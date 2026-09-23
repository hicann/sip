#ifndef FFT_KERNEL_UTILS_H
#define FFT_KERNEL_UTILS_H

#include "catlass/catlass.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/status.hpp"
#include "fft_tiling_def.h"
#include "fft_types.h"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

using namespace Catlass;
using namespace AscendC;

constexpr uint64_t FFT_SYNC_MODE = 2;   // CrossCore mode=2（AI Core 内 AIC↔AIV）

// 核内 EventID：双流水线按 pair 奇偶乒乓（核内同步范式见 coding-in-ascendc/sync-guide.md §五）
constexpr event_t SYNC_ID0 = EVENT_ID0;   // 偶数 pair 流水线
constexpr event_t SYNC_ID1 = EVENT_ID1;   // 奇数 pair 流水线

constexpr uint32_t GATHER_MASK_PER_REPEAT = 64;

using FftElement = float;
using FftLayoutTag = layout::RowMajor;
using FftArchTag = Arch::Ascend950;

// ── GEMM 类型定义【实例相关：final（目标形态）Butterfly 2 + Gemm 16/16 + Butterfly 2，两级 GEMM】──
//
// MmadPingpong 模板参数（按位序，语义源自 {Catlass repo}
//   include/catlass/gemm/dispatch_policy.hpp 的 MmadPingpong 定义（L298-310），勿凭记忆改）：
//   1) ArchTag         ：硬件架构（Ascend950）
//   2) ENABLE_UNIT_FLAG（false；int8 输入时必须 false）
//   3) USE_HF32_MODE   （false）
//   4) L0C_STAGES      （1）
//   5) ENABLE_L1_RESIDENT——W 常驻 L1（首次拷 L1A 后命中即跳过重拷；
//                        GEMM 次数越多收益越大，默认开启）
//   6-8) L1A/L1B/L0A 的 stages 数（pingpong 缓冲份数）
//   9) L0B_STAGES      ：L0B 缓冲份数——B ≤ 64KB 时必须 =1
//
// TileShape（tla::Shape<M, N, K>）：L1/L0 的分块尺寸，三处联动（gemm-usage-guide §2）：
//   1) N ≥ GEMM 的 N（N 不足则 B 加载被截断——历史上 N 不足导致 B 不完整进 L1B 的教训）
//   2) L0B_STAGES 按 B 尺寸重判（见上第 9 参）
//   3) 与 fft_types.h GetMm* 同源推导
// final 推导（目标形态）：GEMM#1 M=K=32 N=512、GEMM#2 M=K=32 N=512
//   → M=32（≥两轮 M）、N=512（≥两轮 N，B 完整进 L1B）、K=32（L0B tile=K×N=32×512×4B=64KB=上限）；
//   B 单份 32×512×4B=64KB→L0B=1；W 常驻 L1（W16 4KB）→L1A=1；
//   B 乒乓 2 份=128KB（已验证 L1B 容量）→L1B=2；L0A tile=32×32×4B=4KB→L0A=1
using FftDispatchPolicy = Gemm::MmadPingpong<
    FftArchTag, false, false, 1, true, 1, 2, 1, 1>;

// TileShape 判法（三处联动，详见上方注释块与 gemm-usage-guide §2）：
//   M ≥ 各轮 GEMM 的 M（=32）；N ≥ 各轮 GEMM 的 N（=512，B 完整加载，否则截断）；
//   K = W 常驻切片粒度（受 L0B tile = K×N ≤ 64KB 约束 → N=512 时 K≤32）
// final mmN 双信号（99 例程口径）：两轮 N=512 → TileShape N=512；
//   L0B tile = K×N = 32×512×4B = 64KB = 上限（L0B_STAGES=1）；L0C tile = 32×512×4B = 64KB
using FftL1TileShape = tla::Shape<tla::Int<32>, tla::Int<512>, tla::Int<32>>;
using FftL0TileShape = tla::Shape<tla::Int<32>, tla::Int<512>, tla::Int<32>>;

// C 直写 UB + SPLIT_N（99 例程口径，mmN 双信号列拼接）：AIV0 得 N 前半（信号0）、AIV1 得后半（信号1）
using FftTileCopyToUb = Gemm::Tile::PackedTileCopyTlaToUB<
    FftArchTag, FftElement, FftLayoutTag, FftElement, FftLayoutTag,
    FftElement, FftLayoutTag, void,
    Gemm::Tile::CopyL0CToUBMode::SPLIT_N>;

using FftBlockMmadToUb = Gemm::Block::BlockMmadTla<
    FftDispatchPolicy, FftL1TileShape, FftL0TileShape,
    FftElement, FftElement, FftElement, void, FftTileCopyToUb>;

// ── kernel 上下文：tiling → 单核视图（InitContext 从 GM tiling 读入；stage 化，
//    对齐概要设计 Part A §2.3）──
struct FftKernelContext {
    // —— 信号描述（ND/任意轴消费）——
    int64_t fftN;                            // 主信号轴长度（1D 定制 = signalLen[0]）
    int32_t signalDims;
    int64_t signalLen[FFT_MAX_SIGNAL_DIM];
    int32_t direction;
    int32_t ubMode;

    // —— stage 清单（族参数按 type 消费：gemm.mm*/opOffset/tOffset/bOffset、
    //    butterfly.layers/twiddleOffset/phaseOffset）——
    int32_t stageCount;
    int32_t gemmCount;                       // GEMM 轮数 = Σ(type==Gemm)
    FftStageDesc stage[FFT_MAX_STAGE];

    // —— 多核侧（单核视图）——
    int64_t rowFloats;
    int64_t wsRegionBytes;                   // 每核 workspace 区大小（B 段基址相对此区）
    uint32_t blockIdx;
    uint32_t rowStart;
    uint32_t rowEnd;

    // —— 其余全局 coeffs 段（float 偏移，InitContext 从字节换算）——
    int64_t gatherTableOffset;               // 步⑤ SIMD gather 偏移表段（GATHER_TABLE_FLOATS，coeffs 末尾）
    int64_t p6t2TableOffset;                 // 步① T2 复乘 SIMD gather 表段（P6T2_TABLE_FLOATS，coeffs 末尾 GATHER_TABLE 之后）
    int64_t p4TableOffset;                   // P4 twiddle SIMD gather 表段（P4_TABLE_FLOATS，coeffs 末尾 P6T2 之后）
};

CATLASS_DEVICE void InitContext(
    __gm__ FftTilingData* tilingData, FftKernelContext& ctx)
{
    auto& inputParams = tilingData->inputParams;
    auto& multiCoreParams = tilingData->multiCoreParams;

    ctx.fftN = inputParams.signalLen[0];
    ctx.signalDims = inputParams.signalDims;
    for (int i = 0; i < FFT_MAX_SIGNAL_DIM; i++) {
        ctx.signalLen[i] = inputParams.signalLen[i];
    }
    ctx.direction = inputParams.direction;
    ctx.ubMode = inputParams.ubMode;
    ctx.rowFloats = ctx.fftN * 2;

    ctx.stageCount = inputParams.stageCount;
    ctx.gemmCount = 0;
    for (int i = 0; i < ctx.stageCount; i++) {
        // __gm__ 地址空间对象不可整体赋值到本地——逐字段拷贝（标量读 __gm__ 合法）
        const auto& src = inputParams.stage[i];
        auto& dst = ctx.stage[i];
        dst.type = src.type;
        dst.radix = src.radix;
        if (src.type == FftStageType::Gemm) {
            dst.gemm.mmM = src.gemm.mmM;
            dst.gemm.mmK = src.gemm.mmK;
            dst.gemm.mmN = src.gemm.mmN;
            dst.gemm.opOffset = src.gemm.opOffset;
            dst.gemm.tOffset = src.gemm.tOffset;
            dst.gemm.bOffset = src.gemm.bOffset;
            ctx.gemmCount++;
            // 偏移：字节 → float（coeffs/workspace 基址 + 偏移处取数）
            dst.gemm.opOffset /= sizeof(float);
            dst.gemm.tOffset /= sizeof(float);
            dst.gemm.bOffset /= sizeof(float);
        } else {
            dst.butterfly.layers = src.butterfly.layers;
            dst.butterfly.twiddleOffset = src.butterfly.twiddleOffset / sizeof(float);   // 字节 → float（Butterfly 尾随 twiddle T0 取数）
            dst.butterfly.phaseOffset = src.butterfly.phaseOffset;
        }
    }
    ctx.wsRegionBytes = multiCoreParams.wsRegionBytes / sizeof(float);   // 字节 → float（kernel bOff 计算用）
    ctx.gatherTableOffset = inputParams.gatherTableOffset / sizeof(float);   // 字节 → float（GATHER_TABLE 段取数）
    ctx.p6t2TableOffset = inputParams.p6t2TableOffset / sizeof(float);       // 字节 → float（P6T2 段取数）
    ctx.p4TableOffset = inputParams.p4TableOffset / sizeof(float);           // 字节 → float（P4 段取数）

    // __mix__(1,2)：AIV 物理 blockIdx 是 AIC 的 2 倍——AIV 侧统一 >>1
    if ASCEND_IS_AIC {
        ctx.blockIdx = AscendC::GetBlockIdx();
    } else {
        ctx.blockIdx = AscendC::GetBlockIdx() >> 1;
    }

    ctx.rowStart = multiCoreParams.rowStartIdx[ctx.blockIdx];
    ctx.rowEnd = multiCoreParams.rowStartIdx[ctx.blockIdx + 1];
}

// ── 单轮 GEMM 标准调用（AIC 侧）：A=W(coeffs) × B(workspace) → C=UB ──
CATLASS_DEVICE void RunGemmToUb(
    FftBlockMmadToUb& blockMmad,
    __gm__ float* gmA, int64_t aOffset,
    __gm__ float* gmB, int64_t bOffset,
    LocalTensor<FftElement>& ubC,
    int32_t M, int32_t K, int32_t N)
{
    GlobalTensor<FftElement> gmA_tensor;
    gmA_tensor.SetGlobalBuffer(gmA + aOffset);
    gmA_tensor.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
    GlobalTensor<FftElement> gmB_tensor;
    gmB_tensor.SetGlobalBuffer(gmB + bOffset);

    auto layoutA = tla::MakeLayout<FftElement, FftLayoutTag>(
        static_cast<uint32_t>(M), static_cast<uint32_t>(K));
    auto layoutB = tla::MakeLayout<FftElement, FftLayoutTag>(
        static_cast<uint32_t>(K), static_cast<uint32_t>(N));
    auto layoutC = tla::MakeLayout<FftElement, FftLayoutTag>(
        static_cast<uint32_t>(M), static_cast<uint32_t>(N));

    auto tensorA = tla::MakeTensor(gmA_tensor, layoutA, Arch::PositionGM{});
    auto tensorB = tla::MakeTensor(gmB_tensor, layoutB, Arch::PositionGM{});
    auto tensorC = tla::MakeTensor(ubC, layoutC, Arch::PositionUB{});

    GemmCoord actualShape(static_cast<uint32_t>(M), static_cast<uint32_t>(N),
                          static_cast<uint32_t>(K));

    auto tensorBlockA = GetTile(tensorA, tla::MakeCoord(0u, 0u),
        tla::MakeShape(static_cast<uint32_t>(M), static_cast<uint32_t>(K)));
    auto tensorBlockB = GetTile(tensorB, tla::MakeCoord(0u, 0u),
        tla::MakeShape(static_cast<uint32_t>(K), static_cast<uint32_t>(N)));
    auto tensorBlockC = GetTile(tensorC, tla::MakeCoord(0u, 0u),
        tla::MakeShape(static_cast<uint32_t>(M), static_cast<uint32_t>(N)));

    blockMmad(tensorBlockA, tensorBlockB, tensorBlockC, actualShape);
}

#endif // FFT_KERNEL_UTILS_H
