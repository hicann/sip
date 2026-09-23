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

constexpr uint64_t FFT_SYNC_MODE = 2;

// renorm 范式：双流水线用 2 个核内 EventID，syncId 按 pair 奇偶切换。
// 靠 (HardEvent, EventID) 组合的独立性区分方向，两个流水线互不阻塞。
constexpr event_t SYNC_ID0 = EVENT_ID0;   // 偶数 pair 流水线
constexpr event_t SYNC_ID1 = EVENT_ID1;   // 奇数 pair 流水线

constexpr uint32_t GATHER_MASK_PER_REPEAT = 64;

using FftElement = float;
using FftLayoutTag = layout::RowMajor;
using FftArchTag = Arch::Ascend950;

// radix 分支：64×64 的 GEMM 是 128×128×128；16×16×16 的 GEMM 是 32×32×512。
// 16×16×16 需 L1/L0 tile 的 N ≥ 512（保证 B 完整加载进 L1B，否则 N 被截断），
// 且 L0B 为 64KB，L0B_STAGES 必须=1（否则 64KB×2 > L0B_SIZE=64KB）。
constexpr bool FFT_IS_RADIX_16 = (Fft::FFT_RADIX == Fft::FftRadix::Radix16x16x16);

using FftDispatchPolicy = Gemm::MmadPingpong<
    FftArchTag, false, false, 1, true, 2, 2, 2,
    (FFT_IS_RADIX_16 ? 1u : 2u)>;

using FftL1TileShape = tla::Shape<
    tla::Int<(FFT_IS_RADIX_16 ? 32 : 128)>,
    tla::Int<(FFT_IS_RADIX_16 ? 512 : 128)>,
    tla::Int<(FFT_IS_RADIX_16 ? 32 : 64)>>;
using FftL0TileShape = tla::Shape<
    tla::Int<(FFT_IS_RADIX_16 ? 32 : 128)>,
    tla::Int<(FFT_IS_RADIX_16 ? 512 : 128)>,
    tla::Int<(FFT_IS_RADIX_16 ? 32 : 64)>>;

using FftTileCopyToUb = Gemm::Tile::PackedTileCopyTlaToUB<
    FftArchTag, FftElement, FftLayoutTag, FftElement, FftLayoutTag,
    FftElement, FftLayoutTag, void,
    Gemm::Tile::CopyL0CToUBMode::SPLIT_N>;

using FftBlockMmadToUb = Gemm::Block::BlockMmadTla<
    FftDispatchPolicy, FftL1TileShape, FftL0TileShape,
    FftElement, FftElement, FftElement, void, FftTileCopyToUb>;

struct FftKernelContext {
    int64_t fftN;
    int32_t N1;
    int32_t N2;
    int32_t direction;
    int32_t ubMode;

    int64_t rowFloats;      // 实数输入行长度 = fftN
    int64_t outRowFloats;   // 半谱输出行长度 = (fftN/2+1)*2（交织复）
    int64_t wsOffset[14];

    int32_t mmM[3];
    int32_t mmK[3];
    int32_t mmN[3];

    int64_t wOffset[3];
    int64_t tOffset;
    int64_t tExpOffset;
    int64_t tRadix2Offset;
    int64_t tPhaseAOffset;
    int64_t t12ExpOffset;
    int64_t tP3LaneOffset;

    uint32_t blockIdx;
    uint32_t rowStart;
    uint32_t rowEnd;
};

CATLASS_DEVICE void InitContext(
    __gm__ FftTilingData* tilingData, FftKernelContext& ctx)
{
    auto& inputParams = tilingData->inputParams;
    auto& multiCoreParams = tilingData->multiCoreParams;

    ctx.fftN = inputParams.fftN;
    ctx.N1 = inputParams.N1;
    ctx.N2 = inputParams.N2;
    ctx.direction = inputParams.direction;
    ctx.ubMode = inputParams.ubMode;
    ctx.rowFloats = ctx.fftN;                     // 实数输入：每行 fftN floats
    ctx.outRowFloats = (ctx.fftN / 2 + 1) * 2;     // 半谱输出：每行 2049 复 = 4098 floats
    for (int i = 0; i < 14; i++) {
        ctx.wsOffset[i] = multiCoreParams.workspaceOffset[i] / sizeof(float);
    }

    for (int32_t i = 0; i < 3; i++) {
        ctx.mmM[i] = inputParams.mmM[i];
        ctx.mmK[i] = inputParams.mmK[i];
        ctx.mmN[i] = inputParams.mmN[i];
        ctx.wOffset[i] = inputParams.wMatrixOffset[i] / sizeof(float);
    }
    ctx.tOffset = inputParams.tMatrixOffset / sizeof(float);
    ctx.tExpOffset = inputParams.tExpandedOffset / sizeof(float);
    ctx.tRadix2Offset = inputParams.tRadix2Offset / sizeof(float);
    ctx.tPhaseAOffset = inputParams.tPhaseAOffset / sizeof(float);
    ctx.t12ExpOffset = inputParams.t12ExpandedOffset / sizeof(float);
    ctx.tP3LaneOffset = inputParams.tP3LaneOffset / sizeof(float);

    if ASCEND_IS_AIC {
        ctx.blockIdx = AscendC::GetBlockIdx();
    } else {
        ctx.blockIdx = AscendC::GetBlockIdx() >> 1;
    }

    ctx.rowStart = multiCoreParams.rowStartIdx[ctx.blockIdx];
    ctx.rowEnd = multiCoreParams.rowStartIdx[ctx.blockIdx + 1];
}

CATLASS_DEVICE void RunGemmToUb(
    FftBlockMmadToUb& blockMmad,
    GM_ADDR gmA, int64_t aOffset,
    GM_ADDR gmB, int64_t bOffset,
    LocalTensor<FftElement>& ubC,
    int32_t M, int32_t K, int32_t N)
{
    GlobalTensor<FftElement> gmA_tensor;
    gmA_tensor.SetGlobalBuffer((__gm__ FftElement*)(gmA) + aOffset);
    gmA_tensor.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
    GlobalTensor<FftElement> gmB_tensor;
    gmB_tensor.SetGlobalBuffer((__gm__ FftElement*)(gmB) + bOffset);

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