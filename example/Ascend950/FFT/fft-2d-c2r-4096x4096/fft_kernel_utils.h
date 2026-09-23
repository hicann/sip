#ifndef FFT_KERNEL_UTILS_H
#define FFT_KERNEL_UTILS_H

#include "catlass/catlass.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/status.hpp"
#include "fft_tiling_def.h"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

using namespace Catlass;
using namespace AscendC;

// ===========================================================================
// 跨核同步（mode 2：AIC↔AIV 1:2，双信号 SPLIT_N）
// ===========================================================================
constexpr uint64_t FFT_SYNC_MODE = 2;
constexpr event_t SYNC_ID0 = EVENT_ID0;   // 偶数 pair 流水线
constexpr event_t SYNC_ID1 = EVENT_ID1;   // 奇数 pair 流水线

constexpr uint32_t GATHER_MASK_PER_REPEAT = 64;

// flag ID：pair 奇偶 × 4 + slot（B0=0,G0=1,B1=2,G1=3）
CATLASS_DEVICE uint16_t FftFlagId(uint32_t pair, uint32_t slot)
{
    return static_cast<uint16_t>((pair & 1) * 4 + slot);
}

// ===========================================================================
// radix-16 GEMM 类型：W16(32×32) @ B(32×512)，SPLIT_N 双 AIV 直达 UB
// L1/L0 tile N=512 保证 B 完整加载；L0B_STAGES=1（64KB 限制）
// ===========================================================================
using FftElement = float;
using FftLayoutTag = layout::RowMajor;
using FftArchTag = Arch::Ascend950;

using FftDispatchPolicy = Gemm::MmadPingpong<
    FftArchTag, false, false, 1, true, 2, 2, 2, 1>;

using FftL1TileShape = tla::Shape<tla::Int<32>, tla::Int<512>, tla::Int<32>>;
using FftL0TileShape = tla::Shape<tla::Int<32>, tla::Int<512>, tla::Int<32>>;

using FftTileCopyToUb = Gemm::Tile::PackedTileCopyTlaToUB<
    FftArchTag, FftElement, FftLayoutTag, FftElement, FftLayoutTag,
    FftElement, FftLayoutTag, void,
    Gemm::Tile::CopyL0CToUBMode::SPLIT_N>;

using FftBlockMmadToUb = Gemm::Block::BlockMmadTla<
    FftDispatchPolicy, FftL1TileShape, FftL0TileShape,
    FftElement, FftElement, FftElement, void, FftTileCopyToUb>;

// ===========================================================================
// 上下文
// ===========================================================================
struct FftC2R2DKernelContext {
    int64_t fftN;
    int32_t batch;
    int32_t N1, N2;
    int32_t direction;
    int32_t mmM, mmK, mmN;
    int32_t halfN;     // fftN/2+1
    int32_t padHalf;   // roundup64(halfN)

    int64_t wOffset;
    int64_t tOffset;
    int64_t tExpandedOffset;
    int64_t tRadix2Offset;
    int64_t tPhaseAOffset;
    int64_t t12ExpandedOffset;
    int64_t tP3LaneOffset;

    uint32_t blockIdx;
    uint32_t rowStart;
    uint32_t rowEnd;
    int64_t inRowStride;   // 当前 pass 输入行距（float 单位）
};

CATLASS_DEVICE void InitC2R2DContext(
    __gm__ FftC2R2DTilingData* tilingData, FftC2R2DKernelContext& ctx)
{
    auto& inputParams = tilingData->inputParams;
    auto& multiCoreParams = tilingData->multiCoreParams;

    ctx.fftN = inputParams.fftN;
    ctx.batch = inputParams.batch;
    ctx.N1 = inputParams.N1;
    ctx.N2 = inputParams.N2;
    ctx.direction = inputParams.direction;
    ctx.mmM = inputParams.mmM;
    ctx.mmK = inputParams.mmK;
    ctx.mmN = inputParams.mmN;
    ctx.halfN = inputParams.halfN;
    ctx.padHalf = inputParams.padHalf;

    ctx.wOffset = inputParams.wOffset / sizeof(float);
    ctx.tOffset = inputParams.tOffset / sizeof(float);
    ctx.tExpandedOffset = inputParams.tExpandedOffset / sizeof(float);
    ctx.tRadix2Offset = inputParams.tRadix2Offset / sizeof(float);
    ctx.tPhaseAOffset = inputParams.tPhaseAOffset / sizeof(float);
    ctx.t12ExpandedOffset = inputParams.t12ExpandedOffset / sizeof(float);
    ctx.tP3LaneOffset = inputParams.tP3LaneOffset / sizeof(float);

    if ASCEND_IS_AIC {
        ctx.blockIdx = AscendC::GetBlockIdx();
    } else {
        ctx.blockIdx = AscendC::GetBlockIdx() >> 1;
    }

    // rowStart/rowEnd 由主 kernel 按 pass 分别设置（瘦列 S2 用 skinnyRowStartIdx，S4 用 fullRowStartIdx）
    ctx.rowStart = 0;
    ctx.rowEnd = 0;
    ctx.inRowStride = 0;
}

// ===========================================================================
// GEMM：W(GM) @ B(GM workspace) -> C(UB)，SPLIT_N
// ===========================================================================
CATLASS_DEVICE void RunGemmRadix16(
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