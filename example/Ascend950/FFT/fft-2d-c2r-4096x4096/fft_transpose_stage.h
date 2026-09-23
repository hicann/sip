// ===========================================================================
// fft_transpose_stage.h — 矩形 64×64 tile 块转置（复数）
//
// 双缓冲乒乓：MTE2 读组 g（bufIdx）∥ SIMT 转置组 g（bufIdx）∥ MTE3 写组 g-1（1-bufIdx）。
// 从 ex85 的 RunTransposeRectStage 移植，供 C2R 逆序瘦列流水使用。
// ===========================================================================

#ifndef FFT_TRANSPOSE_STAGE_H
#define FFT_TRANSPOSE_STAGE_H

#include "catlass/arch/arch.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/catlass.hpp"

#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "simt_api/vector_functions.h"

#include "fft_kernel_utils.h"
#include "helper.hpp"

namespace Fft {

using namespace AscendC;
using namespace Catlass;

// 转置 stage 专属的核内事件 ID（与 kernel_utils 的 mode-2 同步分离）
constexpr int32_t EVENT_MTE2_V = 0;
constexpr int32_t EVENT_V_MTE3 = 1;
constexpr int32_t EVENT_MTE3_MTE2 = 2;

constexpr uint32_t TR_TILE_C = 64;
constexpr uint32_t TR_TILE_R = 64;
constexpr uint32_t TR_W_IN   = TR_TILE_C * 2;   // 128
constexpr uint32_t TR_W_OUT  = TR_TILE_R * 2;   // 128
constexpr uint32_t TR_TILE_FLOATS = TR_TILE_R * TR_W_IN;   // 8192
constexpr uint32_t TR_SHARED_STRIDE = (TR_TILE_C + 3) * 2; // 134
constexpr uint32_t TR_TILE_HALF = TR_TILE_R / 2;           // 32
constexpr uint32_t TR_THREADS = TR_TILE_HALF * TR_TILE_HALF; // 1024
constexpr uint32_t TR_SHARED_FLOATS = TR_TILE_R * TR_SHARED_STRIDE;  // 8576
constexpr uint32_t TR_BUF_FLOATS = 2 * TR_TILE_FLOATS + TR_SHARED_FLOATS; // 24960

// ─── 复数 64×64 tile 转置（读/写侧 512B，float2 8B 访问）────────────────
__simt_vf__ __launch_bounds__(TR_THREADS) inline void simt_transpose_c64x64_rect(
    __ubuf__ float* dst, const __ubuf__ float* src, __ubuf__ float* shared)
{
    uint32_t ty = threadIdx.y;
    uint32_t tx = threadIdx.x;

    const __ubuf__ float2* src2 = (const __ubuf__ float2*)src;
    __ubuf__ float2* shared2 = (__ubuf__ float2*)shared;
    __ubuf__ float2* dst2 = (__ubuf__ float2*)dst;

    shared2[(ty * TR_SHARED_STRIDE + tx * 2) >> 1] =
        src2[(ty * TR_W_IN + tx * 2) >> 1];
    shared2[(ty * TR_SHARED_STRIDE + (tx + TR_TILE_HALF) * 2) >> 1] =
        src2[(ty * TR_W_IN + (tx + TR_TILE_HALF) * 2) >> 1];
    shared2[((ty + TR_TILE_HALF) * TR_SHARED_STRIDE + tx * 2) >> 1] =
        src2[((ty + TR_TILE_HALF) * TR_W_IN + tx * 2) >> 1];
    shared2[((ty + TR_TILE_HALF) * TR_SHARED_STRIDE + (tx + TR_TILE_HALF) * 2) >> 1] =
        src2[((ty + TR_TILE_HALF) * TR_W_IN + (tx + TR_TILE_HALF) * 2) >> 1];

    asc_syncthreads();

    dst2[(ty * TR_W_OUT + tx * 2) >> 1] =
        shared2[(tx * TR_SHARED_STRIDE + ty * 2) >> 1];
    dst2[(ty * TR_W_OUT + (tx + TR_TILE_HALF) * 2) >> 1] =
        shared2[((tx + TR_TILE_HALF) * TR_SHARED_STRIDE + ty * 2) >> 1];
    dst2[((ty + TR_TILE_HALF) * TR_W_OUT + tx * 2) >> 1] =
        shared2[(tx * TR_SHARED_STRIDE + (ty + TR_TILE_HALF) * 2) >> 1];
    dst2[((ty + TR_TILE_HALF) * TR_W_OUT + (tx + TR_TILE_HALF) * 2) >> 1] =
        shared2[((tx + TR_TILE_HALF) * TR_SHARED_STRIDE + (ty + TR_TILE_HALF) * 2) >> 1];
}

// ─── 矩形复数转置 stage（双缓冲）─────────────────────────────────────
// srcRows × srcCols（均为 64 倍数），srcRowFloats/dstRowFloats 为各自行距（float）。
__aicore__ inline void RunTransposeRectStage(
    Arch::Resource<FftArchTag>& resource, GM_ADDR src, GM_ADDR dst,
    int32_t srcRows, int32_t srcCols,
    int32_t srcRowFloats, int32_t dstRowFloats,
    int32_t cubeCoreNum)
{
    if ASCEND_IS_AIC {
        return;
    }

    uint32_t vecIdx = AscendC::GetBlockIdx();
    uint32_t vecCoreNum = static_cast<uint32_t>(cubeCoreNum) * 2;
    if (vecIdx >= vecCoreNum) {
        return;
    }

    const uint32_t numRowBlocks = static_cast<uint32_t>(srcRows) / TR_TILE_R;
    const uint32_t numColBlocks = static_cast<uint32_t>(srcCols) / TR_TILE_C;
    const uint32_t totalTiles = numRowBlocks * numColBlocks;

    const uint32_t tilesPerAiv = totalTiles / vecCoreNum;
    const uint32_t tilesRem = totalTiles % vecCoreNum;
    const uint32_t myStart = vecIdx * tilesPerAiv + (vecIdx < tilesRem ? vecIdx : tilesRem);
    const uint32_t myCount = tilesPerAiv + (vecIdx < tilesRem ? 1 : 0);
    if (myCount == 0) { return; }

    LocalTensor<float> srcBuf[2], dstBuf[2], shBuf[2];
    __ubuf__ float* srcAddr[2]; __ubuf__ float* dstAddr[2]; __ubuf__ float* shAddr[2];
    for (uint32_t b = 0; b < 2; b++) {
        srcBuf[b] = resource.ubBuf.template GetBufferByByte<float>(b * TR_BUF_FLOATS * sizeof(float));
        dstBuf[b] = resource.ubBuf.template GetBufferByByte<float>((b * TR_BUF_FLOATS + TR_TILE_FLOATS) * sizeof(float));
        shBuf[b] = resource.ubBuf.template GetBufferByByte<float>((b * TR_BUF_FLOATS + 2 * TR_TILE_FLOATS) * sizeof(float));
        srcAddr[b] = (__ubuf__ float*)srcBuf[b].GetPhyAddr();
        dstAddr[b] = (__ubuf__ float*)dstBuf[b].GetPhyAddr();
        shAddr[b] = (__ubuf__ float*)shBuf[b].GetPhyAddr();
    }

    __gm__ float* gmSrcBase = (__gm__ float*)src;
    __gm__ float* gmDstBase = (__gm__ float*)dst;

    const int64_t sRowFloats = static_cast<int64_t>(srcRowFloats);
    const int64_t dRowFloats = static_cast<int64_t>(dstRowFloats);
    const uint16_t readGap = static_cast<uint16_t>((sRowFloats - TR_W_IN) * sizeof(float) / 32);
    const uint16_t writeGap = static_cast<uint16_t>((dRowFloats - TR_W_OUT) * sizeof(float) / 32);

    for (uint32_t g = 0; g < myCount; g++) {
        uint32_t bufIdx = g & 1u;
        uint32_t tile = myStart + g;
        uint32_t tr = tile / numColBlocks;
        uint32_t tc = tile - tr * numColBlocks;

        // MTE2 读组 g
        {
            GlobalTensor<float> gmSrc;
            int64_t rowBase = static_cast<int64_t>(tr * TR_TILE_R) * sRowFloats
                            + static_cast<int64_t>(tc * TR_W_IN);
            gmSrc.SetGlobalBuffer(gmSrcBase + rowBase);
            DataCopyParams cp;
            cp.blockCount = TR_TILE_R;
            cp.blockLen = (TR_W_IN * sizeof(float)) / 32;
            cp.srcGap = readGap;
            cp.dstGap = 0;
            DataCopy(srcBuf[bufIdx], gmSrc, cp);
        }
        SetFlag<HardEvent::MTE2_V>(EVENT_MTE2_V);
        WaitFlag<HardEvent::MTE2_V>(EVENT_MTE2_V);

        // MTE3 写组 g-1
        if (g >= 1) {
            uint32_t prevIdx = 1u - bufIdx;
            uint32_t ptile = myStart + g - 1;
            uint32_t ptr = ptile / numColBlocks;
            uint32_t ptc = ptile - ptr * numColBlocks;

            AscendC::WaitFlag<HardEvent::V_MTE3>(EVENT_V_MTE3);
            GlobalTensor<float> gmDst;
            int64_t dstRowBase = static_cast<int64_t>(ptc * TR_TILE_C) * dRowFloats
                               + static_cast<int64_t>(ptr * TR_W_OUT);
            gmDst.SetGlobalBuffer(gmDstBase + dstRowBase);
            DataCopyParams cpw;
            cpw.blockCount = TR_TILE_C;
            cpw.blockLen = (TR_W_OUT * sizeof(float)) / 32;
            cpw.srcGap = 0;
            cpw.dstGap = writeGap;
            DataCopy(gmDst, dstBuf[prevIdx], cpw);
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_MTE3_MTE2);
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_MTE3_MTE2);
        }

        // V 转置组 g
        asc_vf_call<simt_transpose_c64x64_rect>(
            dim3(TR_TILE_HALF, TR_TILE_HALF, 1), dstAddr[bufIdx], srcAddr[bufIdx], shAddr[bufIdx]);
        AscendC::DataSyncBarrier<AscendC::MemDsbT::UB>();
        SetFlag<HardEvent::V_MTE3>(EVENT_V_MTE3);
    }

    // tail 排空最后一组
    {
        uint32_t lastIdx = (myCount - 1) & 1u;
        uint32_t lTile = myStart + myCount - 1;
        uint32_t ltr = lTile / numColBlocks;
        uint32_t ltc = lTile - ltr * numColBlocks;
        AscendC::WaitFlag<HardEvent::V_MTE3>(EVENT_V_MTE3);
        GlobalTensor<float> gmDst;
        int64_t dstRowBase = static_cast<int64_t>(ltc * TR_TILE_C) * dRowFloats
                           + static_cast<int64_t>(ltr * TR_W_OUT);
        gmDst.SetGlobalBuffer(gmDstBase + dstRowBase);
        DataCopyParams cpw;
        cpw.blockCount = TR_TILE_C;
        cpw.blockLen = (TR_W_OUT * sizeof(float)) / 32;
        cpw.srcGap = 0;
        cpw.dstGap = writeGap;
        DataCopy(gmDst, dstBuf[lastIdx], cpw);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_MTE3_MTE2);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_MTE3_MTE2);
    }

    AscendC::PipeBarrier<PIPE_MTE3>();
}

} // namespace Fft

#endif // FFT_TRANSPOSE_STAGE_H