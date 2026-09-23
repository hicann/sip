#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include "catlass/arch/arch.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/catlass.hpp"
#include "catlass/status.hpp"

#include "fft_kernel.h"
#include "fft_kernel_utils.h"
#include "fft_kernel_common.h"
#include "fft_r2c_unpack.h"
#include "fft_tiling_def.h"

#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "simt_api/vector_functions.h"
#include "helper.hpp"

using namespace Catlass;
using namespace AscendC;

namespace Fft {

using ArchTag = Arch::Ascend950;

// ═══════════════════════════════════════════════════════════════════════════
// Stage 1：横向 R2C 1D（两实行打包 → 16×16×16 C2C → 共轭拆分）
// 与 fft_r2c1d_kernel.cpp 的 FftR2c1dKernel 主体一致，仅去 ubMode/mixMode 参数，
// 末尾补 PipeBarrier<PIPE_ALL>（SyncAll 前置）。
// ═══════════════════════════════════════════════════════════════════════════

CATLASS_DEVICE void Phase1R2cAivFused(
    FftKernelContext& ctx, uint32_t subIdx, uint32_t globalPairIdx,
    event_t syncId,
    LocalTensor<float>& ubSepReal, LocalTensor<float>& ubSepImag,
    __gm__ float* gmInputBase, __gm__ float* gmWsBase)
{
    uint32_t mySignal = globalPairIdx * 2 + subIdx;
    int64_t subCount = static_cast<int64_t>(ctx.N2) * ctx.N2;
    int64_t colOffset = static_cast<int64_t>(subIdx) * subCount;

    if constexpr (!SYNC_ONLY) {
        int64_t inOffA = static_cast<int64_t>(2 * mySignal) * ctx.rowFloats;
        int64_t inOffB = static_cast<int64_t>(2 * mySignal + 1) * ctx.rowFloats;
        GlobalTensor<float> gmA;
        gmA.SetGlobalBuffer(gmInputBase + inOffA);
        GlobalTensor<float> gmB;
        gmB.SetGlobalBuffer(gmInputBase + inOffB);
        DataCopy(ubSepReal, gmA, static_cast<uint32_t>(ctx.rowFloats));
        DataCopy(ubSepImag, gmB, static_cast<uint32_t>(ctx.rowFloats));
    }

    SetFlag<HardEvent::MTE2_MTE3>(syncId);
    WaitFlag<HardEvent::MTE2_MTE3>(syncId);

    if constexpr (!SYNC_ONLY) {
        int64_t bMatrixFloats = static_cast<int64_t>(ctx.mmK[0]) * ctx.mmN[0];
        int64_t baseOff = static_cast<int64_t>(ctx.blockIdx) * (6 * bMatrixFloats)
                              + static_cast<int64_t>(globalPairIdx & 1) * bMatrixFloats
                              + ctx.wsOffset[0];

        DataCopyParams copyParams;
        copyParams.blockCount = static_cast<uint16_t>(ctx.N1);
        copyParams.blockLen = static_cast<uint16_t>(subCount * sizeof(float) / 32);
        copyParams.srcGap = 0;
        copyParams.dstGap = static_cast<uint16_t>(subCount * sizeof(float) / 32);

        GlobalTensor<float> gmReal;
        gmReal.SetGlobalBuffer(gmWsBase + baseOff + colOffset);
        DataCopy(gmReal, ubSepReal, copyParams);

        GlobalTensor<float> gmImag;
        gmImag.SetGlobalBuffer(gmWsBase + baseOff
            + static_cast<int64_t>(ctx.N1) * ctx.mmN[0] + colOffset);
        DataCopy(gmImag, ubSepImag, copyParams);
    }
}

CATLASS_DEVICE void Phase7R2cAivFused(
    FftKernelContext& ctx, uint32_t subIdx, uint32_t globalPairIdx,
    event_t syncId,
    LocalTensor<float>& ubSepRe, LocalTensor<float>& ubSepIm,
    __gm__ float* gmOutputBase)
{
    uint32_t mySignal = globalPairIdx * 2 + subIdx;
    UnpackTwoRealStore(gmOutputBase, ubSepRe, ubSepIm,
                       mySignal, static_cast<uint32_t>(ctx.outRowFloats));
    SetFlag<HardEvent::V_MTE3>(syncId);
    WaitFlag<HardEvent::V_MTE3>(syncId);
}

__aicore__ inline void RunR2c1dStage(
    GM_ADDR input, GM_ADDR output, GM_ADDR coeffs, GM_ADDR workspace, GM_ADDR tiling)
{
    FftKernelContext ctx;
    InitContext((__gm__ FftTilingData*)tiling, ctx);
    ctx.outRowFloats = ((ctx.fftN / 2 + 1 + 63) & ~int64_t(63)) * 2;

    if (ctx.blockIdx >= static_cast<uint32_t>(
            ((__gm__ FftTilingData*)tiling)->multiCoreParams.coreNum)) {
        AscendC::PipeBarrier<PIPE_ALL>();
        return;
    }

    Arch::Resource<ArchTag> resource;

    int64_t elemCount = static_cast<int64_t>(ctx.N1) * ctx.N2 * ctx.N2;

    __gm__ float* gmInputBase = (__gm__ float*)(input);
    __gm__ float* gmOutputBase = (__gm__ float*)(output);
    __gm__ float* gmWsBase = (__gm__ float*)(workspace);
    __gm__ float* gmCoeffsBase = (__gm__ float*)(coeffs);

    uint32_t numPairs = (ctx.rowEnd - ctx.rowStart) / 2;

    if ASCEND_IS_AIC {
        FftBlockMmadToUb blockMmadToUb(resource, 0);
        LocalTensor<float> ubGemmBuf0 = resource.ubBuf.template GetBufferByByte<float>(
            UB_OFFSET_GEMM * sizeof(float));
        LocalTensor<float> ubGemmBuf1 = resource.ubBuf.template GetBufferByByte<float>(
            UB_OFFSET_GEMM_1 * sizeof(float));

        int64_t bMatrixFloats = static_cast<int64_t>(ctx.mmK[0]) * ctx.mmN[0];

        uint32_t groups = (numPairs + 1) / 2;
        for (uint32_t i = 0; i < groups; i++) {
            uint32_t pairsInGroup = (i == groups - 1 && numPairs % 2 != 0) ? 1 : 2;

            for (uint32_t j = 0; j < pairsInGroup; j++) {
                uint32_t pair = 2 * i + j;
                if (pair >= numPairs) break;
                uint32_t globalPairIdx = (ctx.rowStart / 2) + pair;
                LocalTensor<float>& ubGemm = (j == 0) ? ubGemmBuf0 : ubGemmBuf1;
                AscendC::CrossCoreWaitFlag<FFT_SYNC_MODE, PIPE_MTE2>(FftFlagId(pair, SLOT_B0));
                if constexpr (!AIC_SKELETON) {
                    int64_t bOff0 = static_cast<int64_t>(ctx.blockIdx) * (6 * bMatrixFloats)
                                  + static_cast<int64_t>(globalPairIdx & 1) * bMatrixFloats
                                  + ctx.wsOffset[0];
                    RunGemmToUb(blockMmadToUb,
                        coeffs, ctx.wOffset[0], workspace, bOff0,
                        ubGemm, ctx.mmM[0], ctx.mmK[0], ctx.mmN[0]);
                }
                AscendC::CrossCoreSetFlag<FFT_SYNC_MODE, PIPE_FIX>(FftFlagId(pair, SLOT_G0));
            }

            for (uint32_t j = 0; j < pairsInGroup; j++) {
                uint32_t pair = 2 * i + j;
                if (pair >= numPairs) break;
                uint32_t globalPairIdx = (ctx.rowStart / 2) + pair;
                LocalTensor<float>& ubGemm = (j == 0) ? ubGemmBuf0 : ubGemmBuf1;
                AscendC::CrossCoreWaitFlag<FFT_SYNC_MODE, PIPE_MTE2>(FftFlagId(pair, SLOT_B1));
                if constexpr (!AIC_SKELETON) {
                    int64_t bOff1 = static_cast<int64_t>(ctx.blockIdx) * (6 * bMatrixFloats)
                                  + static_cast<int64_t>(globalPairIdx & 1) * bMatrixFloats
                                  + ctx.wsOffset[2];
                    RunGemmToUb(blockMmadToUb,
                        coeffs, ctx.wOffset[0], workspace, bOff1,
                        ubGemm, ctx.mmM[1], ctx.mmK[1], ctx.mmN[1]);
                }
                AscendC::CrossCoreSetFlag<FFT_SYNC_MODE, PIPE_FIX>(FftFlagId(pair, SLOT_G1));
            }
        }
    }

    if ASCEND_IS_AIV {
        uint32_t subIdx = AscendC::GetSubBlockIdx();
        int64_t tOff = ctx.tOffset;

        LocalTensor<float> ubSepReal0 = resource.ubBuf.template GetBufferByByte<float>(
            UB_OFFSET_SEP0 * sizeof(float));
        LocalTensor<float> ubSepImag0 = resource.ubBuf.template GetBufferByByte<float>(
            (UB_OFFSET_SEP0 + elemCount) * sizeof(float));
        LocalTensor<float> ubSepReal1 = resource.ubBuf.template GetBufferByByte<float>(
            UB_OFFSET_SEP1 * sizeof(float));
        LocalTensor<float> ubSepImag1 = resource.ubBuf.template GetBufferByByte<float>(
            (UB_OFFSET_SEP1 + elemCount) * sizeof(float));
        LocalTensor<float> ubGemmBuf0 = resource.ubBuf.template GetBufferByByte<float>(
            UB_OFFSET_GEMM * sizeof(float));
        LocalTensor<float> ubGemmBuf1 = resource.ubBuf.template GetBufferByByte<float>(
            UB_OFFSET_GEMM_1 * sizeof(float));
        LocalTensor<float> ubT = resource.ubBuf.template GetBufferByByte<float>(
            UB_OFFSET_T * sizeof(float));
        LocalTensor<float> ubT2Exp = resource.ubBuf.template GetBufferByByte<float>(
            UB_OFFSET_T2_EXP * sizeof(float));
        LocalTensor<float> ubRadix2TW = resource.ubBuf.template GetBufferByByte<float>(
            UB_OFFSET_RADIX2_TW * sizeof(float));
        LocalTensor<float> ubPhaseAOff = resource.ubBuf.template GetBufferByByte<float>(
            UB_OFFSET_PHASE_A * sizeof(float));
        LocalTensor<float> ubT12Exp = resource.ubBuf.template GetBufferByByte<float>(
            UB_OFFSET_T12_EXP * sizeof(float));
        LocalTensor<float> ubP3Lane = resource.ubBuf.template GetBufferByByte<float>(
            UB_OFFSET_P3_LANE * sizeof(float));

        if constexpr (!SYNC_ONLY) {
            GlobalTensor<float> gmT;
            gmT.SetGlobalBuffer(gmCoeffsBase + tOff);
            DataCopy(ubT, gmT, 3 * T_SINGLE_FLOATS);

            GlobalTensor<float> gmTExp;
            gmTExp.SetGlobalBuffer(gmCoeffsBase + ctx.tExpOffset);
            DataCopy(ubT2Exp, gmTExp, 2 * elemCount);

            GlobalTensor<float> gmRadix2TW;
            gmRadix2TW.SetGlobalBuffer(gmCoeffsBase + ctx.tRadix2Offset);
            DataCopy(ubRadix2TW, gmRadix2TW, RADIX2_TW_FLOATS * 2);

            if constexpr (R2_USE_GATHER) {
                GlobalTensor<float> gmPhaseAOff;
                gmPhaseAOff.SetGlobalBuffer(gmCoeffsBase + ctx.tPhaseAOffset);
                DataCopy(ubPhaseAOff, gmPhaseAOff, PHASE_A_OFFSET_FLOATS);
            }

            GlobalTensor<float> gmT12Exp;
            gmT12Exp.SetGlobalBuffer(gmCoeffsBase + ctx.t12ExpOffset);
            DataCopy(ubT12Exp, gmT12Exp, P3_T12_EXP_FLOATS);

            GlobalTensor<float> gmP3Lane;
            gmP3Lane.SetGlobalBuffer(gmCoeffsBase + ctx.tP3LaneOffset);
            DataCopy(ubP3Lane, gmP3Lane, P3_LANE_COUNT);

            SetFlag<HardEvent::MTE2_V>(SYNC_ID0);
            WaitFlag<HardEvent::MTE2_V>(SYNC_ID0);
        }

        __ubuf__ float* t2ExpRe = (__ubuf__ float*)ubT2Exp.GetPhyAddr();
        __ubuf__ float* t2ExpIm = t2ExpRe + elemCount;
        __ubuf__ float* radix2TwTable = (__ubuf__ float*)ubRadix2TW.GetPhyAddr();
        __ubuf__ uint32_t* phaseAOffsets = (__ubuf__ uint32_t*)ubPhaseAOff.GetPhyAddr();
        __ubuf__ float* t12ExpRe = (__ubuf__ float*)ubT12Exp.GetPhyAddr();
        __ubuf__ float* t12ExpIm = t12ExpRe + elemCount;
        __ubuf__ uint32_t* p3Lane = (__ubuf__ uint32_t*)ubP3Lane.GetPhyAddr();

        SetFlag<HardEvent::MTE3_MTE2>(SYNC_ID0);
        SetFlag<HardEvent::MTE3_MTE2>(SYNC_ID1);

        uint32_t groups = (numPairs + 1) / 2;
        for (uint32_t i = 0; i < groups; i++) {
            uint32_t pairsInGroup = (i == groups - 1 && numPairs % 2 != 0) ? 1 : 2;

            for (uint32_t j = 0; j < pairsInGroup; j++) {
                uint32_t pair = 2 * i + j;
                if (pair >= numPairs) break;
                uint32_t gp = (ctx.rowStart / 2) + pair;
                event_t syncId = (pair & 1) ? SYNC_ID1 : SYNC_ID0;
                LocalTensor<float>& sepReal = (pair & 1) ? ubSepReal1 : ubSepReal0;
                LocalTensor<float>& sepImag = (pair & 1) ? ubSepImag1 : ubSepImag0;

                WaitFlag<HardEvent::MTE3_MTE2>(syncId);
                if constexpr (!P1357_SKELETON) {
                    Phase1R2cAivFused(ctx, subIdx, gp, syncId,
                        sepReal, sepImag, gmInputBase, gmWsBase);
                }
                AscendC::CrossCoreSetFlag<FFT_SYNC_MODE, PIPE_MTE3>(FftFlagId(pair, SLOT_B0));
            }

            for (uint32_t j = 0; j < pairsInGroup; j++) {
                uint32_t pair = 2 * i + j;
                if (pair >= numPairs) break;
                uint32_t gp = (ctx.rowStart / 2) + pair;
                event_t syncId = (pair & 1) ? SYNC_ID1 : SYNC_ID0;
                LocalTensor<float>& ubS = (pair & 1) ? ubGemmBuf1 : ubGemmBuf0;
                LocalTensor<float>& sepReal = (pair & 1) ? ubSepReal1 : ubSepReal0;
                LocalTensor<float>& sepImag = (pair & 1) ? ubSepImag1 : ubSepImag0;

                AscendC::CrossCoreWaitFlag<FFT_SYNC_MODE, PIPE_V>(FftFlagId(pair, SLOT_G0));
                if constexpr (!P1357_SKELETON) {
                    Phase3Aiv(ctx, subIdx, gp, syncId,
                        ubS, sepReal, sepImag, t12ExpRe, t12ExpIm, p3Lane, gmWsBase);
                }
                AscendC::CrossCoreSetFlag<FFT_SYNC_MODE, PIPE_MTE3>(FftFlagId(pair, SLOT_B1));
            }

            for (uint32_t j = 0; j < pairsInGroup; j++) {
                uint32_t pair = 2 * i + j;
                if (pair >= numPairs) break;
                uint32_t gp = (ctx.rowStart / 2) + pair;
                event_t syncId = (pair & 1) ? SYNC_ID1 : SYNC_ID0;
                LocalTensor<float>& ubS = (pair & 1) ? ubGemmBuf1 : ubGemmBuf0;
                LocalTensor<float>& sepRe = (pair & 1) ? ubSepReal1 : ubSepReal0;
                LocalTensor<float>& sepIm = (pair & 1) ? ubSepImag1 : ubSepImag0;

                AscendC::CrossCoreWaitFlag<FFT_SYNC_MODE, PIPE_V>(FftFlagId(pair, SLOT_G1));
                if constexpr (!SYNC_ONLY) {
                    Phase5Radix2Aiv(ctx, subIdx, gp, syncId,
                        ubS, sepRe, sepIm,
                        t2ExpRe, t2ExpIm, radix2TwTable, phaseAOffsets);
                }
                AscendC::PipeBarrier<PIPE_V>();
            }

            for (uint32_t j = 0; j < pairsInGroup; j++) {
                uint32_t pair = 2 * i + j;
                if (pair >= numPairs) break;
                uint32_t gp = (ctx.rowStart / 2) + pair;
                event_t syncId = (pair & 1) ? SYNC_ID1 : SYNC_ID0;
                LocalTensor<float>& sepRe = (pair & 1) ? ubSepReal1 : ubSepReal0;
                LocalTensor<float>& sepIm = (pair & 1) ? ubSepImag1 : ubSepImag0;

                if constexpr (!P1357_SKELETON) {
                    Phase7R2cAivFused(ctx, subIdx, gp, syncId,
                        sepRe, sepIm, gmOutputBase);
                }
                SetFlag<HardEvent::MTE3_MTE2>(syncId);
            }
        }

        WaitFlag<HardEvent::MTE3_MTE2>(SYNC_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(SYNC_ID1);
    }

    AscendC::PipeBarrier<PIPE_ALL>();
}

// ═══════════════════════════════════════════════════════════════════════════
// Stage 2/4：64×64 复 tile 矩形块转置（与 fft_transpose_kernel.cpp 一致）
// ═══════════════════════════════════════════════════════════════════════════

constexpr uint32_t TR_TILE_C = 64;
constexpr uint32_t TR_TILE_R = 64;
constexpr uint32_t TR_W_IN  = TR_TILE_C * 2;
constexpr uint32_t TR_W_OUT = TR_TILE_R * 2;
constexpr uint32_t TR_TILE_FLOATS = TR_TILE_R * TR_W_IN;
constexpr uint32_t TR_PAD_COMPLEX = TR_TILE_C + 3;
constexpr uint32_t TR_SHARED_STRIDE = TR_PAD_COMPLEX * 2;
constexpr uint32_t TR_TILE_HALF = TR_TILE_R / 2;
constexpr uint32_t TR_THREADS = TR_TILE_HALF * TR_TILE_HALF;
constexpr uint32_t TR_SHARED_FLOATS = TR_TILE_R * TR_SHARED_STRIDE;
constexpr uint32_t TR_GROUP_FLOATS = TR_TILE_FLOATS;
constexpr uint32_t TR_BUF_FLOATS = 2 * TR_GROUP_FLOATS + TR_SHARED_FLOATS;

__simt_vf__ __launch_bounds__(TR_THREADS) inline void simt_transpose_c64x64(
    __ubuf__ float* dst, const __ubuf__ float* src, __ubuf__ float* sharedTile)
{
    uint32_t ty = threadIdx.y;
    uint32_t tx = threadIdx.x;

    const __ubuf__ float2* src2 = (const __ubuf__ float2*)src;
    __ubuf__ float2* shared2 = (__ubuf__ float2*)sharedTile;
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

__aicore__ inline void RunTransposeRectStage(
    GM_ADDR src, GM_ADDR dst,
    int32_t srcRows, int32_t srcCols,
    int32_t srcRowFloats, int32_t dstRowFloats,
    int32_t vecCoreNumIn)
{
    if ASCEND_IS_AIC {
        AscendC::PipeBarrier<PIPE_ALL>();
        return;
    }

    uint32_t vecIdx = AscendC::GetBlockIdx();
    uint32_t vecCoreNum = static_cast<uint32_t>(vecCoreNumIn);
    if (vecIdx >= vecCoreNum) {
        AscendC::PipeBarrier<PIPE_ALL>();
        return;
    }

    const int64_t sRowFloats = static_cast<int64_t>(srcRowFloats);
    const int64_t dRowFloats = static_cast<int64_t>(dstRowFloats);
    const uint32_t numRowBlocks = static_cast<uint32_t>(srcRows) / TR_TILE_R;
    const uint32_t numColBlocks = static_cast<uint32_t>(srcCols) / TR_TILE_C;
    const uint32_t totalTiles = numRowBlocks * numColBlocks;

    const uint32_t tilesPerAiv = totalTiles / vecCoreNum;
    const uint32_t tilesRemain = totalTiles % vecCoreNum;
    const uint32_t myStart = vecIdx * tilesPerAiv + (vecIdx < tilesRemain ? vecIdx : tilesRemain);
    const uint32_t myCount = tilesPerAiv + (vecIdx < tilesRemain ? 1 : 0);
    if (myCount == 0) {
        AscendC::PipeBarrier<PIPE_ALL>();
        return;
    }

    Arch::Resource<FftArchTag> resource;

    __gm__ float* gmIn = (__gm__ float*)src;
    __gm__ float* gmOut = (__gm__ float*)dst;

    LocalTensor<float> srcTensor[2];
    LocalTensor<float> dstTensor[2];
    __ubuf__ float* srcAddr[2];
    __ubuf__ float* dstAddr[2];
    __ubuf__ float* shAddr[2];
    for (uint32_t b = 0; b < 2; b++) {
        srcTensor[b] = resource.ubBuf.template GetBufferByByte<float>(
            static_cast<int64_t>(b) * TR_BUF_FLOATS * sizeof(float));
        dstTensor[b] = resource.ubBuf.template GetBufferByByte<float>(
            (static_cast<int64_t>(b) * TR_BUF_FLOATS + TR_GROUP_FLOATS) * sizeof(float));
        srcAddr[b] = (__ubuf__ float*)srcTensor[b].GetPhyAddr();
        dstAddr[b] = (__ubuf__ float*)dstTensor[b].GetPhyAddr();
        shAddr[b] = (__ubuf__ float*)resource.ubBuf.template GetBufferByByte<float>(
            (static_cast<int64_t>(b) * TR_BUF_FLOATS + 2 * TR_GROUP_FLOATS) * sizeof(float))
            .GetPhyAddr();
    }

    constexpr event_t EVT_M2_V = EVENT_ID0;
    constexpr event_t EVT_V_M3 = EVENT_ID1;
    constexpr event_t EVT_M3_M2 = EVENT_ID2;

    const uint16_t readGap = static_cast<uint16_t>((sRowFloats - TR_W_IN) * sizeof(float) / 32);
    const uint16_t writeGap = static_cast<uint16_t>((dRowFloats - TR_W_OUT) * sizeof(float) / 32);

    for (uint32_t g = 0; g < myCount; g++) {
        uint32_t bufIdx = g & 1u;
        uint32_t tile = myStart + g;
        uint32_t tr = tile / numColBlocks;
        uint32_t tc = tile - tr * numColBlocks;

        {
            GlobalTensor<float> gmSrc;
            int64_t rowBase = static_cast<int64_t>(tr * TR_TILE_R) * sRowFloats
                            + static_cast<int64_t>(tc * TR_W_IN);
            gmSrc.SetGlobalBuffer(gmIn + rowBase);
            DataCopyParams cp;
            cp.blockCount = TR_TILE_R;
            cp.blockLen = TR_W_IN * sizeof(float) / 32;
            cp.srcGap = readGap;
            cp.dstGap = 0;
            DataCopy(srcTensor[bufIdx], gmSrc, cp);
        }
        SetFlag<HardEvent::MTE2_V>(EVT_M2_V);
        WaitFlag<HardEvent::MTE2_V>(EVT_M2_V);

        if (g >= 1) {
            uint32_t prevIdx = 1u - bufIdx;
            uint32_t ptile = myStart + g - 1;
            uint32_t ptr = ptile / numColBlocks;
            uint32_t ptc = ptile - ptr * numColBlocks;

            WaitFlag<HardEvent::V_MTE3>(EVT_V_M3);
            GlobalTensor<float> gmDst;
            int64_t dstRowBase = static_cast<int64_t>(ptc * TR_TILE_C) * dRowFloats
                               + static_cast<int64_t>(ptr * TR_W_OUT);
            gmDst.SetGlobalBuffer(gmOut + dstRowBase);
            DataCopyParams cpw;
            cpw.blockCount = TR_TILE_C;
            cpw.blockLen = TR_W_OUT * sizeof(float) / 32;
            cpw.srcGap = 0;
            cpw.dstGap = writeGap;
            DataCopy(gmDst, dstTensor[prevIdx], cpw);
            SetFlag<HardEvent::MTE3_MTE2>(EVT_M3_M2);
            WaitFlag<HardEvent::MTE3_MTE2>(EVT_M3_M2);
        }

        asc_vf_call<simt_transpose_c64x64>(
            dim3(TR_TILE_HALF, TR_TILE_HALF, 1), dstAddr[bufIdx], srcAddr[bufIdx], shAddr[bufIdx]);
        AscendC::DataSyncBarrier<AscendC::MemDsbT::UB>();
        SetFlag<HardEvent::V_MTE3>(EVT_V_M3);
    }

    {
        uint32_t lastIdx = (myCount - 1) & 1u;
        uint32_t lTile = myStart + myCount - 1;
        uint32_t ltr = lTile / numColBlocks;
        uint32_t ltc = lTile - ltr * numColBlocks;

        WaitFlag<HardEvent::V_MTE3>(EVT_V_M3);
        GlobalTensor<float> gmDst;
        int64_t dstRowBase = static_cast<int64_t>(ltc * TR_TILE_C) * dRowFloats
                           + static_cast<int64_t>(ltr * TR_W_OUT);
        gmDst.SetGlobalBuffer(gmOut + dstRowBase);
        DataCopyParams cpw;
        cpw.blockCount = TR_TILE_C;
        cpw.blockLen = TR_W_OUT * sizeof(float) / 32;
        cpw.srcGap = 0;
        cpw.dstGap = writeGap;
        DataCopy(gmDst, dstTensor[lastIdx], cpw);
        SetFlag<HardEvent::MTE3_MTE2>(EVT_M3_M2);
        WaitFlag<HardEvent::MTE3_MTE2>(EVT_M3_M2);
    }

    AscendC::PipeBarrier<PIPE_MTE3>();
    AscendC::PipeBarrier<PIPE_ALL>();
}

// ═══════════════════════════════════════════════════════════════════════════
// Stage 3：纵向 C2C 1D（与 fft_c2c1d_kernel.cpp 的 FftC2c1dKernel 主体一致）
// ═══════════════════════════════════════════════════════════════════════════

CATLASS_DEVICE void Phase1C2cAivFused(
    FftKernelContext& ctx, uint32_t subIdx, uint32_t globalPairIdx,
    event_t syncId,
    LocalTensor<float>& ubInput, LocalTensor<float>& ubSepReal, LocalTensor<float>& ubSepImag,
    __ubuf__ float* inputAddr, __ubuf__ float* sepRealAddr, __ubuf__ float* sepImagAddr,
    uint16_t deinterleaveIters,
    __gm__ float* gmInputBase, __gm__ float* gmWsBase)
{
    uint32_t myRow = globalPairIdx * 2 + subIdx;
    int64_t subCount = static_cast<int64_t>(ctx.N2) * ctx.N2;
    int64_t colOffset = static_cast<int64_t>(subIdx) * subCount;

    if constexpr (!SYNC_ONLY) {
        int64_t inOff = static_cast<int64_t>(myRow) * ctx.rowFloats;
        GlobalTensor<float> gmIn;
        gmIn.SetGlobalBuffer(gmInputBase + inOff);
        DataCopy(ubInput, gmIn, static_cast<uint32_t>(ctx.rowFloats));
    }

    SetFlag<HardEvent::MTE2_V>(syncId);
    WaitFlag<HardEvent::MTE2_V>(syncId);

    asc_vf_call<simd_deinterleave_row>(sepRealAddr, sepImagAddr, inputAddr, deinterleaveIters);
    AscendC::DataSyncBarrier<AscendC::MemDsbT::UB>();

    SetFlag<HardEvent::V_MTE3>(syncId);
    WaitFlag<HardEvent::V_MTE3>(syncId);

    if constexpr (!SYNC_ONLY) {
        int64_t bMatrixFloats = static_cast<int64_t>(ctx.mmK[0]) * ctx.mmN[0];
        int64_t baseOff = static_cast<int64_t>(ctx.blockIdx) * (6 * bMatrixFloats)
                              + static_cast<int64_t>(globalPairIdx & 1) * bMatrixFloats
                              + ctx.wsOffset[0];

        DataCopyParams copyParams;
        copyParams.blockCount = static_cast<uint16_t>(ctx.N1);
        copyParams.blockLen = static_cast<uint16_t>(subCount * sizeof(float) / 32);
        copyParams.srcGap = 0;
        copyParams.dstGap = static_cast<uint16_t>(subCount * sizeof(float) / 32);

        GlobalTensor<float> gmReal;
        gmReal.SetGlobalBuffer(gmWsBase + baseOff + colOffset);
        DataCopy(gmReal, ubSepReal, copyParams);

        GlobalTensor<float> gmImag;
        gmImag.SetGlobalBuffer(gmWsBase + baseOff
            + static_cast<int64_t>(ctx.N1) * ctx.mmN[0] + colOffset);
        DataCopy(gmImag, ubSepImag, copyParams);
    }
}

CATLASS_DEVICE void Phase7C2cAivFused(
    FftKernelContext& ctx, uint32_t subIdx, uint32_t globalPairIdx,
    event_t syncId,
    LocalTensor<float>& ubSepRe, LocalTensor<float>& ubSepIm,
    LocalTensor<float>& ubInter,
    __gm__ float* gmOutputBase)
{
    uint32_t myRow = globalPairIdx * 2 + subIdx;
    int64_t subCount = static_cast<int64_t>(ctx.N2) * ctx.N2;

    __ubuf__ float* sepReAddr = (__ubuf__ float*)ubSepRe.GetPhyAddr();
    __ubuf__ float* sepImAddr = (__ubuf__ float*)ubSepIm.GetPhyAddr();
    __ubuf__ float* interAddr = (__ubuf__ float*)ubInter.GetPhyAddr();

    uint32_t interRows = static_cast<uint32_t>(ctx.N1);
    asc_vf_call<simd_interleave_rows_vf>(
        interAddr, sepReAddr, sepImAddr, static_cast<uint16_t>(interRows));

    AscendC::PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_MTE3>(syncId);
    WaitFlag<HardEvent::V_MTE3>(syncId);

    if constexpr (!SYNC_ONLY) {
        DataCopyParams outParams;
        outParams.blockCount = static_cast<uint16_t>(ctx.N1);
        outParams.blockLen = static_cast<uint16_t>(subCount * 2 * sizeof(float) / 32);
        outParams.srcGap = 0;
        outParams.dstGap = 0;

        GlobalTensor<float> gmOut;
        gmOut.SetGlobalBuffer(gmOutputBase + static_cast<int64_t>(myRow) * ctx.outRowFloats);
        DataCopy(gmOut, ubInter, outParams);
    }
}

__aicore__ inline void RunC2c1dStage(
    GM_ADDR input, GM_ADDR output, GM_ADDR coeffs, GM_ADDR workspace, GM_ADDR tiling)
{
    FftKernelContext ctx;
    InitContext((__gm__ FftTilingData*)tiling, ctx);
    ctx.rowFloats = ctx.fftN * 2;
    ctx.outRowFloats = ctx.fftN * 2;

    if (ctx.blockIdx >= static_cast<uint32_t>(
            ((__gm__ FftTilingData*)tiling)->multiCoreParams.coreNum)) {
        AscendC::PipeBarrier<PIPE_ALL>();
        return;
    }

    Arch::Resource<ArchTag> resource;

    int64_t elemCount = static_cast<int64_t>(ctx.N1) * ctx.N2 * ctx.N2;

    __gm__ float* gmInputBase = (__gm__ float*)(input);
    __gm__ float* gmOutputBase = (__gm__ float*)(output);
    __gm__ float* gmWsBase = (__gm__ float*)(workspace);
    __gm__ float* gmCoeffsBase = (__gm__ float*)(coeffs);

    uint32_t numPairs = (ctx.rowEnd - ctx.rowStart) / 2;

    if ASCEND_IS_AIC {
        FftBlockMmadToUb blockMmadToUb(resource, 0);
        LocalTensor<float> ubGemmBuf0 = resource.ubBuf.template GetBufferByByte<float>(
            UB_OFFSET_GEMM * sizeof(float));
        LocalTensor<float> ubGemmBuf1 = resource.ubBuf.template GetBufferByByte<float>(
            UB_OFFSET_GEMM_1 * sizeof(float));

        int64_t bMatrixFloats = static_cast<int64_t>(ctx.mmK[0]) * ctx.mmN[0];

        uint32_t groups = (numPairs + 1) / 2;
        for (uint32_t i = 0; i < groups; i++) {
            uint32_t pairsInGroup = (i == groups - 1 && numPairs % 2 != 0) ? 1 : 2;

            for (uint32_t j = 0; j < pairsInGroup; j++) {
                uint32_t pair = 2 * i + j;
                if (pair >= numPairs) break;
                uint32_t globalPairIdx = (ctx.rowStart / 2) + pair;
                LocalTensor<float>& ubGemm = (j == 0) ? ubGemmBuf0 : ubGemmBuf1;
                AscendC::CrossCoreWaitFlag<FFT_SYNC_MODE, PIPE_MTE2>(FftFlagId(pair, SLOT_B0));
                if constexpr (!AIC_SKELETON) {
                    int64_t bOff0 = static_cast<int64_t>(ctx.blockIdx) * (6 * bMatrixFloats)
                                  + static_cast<int64_t>(globalPairIdx & 1) * bMatrixFloats
                                  + ctx.wsOffset[0];
                    RunGemmToUb(blockMmadToUb,
                        coeffs, ctx.wOffset[0], workspace, bOff0,
                        ubGemm, ctx.mmM[0], ctx.mmK[0], ctx.mmN[0]);
                }
                AscendC::CrossCoreSetFlag<FFT_SYNC_MODE, PIPE_FIX>(FftFlagId(pair, SLOT_G0));
            }

            for (uint32_t j = 0; j < pairsInGroup; j++) {
                uint32_t pair = 2 * i + j;
                if (pair >= numPairs) break;
                uint32_t globalPairIdx = (ctx.rowStart / 2) + pair;
                LocalTensor<float>& ubGemm = (j == 0) ? ubGemmBuf0 : ubGemmBuf1;
                AscendC::CrossCoreWaitFlag<FFT_SYNC_MODE, PIPE_MTE2>(FftFlagId(pair, SLOT_B1));
                if constexpr (!AIC_SKELETON) {
                    int64_t bOff1 = static_cast<int64_t>(ctx.blockIdx) * (6 * bMatrixFloats)
                                  + static_cast<int64_t>(globalPairIdx & 1) * bMatrixFloats
                                  + ctx.wsOffset[2];
                    RunGemmToUb(blockMmadToUb,
                        coeffs, ctx.wOffset[0], workspace, bOff1,
                        ubGemm, ctx.mmM[1], ctx.mmK[1], ctx.mmN[1]);
                }
                AscendC::CrossCoreSetFlag<FFT_SYNC_MODE, PIPE_FIX>(FftFlagId(pair, SLOT_G1));
            }
        }
    }

    if ASCEND_IS_AIV {
        uint32_t subIdx = AscendC::GetSubBlockIdx();
        int64_t tOff = ctx.tOffset;

        LocalTensor<float> ubSepReal0 = resource.ubBuf.template GetBufferByByte<float>(
            UB_OFFSET_SEP0 * sizeof(float));
        LocalTensor<float> ubSepImag0 = resource.ubBuf.template GetBufferByByte<float>(
            (UB_OFFSET_SEP0 + elemCount) * sizeof(float));
        LocalTensor<float> ubSepReal1 = resource.ubBuf.template GetBufferByByte<float>(
            UB_OFFSET_SEP1 * sizeof(float));
        LocalTensor<float> ubSepImag1 = resource.ubBuf.template GetBufferByByte<float>(
            (UB_OFFSET_SEP1 + elemCount) * sizeof(float));
        LocalTensor<float> ubGemmBuf0 = resource.ubBuf.template GetBufferByByte<float>(
            UB_OFFSET_GEMM * sizeof(float));
        LocalTensor<float> ubGemmBuf1 = resource.ubBuf.template GetBufferByByte<float>(
            UB_OFFSET_GEMM_1 * sizeof(float));
        LocalTensor<float> ubT = resource.ubBuf.template GetBufferByByte<float>(
            UB_OFFSET_T * sizeof(float));
        LocalTensor<float> ubT2Exp = resource.ubBuf.template GetBufferByByte<float>(
            UB_OFFSET_T2_EXP * sizeof(float));
        LocalTensor<float> ubRadix2TW = resource.ubBuf.template GetBufferByByte<float>(
            UB_OFFSET_RADIX2_TW * sizeof(float));
        LocalTensor<float> ubPhaseAOff = resource.ubBuf.template GetBufferByByte<float>(
            UB_OFFSET_PHASE_A * sizeof(float));
        LocalTensor<float> ubT12Exp = resource.ubBuf.template GetBufferByByte<float>(
            UB_OFFSET_T12_EXP * sizeof(float));
        LocalTensor<float> ubP3Lane = resource.ubBuf.template GetBufferByByte<float>(
            UB_OFFSET_P3_LANE * sizeof(float));

        if constexpr (!SYNC_ONLY) {
            GlobalTensor<float> gmT;
            gmT.SetGlobalBuffer(gmCoeffsBase + tOff);
            DataCopy(ubT, gmT, 3 * T_SINGLE_FLOATS);

            GlobalTensor<float> gmTExp;
            gmTExp.SetGlobalBuffer(gmCoeffsBase + ctx.tExpOffset);
            DataCopy(ubT2Exp, gmTExp, 2 * elemCount);

            GlobalTensor<float> gmRadix2TW;
            gmRadix2TW.SetGlobalBuffer(gmCoeffsBase + ctx.tRadix2Offset);
            DataCopy(ubRadix2TW, gmRadix2TW, RADIX2_TW_FLOATS * 2);

            if constexpr (R2_USE_GATHER) {
                GlobalTensor<float> gmPhaseAOff;
                gmPhaseAOff.SetGlobalBuffer(gmCoeffsBase + ctx.tPhaseAOffset);
                DataCopy(ubPhaseAOff, gmPhaseAOff, PHASE_A_OFFSET_FLOATS);
            }

            GlobalTensor<float> gmT12Exp;
            gmT12Exp.SetGlobalBuffer(gmCoeffsBase + ctx.t12ExpOffset);
            DataCopy(ubT12Exp, gmT12Exp, P3_T12_EXP_FLOATS);

            GlobalTensor<float> gmP3Lane;
            gmP3Lane.SetGlobalBuffer(gmCoeffsBase + ctx.tP3LaneOffset);
            DataCopy(ubP3Lane, gmP3Lane, P3_LANE_COUNT);

            SetFlag<HardEvent::MTE2_V>(SYNC_ID0);
            WaitFlag<HardEvent::MTE2_V>(SYNC_ID0);
        }

        __ubuf__ float* t2ExpRe = (__ubuf__ float*)ubT2Exp.GetPhyAddr();
        __ubuf__ float* t2ExpIm = t2ExpRe + elemCount;
        __ubuf__ float* radix2TwTable = (__ubuf__ float*)ubRadix2TW.GetPhyAddr();
        __ubuf__ uint32_t* phaseAOffsets = (__ubuf__ uint32_t*)ubPhaseAOff.GetPhyAddr();
        __ubuf__ float* t12ExpRe = (__ubuf__ float*)ubT12Exp.GetPhyAddr();
        __ubuf__ float* t12ExpIm = t12ExpRe + elemCount;
        __ubuf__ uint32_t* p3Lane = (__ubuf__ uint32_t*)ubP3Lane.GetPhyAddr();

        uint16_t deinterleaveIters = static_cast<uint16_t>(ctx.fftN / GATHER_MASK_PER_REPEAT);

        SetFlag<HardEvent::MTE3_MTE2>(SYNC_ID0);
        SetFlag<HardEvent::MTE3_MTE2>(SYNC_ID1);

        uint32_t groups = (numPairs + 1) / 2;
        for (uint32_t i = 0; i < groups; i++) {
            uint32_t pairsInGroup = (i == groups - 1 && numPairs % 2 != 0) ? 1 : 2;

            for (uint32_t j = 0; j < pairsInGroup; j++) {
                uint32_t pair = 2 * i + j;
                if (pair >= numPairs) break;
                uint32_t gp = (ctx.rowStart / 2) + pair;
                event_t syncId = (pair & 1) ? SYNC_ID1 : SYNC_ID0;
                LocalTensor<float>& ubS = (pair & 1) ? ubGemmBuf1 : ubGemmBuf0;
                LocalTensor<float>& sepReal = (pair & 1) ? ubSepReal1 : ubSepReal0;
                LocalTensor<float>& sepImag = (pair & 1) ? ubSepImag1 : ubSepImag0;
                __ubuf__ float* inputAddr = (__ubuf__ float*)ubS.GetPhyAddr();
                __ubuf__ float* sepRealAddr = (__ubuf__ float*)sepReal.GetPhyAddr();
                __ubuf__ float* sepImagAddr = (__ubuf__ float*)sepImag.GetPhyAddr();

                WaitFlag<HardEvent::MTE3_MTE2>(syncId);
                if constexpr (!P1357_SKELETON) {
                    Phase1C2cAivFused(ctx, subIdx, gp, syncId,
                        ubS, sepReal, sepImag,
                        inputAddr, sepRealAddr, sepImagAddr, deinterleaveIters,
                        gmInputBase, gmWsBase);
                }
                AscendC::CrossCoreSetFlag<FFT_SYNC_MODE, PIPE_MTE3>(FftFlagId(pair, SLOT_B0));
            }

            for (uint32_t j = 0; j < pairsInGroup; j++) {
                uint32_t pair = 2 * i + j;
                if (pair >= numPairs) break;
                uint32_t gp = (ctx.rowStart / 2) + pair;
                event_t syncId = (pair & 1) ? SYNC_ID1 : SYNC_ID0;
                LocalTensor<float>& ubS = (pair & 1) ? ubGemmBuf1 : ubGemmBuf0;
                LocalTensor<float>& sepReal = (pair & 1) ? ubSepReal1 : ubSepReal0;
                LocalTensor<float>& sepImag = (pair & 1) ? ubSepImag1 : ubSepImag0;

                AscendC::CrossCoreWaitFlag<FFT_SYNC_MODE, PIPE_V>(FftFlagId(pair, SLOT_G0));
                if constexpr (!P1357_SKELETON) {
                    Phase3Aiv(ctx, subIdx, gp, syncId,
                        ubS, sepReal, sepImag, t12ExpRe, t12ExpIm, p3Lane, gmWsBase);
                }
                AscendC::CrossCoreSetFlag<FFT_SYNC_MODE, PIPE_MTE3>(FftFlagId(pair, SLOT_B1));
            }

            for (uint32_t j = 0; j < pairsInGroup; j++) {
                uint32_t pair = 2 * i + j;
                if (pair >= numPairs) break;
                uint32_t gp = (ctx.rowStart / 2) + pair;
                event_t syncId = (pair & 1) ? SYNC_ID1 : SYNC_ID0;
                LocalTensor<float>& ubS = (pair & 1) ? ubGemmBuf1 : ubGemmBuf0;
                LocalTensor<float>& sepRe = (pair & 1) ? ubSepReal1 : ubSepReal0;
                LocalTensor<float>& sepIm = (pair & 1) ? ubSepImag1 : ubSepImag0;

                AscendC::CrossCoreWaitFlag<FFT_SYNC_MODE, PIPE_V>(FftFlagId(pair, SLOT_G1));
                if constexpr (!SYNC_ONLY) {
                    Phase5Radix2Aiv(ctx, subIdx, gp, syncId,
                        ubS, sepRe, sepIm,
                        t2ExpRe, t2ExpIm, radix2TwTable, phaseAOffsets);
                }
                AscendC::PipeBarrier<PIPE_V>();
            }

            for (uint32_t j = 0; j < pairsInGroup; j++) {
                uint32_t pair = 2 * i + j;
                if (pair >= numPairs) break;
                uint32_t gp = (ctx.rowStart / 2) + pair;
                event_t syncId = (pair & 1) ? SYNC_ID1 : SYNC_ID0;
                LocalTensor<float>& ubInter = (pair & 1) ? ubGemmBuf1 : ubGemmBuf0;
                LocalTensor<float>& sepRe = (pair & 1) ? ubSepReal1 : ubSepReal0;
                LocalTensor<float>& sepIm = (pair & 1) ? ubSepImag1 : ubSepImag0;

                if constexpr (!P1357_SKELETON) {
                    Phase7C2cAivFused(ctx, subIdx, gp, syncId,
                        sepRe, sepIm, ubInter, gmOutputBase);
                }
                SetFlag<HardEvent::MTE3_MTE2>(syncId);
            }
        }

        WaitFlag<HardEvent::MTE3_MTE2>(SYNC_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(SYNC_ID1);
    }

    AscendC::PipeBarrier<PIPE_ALL>();
}

// ═══════════════════════════════════════════════════════════════════════════
// Stage 5：裁剪 w2[4096,2112] -> out[4096,2049]（DataCopyPad<Compact>）
// ═══════════════════════════════════════════════════════════════════════════

constexpr uint32_t CROP_VEC_CORES = 56;

__aicore__ inline void RunCropStage(
    GM_ADDR src, GM_ADDR dst, int32_t numRows, int32_t srcRowFloats, int32_t dstRowFloats)
{
    if ASCEND_IS_AIC {
        AscendC::PipeBarrier<PIPE_ALL>();
        return;
    }

    uint32_t vecIdx = AscendC::GetBlockIdx();
    if (vecIdx >= CROP_VEC_CORES) {
        AscendC::PipeBarrier<PIPE_ALL>();
        return;
    }

    uint32_t nRows = static_cast<uint32_t>(numRows);
    uint32_t rowsPerAiv = nRows / CROP_VEC_CORES;
    uint32_t rem = nRows % CROP_VEC_CORES;
    uint32_t start = vecIdx * rowsPerAiv + (vecIdx < rem ? vecIdx : rem);
    uint32_t count = rowsPerAiv + (vecIdx < rem ? 1 : 0);

    Arch::Resource<FftArchTag> resource;
    LocalTensor<float> ubBuf = resource.ubBuf.template GetBufferByByte<float>(0);

    __gm__ float* gmIn = (__gm__ float*)src;
    __gm__ float* gmOut = (__gm__ float*)dst;

    const uint32_t copyBytes = static_cast<uint32_t>(dstRowFloats * 4);
    constexpr event_t EVT = EVENT_ID0;

    for (uint32_t r = 0; r < count; r++) {
        uint32_t row = start + r;

        GlobalTensor<float> gmSrc;
        gmSrc.SetGlobalBuffer(gmIn + static_cast<int64_t>(row) * srcRowFloats);
        DataCopyPad<float, AscendC::PaddingMode::Compact>(
            ubBuf, gmSrc,
            DataCopyExtParams{1, copyBytes, 0, 0, 0},
            DataCopyPadExtParams<float>{false, 0, 0, 0});
        SetFlag<HardEvent::MTE2_MTE3>(EVT);
        WaitFlag<HardEvent::MTE2_MTE3>(EVT);

        GlobalTensor<float> gmDst;
        gmDst.SetGlobalBuffer(gmOut + static_cast<int64_t>(row) * dstRowFloats);
        DataCopyPad<float, AscendC::PaddingMode::Compact>(
            gmDst, ubBuf,
            DataCopyExtParams{1, copyBytes, 0, 0, 0});
        SetFlag<HardEvent::MTE3_MTE2>(EVT);
        WaitFlag<HardEvent::MTE3_MTE2>(EVT);
    }

    AscendC::PipeBarrier<PIPE_MTE3>();
    AscendC::PipeBarrier<PIPE_ALL>();
}

// ═══════════════════════════════════════════════════════════════════════════
// 融合 Kernel：单 launch 完成 R2C1D → 转置 → C2C1D → 转置 → 裁剪
// 阶段间 SyncAll<false>()；每段 stage 尾部已 PipeBarrier<PIPE_ALL> 排空。
// ═══════════════════════════════════════════════════════════════════════════

CATLASS_GLOBAL __mix__(1, 2) void FftR2C2DFusedKernel(
    GM_ADDR input, GM_ADDR output, GM_ADDR coeffs, GM_ADDR workspace,
    GM_ADDR tiling1, GM_ADDR tiling2,
    GM_ADDR y, GM_ADDR z, GM_ADDR w, GM_ADDR w2,
    int32_t vecCoreNum)
{
    const int32_t fftN = 4096;
    const int32_t halfN = fftN / 2 + 1;
    const int32_t padHalf = (halfN + 63) & ~int32_t(63);

    // S1: 横向 R2C 1D：x -> y[fftN,padHalf]
    RunR2c1dStage(input, y, coeffs, workspace, tiling1);
    AscendC::SyncAll<false>();

    // S2: 转置 y[fftN,padHalf] -> z[padHalf,fftN]
    RunTransposeRectStage(y, z, fftN, padHalf, padHalf * 2, fftN * 2, vecCoreNum);
    AscendC::SyncAll<false>();

    // S3: 纵向 C2C 1D：z[padHalf,fftN] -> w[padHalf,fftN]
    RunC2c1dStage(z, w, coeffs, workspace, tiling2);
    AscendC::SyncAll<false>();

    // S4: 转置 w[padHalf,fftN] -> w2[fftN,padHalf]
    RunTransposeRectStage(w, w2, padHalf, fftN, fftN * 2, padHalf * 2, vecCoreNum);
    AscendC::SyncAll<false>();

    // S5: 裁剪 w2[fftN,padHalf] -> out[fftN,halfN]
    RunCropStage(w2, output, fftN, padHalf * 2, halfN * 2);
}

void FftR2C2DFusedKernelLaunch(
    GM_ADDR input, GM_ADDR output, GM_ADDR coeffs, GM_ADDR workspace,
    GM_ADDR tiling1, GM_ADDR tiling2,
    GM_ADDR y, GM_ADDR z, GM_ADDR w, GM_ADDR w2,
    uint32_t blockDim, aclrtStream stream)
{
    constexpr uint32_t workBuff = 216 * 1024;
    FftR2C2DFusedKernel<<<blockDim, workBuff, stream>>>(
        input, output, coeffs, workspace, tiling1, tiling2, y, z, w, w2,
        static_cast<int32_t>(blockDim * 2));
}

} // namespace Fft
