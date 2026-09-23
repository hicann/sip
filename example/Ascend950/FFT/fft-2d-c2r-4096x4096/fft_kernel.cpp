#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include "catlass/arch/arch.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/catlass.hpp"
#include "catlass/status.hpp"

#include "fft_kernel.h"
#include "fft_kernel_utils.h"
#include "fft_tiling_def.h"
#include "fft_simt_ops.h"
#include "fft_radix16_ops.h"
#include "fft_transpose_stage.h"

#include "kernel_operator.h"
#include "helper.hpp"

using namespace Catlass;
using namespace AscendC;

namespace Fft {

using ArchTag = Arch::Ascend950;

// ===========================================================================
// radix-16 C2R UB 布局（float offsets）
//   Sep0 re/im + Sep1 re/im = 4×4096
//   T2Exp(8192) + Radix2TW(1920) + T(1536=T12/T13/T23) + PhaseA(4096 uint32)
//   S0/S1(各 8196 = GEMM 输出半片 + Pass1 的 Hermitian 输入 staging)
// ===========================================================================
constexpr int64_t UB_SEP0     = 0;        // 4×4096 = 16384
constexpr int64_t UB_T2EXP    = 16384;    // 8192
constexpr int64_t UB_RADIX2TW = 24576;    // 1920
constexpr int64_t UB_PHASEA   = 26496;    // 4096 (uint32)
constexpr int64_t UB_S0       = 30592;    // 8208
constexpr int64_t UB_S1       = 38800;    // 8208
constexpr int64_t UB_T12_EXP  = 47008;    // 8192 (P3 SIMD twiddle 展开表)
constexpr int64_t UB_P3_LANE  = 55200;    // 64 (uint32)

constexpr int32_t R16_ELEM = 4096; // N1*N2*N2（= ops 的 R16_TOTAL）

// 融合 1D Hermitian 展开 + deinterleave：dst[re]+dst[im] 从本行 packed 半谱
// src = 本行半谱基址（共 halfN 个复数），无需镜像行（irfft_axis1 的一维展开）。
// start 之前的元素由 SIMD deinterleave 已处理，此处只处理 [start, fftN)
__simt_vf__ __launch_bounds__(2048) inline void simt_hermitian_deint(
    __ubuf__ float* dstRe, __ubuf__ float* dstIm,
    const __ubuf__ float* src,
    uint32_t fftN, uint32_t halfN, uint32_t start)
{
    uint32_t tid = threadIdx.x;
    for (uint32_t v = start + tid; v < fftN; v += 2048) {
        if (v < halfN) {
            dstRe[v] = src[v * 2];
            dstIm[v] = src[v * 2 + 1];
        } else {
            uint32_t c = fftN - v;
            dstRe[v] = src[c * 2];
            dstIm[v] = -src[c * 2 + 1];
        }
    }
}

// 核心 FF 阶段：HERMITIAN（Pass4 融合 Hermitian 展开）/ REAL_OUT（Pass4 取实部）
// 未用 REAL_OUT 时输出 interleave 复
template <bool HERMITIAN, bool REAL_OUT>
CATLASS_DEVICE void RunFftPass(
    Arch::Resource<ArchTag>& resource, FftC2R2DKernelContext& ctx,
    GM_ADDR srcGM, GM_ADDR dstGM, GM_ADDR bws, GM_ADDR coeffs)
{
    int64_t fftN = ctx.fftN;
    int32_t halfN = ctx.halfN;
    int32_t mmM = ctx.mmM, mmK = ctx.mmK, mmN = ctx.mmN;

    __gm__ float* gmSrcBase = (__gm__ float*)(srcGM);
    __gm__ float* gmDstBase = (__gm__ float*)(dstGM);
    __gm__ float* gmBwsBase = (__gm__ float*)(bws);
    __gm__ float* gmCoeffsBase = (__gm__ float*)(coeffs);

    int64_t bMatrixFloats = static_cast<int64_t>(mmK) * mmN;   // 32*512=16384
    int64_t inRowStride = ctx.inRowStride;                     // 输入行距（float）
    int64_t readFloats = HERMITIAN ? (static_cast<int64_t>(halfN) * 2) : (fftN * 2);
    int64_t outRowFloats = REAL_OUT ? fftN : (fftN * 2);

    uint32_t rowStart = ctx.rowStart;
    uint32_t rowEnd = ctx.rowEnd;
    uint32_t numPairs = (rowEnd - rowStart) / 2;

    // ─── AIC：2 轮 GEMM（group 流水）───
    if ASCEND_IS_AIC {
        FftBlockMmadToUb blockMmad(resource, 0);
        LocalTensor<float> ubGemm0 = resource.ubBuf.template GetBufferByByte<float>(UB_S0 * sizeof(float));
        LocalTensor<float> ubGemm1 = resource.ubBuf.template GetBufferByByte<float>(UB_S1 * sizeof(float));

        uint32_t groups = (numPairs + 1) / 2;
        for (uint32_t i = 0; i < groups; i++) {
            uint32_t pairsInGroup = (i == groups - 1 && (numPairs % 2 != 0)) ? 1 : 2;
            for (uint32_t j = 0; j < pairsInGroup; j++) {
                uint32_t pair = 2 * i + j;
                if (pair >= numPairs) break;
                uint32_t gp = (rowStart / 2) + pair;
                LocalTensor<float>& ubGemm = (j == 0) ? ubGemm0 : ubGemm1;
                AscendC::CrossCoreWaitFlag<FFT_SYNC_MODE, PIPE_MTE2>(FftFlagId(pair, 0));
                int64_t bOff0 = static_cast<int64_t>(ctx.blockIdx) * (4 * bMatrixFloats)
                    + static_cast<int64_t>(gp & 1) * bMatrixFloats;
                RunGemmRadix16(blockMmad, coeffs, ctx.wOffset, bws, bOff0,
                               ubGemm, mmM, mmK, mmN);
                AscendC::CrossCoreSetFlag<FFT_SYNC_MODE, PIPE_FIX>(FftFlagId(pair, 1));
            }
            for (uint32_t j = 0; j < pairsInGroup; j++) {
                uint32_t pair = 2 * i + j;
                if (pair >= numPairs) break;
                uint32_t gp = (rowStart / 2) + pair;
                LocalTensor<float>& ubGemm = (j == 0) ? ubGemm0 : ubGemm1;
                AscendC::CrossCoreWaitFlag<FFT_SYNC_MODE, PIPE_MTE2>(FftFlagId(pair, 2));
                int64_t bOff1 = static_cast<int64_t>(ctx.blockIdx) * (4 * bMatrixFloats)
                    + static_cast<int64_t>(gp & 1) * bMatrixFloats + 2 * bMatrixFloats;
                RunGemmRadix16(blockMmad, coeffs, ctx.wOffset, bws, bOff1,
                               ubGemm, mmM, mmK, mmN);
                AscendC::CrossCoreSetFlag<FFT_SYNC_MODE, PIPE_FIX>(FftFlagId(pair, 3));
            }
        }
    }

    // ─── AIV：P1 → P3 → P5' → P7 ───
    if ASCEND_IS_AIV {
        uint32_t subIdx = AscendC::GetSubBlockIdx();
        int64_t colShift = static_cast<int64_t>(subIdx) * R16_SUB;

        LocalTensor<float> sepRe0 = resource.ubBuf.template GetBufferByByte<float>(UB_SEP0 * sizeof(float));
        LocalTensor<float> sepIm0 = resource.ubBuf.template GetBufferByByte<float>((UB_SEP0 + R16_ELEM) * sizeof(float));
        LocalTensor<float> sepRe1 = resource.ubBuf.template GetBufferByByte<float>((UB_SEP0 + 2 * R16_ELEM) * sizeof(float));
        LocalTensor<float> sepIm1 = resource.ubBuf.template GetBufferByByte<float>((UB_SEP0 + 3 * R16_ELEM) * sizeof(float));
        LocalTensor<float> s0 = resource.ubBuf.template GetBufferByByte<float>(UB_S0 * sizeof(float));
        LocalTensor<float> s1 = resource.ubBuf.template GetBufferByByte<float>(UB_S1 * sizeof(float));
        LocalTensor<float> ubT2Exp = resource.ubBuf.template GetBufferByByte<float>(UB_T2EXP * sizeof(float));
        LocalTensor<float> ubR2TW = resource.ubBuf.template GetBufferByByte<float>(UB_RADIX2TW * sizeof(float));
        LocalTensor<float> ubPhaseA = resource.ubBuf.template GetBufferByByte<float>(UB_PHASEA * sizeof(float));
        LocalTensor<float> ubT12Exp = resource.ubBuf.template GetBufferByByte<float>(UB_T12_EXP * sizeof(float));
        LocalTensor<float> ubP3Lane = resource.ubBuf.template GetBufferByByte<float>(UB_P3_LANE * sizeof(float));

        // 常驻表加载
        {
            GlobalTensor<float> gmTExp;
            gmTExp.SetGlobalBuffer(gmCoeffsBase + ctx.tExpandedOffset);
            DataCopy(ubT2Exp, gmTExp, 2 * R16_ELEM);
            GlobalTensor<float> gmR2;
            gmR2.SetGlobalBuffer(gmCoeffsBase + ctx.tRadix2Offset);
            DataCopy(ubR2TW, gmR2, RADIX2_TW_FLOATS_OPS * 2);
            GlobalTensor<float> gmPA;
            gmPA.SetGlobalBuffer(gmCoeffsBase + ctx.tPhaseAOffset);
            DataCopy(ubPhaseA, gmPA, 4096);
            GlobalTensor<float> gmT12E;
            gmT12E.SetGlobalBuffer(gmCoeffsBase + ctx.t12ExpandedOffset);
            DataCopy(ubT12Exp, gmT12E, P3_T12_EXP_FLOATS_OPS);
            GlobalTensor<float> gmP3L;
            gmP3L.SetGlobalBuffer(gmCoeffsBase + ctx.tP3LaneOffset);
            DataCopy(ubP3Lane, gmP3L, P3_LANE_COUNT_OPS);
            SetFlag<HardEvent::MTE2_V>(SYNC_ID0);
            WaitFlag<HardEvent::MTE2_V>(SYNC_ID0);
        }

        __ubuf__ float* t2Re = (__ubuf__ float*)ubT2Exp.GetPhyAddr();
        __ubuf__ float* t2Im = t2Re + R16_ELEM;
        __ubuf__ float* r2Tw = (__ubuf__ float*)ubR2TW.GetPhyAddr();
        __ubuf__ uint32_t* phaseA = (__ubuf__ uint32_t*)ubPhaseA.GetPhyAddr();
        __ubuf__ float* t12ExpRe = (__ubuf__ float*)ubT12Exp.GetPhyAddr();
        __ubuf__ float* t12ExpIm = t12ExpRe + R16_ELEM;
        __ubuf__ uint32_t* p3Lane = (__ubuf__ uint32_t*)ubP3Lane.GetPhyAddr();

        SetFlag<HardEvent::MTE3_MTE2>(SYNC_ID0);
        SetFlag<HardEvent::MTE3_MTE2>(SYNC_ID1);

        uint32_t groups = (numPairs + 1) / 2;
        for (uint32_t i = 0; i < groups; i++) {
            uint32_t pairsInGroup = (i == groups - 1 && (numPairs % 2 != 0)) ? 1 : 2;

            // P1：读（Hermitian 或整行）→ deinterleave → 写 B0
            for (uint32_t j = 0; j < pairsInGroup; j++) {
                uint32_t pair = 2 * i + j;
                if (pair >= numPairs) break;
                uint32_t gp = (rowStart / 2) + pair;
                uint32_t myRow = gp * 2 + subIdx;
                event_t sid = (pair & 1) ? SYNC_ID1 : SYNC_ID0;
                LocalTensor<float>& ubS = (pair & 1) ? s1 : s0;
                LocalTensor<float>& sepRe = (pair & 1) ? sepRe1 : sepRe0;
                LocalTensor<float>& sepIm = (pair & 1) ? sepIm1 : sepIm0;

                WaitFlag<HardEvent::MTE3_MTE2>(sid);

                if constexpr (HERMITIAN) {
                    // 读本行半谱（非 32B 对齐 → DataCopyPad），1D Hermitian 展开只依赖本行
                    GlobalTensor<float> gmI;
                    gmI.SetGlobalBuffer(gmSrcBase + static_cast<int64_t>(myRow) * inRowStride);
                    DataCopyPad<float, AscendC::PaddingMode::Compact>(
                        ubS, gmI,
                        DataCopyExtParams{1, static_cast<uint32_t>(readFloats * sizeof(float)), 0, 0, 0},
                        DataCopyPadExtParams<float>{false, 0, 0, 0});

                    SetFlag<HardEvent::MTE2_V>(sid);
                    WaitFlag<HardEvent::MTE2_V>(sid);

                    __ubuf__ float* sAddr = (__ubuf__ float*)ubS.GetPhyAddr();
                    __ubuf__ float* sepR = (__ubuf__ float*)sepRe.GetPhyAddr();
                    __ubuf__ float* sepI = (__ubuf__ float*)sepIm.GetPhyAddr();
                    AscendC::PipeBarrier<PIPE_V>();
                    uint32_t simdComplex = (static_cast<uint32_t>(halfN) / 64) * 64;  // 2048
                    // 直读部分 SIMD deinterleave（[0,simdComplex) 整除 64）+ 逆序 Hermitian 尾 SIMT
                    asc_vf_call<simd_deinterleave_row>(sepR, sepI, sAddr,
                        static_cast<uint16_t>(simdComplex / 64));
                    AscendC::DataSyncBarrier<AscendC::MemDsbT::UB>();
                    asc_vf_call<simt_hermitian_deint>(dim3(2048), sepR, sepI, sAddr,
                        static_cast<uint32_t>(fftN), static_cast<uint32_t>(halfN), simdComplex);
                    AscendC::DataSyncBarrier<AscendC::MemDsbT::UB>();
                } else {
                    GlobalTensor<float> gmI;
                    gmI.SetGlobalBuffer(gmSrcBase + static_cast<int64_t>(myRow) * inRowStride);
                    DataCopy(ubS, gmI, static_cast<uint32_t>(readFloats));
                    SetFlag<HardEvent::MTE2_V>(sid);
                    WaitFlag<HardEvent::MTE2_V>(sid);
                    DeinterleaveRow(sepRe, sepIm, ubS, static_cast<uint16_t>(fftN / GATHER_MASK_PER_REPEAT));
                }

                SetFlag<HardEvent::V_MTE3>(sid);
                WaitFlag<HardEvent::V_MTE3>(sid);

                // 写 B0
                int64_t baseOff = static_cast<int64_t>(ctx.blockIdx) * (4 * bMatrixFloats)
                    + static_cast<int64_t>(gp & 1) * bMatrixFloats;
                DataCopyParams cp;
                cp.blockCount = static_cast<uint16_t>(ctx.N1);
                cp.blockLen = static_cast<uint16_t>(R16_SUB * sizeof(float) / 32);
                cp.srcGap = 0;
                cp.dstGap = static_cast<uint16_t>(R16_SUB * sizeof(float) / 32);
                GlobalTensor<float> gmR;
                gmR.SetGlobalBuffer(gmBwsBase + baseOff + colShift);
                DataCopy(gmR, sepRe, cp);
                GlobalTensor<float> gmI;
                gmI.SetGlobalBuffer(gmBwsBase + baseOff + static_cast<int64_t>(ctx.N1) * mmN + colShift);
                DataCopy(gmI, sepIm, cp);
                AscendC::CrossCoreSetFlag<FFT_SYNC_MODE, PIPE_MTE3>(FftFlagId(pair, 0));
            }

            // P3：wait C0 → twiddle(T12)+轴旋转 → 写 B1
            for (uint32_t j = 0; j < pairsInGroup; j++) {
                uint32_t pair = 2 * i + j;
                if (pair >= numPairs) break;
                uint32_t gp = (rowStart / 2) + pair;
                LocalTensor<float>& ubS = (pair & 1) ? s1 : s0;
                LocalTensor<float>& sepRe = (pair & 1) ? sepRe1 : sepRe0;
                LocalTensor<float>& sepIm = (pair & 1) ? sepIm1 : sepIm0;

                AscendC::CrossCoreWaitFlag<FFT_SYNC_MODE, PIPE_V>(FftFlagId(pair, 1));
                __ubuf__ float* sAddr = (__ubuf__ float*)ubS.GetPhyAddr();
                __ubuf__ float* sepR = (__ubuf__ float*)sepRe.GetPhyAddr();
                __ubuf__ float* sepI = (__ubuf__ float*)sepIm.GetPhyAddr();
                AscendC::PipeBarrier<PIPE_V>();
                simd_twiddle_rotate_phase3(sepR, sepI, sAddr, sAddr + R16_ELEM,
                    t12ExpRe, t12ExpIm, p3Lane);
                AscendC::PipeBarrier<PIPE_V>();
                SetFlag<HardEvent::V_MTE3>((pair & 1) ? SYNC_ID1 : SYNC_ID0);
                WaitFlag<HardEvent::V_MTE3>((pair & 1) ? SYNC_ID1 : SYNC_ID0);

                int64_t baseOff = static_cast<int64_t>(ctx.blockIdx) * (4 * bMatrixFloats)
                    + static_cast<int64_t>(gp & 1) * bMatrixFloats + 2 * bMatrixFloats;
                DataCopyParams cp;
                cp.blockCount = static_cast<uint16_t>(ctx.N1);
                cp.blockLen = static_cast<uint16_t>(R16_SUB * sizeof(float) / 32);
                cp.srcGap = 0;
                cp.dstGap = static_cast<uint16_t>(R16_SUB * sizeof(float) / 32);
                GlobalTensor<float> gmR;
                gmR.SetGlobalBuffer(gmBwsBase + baseOff + colShift);
                DataCopy(gmR, sepRe, cp);
                GlobalTensor<float> gmI;
                gmI.SetGlobalBuffer(gmBwsBase + baseOff + static_cast<int64_t>(ctx.N1) * mmN + colShift);
                DataCopy(gmI, sepIm, cp);
                AscendC::CrossCoreSetFlag<FFT_SYNC_MODE, PIPE_MTE3>(FftFlagId(pair, 2));
            }

            // P5'：wait C1 → T2 Gather + radix2 butterfly → C2 in sep
            for (uint32_t j = 0; j < pairsInGroup; j++) {
                uint32_t pair = 2 * i + j;
                if (pair >= numPairs) break;
                LocalTensor<float>& ubS = (pair & 1) ? s1 : s0;
                LocalTensor<float>& sepRe = (pair & 1) ? sepRe1 : sepRe0;
                LocalTensor<float>& sepIm = (pair & 1) ? sepIm1 : sepIm0;

                AscendC::CrossCoreWaitFlag<FFT_SYNC_MODE, PIPE_V>(FftFlagId(pair, 3));
                Radix16Phase5(ubS, sepRe, sepIm, t2Re, t2Im, r2Tw, phaseA);
                AscendC::PipeBarrier<PIPE_V>();
            }

            // P7：interleave 或 取实部 → 写输出
            for (uint32_t j = 0; j < pairsInGroup; j++) {
                uint32_t pair = 2 * i + j;
                if (pair >= numPairs) break;
                uint32_t gp = (rowStart / 2) + pair;
                uint32_t myRow = gp * 2 + subIdx;
                event_t sid = (pair & 1) ? SYNC_ID1 : SYNC_ID0;
                LocalTensor<float>& ubS = (pair & 1) ? s1 : s0;
                LocalTensor<float>& sepRe = (pair & 1) ? sepRe1 : sepRe0;
                LocalTensor<float>& sepIm = (pair & 1) ? sepIm1 : sepIm0;

                if constexpr (REAL_OUT) {
                    RealExtract(ubS, sepRe, static_cast<uint32_t>(ctx.N1), static_cast<uint32_t>(R16_SUB));
                } else {
                    InterleaveC1(ubS, sepRe, sepIm, static_cast<uint32_t>(ctx.N1), static_cast<uint32_t>(R16_SUB));
                }
                SetFlag<HardEvent::V_MTE3>(sid);
                WaitFlag<HardEvent::V_MTE3>(sid);

                GlobalTensor<float> gmOut;
                gmOut.SetGlobalBuffer(gmDstBase + static_cast<int64_t>(myRow) * outRowFloats);
                DataCopy(gmOut, ubS, static_cast<uint32_t>(outRowFloats));

                SetFlag<HardEvent::MTE3_MTE2>(sid);
            }
        }

        WaitFlag<HardEvent::MTE3_MTE2>(SYNC_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(SYNC_ID1);
    }
}

// ===========================================================================
// 方案 B：共轭合并 irfft（S4）—— 两条实输出合并成一条复 IFFT
//   P1: 读两条半谱 -> 各自 Hermitian 展开 -> 复数合并 C = A + iB -> 写 B0
//   P3/P5': 复 IFFT（与普通 C2C 一致）
//   P7: 拆两条实输出（c_re -> 行 2i，c_im -> 行 2i+1）
// ===========================================================================
CATLASS_DEVICE void RunC2rIrfftMerged(
    Arch::Resource<ArchTag>& resource, FftC2R2DKernelContext& ctx,
    GM_ADDR srcGM, GM_ADDR dstGM, GM_ADDR bws, GM_ADDR coeffs)
{
    int64_t fftN = ctx.fftN;
    int32_t halfN = ctx.halfN;
    int32_t mmM = ctx.mmM, mmK = ctx.mmK, mmN = ctx.mmN;

    __gm__ float* gmSrcBase = (__gm__ float*)(srcGM);
    __gm__ float* gmDstBase = (__gm__ float*)(dstGM);
    __gm__ float* gmBwsBase = (__gm__ float*)(bws);
    __gm__ float* gmCoeffsBase = (__gm__ float*)(coeffs);

    int64_t bMatrixFloats = static_cast<int64_t>(mmK) * mmN;
    int64_t inRowStride = ctx.inRowStride;
    int64_t readFloats = static_cast<int64_t>(halfN) * 2;
    int64_t outRowFloats = fftN;
    uint32_t simdComplex = (static_cast<uint32_t>(halfN) / 64) * 64;   // 2048

    uint32_t rowStart = ctx.rowStart;   // dual-row 单位
    uint32_t rowEnd = ctx.rowEnd;
    uint32_t numPairs = (rowEnd - rowStart) / 2;

    // ─── AIC：2 轮 GEMM（group 流水）───
    if ASCEND_IS_AIC {
        FftBlockMmadToUb blockMmad(resource, 0);
        LocalTensor<float> ubGemm0 = resource.ubBuf.template GetBufferByByte<float>(UB_S0 * sizeof(float));
        LocalTensor<float> ubGemm1 = resource.ubBuf.template GetBufferByByte<float>(UB_S1 * sizeof(float));

        uint32_t groups = (numPairs + 1) / 2;
        for (uint32_t i = 0; i < groups; i++) {
            uint32_t pairsInGroup = (i == groups - 1 && (numPairs % 2 != 0)) ? 1 : 2;
            for (uint32_t j = 0; j < pairsInGroup; j++) {
                uint32_t pair = 2 * i + j;
                if (pair >= numPairs) break;
                uint32_t gp = (rowStart / 2) + pair;
                LocalTensor<float>& ubGemm = (j == 0) ? ubGemm0 : ubGemm1;
                AscendC::CrossCoreWaitFlag<FFT_SYNC_MODE, PIPE_MTE2>(FftFlagId(pair, 0));
                int64_t bOff0 = static_cast<int64_t>(ctx.blockIdx) * (4 * bMatrixFloats)
                    + static_cast<int64_t>(gp & 1) * bMatrixFloats;
                RunGemmRadix16(blockMmad, coeffs, ctx.wOffset, bws, bOff0, ubGemm, mmM, mmK, mmN);
                AscendC::CrossCoreSetFlag<FFT_SYNC_MODE, PIPE_FIX>(FftFlagId(pair, 1));
            }
            for (uint32_t j = 0; j < pairsInGroup; j++) {
                uint32_t pair = 2 * i + j;
                if (pair >= numPairs) break;
                uint32_t gp = (rowStart / 2) + pair;
                LocalTensor<float>& ubGemm = (j == 0) ? ubGemm0 : ubGemm1;
                AscendC::CrossCoreWaitFlag<FFT_SYNC_MODE, PIPE_MTE2>(FftFlagId(pair, 2));
                int64_t bOff1 = static_cast<int64_t>(ctx.blockIdx) * (4 * bMatrixFloats)
                    + static_cast<int64_t>(gp & 1) * bMatrixFloats + 2 * bMatrixFloats;
                RunGemmRadix16(blockMmad, coeffs, ctx.wOffset, bws, bOff1, ubGemm, mmM, mmK, mmN);
                AscendC::CrossCoreSetFlag<FFT_SYNC_MODE, PIPE_FIX>(FftFlagId(pair, 3));
            }
        }
    }

    // ─── AIV ───
    if ASCEND_IS_AIV {
        uint32_t subIdx = AscendC::GetSubBlockIdx();
        int64_t colShift = static_cast<int64_t>(subIdx) * R16_SUB;

        LocalTensor<float> aRe = resource.ubBuf.template GetBufferByByte<float>(UB_SEP0 * sizeof(float));
        LocalTensor<float> aIm = resource.ubBuf.template GetBufferByByte<float>((UB_SEP0 + R16_ELEM) * sizeof(float));
        LocalTensor<float> bRe = resource.ubBuf.template GetBufferByByte<float>((UB_SEP0 + 2 * R16_ELEM) * sizeof(float));
        LocalTensor<float> bIm = resource.ubBuf.template GetBufferByByte<float>((UB_SEP0 + 3 * R16_ELEM) * sizeof(float));
        LocalTensor<float> s0 = resource.ubBuf.template GetBufferByByte<float>(UB_S0 * sizeof(float));
        LocalTensor<float> s1 = resource.ubBuf.template GetBufferByByte<float>(UB_S1 * sizeof(float));
        LocalTensor<float> ubT2Exp = resource.ubBuf.template GetBufferByByte<float>(UB_T2EXP * sizeof(float));
        LocalTensor<float> ubR2TW = resource.ubBuf.template GetBufferByByte<float>(UB_RADIX2TW * sizeof(float));
        LocalTensor<float> ubPhaseA = resource.ubBuf.template GetBufferByByte<float>(UB_PHASEA * sizeof(float));
        LocalTensor<float> ubT12Exp = resource.ubBuf.template GetBufferByByte<float>(UB_T12_EXP * sizeof(float));
        LocalTensor<float> ubP3Lane = resource.ubBuf.template GetBufferByByte<float>(UB_P3_LANE * sizeof(float));

        // 常驻表加载
        {
            GlobalTensor<float> gmTExp;
            gmTExp.SetGlobalBuffer(gmCoeffsBase + ctx.tExpandedOffset);
            DataCopy(ubT2Exp, gmTExp, 2 * R16_ELEM);
            GlobalTensor<float> gmR2;
            gmR2.SetGlobalBuffer(gmCoeffsBase + ctx.tRadix2Offset);
            DataCopy(ubR2TW, gmR2, RADIX2_TW_FLOATS_OPS * 2);
            GlobalTensor<float> gmPA;
            gmPA.SetGlobalBuffer(gmCoeffsBase + ctx.tPhaseAOffset);
            DataCopy(ubPhaseA, gmPA, 4096);
            GlobalTensor<float> gmT12E;
            gmT12E.SetGlobalBuffer(gmCoeffsBase + ctx.t12ExpandedOffset);
            DataCopy(ubT12Exp, gmT12E, P3_T12_EXP_FLOATS_OPS);
            GlobalTensor<float> gmP3L;
            gmP3L.SetGlobalBuffer(gmCoeffsBase + ctx.tP3LaneOffset);
            DataCopy(ubP3Lane, gmP3L, P3_LANE_COUNT_OPS);
            SetFlag<HardEvent::MTE2_V>(SYNC_ID0);
            WaitFlag<HardEvent::MTE2_V>(SYNC_ID0);
        }

        __ubuf__ float* t2Re = (__ubuf__ float*)ubT2Exp.GetPhyAddr();
        __ubuf__ float* t2Im = t2Re + R16_ELEM;
        __ubuf__ float* r2Tw = (__ubuf__ float*)ubR2TW.GetPhyAddr();
        __ubuf__ uint32_t* phaseA = (__ubuf__ uint32_t*)ubPhaseA.GetPhyAddr();
        __ubuf__ float* t12ExpRe = (__ubuf__ float*)ubT12Exp.GetPhyAddr();
        __ubuf__ float* t12ExpIm = t12ExpRe + R16_ELEM;
        __ubuf__ uint32_t* p3Lane = (__ubuf__ uint32_t*)ubP3Lane.GetPhyAddr();

        __ubuf__ float* aReAddr = (__ubuf__ float*)aRe.GetPhyAddr();
        __ubuf__ float* aImAddr = (__ubuf__ float*)aIm.GetPhyAddr();
        __ubuf__ float* bReAddr = (__ubuf__ float*)bRe.GetPhyAddr();
        __ubuf__ float* bImAddr = (__ubuf__ float*)bIm.GetPhyAddr();
        __ubuf__ float* s0Addr = (__ubuf__ float*)s0.GetPhyAddr();
        __ubuf__ float* s1Addr = (__ubuf__ float*)s1.GetPhyAddr();

        // SEP / staging ping-pong（pair&1）
        LocalTensor<float> sepReArr[2] = {aRe, bRe};
        LocalTensor<float> sepImArr[2] = {aIm, bIm};
        LocalTensor<float> sArr[2] = {s0, s1};
        __ubuf__ float* sepReAddrArr[2] = {aReAddr, bReAddr};
        __ubuf__ float* sepImAddrArr[2] = {aImAddr, bImAddr};
        __ubuf__ float* sAddrArr[2] = {s0Addr, s1Addr};

        SetFlag<HardEvent::MTE3_MTE2>(SYNC_ID0);
        SetFlag<HardEvent::MTE3_MTE2>(SYNC_ID1);

        DataCopyParams cp;
        cp.blockCount = static_cast<uint16_t>(ctx.N1);
        cp.blockLen = static_cast<uint16_t>(R16_SUB * sizeof(float) / 32);
        cp.srcGap = 0;
        cp.dstGap = static_cast<uint16_t>(R16_SUB * sizeof(float) / 32);

        uint32_t groups = (numPairs + 1) / 2;
        for (uint32_t i = 0; i < groups; i++) {
            uint32_t pairsInGroup = (i == groups - 1 && (numPairs % 2 != 0)) ? 1 : 2;

            // P1：读 m0 → Hermitian → sep；读 m1（复用 staging）→ 融合 Hermitian+合并 → sep；写 B0
            for (uint32_t j = 0; j < pairsInGroup; j++) {
                uint32_t pair = 2 * i + j;
                if (pair >= numPairs) break;
                uint32_t gp = (rowStart / 2) + pair;
                uint32_t myDual = gp * 2 + subIdx;
                uint32_t m0 = myDual * 2;
                uint32_t m1 = myDual * 2 + 1;
                uint32_t b = pair & 1;
                event_t syncId = b ? SYNC_ID1 : SYNC_ID0;
                LocalTensor<float>& sepRe = sepReArr[b];
                LocalTensor<float>& sepIm = sepImArr[b];
                LocalTensor<float>& s = sArr[b];
                __ubuf__ float* sepReA = sepReAddrArr[b];
                __ubuf__ float* sepImA = sepImAddrArr[b];
                __ubuf__ float* sA = sAddrArr[b];

                AscendC::WaitFlag<HardEvent::MTE3_MTE2>(syncId);

                {
                    GlobalTensor<float> gmI0;
                    gmI0.SetGlobalBuffer(gmSrcBase + static_cast<int64_t>(m0) * inRowStride);
                    DataCopyPad<float, AscendC::PaddingMode::Compact>(
                        s, gmI0,
                        DataCopyExtParams{1, static_cast<uint32_t>(readFloats * sizeof(float)), 0, 0, 0},
                        DataCopyPadExtParams<float>{false, 0, 0, 0});
                    SetFlag<HardEvent::MTE2_V>(syncId);
                    WaitFlag<HardEvent::MTE2_V>(syncId);
                    AscendC::PipeBarrier<PIPE_V>();
                    asc_vf_call<simd_deinterleave_row>(sepReA, sepImA, sA,
                        static_cast<uint16_t>(simdComplex / 64));
                    AscendC::DataSyncBarrier<AscendC::MemDsbT::UB>();
                    asc_vf_call<simt_hermitian_deint>(dim3(2048), sepReA, sepImA, sA,
                        static_cast<uint32_t>(fftN), static_cast<uint32_t>(halfN), simdComplex);
                    AscendC::DataSyncBarrier<AscendC::MemDsbT::UB>();
                }

                {
                    SetFlag<HardEvent::V_MTE2>(syncId);
                    WaitFlag<HardEvent::V_MTE2>(syncId);
                    GlobalTensor<float> gmI1;
                    gmI1.SetGlobalBuffer(gmSrcBase + static_cast<int64_t>(m1) * inRowStride);
                    DataCopyPad<float, AscendC::PaddingMode::Compact>(
                        s, gmI1,
                        DataCopyExtParams{1, static_cast<uint32_t>(readFloats * sizeof(float)), 0, 0, 0},
                        DataCopyPadExtParams<float>{false, 0, 0, 0});
                    SetFlag<HardEvent::MTE2_V>(syncId);
                    WaitFlag<HardEvent::MTE2_V>(syncId);
                    AscendC::PipeBarrier<PIPE_V>();
                    asc_vf_call<simd_deint_combine_conj>(sepReA, sepImA, sA,
                        static_cast<uint16_t>(simdComplex / 64));
                    AscendC::DataSyncBarrier<AscendC::MemDsbT::UB>();
                    asc_vf_call<simt_hermitian_deint_combine>(dim3(2048), sepReA, sepImA, sA,
                        static_cast<uint32_t>(fftN), static_cast<uint32_t>(halfN), simdComplex);
                    AscendC::DataSyncBarrier<AscendC::MemDsbT::UB>();
                }

                AscendC::PipeBarrier<PIPE_V>();
                SetFlag<HardEvent::V_MTE3>(syncId);
                WaitFlag<HardEvent::V_MTE3>(syncId);

                {
                    int64_t baseOff = static_cast<int64_t>(ctx.blockIdx) * (4 * bMatrixFloats)
                        + static_cast<int64_t>(gp & 1) * bMatrixFloats;
                    GlobalTensor<float> gmR;
                    gmR.SetGlobalBuffer(gmBwsBase + baseOff + colShift);
                    DataCopy(gmR, sepRe, cp);
                    GlobalTensor<float> gmI;
                    gmI.SetGlobalBuffer(gmBwsBase + baseOff + static_cast<int64_t>(ctx.N1) * mmN + colShift);
                    DataCopy(gmI, sepIm, cp);
                    AscendC::CrossCoreSetFlag<FFT_SYNC_MODE, PIPE_MTE3>(FftFlagId(pair, 0));
                }
            }

            // P3：wait C0 → twiddle → 写 B1
            for (uint32_t j = 0; j < pairsInGroup; j++) {
                uint32_t pair = 2 * i + j;
                if (pair >= numPairs) break;
                uint32_t gp = (rowStart / 2) + pair;
                uint32_t b = pair & 1;
                event_t syncId = b ? SYNC_ID1 : SYNC_ID0;
                LocalTensor<float>& sepRe = sepReArr[b];
                LocalTensor<float>& sepIm = sepImArr[b];
                __ubuf__ float* sepReA = sepReAddrArr[b];
                __ubuf__ float* sepImA = sepImAddrArr[b];
                __ubuf__ float* sA = sAddrArr[b];

                AscendC::CrossCoreWaitFlag<FFT_SYNC_MODE, PIPE_V>(FftFlagId(pair, 1));
                AscendC::PipeBarrier<PIPE_V>();
                simd_twiddle_rotate_phase3(sepReA, sepImA, sA, sA + R16_ELEM,
                    t12ExpRe, t12ExpIm, p3Lane);
                AscendC::PipeBarrier<PIPE_V>();
                SetFlag<HardEvent::V_MTE3>(syncId);
                WaitFlag<HardEvent::V_MTE3>(syncId);
                {
                    int64_t baseOff = static_cast<int64_t>(ctx.blockIdx) * (4 * bMatrixFloats)
                        + static_cast<int64_t>(gp & 1) * bMatrixFloats + 2 * bMatrixFloats;
                    GlobalTensor<float> gmR;
                    gmR.SetGlobalBuffer(gmBwsBase + baseOff + colShift);
                    DataCopy(gmR, sepRe, cp);
                    GlobalTensor<float> gmI;
                    gmI.SetGlobalBuffer(gmBwsBase + baseOff + static_cast<int64_t>(ctx.N1) * mmN + colShift);
                    DataCopy(gmI, sepIm, cp);
                    AscendC::CrossCoreSetFlag<FFT_SYNC_MODE, PIPE_MTE3>(FftFlagId(pair, 2));
                }
            }

            // P5'：wait C1 → radix2 → sep
            for (uint32_t j = 0; j < pairsInGroup; j++) {
                uint32_t pair = 2 * i + j;
                if (pair >= numPairs) break;
                uint32_t b = pair & 1;
                LocalTensor<float>& sepRe = sepReArr[b];
                LocalTensor<float>& sepIm = sepImArr[b];
                LocalTensor<float>& s = sArr[b];

                AscendC::CrossCoreWaitFlag<FFT_SYNC_MODE, PIPE_V>(FftFlagId(pair, 3));
                Radix16Phase5(s, sepRe, sepIm, t2Re, t2Im, r2Tw, phaseA);
                AscendC::PipeBarrier<PIPE_V>();
            }

            // P7：拆两条实输出（c_re → m0，c_im → m1）
            for (uint32_t j = 0; j < pairsInGroup; j++) {
                uint32_t pair = 2 * i + j;
                if (pair >= numPairs) break;
                uint32_t gp = (rowStart / 2) + pair;
                uint32_t myDual = gp * 2 + subIdx;
                uint32_t m0 = myDual * 2;
                uint32_t m1 = myDual * 2 + 1;
                uint32_t b = pair & 1;
                event_t syncId = b ? SYNC_ID1 : SYNC_ID0;
                LocalTensor<float>& sepRe = sepReArr[b];
                LocalTensor<float>& sepIm = sepImArr[b];

                SetFlag<HardEvent::V_MTE3>(syncId);
                WaitFlag<HardEvent::V_MTE3>(syncId);
                {
                    GlobalTensor<float> gmO0;
                    gmO0.SetGlobalBuffer(gmDstBase + static_cast<int64_t>(m0) * outRowFloats);
                    DataCopy(gmO0, sepRe, static_cast<uint32_t>(outRowFloats));
                    GlobalTensor<float> gmO1;
                    gmO1.SetGlobalBuffer(gmDstBase + static_cast<int64_t>(m1) * outRowFloats);
                    DataCopy(gmO1, sepIm, static_cast<uint32_t>(outRowFloats));
                    SetFlag<HardEvent::MTE3_MTE2>(syncId);
                }
            }
        }

        WaitFlag<HardEvent::MTE3_MTE2>(SYNC_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(SYNC_ID1);
    }
}

// ===========================================================================
// 主 Kernel：逆序瘦列 4-stage
//   S1 转置 X[fftN,padHalf] -> Xt[padHalf,fftN]
//   S2 C2C 逆变换（axis0，padHalf 条） -> Y[padHalf,fftN]
//   S3 转置 Y -> Yt[fftN,padHalf]
//   S4 irfft（axis1，Hermitian 展开 + IDFT + 取实） -> output[fftN,fftN] real
// ===========================================================================
CATLASS_GLOBAL __mix__(1, 2) void FftC2R2DKernel(
    GM_ADDR input, GM_ADDR output, GM_ADDR workspace, GM_ADDR bws,
    GM_ADDR coeffs, GM_ADDR tiling)
{
    __gm__ FftC2R2DTilingData* td = (__gm__ FftC2R2DTilingData*)tiling;
    FftC2R2DKernelContext ctx;
    InitC2R2DContext(td, ctx);

    int32_t cubeCoreNum = td->multiCoreParams.coreNum;
    if (ctx.blockIdx >= static_cast<uint32_t>(cubeCoreNum)) {
        return;
    }

    Arch::Resource<ArchTag> resource;
    int32_t fftN = static_cast<int32_t>(ctx.fftN);
    int32_t padHalf = ctx.padHalf;

    // 三个瘦副本：wsA=Xt[padHalf,fftN]  wsB=Y[padHalf,fftN]  wsC=Yt[fftN,padHalf]
    int64_t wsBufBytes = static_cast<int64_t>(fftN) * padHalf * 2 * sizeof(float);
    GM_ADDR wsA = workspace;
    GM_ADDR wsB = workspace + wsBufBytes;
    GM_ADDR wsC = workspace + 2 * wsBufBytes;

    // S1：转置 input[fftN, padHalf] -> wsA Xt[padHalf, fftN]
    RunTransposeRectStage(resource, input, wsA,
        fftN, padHalf, padHalf * 2, fftN * 2, cubeCoreNum);
    AscendC::SyncAll<false>();

    // S2：C2C 逆变换（axis0）wsA -> wsB
    ctx.rowStart = td->multiCoreParams.skinnyRowStartIdx[ctx.blockIdx];
    ctx.rowEnd = td->multiCoreParams.skinnyRowStartIdx[ctx.blockIdx + 1];
    ctx.inRowStride = static_cast<int64_t>(fftN) * 2;
    RunFftPass<false, false>(resource, ctx, wsA, wsB, bws, coeffs);
    AscendC::SyncAll<false>();

    // S3：转置 wsB Y[padHalf, fftN] -> wsC Yt[fftN, padHalf]
    RunTransposeRectStage(resource, wsB, wsC,
        padHalf, fftN, fftN * 2, padHalf * 2, cubeCoreNum);
    AscendC::SyncAll<false>();

    // S4：irfft（axis1）共轭合并，wsC -> output（real）
    ctx.rowStart = td->multiCoreParams.mergedRowStartIdx[ctx.blockIdx];
    ctx.rowEnd = td->multiCoreParams.mergedRowStartIdx[ctx.blockIdx + 1];
    ctx.inRowStride = static_cast<int64_t>(padHalf) * 2;
    RunC2rIrfftMerged(resource, ctx, wsC, output, bws, coeffs);
}

void FftC2R2DKernelLaunch(
    GM_ADDR input, GM_ADDR output, GM_ADDR workspace, GM_ADDR bws,
    GM_ADDR coeffs, GM_ADDR tiling,
    uint32_t blockDim, aclrtStream stream)
{
    constexpr uint32_t workBuff = 216 * 1024;
    FftC2R2DKernel<<<blockDim, workBuff, stream>>>(
        input, output, workspace, bws, coeffs, tiling);
}

} // namespace Fft
