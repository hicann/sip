#ifndef FFT_KERNEL_COMMON_H
#define FFT_KERNEL_COMMON_H

#include "fft_kernel_utils.h"
#include "fft_simt_ops.h"
#include "fft_radix2_ops.h"

namespace Fft {

// ═══════════════════════════════════════════════════════════════════════════
// R2C 1D 与 C2C 1D 共用的 16×16×16 引擎中间阶段（P3 twiddle/轴旋转 + P5' radix2）
// 以及 UB 布局 / 跨核 flag 常量。
// ═══════════════════════════════════════════════════════════════════════════

constexpr int64_t UB_OFFSET_SEP0      = 0;
constexpr int64_t UB_OFFSET_SEP1      = 8192;
constexpr int64_t UB_OFFSET_T2_EXP    = 16384;
constexpr int64_t UB_OFFSET_RADIX2_TW = 24576;
constexpr int64_t UB_OFFSET_T         = 26752;
constexpr int64_t UB_OFFSET_T12_EXP   = 28288;   // P3 SIMD twiddle 展开表（8192 floats = 32KB）
constexpr int64_t UB_OFFSET_P3_LANE   = 36480;   // P3 Gather lane 表（64 u32）
constexpr int64_t UB_OFFSET_GEMM      = 36864;
constexpr int64_t UB_OFFSET_GEMM_1    = 45056;
constexpr int64_t UB_OFFSET_PHASE_A   = 53248;

constexpr int64_t T_SINGLE_FLOATS     = 512;
constexpr int64_t T_HALF_FLOATS       = 256;

constexpr bool SYNC_ONLY = false;
constexpr bool AIC_SKELETON = false;
constexpr bool P1357_SKELETON = false;

constexpr uint32_t SLOT_B0 = 0;
constexpr uint32_t SLOT_G0 = 1;
constexpr uint32_t SLOT_B1 = 2;
constexpr uint32_t SLOT_G1 = 3;

CATLASS_DEVICE uint16_t FftFlagId(uint32_t pair, uint32_t slot)
{
    return static_cast<uint16_t>((pair & 1) * 4 + slot);
}

// ─── Phase3: Wait GEMM0 → twiddle T12 + 轴旋转 → 写 B1（R2C/C2C 共用）───
CATLASS_DEVICE void Phase3Aiv(
    FftKernelContext& ctx, uint32_t subIdx, uint32_t globalPairIdx,
    event_t syncId,
    LocalTensor<float>& ubGemmBuf, LocalTensor<float>& ubOutR, LocalTensor<float>& ubOutI,
    __ubuf__ float* t12ExpRe, __ubuf__ float* t12ExpIm, __ubuf__ uint32_t* p3Lane,
    __gm__ float* gmWsBase)
{
    int64_t elemCount = static_cast<int64_t>(ctx.N1) * ctx.N2 * ctx.N2;
    int64_t subCount = static_cast<int64_t>(ctx.N2) * ctx.N2;
    int64_t colOffset = static_cast<int64_t>(subIdx) * subCount;

    __ubuf__ float* sAddr = (__ubuf__ float*)ubGemmBuf.GetPhyAddr();
    __ubuf__ float* outRAddr = (__ubuf__ float*)ubOutR.GetPhyAddr();
    __ubuf__ float* outIAddr = (__ubuf__ float*)ubOutI.GetPhyAddr();

    AscendC::PipeBarrier<PIPE_V>();
    simd_twiddle_rotate_phase3(outRAddr, outIAddr, sAddr, sAddr + elemCount,
                               t12ExpRe, t12ExpIm, p3Lane);
    AscendC::PipeBarrier<PIPE_V>();

    SetFlag<HardEvent::V_MTE3>(syncId);
    WaitFlag<HardEvent::V_MTE3>(syncId);

    if constexpr (!SYNC_ONLY) {
        int64_t bMatrixFloats = static_cast<int64_t>(ctx.mmK[0]) * ctx.mmN[0];
        int64_t baseOff = static_cast<int64_t>(ctx.blockIdx) * (6 * bMatrixFloats)
                              + static_cast<int64_t>(globalPairIdx & 1) * bMatrixFloats
                              + ctx.wsOffset[2];

        DataCopyParams copyParams;
        copyParams.blockCount = static_cast<uint16_t>(ctx.N1);
        copyParams.blockLen = static_cast<uint16_t>(subCount * sizeof(float) / 32);
        copyParams.srcGap = 0;
        copyParams.dstGap = static_cast<uint16_t>(subCount * sizeof(float) / 32);

        GlobalTensor<float> gmReal;
        gmReal.SetGlobalBuffer(gmWsBase + baseOff + colOffset);
        DataCopy(gmReal, ubOutR, copyParams);

        GlobalTensor<float> gmImag;
        gmImag.SetGlobalBuffer(gmWsBase + baseOff
            + static_cast<int64_t>(ctx.N1) * ctx.mmN[0] + colOffset);
        DataCopy(gmImag, ubOutI, copyParams);
    }
}

// ─── Phase5' Radix2: Wait GEMM1 → T2 复乘 + bit-reverse + 4 层 radix-2 → Z in Sep ──
CATLASS_DEVICE void Phase5Radix2Aiv(
    FftKernelContext& ctx, uint32_t subIdx, uint32_t globalPairIdx,
    event_t syncId,
    LocalTensor<float>& ubGemmBuf, LocalTensor<float>& ubSepRe, LocalTensor<float>& ubSepIm,
    __ubuf__ float* t2ExpRe, __ubuf__ float* t2ExpIm,
    __ubuf__ float* radix2TwTable, __ubuf__ uint32_t* phaseAOffsets)
{
    Phase5Radix2(ubGemmBuf, ubSepRe, ubSepIm,
                 t2ExpRe, t2ExpIm, radix2TwTable, phaseAOffsets);
}

} // namespace Fft

#endif // FFT_KERNEL_COMMON_H