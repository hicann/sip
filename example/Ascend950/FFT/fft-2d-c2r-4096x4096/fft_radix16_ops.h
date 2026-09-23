// ===========================================================================
// fft_radix16_ops.h — radix-16（16×16×16）SIMT/SIMD 计算阶段
// 移植自 ex99 fft_simt_ops.h 与 fft_radix2_ops.h 的 16×16×16 路径。
//   P3: twiddle T12 + 轴旋转（SIMT）
//   P5': T2 复乘 + bit-reverse + 16 点 radix-2 butterfly（SIMD Gather + SIMD butterfly）
// ===========================================================================

#ifndef FFT_RADIX16_OPS_H
#define FFT_RADIX16_OPS_H

#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "catlass/arch/arch.hpp"
#include "catlass/arch/resource.hpp"

namespace Fft {

using namespace AscendC;

constexpr uint32_t R16_THREADS = 2048;
constexpr uint32_t R16_R = 16;
constexpr uint32_t R16_SUB = 256;       // R*R
constexpr uint32_t R16_TOTAL = 4096;    // R*SUB
constexpr uint32_t R16_VEC = 64;
constexpr uint32_t R16_COLBLOCKS = R16_SUB / R16_VEC;  // 4

constexpr int32_t RADIX2_TW_FLOATS_OPS = 15 * 64;  // 960
constexpr int32_t P3_T12_EXP_FLOATS_OPS = 8192;    // 4096 re + 4096 im
constexpr int32_t P3_LANE_COUNT_OPS = 64;

// ─── P3: C0[k3,(n2,n3)] × T12(k3,n2) → B1[n2,(k3,n3)] ────────────────────
// srcIdx = k3*256 + n2*16 + n3；dstIdx = i = n2*256 + k3*16 + n3
__simt_vf__ __launch_bounds__(R16_THREADS) inline void simt_twiddle_rotate_p3(
    __ubuf__ float* dstRe, __ubuf__ float* dstIm,
    __ubuf__ float* srcRe, __ubuf__ float* srcIm,
    __ubuf__ float* t12Re, __ubuf__ float* t12Im)
{
    constexpr uint32_t R = 16;
    constexpr uint32_t SUB = R * R;
    constexpr uint32_t TOTAL = R * SUB;
    uint32_t tid = threadIdx.x;

    for (uint32_t i = tid; i < TOTAL; i += R16_THREADS) {
        uint32_t n2 = i / SUB;
        uint32_t c = i - n2 * SUB;
        uint32_t n3 = c & (R - 1);
        uint32_t k3 = c >> 4;
        uint32_t srcIdx = k3 * SUB + n2 * R + n3;
        uint32_t tIdx = k3 * R + n2;

        float ar = srcRe[srcIdx];
        float ai = srcIm[srcIdx];
        float tr = t12Re[tIdx];
        float ti = t12Im[tIdx];
        dstRe[i] = ar * tr - ai * ti;
        dstIm[i] = ar * ti + ai * tr;
    }
}

// ─── 共轭合并（方案 B）：C_re = A_re - B_im, C_im = A_im + B_re ──────────
// 两条实信号（各自 Hermitian 展开后）合并成一条复信号做一次 IFFT。
__simd_vf__ inline void simd_combine_conj(
    __ubuf__ float* dstRe, __ubuf__ float* dstIm,
    __ubuf__ float* srcARe, __ubuf__ float* srcAIm,
    __ubuf__ float* srcBRe, __ubuf__ float* srcBIm,
    uint32_t total)
{
    Reg::RegTensor<float> aRe, aIm, bRe, bIm, outRe, outIm;
    Reg::MaskReg mask = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
    constexpr uint32_t VEC = 64;
    for (uint32_t off = 0; off < total; off += VEC) {
        Reg::LoadAlign(aRe, srcARe + off);
        Reg::LoadAlign(aIm, srcAIm + off);
        Reg::LoadAlign(bRe, srcBRe + off);
        Reg::LoadAlign(bIm, srcBIm + off);
        Reg::Sub(outRe, aRe, bIm, mask);
        Reg::Add(outIm, aIm, bRe, mask);
        Reg::StoreAlign(dstRe + off, outRe, mask);
        Reg::StoreAlign(dstIm + off, outIm, mask);
    }
}

// ─── 融合 deinterleave + 共轭合并（方案 B 的 UB 优化）─────────────────────
// 就地：dst(Re/Im) 为 A 的 re/im，读 m1 半谱 src，deinterleave 出 B，
// 立即合并 C_re=A_re-B_im、C_im=A_im+B_re，写回 dst。省掉独立 B 缓冲，
// 使每 pair 只需 8192 floats 的 SEP，可做 2-deep 流水。
__simd_vf__ inline void simd_deint_combine_conj(
    __ubuf__ float* dstRe, __ubuf__ float* dstIm,
    __ubuf__ float* src, uint16_t repeatTimes)
{
    constexpr uint16_t oneRepeatSize = AscendC::GetVecLen() / sizeof(float);
    Reg::RegTensor<float> srcReg0, srcReg1, bRe, bIm, aRe, aIm, outRe, outIm;
    Reg::MaskReg mask = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
    for (uint16_t i = 0; i < repeatTimes; i++) {
        Reg::LoadAlign(srcReg0, src + i * oneRepeatSize * 2);
        Reg::LoadAlign(srcReg1, src + i * oneRepeatSize * 2 + oneRepeatSize);
        Reg::DeInterleave<float>(bRe, bIm, srcReg0, srcReg1);
        Reg::LoadAlign(aRe, dstRe + i * oneRepeatSize);
        Reg::LoadAlign(aIm, dstIm + i * oneRepeatSize);
        Reg::Sub(outRe, aRe, bIm, mask);
        Reg::Add(outIm, aIm, bRe, mask);
        Reg::StoreAlign(dstRe + i * oneRepeatSize, outRe, mask);
        Reg::StoreAlign(dstIm + i * oneRepeatSize, outIm, mask);
    }
}

// 融合 Hermitian 尾 + 共轭合并（就地）：处理 [start, fftN)。
__simt_vf__ __launch_bounds__(R16_THREADS) inline void simt_hermitian_deint_combine(
    __ubuf__ float* dstRe, __ubuf__ float* dstIm,
    const __ubuf__ float* src,
    uint32_t fftN, uint32_t halfN, uint32_t start)
{
    uint32_t tid = threadIdx.x;
    for (uint32_t v = start + tid; v < fftN; v += R16_THREADS) {
        if (v < halfN) {
            dstRe[v] = dstRe[v] - src[v * 2 + 1];
            dstIm[v] = dstIm[v] + src[v * 2];
        } else {
            uint32_t c = fftN - v;
            dstRe[v] = dstRe[v] + src[c * 2 + 1];
            dstIm[v] = dstIm[v] + src[c * 2];
        }
    }
}

// ─── P3 SIMD 版：twiddle T12 + 轴旋转（Reg::Gather + 展开表）──────────────
// 输出 dstRe/dstIm 布局：i = n2*256 + k3*16 + n3（与 simt 版一致）。
__simd_vf__ inline void simd_twiddle_rotate_phase3(
    __ubuf__ float* dstRe, __ubuf__ float* dstIm,
    __ubuf__ float* srcRe, __ubuf__ float* srcIm,
    __ubuf__ float* t12ExpRe, __ubuf__ float* t12ExpIm,
    __ubuf__ uint32_t* p3Lane)
{
    constexpr uint32_t R = 16;
    constexpr uint32_t SUB = R * R;      // 256
    constexpr uint32_t VEC = 64;

    Reg::RegTensor<float> srcReReg, srcImReg, t12ReReg, t12ImReg;
    Reg::RegTensor<float> outReReg, outImReg, negImReg;
    Reg::RegTensor<uint32_t> idxReg;
    Reg::MaskReg mask = Reg::CreateMask<float, Reg::MaskPattern::ALL>();

    for (uint32_t n2 = 0; n2 < R; n2++) {
        for (uint32_t k3blk = 0; k3blk < 4; k3blk++) {
            Reg::LoadAlign(idxReg, p3Lane);
            Reg::Adds<uint32_t>(idxReg, idxReg, (k3blk << 10) + (n2 << 4), mask);
            Reg::Gather(srcReReg, srcRe, idxReg, mask);
            Reg::Gather(srcImReg, srcIm, idxReg, mask);

            uint32_t tOff = ((n2 << 2) + k3blk) * VEC;
            Reg::LoadAlign(t12ReReg, t12ExpRe + tOff);
            Reg::LoadAlign(t12ImReg, t12ExpIm + tOff);

            Reg::Mul(outReReg, srcReReg, t12ReReg, mask);
            Reg::Mul(outImReg, srcReReg, t12ImReg, mask);
            Reg::Neg(negImReg, srcImReg, mask);
            Reg::MulAddDst(outReReg, negImReg, t12ImReg, mask);
            Reg::MulAddDst(outImReg, srcImReg, t12ReReg, mask);

            uint32_t dstOff = n2 * SUB + k3blk * VEC;
            Reg::StoreAlign(dstRe + dstOff, outReReg, mask);
            Reg::StoreAlign(dstIm + dstOff, outImReg, mask);
        }
    }
}

// ─── P5' 阶段 B：SIMD 4 层 radix-2 butterfly（16 点 DFT）──────────────────
// 每蝶形一列块：加载 2 行 → 复乘 + 加减 → 写回
__simd_vf__ inline void simd_radix2_butterfly_16(
    __ubuf__ float* xRe, __ubuf__ float* xIm, __ubuf__ float* twTable)
{
    Reg::RegTensor<float> reA, imA, reB, imB;
    Reg::RegTensor<float> twRe, twIm, tRe, tIm;
    Reg::MaskReg mask = Reg::CreateMask<float, Reg::MaskPattern::ALL>();

    uint16_t twOff = 0;
    for (uint16_t stage = 0; stage < 4; stage++) {
        uint16_t half = 1 << stage;
        uint16_t step = 2 * half;
        for (uint16_t base = 0; base < R16_R; base += step) {
            for (uint16_t j = 0; j < half; j++) {
                uint16_t i = base + j;
                uint16_t i2 = i + half;

                Reg::LoadAlign(twRe, twTable + twOff + j * R16_VEC);
                Reg::LoadAlign(twIm, twTable + RADIX2_TW_FLOATS_OPS + twOff + j * R16_VEC);

                for (uint16_t blk = 0; blk < R16_COLBLOCKS; blk++) {
                    uint16_t colOff = blk * R16_VEC;

                    Reg::LoadAlign(reA, xRe + i * R16_SUB + colOff);
                    Reg::LoadAlign(imA, xIm + i * R16_SUB + colOff);
                    Reg::LoadAlign(reB, xRe + i2 * R16_SUB + colOff);
                    Reg::LoadAlign(imB, xIm + i2 * R16_SUB + colOff);

                    Reg::Mul(tRe, reB, twRe, mask);
                    Reg::Mul(tIm, reB, twIm, mask);
                    Reg::Mul(reB, imB, twIm, mask);
                    Reg::Mul(imB, imB, twRe, mask);
                    Reg::Sub(tRe, tRe, reB, mask);
                    Reg::Add(tIm, tIm, imB, mask);

                    Reg::Sub(reB, reA, tRe, mask);
                    Reg::Add(reA, reA, tRe, mask);
                    Reg::Sub(imB, imA, tIm, mask);
                    Reg::Add(imA, imA, tIm, mask);

                    Reg::StoreAlign(xRe + i * R16_SUB + colOff, reA, mask);
                    Reg::StoreAlign(xIm + i * R16_SUB + colOff, imA, mask);
                    Reg::StoreAlign(xRe + i2 * R16_SUB + colOff, reB, mask);
                    Reg::StoreAlign(xIm + i2 * R16_SUB + colOff, imB, mask);
                }
            }
        }
        twOff += half * R16_VEC;
    }
}

// ─── P5' 阶段 A（SIMD Gather 版）：T2 复乘 + 轴旋转 + bit-reverse ─────────
__simd_vf__ inline void simd_twiddle_rotate_brev_p5_gather(
    __ubuf__ float* dstRe, __ubuf__ float* dstIm,
    __ubuf__ float* srcRe, __ubuf__ float* srcIm,
    __ubuf__ float* t2Re, __ubuf__ float* t2Im,
    __ubuf__ uint32_t* offsetTable)
{
    Reg::RegTensor<float> srcReReg, srcImReg, t2ReReg, t2ImReg;
    Reg::RegTensor<float> outReReg, outImReg, tReReg, tImReg;
    Reg::RegTensor<float> tmpReReg, tmpImReg;
    Reg::RegTensor<uint32_t> idxReg;
    Reg::MaskReg mask = Reg::CreateMask<float, Reg::MaskPattern::ALL>();

    for (uint16_t n3 = 0; n3 < 16; n3++) {
        uint16_t br = (n3 & 1u) << 3 | (n3 & 2u) << 1 | (n3 & 4u) >> 1 | (n3 & 8u) >> 3;
        for (uint16_t colBlk = 0; colBlk < 4; colBlk++) {
            Reg::LoadAlign(idxReg, offsetTable + n3 * R16_SUB + colBlk * R16_VEC);
            Reg::Gather(srcReReg, srcRe, idxReg, mask);
            Reg::Gather(srcImReg, srcIm, idxReg, mask);

            Reg::LoadAlign(t2ReReg, t2Re + n3 * R16_SUB + colBlk * R16_VEC);
            Reg::LoadAlign(t2ImReg, t2Im + n3 * R16_SUB + colBlk * R16_VEC);

            Reg::Mul(tReReg, srcReReg, t2ReReg, mask);
            Reg::Mul(tImReg, srcReReg, t2ImReg, mask);
            Reg::Mul(tmpReReg, srcImReg, t2ImReg, mask);
            Reg::Mul(tmpImReg, srcImReg, t2ReReg, mask);
            Reg::Sub(outReReg, tReReg, tmpReReg, mask);
            Reg::Add(outImReg, tImReg, tmpImReg, mask);

            uint16_t dstOff = br * R16_SUB + colBlk * R16_VEC;
            Reg::StoreAlign(dstRe + dstOff, outReReg, mask);
            Reg::StoreAlign(dstIm + dstOff, outImReg, mask);
        }
    }
}

// ─── P5' 组合：阶段 A（Gather）+ 阶段 B（butterfly）───────────────────────
// 输入：ubS 为 GEMM1 结果 C1（16×256 复，[Re;Im] 分开），输出到 ubXRe/ubXIm（bit-reverse 后 butterfly）
__aicore__ inline void Radix16Phase5(
    LocalTensor<float>& ubS, LocalTensor<float>& ubXRe, LocalTensor<float>& ubXIm,
    __ubuf__ float* t2Re, __ubuf__ float* t2Im,
    __ubuf__ float* twTable, __ubuf__ uint32_t* phaseAOffsets)
{
    AscendC::PipeBarrier<PIPE_V>();

    __ubuf__ float* sAddr = (__ubuf__ float*)ubS.GetPhyAddr();
    __ubuf__ float* xReAddr = (__ubuf__ float*)ubXRe.GetPhyAddr();
    __ubuf__ float* xImAddr = (__ubuf__ float*)ubXIm.GetPhyAddr();

    simd_twiddle_rotate_brev_p5_gather(
        xReAddr, xImAddr, sAddr, sAddr + R16_TOTAL, t2Re, t2Im, phaseAOffsets);
    AscendC::DataSyncBarrier<AscendC::MemDsbT::UB>();

    simd_radix2_butterfly_16(xReAddr, xImAddr, twTable);
    AscendC::DataSyncBarrier<AscendC::MemDsbT::UB>();
}

} // namespace Fft

#endif // FFT_RADIX16_OPS_H