#ifndef FFT_RADIX2_OPS_H
#define FFT_RADIX2_OPS_H

#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "fft_tiling_def.h"
#include "catlass/arch/arch.hpp"
#include "catlass/arch/resource.hpp"

constexpr uint32_t R2_THREADS = 2048;
constexpr uint32_t R2_R = 16;
constexpr uint32_t R2_SUB = 256;
constexpr uint32_t R2_TOTAL = 4096;
constexpr uint32_t R2_VEC = 64;
constexpr uint32_t R2_COLBLOCKS = R2_SUB / R2_VEC;  // 4

// 16 点 bit-reverse 查找表
constexpr uint32_t R2_BR_TABLE[16] = {0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15};

// ═══════════════════════════════════════════════════════════════════════════
// P5' 阶段 A（SIMD Gather 版）：T2 复乘 + 轴旋转 + bit-reverse（ex99 原样）
// R2_USE_MULADD=1（方案 A）：复乘用 MulAddDst 融合（vmula 单指令替代
// Mul+Add/Sub 对）。形式：outRe = sRe*t2Re - sIm*t2Im：
//   t = Mul(sIm, t2Im) → t = MulAddDst(t, sRe, ...) 需要 -t2Re? 不——
//   融合顺序：t = Mul(sRe, t2Re)；t = MulAddDst(t, sIm, negT2Im)。
//   UB 无空间放 -t2Im 常量表（T2 表 32KB 已满），用 Neg(sImNeg, sIm) 折中：
//   4 Mul + Sub + Add = 6 算术 → Neg + 2 Mul + 2 MulAddDst = 5（-1/组）。
//
// staggered 轮：offsetTable 从 4096 项完整表（16KB）压缩为 64 项 lane 表
// （256B）。完整索引 = lane 表 f(lane) + 标量 base（colBlk*1024 + n3），
// 每组一条 Reg::Adds 补 base —— 用 64 条/pair 的 V 指令换 16KB UB 空间
// （ubG 3-deep 的空间来源）。等价性：off = k2*256 + k3*16 + n3，
// k2 = colBlk*4 + (lane>>4) ⇒ colBlk*1024 项；k3 = lane&15 ⇒ (lane&15)*16 项。
// ═══════════════════════════════════════════════════════════════════════════
constexpr bool R2_USE_MULADD = true;

__simd_vf__ inline void simd_twiddle_rotate_brev_phase5_gather(
    __ubuf__ float* dstRe, __ubuf__ float* dstIm,
    __ubuf__ float* srcRe, __ubuf__ float* srcIm,
    __ubuf__ float* t2Re, __ubuf__ float* t2Im,
    __ubuf__ uint32_t* laneTable)
{
    AscendC::Reg::RegTensor<float> srcReReg, srcImReg, t2ReReg, t2ImReg;
    AscendC::Reg::RegTensor<float> outReReg, outImReg, tReReg, tImReg;
    AscendC::Reg::RegTensor<float> tmpReReg, tmpImReg, negImReg;
    AscendC::Reg::RegTensor<uint32_t> idxReg;
    AscendC::Reg::MaskReg mask = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();

    for (uint16_t n3 = 0; n3 < 16; n3++) {
        uint16_t br = (n3 & 1u) << 3 | (n3 & 2u) << 1 | (n3 & 4u) >> 1 | (n3 & 8u) >> 3;
        for (uint16_t colBlk = 0; colBlk < 4; colBlk++) {
            // 索引向量 = lane 表 + 标量 base（colBlk*1024 + n3）
            AscendC::Reg::LoadAlign(idxReg, laneTable);
            AscendC::Reg::Adds<uint32_t>(idxReg, idxReg,
                static_cast<uint32_t>(colBlk) * 1024u + n3, mask);
            AscendC::Reg::Gather(srcReReg, srcRe, idxReg, mask);
            AscendC::Reg::Gather(srcImReg, srcIm, idxReg, mask);

            AscendC::Reg::LoadAlign(t2ReReg, t2Re + n3 * R2_SUB + colBlk * R2_VEC);
            AscendC::Reg::LoadAlign(t2ImReg, t2Im + n3 * R2_SUB + colBlk * R2_VEC);

#if R2_USE_MULADD
            // outRe = sRe*t2Re - sIm*t2Im; outIm = sRe*t2Im + sIm*t2Re
            AscendC::Reg::Mul(outReReg, srcReReg, t2ReReg, mask);      // sRe*t2Re
            AscendC::Reg::Mul(outImReg, srcReReg, t2ImReg, mask);      // sRe*t2Im
            AscendC::Reg::Neg(negImReg, srcImReg, mask);               // -sIm
            AscendC::Reg::MulAddDst(outReReg, negImReg, t2ImReg, mask); // + (-sIm)*t2Im
            AscendC::Reg::MulAddDst(outImReg, srcImReg, t2ReReg, mask); // + sIm*t2Re
#else
            AscendC::Reg::Mul(tReReg, srcReReg, t2ReReg, mask);
            AscendC::Reg::Mul(tImReg, srcReReg, t2ImReg, mask);
            AscendC::Reg::Mul(tmpReReg, srcImReg, t2ImReg, mask);
            AscendC::Reg::Mul(tmpImReg, srcImReg, t2ReReg, mask);
            AscendC::Reg::Sub(outReReg, tReReg, tmpReReg, mask);
            AscendC::Reg::Add(outImReg, tImReg, tmpImReg, mask);
#endif

            uint16_t dstOff = br * R2_SUB + colBlk * R2_VEC;
            AscendC::Reg::StoreAlign(dstRe + dstOff, outReReg, mask);
            AscendC::Reg::StoreAlign(dstIm + dstOff, outImReg, mask);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// P5' 阶段 B：SIMD 4 层 radix-2 butterfly（ex99 原样；R2_USE_MULADD=1 时
// twiddle 复乘融合同 gather 段：Neg + 2 Mul + 2 MulAddDst 替代 4 Mul+Add+Sub）
// ═══════════════════════════════════════════════════════════════════════════
__simd_vf__ inline void simd_radix2_butterfly_4stage(
    __ubuf__ float* xRe, __ubuf__ float* xIm,
    __ubuf__ float* twTable)
{
    AscendC::Reg::RegTensor<float> reA, imA, reB, imB;
    AscendC::Reg::RegTensor<float> twRe, twIm, tRe, tIm;
    AscendC::Reg::RegTensor<float> negImB;
    AscendC::Reg::MaskReg mask = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();

    uint16_t twOff = 0;
    for (uint16_t stage = 0; stage < 4; stage++) {
        uint16_t half = 1 << stage;
        uint16_t step = 2 * half;

        for (uint16_t base = 0; base < R2_R; base += step) {
            for (uint16_t j = 0; j < half; j++) {
                uint16_t i = base + j;
                uint16_t i2 = i + half;

                AscendC::Reg::LoadAlign(twRe, twTable + twOff + j * R2_VEC);
                AscendC::Reg::LoadAlign(twIm, twTable + RADIX2_TW_FLOATS + twOff + j * R2_VEC);

                for (uint16_t blk = 0; blk < R2_COLBLOCKS; blk++) {
                    uint16_t colOff = blk * R2_VEC;

                    AscendC::Reg::LoadAlign(reA, xRe + i * R2_SUB + colOff);
                    AscendC::Reg::LoadAlign(imA, xIm + i * R2_SUB + colOff);
                    AscendC::Reg::LoadAlign(reB, xRe + i2 * R2_SUB + colOff);
                    AscendC::Reg::LoadAlign(imB, xIm + i2 * R2_SUB + colOff);

#if R2_USE_MULADD
                    // t = reB*twRe - imB*twIm; tI = reB*twIm + imB*twRe
                    AscendC::Reg::Mul(tRe, reB, twRe, mask);
                    AscendC::Reg::Mul(tIm, reB, twIm, mask);
                    AscendC::Reg::Neg(negImB, imB, mask);
                    AscendC::Reg::MulAddDst(tRe, negImB, twIm, mask);
                    AscendC::Reg::MulAddDst(tIm, imB, twRe, mask);
#else
                    AscendC::Reg::Mul(tRe, reB, twRe, mask);
                    AscendC::Reg::Mul(tIm, reB, twIm, mask);
                    AscendC::Reg::Mul(reB, imB, twIm, mask);
                    AscendC::Reg::Mul(imB, imB, twRe, mask);
                    AscendC::Reg::Sub(tRe, tRe, reB, mask);
                    AscendC::Reg::Add(tIm, tIm, imB, mask);
#endif

                    AscendC::Reg::Sub(reB, reA, tRe, mask);
                    AscendC::Reg::Add(reA, reA, tRe, mask);
                    AscendC::Reg::Sub(imB, imA, tIm, mask);
                    AscendC::Reg::Add(imA, imA, tIm, mask);

                    AscendC::Reg::StoreAlign(xRe + i * R2_SUB + colOff, reA, mask);
                    AscendC::Reg::StoreAlign(xIm + i * R2_SUB + colOff, imA, mask);
                    AscendC::Reg::StoreAlign(xRe + i2 * R2_SUB + colOff, reB, mask);
                    AscendC::Reg::StoreAlign(xIm + i2 * R2_SUB + colOff, imB, mask);
                }
            }
        }

        twOff += half * R2_VEC;
    }
}

constexpr bool R2_USE_GATHER = true;

__aicore__ inline void Phase5Radix2(
    LocalTensor<float>& ubS, LocalTensor<float>& ubXRe, LocalTensor<float>& ubXIm,
    __ubuf__ float* t2Re, __ubuf__ float* t2Im,
    __ubuf__ float* twTable, __ubuf__ uint32_t* phaseAOffsets)
{
    AscendC::PipeBarrier<PIPE_V>();

    __ubuf__ float* sAddr = (__ubuf__ float*)ubS.GetPhyAddr();
    __ubuf__ float* xReAddr = (__ubuf__ float*)ubXRe.GetPhyAddr();
    __ubuf__ float* xImAddr = (__ubuf__ float*)ubXIm.GetPhyAddr();

    if constexpr (R2_USE_GATHER) {
        simd_twiddle_rotate_brev_phase5_gather(
            xReAddr, xImAddr, sAddr, sAddr + R2_TOTAL, t2Re, t2Im, phaseAOffsets);
    }
    AscendC::DataSyncBarrier<AscendC::MemDsbT::UB>();

    simd_radix2_butterfly_4stage(xReAddr, xImAddr, twTable);
    AscendC::DataSyncBarrier<AscendC::MemDsbT::UB>();
}

#endif // FFT_RADIX2_OPS_H
