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

// 16 点 bit-reverse 查找表（与 fft_twiddle.h 的 BuildBitReverse16 一致）
constexpr uint32_t R2_BR_TABLE[16] = {0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15};

// ═══════════════════════════════════════════════════════════════════════════
// P5' 阶段 A：SIMT T2 复乘 + 轴旋转 + bit-reverse 重排（DIT 输入重排）
// ═══════════════════════════════════════════════════════════════════════════
__simt_vf__ __launch_bounds__(R2_THREADS) inline void simt_twiddle_rotate_brev_phase5(
    __ubuf__ float* dstRe, __ubuf__ float* dstIm,
    __ubuf__ float* srcRe, __ubuf__ float* srcIm,
    __ubuf__ float* t2Re, __ubuf__ float* t2Im)
{
    uint32_t tid = threadIdx.x;

    for (uint32_t i = tid; i < R2_TOTAL; i += R2_THREADS) {
        uint32_t n3 = i / R2_SUB;
        uint32_t c  = i - n3 * R2_SUB;
        uint32_t k3 = c & (R2_R - 1);
        uint32_t k2 = c >> 4;
        uint32_t srcIdx = k2 * R2_SUB + k3 * R2_R + n3;

        uint32_t br = (n3 & 1u) << 3 | (n3 & 2u) << 1 | (n3 & 4u) >> 1 | (n3 & 8u) >> 3;
        uint32_t dstIdx = br * R2_SUB + c;

        float tr = t2Re[i];
        float ti = t2Im[i];
        float ar = srcRe[srcIdx];
        float ai = srcIm[srcIdx];
        dstRe[dstIdx] = ar * tr - ai * ti;
        dstIm[dstIdx] = ar * ti + ai * tr;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 单个蝶形辅助函数（寄存器间操作，无 UB 访存）
// ═══════════════════════════════════════════════════════════════════════════
__simd_callee__ inline void butterfly_reg(
    AscendC::Reg::RegTensor<float>& reA, AscendC::Reg::RegTensor<float>& imA,
    AscendC::Reg::RegTensor<float>& reB, AscendC::Reg::RegTensor<float>& imB,
    AscendC::Reg::RegTensor<float>& twRe, AscendC::Reg::RegTensor<float>& twIm,
    AscendC::Reg::RegTensor<float>& tRe, AscendC::Reg::RegTensor<float>& tIm,
    AscendC::Reg::RegTensor<float>& tmpRe, AscendC::Reg::RegTensor<float>& tmpIm,
    AscendC::Reg::MaskReg& mask)
{
    AscendC::Reg::Mul(tRe, reB, twRe, mask);
    AscendC::Reg::Mul(tIm, reB, twIm, mask);
    AscendC::Reg::Mul(tmpRe, imB, twIm, mask);
    AscendC::Reg::Mul(tmpIm, imB, twRe, mask);
    AscendC::Reg::Sub(tRe, tRe, tmpRe, mask);
    AscendC::Reg::Add(tIm, tIm, tmpIm, mask);

    AscendC::Reg::Sub(reB, reA, tRe, mask);
    AscendC::Reg::Add(reA, reA, tRe, mask);
    AscendC::Reg::Sub(imB, imA, tIm, mask);
    AscendC::Reg::Add(imA, imA, tIm, mask);
}

// ─── 单蝶形宏：对具名寄存器 re##a/im##a 与 re##b/im##b 做蝶形，twiddle 用 j ───
#define R2_BF(a, b, j) \
    AscendC::Reg::LoadAlign(twRe, twTable + twOff + (j) * R2_VEC); \
    AscendC::Reg::LoadAlign(twIm, twTable + RADIX2_TW_FLOATS + twOff + (j) * R2_VEC); \
    butterfly_reg(re##a, im##a, re##b, im##b, twRe, twIm, tRe, tIm, tmpRe, tmpIm, mask)

// ─── 加载全部 16 行到具名寄存器 ───
#define R2_LOAD_ALL() \
    AscendC::Reg::LoadAlign(re0,  xRe + 0 * R2_SUB + colOff); AscendC::Reg::LoadAlign(im0,  xIm + 0 * R2_SUB + colOff); \
    AscendC::Reg::LoadAlign(re1,  xRe + 1 * R2_SUB + colOff); AscendC::Reg::LoadAlign(im1,  xIm + 1 * R2_SUB + colOff); \
    AscendC::Reg::LoadAlign(re2,  xRe + 2 * R2_SUB + colOff); AscendC::Reg::LoadAlign(im2,  xIm + 2 * R2_SUB + colOff); \
    AscendC::Reg::LoadAlign(re3,  xRe + 3 * R2_SUB + colOff); AscendC::Reg::LoadAlign(im3,  xIm + 3 * R2_SUB + colOff); \
    AscendC::Reg::LoadAlign(re4,  xRe + 4 * R2_SUB + colOff); AscendC::Reg::LoadAlign(im4,  xIm + 4 * R2_SUB + colOff); \
    AscendC::Reg::LoadAlign(re5,  xRe + 5 * R2_SUB + colOff); AscendC::Reg::LoadAlign(im5,  xIm + 5 * R2_SUB + colOff); \
    AscendC::Reg::LoadAlign(re6,  xRe + 6 * R2_SUB + colOff); AscendC::Reg::LoadAlign(im6,  xIm + 6 * R2_SUB + colOff); \
    AscendC::Reg::LoadAlign(re7,  xRe + 7 * R2_SUB + colOff); AscendC::Reg::LoadAlign(im7,  xIm + 7 * R2_SUB + colOff); \
    AscendC::Reg::LoadAlign(re8,  xRe + 8 * R2_SUB + colOff); AscendC::Reg::LoadAlign(im8,  xIm + 8 * R2_SUB + colOff); \
    AscendC::Reg::LoadAlign(re9,  xRe + 9 * R2_SUB + colOff); AscendC::Reg::LoadAlign(im9,  xIm + 9 * R2_SUB + colOff); \
    AscendC::Reg::LoadAlign(re10, xRe + 10 * R2_SUB + colOff); AscendC::Reg::LoadAlign(im10, xIm + 10 * R2_SUB + colOff); \
    AscendC::Reg::LoadAlign(re11, xRe + 11 * R2_SUB + colOff); AscendC::Reg::LoadAlign(im11, xIm + 11 * R2_SUB + colOff); \
    AscendC::Reg::LoadAlign(re12, xRe + 12 * R2_SUB + colOff); AscendC::Reg::LoadAlign(im12, xIm + 12 * R2_SUB + colOff); \
    AscendC::Reg::LoadAlign(re13, xRe + 13 * R2_SUB + colOff); AscendC::Reg::LoadAlign(im13, xIm + 13 * R2_SUB + colOff); \
    AscendC::Reg::LoadAlign(re14, xRe + 14 * R2_SUB + colOff); AscendC::Reg::LoadAlign(im14, xIm + 14 * R2_SUB + colOff); \
    AscendC::Reg::LoadAlign(re15, xRe + 15 * R2_SUB + colOff); AscendC::Reg::LoadAlign(im15, xIm + 15 * R2_SUB + colOff)

// ─── 写回全部 16 行 ───
#define R2_STORE_ALL() \
    AscendC::Reg::StoreAlign(xRe + 0 * R2_SUB + colOff, re0, mask);  AscendC::Reg::StoreAlign(xIm + 0 * R2_SUB + colOff, im0, mask); \
    AscendC::Reg::StoreAlign(xRe + 1 * R2_SUB + colOff, re1, mask);  AscendC::Reg::StoreAlign(xIm + 1 * R2_SUB + colOff, im1, mask); \
    AscendC::Reg::StoreAlign(xRe + 2 * R2_SUB + colOff, re2, mask);  AscendC::Reg::StoreAlign(xIm + 2 * R2_SUB + colOff, im2, mask); \
    AscendC::Reg::StoreAlign(xRe + 3 * R2_SUB + colOff, re3, mask);  AscendC::Reg::StoreAlign(xIm + 3 * R2_SUB + colOff, im3, mask); \
    AscendC::Reg::StoreAlign(xRe + 4 * R2_SUB + colOff, re4, mask);  AscendC::Reg::StoreAlign(xIm + 4 * R2_SUB + colOff, im4, mask); \
    AscendC::Reg::StoreAlign(xRe + 5 * R2_SUB + colOff, re5, mask);  AscendC::Reg::StoreAlign(xIm + 5 * R2_SUB + colOff, im5, mask); \
    AscendC::Reg::StoreAlign(xRe + 6 * R2_SUB + colOff, re6, mask);  AscendC::Reg::StoreAlign(xIm + 6 * R2_SUB + colOff, im6, mask); \
    AscendC::Reg::StoreAlign(xRe + 7 * R2_SUB + colOff, re7, mask);  AscendC::Reg::StoreAlign(xIm + 7 * R2_SUB + colOff, im7, mask); \
    AscendC::Reg::StoreAlign(xRe + 8 * R2_SUB + colOff, re8, mask);  AscendC::Reg::StoreAlign(xIm + 8 * R2_SUB + colOff, im8, mask); \
    AscendC::Reg::StoreAlign(xRe + 9 * R2_SUB + colOff, re9, mask);  AscendC::Reg::StoreAlign(xIm + 9 * R2_SUB + colOff, im9, mask); \
    AscendC::Reg::StoreAlign(xRe + 10 * R2_SUB + colOff, re10, mask); AscendC::Reg::StoreAlign(xIm + 10 * R2_SUB + colOff, im10, mask); \
    AscendC::Reg::StoreAlign(xRe + 11 * R2_SUB + colOff, re11, mask); AscendC::Reg::StoreAlign(xIm + 11 * R2_SUB + colOff, im11, mask); \
    AscendC::Reg::StoreAlign(xRe + 12 * R2_SUB + colOff, re12, mask); AscendC::Reg::StoreAlign(xIm + 12 * R2_SUB + colOff, im12, mask); \
    AscendC::Reg::StoreAlign(xRe + 13 * R2_SUB + colOff, re13, mask); AscendC::Reg::StoreAlign(xIm + 13 * R2_SUB + colOff, im13, mask); \
    AscendC::Reg::StoreAlign(xRe + 14 * R2_SUB + colOff, re14, mask); AscendC::Reg::StoreAlign(xIm + 14 * R2_SUB + colOff, im14, mask); \
    AscendC::Reg::StoreAlign(xRe + 15 * R2_SUB + colOff, re15, mask); AscendC::Reg::StoreAlign(xIm + 15 * R2_SUB + colOff, im15, mask)

// ═══════════════════════════════════════════════════════════════════════════
// P5' 阶段 B：SIMD 4 层 radix-2 butterfly
// 每蝶形一列块：加载 2 行 → 复乘 + 加减 → 写回
// 注：寄存器优化尝试（一次加载 16 行到 38 个 RegTensor）导致寄存器溢出，
// 性能从 308us 降到 785us，故保留此逐蝶形逐列块版本。
// ═══════════════════════════════════════════════════════════════════════════
__simd_vf__ inline void simd_radix2_butterfly_4stage(
    __ubuf__ float* xRe, __ubuf__ float* xIm,
    __ubuf__ float* twTable)
{
    AscendC::Reg::RegTensor<float> reA, imA, reB, imB;
    AscendC::Reg::RegTensor<float> twRe, twIm, tRe, tIm;
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

                    AscendC::Reg::Mul(tRe, reB, twRe, mask);
                    AscendC::Reg::Mul(tIm, reB, twIm, mask);
                    AscendC::Reg::Mul(reB, imB, twIm, mask);
                    AscendC::Reg::Mul(imB, imB, twRe, mask);
                    AscendC::Reg::Sub(tRe, tRe, reB, mask);
                    AscendC::Reg::Add(tIm, tIm, imB, mask);

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

// ═══════════════════════════════════════════════════════════════════════════
// P5' 阶段 A（SIMD Gather 版）：T2 复乘 + 轴旋转 + bit-reverse
// 用 Reg::Gather 做 stride-16 的 gather，SIMD 复乘，StoreAlign 写回
// 验证：Gather(vgather2) 使用元素偏移（非字节偏移），已通过 gather_test 确认
// ═══════════════════════════════════════════════════════════════════════════
__simd_vf__ inline void simd_twiddle_rotate_brev_phase5_gather(
    __ubuf__ float* dstRe, __ubuf__ float* dstIm,
    __ubuf__ float* srcRe, __ubuf__ float* srcIm,
    __ubuf__ float* t2Re, __ubuf__ float* t2Im,
    __ubuf__ uint32_t* offsetTable)
{
    AscendC::Reg::RegTensor<float> srcReReg, srcImReg, t2ReReg, t2ImReg;
    AscendC::Reg::RegTensor<float> outReReg, outImReg, tReReg, tImReg;
    AscendC::Reg::RegTensor<float> tmpReReg, tmpImReg;
    AscendC::Reg::RegTensor<uint32_t> idxReg;
    AscendC::Reg::MaskReg mask = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();

    for (uint16_t n3 = 0; n3 < 16; n3++) {
        uint16_t br = (n3 & 1u) << 3 | (n3 & 2u) << 1 | (n3 & 4u) >> 1 | (n3 & 8u) >> 3;
        for (uint16_t colBlk = 0; colBlk < 4; colBlk++) {
            AscendC::Reg::LoadAlign(idxReg, offsetTable + n3 * R2_SUB + colBlk * R2_VEC);
            AscendC::Reg::Gather(srcReReg, srcRe, idxReg, mask);
            AscendC::Reg::Gather(srcImReg, srcIm, idxReg, mask);

            AscendC::Reg::LoadAlign(t2ReReg, t2Re + n3 * R2_SUB + colBlk * R2_VEC);
            AscendC::Reg::LoadAlign(t2ImReg, t2Im + n3 * R2_SUB + colBlk * R2_VEC);

            AscendC::Reg::Mul(tReReg, srcReReg, t2ReReg, mask);
            AscendC::Reg::Mul(tImReg, srcReReg, t2ImReg, mask);
            AscendC::Reg::Mul(tmpReReg, srcImReg, t2ImReg, mask);
            AscendC::Reg::Mul(tmpImReg, srcImReg, t2ReReg, mask);
            AscendC::Reg::Sub(outReReg, tReReg, tmpReReg, mask);
            AscendC::Reg::Add(outImReg, tImReg, tmpImReg, mask);

            uint16_t dstOff = br * R2_SUB + colBlk * R2_VEC;
            AscendC::Reg::StoreAlign(dstRe + dstOff, outReReg, mask);
            AscendC::Reg::StoreAlign(dstIm + dstOff, outImReg, mask);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// P5' 组合调用：SIMT/SIMD-Gather 阶段 A → SIMD 4 层 butterfly
// R2_PHASE_SPLIT: 0=两段全跑，1=只跑阶段A，2=只跑阶段B（拆段测时用）
// R2_USE_GATHER: 1=阶段A用SIMD Gather（快），0=阶段A用SIMT
// ═══════════════════════════════════════════════════════════════════════════
constexpr int32_t R2_PHASE_SPLIT = 0;
constexpr bool R2_USE_GATHER = true;  // Gather+字节偏移，测试中

__aicore__ inline void Phase5Radix2(
    LocalTensor<float>& ubS, LocalTensor<float>& ubXRe, LocalTensor<float>& ubXIm,
    __ubuf__ float* t2Re, __ubuf__ float* t2Im,
    __ubuf__ float* twTable, __ubuf__ uint32_t* phaseAOffsets)
{
    AscendC::PipeBarrier<PIPE_V>();

    __ubuf__ float* sAddr = (__ubuf__ float*)ubS.GetPhyAddr();
    __ubuf__ float* xReAddr = (__ubuf__ float*)ubXRe.GetPhyAddr();
    __ubuf__ float* xImAddr = (__ubuf__ float*)ubXIm.GetPhyAddr();

    if constexpr (R2_PHASE_SPLIT != 2) {
        if constexpr (R2_USE_GATHER) {
            // 阶段 A：SIMD Gather 版（T2 + 旋转 + bit-reverse）
            simd_twiddle_rotate_brev_phase5_gather(
                xReAddr, xImAddr, sAddr, sAddr + R2_TOTAL, t2Re, t2Im, phaseAOffsets);
        } else {
            // 阶段 A：SIMT 版（T2 + 旋转 + bit-reverse）
            asc_vf_call<simt_twiddle_rotate_brev_phase5>(
                dim3(R2_THREADS), xReAddr, xImAddr,
                sAddr, sAddr + R2_TOTAL, t2Re, t2Im);
        }
        AscendC::DataSyncBarrier<AscendC::MemDsbT::UB>();
    }

    if constexpr (R2_PHASE_SPLIT != 1) {
        // 阶段 B：SIMD 4 层 radix-2 butterfly
        simd_radix2_butterfly_4stage(xReAddr, xImAddr, twTable);
        AscendC::DataSyncBarrier<AscendC::MemDsbT::UB>();
    }
}

#endif // FFT_RADIX2_OPS_H