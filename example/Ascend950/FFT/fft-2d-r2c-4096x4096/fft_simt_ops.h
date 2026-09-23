#ifndef FFT_SIMT_OPS_H
#define FFT_SIMT_OPS_H

#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "simt_api/vector_functions.h"
#include "fft_tiling_def.h"
#include "catlass/arch/arch.hpp"
#include "catlass/arch/resource.hpp"

constexpr uint32_t TS_TILE_DIM = 16;
constexpr uint32_t TS_THREADS = 2048;
constexpr uint32_t EVT_V_MTE3 = 11;

constexpr uint32_t TS_PAD_COLS = 8;

__simd_vf__ inline void simd_deinterleave_row(
    __ubuf__ float* dstReal, __ubuf__ float* dstImag, __ubuf__ float* src,
    uint16_t repeatTimes)
{
    constexpr uint16_t oneRepeatSize = AscendC::GetVecLen() / sizeof(float);
    AscendC::Reg::RegTensor<float> srcReg0, srcReg1;
    AscendC::Reg::RegTensor<float> dstReg0, dstReg1;
    AscendC::Reg::MaskReg mask = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();

    for (uint16_t i = 0; i < repeatTimes; i++) {
        AscendC::Reg::LoadAlign(srcReg0, src + i * oneRepeatSize * 2);
        AscendC::Reg::LoadAlign(srcReg1, src + i * oneRepeatSize * 2 + oneRepeatSize);
        AscendC::Reg::DeInterleave<float>(dstReg0, dstReg1, srcReg0, srcReg1);
        AscendC::Reg::StoreAlign(dstReal + i * oneRepeatSize, dstReg0, mask);
        AscendC::Reg::StoreAlign(dstImag + i * oneRepeatSize, dstReg1, mask);
    }
}

constexpr uint32_t TS_TILE_32 = 32;
constexpr uint32_t TS_PAD_2 = 2;
constexpr uint32_t TS_PAD_STRIDE_34 = TS_TILE_32 + TS_PAD_2;
constexpr uint32_t TS_TILE_32_ELEMS = TS_TILE_32 * TS_TILE_32;
constexpr uint32_t TS_TILES_PER_BLOCK_2 = TS_THREADS / TS_TILE_32_ELEMS;

__simt_vf__ __launch_bounds__(TS_THREADS) inline void simt_transpose_padded_sync(
    __ubuf__ float* dst, __ubuf__ float* src,
    __ubuf__ float* sharedTile,
    uint32_t srcRows, uint32_t srcCols)
{
    constexpr uint32_t TILE = TS_TILE_32;
    constexpr uint32_t PAD_STRIDE = TS_PAD_STRIDE_34;
    constexpr uint32_t TILE_ELEMS = TS_TILE_32_ELEMS;
    constexpr uint32_t TILES_PER_BLOCK = TS_TILES_PER_BLOCK_2;

    uint32_t tilesRow = (srcRows + TILE - 1) / TILE;
    uint32_t tilesCol = (srcCols + TILE - 1) / TILE;
    uint32_t totalTiles = tilesRow * tilesCol;

    uint32_t localTile = threadIdx.y >> 5;
    uint32_t ty = threadIdx.y & (TILE - 1);
    uint32_t tx = threadIdx.x;

    for (uint32_t t = localTile; t < totalTiles; t += TILES_PER_BLOCK) {
        uint32_t tr = t / tilesCol;
        uint32_t tc = t - tr * tilesCol;

        uint32_t srcR = tr * TILE + ty;
        uint32_t srcC = tc * TILE + tx;

        if (srcR < srcRows && srcC < srcCols) {
            sharedTile[localTile * TILE * PAD_STRIDE + ty * PAD_STRIDE + tx] =
                src[srcR * srcCols + srcC];
        } else {
            sharedTile[localTile * TILE * PAD_STRIDE + ty * PAD_STRIDE + tx] = 0.0f;
        }

        asc_syncthreads();

        uint32_t dstR = tc * TILE + ty;
        uint32_t dstC = tr * TILE + tx;

        if (dstR < srcCols && dstC < srcRows) {
            dst[dstR * srcRows + dstC] =
                sharedTile[localTile * TILE * PAD_STRIDE + tx * PAD_STRIDE + ty];
        }

        asc_syncthreads();
    }
}

__simt_vf__ __launch_bounds__(TS_THREADS) inline void simt_interleave_ub_to_ub(
    __ubuf__ float* dst, __ubuf__ float* srcR, __ubuf__ float* srcI,
    uint32_t srcRows, uint32_t srcCols)
{
    uint32_t total = srcRows * srcCols;
    uint32_t tid = threadIdx.x;
    uint32_t rStride = TS_THREADS / srcCols;
    uint32_t r = tid / srcCols;
    uint32_t c = tid - r * srcCols;
    uint32_t i = tid;
    uint32_t rowBase = r * srcCols * 2;
    for (; i < total; i += TS_THREADS, r += rStride, rowBase += rStride * srcCols * 2) {
        uint32_t srcIdx = r * srcCols + c;
        uint32_t dstIdx = rowBase + c * 2;
        dst[dstIdx]     = srcR[srcIdx];
        dst[dstIdx + 1] = srcI[srcIdx];
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// P7 interleave 的 VF 版（替代 simt_interleave_ub_to_ub）：sepRe[16][256] +
// sepIm[16][256] → interleave 缓冲 [16 行 × 512 floats]。语义一致：
// dst[r][2c]=srcR[r][c], dst[r][2c+1]=srcI[r][c]。Reg 级 API 全 __simd_callee__，
// 单线程 VEC 指令（无 2048 线程 SIMT 发射），ex101 已验证。
// ═══════════════════════════════════════════════════════════════════════════
__simd_vf__ inline void simd_interleave_rows_vf(
    __ubuf__ float* dst, __ubuf__ float* srcR, __ubuf__ float* srcI,
    uint16_t rows)
{
    constexpr uint16_t REG = 64;                       // 每寄存器 64 floats
    constexpr uint16_t ROW_FLOATS = 256;               // 源行长度（N2*N2）
    constexpr uint16_t HALF = ROW_FLOATS / 2;          // 128
    constexpr uint16_t REPS = HALF / REG;              // 2
    AscendC::Reg::RegTensor<float> aReg, bReg, dReg0, dReg1;
    AscendC::Reg::MaskReg mask = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();

    for (uint16_t r = 0; r < rows; r++) {
        uint32_t s = static_cast<uint32_t>(r) * ROW_FLOATS;
        uint32_t d = static_cast<uint32_t>(r) * 2 * ROW_FLOATS;
        for (uint16_t i = 0; i < REPS; i++) {
            AscendC::Reg::LoadAlign(aReg, srcR + s + i * REG);
            AscendC::Reg::LoadAlign(bReg, srcI + s + i * REG);
            AscendC::Reg::Interleave<float>(dReg0, dReg1, aReg, bReg);
            AscendC::Reg::StoreAlign(dst + d + i * 2 * REG, dReg0, mask);
            AscendC::Reg::StoreAlign(dst + d + i * 2 * REG + REG, dReg1, mask);

            AscendC::Reg::LoadAlign(aReg, srcR + s + HALF + i * REG);
            AscendC::Reg::LoadAlign(bReg, srcI + s + HALF + i * REG);
            AscendC::Reg::Interleave<float>(dReg0, dReg1, aReg, bReg);
            AscendC::Reg::StoreAlign(dst + d + HALF * 2 + i * 2 * REG, dReg0, mask);
            AscendC::Reg::StoreAlign(dst + d + HALF * 2 + i * 2 * REG + REG, dReg1, mask);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// P3 twiddle + 轴旋转的 VF 版（替代 simt_twiddle_rotate_phase3）：
// C0[k3,(n2,n3)] × T12(k3,n2) → B1[n2,(k3,n3)]。Reg Gather 做轴旋转、Reg 复乘
// 做 twiddle，T12 用展开表连续 LoadAlign。ex101 已验证。
// ═══════════════════════════════════════════════════════════════════════════
__simd_vf__ inline void simd_twiddle_rotate_phase3(
    __ubuf__ float* dstRe, __ubuf__ float* dstIm,
    __ubuf__ float* srcRe, __ubuf__ float* srcIm,
    __ubuf__ float* t12ExpRe, __ubuf__ float* t12ExpIm,
    __ubuf__ uint32_t* p3Lane)
{
    constexpr uint32_t R = 16;
    constexpr uint32_t SUB = R * R;      // 256
    constexpr uint32_t VEC = 64;

    AscendC::Reg::RegTensor<float> srcReReg, srcImReg, t12ReReg, t12ImReg;
    AscendC::Reg::RegTensor<float> outReReg, outImReg, negImReg;
    AscendC::Reg::RegTensor<uint32_t> idxReg;
    AscendC::Reg::MaskReg mask = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();

    for (uint32_t n2 = 0; n2 < R; n2++) {
        for (uint32_t k3blk = 0; k3blk < 4; k3blk++) {
            AscendC::Reg::LoadAlign(idxReg, p3Lane);
            AscendC::Reg::Adds<uint32_t>(idxReg, idxReg,
                (k3blk << 10) + (n2 << 4), mask);
            AscendC::Reg::Gather(srcReReg, srcRe, idxReg, mask);
            AscendC::Reg::Gather(srcImReg, srcIm, idxReg, mask);

            uint32_t tOff = ((n2 << 2) + k3blk) * VEC;
            AscendC::Reg::LoadAlign(t12ReReg, t12ExpRe + tOff);
            AscendC::Reg::LoadAlign(t12ImReg, t12ExpIm + tOff);

            AscendC::Reg::Mul(outReReg, srcReReg, t12ReReg, mask);
            AscendC::Reg::Mul(outImReg, srcReReg, t12ImReg, mask);
            AscendC::Reg::Neg(negImReg, srcImReg, mask);
            AscendC::Reg::MulAddDst(outReReg, negImReg, t12ImReg, mask);
            AscendC::Reg::MulAddDst(outImReg, srcImReg, t12ReReg, mask);

            uint32_t dstOff = n2 * SUB + k3blk * VEC;
            AscendC::Reg::StoreAlign(dstRe + dstOff, outReReg, mask);
            AscendC::Reg::StoreAlign(dstIm + dstOff, outImReg, mask);
        }
    }
}

__simt_vf__ __launch_bounds__(TS_THREADS) inline void simt_transpose_interleaved_pairs(
    __ubuf__ float* dst, __ubuf__ float* src,
    uint32_t srcRows, uint32_t srcCols)
{
    uint32_t total = srcRows * srcCols;
    uint32_t tid = threadIdx.x;
    uint32_t r = tid / srcCols;
    uint32_t c = tid - r * srcCols;

    for (uint32_t i = tid; i < total; i += TS_THREADS) {
        uint32_t srcIdx = r * srcCols * 2 + c * 2;
        uint32_t dstIdx = c * srcRows * 2 + r * 2;
        dst[dstIdx]     = src[srcIdx];
        dst[dstIdx + 1] = src[srcIdx + 1];
        r += TS_THREADS / srcCols;
    }
}

__aicore__ inline void ContiguousTransposeWrite(
    LocalTensor<float>& srcR, LocalTensor<float>& srcI,
    LocalTensor<float>& tmpR, LocalTensor<float>& tmpI,
    LocalTensor<float>& sharedTile,
    GM_ADDR dstBase, int64_t dstOffset, int64_t elemCount,
    uint32_t srcRows, uint32_t srcCols, uint32_t colOffset)
{
    AscendC::PipeBarrier<PIPE_V>();

    __ubuf__ float* srcRAddr = (__ubuf__ float*)srcR.GetPhyAddr();
    __ubuf__ float* srcIAddr = (__ubuf__ float*)srcI.GetPhyAddr();
    __ubuf__ float* tmpRAddr = (__ubuf__ float*)tmpR.GetPhyAddr();
    __ubuf__ float* tmpIAddr = (__ubuf__ float*)tmpI.GetPhyAddr();
    __ubuf__ float* sharedTileAddr = (__ubuf__ float*)sharedTile.GetPhyAddr();

    uint32_t totalElems = srcRows * srcCols;
    int64_t realOff = dstOffset + static_cast<int64_t>(colOffset) * srcRows;
    int64_t imagOff = dstOffset + elemCount + static_cast<int64_t>(colOffset) * srcRows;

    dim3 threadCfg(TS_TILE_32, TS_TILE_32 * TS_TILES_PER_BLOCK_2, 1);

    asc_vf_call<simt_transpose_padded_sync>(
        threadCfg, tmpRAddr, srcRAddr, sharedTileAddr, srcRows, srcCols);

    SetFlag<HardEvent::V_MTE3>(EVT_V_MTE3);
    WaitFlag<HardEvent::V_MTE3>(EVT_V_MTE3);

    GlobalTensor<float> gmDstR;
    gmDstR.SetGlobalBuffer((__gm__ float*)(dstBase) + realOff);
    DataCopy(gmDstR, tmpR, totalElems);

    asc_vf_call<simt_transpose_padded_sync>(
        threadCfg, tmpIAddr, srcIAddr, sharedTileAddr, srcRows, srcCols);

    SetFlag<HardEvent::V_MTE3>(EVT_V_MTE3);
    WaitFlag<HardEvent::V_MTE3>(EVT_V_MTE3);

    GlobalTensor<float> gmDstI;
    gmDstI.SetGlobalBuffer((__gm__ float*)(dstBase) + imagOff);
    DataCopy(gmDstI, tmpI, totalElems);


}

__aicore__ inline void InterleaveRow(
    LocalTensor<float>& srcR, LocalTensor<float>& srcI,
    LocalTensor<float>& tmpInterleaved,
    uint32_t srcRows, uint32_t srcCols)
{
    AscendC::PipeBarrier<PIPE_V>();

    __ubuf__ float* srcRAddr = (__ubuf__ float*)srcR.GetPhyAddr();
    __ubuf__ float* srcIAddr = (__ubuf__ float*)srcI.GetPhyAddr();
    __ubuf__ float* tmpAddr  = (__ubuf__ float*)tmpInterleaved.GetPhyAddr();

    asc_vf_call<simt_interleave_ub_to_ub>(
        dim3(TS_THREADS), tmpAddr, srcRAddr, srcIAddr,
        srcRows, srcCols);
}

__aicore__ inline void InterleaveTransposeRow(
    LocalTensor<float>& srcR, LocalTensor<float>& srcI,
    LocalTensor<float>& tmpInterleaved, LocalTensor<float>& tmpRaw,
    uint32_t srcRows, uint32_t srcCols)
{
    AscendC::PipeBarrier<PIPE_V>();

    __ubuf__ float* srcRAddr = (__ubuf__ float*)srcR.GetPhyAddr();
    __ubuf__ float* srcIAddr = (__ubuf__ float*)srcI.GetPhyAddr();
    __ubuf__ float* tmpAddr  = (__ubuf__ float*)tmpInterleaved.GetPhyAddr();
    __ubuf__ float* tmpRawAddr = (__ubuf__ float*)tmpRaw.GetPhyAddr();

    asc_vf_call<simt_interleave_ub_to_ub>(
        dim3(TS_THREADS), tmpRawAddr, srcRAddr, srcIAddr,
        srcRows, srcCols);

    asc_vf_call<simt_transpose_interleaved_pairs>(
        dim3(TS_THREADS), tmpAddr, tmpRawAddr, srcRows, srcCols);
}

// ═══════════════════════════════════════════════════════════════════════════
// 16×16×16 专用：twiddle + 轴旋转（P3/P5 阶段）
// 说明：16×16×16 的"转置"是 3D 轴旋转（256 列按 [slow*16+fast] 布局，块内并
// 非连续 16×16 tile），不复用 32×32 整块转置。改为 SIMT 直接
// "读 S(GEMM 结果) → 乘 twiddle → 写旋转后结果到 Sep"，每个输出元素只依赖
// 一个输入元素 → 无需 sharedTile，也无需 syncthreads。
// 索引约定（对齐 py_verify/fft_16x16x16.py 与设计文档）：
//   C0[k3,(n2,n3)] 列 = n2*16+n3（n3 最快）→ srcIdx = k3*256 + n2*16 + n3
//   B1[n2,(k3,n3)] 列 = k3*16+n3（n3 最快）→ dstIdx = n2*256 + k3*16 + n3
//   T12[k3,n2]  idx = k3*16 + n2（Re/Im 各 256 floats）
// ═══════════════════════════════════════════════════════════════════════════

// 纯轴旋转（不乘 twiddle）：P3 的 C0[k3,(n2,n3)] → B1[n2,(k3,n3)]
// 配合"向量 twiddle（展开表）"使用：S 已就地乘过 T12，这里只做索引重排。
// 索引与 simt_twiddle_rotate_phase3 一致：srcIdx = k3*256 + n2*16 + n3 → dstIdx = i
__simt_vf__ __launch_bounds__(TS_THREADS) inline void simt_rotate_phase3(
    __ubuf__ float* dstRe, __ubuf__ float* dstIm,
    __ubuf__ float* srcRe, __ubuf__ float* srcIm)
{
    constexpr uint32_t R = 16;
    constexpr uint32_t SUB = R * R;          // 256 子信号
    constexpr uint32_t TOTAL = R * SUB;      // 4096 复元素/信号
    uint32_t tid = threadIdx.x;

    for (uint32_t i = tid; i < TOTAL; i += TS_THREADS) {
        uint32_t n2 = i / SUB;               // 输出行（旋转后）
        uint32_t c  = i - n2 * SUB;
        uint32_t n3 = c & (R - 1);
        uint32_t k3 = c >> 4;
        uint32_t srcIdx = k3 * SUB + n2 * R + n3;
        dstRe[i] = srcRe[srcIdx];
        dstIm[i] = srcIm[srcIdx];
    }
}

// P3: C0[k3,(n2,n3)] ×T12(k3,n2)[广播到 n3] → B1[n2,(k3,n3)]
// P3_MODE: 0=完整(twiddle+轴旋转)  1=只轴旋转(不乘twiddle)  2=只twiddle(不重排)
// 用于 profiling 拆解 P3 两大逻辑步（twiddle 与轴旋转）各自开销。
constexpr int32_t P3_MODE = 0;
__simt_vf__ __launch_bounds__(TS_THREADS) inline void simt_twiddle_rotate_phase3(
    __ubuf__ float* dstRe, __ubuf__ float* dstIm,
    __ubuf__ float* srcRe, __ubuf__ float* srcIm,
    __ubuf__ float* t12Re, __ubuf__ float* t12Im)
{
    constexpr uint32_t R = 16;
    constexpr uint32_t SUB = R * R;          // 256 子信号
    constexpr uint32_t TOTAL = R * SUB;      // 4096 复元素/信号/轮
    uint32_t tid = threadIdx.x;

    for (uint32_t i = tid; i < TOTAL; i += TS_THREADS) {
        uint32_t n2 = i / SUB;               // 输出行（旋转后）
        uint32_t c  = i - n2 * SUB;          // 输出列 = k3*16 + n3
        uint32_t n3 = c & (R - 1);           // 快下标
        uint32_t k3 = c >> 4;                // 慢下标
        uint32_t srcIdx = k3 * SUB + n2 * R + n3;
        uint32_t tIdx   = k3 * R + n2;

        if constexpr (P3_MODE == 1) {
            // 只轴旋转（不乘 twiddle）
            dstRe[i] = srcRe[srcIdx];
            dstIm[i] = srcIm[srcIdx];
        } else if constexpr (P3_MODE == 2) {
            // 只 twiddle（不重排：直接读 src[i]）
            float ar = srcRe[i];
            float ai = srcIm[i];
            float tr = t12Re[tIdx];
            float ti = t12Im[tIdx];
            dstRe[i] = ar * tr - ai * ti;
            dstIm[i] = ar * ti + ai * tr;
        } else {
            // 完整：轴旋转 + twiddle
            float ar = srcRe[srcIdx];
            float ai = srcIm[srcIdx];
            float tr = t12Re[tIdx];
            float ti = t12Im[tIdx];
            dstRe[i] = ar * tr - ai * ti;
            dstIm[i] = ar * ti + ai * tr;
        }
    }
}

// P5: C1[k2,(k3,n3)] ×T23(k2,n3)[广播到 k3]×T13(k3,n3)[广播到 k2] → B2[n3,(k2,k3)]
//   S  : srcIdx = k2*256 + k3*16 + n3
//   Sep: dstIdx = n3*256 + k2*16 + k3
//   T23[k2,n3] idx = k2*16+n3；T13[k3,n3] idx = k3*16+n3（Re/Im 各 256 floats）
__simt_vf__ __launch_bounds__(TS_THREADS) inline void simt_twiddle_rotate_phase5(
    __ubuf__ float* dstRe, __ubuf__ float* dstIm,
    __ubuf__ float* srcRe, __ubuf__ float* srcIm,
    __ubuf__ float* t23Re, __ubuf__ float* t23Im,
    __ubuf__ float* t13Re, __ubuf__ float* t13Im)
{
    constexpr uint32_t R = 16;
    constexpr uint32_t SUB = R * R;
    constexpr uint32_t TOTAL = R * SUB;
    uint32_t tid = threadIdx.x;

    for (uint32_t i = tid; i < TOTAL; i += TS_THREADS) {
        uint32_t n3 = i / SUB;               // 输出行（旋转后）
        uint32_t c  = i - n3 * SUB;          // 输出列 = k2*16 + k3
        uint32_t k3 = c & (R - 1);
        uint32_t k2 = c >> 4;
        uint32_t srcIdx = k2 * SUB + k3 * R + n3;

        // 合并 T2 = T23·T13（复乘）
        uint32_t t23Idx = k2 * R + n3;
        uint32_t t13Idx = k3 * R + n3;
        float aR = t23Re[t23Idx], aI = t23Im[t23Idx];
        float bR = t13Re[t13Idx], bI = t13Im[t13Idx];
        float tr = aR * bR - aI * bI;
        float ti = aR * bI + aI * bR;

        float ar = srcRe[srcIdx];
        float ai = srcIm[srcIdx];
        dstRe[i] = ar * tr - ai * ti;
        dstIm[i] = ar * ti + ai * tr;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// P5 tile 化转置 + 融合 twiddle（替代上面 stride-16 直读版本）
// 思路：按 k2 行并行，读 S(源) 连续 → 转置存 sharedTile → 连续读 sharedTile →
// 融合 twiddle(T23·T13) 写 Sep(输出)。把原"连续线程 stride-16 读源"转移到
// sharedTile（小热缓冲），源读与输出写都变成连续/16 连续块。
// 每波并行 P5_PARR 行，共用 2048 线程（1024 re + 1024 im）。
// sharedTile 需求 = P5_PARR × P5_PAD × 2(re/im) floats。
// ═══════════════════════════════════════════════════════════════════════════
constexpr uint32_t P5_PARR = 4;             // 每波并行 k2 行数
constexpr uint32_t P5_PAD  = 264;           // sharedTile 行 padding（256+8，防 bank 冲突）
constexpr uint32_t P5_TILE_FLOATS = P5_PARR * P5_PAD;   // 1056（re 或 im 各一份）

__simt_vf__ __launch_bounds__(TS_THREADS) inline void simt_twiddle_rotate_phase5_tiled(
    __ubuf__ float* dstRe, __ubuf__ float* dstIm,
    __ubuf__ float* srcRe, __ubuf__ float* srcIm,
    __ubuf__ float* t23Re, __ubuf__ float* t23Im,
    __ubuf__ float* t13Re, __ubuf__ float* t13Im,
    __ubuf__ float* shRe, __ubuf__ float* shIm)
{
    constexpr uint32_t R = 16;
    constexpr uint32_t SUB = R * R;         // 256
    constexpr uint32_t ROWS = R;            // 16（k2）
    constexpr uint32_t PARR = P5_PARR;      // 4
    constexpr uint32_t PAD  = P5_PAD;       // 264
    constexpr uint32_t HALF = PARR * SUB;   // 1024

    uint32_t tid = threadIdx.x;             // 0..2047
    uint32_t half = tid / HALF;             // 0=re, 1=im
    uint32_t t2 = tid - half * HALF;        // 0..1023
    uint32_t r = t2 / SUB;                  // 并行行 0..PARR-1
    uint32_t idx = t2 - r * SUB;            // 0..255

    for (uint32_t w = 0; w < ROWS / PARR; w++) {
        uint32_t k2 = w * PARR + r;         // 源行（k2）

        // ── Load：源序 idx=k3*16+n3 → 读 S 连续(256)，转置写 sharedTile ──
        uint32_t k3 = idx >> 4;             // idx/16
        uint32_t n3 = idx & (R - 1);        // idx%16
        uint32_t srcIdx = k2 * SUB + idx;   // S[k2][k3,n3]（连续）
        uint32_t shIdx  = r * PAD + n3 * R + k3;   // sh[n3][k3] = S[k2][k3,n3]
        if (half == 0) shRe[shIdx] = srcRe[srcIdx];
        else           shIm[shIdx] = srcIm[srcIdx];

        asc_syncthreads();

        // ── Store：输出序 idx=n3*16+k3（k3 最快）→ 连续读 sh，融合 twiddle 写 Sep ──
        uint32_t n3o = idx >> 4;            // 输出序解读
        uint32_t k3o = idx & (R - 1);
        float ar = shRe[r * PAD + n3o * R + k3o];
        float ai = shIm[r * PAD + n3o * R + k3o];

        // T2 = T23[k2,n3]·T13[k3,n3]
        uint32_t t23Idx = k2 * R + n3o;
        uint32_t t13Idx = k3o * R + n3o;
        float aR = t23Re[t23Idx], aI = t23Im[t23Idx];
        float bR = t13Re[t13Idx], bI = t13Im[t13Idx];
        float tr = aR * bR - aI * bI;
        float ti = aR * bI + aI * bR;

        uint32_t dstIdx = n3o * SUB + k2 * R + k3o;   // Sep[n3][k2,k3]
        if (half == 0) dstRe[dstIdx] = ar * tr - ai * ti;
        else           dstIm[dstIdx] = ar * ti + ai * tr;

        asc_syncthreads();
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// P5 block17：16×16 块 + PAD=17（低 bank 冲突）+ 16 行单波（仅 2 sync/调用）
// 修正版：Tiled16 用 256-长行 PAD=264，块内 shared 访问仍是 stride-16（高冲突）；
// 本版把每 k2 行看作 16×16(k3×n3) 块，用行距 17 存储（16 数据+1 padding），
// 使 shared 读/写都是 stride-17（与 bank 周期互质，低冲突）。
// 源读(S)连续、shared 访问 stride-17、输出写(Sep)16 连续块。
// shared 需求 = 16 行 × 16 × 17 × 2(re/im) = 8704 floats（复用 T12 展开表区）。
// ═══════════════════════════════════════════════════════════════════════════
constexpr uint32_t P5_PAD17         = 17;
constexpr uint32_t P5_BLOCK17_STRIDE = 16 * P5_PAD17;         // 272
constexpr uint32_t P5_BLOCK17_SH     = 16 * P5_BLOCK17_STRIDE; // 4352（re 或 im 各一份）

__simt_vf__ __launch_bounds__(TS_THREADS) inline void simt_twiddle_rotate_phase5_block17(
    __ubuf__ float* dstRe, __ubuf__ float* dstIm,
    __ubuf__ float* srcRe, __ubuf__ float* srcIm,
    __ubuf__ float* t23Re, __ubuf__ float* t23Im,
    __ubuf__ float* t13Re, __ubuf__ float* t13Im,
    __ubuf__ float* shRe, __ubuf__ float* shIm)
{
    constexpr uint32_t R = 16;
    constexpr uint32_t SUB = R * R;                 // 256
    constexpr uint32_t TOTAL = R * SUB;             // 4096 复
    constexpr uint32_t ALL = 2 * TOTAL;             // 8192（re+im 位置）
    constexpr uint32_t PAD = P5_PAD17;              // 17
    constexpr uint32_t STRIDE = P5_BLOCK17_STRIDE;  // 272
    constexpr uint32_t ITERS = ALL / TS_THREADS;    // 4

    uint32_t tid = threadIdx.x;                     // 0..2047

    // ── Load：源序 idx=k3*16+n3 → 读 S 连续，写 shared（k3 行距 17）──
    for (uint32_t it = 0; it < ITERS; it++) {
        uint32_t p = it * TS_THREADS + tid;
        uint32_t half = p / TOTAL;                  // 0=re, 1=im
        uint32_t q = p - half * TOTAL;              // 0..4095
        uint32_t r = q / SUB;                       // k2 行 0..15
        uint32_t idx = q - r * SUB;                 // 源列 k3*16+n3
        uint32_t k3 = idx >> 4;
        uint32_t n3 = idx & (R - 1);
        uint32_t shIdx = r * STRIDE + k3 * PAD + n3;
        if (half == 0) shRe[shIdx] = srcRe[q];
        else           shIm[shIdx] = srcIm[q];
    }

    asc_syncthreads();

    // ── Store：输出序 idx=n3*16+k3（k3 最快）→ 读 shared(stride 17) + twiddle → 写 Sep ──
    for (uint32_t it = 0; it < ITERS; it++) {
        uint32_t p = it * TS_THREADS + tid;
        uint32_t half = p / TOTAL;
        uint32_t q = p - half * TOTAL;
        uint32_t r = q / SUB;
        uint32_t idx = q - r * SUB;                 // 输出序 n3*16+k3
        uint32_t n3o = idx >> 4;
        uint32_t k3o = idx & (R - 1);

        float ar = shRe[r * STRIDE + k3o * PAD + n3o];
        float ai = shIm[r * STRIDE + k3o * PAD + n3o];

        // T2 = T23[k2,n3]·T13[k3,n3]
        uint32_t t23Idx = r * R + n3o;
        uint32_t t13Idx = k3o * R + n3o;
        float aR = t23Re[t23Idx], aI = t23Im[t23Idx];
        float bR = t13Re[t13Idx], bI = t13Im[t13Idx];
        float tr = aR * bR - aI * bI;
        float ti = aR * bI + aI * bR;

        uint32_t dstIdx = n3o * SUB + r * R + k3o;  // Sep[n3][k2,k3]
        if (half == 0) dstRe[dstIdx] = ar * tr - ai * ti;
        else           dstIm[dstIdx] = ar * ti + ai * tr;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// P5 源序版（A0 不共享 + 侧交换）：每线程 = 一个 (k2,k3)，处理 n3=0..15。
// 读：src[k2*256 + k3*16 + n3]，每线程 16 个连续（可向量化，64B 全利用率）
// 写：dst[n3*256 + k2*16 + k3]，每线程 16 个散开（stride-256）
// 与 Stride16 正好相反：读全利用、写散开。256 线程。
// 目的：测"哪一侧的散访问更贵"（读散 vs 写散）。
// ═══════════════════════════════════════════════════════════════════════════
constexpr uint32_t P5_SRC_THREADS = 256;

__simt_vf__ __launch_bounds__(P5_SRC_THREADS) inline void simt_twiddle_rotate_phase5_srcorder(
    __ubuf__ float* dstRe, __ubuf__ float* dstIm,
    __ubuf__ float* srcRe, __ubuf__ float* srcIm,
    __ubuf__ float* t23Re, __ubuf__ float* t23Im,
    __ubuf__ float* t13Re, __ubuf__ float* t13Im)
{
    constexpr uint32_t R = 16;
    constexpr uint32_t SUB = R * R;          // 256
    uint32_t t = threadIdx.x;                // 0..255
    uint32_t k2 = t >> 4;                    // t/16
    uint32_t k3 = t & (R - 1);               // t%16
    uint32_t base = k2 * SUB + k3 * R;       // k2*256 + k3*16

    for (uint32_t n3 = 0; n3 < R; n3++) {
        float ar = srcRe[base + n3];         // 每线程 16 连续读
        float ai = srcIm[base + n3];

        // T2 = T23[k2,n3]·T13[k3,n3]
        uint32_t t23Idx = k2 * R + n3;
        uint32_t t13Idx = k3 * R + n3;
        float aR = t23Re[t23Idx], aI = t23Im[t23Idx];
        float bR = t13Re[t13Idx], bI = t13Im[t13Idx];
        float tr = aR * bR - aI * bI;
        float ti = aR * bI + aI * bR;

        uint32_t dstIdx = n3 * SUB + k2 * R + k3;   // 每线程 16 散写（stride-256）
        dstRe[dstIdx] = ar * tr - ai * ti;
        dstIm[dstIdx] = ar * ti + ai * tr;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// P5 输出序 ILP 版（A0 不共享 + 少线程多元素）：512 线程 × 8 元素（grid-stride）。
// 与 Stride16 相同的读/写模式（读 stride-16、写连续），但每线程 8 个独立 load
// （ILP=8）→ 测"延迟隐藏不足"假设：若 233us 是延迟没藏住，本版应更快。
// ═══════════════════════════════════════════════════════════════════════════
constexpr uint32_t P5_ILP_THREADS = 512;

__simt_vf__ __launch_bounds__(P5_ILP_THREADS) inline void simt_twiddle_rotate_phase5_outilp(
    __ubuf__ float* dstRe, __ubuf__ float* dstIm,
    __ubuf__ float* srcRe, __ubuf__ float* srcIm,
    __ubuf__ float* t23Re, __ubuf__ float* t23Im,
    __ubuf__ float* t13Re, __ubuf__ float* t13Im)
{
    constexpr uint32_t R = 16;
    constexpr uint32_t SUB = R * R;                 // 256
    constexpr uint32_t TOTAL = R * SUB;             // 4096
    constexpr uint32_t ITERS = TOTAL / P5_ILP_THREADS;  // 8
    uint32_t tid = threadIdx.x;                     // 0..511

    for (uint32_t it = 0; it < ITERS; it++) {
        uint32_t i = it * P5_ILP_THREADS + tid;     // 输出序
        uint32_t n3 = i / SUB;
        uint32_t c = i - n3 * SUB;
        uint32_t k3 = c & (R - 1);
        uint32_t k2 = c >> 4;
        uint32_t srcIdx = k2 * SUB + k3 * R + n3;

        // T2 = T23[k2,n3]·T13[k3,n3]
        uint32_t t23Idx = k2 * R + n3;
        uint32_t t13Idx = k3 * R + n3;
        float aR = t23Re[t23Idx], aI = t23Im[t23Idx];
        float bR = t13Re[t13Idx], bI = t13Im[t13Idx];
        float tr = aR * bR - aI * bI;
        float ti = aR * bI + aI * bR;

        float ar = srcRe[srcIdx];
        float ai = srcIm[srcIdx];
        dstRe[i] = ar * tr - ai * ti;
        dstIm[i] = ar * ti + ai * tr;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// P5 纯布局调整（无任何数值计算）：在 Stride16 基础上去掉 T2=T23·T13 复乘与
// twiddle 乘，只保留"stride-16 读 + 连续写"的搬移逻辑。
// 目的：隔离 P5 这一步骤的纯访存/搬移成本，与完整版对比可拆出算术开销。
// ═══════════════════════════════════════════════════════════════════════════
__simt_vf__ __launch_bounds__(TS_THREADS) inline void simt_rotate_phase5_nocalc(
    __ubuf__ float* dstRe, __ubuf__ float* dstIm,
    __ubuf__ float* srcRe, __ubuf__ float* srcIm)
{
    constexpr uint32_t R = 16;
    constexpr uint32_t SUB = R * R;          // 256
    constexpr uint32_t TOTAL = R * SUB;      // 4096
    uint32_t tid = threadIdx.x;

    for (uint32_t i = tid; i < TOTAL; i += TS_THREADS) {
        uint32_t n3 = i / SUB;
        uint32_t c  = i - n3 * SUB;
        uint32_t k3 = c & (R - 1);
        uint32_t k2 = c >> 4;
        uint32_t srcIdx = k2 * SUB + k3 * R + n3;
        dstRe[i] = srcRe[srcIdx];
        dstIm[i] = srcIm[srcIdx];
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// P5 融合 SIMT + 输出布局 T2 展开表（方案 A）：
// 与 Stride16 同样的单遍结构，但 twiddle 不现算——直接读宿主预展开的
// T2e[i]（按输出 i 排布，连续读），只做 1 次复乘。
// 去掉：T23/T13 两张 strided 读 + T2=T23·T13 计算（4mul2add）+ 两级依赖链。
// ═══════════════════════════════════════════════════════════════════════════
__simt_vf__ __launch_bounds__(TS_THREADS) inline void simt_twiddle_rotate_phase5_t2exp(
    __ubuf__ float* dstRe, __ubuf__ float* dstIm,
    __ubuf__ float* srcRe, __ubuf__ float* srcIm,
    __ubuf__ float* t2Re, __ubuf__ float* t2Im)
{
    constexpr uint32_t R = 16;
    constexpr uint32_t SUB = R * R;          // 256
    constexpr uint32_t TOTAL = R * SUB;      // 4096
    uint32_t tid = threadIdx.x;

    for (uint32_t i = tid; i < TOTAL; i += TS_THREADS) {
        uint32_t n3 = i / SUB;
        uint32_t c  = i - n3 * SUB;
        uint32_t k3 = c & (R - 1);
        uint32_t k2 = c >> 4;
        uint32_t srcIdx = k2 * SUB + k3 * R + n3;

        float tr = t2Re[i];                  // 连续（输出布局）
        float ti = t2Im[i];
        float ar = srcRe[srcIdx];
        float ai = srcIm[srcIdx];
        dstRe[i] = ar * tr - ai * ti;
        dstIm[i] = ar * ti + ai * tr;
    }
}

#endif // FFT_SIMT_OPS_H