#ifndef FFT_SIMT_OPS_H
#define FFT_SIMT_OPS_H

#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "fft_tiling_def.h"
#include "catlass/arch/arch.hpp"
#include "catlass/arch/resource.hpp"

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
constexpr uint32_t TS_THREADS = 2048;
constexpr uint32_t TS_TILE_32 = 32;
constexpr uint32_t TS_PAD_COLS = 2;
constexpr uint32_t TS_PAD_STRIDE_34 = TS_TILE_32 + TS_PAD_COLS;
constexpr uint32_t TS_TILE_32_ELEMS = TS_TILE_32 * TS_TILE_32;
constexpr uint32_t TS_TILES_PER_BLOCK_2 = TS_THREADS / TS_TILE_32_ELEMS;

// ===========================================================================
// SIMD DeInterleave: separate interleaved [re,im,re,im,...] -> [re,re,...] + [im,im,...]
// Reused from ex85 — identical operation for reading complex input rows.
//   src: contiguous interleaved complex (N*2 floats)
//   dstReal / dstImag: separated output (N floats each)
//   repeatTimes = N / 64 (for float)
// ---------------------------------------------------------------------------
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

// ===========================================================================
// SIMT VF: Hermitian expansion — complete half-spectrum to full complex row.
//
// For C2R (inverse FFT), the input is a half-spectrum [halfN] complex.
// The full spectrum [fftN] is reconstructed via conjugate symmetric completion:
//   X_full[0..halfN-1] = X_input[0..halfN-1]  (direct copy)
//   X_full[v] = conj(X_input[fftN - v])        for v = halfN..fftN-1
//
// The mirror row index is (fftN - row) % fftN, so for row m:
//   X_full[m, v] = conj(X_input[(fftN-m)%fftN, fftN-v])
//
// Parameters:
//   dst:          output [fftN*2] interleaved complex (full row)
//   srcRow:       input row [halfN*2] interleaved complex (this row's half-spectrum)
//   srcMirrorRow: input mirror row [halfN*2] interleaved complex
//   fftN:         FFT size (e.g. 4096)
//   halfN:        half-spectrum size (fftN/2 + 1, e.g. 2049)
// ---------------------------------------------------------------------------
__simt_vf__ __launch_bounds__(TS_THREADS) inline void simt_hermitian_expand_row(
    __ubuf__ float* dst,
    __ubuf__ float* srcRow,
    __ubuf__ float* srcMirrorRow,
    uint32_t fftN,
    uint32_t halfN)
{
    uint32_t tid = threadIdx.x;

    for (uint32_t v = tid; v < fftN; v += TS_THREADS) {
        if (v < halfN) {
            // Direct copy: X_full[v] = X[v]
            dst[v * 2] = srcRow[v * 2];
            dst[v * 2 + 1] = srcRow[v * 2 + 1];
        } else {
            // Hermitian: X_full[v] = conj(X_mirror[fftN - v])
            uint32_t mirrorCol = fftN - v;  // in range [1, halfN-2]
            float re = srcMirrorRow[mirrorCol * 2];
            float im = srcMirrorRow[mirrorCol * 2 + 1];
            dst[v * 2] = re;
            dst[v * 2 + 1] = -im;  // conjugate
        }
    }
}

// ===========================================================================
// SIMT VF: 2-stage twiddle multiply + rearrange.
// Reused from ex85 — same operation, runtime N1=N2=64 for C2R.
//
//   srcR/srcI: GEMM0 output C0 [N1×N2] real/imag halves (separated)
//   twR/twI: twiddle [N1×N2] real/imag
//   dst: B1 [2*N2×N1] rearranged output (rows [0,N2)=real, [N2,2*N2)=imag)
//
// For each (k1, n2) in [N1, N2):
//   yr = srcR[k1*N2 + n2], yi = srcI[k1*N2 + n2]
//   tr = twR[k1*N2 + n2],  ti = twI[k1*N2 + n2]
//   outR = yr*tr - yi*ti,  outI = yr*ti + yi*tr
//   dst[n2*N1 + k1]             = outR
//   dst[N2*N1 + n2*N1 + k1]     = outI
// ---------------------------------------------------------------------------
__simt_vf__ __launch_bounds__(TS_THREADS) inline void simt_twiddle_rearrange_2stage(
    __ubuf__ float* dst, __ubuf__ float* srcR, __ubuf__ float* srcI,
    __ubuf__ float* twR, __ubuf__ float* twI,
    uint32_t N1, uint32_t N2)
{
    uint32_t total = N1 * N2;
    uint32_t tid = threadIdx.x;
    for (uint32_t idx = tid; idx < total; idx += TS_THREADS) {
        uint32_t k1 = idx / N2;
        uint32_t n2 = idx - k1 * N2;

        uint32_t srcIdx = k1 * N2 + n2;
        float yr = srcR[srcIdx];
        float yi = srcI[srcIdx];
        float tr = twR[srcIdx];
        float ti = twI[srcIdx];

        float outR = yr * tr - yi * ti;
        float outI = yr * ti + yi * tr;

        uint32_t dstIdx = n2 * N1 + k1;
        dst[dstIdx] = outR;
        dst[N2 * N1 + dstIdx] = outI;
    }
}

// ===========================================================================
// SIMD VF: Real extraction — take real part from separated GEMM1 output.
// 纯连续拷贝（dst[i] = srcR[i]），SIMD 256B/cyc。
// ---------------------------------------------------------------------------
__simd_vf__ inline void simd_real_extract(
    __ubuf__ float* dst,
    __ubuf__ float* srcR,
    uint32_t total)
{
    Reg::RegTensor<float> r;
    Reg::MaskReg mask = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
    constexpr uint32_t VEC = 64;
    for (uint32_t off = 0; off < total; off += VEC) {
        Reg::LoadAlign(r, srcR + off);
        Reg::StoreAlign(dst + off, r, mask);
    }
}

// ===========================================================================
// SIMT scatter transpose: each thread writes one complex from UB to GM
// Reused from ex85 — for non-square matrix transpose (ws0 -> ws1)
// ---------------------------------------------------------------------------
__simt_vf__ __launch_bounds__(TS_THREADS) inline void simt_scatter_transpose_row(
    __gm__ float* gmDst,
    __ubuf__ float* ubSrc,
    uint32_t srcCols,
    uint32_t dstRowFloats,
    uint32_t dstColFloatOff,
    uint32_t dstOffset)
{
    uint32_t tid = threadIdx.x;
    for (uint32_t c = tid; c < srcCols; c += TS_THREADS) {
        float re = ubSrc[c * 2];
        float im = ubSrc[c * 2 + 1];
        uint32_t gmOff = dstOffset + c * dstRowFloats + dstColFloatOff;
        gmDst[gmOff] = re;
        gmDst[gmOff + 1] = im;
    }
}

// ===========================================================================
// Host-callable wrappers
// ---------------------------------------------------------------------------
__aicore__ inline void DeinterleaveRow(
    LocalTensor<float>& dstReal, LocalTensor<float>& dstImag,
    LocalTensor<float>& src, uint16_t repeatTimes)
{
    __ubuf__ float* dstR = (__ubuf__ float*)dstReal.GetPhyAddr();
    __ubuf__ float* dstI = (__ubuf__ float*)dstImag.GetPhyAddr();
    __ubuf__ float* s = (__ubuf__ float*)src.GetPhyAddr();
    asc_vf_call<simd_deinterleave_row>(dstR, dstI, s, repeatTimes);
    AscendC::DataSyncBarrier<AscendC::MemDsbT::UB>();
}

__aicore__ inline void HermitianExpandRow(
    LocalTensor<float>& dst,
    LocalTensor<float>& srcRow,
    LocalTensor<float>& srcMirrorRow,
    uint32_t fftN, uint32_t halfN)
{
    AscendC::PipeBarrier<PIPE_V>();
    __ubuf__ float* d = (__ubuf__ float*)dst.GetPhyAddr();
    __ubuf__ float* sr = (__ubuf__ float*)srcRow.GetPhyAddr();
    __ubuf__ float* sm = (__ubuf__ float*)srcMirrorRow.GetPhyAddr();
    asc_vf_call<simt_hermitian_expand_row>(
        dim3(TS_THREADS), d, sr, sm, fftN, halfN);
}

__aicore__ inline void TwiddleRearrange2Stage(
    LocalTensor<float>& dst,
    LocalTensor<float>& srcR, LocalTensor<float>& srcI,
    LocalTensor<float>& twR, LocalTensor<float>& twI,
    uint32_t N1, uint32_t N2)
{
    AscendC::PipeBarrier<PIPE_V>();
    __ubuf__ float* d = (__ubuf__ float*)dst.GetPhyAddr();
    __ubuf__ float* sR = (__ubuf__ float*)srcR.GetPhyAddr();
    __ubuf__ float* sI = (__ubuf__ float*)srcI.GetPhyAddr();
    __ubuf__ float* tR = (__ubuf__ float*)twR.GetPhyAddr();
    __ubuf__ float* tI = (__ubuf__ float*)twI.GetPhyAddr();
    asc_vf_call<simt_twiddle_rearrange_2stage>(
        dim3(TS_THREADS), d, sR, sI, tR, tI, N1, N2);
}

__aicore__ inline void RealExtract(
    LocalTensor<float>& dst,
    LocalTensor<float>& srcR,
    uint32_t N1, uint32_t N2)
{
    AscendC::PipeBarrier<PIPE_V>();
    __ubuf__ float* d = (__ubuf__ float*)dst.GetPhyAddr();
    __ubuf__ float* sR = (__ubuf__ float*)srcR.GetPhyAddr();
    simd_real_extract(d, sR, N1 * N2);
    AscendC::DataSyncBarrier<AscendC::MemDsbT::UB>();
}

__aicore__ inline void ScatterTransposeRow(
    __gm__ float* gmDst, LocalTensor<float>& ubSrc,
    uint32_t srcCols, uint32_t dstRowFloats,
    uint32_t dstColFloatOff, uint32_t dstOffset)
{
    AscendC::PipeBarrier<PIPE_V>();
    __ubuf__ float* ubSrcPtr = (__ubuf__ float*)ubSrc.GetPhyAddr();
    asc_vf_call<simt_scatter_transpose_row>(
        dim3(TS_THREADS), gmDst, ubSrcPtr,
        srcCols, dstRowFloats, dstColFloatOff, dstOffset);
}

// ===========================================================================
// SIMT VF: Scatter real values to transposed GM positions.
// Each thread writes one float from UB to GM at transposed offset.
//   gmDst[col * fftN + rowIdx] = ubSrc[col]
// ---------------------------------------------------------------------------
__simt_vf__ __launch_bounds__(TS_THREADS) inline void simt_scatter_real_transpose(
    __gm__ float* gmDst, __ubuf__ float* ubSrc,
    uint32_t fftN, uint32_t rowIdx)
{
    uint32_t tid = threadIdx.x;
    for (uint32_t col = tid; col < fftN; col += TS_THREADS) {
        gmDst[col * fftN + rowIdx] = ubSrc[col];
    }
}

__aicore__ inline void ScatterRealTranspose(
    __gm__ float* gmDst, LocalTensor<float>& ubSrc,
    uint32_t fftN, uint32_t rowIdx)
{
    AscendC::PipeBarrier<PIPE_V>();
    __ubuf__ float* ubSrcPtr = (__ubuf__ float*)ubSrc.GetPhyAddr();
    asc_vf_call<simt_scatter_real_transpose>(
        dim3(TS_THREADS), gmDst, ubSrcPtr, fftN, rowIdx);
}

// ===========================================================================
// SIMD VF: Interleave separated [re,re,...] + [im,im,...] → [re,im,re,im,...]
// C1 GEMM output is in (n2, n1) order where n = n2*N1 + n1 (natural CRT order).
// 用 Reg::Interleave（DeInterleave 的逆）做向量交织。
// ---------------------------------------------------------------------------
__simd_vf__ inline void simd_interleave_c1(
    __ubuf__ float* dst, __ubuf__ float* srcR, __ubuf__ float* srcI,
    uint32_t total)
{
    Reg::RegTensor<float> r, i, o0, o1;
    Reg::MaskReg mask = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
    constexpr uint32_t VEC = 64;
    for (uint32_t off = 0; off < total; off += VEC) {
        Reg::LoadAlign(r, srcR + off);
        Reg::LoadAlign(i, srcI + off);
        Reg::Interleave(o0, o1, r, i);
        Reg::StoreAlign(dst + 2 * off, o0, mask);
        Reg::StoreAlign(dst + 2 * off + VEC, o1, mask);
    }
}

// Host wrapper
__aicore__ inline void InterleaveC1(
    LocalTensor<float>& dst,
    LocalTensor<float>& srcR, LocalTensor<float>& srcI,
    uint32_t N1, uint32_t N2)
{
    AscendC::PipeBarrier<PIPE_V>();
    __ubuf__ float* d = (__ubuf__ float*)dst.GetPhyAddr();
    __ubuf__ float* sR = (__ubuf__ float*)srcR.GetPhyAddr();
    __ubuf__ float* sI = (__ubuf__ float*)srcI.GetPhyAddr();
    uint32_t total = N1 * N2;
    simd_interleave_c1(d, sR, sI, total);
    AscendC::DataSyncBarrier<AscendC::MemDsbT::UB>();
}

#endif // FFT_SIMT_OPS_H
