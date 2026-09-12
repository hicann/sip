/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
// Selected arithmetic and movement from the frozen standalone RegBase import.
// See ../../../IMPORT.md for the source identity and adapter changes.
#ifndef ASDSIP_FFT_REGBASE_IMPL_H
#define ASDSIP_FFT_REGBASE_IMPL_H
#include "c_api/asc_simd.h"
#include "kernel_operator.h"
#include "../../tiling/fft_regbase_tiling_data.h"
#include <cstdint>
namespace {
constexpr int32_t kComplexPerVector = 64;
constexpr int32_t kFloatsPerVector = 64;
constexpr int32_t kComplexLoadFloats = 2 * kFloatsPerVector;
constexpr int32_t kComplexBytes = 2 * static_cast<int32_t>(sizeof(float));

template <int32_t N>
struct FftTraits {
    static constexpr int32_t kStages = N == 16   ? 4 :
                                       N == 32   ? 5 :
                                       N == 64   ? 6 :
                                       N == 128  ? 7 :
                                       N == 256  ? 8 :
                                       N == 512  ? 9 :
                                       N == 1024 ? 10 :
                                       N == 2048 ? 11 :
                                       N == 4096 ? 12 :
                                       N == 8192 ? 13 :
                                                   0;
};

// One radix-2 Stockham stage in the notation used by host_fft.cpp.
//
// q is vectorized directly. u=input[q] and v=input[q+N/2] are contiguous,
// while the Stockham output permutation is expressed through Scatter:
//   block   = q / half
//   j       = q % half
//   output0 = block * length + j = 2*q - j
//   output1 = output0 + half
//
// Gather and Scatter indexes are float-element offsets from aligned UB bases.
// All active Scatter indexes are unique. The inactive lanes are masked.
template <int32_t N>
__simd_callee__ inline void StockhamRadix2StageBody(__ubuf__ float* input, __ubuf__ float* twiddles,
                                                    __ubuf__ float* output, uint16_t tileBatch, uint32_t half)
{
    using namespace AscendC;

    constexpr uint16_t qChunkCount = static_cast<uint16_t>(((N / 2) + kComplexPerVector - 1) / kComplexPerVector);
    constexpr uint32_t activeButterflies = (N / 2 < kComplexPerVector) ? N / 2 : kComplexPerVector;

    for (uint16_t qChunk = 0; qChunk < qChunkCount; ++qChunk) {
        const uint32_t qBase = static_cast<uint32_t>(qChunk) * static_cast<uint32_t>(kComplexPerVector);

        for (uint16_t transform = 0; transform < tileBatch; ++transform) {
            Reg::RegTensor<float> uReal, uImag, vReal, vImag;
            Reg::RegTensor<float> wReal, wImag, tReal, tImag, temporary;
            Reg::RegTensor<uint32_t> q, j, halfMask, index, output0;

            uint32_t active = activeButterflies;
            Reg::MaskReg mask = Reg::UpdateMask<float>(active);

            const uint32_t transformFloatBase = static_cast<uint32_t>(transform) * static_cast<uint32_t>(N * 2);
            __ubuf__ float* u = input + transformFloatBase + (qBase << 1);
            __ubuf__ float* v = u + N;
            __ubuf__ float* transformOutput = output + transformFloatBase;

            // Load 64 interleaved complex values as two 64-float registers.
            // The allocation has a 64-complex guard, because LoadAlign itself
            // is unmasked even when only the first active lanes are consumed.
            Reg::LoadAlign<float, Reg::LoadDist::DIST_DINTLV_B32>(uReal, uImag, u);
            Reg::LoadAlign<float, Reg::LoadDist::DIST_DINTLV_B32>(vReal, vImag, v);

            // Arange is documented for int32_t, while float Gather/Scatter
            // require uint32_t indexes. q is non-negative, so the bit pattern
            // is preserved by this register-view cast.
            Reg::Arange<int32_t>(reinterpret_cast<Reg::RegTensor<int32_t>&>(q), static_cast<int32_t>(qBase));
            Reg::Duplicate(halfMask, half - 1U, mask);
            Reg::And(j, q, halfMask, mask);

            // Compact host twiddle offset: (half - 1) + j. Gather uses float
            // element offsets, hence the multiplication by two.
            Reg::Adds(index, j, half - 1U, mask);
            Reg::Muls(index, index, uint32_t{2}, mask);
            Reg::Gather(wReal, twiddles, index, mask);
            Reg::Adds(index, index, uint32_t{1}, mask);
            Reg::Gather(wImag, twiddles, index, mask);

            // t = v * w, followed by a = u + t and b = u - t.
            Reg::Mul(tReal, vReal, wReal, mask);
            Reg::Mul(temporary, vImag, wImag, mask);
            Reg::Sub(tReal, tReal, temporary, mask);

            Reg::Mul(tImag, vReal, wImag, mask);
            Reg::Mul(temporary, vImag, wReal, mask);
            Reg::Add(tImag, tImag, temporary, mask);

            Reg::Sub(vReal, uReal, tReal, mask);
            Reg::Sub(vImag, uImag, tImag, mask);
            Reg::Add(uReal, uReal, tReal, mask);
            Reg::Add(uImag, uImag, tImag, mask);

            if (half >= static_cast<uint32_t>(kComplexPerVector)) {
                // A full 64-butterfly q chunk stays inside one Stockham
                // block. Its a and b ranges are therefore contiguous. The
                // documented dual store interleaves real/imag in one 512-B
                // operation; it ignores mask, hence this path is full-only.
                const uint32_t jBase = qBase & (half - 1U);
                const uint32_t output0Base = 2U * qBase - jBase;
                Reg::StoreAlign<float, Reg::StoreDist::DIST_INTLV_B32>(transformOutput + 2U * output0Base, uReal, uImag,
                                                                       mask);
                Reg::StoreAlign<float, Reg::StoreDist::DIST_INTLV_B32>(transformOutput + 2U * (output0Base + half),
                                                                       vReal, vImag, mask);
            } else {
                // output0 = 2*q-j. Reuse one index register for real/imag
                // and output0/output1. Scatter handles early-stage runs that
                // alternate at a granularity smaller than one vector.
                Reg::Muls(output0, q, uint32_t{2}, mask);
                Reg::Sub(output0, output0, j, mask);
                Reg::Muls(index, output0, uint32_t{2}, mask);
                Reg::Scatter(transformOutput, uReal, index, mask);
                Reg::Adds(index, index, uint32_t{1}, mask);
                Reg::Scatter(transformOutput, uImag, index, mask);
                Reg::Adds(index, index, 2U * half - 1U, mask);
                Reg::Scatter(transformOutput, vReal, index, mask);
                Reg::Adds(index, index, uint32_t{1}, mask);
                Reg::Scatter(transformOutput, vImag, index, mask);
            }
        }
    }
}

template <int32_t N, int32_t Stage>
__simd_callee__ inline void StockhamRadix2UnrolledStages(__ubuf__ float* buffer0, __ubuf__ float* twiddles,
                                                         __ubuf__ float* buffer1, uint16_t tileBatch)
{
    constexpr uint32_t half = uint32_t{1} << Stage;
    if constexpr ((Stage & 1) == 0) {
        StockhamRadix2StageBody<N>(buffer0, twiddles, buffer1, tileBatch, half);
    } else {
        StockhamRadix2StageBody<N>(buffer1, twiddles, buffer0, tileBatch, half);
    }

    if constexpr (Stage + 1 < FftTraits<N>::kStages) {
        AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();
        StockhamRadix2UnrolledStages<N, Stage + 1>(buffer0, twiddles, buffer1, tileBatch);
    }
}

template <int32_t N>
__no_simd_vf_fusion__ __simd_vf__ inline void StockhamRadix2FusedUnrolledDirectVf(__ubuf__ float* buffer0,
                                                                                  __ubuf__ float* twiddles,
                                                                                  __ubuf__ float* buffer1,
                                                                                  uint16_t tileBatch)
{
    StockhamRadix2UnrolledStages<N, 0>(buffer0, twiddles, buffer1, tileBatch);
}

// 64-lane vector with one logical q chunk, so every lane is active and the
// Stockham repack can be represented by the physical lane order instead of
// register Gather+Select:
//
//   before stage s, physical lane l represents
//     q_s(l) = high_bits(l) | bit_reverse_s(low_s_bits(l));
//   Interleave(A, B) advances q_s to q_(s+1).
//
// The compact host twiddle address remains (half - 1) + j.  j is advanced by
// the same perfect shuffle as the data, so it always equals q % half in the
// current physical lane order.  Only the final Scatter converts the lazy lane
// order back to natural batch-major Stockham order.
template <uint32_t Half, uint32_t JOffset = 0U>
__simd_callee__ inline void StockhamRadix2LazyLoadTwiddle(
    __ubuf__ float* twiddles, AscendC::Reg::RegTensor<uint32_t>& j, AscendC::Reg::RegTensor<uint32_t>& index,
    AscendC::Reg::RegTensor<float>& wReal, AscendC::Reg::RegTensor<float>& wImag, AscendC::Reg::MaskReg& fullMask)
{
    using namespace AscendC;
    static_assert(Half > 1U, "the unit-twiddle stage must not load the table");

    // Gather indexes are FP32 element offsets into the interleaved complex
    // table.  N=256's final second q chunk supplies JOffset=64.
    Reg::Adds(index, j, Half - 1U + JOffset, fullMask);
    Reg::Muls(index, index, uint32_t{2}, fullMask);
    Reg::Gather(wReal, twiddles, index, fullMask);
    Reg::Adds(index, index, uint32_t{1}, fullMask);
    Reg::Gather(wImag, twiddles, index, fullMask);
}

// UnitTwiddle selects only the h=1 add/sub shortcut. All later stages use the
// same complex multiply; their actual half remains exclusively in j/twiddles.
template <bool UnitTwiddle>
__simd_callee__ inline void StockhamRadix2LazyButterfly(
    AscendC::Reg::RegTensor<float>& uReal, AscendC::Reg::RegTensor<float>& uImag, AscendC::Reg::RegTensor<float>& vReal,
    AscendC::Reg::RegTensor<float>& vImag, AscendC::Reg::RegTensor<float>& wReal, AscendC::Reg::RegTensor<float>& wImag,
    AscendC::Reg::RegTensor<float>& tReal, AscendC::Reg::RegTensor<float>& tImag,
    AscendC::Reg::RegTensor<float>& temporary, AscendC::Reg::MaskReg& fullMask)
{
    using namespace AscendC;

    if constexpr (UnitTwiddle) {
        // Save b before overwriting u with a.  Register assignment below is a
        // rename/copy of the complete vector, not a UB access.
        Reg::Sub(tReal, uReal, vReal, fullMask);
        Reg::Sub(tImag, uImag, vImag, fullMask);
        Reg::Add(uReal, uReal, vReal, fullMask);
        Reg::Add(uImag, uImag, vImag, fullMask);
        vReal = tReal;
        vImag = tImag;
    } else {
        // t = v*w; then retain a in u and b in v.  This exact arithmetic order
        // matches the existing primitive Stockham kernels and CPU oracle.
        Reg::Mul(tReal, vReal, wReal, fullMask);
        Reg::Mul(temporary, vImag, wImag, fullMask);
        Reg::Sub(tReal, tReal, temporary, fullMask);
        Reg::Mul(tImag, vReal, wImag, fullMask);
        Reg::Mul(temporary, vImag, wReal, fullMask);
        Reg::Add(tImag, tImag, temporary, fullMask);

        Reg::Sub(vReal, uReal, tReal, fullMask);
        Reg::Sub(vImag, uImag, tImag, fullMask);
        Reg::Add(uReal, uReal, tReal, fullMask);
        Reg::Add(uImag, uImag, tImag, fullMask);
    }
}

template <uint32_t Half>
__simd_callee__ inline void StockhamRadix2LazyAdvanceJ(AscendC::Reg::RegTensor<uint32_t>& j,
                                                       AscendC::Reg::RegTensor<uint32_t>& index,
                                                       AscendC::Reg::MaskReg& fullMask)
{
    using namespace AscendC;
    Reg::Adds(index, j, Half, fullMask);
    // Source/destination aliasing is permitted and the two destinations remain
    // distinct.  The second result can reuse index because it is no longer live.
    Reg::Interleave(j, index, j, index);
}

template <uint32_t Half>
__simd_callee__ inline void StockhamRadix2LazyStage128(
    __ubuf__ float* twiddles, AscendC::Reg::RegTensor<float>& uReal, AscendC::Reg::RegTensor<float>& uImag,
    AscendC::Reg::RegTensor<float>& vReal, AscendC::Reg::RegTensor<float>& vImag, AscendC::Reg::RegTensor<float>& work0,
    AscendC::Reg::RegTensor<float>& work1, AscendC::Reg::RegTensor<float>& work2, AscendC::Reg::RegTensor<uint32_t>& j,
    AscendC::Reg::RegTensor<uint32_t>& index, AscendC::Reg::MaskReg& fullMask)
{
    using namespace AscendC;
    if constexpr (Half == 1U) {
        // Form a/b in two scratch registers and feed them directly to the
        // perfect shuffle.  This avoids aggregate RegTensor assignments and
        // keeps only two arithmetic temporaries live in the unit stage.
        Reg::Sub(work0, uReal, vReal, fullMask);
        Reg::Add(work1, uReal, vReal, fullMask);
        Reg::Interleave(uReal, vReal, work1, work0);
        Reg::Sub(work0, uImag, vImag, fullMask);
        Reg::Add(work1, uImag, vImag, fullMask);
        Reg::Interleave(uImag, vImag, work1, work0);
    } else {
        // Stream the two twiddle components through work0.  Once the real
        // products are captured, the consumed v registers hold ad/bd.  The
        // operation order is still t.real=ac-bd and t.imag=ad+bc.
        Reg::Adds(index, j, Half - 1U, fullMask);
        Reg::Muls(index, index, uint32_t{2}, fullMask);
        Reg::Gather(work0, twiddles, index, fullMask);
        Reg::Mul(work1, vReal, work0, fullMask); // ac
        Reg::Mul(work2, vImag, work0, fullMask); // bc
        Reg::Adds(index, index, uint32_t{1}, fullMask);
        Reg::Gather(work0, twiddles, index, fullMask);
        Reg::Mul(vReal, vReal, work0, fullMask); // ad
        Reg::Mul(vImag, vImag, work0, fullMask); // bd
        Reg::Sub(work1, work1, vImag, fullMask);
        Reg::Add(work2, vReal, work2, fullMask);
        Reg::Sub(vReal, uReal, work1, fullMask);
        Reg::Sub(vImag, uImag, work2, fullMask);
        Reg::Add(uReal, uReal, work1, fullMask);
        Reg::Add(uImag, uImag, work2, fullMask);
        Reg::Interleave(uReal, vReal, uReal, vReal);
        Reg::Interleave(uImag, vImag, uImag, vImag);
    }
    StockhamRadix2LazyAdvanceJ<Half>(j, index, fullMask);
}

template <uint32_t Half>
__simd_callee__ inline void StockhamRadix2LazyFinal128(
    __ubuf__ float* twiddles, AscendC::Reg::RegTensor<float>& uReal, AscendC::Reg::RegTensor<float>& uImag,
    AscendC::Reg::RegTensor<float>& vReal, AscendC::Reg::RegTensor<float>& vImag, AscendC::Reg::RegTensor<float>& work0,
    AscendC::Reg::RegTensor<float>& work1, AscendC::Reg::RegTensor<float>& work2, AscendC::Reg::RegTensor<uint32_t>& j,
    AscendC::Reg::RegTensor<uint32_t>& index, AscendC::Reg::MaskReg& fullMask)
{
    using namespace AscendC;
    static_assert(Half == 64U, "N=128 final Stockham half must be 64");
    Reg::Adds(index, j, Half - 1U, fullMask);
    Reg::Muls(index, index, uint32_t{2}, fullMask);
    Reg::Gather(work0, twiddles, index, fullMask);
    Reg::Mul(work1, vReal, work0, fullMask);
    Reg::Mul(work2, vImag, work0, fullMask);
    Reg::Adds(index, index, uint32_t{1}, fullMask);
    Reg::Gather(work0, twiddles, index, fullMask);
    Reg::Mul(vReal, vReal, work0, fullMask);
    Reg::Mul(vImag, vImag, work0, fullMask);
    Reg::Sub(work1, work1, vImag, fullMask);
    Reg::Add(work2, vReal, work2, fullMask);
    Reg::Sub(vReal, uReal, work1, fullMask);
    Reg::Sub(vImag, uImag, work2, fullMask);
    Reg::Add(uReal, uReal, work1, fullMask);
    Reg::Add(uImag, uImag, work2, fullMask);
}

__simd_callee__ inline void StockhamRadix2LazyStore128(__ubuf__ float* output, AscendC::Reg::RegTensor<float>& aReal,
                                                       AscendC::Reg::RegTensor<float>& aImag,
                                                       AscendC::Reg::RegTensor<float>& bReal,
                                                       AscendC::Reg::RegTensor<float>& bImag,
                                                       AscendC::Reg::RegTensor<uint32_t>& bitReverse6,
                                                       AscendC::Reg::RegTensor<uint32_t>& index,
                                                       AscendC::Reg::MaskReg& fullMask)
{
    using namespace AscendC;
    Reg::Muls(index, bitReverse6, uint32_t{2}, fullMask);
    Reg::Scatter(output, aReal, index, fullMask);
    Reg::Adds(index, index, uint32_t{1}, fullMask);
    Reg::Scatter(output, aImag, index, fullMask);
    Reg::Adds(index, index, uint32_t{127}, fullMask);
    Reg::Scatter(output, bReal, index, fullMask);
    Reg::Adds(index, index, uint32_t{1}, fullMask);
    Reg::Scatter(output, bImag, index, fullMask);
}

__simd_callee__ inline void StockhamRadix2LazyStoreContiguous128(
    __ubuf__ float* output, AscendC::Reg::RegTensor<float>& aReal, AscendC::Reg::RegTensor<float>& aImag,
    AscendC::Reg::RegTensor<float>& bReal, AscendC::Reg::RegTensor<float>& bImag,
    AscendC::Reg::RegTensor<float>& naturalReal, AscendC::Reg::RegTensor<float>& naturalImag,
    AscendC::Reg::RegTensor<uint32_t>& bitReverse6, AscendC::Reg::MaskReg& fullMask)
{
    using namespace AscendC;
    // bit_reverse_6 is self-inverse. Gather therefore converts each physical
    // bit-reversed register to natural q order before one aligned AoS store.
    Reg::Gather(naturalReal, aReal, bitReverse6);
    Reg::Gather(naturalImag, aImag, bitReverse6);
    Reg::StoreAlign<float, Reg::StoreDist::DIST_INTLV_B32>(output, naturalReal, naturalImag, fullMask);
    Reg::Gather(naturalReal, bReal, bitReverse6);
    Reg::Gather(naturalImag, bImag, bitReverse6);
    Reg::StoreAlign<float, Reg::StoreDist::DIST_INTLV_B32>(output + 128U, naturalReal, naturalImag, fullMask);
}

template <bool ContiguousStore>
__no_simd_vf_fusion__ __simd_vf__ inline void StockhamRadix2FusedRegisterLazy128DirectVf(__ubuf__ float* buffer0,
                                                                                         __ubuf__ float* twiddles,
                                                                                         __ubuf__ float* buffer1,
                                                                                         uint16_t tileBatch)
{
    using namespace AscendC;
    Reg::RegTensor<float> uReal, uImag, vReal, vImag;
    Reg::RegTensor<float> work0, work1, work2;
    Reg::RegTensor<uint32_t> j, index;
    Reg::MaskReg fullMask = Reg::CreateMask<float, Reg::MaskPattern::ALL>();

#pragma unroll 1
    for (uint16_t transform = 0; transform < tileBatch; ++transform) {
        const uint32_t transformFloatBase = static_cast<uint32_t>(transform) * 256U;
        __ubuf__ float* inputTransform = buffer0 + transformFloatBase;
        __ubuf__ float* outputTransform = buffer1 + transformFloatBase;

        // Each load consumes exactly 64 complex FP32 values.  Both halves are
        // resident before the first final Scatter, making buffer0==buffer1 safe.
        Reg::LoadAlign<float, Reg::LoadDist::DIST_DINTLV_B32>(uReal, uImag, inputTransform);
        Reg::LoadAlign<float, Reg::LoadDist::DIST_DINTLV_B32>(vReal, vImag, inputTransform + 128U);
        Reg::Duplicate(j, uint32_t{0}, fullMask);

        // half=1 is kept branch-free.  The remaining non-final stages use a
        // short Aux-Scalar loop rather than cloning five copies of the data
        // flow.  Bisheng otherwise extends SSA live ranges across stages and
        // spills even though the mathematical live set is only nine VRegs.
        Reg::Sub(work0, uReal, vReal, fullMask);
        Reg::Add(work1, uReal, vReal, fullMask);
        Reg::Interleave(uReal, vReal, work1, work0);
        Reg::Sub(work0, uImag, vImag, fullMask);
        Reg::Add(work1, uImag, vImag, fullMask);
        Reg::Interleave(uImag, vImag, work1, work0);
        Reg::Adds(index, j, uint32_t{1}, fullMask);
        Reg::Interleave(j, index, j, index);

#pragma unroll 1
        for (uint32_t half = 2U; half < 64U; half <<= 1U) {
            Reg::Adds(index, j, half - 1U, fullMask);
            Reg::Muls(index, index, uint32_t{2}, fullMask);
            Reg::Gather(work0, twiddles, index, fullMask);
            Reg::Mul(work1, vReal, work0, fullMask);
            Reg::Mul(work2, vImag, work0, fullMask);
            Reg::Adds(index, index, uint32_t{1}, fullMask);
            Reg::Gather(work0, twiddles, index, fullMask);
            Reg::Mul(vReal, vReal, work0, fullMask);
            Reg::Mul(vImag, vImag, work0, fullMask);
            Reg::Sub(work1, work1, vImag, fullMask);
            Reg::Add(work2, vReal, work2, fullMask);
            Reg::Sub(vReal, uReal, work1, fullMask);
            Reg::Sub(vImag, uImag, work2, fullMask);
            Reg::Add(uReal, uReal, work1, fullMask);
            Reg::Add(uImag, uImag, work2, fullMask);
            Reg::Interleave(uReal, vReal, uReal, vReal);
            Reg::Interleave(uImag, vImag, uImag, vImag);
            Reg::Adds(index, j, half, fullMask);
            Reg::Interleave(j, index, j, index);
        }

        // Final half=64 stage: j is bit_reverse_6(lane).  Do not Interleave;
        // direct Scatter restores natural q and q+N/2 locations.
        StockhamRadix2LazyFinal128<64U>(twiddles, uReal, uImag, vReal, vImag, work0, work1, work2, j, index, fullMask);
        if constexpr (ContiguousStore) {
            StockhamRadix2LazyStoreContiguous128(outputTransform, uReal, uImag, vReal, vImag, work0, work1, j,
                                                 fullMask);
        } else {
            StockhamRadix2LazyStore128(outputTransform, uReal, uImag, vReal, vImag, j, index, fullMask);
        }
    }
}

template <uint32_t Half, bool Layout0>
__simd_callee__ inline void StockhamRadix2LazyInterleaveStage256(
    __ubuf__ float* twiddles, AscendC::Reg::RegTensor<float>& r0Real, AscendC::Reg::RegTensor<float>& r0Imag,
    AscendC::Reg::RegTensor<float>& r1Real, AscendC::Reg::RegTensor<float>& r1Imag,
    AscendC::Reg::RegTensor<float>& r2Real, AscendC::Reg::RegTensor<float>& r2Imag,
    AscendC::Reg::RegTensor<float>& r3Real, AscendC::Reg::RegTensor<float>& r3Imag,
    AscendC::Reg::RegTensor<float>& wReal, AscendC::Reg::RegTensor<float>& wImag, AscendC::Reg::RegTensor<float>& tReal,
    AscendC::Reg::RegTensor<float>& tImag, AscendC::Reg::RegTensor<float>& temporary,
    AscendC::Reg::RegTensor<uint32_t>& j, AscendC::Reg::RegTensor<uint32_t>& index, AscendC::Reg::MaskReg& fullMask)
{
    using namespace AscendC;
    if constexpr (Half > 1U) {
        StockhamRadix2LazyLoadTwiddle<Half>(twiddles, j, index, wReal, wImag, fullMask);
    }

    if constexpr (Layout0) {
        // L0=[U0,U1,V0,V1].  In-place Interleave leaves
        // L1=[U0,V0,U1,V1] for the next stage.
        StockhamRadix2LazyButterfly<Half == 1U>(r0Real, r0Imag, r2Real, r2Imag, wReal, wImag, tReal, tImag, temporary,
                                                fullMask);
        StockhamRadix2LazyButterfly<Half == 1U>(r1Real, r1Imag, r3Real, r3Imag, wReal, wImag, tReal, tImag, temporary,
                                                fullMask);
        Reg::Interleave(r0Real, r2Real, r0Real, r2Real);
        Reg::Interleave(r0Imag, r2Imag, r0Imag, r2Imag);
        Reg::Interleave(r1Real, r3Real, r1Real, r3Real);
        Reg::Interleave(r1Imag, r3Imag, r1Imag, r3Imag);
    } else {
        // L1=[U0,V0,U1,V1].  The same operation returns to L0.
        StockhamRadix2LazyButterfly<Half == 1U>(r0Real, r0Imag, r1Real, r1Imag, wReal, wImag, tReal, tImag, temporary,
                                                fullMask);
        StockhamRadix2LazyButterfly<Half == 1U>(r2Real, r2Imag, r3Real, r3Imag, wReal, wImag, tReal, tImag, temporary,
                                                fullMask);
        Reg::Interleave(r0Real, r1Real, r0Real, r1Real);
        Reg::Interleave(r0Imag, r1Imag, r0Imag, r1Imag);
        Reg::Interleave(r2Real, r3Real, r2Real, r3Real);
        Reg::Interleave(r2Imag, r3Imag, r2Imag, r3Imag);
    }
    StockhamRadix2LazyAdvanceJ<Half>(j, index, fullMask);
}

__simd_callee__ inline void StockhamRadix2LazyStore256(
    __ubuf__ float* output, AscendC::Reg::RegTensor<float>& a0Real, AscendC::Reg::RegTensor<float>& a0Imag,
    AscendC::Reg::RegTensor<float>& b0Real, AscendC::Reg::RegTensor<float>& b0Imag,
    AscendC::Reg::RegTensor<float>& a1Real, AscendC::Reg::RegTensor<float>& a1Imag,
    AscendC::Reg::RegTensor<float>& b1Real, AscendC::Reg::RegTensor<float>& b1Imag,
    AscendC::Reg::RegTensor<uint32_t>& bitReverse6, AscendC::Reg::RegTensor<uint32_t>& index,
    AscendC::Reg::MaskReg& fullMask)
{
    using namespace AscendC;
    Reg::Muls(index, bitReverse6, uint32_t{2}, fullMask);
    Reg::Scatter(output, a0Real, index, fullMask);
    Reg::Adds(index, index, uint32_t{1}, fullMask);
    Reg::Scatter(output, a0Imag, index, fullMask);
    Reg::Adds(index, index, uint32_t{127}, fullMask);
    Reg::Scatter(output, a1Real, index, fullMask);
    Reg::Adds(index, index, uint32_t{1}, fullMask);
    Reg::Scatter(output, a1Imag, index, fullMask);
    Reg::Adds(index, index, uint32_t{127}, fullMask);
    Reg::Scatter(output, b0Real, index, fullMask);
    Reg::Adds(index, index, uint32_t{1}, fullMask);
    Reg::Scatter(output, b0Imag, index, fullMask);
    Reg::Adds(index, index, uint32_t{127}, fullMask);
    Reg::Scatter(output, b1Real, index, fullMask);
    Reg::Adds(index, index, uint32_t{1}, fullMask);
    Reg::Scatter(output, b1Imag, index, fullMask);
}

__simd_callee__ inline void StockhamRadix2LazyStoreContiguous256(
    __ubuf__ float* output, AscendC::Reg::RegTensor<float>& a0Real, AscendC::Reg::RegTensor<float>& a0Imag,
    AscendC::Reg::RegTensor<float>& b0Real, AscendC::Reg::RegTensor<float>& b0Imag,
    AscendC::Reg::RegTensor<float>& a1Real, AscendC::Reg::RegTensor<float>& a1Imag,
    AscendC::Reg::RegTensor<float>& b1Real, AscendC::Reg::RegTensor<float>& b1Imag,
    AscendC::Reg::RegTensor<float>& naturalReal, AscendC::Reg::RegTensor<float>& naturalImag,
    AscendC::Reg::RegTensor<uint32_t>& bitReverse6, AscendC::Reg::MaskReg& fullMask)
{
    using namespace AscendC;
    Reg::Gather(naturalReal, a0Real, bitReverse6);
    Reg::Gather(naturalImag, a0Imag, bitReverse6);
    Reg::StoreAlign<float, Reg::StoreDist::DIST_INTLV_B32>(output, naturalReal, naturalImag, fullMask);
    Reg::Gather(naturalReal, a1Real, bitReverse6);
    Reg::Gather(naturalImag, a1Imag, bitReverse6);
    Reg::StoreAlign<float, Reg::StoreDist::DIST_INTLV_B32>(output + 128U, naturalReal, naturalImag, fullMask);
    Reg::Gather(naturalReal, b0Real, bitReverse6);
    Reg::Gather(naturalImag, b0Imag, bitReverse6);
    Reg::StoreAlign<float, Reg::StoreDist::DIST_INTLV_B32>(output + 256U, naturalReal, naturalImag, fullMask);
    Reg::Gather(naturalReal, b1Real, bitReverse6);
    Reg::Gather(naturalImag, b1Imag, bitReverse6);
    Reg::StoreAlign<float, Reg::StoreDist::DIST_INTLV_B32>(output + 384U, naturalReal, naturalImag, fullMask);
}

template <bool ContiguousStore>
__no_simd_vf_fusion__ __simd_vf__ inline void StockhamRadix2FusedRegisterLazy256DirectVf(__ubuf__ float* buffer0,
                                                                                         __ubuf__ float* twiddles,
                                                                                         __ubuf__ float* buffer1,
                                                                                         uint16_t tileBatch)
{
    using namespace AscendC;
    Reg::RegTensor<float> r0Real, r0Imag, r1Real, r1Imag;
    Reg::RegTensor<float> r2Real, r2Imag, r3Real, r3Imag;
    Reg::RegTensor<float> wReal, wImag, tReal, tImag, temporary;
    Reg::RegTensor<uint32_t> j, index;
    Reg::MaskReg fullMask = Reg::CreateMask<float, Reg::MaskPattern::ALL>();

#pragma unroll 1
    for (uint16_t transform = 0; transform < tileBatch; ++transform) {
        const uint32_t transformFloatBase = static_cast<uint32_t>(transform) * 512U;
        __ubuf__ float* inputTransform = buffer0 + transformFloatBase;
        __ubuf__ float* outputTransform = buffer1 + transformFloatBase;

        // Initial L0=[U0,U1,V0,V1], each entry one full 64-complex chunk.
        Reg::LoadAlign<float, Reg::LoadDist::DIST_DINTLV_B32>(r0Real, r0Imag, inputTransform);
        Reg::LoadAlign<float, Reg::LoadDist::DIST_DINTLV_B32>(r1Real, r1Imag, inputTransform + 128U);
        Reg::LoadAlign<float, Reg::LoadDist::DIST_DINTLV_B32>(r2Real, r2Imag, inputTransform + 256U);
        Reg::LoadAlign<float, Reg::LoadDist::DIST_DINTLV_B32>(r3Real, r3Imag, inputTransform + 384U);
        Reg::Duplicate(j, uint32_t{0}, fullMask);

        StockhamRadix2LazyInterleaveStage256<1U, true>(twiddles, r0Real, r0Imag, r1Real, r1Imag, r2Real, r2Imag, r3Real,
                                                       r3Imag, wReal, wImag, tReal, tImag, temporary, j, index,
                                                       fullMask);

        // Keep h={2,4,8,16,32} in one Aux-Scalar loop.  The physical layout
        // alternates L1/L0; the explicit stage parity identifies the input
        // layout (h=2/8/32 consume L1, h=4/16 consume L0).  As
        // for N=128, preventing stage unrolling is essential to keep Bisheng
        // from extending SSA live ranges beyond the physical register file.
#pragma unroll 1
        for (uint32_t half = 2U, stage = 1U; half <= 32U; half <<= 1U, ++stage) {
            Reg::Adds(index, j, half - 1U, fullMask);
            Reg::Muls(index, index, uint32_t{2}, fullMask);
            Reg::Gather(wReal, twiddles, index, fullMask);
            Reg::Adds(index, index, uint32_t{1}, fullMask);
            Reg::Gather(wImag, twiddles, index, fullMask);

            if ((stage & uint32_t{1}) != 0U) {
                // L1=[U0,V0,U1,V1] -> L0.
                StockhamRadix2LazyButterfly<false>(r0Real, r0Imag, r1Real, r1Imag, wReal, wImag, tReal, tImag,
                                                   temporary, fullMask);
                StockhamRadix2LazyButterfly<false>(r2Real, r2Imag, r3Real, r3Imag, wReal, wImag, tReal, tImag,
                                                   temporary, fullMask);
                Reg::Interleave(r0Real, r1Real, r0Real, r1Real);
                Reg::Interleave(r0Imag, r1Imag, r0Imag, r1Imag);
                Reg::Interleave(r2Real, r3Real, r2Real, r3Real);
                Reg::Interleave(r2Imag, r3Imag, r2Imag, r3Imag);
            } else {
                // L0=[U0,U1,V0,V1] -> L1.
                StockhamRadix2LazyButterfly<false>(r0Real, r0Imag, r2Real, r2Imag, wReal, wImag, tReal, tImag,
                                                   temporary, fullMask);
                StockhamRadix2LazyButterfly<false>(r1Real, r1Imag, r3Real, r3Imag, wReal, wImag, tReal, tImag,
                                                   temporary, fullMask);
                Reg::Interleave(r0Real, r2Real, r0Real, r2Real);
                Reg::Interleave(r0Imag, r2Imag, r0Imag, r2Imag);
                Reg::Interleave(r1Real, r3Real, r1Real, r3Real);
                Reg::Interleave(r1Imag, r3Imag, r1Imag, r3Imag);
            }
            Reg::Adds(index, j, half, fullMask);
            Reg::Interleave(j, index, j, index);
        }

        // half=64 consumes L0 pairs (0,2)/(1,3).  Its Stockham repack is a
        // free reinterpretation of [A0,A1,B0,B1] as L1=[U0,V0,U1,V1].
        StockhamRadix2LazyLoadTwiddle<64U>(twiddles, j, index, wReal, wImag, fullMask);
        StockhamRadix2LazyButterfly<false>(r0Real, r0Imag, r2Real, r2Imag, wReal, wImag, tReal, tImag, temporary,
                                           fullMask);
        StockhamRadix2LazyButterfly<false>(r1Real, r1Imag, r3Real, r3Imag, wReal, wImag, tReal, tImag, temporary,
                                           fullMask);

        // Final half=128 consumes L1 pairs.  Chunk one needs j+64 in the
        // compact stage segment.  The final register meanings are
        // [A0,B0,A1,B1].
        StockhamRadix2LazyLoadTwiddle<128U, 0U>(twiddles, j, index, wReal, wImag, fullMask);
        StockhamRadix2LazyButterfly<false>(r0Real, r0Imag, r1Real, r1Imag, wReal, wImag, tReal, tImag, temporary,
                                           fullMask);
        StockhamRadix2LazyLoadTwiddle<128U, 64U>(twiddles, j, index, wReal, wImag, fullMask);
        StockhamRadix2LazyButterfly<false>(r2Real, r2Imag, r3Real, r3Imag, wReal, wImag, tReal, tImag, temporary,
                                           fullMask);
        if constexpr (ContiguousStore) {
            StockhamRadix2LazyStoreContiguous256(outputTransform, r0Real, r0Imag, r1Real, r1Imag, r2Real, r2Imag,
                                                 r3Real, r3Imag, wReal, wImag, j, fullMask);
        } else {
            StockhamRadix2LazyStore256(outputTransform, r0Real, r0Imag, r1Real, r1Imag, r2Real, r2Imag, r3Real, r3Imag,
                                       j, index, fullMask);
        }
    }
}

template <int32_t N>
__aicore__ inline void RunCompute(__ubuf__ float* buffer0, __ubuf__ float* twiddles, __ubuf__ float* buffer1,
                                  uint16_t tileBatch)
{
    if constexpr (N == 128) {
        StockhamRadix2FusedRegisterLazy128DirectVf<false>(buffer0, twiddles, buffer1, tileBatch);
    } else if constexpr (N == 256) {
        StockhamRadix2FusedRegisterLazy256DirectVf<false>(buffer0, twiddles, buffer1, tileBatch);
    } else {
        StockhamRadix2FusedUnrolledDirectVf<N>(buffer0, twiddles, buffer1, tileBatch);
    }
}

template <int32_t N>
__aicore__ inline __ubuf__ float* FinalBuffer(__ubuf__ float* buffer0, __ubuf__ float* buffer1)
{
    if constexpr (N <= 256 || (FftTraits<N>::kStages & 1) != 0) {
        return buffer1;
    } else {
        return buffer0;
    }
}

struct CoreBatchRange {
    int64_t start;
    int64_t count;
};

__aicore__ inline CoreBatchRange GetCoreBatchRange(int64_t batch)
{
    const int64_t blockCount = static_cast<int64_t>(AscendC::GetBlockNum());
    const int64_t blockIndex = static_cast<int64_t>(AscendC::GetBlockIdx());
    const int64_t quotient = batch / blockCount;
    const int64_t remainder = batch - quotient * blockCount;

    CoreBatchRange range;
    range.count = quotient + (blockIndex < remainder ? 1 : 0);
    range.start = blockIndex * quotient + (blockIndex < remainder ? blockIndex : remainder);
    return range;
}

template <int32_t N, int32_t TilePoints>
__aicore__ inline void RunSingleBuffered(__gm__ float* input, __gm__ float* twiddles, __gm__ float* output,
                                         int64_t batch)
{
    static_assert(N >= 16 && N <= 8192 && (N & (N - 1)) == 0 && TilePoints >= N && TilePoints % N == 0);
    constexpr bool registerResident = N <= 256;
    constexpr int32_t tileBatchCapacity = TilePoints / N;
    constexpr int32_t dataFloats = TilePoints * 2 + (registerResident ? 0 : kComplexLoadFloats);
    constexpr int32_t dataBufferCount = registerResident ? 1 : 2;
    static_assert((dataFloats * static_cast<int32_t>(sizeof(float))) % 32 == 0,
                  "consecutive UB data allocations must stay 32-B aligned");

    // Register-resident compute gathers one complete group before writing it,
    // so its input and final output safely alias one UB buffer and it needs no
    // unconditional-LoadAlign guard. Other variants retain two guarded
    // Stockham ping/pong buffers.
    __ubuf__ float dataStorage[dataBufferCount * dataFloats];
    __ubuf__ float* buffer0 = dataStorage;
    __ubuf__ float* buffer1 = registerResident ? dataStorage : dataStorage + dataFloats;
    __ubuf__ float twiddleLocal[N * 2];

    const CoreBatchRange range = GetCoreBatchRange(batch);
    if (range.count == 0) {
        return;
    }

    asc_copy_gm2ub_align(twiddleLocal, twiddles, 1, static_cast<uint32_t>(N * kComplexBytes), 0, 0, false, 0, 0, 0);

    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    for (int64_t localStart = 0; localStart < range.count; localStart += tileBatchCapacity) {
        const int64_t remaining = range.count - localStart;
        const uint16_t tileBatch = static_cast<uint16_t>(remaining < tileBatchCapacity ? remaining : tileBatchCapacity);
        const int64_t globalBatch = range.start + localStart;
        const int64_t globalFloatOffset = globalBatch * N * 2;
        const uint32_t tileBytes = static_cast<uint32_t>(tileBatch) * N * kComplexBytes;
        __gm__ float* inputTile = input + globalFloatOffset;
        __gm__ float* outputTile = output + globalFloatOffset;

        // MTE3->MTE2 protects the reused UB slot. MTE2->V makes the input and
        // the earlier twiddle transfer visible to Vector. V->MTE3 protects the
        // final result before its asynchronous copy to GM.
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        asc_copy_gm2ub_align(buffer0, inputTile, 1, tileBytes, 0, 0, false, 0, 0, 0);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

        RunCompute<N>(buffer0, twiddleLocal, buffer1, tileBatch);

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        __ubuf__ float* finalBuffer = FinalBuffer<N>(buffer0, buffer1);
        asc_copy_ub2gm_align(outputTile, finalBuffer, 1, tileBytes, 0, 0, 0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    }
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
}

// tile 0. Each loop iteration then queues the MTE2 copy for tile t+1 into the
// other slot before executing Vector work for tile t. MTE3_MTE2 protects slot
// reuse, MTE2_V publishes a prefetched input to Vector, and V_MTE3 publishes
// the final Stockham buffer to the output copy. The paired events are drained
// even when a core owns only one tile.
template <int32_t N, int32_t TilePoints>
__aicore__ inline void RunDoubleBufferedDirectPrefetch(__gm__ float* input, __gm__ float* twiddles,
                                                       __gm__ float* output, int64_t batch)
{
    static_assert(N >= 16 && N <= 4096 && (N & (N - 1)) == 0 && TilePoints >= N && TilePoints % N == 0);
    constexpr bool registerResident = N <= 256;
    constexpr int32_t tileBatchCapacity = TilePoints / N;
    constexpr int32_t dataFloats = TilePoints * 2 + (registerResident ? 0 : kComplexLoadFloats);
    constexpr int32_t dataBuffersPerSlot = registerResident ? 1 : 2;
    static_assert((dataFloats * static_cast<int32_t>(sizeof(float))) % 32 == 0,
                  "consecutive UB data allocations must stay 32-B aligned");

    __ubuf__ float pingStorage[dataBuffersPerSlot * dataFloats];
    __ubuf__ float pongStorage[dataBuffersPerSlot * dataFloats];
    __ubuf__ float* buffer0Ping = pingStorage;
    __ubuf__ float* buffer1Ping = registerResident ? pingStorage : pingStorage + dataFloats;
    __ubuf__ float* buffer0Pong = pongStorage;
    __ubuf__ float* buffer1Pong = registerResident ? pongStorage : pongStorage + dataFloats;
    __ubuf__ float twiddleLocal[N * 2];

    const CoreBatchRange range = GetCoreBatchRange(batch);
    if (range.count == 0) {
        return;
    }

    asc_copy_gm2ub_align(twiddleLocal, twiddles, 1, static_cast<uint32_t>(N * kComplexBytes), 0, 0, false, 0, 0, 0);

    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);

    // Prologue: load tile 0 into ping and publish it to Vector.
    const uint16_t firstTileBatch = static_cast<uint16_t>(range.count < tileBatchCapacity ? range.count :
                                                                                            tileBatchCapacity);
    const uint32_t firstTileBytes = static_cast<uint32_t>(firstTileBatch) * N * kComplexBytes;
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    asc_copy_gm2ub_align(buffer0Ping, input + range.start * N * 2, 1, firstTileBytes, 0, 0, false, 0, 0, 0);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

    int64_t tileIndex = 0;
    for (int64_t localStart = 0; localStart < range.count; localStart += tileBatchCapacity, ++tileIndex) {
        const int64_t remaining = range.count - localStart;
        const uint16_t tileBatch = static_cast<uint16_t>(remaining < tileBatchCapacity ? remaining : tileBatchCapacity);
        const int64_t globalBatch = range.start + localStart;
        const uint32_t tileBytes = static_cast<uint32_t>(tileBatch) * N * kComplexBytes;
        const int32_t eventId = ((tileIndex & 1) == 0) ? EVENT_ID0 : EVENT_ID1;
        __ubuf__ float* buffer0 = ((tileIndex & 1) == 0) ? buffer0Ping : buffer0Pong;
        __ubuf__ float* buffer1 = ((tileIndex & 1) == 0) ? buffer1Ping : buffer1Pong;

        const int64_t nextLocalStart = localStart + tileBatchCapacity;
        if (nextLocalStart < range.count) {
            const int64_t nextRemaining = range.count - nextLocalStart;
            const uint16_t nextTileBatch = static_cast<uint16_t>(nextRemaining < tileBatchCapacity ? nextRemaining :
                                                                                                     tileBatchCapacity);
            const uint32_t nextTileBytes = static_cast<uint32_t>(nextTileBatch) * N * kComplexBytes;
            const int64_t nextTileIndex = tileIndex + 1;
            const int32_t nextEventId = ((nextTileIndex & 1) == 0) ? EVENT_ID0 : EVENT_ID1;
            __ubuf__ float* nextBuffer0 = ((nextTileIndex & 1) == 0) ? buffer0Ping : buffer0Pong;
            const int64_t nextGlobalBatch = range.start + nextLocalStart;

            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(nextEventId);
            asc_copy_gm2ub_align(nextBuffer0, input + nextGlobalBatch * N * 2, 1, nextTileBytes, 0, 0, false, 0, 0, 0);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(nextEventId);
        }

        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventId);
        RunCompute<N>(buffer0, twiddleLocal, buffer1, tileBatch);

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventId);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventId);
        __ubuf__ float* finalBuffer = FinalBuffer<N>(buffer0, buffer1);
        asc_copy_ub2gm_align(output + globalBatch * N * 2, finalBuffer, 1, tileBytes, 0, 0, 0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
    }

    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
}

template <int32_t N, int32_t TilePoints>
__aicore__ inline int64_t LoadBatch(__gm__ uint8_t* tiling)
{
    const auto* data = reinterpret_cast<__gm__ AsdSip::FftC2CRegBaseTilingData*>(tiling);
    if (data->n != N || data->tilePoints != TilePoints || data->batch <= 0) {
        // A host/specialization ABI mismatch must not make an AIV address GM
        // using the wrong transform or tile size. The host accuracy sentinel
        // turns this fail-closed return into a visible correctness failure.
        return 0;
    }
    return data->batch;
}

template <int32_t N, int32_t TilePoints, bool DoubleBuffer>
__aicore__ inline void RunRegBase(__gm__ float* input, __gm__ float* coeff, __gm__ float* output,
                                  __gm__ uint8_t* tiling)
{
    AscendC::InitSocState();
    const int64_t batch = LoadBatch<N, TilePoints>(tiling);
    if (batch <= 0) {
        return;
    }
    if constexpr (DoubleBuffer) {
        RunDoubleBufferedDirectPrefetch<N, TilePoints>(input, coeff, output, batch);
    } else {
        RunSingleBuffered<N, TilePoints>(input, coeff, output, batch);
    }
    AscendC::PipeBarrier<PIPE_ALL>();
}
} // namespace
#endif
