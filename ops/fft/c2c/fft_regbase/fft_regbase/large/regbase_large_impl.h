/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "acl/acl.h"
#include "c_api/asc_simd.h"
#include "kernel_operator.h"

#include "regbase_large.h"

#include <cstdint>

namespace {

// First scalable power-of-two candidate.  The natural-order N=16384 signal is
// viewed as x[n0][n1], where n=n0*128+n1.  Phase A transforms n0 and writes
// T[k0][n1]; phase B transforms n1 and writes X[k1][k0] in natural order.
constexpr int32_t kLargeN = 16384;
constexpr int32_t kFactor = 128;
constexpr int32_t kComplexBytes = 2 * static_cast<int32_t>(sizeof(float));
constexpr int32_t kFft128Floats = 2 * kFactor;
constexpr int32_t kPhysicalTwiddleStages = 6;
constexpr int32_t kPhysicalTwiddleStageFloats = 128;
constexpr int32_t kPhysicalTwiddleFloats = kPhysicalTwiddleStages * kPhysicalTwiddleStageFloats;
constexpr uint8_t kL2CacheNormal = 1;

struct CoreTaskRange {
    int64_t start;
    int64_t count;
};

__aicore__ inline CoreTaskRange GetCoreTaskRange(int64_t taskCount)
{
    const int64_t blockCount = static_cast<int64_t>(AscendC::GetBlockNum());
    const int64_t blockIndex = static_cast<int64_t>(AscendC::GetBlockIdx());
    const int64_t quotient = taskCount / blockCount;
    const int64_t remainder = taskCount - quotient * blockCount;

    CoreTaskRange range;
    range.count = quotient + (blockIndex < remainder ? 1 : 0);
    range.start = blockIndex * quotient + (blockIndex < remainder ? blockIndex : remainder);
    return range;
}

// Build the final lazy Stockham lane map once per VF task.  The physical-
// twiddle FFT no longer needs j for coefficient lookup, so all transforms in
// one movement group can share this read-only bit-reverse-6 register.
__simd_callee__ inline void BuildFft128FinalLaneMap(AscendC::Reg::RegTensor<uint32_t>& j,
                                                    AscendC::Reg::RegTensor<uint32_t>& index,
                                                    AscendC::Reg::MaskReg& fullMask)
{
    using namespace AscendC;

    Reg::Duplicate(j, uint32_t{0}, fullMask);
    uint32_t half = 1U;
#pragma unroll 1
    for (uint16_t bit = 0; bit < 6; ++bit) {
        Reg::Adds(index, j, half, fullMask);
        Reg::Interleave(j, index, j, index);
        half <<= 1U;
    }
}

// FFT128 arithmetic with coefficients pre-expanded into the physical lane
// order produced by the lazy Stockham Interleave sequence.  Each real or
// imaginary coefficient plane is exactly one aligned 64-lane FP32 load.
__simd_callee__ inline void RunFft128RegistersPhysicalTwiddle(
    __ubuf__ float* twiddles, AscendC::Reg::RegTensor<float>& uReal, AscendC::Reg::RegTensor<float>& uImag,
    AscendC::Reg::RegTensor<float>& vReal, AscendC::Reg::RegTensor<float>& vImag, AscendC::Reg::RegTensor<float>& work0,
    AscendC::Reg::RegTensor<float>& work1, AscendC::Reg::RegTensor<float>& work2, AscendC::Reg::MaskReg& fullMask)
{
    using namespace AscendC;

    // Stockham half=1 has a unit twiddle and remains explicit.
    Reg::Sub(work0, uReal, vReal, fullMask);
    Reg::Add(work1, uReal, vReal, fullMask);
    Reg::Interleave(uReal, vReal, work1, work0);
    Reg::Sub(work0, uImag, vImag, fullMask);
    Reg::Add(work1, uImag, vImag, fullMask);
    Reg::Interleave(uImag, vImag, work1, work0);

    // Physical stages 0..4 are Stockham half=2..32.  The uint16 induction,
    // zero start, unit step, and branch-free body are the documented VF
    // hardware-loop shape.  Keep unroll(1) to bound register live ranges.
#pragma unroll 1
    for (uint16_t stage = 0; stage < 5; ++stage) {
        __ubuf__ float* stageTwiddles = twiddles + static_cast<uint32_t>(stage) *
                                                       static_cast<uint32_t>(kPhysicalTwiddleStageFloats);
        Reg::LoadAlign(work0, stageTwiddles);
        Reg::Mul(work1, vReal, work0, fullMask);
        Reg::Mul(work2, vImag, work0, fullMask);
        Reg::LoadAlign(work0, stageTwiddles + 64U);
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
    }

    // Physical stage 5 is Stockham half=64.  As in the compact control, the
    // final outputs are not interleaved again.
    __ubuf__ float* finalTwiddles = twiddles + 5U * static_cast<uint32_t>(kPhysicalTwiddleStageFloats);
    Reg::LoadAlign(work0, finalTwiddles);
    Reg::Mul(work1, vReal, work0, fullMask);
    Reg::Mul(work2, vImag, work0, fullMask);
    Reg::LoadAlign(work0, finalTwiddles + 64U);
    Reg::Mul(vReal, vReal, work0, fullMask);
    Reg::Mul(vImag, vImag, work0, fullMask);
    Reg::Sub(work1, work1, vImag, fullMask);
    Reg::Add(work2, vReal, work2, fullMask);
    Reg::Sub(vReal, uReal, work1, fullMask);
    Reg::Sub(vImag, uImag, work2, fullMask);
    Reg::Add(uReal, uReal, work1, fullMask);
    Reg::Add(uImag, uImag, work2, fullMask);
}

// Multiply one physical FFT128 half by a natural-order cross-twiddle plane.
// naturalOffset is 0 for u and 64 for v.  j maps physical lanes to natural k0.
__simd_callee__ inline void MultiplyCrossTwiddle(
    __ubuf__ float* crossTwiddles, AscendC::Reg::RegTensor<float>& real, AscendC::Reg::RegTensor<float>& imag,
    AscendC::Reg::RegTensor<float>& work0, AscendC::Reg::RegTensor<float>& work1, AscendC::Reg::RegTensor<float>& work2,
    AscendC::Reg::RegTensor<uint32_t>& j, AscendC::Reg::RegTensor<uint32_t>& index, uint32_t naturalOffset,
    AscendC::Reg::MaskReg& fullMask)
{
    using namespace AscendC;

    Reg::Adds(index, j, naturalOffset, fullMask);
    Reg::Muls(index, index, uint32_t{2}, fullMask);
    Reg::Gather(work0, crossTwiddles, index, fullMask);
    Reg::Mul(work1, real, work0, fullMask); // ac
    Reg::Mul(work2, imag, work0, fullMask); // bc
    Reg::Adds(index, index, uint32_t{1}, fullMask);
    Reg::Gather(work0, crossTwiddles, index, fullMask);
    Reg::Mul(real, real, work0, fullMask); // ad
    Reg::Mul(imag, imag, work0, fullMask); // bd
    Reg::Sub(work1, work1, imag, fullMask);
    Reg::Add(work2, real, work2, fullMask);
    real = work1;
    imag = work2;
}
// FFT128 columns with physical-order coefficients, an amortized final
// lane map, and fused cross-twiddle multiplication.
template <int32_t Group>
__no_simd_vf_fusion__ __simd_vf__ inline void PhaseAPhysicalTwiddleVf(__ubuf__ float* tile,
                                                                      __ubuf__ float* localTwiddles,
                                                                      __ubuf__ float* crossTwiddles)
{
    using namespace AscendC;
    static_assert(Group == 4 || Group == 8 || Group == 16);

    Reg::RegTensor<float> uReal, uImag, vReal, vImag;
    Reg::RegTensor<float> work0, work1, work2;
    Reg::RegTensor<uint32_t> j, index;
    Reg::MaskReg fullMask = Reg::CreateMask<float, Reg::MaskPattern::ALL>();

    BuildFft128FinalLaneMap(j, index, fullMask);

#pragma unroll 1
    for (uint16_t transform = 0; transform < Group; ++transform) {
        Reg::Arange<int32_t>(reinterpret_cast<Reg::RegTensor<int32_t>&>(index), 0);
        Reg::Muls(index, index, static_cast<uint32_t>(2 * Group), fullMask);
        Reg::Adds(index, index, static_cast<uint32_t>(2 * transform), fullMask);
        Reg::Gather(uReal, tile, index, fullMask);
        Reg::Adds(index, index, uint32_t{1}, fullMask);
        Reg::Gather(uImag, tile, index, fullMask);
        Reg::Adds(index, index, static_cast<uint32_t>(128 * Group - 1), fullMask);
        Reg::Gather(vReal, tile, index, fullMask);
        Reg::Adds(index, index, uint32_t{1}, fullMask);
        Reg::Gather(vImag, tile, index, fullMask);

        RunFft128RegistersPhysicalTwiddle(localTwiddles, uReal, uImag, vReal, vImag, work0, work1, work2, fullMask);

        __ubuf__ float* crossPlane = crossTwiddles +
                                     static_cast<uint32_t>(transform) * static_cast<uint32_t>(kFft128Floats);
        MultiplyCrossTwiddle(crossPlane, uReal, uImag, work0, work1, work2, j, index, 0U, fullMask);
        MultiplyCrossTwiddle(crossPlane, vReal, vImag, work0, work1, work2, j, index, 64U, fullMask);

        Reg::Muls(index, j, static_cast<uint32_t>(2 * Group), fullMask);
        Reg::Adds(index, index, static_cast<uint32_t>(2 * transform), fullMask);
        Reg::Scatter(tile, uReal, index, fullMask);
        Reg::Adds(index, index, uint32_t{1}, fullMask);
        Reg::Scatter(tile, uImag, index, fullMask);
        Reg::Adds(index, index, static_cast<uint32_t>(128 * Group - 1), fullMask);
        Reg::Scatter(tile, vReal, index, fullMask);
        Reg::Adds(index, index, uint32_t{1}, fullMask);
        Reg::Scatter(tile, vImag, index, fullMask);
    }
}

// FFT128 rows with DINTLV loads and natural-order transpose Scatter.
template <int32_t Group>
__no_simd_vf_fusion__ __simd_vf__ inline void PhaseBPhysicalTwiddleVf(__ubuf__ float* inputTile,
                                                                      __ubuf__ float* localTwiddles,
                                                                      __ubuf__ float* outputTile)
{
    using namespace AscendC;
    static_assert(Group == 4 || Group == 8 || Group == 16);

    Reg::RegTensor<float> uReal, uImag, vReal, vImag;
    Reg::RegTensor<float> work0, work1, work2;
    Reg::RegTensor<uint32_t> j, index;
    Reg::MaskReg fullMask = Reg::CreateMask<float, Reg::MaskPattern::ALL>();

    BuildFft128FinalLaneMap(j, index, fullMask);

#pragma unroll 1
    for (uint16_t transform = 0; transform < Group; ++transform) {
        __ubuf__ float* inputTransform = inputTile +
                                         static_cast<uint32_t>(transform) * static_cast<uint32_t>(kFft128Floats);
        Reg::LoadAlign<float, Reg::LoadDist::DIST_DINTLV_B32>(uReal, uImag, inputTransform);
        Reg::LoadAlign<float, Reg::LoadDist::DIST_DINTLV_B32>(vReal, vImag, inputTransform + 128U);

        RunFft128RegistersPhysicalTwiddle(localTwiddles, uReal, uImag, vReal, vImag, work0, work1, work2, fullMask);

        Reg::Muls(index, j, static_cast<uint32_t>(2 * Group), fullMask);
        Reg::Adds(index, index, static_cast<uint32_t>(2 * transform), fullMask);
        Reg::Scatter(outputTile, uReal, index, fullMask);
        Reg::Adds(index, index, uint32_t{1}, fullMask);
        Reg::Scatter(outputTile, uImag, index, fullMask);
        Reg::Adds(index, index, static_cast<uint32_t>(128 * Group - 1), fullMask);
        Reg::Scatter(outputTile, vReal, index, fullMask);
        Reg::Adds(index, index, uint32_t{1}, fullMask);
        Reg::Scatter(outputTile, vImag, index, fullMask);
    }
}

// FFT256: two FFT128s over even/odd input, then E[j] +/- W256^j O[j].
// Phase A gathers columns, applies the cross twiddle and scatters in-place;
// Phase B gathers rows and fuses the final transpose into its scatter.
template <int32_t Group, bool PhaseA = false>
__no_simd_vf_fusion__ __simd_vf__ inline void Fft256PhysicalVf(__ubuf__ float* inputTile, __ubuf__ float* twiddles,
                                                               __ubuf__ float* outputTile,
                                                               __ubuf__ float* crossTwiddles = nullptr)
{
    using namespace AscendC;
    Reg::RegTensor<float> e0r, e0i, e1r, e1i, o0r, o0i, o1r, o1i;
    Reg::RegTensor<float> w0, w1, w2;
    Reg::RegTensor<uint32_t> j, index;
    auto mask = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
    BuildFft128FinalLaneMap(j, index, mask);
#pragma unroll 1
    for (uint16_t transform = 0; transform < Group; ++transform) {
        constexpr uint32_t inputStride = PhaseA ? Group : 1;
        __ubuf__ float* input = inputTile + (PhaseA ? 2U : 512U) * transform;
        Reg::Arange<int32_t>(reinterpret_cast<Reg::RegTensor<int32_t>&>(index), 0);
        Reg::Muls(index, index, uint32_t{4 * inputStride}, mask);
        Reg::Gather(e0r, input, index, mask);
        Reg::Gather(e1r, input + 256 * inputStride, index, mask);
        Reg::Gather(o0r, input + 2 * inputStride, index, mask);
        Reg::Gather(o1r, input + 258 * inputStride, index, mask);
        Reg::Adds(index, index, uint32_t{1}, mask);
        Reg::Gather(e0i, input, index, mask);
        Reg::Gather(e1i, input + 256 * inputStride, index, mask);
        Reg::Gather(o0i, input + 2 * inputStride, index, mask);
        Reg::Gather(o1i, input + 258 * inputStride, index, mask);
        RunFft128RegistersPhysicalTwiddle(twiddles, e0r, e0i, e1r, e1i, w0, w1, w2, mask);
        RunFft128RegistersPhysicalTwiddle(twiddles, o0r, o0i, o1r, o1i, w0, w1, w2, mask);
        // Final Stockham butterfly, first physical 64-lane chunk.
        Reg::LoadAlign(w0, twiddles + 768);
        Reg::Mul(w1, o0r, w0, mask);
        Reg::Mul(w2, o0i, w0, mask);
        Reg::LoadAlign(w0, twiddles + 832);
        Reg::Mul(o0r, o0r, w0, mask);
        Reg::Mul(o0i, o0i, w0, mask);
        Reg::Sub(w1, w1, o0i, mask);
        Reg::Add(w2, w2, o0r, mask);
        Reg::Sub(o0r, e0r, w1, mask);
        Reg::Sub(o0i, e0i, w2, mask);
        Reg::Add(e0r, e0r, w1, mask);
        Reg::Add(e0i, e0i, w2, mask);
        Reg::LoadAlign(w0, twiddles + 896);
        Reg::Mul(w1, o1r, w0, mask);
        Reg::Mul(w2, o1i, w0, mask);
        Reg::LoadAlign(w0, twiddles + 960);
        Reg::Mul(o1r, o1r, w0, mask);
        Reg::Mul(o1i, o1i, w0, mask);
        Reg::Sub(w1, w1, o1i, mask);
        Reg::Add(w2, w2, o1r, mask);
        Reg::Sub(o1r, e1r, w1, mask);
        Reg::Sub(o1i, e1i, w2, mask);
        Reg::Add(e1r, e1r, w1, mask);
        Reg::Add(e1i, e1i, w2, mask);
        if constexpr (!PhaseA) {
            // Preserve the validated Phase-B scatter order and shared index.
            Reg::Muls(index, j, uint32_t{2 * Group}, mask);
            Reg::Adds(index, index, uint32_t{2} * transform, mask);
            Reg::Scatter(outputTile, e0r, index, mask);
            Reg::Scatter(outputTile + 128 * Group, e1r, index, mask);
            Reg::Scatter(outputTile + 256 * Group, o0r, index, mask);
            Reg::Scatter(outputTile + 384 * Group, o1r, index, mask);
            Reg::Adds(index, index, uint32_t{1}, mask);
            Reg::Scatter(outputTile, e0i, index, mask);
            Reg::Scatter(outputTile + 128 * Group, e1i, index, mask);
            Reg::Scatter(outputTile + 256 * Group, o0i, index, mask);
            Reg::Scatter(outputTile + 384 * Group, o1i, index, mask);
        } else {
            // Store each finished chunk immediately to shorten cross-product lifetimes.
            MultiplyCrossTwiddle(crossTwiddles + 512U * transform, e0r, e0i, w0, w1, w2, j, index, 0U, mask);
            Reg::Muls(index, j, uint32_t{2 * Group}, mask);
            Reg::Adds(index, index, uint32_t{2} * transform, mask);
            Reg::Scatter(outputTile, e0r, index, mask);
            Reg::Adds(index, index, uint32_t{1}, mask);
            Reg::Scatter(outputTile, e0i, index, mask);
            MultiplyCrossTwiddle(crossTwiddles + 512U * transform, e1r, e1i, w0, w1, w2, j, index, 64U, mask);
            Reg::Muls(index, j, uint32_t{2 * Group}, mask);
            Reg::Adds(index, index, uint32_t{2} * transform, mask);
            Reg::Scatter(outputTile + 128 * Group, e1r, index, mask);
            Reg::Adds(index, index, uint32_t{1}, mask);
            Reg::Scatter(outputTile + 128 * Group, e1i, index, mask);
            MultiplyCrossTwiddle(crossTwiddles + 512U * transform, o0r, o0i, w0, w1, w2, j, index, 128U, mask);
            Reg::Muls(index, j, uint32_t{2 * Group}, mask);
            Reg::Adds(index, index, uint32_t{2} * transform, mask);
            Reg::Scatter(outputTile + 256 * Group, o0r, index, mask);
            Reg::Adds(index, index, uint32_t{1}, mask);
            Reg::Scatter(outputTile + 256 * Group, o0i, index, mask);
            MultiplyCrossTwiddle(crossTwiddles + 512U * transform, o1r, o1i, w0, w1, w2, j, index, 192U, mask);
            Reg::Muls(index, j, uint32_t{2 * Group}, mask);
            Reg::Adds(index, index, uint32_t{2} * transform, mask);
            Reg::Scatter(outputTile + 384 * Group, o1r, index, mask);
            Reg::Adds(index, index, uint32_t{1}, mask);
            Reg::Scatter(outputTile + 384 * Group, o1i, index, mask);
        }
    }
}

// FFT256 arithmetic for FFT512; leave the measured FFT256 VF unchanged.
__simd_callee__ inline void RunFft256RegistersPhysicalTwiddle(
    __ubuf__ float* twiddles, AscendC::Reg::RegTensor<float>& e0r, AscendC::Reg::RegTensor<float>& e0i,
    AscendC::Reg::RegTensor<float>& e1r, AscendC::Reg::RegTensor<float>& e1i, AscendC::Reg::RegTensor<float>& o0r,
    AscendC::Reg::RegTensor<float>& o0i, AscendC::Reg::RegTensor<float>& o1r, AscendC::Reg::RegTensor<float>& o1i,
    AscendC::Reg::RegTensor<float>& w0, AscendC::Reg::RegTensor<float>& w1, AscendC::Reg::RegTensor<float>& w2,
    AscendC::Reg::MaskReg& mask)
{
    using namespace AscendC;
    RunFft128RegistersPhysicalTwiddle(twiddles, e0r, e0i, e1r, e1i, w0, w1, w2, mask);
    RunFft128RegistersPhysicalTwiddle(twiddles, o0r, o0i, o1r, o1i, w0, w1, w2, mask);
    // Final Stockham butterfly, first physical 64-lane chunk.
    Reg::LoadAlign(w0, twiddles + 768);
    Reg::Mul(w1, o0r, w0, mask);
    Reg::Mul(w2, o0i, w0, mask);
    Reg::LoadAlign(w0, twiddles + 832);
    Reg::Mul(o0r, o0r, w0, mask);
    Reg::Mul(o0i, o0i, w0, mask);
    Reg::Sub(w1, w1, o0i, mask);
    Reg::Add(w2, w2, o0r, mask);
    Reg::Sub(o0r, e0r, w1, mask);
    Reg::Sub(o0i, e0i, w2, mask);
    Reg::Add(e0r, e0r, w1, mask);
    Reg::Add(e0i, e0i, w2, mask);
    Reg::LoadAlign(w0, twiddles + 896);
    Reg::Mul(w1, o1r, w0, mask);
    Reg::Mul(w2, o1i, w0, mask);
    Reg::LoadAlign(w0, twiddles + 960);
    Reg::Mul(o1r, o1r, w0, mask);
    Reg::Mul(o1i, o1i, w0, mask);
    Reg::Sub(w1, w1, o1i, mask);
    Reg::Add(w2, w2, o1r, mask);
    Reg::Sub(o1r, e1r, w1, mask);
    Reg::Sub(o1i, e1i, w2, mask);
    Reg::Add(e1r, e1r, w1, mask);
    Reg::Add(e1i, e1i, w2, mask);
}

// FFT512 rows: FFT256(even), FFT256(odd), then E[j] +/- W512^j O[j].
// Store each completed pair immediately to bound register live ranges.
template <int32_t Group>
__no_simd_vf_fusion__ __simd_vf__ inline void Fft512PhysicalVf(__ubuf__ float* inputTile, __ubuf__ float* twiddles,
                                                               __ubuf__ float* outputTile)
{
    using namespace AscendC;
    Reg::RegTensor<float> e0r, e0i, e1r, e1i, e2r, e2i, e3r, e3i;
    Reg::RegTensor<float> o0r, o0i, o1r, o1i, o2r, o2i, o3r, o3i;
    Reg::RegTensor<float> w0, w1, w2;
    Reg::RegTensor<uint32_t> j, index;
    auto mask = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
#pragma unroll 1
    for (uint16_t transform = 0; transform < Group; ++transform) {
        __ubuf__ float* input = inputTile + 1024U * transform;
        // Keep every Gather base 32-byte aligned; offsets belong in index.
        Reg::Arange<int32_t>(reinterpret_cast<Reg::RegTensor<int32_t>&>(index), 0);
        Reg::Muls(index, index, uint32_t{8}, mask);
        Reg::Gather(e0r, input, index, mask);
        Reg::Gather(e1r, input + 512U, index, mask);
        Reg::Adds(index, index, uint32_t{4}, mask);
        Reg::Gather(e2r, input, index, mask);
        Reg::Gather(e3r, input + 512U, index, mask);
        Reg::Adds(index, index, uint32_t{1}, mask);
        Reg::Gather(e2i, input, index, mask);
        Reg::Gather(e3i, input + 512U, index, mask);
        Reg::Arange<int32_t>(reinterpret_cast<Reg::RegTensor<int32_t>&>(index), 0);
        Reg::Muls(index, index, uint32_t{8}, mask);
        Reg::Adds(index, index, uint32_t{1}, mask);
        Reg::Gather(e0i, input, index, mask);
        Reg::Gather(e1i, input + 512U, index, mask);
        RunFft256RegistersPhysicalTwiddle(twiddles, e0r, e0i, e1r, e1i, e2r, e2i, e3r, e3i, w0, w1, w2, mask);
        // Park the final even chunk in its private output rows, in physical
        // lane order. It is reloaded before those rows receive final output.
        // This bounded 512-byte temporary replaces implicit compiler spilling.
        Reg::Arange<int32_t>(reinterpret_cast<Reg::RegTensor<int32_t>&>(index), 0);
        Reg::Muls(index, index, uint32_t{2 * Group}, mask);
        Reg::Adds(index, index, uint32_t{2} * transform, mask);
        Reg::Scatter(outputTile + 384 * Group, e3r, index, mask);
        Reg::Adds(index, index, uint32_t{1}, mask);
        Reg::Scatter(outputTile + 384 * Group, e3i, index, mask);
        // Keep every Gather base 32-byte aligned; offsets belong in index.
        Reg::Arange<int32_t>(reinterpret_cast<Reg::RegTensor<int32_t>&>(index), 0);
        Reg::Muls(index, index, uint32_t{8}, mask);
        Reg::Adds(index, index, uint32_t{2}, mask);
        Reg::Gather(o0r, input, index, mask);
        Reg::Gather(o1r, input + 512U, index, mask);
        Reg::Adds(index, index, uint32_t{4}, mask);
        Reg::Gather(o2r, input, index, mask);
        Reg::Gather(o3r, input + 512U, index, mask);
        Reg::Adds(index, index, uint32_t{1}, mask);
        Reg::Gather(o2i, input, index, mask);
        Reg::Gather(o3i, input + 512U, index, mask);
        Reg::Arange<int32_t>(reinterpret_cast<Reg::RegTensor<int32_t>&>(index), 0);
        Reg::Muls(index, index, uint32_t{8}, mask);
        Reg::Adds(index, index, uint32_t{3}, mask);
        Reg::Gather(o0i, input, index, mask);
        Reg::Gather(o1i, input + 512U, index, mask);
        RunFft256RegistersPhysicalTwiddle(twiddles, o0r, o0i, o1r, o1i, o2r, o2i, o3r, o3i, w0, w1, w2, mask);
        BuildFft128FinalLaneMap(j, index, mask);
        // Natural j + 0, paired with j + 256.
        Reg::LoadAlign(w0, twiddles + 1024);
        Reg::Mul(w1, o0r, w0, mask);
        Reg::Mul(w2, o0i, w0, mask);
        Reg::LoadAlign(w0, twiddles + 1088);
        Reg::Mul(o0r, o0r, w0, mask);
        Reg::Mul(o0i, o0i, w0, mask);
        Reg::Sub(w1, w1, o0i, mask);
        Reg::Add(w2, w2, o0r, mask);
        Reg::Sub(o0r, e0r, w1, mask);
        Reg::Sub(o0i, e0i, w2, mask);
        Reg::Add(e0r, e0r, w1, mask);
        Reg::Add(e0i, e0i, w2, mask);
        Reg::Muls(index, j, uint32_t{2 * Group}, mask);
        Reg::Adds(index, index, uint32_t{2} * transform, mask);
        Reg::Scatter(outputTile + 0 * Group, e0r, index, mask);
        Reg::Scatter(outputTile + 512 * Group, o0r, index, mask);
        Reg::Adds(index, index, uint32_t{1}, mask);
        Reg::Scatter(outputTile + 0 * Group, e0i, index, mask);
        Reg::Scatter(outputTile + 512 * Group, o0i, index, mask);
        // Natural j + 64, paired with j + 320.
        Reg::LoadAlign(w0, twiddles + 1152);
        Reg::Mul(w1, o1r, w0, mask);
        Reg::Mul(w2, o1i, w0, mask);
        Reg::LoadAlign(w0, twiddles + 1216);
        Reg::Mul(o1r, o1r, w0, mask);
        Reg::Mul(o1i, o1i, w0, mask);
        Reg::Sub(w1, w1, o1i, mask);
        Reg::Add(w2, w2, o1r, mask);
        Reg::Sub(o1r, e1r, w1, mask);
        Reg::Sub(o1i, e1i, w2, mask);
        Reg::Add(e1r, e1r, w1, mask);
        Reg::Add(e1i, e1i, w2, mask);
        Reg::Muls(index, j, uint32_t{2 * Group}, mask);
        Reg::Adds(index, index, uint32_t{2} * transform, mask);
        Reg::Scatter(outputTile + 128 * Group, e1r, index, mask);
        Reg::Scatter(outputTile + 640 * Group, o1r, index, mask);
        Reg::Adds(index, index, uint32_t{1}, mask);
        Reg::Scatter(outputTile + 128 * Group, e1i, index, mask);
        Reg::Scatter(outputTile + 640 * Group, o1i, index, mask);
        // Natural j + 128, paired with j + 384.
        Reg::LoadAlign(w0, twiddles + 1280);
        Reg::Mul(w1, o2r, w0, mask);
        Reg::Mul(w2, o2i, w0, mask);
        Reg::LoadAlign(w0, twiddles + 1344);
        Reg::Mul(o2r, o2r, w0, mask);
        Reg::Mul(o2i, o2i, w0, mask);
        Reg::Sub(w1, w1, o2i, mask);
        Reg::Add(w2, w2, o2r, mask);
        Reg::Sub(o2r, e2r, w1, mask);
        Reg::Sub(o2i, e2i, w2, mask);
        Reg::Add(e2r, e2r, w1, mask);
        Reg::Add(e2i, e2i, w2, mask);
        Reg::Muls(index, j, uint32_t{2 * Group}, mask);
        Reg::Adds(index, index, uint32_t{2} * transform, mask);
        Reg::Scatter(outputTile + 256 * Group, e2r, index, mask);
        Reg::Scatter(outputTile + 768 * Group, o2r, index, mask);
        Reg::Adds(index, index, uint32_t{1}, mask);
        Reg::Scatter(outputTile + 256 * Group, e2i, index, mask);
        Reg::Scatter(outputTile + 768 * Group, o2i, index, mask);
        // The parked chunk creates a real UB store->load dependency. Do not
        // rely on the allocator reusing the same physical data registers.
        Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
        Reg::Arange<int32_t>(reinterpret_cast<Reg::RegTensor<int32_t>&>(index), 0);
        Reg::Muls(index, index, uint32_t{2 * Group}, mask);
        Reg::Adds(index, index, uint32_t{2} * transform, mask);
        Reg::Gather(e3r, outputTile + 384 * Group, index, mask);
        Reg::Adds(index, index, uint32_t{1}, mask);
        Reg::Gather(e3i, outputTile + 384 * Group, index, mask);
        Reg::LocalMemBar<Reg::MemType::VEC_LOAD, Reg::MemType::VEC_STORE>();
        // Natural j + 192, paired with j + 448.
        Reg::LoadAlign(w0, twiddles + 1408);
        Reg::Mul(w1, o3r, w0, mask);
        Reg::Mul(w2, o3i, w0, mask);
        Reg::LoadAlign(w0, twiddles + 1472);
        Reg::Mul(o3r, o3r, w0, mask);
        Reg::Mul(o3i, o3i, w0, mask);
        Reg::Sub(w1, w1, o3i, mask);
        Reg::Add(w2, w2, o3r, mask);
        Reg::Sub(o3r, e3r, w1, mask);
        Reg::Sub(o3i, e3i, w2, mask);
        Reg::Add(e3r, e3r, w1, mask);
        Reg::Add(e3i, e3i, w2, mask);
        Reg::Muls(index, j, uint32_t{2 * Group}, mask);
        Reg::Adds(index, index, uint32_t{2} * transform, mask);
        Reg::Scatter(outputTile + 384 * Group, e3r, index, mask);
        Reg::Scatter(outputTile + 896 * Group, o3r, index, mask);
        Reg::Adds(index, index, uint32_t{1}, mask);
        Reg::Scatter(outputTile + 384 * Group, e3i, index, mask);
        Reg::Scatter(outputTile + 896 * Group, o3i, index, mask);
    }
}

template <int32_t Group>
__aicore__ inline int64_t LoadTaskCount(__gm__ uint8_t* tiling)
{
    const auto* data = reinterpret_cast<__gm__ FftC2CRegBaseLargeTilingData*>(tiling);
    if (data->batch <= 0 || data->n != kLargeN || data->groupSize != Group) {
        return 0;
    }
    return data->batch * static_cast<int64_t>(kFactor / Group);
}

template <int32_t Q = 128, int32_t P = 128, int32_t Group = 4>
__aicore__ inline void RunPhaseADoubleBufferedPhysicalTwiddle(__gm__ float* input, __gm__ float* localTwiddles,
                                                              __gm__ float* crossTwiddles, __gm__ float* workspace,
                                                              int64_t taskCount)
{
    static_assert(P == 128 || P == 256);
    constexpr int32_t groupsPerBatch = Q / Group;
    constexpr uint32_t rowBytes = Group * kComplexBytes;
    constexpr uint32_t tileFloats = 2 * P * Group;
    constexpr uint32_t tileBytes = tileFloats * sizeof(float);

    __ubuf__ float tilePing[tileFloats];
    __ubuf__ float crossPing[tileFloats];
    __ubuf__ float tilePong[tileFloats];
    __ubuf__ float crossPong[tileFloats];
    __ubuf__ float twiddleLocal[P == 128 ? kPhysicalTwiddleFloats : 1024];

    const CoreTaskRange range = GetCoreTaskRange(taskCount);
    if (range.count == 0) {
        return;
    }

    asc_copy_gm2ub_align(twiddleLocal, localTwiddles, 1, static_cast<uint32_t>(sizeof(twiddleLocal)), 0, 0, false, 0, 0,
                         0);

    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);

    const int64_t firstTask = range.start;
    const int64_t firstBatch = firstTask / groupsPerBatch;
    const int32_t firstGroup = static_cast<int32_t>(firstTask - firstBatch * groupsPerBatch);
    const int32_t firstN1Base = firstGroup * Group;
    const int64_t firstBatchComplexBase = firstBatch * (P * Q);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    asc_copy_gm2ub_align(tilePing, input + 2 * (firstBatchComplexBase + firstN1Base), P, rowBytes, 0, 0, false, 0,
                         static_cast<uint64_t>(Q * kComplexBytes), rowBytes);
    asc_copy_gm2ub_align(crossPing, crossTwiddles + 2 * static_cast<int64_t>(firstN1Base) * P, 1, tileBytes, 0, 0,
                         false, kL2CacheNormal, 0, 0);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

    for (int64_t localTask = 0; localTask < range.count; ++localTask) {
        const bool usePing = (localTask & int64_t{1}) == 0;
        const int32_t eventId = usePing ? EVENT_ID0 : EVENT_ID1;
        __ubuf__ float* tile = usePing ? tilePing : tilePong;
        __ubuf__ float* crossLocal = usePing ? crossPing : crossPong;

        const int64_t nextLocalTask = localTask + 1;
        if (nextLocalTask < range.count) {
            const bool nextUsesPing = (nextLocalTask & int64_t{1}) == 0;
            const int32_t nextEventId = nextUsesPing ? EVENT_ID0 : EVENT_ID1;
            __ubuf__ float* nextTile = nextUsesPing ? tilePing : tilePong;
            __ubuf__ float* nextCross = nextUsesPing ? crossPing : crossPong;
            const int64_t nextTask = range.start + nextLocalTask;
            const int64_t nextBatch = nextTask / groupsPerBatch;
            const int32_t nextGroup = static_cast<int32_t>(nextTask - nextBatch * groupsPerBatch);
            const int32_t nextN1Base = nextGroup * Group;
            const int64_t nextBatchComplexBase = nextBatch * (P * Q);

            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(nextEventId);
            asc_copy_gm2ub_align(nextTile, input + 2 * (nextBatchComplexBase + nextN1Base), P, rowBytes, 0, 0, false, 0,
                                 static_cast<uint64_t>(Q * kComplexBytes), rowBytes);
            asc_copy_gm2ub_align(nextCross, crossTwiddles + 2 * static_cast<int64_t>(nextN1Base) * P, 1, tileBytes, 0,
                                 0, false, kL2CacheNormal, 0, 0);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(nextEventId);
        }

        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventId);
        if constexpr (P == 128) {
            PhaseAPhysicalTwiddleVf<Group>(tile, twiddleLocal, crossLocal);
        } else {
            Fft256PhysicalVf<Group, true>(tile, twiddleLocal, tile, crossLocal);
        }

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventId);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventId);

        const int64_t task = range.start + localTask;
        const int64_t batch = task / groupsPerBatch;
        const int32_t groupIndex = static_cast<int32_t>(task - batch * groupsPerBatch);
        const int32_t n1Base = groupIndex * Group;
        const int64_t batchComplexBase = batch * (P * Q);
        asc_copy_ub2gm_align(workspace + 2 * (batchComplexBase + n1Base), tile, P, rowBytes, 0,
                             static_cast<uint64_t>(Q * kComplexBytes), rowBytes);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
    }

    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
}

// Phase B keeps batch-major group tasks and uses independent input/output
// ping/pong slots. No cross-twiddle table or inter-core wait is needed here.
template <int32_t LocalLength = 128, int32_t Group = 16, int32_t P = 128>
__aicore__ inline void RunPhaseBDoubleBufferedPhysicalTwiddle(__gm__ float* workspace, __gm__ float* localTwiddles,
                                                              __gm__ float* output, int64_t taskCount)
{
    constexpr int32_t groupsPerBatch = P / Group;
    constexpr uint32_t rowBytes = Group * kComplexBytes;
    constexpr uint32_t tileFloats = 2 * LocalLength * Group;
    constexpr uint32_t tileBytes = tileFloats * sizeof(float);

    __ubuf__ float inputPing[tileFloats];
    __ubuf__ float outputPing[tileFloats];
    __ubuf__ float inputPong[tileFloats];
    __ubuf__ float outputPong[tileFloats];
    static_assert(LocalLength == 128 || LocalLength == 256 || LocalLength == 512);
    static_assert(LocalLength != 512 || Group == 4 || Group == 8);
    __ubuf__ float twiddleLocal[LocalLength == 128 ? 768 : (LocalLength == 256 ? 1024 : 1536)];

    const CoreTaskRange range = GetCoreTaskRange(taskCount);
    if (range.count == 0) {
        return;
    }

    asc_copy_gm2ub_align(twiddleLocal, localTwiddles, 1, static_cast<uint32_t>(sizeof(twiddleLocal)), 0, 0, false, 0, 0,
                         0);

    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);

    for (int64_t localTask = 0; localTask < range.count; ++localTask) {
        const bool usePing = (localTask & int64_t{1}) == 0;
        const int32_t eventId = usePing ? EVENT_ID0 : EVENT_ID1;
        __ubuf__ float* inputTile = usePing ? inputPing : inputPong;
        __ubuf__ float* outputTile = usePing ? outputPing : outputPong;

        const int64_t task = range.start + localTask;
        const int64_t batch = task / groupsPerBatch;
        const int32_t groupIndex = static_cast<int32_t>(task - batch * groupsPerBatch);
        const int32_t k0Base = groupIndex * Group;
        const int64_t batchComplexBase = batch * (P * LocalLength);

        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventId);
        asc_copy_gm2ub_align(inputTile, workspace + 2 * (batchComplexBase + static_cast<int64_t>(k0Base) * LocalLength),
                             1, tileBytes, 0, 0, false, 0, 0, 0);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventId);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventId);

        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventId);
        if constexpr (LocalLength == 128) {
            PhaseBPhysicalTwiddleVf<Group>(inputTile, twiddleLocal, outputTile);
        } else if constexpr (LocalLength == 256) {
            Fft256PhysicalVf<Group>(inputTile, twiddleLocal, outputTile);
        } else {
            Fft512PhysicalVf<Group>(inputTile, twiddleLocal, outputTile);
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventId);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventId);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventId);

        asc_copy_ub2gm_align(output + 2 * (batchComplexBase + k0Base), outputTile, LocalLength, rowBytes, 0,
                             static_cast<uint64_t>(P * kComplexBytes), rowBytes);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventId);
    }

    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
}

} // namespace
