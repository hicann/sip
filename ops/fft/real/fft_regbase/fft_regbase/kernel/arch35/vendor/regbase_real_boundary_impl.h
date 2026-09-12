/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "c_api/asc_simd.h"
#include "kernel_operator.h"

namespace {
template <AscendC::HardEvent Event>
__aicore__ inline void Fence()
{
    AscendC::SetFlag<Event>(EVENT_ID0);
    AscendC::WaitFlag<Event>(EVENT_ID0);
}

// Interior k in [1,M): a=Z[k]/X[k], b=conj(Z[M-k]/X[M-k]).
// B arrives in ascending GM order and is reversed by masked UB Gather.
// All indexes are float offsets from aligned private UB bases.
template <bool C2R>
__no_simd_vf_fusion__ __simd_vf__ inline void BoundaryVf(__ubuf__ float* a, __ubuf__ float* b, __ubuf__ float* tw,
                                                         __ubuf__ float* output, uint32_t count)
{
    using namespace AscendC;
    uint32_t remaining = count;
    const uint16_t repeats = (count + 63U) / 64U;
#pragma unroll 1
    for (uint16_t repeat = 0; repeat < repeats; ++repeat) {
        Reg::MaskReg mask = Reg::UpdateMask<float>(remaining);
        Reg::RegTensor<uint32_t> q, index, reverse, last;
        Reg::RegTensor<float> ar, ai, br, bi, wr, wi;
        Reg::RegTensor<float> er, ei, dr, di, yr, yi, temporary;
        Reg::Arange<int32_t>(reinterpret_cast<Reg::RegTensor<int32_t>&>(q), repeat * 64);
        Reg::Muls(index, q, uint32_t{2}, mask);
        Reg::Gather(ar, a, index, mask);
        Reg::Gather(wr, tw, index, mask);
        Reg::Adds(index, index, uint32_t{1}, mask);
        Reg::Gather(ai, a, index, mask);
        Reg::Gather(wi, tw, index, mask);
        Reg::Duplicate(last, count - 1U, mask);
        Reg::Sub(reverse, last, q, mask);
        Reg::Muls(reverse, reverse, uint32_t{2}, mask);
        Reg::Gather(br, b, reverse, mask);
        Reg::Adds(reverse, reverse, uint32_t{1}, mask);
        Reg::Gather(bi, b, reverse, mask);

        Reg::Add(er, ar, br, mask);
        Reg::Sub(ei, ai, bi, mask); // b is conjugated
        Reg::Sub(dr, ar, br, mask);
        Reg::Add(di, ai, bi, mask);
        if constexpr (C2R) {
            // C[k]=(a+b)+i*W_N[k]*(a-b). NO 1/2: FFT_M(C)
            // directly produces the unnormalized N-point real transform.
            Reg::Mul(temporary, wr, di, mask);
            Reg::Sub(yr, er, temporary, mask);
            Reg::Mul(temporary, wi, dr, mask);
            Reg::Sub(yr, yr, temporary, mask);
            Reg::Mul(temporary, wr, dr, mask);
            Reg::Add(yi, ei, temporary, mask);
            Reg::Mul(temporary, wi, di, mask);
            Reg::Sub(yi, yi, temporary, mask);
        } else {
            // X[k]=((a+b)-i*W_N[k]*(a-b))/2.
            Reg::Mul(temporary, wr, di, mask);
            Reg::Add(yr, er, temporary, mask);
            Reg::Mul(temporary, wi, dr, mask);
            Reg::Add(yr, yr, temporary, mask);
            Reg::Mul(temporary, wi, di, mask);
            Reg::Add(yi, ei, temporary, mask);
            Reg::Mul(temporary, wr, dr, mask);
            Reg::Sub(yi, yi, temporary, mask);
            Reg::Muls(yr, yr, 0.5f, mask);
            Reg::Muls(yi, yi, 0.5f, mask);
        }
        Reg::Muls(index, q, uint32_t{2}, mask);
        Reg::Scatter(output, yr, index, mask);
        Reg::Adds(index, index, uint32_t{1}, mask);
        Reg::Scatter(output, yi, index, mask);
    }
}

template <bool C2R>
__no_simd_vf_fusion__ __simd_vf__ inline void EndpointsVf(__ubuf__ float* input, __ubuf__ float* output)
{
    using namespace AscendC;
    uint32_t count = 1;
    Reg::MaskReg mask = Reg::UpdateMask<float>(count);
    Reg::RegTensor<uint32_t> index;
    Reg::RegTensor<float> a, b, sum, difference, zero;
    Reg::Duplicate(index, uint32_t{0}, mask);
    Reg::Gather(a, input, index, mask);
    Reg::Duplicate(index, uint32_t{C2R ? 8 : 1}, mask);
    Reg::Gather(b, input, index, mask);
    Reg::Add(sum, a, b, mask);
    Reg::Sub(difference, a, b, mask);
    Reg::Duplicate(index, uint32_t{0}, mask);
    Reg::Scatter(output, sum, index, mask);
    Reg::Duplicate(index, uint32_t{C2R ? 1 : 8}, mask);
    Reg::Scatter(output, difference, index, mask);
    if constexpr (!C2R) {
        Reg::Duplicate(zero, 0.0f, mask);
        Reg::Duplicate(index, uint32_t{1}, mask);
        Reg::Scatter(output, zero, index, mask);
        Reg::Duplicate(index, uint32_t{9}, mask);
        Reg::Scatter(output, zero, index, mask);
    }
}

template <bool C2R, int32_t kTile>
__aicore__ inline void RunBoundary(__gm__ float* input, __gm__ float* tw, __gm__ float* output, int32_t n,
                                   int64_t batch)
{
    static_assert(kTile == 256 || kTile == 1024 || kTile == 4096);
    static_assert(8 * kTile * sizeof(float) + 64 <= 253952);
    const int32_t m = n / 2;
    const int32_t chunks = (m - 1 + kTile - 1) / kTile;
    const int64_t tasks = batch * chunks;
    __ubuf__ float a[2 * kTile], b[2 * kTile], w[2 * kTile], y[2 * kTile];
    __ubuf__ float endpoints[16];
    // GM is shared, but each (batch,chunk) has unique output ownership.
    // There is no inter-core wait. Caller stream orders boundary and FFT.
    for (int64_t task = AscendC::GetBlockIdx(); task < tasks; task += AscendC::GetBlockNum()) {
        const int64_t batchIndex = task / chunks;
        const int32_t chunk = task % chunks;
        const int32_t k = 1 + chunk * kTile;
        const uint32_t count = m - k < kTile ? m - k : kTile;
        __gm__ float* src = input + batchIndex * (C2R ? n + 2 : n);
        __gm__ float* dst = output + batchIndex * (C2R ? n : n + 2);
        // PIPE_S schedules; MTE3->MTE2 also prevents the subsequent V from
        // overwriting y while its preceding MTE3 store is still in flight.
        Fence<AscendC::HardEvent::MTE3_MTE2>();
        asc_copy_gm2ub_align(a, src + 2 * k, 1, count * 8, 0, 0, false, 0, 0, 0);
        asc_copy_gm2ub_align(b, src + 2 * (m - k - count + 1), 1, count * 8, 0, 0, false, 0, 0, 0);
        asc_copy_gm2ub_align(w, tw + 2 * k, 1, count * 8, 0, 0, false, 0, 0, 0);
        Fence<AscendC::HardEvent::MTE2_V>();
        BoundaryVf<C2R>(a, b, w, y, count);
        Fence<AscendC::HardEvent::V_MTE3>();
        asc_copy_ub2gm_align(dst + 2 * k, y, 1, count * 8, 0, 0, 0);
        if (chunk == 0) {
            Fence<AscendC::HardEvent::MTE3_MTE2>();
            asc_copy_gm2ub_align(endpoints, src, 1, 8, 0, 0, false, 0, 0, 0);
            if constexpr (C2R) {
                // Only endpoint REAL parts are consumed; imaginary values,
                // including NaNs, do not enter arithmetic.
                asc_copy_gm2ub_align(endpoints + 8, src + 2 * m, 1, 8, 0, 0, false, 0, 0, 0);
            }
            Fence<AscendC::HardEvent::MTE2_V>();
            EndpointsVf<C2R>(endpoints, y);
            Fence<AscendC::HardEvent::V_MTE3>();
            asc_copy_ub2gm_align(dst, y, 1, 8, 0, 0, 0);
            if constexpr (!C2R)
                asc_copy_ub2gm_align(dst + 2 * m, y + 8, 1, 8, 0, 0, 0);
        }
    }
    AscendC::PipeBarrier<PIPE_ALL>();
}
} // namespace
