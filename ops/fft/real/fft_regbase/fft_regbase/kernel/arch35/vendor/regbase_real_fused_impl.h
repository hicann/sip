/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef FFT_REAL_FUSED_IMPL_H
#define FFT_REAL_FUSED_IMPL_H

namespace real_fused {

// One whole transform per row. Half spectra are padded only in private UB:
// N+8 floats keeps every row base 32-byte aligned; GM still has N+2 floats.
template <int N, bool C2R>
__no_simd_vf_fusion__ __simd_vf__ inline void ConvertRows(__ubuf__ float* input, __ubuf__ float* tw,
                                                          __ubuf__ float* output, uint16_t tileBatch)
{
    using namespace AscendC;
    constexpr uint32_t m = N / 2;
    constexpr uint32_t inputStride = C2R ? N + 8 : N;
    constexpr uint32_t outputStride = C2R ? N : N + 8;
#pragma unroll 1
    for (uint16_t transform = 0; transform < tileBatch; ++transform) {
        __ubuf__ float* src = input + transform * inputStride;
        __ubuf__ float* dst = output + transform * outputStride;
        uint32_t remaining = m - 1;
#pragma unroll 1
        for (uint16_t repeat = 0; repeat < (m - 1 + 63) / 64; ++repeat) {
            Reg::MaskReg mask = Reg::UpdateMask<float>(remaining);
            Reg::RegTensor<uint32_t> k, index, mirror, limit;
            Reg::RegTensor<float> ar, ai, br, bi, wr, wi;
            Reg::RegTensor<float> er, ei, dr, di, yr, yi, temp;
            Reg::Arange<int32_t>(reinterpret_cast<Reg::RegTensor<int32_t>&>(k), 1 + repeat * 64);
            Reg::Muls(index, k, uint32_t{2}, mask);
            Reg::Gather(ar, src, index, mask);
            if constexpr (N == 16384) {
                // Store W[0..M/2] only. The upper interval follows
                // W[M-k] = -conj(W[k]); its boundary is between VF repeats.
                if (repeat >= m / 128) {
                    Reg::Duplicate(limit, 2U * m, mask);
                    Reg::Sub(index, limit, index, mask);
                }
                Reg::Gather(wr, tw, index, mask);
                Reg::Adds(index, index, uint32_t{1}, mask);
                Reg::Gather(wi, tw, index, mask);
                if (repeat >= m / 128)
                    Reg::Muls(wr, wr, -1.0f, mask);
                Reg::Muls(index, k, uint32_t{2}, mask);
                Reg::Adds(index, index, uint32_t{1}, mask);
                Reg::Gather(ai, src, index, mask);
            } else {
                Reg::Gather(wr, tw, index, mask);
                Reg::Adds(index, index, uint32_t{1}, mask);
                Reg::Gather(ai, src, index, mask);
                Reg::Gather(wi, tw, index, mask);
            }
            Reg::Duplicate(limit, m, mask);
            Reg::Sub(mirror, limit, k, mask);
            Reg::Muls(mirror, mirror, uint32_t{2}, mask);
            Reg::Gather(br, src, mirror, mask);
            Reg::Adds(mirror, mirror, uint32_t{1}, mask);
            Reg::Gather(bi, src, mirror, mask);
            Reg::Add(er, ar, br, mask);
            Reg::Sub(ei, ai, bi, mask);
            Reg::Sub(dr, ar, br, mask);
            Reg::Add(di, ai, bi, mask);
            if constexpr (C2R) {
                // C[k]=(a+b)+i*W_N[k]*(a-b); factor two is intentional.
                Reg::Mul(temp, wr, di, mask);
                Reg::Sub(yr, er, temp, mask);
                Reg::Mul(temp, wi, dr, mask);
                Reg::Sub(yr, yr, temp, mask);
                Reg::Mul(temp, wr, dr, mask);
                Reg::Add(yi, ei, temp, mask);
                Reg::Mul(temp, wi, di, mask);
                Reg::Sub(yi, yi, temp, mask);
            } else {
                Reg::Mul(temp, wr, di, mask);
                Reg::Add(yr, er, temp, mask);
                Reg::Mul(temp, wi, dr, mask);
                Reg::Add(yr, yr, temp, mask);
                Reg::Mul(temp, wi, di, mask);
                Reg::Add(yi, ei, temp, mask);
                Reg::Mul(temp, wr, dr, mask);
                Reg::Sub(yi, yi, temp, mask);
                Reg::Muls(yr, yr, 0.5f, mask);
                Reg::Muls(yi, yi, 0.5f, mask);
            }
            Reg::Muls(index, k, uint32_t{2}, mask);
            Reg::Scatter(dst, yr, index, mask);
            Reg::Adds(index, index, uint32_t{1}, mask);
            Reg::Scatter(dst, yi, index, mask);
        }
        // Endpoints stay in the same UB row: no additional DMA or VF call.
        // C2R deliberately never reads the endpoint imaginary components.
        uint32_t one = 1;
        Reg::MaskReg mask = Reg::UpdateMask<float>(one);
        Reg::RegTensor<uint32_t> index;
        Reg::RegTensor<float> a, b, sum, difference, zero;
        Reg::Duplicate(index, uint32_t{0}, mask);
        Reg::Gather(a, src, index, mask);
        Reg::Duplicate(index, uint32_t{C2R ? N : 1}, mask);
        Reg::Gather(b, src, index, mask);
        Reg::Add(sum, a, b, mask);
        Reg::Sub(difference, a, b, mask);
        Reg::Duplicate(index, uint32_t{0}, mask);
        Reg::Scatter(dst, sum, index, mask);
        Reg::Duplicate(index, uint32_t{C2R ? 1 : N}, mask);
        Reg::Scatter(dst, difference, index, mask);
        if constexpr (!C2R) {
            Reg::Duplicate(zero, 0.0f, mask);
            Reg::Duplicate(index, uint32_t{1}, mask);
            Reg::Scatter(dst, zero, index, mask);
            Reg::Duplicate(index, uint32_t{N + 1}, mask);
            Reg::Scatter(dst, zero, index, mask);
        }
    }
}

template <int N, bool C2R>
__aicore__ inline void LoadTile(__ubuf__ float* packed, __ubuf__ float* half, __gm__ float* input, int64_t globalBatch,
                                uint16_t count)
{
    if constexpr (C2R) {
        // Byte-length DMA: source/destination strides are head-to-head.
        asc_copy_gm2ub_align(half, input + globalBatch * (N + 2), count, (N + 2) * 4, 0, 0, false, 0, (N + 2) * 4,
                             (N + 8) * 4);
    } else {
        asc_copy_gm2ub_align(packed, input + globalBatch * N, 1, count * N * 4, 0, 0, false, 0, 0, 0);
    }
}

template <int N, bool C2R, bool Contiguous, bool Inverse, bool DoubleBuffer>
__aicore__ inline void Run(__gm__ float* input, __gm__ float* coreTwiddles, __gm__ float* boundaryTwiddles,
                           __gm__ float* output, int64_t batch)
{
    static_assert(N >= 128 && N <= 16384 && (N & (N - 1)) == 0);
    static_assert(N <= 8192 || !DoubleBuffer);
    constexpr int m = N / 2;
    constexpr int tileCapacity = m <= 256 ? (m == 256 && Contiguous ? 4 : 256 / m) : (m == 512 ? 2 : 1);
    // Radix-4 ping/pong buffers also serve the conversion. Either can be the
    // final FFT buffer; both therefore fit padded half rows and LoadAlign's
    // 64-complex guard. No third tile or additional GM workspace is needed.
    constexpr int packedFloats = m >= 512 ? tileCapacity * (N + 8) + 128 : tileCapacity * N;
    constexpr int halfFloats = m >= 512 ? packedFloats : tileCapacity * (N + 8);
    constexpr int slots = DoubleBuffer ? 2 : 1;
    constexpr int realTwFloats = N == 16384 ? N / 2 + 8 : N + 8;
    static_assert(4 * (slots * (packedFloats + halfFloats) + N + realTwFloats) <= 253952);
    __ubuf__ float packedStorage[slots * packedFloats];
    __ubuf__ float halfStorage[slots * halfFloats];
    __ubuf__ float fftTw[N];
    __ubuf__ float realTw[realTwFloats];
    const CoreBatchRange range = GetCoreBatchRange(batch);
    if (!range.count)
        return;
    asc_copy_gm2ub_align(fftTw, coreTwiddles, 1, N * 4, 0, 0, false, 0, 0, 0);
    asc_copy_gm2ub_align(realTw, boundaryTwiddles, 1, (realTwFloats - 6) * 4, 0, 0, false, 0, 0, 0);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    if constexpr (DoubleBuffer) {
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
        const uint16_t firstCount = static_cast<uint16_t>(range.count < tileCapacity ? range.count : tileCapacity);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        LoadTile<N, C2R>(packedStorage, halfStorage, input, range.start, firstCount);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
    }
    int64_t tileIndex = 0;
    for (int64_t local = 0; local < range.count; local += tileCapacity, ++tileIndex) {
        const int64_t left = range.count - local;
        const uint16_t count = static_cast<uint16_t>(left < tileCapacity ? left : tileCapacity);
        const int64_t globalBatch = range.start + local;
        const int slot = DoubleBuffer ? (tileIndex & 1) : 0;
        const int eventId = slot ? EVENT_ID1 : EVENT_ID0;
        __ubuf__ float* packed = packedStorage + slot * packedFloats;
        __ubuf__ float* half = halfStorage + slot * halfFloats;
        if constexpr (DoubleBuffer) {
            const int64_t next = local + tileCapacity;
            if (next < range.count) {
                const int nextSlot = 1 - slot;
                const int nextEvent = nextSlot ? EVENT_ID1 : EVENT_ID0;
                const int64_t nextLeft = range.count - next;
                const uint16_t nextCount = static_cast<uint16_t>(nextLeft < tileCapacity ? nextLeft : tileCapacity);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(nextEvent);
                LoadTile<N, C2R>(packedStorage + nextSlot * packedFloats, halfStorage + nextSlot * halfFloats, input,
                                 range.start + next, nextCount);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(nextEvent);
            }
        } else {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
            LoadTile<N, C2R>(packed, half, input, globalBatch, count);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventId);
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventId);
        if constexpr (C2R) {
            ConvertRows<N, true>(half, realTw, packed, count);
            // Separate VFs share this core's UB. Complete its writes before
            // the unchanged FFT VF reads them; no inter-core synchronization.
            AscendC::PipeBarrier<PIPE_V>();
        }
        __ubuf__ float* fftOutput = packed;
        __ubuf__ float* halfOutput = half;
        if constexpr (m == 64) {
            StockhamRadix2FusedRegisterDirectVf<64>(packed, fftTw, packed, count);
        } else if constexpr (m == 128) {
            StockhamRadix2FusedRegisterLazy128DirectVf<Contiguous>(packed, fftTw, packed, count);
        } else if constexpr (m == 256) {
            StockhamRadix2FusedRegisterLazy256DirectVf<Contiguous>(packed, fftTw, packed, count);
        } else {
            StockhamRadix4FusedUnrolledDirectVf<m, Inverse>(packed, fftTw, half, count);
            if constexpr ((FftTraits<m>::kRadix4Passes & 1) != 0) {
                fftOutput = half;
                halfOutput = packed;
            }
        }
        if constexpr (!C2R) {
            AscendC::PipeBarrier<PIPE_V>();
            ConvertRows<N, false>(fftOutput, realTw, halfOutput, count);
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventId);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventId);
        if constexpr (C2R) {
            asc_copy_ub2gm_align(output + globalBatch * N, fftOutput, 1, count * N * 4, 0, 0, 0);
        } else {
            asc_copy_ub2gm_align(output + globalBatch * (N + 2), halfOutput, count, (N + 2) * 4, 0, (N + 2) * 4,
                                 (N + 8) * 4);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
    }
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    if constexpr (DoubleBuffer)
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
}
} // namespace real_fused

#endif
