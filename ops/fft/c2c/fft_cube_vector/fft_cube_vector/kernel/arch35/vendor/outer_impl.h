/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
// Selected from the experiment; provenance: ../SOURCE_MANIFEST.json.
// N = Q * 16384. Transform the strided Q dimension, then apply W_N^(k*n).
// The output remains [batch,Q,16384], ready for batch*Q inner FFT16384s.
// Each AIV owns disjoint column tiles of at most 4096 complex values. UB is private; the kernel boundary
// supplies global visibility before the inner FFT. No inter-core flag is used.
#include "kernel_operator.h"
#include "c_api/asc_simd.h"
#include "simt_api/asc_simt.h"
#include "acl/acl.h"
#include "outer_regbase.h"

template <int Q, int Half>
__simt_callee__ __attribute__((always_inline)) inline void OuterRegisterStage(float (&real)[Q], float (&imag)[Q],
                                                                              __ubuf__ float* tw)
{
#pragma unroll
    for (int group = 0; group < Q; group += 2 * Half) {
#pragma unroll
        for (int j = 0; j < Half; ++j) {
            int a = group + j, b = a + Half;
            float ar = real[a], ai = imag[a], br = real[b], bi = imag[b], r, im;
            if (j == 0) {
                r = br;
                im = bi;
            } else {
                float wr = tw[2 * (Half - 1 + j)], wi = tw[2 * (Half - 1 + j) + 1];
                r = br * wr - bi * wi;
                im = br * wi + bi * wr;
            }
            real[a] = ar + r;
            imag[a] = ai + im;
            real[b] = ar - r;
            imag[b] = ai - im;
        }
    }
    if constexpr (Half * 2 < Q)
        OuterRegisterStage<Q, Half * 2>(real, imag, tw);
}

template <int Q, int Threads>
__simt_vf__ __launch_bounds__(Threads) inline void OuterRegisterCompute(__ubuf__ float* raw, __ubuf__ float* tw,
                                                                        __ubuf__ float* cross, int colLog2)
{
    constexpr int bits = __builtin_ctz((unsigned)Q);
    const int columns = 1 << colLog2;
    for (int col = threadIdx.x; col < columns; col += Threads) {
        float real[Q], imag[Q];
#pragma unroll
        for (int k = 0; k < Q; ++k) {
            int rev = 0;
#pragma unroll
            for (int bit = 0; bit < bits; ++bit)
                rev = (rev << 1) | ((k >> bit) & 1);
            real[k] = raw[2 * (rev * columns + col)];
            imag[k] = raw[2 * (rev * columns + col) + 1];
        }
        OuterRegisterStage<Q, 1>(real, imag, tw);
#pragma unroll
        for (int k = 0; k < Q; ++k) {
            int i = k * columns + col;
            float c = cross[2 * i], s = cross[2 * i + 1];
            raw[2 * i] = real[k] * c - imag[k] * s;
            raw[2 * i + 1] = real[k] * s + imag[k] * c;
        }
    }
}

__aicore__ inline void RunOuterRegister(__gm__ float* input, __gm__ float* twiddles, __gm__ float* crossTwiddles,
                                        __gm__ float* output, int batch, int qLog2, int colLog2)
{
    AscendC::InitSocState();
    AscendC::LocalTensor<float> raw(AscendC::TPosition::VECIN, 0, 8192);
    AscendC::LocalTensor<float> values(AscendC::TPosition::VECCALC, 32768, 8192);
    AscendC::LocalTensor<float> cross(AscendC::TPosition::VECIN, 65536, 8192);
    AscendC::LocalTensor<float> tw(AscendC::TPosition::VECIN, 98304, 128);
    AscendC::GlobalTensor<float> in, out, table, local;
    in.SetGlobalBuffer(input);
    out.SetGlobalBuffer(output);
    table.SetGlobalBuffer(crossTwiddles);
    local.SetGlobalBuffer(twiddles);
    AscendC::DataCopy(tw, local, 128);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    const int q = 1 << qLog2, columns = 1 << colLog2, tiles = 16384 >> colLog2, tasks = batch * tiles;
    for (int task = AscendC::GetBlockIdx(); task < tasks; task += AscendC::GetBlockNum()) {
        int b = task >> (14 - colLog2), col = (task & (tiles - 1)) << colLog2;
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::DataCopyParams cp(q, columns / 4, (16384 - columns) / 4, 0);
        AscendC::DataCopy(raw, in[(int64_t)b * q * 32768 + col * 2], cp);
        AscendC::DataCopy(cross, table[col * 2], cp);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        switch (qLog2) {
            case 2:
                asc_vf_call<OuterRegisterCompute<4, 256>>(dim3(256), (__ubuf__ float*)raw.GetPhyAddr(),
                                                          (__ubuf__ float*)tw.GetPhyAddr(),
                                                          (__ubuf__ float*)cross.GetPhyAddr(), colLog2);
                break;
            case 3:
                asc_vf_call<OuterRegisterCompute<8, 256>>(dim3(256), (__ubuf__ float*)raw.GetPhyAddr(),
                                                          (__ubuf__ float*)tw.GetPhyAddr(),
                                                          (__ubuf__ float*)cross.GetPhyAddr(), colLog2);
                break;
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        cp.srcStride = 0;
        cp.dstStride = (16384 - columns) / 4;
        AscendC::DataCopy(out[(int64_t)b * q * 32768 + col * 2], raw, cp);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    }
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    AscendC::PipeBarrier<PIPE_ALL>();
}

__aicore__ inline void RunOuterSimd(__gm__ float* input, __gm__ float* twiddles, __gm__ float* crossTwiddles,
                                    __gm__ float* output, int batch, int qLog2, int colLog2)
{
    AscendC::InitSocState();
    AscendC::LocalTensor<float> raw(AscendC::TPosition::VECIN, 0, 8192);
    AscendC::LocalTensor<float> values(AscendC::TPosition::VECCALC, 32768, 8192);
    AscendC::LocalTensor<float> cross(AscendC::TPosition::VECIN, 65536, 8192);
    AscendC::LocalTensor<float> tw(AscendC::TPosition::VECIN, 98304, 128);
    AscendC::GlobalTensor<float> in, out, table, local;
    in.SetGlobalBuffer(input);
    out.SetGlobalBuffer(output);
    table.SetGlobalBuffer(crossTwiddles);
    local.SetGlobalBuffer(twiddles);
    AscendC::DataCopy(tw, local, 128);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    const int q = 1 << qLog2, columns = 1 << colLog2, tiles = 16384 >> colLog2, tasks = batch * tiles;
    for (int task = AscendC::GetBlockIdx(); task < tasks; task += AscendC::GetBlockNum()) {
        int b = task >> (14 - colLog2), col = (task & (tiles - 1)) << colLog2;
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::DataCopyParams cp(q, columns / 4, (16384 - columns) / 4, 0);
        AscendC::DataCopy(raw, in[(int64_t)b * q * 32768 + col * 2], cp);
        AscendC::DataCopy(cross, table[col * 2], cp);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        switch (qLog2) {
            case 4:
                OuterSimdCompute<16>((__ubuf__ float*)raw.GetPhyAddr(), (__ubuf__ float*)tw.GetPhyAddr(),
                                     (__ubuf__ float*)cross.GetPhyAddr(), colLog2);
                break;
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        cp.srcStride = 0;
        cp.dstStride = (16384 - columns) / 4;
        AscendC::DataCopy(out[(int64_t)b * q * 32768 + col * 2], raw, cp);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    }
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    AscendC::PipeBarrier<PIPE_ALL>();
}

template <int Half>
__simt_callee__ __attribute__((always_inline)) inline void OuterOctetStage(float (&real)[8], float (&imag)[8],
                                                                           __ubuf__ float* tw, int baseJ, int spacing)
{
#pragma unroll
    for (int group = 0; group < 8; group += 2 * Half) {
#pragma unroll
        for (int k = 0; k < Half; ++k) {
            const int a = group + k, b = a + Half, j = baseJ + k * spacing;
            float br = real[b], bi = imag[b];
            if (j != 0) {
                const int t = Half * spacing - 1 + j;
                float wr = tw[2 * t], wi = tw[2 * t + 1], r = br * wr - bi * wi;
                bi = br * wi + bi * wr;
                br = r;
            }
            const float ar = real[a], ai = imag[a];
            real[a] = ar + br;
            imag[a] = ai + bi;
            real[b] = ar - br;
            imag[b] = ai - bi;
        }
    }
    if constexpr (Half < 4)
        OuterOctetStage<Half * 2>(real, imag, tw, baseJ, spacing);
}

// Q64 only: three radix-2 rounds per eight-value register-owned group.
// Two rounds of UB exchange, instead of three for radix4 or six for radix2.
__simt_vf__ __launch_bounds__(256) inline void OuterRadix8Compute(__ubuf__ float* raw, __ubuf__ float* values,
                                                                  __ubuf__ float* tw, __ubuf__ float* cross,
                                                                  int colLog2)
{
    const int columns = 1 << colLog2;
    for (int i = threadIdx.x; i < 64 * columns; i += 256) {
        int k = i >> colLog2, col = i & (columns - 1), rev = 0;
#pragma unroll
        for (int bit = 0; bit < 6; ++bit)
            rev = (rev << 1) | ((k >> bit) & 1);
        values[2 * (rev * columns + col)] = raw[2 * i];
        values[2 * (rev * columns + col) + 1] = raw[2 * i + 1];
    }
    asc_syncthreads();
#pragma unroll 1
    for (int stage = 0; stage < 6; stage += 3) {
        const int spacing = 1 << stage;
        for (int i = threadIdx.x; i < 8 * columns; i += 256) {
            const int group = i >> colLog2, col = i & (columns - 1), j = group & (spacing - 1);
            const int base = ((group >> stage) << (stage + 3)) + j;
            float real[8], imag[8];
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                const int index = (base + r * spacing) * columns + col;
                real[r] = values[2 * index];
                imag[r] = values[2 * index + 1];
            }
            OuterOctetStage<1>(real, imag, tw, j, spacing);
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                const int index = (base + r * spacing) * columns + col;
                values[2 * index] = real[r];
                values[2 * index + 1] = imag[r];
            }
        }
        asc_syncthreads();
    }
    for (int i = threadIdx.x; i < 64 * columns; i += 256) {
        float r = values[2 * i], im = values[2 * i + 1], c = cross[2 * i], s = cross[2 * i + 1];
        raw[2 * i] = r * c - im * s;
        raw[2 * i + 1] = r * s + im * c;
    }
}

// Two radix-2 rounds per register-owned quartet, with one private-UB barrier.
// The final odd round remains radix 2. No thread owns a full FFT64 column.
__simt_vf__ __launch_bounds__(256) inline void OuterRadix4Compute(__ubuf__ float* raw, __ubuf__ float* values,
                                                                  __ubuf__ float* tw, __ubuf__ float* cross, int bits,
                                                                  int colLog2)
{
    const int q = 1 << bits, columns = 1 << colLog2;
    for (int i = threadIdx.x; i < q * columns; i += 256) {
        int k = i >> colLog2, col = i & (columns - 1), rev = 0;
        for (int bit = 0; bit < bits; ++bit)
            rev = (rev << 1) | ((k >> bit) & 1);
        values[2 * (rev * columns + col)] = raw[2 * i];
        values[2 * (rev * columns + col) + 1] = raw[2 * i + 1];
    }
    asc_syncthreads();
    int stage = 0;
    for (; stage + 1 < bits; stage += 2) {
        const int half = 1 << stage;
        for (int i = threadIdx.x; i < (q / 4) * columns; i += 256) {
            int quartet = i >> colLog2, col = i & (columns - 1), j = quartet & (half - 1);
            int a = (((quartet >> stage) << (stage + 2)) + j) * columns + col;
            int b = a + half * columns, c = b + half * columns, d = c + half * columns;
            float ar = values[2 * a], ai = values[2 * a + 1], br = values[2 * b], bi = values[2 * b + 1];
            float cr = values[2 * c], ci = values[2 * c + 1], dr = values[2 * d], di = values[2 * d + 1];
            if (j != 0) {
                int t = half - 1 + j;
                float wr = tw[2 * t], wi = tw[2 * t + 1];
                float r = br * wr - bi * wi;
                bi = br * wi + bi * wr;
                br = r;
                r = dr * wr - di * wi;
                di = dr * wi + di * wr;
                dr = r;
            }
            float ur = ar + br, ui = ai + bi, vr = ar - br, vi = ai - bi;
            float pr = cr + dr, pi = ci + di, sr = cr - dr, si = ci - di;
            int t = 2 * half - 1 + j;
            if (j != 0) {
                float wr = tw[2 * t], wi = tw[2 * t + 1];
                float r = pr * wr - pi * wi;
                pi = pr * wi + pi * wr;
                pr = r;
            }
            t += half;
            float wr = tw[2 * t], wi = tw[2 * t + 1];
            float r = sr * wr - si * wi;
            si = sr * wi + si * wr;
            sr = r;
            values[2 * a] = ur + pr;
            values[2 * a + 1] = ui + pi;
            values[2 * c] = ur - pr;
            values[2 * c + 1] = ui - pi;
            values[2 * b] = vr + sr;
            values[2 * b + 1] = vi + si;
            values[2 * d] = vr - sr;
            values[2 * d + 1] = vi - si;
        }
        asc_syncthreads();
    }
    if (stage < bits) {
        int half = 1 << stage;
        for (int i = threadIdx.x; i < (q / 2) * columns; i += 256) {
            int butterfly = i >> colLog2, col = i & (columns - 1), j = butterfly & (half - 1);
            int a = (((butterfly >> stage) << (stage + 1)) + j) * columns + col, b = a + half * columns;
            float ar = values[2 * a], ai = values[2 * a + 1], br = values[2 * b], bi = values[2 * b + 1];
            if (j != 0) {
                int t = half - 1 + j;
                float wr = tw[2 * t], wi = tw[2 * t + 1];
                float r = br * wr - bi * wi;
                bi = br * wi + bi * wr;
                br = r;
            }
            values[2 * a] = ar + br;
            values[2 * a + 1] = ai + bi;
            values[2 * b] = ar - br;
            values[2 * b + 1] = ai - bi;
        }
        asc_syncthreads();
    }
    for (int i = threadIdx.x; i < q * columns; i += 256) {
        float r = values[2 * i], im = values[2 * i + 1], c = cross[2 * i], s = cross[2 * i + 1];
        raw[2 * i] = r * c - im * s;
        raw[2 * i + 1] = r * s + im * c;
    }
}

__aicore__ inline void OuterPrefetch(AscendC::GlobalTensor<float>& in, AscendC::GlobalTensor<float>& table, int task,
                                     int slot, int q, int colLog2)
{
    const int columns = 1 << colLog2, tiles = 16384 >> colLog2;
    int batch = task >> (14 - colLog2), col = (task & (tiles - 1)) << colLog2;
    AscendC::LocalTensor<float> raw(AscendC::TPosition::VECIN, slot ? 98304 : 0, 8192);
    AscendC::LocalTensor<float> cross(AscendC::TPosition::VECIN, slot ? 131072 : 65536, 8192);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(slot);
    AscendC::DataCopyParams cp(q, columns / 4, (16384 - columns) / 4, 0);
    AscendC::DataCopy(raw, in[(int64_t)batch * q * 32768 + col * 2], cp);
    AscendC::DataCopy(cross, table[col * 2], cp);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(slot);
}

template <bool DoubleBuffered, bool Radix4, bool Radix8 = false>
__aicore__ inline void OuterOptimized(__gm__ float* input, __gm__ float* twiddles, __gm__ float* crossTwiddles,
                                      __gm__ float* output, int batch, int qLog2, int colLog2)
{
    static_assert(DoubleBuffered && (Radix4 || Radix8), "Only retained buffered outer engines");
    const int q = 1 << qLog2, columns = 1 << colLog2, tiles = 16384 >> colLog2, tasks = batch * tiles;
    const int core = AscendC::GetBlockIdx(), step = AscendC::GetBlockNum();
    if (core >= tasks)
        return;
    AscendC::LocalTensor<float> values(AscendC::TPosition::VECCALC, 32768, 8192);
    AscendC::LocalTensor<float> tw(AscendC::TPosition::VECIN, DoubleBuffered ? 163840 : 98304, 128);
    AscendC::GlobalTensor<float> in, out, table, local;
    in.SetGlobalBuffer(input);
    out.SetGlobalBuffer(output);
    table.SetGlobalBuffer(crossTwiddles);
    local.SetGlobalBuffer(twiddles);
    AscendC::DataCopy(tw, local, 128);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    if constexpr (DoubleBuffered)
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    OuterPrefetch(in, table, core, 0, q, colLog2);
    int ordinal = 0;
    for (int task = core; task < tasks; task += step, ++ordinal) {
        int slot = DoubleBuffered ? (ordinal & 1) : 0, next = task + step;
        if constexpr (DoubleBuffered) {
            if (next < tasks)
                OuterPrefetch(in, table, next, slot ^ 1, q, colLog2);
        }
        AscendC::LocalTensor<float> raw(AscendC::TPosition::VECIN, slot ? 98304 : 0, 8192);
        AscendC::LocalTensor<float> cross(AscendC::TPosition::VECIN, slot ? 131072 : 65536, 8192);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(slot);
        if constexpr (Radix8) {
            asc_vf_call<OuterRadix8Compute>(dim3(256), (__ubuf__ float*)raw.GetPhyAddr(),
                                            (__ubuf__ float*)values.GetPhyAddr(), (__ubuf__ float*)tw.GetPhyAddr(),
                                            (__ubuf__ float*)cross.GetPhyAddr(), colLog2);
        } else if constexpr (Radix4) {
            asc_vf_call<OuterRadix4Compute>(dim3(256), (__ubuf__ float*)raw.GetPhyAddr(),
                                            (__ubuf__ float*)values.GetPhyAddr(), (__ubuf__ float*)tw.GetPhyAddr(),
                                            (__ubuf__ float*)cross.GetPhyAddr(), qLog2, colLog2);
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(slot);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(slot);
        int b = task >> (14 - colLog2), col = (task & (tiles - 1)) << colLog2;
        AscendC::DataCopyParams cp(q, columns / 4, 0, (16384 - columns) / 4);
        AscendC::DataCopy(out[(int64_t)b * q * 32768 + col * 2], raw, cp);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(slot);
        if constexpr (!DoubleBuffered) {
            if (next < tasks)
                OuterPrefetch(in, table, next, 0, q, colLog2);
        }
    }
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    if constexpr (DoubleBuffered)
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
}
