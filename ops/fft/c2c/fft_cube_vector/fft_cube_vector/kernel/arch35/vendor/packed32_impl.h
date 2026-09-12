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
// FFT128 = four DFT32 subsequences + twiddles + FFT4.
// One real block-matrix product computes real and imaginary results directly.
// This avoids separately rounding four real partial products before cancellation.
// Direct A5 UB->L1 and L0C->UB, with separate ownership in both directions.
#include "cube_common.h"
#include "simt_api/asc_simt.h"
#include "acl/acl.h"

constexpr int kPacked32Rows = 128, kPacked32K = 64, kPacked32OriginalRows = 32;
constexpr int kPacked32Half = 64, kPacked32Plane = 4096, kPacked32HalfPlane = 2048;

__simt_vf__ __launch_bounds__(256) inline void Packed32Pack(__ubuf__ float* raw, __ubuf__ float* packed, int liveRows)
{
    for (int i = threadIdx.x; i < 2048; i += 256) {
        // Consecutive threads read consecutive complex inputs. The fractal write
        // permutes each 32-thread group across 32 different float positions.
        int row = i / 128, n = i % 128, s = n % 4, j = n / 4, m = row * 4 + s;
        float r = 0, im = 0;
        if (row < liveRows) {
            r = raw[2 * i];
            im = raw[2 * i + 1];
        }
        // Supplied port's zZ L1 layout, with 16-row/8-float fractals.
        int kk = 2 * j;
        int dst = (m / 16) * 1024 + (kk / 8) * 128 + (m % 16) * 8 + kk % 8;
        packed[dst] = r;
        packed[dst + 1] = im;
    }
}

__simt_vf__ __launch_bounds__(256) inline void Packed32Finish(__ubuf__ float* products, __ubuf__ float* tw,
                                                              __gm__ float* output, int firstRow, int liveRows,
                                                              bool inverse, int outputLog2)
{
    for (int i = threadIdx.x; i < 16 * 32; i += 256) {
        // 16 adjacent FFT rows map to contiguous natural-order large FFT stores.
        int row = i % 16, k = i / 16;
        if (row >= liveRows)
            continue;
        float rr[4], ii[4];
#pragma unroll
        for (int s = 0; s < 4; ++s) {
            int src = (2 * k) * 64 + row * 4 + s; // DN: [64 real/imag columns,64 rows]
            float r = products[src], im = products[src + 64];
            float c = tw[2 * (s * 32 + k)], t = tw[2 * (s * 32 + k) + 1];
            rr[s] = r * c - im * t;
            ii[s] = r * t + im * c;
        }
        float ur = rr[0] + rr[2], ui = ii[0] + ii[2], vr = rr[1] + rr[3], vi = ii[1] + ii[3];
        float pr = rr[0] - rr[2], pi = ii[0] - ii[2], qr = rr[1] - rr[3], qi = ii[1] - ii[3];
        float sign = inverse ? 1.f : -1.f;
        float yr[4] = {ur + vr, pr - sign * qi, ur - vr, pr + sign * qi};
        float yi[4] = {ui + vi, pi + sign * qr, ui - vi, pi - sign * qr};
        int globalRow = firstRow + row;
#pragma unroll
        for (int s = 0; s < 4; ++s) {
            int frequency = k + 32 * s;
            int64_t dst = outputLog2 ? ((int64_t)(globalRow >> outputLog2) << (outputLog2 + 7)) +
                                           (frequency << outputLog2) + (globalRow & ((1 << outputLog2) - 1)) :
                                       (int64_t)globalRow * 128 + frequency;
            output[2 * dst] = yr[s];
            output[2 * dst + 1] = yi[s];
        }
    }
}

__aicore__ inline void Packed32Cube(__gm__ float* coefficients, int rows)
{
    AscendC::GlobalTensor<float> coeff;
    coeff.SetGlobalBuffer(coefficients);
    AscendC::LocalTensor<float> b1(AscendC::TPosition::B1, 128 * 1024, 4096);
    AscendC::LocalTensor<float> b0(AscendC::TPosition::B2, 0, 4096);
    AscendC::LocalTensor<float> a0(AscendC::TPosition::A2, 0, 2 * 4096);
    SetPadding(0);
    AscendC::SetAtomicNone();
    load_matrix_zZ<float>(b1, coeff, 64, 64, 64, 64, 64);
    SET_FLAG(MTE2, MTE1, EVENT_ID2);
    WAIT_FLAG(MTE2, MTE1, EVENT_ID2);
    for (int i = 0; i < 4; ++i)
        AscendC::LoadDataWithTranspose(b0[i * 128], b1[i * 256],
                                       AscendC::LoadData2dTransposeParams(0, 4, 4, 7, 3, inc));
    SET_FLAG(MTE1, M, EVENT_ID2);
    WAIT_FLAG(MTE1, M, EVENT_ID2);
    SET_FLAG(M, MTE1, EVENT_ID2);
    SET_FLAG(FIX, M, EVENT_ID0);
    SET_FLAG(FIX, M, EVENT_ID1);
    AscendC::CrossCoreSetFlag<2, PIPE_MTE1>(0);
    AscendC::CrossCoreSetFlag<2, PIPE_MTE1>(1);
    int ordinal = 0, tasks = (rows + 31) / 32;
    for (int task = AscendC::GetBlockIdx(); task < tasks; task += AscendC::GetBlockNum(), ++ordinal) {
        int slot = ordinal % 2;
        AscendC::LocalTensor<float> a1(AscendC::TPosition::A1, slot * 32768, 2 * 4096);
        AscendC::LocalTensor<float> c(AscendC::TPosition::CO1, slot * 32768, 8192);
        AscendC::CrossCoreWaitFlag<2, PIPE_MTE1>(2 + slot);
        WAIT_FLAG(M, MTE1, EVENT_ID2);
        for (int i = 0; i < 8; ++i)
            AscendC::LoadData(a0[i * 128], a1[i * 1024], AscendC::LoadData2dParams(0, 8, 1, 0, 7, false, inc));
        AscendC::CrossCoreSetFlag<2, PIPE_MTE1>(slot);
        SET_FLAG(MTE1, M, slot);
        WAIT_FLAG(MTE1, M, slot);
        WAIT_FLAG(FIX, M, slot);
        AscendC::Mmad(c, a0, b0, AscendC::MmadParams(128, 64, 64, 0, false, true));
        SET_FLAG(M, MTE1, EVENT_ID2);
        SET_FLAG(M, FIX, slot);
        WAIT_FLAG(M, FIX, slot);
        AscendC::CrossCoreWaitFlag<2, PIPE_FIX>(6 + slot);
        AscendC::LocalTensor<float> receive(AscendC::TPosition::VECOUT, slot * 65536 + 32768, 4096);
        AscendC::FixpipeParamsArch3510<AscendC::CO2Layout::COLUMN_MAJOR> fp;
        fp.nSize = 64;
        fp.mSize = 64;
        fp.srcStride = 128;
        fp.dstStride = 64;
        fp.quantPre = QuantMode_t::NoQuant;
        fp.params.dnNum = 1;
        fp.params.srcNzC0Stride = 1;
        fp.params.srcNzMatrixStride = 0;
        fp.params.dstDnMatrixStride = 0;
        fp.subBlockId = 0;
        AscendC::Fixpipe<float, float, kDnUb>(receive, c, fp);
        fp.subBlockId = 1;
        AscendC::Fixpipe<float, float, kDnUb>(receive, c[64 * 16], fp);
        AscendC::CrossCoreSetFlag<2, PIPE_FIX>(8 + slot);
        SET_FLAG(FIX, M, slot);
    }
    AscendC::CrossCoreWaitFlag<2, PIPE_FIX>(6);
    AscendC::CrossCoreWaitFlag<2, PIPE_FIX>(7);
    WAIT_FLAG(M, MTE1, EVENT_ID2);
    WAIT_FLAG(FIX, M, EVENT_ID0);
    WAIT_FLAG(FIX, M, EVENT_ID1);
}

__aicore__ inline void Packed32Prepare(__gm__ float* input, int rows, int task, int slot)
{
    int sub = AscendC::GetSubBlockIdx(), firstRow = task * 32 + sub * 16, live = rows - firstRow;
    if (live > 16)
        live = 16;
    if (live < 0)
        live = 0;
    AscendC::GlobalTensor<float> x;
    x.SetGlobalBuffer(input);
    AscendC::LocalTensor<float> raw(AscendC::TPosition::VECIN, slot * 65536, 4096);
    AscendC::LocalTensor<float> pack(AscendC::TPosition::VECOUT, slot * 65536 + 16384, 2 * 2048);
    AscendC::CrossCoreWaitFlag<2, PIPE_MTE2>(slot);
    if (live)
        AscendC::DataCopy(raw, x[(int64_t)firstRow * 256], live * 256);
    SET_FLAG(MTE2, V, slot);
    WAIT_FLAG(MTE2, V, slot);
    asc_vf_call<Packed32Pack>(dim3(256), (__ubuf__ float*)raw.GetPhyAddr(), (__ubuf__ float*)pack.GetPhyAddr(), live);
    SET_FLAG(V, MTE3, slot);
    WAIT_FLAG(V, MTE3, slot);
    AscendC::LocalTensor<float> a1(AscendC::TPosition::A1, slot * 32768 + sub * 16384, 4096);
    AscendC::DataCopy(a1, pack, 4096);
    AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(2 + slot);
}
__aicore__ inline void Packed32Consume(__gm__ float* output, int rows, int task, int slot, bool inverse, int outputLog2)
{
    int firstRow = task * 32 + AscendC::GetSubBlockIdx() * 16, live = rows - firstRow;
    if (live > 16)
        live = 16;
    if (live < 0)
        live = 0;
    AscendC::LocalTensor<float> products(AscendC::TPosition::VECIN, slot * 65536 + 32768, 4096);
    AscendC::LocalTensor<float> tw(AscendC::TPosition::VECIN, 131072, 256);
    AscendC::CrossCoreWaitFlag<2, PIPE_V>(8 + slot);
    asc_vf_call<Packed32Finish>(dim3(256), (__ubuf__ float*)products.GetPhyAddr(), (__ubuf__ float*)tw.GetPhyAddr(),
                                output, firstRow, live, inverse, outputLog2);
    AscendC::CrossCoreSetFlag<2, PIPE_V>(6 + slot);
}

template <bool Pipelined>
__aicore__ inline void Packed32Vector(__gm__ float* input, __gm__ float* coefficients, __gm__ float* output, int rows,
                                      bool inverse, int outputLog2)
{
    AscendC::GlobalTensor<float> coeff;
    coeff.SetGlobalBuffer(coefficients);
    AscendC::LocalTensor<float> tw(AscendC::TPosition::VECIN, 131072, 256);
    AscendC::DataCopy(tw, coeff[4096], 256);
    AscendC::CrossCoreSetFlag<2, PIPE_V>(6);
    AscendC::CrossCoreSetFlag<2, PIPE_V>(7);
    const int tasks = (rows + 31) / 32, core = AscendC::GetBlockIdx() / 2, step = AscendC::GetBlockNum();
    const int count = (tasks + step - 1 - core) / step, depth = Pipelined ? 2 : 1;
    for (int j = 0; j < depth && j < count; ++j) {
        Packed32Prepare(input, rows, core + j * step, j % 2);
    }
    for (int j = 0; j < count; ++j) {
        Packed32Consume(output, rows, core + j * step, j % 2, inverse, outputLog2);
        int next = j + depth;
        if (next < count) {
            Packed32Prepare(input, rows, core + next * step, next % 2);
        }
    }
    AscendC::CrossCoreWaitFlag<2, PIPE_MTE3>(0);
    AscendC::CrossCoreWaitFlag<2, PIPE_MTE3>(1);
}
