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
// Three-real-product complex multiplication; all operations remain FP32.
// Direct A5 UB->L1 and L0C->UB, with separate ownership in both directions.
#include "cube_common.h"
#include "simt_api/asc_simt.h"
#include "acl/acl.h"

constexpr int kGaussRows = 128, kGaussK = 32, kGaussOriginalRows = 32;
constexpr int kGaussHalf = 64, kGaussPlane = 4096, kGaussHalfPlane = 2048;

__simt_vf__ __launch_bounds__(256) inline void GaussPack(__ubuf__ float* raw, __ubuf__ float* packed, int liveRows)
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
        int dst = (m / 16) * 512 + (j / 8) * 128 + (m % 16) * 8 + j % 8;
        packed[dst] = r;
        packed[2048 + dst] = im;
        packed[4096 + dst] = r + im;
    }
}

__simt_vf__ __launch_bounds__(256) inline void GaussFinish(__ubuf__ float* products, __ubuf__ float* tw,
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
            int src = k * 64 + row * 4 + s; // Fixpipe DN, each product [32,64]
            float p = products[src], q = products[2048 + src], v = products[4096 + src];
            float r = p - q, im = (v - p) - q;
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

__aicore__ inline void GaussCube(__gm__ float* coefficients, int rows)
{
    AscendC::GlobalTensor<float> coeff;
    coeff.SetGlobalBuffer(coefficients);
    AscendC::LocalTensor<float> b1(AscendC::TPosition::B1, 128 * 1024, 3072);
    AscendC::LocalTensor<float> b0(AscendC::TPosition::B2, 0, 3072);
    AscendC::LocalTensor<float> a0(AscendC::TPosition::A2, 0, 3 * 4096);
    SetPadding(0);
    AscendC::SetAtomicNone();
    for (int p = 0; p < 3; ++p)
        load_matrix_zZ<float>(b1[p * 1024], coeff[p * 1024], 32, 32, 32, 32, 32);
    SET_FLAG(MTE2, MTE1, EVENT_ID2);
    WAIT_FLAG(MTE2, MTE1, EVENT_ID2);
    for (int p = 0; p < 3; ++p)
        for (int i = 0; i < 2; ++i)
            AscendC::LoadDataWithTranspose(b0[p * 1024 + i * 128], b1[p * 1024 + i * 256],
                                           AscendC::LoadData2dTransposeParams(0, 2, 2, 3, 1, inc));
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
        AscendC::LocalTensor<float> a1(AscendC::TPosition::A1, slot * 49152, 3 * 4096);
        AscendC::LocalTensor<float> c(AscendC::TPosition::CO1, slot * 65536, 3 * 4096);
        AscendC::CrossCoreWaitFlag<2, PIPE_MTE1>(2 + slot);
        WAIT_FLAG(M, MTE1, EVENT_ID2);
        for (int p = 0; p < 3; ++p)
            for (int i = 0; i < 8; ++i)
                AscendC::LoadData(a0[p * 4096 + i * 128], a1[p * 4096 + i * 512],
                                  AscendC::LoadData2dParams(0, 4, 1, 0, 7, false, inc));
        AscendC::CrossCoreSetFlag<2, PIPE_MTE1>(slot);
        SET_FLAG(MTE1, M, slot);
        WAIT_FLAG(MTE1, M, slot);
        WAIT_FLAG(FIX, M, slot);
        for (int p = 0; p < 3; ++p)
            AscendC::Mmad(c[p * 4096], a0[p * 4096], b0[p * 1024], AscendC::MmadParams(128, 32, 32, 0, false, true));
        SET_FLAG(M, MTE1, EVENT_ID2);
        SET_FLAG(M, FIX, slot);
        WAIT_FLAG(M, FIX, slot);
        AscendC::CrossCoreWaitFlag<2, PIPE_FIX>(6 + slot);
        AscendC::LocalTensor<float> receive(AscendC::TPosition::VECOUT, slot * 65536 + 40960, 3 * 2048);
        AscendC::FixpipeParamsArch3510<AscendC::CO2Layout::COLUMN_MAJOR> fp;
        fp.nSize = 32;
        fp.mSize = 64;
        fp.srcStride = 128;
        fp.dstStride = 64;
        fp.quantPre = QuantMode_t::NoQuant;
        fp.params.dnNum = 1;
        fp.params.srcNzC0Stride = 1;
        fp.params.srcNzMatrixStride = 0;
        fp.params.dstDnMatrixStride = 0;
        for (int p = 0; p < 3; ++p) {
            fp.subBlockId = 0;
            AscendC::Fixpipe<float, float, kDnUb>(receive[p * 2048], c[p * 4096], fp);
            fp.subBlockId = 1;
            AscendC::Fixpipe<float, float, kDnUb>(receive[p * 2048], c[p * 4096 + 64 * 16], fp);
        }
        AscendC::CrossCoreSetFlag<2, PIPE_FIX>(8 + slot);
        SET_FLAG(FIX, M, slot);
    }
    AscendC::CrossCoreWaitFlag<2, PIPE_FIX>(6);
    AscendC::CrossCoreWaitFlag<2, PIPE_FIX>(7);
    WAIT_FLAG(M, MTE1, EVENT_ID2);
    WAIT_FLAG(FIX, M, EVENT_ID0);
    WAIT_FLAG(FIX, M, EVENT_ID1);
}

__aicore__ inline void GaussPrepare(__gm__ float* input, int rows, int task, int slot)
{
    int sub = AscendC::GetSubBlockIdx(), firstRow = task * 32 + sub * 16, live = rows - firstRow;
    if (live > 16)
        live = 16;
    if (live < 0)
        live = 0;
    AscendC::GlobalTensor<float> x;
    x.SetGlobalBuffer(input);
    AscendC::LocalTensor<float> raw(AscendC::TPosition::VECIN, slot * 65536, 4096);
    AscendC::LocalTensor<float> pack(AscendC::TPosition::VECOUT, slot * 65536 + 16384, 3 * 2048);
    AscendC::CrossCoreWaitFlag<2, PIPE_MTE2>(slot);
    if (live)
        AscendC::DataCopy(raw, x[(int64_t)firstRow * 256], live * 256);
    SET_FLAG(MTE2, V, slot);
    WAIT_FLAG(MTE2, V, slot);
    asc_vf_call<GaussPack>(dim3(256), (__ubuf__ float*)raw.GetPhyAddr(), (__ubuf__ float*)pack.GetPhyAddr(), live);
    SET_FLAG(V, MTE3, slot);
    WAIT_FLAG(V, MTE3, slot);
    for (int p = 0; p < 3; ++p) {
        AscendC::LocalTensor<float> a1(AscendC::TPosition::A1, slot * 49152 + p * 16384 + sub * 8192, 2048);
        AscendC::DataCopy(a1, pack[p * 2048], 2048);
    }
    AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(2 + slot);
}
__aicore__ inline void GaussConsume(__gm__ float* output, int rows, int task, int slot, bool inverse, int outputLog2)
{
    int firstRow = task * 32 + AscendC::GetSubBlockIdx() * 16, live = rows - firstRow;
    if (live > 16)
        live = 16;
    if (live < 0)
        live = 0;
    AscendC::LocalTensor<float> products(AscendC::TPosition::VECIN, slot * 65536 + 40960, 3 * 2048);
    AscendC::LocalTensor<float> tw(AscendC::TPosition::VECIN, 131072, 256);
    AscendC::CrossCoreWaitFlag<2, PIPE_V>(8 + slot);
    asc_vf_call<GaussFinish>(dim3(256), (__ubuf__ float*)products.GetPhyAddr(), (__ubuf__ float*)tw.GetPhyAddr(),
                             output, firstRow, live, inverse, outputLog2);
    AscendC::CrossCoreSetFlag<2, PIPE_V>(6 + slot);
}

template <bool Pipelined>
__aicore__ inline void GaussVector(__gm__ float* input, __gm__ float* coefficients, __gm__ float* output, int rows,
                                   bool inverse, int outputLog2)
{
    AscendC::GlobalTensor<float> coeff;
    coeff.SetGlobalBuffer(coefficients);
    AscendC::LocalTensor<float> tw(AscendC::TPosition::VECIN, 131072, 256);
    AscendC::DataCopy(tw, coeff[3072], 256);
    AscendC::CrossCoreSetFlag<2, PIPE_V>(6);
    AscendC::CrossCoreSetFlag<2, PIPE_V>(7);
    const int tasks = (rows + 31) / 32, core = AscendC::GetBlockIdx() / 2, step = AscendC::GetBlockNum();
    const int count = (tasks + step - 1 - core) / step, depth = Pipelined ? 2 : 1;
    for (int j = 0; j < depth && j < count; ++j)
        GaussPrepare(input, rows, core + j * step, j % 2);
    for (int j = 0; j < count; ++j) {
        GaussConsume(output, rows, core + j * step, j % 2, inverse, outputLog2);
        int next = j + depth;
        if (next < count)
            GaussPrepare(input, rows, core + next * step, next % 2);
    }
    AscendC::CrossCoreWaitFlag<2, PIPE_MTE3>(0);
    AscendC::CrossCoreWaitFlag<2, PIPE_MTE3>(1);
}
