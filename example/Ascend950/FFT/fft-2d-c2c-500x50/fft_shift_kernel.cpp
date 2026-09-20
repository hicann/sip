/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// ===========================================================================
// FftShift2DKernel — 2D fftshift / ifftshift ([b1, fftN1, fftN2] = [*, 500, 50]
// complex64), AIV-only (__vector__) SIMD kernel.
//
//   Per slice: split the [500, 100] float view into two halves of 250 full rows.
//   Each half is moved GM<->UB as one contiguous 100KB burst (HBM-friendly).
//   The per-row [L|R] -> [R|L] swap is done *inside UB* with the RegBase
//   unaligned load/store (asc_vf_call + Load/Store wrappers).
//
//   The two halves are ping-pong double-buffered: while MTE3 stores half h and
//   the VF swaps half h, MTE2 prefetches half h+1 into the other UB buffer, so
//   the RegBase reorder latency is hidden behind the contiguous GM traffic.
//
//   The inverse (ifftshift) direction applies a 1/N scale in UB via Muls.
// ===========================================================================

#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/status.hpp"

#include "fft_shift_kernel.h"
#include "fft_types.h"

#include "kernel_operator.h"

// helper.hpp LAST — it includes <opdev/fp16_t.h> defining macros that conflict
// with kernel headers already included above.
#include "helper.hpp"

using namespace Catlass;
using namespace AscendC;

namespace Fft {

constexpr uint32_t SHIFT_ROWS = 250;                      // rows per half (fftN1 / 2)
constexpr uint32_t SHIFT_COLS = 50;                       // floats per half-row (fftN2)
constexpr uint32_t ROW_FLOATS = 2 * SHIFT_COLS;           // 100 (full row)
constexpr uint32_t HALF_FLOATS = SHIFT_ROWS * ROW_FLOATS; // 25000 (100KB)
constexpr uint32_t HALF_PAD_FLOATS = HALF_FLOATS + 64;    // pad for unaligned over-read

// ===========================================================================
// RegBase VF: swap the two 50-float halves of every 100-float row in place.
//   row[0..49] = L, row[50..99] = R  ->  row[0..49] = R, row[50..99] = L
// The RegTensor is VL=256B (64 floats), so a half (50 floats) is one
// unaligned Load (64) + one partial Store (count=50).
// ===========================================================================
__simd_vf__ inline void SwapHalvesVF(__ubuf__ float* buf, uint32_t rows)
{
    AscendC::Reg::RegTensor<float> regL;
    AscendC::Reg::RegTensor<float> regR;

    for (uint32_t j = 0; j < rows; ++j) {
        __ubuf__ float* row = buf + j * ROW_FLOATS;
        AscendC::Reg::Load(regL, row);                           // L (first SHIFT_COLS valid)
        AscendC::Reg::Load(regR, row + SHIFT_COLS);              // R (first SHIFT_COLS valid)
        AscendC::Reg::Store(row, regR, SHIFT_COLS);              // R -> [0, 50)
        AscendC::Reg::Store(row + SHIFT_COLS, regL, SHIFT_COLS); // L -> [50, 100)
    }
}

__global__ __vector__ void FftShift2DKernel(GM_ADDR input, GM_ADDR output, GM_ADDR tiling)
{
    AscendC::InitSocState();
    __gm__ FftShiftTilingData* td = (__gm__ FftShiftTilingData*)tiling;
    const int64_t b1 = td->b1;
    const uint32_t coreNum = td->coreNum;
    const int32_t inverse = td->inverse;

    const uint32_t blockIdx = AscendC::GetBlockIdx();
    if (blockIdx >= coreNum) {
        return;
    }

    int64_t matStart = 0;
    int64_t matEnd = 0;
    FFT89_CORE_MAT_RANGE(blockIdx, b1, coreNum, matStart, matEnd);

    GlobalTensor<float> gmSrc;
    gmSrc.SetGlobalBuffer((__gm__ float*)input);
    GlobalTensor<float> gmDst;
    gmDst.SetGlobalBuffer((__gm__ float*)output);

    LocalMemAllocator<Hardware::UB> ubAllocator;
    LocalTensor<float> ubHalf[2];
    ubHalf[0] = ubAllocator.Alloc<float, HALF_PAD_FLOATS>();
    ubHalf[1] = ubAllocator.Alloc<float, HALF_PAD_FLOATS>();
    __ubuf__ float* halfAddr[2];
    halfAddr[0] = reinterpret_cast<__ubuf__ float*>(ubHalf[0].GetPhyAddr());
    halfAddr[1] = reinterpret_cast<__ubuf__ float*>(ubHalf[1].GetPhyAddr());

    const float scale = inverse ? (1.0f / static_cast<float>(FFT_N1_DEFAULT * FFT_N2_DEFAULT)) : 1.0f;

    // Per-matrix half source/destination offsets (in floats, within GM_MAT_FLOATS).
    const uint32_t srcOff[2] = {0, 25000};
    const uint32_t dstOff[2] = {25000, 0};

    const int64_t numChunks = 2 * (matEnd - matStart);

    // Mark both buffers initially free (store completed).
    SetFlag<HardEvent::MTE3_MTE2>(0);
    SetFlag<HardEvent::MTE3_MTE2>(1);

    // Prologue: load the first half (matStart, top half) into buffer 0.
    {
        const uint32_t base = static_cast<uint32_t>(matStart * GM_MAT_FLOATS);
        WaitFlag<HardEvent::MTE3_MTE2>(0);
        DataCopy(ubHalf[0], gmSrc[base + srcOff[0]], HALF_FLOATS);
        SetFlag<HardEvent::MTE2_V>(0);
    }

    for (int64_t t = 0; t < numChunks; ++t) {
        const uint32_t cur = t & 1;
        const uint32_t nxt = (t + 1) & 1;
        const int64_t m = matStart + t / 2;
        const uint32_t h = t % 2;
        const uint32_t base = static_cast<uint32_t>(m * GM_MAT_FLOATS);

        // Wait current buffer loaded.
        WaitFlag<HardEvent::MTE2_V>(cur);

        // Prefetch the next half into the other buffer (overlaps reorder+store).
        if (t + 1 < numChunks) {
            const int64_t mn = matStart + (t + 1) / 2;
            const uint32_t hn = (t + 1) % 2;
            const uint32_t basen = static_cast<uint32_t>(mn * GM_MAT_FLOATS);
            WaitFlag<HardEvent::MTE3_MTE2>(nxt);
            DataCopy(ubHalf[nxt], gmSrc[basen + srcOff[hn]], HALF_FLOATS);
            SetFlag<HardEvent::MTE2_V>(nxt);
        }

        // Reorder current buffer in place (V): scale then RegBase swap.
        Muls(ubHalf[cur], ubHalf[cur], scale, HALF_FLOATS);
        PipeBarrier<PIPE_V>();
        asc_vf_call<SwapHalvesVF>(halfAddr[cur], SHIFT_ROWS);
        AscendC::DataSyncBarrier<MemDsbT::UB>();

        // Store current buffer (MTE3).
        SetFlag<HardEvent::V_MTE3>(cur);
        WaitFlag<HardEvent::V_MTE3>(cur);
        DataCopy(gmDst[base + dstOff[h]], ubHalf[cur], HALF_FLOATS);
        SetFlag<HardEvent::MTE3_MTE2>(cur);
    }
}

// Host-side launch wrapper
void FftShift2DKernelLaunch(GM_ADDR input, GM_ADDR output, GM_ADDR tiling, uint32_t blockDim, aclrtStream stream)
{
    constexpr uint32_t dynUbSize = 2 * HALF_PAD_FLOATS * sizeof(float);
    FftShift2DKernel<<<blockDim, dynUbSize, stream>>>(input, output, tiling);
}

} // namespace Fft
