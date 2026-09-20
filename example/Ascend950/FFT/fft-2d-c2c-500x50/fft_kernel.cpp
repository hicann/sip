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
// FftC2C2DFusedKernel — 2D C2C FFT ([b1, fftN1, fftN2] = [12000, 500, 50],
// axes (2,3)) fused in ONE mix kernel:
//
//   AIC  (Stage R): per matrix, TWO M-half GEMMs
//       Y[Mh, 104] = X[Mh, 100] @ W50[100, 104]
//     whose Fixpipe L0C->UB egress (NZ2ND + subBlockId routing) writes the
//     [500, 104] row-major tile straight into the OWNER AIV's UB bank
//     (half 0 -> tile rows [0,250), half 1 -> rows [250,500)).
//   AIV (Stage C): per owned matrix, ONE asc_vf_call of simt_fft_500x50 —
//     the 4-stage in-place DIF column FFT + digit-reversal GM scatter.
//
// Work split: b1 balanced across CoreNum blocks (1 AIC + 2 AIV); within a
// block, matrix m routes to AIV (m & 1) — matrix-level pingpong.  Every
// AIV's GM output rows are exclusively owned -> no workspace, no SyncAll,
// no cross-core data traffic beyond the flag choreography.
//
// Block-id derivation (mode 4, ex83-3 InitContext pattern): the AIC's
// GetBlockIdx() is the core index directly; an AIV's core index is
// GetBlockIdx() >> 1.
// ===========================================================================

#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include "catlass/arch/arch.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/catlass.hpp"
#include "catlass/status.hpp"

#include "fft_kernel.h"
#include "fft_kernel_utils.h"
#include "fft_simt_ops.h"
#include "fft_tiling_def.h"
#include "fft_types.h"

#include "kernel_operator.h"

// helper.hpp LAST — it includes <opdev/fp16_t.h> defining macros that conflict
// with kernel headers already included above.
#include "helper.hpp"

using namespace Catlass;
using namespace AscendC;

namespace Fft {

using ArchTag = Arch::Ascend950;

CATLASS_GLOBAL __mix__(1, 2) void FftC2C2DFusedKernel(GM_ADDR input, GM_ADDR output, GM_ADDR coeffs, GM_ADDR tiling)
{
    __gm__ FftTilingData* td = (__gm__ FftTilingData*)tiling;
    const int64_t b1 = td->b1;
    const uint32_t coreNum = td->coreNum;

    // Block id: AIC -> raw GetBlockIdx(); AIV -> GetBlockIdx() >> 1
    // (mode-4 interleaving; ex83-3 InitContext / ex89 pattern).
    uint32_t blockIdx;
    if ASCEND_IS_AIC {
        blockIdx = AscendC::GetBlockIdx();
    } else {
        blockIdx = AscendC::GetBlockIdx() >> 1;
    }
    if (blockIdx >= coreNum) {
        return;
    }

    // ---- this block's matrix range (same formula as host tiling) ----
    // Macro (not CoreMatrixRange()): the __global__ __aicore__ body
    // cannot call a plain __host__ inline function.
    int64_t matStart = 0;
    int64_t matEnd = 0;
    FFT89_CORE_MAT_RANGE(blockIdx, b1, coreNum, matStart, matEnd);
    const int64_t matCount = matEnd - matStart;

    // +1 forward / -1 inverse: selects the symmetric-DFT-5 butterfly's
    // output sign pattern (ex89 convention).
    const int32_t dirSign = (td->direction == 1) ? -1 : 1;

    if ASCEND_IS_AIC {
        // ===================================================================
        // AIC: Stage R pipeline — walk matrices in local order, route each
        // to the owner AIV (li & 1).  For li >= 2, wait until the owner
        // freed the UB bank two matrices back (the pingpong period).
        // ===================================================================
        Arch::Resource<ArchTag> resource;
        {
            FftBlockMmadToUb blockMmad(resource, 0);

            // The AIC addresses UB relative to its own sub-block view; the
            // Fixpipe hardware redirects writes to the owner AIV's bank via
            // subBlockId (ex83-3's dual-AIV RunGemmToUb pattern).
            LocalTensor<float> ubTile = resource.ubBuf.template GetBufferByByte<float>(0);

            // RunGemmToUb takes FLOAT offsets (ex83-3 InitContext converts
            // the tiling's byte offsets once, up front — same here).
            const int64_t w50FloatOff = td->w50Offset / static_cast<int64_t>(sizeof(float));

            for (int64_t li = 0; li < matCount; ++li) {
                const uint32_t owner = static_cast<uint32_t>(li & 1);
                const int64_t matFloatOff = (matStart + li) * GM_MAT_FLOATS;

                // The owner AIV finished with this UB bank two matrices ago
                // (level: one set enables exactly one wait, 83-2/83-3
                // software-pipeline flag discipline).
                if (li >= 2) {
                    AscendC::CrossCoreWaitFlag<FFT_SYNC_MODE, PIPE_FIX>(FFT_SYNC_UB_FREE[owner]);
                }

                // Half 0: input rows [0, 250) -> tile rows [0, 250).
                RunGemmToUb(blockMmad, input, matFloatOff, coeffs, w50FloatOff, ubTile, M_HALF, GEMM_N, WS_STRIDE,
                            WS_STRIDE,
                            /*cRowOffset*/ 0, owner != 0);

                // Half 1: input rows [250, 500) -> tile rows [250, 500).
                RunGemmToUb(blockMmad, input, matFloatOff + static_cast<int64_t>(M_HALF) * GEMM_N, coeffs, w50FloatOff,
                            ubTile, M_HALF, GEMM_N, WS_STRIDE, WS_STRIDE,
                            /*cRowOffset*/ M_HALF, owner != 0);

                // Both halves fixed into the owner's UB — notify.
                AscendC::CrossCoreSetFlag<FFT_SYNC_MODE, PIPE_FIX>(FFT_SYNC_GEMM_DONE[owner]);
            }
        } // blockMmad destructed (drains all pipe events)

        AscendC::PipeBarrier<PIPE_FIX>();
        return;
    }

    if ASCEND_IS_AIV {
        // ===================================================================
        // AIV: Stage C — consume owned matrices.  subIdx selects the UB bank
        // this AIV's Fixpipe writes land in (matches the AIC's owner route).
        // ===================================================================
        const uint32_t subIdx = AscendC::GetSubBlockIdx(); // 0=AIV0, 1=AIV1
        Arch::Resource<ArchTag> resource;

        // UB layout (floats):
        //   [0, 52000):   ubTile — [500, 104] fused GEMM output tile
        //   [52000, 53040): ubTw  — 1040-float twiddle table
        // Total 53040 floats = 212,160B < 216KB.
        LocalTensor<float> ubTile = resource.ubBuf.template GetBufferByByte<float>(0);
        LocalTensor<float> ubTw = resource.ubBuf.template GetBufferByByte<float>(TILE_FLOATS * sizeof(float));

        __gm__ float* gmOut = (__gm__ float*)(output);
        __gm__ float* gmCoeffs = (__gm__ float*)(coeffs);

        // ---- one-time twiddle load (direction-selected table) ----
        {
            GlobalTensor<float> gmTw;
            gmTw.SetGlobalBuffer(gmCoeffs + td->twiddleOffset / static_cast<int64_t>(sizeof(float)));
            DataCopy(ubTw, gmTw, static_cast<int32_t>(COEFF_FLOATS_PER_DIR));
            SetFlag<HardEvent::MTE2_V>(0);
            WaitFlag<HardEvent::MTE2_V>(0);
        }

        // This AIV owns local matrices with (li & 1) == subIdx.
        for (int64_t li = subIdx; li < matCount; li += 2) {
            const int64_t m = matStart + li;
            const int64_t matFloatOff = m * GM_MAT_FLOATS;

            // Wait for the AIC's GEMM+Fixpipe of BOTH halves (own-namespace
            // id — the owner routing is encoded in the AIC's set id).
            AscendC::CrossCoreWaitFlag<FFT_SYNC_MODE, PIPE_V>(FFT_SYNC_AIV_WAIT_DONE);

            {
                AscendC::PipeBarrier<PIPE_V>();
                __ubuf__ float* tileAddr = (__ubuf__ float*)ubTile.GetPhyAddr();
                __ubuf__ float* twAddr = (__ubuf__ float*)ubTw.GetPhyAddr();
                // Direction is fixed per launch: dispatch the compile-time
                // sign instantiation OUTSIDE the SIMT body (the +/- folds
                // into the butterfly arithmetic; no runtime branch inside).
                if (dirSign > 0) {
                    asc_vf_call<simt_fft_500x50_impl<1>>(dim3(SIMT_THREADS), tileAddr, twAddr, gmOut + matFloatOff);
                } else {
                    asc_vf_call<simt_fft_500x50_impl<-1>>(dim3(SIMT_THREADS), tileAddr, twAddr, gmOut + matFloatOff);
                }
                AscendC::DataSyncBarrier<MemDsbT::UB>();

                // Order the SIMT (V-pipe) tile reads before the cross-core
                // FREE flag: the flag-set is enqueued on MTE3, so first drain
                // V into an MTE3-blocking event (ex83-3 Pass-2 Step B+C tail).
                SetFlag<HardEvent::V_MTE3>(0);
                WaitFlag<HardEvent::V_MTE3>(0);
            }

            // Notify the AIC: this UB bank is free.
            AscendC::CrossCoreSetFlag<FFT_SYNC_MODE, PIPE_MTE3>(FFT_SYNC_AIV_SET_FREE);
        }

        AscendC::PipeBarrier<PIPE_MTE3>();
    }
}

// Host-side launch wrapper
void FftC2C2DFusedKernelLaunch(GM_ADDR input, GM_ADDR output, GM_ADDR coeffs, GM_ADDR tiling, uint32_t blockDim,
                               aclrtStream stream)
{
    constexpr uint32_t workBuff = 216 * 1024;
    FftC2C2DFusedKernel<<<blockDim, workBuff, stream>>>(input, output, coeffs, tiling);
}

} // namespace Fft
