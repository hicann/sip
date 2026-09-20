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

#ifndef FFT_KERNEL_UTILS_H
#define FFT_KERNEL_UTILS_H

#ifdef ENABLE_CV_COMM_VIA_SSBUF
#pragma message("FFT90: ENABLE_CV_COMM_VIA_SSBUF is DEFINED")
#else
#pragma message("FFT90: ENABLE_CV_COMM_VIA_SSBUF is NOT DEFINED")
#endif

#include "catlass/catlass.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/status.hpp"
#include "fft_tiling_def.h"
#include "fft_types.h"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

using namespace Catlass;
using namespace AscendC;

namespace Fft {

// ---------------------------------------------------------------------------
// Cross-core sync (mode 4: AIC <-> AIV point-to-point, Ascend 950).
// ID ROUTING (ex83-3/ex86 NPU-verified semantics):
//   * AIC set(F)      -> AIV0's own flag F        (F in [0,16))
//   * AIC set(F+16)   -> AIV1's own flag F        (the +16 is the ROUTE)
//   * AIV0 set(F)     -> AIC's flag F
//   * AIV1 set(F)     -> AIC's flag F+16          (hardware applies the offset)
// So the AIV side ALWAYS waits/sets its own [0,16) ids; the owner routing is
// encoded entirely in the AIC-side id (0 vs +16).
//
// Choreography (per matrix m, owner AIV = m & 1):
//   AIC: RunGemmToUb x2 (both M-halves, routed to owner's UB bank)
//        -> CrossCoreSetFlag<PIPE_FIX>(GEMM_DONE[owner])
//   AIV(owner): CrossCoreWaitFlag<PIPE_V>(AIV_WAIT_DONE)
//        -> simt_fft_500x50 (column FFT + GM scatter)
//        -> V_MTE3 drain -> CrossCoreSetFlag<PIPE_MTE3>(AIV_SET_FREE)
//   AIC (m >= 2): CrossCoreWaitFlag<PIPE_FIX>(UB_FREE[owner]) before the
//        fixpipe of matrix m-2 that reuses the same UB bank.
// ---------------------------------------------------------------------------
constexpr uint64_t FFT_SYNC_MODE = 4;

// AIC-side ids (route included): set GEMM_DONE[owner] after both halves'
// fixpipe; wait UB_FREE[owner] before reusing the owner's UB bank.
constexpr uint64_t FFT_SYNC_GEMM_DONE[2] = {0, 16};
constexpr uint64_t FFT_SYNC_UB_FREE[2] = {4, 20};

// AIV-side ids (own namespace — hardware applies the +16 offset for AIV1):
constexpr uint64_t FFT_SYNC_AIV_WAIT_DONE = 0; // AIC's GEMM_DONE arrives here
constexpr uint64_t FFT_SYNC_AIV_SET_FREE = 4;  // AIC sees 4 (AIV0) / 20 (AIV1)

// ---------------------------------------------------------------------------
// GEMM type definitions — ex88's dispatch policy and tile shapes verbatim
// (single-B-tile, L1-resident W; L0C_STAGES=2 double-buffers the [256,128]
// L0C tile so back-to-back half-GEMMs overlap fixpipe with MMAD).
// ---------------------------------------------------------------------------
using FftElement = float;
using FftLayoutTag = layout::RowMajor;
using FftArchTag = Arch::Ascend950;

using FftDispatchPolicy = Gemm::MmadPingpong<FftArchTag, /*enableUnitFlag*/ false, /*useHF32*/ false, /*l0CStages*/ 2,
                                             /*enableL1Resident*/ true, /*l1AStages*/ 2, /*l1BStages*/ 1,
                                             /*l0AStages*/ 2, /*l0BStages*/ 2>;

using FftL1TileShape = tla::Shape<tla::Int<256>, tla::Int<128>, tla::Int<128>>;
using FftL0TileShape = tla::Shape<tla::Int<256>, tla::Int<128>, tla::Int<32>>;

// L0C->UB Fixpipe with caller-specified subBlockId routing (NO_SPLIT_SUBBLOCK):
// the AIC addresses UB relative to ITS sub-block view and the hardware
// redirects the write to the owner AIV's private UB bank.
using FftTileCopyToUb = Gemm::Tile::PackedTileCopyTlaToUB<FftArchTag, FftElement, FftLayoutTag, FftElement,
                                                          FftLayoutTag, FftElement, FftLayoutTag, void,
                                                          Gemm::Tile::CopyL0CToUBMode::NO_SPLIT_SUBBLOCK>;

using FftBlockMmadToUb = Gemm::Block::BlockMmadTla<FftDispatchPolicy, FftL1TileShape, FftL0TileShape, FftElement,
                                                   FftElement, FftElement, void, FftTileCopyToUb>;

// ---------------------------------------------------------------------------
// RunGemmToUb — C[M x N] = A[M x K] x B[K, N], output to UB via Fixpipe
// (83-3's RunGemmToUb pattern).  A = the input matrix half read straight
// from GM (row-major, K=100); B = the padded W50 [100, 104] (row-major);
// C lands row-major [M, 104] in the owner AIV's UB bank (subBlockId), at
// row cRowOffset of the [500, 104] tile.
//
// M is fixed at M_HALF=250 (<= L1_TILE_M=256): one L1 tile per operand, no
// M_L0_LOOP, K=100 splits as 32/32/32/4 on the L0 tile.  N=104 (= WS_STRIDE,
// includes W's 4 zero pad columns — the same pass-the-pad GEMM pattern as
// ex83-3's KPad=104 B operand), so the fixpipe's pad lanes are deterministic
// zeros rather than uninitialized L0C fractal data.
// ---------------------------------------------------------------------------
CATLASS_DEVICE void RunGemmToUb(FftBlockMmadToUb& blockMmad, GM_ADDR gmA, int64_t aOffset, GM_ADDR gmB, int64_t bOffset,
                                LocalTensor<FftElement>& ubC, int32_t M, int32_t K, int32_t N, int32_t layoutN,
                                int32_t cRowOffset, bool subBlockId = false)
{
    GlobalTensor<FftElement> gmA_tensor;
    gmA_tensor.SetGlobalBuffer((__gm__ FftElement*)(gmA) + aOffset);
    GlobalTensor<FftElement> gmB_tensor;
    gmB_tensor.SetGlobalBuffer((__gm__ FftElement*)(gmB) + bOffset);

    auto layoutA = tla::MakeLayout<FftElement, FftLayoutTag>(static_cast<uint32_t>(M), static_cast<uint32_t>(K));
    auto layoutB = tla::MakeLayout<FftElement, FftLayoutTag>(static_cast<uint32_t>(K), static_cast<uint32_t>(layoutN));
    // The C tensor spans the FULL tile [500, 104]; the tile anchor at row
    // cRowOffset places the fixpipe dst at tile row cRowOffset.
    auto layoutC = tla::MakeLayout<FftElement, FftLayoutTag>(static_cast<uint32_t>(FFT_N1_DEFAULT),
                                                             static_cast<uint32_t>(layoutN));

    auto tensorA = tla::MakeTensor(gmA_tensor, layoutA, Arch::PositionGM{});
    auto tensorB = tla::MakeTensor(gmB_tensor, layoutB, Arch::PositionGM{});
    auto tensorC = tla::MakeTensor(ubC, layoutC, Arch::PositionUB{});

    GemmCoord actualShape(static_cast<uint32_t>(M), static_cast<uint32_t>(N), static_cast<uint32_t>(K));

    auto tensorBlockA = GetTile(tensorA, tla::MakeCoord(0u, 0u),
                                tla::MakeShape(static_cast<uint32_t>(M), static_cast<uint32_t>(K)));
    auto tensorBlockB = GetTile(tensorB, tla::MakeCoord(0u, 0u),
                                tla::MakeShape(static_cast<uint32_t>(K), static_cast<uint32_t>(layoutN)));
    auto tensorBlockC = GetTile(tensorC, tla::MakeCoord(static_cast<uint32_t>(cRowOffset), 0u),
                                tla::MakeShape(static_cast<uint32_t>(M), static_cast<uint32_t>(N)));
    blockMmad(tensorBlockA, tensorBlockB, tensorBlockC, actualShape, {}, subBlockId);
}

} // namespace Fft

#endif // FFT_KERNEL_UTILS_H
