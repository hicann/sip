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

#ifndef FFT_TYPES_H
#define FFT_TYPES_H

#include <cstdint>
#include <cmath>

namespace Fft {

constexpr double FFT_PI = 3.14159265358979323846;
constexpr double FFT_2PI = 2.0 * FFT_PI;

// ===========================================================================
// Problem shape: [b1, fftN1, fftN2] = [12000, 500, 50] complex64 (interleaved
// [re, im]).  2D C2C FFT over axes (1, 2), fused inside ONE mix kernel:
//
//   Stage R (AIC, ex88 embedded-rotation GEMM):
//       row FFT over fftN2=50 (the last dim).  Each [500, 50] matrix is
//       TWO M-half GEMMs:
//           Y[Mhalf, 104] = X[Mhalf, 100] @ W[100, 104]
//       W  = ex88's rotation-embedded [100, 100] matrix padded with 4 zero
//            columns to 104 (8-float alignment for the Fixpipe nSize).
//       Mhalf = 250 (<= L1_TILE_M=256 -> single M-tile per GEMM call, no
//            M_L0_LOOP, avoiding the WaitFlag<FIX_M>-inside-loop hazard).
//       The Fixpipe L0C->UB egress (NZ2ND) writes each half's [250, 104]
//       result row-major into the target AIV's UB tile: half 0 -> tile rows
//       [0, 250), half 1 -> tile rows [250, 500).
//
//   Stage C (AIV, ex89 pure-SIMT port):
//       column FFT over fftN1=500 on the UB tile [500, 104] (50 valid
//       interleaved columns = the 50 row-FFT outputs per matrix row,
//       columns [100, 104) are fixpipe padding).  4-stage in-place DIF
//       (radices 4,5,5,5) + digit-reversal folded into the GM scatter.
//
// Parallelization: b1 split across CoreNum blocks (1 AIC + 2 AIV each);
// within a block, matrices pingpong AIV0/AIV1 by parity (NO_SPLIT_SUBBLOCK
// Fixpipe routes the whole [250,104] half to the owner AIV's UB bank).
// No workspace, no SyncAll: every AIV's GM output region is disjoint.
// ===========================================================================

constexpr int32_t FFT_N1_DEFAULT = 500; // middle dim: column FFT (SIMT)
constexpr int32_t FFT_N2_DEFAULT = 50;  // last dim: row FFT (GEMM)
constexpr int32_t B1_DEFAULT = 12000;   // leading batch dim (split by core)

// ---- Stage R (GEMM) geometry ----
constexpr int32_t GEMM_N = 2 * FFT_N2_DEFAULT;                                 // 100 real cols (= K)
constexpr int32_t WS_STRIDE = 104;                                             // padded row pitch (floats);
                                                                               // 104 = RoundUp(100, 8): Fixpipe
                                                                               // nSize must be 8-float aligned
constexpr int32_t M_HALF = 250;                                                // rows per GEMM call (2 halves)
constexpr int32_t W_FLOATS_PER_DIR = static_cast<int64_t>(GEMM_N) * WS_STRIDE; // 100*104

// ---- Stage C (SIMT) geometry — ex89 port with a wider row ----
constexpr int32_t B2H = FFT_N2_DEFAULT;                      // 50 valid complex cols/tile row
constexpr int32_t ROW_STRIDE = WS_STRIDE;                    // 104 (must equal fixpipe dstStride)
constexpr int32_t TILE_FLOATS = FFT_N1_DEFAULT * ROW_STRIDE; // 52000 (208KB)
constexpr int32_t TILE_BYTES = TILE_FLOATS * 4;              // 208000B < 216KB UB

// ---- GM geometry (floats): element (b, i1, i2) at 2*(b*fftN1*fftN2 + i1*fftN2 + i2)
constexpr int32_t GM_ROW_FLOATS = 2 * FFT_N2_DEFAULT;             // 100
constexpr int32_t GM_MAT_FLOATS = FFT_N1_DEFAULT * GM_ROW_FLOATS; // 50000

// ---------------------------------------------------------------------------
// Twiddle tables for the column FFT (ex89 verbatim; both directions,
// 1040 floats each, interleaved [re, im]):
//   r4[125]   W_500^j          radix-4 stage (X1 twiddle)
//   r4[125]   W_250^j          radix-4 stage (X2 = W_500^(2j))
//   r4[125]   W_500^(3*j)      radix-4 stage (X3 twiddle)
//   tw3[100]  W_125^(j*t)      radix-5 stage (L=125), layout [j][t-1], t=1..4
//   tw4[20]   W_25^(j*t)       radix-5 stage (L=25),  layout [j][t-1], t=1..4
//   w5[25]    W_5^(s*t)        radix-5 DFT matrix (unused by SIMT device)
// ---------------------------------------------------------------------------
constexpr int32_t R4_T1_COUNT = 125;
constexpr int32_t R4_T2_COUNT = 125;
constexpr int32_t R4_T3_COUNT = 125;
constexpr int32_t TW3_COUNT = 100;
constexpr int32_t TW4_COUNT = 20;
constexpr int32_t W5_COUNT = 25;

constexpr int32_t R4_T1_OFF = 0;
constexpr int32_t R4_T2_OFF = R4_T1_OFF + 2 * R4_T1_COUNT;      // 250
constexpr int32_t R4_T3_OFF = R4_T2_OFF + 2 * R4_T2_COUNT;      // 500
constexpr int32_t TW3_OFF = R4_T3_OFF + 2 * R4_T3_COUNT;        // 750
constexpr int32_t TW4_OFF = TW3_OFF + 2 * TW3_COUNT;            // 950
constexpr int32_t W5_OFF = TW4_OFF + 2 * TW4_COUNT;             // 990
constexpr int32_t COEFF_FLOATS_PER_DIR = W5_OFF + 2 * W5_COUNT; // 1040

// Coefficient buffer layout (floats):
//   [0, 2*1040)                                twiddles fwd/inv (Stage C)
//   [2*1040, 2*1040 + 100*104)                 W50 forward  (Stage R, row-major)
//   [2*1040 + 100*104, 2*1040 + 2*100*104)     W50 inverse
constexpr int64_t TW_TOTAL_FLOATS = 2 * COEFF_FLOATS_PER_DIR; // 2080
constexpr int64_t W_FWD_OFF = TW_TOTAL_FLOATS;                // 2080
constexpr int64_t W_INV_OFF = W_FWD_OFF + W_FLOATS_PER_DIR;   // 12480

// ---------------------------------------------------------------------------
// Digit reversal for the column-FFT stage order (4,5,5,5) — ex89 verbatim:
//   position p = t1*125 + t2*25 + t3*5 + t4  holds X[k],
//   k = t1 + 4*t2 + 20*t3 + 100*t4  (t1 in [0,4), t2..t4 in [0,5)).
// Macro (not inline fn): __simt_vf__ bodies may only call __simt_callee__
// functions, so device code expands the macro while host tests use
// DigitReverse500() — single source of truth.
// ---------------------------------------------------------------------------
#define FFT89_DIGREV500(p) (((p) / 125) + 4 * (((p) / 25) % 5) + 20 * (((p) / 5) % 5) + 100 * ((p) % 5))

inline uint32_t DigitReverse500(uint32_t p) { return FFT89_DIGREV500(p); }

// Balanced b1 split across cores (ex89 verbatim; also a MACRO because the
// __global__ __aicore__ body cannot call plain __host__ inline functions).
#define FFT89_CORE_MAT_RANGE(core_, b1_, coreNum_, matStart_, matEnd_) \
    do {                                                               \
        const int64_t per_ = (b1_) / static_cast<int64_t>(coreNum_);   \
        const int64_t rem_ = (b1_) % static_cast<int64_t>(coreNum_);   \
        const int64_t c_ = static_cast<int64_t>(core_);                \
        (matStart_) = c_ * per_ + ((c_ < rem_) ? c_ : rem_);           \
        (matEnd_) = (matStart_) + per_ + ((c_ < rem_) ? 1 : 0);        \
    } while (0)

inline void CoreMatrixRange(uint32_t core, int64_t b1, uint32_t coreNum, int64_t& matStart, int64_t& matEnd)
{
    FFT89_CORE_MAT_RANGE(core, b1, coreNum, matStart, matEnd);
}

inline int32_t CeilDiv(int32_t a, int32_t b) { return (a + b - 1) / b; }

inline int64_t CeilDiv(int64_t a, int64_t b) { return (a + b - 1) / b; }

} // namespace Fft

#endif // FFT_TYPES_H
