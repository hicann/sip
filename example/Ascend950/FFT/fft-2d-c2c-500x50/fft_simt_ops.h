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

#ifndef FFT_SIMT_OPS_H
#define FFT_SIMT_OPS_H

#ifdef FFT90_HOST_SIM
#include <cstdint>
struct Fft90ThreadIdx {
    uint32_t x;
};
namespace Fft {
extern thread_local Fft90ThreadIdx threadIdx;
void asc_syncthreads();
struct float2 {
    float x, y;
};
inline float2 make_float2(float a, float b) { return float2{a, b}; }
} // namespace Fft
#define __simt_vf__
#define __launch_bounds__(x)
#define __ubuf__
#define __gm__
#else
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "simt_api/vector_functions.h"

#include "catlass/catlass.hpp"
#include "catlass/status.hpp"
#include "fft_types.h"

using namespace Catlass;
using namespace AscendC;
#endif

#include "fft_types.h"

namespace Fft {

constexpr uint32_t SIMT_THREADS = 256;

// Symmetric DFT-5 butterfly constants (ex89 R1).
constexpr float R5_C72 = 0.30901699437494745f;
constexpr float R5_C144 = -0.80901699494794734f;
constexpr float R5_S72 = 0.95105651629515357f;
constexpr float R5_S144 = 0.58778525229247314f;

// ===========================================================================
// Register-fused 2-stage column FFT (radix-20 + radix-25), reusing the SAME
// radix 4/5/5/5 DIF decomposition, twiddle table and digit-reversal as the
// 4-stage version.  Stages 1+2 (radix-4 + radix-5) and 3+4 (radix-5 + radix-5)
// are fused IN REGISTERS (no UB write/read between them), cutting the UB
// traffic from 4 reads + 3 writes to 2 reads + 1 write.
//
//   data : UB tile [500, ROW_STRIDE=104] floats, row-major, interleaved.
//   tw   : UB twiddle table (unchanged 1040 floats, ex89 layout).
//   gmOut: GM base of THIS matrix.
//   SGN  : +1 forward / -1 inverse (compile-time).
// ===========================================================================

// Symmetric DFT-5 on 5 named complex values (in place).  Output order 0,1,4,2,3.
#define FFT90_DFT5(x0r, x0i, x1r, x1i, x2r, x2i, x3r, x3i, x4r, x4i)          \
    do {                                                                      \
        const float s1r = (x1r) + (x4r), s1i = (x1i) + (x4i);                 \
        const float s2r = (x2r) + (x3r), s2i = (x2i) + (x3i);                 \
        const float d1r = (x1r) - (x4r), d1i = (x1i) - (x4i);                 \
        const float d2r = (x2r) - (x3r), d2i = (x2i) - (x3i);                 \
        const float A1 = R5_C72 * s1r + R5_C144 * s2r;                        \
        const float B1 = R5_C72 * s1i + R5_C144 * s2i;                        \
        const float D1 = R5_S72 * d1r + R5_S144 * d2r;                        \
        const float E1 = R5_S72 * d1i + R5_S144 * d2i;                        \
        const float A2 = R5_C144 * s1r + R5_C72 * s2r;                        \
        const float B2 = R5_C144 * s1i + R5_C72 * s2i;                        \
        const float D2 = R5_S144 * d1r - R5_S72 * d2r;                        \
        const float E2 = R5_S144 * d1i - R5_S72 * d2i;                        \
        const float t0r = (x0r) + s1r + s2r, t0i = (x0i) + s1i + s2i;         \
        const float t1r = (x0r) + A1 + sgn * E1, t1i = (x0i) + B1 - sgn * D1; \
        const float t4r = (x0r) + A1 - sgn * E1, t4i = (x0i) + B1 + sgn * D1; \
        const float t2r = (x0r) + A2 + sgn * E2, t2i = (x0i) + B2 - sgn * D2; \
        const float t3r = (x0r) + A2 - sgn * E2, t3i = (x0i) + B2 + sgn * D2; \
        (x0r) = t0r;                                                          \
        (x0i) = t0i;                                                          \
        (x1r) = t1r;                                                          \
        (x1i) = t1i;                                                          \
        (x4r) = t4r;                                                          \
        (x4i) = t4i;                                                          \
        (x2r) = t2r;                                                          \
        (x2i) = t2i;                                                          \
        (x3r) = t3r;                                                          \
        (x3i) = t3i;                                                          \
    } while (0)

// radix-4 DIF on 4 named complex values (in place), with 3 twiddles.
#define FFT90_R4(x0r, x0i, x1r, x1i, x2r, x2i, x3r, x3i, w1r, w1i, w2r, w2i, w3r, w3i) \
    do {                                                                               \
        const float a0r = (x0r) + (x2r), a0i = (x0i) + (x2i);                          \
        const float a1r = (x0r) - (x2r), a1i = (x0i) - (x2i);                          \
        const float a2r = (x1r) + (x3r), a2i = (x1i) + (x3i);                          \
        const float a3r = (x1r) - (x3r), a3i = (x1i) - (x3i);                          \
        const float y0r = a0r + a2r, y0i = a0i + a2i;                                  \
        const float b1r = a1r + sgn * a3i, b1i = a1i - sgn * a3r;                      \
        const float b3r = a1r - sgn * a3i, b3i = a1i + sgn * a3r;                      \
        const float b2r = a0r - a2r, b2i = a0i - a2i;                                  \
        const float y1r = b1r * (w1r) - b1i * (w1i), y1i = b1r * (w1i) + b1i * (w1r);  \
        const float y2r = b2r * (w2r) - b2i * (w2i), y2i = b2r * (w2i) + b2i * (w2r);  \
        const float y3r = b3r * (w3r) - b3i * (w3i), y3i = b3r * (w3i) + b3i * (w3r);  \
        (x0r) = y0r;                                                                   \
        (x0i) = y0i;                                                                   \
        (x1r) = y1r;                                                                   \
        (x1i) = y1i;                                                                   \
        (x2r) = y2r;                                                                   \
        (x2i) = y2i;                                                                   \
        (x3r) = y3r;                                                                   \
        (x3i) = y3i;                                                                   \
    } while (0)

// complex multiply by twiddle (cr, ci) := (cr,ci) * (tr,ti)
#define FFT90_TWMUL(cr, ci, tr, ti)                 \
    do {                                            \
        const float zr = (cr) * (tr) - (ci) * (ti); \
        const float zi = (cr) * (ti) + (ci) * (tr); \
        (cr) = zr;                                  \
        (ci) = zi;                                  \
    } while (0)

// digit reversal of the radix-4 x radix-5 outer part (f' = t1*5 + t2 -> t1 + 4*t2)
#define FFT90_DIGREV20(f) ((f) / 5 + 4 * ((f) % 5))

// ---------------------------------------------------------------------------
// Stage A: radix-20 (fused radix-4 D=125 + radix-5 D=25).  25 groups x 50 cols.
// ---------------------------------------------------------------------------
#define FFT90_STAGE20                                                                                   \
    do {                                                                                                \
        const uint32_t total = 25 * static_cast<uint32_t>(B2H);                                         \
        const uint32_t step25 = 25 * static_cast<uint32_t>(ROW_STRIDE);                                 \
        const uint32_t iters = (total + SIMT_THREADS - 1) / SIMT_THREADS;                               \
        for (uint32_t it = 0; it < iters; ++it) {                                                       \
            const uint32_t e = it * SIMT_THREADS + threadIdx.x;                                         \
            if (e >= total)                                                                             \
                continue;                                                                               \
            const uint32_t jj = e % static_cast<uint32_t>(B2H);                                         \
            const uint32_t g = e / static_cast<uint32_t>(B2H);                                          \
            const uint32_t base = g * static_cast<uint32_t>(ROW_STRIDE) + 2 * jj;                       \
            float x0r = data[base];                                                                     \
            float x0i = data[base + 1];                                                                 \
            float x1r = data[base + step25];                                                            \
            float x1i = data[base + step25 + 1];                                                        \
            float x2r = data[base + 2 * step25];                                                        \
            float x2i = data[base + 2 * step25 + 1];                                                    \
            float x3r = data[base + 3 * step25];                                                        \
            float x3i = data[base + 3 * step25 + 1];                                                    \
            float x4r = data[base + 4 * step25];                                                        \
            float x4i = data[base + 4 * step25 + 1];                                                    \
            float x5r = data[base + 5 * step25];                                                        \
            float x5i = data[base + 5 * step25 + 1];                                                    \
            float x6r = data[base + 6 * step25];                                                        \
            float x6i = data[base + 6 * step25 + 1];                                                    \
            float x7r = data[base + 7 * step25];                                                        \
            float x7i = data[base + 7 * step25 + 1];                                                    \
            float x8r = data[base + 8 * step25];                                                        \
            float x8i = data[base + 8 * step25 + 1];                                                    \
            float x9r = data[base + 9 * step25];                                                        \
            float x9i = data[base + 9 * step25 + 1];                                                    \
            float x10r = data[base + 10 * step25];                                                      \
            float x10i = data[base + 10 * step25 + 1];                                                  \
            float x11r = data[base + 11 * step25];                                                      \
            float x11i = data[base + 11 * step25 + 1];                                                  \
            float x12r = data[base + 12 * step25];                                                      \
            float x12i = data[base + 12 * step25 + 1];                                                  \
            float x13r = data[base + 13 * step25];                                                      \
            float x13i = data[base + 13 * step25 + 1];                                                  \
            float x14r = data[base + 14 * step25];                                                      \
            float x14i = data[base + 14 * step25 + 1];                                                  \
            float x15r = data[base + 15 * step25];                                                      \
            float x15i = data[base + 15 * step25 + 1];                                                  \
            float x16r = data[base + 16 * step25];                                                      \
            float x16i = data[base + 16 * step25 + 1];                                                  \
            float x17r = data[base + 17 * step25];                                                      \
            float x17i = data[base + 17 * step25 + 1];                                                  \
            float x18r = data[base + 18 * step25];                                                      \
            float x18i = data[base + 18 * step25 + 1];                                                  \
            float x19r = data[base + 19 * step25];                                                      \
            float x19i = data[base + 19 * step25 + 1];                                                  \
            /* 5 radix-4 sub-butterflies (local {s0, s0+5, s0+10, s0+15}) */                            \
            /* twiddle j = g + 25*s0 */                                                                 \
            {                                                                                           \
                const uint32_t j0 = g + 25 * 0;                                                         \
                float w1r = tw[(R4_T1_OFF) + 2 * j0], w1i = tw[(R4_T1_OFF) + 2 * j0 + 1];               \
                float w2r = tw[(R4_T2_OFF) + 2 * j0], w2i = tw[(R4_T2_OFF) + 2 * j0 + 1];               \
                float w3r = tw[(R4_T3_OFF) + 2 * j0], w3i = tw[(R4_T3_OFF) + 2 * j0 + 1];               \
                FFT90_R4(x0r, x0i, x5r, x5i, x10r, x10i, x15r, x15i, w1r, w1i, w2r, w2i, w3r, w3i);     \
            }                                                                                           \
            {                                                                                           \
                const uint32_t j1 = g + 25 * 1;                                                         \
                float w1r = tw[(R4_T1_OFF) + 2 * j1], w1i = tw[(R4_T1_OFF) + 2 * j1 + 1];               \
                float w2r = tw[(R4_T2_OFF) + 2 * j1], w2i = tw[(R4_T2_OFF) + 2 * j1 + 1];               \
                float w3r = tw[(R4_T3_OFF) + 2 * j1], w3i = tw[(R4_T3_OFF) + 2 * j1 + 1];               \
                FFT90_R4(x1r, x1i, x6r, x6i, x11r, x11i, x16r, x16i, w1r, w1i, w2r, w2i, w3r, w3i);     \
            }                                                                                           \
            {                                                                                           \
                const uint32_t j2 = g + 25 * 2;                                                         \
                float w1r = tw[(R4_T1_OFF) + 2 * j2], w1i = tw[(R4_T1_OFF) + 2 * j2 + 1];               \
                float w2r = tw[(R4_T2_OFF) + 2 * j2], w2i = tw[(R4_T2_OFF) + 2 * j2 + 1];               \
                float w3r = tw[(R4_T3_OFF) + 2 * j2], w3i = tw[(R4_T3_OFF) + 2 * j2 + 1];               \
                FFT90_R4(x2r, x2i, x7r, x7i, x12r, x12i, x17r, x17i, w1r, w1i, w2r, w2i, w3r, w3i);     \
            }                                                                                           \
            {                                                                                           \
                const uint32_t j3 = g + 25 * 3;                                                         \
                float w1r = tw[(R4_T1_OFF) + 2 * j3], w1i = tw[(R4_T1_OFF) + 2 * j3 + 1];               \
                float w2r = tw[(R4_T2_OFF) + 2 * j3], w2i = tw[(R4_T2_OFF) + 2 * j3 + 1];               \
                float w3r = tw[(R4_T3_OFF) + 2 * j3], w3i = tw[(R4_T3_OFF) + 2 * j3 + 1];               \
                FFT90_R4(x3r, x3i, x8r, x8i, x13r, x13i, x18r, x18i, w1r, w1i, w2r, w2i, w3r, w3i);     \
            }                                                                                           \
            {                                                                                           \
                const uint32_t j4 = g + 25 * 4;                                                         \
                float w1r = tw[(R4_T1_OFF) + 2 * j4], w1i = tw[(R4_T1_OFF) + 2 * j4 + 1];               \
                float w2r = tw[(R4_T2_OFF) + 2 * j4], w2i = tw[(R4_T2_OFF) + 2 * j4 + 1];               \
                float w3r = tw[(R4_T3_OFF) + 2 * j4], w3i = tw[(R4_T3_OFF) + 2 * j4 + 1];               \
                FFT90_R4(x4r, x4i, x9r, x9i, x14r, x14i, x19r, x19i, w1r, w1i, w2r, w2i, w3r, w3i);     \
            }                                                                                           \
            /* 4 radix-5 sub-butterflies (local consecutive), twiddle W_125^{g*t} */                    \
            {                                                                                           \
                float t1r = tw[(TW3_OFF) + 2 * (g * 4 + 0)], t1i = tw[(TW3_OFF) + 2 * (g * 4 + 0) + 1]; \
                float t2r = tw[(TW3_OFF) + 2 * (g * 4 + 1)], t2i = tw[(TW3_OFF) + 2 * (g * 4 + 1) + 1]; \
                float t3r = tw[(TW3_OFF) + 2 * (g * 4 + 2)], t3i = tw[(TW3_OFF) + 2 * (g * 4 + 2) + 1]; \
                float t4r = tw[(TW3_OFF) + 2 * (g * 4 + 3)], t4i = tw[(TW3_OFF) + 2 * (g * 4 + 3) + 1]; \
                FFT90_DFT5(x0r, x0i, x1r, x1i, x2r, x2i, x3r, x3i, x4r, x4i);                           \
                FFT90_TWMUL(x1r, x1i, t1r, t1i);                                                        \
                FFT90_TWMUL(x4r, x4i, t4r, t4i);                                                        \
                FFT90_TWMUL(x2r, x2i, t2r, t2i);                                                        \
                FFT90_TWMUL(x3r, x3i, t3r, t3i);                                                        \
            }                                                                                           \
            {                                                                                           \
                float t1r = tw[(TW3_OFF) + 2 * (g * 4 + 0)], t1i = tw[(TW3_OFF) + 2 * (g * 4 + 0) + 1]; \
                float t2r = tw[(TW3_OFF) + 2 * (g * 4 + 1)], t2i = tw[(TW3_OFF) + 2 * (g * 4 + 1) + 1]; \
                float t3r = tw[(TW3_OFF) + 2 * (g * 4 + 2)], t3i = tw[(TW3_OFF) + 2 * (g * 4 + 2) + 1]; \
                float t4r = tw[(TW3_OFF) + 2 * (g * 4 + 3)], t4i = tw[(TW3_OFF) + 2 * (g * 4 + 3) + 1]; \
                FFT90_DFT5(x5r, x5i, x6r, x6i, x7r, x7i, x8r, x8i, x9r, x9i);                           \
                FFT90_TWMUL(x6r, x6i, t1r, t1i);                                                        \
                FFT90_TWMUL(x9r, x9i, t4r, t4i);                                                        \
                FFT90_TWMUL(x7r, x7i, t2r, t2i);                                                        \
                FFT90_TWMUL(x8r, x8i, t3r, t3i);                                                        \
            }                                                                                           \
            {                                                                                           \
                float t1r = tw[(TW3_OFF) + 2 * (g * 4 + 0)], t1i = tw[(TW3_OFF) + 2 * (g * 4 + 0) + 1]; \
                float t2r = tw[(TW3_OFF) + 2 * (g * 4 + 1)], t2i = tw[(TW3_OFF) + 2 * (g * 4 + 1) + 1]; \
                float t3r = tw[(TW3_OFF) + 2 * (g * 4 + 2)], t3i = tw[(TW3_OFF) + 2 * (g * 4 + 2) + 1]; \
                float t4r = tw[(TW3_OFF) + 2 * (g * 4 + 3)], t4i = tw[(TW3_OFF) + 2 * (g * 4 + 3) + 1]; \
                FFT90_DFT5(x10r, x10i, x11r, x11i, x12r, x12i, x13r, x13i, x14r, x14i);                 \
                FFT90_TWMUL(x11r, x11i, t1r, t1i);                                                      \
                FFT90_TWMUL(x14r, x14i, t4r, t4i);                                                      \
                FFT90_TWMUL(x12r, x12i, t2r, t2i);                                                      \
                FFT90_TWMUL(x13r, x13i, t3r, t3i);                                                      \
            }                                                                                           \
            {                                                                                           \
                float t1r = tw[(TW3_OFF) + 2 * (g * 4 + 0)], t1i = tw[(TW3_OFF) + 2 * (g * 4 + 0) + 1]; \
                float t2r = tw[(TW3_OFF) + 2 * (g * 4 + 1)], t2i = tw[(TW3_OFF) + 2 * (g * 4 + 1) + 1]; \
                float t3r = tw[(TW3_OFF) + 2 * (g * 4 + 2)], t3i = tw[(TW3_OFF) + 2 * (g * 4 + 2) + 1]; \
                float t4r = tw[(TW3_OFF) + 2 * (g * 4 + 3)], t4i = tw[(TW3_OFF) + 2 * (g * 4 + 3) + 1]; \
                FFT90_DFT5(x15r, x15i, x16r, x16i, x17r, x17i, x18r, x18i, x19r, x19i);                 \
                FFT90_TWMUL(x16r, x16i, t1r, t1i);                                                      \
                FFT90_TWMUL(x19r, x19i, t4r, t4i);                                                      \
                FFT90_TWMUL(x17r, x17i, t2r, t2i);                                                      \
                FFT90_TWMUL(x18r, x18i, t3r, t3i);                                                      \
            }                                                                                           \
            /* write back */                                                                            \
            data[base] = x0r;                                                                           \
            data[base + 1] = x0i;                                                                       \
            data[base + step25] = x1r;                                                                  \
            data[base + step25 + 1] = x1i;                                                              \
            data[base + 2 * step25] = x2r;                                                              \
            data[base + 2 * step25 + 1] = x2i;                                                          \
            data[base + 3 * step25] = x3r;                                                              \
            data[base + 3 * step25 + 1] = x3i;                                                          \
            data[base + 4 * step25] = x4r;                                                              \
            data[base + 4 * step25 + 1] = x4i;                                                          \
            data[base + 5 * step25] = x5r;                                                              \
            data[base + 5 * step25 + 1] = x5i;                                                          \
            data[base + 6 * step25] = x6r;                                                              \
            data[base + 6 * step25 + 1] = x6i;                                                          \
            data[base + 7 * step25] = x7r;                                                              \
            data[base + 7 * step25 + 1] = x7i;                                                          \
            data[base + 8 * step25] = x8r;                                                              \
            data[base + 8 * step25 + 1] = x8i;                                                          \
            data[base + 9 * step25] = x9r;                                                              \
            data[base + 9 * step25 + 1] = x9i;                                                          \
            data[base + 10 * step25] = x10r;                                                            \
            data[base + 10 * step25 + 1] = x10i;                                                        \
            data[base + 11 * step25] = x11r;                                                            \
            data[base + 11 * step25 + 1] = x11i;                                                        \
            data[base + 12 * step25] = x12r;                                                            \
            data[base + 12 * step25 + 1] = x12i;                                                        \
            data[base + 13 * step25] = x13r;                                                            \
            data[base + 13 * step25 + 1] = x13i;                                                        \
            data[base + 14 * step25] = x14r;                                                            \
            data[base + 14 * step25 + 1] = x14i;                                                        \
            data[base + 15 * step25] = x15r;                                                            \
            data[base + 15 * step25 + 1] = x15i;                                                        \
            data[base + 16 * step25] = x16r;                                                            \
            data[base + 16 * step25 + 1] = x16i;                                                        \
            data[base + 17 * step25] = x17r;                                                            \
            data[base + 17 * step25 + 1] = x17i;                                                        \
            data[base + 18 * step25] = x18r;                                                            \
            data[base + 18 * step25 + 1] = x18i;                                                        \
            data[base + 19 * step25] = x19r;                                                            \
            data[base + 19 * step25 + 1] = x19i;                                                        \
        }                                                                                               \
    } while (0)

// ---------------------------------------------------------------------------
// Stage B: radix-25 (fused radix-5 D=5 + radix-5 D=1) + digit-reversal scatter.
// ---------------------------------------------------------------------------
#define FFT90_STAGE25_SCATTER                                                                           \
    do {                                                                                                \
        const uint32_t total = 20 * static_cast<uint32_t>(B2H);                                         \
        const uint32_t step = static_cast<uint32_t>(ROW_STRIDE);                                        \
        const uint32_t dk3 = 20 * static_cast<uint32_t>(GM_ROW_FLOATS) / 2;  /* 1000 f2 = 20 rows */    \
        const uint32_t dk4 = 100 * static_cast<uint32_t>(GM_ROW_FLOATS) / 2; /* 5000 f2 = 100 rows */   \
        const uint32_t iters = (total + SIMT_THREADS - 1) / SIMT_THREADS;                               \
        for (uint32_t it = 0; it < iters; ++it) {                                                       \
            const uint32_t e = it * SIMT_THREADS + threadIdx.x;                                         \
            if (e >= total)                                                                             \
                continue;                                                                               \
            const uint32_t jj = e % static_cast<uint32_t>(B2H);                                         \
            const uint32_t fp = e / static_cast<uint32_t>(B2H);                                         \
            const uint32_t base = (25 * fp) * step + 2 * jj;                                            \
            float x0r = data[base];                                                                     \
            float x0i = data[base + 1];                                                                 \
            float x1r = data[base + step];                                                              \
            float x1i = data[base + step + 1];                                                          \
            float x2r = data[base + 2 * step];                                                          \
            float x2i = data[base + 2 * step + 1];                                                      \
            float x3r = data[base + 3 * step];                                                          \
            float x3i = data[base + 3 * step + 1];                                                      \
            float x4r = data[base + 4 * step];                                                          \
            float x4i = data[base + 4 * step + 1];                                                      \
            float x5r = data[base + 5 * step];                                                          \
            float x5i = data[base + 5 * step + 1];                                                      \
            float x6r = data[base + 6 * step];                                                          \
            float x6i = data[base + 6 * step + 1];                                                      \
            float x7r = data[base + 7 * step];                                                          \
            float x7i = data[base + 7 * step + 1];                                                      \
            float x8r = data[base + 8 * step];                                                          \
            float x8i = data[base + 8 * step + 1];                                                      \
            float x9r = data[base + 9 * step];                                                          \
            float x9i = data[base + 9 * step + 1];                                                      \
            float x10r = data[base + 10 * step];                                                        \
            float x10i = data[base + 10 * step + 1];                                                    \
            float x11r = data[base + 11 * step];                                                        \
            float x11i = data[base + 11 * step + 1];                                                    \
            float x12r = data[base + 12 * step];                                                        \
            float x12i = data[base + 12 * step + 1];                                                    \
            float x13r = data[base + 13 * step];                                                        \
            float x13i = data[base + 13 * step + 1];                                                    \
            float x14r = data[base + 14 * step];                                                        \
            float x14i = data[base + 14 * step + 1];                                                    \
            float x15r = data[base + 15 * step];                                                        \
            float x15i = data[base + 15 * step + 1];                                                    \
            float x16r = data[base + 16 * step];                                                        \
            float x16i = data[base + 16 * step + 1];                                                    \
            float x17r = data[base + 17 * step];                                                        \
            float x17i = data[base + 17 * step + 1];                                                    \
            float x18r = data[base + 18 * step];                                                        \
            float x18i = data[base + 18 * step + 1];                                                    \
            float x19r = data[base + 19 * step];                                                        \
            float x19i = data[base + 19 * step + 1];                                                    \
            float x20r = data[base + 20 * step];                                                        \
            float x20i = data[base + 20 * step + 1];                                                    \
            float x21r = data[base + 21 * step];                                                        \
            float x21i = data[base + 21 * step + 1];                                                    \
            float x22r = data[base + 22 * step];                                                        \
            float x22i = data[base + 22 * step + 1];                                                    \
            float x23r = data[base + 23 * step];                                                        \
            float x23i = data[base + 23 * step + 1];                                                    \
            float x24r = data[base + 24 * step];                                                        \
            float x24i = data[base + 24 * step + 1];                                                    \
            /* 5 radix-5 sub-butterflies (local {s0, s0+5, .., s0+20}), twiddle W_25 */                 \
            {                                                                                           \
                float t1r = tw[(TW4_OFF) + 2 * (0 * 4 + 0)], t1i = tw[(TW4_OFF) + 2 * (0 * 4 + 0) + 1]; \
                float t2r = tw[(TW4_OFF) + 2 * (0 * 4 + 1)], t2i = tw[(TW4_OFF) + 2 * (0 * 4 + 1) + 1]; \
                float t3r = tw[(TW4_OFF) + 2 * (0 * 4 + 2)], t3i = tw[(TW4_OFF) + 2 * (0 * 4 + 2) + 1]; \
                float t4r = tw[(TW4_OFF) + 2 * (0 * 4 + 3)], t4i = tw[(TW4_OFF) + 2 * (0 * 4 + 3) + 1]; \
                FFT90_DFT5(x0r, x0i, x5r, x5i, x10r, x10i, x15r, x15i, x20r, x20i);                     \
                FFT90_TWMUL(x5r, x5i, t1r, t1i);                                                        \
                FFT90_TWMUL(x20r, x20i, t4r, t4i);                                                      \
                FFT90_TWMUL(x10r, x10i, t2r, t2i);                                                      \
                FFT90_TWMUL(x15r, x15i, t3r, t3i);                                                      \
            }                                                                                           \
            {                                                                                           \
                float t1r = tw[(TW4_OFF) + 2 * (1 * 4 + 0)], t1i = tw[(TW4_OFF) + 2 * (1 * 4 + 0) + 1]; \
                float t2r = tw[(TW4_OFF) + 2 * (1 * 4 + 1)], t2i = tw[(TW4_OFF) + 2 * (1 * 4 + 1) + 1]; \
                float t3r = tw[(TW4_OFF) + 2 * (1 * 4 + 2)], t3i = tw[(TW4_OFF) + 2 * (1 * 4 + 2) + 1]; \
                float t4r = tw[(TW4_OFF) + 2 * (1 * 4 + 3)], t4i = tw[(TW4_OFF) + 2 * (1 * 4 + 3) + 1]; \
                FFT90_DFT5(x1r, x1i, x6r, x6i, x11r, x11i, x16r, x16i, x21r, x21i);                     \
                FFT90_TWMUL(x6r, x6i, t1r, t1i);                                                        \
                FFT90_TWMUL(x21r, x21i, t4r, t4i);                                                      \
                FFT90_TWMUL(x11r, x11i, t2r, t2i);                                                      \
                FFT90_TWMUL(x16r, x16i, t3r, t3i);                                                      \
            }                                                                                           \
            {                                                                                           \
                float t1r = tw[(TW4_OFF) + 2 * (2 * 4 + 0)], t1i = tw[(TW4_OFF) + 2 * (2 * 4 + 0) + 1]; \
                float t2r = tw[(TW4_OFF) + 2 * (2 * 4 + 1)], t2i = tw[(TW4_OFF) + 2 * (2 * 4 + 1) + 1]; \
                float t3r = tw[(TW4_OFF) + 2 * (2 * 4 + 2)], t3i = tw[(TW4_OFF) + 2 * (2 * 4 + 2) + 1]; \
                float t4r = tw[(TW4_OFF) + 2 * (2 * 4 + 3)], t4i = tw[(TW4_OFF) + 2 * (2 * 4 + 3) + 1]; \
                FFT90_DFT5(x2r, x2i, x7r, x7i, x12r, x12i, x17r, x17i, x22r, x22i);                     \
                FFT90_TWMUL(x7r, x7i, t1r, t1i);                                                        \
                FFT90_TWMUL(x22r, x22i, t4r, t4i);                                                      \
                FFT90_TWMUL(x12r, x12i, t2r, t2i);                                                      \
                FFT90_TWMUL(x17r, x17i, t3r, t3i);                                                      \
            }                                                                                           \
            {                                                                                           \
                float t1r = tw[(TW4_OFF) + 2 * (3 * 4 + 0)], t1i = tw[(TW4_OFF) + 2 * (3 * 4 + 0) + 1]; \
                float t2r = tw[(TW4_OFF) + 2 * (3 * 4 + 1)], t2i = tw[(TW4_OFF) + 2 * (3 * 4 + 1) + 1]; \
                float t3r = tw[(TW4_OFF) + 2 * (3 * 4 + 2)], t3i = tw[(TW4_OFF) + 2 * (3 * 4 + 2) + 1]; \
                float t4r = tw[(TW4_OFF) + 2 * (3 * 4 + 3)], t4i = tw[(TW4_OFF) + 2 * (3 * 4 + 3) + 1]; \
                FFT90_DFT5(x3r, x3i, x8r, x8i, x13r, x13i, x18r, x18i, x23r, x23i);                     \
                FFT90_TWMUL(x8r, x8i, t1r, t1i);                                                        \
                FFT90_TWMUL(x23r, x23i, t4r, t4i);                                                      \
                FFT90_TWMUL(x13r, x13i, t2r, t2i);                                                      \
                FFT90_TWMUL(x18r, x18i, t3r, t3i);                                                      \
            }                                                                                           \
            {                                                                                           \
                float t1r = tw[(TW4_OFF) + 2 * (4 * 4 + 0)], t1i = tw[(TW4_OFF) + 2 * (4 * 4 + 0) + 1]; \
                float t2r = tw[(TW4_OFF) + 2 * (4 * 4 + 1)], t2i = tw[(TW4_OFF) + 2 * (4 * 4 + 1) + 1]; \
                float t3r = tw[(TW4_OFF) + 2 * (4 * 4 + 2)], t3i = tw[(TW4_OFF) + 2 * (4 * 4 + 2) + 1]; \
                float t4r = tw[(TW4_OFF) + 2 * (4 * 4 + 3)], t4i = tw[(TW4_OFF) + 2 * (4 * 4 + 3) + 1]; \
                FFT90_DFT5(x4r, x4i, x9r, x9i, x14r, x14i, x19r, x19i, x24r, x24i);                     \
                FFT90_TWMUL(x9r, x9i, t1r, t1i);                                                        \
                FFT90_TWMUL(x24r, x24i, t4r, t4i);                                                      \
                FFT90_TWMUL(x14r, x14i, t2r, t2i);                                                      \
                FFT90_TWMUL(x19r, x19i, t3r, t3i);                                                      \
            }                                                                                           \
            /* 5 radix-5 sub-butterflies (consecutive), no twiddle */                                   \
            FFT90_DFT5(x0r, x0i, x1r, x1i, x2r, x2i, x3r, x3i, x4r, x4i);                               \
            FFT90_DFT5(x5r, x5i, x6r, x6i, x7r, x7i, x8r, x8i, x9r, x9i);                               \
            FFT90_DFT5(x10r, x10i, x11r, x11i, x12r, x12i, x13r, x13i, x14r, x14i);                     \
            FFT90_DFT5(x15r, x15i, x16r, x16i, x17r, x17i, x18r, x18i, x19r, x19i);                     \
            FFT90_DFT5(x20r, x20i, x21r, x21i, x22r, x22i, x23r, x23i, x24r, x24i);                     \
            /* digit-reversal scatter: X[s=t3*5+t4] -> row fr + 20*t3 + 100*t4 */                       \
            const uint32_t fr = FFT90_DIGREV20(fp);                                                     \
            const uint32_t gmBase = fr * static_cast<uint32_t>(GM_ROW_FLOATS) + 2 * jj;                 \
            __gm__ float2* d2 = (__gm__ float2*)(gmOut + gmBase);                                       \
            d2[0] = make_float2(x0r, x0i);                                                              \
            d2[dk4] = make_float2(x1r, x1i);                                                            \
            d2[2 * dk4] = make_float2(x2r, x2i);                                                        \
            d2[3 * dk4] = make_float2(x3r, x3i);                                                        \
            d2[4 * dk4] = make_float2(x4r, x4i);                                                        \
            d2[dk3] = make_float2(x5r, x5i);                                                            \
            d2[dk3 + dk4] = make_float2(x6r, x6i);                                                      \
            d2[dk3 + 2 * dk4] = make_float2(x7r, x7i);                                                  \
            d2[dk3 + 3 * dk4] = make_float2(x8r, x8i);                                                  \
            d2[dk3 + 4 * dk4] = make_float2(x9r, x9i);                                                  \
            d2[2 * dk3] = make_float2(x10r, x10i);                                                      \
            d2[2 * dk3 + dk4] = make_float2(x11r, x11i);                                                \
            d2[2 * dk3 + 2 * dk4] = make_float2(x12r, x12i);                                            \
            d2[2 * dk3 + 3 * dk4] = make_float2(x13r, x13i);                                            \
            d2[2 * dk3 + 4 * dk4] = make_float2(x14r, x14i);                                            \
            d2[3 * dk3] = make_float2(x15r, x15i);                                                      \
            d2[3 * dk3 + dk4] = make_float2(x16r, x16i);                                                \
            d2[3 * dk3 + 2 * dk4] = make_float2(x17r, x17i);                                            \
            d2[3 * dk3 + 3 * dk4] = make_float2(x18r, x18i);                                            \
            d2[3 * dk3 + 4 * dk4] = make_float2(x19r, x19i);                                            \
            d2[4 * dk3] = make_float2(x20r, x20i);                                                      \
            d2[4 * dk3 + dk4] = make_float2(x21r, x21i);                                                \
            d2[4 * dk3 + 2 * dk4] = make_float2(x22r, x22i);                                            \
            d2[4 * dk3 + 3 * dk4] = make_float2(x23r, x23i);                                            \
            d2[4 * dk3 + 4 * dk4] = make_float2(x24r, x24i);                                            \
        }                                                                                               \
    } while (0)

template <int32_t SGN>
__simt_vf__ __launch_bounds__(SIMT_THREADS) inline void simt_fft_500x50_impl(__ubuf__ float* data, __ubuf__ float* tw,
                                                                             __gm__ float* gmOut)
{
    constexpr float sgn = static_cast<float>(SGN);

    FFT90_STAGE20;
    asc_syncthreads();
    FFT90_STAGE25_SCATTER;
}

#undef FFT90_DFT5
#undef FFT90_R4
#undef FFT90_TWMUL
#undef FFT90_DIGREV20
#undef FFT90_STAGE20
#undef FFT90_STAGE25_SCATTER

#ifdef FFT90_HOST_SIM
inline void simt_fft_500x50(__ubuf__ float* data, __ubuf__ float* tw, __gm__ float* gmOut, int32_t dirSign)
{
    if (dirSign > 0) {
        simt_fft_500x50_impl<1>(data, tw, gmOut);
    } else {
        simt_fft_500x50_impl<-1>(data, tw, gmOut);
    }
}
#endif

} // namespace Fft

#endif // FFT_SIMT_OPS_H
