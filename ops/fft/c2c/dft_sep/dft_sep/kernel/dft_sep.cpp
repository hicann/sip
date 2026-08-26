/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_ARCH
#define CATLASS_ARCH 2201
#endif

#include "kernel_operator.h"
#include "dft_sep_matmul_kernel.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/device/device_gemm.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/status.hpp"

using T_INPUT = float;
using T_OUTPUT = float;

using namespace Catlass;

constexpr uint32_t L1_SHAPE_M = 128;
constexpr uint32_t L1_SHAPE_N = 128;
constexpr uint32_t L1_SHAPE_K = 128;

constexpr uint32_t L0_SHAPE_M = 128;
constexpr uint32_t L0_SHAPE_N = 128;
constexpr uint32_t L0_SHAPE_K = 64;

extern "C" __global__ __aicore__ void dft_sep(GM_ADDR ffts_addr, GM_ADDR gm_a_real, GM_ADDR gm_a_imag, GM_ADDR gm_b,
                                              GM_ADDR gm_c_real, GM_ADDR gm_c_imag, GM_ADDR workspace,
                                              GM_ADDR tiling_para_gm)
{
    AscendC::SetSyncBaseAddr((uint64_t)ffts_addr);
    uint32_t batchSize, m, n, k, trans_a, trans_b, M0, N0, K0;

    auto tiling_para = reinterpret_cast<__gm__ int32_t*>(tiling_para_gm);
    batchSize = tiling_para[0]; // set to 1
    m = tiling_para[1];         // batch size
    n = tiling_para[2];         // fftN
    k = tiling_para[3];         // fftN

    // 指针偏移用 64 位计算：n*k 在 uint32_t 域相乘，fftN > 65536 时溢出回绕导致越界（issue #113）
    uint64_t bMatrixSize = static_cast<uint64_t>(n) * static_cast<uint64_t>(k) * sizeof(float);
    auto gm_b_real = gm_b;
    auto gm_b_imag = gm_b + bMatrixSize;
    auto gm_b_imag_neg = gm_b + 2 * bMatrixSize;

    using LayoutA = layout::RowMajor;
    using LayoutB = layout::RowMajor;
    using LayoutC = layout::RowMajor;
    LayoutA layoutA{m, k};
    LayoutB layoutB{k, n};
    LayoutC layoutC{m, n};

    using ArchTag = Arch::AtlasA2;
    using DispatchPolicy = Gemm::MmadAtlasA2Pingpong<true>;
    using L1TileShape = GemmShape<L1_SHAPE_M, L1_SHAPE_N, L1_SHAPE_K>;

    using L0TileShape = GemmShape<L0_SHAPE_M, L0_SHAPE_N, L0_SHAPE_K>;

    using AType = Gemm::GemmType<T_INPUT, LayoutA>;
    using BType = Gemm::GemmType<T_INPUT, LayoutB>;
    using CType = Gemm::GemmType<T_OUTPUT, LayoutC>;

    using BlockMmad = Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType>;
    using BlockEpilogue = void;

    // Swizzle offset is 3 and direction is 0.
    using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;

    // kernel level
    using MatmulKernel = Gemm::Kernel::CustomBasicMatmulForDft<BlockMmad, BlockEpilogue, BlockScheduler>;

    Catlass::GemmCoord problemShape{m, n, k};
    Arch::Resource<ArchTag> resource;

    AscendC::SetAtomicNone();

    MatmulKernel matmul_op;
    MatmulKernel::Params params{problemShape, gm_a_real, layoutA, gm_b_real, layoutB, gm_c_real, layoutC};
    matmul_op(params, resource);

    params = {problemShape, gm_a_real, layoutA, gm_b_imag, layoutB, gm_c_imag, layoutC};
    matmul_op(params, resource);

    AscendC::SetAtomicAdd<float>();

    params = {problemShape, gm_a_imag, layoutA, gm_b_imag_neg, layoutB, gm_c_real, layoutC};
    matmul_op(params, resource);

    params = {problemShape, gm_a_imag, layoutA, gm_b_real, layoutB, gm_c_imag, layoutC};
    matmul_op(params, resource);
}
