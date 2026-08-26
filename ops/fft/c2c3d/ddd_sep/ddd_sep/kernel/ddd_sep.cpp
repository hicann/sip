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
#include "kernel/fft_batched_matmul_kernel.hpp"
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

constexpr uint32_t MATMUL_READY_ID = 1;

struct DDDSepKernelParams {
    GM_ADDR gm_signal_real;
    GM_ADDR gm_signal_imag;
    GM_ADDR gm_rot_x;
    GM_ADDR gm_rot_y;
    GM_ADDR gm_rot_z;
    GM_ADDR gm_out_real;
    GM_ADDR gm_out_imag;
    GM_ADDR workspace;
    GM_ADDR tiling_para_gm;

    // Methods
    CATLASS_DEVICE
    DDDSepKernelParams() {}

    CATLASS_DEVICE
    DDDSepKernelParams(GM_ADDR gm_signal_real_, GM_ADDR gm_signal_imag_, GM_ADDR gm_rot_x_, GM_ADDR gm_rot_y_,
                       GM_ADDR gm_rot_z_, GM_ADDR gm_out_real_, GM_ADDR gm_out_imag_, GM_ADDR workspace_,
                       GM_ADDR tiling_para_gm_)
        : gm_signal_real(gm_signal_real_),
          gm_signal_imag(gm_signal_imag_),
          gm_rot_x(gm_rot_x_),
          gm_rot_y(gm_rot_y_),
          gm_rot_z(gm_rot_z_),
          gm_out_real(gm_out_real_),
          gm_out_imag(gm_out_imag_),
          workspace(workspace_),
          tiling_para_gm(tiling_para_gm_)
    {}
};

template <class ArchTag_, class BatchMatmulKernel_>
class DDDSepKernel {
public:
    using ArchTag = ArchTag_;
    using MatmulKernel = BatchMatmulKernel_;
    using MatmulParams = typename MatmulKernel::Params;

    // Methods
    CATLASS_DEVICE
    DDDSepKernel() {}

    CATLASS_DEVICE void getMatmulParams(MatmulParams& matmulParams, GM_ADDR gmA, GM_ADDR gmB, GM_ADDR gmC,
                                        uint32_t batchCount, uint32_t m, uint32_t n, uint32_t k, uint32_t strideA,
                                        uint32_t strideB, uint32_t strideC)
    {
        matmulParams.batchCount = batchCount;
        matmulParams.problemShape = {m, n, k};
        matmulParams.layoutA = {m, k};
        matmulParams.strideA = strideA;
        matmulParams.layoutB = {k, n};
        matmulParams.strideB = strideB;
        matmulParams.layoutC = {m, n};
        matmulParams.strideC = strideC;

        matmulParams.ptrA = gmA;
        matmulParams.ptrB = gmB;
        matmulParams.ptrC = gmC;
    }

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE void operator()(DDDSepKernelParams const& params);

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIC>(DDDSepKernelParams const& params)
    {
        uint32_t batchSize, fftSizeX, fftSizeY, fftSizeZ;
        auto tiling_para = reinterpret_cast<__gm__ uint32_t*>(params.tiling_para_gm);
        batchSize = tiling_para[0];
        fftSizeX = tiling_para[1];
        fftSizeY = tiling_para[2];
        fftSizeZ = tiling_para[3];

        auto gm_signal_real = params.gm_signal_real;
        auto gm_signal_imag = params.gm_signal_imag;
        auto gm_out_real = params.gm_out_real;
        auto gm_out_imag = params.gm_out_imag;

        // 指针偏移用 64 位计算：fftSizeX*fftSizeX 等在 uint32_t 域相乘，
        // fftSize > 65536 时溢出回绕导致越界（issue #114）
        uint64_t rotXSize = static_cast<uint64_t>(fftSizeX) * static_cast<uint64_t>(fftSizeX) * sizeof(T_INPUT);
        uint64_t rotYSize = static_cast<uint64_t>(fftSizeY) * static_cast<uint64_t>(fftSizeY) * sizeof(T_INPUT);
        uint64_t rotZSize = static_cast<uint64_t>(fftSizeZ) * static_cast<uint64_t>(fftSizeZ) * sizeof(T_INPUT);

        auto gm_rot_x_real = params.gm_rot_x;
        auto gm_rot_x_imag = params.gm_rot_x + rotXSize;
        auto gm_rot_x_imag_neg = params.gm_rot_x + 2 * rotXSize;

        auto gm_rot_y_real = params.gm_rot_y;
        auto gm_rot_y_imag = params.gm_rot_y + rotYSize;
        auto gm_rot_y_imag_neg = params.gm_rot_y + 2 * rotYSize;

        auto gm_rot_z_real = params.gm_rot_z;
        auto gm_rot_z_imag = params.gm_rot_z + rotZSize;
        auto gm_rot_z_imag_neg = params.gm_rot_z + 2 * rotZSize;

        uint64_t tempSize = static_cast<uint64_t>(batchSize) * static_cast<uint64_t>(fftSizeX) *
                            static_cast<uint64_t>(fftSizeY) * static_cast<uint64_t>(fftSizeZ) * sizeof(T_INPUT);
        auto gm_temp_real = params.workspace;
        auto gm_temp_imag = params.workspace + tempSize;

        MatmulKernel matmulOp;
        MatmulParams matmulParams;

        // x-axis DFT, rot_x @ [batch, x, y * z]
        auto input_real = gm_signal_real;
        auto input_imag = gm_signal_imag;
        auto out_real = gm_out_real;
        auto out_imag = gm_out_imag;

        getMatmulParams(matmulParams, gm_rot_x_real, input_real, out_real, batchSize, fftSizeX, fftSizeY * fftSizeZ,
                        fftSizeX, 0, fftSizeX * fftSizeY * fftSizeZ, fftSizeX * fftSizeY * fftSizeZ);

        // R * R
        matmulParams.ptrA = gm_rot_x_real;
        matmulParams.ptrB = input_real;
        matmulParams.ptrC = out_real;
        matmulOp(matmulParams, resource);

        // R * I
        matmulParams.ptrA = gm_rot_x_real;
        matmulParams.ptrB = input_imag;
        matmulParams.ptrC = out_imag;
        matmulOp(matmulParams, resource);

        AscendC::SetAtomicAdd<float>();

        // I * I
        matmulParams.ptrA = gm_rot_x_imag_neg;
        matmulParams.ptrB = input_imag;
        matmulParams.ptrC = out_real;
        matmulOp(matmulParams, resource);

        // I * R
        matmulParams.ptrA = gm_rot_x_imag;
        matmulParams.ptrB = input_real;
        matmulParams.ptrC = out_imag;
        matmulOp(matmulParams, resource);

        AscendC::SetAtomicNone();

        AscendC::SyncAll();

        // y-axis DFT, rot_y @ [batch *x, y, z]
        input_real = gm_out_real;
        input_imag = gm_out_imag;
        out_real = gm_temp_real;
        out_imag = gm_temp_imag;

        getMatmulParams(matmulParams, gm_rot_y_real, input_real, out_real, batchSize * fftSizeX, fftSizeY, fftSizeZ,
                        fftSizeY, 0, fftSizeY * fftSizeZ, fftSizeY * fftSizeZ);

        // R * R
        matmulParams.ptrA = gm_rot_y_real;
        matmulParams.ptrB = input_real;
        matmulParams.ptrC = out_real;
        matmulOp(matmulParams, resource);

        // R * I
        matmulParams.ptrA = gm_rot_y_real;
        matmulParams.ptrB = input_imag;
        matmulParams.ptrC = out_imag;
        matmulOp(matmulParams, resource);

        AscendC::SetAtomicAdd<float>();

        // I * I
        matmulParams.ptrA = gm_rot_y_imag_neg;
        matmulParams.ptrB = input_imag;
        matmulParams.ptrC = out_real;
        matmulOp(matmulParams, resource);

        // I * R
        matmulParams.ptrA = gm_rot_y_imag;
        matmulParams.ptrB = input_real;
        matmulParams.ptrC = out_imag;
        matmulOp(matmulParams, resource);

        AscendC::SetAtomicNone();

        AscendC::SyncAll();

        // z-axis DFT, [batch *x * y, z] @ rot_z
        input_real = gm_temp_real;
        input_imag = gm_temp_imag;
        out_real = gm_out_real;
        out_imag = gm_out_imag;

        getMatmulParams(matmulParams, input_real, gm_rot_z_real, out_real, 1, batchSize * fftSizeX * fftSizeY, fftSizeZ,
                        fftSizeZ, 0, 0, 0);

        // R * R
        matmulParams.ptrA = input_real;
        matmulParams.ptrB = gm_rot_z_real;
        matmulParams.ptrC = out_real;
        matmulOp(matmulParams, resource);

        // R * I
        matmulParams.ptrA = input_real;
        matmulParams.ptrB = gm_rot_z_imag;
        matmulParams.ptrC = out_imag;
        matmulOp(matmulParams, resource);

        AscendC::SetAtomicAdd<float>();

        // I * I
        matmulParams.ptrA = input_imag;
        matmulParams.ptrB = gm_rot_z_imag_neg;
        matmulParams.ptrC = out_real;
        matmulOp(matmulParams, resource);

        // I * R
        matmulParams.ptrA = input_imag;
        matmulParams.ptrB = gm_rot_z_real;
        matmulParams.ptrC = out_imag;
        matmulOp(matmulParams, resource);

        AscendC::SetAtomicNone();
    }

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIV>(DDDSepKernelParams const& params)
    {}

private:
    Arch::Resource<ArchTag> resource;
    Arch::CrossCoreFlag matmulReady{MATMUL_READY_ID};
};

extern "C" CATLASS_GLOBAL void ddd_sep(GM_ADDR ffts_addr, GM_ADDR gm_signal_real, GM_ADDR gm_signal_imag,
                                       GM_ADDR gm_rot_x, GM_ADDR gm_rot_y, GM_ADDR gm_rot_z, GM_ADDR gm_out_real,
                                       GM_ADDR gm_out_imag, GM_ADDR workspace, GM_ADDR tiling_para_gm)
{
    AscendC::SetSyncBaseAddr((uint64_t)ffts_addr);

    using ArchTag = Arch::AtlasA2;
    using LayoutA = layout::RowMajor;
    using LayoutB = layout::RowMajor;
    using LayoutC = layout::RowMajor;
    using AType = Gemm::GemmType<T_INPUT, LayoutA>;
    using BType = Gemm::GemmType<T_INPUT, LayoutB>;
    using CType = Gemm::GemmType<T_OUTPUT, LayoutC>;
    using DispatchPolicy = Gemm::MmadAtlasA2Pingpong<true>;
    using L1TileShape = GemmShape<L1_SHAPE_M, L1_SHAPE_N, L1_SHAPE_K>;
    using L0TileShape = GemmShape<L0_SHAPE_M, L0_SHAPE_N, L0_SHAPE_K>;
    using BlockMmad = Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType>;
    using BlockEpilogue = void;
    // Swizzle offset is 3 and direction is 0.
    using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;
    // kernel level
    using MatmulKernel = Gemm::Kernel::FftBatchedMatmul<BlockMmad, BlockEpilogue, BlockScheduler>;
    using MatmulParams = typename MatmulKernel::Params;
    using DDDSepKernel = DDDSepKernel<ArchTag, MatmulKernel>;

    DDDSepKernel dddSepOp;

    DDDSepKernelParams params{gm_signal_real, gm_signal_imag, gm_rot_x,  gm_rot_y,      gm_rot_z,
                              gm_out_real,    gm_out_imag,    workspace, tiling_para_gm};

    dddSepOp(params);
}
