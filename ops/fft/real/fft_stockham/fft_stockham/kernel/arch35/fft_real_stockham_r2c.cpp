/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#define CATLASS_ARCH 3510

#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "vendor/fft_stockham_mix.h"
#include "params/fft_real_stockham_tiling.h"

using namespace AscendC;

constexpr uint32_t VF_MAX_THREAD_NUM = 256;

__simt_callee__ inline void ComplexMul(float ar, float ai, float br, float bi, float& cr, float& ci)
{
    cr = ar * br - ai * bi;
    ci = ar * bi + ai * br;
}

__simt_vf__ LAUNCH_BOUND(VF_MAX_THREAD_NUM)
__aicore__ void FftR2CPackRealToComplex(__gm__ float* __restrict__ gm_input, __gm__ float* __restrict__ gm_workspace,
                                        int64_t batchSize, int64_t fftN)
{
    int64_t fftPointCount = fftN / 2;
    int64_t totalComplex = batchSize * fftPointCount;
    int64_t tid = static_cast<int64_t>(Simt::GetThreadIdx<0>());
    int64_t stride = static_cast<int64_t>(Simt::GetThreadNum<0>());

    for (int64_t idx = tid; idx < totalComplex; idx += stride) {
        int64_t batch = idx / fftPointCount;
        int64_t k = idx - batch * fftPointCount;
        int64_t outOffset = idx * 2;
        int64_t inOffset = batch * fftN + k * 2;
        gm_workspace[outOffset] = gm_input[inOffset];
        gm_workspace[outOffset + 1] = gm_input[inOffset + 1];
    }
}

__simt_vf__ LAUNCH_BOUND(VF_MAX_THREAD_NUM)
__aicore__ void FftR2COddPackRealToComplex(__gm__ float* __restrict__ gm_input, __gm__ float* __restrict__ gm_workspace,
                                           int64_t totalElements)
{
    int64_t tid = static_cast<int64_t>(Simt::GetThreadIdx<0>());
    int64_t stride = static_cast<int64_t>(Simt::GetThreadNum<0>());

    for (int64_t idx = tid; idx < totalElements; idx += stride) {
        gm_workspace[idx * 2] = gm_input[idx];
        gm_workspace[idx * 2 + 1] = 0.0f;
    }
}

__simt_vf__ LAUNCH_BOUND(VF_MAX_THREAD_NUM)
__aicore__ void FftR2CPostProcess(__gm__ float* __restrict__ fftOutput, __gm__ float* __restrict__ gm_tw_post,
                                  __gm__ float* __restrict__ gm_output, int64_t batchSize, int64_t fftN)
{
    int64_t fftPointCount = fftN / 2;
    int64_t spectrumSize = fftPointCount + 1;
    int64_t tid = static_cast<int64_t>(Simt::GetThreadIdx<0>());
    int64_t stride = static_cast<int64_t>(Simt::GetThreadNum<0>());

    for (int64_t idx = tid; idx < batchSize * spectrumSize; idx += stride) {
        int64_t batch = idx / spectrumSize;
        int64_t k = idx - batch * spectrumSize;
        int64_t outOffset = idx * 2;
        int64_t z0Offset = batch * fftPointCount * 2;

        if (k == 0) {
            float z0Re = fftOutput[z0Offset];
            float z0Im = fftOutput[z0Offset + 1];
            gm_output[outOffset] = z0Re + z0Im;
            gm_output[outOffset + 1] = 0.0f;
        } else if (k == fftPointCount) {
            float z0Re = fftOutput[z0Offset];
            float z0Im = fftOutput[z0Offset + 1];
            gm_output[outOffset] = z0Re - z0Im;
            gm_output[outOffset + 1] = 0.0f;
        } else {
            int64_t zkOffset = (batch * fftPointCount + k) * 2;
            int64_t znkOffset = (batch * fftPointCount + (fftPointCount - k)) * 2;
            float zkRe = fftOutput[zkOffset];
            float zkIm = fftOutput[zkOffset + 1];
            float znkRe = fftOutput[znkOffset];
            float znkIm = -fftOutput[znkOffset + 1];

            float evenRe = (zkRe + znkRe) * 0.5f;
            float evenIm = (zkIm + znkIm) * 0.5f;
            float oddRe = (zkIm - znkIm) * 0.5f;
            float oddIm = (znkRe - zkRe) * 0.5f;

            float twiddledRe;
            float twiddledIm;
            ComplexMul(gm_tw_post[k * 2], gm_tw_post[k * 2 + 1], oddRe, oddIm, twiddledRe, twiddledIm);
            gm_output[outOffset] = evenRe + twiddledRe;
            gm_output[outOffset + 1] = evenIm + twiddledIm;
        }
    }
}

__simt_vf__ LAUNCH_BOUND(VF_MAX_THREAD_NUM)
__aicore__ void FftR2COddTruncateOutput(__gm__ float* __restrict__ fftOutput, __gm__ float* __restrict__ gm_output,
                                        int64_t batchSize, int64_t fftN)
{
    int64_t spectrumSize = fftN / 2 + 1;
    int64_t totalOutput = batchSize * spectrumSize;
    int64_t tid = static_cast<int64_t>(Simt::GetThreadIdx<0>());
    int64_t stride = static_cast<int64_t>(Simt::GetThreadNum<0>());

    for (int64_t idx = tid; idx < totalOutput; idx += stride) {
        int64_t batch = idx / spectrumSize;
        int64_t k = idx - batch * spectrumSize;
        int64_t inOffset = (batch * fftN + k) * 2;
        int64_t outOffset = idx * 2;
        gm_output[outOffset] = fftOutput[inOffset];
        gm_output[outOffset + 1] = fftOutput[inOffset + 1];
    }
}

class FftR2CKernelMultiCore {
public:
    __aicore__ inline void Init(__gm__ float* __restrict__ gm_input, __gm__ float* __restrict__ gm_dft_matrix_array,
                                __gm__ float* __restrict__ gm_tw_matrix_array,
                                __gm__ float* __restrict__ gm_tw_post_process, __gm__ int32_t* __restrict__ radix_list,
                                __gm__ float* __restrict__ gm_output, __gm__ float* __restrict__ gm_workspace,
                                __gm__ uint8_t* __restrict__ gm_tiling_para);

    __aicore__ inline void Process();

private:
    __gm__ float* __restrict__ gm_input_core_;
    __gm__ float* __restrict__ gm_dft_matrix_array_;
    __gm__ float* __restrict__ gm_tw_matrix_array_;
    __gm__ float* __restrict__ gm_tw_post_process_;
    __gm__ int32_t* __restrict__ gm_radix_list_;
    __gm__ float* __restrict__ gm_output_core_;
    __gm__ float* __restrict__ workspace0_;
    __gm__ float* __restrict__ workspace1_;

    int64_t fftN_;
    int32_t radixListLen_;
    int32_t isOddN_;
    int64_t localBatchSize_;
};

__aicore__ inline void FftR2CKernelMultiCore::Init(
    __gm__ float* __restrict__ gm_input, __gm__ float* __restrict__ gm_dft_matrix_array,
    __gm__ float* __restrict__ gm_tw_matrix_array, __gm__ float* __restrict__ gm_tw_post_process,
    __gm__ int32_t* __restrict__ radix_list, __gm__ float* __restrict__ gm_output,
    __gm__ float* __restrict__ gm_workspace, __gm__ uint8_t* __restrict__ gm_tiling_para)
{
    auto tiling = reinterpret_cast<__gm__ AsdSip::FftRealStockhamTiling*>(gm_tiling_para);
    int64_t batchSize = tiling->batchSize;
    fftN_ = tiling->fftN;
    radixListLen_ = tiling->radixListLen;
    isOddN_ = tiling->isOddN;

    int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
    int64_t blockNum = static_cast<int64_t>(GetBlockNum());
    int64_t batchesPerCore = batchSize / blockNum;
    int64_t extraBatches = batchSize % blockNum;
    int64_t batchStart;

    if (blockIdx < extraBatches) {
        batchStart = blockIdx * (batchesPerCore + 1);
        localBatchSize_ = batchesPerCore + 1;
    } else {
        batchStart = extraBatches * (batchesPerCore + 1) + (blockIdx - extraBatches) * batchesPerCore;
        localBatchSize_ = batchesPerCore;
    }

    int64_t fftPointCount = isOddN_ ? fftN_ : fftN_ / 2;
    int64_t inputOffset = batchStart * fftN_;
    int64_t outputOffset = batchStart * (fftN_ / 2 + 1) * 2;
    int64_t workspaceBatchBytes = batchStart * fftPointCount * 2 * sizeof(float);

    gm_input_core_ = gm_input + inputOffset;
    gm_output_core_ = gm_output + outputOffset;
    gm_dft_matrix_array_ = gm_dft_matrix_array;
    gm_tw_matrix_array_ = gm_tw_matrix_array;
    gm_tw_post_process_ = gm_tw_post_process;
    gm_radix_list_ = radix_list;
    workspace0_ = reinterpret_cast<__gm__ float*>(reinterpret_cast<__gm__ uint8_t*>(gm_workspace) +
                                                  tiling->workspaceOffsets[0] + workspaceBatchBytes);
    workspace1_ = reinterpret_cast<__gm__ float*>(reinterpret_cast<__gm__ uint8_t*>(gm_workspace) +
                                                  tiling->workspaceOffsets[1] + workspaceBatchBytes);
}

__aicore__ inline void FftR2CKernelMultiCore::Process()
{
    if (localBatchSize_ <= 0) {
        return;
    }

    int64_t fftPointCount = isOddN_ ? fftN_ : fftN_ / 2;
    if (isOddN_) {
        Simt::VF_CALL<FftR2COddPackRealToComplex>(Simt::Dim3{VF_MAX_THREAD_NUM, 1, 1}, gm_input_core_, workspace0_,
                                                  localBatchSize_ * fftN_);
    } else {
        Simt::VF_CALL<FftR2CPackRealToComplex>(Simt::Dim3{VF_MAX_THREAD_NUM, 1, 1}, gm_input_core_, workspace0_,
                                               localBatchSize_, fftN_);
    }

    __gm__ float* fftOutput = AsdSip::stockham::runStockhamForward(
        workspace0_, workspace1_, gm_dft_matrix_array_, gm_tw_matrix_array_, gm_radix_list_, radixListLen_,
        static_cast<int32_t>(localBatchSize_), static_cast<int32_t>(fftPointCount));

    AscendC::PipeBarrier<PIPE_V>();
    if (isOddN_) {
        Simt::VF_CALL<FftR2COddTruncateOutput>(Simt::Dim3{VF_MAX_THREAD_NUM, 1, 1}, fftOutput, gm_output_core_,
                                               localBatchSize_, fftN_);
    } else {
        Simt::VF_CALL<FftR2CPostProcess>(Simt::Dim3{VF_MAX_THREAD_NUM, 1, 1}, fftOutput, gm_tw_post_process_,
                                         gm_output_core_, localBatchSize_, fftN_);
    }
}

extern "C" __global__ __vector__ void fft_real_stockham_r2c(
    __gm__ float* __restrict__ gm_input, __gm__ float* __restrict__ gm_dft_matrix_array,
    __gm__ float* __restrict__ gm_tw_matrix_array, __gm__ float* __restrict__ gm_tw_post_process,
    __gm__ int32_t* __restrict__ radix_list, __gm__ float* __restrict__ gm_output,
    __gm__ float* __restrict__ gm_workspace, __gm__ uint8_t* __restrict__ gm_tiling_para)
{
    FftR2CKernelMultiCore kernel;
    kernel.Init(gm_input, gm_dft_matrix_array, gm_tw_matrix_array, gm_tw_post_process, radix_list, gm_output,
                gm_workspace, gm_tiling_para);
    kernel.Process();
}
