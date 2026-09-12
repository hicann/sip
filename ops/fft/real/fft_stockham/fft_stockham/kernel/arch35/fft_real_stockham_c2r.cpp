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

__simt_vf__ LAUNCH_BOUND(VF_MAX_THREAD_NUM)
__aicore__ void FftC2RHermitianExpand(__gm__ float* __restrict__ gm_input, __gm__ float* __restrict__ gm_workspace,
                                      int64_t batchSize, int64_t fftN)
{
    int64_t spectrumSize = fftN / 2 + 1;
    int64_t totalComplex = batchSize * fftN;
    int64_t tid = static_cast<int64_t>(Simt::GetThreadIdx<0>());
    int64_t stride = static_cast<int64_t>(Simt::GetThreadNum<0>());

    for (int64_t idx = tid; idx < totalComplex; idx += stride) {
        int64_t batch = idx / fftN;
        int64_t k = idx - batch * fftN;
        int64_t outOffset = idx * 2;

        if (k < spectrumSize) {
            int64_t inOffset = (batch * spectrumSize + k) * 2;
            gm_workspace[outOffset] = gm_input[inOffset];
            gm_workspace[outOffset + 1] = gm_input[inOffset + 1];
        } else {
            int64_t mirror = fftN - k;
            int64_t inOffset = (batch * spectrumSize + mirror) * 2;
            gm_workspace[outOffset] = gm_input[inOffset];
            gm_workspace[outOffset + 1] = -gm_input[inOffset + 1];
        }
    }
}

__simt_vf__ LAUNCH_BOUND(VF_MAX_THREAD_NUM)
__aicore__ void FftC2RStoreReal(__gm__ float* __restrict__ fftOutput, __gm__ float* __restrict__ gm_output,
                                int64_t totalElements)
{
    int64_t tid = static_cast<int64_t>(Simt::GetThreadIdx<0>());
    int64_t stride = static_cast<int64_t>(Simt::GetThreadNum<0>());

    for (int64_t idx = tid; idx < totalElements; idx += stride) {
        gm_output[idx] = fftOutput[idx * 2];
    }
}

__simt_vf__ LAUNCH_BOUND(VF_MAX_THREAD_NUM)
__aicore__ void FftC2REvenPreprocess(__gm__ float* __restrict__ input, __gm__ float* __restrict__ twiddle,
                                     __gm__ float* __restrict__ output, int64_t batchSize, int64_t fftN)
{
    int64_t m = fftN / 2;
    int64_t spectrumN = m + 1;
    int64_t tid = static_cast<int64_t>(Simt::GetThreadIdx<0>());
    int64_t stride = static_cast<int64_t>(Simt::GetThreadNum<0>());
    for (int64_t idx = tid; idx < batchSize * m; idx += stride) {
        int64_t b = idx / m;
        int64_t k = idx - b * m;
        int64_t a = (b * spectrumN + k) * 2;
        int64_t mirror = (b * spectrumN + (m - k)) * 2;
        if (k == 0) {
            float dc = input[a];
            float nyquist = input[mirror];
            output[idx * 2] = 0.5f * (dc + nyquist);
            output[idx * 2 + 1] = 0.5f * (dc - nyquist);
            continue;
        }
        float ar = input[a];
        float ai = input[a + 1];
        float br = input[mirror];
        float bi = -input[mirror + 1];
        float er = 0.5f * (ar + br);
        float ei = 0.5f * (ai + bi);
        float dr = 0.5f * (ar - br);
        float di = 0.5f * (ai - bi);
        float wr = twiddle[k * 2];
        float wi = twiddle[k * 2 + 1];
        float oddr = wr * dr - wi * di;
        float oddi = wr * di + wi * dr;
        output[idx * 2] = er - oddi;
        output[idx * 2 + 1] = ei + oddr;
    }
}

__simt_vf__ LAUNCH_BOUND(VF_MAX_THREAD_NUM)
__aicore__ void FftC2REvenUnpack(__gm__ float* __restrict__ input, __gm__ float* __restrict__ output, int64_t totalReal)
{
    int64_t tid = static_cast<int64_t>(Simt::GetThreadIdx<0>());
    int64_t stride = static_cast<int64_t>(Simt::GetThreadNum<0>());
    for (int64_t idx = tid; idx < totalReal / 2; idx += stride) {
        output[idx * 2] = 2.0f * input[idx * 2];
        output[idx * 2 + 1] = 2.0f * input[idx * 2 + 1];
    }
}

class FftC2RKernelMultiCore {
public:
    __aicore__ inline void Init(__gm__ float* __restrict__ gm_input, __gm__ float* __restrict__ gm_dft_matrix_array,
                                __gm__ float* __restrict__ gm_tw_matrix_array,
                                __gm__ float* __restrict__ gm_tw_real_process, __gm__ int32_t* __restrict__ radix_list,
                                __gm__ float* __restrict__ gm_output, __gm__ float* __restrict__ gm_workspace,
                                __gm__ uint8_t* __restrict__ gm_tiling_para);

    __aicore__ inline void Process();

private:
    __gm__ float* __restrict__ gm_input_core_;
    __gm__ float* __restrict__ gm_dft_matrix_array_;
    __gm__ float* __restrict__ gm_tw_matrix_array_;
    __gm__ float* __restrict__ gm_tw_real_process_;
    __gm__ int32_t* __restrict__ gm_radix_list_;
    __gm__ float* __restrict__ gm_output_core_;
    __gm__ float* __restrict__ workspace0_;
    __gm__ float* __restrict__ workspace1_;

    int64_t fftN_;
    int32_t radixListLen_;
    int32_t isOddN_;
    int64_t localBatchSize_;
};

__aicore__ inline void FftC2RKernelMultiCore::Init(
    __gm__ float* __restrict__ gm_input, __gm__ float* __restrict__ gm_dft_matrix_array,
    __gm__ float* __restrict__ gm_tw_matrix_array, __gm__ float* __restrict__ gm_tw_real_process,
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

    int64_t inputOffset = batchStart * (fftN_ / 2 + 1) * 2;
    int64_t outputOffset = batchStart * fftN_;
    int64_t complexN = isOddN_ ? fftN_ : fftN_ / 2;
    int64_t workspaceBatchBytes = batchStart * complexN * 2 * sizeof(float);

    gm_input_core_ = gm_input + inputOffset;
    gm_output_core_ = gm_output + outputOffset;
    gm_dft_matrix_array_ = gm_dft_matrix_array;
    gm_tw_matrix_array_ = gm_tw_matrix_array;
    gm_tw_real_process_ = gm_tw_real_process;
    gm_radix_list_ = radix_list;
    workspace0_ = reinterpret_cast<__gm__ float*>(reinterpret_cast<__gm__ uint8_t*>(gm_workspace) +
                                                  tiling->workspaceOffsets[0] + workspaceBatchBytes);
    workspace1_ = reinterpret_cast<__gm__ float*>(reinterpret_cast<__gm__ uint8_t*>(gm_workspace) +
                                                  tiling->workspaceOffsets[1] + workspaceBatchBytes);
}

__aicore__ inline void FftC2RKernelMultiCore::Process()
{
    if (localBatchSize_ <= 0) {
        return;
    }

    int64_t complexN = isOddN_ ? fftN_ : fftN_ / 2;
    if (isOddN_) {
        Simt::VF_CALL<FftC2RHermitianExpand>(Simt::Dim3{VF_MAX_THREAD_NUM, 1, 1}, gm_input_core_, workspace0_,
                                             localBatchSize_, fftN_);
    } else {
        Simt::VF_CALL<FftC2REvenPreprocess>(Simt::Dim3{VF_MAX_THREAD_NUM, 1, 1}, gm_input_core_, gm_tw_real_process_,
                                            workspace0_, localBatchSize_, fftN_);
    }

    __gm__ float* fftOutput = AsdSip::stockham::runStockhamForward(
        workspace0_, workspace1_, gm_dft_matrix_array_, gm_tw_matrix_array_, gm_radix_list_, radixListLen_,
        static_cast<int32_t>(localBatchSize_), static_cast<int32_t>(complexN));

    AscendC::PipeBarrier<PIPE_V>();
    if (isOddN_) {
        Simt::VF_CALL<FftC2RStoreReal>(Simt::Dim3{VF_MAX_THREAD_NUM, 1, 1}, fftOutput, gm_output_core_,
                                       localBatchSize_ * fftN_);
    } else {
        Simt::VF_CALL<FftC2REvenUnpack>(Simt::Dim3{VF_MAX_THREAD_NUM, 1, 1}, fftOutput, gm_output_core_,
                                        localBatchSize_ * fftN_);
    }
}

extern "C" __global__ __vector__ void fft_real_stockham_c2r(
    __gm__ float* __restrict__ gm_input, __gm__ float* __restrict__ gm_dft_matrix_array,
    __gm__ float* __restrict__ gm_tw_matrix_array, __gm__ float* __restrict__ gm_tw_real_process,
    __gm__ int32_t* __restrict__ radix_list, __gm__ float* __restrict__ gm_output,
    __gm__ float* __restrict__ gm_workspace, __gm__ uint8_t* __restrict__ gm_tiling_para)
{
    FftC2RKernelMultiCore kernel;
    kernel.Init(gm_input, gm_dft_matrix_array, gm_tw_matrix_array, gm_tw_real_process, radix_list, gm_output,
                gm_workspace, gm_tiling_para);
    kernel.Process();
}
