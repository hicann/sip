/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cmath>
#include <vector>
#include <numeric>
#include <complex>

#include <mki/utils/rt/rt.h>
#include <mki/utils/platform/platform_info.h>
#include <mki/utils/SVector/SVector.h>
#include "utils/assert.h"
#include "log/log.h"
#include "utils/rotate.h"
#include "utils/aspb_status.h"
#include "utils/fft_mode.h"
#include "fft_api.h"
#include "base_api.h"
#include "base_inner_api.h"
#include "utils/ops_base.h"

#include "fftcore/fft_core_any.h"

constexpr int64_t FACTOR_BOUND = 32;

constexpr int64_t NDIM_BOUND = 5;

constexpr int64_t FOUR_G_IN_BYTES = 4294967296L; // 4L * 1024L * 1024L * 1024L

constexpr int64_t K_SIZE_OF_COMPLEX_64 = sizeof(float) * 2;

constexpr int64_t BUFFER_NUM = 2;

constexpr int64_t FIRST_FACTOR_MIN = 16;

constexpr int64_t TIMES_FOUR = 4;

constexpr int64_t DOUBLE_SIZE_OF_COMPLEX_64 = 2;

constexpr int64_t DOUBLE_SIZE_OF_DATASIZE = 2;

constexpr int64_t TENSOR_TWO_DIMENSION = 2;

constexpr int64_t TENSOR_THREE_DIMENSION = 3;

using namespace AsdSip;
namespace {
template <typename E, typename T>
void CopyVector(const std::vector<E>& in, std::vector<T>& out)
{
    for (E e : in) {
        out.push_back(static_cast<T>(e));
    }
}

std::vector<int64_t> Factorize(int64_t size)
{
    std::vector<int64_t> factors{};

    if (size <= 0) {
        return factors;
    }

    int64_t bound = static_cast<int64_t>(std::sqrt(size));
    for (int64_t factor = 2; factor <= bound;) {
        if (size % factor == 0) {
            factors.push_back(factor);
            size /= factor;
            bound = static_cast<int64_t>(std::sqrt(size));
        } else {
            factor++;
        }
    }

    if (size != 1) {
        factors.push_back(size);
    }
    return factors;
}

std::vector<int64_t> MakeSureFirstAlpha(std::vector<int64_t>& factors)
{
    if ((factors.size() == 1) || (factors[0] >= FIRST_FACTOR_MIN)) {
        return factors;
    }
    for (int64_t i = 1; i < static_cast<int64_t>(factors.size()); i++) {
        if (factors[i] >= FIRST_FACTOR_MIN) {
            int64_t tmp = factors[0];
            factors[0] = factors[i];
            factors[i] = tmp;
            break;
        }
    }
    return factors;
}

std::vector<int64_t> Merge(const std::vector<int64_t>& factors_)
{
    std::vector<int64_t> factors(factors_.size());
    std::copy(factors_.rbegin(), factors_.rend(), factors.begin());
    std::vector<int64_t> mergedFactors{};
    std::vector<bool> isMerged(factors.size());
    for (size_t i = 0; i < isMerged.size(); i++) {
        isMerged[i] = false;
    }
    for (size_t i = 0; i < factors.size(); i++) {
        int64_t factor = 1;
        for (size_t j = i; j < factors.size(); j++) {
            if (isMerged[j]) {
                continue;
            }
            if ((factor == 1) || ((factors[j] * factor) <= FACTOR_BOUND)) {
                factor *= factors[j];
                isMerged[j] = true;
            }
        }
        if (factor == 1) {
            break;
        }
        mergedFactors.push_back(factor);
    }
    std::sort(mergedFactors.begin(), mergedFactors.end());
    if (mergedFactors.size() > NDIM_BOUND) {
        std::vector<int64_t> mergedFactors_(NDIM_BOUND);
        std::copy(mergedFactors.begin() + mergedFactors.size() - NDIM_BOUND, mergedFactors.end(),
                  mergedFactors_.begin());
        for (int64_t i = 0; i < (static_cast<int64_t>(mergedFactors.size()) - NDIM_BOUND); i++) {
            auto min_ = std::min_element(mergedFactors_.begin(), mergedFactors_.end());
            *min_ *= mergedFactors[i];
        }
        std::sort(mergedFactors_.begin(), mergedFactors_.end());
        return MakeSureFirstAlpha(mergedFactors_);
    }
    return MakeSureFirstAlpha(mergedFactors);
}

int64_t CalNumel(const SVector<int64_t>& value, uint32_t start, uint32_t end)
{
    int64_t elementCount = 1;
    int64_t maxVal = std::numeric_limits<int64_t>::max();
    int64_t maxDimValue = std::numeric_limits<int32_t>::max();
    for (uint32_t idx = start; idx < end; idx++) {
        int64_t tmpVal = value[idx];
        if (tmpVal <= 0 || tmpVal > maxDimValue) {
            ASDSIP_LOG(ERROR) << "dims : " << tmpVal << " is invalid!";
            throw std::runtime_error("dims is invalid!");
        }
        if (maxVal / elementCount < tmpVal) {
            ASDSIP_LOG(ERROR) << "Tensor size is overflow!";
            throw std::runtime_error("Tensor size is overflow!");
        }
        elementCount *= tmpVal;
    }
    return elementCount;
}

void Permute(Mki::Tensor& input, Mki::Tensor& output, std::vector<int64_t> dimPermute, SVector<int64_t> stride,
             void* stream, uint8_t* deviceBuffer = nullptr)
{
    SVector<int64_t> stride_;
    for (int64_t i = 0; i < static_cast<int64_t>(input.desc.dims.size()); i++) {
        stride_.push_back(stride[dimPermute[i]]);
    }
    SVector<int64_t> tempPermute;
    for (auto i : dimPermute) {
        tempPermute.push_back(input.desc.dims[i]);
    }
    AsStrided(input, output, tempPermute, stride_, stream, 0, deviceBuffer);
}

SVector<int64_t> GetStride(Mki::Tensor& input)
{
    SVector<int64_t> stride_;

    if (input.desc.dims.size() == 0 || input.desc.dims[0] == 0) {
        ASDSIP_LOG(ERROR) << "input.desc.dims.size() == 0 or input.desc.dims[0] == 0!";
        return stride_;
    }

    stride_.push_back(CalNumel(input.desc.dims, 0, input.desc.dims.size()) / input.desc.dims[0]);
    for (int64_t i = 1; i < static_cast<int64_t>(input.desc.dims.size()); i++) {
        if (input.desc.dims[i] != 0) {
            auto tmpStride = stride_[i - 1] / input.desc.dims[i];
            stride_.push_back(tmpStride);
        }
    }
    return stride_;
}

AspbStatus CopyDeviceToDeviceAsync(Tensor& inTensor, Tensor& outTensor, size_t size, void* stream)
{
    int st = MkiRtMemCopyAsync(outTensor.data, size, inTensor.data, size, MKIRT_MEMCOPY_DEVICE_TO_DEVICE, stream);
    if (st != MKIRT_SUCCESS) {
        ASDSIP_LOG(ERROR) << "Memcpy device to device failed.";
        return AsdSip::ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }
    ASDSIP_LOG(INFO) << "Copy device to device async success.";
    return AsdSip::ErrorType::ACL_SUCCESS;
}
} // namespace

void FFTCoreAny::InitRadix()
{
    std::vector<int64_t> factors = Factorize(problemDesc.nDoing);
    if (factors.size() != 0) {
        CopyVector(Merge(factors), radixVec);
        ASDSIP_LOG(INFO) << "FFTCoreAny init radix success.";
    } else {
        ASDSIP_LOG(ERROR) << "size must be greater than 0.";
        throw std::runtime_error("factors size must be greater than 0.");
    }
}

bool FFTCoreAny::PreAllocateInDevice()
{
    FftMode fftMode = convert(problemDesc.fftType);
    std::vector<int64_t> factors;
    CopyVector(radixVec, factors);

    int64_t prevN = 1;
    for (int64_t index = 0; index < static_cast<int64_t>(factors.size()); index++) {
        auto [outN, outComplex, inN, inComplex] = rotate::computeOutShape(factors, index, fftMode);
        if (prevN * outN * outComplex * inN * inComplex * sizeof(float) > FOUR_G_IN_BYTES * TIMES_FOUR) {
            return false;
        }

        prevN *= factors[index];
    }

    prevN = 1;
    for (int64_t index = 0; index < static_cast<int64_t>(factors.size()); index++) {
        std::function<FFTensor*()> func = [=]() -> FFTensor* {
            wten::TheTensor<float> tensor = rotate::OneRotateMatrix(factors, index, fftMode, problemDesc.forward);

            float* hostData = tensor.move_data();
            FFTensor* coeffMatrixPtr = new FFTensor;
            FFTensor& coeffMatrix = *coeffMatrixPtr;

            SVector<int64_t> dims;
            for (auto dim : tensor.shape()) {
                dims.push_back(dim);
            }

            coeffMatrix.desc = {Mki::TensorDType::TENSOR_DTYPE_FLOAT, Mki::TensorFormat::TENSOR_FORMAT_ND, dims, {}, 0};
            coeffMatrix.hostData = hostData;
            coeffMatrix.dataSize = tensor.size() * sizeof(float);

            return coeffMatrixPtr;
        };

        auto [outN, outComplex, inN, inComplex] = rotate::computeOutShape(factors, index, fftMode);
        CoeffKey key = {coreType, 0, {prevN, outN * outComplex, inComplex * inN}, problemDesc.forward};
        coeffMatrices.push_back(FFTensorCache::getCoeff(key, func));

        prevN *= factors[index];
    }
    ASDSIP_LOG(INFO) << "DdCore PreAllocateInDevice success.";
    return true;
}

size_t FFTCoreAny::EstimateWorkspaceSize()
{
    if (problemDesc.fftType == asdFftType::ASCEND_FFT_C2R) {
        return getAlignedSize(problemDesc.nDoing * problemDesc.batch * K_SIZE_OF_COMPLEX_64 *
                              DOUBLE_SIZE_OF_COMPLEX_64 * BUFFER_NUM) +
               getAlignedSize(ASYNC_WORKSPACE_SIZE);
    } else {
        return getAlignedSize(problemDesc.nDoing * problemDesc.batch * K_SIZE_OF_COMPLEX_64 * BUFFER_NUM) +
               getAlignedSize(ASYNC_WORKSPACE_SIZE);
    }
}

void FFTCoreAny::Run(Mki::Tensor& inTensor, Mki::Tensor& outTensor, void* stream, workspace::Workspace& workspace)
{
    size_t bufferSize = 0;
    if (problemDesc.fftType == asdFftType::ASCEND_FFT_C2R) {
        bufferSize = problemDesc.nDoing * problemDesc.batch * K_SIZE_OF_COMPLEX_64 * DOUBLE_SIZE_OF_COMPLEX_64 *
                     BUFFER_NUM;
    } else {
        bufferSize = problemDesc.nDoing * problemDesc.batch * K_SIZE_OF_COMPLEX_64 * BUFFER_NUM;
    }

    buffer = (uint8_t*)workspace.allocate(bufferSize);
    deviceBuffer = (uint8_t*)workspace.allocate(ASYNC_WORKSPACE_SIZE);

    auto mode = problemDesc.fftType;

    Mki::Tensor self = inTensor;
    const int64_t ndim = static_cast<int64_t>(self.desc.dims.size());
    const auto originSize = self.desc.dims;
    const int64_t signalNdim = 1;
    const int64_t batchDims = ndim - signalNdim;

    Tensor tmpPing;
    tmpPing.data = buffer;
    tmpPing.dataSize = inTensor.dataSize;

    // 创建temp tensor
    Tensor tmpPong;
    if (mode == asdFftType::ASCEND_FFT_R2C) {
        tmpPong.data = buffer + tmpPing.dataSize * DOUBLE_SIZE_OF_DATASIZE;
        tmpPong.dataSize = tmpPing.dataSize;
    } else if (mode == asdFftType::ASCEND_FFT_C2C) {
        tmpPong.data = buffer + tmpPing.dataSize;
        tmpPong.dataSize = tmpPing.dataSize;
    }

    Tensor tmpC2r;
    // 补齐镜像
    if (mode == asdFftType::ASCEND_FFT_C2R) {
        tmpC2r.data = buffer + tmpPing.dataSize * DOUBLE_SIZE_OF_DATASIZE;
        tmpC2r.dataSize = tmpPing.dataSize * DOUBLE_SIZE_OF_DATASIZE;

        if (Mki::PlatformInfo::Instance().GetPlatformType() == Mki::PlatformType::ASCEND_910B) {
            Conj(self, tmpPing, stream, deviceBuffer);
        } else if (Mki::PlatformInfo::Instance().GetPlatformType() == Mki::PlatformType::ASCEND_950) {
            tmpPing.desc.dims = self.desc.dims;
        }
        tmpPing.desc.dims.push_back(TENSOR_TWO_DIMENSION);
        auto doubleSizeSelf = CalNumel(tmpPing.desc.dims, 0, tmpPing.desc.dims.size());
        tmpPing.desc = {TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, tmpPing.desc.dims, {}, 0};
        tmpPing.dataSize = static_cast<size_t>(doubleSizeSelf);

        if (Mki::PlatformInfo::Instance().GetPlatformType() == Mki::PlatformType::ASCEND_950) {
            Mki::Tensor selfConj = inTensor;
            selfConj.desc.dims.push_back(TENSOR_TWO_DIMENSION);
            selfConj.desc = {TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, selfConj.desc.dims, {}, 0};

            Tensor conjReal;
            Tensor conjImag;
            conjReal.desc.dims = self.desc.dims;
            conjReal.desc.dims.push_back(1);
            conjReal.desc = {TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, conjReal.desc.dims, {}, 0};
            conjImag.desc = {TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, conjReal.desc.dims, {}, 0};

            conjReal.data = tmpC2r.data;
            conjReal.dataSize = selfConj.dataSize / 2;
            conjImag.data = static_cast<void*>(static_cast<uint8_t*>(tmpC2r.data) + selfConj.dataSize / 2);
            conjImag.dataSize = selfConj.dataSize / 2;

            Slice(selfConj, conjImag, selfConj.desc.dims.size() - 1, 1, TENSOR_TWO_DIMENSION, 1, stream, deviceBuffer);

            Muls(conjImag, float(-1.0), conjReal, stream, deviceBuffer);

            Slice(selfConj, conjImag, selfConj.desc.dims.size() - 1, 0, 1, 1, stream, deviceBuffer);
            Concat(conjImag, conjReal, tmpPing, selfConj.desc.dims.size() - 1, stream, deviceBuffer);
        }
        auto stridePing = GetStride(tmpPing);
        auto asStridedShape = tmpPing.desc.dims;

        asStridedShape[asStridedShape.size() - TENSOR_TWO_DIMENSION]--;

        AsStrided(tmpPing, tmpC2r, asStridedShape, stridePing, stream, 2, deviceBuffer);

        if (problemDesc.nDoing % TENSOR_TWO_DIMENSION == 0) {
            auto strideC2r = GetStride(tmpC2r);
            asStridedShape[asStridedShape.size() - TENSOR_TWO_DIMENSION]--;

            AsStrided(tmpC2r, tmpPing, asStridedShape, strideC2r, stream, 0, deviceBuffer);

            if (Reverse(tmpPing, tmpC2r, {static_cast<int64_t>(tmpPing.desc.dims.size() - TENSOR_TWO_DIMENSION)},
                        stream, deviceBuffer) != AsdSip::ErrorType::ACL_SUCCESS) {
                throw std::runtime_error("Failed to execute the op reverse!");
            }
        } else {
            if (Reverse(tmpC2r, tmpPing, {static_cast<int64_t>(tmpC2r.desc.dims.size() - TENSOR_TWO_DIMENSION)}, stream,
                        deviceBuffer) != AsdSip::ErrorType::ACL_SUCCESS) {
                throw std::runtime_error("Failed to execute the op reverse!");
            }
        }
    }
    // transpose 调整参与fft计算的维度，[2,3,4] -> [4,2,3] 原地更新不可用
    std::vector<int64_t> dimPermute;
    for (int64_t i = 0; i < ndim - 1; i++) {
        dimPermute.push_back(i);
    }

    dimPermute.insert(dimPermute.begin(), ndim - 1);
    if (mode != asdFftType::ASCEND_FFT_R2C) {
        dimPermute.insert(dimPermute.begin(), ndim);

        // C2C需要将输入的complex数据转化为float类型
        auto itSelf = self.desc.dims.end();
        self.desc.dims.insert(itSelf, TENSOR_TWO_DIMENSION);
        auto doubleSizeSelf = CalNumel(self.desc.dims, 0, self.desc.dims.size());
        self.desc = {TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, self.desc.dims, {}, 0};
        self.dataSize = static_cast<size_t>(doubleSizeSelf);
    }
    // 维度重排
    SVector<int64_t> stride;
    SVector<Tensor> tmpPingpong;
    if (mode == asdFftType::ASCEND_FFT_C2R) {
        if (problemDesc.nDoing % TENSOR_TWO_DIMENSION == 0) {
            Concat(self, tmpC2r, tmpPing, self.desc.dims.size() - TENSOR_TWO_DIMENSION, stream, deviceBuffer);
            stride = GetStride(tmpPing);

            Permute(tmpPing, tmpC2r, dimPermute, stride, stream, deviceBuffer);

            tmpPingpong.push_back(tmpC2r);
            tmpPingpong.push_back(tmpPing);
        } else {
            Concat(self, tmpPing, tmpC2r, self.desc.dims.size() - TENSOR_TWO_DIMENSION, stream, deviceBuffer);
            stride = GetStride(tmpC2r);
            Permute(tmpC2r, tmpPing, dimPermute, stride, stream, deviceBuffer);

            tmpPingpong.push_back(tmpPing);
            tmpPingpong.push_back(tmpC2r);
        }
    } else {
        stride = GetStride(self);
        Permute(self, tmpPing, dimPermute, stride, stream, deviceBuffer);

        tmpPingpong.push_back(tmpPing);
        tmpPingpong.push_back(tmpPong);
    }
    // 计算维度，原始tensor的-1，等于self_begin
    int64_t signalTotal = originSize[originSize.size() - 1];
    int64_t batchTotal = CalNumel(inTensor.desc.dims, 0, batchDims);

    // 三个维度
    SVector<int64_t> collapsedSizes(TENSOR_THREE_DIMENSION);
    uint32_t ping = 0;
    uint32_t pong = 1;

    // get plan
    auto factorsList = radixVec;
    int64_t factorsNum = static_cast<int64_t>(coeffMatrices.size());

    // fft iteration
    for (int64_t index = 0; index < factorsNum; index++) {
        auto& coefficientMatrix = *(coeffMatrices[index]);
        if (coefficientMatrix.desc.dims[TENSOR_TWO_DIMENSION] == 0 || coefficientMatrix.desc.dims[0] == 0) {
            ASDSIP_LOG(ERROR) << "coefficientMatrix.desc.dims[2] == 0 || coefficientMatrix.desc.dims[0] == 0.";
            return;
        }

        collapsedSizes[0] = coefficientMatrix.desc.dims[0];
        collapsedSizes[1] = coefficientMatrix.desc.dims[TENSOR_TWO_DIMENSION];
        collapsedSizes[TENSOR_TWO_DIMENSION] = tmpPingpong.at(ping).Numel() /
                                               (coefficientMatrix.desc.dims[TENSOR_TWO_DIMENSION] *
                                                coefficientMatrix.desc.dims[0]);

        tmpPingpong.at(ping).View(collapsedSizes);
        if (MatMul(coefficientMatrix, tmpPingpong.at(ping), tmpPingpong.at(pong), coefficientMatrix.desc.dims[1],
                   coefficientMatrix.desc.dims[TENSOR_TWO_DIMENSION],
                   tmpPingpong.at(ping).desc.dims[TENSOR_TWO_DIMENSION], stream,
                   deviceBuffer) != AsdSip::ErrorType::ACL_SUCCESS) {
            ASDSIP_LOG(ERROR) << "Fail to execute op MatMul!";
            throw std::runtime_error("Fail to execute op MatMul!");
        }

        ping = 1 - ping;
        pong = 1 - pong;
    }
    // reshape
    SVector<int64_t> reshapeSizes(factorsNum + TENSOR_THREE_DIMENSION);
    if (mode == asdFftType::ASCEND_FFT_C2R) {
        reshapeSizes[factorsNum] = 1;
    } else {
        reshapeSizes[factorsNum] = TENSOR_TWO_DIMENSION;
    }

    reshapeSizes[factorsNum + 1] = 1;
    reshapeSizes[factorsNum + TENSOR_TWO_DIMENSION] = batchTotal;
    if (factorsList.size() > reshapeSizes.size()) {
        ASDSIP_LOG(ERROR) << "factorsList size is larger than reshapeSizes!";
        return;
    }
    if (factorsList.size() == 0) {
        ASDSIP_LOG(ERROR) << "factorsList size is 0!";
        return;
    }
    std::copy(factorsList.cbegin(), factorsList.cend(), reshapeSizes.begin());

    if (mode == asdFftType::ASCEND_FFT_R2C) {
        // r2c change
        if (factorsList.back() == 0) {
            ASDSIP_LOG(ERROR) << "factorsList.back() == 0.";
            return;
        }
        if (factorsNum < 1) {
            ASDSIP_LOG(ERROR) << "factorsNum < 1.";
            return;
        }
        signalTotal /= factorsList.back();
        signalTotal *= (factorsList.back() / TENSOR_TWO_DIMENSION + 1);
        reshapeSizes[factorsNum - 1] = reshapeSizes[factorsNum - 1] / TENSOR_TWO_DIMENSION + 1;
    }

    tmpPingpong.at(ping).View(reshapeSizes);

    // Permute
    std::vector<int64_t> dimPer(factorsNum + TENSOR_THREE_DIMENSION);
    std::iota(dimPer.rbegin() + 1, dimPer.rbegin() + factorsNum + 1, int64_t{0});
    dimPer[0] = factorsNum + TENSOR_TWO_DIMENSION;
    dimPer[1] = factorsNum + 1;
    dimPer[factorsNum + TENSOR_TWO_DIMENSION] = factorsNum;

    // transpose 调整参与fft计算的维度，[2,3,4] -> [4,2,3]
    SVector<int64_t> stri;
    stri = GetStride(tmpPingpong.at(ping));

    Permute(tmpPingpong.at(ping), tmpPingpong.at(pong), dimPer, stri, stream, deviceBuffer);

    auto finalSizes = inTensor.desc.dims;
    if (mode == asdFftType::ASCEND_FFT_R2C) {
        finalSizes[batchDims] = tmpPingpong.at(pong).Numel() /
                                (CalNumel(inTensor.desc.dims, 0, batchDims) * TENSOR_TWO_DIMENSION);
        finalSizes.insert(finalSizes.end(), TENSOR_TWO_DIMENSION);
    } else if (mode == asdFftType::ASCEND_FFT_C2R) {
        finalSizes[ndim - 1] = tmpPingpong.at(pong).Numel() / CalNumel(inTensor.desc.dims, 0, batchDims);
    } else {
        finalSizes.insert(finalSizes.end(), TENSOR_TWO_DIMENSION);
    }

    tmpPingpong.at(pong).View(finalSizes);

    // 计算结果shape
    auto outSize = inTensor.desc.dims;

    if (mode == asdFftType::ASCEND_FFT_R2C) {
        outSize[outSize.size() - 1] = (outSize[outSize.size() - 1] / TENSOR_TWO_DIMENSION) + 1;

        auto tmpOutSize = outSize;
        tmpOutSize.push_back(TENSOR_TWO_DIMENSION);

        if (tmpPingpong.at(pong).desc.dims[0] == 0) {
            return;
        }

        SVector<int64_t> strideNew{CalNumel(tmpPingpong.at(pong).desc.dims, 0, tmpPingpong.at(pong).desc.dims.size()) /
                                   tmpPingpong.at(pong).desc.dims[0]};
        for (int64_t i = 1; i < static_cast<int64_t>(tmpPingpong.at(pong).desc.dims.size()); i++) {
            if (tmpPingpong.at(pong).desc.dims[i] != 0) {
                auto tmp_stride = strideNew[i - 1] / tmpPingpong.at(pong).desc.dims[i];
                strideNew.push_back(tmp_stride);
            }
        }
        AsStrided(tmpPingpong.at(pong), tmpPingpong.at(ping), tmpOutSize, strideNew, stream, 0, deviceBuffer);
        ping = 1 - ping;
        pong = 1 - pong;
    }

    CopyDeviceToDeviceAsync(tmpPingpong.at(pong), outTensor, tmpPingpong.at(pong).Numel() * sizeof(float), stream);

    if (mode != asdFftType::ASCEND_FFT_C2R) {
        // HackFloat32intoComplex64
        outTensor.desc = {TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, outSize, {}, 0};
        outTensor.dataSize = tmpPingpong.at(pong).dataSize;
    }
    workspace.recycleLast();
    workspace.recycleLast();
    ASDSIP_LOG(INFO) << "FFTCoreAny run success.";
}

void FFTCoreAny::Run(void* input, void* output, void* stream, workspace::Workspace& workspace)
{
    Tensor inTensor;
    Tensor outTensor;
    TensorDType dtype = TENSOR_DTYPE_COMPLEX64;
    SVector<int64_t> inShape{problemDesc.batch};
    if (problemDesc.fftType == asdFftType::ASCEND_FFT_C2R) {
        inShape.push_back(problemDesc.nDoing / 2 + 1);
    } else {
        inShape.push_back(problemDesc.nDoing);
    }

    if (problemDesc.fftType == asdFftType::ASCEND_FFT_R2C) {
        dtype = TENSOR_DTYPE_FLOAT;
        inTensor.dataSize = problemDesc.batch * problemDesc.nDoing * sizeof(float);
    } else {
        inTensor.dataSize = problemDesc.batch * problemDesc.nDoing * sizeof(float) * 2;
    }
    inTensor.desc = {dtype, TENSOR_FORMAT_ND, inShape, {}, 0};
    inTensor.data = input;
    outTensor.data = output;

    Run(inTensor, outTensor, stream, workspace);
}
