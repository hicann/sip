/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>
#include <complex>
#include <algorithm>

#include <mki/tensor.h>
#include <mki/types.h>
#include <mki/utils/platform/platform_info.h>
#include "log/log.h"
#include "ops.h"
#include "utils/ops_base.h"
#include "utils/aspb_status.h"

#include "fftcore/fft_core_base.h"
#include "fftcore/fft_core_common_func_utils.h"
#include "fftcore/fft_core_common_func.h"

constexpr double K_PI = 3.14159265358979323846;
constexpr double K_2PI = 2 * K_PI;
constexpr int64_t BIAS_SIZE = 2048;
constexpr int64_t BIASC_SIZE = 4096;

using namespace AsdSip;

namespace {

constexpr int64_t L2_CACHE_MAX_FLOAT_22 = (1 << 22);
constexpr int64_t L2_CACHE_MAX_FLOAT_21 = (1 << 21);
constexpr int64_t EVEN_THRESHOLD = 524288 / 2;

void InitMem(void* dest, uint8_t ch, size_t count)
{
    if (dest == nullptr) {
        ASDSIP_LOG(ERROR) << "dest is nullptr";
        throw std::runtime_error("Invalid dest.");
    }

    for (size_t i = 0; i < count; i++) {
        ((uint8_t*)dest)[i] = ch;
    }
}

int64_t GetAllWMatrixSize(std::vector<int64_t>& radixVec)
{
    int64_t size = 0;
    int64_t tempN;
    for (std::vector<int64_t>::iterator it = radixVec.begin(); it != radixVec.end(); ++it) {
        tempN = *it;
        size += TWO_MUL * tempN * TWO_MUL * tempN;
        if (size > INT64_MAX) {
            throw std::runtime_error("WMatrixSize large than INT64_MAX.");
        }
    }

    return size;
}

int64_t GetTMatrixCol(std::vector<int64_t>& radixVec, int64_t iter)
{
    int64_t col = 1;
    for (int64_t i = iter + 1; i < static_cast<int64_t>(radixVec.size()); ++i) {
        col *= radixVec[i];
    }
    return col;
}

int64_t GetAllTMatrixSize(std::vector<int64_t>& radixVec)
{
    int64_t size = 0;
    int64_t tempRow;
    int64_t tempCol;
    for (int64_t it = 0; it < static_cast<int64_t>(radixVec.size()) - 1; ++it) {
        tempRow = radixVec[it] * TWO_MUL;
        tempCol = GetTMatrixCol(radixVec, it);
        size += tempRow * tempCol;
        if (size > INT64_MAX) {
            throw std::runtime_error("TMatrixSize large than INT64_MAX.");
        }
    }

    return size;
}

int64_t GetAllSMatrixSize(std::vector<int64_t>& radixVec)
{
    int64_t size = 0;
    int64_t tempN;
    int64_t pre = 1;
    for (std::vector<int64_t>::iterator it = radixVec.begin(); it != radixVec.end(); ++it) {
        tempN = *it;
        size += pre * TWO_MUL * tempN * TWO_MUL * tempN;
        if (size > INT64_MAX) {
            throw std::runtime_error("SMatrixSize large than INT64_MAX.");
        }
        pre *= tempN;
    }

    return size;
}

// 两处调用逻辑均保障n、factors[i]不会为0
std::vector<int64_t> FindTwoRadix(std::vector<int64_t>& factors, int64_t n)
{
    if (n == 0) {
        n = 1;
    }
    int64_t n1 = factors[0];
    int64_t minN1 = n1;
    if (minN1 == 0) {
        minN1 = 1;
    }
    float minRatio = std::max(n / std::pow(n1, 2), std::pow(n1, 2) / n); // (n / n1) 为N2, 此处为比较N2比N1的最小值

    int64_t fLen = static_cast<int64_t>(factors.size());
    for (int64_t i = 0; i < (1 << fLen); i++) {
        n1 = 1;
        for (int64_t j = 0; j < fLen; j++) {
            if ((static_cast<uint64_t>(i) & (1 << j)) != 0) {
                n1 *= factors[j];
            }
        }
        float ratio = std::max(n / std::pow(n1, 2), std::pow(n1, 2) / n);
        if (ratio < minRatio) {
            minN1 = n1;
            minRatio = ratio;
        }
    }

    std::vector<int64_t> radixList = {minN1, n / minN1};
    std::sort(radixList.begin(), radixList.end());

    return radixList;
}

int64_t GetTwiddleMatrixLen(int64_t fftN, const std::vector<int64_t>& radixVecRef)
{
    int64_t n = fftN;
    int64_t dftMatrixLen = 0;
    int64_t stepLen = static_cast<int64_t>(radixVecRef.size());
    int64_t n0 = 1;
    for (int64_t stepIndex = 0; stepIndex < stepLen; stepIndex++) {
        int64_t n1 = radixVecRef[stepIndex];
        if (n1 == 0) {
            continue;
        }
        int64_t n2 = n / n1 / n0;

        int32_t tilM0;
        int32_t tilK0;
        int32_t tilN0;

        getTile(n1, (n2 > 1) ? n2 : n0, stepIndex, stepLen, tilM0, tilN0, tilK0);

        dftMatrixLen += ROUND(TWO_MUL * n1, tilM0) * ROUND(TWO_MUL * n1, tilK0);
        n0 *= n1;
    }

    return dftMatrixLen;
}

} // namespace

AspbStatus InitWMatrixCommon(FFTCoreType coreType, std::vector<int64_t>& radixVec, int64_t n, bool forward,
                             std::shared_ptr<AsdSip::FFTensor>& wMatrix)
{
    std::function<AsdSip::FFTensor*()> func = [=]() mutable -> AsdSip::FFTensor* {
        AsdSip::FFTensor* wMatrixPtr = new AsdSip::FFTensor;
        AsdSip::FFTensor& wMatrix_ = *wMatrixPtr;

        int64_t size = GetAllWMatrixSize(radixVec);

        wMatrix_.desc.dtype = TENSOR_DTYPE_FLOAT;
        wMatrix_.desc.format = TENSOR_FORMAT_ND;
        wMatrix_.desc.dims = {size};
        wMatrix_.dataSize = sizeof(float) * size;

        float* wMatrixHost = nullptr;
        try {
            wMatrixHost = new float[size];
        } catch (std::bad_alloc& e) {
            delete wMatrixPtr;
            ASDSIP_LOG(ERROR) << "wMatrixHost alloc failed: " << e.what();
            throw std::runtime_error("wMatrixHost alloc failed:");
        }
        InitMem(wMatrixHost, 0, wMatrix_.dataSize);

        int64_t tempNi;
        int64_t singleSize;
        int64_t offset = 0;

        for (int64_t it = 0; it < static_cast<int64_t>(radixVec.size()); ++it) {
            tempNi = radixVec[it];
            if (tempNi == 0) {
                continue;
            }

            singleSize = tempNi * tempNi * 2 * 2;

            if (radixVec.size() > 1 && it == static_cast<int64_t>(radixVec.size()) - 1) {
                for (int64_t k = 0; k < tempNi * tempNi; ++k) {
                    int64_t i = k / tempNi;
                    int64_t j = k % tempNi;
                    *(wMatrixHost + offset + i * 2 * tempNi + j) = cos(-1.0 * K_2PI / tempNi * i * j);
                    *(wMatrixHost + offset + i * 2 * tempNi + tempNi + j) = sin(-1.0 * K_2PI / tempNi * i * j) *
                                                                            (forward ? (-1.0) : (1.0));
                    *(wMatrixHost + offset + (i + tempNi) * 2 * tempNi + j) = sin(1.0 * K_2PI / tempNi * i * j) *
                                                                              (forward ? (-1.0) : (1.0));
                    *(wMatrixHost + offset + (i + tempNi) * 2 * tempNi + tempNi +
                      j) = cos(-1.0 * K_2PI / tempNi * i * j);
                }
            } else {
                for (int64_t k = 0; k < tempNi * tempNi; ++k) {
                    int64_t i = k / tempNi;
                    int64_t j = k % tempNi;
                    *(wMatrixHost + offset + 2 * i * 2 * tempNi + j) = cos(-1.0 * K_2PI / tempNi * i * j);
                    *(wMatrixHost + offset + 2 * i * 2 * tempNi + tempNi + j) = sin(-1.0 * K_2PI / tempNi * i * j) *
                                                                                (forward ? (-1.0) : (1.0));
                    *(wMatrixHost + offset + (2 * i + 1) * 2 * tempNi + j) = sin(1.0 * K_2PI / tempNi * i * j) *
                                                                             (forward ? (-1.0) : (1.0));
                    *(wMatrixHost + offset + (2 * i + 1) * 2 * tempNi + tempNi +
                      j) = cos(-1.0 * K_2PI / tempNi * i * j);
                }
            }

            offset += singleSize;
        }

        wMatrix_.hostData = wMatrixHost;
        return wMatrixPtr;
    };

    AsdSip::CoeffKey key = {coreType, W_MARK, {n}, forward};
    wMatrix = FFTensorCache::getCoeff(key, func);

    ASDSIP_LOG(INFO) << "Init WMatrixCommon success.";
    return AsdSip::ErrorType::ACL_SUCCESS;
}

AspbStatus InitTMatrixCommon(FFTCoreType coreType, std::vector<int64_t>& radixVec, int64_t n,
                             std::shared_ptr<AsdSip::FFTensor>& tMatrix)
{
    if (n == RADIX_8K) {
        radixVec = {2, 64, 64};
    }
    if (n == RADIX_16K) {
        radixVec = {4, 64, 64};
    }

    std::function<AsdSip::FFTensor*()> func = [=]() mutable -> AsdSip::FFTensor* {
        AsdSip::FFTensor* tMatrixPtr = new AsdSip::FFTensor;
        AsdSip::FFTensor& tMatrix_ = *tMatrixPtr;

        tMatrix_.desc.dtype = TENSOR_DTYPE_FLOAT;
        tMatrix_.desc.format = TENSOR_FORMAT_ND;

        int64_t eleNum = GetAllTMatrixSize(radixVec);
        tMatrix_.desc.dims = {eleNum};
        tMatrix_.dataSize = sizeof(float) * eleNum;

        float* tMatrixHost = nullptr;
        try {
            tMatrixHost = new float[eleNum];
        } catch (std::bad_alloc& e) {
            delete tMatrixPtr;
            ASDSIP_LOG(ERROR) << "tMatrixHost alloc failed: " << e.what();
            throw std::runtime_error("tMatrixHost alloc failed:.");
        }
        InitMem(tMatrixHost, 0, tMatrix_.dataSize);

        int64_t tempRow;
        int64_t tempCol;
        int64_t singleSize;
        int64_t offset = 0;

        for (int64_t it = 0; it < static_cast<int64_t>(radixVec.size()) - 1; ++it) {
            tempRow = radixVec[it];
            tempCol = GetTMatrixCol(radixVec, it);
            singleSize = tempRow * 2 * tempCol;

            for (int64_t i = 0; i < tempRow; ++i) {
                for (int64_t j = 0; j < tempCol; ++j) {
                    *((tMatrixHost + offset) + 2 * i * tempCol + j) = cos(-1.0 * K_2PI / (tempRow * tempCol) * i * j);
                    *((tMatrixHost + offset) + (2 * i + 1) * tempCol +
                      j) = sin(-1.0 * K_2PI / (tempRow * tempCol) * i * j);
                }
            }

            offset += singleSize;
        }

        tMatrix_.hostData = tMatrixHost;

        return tMatrixPtr;
    };

    AsdSip::CoeffKey key = {coreType, T_MARK, {n}, 0};
    tMatrix = FFTensorCache::getCoeff(key, func);

    if (n == RADIX_8K || n == RADIX_16K) {
        radixVec = {64, 64};
    }

    ASDSIP_LOG(INFO) << "Init TMatrixCommon success.";
    return AsdSip::ErrorType::ACL_SUCCESS;
}

AspbStatus InitSMatrixCommon(FFTCoreType coreType, std::vector<int64_t>& radixVec, int64_t n, bool forward,
                             std::shared_ptr<AsdSip::FFTensor>& sMatrix)
{
    std::function<AsdSip::FFTensor*()> func = [=]() mutable -> AsdSip::FFTensor* {
        AsdSip::FFTensor* sMatrixPtr = new AsdSip::FFTensor;
        AsdSip::FFTensor& sMatrix_ = *sMatrixPtr;

        int64_t size = GetAllSMatrixSize(radixVec);

        sMatrix_.desc.dtype = TENSOR_DTYPE_FLOAT;
        sMatrix_.desc.format = TENSOR_FORMAT_ND;
        sMatrix_.desc.dims = {size};
        sMatrix_.dataSize = sizeof(float) * size;

        float* sMatrixHost = nullptr;
        try {
            sMatrixHost = new float[size];
        } catch (std::bad_alloc& e) {
            delete sMatrixPtr;
            ASDSIP_LOG(ERROR) << "sMatrixHost alloc failed: " << e.what();
            throw std::runtime_error("sMatrixHost alloc failed:.");
        }
        InitMem(sMatrixHost, 0, sMatrix_.dataSize);

        int64_t tempRepeat = 1;
        int64_t tempN;
        int64_t singleSize;
        int64_t offset = 0;

        for (int64_t it = 0; it < static_cast<int64_t>(radixVec.size()); ++it) {
            tempN = radixVec[it];
            if (tempN == 0) {
                continue;
            }
            singleSize = tempRepeat * tempN * tempN * 2 * 2;
            for (int64_t i = 0; i < tempRepeat; ++i) {
                for (int64_t m = 0; m < tempN * tempN; ++m) {
                    int64_t j = m / tempN;
                    int64_t k = m % tempN;
                    *((sMatrixHost + offset) + i * 4 * tempN * tempN + 2 * j * 2 * tempN +
                      k) = cos(-K_2PI * (i + j * tempRepeat) * k / (tempRepeat * tempN));
                    *((sMatrixHost + offset) + i * 4 * tempN * tempN + 2 * j * 2 * tempN + k +
                      tempN) = sin(-K_2PI * (i + j * tempRepeat) * k / (tempRepeat * tempN)) *
                               (forward ? (-1.0) : (1.0));
                    *((sMatrixHost + offset) + i * 4 * tempN * tempN + (2 * j + 1) * 2 * tempN +
                      k) = sin(-K_2PI * (i + j * tempRepeat) * k / (tempRepeat * tempN)) * (forward ? (1.0) : (-1.0));
                    *((sMatrixHost + offset) + i * 4 * tempN * tempN + (2 * j + 1) * 2 * tempN + k +
                      tempN) = cos(-K_2PI * (i + j * tempRepeat) * k / (tempRepeat * tempN));
                }
            }

            offset += singleSize;
            tempRepeat *= tempN;
        }

        sMatrix_.hostData = sMatrixHost;
        return sMatrixPtr;
    };

    AsdSip::CoeffKey key = {coreType, S_MARK, {n}, forward};
    sMatrix = FFTensorCache::getCoeff(key, func);

    ASDSIP_LOG(INFO) << "Init SMatrixCommon success.";
    return AsdSip::ErrorType::ACL_SUCCESS;
}

// inputIndexDevice && outputIndexDevices
AspbStatus InitInputOutputIndex(FFTCoreType coreType, int64_t fftN, int parity,
                                std::shared_ptr<AsdSip::FFTensor>& inputIndex,
                                std::shared_ptr<AsdSip::FFTensor>& outputIndex)
{
    int64_t n = fftN * 2;
    int64_t biasc = BIASC_SIZE;
    int64_t bias = BIAS_SIZE;
    int64_t indexSize = n >= biasc ? bias : ((n / 2 + 63) / 64) * 64;
    int64_t outIndexSize = indexSize * 2;

    bool isR2c = coreType == FFTCoreType::kDftR2C || coreType == FFTCoreType::kFftR2C;
    bool isDft = coreType == FFTCoreType::kDftR2C || coreType == FFTCoreType::kDftC2R;

    std::function<AsdSip::FFTensor*()> input_index_func = [=]() -> AsdSip::FFTensor* {
        AsdSip::FFTensor* _input_index_ptr = new AsdSip::FFTensor;
        AsdSip::FFTensor& _input_index = *_input_index_ptr;

        uint32_t* _input_index_host = nullptr;
        try {
            _input_index_host = new uint32_t[biasc * 4];
        } catch (std::bad_alloc& e) {
            delete _input_index_ptr;
            ASDSIP_LOG(ERROR) << "_input_index_host alloc failed: " << e.what();
            throw std::runtime_error("_input_index_host alloc failed:.");
        }
        InitMem(_input_index_host, 0, biasc * 4 * sizeof(uint32_t));

        if (isR2c) {
            if (isDft) {
                for (uint32_t i = 0; i < BIASC_SIZE; i++) {
                    *(_input_index_host + RADIX_8K + 2 * i) = static_cast<uint32_t>((i) * sizeof(float));
                    *(_input_index_host + RADIX_8K + 2 * i +
                      1) = static_cast<uint32_t>((BIASC_SIZE + i) * sizeof(float));
                }
            }
            if (indexSize == bias) {
                if (n > biasc) {
                    for (uint32_t i = 0; i < indexSize; i++) {
                        *(_input_index_host + i) = static_cast<uint32_t>((bias - 1 - i) * sizeof(float));
                    }
                    uint32_t remainSize = (n - 2) % biasc; // 超过 4096长度的尾块处理
                    for (uint32_t i = 0; i < remainSize / 2; i++) {
                        *(_input_index_host + indexSize +
                          i) = static_cast<uint32_t>((remainSize / 2 - 1 - i) * sizeof(float));
                    }
                    for (uint32_t i = (remainSize / 2); i < indexSize; i++) {
                        *(_input_index_host + indexSize + i) = static_cast<uint32_t>((0) * sizeof(float));
                    }
                } else {
                    for (int i = 0; i < (n / 2 - 1); i++) {
                        *(_input_index_host + i) = (n / 2 - 2 - i) * sizeof(float);
                    }
                    for (int i = n / 2 - 1; i < indexSize; i++) {
                        *(_input_index_host + i) = 0 * sizeof(float);
                    }
                }
            } else {
                for (int i = 0; i < (n / 2 - 1); i++) {
                    *(_input_index_host + i) = (n / 2 - 2 - i) * sizeof(float);
                }
                for (int i = n / 2 - 1; i < indexSize; i++) {
                    *(_input_index_host + i) = 0 * sizeof(float);
                }
            }
        } else {
            if (parity == 1) {
                if ((fftN / 2 * 2 > BIASC_SIZE)) { // 需要分多次进行C2R的共轭操作
                    uint32_t idx_size = biasc;
                    for (uint32_t i = 0; i < idx_size / 2; i++) {
                        *(_input_index_host + i * 2) = static_cast<uint32_t>((idx_size - i * 2) * sizeof(float));
                        *(_input_index_host + i * 2 +
                          1) = static_cast<uint32_t>((idx_size - i * 2 + 1) * sizeof(float));
                    }

                    for (uint32_t i = 0; i < idx_size / 2; i++) {
                        *(_input_index_host + idx_size +
                          i * 2) = static_cast<uint32_t>((idx_size - i * 2 - 2) * sizeof(float));
                        *(_input_index_host + idx_size + i * 2 +
                          1) = static_cast<uint32_t>((idx_size - i * 2 - 1) * sizeof(float));
                    }
                    uint32_t idx_remain = (fftN / 2 * 2) % biasc;
                    uint32_t idx_remain_padding = (idx_remain + 63) / 64 * 64;
                    for (uint32_t i = 0; i < idx_remain / 2; i++) {
                        *(_input_index_host + idx_size * 2 +
                          i * 2) = static_cast<uint32_t>((idx_remain - i * 2 - 2) * sizeof(float));
                        *(_input_index_host + idx_size * 2 + i * 2 +
                          1) = static_cast<uint32_t>((idx_remain - i * 2 - 1) * sizeof(float));
                    }
                    for (uint32_t i = 0; i < (idx_remain_padding - idx_remain); i++) {
                        *(_input_index_host + idx_size * 2 + idx_remain + i) = 0;
                    }
                } else {
                    uint32_t idx_size = (fftN / 2) * 2; // 与算子逻辑一致
                    uint32_t idx_padding = (idx_size + 63) / 64 * 64;
                    for (uint32_t i = 0; i < idx_size / 2; i++) {
                        *(_input_index_host +
                          i * 2) = static_cast<uint32_t>((idx_size + 2 - i * 2 - 2) * sizeof(float));
                        *(_input_index_host + i * 2 +
                          1) = static_cast<uint32_t>((idx_size + 2 - i * 2 - 1) * sizeof(float));
                    }
                    for (uint32_t i = 0; i < (idx_padding - idx_size); i++) {
                        *(_input_index_host + idx_size + i) = 0;
                    }
                }
            } else {
                if (indexSize == bias) {
                    if (n > biasc) {
                        for (uint32_t i = 0; i < indexSize; i++) {
                            *(_input_index_host + i) = static_cast<uint32_t>((bias - 1 - i) * sizeof(float));
                        }
                        uint32_t remainSize = (n + 2) % biasc; // 超过 4096长度的尾块处理
                        for (uint32_t i = 0; i < remainSize / 2; i++) {
                            *(_input_index_host + indexSize +
                              i) = static_cast<uint32_t>((remainSize / 2 - 1 - i) * sizeof(float));
                        }
                        for (uint32_t i = (remainSize / 2); i < indexSize; i++) {
                            *(_input_index_host + indexSize + i) = static_cast<uint32_t>((0) * sizeof(float));
                        }
                    } else {
                        if (n == BIASC_SIZE) { // 此分支只有C2R 4096会走到
                            for (int i = 0; i < (n / 2); i++) {
                                *(_input_index_host + i) = (n / 2 - i - 1) * sizeof(float);
                            }
                        } else {
                            for (int i = 0; i < (n / 2); i++) {
                                *(_input_index_host + i) = (n / 2 - i) * sizeof(float);
                            }
                        }
                        int32_t gather_num = n + 2;
                        int32_t index_size_padding = ((gather_num / 2 + 63) / 64) * 64;
                        for (int i = (n / 2); i < index_size_padding; i++) {
                            *(_input_index_host + i) = 0 * sizeof(float);
                        }
                    }
                } else {
                    for (int i = 0; i < (n / 2); i++) {
                        *(_input_index_host + i) = (n / 2 - i) * sizeof(float);
                    }
                    int32_t gather_num = n + 2;
                    int32_t index_size_padding = ((gather_num / 2 + 63) / 64) * 64;
                    for (int i = (n / 2); i < index_size_padding; i++) {
                        *(_input_index_host + i) = 0 * sizeof(float);
                    }
                }
            }
        };

        _input_index.desc = {
            Mki::TensorDType::TENSOR_DTYPE_UINT32, Mki::TensorFormat::TENSOR_FORMAT_ND, {biasc * 4}, {}, 0};
        _input_index.hostData = _input_index_host;
        _input_index.dataSize = biasc * 4 * sizeof(uint32_t);

        return _input_index_ptr;
    };

    AsdSip::CoeffKey input_index_key = {coreType, INPUT_MARK, {parity == 1 ? fftN : fftN * 2}, 0};
    inputIndex = FFTensorCache::getCoeff(input_index_key, input_index_func);

    std::function<AsdSip::FFTensor*()> output_index_func = [=]() -> AsdSip::FFTensor* {
        AsdSip::FFTensor* _output_index_ptr = new AsdSip::FFTensor;
        AsdSip::FFTensor& _output_index = *_output_index_ptr;

        uint32_t* _output_index_host = nullptr;
        try {
            _output_index_host = new uint32_t[outIndexSize];
        } catch (std::bad_alloc& e) {
            delete _output_index_ptr;
            ASDSIP_LOG(ERROR) << "_output_index_host alloc failed: " << e.what();
            throw std::runtime_error("_output_index_host alloc failed:.");
        }
        InitMem(_output_index_host, 0, outIndexSize * sizeof(uint32_t));

        for (int i = 0; i < indexSize; i++) {
            *(_output_index_host + i * 2) = i * 4;
            *(_output_index_host + i * 2 + 1) = (BIAS_SIZE + i) * 4;
        }

        _output_index.desc = {
            Mki::TensorDType::TENSOR_DTYPE_UINT32, Mki::TensorFormat::TENSOR_FORMAT_ND, {outIndexSize}, {}, 0};
        _output_index.hostData = _output_index_host;
        _output_index.dataSize = outIndexSize * sizeof(uint32_t);

        return _output_index_ptr;
    };

    AsdSip::CoeffKey output_index_key = {coreType, OUTPUT_MARK, {parity == 1 ? fftN : fftN * 2}, 0};
    outputIndex = FFTensorCache::getCoeff(output_index_key, output_index_func);

    return AsdSip::ErrorType::ACL_SUCCESS;
}

// inputADevice && inputBDevice
AspbStatus InitABTable(FFTCoreType coreType, int64_t fftN, bool forward, int parity,
                       std::shared_ptr<AsdSip::FFTensor>& aTable, std::shared_ptr<AsdSip::FFTensor>& bTable)
{
    int32_t tableSize = 0;

    std::unique_ptr<double[]> cosTable{nullptr};
    std::unique_ptr<double[]> sinTable{nullptr};
    if (parity == 0) {
        tableSize = fftN;

        cosTable.reset(new double[tableSize]);
        sinTable.reset(new double[tableSize]);
        InitMem(cosTable.get(), 0, tableSize * sizeof(double));
        InitMem(sinTable.get(), 0, tableSize * sizeof(double));

        for (int i = 0; i < tableSize; i++) {
            cosTable[i] = std::cos(K_PI * i / tableSize);
            sinTable[i] = std::sin(K_PI * i / tableSize);
        }
    }

    bool isR2c = coreType == FFTCoreType::kDftR2C || coreType == FFTCoreType::kFftR2C;
    double factor = 1.0;
    if ((isR2c && !forward) || (!isR2c && forward)) {
        factor = -1.0;
    }

    std::function<AsdSip::FFTensor*()> a_table_func = [=, &cosTable, &sinTable]() -> AsdSip::FFTensor* {
        AsdSip::FFTensor* aTablePtr = new AsdSip::FFTensor;
        AsdSip::FFTensor& aTableRef = *aTablePtr;

        float* _a_table_host = nullptr;

        if (parity == 0) {
            try {
                _a_table_host = new float[tableSize * 2];
            } catch (std::bad_alloc& e) {
                delete aTablePtr;
                ASDSIP_LOG(ERROR) << "_a_table_host alloc failed: " << e.what();
                throw std::runtime_error("_a_table_host alloc failed:.");
            }
            InitMem(_a_table_host, 0, sizeof(float) * tableSize * 2);

            for (int32_t i = 0; i < tableSize; ++i) {
                if (isR2c) {
                    *(_a_table_host + i) = (1.0 - factor * (sinTable[i])) / 2.0;
                    *(_a_table_host + tableSize + i) = cosTable[i] / (-2.0);
                } else {
                    *(_a_table_host + i) = (1.0 - factor * (sinTable[i]));
                    *(_a_table_host + tableSize + i) = cosTable[i];
                }
            }
        }

        aTableRef.desc = {
            Mki::TensorDType::TENSOR_DTYPE_FLOAT, Mki::TensorFormat::TENSOR_FORMAT_ND, {tableSize * 2}, {}, 0};
        aTableRef.hostData = _a_table_host;
        aTableRef.dataSize = tableSize * 2 * sizeof(float);

        return aTablePtr;
    };

    AsdSip::CoeffKey a_table_key = {coreType, A_MARK, {parity == 1 ? fftN : fftN * 2}, 0};
    aTable = FFTensorCache::getCoeff(a_table_key, a_table_func);

    std::function<AsdSip::FFTensor*()> b_table_func = [=, &cosTable, &sinTable]() -> AsdSip::FFTensor* {
        AsdSip::FFTensor* _b_table_ptr = new AsdSip::FFTensor;
        AsdSip::FFTensor& _b_table = *_b_table_ptr;

        float* _b_table_host = nullptr;

        if (parity == 0) {
            try {
                _b_table_host = new float[tableSize * 2];
            } catch (std::bad_alloc& e) {
                delete _b_table_ptr;
                ASDSIP_LOG(ERROR) << "_b_table_host alloc failed: " << e.what();
                throw std::runtime_error("_b_table_host alloc failed:.");
            }
            InitMem(_b_table_host, 0, sizeof(float) * tableSize * 2);

            for (int32_t i = 0; i < tableSize; ++i) {
                if (isR2c) {
                    *(_b_table_host + i) = (1.0 + factor * (sinTable[i])) / 2.0;
                    *(_b_table_host + tableSize + i) = cosTable[i] / 2.0;
                } else {
                    *(_b_table_host + i) = (1.0 + factor * (sinTable[i]));
                    *(_b_table_host + tableSize + i) = -1.0 * cosTable[i];
                }
            }
        }

        _b_table.desc = {
            Mki::TensorDType::TENSOR_DTYPE_FLOAT, Mki::TensorFormat::TENSOR_FORMAT_ND, {tableSize * 2}, {}, 0};
        _b_table.hostData = _b_table_host;
        _b_table.dataSize = tableSize * 2 * sizeof(float);

        return _b_table_ptr;
    };

    AsdSip::CoeffKey b_table_key = {coreType, B_MARK, {parity == 1 ? fftN : fftN * 2}, 0};
    bTable = FFTensorCache::getCoeff(b_table_key, b_table_func);

    return AsdSip::ErrorType::ACL_SUCCESS;
}

void InitMixRadixLong(int64_t fftN, std::vector<int64_t>& radixVecRef)
{
    std::vector<int64_t> radixArr;
    for (int64_t i = RADIX_ADDR_START; i >= RADIX_ADDR_END; i--) {
        radixArr.push_back(i);
    }

    std::vector<int64_t> radixList;

    int64_t inputLen = fftN;

    if (inputLen == RADIX_LEN_19683) {
        radixList = {27, 9, 81};
        radixVecRef = radixList;
        return;
    } else if (inputLen == RADIX_LEN_243) {
        radixList = {27, 9, 81};
        radixVecRef = radixList;
        return;
    } else if (inputLen == RADIX_LEN_729) {
        radixList = {27, 27};
        radixVecRef = radixList;
        return;
    } else if (inputLen == RADIX_LEN_59049) {
        radixList = {27, 27, 81};
        radixVecRef = radixList;
        return;
    } else if (inputLen == RADIX_LEN_177147) {
        radixList = {27, 81, 81};
        radixVecRef = radixList;
        return;
    } else if (inputLen == RADIX_LEN_1594323) {
        radixList = {27, 27, 27, 81};
        radixVecRef = radixList;
        return;
    } else if (inputLen == RADIX_LEN_129140163) {
        radixList = {27, 27, 27, 81, 81};
        radixVecRef = radixList;
        return;
    } else if (inputLen == RADIX_LEN_387420489) {
        radixList = {27, 27, 81, 81, 81};
        radixVecRef = radixList;
        return;
    } else if (inputLen == RADIX_LEN_625) {
        radixList = {25, 25};
        radixVecRef = radixList;
        return;
    } else if (inputLen == RADIX_LEN_3125) {
        radixList = {25, 125};
        radixVecRef = radixList;
        return;
    } else if (inputLen == RADIX_LEN_15625) {
        radixList = {125, 125};
        radixVecRef = radixList;
        return;
    } else if (inputLen == RADIX_LEN_78125) {
        radixList = {25, 25, 125};
        radixVecRef = radixList;
        return;
    } else if (inputLen == RADIX_LEN_390625) {
        radixList = {25, 125, 125};
        radixVecRef = radixList;
        return;
    } else if (inputLen == RADIX_LEN_2401) {
        radixList = {49, 49};
        radixVecRef = radixList;
        return;
    } else if (inputLen == RADIX_LEN_16807) {
        radixList = {7, 49, 49};
        radixVecRef = radixList;
        return;
    } else if (inputLen == RADIX_LEN_117649) {
        radixList = {49, 49, 49};
        radixVecRef = radixList;
        return;
    } else if (inputLen == RADIX_LEN_76800) {
        radixList = {8, 80, 120};
        radixVecRef = radixList;
        return;
    }

    for (int64_t r : radixArr) {
        while (inputLen % r == 0) {
            radixList.push_back(r);
            inputLen /= r;
        }
    }

    std::sort(radixList.begin(), radixList.end());
    radixVecRef = radixList;
}

void InitMixRadixShort(int64_t fftN, std::vector<int64_t>& radixVecRef)
{
    int64_t n = fftN;
    std::vector<int64_t> radixList = {};
    if (fftN <= RADIX_ADDR_START) {
        radixList.push_back(fftN);
        radixVecRef = radixList;
        return;
    }
    // 埃式筛筛选质数
    std::vector<int64_t> primes = {};
    std::vector<bool> minp(n + 1, true);
    for (int64_t i = 2; i < (n + 1); i++) {
        if (minp[i]) {
            primes.push_back(i);
        }
        for (int64_t j = i * i; j < (n + 1); j += i) {
            minp[j] = false;
        }
    }
    minp.clear();
    // 质因数分解
    std::vector<int64_t> factors = {};
    for (int64_t i = 0; i < static_cast<int64_t>(primes.size()); i++) {
        int64_t prime = primes[i];
        if (prime == 0) {
            continue;
        }
        while (n % prime == 0) {
            factors.push_back(prime);
            n /= prime;
        }
        if (n == 1) {
            break;
        }
    }
    primes.clear();
    // 基于因数分解选择均衡的基
    radixList = FindTwoRadix(factors, fftN);
    if (radixList[1] > RADIX_ADDR_START) {
        std::vector<int64_t> newRadixList = {};
        int64_t len = static_cast<int64_t>(factors.size());
        for (int64_t i = 0; i < len; i++) {
            int64_t t = *(factors.begin() + i);
            if (t == 0) {
                continue;
            }
            factors.erase(factors.begin() + i);
            newRadixList = FindTwoRadix(factors, fftN / t);
            factors.insert(factors.begin() + i, t);
            if (newRadixList[1] <= RADIX_ADDR_START) {
                newRadixList.insert(newRadixList.begin(), t);
                radixList = newRadixList;
                break;
            }
        }
    }

    radixVecRef = radixList;
}

AspbStatus InitMixRadixList(FFTCoreType coreType, int64_t fftN, bool forward, std::vector<int64_t>& radixVecRef,
                            std::shared_ptr<AsdSip::FFTensor>& radixList)
{
    int64_t* radixListRefPtr = static_cast<int64_t*>(radixVecRef.data());
    size_t radixListLen = radixVecRef.size();

    std::function<AsdSip::FFTensor*()> func = [=]() mutable -> AsdSip::FFTensor* {
        AsdSip::FFTensor* radixListPtr = new AsdSip::FFTensor;
        AsdSip::FFTensor& radixList_ = *radixListPtr;

        int32_t* radixListHost = new (std::nothrow) int32_t[radixListLen];
        if (radixListHost == nullptr) {
            delete radixListPtr;
            ASDSIP_LOG(ERROR) << "radixListHost alloc failed: ";
            throw std::runtime_error("radixListHost alloc failed:.");
        }
        InitMem(radixListHost, 0, sizeof(int32_t) * radixListLen);

        for (int64_t i = 0; i < static_cast<int64_t>(radixListLen); i++) {
            radixListHost[i] = radixListRefPtr[i];
        }

        radixList_.desc = {Mki::TensorDType::TENSOR_DTYPE_INT32,
                           Mki::TensorFormat::TENSOR_FORMAT_ND,
                           {static_cast<int64_t>(radixListLen)},
                           {},
                           0};
        radixList_.hostData = radixListHost;
        radixList_.dataSize = sizeof(int32_t) * radixListLen;

        return radixListPtr;
    };

    AsdSip::CoeffKey key = {coreType, MIX_RADIX_LIST_MARK, {fftN}, forward};
    radixList = FFTensorCache::getCoeff(key, func);

    return AsdSip::ErrorType::ACL_SUCCESS;
}

namespace {

AspbStatus InitMixRadixForwardTwiddleMatrix(FFTCoreType coreType, int64_t fftN, int64_t batchSize,
                                            std::vector<int64_t>& radixVecRef,
                                            std::shared_ptr<AsdSip::FFTensor>& dftMatrixArray)
{
    int64_t dftMatrixLen = GetTwiddleMatrixLen(fftN, radixVecRef);
    int64_t* radixListPtr = static_cast<int64_t*>(radixVecRef.data());
    int64_t radixListLen = static_cast<int64_t>(radixVecRef.size());

    std::function<AsdSip::FFTensor*()> funcMultiSteplen = [=]() mutable -> AsdSip::FFTensor* {
        AsdSip::FFTensor* dftMatrixArrayPtr = new AsdSip::FFTensor;
        AsdSip::FFTensor& dftMatrixArray_ = *dftMatrixArrayPtr;

        float* matrixArrayHost = new (std::nothrow) float[dftMatrixLen];
        if (matrixArrayHost == nullptr) {
            delete dftMatrixArrayPtr;
            ASDSIP_LOG(ERROR) << "matrixArrayHost alloc failed: ";
            throw std::runtime_error("matrixArrayHost alloc failed:.");
        }
        InitMem(matrixArrayHost, 0, dftMatrixLen * sizeof(float));

        // 构造DFT矩阵
        auto nowMatrixArrayHost = matrixArrayHost;
        int64_t stepLen = radixListLen;

        // 生成正向所需的W矩阵
        GenWMatrixForwardForMultiLen(stepLen, radixListPtr, fftN, nowMatrixArrayHost);

        dftMatrixArray_.desc = {
            Mki::TensorDType::TENSOR_DTYPE_FLOAT, Mki::TensorFormat::TENSOR_FORMAT_ND, {dftMatrixLen}, {}, 0};
        dftMatrixArray_.hostData = matrixArrayHost;
        dftMatrixArray_.dataSize = dftMatrixLen * sizeof(float);

        return dftMatrixArrayPtr;
    };

    std::function<AsdSip::FFTensor*()> funcOneSteplen = [=]() mutable -> AsdSip::FFTensor* {
        AsdSip::FFTensor* dftMatrixArrayPtr = new AsdSip::FFTensor;
        AsdSip::FFTensor& dftMatrixArray_ = *dftMatrixArrayPtr;

        float* dftMatrixArrayHost = new (std::nothrow) float[dftMatrixLen];
        if (dftMatrixArrayHost == nullptr) {
            delete dftMatrixArrayPtr;
            ASDSIP_LOG(ERROR) << "dftMatrixArrayHost alloc failed: ";
            throw std::runtime_error("dftMatrixArrayHost alloc failed:.");
        }
        InitMem(dftMatrixArrayHost, 0, dftMatrixLen * sizeof(float));

        // 构造DFT矩阵
        auto nowDftMatrixArrayHost = dftMatrixArrayHost;
        int64_t stepLen = radixListLen;

        GenWMatrixForwardForOneLen(stepLen, radixListPtr, fftN, nowDftMatrixArrayHost);

        dftMatrixArray_.desc = {
            Mki::TensorDType::TENSOR_DTYPE_FLOAT, Mki::TensorFormat::TENSOR_FORMAT_ND, {dftMatrixLen}, {}, 0};
        dftMatrixArray_.hostData = dftMatrixArrayHost;
        dftMatrixArray_.dataSize = dftMatrixLen * sizeof(float);

        return dftMatrixArrayPtr;
    };

    AsdSip::CoeffKey key = {coreType, MIX_FORWARD_MARK, {batchSize, fftN}, true};
    dftMatrixArray = FFTensorCache::getCoeff(key, radixListLen > 1 ? funcMultiSteplen : funcOneSteplen);

    return AsdSip::ErrorType::ACL_SUCCESS;
}

AspbStatus InitMixRadixInverseTwiddleMatrix(FFTCoreType coreType, int64_t fftN, int64_t batchSize,
                                            std::vector<int64_t>& radixVecRef,
                                            std::shared_ptr<AsdSip::FFTensor>& dftMatrixArrayInverse)
{
    int64_t dftMatrixLen = GetTwiddleMatrixLen(fftN, radixVecRef);
    int64_t* radixListPtr = static_cast<int64_t*>(radixVecRef.data());
    int64_t radixListLen = static_cast<int64_t>(radixVecRef.size());

    std::function<AsdSip::FFTensor*()> funcMultiSteplen = [=]() mutable -> AsdSip::FFTensor* {
        AsdSip::FFTensor* dftMatrixArrayInversePtr = new AsdSip::FFTensor;
        AsdSip::FFTensor& dftMatrixArrayInverse_ = *dftMatrixArrayInversePtr;

        float* dftMatrixArrayInverseHost = new (std::nothrow) float[dftMatrixLen];
        if (dftMatrixArrayInverseHost == nullptr) {
            delete dftMatrixArrayInversePtr;
            ASDSIP_LOG(ERROR) << "dftMatrixArrayInverseHost malloc failed: ";
            throw std::runtime_error("dftMatrixArrayInverseHost malloc failed:.");
        }

        InitMem(dftMatrixArrayInverseHost, 0, dftMatrixLen * sizeof(float));

        auto nowDftMatrixArrayInverseHost = dftMatrixArrayInverseHost;
        int64_t stepLen = radixListLen;

        GenWMatrixInverseForMultiLen(stepLen, radixListPtr, fftN, nowDftMatrixArrayInverseHost);

        dftMatrixArrayInverse_.desc = {
            Mki::TensorDType::TENSOR_DTYPE_FLOAT, Mki::TensorFormat::TENSOR_FORMAT_ND, {dftMatrixLen}, {}, 0};
        dftMatrixArrayInverse_.hostData = dftMatrixArrayInverseHost;
        dftMatrixArrayInverse_.dataSize = dftMatrixLen * sizeof(float);

        return dftMatrixArrayInversePtr;
    };

    std::function<AsdSip::FFTensor*()> funcOneSteplen = [=]() mutable -> AsdSip::FFTensor* {
        AsdSip::FFTensor* dftMatrixArrayInversePtr = new AsdSip::FFTensor;
        AsdSip::FFTensor& dftMatrixArrayInverse_ = *dftMatrixArrayInversePtr;

        float* dftMatrixArrayInverseHost = new (std::nothrow) float[dftMatrixLen];
        if (dftMatrixArrayInverseHost == nullptr) {
            delete dftMatrixArrayInversePtr;
            ASDSIP_LOG(ERROR) << "dftMatrixArrayInverseHost malloc failed: ";
            throw std::runtime_error("dftMatrixArrayInverseHost malloc failed:.");
        }

        InitMem(dftMatrixArrayInverseHost, 0, dftMatrixLen * sizeof(float));

        auto nowDftMatrixArrayInverseHost = dftMatrixArrayInverseHost;
        int64_t stepLen = radixListLen;
        GenWMatrixInverseForOneLen(stepLen, radixListPtr, fftN, nowDftMatrixArrayInverseHost);

        dftMatrixArrayInverse_.desc = {
            Mki::TensorDType::TENSOR_DTYPE_FLOAT, Mki::TensorFormat::TENSOR_FORMAT_ND, {dftMatrixLen}, {}, 0};
        dftMatrixArrayInverse_.hostData = dftMatrixArrayInverseHost;
        dftMatrixArrayInverse_.dataSize = dftMatrixLen * sizeof(float);

        return dftMatrixArrayInversePtr;
    };

    AsdSip::CoeffKey key = {coreType, MIX_BACKWARD_MARK, {batchSize, fftN}, true};
    dftMatrixArrayInverse = FFTensorCache::getCoeff(key, radixListLen > 1 ? funcMultiSteplen : funcOneSteplen);

    return AsdSip::ErrorType::ACL_SUCCESS;
}

} // namespace

AspbStatus InitMixRadixTwiddleMatrix(FFTCoreType coreType, int64_t fftN, int64_t batchSize, bool forward,
                                     std::vector<int64_t>& radixVecRef,
                                     std::shared_ptr<AsdSip::FFTensor>& twiddleMatrixArray)
{
    if (forward) {
        return InitMixRadixForwardTwiddleMatrix(coreType, fftN, batchSize, radixVecRef, twiddleMatrixArray);
    } else {
        return InitMixRadixInverseTwiddleMatrix(coreType, fftN, batchSize, radixVecRef, twiddleMatrixArray);
    }
}

namespace {

int64_t GetTwMatrixLen(int64_t fftN, const std::vector<int64_t>& radixVecRef)
{
    int64_t n = fftN;
    int64_t twMatrixLen = 0;
    if (radixVecRef.size() == 1) {
        return twMatrixLen;
    }
    int64_t n0 = 1;
    int64_t stepLen = static_cast<int64_t>(radixVecRef.size());
    for (int64_t stepIndex = 0; stepIndex < stepLen; stepIndex++) {
        int64_t n1 = radixVecRef[stepIndex];
        if (n1 == 0) {
            continue;
        }
        int64_t n2 = n / n1 / n0;

        int32_t tileM0;
        int32_t tileK0;
        int32_t tileN0;

        getTile(n1, (n2 > 1) ? n2 : n0, stepIndex, stepLen, tileM0, tileN0, tileK0);

        twMatrixLen += TWO_MUL * n1 * ROUND(n2, tileN0);
        n0 *= n1;
    }

    return twMatrixLen;
}

} // namespace

AspbStatus InitMixRadixTWMatrix(FFTCoreType coreType, int64_t fftN, bool forward, std::vector<int64_t>& radixVecRef,
                                std::shared_ptr<AsdSip::FFTensor>& twMatrixArray)
{
    int64_t twMatrixLen = GetTwMatrixLen(fftN, radixVecRef);
    int64_t* radixListPtr = static_cast<int64_t*>(radixVecRef.data());
    int64_t radixListLen = static_cast<int64_t>(radixVecRef.size());

    std::function<AsdSip::FFTensor*()> func = [=]() mutable -> AsdSip::FFTensor* {
        AsdSip::FFTensor* twMatrixArrayPtr = new AsdSip::FFTensor;
        AsdSip::FFTensor& twMatrixArray_ = *twMatrixArrayPtr;

        float* twMatrixArrayHost = nullptr;

        if (twMatrixLen > 0) {
            try {
                twMatrixArrayHost = new float[twMatrixLen];
            } catch (std::bad_alloc& e) {
                delete twMatrixArrayPtr;
                ASDSIP_LOG(ERROR) << "twMatrixArrayHost alloc failed: " << e.what();
                throw std::runtime_error("twMatrixArrayHost alloc failed:.");
            }

            InitMem(twMatrixArrayHost, 0, twMatrixLen * sizeof(float));

            // 构造tw矩阵
            auto nowHTwMatrixArrayHost = twMatrixArrayHost;
            int64_t stepLen = radixListLen;
            int64_t n0 = 1;
            for (int64_t stepIndex = 0; stepIndex < stepLen - 1; stepIndex++) {
                int64_t n1 = radixListPtr[stepIndex];
                if (n1 == 0) {
                    continue;
                }
                int64_t n2 = fftN / n1 / n0;

                int32_t tileM0;
                int32_t tileK0;
                int32_t tileN0;

                getTile(n1, (n2 > 1) ? n2 : n0, stepIndex, stepLen, tileM0, tileN0, tileK0);

                for (int64_t k = 0; k < n1 * n2; k++) {
                    int64_t i = k / n2;
                    int64_t j = k % n2;
                    nowHTwMatrixArrayHost[i * 2 * tileN0 + (j / tileN0) * 2 * n1 * tileN0 + (j % tileN0)] = cos(
                        -1.0 * K_2PI / (n1 * n2) * i * j);
                    nowHTwMatrixArrayHost[i * 2 * tileN0 + tileN0 + (j / tileN0) * 2 * n1 * tileN0 +
                                          (j % tileN0)] = sin(-1.0 * K_2PI / (n1 * n2) * i * j);
                }
                nowHTwMatrixArrayHost += 2 * n1 * ROUND(n2, tileN0);
                n0 *= n1;
            }
        }

        twMatrixArray_.desc = {
            Mki::TensorDType::TENSOR_DTYPE_FLOAT, Mki::TensorFormat::TENSOR_FORMAT_ND, {twMatrixLen}, {}, 0};
        twMatrixArray_.hostData = twMatrixArrayHost;
        twMatrixArray_.dataSize = twMatrixLen * sizeof(float);

        return twMatrixArrayPtr;
    };

    AsdSip::CoeffKey key = {coreType, MIX_TW_MARK, {fftN}, forward};
    twMatrixArray = FFTensorCache::getCoeff(key, func);

    return AsdSip::ErrorType::ACL_SUCCESS;
}

// to estimate workspace
void InitMixRadixParam(int64_t parity, int64_t fftN, int64_t batchSize, std::vector<int64_t>& radixVecRef,
                       int64_t& workspaceInputSize, int64_t& workspaceOutputSize, int64_t& workspaceSyncSize,
                       int64_t& workspaceC2cOutputSize, int64_t& workspaceAuxilSize)
{
    if (fftN == 0) {
        fftN = 1;
    }
    int64_t n = fftN;

    int64_t* radixListPtr = static_cast<int64_t*>(radixVecRef.data());
    int64_t stepLen = static_cast<int64_t>(radixVecRef.size());
    int64_t tmpBatchSize = batchSize;
    // 让每次处理的数据小于 2 ^ 21/2 ^ 22 次方
    int64_t maxFloat = parity == 0 && fftN <= EVEN_THRESHOLD ? L2_CACHE_MAX_FLOAT_22 : L2_CACHE_MAX_FLOAT_21;
    int64_t batchPartLen = (maxFloat + fftN - 1) / fftN;
    if (batchPartLen == 0) {
        batchPartLen = 1;
    }
    int64_t batchLoop = (tmpBatchSize + batchPartLen - 1) / batchPartLen;
    int64_t batchRemain = tmpBatchSize % batchPartLen;

    if (batchLoop == 1 && batchRemain > 0) {
        batchPartLen = batchRemain;
    }

    if (n > maxFloat * RADIX_LEN_FIVE_NUM) {
        batchPartLen = tmpBatchSize;
    }

    int64_t tmpWorkspaceMaxBytes = (stepLen == 1) ? 0 : batchPartLen * 2 * fftN * sizeof(float);
    int64_t n0 = 1;
    int64_t tmpWorkspaceSyncBytes = 0;
    for (int64_t stepIndex = 0; stepIndex < stepLen; stepIndex++) {
        int64_t n1 = radixListPtr[stepIndex];
        if (n1 == 0) {
            continue;
        }
        int64_t n2 = fftN / n1 / n0;
        int32_t tileM0;
        int32_t tileK0;
        int32_t tileN0;
        getTile(n1, (n2 > 1) ? n2 : n0, stepIndex, stepLen, tileM0, tileN0, tileK0);
        int64_t N2_padding = ROUND(n2, tileN0);
        int64_t batchLen = L0AB_PINGPONG_BUFFER_LEN * 2 / (tileK0 * tileN0);
        batchLen += static_cast<int64_t>(batchLen == 0);
        int64_t tmpNowWorkspaceMaxBytes = 32;
        if (stepIndex == 0 ||
            (stepLen >= RADIX_LEN_FOUR_NUM && stepIndex < stepLen - RADIX_LEN_EVEN_NUM && stepIndex != 0)) {
            tmpNowWorkspaceMaxBytes = (batchPartLen * n0 * TWO_MUL * n1 * N2_padding) * sizeof(float);
        }
        if (stepLen >= RADIX_LEN_THREE_NUM && stepIndex == stepLen - RADIX_LEN_EVEN_NUM) {
            tmpNowWorkspaceMaxBytes = batchPartLen * n0 * TWO_MUL * n1 * (tileN0) * sizeof(float);
        }
        if (stepIndex == stepLen - 1) {
            tmpNowWorkspaceMaxBytes = batchPartLen * TWO_MUL * ROUND(n1, ROUND_16) * ROUND(n0, tileN0) * sizeof(float);
        }
        tmpNowWorkspaceMaxBytes = ROUND(tmpNowWorkspaceMaxBytes, RADIX_ADDR_START);
        int64_t tmpNowWorkspaceSyncBytes = 20 * 2 * batchLen * tileM0 * tileN0 * sizeof(float);
        tmpWorkspaceMaxBytes = tmpNowWorkspaceMaxBytes > tmpWorkspaceMaxBytes ? tmpNowWorkspaceMaxBytes :
                                                                                tmpWorkspaceMaxBytes;
        tmpWorkspaceSyncBytes = tmpNowWorkspaceSyncBytes > tmpWorkspaceSyncBytes ? tmpNowWorkspaceSyncBytes :
                                                                                   tmpWorkspaceSyncBytes;
        n0 *= n1;
    }

    workspaceInputSize = ROUND(TWO_MUL * tmpWorkspaceMaxBytes, ROUND_32);
    workspaceOutputSize = ROUND(TWO_MUL * tmpWorkspaceMaxBytes, ROUND_32);

    workspaceSyncSize = ROUND(tmpWorkspaceSyncBytes, ROUND_32);

    workspaceC2cOutputSize = ROUND(sizeof(std::complex<float>) * fftN * batchSize, ROUND_32);

    workspaceAuxilSize = AUXILSIZE;
}

void InitAiVSplitWay(FFTCoreType coreType, int64_t fftN, std::vector<int64_t>& radixVecRef, int32_t& aivSplitWay)
{
    int64_t* radixListPtr = static_cast<int64_t*>(radixVecRef.data());
    int64_t stepLen = static_cast<int64_t>(radixVecRef.size());
    int64_t n = fftN;

    bool isVecVtransposeLoad = false;
    {
        int64_t n0 = 1;
        for (int64_t stepIndex = 0; stepIndex < stepLen - 1; stepIndex++) {
            int64_t n1 = radixListPtr[stepIndex];
            if (n1 == 0) {
                continue;
            }
            int64_t n2 = n / n1 / n0;

            int32_t tileM0;
            int32_t tileK0;
            int32_t tileN0;

            getTile(n1, (n2 > 1) ? n2 : n0, stepIndex, stepLen, tileM0, tileN0, tileK0);

            if (stepIndex == 0 && stepLen >= RADIX_LEN_EVEN_NUM) {
                isVecVtransposeLoad = ROUND(n2, tileN0) <= VECVTRANS_LEN;
            }
            n0 *= n1;
        }
    }

    {
        int64_t n1 = radixListPtr[stepLen - 1];
        if (n1 == 0) {
            n1 = 1;
        }
        int64_t n0 = n / n1;

        int32_t tileM0;
        int32_t tileN0;
        int32_t tileK0;

        getTile(n1, n0, stepLen - 1, stepLen, tileM0, tileN0, tileK0);
        if (tileN0 == 0) {
            tileN0 = 1;
        }
        if (tileK0 == 0) {
            tileK0 = 1;
        }
        int64_t batchLen = L0AB_PINGPONG_BUFFER_LEN * 2 / (tileN0 * tileK0);
        batchLen += static_cast<int>(batchLen == 0);

        aivSplitWay = (batchLen > 1 && (n1 <= RADIX_LIST_LEN || (n1 <= AUXILSIZE && n0 <= MIX_TW_MARK))) ?
                          1 :
                          ((L0AB_PINGPONG_BUFFER_LEN / (tileK0 * tileN0) >= 1) ? AIVSPLITWAY_TWO : AIVSPLITWAY_THREE);

        if (coreType == FFTCoreType::kFftMix) {
            aivSplitWay += isVecVtransposeLoad ? RADIX_LEN_THREE_NUM : 0;
        }
    }
}

void GenWMatrixForwardForMultiLen(int64_t stepLen, int64_t* radixListPtr, int64_t fftN, float* nowDftMatrixArrayHost)
{
    int64_t n0Num = 1;
    // 生成正向所需的W矩阵
    for (int64_t stepIndex = 0; stepIndex < stepLen; stepIndex++) {
        int64_t n1Num = radixListPtr[stepIndex];
        if (n1Num == 0) {
            continue;
        }
        int64_t n2 = fftN / n1Num / n0Num;

        int32_t tileM0;
        int32_t tileK0;
        int32_t tileN0;

        getTile(n1Num, (n2 > 1) ? n2 : n0Num, stepIndex, stepLen, tileM0, tileN0, tileK0);

        int64_t batchLen = L0AB_PINGPONG_BUFFER_LEN * 2 / (tileK0 * tileN0);
        batchLen += static_cast<int64_t>(batchLen == 0);

        // 对应虚实结合的三种情况
        if (stepIndex == stepLen - 1 && batchLen > 1 && (n1Num <= AUXILSIZE && n0Num <= MIX_TW_MARK)) {
            for (int64_t k = 0; k < n1Num * n1Num; k++) {
                int64_t i = k / n1Num;
                int64_t j = k % n1Num;
                nowDftMatrixArrayHost[i * TWO_MUL * ROUND(n1Num, ROUND_16) + j] = cos(-1.0 * K_2PI / n1Num * i * j);
                nowDftMatrixArrayHost[i * TWO_MUL * ROUND(n1Num, ROUND_16) + ROUND(n1Num, ROUND_16) + j] = -sin(
                    -1.0 * K_2PI / n1Num * i * j);
                nowDftMatrixArrayHost[n1Num * TWO_MUL * ROUND(n1Num, ROUND_16) + i * TWO_MUL * ROUND(n1Num, ROUND_16) +
                                      j] = sin(-1.0 * K_2PI / n1Num * i * j);
                nowDftMatrixArrayHost[n1Num * TWO_MUL * ROUND(n1Num, ROUND_16) + i * TWO_MUL * ROUND(n1Num, ROUND_16) +
                                      ROUND(n1Num, ROUND_16) + j] = cos(-1.0 * K_2PI / n1Num * i * j);
            }
        } else if (stepIndex == stepLen - 1 && L0AB_PINGPONG_BUFFER_LEN / (tileK0 * tileN0) >= 1) {
            // 将2 * n1Num 按照tile_M0分块，每个tile_M0块，上面是实部，下面是虚部。
            int64_t n1Loop = (2 * n1Num + tileM0 - 1) / tileM0;
            for (int64_t k = 0; k < n1Num * n1Num; k++) {
                int64_t i = k / n1Num;
                int64_t j = k % n1Num;
                int64_t n1Index = i / (tileM0 / 2);
                int64_t n1InIndex = i % (tileM0 / 2);
                int64_t n1Actual = (n1Index == n1Loop - 1) ? (2 * n1Num - n1Index * tileM0) : tileM0;
                nowDftMatrixArrayHost[n1Index * tileM0 * TWO_MUL * ROUND(n1Num, ROUND_16) +
                                      n1InIndex * TWO_MUL * ROUND(n1Num, ROUND_16) + j] = cos(-1.0 * K_2PI / n1Num * i *
                                                                                              j);
                nowDftMatrixArrayHost[n1Index * tileM0 * TWO_MUL * ROUND(n1Num, ROUND_16) +
                                      n1InIndex * TWO_MUL * ROUND(n1Num, ROUND_16) + ROUND(n1Num, ROUND_16) +
                                      j] = -sin(-1.0 * K_2PI / n1Num * i * j);
                nowDftMatrixArrayHost[n1Index * tileM0 * TWO_MUL * ROUND(n1Num, ROUND_16) +
                                      n1InIndex * TWO_MUL * ROUND(n1Num, ROUND_16) +
                                      (n1Actual / TWO_MUL) * TWO_MUL * ROUND(n1Num, ROUND_16) + j] = sin(-1.0 * K_2PI /
                                                                                                         n1Num * i * j);
                nowDftMatrixArrayHost[n1Index * tileM0 * TWO_MUL * ROUND(n1Num, ROUND_16) +
                                      n1InIndex * TWO_MUL * ROUND(n1Num, ROUND_16) +
                                      (n1Actual / TWO_MUL) * TWO_MUL * ROUND(n1Num, ROUND_16) + ROUND(n1Num, ROUND_16) +
                                      j] = cos(-1.0 * K_2PI / n1Num * i * j);
            }
        } else if (stepIndex == stepLen - 1) {
            for (int64_t k = 0; k < n1Num * n1Num; k++) {
                int64_t i = k / n1Num;
                int64_t j = k % n1Num;
                nowDftMatrixArrayHost[i * TWO_MUL * TWO_MUL * ROUND(n1Num, ROUND_16) + j] = cos(-1.0 * K_2PI / n1Num *
                                                                                                i * j);
                nowDftMatrixArrayHost[i * TWO_MUL * TWO_MUL * ROUND(n1Num, ROUND_16) + ROUND(n1Num, ROUND_16) +
                                      j] = -sin(-1.0 * K_2PI / n1Num * i * j);
                nowDftMatrixArrayHost[i * TWO_MUL * TWO_MUL * ROUND(n1Num, ROUND_16) +
                                      TWO_MUL * ROUND(n1Num, ROUND_16) + j] = sin(-1.0 * K_2PI / n1Num * i * j);
                nowDftMatrixArrayHost[i * TWO_MUL * TWO_MUL * ROUND(n1Num, ROUND_16) +
                                      TWO_MUL * ROUND(n1Num, ROUND_16) + ROUND(n1Num, ROUND_16) + j] = cos(-1.0 *
                                                                                                           K_2PI /
                                                                                                           n1Num * i *
                                                                                                           j);
            }
        } else {
            for (int64_t k = 0; k < n1Num * n1Num; k++) {
                int64_t i = k / n1Num;
                int64_t j = k % n1Num;
                nowDftMatrixArrayHost[i * TWO_MUL * ROUND(TWO_MUL * n1Num, tileM0) + j] = cos(-1.0 * K_2PI / n1Num * i *
                                                                                              j);
                nowDftMatrixArrayHost[i * TWO_MUL * ROUND(TWO_MUL * n1Num, tileM0) + n1Num + j] = -sin(-1.0 * K_2PI /
                                                                                                       n1Num * i * j);
                nowDftMatrixArrayHost[i * TWO_MUL * ROUND(TWO_MUL * n1Num, tileM0) + ROUND(TWO_MUL * n1Num, tileM0) +
                                      j] = sin(-1.0 * K_2PI / n1Num * i * j);
                nowDftMatrixArrayHost[i * TWO_MUL * ROUND(TWO_MUL * n1Num, tileM0) + ROUND(TWO_MUL * n1Num, tileM0) +
                                      n1Num + j] = cos(-1.0 * K_2PI / n1Num * i * j);
            }
            nowDftMatrixArrayHost += ROUND(TWO_MUL * n1Num, tileM0) * ROUND(TWO_MUL * n1Num, tileK0);
            n0Num *= n1Num;
        }
    }
}

void GenWMatrixForwardForOneLen(int64_t stepLen, int64_t* radixListPtr, int64_t fftN, float* nowDftMatrixArrayHost)
{
    int64_t n0 = 1;
    int64_t n = fftN;

    // 生成正向所需的W矩阵
    for (int64_t stepIndex = 0; stepIndex < stepLen; stepIndex++) {
        int64_t n1Radix = radixListPtr[stepIndex];
        if (n1Radix == 0) {
            continue;
        }

        int32_t tileM0;
        int32_t tileK0;

        if (n <= RADIX_LIST_LEN) {
            tileM0 = ROUND(TWO_MUL * n, ROUND_16);
            tileK0 = tileM0;
        } else {
            tileM0 = RADIX_ADDR_START;
            tileK0 = RADIX_ADDR_START;
            if (tileK0 > TWO_MUL * n / TWO_MUL && tileK0 < TWO_MUL * n) {
                tileK0 = MIN(tileK0, ROUND(TWO_MUL * n / TWO_MUL, ROUND_16));
            }
            if (tileM0 > TWO_MUL * n / TWO_MUL && tileM0 < TWO_MUL * n) {
                tileM0 = MIN(tileM0, ROUND(TWO_MUL * n / TWO_MUL, ROUND_16));
            }
            tileM0 = MIN(tileM0, ROUND(TWO_MUL * n, ROUND_16));
            tileK0 = MIN(tileK0, ROUND(TWO_MUL * n, ROUND_16));
        }

        for (int64_t k = 0; k < n1Radix * n1Radix; k++) {
            int64_t i = k / n1Radix;
            int64_t j = k % n1Radix;
            nowDftMatrixArrayHost[i * TWO_MUL * ROUND(TWO_MUL * n1Radix, tileM0) + TWO_MUL * j] = cos(-1.0 * K_2PI /
                                                                                                      n1Radix * i * j);
            nowDftMatrixArrayHost[i * TWO_MUL * ROUND(TWO_MUL * n1Radix, tileM0) + TWO_MUL * j + 1] = -sin(
                -1.0 * K_2PI / n1Radix * i * j);
            nowDftMatrixArrayHost[i * TWO_MUL * ROUND(TWO_MUL * n1Radix, tileM0) + ROUND(TWO_MUL * n1Radix, tileM0) +
                                  TWO_MUL * j] = sin(-1.0 * K_2PI / n1Radix * i * j);
            nowDftMatrixArrayHost[i * TWO_MUL * ROUND(TWO_MUL * n1Radix, tileM0) + ROUND(TWO_MUL * n1Radix, tileM0) +
                                  TWO_MUL * j + 1] = cos(-1.0 * K_2PI / n1Radix * i * j);
        }
        nowDftMatrixArrayHost += ROUND(TWO_MUL * n1Radix, tileM0) * ROUND(TWO_MUL * n1Radix, tileK0);
        n0 *= n1Radix;
    }
}

void GenWMatrixInverseForMultiLen(int64_t stepLen, int64_t* radixListPtr, int64_t fftN,
                                  float* nowDftMatrixArrayInverseHost)
{
    int64_t n0 = 1;

    for (int64_t stepIndex = 0; stepIndex < stepLen; stepIndex++) {
        int64_t n1 = radixListPtr[stepIndex];
        if (n1 == 0) {
            continue;
        }
        int64_t n2 = fftN / n1 / n0;

        int32_t tileM0;
        int32_t tileK0;
        int32_t tileN0;
        getTile(n1, (n2 > 1) ? n2 : n0, stepIndex, stepLen, tileM0, tileN0, tileK0);

        if (tileK0 == 0) {
            tileK0 = 1;
        }
        if (tileN0 == 0) {
            tileN0 = 1;
        }
        int64_t batchLen = L0AB_PINGPONG_BUFFER_LEN * 2 / (tileK0 * tileN0);
        batchLen += static_cast<int64_t>(batchLen == 0);

        // 对应虚实结合的三种情况
        if (stepIndex == stepLen - 1 && batchLen > 1 && (n1 <= AUXILSIZE && n0 <= MIX_TW_MARK)) {
            for (int64_t k = 0; k < n1 * n1; k++) {
                int64_t i = k / n1;
                int64_t j = k % n1;
                nowDftMatrixArrayInverseHost[i * TWO_MUL * ROUND(n1, ROUND_16) + j] = cos(-1.0 * K_2PI / n1 * i * j);
                nowDftMatrixArrayInverseHost[i * TWO_MUL * ROUND(n1, ROUND_16) + ROUND(n1, ROUND_16) + j] = sin(
                    -1.0 * K_2PI / n1 * i * j);
                nowDftMatrixArrayInverseHost[n1 * TWO_MUL * ROUND(n1, ROUND_16) + i * TWO_MUL * ROUND(n1, ROUND_16) +
                                             j] = -sin(-1.0 * K_2PI / n1 * i * j);
                nowDftMatrixArrayInverseHost[n1 * TWO_MUL * ROUND(n1, ROUND_16) + i * TWO_MUL * ROUND(n1, ROUND_16) +
                                             ROUND(n1, ROUND_16) + j] = cos(-1.0 * K_2PI / n1 * i * j);
            }
        } else if (stepIndex == stepLen - 1 && L0AB_PINGPONG_BUFFER_LEN / (tileK0 * tileN0) >= 1) {
            // 将2 * n1 按照tile_M0分块，每个tile_M0块，上面是实部，下面是虚部。
            int64_t n1Loop = (2 * n1 + tileM0 - 1) / tileM0;
            for (int64_t k = 0; k < n1 * n1; k++) {
                int64_t i = k / n1;
                int64_t j = k % n1;
                int64_t n1Index = i / (tileM0 / 2);
                int64_t n1InIndex = i % (tileM0 / 2);
                int64_t n1Actual = (n1Index == n1Loop - 1) ? (2 * n1 - n1Index * tileM0) : tileM0;
                nowDftMatrixArrayInverseHost[n1Index * tileM0 * TWO_MUL * ROUND(n1, ROUND_16) +
                                             n1InIndex * TWO_MUL * ROUND(n1, ROUND_16) + j] = cos(-1.0 * K_2PI / n1 *
                                                                                                  i * j);
                nowDftMatrixArrayInverseHost[n1Index * tileM0 * TWO_MUL * ROUND(n1, ROUND_16) +
                                             n1InIndex * TWO_MUL * ROUND(n1, ROUND_16) + ROUND(n1, ROUND_16) +
                                             j] = sin(-1.0 * K_2PI / n1 * i * j);
                nowDftMatrixArrayInverseHost[n1Index * tileM0 * TWO_MUL * ROUND(n1, ROUND_16) +
                                             n1InIndex * TWO_MUL * ROUND(n1, ROUND_16) +
                                             (n1Actual / TWO_MUL) * TWO_MUL * ROUND(n1, ROUND_16) + j] = -sin(-1.0 *
                                                                                                              K_2PI /
                                                                                                              n1 * i *
                                                                                                              j);
                nowDftMatrixArrayInverseHost[n1Index * tileM0 * TWO_MUL * ROUND(n1, ROUND_16) +
                                             n1InIndex * TWO_MUL * ROUND(n1, ROUND_16) +
                                             (n1Actual / TWO_MUL) * TWO_MUL * ROUND(n1, ROUND_16) +
                                             ROUND(n1, ROUND_16) + j] = cos(-1.0 * K_2PI / n1 * i * j);
            }
        } else if (stepIndex == stepLen - 1) {
            for (int64_t k = 0; k < n1 * n1; k++) {
                int64_t i = k / n1;
                int64_t j = k % n1;
                nowDftMatrixArrayInverseHost[i * TWO_MUL * TWO_MUL * ROUND(n1, ROUND_16) + j] = cos(-1.0 * K_2PI / n1 *
                                                                                                    i * j);
                nowDftMatrixArrayInverseHost[i * TWO_MUL * TWO_MUL * ROUND(n1, ROUND_16) + ROUND(n1, ROUND_16) +
                                             j] = sin(-1.0 * K_2PI / n1 * i * j);
                nowDftMatrixArrayInverseHost[i * TWO_MUL * TWO_MUL * ROUND(n1, ROUND_16) +
                                             TWO_MUL * ROUND(n1, ROUND_16) + j] = -sin(-1.0 * K_2PI / n1 * i * j);
                nowDftMatrixArrayInverseHost[i * TWO_MUL * TWO_MUL * ROUND(n1, ROUND_16) +
                                             TWO_MUL * ROUND(n1, ROUND_16) + ROUND(n1, ROUND_16) + j] = cos(-1.0 *
                                                                                                            K_2PI / n1 *
                                                                                                            i * j);
            }
        } else {
            for (int64_t k = 0; k < n1 * n1; k++) {
                int64_t i = k / n1;
                int64_t j = k % n1;
                nowDftMatrixArrayInverseHost[i * TWO_MUL * ROUND(TWO_MUL * n1, tileM0) + j] = cos(-1.0 * K_2PI / n1 *
                                                                                                  i * j);
                nowDftMatrixArrayInverseHost[i * TWO_MUL * ROUND(TWO_MUL * n1, tileM0) + n1 + j] = sin(-1.0 * K_2PI /
                                                                                                       n1 * i * j);
                nowDftMatrixArrayInverseHost[i * TWO_MUL * ROUND(TWO_MUL * n1, tileM0) + ROUND(TWO_MUL * n1, tileM0) +
                                             j] = -sin(-1.0 * K_2PI / n1 * i * j);
                nowDftMatrixArrayInverseHost[i * TWO_MUL * ROUND(TWO_MUL * n1, tileM0) + ROUND(TWO_MUL * n1, tileM0) +
                                             n1 + j] = cos(-1.0 * K_2PI / n1 * i * j);
            }
        }
        nowDftMatrixArrayInverseHost += ROUND(TWO_MUL * n1, tileM0) * ROUND(TWO_MUL * n1, tileK0);
        n0 *= n1;
    }
}

void GenWMatrixInverseForOneLen(int64_t stepLen, int64_t* radixListPtr, int64_t fftN,
                                float* nowDftMatrixArrayInverseHost)
{
    int64_t n0 = 1;
    int64_t n = fftN;

    for (int64_t stepIndex = 0; stepIndex < stepLen; stepIndex++) {
        int64_t n1 = radixListPtr[stepIndex];
        if (n1 == 0) {
            continue;
        }

        int32_t tileM0;
        int32_t tileK0;

        if (n <= RADIX_LIST_LEN) {
            tileM0 = ROUND(TWO_MUL * n, ROUND_16);
            tileK0 = tileM0;
        } else {
            tileK0 = RADIX_ADDR_START;
            tileM0 = RADIX_ADDR_START;
            if (tileK0 > TWO_MUL * n / TWO_MUL && tileK0 < TWO_MUL * n) {
                tileK0 = MIN(tileK0, ROUND(TWO_MUL * n / TWO_MUL, ROUND_16));
            }
            if (tileM0 > TWO_MUL * n / TWO_MUL && tileM0 < TWO_MUL * n) {
                tileM0 = MIN(tileM0, ROUND(TWO_MUL * n / TWO_MUL, ROUND_16));
            }
            tileM0 = MIN(tileM0, ROUND(TWO_MUL * n, ROUND_16));
            tileK0 = MIN(tileK0, ROUND(TWO_MUL * n, ROUND_16));
        }

        for (int64_t k = 0; k < n1 * n1; k++) {
            int64_t i = k / n1;
            int64_t j = k % n1;
            nowDftMatrixArrayInverseHost[i * TWO_MUL * ROUND(TWO_MUL * n1, tileM0) + TWO_MUL * j] = cos(-1.0 * K_2PI /
                                                                                                        n1 * i * j);
            nowDftMatrixArrayInverseHost[i * TWO_MUL * ROUND(TWO_MUL * n1, tileM0) + TWO_MUL * j + 1] = sin(
                -1.0 * K_2PI / n1 * i * j);
            nowDftMatrixArrayInverseHost[i * TWO_MUL * ROUND(TWO_MUL * n1, tileM0) + ROUND(TWO_MUL * n1, tileM0) +
                                         TWO_MUL * j] = -sin(-1.0 * K_2PI / n1 * i * j);
            nowDftMatrixArrayInverseHost[i * TWO_MUL * ROUND(TWO_MUL * n1, tileM0) + ROUND(TWO_MUL * n1, tileM0) +
                                         TWO_MUL * j + 1] = cos(-1.0 * K_2PI / n1 * i * j);
        }
        nowDftMatrixArrayInverseHost += ROUND(TWO_MUL * n1, tileM0) * ROUND(TWO_MUL * n1, tileK0);
        n0 *= n1;
    }
}

// PQMatrix
AspbStatus InitPQMatrix(FFTCoreType coreType, int64_t fftN, bool forward, bool isP,
                        std::shared_ptr<AsdSip::FFTensor>& pqMatrix)
{
    int64_t inSize = 2 * fftN;
    int64_t outSize = 2 * fftN;
    int32_t tableSize = fftN;

    std::unique_ptr<double[]> cosTable{new double[tableSize]};
    std::unique_ptr<double[]> sinTable{new double[tableSize]};

    InitMem(cosTable.get(), 0, tableSize * sizeof(double));
    InitMem(sinTable.get(), 0, tableSize * sizeof(double));

    for (int i = 0; i < fftN; i++) {
        cosTable[i] = cos(K_2PI * i / fftN);
        sinTable[i] = sin(K_2PI * i / fftN);
    }

    std::function<AsdSip::FFTensor*()> func = [=, &cosTable, &sinTable]() -> AsdSip::FFTensor* {
        AsdSip::FFTensor* _pqMatrix_ptr = new AsdSip::FFTensor;
        AsdSip::FFTensor& _pqMatrix = *_pqMatrix_ptr;

        float* _pqMatrix_host = nullptr;
        try {
            _pqMatrix_host = new float[outSize * inSize];
        } catch (std::bad_alloc& e) {
            delete _pqMatrix_ptr;
            ASDSIP_LOG(ERROR) << "_pqMatrix_host alloc failed: " << e.what();
            throw std::runtime_error("_pqMatrix_host alloc failed:.");
        }
        InitMem(_pqMatrix_host, 0, sizeof(float) * outSize * inSize);

        for (int i = 0; i < fftN; i++) {
            for (int j = 0; j < fftN; j++) {
                if (isP) {
                    *(reinterpret_cast<float*>(_pqMatrix_host) + (2 * i) * (2 * fftN) + j) = cosTable[(i * j) % fftN];
                    *(reinterpret_cast<float*>(_pqMatrix_host) + (2 * i + 1) * (2 * fftN) +
                      j) = (forward ? -1 : 1) * sinTable[(i * j) % fftN];
                    *(reinterpret_cast<float*>(_pqMatrix_host) + (2 * i) * (2 * fftN) + j +
                      fftN) = (forward ? 1 : -1) * sinTable[(i * j) % fftN];
                    *(reinterpret_cast<float*>(_pqMatrix_host) + (2 * i + 1) * (2 * fftN) + j +
                      fftN) = cosTable[(i * j) % fftN];
                    continue;
                }

                *(reinterpret_cast<float*>(_pqMatrix_host) + j * (2 * fftN) + (2 * i)) = cosTable[(i * j) % fftN];
                *(reinterpret_cast<float*>(_pqMatrix_host) + j * (2 * fftN) + (2 * i + 1)) = (forward ? -1 : 1) *
                                                                                             sinTable[(i * j) % fftN];
                *(reinterpret_cast<float*>(_pqMatrix_host) + (j + fftN) * (2 * fftN) +
                  (2 * i)) = (forward ? 1 : -1) * sinTable[(i * j) % fftN];
                *(reinterpret_cast<float*>(_pqMatrix_host) + (j + fftN) * (2 * fftN) +
                  (2 * i + 1)) = cosTable[(i * j) % fftN];
            }
        }

        _pqMatrix.desc = {
            Mki::TensorDType::TENSOR_DTYPE_FLOAT, Mki::TensorFormat::TENSOR_FORMAT_ND, {outSize, inSize}, {}, 0};
        _pqMatrix.hostData = _pqMatrix_host;
        _pqMatrix.dataSize = sizeof(float) * outSize * inSize;

        return _pqMatrix_ptr;
    };

    AsdSip::CoeffKey key = {coreType, static_cast<unsigned>(isP ? DD_P_MARK : DD_Q_MARK), {fftN}, forward};
    pqMatrix = FFTensorCache::getCoeff(key, func);

    return AsdSip::ErrorType::ACL_SUCCESS;
}
