/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <mki_loader/op_register.h>
#include <mki/utils/SVector/SVector.h>
#include <mki/utils/platform/platform_info.h>
#include <mki/base/operation_base.h>
#include "utils/assert.h"
#include "log/log.h"
#include "fft_all_mix.h"
#include "fft_r2c_arch35.h"

namespace {
constexpr int EVEN_THRESHOLD = 524288 / 2;
}

namespace AsdSip {
using namespace Mki;
constexpr int AIVWAYC2R2C = 2;
constexpr int AIVWAYC2R3C = 3;
constexpr int64_t INPUT_NUM = 8;
class FftR2COperation : public OperationBase {
public:
    explicit FftR2COperation(const std::string& opName) noexcept : OperationBase(opName) {}

    Kernel* GetBestKernel(const LaunchParam& launchParam) const override
    {
        if (Mki::PlatformInfo::Instance().GetPlatformType() == Mki::PlatformType::ASCEND_950) {
            return GetKernelByName("FftR2CC64Kernel");
        } else if (Mki::PlatformInfo::Instance().GetPlatformType() == Mki::PlatformType::ASCEND_910B) {
            ASDSIP_CHECK(IsConsistent(launchParam), "Failed to check consistent", return nullptr);
            const auto& param = AnyCast<OpParam::FftAllMix>(launchParam.GetParam());

            if (param.aiv_split_way == 1 && param.parity == 1) {
                return GetKernelByName("FftR2COdd1C64Kernel");
            } else if (param.aiv_split_way == 1 && param.parity == 0) {
                return param.fftN <= EVEN_THRESHOLD ? GetKernelByName("FftR2CEven1C64Kernel") :
                                                      GetKernelByName("FftR2C1C64Kernel");
            } else if (param.aiv_split_way == AIVWAYC2R2C && param.parity == 1) {
                return GetKernelByName("FftR2COdd2C64Kernel");
            } else if (param.aiv_split_way == AIVWAYC2R2C && param.parity == 0) {
                return param.fftN <= EVEN_THRESHOLD ? GetKernelByName("FftR2CEven2C64Kernel") :
                                                      GetKernelByName("FftR2C2C64Kernel");
            } else if (param.aiv_split_way == AIVWAYC2R3C && param.parity == 1) {
                return GetKernelByName("FftR2COdd3C64Kernel");
            } else if (param.aiv_split_way == AIVWAYC2R3C && param.parity == 0) {
                return param.fftN <= EVEN_THRESHOLD ? GetKernelByName("FftR2CEven3C64Kernel") :
                                                      GetKernelByName("FftR2C3C64Kernel");
            }
        }
        return nullptr;
    }

protected:
    Status InferShapeImpl(const LaunchParam& launchParam, SVector<Tensor>& outTensors) const override
    {
        if (Mki::PlatformInfo::Instance().GetPlatformType() == Mki::PlatformType::ASCEND_950) {
            ASDSIP_CHECK(launchParam.GetParam().Type() == typeid(OpParam::FftR2CArch35), "OpParam is invalid",
                         return Status::FailStatus(ERROR_INFERSHAPE_ERROR, "no match specificParam type."));

            outTensors[0].desc.dtype = TENSOR_DTYPE_COMPLEX64;
            outTensors[0].desc.format = launchParam.GetInTensor(0).desc.format;
            // 空 dims 防护（issue #397）：0 维输入时 dims[0] 为越界读，先拒绝
            const auto& inDims = launchParam.GetInTensor(0).desc.dims;
            ASDSIP_CHECK(!inDims.empty(), "input dims is empty",
                         return Status::FailStatus(ERROR_INFERSHAPE_ERROR, "input dims is empty."));
            int64_t batch = inDims[0];
            outTensors[0].desc.dims = {batch, *(inDims.end() - 1) / 2 + 1};

        } else if (Mki::PlatformInfo::Instance().GetPlatformType() == Mki::PlatformType::ASCEND_910B) {
            ASDSIP_CHECK(launchParam.GetParam().Type() == typeid(OpParam::FftAllMix), "OpParam is invalid",
                         return Status::FailStatus(ERROR_INFERSHAPE_ERROR, "no match specificParam type."));

            outTensors[0].desc.dtype = TENSOR_DTYPE_COMPLEX64;
            outTensors[0].desc.format = launchParam.GetInTensor(0).desc.format;
            // 空 dims 防护（issue #397）：0 维输入时 dims[0] 为越界读，先拒绝
            const auto& inDims = launchParam.GetInTensor(0).desc.dims;
            ASDSIP_CHECK(!inDims.empty(), "input dims is empty",
                         return Status::FailStatus(ERROR_INFERSHAPE_ERROR, "input dims is empty."));
            int64_t batch = inDims[0];
            outTensors[0].desc.dims = {batch, *(inDims.end() - 1) / 2 + 1};
        }

        ASDSIP_CHECK(launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_FLOAT, "error dataDype",
                     return Status::FailStatus(ERROR_INFERSHAPE_ERROR));

        return Status::OkStatus();
    }

    int64_t GetInputNum(const Any& specificParam) const override
    {
        if (Mki::PlatformInfo::Instance().GetPlatformType() == Mki::PlatformType::ASCEND_950) {
            ASDSIP_CHECK(specificParam.Type() == typeid(OpParam::FftR2CArch35), "OpParam is invalid", return 0);
            return 5;
        } else if (Mki::PlatformInfo::Instance().GetPlatformType() == Mki::PlatformType::ASCEND_910B) {
            ASDSIP_CHECK(specificParam.Type() == typeid(OpParam::FftAllMix), "OpParam is invalid", return 0);
            return INPUT_NUM;
        }
        return INPUT_NUM;
    }
};
REG_OPERATION(FftR2COperation);
} //    namespace AsdSip
