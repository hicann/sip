/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <mki/base/operation_base.h>
#include <mki_loader/op_register.h>
#include <mki/utils/SVector/SVector.h>
#include "utils/assert.h"
#include "log/log.h"
#include "dft_c2r.h"

namespace AsdSip {
using namespace Mki;

constexpr int64_t INPUT_NUM = 2;

class DftC2ROperation : public OperationBase {
public:
    explicit DftC2ROperation(const std::string& opName) noexcept : OperationBase(opName) {}

    Kernel* GetBestKernel(const LaunchParam& launchParam) const override
    {
        ASDSIP_CHECK(IsConsistent(launchParam), "Failed to check consistent", return nullptr);
        return GetKernelByName("DftC2RC64Kernel");
    }

    int64_t GetInputNum(const Any& specificParam) const override
    {
        ASDSIP_CHECK(specificParam.Type() == typeid(OpParam::DftC2R), "OpParam is invalid", return 0);
        return INPUT_NUM;
    }

protected:
    Status InferShapeImpl(const LaunchParam& launchParam, SVector<Tensor>& outTensors) const override
    {
        ASDSIP_CHECK(launchParam.GetParam().Type() == typeid(OpParam::DftC2R), "OpParam is invalid",
                     return Status::FailStatus(ERROR_INFERSHAPE_ERROR, "no match specificParam type."));
        ASDSIP_CHECK(launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_COMPLEX64, "error opId",
                     return Status::FailStatus(ERROR_INFERSHAPE_ERROR));

        const auto& param = AnyCast<OpParam::DftC2R>(launchParam.GetParam());

        outTensors[0].desc.dtype = TENSOR_DTYPE_FLOAT;
        outTensors[0].desc.format = TensorFormat::TENSOR_FORMAT_ND;
        // 空 dims 防护（issue #397）：0 维输入时 dims[0] 为越界读，先拒绝
        const auto& inDims = launchParam.GetInTensor(0).desc.dims;
        ASDSIP_CHECK(!inDims.empty(), "input dims is empty",
                     return Status::FailStatus(ERROR_INFERSHAPE_ERROR, "input dims is empty."));
        int64_t batch = inDims[0];
        outTensors[0].desc.dims = {batch, param.fftN};

        return Status::OkStatus();
    }
};
REG_OPERATION(DftC2ROperation);
} //    namespace AsdSip
