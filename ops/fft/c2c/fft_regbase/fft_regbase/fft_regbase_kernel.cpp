/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <mki_loader/op_register.h>
#include <mki/utils/platform/platform_info.h>
#include <limits>
#include "params/fft_c2c_regbase.h"
#include "tiling/fft_regbase_tiling.h"
#include "tiling/fft_regbase_tiling_data.h"
#include "utils/assert.h"

namespace AsdSip {
using namespace Mki;

class FftC2CRegBaseC64Kernel : public KernelBase {
public:
    explicit FftC2CRegBaseC64Kernel(const std::string& kernelName, const BinHandle* handle) noexcept
        : KernelBase(kernelName, handle)
    {}

    bool CanSupport(const LaunchParam& launchParam) const override
    {
        ASDSIP_CHECK(PlatformInfo::Instance().GetPlatformType() == PlatformType::ASCEND_950,
                     "RegBase requires Ascend 950", return false);
        ASDSIP_CHECK(launchParam.GetParam().Type() == typeid(OpParam::FftC2CRegBase),
                     "RegBase parameter type is invalid", return false);
        ASDSIP_CHECK(launchParam.GetInTensorCount() == 2 && launchParam.GetOutTensorCount() == 1,
                     "RegBase requires two inputs and one output", return false);
        const auto& param = AnyCast<OpParam::FftC2CRegBase>(launchParam.GetParam());
        ASDSIP_CHECK(OpParam::IsFftC2CRegBaseSupported(param.fftN) && param.batchSize > 0 &&
                         (param.isInverse == 0 || param.isInverse == 1),
                     "RegBase parameters are unsupported", return false);
        ASDSIP_CHECK(param.batchSize <= std::numeric_limits<int64_t>::max() / (param.fftN * 8),
                     "RegBase signal byte count overflows", return false);
        const auto& input = launchParam.GetInTensor(0);
        const auto& coeff = launchParam.GetInTensor(1);
        const auto& output = launchParam.GetOutTensor(0);
        ASDSIP_CHECK(input.desc.dtype == TENSOR_DTYPE_COMPLEX64 && output.desc.dtype == TENSOR_DTYPE_COMPLEX64,
                     "RegBase signal dtype must be complex64", return false);
        ASDSIP_CHECK(input.desc.dims.size() == 2 && input.desc.dims[0] == param.batchSize &&
                         input.desc.dims[1] == param.fftN && output.desc.dims == input.desc.dims,
                     "RegBase requires batch-major [batch, N] signal tensors", return false);
        ASDSIP_CHECK(coeff.desc.dtype == TENSOR_DTYPE_FLOAT && coeff.desc.dims.size() == 1 &&
                         coeff.desc.dims[0] == param.fftN * 2,
                     "RegBase requires 2*N FP32 twiddle values", return false);
        ASDSIP_CHECK(input.IsContiguous() && coeff.IsContiguous() && output.IsContiguous() && input.desc.offset == 0 &&
                         coeff.desc.offset == 0 && output.desc.offset == 0,
                     "RegBase requires contiguous tensors with zero descriptor offset", return false);
        ASDSIP_CHECK(input.data == nullptr || output.data == nullptr || input.data != output.data,
                     "RegBase requires separate input/output; planner must supply scratch for aliasing", return false);
        return true;
    }

    uint64_t GetTilingSize(const LaunchParam&) const override { return sizeof(FftC2CRegBaseTilingData); }

    Status InitImpl(const LaunchParam& launchParam) override
    {
        ASDSIP_CHECK(FftC2CRegBaseTiling(launchParam, kernelInfo_) == AsdSip::ErrorType::ACL_SUCCESS,
                     "RegBase tiling failed", return Status::FailStatus(ERROR_INVALID_VALUE));
        return Status::OkStatus();
    }
};

REG_KERNEL_BASE(FftC2CRegBaseC64Kernel);
} // namespace AsdSip
