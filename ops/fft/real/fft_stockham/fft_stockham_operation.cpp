/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <mki/base/operation_base.h>
#include <mki/utils/platform/platform_info.h>
#include <mki_loader/op_register.h>
#include "params/fft_real_stockham.h"
#include "params/fft_real_stockham_tiling.h"
#include "utils/assert.h"
#include <algorithm>

namespace AsdSip {
using namespace Mki;
using OpParam::FftRealStockham;
using OpParam::RealStockhamPlan;

class FftRealStockhamOperation : public OperationBase {
public:
    explicit FftRealStockhamOperation(const std::string& name) noexcept : OperationBase(name) {}
    int64_t GetInputNum(const Any& param) const override { return param.Type() == typeid(FftRealStockham) ? 5 : 0; }
    Kernel* GetBestKernel(const LaunchParam& launch) const override
    {
        if (!IsConsistent(launch) || launch.GetParam().Type() != typeid(FftRealStockham))
            return nullptr;
        return GetKernelByName(OpParam::StockhamKernelName(AnyCast<FftRealStockham>(launch.GetParam()).isC2R));
    }

protected:
    Status InferShapeImpl(const LaunchParam& launch, SVector<Tensor>& outputs) const override
    {
        ASDSIP_CHECK(launch.GetParam().Type() == typeid(FftRealStockham) && launch.GetInTensorCount() == 5 &&
                         outputs.size() == 1,
                     "Real Stockham expects five inputs and one output",
                     return Status::FailStatus(ERROR_INFERSHAPE_ERROR));
        const auto& p = AnyCast<FftRealStockham>(launch.GetParam());
        try {
            (void)RealStockhamPlan(p);
        } catch (const std::exception&) {
            return Status::FailStatus(ERROR_INFERSHAPE_ERROR);
        }
        outputs[0].desc = {OpParam::StockhamComplexSignal(p, true) ? TENSOR_DTYPE_COMPLEX64 : TENSOR_DTYPE_FLOAT,
                           TENSOR_FORMAT_ND,
                           {p.batchSize, OpParam::StockhamSignalElements(p, true)},
                           {},
                           0};
        return Status::OkStatus();
    }
};

class FftRealStockhamKernel : public KernelBase {
public:
    FftRealStockhamKernel(const std::string& name, const BinHandle* handle, bool c2r)
        : KernelBase(name, handle), c2r_(c2r)
    {}
    bool CanSupport(const LaunchParam& launch) const override
    {
        if (PlatformInfo::Instance().GetPlatformType() != PlatformType::ASCEND_950 ||
            launch.GetParam().Type() != typeid(FftRealStockham) || launch.GetInTensorCount() != 5 ||
            launch.GetOutTensorCount() != 1)
            return false;
        const auto& p = AnyCast<FftRealStockham>(launch.GetParam());
        if ((p.isC2R != 0) != c2r_)
            return false;
        try {
            const RealStockhamPlan d(p);
            const size_t coefficientSizes[] = {0, d.dftFloats, d.twiddleFloats, d.realFloats, d.radices.size()};
            for (int i = 0; i < 6; ++i) {
                const auto& tensor = i == 5 ? launch.GetOutTensor(0) : launch.GetInTensor(i);
                if (tensor.desc.format != TENSOR_FORMAT_ND || tensor.desc.offset != 0 || !tensor.IsContiguous())
                    return false;
                if (i == 0 || i == 5) {
                    const auto dtype = OpParam::StockhamComplexSignal(p, i == 5) ? TENSOR_DTYPE_COMPLEX64 :
                                                                                   TENSOR_DTYPE_FLOAT;
                    if (tensor.desc.dtype != dtype || tensor.desc.dims.size() != 2 ||
                        tensor.desc.dims[0] != p.batchSize ||
                        tensor.desc.dims[1] != OpParam::StockhamSignalElements(p, i == 5))
                        return false;
                } else if (tensor.desc.dtype != (i == 4 ? TENSOR_DTYPE_INT32 : TENSOR_DTYPE_FLOAT) ||
                           tensor.desc.dims.size() != 1 || size_t(tensor.desc.dims[0]) != coefficientSizes[i])
                    return false;
                if (reinterpret_cast<uintptr_t>(tensor.data) % 32 != 0)
                    return false;
            }
            const auto x = reinterpret_cast<uintptr_t>(launch.GetInTensor(0).data);
            const auto y = reinterpret_cast<uintptr_t>(launch.GetOutTensor(0).data);
            if (x && y && (x <= y ? y - x < d.inputBytes : x - y < d.outputBytes))
                return false;
        } catch (const std::exception&) {
            return false;
        }
        return true;
    }
    uint64_t GetTilingSize(const LaunchParam&) const override { return sizeof(FftRealStockhamTiling); }
    Status InitImpl(const LaunchParam& launch) override
    {
        ASDSIP_CHECK(CanSupport(launch), "Invalid real Stockham launch",
                     return Status::FailStatus(ERROR_INVALID_VALUE));
        const auto& p = AnyCast<FftRealStockham>(launch.GetParam());
        const RealStockhamPlan d(p);
        const auto cores = PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_VECTOR);
        ASDSIP_CHECK(cores > 0 && kernelInfo_.GetTilingHostAddr(), "Missing Stockham tiling or AIVs",
                     return Status::FailStatus(ERROR_INVALID_VALUE));
        auto* t = reinterpret_cast<FftRealStockhamTiling*>(kernelInfo_.GetTilingHostAddr());
        // Only fields read by the donor mixed kernels are transferred.
        *t = {p.batchSize, p.fftN, {0, d.bufferBytes}, static_cast<int32_t>(d.radices.size()), int32_t(p.fftN & 1)};
        kernelInfo_.SetTilingId(0);
        kernelInfo_.SetBlockDim(static_cast<uint32_t>(std::min<int64_t>(p.batchSize, cores)));
        kernelInfo_.SetLocalMemorySize(0);
        kernelInfo_.GetScratchSizes() = {2 * d.bufferBytes};
        return Status::OkStatus();
    }

private:
    bool c2r_;
};

class FftRealStockhamC2RKernel : public FftRealStockhamKernel {
public:
    FftRealStockhamC2RKernel(const std::string& name, const BinHandle* handle) noexcept
        : FftRealStockhamKernel(name, handle, true)
    {}
};
class FftRealStockhamR2CKernel : public FftRealStockhamKernel {
public:
    FftRealStockhamR2CKernel(const std::string& name, const BinHandle* handle) noexcept
        : FftRealStockhamKernel(name, handle, false)
    {}
};
REG_OPERATION(FftRealStockhamOperation);
REG_KERNEL_BASE(FftRealStockhamC2RKernel);
REG_KERNEL_BASE(FftRealStockhamR2CKernel);
} // namespace AsdSip
