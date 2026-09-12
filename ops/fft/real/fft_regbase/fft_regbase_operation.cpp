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
#include "params/fft_real_regbase.h"
#include "params/fft_real_regbase_tiling.h"
#include "utils/assert.h"

namespace AsdSip {
using namespace Mki;
using OpParam::FftRealRegBase;
using OpParam::FftRealStage;

class FftRealRegBaseOperation : public OperationBase {
public:
    explicit FftRealRegBaseOperation(const std::string& name) noexcept : OperationBase(name) {}
    int64_t GetInputNum(const Any& param) const override { return param.Type() == typeid(FftRealRegBase) ? 3 : 0; }
    Kernel* GetBestKernel(const LaunchParam& launch) const override
    {
        if (!IsConsistent(launch) || launch.GetParam().Type() != typeid(FftRealRegBase))
            return nullptr;
        return GetKernelByName(OpParam::RealKernelName(AnyCast<FftRealRegBase>(launch.GetParam()).stage));
    }

protected:
    Status InferShapeImpl(const LaunchParam& launch, SVector<Tensor>& outputs) const override
    {
        ASDSIP_CHECK(
            launch.GetParam().Type() == typeid(FftRealRegBase) && launch.GetInTensorCount() == 3 && outputs.size() == 1,
            "Real RegBase expects three inputs and one output", return Status::FailStatus(ERROR_INFERSHAPE_ERROR));
        const auto& p = AnyCast<FftRealRegBase>(launch.GetParam());
        try {
            (void)OpParam::RealDescriptor(p, 1);
        } catch (const std::exception&) {
            return Status::FailStatus(ERROR_INFERSHAPE_ERROR);
        }
        outputs[0].desc = {
            TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {p.batchSize, OpParam::RealSignalFloats(p, true)}, {}, 0};
        return Status::OkStatus();
    }
};

class FftRealRegBaseKernel : public KernelBase {
public:
    FftRealRegBaseKernel(const std::string& name, const BinHandle* handle, FftRealStage stage)
        : KernelBase(name, handle), stage_(stage)
    {}
    bool CanSupport(const LaunchParam& launch) const override
    {
        if (PlatformInfo::Instance().GetPlatformType() != PlatformType::ASCEND_950 ||
            launch.GetParam().Type() != typeid(FftRealRegBase) || launch.GetInTensorCount() != 3 ||
            launch.GetOutTensorCount() != 1)
            return false;
        const auto& p = AnyCast<FftRealRegBase>(launch.GetParam());
        if (p.stage != stage_)
            return false;
        try {
            const auto d = OpParam::RealDescriptor(p, 1);
            if ((stage_ == FftRealStage::Fused) != d.IsFused() || (stage_ == FftRealStage::Small && d.core.Large()))
                return false;
            for (int i = 0; i < 4; ++i) {
                const auto& tensor = i == 3 ? launch.GetOutTensor(0) : launch.GetInTensor(i);
                if (tensor.desc.dtype != TENSOR_DTYPE_FLOAT || tensor.desc.offset != 0 || !tensor.IsContiguous())
                    return false;
                if (i == 0 || i == 3) {
                    if (tensor.desc.dims.size() != 2 || tensor.desc.dims[0] != p.batchSize ||
                        tensor.desc.dims[1] != OpParam::RealSignalFloats(p, i == 3))
                        return false;
                } else if (tensor.desc.dims.size() != 1 ||
                           tensor.desc.dims[0] != OpParam::RealCoefficientFloats(p, d, i))
                    return false;
            }
            const auto x = reinterpret_cast<uintptr_t>(launch.GetInTensor(0).data);
            const auto y = reinterpret_cast<uintptr_t>(launch.GetOutTensor(0).data);
            if ((x % 32) || (y % 32))
                return false;
            if (x && y &&
                (x <= y ? y - x < size_t(p.batchSize) * OpParam::RealSignalFloats(p, false) * 4 :
                          x - y < size_t(p.batchSize) * OpParam::RealSignalFloats(p, true) * 4))
                return false;
        } catch (const std::exception&) {
            return false;
        }
        return true;
    }
    uint64_t GetTilingSize(const LaunchParam&) const override { return sizeof(FftRealRegBaseTiling); }
    Status InitImpl(const LaunchParam& launch) override
    {
        ASDSIP_CHECK(CanSupport(launch), "Invalid real RegBase launch", return Status::FailStatus(ERROR_INVALID_VALUE));
        const auto& p = AnyCast<FftRealRegBase>(launch.GetParam());
        const auto cores = PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_VECTOR);
        ASDSIP_CHECK(cores > 0 && kernelInfo_.GetTilingHostAddr(), "Missing real FFT tiling or AIVs",
                     return Status::FailStatus(ERROR_INVALID_VALUE));
        const auto d = OpParam::RealDescriptor(p, cores);
        auto* t = reinterpret_cast<FftRealRegBaseTiling*>(kernelInfo_.GetTilingHostAddr());
        *t = {p.batchSize, d.core.n, d.core.tilePoints, p.fftN, p.isInverse, p.isC2R, d.boundaryTilePoints};
        kernelInfo_.SetTilingId(OpParam::RealTilingKey(p, d));
        kernelInfo_.SetBlockDim(stage_ == FftRealStage::Fused    ? d.fusedBlocks :
                                stage_ == FftRealStage::Boundary ? d.boundaryBlocks :
                                                                   d.core.phaseABlocks);
        // Static UB arrays, matching donor <<<blocks,0,stream>>> launches.
        kernelInfo_.SetLocalMemorySize(0);
        kernelInfo_.GetScratchSizes() = {0};
        return Status::OkStatus();
    }

private:
    FftRealStage stage_;
};

#define REAL_KERNEL(Name, Stage)                                        \
    class Name : public FftRealRegBaseKernel {                          \
    public:                                                             \
        Name(const std::string& name, const BinHandle* handle) noexcept \
            : FftRealRegBaseKernel(name, handle, FftRealStage::Stage)   \
        {}                                                              \
    };                                                                  \
    REG_KERNEL_BASE(Name)

REG_OPERATION(FftRealRegBaseOperation);
REAL_KERNEL(FftRealRegBaseFusedKernel, Fused);
REAL_KERNEL(FftRealRegBaseBoundaryKernel, Boundary);
REAL_KERNEL(FftRealRegBaseSmallKernel, Small);
} // namespace AsdSip
