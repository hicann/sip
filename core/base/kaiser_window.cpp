/**
 * Copyright (c) 2026 gcw_2GoAgJFe. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <limits>
#include <memory>
#include <cstdint>
#include "base_api.h"
#include "aclnn/acl_meta.h"
#include "utils/assert.h"
#include "utils/ops_base.h"
#include "ops.h"
#include "kaiser_window.h"

namespace AsdSip {
namespace {
AspbStatus CheckKaiserOutput(aclTensor* out, int64_t length, Mki::TensorDType& dtype)
{
    ASDSIP_ECHECK(out != nullptr, "KaiserWindow: null output", ErrorType::ACL_ERROR_INVALID_PARAM);
    ASDSIP_ECHECK(length >= 0 && length <= std::numeric_limits<int32_t>::max(),
                  "KaiserWindow: windowLength must be in [0, INT32_MAX]", ErrorType::ACL_ERROR_INVALID_PARAM);
    aclDataType aclDtype;
    CHECK_STATUS_WITH_ACL_RETURN(aclGetDataType(out, &aclDtype), "KaiserWindow: aclGetDataType");
    ASDSIP_ECHECK(aclDtype == ACL_FLOAT || aclDtype == ACL_FLOAT16 || aclDtype == ACL_BF16,
                  "KaiserWindow: unsupported dtype", ErrorType::ACL_ERROR_UNSUPPORTED_DATA_TYPE);
    dtype = static_cast<Mki::TensorDType>(aclDtype);
    aclFormat format;
    CHECK_STATUS_WITH_ACL_RETURN(aclGetFormat(out, &format), "KaiserWindow: aclGetFormat");
    ASDSIP_ECHECK(format == ACL_FORMAT_ND, "KaiserWindow: output must use ND format",
                  ErrorType::ACL_ERROR_INVALID_PARAM);

    int64_t* rawDims = nullptr;
    uint64_t rank = 0;
    auto ret = aclGetViewShape(out, &rawDims, &rank);
    std::unique_ptr<int64_t[]> dims(rawDims);
    ASDSIP_ECHECK(ret == 0 && rank == 1 && dims && dims[0] == length,
                  "KaiserWindow: output shape must equal [windowLength]", ErrorType::ACL_ERROR_INVALID_PARAM);
    int64_t* rawStrides = nullptr;
    uint64_t strideRank = 0;
    ret = aclGetViewStrides(out, &rawStrides, &strideRank);
    std::unique_ptr<int64_t[]> strides(rawStrides);
    ASDSIP_ECHECK(ret == 0 && strideRank == 1 && strides && (length <= 1 || strides[0] == 1),
                  "KaiserWindow: output must be contiguous", ErrorType::ACL_ERROR_INVALID_PARAM);
    return ErrorType::ACL_SUCCESS;
}

AspbStatus CreateKaiserView(aclTensor* out, int64_t length, Mki::TensorDType dtype, aclTensor*& view)
{
    int64_t offset = 0;
    CHECK_STATUS_WITH_ACL_RETURN(aclGetViewOffset(out, &offset), "KaiserWindow: aclGetViewOffset");
    ASDSIP_ECHECK(offset >= 0, "KaiserWindow: negative output offset", ErrorType::ACL_ERROR_INVALID_PARAM);
    int64_t* rawStorageDims = nullptr;
    uint64_t storageRank = 0;
    const auto ret = aclGetStorageShape(out, &rawStorageDims, &storageRank);
    std::unique_ptr<int64_t[]> storageDims(rawStorageDims);
    ASDSIP_ECHECK(ret == 0 && storageDims && storageRank > 0, "KaiserWindow: invalid storage shape",
                  ErrorType::ACL_ERROR_INVALID_PARAM);
    uint64_t storageElements = 1;
    for (uint64_t dim = 0; dim < storageRank; ++dim) {
        ASDSIP_ECHECK(storageDims[dim] > 0 && static_cast<uint64_t>(storageDims[dim]) <=
                                                  std::numeric_limits<uint64_t>::max() / storageElements,
                      "KaiserWindow: invalid storage size", ErrorType::ACL_ERROR_INVALID_PARAM);
        storageElements *= static_cast<uint64_t>(storageDims[dim]);
    }
    const uint64_t viewOffset = static_cast<uint64_t>(offset);
    ASDSIP_ECHECK(viewOffset <= storageElements && static_cast<uint64_t>(length) <= storageElements - viewOffset,
                  "KaiserWindow: output view exceeds storage", ErrorType::ACL_ERROR_INVALID_PARAM);
    void* rawData = nullptr;
    CHECK_STATUS_WITH_ACL_RETURN(aclGetRawTensorAddr(out, &rawData), "KaiserWindow: aclGetRawTensorAddr");
    const uint64_t itemSize = dtype == Mki::TENSOR_DTYPE_FLOAT ? sizeof(float) : sizeof(uint16_t);
    const uintptr_t baseAddress = reinterpret_cast<uintptr_t>(rawData);
    ASDSIP_ECHECK(rawData != nullptr && viewOffset <= (std::numeric_limits<uintptr_t>::max() - baseAddress) / itemSize,
                  "KaiserWindow: invalid output address", ErrorType::ACL_ERROR_INVALID_PARAM);
    if (viewOffset == 0 && storageRank == 1 && storageDims[0] == length) {
        view = out;
        return ErrorType::ACL_SUCCESS;
    }
    auto* viewData = reinterpret_cast<void*>(baseAddress + viewOffset * itemSize);
    const int64_t stride = 1;
    // MKI uses storage addresses/shapes. Normalize the contiguous view so its
    // offset is respected without copying or allocating any output data.
    view = aclCreateTensor(&length, 1, static_cast<aclDataType>(dtype), &stride, 0, ACL_FORMAT_ND, &length, 1,
                           viewData);
    ASDSIP_ECHECK(view != nullptr, "KaiserWindow: create output view failed", ErrorType::ACL_ERROR_INTERNAL_ERROR);
    return ErrorType::ACL_SUCCESS;
}

struct KaiserKernelCache {
    aclrtContext context = nullptr;
    OpParam::KaiserWindow param;
    std::unique_ptr<Mki::Kernel> kernel;
    Mki::LaunchParam launch;
};

AspbStatus PrepareKaiserKernel(const OpParam::KaiserWindow& param, aclTensor* output, KaiserKernelCache*& state)
{
    // Bounded host-only cache: no output tensor or GM allocation is retained.
    // A thread owns its launch arguments; a context change invalidates tiling.
    // 线程模型假设（issue #399 已核实并在 PTA 侧注释固化）：handler 在入队期间于
    // 调用线程内同步执行，thread_local + 当前 context 作缓存键在该模型下成立；
    // 若后续引入异步 handler 投递（非调用线程执行），此处键需改为显式 device 绑定。
    thread_local KaiserKernelCache cache;
    aclrtContext context = nullptr;
    CHECK_STATUS_WITH_ACL_RETURN(aclrtGetCurrentContext(&context), "KaiserWindow: get current context");
    if (!cache.kernel || cache.context != context || !(cache.param == param)) {
        cache.kernel.reset();
        OpDesc desc;
        desc.opName = "KaiserWindowOperation";
        desc.specificParam = param;
        auto* operation = Ops::Instance().GetOperationByName(desc.opName);
        ASDSIP_ECHECK(operation != nullptr, "KaiserWindow: operation not found", ErrorType::ACL_ERROR_INTERNAL_ERROR);
        cache.launch.Reset();
        cache.launch.SetParam(desc.specificParam);
        cache.launch.AddOutTensor(output);
        const auto shapeStatus = operation->InferShape(cache.launch);
        ASDSIP_ECHECK(shapeStatus.Ok(), shapeStatus.Message(), ErrorType::ACL_ERROR_INTERNAL_ERROR);
        std::unique_ptr<Mki::Kernel> prepared(operation->GetBestKernel(cache.launch));
        ASDSIP_ECHECK(prepared != nullptr, "KaiserWindow: kernel not found", ErrorType::ACL_ERROR_INTERNAL_ERROR);
        prepared->SetLaunchWithTiling(true);
        const auto status = prepared->Init(cache.launch);
        ASDSIP_ECHECK(status.Ok(), status.Message(), ErrorType::ACL_ERROR_INTERNAL_ERROR);
        cache.context = context;
        cache.param = param;
        cache.kernel = std::move(prepared);
    }
    // Metadata is unchanged on a hit, but each factory call owns a fresh output.
    cache.launch.GetOutTensor(0).data = Mki::GetStorageAddr(output);
    cache.launch.GetOutTensor(0).hostData = nullptr;
    state = &cache;
    return ErrorType::ACL_SUCCESS;
}
} // namespace

extern "C" AspbStatus asdKaiserWindowGetWorkspaceSize(size_t* workspaceSize)
{
    ASDSIP_ECHECK(workspaceSize != nullptr, "KaiserWindow: null workspaceSize", ErrorType::ACL_ERROR_INVALID_PARAM);
    *workspaceSize = 0;
    return ErrorType::ACL_SUCCESS;
}

extern "C" AspbStatus asdKaiserWindow(aclTensor* outTensor, int64_t windowLength, bool periodic, double beta,
                                      void* stream, void* workspace)
{
    Mki::TensorDType dtype;
    const auto ret = CheckKaiserOutput(outTensor, windowLength, dtype);
    if (ret != ErrorType::ACL_SUCCESS || windowLength == 0) {
        return ret;
    }
    ASDSIP_ECHECK(stream != nullptr, "KaiserWindow: null stream", ErrorType::ACL_ERROR_INVALID_PARAM);
    aclTensor* rawView = nullptr;
    const auto viewStatus = CreateKaiserView(outTensor, windowLength, dtype, rawView);
    if (viewStatus != ErrorType::ACL_SUCCESS) {
        return viewStatus;
    }
    std::unique_ptr<aclTensor, decltype(&aclDestroyTensor)> ownedView(rawView == outTensor ? nullptr : rawView,
                                                                      &aclDestroyTensor);
    OpParam::KaiserWindow param;
    param.windowLength = windowLength;
    param.periodic = periodic;
    param.beta = beta;
    param.dtype = dtype;
    KaiserKernelCache* state = nullptr;
    const auto prepareStatus = PrepareKaiserKernel(param, rawView, state);
    if (prepareStatus != ErrorType::ACL_SUCCESS) {
        return prepareStatus;
    }
    const auto status = RunAsdOpsV2(stream, state->launch, static_cast<uint8_t*>(workspace), *state->kernel);
    state->launch.GetOutTensor(0).data = nullptr;
    ASDSIP_ECHECK(status.Ok(), status.Message(), ErrorType::ACL_ERROR_INTERNAL_ERROR);
    return ErrorType::ACL_SUCCESS;
}
} // namespace AsdSip
