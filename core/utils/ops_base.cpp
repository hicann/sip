/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "utils/ops_base.h"
#include <memory>
#include <iostream>
#include "ops.h"
#include "mki/types.h"
#include "log/log.h"
#include "mki/utils/platform/platform_info.h"
#include "acl_meta.h"

using namespace Mki;
using namespace AsdSip;

static Status MallocOutTensor(LaunchParam& launchParam, const Operation* op, SVector<Tensor>& outTensorList)
{
    AsdSip::OpDesc opDesc;
    opDesc.specificParam = launchParam.GetParam();
    int64_t outTensorNum = op->GetOutputNum(launchParam.GetParam());
    if (outTensorList.size() == 0) {
        for (int64_t i = 0; i < outTensorNum; i++) {
            Tensor tensor;
            launchParam.AddOutTensor(tensor);
            outTensorList.push_back(tensor);
        }
    } else if (outTensorList.size() == static_cast<size_t>(outTensorNum)) {
        for (int64_t i = 0; i < outTensorNum; i++) {
            launchParam.AddOutTensor(outTensorList.at(i));
        }
    } else {
        return Status::FailStatus(-1, "outTensorList size is wrong.");
    }

    Status status = op->InferShape(launchParam);
    if (!status.Ok()) {
        return status;
    }

    for (int64_t i = 0; i < outTensorNum; i++) {
        Tensor& tensor = launchParam.GetOutTensor(i);
        Tensor& outTensor = outTensorList.at(i);

        tensor.dataSize = static_cast<size_t>(tensor.Numel()) *
                          static_cast<size_t>(GetTensorElementSize(tensor.desc.dtype));

        outTensor.dataSize = tensor.dataSize;
        outTensor.desc = tensor.desc;
        if (tensor.data == nullptr) {
            int st = MkiRtMemMallocDevice(&tensor.data, tensor.dataSize, MKIRT_MEM_DEFAULT);
            if (st != MKIRT_SUCCESS) {
                return Status::FailStatus(-1, "Device malloc outtensor failed.");
            }
            outTensor.data = tensor.data;
        }
    }

    return Status::OkStatus();
}

static Status MallocAndSetWorkspace(const KernelInfo& kernelInfo, RunInfo& runInfo)
{
    size_t bufferSize = kernelInfo.GetTotalScratchSize();
    if (bufferSize == 0) {
        ASDSIP_LOG(INFO) << "no workspace";
        return Status::OkStatus();
    }
    uint8_t* deviceBuffer = nullptr;
    void* tempDevicePtr = nullptr;
    int ret = MkiRtMemMallocDevice(&tempDevicePtr, bufferSize, MKIRT_MEM_DEFAULT);
    if (ret != MKIRT_SUCCESS) {
        ASDSIP_LOG(ERROR) << "MkiRtMemMallocDevice fail, errCode:" << ret << ", errName:" << MkiRtErrorName(ret)
                          << "errDesc:" << MkiRtErrorDesc(ret);
        return Status::FailStatus(-1, "malloc Workspace memory fail");
    }
    deviceBuffer = static_cast<uint8_t*>(tempDevicePtr);
    runInfo.SetScratchDeviceAddr(deviceBuffer);
    return Status::OkStatus();
}

static Status FreeWorkspace(const KernelInfo& kernelInfo, RunInfo& runInfo)
{
    uint8_t* deviceBuffer = runInfo.GetScratchDeviceAddr();
    size_t bufferSize = kernelInfo.GetTotalScratchSize();
    if (deviceBuffer != nullptr && bufferSize != 0) {
        MkiRtStreamSynchronize(runInfo.GetStream());
        MkiRtMemFreeDevice(deviceBuffer);
    }

    return Status::OkStatus();
}

static Status RunAsdOpsImpl(LaunchParam& launchParam, const AsdSip::OpDesc& opDesc, RunInfo& runInfo,
                            const SVector<Tensor>& inTensorList, SVector<Tensor>& outTensorList, uint8_t* workspace)
{
    Operation* op = AsdSip::Ops::Instance().GetOperationByName(opDesc.opName);
    if (op == nullptr) {
        return Status::FailStatus(-1, "Get operation failed.");
    }

    int64_t inTensorNum = op->GetInputNum(launchParam.GetParam());
    if (inTensorList.size() != static_cast<size_t>(inTensorNum)) {
        return Status::FailStatus(-1, "Check inTensorList size failed.");
    }

    Status statusInfo = MallocOutTensor(launchParam, op, outTensorList);
    if (!statusInfo.Ok()) {
        return statusInfo;
    }
    std::unique_ptr<Kernel> kernel = static_cast<std::unique_ptr<Kernel>>(op->GetBestKernel(launchParam));
    if (kernel == nullptr) {
        return Status::FailStatus(-1, "Get best kernel failed.");
    }
    kernel->SetLaunchWithTiling(true);
    kernel->Init(launchParam);

    if (workspace == nullptr) {
        const KernelInfo& kernelInfo = kernel->GetKernelInfo();
        statusInfo = MallocAndSetWorkspace(kernelInfo, runInfo);
        if (!statusInfo.Ok()) {
            return statusInfo;
        }
    } else {
        runInfo.SetScratchDeviceAddr(workspace);
    }

    ASDSIP_LOG(DEBUG) << kernel->GetName() << " run start, LaunchParam:\n" << launchParam.ToString();
    ASDSIP_LOG(DEBUG) << "RunInfo:\n" << runInfo.ToString();
    statusInfo = kernel->Run(launchParam, runInfo);

    ASDSIP_LOG_IF(!statusInfo.Ok(), ERROR) << kernel->GetName() << " run fail, error:" << statusInfo.ToString();
    if (!statusInfo.Ok()) {
        return statusInfo;
    }

    if (workspace == nullptr) {
        statusInfo = FreeWorkspace(kernel->GetKernelInfo(), runInfo);
        if (!statusInfo.Ok()) {
            return statusInfo;
        }
    }

    return statusInfo;
}

Status RunAsdOps(MkiRtStream stream, const AsdSip::OpDesc& opDesc, const SVector<Tensor>& inTensorList,
                 SVector<Tensor>& outTensorList, uint8_t* workspace)
{
    if (stream == nullptr) {
        ASDSIP_LOG(ERROR) << "stream is nullptr!";
        return Status::FailStatus(-1, "stream is nullptr!");
    }
    RunInfo runInfo;
    runInfo.SetStream(stream);
    LaunchParam launchParam;
    launchParam.SetParam(opDesc.specificParam);

    for (size_t i = 0; i < inTensorList.size(); i++) {
        launchParam.AddInTensor(inTensorList.at(i));
    }

    Status status = RunAsdOpsImpl(launchParam, opDesc, runInfo, inTensorList, outTensorList, workspace);
    if (!status.Ok()) {
        return status;
    }

    return Status::OkStatus();
}

Status MallocTensorInDevice(Tensor& tensor)
{
    int st = MkiRtMemMallocDevice(&tensor.data, tensor.dataSize, MKIRT_MEM_DEFAULT);
    if (st != MKIRT_SUCCESS) {
        ASDSIP_LOG(ERROR) << "Device malloc intensor failed.";
        return Status::FailStatus(-1, "Device malloc intensor failed.");
    }

    st = MkiRtMemCopy(tensor.data, tensor.dataSize, tensor.hostData, tensor.dataSize, MKIRT_MEMCOPY_HOST_TO_DEVICE);
    if (st != MKIRT_SUCCESS) {
        ASDSIP_LOG(ERROR) << "Memcpy host to device failed.";
        return Status::FailStatus(-1, "Memcpy host to device failed.");
    }

    return Status::OkStatus();
}

Status CopyOutTensorToHost(Tensor& tensor)
{
    if (tensor.hostData == nullptr) {
        tensor.hostData = malloc(tensor.dataSize);
        if (tensor.hostData == nullptr) {
            return Status::FailStatus(-1, "Host malloc outtensor failed.");
        }
    }

    int st = MkiRtMemCopy(tensor.hostData, tensor.dataSize, tensor.data, tensor.dataSize, MKIRT_MEMCOPY_DEVICE_TO_HOST);
    if (st != MKIRT_SUCCESS) {
        ASDSIP_LOG(ERROR) << "MkiRtMemCopy";
        return Status::FailStatus(-1, "Memcpy outtensor device to host failed");
    }

    return Status::OkStatus();
}

Status FreeTensorInDevice(const Tensor& tensor)
{
    MkiRtMemFreeDevice(tensor.data);
    return Status::OkStatus();
}

static Status RunAsdOpsImplV2(LaunchParam& launchParam, const AsdSip::OpDesc& opDesc, RunInfo& runInfo,
                              uint8_t* workspace)
{
    Operation* op = AsdSip::Ops::Instance().GetOperationByName(opDesc.opName);
    if (op == nullptr) {
        return Status::FailStatus(-1, "Get operation failed.");
    }

    Status status = op->InferShape(launchParam);
    if (!status.Ok()) {
        return status;
    }
    std::unique_ptr<Kernel> kernel = static_cast<std::unique_ptr<Kernel>>(op->GetBestKernel(launchParam));
    if (kernel == nullptr) {
        return Status::FailStatus(-1, "Get best kernel failed.");
    }
    kernel->SetLaunchWithTiling(true);
    kernel->Init(launchParam);
    if (workspace == nullptr) {
        const KernelInfo& kernelInfo = kernel->GetKernelInfo();
        status = MallocAndSetWorkspace(kernelInfo, runInfo);
        if (!status.Ok()) {
            return status;
        }
    } else {
        runInfo.SetScratchDeviceAddr(workspace);
    }
    ASDSIP_LOG(INFO) << kernel->GetName() << " run start, LaunchParam:\n" << launchParam.ToString();
    ASDSIP_LOG(INFO) << "RunInfo:\n" << runInfo.ToString();
    status = kernel->Run(launchParam, runInfo);

    ASDSIP_LOG_IF(!status.Ok(), ERROR) << kernel->GetName() << " run fail, error:" << status.ToString();
    if (!status.Ok()) {
        return status;
    }

    if (workspace == nullptr) {
        status = FreeWorkspace(kernel->GetKernelInfo(), runInfo);
        if (!status.Ok()) {
            return status;
        }
    }

    return status;
}

Status RunAsdOpsV2(MkiRtStream stream, const AsdSip::OpDesc& opDesc, const SVector<aclTensor*>& inTensorList,
                   SVector<aclTensor*>& outTensorList, uint8_t* workspace)
{
    if (stream == nullptr) {
        ASDSIP_LOG(ERROR) << "stream is nullptr!";
        return Status::FailStatus(-1, "stream is nullptr!");
    }
    RunInfo runInfo;
    runInfo.SetStream(stream);
    LaunchParam launchParam;
    launchParam.SetParam(opDesc.specificParam);

    for (size_t i = 0; i < inTensorList.size(); i++) {
        launchParam.AddInTensor(inTensorList.at(i));
    }

    for (size_t i = 0; i < outTensorList.size(); i++) {
        launchParam.AddOutTensor(outTensorList.at(i));
    }
    Status status = RunAsdOpsImplV2(launchParam, opDesc, runInfo, workspace);
    if (!status.Ok()) {
        ASDSIP_LOG(ERROR) << "Execute RunAsdOpsImplV2 failed.";
        return status;
    }

    ASDSIP_LOG(INFO) << "Execute RunAsdOpsV2 success.";
    return Status::OkStatus();
}

// #ifndef UNITTEST
Status toAclTensor(const Tensor& inTensor, aclTensor*& outTensor, std::vector<int64_t> stride)
{
    SVector<int64_t> shape = inTensor.desc.dims;

    if (stride.size() == 0) {
        stride.resize(shape.size(), 1);
        for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; i--) {
            stride[i] = shape[i + 1] * stride[i + 1];
        }
        outTensor = aclCreateTensor(shape.data(), shape.size(), static_cast<aclDataType>(inTensor.desc.dtype),
                                    stride.data(), inTensor.desc.offset, static_cast<aclFormat>(inTensor.desc.format),
                                    shape.data(), shape.size(), inTensor.data);
    } else {
        int64_t max_id = 0;
        if (stride.size() < shape.size()) {
            ASDSIP_LOG(ERROR) << "tensor stride size is not equal shape size!"
                              << "expected stride size is [ " << shape.size() << " ],"
                              << "actually is [ " << stride.size() << " ].";
            size_t strideOriginalSize = stride.size();
            for (auto i = strideOriginalSize; i < shape.size(); i++) {
                stride.push_back(1);
            }
            for (auto i = shape.size() - 2; i >= strideOriginalSize; i--) {
                stride[i] = shape[i + 1] * stride[i + 1];
            }
        }
        for (int64_t i = 1; i < static_cast<int64_t>(shape.size()); i++) {
            if (stride[i] > stride[max_id]) {
                max_id = i;
            }
        }
        std::vector<int64_t> storageShape{stride[max_id] * shape[max_id] + inTensor.desc.offset};
        outTensor = aclCreateTensor(shape.data(), shape.size(), static_cast<aclDataType>(inTensor.desc.dtype),
                                    stride.data(), inTensor.desc.offset, static_cast<aclFormat>(inTensor.desc.format),
                                    storageShape.data(), storageShape.size(), inTensor.data);
    }
    return Status::OkStatus();
}
// #endif
