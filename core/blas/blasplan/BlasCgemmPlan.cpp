/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "acl/acl.h"
#include "aclnn/acl_meta.h"
#include "utils/ops_base.h"
#include "utils/mem_base_inner.h"
#include "log/log.h"
#include "utils/aspb_status.h"
#include "include/blasplan/BlasCgemmPlan.h"

constexpr int TWO = 2;
constexpr int THREE = 3;
constexpr int64_t PAD_NUM = 128;
constexpr int64_t M0 = 128;
constexpr int64_t N0 = 128;
constexpr int64_t K0 = 128;

constexpr uint32_t BLAS_CGEMM_WORKSPACE_SIZE = 16 * 1024 * 1024; // 16:flattenNum

using namespace Mki;

namespace AsdSip {
BlasCgemmPlan::BlasCgemmPlan(CgemmPlanParam planParam) : BlasPlan(), planParam{planParam} {};

AsdSip::AspbStatus MallocTensorImpl(SVector<Tensor>& tensorList, int tensorNum, int tensorSize,
                                    SVector<aclTensor*>& aclTensorList)
{
    for (int i = 0; i < tensorNum; i++) {
        float* tensorData = nullptr;
        try {
            tensorData = new float[tensorSize];
        } catch (std::bad_alloc& e) {
            ASDSIP_LOG(ERROR) << "BlasCgemmPlan failed: " << e.what();
            ASDSIP_LOG(ERROR) << "malloc failed.";
            return ErrorType::ACL_ERROR_INTERNAL_ERROR;
        }
        Mki::Tensor curr;
        curr.desc = {TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {tensorSize}, {}, 0};
        curr.hostData = tensorData;
        curr.dataSize = tensorSize * sizeof(float);
        if (!MallocTensorInDevice((curr)).Ok()) {
            delete[] tensorData;
            curr.hostData = nullptr;
            tensorData = nullptr;
            ASDSIP_LOG(ERROR) << "BlasCgemmPlan malloc tensor in device failed.";
            return ErrorType::ACL_ERROR_INTERNAL_ERROR;
        }
        tensorList.push_back(curr);

        aclTensor* augAclTensor = nullptr;
        toAclTensor(curr, augAclTensor);
        if (augAclTensor == nullptr) {
            ASDSIP_LOG(ERROR) << "convert to aclTensor fail.";
            return ErrorType::ACL_ERROR_INTERNAL_ERROR;
        }

        aclTensorList.push_back(augAclTensor);

        tensorData = nullptr;
    }
    return ErrorType::ACL_SUCCESS;
}

AsdSip::AspbStatus BlasCgemmPlan::CreateTensor()
{
    int augABNum = 2;
    int augCNum = 4;

    int64_t lda_pad = (planParam.transa == asdBlasOperation_t::ASDBLAS_OP_N) ?
                          ((planParam.m + PAD_NUM - 1) / PAD_NUM * PAD_NUM) :
                          ((planParam.k + PAD_NUM - 1) / PAD_NUM * PAD_NUM);
    int64_t ldb_pad = (planParam.transb == asdBlasOperation_t::ASDBLAS_OP_N) ?
                          ((planParam.k + PAD_NUM - 1) / PAD_NUM * PAD_NUM) :
                          ((planParam.n + PAD_NUM - 1) / PAD_NUM * PAD_NUM);

    int64_t ASize = ((planParam.transa == asdBlasOperation_t::ASDBLAS_OP_N) ?
                         (lda_pad * (planParam.k + K0 - 1) / K0 * K0) :
                         (lda_pad * (planParam.m + M0 - 1) / M0 * M0));
    int64_t BSize = ((planParam.transb == asdBlasOperation_t::ASDBLAS_OP_N) ?
                         (ldb_pad * (planParam.n + N0 - 1) / N0 * N0) :
                         (ldb_pad * (planParam.k + K0 - 1) / K0 * K0));
    int64_t CSize = planParam.ldc * planParam.n;

    AsdSip::AspbStatus status;

    status = MallocTensorImpl(this->augATensors, augABNum, ASize, this->augAAclTensors);
    if (status != ErrorType::ACL_SUCCESS) {
        ASDSIP_LOG(ERROR) << "BlasCgemmPlan CreateTensor failed.";
        return ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }

    status = MallocTensorImpl(this->augBTensors, augABNum, BSize, this->augBAclTensors);
    if (status != ErrorType::ACL_SUCCESS) {
        ASDSIP_LOG(ERROR) << "BlasCgemmPlan CreateTensor failed.";
        return ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }

    status = MallocTensorImpl(this->augCTensors, augCNum, CSize, this->augCAclTensors);
    if (status != ErrorType::ACL_SUCCESS) {
        ASDSIP_LOG(ERROR) << "BlasCgemmPlan CreateTensor failed.";
        return ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }

    return ErrorType::ACL_SUCCESS;
}

void BlasCgemmPlan::FreeGivenTensor(Mki::Tensor tensor) const
{
    if (tensor.data != nullptr) {
        FreeTensorInDevice(tensor);
        tensor.data = nullptr;
    }
    if (tensor.hostData != nullptr) {
        delete[] static_cast<float*>(tensor.hostData);
        tensor.hostData = nullptr;
    }
}

void BlasCgemmPlan::FreeAclTensors(Mki::SVector<aclTensor*> aclTensorList) const
{
    for (auto* ptr : aclTensorList) {
        aclDestroyTensor(ptr);
    }
    aclTensorList.clear();
}

AsdSip::AspbStatus BlasCgemmPlan::FreeTensor()
{
    for (auto tensor : this->augATensors) {
        FreeGivenTensor(tensor);
    }
    for (auto tensor : this->augBTensors) {
        FreeGivenTensor(tensor);
    }
    for (auto tensor : this->augCTensors) {
        FreeGivenTensor(tensor);
    }
    FreeAclTensors(this->augAAclTensors);
    FreeAclTensors(this->augBAclTensors);
    FreeAclTensors(this->augCAclTensors);
    return ErrorType::ACL_SUCCESS;
}

BlasCgemmPlan::~BlasCgemmPlan() { BlasPlan::DestroyPlanData(); }

int64_t BlasCgemmPlan::GetWorkspaceSize() { return BLAS_CGEMM_WORKSPACE_SIZE; }
} // namespace AsdSip
