/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "include/blasplan/BlasPlan.h"
#include "mki/utils/rt/rt.h"
#include "utils/ops_base.h"
#include "utils/mem_base_inner.h"
#include "utils/aspb_status.h"

constexpr uint32_t BLAS_DEFAULT_WORKSPACE_SIZE = 1024;

namespace AsdSip {

BlasPlan::BlasPlan() : blasStatus(AsdSip::AsdBlasPlanStatus::PLAN_UNINITIALIZED), stream(nullptr){};

void BlasPlan::MarkInitialized()
{
    blasStatus = AsdSip::AsdBlasPlanStatus::PLAN_INITIALIZED;
}

void BlasPlan::MarkFailed()
{
    blasStatus = AsdSip::AsdBlasPlanStatus::PLAN_FAILED;
}

bool BlasPlan::IsInitialized() const
{
    return blasStatus == AsdSip::AsdBlasPlanStatus::PLAN_INITIALIZED;
}

int64_t BlasPlan::GetWorkspaceSize()
{
    return BLAS_DEFAULT_WORKSPACE_SIZE;
}

bool BlasPlan::HasWorkSpace() const
{
    return workspace != nullptr;
}

void BlasPlan::SetWorkspace(uint8_t *workspacePtr)
{
    this->workspace = workspacePtr;
}

uint8_t *BlasPlan::GetWorkspace()
{
    return workspace;
}

AsdSip::AspbStatus BlasPlan::CreateTensor()
{
    return ErrorType::ACL_SUCCESS;
}

// 释放中间存储的tensor
AsdSip::AspbStatus BlasPlan::FreeTensor()
{
    return ErrorType::ACL_SUCCESS;
}

void *BlasPlan::GetStream()
{
    return stream;
}

void BlasPlan::SetStream(void *asdBalsStream)
{
    this->stream = asdBalsStream;
}

// 释放Plan相关的data
void BlasPlan::DestroyPlanData()
{
    workspace = nullptr;
    FreeTensor();
}

BlasPlan::~BlasPlan()
{
    DestroyPlanData();
}
}
