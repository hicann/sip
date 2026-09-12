/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once

#include "BlasPlan.h"

namespace AsdSip {
class BlasStrmvPlan : public BlasPlan {
public:
    BlasStrmvPlan(asdBlasFillMode_t uplo, asdBlasOperation_t trans, int64_t n);
    ~BlasStrmvPlan() override;
    AsdSip::AspbStatus CreateTensor() override;
    AsdSip::AspbStatus SetMaskTensor();
    AsdSip::AspbStatus FreeTensor() override;
    Mki::Tensor GetMaskTensor() { return this->maskTensor; }
    aclTensor* GetMaskAclTensor() { return this->maskAclTensor; }
    int64_t GetWorkspaceSize() override;

private:
    asdBlasFillMode_t uplo;   // 上下三角矩阵
    asdBlasOperation_t trans; // 方阵的阶
    int64_t n;
    Mki::Tensor maskTensor;
    aclTensor* maskAclTensor = nullptr;
};
} // namespace AsdSip
