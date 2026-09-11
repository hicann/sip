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

namespace BlasPlanCache {

// 绑定 handle 与 plan；handle 已绑定 plan 时返回 false 且不接管 plan 所有权（issue #129）
bool MakePlan(AsdSip::asdBlasHandle& handle, AsdSip::BlasPlan* plan);

AsdSip::asdBlasHandle InitHandle();

bool doesPlanExist(AsdSip::asdBlasHandle& handle);

AsdSip::BlasPlan& getPlan(AsdSip::asdBlasHandle& handle);

void destroy_plan(AsdSip::asdBlasHandle& handle);
} // namespace BlasPlanCache
