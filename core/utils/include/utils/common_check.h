/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASDSIP_COMMON_CHECK_H
#define ASDSIP_COMMON_CHECK_H

#include <cstdint>
#include <optional>
#include <vector>
#include "utils/aspb_status.h"
#include "utils/mem_base.h"
#include "acl/acl.h"
#include "aclnn/acl_meta.h"

std::optional<int64_t> GetTensorNum(uint64_t dimsNum, int64_t* storageDims);

AsdSip::AspbStatus AsdTensorDtypeCheck(const aclTensor* tensor, const aclDataType expectedDtype);

AsdSip::AspbStatus AsdTensorShapeCheck(const aclTensor* tensor, const std::vector<int64_t> expectedDims);

AsdSip::AspbStatus AsdTensorNumCheck(const aclTensor* tensor, int64_t expectedNum);

AsdSip::AspbStatus AsdTensorInvalidShapeCheck(const aclTensor* tensor);

#define SIP_OP_CHECK_DTYPE_NOT_MATCH(tensor, expectedDtype, ret)                \
    if (AsdTensorDtypeCheck(tensor, expectedDtype) != ErrorType::ACL_SUCCESS) { \
        ASDSIP_ELOG(ret) << #tensor << " get wrong dtype.";                     \
        return ret;                                                             \
    }

#define SIP_OP_CHECK_SHAPE_NOT_MATCH(tensor, expectedDims, ret)                \
    if (AsdTensorShapeCheck(tensor, expectedDims) != ErrorType::ACL_SUCCESS) { \
        ASDSIP_ELOG(ret) << #tensor << " get wrong shape.";                    \
        return ret;                                                            \
    }

#define SIP_OP_CHECK_NUM_NOT_MATCH(tensor, expectedNum, ret)                \
    if (AsdTensorNumCheck(tensor, expectedNum) != ErrorType::ACL_SUCCESS) { \
        ASDSIP_ELOG(ret) << #tensor << " get wrong element num.";           \
        return ret;                                                         \
    }

#define SIP_OP_CHECK_INVALID_SHAPE(tensor, ret)                         \
    if (AsdTensorInvalidShapeCheck(tensor) != ErrorType::ACL_SUCCESS) { \
        ASDSIP_ELOG(ret) << #tensor << " get invalid shape.";           \
        return ret;                                                     \
    }
#endif
