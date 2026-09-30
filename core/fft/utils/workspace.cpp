/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "log/log.h"
#include "utils/workspace.h"

using Mki::Tensor;

namespace workspace {

Workspace::Workspace(const Tensor& dataSegment)
{
    dataPtr_ = dataSegment.data;
    dataSize_ = dataSegment.dataSize;
}

Workspace::Workspace(void* workSpace) { dataPtr_ = workSpace; }

Workspace::Workspace(void* workSpace, size_t capacity)
{
    dataPtr_ = workSpace;
    dataSize_ = capacity;
    capacity_ = capacity;
}

bool Workspace::isInitialized() const { return dataPtr_ != nullptr; }

void Workspace::Reset() { offsets_.resize(1); }

void* Workspace::allocate(size_t dataSize)
{
    dataSize = getAlignedSize(dataSize);
    size_t offset = offsets_.back();

    if (capacity_ != 0 && offset + dataSize > capacity_) {
        ASDSIP_LOG(ERROR) << "Workspace allocation exceeds capacity: need " << offset + dataSize << ", capacity "
                          << capacity_ << ".";
        throw std::runtime_error("Workspace allocation exceeds capacity.");
    }

    offsets_.push_back(offset + dataSize);

    return (uint8_t*)dataPtr_ + offset;
}

void Workspace::recycleLast()
{
    if (offsets_.size() <= 1) {
        ASDSIP_LOG(ERROR) << "There are no allocated tensors.";
        throw std::runtime_error("There are no allocated tensors.");
    }

    offsets_.pop_back();
}

} // namespace workspace
