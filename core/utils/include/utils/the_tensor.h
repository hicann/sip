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

#include <vector>
#include "log/log.h"

namespace wten {

template <typename T>
class TheTensor {
public:
    // 禁用拷贝: data_ 为裸指针, 浅拷贝会导致析构时 double-free
    TheTensor(const TheTensor&) = delete;
    TheTensor& operator=(const TheTensor&) = delete;

    TheTensor(const std::vector<int64_t>& shape)
        : shape_(shape), strides_(std::vector<int64_t>(shape.size())), size_(compute_size(shape)), data_(new T[size_])
    {
        update_strides();
    }

    TheTensor(TheTensor&& tensor)
        : shape_(tensor.shape_), strides_(tensor.strides_), size_(tensor.size_), data_(tensor.move_data())
    {}

    TheTensor<T>& operator=(TheTensor<T>&& tensor)
    {
        delete[] data_;
        data_ = tensor.data_;
        size_ = tensor.size_;
        shape_ = tensor.shape_;
        strides_ = tensor.strides_;
        tensor.data_ = nullptr;
        return *this;
    }

    T& operator[](size_t index) const { return data_[index]; }

    T* data() const { return data_; }

    T* move_data()
    {
        T* data = data_;
        data_ = nullptr;
        return data;
    }

    void free_data() { delete[] move_data(); }

    size_t size() const { return size_; }

    size_t ndim() const { return shape_.size(); }

    const std::vector<int64_t>& shape() const { return shape_; }

    const std::vector<int64_t>& strides() const { return strides_; }

    void reshape(const std::vector<int64_t>& shape)
    {
        if (size_ != compute_size(shape)) {
            ASDSIP_LOG(ERROR) << "Invalid shape.";
            return;
        }

        shape_ = shape;
        update_strides();
    }

    ~TheTensor() { delete[] data_; };

private:
    std::vector<int64_t> shape_;
    std::vector<int64_t> strides_;
    size_t size_;
    T* data_;

    size_t compute_size(const std::vector<int64_t>& shape) const
    {
        size_t size = 1;
        for (size_t dim : shape) {
            size *= dim;
        }

        return size;
    }

    void update_strides()
    {
        strides_.resize(ndim());
        int64_t stride = 1;
        for (int64_t i = static_cast<int64_t>(shape_.size()) - 1; i >= 0; i--) {
            strides_[i] = stride;
            stride *= shape_[i];
        }
    }
};

} // namespace wten
