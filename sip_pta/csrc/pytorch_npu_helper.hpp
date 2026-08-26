/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef PYTORCH_NPU_HELPER_HPP_
#define PYTORCH_NPU_HELPER_HPP_

#include <vector>
#include "BlasMixCache.h"
#include "FFTMixCache.h"
#include "constants.h"
#include "pytorch_npu_helper_utils.hpp"

inline at::Tensor CopyTensorHostToDevice(const at::Tensor& cpu_tensor)
{
    at::Tensor cpuPinMemTensor = cpu_tensor.pin_memory();
    int deviceIndex = 0;
    return cpuPinMemTensor.to(c10::Device(torch_npu::utils::get_npu_device_type(), deviceIndex),
                              cpuPinMemTensor.scalar_type(), true, true);
}

inline at::Tensor CopyScalarToDevice(const c10::Scalar& cpu_scalar, at::ScalarType scalar_data_type)
{
    return CopyTensorHostToDevice(scalar_to_tensor(cpu_scalar).to(scalar_data_type));
}

inline aclTensor* ConvertType(const at::Tensor& at_tensor)
{
    static const auto aclCreateTensor = GET_OP_API_FUNC(aclCreateTensor);
    if (aclCreateTensor == nullptr || !at_tensor.defined()) {
        return nullptr;
    }
    at::ScalarType scalar_data_type = at_tensor.scalar_type();
    aclDataType acl_data_type = kATenScalarTypeToAclDataTypeTable[static_cast<int64_t>(scalar_data_type)];
    TORCH_CHECK(acl_data_type != ACL_DT_UNDEFINED,
                std::string(c10::toString(scalar_data_type)) + " has not been supported")
    c10::SmallVector<int64_t, sip_pta::DIM_5> storageDims;
    // if acl_data_type is ACL_STRING, storageDims is empty.
    auto itemsize = at_tensor.itemsize();
    if (itemsize == 0) {
        AT_ERROR("When ConvertType, tensor item size of cannot be zero.");
        return nullptr;
    }
    if (acl_data_type != ACL_STRING) {
        storageDims.push_back(at_tensor.storage().nbytes() / itemsize);
    }

    const auto dimNum = at_tensor.sizes().size();
    aclFormat format = ACL_FORMAT_ND;
    switch (dimNum) {
        case sip_pta::DIM_2:
            format = ACL_FORMAT_NCL;
            break;
        case sip_pta::DIM_4:
            format = ACL_FORMAT_NCHW;
            break;
        case sip_pta::DIM_5:
            format = ACL_FORMAT_NCDHW;
            break;
        default:
            format = ACL_FORMAT_ND;
    }

    if (at_tensor.unsafeGetTensorImpl()->is_wrapped_number()) {
        c10::Scalar expScalar = ConvertTensorToScalar(at_tensor);
        // 局部张量在函数返回时析构释放设备存储，返回的 aclTensor* 持有悬垂指针
        // 构成 use-after-free（issue #119）。一次算子执行可能连续转换多个标量，
        // 单个 static 槽位会被后续调用覆盖；与 #120 一致，用 thread_local 容器
        // 为每次调用持有独立张量（无覆盖/无别名/无线程竞态，覆盖 EXEC_FUNC 窗口）。
        thread_local std::vector<at::Tensor> scalarTensorHolder;
        scalarTensorHolder.push_back(CopyScalarToDevice(expScalar, scalar_data_type));
        const at::Tensor& aclInput = scalarTensorHolder.back();
        return aclCreateTensor(aclInput.sizes().data(), aclInput.sizes().size(), acl_data_type,
                               aclInput.strides().data(), aclInput.storage_offset(), format, storageDims.data(),
                               storageDims.size(), const_cast<void*>(aclInput.storage().data()));
    }

    auto acl_tensor = aclCreateTensor(at_tensor.sizes().data(), at_tensor.sizes().size(), acl_data_type,
                                      at_tensor.strides().data(), at_tensor.storage_offset(), format,
                                      storageDims.data(), storageDims.size(),
                                      const_cast<void*>(at_tensor.storage().data()));
    return acl_tensor;
}

inline aclScalar* ConvertType(const at::Scalar& at_scalar)
{
    static const auto aclCreateScalar = GET_OP_API_FUNC(aclCreateScalar);
    if (aclCreateScalar == nullptr) {
        return nullptr;
    }

    at::ScalarType scalar_data_type = at_scalar.type();
    aclDataType acl_data_type = kATenScalarTypeToAclDataTypeTable[static_cast<int64_t>(scalar_data_type)];
    TORCH_CHECK(acl_data_type != ACL_DT_UNDEFINED,
                std::string(c10::toString(scalar_data_type)) + " has not been supported")
    aclScalar* acl_scalar = nullptr;
    switch (scalar_data_type) {
        case at::ScalarType::Double: {
            double value = at_scalar.toDouble();
            acl_scalar = aclCreateScalar(&value, acl_data_type);
            break;
        }
        case at::ScalarType::Long: {
            int64_t value = at_scalar.toLong();
            acl_scalar = aclCreateScalar(&value, acl_data_type);
            break;
        }
        case at::ScalarType::Bool: {
            bool value = at_scalar.toBool();
            acl_scalar = aclCreateScalar(&value, acl_data_type);
            break;
        }
        case at::ScalarType::ComplexDouble: {
            auto value = at_scalar.toComplexDouble();
            acl_scalar = aclCreateScalar(&value, acl_data_type);
            break;
        }
        default:
            acl_scalar = nullptr;
            break;
    }
    return acl_scalar;
}

inline aclIntArray* ConvertType(const at::IntArrayRef& at_array)
{
    static const auto aclCreateIntArray = GET_OP_API_FUNC(aclCreateIntArray);
    if (aclCreateIntArray == nullptr) {
        return nullptr;
    }
    auto array = aclCreateIntArray(at_array.data(), at_array.size());
    return array;
}

template <std::size_t N>
inline aclBoolArray* ConvertType(const std::array<bool, N>& value)
{
    static const auto aclCreateBoolArray = GET_OP_API_FUNC(aclCreateBoolArray);
    if (aclCreateBoolArray == nullptr) {
        return nullptr;
    }

    auto array = aclCreateBoolArray(value.data(), value.size());
    return array;
}

inline aclBoolArray* ConvertType(const at::ArrayRef<bool>& value)
{
    static const auto aclCreateBoolArray = GET_OP_API_FUNC(aclCreateBoolArray);
    if (aclCreateBoolArray == nullptr) {
        return nullptr;
    }

    auto array = aclCreateBoolArray(value.data(), value.size());
    return array;
}

inline aclTensorList* ConvertType(const at::TensorList& at_tensor_list)
{
    static const auto aclCreateTensorList = GET_OP_API_FUNC(aclCreateTensorList);
    if (aclCreateTensorList == nullptr) {
        return nullptr;
    }

    std::vector<const aclTensor*> tensor_list(at_tensor_list.size());
    for (size_t i = 0; i < at_tensor_list.size(); i++) {
        tensor_list[i] = ConvertType(at_tensor_list[i]);
    }
    auto acl_tensor_list = aclCreateTensorList(tensor_list.data(), tensor_list.size());
    return acl_tensor_list;
}

inline aclTensor* ConvertType(const c10::optional<at::Tensor>& opt_tensor)
{
    if (opt_tensor.has_value() && opt_tensor.value().defined()) {
        return ConvertType(opt_tensor.value());
    }
    return nullptr;
}

inline aclIntArray* ConvertType(const c10::optional<at::IntArrayRef>& opt_array)
{
    if (opt_array.has_value()) {
        return ConvertType(opt_array.value());
    }
    return nullptr;
}

inline aclScalar* ConvertType(const c10::optional<at::Scalar>& opt_scalar)
{
    if (opt_scalar.has_value()) {
        return ConvertType(opt_scalar.value());
    }
    return nullptr;
}

inline aclDataType ConvertType(const at::ScalarType scalarType)
{
    return kATenScalarTypeToAclDataTypeTable[static_cast<int64_t>(scalarType)];
}

template <typename T>
T ConvertType(T value)
{
    return value;
}

template <typename Tuple, size_t... I>
auto ConvertToOpApiFunc(const Tuple& params, void* opApiAddr, std::index_sequence<I...>)
{
    using OpApiFunc = int (*)(typename std::decay<decltype(std::get<I>(params))>::type...);
    auto func = reinterpret_cast<OpApiFunc>(opApiAddr);
    return func;
}

template <typename Tuple>
auto ConvertToOpApiFunc(const Tuple& params, void* opApiAddr)
{
    static constexpr auto size = std::tuple_size<Tuple>::value;
    return ConvertToOpApiFunc(params, opApiAddr, std::make_index_sequence<size>{});
}

template <typename... Ts>
constexpr auto ConvertTypes(Ts&... args)
{
    return std::make_tuple(ConvertType(args)...);
}

template <typename Function, typename Tuple, size_t... I>
auto call(Function f, Tuple t, std::index_sequence<I...>)
{
    return f(std::get<I>(t)...);
}

template <typename Function, typename Tuple>
auto call(Function f, Tuple t)
{
    static constexpr auto size = std::tuple_size<Tuple>::value;
    return call(f, t, std::make_index_sequence<size>{});
}

inline aclDataType MapAtToAclDataType(at::ScalarType scalar_type)
{
    switch (scalar_type) {
        case at::kFloat:
            return ACL_FLOAT;
        case at::kHalf:
            return ACL_FLOAT16;
        case at::kInt:
            return ACL_INT32;
        case at::kLong:
            return ACL_INT64;
        case at::kDouble:
            return ACL_DOUBLE;
        case at::kByte:
            return ACL_UINT8;
        case at::kChar:
            return ACL_INT8;
        case at::kShort:
            return ACL_INT16;
        case at::kBool:
            return ACL_BOOL;
        case at::kBFloat16:
            return ACL_BF16;
        case at::kComplexFloat:
            return ACL_COMPLEX64;
        case at::kComplexDouble:
            return ACL_COMPLEX128;
        case at::kComplexHalf:
            return ACL_COMPLEX32;
        default:
            return ACL_DT_UNDEFINED;
    }
}
inline Mki::TensorDType ConvertToMkiDType(const at::ScalarType scalarType)
{
    int acl_dtype = int(MapAtToAclDataType(scalarType));
    return static_cast<Mki::TensorDType>(acl_dtype);
}

struct MkiConvertInput {
    const at::Tensor& tensor;
};

inline Mki::Tensor ConvertType(const MkiConvertInput& mkiInput)
{
    at::Tensor at_tensor = mkiInput.tensor;
    TORCH_CHECK(at_tensor.defined(), "at_tensor not defined!");
    TORCH_CHECK(torch_npu::utils::is_npu(at_tensor), "Expected NPU tensor");
    TORCH_CHECK(at_tensor.is_contiguous(), "Expected tensor is contiguous");

    Mki::Tensor mki_tensor;

    mki_tensor.desc.dtype = ConvertToMkiDType(at_tensor.scalar_type());
    mki_tensor.desc.format = Mki::TENSOR_FORMAT_ND;

    auto at_sizes = at_tensor.sizes();
    for (int64_t s : at_sizes) {
        mki_tensor.desc.dims.push_back(s);
    }

    auto at_strides = at_tensor.strides();
    for (int64_t s : at_strides) {
        mki_tensor.desc.strides.push_back(s);
    }

    mki_tensor.desc.offset = at_tensor.storage_offset() * at_tensor.itemsize();
    mki_tensor.data = const_cast<void*>(at_tensor.storage().data());
    mki_tensor.dataSize = at_tensor.nbytes();

    return mki_tensor;
}

inline aclTensor* CreateAclTensorFromAtTensor(const at::Tensor& at_tensor)
{
    // 非连续输入时 .contiguous() 产生的新张量若为局部变量，函数返回即析构释放
    // 设备存储，返回的 aclTensor* 持有悬垂指针构成 use-after-free（issue #120）。
    // 一次算子执行前会连续转换多个张量（如 convolve 的 signal/kernel/output），
    // 单个 static 槽位会被后续调用覆盖导致前序指针再次悬垂；改用 thread_local
    // 容器为每次调用持有独立张量：无跨调用覆盖、无别名、无线程间数据竞态，
    // 生命周期延续至该线程后续调用（覆盖 EXEC_FUNC 同步执行窗口）。
    thread_local std::vector<at::Tensor> tensorContiguousHolder;
    tensorContiguousHolder.push_back(at_tensor.is_contiguous() ? at_tensor : at_tensor.contiguous());
    const at::Tensor& tensor_contiguous = tensorContiguousHolder.back();

    auto shape = tensor_contiguous.sizes().vec();
    auto strides = tensor_contiguous.strides().vec();
    int64_t dim_num = static_cast<int64_t>(shape.size());
    int64_t offset = 0;
    aclDataType acl_data_type = MapAtToAclDataType(at_tensor.scalar_type());

    void* device_ptr = const_cast<void*>(tensor_contiguous.storage().data());

    // 4. 获取函数指针 (注意：aclCreateTensor 是 OP_API 下的接口)
    static const auto aclCreateTensorFunc = GET_OP_API_FUNC(aclCreateTensor);
    if (aclCreateTensorFunc == nullptr) {
        AT_ERROR("Failed to get aclCreateTensor function pointer.");
        return nullptr;
    }

    aclTensor* tensor = aclCreateTensorFunc(shape.data(),   // 1. View Shape
                                            dim_num,        // 2. View Dim Num
                                            acl_data_type,  // 3. DataType
                                            strides.data(), // 4. Strides
                                            offset,         // 5. Offset
                                            ACL_FORMAT_ND,  // 6. Format (固定使用 ND)
                                            shape.data(), // 7. Storage Shape (关键：传原始 shape，满足 > 1 的校验)
                                            dim_num,   // 8. Storage Dim Num
                                            device_ptr // 9. Data Addr
    );

    return tensor;
}

uint64_t CalcHashId();

// 通过so文件函数名调用
#define EXEC_FUNC_NAME(ops_api_name, ...)                                                               \
    do {                                                                                                \
        static const auto opApiFuncAddr = GetAsdSipApiFuncAddr(#ops_api_name);                          \
        TORCH_CHECK(opApiFuncAddr != nullptr, #ops_api_name, " not found.");                            \
        auto converted_params = ConvertTypes(__VA_ARGS__);                                              \
        auto acl_call = [converted_params]() -> int {                                                   \
            static auto opsFunc = ConvertToOpApiFunc(converted_params, opApiFuncAddr);                  \
            auto opsStats = call(opsFunc, converted_params);                                            \
            TORCH_CHECK(opsStats == 0, "call " #ops_api_name " failed, detail:", aclGetRecentErrMsg()); \
            return opsStats;                                                                            \
        };                                                                                              \
        at_npu::native::OpCommand cmd;                                                                  \
        cmd.Name(#ops_api_name);                                                                        \
        cmd.SetCustomHandler(acl_call);                                                                 \
        cmd.Run();                                                                                      \
    } while (false)

#define EXEC_FUNC(ops_api, ...)                                                                    \
    do {                                                                                           \
        auto converted_params = ConvertTypes(__VA_ARGS__);                                         \
        auto acl_call = [converted_params]() -> int {                                              \
            auto opsStats = call(ops_api, converted_params);                                       \
            TORCH_CHECK(opsStats == 0, "call " #ops_api " failed, detail:", aclGetRecentErrMsg()); \
            return opsStats;                                                                       \
        };                                                                                         \
        at_npu::native::OpCommand cmd;                                                             \
        cmd.Name(#ops_api);                                                                        \
        cmd.SetCustomHandler(acl_call);                                                            \
        cmd.Run();                                                                                 \
    } while (false)

#define EXEC_BLAS_FUNC(ops_api, makePlanFunc, planParam, ...)                                                     \
    do {                                                                                                          \
        auto sip_stream = c10_npu::getCurrentNPUStream().stream(false);                                           \
        AsdSip::asdBlasHandle handle = op_api::getBlasHandle(#ops_api, planParam, makePlanFunc);                  \
        size_t workspace_size = 0;                                                                                \
        void* workspace_addr = nullptr;                                                                           \
        at::Tensor workspace_tensor;                                                                              \
        AsdSip::asdBlasGetWorkspaceSize(handle, workspace_size);                                                  \
        if (workspace_size > 0) {                                                                                 \
            workspace_tensor = at_npu::native::allocate_workspace(static_cast<long>(workspace_size), sip_stream); \
            workspace_addr = const_cast<void*>(workspace_tensor.storage().data());                                \
        }                                                                                                         \
        AsdSip::asdBlasSetWorkspace(handle, workspace_addr);                                                      \
        AsdSip::asdBlasSetStream(handle, sip_stream);                                                             \
        EXEC_FUNC(ops_api, handle, __VA_ARGS__);                                                                  \
    } while (false)

#define EXEC_FFT_FUNC(ops_api, fftParam, ...)                                                                     \
    do {                                                                                                          \
        auto sip_stream = c10_npu::getCurrentNPUStream().stream(false);                                           \
        AsdSip::asdFftHandle handle = op_api::getFftHandle(fftParam);                                             \
        size_t workspace_size = 0;                                                                                \
        AsdSip::asdFftGetWorkspaceSize(handle, workspace_size);                                                   \
        void* workspace_addr = nullptr;                                                                           \
        at::Tensor workspace_tensor;                                                                              \
        if (workspace_size > 0) {                                                                                 \
            workspace_tensor = at_npu::native::allocate_workspace(static_cast<long>(workspace_size), sip_stream); \
            workspace_addr = const_cast<void*>(workspace_tensor.storage().data());                                \
        }                                                                                                         \
        AsdSip::asdFftSetWorkspace(handle, workspace_addr);                                                       \
        AsdSip::asdFftSetStream(handle, sip_stream);                                                              \
        EXEC_FUNC(ops_api, handle, __VA_ARGS__);                                                                  \
    } while (false)
#endif // PYTORCH_NPU_HELPER_HPP_
