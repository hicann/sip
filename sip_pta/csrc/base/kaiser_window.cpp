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
#include <torch/extension.h>
#include <torch_npu/csrc/core/npu/NPUGuard.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include "base_api.h"
#include "pytorch_npu_helper.hpp"

namespace sip_pta {
at::Tensor asdKaiserWindow(int64_t windowLength, bool periodic, double beta, c10::optional<at::ScalarType> dtype,
                           c10::optional<at::Layout> layout, c10::optional<at::Device> device, bool requiresGrad)
{
    TORCH_CHECK(windowLength >= 0 && windowLength <= std::numeric_limits<int32_t>::max(),
                "asd_kaiser_window: window_length must be in [0, 2147483647]");
    const auto outputDtype = dtype.value_or(at::kFloat);
    TORCH_CHECK(outputDtype == at::kHalf || outputDtype == at::kBFloat16 || outputDtype == at::kFloat,
                "asd_kaiser_window: dtype must be float16, bfloat16 or float32");
    TORCH_CHECK(layout.value_or(at::kStrided) == at::kStrided, "asd_kaiser_window: only strided layout is supported");
    TORCH_CHECK(!device.has_value() || device->type() == c10::DeviceType::PrivateUse1,
                "asd_kaiser_window: device must be an NPU");

    auto currentStream = c10_npu::getCurrentNPUStream();
    const auto deviceIndex = device.has_value() && device->has_index() ? device->index() : currentStream.device_index();
    const c10::Device outputDevice(c10::DeviceType::PrivateUse1, deviceIndex);
    c10_npu::OptionalNPUGuard guard;
    if (deviceIndex != currentStream.device_index()) {
        guard.reset_device(outputDevice);
        currentStream = c10_npu::getCurrentNPUStream();
    }
    auto output = at::empty({windowLength}, at::TensorOptions().dtype(outputDtype).device(outputDevice));

    size_t workspaceSize = 0;
    TORCH_CHECK(AsdSip::asdKaiserWindowGetWorkspaceSize(&workspaceSize) == AsdSip::ErrorType::ACL_SUCCESS,
                "asd_kaiser_window: get workspace size failed");
    TORCH_CHECK(workspaceSize == 0, "asd_kaiser_window: expected zero workspace");
    if (windowLength != 0) {
        auto stream = currentStream.stream(false);
        void* workspace = nullptr;
        // Snapshot tensor metadata before enqueueing, using the same conversion and
        // release helpers as EXEC_FUNC. RunOpApiV2 preserves the NPU task queue.
        auto params = ConvertTypes(output, windowLength, periodic, beta, stream, workspace);
        auto handler = [params]() mutable -> int {
            const auto status = call(AsdSip::asdKaiserWindow, params);
            ReleaseConvertTypes(params);
            TORCH_CHECK(status == AsdSip::ErrorType::ACL_SUCCESS,
                        "asd_kaiser_window: kernel launch failed, status: ", status);
            return status;
        };
        // RunOpApiV2 错误契约（torch_npu OpCommand.h 已核实, issue #399）：签名为
        // `static void RunOpApiV2(const string&, const PROC_FUNC&, bool sync = false)`，
        // 以 void 返回、无错误码返回值；失败上报走 handler 内部异常（此处 TORCH_CHECK）
        // 与 NPU 任务队列自身的异步错误回调，故无需接收返回值。
        // handler 在入队期间于调用线程内同步执行（与 EXEC_FUNC 的 SetCustomHandler
        // 同线程模型），core 侧 thread_local 上下文缓存键在该模型下成立。
        at_npu::native::OpCommand::RunOpApiV2("asdKaiserWindow", handler);
    }
    if (requiresGrad) {
        output.set_requires_grad(true);
    }
    return output;
}

TORCH_LIBRARY_FRAGMENT(torch_sip, m)
{
    m.def("asd_kaiser_window(int window_length, bool periodic=True, float beta=12., *, "
          "ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool requires_grad=False) -> Tensor");
}

// A factory has no tensor arguments from which the dispatcher could infer PrivateUse1.
TORCH_LIBRARY_IMPL(torch_sip, BackendSelect, m) { m.impl("asd_kaiser_window", &asdKaiserWindow); }

TORCH_LIBRARY_IMPL(torch_sip, PrivateUse1, m) { m.impl("asd_kaiser_window", &asdKaiserWindow); }
} // namespace sip_pta
