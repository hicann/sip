/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * You may refer to the License for details. You should not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_FFT_KERNEL_STUB_H
#define OPS_FFT_KERNEL_STUB_H

#include <cstdint>
#include <cstdlib>
#include <cstring>

#include "utils/aspb_status.h"

// ops-fft 预编译制品 libcann_ops_fft.so 导出的 kernel host stub (bisheng <<<>>> 生成物):
//   - fft_b: 910B(arch32), tiling 对应 sip FftBTilingData/Fft1DBTilingData
//   - fft_c2c_arch35_mix_multi_core: 950(arch35) mixed-radix, tiling 对应 sip FftAllMixTilingData
// stub 声明为 weak (存件 so 只含单一 SoC 的 arch 产物): 任一存件下均可加载, 缺失侧为 nullptr
extern "C" {
__attribute__((weak)) void fft_b(uint32_t blocks, void *unused, void *stream,
            void *ffts_addr, void *gm_input, void *gm_w, void *gm_t, void *gm_index,
            void *gm_output, void *gm_workspace, void *gm_tiling);
__attribute__((weak)) void fft_c2c_arch35_mix_multi_core(uint32_t blocks, void *unused, void *stream,
            void *gm_input, void *gm_dft_matrix, void *gm_tw_matrix, void *gm_radix_list,
            void *gm_output, void *gm_workspace, void *gm_tiling);
}

// CANN runtime: 硬件同步地址 (kernel 跨核同步用), 原型声明避免引入 acl 头依赖
extern "C" int aclrtGetHardwareSyncAddr(void **addr);

namespace AsdSip {

// 后端开关: SIP_FFT_BACKEND=ops-fft 为强制请求(未适配/缺件即报错, 不回退),
// 未设置或其他值时使用 internal (sip 自身 kernel)
inline bool UseOpsFftKernelBackend()
{
    const char *backend = std::getenv("SIP_FFT_BACKEND");
    return backend != nullptr && std::strcmp(backend, "ops-fft") == 0;
}

// stub 可用性判空 (weak 符号在当前 so 中缺失时为 nullptr)
inline bool OpsFftBStubAvailable()
{
    return fft_b != nullptr;
}

inline bool OpsFftArch35MixStubAvailable()
{
    return fft_c2c_arch35_mix_multi_core != nullptr;
}

class FFTPlan;

// ops-fft 后端统一拦截 (定义于 fft_api.cpp): 请求 ops-fft 时逐 step 校验
// 适配性与 stub 可用性, 任一不满足即报错中止(ACL_ERROR_API_NOT_SUPPORT), 不回退 internal
AspbStatus OpsFftBackendUnifyIntercept(const FFTPlan &plan);

} // namespace AsdSip

#endif // OPS_FFT_KERNEL_STUB_H
