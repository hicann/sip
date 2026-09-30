/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// ops-fft 后端 A/B 验证样例: 同一输入分别以 internal 与 ops-fft 后端执行 C2C 1D FFT,
// 逐位比对输出并给出耗时; ops-fft 为强制请求, 能执行到对比阶段即已生效
// 用法: ./example [n] [batch]   (默认 8192 4; 950 请用混合基形状如 15000)

#include <acl/acl.h>
#include "aclnn/acl_meta.h"
#include "asdsip.h"
#include <chrono>
#include <cstdarg>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

using c64 = std::complex<float>;
using namespace AsdSip;

// [AB] 输出统一走此接口并立即刷出, 避免与 slog 输出交错
static void AbLog(const char* fmt, ...)
{
    va_list args;
    va_start(args, fmt);
    vprintf(fmt, args);
    va_end(args);
    fflush(stdout);
}

#define AB_CHECK(cond, msg)                                    \
    do {                                                       \
        if (!(cond)) {                                         \
            AbLog("[AB][FAIL] %s (line %d)\n", msg, __LINE__); \
            return false;                                      \
        }                                                      \
    } while (0)

#define AB_CHECK_MAIN(cond, msg)                               \
    do {                                                       \
        if (!(cond)) {                                         \
            AbLog("[AB][FAIL] %s (line %d)\n", msg, __LINE__); \
            return -1;                                         \
        }                                                      \
    } while (0)

static bool RunOnce(int n, int batch, const std::vector<c64>& input, std::vector<c64>& output, int warmup, int loops,
                    double& msPerExec)
{
    asdFftHandle handle;
    AB_CHECK(asdFftCreate(handle) == ErrorType::ACL_SUCCESS, "asdFftCreate");
    AB_CHECK(asdFftMakePlan1D(handle, n, asdFftType::ASCEND_FFT_C2C, asdFftDirection::ASCEND_FFT_FORWARD, batch) ==
                 ErrorType::ACL_SUCCESS,
             "asdFftMakePlan1D");
    size_t wsSize = 0;
    asdFftGetWorkspaceSize(handle, wsSize);
    void* wsDev = nullptr;
    if (wsSize > 0) {
        AB_CHECK(aclrtMalloc(&wsDev, (int64_t)wsSize, ACL_MEM_MALLOC_HUGE_FIRST) == ::ACL_SUCCESS, "malloc workspace");
    }
    asdFftSetWorkspace(handle, (uint8_t*)wsDev);
    aclrtStream stream = nullptr;
    AB_CHECK(aclrtCreateStream(&stream) == ::ACL_SUCCESS, "create stream");
    asdFftSetStream(handle, stream);

    size_t bytes = input.size() * sizeof(c64);
    const int64_t shape[2] = {batch, n};
    const int64_t strides[2] = {n, 1};
    void* inDev = nullptr;
    void* outDev = nullptr;
    AB_CHECK(aclrtMalloc(&inDev, (int64_t)bytes, ACL_MEM_MALLOC_HUGE_FIRST) == ::ACL_SUCCESS, "malloc input");
    AB_CHECK(aclrtMalloc(&outDev, (int64_t)bytes, ACL_MEM_MALLOC_HUGE_FIRST) == ::ACL_SUCCESS, "malloc output");
    aclrtMemcpy(inDev, bytes, input.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    aclTensor* inT = aclCreateTensor(shape, 2, ACL_COMPLEX64, strides, 0, ACL_FORMAT_ND, shape, 2, inDev);
    aclTensor* outT = aclCreateTensor(shape, 2, ACL_COMPLEX64, strides, 0, ACL_FORMAT_ND, shape, 2, outDev);

    for (int i = 0; i < warmup; i++) {
        asdFftExecC2C(handle, inT, outT);
    }
    aclrtSynchronizeStream(stream);
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < loops; i++) {
        AB_CHECK(asdFftExecC2C(handle, inT, outT) == ErrorType::ACL_SUCCESS, "asdFftExecC2C");
    }
    aclrtSynchronizeStream(stream);
    msPerExec = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / loops;

    output.resize(input.size());
    aclrtMemcpy(output.data(), bytes, outDev, bytes, ACL_MEMCPY_DEVICE_TO_HOST);

    aclDestroyTensor(inT);
    aclDestroyTensor(outT);
    aclrtFree(inDev);
    aclrtFree(outDev);
    if (wsDev != nullptr) {
        aclrtFree(wsDev);
    }
    aclrtDestroyStream(stream);
    asdFftDestroy(handle);
    return true;
}

int main(int argc, char** argv)
{
    int n = argc > 1 ? std::atoi(argv[1]) : 8192;
    int batch = argc > 2 ? std::atoi(argv[2]) : 4;
    AbLog("[AB] C2C 1D FFT, n=%d batch=%d (910B: n=2^k 且 256<n<32768; 950: 混合基如 15000)\n", n, batch);

    AB_CHECK_MAIN(aclInit(nullptr) == ::ACL_SUCCESS, "aclInit");
    // 优先从环境变量 ASDSIP_DEVICE_ID 获取实际 device id(build.sh 按 CI 映射卡注入), 未设置时默认 0
    int deviceId = 0;
    const char* envDeviceId = std::getenv("ASDSIP_DEVICE_ID");
    if (envDeviceId != nullptr) {
        deviceId = std::atoi(envDeviceId);
    }
    AB_CHECK_MAIN(aclrtSetDevice(deviceId) == ::ACL_SUCCESS, "aclrtSetDevice");

    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<c64> input((size_t)n * batch);
    for (auto& v : input) {
        v = c64(dist(rng), dist(rng));
    }

    std::vector<c64> outInternal;
    std::vector<c64> outOpsFft;
    double msInternal = 0.0;
    double msOpsFft = 0.0;
    const int warmup = 10;
    const int loops = 100;

    try {
        AbLog("[AB] backend=internal ...\n");
        AB_CHECK_MAIN(RunOnce(n, batch, input, outInternal, warmup, loops, msInternal), "internal run");
        AbLog("[AB] internal: %.3f ms/exec\n", msInternal);

        // 同进程切换到 ops-fft 后端(后端开关为进程级环境变量, 每次 exec 实时读取)
        AB_CHECK_MAIN(setenv("SIP_FFT_BACKEND", "ops-fft", 1) == 0, "setenv");
        AbLog("[AB] backend=ops-fft ...\n");
        if (!RunOnce(n, batch, input, outOpsFft, warmup, loops, msOpsFft)) {
            AbLog("[AB][FAIL] ops-fft 后端不可用, asdFftExec 已报错中止(严格语义, 不回退):\n");
            AbLog("[AB]       当前形状未命中覆盖路径(见 README 路由表), 或本安装包为 internal-only 构建。\n");
            return -1;
        }
        unsetenv("SIP_FFT_BACKEND");
        AbLog("[AB] ops-fft: %.3f ms/exec\n", msOpsFft);
    } catch (const std::exception& e) {
        AbLog("[AB][FAIL] %s (形状不被当前平台支持?)\n", e.what());
        return -1;
    } catch (...) {
        AbLog("[AB][FAIL] unknown exception (形状不被当前平台支持?)\n");
        return -1;
    }

    bool same = outInternal.size() == outOpsFft.size() &&
                std::memcmp(outInternal.data(), outOpsFft.data(), outInternal.size() * sizeof(c64)) == 0;
    AbLog("[AB] 双后端输出%s一致 (internal %.3f ms/exec, ops-fft %.3f ms/exec)\n", same ? "逐位" : "不", msInternal,
          msOpsFft);
    AbLog("[AB] 判读: 能执行到本行即 ops-fft 已生效, 日志 'run via ops-fft kernel ... success' 为直调命中证据。\n");

    aclrtResetDevice(0);
    aclFinalize();
    return same ? 0 : 1;
}
