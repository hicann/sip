/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// ===========================================================================
// Host entry: FFT C2C 2D over axes (1,2) of [b1, fftN1, fftN2] = [12000,
// 500, 50] complex64 — FUSED AIC-GEMM + AIV-SIMT implementation (example 90).
//
// Usage:
//   90_ascend950_fft_c2c_2d_fused [fftN1] [direction] [device_id] [data_dir]
//                                 [--benchmark] [--perf]
//     fftN1:     500 (default; the operator is specialized to 500x50)
//     direction: -1 (forward, default) or 1 (inverse, unnormalized)
//     device_id: default 7
//     data_dir:  directory for input/golden/output bin files (default: data)
//     --benchmark: random input, skip verification
//     --perf: 10 warmup + 10 measured iterations with aclrtEvent timing
// ===========================================================================

#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <acl/acl.h>
#include <tiling/platform/platform_ascendc.h>

#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/status.hpp"
#include "fft_kernel.h"
#include "fft_plan.h"
#include "fft_shift_kernel.h"
#include "fft_types.h"
#include "helper.hpp"

using namespace Fft;

struct FftOptions {
    static constexpr auto HELPER = "Usage: 90_ascend950_fft_c2c_2d_fused [fftN1] [direction] [device_id] [data_dir] "
                                   "[--benchmark] [--perf] [--roundtrip]\n"
                                   "  fftN1: 500 (default; middle-dim signal length, specialized to [12000, 500, 50])\n"
                                   "  direction: -1 (forward, default) or 1 (inverse, unnormalized)\n"
                                   "  device_id: default 7\n"
                                   "  data_dir: directory for input/golden/output bin files (default: data)\n"
                                   "  --benchmark: skip verification for clean profiling.\n"
                                   "  --perf: 10 warmup + 10 measured iterations with timing.\n"
                                   "  --roundtrip: reproduce fft_demo.cu loop (RO x CH loop, forward+inverse,\n"
                                   "               no fftshift / no 1/N); prints Time / Total FFTs / FFTs/sec /\n"
                                   "               Max diff / PASSED.\n";

    int64_t b1{B1_DEFAULT};
    int64_t fftN1{FFT_N1_DEFAULT};
    int64_t fftN2{FFT_N2_DEFAULT};
    int32_t direction{-1};
    int32_t deviceId{7};
    std::string dataDir{"data"};
    bool benchmark{false};
    bool perf{false};
    bool roundtrip{false};

    int Parse(int argc, const char** argv)
    {
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--benchmark" || arg == "-b")
                benchmark = true;
            if (arg == "--perf" || arg == "-p") {
                perf = true;
                benchmark = true;
            }
            if (arg == "--roundtrip")
                roundtrip = true;
        }

        std::vector<std::string> posArgs;
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--benchmark" || arg == "-b" || arg == "--perf" || arg == "-p" || arg == "--roundtrip")
                continue;
            posArgs.push_back(arg);
        }

        if (posArgs.size() >= 1)
            fftN1 = std::atoll(posArgs[0].c_str());
        if (posArgs.size() >= 2)
            direction = std::atoi(posArgs[1].c_str());
        if (posArgs.size() >= 3)
            deviceId = std::atoi(posArgs[2].c_str());
        if (posArgs.size() >= 4)
            dataDir = posArgs[3];

        if (fftN1 != FFT_N1_DEFAULT || fftN2 != FFT_N2_DEFAULT) {
            std::cerr << "Warning: fftN1 x fftN2 = " << fftN1 << " x " << fftN2
                      << " (this operator is specialized for 500 x 50)" << std::endl;
        }
        if (!roundtrip && direction != -1 && direction != 1) {
            std::cerr << "Error: direction must be -1 or 1" << std::endl;
            return -1;
        }
        return 0;
    }
};

static std::string ShapeTag(const FftOptions& options)
{
    return std::to_string(options.b1) + "x" + std::to_string(options.fftN1) + "x" + std::to_string(options.fftN2);
}

static bool ReadBinFile(const std::string& path, void* buffer, size_t expectedBytes)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::cerr << "Cannot open: " << path << std::endl;
        return false;
    }
    f.read(static_cast<char*>(buffer), static_cast<std::streamsize>(expectedBytes));
    return static_cast<size_t>(f.gcount()) == expectedBytes;
}

static bool WriteBinFile(const std::string& path, const void* buffer, size_t bytes)
{
    std::ofstream f(path, std::ios::binary);
    if (!f)
        return false;
    f.write(static_cast<const char*>(buffer), static_cast<std::streamsize>(bytes));
    return f.good();
}

static void Cleanup(aclrtStream stream, int32_t deviceId)
{
    if (stream != nullptr)
        ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(deviceId));
    ACL_CHECK(aclFinalize());
}

static void Run(const FftOptions& options)
{
    aclrtStream stream{nullptr};
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(options.deviceId));
    ACL_CHECK(aclrtCreateStream(&stream));

    int64_t b1 = options.b1;
    int64_t fftN1 = options.fftN1;
    int64_t fftN2 = options.fftN2;
    int32_t direction = options.direction;

    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    int32_t coreNum = static_cast<int32_t>(platform->GetCoreNumAic());
    std::cout << "Block (AIC) core count: " << coreNum << " -> " << coreNum * 2 << " AIVs (1 AIC + 2 AIV per block)"
              << std::endl;

    std::string tag = ShapeTag(options) + "_" + std::to_string(direction);
    std::string inputPath = options.dataDir + "/input_" + tag + ".bin";
    std::string goldenPath = options.dataDir + "/golden_" + tag + ".bin";
    std::string outputPath = options.dataDir + "/output_" + tag + ".bin";

    int64_t inputFloats = b1 * fftN1 * fftN2 * 2;
    size_t inputSize = inputFloats * sizeof(float);

    std::cout << "Input shape: [" << b1 << ", " << fftN1 << ", " << fftN2 << "]"
              << " (2D FFT on axes (1,2), fused AIC-GEMM + AIV-SIMT)" << std::endl;
    std::cout << "Matrices: " << b1 << "  Input size: " << inputSize / 1024 / 1024 << " MB" << std::endl;

    std::vector<float> hostInput(inputFloats);
    if (options.benchmark) {
        unsigned int seed = 42;
        for (int64_t i = 0; i < inputFloats; ++i) {
            seed = seed * 1103515245u + 12345u;
            hostInput[i] = static_cast<float>(static_cast<int32_t>(seed >> 16) & 0xffff) / 32768.0f - 1.0f;
        }
        std::cout << "[benchmark] Generated random input (" << inputSize << " bytes)" << std::endl;
    } else {
        if (!ReadBinFile(inputPath, hostInput.data(), inputSize)) {
            std::cerr << "Failed to read input. Run gen_data.py first." << std::endl;
            Cleanup(stream, options.deviceId);
            return;
        }
        std::cout << "Loaded input: " << inputPath << std::endl;
    }

    uint8_t* deviceInput{nullptr};
    uint8_t* deviceOutput{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceInput), inputSize, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceOutput), inputSize, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceInput, inputSize, hostInput.data(), inputSize, ACL_MEMCPY_HOST_TO_DEVICE));

    FftPlan plan;
    if (!FftPlanCreate(plan, b1, fftN1, fftN2, direction, options.deviceId, coreNum)) {
        std::cerr << "FftPlanCreate failed" << std::endl;
        ACL_CHECK(aclrtFree(deviceInput));
        ACL_CHECK(aclrtFree(deviceOutput));
        Cleanup(stream, options.deviceId);
        return;
    }

    if (options.perf) {
        constexpr int WARMUP_ITERS = 10;
        constexpr int MEASURE_ITERS = 10;

        std::cout << "=== Performance Test ===" << std::endl;
        std::cout << "  shape=[" << b1 << "," << fftN1 << "," << fftN2 << "]"
                  << " blocks=" << coreNum << " (AIC GEMM + AIV SIMT fused)" << std::endl;

        std::cout << "Warmup (" << WARMUP_ITERS << " iters)..." << std::flush;
        for (int i = 0; i < WARMUP_ITERS; ++i) {
            FftExec2D(plan, deviceInput, deviceOutput, stream, coreNum);
        }
        std::cout << " done." << std::endl;

        aclrtEvent startEvent{nullptr}, endEvent{nullptr};
        ACL_CHECK(aclrtCreateEvent(&startEvent));
        ACL_CHECK(aclrtCreateEvent(&endEvent));

        ACL_CHECK(aclrtRecordEvent(startEvent, stream));
        for (int i = 0; i < MEASURE_ITERS; ++i) {
            FftExec2D(plan, deviceInput, deviceOutput, stream, coreNum);
        }
        ACL_CHECK(aclrtRecordEvent(endEvent, stream));
        ACL_CHECK(aclrtSynchronizeStream(stream));

        float elapsedMs = 0.0f;
        ACL_CHECK(aclrtEventElapsedTime(&elapsedMs, startEvent, endEvent));
        float avgMs = elapsedMs / static_cast<float>(MEASURE_ITERS);

        std::cout << "=== Results ===" << std::endl;
        std::cout << "  Total time (" << MEASURE_ITERS << " iters): " << elapsedMs << " ms" << std::endl;
        std::cout << "  Average latency: " << avgMs << " ms (" << avgMs * 1000.0f << " us)" << std::endl;
        std::cout << "  Effective bandwidth: " << (2.0 * inputSize / 1e9) / (avgMs / 1e3) << " GB/s" << std::endl;
        // Each iteration processes b1 independent 2D FFTs of shape [fftN1, fftN2],
        // so throughput in [500, 50] FFT-2D per second is b1 / (avgMs / 1000).
        double fft2dPerSec = static_cast<double>(b1) * 1000.0 / static_cast<double>(avgMs);
        std::cout << "  FFT-2D [" << fftN1 << ", " << fftN2 << "] throughput: " << fft2dPerSec << " FFTs/s"
                  << std::endl;

        ACL_CHECK(aclrtDestroyEvent(startEvent));
        ACL_CHECK(aclrtDestroyEvent(endEvent));

        FftPlanDestroy(plan);
        ACL_CHECK(aclrtFree(deviceInput));
        ACL_CHECK(aclrtFree(deviceOutput));
        Cleanup(stream, options.deviceId);
        return;
    }

    std::cout << "Starting 2D FFT (fused AIC + AIV)..." << std::endl;
    FftExec2D(plan, deviceInput, deviceOutput, stream, coreNum);
    std::cout << "2D FFT complete." << std::endl;

    if (options.benchmark) {
        std::cout << "[benchmark] Skipped verification." << std::endl;
    } else {
        std::vector<float> hostOutput(inputFloats);
        ACL_CHECK(aclrtMemcpy(hostOutput.data(), inputSize, deviceOutput, inputSize, ACL_MEMCPY_DEVICE_TO_HOST));

        std::vector<float> hostGolden(inputFloats);
        bool hasGolden = ReadBinFile(goldenPath, hostGolden.data(), inputSize);

        if (hasGolden) {
            double maxAbsErr = 0.0, sumAbsErr = 0.0, sumGoldMag = 0.0, sumSqErr = 0.0;
            int64_t totalChecked = inputFloats / 2;
            for (int64_t i = 0; i < totalChecked; i++) {
                float outR = hostOutput[i * 2], outI = hostOutput[i * 2 + 1];
                float gR = hostGolden[i * 2], gI = hostGolden[i * 2 + 1];
                double absErr = std::sqrt((double)(outR - gR) * (outR - gR) + (double)(outI - gI) * (outI - gI));
                double goldMag = std::sqrt((double)gR * gR + (double)gI * gI);
                maxAbsErr = std::max(maxAbsErr, absErr);
                sumAbsErr += absErr;
                sumGoldMag += goldMag;
                sumSqErr += absErr * absErr;
            }
            double mare = sumAbsErr / totalChecked;
            double mere = sumAbsErr / (sumGoldMag > 0 ? sumGoldMag : 1.0);
            double rmse = std::sqrt(sumSqErr / totalChecked);

            std::cout << "\n=== Precision Comparison ===" << std::endl;
            std::cout << "Elements: " << totalChecked << std::endl;
            std::cout << "MARE (mean abs err): " << mare << std::endl;
            std::cout << "MERE: " << mere << "  RMSE: " << rmse << std::endl;
            std::cout << "MaxAbsErr: " << maxAbsErr << std::endl;

            std::cout << "\nSample output (first 4 elements):" << std::endl;
            for (int i = 0; i < 4; i++) {
                std::cout << "  npu: (" << hostOutput[i * 2] << ", " << hostOutput[i * 2 + 1] << ")"
                          << "  golden: (" << hostGolden[i * 2] << ", " << hostGolden[i * 2 + 1] << ")" << std::endl;
            }
        }

        if (WriteBinFile(outputPath, hostOutput.data(), inputSize)) {
            std::cout << "\nWrote output: " << outputPath << std::endl;
        }
    }

    FftPlanDestroy(plan);
    ACL_CHECK(aclrtFree(deviceInput));
    ACL_CHECK(aclrtFree(deviceOutput));
    Cleanup(stream, options.deviceId);
}

// ---------------------------------------------------------------------------
// Round-trip work flow, using the fused Ascend950 2D C2C FFT operator plus the
// AIV fftshift / ifftshift kernels:
//
//   for idxCh2 in [0, CH):                     (CH=24)
//       fftshift           (b1 = RO x CH = 12000)
//       forward 2D FFT     (b1 = RO x CH = 12000)
//       inverse 2D FFT     (b1 = RO x CH = 12000)
//       ifftshift + 1/N    (b1 = RO x CH = 12000)
//
// The forward/inverse FFTs are unnormalized and ifftshift applies 1/N
// (N = PE x SPE = 500 x 50), so the full round-trip restores the original
// input; verification compares the output against the original input with a
// 1e-5 gate.
// ---------------------------------------------------------------------------
static void RunRoundTrip(const FftOptions& options)
{
    aclrtStream stream{nullptr};
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(options.deviceId));
    ACL_CHECK(aclrtCreateStream(&stream));

    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    int32_t coreNum = static_cast<int32_t>(platform->GetCoreNumAic());
    int32_t aivCoreNum = static_cast<int32_t>(platform->GetCoreNumAiv());
    std::cout << "Block (AIC) core count: " << coreNum << " -> " << coreNum * 2 << " AIVs (1 AIC + 2 AIV per block)"
              << std::endl;

    // ---- fft_demo.cu 的维度 (RO / PE / CH / SPE) ----
    const int64_t lROSize = 500;                      // RO  readout
    const int64_t lPESize = 500;                      // PE  = fftN1 (column FFT)
    const int64_t lChaNum = 24;                       // CH  channel count
    const int64_t lSPESize = 50;                      // SPE = fftN2 (row FFT)
    const int64_t lSlcSize = lPESize * lSPESize;      // 25000 = one 2D FFT plane
    const int64_t lSlabSize = lSlcSize * lChaNum;     // 600000 = one RO position
    const int64_t totalComplex = lSlabSize * lROSize; // 300000000 complex
    const int64_t totalFloats = totalComplex * 2;

    std::cout << "2D FFT round-trip demo (Ascend950, fftshift -> fwd -> inv -> ifftshift+1/N)\n";
    std::cout << "RO: " << lROSize << " PE: " << lPESize << " CH: " << lChaNum << " SPE: " << lSPESize << "\n";
    std::cout << "FFT size: " << lPESize << " x " << lSPESize << "  Batch: " << lChaNum << "\n";
    std::cout << "Total data: " << static_cast<double>(totalFloats) * sizeof(float) / (1024.0 * 1024 * 1024) << " GB\n";

    // ---- 随机复数输入 (seed 42, 与 --benchmark 相同的 LCG) ----
    std::vector<float> hostInput(static_cast<size_t>(totalFloats));
    {
        unsigned int seed = 42;
        for (int64_t i = 0; i < totalFloats; ++i) {
            seed = seed * 1103515245u + 12345u;
            hostInput[static_cast<size_t>(i)] = static_cast<float>(static_cast<int32_t>(seed >> 16) & 0xffff) /
                                                    32768.0f -
                                                1.0f;
        }
    }
    const size_t totalBytes = static_cast<size_t>(totalFloats) * sizeof(float);

    // ---- 设备缓冲: d_original(基准输入) / d_src(输出) / d_tmpA / d_tmpB ----
    // d_tmpA: fftshift 输出，随后复用为 inverse FFT 输出。
    // d_tmpB: forward FFT 输出，随后作为 inverse FFT 输入。
    uint8_t* d_original = nullptr;
    uint8_t* d_src = nullptr;
    uint8_t* d_tmpA = nullptr;
    uint8_t* d_tmpB = nullptr;
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&d_original), totalBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&d_src), totalBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&d_tmpA), totalBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&d_tmpB), totalBytes, ACL_MEM_MALLOC_HUGE_FIRST));

    // ---- shift tiling (fftshift / ifftshift) ----
    FftShiftTilingData shiftFwdHost{lROSize * lChaNum, static_cast<uint32_t>(aivCoreNum), 0};
    FftShiftTilingData shiftInvHost{lROSize * lChaNum, static_cast<uint32_t>(aivCoreNum), 1};
    uint8_t* dShiftFwd = nullptr;
    uint8_t* dShiftInv = nullptr;
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&dShiftFwd), sizeof(FftShiftTilingData), ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&dShiftInv), sizeof(FftShiftTilingData), ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(dShiftFwd, sizeof(FftShiftTilingData), &shiftFwdHost, sizeof(FftShiftTilingData),
                          ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(dShiftInv, sizeof(FftShiftTilingData), &shiftInvHost, sizeof(FftShiftTilingData),
                          ACL_MEMCPY_HOST_TO_DEVICE));

    ACL_CHECK(aclrtMemcpy(d_original, totalBytes, hostInput.data(), totalBytes, ACL_MEMCPY_HOST_TO_DEVICE));

    // ---- 计划: 正向 + 逆向 (b1 = RO x CH = 500 x 24 = 12000) ----
    FftPlan fwd, inv;
    const int64_t b1 = lROSize * lChaNum;

    bool ok = true;
    ok = ok && FftPlanCreate(fwd, b1, lPESize, lSPESize, -1, options.deviceId, coreNum);
    ok = ok && FftPlanCreate(inv, b1, lPESize, lSPESize, +1, options.deviceId, coreNum);
    if (!ok) {
        std::cerr << "FftPlanCreate failed for round-trip" << std::endl;
        FftPlanDestroy(fwd);
        FftPlanDestroy(inv);
        ACL_CHECK(aclrtFree(d_original));
        ACL_CHECK(aclrtFree(d_src));
        ACL_CHECK(aclrtFree(d_tmpA));
        ACL_CHECK(aclrtFree(d_tmpB));
        ACL_CHECK(aclrtFree(dShiftFwd));
        ACL_CHECK(aclrtFree(dShiftInv));
        Cleanup(stream, options.deviceId);
        return;
    }

    const uint32_t aicBlockDim = static_cast<uint32_t>(coreNum);
    const uint32_t aivBlockDim = static_cast<uint32_t>(aivCoreNum);

    aclrtEvent startEvent{nullptr}, stopEvent{nullptr};
    ACL_CHECK(aclrtCreateEvent(&startEvent));
    ACL_CHECK(aclrtCreateEvent(&stopEvent));

    ACL_CHECK(aclrtRecordEvent(startEvent, stream));

    for (int64_t idxCh2 = 0; idxCh2 < lChaNum; ++idxCh2) {
        // fftshift -> forward FFT -> inverse FFT -> ifftshift(+1/N)
        FftShift2DKernelLaunch(d_original, d_tmpA, dShiftFwd, aivBlockDim, stream);
        FftC2C2DFusedKernelLaunch(d_tmpA, d_tmpB, fwd.dCoeffs, fwd.dTiling, aicBlockDim, stream);
        FftC2C2DFusedKernelLaunch(d_tmpB, d_tmpA, inv.dCoeffs, inv.dTiling, aicBlockDim, stream);
        FftShift2DKernelLaunch(d_tmpA, d_src, dShiftInv, aivBlockDim, stream);
    }

    ACL_CHECK(aclrtRecordEvent(stopEvent, stream));
    ACL_CHECK(aclrtSynchronizeStream(stream));

    float elapsedMs = 0.0f;
    ACL_CHECK(aclrtEventElapsedTime(&elapsedMs, startEvent, stopEvent));
    const double elapsed = elapsedMs / 1000.0;

    // ---- 精度统计: round-trip 输出 vs 原始输入 ----
    std::vector<float> hostOutput(static_cast<size_t>(totalFloats));
    ACL_CHECK(aclrtMemcpy(hostOutput.data(), totalBytes, d_src, totalBytes, ACL_MEMCPY_DEVICE_TO_HOST));

    double maxDiff = 0.0;
    for (int64_t i = 0; i < totalComplex; ++i) {
        const double refR = static_cast<double>(hostInput[static_cast<size_t>(2 * i)]);
        const double refI = static_cast<double>(hostInput[static_cast<size_t>(2 * i + 1)]);
        const double dR = std::fabs(static_cast<double>(hostOutput[static_cast<size_t>(2 * i)]) - refR);
        const double dI = std::fabs(static_cast<double>(hostOutput[static_cast<size_t>(2 * i + 1)]) - refI);
        const double localMax = std::fmax(dR, dI);
        if (localMax > maxDiff)
            maxDiff = localMax;
    }
    const bool passed = maxDiff < 1e-5;

    // ---- 统计打印 (对齐 fft_demo.cu) ----
    const int64_t totalFFTs = lROSize * lChaNum * lChaNum * 2;
    std::cout << "\n========================================\n";
    std::cout << "Time: " << elapsed << " s\n";
    std::cout << "Total FFTs: " << totalFFTs << " (= RO×CH²×2 = " << lROSize << "×" << lChaNum << "²×2)\n";
    std::cout << "FFTs/sec: " << std::fixed << std::setprecision(0)
              << (elapsed > 0.0 ? static_cast<double>(totalFFTs) / elapsed : 0.0) << "\n";
    std::cout << "Max diff: " << std::setprecision(6) << maxDiff << "\n";
    std::cout << "Test: " << (passed ? "PASSED" : "FAILED") << "\n";
    std::cout << "========================================\n";

    FftPlanDestroy(fwd);
    FftPlanDestroy(inv);
    ACL_CHECK(aclrtDestroyEvent(startEvent));
    ACL_CHECK(aclrtDestroyEvent(stopEvent));
    ACL_CHECK(aclrtFree(d_original));
    ACL_CHECK(aclrtFree(d_src));
    ACL_CHECK(aclrtFree(d_tmpA));
    ACL_CHECK(aclrtFree(d_tmpB));
    ACL_CHECK(aclrtFree(dShiftFwd));
    ACL_CHECK(aclrtFree(dShiftInv));
    Cleanup(stream, options.deviceId);
}

int main(int argc, const char** argv)
{
    FftOptions options;
    if (options.Parse(argc, argv) != 0) {
        std::cerr << FftOptions::HELPER;
        return 1;
    }
    if (options.roundtrip) {
        RunRoundTrip(options);
        return 0;
    }
    Run(options);
    return 0;
}
