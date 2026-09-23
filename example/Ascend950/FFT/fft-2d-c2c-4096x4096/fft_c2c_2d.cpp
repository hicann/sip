#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
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
#include "fft_tiling.h"
#include "fft_types.h"
#include "helper.hpp"

using namespace Fft;

struct FftOptions {
    static constexpr auto HELPER =
        "Usage: fft_2d_c2c_4096x4096 [direction] [device_id] [data_dir] [--perf]\n"
        "  input:      [4096, 4096] complex64 (FFT2D, both axes)\n"
        "  direction:  -1 (forward, default) or 1 (inverse, backward norm)\n"
        "  device_id:  default 3\n"
        "  data_dir:   directory for input/golden/output bin files (default: data)\n"
        "  --perf:     performance test mode. 10 warmup + 10 measured iterations.\n";

    int32_t direction{-1};
    int32_t deviceId{3};
    std::string dataDir{"data"};
    bool perf{false};

    int Parse(int argc, const char** argv)
    {
        if (argc < 1) {
            std::cerr << HELPER;
            return -1;
        }

        std::vector<std::string> posArgs;
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--perf" || arg == "-p") {
                perf = true;
                continue;
            }
            posArgs.push_back(arg);
        }

        if (posArgs.size() >= 1) direction = std::atoi(posArgs[0].c_str());
        if (posArgs.size() >= 2) deviceId = std::atoi(posArgs[1].c_str());
        if (posArgs.size() >= 3) dataDir = posArgs[2];

        if (direction != -1 && direction != 1) {
            std::cerr << "Error: direction must be -1 or 1, got " << direction << std::endl;
            return -1;
        }
        return 0;
    }
};

int main(int argc, const char** argv)
{
    FftOptions opts;
    if (opts.Parse(argc, argv) != 0) {
        return 1;
    }

    constexpr int64_t kRows = 4096;
    constexpr int64_t kCols = 4096;

    std::cout << "FFT C2C 2D [4096,4096] (transpose-absorbed fused kernel)" << std::endl;
    std::cout << "  direction=" << opts.direction
              << " deviceId=" << opts.deviceId << std::endl;

    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(opts.deviceId));

    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    int32_t cubeCoreNum = static_cast<int32_t>(platform->GetCoreNumAic());
    int32_t vecCoreNum = static_cast<int32_t>(platform->GetCoreNumAiv());
    std::cout << "  AIC cores: " << cubeCoreNum << ", AIV cores: " << vecCoreNum << std::endl;

    // K1 的信号按 4 对齐聚合组分配（每核行数为 4 的倍数；4096/28 核，
    // groupsPerCore=36 余 16 —— rowStartIdx 均为 4 的倍数）。
    FftPlan plan;
    if (!FftPlanCreate(plan, kCols, kRows, opts.direction, opts.deviceId,
                       cubeCoreNum, vecCoreNum)) {
        std::cerr << "FftPlanCreate failed" << std::endl;
        aclFinalize();
        return 1;
    }

    int64_t totalFloats = kRows * kCols * 2;
    size_t dataSize = totalFloats * sizeof(float);

    uint8_t* dInput = nullptr;
    uint8_t* dOutput = nullptr;

    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&dInput), dataSize, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&dOutput), dataSize, ACL_MEM_MALLOC_HUGE_FIRST));

    std::string suffix = "2d_4096x4096_fft2_" + std::to_string(opts.direction);

    if (opts.perf) {
        std::vector<float> hostInput(totalFloats);
        unsigned int seed = 42;
        for (int64_t i = 0; i < totalFloats; ++i) {
            seed = seed * 1103515245u + 12345u;
            hostInput[i] = static_cast<float>(static_cast<int32_t>(seed >> 16) & 0xffff) / 32768.0f - 1.0f;
        }
        ACL_CHECK(aclrtMemcpy(dInput, dataSize, hostInput.data(), dataSize,
                              ACL_MEMCPY_HOST_TO_DEVICE));
        std::cout << "  [perf] Generated random input (" << dataSize << " bytes, no file I/O)" << std::endl;
    } else {
        std::string inputFile = opts.dataDir + "/input_" + suffix + ".bin";
        std::ifstream ifs(inputFile, std::ios::binary);
        if (ifs.is_open()) {
            std::vector<float> hostInput(totalFloats);
            ifs.read(reinterpret_cast<char*>(hostInput.data()), dataSize);
            ifs.close();
            ACL_CHECK(aclrtMemcpy(dInput, dataSize, hostInput.data(), dataSize,
                                  ACL_MEMCPY_HOST_TO_DEVICE));
            std::cout << "  Loaded input from " << inputFile << std::endl;
        } else {
            std::cerr << "  Error: cannot open " << inputFile << std::endl;
            std::cerr << "  Run gen_data.py first" << std::endl;
            aclrtFree(dInput);
            aclrtFree(dOutput);
            aclrtResetDevice(opts.deviceId);
            aclFinalize();
            return 1;
        }
    }

    aclrtStream stream;
    ACL_CHECK(aclrtCreateStream(&stream));

    if (opts.perf) {
        constexpr int WARMUP_ITERS = 10;
        constexpr int MEASURE_ITERS = 10;

        std::cout << "=== Performance Test ===" << std::endl;
        std::cout << "  warmup iters: " << WARMUP_ITERS
                  << "  measure iters: " << MEASURE_ITERS << std::endl;

        std::cout << "Warmup (" << WARMUP_ITERS << " iters)..." << std::flush;
        for (int i = 0; i < WARMUP_ITERS; ++i) {
            FftExec2D(plan, dInput, dOutput, stream, cubeCoreNum, vecCoreNum);
        }
        std::cout << " done." << std::endl;

        aclrtEvent startEvent{nullptr};
        aclrtEvent endEvent{nullptr};
        ACL_CHECK(aclrtCreateEvent(&startEvent));
        ACL_CHECK(aclrtCreateEvent(&endEvent));

        ACL_CHECK(aclrtRecordEvent(startEvent, stream));
        for (int i = 0; i < MEASURE_ITERS; ++i) {
            FftExec2D(plan, dInput, dOutput, stream, cubeCoreNum, vecCoreNum);
        }
        ACL_CHECK(aclrtRecordEvent(endEvent, stream));
        ACL_CHECK(aclrtSynchronizeStream(stream));

        float elapsedMs = 0.0f;
        ACL_CHECK(aclrtEventElapsedTime(&elapsedMs, startEvent, endEvent));
        float avgMs = elapsedMs / static_cast<float>(MEASURE_ITERS);

        std::cout << "=== Results ===" << std::endl;
        std::cout << "  Total time (" << MEASURE_ITERS << " iters): " << elapsedMs << " ms" << std::endl;
        std::cout << "  Average latency: " << avgMs << " ms ("
                  << avgMs * 1000.0f << " us)" << std::endl;

        ACL_CHECK(aclrtDestroyEvent(startEvent));
        ACL_CHECK(aclrtDestroyEvent(endEvent));

        ACL_CHECK(aclrtDestroyStream(stream));
        aclrtFree(dInput);
        aclrtFree(dOutput);
        FftPlanDestroy(plan);
        aclrtResetDevice(opts.deviceId);
        aclFinalize();
        return 0;
    }

    FftExec2D(plan, dInput, dOutput, stream, cubeCoreNum, vecCoreNum);

    std::vector<float> hostOutput(totalFloats);
    ACL_CHECK(aclrtMemcpy(hostOutput.data(), dataSize, dOutput, dataSize,
                          ACL_MEMCPY_DEVICE_TO_HOST));
    std::string outputFile = opts.dataDir + "/output_" + suffix + ".bin";
    std::ofstream ofs(outputFile, std::ios::binary);
    ofs.write(reinterpret_cast<char*>(hostOutput.data()), dataSize);
    ofs.close();
    std::cout << "  Output saved to " << outputFile << std::endl;

    ACL_CHECK(aclrtDestroyStream(stream));
    aclrtFree(dInput);
    aclrtFree(dOutput);
    FftPlanDestroy(plan);

    aclrtResetDevice(opts.deviceId);
    aclFinalize();

    std::cout << std::endl << "FFT C2C axis=0 completed." << std::endl;
    return 0;
}
