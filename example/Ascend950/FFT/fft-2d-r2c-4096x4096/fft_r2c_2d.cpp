#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

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

struct FftR2C2DOptions {
    static constexpr auto HELPER =
        "Usage: fft_2d_r2c_4096x4096 fftN [batch] [device_id] [data_dir] [--perf]\n"
        "  fftN:     4096 (fixed)\n"
        "  batch:    1 (fixed for radix-16 2D)\n"
        "  device_id: default 0\n"
        "  data_dir:  directory for input/golden/output bin files (default: data)\n"
        "  --perf:    performance test mode (10 warmup + 10 measured).\n";

    int64_t fftN{4096};
    int32_t batch{1};
    int32_t deviceId{0};
    std::string dataDir{"data"};
    bool perf{false};

    int Parse(int argc, const char** argv)
    {
        if (argc < 2) { std::cerr << HELPER; return -1; }
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--perf" || arg == "-p") perf = true;
        }
        std::vector<std::string> posArgs;
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--perf" || arg == "-p") continue;
            posArgs.push_back(arg);
        }
        if (posArgs.size() >= 1) fftN = std::atoll(posArgs[0].c_str());
        if (posArgs.size() >= 2) batch = std::atoi(posArgs[1].c_str());
        if (posArgs.size() >= 3) deviceId = std::atoi(posArgs[2].c_str());
        if (posArgs.size() >= 4) dataDir = posArgs[3];
        if (fftN != 4096) { std::cerr << "Error: only fftN=4096 supported" << std::endl; return -1; }
        if (batch != 1) { std::cerr << "Error: only batch=1 supported" << std::endl; return -1; }
        return 0;
    }
};

int main(int argc, const char** argv)
{
    FftR2C2DOptions opts;
    if (opts.Parse(argc, argv) != 0) return 1;

    std::cout << "FFT R2C 2D (Pass1 R2C-1D + Pass2 C2C-1D, radix-16)" << std::endl;
    std::cout << "  fftN=" << opts.fftN << " batch=" << opts.batch
              << " deviceId=" << opts.deviceId << std::endl;

    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(opts.deviceId));

    int64_t fftN = opts.fftN;
    int32_t batch = opts.batch;
    int64_t halfN = fftN / 2 + 1;

    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    int32_t cubeCoreNum = static_cast<int32_t>(platform->GetCoreNumAic());
    std::cout << "  AIC cores: " << cubeCoreNum << std::endl;

    int64_t inputFloats = static_cast<int64_t>(batch) * fftN * fftN;
    int64_t outputFloats = static_cast<int64_t>(batch) * fftN * halfN * 2;
    size_t inputSize = inputFloats * sizeof(float);
    size_t outputSize = outputFloats * sizeof(float);

    uint8_t* dInput = nullptr;
    uint8_t* dOutput = nullptr;
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&dInput), inputSize, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&dOutput), outputSize, ACL_MEM_MALLOC_HUGE_FIRST));

    FftR2C2DPlan plan;
    if (!FftR2C2DPlanCreate(plan, fftN, batch, opts.deviceId, cubeCoreNum)) {
        std::cerr << "FftR2C2DPlanCreate failed" << std::endl;
        aclrtFree(dInput); aclrtFree(dOutput);
        aclrtResetDevice(opts.deviceId); aclFinalize();
        return 1;
    }

    aclrtStream stream;
    ACL_CHECK(aclrtCreateStream(&stream));

    std::string suffix = std::to_string(fftN) + "_" + std::to_string(batch);

    if (opts.perf) {
        std::vector<float> hostInput(inputFloats);
        unsigned int seed = 42;
        for (int64_t i = 0; i < inputFloats; ++i) {
            seed = seed * 1103515245u + 12345u;
            hostInput[i] = static_cast<float>(static_cast<int32_t>(seed >> 16) & 0xffff) / 32768.0f - 1.0f;
        }
        ACL_CHECK(aclrtMemcpy(dInput, inputSize, hostInput.data(), inputSize, ACL_MEMCPY_HOST_TO_DEVICE));

        constexpr int WARMUP_ITERS = 10;
        constexpr int MEASURE_ITERS = 10;
        for (int i = 0; i < WARMUP_ITERS; ++i) FftR2C2DExec(plan, dInput, dOutput, stream, cubeCoreNum);
        ACL_CHECK(aclrtSynchronizeStream(stream));

        aclrtEvent startEvent{nullptr}, endEvent{nullptr};
        ACL_CHECK(aclrtCreateEvent(&startEvent));
        ACL_CHECK(aclrtCreateEvent(&endEvent));
        ACL_CHECK(aclrtRecordEvent(startEvent, stream));
        for (int i = 0; i < MEASURE_ITERS; ++i) FftR2C2DExec(plan, dInput, dOutput, stream, cubeCoreNum);
        ACL_CHECK(aclrtRecordEvent(endEvent, stream));
        ACL_CHECK(aclrtSynchronizeStream(stream));

        float elapsedMs = 0.0f;
        ACL_CHECK(aclrtEventElapsedTime(&elapsedMs, startEvent, endEvent));
        std::cout << "=== Results ===" << std::endl;
        std::cout << "  Total time (" << MEASURE_ITERS << " iters): " << elapsedMs << " ms" << std::endl;
        std::cout << "  Average latency: " << elapsedMs / MEASURE_ITERS << " ms ("
                  << elapsedMs / MEASURE_ITERS * 1000.0f << " us)" << std::endl;

        ACL_CHECK(aclrtDestroyEvent(startEvent));
        ACL_CHECK(aclrtDestroyEvent(endEvent));
        ACL_CHECK(aclrtDestroyStream(stream));
        aclrtFree(dInput); aclrtFree(dOutput);
        FftR2C2DPlanDestroy(plan);
        aclrtResetDevice(opts.deviceId); aclFinalize();
        return 0;
    }

    std::string inputFile = opts.dataDir + "/input_" + suffix + ".bin";
    std::vector<float> hostInput(inputFloats);
    std::ifstream ifs(inputFile, std::ios::binary);
    if (!ifs.is_open()) {
        std::cerr << "Error: cannot open " << inputFile << std::endl;
        std::cerr << "  Run gen_data.py first." << std::endl;
        aclrtFree(dInput); aclrtFree(dOutput);
        aclrtResetDevice(opts.deviceId); aclFinalize();
        return 1;
    }
    ifs.read(reinterpret_cast<char*>(hostInput.data()), inputSize);
    ifs.close();
    ACL_CHECK(aclrtMemcpy(dInput, inputSize, hostInput.data(), inputSize, ACL_MEMCPY_HOST_TO_DEVICE));
    std::cout << "  Loaded input: " << inputSize << " bytes" << std::endl;

    FftR2C2DExec(plan, dInput, dOutput, stream, cubeCoreNum);

    std::vector<float> hostOutput(outputFloats);
    ACL_CHECK(aclrtMemcpy(hostOutput.data(), outputSize, dOutput, outputSize, ACL_MEMCPY_DEVICE_TO_HOST));
    std::string outputFile = opts.dataDir + "/output_" + suffix + ".bin";
    std::ofstream ofs(outputFile, std::ios::binary);
    ofs.write(reinterpret_cast<char*>(hostOutput.data()), outputSize);
    ofs.close();
    std::cout << "  Output saved to " << outputFile << std::endl;

    ACL_CHECK(aclrtDestroyStream(stream));
    aclrtFree(dInput); aclrtFree(dOutput);
    FftR2C2DPlanDestroy(plan);
    aclrtResetDevice(opts.deviceId); aclFinalize();

    std::cout << "FFT R2C 2D completed." << std::endl;
    return 0;
}