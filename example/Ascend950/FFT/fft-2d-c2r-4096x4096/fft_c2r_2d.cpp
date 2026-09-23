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

struct FftC2R2DOptions {
    static constexpr auto HELPER =
        "Usage: fft_2d_c2r_4096x4096 fftN [batch] [device_id] [data_dir] [--benchmark] [--perf] [--stage N]\n"
        "  fftN: 4096 (default) or 2048\n"
        "  batch: 1 (default), up to 16\n"
        "  device_id: default 0\n"
        "  data_dir: directory for input/golden/output bin files (default: data)\n"
        "  --benchmark: skip verification for clean profiling.\n"
        "  --perf: performance test mode. 10 warmup + 10 measured iterations.\n"
        "  --stage N: debug stage execution (0=K1+transpose, 1=K1+transpose, 2=full pipeline)\n"
        "    Note: K2 is fused into K1. stage 0 and 1 both output K1's transposed complex output.\n";

    int64_t fftN{4096};
    int32_t batch{1};
    int32_t deviceId{0};
    std::string dataDir{"data"};
    bool benchmark{false};
    bool perf{false};
    int32_t stage{-1};  // -1 = full pipeline (no debug)

    int Parse(int argc, const char** argv)
    {
        if (argc < 2) { std::cerr << HELPER; return -1; }
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--benchmark" || arg == "-b") benchmark = true;
            if (arg == "--perf" || arg == "-p") { perf = true; benchmark = true; }
            if (arg == "--stage" && i + 1 < argc) { stage = std::atoi(argv[++i]); }
        }
        std::vector<std::string> posArgs;
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--benchmark" || arg == "-b" || arg == "--perf" || arg == "-p") continue;
            if (arg == "--stage") { i++; continue; }
            posArgs.push_back(arg);
        }
        if (posArgs.size() >= 1) fftN = std::atoll(posArgs[0].c_str());
        if (posArgs.size() >= 2) batch = std::atoi(posArgs[1].c_str());
        if (posArgs.size() >= 3) deviceId = std::atoi(posArgs[2].c_str());
        if (posArgs.size() >= 4) dataDir = posArgs[3];
        if (fftN != 4096 && fftN != 2048) {
            std::cerr << "Error: fftN must be 4096 or 2048, got " << fftN << std::endl;
            return -1;
        }
        if (batch < 1 || batch > 16) {
            std::cerr << "Error: batch must be 1..16, got " << batch << std::endl;
            return -1;
        }
        return 0;
    }
};

static bool ReadBinFile(const std::string& path, void* buffer, size_t expectedBytes)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::cerr << "Cannot open file: " << path << std::endl; return false; }
    f.read(static_cast<char*>(buffer), static_cast<std::streamsize>(expectedBytes));
    if (static_cast<size_t>(f.gcount()) != expectedBytes) {
        std::cerr << "File size mismatch: " << path << " expected " << expectedBytes << " got " << f.gcount() << std::endl;
        return false;
    }
    return true;
}

static bool WriteBinFile(const std::string& path, const void* buffer, size_t bytes)
{
    std::ofstream f(path, std::ios::binary);
    if (!f) { std::cerr << "Cannot create file: " << path << std::endl; return false; }
    f.write(static_cast<const char*>(buffer), static_cast<std::streamsize>(bytes));
    return f.good();
}

static void Run(const FftC2R2DOptions& options)
{
    aclrtStream stream{nullptr};

    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(options.deviceId));
    ACL_CHECK(aclrtCreateStream(&stream));

    int64_t fftN = options.fftN;
    int32_t batch = options.batch;

    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    int32_t cubeCoreNum = static_cast<int32_t>(platform->GetCoreNumAic());
    std::cout << "Cube core count: " << cubeCoreNum << std::endl;

    // Input: [batch, fftN, fftN/2+1] complex64 (interleaved [re, im])
    // kernel 内部需要 padHalf（64 倍）补齐，host 侧在 H2D 前做零填充。
    int64_t halfN = fftN / 2 + 1;
    int64_t padHalf = ((halfN + 63) / 64) * 64;
    int64_t inputFloats = batch * fftN * halfN * 2;
    int64_t paddedInputFloats = batch * fftN * padHalf * 2;
    size_t inputSize = inputFloats * sizeof(float);
    size_t paddedInputSize = paddedInputFloats * sizeof(float);

    // Output: [batch, fftN, fftN] float32 (real)
    int64_t outputFloats = batch * fftN * fftN;
    size_t outputSize = outputFloats * sizeof(float);

    std::vector<float> hostInput(inputFloats);
    if (options.benchmark) {
        unsigned int seed = 42;
        for (int64_t i = 0; i < inputFloats; ++i) {
            seed = seed * 1103515245u + 12345u;
            hostInput[i] = static_cast<float>(static_cast<int32_t>(seed >> 16) & 0xffff) / 32768.0f - 1.0f;
        }
        std::cout << "[benchmark] Generated random input (" << inputSize << " bytes)" << std::endl;
    } else {
        std::string inputPath = options.dataDir + "/input_" + std::to_string(fftN) + "_" +
                                std::to_string(batch) + ".bin";
        if (!ReadBinFile(inputPath, hostInput.data(), inputSize)) {
            std::cerr << "Failed to read input. Run gen_data.py first." << std::endl;
            ACL_CHECK(aclrtDestroyStream(stream));
            ACL_CHECK(aclrtResetDevice(options.deviceId));
            ACL_CHECK(aclFinalize());
            return;
        }
        std::cout << "Loaded input: " << inputSize << " bytes" << std::endl;
    }

    uint8_t* deviceInput{nullptr};
    uint8_t* deviceOutput{nullptr};

    // host 侧零填充到 padHalf（2112），补齐 63 列
    std::vector<float> hostInputPadded(paddedInputFloats, 0.0f);
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t r = 0; r < fftN; ++r) {
            int64_t srcOff = (b * fftN + r) * halfN * 2;
            int64_t dstOff = (b * fftN + r) * padHalf * 2;
            std::memcpy(&hostInputPadded[dstOff], &hostInput[srcOff],
                        static_cast<size_t>(halfN * 2 * sizeof(float)));
        }
    }

    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceInput), paddedInputSize, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceOutput), outputSize, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceInput, paddedInputSize, hostInputPadded.data(), paddedInputSize, ACL_MEMCPY_HOST_TO_DEVICE));

    FftC2R2DPlan plan;
    if (!FftC2R2DPlanCreate(plan, fftN, batch, options.deviceId, cubeCoreNum)) {
        std::cerr << "FftC2R2DPlanCreate failed" << std::endl;
        ACL_CHECK(aclrtFree(deviceInput));
        ACL_CHECK(aclrtFree(deviceOutput));
        ACL_CHECK(aclrtDestroyStream(stream));
        ACL_CHECK(aclrtResetDevice(options.deviceId));
        ACL_CHECK(aclFinalize());
        return;
    }

    if (options.perf) {
        constexpr int WARMUP_ITERS = 10;
        constexpr int MEASURE_ITERS = 10;
        std::cout << "=== Performance Test ===" << std::endl;
        std::cout << "  fftN: " << fftN << "  batch: " << batch
                  << "  cubeCores: " << cubeCoreNum << std::endl;

        std::cout << "Warmup (" << WARMUP_ITERS << " iters)..." << std::flush;
        for (int i = 0; i < WARMUP_ITERS; ++i) {
            FftC2R2DExec(plan, deviceInput, deviceOutput, stream, cubeCoreNum);
        }
        std::cout << " done." << std::endl;

        aclrtEvent startEvent{nullptr}, endEvent{nullptr};
        ACL_CHECK(aclrtCreateEvent(&startEvent));
        ACL_CHECK(aclrtCreateEvent(&endEvent));

        ACL_CHECK(aclrtRecordEvent(startEvent, stream));
        for (int i = 0; i < MEASURE_ITERS; ++i) {
            FftC2R2DExec(plan, deviceInput, deviceOutput, stream, cubeCoreNum);
        }
        ACL_CHECK(aclrtRecordEvent(endEvent, stream));
        ACL_CHECK(aclrtSynchronizeStream(stream));

        float elapsedMs = 0.0f;
        ACL_CHECK(aclrtEventElapsedTime(&elapsedMs, startEvent, endEvent));
        float avgMs = elapsedMs / static_cast<float>(MEASURE_ITERS);

        std::cout << "=== Results ===" << std::endl;
        std::cout << "  Total time (" << MEASURE_ITERS << " iters): " << elapsedMs << " ms" << std::endl;
        std::cout << "  Average latency: " << avgMs << " ms (" << avgMs * 1000.0f << " us)" << std::endl;

        ACL_CHECK(aclrtDestroyEvent(startEvent));
        ACL_CHECK(aclrtDestroyEvent(endEvent));

        FftC2R2DPlanDestroy(plan);
        ACL_CHECK(aclrtFree(deviceInput));
        ACL_CHECK(aclrtFree(deviceOutput));
        ACL_CHECK(aclrtDestroyStream(stream));
        ACL_CHECK(aclrtResetDevice(options.deviceId));
        ACL_CHECK(aclFinalize());
        return;
    }

    if (options.stage >= 0) {
        std::cout << "Starting 2D C2R FFT (stage=" << options.stage
                  << (options.stage <= 1 ? ", K1+fused transpose" : ", full pipeline")
                  << ")..." << std::endl;
        FftC2R2DExecStage(plan, deviceInput, deviceOutput, stream, cubeCoreNum, options.stage);
        std::cout << "Stage " << options.stage << " complete." << std::endl;
    } else {
        std::cout << "Starting 2D C2R FFT..." << std::endl;
        FftC2R2DExec(plan, deviceInput, deviceOutput, stream, cubeCoreNum);
        std::cout << "2D C2R FFT complete." << std::endl;
    }

    if (!options.benchmark) {
        std::vector<float> hostOutput(outputFloats);
        ACL_CHECK(aclrtMemcpy(hostOutput.data(), outputSize, deviceOutput, outputSize, ACL_MEMCPY_DEVICE_TO_HOST));

        // C2R 2D output: real [batch, fftN, fftN] float32
        std::cout << "\n=== C2R 2D Output ===" << std::endl;
        std::cout << "  Shape: [1, " << fftN << ", " << fftN << "] real" << std::endl;
        for (int ri = 0; ri < 4; ri++) {
            int64_t row = ri;
            std::cout << "  row " << row << " [0:8]: ";
            for (int c = 0; c < 8; c++) {
                int64_t off = row * fftN + c;
                std::cout << hostOutput[off] << " ";
            }
            std::cout << std::endl;
        }

        // Write output bin for gen_data.py
        std::string outputPath = options.dataDir + "/output_" + std::to_string(fftN) + "_" +
                                 std::to_string(batch) + ".bin";
        if (WriteBinFile(outputPath, hostOutput.data(), outputSize)) {
            std::cout << "\nWrote output: " << outputPath << std::endl;
        }
    }

    FftC2R2DPlanDestroy(plan);
    ACL_CHECK(aclrtFree(deviceInput));
    ACL_CHECK(aclrtFree(deviceOutput));
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(options.deviceId));
    ACL_CHECK(aclFinalize());
}

int main(int argc, const char** argv)
{
    FftC2R2DOptions options;
    if (options.Parse(argc, argv) != 0) return 1;
    Run(options);
    return 0;
}
