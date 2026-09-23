#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/status.hpp"

#include "fft_plan.h"
#include "fft_kernel.h"
#include "fft_twiddle.h"
#include "fft_tiling_def.h"

#include "helper.hpp"

#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

namespace Fft {

static bool BuildAndCopyCoeffs(uint8_t* dCoeffs, int32_t N1, int32_t direction, int64_t fftN)
{
    int32_t w0Dim = 2 * N1;
    int64_t w0Floats = static_cast<int64_t>(w0Dim) * w0Dim;
    int64_t tFloats = 3 * (2 * N1) * N1;                       // T12/T13/T23
    int64_t tExpFloats = static_cast<int64_t>(4096) * 2;        // T2 展开表
    int64_t tRadix2Floats = static_cast<int64_t>(RADIX2_TW_FLOATS) * 2;
    int64_t tPhaseAFloats = static_cast<int64_t>(PHASE_A_OFFSET_FLOATS);
    int64_t t12ExpFloats = static_cast<int64_t>(P3_T12_EXP_FLOATS);
    int64_t tP3LaneFloats = static_cast<int64_t>(P3_LANE_COUNT);

    std::vector<float> hostW0(w0Floats);
    std::vector<float> hostT(tFloats);
    std::vector<float> hostTExp(tExpFloats);
    std::vector<float> hostRadix2TW(tRadix2Floats);
    std::vector<uint32_t> hostPhaseA(PHASE_A_OFFSET_COUNT);
    std::vector<float> hostT12Exp(t12ExpFloats);
    std::vector<uint32_t> hostP3Lane(P3_LANE_COUNT);

    BuildWMatrix(hostW0.data(), N1, direction);
    BuildTMatrix3(hostT.data(), N1, direction);
    BuildT2Expanded(hostTExp.data(), N1, direction);
    BuildRadix2Twiddles(hostRadix2TW.data(), direction);
    BuildPhaseAOffsets(hostPhaseA.data());
    BuildT12Expanded(hostT12Exp.data(), N1, direction);
    BuildP3LaneOffsets(hostP3Lane.data());

    size_t offset = 0;
    auto cp = [&](const void* src, size_t bytes, const char* what) -> bool {
        aclError r = aclrtMemcpy(dCoeffs + offset, bytes, src, bytes, ACL_MEMCPY_HOST_TO_DEVICE);
        if (r != ACL_ERROR_NONE) { std::cerr << "copy " << what << " failed" << std::endl; return false; }
        offset += bytes;
        return true;
    };

    if (!cp(hostW0.data(), w0Floats * sizeof(float), "W0")) return false;
    if (!cp(hostT.data(), tFloats * sizeof(float), "T")) return false;
    if (!cp(hostTExp.data(), tExpFloats * sizeof(float), "T2Exp")) return false;
    if (!cp(hostRadix2TW.data(), tRadix2Floats * sizeof(float), "Radix2TW")) return false;
    if (!cp(hostPhaseA.data(), tPhaseAFloats * sizeof(uint32_t), "PhaseA")) return false;
    if (!cp(hostT12Exp.data(), t12ExpFloats * sizeof(float), "T12Exp")) return false;
    if (!cp(hostP3Lane.data(), tP3LaneFloats * sizeof(uint32_t), "P3Lane")) return false;
    return true;
}

bool FftR2C2DPlanCreate(FftR2C2DPlan& plan, int64_t fftN, int32_t batch,
                        int32_t deviceId, int32_t cubeCoreNum)
{
    if (fftN != 4096) {
        std::cerr << "Error: only fftN=4096 supported, got " << fftN << std::endl;
        return false;
    }

    plan.fftN = fftN;
    plan.batch = batch;
    plan.deviceId = deviceId;

    int32_t direction = -1;   // R2C 2D 正向
    int64_t halfN = fftN / 2 + 1;             // 2049
    int64_t padHalf = (halfN + 63) & ~int64_t(63);   // pad 到 4 的倍数 = 2052（行宽 32B 对齐）
    int64_t numSignals1 = fftN / 2;           // Pass1：两实行打包 -> 2048 复信号

    ComputeTiling(plan.tiling1, fftN, numSignals1, direction, cubeCoreNum, 0, 2);
    ComputeTiling(plan.tiling2, fftN, padHalf, direction, cubeCoreNum, 0, 2);

    plan.coeffsSize = GetCoeffsSize(fftN);
    plan.workspaceSize = GetWorkspaceSize(fftN, numSignals1, cubeCoreNum);
    plan.tilingSize = sizeof(FftTilingData);

    plan.ySize = static_cast<size_t>(fftN) * padHalf * 2 * sizeof(float);
    plan.zSize = static_cast<size_t>(padHalf) * fftN * 2 * sizeof(float);
    plan.wSize = plan.zSize;
    plan.w2Size = plan.ySize;

    ACL_CHECK(aclrtSetDevice(deviceId));

    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&plan.dCoeffs), plan.coeffsSize, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&plan.dWorkspace), plan.workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&plan.dTiling1), plan.tilingSize, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&plan.dTiling2), plan.tilingSize, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&plan.dY), plan.ySize, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&plan.dZ), plan.zSize, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&plan.dW), plan.wSize, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&plan.dW2), plan.w2Size, ACL_MEM_MALLOC_HUGE_FIRST));

    if (!BuildAndCopyCoeffs(plan.dCoeffs, plan.tiling1.N1, direction, fftN)) {
        return false;
    }

    FftTilingData devTiling;
    FillDeviceTiling(plan.tiling1, devTiling);
    ACL_CHECK(aclrtMemcpy(plan.dTiling1, plan.tilingSize, &devTiling, plan.tilingSize, ACL_MEMCPY_HOST_TO_DEVICE));

    FillDeviceTiling(plan.tiling2, devTiling);
    ACL_CHECK(aclrtMemcpy(plan.dTiling2, plan.tilingSize, &devTiling, plan.tilingSize, ACL_MEMCPY_HOST_TO_DEVICE));

    std::cout << "FftR2C2DPlanCreate: fftN=" << fftN << " halfN=" << halfN
              << " padHalf=" << padHalf
              << " coeffsSize=" << plan.coeffsSize
              << " workspaceSize=" << plan.workspaceSize
              << " ySize=" << plan.ySize << " zSize=" << plan.zSize << std::endl;
    return true;
}

void FftR2C2DExec(FftR2C2DPlan& plan, uint8_t* dInput, uint8_t* dOutput,
                  aclrtStream stream, int32_t cubeCoreNum)
{
    uint32_t blockDim = static_cast<uint32_t>(cubeCoreNum);
    FftR2C2DFusedKernelLaunch(dInput, dOutput, plan.dCoeffs, plan.dWorkspace,
                              plan.dTiling1, plan.dTiling2,
                              plan.dY, plan.dZ, plan.dW, plan.dW2,
                              blockDim, stream);
    ACL_CHECK(aclrtSynchronizeStream(stream));
}

void FftR2C2DPlanDestroy(FftR2C2DPlan& plan)
{
    if (plan.dCoeffs) { aclrtFree(plan.dCoeffs); plan.dCoeffs = nullptr; }
    if (plan.dWorkspace) { aclrtFree(plan.dWorkspace); plan.dWorkspace = nullptr; }
    if (plan.dTiling1) { aclrtFree(plan.dTiling1); plan.dTiling1 = nullptr; }
    if (plan.dTiling2) { aclrtFree(plan.dTiling2); plan.dTiling2 = nullptr; }
    if (plan.dY) { aclrtFree(plan.dY); plan.dY = nullptr; }
    if (plan.dZ) { aclrtFree(plan.dZ); plan.dZ = nullptr; }
    if (plan.dW) { aclrtFree(plan.dW); plan.dW = nullptr; }
    if (plan.dW2) { aclrtFree(plan.dW2); plan.dW2 = nullptr; }
}

} // namespace Fft