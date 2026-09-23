#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/status.hpp"

#include "fft_plan.h"
#include "fft_kernel.h"
#include "fft_tiling_def.h"
#include "fft_radix16_twiddle.h"

#include "helper.hpp"

#include <cstddef>
#include <cstring>
#include <iostream>
#include <vector>

namespace Fft {

static constexpr int32_t N1_16 = RADIX16_N;
static constexpr int32_t N2_16 = RADIX16_N;

bool FftC2R2DPlanCreate(FftC2R2DPlan& plan, int64_t fftN, int32_t batch,
                        int32_t deviceId, int32_t cubeCoreNum)
{
    if (fftN != 4096) {
        std::cerr << "FftC2R2DPlanCreate: only fftN=4096 supported, got " << fftN << std::endl;
        return false;
    }

    plan.fftN = fftN;
    plan.batch = batch;
    plan.deviceId = deviceId;

    ComputeC2R2DTiling(plan.tiling, fftN, batch, cubeCoreNum);
    plan.coeffsSize = GetC2R2DCoeffsSize(fftN);
    plan.bWorkspaceSize = GetC2R2DBWorkspaceSize(fftN, batch, cubeCoreNum);
    plan.workspaceSize = GetC2R2DTransposeWorkspaceSize(fftN, batch);
    plan.tilingSize = sizeof(FftC2R2DTilingData);

    // 系数构造
    int64_t w0Floats = static_cast<int64_t>(2 * N1_16) * (2 * N1_16);       // 1024
    int64_t tFloats = static_cast<int64_t>(2 * N1_16) * N2_16 * 3;           // 1536
    int64_t tExpFloats = static_cast<int64_t>(4096) * 2;                     // 8192
    int64_t tR2Floats = static_cast<int64_t>(RADIX2_TW_FLOATS) * 2;          // 1920
    int64_t tPhaseAFloats = static_cast<int64_t>(PHASE_A_OFFSET_COUNT);      // 4096
    int64_t t12ExpFloats = static_cast<int64_t>(P3_T12_EXP_FLOATS);          // 8192
    int64_t p3LaneFloats = static_cast<int64_t>(P3_LANE_COUNT);              // 64

    std::vector<float> hostW(w0Floats);
    std::vector<float> hostT(tFloats);
    std::vector<float> hostTExp(tExpFloats);
    std::vector<float> hostR2(tR2Floats);
    std::vector<uint32_t> hostPhaseA(tPhaseAFloats);
    std::vector<float> hostT12Exp(t12ExpFloats);
    std::vector<uint32_t> hostP3Lane(p3LaneFloats);

    BuildWMatrix16(hostW.data(), FFT_INVERSE);
    BuildTMatrix3_16(hostT.data(), FFT_INVERSE);
    BuildT2Expanded16(hostTExp.data(), FFT_INVERSE);
    BuildRadix2Twiddles16(hostR2.data(), FFT_INVERSE);
    BuildPhaseAOffsets16(hostPhaseA.data());
    BuildT12Expanded16(hostT12Exp.data(), FFT_INVERSE);
    BuildP3LaneOffsets16(hostP3Lane.data());

    aclError ret = aclrtSetDevice(deviceId);
    if (ret != ACL_ERROR_NONE) {
        std::cerr << "FftC2R2DPlanCreate: aclrtSetDevice failed: " << ret << std::endl;
        return false;
    }

    ret = aclrtMalloc(reinterpret_cast<void**>(&plan.dCoeffs), plan.coeffsSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_ERROR_NONE) { std::cerr << "malloc coeffs failed" << std::endl; return false; }

    size_t off = 0;
    auto copy = [&](const void* src, size_t bytes) -> bool {
        ret = aclrtMemcpy(plan.dCoeffs + off, bytes, src, bytes, ACL_MEMCPY_HOST_TO_DEVICE);
        if (ret != ACL_ERROR_NONE) { std::cerr << "copy coeffs failed @" << off << std::endl; return false; }
        off += bytes;
        return true;
    };
    if (!copy(hostW.data(), w0Floats * sizeof(float))) return false;
    if (!copy(hostT.data(), tFloats * sizeof(float))) return false;
    if (!copy(hostTExp.data(), tExpFloats * sizeof(float))) return false;
    if (!copy(hostR2.data(), tR2Floats * sizeof(float))) return false;
    if (!copy(hostPhaseA.data(), tPhaseAFloats * sizeof(uint32_t))) return false;
    if (!copy(hostT12Exp.data(), t12ExpFloats * sizeof(float))) return false;
    if (!copy(hostP3Lane.data(), p3LaneFloats * sizeof(uint32_t))) return false;

    ret = aclrtMalloc(reinterpret_cast<void**>(&plan.dBWorkspace), plan.bWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_ERROR_NONE) { std::cerr << "malloc B workspace failed" << std::endl; aclrtFree(plan.dCoeffs); plan.dCoeffs = nullptr; return false; }

    ret = aclrtMalloc(reinterpret_cast<void**>(&plan.dWorkspace), plan.workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_ERROR_NONE) { std::cerr << "malloc transpose workspace failed" << std::endl; aclrtFree(plan.dCoeffs); plan.dCoeffs = nullptr; aclrtFree(plan.dBWorkspace); plan.dBWorkspace = nullptr; return false; }

    ret = aclrtMalloc(reinterpret_cast<void**>(&plan.dTiling), plan.tilingSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_ERROR_NONE) { std::cerr << "malloc tiling failed" << std::endl; aclrtFree(plan.dCoeffs); plan.dCoeffs = nullptr; aclrtFree(plan.dBWorkspace); plan.dBWorkspace = nullptr; aclrtFree(plan.dWorkspace); plan.dWorkspace = nullptr; return false; }

    FftC2R2DTilingData devTiling;
    FillC2R2DDeviceTiling(plan.tiling, devTiling);
    ret = aclrtMemcpy(plan.dTiling, plan.tilingSize, &devTiling, plan.tilingSize, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_ERROR_NONE) { std::cerr << "copy tiling failed" << std::endl; return false; }

    std::cout << "FftC2R2DPlanCreate: fftN=" << fftN << " batch=" << batch
              << " (N1=" << plan.tiling.N1 << ",N2=" << plan.tiling.N2 << ")"
              << " halfN=" << plan.tiling.halfN << " padHalf=" << plan.tiling.padHalf
              << " cubeCores=" << cubeCoreNum
              << " coeffsSize=" << plan.coeffsSize
              << " bWorkspaceSize=" << plan.bWorkspaceSize
              << " transposeWorkspaceSize=" << plan.workspaceSize << std::endl;
    return true;
}

void FftC2R2DExec(FftC2R2DPlan& plan, uint8_t* dInput, uint8_t* dOutput,
                  aclrtStream stream, int32_t cubeCoreNum)
{
    uint32_t blockDim = static_cast<uint32_t>(cubeCoreNum);
    FftC2R2DKernelLaunch(dInput, dOutput, plan.dWorkspace, plan.dBWorkspace,
                         plan.dCoeffs, plan.dTiling, blockDim, stream);
    ACL_CHECK(aclrtSynchronizeStream(stream));
}

void FftC2R2DExecStage(FftC2R2DPlan& plan, uint8_t* dInput, uint8_t* dOutput,
                       aclrtStream stream, int32_t cubeCoreNum, int32_t stage)
{
    (void)stage;
    FftC2R2DExec(plan, dInput, dOutput, stream, cubeCoreNum);
}

void FftC2R2DPlanDestroy(FftC2R2DPlan& plan)
{
    if (plan.dCoeffs != nullptr) { aclrtFree(plan.dCoeffs); plan.dCoeffs = nullptr; }
    if (plan.dBWorkspace != nullptr) { aclrtFree(plan.dBWorkspace); plan.dBWorkspace = nullptr; }
    if (plan.dWorkspace != nullptr) { aclrtFree(plan.dWorkspace); plan.dWorkspace = nullptr; }
    if (plan.dTiling != nullptr) { aclrtFree(plan.dTiling); plan.dTiling = nullptr; }
}

} // namespace Fft