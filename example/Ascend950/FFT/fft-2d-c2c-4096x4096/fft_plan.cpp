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

// host 侧 Plan/Exec/Destroy。2D = 横向 FFT(axis=1) + 纵向 FFT(axis=0)，
// 复用 release 1D 引擎（2×16×16×2×2×2），融合单 launch 内完成 S1→S2→S3→S4。
namespace Fft {

bool FftPlanCreate(FftPlan& plan, int64_t fftN, int64_t batch, int32_t direction,
                   int32_t deviceId, int32_t cubeCoreNum, int32_t vecCoreNum)
{
    // 定制化：仅支持 4096 点 FFT、[4096,4096] 输入
    if (fftN != FFT_N_REQUIRED || batch != 4096) {
        std::cerr << "Error: only [4096,4096] is supported, got fftN=" << fftN
                  << " batch=" << batch << std::endl;
        return false;
    }

    plan.fftN = fftN;
    plan.batch = batch;
    plan.direction = direction;
    plan.deviceId = deviceId;
    plan.decomp.Decompose(fftN);

    ComputeTiling(plan.tiling, plan.decomp, fftN, batch, direction, cubeCoreNum);

    plan.coeffsSize = GetCoeffsSize(plan.decomp);
    plan.workspaceSize = GetWorkspaceSize(plan.decomp, batch, cubeCoreNum);
    plan.wsTransposeSize = static_cast<size_t>(batch) * fftN * 2 * sizeof(float);
    plan.tmpSize = static_cast<size_t>(batch) * fftN * 2 * sizeof(float);
    plan.horiSize = static_cast<size_t>(batch) * fftN * 2 * sizeof(float);
    plan.tilingSize = sizeof(FftTilingData);

    // ── host 侧构造系数（stage 化，与 release fft_plan.cpp 同源）──
    // coeffs 布局约定：W 系（按 stage 序、同基去重）→ T 系（按级间序）
    //   → T_expanded → GATHER_TABLE → P6T2 → P4
    std::vector<float> hostCoeffs(plan.coeffsSize / sizeof(float));
    {
        int64_t coeffFloats = 0;
        int64_t seenRadix[FFT_MAX_STAGE] = {0};
        int32_t seenN = 0;
        for (int32_t i = 0; i < plan.decomp.stageCount; i++) {
            if (plan.decomp.stage[i].type != FftStageType::Gemm) { continue; }
            int32_t r = plan.decomp.stage[i].radix;
            bool dup = false;
            for (int32_t j = 0; j < seenN; j++) {
                if (seenRadix[j] == r) { dup = true; break; }
            }
            if (dup) { continue; }
            seenRadix[seenN++] = r;
            BuildWMatrix(hostCoeffs.data() + coeffFloats, r, direction);
            coeffFloats += (2 * r) * (2 * r);
        }
        for (int32_t i = 0; i < plan.decomp.stageCount - 1; i++) {
            if (plan.decomp.stage[i].type == FftStageType::Butterfly) {
                BuildT0TwiddleCompact(hostCoeffs.data() + coeffFloats, direction);
                coeffFloats += T0_COMPACT_FLOATS;
            } else if (i == plan.decomp.stageCount - 2) {
                BuildT2Expanded(hostCoeffs.data() + coeffFloats, direction);
                coeffFloats += T2_EXPANDED_FLOATS;
            } else {
                BuildTMatrix(hostCoeffs.data() + coeffFloats,
                             plan.decomp.GetTGroups(i), plan.decomp.stage[i].radix,
                             direction, plan.decomp.GetTDenominator(i));
                coeffFloats += plan.decomp.GetTFloats(i);
            }
        }
        BuildP6Radix2TwiddleExpanded(hostCoeffs.data() + coeffFloats, direction);
        coeffFloats += 896;
        BuildInterleaveBitrevGatherTable(
            reinterpret_cast<uint32_t*>(hostCoeffs.data() + coeffFloats));
        coeffFloats += GATHER_TABLE_FLOATS;
        BuildP6T2GatherTables(
            reinterpret_cast<uint32_t*>(hostCoeffs.data() + coeffFloats));
        coeffFloats += P6T2_TABLE_FLOATS;
        BuildP4GatherTables(
            reinterpret_cast<uint32_t*>(hostCoeffs.data() + coeffFloats));
        coeffFloats += P4_TABLE_FLOATS;
    }

    aclError ret = aclrtSetDevice(deviceId);
    if (ret != ACL_ERROR_NONE) {
        std::cerr << "FftPlanCreate: aclrtSetDevice failed: " << ret << std::endl;
        return false;
    }

    ret = aclrtMalloc(reinterpret_cast<void**>(&plan.dCoeffs), plan.coeffsSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_ERROR_NONE) {
        std::cerr << "FftPlanCreate: aclrtMalloc coeffs failed: " << ret << std::endl;
        return false;
    }

    ret = aclrtMemcpy(plan.dCoeffs, plan.coeffsSize, hostCoeffs.data(),
                      plan.coeffsSize, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_ERROR_NONE) { std::cerr << "copy coeffs failed" << std::endl; return false; }

    ret = aclrtMalloc(reinterpret_cast<void**>(&plan.dWorkspace), plan.workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_ERROR_NONE) {
        std::cerr << "FftPlanCreate: aclrtMalloc workspace failed: " << ret << std::endl;
        aclrtFree(plan.dCoeffs); plan.dCoeffs = nullptr;
        return false;
    }

    ret = aclrtMalloc(reinterpret_cast<void**>(&plan.dWsTranspose), plan.wsTransposeSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_ERROR_NONE) {
        std::cerr << "FftPlanCreate: aclrtMalloc wsTranspose failed: " << ret << std::endl;
        aclrtFree(plan.dCoeffs); aclrtFree(plan.dWorkspace);
        plan.dCoeffs = nullptr; plan.dWorkspace = nullptr;
        return false;
    }

    ret = aclrtMalloc(reinterpret_cast<void**>(&plan.dTmp), plan.tmpSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_ERROR_NONE) {
        std::cerr << "FftPlanCreate: aclrtMalloc tmp failed: " << ret << std::endl;
        aclrtFree(plan.dCoeffs); aclrtFree(plan.dWorkspace); aclrtFree(plan.dWsTranspose);
        plan.dCoeffs = nullptr; plan.dWorkspace = nullptr; plan.dWsTranspose = nullptr;
        return false;
    }

    ret = aclrtMalloc(reinterpret_cast<void**>(&plan.dHori), plan.horiSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_ERROR_NONE) {
        std::cerr << "FftPlanCreate: aclrtMalloc hori failed: " << ret << std::endl;
        aclrtFree(plan.dCoeffs); aclrtFree(plan.dWorkspace); aclrtFree(plan.dWsTranspose); aclrtFree(plan.dTmp);
        plan.dCoeffs = nullptr; plan.dWorkspace = nullptr; plan.dWsTranspose = nullptr; plan.dTmp = nullptr;
        return false;
    }

    ret = aclrtMalloc(reinterpret_cast<void**>(&plan.dTiling), plan.tilingSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_ERROR_NONE) {
        std::cerr << "FftPlanCreate: aclrtMalloc tiling failed: " << ret << std::endl;
        aclrtFree(plan.dCoeffs); aclrtFree(plan.dWorkspace); aclrtFree(plan.dWsTranspose);
        aclrtFree(plan.dTmp); aclrtFree(plan.dHori);
        plan.dCoeffs = nullptr; plan.dWorkspace = nullptr; plan.dWsTranspose = nullptr;
        plan.dTmp = nullptr; plan.dHori = nullptr;
        return false;
    }

    FftTilingData devTiling;
    FillDeviceTiling(plan.tiling, plan.decomp, devTiling);

    ret = aclrtMemcpy(plan.dTiling, plan.tilingSize, &devTiling, plan.tilingSize, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_ERROR_NONE) {
        std::cerr << "FftPlanCreate: copy tiling failed: " << ret << std::endl;
        FftPlanDestroy(plan);
        return false;
    }

    std::cout << "FftPlanCreate: fftN=" << fftN << " batch=" << batch
              << " radix=" << FftRadixName()
              << " gemmStages=" << plan.decomp.GetGemmCount()
              << " cubeCores=" << cubeCoreNum << " vecCores=" << vecCoreNum
              << " coeffsSize=" << plan.coeffsSize
              << " workspaceSize=" << plan.workspaceSize
              << " wsTransposeSize=" << plan.wsTransposeSize
              << " tmpSize=" << plan.tmpSize
              << " horiSize=" << plan.horiSize << std::endl;

    return true;
}

void FftPlanDestroy(FftPlan& plan)
{
    if (plan.dCoeffs != nullptr) { aclrtFree(plan.dCoeffs); plan.dCoeffs = nullptr; }
    if (plan.dWorkspace != nullptr) { aclrtFree(plan.dWorkspace); plan.dWorkspace = nullptr; }
    if (plan.dWsTranspose != nullptr) { aclrtFree(plan.dWsTranspose); plan.dWsTranspose = nullptr; }
    if (plan.dTmp != nullptr) { aclrtFree(plan.dTmp); plan.dTmp = nullptr; }
    if (plan.dHori != nullptr) { aclrtFree(plan.dHori); plan.dHori = nullptr; }
    if (plan.dTiling != nullptr) { aclrtFree(plan.dTiling); plan.dTiling = nullptr; }
}

void FftExec2D(FftPlan& plan, uint8_t* dInput, uint8_t* dOutput,
               aclrtStream stream, int32_t cubeCoreNum, int32_t vecCoreNum)
{
    // FFT2D = 横向 FFT(axis=1) 后纵向 FFT(axis=0)：
    //   S1(K1 行FFT) X -> h;  S2(转置) h -> ws;
    //   S3(K1 行FFT) ws -> t;  S4(转置) t -> out = fft2(X)
    // 4 次独立 launch（同一 stream 串行）：1D FFT 引擎需 248KB UB（无 SIMT）、
    // 转置用 SIMT（UB 上限 216KB），两者 UB 预算不同故不能融合为单 kernel。
    uint32_t blockDim = static_cast<uint32_t>(cubeCoreNum);

    // S1: 横向 FFT：X -> h（release 2×16×16×2×2×2 引擎）
    FftKernelLaunch(dInput, plan.dHori, plan.dCoeffs, plan.dWorkspace, plan.dTiling,
                    blockDim, stream, 0, 2);

    // S2: 转置：h -> ws
    FftTransposeKernelLaunch(plan.dHori, plan.dWsTranspose, plan.dTiling,
                             blockDim, stream, vecCoreNum);

    // S3: 纵向 FFT：ws -> t
    FftKernelLaunch(plan.dWsTranspose, plan.dTmp, plan.dCoeffs, plan.dWorkspace, plan.dTiling,
                    blockDim, stream, 0, 2);

    // S4: 转置：t -> out
    FftTransposeKernelLaunch(plan.dTmp, dOutput, plan.dTiling,
                             blockDim, stream, vecCoreNum);

    ACL_CHECK(aclrtSynchronizeStream(stream));
}

} // namespace Fft