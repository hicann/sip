#ifndef FFT_PLAN_H
#define FFT_PLAN_H

#include <cstdint>
#include <acl/acl.h>
#include "fft_types.h"
#include "fft_tiling.h"
#include "fft_tiling_def.h"

namespace Fft {

struct FftC2R2DPlan {
    int64_t fftN;
    int32_t batch;
    int32_t deviceId;

    FftC2R2DHostTiling tiling;

    uint8_t* dCoeffs;
    size_t coeffsSize;

    uint8_t* dBWorkspace;       // B0/B1 GEMM 工作区
    size_t bWorkspaceSize;

    uint8_t* dWorkspace;        // 转置工作区（ws0 + ws1）
    size_t workspaceSize;

    uint8_t* dTiling;
    size_t tilingSize;

    FftC2R2DPlan()
        : fftN(0), batch(0), deviceId(0),
          dCoeffs(nullptr), coeffsSize(0),
          dBWorkspace(nullptr), bWorkspaceSize(0),
          dWorkspace(nullptr), workspaceSize(0),
          dTiling(nullptr), tilingSize(0)
    {}
};

bool FftC2R2DPlanCreate(FftC2R2DPlan& plan, int64_t fftN, int32_t batch,
                        int32_t deviceId, int32_t cubeCoreNum);
void FftC2R2DExec(FftC2R2DPlan& plan, uint8_t* dInput, uint8_t* dOutput,
                  aclrtStream stream, int32_t cubeCoreNum);
void FftC2R2DExecStage(FftC2R2DPlan& plan, uint8_t* dInput, uint8_t* dOutput,
                       aclrtStream stream, int32_t cubeCoreNum, int32_t stage);
void FftC2R2DPlanDestroy(FftC2R2DPlan& plan);

} // namespace Fft

#endif // FFT_PLAN_H