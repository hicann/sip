#ifndef FFT_PLAN_H
#define FFT_PLAN_H

#include <cstdint>
#include <acl/acl.h>
#include "fft_types.h"
#include "fft_tiling.h"
#include "fft_tiling_def.h"

namespace Fft {

struct FftR2C2DPlan {
    int64_t fftN;
    int32_t batch;
    int32_t deviceId;

    FftHostTilingData tiling1;   // Pass 1（R2C，numSignals = fftN/2）
    FftHostTilingData tiling2;   // Pass 2（C2C，numSignals = halfN pad 到偶数）

    uint8_t* dCoeffs;
    size_t coeffsSize;
    uint8_t* dWorkspace;        // 两块 pass 共用 B0/B1 GEMM 工作区（顺序执行）
    size_t workspaceSize;
    uint8_t* dTiling1;
    uint8_t* dTiling2;
    size_t tilingSize;

    // 转置工作区（内部）
    uint8_t* dY;   // [fftN, padHalf] complex（Pass1 输出，pad 到 64 倍数）
    uint8_t* dZ;   // [padHalf, fftN] complex（转置后）
    uint8_t* dW;   // [padHalf, fftN] complex（Pass2 输出）
    uint8_t* dW2;  // [fftN, padHalf] complex（Pass2 转置回，pad 到 64 倍数）
    size_t ySize, zSize, wSize, w2Size;

    FftR2C2DPlan()
        : fftN(0), batch(0), deviceId(0),
          dCoeffs(nullptr), coeffsSize(0),
          dWorkspace(nullptr), workspaceSize(0),
          dTiling1(nullptr), dTiling2(nullptr), tilingSize(0),
          dY(nullptr), dZ(nullptr), dW(nullptr), dW2(nullptr),
          ySize(0), zSize(0), wSize(0), w2Size(0)
    {}
};

bool FftR2C2DPlanCreate(FftR2C2DPlan& plan, int64_t fftN, int32_t batch,
                        int32_t deviceId, int32_t cubeCoreNum);
void FftR2C2DExec(FftR2C2DPlan& plan, uint8_t* dInput, uint8_t* dOutput,
                  aclrtStream stream, int32_t cubeCoreNum);
void FftR2C2DPlanDestroy(FftR2C2DPlan& plan);

} // namespace Fft

#endif // FFT_PLAN_H