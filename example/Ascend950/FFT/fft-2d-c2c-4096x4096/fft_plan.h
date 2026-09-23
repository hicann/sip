#ifndef FFT_PLAN_H
#define FFT_PLAN_H

#include <cstdint>
#include <acl/acl.h>
#include "fft_types.h"
#include "fft_tiling.h"
#include "fft_tiling_def.h"

namespace Fft {

struct FftPlan {
    int64_t fftN;
    int64_t batch;
    int32_t direction;
    int32_t deviceId;

    RadixDecomp decomp;
    FftHostTilingData tiling;

    uint8_t* dCoeffs;
    size_t coeffsSize;
    uint8_t* dWorkspace;
    size_t workspaceSize;
    uint8_t* dWsTranspose;     // K0 输出 / K1 输入（ws = X.T）
    size_t wsTransposeSize;
    uint8_t* dTmp;             // K1 输出（tmp = fft(X.T, axis=1) = (F·X).T）
    size_t tmpSize;
    uint8_t* dHori;            // K1 横向输出（h = fft(X, axis=1)，FFT2D Pass1）
    size_t horiSize;
    uint8_t* dTiling;
    size_t tilingSize;

    FftPlan()
        : fftN(0), batch(1), direction(-1), deviceId(0),
        dCoeffs(nullptr), coeffsSize(0),
        dWorkspace(nullptr), workspaceSize(0),
        dWsTranspose(nullptr), wsTransposeSize(0),
        dTmp(nullptr), tmpSize(0),
        dHori(nullptr), horiSize(0),
        dTiling(nullptr), tilingSize(0)
    {}
};

bool FftPlanCreate(FftPlan& plan, int64_t fftN, int64_t batch, int32_t direction,
                   int32_t deviceId, int32_t cubeCoreNum, int32_t vecCoreNum);

// FFT2D：横向 FFT(axis=1) 后纵向 FFT(axis=0)，复用同一引擎（融合单 launch）
void FftExec2D(FftPlan& plan, uint8_t* dInput, uint8_t* dOutput,
               aclrtStream stream, int32_t cubeCoreNum, int32_t vecCoreNum);
void FftPlanDestroy(FftPlan& plan);

} // namespace Fft

#endif // FFT_PLAN_H