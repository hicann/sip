#ifndef FFT_TILING_H
#define FFT_TILING_H

#include <cstddef>
#include <cstdint>
#include "fft_types.h"
#include "fft_tiling_def.h"

namespace Fft {

// host 侧 tiling（推导中间量与尺寸；FillDeviceTiling 时收敛进 FftTilingData）
struct FftHostTilingData {
    // —— 信号描述（与 FftPlan 入参一致）——
    int64_t batch;
    int64_t fftN;
    int32_t direction;

    // —— 多核划分 ——
    int32_t cubeCoreNum;
    int32_t totalRows;
    int32_t rowsPerCore;
    int32_t rowRemainder;

    // —— stage 化（对齐概要设计 Part A §2.3）：偏移/尺寸直接在 FftTilingData 的
    //    stage[].gemm/butterfly 族参数上填，host 结构只留派生汇总量 ——
    int32_t stageCount;
    int32_t gemmCount;        // GEMM 轮数 = Σ(type==Gemm)

    size_t coeffsSize;        // coeffs 总大小（W 系去重 + T 系 + 展开表）
    size_t workspaceSize;     // workspace 总大小 = coreNum × wsRegionBytes
    int64_t wsRegionBytes;    // 每核 workspace 区大小
};

void FillDeviceTiling(const FftHostTilingData& hostTiling, const RadixDecomp& decomp,
                      FftTilingData& devTiling);
void ComputeTiling(FftHostTilingData& tiling, const RadixDecomp& decomp, int64_t fftN,
                   int64_t batch, int32_t direction, int32_t cubeCoreNum);
size_t GetCoeffsSize(const RadixDecomp& decomp);
size_t GetWorkspaceSize(const RadixDecomp& decomp, int64_t batch, int32_t cubeCoreNum);

} // namespace Fft

#endif // FFT_TILING_H