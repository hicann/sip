#ifndef FFT_KERNEL_H
#define FFT_KERNEL_H

#include <cstdint>
#include <acl/acl.h>
#include "fft_kernel_utils.h"

namespace Fft {

// 1D FFT kernel（release 2×16×16×2×2×2 引擎，248KB UB，无 SIMT）
void FftKernelLaunch(
    GM_ADDR input, GM_ADDR output, GM_ADDR coeffs, GM_ADDR workspace, GM_ADDR tiling,
    uint32_t blockDim, aclrtStream stream, int32_t ubMode = 0, int32_t mixMode = 2);

// 转置 kernel（64×64 SIMT，216KB UB）
void FftTransposeKernelLaunch(
    GM_ADDR input, GM_ADDR output, GM_ADDR tiling,
    uint32_t blockDim, aclrtStream stream, int32_t vecCoreNum);

} // namespace Fft

#endif // FFT_KERNEL_H