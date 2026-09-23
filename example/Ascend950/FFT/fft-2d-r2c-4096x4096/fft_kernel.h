#ifndef FFT_KERNEL_H
#define FFT_KERNEL_H

#include <cstdint>
#include <acl/acl.h>

namespace Fft {

// 融合单 launch：R2C1D → 转置 → C2C1D → 转置 → 裁剪，5 个 stage 阶段间 SyncAll<false>。
// 数据流：x -> y[fftN,padHalf] -> z[padHalf,fftN] -> w[padHalf,fftN] -> w2[fftN,padHalf] -> out[fftN,halfN]
// y/z/w/w2 为 GM 中间缓冲（stage 间跨 block 数据依赖必须经 GM 传递）。
void FftR2C2DFusedKernelLaunch(
    GM_ADDR input, GM_ADDR output, GM_ADDR coeffs, GM_ADDR workspace,
    GM_ADDR tiling1, GM_ADDR tiling2,
    GM_ADDR y, GM_ADDR z, GM_ADDR w, GM_ADDR w2,
    uint32_t blockDim, aclrtStream stream);

} // namespace Fft

#endif // FFT_KERNEL_H
