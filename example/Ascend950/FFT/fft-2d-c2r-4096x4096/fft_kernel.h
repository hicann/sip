#ifndef FFT_KERNEL_H
#define FFT_KERNEL_H

#include <cstdint>
#include <acl/acl.h>

namespace Fft {

// radix-16 C2R 2D FFT kernel — 逆序瘦列 4-stage in one launch:
//   S1 transpose input[fftN,padHalf] -> Xt[padHalf,fftN]
//   S2 C2C 逆变换 (axis0, padHalf 条) -> Y[padHalf,fftN]
//   S3 transpose Y -> Yt[fftN,padHalf]
//   S4 irfft (axis1, Hermitian 展开 + IDFT + 取实) -> output[fftN,fftN] real
//   input     [batch, fftN, padHalf] complex64（半谱，pad 到 64 倍数）
//   output    [batch, fftN, fftN]    float32（实数）
//   workspace 三个瘦副本 wsA/wsB/wsC
//   bws       radix-16 GEMM B0/B1 workspace  coeffs radix-16 系数  tiling 设备 tiling
void FftC2R2DKernelLaunch(
    GM_ADDR input, GM_ADDR output, GM_ADDR workspace, GM_ADDR bws,
    GM_ADDR coeffs, GM_ADDR tiling,
    uint32_t blockDim, aclrtStream stream);

} // namespace Fft

#endif // FFT_KERNEL_H