#ifndef FFT_TYPES_H
#define FFT_TYPES_H

#include <cstdint>

namespace Fft {

constexpr double FFT_PI = 3.14159265358979323846;
constexpr double FFT_2PI = 2.0 * FFT_PI;

// FFT direction: +1 inverse (C2R = irfft2), -1 forward
constexpr int32_t FFT_INVERSE = 1;
constexpr int32_t FFT_FORWARD = -1;

// radix-16（16×16×16）：4096 = 16 × 16 × 16
// 三轮 GEMM：W16(32×32) @ B(32×512)，第三轮由 radix-2 butterfly 替代
//   iter0 GEMM0: W16 @ B0 -> C0  (16 点 DFT)
//   iter1 GEMM1: W16 @ B1 -> C1  (16 点 DFT)
//   iter2: T2 复乘 + bit-reverse + 16 点 radix-2 butterfly（无 GEMM2）
constexpr int32_t RADIX16_N = 16;

struct RadixDecomp2Stage {
    int32_t N1;
    int32_t N2;
    int32_t iterCount;

    RadixDecomp2Stage() : N1(0), N2(0), iterCount(0) {}

    void Decompose(int64_t fftN)
    {
        (void)fftN;
        N1 = RADIX16_N;   // 16
        N2 = RADIX16_N;   // 16
        iterCount = 3;
    }

    bool IsValid() const { return N1 > 0 && N2 > 0; }

    // W16 矩阵（实/虚分离）：(2N) × (2N) = 32×32，三轮复用
    int32_t GetW0Rows() const { return 2 * N1; }
    int32_t GetW0Cols() const { return 2 * N1; }

    // GEMM 各轮尺寸（三轮相同）
    int32_t GetGemmM(int32_t) const { return 2 * N1; }   // 32
    int32_t GetGemmK(int32_t) const { return 2 * N1; }   // 32
    int32_t GetGemmN(int32_t) const { return 2 * N2 * N2; }  // 512（双信号 SPLIT_N）
};

inline int32_t CeilDiv(int32_t a, int32_t b) { return (a + b - 1) / b; }
inline int64_t CeilDiv(int64_t a, int64_t b) { return (a + b - 1) / b; }
inline int32_t RoundUp64(int32_t v) { return (v + 63) & ~63; }

} // namespace Fft

#endif // FFT_TYPES_H