#ifndef FFT_TYPES_H
#define FFT_TYPES_H

#include <cstdint>
#include <cmath>
#include <stdexcept>
#include <string>

namespace Fft {

constexpr double FFT_PI = 3.14159265358979323846;
constexpr double FFT_2PI = 2.0 * FFT_PI;

// 定制化：该算子仅支持 4096 点 FFT
constexpr int64_t FFT_N_REQUIRED = 4096;

// radix 策略：编译期唯一生效，切换策略只需修改 FFT_RADIX（无需动态传参）
enum class FftRadix {
    Radix64x64,      // 64 x 64，两轮分解
    Radix16x16x16,   // 16 x 16 x 16，三轮分解（预留）
};

constexpr FftRadix FFT_RADIX = FftRadix::Radix16x16x16;

// 当前生效策略名（编译期求值）
inline const char* FftRadixName()
{
    if constexpr (FFT_RADIX == FftRadix::Radix64x64) {
        return "64x64";
    } else if constexpr (FFT_RADIX == FftRadix::Radix16x16x16) {
        return "16x16x16";
    } else {
        return "unknown";
    }
}

// 信号长度校验：非 4096 直接报错
inline void CheckFftN(int64_t fftN)
{
    if (fftN != FFT_N_REQUIRED) {
        throw std::invalid_argument(
            "FFT: only fftN=" + std::to_string(FFT_N_REQUIRED) +
            " is supported, got " + std::to_string(fftN));
    }
}

struct RadixDecomp {
    int32_t N1;
    int32_t N2;
    int32_t iterCount;

    RadixDecomp() : N1(0), N2(0), iterCount(0) {}

    void Decompose(int64_t fftN)
    {
        CheckFftN(fftN);
        if constexpr (FFT_RADIX == FftRadix::Radix64x64) {
            N1 = 64;
            N2 = 64;
            iterCount = 2;
        } else if constexpr (FFT_RADIX == FftRadix::Radix16x16x16) {
            N1 = 16;
            N2 = 16;
            iterCount = 3;
        }
    }

    int32_t GetWMatrixRows(int32_t iter) const
    {
        return (iter == 0) ? (2 * N1) : (2 * N2);
    }

    int32_t GetWMatrixCols(int32_t iter) const
    {
        return (iter == 0) ? (2 * N1) : (2 * N2);
    }

    int32_t GetTMatrixRows() const { return 2 * N1; }
    int32_t GetTMatrixCols() const { return N2; }

    // 单 T 矩阵尺寸（floats），多 T 时乘以 GetTMatrixCount()
    int32_t GetTSingleFloats() const { return (2 * N1) * N2; }

    // 16×16×16 需要 3 个 16×16 复 twiddle（T1(k3,n2)/T2a(k3,n3)/T2b(k2,n3)）
    int32_t GetTMatrixCount() const
    {
        if constexpr (FFT_RADIX == FftRadix::Radix16x16x16) {
            return 3;
        } else {
            return 1;
        }
    }

    int32_t GetMmM(int32_t iter) const { return (iter == 0) ? (2 * N1) : (2 * N2); }
    int32_t GetMmK(int32_t iter) const { return (iter == 0) ? (2 * N1) : (2 * N2); }
    int32_t GetMmN(int32_t iter) const
    {
        if constexpr (FFT_RADIX == FftRadix::Radix16x16x16) {
            (void)iter;   // 三轮相同：每信号每轮复 N = n2×n3 等 = N2×N2 = 256
            return N2 * N2;
        } else {
            return (iter == 0) ? N2 : N1;
        }
    }
};

inline int32_t CeilDiv(int32_t a, int32_t b)
{
    return (a + b - 1) / b;
}

inline int64_t CeilDiv(int64_t a, int64_t b)
{
    return (a + b - 1) / b;
}

} // namespace Fft

#endif // FFT_TYPES_H