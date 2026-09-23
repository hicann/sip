// ===========================================================================
// fft_radix16_twiddle.h — radix-16（16×16×16）系数构造（host 侧）
// 移植自 ex99 fft_twiddle.h，仅保留 16×16×16 路径：
//   W16(32×32 实/虚分离) + T12/T13/T23 + T2 展开表 + radix-2 twiddle + PhaseA 偏移表
// ===========================================================================

#ifndef FFT_RADIX16_TWIDDLE_H
#define FFT_RADIX16_TWIDDLE_H

#include <cstdint>
#include <cmath>
#include <vector>

namespace Fft {

constexpr double FFT_PI_16 = 3.14159265358979323846;
constexpr double FFT_2PI_16 = 2.0 * FFT_PI_16;

// radix-2 展开表尺寸（16 点 DFT 4 层，twiddle 4/2/1 ... 共 15 复 × 64 lane）
constexpr int32_t RADIX2_TW_REPS = 64;
constexpr int32_t RADIX2_TW_STAGE_CNT = 15;
constexpr int32_t RADIX2_TW_FLOATS = RADIX2_TW_STAGE_CNT * RADIX2_TW_REPS;  // 960

// Phase A SIMD Gather 偏移表：16(n3) × 256(k2,k3) = 4096 uint32
constexpr int32_t PHASE_A_OFFSET_COUNT = 16 * 256;

// 16 点 DFT W 矩阵：32×32（16 实 + 16 虚）。
// 注意：C2R 参考 torch irfft2(norm="forward")，逆变换无缩放（与 ex86 旧 radix-64 一致），
// 故不带 1/N 因子（ex99 C2C 的 BuildWMatrix 带 1/N，此处不适用）。
inline void BuildWMatrix16(float* hostW, int32_t direction)
{
    const int32_t N = 16;
    const int32_t dim = 2 * N;   // 32
    double sign = static_cast<double>(direction);

    for (int32_t k = 0; k < N; k++) {
        for (int32_t n = 0; n < N; n++) {
            double angle = FFT_2PI_16 * static_cast<double>(k * n) / static_cast<double>(N);
            double c = std::cos(angle);
            double s = std::sin(angle) * sign;
            hostW[k * dim + n]           = static_cast<float>(c);
            hostW[k * dim + n + N]       = static_cast<float>(-s);
            hostW[(k + N) * dim + n]     = static_cast<float>(s);
            hostW[(k + N) * dim + n + N] = static_cast<float>(c);
        }
    }
}

// 单个 T 矩阵（R×R 复 = 2*R*R floats），denominator 为归一化分母
inline void BuildTMatrixGeneric(float* hostT, int32_t R, int32_t direction, int32_t denominator)
{
    double sign = static_cast<double>(direction);
    int32_t realFloats = R * R;
    for (int32_t k = 0; k < R; k++) {
        for (int32_t n = 0; n < R; n++) {
            double angle = FFT_2PI_16 * static_cast<double>(k * n) / static_cast<double>(denominator);
            hostT[k * R + n]                = static_cast<float>(std::cos(angle));
            hostT[realFloats + k * R + n]  = static_cast<float>(std::sin(angle) * sign);
        }
    }
}

// T12/T13/T23 三张表，按 [T12, T13, T23] 顺序，每个 2*16*16=512 floats
inline void BuildTMatrix3_16(float* hostT, int32_t direction)
{
    const int32_t R = 16;
    const int32_t single = (2 * R) * R;   // 512
    BuildTMatrixGeneric(hostT + 0 * single, R, direction, R * R);        // T12 denom=256
    BuildTMatrixGeneric(hostT + 1 * single, R, direction, R * R * R);    // T13 denom=4096
    BuildTMatrixGeneric(hostT + 2 * single, R, direction, R * R);        // T23 denom=256
}

// T2 展开表：T2[i] = T23[k2,n3] * T13[k3,n3]，按输出索引 i=n3*256+k2*16+k3 排布
inline void BuildT2Expanded16(float* hostTE, int32_t direction)
{
    const int32_t R = 16;
    const int32_t single = (2 * R) * R;   // 512
    const int32_t half = R * R;           // 256
    const int32_t total = R * half;       // 4096
    std::vector<float> t(static_cast<size_t>(3 * single));
    BuildTMatrix3_16(t.data(), direction);
    const float* t13Re = t.data() + 1 * single;
    const float* t13Im = t.data() + 1 * single + half;
    const float* t23Re = t.data() + 2 * single;
    const float* t23Im = t.data() + 2 * single + half;

    for (int32_t i = 0; i < total; i++) {
        int32_t n3 = i / half;
        int32_t c = i - n3 * half;
        int32_t k3 = c & (R - 1);
        int32_t k2 = c >> 4;
        int32_t t23Idx = k2 * R + n3;
        int32_t t13Idx = k3 * R + n3;
        float aR = t23Re[t23Idx], aI = t23Im[t23Idx];
        float bR = t13Re[t13Idx], bI = t13Im[t13Idx];
        hostTE[i]         = aR * bR - aI * bI;
        hostTE[total + i] = aR * bI + aI * bR;
    }
}

// radix-2 各层 twiddle（SIMD 预展开，每 twiddle 复制 64 份）
inline void BuildRadix2Twiddles16(float* hostTW, int32_t direction)
{
    double sign = static_cast<double>(direction);
    int32_t count = 0;
    for (int32_t stage = 0; stage < 4; stage++) {
        int32_t half = 1 << stage;
        int32_t step = 2 * half;
        for (int32_t j = 0; j < half; j++) {
            double angle = FFT_2PI_16 * static_cast<double>(j) / static_cast<double>(step);
            float re = static_cast<float>(std::cos(angle));
            float im = static_cast<float>(std::sin(angle) * sign);
            for (int32_t r = 0; r < RADIX2_TW_REPS; r++) {
                hostTW[count * RADIX2_TW_REPS + r] = re;
                hostTW[RADIX2_TW_FLOATS + count * RADIX2_TW_REPS + r] = im;
            }
            count++;
        }
    }
}

// Phase A Gather 偏移表：offset[n3*256 + colBlk*64 + lane] = k2*256 + k3*16 + n3
inline void BuildPhaseAOffsets16(uint32_t* hostOffsets)
{
    for (int32_t n3 = 0; n3 < 16; n3++) {
        for (int32_t colBlk = 0; colBlk < 4; colBlk++) {
            for (int32_t lane = 0; lane < 64; lane++) {
                int32_t k2 = colBlk * 4 + (lane >> 4);
                int32_t k3 = lane & 15;
                int32_t idx = n3 * 256 + colBlk * 64 + lane;
                hostOffsets[idx] = static_cast<uint32_t>(k2 * 256 + k3 * 16 + n3);
            }
        }
    }
}

// P3 SIMD twiddle 用的 T12 展开表（8192 floats = 4096 re + 4096 im）
constexpr int32_t P3_T12_EXP_FLOATS = 8192;
constexpr int32_t P3_LANE_COUNT = 64;

inline void BuildT12Expanded16(float* hostT12E, int32_t direction)
{
    const int32_t R = 16;
    const int32_t single = (2 * R) * R;   // 512
    const int32_t half = R * R;           // 256
    const int32_t total = R * half;       // 4096
    std::vector<float> t12(static_cast<size_t>(single));
    BuildTMatrixGeneric(t12.data(), R, direction, R * R);   // T12 denom=256
    const float* t12Re = t12.data();
    const float* t12Im = t12.data() + half;

    for (int32_t n2 = 0; n2 < R; n2++) {
        for (int32_t k3blk = 0; k3blk < 4; k3blk++) {
            for (int32_t m = 0; m < 64; m++) {
                int32_t k3Local = m >> 4;
                int32_t k3 = k3blk * 4 + k3Local;
                int32_t tIdx = k3 * R + n2;
                int32_t j = (n2 * 4 + k3blk) * 64 + m;
                hostT12E[j]         = t12Re[tIdx];
                hostT12E[total + j] = t12Im[tIdx];
            }
        }
    }
}

// P3 数据 Gather 的 64 项 lane 表：f(lane) = (lane>>4)*256 + (lane&15)
inline void BuildP3LaneOffsets16(uint32_t* hostP3Lane)
{
    for (int32_t lane = 0; lane < 64; lane++) {
        hostP3Lane[lane] = static_cast<uint32_t>((lane >> 4) * 256 + (lane & 15));
    }
}

} // namespace Fft

#endif // FFT_RADIX16_TWIDDLE_H