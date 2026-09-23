#ifndef FFT_TWIDDLE_H
#define FFT_TWIDDLE_H

#include <cstdint>
#include <cmath>
#include <vector>
#include "fft_types.h"

namespace Fft {

inline void BuildWMatrix(float* hostW, int32_t N, int32_t direction)
{
    int32_t dim = 2 * N;
    double sign = static_cast<double>(direction);
    double scale = (direction == 1) ? (1.0 / static_cast<double>(N)) : 1.0;

    for (int32_t k = 0; k < N; k++) {
        for (int32_t n = 0; n < N; n++) {
            double angle = FFT_2PI * static_cast<double>(k * n) / static_cast<double>(N);
            double cosVal = std::cos(angle);
            double sinVal = std::sin(angle) * sign;

            hostW[k * dim + n] = static_cast<float>(cosVal * scale);
            hostW[k * dim + n + N] = static_cast<float>(-sinVal * scale);
            hostW[(k + N) * dim + n] = static_cast<float>(sinVal * scale);
            hostW[(k + N) * dim + n + N] = static_cast<float>(cosVal * scale);
        }
    }
}

inline void BuildTMatrix(float* hostT, int32_t N1, int32_t N2, int32_t direction,
                         int32_t denominator = 0)
{
    // denominator=0 时默认 N1*N2（64×64 语义：T = W_{N1*N2}^{k1·n2}）
    int32_t totalN = (denominator > 0) ? denominator : (N1 * N2);
    int32_t realFloats = N1 * N2;
    double sign = static_cast<double>(direction);

    for (int32_t k1 = 0; k1 < N1; k1++) {
        for (int32_t n2 = 0; n2 < N2; n2++) {
            double angle = FFT_2PI * static_cast<double>(k1 * n2) / static_cast<double>(totalN);
            hostT[k1 * N2 + n2] = static_cast<float>(std::cos(angle));
            hostT[realFloats + k1 * N2 + n2] = static_cast<float>(std::sin(angle) * sign);
        }
    }
}

// 16×16×16：按 [T12, T13, T23] 顺序写入 hostT（每个 32×16 复矩阵 = N1*N2 实 + N1*N2 虚）
//   T12[k3,n2] = W_256^{k3·n2}     （第 1 轮 GEMM0 后，广播到 n3）
//   T13[k3,n3] = W_4096^{k3·n3}    （跳级项，第 2 轮 GEMM1 后，广播到 k2）
//   T23[k2,n3] = W_256^{k2·n3}     （第 2 轮 GEMM1 后，广播到 k3）
inline void BuildTMatrix3(float* hostT, int32_t R, int32_t direction)
{
    int32_t single = (2 * R) * R;
    BuildTMatrix(hostT + 0 * single, R, R, direction, R * R);        // T12: denom=256
    BuildTMatrix(hostT + 1 * single, R, R, direction, R * R * R);    // T13: denom=4096
    BuildTMatrix(hostT + 2 * single, R, R, direction, R * R);        // T23: denom=256
}

// T12 展开表（P3 SIMD twiddle 用）：4096 实 + 4096 虚 = 8192 floats = 32KB。
// 按输出块 (n2, k3blk) 的 64 元素 m 展开（m = k3Local*16 + n3，k3Local=m>>4）：
//   j = (n2*4 + k3blk)*64 + m
//   T12e_re[j] = Re(T12[k3blk*4 + k3Local][n2])，Im 在 total 偏移后。
// 使 P3 的 twiddle 对 64 元素连续可 Reg::LoadAlign（ex101 模式）。
inline void BuildT12Expanded(float* hostT12E, int32_t R, int32_t direction)
{
    const int32_t single = (2 * R) * R;   // 512
    const int32_t half   = R * R;         // 256
    const int32_t total  = R * half;      // 4096
    std::vector<float> t(static_cast<size_t>(single));
    BuildTMatrix(t.data(), R, R, direction, R * R);   // T12, denom=256
    const float* t12Re = t.data();
    const float* t12Im = t.data() + half;

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

// P3 数据 Gather 的 64 项 lane 表：f(lane) = (lane>>4)*256 + (lane&15)。
inline void BuildP3LaneOffsets(uint32_t* hostP3Lane)
{
    for (int32_t lane = 0; lane < 64; lane++) {
        hostP3Lane[lane] = static_cast<uint32_t>((lane >> 4) * 256 + (lane & 15));
    }
}

// T2 展开表（P5 融合 SIMT 用）：4096 实 + 4096 虚 = 8192 floats = 32KB。
// 按"输出索引" i = n3*256 + k2*16 + k3 排布（与 dst/B2 布局一致）：
//   T2e_re[i] = Re(T23[k2,n3]·T13[k3,n3])，T2e_im[i] = Im(...)
// SIMT 里连续读 T2e[i]，省掉 T13 的 stride-16 读 + T2 计算 + 两级依赖。
inline void BuildT2Expanded(float* hostTE, int32_t R, int32_t direction)
{
    const int32_t single = (2 * R) * R;    // 512（单 T 矩阵：256 re + 256 im）
    const int32_t half   = R * R;          // 256
    const int32_t total  = R * half;       // 4096
    std::vector<float> t(static_cast<size_t>(3 * single));
    BuildTMatrix3(t.data(), R, direction); // 依序 T12/T13/T23
    const float* t13Re = t.data() + 1 * single;
    const float* t13Im = t.data() + 1 * single + half;
    const float* t23Re = t.data() + 2 * single;
    const float* t23Im = t.data() + 2 * single + half;

    for (int32_t i = 0; i < total; i++) {
        int32_t n3 = i / half;             // 输出行
        int32_t c  = i - n3 * half;        // k2*16+k3
        int32_t k3 = c & (R - 1);
        int32_t k2 = c >> 4;
        int32_t t23Idx = k2 * R + n3;
        int32_t t13Idx = k3 * R + n3;
        float aR = t23Re[t23Idx], aI = t23Im[t23Idx];
        float bR = t13Re[t13Idx], bI = t13Im[t13Idx];
        hostTE[i]         = aR * bR - aI * bI;   // Re(T23·T13)
        hostTE[total + i] = aR * bI + aI * bR;   // Im(T23·T13)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Radix-2 方案专用表（16 点 DFT 第 3 轮）
// ═══════════════════════════════════════════════════════════════════════════

// 16 点 bit-reverse 索引表（DIT 输入重排，与 verify_radix2_vs_gemm.py 一致）
//   br[0..15] = [0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15]
inline void BuildBitReverse16(int32_t* br)
{
    const int32_t rev4[16] = {0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15};
    for (int32_t i = 0; i < 16; i++) br[i] = rev4[i];
}

// radix-2 各层 twiddle 表（SIMD 预展开，每 twiddle 复制 64 份 = 一个向量重复）
// 4 层 butterfly：stage s (0..3) 有 half=2^s 个 twiddle，w[j] = exp(direction·2πi·j/step)
//   step = 2*half；总复 twiddle = 1+2+4+8 = 15。
// 排布（float）：
//   [0,  64)              : stage0 tw[0] = 1（64 份）
//   [64,  64+128)         : stage1 tw[0],tw[1]（各 64 份）
//   [192, 192+256)        : stage2 tw[0..3]
//   [448, 448+512)        : stage3 tw[0..7]
//   实部 960 floats + 虚部 960 floats = 1920 floats
// 说明：stage0 的 twiddle=1，SIMD 可跳过复乘；此处仍生成以简化索引。
// 说明：RADIX2_TW_REPS / RADIX2_TW_STAGE_CNT / RADIX2_TW_FLOATS 定义在 fft_tiling_def.h

// 16 点 bit-reverse 查找表（展开为 16 个位置，供 SIMD 查表用）
constexpr int32_t BR_TABLE[16] = {0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15};

// Phase A 的 Gather 偏移量（每 n3 一组，共 16 组，每组 256 个偏移量）
// 布局：offset[n3 * 256 + colBlk * 64 + lane] = k2*256 + k3*16 + n3
// 其中 colBlk=0..3, lane=0..63, k2 = colBlk*4 + (lane>>4), k3 = lane&15
// 已验证：Gather(vgather2) 使用元素偏移
inline void BuildPhaseAOffsets(uint32_t* hostOffsets)
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
inline void BuildRadix2Twiddles(float* hostTW, int32_t direction)
{
    int32_t count = 0;
    for (int32_t stage = 0; stage < 4; stage++) {
        int32_t half = 1 << stage;
        int32_t step = 2 * half;
        for (int32_t j = 0; j < half; j++) {
            double angle = FFT_2PI * static_cast<double>(j) / static_cast<double>(step);
            float re = static_cast<float>(std::cos(angle));
            float im = static_cast<float>(std::sin(angle) * static_cast<double>(direction));
            for (int32_t r = 0; r < RADIX2_TW_REPS; r++) {
                hostTW[count * RADIX2_TW_REPS + r]               = re;
                hostTW[RADIX2_TW_FLOATS + count * RADIX2_TW_REPS + r] = im;
            }
            count++;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// V2 C16 (2×16×16×8) 专用 twiddle 表
// ═══════════════════════════════════════════════════════════════════════════

// tw1a[k0, n1'] = W_32^{k0·n1'}  因式分解: W_4096^{k0·n0} = tw1a·tw1b
// tw1b[k0, n0'] = W_4096^{k0·n0'}   n0 = n1'·128 + n0'
inline void BuildV2Tw1(float* hostTw1, int32_t direction)
{
    double sign = static_cast<double>(direction);
    // tw1a: W_32^{k0·n1'}, 2×16
    for (int32_t k0 = 0; k0 < 2; k0++) {
        for (int32_t n1p = 0; n1p < 16; n1p++) {
            double angle = FFT_2PI * static_cast<double>(k0 * n1p) / 32.0;
            hostTw1[(k0 * 16 + n1p)] = static_cast<float>(std::cos(angle));
            hostTw1[32 + (k0 * 16 + n1p)] = static_cast<float>(std::sin(angle) * sign);
        }
    }
    // tw1b: W_4096^{k0·n0'}, 2×128
    for (int32_t k0 = 0; k0 < 2; k0++) {
        for (int32_t n0p = 0; n0p < 128; n0p++) {
            double angle = FFT_2PI * static_cast<double>(k0 * n0p) / 4096.0;
            hostTw1[64 + (k0 * 128 + n0p)] = static_cast<float>(std::cos(angle));
            hostTw1[64 + 256 + (k0 * 128 + n0p)] = static_cast<float>(std::sin(angle) * sign);
        }
    }
}

// tw2a[k1', n2] = W_256^{k1'·n2}  因式分解: W_2048^{k1'·n0'} = tw2a·tw2b, n0'=n2·8+n3
// tw2b[k1', n3] = W_2048^{k1'·n3}
inline void BuildV2Tw2(float* hostTw2, int32_t direction)
{
    double sign = static_cast<double>(direction);
    // tw2a: W_256^{k1'·n2}, 16×16
    for (int32_t k1p = 0; k1p < 16; k1p++) {
        for (int32_t n2 = 0; n2 < 16; n2++) {
            double angle = FFT_2PI * static_cast<double>(k1p * n2) / 256.0;
            hostTw2[(k1p * 16 + n2)] = static_cast<float>(std::cos(angle));
            hostTw2[256 + (k1p * 16 + n2)] = static_cast<float>(std::sin(angle) * sign);
        }
    }
    // tw2b: W_2048^{k1'·n3}, 16×8
    for (int32_t k1p = 0; k1p < 16; k1p++) {
        for (int32_t n3 = 0; n3 < 8; n3++) {
            double angle = FFT_2PI * static_cast<double>(k1p * n3) / 2048.0;
            hostTw2[512 + (k1p * 8 + n3)] = static_cast<float>(std::cos(angle));
            hostTw2[512 + 128 + (k1p * 8 + n3)] = static_cast<float>(std::sin(angle) * sign);
        }
    }
}

// tw3[k2, n3] = W_128^{k2·n3}, 16×8
inline void BuildV2Tw3(float* hostTw3, int32_t direction)
{
    double sign = static_cast<double>(direction);
    for (int32_t k2 = 0; k2 < 16; k2++) {
        for (int32_t n3 = 0; n3 < 8; n3++) {
            double angle = FFT_2PI * static_cast<double>(k2 * n3) / 128.0;
            hostTw3[(k2 * 8 + n3)] = static_cast<float>(std::cos(angle));
            hostTw3[128 + (k2 * 8 + n3)] = static_cast<float>(std::sin(angle) * sign);
        }
    }
}

// 8-pt radix-2 twiddle 预展开表（SIMD 用，每 twiddle 复制 64 份）
// 3 层：W_2^0=1（跳过）, W_4^0..1（2个）, W_8^0..3（4个）= 6 个非平凡 twiddle
// 总复数 = 6，实浮点 = 6×64×2 = 768 floats
inline void BuildRadix2_8Twiddles(float* hostTW, int32_t direction)
{
    double sign = static_cast<double>(direction);
    int32_t count = 0;
    for (int32_t stage = 0; stage < 3; stage++) {
        int32_t half = 1 << stage;
        int32_t step = 2 * half;
        for (int32_t j = 0; j < half; j++) {
            double angle = FFT_2PI * static_cast<double>(j) / static_cast<double>(step);
            float re = static_cast<float>(std::cos(angle));
            float im = static_cast<float>(std::sin(angle) * sign);
            // 跳过的 j=0 在 stage=0 （W_2^0=1），仍生成但SIMD可跳过复乘
            for (int32_t r = 0; r < 64; r++) {
                hostTW[count * 64 + r] = re;
                hostTW[768 + count * 64 + r] = im;
            }
            count++;
        }
    }
}

} // namespace Fft

#endif // FFT_TWIDDLE_H