#ifndef FFT_TWIDDLE_H
#define FFT_TWIDDLE_H

#include <cstdint>
#include <cmath>
#include <vector>
#include "fft_types.h"

// W/T 矩阵构造（host 端纯函数，已验证实展开口径）。
// W 实展开 [[Re, −Im], [Im, Re]]（2N×2N）；T 为紧凑复表（前 N1*N2 实、后 N1*N2 虚）。
// 排布必须与 GEMM 实展开口径一致（gemm-usage-guide §1）。
namespace Fft {

inline void BuildWMatrix(float* hostW, int32_t N, int32_t direction)
{
    int32_t dim = 2 * N;
    double sign = static_cast<double>(direction);
    double scale = 1.0;   // norm="forward"：正/反变换均不归一化（方向仅影响指数符号）

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
    // denominator=0 时默认 N1*N2（两级分解语义：T = W_{N1*N2}^{k1·n2}）
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

// T0（Butterfly 尾随 twiddle）紧凑表：只含 kernel 消费的 k0=1 流（规范形 8192 floats → 4096 floats）。
// 规范形 T0 = W_4096^(m·k0)（k0∈[0,2)、m∈[0,2048)），BuildTMatrix 排布为
//   Re(k0,m)=tw[m*2+k0]、Im(k0,m)=tw[4096+m*2+k0]；kernel 只读 k0=1 流（tw[m*2+1]/tw[4096+m*2+1]）。
// 紧凑表（dst[4096]）：dst[m]=Re(k0=1,m)=cos(2πm/4096)、dst[2048+m]=Im(k0=1,m)=sin(2πm/4096)·sign。
// 消费端 simd_p1_butterfly 读 dst[m]（Re）/dst[2048+m]（Im）。
inline void BuildT0TwiddleCompact(float* dst, int32_t direction)
{
    constexpr int32_t M = 2048;   // 组数（m∈[0,2048)）
    constexpr int32_t N = 4096;   // W 的 N（W_4096^m）
    double sign = static_cast<double>(direction);
    for (int32_t m = 0; m < M; m++) {
        double angle = FFT_2PI * static_cast<double>(m) / static_cast<double>(N);
        dst[m] = static_cast<float>(std::cos(angle));
        dst[M + m] = static_cast<float>(std::sin(angle) * sign);
    }
}

// 展开表（向量 twiddle 用）：按数据索引逐元素排布，空间换一次整块向量 Mul。
// 排布公式必须与 kernel 侧消费索引逐元素一致。

// P6 radix-2 蝶形 twiddle 展开表（SIMD 用，每 twiddle 复制 VEC=64 份成向量）：
//   Re 段在前（448 floats）、Im 段在后（448 floats），总 896 floats；
//   段内级序 L3(W_8^0..3) → L4(W_4^0..1) → L5(W_2^0=1，免乘) 共 7 个 twiddle
//   消费端 simd_radix2_l3 / simd_radix2_l4 按各自段偏移读取
inline void BuildP6Radix2TwiddleExpanded(float* hostTE, int32_t direction)
{
    constexpr int32_t VEC = 64;
    constexpr int32_t RE = (4 + 2 + 1) * VEC;   // 448（Re 段在前，Im 段在后）
    // 级定义 {twiddle 个数, W 的 N}：L3 W_8(4)、L4 W_4(2)、L5 W_2(1)
    const int32_t cnt[3] = {4, 2, 1};
    const int32_t N[3] = {8, 4, 2};
    double sign = static_cast<double>(direction);
    int32_t idx = 0;
    for (int32_t s = 0; s < 3; s++) {
        for (int32_t j = 0; j < cnt[s]; j++) {
            double angle = FFT_2PI * static_cast<double>(j) / static_cast<double>(N[s]);
            float cr = static_cast<float>(std::cos(angle));
            float ci = static_cast<float>(std::sin(angle) * sign);
            for (int32_t v = 0; v < VEC; v++) {
                hostTE[idx * VEC + v] = cr;
                hostTE[RE + idx * VEC + v] = ci;
            }
            idx++;
        }
    }
}

// 步⑤ gather 偏移表段（GATHER_TABLE_FLOATS=64 个 uint32_t，coeffs 末尾）：
//   DIT 版步⑤ 为纯 interleave（零 Gather），kernel 不消费本表；保留构造以维持 coeffs 段序与偏移口径。
//   M[lane] = (lane&1)·4096 + ((lane>>1)&1)·256 + ((lane>>2)&15)·16（Re/Im 平面基址 4096 分档，单次 Gather 覆盖双平面）
inline void BuildInterleaveBitrevGatherTable(uint32_t* dst)
{
    for (int32_t lane = 0; lane < GATHER_TABLE_FLOATS; lane++) {
        dst[lane] = static_cast<uint32_t>(
            (lane & 1) * 4096 + ((lane >> 1) & 1) * 256 + ((lane >> 2) & 15) * 16);
    }
}

// P6 步① T2 复乘 SIMD gather 表（P6T2_TABLE_FLOATS=128 个 uint32_t，coeffs 末尾 GATHER_TABLE 之后）：
//   前半 P6T2_OFF（64）：C1 源偏移 M_OFF'[lane] = (lane>>5)·256 + (lane&1)·128 + ((lane>>1)&15)·8
//     【DIT 物理块口径】blk = 物理块（= k2>>1）：lane → k0=lane&1、k1=(lane>>1)&15、k2=2·blk+(lane>>5)；
//     base(blk,m2) = 512·blk + m2 由 kernel 标量现算广播；srcOff = k2·256 + k0·128 + k1·8 + m2
//   后半 P6T2_TW（64）：T2 twiddle 索引 T2_TW[lane] = lane&15（T2 改全展开表后不消费，仅占位维持段长）
//   仅 P6T2_OFF 前半被 simd_p6_t2_expanded 消费：srcOff_Re = base + M_OFF'[lane]、
//   srcOff_Im = srcOff_Re + 4096（Im 行 k2+16）。
inline void BuildP6T2GatherTables(uint32_t* dst)
{
    constexpr int32_t LANES = 64;
    for (int32_t lane = 0; lane < LANES; lane++) {
        dst[lane] = static_cast<uint32_t>(
            (lane >> 5) * 256 + (lane & 1) * 128 + ((lane >> 1) & 15) * 8);
        dst[LANES + lane] = static_cast<uint32_t>(lane & 15);
    }
}

// T2 全展开表（P6 步① T2 复乘，按输出平面预展开——空间换时间，消除 T2 侧 8 次 Gather）：
//   【DIT 口径】排布按 P6 Re/Im 分离平面的**物理列** rp ∈ [0,512)（数学列布局 [k0|k1|k2]：
//     rp bit0=k0、bit1-4=k1、bit5-8=k2）：T2Exp[m2·512 + rp] = W_128^(m2·k2(rp))，k2(rp) = (rp>>5)&15。
//   Re 段 4096 floats 在前、Im 段 4096 在后（总 8192 floats = T2_EXPANDED_FLOATS，32KB）。
//   数学等价：W_128^(m2·k2)，k2 为逻辑 k2（r = k0·256+k1·16+k2 的低 4 位 = 物理列 rp 的 bit5-8）。
//   消费端 simd_p6_t2_expanded 按物理列连续 LoadAlign（零 Gather）。
inline void BuildT2Expanded(float* hostTE, int32_t direction)
{
    constexpr int32_t M2 = 8;         // 平面行 m2 ∈ [0,8)
    constexpr int32_t COL = 512;      // 物理列 rp ∈ [0,512)
    constexpr int32_t RE = M2 * COL;  // Re 段 4096
    double sign = static_cast<double>(direction);
    for (int32_t m2 = 0; m2 < M2; m2++) {
        for (int32_t rp = 0; rp < COL; rp++) {
            int32_t k2 = (rp >> 5) & 15;   // 物理列 rp 的 bit5-8 = 逻辑 k2（数学列布局）
            double angle = FFT_2PI * static_cast<double>(m2 * k2) / 128.0;
            hostTE[m2 * COL + rp] = static_cast<float>(std::cos(angle));
            hostTE[RE + m2 * COL + rp] = static_cast<float>(std::sin(angle) * sign);
        }
    }
}

// P4 twiddle SIMD gather 表（P4_TABLE_FLOATS=128 个 uint32_t，coeffs 末尾 P6T2 之后）：
//   前半 P4_OFF（64）：C0 源偏移 M_OFF[lane] = (lane>>3)·256 + (lane&7)（C0 行内列偏移分量；
//     base(blk) = (blk&1)·2048 + (blk&2)·64 + ((blk>>2)&15)·8 由 kernel 标量现算广播）
//   后半 P4_TW（64）：T1 twiddle 索引 TW_OFF[lane] = (lane&7)·16 + (lane>>3)
//     （twIdx = baseTw(blk) + TW_OFF[lane]，baseTw(blk) = ((blk>>2)&15)·128 + (blk&1)·8）
//   消费端 simd_twiddle_rotate_r4：srcOff_Re = base + M_OFF[lane]、srcOff_Im = srcOff_Re + 4096（Im 行 k1+16）。
inline void BuildP4GatherTables(uint32_t* dst)
{
    constexpr int32_t LANES = 64;
    for (int32_t lane = 0; lane < LANES; lane++) {
        dst[lane] = static_cast<uint32_t>(
            (lane >> 3) * 256 + (lane & 7));
        dst[LANES + lane] = static_cast<uint32_t>(
            (lane & 7) * 16 + (lane >> 3));
    }
}

} // namespace Fft

#endif // FFT_TWIDDLE_H
