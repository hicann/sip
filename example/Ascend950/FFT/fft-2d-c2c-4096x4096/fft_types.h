#ifndef FFT_TYPES_H
#define FFT_TYPES_H

#include <cstdint>
#include <cmath>
#include <stdexcept>
#include <string>
#include "fft_tiling_def.h"

// 序列常量与分解参数（host 侧推导，与 tiling/kernel 共享）。
// **stage 化形态（对齐概要设计 Part A §2.3）**：RadixDecomp 产出 stage 清单
// （每级 type/radix/族参数），GEMM shape 按各 Gemm stage 的 radix 推导——
// 不再有 mmM[3] 平铺数组与"两轮同基"的隐含假设。
// TODO 标记处为实例相关（每序列必改）；工具函数（CeilDiv/CheckFftN 等）跨序列不改。
namespace Fft {

constexpr double FFT_PI = 3.14159265358979323846;
constexpr double FFT_2PI = 2.0 * FFT_PI;

// 本工程固定信号长度（N=4096，与 kernel 文件名一致；一工程一序列）
constexpr int64_t FFT_N_REQUIRED = 4096;

// radix 策略：编译期唯一生效（一工程一序列，不做运行时切换）
// final（目标形态 2×16×16×2×2×2）：首级 Butterfly 2 + Gemm 16/16 + 末级连续 3 级 radix-2（L3/L4/L5 归并 stage[3]）
enum class FftRadix {
    Radix2x16x16x2x2x2,   // Butterfly 2 → Gemm 16 → Gemm 16 → Butterfly 2（L3/L4/L5 归并）
};

constexpr FftRadix FFT_RADIX = FftRadix::Radix2x16x16x2x2x2;

// 当前生效策略名（编译期求值，log 用）
inline const char* FftRadixName()
{
    return "2x16x16x2x2x2";
}

// 信号长度校验：非本工程长度直接报错
inline void CheckFftN(int64_t fftN)
{
    // 运行时可变轴（_N_）时改为按 tiling.signalLen[] 校验，不用 constexpr 拦截
    if (fftN != FFT_N_REQUIRED) {
        throw std::invalid_argument(
            "FFT: only fftN=" + std::to_string(FFT_N_REQUIRED) +
            " is supported, got " + std::to_string(fftN));
    }
}

// 序列分解（stage 化）：Decompose 填 stage 清单，Get* 按 stage 查询
struct RadixDecomp {
    int32_t stageCount;
    FftStageDesc stage[FFT_MAX_STAGE];   // 每级：type + radix + 族参数（mm/偏移由 tiling 填）

    RadixDecomp() : stageCount(0) {}

    void Decompose(int64_t fftN)
    {
        // final（目标形态）：Butterfly 2 → Gemm 16 → Gemm 16 → Butterfly 2（L3/L4/L5 连续 3 级 radix-2 归并）
        // 族参数 mm*/偏移不在此填，由 ComputeTiling/plan 按 stage 布局计算
        (void)fftN;
        stageCount = 4;
        stage[0].type = FftStageType::Butterfly;
        stage[0].radix = 2;
        stage[1].type = FftStageType::Gemm;
        stage[1].radix = 16;
        stage[2].type = FftStageType::Gemm;
        stage[2].radix = 16;
        stage[3].type = FftStageType::Butterfly;
        stage[3].radix = 2;   // L3/L4/L5 归并；T3/T4/T5 固化代码不占 coeffs
    }

    // —— 按 stage 查询（tiling/plan 消费）——
    int32_t GetGemmCount() const
    {
        // GEMM 轮数 = Σ(type==Gemm) 的 stage 数
        int32_t n = 0;
        for (int32_t i = 0; i < stageCount; i++) {
            if (stage[i].type == FftStageType::Gemm) { n++; }
        }
        return n;
    }

    // 各 Gemm stage 的实展开 MMAD 形状（M=K=2r；N 侧含双信号 ×2，由 tiling 乘）
    // 口径 ref_fft_math §5：mmM = mmK = 2r，mmN = N/r（每信号组数）
    int32_t GemmStageRadix(int32_t gemmIdx) const
    {
        // 按 stage 序数 Gemm，返回第 gemmIdx 个 Gemm stage 的 radix
        int32_t n = 0;
        for (int32_t i = 0; i < stageCount; i++) {
            if (stage[i].type == FftStageType::Gemm) {
                if (n == gemmIdx) { return stage[i].radix; }
                n++;
            }
        }
        return 0;
    }
    int32_t GetMmM(int32_t gemmIdx) const { return 2 * GemmStageRadix(gemmIdx); }
    int32_t GetMmK(int32_t gemmIdx) const { return GetMmM(gemmIdx); }
    int32_t GetMmN(int32_t gemmIdx) const
    {
        return static_cast<int32_t>(FFT_N_REQUIRED / GemmStageRadix(gemmIdx));
    }

    // coeffs 尺寸（W 系按 stage 去重 + T 系按级间，含 Butterfly 尾随 twiddle）
    // 级间 T_i = W_den^(组·k_i) 逐级间一张，大小 = 2×r_i×组数_i floats（复表实虚各一份）：
    //   组数_i = N / (r_0·…·r_i)（第 i+1 级前的组数）：
    //   T0（Butterfly 尾随）= 2×2×2048 = 8192 floats（32KB，W_4096^(m·k0)，规范形）
    //   T1 = 2×16×128 = 4096 floats（16KB，W_2048^(m2·k1)）
    //   T2 = 2×16×8 = 256 floats（1KB，W_128^(m'·k2)）
    //   T3/T4/T5（L3/L4/L5 蝶形旋转）r² 级常数规模，固化在代码/寄存器，不占 coeffs（详A §2.2.6 / 详B §3.1）
    int32_t GetTFloats(int32_t i) const
    {
        int32_t acc = 1;
        for (int32_t j = 0; j <= i; j++) { acc *= stage[j].radix; }   // 前 i+1 级 radix 乘积
        return 2 * stage[i].radix * static_cast<int32_t>(FFT_N_REQUIRED / acc);
    }
    // 级间 T_i 的 BuildTMatrix 参数（host 构造用）：N1=组数_i、N2=当前级 radix、den=W 的 N
    int32_t GetTGroups(int32_t i) const
    {
        int32_t acc = 1;
        for (int32_t j = 0; j <= i; j++) { acc *= stage[j].radix; }
        return static_cast<int32_t>(FFT_N_REQUIRED / acc);
    }
    int32_t GetTDenominator(int32_t i) const
    {
        int32_t acc = 1;
        for (int32_t j = 0; j < i; j++) { acc *= stage[j].radix; }   // 前 i 级 radix 乘积
        return static_cast<int32_t>(FFT_N_REQUIRED / acc);
    }
    int32_t GetTMatrixCount() const { return stageCount - 1; }   // 级间数（末级无）
};

// T0 紧凑表尺寸（coeffs/tiling 布局）：kernel 只消费 k0=1 流，规范形 8192 floats → 紧凑 4096 floats。
//   与 fft_twiddle.h BuildT0TwiddleCompact 同源；GetCoeffsSize/FillDeviceTiling/fft_plan 构建三处按此对齐。
constexpr int32_t T0_COMPACT_FLOATS = 4096;

// T2 全展开表尺寸（coeffs/tiling 布局，P6 步① 衔接 twiddle）：按 P6 输出平面（8 行 m2 × 512 列 r）预展开，
//   Re 段 4096 + Im 段 4096 = 8192 floats（紧凑规范形 256 floats → 全展开 8192，空间换时间消除 T2 侧 Gather）。
//   与 fft_twiddle.h BuildT2Expanded 同源；GetCoeffsSize/FillDeviceTiling/fft_plan 构建三处按此对齐。
constexpr int32_t T2_EXPANDED_FLOATS = 8192;

inline int32_t CeilDiv(int32_t a, int32_t b) { return (a + b - 1) / b; }
inline int64_t CeilDiv(int64_t a, int64_t b) { return (a + b - 1) / b; }

} // namespace Fft

#endif // FFT_TYPES_H