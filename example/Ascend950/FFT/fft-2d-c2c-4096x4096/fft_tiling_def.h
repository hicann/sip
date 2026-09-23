#ifndef FFT_TILING_DEF_H
#define FFT_TILING_DEF_H

#include <cstdint>

// 【共享契约①】Device 侧 tiling 结构体：host 填（FillDeviceTiling）、kernel 读（InitContext）。
// **结构对齐概要设计 Part A §2.3（stage 化通用形态）**——字段/枚举/族参数为设计契约的
// 直译，实现不得偏离；通用形态支持：任意 radix 序列（stageCount 无三轮上限）、
// 蝶形族（Butterfly，AIV）、任意信号轴（signalAxis[]）、运行时可变轴长（signalLen[]）。
//
// 与平铺简化形态（旧 mmM[3]/wMatrixOffset[3]，纯 GEMM 同基两轮序列适用）的差异：
// stage[] 数组承载任意级数与异构族；GEMM 轮数 = Σ(type==Gemm) 的 stage 数。
constexpr uint32_t MAX_CORE_NUM = 64;
constexpr uint32_t FFT_MAX_RANK = 8;          // 输入 Tensor 最大维度数（占位上界，按算子需求定）
constexpr uint32_t FFT_MAX_SIGNAL_DIM = 3;    // 最大信号轴个数（1D/2D/3D）
constexpr uint32_t FFT_MAX_STAGE = 16;        // 最大算法阶段数（占位上界：2^k 级联折进一个 Butterfly stage）

// 步⑤ interleave+bitrev SIMD gather 偏移表尺寸（coeffs float 口径）：64 个 uint32_t 元素偏移
//   （仅存 64 个 lane 的 M[lane]，base(blk) 由 kernel 标量现算广播；1 uint32 = 1 float 槽，共 64 floats = 256B）
constexpr int32_t GATHER_TABLE_FLOATS = 64;

// 步① T2 复乘 SIMD gather 表尺寸（coeffs float 口径，GATHER_TABLE 之后）：
//   P6T2_OFF（64 uint32）：C1 源偏移 M_OFF'[lane] = (lane>>5)·256 + (lane&1)·128 + ((lane>>1)&15)·8
//   P6T2_TW（64 uint32）：T2 twiddle 索引 T2_TW[lane] = lane&15（T2 全展开后弃用）
//   各 64 uint32 = 64 float 槽；两表合计 128 floats = 512B；仅 P6T2_OFF 前半被 simd_p6_t2_expanded 消费
constexpr int32_t P6T2_TABLE_FLOATS = 128;

// P4 twiddle SIMD gather 表尺寸（coeffs float 口径，P6T2 之后）：
//   P4_OFF（64 uint32）：C0 源偏移 M_OFF[lane] = (lane>>3)·256 + (lane&7)
//   P4_TW（64 uint32）：T1 twiddle 索引 TW_OFF[lane] = (lane&7)·16 + (lane>>3)
//   各 64 uint32 = 64 float 槽；两表合计 128 floats = 512B
constexpr int32_t P4_TABLE_FLOATS = 128;

// 每级算法族：单元隐含（Gemm→AIC；Butterfly→AIV）
enum class FftStageType : int32_t {
    Gemm,      // 矩阵族：AIC GEMM（典型 radix≥16）；AIV 向量点积复用时可置 mm=1、opOffset 指展开表
    Butterfly, // 蝶形族：仅 AIV（radix 2/4，或 2^k 级联折进一个 stage）
};

struct GemmStageParams {
    int32_t mmM, mmK, mmN;   // AIC MMAD 形状（AIV 向量路径时置 1）
    int64_t opOffset;        // W 矩阵 / 向量展开表偏移（相对 coeffs 基址，字节）
    int64_t tOffset;         // T（twiddle 复乘）表偏移，无则置 0
    int64_t bOffset;         // B 矩阵段在该核 workspace 区内的基址偏移（字节）；乒乓由 (pairIdx&1)×stride 现算
};

struct ButterflyStageParams {
    int32_t layers;          // 级联层数（radix=2^k → k；radix=4 → 2）
    int64_t twiddleOffset;   // 连续蝶形级共享 twiddle 表偏移
    int64_t phaseOffset;     // 重排/gather 表偏移，可省置 0
};

struct FftStageDesc {
    FftStageType type;   // 算法族（单元隐含：Gemm→AIC、Butterfly→AIV）
    int32_t radix;       // 该级基数
    union { GemmStageParams gemm; ButterflyStageParams butterfly; };
};

class FftInputParams {               // 计算侧：信号描述 + 每级阶段描述
public:
    // —— 信号描述（与 FftPlan ① 入参一致，plan 创建时写入 tiling）——
    int32_t inputRank;                      // 输入 Tensor 维度数
    int64_t inputShape[FFT_MAX_RANK];       // 输入 Tensor 各维长度（含 batch 与全部非信号维）
    int32_t signalDims;                     // 信号轴个数（= FFT 维度 1D/2D/3D）
    int32_t signalAxis[FFT_MAX_SIGNAL_DIM]; // 信号所在轴下标（任意轴，不限于尾轴）
    int64_t signalLen[FFT_MAX_SIGNAL_DIM];  // 各信号轴 FFT 长度（_N_ 轴为运行时值，经此传入）
    int32_t direction;                      // forward(-1) / inverse(+1)
    int32_t ubMode;                         // 0=GM roundtrip, 1=UB->L1 direct（host 传参；kernel InitContext 读）

    // —— 算法阶段描述：stageCount = radix 因子个数（非 GEMM 轮数）——
    int32_t stageCount;                  // 算法阶段数；GEMM 轮数 = Σ(type==Gemm)
    FftStageDesc stage[FFT_MAX_STAGE];   // 每级：类型 + radix + 族参数（含各自 coeffs 段偏移）

    // —— 其余全局 coeffs 段偏移（跨级共享 / 按变体增补，无则置 0）——
    int64_t tExpandedOffset;             // T_expanded（向量阶段展开表）段偏移
    int64_t gatherTableOffset;           // 步⑤ SIMD gather 偏移表（GATHER_TABLE_FLOATS，coeffs 末尾）段偏移，无则置 0
    int64_t p6t2TableOffset;             // 步① T2 复乘 SIMD gather 表（P6T2_TABLE_FLOATS，coeffs 末尾 GATHER_TABLE 之后）段偏移，无则置 0
    int64_t p4TableOffset;               // P4 twiddle SIMD gather 表（P4_TABLE_FLOATS，coeffs 末尾 P6T2 之后）段偏移，无则置 0
};

class FftMultiCoreParams {           // 多核侧：核划分 + workspace 区布局
public:
    int32_t coreNum, totalRows, rowsPerCore, rowRemainder;   // totalRows = 合并相邻非信号轴后的信号平面数
    uint32_t rowStartIdx[MAX_CORE_NUM + 1];
    int64_t wsRegionBytes;           // 每核 workspace 区大小（含各 stage B 段及乒乓份；段基址在各 stage.gemm.bOffset）
};

class FftTilingData {
public:
    FftInputParams inputParams;
    FftMultiCoreParams multiCoreParams;
};

#endif // FFT_TILING_DEF_H