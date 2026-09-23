#ifndef FFT_TILING_DEF_H
#define FFT_TILING_DEF_H

#include <cstdint>

constexpr uint32_t MAX_CORE_NUM = 64;

// radix-16 C2R 2D 设备侧 tiling 参数。
// 布局必须与 host 侧编译单元逐字节一致。
//
// 流水线（逆序瘦列方案）：
//   input  [batch, fftN, padHalf] complex（半谱，padHalf 补齐到 64 倍）
//   → S1 转置  -> Xt[padHalf, fftN] complex
//   → S2 C2C 逆变换（axis0，padHalf 条） -> Y[padHalf, fftN]
//   → S3 转置  -> Yt[fftN, padHalf] complex
//   → S4 irfft（axis1，fftN 条，Hermitian 展开 + IDFT + 取实） -> output[fftN, fftN] real

class FftC2R2DInputParams {
public:
    int64_t fftN;              // 4096
    int32_t batch;
    int32_t direction;         // +1 inverse（C2R = irfft2）
    int32_t N1;                // 16
    int32_t N2;                // 16
    int32_t iterCount;         // 3
    int32_t halfN;             // 2049
    int32_t padHalf;           // 2112（64 倍数）

    int32_t mmM;               // 32
    int32_t mmK;               // 32
    int32_t mmN;               // 512（双信号 SPLIT_N）

    // 系数偏移（bytes）
    int64_t wOffset;           // W16 (32×32)
    int64_t tOffset;           // T12/T13/T23
    int64_t tExpandedOffset;   // T2 展开表
    int64_t tRadix2Offset;     // radix-2 twiddle
    int64_t tPhaseAOffset;     // PhaseA Gather 偏移表
    int64_t t12ExpandedOffset; // P3 T12 展开表
    int64_t tP3LaneOffset;     // P3 Gather lane 表
};

class FftC2R2DMultiCoreParams {
public:
    int32_t coreNum;
    int32_t skinnyRows;        // batch * padHalf（S2 的行数）
    int32_t fullRows;          // batch * fftN（S4 的输出行数）
    int32_t mergedRows;        // fullRows / 2（S4 共轭合并的 dual-row 数）
    uint32_t skinnyRowStartIdx[MAX_CORE_NUM + 1];
    uint32_t fullRowStartIdx[MAX_CORE_NUM + 1];
    uint32_t mergedRowStartIdx[MAX_CORE_NUM + 1];
};

class FftC2R2DTilingData {
public:
    FftC2R2DInputParams inputParams;
    FftC2R2DMultiCoreParams multiCoreParams;
    int32_t stage;             // 0/1 = 兼容旧接口（S1..S3 部分），2 = full pipeline
};

#endif // FFT_TILING_DEF_H
