#ifndef FFT_TILING_DEF_H
#define FFT_TILING_DEF_H

#include <cstdint>

constexpr uint32_t MAX_CORE_NUM = 64;

// radix-2 方案：SIMD 预展开 twiddle 表尺寸（16 点 DFT 4 层，每 twiddle 64 份）
constexpr int32_t RADIX2_TW_REPS = 64;       // 每 twiddle 复制份数（SIMD lane）
constexpr int32_t RADIX2_TW_STAGE_CNT = 15;  // 总 twiddle 数（1+2+4+8）
constexpr int32_t RADIX2_TW_FLOATS = RADIX2_TW_STAGE_CNT * RADIX2_TW_REPS;  // 960

// Phase A SIMD Gather 偏移表：16(n3) × 256(k2=0..15,k3=0..15) = 4096 uint32 = 16KB
constexpr int32_t PHASE_A_OFFSET_COUNT = 16 * 256;  // 4096
constexpr int32_t PHASE_A_OFFSET_FLOATS = PHASE_A_OFFSET_COUNT;  // 按 float 计数（16KB）

// P3 SIMD twiddle 展开表 + 数据 Gather lane 表（ex101 模式）
constexpr int32_t P3_T12_EXP_FLOATS = 4096 * 2;  // T12 展开：4096 实 + 4096 虚 = 32KB
constexpr int32_t P3_LANE_COUNT = 64;            // f(lane)=(lane>>4)*256+(lane&15)

class FftInputParams {
public:
    int64_t fftN;
    int32_t N1;
    int32_t N2;
    int32_t direction;
    int32_t batch;
    int32_t iterCount;

    int32_t mmM[3];
    int32_t mmK[3];
    int32_t mmN[3];

    int64_t wMatrixOffset[3];
    int64_t tMatrixOffset;
    int64_t tExpandedOffset;
    int64_t tRadix2Offset;
    int64_t tPhaseAOffset;
    int64_t t12ExpandedOffset;
    int64_t tP3LaneOffset;

    int32_t ubMode;
    int32_t mixMode;
};

class FftMultiCoreParams {
public:
    int32_t coreNum;
    int32_t totalRows;
    int32_t rowsPerCore;
    int32_t rowRemainder;

    uint32_t rowStartIdx[MAX_CORE_NUM + 1];

    int64_t workspaceOffset[14];
};

class FftTilingData {
public:
    FftInputParams inputParams;
    FftMultiCoreParams multiCoreParams;
};

#endif // FFT_TILING_DEF_H