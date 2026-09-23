#include "fft_tiling.h"
#include "fft_twiddle.h"

namespace Fft {

size_t GetCoeffsSize(int64_t fftN)
{
    RadixDecomp decomp;
    decomp.Decompose(fftN);

    int64_t w0Size = static_cast<int64_t>(decomp.GetWMatrixRows(0)) * decomp.GetWMatrixCols(0);
    int64_t tSize = static_cast<int64_t>(decomp.GetTSingleFloats()) * decomp.GetTMatrixCount();

    int64_t total = w0Size + tSize;
    if constexpr (FFT_RADIX == FftRadix::Radix16x16x16) {
        total += static_cast<int64_t>(4096) * 2;                    // T2 展开表（P5 融合 SIMT）
        total += static_cast<int64_t>(RADIX2_TW_FLOATS) * 2;        // radix-2 各层 twiddle 展开表
        total += static_cast<int64_t>(PHASE_A_OFFSET_FLOATS);       // Phase A SIMD Gather 偏移表
        total += static_cast<int64_t>(P3_T12_EXP_FLOATS);           // P3 T12 展开表（SIMD twiddle）
        total += static_cast<int64_t>(P3_LANE_COUNT);               // P3 数据 Gather lane 表
    }
    return static_cast<size_t>(total * sizeof(float));
}

size_t GetWorkspaceSize(int64_t fftN, int64_t batch, int32_t cubeCoreNum)
{
    // 每核 B 缓冲 = 轮数 × ping/pong；每份 = (2*N1) × (2*每信号复N)
    // 64×64:  4 × 128×128 = 256KB；16×16×16: 6 × 32×512 = 384KB
    RadixDecomp decomp;
    decomp.Decompose(fftN);
    int64_t bRows = static_cast<int64_t>(2) * decomp.N1;
    int64_t bCols = static_cast<int64_t>(2) * decomp.GetMmN(0);
    int64_t bufferCount = static_cast<int64_t>(decomp.iterCount) * 2;
    int64_t perCoreFloats = bufferCount * bRows * bCols;
    return static_cast<size_t>(cubeCoreNum * perCoreFloats * sizeof(float));
}

void ComputeTiling(FftHostTilingData& tiling, int64_t fftN, int64_t batch, int32_t direction,
                   int32_t cubeCoreNum, int32_t ubMode, int32_t mixMode)
{
    tiling.batch = batch;
    tiling.fftN = fftN;
    tiling.direction = direction;
    tiling.ubMode = ubMode;
    tiling.mixMode = mixMode;

    RadixDecomp decomp;
    decomp.Decompose(fftN);
    tiling.N1 = decomp.N1;
    tiling.N2 = decomp.N2;
    tiling.iterCount = decomp.iterCount;

    tiling.cubeCoreNum = cubeCoreNum;
    tiling.totalRows = static_cast<int32_t>(batch);

    int32_t totalPairs = tiling.totalRows / 2;
    int32_t pairsPerCore = CeilDiv(totalPairs, cubeCoreNum);

    int32_t fullPairs = pairsPerCore * cubeCoreNum;
    if (fullPairs > totalPairs) {
        tiling.rowsPerCore = (pairsPerCore - 1) * 2;
        tiling.rowRemainder = totalPairs - (pairsPerCore - 1) * cubeCoreNum;
    } else {
        tiling.rowsPerCore = pairsPerCore * 2;
        tiling.rowRemainder = 0;
    }

    for (int32_t iter = 0; iter < tiling.iterCount; iter++) {
        tiling.mmM[iter] = decomp.GetMmM(iter);
        tiling.mmK[iter] = decomp.GetMmK(iter);
        tiling.mmN[iter] = decomp.GetMmN(iter) * 2;
    }

    int64_t w0Size = static_cast<int64_t>(decomp.GetWMatrixRows(0)) * decomp.GetWMatrixCols(0);
    int64_t tSize = static_cast<int64_t>(decomp.GetTSingleFloats()) * decomp.GetTMatrixCount();

    int64_t coeffOffset = 0;
    tiling.wMatrixOffset[0] = coeffOffset;
    coeffOffset += w0Size * sizeof(float);
    tiling.wMatrixOffset[1] = tiling.wMatrixOffset[0];  // W1 = W0（都是 16/64 点 DFT 矩阵），直接复用
    if constexpr (FFT_RADIX == FftRadix::Radix16x16x16) {
        tiling.wMatrixOffset[2] = tiling.wMatrixOffset[0];  // W2 = W0（三轮复用同一份）
    }
    tiling.tMatrixOffset = coeffOffset;
    coeffOffset += tSize * sizeof(float);
    if constexpr (FFT_RADIX == FftRadix::Radix16x16x16) {
        tiling.tExpandedOffset = coeffOffset;   // T2 展开表（P5 融合 SIMT）
        coeffOffset += static_cast<int64_t>(4096) * 2 * sizeof(float);
        tiling.tRadix2Offset = coeffOffset;     // radix-2 各层 twiddle 展开表
        coeffOffset += static_cast<int64_t>(RADIX2_TW_FLOATS) * 2 * sizeof(float);
        tiling.tPhaseAOffset = coeffOffset;     // Phase A SIMD Gather 偏移表
        coeffOffset += static_cast<int64_t>(PHASE_A_OFFSET_FLOATS) * sizeof(float);
        tiling.t12ExpandedOffset = coeffOffset;  // P3 T12 展开表（SIMD twiddle）
        coeffOffset += static_cast<int64_t>(P3_T12_EXP_FLOATS) * sizeof(float);
        tiling.tP3LaneOffset = coeffOffset;      // P3 数据 Gather lane 表
        coeffOffset += static_cast<int64_t>(P3_LANE_COUNT) * sizeof(float);
    } else {
        tiling.tExpandedOffset = 0;
        tiling.tRadix2Offset = 0;
        tiling.tPhaseAOffset = 0;
        tiling.t12ExpandedOffset = 0;
        tiling.tP3LaneOffset = 0;
    }

    tiling.coeffsSize = GetCoeffsSize(fftN);

    // per-pair workspace: B0/B1(/B2) × ping/pong
    int64_t perBufferFloats = static_cast<int64_t>(2 * decomp.N1) * (2 * decomp.GetMmN(0));
    tiling.workspaceOffset[0] = 0;
    tiling.workspaceOffset[2] = 2 * perBufferFloats;  // B1 在 B0 的 2 份之后
    if constexpr (FFT_RADIX == FftRadix::Radix16x16x16) {
        tiling.workspaceOffset[4] = 4 * perBufferFloats;  // B2 在 B0/B1 的 4 份之后
    }
    for (int i = 0; i < 14; i++) {
        if (i != 0 && i != 2 && i != 4) {
            tiling.workspaceOffset[i] = 0;
        }
    }
    tiling.workspaceSize = GetWorkspaceSize(fftN, batch, cubeCoreNum);
}

void FillDeviceTiling(const FftHostTilingData& host, FftTilingData& dev)
{
    dev.inputParams.fftN = host.fftN;
    dev.inputParams.N1 = host.N1;
    dev.inputParams.N2 = host.N2;
    dev.inputParams.direction = host.direction;
    dev.inputParams.batch = static_cast<int32_t>(host.batch);
    dev.inputParams.iterCount = host.iterCount;

    for (int32_t i = 0; i < 3; i++) {
        dev.inputParams.mmM[i] = host.mmM[i];
        dev.inputParams.mmK[i] = host.mmK[i];
        dev.inputParams.mmN[i] = host.mmN[i];
        dev.inputParams.wMatrixOffset[i] = host.wMatrixOffset[i];
    }
    dev.inputParams.tMatrixOffset = host.tMatrixOffset;
    dev.inputParams.tExpandedOffset = host.tExpandedOffset;
    dev.inputParams.tRadix2Offset = host.tRadix2Offset;
    dev.inputParams.tPhaseAOffset = host.tPhaseAOffset;
    dev.inputParams.t12ExpandedOffset = host.t12ExpandedOffset;
    dev.inputParams.tP3LaneOffset = host.tP3LaneOffset;
    dev.inputParams.ubMode = host.ubMode;
    dev.inputParams.mixMode = host.mixMode;

    dev.multiCoreParams.coreNum = host.cubeCoreNum;
    dev.multiCoreParams.totalRows = host.totalRows;
    dev.multiCoreParams.rowsPerCore = host.rowsPerCore;
    dev.multiCoreParams.rowRemainder = host.rowRemainder;

    uint32_t offset = 0;
    for (int32_t i = 0; i < host.cubeCoreNum; i++) {
        dev.multiCoreParams.rowStartIdx[i] = offset;
        int32_t rows = host.rowsPerCore;
        if (i < host.rowRemainder) {
            rows = host.rowsPerCore + 2;
        }
        offset += static_cast<uint32_t>(rows);
    }
    dev.multiCoreParams.rowStartIdx[host.cubeCoreNum] = static_cast<uint32_t>(host.totalRows);

    for (int i = 0; i < 14; i++) {
        dev.multiCoreParams.workspaceOffset[i] = host.workspaceOffset[i];
    }
}

} // namespace Fft