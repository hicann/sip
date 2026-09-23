#include "fft_tiling.h"
#include "fft_radix16_twiddle.h"

namespace Fft {

static constexpr int32_t N1_16 = RADIX16_N;  // 16
static constexpr int32_t N2_16 = RADIX16_N;  // 16

size_t GetC2R2DCoeffsSize(int64_t fftN)
{
    (void)fftN;
    int64_t w0 = static_cast<int64_t>(2 * N1_16) * (2 * N1_16);            // 1024
    int64_t t = static_cast<int64_t>(2 * N1_16) * N2_16 * 3;               // 1536
    int64_t tExp = static_cast<int64_t>(4096) * 2;                         // 8192
    int64_t tR2 = static_cast<int64_t>(RADIX2_TW_FLOATS) * 2;              // 1920
    int64_t tPhaseA = static_cast<int64_t>(PHASE_A_OFFSET_COUNT);          // 4096
    int64_t t12E = static_cast<int64_t>(P3_T12_EXP_FLOATS);                // 8192
    int64_t p3Lane = static_cast<int64_t>(P3_LANE_COUNT);                  // 64
    return static_cast<size_t>((w0 + t + tExp + tR2 + tPhaseA + t12E + p3Lane) * sizeof(float));
}

size_t GetC2R2DBWorkspaceSize(int64_t fftN, int32_t batch, int32_t cubeCoreNum)
{
    (void)fftN; (void)batch;
    int64_t bMF = static_cast<int64_t>(2 * N1_16) * (2 * N2_16 * N2_16);   // 32*512=16384
    int64_t perCore = 4 * bMF;   // B0×2 + B1×2
    return static_cast<size_t>(cubeCoreNum * perCore * sizeof(float));
}

size_t GetC2R2DTransposeWorkspaceSize(int64_t fftN, int32_t batch)
{
    int32_t padHalf = RoundUp64(static_cast<int32_t>(fftN / 2 + 1));       // 2112
    int64_t ws = static_cast<int64_t>(batch) * fftN * padHalf * 2;          // complex
    return static_cast<size_t>(3 * ws * sizeof(float));                     // wsA+wsB+wsC
}

// 按 pair（2 行）分配 row range，返回 cur 之后 rowStartIdx[cur]==totalRows。
static void ComputeRowRanges(int32_t totalRows, int32_t coreNum, uint32_t* out)
{
    int32_t totalPairs = totalRows / 2;
    int32_t pairsPerCore = CeilDiv(totalPairs, coreNum);
    int32_t fullPairs = pairsPerCore * coreNum;
    int32_t rowsPerCore;
    int32_t rowRemainder;
    if (fullPairs > totalPairs) {
        rowsPerCore = (pairsPerCore - 1) * 2;
        rowRemainder = totalPairs - (pairsPerCore - 1) * coreNum;
    } else {
        rowsPerCore = pairsPerCore * 2;
        rowRemainder = 0;
    }
    uint32_t offset = 0;
    for (int32_t i = 0; i < coreNum; i++) {
        out[i] = offset;
        int32_t rows = rowsPerCore + (i < rowRemainder ? 2 : 0);
        offset += static_cast<uint32_t>(rows);
    }
    out[coreNum] = static_cast<uint32_t>(totalRows);
}

void ComputeC2R2DTiling(FftC2R2DHostTiling& tiling, int64_t fftN, int32_t batch,
                        int32_t cubeCoreNum)
{
    tiling.fftN = fftN;
    tiling.batch = batch;
    tiling.direction = FFT_INVERSE;
    tiling.N1 = N1_16;
    tiling.N2 = N2_16;
    tiling.iterCount = 3;
    tiling.halfN = static_cast<int32_t>(fftN / 2 + 1);                     // 2049
    tiling.padHalf = RoundUp64(tiling.halfN);                              // 2112

    tiling.mmM = 2 * N1_16;          // 32
    tiling.mmK = 2 * N1_16;          // 32
    tiling.mmN = 2 * N2_16 * N2_16;  // 512

    tiling.cubeCoreNum = cubeCoreNum;
    tiling.skinnyRows = batch * tiling.padHalf;
    tiling.fullRows = static_cast<int32_t>(batch * fftN);
    tiling.mergedRows = tiling.fullRows / 2;

    // 系数偏移
    int64_t w0 = static_cast<int64_t>(2 * N1_16) * (2 * N1_16);
    int64_t t = static_cast<int64_t>(2 * N1_16) * N2_16 * 3;
    int64_t tExp = static_cast<int64_t>(4096) * 2;
    int64_t tR2 = static_cast<int64_t>(RADIX2_TW_FLOATS) * 2;

    int64_t off = 0;
    tiling.wOffset = off;
    off += w0 * sizeof(float);
    tiling.tOffset = off;
    off += t * sizeof(float);
    tiling.tExpandedOffset = off;
    off += tExp * sizeof(float);
    tiling.tRadix2Offset = off;
    off += tR2 * sizeof(float);
    tiling.tPhaseAOffset = off;
    off += static_cast<int64_t>(PHASE_A_OFFSET_COUNT) * sizeof(float);
    tiling.t12ExpandedOffset = off;
    off += static_cast<int64_t>(P3_T12_EXP_FLOATS) * sizeof(float);
    tiling.tP3LaneOffset = off;
    off += static_cast<int64_t>(P3_LANE_COUNT) * sizeof(uint32_t);

    tiling.coeffsSize = GetC2R2DCoeffsSize(fftN);
    tiling.bWorkspaceSize = GetC2R2DBWorkspaceSize(fftN, batch, cubeCoreNum);
    tiling.transposeWorkspaceSize = GetC2R2DTransposeWorkspaceSize(fftN, batch);

    int64_t wsFloats = static_cast<int64_t>(batch) * fftN * tiling.padHalf * 2;
    tiling.wsAOffset = 0;
    tiling.wsBOffset = wsFloats * sizeof(float);
    tiling.wsCOffset = 2 * wsFloats * sizeof(float);
}

void FillC2R2DDeviceTiling(const FftC2R2DHostTiling& host, FftC2R2DTilingData& dev,
                           int32_t stage)
{
    dev.inputParams.fftN = host.fftN;
    dev.inputParams.batch = host.batch;
    dev.inputParams.direction = host.direction;
    dev.inputParams.N1 = host.N1;
    dev.inputParams.N2 = host.N2;
    dev.inputParams.iterCount = host.iterCount;
    dev.inputParams.halfN = host.halfN;
    dev.inputParams.padHalf = host.padHalf;
    dev.inputParams.mmM = host.mmM;
    dev.inputParams.mmK = host.mmK;
    dev.inputParams.mmN = host.mmN;
    dev.inputParams.wOffset = host.wOffset;
    dev.inputParams.tOffset = host.tOffset;
    dev.inputParams.tExpandedOffset = host.tExpandedOffset;
    dev.inputParams.tRadix2Offset = host.tRadix2Offset;
    dev.inputParams.tPhaseAOffset = host.tPhaseAOffset;
    dev.inputParams.t12ExpandedOffset = host.t12ExpandedOffset;
    dev.inputParams.tP3LaneOffset = host.tP3LaneOffset;

    dev.multiCoreParams.coreNum = host.cubeCoreNum;
    dev.multiCoreParams.skinnyRows = host.skinnyRows;
    dev.multiCoreParams.fullRows = host.fullRows;
    dev.multiCoreParams.mergedRows = host.mergedRows;
    ComputeRowRanges(host.skinnyRows, host.cubeCoreNum, dev.multiCoreParams.skinnyRowStartIdx);
    ComputeRowRanges(host.fullRows, host.cubeCoreNum, dev.multiCoreParams.fullRowStartIdx);
    ComputeRowRanges(host.mergedRows, host.cubeCoreNum, dev.multiCoreParams.mergedRowStartIdx);

    dev.stage = stage;
}

} // namespace Fft