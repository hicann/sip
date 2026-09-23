#ifndef FFT_TILING_H
#define FFT_TILING_H

#include <cstddef>
#include <cstdint>
#include "fft_types.h"
#include "fft_tiling_def.h"

namespace Fft {

struct FftHostTilingData {
    int64_t batch;
    int64_t fftN;
    int32_t N1;
    int32_t N2;
    int32_t iterCount;
    int32_t direction;

    int32_t cubeCoreNum;
    int32_t totalRows;
    int32_t rowsPerCore;
    int32_t rowRemainder;

    int32_t mmM[3];
    int32_t mmK[3];
    int32_t mmN[3];

    size_t coeffsSize;
    size_t workspaceSize;

    int64_t wMatrixOffset[3];
    int64_t tMatrixOffset;
    int64_t tExpandedOffset;
    int64_t tRadix2Offset;
    int64_t tPhaseAOffset;
    int64_t t12ExpandedOffset;
    int64_t tP3LaneOffset;
    int64_t workspaceOffset[14];

    int32_t ubMode;
    int32_t mixMode;
};

void FillDeviceTiling(const FftHostTilingData& hostTiling, FftTilingData& devTiling);
void ComputeTiling(FftHostTilingData& tiling, int64_t fftN, int64_t batch, int32_t direction,
                   int32_t cubeCoreNum, int32_t ubMode = 0, int32_t mixMode = 2);
size_t GetCoeffsSize(int64_t fftN);
size_t GetWorkspaceSize(int64_t fftN, int64_t batch, int32_t cubeCoreNum = 28);

} // namespace Fft

#endif // FFT_TILING_H