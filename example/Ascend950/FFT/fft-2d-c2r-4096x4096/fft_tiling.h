#ifndef FFT_TILING_H
#define FFT_TILING_H

#include <cstddef>
#include <cstdint>
#include "fft_types.h"
#include "fft_tiling_def.h"

namespace Fft {

struct FftC2R2DHostTiling {
    int64_t fftN;
    int32_t batch;
    int32_t direction;
    int32_t N1;          // 16
    int32_t N2;          // 16
    int32_t iterCount;   // 3
    int32_t halfN;       // 2049
    int32_t padHalf;     // 2112

    int32_t mmM;         // 32
    int32_t mmK;         // 32
    int32_t mmN;         // 512

    int32_t cubeCoreNum;

    int32_t skinnyRows;  // batch * padHalf
    int32_t fullRows;    // batch * fftN
    int32_t mergedRows;  // fullRows / 2

    int64_t wOffset;
    int64_t tOffset;
    int64_t tExpandedOffset;
    int64_t tRadix2Offset;
    int64_t tPhaseAOffset;
    int64_t t12ExpandedOffset;
    int64_t tP3LaneOffset;

    size_t coeffsSize;
    size_t bWorkspaceSize;               // B0/B1 每核 pingpong buffer
    size_t transposeWorkspaceSize;        // wsA + wsB + wsC（三个瘦副本）
    int64_t wsAOffset;                    // Xt[padHalf, fftN]
    int64_t wsBOffset;                    // Y[padHalf, fftN]
    int64_t wsCOffset;                    // Yt[fftN, padHalf]
};

void ComputeC2R2DTiling(FftC2R2DHostTiling& tiling, int64_t fftN, int32_t batch,
                        int32_t cubeCoreNum);
void FillC2R2DDeviceTiling(const FftC2R2DHostTiling& host, FftC2R2DTilingData& dev,
                           int32_t stage = 2);
size_t GetC2R2DCoeffsSize(int64_t fftN);
size_t GetC2R2DBWorkspaceSize(int64_t fftN, int32_t batch, int32_t cubeCoreNum);
size_t GetC2R2DTransposeWorkspaceSize(int64_t fftN, int32_t batch);

} // namespace Fft

#endif // FFT_TILING_H