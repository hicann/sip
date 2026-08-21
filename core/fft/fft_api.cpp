/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <iostream>
#include <algorithm>
#include <mki/utils/platform/platform_info.h>
#include "aclnn/acl_meta.h"
#include "log/log.h"
#include "fftoperation/transpose.h"

#include "fftcore/dft_core.h"
#include "fftcore/fft_core_b.h"
#include "fftcore/fft_core_n.h"
#include "fftcore/fft_core_mix.h"
#include "fftcore/fft_core_stride.h"
#include "fftcore/dft_c2r_core.h"
#include "fftcore/fft_c2r_core.h"
#include "fftcore/dft_r2c_core.h"
#include "fftcore/fft_r2c_core.h"
#include "fftcore/fft_core_any.h"
#include "fftcore/dd_core.h"
#include "fftcore/dft_core_sep.h"
#include "fftcore/ddd_core_sep.h"
#include "fftcore/fft_core_b_sep.h"

// arch35 fft cores
#include "fftcore/fft_c2r_arch35_core.h"
#include "fftcore/fft_r2c_arch35_core.h"
#include "fftcore/fft_c2c_arch35_core.h"
#include "fftcore/fft_c2c2d_arch35_core.h"
#include "params/fft_c2c2d_arch35.h"

#include "fftplan/fft_plan_cache.h"
#include "fftcore/select_core.h"
#include "utils/include/utils/fft_common_func.h"
#include "utils/sip_lock.h"
#include "utils/assert.h"
#include "fft_api.h"

namespace AsdSip {
using namespace Mki;
using namespace AsdSip;

constexpr int K_FACTOR_2 = 2;
constexpr int K_LAST_DIM = -1;
constexpr int K_PENULTIMATE_DIM = -2;
constexpr int K_ANTEPENULTIMATE_DIM = -3;
constexpr int K_BLOCK_SIZE = 32;
constexpr int K_RADIX_2 = 2;
constexpr int K_RADIX_MIX = -2;
constexpr int K_RADIX_ANY = -1;
constexpr int K_N_FFT_32 = 32;
constexpr int K_N_FFT_128 = 128;
constexpr int K_N_FFT_256 = 256;
constexpr int K_N_FFT_1024 = 1024;
constexpr int K_N_FFT_2048 = 2048;
constexpr int K_N_FFT_4096 = 4096;
constexpr int K_N_FFT_8192 = 8192;
constexpr int K_N_FFT_32768 = 32768;
constexpr int K_N_FFT_65536 = 65536;
constexpr int K_VERTICAL_FACTOR_128 = 128;
constexpr int MAX_FFT_SIZE = 1 << 27;

std::vector<int64_t> RADIX_2 = {2};
std::vector<int64_t> RADIX_MIX = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47};
std::vector<int64_t> RADIX_ARCH35_C2C_MIX = {2, 3, 5, 7, 11, 13, 17, 19};

std::vector<int64_t> deDuplicates(const std::vector<int64_t>& duplicates)
{
    std::vector<int64_t> uniques;
    for (int64_t item : duplicates) {
        if (uniques.empty() || uniques.back() != item) {
            uniques.push_back(item);
        }
    }

    return uniques;
}

bool Support(const std::vector<int64_t>& uniques, const std::vector<int64_t>& radixSet)
{
    for (int64_t factor : uniques) {
        if (std::find(radixSet.begin(), radixSet.end(), factor) == radixSet.end()) {
            return false;
        }
    }

    return true;
}

int ChooseRadix(asdFftType fftType, const std::vector<int64_t>& uniques)
{
    if (fftType == asdFftType::ASCEND_FFT_C2C) {
        if (Support(uniques, RADIX_2)) {
            return K_RADIX_2;
        }
    }

    if (Support(uniques, RADIX_MIX)) {
        return K_RADIX_MIX;
    }

    return K_RADIX_ANY;
}

int ChooseRadix(AsdSip::asdFftType fftType, int64_t fftSize)
{
    std::vector<int64_t> factors = orderedFactorize(fftSize);
    std::vector<int64_t> uniques = deDuplicates(factors);

    return ChooseRadix(fftType, uniques);
}

bool Fft2dSupportVertical(int64_t batchSize, int64_t fftSizeX, int64_t fftSizeY, asdFftType fftType)
{
    if (Mki::PlatformInfo::Instance().GetPlatformType() == Mki::PlatformType::ASCEND_950) {
        return false;
    }

    if (batchSize != 1) {
        return false;
    }

    switch (fftType) {
        case asdFftType::ASCEND_FFT_C2R:
            if (GetC2RInputSize(fftSizeY) % K_VERTICAL_FACTOR_128 != 0) {
                return false;
            }
            break;
        case asdFftType::ASCEND_FFT_R2C:
            if (GetR2COutputSize(fftSizeY) % K_VERTICAL_FACTOR_128 != 0) {
                return false;
            }
            break;
        case asdFftType::ASCEND_FFT_C2C:
            if (fftSizeY % K_VERTICAL_FACTOR_128 != 0) {
                return false;
            }
            break;
        default:
            break;
    }

    std::vector<int64_t> factors = orderedFactorize(fftSizeX);
    std::vector<int64_t> uniques = deDuplicates(factors);

    if (!Support(uniques, RADIX_2)) {
        return false;
    }

    if (fftSizeX < K_N_FFT_256 || fftSizeX > K_N_FFT_65536) {
        return false;
    }

    return true;
}

bool Fft2dSupportFusing(int64_t fftSizeX, int64_t fftSizeY, int radixX, int radixY, AsdSip::asdFftType fftType)
{
    if (Mki::PlatformInfo::Instance().GetPlatformType() == Mki::PlatformType::ASCEND_950) {
        return false;
    }

    switch (fftType) {
        case asdFftType::ASCEND_FFT_C2C:
            if (radixX == K_RADIX_2 && radixY == K_RADIX_2 && fftSizeX >= K_N_FFT_32 && fftSizeY >= K_N_FFT_32 &&
                fftSizeX <= K_N_FFT_128 && fftSizeY <= K_N_FFT_128) {
                return true;
            }
            break;
        default:
            break;
    }

    return false;
}

void getC2RCore(std::optional<FFTCoreType>& coreTypeOpt, int radix, unsigned nDoing, bool forward)
{
    if (forward) {
        ASDSIP_LOG(DEBUG) << "C2RCore forward.";
    } else {
        ASDSIP_LOG(DEBUG) << "C2RCore backward.";
    }

    if (Mki::PlatformInfo::Instance().GetPlatformType() == Mki::PlatformType::ASCEND_910B) {
        if (nDoing <= K_N_FFT_1024) {
            coreTypeOpt = FFTCoreType::kDftC2R;
        } else {
            if (radix == K_RADIX_MIX) {
                coreTypeOpt = FFTCoreType::kFftC2R;
            }
        }
    } else if (Mki::PlatformInfo::Instance().GetPlatformType() == Mki::PlatformType::ASCEND_950) {
        if (nDoing <= K_N_FFT_1024) {
            coreTypeOpt = FFTCoreType::kDftC2R;
        } else {
            if (radix == K_RADIX_MIX) { // currently supports radix=2,3,5,7
                coreTypeOpt = FFTCoreType::kFftC2RArch35;
            }
        }
    }
    return;
}

void getR2CCore(std::optional<FFTCoreType>& coreTypeOpt, int radix, unsigned nDoing, bool forward)
{
    if (forward) {
        ASDSIP_LOG(DEBUG) << "R2CCore forward.";
    } else {
        ASDSIP_LOG(DEBUG) << "R2CCore backward.";
    }

    if (Mki::PlatformInfo::Instance().GetPlatformType() == Mki::PlatformType::ASCEND_910B) {
        if (nDoing <= K_N_FFT_1024) {
            coreTypeOpt = FFTCoreType::kDftR2C;
        } else {
            if (radix == K_RADIX_MIX) {
                coreTypeOpt = FFTCoreType::kFftR2C;
            }
        }
    } else if (Mki::PlatformInfo::Instance().GetPlatformType() == Mki::PlatformType::ASCEND_950) {
        if (nDoing <= K_N_FFT_1024) {
            coreTypeOpt = FFTCoreType::kDftR2C;
        } else {
            if (radix == K_RADIX_MIX) { // currently supports radix=2,3,5,7
                coreTypeOpt = FFTCoreType::kFftR2CArch35;
            }
        }
    }
    return;
}

void getC2CCore(std::optional<FFTCoreType>& coreTypeOpt, int radix, unsigned nDoing, bool forward)
{
    if (forward) {
        ASDSIP_LOG(DEBUG) << "C2CCore forward.";
    } else {
        ASDSIP_LOG(DEBUG) << "C2CCore backward.";
    }

    if (Mki::PlatformInfo::Instance().GetPlatformType() == Mki::PlatformType::ASCEND_950) {
        bool arch35Supported = nDoing > 1 && (radix == K_RADIX_2 ||
                                              (radix == K_RADIX_MIX &&
                                               Support(deDuplicates(orderedFactorize(nDoing)), RADIX_ARCH35_C2C_MIX)));
        if (arch35Supported) {
            coreTypeOpt = FFTCoreType::kFftC2CArch35;
        } else if (nDoing <= K_N_FFT_256) {
            coreTypeOpt = FFTCoreType::kDft;
        } else {
            ASDSIP_LOG(ERROR) << "ASCEND_950 C2C arch35 unsupported factorization for nDoing=" << nDoing;
            throw std::runtime_error("ASCEND_950 C2C arch35 unsupported factorization.");
        }
        return;
    }

    if (nDoing <= K_N_FFT_256) {
        coreTypeOpt = FFTCoreType::kDft;
        return;
    }
    switch (radix) {
        case K_RADIX_2:
            coreTypeOpt = nDoing < K_N_FFT_32768 ? FFTCoreType::kFftB : FFTCoreType::kFftN;
            return;
        case K_RADIX_MIX:
            if (nDoing > K_N_FFT_256) {
                coreTypeOpt = FFTCoreType::kFftMix;
            }
            return;
        default:
            return;
    }
}

// initialize FftCore according to its core_type
std::unique_ptr<FftOperation> InitFftOpPtr(std::optional<FFTCoreType> coreTypeOpt, unsigned nDone, unsigned nDoing,
                                           unsigned nLeft, unsigned batch, asdFftType fftType, bool forward,
                                           FFTPlan& plan)
{
    std::unique_ptr<FftOperation> unique(nullptr);

    if (coreTypeOpt) {
        switch (coreTypeOpt.value()) {
            // radix2 stride for c2c
            case FFTCoreType::kFftStride:
                unique.reset(new FFTCoreStride(nDone, nDoing, nLeft, batch, fftType, forward));
                break;
            // radixAny for c2c
            case FFTCoreType::kDft:
                unique.reset(new DFTCore(nDone, nDoing, nLeft, batch, fftType, forward));
                break;
            // radix2 for c2c
            case FFTCoreType::kFftB:
                unique.reset(new FFTCoreB(nDone, nDoing, nLeft, batch, fftType, forward));
                break;
            // radix2 for c2c
            case FFTCoreType::kFftN:
                unique.reset(new FFTCoreN(nDone, nDoing, nLeft, batch, fftType, forward));
                break;
            // radixMix for c2c
            case FFTCoreType::kFftMix:
                unique.reset(new FftCoreMix(nDone, nDoing, nLeft, batch, fftType, forward));
                break;
            // radixMix for c2r
            case FFTCoreType::kDftC2R:
                unique.reset(new DftC2RCore(nDoing, batch, fftType, forward));
                break;
            // radixMix for c2r
            case FFTCoreType::kFftC2R:
                unique.reset(new FftC2RCore(nDone, nDoing, nLeft, batch, fftType, forward));
                break;
            // radixMix for c2r arch35
            case FFTCoreType::kFftC2RArch35:
                unique.reset(new FftC2RCoreArch35(nDone, nDoing, nLeft, batch, fftType, forward));
                break;

            // radixMix for r2c
            case FFTCoreType::kDftR2C:
                unique.reset(new DftR2CCore(nDone, nDoing, nLeft, batch, fftType, forward));
                break;
            // radixMix for r2c
            case FFTCoreType::kFftR2C:
                unique.reset(new FftR2CCore(nDone, nDoing, nLeft, batch, fftType, forward));
                break;
            // radixMix for r2c arch35
            case FFTCoreType::kFftR2CArch35:
                unique.reset(new FftR2CCoreArch35(nDone, nDoing, nLeft, batch, fftType, forward));
                break;
            // radixMix for c2c arch35
            case FFTCoreType::kFftC2CArch35:
                unique.reset(new FftC2CCoreArch35(nDone, nDoing, nLeft, batch, fftType, forward));
                break;

            // radixAny for c2c c2r r2c
            case FFTCoreType::kAny:
                unique.reset(new FFTCoreAny(nDone, nDoing, nLeft, batch, fftType, forward));
                break;

            case FFTCoreType::kDftSep:
                unique.reset(new DFTCoreSep(nDone, nDoing, nLeft, batch, fftType, forward));
                break;

            case FFTCoreType::kFftBSep:
                unique.reset(new FFTCoreBSep(nDone, nDoing, nLeft, batch, fftType, forward));
                break;

            default:
                throw std::runtime_error("Invalid coreType.");
                break;
        }

        if (unique == nullptr) {
            ASDSIP_LOG(ERROR) << "initialize fftcore failed, ptr is nullptr.";
            throw std::runtime_error("ptr is nullptr.");
        }

        if (!unique->init()) {
            plan.markFailed();
            ASDSIP_LOG(ERROR) << "initialize fftcore failed.";
            throw std::runtime_error("initialize fftcore failed.");
        }
    }

    return unique;
}

std::unique_ptr<FftOperation> InitFft2DOpPtr(
    std::optional<FFTCoreType> coreTypeOpt, int64_t fftSizeX, int64_t fftSizeY, int64_t batchSize,
    AsdSip::asdFftType fftType, bool forward, FFTPlan& plan,
    int32_t arch35Mode = static_cast<int32_t>(OpParam::FftC2C2DArch35Mode::RADIX2))
{
    std::unique_ptr<FftOperation> unique(nullptr);

    if (coreTypeOpt) {
        switch (coreTypeOpt.value()) {
            case FFTCoreType::kDd:
                unique.reset(new DdCore(fftSizeX, fftSizeY, batchSize, fftType, forward));
                break;
            case FFTCoreType::kFftC2C2DArch35:
                unique.reset(new FftC2C2DCoreArch35(fftSizeX, fftSizeY, batchSize, fftType, forward, arch35Mode));
                break;
            default:
                break;
        }

        if (unique == nullptr) {
            ASDSIP_LOG(ERROR) << "initialize fftcore failed, ptr is nullptr.";
            throw std::runtime_error("ptr is nullptr.");
        }

        if (!unique->init()) {
            plan.markFailed();
            ASDSIP_LOG(ERROR) << "initialize fftcore failed.";
        }
    }

    return unique;
}

std::unique_ptr<FftOperation> InitFft3DOpPtr(std::optional<FFTCoreType> coreTypeOpt, int64_t fftSizeX, int64_t fftSizeY,
                                             int64_t fftSizeZ, int64_t batchSize, AsdSip::asdFftType fftType,
                                             bool forward, FFTPlan& plan)
{
    std::unique_ptr<FftOperation> unique(nullptr);

    if (coreTypeOpt) {
        switch (coreTypeOpt.value()) {
            case FFTCoreType::kDddSep:
                unique.reset(new DddCoreSep(fftSizeX, fftSizeY, fftSizeZ, batchSize, fftType, forward));
                break;
            default:
                break;
        }

        if (unique == nullptr) {
            ASDSIP_LOG(ERROR) << "initialize fftcore failed, ptr is nullptr.";
            throw std::runtime_error("ptr is nullptr.");
        }

        if (!unique->init()) {
            plan.markFailed();
            ASDSIP_LOG(ERROR) << "initialize fftcore failed.";
        }
    }

    return unique;
}

std::unique_ptr<FftOperation> getCore(const std::vector<int64_t>& uniques, unsigned nDone, unsigned nDoing,
                                      unsigned nLeft, unsigned stride, unsigned batch, asdFftType fftType, bool forward,
                                      FFTPlan& plan)
{
    std::optional<FFTCoreType> coreTypeOpt = std::nullopt;

    // only support Ascend 910B
    if (Mki::PlatformInfo::Instance().GetPlatformType() == Mki::PlatformType::ASCEND_910B) {
        if (fftType == asdFftType::ASCEND_FFT_C2C) {
            if (stride > 1) {
                return InitFftOpPtr(FFTCoreType::kFftStride, nDone, nDoing, stride, batch, fftType, forward, plan);
            }
        }

        if (fftType == asdFftType::ASCEND_FFT_C2C_SEP) {
            if ((nDoing == 256 || nDoing == 512) && forward) {
                return InitFftOpPtr(FFTCoreType::kFftBSep, nDone, nDoing, stride, batch, fftType, forward, plan);
            }
            return InitFftOpPtr(FFTCoreType::kDftSep, nDone, nDoing, stride, batch, fftType, forward, plan);
        }

        // getCore from configs
        if (fftType == asdFftType::ASCEND_FFT_C2C) {
            SelectCore::InitializeConfigs();
            coreTypeOpt = SelectCore::FindConfig(batch, nDoing);
            if (coreTypeOpt) {
                return InitFftOpPtr(coreTypeOpt, nDone, nDoing, nLeft, batch, fftType, forward, plan);
            }
        }
    }

    // the default rule of get core
    int radix = ChooseRadix(fftType, uniques);
    switch (fftType) {
        case asdFftType::ASCEND_FFT_C2R:
            getC2RCore(coreTypeOpt, radix, nDoing, forward);

            break;
        case asdFftType::ASCEND_FFT_R2C:
            getR2CCore(coreTypeOpt, radix, nDoing, forward);

            break;
        case asdFftType::ASCEND_FFT_C2C:
            getC2CCore(coreTypeOpt, radix, nDoing, forward);

            break;
        default:
            break;
    }

    if (!coreTypeOpt) {
        coreTypeOpt = FFTCoreType::kAny;
    }
    return InitFftOpPtr(coreTypeOpt, nDone, nDoing, nLeft, batch, fftType, forward, plan);
}

std::unique_ptr<FftOperation> GetCore2D(int64_t fftSizeX, int64_t fftSizeY, int radixX, int radixY, int64_t batchSize,
                                        AsdSip::asdFftType fftType, bool forward, FFTPlan& plan)
{
    if (Mki::PlatformInfo::Instance().GetPlatformType() == Mki::PlatformType::ASCEND_950) {
        if (fftType == asdFftType::ASCEND_FFT_C2C && radixX == K_RADIX_2 && radixY == K_RADIX_2) {
            return InitFft2DOpPtr(FFTCoreType::kFftC2C2DArch35, fftSizeX, fftSizeY, batchSize, fftType, forward, plan,
                                  static_cast<int32_t>(OpParam::FftC2C2DArch35Mode::RADIX2));
        }
        if (fftType == asdFftType::ASCEND_FFT_C2C &&
            Support(deDuplicates(orderedFactorize(fftSizeX)), RADIX_ARCH35_C2C_MIX) &&
            Support(deDuplicates(orderedFactorize(fftSizeY)), RADIX_ARCH35_C2C_MIX)) {
            return InitFft2DOpPtr(FFTCoreType::kFftC2C2DArch35, fftSizeX, fftSizeY, batchSize, fftType, forward, plan,
                                  static_cast<int32_t>(OpParam::FftC2C2DArch35Mode::MIXED_RADIX));
        }
        if (fftType == asdFftType::ASCEND_FFT_C2C) {
            ASDSIP_LOG(ERROR) << "ASCEND_950 2D C2C arch35 unsupported factorization: fftSizeX=" << fftSizeX
                              << ", fftSizeY=" << fftSizeY;
            throw std::runtime_error("ASCEND_950 2D C2C arch35 unsupported factorization.");
        }
        return nullptr;
    }

    std::optional<FFTCoreType> coreTypeOpt = std::nullopt;

    if (fftType == asdFftType::ASCEND_FFT_C2C) {
        if (radixX == K_RADIX_2 && radixY == K_RADIX_2 && fftSizeX >= K_N_FFT_32 && fftSizeY >= K_N_FFT_32 &&
            fftSizeX <= K_N_FFT_128 && fftSizeY <= K_N_FFT_128) {
            coreTypeOpt = FFTCoreType::kDd;
        }
    }

    return InitFft2DOpPtr(coreTypeOpt, fftSizeX, fftSizeY, batchSize, fftType, forward, plan);
}

std::unique_ptr<FftOperation> GetCore3D(int64_t fftSizeX, int64_t fftSizeY, int64_t fftSizeZ, int64_t batchSize,
                                        AsdSip::asdFftType fftType, bool forward, FFTPlan& plan)
{
    if (Mki::PlatformInfo::Instance().GetPlatformType() == Mki::PlatformType::ASCEND_950) {
        return nullptr;
    }

    std::optional<FFTCoreType> coreTypeOpt = std::nullopt;
    // currently only supports DFT 3D kernel.
    if (fftType == asdFftType::ASCEND_FFT_C2C_SEP) {
        coreTypeOpt = FFTCoreType::kDddSep;
    }

    return InitFft3DOpPtr(coreTypeOpt, fftSizeX, fftSizeY, fftSizeZ, batchSize, fftType, forward, plan);
}

bool commonMatchFunc(const FFTPlan& plan, const Tensor& inData)
{
    int64_t last = static_cast<int64_t>(inData.desc.dims.size()) - 1;
    int64_t i = last;
    if (static_cast<int64_t>(plan.fftSizes.size()) > last + 1) {
        return false;
    }
    for (auto rit = plan.fftSizes.rbegin(); rit != plan.fftSizes.rend(); rit++, i--) {
        int64_t dim = inData.desc.dims[i];
        if (dim != *rit) {
            return false;
        }
    }

    return true;
}

void addFFTTransposeStep(FFTPlan& plan, int axis0, int axis1, const SVector<int64_t>& dims)
{
    plan.steps.push_back(PlanStep{});
    PlanStep& step = plan.steps.back();
    step.operation = std::unique_ptr<FftOperation>(std::make_unique<Transpose>(axis0, axis1, dims));
}

void addFFTSteps(FFTPlan& plan, int dim, int batchSize, asdFftType fftType)
{
    if (plan.fftSizes.size() <= static_cast<size_t>(dim) || plan.fftStrides.size() <= static_cast<size_t>(dim)) {
        throw std::runtime_error("Invalid dim.");
    }
    int fftSize = plan.fftSizes[dim];
    int fftStride = plan.fftStrides[dim];
    std::vector<int64_t> factors = orderedFactorize(fftSize);
    std::vector<int64_t> uniques = deDuplicates(factors);

    unsigned nLeft = 1;
    plan.steps.push_back(PlanStep{});
    PlanStep& step = plan.steps.back();
    step.operation = getCore(uniques, 1, fftSize, nLeft, fftStride, batchSize, fftType, plan.isForward(), plan);
}

inline void addFFTSteps(FFTPlan& plan, int dim, int batchSize) { addFFTSteps(plan, dim, batchSize, plan.fftType); }

void addFFT2DStep(FFTPlan& plan, int radixX, int radixY)
{
    int64_t fftSizeX = plan.fftSizes[0];
    int64_t fftSizeY = plan.fftSizes[1];

    plan.steps.push_back(PlanStep{});
    PlanStep& step = plan.steps.back();

    step.operation = GetCore2D(fftSizeX, fftSizeY, radixX, radixY, plan.batchSize, plan.fftType, plan.isForward(),
                               plan);
}

void addFFT3DStep(FFTPlan& plan, int64_t fftSizeX, int64_t fftSizeY, int64_t fftSizeZ)
{
    plan.steps.push_back(PlanStep{});
    PlanStep& step = plan.steps.back();

    step.operation = GetCore3D(fftSizeX, fftSizeY, fftSizeZ, plan.batchSize, plan.fftType, plan.isForward(), plan);
}

void init2DSteps(FFTPlan& plan)
{
    int64_t batchSize = plan.batchSize;
    int64_t fftSizeX = plan.fftSizes[0];
    int64_t fftSizeY = plan.fftSizes[1];

    int radixX = ChooseRadix(plan.fftType, fftSizeX);
    int radixY = ChooseRadix(plan.fftType, fftSizeY);

    SVector<int64_t> dimsStepOne;
    SVector<int64_t> dimsStepTwo;
    switch (plan.fftType) {
        case asdFftType::ASCEND_FFT_C2R:
            if (Fft2dSupportVertical(batchSize, fftSizeX, fftSizeY, plan.fftType)) {
                plan.fftStrides[0] = GetC2RInputSize(fftSizeY);
                addFFTSteps(plan, 0, batchSize, asdFftType::ASCEND_FFT_C2C);
                addFFTSteps(plan, 1, batchSize * fftSizeX, asdFftType::ASCEND_FFT_C2R);
                break;
            }
            dimsStepOne.push_back(batchSize);
            dimsStepOne.push_back(fftSizeX);
            dimsStepOne.push_back(GetC2RInputSize(fftSizeY));
            addFFTTransposeStep(plan, K_LAST_DIM, K_PENULTIMATE_DIM, dimsStepOne);
            addFFTSteps(plan, 0, batchSize * GetC2RInputSize(fftSizeY), asdFftType::ASCEND_FFT_C2C);
            dimsStepTwo.push_back(batchSize);
            dimsStepTwo.push_back(GetC2RInputSize(fftSizeY));
            dimsStepTwo.push_back(fftSizeX);
            addFFTTransposeStep(plan, K_LAST_DIM, K_PENULTIMATE_DIM, dimsStepTwo);
            addFFTSteps(plan, 1, batchSize * fftSizeX, asdFftType::ASCEND_FFT_C2R);
            break;
        case asdFftType::ASCEND_FFT_R2C:
            if (Fft2dSupportVertical(batchSize, fftSizeX, fftSizeY, plan.fftType)) {
                plan.fftStrides[0] = GetC2RInputSize(fftSizeY);
                addFFTSteps(plan, 1, batchSize * fftSizeX, asdFftType::ASCEND_FFT_R2C);
                addFFTSteps(plan, 0, batchSize, asdFftType::ASCEND_FFT_C2C);
                break;
            }
            addFFTSteps(plan, 1, batchSize * fftSizeX, asdFftType::ASCEND_FFT_R2C);
            dimsStepOne.push_back(batchSize);
            dimsStepOne.push_back(fftSizeX);
            dimsStepOne.push_back(GetR2COutputSize(fftSizeY));
            addFFTTransposeStep(plan, K_LAST_DIM, K_PENULTIMATE_DIM, dimsStepOne);
            addFFTSteps(plan, 0, batchSize * GetR2COutputSize(fftSizeY), asdFftType::ASCEND_FFT_C2C);
            dimsStepTwo.push_back(batchSize);
            dimsStepTwo.push_back(GetR2COutputSize(fftSizeY));
            dimsStepTwo.push_back(fftSizeX);
            addFFTTransposeStep(plan, K_LAST_DIM, K_PENULTIMATE_DIM, dimsStepTwo);
            break;
        case asdFftType::ASCEND_FFT_C2C:
            if (Mki::PlatformInfo::Instance().GetPlatformType() == Mki::PlatformType::ASCEND_950) {
                if ((radixX == K_RADIX_2 && radixY == K_RADIX_2) ||
                    (Support(deDuplicates(orderedFactorize(fftSizeX)), RADIX_ARCH35_C2C_MIX) &&
                     Support(deDuplicates(orderedFactorize(fftSizeY)), RADIX_ARCH35_C2C_MIX))) {
                    addFFT2DStep(plan, radixX, radixY);
                    break;
                }
                ASDSIP_LOG(ERROR) << "ASCEND_950 2D C2C arch35 unsupported factorization: fftSizeX=" << fftSizeX
                                  << ", fftSizeY=" << fftSizeY;
                throw std::runtime_error("ASCEND_950 2D C2C arch35 unsupported factorization.");
            }
            if (Fft2dSupportFusing(fftSizeX, fftSizeY, radixX, radixY, plan.fftType)) {
                addFFT2DStep(plan, radixX, radixY);
                break;
            }
            if (Fft2dSupportVertical(batchSize, fftSizeX, fftSizeY, plan.fftType)) {
                plan.fftStrides[0] = fftSizeY;
                addFFTSteps(plan, 1, batchSize * fftSizeX, asdFftType::ASCEND_FFT_C2C);
                addFFTSteps(plan, 0, batchSize, asdFftType::ASCEND_FFT_C2C);
                break;
            }
            addFFTSteps(plan, 1, batchSize * fftSizeX, asdFftType::ASCEND_FFT_C2C);
            dimsStepOne.push_back(batchSize);
            dimsStepOne.push_back(fftSizeX);
            dimsStepOne.push_back(fftSizeY);
            addFFTTransposeStep(plan, K_LAST_DIM, K_PENULTIMATE_DIM, dimsStepOne);
            addFFTSteps(plan, 0, batchSize * fftSizeY, asdFftType::ASCEND_FFT_C2C);
            dimsStepTwo.push_back(batchSize);
            dimsStepTwo.push_back(fftSizeY);
            dimsStepTwo.push_back(fftSizeX);
            addFFTTransposeStep(plan, K_LAST_DIM, K_PENULTIMATE_DIM, dimsStepTwo);
            break;
        default:
            break;
    }

    plan.markInitialized();
}

void addFFTC2CFirstTwoDimsStep(FFTPlan& plan, int batchSize, int64_t fftSizeX, int64_t fftSizeY, int64_t fftSizeZ)
{
    SVector<int64_t> dimsStepOne;
    SVector<int64_t> dimsStepTwo;
    SVector<int64_t> dimsStepThree;
    SVector<int64_t> dimsStepFour;
    bool strideTag = false;

    if (Fft2dSupportVertical(batchSize, fftSizeX, fftSizeY * fftSizeZ, plan.fftType)) {
        plan.fftStrides[0] = fftSizeY * fftSizeZ;
        strideTag = true;
        addFFTSteps(plan, 0, batchSize, asdFftType::ASCEND_FFT_C2C);
    } else {
        dimsStepOne = {batchSize, fftSizeX, fftSizeY, fftSizeZ};
        addFFTTransposeStep(plan, K_LAST_DIM, K_ANTEPENULTIMATE_DIM, dimsStepOne);
        addFFTSteps(plan, 0, batchSize * fftSizeY * fftSizeZ, asdFftType::ASCEND_FFT_C2C);
    }
    if ((fftSizeX == 1) && Fft2dSupportVertical(batchSize, fftSizeY, fftSizeZ, plan.fftType)) {
        plan.fftStrides[1] = fftSizeZ;
        addFFTSteps(plan, 1, batchSize * fftSizeX, asdFftType::ASCEND_FFT_C2C);
    } else {
        if (strideTag) {
            dimsStepOne = {batchSize, fftSizeX, fftSizeY, fftSizeZ};
            addFFTTransposeStep(plan, K_LAST_DIM, K_PENULTIMATE_DIM, dimsStepOne);
            addFFTSteps(plan, 1, batchSize * fftSizeX * fftSizeZ, asdFftType::ASCEND_FFT_C2C);
            dimsStepTwo = {batchSize, fftSizeX, fftSizeZ, fftSizeY};
            addFFTTransposeStep(plan, K_LAST_DIM, K_PENULTIMATE_DIM, dimsStepTwo);
        } else {
            dimsStepTwo = {batchSize, fftSizeZ, fftSizeY, fftSizeX};
            addFFTTransposeStep(plan, K_LAST_DIM, K_PENULTIMATE_DIM, dimsStepTwo);
            addFFTSteps(plan, 1, batchSize * fftSizeX * fftSizeZ, asdFftType::ASCEND_FFT_C2C);
            dimsStepThree = {batchSize, fftSizeZ, fftSizeX, fftSizeY};
            addFFTTransposeStep(plan, K_LAST_DIM, K_PENULTIMATE_DIM, dimsStepThree);
            dimsStepFour = {batchSize, fftSizeZ, fftSizeY, fftSizeX};
            addFFTTransposeStep(plan, K_LAST_DIM, K_ANTEPENULTIMATE_DIM, dimsStepFour);
        }
    }
}

void init3DSteps(FFTPlan& plan)
{
    int64_t batchSize = plan.batchSize;
    int64_t fftSizeX = plan.fftSizes[0];
    int64_t fftSizeY = plan.fftSizes[1];
    int64_t fftSizeZ = plan.fftSizes[2];

    int radixY = ChooseRadix(plan.fftType, fftSizeY);
    int radixZ = ChooseRadix(plan.fftType, fftSizeZ);

    SVector<int64_t> dimsStepOne;
    SVector<int64_t> dimsStepTwo;

    switch (plan.fftType) {
        case asdFftType::ASCEND_FFT_C2C_SEP:
            plan.fftStrides = {fftSizeY * fftSizeZ, fftSizeZ, 1};
            addFFT3DStep(plan, fftSizeX, fftSizeY, fftSizeZ);
            break;
        case asdFftType::ASCEND_FFT_C2R:
            addFFTC2CFirstTwoDimsStep(plan, batchSize, fftSizeX, fftSizeY, GetC2RInputSize(fftSizeZ));
            addFFTSteps(plan, 2, batchSize * fftSizeX * fftSizeY, asdFftType::ASCEND_FFT_C2R);
            break;
        case asdFftType::ASCEND_FFT_R2C:
            addFFTSteps(plan, 2, batchSize * fftSizeX * fftSizeY, asdFftType::ASCEND_FFT_R2C);
            addFFTC2CFirstTwoDimsStep(plan, batchSize, fftSizeX, fftSizeY, GetR2COutputSize(fftSizeZ));
            break;
        case asdFftType::ASCEND_FFT_C2C:
            if (Fft2dSupportFusing(fftSizeY, fftSizeZ, radixY, radixZ, plan.fftType)) {
                plan.batchSize = batchSize * fftSizeX;
                plan.fftSizes = {fftSizeY, fftSizeZ};
                radixZ = ChooseRadix(plan.fftType, fftSizeZ);
                addFFT2DStep(plan, radixY, radixZ);
                plan.batchSize = batchSize;
                plan.fftSizes = {fftSizeX, fftSizeY, fftSizeZ};
                if (Fft2dSupportVertical(batchSize, fftSizeX, fftSizeY * fftSizeZ, plan.fftType)) {
                    plan.fftStrides[0] = fftSizeY * fftSizeZ;
                    addFFTSteps(plan, 0, batchSize, asdFftType::ASCEND_FFT_C2C);
                } else {
                    dimsStepOne = {batchSize, fftSizeX, fftSizeY, fftSizeZ};
                    addFFTTransposeStep(plan, K_LAST_DIM, K_ANTEPENULTIMATE_DIM, dimsStepOne);
                    addFFTSteps(plan, 0, batchSize * fftSizeY * fftSizeZ, asdFftType::ASCEND_FFT_C2C);
                    dimsStepTwo = {batchSize, fftSizeZ, fftSizeY, fftSizeX};
                    addFFTTransposeStep(plan, K_LAST_DIM, K_ANTEPENULTIMATE_DIM, dimsStepTwo);
                }
                break;
            }
            addFFTC2CFirstTwoDimsStep(plan, batchSize, fftSizeX, fftSizeY, fftSizeZ);
            addFFTSteps(plan, 2, batchSize * fftSizeX * fftSizeY, asdFftType::ASCEND_FFT_C2C);
            break;
        default:
            break;
    }

    plan.markInitialized();
}

inline size_t computeTempCachesSize(const FFTPlan& plan)
{
    size_t num = plan.steps.size() <= K_FACTOR_2 ? 1 : K_FACTOR_2;
    return num * getAlignedSize(plan.eleNum() * GetTensorElementSize(Mki::TensorDType::TENSOR_DTYPE_COMPLEX64) +
                                K_BLOCK_SIZE);
}

inline size_t computeWorkspaceSize(const FFTPlan& plan)
{
    size_t workspaceSize = 0;
    for (int64_t i = 0; i < static_cast<int64_t>(plan.steps.size()); i++) {
        workspaceSize = std::max(workspaceSize, plan.steps[i].operation->EstimateWorkspaceSize());
    }

    return workspaceSize;
}

inline bool shouldAllocTempCaches(const FFTPlan& plan) { return plan.steps.size() > 1; }

inline bool shouldAllocWorkspace(const FFTPlan& plan) { return computeWorkspaceSize(plan) != 0; }

void recycleInterCaches(FFTPlan& plan, workspace::Workspace& wkspace)
{
    int64_t num = plan.steps.size() <= K_FACTOR_2 ? 1 : K_FACTOR_2;
    for (int64_t i = 0; i < num; i++) {
        wkspace.recycleLast();
    }
}

// public interface
AspbStatus asdFftCreate(asdFftHandle& handle)
{
    std::lock_guard<std::mutex> lock(fft_mtx);
    handle = FFTPlanCache::makePlan();
    return AsdSip::ErrorType::ACL_SUCCESS;
}

AspbStatus asdFftSetStream(asdFftHandle handle, void* stream)
{
    std::lock_guard<std::mutex> lock(fft_mtx);
    if (stream == nullptr) {
        ASDSIP_LOG(ERROR) << "stream is nullptr.";
        return ErrorType::ACL_ERROR_INVALID_PARAM;
    }
    if (!FFTPlanCache::doesPlanExist(handle)) {
        ASDSIP_LOG(ERROR) << "Invalid handle.";
        return ErrorType::ACL_ERROR_INVALID_PARAM;
    }
    FFTPlan& plan = FFTPlanCache::getPlan(handle);

    plan.stream = stream;
    return AsdSip::ErrorType::ACL_SUCCESS;
}

AspbStatus asdFftMakePlan1D(asdFftHandle handle, int64_t fftSize, asdFftType fftType, asdFftDirection direction,
                            int64_t batchSize, asdFft1dDimType dimType)
{
    std::lock_guard<std::mutex> lock(fft_mtx);
    if (!FFTPlanCache::doesPlanExist(handle)) {
        ASDSIP_LOG(ERROR) << "Invalid handle.";
        return ErrorType::ACL_ERROR_INVALID_PARAM;
    }
    FFTPlan& plan = FFTPlanCache::getPlan(handle);

    if (fftSize <= 0 || fftSize > MAX_FFT_SIZE) {
        ASDSIP_LOG(ERROR) << "Invalid fft_size.";
        return ErrorType::ACL_ERROR_INVALID_PARAM;
    }

    if (batchSize <= 0) {
        ASDSIP_LOG(ERROR) << "Invalid batch_size.";
        return ErrorType::ACL_ERROR_INVALID_PARAM;
    }

    plan.fftType = fftType;
    plan.direction = direction;
    plan.fftSizes = {fftSize};

    if (dimType == asdFft1dDimType::ASCEND_FFT_VERTICAL) {
        plan.batchSize = 1;
        plan.fftStrides = {batchSize};
    } else {
        plan.batchSize = batchSize;
        plan.fftStrides = {1};
    }
    addFFTSteps(plan, 0, batchSize);
    plan.markInitialized();
    return AsdSip::ErrorType::ACL_SUCCESS;
}

AspbStatus commonParamCheck(asdFftHandle handle, int64_t fftSizeX, int64_t fftSizeY, int64_t batchSize)
{
    if (!FFTPlanCache::doesPlanExist(handle)) {
        ASDSIP_LOG(ERROR) << "Invalid handle.";
        return ErrorType::ACL_ERROR_INVALID_PARAM;
    }

    if (fftSizeX <= 0 || fftSizeX > MAX_FFT_SIZE) {
        ASDSIP_LOG(ERROR) << "Invalid fftSizeX.";
        return ErrorType::ACL_ERROR_INVALID_PARAM;
    }

    if (fftSizeY <= 0 || fftSizeY > MAX_FFT_SIZE) {
        ASDSIP_LOG(ERROR) << "Invalid fftSizeY.";
        return ErrorType::ACL_ERROR_INVALID_PARAM;
    }

    if (batchSize <= 0) {
        ASDSIP_LOG(ERROR) << "Invalid batchSize.";
        return ErrorType::ACL_ERROR_INVALID_PARAM;
    }

    return AsdSip::ErrorType::ACL_SUCCESS;
}

AspbStatus commonParamCheck3D(asdFftHandle handle, int64_t fftSizeX, int64_t fftSizeY, int64_t fftSizeZ,
                              int64_t batchSize)
{
    AspbStatus checkStatus = commonParamCheck(handle, fftSizeX, fftSizeY, batchSize);
    if (checkStatus != AsdSip::ErrorType::ACL_SUCCESS) {
        return checkStatus;
    }
    if (fftSizeZ <= 0 || fftSizeZ > MAX_FFT_SIZE) {
        ASDSIP_LOG(ERROR) << "Invalid fftSizeZ.";
        return ErrorType::ACL_ERROR_INVALID_PARAM;
    }
    return AsdSip::ErrorType::ACL_SUCCESS;
}

AspbStatus asdFftMakePlan1D(asdFftHandle handle, int64_t fftSizeX, int64_t fftSizeY, asdFftType fftType,
                            asdFftDirection direction, int64_t batchSize, int32_t dim)
{
    std::lock_guard<std::mutex> lock(fft_mtx);
    AspbStatus checkStatus = commonParamCheck(handle, fftSizeX, fftSizeY, batchSize);
    if (checkStatus != AsdSip::ErrorType::ACL_SUCCESS) {
        return checkStatus;
    }

    // 支持 dim 负索引语义: -1 归一化为 0(1D plan 的 fftSizes 仅一维), 更小的负数视为非法
    if (dim < 0) {
        dim = dim + 1;
    }
    if (dim < 0) {
        ASDSIP_LOG(ERROR) << "dim is invalid.";
        return ErrorType::ACL_ERROR_INVALID_PARAM;
    }
    FFTPlan& plan = FFTPlanCache::getPlan(handle);

    plan.fftType = fftType;
    plan.direction = direction;
    plan.batchSize = batchSize;
    plan.fftSizes = {fftSizeX};
    plan.fftStrides = {1};
    if (dim == 1) {
        plan.fftStrides = {fftSizeY};
    }

    addFFTSteps(plan, 0, batchSize);
    plan.markInitialized();

    return AsdSip::ErrorType::ACL_SUCCESS;
}

AspbStatus asdFftDestroy(asdFftHandle handle)
{
    std::lock_guard<std::mutex> lock(fft_mtx);
    if (!FFTPlanCache::doesPlanExist(handle)) {
        ASDSIP_LOG(ERROR) << "Invalid handle.";
        return ErrorType::ACL_ERROR_INVALID_PARAM;
    }
    FFTPlanCache::destroyPlan(handle);
    return AsdSip::ErrorType::ACL_SUCCESS;
}

AspbStatus asdFftMakePlan2D(asdFftHandle handle, int64_t fftSizeX, int64_t fftSizeY, asdFftType fftType,
                            asdFftDirection direction, int32_t batchSize)
{
    std::lock_guard<std::mutex> lock(fft_mtx);
    AspbStatus checkStatus = commonParamCheck(handle, fftSizeX, fftSizeY, batchSize);
    if (checkStatus != AsdSip::ErrorType::ACL_SUCCESS) {
        return checkStatus;
    }

    FFTPlan& plan = FFTPlanCache::getPlan(handle);

    plan.fftType = fftType;
    plan.direction = direction;
    plan.batchSize = batchSize;
    plan.fftSizes = {fftSizeX, fftSizeY};
    plan.fftStrides = {1, 1};

    init2DSteps(plan);

    return AsdSip::ErrorType::ACL_SUCCESS;
}

AspbStatus asdFftMakePlan3D(asdFftHandle handle, int64_t fftSizeX, int64_t fftSizeY, int64_t fftSizeZ,
                            asdFftType fftType, asdFftDirection direction, int32_t batchSize)
{
    std::lock_guard<std::mutex> lock(fft_mtx);
    AspbStatus checkStatus = commonParamCheck3D(handle, fftSizeX, fftSizeY, fftSizeZ, batchSize);
    if (checkStatus != AsdSip::ErrorType::ACL_SUCCESS) {
        return checkStatus;
    }

    FFTPlan& plan = FFTPlanCache::getPlan(handle);

    plan.fftType = fftType;
    plan.direction = direction;
    plan.batchSize = batchSize;
    plan.fftSizes = {fftSizeX, fftSizeY, fftSizeZ};
    plan.fftStrides = {1, 1, 1};

    init3DSteps(plan);
    return AsdSip::ErrorType::ACL_SUCCESS;
}

AspbStatus asdFftGetWorkspaceSize(asdFftHandle handle, size_t& workspaceSize)
{
    std::lock_guard<std::mutex> lock(fft_mtx);
    if (!FFTPlanCache::doesPlanExist(handle)) {
        ASDSIP_LOG(ERROR) << "Invalid handle.";
        return ErrorType::ACL_ERROR_INVALID_PARAM;
    }
    FFTPlan& plan = FFTPlanCache::getPlan(handle);
    workspaceSize = 0;
    if (shouldAllocTempCaches(plan)) {
        workspaceSize += computeTempCachesSize(plan);
    }
    workspaceSize += computeWorkspaceSize(plan);
    return AsdSip::ErrorType::ACL_SUCCESS;
}

AspbStatus asdFftSetWorkspace(asdFftHandle handle, void* workspace)
{
    std::lock_guard<std::mutex> lock(fft_mtx);
    if (!FFTPlanCache::doesPlanExist(handle)) {
        ASDSIP_LOG(ERROR) << "Invalid handle.";
        return ErrorType::ACL_ERROR_INVALID_PARAM;
    }
    FFTPlan& plan = FFTPlanCache::getPlan(handle);

    plan.workspaceAddr = workspace;

    return AsdSip::ErrorType::ACL_SUCCESS;
}

AspbStatus asdFftSynchronize(asdFftHandle handle)
{
    std::lock_guard<std::mutex> lock(fft_mtx);
    if (!FFTPlanCache::doesPlanExist(handle)) {
        ASDSIP_LOG(ERROR) << "Invalid handle.";
        return AsdSip::ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }
    FFTPlan& plan = FFTPlanCache::getPlan(handle);
    MkiRtStreamSynchronize(plan.stream);
    return AsdSip::ErrorType::ACL_SUCCESS;
}

AspbStatus asdFftGetType(asdFftHandle handle, asdFftType& fftType)
{
    std::lock_guard<std::mutex> lock(fft_mtx);
    if (!FFTPlanCache::doesPlanExist(handle)) {
        ASDSIP_LOG(ERROR) << "Invalid handle.";
        return AsdSip::ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }
    FFTPlan& plan = FFTPlanCache::getPlan(handle);

    fftType = plan.fftType;

    return AsdSip::ErrorType::ACL_SUCCESS;
}

std::vector<void*> allocInterCachesV2(FFTPlan& plan, workspace::Workspace& wkspace)
{
    std::vector<void*> cache;
    int64_t num = plan.steps.size() <= K_FACTOR_2 ? 1 : K_FACTOR_2;
    for (int64_t i = 0; i < num; i++) {
        size_t dataSize = getAlignedSize(
            plan.eleNum() * GetTensorElementSize(Mki::TensorDType::TENSOR_DTYPE_COMPLEX64) + K_BLOCK_SIZE);
        cache.push_back(wkspace.allocate(dataSize));
    }
    return cache;
}

AspbStatus asdFftExecV2(FFTPlan& plan, const aclTensor* input, const aclTensor* output)
{
    if (!plan.isInitialized()) {
        ASDSIP_LOG(ERROR) << "plan is not initialized.";
        return ErrorType::ACL_ERROR_INVALID_PARAM;
    }

    if (shouldAllocTempCaches(plan) && shouldAllocWorkspace(plan) && !plan.hasWorkspace()) {
        ASDSIP_LOG(ERROR) << "workspace has not been allocated.";
        return AsdSip::ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }

    workspace::Workspace wkspace(plan.workspaceAddr);
    wkspace.Reset();

    void* inputData = Mki::GetStorageAddr(input);
    if (inputData == nullptr) {
        ASDSIP_LOG(ERROR) << "input aclTensor data is nullptr.";
        return AsdSip::ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }
    void* outputData = Mki::GetStorageAddr(output);
    if (outputData == nullptr) {
        ASDSIP_LOG(ERROR) << "output aclTensor data is nullptr.";
        return AsdSip::ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }

    if (plan.steps.size() == 1) {
        plan.steps[0].operation->Run(inputData, outputData, plan.stream, wkspace);
        return AsdSip::ErrorType::ACL_SUCCESS;
    }

    std::vector<void*> tmpCache = allocInterCachesV2(plan, wkspace);
    int ping = 0;
    for (int64_t i = 0; i < static_cast<int64_t>(plan.steps.size()); i++) {
        void* tmpIn = i == 0 ? inputData : tmpCache[1 - ping];
        void* tmpOut = i == static_cast<int64_t>(plan.steps.size()) - 1 ? outputData : tmpCache[ping];

        plan.steps[i].operation->Run(tmpIn, tmpOut, plan.stream, wkspace);

        ping = 1 - ping;
    }
    recycleInterCaches(plan, wkspace);

    return AsdSip::ErrorType::ACL_SUCCESS;
}

bool matchC2C_(const FFTPlan& plan, const aclTensor* input)
{
    if (input == nullptr) {
        return false;
    }

    aclDataType dataType = aclDataType::ACL_DT_UNDEFINED;
    auto ret = aclGetDataType(input, &dataType);
    if (ret != 0 || dataType != aclDataType::ACL_COMPLEX64) {
        return false;
    }

    int64_t* dims = nullptr;
    uint64_t tensorDimSize = 0;
    ret = aclGetViewShape(input, &dims, &tensorDimSize);
    if (ret != 0) {
        delete[] dims;
        dims = nullptr;
        return false;
    }

    // avoid check for fft stride
    if (plan.fftStrides[0] != 1) {
        delete[] dims;
        dims = nullptr;
        return true;
    }

    int64_t lastDim = static_cast<int64_t>(tensorDimSize) - 1;
    // 防护: 输入张量维度数小于 plan 的 fftSizes 维度数时, 循环索引会变为负值导致越界读
    if (static_cast<int64_t>(plan.fftSizes.size()) > lastDim + 1) {
        delete[] dims;
        dims = nullptr;
        return false;
    }
    int64_t i = lastDim;
    for (auto iterPtr = plan.fftSizes.rbegin(); iterPtr != plan.fftSizes.rend(); iterPtr++, i--) {
        int64_t dim = dims[i];
        if (dim != *iterPtr) {
            delete[] dims;
            dims = nullptr;
            return false;
        }
    }

    delete[] dims;
    return true;
}

bool matchC2R_(const FFTPlan& plan, const aclTensor* input)
{
    if (input == nullptr) {
        return false;
    }

    aclDataType dataType = aclDataType::ACL_DT_UNDEFINED;
    auto ret = aclGetDataType(input, &dataType);
    if (ret != 0 || dataType != aclDataType::ACL_COMPLEX64) {
        return false;
    }

    int64_t* numOfDims = nullptr;
    uint64_t dimSize = 0;
    ret = aclGetViewShape(input, &numOfDims, &dimSize);
    if (ret != 0) {
        delete[] numOfDims;
        numOfDims = nullptr;
        return false;
    }

    int64_t lastDim = static_cast<int64_t>(dimSize) - 1;
    // 防护: 输入张量维度数小于 plan 的 fftSizes 维度数时, 循环索引会变为负值导致越界读
    if (static_cast<int64_t>(plan.fftSizes.size()) > lastDim + 1) {
        delete[] numOfDims;
        numOfDims = nullptr;
        return false;
    }
    int64_t dimIterIdx = lastDim;
    for (auto rit = plan.fftSizes.rbegin(); rit != plan.fftSizes.rend(); rit++, dimIterIdx--) {
        int64_t dim = numOfDims[dimIterIdx];
        if (dimIterIdx == lastDim) {
            if (dim != (*rit / K_FACTOR_2 + 1)) {
                delete[] numOfDims;
                numOfDims = nullptr;
                return false;
            }
        } else {
            if (*rit != dim) {
                delete[] numOfDims;
                numOfDims = nullptr;
                return false;
            }
        }
    }
    delete[] numOfDims;
    return true;
}

bool matchR2C_(const FFTPlan& plan, const aclTensor* input)
{
    if (input == nullptr) {
        return false;
    }

    aclDataType dataType = aclDataType::ACL_DT_UNDEFINED;
    auto ret = aclGetDataType(input, &dataType);
    if (ret != 0 || dataType != aclDataType::ACL_FLOAT) {
        return false;
    }

    int64_t* dims = nullptr;
    uint64_t dimsNum = 0;
    ret = aclGetViewShape(input, &dims, &dimsNum);
    if (ret != 0) {
        delete[] dims;
        dims = nullptr;
        return false;
    }

    int64_t last = static_cast<int64_t>(dimsNum) - 1;
    int64_t i = last;
    for (auto rit = plan.fftSizes.rbegin(); rit != plan.fftSizes.rend(); rit++, i--) {
        int64_t dim = dims[i];
        if (dim != *rit) {
            delete[] dims;
            dims = nullptr;
            return false;
        }
    }
    delete[] dims;
    dims = nullptr;
    return true;
}

AsdSip::AspbStatus asdFftExecC2C(asdFftHandle handle, const aclTensor* input, const aclTensor* output)
{
    std::lock_guard<std::mutex> lock(fft_mtx);
    if (!FFTPlanCache::doesPlanExist(handle)) {
        ASDSIP_LOG(ERROR) << "Invalid handle.";
        return ErrorType::ACL_ERROR_INVALID_PARAM;
    }
    FFTPlan& plan = FFTPlanCache::getPlan(handle);

    if (!matchC2C_(plan, input)) {
        ASDSIP_LOG(ERROR) << "input does not match plan.";
        return ErrorType::ACL_ERROR_INVALID_PARAM;
    }

    if (!matchC2C_(plan, output)) {
        ASDSIP_LOG(ERROR) << "output does not match plan.";
        return ErrorType::ACL_ERROR_INVALID_PARAM;
    }

    return asdFftExecV2(plan, input, output);
}

AspbStatus asdFftExecC2R(asdFftHandle handle, const aclTensor* input, const aclTensor* output)
{
    std::lock_guard<std::mutex> lock(fft_mtx);
    if (!FFTPlanCache::doesPlanExist(handle)) {
        ASDSIP_LOG(ERROR) << "Invalid handle.";
        return ErrorType::ACL_ERROR_INVALID_PARAM;
    }
    FFTPlan& plan = FFTPlanCache::getPlan(handle);

    if (!matchC2R_(plan, input)) {
        ASDSIP_LOG(ERROR) << "input does not match plan.";
        return ErrorType::ACL_ERROR_INVALID_PARAM;
    }

    if (!matchR2C_(plan, output)) {
        ASDSIP_LOG(ERROR) << "output does not match plan.";
        return ErrorType::ACL_ERROR_INVALID_PARAM;
    }

    return asdFftExecV2(plan, input, output);
}

AspbStatus asdFftExecR2C(asdFftHandle handle, const aclTensor* input, const aclTensor* output)
{
    std::lock_guard<std::mutex> lock(fft_mtx);
    if (!FFTPlanCache::doesPlanExist(handle)) {
        ASDSIP_LOG(ERROR) << "Invalid handle.";
        return ErrorType::ACL_ERROR_INVALID_PARAM;
    }
    FFTPlan& plan = FFTPlanCache::getPlan(handle);

    if (!matchR2C_(plan, input)) {
        ASDSIP_LOG(ERROR) << "input does not match plan.";
        return ErrorType::ACL_ERROR_INVALID_PARAM;
    }

    if (!matchC2R_(plan, output)) {
        ASDSIP_LOG(ERROR) << "output does not match plan.";
        return ErrorType::ACL_ERROR_INVALID_PARAM;
    }

    return asdFftExecV2(plan, input, output);
}

AspbStatus asdFftExecV2Separated(FFTPlan& plan, const aclTensor* inputReal, const aclTensor* inputImag,
                                 const aclTensor* outputReal, const aclTensor* outputImag)
{
    if (!plan.isInitialized()) {
        ASDSIP_LOG(ERROR) << "plan is not initilized.";
        return ErrorType::ACL_ERROR_INVALID_PARAM;
    }

    if (shouldAllocTempCaches(plan) && shouldAllocWorkspace(plan) && !plan.hasWorkspace()) {
        ASDSIP_LOG(ERROR) << "workspace has not been allocated.";
        return AsdSip::ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }

    workspace::Workspace wkspace(plan.workspaceAddr);
    wkspace.Reset();

    void* inputRealData = Mki::GetStorageAddr(inputReal);
    if (inputRealData == nullptr) {
        ASDSIP_LOG(ERROR) << "input aclTensor data is nullptr.";
        return AsdSip::ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }
    void* inputImagData = Mki::GetStorageAddr(inputImag);
    if (inputImagData == nullptr) {
        ASDSIP_LOG(ERROR) << "input aclTensor data is nullptr.";
        return AsdSip::ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }
    void* outputRealData = Mki::GetStorageAddr(outputReal);
    if (outputRealData == nullptr) {
        ASDSIP_LOG(ERROR) << "output aclTensor data is nullptr.";
        return AsdSip::ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }
    void* outputImagData = Mki::GetStorageAddr(outputImag);
    if (outputImagData == nullptr) {
        ASDSIP_LOG(ERROR) << "output aclTensor data is nullptr.";
        return AsdSip::ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }

    if (plan.steps.size() == 1) {
        plan.steps[0].operation->Run(inputRealData, inputImagData, outputRealData, outputImagData, plan.stream,
                                     wkspace);
        return AsdSip::ErrorType::ACL_SUCCESS;
    }
    return AsdSip::ErrorType::ACL_SUCCESS;
}

AspbStatus asdFftExecC2CSeparated(asdFftHandle handle, const aclTensor* inputReal, const aclTensor* inputImag,
                                  const aclTensor* outputReal, const aclTensor* outputImag)
{
    std::lock_guard<std::mutex> lock(fft_mtx);
    if (!FFTPlanCache::doesPlanExist(handle)) {
        ASDSIP_LOG(ERROR) << "Invalid handle.";
        return ErrorType::ACL_ERROR_INVALID_PARAM;
    }

    int64_t* viewDimsInReal = nullptr;
    uint64_t viewDimsNumInReal = 0;
    CHECK_STATUS_WITH_ACL_RETURN(aclGetViewShape(inputReal, &viewDimsInReal, &viewDimsNumInReal), "aclGetViewShape");

    int64_t* viewDimsInImag = nullptr;
    uint64_t viewDimsNumInImag = 0;
    CHECK_STATUS_WITH_ACL_RETURN(aclGetViewShape(inputImag, &viewDimsInImag, &viewDimsNumInImag), "aclGetViewShape");

    int64_t* viewDimsOutReal = nullptr;
    uint64_t viewDimsNumOutReal = 0;
    CHECK_STATUS_WITH_ACL_RETURN(aclGetViewShape(outputReal, &viewDimsOutReal, &viewDimsNumOutReal), "aclGetViewShape");

    int64_t* viewDimsOutImag = nullptr;
    uint64_t viewDimsNumOutImag = 0;
    CHECK_STATUS_WITH_ACL_RETURN(aclGetViewShape(outputImag, &viewDimsOutImag, &viewDimsNumOutImag), "aclGetViewShape");

    if ((viewDimsNumInReal != viewDimsNumInImag) || (viewDimsNumOutReal != viewDimsNumOutImag) ||
        (viewDimsNumInReal != viewDimsNumOutReal)) {
        ASDSIP_ELOG(ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH) << "invalid input/output format.";
        delete[] viewDimsInReal;
        delete[] viewDimsInImag;
        delete[] viewDimsOutReal;
        delete[] viewDimsOutImag;
        return AsdSip::ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }

    bool validFlag = true;
    for (uint64_t dim = 0; dim < viewDimsNumInReal; dim++) {
        if ((viewDimsInReal[dim] != viewDimsInImag[dim]) || (viewDimsInReal[dim] != viewDimsOutReal[dim]) ||
            (viewDimsOutReal[dim] != viewDimsOutImag[dim])) {
            validFlag = false;
            break;
        }
    }

    if (!validFlag) {
        ASDSIP_ELOG(ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH) << "invalid input/output shape.";
        delete[] viewDimsInReal;
        delete[] viewDimsInImag;
        delete[] viewDimsOutReal;
        delete[] viewDimsOutImag;
        return AsdSip::ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }

    delete[] viewDimsInReal;
    delete[] viewDimsInImag;
    delete[] viewDimsOutReal;
    delete[] viewDimsOutImag;

    FFTPlan& plan = FFTPlanCache::getPlan(handle);

    return asdFftExecV2Separated(plan, inputReal, inputImag, outputReal, outputImag);
}

} // namespace AsdSip
