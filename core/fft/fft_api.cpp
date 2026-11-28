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
#include <cstdlib>
#include <limits>
#include <memory>
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
#include "fftcore/fft_c2c_regbase_core.h"
#include "fftcore/fft_c2c_large_core.h"
#include "fftcore/fft_real_regbase_core.h"
#include "fftcore/fft_real_stockham_core.h"
#include "params/fft_real_dispatch.h"
#include "params/fft_c2c_dispatch.h"
#include "params/fft_c2c_regbase.h"
#include "fftcore/fft_c2c2d_arch35_core.h"

#include "fftcore/ops_fft_kernel_stub.h"

#include "fftplan/fft_plan_cache.h"
#include "fftcore/select_core.h"
#include "utils/include/utils/fft_common_func.h"
#include "utils/sip_lock.h"
#include "utils/assert.h"
#include "fft1d_optimized_selection.h"
#include "fft_mixed_selection.h"
#include "fftcore/fft_mixed_common_core.h"
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

void getC2RCore(std::optional<FFTCoreType>& coreTypeOpt, int radix, unsigned nDoing, unsigned batch, bool forward)
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
        if (fft::a5::SelectReal(nDoing, batch, true) == fft::a5::RealRoute::Regbase) {
            coreTypeOpt = FFTCoreType::kFftC2RRegBase;
        } else if (fft::a5::SelectReal(nDoing, batch, true) == fft::a5::RealRoute::Stockham) {
            coreTypeOpt = FFTCoreType::kFftC2RStockham;
        } else if (nDoing <= K_N_FFT_1024) {
            coreTypeOpt = FFTCoreType::kDftC2R;
        } else {
            if (radix == K_RADIX_MIX) { // currently supports radix=2,3,5,7
                coreTypeOpt = FFTCoreType::kFftC2RArch35;
            }
        }
    }
    return;
}

void getR2CCore(std::optional<FFTCoreType>& coreTypeOpt, int radix, unsigned nDoing, unsigned batch, bool forward)
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
        if (fft::a5::SelectReal(nDoing, batch, false) == fft::a5::RealRoute::Regbase) {
            coreTypeOpt = FFTCoreType::kFftR2CRegBase;
        } else if (fft::a5::SelectReal(nDoing, batch, false) == fft::a5::RealRoute::Stockham) {
            coreTypeOpt = FFTCoreType::kFftR2CStockham;
        } else if (nDoing <= K_N_FFT_1024) {
            coreTypeOpt = FFTCoreType::kDftR2C;
        } else {
            if (radix == K_RADIX_MIX) { // currently supports radix=2,3,5,7
                coreTypeOpt = FFTCoreType::kFftR2CArch35;
            }
        }
    }
    return;
}

void getC2CCore(std::optional<FFTCoreType>& coreTypeOpt, int radix, unsigned nDoing, unsigned batch, bool forward)
{
    if (forward) {
        ASDSIP_LOG(DEBUG) << "C2CCore forward.";
    } else {
        ASDSIP_LOG(DEBUG) << "C2CCore backward.";
    }

    if (Mki::PlatformInfo::Instance().GetPlatformType() == Mki::PlatformType::ASCEND_950) {
        const auto route = fft::a5::Select(nDoing, batch);
        if (route == fft::a5::Route::Dft) {
            coreTypeOpt = FFTCoreType::kDft;
            return;
        }
        if (route == fft::a5::Route::Regbase) {
            coreTypeOpt = FFTCoreType::kFftC2CRegBase;
            return;
        }
        if (route != fft::a5::Route::Existing) {
            coreTypeOpt = FFTCoreType::kFftC2CCubeVector;
            return;
        }
        bool arch35Supported = nDoing > 1 && (radix == K_RADIX_2 ||
                                              (radix == K_RADIX_MIX &&
                                               Support(deDuplicates(orderedFactorize(nDoing)), RADIX_ARCH35_C2C_MIX)));
        if (arch35Supported) {
            coreTypeOpt = FFTCoreType::kFftC2CSimt;
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
            case FFTCoreType::kFftR2CRegBase:
            case FFTCoreType::kFftC2RRegBase:
                unique.reset(new FftRealRegBaseCore(nDone, nDoing, nLeft, batch, fftType, forward));
                break;
            case FFTCoreType::kFftR2CStockham:
            case FFTCoreType::kFftC2RStockham:
                unique.reset(new FftRealStockhamCore(nDone, nDoing, nLeft, batch, fftType, forward));
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
            case FFTCoreType::kFftC2CRegBase:
                if (nDoing <= 8192) {
                    unique.reset(new FftC2CRegBaseCore(nDone, nDoing, nLeft, batch, fftType, forward));
                } else {
                    unique.reset(
                        new FftC2CLargeCore(coreTypeOpt.value(), nDone, nDoing, nLeft, batch, fftType, forward));
                }
                break;
            case FFTCoreType::kFftC2CCubeVector:
                unique.reset(new FftC2CLargeCore(coreTypeOpt.value(), nDone, nDoing, nLeft, batch, fftType, forward));
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
        if (Mki::PlatformInfo::Instance().GetPlatformType() == Mki::PlatformType::ASCEND_950) {
            const char* backend = "existing";
            switch (coreTypeOpt.value()) {
                case FFTCoreType::kDft:
                    backend = "DFT-C2C";
                    break;
                case FFTCoreType::kDftC2R:
                    backend = "DFT-C2R";
                    break;
                case FFTCoreType::kFftC2RRegBase:
                    backend = "RegBase-C2R";
                    break;
                case FFTCoreType::kFftR2CRegBase:
                    backend = "RegBase-R2C";
                    break;
                case FFTCoreType::kFftC2RStockham:
                    backend = "Stockham-C2R";
                    break;
                case FFTCoreType::kFftR2CStockham:
                    backend = "Stockham-R2C";
                    break;
                case FFTCoreType::kDftR2C:
                    backend = "DFT-R2C";
                    break;
                case FFTCoreType::kFftC2CRegBase:
                    backend = "RegBase";
                    break;
                case FFTCoreType::kFftC2CCubeVector:
                    backend = fft::a5::Select(nDoing, batch) == fft::a5::Route::VectorExtended ? "Vector-C2C" :
                                                                                                 "CubeVector";
                    break;
                case FFTCoreType::kFftC2CSimt:
                    backend = "SIMT-C2C";
                    break;
                case FFTCoreType::kFftC2RArch35:
                    backend = "SIMT-C2R";
                    break;
                case FFTCoreType::kFftR2CArch35:
                    backend = "SIMT-R2C";
                    break;
                default:
                    break;
            }
            const char* trace = std::getenv("ASDSIP_FFT_PLAN_TRACE");
            if (trace != nullptr && std::string(trace) == "1") {
                std::cerr << "A5_FFT_BACKEND backend=" << backend << " N=" << nDoing << " effective_batch=" << batch
                          << " forward=" << forward << " workspace=" << unique->EstimateWorkspaceSize() << '\n';
            }
        }
    }

    return unique;
}

std::unique_ptr<FftOperation> InitFft2DOpPtr(std::optional<FFTCoreType> coreTypeOpt, int64_t fftSizeX, int64_t fftSizeY,
                                             int64_t batchSize, AsdSip::asdFftType fftType, bool forward, FFTPlan& plan)
{
    std::unique_ptr<FftOperation> unique(nullptr);

    if (coreTypeOpt) {
        switch (coreTypeOpt.value()) {
            case FFTCoreType::kDd:
                unique.reset(new DdCore(fftSizeX, fftSizeY, batchSize, fftType, forward));
                break;
            case FFTCoreType::kFftC2C2DArch35:
                // mode 死成员删除（issue #356）：构造入参从未被核内读取，RADIX2 枚举分支
                // 亦静态不可达（issue #357）
                unique.reset(new FftC2C2DCoreArch35(fftSizeX, fftSizeY, batchSize, fftType, forward));
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
            // 与 1D InitFftOpPtr 口径对齐：失败必须抛出，由 API 边界翻译为
            // 错误码——否则尾部 markInitialized 会把 FAILED 覆写回 INITIALIZED，
            // 空 kernel 的 plan 通过守卫后崩溃（issue #169）
            throw std::runtime_error("initialize fftcore failed.");
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
            // 与 1D/2D 口径对齐：失败必须抛出，由 API 边界翻译为错误码
            // （issue #169，防止 FAILED 被尾部 markInitialized 覆写）
            throw std::runtime_error("initialize fftcore failed.");
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
            getC2RCore(coreTypeOpt, radix, nDoing, batch, forward);

            break;
        case asdFftType::ASCEND_FFT_R2C:
            getR2CCore(coreTypeOpt, radix, nDoing, batch, forward);

            break;
        case asdFftType::ASCEND_FFT_C2C:
            getC2CCore(coreTypeOpt, radix, nDoing, batch, forward);

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
        // radix-2×radix-2 组合在 init2DSteps 已显式导流至 ordinary transpose/1D 组合路径，
        // 不进入 addFFT2DStep（issue #357）：原 RADIX2 分支静态不可达，随 mode 死成员一并删除
        if (fftType == asdFftType::ASCEND_FFT_C2C &&
            Support(deDuplicates(orderedFactorize(fftSizeX)), RADIX_ARCH35_C2C_MIX) &&
            Support(deDuplicates(orderedFactorize(fftSizeY)), RADIX_ARCH35_C2C_MIX)) {
            return InitFft2DOpPtr(FFTCoreType::kFftC2C2DArch35, fftSizeX, fftSizeY, batchSize, fftType, forward, plan);
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

// commonMatchFunc 已删除（issue #232）：全仓零调用的死代码，无头文件声明；
// 其"plan 与输入张量尾维匹配"逻辑未来需要时应在有消费方的前提下重新实现，
// 避免随 Tensor/FFTPlan 结构演进而悄然失配

void addFFTTransposeStep(FFTPlan& plan, int axis0, int axis1, const SVector<int64_t>& dims)
{
    plan.steps.push_back(PlanStep{});
    PlanStep& step = plan.steps.back();
    step.operation = std::unique_ptr<FftOperation>(std::make_unique<Transpose>(axis0, axis1, dims));
    if (!step.operation->init()) {
        throw std::runtime_error("initialize transpose failed");
    }
}

void addFFTSteps(FFTPlan& plan, int dim, int64_t batchSize, asdFftType fftType)
{
    if (plan.fftSizes.size() <= static_cast<size_t>(dim) || plan.fftStrides.size() <= static_cast<size_t>(dim)) {
        throw std::runtime_error("Invalid dim.");
    }
    // 2D/3D 派生 batch 乘积（batchSize * fftSizeX[*fftSizeY]）以 int64 传入后，必须能
    // 无损落入 getCore 的 unsigned 形参；超过 INT32_MAX 截断为负/错值会静默产出错误
    // plan（issue #214，与 #175 的 1D 顶层 batchSize 入口守卫同口径）
    if (batchSize > static_cast<int64_t>(std::numeric_limits<int32_t>::max())) {
        ASDSIP_LOG(ERROR) << "derived batch size exceeds INT32_MAX.";
        throw std::runtime_error("derived batch size exceeds INT32_MAX.");
    }
    // fftSizes 为 int64，窄化到 int 前先做范围校验（issue #214）
    if (plan.fftSizes[dim] > static_cast<int64_t>(std::numeric_limits<int32_t>::max())) {
        ASDSIP_LOG(ERROR) << "fft size exceeds INT32_MAX.";
        throw std::runtime_error("fft size exceeds INT32_MAX.");
    }
    int fftSize = static_cast<int>(plan.fftSizes[dim]);
    int fftStride = plan.fftStrides[dim];
    const auto kindOf = [](asdFftType type) {
        return type == asdFftType::ASCEND_FFT_C2C ? fft1d::Kind::C2C :
               type == asdFftType::ASCEND_FFT_R2C ? fft1d::Kind::R2C :
               type == asdFftType::ASCEND_FFT_C2R ? fft1d::Kind::C2R :
                                                    fft1d::Kind::Other;
    };
    const auto kind = kindOf(fftType);
    const bool ascend950 = Mki::PlatformInfo::Instance().GetPlatformType() == Mki::PlatformType::ASCEND_950;
    const int mixedPass = kind == fft1d::Kind::C2C ? 0 :
                          kind == fft1d::Kind::R2C ? 1 :
                          kind == fft1d::Kind::C2R ? 2 :
                                                     -1;
    const bool realPlan = plan.fftType == asdFftType::ASCEND_FFT_R2C || plan.fftType == asdFftType::ASCEND_FFT_C2R;
    const bool axis2d = plan.fftSizes.size() == 2 && ascend950 && fftStride == 1 && !UseOpsFftKernelBackend();
    const bool real2d = axis2d && realPlan;
    using RealRoute = fft1d::Real2dAxisRoute;
    const auto pass = static_cast<OpParam::MixedCommonPass>(mixedPass);
    const bool square = real2d && plan.batchSize == 1 && plan.fftSizes[0] == plan.fftSizes[1] &&
                        (plan.fftType == asdFftType::ASCEND_FFT_R2C) == plan.isForward();
    const auto realRoute = real2d ? fft1d::SelectReal2dAxis(fftSize, batchSize, pass, !plan.isForward(), square) :
                                    RealRoute::Native;
    const bool complex2dAxis = axis2d && mixedPass == 0 &&
                               (plan.fftType == asdFftType::ASCEND_FFT_C2C || realRoute != RealRoute::Native);
    const bool regBaseAxis = complex2dAxis && fftSize >= 32 && fftSize <= 8192 && (fftSize & (fftSize - 1)) == 0;
    const bool realRegBase = mixedPass > 0 && realRoute == RealRoute::RegBase;
    const bool realMixedAxis = mixedPass > 0 && realRoute == RealRoute::Mixed;
    const bool allowCube = plan.fftType == asdFftType::ASCEND_FFT_C2C || complex2dAxis || realMixedAxis ||
                           (plan.fftSizes.size() == 1 && realPlan && OpParam::RealMixedLength(fftSize));
    const OpParam::FftMixedCommon axisParam{fftSize,       batchSize, pass, !plan.isForward(),
                                            complex2dAxis, allowCube, true, realMixedAxis};
    if ((((complex2dAxis && !regBaseAxis) || realMixedAxis) && OpParam::ValidMixedCommon(axisParam)) ||
        (plan.fftSizes.size() == 1 && !UseOpsFftKernelBackend() &&
         fft1d::CommonMixedEligible(fftSize, batchSize, mixedPass, !plan.isForward(), ascend950, fftStride == 1,
                                    allowCube))) {
        auto core = std::make_unique<FftMixedCommonCore>(axisParam);
        if (core->init()) {
            plan.steps.push_back(PlanStep{std::move(core)});
            return;
        }
        ASDSIP_LOG(WARN) << "Common mixed FFT unavailable; using original route. N=" << fftSize;
    }
    int config = 0;
    if (plan.fftSizes.size() == 1) {
        config = fft1d::OptimizedConfig(fftSize, batchSize, kind, !plan.isForward(), ascend950, fftStride == 1);
    } else if (regBaseAxis || realRegBase) {
        config = fft1d::OptimizedConfig(fftSize, fftSize, kind, !plan.isForward(), ascend950, true);
    } else if (plan.fftSizes.size() == 2 && !real2d) {
        config = fft1d::OptimizedSquareConfig(fftSize, plan.fftSizes[1 - dim], plan.batchSize, batchSize, kind,
                                              kindOf(plan.fftType), !plan.isForward(), ascend950, fftStride == 1);
    }
    const bool multirow1d = ascend950 && !UseOpsFftKernelBackend() && plan.fftSizes.size() == 1 && fftStride == 1 &&
                            batchSize >= 1 && (kind != fft1d::Kind::C2C || batchSize <= 8192) && fftSize >= 32 &&
                            fftSize <= (kind == fft1d::Kind::C2C ? 4096 : 8192) && (fftSize & (fftSize - 1)) == 0 &&
                            (kind == fft1d::Kind::C2C || (kind == fft1d::Kind::R2C && plan.isForward()) ||
                             (kind == fft1d::Kind::C2R && !plan.isForward()));
    const bool multirow = multirow1d || (realRegBase && batchSize != fftSize);
    if (multirow)
        config = fft1d::OptimizedConfig(fftSize, fftSize, kind, !plan.isForward(), true, true);
    if (config != 0) {
        std::unique_ptr<FftOperation> core;
        if (kind == fft1d::Kind::C2C)
            core.reset(new FftC2CRegBaseCore(1, fftSize, 1, batchSize, fftType, plan.isForward(), config, multirow));
        else
            core.reset(new FftRealRegBaseCore(1, fftSize, 1, batchSize, fftType, plan.isForward(), config, multirow));
        if (core->init()) {
            const char* trace = std::getenv("ASDSIP_FFT_PLAN_TRACE");
            if ((regBaseAxis || realRegBase) && trace && std::string(trace) == "1")
                std::cerr << "ASDSIP_REGBASE_FFT N=" << fftSize << " batch=" << batchSize
                          << " inverse=" << !plan.isForward() << std::endl;
            plan.steps.push_back(PlanStep{std::move(core)});
            return;
        }
        // Capability/initialization rejection must not remove the original route.
        // Exceptions still propagate to the public API's existing error handling.
        ASDSIP_LOG(WARN) << "Optimized FFT axis unavailable; using original route. N=" << fftSize;
    }
    std::vector<int64_t> factors = orderedFactorize(fftSize);
    std::vector<int64_t> uniques = deDuplicates(factors);

    unsigned nLeft = 1;
    plan.steps.push_back(PlanStep{});
    PlanStep& step = plan.steps.back();
    // 入口已校验 batchSize/fftSize <= INT32_MAX，窄化到 getCore 的 unsigned 形参无损
    step.operation = getCore(uniques, 1, fftSize, nLeft, fftStride, static_cast<unsigned>(batchSize), fftType,
                             plan.isForward(), plan);
}

inline void addFFTSteps(FFTPlan& plan, int dim, int64_t batchSize) { addFFTSteps(plan, dim, batchSize, plan.fftType); }

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
    auto operation = GetCore3D(fftSizeX, fftSizeY, fftSizeZ, plan.batchSize, plan.fftType, plan.isForward(), plan);
    // 950 平台 GetCore3D 返回 nullptr：plan 置 FAILED 并抛出，由 API 边界
    // 统一翻译为错误码，杜绝空 operation 入 steps 后被 markInitialized
    // 放行、GetWorkspaceSize/exec 空指针崩溃（issue #168）
    if (operation == nullptr) {
        plan.markFailed();
        ASDSIP_LOG(ERROR) << "GetCore3D returned null operation (unsupported type on this platform).";
        throw std::runtime_error("make 3D plan failed: core is null.");
    }
    plan.steps.push_back(PlanStep{std::move(operation)});
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
                // Check both contiguous passes using their actual row counts, including rectangular batches.
                const auto axisSupported = [batchSize](int64_t n, int64_t other) {
                    if (n < 1 || n > 8192 || other < 1 || other > 8192)
                        return false;
                    return OpParam::ValidMixedCommon(
                        {n, int64_t(batchSize) * other, OpParam::MixedCommonPass::Complex, false, true});
                };
                const bool commonAxes = !UseOpsFftKernelBackend() && axisSupported(fftSizeX, fftSizeY) &&
                                        axisSupported(fftSizeY, fftSizeX);
                if (commonAxes || (radixX == K_RADIX_2 && radixY == K_RADIX_2)) {
                    // Continue into the ordinary transpose/1D composition below.
                } else if (fftSizeX > 1 && fftSizeY > 1 &&
                           Support(deDuplicates(orderedFactorize(fftSizeX)), RADIX_ARCH35_C2C_MIX) &&
                           Support(deDuplicates(orderedFactorize(fftSizeY)), RADIX_ARCH35_C2C_MIX)) {
                    // 任一维长度为 1 时 orderedFactorize 得到空因子集、Support 对空集恒真，
                    // 会让空 radix 表进入混合基核（stage 循环零迭代、输出缓冲不写入仍返回
                    // 成功），准入侧显式拒绝（issue #352）
                    addFFT2DStep(plan, radixX, radixY);
                    break;
                } else {
                    ASDSIP_LOG(ERROR) << "ASCEND_950 2D C2C arch35 unsupported factorization: fftSizeX=" << fftSizeX
                                      << ", fftSizeY=" << fftSizeY;
                    throw std::runtime_error("ASCEND_950 2D C2C arch35 unsupported factorization.");
                }
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

void addFFTC2CFirstTwoDimsStep(FFTPlan& plan, int64_t batchSize, int64_t fftSizeX, int64_t fftSizeY, int64_t fftSizeZ)
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

// workspace 绑定/需求量双重守卫（issue #162/#486，asdFftExecV2/asdFftExecV2Separated 共用）：
// 1) 绑定守卫（#162）：需要 workspace/temp cache 的 plan 必须已 SetWorkspace；
// 2) 需求量守卫（#486）：plan.workspaceSize 仅由 asdFftGetWorkspaceSize 记录，跳过
//    GetWorkspaceSize 直接 SetWorkspace+Exec 时为 0，Workspace 以 capacity_==0
//    "容量未知"语义构造，#195 的 allocate 越界防护被整体旁路，必须先记录需求量
inline AspbStatus CheckWorkspaceBinding(const FFTPlan& plan)
{
    const bool workspaceRequired = shouldAllocTempCaches(plan) || shouldAllocWorkspace(plan);
    if (workspaceRequired && !plan.hasWorkspace()) {
        ASDSIP_LOG(ERROR) << "workspace has not been allocated.";
        return AsdSip::ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }
    if (workspaceRequired && plan.workspaceSize == 0) {
        ASDSIP_LOG(ERROR) << "workspace size has not been queried, call asdFftGetWorkspaceSize before exec.";
        return AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM;
    }
    return AsdSip::ErrorType::ACL_SUCCESS;
}

void recycleInterCaches(FFTPlan& plan, workspace::Workspace& wkspace)
{
    int64_t num = plan.steps.size() <= K_FACTOR_2 ? 1 : K_FACTOR_2;
    for (int64_t i = 0; i < num; i++) {
        // 容错回收（issue #486）：异常 catch 内清理时 allocate 可能部分完成后即抛出，
        // 此时无可回收项，直接 recycleLast 会在 catch 内二次抛异常穿透 C 接口
        if (wkspace.CanRecycle()) {
            wkspace.recycleLast();
        }
    }
}

// public interface
AspbStatus asdFftCreate(asdFftHandle& handle)
{
    std::lock_guard<std::mutex> lock(fft_mtx);
    // makePlan 的 new/哈希表插入可能因宿主内存不足抛 std::bad_alloc：在 API 边界统一
    // 翻译为错误码，异常不逃逸 AspbStatus API（issue #563，对齐 MakePlan1D/addFFTSteps
    // 等入口的 #131 约定）
    try {
        handle = FFTPlanCache::makePlan();
    } catch (const std::exception& e) {
        try {
            ASDSIP_LOG(ERROR) << "asdFftCreate failed: " << e.what();
        } catch (...) {
            // 日志格式化自身在 OOM 下可能再抛 bad_alloc，不得逃出错误码翻译路径
        }
        handle = nullptr;
        return ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }
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

// direction 白名单校验：plan 正反向语义由 FFTPlan::isForward() 以 direction ==
// ASCEND_FFT_FORWARD 判定（fft_plan.cpp），非法枚举值一律被判为反向，plan 被
// 静默按逆变换执行并返回成功、调用方得到符号相反的结果且无告警（issue #562，
// 与 fftType 白名单 2D/3D #124、1D #551 口径对齐）
AspbStatus CheckFftDirection(asdFftDirection direction)
{
    if (direction != asdFftDirection::ASCEND_FFT_FORWARD && direction != asdFftDirection::ASCEND_FFT_INVERSE) {
        ASDSIP_LOG(ERROR) << "Invalid direction " << static_cast<int>(direction)
                          << ", only FORWARD/INVERSE are supported.";
        return ErrorType::ACL_ERROR_INVALID_PARAM;
    }
    return AsdSip::ErrorType::ACL_SUCCESS;
}

// 1D plan 的 batch 布局设置：纵向 plan 的 batch 记录在 fftStrides（fft 沿跨步内存执行），
// 横向 plan 的 batch 记录在 batchSize（fft 沿连续尾维执行）
AspbStatus Setup1dBatchLayout(FFTPlan& plan, asdFft1dDimType dimType, int64_t batchSize)
{
    // dimType 白名单：非法枚举值落入 else 分支会静默按 HORIZONTAL 处理，还会绕过
    // 950 平台 VERTICAL+batchSize>1 的不支持拦截（issue #562）
    if (dimType != asdFft1dDimType::ASCEND_FFT_HORIZONTAL && dimType != asdFft1dDimType::ASCEND_FFT_VERTICAL) {
        ASDSIP_LOG(ERROR) << "Invalid dimType " << static_cast<int>(dimType) << " for 1D plan.";
        return ErrorType::ACL_ERROR_INVALID_PARAM;
    }
    if (dimType == asdFft1dDimType::ASCEND_FFT_VERTICAL) {
        // 950 平台无 stride 核心：纵向 1D FFT 依赖的 stride 会被核心静默丢弃、按连续内存
        // 计算导致结果错误，对 stride>1（即 batchSize>1）显式返回不支持而非静默算错（issue #159）
        if (batchSize > 1 && Mki::PlatformInfo::Instance().GetPlatformType() == Mki::PlatformType::ASCEND_950) {
            ASDSIP_LOG(ERROR) << "vertical 1D FFT with batchSize > 1 is not supported on ASCEND_950.";
            return ErrorType::ACL_ERROR_API_NOT_SUPPORT;
        }
        plan.batchSize = 1;
        plan.fftStrides = {batchSize};
    } else {
        plan.batchSize = batchSize;
        plan.fftStrides = {1};
    }
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
    // 步骤构造链 addFFTSteps(int)/getCore(unsigned) 存在 int64→int→unsigned
    // 窄化（issue #175）：batch 超过 INT32_MAX 时截断为错误值，与 plan.batchSize
    // 真值形成两套矛盾语义。入口拦截，保证窄化无损。
    if (batchSize > static_cast<int64_t>(std::numeric_limits<int32_t>::max())) {
        ASDSIP_LOG(ERROR) << "batch_size exceeds INT32_MAX.";
        return ErrorType::ACL_ERROR_INVALID_PARAM;
    }

    // 1D 仅支持 C2C/C2R/R2C/C2C_SEP：STFT_* 或非法枚举值会落入 getCore 的 default 分支
    // 产出 FFTCoreAny 内核的 plan 且被标记为已初始化并返回成功，后续所有 exec 调用均被
    // 拒绝、句柄永久不可执行（issue #551，与 2D/3D 的 fftType 白名单口径对齐，issue #124）
    if (fftType != asdFftType::ASCEND_FFT_C2C && fftType != asdFftType::ASCEND_FFT_C2R &&
        fftType != asdFftType::ASCEND_FFT_R2C && fftType != asdFftType::ASCEND_FFT_C2C_SEP) {
        ASDSIP_LOG(ERROR) << "asdFftMakePlan1D: unsupported fftType " << static_cast<int>(fftType) << " for 1D plan.";
        return ErrorType::ACL_ERROR_INVALID_PARAM;
    }

    // direction 白名单校验：非法枚举值静默按反向变换执行并返回成功（issue #562）
    if (CheckFftDirection(direction) != AsdSip::ErrorType::ACL_SUCCESS) {
        return ErrorType::ACL_ERROR_INVALID_PARAM;
    }

    // 重复初始化守卫：与头文件"Any given handle can only be initialized once"契约对齐（issue #137）
    if (plan.isInitialized()) {
        ASDSIP_LOG(ERROR) << "plan is already initialized, repeated initialization is not allowed.";
        return ErrorType::ACL_ERROR_INVALID_PARAM;
    }

    plan.fftType = fftType;
    plan.direction = direction;
    plan.fftSizes = {fftSize};

    // 纵向/横向 batch 布局设置抽取为独立函数：1D 入口校验链较长，保持单函数规模合规
    AspbStatus layoutStatus = Setup1dBatchLayout(plan, dimType, batchSize);
    if (layoutStatus != AsdSip::ErrorType::ACL_SUCCESS) {
        return layoutStatus;
    }
    // 内部初始化路径可能抛出异常（如不支持的长度），在 API 边界统一翻译为错误码（issue #131）
    try {
        addFFTSteps(plan, 0, batchSize);
    } catch (const std::exception& e) {
        plan.steps.clear();
        ASDSIP_LOG(ERROR) << "asdFftMakePlan1D failed: " << e.what();
        return ErrorType::ACL_ERROR_INVALID_PARAM;
    }
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

// 双尺寸 asdFftMakePlan1D 重载已删除（issue #233）：无头文件声明、全仓零调用的
// 死代码；且其 fftSizeY 仅在 dim==1 时被消费、dim>=2 时被静默忽略（无校验无告
// 警），一旦未来被暴露 dim>=2 的调用会静默产出与预期不符的 plan。
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

    // 2D 仅支持 C2C/C2R/R2C：如 C2C_SEP 落入 init2DSteps 的 default 分支会得到 0 步 plan
    // 且仍被标记为已初始化，exec 阶段静默无计算（issue #124）
    if (fftType != asdFftType::ASCEND_FFT_C2C && fftType != asdFftType::ASCEND_FFT_C2R &&
        fftType != asdFftType::ASCEND_FFT_R2C) {
        ASDSIP_LOG(ERROR) << "asdFftMakePlan2D: unsupported fftType " << static_cast<int>(fftType) << " for 2D plan.";
        return ErrorType::ACL_ERROR_INVALID_PARAM;
    }

    // direction 白名单校验：非法枚举值静默按反向变换执行并返回成功（issue #562）
    if (CheckFftDirection(direction) != AsdSip::ErrorType::ACL_SUCCESS) {
        return ErrorType::ACL_ERROR_INVALID_PARAM;
    }

    FFTPlan& plan = FFTPlanCache::getPlan(handle);

    // 重复初始化守卫：与头文件"Any given handle can only be initialized once"契约对齐（issue #137）
    if (plan.isInitialized()) {
        ASDSIP_LOG(ERROR) << "plan is already initialized, repeated initialization is not allowed.";
        return ErrorType::ACL_ERROR_INVALID_PARAM;
    }

    plan.fftType = fftType;
    plan.direction = direction;
    plan.batchSize = batchSize;
    plan.fftSizes = {fftSizeX, fftSizeY};
    plan.fftStrides = {1, 1};

    // 内部初始化路径可能抛出异常（如 950 不支持的分解），在 API 边界统一翻译为错误码（issue #131）
    try {
        init2DSteps(plan);
    } catch (const std::exception& e) {
        plan.steps.clear();
        ASDSIP_LOG(ERROR) << "asdFftMakePlan2D failed: " << e.what();
        return ErrorType::ACL_ERROR_INVALID_PARAM;
    }
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

    // 3D 支持 C2C/C2R/R2C/C2C_SEP，其余类型（如 STFT_*）落入 init3DSteps 的 default 分支会得到
    // 0 步 plan 且仍被标记为已初始化（issue #124）
    if (fftType != asdFftType::ASCEND_FFT_C2C && fftType != asdFftType::ASCEND_FFT_C2R &&
        fftType != asdFftType::ASCEND_FFT_R2C && fftType != asdFftType::ASCEND_FFT_C2C_SEP) {
        ASDSIP_LOG(ERROR) << "asdFftMakePlan3D: unsupported fftType " << static_cast<int>(fftType) << " for 3D plan.";
        return ErrorType::ACL_ERROR_INVALID_PARAM;
    }

    // direction 白名单校验：非法枚举值静默按反向变换执行并返回成功（issue #562）
    if (CheckFftDirection(direction) != AsdSip::ErrorType::ACL_SUCCESS) {
        return ErrorType::ACL_ERROR_INVALID_PARAM;
    }

    FFTPlan& plan = FFTPlanCache::getPlan(handle);

    // 重复初始化守卫：与头文件"Any given handle can only be initialized once"契约对齐（issue #137）
    if (plan.isInitialized()) {
        ASDSIP_LOG(ERROR) << "plan is already initialized, repeated initialization is not allowed.";
        return ErrorType::ACL_ERROR_INVALID_PARAM;
    }

    plan.fftType = fftType;
    plan.direction = direction;
    plan.batchSize = batchSize;
    plan.fftSizes = {fftSizeX, fftSizeY, fftSizeZ};
    plan.fftStrides = {1, 1, 1};

    // 内部初始化路径可能抛出异常（如 950 不支持的分解），在 API 边界统一翻译为错误码（issue #131）
    try {
        init3DSteps(plan);
    } catch (const std::exception& e) {
        plan.steps.clear();
        ASDSIP_LOG(ERROR) << "asdFftMakePlan3D failed: " << e.what();
        return ErrorType::ACL_ERROR_INVALID_PARAM;
    }
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
    try {
        workspaceSize = 0;
        if (shouldAllocTempCaches(plan)) {
            workspaceSize += computeTempCachesSize(plan);
        }
        workspaceSize += computeWorkspaceSize(plan);
        // 记录需求量供执行期 Workspace 越界校验使用（issue #195）
        plan.workspaceSize = workspaceSize;
    } catch (const std::exception& e) {
        ASDSIP_LOG(ERROR) << "asdFftGetWorkspaceSize failed: " << e.what();
        return AsdSip::ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }
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
    // 同步失败（stream 非法/运行时错误等）必须上抛错误码，不得静默吞掉（issue #177）
    if (MkiRtStreamSynchronize(plan.stream) != 0) {
        ASDSIP_LOG(ERROR) << "MkiRtStreamSynchronize failed.";
        return AsdSip::ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }
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

// ops-fft 后端统一拦截: 请求 ops-fft 时校验 plan 各 step 的适配性与 stub 可用性,
// 严格档 (ops-fft): 任一不满足即返回 ACL_ERROR_API_NOT_SUPPORT, 不回退 internal;
// 自动档 (ops-fft-auto): 不满足的 step 打 ERROR 日志后放行, 由各 core Run 内的
// 逐 step 分支 (backend 开启且 stub 就绪才走 RunViaOpsFft) 自动回退 internal
AspbStatus OpsFftBackendUnifyIntercept(const FFTPlan& plan)
{
    if (!UseOpsFftKernelBackend()) {
        return ErrorType::ACL_SUCCESS;
    }
    const bool autoFallback = UseOpsFftAutoFallback();
    for (const auto& step : plan.steps) {
        const bool adapted = step.operation->OpsFftBackendAdapted();
        if (adapted && step.operation->OpsFftBackendStubReady()) {
            continue;
        }
        // 回退提示用 ERROR 级: 默认日志级别下 WARN 会被完全过滤, 静默回退不可见
        const char* reason = adapted ?
                                 "ops-fft kernel stub unavailable (ops-fft library not linked or symbol missing)" :
                                 "current FFT path is not adapted to ops-fft kernel";
        ASDSIP_LOG(ERROR) << "SIP_FFT_BACKEND=" << (autoFallback ? "ops-fft-auto" : "ops-fft") << ": " << reason
                          << (autoFallback ? ", auto fallback to internal kernel." : ", aborting.");
        if (!autoFallback) {
            return ErrorType::ACL_ERROR_API_NOT_SUPPORT;
        }
    }
    return ErrorType::ACL_SUCCESS;
}

AspbStatus asdFftExecV2(FFTPlan& plan, const aclTensor* input, const aclTensor* output)
{
    if (!plan.isInitialized()) {
        ASDSIP_LOG(ERROR) << "plan is not initialized.";
        return ErrorType::ACL_ERROR_INVALID_PARAM;
    }

    // workspace 绑定/需求量双重守卫（issue #162/#486）
    AspbStatus wsGuardStatus = CheckWorkspaceBinding(plan);
    if (wsGuardStatus != ErrorType::ACL_SUCCESS) {
        return wsGuardStatus;
    }

    // 带容量构造：allocate 越界即抛异常，防止用户传入小于需求量的 workspace 静默越界写（issue #195）
    workspace::Workspace wkspace(plan.workspaceAddr, plan.workspaceSize);
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

    AspbStatus interceptStatus = OpsFftBackendUnifyIntercept(plan);
    if (interceptStatus != ErrorType::ACL_SUCCESS) {
        return interceptStatus;
    }

    if (plan.steps.size() == 1) {
        // 核心在 Run 内做对齐/重叠校验并可能抛出异常，在 API 边界统一翻译为错误码，
        // 避免异常穿透 C 接口（issue #160）
        try {
            plan.steps[0].operation->Run(inputData, outputData, plan.stream, wkspace);
        } catch (const std::exception& e) {
            ASDSIP_LOG(ERROR) << "asdFftExecV2 failed: " << e.what();
            return AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM;
        }
        return AsdSip::ErrorType::ACL_SUCCESS;
    }
    int ping = 0;
    // 多步临时缓存分配纳入异常穿透防护（issue #486）：用户绑定小容量 workspace 触发
    // allocate 越界时异常不得穿透 C 接口；成功路径与异常路径统一由回收容错保障
    try {
        std::vector<void*> tmpCache = allocInterCachesV2(plan, wkspace);
        for (int64_t i = 0; i < static_cast<int64_t>(plan.steps.size()); i++) {
            void* tmpIn = i == 0 ? inputData : tmpCache[1 - ping];
            void* tmpOut = i == static_cast<int64_t>(plan.steps.size()) - 1 ? outputData : tmpCache[ping];

            plan.steps[i].operation->Run(tmpIn, tmpOut, plan.stream, wkspace);

            ping = 1 - ping;
        }
    } catch (const std::exception& e) {
        recycleInterCaches(plan, wkspace);
        ASDSIP_LOG(ERROR) << "asdFftExecV2 failed: " << e.what();
        return AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM;
    }
    recycleInterCaches(plan, wkspace);

    return AsdSip::ErrorType::ACL_SUCCESS;
}

// plan 的 batch 规模推导：纵向 1D plan 的 batch 记录在 fftStrides[0]（plan.batchSize
// 固定为 1），横向 1D 与多维 plan 记录在 plan.batchSize。多维 plan 的 fftStrides 由
// init*Steps 改写为变换维元素步长（如 3D C2C_SEP 改写为 {Y*Z, Z, 1}），不承载 batch
// 语义，不得参与推导（matchSeparatedPlan_/#550 与 matchC2R_/matchR2C_/#561 共用）
int64_t PlanBatchExtent(const FFTPlan& plan)
{
    int64_t planBatch = plan.batchSize;
    if (plan.fftSizes.size() == 1 && plan.fftStrides.size() == 1 && plan.fftStrides[0] > planBatch) {
        planBatch = plan.fftStrides[0];
    }
    return planBatch;
}

// 前导维度（索引 0..leadingDimCount-1）元素规模须恰为 plan 的 batch：不足会越界读写，
// 超出语义不符（matchSeparatedPlan_/#550 与 matchC2R_/matchR2C_/#561 共用）；乘法上界
// 先判后乘防溢出，dims[j] 为 0/负先行拒绝、除法安全
bool LeadingExtentMatchesBatch(const int64_t* dims, int64_t leadingDimCount, int64_t planBatch)
{
    if (planBatch <= 0) {
        return false;
    }
    int64_t leadingExtent = 1;
    for (int64_t j = 0; j < leadingDimCount; j++) {
        if (dims[j] <= 0 || leadingExtent > planBatch / dims[j]) {
            return false;
        }
        leadingExtent *= dims[j];
    }
    return leadingExtent == planBatch;
}

// 1D C2C plan 的张量契约校验（issue #471/#472）：rank 恰为 2，横向 (batchSize,
// fftSize)、纵向 (fftSize, batchSize)。仅比对尾维既不校验 rank 也不校验 batch 维：
// [1, fftSize]、[7, batch, fftSize] 等错误张量会通过 match 直达内核按 plan 的 batch
// 规模读写造成设备越界（issue #471）；纵向 plan 尾维是 batch 维，仅比对尾维会把
// 合法的 (fftSize, batchSize) 误判为不匹配（issue #472）。
bool matchC2C1D_(const FFTPlan& plan, const int64_t* dims, uint64_t tensorDimSize)
{
    if (tensorDimSize != 2) {
        return false;
    }
    // 纵向 plan（batch 记录在 fftStrides[0]、plan.batchSize 置 1，见 Setup1dBatchLayout）：
    // 接受文档承诺的 (fftSize, batchSize)。
    if (plan.batchSize == 1 && plan.fftStrides[0] != 1) {
        return (dims[0] == plan.fftSizes[0]) && (dims[1] == plan.fftStrides[0]);
    }
    // 纵向 batchSize==1 退化为 fftStrides={1}，与横向 plan 内部描述一致、数值等价，
    // 无法从 plan 字段反推布局：(batchSize, fftSize) 与 (fftSize, batchSize) 两种
    // 形状均接受；横向 batchSize>1 时仅接受 (batchSize, fftSize)。
    if (plan.batchSize == 1 && plan.fftStrides[0] == 1) {
        return (dims[0] == 1 && dims[1] == plan.fftSizes[0]) || (dims[0] == plan.fftSizes[0] && dims[1] == 1);
    }
    return (dims[0] == plan.batchSize) && (dims[1] == plan.fftSizes[0]);
}

bool matchC2C_(const FFTPlan& plan, const aclTensor* input)
{
    if (input == nullptr || plan.fftStrides.empty() || plan.fftSizes.empty()) {
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

    if (plan.fftSizes.size() == 1) {
        bool matched = matchC2C1D_(plan, dims, tensorDimSize);
        delete[] dims;
        dims = nullptr;
        return matched;
    }

    // 尾维形状比较与 stride 语义正交：纵向（strided）plan 不能跳过——否则
    // 尺寸不符的张量通过 match 直达内核造成设备越界读写；与 matchC2R_/
    // matchR2C_ 口径对齐（issue #172）
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
    if (input == nullptr || plan.fftStrides.empty() || plan.fftSizes.empty()) {
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
    // batch 覆盖校验：尾维匹配只核对 fft 尺寸，前导维度元素规模不足 plan 的 batch 时
    // （如 batch=1000 的 plan 配 [1, N/2+1] 张量）张量直达核心，核心按 plan.batchSize
    // 读写造成设备越界（issue #561，对齐 matchSeparatedPlan_ #550 口径）
    if (!LeadingExtentMatchesBatch(numOfDims, dimIterIdx + 1, PlanBatchExtent(plan))) {
        delete[] numOfDims;
        return false;
    }
    delete[] numOfDims;
    return true;
}

bool matchR2C_(const FFTPlan& plan, const aclTensor* input)
{
    if (input == nullptr || plan.fftStrides.empty() || plan.fftSizes.empty()) {
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
    // 防护: 输入张量维度数小于 plan 的 fftSizes 维度数时, 循环索引会变为负值导致越界读
    if (static_cast<int64_t>(plan.fftSizes.size()) > last + 1) {
        delete[] dims;
        dims = nullptr;
        return false;
    }
    int64_t i = last;
    for (auto rit = plan.fftSizes.rbegin(); rit != plan.fftSizes.rend(); rit++, i--) {
        int64_t dim = dims[i];
        if (dim != *rit) {
            delete[] dims;
            dims = nullptr;
            return false;
        }
    }
    // batch 覆盖校验：口径与 matchC2R_ 一致，缺 batch 张量直达核心造成设备越界
    // （issue #561，对齐 matchSeparatedPlan_ #550 口径）
    if (!LeadingExtentMatchesBatch(dims, i + 1, PlanBatchExtent(plan))) {
        delete[] dims;
        return false;
    }
    delete[] dims;
    dims = nullptr;
    return true;
}

// 分离执行 plan 与张量形状一致性校验：分离核心按 plan 记录的 batch/fft 尺寸启动读写，
// 形状不符的张量直达核心会造成设备越界读写且返回成功（issue #550）。尾维口径与
// matchC2C_/matchR2C_ 对齐（#172 修复思路）；batch 维按剩余前导维度的元素规模与 plan
// 的 batch 规模比对（1D 纵向 plan 的 batch 记录在 fftStrides 中）
bool matchSeparatedPlan_(const FFTPlan& plan, const int64_t* dims, uint64_t dimsNum)
{
    if (dims == nullptr || plan.fftSizes.empty() || plan.fftStrides.empty()) {
        return false;
    }

    // 防护: 张量维度数小于 plan 的 fftSizes 维度数时, 循环索引会变为负值导致越界读
    if (static_cast<int64_t>(plan.fftSizes.size()) > static_cast<int64_t>(dimsNum)) {
        return false;
    }

    int64_t i = static_cast<int64_t>(dimsNum) - 1;
    for (auto iterPtr = plan.fftSizes.rbegin(); iterPtr != plan.fftSizes.rend(); iterPtr++, i--) {
        if (dims[i] != *iterPtr) {
            return false;
        }
    }

    // batch 规模推导仅限 1D plan（推导口径详见 PlanBatchExtent）：前导维度（索引 0..i）
    // 元素规模须恰为 plan 的 batch，不足越界读写、超出语义不符（口径详见
    // LeadingExtentMatchesBatch，#550/#561 统一实现）
    return LeadingExtentMatchesBatch(dims, i + 1, PlanBatchExtent(plan));
}

AsdSip::AspbStatus asdFftExecC2C(asdFftHandle handle, const aclTensor* input, const aclTensor* output)
{
    std::lock_guard<std::mutex> lock(fft_mtx);
    if (!FFTPlanCache::doesPlanExist(handle)) {
        ASDSIP_LOG(ERROR) << "Invalid handle.";
        return ErrorType::ACL_ERROR_INVALID_PARAM;
    }
    FFTPlan& plan = FFTPlanCache::getPlan(handle);

    ASDSIP_ECHECK(plan.isInitialized(), "plan is not initialized.", ErrorType::ACL_ERROR_INVALID_PARAM);
    // 变体与 plan 类型一致性校验：match* 仅检查 dtype/形状，C2R plan 可经 C2C 入口
    // 以错误核心静默执行并返回成功（issue #213）
    if (plan.fftType != asdFftType::ASCEND_FFT_C2C) {
        ASDSIP_LOG(ERROR) << "plan fftType does not match asdFftExecC2C.";
        return ErrorType::ACL_ERROR_INVALID_PARAM;
    }
    if (!matchC2C_(plan, input)) {
        ASDSIP_LOG(ERROR) << "input does not match plan.";
        return ErrorType::ACL_ERROR_INVALID_PARAM;
    }

    if (!matchC2C_(plan, output)) {
        ASDSIP_LOG(ERROR) << "output does not match plan.";
        return ErrorType::ACL_ERROR_INVALID_PARAM;
    }

    // 纵向 stride>1 的 1D C2C 核心存在数值缺陷（910B/CANN 9.3.0 实测纵向输出与
    // 横向参照 maxRelErr 约 4.3e11，见 issue #472 评论）：修复合入前在 API 边界显式
    // 拒绝执行，与 Setup1dBatchLayout 对 950 平台纵向 stride 的 API_NOT_SUPPORT
    // 口径一致，避免 ACL_SUCCESS 下的静默数据错误（检视意见：不静默放行）。
    if (plan.fftSizes.size() == 1 && plan.batchSize == 1 && plan.fftStrides[0] > 1) {
        ASDSIP_LOG(ERROR) << "1D C2C with vertical stride > 1 is not supported until the stride kernel is fixed.";
        return ErrorType::ACL_ERROR_API_NOT_SUPPORT;
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

    ASDSIP_ECHECK(plan.isInitialized(), "plan is not initialized.", ErrorType::ACL_ERROR_INVALID_PARAM);
    // 变体与 plan 类型一致性校验（issue #213）
    if (plan.fftType != asdFftType::ASCEND_FFT_C2R) {
        ASDSIP_LOG(ERROR) << "plan fftType does not match asdFftExecC2R.";
        return ErrorType::ACL_ERROR_INVALID_PARAM;
    }
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

    ASDSIP_ECHECK(plan.isInitialized(), "plan is not initialized.", ErrorType::ACL_ERROR_INVALID_PARAM);
    // 变体与 plan 类型一致性校验（issue #213）
    if (plan.fftType != asdFftType::ASCEND_FFT_R2C) {
        ASDSIP_LOG(ERROR) << "plan fftType does not match asdFftExecR2C.";
        return ErrorType::ACL_ERROR_INVALID_PARAM;
    }
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
        ASDSIP_LOG(ERROR) << "plan is not initialized.";
        return ErrorType::ACL_ERROR_INVALID_PARAM;
    }

    // workspace 绑定/需求量双重守卫（issue #162/#486），口径与 asdFftExecV2 一致
    AspbStatus wsGuardStatus = CheckWorkspaceBinding(plan);
    if (wsGuardStatus != ErrorType::ACL_SUCCESS) {
        return wsGuardStatus;
    }

    // 带容量构造：allocate 越界即抛异常，防止用户传入小于需求量的 workspace 静默越界写（issue #195）
    workspace::Workspace wkspace(plan.workspaceAddr, plan.workspaceSize);
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

    AspbStatus interceptStatus = OpsFftBackendUnifyIntercept(plan);
    if (interceptStatus != ErrorType::ACL_SUCCESS) {
        return interceptStatus;
    }

    if (plan.steps.size() == 1) {
        // 4 指针分离实虚部 Run 的默认实现为空：非分离核心会静默无操作却返回成功，
        // 先探测核心能力再执行（issue #161）
        if (!plan.steps[0].operation->SupportsSeparatedExec()) {
            ASDSIP_LOG(ERROR) << "the core of current plan does not implement separated execution.";
            return AsdSip::ErrorType::ACL_ERROR_INTERNAL_ERROR;
        }
        // 核心异常穿透防护：在 API 边界统一翻译为错误码（issue #160）
        try {
            plan.steps[0].operation->Run(inputRealData, inputImagData, outputRealData, outputImagData, plan.stream,
                                         wkspace);
        } catch (const std::exception& e) {
            ASDSIP_LOG(ERROR) << "asdFftExecV2Separated failed: " << e.what();
            return AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM;
        }
        return AsdSip::ErrorType::ACL_SUCCESS;
    }
    // 步骤数不为 1 时分离实虚部执行无多步 ping-pong 支持，显式报错而非静默返回成功（issue #124）
    ASDSIP_LOG(ERROR) << "asdFftExecV2Separated expects a single-step plan, but got " << plan.steps.size() << " steps.";
    return AsdSip::ErrorType::ACL_ERROR_INTERNAL_ERROR;
}

AspbStatus asdFftExecC2CSeparated(asdFftHandle handle, const aclTensor* inputReal, const aclTensor* inputImag,
                                  const aclTensor* outputReal, const aclTensor* outputImag)
{
    std::lock_guard<std::mutex> lock(fft_mtx);
    if (!FFTPlanCache::doesPlanExist(handle)) {
        ASDSIP_LOG(ERROR) << "Invalid handle.";
        return ErrorType::ACL_ERROR_INVALID_PARAM;
    }

    // RAII 管理：任一 aclGetViewShape 失败早退时，先前已获取的维度数组自动
    // 释放，消除泄漏（issue #176）
    int64_t* rawInReal = nullptr;
    uint64_t viewDimsNumInReal = 0;
    CHECK_STATUS_WITH_ACL_RETURN(aclGetViewShape(inputReal, &rawInReal, &viewDimsNumInReal), "aclGetViewShape");
    std::unique_ptr<int64_t[]> viewDimsInReal(rawInReal);

    int64_t* rawInImag = nullptr;
    uint64_t viewDimsNumInImag = 0;
    CHECK_STATUS_WITH_ACL_RETURN(aclGetViewShape(inputImag, &rawInImag, &viewDimsNumInImag), "aclGetViewShape");
    std::unique_ptr<int64_t[]> viewDimsInImag(rawInImag);

    int64_t* rawOutReal = nullptr;
    uint64_t viewDimsNumOutReal = 0;
    CHECK_STATUS_WITH_ACL_RETURN(aclGetViewShape(outputReal, &rawOutReal, &viewDimsNumOutReal), "aclGetViewShape");
    std::unique_ptr<int64_t[]> viewDimsOutReal(rawOutReal);

    int64_t* rawOutImag = nullptr;
    uint64_t viewDimsNumOutImag = 0;
    CHECK_STATUS_WITH_ACL_RETURN(aclGetViewShape(outputImag, &rawOutImag, &viewDimsNumOutImag), "aclGetViewShape");
    std::unique_ptr<int64_t[]> viewDimsOutImag(rawOutImag);

    if ((viewDimsNumInReal != viewDimsNumInImag) || (viewDimsNumOutReal != viewDimsNumOutImag) ||
        (viewDimsNumInReal != viewDimsNumOutReal)) {
        ASDSIP_ELOG(ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH) << "invalid input/output format.";
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
        return AsdSip::ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }

    // 白盒扫描修复（!140）新增的分离张量 dtype 校验：与 #176 的 RAII 管理共存，
    // 视图数组已由 unique_ptr 管理，早退路径自动释放，无需手动 delete[]
    const aclTensor* sepTensors[4] = {inputReal, inputImag, outputReal, outputImag};
    const char* sepNames[4] = {"inputReal", "inputImag", "outputReal", "outputImag"};
    for (int i = 0; i < 4; i++) {
        aclDataType sepDtype = aclDataType::ACL_DT_UNDEFINED;
        auto dtypeRet = aclGetDataType(sepTensors[i], &sepDtype);
        if (dtypeRet != 0 || sepDtype != aclDataType::ACL_FLOAT) {
            ASDSIP_ELOG(ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH)
                << "asdFftExecC2CSeparated requires FLOAT " << sepNames[i] << " tensor.";
            return AsdSip::ErrorType::ACL_ERROR_OP_INPUT_NOT_MATCH;
        }
    }
    FFTPlan& plan = FFTPlanCache::getPlan(handle);
    ASDSIP_ECHECK(plan.isInitialized(), "plan is not initialized.", ErrorType::ACL_ERROR_INVALID_PARAM);
    // plan 与张量形状一致性校验（issue #550）：四个张量形状已确认两两一致，取 inputReal
    // 视图与 plan 比对即可；不符直接拒绝，防止小形状张量直达核心越界读写
    if (!matchSeparatedPlan_(plan, viewDimsInReal.get(), viewDimsNumInReal)) {
        ASDSIP_LOG(ERROR) << "input/output shape does not match plan.";
        return ErrorType::ACL_ERROR_INVALID_PARAM;
    }
    return asdFftExecV2Separated(plan, inputReal, inputImag, outputReal, outputImag);
}

} // namespace AsdSip
