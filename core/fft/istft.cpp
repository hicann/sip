/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "utils/assert.h"
#include "utils/sip_lock.h"
#include "fftplan/fft_plan_cache.h"
#include "fft_api.h"
#include "fftcore/fft_core_istft_any.h"
#include "fftoperation/transpose.h"

namespace AsdSip {
using namespace Mki;
using namespace AsdSip;

constexpr int64_t ISTFTANY_CORE_STEP = 2;
constexpr int64_t ISTFT_K_FACTOR_2 = 2;
constexpr int64_t INPUT_SUPPORTED_DIM = 3;
constexpr int64_t OUTPUT_SUPPORTED_DIM = 2;
constexpr int64_t WINDOW_SUPPORTED_DIM = 1;
constexpr int64_t OUT_SIGNAL_LEN_DIM = 1;
constexpr int64_t N_FRAME_DIM = 2;
constexpr int64_t FFT_SIZE_DIM = 1;
constexpr int64_t CHANNEL_DIM = 0;
constexpr int64_t SUPPORTED_MAX_N_FFT = 1500;
constexpr int64_t SUPPORTED_MAX_HOP_LEN = 1500;

inline size_t IstftComputeWorkspaceSize(const AsdSip::FFTPlan& plan)
{
    size_t workspaceSize = 0;
    for (int64_t i = 0; i < static_cast<int64_t>(plan.steps.size()); i++) {
        workspaceSize = std::max(workspaceSize, plan.steps[i].operation->EstimateWorkspaceSize());
    }

    return workspaceSize;
}

inline bool IstftShouldAllocTempCaches(const AsdSip::FFTPlan& plan) { return plan.steps.size() > 1; }

inline bool IstftShouldAllocWorkspace(const AsdSip::FFTPlan& plan) { return IstftComputeWorkspaceSize(plan) != 0; }

std::vector<void*> IstftAllocInterCaches(FFTPlan& plan, workspace::Workspace& wkspace)
{
    std::vector<void*> cache;
    int64_t num = plan.steps.size() <= ISTFT_K_FACTOR_2 ? 1 : ISTFT_K_FACTOR_2;
    for (int64_t i = 0; i < num; i++) {
        size_t dataSize = getAlignedSize(
            plan.eleNum() * GetTensorElementSize(Mki::TensorDType::TENSOR_DTYPE_COMPLEX64) + K_BLOCK_SIZE);
        cache.push_back(wkspace.allocate(dataSize));
    }
    return cache;
}

void IstftRecycleInterCaches(FFTPlan& plan, workspace::Workspace& wkspace)
{
    int64_t num = plan.steps.size() <= ISTFT_K_FACTOR_2 ? 1 : ISTFT_K_FACTOR_2;
    for (int64_t i = 0; i < num; i++) {
        wkspace.recycleLast();
    }
}

void addIstftransposeStep(FFTPlan& plan, int axis0, int axis1, const SVector<int64_t>& dims)
{
    plan.steps.push_back(PlanStep{});
    PlanStep& step = plan.steps.back();
    step.operation = std::unique_ptr<FftOperation>(std::make_unique<Transpose>(axis0, axis1, dims));
}

// 获取istft的core
std::unique_ptr<FftOperation> getIstftCore(const struct IstftDesc& istftAnyParms, FFTPlan& plan)
{
    (void)plan;
    std::unique_ptr<FftOperation> unique(nullptr);
    unique.reset(new FFTCoreIstftAny(istftAnyParms));

    if (!unique->init()) {
        plan.markFailed();
        ASDSIP_LOG(ERROR) << "initialize istft fftcore failed.";
        throw std::runtime_error("initialize istft fftcore failed.");
    }

    return unique;
}

// 增加istft any step
void AddFFTIstftSteps(FFTPlan& plan, struct IstftDesc& istftAnyParms)
{
    plan.steps.push_back(PlanStep{getIstftCore(istftAnyParms, plan)});
}

void InitIstftSteps(FFTPlan& plan, struct IstftDesc& istftAnyParms)
{
    // 由于第一步是transpose, 但在MakePlan1DFft已经加了首个step, 需要调整一下step顺序。
    addIstftransposeStep(plan, FFT_SIZE_DIM, N_FRAME_DIM,
                         {istftAnyParms.channel, istftAnyParms.fftSize, istftAnyParms.nFrames});

    // c2c or c2r
    AddFFTIstftSteps(plan, istftAnyParms);

    // 根据计算逻辑，第一步是本应该transpose, 但在MakePlan1DFft已经加了首个step, 需要调整一下step顺序。
    std::swap(plan.steps[0], plan.steps[1]);

    plan.markInitialized();
}

// c2r 运行前参数校验
bool MatchIstftC2R(const FFTPlan& plan, const aclTensor* input)
{
    aclDataType dataType = aclDataType::ACL_DT_UNDEFINED;
    auto ret = aclGetDataType(input, &dataType);
    if (ret != ::ACL_SUCCESS || dataType != aclDataType::ACL_COMPLEX64) {
        ASDSIP_LOG(ERROR) << "Invalid input or invalid input dtype! Only Support complex64!";
        return false;
    }

    int64_t* realInputDims = nullptr;
    uint64_t realDim = 0;
    ret = aclGetViewShape(input, &realInputDims, &realDim);
    if (ret != ::ACL_SUCCESS || realDim != INPUT_SUPPORTED_DIM) {
        if (realInputDims != nullptr) {
            delete[] realInputDims;
            realInputDims = nullptr;
        }
        ASDSIP_LOG(ERROR) << "Invalid input! Only 3-dimensional tensors are supported!";
        return false;
    }

    int64_t channel = realInputDims[CHANNEL_DIM];
    int64_t fftSize = realInputDims[FFT_SIZE_DIM];
    int64_t nFrames = realInputDims[N_FRAME_DIM];

    delete[] realInputDims;
    realInputDims = nullptr;

    // 校验 俩次的input shape是否一样
    if (channel != plan.istftDesc.channel || fftSize != plan.istftDesc.fftSize || nFrames != plan.istftDesc.nFrames) {
        ASDSIP_LOG(ERROR) << "The input shape is not equal to the shape recorded by plan!";
        return false;
    }

    // 校验1d fft n_fft 和 fft_size 关系约束
    if (fftSize != (plan.fftSizes[0] / ISTFT_K_FACTOR_2 + 1)) {
        ASDSIP_LOG(ERROR) << "Invalid input shape!";
        return false;
    }

    return true;
}

// c2c 运行前参数校验
bool MatchIstftC2C(const FFTPlan& plan, const aclTensor* input)
{
    aclDataType dataType = aclDataType::ACL_DT_UNDEFINED;
    auto ret = aclGetDataType(input, &dataType);
    if (ret != ::ACL_SUCCESS || dataType != aclDataType::ACL_COMPLEX64) {
        ASDSIP_LOG(ERROR) << "Invalid input or invalid input dtype! Only Support complex64!";
        return false;
    }

    int64_t* realInputDims = nullptr;
    uint64_t realDim = 0;
    ret = aclGetViewShape(input, &realInputDims, &realDim);
    if (ret != ::ACL_SUCCESS || realDim != INPUT_SUPPORTED_DIM) {
        if (realInputDims != nullptr) {
            delete[] realInputDims;
            realInputDims = nullptr;
        }
        ASDSIP_LOG(ERROR) << "Invalid input! Failed to get input shape!";
        return false;
    }

    int64_t channel = realInputDims[CHANNEL_DIM];
    int64_t fftSize = realInputDims[FFT_SIZE_DIM];
    int64_t nFrames = realInputDims[N_FRAME_DIM];

    delete[] realInputDims;
    realInputDims = nullptr;

    // 校验 俩次的input shape是否一样
    if (channel != plan.istftDesc.channel || fftSize != plan.istftDesc.fftSize || nFrames != plan.istftDesc.nFrames) {
        ASDSIP_LOG(ERROR) << "The input shape is not equal to the shape recorded by plan!";
        return false;
    }

    // avoid check for fft stride
    if (plan.fftStrides[0] != 1) {
        return true;
    }

    if (fftSize != plan.fftSizes[0]) {
        ASDSIP_LOG(ERROR) << "Invalid input shape!";
        return false;
    }
    return true;
}

AspbStatus asdFftExecIstftV2(FFTPlan& plan, const aclTensor* input, const aclTensor* window_opt,
                             const aclTensor* output)
{
    if (!plan.isInitialized()) {
        ASDSIP_LOG(ERROR) << "plan is not initialized.";
        return ErrorType::ACL_ERROR_INVALID_PARAM;
    }

    // 与 asdFftExecV2 的口径一致：多步临时缓存与单步 scratch 任一需要即要求已设置
    // workspace，否则需 workspace 的 plan 会以空基址进入 Run（issue #162）
    if ((IstftShouldAllocTempCaches(plan) || IstftShouldAllocWorkspace(plan)) && !plan.hasWorkspace()) {
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

    void* windowData = Mki::GetStorageAddr(window_opt);
    if (windowData == nullptr) {
        ASDSIP_LOG(ERROR) << "window_opt aclTensor data is nullptr.";
        return AsdSip::ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }

    void* outputData = Mki::GetStorageAddr(output);
    if (outputData == nullptr) {
        ASDSIP_LOG(ERROR) << "output aclTensor data is nullptr.";
        return AsdSip::ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }

    std::vector<void*> tmpCache = IstftAllocInterCaches(plan, wkspace);

    // 1 transpose transposeOut size: (channel, n_frames, fft_size)
    // 2 c2c or c2r size: (channel, n_frames, n_fft) c2c n_fft = fft_size; c2r n_fft = (fft_size - 1) * 2
    int ping = 0;
    for (int64_t i = 0; i < static_cast<int64_t>(plan.steps.size()); i++) {
        void* tmpIn = i == 0 ? inputData : tmpCache[1 - ping];
        void* tmpOut = i == static_cast<int64_t>(plan.steps.size()) - 1 ? outputData : tmpCache[ping];

        if (i != ISTFTANY_CORE_STEP) {
            plan.steps[i].operation->Run(tmpIn, tmpOut, plan.stream, wkspace);
        } else {
            plan.steps[i].operation->Run(tmpIn, windowData, tmpOut, plan.stream, wkspace);
        }
        ping = 1 - ping;
    }

    IstftRecycleInterCaches(plan, wkspace);

    return AsdSip::ErrorType::ACL_SUCCESS;
}

// 计算out expected signal len
int64_t ComputerExpectedSliceSignalLen(struct IstftDesc& istftAnyParms)
{
    const bool center = istftAnyParms.center;
    const int64_t nFft = istftAnyParms.nFft;
    int64_t expectedOutputSignalLen = nFft + istftAnyParms.hopLengthOpt * (istftAnyParms.nFrames - 1);
    const auto lengthOpt = istftAnyParms.lengthOpt;
    const auto start = center ? nFft / 2 : 0;
    const auto end = [&]() -> int64_t {
        if (lengthOpt > 0) {
            return start + lengthOpt;
        }
        if (center) {
            return -(nFft / 2);
        }
        return expectedOutputSignalLen;
    }();
    return end - start + expectedOutputSignalLen;
}

// istft params check
bool IstftParamsCheck(struct IstftDesc& istftAnyParms, const aclTensor* input)
{
    aclDataType dataType = aclDataType::ACL_DT_UNDEFINED;
    auto ret = aclGetDataType(input, &dataType);
    if (ret != ::ACL_SUCCESS || dataType != aclDataType::ACL_COMPLEX64) {
        ASDSIP_LOG(ERROR) << "Invalid input or invalid input dtype! Only Support complex64!";
        return false;
    }

    int64_t* realInputDims = nullptr;
    uint64_t realDim = 0;
    ret = aclGetViewShape(input, &realInputDims, &realDim);
    if (ret != ::ACL_SUCCESS || realDim != INPUT_SUPPORTED_DIM) {
        if (realInputDims != nullptr) {
            delete[] realInputDims;
            realInputDims = nullptr;
        }
        ASDSIP_LOG(ERROR) << "Invalid input! Only 3-dimensional tensors are supported!";
        return false;
    }

    // record input shape
    auto nFrames = realInputDims[N_FRAME_DIM];
    auto fftSize = realInputDims[FFT_SIZE_DIM];
    auto channel = realInputDims[CHANNEL_DIM];
    istftAnyParms.channel = channel;
    istftAnyParms.nFrames = nFrames;
    istftAnyParms.fftSize = fftSize;

    delete[] realInputDims;
    realInputDims = nullptr;

    // center check
    if (!istftAnyParms.center) {
        istftAnyParms.center = true;
        ASDSIP_LOG(WARN) << "warning: Currently, the supported center is only true.";
    }

    // normalized check
    if (istftAnyParms.normalized) {
        istftAnyParms.normalized = false;
        ASDSIP_LOG(WARN) << "warning: Currently, the supported normalized is only false.";
    }

    // onesidedOpt check
    if (istftAnyParms.onesidedOpt) {
        istftAnyParms.onesidedOpt = false;
        ASDSIP_LOG(WARN) << "warning: Currently, the supported onesidedOpt is only false.";
    }

    // returnComplex check
    if (!istftAnyParms.returnComplex) {
        istftAnyParms.returnComplex = true;
        ASDSIP_LOG(WARN) << "warning: Currently, the supported returnComplex is only true.";
    }

    // framee check
    ASDSIP_CHECK(istftAnyParms.nFrames > 1, "Currently, the nFrames dim value of input should be more than 1",
                 return false);

    // lengthOpt check
    if (istftAnyParms.lengthOpt != 0) {
        istftAnyParms.lengthOpt = 0;
        ASDSIP_LOG(WARN) << "warning: Currently, the supported lengthOpt is only 0.";
    }

    // onesided = true 则 n_fft / 2 + 1 = fft_size; 否则 n_fft = fft_size
    if (istftAnyParms.onesidedOpt) {
        if (istftAnyParms.nFft / 2 + 1 != fftSize) {
            ASDSIP_LOG(ERROR) << "expected the frequency dimension (3rd to the last) of the input tensor to "
                              << "match n_fft / 2 + 1 when onesided=True, but got " << fftSize;
            return false;
        }
    } else {
        if (istftAnyParms.nFft != fftSize) {
            ASDSIP_LOG(ERROR) << "expected the frequency dimension (3rd to the last) of the input tensor to "
                              << "match n_fft when onesided=True, but got " << fftSize;
            return false;
        }
    }

    // 0 < hop_length && hop_length <= win_length)
    if (!(0 < istftAnyParms.hopLengthOpt && istftAnyParms.hopLengthOpt <= istftAnyParms.winLengthOpt)) {
        ASDSIP_LOG(ERROR) << "expected 0 < hop_length <= win_length ";
        return false;
    }

    // 0 < win_length && win_length == n_fft
    if (!(istftAnyParms.winLengthOpt > 0 && istftAnyParms.winLengthOpt == istftAnyParms.nFft)) {
        ASDSIP_LOG(ERROR) << "expected 0 < win_length == n_fft ";
        return false;
    }

    ASDSIP_CHECK(istftAnyParms.nFft < SUPPORTED_MAX_N_FFT, "Currently, the nfft size should be less than 1500",
                 return false);
    ASDSIP_CHECK(istftAnyParms.hopLengthOpt < SUPPORTED_MAX_N_FFT,
                 "Currently, the hoplen size should be less than 1500.", return false);

    int64_t expectedSliceSignalLen = ComputerExpectedSliceSignalLen(istftAnyParms);
    istftAnyParms.outSignalLen = expectedSliceSignalLen;

    return true;
}

bool IstftWindowTensorCheck(struct IstftDesc& istftAnyParms, const aclTensor* window)
{
    int64_t* realInputDims = nullptr;
    uint64_t realDim = 0;
    aclDataType dataType = aclDataType::ACL_DT_UNDEFINED;

    // window dtype float
    auto ret = aclGetDataType(window, &dataType);
    if (ret != ::ACL_SUCCESS) {
        ASDSIP_LOG(ERROR) << "Invalid window tensor!";
        return false;
    }

    if (dataType != aclDataType::ACL_FLOAT) {
        ASDSIP_LOG(ERROR) << "Invalid iwindow tensor dtype! Only support float or complex64!";
        return false;
    }
    istftAnyParms.windowDtype = static_cast<int64_t>(dataType);

    // shape [window_len] window_len = n_fft
    ret = aclGetViewShape(window, &realInputDims, &realDim);
    if (ret != ::ACL_SUCCESS || realDim != WINDOW_SUPPORTED_DIM) {
        ASDSIP_LOG(ERROR) << "Invalid window tensor! Only 1-dimensional tensors are supported!";
        if (realInputDims != nullptr) {
            delete[] realInputDims;
            realInputDims = nullptr;
        }
        return false;
    }
    if (*realInputDims != istftAnyParms.winLengthOpt || *realInputDims != istftAnyParms.nFft) {
        ASDSIP_LOG(ERROR) << "Invalid window tensor! Window length should be equal to n_fft, "
                          << "and Window length should be equal to winLengthOpt";
        delete[] realInputDims;
        realInputDims = nullptr;
        return false;
    }

    delete[] realInputDims;
    realInputDims = nullptr;

    return true;
}

bool IstftOutTensorCheck(struct IstftDesc& istftAnyParms, const aclTensor* output)
{
    // dtype cp64 or fp32
    int64_t* realInputDims = nullptr;
    uint64_t realDim = 0;
    aclDataType dataType = aclDataType::ACL_DT_UNDEFINED;
    auto ret = aclGetDataType(output, &dataType);
    if (ret != ::ACL_SUCCESS) {
        ASDSIP_LOG(ERROR) << "Invalid output tensor!";
        return false;
    }

    if (istftAnyParms.returnComplex && dataType != aclDataType::ACL_COMPLEX64) {
        ASDSIP_LOG(ERROR) << "Invalid output tensor! It should be complex64!";
        return false;
    }

    if (!istftAnyParms.returnComplex && dataType != aclDataType::ACL_FLOAT) {
        ASDSIP_LOG(ERROR) << "Invalid output tensor! It should be float!";
        return false;
    }

    // shape [channel, signalLen]
    ret = aclGetViewShape(output, &realInputDims, &realDim);
    if (ret != ::ACL_SUCCESS || realDim != OUTPUT_SUPPORTED_DIM) {
        if (realInputDims != nullptr) {
            delete[] realInputDims;
            realInputDims = nullptr;
        }
        ASDSIP_LOG(ERROR) << "Invalid out tensor! Only 2-dimensional tensors are supported!";
        return false;
    }

    // compare out and in
    if (istftAnyParms.channel != realInputDims[CHANNEL_DIM]) {
        ASDSIP_LOG(ERROR)
            << "Invalid out tensor shape! The channel dim in out tensor should be equal to in input tensor!";
        delete[] realInputDims;
        realInputDims = nullptr;
        return false;
    }

    int64_t expectedSliceSignalLen = ComputerExpectedSliceSignalLen(istftAnyParms);
    if (expectedSliceSignalLen != realInputDims[OUT_SIGNAL_LEN_DIM]) {
        ASDSIP_LOG(ERROR) << "Invalid signal dim in out tensor! Expected is " << expectedSliceSignalLen
                          << " , actually is " << realInputDims[OUT_SIGNAL_LEN_DIM];
        delete[] realInputDims;
        realInputDims = nullptr;
        return false;
    }

    delete[] realInputDims;
    realInputDims = nullptr;
    return true;
}

bool IstftWindowAndOutTensorCheck(struct IstftDesc& istftAnyParms, const aclTensor* window, const aclTensor* output)
{
    if (!IstftOutTensorCheck(istftAnyParms, output)) {
        return false;
    }
    return IstftWindowTensorCheck(istftAnyParms, window);
}

bool MakePlan1DFft(asdFftHandle handle, struct IstftDesc& istftAnyParms)
{
    // 维度重排，转化为库上能处理的shape
    int64_t sipFftSize = istftAnyParms.fftSize;
    int64_t batch = istftAnyParms.channel * istftAnyParms.nFrames;

    istftAnyParms.sipFftSize = sipFftSize;
    istftAnyParms.sipFftBatch = batch;

    // 根据return_complexOpt 选择c2c 或者 c2r 进行make 1d fft
    if (istftAnyParms.returnComplex) {
        ASDSIP_LOG(DEBUG) << "MakePlan1DFft:" << "ASCEND_FFT_C2C ";
        ASDSIP_CHECK(asdFftMakePlan1D(handle, istftAnyParms.nFft, asdFftType::ASCEND_FFT_C2C,
                                      asdFftDirection::ASCEND_FFT_INVERSE, batch) == AsdSip::ErrorType::ACL_SUCCESS,
                     "c2c asdFftMakePlan1D make plan failed.", return false);
    } else {
        ASDSIP_LOG(DEBUG) << "MakePlan1DFft:" << "ASCEND_FFT_C2R ";
        ASDSIP_CHECK(asdFftMakePlan1D(handle, istftAnyParms.nFft, asdFftType::ASCEND_FFT_C2R,
                                      asdFftDirection::ASCEND_FFT_FORWARD, batch) == AsdSip::ErrorType::ACL_SUCCESS,
                     "c2r asdFftMakePlan1D make plan failed.", return false);
    }
    ASDSIP_LOG(DEBUG) << "MakePlan1DFft" << " success!";
    return true;
}

/*
 * istft make plan
 * 1: make c2c/c2r plan; 2: add istft exec steps
 */
AspbStatus asdFftIstftMakePlan(asdFftHandle handle, const aclTensor* input, const int64_t nFft,
                               const int64_t hopLengthOpt, const int64_t winLengthOpt, const bool center,
                               const bool normalized, const bool onesidedOpt, int64_t lengthOpt,
                               const bool returnComplex)
{
    ASDSIP_ECHECK(FFTPlanCache::doesPlanExist(handle), "fft istft get cached plan failed.",
                  ErrorType::ACL_ERROR_INVALID_PARAM);

    struct IstftDesc istftAnyParms = {0,
                                      0,
                                      0,
                                      0,
                                      0,
                                      0,
                                      0,
                                      nFft,
                                      hopLengthOpt,
                                      winLengthOpt,
                                      center,
                                      normalized,
                                      onesidedOpt,
                                      lengthOpt,
                                      returnComplex};
    ASDSIP_ECHECK(IstftParamsCheck(istftAnyParms, input), "fft istft params check failed.",
                  ErrorType::ACL_ERROR_INVALID_PARAM);

    // make c2c/c2r plan
    ASDSIP_ECHECK(MakePlan1DFft(handle, istftAnyParms), "fft istft make plan failed.",
                  ErrorType::ACL_ERROR_INTERNAL_ERROR);

    FFTPlan& plan = FFTPlanCache::getPlan(handle);
    plan.istftDesc = istftAnyParms;

    // getIstftCore 初始化失败会抛异常，在 API 边界统一翻译为错误码（issue #131 同类加固）
    try {
        InitIstftSteps(plan, istftAnyParms);
    } catch (const std::exception& e) {
        plan.steps.clear();
        ASDSIP_LOG(ERROR) << "asdFftIstftMakePlan failed: " << e.what();
        return ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }
    return AsdSip::ErrorType::ACL_SUCCESS;
}

AspbStatus asdFftExecIstft(asdFftHandle handle, const aclTensor* input, const aclTensor* windowOpt,
                           const aclTensor* output)
{
    std::lock_guard<std::mutex> lock(fft_mtx);
    if (!FFTPlanCache::doesPlanExist(handle)) {
        ASDSIP_LOG(ERROR) << "Invalid handle.";
        return ErrorType::ACL_ERROR_INVALID_PARAM;
    }
    FFTPlan& plan = FFTPlanCache::getPlan(handle);

    // plan and input check
    if (plan.fftType == asdFftType::ASCEND_FFT_C2C) {
        ASDSIP_ECHECK(MatchIstftC2C(plan, input), "input does not match c2c plan", ErrorType::ACL_ERROR_INVALID_PARAM);
    } else {
        ASDSIP_ECHECK(MatchIstftC2R(plan, input), "input does not match c2r plan", ErrorType::ACL_ERROR_INVALID_PARAM);
    }

    // window and out check
    ASDSIP_ECHECK(IstftWindowAndOutTensorCheck(plan.istftDesc, windowOpt, output), "Invalid window or out param!",
                  ErrorType::ACL_ERROR_INVALID_PARAM);

    return asdFftExecIstftV2(plan, input, windowOpt, output);
}
} // namespace AsdSip
