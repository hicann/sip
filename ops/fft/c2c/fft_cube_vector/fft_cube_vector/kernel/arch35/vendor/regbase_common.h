/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
// Selected from the experiment; provenance: ../SOURCE_MANIFEST.json.
#pragma once
#include "acl/acl.h"
#include "c_api/asc_simd.h"
#include "kernel_operator.h"

#include "regbase_large.h"

#include <cstdint>

namespace {

// First scalable power-of-two candidate.  The natural-order N=16384 signal is
// viewed as x[n0][n1], where n=n0*128+n1.  Phase A transforms n0 and writes
// T[k0][n1]; phase B transforms n1 and writes X[k1][k0] in natural order.
constexpr int32_t kLargeN = 16384;
constexpr int32_t kFactor = 128;
constexpr int32_t kComplexBytes = 2 * static_cast<int32_t>(sizeof(float));
constexpr int32_t kFft128Floats = 2 * kFactor;
constexpr uint8_t kL2CacheNormal = 1;

struct CoreTaskRange {
    int64_t start;
    int64_t count;
};

__aicore__ inline CoreTaskRange GetCoreTaskRange(int64_t taskCount)
{
    const int64_t blockCount = static_cast<int64_t>(AscendC::GetBlockNum());
    const int64_t blockIndex = static_cast<int64_t>(AscendC::GetBlockIdx());
    const int64_t quotient = taskCount / blockCount;
    const int64_t remainder = taskCount - quotient * blockCount;

    CoreTaskRange range;
    range.count = quotient + (blockIndex < remainder ? 1 : 0);
    range.start = blockIndex * quotient + (blockIndex < remainder ? blockIndex : remainder);
    return range;
}

// This is the arithmetic core of the device-tested FFT128 lazy Stockham path
// in regbase_small.asc.  The two input halves are already deinterleaved in
// registers.  At return, u/v contain natural outputs 0..63/64..127 in the
// physical lane order described by j=bit_reverse_6(lane).  Keeping this body
// unchanged isolates the new experiment to movement, tiling, and final layout.
__simd_callee__ inline void RunFft128Registers(
    __ubuf__ float* twiddles, AscendC::Reg::RegTensor<float>& uReal, AscendC::Reg::RegTensor<float>& uImag,
    AscendC::Reg::RegTensor<float>& vReal, AscendC::Reg::RegTensor<float>& vImag, AscendC::Reg::RegTensor<float>& work0,
    AscendC::Reg::RegTensor<float>& work1, AscendC::Reg::RegTensor<float>& work2, AscendC::Reg::RegTensor<uint32_t>& j,
    AscendC::Reg::RegTensor<uint32_t>& index, AscendC::Reg::MaskReg& fullMask)
{
    using namespace AscendC;

    Reg::Duplicate(j, uint32_t{0}, fullMask);

    // Stockham half=1.  Interleave advances the lazy physical q order.
    Reg::Sub(work0, uReal, vReal, fullMask);
    Reg::Add(work1, uReal, vReal, fullMask);
    Reg::Interleave(uReal, vReal, work1, work0);
    Reg::Sub(work0, uImag, vImag, fullMask);
    Reg::Add(work1, uImag, vImag, fullMask);
    Reg::Interleave(uImag, vImag, work1, work0);
    Reg::Adds(index, j, uint32_t{1}, fullMask);
    Reg::Interleave(j, index, j, index);

    // Preserve unroll(1): fully cloning these stages increases SSA live ranges
    // enough to spill on DAV3510 in the existing FFT128 implementation.
#pragma unroll 1
    for (uint32_t half = 2U; half < 64U; half <<= 1U) {
        Reg::Adds(index, j, half - 1U, fullMask);
        Reg::Muls(index, index, uint32_t{2}, fullMask);
        Reg::Gather(work0, twiddles, index, fullMask);
        Reg::Mul(work1, vReal, work0, fullMask);
        Reg::Mul(work2, vImag, work0, fullMask);
        Reg::Adds(index, index, uint32_t{1}, fullMask);
        Reg::Gather(work0, twiddles, index, fullMask);
        Reg::Mul(vReal, vReal, work0, fullMask);
        Reg::Mul(vImag, vImag, work0, fullMask);
        Reg::Sub(work1, work1, vImag, fullMask);
        Reg::Add(work2, vReal, work2, fullMask);
        Reg::Sub(vReal, uReal, work1, fullMask);
        Reg::Sub(vImag, uImag, work2, fullMask);
        Reg::Add(uReal, uReal, work1, fullMask);
        Reg::Add(uImag, uImag, work2, fullMask);
        Reg::Interleave(uReal, vReal, uReal, vReal);
        Reg::Interleave(uImag, vImag, uImag, vImag);
        Reg::Adds(index, j, half, fullMask);
        Reg::Interleave(j, index, j, index);
    }

    // Final Stockham half=64.  Do not interleave after this stage: j is the
    // final bit-reverse-6 lane map used by each output policy below.
    constexpr uint32_t half = 64U;
    Reg::Adds(index, j, half - 1U, fullMask);
    Reg::Muls(index, index, uint32_t{2}, fullMask);
    Reg::Gather(work0, twiddles, index, fullMask);
    Reg::Mul(work1, vReal, work0, fullMask);
    Reg::Mul(work2, vImag, work0, fullMask);
    Reg::Adds(index, index, uint32_t{1}, fullMask);
    Reg::Gather(work0, twiddles, index, fullMask);
    Reg::Mul(vReal, vReal, work0, fullMask);
    Reg::Mul(vImag, vImag, work0, fullMask);
    Reg::Sub(work1, work1, vImag, fullMask);
    Reg::Add(work2, vReal, work2, fullMask);
    Reg::Sub(vReal, uReal, work1, fullMask);
    Reg::Sub(vImag, uImag, work2, fullMask);
    Reg::Add(uReal, uReal, work1, fullMask);
    Reg::Add(uImag, uImag, work2, fullMask);
}

// Multiply one physical FFT128 half by a natural-order cross-twiddle plane.
// naturalOffset is 0 for u and 64 for v.  j maps physical lanes to natural k0.
__simd_callee__ inline void MultiplyCrossTwiddle(
    __ubuf__ float* crossTwiddles, AscendC::Reg::RegTensor<float>& real, AscendC::Reg::RegTensor<float>& imag,
    AscendC::Reg::RegTensor<float>& work0, AscendC::Reg::RegTensor<float>& work1, AscendC::Reg::RegTensor<float>& work2,
    AscendC::Reg::RegTensor<uint32_t>& j, AscendC::Reg::RegTensor<uint32_t>& index, uint32_t naturalOffset,
    AscendC::Reg::MaskReg& fullMask)
{
    using namespace AscendC;

    Reg::Adds(index, j, naturalOffset, fullMask);
    Reg::Muls(index, index, uint32_t{2}, fullMask);
    Reg::Gather(work0, crossTwiddles, index, fullMask);
    Reg::Mul(work1, real, work0, fullMask); // ac
    Reg::Mul(work2, imag, work0, fullMask); // bc
    Reg::Adds(index, index, uint32_t{1}, fullMask);
    Reg::Gather(work0, crossTwiddles, index, fullMask);
    Reg::Mul(real, real, work0, fullMask); // ad
    Reg::Mul(imag, imag, work0, fullMask); // bd
    Reg::Sub(work1, work1, imag, fullMask);
    Reg::Add(work2, real, work2, fullMask);
    real = work1;
    imag = work2;
}

// Phase A load policy gathers four/eight/sixteen n1 columns from the compact
// [n0][column] UB tile.  Its store policy fuses W_16384^(n1*k0) and scatters
// into [k0][column], ready for one 2-D MTE3 transfer to workspace T[k0][n1].
template <int32_t Group>
__no_simd_vf_fusion__ __simd_vf__ inline void PhaseAVf(__ubuf__ float* tile, __ubuf__ float* localTwiddles,
                                                       __ubuf__ float* crossTwiddles)
{
    using namespace AscendC;
    static_assert(Group == 4 || Group == 8 || Group == 16);

    Reg::RegTensor<float> uReal, uImag, vReal, vImag;
    Reg::RegTensor<float> work0, work1, work2;
    Reg::RegTensor<uint32_t> j, index;
    Reg::MaskReg fullMask = Reg::CreateMask<float, Reg::MaskPattern::ALL>();

#pragma unroll 1
    for (uint16_t transform = 0; transform < Group; ++transform) {
        // Raw tile layout is [n0][transform], interleaved complex FP32.
        Reg::Arange<int32_t>(reinterpret_cast<Reg::RegTensor<int32_t>&>(index), 0);
        Reg::Muls(index, index, static_cast<uint32_t>(2 * Group), fullMask);
        Reg::Adds(index, index, static_cast<uint32_t>(2 * transform), fullMask);
        Reg::Gather(uReal, tile, index, fullMask);
        Reg::Adds(index, index, uint32_t{1}, fullMask);
        Reg::Gather(uImag, tile, index, fullMask);
        Reg::Adds(index, index, static_cast<uint32_t>(128 * Group - 1), fullMask);
        Reg::Gather(vReal, tile, index, fullMask);
        Reg::Adds(index, index, uint32_t{1}, fullMask);
        Reg::Gather(vImag, tile, index, fullMask);

        RunFft128Registers(localTwiddles, uReal, uImag, vReal, vImag, work0, work1, work2, j, index, fullMask);

        __ubuf__ float* crossPlane = crossTwiddles +
                                     static_cast<uint32_t>(transform) * static_cast<uint32_t>(kFft128Floats);
        MultiplyCrossTwiddle(crossPlane, uReal, uImag, work0, work1, work2, j, index, 0U, fullMask);
        MultiplyCrossTwiddle(crossPlane, vReal, vImag, work0, work1, work2, j, index, 64U, fullMask);

        // Scatter to [k0][transform].  All source values for this transform
        // are already in registers; columns are disjoint, so reusing tile is
        // safe without a VEC_STORE->VEC_LOAD barrier.
        Reg::Muls(index, j, static_cast<uint32_t>(2 * Group), fullMask);
        Reg::Adds(index, index, static_cast<uint32_t>(2 * transform), fullMask);
        Reg::Scatter(tile, uReal, index, fullMask);
        Reg::Adds(index, index, uint32_t{1}, fullMask);
        Reg::Scatter(tile, uImag, index, fullMask);
        Reg::Adds(index, index, static_cast<uint32_t>(128 * Group - 1), fullMask);
        Reg::Scatter(tile, vReal, index, fullMask);
        Reg::Adds(index, index, uint32_t{1}, fullMask);
        Reg::Scatter(tile, vImag, index, fullMask);
    }
}

// Phase B consumes Group contiguous workspace rows [k0][n1].  Each row is one
// FFT128 input.  The final Scatter transposes the natural FFT result into
// [k1][group-column], which MTE3 writes directly to natural output X[k1][k0].
template <int32_t Group>
__no_simd_vf_fusion__ __simd_vf__ inline void PhaseBVf(__ubuf__ float* inputTile, __ubuf__ float* localTwiddles,
                                                       __ubuf__ float* outputTile)
{
    using namespace AscendC;
    static_assert(Group == 4 || Group == 8 || Group == 16);

    Reg::RegTensor<float> uReal, uImag, vReal, vImag;
    Reg::RegTensor<float> work0, work1, work2;
    Reg::RegTensor<uint32_t> j, index;
    Reg::MaskReg fullMask = Reg::CreateMask<float, Reg::MaskPattern::ALL>();

#pragma unroll 1
    for (uint16_t transform = 0; transform < Group; ++transform) {
        __ubuf__ float* inputTransform = inputTile +
                                         static_cast<uint32_t>(transform) * static_cast<uint32_t>(kFft128Floats);
        Reg::LoadAlign<float, Reg::LoadDist::DIST_DINTLV_B32>(uReal, uImag, inputTransform);
        Reg::LoadAlign<float, Reg::LoadDist::DIST_DINTLV_B32>(vReal, vImag, inputTransform + 128U);

        RunFft128Registers(localTwiddles, uReal, uImag, vReal, vImag, work0, work1, work2, j, index, fullMask);

        Reg::Muls(index, j, static_cast<uint32_t>(2 * Group), fullMask);
        Reg::Adds(index, index, static_cast<uint32_t>(2 * transform), fullMask);
        Reg::Scatter(outputTile, uReal, index, fullMask);
        Reg::Adds(index, index, uint32_t{1}, fullMask);
        Reg::Scatter(outputTile, uImag, index, fullMask);
        Reg::Adds(index, index, static_cast<uint32_t>(128 * Group - 1), fullMask);
        Reg::Scatter(outputTile, vReal, index, fullMask);
        Reg::Adds(index, index, uint32_t{1}, fullMask);
        Reg::Scatter(outputTile, vImag, index, fullMask);
    }
}

template <int32_t Group>
__aicore__ inline int64_t LoadTaskCount(__gm__ uint8_t* tiling)
{
    const auto* data = reinterpret_cast<__gm__ FftC2CRegBaseLargeTilingData*>(tiling);
    if (data->batch <= 0 || data->n != kLargeN || data->groupSize != Group) {
        return 0;
    }
    return data->batch * static_cast<int64_t>(kFactor / Group);
}

__aicore__ inline void RunPhaseADoubleBufferedG4(__gm__ float* input, __gm__ float* localTwiddles,
                                                 __gm__ float* crossTwiddles, __gm__ float* workspace,
                                                 int64_t taskCount)
{
    constexpr int32_t Group = 4;
    constexpr int32_t groupsPerBatch = kFactor / Group;
    constexpr uint32_t rowBytes = Group * kComplexBytes;
    constexpr uint32_t tileFloats = 2 * kFactor * Group;
    constexpr uint32_t tileBytes = tileFloats * sizeof(float);

    __ubuf__ float tilePing[tileFloats];
    __ubuf__ float crossPing[tileFloats];
    __ubuf__ float tilePong[tileFloats];
    __ubuf__ float crossPong[tileFloats];
    __ubuf__ float twiddleLocal[kFft128Floats];

    const CoreTaskRange range = GetCoreTaskRange(taskCount);
    if (range.count == 0) {
        return;
    }

    asc_copy_gm2ub_align(twiddleLocal, localTwiddles, 1, static_cast<uint32_t>(kFactor * kComplexBytes), 0, 0, false, 0,
                         0, 0);

    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);

    // Prologue: load the first owned task into ping.  The MTE2_V event also
    // orders the earlier local-twiddle copy before the first Vector read.
    const int64_t firstTask = range.start;
    const int64_t firstBatch = firstTask / groupsPerBatch;
    const int32_t firstGroup = static_cast<int32_t>(firstTask - firstBatch * groupsPerBatch);
    const int32_t firstN1Base = firstGroup * Group;
    const int64_t firstBatchComplexBase = firstBatch * kLargeN;
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    asc_copy_gm2ub_align(tilePing, input + 2 * (firstBatchComplexBase + firstN1Base), kFactor, rowBytes, 0, 0, false, 0,
                         static_cast<uint64_t>(kFactor * kComplexBytes), rowBytes);
    asc_copy_gm2ub_align(crossPing, crossTwiddles + 2 * static_cast<int64_t>(firstN1Base) * kFactor, 1, tileBytes, 0, 0,
                         false, kL2CacheNormal, 0, 0);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

    for (int64_t localTask = 0; localTask < range.count; ++localTask) {
        const bool usePing = (localTask & int64_t{1}) == 0;
        const int32_t eventId = usePing ? EVENT_ID0 : EVENT_ID1;
        __ubuf__ float* tile = usePing ? tilePing : tilePong;
        __ubuf__ float* crossLocal = usePing ? crossPing : crossPong;

        // Explicit next-task prefetch is what creates overlap: its MTE2 work
        // is queued before the current task's Vector function.
        const int64_t nextLocalTask = localTask + 1;
        if (nextLocalTask < range.count) {
            const bool nextUsesPing = (nextLocalTask & int64_t{1}) == 0;
            const int32_t nextEventId = nextUsesPing ? EVENT_ID0 : EVENT_ID1;
            __ubuf__ float* nextTile = nextUsesPing ? tilePing : tilePong;
            __ubuf__ float* nextCross = nextUsesPing ? crossPing : crossPong;
            const int64_t nextTask = range.start + nextLocalTask;
            const int64_t nextBatch = nextTask / groupsPerBatch;
            const int32_t nextGroup = static_cast<int32_t>(nextTask - nextBatch * groupsPerBatch);
            const int32_t nextN1Base = nextGroup * Group;
            const int64_t nextBatchComplexBase = nextBatch * kLargeN;

            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(nextEventId);
            asc_copy_gm2ub_align(nextTile, input + 2 * (nextBatchComplexBase + nextN1Base), kFactor, rowBytes, 0, 0,
                                 false, 0, static_cast<uint64_t>(kFactor * kComplexBytes), rowBytes);
            asc_copy_gm2ub_align(nextCross, crossTwiddles + 2 * static_cast<int64_t>(nextN1Base) * kFactor, 1,
                                 tileBytes, 0, 0, false, kL2CacheNormal, 0, 0);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(nextEventId);
        }

        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventId);
        PhaseAVf<Group>(tile, twiddleLocal, crossLocal);

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventId);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventId);

        const int64_t task = range.start + localTask;
        const int64_t batch = task / groupsPerBatch;
        const int32_t groupIndex = static_cast<int32_t>(task - batch * groupsPerBatch);
        const int32_t n1Base = groupIndex * Group;
        const int64_t batchComplexBase = batch * kLargeN;
        asc_copy_ub2gm_align(workspace + 2 * (batchComplexBase + n1Base), tile, kFactor, rowBytes, 0,
                             static_cast<uint64_t>(kFactor * kComplexBytes), rowBytes);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
    }

    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
}

__aicore__ inline void RunPhaseBDoubleBufferedG16(__gm__ float* workspace, __gm__ float* localTwiddles,
                                                  __gm__ float* output, int64_t taskCount)
{
    constexpr int32_t Group = 16;
    constexpr int32_t groupsPerBatch = kFactor / Group;
    constexpr uint32_t rowBytes = Group * kComplexBytes;
    constexpr uint32_t tileFloats = 2 * kFactor * Group;
    constexpr uint32_t tileBytes = tileFloats * sizeof(float);

    __ubuf__ float inputPing[tileFloats];
    __ubuf__ float outputPing[tileFloats];
    __ubuf__ float inputPong[tileFloats];
    __ubuf__ float outputPong[tileFloats];
    __ubuf__ float twiddleLocal[kFft128Floats];

    const CoreTaskRange range = GetCoreTaskRange(taskCount);
    if (range.count == 0) {
        return;
    }

    asc_copy_gm2ub_align(twiddleLocal, localTwiddles, 1, static_cast<uint32_t>(kFactor * kComplexBytes), 0, 0, false, 0,
                         0, 0);

    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);

    for (int64_t localTask = 0; localTask < range.count; ++localTask) {
        const bool usePing = (localTask & int64_t{1}) == 0;
        const int32_t eventId = usePing ? EVENT_ID0 : EVENT_ID1;
        __ubuf__ float* inputTile = usePing ? inputPing : inputPong;
        __ubuf__ float* outputTile = usePing ? outputPing : outputPong;

        const int64_t task = range.start + localTask;
        const int64_t batch = task / groupsPerBatch;
        const int32_t groupIndex = static_cast<int32_t>(task - batch * groupsPerBatch);
        const int32_t k0Base = groupIndex * Group;
        const int64_t batchComplexBase = batch * kLargeN;

        // PIPE_MTE2 cannot overwrite this input slot until PIPE_V publishes
        // completion of its previous read from the same ping/pong slot.
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventId);
        asc_copy_gm2ub_align(inputTile, workspace + 2 * (batchComplexBase + static_cast<int64_t>(k0Base) * kFactor), 1,
                             tileBytes, 0, 0, false, 0, 0, 0);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventId);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventId);

        // PIPE_V cannot overwrite this output slot until PIPE_MTE3 publishes
        // completion of its previous read from the same ping/pong slot.
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventId);
        PhaseBVf<Group>(inputTile, twiddleLocal, outputTile);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventId);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventId);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventId);

        asc_copy_ub2gm_align(output + 2 * (batchComplexBase + k0Base), outputTile, kFactor, rowBytes, 0,
                             static_cast<uint64_t>(kFactor * kComplexBytes), rowBytes);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventId);
    }

    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
}

} // namespace
