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
// Derived from the frozen RegBase large phase A/B movement policies.
// Arithmetic and per-core ping/pong synchronization are unchanged.
// A writes [batch,k0,Q,n1]; B consumes rows and writes [batch,k1,k0,Q].
namespace {
__aicore__ inline void RunPhaseAExtended(__gm__ float* input, __gm__ float* localTwiddles, __gm__ float* crossTwiddles,
                                         __gm__ float* workspace, int64_t taskCount, int32_t outerFactor)
{
    int qShift = 0;
    for (int v = outerFactor; v > 1; v >>= 1)
        ++qShift;
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
        const int64_t batchComplexBase = (batch >> qShift) * outerFactor * kLargeN +
                                         (batch & (outerFactor - 1)) * kFactor;
        asc_copy_ub2gm_align(workspace + 2 * (batchComplexBase + n1Base), tile, kFactor, rowBytes, 0,
                             static_cast<uint64_t>(outerFactor * kFactor * kComplexBytes), rowBytes);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
    }

    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
}

__aicore__ inline void RunPhaseBExtended(__gm__ float* workspace, __gm__ float* localTwiddles, __gm__ float* output,
                                         int64_t taskCount, int32_t outputWidth)
{
    constexpr int32_t Group = 16;
    const int32_t groupsPerBatch = outputWidth / Group;
    int groupShift = 0;
    for (int v = groupsPerBatch; v > 1; v >>= 1)
        ++groupShift;
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
        const int64_t batch = task >> groupShift;
        const int32_t groupIndex = static_cast<int32_t>(task - batch * groupsPerBatch);
        const int32_t k0Base = groupIndex * Group;
        const int64_t batchComplexBase = batch * outputWidth * kFactor;

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
                             static_cast<uint64_t>(outputWidth * kComplexBytes), rowBytes);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventId);
    }

    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
}
} // namespace
