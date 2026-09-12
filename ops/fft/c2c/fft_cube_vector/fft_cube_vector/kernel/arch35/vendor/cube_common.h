// Selected from the experiment; provenance: ../SOURCE_MANIFEST.json.
#pragma once
/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_operator.h"
#define SET_FLAG(trigger, waiter, e) AscendC::SetFlag<AscendC::HardEvent::trigger##_##waiter>((e))
#define WAIT_FLAG(trigger, waiter, e) AscendC::WaitFlag<AscendC::HardEvent::trigger##_##waiter>((e))
#define PIPE_BARRIER(pipe) AscendC::PipeBarrier<PIPE_##pipe>()

template <typename IN_DTYPE>
__aicore__ inline void SetPadding(IN_DTYPE padValue)
{
    AscendC::SetLoadDataPaddingValue<IN_DTYPE>(padValue);
}

constexpr AscendC::FixpipeConfig kDnUb = {AscendC::CO2Layout::COLUMN_MAJOR, true};
template <typename DTYPE = float>
__aicore__ __attribute__((always_inline)) inline void load_matrix_zZ(AscendC::LocalTensor<DTYPE> dst,
                                                                     AscendC::GlobalTensor<DTYPE> src, int32_t R,
                                                                     int32_t C, int32_t valid_row, int32_t valid_col,
                                                                     int32_t stride)
{
    constexpr int32_t R0 = 16;                 // fractal-группа по M: 16 строк
    constexpr int32_t C0 = 32 / sizeof(DTYPE); // fractal-группа по K/N: 8 float
    constexpr int STRIDE_LIMIT = 65536;        // лимит, который принимает Nd2NzParams

    int64_t srcNdStride = R0 * stride; // шаг между блоками из 16 строк в GM
    int64_t srcNStride = stride;       // шаг между соседними строками в GM

    if (srcNdStride < STRIDE_LIMIT) {
        // Быстрый путь: можно одним или двумя Nd2Nz-копированиями забрать все строки.
        int32_t ndNum = valid_row / R0;   // полные группы по 16 строк
        int32_t remains = valid_row % R0; // хвост меньше 16 строк

        if (ndNum > 0) {
            AscendC::DataCopy(dst, src,
                              AscendC::Nd2NzParams(ndNum, // число ND-матриц (групп по 16 строк)
                                                   R0,    // nValue — строк в одной группе
                                                   valid_col, // dValue — столбцов для копирования
                                                   R0 * stride, // srcNdMatrixStride — шаг между группами в GM
                                                   srcNStride, // srcDValue — шаг между строками внутри группы
                                                   R0,     // dstNzC0Stride — шаг в L1 по fractal C0
                                                   1,      // dstNzNStride
                                                   R0 * C) // dstNzMatrixStride — шаг между группами в L1
            );
        }
        if (remains > 0) {
            // Докачиваем оставшиеся строки (< 16).
            AscendC::DataCopy(dst[ndNum * R0 * C], src[ndNum * R0 * stride],
                              AscendC::Nd2NzParams(1,          // одна неполная группа
                                                   remains,    // сколько строк в хвосте
                                                   valid_col,  // dValue
                                                   0,          // srcNdMatrixStride
                                                   srcNStride, // srcDValue
                                                   R0,         // dstNzC0Stride
                                                   1,          // dstNzNStride
                                                   0)          // dstNzMatrixStride
            );
        }
    } else if (srcNStride < STRIDE_LIMIT) {
        // Средний путь: большой межблочный stride, но построчный stride ещё допустим.
        // Копируем по одной группе из 16 строк за раз.
        int32_t ndNum = valid_row / R0;
        int32_t remains = valid_row % R0;
        for (int32_t i = 0; i < ndNum; i++) {
            AscendC::DataCopy(dst[i * R0 * C], src[i * R0 * stride],
                              AscendC::Nd2NzParams(1,          // ndNum
                                                   R0,         // nValue
                                                   valid_col,  // dValue
                                                   0,          // srcNdMatrixStride
                                                   srcNStride, // srcDValue
                                                   R0,         // dstNzC0Stride
                                                   1,          // dstNzNStride
                                                   0)          // dstNzMatrixStride
            );
        }
        if (remains > 0) {
            AscendC::DataCopy(dst[ndNum * R0 * C], src[ndNum * R0 * stride],
                              AscendC::Nd2NzParams(1,          // ndNum
                                                   remains,    // nValue
                                                   valid_col,  // dValue
                                                   0,          // srcNdMatrixStride
                                                   srcNStride, // srcDValue
                                                   R0,         // dstNzC0Stride
                                                   1,          // dstNzNStride
                                                   0)          // dstNzMatrixStride
            );
        }
    } else {
        // Медленный fallback: stride слишком большой даже для одной строки.
        // Копируем каждую логическую строку отдельно.
        for (int32_t i = 0; i < valid_row; i++) {
            int32_t idxR0 = i / R0;   // индекс fractal-группы по M
            int32_t idxInR0 = i % R0; // позиция строки внутри группы

            AscendC::DataCopy(dst[idxR0 * R0 * C + idxInR0 * C0], src[i * stride],
                              AscendC::Nd2NzParams(1,         // ndNum
                                                   1,         // nValue — одна строка
                                                   valid_col, // dValue
                                                   0,         // srcNdMatrixStride
                                                   0,         // srcDValue
                                                   R0,        // dstNzC0Stride
                                                   0,         // dstNzNStride
                                                   0)         // dstNzMatrixStride
            );
        }
    }
}
