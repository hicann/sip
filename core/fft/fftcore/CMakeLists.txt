# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/fftoperation)

add_library(fft_core STATIC
    fft_core_common_func.cpp
    fft_core_common_func_utils.cpp
    fft_core_mix_base.cpp

    dft_core.cpp
    dft_core_sep.cpp

    fft_core_b.cpp
    fft_core_b_sep.cpp
    fft_core_n.cpp
    fft_core_stride.cpp

    fft_core_mix.cpp

    dft_c2r_core.cpp
    fft_c2r_core.cpp
    fft_c2r_arch35.cpp

    dft_r2c_core.cpp
    fft_r2c_core.cpp
    fft_r2c_arch35.cpp
    fft_c2c_arch35.cpp
    fft_c2c_regbase.cpp
    fft_c2c_large.cpp
    fft_real_regbase.cpp
    fft_real_stockham.cpp
    fft_c2c2d_arch35.cpp

    fft_core_any.cpp

    dd_core.cpp

    ddd_core_sep.cpp

    select_core.cpp

    fft_core_istft_any.cpp
)

target_include_directories(fft_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(fft_core PRIVATE ${PROJECT_SOURCE_DIR}/ops/fft/c2c/fft_regbase/fft_regbase/large)

target_link_libraries(fft_core PUBLIC fft_utils)
target_link_libraries(fft_core PUBLIC fft_operation)

target_include_directories(fft_core PRIVATE ${PROJECT_SOURCE_DIR}/ops/fft/real/fft_regbase/include)
