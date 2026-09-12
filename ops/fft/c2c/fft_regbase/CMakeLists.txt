# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
set(fft_regbase_src
    ${CMAKE_CURRENT_LIST_DIR}/fft_regbase_operation.cpp
    ${CMAKE_CURRENT_LIST_DIR}/fft_regbase/fft_regbase_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/fft_regbase/tiling/fft_regbase_tiling.cpp
)
add_operation(FftC2CRegBaseOperation "${fft_regbase_src}")
if(BUILD_FftC2CRegBaseOperation)
    target_include_directories(FftC2CRegBaseOperation PRIVATE ${PROJECT_SOURCE_DIR}/ops/fft/common/include)
endif()
add_kernel(fft_c2c_regbase ascend950 vector
    fft_regbase/kernel/arch35/fft_c2c_regbase.asc
    FftC2CRegBaseC64Kernel
    CACHE_POLICY compiler-default
    INCLUDE_DIRECTORIES $ENV{ASCEND_HOME_PATH}/asc/include
        $ENV{ASCEND_HOME_PATH}/asc/include/basic_api
        $ENV{ASCEND_HOME_PATH}/asc/include/adv_api
        $ENV{ASCEND_HOME_PATH}/asc/impl/basic_api
        $ENV{ASCEND_HOME_PATH}/asc/impl/utils
    DEPENDS ${CMAKE_CURRENT_LIST_DIR}/fft_regbase/kernel/arch35/fft_regbase_impl.h
        ${CMAKE_CURRENT_LIST_DIR}/fft_regbase/tiling/fft_regbase_tiling_data.h
)

set(large_dir ${CMAKE_CURRENT_LIST_DIR}/fft_regbase/large)
set(large_depends ${large_dir}/regbase_large_impl.h ${large_dir}/regbase_large.h
    ${large_dir}/fft_c2c_plan.h)
add_kernel(fft_c2c_regbase_large_a ascend950 vector
    fft_regbase/large/fft_c2c_regbase_large_a.asc FftC2CRegBaseLargeAKernel
    CACHE_POLICY compiler-default DEPENDS ${large_depends})
add_kernel(fft_c2c_regbase_large_b ascend950 vector
    fft_regbase/large/fft_c2c_regbase_large_b.asc FftC2CRegBaseLargeBKernel
    CACHE_POLICY compiler-default DEPENDS ${large_depends})
