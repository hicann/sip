# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
set(real_regbase_dir ${CMAKE_CURRENT_LIST_DIR})
set(real_vendor ${real_regbase_dir}/fft_regbase/kernel/arch35/vendor)
set(real_large ${PROJECT_SOURCE_DIR}/ops/fft/c2c/fft_regbase/fft_regbase/large)
add_operation(FftRealRegBaseOperation "${real_regbase_dir}/fft_regbase_operation.cpp")
if(BUILD_FftRealRegBaseOperation)
    target_include_directories(FftRealRegBaseOperation PRIVATE ${real_regbase_dir}/include ${real_large})
endif()
set(real_dependencies ${real_regbase_dir}/include/fft_real_plan.h
    ${PROJECT_SOURCE_DIR}/ops/include/params/fft_real_regbase_tiling.h
    ${real_vendor}/regbase_small.h ${real_large}/fft_c2c_plan.h
    ${real_vendor}/regbase_small_impl.h ${real_vendor}/regbase_real_fused_impl.h
    ${real_vendor}/regbase_real_boundary_impl.h)
foreach(stage IN ITEMS fused boundary small)
    if(stage STREQUAL "fused")
        set(real_kernel FftRealRegBaseFusedKernel)
    elseif(stage STREQUAL "boundary")
        set(real_kernel FftRealRegBaseBoundaryKernel)
    else()
        set(real_kernel FftRealRegBaseSmallKernel)
    endif()
    add_kernel(fft_real_regbase_${stage} ascend950 vector
        fft_regbase/kernel/arch35/fft_real_regbase_${stage}.asc ${real_kernel}
        CACHE_POLICY compiler-default INCLUDE_DIRECTORIES ${real_large} ${PROJECT_SOURCE_DIR}/ops/include
        DEPENDS ${real_dependencies})
endforeach()
