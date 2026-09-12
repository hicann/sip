# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
add_operation(FftC2CCubeVectorOperation "${CMAKE_CURRENT_LIST_DIR}/fft_cube_vector_operation.cpp")
if(BUILD_FftC2CCubeVectorOperation)
    target_include_directories(FftC2CCubeVectorOperation PRIVATE ${PROJECT_SOURCE_DIR}/ops/fft/common/include)
endif()

set(kernel_dir fft_cube_vector/kernel/arch35)
file(GLOB retained_headers ${CMAKE_CURRENT_LIST_DIR}/${kernel_dir}/vendor/*.h)
set(stage_headers ${retained_headers} ${PROJECT_SOURCE_DIR}/ops/include/params/fft_c2c_large_tiling.h)
add_kernel(fft_c2c_outer ascend950 vector ${kernel_dir}/fft_c2c_outer.asc FftC2COuterKernel
    CACHE_POLICY compiler-default INCLUDE_DIRECTORIES ${PROJECT_SOURCE_DIR}/ops/include DEPENDS ${stage_headers})
add_kernel(fft_c2c_cube_vector_a ascend950 vector ${kernel_dir}/fft_c2c_cube_vector_a.asc FftC2CCubeVectorAKernel
    CACHE_POLICY compiler-default INCLUDE_DIRECTORIES ${PROJECT_SOURCE_DIR}/ops/include DEPENDS ${stage_headers})
add_kernel(fft_c2c_cube_vector_b ascend950 vector ${kernel_dir}/fft_c2c_cube_vector_b.asc FftC2CCubeVectorBKernel
    CACHE_POLICY compiler-default INCLUDE_DIRECTORIES ${PROJECT_SOURCE_DIR}/ops/include DEPENDS ${stage_headers})
add_kernel(fft_c2c_packed ascend950 mix ${kernel_dir}/fft_c2c_packed.asc FftC2CPackedKernel
    CACHE_POLICY compiler-default INCLUDE_DIRECTORIES ${PROJECT_SOURCE_DIR}/ops/include DEPENDS ${stage_headers})
add_kernel(fft_c2c_gauss ascend950 mix ${kernel_dir}/fft_c2c_gauss.asc FftC2CGaussKernel
    CACHE_POLICY compiler-default INCLUDE_DIRECTORIES ${PROJECT_SOURCE_DIR}/ops/include DEPENDS ${stage_headers})
