# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

set(dft_src
    ${CMAKE_CURRENT_LIST_DIR}/dft_operation.cpp
    ${CMAKE_CURRENT_LIST_DIR}/dft/dft_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/dft/tiling/dft_tiling.cpp
)

set(CATLASS_INCLUDES
    ${OPS_THIRD_PARTY_DIR}/catlass/include
    $ENV{ASCEND_HOME_PATH}/include
)

add_operation(DftOperation "${dft_src}")

add_kernel(dft ascend910b cube
    dft/kernel/dft.cce
    DftC64Kernel)

add_kernel(dft_c2c ascend950 cube
    dft/kernel/arch35/dft_c2c.cce
    DftC64Kernel
    CACHE_POLICY compiler-default)

add_kernel(dft_complex32 ascend910b mix
    dft/kernel/dft_complex32.cpp
    DftC32Kernel
    INCLUDE_DIRECTORIES ${CATLASS_INCLUDES})
