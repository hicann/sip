# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

set(dft_c2r_src
    ${CMAKE_CURRENT_LIST_DIR}/dft_c2r_operation.cpp
    ${CMAKE_CURRENT_LIST_DIR}/dft/dft_c2r_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/dft/tiling/dft_c2r_tiling.cpp
)

add_operation(DftC2ROperation "${dft_c2r_src}")

add_kernel(dft_c2r ascend910b cube
    dft/kernel/dft_c2r.cce
    DftC2RC64Kernel)

add_kernel(dft_c2r ascend950 cube
    dft/kernel/arch35/dft_c2r.cce
    DftC2RC64Kernel
    CACHE_POLICY compiler-default)
