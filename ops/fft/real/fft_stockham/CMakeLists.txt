# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
set(real_stockham_dir ${CMAKE_CURRENT_LIST_DIR})
add_operation(FftRealStockhamOperation "${real_stockham_dir}/fft_stockham_operation.cpp")
foreach(kind IN ITEMS c2r r2c)
    if(kind STREQUAL "c2r")
        set(stockham_kernel FftRealStockhamC2RKernel)
    else()
        set(stockham_kernel FftRealStockhamR2CKernel)
    endif()
    add_kernel(fft_real_stockham_${kind} ascend950 vector
        fft_stockham/kernel/arch35/fft_real_stockham_${kind}.cpp ${stockham_kernel}
        CACHE_POLICY compiler-default INCLUDE_DIRECTORIES ${PROJECT_SOURCE_DIR}/ops/include
        DEPENDS ${real_stockham_dir}/fft_stockham/kernel/arch35/vendor/fft_stockham_mix.h
                ${PROJECT_SOURCE_DIR}/ops/include/params/fft_real_stockham_tiling.h)
endforeach()
