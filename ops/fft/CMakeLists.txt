# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

add_subdirectory(c2c/dft)
add_subdirectory(c2c/dft_sep)
add_subdirectory(c2c/fft_b)
add_subdirectory(c2c/fft_b_sep)
add_subdirectory(c2c/fft_mix)
add_subdirectory(c2c/fft_n)
add_subdirectory(c2c/fft_stride)
add_subdirectory(c2c/arch35)
add_subdirectory(c2c/fft_regbase)
add_subdirectory(c2c/fft_cube_vector)
add_subdirectory(c2c2d/dd)
add_subdirectory(c2r/dft)
add_subdirectory(c2r/fft)
add_subdirectory(r2c/dft)
add_subdirectory(r2c/fft)
add_subdirectory(real/fft_regbase)
add_subdirectory(real/fft_stockham)
add_subdirectory(c2c3d/ddd_sep)


set(operations_objects ${operations_objects} ${ops_objects} PARENT_SCOPE)
message("ops ops_objects: ${ops_objects}")
message("ops fft_objects: ${fft_objects}")
set_source_files_properties(${BINARY_SRC_LIST} PROPERTIES GENERATED TRUE)
message("BINARY_SRC_LIST: ${BINARY_SRC_LIST}")
message("BINARY_TARGET_LIST: ${BINARY_TARGET_LIST}")

add_custom_command(
    OUTPUT ${BINARY_SRC_LIST}
    DEPENDS ${BINARY_TARGET_LIST}
    COMMAND python3 ${MKI_PACKAGE_DIR}/scripts/build_util.py --binary_dir ${CMAKE_BINARY_DIR} --op_type "ascendc"
)

add_custom_target(FFT_BINARY_SRC_TARGET ALL
    DEPENDS ${BINARY_SRC_LIST}
)

add_library(fft_obj OBJECT ${BINARY_SRC_LIST})
add_dependencies(fft_obj FFT_BINARY_SRC_TARGET)
