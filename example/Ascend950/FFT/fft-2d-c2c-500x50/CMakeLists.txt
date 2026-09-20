# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# Standalone single-operator CMake project (reference: asc-devkit-cann single-op
# samples).  The operator is a fused 2D C2C FFT over [12000, 500, 50] complex64
# (axes 1,2): AIC runs the 50-point row FFT as an embedded-rotation GEMM, AIV runs
# the 500-point column FFT (2-stage radix-20/25 register fusion) as SIMT.
#
# Build:  bash build.sh   (requires source set_env.sh first / inside script)

cmake_minimum_required(VERSION 3.16)

set(CMAKE_ASC_ARCHITECTURES "dav-3510" CACHE STRING "NPU architecture: dav-3510")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ASC REQUIRED)

project(fft_2d_c2c_500x50 LANGUAGES ASC CXX)

# ---------------------------------------------------------------------------
# Sources: the whole program (host entry + kernel) is compiled as ASC, the same
# convention used by the Catlass examples.  Host code (aclInit / memcpy / file
# IO) and the __mix__ kernel share the fft_*.h headers, so a single translation
# unit set keeps header references trivially consistent.
# ---------------------------------------------------------------------------
set(FFT_SOURCES
    fft_c2c_2d.cpp
    fft_plan.cpp
    fft_coeffs.cpp
    fft_tiling.cpp
    fft_kernel.cpp
    fft_shift_kernel.cpp
)

set_source_files_properties(${FFT_SOURCES} PROPERTIES LANGUAGE ASC)

add_executable(fft_2d_c2c_500x50 ${FFT_SOURCES})

set_target_properties(fft_2d_c2c_500x50 PROPERTIES LINKER_LANGUAGE ASC)

# ---------------------------------------------------------------------------
# Include paths: local headers (fft_*.h), vendored Catlass/TLA templates, CANN.
# ---------------------------------------------------------------------------
target_include_directories(fft_2d_c2c_500x50 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    $ENV{ASCEND_HOME_PATH}/include
    $ENV{ASCEND_HOME_PATH}/include/aclnn
)

# ---------------------------------------------------------------------------
# Compile options / definitions (mirror the verified Catlass example flags).
# ---------------------------------------------------------------------------
target_compile_options(fft_2d_c2c_500x50 PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>
    $<$<COMPILE_LANGUAGE:ASC>:--cce-auto-sync=off>
    $<$<COMPILE_LANGUAGE:ASC>:-DENABLE_CV_COMM_VIA_SSBUF=true>
)

target_compile_definitions(fft_2d_c2c_500x50 PRIVATE
    CATLASS_ARCH=3510
    ENABLE_CV_COMM_VIA_SSBUF=true
)

# ---------------------------------------------------------------------------
# Link libraries (AscendC runtime + CANN host libs).
# ---------------------------------------------------------------------------
link_directories(fft_2d_c2c_500x50 PRIVATE $ENV{ASCEND_HOME_PATH}/lib64)

target_link_libraries(fft_2d_c2c_500x50 PRIVATE
    ascendc_runtime runtime profapi mmpa ascend_dump c_sec error_manager ascendcl
    tiling_api nnopbase platform
    dl
    m
)

if(EXISTS $ENV{ASCEND_HOME_PATH}/lib64/libascendalog.so)
    target_link_libraries(fft_2d_c2c_500x50 PRIVATE ascendalog)
endif()
if(EXISTS $ENV{ASCEND_HOME_PATH}/lib64/libunified_dlog.so)
    target_link_libraries(fft_2d_c2c_500x50 PRIVATE unified_dlog)
endif()

target_link_libraries(fft_2d_c2c_500x50 PRIVATE $ENV{ASCEND_HOME_PATH}/x86_64-linux/devlib/libascend_hal.so)
