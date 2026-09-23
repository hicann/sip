# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
# -----------------------------------------------------------------------------------------------------------
# Standalone single-operator CMake project.  The operator is a fused 2D C2C FFT
# over [4096, 4096] complex64 (both axes).  One __mix__(1,2) kernel performs four
# stages: row FFT (ex99 engine) -> transpose -> row FFT -> transpose, i.e.
#   fft2(X) = ( fft( (fft(X, axis=1)).T, axis=1 ) ).T
# AIC runs the row FFT GEMM (16x16x16), AIV runs deinterleave/twiddle/radix-2 and
# the 64x64 SIMT transpose stages.
#
# Build:  bash build.sh   (sources set_env.sh if ASCEND_HOME_PATH is unset)

cmake_minimum_required(VERSION 3.16)

set(CMAKE_ASC_ARCHITECTURES "dav-3510" CACHE STRING "NPU architecture: dav-3510")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ASC REQUIRED)

project(fft_2d_c2c_4096x4096 LANGUAGES ASC CXX)

# Whole program (host entry + kernel) is compiled as ASC, the same convention
# used by the Catlass examples.  Host code and the __mix__ kernel share the
# fft_*.h headers.
set(FFT_SOURCES
    fft_c2c_2d.cpp
    fft_plan.cpp
    fft_tiling.cpp
    fft_kernel_fft.cpp
)

set_source_files_properties(${FFT_SOURCES} PROPERTIES LANGUAGE ASC)

add_executable(fft_2d_c2c_4096x4096 ${FFT_SOURCES})

set_target_properties(fft_2d_c2c_4096x4096 PROPERTIES LINKER_LANGUAGE ASC)

target_include_directories(fft_2d_c2c_4096x4096 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    $ENV{ASCEND_HOME_PATH}/include
    $ENV{ASCEND_HOME_PATH}/include/aclnn
)

target_compile_options(fft_2d_c2c_4096x4096 PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>
    $<$<COMPILE_LANGUAGE:ASC>:--cce-auto-sync=off>
    $<$<COMPILE_LANGUAGE:ASC>:-DENABLE_CV_COMM_VIA_SSBUF=true>
)

target_compile_definitions(fft_2d_c2c_4096x4096 PRIVATE
    CATLASS_ARCH=3510
    ENABLE_CV_COMM_VIA_SSBUF=true
)

link_directories(fft_2d_c2c_4096x4096 PRIVATE $ENV{ASCEND_HOME_PATH}/lib64)

target_link_libraries(fft_2d_c2c_4096x4096 PRIVATE
    ascendc_runtime runtime profapi mmpa ascend_dump c_sec error_manager ascendcl
    tiling_api nnopbase platform
    dl
    m
)

if(EXISTS $ENV{ASCEND_HOME_PATH}/lib64/libascendalog.so)
    target_link_libraries(fft_2d_c2c_4096x4096 PRIVATE ascendalog)
endif()
if(EXISTS $ENV{ASCEND_HOME_PATH}/lib64/libunified_dlog.so)
    target_link_libraries(fft_2d_c2c_4096x4096 PRIVATE unified_dlog)
endif()

target_link_libraries(fft_2d_c2c_4096x4096 PRIVATE $ENV{ASCEND_HOME_PATH}/x86_64-linux/devlib/libascend_hal.so)