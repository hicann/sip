# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

add_subdirectory($ENV{CODE_ROOT}/3rdparty/googletest-1.14.0 googletest EXCLUDE_FROM_ALL)
add_subdirectory(base)

include_directories(${CMAKE_CURRENT_LIST_DIR} $ENV{CODE_ROOT}/3rdparty/googletest/include
    $ENV{ASCEND_HOME_PATH}/include
    $ENV{ASCEND_HOME_PATH}/include/aclnn
    ${PROJECT_SOURCE_DIR}/ops/include
    ${PROJECT_SOURCE_DIR}/include
)

link_libraries(asdsip_test_util asdsip_host gtest)

aux_source_directory(${CMAKE_CURRENT_LIST_DIR} TEST_SRC)
add_library(test_src OBJECT ${TEST_SRC})

aux_source_directory(${CMAKE_CURRENT_LIST_DIR}/base/conj TEST_SRC_CONJ)
add_library(test_src_conj OBJECT ${TEST_SRC_CONJ})

aux_source_directory(${CMAKE_CURRENT_LIST_DIR}/fft/1d/c2r TEST_SRC_FFT_C2R)
add_library(test_src_fft_c2r OBJECT ${TEST_SRC_FFT_C2R})

aux_source_directory(${CMAKE_CURRENT_LIST_DIR}/fft/1d/c2c TEST_SRC_FFT_C2C)
add_library(test_src_fft_c2c OBJECT ${TEST_SRC_FFT_C2C})

aux_source_directory(${CMAKE_CURRENT_LIST_DIR}/fft/1d/r2c TEST_SRC_FFT_R2C)
add_library(test_src_fft_r2c OBJECT ${TEST_SRC_FFT_R2C})

aux_source_directory(${CMAKE_CURRENT_LIST_DIR}/base/asd_mul TEST_SRC_ASD_MUL)
add_library(test_src_asd_mul OBJECT ${TEST_SRC_ASD_MUL})

# aux_source_directory(${CMAKE_CURRENT_LIST_DIR}/base/swaplast2axes TEST_SRC_SWAPLAST2AXES)
# add_library(test_src_swaplast2axes OBJECT ${TEST_SRC_SWAPLAST2AXES})

# aux_source_directory(${CMAKE_CURRENT_LIST_DIR}/base/function TEST_SRC_FUNCTION)
# add_library(test_src_function OBJECT ${TEST_SRC_FUNCTION})

# aux_source_directory(${CMAKE_CURRENT_LIST_DIR}/base/concat TEST_SRC_CONCAT)
# add_library(test_src_concat OBJECT ${TEST_SRC_CONCAT})

# aux_source_directory(${CMAKE_CURRENT_LIST_DIR}/base/reverse TEST_SRC_REVERSE)
# add_library(test_src_reverse OBJECT ${TEST_SRC_REVERSE})

# aux_source_directory(${CMAKE_CURRENT_LIST_DIR}/base/matmul TEST_SRC_MATMUL)
# add_library(test_src_matmul OBJECT ${TEST_SRC_MATMUL})

# aux_source_directory(${CMAKE_CURRENT_LIST_DIR}/blas/blas_api TEST_SRC_BLAS_API)
# add_library(test_src_blas_api OBJECT ${TEST_SRC_BLAS_API})

aux_source_directory(${CMAKE_CURRENT_LIST_DIR}/blas/strmv TEST_SRC_STRMV)
add_library(test_src_strmv OBJECT ${TEST_SRC_STRMV})

# aux_source_directory(${CMAKE_CURRENT_LIST_DIR}/blas/isamax TEST_SRC_ISAMAX)
# add_library(test_src_isamax OBJECT ${TEST_SRC_ISAMAX})

# aux_source_directory(${CMAKE_CURRENT_LIST_DIR}/blas/icamax TEST_SRC_ICAMAX)
# add_library(test_src_icamax OBJECT ${TEST_SRC_ICAMAX})

aux_source_directory(${CMAKE_CURRENT_LIST_DIR}/blas/ssyr2 TEST_SRC_SSYR2)
add_library(test_src_ssyr2 OBJECT ${TEST_SRC_SSYR2})

# aux_source_directory(${CMAKE_CURRENT_LIST_DIR}/blas/ctrmv TEST_SRC_CTRMV)
# add_library(test_src_ctrmv OBJECT ${TEST_SRC_CTRMV})

aux_source_directory(${CMAKE_CURRENT_LIST_DIR}/blas/strmm TEST_SRC_STRMM)
add_library(test_src_strmm OBJECT ${TEST_SRC_STRMM})

# aux_source_directory(${CMAKE_CURRENT_LIST_DIR}/blas/cgemm TEST_SRC_CGEMM)
# add_library(test_src_cgemm OBJECT ${TEST_SRC_CGEMM})

# aux_source_directory(${CMAKE_CURRENT_LIST_DIR}/blas/sasum TEST_SRC_SASUM)
# add_library(test_src_sasum OBJECT ${TEST_SRC_SASUM})

# aux_source_directory(${CMAKE_CURRENT_LIST_DIR}/blas/scasum TEST_SRC_SCASUM)
# add_library(test_src_scasum OBJECT ${TEST_SRC_SCASUM})

aux_source_directory(${CMAKE_CURRENT_LIST_DIR}/blas/snrm2 TEST_SRC_SNRM2)
add_library(test_src_snrm2 OBJECT ${TEST_SRC_SNRM2})

aux_source_directory(${CMAKE_CURRENT_LIST_DIR}/blas/scnrm2 TEST_SRC_SCNRM2)
add_library(test_src_scnrm2 OBJECT ${TEST_SRC_SCNRM2})

# aux_source_directory(${CMAKE_CURRENT_LIST_DIR}/blas/sscal TEST_SRC_SSCAL)
# add_library(test_src_sscal OBJECT ${TEST_SRC_SSCAL})

# aux_source_directory(${CMAKE_CURRENT_LIST_DIR}/blas/csscal TEST_SRC_CSSCAL)
# add_library(test_src_csscal OBJECT ${TEST_SRC_CSSCAL})

# aux_source_directory(${CMAKE_CURRENT_LIST_DIR}/blas/cscal TEST_SRC_CSCAL)
# add_library(test_src_cscal OBJECT ${TEST_SRC_CSCAL})

aux_source_directory(${CMAKE_CURRENT_LIST_DIR}/blas/scopy TEST_SRC_SCOPY)
add_library(test_src_scopy OBJECT ${TEST_SRC_SCOPY})

aux_source_directory(${CMAKE_CURRENT_LIST_DIR}/blas/ccopy TEST_SRC_CCOPY)
add_library(test_src_ccopy OBJECT ${TEST_SRC_CCOPY})

# aux_source_directory(${CMAKE_CURRENT_LIST_DIR}/blas/caxpy TEST_SRC_CAXPY)
# add_library(test_src_caxpy OBJECT ${TEST_SRC_CAXPY})

aux_source_directory(${CMAKE_CURRENT_LIST_DIR}/blas/sdot TEST_SRC_SDOT)
add_library(test_src_sdot OBJECT ${TEST_SRC_SDOT})

aux_source_directory(${CMAKE_CURRENT_LIST_DIR}/blas/cdotu TEST_SRC_CDOTU)
add_library(test_src_cdotu OBJECT ${TEST_SRC_CDOTU})

aux_source_directory(${CMAKE_CURRENT_LIST_DIR}/blas/cdotc TEST_SRC_CDOTC)
add_library(test_src_cdotc OBJECT ${TEST_SRC_CDOTC})

# aux_source_directory(${CMAKE_CURRENT_LIST_DIR}/blas/sswap TEST_SRC_SSWAP)
# add_library(test_src_sswap OBJECT ${TEST_SRC_SSWAP})

# aux_source_directory(${CMAKE_CURRENT_LIST_DIR}/blas/cswap TEST_SRC_CSWAP)
# add_library(test_src_cswap OBJECT ${TEST_SRC_CSWAP})

aux_source_directory(${CMAKE_CURRENT_LIST_DIR}/blas/ssyr TEST_SRC_SSYR)
add_library(test_src_ssyr OBJECT ${TEST_SRC_SSYR})

# aux_source_directory(${CMAKE_CURRENT_LIST_DIR}/blas/cgemv TEST_SRC_CGEMV)
# add_library(test_src_cgemv OBJECT ${TEST_SRC_CGEMV})

# aux_source_directory(${CMAKE_CURRENT_LIST_DIR}/blas/interpolation TEST_SRC_INTERPOLATION)
# add_library(test_src_interpolation OBJECT ${TEST_SRC_INTERPOLATION})

# aux_source_directory(${CMAKE_CURRENT_LIST_DIR}/blas/cgerc TEST_SRC_CGERC)
# add_library(test_src_cgerc OBJECT ${TEST_SRC_CGERC})

# aux_source_directory(${CMAKE_CURRENT_LIST_DIR}/blas/csrot TEST_SRC_CSROT)
# add_library(test_src_csrot OBJECT ${TEST_SRC_CSROT})

# aux_source_directory(${CMAKE_CURRENT_LIST_DIR}/blas/colwise_mul TEST_SRC_COLWISE_MUL)
# add_library(test_src_colwise_mul OBJECT ${TEST_SRC_COLWISE_MUL})

# aux_source_directory(${CMAKE_CURRENT_LIST_DIR}/blas/complex_mat_dot TEST_SRC_COMPLEX_MAT_DOT)
# add_library(test_src_complex_mat_dot OBJECT ${TEST_SRC_COMPLEX_MAT_DOT})

# aux_source_directory(${CMAKE_CURRENT_LIST_DIR}/blas/cgemv_batched TEST_SRC_CGEMV_BATCHED)
# add_library(test_src_cgemv_batched OBJECT ${TEST_SRC_CGEMV_BATCHED})

# aux_source_directory(${CMAKE_CURRENT_LIST_DIR}/fft/1d TEST_SRC_1D)
# add_library(test_src_1d OBJECT ${TEST_SRC_1D})

# aux_source_directory(${CMAKE_CURRENT_LIST_DIR}/fft/2d TEST_SRC_2D)
# add_library(test_src_2d OBJECT ${TEST_SRC_2D})

# aux_source_directory(${CMAKE_CURRENT_LIST_DIR}/blas/cgemm_batched TEST_SRC_CGEMM_BATCHED)
# add_library(test_src_cgemm_batched OBJECT ${TEST_SRC_CGEMM_BATCHED})

# aux_source_directory(${CMAKE_CURRENT_LIST_DIR}/blas/hcgemm_batched TEST_SRC_HCGEMM_BATCHED)
# add_library(test_src_hcgemm_batched OBJECT ${TEST_SRC_HCGEMM_BATCHED})

aux_source_directory(${CMAKE_CURRENT_LIST_DIR}/blas/cmatinv_batched TEST_SRC_CMATINV_BATCHED)
add_library(test_src_cmatinv_batched OBJECT ${TEST_SRC_CMATINV_BATCHED})

# aux_source_directory(${CMAKE_CURRENT_LIST_DIR}/filter/convolve TEST_SRC_CONVOLVE)
# add_library(test_src_convolve OBJECT ${TEST_SRC_CONVOLVE})

add_executable(ops_unittest
# ${CMAKE_CURRENT_LIST_DIR}/custom_main.cpp
    $<TARGET_OBJECTS:test_src>
    $<TARGET_OBJECTS:test_src_asd_mul>
    # $<TARGET_OBJECTS:test_src_swaplast2axes>
    # $<TARGET_OBJECTS:test_src_function>
    $<TARGET_OBJECTS:test_src_conj>
    $<TARGET_OBJECTS:test_src_fft_c2r>
    $<TARGET_OBJECTS:test_src_fft_c2c>
    $<TARGET_OBJECTS:test_src_fft_r2c>
    # $<TARGET_OBJECTS:test_src_concat>
    # $<TARGET_OBJECTS:test_src_matmul>
    # $<TARGET_OBJECTS:test_src_reverse>
    # $<TARGET_OBJECTS:test_src_blas_api>
    # $<TARGET_OBJECTS:test_src_isamax>
    # $<TARGET_OBJECTS:test_src_icamax>
    $<TARGET_OBJECTS:test_src_strmv>
    $<TARGET_OBJECTS:test_src_ssyr2>
    # $<TARGET_OBJECTS:test_src_ctrmv>
    $<TARGET_OBJECTS:test_src_strmm>
    # $<TARGET_OBJECTS:test_src_cgemm>
    # $<TARGET_OBJECTS:test_src_sasum>
    # $<TARGET_OBJECTS:test_src_scasum>
    $<TARGET_OBJECTS:test_src_snrm2>
    $<TARGET_OBJECTS:test_src_scnrm2>
    # $<TARGET_OBJECTS:test_src_sscal>
    # $<TARGET_OBJECTS:test_src_csscal>
    # $<TARGET_OBJECTS:test_src_cscal>
    $<TARGET_OBJECTS:test_src_scopy>
    $<TARGET_OBJECTS:test_src_ccopy>
    # $<TARGET_OBJECTS:test_src_caxpy>
    $<TARGET_OBJECTS:test_src_sdot>
    $<TARGET_OBJECTS:test_src_cdotu>
    $<TARGET_OBJECTS:test_src_cdotc>
    # $<TARGET_OBJECTS:test_src_sswap>
    # $<TARGET_OBJECTS:test_src_cswap>
    $<TARGET_OBJECTS:test_src_ssyr>
    $<TARGET_OBJECTS:test_src_cmatinv_batched>
    # $<TARGET_OBJECTS:test_src_cgemv>
	# $<TARGET_OBJECTS:test_src_cgerc>
    # $<TARGET_OBJECTS:test_src_csrot>
    # $<TARGET_OBJECTS:test_src_colwise_mul>
    # $<TARGET_OBJECTS:test_src_complex_mat_dot>
    # $<TARGET_OBJECTS:test_src_cgemv_batched>
    # $<TARGET_OBJECTS:test_src_1d>
    # $<TARGET_OBJECTS:test_src_2d>
    # $<TARGET_OBJECTS:test_src_interpolation>
    # $<TARGET_OBJECTS:test_src_cgemm_batched>
    # $<TARGET_OBJECTS:test_src_hcgemm_batched>
    # $<TARGET_OBJECTS:test_src_convolve>
)


target_include_directories(ops_unittest PRIVATE ${CMAKE_CURRENT_LIST_DIR}
                                                $ENV{CODE_ROOT}/3rdparty/googletest/include)

target_link_directories(ops_unittest PRIVATE $ENV{CODE_ROOT}/3rdparty/googletest/lib)
target_link_libraries(ops_unittest PRIVATE asdsip_test_util asdsip_host gtest
    $ENV{ASCEND_HOME_PATH}/lib64/libascendcl.so
    $ENV{ASCEND_HOME_PATH}/lib64/libnnopbase.so
    $ENV{ASCEND_HOME_PATH}/lib64/libopapi.so
)

target_compile_options(ops_unittest PRIVATE
    -Wno-sign-compare
    -Wno-narrowing
)

install(TARGETS ops_unittest DESTINATION bin)
