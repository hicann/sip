# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

if("$ENV{TEST_TYPE}" STREQUAL "UT")
    include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/fft/fftplan/include
        ${PROJECT_SOURCE_DIR}/ops/include
        $ENV{ATB_HOME_PATH}/include
    )
else()
    include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/fft/fftplan/include
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/ops/include
    )
endif()


add_subdirectory(utils)
add_subdirectory(base)
add_subdirectory(blas)
add_subdirectory(fft)
add_subdirectory(filter)
add_subdirectory(interpolation)

if(NOT "$ENV{TEST_TYPE}" STREQUAL "ST")
    add_library(asdsip SHARED $<TARGET_OBJECTS:utils> $<TARGET_OBJECTS:blas> $<TARGET_OBJECTS:fft> $<TARGET_OBJECTS:base> $<TARGET_OBJECTS:filter> $<TARGET_OBJECTS:interpolation>)
    if("$ENV{TEST_TYPE}" STREQUAL "UT")
        target_link_libraries(asdsip PUBLIC utils base blas fft filter interpolation mki asdsip_core)
    else()
        target_link_libraries(asdsip PUBLIC utils base blas fft filter interpolation asdsip_core)
    endif()
    if(TARGET ops_fft)
        # 项目全局 CMAKE_SKIP_RPATH=TRUE (host_config.cmake), 运行时靠 set_env.sh 的
        # LD_LIBRARY_PATH 定位 libcann_ops_fft.so (与 libasdsip.so 同目录);
        # weak 引用不会触发 --as-needed 记录 DT_NEEDED, 须显式 --no-as-needed 强制依赖
        target_link_libraries(asdsip PRIVATE "-Wl,--no-as-needed,${OPS_FFT_LIB},--as-needed")
    endif()
    add_library(asdsip_static STATIC $<TARGET_OBJECTS:utils> $<TARGET_OBJECTS:blas> $<TARGET_OBJECTS:fft> $<TARGET_OBJECTS:base> $<TARGET_OBJECTS:filter> $<TARGET_OBJECTS:interpolation>)
    target_link_libraries(asdsip_static PUBLIC utils base blas fft filter interpolation)
    target_include_directories(asdsip_static PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_include_directories(asdsip_static PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/utils/include)

    install(TARGETS asdsip DESTINATION lib)
    install(TARGETS asdsip_static DESTINATION lib)
endif()

install(FILES ${PROJECT_SOURCE_DIR}/include/blas_api.h ${PROJECT_SOURCE_DIR}/include/fft_api.h ${PROJECT_SOURCE_DIR}/include/base_api.h ${PROJECT_SOURCE_DIR}/include/filter_api.h ${PROJECT_SOURCE_DIR}/include/interp_api.h ${PROJECT_SOURCE_DIR}/include/asdsip.h DESTINATION include)
install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/domain DESTINATION include)
