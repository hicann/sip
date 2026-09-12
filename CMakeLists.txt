# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.8)
project("asdsipboost")

option(USE_CXX11_ABI "USE_CXX11_ABI" OFF)
option(USE_MSDEBUG "USE_MSDEBUG" OFF)
option(USE_MSSANITIZER "USE_MSSANITIZER" OFF)

if(NOT "$ENV{NEED_COMPILE_RT}" STREQUAL  "TRUE")
    add_compile_options(-Wfloat-equal -fno-common)
endif()

message(STATUS "USE_CXX11_ABI:${USE_CXX11_ABI}")
message(STATUS "USE_MSDEBUG:${USE_MSDEBUG}")
message(STATUS "USE_MSSANITIZER:${USE_MSSANITIZER}")

# compile flag
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${PROJECT_SOURCE_DIR}/output" CACHE PATH "..." FORCE)
endif()
message(STATUS "CMAKE_INSTALL_PREFIX:${CMAKE_INSTALL_PREFIX}")

set(NAMESPACE "AsdSip")
set(THIRD_PARTY_DIR "${CMAKE_CURRENT_LIST_DIR}/3rdparty")
set(MKI_PACKAGE_DIR "${THIRD_PARTY_DIR}/mki")
set(OP_LIST_YAML_DIR "${CMAKE_CURRENT_LIST_DIR}/configs")
set(MKI_SCRIPT_DIR "${MKI_PACKAGE_DIR}/scripts")

include(${PROJECT_SOURCE_DIR}/cmake/host_config.cmake)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pipe -fstack-protector-all")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-copy")
message(STATUS "CMAKE_CXX_FLAGS:${CMAKE_CXX_FLAGS}")

if("$ENV{TEST_TYPE}" STREQUAL "UT")
    add_definitions(-DUNITTEST)
endif()

ADD_LINK_OPTIONS(-s)
add_compile_options(
    -D_FORTIFY_SOURCE=2 -O2
    )

include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/core/utils/include
    $ENV{ASCEND_HOME_PATH}/include
    $ENV{ASCEND_HOME_PATH}/include/aclnn
)

link_directories(
    $ENV{ASCEND_HOME_PATH}/lib64
)


if("$ENV{NEED_COMPILE_RT}" STREQUAL "TRUE")
    if("$ENV{TEST_TYPE}" STREQUAL "UT")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-arcs -ftest-coverage -Wno-error")
        set(CMAKE_CXX_OUTPUT_EXTENSION_REPLACE 1)
    elseif("$ENV{TEST_TYPE}" STREQUAL "FT")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-arcs -ftest-coverage -O0 -fsanitize-coverage=trace-pc -Wno-error")
        set(CMAKE_CXX_OUTPUT_EXTENSION_REPLACE 1)
        if("$ENV{USE_ASAN}" STREQUAL "ON")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fsanitize=undefined -fPIC")
        endif()
    elseif("$ENV{TEST_TYPE}" STREQUAL "ST")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-arcs -ftest-coverage -Wno-error")
        set(CMAKE_CXX_OUTPUT_EXTENSION_REPLACE 1)
    endif()
    add_subdirectory(tests)
# else()
#     add_subdirectory(example)
endif()

add_subdirectory(imported_libs)
add_subdirectory(ops)
add_subdirectory(core)

install(FILES ${MKI_PACKAGE_DIR}/lib/libmki_static.a DESTINATION lib)
install(FILES ${MKI_PACKAGE_DIR}/lib/libmki.so DESTINATION lib)
