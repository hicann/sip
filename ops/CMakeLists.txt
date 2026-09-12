# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

message(STATUS "MKI_PACKAGE_DIR2 : ${MKI_PACKAGE_DIR}")
set(OPS_THIRD_PARTY_DIR "${CMAKE_CURRENT_LIST_DIR}/../3rdparty")

include(${PROJECT_SOURCE_DIR}/cmake/kernel_config.cmake)

include_directories(
    ${MKI_PACKAGE_DIR}/include
    ${CMAKE_CURRENT_LIST_DIR}/include
    ${CMAKE_CURRENT_LIST_DIR}/include/params
    ${CMAKE_CURRENT_LIST_DIR}/blas/common_tiling/include
    ${CMAKE_CURRENT_LIST_DIR}/utils/include
    $ENV{ASDOPS_CODE_PATH}/3rdparty/securec/include
    $ENV{ASDOPS_CODE_PATH}/3rdparty/metadef/inc/external
    $ENV{ASDOPS_CODE_PATH}/core
)

if(NOT EXISTS ${OP_LIST_YAML_DIR}/op_list.yaml)
    execute_process(COMMAND python3 ${MKI_PACKAGE_DIR}/scripts/op_list_utils.py -s ${CMAKE_CURRENT_LIST_DIR} -d ${OP_LIST_YAML_DIR}
                    ERROR_VARIABLE RESULT_INFO
                    RESULT_VARIABLE RESULT)
    if(NOT RESULT EQUAL 0)
        message(FATAL_ERROR "create op_build.yaml failed, error code: ${RESULT}, error info:\n${RESULT_INFO}")
    endif()
endif()

set(OP_LIST_CMD_PARAMS "'${OP_LIST_YAML_DIR}/op_list.yaml', '${MKI_PACKAGE_DIR}/cmake/op_build.cmake'")
execute_process(COMMAND python3 -c "import op_list_utils\nop_list_utils.build_cmake_options(${OP_LIST_CMD_PARAMS})"
                WORKING_DIRECTORY ${MKI_PACKAGE_DIR}/scripts
                ERROR_VARIABLE RESULT_INFO
                RESULT_VARIABLE RESULT)
if(NOT RESULT EQUAL 0)
    message(FATAL_ERROR "create op_build.cmake failed, error code: ${RESULT}, error info:\n${RESULT_INFO}")
endif()

include(${MKI_PACKAGE_DIR}/cmake/op_build.cmake)
sip_configure_kernel_targets()

add_subdirectory(base)

add_subdirectory(fft)
add_subdirectory(blas)
add_subdirectory(filter)
add_subdirectory(utils)

add_library(asdsip_core SHARED ops.cpp)
target_link_libraries(asdsip_core PUBLIC "-Wl,--whole-archive" blas_obj fft_obj base_obj filter_obj "-Wl,--no-whole-archive")

message("ops ops_objects: ${ops_objects}")
if("$ENV{TEST_TYPE}" STREQUAL "UT" OR "$ENV{TEST_TYPE}" STREQUAL "FT")
    target_link_libraries(asdsip_core PRIVATE utils mki tiling_api register graph ${operations_objects})
else()
    target_link_libraries(asdsip_core PRIVATE "-Wl,--whole-archive" mki_static "-Wl,--no-whole-archive" utils tiling_api register graph ${operations_objects}
        -Wl,--exclude-libs=ALL
    )
endif()
message("ops operations_objects: ${operations_objects}")

install(TARGETS asdsip_core DESTINATION lib)

add_library(asdsip_host SHARED)
target_link_libraries(asdsip_host PRIVATE ops_utils)
target_include_directories(asdsip_host PUBLIC ${CMAKE_CURRENT_LIST_DIR}/include)
target_include_directories(asdsip_host PUBLIC ${CMAKE_CURRENT_LIST_DIR}/include/params)

install(TARGETS asdsip_host DESTINATION lib)
