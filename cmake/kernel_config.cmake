# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# Filter generated kernel switches before any device build targets are created.
# Host operation factories remain available; MKI attaches only present binaries.
function(sip_configure_kernel_targets)
    set(cann_version "")
    set(version_header "$ENV{ASCEND_HOME_PATH}/include/version/cann_version.h")
    if(EXISTS "${version_header}")
        file(READ "${version_header}" version_text)
        foreach(part MAJOR MINOR PATCH)
            string(REGEX MATCH "#[ \t]*define[ \t]+CANN_${part}[ \t]+([0-9]+)" match "${version_text}")
            set(cann_${part} "${CMAKE_MATCH_1}")
        endforeach()
        if(NOT "${cann_MAJOR}" STREQUAL "" AND NOT "${cann_MINOR}" STREQUAL "" AND
           NOT "${cann_PATCH}" STREQUAL "")
            set(cann_version "${cann_MAJOR}.${cann_MINOR}.${cann_PATCH}")
        endif()
    endif()
    if("${cann_version}" STREQUAL "" AND EXISTS "$ENV{ASCEND_HOME_PATH}/compiler/version.info")
        file(STRINGS "$ENV{ASCEND_HOME_PATH}/compiler/version.info" version_lines REGEX "^[Vv]ersion=")
        foreach(line ${version_lines})
            if(line MATCHES "^[Vv]ersion=([0-9]+\\.[0-9]+\\.[0-9]+)")
                set(cann_version "${CMAKE_MATCH_1}")
                break()
            endif()
        endforeach()
    endif()

    set(requested_targets "")
    set(effective_targets "")
    get_cmake_property(variables VARIABLES)
    foreach(variable ${variables})
        if(variable MATCHES "^BUILD_.+_ascend[0-9a-z]+$" AND ${variable})
            string(REGEX REPLACE "^.*_(ascend[0-9a-z]+)$" "\\1" soc "${variable}")
            list(APPEND requested_targets "${soc}")
            if(soc STREQUAL "ascend950")
                if("${cann_version}" STREQUAL "")
                    message(FATAL_ERROR
                        "Cannot determine CANN version for requested ascend950 kernels. "
                        "Source the CANN environment with include/version/cann_version.h or compiler/version.info.")
                endif()
                if(cann_version VERSION_LESS "9.1.0")
                    set(${variable} OFF PARENT_SCOPE)
                    continue()
                endif()
            endif()
            list(APPEND effective_targets "${soc}")
        endif()
    endforeach()
    list(REMOVE_DUPLICATES requested_targets)
    list(REMOVE_DUPLICATES effective_targets)
    list(SORT requested_targets)
    list(SORT effective_targets)
    if("${cann_version}" STREQUAL "")
        set(cann_version "unknown (A5 not requested)")
    endif()
    message(STATUS "SiP CANN version: ${cann_version}; requested device targets: ${requested_targets}")
    message(STATUS "SiP effective device targets: ${effective_targets}")
    if("ascend950" IN_LIST requested_targets AND NOT "ascend950" IN_LIST effective_targets)
        message(STATUS "Omitting all A5 device kernels: this SiP build requires CANN >= 9.1.0 for A5 support")
    endif()
    if(NOT effective_targets)
        message(FATAL_ERROR "No device kernel target remains enabled. Enable a compatible target or use CANN >= 9.1.0.")
    endif()
endfunction()

macro(add_operation op srcs)
    if (BUILD_${op})
        add_compile_definitions(OperationPlaceHolder="${op}")
        add_library(${op} OBJECT ${srcs})
        set(op_name ${op})
        set(ops_objects ${ops_objects} ${op} PARENT_SCOPE)
    endif()
endmacro()

macro(add_kernel kernel soc channel srcs tac)
    if (BUILD_${op_name}_${tac}_${soc})
        string(TOLOWER ${soc} soc_lower)
        string(LENGTH ${soc} soc_length)
        string(SUBSTRING "${CHIP_TYPE}" 0 ${soc_length} chip_type_prefix)
        if ((NOT USE_MSDEBUG) OR (USE_MSDEBUG AND ("${soc_lower}" STREQUAL "${chip_type_prefix}")))
            # build target: op_kernels/soc/op/kernel/kernel.o
            set(${kernel}_${soc}_output
                ${CMAKE_BINARY_DIR}/op_kernels/${soc}/${op_name}/${tac}/${kernel}.o)
            set(multiValueArgs INCLUDE_DIRECTORIES DEPENDS)
            set(oneValueArgs CACHE_POLICY)
            cmake_parse_arguments(arg_add_kernel "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
            set(PYTHON_ARGS
                "--soc" "${soc}"
                "--channel" "${channel}"
                "--srcs" "${CMAKE_CURRENT_LIST_DIR}/${srcs}"
                "--dst" "${${kernel}_${soc}_output}"
                "--code_root" "${OPS_THIRD_PARTY_DIR}/.."
                "--kernel" "${kernel}"
                "--use_msdebug" "${USE_MSDEBUG}"
                "--use_mssanitizer" "${USE_MSSANITIZER}"
                "--no_warning"
            )
            if(NOT "${arg_add_kernel_INCLUDE_DIRECTORIES}" STREQUAL "")
                set(PYTHON_ARGS ${PYTHON_ARGS} "--include_directories" "${arg_add_kernel_INCLUDE_DIRECTORIES}")
            endif()
            if(NOT "${arg_add_kernel_CACHE_POLICY}" STREQUAL "")
                if(NOT "${arg_add_kernel_CACHE_POLICY}" STREQUAL "compiler-default")
                    message(FATAL_ERROR
                        "${kernel}: only CACHE_POLICY compiler-default is supported; "
                        "A5 cache-policy exceptions require separate path-specific qualification")
                endif()
                set(PYTHON_ARGS ${PYTHON_ARGS} "--cache-policy" "${arg_add_kernel_CACHE_POLICY}")
            endif()
            add_custom_command(
                OUTPUT ${${kernel}_${soc}_output}
                DEPENDS ${srcs} ${arg_add_kernel_DEPENDS} ${PROJECT_SOURCE_DIR}/scripts/compile_ascendc.py
                WORKING_DIRECTORY ${OPS_PROJECT_ROOT_DIR}
                COMMAND python3 ${PROJECT_SOURCE_DIR}/scripts/compile_ascendc.py ${PYTHON_ARGS}
            )
            # build target: obj/soc/op/kernel.cpp
            set(${kernel}_${soc}_cpp_output
                ${CMAKE_BINARY_DIR}/obj/${soc}/${op_name}/${kernel}.cpp)
            add_custom_command(
                OUTPUT ${${kernel}_${soc}_cpp_output}
                DEPENDS ${${kernel}_${soc}_output}
                WORKING_DIRECTORY ${MKI_SCRIPT_DIR}
                COMMAND python3 -c "import build_util; build_util.compile_ascendc_code('${${kernel}_${soc}_output}', '${${kernel}_${soc}_cpp_output}')"
                VERBATIM
            )
            add_custom_target(ascendc_cpp_${kernel}_${soc} ALL
                DEPENDS ${${kernel}_${soc}_cpp_output}
            )
            # collect targets
            set(LOCAL_BINARY_SRC_LIST ${LOCAL_BINARY_SRC_LIST} ${${kernel}_${soc}_cpp_output})
            set(BINARY_SRC_LIST ${BINARY_SRC_LIST} ${LOCAL_BINARY_SRC_LIST} PARENT_SCOPE)
            set(LOCAL_BINARY_TARGET_LIST ${LOCAL_BINARY_TARGET_LIST} ascendc_cpp_${kernel}_${soc})
            set(BINARY_TARGET_LIST ${BINARY_TARGET_LIST} ${LOCAL_BINARY_TARGET_LIST} PARENT_SCOPE)
        endif()
    endif()
endmacro()

macro(add_kernel_bin tac soc)
    set(${tac}_${soc}_dir
        ${CMAKE_BINARY_DIR}/op_kernels/${soc}/${op_name}/${tac})
    file(MAKE_DIRECTORY ${${tac}_${soc}_dir})
    set(${tac}_${soc}_output ${${tac}_${soc}_dir}/${tac}.o)
    add_custom_command(
        OUTPUT ${${tac}_${soc}_output}
        DEPENDS ${CMAKE_CURRENT_LIST_DIR}/kernel/${tac}_${soc}.txt
        WORKING_DIRECTORY ${OPS_PROJECT_ROOT_DIR}
        COMMAND xxd -r -ps
            ${CMAKE_CURRENT_LIST_DIR}/kernel/${tac}_${soc}.txt
            ${${tac}_${soc}_output}
        COMMAND cp
            ${CMAKE_CURRENT_LIST_DIR}/kernel/${tac}_${soc}.json
            ${${tac}_${soc}_dir}/${tac}.json
    )
    add_custom_target(binary_${tac}_${soc} ALL
        DEPENDS ${${tac}_${soc}_output}
    )
endmacro()

macro(add_aicpu_kernel kernel srcs tac)
    add_compile_definitions(${tac}AicpuKernelPlaceHolder=${kernel})
    set(AICPU_BINARY_SRC_LIST ${AICPU_BINARY_SRC_LIST} ${CMAKE_CURRENT_LIST_DIR}/${srcs} PARENT_SCOPE)
endmacro()

macro(return_value_check return_value msg)
    if(NOT return_value EQUAL 0)
        message(FATAL_ERROR ${msg})
    endif()
endmacro()
