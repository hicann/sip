#!/bin/bash
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#

set -e

function fn_init_use_cxx11_abi()
{
    res=$(python3 -c "import torch" &> /dev/null || echo "torch_not_exist")
    if [ "$res" == "torch_not_exist" ]; then
        echo "Warning: Torch is not installed!"
        [[ "$USE_CXX11_ABI" == "" ]] && USE_CXX11_ABI=ON
        echo "USE_CXX11_ABI=$USE_CXX11_ABI"
        return 0
    fi

    if [ "$USE_CXX11_ABI" == "" ]; then
        if [ "$(python3 -c 'import torch; print(torch.compiled_with_cxx11_abi())')" == "True" ]; then
            USE_CXX11_ABI=ON
        else
            USE_CXX11_ABI=OFF
        fi
    fi
    echo "USE_CXX11_ABI=$USE_CXX11_ABI"
}

function fn_build_mki()
{
    cd $MKI_ROOT
    [[ -d "$MKI_ROOT"/build ]] && rm -rf $MKI_ROOT/build
    [[ -d "$MKI_ROOT"/output ]] && rm -rf $MKI_ROOT/output

    echo  "current commid id of ascend-boost-comm: $(git rev-parse HEAD)"

    build_options="--use_cxx11_abi=0"
    bash scripts/build.sh $build_options
}

function fn_select_ops_fft_soc()
{
    # 自动推导 ops-fft 存件 SoC, 不提供手动开关:
    #   仅启用 910b/950 之一 → 跟随该目标; 两者均启用 → 按构建机架构选取
    #   (910B 存件为 aarch64, 950 存件为 x86_64); 均未启用 → 输出空, 跳过联合构建
    local targets
    targets=$(python3 -c "import sys; sys.path.insert(0, '${CODE_ROOT}/scripts'); \
import build_util; print(' '.join(build_util.get_build_target_list()))" 2>/dev/null)
    local has_910b=0
    local has_950=0
    [[ "${targets}" == *"ascend910b"* ]] && has_910b=1
    [[ "${targets}" == *"ascend950"* ]] && has_950=1

    if [ ${has_910b} -eq 1 ] && [ ${has_950} -eq 0 ]; then
        echo "Ascend910B"
    elif [ ${has_950} -eq 1 ] && [ ${has_910b} -eq 0 ]; then
        echo "Ascend950"
    elif [ ${has_910b} -eq 1 ] && [ ${has_950} -eq 1 ]; then
        if [ "$(uname -m)" = "aarch64" ]; then
            echo "Ascend910B"
        else
            echo "Ascend950"
        fi
    fi
}

function fn_prepare_ops_fft_package()
{
    # ops-fft 联合构建为显式开启(--use_ops_fft), 默认不构建:
    # 开启时存件为强依赖, 缺失/失配/目标不含 910b/950 报错终止
    if [ "${USE_OPS_FFT}" != "ON" ]; then
        echo "ops-fft: 未开启联合构建(默认纯 internal, 需要时加 --use_ops_fft)"
        export OPS_FFT_PKG_VALID=0
        return 0
    fi
    local pkg="${THIRD_PARTY_DIR}/ops-fft"
    local soc
    soc=$(fn_select_ops_fft_soc)
    if [ -z "${soc}" ]; then
        echo "[ERROR] ops-fft: --use_ops_fft 但 build targets 不含 ascend910b/ascend950"
        exit 1
    fi
    if [ -n "${OPS_FFT_SOC}" ] && [ "${OPS_FFT_SOC}" != "${soc}" ]; then
        echo "ops-fft: 环境变量 OPS_FFT_SOC=${OPS_FFT_SOC} 已忽略, 自动推导为 ${soc}"
    fi
    export OPS_FFT_SOC="${soc}"   # 导出供 cmake 选择制品目录
    local so="${pkg}/lib/${soc}/libcann_ops_fft.so"
    local manifest="${pkg}/lib/${soc}/manifest.info"

    # 逐项校验: soc/arch 为硬门禁, cann_version/abi 仅信息性记录
    # (CANN ABI 向后兼容且接口为 extern "C" 纯 C, 均不影响二进制兼容)
    local host_arch="$(uname -m)"
    local so_arch="unknown"
    case "$(readelf -h "${so}" 2>/dev/null | sed -n 's/^ *Machine: *//p')" in
        AArch64) so_arch="aarch64" ;;
        *X86-64*) so_arch="x86_64" ;;
    esac
    local mismatch=""
    if [ ! -f "${so}" ]; then
        mismatch="so 缺失: ${so}"
    elif [ ! -f "${manifest}" ]; then
        mismatch="manifest 缺失: ${manifest}"
    elif ! grep -q "^soc=${soc}$" "${manifest}"; then
        mismatch="soc 不匹配 (需 ${soc}, manifest: $(grep '^soc=' ${manifest}))"
    elif [ "${so_arch}" != "unknown" ] && [ "${so_arch}" != "${host_arch}" ]; then
        mismatch="arch 不匹配 (本机 ${host_arch}, 存件 ${so_arch}: ${so})"
    fi

    if [ -z "${mismatch}" ]; then
        echo "ops-fft binary package ready: ${so} (soc=${soc}, arch=${so_arch})"
        export OPS_FFT_PKG_VALID=1
        return 0
    fi

    # --use_ops_fft 下存件为强依赖: 缺失/失配直接报错终止
    echo "================================================================================"
    echo "[ERROR] ops-fft binary package validation failed (--use_ops_fft), build aborted:"
    echo "[ERROR]   ${mismatch}"
    echo "[ERROR] see docs/ops_fft_joint_build.md (存件更新) or use a matched-arch host"
    echo "================================================================================"
    exit 1
}

function fn_collect_ops_fft_artifacts()
{
    local pkg="${THIRD_PARTY_DIR}/ops-fft"
    local soc="${OPS_FFT_SOC}"
    if [ "${OPS_FFT_PKG_VALID}" != "1" ]; then
        return
    fi
    mkdir -p ${OUTPUT_DIR}/lib ${OUTPUT_DIR}/include
    cp -af "${pkg}/lib/${soc}"/libcann_ops_fft.so* ${OUTPUT_DIR}/lib/
    cp -f "${pkg}/include/cann_ops_fft.h" ${OUTPUT_DIR}/include/ 2>/dev/null
    echo "collect ops-fft artifacts: libcann_ops_fft.so, cann_ops_fft.h -> ${OUTPUT_DIR}"
}

function fn_make_run_package()
{
    if [ $( uname -a | grep -c -i "x86_64" ) -ne 0 ]; then
        echo "it is system of x86_64"
        ARCH="x86_64"
    elif [ $( uname -a | grep -c -i "aarch64" ) -ne 0 ]; then
        echo "it is system of aarch64"
        ARCH="aarch64"
    else
        echo "it is not system of aarch64 or x86_64"
        exit 1
    fi
    branch=$(git symbolic-ref -q --short HEAD || git describe --tags --exact-match 2> /dev/null || echo $branch)
    commit_id=$(git rev-parse HEAD)
    touch $OUTPUT_DIR/version.info
    cat>$OUTPUT_DIR/version.info<<EOF
    Ascend-cann-asdsip : ${VERSION}
    Ascend-cann-asdsip Version : ${VERSION_B}
    Platform : ${ARCH}
    branch : ${branch}
    commit id : ${commit_id}
EOF

    rm -rf $CODE_ROOT/output/host
    rm -rf $CODE_ROOT/output/device/

    # api.h 归档
    chmod 755 -R $CODE_ROOT/scripts/install.sh
    cp $CODE_ROOT/scripts/install.sh $CODE_ROOT/output/
    cp $CODE_ROOT/scripts/set_env.sh $CODE_ROOT/output/


    chmod 755 -R $CODE_ROOT/output

    ARCH=`uname -m`

    mkdir -p $OUTPUT_DIR/scripts
    cp $CODE_ROOT/scripts/install.sh $OUTPUT_DIR
    cp $CODE_ROOT/scripts/set_env.sh $OUTPUT_DIR
    cp $CODE_ROOT/scripts/uninstall.sh $OUTPUT_DIR/scripts
    cp $CODE_ROOT/scripts/filelist.csv $OUTPUT_DIR/scripts
    # 收编的 ops-fft 制品登记进 filelist.csv, 否则 --uninstall 按 csv 清理会遗留孤儿文件
    if ls ${OUTPUT_DIR}/lib/libcann_ops_fft.so* > /dev/null 2>&1; then
        for f in ${OUTPUT_DIR}/lib/libcann_ops_fft.so*; do
            echo "lib/$(basename ${f})"
        done >> ${OUTPUT_DIR}/scripts/filelist.csv
        if [ -f ${OUTPUT_DIR}/include/cann_ops_fft.h ]; then
            echo "include/cann_ops_fft.h" >> ${OUTPUT_DIR}/scripts/filelist.csv
        fi
    fi
    sed -i "s/ASDSIPPKGARCH/${ARCH}/" $OUTPUT_DIR/install.sh
    sed -i "s!VERSION_PLACEHOLDER!${VERSION}!" $OUTPUT_DIR/install.sh
    sed -i "s!LOG_PATH_PLACEHOLDER!${LOG_PATH}!" $OUTPUT_DIR/install.sh
    sed -i "s!LOG_NAME_PLACEHOLDER!${LOG_NAME}!" $OUTPUT_DIR/install.sh
    sed -i "s!VERSION_PLACEHOLDER!${VERSION}!" $OUTPUT_DIR/scripts/uninstall.sh
    sed -i "s!LOG_PATH_PLACEHOLDER!${LOG_PATH}!" $OUTPUT_DIR/scripts/uninstall.sh
    sed -i "s!LOG_NAME_PLACEHOLDER!${LOG_NAME}!" $OUTPUT_DIR/scripts/uninstall.sh

    makeself_dir=${ASCEND_HOME_PATH}/toolkit/tools/op_project_templates/ascendc/customize/cmake/util/makeself
    $makeself_dir/makeself.sh --header $makeself_dir/makeself-header.sh \
        --help-header $CODE_ROOT/scripts/help.info --pigz --complevel 4 --nomd5 --sha256 --chown \
        $CODE_ROOT/output $OUTPUT_DIR/Ascend-cann-SIP_${VERSION}_linux-${ARCH}.run ASCEND_SIP_RUN_PACKAGE ./install.sh

    echo "Ascend-cann-SIP_${VERSION}_linux-${ARCH}.run is successfully generated in $OUTPUT_DIR"
}

function fn_compile_and_pack()
{
    cmake $1 $2
    if [ "$USE_VERBOSE" == "ON" ];then
        VERBOSE=1 make -j
    else
        make -j64
    fi
    make install
    fn_collect_ops_fft_artifacts
    fn_make_run_package
}

function fn_install_lcov()
{
    LCOV_PACK_PATH=${CODE_ROOT}/lcov-1.16
    LCOV_BIN_PATH=${CODE_ROOT}/lcov
    if [ ! -d $LCOV_BIN_PATH ]; then
        if [ ! -d $LCOV_PACK_PATH ]; then
            git clone --branch v1.16 --depth 1 https://gitcode.com/gh_mirrors/lc/lcov.git $LCOV_PACK_PATH
        fi
        cd ${LCOV_PACK_PATH}
        make -j
        make PREFIX=${LCOV_BIN_PATH} install
    fi
}

function fn_build_googletest()
{
    THIRD_PARTY_DIR_PATH=${THIRD_PARTY_DIR}
    GTEST_DIR=$THIRD_PARTY_DIR_PATH/googletest
    if [ ! -d $GTEST_DIR ]; then
        [[ ! -d $THIRD_PARTY_DIR_PATH ]] && mkdir -p $THIRD_PARTY_DIR_PATH
        cd $THIRD_PARTY_DIR_PATH
        wget https://gitcode.com/cann-src-third-party/googletest/releases/download/v1.14.0/googletest-1.14.0.tar.gz
        tar -xf googletest-1.14.0.tar.gz
        rm googletest-1.14.0.tar.gz
    fi
    cd $CODE_ROOT
}

function fn_ut_test_needed()
{
    export NEED_COMPILE_RT="TRUE"
    export TEST_TYPE="UT"
    fn_install_lcov
    fn_build_googletest
}

function fn_build_coverage()
{

    export GCOV_DIR=$CACHE_DIR/gcov
    PYTHON_FILTER_TOOL=$CODE_ROOT/tests/ut/framework/test_util/FilterTool.py
    LCOV_PATH=${CODE_ROOT}/lcov/bin/lcov

    rm -rf $CACHE_DIR/core/CMakeFiles/asdops_static.dir/

    [ -n "$GCOV_DIR" ] && rm -rf $GCOV_DIR
    mkdir $GCOV_DIR

    if [ "$TEST_TYPE" == "UT" ]; then
        fn_run_unittest
    fi
    # if [ "$TEST_TYPE" == "FT" ]; then
    #     fn_run_fuzztest
    # fi

    # cd $GCOV_DIR
    # echo "CURRENT_DIR=${CURRENT_DIR}"
    # $LCOV_PATH -c --directory ${CURRENT_DIR} --output-file tmp_coverage.info --rc lcov_branch_coverage=1 >> $GCOV_DIR/log.txt
    # $LCOV_PATH -r tmp_coverage.info '*/3rdparty/*' '*/build/*' '*torch/*' '*c10/*' '*ATen/*' '*/c++/7*' '*tests/*' '*tools/*' '*torch_extension/*' '/opt/*'  '*/core/tbe/stubs/*' '/usr/*' '*/ascend-op-common-lib/*' '*/asdops/*' '*/Ascend/*' -output-file test_coverage.info --rc lcov_branch_coverage=1 >> $GCOV_DIR/log.txt
    # $LCOV_PATH -a test_coverage.info -o main_coverage.info --rc lcov_branch_coverage=1 >> $GCOV_DIR/log.txt
    # python3 $PYTHON_FILTER_TOOL --input ./main_coverage.info --output ./final.info --root $CACHE_DIR --debug 1 >> $GCOV_DIR/log.txt
    # ${CODE_ROOT}/lcov/bin/genhtml --branch-coverage final.info -o cover_result --rc lcov_branch_coverage=1 >> $GCOV_DIR/log.txt
    # [[ ! -d ./cov_info ]] && mkdir cov_info
    # cp final.info ./cov_info
    # tail -n 4 $GCOV_DIR/log.txt
    # cd ..
    # tar -czf gcov.tar.gz gcov
    # mv gcov.tar.gz $OUTPUT_DIR/
}

function fn_run_unittest()
{
    echo " CURRENT DIRECTORY: $(pwd)"
    echo " UT CURRENT_DIR=${CURRENT_DIR}"
    echo " UT OUTPUT_DIR=${OUTPUT_DIR}"

    export LD_LIBRARY_PATH=$OUTPUT_DIR/lib/:$LD_LIBRARY_PATH
    $OUTPUT_DIR/bin/ops_unittest --gtest_output=xml:test_detail.xml
    cp test_detail.xml unittest_result.xml
}

function fn_build()
{
    # check dependency mki
    MKI_PATH="${THIRD_PARTY_DIR}/mki"
    COMPILER_PATH="${THIRD_PARTY_DIR}/compiler"
    if [ ! -d "${MKI_PATH}" ] || [ ! -d "${COMPILER_PATH}" ]; then
        mkdir -p 3rdparty
        if [ ! -d "${MKI_ROOT}" ]; then
            echo "Third_party dir does not complete and ascend-boost-comm does not exit!"
            cd ${THIRD_PARTY_DIR}/
            git clone https://gitcode.com/cann/ascend-boost-comm.git -b master
        fi
        cd $CODE_ROOT/
        rm -rf MKI_PATH
        rm -rf COMPILER_PATH
        fn_build_mki
        cp -r ${THIRD_PARTY_DIR}/ascend-boost-comm/output/mki ${THIRD_PARTY_DIR}/
        cp -r ${THIRD_PARTY_DIR}/ascend-boost-comm/3rdparty/compiler ${THIRD_PARTY_DIR}/
    fi

    # check dependency catlass
    CATLASS_PATH="${THIRD_PARTY_DIR}/catlass"
    if [ ! -d "${CATLASS_PATH}" ]; then
        echo "Third_party dir catlass does not exit!"
        cd ${THIRD_PARTY_DIR}/
        git clone https://gitcode.com/cann/catlass.git -b master
        cd $CODE_ROOT/
    fi

    # ops-fft 二进制存件准入 (默认不构建; --use_ops_fft 显式开启, 缺失/失配时报错终止)
    fn_prepare_ops_fft_package

    cmake -B build -S . -DCURRENT_DIR="$CURRENT_DIR"

    cd $CODE_ROOT/
    echo  "current commid id of ascendSipBoost: $(git rev-parse HEAD)"

    rm -rf build output
    [[ ! -d build ]] && mkdir build
    cd build

    echo "COMPILE_OPTIONS:$COMPILE_OPTIONS"
    COMPILE_OPTIONS="${COMPILE_OPTIONS}  .."

    fn_compile_and_pack "$CODE_ROOT" "$COMPILE_OPTIONS"
}

function help_info() {
    echo "Usage: bash build.sh [type] [options]"
    echo
    echo "type:"
    echo "--help                         Displays help message."
    echo "--dev                          仅编译算子库, 若type为空，默认为dev."
    echo "--clean                        清除缓存和依赖的三方库."
    echo "--ut                           编译执行单元测试用例."
    echo
    echo "options:"
    echo "--output=<dir>               指定编译输出目录，默认为${repo}/output目录."
    echo "--use_cxx11_abi=0            设置-D_GLIBCXX_USE_CXX11_ABI=0 (默认选项)."
    echo "--use_cxx11_abi=1            设置-D_GLIBCXX_USE_CXX11_ABI=1."
    echo "--use_ops_fft                开启 ops-fft 联合构建(默认不构建; 开启后存件缺失/失配时报错终止)."
    echo "--verbose                    打印详细的编译命令."
    echo "--mssanitizer                启用mssanitizer."
    echo
}

function fn_main()
{
     if [[ "$BUILD_OPTION_LIST" =~ "$1" ]];then
        if [[ -z "$1" ]];then
            arg1="--dev"
        else
            arg1=$1
            shift
        fi
    else
        cfg_flag=0
        for item in ${BUILD_CONFIGURE_LIST[*]};do
            if [[ "$1" =~ $item ]];then
                cfg_flag=1
                break 1
            fi
        done
        if [[ "$cfg_flag" == 1 ]];then
            arg1="--dev"
        else
            echo "argument $1 is unknown, please type 'build.sh --help' for more information"
            exit 1
        fi
    fi

    until [[ -z "$1" ]]
    do {
        arg2=$1
        case "${arg2}" in
        --output=*)
            arg2=${arg2#*=}
            if [ -z "$arg2" ];then
                echo "the output directory is not set. This should be set like --output=<outputDir>"
            else
                cd $CURRENT_DIR
                if [ ! -d "$arg2" ];then
                    mkdir -p $arg2
                fi
                export OUTPUT_DIR=$(cd $arg2; pwd)
            fi
            ;;
        "--use_cxx11_abi=1")
            USE_CXX11_ABI=ON
            ;;
        "--use_cxx11_abi=0")
            USE_CXX11_ABI=OFF
            ;;
        "--use_ops_fft")
            USE_OPS_FFT=ON
            ;;
        "--verbose")
            USE_VERBOSE=ON
            export VERBOSE=1
            ;;
        "--mssanitizer")
            COMPILE_OPTIONS="${COMPILE_OPTIONS} -DUSE_MSSANITIZER=ON"
            ;;
        esac
        shift
    }
    done

    COMPILE_OPTIONS="${COMPILE_OPTIONS} -DUSE_CXX11_ABI=$USE_CXX11_ABI"
    COMPILE_OPTIONS="${COMPILE_OPTIONS} -DCMAKE_BUILD_TYPE=Release"
    case "${arg1}" in
        --dev)
            fn_build
            ;;
        "ut")
            fn_ut_test_needed
            fn_build
            fn_build_coverage
            ;;
        --ut)
            fn_ut_test_needed
            fn_build
            fn_build_coverage
            ;;
        "smoke_pr")
            fn_build
            bash $CODE_ROOT/scripts/build_test.sh example_test
            ;;
        "smoke_all")
            fn_build
            bash $CODE_ROOT/scripts/build_test.sh default
            ;;
        --clean)
            [[ -d "$OUTPUT_DIR" ]] && rm -rf $OUTPUT_DIR
            [[ -d "$CACHE_DIR" ]] && rm -rf $CACHE_DIR
            [[ -d "$THIRD_PARTY_DIR" ]] && rm -rf $THIRD_PARTY_DIR
            echo "clear all build history."
            ;;
        *)
            help_info
            ;;
    esac
}

set -e
cd $(dirname $0)
export CURRENT_DIR=$(pwd)

CURRENT_DIR=$(pwd)
export CODE_ROOT=${CURRENT_DIR}
export CACHE_DIR=$CODE_ROOT/build
export OUTPUT_DIR=$CODE_ROOT/output
THIRD_PARTY_DIR=$CODE_ROOT/3rdparty
ASDSIP_DIR=$CODE_ROOT
RELEASE_DIR=$CODE_ROOT/ci/release
MKI_ROOT=$THIRD_PARTY_DIR/ascend-boost-comm
VERSION="9.1.0"
VERSION_B="9.1.0"
LOG_PATH="/var/log/cann_asdsip_log/"
LOG_NAME="cann_asdsip_install.log"

export COMPILE_OPTIONS="-DNO_WERROR=ON"
export USE_VERBOSE=OFF
export USE_CXX11_ABI="OFF"
export USE_OPS_FFT="OFF"
BUILD_OPTION_LIST="ops_unit ut st ft smoke_pr smoke_all --ut --dev --clean --help"
BUILD_CONFIGURE_LIST=("--output=.*" "--use_cxx11_abi=0" "--use_cxx11_abi=1 --verbose --mssanitizer" "--use_ops_fft")

fn_main "$@"
