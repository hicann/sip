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

BUILD_DIR=$(dirname $(readlink -f $0))
CANN_DIR=$(readlink -f $BUILD_DIR/../CANN)

# install cann
function fn_install_cann_and_kernel()
{
    cd $CANN_DIR
    chmod +x *.run
    cann_install_path="/home/slave1/Ascend/ascend-toolkit"
    if [ ! -d "$cann_install_path" ];then
        ./CANN-runtime-*.run --full --quiet --nox11 --install-path=${cann_install_path}
        ./CANN-compiler-*.run --full --pylocal --quiet --nox11 --install-path=${cann_install_path}
        ./CANN-opp-*.run --full --quiet --nox11 --install-path=${cann_install_path}
        ./CANN-toolkit-*.run --full --pylocal --quiet --nox11 --install-path=${cann_install_path}
        ./CANN-aoe-*.run --full --quiet --nox11 --install-path=${cann_install_path}
        ./Ascend910B-opp_kernel-*.run --full --quiet --nox11 --install-path=${cann_install_path}
        ./Ascend310P-opp_kernel-*.run --full --quiet --nox11 --install-path=${cann_install_path}
        ./CANN-hccl-*.run --full --quiet --nox11 --install-path=${cann_install_path}
    fi
    set +e
    source /home/slave1/Ascend/ascend-toolkit/latest/bin/setenv.bash
    export ASCEND_HOME_PATH=/home/slave1/Ascend/ascend-toolkit/latest
    set -e
    cd -
}

function fn_build_mki()
{
    cd $MKI_ROOT
    [[ -d "$CODE_ROOT"/../Mind-KernelInfra/build ]] && rm -rf $CODE_ROOT/../Mind-KernelInfra/build
    [[ -d "$CODE_ROOT"/../Mind-KernelInfra/output ]] && rm -rf $CODE_ROOT/../Mind-KernelInfra/output

    echo  "current commid id of Mind-KernelInfra: $(git rev-parse HEAD)"

    build_options="--use_cxx11_abi=0"
    bash scripts/build.sh release $build_options
    cd $MKI_ROOT/output
    tar -xf mki.tar.gz
    rm -f mki.tar.gz
}

function fn_build_asdops()
{
    COMMONLIB_THIRD_PARTY_DIR=$ASDOPS_ROOT/3rdparty
    [[ -d "$CODE_ROOT"/../ascend-op-common-lib/build ]] && rm -rf $CODE_ROOT/../ascend-op-common-lib/build
    [[ -d "$CODE_ROOT"/../ascend-op-common-lib/output ]] && rm -rf $CODE_ROOT/../ascend-op-common-lib/output

    [[ ! -d "$COMMONLIB_THIRD_PARTY_DIR" ]] && mkdir $COMMONLIB_THIRD_PARTY_DIR
    [[ -d "$COMMONLIB_THIRD_PARTY_DIR/mki" ]] && rm -rf $COMMONLIB_THIRD_PARTY_DIR/mki

    cd $ASDOPS_ROOT

    # copy asdops output
    cp -r $MKI_ROOT/output/mki 3rdparty

    echo  "current commid id of ascend-op-common-lib: $(git rev-parse HEAD)"
    bash scripts/build.sh release --use_cxx11_abi=0 --no_werror
}

function fn_init_makeself()
{
    if [ -d "$ASDSIP_DIR/opensource/makeself" ]; then
        echo "makeself exist in $ASDSIP_DIR, not build it"
        return
    fi
    [[ ! -d "$ASDSIP_DIR/opensource" ]] && mkdir $ASDSIP_DIR/opensource
    cd $ASDSIP_DIR/opensource/
    rm -rf makeself
    git clone -b release-2.5.0 https://gitcode.com/gh_mirrors/ma/makeself.git
}

function fn_build_catlass()
{
    if [ -d "$THIRD_PARTY_DIR/catlass" ]; then
        echo "catlass already exists in $THIRD_PARTY_DIR, skip cloning"
        return 0
    fi
    [[ ! -d "$THIRD_PARTY_DIR" ]] && mkdir $THIRD_PARTY_DIR
    cd $THIRD_PARTY_DIR
    branch=9.0.0-beta.1
    echo "current branch for catlass: $branch"
    git clone --branch $branch --depth 1 https://gitcode.com/cann/catlass.git
}

function config_asdsip_version()
{
    if [ -z "$VERSION" ]; then
        echo "PackageName is not set, use default setting!"
        VERSION='1.0.RC1'
        echo "PackageName: $VERSION"
    fi
    if [ -z "$VERSION_B" ]; then
        echo "CANNVersion is not set, use default setting!"
        VERSION_B='1.0.RC1'
        echo "CANNVersion: $VERSION_B"
    fi
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
    rm -rf $CODE_ROOT/output/bin/

    # api.h 归档
    chmod 750 -R $CODE_ROOT/scripts/install.sh
    cp $CODE_ROOT/scripts/install.sh $CODE_ROOT/output/
    cp $CODE_ROOT/scripts/set_env.sh $CODE_ROOT/output/


    chmod 750 -R $CODE_ROOT/output
    chmod 750 -R $CODE_ROOT/opensource/makeself

    ARCH=`uname -i`

    mkdir -p $OUTPUT_DIR/scripts
    mkdir -p $RELEASE_DIR/$ARCH
    cp $CODE_ROOT/scripts/install.sh $OUTPUT_DIR
    cp $CODE_ROOT/scripts/set_env.sh $OUTPUT_DIR
    cp $CODE_ROOT/scripts/uninstall.sh $OUTPUT_DIR/scripts
    cp $CODE_ROOT/scripts/filelist.csv $OUTPUT_DIR/scripts
    sed -i "s/ASDSIPPKGARCH/${ARCH}/" $OUTPUT_DIR/install.sh
    sed -i "s!VERSION_PLACEHOLDER!${VERSION}!" $OUTPUT_DIR/install.sh
    sed -i "s!LOG_PATH_PLACEHOLDER!${LOG_PATH}!" $OUTPUT_DIR/install.sh
    sed -i "s!LOG_NAME_PLACEHOLDER!${LOG_NAME}!" $OUTPUT_DIR/install.sh
    sed -i "s!VERSION_PLACEHOLDER!${VERSION}!" $OUTPUT_DIR/scripts/uninstall.sh
    sed -i "s!LOG_PATH_PLACEHOLDER!${LOG_PATH}!" $OUTPUT_DIR/scripts/uninstall.sh
    sed -i "s!LOG_NAME_PLACEHOLDER!${LOG_NAME}!" $OUTPUT_DIR/scripts/uninstall.sh

    $CODE_ROOT/opensource/makeself/makeself.sh --header $CODE_ROOT/opensource/makeself/makeself-header.sh \
        --help-header $CODE_ROOT/scripts/help.info --pigz --complevel 4 --nomd5 --sha256 --chown \
        $CODE_ROOT/output $RELEASE_DIR/$ARCH/Ascend-cann-SIP_${VERSION}_linux-${ARCH}.run ASCEND_SIP_RUN_PACKAGE ./install.sh \

    rm -rf  $OUTPUT_DIR/*
    mkdir -p $OUTPUT_DIR/$ARCH
    cp $RELEASE_DIR/$ARCH/Ascend-cann-SIP_${VERSION}_linux-${ARCH}.run $OUTPUT_DIR/$ARCH
    echo "Ascend-cann-SIP_${VERSION}_linux-${ARCH}.run is successfully generated in $OUTPUT_DIR/$ARCH"
}


function fn_build()
{
    export ASCEND_HOME_PATH=/home/slave1/Ascend/ascend-toolkit/latest
    export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH}

    source_file="${ASCEND_HOME_PATH}/lib64/libopapi.so"

    # 检查源文件是否存在
    if [ ! -e "$source_file" ]; then
        echo "error '$source_file':File not exists!"
        exit 1
    fi

    if [ ! -d "$CODE_ROOT"/3rdparty ]; then
        cd $CODE_ROOT/
        mkdir 3rdparty
    else
        [[ -d "$CODE_ROOT"/3rdparty/mki ]] && rm -rf $CODE_ROOT/3rdparty/mki
        [[ -d "$CODE_ROOT"/3rdparty/compiler ]] && rm -rf $CODE_ROOT/3rdparty/compiler
    fi

    cp -r $CODE_ROOT/../Mind-KernelInfra/output/mki $CODE_ROOT/3rdparty/
    cp -r $CODE_ROOT/../Mind-KernelInfra/3rdparty/compiler $CODE_ROOT/3rdparty/

    cd $CODE_ROOT/
    echo  "current commid id of ascendSipBoost: $(git rev-parse HEAD)"

    rm -rf build output
    mkdir build
    cd build

    cmake -DUSE_CXX11_ABI=OFF -DCMAKE_BUILD_TYPE=Release -DNO_WERROR=ON ..
    make install -j64
}

function fn_init_cann_env()
{
    cann_default_install_path_1="/usr/local/Ascend/ascend-toolkit"
    cann_default_install_path_2="/home/slave1/Ascend/ascend-toolkit/latest"

    set +e
    if [ -d "${cann_default_install_path_1}" ];then
        source /usr/local/Ascend/ascend-toolkit/set_env.sh
    elif [ -d "${cann_default_install_path_2}" ]; then
        source /home/slave1/Ascend/ascend-toolkit/latest/set_env.sh
    else
        fn_install_cann_and_kernel
        if [ -z $ASCEND_HOME_PATH ];then
            echo "env ASCEND_HOME_PATH not exists, build fail"
            exit 1
        fi
    fi
    set -e
}

function bep_env_init() {
    # bep 消除二进制
    local bep_env_config=$ASDSIP_DIR/scripts/bep_env.conf
    # 检查BepKit预置环境
    local bep_sh=$(which bep_env.sh)
    echo "has bep sh :${bep_sh}"
    # 执行bep脚本
    if [ ! -d "${SECBEPKIT_HOME}" ] && [ ! -f "$bep_sh" ]; then
        echo "BepKit is uninstalled, Please install the tool and configure the env var \$SECBEPKIT_HOME"
    else
        source ${SECBEPKIT_HOME}/bep_env.sh -s $bep_env_config
        if [ $? -ne 0 ]; then
            echo "build bep failed!"
            exit 1
        else
            echo "build bep success."
        fi
    fi
}

function fn_main()
{
    if [[ "$1" == "pack" ]]; then
        arg1="pack"
        shift
    else
        arg1="compile"
        shift
    fi

    until [[ -z "$1" ]]
    do {
        case "$1" in
        "--PackageName="*)
            VERSION="${1#*=}"
            echo "PackageName set to: $VERSION"
            ;;
        "--CANNVersion="*)
            VERSION_B="${1#*=}"
            echo "CANNVersion set to: $VERSION_B"
            ;;
        "--use_cxx11_abi=1")
            USE_CXX11_ABI=ON
            ;;
        "--use_cxx11_abi=0")
            USE_CXX11_ABI=OFF
            ;;
        esac
        shift
    }
    done

    case "${arg1}" in
        "pack")
            config_asdsip_version
            fn_init_makeself
            fn_make_run_package
            ;;
        "compile")
            fn_build_mki
            fn_build_catlass
            fn_build
            ;;
    esac
}

set -e

SCRIPT_DIR=$(cd $(dirname $0); pwd)
cd $SCRIPT_DIR
cd ..
export CODE_ROOT=$(pwd)
MKI_ROOT=$CODE_ROOT/../Mind-KernelInfra
ASDOPS_ROOT=$CODE_ROOT/../ascend-op-common-lib
export CACHE_DIR=$CODE_ROOT/build
export OUTPUT_DIR=$CODE_ROOT/output
THIRD_PARTY_DIR=$CODE_ROOT/3rdparty
export ASDOPS_KERNEL_PATH=/tmp/asdops_dependency/2023-11-10/opp_kernel
ASDSIP_DIR=$CODE_ROOT
RELEASE_DIR=$CODE_ROOT/ci/release
VERSION="8.0.0"
LOG_PATH="/var/log/cann_asdsip_log/"
LOG_NAME="cann_asdsip_install.log"

bep_env_init
fn_main "$@"
