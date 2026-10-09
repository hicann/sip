#!/bin/bash
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
# torch_sip (sip_pta) 一键构建脚本：
#   bash build.sh            # 构建 whl
#   bash build.sh --install  # 构建并安装到当前 python 环境（pip install --force-reinstall --no-deps）
#   bash build.sh --clean    # 清理构建产物与三方依赖缓存
#
# 脚本自动完成：
#   1. 环境自检（python3 / torch / torch_npu / CANN set_env.sh）
#   2. 检查并自动拉取 ascend-boost-comm（提供 MKI 头文件，BOOST_COMM_PATH 默认指向它）
#   3. python3 setup.py build bdist_wheel 生成 dist/torch_sip-*.whl

set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CODE_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)               # sip 仓库根目录
THIRD_PARTY_DIR="${CODE_ROOT}/3rdparty"
BOOST_COMM_DIR="${THIRD_PARTY_DIR}/ascend-boost-comm"
BOOST_COMM_URL="https://gitcode.com/cann/ascend-boost-comm.git"
BOOST_COMM_BRANCH="master"
# 与 sip 仓 build.sh 保持一致的 MKI 头文件目录（含 mki/ 子目录）
DEFAULT_BOOST_COMM_INCLUDE="${BOOST_COMM_DIR}/src/include"

ACTION=""
for arg in "$@"; do
    case "${arg}" in
        --install) ACTION="install" ;;
        --clean)   ACTION="clean" ;;
        -h|--help) ACTION="help" ;;
        *) echo "unknown option: ${arg}"; ACTION="help" ;;
    esac
done

function help_info() {
    echo "Usage: bash build.sh [options]"
    echo
    echo "options:"
    echo "  (no option)   仅构建 whl 包（默认）"
    echo "  --install     构建并安装 whl 到当前 python 环境"
    echo "  --clean       清理构建产物、三方依赖缓存"
    echo "  -h, --help    显示帮助信息"
}

function fn_check_cmd() {
    if ! command -v "$1" &> /dev/null; then
        echo "[ERROR] command not found: $1, please install it first."
        exit 1
    fi
}

function fn_check_env()
{
    fn_check_cmd python3
    fn_check_cmd pip3
    fn_check_cmd git

    # torch / torch_npu 必须已安装
    if ! python3 -c "import torch" &> /dev/null; then
        echo "[ERROR] torch is not installed in current python env."
        echo "        please 'conda activate your_env' and install torch + torch_npu first."
        exit 1
    fi
    if ! python3 -c "import torch_npu" &> /dev/null; then
        echo "[ERROR] torch_npu is not installed in current python env."
        exit 1
    fi

    # CANN 环境：ASCEND_HOME_PATH 未设置时尝试自动 source 常见路径
    if [ -z "${ASCEND_HOME_PATH}" ]; then
        local candidates=(
            "/usr/local/Ascend/ascend-toolkit/set_env.sh"
            "/usr/local/Ascend/latest/set_env.sh"
            "${HOME}/Ascend/ascend-toolkit/set_env.sh"
        )
        for f in "${candidates[@]}"; do
            if [ -f "${f}" ]; then
                echo "[INFO] ASCEND_HOME_PATH not set, auto source ${f}"
                # shellcheck disable=SC1090
                source "${f}"
                break
            fi
        done
        if [ -z "${ASCEND_HOME_PATH}" ]; then
            echo "[WARN] ASCEND_HOME_PATH is empty; if build fails on acl headers,"
            echo "       please 'source <cann>/set_env.sh' and retry."
        fi
    fi

    # asdsip 库：链接期必需（libasdsip.so）。ASDSIP_HOME_PATH 未设置时不能盲信默认路径
    # /usr/local/Ascend/asdsip/latest（可能是不完整安装，缺 libasdsip.so），需实际探测
    if [ -z "${ASDSIP_HOME_PATH}" ]; then
        local default_sip="/usr/local/Ascend/asdsip/latest"
        if [ -f "${default_sip}/lib/libasdsip.so" ]; then
            export ASDSIP_HOME_PATH="${default_sip}"
        else
            echo "[ERROR] ASDSIP_HOME_PATH is not set and ${default_sip}/lib/libasdsip.so not found."
            echo "        please source <asdsip install dir>/set_env.sh first, e.g.:"
            echo "        source /usr/local/Ascend/asdsip/set_env.sh"
            exit 1
        fi
    fi
    if [ ! -f "${ASDSIP_HOME_PATH}/lib/libasdsip.so" ]; then
        echo "[ERROR] ${ASDSIP_HOME_PATH}/lib/libasdsip.so not found,"
        echo "        ASDSIP_HOME_PATH=${ASDSIP_HOME_PATH} may be an incomplete install."
        echo "        please source <asdsip install dir>/set_env.sh first."
        exit 1
    fi
    echo "[INFO] ASDSIP_HOME_PATH=${ASDSIP_HOME_PATH}"

    echo "[INFO] python   : $(python3 --version)"
    echo "[INFO] torch    : $(python3 -c 'import torch; print(torch.__version__)')"
    echo "[INFO] torch_npu: $(python3 -c 'import torch_npu; print(torch_npu.__version__)')"
}

function fn_prepare_boost_comm()
{
    # MKI 头文件来源：
    #   1. BOOST_COMM_PATH 显式指定 → 信任用户配置
    #   2. 仓库 3rdparty/ascend-boost-comm 已存在（执行过 sip 仓 build.sh）→ 直接复用
    #   3. 都没有 → 自动 git clone 到 3rdparty/
    if [ -n "${BOOST_COMM_PATH}" ] && [ -d "${BOOST_COMM_PATH}/mki" ]; then
        echo "[INFO] BOOST_COMM_PATH=${BOOST_COMM_PATH} (from env, mki headers found)"
        return 0
    fi

    if [ ! -d "${DEFAULT_BOOST_COMM_INCLUDE}/mki" ]; then
        echo "[INFO] ascend-boost-comm not found, cloning from ${BOOST_COMM_URL} ..."
        mkdir -p "${THIRD_PARTY_DIR}"
        if [ -d "${BOOST_COMM_DIR}" ]; then
            rm -rf "${BOOST_COMM_DIR}"
        fi
        git clone "${BOOST_COMM_URL}" -b "${BOOST_COMM_BRANCH}" "${BOOST_COMM_DIR}"
    fi

    if [ -d "${DEFAULT_BOOST_COMM_INCLUDE}/mki" ]; then
        export BOOST_COMM_PATH="${DEFAULT_BOOST_COMM_INCLUDE}"
        echo "[INFO] BOOST_COMM_PATH=${BOOST_COMM_PATH}"
    else
        echo "[ERROR] mki headers not found in ${DEFAULT_BOOST_COMM_INCLUDE}"
        echo "        please check ascend-boost-comm repo layout or set BOOST_COMM_PATH manually."
        exit 1
    fi
}

function fn_clean()
{
    echo "[INFO] cleaning build artifacts ..."
    rm -rf ${SCRIPT_DIR}/build ${SCRIPT_DIR}/dist ${SCRIPT_DIR}/*.egg-info
    find "${SCRIPT_DIR}" -maxdepth 4 -type d -name "__pycache__" -exec rm -rf {} + 2> /dev/null || true
    # ascend-boost-comm 三方依赖目录默认保留（避免误删非本脚本管理的文件），如需清理请手动删除
    if [[ -d ${BOOST_COMM_DIR} ]]; then
        echo "[INFO] keep ${BOOST_COMM_DIR} (delete it manually if not needed)"
    fi
    echo "[INFO] clean done."
}

function fn_build()
{
    fn_check_env
    fn_prepare_boost_comm

    # 并行编译优化：
    #   1. 优先启用 ninja（增量构建 + 自动并行），未安装时回退 make
    #   2. 回退 make 时通过 MAX_JOBS 控制并行度（默认 CPU 核数，上限 64 防 OOM）
    local nproc_num
    nproc_num=$(nproc)
    local max_jobs="${MAX_JOBS:-$(( nproc_num > 64 ? 64 : nproc_num ))}"
    export MAX_JOBS="${max_jobs}"
    if command -v ninja &> /dev/null; then
        export USE_NINJA=1
        echo "[INFO] USE_NINJA=1, MAX_JOBS=${max_jobs}"
    else
        echo "[INFO] ninja not found, fallback to make with MAX_JOBS=${max_jobs}"
        echo "       (pip install ninja for parallel incremental build)"
    fi

    cd "${SCRIPT_DIR}"
    echo "[INFO] building torch_sip whl ..."
    python3 setup.py build bdist_wheel

    local whl
    whl=$(ls -t "${SCRIPT_DIR}"/dist/torch_sip-*.whl 2> /dev/null | head -1)
    if [ -z "${whl}" ]; then
        echo "[ERROR] whl not found in ${SCRIPT_DIR}/dist"
        exit 1
    fi
    echo "[INFO] build success: ${whl}"

    if [ "${ACTION}" == "install" ]; then
        echo "[INFO] installing ${whl} ..."
        pip3 install --force-reinstall --no-deps "${whl}"
        echo "[INFO] installed. Run-time env setup:"
        echo "       source <asdsip install dir>/set_env.sh"
        echo "       source <cann install dir>/set_env.sh"
        echo "       conda activate your_env"
    else
        echo "[INFO] install manually with:"
        echo "       pip3 install --force-reinstall --no-deps ${whl}"
    fi
}

case "${ACTION}" in
    clean) fn_clean ;;
    help)  help_info ;;
    *)     fn_build ;;
esac
