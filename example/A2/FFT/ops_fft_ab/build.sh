#!/bin/bash
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# You may refer to the License for details. You should not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#

# ops-fft 联合构建后端 A/B 验证样例: 编译并运行, 参数透传 (n batch, 默认 8192 4)
# 结果判读: "[AB] 双后端输出逐位一致" + 日志 "run via ops-fft kernel ... success"

# 检查环境变量 ASDSIP_HOME_PATH 是否已设置
if [ -z "$ASDSIP_HOME_PATH" ]; then
    echo "the env params ASDSIP_HOME_PATH is not set."
    exit 1
fi
# 检查目录是否存在，兼容路径以 latest 或 latest/ 结尾的情况
if [ ! -d "$ASDSIP_HOME_PATH" ]; then
    new_path="${ASDSIP_HOME_PATH%latest/}"
    new_path="${new_path%latest}"
    if [ ! -d "$new_path" ]; then
        echo "the env params ASDSIP_HOME_PATH is not exist."
        exit 1
    fi
    export ASDSIP_HOME_PATH=$new_path
fi
export LD_LIBRARY_PATH=$ASDSIP_HOME_PATH/lib:$LD_LIBRARY_PATH

g++  example_fft_backend_ab.cpp \
    -I${ASCEND_HOME_PATH}/include/aclnn \
    -I${ASCEND_HOME_PATH}/include \
    -L${ASCEND_HOME_PATH}/lib64/ -lascendcl -lopapi -lnnopbase \
    -I${ASDSIP_HOME_PATH}/include \
    -L${ASDSIP_HOME_PATH}/lib -lmki \
    -L${ASDSIP_HOME_PATH}/lib -lasdsip \
    -L${ASDSIP_HOME_PATH}/lib -lasdsip_core \
    -L${ASDSIP_HOME_PATH}/lib -lasdsip_host \
    -o example

echo "A/B verify: internal vs ops-fft backend (args: n batch)"
# 开启 INFO 日志以捕获 "run via ops-fft kernel" 命中日志; grep 锚定行首
# (时间戳前缀)过滤不完整行, awk 剥离时间戳后按内容去重(计时循环产生大量重复日志)
ASCEND_GLOBAL_LOG_LEVEL=1 ASCEND_SLOG_PRINT_TO_STDOUT=1 ./example "$@" 2>&1 \
    | grep -E "^\[AB\]|^\[[0-9]{4}-[0-9]{2}-[0-9]{2} .*(run via ops-fft|aborting)" \
    | awk '{k=$0; sub(/^\[[0-9-]+ [0-9:.]+\] /, "", k); if (!seen[k]++) print}'
rc=${PIPESTATUS[0]}
rm -f example
exit $rc
