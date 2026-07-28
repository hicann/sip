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

g++  example_scopy.cpp \
    -I${ASCEND_HOME_PATH}/include/aclnn \
    -I${ASCEND_HOME_PATH}/include \
    -L${ASCEND_HOME_PATH}/lib64/ -lascendcl -lopapi -lnnopbase \
    -I${ASDSIP_HOME_PATH}/include \
    -L${ASDSIP_HOME_PATH}/lib -lmki \
    -L${ASDSIP_HOME_PATH}/lib -lasdsip \
    -L${ASDSIP_HOME_PATH}/lib -lasdsip_core \
    -L${ASDSIP_HOME_PATH}/lib -lasdsip_host \
    -o example

echo "Execute the example of CopyOperation, calculate  vector copy:"
./example
rm example
