#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
set -e

DEVICE_ID="${1:-3}"
SKIP_BUILD="${2:-0}"
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DATA_DIR="$SCRIPT_DIR/data"
EXE_NAME=fft_2d_c2c_4096x4096

echo "============================================"
echo " FFT C2C 2D [4096,4096] (FFT2D) Precision Test"
echo " device=${DEVICE_ID}  directions=-1,+1  (gate MARE < 2e-5)"
echo "============================================"

echo ""
echo "[1/3] Generating input and golden (numpy.fft2)..."
mkdir -p "$DATA_DIR"
# 清理上次运行残留的输出，保证首次 gen_data.py 仅生成输入与 golden；
# 精度比对在设备程序执行后的第二次 gen_data.py 中进行
rm -f "$DATA_DIR"/output_*.bin
for DIRECTION in -1 1; do
    python3 "$SCRIPT_DIR/gen_data.py" "$DIRECTION" 42 "$DATA_DIR"
done

if [ "$SKIP_BUILD" != "1" ]; then
    echo ""
    echo "[2/3] Building operator..."
    bash "$SCRIPT_DIR/build.sh"
fi

echo ""
echo "[3/3] Running NPU FFT operator and comparing precision..."
cd "$SCRIPT_DIR/build"
for DIRECTION in -1 1; do
    echo ""
    echo "---- direction=${DIRECTION} ----"
    ./${EXE_NAME} "$DIRECTION" "$DEVICE_ID" "$DATA_DIR"
    python3 "$SCRIPT_DIR/gen_data.py" "$DIRECTION" 42 "$DATA_DIR"
done

echo ""
echo "============================================"
echo " Test Complete (both directions)"
echo "============================================"
