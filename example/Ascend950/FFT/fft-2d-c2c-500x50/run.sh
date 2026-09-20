#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

set -e

DEVICE_ID="${1:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$SCRIPT_DIR/data"
BIN="$SCRIPT_DIR/build/fft_2d_c2c_500x50"

# Locate and source the CANN set_env.sh (needed for LD_LIBRARY_PATH when running).
if [ -z "${ASCEND_HOME_PATH:-}" ]; then
    for cand in \
        "${HOME}/cann/ascend-toolkit/set_env.sh" \
        "/usr/local/Ascend/ascend-toolkit/set_env.sh" \
        "/usr/local/Ascend/latest/set_env.sh"; do
        if [ -f "$cand" ]; then
            # shellcheck disable=SC1090
            source "$cand"
            break
        fi
    done
fi
if [ -z "${ASCEND_HOME_PATH:-}" ]; then
    echo "[ERROR] ASCEND_HOME_PATH is not set. Please source set_env.sh first." >&2
    exit 1
fi

bash "$SCRIPT_DIR/build.sh"

mkdir -p "$DATA_DIR"

# Generate input + golden (numpy.fft.fft2, seed 42) for both directions.
python3 "$SCRIPT_DIR/gen_data.py" 500 -1 12000 50
python3 "$SCRIPT_DIR/gen_data.py" 500 1 12000 50

echo ""
echo "===== direction = -1 (forward) ====="
"$BIN" 500 -1 "$DEVICE_ID" "$DATA_DIR"

echo ""
echo "===== direction = +1 (inverse, unnormalized) ====="
"$BIN" 500 1 "$DEVICE_ID" "$DATA_DIR"

echo ""
echo "===== precision gate (MARE < 1e-4) ====="
python3 "$SCRIPT_DIR/gen_data.py" 500 -1 12000 50
python3 "$SCRIPT_DIR/gen_data.py" 500 1 12000 50
