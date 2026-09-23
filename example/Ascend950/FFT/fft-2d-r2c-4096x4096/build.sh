#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"

if [ -z "${ASCEND_HOME_PATH:-}" ]; then
    for cand in \
        "${HOME}/cann/ascend-toolkit/set_env.sh" \
        "/usr/local/Ascend/ascend-toolkit/set_env.sh" \
        "/usr/local/Ascend/latest/set_env.sh"; do
        if [ -f "$cand" ]; then
            source "$cand"
            break
        fi
    done
fi

if [ -z "${ASCEND_HOME_PATH:-}" ]; then
    echo "[ERROR] ASCEND_HOME_PATH is not set. Please source set_env.sh first." >&2
    exit 1
fi

cmake -S "$SCRIPT_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD_DIR" -j

echo "[INFO] Built: $BUILD_DIR/fft_2d_r2c_4096x4096"