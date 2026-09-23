#!/bin/bash
set -e
DEVICE_ID="${1:-0}"
SKIP_BUILD="${2:-0}"
FFT_N=4096
BATCH=1
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
EXE_NAME=fft_2d_r2c_4096x4096

echo "=== FFT R2C 2D [4096,4096] Perf Test (device=${DEVICE_ID}) ==="
if [ "$SKIP_BUILD" != "1" ]; then
    bash "$SCRIPT_DIR/build.sh"
fi
cd "$SCRIPT_DIR/build"
./${EXE_NAME} ${FFT_N} ${BATCH} ${DEVICE_ID} --perf