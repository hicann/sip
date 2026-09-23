#!/bin/bash
set -e

DEVICE_ID="${1:-3}"
SKIP_BUILD="${2:-0}"
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
EXE_NAME=fft_2d_c2c_4096x4096

echo "============================================"
echo " FFT C2C 2D [4096,4096] (FFT2D) Perf Test"
echo " device = ${DEVICE_ID}  (10 warmup + 10 measured)"
echo "============================================"

if [ "$SKIP_BUILD" != "1" ]; then
    echo ""
    echo "[1/2] Building operator..."
    bash "$SCRIPT_DIR/build.sh"
fi

echo ""
echo "[2/2] Running performance test (forward + inverse)..."
cd "$SCRIPT_DIR/build"
for DIRECTION in -1 1; do
    ./${EXE_NAME} "$DIRECTION" "$DEVICE_ID" "$SCRIPT_DIR/data" --perf
done