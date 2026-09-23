#!/bin/bash
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