#!/bin/bash
set -e
DEVICE_ID="${1:-0}"
SKIP_BUILD="${2:-0}"
FFT_N=4096
BATCH=1
STAGE=2
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DATA_DIR="$SCRIPT_DIR/data"
EXE_NAME=fft_2d_c2r_4096x4096

echo "=== FFT C2R 2D [4096,4096] Precision Test (device=${DEVICE_ID}) ==="
mkdir -p "$DATA_DIR"
python3 "$SCRIPT_DIR/gen_data.py" ${FFT_N} ${BATCH} --stage ${STAGE}
if [ "$SKIP_BUILD" != "1" ]; then
    bash "$SCRIPT_DIR/build.sh"
fi
cd "$SCRIPT_DIR/build"
./${EXE_NAME} ${FFT_N} ${BATCH} ${DEVICE_ID} "$DATA_DIR" --stage ${STAGE}
cd "$SCRIPT_DIR"
python3 "$SCRIPT_DIR/gen_data.py" ${FFT_N} ${BATCH} --stage ${STAGE}