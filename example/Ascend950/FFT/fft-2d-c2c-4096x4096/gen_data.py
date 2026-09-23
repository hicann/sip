#!/usr/bin/env python3
"""Generate test data and compare precision for example 101.

覆盖两个模式（dim）：
  dim=1（axis=0 纵向 FFT）：
    forward  (direction=-1): np.fft.fft(x, axis=0)
    inverse  (direction=+1): np.fft.ifft(x, axis=0, norm="forward")   （不归一化）
  dim=2（FFT2D 双向）：
    forward  (direction=-1): np.fft.fft2(x)
    inverse  (direction=+1): np.fft.ifft2(x, norm="forward")          （不归一化）

引擎归一化约定：norm="forward"——正/反变换均不归一化（方向仅影响指数符号）。

Data files (relative to <data_dir>, default ./data):
  input_2d_4096x4096_axis0_<direction>.bin     （dim=1）
  input_2d_4096x4096_fft2_<direction>.bin      （dim=2）
  golden_*/output_* 同名。

Usage:
  python3 gen_data.py <dim> <direction> [seed] [data_dir]
"""

from __future__ import annotations

import os
import sys

import numpy as np

ROWS = COLS = 4096
MARE_GATE = 2e-5


def gen_input(rows: int, cols: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((rows, cols)) + 1j * rng.standard_normal((rows, cols))
    return x.astype(np.complex64)


def fft_golden(x: np.ndarray, direction: int) -> np.ndarray:
    x64 = x.astype(np.complex128)
    # 与引擎一致：norm="forward"（不归一化）。
    y = np.fft.fft2(x64) if direction == -1 else np.fft.ifft2(x64, norm="forward")
    return y.astype(np.complex64)


def write_bin(path: str, arr: np.ndarray):
    arr = np.ascontiguousarray(arr, dtype=np.complex64)
    flat = arr.view(np.float32)
    flat.tofile(path)
    print(f"  written: {path}  ({flat.size} floats, {flat.nbytes / 1024 / 1024:.1f} MB)")


def compare(got: np.ndarray, golden: np.ndarray) -> bool:
    abs_err = np.abs(got.astype(np.complex128) - golden.astype(np.complex128))
    denom = np.abs(golden.astype(np.complex128)) + 1e-12
    mare = float(np.mean(abs_err / denom))
    max_abs = float(np.max(abs_err))

    print("\n=== Precision Comparison ===")
    print(f"  elements compared: {golden.size}")
    print(f"  MARE (mean absolute relative error): {mare:.6e}")
    print(f"  max absolute error:                 {max_abs:.6e}")

    passed = mare < MARE_GATE
    print(f"  Compare: {'PASS' if passed else 'FAIL'} (gate MARE < {MARE_GATE:g})")
    return passed


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2

    direction = int(sys.argv[1])
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    data_dir = sys.argv[3] if len(sys.argv) > 3 else "data"

    if direction not in (-1, 1):
        print(f"Error: direction must be -1 or 1, got {direction}")
        return 2

    print(f"Generating data: [{ROWS},{COLS}] FFT2D, direction={direction}, seed={seed}")

    x = gen_input(ROWS, COLS, seed)
    golden = fft_golden(x, direction)

    os.makedirs(data_dir, exist_ok=True)
    tag = f"2d_{ROWS}x{COLS}_fft2_{direction}"

    input_file = os.path.join(data_dir, f"input_{tag}.bin")
    golden_file = os.path.join(data_dir, f"golden_{tag}.bin")
    write_bin(input_file, x)
    write_bin(golden_file, golden)

    output_file = os.path.join(data_dir, f"output_{tag}.bin")
    if not os.path.exists(output_file):
        print(f"\nNPU output not found: {output_file}")
        print("Run the NPU executable first, then re-run this script to compare.")
        return 0

    print(f"\nFound NPU output: {output_file}")
    flat = np.fromfile(output_file, dtype=np.float32)
    expected_floats = ROWS * COLS * 2
    if flat.size != expected_floats:
        print(f"Error: expected {expected_floats} floats, got {flat.size}")
        return 1

    got = (flat[0::2] + 1j * flat[1::2]).reshape(ROWS, COLS).astype(np.complex64)
    return 0 if compare(got, golden) else 1


if __name__ == "__main__":
    sys.exit(main())