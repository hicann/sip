#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Generate test data and compare precision for the fused FFT C2C 2D operator
(example 90).

Golden reference: numpy.fft.fft2 (axes=(1,2) of [b1, fftN1, fftN2]).
The NPU computes the same 2D DFT with a fused AIC+AIV kernel: the AIC runs
the row FFT (fftN2=50) as an embedded-rotation GEMM whose Fixpipe L0C->UB
egress lands the [500, 104] tile in the owner AIV's UB; the AIV runs the
column FFT (fftN1=500) on that tile in-place (radices 4,5,5,5 + digit-
reversal folded into the GM scatter).

Input shape: [b1=12000, fftN1=500, fftN2=50] complex64
             (stored as float32 interleaved [re, im])
Output shape: same — 2D FFT over axes (1, 2).

Golden:
  forward  (direction=-1): np.fft.fft2(x, axes=(1, 2))
  inverse  (direction=+1): np.fft.ifft2(x, axes=(1, 2), norm='forward')
                            (unnormalized — no division by N)

Usage:
    python3 gen_data.py [fftN1] [direction] [b1] [fftN2]

After an NPU run, re-run this script to compare output_*.bin against golden.
"""

import os
import sys
import numpy as np


def generate_golden(b1, fftN1, fftN2, direction):
    """Generate random complex input and the golden 2D FFT."""
    np.random.seed(42)
    x = np.random.randn(b1, fftN1, fftN2).astype(np.float32) + 1j * np.random.randn(
        b1, fftN1, fftN2
    ).astype(np.float32)

    if direction == -1:
        golden = np.fft.fft2(x, axes=(1, 2))
    else:
        # Unnormalized inverse
        golden = np.fft.ifft2(x, axes=(1, 2), norm="forward")

    return x.astype(np.complex64), golden.astype(np.complex64)


def write_binary(filename, data):
    """Write complex64 data as interleaved float32 binary (C-contiguous)."""
    data = np.ascontiguousarray(data, dtype=np.complex64)
    flat = data.view(np.float32)
    flat.tofile(filename)
    print(
        f"Written {filename}: {flat.size} floats ({flat.nbytes / 1024 / 1024:.1f} MB)"
    )


def compare_precision(npu_output, golden, b1, fftN1, fftN2, direction):
    """Compare NPU output with golden reference (numpy.fft.fft2 axes=(1,2))."""
    total_elements = golden.size
    npu_complex = (
        npu_output[0 : 2 * total_elements : 2]
        + 1j * npu_output[1 : 2 * total_elements : 2]
    )
    npu_complex = npu_complex.reshape(golden.shape)

    abs_error = np.abs(npu_complex - golden)
    golden_mag = np.abs(golden)
    rel_error = np.where(
        golden_mag > 1e-6, abs_error / np.maximum(golden_mag, 1e-30), abs_error
    )

    mare = float(np.mean(abs_error))
    mere = float(np.mean(rel_error))
    max_abs = float(np.max(abs_error))
    rmse = float(np.sqrt(np.mean(abs_error**2)))

    print(f"  Elements: {total_elements}")
    print(f"  MARE (mean abs err): {mare:.6e}")
    print(f"  MERE (mean rel err): {mere:.6e}")
    print(f"  RMSE: {rmse:.6e}")
    print(f"  MaxAbsErr: {max_abs:.6e}")

    # Gate: MARE < 1e-4 (both directions; expected ~1e-6)
    passed = mare < 1e-4
    print(f"  Gate MARE < 1e-4: {'PASS' if passed else 'FAIL'}")
    return passed


def main():
    fftN1 = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    direction = int(sys.argv[2]) if len(sys.argv) > 2 else -1
    b1 = int(sys.argv[3]) if len(sys.argv) > 3 else 12000
    fftN2 = int(sys.argv[4]) if len(sys.argv) > 4 else 50

    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(data_dir, exist_ok=True)

    tag = f"{b1}x{fftN1}x{fftN2}_{direction}"
    input_path = os.path.join(data_dir, f"input_{tag}.bin")
    golden_path = os.path.join(data_dir, f"golden_{tag}.bin")
    output_path = os.path.join(data_dir, f"output_{tag}.bin")

    # Golden always regenerated (it is deterministic: seed 42).
    x, golden = generate_golden(b1, fftN1, fftN2, direction)
    write_binary(input_path, x)
    write_binary(golden_path, golden)

    if os.path.exists(output_path):
        npu_output = np.fromfile(output_path, dtype=np.float32)
        expected_floats = 2 * golden.size
        if npu_output.size != expected_floats:
            print(
                f"Output size mismatch: got {npu_output.size} floats, "
                f"expected {expected_floats}"
            )
            return 1
        print(f"Comparing NPU output against golden (direction={direction}):")
        passed = compare_precision(npu_output, golden, b1, fftN1, fftN2, direction)
        return 0 if passed else 1
    else:
        print(
            f"(output_{tag}.bin not found — data generated, run the NPU binary first)"
        )
        return 0


if __name__ == "__main__":
    sys.exit(main())
