#!/usr/bin/env python3
"""
Generate test data and compare precision for FFT R2C 2D operator.
Uses torch.fft.rfft2 as golden reference.

Usage:
    python3 gen_data.py [fftN] [batch]
    fftN: 4096 (default), 2048
    batch: 1 (default)
"""

import os
import sys
import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def generate_golden_torch(fftN, batch=1):
    np.random.seed(42)
    x = np.random.randn(batch, fftN, fftN).astype(np.float32)
    x_tensor = torch.from_numpy(x)
    golden = torch.fft.rfft2(x_tensor, norm='backward')
    return x, golden.numpy().astype(np.complex64)


def generate_golden_numpy(fftN, batch=1):
    np.random.seed(42)
    x = np.random.randn(batch, fftN, fftN).astype(np.float32)
    golden = np.fft.rfft2(x, axes=(-2, -1))
    return x, golden.astype(np.complex64)


def write_binary(filename, data):
    if np.iscomplexobj(data):
        flat = data.view(np.float32)
    else:
        flat = data.astype(np.float32).ravel()
    flat.tofile(filename)
    print(f"Written {filename}: {flat.size} floats ({flat.nbytes} bytes)")


def read_binary(filename, count):
    return np.fromfile(filename, dtype=np.float32, count=count)


def compare_precision(npu_output, golden, fftN, batch):
    totalElements = golden.size
    npu_complex = npu_output[0:2*totalElements:2] + 1j * npu_output[1:2*totalElements:2]
    npu_complex = npu_complex.reshape(golden.shape)
    abs_error = np.abs(npu_complex - golden)
    gold_mag = np.abs(golden)
    rel_error = np.where(gold_mag > 1e-8, abs_error / gold_mag, abs_error)
    mare = float(np.mean(rel_error))
    mere = float(np.mean(abs_error) / float(np.mean(gold_mag)))
    rmse = float(np.sqrt(np.mean(abs_error ** 2)))
    max_abs = float(np.max(abs_error))
    print(f"\n=== Precision Comparison (fftN={fftN}, batch={batch}) ===")
    print(f"Elements: {totalElements}")
    print(f"MARE: {mare:.6e}  MERE: {mere:.6e}  RMSE: {rmse:.6e}  MaxAbs: {max_abs:.6e}")
    passed = (mare < 1e-3) and (mere < 0.5) and (rmse < 0.5)
    print(f"Result: {'PASS' if passed else 'FAIL'}")

    print(f"\n=== Diagnosis ===")
    for r in range(min(4, golden.shape[1])):
        row = 0 if batch == 1 else 0
        print(f"  npu[{row},{r},0:4]: {npu_complex[row, r, 0:4]}")
        print(f"  golden[{row},{r},0:4]: {golden[row, r, 0:4]}")

    return passed


def main():
    fftN = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
    batch = int(sys.argv[2]) if len(sys.argv) > 2 else 1

    if fftN not in [2048, 4096]:
        print(f"Error: fftN must be 2048 or 4096, got {fftN}")
        sys.exit(1)

    print(f"Generating test data: fftN={fftN}, batch={batch}")
    if HAS_TORCH:
        print("Using torch.fft.rfft2 as golden reference...")
        input_data, golden = generate_golden_torch(fftN, batch)
    else:
        print("Using numpy.fft.rfft2 as golden reference...")
        input_data, golden = generate_golden_numpy(fftN, batch)

    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(data_dir, exist_ok=True)

    write_binary(os.path.join(data_dir, f"input_{fftN}_{batch}.bin"), input_data)
    write_binary(os.path.join(data_dir, f"golden_{fftN}_{batch}.bin"), golden)

    npu_output_file = os.path.join(data_dir, f"output_{fftN}_{batch}.bin")
    if os.path.exists(npu_output_file):
        print(f"\nFound NPU output: {npu_output_file}")
        total_floats = batch * fftN * (fftN // 2 + 1) * 2
        npu_output = read_binary(npu_output_file, total_floats)
        if len(npu_output) == total_floats:
            compare_precision(npu_output, golden, fftN, batch)
        else:
            print(f"Error: expected {total_floats} floats, got {len(npu_output)}")
    else:
        print(f"\nNPU output not found. Run the NPU executable first.")
        print(f"  bash scripts/build.sh -DCATLASS_ARCH=3510 85_ascend950_fft_r2c_2d --clean")
        print(f"  ./output/bin/85_ascend950_fft_r2c_2d {fftN} {batch} 0")


if __name__ == "__main__":
    main()
