#!/usr/bin/env python3
"""
Generate test data and compare precision for FFT C2R 2D operator.
Uses torch.fft.irfft2 as golden reference (norm="forward").

Input:  [batch, fftN, fftN//2+1] complex64 (half-spectrum)
Output: [batch, fftN, fftN]      float32  (real)

Usage:
    python3 gen_data.py [fftN] [batch] [--stage N]

    fftN: 4096 (default), 2048
    batch: 1 (default)
    --stage N: verify a specific kernel stage
        0: K1 only (Hermitian expansion + Row IDFT) → ws0 complex output
        1: K1 + K2 (Hermitian + IDFT + Transpose) → ws1 complex output
        2: full pipeline (default) → real output
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
    X_freq = torch.fft.rfft2(x_tensor, norm='forward')
    golden = torch.fft.irfft2(X_freq, s=(fftN, fftN), norm='forward')
    return X_freq.numpy().astype(np.complex64), golden.numpy().astype(np.float32), x


def generate_golden_numpy(fftN, batch=1):
    np.random.seed(42)
    x = np.random.randn(batch, fftN, fftN).astype(np.float32)
    X_freq = np.fft.rfft2(x, axes=(-2, -1), norm='forward').astype(np.complex64)
    golden = np.fft.irfft2(X_freq, s=(fftN, fftN), norm='forward').astype(np.float32)
    return X_freq, golden, x


def hermitian_expand(X_freq, fftN):
    """Stage 0 golden: Hermitian expand half-spectrum to full complex.

    Input:  [batch, fftN, halfN] complex64
    Output: [batch, fftN, fftN] complex128 (full spectrum)
    """
    batch, M, halfN = X_freq.shape
    X_full = np.zeros((batch, M, fftN), dtype=np.complex128)
    X_full[:, :, :halfN] = X_freq

    u_sym = (M - np.arange(M)) % M
    v_right = np.arange(halfN, fftN)
    v_left = fftN - v_right
    for b in range(batch):
        # Index batch first to keep 2D semantics (avoids numpy mixed-indexing axis reorder)
        X_full[b][:, v_right] = np.conj(X_freq[b][u_sym[:, None], v_left[None, :]])

    return X_full


# Radix-64 1D IDFT (matches numpy_prototype.py idft1d_radix64)
N1_64 = 64
N2_64 = 64

def build_w_matrix(N, sign=1.0):
    """Build N-point IDFT matrix in separated real/imag format (2N, 2N)."""
    dim = 2 * N
    W = np.zeros((dim, dim), dtype=np.float64)
    for k in range(N):
        for n in range(N):
            angle = 2.0 * np.pi * k * n / N
            cos_val = np.cos(angle)
            sin_val = np.sin(angle) * sign
            W[k, n] = cos_val
            W[k, n + N] = -sin_val
            W[k + N, n] = sin_val
            W[k + N, n + N] = cos_val
    return W

def build_t_matrix(N1, N2, total_N, sign=1.0):
    """Build inter-stage twiddle (2*N1, N2)."""
    T = np.zeros((2 * N1, N2), dtype=np.float64)
    for k1 in range(N1):
        for n2 in range(N2):
            angle = 2.0 * np.pi * k1 * n2 / total_N
            T[k1, n2] = np.cos(angle)
            T[k1 + N1, n2] = np.sin(angle) * sign
    return T

W64_gold = build_w_matrix(64)
T64_gold = build_t_matrix(64, 64, 4096)

def idft1d_radix64(mat, axis=0):
    """1D IDFT via radix-64 GEMM — matches NPU kernel exactly."""
    if axis == 1:
        return idft1d_radix64(mat.T, axis=0).T

    L, K = mat.shape
    N1, N2 = N1_64, N2_64

    X = mat.reshape(N1, N2, K)

    X_real = X.real.reshape(N1, N2 * K)
    X_imag = X.imag.reshape(N1, N2 * K)
    X_sep = np.vstack([X_real, X_imag])

    C0 = W64_gold @ X_sep

    C0_real = C0[:N1, :].reshape(N1, N2, K)
    C0_imag = C0[N1:, :].reshape(N1, N2, K)

    T_real = T64_gold[:N1, :]
    T_imag = T64_gold[N1:, :]

    B_real = C0_real * T_real[:, :, None] - C0_imag * T_imag[:, :, None]
    B_imag = C0_real * T_imag[:, :, None] + C0_imag * T_real[:, :, None]

    B_swap_real = B_real.transpose(1, 0, 2).reshape(N2, N1 * K)
    B_swap_imag = B_imag.transpose(1, 0, 2).reshape(N2, N1 * K)
    B_swap_sep = np.vstack([B_swap_real, B_swap_imag])

    C1 = W64_gold @ B_swap_sep

    C1_real = C1[:N2, :].reshape(N2, N1, K)
    C1_imag = C1[N2:, :].reshape(N2, N1, K)
    C1_complex = C1_real + 1j * C1_imag

    output = C1_complex.reshape(N2 * N1, K)
    return output


def kernel1_golden(X_freq, fftN):
    """Stage 0 golden: Hermitian expand + Row IDFT (axis 1).

    Input:  [batch, fftN, halfN] complex64
    Output: [batch, fftN, fftN] complex (IDFT result)
    """
    batch, M, halfN = X_freq.shape
    X_full = hermitian_expand(X_freq, fftN)

    Y = np.zeros((batch, M, fftN), dtype=np.complex128)
    for b in range(batch):
        Y[b] = idft1d_radix64(X_full[b], axis=1)

    return Y


def write_binary(filename, data):
    if np.iscomplexobj(data):
        flat = data.view(np.float32)
    else:
        flat = data.astype(np.float32).ravel()
    flat.tofile(filename)
    print(f"Written {filename}: {flat.size} floats ({flat.nbytes} bytes)")


def read_binary(filename, count):
    return np.fromfile(filename, dtype=np.float32, count=count)


def compare_real(npu_output, golden, fftN, batch, label=""):
    """Compare real-valued NPU output against golden."""
    totalElements = golden.size
    npu_real = npu_output[0:totalElements].reshape(golden.shape)

    abs_error = np.abs(npu_real - golden)
    gold_mag = np.abs(golden)
    rel_error = np.where(gold_mag > 1e-8, abs_error / gold_mag, abs_error)

    mare = float(np.mean(rel_error))
    mere = float(np.mean(abs_error) / float(np.mean(gold_mag)))
    rmse = float(np.sqrt(np.mean(abs_error ** 2)))
    max_abs = float(np.max(abs_error))

    print(f"\n=== Precision Comparison {label} (fftN={fftN}, batch={batch}) ===")
    print(f"Elements: {totalElements}")
    print(f"MARE: {mare:.6e}  MERE: {mere:.6e}  RMSE: {rmse:.6e}  MaxAbs: {max_abs:.6e}")

    passed = (mare < 1e-3) and (mere < 0.5) and (rmse < 0.5)
    print(f"Result: {'PASS' if passed else 'FAIL'}")

    print(f"\n=== Diagnosis ===")
    for r in range(min(4, golden.shape[1])):
        row = 0 if batch == 1 else 0
        print(f"  npu[{row},{r},0:8]:   {npu_real[row, r, 0:8]}")
        print(f"  golden[{row},{r},0:8]: {golden[row, r, 0:8]}")

    return passed


def compare_complex(npu_output_floats, golden_complex, fftN, batch, label=""):
    """Compare complex-valued NPU output (interleaved floats) against golden."""
    totalComplex = golden_complex.size
    npu_complex = npu_output_floats[0:2*totalComplex:2] + 1j * npu_output_floats[1:2*totalComplex:2]
    npu_complex = npu_complex.reshape(golden_complex.shape)

    abs_error = np.abs(npu_complex - golden_complex)
    gold_mag = np.abs(golden_complex)
    rel_error = np.where(gold_mag > 1e-8, abs_error / gold_mag, abs_error)

    mare = float(np.mean(rel_error))
    mere = float(np.mean(abs_error) / float(np.mean(gold_mag) + 1e-12))
    rmse = float(np.sqrt(np.mean(abs_error ** 2)))
    max_abs = float(np.max(abs_error))

    print(f"\n=== Precision Comparison {label} (fftN={fftN}, batch={batch}) ===")
    print(f"Complex elements: {totalComplex}")
    print(f"MARE: {mare:.6e}  MERE: {mere:.6e}  RMSE: {rmse:.6e}  MaxAbs: {max_abs:.6e}")

    passed = (mare < 1e-3) and (mere < 0.5) and (rmse < 0.5)
    print(f"Result: {'PASS' if passed else 'FAIL'}")

    print(f"\n=== Diagnosis ===")
    for r in range(min(4, golden_complex.shape[1])):
        row = 0 if batch == 1 else 0
        print(f"  npu[{row},{r},0:4]:   {npu_complex[row, r, 0:4]}")
        print(f"  golden[{row},{r},0:4]: {golden_complex[row, r, 0:4]}")

    return passed


def main():
    # Parse --stage
    stage = 2  # default: full pipeline
    pos_args = []
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--stage" and i + 1 < len(args):
            stage = int(args[i + 1])
            i += 2
        else:
            pos_args.append(args[i])
            i += 1

    fftN = int(pos_args[0]) if len(pos_args) > 0 else 4096
    batch = int(pos_args[1]) if len(pos_args) > 1 else 1

    if fftN not in [2048, 4096]:
        print(f"Error: fftN must be 2048 or 4096, got {fftN}")
        sys.exit(1)

    print(f"Generating test data: fftN={fftN}, batch={batch}, stage={stage}")
    if HAS_TORCH:
        print("Using torch.fft as golden reference...")
        X_freq, golden, x_orig = generate_golden_torch(fftN, batch)
    else:
        print("Using numpy.fft as golden reference...")
        X_freq, golden, x_orig = generate_golden_numpy(fftN, batch)

    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(data_dir, exist_ok=True)

    write_binary(os.path.join(data_dir, f"input_{fftN}_{batch}.bin"), X_freq)

    if stage == 2:
        write_binary(os.path.join(data_dir, f"golden_{fftN}_{batch}.bin"), golden)
    elif stage == 0:
        # Stage 0 golden: Hermitian expand + Row IDFT + fused transpose
        # K1 now scatter-writes transposed output directly to ws1.
        Y_k1 = kernel1_golden(X_freq, fftN)
        Y_transposed = np.ascontiguousarray(np.transpose(Y_k1, (0, 2, 1)))
        write_binary(os.path.join(data_dir, f"golden_{fftN}_{batch}.bin"), Y_transposed)
        print(f"  Stage 0 golden: Hermitian + Row IDFT + transpose [batch, {fftN}, {fftN}] complex")
    elif stage == 1:
        # Stage 1 golden: Hermitian expand + Row IDFT + Transpose
        Y_k1 = kernel1_golden(X_freq, fftN)
        Y_transposed = np.ascontiguousarray(np.transpose(Y_k1, (0, 2, 1)))
        write_binary(os.path.join(data_dir, f"golden_{fftN}_{batch}.bin"), Y_transposed)
        print(f"  Stage 1 golden: Hermitian + IDFT + transpose [batch, {fftN}, {fftN}] complex")

    npu_output_file = os.path.join(data_dir, f"output_{fftN}_{batch}.bin")
    if os.path.exists(npu_output_file):
        print(f"\nFound NPU output: {npu_output_file}")
        if stage == 2:
            total_floats = batch * fftN * fftN
            npu_output = read_binary(npu_output_file, total_floats)
            if len(npu_output) == total_floats:
                compare_real(npu_output, golden, fftN, batch, label="(full pipeline)")
            else:
                print(f"Error: expected {total_floats} floats, got {len(npu_output)}")
        elif stage == 0:
            total_floats = batch * fftN * fftN * 2
            npu_output = read_binary(npu_output_file, total_floats)
            Y_k1 = kernel1_golden(X_freq, fftN)
            Y_transposed = np.ascontiguousarray(np.transpose(Y_k1, (0, 2, 1)))
            if len(npu_output) == total_floats:
                compare_complex(npu_output, Y_transposed, fftN, batch, label="(K1: Hermitian+IDFT+transpose)")
            else:
                print(f"Error: expected {total_floats} floats, got {len(npu_output)}")
        elif stage == 1:
            total_floats = batch * fftN * fftN * 2
            npu_output = read_binary(npu_output_file, total_floats)
            Y_k1 = kernel1_golden(X_freq, fftN)
            Y_transposed = np.ascontiguousarray(np.transpose(Y_k1, (0, 2, 1)))
            if len(npu_output) == total_floats:
                compare_complex(npu_output, Y_transposed, fftN, batch, label="(K1: Hermitian+IDFT+transpose)")
            else:
                print(f"Error: expected {total_floats} floats, got {len(npu_output)}")
    else:
        print(f"\nNPU output not found. Run the NPU executable first:")
        print(f"  bash scripts/build.sh -DCATLASS_ARCH=3510 86_ascend950_fft_c2r_2d --clean")
        if stage == 0:
            print(f"  ./output/bin/86_ascend950_fft_c2r_2d {fftN} {batch} 0 --stage 0")
        elif stage == 1:
            print(f"  ./output/bin/86_ascend950_fft_c2r_2d {fftN} {batch} 0 --stage 1")
        else:
            print(f"  ./output/bin/86_ascend950_fft_c2r_2d {fftN} {batch} 0")


if __name__ == "__main__":
    main()
