#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
"""

import argparse
import os

import numpy as np


def load_tensor_from_bin(filename):
    """Read tensor saved by SaveTensorToBin: int32 dtype + int32 ndim + int64*ndim dims + data."""
    dtype_map = {
        0: "undefined",
        1: np.float32,
        2: np.float16,
        3: np.int32,
        4: np.int64,
        16: np.complex64,
    }
    with open(filename, "rb") as f:
        dtype = np.frombuffer(f.read(4), dtype=np.int32)[0]
        dim_count = np.frombuffer(f.read(4), dtype=np.int32)[0]
        dims = np.frombuffer(f.read(dim_count * 8), dtype=np.int64)
        data = np.frombuffer(f.read(), dtype=dtype_map[int(dtype)])
    return data.reshape(tuple(dims))


def compute_istft_golden(spec, window, nfft, hop):
    """Golden istft for center=True, normalized=False, onesided=False, returnComplex=True.

    等效性论证（适用边界）：本 golden 复刻 FFTCoreIstftAny::Run 的五步结构
      1/nFft 预归一化 -> ifft -> 乘窗 OLA -> 除 w^2 包络 -> 中心裁剪。
    其中除 np.fft.ifft 外的 OLA 偏移/裁剪区间/包络归一与被测实现同构，为缓解
    "golden 复刻 kernel bug 自证"的风险，main() 中另做一项独立往返自检
    （roundtrip_check，纯 numpy 语义、不经被测 kernel），确认本 OLA/裁剪/包络
    结构对标准 istft 语义（信号经等效 stft 后能量/波形可恢复）成立。

    window comes from the window bin saved by test_fft_istft.cpp so that both sides
    use bit-exact identical window values.
    """
    frames = np.fft.ifft(spec.astype(np.complex128), axis=1)
    frames_w = frames * window[None, :, None]

    batch, _, n_frames = spec.shape
    out_len_full = nfft + hop * (n_frames - 1)
    y = np.zeros((batch, out_len_full), dtype=np.complex128)
    wsum = np.zeros(out_len_full, dtype=np.float64)
    w2 = window * window
    for t in range(n_frames):
        y[:, t * hop : t * hop + nfft] += frames_w[:, :, t]
        wsum[t * hop : t * hop + nfft] += w2

    out = y / wsum[None, :]
    start = nfft // 2
    return out[:, start : start + hop * (n_frames - 1)]


def roundtrip_check(nfft, hop):
    """独立往返自检：不依赖被测 kernel，验证本 golden 的 OLA/裁剪/包络结构
    对标准 istft 语义成立——时域正弦信号的等效 stft 频谱经本 golden 流程后
    能恢复原波形（center=True 等效 stft：补零 nFft//2 两侧 + 全帧窗覆盖）。
    """
    # 自建同款恒正窗（与 test_fft_istft.cpp MakeIstftWindow 公式一致）
    k = np.arange(nfft, dtype=np.float64)
    win = 0.6 + 0.4 * np.sin(2.0 * np.pi * k / nfft + 0.7)

    rng = np.random.default_rng(42)
    sig_len = nfft + hop * 5
    t = np.arange(sig_len)
    signal = 0.5 * np.sin(2 * np.pi * 3.0 * t / nfft) + 0.3 * rng.standard_normal(
        sig_len
    )

    # center=True 等效 stft：两侧 pad nFft//2，逐 hop 取 nFft 帧加窗做 fft
    n_frames = 1 + sig_len // hop
    padded = np.pad(signal, (nfft // 2, nfft // 2))
    frames = np.stack(
        [padded[i * hop : i * hop + nfft] for i in range(n_frames)], axis=1
    )
    spec_rt = np.fft.fft(frames * win[:, None], axis=0)[None, :, :]  # (1, nFft, T)

    restored = compute_istft_golden(spec_rt, win, nfft, hop)[0]
    ref = signal[: restored.shape[0]]
    err = np.max(np.abs(restored - ref)) / (np.max(np.abs(ref)) + 1e-30)
    assert err < 1e-6, f"golden roundtrip self-check failed: rel_err={err}"
    print(f"golden roundtrip self-check PASSED (rel_err={err:.2e})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--nFft", type=int, required=True)
    parser.add_argument("--hop", type=int, required=True)
    parser.add_argument("--frames", type=int, required=True)
    parser.add_argument("--suffix", type=str, default="")
    args = parser.parse_args()

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(curr_dir, "complex64_input_0.bin")
    spec = load_tensor_from_bin(input_file)
    spec = spec.reshape((args.batch, args.nFft, args.frames))

    # window is generated and saved by test_fft_istft.cpp (bit-exact with the kernel)
    window = np.fromfile(
        os.path.join(curr_dir, "float_window" + args.suffix + ".bin"), dtype=np.float32
    )
    assert window.shape[0] == args.nFft, "window length mismatch"

    # 独立往返自检（不依赖被测 kernel），缓解 golden 复刻 kernel 语义的"自证"风险
    roundtrip_check(args.nFft, args.hop)

    golden = compute_istft_golden(spec, window.astype(np.float64), args.nFft, args.hop)
    out_name = os.path.join(curr_dir, "complex64_golden_istft" + args.suffix + ".bin")
    golden.astype(np.complex64).tofile(out_name)
    print("golden saved to " + out_name + " (shape=" + str(golden.shape) + ")")


if __name__ == "__main__":
    main()
