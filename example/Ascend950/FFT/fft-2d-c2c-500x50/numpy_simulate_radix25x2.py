#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""P0 仿真：radix-25x2 行 FFT（Formulation A，M=500 视角）。

验证「偶数/奇数 planar GEMM + radix-2」与 np.fft.fft(50) 一致，并给出
W25 的 planar 系数布局（供 device GEMM 使用）。

  50 = 25 x 2（Cooley-Tukey DIT）:
    x_even[n] = x[2n], x_odd[n] = x[2n+1]
    F_even = DFT_25(x_even), F_odd = DFT_25(x_odd)
    X[k]   = F_even[k] + W_50^k * F_odd[k]
    X[k+25]= F_even[k] - W_50^k * F_odd[k]
"""

import numpy as np

N = 50
N1 = 25


def build_w25_planar(direction):
    """25 点 DFT 矩阵的 planar [50,50] 形式。

    planar 复数 (re0..re24, im0..im24)，矩阵分块:
      [Re  -dir*Im]    前 25 行 = 实数输出分量
      [dir*Im  Re]     后 25 行 = 虚数输出分量
    """
    w = np.zeros((50, 50), dtype=np.float64)
    for f in range(N1):
        for t in range(N1):
            th = 2 * np.pi * f * t / N1
            c = np.cos(th)
            s = np.sin(th)
            w[f, t] = c
            w[f, N1 + t] = -direction * s
            w[N1 + f, t] = direction * s
            w[N1 + f, N1 + t] = c
    return w


def build_twiddle(direction):
    k = np.arange(N1)
    th = 2 * np.pi * k / N
    return np.cos(th) + 1j * direction * np.sin(th)


def row_fft_radix25x2(x, direction):
    """x: [..., 50] complex -> 50 点 FFT（未归一化）。"""
    x_even = x[..., 0::2]  # [..., 25]
    x_odd = x[..., 1::2]  # [..., 25]

    def planar(cx):
        return np.concatenate([cx.real, cx.imag], axis=-1)  # [..., 50]

    def unplanar(p):
        h = p.shape[-1] // 2
        return p[..., :h] + 1j * p[..., h:]

    # GEMM: F = planar(x) @ W^T（行主序 [..., 50] @ [50, 50]）
    Fe = unplanar(planar(x_even) @ build_w25_planar(direction).T)
    Fo = unplanar(planar(x_odd) @ build_w25_planar(direction).T)

    tw = build_twiddle(direction)  # [25]
    p = tw * Fo
    Y = np.empty_like(x)
    Y[..., 0:25] = Fe + p
    Y[..., 25:50] = Fe - p
    return Y


def golden(x, direction):
    if direction == -1:
        return np.fft.fft(x, n=N, axis=-1)
    return np.fft.ifft(x, n=N, axis=-1, norm="forward")


def mare(y, g):
    return float(np.mean(np.abs(y - g) / np.maximum(np.abs(g), 1e-6)))


def main():
    rng = np.random.default_rng(0)
    for direction in (-1, 1):
        x = (rng.standard_normal((4, N)) + 1j * rng.standard_normal((4, N))).astype(
            np.complex64
        )
        y = row_fft_radix25x2(x, direction)
        g = golden(x, direction)
        m = mare(y, g)
        print(
            f"direction={direction:+d}  MARE={m:.3e}  {'PASS' if m < 1e-6 else 'FAIL'}"
        )


if __name__ == "__main__":
    main()
