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


def compute_fft_c2c_3d_golden(input_data, direction):
    """Golden of 3D c2c fft: forward=fftn over last 3 axes.

    sip 的逆变换不归一化（输出 = unnormalized ifft，即 ifftn*N），与 1D c2c
    golden 的 norm='forward' 口径一致，inverse 侧用 norm='forward' 的 ifftn。
    """
    if direction == "forward":
        return np.fft.fftn(input_data, axes=(-3, -2, -1)).astype(np.complex64)
    return np.fft.ifftn(input_data, axes=(-3, -2, -1), norm="forward").astype(
        np.complex64
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizeX", type=int, required=True)
    parser.add_argument("--sizeY", type=int, required=True)
    parser.add_argument("--sizeZ", type=int, required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument(
        "--direction", type=str, required=True, choices=["forward", "inverse"]
    )
    parser.add_argument("--suffix", type=str, default="")
    args = parser.parse_args()

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(curr_dir, "complex64_input_0.bin")
    input_data = load_tensor_from_bin(input_file)
    input_data = input_data.reshape((args.batch, args.sizeX, args.sizeY, args.sizeZ))

    golden = compute_fft_c2c_3d_golden(input_data, args.direction)
    out_name = os.path.join(curr_dir, "complex64_golden_c2c_3d" + args.suffix + ".bin")
    golden.astype(np.complex64).tofile(out_name)
    print("golden saved to " + out_name + " (shape=" + str(golden.shape) + ")")


if __name__ == "__main__":
    main()
