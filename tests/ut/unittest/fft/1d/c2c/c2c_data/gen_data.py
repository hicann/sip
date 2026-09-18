#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import sys
import os
import numpy as np
import argparse

curr_dir = os.path.dirname(os.path.realpath(__file__))


def load_tensor_from_bin(filename):
    dtype_map = {
        0: "undefined",
        1: np.float32,
        2: np.float16,
        3: np.int32,
        4: np.int64,
        16: np.complex64,
        6: np.uint8,
    }

    with open(filename, "rb") as f:
        dtype_int = np.frombuffer(f.read(4), dtype=np.int32)[0]
        np_type = dtype_map.get(dtype_int, np.float32)
        dim_count = np.frombuffer(f.read(4), dtype=np.int32)[0]
        dims = np.frombuffer(f.read(8 * dim_count), dtype=np.int64)
        data = np.frombuffer(f.read(), dtype=np_type)
        data = data.reshape(dims)
        return data, dims, np_type


def compute_fft_c2c_golden(input_data, nfft, direction):
    if direction == "forward":
        golden = np.fft.fft(input_data, n=nfft)
    else:
        golden = np.fft.ifft(input_data, n=nfft, norm="forward")
    return golden.astype(np.complex64)


def generate_golden_data(batch, nfft, direction):
    input_filename = curr_dir + "/complex64_input_0.bin"

    if not os.path.exists(input_filename):
        print(f"Input file not found: {input_filename}")
        sys.exit(1)

    input_data, dims, dtype = load_tensor_from_bin(input_filename)
    print(f"Loaded input: shape={dims}, dtype={dtype}")
    print(f"Input data shape: {input_data.shape}")

    golden_data = np.zeros((batch, nfft), dtype=np.complex64)

    for b in range(batch):
        batch_input = input_data[b, :]
        golden_data[b, :] = compute_fft_c2c_golden(batch_input, nfft, direction)

    suffix = "" if direction == "forward" else "_inv"
    golden_filename = curr_dir + f"/complex64_golden_c2c{suffix}.bin"
    golden_data.astype(np.complex64).tofile(golden_filename)
    print(f"Golden data saved: {golden_filename}, shape={golden_data.shape}")

    input_simple_filename = curr_dir + "/complex64_input_simple.bin"
    input_data.astype(np.complex64).tofile(input_simple_filename)

    return golden_data


def main():
    parser = argparse.ArgumentParser(
        description="Generate golden data for FFT C2C test"
    )
    parser.add_argument("--batch", type=int, default=1, help="Batch size")
    parser.add_argument("--nfft", type=int, default=16, help="FFT size")
    parser.add_argument(
        "--direction", type=str, default="forward", help="forward or inverse"
    )
    args = parser.parse_args()

    print(
        f"Generating golden data: batch={args.batch}, nfft={args.nfft}, direction={args.direction}"
    )
    golden_data = generate_golden_data(args.batch, args.nfft, args.direction)
    print(f"golden_data = {golden_data.flatten()[:10]}")
    print("Done!")


if __name__ == "__main__":
    main()
