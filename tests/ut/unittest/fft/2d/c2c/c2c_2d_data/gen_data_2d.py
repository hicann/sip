#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import argparse
import os
import sys
import numpy as np

curr_dir = os.path.dirname(os.path.realpath(__file__))


def load_tensor_from_bin(filename):
    dtype_map = {
        0: "undefined",
        1: np.float32,
        2: np.float16,
        3: np.int32,
        4: np.int64,
        16: np.complex64,
    }
    with open(filename, "rb") as f:
        dtype_int = np.frombuffer(f.read(4), dtype=np.int32)[0]
        np_type = dtype_map.get(dtype_int, np.float32)
        dim_count = np.frombuffer(f.read(4), dtype=np.int32)[0]
        dims = np.frombuffer(f.read(8 * dim_count), dtype=np.int64)
        data = np.frombuffer(f.read(), dtype=np_type)
        data = data.reshape(dims)
        return data


def main():
    parser = argparse.ArgumentParser(
        description="Generate golden data for FFT 2D C2C test"
    )
    parser.add_argument("--sizeX", type=int, required=True)
    parser.add_argument("--sizeY", type=int, required=True)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--direction", type=str, default="forward")
    parser.add_argument(
        "--suffix",
        type=str,
        default="",
        help="golden 文件名后缀，区分 batch 等不同场景",
    )
    args = parser.parse_args()

    input_file = curr_dir + "/complex64_input_0.bin"
    if not os.path.exists(input_file):
        print(f"Input file not found: {input_file}")
        sys.exit(1)

    input_data = load_tensor_from_bin(input_file)
    input_data = input_data.reshape(args.batch, args.sizeX, args.sizeY)

    if args.direction == "forward":
        golden = np.fft.fft2(input_data, s=(args.sizeX, args.sizeY))
    else:
        golden = np.fft.ifft2(input_data, s=(args.sizeX, args.sizeY), norm="forward")
    golden = golden.astype(np.complex64)

    golden_file = curr_dir + "/complex64_golden_c2c_2d" + args.suffix + ".bin"
    golden.tofile(golden_file)
    print(f"Golden saved: {golden_file}, shape={golden.shape}")


if __name__ == "__main__":
    main()
