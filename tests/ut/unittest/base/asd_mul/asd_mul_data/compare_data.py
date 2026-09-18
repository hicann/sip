#!/usr/bin/env python3
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
import numpy as np
import glob
import os
import argparse

curr_dir = os.path.dirname(os.path.realpath(__file__))


def compare_data(dtype):
    if dtype == "c64":
        precision = 0.001
        golden_files = sorted(
            glob.glob(os.path.join(curr_dir, "complex64*golden*.bin"))
        )
    elif dtype == "c32":
        precision = 0.01
        golden_files = sorted(
            glob.glob(os.path.join(curr_dir, "complex32*golden*.bin"))
        )
    else:
        raise ValueError(f"Unknown dtype: {dtype}")

    # 按 dtype 过滤 output 文件：同目录多用例（c32/c64）共享，"*.bin" 全量 glob 会把
    # 其它用例残留的 output 也配进来（sorted-zip 错配，报 size mismatch），必须限定前缀
    if dtype == "c64":
        output_files = sorted(
            glob.glob(os.path.join(curr_dir, "complex64*output*.bin"))
        )
    else:
        output_files = sorted(
            glob.glob(os.path.join(curr_dir, "complex32*output*.bin"))
        )

    if not golden_files or not output_files:
        print(f"No golden or output files found for dtype={dtype}")
        sys.exit(1)

    data_same = True
    for gold, out in zip(golden_files, output_files):
        # golden 文件是纯数据，无 header
        if dtype == "c64":
            tmp_gold = np.fromfile(gold, np.complex64)
            # output 文件可能有 header，尝试读取
            with open(out, "rb") as f:
                dtype_int = np.frombuffer(f.read(4), dtype=np.int32)[0]
                if dtype_int == 16:  # 有 header
                    dim_count = np.frombuffer(f.read(4), dtype=np.int32)[0]
                    f.read(8 * dim_count)  # 跳过 dims 字段（比对只关心数据本身）
                    tmp_out = np.frombuffer(f.read(), dtype=np.complex64)
                else:
                    # 没有 header，重新读取
                    tmp_out = np.fromfile(out, np.complex64)
        else:
            tmp_gold_raw = np.fromfile(gold, np.float16)
            n_gold = len(tmp_gold_raw) // 2
            tmp_gold = np.zeros(n_gold, dtype=np.complex64)
            for i in range(n_gold):
                tmp_gold[i] = complex(
                    float(tmp_gold_raw[2 * i]), float(tmp_gold_raw[2 * i + 1])
                )

            with open(out, "rb") as f:
                dtype_int = np.frombuffer(f.read(4), dtype=np.int32)[0]
                if dtype_int == 33:  # 有 header
                    dim_count = np.frombuffer(f.read(4), dtype=np.int32)[0]
                    f.read(8 * dim_count)  # 跳过 dims 字段（比对只关心数据本身）
                    tmp_out_raw = np.frombuffer(f.read(), dtype=np.float16)
                else:
                    tmp_out_raw = np.fromfile(out, np.float16)

            n_out = len(tmp_out_raw) // 2
            tmp_out = np.zeros(n_out, dtype=np.complex64)
            for i in range(n_out):
                tmp_out[i] = complex(
                    float(tmp_out_raw[2 * i]), float(tmp_out_raw[2 * i + 1])
                )

        if len(tmp_out) != len(tmp_gold):
            print(
                f"FAILED! {out}: size mismatch (out={len(tmp_out)}, gold={len(tmp_gold)})"
            )
            data_same = False
            continue

        diff_res = np.isclose(tmp_out, tmp_gold, precision, precision)
        diff_idx = np.where(~diff_res)[0]

        if len(diff_idx) == 0:
            print(f"PASSED! {out}")
            print(f"Max absolute error: {np.max(np.abs(tmp_out - tmp_gold))}")
            print(
                f"Max relative error: {np.max(np.abs(tmp_out - tmp_gold) / (np.abs(tmp_gold) + 1e-10))}"
            )
        else:
            print(f"FAILED! {out}")
            for idx in diff_idx[:5]:
                print(
                    f"  index: {idx}, output: {tmp_out[idx]}, golden: {tmp_gold[idx]}, diff: {tmp_out[idx] - tmp_gold[idx]}"
                )
            data_same = False

    sys.exit(0 if data_same else 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare output with golden data for asdMul"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        required=True,
        choices=["c64", "c32"],
        help="Data type: c64 or c32",
    )
    args = parser.parse_args()
    compare_data(args.dtype)
