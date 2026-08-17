#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import argparse
import json
import logging
import os
import re
import stat
import sys


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--soc", type=str, required=True)
    parser.add_argument("--channel", type=str, required=True)
    parser.add_argument("--srcs", type=str, required=True)
    parser.add_argument("--dst", type=str, required=True)
    parser.add_argument("--code_root", type=str, required=True)
    parser.add_argument("--kernel", type=str, required=True)
    parser.add_argument("--use_msdebug", type=str)
    parser.add_argument("--use_mssanitizer", type=str, required=True)
    parser.add_argument("--no_warning", action="store_true")
    parser.add_argument("--include_directories", type=str, required=False, nargs="+")
    parser.add_argument("--use_ascendc_dump", action="store_true")
    return parser.parse_args()


def gen_compile_cmd(args, dst: str, sub_arch: str, compile_options):
    compile_cmd = [
        os.path.join(
            args.code_root, "3rdparty", "compiler", "ccec_compiler", "bin", "ccec"
        ),
        "-c",
    ]
    if args.use_msdebug == "ON":
        compile_cmd += ["-O0", "-g", "--cce-ignore-always-inline=true"]
    else:
        compile_cmd += ["-O2"]
    compile_cmd += compile_options
    compile_cmd += [
        args.srcs,
        "--cce-aicore-arch=%s" % sub_arch,
        "--cce-aicore-only",
        "-o",
        dst,
        "-mllvm",
        "-cce-aicore-fp-ceiling=2",
    ]
    if args.use_mssanitizer == "ON" and args.soc in ["ascend310p", "ascend910b"]:
        compile_cmd += [
            "-g",
            "--cce-enable-sanitizer",
            "-mllvm",
            "-cce-aicore-long-call",
            "-mllvm",
            "-cce-aicore-jump-expand=true",
        ]
    compile_cmd += ["-std=c++17"]
    compile_cmd += ["--cce-mask-opt"]
    if args.use_ascendc_dump:
        compile_cmd += ["-mllvm", "-cce-aicore-function-stack-size=0x4000"]
        compile_cmd += ["--cce-long-call=true"]
        compile_cmd += ["-DASCENDC_DUMP=1"]
        compile_cmd += ["-DASCENDC_DEBUG"]
    return compile_cmd


def gen_compile_cmd_v220(args, dst: str, sub_arch: str, compile_options):
    compile_cmd = [
        os.path.join(
            args.code_root, "3rdparty", "compiler", "ccec_compiler", "bin", "ccec"
        ),
        "-c",
    ]
    if args.use_msdebug == "ON":
        compile_cmd += ["-O0", "-g", "--cce-ignore-always-inline=true"]
    else:
        compile_cmd += ["-O3"]
    compile_cmd += compile_options
    compile_cmd += [
        args.srcs,
        "--cce-aicore-arch=%s" % sub_arch,
        "--cce-aicore-only",
        "-o",
        dst,
        "-mllvm",
        "-cce-aicore-stack-size=0x8000",
        "-mllvm",
        "-cce-aicore-function-stack-size=0x8000",
        "-mllvm",
        "-cce-aicore-record-overflow=true",
        "-mllvm",
        "-cce-aicore-addr-transform",
        "-mllvm",
        "-cce-aicore-dcci-insert-for-scalar=false",
    ]
    if args.use_mssanitizer == "ON":
        compile_cmd += [
            "-g",
            "--cce-enable-sanitizer",
            "-mllvm",
            "-cce-aicore-long-call",
            "-mllvm",
            "-cce-aicore-jump-expand=true",
        ]
    compile_cmd += ["-std=c++17"]
    if args.use_ascendc_dump:
        compile_cmd += ["--cce-long-call=true"]
        compile_cmd += ["-DASCENDC_DUMP=1"]
        compile_cmd += ["-DASCENDC_DEBUG"]
    return compile_cmd


def gen_compile_cmd_v300(args, dst: str, sub_arch: str, compile_options):
    compile_cmd = [
        os.path.join(
            args.code_root, "3rdparty", "compiler", "ccec_compiler", "bin", "ccec"
        ),
        "-c",
    ]
    if args.use_msdebug == "ON":
        compile_cmd += ["-O0", "-g", "--cce-ignore-always-inline=true"]
    else:
        compile_cmd += ["-O3"]
    compile_cmd += compile_options
    compile_cmd += [
        args.srcs,
        "--cce-aicore-arch=%s" % sub_arch,
        "--cce-aicore-only",
        "-o",
        dst,
        "-mllvm",
        "-cce-aicore-function-stack-size=0x4000",
        "-mllvm",
        "-cce-aicore-addr-transform",
        "-mllvm",
        "--cce-aicore-or-combine=false",
        "-mllvm",
        "-instcombine-code-sinking=false",
        "-mllvm",
        "-cce-aicore-jump-expand=false",
        "-mllvm",
        "-cce-aicore-mask-opt=false",
    ]
    compile_cmd += ["-std=c++17"]
    if args.use_ascendc_dump:
        compile_cmd += ["--cce-long-call=true"]
        compile_cmd += ["-DASCENDC_DUMP=1"]
        compile_cmd += ["-DASCENDC_DEBUG"]
    return compile_cmd


def gen_compile_cmd_c310(args, dst: str, sub_arch: str, compile_options):
    compile_cmd = [
        os.path.join(
            args.code_root, "3rdparty", "compiler", "ccec_compiler", "bin", "bisheng"
        ),
        "-c",
    ]
    compile_cmd += compile_options
    compile_cmd += [
        args.srcs,
        "--cce-aicore-arch=%s" % sub_arch,
        "--cce-aicore-only",
        "-o",
        dst,
        "-mllvm",
        "-cce-aicore-stack-size=0x8000",
        "-mllvm",
        "-cce-aicore-function-stack-size=0x8000",
        "-mllvm",
        "-cce-aicore-record-overflow=true",
        "-mllvm",
        "-cce-aicore-addr-transform",
        "-mllvm",
        "-cce-aicore-jump-expand=true",
        "-mllvm",
        "-cce-aicore-dcci-insert-for-scalar=false",
        "-mllvm",
        "-cce-aicore-dcci-before-kernel-end=false",
    ]
    compile_cmd += ["-std=c++17"]
    return compile_cmd


def gen_fatbin_cmd(args, obj_file: list, dst_file: str):
    compile_cmd = [
        os.path.join(
            args.code_root, "3rdparty", "compiler", "ccec_compiler", "bin", "ld.lld"
        ),
        "-m",
        "aicorelinux",
        "-Ttext=0",
    ]
    compile_cmd += obj_file
    compile_cmd += ["-static", "-o", "%s" % dst_file]
    return compile_cmd


def gen_json(args, kernels):
    json_template = {
        "binFileName": "",
        "binFileSuffix": ".o",
        "blockDim": -1,
        "coreType": "",
        "core_type": "",
        "intercoreSync": 0,
        "magic": "",
        "memoryStamping": [],
        "opParaSize": 0,
        "parameters": [],
        "sha256": "",
        "kernelList": [],
        "compileInfo": {},
    }
    json_template["binFileName"] = args.kernel
    for kernel in kernels:
        json_template["kernelList"].append({"kernelName": kernel})
    if args.soc == "ascend910" or args.soc == "ascend310p" or args.soc == "ascend310b":
        json_template["coreType"] = "AiCore"
        json_template["core_type"] = "AIC"
        json_template["magic"] = "RT_DEV_BINARY_MAGIC_ELF"
    elif args.channel == "cube":
        json_template["coreType"] = "AiCore"
        json_template["core_type"] = "AIC"
        json_template["magic"] = "RT_DEV_BINARY_MAGIC_ELF_AICUBE"
    elif args.channel == "vector":
        json_template["coreType"] = "VectorCore"
        json_template["core_type"] = "AIV"
        json_template["magic"] = "RT_DEV_BINARY_MAGIC_ELF_AIVEC"
    elif args.channel == "mix":
        json_template["coreType"] = "MIX"
        json_template["core_type"] = "MIX"
        json_template["magic"] = "RT_DEV_BINARY_MAGIC_ELF"

    with os.fdopen(
        os.open(
            os.path.splitext(args.dst)[0] + ".json",
            os.O_TRUNC | os.O_WRONLY | os.O_CREAT,
            stat.S_IWUSR | stat.S_IRUSR,
        ),
        "w",
    ) as f:
        json.dump(json_template, f, indent=4)


def get_common_options(args):
    tikcpp_path = os.path.join(args.code_root, "3rdparty", "compiler", "tikcpp")
    options = ["-x", "cce"]
    options.append("-I.")
    options.append("-I" + tikcpp_path)
    options.append("-I" + os.path.join(tikcpp_path, "tikcfw"))
    options.append("-I" + os.path.join(tikcpp_path, "tikcfw", "impl"))
    options.append("-I" + os.path.join(tikcpp_path, "tikcfw", "interface"))
    if args.include_directories:
        for dir in args.include_directories:
            options.append("-I" + dir)
    if args.no_warning:
        options.append("-Wno-deprecated-declarations")
        options.append("-Wno-array-bounds")
    return options


def get_tiling_key_ids(src_file):
    with open(src_file) as f:
        s = f.read()
        patt = r"TILING_KEY_IS\((\d+)\)"
        pattern = re.compile(patt)
        tiling_key_list = pattern.findall(s)
        if tiling_key_list:
            return tiling_key_list
        else:
            return [0]


def get_arch(soc, channel):
    arch_dict = {
        "ascend310b": {"vector": "dav-m300", "cube": "dav-m300"},
        "ascend310p": {"vector": "dav-m200", "cube": "dav-m200"},
        "ascend910": {"vector": "dav-c100", "cube": "dav-c100"},
        "ascend910b": {"vector": "dav-c220-vec", "cube": "dav-c220-cube", "mix": "mix"},
        "ascend950": {"vector": "dav-c310", "cube": "dav-c310", "mix": "mix"},
    }
    try:
        return arch_dict[soc][channel]
    except KeyError:
        logging.error("input soc %s or channel %s is invalid", soc, channel)
        return "None"


def exe_cmd(cmd):
    if os.system(cmd) != 0:
        logging.error("execute command failed")
        logging.debug("command: %s", cmd)
        return -1
    return 0


def compile_ascendc_operation(args):
    dsts = []
    kernels = []
    options = get_common_options(args)
    arch = get_arch(args.soc, args.channel)
    compile_cmd = ""
    link_cmd = ""
    ascend_home_path = os.getenv("ASCEND_HOME_PATH", "ASCEND_HOME_PATH does not exist.")
    mssanitizer_path = os.path.join(ascend_home_path, "tools", "mssanitizer", "lib64")

    if arch == "None":
        return -1
    tiling_key_ids = get_tiling_key_ids(args.srcs)
    logging.debug("tiling_key_ids: ", tiling_key_ids)
    for key in tiling_key_ids:
        if args.soc == "ascend310p" or args.soc == "ascend910":
            dst = os.path.splitext(args.dst)[0] + f"_{key}.o"
            opt = options + [
                f"-D{args.kernel}={args.kernel}_{key}",
                f"-DTILING_KEY_VAR={key}",
            ]
            compile_cmd = " ".join(gen_compile_cmd(args, dst, arch, opt))
            if (exe_cmd(compile_cmd)) != 0:
                return -1
            dsts.append(dst)
            if args.use_mssanitizer == "ON" and args.soc == "ascend310p":
                dsts.append("--dependent-libraries")
                dsts.append(
                    os.path.join(mssanitizer_path, "libsanitizer_stub_dav-m200.a")
                )
        elif args.soc == "ascend910b":
            if args.channel != "mix":
                dst = os.path.splitext(args.dst)[0] + f"_{key}.o"
                opt = options + [
                    f"-D{args.kernel}={args.kernel}_{key}",
                    f"-DTILING_KEY_VAR={key}",
                ]
                compile_cmd = " ".join(gen_compile_cmd_v220(args, dst, arch, opt))
                if (exe_cmd(compile_cmd)) != 0:
                    return -1
                dsts.append(dst)
                if args.use_mssanitizer == "ON":
                    dsts.append("--dependent-libraries")
                    dsts.append(
                        os.path.join(
                            mssanitizer_path, "libsanitizer_stub_dav-c220-cube.a"
                        )
                    )
                    dsts.append(
                        os.path.join(
                            mssanitizer_path, "libsanitizer_stub_dav-c220-vec.a"
                        )
                    )
            else:
                dst = os.path.splitext(args.dst)[0] + f"_mix_aic_{key}.o"
                aic_opt = options + [
                    f"-D{args.kernel}={args.kernel}_{key}_mix_aic",
                    f"-DTILING_KEY_VAR={key}",
                ]
                compile_cmd = " ".join(
                    gen_compile_cmd_v220(args, dst, "dav-c220-cube", aic_opt)
                )
                if (exe_cmd(compile_cmd)) != 0:
                    return -1
                dsts.append(dst)
                if args.use_mssanitizer == "ON":
                    dsts.append("--dependent-libraries")
                    dsts.append(
                        os.path.join(
                            mssanitizer_path, "libsanitizer_stub_dav-c220-cube.a"
                        )
                    )
                dst = os.path.splitext(args.dst)[0] + f"_mix_aiv_{key}.o"
                aiv_opt = options + [
                    f"-D{args.kernel}={args.kernel}_{key}_mix_aiv",
                    f"-DTILING_KEY_VAR={key}",
                ]
                compile_cmd = " ".join(
                    gen_compile_cmd_v220(args, dst, "dav-c220-vec", aiv_opt)
                )
                if (exe_cmd(compile_cmd)) != 0:
                    return -1
                dsts.append(dst)
                if args.use_mssanitizer == "ON":
                    dsts.append("--dependent-libraries")
                    dsts.append(
                        os.path.join(
                            mssanitizer_path, "libsanitizer_stub_dav-c220-vec.a"
                        )
                    )
        elif args.soc == "ascend310b":
            dst = os.path.splitext(args.dst)[0] + f"_{key}.o"
            opt = options + [
                f"-D{args.kernel}={args.kernel}_{key}",
                f"-DTILING_KEY_VAR={key}",
            ]
            compile_cmd = " ".join(gen_compile_cmd_v300(args, dst, arch, opt))
            if (exe_cmd(compile_cmd)) != 0:
                return -1
            dsts.append(dst)
        elif args.soc == "ascend950":
            if args.channel != "mix":
                dst = os.path.splitext(args.dst)[0] + f"_{key}.o"
                opt = options + [
                    f"-D{args.kernel}={args.kernel}_{key}",
                    f"-DTILING_KEY_VAR={key}",
                ]
                compile_cmd = " ".join(gen_compile_cmd_c310(args, dst, arch, opt))
                if (exe_cmd(compile_cmd)) != 0:
                    return -1
                dsts.append(dst)
            else:
                dst = os.path.splitext(args.dst)[0] + f"_mix_aic_{key}.o"
                aic_opt = options + [
                    f"-D{args.kernel}={args.kernel}_{key}_mix_aic",
                    f"-DTILING_KEY_VAR={key}",
                ]
                compile_cmd = " ".join(
                    gen_compile_cmd_c310(args, dst, "dav-c310", aic_opt)
                )
                if (exe_cmd(compile_cmd)) != 0:
                    return -1
                dsts.append(dst)
                dst = os.path.splitext(args.dst)[0] + f"_mix_aiv_{key}.o"
                aiv_opt = options + [
                    f"-D{args.kernel}={args.kernel}_{key}_mix_aiv",
                    f"-DTILING_KEY_VAR={key}",
                ]
                compile_cmd = " ".join(
                    gen_compile_cmd_c310(args, dst, "dav-c310", aiv_opt)
                )
                if (exe_cmd(compile_cmd)) != 0:
                    return -1
                dsts.append(dst)
        else:
            logging.error("soc version %s is not supported", args.soc)
            sys.exit(1)
        kernels.append(f"{args.kernel}_{key}")

    link_cmd = " ".join(gen_fatbin_cmd(args, dsts, args.dst))
    if (exe_cmd(link_cmd)) != 0:
        return -1
    gen_json(args, kernels)
    return 0


if __name__ == "__main__":
    input_args = parse_args()
    output_dir = os.path.dirname(input_args.dst)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    res = compile_ascendc_operation(input_args)
    if res != 0:
        logging.error("compile ascend C failed!")
        sys.exit(1)
