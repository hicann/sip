#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
"""

import os
import glob
import sysconfig
import torch
from torch.utils import cpp_extension
import torch_npu
from torch_npu.utils.cpp_extension import NpuExtension
from setuptools import setup, find_packages

USE_NINJA = os.getenv("USE_NINJA") == "1"
BASE_DIR = os.path.dirname(os.path.realpath(__file__))
# 从环境变量读取sip安装路径
SIP_ROOT_DIR = os.getenv("ASDSIP_HOME_PATH", "/usr/local/Ascend/asdsip/latest")


def get_dependency_paths():
    python_include = sysconfig.get_config_var("INCLUDEPY")
    default_boost_include = os.path.abspath(
        os.path.join(BASE_DIR, "..", "3rdparty", "ascend-boost-comm", "src", "include")
    )
    # ascend-boost-comm include路径
    boost_comm_include = os.environ.get("BOOST_COMM_PATH", default_boost_include)
    torch_include_paths = cpp_extension.include_paths()
    torch_lib_dir = os.path.join(os.path.dirname(torch.__file__), "lib")

    torch_npu_path = os.path.dirname(torch_npu.__file__)
    torch_npu_include = os.path.join(torch_npu_path, "include")
    torch_npu_lib = os.path.join(torch_npu_path, "lib")

    # Acl 路径
    acl_include = os.path.join(torch_npu_path, "include/third_party/acl/inc")
    ascend_home = os.environ.get("ASCEND_HOME_PATH", "")
    ascend_include = os.path.join(ascend_home, "include")

    sip_include_dir = os.path.join(SIP_ROOT_DIR, "include")
    sip_lib_dir = os.path.join(SIP_ROOT_DIR, "lib")
    util_include_dir = os.path.join(BASE_DIR, "csrc")
    return {
        "includes": [
            python_include,
            torch_npu_include,
            sip_include_dir,
            acl_include,
            *torch_include_paths,
            boost_comm_include,
            ascend_include,
            util_include_dir,
        ],
        "libs": [torch_lib_dir, torch_npu_lib, sip_lib_dir],
    }


dep_paths = get_dependency_paths()
your_ext = NpuExtension(
    name="sip_custom_ops_lib",
    sources=glob.glob(os.path.join(BASE_DIR, "csrc", "**", "*.cpp"), recursive=True),
    include_dirs=dep_paths["includes"],
    library_dirs=dep_paths["libs"],
    libraries=["torch", "torch_npu", "c10", "asdsip", "mki"],
    language="c++",
    extra_compile_args={
        "cxx": [
            "-std=c++17",
            f"-D_GLIBCXX_USE_CXX11_ABI={1 if torch.compiled_with_cxx11_abi() else 0}",
            "-D_FORTIFY_SOURCE=2",
            "-O2",
            "-fstack-protector-strong",
            "-ftrapv",
            "-Wno-unused-function",
            "-Wno-unused-variable",
        ],
    },
    extra_link_args=["-Wl,-s,-z,now"],
)

setup(
    name="torch_sip",
    version="0.1.0",
    ext_modules=[your_ext],
    packages=find_packages(),
    cmdclass={
        "build_ext": cpp_extension.BuildExtension.with_options(use_ninja=USE_NINJA)
    },
)
