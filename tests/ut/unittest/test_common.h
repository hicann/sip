/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCEND_UNIT_TEST_COMMON_H
#define ASCEND_UNIT_TEST_COMMON_H

#include <cmath>
#include <complex>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <set>
#include <string>
#include <vector>

#include "mki/utils/SVector/SVector.h"

int64_t Prod(const Mki::SVector<int64_t>& vec);

/**
 * @brief 读取 golden 二进制文件（纯数据，无 header）并按 isclose(rtol, atol) 语义
 *        与 NPU 输出 host 数据直接比对，等价于原 compare_data.py 的 numpy 流程。
 *        消除每用例一次 python3 进程启动开销（gen_data.py 精度基准不变，golden
 *        仍由 python 侧预生成，见各 FFT 数据目录）。
 * @param goldenPath golden 文件路径（np.complex64/float32 纯数据 dump）
 * @param outHost    NPU 输出 host 内存首地址
 * @param outElems   NPU 输出元素个数
 * @param isComplex  true=complex64（元素 8 字节），false=float32（元素 4 字节）
 * @param rtol/atol  numpy.isclose 同名参数语义
 * @param maxAbsErr/maxRelErr 输出误差统计（可传 nullptr 忽略）
 * @return 0=全部在容差内；-1=文件读取失败；>0=超差元素个数
 */
static inline int CompareGoldenWithOutput(const std::string& goldenPath, const void* outHost, size_t outElems,
                                          bool isComplex, double rtol, double atol, double* maxAbsErr,
                                          double* maxRelErr)
{
    std::ifstream file(goldenPath, std::ios::binary);
    if (!file.is_open()) {
        std::cout << "CompareGoldenWithOutput: open golden failed: " << goldenPath << std::endl;
        return -1;
    }
    file.seekg(0, std::ios::end);
    std::streamoff byteSize = file.tellg();
    file.seekg(0, std::ios::beg);
    size_t goldenElems = static_cast<size_t>(byteSize) / (isComplex ? sizeof(std::complex<float>) : sizeof(float));
    if (goldenElems != outElems) {
        std::cout << "CompareGoldenWithOutput: size mismatch (out=" << outElems << ", gold=" << goldenElems << ")"
                  << std::endl;
        return -1;
    }

    // 一次性读入 golden 缓冲（避免逐元素 iostream 调用开销）
    std::vector<char> goldenBuf(static_cast<size_t>(byteSize));
    file.read(goldenBuf.data(), byteSize);
    file.close();

    int mismatch = 0;
    double localMaxAbs = 0.0, localMaxRel = 0.0;
    for (size_t i = 0; i < outElems; ++i) {
        std::complex<double> o, g;
        if (isComplex) {
            // golden 与输出均为 planar 复数（实部虚部各 4 字节连续）
            const float* goldPtr = reinterpret_cast<const float*>(goldenBuf.data()) + i * 2;
            g = std::complex<double>(goldPtr[0], goldPtr[1]);
            const float* outPtr = reinterpret_cast<const float*>(outHost) + i * 2;
            o = std::complex<double>(outPtr[0], outPtr[1]);
        } else {
            g = static_cast<double>(reinterpret_cast<const float*>(goldenBuf.data())[i]);
            o = static_cast<double>(reinterpret_cast<const float*>(outHost)[i]);
        }
        double diff = std::abs(o - g);
        double tol = atol + rtol * std::abs(g);
        // NaN 安全：diff 为 NaN 时 (diff > tol) 恒 false 会漏检，用 !(diff <= tol)
        // 使 NaN 自然落入 mismatch 分支（对齐 numpy.isclose equal_nan=False 语义）
        if (!(diff <= tol)) {
            if (mismatch < 5) {
                std::cout << "  mismatch idx=" << i << " out=" << o << " gold=" << g << " diff=" << diff
                          << " tol=" << tol << std::endl;
            }
            ++mismatch;
        }
        if (diff > localMaxAbs) {
            localMaxAbs = diff;
        }
        double rel = diff / (std::abs(g) + 1e-10);
        if (rel > localMaxRel) {
            localMaxRel = rel;
        }
    }
    if (maxAbsErr != nullptr) {
        *maxAbsErr = localMaxAbs;
    }
    if (maxRelErr != nullptr) {
        *maxRelErr = localMaxRel;
    }
    return mismatch;
}

/**
 * @brief 对字符串进行 shell 单引号包裹，防止路径中的空格或特殊字符导致 system() 调用解析失败。
 * @param s 原始字符串（如文件路径）
 * @return 被单引号包裹并转义内部单引号的字符串
 */
static inline std::string ShellQuote(const std::string& s)
{
    std::string result = "'";
    for (char c : s) {
        if (c == '\'') {
            result += "'\\''";
        } else {
            result += c;
        }
    }
    result += "'";
    return result;
}

/**
 * @brief suite 级一次性的数据目录准备：按目标目录 key 记忆化，每个 key 仅执行一次
 *        mkdir/cp/chmod，消除原实现每用例 3 次 system() 的 fork 开销。
 *        调用方应以 static local 形式持有返回值（C++11 初始化 once 语义）。
 *        注意：函数级 static 非线程安全，仅限 gtest 串行执行场景使用。
 * @return 目标目录路径（不含结尾 data 子目录名，数据子目录为 destPath/dataDirName）
 */
static inline std::string PrepareDataDirOnce(const std::string& currentDir, const std::string& relUnderTests,
                                             const std::string& dataDirName)
{
    std::string originPath = currentDir + "/tests/ut/unittest/" + relUnderTests + "/" + dataDirName;
    std::string destPath = currentDir + "/build/tests/ut/unittest/" + relUnderTests;
    static std::set<std::string> doneKeys;
    if (doneKeys.count(destPath) != 0) {
        return destPath;
    }
    std::string cmd = "mkdir -p " + ShellQuote(destPath) + " && cp -rf " + ShellQuote(originPath) + " " +
                      ShellQuote(destPath) + " && chmod -R 755 " + ShellQuote(destPath + "/" + dataDirName + "/");
    (void)system(cmd.c_str());
    doneKeys.insert(destPath);
    return destPath;
}

#endif // ASCEND_UNIT_TEST_COMMON_H
