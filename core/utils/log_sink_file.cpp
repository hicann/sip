/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <string>
#include <cstdlib>
#include <cstring>
#include <clocale>
#include <cctype>
#include <syscall.h>
#include <mutex>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <sys/stat.h>
#include <pwd.h>
#include <fcntl.h>
#include <unistd.h>
#include <dirent.h>
#include <sys/statvfs.h>
#include <securec.h>
#include "log/log_core_sip.h"
#include "utils/env.h"
#include "mki/utils/strings/match.h"
#include "mki/utils/strings/str_checker.h"
#include "mki/utils/strings/str_split.h"
#include "log/log_sink_file_sip.h"

namespace AsdSip {
constexpr size_t MAX_LOG_FILE_COUNT = 50;                         // 50 回滚管理50个日志文件
constexpr size_t MAX_FILE_NAME_LEN = 128;                         // 128: max file length
constexpr uint64_t MAX_FILE_SIZE_THRESHOLD = 20 * 1024 * 1024;    // 20,971,520 当前单个日志文件最大20M
constexpr uint64_t DISK_AVAILABEL_LIMIT = 1 * 1024 * 1024 * 1024; // 磁盘剩余空间门限1G

static bool isValidChar(unsigned char c)
{
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || (c == '.') || (c == '_') ||
           (c == '-') || (c == '/');
}

std::mutex localeMutex;

static bool IsValidFileName(const char* name)
{
    std::lock_guard<std::mutex> lock(localeMutex);

    size_t len = strlen(name);
    if (len == 0 || len > MAX_FILE_NAME_LEN) {
        return false;
    }

    // 保存当前的locale设置
    std::string original_locale = std::setlocale(LC_CTYPE, nullptr);

    // 设置新的locale环境
    std::setlocale(LC_CTYPE, "en_US.UTF-8");

    for (size_t i = 0; i < len; ++i) {
        char c = name[i];
        if (!isValidChar(static_cast<unsigned char>(c))) {
            // 恢复原来的locale设置
            std::setlocale(LC_CTYPE, original_locale.c_str());
            return false;
        }
    }

    // 恢复原来的locale设置
    std::setlocale(LC_CTYPE, original_locale.c_str());
    return true;
}

LogSinkFileSip::LogSinkFileSip() { Init(); }

LogSinkFileSip::~LogSinkFileSip() { CloseFile(); }

void LogSinkFileSip::LogSip(const char* log, uint64_t logLen)
{
    std::lock_guard<std::mutex> guard(mutex_);
    if (currentFileSize_ + logLen >= MAX_FILE_SIZE_THRESHOLD) {
        CloseFile();
    }

    if (currentFd_ < 0) {
        OpenFile();
    }

    if (currentFd_ < 0) {
        return;
    }

    ssize_t writeSize = write(currentFd_, log, logLen);
    if (writeSize != static_cast<ssize_t>(logLen)) {
        std::cout << "asdsip_log write file fail, want to write size: " << logLen
                  << ", success write size:" << writeSize;
        CloseFile();
        return;
    }

    currentFileSize_ += writeSize;
}

// 辅助函数：检查路径是否存在且为目录
static bool IsDirectory(const std::string& path)
{
    struct stat statbuf;
    if (stat(path.c_str(), &statbuf) != 0) {
        return false;
    }
    return S_ISDIR(statbuf.st_mode);
}

// 辅助函数：检查路径是否可写
static bool IsWritable(const std::string& path) { return access(path.c_str(), W_OK) == 0; }

// 辅助函数：替换字符串中的所有指定子串
static void ReplaceAll(std::string& str, const std::string& from, const std::string& to)
{
    if (from.empty()) {
        return;
    }
    size_t start_pos = 0;
    while ((start_pos = str.find(from, start_pos)) != std::string::npos) {
        str.replace(start_pos, from.length(), to);
        start_pos += to.length(); // 防止无限循环（如果to包含from）
    }
}

// 辅助函数：安全规范化路径（处理各种绕过技术）
static std::string SafeNormalizePath(const std::string& inputPath)
{
    if (inputPath.empty()) {
        return "";
    }

    // 处理URI编码
    std::string path;
    size_t i = 0;

    while (i < inputPath.size()) {
        if (inputPath[i] == '%' && i + 2 < inputPath.size()) {
            // 取 % 后面两个字符
            char hex[3] = {inputPath[i + 1], inputPath[i + 2], '\0'};
            char* end;
            long val = strtol(hex, &end, 16);

            // 检查是否成功解析了两个十六进制字符（end 应该指向末尾 '\0'）
            if (end == hex + 2) { // 关键：必须完整解析两个字符
                path += static_cast<char>(val);
                i += 3; // 跳过 '%' 和两个十六进制字符
                continue;
            }
        }

        // 普通字符或无效 %XX 编码，直接添加
        path += inputPath[i];
        ++i;
    }

    // 2. 处理Unicode欺骗字符 - 使用字符串替换
    ReplaceAll(path, "．", "."); // 全角点 U+FF0E
    ReplaceAll(path, "﹒", "."); // 小点 U+FE52
    ReplaceAll(path, "。", "."); // 中文句号 U+3002
    ReplaceAll(path, "／", "/"); // 全角斜杠 U+FF0F

    // 处理尾部截断攻击
    size_t null_pos = path.find('\0');
    if (null_pos != std::string::npos) {
        path = path.substr(0, null_pos);
    }

    // 处理特殊分隔符
    std::replace(path.begin(), path.end(), '\\', '/');

    // 处理相对路径和重复分隔符
    std::vector<std::string> parts;
    std::string current;
    bool isAbsolute = (path[0] == '/');

    for (char c : path) {
        if (c == '/') {
            if (!current.empty()) {
                if (current == ".") {
                    // 忽略当前目录
                } else if (current == "..") {
                    if (!parts.empty() && parts.back() != "..") {
                        parts.pop_back();
                    } else if (!isAbsolute) {
                        parts.push_back(current);
                    }
                } else {
                    parts.push_back(current);
                }
                current.clear();
            }
        } else {
            current += c;
        }
    }

    if (!current.empty()) {
        if (current == ".") {
            // 忽略
        } else if (current == "..") {
            if (!parts.empty() && parts.back() != "..") {
                parts.pop_back();
            } else if (!isAbsolute) {
                parts.push_back(current);
            }
        } else {
            parts.push_back(current);
        }
    }

    // 构建规范化路径
    std::string normalized;
    for (const auto& part : parts) {
        if (!normalized.empty() || isAbsolute) {
            normalized += "/";
        }
        normalized += part;
    }

    if (normalized.empty()) {
        normalized = isAbsolute ? "/" : ".";
    }

    return normalized;
}

// 解析真实路径并验证安全性
static std::string ResolveAndValidatePath(const std::string& inputPath)
{
    // 1. 安全规范化路径
    std::string normalized = SafeNormalizePath(inputPath);
    char resolved[PATH_MAX] = {0};

    // 2. 解析真实路径
    if (realpath(normalized.c_str(), resolved) == nullptr) {
        // 路径不存在，尝试创建
        if (mkdir(normalized.c_str(), 0755) != 0 && errno != EEXIST) {
            return ""; // 创建失败
        }
        if (realpath(normalized.c_str(), resolved) == nullptr) {
            return ""; // 仍然解析失败
        }
    }

    std::string realPath = resolved;

    // 3. 验证路径属性
    if (!IsDirectory(realPath)) {
        return ""; // 不是目录
    }

    if (!IsWritable(realPath)) {
        return ""; // 不可写
    }

    // 4. 防止敏感目录访问
    const std::vector<std::string> forbiddenPaths = {"/",     "/bin", "/sbin", "/usr", "/etc",
                                                     "/root", "/var", "/sys",  "/proc"};

    for (const auto& forbidden : forbiddenPaths) {
        if (realPath == forbidden || realPath.find(forbidden + "/") == 0) {
            return ""; // 禁止访问系统目录
        }
    }

    // 5. 防止通过符号链接逃逸
    std::string current = realPath;
    while (!current.empty()) {
        char linkPath[PATH_MAX] = {0};
        ssize_t len = readlink(current.c_str(), linkPath, sizeof(linkPath) - 1);
        if (len > 0) {
            linkPath[len] = '\0';
            std::string linkTarget = SafeNormalizePath(linkPath);
            if (!linkTarget.empty() && linkTarget[0] != '/') {
                // 相对路径符号链接
                size_t pos = current.find_last_of('/');
                if (pos != std::string::npos) {
                    linkTarget = current.substr(0, pos) + "/" + linkTarget;
                }
            }

            // 检查符号链接目标是否在禁止目录
            for (const auto& forbidden : forbiddenPaths) {
                if (linkTarget.find(forbidden) == 0) {
                    return ""; // 符号链接指向禁止目录
                }
            }

            // 继续检查符号链接目标的父目录
            size_t pos = current.find_last_of('/');
            if (pos == 0) {
                break; // 到达根目录
            }
            current = current.substr(0, pos);
        } else {
            break; // 不是符号链接或错误
        }
    }

    return realPath;
}

void LogSinkFileSip::Init()
{
    const char* env = "asdsip";
    boostType_ = env && strlen(env) <= MAX_ENV_STRING_LEN && IsValidFileName(env) ? std::string(env) : "asdsip";

    env = std::getenv("ASCEND_PROCESS_LOG_PATH") ? std::getenv("ASCEND_PROCESS_LOG_PATH") : "asdsip";
    std::string logRootDir = env && strlen(env) <= MAX_ENV_STRING_LEN && IsValidFileName(env) ? std::string(env) :
                                                                                                GetHomeDir();

    logRootDir = ResolveAndValidatePath(logRootDir);
    if (logRootDir.empty()) {
        std::cerr << "Warning: User-specified log path is invalid: " << env;
        // 回退到默认安全路径
        std::string defaultPath = GetHomeDir();
        logRootDir = ResolveAndValidatePath(defaultPath);
    }

    logDir_ = logRootDir + "/log" + "/" + boostType_;

    env = "1";
    isFlush_ = env && strlen(env) <= MAX_ENV_STRING_LEN ? std::string(env) == "1" : false;
}

bool LogSinkFileSip::IsFileNameMatched(const std::string& fileName, std::string& createTime)
{
    std::string prefix = boostType_ + '_';
    bool match = Mki::StartsWith(fileName, prefix);
    if (!match) {
        return false;
    }
    match = Mki::EndsWith(fileName, ".log");
    if (!match) {
        return false;
    }
    size_t subStrLen = fileName.length() - prefix.length() - 4; // 4: length of ".log" postfix
    std::string subStr = fileName.substr(prefix.length(), subStrLen);
    std::vector<std::string> splitResult;
    Mki::StrSplit(subStr, '_', splitResult);
    if (splitResult.size() != 2) { // 2: time & pid
        return false;
    }
    for (auto& str : splitResult) {
        if (str.empty() || !std::all_of(str.begin(), str.end(), ::isdigit)) {
            return false;
        }
    }
    createTime = splitResult.at(1);
    return true;
}

void LogSinkFileSip::DeleteOldestFile()
{
    std::vector<std::pair<std::string, std::string>> logFiles;
    DIR* dir = opendir(logDir_.c_str());
    if (!dir) {
        return;
    }
    struct dirent* ptr = nullptr;
    while ((ptr = readdir(dir)) != nullptr) {
        if (ptr->d_name[0] != '.') {
            std::string fileName = ptr->d_name;
            std::string createTime;
            if (IsFileNameMatched(fileName, createTime)) {
                logFiles.emplace_back(logDir_ + "/" + fileName, createTime);
            }
        }
    }
    closedir(dir);

    std::sort(logFiles.begin(), logFiles.end(),
              [](std::pair<std::string, std::string>& a, std::pair<std::string, std::string>& b) {
                  return a.second < b.second;
              });

    if (logFiles.size() >= MAX_LOG_FILE_COUNT) {
        size_t deleteCount = logFiles.size() - MAX_LOG_FILE_COUNT + 1;
        for (size_t i = 0; i < deleteCount; ++i) {
            std::cout << "asdsip_log delete old file:" << logFiles[i].first << std::endl;
            remove(logFiles[i].first.c_str());
        }
    }
}

void LogSinkFileSip::OpenFile()
{
    MakeLogDir();

    DeleteOldestFile();

    if (!IsDiskAvailable()) {
        return;
    }

    std::string logFilePath = GetNewLogFilePath();
    if (currentFd_ >= 0) {
        CloseFile();
    }

    currentFd_ = open(logFilePath.c_str(), O_RDWR | O_CREAT | O_APPEND, S_IRUSR | S_IWUSR | S_IRGRP);
    if (currentFd_ < 0) {
        std::cout << "asdsip_log open " << logFilePath << " fail" << std::endl;
        if (remove(logFilePath.c_str()) == 0) {
            std::cerr << "Successfully removed residual file: " << logFilePath << std::endl;
        } else {
            std::cerr << "Failed to remove residual file: " << logFilePath << " - " << strerror(errno) << std::endl;
        }
    }
}

std::string LogSinkFileSip::GetNewLogFilePath()
{
    std::time_t tmpTime = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    struct tm* nowTime = std::localtime(&tmpTime);

    std::stringstream filePath;
    filePath << logDir_ << "/" << boostType_ << "_" << std::to_string(syscall(SYS_getpid)) << "_"
             << std::put_time(nowTime, "%Y%m%d%H%M%S") << ".log";
    return filePath.str();
}

bool LogSinkFileSip::IsDiskAvailable()
{
    struct statvfs vfs;
    if (statvfs(logDir_.c_str(), &vfs) == -1) {
        std::cout << "asdsip_log get current disk stats fail" << std::endl;
        return false;
    }

    uint64_t availableSize = vfs.f_bsize * vfs.f_bfree;
    if (availableSize <= DISK_AVAILABEL_LIMIT) {
        std::cout << "asdsip_log disk available space it too low, available size:" << availableSize
                  << ", limit size:" << DISK_AVAILABEL_LIMIT << std::endl;
        return false;
    }

    return true;
}

void LogSinkFileSip::MakeLogDir()
{
    struct stat st;
    if (stat(logDir_.c_str(), &st) == 0) { // 目录已经存在，就不创建
        return;
    }

    mode_t mode = S_IRWXU;
    uint32_t offset = 0;
    uint32_t pathLen = logDir_.size();
    do {
        const char* str = strchr(logDir_.c_str() + offset, '/');
        offset = (str == nullptr) ? pathLen : str - logDir_.c_str() + 1;
        std::string childDir = logDir_.substr(0, offset);
        if (stat(childDir.c_str(), &st) < 0) {
            std::cout << "asdsip_log mkdir " << childDir << std::endl;
            if (mkdir(childDir.c_str(), mode) < 0) {
                std::cerr << "Failed to create directory: " << childDir << ", error: " << strerror(errno) << std::endl;
                return;
            }
        }
    } while (offset != pathLen);
}

void LogSinkFileSip::CloseFile()
{
    // fd 0 是合法的文件描述符（如进程启动时 stdin 已关闭，open() 可能返回 0），
    // 需与其他非负 fd 一致处理，否则日志轮转与析构清理会失效（issue #108）
    if (currentFd_ >= 0) {
        (void)fchmod(currentFd_, S_IRUSR | S_IRGRP);
        close(currentFd_);
        currentFd_ = -1;
        currentFileSize_ = 0;
    }
}

std::string LogSinkFileSip::GetHomeDir()
{
    int bufsize;
    if ((bufsize = sysconf(_SC_GETPW_R_SIZE_MAX)) == -1) {
        return "";
    }

    char buffer[bufsize] = {0};
    struct passwd pwd = {};
    struct passwd* result = nullptr;
    // POSIX: getpwuid_r 返回 0 但 *result 为 NULL 表示用户未找到，此时 pwd 内容未定义。
    // 两种失败情况（返回非 0 / result 为 NULL）都必须视为错误，条件应为 || 而非 &&
    if (getpwuid_r(getuid(), &pwd, buffer, bufsize, &result) != 0 || result == nullptr) {
        return "";
    }

    return std::string(pwd.pw_dir);
}
} // namespace AsdSip
