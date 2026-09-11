/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef LOG_SINK_FILE_H
#define LOG_SINK_FILE_H

#include <fstream>
#include <mutex>
#include "log/log_sink_sip.h"

namespace AsdSip {
class LogSinkFileSip : public LogSinkSip {
public:
    LogSinkFileSip();
    ~LogSinkFileSip() override;
    void LogSip(const char* log, uint64_t logLen) override;

private:
    LogSinkFileSip(const LogSinkFileSip&) = delete;
    const LogSinkFileSip& operator=(const LogSinkFileSip&) = delete;
    void Init();
    void OpenFile();
    void DeleteOldestFile();
    std::string GetNewLogFilePath();
    bool IsDiskAvailable();
    void MakeLogDir();
    void CloseFile();
    std::string GetHomeDir();
    bool IsFileNameMatched(const std::string& fileName, std::string& createTime);
    ssize_t safeWriteAll(int fd, const void* buf, size_t count);

private:
    std::string boostType_;
    std::string logDir_;
    int currentFd_ = -1;
    uint64_t currentFileSize_ = 0;
    std::mutex mutex_;
};
} // namespace AsdSip
#endif
