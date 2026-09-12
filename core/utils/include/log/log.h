/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASDSIP_LOG_H
#define ASDSIP_LOG_H

#ifdef UNITTEST

#include <mki/utils/log/log_stream.h>
#include <mki/utils/log/log_core.h>
#include <mki/utils/log/log_sink.h>
#include <mki/utils/log/log_entity.h>

#else

#include "log/log_stream_sip.h"
#include "log/log_core_sip.h"
#include "log/log_sink_sip.h"
#include "log/log_entity.h"

#endif

#ifdef UNITTEST

#define ASDSIP_LOG(level) ASDSIP_LOG_##level
#define ASDSIP_ELOG(err_code)                                           \
    if (Mki::LogLevel::ERROR >= Mki::LogCore::Instance().GetLogLevel()) \
    Mki::LogStream(__FILE__, __LINE__, __FUNCTION__, Mki::LogLevel::ERROR)

#define ASDSIP_FLOG(level, format, ...) ASDSIP_FLOG_##level(format, __VA_ARGS__)

#define ASDSIP_LOG_IF(condition, level) \
    if (condition)                      \
    ASDSIP_LOG(level)

#define ASDSIP_LOG_TRACE                                                \
    if (Mki::LogLevel::TRACE >= Mki::LogCore::Instance().GetLogLevel()) \
    Mki::LogStream(__FILE__, __LINE__, __FUNCTION__, Mki::LogLevel::TRACE)
#define ASDSIP_LOG_DEBUG                                                \
    if (Mki::LogLevel::DEBUG >= Mki::LogCore::Instance().GetLogLevel()) \
    Mki::LogStream(__FILE__, __LINE__, __FUNCTION__, Mki::LogLevel::DEBUG)
#define ASDSIP_LOG_INFO                                                \
    if (Mki::LogLevel::INFO >= Mki::LogCore::Instance().GetLogLevel()) \
    Mki::LogStream(__FILE__, __LINE__, __FUNCTION__, Mki::LogLevel::INFO)
#define ASDSIP_LOG_WARN                                                \
    if (Mki::LogLevel::WARN >= Mki::LogCore::Instance().GetLogLevel()) \
    Mki::LogStream(__FILE__, __LINE__, __FUNCTION__, Mki::LogLevel::WARN)
#define ASDSIP_LOG_ERROR                                                \
    if (Mki::LogLevel::ERROR >= Mki::LogCore::Instance().GetLogLevel()) \
    Mki::LogStream(__FILE__, __LINE__, __FUNCTION__, Mki::LogLevel::ERROR)
#define ASDSIP_LOG_FATAL                                                \
    if (Mki::LogLevel::FATAL >= Mki::LogCore::Instance().GetLogLevel()) \
    Mki::LogStream(__FILE__, __LINE__, __FUNCTION__, Mki::LogLevel::FATAL)

#define ASDSIP_FLOG_TRACE(format, ...)                                  \
    if (Mki::LogLevel::TRACE >= Mki::LogCore::Instance().GetLogLevel()) \
    Mki::LogStream(__FILE__, __LINE__, __FUNCTION__, Mki::LogLevel::TRACE).Format(format, __VA_ARGS__)
#define ASDSIP_FLOG_DEBUG(format, ...)                                  \
    if (Mki::LogLevel::DEBUG >= Mki::LogCore::Instance().GetLogLevel()) \
    Mki::LogStream(__FILE__, __LINE__, __FUNCTION__, Mki::LogLevel::DEBUG).Format(format, __VA_ARGS__)
#define ASDSIP_FLOG_INFO(format, ...)                                  \
    if (Mki::LogLevel::INFO >= Mki::LogCore::Instance().GetLogLevel()) \
    Mki::LogStream(__FILE__, __LINE__, __FUNCTION__, Mki::LogLevel::INFO).Format(format, __VA_ARGS__)
#define ASDSIP_FLOG_WARN(format, ...)                                  \
    if (Mki::LogLevel::WARN >= Mki::LogCore::Instance().GetLogLevel()) \
    Mki::LogStream(__FILE__, __LINE__, __FUNCTION__, Mki::LogLevel::WARN).Format(format, __VA_ARGS__)
#define ASDSIP_FLOG_ERROR(format, ...)                                  \
    if (Mki::LogLevel::ERROR >= Mki::LogCore::Instance().GetLogLevel()) \
    Mki::LogStream(__FILE__, __LINE__, __FUNCTION__, Mki::LogLevel::ERROR).Format(format, __VA_ARGS__)
#define ASDSIP_FLOG_FATAL(format, ...)                                  \
    if (Mki::LogLevel::FATAL >= Mki::LogCore::Instance().GetLogLevel()) \
    Mki::LogStream(__FILE__, __LINE__, __FUNCTION__, Mki::LogLevel::FATAL).Format(format, __VA_ARGS__)

#else

#define ASDSIP_LOG(level) ASDSIP_LOG_##level

#define ASDSIP_ELOG(err_code)                                                       \
    if (AsdSip::LogLevel::ERROR >= AsdSip::LogCoreSip::InstanceSip().GetLogLevel()) \
    AsdSip::LogStreamSip(__FILE__, __LINE__, __FUNCTION__, AsdSip::LogLevel::ERROR, err_code)

#define ASDSIP_FLOG(level, format, ...) ASDSIP_FLOG_##level(format, __VA_ARGS__)

#define ASDSIP_LOG_IF(condition, level) \
    if (condition)                      \
    ASDSIP_LOG(level)

#define ASDSIP_LOG_DEBUG                                                            \
    if (AsdSip::LogLevel::DEBUG >= AsdSip::LogCoreSip::InstanceSip().GetLogLevel()) \
    AsdSip::LogStreamSip(__FILE__, __LINE__, __FUNCTION__, AsdSip::LogLevel::DEBUG)
#define ASDSIP_LOG_INFO                                                            \
    if (AsdSip::LogLevel::INFO >= AsdSip::LogCoreSip::InstanceSip().GetLogLevel()) \
    AsdSip::LogStreamSip(__FILE__, __LINE__, __FUNCTION__, AsdSip::LogLevel::INFO)
#define ASDSIP_LOG_WARN                                                            \
    if (AsdSip::LogLevel::WARN >= AsdSip::LogCoreSip::InstanceSip().GetLogLevel()) \
    AsdSip::LogStreamSip(__FILE__, __LINE__, __FUNCTION__, AsdSip::LogLevel::WARN)
#define ASDSIP_LOG_ERROR                                                            \
    if (AsdSip::LogLevel::ERROR >= AsdSip::LogCoreSip::InstanceSip().GetLogLevel()) \
    AsdSip::LogStreamSip(__FILE__, __LINE__, __FUNCTION__, AsdSip::LogLevel::ERROR)
#define ASDSIP_LOG_FATAL                                                            \
    if (AsdSip::LogLevel::FATAL >= AsdSip::LogCoreSip::InstanceSip().GetLogLevel()) \
    AsdSip::LogStreamSip(__FILE__, __LINE__, __FUNCTION__, AsdSip::LogLevel::FATAL)

#define ASDSIP_FLOG_DEBUG(format, ...)                                              \
    if (AsdSip::LogLevel::DEBUG >= AsdSip::LogCoreSip::InstanceSip().GetLogLevel()) \
    AsdSip::LogStreamSip(__FILE__, __LINE__, __FUNCTION__, AsdSip::LogLevel::DEBUG).FormatSip(format, __VA_ARGS__)
#define ASDSIP_FLOG_INFO(format, ...)                                              \
    if (AsdSip::LogLevel::INFO >= AsdSip::LogCoreSip::InstanceSip().GetLogLevel()) \
    AsdSip::LogStreamSip(__FILE__, __LINE__, __FUNCTION__, AsdSip::LogLevel::INFO).FormatSip(format, __VA_ARGS__)
#define ASDSIP_FLOG_WARN(format, ...)                                              \
    if (AsdSip::LogLevel::WARN >= AsdSip::LogCoreSip::InstanceSip().GetLogLevel()) \
    AsdSip::LogStreamSip(__FILE__, __LINE__, __FUNCTION__, AsdSip::LogLevel::WARN).FormatSip(format, __VA_ARGS__)
#define ASDSIP_FLOG_ERROR(format, ...)                                              \
    if (AsdSip::LogLevel::ERROR >= AsdSip::LogCoreSip::InstanceSip().GetLogLevel()) \
    AsdSip::LogStreamSip(__FILE__, __LINE__, __FUNCTION__, AsdSip::LogLevel::ERROR).FormatSip(format, __VA_ARGS__)
#define ASDSIP_FLOG_FATAL(format, ...)                                              \
    if (AsdSip::LogLevel::FATAL >= AsdSip::LogCoreSip::InstanceSip().GetLogLevel()) \
    AsdSip::LogStreamSip(__FILE__, __LINE__, __FUNCTION__, AsdSip::LogLevel::FATAL).FormatSip(format, __VA_ARGS__)
#endif

#endif
