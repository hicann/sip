# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import argparse
import os
import re


def detect_encoding(file_path):
    """用标准库试探文件编码，替代 chardet 依赖"""
    with open(file_path, "rb") as f:
        raw = f.read()
    for enc in ("utf-8", "gbk", "latin-1"):
        try:
            raw.decode(enc)
            return enc
        except (UnicodeDecodeError, LookupError):
            continue
    return "latin-1"


def FilterCommentsStringMacro(codeLine):
    regMacro = r"\s*#.*$"
    regComments = r"/\*.*?\*/|//.*$"
    regString1 = r'\\"'
    regString2 = r'"[^"]*"'
    codeLine = re.sub(regMacro, "", codeLine)
    codeLine = re.sub(regComments, "", codeLine)
    codeLine = re.sub(regString1, "", codeLine)
    codeLine = re.sub(regString2, "", codeLine)
    return codeLine


def FilterTemplates(codeLine):
    regPrefix = r"\b(const|volatile|typename)\b"
    codeLine = re.sub(regPrefix, "", codeLine)
    regTemp = (
        r"<\s*(::)?\w+\s*(::\s*\w+\s*)*(,\s*(::)?\w+\s*(::\s*\w+\s*)*)*([*&]\s*)?>"
    )
    i = 0
    while i < 50:
        currentLine = codeLine
        codeLine = re.sub(regTemp, "", codeLine)
        if codeLine == currentLine:
            break
        i += 1
    regSpecial = r"(->|>>|<<|::)"
    codeLine = re.sub(regSpecial, "", codeLine)
    return codeLine


def WhetherConditionals(codeLine):
    regulation = r"([?|!~><]|&&|==|!=|\b(if|switch|case|while|for)\b)"
    newCodeLine = re.sub(regulation, "", codeLine)
    if len(newCodeLine) < len(codeLine):
        return True
    else:
        return False


class BranchFilter:
    def __init__(self, debug):
        self.iter_ = 0
        self.limit_ = 5
        self.macroList = []
        self.debugSwitch = int(debug)
        self.dirDirtyWords = [
            "/llt/",
            "/ai/",
            "/vector/",
            "/x86/",
            "/sve/dev/operators/",
            "/test/",
            "/adapter/",
        ]

    def DEBUG_LOG(self, args):
        if self.debugSwitch:
            print("[DEBUG] ", args)

    def LoadFile(self, cppFile):
        # 用标准库检测文件编码（替代 chardet 依赖）
        encoding = detect_encoding(cppFile)

        # 用正确的编码打开文件
        with open(cppFile, "r", encoding=encoding) as fw:
            fileLineList = fw.readlines()
            return fileLineList

    def FindAllMacros(self, codeLines):
        for line in codeLines:
            line = line[0:-1]
            regMacro = r"\s*#define\s*"
            newLine = re.sub(regMacro, "", line)
            if len(newLine) < len(line):
                regSpace = r"\s*\\"
                newLine = re.sub(regSpace, "", newLine)
                regBraces = r"\(.*\)"
                newLine = re.sub(regBraces, "", newLine)
                self.DEBUG_LOG(newLine)
                self.macroList.append(newLine)

    def FindAllCodeFiles(self, rootDir):
        codeFileList = []
        self.DEBUG_LOG(rootDir)
        for lists in os.listdir(rootDir):
            path = os.path.join(rootDir, lists)
            if os.path.isdir(path):
                if path.find("/.git") != -1 or path.find("/build") != -1:
                    continue
                for dirWord in self.dirDirtyWords:
                    if path.find(dirWord) != -1:
                        self.DEBUG_LOG("Pass dir : " + path)
                        break
                else:
                    codeFileList += self.FindAllCodeFiles(path)
            else:
                if (
                    path.endswith(".cpp")
                    or path.endswith(".cc")
                    or path.endswith(".c")
                    or path.endswith(".h")
                ):
                    if path.find("/test/") == -1:
                        codeFileList.append(path)
                        self.DEBUG_LOG("Add code file : " + path)
        return codeFileList

    def GetBranchInfo(self, infoLine):
        infoParts = re.split(r"[,:]", infoLine)
        cppLineNumber = int(infoParts[1])
        return cppLineNumber

    def BranchInfoFilterMacros(self, infoLine, cppLines):
        infoParts = re.split(r"[,:]", infoLine)
        cppLineNumber = int(infoParts[1])

        # load cpp file and find macro in it
        cppLine = cppLines[cppLineNumber - 1]
        cppLine = FilterCommentsStringMacro(cppLine)
        cppLine = FilterTemplates(cppLine)
        self.DEBUG_LOG(cppLine)
        for macro in self.macroList:
            if re.match(".*" + macro + ".*", cppLine):
                return True

        if WhetherConditionals(cppLine):
            return False

        # pass the branch info which is related to template
        if re.match(r".*<.*>.*", cppLine):
            self.DEBUG_LOG("find template ")
            return True

        return False

    def ProcessOneLine(self, infoLines, cppLines, newInfoData):
        infoLine = infoLines[self.iter_]
        infoParts = re.split(r"[,:]", infoLine)
        cppLineNumber = int(infoParts[1])
        currentCppLineNumber = cppLineNumber
        Flag = self.BranchInfoFilterMacros(infoLines[self.iter_], cppLines)

        """
        Conditions above decides whether this infoLine should be push_back
        """
        while cppLineNumber == currentCppLineNumber:
            self.DEBUG_LOG(infoLines[self.iter_][0:-1])
            # push_back or not
            if Flag:
                newInfoData += infoLines[self.iter_]
            self.iter_ += 1
            cppLineNumber = self.GetBranchInfo(infoLines[self.iter_][0:-1])

        return newInfoData

    def ProcessOneFile(self, infoLines, line, newInfoData):
        """
        load the cpp file and delete some branches in info file
        """
        fileAbsPath = line[3:-1]
        cppLines = self.LoadFile(fileAbsPath)

        while infoLines[self.iter_][0:13] != "end_of_record":
            prefix = infoLines[self.iter_][0:4]
            if prefix == "BRDA":  # prefix 'BRF:' or prefix 'BRH:' is not useful!
                newInfoData = self.ProcessOneLine(infoLines, cppLines, newInfoData)
            else:
                # not BRDA, just push back
                newInfoData += infoLines[self.iter_]
                self.iter_ += 1
        return newInfoData

    def MainLoopInfo(self, infoLines, newInfoFile):
        newInfoData = ""

        while self.iter_ < len(infoLines):
            line = infoLines[self.iter_]
            if self.WhetherCounterNewFile(line):
                self.DEBUG_LOG("[INFO] Process File in:" + line[3:-1])
                newInfoData = self.ProcessOneFile(infoLines, line, newInfoData)
            else:
                self.iter_ += 1
                newInfoData += line

        with open(newInfoFile, "w") as fw:
            fw.write(newInfoData)

    def Filter(self, inputFile, outFile, rootDir):
        self.FindAllCodeFiles(rootDir)
        infoLines = self.LoadFile(inputFile)
        self.MainLoopInfo(infoLines, outFile)

    def WhetherCounterNewFile(self, line):
        if line[0:3] == "SF:":
            return True
        else:
            return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input_info_file",
        dest="infoFile",
        type=str,
        required=True,
        help="input info file",
    )
    parser.add_argument(
        "-o",
        "--output_info_file",
        dest="outFile",
        type=str,
        required=True,
        help="output info file",
    )
    parser.add_argument(
        "-s",
        "--source_dir",
        dest="sourceDir",
        type=str,
        required=True,
        help="the root directory of source code",
    )
    parser.add_argument(
        "-d",
        "--debug",
        dest="debug",
        type=str,
        required=False,
        default="0",
        help="debug mode",
    )
    args = parser.parse_args()

    obj = BranchFilter(args.debug)
    obj.FindAllMacros(obj.LoadFile(args.infoFile))
    obj.Filter(args.infoFile, args.outFile, args.sourceDir)


if __name__ == "__main__":
    main()
