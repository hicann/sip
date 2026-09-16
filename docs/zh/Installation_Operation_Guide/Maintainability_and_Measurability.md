# 可维可测能力

信号处理加速库当前提供以下可维可测能力：

## 返回值

调用信号处理加速库算子API时，接口返回值如下表所示。

<table style="undefined;table-layout: fixed; width: 820px"><colgroup>
    <col style="width: 250px">
    <col style="width: 120px">
    <col style="width: 200px">
    <col style="width: 250px">
  </colgroup>
  <thead>
      <tr>
        <th>状态码名称</th>
        <th>状态码值</th>
        <th>错误码说明</th>
        <th>故障定位方法</th>
      </tr></thead>
  <tbody>
    <tr>
      <td>ACL_SUCCESS</td>
      <td>0</td>
      <td>执行成功。</td>
      <td>-</td>
    </tr>
    </tbody>
    <tbody>
    <tr>
      <td>ACL_ERROR_INVALID_PARAM</td>
      <td>100000</td>
      <td>参数校验失败。</td>
      <td>请检查接口的入参值是否正确。</td>
    </tr>
    </tbody>
    <tbody>
    <tr>
      <td>ACL_ERROR_OP_INPUT_NOT_MATCH</td>
      <td>100021</td>
      <td>单算子的输入不匹配。</td>
      <td>请检查算子的输入是否正确。</td>
    </tr>
    </tbody>
    <tbody>
    <tr>
      <td>ACL_ERROR_OP_OUTPUT_NOT_MATCH</td>
      <td>100022</td>
      <td>单算子的输出不匹配。</td>
      <td>请检查算子的输出是否正确。</td>
    </tr>
    </tbody>
    <tbody>
    <tr>
      <td>ACL_ERROR_UNSUPPORTED_DATA_TYPE</td>
      <td>100026</td>
      <td>不支持的数据类型。</td>
      <td>请检查数据类型是否存在或当前是否支持。</td>
    </tr>
    </tbody>
    <tbody>
    <tr>
      <td>ACL_ERROR_FORMAT_NOT_MATCH</td>
      <td>100027</td>
      <td>Format不匹配。</td>
      <td>请检查Format是否正确。</td>
    </tr>
    </tbody>
    <tbody>
    <tr>
      <td>ACL_ERROR_API_NOT_SUPPORT</td>
      <td>200001</td>
      <td>接口不支持。</td>
      <td>请检查调用的接口当前是否支持。</td>
    </tr>
    </tbody>
    <tbody>
    <tr>
      <td>ACL_ERROR_INTERNAL_ERROR</td>
      <td>500000</td>
      <td>未知内部错误。</td>
      <td>-</td>
    </tr>
    </tbody>
    </table>

## 日志系统

信号处理加速库的日志系统支持日志分级、日志输出到标准输出、日志输出到文件。

- 日志分级

  日志的严重级别从高到低分为ERROR、WARN、INFO、DEBUG四个级别。日志级别由环境变量“ASCEND_GLOBAL_LOG_LEVEL”控制，默认为“INFO”。

   <table style="undefined;table-layout: fixed; width: 500px"><colgroup>
    <col style="width: 180px">
    <col style="width: 250px">
  </colgroup>
  <thead>
      <tr>
        <th>级别</th>
        <th>含义</th>
      </tr></thead>
  <tbody>
    <tr>
      <td>ERROR</td>
      <td>错误信息，该级别打印错误与异常信息。</td>
    </tr>
    </tbody>
    <tbody>
    <tr>
      <td>WARN</td>
      <td>警告信息，表明会出现潜在错误的情形，给开发者一些提示。</td>
    </tr>
    </tbody>
    <tbody>
    <tr>
      <td>INFO（默认）</td>
      <td>数据信息，打印算子与整图相关的信息，用户通过观察INFO日志就可以得知整图或单算子的运行状态。</td>
    </tr>
    </tbody>
    <tbody>
    <tr>
      <td>DEBUG</td>
      <td>调试信息，打印加速库代码细节信息，加速库开发者可以通过查看DEBUG日志来调试框架代码。</td>
    </tr>
    </tbody>
    </table>

- 日志保存
  - 日志文件保存在“[LOG_PATH]/log/asdsip”下。\
  [LOG_PATH]由环境变量（请参见[环境变量参考](./environment_variable.md)）ASCEND_PROCESS_LOG_PATH控制，默认为"~/ascend"；
  - 日志文件的命名格式为asdsip_[PID]_[年][月][日][时][分][秒].log。\
  [PID]为线程号。例如：asdsip_253440_20231102065052.log。

- 空间管理
  - 每个日志文件大小最大为20MB，最多存50个文件。如当前保存目录下的日志文件（以标准命名格式存储的日志文件）达到最高存储数量，将根据时间戳，删除最早时间的日志文件。
  - 在生成日志文件前，将会对日志保存目录的空间大小进行判断，如果空间不足1GB，将不会继续生成日志文件。

## DumpTensor能力

信号处理加速库Dump Tensor功能是在算子运行过程中，将算子计算过程中产生的中间数据，或算子的输入、输出进行打印或保存。具体包括下述两种场景：\
用户使用信号加速库算子、自定义计算流程场景\
用户使用信号加速库算子，在业务流程中，对信号加速库算子的输入或输出进行打印或保存，辅助用户分析或定位业务流程中的计算结果是否正确。\
cpp侧调用信号加速库算子、自定义计算流程
在cpp侧调用时，可基于c++自身的函数进行打印或者数据保存。示例如下：

```Cpp
#include <iostream>
#include <fstream>
#include <cmath>
#include <random>
#include <complex>
#include "asdsip.h"
#include "acl/acl.h"
#include "acl_meta.h"
using namespace AsdSip;
#define ASD_STATUS_CHECK(err)                                                \
    do {                                                                     \
        AsdSip::AspbStatus err_ = (err);                                     \
        if (err_ != AsdSip::NO_ERROR) {                                      \
            std::cout << "Execute failed." << std::endl; \
            exit(-1);                                                        \
        } else {                                                             \
            std::cout << "Execute successfully." << std::endl;               \
        }                                                                    \
    } while (0)
void printTensor(const std::complex<float> *tensorData, int64_t tensorSize)
{
    for (int64_t i = 0; i < tensorSize; i++) {
        std::cout << tensorData[i] << " ";
    }
    std::cout << std::endl;
}
#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)
#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)
int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}
int Init(int32_t deviceId, aclrtStream *stream)
{
    // 固定写法，acl初始化
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}
template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
    aclDataType dataType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // 调用aclrtMalloc申请device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // 调用aclrtMemcpy将host侧数据复制到device侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
    // 计算连续tensor的strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    // 调用aclCreateTensor接口创建aclTensor
    *tensor = aclCreateTensor(shape.data(),
        shape.size(),
        dataType,
        strides.data(),
        0,
        aclFormat::ACL_FORMAT_ND,
        shape.data(),
        shape.size(),
        *deviceAddr);
    return 0;
}
void printTensor(std::vector<std::complex<float>> tensorData, int64_t tensorSize)
{
    for (int64_t i = 0; i < tensorSize; i++) {
        std::cout << tensorData[i] << " ";
    }
    std::cout << std::endl;
}
int main(int argc, char **argv)
{
    int deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    int64_t n = 8;
    int64_t xSize = 8;
    int64_t ySize = 8;
    std::vector<std::complex<float>> tensorInXData;
    tensorInXData.reserve(xSize);
    for (int64_t i = 0; i < xSize; i++) {
        tensorInXData[i] = {2.0, (float)(1.0 + i)};
    }
    std::vector<std::complex<float>> tensorInYData;
    tensorInYData.reserve(ySize);
    for (int64_t i = 0; i < ySize; i++) {
        tensorInYData[i] = {3.0, 4.0};
    }
    int64_t resultSize = 1;
    std::vector<std::complex<float>> resultData;
    resultData.reserve(resultSize);
    std::cout << "------- input TensorInX -------" << std::endl;
    printTensor(tensorInXData.data(), xSize);
    std::cout << "------- input TensorInY -------" << std::endl;
    printTensor(tensorInYData.data(), ySize);
    std::vector<int64_t> xShape = {xSize};
    std::vector<int64_t> yShape = {ySize};
    std::vector<int64_t> resultShape = {resultSize};
    aclTensor *inputX = nullptr;
    aclTensor *inputY = nullptr;
    aclTensor *result = nullptr;
    void *inputXDeviceAddr = nullptr;
    void *inputYDeviceAddr = nullptr;
    void *resultDeviceAddr = nullptr;
    ret = CreateAclTensor(tensorInXData, xShape, &inputXDeviceAddr, aclDataType::ACL_COMPLEX64, &inputX);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(tensorInYData, yShape, &inputYDeviceAddr, aclDataType::ACL_COMPLEX64, &inputY);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(resultData, resultShape, &resultDeviceAddr, aclDataType::ACL_COMPLEX64, &result);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    asdBlasHandle handle;
    asdBlasCreate(handle);
    size_t lwork = 0;
    void *buffer = nullptr;
    asdBlasMakeDotPlan(handle);
    asdBlasGetWorkspaceSize(handle, &lwork);
    std::cout << "lwork = " << lwork << std::endl;
    if (lwork > 0) {
        ret = aclrtMalloc(&buffer, static_cast<int64_t>(lwork), ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    asdBlasSetWorkspace(handle, buffer);
    asdBlasSetStream(handle, stream);
    ASD_STATUS_CHECK(asdBlasCdotu(handle, n, inputX, 1, inputY, 1, result));
    asdBlasSynchronize(handle);
    asdBlasDestroy(handle);
    ret = aclrtMemcpy(resultData.data(),
        resultSize * sizeof(std::complex<float>),
        resultDeviceAddr,
        resultSize * sizeof(std::complex<float>),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    std::cout << "------- result -------" << std::endl;
    printTensor(resultData.data(), resultSize);
    std::ofstream file("result.bin", std::ios::binary | std::ios::out);
    file.write((const char *)resultData.data(), sizeof(std::complex<float>) * resultSize);
    file.close();
    std::cout << "result.bin saved." << std::endl;
    aclDestroyTensor(inputX);
    aclDestroyTensor(inputY);
    aclDestroyTensor(result);
    aclrtFree(inputXDeviceAddr);
    aclrtFree(inputYDeviceAddr);
    aclrtFree(resultDeviceAddr);
    if (lwork > 0) {
        aclrtFree(buffer);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
