# Maintainability and Measurability

<!-- md-trans-meta sourceCommit=cec88e057607a630073cce4bbace3c21f8d93fe7 translatedAt=2026-08-12T10:59:50.347Z pushedAt=2026-08-20T11:47:59.845Z -->

The signal processing acceleration library provides the following maintainability and measurability capabilities:

## Return Values

When a signal processing acceleration library operator API is called, the return values are as shown in the following table.

<table style="undefined;table-layout: fixed; width: 820px"><colgroup>
    <col style="width: 250px">
    <col style="width: 120px">
    <col style="width: 200px">
    <col style="width: 250px">
  </colgroup>
  <thead>
      <tr>
        <th>Status Code Name</th>
        <th>Status Code Value</th>
        <th>Error Code Description</th>
        <th>Fault Location Method</th>
      </tr></thead>
  <tbody>
    <tr>
      <td>ACL_SUCCESS</td>
      <td>0</td>
      <td>Execution successful.</td>
      <td>-</td>
    </tr>
    </tbody>
    <tbody>
    <tr>
      <td>ACL_ERROR_INVALID_PARAM</td>
      <td>100000</td>
      <td>Parameter verification failed.</td>
      <td>Check whether the input parameter values of the API are correct.</td>
    </tr>
    </tbody>
    <tbody>
    <tr>
      <td>ACL_ERROR_OP_INPUT_NOT_MATCH</td>
      <td>100021</td>
      <td>Single-operator input mismatch.</td>
      <td>Check whether the operator input is correct.</td>
    </tr>
    </tbody>
    <tbody>
    <tr>
      <td>ACL_ERROR_OP_OUTPUT_NOT_MATCH</td>
      <td>100022</td>
      <td>The output of a single operator does not match.</td>
      <td>Check whether the operator output is correct.</td>
    </tr>
    </tbody>
    <tbody>
    <tr>
      <td>ACL_ERROR_UNSUPPORTED_DATA_TYPE</td>
      <td>100026</td>
      <td>Unsupported data type.</td>
      <td>Check whether the data type exists or is currently supported.</td>
    </tr>
    </tbody>
    <tbody>
    <tr>
      <td>ACL_ERROR_FORMAT_NOT_MATCH</td>
      <td>100027</td>
      <td>Format mismatch.</td>
      <td>Check whether the format is correct.</td>
    </tr>
    </tbody>
    <tbody>
    <tr>
      <td>ACL_ERROR_API_NOT_SUPPORT</td>
      <td>200001</td>
      <td>The API is not supported.</td>
      <td>Check whether the called API is currently supported.</td>
    </tr>
    </tbody>
    <tbody>
    <tr>
      <td>ACL_ERROR_INTERNAL_ERROR</td>
      <td>500000</td>
      <td>Unknown internal error.</td>
      <td>-</td>
    </tr>
    </tbody>
    </table>

## Logging System

The logging system of the signal processing acceleration library supports log levels, output to standard output, and output to files.

- Log level

  The severity levels from high to low are ERROR, WARN, INFO, and DEBUG. The log level is controlled by the environment variable `ASCEND_GLOBAL_LOG_LEVEL`, and the default value is `INFO`.

   <table style="undefined;table-layout: fixed; width: 500px"><colgroup>
    <col style="width: 180px">
    <col style="width: 250px">
  </colgroup>
  <thead>
      <tr>
        <th>Level</th>
        <th>Description</th>
      </tr></thead>
  <tbody>
    <tr>
      <td>ERROR</td>
      <td>Error information. This level prints error and exception information.</td>
    </tr>
    </tbody>
    <tbody>
    <tr>
      <td>WARN</td>
      <td>Warning information, indicating situations where potential errors may occur and providing hints to developers.</td>
    </tr>
    </tbody>
    <tbody>
    <tr>
      <td>INFO (default)</td>
      <td>Data information. Prints operator and graph-related information. Users can learn the running status of the entire graph or a single operator by observing INFO logs.</td>
    </tr>
    </tbody>
    <tbody>
    <tr>
      <td>DEBUG</td>
      <td>Debug information. Prints detailed code information of the acceleration library. Acceleration library developers can debug framework code by viewing DEBUG logs.</td>
    </tr>
    </tbody>
    </table>

- Log storage

  - Log files are stored under `[LOG_PATH]/log/asdsip`.\

  `[LOG_PATH]` is controlled by the environment variable `ASCEND_PROCESS_LOG_PATH` (see [Environment Variable Reference](./environment_variable.md)), and the default value is `~/ascend`;

  - The log file naming format is `asdsip_[PID]_[YYYY][MM][DD][HH][MM][SS].log`.\

  `[PID]` is the thread ID. For example: `asdsip_253440_20231102065052.log`.

- Space management

  - Each log file is limited to a maximum size of 20 MB, with a maximum of 50 files stored. If the number of log files (log files stored with the standard naming format) in the current storage directory reaches the maximum, the earliest log file will be deleted based on its timestamp.

  - Before generating a log file, the available space in the log storage directory is checked. If the available space is less than 1 GB, no further log files will be generated.

## DumpTensor Capability

The DumpTensor feature of the signal processing acceleration library prints or saves the intermediate data generated during operator computation, or the input and output of the operator. It covers the following two scenarios: users using signal acceleration library operators, and custom computation flow scenarios.\
Scenario 1: When users use signal acceleration library operators in a service flow, the input or output of the signal acceleration library operators can be printed or saved to help users analyze or locate whether the computation results in the service flow are correct.\
Scenario 2: When calling signal acceleration library operators or custom computation flows on the C++ side, users can print or save data using C++ native functions. An example is as follows:

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
    // Boilerplate: Initialize ACL.
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
    // Call aclrtMalloc to allocate device-side memory.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // Call aclrtMemcpy to copy host-side data to device-side memory.
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
    // Compute the strides of a contiguous tensor.
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    // Call the aclCreateTensor API to create an aclTensor.
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
