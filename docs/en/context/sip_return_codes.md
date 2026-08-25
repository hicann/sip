# SiP Return Codes

<!-- md-trans-meta sourceCommit=4a22b17b608bcc429ba2f8a5211d79cc753fb40f translatedAt=2026-08-12T10:57:55.620Z pushedAt=2026-08-20T11:47:59.806Z -->

When calling the signal processing acceleration library operator APIs, the return values are as shown in the following table.

**Table 1** Return values

<a name="zh-cn_topic_0000001563019104_table8155243135018"></a>

<table><thead align="left"><tr id="zh-cn_topic_0000001563019104_row111561243135019"><th class="cellrowborder" valign="top" width="30.543054305430545%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0000001563019104_p6676115185014"><a name="zh-cn_topic_0000001563019104_p6676115185014"></a><a name="zh-cn_topic_0000001563019104_p6676115185014"></a>Status Code Name</p></th>
<th class="cellrowborder" valign="top" width="15.971597159715973%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0000001563019104_p16690195185015"><a name="zh-cn_topic_0000001563019104_p16690195185015"></a><a name="zh-cn_topic_0000001563019104_p16690195185015"></a>Status Code Value</p></th>
<th class="cellrowborder" valign="top" width="53.48534853485349%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0000001563019104_p107021951145010"><a name="zh-cn_topic_0000001563019104_p107021951145010"></a><a name="zh-cn_topic_0000001563019104_p107021951145010"></a>Status Code Description</p></th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001563019104_row2015624345019"><td class="cellrowborder" valign="top" width="30.543054305430545%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001563019104_p45716143512"><a name="zh-cn_topic_0000001563019104_p45716143512"></a><a name="zh-cn_topic_0000001563019104_p45716143512"></a>ACL_SUCCESS</p></td>
<td class="cellrowborder" valign="top" width="15.971597159715973%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001563019104_p205761419512"><a name="zh-cn_topic_0000001563019104_p205761419512"></a><a name="zh-cn_topic_0000001563019104_p205761419512"></a>0</p></td>
<td class="cellrowborder" valign="top" width="53.48534853485349%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001563019104_p95741410511"><a name="zh-cn_topic_0000001563019104_p95741410511"></a><a name="zh-cn_topic_0000001563019104_p95741410511"></a>Execution successful.</p></td>
</tr>
<tr id="zh-cn_topic_0000001563019104_row9156144365013"><td class="cellrowborder" valign="top" width="30.543054305430545%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001563019104_p14704133965112"><a name="zh-cn_topic_0000001563019104_p14704133965112"></a><a name="zh-cn_topic_0000001563019104_p14704133965112"></a>ACL_ERROR_INVALID_PARAM</p></td>
<td class="cellrowborder" valign="top" width="15.971597159715973%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001563019104_p1156543125020"><a name="zh-cn_topic_0000001563019104_p1156543125020"></a><a name="zh-cn_topic_0000001563019104_p1156543125020"></a>100000</p></td>
<td class="cellrowborder" valign="top" width="53.48534853485349%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001563019104_p1015624311507"><a name="zh-cn_topic_0000001563019104_p1015624311507"></a><a name="zh-cn_topic_0000001563019104_p1015624311507"></a>Parameter verification failed.</p></td>
</tr>
<tr id="zh-cn_topic_0000001563019104_row315644318505"><td class="cellrowborder" valign="top" width="30.543054305430545%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001563019104_p11156144312509"><a name="zh-cn_topic_0000001563019104_p11156144312509"></a><a name="zh-cn_topic_0000001563019104_p11156144312509"></a>ACL_ERROR_OP_INPUT_NOT_MATCH</p></td>
<td class="cellrowborder" valign="top" width="15.971597159715973%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001563019104_p915619437501"><a name="zh-cn_topic_0000001563019104_p915619437501"></a><a name="zh-cn_topic_0000001563019104_p915619437501"></a>100021</p></td>
<td class="cellrowborder" valign="top" width="53.48534853485349%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001563019104_p17570123425314"><a name="zh-cn_topic_0000001563019104_p17570123425314"></a><a name="zh-cn_topic_0000001563019104_p17570123425314"></a>Single-operator input mismatch..</p></td>
</tr>
<tr id="zh-cn_topic_0000001563019104_row1215674375018"><td class="cellrowborder" valign="top" width="30.543054305430545%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001563019104_p11156174305019"><a name="zh-cn_topic_0000001563019104_p11156174305019"></a><a name="zh-cn_topic_0000001563019104_p11156174305019"></a>ACL_ERROR_OP_OUTPUT_NOT_MATCH</p></td>
<td class="cellrowborder" valign="top" width="15.971597159715973%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001563019104_p9156443185011"><a name="zh-cn_topic_0000001563019104_p9156443185011"></a><a name="zh-cn_topic_0000001563019104_p9156443185011"></a>100022</p></td>
<td class="cellrowborder" valign="top" width="53.48534853485349%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001563019104_p4156543185018"><a name="zh-cn_topic_0000001563019104_p4156543185018"></a><a name="zh-cn_topic_0000001563019104_p4156543185018"></a>The output of a single operator does not match.</p></td>
</tr>
<tr id="zh-cn_topic_0000001563019104_row11561143115015"><td class="cellrowborder" valign="top" width="30.543054305430545%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001563019104_p107381545195210"><a name="zh-cn_topic_0000001563019104_p107381545195210"></a><a name="zh-cn_topic_0000001563019104_p107381545195210"></a>ACL_ERROR_UNSUPPORTED_DATA_TYPE</p></td>
<td class="cellrowborder" valign="top" width="15.971597159715973%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001563019104_p14156144313508"><a name="zh-cn_topic_0000001563019104_p14156144313508"></a><a name="zh-cn_topic_0000001563019104_p14156144313508"></a>100026</p></td>
<td class="cellrowborder" valign="top" width="53.48534853485349%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001563019104_p10850181263218"><a name="zh-cn_topic_0000001563019104_p10850181263218"></a><a name="zh-cn_topic_0000001563019104_p10850181263218"></a>Unsupported data type.</p></td>
</tr>
<tr id="zh-cn_topic_0000001563019104_row11561143115015"><td class="cellrowborder" valign="top" width="30.543054305430545%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001563019104_p107381545195210"><a name="zh-cn_topic_0000001563019104_p107381545195210"></a><a name="zh-cn_topic_0000001563019104_p107381545195210"></a>ACL_ERROR_FORMAT_NOT_MATCH</p></td>
<td class="cellrowborder" valign="top" width="15.971597159715973%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001563019104_p14156144313508"><a name="zh-cn_topic_0000001563019104_p14156144313508"></a><a name="zh-cn_topic_0000001563019104_p14156144313508"></a>100027</p></td>
<td class="cellrowborder" valign="top" width="53.48534853485349%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001563019104_p10850181263218"><a name="zh-cn_topic_0000001563019104_p10850181263218"></a><a name="zh-cn_topic_0000001563019104_p10850181263218"></a>Format mismatch.</p></td>
</tr>
<tr id="zh-cn_topic_0000001563019104_row11561143115015"><td class="cellrowborder" valign="top" width="30.543054305430545%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001563019104_p107381545195210"><a name="zh-cn_topic_0000001563019104_p107381545195210"></a><a name="zh-cn_topic_0000001563019104_p107381545195210"></a>ACL_ERROR_API_NOT_SUPPORT</p></td>
<td class="cellrowborder" valign="top" width="15.971597159715973%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001563019104_p14156144313508"><a name="zh-cn_topic_0000001563019104_p14156144313508"></a><a name="zh-cn_topic_0000001563019104_p14156144313508"></a>200001</p></td>
<td class="cellrowborder" valign="top" width="53.48534853485349%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001563019104_p10850181263218"><a name="zh-cn_topic_0000001563019104_p10850181263218"></a><a name="zh-cn_topic_0000001563019104_p10850181263218"></a>The API is not supported.</p></td>
</tr>
<tr id="zh-cn_topic_0000001563019104_row11561143115015"><td class="cellrowborder" valign="top" width="30.543054305430545%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001563019104_p107381545195210"><a name="zh-cn_topic_0000001563019104_p107381545195210"></a><a name="zh-cn_topic_0000001563019104_p107381545195210"></a>ACL_ERROR_INTERNAL_ERROR</p></td>
<td class="cellrowborder" valign="top" width="15.971597159715973%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001563019104_p14156144313508"><a name="zh-cn_topic_0000001563019104_p14156144313508"></a><a name="zh-cn_topic_0000001563019104_p14156144313508"></a>500000</p></td>
<td class="cellrowborder" valign="top" width="53.48534853485349%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001563019104_p95741410511"><a name="zh-cn_topic_0000001563019104_p95741410511"></a><a name="zh-cn_topic_0000001563019104_p95741410511"></a>Internal unknown error.</p></td>
</tr>
</tbody>
</table>

# Logging System

The logging system of the signal processing acceleration library supports log levels, output to standard output, and output to log files.

- **Log level:**

The severity levels of logs, from high to low, are ERROR, WARN, INFO, and DEBUG, as shown in the following table. The log level is controlled by the environment variable `ASCEND_GLOBAL_LOG_LEVEL`, and the default value is `INFO`.
**Table 2** Log levels

<a name="zh-cn_topic_0000001563019104_table8155243135018"></a>

<table><thead align="left"><tr id="zh-cn_topic_0000001563019104_row111561243135019"><th class="cellrowborder" valign="top" width="30.543054305430545%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0000001563019104_p6676115185014"><a name="zh-cn_topic_0000001563019104_p6676115185014"></a><a name="zh-cn_topic_0000001563019104_p6676115185014"></a>Level</p></th>
<th class="cellrowborder" valign="top" width="15.971597159715973%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0000001563019104_p16690195185015"><a name="zh-cn_topic_0000001563019104_p16690195185015"></a><a name="zh-cn_topic_0000001563019104_p16690195185015"></a>Meaning</p></th>
</tr>
</thead>
<tbody>
<tr id="zh-cn_topic_0000001563019104_row9156144365013"><td class="cellrowborder" valign="top" width="30.543054305430545%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001563019104_p14704133965112"><a name="zh-cn_topic_0000001563019104_p14704133965112"></a><a name="zh-cn_topic_0000001563019104_p14704133965112"></a>ERROR</p></td>
<td class="cellrowborder" valign="top" width="15.971597159715973%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001563019104_p1156543125020"><a name="zh-cn_topic_0000001563019104_p1156543125020"></a><a name="zh-cn_topic_0000001563019104_p1156543125020"></a>Error information. This level prints error and exception information.</p></td>
</tr>
<tr id="zh-cn_topic_0000001563019104_row9156144365013"><td class="cellrowborder" valign="top" width="30.543054305430545%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001563019104_p14704133965112"><a name="zh-cn_topic_0000001563019104_p14704133965112"></a><a name="zh-cn_topic_0000001563019104_p14704133965112"></a>WARN</p></td>
<td class="cellrowborder" valign="top" width="15.971597159715973%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001563019104_p1156543125020"><a name="zh-cn_topic_0000001563019104_p1156543125020"></a><a name="zh-cn_topic_0000001563019104_p1156543125020"></a>Warning information, indicating potential error conditions and providing hints to developers.</p></td>
</tr>
<tr id="zh-cn_topic_0000001563019104_row9156144365013"><td class="cellrowborder" valign="top" width="30.543054305430545%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001563019104_p14704133965112"><a name="zh-cn_topic_0000001563019104_p14704133965112"></a><a name="zh-cn_topic_0000001563019104_p14704133965112"></a>INFO (default)</p></td>
<td class="cellrowborder" valign="top" width="15.971597159715973%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001563019104_p1156543125020"><a name="zh-cn_topic_0000001563019104_p1156543125020"></a><a name="zh-cn_topic_0000001563019104_p1156543125020"></a>Data information, printing information related to operators and the entire graph. By observing INFO logs, users can learn the running status of the entire graph or individual operators.</p></td>
</tr>
<tr id="zh-cn_topic_0000001563019104_row9156144365013"><td class="cellrowborder" valign="top" width="30.543054305430545%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001563019104_p14704133965112"><a name="zh-cn_topic_0000001563019104_p14704133965112"></a><a name="zh-cn_topic_0000001563019104_p14704133965112"></a>DEBUG</p></td>
<td class="cellrowborder" valign="top" width="15.971597159715973%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001563019104_p1156543125020"><a name="zh-cn_topic_0000001563019104_p1156543125020"></a><a name="zh-cn_topic_0000001563019104_p1156543125020"></a>Debug information, printing detailed code information of the acceleration library. Acceleration library developers can debug framework code by viewing DEBUG logs.</p></td>
</tr>
</tbody>
</table>

- **Log storage:**

Log files are stored under `[LOG_PATH]/log/asdsip`.
   <ul><li><code>[LOG_PATH]</code> is controlled by the environment variable <code>ASCEND_PROCESS_LOG_PATH</code>, and the default value is <code>~/ascend</code>.</li><li>The log file naming format is <code>asdsip_[PID]_[YYYY][MM][DD][HH][MM][SS].log</code>, where <code>[PID]</code> is the process ID. For example: <code>asdsip_253440_20231102065052.log</code>.</li></ul>

- **Space management:**

   <ul><li>Each log file has a maximum size of 20 MB, and up to 50 files can be stored. If the number of log files (stored in the standard naming format) in the current storage directory reaches the maximum storage count, the earliest log file will be deleted based on its timestamp.</li><li>Before generating a log file, the available space in the log storage directory is checked. If the available space is less than 1 GB, no more log files will be generated.</li></ul>

# DumpTensor Capability

The DumpTensor feature of the signal processing acceleration library prints or saves the intermediate data generated during operator computation, or the input and output of the operator. It covers the following two scenarios: users using signal acceleration library operators, and custom computation flow scenarios.

Scenario 1: When users use signal acceleration library operators in a service flow, the input or output of the signal acceleration library operators can be printed or saved to help users analyze or locate whether the computation results in the service flow are correct.
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
    // Copy the host-side data to the device-side memory using aclrtMemcpy.
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
    // Compute the strides of a contiguous tensor.
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    // Create an aclTensor by calling the aclCreateTensor API.
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
