# SiP返回码

调用信号处理加速库算子API时，接口返回值如下表所示。

**表 1**  返回状态码

<a name="zh-cn_topic_0000001563019104_table8155243135018"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001563019104_row111561243135019"><th class="cellrowborder" valign="top" width="30.543054305430545%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0000001563019104_p6676115185014"><a name="zh-cn_topic_0000001563019104_p6676115185014"></a><a name="zh-cn_topic_0000001563019104_p6676115185014"></a>状态码名称</p>
</th>
<th class="cellrowborder" valign="top" width="15.971597159715973%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0000001563019104_p16690195185015"><a name="zh-cn_topic_0000001563019104_p16690195185015"></a><a name="zh-cn_topic_0000001563019104_p16690195185015"></a>状态码值</p>
</th>
<th class="cellrowborder" valign="top" width="53.48534853485349%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0000001563019104_p107021951145010"><a name="zh-cn_topic_0000001563019104_p107021951145010"></a><a name="zh-cn_topic_0000001563019104_p107021951145010"></a>状态码说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001563019104_row2015624345019"><td class="cellrowborder" valign="top" width="30.543054305430545%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001563019104_p45716143512"><a name="zh-cn_topic_0000001563019104_p45716143512"></a><a name="zh-cn_topic_0000001563019104_p45716143512"></a>ACL_SUCCESS</p>
</td>
<td class="cellrowborder" valign="top" width="15.971597159715973%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001563019104_p205761419512"><a name="zh-cn_topic_0000001563019104_p205761419512"></a><a name="zh-cn_topic_0000001563019104_p205761419512"></a>0</p>
</td>
<td class="cellrowborder" valign="top" width="53.48534853485349%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001563019104_p95741410511"><a name="zh-cn_topic_0000001563019104_p95741410511"></a><a name="zh-cn_topic_0000001563019104_p95741410511"></a>执行成功。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001563019104_row9156144365013"><td class="cellrowborder" valign="top" width="30.543054305430545%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001563019104_p14704133965112"><a name="zh-cn_topic_0000001563019104_p14704133965112"></a><a name="zh-cn_topic_0000001563019104_p14704133965112"></a>ACL_ERROR_INVALID_PARAM</p>
</td>
<td class="cellrowborder" valign="top" width="15.971597159715973%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001563019104_p1156543125020"><a name="zh-cn_topic_0000001563019104_p1156543125020"></a><a name="zh-cn_topic_0000001563019104_p1156543125020"></a>100000</p>
</td>
<td class="cellrowborder" valign="top" width="53.48534853485349%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001563019104_p1015624311507"><a name="zh-cn_topic_0000001563019104_p1015624311507"></a><a name="zh-cn_topic_0000001563019104_p1015624311507"></a>参数校验失败。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001563019104_row315644318505"><td class="cellrowborder" valign="top" width="30.543054305430545%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001563019104_p11156144312509"><a name="zh-cn_topic_0000001563019104_p11156144312509"></a><a name="zh-cn_topic_0000001563019104_p11156144312509"></a>ACL_ERROR_OP_INPUT_NOT_MATCH</p>
</td>
<td class="cellrowborder" valign="top" width="15.971597159715973%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001563019104_p915619437501"><a name="zh-cn_topic_0000001563019104_p915619437501"></a><a name="zh-cn_topic_0000001563019104_p915619437501"></a>100021</p>
</td>
<td class="cellrowborder" valign="top" width="53.48534853485349%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001563019104_p17570123425314"><a name="zh-cn_topic_0000001563019104_p17570123425314"></a><a name="zh-cn_topic_0000001563019104_p17570123425314"></a>单算子的输入不匹配。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001563019104_row1215674375018"><td class="cellrowborder" valign="top" width="30.543054305430545%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001563019104_p11156174305019"><a name="zh-cn_topic_0000001563019104_p11156174305019"></a><a name="zh-cn_topic_0000001563019104_p11156174305019"></a>ACL_ERROR_OP_OUTPUT_NOT_MATCH</p>
</td>
<td class="cellrowborder" valign="top" width="15.971597159715973%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001563019104_p9156443185011"><a name="zh-cn_topic_0000001563019104_p9156443185011"></a><a name="zh-cn_topic_0000001563019104_p9156443185011"></a>100022</p>
</td>
<td class="cellrowborder" valign="top" width="53.48534853485349%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001563019104_p4156543185018"><a name="zh-cn_topic_0000001563019104_p4156543185018"></a><a name="zh-cn_topic_0000001563019104_p4156543185018"></a>单算子的输出不匹配。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001563019104_row11561143115015"><td class="cellrowborder" valign="top" width="30.543054305430545%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001563019104_p107381545195210"><a name="zh-cn_topic_0000001563019104_p107381545195210"></a><a name="zh-cn_topic_0000001563019104_p107381545195210"></a>ACL_ERROR_UNSUPPORTED_DATA_TYPE</p>
</td>
<td class="cellrowborder" valign="top" width="15.971597159715973%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001563019104_p14156144313508"><a name="zh-cn_topic_0000001563019104_p14156144313508"></a><a name="zh-cn_topic_0000001563019104_p14156144313508"></a>100026</p>
</td>
<td class="cellrowborder" valign="top" width="53.48534853485349%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001563019104_p10850181263218"><a name="zh-cn_topic_0000001563019104_p10850181263218"></a><a name="zh-cn_topic_0000001563019104_p10850181263218"></a>不支持的数据类型。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001563019104_row11561143115015"><td class="cellrowborder" valign="top" width="30.543054305430545%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001563019104_p107381545195210"><a name="zh-cn_topic_0000001563019104_p107381545195210"></a><a name="zh-cn_topic_0000001563019104_p107381545195210"></a>ACL_ERROR_FORMAT_NOT_MATCH</p>
</td>
<td class="cellrowborder" valign="top" width="15.971597159715973%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001563019104_p14156144313508"><a name="zh-cn_topic_0000001563019104_p14156144313508"></a><a name="zh-cn_topic_0000001563019104_p14156144313508"></a>100027</p>
</td>
<td class="cellrowborder" valign="top" width="53.48534853485349%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001563019104_p10850181263218"><a name="zh-cn_topic_0000001563019104_p10850181263218"></a><a name="zh-cn_topic_0000001563019104_p10850181263218"></a>Format不匹配。</p>

</td>
</tr>
<tr id="zh-cn_topic_0000001563019104_row11561143115015"><td class="cellrowborder" valign="top" width="30.543054305430545%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001563019104_p107381545195210"><a name="zh-cn_topic_0000001563019104_p107381545195210"></a><a name="zh-cn_topic_0000001563019104_p107381545195210"></a>ACL_ERROR_API_NOT_SUPPORT</p>
</td>
<td class="cellrowborder" valign="top" width="15.971597159715973%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001563019104_p14156144313508"><a name="zh-cn_topic_0000001563019104_p14156144313508"></a><a name="zh-cn_topic_0000001563019104_p14156144313508"></a>200001</p>
</td>
<td class="cellrowborder" valign="top" width="53.48534853485349%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001563019104_p10850181263218"><a name="zh-cn_topic_0000001563019104_p10850181263218"></a><a name="zh-cn_topic_0000001563019104_p10850181263218"></a>接口不支持。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001563019104_row11561143115015"><td class="cellrowborder" valign="top" width="30.543054305430545%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001563019104_p107381545195210"><a name="zh-cn_topic_0000001563019104_p107381545195210"></a><a name="zh-cn_topic_0000001563019104_p107381545195210"></a>ACL_ERROR_INTERNAL_ERROR</p>
</td>
<td class="cellrowborder" valign="top" width="15.971597159715973%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001563019104_p14156144313508"><a name="zh-cn_topic_0000001563019104_p14156144313508"></a><a name="zh-cn_topic_0000001563019104_p14156144313508"></a>500000</p>
</td>
<td class="cellrowborder" valign="top" width="53.48534853485349%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001563019104_p95741410511"><a name="zh-cn_topic_0000001563019104_p95741410511"></a><a name="zh-cn_topic_0000001563019104_p95741410511"></a>内部未知错误。</p>
</td>
</tr>
</tbody>
</table>

# 日志系统

信号处理加速库的日志系统支持日志分级、日志输出到标准输出、日志输出到文件。

- **日志分级**：

日志的严重级别从高到低分为ERROR、WARN、INFO、DEBUG四个级别，如下表所示。日志级别由环境变量“ASCEND_GLOBAL_LOG_LEVEL”控制，默认为“INFO”。\
**表 2**  日志等级

<a name="zh-cn_topic_0000001563019104_table8155243135018"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001563019104_row111561243135019"><th class="cellrowborder" valign="top" width="30.543054305430545%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0000001563019104_p6676115185014"><a name="zh-cn_topic_0000001563019104_p6676115185014"></a><a name="zh-cn_topic_0000001563019104_p6676115185014"></a>级别</p>
</th>
<th class="cellrowborder" valign="top" width="15.971597159715973%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0000001563019104_p16690195185015"><a name="zh-cn_topic_0000001563019104_p16690195185015"></a><a name="zh-cn_topic_0000001563019104_p16690195185015"></a>含义</p>
</th>
</tr>
</thead>
<tbody>
<tr id="zh-cn_topic_0000001563019104_row9156144365013"><td class="cellrowborder" valign="top" width="30.543054305430545%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001563019104_p14704133965112"><a name="zh-cn_topic_0000001563019104_p14704133965112"></a><a name="zh-cn_topic_0000001563019104_p14704133965112"></a>ERROR</p>
</td>
<td class="cellrowborder" valign="top" width="15.971597159715973%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001563019104_p1156543125020"><a name="zh-cn_topic_0000001563019104_p1156543125020"></a><a name="zh-cn_topic_0000001563019104_p1156543125020"></a>错误信息，该级别打印错误与异常信息。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001563019104_row9156144365013"><td class="cellrowborder" valign="top" width="30.543054305430545%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001563019104_p14704133965112"><a name="zh-cn_topic_0000001563019104_p14704133965112"></a><a name="zh-cn_topic_0000001563019104_p14704133965112"></a>WARN</p>
</td>
<td class="cellrowborder" valign="top" width="15.971597159715973%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001563019104_p1156543125020"><a name="zh-cn_topic_0000001563019104_p1156543125020"></a><a name="zh-cn_topic_0000001563019104_p1156543125020"></a>警告信息，表明会出现潜在错误的情形，给开发者一些提示。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001563019104_row9156144365013"><td class="cellrowborder" valign="top" width="30.543054305430545%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001563019104_p14704133965112"><a name="zh-cn_topic_0000001563019104_p14704133965112"></a><a name="zh-cn_topic_0000001563019104_p14704133965112"></a>INFO（默认）</p>
</td>
<td class="cellrowborder" valign="top" width="15.971597159715973%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001563019104_p1156543125020"><a name="zh-cn_topic_0000001563019104_p1156543125020"></a><a name="zh-cn_topic_0000001563019104_p1156543125020"></a>数据信息，打印算子与整图相关的信息，用户通过观察INFO日志就可以得知整图或单算子的运行状态。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001563019104_row9156144365013"><td class="cellrowborder" valign="top" width="30.543054305430545%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001563019104_p14704133965112"><a name="zh-cn_topic_0000001563019104_p14704133965112"></a><a name="zh-cn_topic_0000001563019104_p14704133965112"></a>DEBUG</p>
</td>
<td class="cellrowborder" valign="top" width="15.971597159715973%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001563019104_p1156543125020"><a name="zh-cn_topic_0000001563019104_p1156543125020"></a><a name="zh-cn_topic_0000001563019104_p1156543125020"></a>调试信息，打印加速库代码细节信息，加速库开发者可以通过查看DEBUG日志来调试框架代码。</p>
</td>
</tr>
</tbody>
</table>

- **日志保存**：
日志文件保存在“[LOG_PATH]/log/asdsip”下。
   <ul><li>[LOG_PATH]由环境变量ASCEND_PROCESS_LOG_PATH控制，默认为“~/ascend”；</li><li>日志文件的命名格式为asdsip_[PID]_[年][月][日][时][分][秒].log。[PID]为进程号。例如：asdsip_253440_20231102065052.log。</li></ul>
- **空间管理**：

   <ul><li>每个日志文件大小最大为20MB，最多存50个文件。如当前保存目录下的日志文件（以标准命名格式存储的日志文件）达到最高存储数量，将根据时间戳，删除最早时间的日志文件。</li><li>在生成日志文件前，将会对日志保存目录的空间大小进行判断，如果空间不足1GB，将不会继续生成日志文件。</li></ul>

# DumpTensor能力

信号处理加速库Dump Tensor功能是在算子运行过程中，将算子计算过程中产生的中间数据，或算子的输入、输出进行打印或保存。具体包括下述两种场景：用户使用信号加速库算子、自定义计算流程场景。

场景一：用户使用信号加速库算子，在业务流程中，对信号加速库算子的输入或输出进行打印或保存，辅助用户分析或定位业务流程中的计算结果是否正确。
场景二：cpp侧调用信号加速库算子、自定义计算流程，在cpp侧调用时，可基于C++自身的函数进行打印或者数据保存。示例如下：

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
    tensorInXData.resize(xSize);
    for (int64_t i = 0; i < xSize; i++) {
        tensorInXData[i] = {2.0, (float)(1.0 + i)};
    }
    std::vector<std::complex<float>> tensorInYData;
    tensorInYData.resize(ySize);
    for (int64_t i = 0; i < ySize; i++) {
        tensorInYData[i] = {3.0, 4.0};
    }
    int64_t resultSize = 1;
    std::vector<std::complex<float>> resultData;
    resultData.resize(resultSize);
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
