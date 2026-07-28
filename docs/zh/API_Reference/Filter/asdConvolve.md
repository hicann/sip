# asdConvolve

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |     ×    |
|  <term>Atlas 训练系列产品</term>    |     ×    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Ascend 950PR/Ascend 950DT</term>   |     ×  |

## 功能说明

- 接口功能：对给定的信号进行一维滤波操作。
- 计算公式：
  $$
  w(k)=\sum _{j}u(j)v(k-j+1)
  $$
  其中，w(k)为输出的k位置的元素，u(j)为输入位置为j的一维信号，v(k-j+1)为位置为k-j+1的滤波卷积核。一维信号为复数向量，滤波卷积核为实数向量。

            示例：
            输入“u”为：
            [[1.+1.j 2.+2.j],
            [1.+1.j 2.+2.j]]
            输入“v”为：
            [1. 2. 3. 4.]
            调用“asdConvolve”算子后，输出“result”为：
            [[4.+4.j, 7.+7.j],
            [4.+4.j, 7.+7.j]]

## 函数原型

```Cpp
AspbStatus asdConvolve(
  const aclTensor *    signal,
  const aclTensor *    kernel,
  aclTensor *          output,
  asdConvolveMode_t    mode,
  void *stream, void * workspace)
```

## asdConvolve

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 880px"><colgroup>
    <col style="width: 250px">
    <col style="width: 120px">
    <col style="width: 510px">
  </colgroup>
  <thead>
      <tr>
        <th>参数名</th>
        <th>输入/输出</th>
        <th>描述</th>
      </tr></thead>
  <tbody>
    <tr>
      <td>signal（aclTensor *）</td>
      <td>输入</td>
      <td><ul><li>输入的一维信号。</li><li>支持的数据类型为COMPLEX32、COMPLEX64。</li><li>输入信号shape为[BatchCount, n]。</li></ul></td>
    </tr>
    <tr>
      <td>kernel（aclTensor *）</td>
      <td>输入</td>
      <td><ul><li>输入的滤波卷积核。</li><li>支持的数据类型为FLOAT16、FLOAT32。</li><li>输入滤波卷积核shape为[k]。</li></ul></td>
    </tr>
    <tr>
      <td>output（aclTensor *）</td>
      <td>输入/输出</td>
      <td><ul><li>输入/输出信号。</li><li>支持的数据类型为COMPLEX32、COMPLEX64。</li><li>输出shape与输入shape保持一致。</li></ul></td>
    </tr>
    <tr>
      <td>mode（asdConvolveMode_t）</td>
      <td>输入</td>
      <td>滤波卷积模式，当前仅支持ASD_CONVOLVE_SAME，即输入和输出的向量维度保持一致。</td>
    </tr>
    <tr>
      <td>stream（void*）</td>
      <td>输入</td>
      <td>算子执行时的stream。</td>
    </tr>
    <tr>
      <td>workspace（void*）</td>
      <td>输入</td>
      <td>算子所需的Workspace指针。</td>
    </tr>
    </tbody>
    </table>
- **返回值**：

  返回状态码，具体参见[SiP返回码](../../context/SiP返回码.md)。

## 约束说明

无

## 调用示例

示例代码如下，该样例旨在提供快速上手、开发和调试算子的最小化实现，其核心目标是使用最精简的代码展示算子的核心功能，而非提供生产级的安全保障。不推荐用户直接将示例代码作为业务代码，若用户将示例代码应用在自身的真实业务场景中且发生了安全问题，则需用户自行承担。

```Cpp
#include <iostream>
#include "asdsip.h"
#include "filter_api.h"
#include "acl/acl.h"
#include "acl/acl_base.h"
#include "acl_meta.h"
#include <complex>
#include <vector>

using namespace AsdSip;

using half = op::fp16_t;

#define ASD_STATUS_CHECK(err)                                                \
    do {                                                                     \
        AsdSip::AspbStatus err_ = (err);                                     \
        if (err_ != AsdSip::ACL_SUCCESS) {                                      \
            std::cout << "Execute failed." << std::endl; \
            exit(-1);                                                        \
        }                                                                    \
    } while (0)

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
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
    aclDataType dataType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // 调用aclrtMalloc申请device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // 调用aclrtMemcpy将host侧数据复制到device侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

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

template <typename T>
void printTensor(std::vector<T> tensorData, int64_t tensorSize)
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
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    int64_t signalLen = 128; // 26208
    int64_t kernelLen = 32;
    int64_t batchCount = 2; // 768

    std::vector<std::complex<half>> tensorSignalData;
    tensorSignalData.resize(signalLen * batchCount);

    std::vector<half> tensorKernelData;
    tensorKernelData.resize(kernelLen);

    for (int64_t i = 0; i < signalLen * batchCount; i++) {
        tensorSignalData[i] = {(half)1.0, (half)1.0};
    }

    for (int64_t i = 0; i < kernelLen; i++) {
        tensorKernelData[i] = (half)(1.0 + i);
        // tensorKernelData[i] = 1.0;
    }

    std::vector<std::complex<half>> tensorOutData;
    tensorOutData.resize(signalLen * batchCount);

    for (int64_t i = 0; i < signalLen * batchCount; i++) {
        tensorOutData[i] = {(half)-1.0, (half)-1.0};
    }

    std::vector<int64_t> signalShape = {batchCount, signalLen};
    std::vector<int64_t> kernelShape = {kernelLen};
    std::vector<int64_t> resultShape = {batchCount, signalLen};

    aclTensor *signal = nullptr;
    aclTensor *kernel = nullptr;
    aclTensor *output = nullptr;
    void *signalDeviceAddr = nullptr;
    void *kernelDeviceAddr = nullptr;
    void *outputDeviceAddr = nullptr;

    ret = CreateAclTensor<std::complex<half>>(
        tensorSignalData, signalShape, &signalDeviceAddr, aclDataType::ACL_COMPLEX32, &signal);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);

    ret = CreateAclTensor<half>(
        tensorKernelData, kernelShape, &kernelDeviceAddr, aclDataType::ACL_FLOAT16, &kernel);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);

    ret = CreateAclTensor<std::complex<half>>(
        tensorOutData, resultShape, &outputDeviceAddr, aclDataType::ACL_COMPLEX32, &output);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);

    size_t lwork = 0;
    AsdSip::asdConvolveGetWorkspaceSize(signalLen, kernelLen, lwork);
    void *buffer = nullptr;

    std::cout << "lwork = " << lwork << std::endl;
    if (lwork > 0) {
        ret = aclrtMalloc(&buffer, static_cast<int64_t>(lwork), ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    ASD_STATUS_CHECK(AsdSip::asdConvolve(signal, kernel, output, asdConvolveMode_t::ASD_CONVOLVE_SAME, stream, buffer));

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    ret = aclrtMemcpy(tensorOutData.data(),
        signalLen * batchCount * sizeof(std::complex<half>),
        outputDeviceAddr,
        signalLen * batchCount * sizeof(std::complex<half>),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

    std::cout << "------- result -------" << std::endl;
    for (int batchIdx = 0; batchIdx < batchCount; batchIdx++) {
        for (int i = 0; i < signalLen; i++) {
            std::cout << "(" << (float)tensorOutData[batchIdx * signalLen + i].real() << ","
                      << (float)tensorOutData[batchIdx * signalLen + i].imag() << ")"
                      << " ";
        }
        std::cout << std::endl;
    }

    aclDestroyTensor(signal);
    aclDestroyTensor(kernel);
    aclDestroyTensor(output);
    aclrtFree(signalDeviceAddr);
    aclrtFree(kernelDeviceAddr);
    aclrtFree(outputDeviceAddr);

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
