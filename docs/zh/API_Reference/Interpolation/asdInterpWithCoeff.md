# asdInterpWithCoeff

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

- 接口功能：\
asdInterpWithCoeffGetWorkspaceSize：计算asdInterpWithCoeff算子所需的workspace大小。\
asdInterpWithCoeff：支持向量插值操作，主要用于数据符号的信道估计，或者均衡系数插值。
- 计算公式：

  $$
  result=A \odot\ B =(A)_{ij}(B)_{ij}
  $$

  示例：

  输入“A”为：\
[ [ 1+1i, 1+1i ],\
  [ 2+2i, 2+2i ] ]\
输入“B”为：\
[ [ 1+1i, 1+1i ],\
  [ 2+2i, 2+2i ] ]\
调用asdInterpWithCoeff算子后，输出“result”为：\
[ [ 0+2i, 0+2i ],\
  [ 0+8i, 0+8i ] ]

## 函数原型

```Cpp
AspbStatus asdInterpWithCoeffGetWorkspaceSize(
  size_t &             workspaceSize)
```

```Cpp
AspbStatus asdInterpWithCoeff(
  const aclTensor *    x,
  const aclTensor *    coefficient,
  aclTensor *          y,
  void *               stream,
  void *               workSpace = nullptr)
```

## asdInterpWithCoeffGetWorkspaceSize

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
      <td>workspaceSize（size_t &）</td>
      <td>输出</td>
      <td>算子所需要的workspace。</td>
    </tr>
  </tbody>
    </table>
- **返回值**：

  返回状态码，具体参见[SiP返回码](../../context/SiP返回码.md)。

## asdInterpWithCoeff

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
      <td>x（aclTensor *）</td>
      <td>输入</td>
      <td><ul><li>对应公式中的'B'。</li><li>数据类型支持COMPLEX32、COMPLEX64</li><li>数据格式支持ND。</li>
      <li>shape为[batch,nRs,totalSubcarrier]。<ul><li>batch：波束数量，取值范围是1~1024 （6G时最大取值为16(终端的流数)*64(基站接收的波束数)=1024）。</li><li>nRs：参考信号数，取值是2、4。</li><li>totalSubcarrier = nRB*12。</li><li>nRB：资源块数，取值范围是1~2730 （每RB包含12个子载波，5G时取值范围是1~273，6G时取值是5G的4倍到10倍）。</li>
      </ul></li></ul></td>
    </tr>
    <tr>
    <td>coefficient（aclTensor *）</td>
      <td>输入</td>
      <td><ul><li>对应公式中的'A'。</li><li>数据类型支持COMPLEX32、COMPLEX64</li><li>数据格式支持ND。</li>
      <li>shape为[batch,14-nRs,nRs]。<ul><li>batch：波束数量，取值范围是1~1024 （6G时最大取值为16(终端的流数)*64(基站接收的波束数)=1024）。</li><li>nRs：参考信号数，取值是2、4。</li></ul></li></ul></td>
    </tr>
    <tr>
      <td>y（aclTensor *）</td>
      <td>输出</td>
      <td><ul><li>对应公式中的'result'。</li><li>数据类型支持COMPLEX32、COMPLEX64</li><li>数据格式支持ND。</li>
      <li>shape为[batch,14-nRs,totalSubcarrier]。<ul><li>batch：波束数量，取值范围是1~1024 （6G时最大取值为16(终端的流数)*64(基站接收的波束数)=1024）。</li><li>nRs：参考信号数，取值是2、4。</li><li>totalSubcarrier = nRB*12。</li><li>nRB：资源块数，取值范围是1~2730 （每RB包含12个子载波，5G时取值范围是1~273, 6G时取值是5G的4倍到10倍）。</li></ul></li></ul></td>
    </tr>
    <tr>
      <td>stream（void *）</td>
      <td>输入</td>
      <td>npu执行流。</td>
    </tr>
    <tr>
      <td>workSpace（void *）</td>
      <td>输入</td>
      <td>asdInterpWithCoeff算子所需要的workspace。</td>
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
#include <complex>
#include <vector>
#include "interp_api.h"
#include "acl/acl.h"
#include "acl_meta.h"

using namespace AsdSip;

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
    aclInit(nullptr);
    aclrtSetDevice(deviceId);
    aclrtCreateStream(stream);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
    aclDataType dataType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T) * 2; // 2 : complex
    // 调用aclrtMalloc申请device侧内存
    aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    // 调用aclrtMemcpy将host侧数据复制到device侧内存上
    aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);

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

int main(int argc, char **argv)
{
    // 设置算子使用的device id
    int deviceId = 0;
    //（固定写法）创造执行流
    aclrtStream stream;
    Init(deviceId, &stream);

    // 创造tensor的Host侧数据
    int64_t batch = 1;
    int64_t nRs = 2;
    int64_t totalSubcarrier = 32;
    int64_t nSignal = 14;

    int64_t xSize = batch * nRs * totalSubcarrier * 2;
    std::vector<float> tensorInXData;
    tensorInXData.resize(xSize);
    for (int64_t i = 0; i < xSize; i++) {
        tensorInXData[i] = 1.0 + i;
    }

    int64_t coeffSize = batch * (nSignal - nRs) * nRs * 2;
    std::vector<float> coeffData;
    coeffData.resize(xSize);
    for (int64_t i = 0; i < coeffSize; i++) {
        coeffData[i] = 1;
    }

    int64_t resultSize = batch * (nSignal - nRs) * totalSubcarrier * 2;
    std::vector<float> resultData;
    resultData.resize(resultSize);
    for (int64_t i = 0; i < resultSize; i++) {
        resultData[i] = 2;
    }

    // int64_t xSize = batch * nRs * totalSubcarrier;
    // std::vector<std::complex<float>> tensorInXData(xSize, std::complex<float>(0, 0));
    // for (int i = 0; i < xSize; i++) {
    //     tensorInXData[i] = std::complex<float>(i * 2, i * 2 + 1);
    // }
    // int64_t coeffSize = batch * (nSignal - nRs) * nRs;
    // std::vector<std::complex<float>> coeffData(xSize, std::complex<float>(0, 0));
    // for (int i = 0; i < coeffSize; i++) {
    //     coeffData[i] = std::complex<float>(1, 1);
    // }
    // int64_t resultSize = batch * (nSignal - nRs) * totalSubcarrier;
    // std::vector<std::complex<float>> resultData(xSize, std::complex<float>(0, 0));
    // for (int i = 0; i < resultSize; i++) {
    //     resultData[i] = std::complex<float>(2, 2);
    // }

    std::cout << "------- input x -------" << std::endl;
    for (int64_t i = 0; i < xSize; i++) {
        std::cout << tensorInXData[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "------- input coeff -------" << std::endl;
    for (int64_t i = 0; i < coeffSize; i++) {
        std::cout << coeffData[i] << " ";
    }
    std::cout << std::endl;

    // 创造输入/输出tensor
    aclTensor *inputX = nullptr;
    aclTensor *inputCoeff = nullptr;
    aclTensor *result = nullptr;
    void *inputXDeviceAddr = nullptr;
    void *inputYDeviceAddr = nullptr;
    void *resultDeviceAddr = nullptr;
    CreateAclTensor(tensorInXData, {batch, nRs, totalSubcarrier}, &inputXDeviceAddr, aclDataType::ACL_COMPLEX64, &inputX);
    CreateAclTensor(coeffData, {batch, nSignal-nRs, nRs}, &inputYDeviceAddr, aclDataType::ACL_COMPLEX64, &inputCoeff);
    CreateAclTensor(resultData, {batch, nSignal-nRs, totalSubcarrier}, &resultDeviceAddr, aclDataType::ACL_COMPLEX64, &result);

    size_t lwork = 0;
    void *buffer = nullptr;
    AsdSip::asdInterpWithCoeffGetWorkspaceSize(lwork);
    if (lwork > 0) {
        aclrtMalloc(&buffer, static_cast<int64_t>(lwork), ACL_MEM_MALLOC_HUGE_FIRST);
    }
    asdInterpWithCoeff(inputX, inputCoeff, result, stream, buffer);
    aclrtSynchronizeStream(stream);
    // 将输出tensor的Device侧数据复制到Host侧内存上
    aclrtMemcpy(resultData.data(),
        resultSize * sizeof(float),
        resultDeviceAddr,
        resultSize * sizeof(float),
        ACL_MEMCPY_DEVICE_TO_HOST);

    std::cout << "------- result -------" << std::endl;
    for (int64_t i = 0; i < nSignal - nRs; i++) {
        for (int64_t j = 0; j < totalSubcarrier * 2; j++) {
            std::cout << resultData[i * totalSubcarrier * 2 + j] << " ";
        }
        std::cout << std::endl;
    }

    // 资源释放
    aclDestroyTensor(inputX);
    aclDestroyTensor(inputCoeff);
    aclDestroyTensor(result);
    aclrtFree(inputXDeviceAddr);
    aclrtFree(inputYDeviceAddr);
    aclrtFree(resultDeviceAddr);
    if (lwork > 0) {
        aclrtFree(buffer);
    }

    // 调度算子后重置算子使用的deviceId
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
