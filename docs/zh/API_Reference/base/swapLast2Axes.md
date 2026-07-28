# swapLast2Axes

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |     ×    |
|  <term>Atlas 训练系列产品</term>    |     ×    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Ascend 950PR/Ascend 950DT</term>   |     ×    |

## 功能说明

- 接口功能：\
swapLast2AxesGetWorkspaceSize：计算swapLast2Axes算子所需的workspace大小。\
swapLast2Axes：交换Tensor的最后两维。
- 计算公式：

  $$
  outTensor_{bij} = inTensor_{bji}\\
  $$
  其中：b为数据的批次号，i为输入数据的行号， j为输入数据的列号。
示例：
  - 示例一：
输入“inTensor”为：\
[[[1.+0.j, 2.+0.j, 3.+0.j]]]\
调用swapLast2Axes算子后，输出“outTensor”为：\
[[[1.+0.j], [2.+0.j], [3.+0.j]]]
  - 示例二：
输入“inTensor”为：\
[[[ 0.+0.j, 1.+0.j, 2.+0.j], \
[ 3.+0.j, 4.+0.j, 5.+0.j]],\
[[ 6.+0.j, 7.+0.j, 8.+0.j], \
[ 9.+0.j, 10.+0.j, 11.+0.j]]]\
调用swapLast2Axes算子后，输出“outTensor”为：\
[[[ 0.+0.j, 3.+0.j], \
[ 1.+0.j, 4.+0.j], \
[ 2.+0.j, 5.+0.j]],\
[[ 6.+0.j, 9.+0.j], \
[ 7.+0.j, 10.+0.j], \
[ 8.+0.j, 11.+0.j]]]

## 函数原型

若需使用“swapLast2Axes”算子，必须先调用“swapLast2AxesGetWorkspaceSize”接口获取入参并根据计算流程计算所需workspace大小，再调用“swapLast2Axes”接口执行计算。

```Cpp
AsdSip::AspbStatus swapLast2AxesGetWorkspaceSize(
  size_t *size)
```

```Cpp
AsdSip::AspbStatus swapLast2Axes(
  const aclTensor *    inTensor,
  aclTensor *          outTensor,
  void *               stream,
  void *               workspace = nullptr)
```

- **返回值**：

  返回状态码，具体参见[SiP返回码](../../context/SiP返回码.md)。

## swapLast2AxesGetWorkspaceSize

- **参数说明**：

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
      <td>size（size_t *）</td>
      <td>输入/输出</td>
      <td>swapLast2Axes算子所需要的workspace。</td>
    </tr>
  </tbody>
  </table>

- **返回值**：

  返回状态码，具体参见[SiP返回码](../../context/SiP返回码.md)。

## swapLast2Axes

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
      <td>inTensor（aclTensor *）</td>
      <td>输入</td>
      <td><ul><li>表示输入的张量数据，对应公式中的'inTensor'。</li><li>输入的最大元素数为3600000000 ([60000, 60000]以内)。</li><li>数据类型仅支持COMPLEX64，数据格式支持ND。</li>
      <li>输入dim限制为2或3。</li></ul>
      </td>
    </tr>
    <tr>
      <td>outTensor（aclTensor *）</td>
      <td>输出</td>
      <td><ul><li>表示输出的张量数据，对应公式中的'outTensor'。</li><li>数据类型仅支持COMPLEX64，数据类型需要与inTensor的数据类型一致。</li><li>如果inTensor的shape为[k,x,y]，outTensor的shape为[k,y,x]。</li><li>数据格式支持ND。</li></ul></td>
    </tr>
    <tr>
      <td>workspace（void *）</td>
      <td>输入</td>
      <td>swapLast2Axes算子所需要的workspace。</td>
    </tr>
    <tr>
      <td>stream(void *)</td>
      <td>输入</td>
      <td>npu执行流。</td>
    </tr>
  </tbody>
  </table>

- **返回值**：

  返回状态码，具体参见[SiP返回码](../../context/SiP返回码.md)。

## 约束说明

  算子实际计算时，不支持ND高维度运算（不支持维度>3的运算）。

## 调用示例

示例代码如下，该样例旨在提供快速上手、开发和调试算子的最小化实现，其核心目标是使用最精简的代码展示算子的核心功能，而非提供生产级的安全保障。不推荐用户直接将示例代码作为业务代码，若用户将示例代码应用在自身的真实业务场景中且发生了安全问题，则需用户自行承担。

```Cpp
#include <iostream>
#include <vector>
#include "asdsip.h"
#include "acl/acl.h"
#include "acl_meta.h"

using namespace AsdSip;

#define ASD_STATUS_CHECK(err)                                                \
    do {                                                                     \
        AsdSip::AspbStatus err_ = (err);                                     \
        if (err_ != AsdSip::ErrorType::ACL_SUCCESS) {                                      \
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
    auto size = GetShapeSize(shape) * sizeof(T) * 2;
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

void printTensor(const float *tensorData, size_t row, size_t col)
{
    for (size_t r = 0; r < row; ++r) {
        for (size_t c = 0; c < col; ++c) {
            size_t index = (r * col + c) * 2;
            std::cout << "(" << int(tensorData[index]) << ", " << int(tensorData[index + 1]) << ") ";
        }
        std::cout << "\n";
    }
}

int main(int argc, char **argv)
{
    int deviceId = 0;

    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    int64_t row = 3;
    int64_t col = 2;
    const int64_t tensorSize = row * col * 2;

    std::vector<float> tensorInData;
    tensorInData.resize(tensorSize);
    for (int64_t i = 0; i < tensorSize; i++) {
        tensorInData[i] = 0.0 + i;
    }
    std::vector<float> tensorOutData;
    tensorOutData.resize(tensorSize);

    std::vector<int64_t> inShape = {row, col};
    std::vector<int64_t> outShape = {col, row};
    aclTensor *input = nullptr;
    aclTensor *output = nullptr;
    void *inputDeviceAddr = nullptr;
    void *outputDeviceAddr = nullptr;
    ret = CreateAclTensor(tensorInData, inShape, &inputDeviceAddr, aclDataType::ACL_COMPLEX64, &input);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);
    ret = CreateAclTensor(tensorOutData, outShape, &outputDeviceAddr, aclDataType::ACL_COMPLEX64, &output);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);

    void *workspace = nullptr;
    size_t lwork = 0;
    swapLast2AxesGetWorkspaceSize(lwork);
    std::cout << "lwork = " << lwork << std::endl;
    if (lwork > 0) {
        ret = aclrtMalloc(&workspace, static_cast<int64_t>(lwork), ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    ASD_STATUS_CHECK(swapLast2Axes(input, output, stream, workspace));

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    ret = aclrtMemcpy(tensorOutData.data(),
        tensorSize * sizeof(float),
        outputDeviceAddr,
        tensorSize * sizeof(float),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("copy output tensor from device to host failed. ERROR: %d\n", ret); return ret);

    std::cout << "row = " << row << ", col = " << col << std::endl;
    std::cout << "------- Input ------- " << std::endl;
    printTensor(tensorInData.data(), row, col);

    std::cout << "------- Output -------" << std::endl;
    printTensor(tensorOutData.data(), col, row);
    std::cout << "Execute successfully." << std::endl;

    aclrtFree(inputDeviceAddr);
    aclrtFree(outputDeviceAddr);
    aclDestroyTensor(input);
    aclDestroyTensor(output);
    if (lwork > 0) {
        aclrtFree(workspace);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
