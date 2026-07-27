# rsInterpolationBySinc

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
rsInterpolationBySincGetWorkspaceSize：计算rsInterpolationBySinc算子所需的workspace大小。\
rsInterpolationBySinc：实现带batch的一维复数向量插值计算，返回结果和插值坐标同样形状大小。
- 计算公式：
  $$
  x(d)=\sum _{n=0}^{N-1}x[n]sinc(d-n)
  $$
  其中,n为实信号索引，x[n]为原实信号序列，sinc(d-n)是插值系数，在本算子中需要作为参数传入。\
  示例：\
输入“inputTensor”为：\
 [ 1 + i , 2 + i ]\
输入“sincTab”为：（intp_num = 2， quant_num = 2）\
  [   [ 1 , 0 ],  [ 0.5 , 0.5 ],   [ 0 , 1 ]  ]\
原始“pos”为：\
  [ 0.2 , 1.6 ] \
转为输入“posFloor”为： floor(Pos)\
  [ 0 , 1 ] \
转为输入“posToTabIndex”为：round((Pos -posFloor) *quant_num)\
  [ 0 , 1 ] \
其中，tab大小为2*3。由于pos[0] = 0.2，取inputTensor[0]及后面一个元素inputTensor[1]共2个元素，与sincTab[posToTabIndex[0]]进行向量点乘，得到outputTensor[0]，依次计算后续元素。\
  pos[0] = 0.2 → outputTensor[0] = [1 + i , 2 + i] · [ 1 , 0 ] = 1 + i \
  pos[1] = 1.6 → outputTensor[1] = [2 + i , 2 + i] · [ 0.5 , 0.5 ] = 2 + i\
调用“rsInterpolationBySinc”算子后，输出“outputTensor”为：\
  [ 1 + i , 2 + i ]
 
## 函数原型

若需使用“rsInterpolationBySinc”算子，需先调用“rsInterpolationBySincGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“rsInterpolationBySinc”接口执行计算。

```Cpp
AspbStatus rsInterpolationBySincGetWorkspaceSize(
  size_t &                   workspaceSize)
```

```Cpp
AspbStatus rsInterpolationBySinc(
  const aclTensor *          inputTensor, 
  const aclTensor *          sincTab,
  const aclTensor *          posFloor, 
  const aclTensor *          posToTabIndex,
  aclTensor *                outputTensor, 
  int                        interpNum, 
  int                        quantNum, 
  int                        interpLength,
  void *                     stream, 
  void *                     workSpace = nullptr)

```

## rsInterpolationBySincGetWorkspaceSize

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
      <td>workspace的地址。</td>
    </tr>
  </tbody>
    </table>
- **返回值**：

  返回状态码，具体参见[SiP返回码](../../context/SiP返回码.md)。

## rsInterpolationBySinc

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
      <td>inputTensor（aclTensor *）</td>
      <td>输入</td>
      <td><ul><li>表示原始信号。</li><li>支持的数据类型为COMPLEX64。</li><li>数据格式支持ND。</li><li>shape为[batch, signalLength]。</li></ul></td>
    </tr>
    <tr>
      <td>sincTab（aclTensor *）</td>
      <td>输入</td>
      <td><ul><li>表示插值系数矩阵。</li><li>支持的数据类型为FLOAT32。</li><li>数据格式支持ND。</li><li>shape为[ 4, ((quantNum + 1) * 2) * (interpNum * 2 + 8)]。</li></ul></td>
    </tr>
    <tr>
      <td>posFloor（aclTensor *）</td>
      <td>输入</td>
      <td><ul><li>表示插值点坐标向下取整后的值。</li><li>支持的数据类型为INT32。</li><li>数据格式支持ND。</li><li>shape为[batch, length]。</li></ul></td>
    </tr>
    <tr>
      <td>posToTabIndex（aclTensor *）</td>
      <td>输入</td>
      <td><ul><li>插值点坐标通过round((Pos -posFloor) *quantNum)计算出对应插值系数矩阵的行号。</li><li>支持的数据类型为INT16_T。</li><li>数据格式支持ND。</li><li>shape为[batch, length]。</li></ul></td>
    </tr>
    <tr>
      <td>outputTensor（aclTensor *）</td>
      <td>输出</td>
      <td><ul><li>插值结果。</li><li>支持的数据类型为COMPLEX64。</li><li>数据格式支持ND。</li><li>shape为[batch, length]。</li></ul></td>
    </tr>
    <tr>
      <td>interpNum（int）</td>
      <td>输入</td>
      <td>插值点数。</td>
    </tr>
    <tr>
      <td>quantNum（int）</td>
      <td>输入</td>
      <td>量化点数。</td>
    </tr>
    <tr>
      <td>interpLength（int）</td>
      <td>输入</td>
      <td>插值长度。</td>
    </tr>
    <tr>
      <td>stream（void *）</td>
      <td>输入</td>
      <td>npu执行流。</td>
    </tr>
    <tr>
      <td>workSpace（void *）</td>
      <td>输入</td>
      <td>workspace的地址。</td>
    </tr>
  </tbody>
    </table>
- **返回值**：

  返回状态码，具体参见[SiP返回码](../../context/SiP返回码.md)。
  
## 约束说明

rsInterpolationBySinc：

- 输入的元素个数理论支持[1，3.93e+09]。
- 算子实际计算时，不支持ND高维度运算（不支持维度≥3的运算）。
- interpNum只支持偶数，通常使用 [8 ，12 ，16] ，当前版本最大支持16。
- quantNum为2的幂，最大32。
- inputTensor、posFloor、posToTabIndex第0维是相同的batch数，outputTensor长度和posFloor、posToTabIndex一致。
- sincTab：为了将复数点乘转化为实数点乘以及更亲和NPU，需要进行预处理，需要扩充成[ ((quantNum+1) *2) * (interpNum*2+8) * 4] ，具体算法参考“调用示例”中的预处理内容。

## 调用示例

示例代码如下，该样例旨在提供快速上手、开发和调试算子的最小化实现，其核心目标是使用最精简的代码展示算子的核心功能，而非提供生产级的安全保障。不推荐用户直接将示例代码作为业务代码，若用户将示例代码应用在自身的真实业务场景中且发生了安全问题，则需用户自行承担。

- rsInterpolationBySinc算子调用说明：\
  - 调用示例中会进行sincTab预处理：\
系数矩阵要和复数点乘，在NPU上会转换为系数矩阵和两组float32相乘，所以需要将系数转为下面格式，此时虚数矩阵会被扩充成 ((quantNum+1)*2)* (interpNum*2)。\
    [系数1,0]\
    [ 0,系数1]\
  - 为了亲和NPU（32字节对齐），需要有四种格式矩阵，每行开头不补0，补2个0，补4个0及补6个0，其中“genTab”函数用于生成如下类似系数矩阵：
  w0,0,w1,0,w2,0,w3,0,w4,0,w5,0,w6,0,w7,0,w8,0,w9,0,w10,0,w11,0,w12,0,w13,0,w14,0,w15,0,0,0,0,0,0,0,0\
0,0,w0,0,w1,0,w2,0,w3,0,w4,0,w5,0,w6,0,w7,0,w8,0,w9,0,w10,0,w11,0,w12,0,w13,0,w14,0,w15,0,0,0,0,0,0\
0,0,0,0,w0,0,w1,0,w2,0,w3,0,w4,0,w5,0,w6,0,w7,0,w8,0,w9,0,w10,0,w11,0,w12,0,w13,0,w14,0,w15,0,0,0,0\
0,0,0,0,0,0,w0,0,w1,0,w2,0,w3,0,w4,0,w5,0,w6,0,w7,0,w8,0,w9,0,w10,0,w11,0,w12,0,w13,0,w14,0,w15,0,0

- rsInterpolationBySinc算子调用示例：

```Cpp
#include <iostream>
#include <vector>
#include <securec.h>
#include "asdsip.h"
#include "acl/acl.h"
#include "acl_meta.h"

using std::complex;
using namespace AsdSip;

#define ASD_STATUS_CHECK(err)                                                \
    do {                                                                     \
        AsdSip::AspbStatus err_ = (err);                                     \
        if (err_ != AsdSip::ErrorType::ACL_SUCCESS) {                                      \
            std::cout << "Execute failed." << std::endl; \
            exit(-1);                                                        \
        }                                                                    \
    } while (0)

#define DINTER_CORE_SIZE 528

static void genTab(float *tab, int tabSize)
{
    static float DINTER_CORE_33x16[DINTER_CORE_SIZE];
    for (int i = 0; i < DINTER_CORE_SIZE; ++i) {
        DINTER_CORE_33x16[i] = ((rand() / (float)RAND_MAX) * 2.0f) - 1.0f;
    }

    for (int i = 0; i < 4; i++) {
        int zeroOffset = i * 2;
        int blockOffset = i * (33 * 2) * (16 * 2 + 8);
        for (int j = 0; j < 33; j++) {
            int rowOffset_real = blockOffset + j * (16 * 2 + 8) * 2;
            int rowOffset_imag = rowOffset_real + (16 * 2 + 8);
            for (int k = 0; k < 16; k++) {
                tab[rowOffset_real + zeroOffset + k * 2] = DINTER_CORE_33x16[j * 16 + k];
                tab[rowOffset_imag + zeroOffset + k * 2 + 1] = DINTER_CORE_33x16[j * 16 + k];
            }
        }
    }
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

int main(int argc, char **argv)
{
    int deviceId = 0;

    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    int batch = 1;
    int signalLength = 64;
    int interpLength = signalLength;
    const int64_t tabSize = (33 * 2) * (16 * 2 + 8) * 4;  // 2 ：虚实系数，8：补零，4：不补0，补2,4,6个零
    const unsigned long inSize = batch * signalLength;
    const unsigned long posSize = batch * interpLength;
    const unsigned long tabIndexSize = batch * interpLength;
    const unsigned long outSize = batch * interpLength;

    float *tabDate = new float[tabSize]();
    genTab(tabDate, tabSize);
    std::vector<float> tab(tabDate, tabDate + tabSize);
    std::vector<complex<float>> inSignal;
    inSignal.reserve(inSize);
    for (long long ii = 0; ii < signalLength; ++ii) {
        inSignal[ii] = complex<float>(ii, ii);
    }
    std::vector<int32_t> intpPos;
    intpPos.reserve(posSize);
    for (long long ii = 0; ii < interpLength; ++ii) {
        intpPos[ii] = ii;
    }
    std::vector<int16_t> tabIndex;
    tabIndex.reserve(tabIndexSize);
    for (long long ii = 0; ii < interpLength; ++ii) {
        tabIndex[ii] = ii % 33;
    }
    std::vector<complex<float>> outSignal;
    outSignal.reserve(outSize);
    for (long long ii = 0; ii < interpLength; ++ii) {
        outSignal[ii] = complex<float>(0, 0);
    }

    aclTensor *tensorIn = nullptr;
    aclTensor *tensorTab = nullptr;
    aclTensor *tensorPos = nullptr;
    aclTensor *tensorTabIndex = nullptr;
    aclTensor *tensorOut = nullptr;
    void *tensorInDeviceAddr = nullptr;
    void *tensorTabDeviceAddr = nullptr;
    void *tensorPosDeviceAddr = nullptr;
    void *tensorTabIndexDeviceAddr = nullptr;
    void *tensorOutDeviceAddr = nullptr;
    ret = CreateAclTensor(inSignal, {batch, signalLength}, &tensorInDeviceAddr, aclDataType::ACL_COMPLEX64, &tensorIn);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);
    ret = CreateAclTensor(tab, {1, tabSize}, &tensorTabDeviceAddr, aclDataType::ACL_FLOAT, &tensorTab);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);
    ret = CreateAclTensor(intpPos, {batch, interpLength}, &tensorPosDeviceAddr, aclDataType::ACL_INT32, &tensorPos);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);
    ret = CreateAclTensor(
        tabIndex, {batch, interpLength}, &tensorTabIndexDeviceAddr, aclDataType::ACL_INT16, &tensorTabIndex);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);
    ret =
        CreateAclTensor(outSignal, {batch, interpLength}, &tensorOutDeviceAddr, aclDataType::ACL_COMPLEX64, &tensorOut);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);

    void *workspace = nullptr;
    size_t workspaceSize = 0;
    rsInterpolationBySincGetWorkspaceSize(workspaceSize);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspace, static_cast<int64_t>(workspaceSize), ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    ASD_STATUS_CHECK(rsInterpolationBySinc(
        tensorIn, tensorTab, tensorPos, tensorTabIndex, tensorOut, 16, 32, interpLength, stream, workspace));

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    ret = aclrtMemcpy(outSignal.data(),
        outSize * sizeof(std::complex<float>),
        tensorOutDeviceAddr,
        outSize * sizeof(std::complex<float>),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

    for (long long ii = 0; ii < interpLength; ++ii) {
        std::cout << outSignal[ii] << "\t";
    }
    std::cout << "\nend result" << std::endl;
    std::cout << "Execute successfully." << std::endl;

    delete[] tabDate;
    aclDestroyTensor(tensorIn);
    aclDestroyTensor(tensorPos);
    aclDestroyTensor(tensorTab);
    aclDestroyTensor(tensorTabIndex);
    aclDestroyTensor(tensorOut);
    aclrtFree(tensorInDeviceAddr);
    aclrtFree(tensorTabDeviceAddr);
    aclrtFree(tensorPosDeviceAddr);
    aclrtFree(tensorTabIndexDeviceAddr);
    aclrtFree(tensorOutDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspace);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
