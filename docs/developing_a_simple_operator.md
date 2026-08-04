# 从零开始为 SiP 库开发一个 Conj 算子

本教程以能够运行为第一目标，先不追求性能极致，力求让第一次接触SiP的开发者 30 分钟内能在本地看到结果。

## 算子开发实例

我们以Conj算子（共轭算子）为例，说明基于SiP库开发一个算子的主要流程。

### 算子功能

输入一个复数向量V = a + bj，进行共轭计算，即Conj(V) = a - bj。

### 新增文件

- 在`core/base`目录下新增`conj.cpp`文件。文件内容如下：

```c++
#include "utils/assert.h"
#include "log/log.h"
#include "base_api.h"
#include "utils/ops_base.h"
#include "conj.h"

using namespace Mki;
using namespace AsdSip;

namespace AsdSip {
AspbStatus Conj(const Tensor &inTensor, Tensor &outTensor, void *stream, uint8_t *workspace)
{
    OpDesc opDesc;
    opDesc.opName = "ConjOperation";
    AsdSip::OpParam::Conj param;
    opDesc.specificParam = param;
    ASDSIP_LOG(DEBUG) << "OpDesc: " << opDesc.opName << "; OpDesc info: " << param.ToString();

    SVector<Tensor> inTensors = {inTensor};
    SVector<Tensor> outTensors = {outTensor};

    Status status = RunAsdOps(stream, opDesc, inTensors, outTensors, workspace);
    ASDSIP_ECHECK(status.Ok(), status.Message(), ErrorType::ACL_ERROR_INTERNAL_ERROR);

    outTensor = outTensors.at(0);

    ASDSIP_LOG(INFO) << "Execute Conj success.";
    return ErrorType::ACL_SUCCESS;
}
}
```

- 在`ops/base`下新增目录`conj`，该目录下主要存放Conj算子接入SiP框架部分以及算子tiling和kernel的代码，目录结构如下：

```txt
conj
├── CMakeLists.txt
├── conj
│   ├── conj_kernel.cpp
│   ├── op_kernel
│   │   ├── conj.cpp
│   │   └── conj.h
│   └── tiling
│       ├── conj_tiling.cpp
│       ├── conj_tiling.h
│       └── tiling_data.h
└── conj_operation.cpp
```

- 新增`ops/include/params/conj.h`文件定义 `Conj` 操作的参数结构体，内容如下：

```c++
#ifndef ASDSIP_PARAMS_CONJ_H
#define ASDSIP_PARAMS_CONJ_H

#include <cstdint>
#include <string>
#include <sstream>
#include <mki/utils/SVector/SVector.h>

namespace AsdSip {
namespace OpParam {
struct Conj {
    int64_t n;
    bool operator==(const Conj &other) const
    {
        return this->n == other.n;
    };

    std::string ToString() const
    {
        std::stringstream ss;
        ss << "OpName: conj";
        return ss.str();
    }
};
}  // namespace OpParam
}  // namespace AsdSip

#endif  // ASDSIP_PARAMS_CONJ_H
```

### 修改文件

- `configs/op_list.yaml`文件直接删除或者增加以下内容（这一操作非常重要，将新增算子信息加入列表，后续构建才会将新增算子的实现和接口真正编译进去）：

```yaml

ConjOperation:
    ConjC64Kernel:
        ascend910b: true
```

- 在`ops/base/CMakeLists.txt`新增以下内容：

```c++
add_subdirectory(conj)
```

- 在`sip/include/base_api.h`新增以下内容：

```c++
AspbStatus Conj(const Tensor &inTensor, Tensor &outTensor, void *stream, uint8_t *workspace)
```

## 环境准备

您可参考[环境准备](../README.md#3--环境构建)进行编译和测试环境搭建，环境准备好之后，就可以开始SiP的算子开发之旅。

## SiP算子实现

SiP算子实现主要包括：kernel侧算子实现和host侧tiling实现。

### tiling开发

tiling开发的核心概念：`TilingData`、`Workspace`、`TilingKey`、`BlockDim`等，可访问[术语表-昇腾社区](https://www.hiascend.com/document/detail/zh/Glossary/gls/gls_0001.html)查看。

#### tiling_data.h

文件路径：`ops/base/conj/conj/tiling/tiling_data.h`  
主要功能：描述算子的输入输出数据的数据结构定义。

```c++
#ifndef ASDOPS_CONJ_TILING_DATA
#define ASDOPS_CONJ_TILING_DATA

#include <cstdint>

namespace AsdSip {
struct ConjTilingData {
    uint32_t dataNum{0};
    uint32_t coreNum{0};
    uint32_t len{0};
    uint32_t tail{0};
};
}
#endif
```

#### conj_tiling.h

文件路径：`ops/base/conj/conj/tiling/conj_tiling.h`  
主要功能：tiling过程主要是完成数据的切分，因此其主体函数是实现切分功能的函数，这里则是函数声明。

```c++
#ifndef ASDOPS_MULTINOMIAL_TILING_H
#define ASDOPS_MULTINOMIAL_TILING_H

#include "mki/kernel_info.h"
#include "mki/launch_param.h"
#include "utils/aspb_status.h"
namespace AsdSip {
AsdSip::AspbStatus ConjTiling(const Mki::LaunchParam &launchParam, Mki::KernelInfo &kernelInfo);
}  // namespace AsdSip
#endif
```

#### conj_tiling.cpp

文件路径：`ops/base/conj/conj/tiling/conj_tiling.cpp`  
主要功能：实现切分功能的主体函数  

```c++
#include "conj_tiling.h"
#include "tiling_data.h"
#include "mki/utils/platform/platform_info.h"
#include "utils/assert.h"
#include "log/log.h"

namespace AsdSip {
using namespace Mki;
AsdSip::AspbStatus ConjTiling(const LaunchParam &launchParam, KernelInfo &kernelInfo)
{
    uint32_t maxCore = static_cast<uint32_t>(PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_VECTOR));
    if (maxCore == 0) {
        maxCore = 1;
    }
    uint32_t size = static_cast<uint32_t>(launchParam.GetInTensor(0).Numel()) * 2;
    uint32_t len = (size / maxCore + 7) / 8 * 8;
    uint32_t seqLenLowerBound = 64;
    if (len < seqLenLowerBound) {
        len = seqLenLowerBound;
    }
    uint32_t needCoreNum = (size + len - 1) / len;
    uint32_t tail = size - len * (needCoreNum - 1);

    ConjTilingData *tilingDataPtr = reinterpret_cast<AsdSip::ConjTilingData *>(kernelInfo.GetTilingHostAddr());
    ASDSIP_CHECK(tilingDataPtr != nullptr, "tilingDataPtr should not be empty",
              return AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);

    tilingDataPtr->coreNum = needCoreNum;
    tilingDataPtr->dataNum = size;
    tilingDataPtr->len = len;
    tilingDataPtr->tail = tail;

    kernelInfo.SetBlockDim(needCoreNum);
    kernelInfo.GetScratchSizes().push_back(0);
    ASDSIP_LOG(DEBUG) << "KernelInfo:\n" << kernelInfo.ToString();

    return AsdSip::ErrorType::ACL_SUCCESS;
}
}  // namespace AsdSip
```

### kernel开发

kernel相关的`Compute`、`CopyIn`、`CopyOut`等概念，可访问[术语表-昇腾社区](https://www.hiascend.com/document/detail/zh/Glossary/gls/gls_0001.html)查看。

#### conj.cpp

文件路径：`ops/base/conj/conj/op_kernel/conj.cpp`  
主要功能：核函数入口。

```c++
#include "../tiling/tiling_data.h"
#include "conj.h"

using namespace AscendC;

inline __aicore__ void InitTilingData(const __gm__ uint8_t *pTilingdata, AsdSip::ConjTilingData *tilingdata)
{
#if defined(__CCE_KT_TEST__) || (__CCE_AICORE__ == 220)
    tilingdata->dataNum = (*(const __gm__ uint32_t *)(pTilingdata + 0));
    tilingdata->coreNum = (*(const __gm__ uint32_t *)(pTilingdata + 4));
    tilingdata->len = (*(const __gm__ uint32_t *)(pTilingdata + 8));
    tilingdata->tail = (*(const __gm__ uint32_t *)(pTilingdata + 12));
#else
    __ubuf__ uint8_t *tilingdataInUb = (__ubuf__ uint8_t *)get_imm(0);
    int32_t tilingBlockNum = sizeof(AsdSip::ConjTilingData) / 32 + 1;
    copy_gm_to_ubuf(((__ubuf__ uint8_t *)tilingdataInUb), pTilingdata, 0, 1, tilingBlockNum, 0, 0);
    pipe_barrier(PIPE_ALL);
    tilingdata->dataNum = (*(__ubuf__ uint32_t *)((__ubuf__ uint8_t *)tilingdataInUb + 0));
    tilingdata->coreNum = (*(__ubuf__ uint32_t *)((__ubuf__ uint8_t *)tilingdataInUb + 4));
    tilingdata->len = (*(__ubuf__ uint32_t *)((__ubuf__ uint8_t *)tilingdataInUb + 8));
    tilingdata->tail = (*(__ubuf__ uint32_t *)((__ubuf__ uint8_t *)tilingdataInUb + 12));
    pipe_barrier(PIPE_ALL);
#endif
}

extern "C" __global__ __aicore__ void conj(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    AsdSip::ConjTilingData tilingData;
    InitTilingData(tiling, &tilingData);

    Conj::Conj op;
    op.Init(x, y, &tilingData);
    op.Process();
}
```

#### conj.h

文件路径：`ops/base/conj/conj/op_kernel/conj.h`
主要功能：核函数主要功能实现。

```c++
#ifndef CONJ_N_D_H
#define CONJ_N_D_H

#include <type_traits>
#include "kernel_operator.h"

namespace Conj {
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t BYTE_BLOCK = 32;
constexpr int32_t BYTES_PER_REPEAT = 256;
constexpr int32_t MAX_CAST_COUNT = 512;
constexpr uint32_t MAX_DATA_COUNT = 8 * 1024;

class Conj {
public:
    __aicore__ inline Conj(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const AsdSip::ConjTilingData *tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(uint32_t offset, uint32_t dataCount);
    __aicore__ inline void Compute(uint32_t dataCount);
    __aicore__ inline void CopyOut(uint32_t offset, uint32_t dataCount);

    template <typename T1, typename T2>
    __aicore__ inline T1 CeilA2B(T1 a, T2 b)
    {
        if (b == 0) {
            return a;
        }
        return (a + b - 1) / b;
    };

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> dataQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue;
    GlobalTensor<float> inTensorsGM;
    GlobalTensor<float> outTensorsGM;

    int64_t blockIdx = 0;

    // tiling params
    uint32_t coreNum = 0;
    uint32_t dataNum = 0;
    uint32_t len = 0;
    uint32_t blockOffset = 0;
};

__aicore__ inline void Conj::Init(GM_ADDR x, GM_ADDR y, const AsdSip::ConjTilingData *tilingData)
{
    blockIdx = GetBlockIdx();

    inTensorsGM.SetGlobalBuffer((__gm__ float *)x);
    outTensorsGM.SetGlobalBuffer((__gm__ float *)y);

    pipe.InitBuffer(dataQueue, BUFFER_NUM, MAX_DATA_COUNT * sizeof(float));
    pipe.InitBuffer(outQueue, BUFFER_NUM, MAX_DATA_COUNT * sizeof(float));

    dataNum = tilingData->dataNum;
    coreNum = tilingData->coreNum;

    len = tilingData->len;
    blockOffset = len * blockIdx;
    if (blockIdx == coreNum - 1) {
        len = tilingData->tail;
    }
}

__aicore__ inline void Conj::Process()
{
    uint32_t times = len / MAX_DATA_COUNT;
    uint32_t reminder = len % MAX_DATA_COUNT;

    uint32_t offset = blockOffset;
    for (uint32_t i = 0; i < times; i++) {
        CopyIn(offset, MAX_DATA_COUNT);
        Compute(MAX_DATA_COUNT);
        CopyOut(offset, MAX_DATA_COUNT);
        offset += MAX_DATA_COUNT;
    }

    if (reminder > 0) {
        uint32_t dataCount = CeilA2B(reminder, 8) * 8;
        CopyIn(offset, dataCount);
        Compute(dataCount);
        CopyOut(offset, dataCount);
    }
}

__aicore__ inline void Conj::CopyIn(uint32_t offset, uint32_t dataCount)
{
    LocalTensor<float> dataLocal = dataQueue.AllocTensor<float>();
    DataCopy(dataLocal, inTensorsGM[offset], dataCount);
    dataQueue.EnQue(dataLocal);
}

__aicore__ inline void Conj::Compute(uint32_t dataCount)
{
    LocalTensor<float> dataLocal = dataQueue.DeQue<float>();
    LocalTensor<float> outLocal = outQueue.AllocTensor<float>();

    uint64_t mask[2] = {6148914691236517205, 0};
    uint64_t mask_sub[2] = {__UINT64_C(12297829382473034410), 0};
    uint64_t repeatTimes = (dataCount * sizeof(float) + BYTES_PER_REPEAT - 1) / BYTES_PER_REPEAT;
    pipe_barrier(PIPE_V);
    Duplicate<float>(outLocal, 0, dataCount);
    pipe_barrier(PIPE_V);
    Copy(outLocal, dataLocal, mask, repeatTimes, {1, 1, 8, 8});
    pipe_barrier(PIPE_V);
    Sub(outLocal, outLocal, dataLocal, mask_sub, repeatTimes, {1, 1, 1, 8, 8, 8});
    pipe_barrier(PIPE_V);

    dataQueue.FreeTensor(dataLocal);
    outQueue.EnQue<float>(outLocal);
}

__aicore__ inline void Conj::CopyOut(uint32_t offset, uint32_t dataCount)
{
    LocalTensor<float> outLocal = outQueue.DeQue<float>();
    DataCopy(outTensorsGM[offset], outLocal, dataCount);
    outQueue.FreeTensor(outLocal);
}
}
#endif  // CONJ_N_D_H
```

#### conj_operation.cpp

文件路径：`ops/base/conj/conj_operation.cpp`  
主要功能：选择最佳的kernel函数（核函数）。

```c++
#include "utils/assert.h"
#include "mki/base/operation_base.h"
#include "log/log.h"
#include "mki_loader/op_register.h"
#include "mki/utils/SVector/SVector.h"
#include "conj.h"

namespace AsdSip {
using namespace Mki;
class ConjOperation : public OperationBase {
public:
    explicit ConjOperation(const std::string &opName) noexcept : OperationBase(opName) {}
    Kernel *GetBestKernel(const LaunchParam &launchParam) const override
    {
        ASDSIP_CHECK(IsConsistent(launchParam), "Failed to check consistent", return nullptr);
        return GetKernelByName("ConjC64Kernel");
    }

protected:
    Status InferShapeImpl(const LaunchParam &launchParam, SVector<Tensor> &outTensors) const override
    {
        const Any &specificParam = launchParam.GetParam();
        ASDSIP_CHECK(specificParam.Type() == typeid(OpParam::Conj), "OpParam is invalid",
                  return Status::FailStatus(ERROR_INVALID_VALUE));

        outTensors[0].desc.dtype = launchParam.GetInTensor(0).desc.dtype;
        outTensors[0].desc.format = launchParam.GetInTensor(0).desc.format;
        outTensors[0].desc.dims = launchParam.GetInTensor(0).desc.dims;

        return Status::OkStatus();
    }
};
REG_OPERATION(ConjOperation);
}  //    namespace AsdSip
```

### conj_kernel.cpp

文件路径：`ops/base/conj/conj/conj_kernel.cpp`  
主要功能：执行tiling，入参校验等。

```c++
#include "mki/base/kernel_base.h"
#include "mki_loader/op_register.h"
#include "utils/assert.h"
#include "log/log.h"
#include "utils/assert.h"
#include "conj.h"
#include "tiling/conj_tiling.h"
#include "tiling/tiling_data.h"

static constexpr uint32_t TENSOR_INPUT_NUM = 1;
static constexpr uint32_t TENSOR_OUTPUT_NUM = 1;
namespace AsdSip {
using namespace Mki;
class ConjKernel : public KernelBase {
public:
    explicit ConjKernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : KernelBase(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        ASDSIP_CHECK(launchParam.GetInTensorCount() == TENSOR_INPUT_NUM, "check inTensor count failed", return false);
        ASDSIP_CHECK(launchParam.GetOutTensorCount() == TENSOR_OUTPUT_NUM,
            "check outTensor count failed", return false);
        ASDSIP_CHECK(launchParam.GetParam().Type() == typeid(OpParam::Conj), "check param type failed!", return false);
        return true;
    }

    uint64_t GetTilingSize(const LaunchParam &launchParam) const override
    {
        (void)launchParam;
        return sizeof(ConjTilingData);
    }

    Status InitImpl(const LaunchParam &launchParam) override
    {
        auto status = ConjTiling(launchParam, kernelInfo_);
        ASDSIP_CHECK(status == AsdSip::ErrorType::ACL_SUCCESS, "InitRunInfoImpl ConjTiling failed",
                    return Status::FailStatus(ERROR_INVALID_VALUE));
        return Status::OkStatus();
    }
};

// ConjC64Kernel
class ConjC64Kernel : public ConjKernel {
public:
    explicit ConjC64Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : ConjKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        ASDSIP_CHECK(ConjKernel::CanSupport(launchParam), "failed to check support", return false);
        ASDSIP_CHECK(launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_COMPLEX64, "tensor dtype unsupported",
                  return false);
        return true;
    }
};
REG_KERNEL_BASE(ConjC64Kernel);

}  // namespace AsdSip
```

### CMakeLists.txt

文件路径：`ops/base/conj/CMakeLists.txt`  
主要功能：文件编译。

```cpp
set(conj_src
    ${CMAKE_CURRENT_LIST_DIR}/conj_operation.cpp
    ${CMAKE_CURRENT_LIST_DIR}/conj/conj_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/conj/tiling/conj_tiling.cpp
)

add_operation(ConjOperation "${conj_src}")

add_kernel(conj ascend910b vector
    conj/op_kernel/conj.cpp
    ConjC64Kernel)
```

### SiP编译与环境变量设置

SiP仓的构建脚本文件为`build.sh`，脚本使用的基本命令是：

```shell
bash build.sh
```
