# Operator Usage Guide

<!-- md-trans-meta sourceCommit=cec88e057607a630073cce4bbace3c21f8d93fe7 translatedAt=2026-08-12T10:59:46.269Z pushedAt=2026-08-20T11:47:59.842Z -->

After installing and deploying the signal processing acceleration library package, users can call the operator APIs provided by the package to implement high-performance, high-precision signal processing applications. This section introduces basic operator calling examples.

## Security Statement

If a user does not call operators in accordance with the standard process sequence in real business scenarios and a security issue occurs, the user shall bear the consequences themselves.

## Operator Calling Flow

![Operator Calling Flow](../API_Reference/figures/en-us_image_0000002537574354.png)

## Procedure

- Create a new sample file named `example.cpp` as shown below.

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
    // Boilerplate: Initialize ACL.
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
    // Call aclrtMalloc to allocate device-side memory.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // Call aclrtMemcpy to copy host-side data to device-side memory.
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

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

int main(int argc, char **argv)
{
    // Set the device ID used by the operator.
    int deviceId = 0;
    // (Standard pattern) Create an execution stream.
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // Create the host-side data for the tensor.
    int64_t n = 5;
    int64_t incx = 1;
    int64_t incy = 1;

    int64_t xSize = 5;
    std::vector<float> tensorInXData;
    tensorInXData.reserve(xSize);
    for (int64_t i = 0; i < xSize; i++) {
        tensorInXData[i] = 1.0 + i;
    }

    int64_t ySize = 5;
    std::vector<float> tensorInYData;
    tensorInYData.reserve(xSize);
    for (int64_t i = 0; i < ySize; i++) {
        tensorInYData[i] = 10.0 + i;
    }

    int64_t resultSize = 1;
    std::vector<float> resultData;
    resultData.reserve(resultSize);

    std::cout << "------- input x -------" << std::endl;
    for (int64_t i = 0; i < xSize; i++) {
        std::cout << tensorInXData[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "------- input y -------" << std::endl;
    for (int64_t i = 0; i < ySize; i++) {
        std::cout << tensorInYData[i] << " ";
    }
    std::cout << std::endl;

    // Create input/output tensors.
    std::vector<int64_t> xShape = {xSize};
    std::vector<int64_t> yShape = {ySize};
    std::vector<int64_t> resultShape = {resultSize};
    aclTensor *inputX = nullptr;
    aclTensor *inputY = nullptr;
    aclTensor *result = nullptr;
    void *inputXDeviceAddr = nullptr;
    void *inputYDeviceAddr = nullptr;
    void *resultDeviceAddr = nullptr;
    ret = CreateAclTensor(tensorInXData, xShape, &inputXDeviceAddr, aclDataType::ACL_FLOAT, &inputX);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);
    ret = CreateAclTensor(tensorInYData, yShape, &inputYDeviceAddr, aclDataType::ACL_FLOAT, &inputY);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);
    ret = CreateAclTensor(resultData, resultShape, &resultDeviceAddr, aclDataType::ACL_FLOAT, &result);
    CHECK_RET(ret == ::ACL_SUCCESS, return ret);

    // Create the operator execution handle.
    asdBlasHandle handle;
    asdBlasCreate(handle);

    // Create the workspace required for operator execution.
    size_t lwork = 0;
    void *buffer = nullptr;
    asdBlasMakeDotPlan(handle);
    asdBlasGetWorkspaceSize(handle, lwork);
    std::cout << "lwork = " << lwork << std::endl;
    if (lwork > 0) {
        ret = aclrtMalloc(&buffer, static_cast<int64_t>(lwork), ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    asdBlasSetWorkspace(handle, buffer);

    // Configure the operator execution information.
    asdBlasSetStream(handle, stream);

    // Call the API to execute the operator (fixed calling logic).
    ASD_STATUS_CHECK(asdBlasSdot(handle, n, inputX, incx, inputY, incy, result));
    asdBlasSynchronize(handle);

    // Destroy the operator handle after scheduling the operator.
    asdBlasDestroy(handle);

    // Copy the output tensor data from the device side to the host side memory.
    ret = aclrtMemcpy(resultData.data(),
        resultSize * sizeof(float),
        resultDeviceAddr,
        resultSize * sizeof(float),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

    std::cout << "------- result -------" << std::endl;
    for (int64_t i = 0; i < 1; i++) {
        std::cout << resultData[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "Execute successfully." << std::endl;

    // Release resources.
    aclDestroyTensor(inputX);
    aclDestroyTensor(inputY);
    aclDestroyTensor(result);
    aclrtFree(inputXDeviceAddr);
    aclrtFree(inputYDeviceAddr);
    aclrtFree(resultDeviceAddr);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```

- Run the following command to compile the example code.

```Cpp
g++  example.cpp \
    -I${ASCEND_HOME_PATH}/include/aclnn \
    -I${ASCEND_HOME_PATH}/include \
    -L${ASCEND_HOME_PATH}/lib64/ -lascendcl -lopapi -lnnopbase \
    -I${ASDSIP_HOME_PATH}/include \
    -L${ASDSIP_HOME_PATH}/lib -lmki \
    -L${ASDSIP_HOME_PATH}/lib -lasdsip \
    -L${ASDSIP_HOME_PATH}/lib -lasdsip_core \
    -L${ASDSIP_HOME_PATH}/lib -lasdsip_host \
    -o example
```

- Run the example:

```Cpp
{PATH_TO_EXAMPLE}/example
```

- The output result is as follows, indicating that the dot product of `x` and `y` is 190:

```Cpp

------- input x -------
1 2 3 4 5
------- input y -------
10 11 12 13 14

------- result -------
190

```
