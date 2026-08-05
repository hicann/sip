# AscendSiPBoost Signal Processing Acceleration Library

🔥 [2025/10] The AscendSiPBoost project (hereinafter referred to as the SiP library) is launched for the first time.

## Table of Contents

1. [Learning Resources](#study)
2. [What is SiP](#sip)
3. [Environment Setup](#build)
4. [Quick Start](#learn)
5. [Custom Operator Development](#develop)
6. [Contributing](#contribute)
7. [Reference Documentation](#doc)

## <a id="study"></a>1.  Learning Resources

- [Compilation and Build](./docs/compilation_build_en.md): Compilation command instructions for the SiP library.
- [API Documentation](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/API/SiP/SIP_0000.html): Introduces the interfaces and related terminology of the SiP library.
- [Issue Tracker](https://gitcode.com/cann/sip/issues): Submit discovered issues through issues.

## <a id="sip"></a>2.  What is SiP

Ascend Signal Processing Boost (hereinafter referred to as the SiP library) is built based on Huawei Ascend AI processors. It is deeply adapted to hardware computing power, storage, and memory bandwidth characteristics. It provides high-performance NPU operators such as FFT, BLAS, FIR filtering, and interpolation to deliver efficient and reliable computing acceleration for the signal processing domain.

The interface functions of the acceleration library are mainly divided into six parts:

- Signal Processing Acceleration Library Framework: Manages operators, handles binary loading of operators on the Device side and tiling on the Host side, and provides upper-layer interfaces to support single operator calls and batch calls of multiple operators.
- FFT Library: Includes a dedicated NPU Kernel and PLAN framework. Implements the FFT series operators and provides interfaces to support C2C, C2R, and R2C functions for developers.
- BLAS Library: Provides dedicated Kernels according to BLAS-related standard definitions, implements BLAS series operator functions, and provides interfaces from level 1 to level 3 for developers.
- Complex Basic Computation Library: Provides basic complex type operator support.
- Signal Domain Fusion Operator Library: Contains fusion operators such as PC, MTD, CFAR, and Interpolation, supporting scenarios such as pulse signal analysis, dynamic target detection, and constant false alarm rate.
- Solver Library: Primarily provides complex linear algebra functions based on BLAS, such as matrix decomposition and eigenvalue computation.

## <a id="build"></a>3.  Environment Setup

### 3.1  Quick Installation of CANN Software

This section provides sample commands for quickly installing CANN software. For more installation steps, refer to the [CANN Software Installation Guide](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/softwareinst/instg/instg_0000.html?OS=openEuler&InstallType=netyum) on the CANN official website.

#### 3.1.1  Pre-installation Preparations

The dependencies required for source code compilation in this project are listed below. Pay attention to the version requirements.

   - python >= 3.7.0
   - pyyaml
   - gcc >= 7.3.0
   - g++ >= 7.3.0
   - cmake >= 3.16.0
   - pigz (Installation can speed up packaging. Recommended version >= 2.4)
   - dos2unix
   - numpy
   - googletest (Only required when running UT. Recommended version [release-1.11.0](https://github.com/google/googletest/releases/tag/release-1.11.0))

You can install the above dependencies with one click using the project script. Follow these steps:

```sh
bash install_deps.sh
```

After the installation completes, continue to install Python third-party library dependencies through the requirements.txt in the project root directory. The command is as follows:

```sh
pip3 install -r requirements.txt
```

#### 3.1.2  Install the Community Edition CANN Toolkit Package

- For Atlas A2/A3 series products: Click the [download link](https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-mirror/software/master/) to obtain the software package. Select the latest version and download the corresponding package based on the product model and environment architecture.

```bash
# Ensure the installation package has executable permissions
chmod +x Ascend-cann-toolkit_${cann_version}_linux-${arch}.run
# Installation command
./Ascend-cann-toolkit_${cann_version}_linux-${arch}.run --install --force --install-path=${install_path}
```

- \$\{cann\_version\}: Indicates the CANN package version number.
- \$\{arch\}: Indicates the CPU architecture, such as aarch64 or x86_64.
- \$\{install\_path\}: Indicates the specified installation path. The default installation directory is `/usr/local/Ascend`.

#### 3.1.3  Install the Community Edition CANN Ops Package

- For Atlas A2/A3 series products: Click the [download link](https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-mirror/software/master/) to obtain the software package. Select the latest version and download the corresponding package based on the product model and environment architecture.

```bash
# Ensure the installation package has executable permissions
chmod +x Ascend-cann-${soc_name}-ops_${cann_version}_linux-${arch}.run
# Installation command
./Ascend-cann-${soc_name}-ops_${cann_version}_linux-${arch}.run --install --install-path=${install_path}
```

- \$\{soc\_name\}: Indicates the NPU model name, which is the remaining content after removing "ascend" from \$\{soc\_version\}.
- \$\{install\_path\}: Indicates the specified installation path. It must be the same path where the toolkit package is installed. The default installation directory is `/usr/local/Ascend`.

#### 3.1.4  Environment Variable Configuration

```bash
# Default path installation, using root user as an example (for non-root users, replace /usr/local with ${HOME})
source /usr/local/Ascend/cann/set_env.sh
# Specified path installation
# source ${install_path}/cann/set_env.sh
```

#### 3.1.5  Tool Version Requirements and Installation

After installing CANN, you can install some tools to facilitate subsequent development. For details, refer to the following:

* [CANN Dependency List](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/softwareinst/instg/instg_0045.html?OS=Debian&InstallType=netapt)
* [Post-installation Operations for CANN](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/softwareinst/instg/instg_0094.html?OS=Debian&InstallType=netapt)

## <a id="learn"></a>4.  Quick Start

### 4.1  SiP Compilation

 - Download the acceleration library

    ```sh
    git clone https://gitcode.com/cann/sip.git
    ```

    You can select any branch as needed.
 - SiP library compilation<br>
    Compile the acceleration library and set the acceleration library environment variables:

    ```sh
    cd ${sip_root_path}
    bash build.sh
    source output/set_env.sh
    ```

    Special notes:
    - `Ascend-cann-SIP_${version}_linux_${arch}.run` is an executable product packaged by makeself after compilation. It contains the files required for operator execution. You can run the package with the following command to install the operator information files to the specified directory.

        ```bash
        # Ensure the installation package has executable permissions
        chmod +x Ascend-cann-SIP_${version}_linux_${arch}.run
        # Installation command
        ./Ascend-cann-SIP_${version}_linux_${arch}.run --install --install-path=${install_path}
        ```

    - The above compilation method only supports compiling the acceleration library downloaded via git. It does not support the compilation of acceleration libraries downloaded as zip archives.
    - The compilation process requires network access to download dependency libraries, so the compilation environment must have network connectivity.
    - The compilation process includes two steps: obtaining and compiling the ascend-boost-comm (Ascend Distributed Communication Acceleration Library) component, and compiling the signal acceleration library. For more command introductions, refer to the `build.sh` file in the SiP repository.

 - For more compilation command descriptions, refer to [Compilation and Build](./docs/compilation_build_en.md)

### 4.2  Call Example Description

The sample code in this section demonstrates how to call operators through C++.

#### 4.2.1  C++

In the `example` directory of the SiP repository, there are multiple operator invocation demo samples that do not depend on a test framework and can be compiled and used immediately. The sample code in this section demonstrates calling the SiP asdBlasSdot operator through C++ to implement vector dot product (inner product) functionality. For the complete code, refer to [example](example/example.cpp). Only the core content is shown below:

```c++
int main(int argc, char **argv)
{
    // Set the device ID used by the operator
    int deviceId = 0;
    // (Fixed写法) Create execution stream
    aclrtStream stream;
    Init(deviceId, &stream);

    // Create Host-side data for tensors
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

    // Create input/output tensors
    std::vector<int64_t> xShape = {xSize};
    std::vector<int64_t> yShape = {ySize};
    std::vector<int64_t> resultShape = {resultSize};
    aclTensor *inputX = nullptr;
    aclTensor *inputY = nullptr;
    aclTensor *result = nullptr;
    void *inputXDeviceAddr = nullptr;
    void *inputYDeviceAddr = nullptr;
    void *resultDeviceAddr = nullptr;
    CreateAclTensor(tensorInXData, xShape, &inputXDeviceAddr, aclDataType::ACL_FLOAT, &inputX);
    CreateAclTensor(tensorInYData, yShape, &inputYDeviceAddr, aclDataType::ACL_FLOAT, &inputY);
    CreateAclTensor(resultData, resultShape, &resultDeviceAddr, aclDataType::ACL_FLOAT, &result);

    // Create operator execution handle
    asdBlasHandle handle;
    asdBlasCreate(handle);

    // Create the workspace required for operator execution
    size_t lwork = 0;
    void *buffer = nullptr;
    asdBlasMakeDotPlan(handle);
    asdBlasGetWorkspaceSize(handle, lwork);
    if (lwork > 0) {
        aclrtMalloc(&buffer, static_cast<int64_t>(lwork), ACL_MEM_MALLOC_HUGE_FIRST);
    }
    asdBlasSetWorkspace(handle, buffer);

    // Configure operator execution information
    asdBlasSetStream(handle, stream);

    // Call the interface to execute the operator (fixed invocation logic)
    asdBlasSdot(handle, n, inputX, incx, inputY, incy, result);
    asdBlasSynchronize(handle);

    // Destroy the operator handle after calling the operator
    asdBlasDestroy(handle);

    // Copy the Device-side data of the output tensor to Host-side memory
    aclrtMemcpy(resultData.data(),
        resultSize * sizeof(float),
        resultDeviceAddr,
        resultSize * sizeof(float),
        ACL_MEMCPY_DEVICE_TO_HOST);

    std::cout << "------- result -------" << std::endl;
    for (int64_t i = 0; i < 1; i++) {
        std::cout << resultData[i] << " ";
    }
    std::cout << std::endl;

    // Resource release
    aclDestroyTensor(inputX);
    aclDestroyTensor(inputY);
    aclDestroyTensor(result);
    aclrtFree(inputXDeviceAddr);
    aclrtFree(inputYDeviceAddr);
    aclrtFree(resultDeviceAddr);
    if (lwork > 0) {
        aclrtFree(buffer);
    }

    // Reset the deviceId used by the operator after calling the operator
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```

File compilation instructions: Enter the example directory and run `bash build.sh` to complete compilation and execution.

```shell
cd example
bash build.sh
```

Operator usage guide: Visit [Header File List - CANN Community Edition - Ascend Community](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/API/SiP/SIP_0000.html).

#### 4.2.2  Sample Security Statement

The samples in the `example` directory are designed to provide a minimal implementation for quickly getting started, developing, and debugging SiP features. The core goal is to demonstrate SiP core functionality using the most concise code, **not to provide production-level security guarantees**. Compared with mature production-level usage methods, the security features in this sample (such as input validation and boundary validation) are relatively limited.

SiP does not recommend that users directly use the samples as production code, nor does it guarantee the security of such usage. If users apply the sample code from the `example` directory in their own real business scenarios and encounter security issues, the users bear the responsibility themselves.

### 4.3  Logs and Environment Variable Description

- The acceleration library logs have now been partially adapted to CANN logs. For the environment variable description, refer to
  **[CANN Community Edition Documentation / Environment Variable Reference](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/maintenref/envvar/envref_07_0001.html)**.

## <a id="develop"></a>5.  Custom Operator Development

For detailed steps, refer to [Developing a Simple Operator from Scratch](./docs/developing_a_simple_operator_en.md)

## <a id="contribute"></a>6.  Contributing

1. Fork the repository
2. Modify and submit code
3. Create a Pull-Request

For detailed steps, refer to [Contribution Guide](./docs/contributing_guide_en.md)

## <a id="doc"></a>7.  Reference Documentation

**[CANN Community Edition Documentation](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/index/index.html)**
**[SiP Community Edition Documentation](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/API/SiP/SIP_0000.html)**
