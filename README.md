# AscendSiPBoost 信号处理加速库

🔥 [2025/10] AscendSiPBoost项目（下文简称为SiP库）首次上线。

## 内容总览

1. [学习资源](#study)
2. [什么是SiP](#sip)
3. [环境构建](#build)
4. [快速上手](#learn)
5. [自定义算子开发](#develop)
6. [参与贡献](#contribute)
7. [参考文档](#doc)

## <a id="study"></a>1.  学习资源

- [编译与构建](./docs/compilation_build.md)：SiP库的编译命令说明。
- [API文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/API/SiP/SIP_0000.html)：介绍了SiP库的接口和相关术语。
- [问题报告](https://gitcode.com/cann/sip/issues)：通过issue提交发现的问题。

## <a id="sip"></a>2.  什么是SiP

Ascend Signal Processing Boost（昇腾信号处理加速库，下文简称为SiP库）基于华为Ascend AI处理器打造，深度适配硬件算力、存储及内存带宽特性，提供FFT、BLAS、FIR滤波、插值等高性能NPU算子，为信号处理领域提供高效可靠的算力加速。

加速库接口功能主要分成六个部分：

- 信号处理加速库框架：负责算子的管理，算子在Device侧的二进制加载以及Host侧的tiling；负责对上层提供接口支持单算子调用、多算子批量调用等。
- FFT库：包括专用的NPU Kernel、PLAN框架；实现FFT系列算子，对外提供接口以支持C2C、C2R和R2C功能，供开发者使用。
- BLAS库：依照BLAS相关的标准定义，提供专用的Kernel，实现BLAS系列算子功能，对外提供从level1到level3的接口，供开发者使用。
- 复数基础计算库：提供基础的复数类型算子支持。
- 信号领域融合算子库：包含PC、MTD、CFAR、Interpolation等融合算子，支撑脉冲信号分析，动态目标检测，恒虚警等场景。
- Solver库：主要提供基于BLAS的复杂线性代数函数，例如矩阵分解、特征值求解等。

## <a id="build"></a>3.  环境构建

### 3.1  快速安装CANN软件

本节提供快速安装CANN软件的示例命令，更多安装步骤请参考CANN官网的[CANN软件安装指南](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/softwareinst/instg/instg_0000.html?OS=openEuler&InstallType=netyum)。

#### 3.1.1  安装前准备

本项目源码编译用到的依赖如下，请注意版本要求。

   - python >= 3.7.0
   - pyyaml
   - gcc >= 7.3.0
   - g++ >= 7.3.0
   - cmake >= 3.16.0
   - pigz（安装后可提升打包速度，建议版本 >= 2.4）
   - dos2unix
   - numpy
   - googletest（仅执行UT时依赖，建议版本 [release-1.11.0](https://github.com/google/googletest/releases/tag/release-1.11.0)）

上述依赖可通过项目脚本一键安装，操作步骤如下：

```sh
bash install_deps.sh
```

安装完后，再通过项目根目录requirements.txt继续安装python三方库依赖，命令如下:

```sh
pip3 install -r requirements.txt
```

#### 3.1.2  安装社区版CANN toolkit包

- Atlas A2/A3系列产品：单击[下载链接](https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-mirror/software/master/)获取软件包，选择最新时间版本，并根据产品型号和环境架构下载对应包。

```bash
# 确保安装包具有可执行权限
chmod +x Ascend-cann-toolkit_${cann_version}_linux-${arch}.run
# 安装命令
./Ascend-cann-toolkit_${cann_version}_linux-${arch}.run --install --force --install-path=${install_path}
```

- \$\{cann\_version\}：表示CANN包版本号。
- \$\{arch\}：表示CPU架构，如aarch64、x86_64。
- \$\{install\_path\}：表示指定安装路径，默认安装在`/usr/local/Ascend`目录。

#### 3.1.3  安装社区版CANN ops包

- Atlas A2/A3系列产品：单击[下载链接](https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-mirror/software/master/)获取软件包，选择最新时间版本，并根据产品型号和环境架构下载对应包。

```bash
# 确保安装包具有可执行权限
chmod +x Ascend-cann-${soc_name}-ops_${cann_version}_linux-${arch}.run
# 安装命令
./Ascend-cann-${soc_name}-ops_${cann_version}_linux-${arch}.run --install --install-path=${install_path}
```

- \$\{soc\_name\}：表示NPU型号名称，即\$\{soc\_version\}删除“ascend”后剩余的内容。
- \$\{install\_path\}：表示指定安装路径，需要与toolkit包安装在相同路径，默认安装在`/usr/local/Ascend`目录。

#### 3.1.4  环境变量配置

```bash
# 默认路径安装，以root用户为例（非root用户，将/usr/local替换为${HOME}）
source /usr/local/Ascend/cann/set_env.sh
# 指定路径安装
# source ${install_path}/cann/set_env.sh
```

#### 3.1.5  基础工具版本要求与安装

安装CANN之后，您可安装一些工具方便后续开发，参见以下内容：

* [CANN依赖列表](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/softwareinst/instg/instg_0045.html?OS=Debian&InstallType=netapt)
* [CANN安装后操作](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/softwareinst/instg/instg_0094.html?OS=Debian&InstallType=netapt)

## <a id="learn"></a>4.  快速上手

### 4.1  SiP编译

 - 加速库下载

    ```sh
    git clone https://gitcode.com/cann/sip.git
    ```

   您可自行选择需要的分支。
 - SiP库编译<br>
    编译加速库：

    ```sh
    cd ${sip_root_path}
    bash build.sh
    ```

    编译前请根据目标硬件修改[`configs/build_config.json`](configs/build_config.json)中的芯片架构配置，将不支持的芯片架构设置为`false`。默认配置同时启用`ascend910b`和`ascend950`，若运行环境仅支持其中一种芯片，需关闭不支持的架构，否则编译会失败。也可通过环境变量`BUILD_CONFIG_FILE`指定自定义配置文件路径：

    ```sh
    export BUILD_CONFIG_FILE=/path/to/your/build_config.json
    bash build.sh
    ```

    特别说明：
    - `Ascend-cann-SIP_${version}_linux_${arch}.run`是编译后通过makeself打包的包含算子执行所需文件的可执行产物，用户可以通过以下命令执行run包将包含算子信息的文件安装到指定目录。

        ```bash
        # 确保安装包具有可执行权限
        chmod +x Ascend-cann-SIP_${version}_linux_${arch}.run
        # 安装命令
        ./Ascend-cann-SIP_${version}_linux_${arch}.run --install --install-path=${install_path}
        # 设置加速库环境变量
        source ${install_path}/asdsip/set_env.sh
        ```

    - 上述编译方式仅支持编译通过git下载的加速库，以zip压缩包方式下载的加速库不支持该编译方式；
    - 由于编译过程需要联网下载依赖库，因此编译环境需要联网；
    - 该编译过程包括获取ascend-boost-comm（昇腾分布式通信加速库）组件并编译该组件，和编译信号加速库两个步骤。更多命令介绍可查看SiP仓库`build.sh`文件。

 - 更多编译命令说明请参考[编译与构建](./docs/compilation_build.md)

### 4.2  调用示例说明

本节示例代码分别展示了如何通过C++调用算子。

#### 4.2.1  C++

在SiP仓库的`example`目录下，存放了多个不依赖测试框架、即编可用的算子调用Demo示例。本节示例代码展示通过C++调用SiP asdBlasSdot算子实现向量点乘（内积）功能，代码完整内容可参考[example](example/example.cpp)，下面仅展示其核心内容：

```c++
int main(int argc, char **argv)
{
    // 设置算子使用的device id
    int deviceId = 0;
    //（固定写法）创建执行流
    aclrtStream stream;
    Init(deviceId, &stream);

    // 创造tensor的Host侧数据
    int64_t n = 5;
    int64_t incx = 1;
    int64_t incy = 1;

    int64_t xSize = 5;
    std::vector<float> tensorInXData;
    tensorInXData.resize(xSize);
    for (int64_t i = 0; i < xSize; i++) {
        tensorInXData[i] = 1.0 + i;
    }

    int64_t ySize = 5;
    std::vector<float> tensorInYData;
    tensorInYData.resize(xSize);
    for (int64_t i = 0; i < ySize; i++) {
        tensorInYData[i] = 10.0 + i;
    }

    int64_t resultSize = 1;
    std::vector<float> resultData;
    resultData.resize(resultSize);

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

    // 创造输入/输出tensor
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

    // 创建算子执行句柄
    asdBlasHandle handle;
    asdBlasCreate(handle);

    // 创造算子执行所需workspace
    size_t lwork = 0;
    void *buffer = nullptr;
    asdBlasMakeDotPlan(handle);
    asdBlasGetWorkspaceSize(handle, lwork);
    if (lwork > 0) {
        aclrtMalloc(&buffer, static_cast<int64_t>(lwork), ACL_MEM_MALLOC_HUGE_FIRST);
    }
    asdBlasSetWorkspace(handle, buffer);

    // 配置算子执行信息
    asdBlasSetStream(handle, stream);

    // 调用接口执行算子（固定调用逻辑）
    asdBlasSdot(handle, n, inputX, incx, inputY, incy, result);
    asdBlasSynchronize(handle);

    // 调用算子后销毁算子句柄
    asdBlasDestroy(handle);

    // 将输出tensor的Device侧数据复制到Host侧内存上
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

    // 资源释放
    aclDestroyTensor(inputX);
    aclDestroyTensor(inputY);
    aclDestroyTensor(result);
    aclrtFree(inputXDeviceAddr);
    aclrtFree(inputYDeviceAddr);
    aclrtFree(resultDeviceAddr);
    if (lwork > 0) {
        aclrtFree(buffer);
    }

    // 调用算子后重置算子使用的deviceId
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```

文件编译说明：进入example目录，执行bash build.sh完成编译和执行。

```shell
cd example
bash build.sh
```

算子使用指导：可访问[头文件列表-CANN社区版-昇腾社区](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/API/SiP/SIP_0000.html)。

#### 4.2.2  样例安全声明

`example`目录下的样例旨在提供快速上手、开发和调试SiP特性的最小化实现，其核心目标是使用最精简的代码展示SiP核心功能，**而非提供生产级的安全保障**。与成熟的生产级使用方法相比，此样例中的安全功能（如输入校验、边界校验）相对有限。

SiP不推荐用户直接将样例作为业务代码，也不保证此种做法的安全性。若用户将`example`中的示例代码应用在自身的真实业务场景中且发生了安全问题，则由用户自行承担。

### 4.3  日志和环境变量说明

- 加速库日志现在已经部分适配CANN日志，环境变量说明请参考
  **[CANN社区版文档/环境变量参考](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/maintenref/envvar/envref_07_0001.html)**。

## <a id="develop"></a>5.  自定义算子开发

详细步骤可参考[从开发一个简单算子出发](./docs/developing_a_simple_operator.md)

## <a id="contribute"></a>6. 参与贡献

1. fork仓库
2. 修改并提交代码
3. 新建Pull-Request

详细步骤可参考[贡献指南](./docs/contributing_guide.md)

## <a id="doc"></a>7.  参考文档

**[CANN社区版文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/index/index.html)**
**[SiP社区版文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/API/SiP/SIP_0000.html)**
