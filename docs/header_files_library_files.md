# 头文件和库文件说明

本文档介绍 Ascend SiP Boost（SiP）库的头文件与库文件，帮助用户在编译和运行应用程序时正确引用依赖。

## 接口分类

AsdSip 接口统一位于 `AsdSip` 命名空间下，按功能分为以下几大类别：

| 接口名前缀 | 所属模块 | 描述 |
| --- | --- | --- |
| swapLast2Axes / asdMul | Base（基础模块） | 基础张量操作，包括轴交换、逐元素乘法等。 |
| asdFft* | FFT（快速傅里叶变换模块） | FFT/STFT/ISTFT 变换，支持 1D/2D/3D，支持 C2C、C2R、R2C 类型。 |
| asdBlas* | BLAS（线性代数模块） | BLAS Level 1/2/3 操作，包括矩阵乘法、向量运算、三角求解等，支持实数和复数类型。 |
| asdConvolve* | Filter（滤波器模块） | 一维卷积操作，支持 full/same/valid 三种模式。 |
| asdInterp* | Interpolation（插值模块） | 基于系数的插值操作。 |
| rs* | Domain（领域模块） | 面向雷达信号处理等特定领域的专用接口，如 Sinc 插值。 |

## 头文件说明

编译 AsdSip 接口程序时，需根据实际使用的接口 include 对应的头文件。头文件位于安装目录的 `include/` 下。

| 头文件 | 用途 | 对应的库文件 |
| --- | --- | --- |
| asdsip.h | **总入口头文件**，聚合所有公开 API 头文件。用户只需 `#include "asdsip.h"` 即可使用全部接口。内部依次 include base_api.h、blas_api.h、fft_api.h、filter_api.h、interp_api.h 和 domain/rs_api.h。 | libasdsip.so 或 libasdsip_static.a |
| base_api.h | **基础操作接口**。定义轴交换（swapLast2Axes）、逐元素乘法（asdMul）等基础张量操作。依赖 utils/aspb_status.h 和 utils/mem_base.h。 | libasdsip.so 或 libasdsip_static.a |
| fft_api.h | **FFT 快速傅里叶变换接口**。定义 FFT Handle 管理（asdFftCreate/Destroy）、流绑定（asdFftSetStream）、1D/2D/3D 计划创建（asdFftMakePlan1D/2D/3D）、ISTFT 计划创建（asdFftIstftMakePlan）、执行接口（asdFftExecC2C/C2R/R2C/C2CSeparated/Istft）、Workspace 管理（asdFftGetWorkspaceSize/SetWorkspace）、同步（asdFftSynchronize）等。同时定义 asdFftType、asdFftDirection、asdFft1dDimType 等枚举类型。 | libasdsip.so 或 libasdsip_static.a |
| blas_api.h | **BLAS 线性代数接口**。定义 BLAS Handle 管理（asdBlasCreate/Destroy）、流绑定（asdBlasSetStream）、计划创建（MakeXxxPlan 系列）、Workspace 管理、同步等。提供的 BLAS 操作包括：矩阵乘法（Cgemm、CgemmBatched、HCgemmBatched）、矩阵向量乘（Cgemv、CgemvBatched、HCgemvBatched）、三角矩阵求解（Strmv、Ctrmv）、矩阵三角乘（Strmm）、秩一更新（Cgerc）、向量点积（Sdot、Cdotu、Cdotc）、向量范数（Snrm2、Scnrm2）、向量绝对值求和（Sasum、Scasum）、向量缩放（Sscal、Cscal、Csscal）、向量拷贝（Scopy、Ccopy）、向量交换（Sswap、Cswap）、向量线性组合（Caxpy）、Givens 旋转（Csrot）、矩阵求逆（CmatinvBatched、HCmatinvBatched）、列向量乘（ColwiseMul）、复矩阵点积（ComplexMatDot）、最大绝对值索引（Isamax、Icamax）等。同时定义 asdBlasStatus、asdBlasSideMode_t、asdBlasOperation_t、asdBlasFillMode_t、asdBlasDiagType_t 等枚举类型。依赖 acl/acl.h 和 aclnn/opdev/fp16_t.h。 | libasdsip.so 或 libasdsip_static.a |
| filter_api.h | **滤波器/卷积接口**。定义一维卷积操作（asdConvolve）及其 Workspace 大小查询（asdConvolveGetWorkspaceSize）。支持 ASD_CONVOLVE_FULL、ASD_CONVOLVE_SAME、ASD_CONVOLVE_VALID 三种卷积模式（asdConvolveMode_t 枚举）。 | libasdsip.so 或 libasdsip_static.a |
| interp_api.h | **插值接口**。定义基于系数的插值操作（asdInterpWithCoeff）及其 Workspace 大小查询（asdInterpWithCoeffGetWorkspaceSize）。 | libasdsip.so 或 libasdsip_static.a |
| domain/rs_api.h | **雷达信号处理领域接口**。定义 Sinc 插值操作（rsInterpolationBySinc）及其 Workspace 大小查询（rsInterpolationBySincGetWorkspaceSize），面向雷达信号处理场景。 | libasdsip.so 或 libasdsip_static.a |

## 库文件说明

构建完成并执行安装后，库文件位于安装目录的 `lib/` 下。

### AsdSip 库文件

| 库文件 | 类型 | 说明 |
| --- | --- | --- |
| libasdsip.so | 动态库（共享库） | **主用户库**。聚合 utils、base、blas、fft、filter、interpolation 全部模块，链接 libasdsip_core.so。用户应用程序应链接此库以使用 AsdSip 全部公开接口。 |
| libasdsip_static.a | 静态库 | **主用户库（静态版本）**。功能与 libasdsip.so 相同，适用于需要静态链接的场景。链接时需额外链接 libasdsip_core.so。 |
| libasdsip_core.so | 动态库（共享库） | **算子核心运行时库**。包含算子注册、Kernel 加载与调度（Ops 单例）、tiling 逻辑等。内部链接 MKI 静态库（libmki_static.a）和 Ascend CANN 算子编译框架库。由 libasdsip.so 自动依赖，用户通常无需单独引用。 |
| libasdsip_host.so | 动态库（共享库） | **主机端工具库**。包含算子参数处理等主机端辅助功能，依赖 ops_utils 模块。 |

### 第三方依赖库

以下库文件由 CANN 软件包或 MKI 框架提供，编译和运行时需确保其在链接路径中可用。

| 库文件 | 来源 | 说明 |
| --- | --- | --- |
| libmki.so / libmki_static.a | MKI 框架（3rdparty/mki） | **MKI 内核抽象框架库**。提供 Tensor、Kernel、Operation 等核心抽象，以及运行时调度能力。libasdsip_core.so 在发布模式下静态链接 libmki_static.a，在测试模式下动态链接 libmki.so。 |
| libascendcl.so | CANN 软件包（$ASCEND_HOME_PATH/lib64） | **Ascend CL 运行时库**。提供 aclTensor 等基础数据类型和设备管理、内存管理、Stream 管理等运行时接口。AsdSip 公开头文件中引用了 acl/acl.h。 |
| libaclnn.so | CANN 软件包（$ASCEND_HOME_PATH/lib64） | **Ascend NN 算子库**。提供 aclnn 系列算子接口，BLAS 模块中部分操作依赖 aclnn 头文件（aclnn/opdev/fp16_t.h、acl/acl_meta.h 等）。 |
