# 信号处理加速库CtrmvOperation C++ Demo

## 介绍

该目录下为信号处理加速库CtrmvOperation C++调用示例。

## 功能说明

- 算子功能：单精度复数矩阵向量乘法算子，用于计算一个复数三角矩阵与一个复数向量的乘积。
- 计算公式：
    $$\mathbf{x} = op(A) \cdot \mathbf{x}$$
    其中 $op(A)$ 取决于 trans 参数：
    - $op(A) = A$     (NoTrans)
    - $op(A) = A^T$   (Trans)  
    - $op(A) = A^H$   (ConjTrans)

## 使用说明

### 环境配置

 - 配置CANN环境变量

    ```sh
    source [CANN安装路径]/set_env.sh
    ```  

    默认：source /usr/local/Ascend/ascend-toolkit/set_env.sh

### SiP编译

 - 用户应进入SiP根目录，执行如下指令进行信号处理加速库的编译，并设置加速库环境变量。

    ```sh
    cd ${SiP_root_path}
    bash build.sh
    source output/set_env.sh
    ```

    特别说明：
    - 上述编译方式仅支持编译通过git下载的加速库，以zip压缩包方式下载的加速库不支持该编译方式；
    - 由于编译过程需要联网下载依赖库，因此编译环境需要联网；
    - 该编译过程包括获取ascend-boost-comm（昇腾分布式通信加速库）组件并编译该组件，和编译信号加速库两个步骤。更多命令介绍可查看SiP仓库的[build.sh](../../../../build.sh)文件。

 - 更多编译命令说明请参考[编译与构建](../../../../docs/compilation_build.md)

### 运行demo

- 进入 `example` 目录并执行构建脚本。

    ```sh
    cd  ${示例所在目录}
    bash build.sh
    ```

## 额外说明

示例中生成的数据不代表实际场景，可根据具体使用场景进行数据修改。

## 产品支持情况

适用于Atlas A2/A3 训练系列产品、Atlas 800I A2 推理产品、Atlas A3 推理系列产品。
