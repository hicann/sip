# 信号处理加速库FFT2DOperation C++ Demo

## 介绍

该目录下为信号处理加速库FFT2DOperation C++调用示例。

## 功能说明

- 算子功能：执行批量2D快速傅里叶变换。
- 计算公式：
    $$
    X[k,l] = \sum_{m=0}^{M-1} \sum_{n=0}^{N-1} x[m,n] \cdot e^{-j2\pi (km/M + ln/N)}
    $$

    其中：
    - $x[m,n]$ 为输入二维时域信号（复数矩阵）
    - $X[k,l]$ 为输出二维频域信号（复数矩阵）
    - $M \times N$ 为信号尺寸
    - $k = 0, 1, \dots, M-1$, $l = 0, 1, \dots, N-1$

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

## 场景说明

提供示例代码分别对应不同场景，编译运行时需要根据具体场景对应更改build脚本：

- **example_acl_fft_c2c_2d.cpp**

    【注】：默认编译脚本可编译运行该示例。

    **输入**

    | TensorName |   DataType  | DataFormat |          Shape          |       Value     |
    | :--------: | :---------: | :--------: | :---------------------: | :-------------: |
    |  fftType   |  asdFftType |     nd     |           [1]           |  ASCEND_FFT_C2C |
    |   input    |  complex64  |     nd     |  [batch, Nfft1, Nfft2]  |       -         |

    **输出**

    | TensorName |  DataType  | DataFormat |           Shape           |       Value     |
    | :--------: | :--------: | :--------: | :-----------------------: | :-------------: |
    |     out    |  complex64 |     nd     |   [batch, Nfft1, Nfft2]   |         -       |

    ---

- **example_acl_fft_c2r_2d.cpp**

    【注】：将编译脚本中的 example_acl_fft_c2c_2d.cpp 替换为 example_acl_fft_c2r_2d.cpp 后，替换后的编译脚本可编译运行。

    **输入**

    | TensorName |   DataType  | DataFormat |              Shape              |       Value     |
    | :--------: | :---------: | :--------: | :-----------------------------: | :-------------: |
    |  fftType   |  asdFftType |     nd     |                [1]              |  ASCEND_FFT_C2R |
    |   input    |  complex64  |     nd     |  [batch, Nfft1, Nfft2 / 2 + 1]  |       -         |

    **输出**

    | TensorName |  DataType  | DataFormat |           Shape           |       Value     |
    | :--------: | :--------: | :--------: | :-----------------------: | :-------------: |
    |     out    |   float32   |     nd     |   [batch, Nfft1, Nfft2]   |         -       |

     ---

- **example_acl_fft_r2c_2d.cpp**

    【注】：将编译脚本中的 example_acl_fft_c2c_2d.cpp 替换为 example_acl_fft_r2c_2d.cpp 后，替换后的编译脚本可编译运行。

    **输入**

    | TensorName |   DataType   | DataFormat |           Shape           |       Value     |
    | :--------: | :----------: | :--------: | :-----------------------: | :-------------: |
    |   fftType  |   asdFftType |     nd     |             [1]           |  ASCEND_FFT_R2C |
    |    input   |    float32    |     nd     |   [batch, Nfft1, Nfft2]   |         -       |

    **输出**

    | TensorName |   DataType  | DataFormat |              Shape              |       Value     |
    | :--------: | :---------: | :--------: | :-----------------------------: | :-------------: |
    |    out     |  complex64  |     nd     |  [batch, Nfft1, Nfft2 / 2 + 1]  |       -         |
