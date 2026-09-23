# fft-2d-c2c-500x50

独立编译执行的 2D C2C FFT 单算子工程。算子对 `[12000, 500, 50]` 复数（`complex64`，交错
`[re, im]`）在 `axes=(1,2)` 上做二维 FFT，采用 **AIC GEMM + AIV SIMT 融合**实现。

| 项 | 值 |
|---|---|
| 输入形状 | `[b1=12000, fftN1=500, fftN2=50]` complex64 |
| 算子 | 2D C2C FFT（axes 1,2），forward / inverse 均未归一化 |
| 架构 | Ascend950（`dav-3510`），`__mix__(1,2)`（1 AIC + 2 AIV per block） |
| 实测耗时 | ≈ 5.86 ms（Task Duration） |
| 实测精度 | MARE ≈ 4e-5（门限 1e-4） |

## 目录结构

```
fft-2d-c2c-500x50/
├── CMakeLists.txt            # 独立 cmake（find_package(ASC)）
├── build.sh                  # 一键编译
├── run.sh                    # 一键编译 + 执行 + 精度比较
├── run_perf.sh               # 性能测试
├── gen_data.py               # 数据生成 + golden + 精度比对
├── include/
│   ├── catlass/              # vendored Catlass 模板库（header-only）
│   └── tla/                  # vendored TLA 库
├── fft_types.h               # 常量 / 数位反转 / tiling 宏
├── fft_tiling_def.h          # 设备侧 tiling POD
├── fft_tiling.h/.cpp         # host tiling 计算
├── fft_coeffs.h/.cpp         # 系数构建（twiddles + W50）
├── fft_plan.h/.cpp           # plan 创建/执行/销毁
├── fft_kernel.h/.cpp         # mix 核 + launch wrapper
├── fft_kernel_utils.h        # AIC GEMM（RunGemmToUb）+ 跨核旗标
├── fft_simt_ops.h            # AIV 2 阶段 radix-20/25 列 FFT（SIMT）
├── fft_c2c_2d.cpp            # host 入口（main）
├── helper.hpp                # ACL_CHECK 等 host 辅助（vendor 自 examples/common）
└── options.hpp               # Gemm/Gemv 选项结构（helper.hpp 依赖，vendor 同上）
```

## 实现概述

- **AIC（Cube）**：50 点行 FFT 用稠密 DFT 矩阵 `W50[100,104]`（8-float 对齐垫宽）做
  两次 M-half GEMM，`NO_SPLIT_SUBBLOCK` Fixpipe 把 `[500,104]` tile 直写目标 AIV 的 UB。
- **AIV（SIMT）**：500 点列 FFT 用 2 阶段寄存器融合 `radix-20（4×5）+ radix-25（5×5）`，
  复用原 twiddle 表与数位反转，UB 流量从 4 读 3 写降到 2 读 1 写。
- 矩阵级 pingpong（AIC/AIV 跨核旗标，mode-4），无 GM workspace、无 SyncAll。

## 依赖

- `include/catlass` 与 `include/tla`：已随目录 vendor，无需额外安装。
- CANN 开发套件：构建前需 `source <cann>/ascend-toolkit/set_env.sh`（`build.sh` 会尝试自动
  定位 `${HOME}/cann/ascend-toolkit/set_env.sh`）。

## 编译

```bash
bash build.sh
```

产物：`build/fft_2d_c2c_500x50`。

## 运行与精度

```bash
bash run.sh [device_id]        # 默认 device_id=0
```

`run.sh` 依次：编译 → 生成 input/golden（`numpy.fft.fft2`，seed 42）→ 跑 forward/inverse →
按 `MARE < 1e-4` 门限比对精度。

## Round-trip 模式

`--roundtrip` 实现多通道 round-trip 工作负载：数据形状 `[RO=500][CH=24][PE=500][SPE=50]`
（即 `[12000, 500, 50]` complex64），在 `CH` 个校准通道上对每个 `[500, 50]` 平面重复执行
`fftshift → forward → inverse → ifftshift(+1/N)` 完整链路，并统计整体耗时与吞吐。

```bash
./build/fft_2d_c2c_500x50 500 -1 <device_id> --roundtrip
```

- 内层循环 `CH = 24` 次，每次对完整 `[12000, 500, 50]` 执行 fftshift、一次正向 + 一次
  逆向 2D FFT、以及 ifftshift（含 `1/N` 缩放，`N = PE×SPE = 500×50 = 25000`）；
- 由于 inverse 后含 `1/N` 归一化，round-trip 输出恢复原始输入（不是 `输入 × N`）；
- 输出统计：`Time / Total FFTs / FFTs/sec / Max diff / Test`
  - `Total FFTs = RO×CH²×2 = 500×24²×2 = 576000`
  - `Max diff` = round-trip 输出与原始输入的最大逐点差，绝对门限 `< 1e-5` 判 PASS（无归一化步骤）；
- 该模式在 host 侧用固定种子（seed 42）生成随机复数输入，精度校验直接对比原始输入；
- 进程退出码：PASSED 为 0，FAILED 为非 0，供脚本/CI 直接判定（issue #181）。

实测结果（Ascend950PR，28 AIC / 56 AIV）：

```text
Time: 0.306828 s
Total FFTs: 576000 (= RO×CH²×2 = 500×24²×2)
FFTs/sec: 1877274
Max diff: 0.000001
Test: PASSED
```

`Max diff ≈ 1e-6`，低于 `1e-5` 绝对门限。

## 性能

```bash
bash run_perf.sh [device_id]   # aclrtEvent：10 预热 + 10 测量
```

复现 msprof `Task Duration`（≈5.86ms）：

```bash
source <cann>/ascend-toolkit/set_env.sh
msprof op build/fft_2d_c2c_500x50 500 -1 <device_id> data
```
