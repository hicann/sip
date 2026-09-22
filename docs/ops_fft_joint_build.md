# ops-fft 联合构建

## 概述

SiP 的 FFT 算子支持两种执行后端：

- **internal（默认）**：使用 SiP 自带 kernel；
- **ops-fft**：直接调用 ops-fft 预编译库 `libcann_ops_fft.so` 导出的 device kernel，常量与 tiling 复用 SiP plan 期产物，两种后端计算结果逐位一致。

联合构建指该能力的全链路：构建期以 `--use_ops_fft` 显式开启，将 ops-fft 二进制存件与 SiP 一起链接并打进安装包（默认构建为纯 internal，不依赖存件）；运行期通过环境变量强制启用 ops-fft 后端（未设置时使用 internal）。

## 使用范围

| SoC | 覆盖路径 | 直调 kernel |
| --- | --- | --- |
| Ascend910B | `asdFftExecV2` 的 FFTCoreB 执行路径 | `fft_b` |
| Ascend950 | FFT C2C 混合基（mixed-radix）路径 | `fft_c2c_arch35_mix_multi_core` |

未覆盖的算子与路径不接入该后端：未设置环境变量时始终走 internal；请求 ops-fft 时直接报错中止（统一拦截）。

## 构建说明

### 默认构建（纯 internal）

`bash build.sh` 默认**不构建** ops-fft 联合构建，产出纯 internal 的标准包；无需关心 `3rdparty/ops-fft/` 存件是否存在。

### 开启联合构建

```sh
bash build.sh --use_ops_fft
```

开启后存件为强依赖：`build.sh` 会按 `lib/<soc>/manifest.info`（每个 SoC 一份，多 SoC 存件并存互不覆盖）逐项校验存件，**SoC / 存件架构**为硬门禁（存件架构由 readelf 直读 `libcann_ops_fft.so` 并与本机构建机比对，当前 910B 存件为 aarch64、950 存件为 x86_64，须在架构匹配的机器上构建）；`cann_version` 与 `abi` 仅作信息性记录不参与校验（`libcann_ops_fft.so` 链接 CANN 运行时，ABI 向后兼容，且对外为 `extern "C"` 纯 C 接口，`_GLIBCXX_USE_CXX11_ABI` 不影响二进制兼容）。存件缺失/失配或编译目标不含 910b/950 时直接报错终止。校验通过时 `libasdsip.so` 链接 `libcann_ops_fft.so`，并将其与头文件收编进安装包（已登记 `filelist.csv`，卸载不残留）；安装布局下 `libcann_ops_fft.so` 与 `libasdsip.so` 同目录，由 `set_env.sh` 的 `LD_LIBRARY_PATH` 定位。

### 目标 SoC 自动推导

ops-fft 存件的 SoC 选择无需手动指定，由构建系统自动推导：

- `configs/build_config.json`（或 `BUILD_CONFIG_FILE` 指定的配置）仅启用 910b/950 之一时，跟随该目标；
- 两者同时启用时，按构建机架构选取：aarch64 链接 910B 存件、x86_64 链接 950 存件（存件架构须与构建机匹配，见上文校验说明）。

即：在匹配架构的机器上 `bash build.sh --use_ops_fft`，即可得到对应 SoC 的联合构建。

### 存件缺失或失配时

`--use_ops_fft` 下存件为强依赖：存件缺失或校验失配（含架构不匹配）时直接报错终止构建，需更新存件（见文末"存件更新"）或换用架构匹配的构建机。

## 运行说明

通过进程级环境变量切换后端：

```sh
export SIP_FFT_BACKEND=ops-fft   # 强制启用 ops-fft 后端（未适配/缺件时报错中止）
unset SIP_FFT_BACKEND            # 恢复 internal 后端（默认）
```

- 后端为强制请求：当前路径未适配（路由条件表之外）或 `libcann_ops_fft.so` 缺失/不包含所需 kernel 时，直接返回 `ACL_ERROR_API_NOT_SUPPORT` 并输出 ERROR 日志（默认日志级别即可见，需已开启日志输出：`ASCEND_SLOG_PRINT_TO_STDOUT=1` 或 `ASCEND_PROCESS_LOG_PATH`），不回退、不执行计算；
- 后端切换只影响 kernel 选择，接口与计算结果不变；
- 成功走 ops-fft 后端时日志包含 `run via ops-fft kernel ... success`，可据此确认后端生效。

### 路由条件

后端开关只对下表范围内的 C2C 1D 计算生效，范围外（如 n ≤ 256 的 DFT、910B 混合基、950 纯 2 幂）不接入该后端——未设置环境变量时走 internal，请求 ops-fft 时报错中止：

| SoC | 命中条件 | 示例 Nfft | 直调 kernel |
| --- | --- | --- | --- |
| Ascend910B | n 为纯 2 幂且 256 < n < 32768（batch 任意） | 8192 | `fft_b` |
| Ascend950 | n 非纯 2 幂，且质因子均在 {2, 3, 5, 7, 11, 13, 17, 19} 内 | 15000 | `fft_c2c_arch35_mix_multi_core` |

### 快速验证

仓库提供独立样例 `example/A2/FFT/ops_fft_ab/`（不依赖、不修改任何现有样例），一条命令完成 internal 与 ops-fft 双后端 A/B 对比（自动编译 + 执行 + 逐位比对 + 日志命中确认）：

1. 配置环境（安装包布局）：

```sh
source <CANN安装路径>/set_env.sh
export ASDSIP_HOME_PATH=<SiP安装路径>
```

2. 运行（参数为 n 与 batch，默认 8192 4）：

```sh
cd example/A2/FFT/ops_fft_ab
bash build.sh            # Ascend910B: n=8192 命中 fft_b 路径
bash build.sh 15000 4    # Ascend950: 混合基 n=15000 命中 arch35 mix 路径
```

3. 结果判读：

```text
[AB] 双后端输出逐位一致 (internal 0.055 ms/exec, ops-fft 0.050 ms/exec)
[2026-09-16 16:25:47.812341] [asdsip] [INFO] ... FFTCoreB run via ops-fft kernel (blocks=4) success.
```

第一行说明双后端计算结果一致——能执行到对比阶段即表示 ops-fft 已生效；第二行是直调命中的日志证据（Ascend950 上对应 `FftC2CCoreArch35 run via ops-fft kernel (arch35 mix, ...)`）。样例已自动开启 `ASCEND_GLOBAL_LOG_LEVEL=1` 与 `ASCEND_SLOG_PRINT_TO_STDOUT=1`（默认日志级别为 ERROR，看不到 INFO 级后端日志）；若形状未命中覆盖路径或安装包为 internal-only 构建，样例会在 ops-fft 阶段直接报错中止（严格语义）。

单元测试默认在 internal 后端下运行；ops-fft 后端请使用上述 A/B 样例验证（UT 用例包含大量未适配形状，整体设置 `SIP_FFT_BACKEND=ops-fft` 会因严格语义报错）。

## 存件更新（维护者）

存件为 ops-fft 侧产出的成品，本仓库只收编、不从源码构建。更新存件（升级 ops-fft / 新增 SoC）时，将对应产物按下述布局放入并提交：

```text
3rdparty/ops-fft/
├── include/cann_ops_fft.h            # ops-fft 对外头文件
└── lib/<soc>/                        # 如 Ascend910B、Ascend950
    ├── libcann_ops_fft.so            # soname 链: .so -> .so.1 -> .so.x.y.z 三件套
    └── manifest.info                 # 见下
```

`manifest.info` 字段（`build.sh` 按 `soc` 校验，其余为溯源信息）：

```text
soc=<Ascend910B|Ascend950>            # 须与所在目录名一致
abi=<ON|OFF>                          # 信息性: 接口为 extern "C", 不影响兼容
cann_version=<x.y.z>                  # 信息性: 制作环境的 CANN 版本
cann_path=<路径>                      # 信息性: 制作环境的 CANN 路径
source_head=<commit>                  # 信息性: ops-fft 源码 commit
so_md5=<md5>                          # md5sum libcann_ops_fft.so 的结果
```
