# fft-2d-c2c-4096x4096 — FFT C2C 2D [4096,4096] 融合单 kernel 算子

> 从 `examples/101_ascend950_fft_c2c_2d_4096_axis0` 重构而来的**独立算子工程**，
> 仅保留 FFT2D（both axes）分支，不依赖仓库外部文件（CATLASS/TLA 已 vendored）。

## 1. 算子语义

输入 `[4096, 4096]` complex64，做 2D C2C FFT（axes 0,1）：

- forward（`direction=-1`）等价 `np.fft.fft2(x)`。
- inverse（`direction=+1`）等价 `np.fft.ifft2(x, norm="forward")`（不归一化）。

## 2. 方案

单个 `__mix__(1,2)` kernel 内完成四阶段，利用两次转置吸收列 FFT：

```
S1: 行 FFT (ex99 引擎)   X  -> h   = fft(X, axis=1)
S2: 转置 (64x64 SIMT)     h  -> ws  = h.T
S3: 行 FFT (ex99 引擎)    ws -> t   = fft(h.T, axis=1) = fft(h, axis=0).T
S4: 转置 (64x64 SIMT)     t  -> out = fft(h, axis=0) = fft2(X)
```

AIC 运行行 FFT 的 16×16×16 GEMM；AIV 运行 deinterleave / twiddle / radix-2 与
64×64 SIMT 转置。阶段间 `SyncAll<false>`。

## 3. 目录结构

| 文件 | 职责 |
|---|---|
| `fft_c2c_2d.cpp` | host 入口（参数、bin I/O、`--perf` 计时） |
| `fft_plan.cpp/.h` | Plan/Exec/Destroy（系数构造、FFT2D 融合单 launch） |
| `fft_tiling.cpp/.h` / `fft_tiling_def.h` | tiling 参数 |
| `fft_twiddle.h` / `fft_types.h` | 系数构造 / 常量 |
| `fft_kernel.h` | 融合 kernel 声明 |
| `fft_kernel_fft.cpp` | 行 FFT 引擎 + 融合四阶段 kernel |
| `fft_kernel_transpose_stage.h` | 转置 stage（inline，融合复用） |
| `fft_kernel_utils.h` / `fft_simt_ops.h` / `fft_radix2_ops.h` | GEMM 块 / SIMT 算子 |
| `include/catlass` / `include/tla` | vendored 模板库 |
| `helper.hpp` | ACL 辅助宏 |

## 4. 构建与运行

```bash
# 构建
bash build.sh

# 精度（双向，门禁 MARE < 2e-5）
bash run_test.sh <device_id>

# 性能（10 warmup + 10 measured，双向）
bash run_perf.sh <device_id>

# 手动
./build/fft_2d_c2c_4096x4096 [direction] [device_id] [data_dir] [--perf]
```

## 5. 基线结果

| 项 | 值 |
|---|---|
| forward (direction=-1) | ~1.59 ms |
| inverse (direction=+1) | ~1.58 ms |
| 精度 | MARE < 2e-5（实测 ~3.6e-7） |

## 6. 链接依赖

需链接 `libascend_hal.so`（全路径 `$ASCEND_HOME_PATH/x86_64-linux/devlib/libascend_hal.so`）。
该 HAL 库为 Ascend 统一链接项，缺失时部分非对齐 data 搬运（如 `DataCopyPad<Compact>`）会静默出错。