# fft-2d-r2c-4096x4096 — FFT R2C 2D [4096,4096] 融合单 kernel 算子

从 `examples/85_ascend950_fft_r2c_2d` 重构而来的**独立算子工程**（自包含，vendored CATLASS/TLA）。

## 1. 算子语义

- 输入：`[1, 4096, 4096]` float32（实数）
- 输出：`[1, 4096, 2049]` complex64（半谱，交织 `[re,im]`）
- forward only，等价 `torch.fft.rfft2(x, norm="backward")`
- 精度 MARE ≈ 4.7e-7

## 2. 方案

单个 `__mix__(1,2)` kernel 内 5 个 stage，阶段间 `SyncAll<false>`：

```
S1 R2C-1D: x -> y[4096,2112]   两实行打包 + 16x16x16 C2C + 共轭拆分
S2 转置:   y -> z[2112,4096]   64x64 复 tile
S3 C2C-1D: z -> w[2112,4096]   16x16x16 C2C
S4 转置:   w -> w2[4096,2112]  64x64 复 tile
S5 裁剪:   w2 -> out[4096,2049] DataCopyPad<Compact>
```

## 3. 构建与运行

```bash
bash build.sh
bash run_test.sh <device_id>   # 精度（MARE < 2e-5）
bash run_perf.sh <device_id>   # 性能
```

## 4. 基线

| 项 | 值 |
|---|---|
| 精度 | MARE ≈ 4.74e-7 |
| 性能 | ~0.90 ms |

## 5. 链接依赖

需链接 `libascend_hal.so`（全路径 `$ASCEND_HOME_PATH/x86_64-linux/devlib/libascend_hal.so`）。
S5 的 `DataCopyPad<Compact>` 非对齐裁剪依赖该 HAL，缺失时精度错误（MARE ~1.4e-1）。