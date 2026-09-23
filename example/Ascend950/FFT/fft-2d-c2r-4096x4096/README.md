# fft-2d-c2r-4096x4096 — FFT C2R 2D [4096,4096] 算子

从 `examples/86_ascend950_fft_c2r_2d` 重构而来的**独立算子工程**（自包含，vendored CATLASS/TLA）。

## 1. 算子语义

- 输入：`[batch, 4096, 2049]` complex64（半频谱，交织 `[re,im]`）
- 输出：`[batch, 4096, 4096]` float32（实数）
- inverse only，等价 `torch.fft.irfft2(X, s=(4096,4096), norm="forward")`（无归一化）
- 精度 MARE ≈ 3.47e-6

## 2. 方案

两阶段流水线（K2 融合进 K1），radix-64 1D IDFT 分解：

```
input [batch, N, N/2+1] complex (半谱)
  -> K1: Hermitian expand + Row IDFT + fused transpose -> ws1 (transposed complex)
  -> K3: Row IDFT + Real extraction -> output [batch, N, N] real
```

## 3. 构建与运行

```bash
bash build.sh
bash run_test.sh <device_id>   # 精度（MARE < 2e-5）
bash run_perf.sh <device_id>   # 性能（--benchmark --perf）
```

## 4. 基线

| 项 | 值 |
|---|---|
| 精度 | MARE ≈ 3.40e-6 |
| 性能 | ~0.83 ms |

## 5. 链接依赖

需链接 `libascend_hal.so`（全路径 `$ASCEND_HOME_PATH/x86_64-linux/devlib/libascend_hal.so`）。