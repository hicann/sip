# torch_sip.solve 批量线性方程组求解 —— 接入指导

面向 PyTorch 用户的 `torch_sip.solve` / `torch_sip.solve_backward` 使用说明。底层纯 C++ 接口（`AsdSip::solve`）参见 [solve.md](solve.md)。

- **环境**: Atlas A2 系列（Ascend910B3）/ CANN 9.2.0 / torch 2.x + torch_npu / AscendSiPBoost 9.1.0
- **源码位置**: [sip_pta/](../../../../sip_pta/)（PyTorch 扩展）、[ops/base/solve/](../../../../ops/base/solve/)（device kernel）、[core/base/solve.cpp](../../../../core/base/solve.cpp)（C 接口层）

---

## 1. 安装与初始化

```bash
# 前置: CANN 环境已 source, 当前 python 环境已安装 torch + torch_npu
cd sip_pta
bash build.sh --install        # 构建 torch_sip whl 并覆盖安装到当前环境
```

验证安装：

```python
import torch, torch_npu, torch_sip
a = torch.tensor([[[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]]], device="npu:0")
b = torch.tensor([[[7.0, 12.0], [15.0, 20.0], [13.0, 16.0]]], device="npu:0")
x = torch_sip.solve(a, b)
print(x.cpu())                 # tensor([[[1., 2.], [3., 4.], [5., 6.]]])
```

## 2. 接口签名

| 接口 | 语义 |
|---|---|
| `torch_sip.solve(A, B)` | 批量求解 A·X = B（fp32），返回 X，shape 为 [batch, n, nrhs]（batch 跟随 A） |
| `torch_sip.solve_backward(A, X, gX)` | 手动解析反向，返回 `(gB, gA)`：`gB = solve(Aᵀ, gX)`，`gA = -(gB·Xᵀ)`；B 为广播形态时 gB 满批量返回，需自行沿 batch 求和 |

### 输入约束

| 项 | 约束 |
|---|---|
| shape | `A [batch, n, n]`、`B [batch|1, n, nrhs]`，输出 `X [batch, n, nrhs]`（batch 跟随 A）；均为 3 维 |
| dtype | `float32`（仅此一种） |
| n | `[2, 256]` |
| nrhs | ≥ 1 |
| 内存布局 | 连续张量最优（非连续自动物化，有拷贝开销；直接调用 C API 时需调用方保证连续） |
| batch 广播 | 支持 B `[1, n, nrhs]` 广播形态：kernel 原生处理，无需调用方展开，各矩阵共享 B[0]，X 的 batch 为 A 的 batch；其余情况 B 的 batch 须与 A 一致 |
| 奇异性 | A 须非奇异；算子不做奇异检测，奇异/近奇异矩阵结果不可用 |

### 执行路径（对用户透明）

| 条件 | 路径 |
|---|---|
| n==2 且右端项列数较小 | 跨矩阵向量化路径（64 个矩阵打包 64 车道） |
| 2 < n ≤ 64 | 整装 LU（A/L 常驻 UB） |
| 64 < n ≤ 256 | 分块 LU（A 副本驻留显存，按 64 列 panel 分解） |

## 3. 前向用法

```python
import torch, torch_npu, torch_sip

batch, n, nrhs = 196, 3, 16384
a = (torch.randn(batch, n, n) + torch.eye(n) * 3).npu()   # 对角增强保证可逆
b = torch.randn(batch, n, nrhs).npu()

x = torch_sip.solve(a, b)                                  # [196, 3, 16384]
```

与 `torch.linalg.solve` 语义一致（同样要求 A 可逆），B 的 batch 可与 A 相同或为 1（广播形态，kernel 原生共享 B[0]，见上方"batch 广播"约束）；区别是本接口固定计算在 NPU 上且仅支持 fp32；不支持 `out=` 参数。

## 4. 反向用法

反向有**自动微分**与**手动两阶段**两种用法，二者调用同一反向实现：

### 4.1 自动微分（推荐，与 torch 用法一致）

`torch_sip.solve` 已通过 `torch.autograd.Function` 注册反向：A/B 任一 `requires_grad` 时自动走 `solve_backward` 计算梯度。

```python
a = a.requires_grad_(True)
b = b.requires_grad_(True)
x = torch_sip.solve(a, b)
x.backward(torch.randn_like(x))               # 上游梯度 dL/dX
# a.grad = gA [batch, n, n]
# b.grad = gB [1, n, nrhs]（B 为广播形态 [1, n, nrhs] 时沿 batch 归约，与实现 ctx.b_bcast 门控一致；
#          B 与 A 同 batch 时为 [batch, n, nrhs]）
```

### 4.2 手动两阶段（推理/训练混合场景，省一次前向记录）

```python
with torch.no_grad():
    x = torch_sip.solve(a, b)          # 前向（保存 X 供反向使用）

with torch.no_grad():
    gb, ga = torch_sip.solve_backward(a, x, gx)   # gB 同 X 形状, gA [batch, n, n]
```

公式：

```text
gB = solve(Aᵀ, gX)          # 复用融合 solve 求解 Aᵀ·gB = gX
gA = -(gB @ Xᵀ)
```

其中 `gA` 的计算按矩阵阶数分派：n≤6 且右端项列数较大时走广播乘+末维归约（向量路径），其余走 ATen bmm（Cube 路径），对用户透明。

## 5. 从 torch.linalg.solve 迁移

```python
# 迁移前
x = torch.linalg.solve(a, b)

# 迁移后（全 NPU，语义一致）
x = torch_sip.solve(a, b)
```

差异说明：
- 仅支持 fp32；`float64`/`float16`/`bfloat16` 会报错。
- 仅支持 3 维批量形式 `[batch, n, n]`，不支持 `torch.linalg.solve` 的广播组合与批量 `[..., n, nrhs]` 任意前导维。
- 奇异矩阵：`torch.linalg.solve` 在 CPU 上抛 `SingularMatrixError`，本接口不检测，结果为 inf/nan，业务侧须保证输入非奇异。

## 6. 精度与性能参考

（数据为 Atlas 800T A2 单卡空闲时段实测。精度标杆为 CPU fp32 `torch.linalg.solve`；性能基线为“小算子拼接”，两者均与仓内 UT/固化套件口径一致。）

**精度**（固化套件 30/30 PASS；C++ UT 150/150；sip_pta solve 14/14。判据 MERE < 1e-4，n>64 放宽至 1e-3）：

| 场景 | MERE 典型值 |
|---|---|
| 需求用例 P0/P1/P2（n=2~4） | 5.0e-8 ~ 2.8e-7 |
| 泛化 n=32/48/64 | 6.2e-6 ~ 1.7e-5 |
| 泛化 n=128/256（分块 LU） | 5.0e-5 ~ 2.1e-4 |
| 反向 gB / gA（n=3~64） | 6.4e-7 ~ 2.5e-3 |

**性能**（设备事件摊销值，正向/反向均为 `torch_sip` vs 小算子拼接；统一用"占小算子拼接基线的耗时比例"表示，验收线 ≤50%）：

| 场景 | torch_sip | 小算子拼接基线 | 占基线 |
|---|---|---|---|
| P0 [196000,2,2]×1 正向 | 156.9 µs | 4.8 ms | 3.3% |
| P1 [196,4,4]×4 正向 | 30.9 µs | 4.2 ms | 0.7% |
| P1 [196,3,3]×16384 正向 | 54.1 µs | 14.6 ms | 0.4% |
| P2 [196,3,3]×3 正向 | 31.4 µs | 2.6 ms | 1.2% |
| P2 [196,3,3]×1000 正向 | 26.4 µs | 2.6 ms | 1.0% |
| P0 [196000,2,2]×1 反向 | 296.0 µs | 4.8 ms | 6.1% |
| P1 [196,4,4]×4 反向 | 130.9 µs | 4.4 ms | 3.0% |
| P1 广播 [196,3,3]×16384 反向 | 311.5 µs | 15.9 ms | 2.0% |
| P2 [196,3,3]×3 反向 | 123.4 µs | 2.7 ms | 4.6% |
| P2 [196,3,3]×1000 反向 | 127.6 µs | 2.6 ms | 4.8% |

反向基线 = 拼接 solve + bmm + neg（与反向 op 同层）。

反向另有 autograd 端到端口径（`x.backward()`，仅反向）：5 个用例为 365.6~411.7 µs，较手动 `solve_backward` 多约 0.25 ms，为 Python 侧 autograd 包装与算子调度开销。性能敏感场景建议用 §4.2 的手动两阶段。

大矩阵为正确性优先的分块实现，耗时随 n 快速上升（固化套件 g-n* 口径）：n=64 约 0.22 ms（batch=8）、n=128 约 1.63 ms（batch=4）、n=256 约 11.1 ms（batch=2）。

## 7. 常见问题

**Q1: 传入 float64/float16 报错？** 当前版本仅支持 fp32；请在上层转换后再调用。

**Q2: B 只有 1 个批次、A 是 batch 个，能直接用吗？** 可以。直接传 `B [1, n, nrhs]`，kernel 原生广播（无需 expand 展开）。自动微分反向的 gB 会自动规约回 `[1, n, nrhs]`；手动调 `solve_backward` 时 gB 为满批量，需自行 `gB.sum(dim=0, keepdim=True)`。

**Q3: n>256 怎么办？** 不支持；请按分块或降阶方式在业务侧处理。

**Q4: 结果出现 inf/nan？** 检查 A 是否奇异或严重病态（本接口不做奇异检测）。对角增强（如 `A + 3*I`）可显著改善条件数。

**Q5: 反向支持自动微分吗？** 支持。A/B 任一 `requires_grad` 时 `torch_sip.solve` 自动走 autograd 路径（内部调用 `solve_backward`）；也可在 `no_grad` 下自行调用 `torch_sip.solve_backward` 做手动反向。
