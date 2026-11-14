# -*- coding: utf-8 -*-
"""
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the License).
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
"""

# torch_sip.npu_inv —— 接口对齐 torch.linalg.inv 的 NPU 批量矩阵求逆（含反向）。
#
# 最终接口集 (python 入口):
#   npu_inv(a, *, out=None)                对齐 torch.linalg.inv, 自动反向 (C++ autograd)
#   npu_inv_backward(y, dy)                手动反向路径-纯反向: dA = -(Y^H dY) Y^H（H 为共轭转置；实数域退化为 Y^T）
#   (手动反向的正向阶段用 torch.no_grad() 包裹 npu_inv 即可, 不建图)
#
# 实现 (dtype 路由):
#   实数:  SiP asdBlasSmatinvBatched / HSmatinvBatched  (Gauss-Jordan 单 kernel, fp32 域计算)
#   复数:  SiP asdBlasCmatinvBatched / HCmatinvBatched  (上游复数 kernel, c64/c32 原生)
#   bf16:  实数 kernel + fp32 域计算 (cast 进出)
#
# shape 路由 (折叠后 [B, N, N]):
#   快速域 (性能优先):  实数 N < 32 且 B <= 3000 (Smatinv kernel 域)
#                      复数 N <= 256 且 B <= 3000 (Cmatinv kernel 域)
#   保底域 (能跑通):   超出上述范围 -> 分块批处理 (batch 维切块) / 大 N LU 分块求逆
import logging

import torch

logger = logging.getLogger(__name__)

# ---- 快速域限制 (kernel 直调) ----
_BATCH_LIMIT = 3000
_N_REAL_LIMIT = 32  # SmatinvBatched: n < 32
_N_CMPLX_LIMIT = 256  # CmatinvBatched: n <= 256

# ---- 支持的 dtype ----
_REAL_DTYPES = (torch.float32, torch.float16, torch.bfloat16)
_CMPLX_DTYPES = (torch.complex64, torch.complex32)
_DTYPES = _REAL_DTYPES + _CMPLX_DTYPES

# ---- 保底分块参数 ----
_FALLBACK_BATCH_CHUNK = 2048  # batch 维分块上限 (留余量)
_LU_BLOCK = 16  # 大 N 分块求逆的子块尺寸


def _check_input(a: torch.Tensor, api: str):
    if a.dim() < 2 or a.size(-1) != a.size(-2):
        raise ValueError(f"{api}: input must have shape (*, N, N)")
    if a.dtype not in _DTYPES:
        raise TypeError(
            f"{api}: unsupported dtype {a.dtype}, expect "
            f"float32/float16/bfloat16/complex64/complex32"
        )
    if a.dtype == torch.complex32 and a.is_npu:
        # torch_npu op_plugin 不支持 c32 设备侧 copy/cast (创建即失败);
        # c32 请在 CPU 构造后 .complex64 再上传
        raise TypeError(
            f"{api}: complex32 on NPU is not supported by this torch_npu "
            f"(no device-side copy/cast); build on CPU and cast to complex64 "
            f"before moving to NPU"
        )


def _materialize(x: torch.Tensor) -> torch.Tensor:
    """确保 tensor 是 storage_offset==0 的连续布局。

    SiP C++ 层 CreateAclTensorFromAtTensor 未处理 storage_offset:
    首维切片 (x[s:e]) 本身连续, .contiguous() 是 no-op, 但数据基址带
    偏移, aclTensor 按存储基址解释 -> 全批错位 (实测 B=953 切片 diff 0.2)。
    物化 (clone) 强制 storage_offset=0 后恢复 bit 级正确。
    """
    if x.storage_offset() != 0:
        return x.clone()
    return x


def _fold_batch(a: torch.Tensor):
    """(*, N, N) -> (B, N, N) + 批维信息, 供 op 调用。
    非连续视图 (stride 视图/转置切片) 先物化, kernel 只接受连续内存。"""
    batch_dims = list(a.shape[:-2])
    n = a.size(-1)
    b = 1
    for d in batch_dims:
        b *= d
    a = a.contiguous()  # no-op for contiguous inputs
    a = _materialize(a)  # 消除 storage_offset (C++ 层未处理)
    return a.reshape(b, n, n), b, n, batch_dims


def _is_cmplx(dt):
    return dt in _CMPLX_DTYPES


def _cmplx_bmm(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """复数 bmm 的 NPU 组合实现 (F2 补齐, 2026-10-24)。

    本 torch_npu 版本 NPU bmm 不支持 complex (BmmKernelNpuOpApi 调
    aclnnBatchMatMul 失败), 历史上 c64 反向因此整体移 CPU。
    此处以实数 bmm 组合 Gauss 乘法 (3 次实数 bmm, 优于朴素 4 次):
        p = (xr + xi) @ y      (a+d 型中间积)
        re = p - (d + xi @ y)  即 ac - bd
        im = p - (c + xr @ y)  即 ad + bc  (推导: p=ac+ad+bc; im=p-ac-bd)
    全程 NPU 原生 op (.real/.imag 视图 + 实数 bmm + torch.complex 组装),
    消除 D2H/H2D 往返 (~400us/次 -> NPU 常驻)。
    """
    xr, xi = x.real, x.imag
    yr, yi = y.real, y.imag
    # Gauss 3 乘法: p/ac/bd 组合
    p = torch.bmm(xr + xi, yr + yi)
    ac = torch.bmm(xr, yr)
    bd = torch.bmm(xi, yi)
    re = ac - bd
    # im = p - ac - bd
    im = p - ac - bd
    return torch.complex(re, im)


# ---------------------------------------------------------------------------
# 复数 autograd (python Function; 反向 dA = -(Y^H dY) Y^H, H=共轭转置)
# ---------------------------------------------------------------------------
class _CmplxInvFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, n):
        if n <= _N_CMPLX_LIMIT and x.shape[0] <= _BATCH_LIMIT:
            y = _inv_cmplx_fast(x, n)
        elif n <= _N_CMPLX_LIMIT:
            y = _inv_batch_fallback(
                _inv_cmplx_fast, x, x.shape[0], n, _FALLBACK_BATCH_CHUNK
            )
        else:
            with torch.no_grad():
                y = _lu_inv_large_n(x)
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx, dy):
        (y,) = ctx.saved_tensors
        # F2 补齐 (2026-10-24): dA = -(Y^H dY) Y^H, H=共轭转置。
        # 对齐 torch 复数 autograd (Wirtinger 语义, CPU c128 对照
        # diff 3.5e-17); 实矩阵时 Y^H=Y^T 退化为实数公式。
        # NPU bmm 无 complex -> _cmplx_bmm 实数组合 (Gauss 3 乘法),
        # 全 NPU 常驻, 消除 D2H/H2D 往返 (~1.5-2ms -> ~0.8ms)。
        yH = y.conj().transpose(-1, -2)
        ga = -_cmplx_bmm(_cmplx_bmm(yH, dy), yH)
        return ga, None


# ---------------------------------------------------------------------------
# 快速域: 单次 kernel 直调
# ---------------------------------------------------------------------------
def _inv_real_fast(x, n):
    """实数 [B,N,N], N<32, B<=3000 -> SmatinvBatched"""
    if torch.is_grad_enabled() and x.requires_grad:
        return torch.ops.torch_sip.smatinv_inv_autograd(x)
    return torch.ops.torch_sip.smatinv_inv_forward(x)


def _inv_cmplx_fast(x, n):
    """复数 [B,N,N], N<=256, B<=3000 -> CmatinvBatched。
    注: complex32 在本 torch_npu 版本的 op_plugin 层不支持设备侧任何
    copy/cast (copy_d2d_baseformat 报错), NPU 上无法创建/搬运 c32 tensor;
    c32 输入在 _check_input 已拦截并给出明确指引 (CPU 构造 + c64 上传),
    此处只处理 c64。"""
    return torch.ops.torch_sip.asd_blas_cmatinv_batched(x)


# ---------------------------------------------------------------------------
# 保底域: 分块批处理 (batch 超限) / LU 分块求逆 (N 超限)
# ---------------------------------------------------------------------------
def _inv_batch_fallback(fast_fn, x, b, n, chunk):
    """B > 3000: 按 batch 维分块, 每块走快速 kernel。
    切片 x[s:e] 带非零 storage_offset (contiguous no-op), C++ 层按存储基址
    解释会错位 —— 分块后必须物化 (见 _materialize)。"""
    out = torch.empty_like(x)
    for s in range(0, b, chunk):
        e = min(s + chunk, b)
        xb = _materialize(x[s:e])
        out[s:e] = fast_fn(xb, n)
    return out


def _inv_real_via_c64(x, n):
    """实数 [B,N,N] 32<=N<=256 融合路径 (v2): 直调 SgetriBatched 真实数
    LU 求逆 kernel (Smatinv 入口 n>=32 自动分派; 入口直拷不拆实虚,
    出口只拷 real plane, 带选主元)。v1 的 c64 包装路径保留为对照。

    F5 pad 方案 (2026-10-24): C++ 路由要求 n%16∈{0,8} (tail 面板有效列宽
    realN×4B 须 32B 对齐 = VEC 指令最小块; 其余 tail 走 MIX)。n%16∉{0,8}
    且 P2 条件满足 (n<=256, b>=2 或 n<=128) 时, 在 host 侧 pad 到 next-8
    对齐后走 P2, 用分块恒等式 [[A,0],[0,I]]^-1 = [[A^-1,0],[0,I]] 裁剪。
    pad 开销 (一次 zeros+copy) 远小于 MIX 差距 (b>=2 时 1.4~4.9x)。
    """
    if _p2_paddable(x, n):
        return _inv_real_pad8(x, n)
    if torch.is_grad_enabled() and x.requires_grad:
        return torch.ops.torch_sip.smatinv_inv_autograd(x)
    return torch.ops.torch_sip.smatinv_inv_forward(x)


def _p2_paddable(x, n):
    """n%16∉{0,8} 且 pad 后 (n<=256, batch 条件) 会命中 P2 路由 -> pad 划算。
    b=1 且 pad 后 n>128 时 C++ 路由仍走 MIX (batch 条件), pad 无收益。"""
    if n % 16 in (0, 8) or n > 256:
        return False
    b = x.shape[0]
    M = (n + 7) // 8 * 8
    # pad 后 M%16∈{8,0} 且满足 P2 的 (batch>=2 || n<=128) 条件
    return (M % 16 in (0, 8)) and (M <= 256) and (b >= 2 or M <= 128)


def _inv_real_pad8(x, n):
    """pad 到 next-8 对齐走 P2 + 分块恒等式裁剪:
    A_pad = [[A,0],[0,I]] -> A_pad^-1 = [[A^-1,0],[0,I]], 取左上角即 A^-1。
    (数学严格: 分块对角矩阵的逆等于对角块各自求逆)"""
    M = (n + 7) // 8 * 8
    b = x.shape[0]
    a_pad = torch.zeros(b, M, M, dtype=x.dtype, device=x.device)
    a_pad[:, :n, :n] = x
    a_pad[:, n:, n:] = torch.eye(M - n, dtype=x.dtype, device=x.device)
    if torch.is_grad_enabled() and x.requires_grad:
        y_pad = torch.ops.torch_sip.smatinv_inv_autograd(a_pad)
        # autograd 场景: pad 区梯度天然为 0 (恒等块), 直接裁剪返回
        return y_pad[:, :n, :n]
    y_pad = torch.ops.torch_sip.smatinv_inv_forward(a_pad)
    return y_pad[:, :n, :n].contiguous()


def _lu_inv_large_n(a):
    """
    大 N 保底: 基于 LU 分解的求逆 (torch 原生 op 组合)。
    A = P L U  ->  A^{-1} = U^{-1} L^{-1} P
    仅在超出 kernel N 上限时调用 (性能不追求, 保正确)。
    """
    b, n, _ = a.shape
    dev = a.device
    # LU 分解 (NPU 支持): 仅显存不足等预期内可恢复异常回退 CPU, 其余异常保留原始栈直接抛出。
    # torch_npu 对 NPU OOM 抛 torch.OutOfMemoryError (NPUException.h TORCH_CHECK_WITH);
    # 裸捕获 Exception 会把设备断言/dispatch 错误等真实故障一并吞掉, 掩盖 NPU 路径失效
    try:
        lu, piv = torch.linalg.lu_factor(a)
    except torch.OutOfMemoryError as e:
        logger.warning("LU factorization failed on NPU (OOM), fallback to CPU: %s", e)
        lu, piv = torch.linalg.lu_factor(a.cpu())
        lu, piv = lu.to(dev), piv.to(dev)
    # eye: NPU 的 eye 不支持 complex —— CPU 构造再上传 (保底路径, 性能不敏感)
    eye = torch.eye(n, dtype=a.dtype).expand(b, n, n).to(dev)
    return torch.linalg.lu_solve(lu, piv, eye)


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def npu_inv(a: torch.Tensor, *, out: torch.Tensor = None) -> torch.Tensor:
    """计算批量方阵的逆矩阵, 接口对齐 torch.linalg.inv (支持反向传播)。

    - 输入: (*, M, M), 任意批维
    - dtype: float32/float16/bfloat16/complex64/complex32
      (实数低精度与 bf16 内部以 float32 计算; complex 走原生复数 kernel)
    - 快速域 (性能最优): 实数 N<32 & B<=3000; 复数 N<=256 & B<=3000
    - 保底域 (能跑通):   更大 B 分块批处理; 更大 N 走 LU 分块求逆
    - out= 参数语义与 torch.linalg.inv 一致
    - requires_grad 时自动建图 (C++ autograd, 仅实数路径; 复数/bf16/保底路径
      经 composite op 自动建图), y.backward(dy) 即得梯度
    """
    _check_input(a, "npu_inv")
    x, b, n, batch_dims = _fold_batch(a)

    if _is_cmplx(a.dtype):
        # ---- 复数路径 (autograd: _CmplxInvFn 建图, 反向 dA = -(Y^H dY) Y^H,
        #      H=共轭转置, 由 bmm 处理) ----
        if torch.is_grad_enabled() and a.requires_grad:
            y = _CmplxInvFn.apply(x, n)
        elif n <= _N_CMPLX_LIMIT and b <= _BATCH_LIMIT:
            y = _inv_cmplx_fast(x, n)
        elif n <= _N_CMPLX_LIMIT:
            y = _inv_batch_fallback(_inv_cmplx_fast, x, b, n, _FALLBACK_BATCH_CHUNK)
        else:
            y = _lu_inv_large_n(x)
    else:
        # ---- 实数路径 (bf16 经 fp32 域) ----
        in_dtype = a.dtype
        if in_dtype == torch.bfloat16:
            x = x.float()
        if n < _N_REAL_LIMIT and b <= _BATCH_LIMIT:
            y = _inv_real_fast(x, n)
        elif n < _N_REAL_LIMIT:
            y = _inv_batch_fallback(_inv_real_fast, x, b, n, _FALLBACK_BATCH_CHUNK)
        elif n <= _N_CMPLX_LIMIT and b <= _BATCH_LIMIT:
            # 32 <= N <= 256: Cgetri 融合 kernel (c64 包装路径)
            y = _inv_real_via_c64(x, n)
        elif n <= _N_CMPLX_LIMIT:
            y = _inv_batch_fallback(_inv_real_via_c64, x, b, n, _FALLBACK_BATCH_CHUNK)
        else:
            y = _lu_inv_large_n(x)
        if in_dtype == torch.bfloat16:
            y = y.to(in_dtype)

    y = y.reshape(*batch_dims, n, n)

    if out is not None:
        out.copy_(y)
        return out
    return y


def npu_inv_backward(y: torch.Tensor, grad_y: torch.Tensor):
    """手动反向路径-纯反向: dA = -(Y^H dY) Y^H（H 为共轭转置；实数域退化为 Y^T）,
    免 autograd 引擎 (~400us/次)。

    用法 (手动反向两阶段):
        with torch.no_grad():               # 正向阶段: 不建图 (等价原 npu_inv_forward)
            y = torch_sip.npu_inv(a)
        ...                                  # loss 计算, 反传产生 dy
        ga = torch_sip.npu_inv_backward(y, dy)
    y 需由调用方保存到反传阶段; dy 为上游产生的 dL/dy。

    实数 fp32/fp16/bf16 均支持 (低精度在 fp32 域计算后回转);
    复数 c64 支持 (F2 补齐, 2026-10-24): _cmplx_bmm 实数组合,
    dA = -(Y^H dY) Y^H, H 为共轭转置（实数域退化为 Y^T）。
    """
    _check_input(y, "npu_inv_backward")
    if grad_y.shape != y.shape or grad_y.dtype != y.dtype:
        raise ValueError("npu_inv_backward: grad_y must match y in shape and dtype")
    if _is_cmplx(y.dtype):
        # F2: 复数手动反向 dA = -(Y^H dY) Y^H (共轭转置, 对齐 torch
        # 复数 autograd Wirtinger 语义); NPU bmm 无 complex -> 实数组合
        x, b, n, batch_dims = _fold_batch(y)
        g = grad_y.reshape(b, n, n)
        yH = x.conj().transpose(-1, -2)
        dA = -_cmplx_bmm(_cmplx_bmm(yH, g), yH)
        return dA.reshape(*batch_dims, n, n)
    # P1' fast path (2026-10-24 晚): fp32 连续 [B,N,N] 常见形态走 aten 直发
    # 表达式 (baddbmm+bit 级等价验证 maxdiff=0.0 —— 同一 bmm kernel, C++ op
    # 仅是 alpha=-1.0 的 baddbmm 封装)。省 torch.ops 自定义 op dispatcher
    # 链 (实测 24.9us/call, case1 0.063→0.039ms kernel 链), 与 GJ 裸表达
    # 式同一派发路径。非典型布局回退 C++ op / 通用路径。
    if (
        y.dtype == torch.float32
        and y.dim() == 3
        and y.is_contiguous()
        and grad_y.is_contiguous()
        and y.storage_offset() == 0
        and grad_y.storage_offset() == 0
    ):
        yT = y.transpose(-1, -2)
        tmp = torch.baddbmm(grad_y, yT, grad_y, beta=0.0, alpha=-1.0)
        return torch.bmm(tmp, yT)
    x, b, n, batch_dims = _fold_batch(y)
    g = grad_y.reshape(b, n, n)
    in_dtype = y.dtype
    if in_dtype != torch.float32:
        x, g = x.float(), g.float()
    dA = torch.ops.torch_sip.smatinv_backward(x, g)
    if in_dtype != torch.float32:
        dA = dA.to(in_dtype)
    return dA.reshape(*batch_dims, n, n)
