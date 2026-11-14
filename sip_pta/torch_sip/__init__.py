# -*- coding: utf-8 -*-
"""
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
"""

__version__ = "0.1.0"

import math
from typing import Optional, Tuple, Union
import torch

# 副作用导入：加载 C++ 扩展以注册 torch.ops.torch_sip 算子族（issue #257 连带：
# 修改本文件触发 ruff F401，noqa 保留必要的副作用导入语义）
import sip_custom_ops_lib  # noqa: F401

# npu_inv 快路径: 顶层绑定实现模块, 避免每次调用的函数内 import 查找 (~10us+/次)
from torch_sip import _npu_inv_impl

# Both entry points dispatch to the same registered factory. Compilers retain
# the operator object; eager calls avoid boxing the scalar factory arguments.
_asd_kaiser_window_op = torch.ops.torch_sip.asd_kaiser_window.default
_asd_kaiser_window = sip_custom_ops_lib._asd_kaiser_window_typed
_kaiser_is_compiling = torch.compiler.is_compiling


def asd_mul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.ops.torch_sip.asd_mul(x, y)


def asd_kaiser_window(
    window_length: int,
    periodic: bool = True,
    beta: float = 12.0,
    *,
    dtype: Optional[torch.dtype] = None,
    layout: Optional[torch.layout] = torch.strided,
    device: Optional[Union[torch.device, str]] = None,
    requires_grad: bool = False,
) -> torch.Tensor:
    """Create a contiguous Kaiser window on NPU using the native SiP kernel.

    ``dtype=None`` selects float32; float16 and bfloat16 are also supported.
    ``device=None`` selects the current NPU. Only strided layout is supported
    (``layout=None`` also selects strided). ``requires_grad=True`` creates a
    leaf tensor whose gradient can be accumulated by subsequent operations.
    """
    if _kaiser_is_compiling():
        return _asd_kaiser_window_op(
            window_length,
            periodic,
            beta,
            dtype=dtype,
            layout=layout,
            device=device,
            requires_grad=requires_grad,
        )
    return _asd_kaiser_window(
        window_length,
        periodic,
        beta,
        dtype=dtype,
        layout=layout,
        device=device,
        requires_grad=requires_grad,
    )


def conj(x: torch.Tensor) -> torch.Tensor:
    return torch.ops.torch_sip.conj(x)


def swap_last2_axes(x: torch.Tensor) -> torch.Tensor:
    return torch.ops.torch_sip.swap_last2_axes(x)


def asd_blas_asum(x: torch.Tensor) -> torch.Tensor:
    return torch.ops.torch_sip.asd_blas_asum(x)


def asd_blas_cal(x: torch.Tensor, alpha: Union[float, complex, int]) -> torch.Tensor:
    return torch.ops.torch_sip.asd_blas_cal(x, alpha)


def asd_blas_caxpy(
    x: torch.Tensor, y: torch.Tensor, alpha: Union[float, complex, int]
) -> torch.Tensor:
    if not isinstance(alpha, complex):
        alpha = complex(alpha)
    return torch.ops.torch_sip.asd_blas_caxpy(x, y, alpha)


def _get_trans_enum(trans: str) -> int:
    """内部辅助：将字符串转为底层 C++ 预期的整数枚举"""
    mapping = {"N": 0, "T": 1, "C": 2}
    t = trans.upper()
    if t not in mapping:
        raise ValueError(f"Invalid trans mode: {trans}. Expected 'N', 'T', or 'C'.")
    return mapping[t]


def asd_blas_cgemm(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    mat_c: torch.Tensor,
    alpha: Union[float, complex] = 1.0,
    beta: Union[float, complex] = 0.0,
    trans_a: str = "N",
    trans_b: str = "N",
) -> torch.Tensor:
    """
    复数矩阵乘加算子 (CGEMM): C = alpha * op(A) * op(B) + beta * C
    支持 A, B, C 的物理列优先排布。
    """
    t_a = _get_trans_enum(trans_a)
    t_b = _get_trans_enum(trans_b)

    # 强制转换为 complex，匹配 PTA 层的 Scalar 转换
    alpha_c = complex(alpha)
    beta_c = complex(beta)

    # 调用 torch.ops 中的 C++ 实现
    return torch.ops.torch_sip.asd_blas_cgemm(
        mat_a, mat_b, mat_c, alpha_c, beta_c, t_a, t_b
    )


def asd_blas_cgemm_batched(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    mat_c: torch.Tensor,
    alpha: Union[float, complex] = 1.0,
    beta: Union[float, complex] = 0.0,
    trans_a: str = "N",
    trans_b: str = "N",
) -> torch.Tensor:
    """
    复数矩阵乘加算子 (Batched CGEMM): C[i] = alpha * op(A[i]) * op(B[i]) + beta * C[i]
    输入 Tensor 形状应为 [Batch, M, N]。
    当前只支持alpha=1, beta=0, trans_a='N', trans_b='N'; 0 < m, k, n <= 32
    """
    t_a = _get_trans_enum(trans_a)
    t_b = _get_trans_enum(trans_b)

    alpha_c = complex(alpha)
    beta_c = complex(beta)

    return torch.ops.torch_sip.asd_blas_cgemm_batched(
        mat_a, mat_b, mat_c, alpha_c, beta_c, t_a, t_b
    )


def asd_blas_cgemv(mat_a, vec_x, vec_y, alpha=1.0, beta=0.0, trans="N"):
    t = _get_trans_enum(trans)

    alpha = complex(alpha)
    beta = complex(beta)
    return torch.ops.torch_sip.asd_blas_cgemv(mat_a, vec_x, vec_y, alpha, beta, t)


def asd_blas_cgerc(mat_a, vec_x, vec_y, alpha=1.0):
    """
    mat_a: (M, N) - 结果矩阵，将被原地修改
    vec_x: (M,)
    vec_y: (N,)
    alpha: complex scalar
    """
    alpha = complex(alpha)
    return torch.ops.torch_sip.asd_blas_cgerc(mat_a, vec_x, vec_y, alpha)


def asd_blas_cmatinv_batched(mat_a):
    """
    计算批处理复数矩阵求逆

    Args:
        mat_a: (Batch, N, N) 复数张量 (支持 torch.complex64, torch.complex32)

    Returns:
        Tensor: 形状与 mat_a 相同的逆矩阵
    """
    return torch.ops.torch_sip.asd_blas_cmatinv_batched(mat_a)


def asd_blas_ctrmv(mat_a, vec_x, uplo="L", trans="N", diag="N"):
    """
    单精度复数三角矩阵-向量乘法 (原地修改 vec_x)

    Args:
        mat_a: (N, N) 复数三角矩阵
        vec_x: (N,) 复数向量
        uplo: "L" (下三角), "U" (上三角)
        trans: "N" (无转置), "T" (转置) 或 "C" (共轭转置)
        diag: "N" (非单位对角) 或 "U" (单位对角，对角线元素视为1)
    """
    uplo_map = {"L": 0, "U": 1}
    trans_map = {"N": 0, "T": 1, "C": 2}
    diag_map = {"N": 0, "U": 1}

    # 非法选项显式报错（issue #257，对齐 strmm/strmv 的 _char_to_enum 口径）：
    # 原 dict.get(key, 0) 把拼错的 uplo/trans/diag 静默映射为默认值按错误语义执行
    u_val = _char_to_enum(uplo, uplo_map)
    t_val = _char_to_enum(trans, trans_map)
    d_val = _char_to_enum(diag, diag_map)

    return torch.ops.torch_sip.asd_blas_ctrmv(mat_a, vec_x, u_val, t_val, d_val)


def asd_blas_dot(vec_x, vec_y, conjugate=False):
    """
    向量点积计算 (支持实数与复数)

    Args:
        vec_x: 输入向量 x
        vec_y: 输入向量 y
        conjugate: 是否计算共轭点积 (仅限复数)
    """
    return torch.ops.torch_sip.asd_blas_dot(vec_x, vec_y, conjugate)


def asd_blas_iamax(vec_x):
    """
    寻找向量中绝对值（或复数实虚部绝对值之和）最大的元素索引

    Args:
        vec_x: (N,) 目标向量，支持 Float32 或 Complex64

    Returns:
        int: 最大元素对应的索引标量
    """
    # 获取底层的 INT32 索引张量
    res_tensor = torch.ops.torch_sip.asd_blas_iamax(vec_x)

    # 转为标量数值
    idx = res_tensor.item()
    return idx


def asd_blas_cgemv_batched(mat_a, vec_x, vec_y, alpha=1.0, beta=0.0, trans="N"):
    t = _get_trans_enum(trans)

    alpha = complex(alpha)
    beta = complex(beta)
    return torch.ops.torch_sip.asd_blas_cgemv_batched(
        mat_a, vec_x, vec_y, alpha, beta, t
    )


def asd_blas_copy(tensor_x, out=None):
    x_input = tensor_x.contiguous()

    if out is not None:
        if not out.is_contiguous():
            raise ValueError("asd_blas_copy: The 'out' tensor must be contiguous.")
        return torch.ops.torch_sip.asd_blas_copy_out(x_input, out)
    else:
        return torch.ops.torch_sip.asd_blas_copy(x_input)


def asd_blas_colwise_mul(mat, vec):
    return torch.ops.torch_sip.asd_blas_colwise_mul(mat, vec)


def asd_blas_complex_mat_dot(mat_x, mat_y):
    return torch.ops.torch_sip.asd_blas_complex_mat_dot(mat_x, mat_y)


def asd_blas_nrm2(vec_x):
    """
    计算向量的欧氏范数 (L2 Norm)
    内部自动根据数据类型分发到 Snrm2 (实数) 或 Scnrm2 (复数) 算子

    Args:
        vec_x: (N,) 目标向量，支持 Float32 或 Complex64

    Returns:
        float: 向量的欧氏范数标量值
    """
    res_tensor = torch.ops.torch_sip.asd_blas_nrm2(vec_x)
    return res_tensor.item()


def asd_blas_csrot(vec_x: torch.Tensor, vec_y: torch.Tensor, c: float, s: float):
    """
    公式:
        x' = c * x + s * y
        y' = -s * x + c * y

    Args:
        vec_x: (N,) 复数向量，将被原地覆盖更新
        vec_y: (N,) 复数向量，将被原地覆盖更新
        c: 旋转矩阵的余弦值 (实数)
        s: 旋转矩阵的正弦值 (实数)

    Returns:
        Tuple[Tensor, Tensor]
    """
    return torch.ops.torch_sip.asd_blas_csrot(vec_x, vec_y, float(c), float(s))


def asd_blas_ssyr(
    mat_a: torch.Tensor, vec_x: torch.Tensor, alpha: float = 1.0, uplo: str = "L"
):
    """
    公式:
        A = alpha * x * x^T + A
    Args:
        mat_a: (N, N) 对称矩阵，将被原地更新
        vec_x: (N,) 列向量
        alpha: 缩放因子
        uplo: "U" (仅更新上三角) 或 "L" (仅更新下三角)
    Returns:
        Tensor: 更新后的 mat_a 引用
    """
    # 映射枚举值
    uplo_map = {"L": 0, "U": 1}
    # 非法选项显式报错（issue #257，对齐 strmm/strmv 口径）
    u_val = _char_to_enum(uplo, uplo_map)

    return torch.ops.torch_sip.asd_blas_ssyr(mat_a, vec_x, alpha, u_val)


def asd_blas_ssyr2(
    mat_a: torch.Tensor,
    vec_x: torch.Tensor,
    vec_y: torch.Tensor,
    alpha: float = 1.0,
    uplo: str = "L",
):
    """
    公式:
        A = alpha * x * y^T + alpha * y * x^T + A
    Args:
        mat_a: (N, N) 目标对称矩阵，其内容将被原地覆盖更新
        vec_x: (N,) 输入向量 x
        vec_y: (N,) 输入向量 y
        alpha: 更新的缩放标量因子
        uplo: "U" (仅更新上三角) 或 "L" (仅更新下三角)
    Returns:
        Tensor: 更新完成后的 mat_a 引用
    """
    # 构建枚举值映射字典
    uplo_map = {"L": 0, "U": 1}
    # 非法选项显式报错（issue #257，对齐 strmm/strmv 口径）
    u_val = _char_to_enum(uplo, uplo_map)
    return torch.ops.torch_sip.asd_blas_ssyr2(mat_a, vec_x, vec_y, alpha, u_val)


def _char_to_enum(char_val: str, valid_dict: dict) -> int:
    char_upper = char_val.upper()
    if char_upper not in valid_dict:
        raise ValueError(
            f"Invalid option '{char_val}'. Expected one of {list(valid_dict.keys())}."
        )
    return valid_dict[char_upper]


def asd_blas_strmm(mat_a, mat_b, alpha=1.0, side="L", uplo="L", trans="N"):
    side_val = _char_to_enum(side, {"L": 0, "R": 1})
    uplo_val = _char_to_enum(uplo, {"L": 0, "U": 1})
    trans_val = _char_to_enum(trans, {"N": 0, "T": 1})

    alpha_val = float(alpha)

    return torch.ops.torch_sip.asd_blas_strmm(
        mat_a, mat_b, alpha_val, side_val, uplo_val, trans_val
    )


def asd_blas_strmv(mat_a, vec_x, uplo="L", trans="N", diag="N"):
    uplo_val = _char_to_enum(uplo, {"L": 0, "U": 1})
    trans_val = _char_to_enum(trans, {"N": 0, "T": 1})
    diag_val = _char_to_enum(diag, {"N": 0, "U": 1})

    return torch.ops.torch_sip.asd_blas_strmv(
        mat_a, vec_x, uplo_val, trans_val, diag_val
    )


def asd_blas_swap(vec_x, vec_y):
    return torch.ops.torch_sip.asd_blas_swap(vec_x, vec_y)


def asd_interp_with_coeff(x, coefficient):
    """
    向量插值算子 (InterpWithCoeff) 简化入口
    :param x: 输入信号张量，形状 [batch, nRs, totalSubcarrier]
    :param coefficient: 插值系数张量，形状 [batch, 14-nRs, nRs]
    :return: 输出张量，形状 [batch, 14-nRs, totalSubcarrier]
    """
    return torch.ops.torch_sip.asd_interp_with_coeff(x, coefficient)


def asd_convolve(signal, kernel):
    """
    一维滤波卷积算子 (Convolve) 简化入口
    :param signal: 输入的复数一维信号，形状 [BatchCount, n]
    :param kernel: 输入的实数滤波卷积核，形状 [k]
    :return: 滤波后的复数信号，形状 [BatchCount, n]
    """
    return torch.ops.torch_sip.asd_convolve(signal, kernel)


def rs_interpolation_by_sinc(
    input_tensor, sinc_tab, pos_floor, pos_to_tab_index, interp_num=16, quant_num=32
):
    """
    批量一维复数向量 Sinc 插值算子
    :param input_tensor: 原始信号，Complex64, 形状 [batch, signal_length]
    :param sinc_tab: 插值系数矩阵，Float32
    :param pos_floor: 坐标向下取整值，Int32, 形状 [batch, interp_length]
    :param pos_to_tab_index: 坐标小数映射的行号索引，Int16, 形状 [batch, interp_length]
    :param interp_num: 插值点数 (默认 16，需为偶数)
    :param quant_num: 量化点数 (默认 32，需为2的幂)
    :return: 插值结果，Complex64, 形状 [batch, interp_length]
    """
    return torch.ops.torch_sip.rs_interpolation_by_sinc(
        input_tensor, sinc_tab, pos_floor, pos_to_tab_index, interp_num, quant_num
    )


def asd_fft_c2c(input_tensor: torch.Tensor, is_forward: bool = True) -> torch.Tensor:
    return torch.ops.torch_sip.asd_fft_c2c(input_tensor, is_forward)


def asd_fft_c2c_sep(
    in_real: torch.Tensor, in_imag: torch.Tensor, is_forward: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.torch_sip.asd_fft_c2c_sep(in_real, in_imag, is_forward)


def asd_fft_r2c(input_tensor: torch.Tensor, is_forward: bool = True) -> torch.Tensor:
    """
    Perform Real-to-Complex FFT on NPU.
    Input: Real tensor (Float32)
    Output: Complex tensor (Complex64) with last dimension size n/2 + 1
    """
    return torch.ops.torch_sip.asd_fft_r2c(input_tensor, is_forward)


def asd_fft_c2r(input_tensor: torch.Tensor, is_forward: bool = True) -> torch.Tensor:
    return torch.ops.torch_sip.asd_fft_c2r(input_tensor, is_forward)


def fft_shift(
    input_tensor: torch.Tensor,
    row: int = None,
    col: int = None,
    batch: int = None,
    dist: int = 0,
) -> torch.Tensor:
    """
    FFT Shift: 将批量复数 slice 的零频分量移到各 slice 中心
    (对齐 CUDA fftShiftBatch2DKernel 语义)。
    当前版本仅支持 2D slice（row x col），其他维度支持待后续版本补齐。

    output[s][(r + (row+1)//2) % row][(c + (col+1)//2) % col] = input[s][r][c]

    Args:
        input_tensor: complex64 张量
            - 3D [batch, row, col]（可省略 row/col/batch 自动推断）
            - 或 1D 扁平布局（须显式指定 row/col/batch/dist）
        row:  每 slice 行数（3D 输入可省略）
        col:  每 slice 列数（3D 输入可省略）
        batch: slice 数（3D 输入可省略）
        dist:  输入 slice 间元素步距；<= 0 表示连续（dist = row*col）。
               dist > row*col 时输入 slice 间有 padding。

    Returns:
        complex64 张量，形状 [batch, row, col]

    Example:
        >>> x = torch.randn(12000, 500, 50, dtype=torch.complex64, device="npu")
        >>> y = fft_shift(x)                    # 自动推断
        >>> y = fft_shift(x, 500, 50, 12000)    # 显式指定
    """
    if row is None or col is None or batch is None:
        # 自动推断：要求 3D [batch, row, col]
        shape = input_tensor.shape
        if len(shape) != 3:
            raise ValueError(
                f"fft_shift: auto-infer requires a 3D [batch, row, col] tensor, "
                f"got {len(shape)}D shape {shape}. "
                f"Explicitly pass row/col/batch for flattened input."
            )
        return torch.ops.torch_sip.asd_fft_shift_auto(input_tensor)
    return torch.ops.torch_sip.asd_fft_shift(input_tensor, row, col, batch, dist)


def ifft_shift(
    input_tensor: torch.Tensor,
    row: int = None,
    col: int = None,
    batch: int = None,
    dist: int = 0,
    scale: float = 1.0,
) -> torch.Tensor:
    """
    IFFT Shift: fft_shift 的逆变换，将批量复数 slice 的零频分量从中心移回角落
    (对齐 CUDA ifftShiftScaleBatch2DKernel 语义)。
    当前版本仅支持 2D slice（row x col），其他维度支持待后续版本补齐。

    output[s][(r + row//2) % row][(c + col//2) % col] = scale * input[s][r][c]

    偶数维度下与 fft_shift 排列一致；奇数维度下两者相差 1（floor 而非 ceil）。
    scale 为逐元素输出乘子（kernel 内 float32 计算），可把逆 FFT 的
    1/(row*col) 归一化折叠进本 pass；scale=1 时数值逐位不变（纯排列）。

    Args:
        input_tensor: complex64 张量
            - 3D [batch, row, col]（可省略 row/col/batch 自动推断）
            - 或 1D 扁平布局（须显式指定 row/col/batch/dist）
        row:  每 slice 行数（3D 输入可省略）
        col:  每 slice 列数（3D 输入可省略）
        batch: slice 数（3D 输入可省略）
        dist:  输入 slice 间元素步距；<= 0 表示连续（dist = row*col）。
               dist > row*col 时输入 slice 间有 padding。
        scale: 逐元素输出乘子（float32），默认 1.0。

    Returns:
        complex64 张量，形状 [batch, row, col]

    Example:
        >>> x = torch.randn(12000, 500, 50, dtype=torch.complex64, device="npu")
        >>> y = ifft_shift(x)                              # 自动推断, scale=1
        >>> y = ifft_shift(x, 500, 50, 12000, scale=1/25000)  # 折叠逆 FFT 归一化
    """
    if row is None or col is None or batch is None:
        # 自动推断：要求 3D [batch, row, col]
        shape = input_tensor.shape
        if len(shape) != 3:
            raise ValueError(
                f"ifft_shift: auto-infer requires a 3D [batch, row, col] tensor, "
                f"got {len(shape)}D shape {shape}. "
                f"Explicitly pass row/col/batch for flattened input."
            )
        # 浮点相等判断使用 math.isclose（G.TYP.02）；与 1.0 近似的 scale 视为默认值
        if dist != 0 or not math.isclose(scale, 1.0):
            return torch.ops.torch_sip.asd_ifft_shift(
                input_tensor, shape[1], shape[2], shape[0], dist, scale
            )
        return torch.ops.torch_sip.asd_ifft_shift_auto(input_tensor)
    return torch.ops.torch_sip.asd_ifft_shift(
        input_tensor, row, col, batch, dist, scale
    )


def npu_inv(a: torch.Tensor, *, out: torch.Tensor = None) -> torch.Tensor:
    """批量实数矩阵求逆 (接口对齐 torch.linalg.inv), 支持反向传播。"""
    return _npu_inv_impl.npu_inv(a, out=out)


def npu_inv_backward(y: torch.Tensor, grad_y: torch.Tensor):
    """手动反向路径-纯反向: ga = npu_inv_backward(y, dy), dA = -(Y^H dY) Y^H（H 为共轭转置；实数域退化为 Y^T）。

    y 为正向阶段 npu_inv(a) 在 no_grad 下的输出 (调用方保存到反传阶段); dy 为上游产生的 dL/dy。
    """
    return _npu_inv_impl.npu_inv_backward(y, grad_y)


class _SolveFn(torch.autograd.Function):
    """solve 的自动微分包装: 反向唤起 solve_backward。

    反向解析式: gB = solve(Aᵀ, gX); gA = -(gB·Xᵀ)。
    B 为广播形态 [1, n, nrhs] 时, gB 需沿 batch 维求和归约 (kernel 前向不展开 B)。
    """

    @staticmethod
    def forward(ctx, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            X = torch.ops.torch_sip.solve_forward(A, B)
        ctx.b_bcast = B.dim() == 3 and B.size(0) == 1 and A.size(0) != 1
        ctx.save_for_backward(A, X)
        return X

    @staticmethod
    def backward(ctx, grad_X: torch.Tensor):
        A, X = ctx.saved_tensors
        gB, gA = torch.ops.torch_sip.solve_backward(A, X, grad_X)
        if ctx.b_bcast:
            gB = gB.sum(
                dim=0, keepdim=True
            )  # 广播 B 的梯度沿 batch 规约回 [1, n, nrhs]
        return gA, gB  # 与 forward 入参 (A, B) 顺序一致


def _solve_check(A: torch.Tensor, B: torch.Tensor) -> None:
    """范围校验须在 op 调用之前: mki 异常后 OpCommand 会跳过后续 handler(返回假成功)."""
    # 结构/维度错误交由 C++ 绑定层拦截, 保持报错语义 (issue #400): 高维 size 访问
    # 必须在 dim 守卫之后, 否则 2D 等维度不足输入会先触发裸 IndexError
    if A.dim() != 3 or B.dim() != 3:
        return
    if A.size(0) < 1:
        raise RuntimeError("solve: batch must be >= 1")
    if A.size(1) < 2 or A.size(1) > 256:
        raise RuntimeError(f"solve: n must be in [2, 256], got {A.size(1)}")
    if B.size(2) < 1:
        raise RuntimeError("solve: cols(nrhs) must be >= 1")
    if B.size(0) != A.size(0) and B.size(0) != 1:
        raise RuntimeError(
            f"solve: B batch must be {A.size(0)} or 1 (broadcast), got {B.size(0)}"
        )


def _solve_materialize(x: torch.Tensor) -> torch.Tensor:
    """非连续/带偏移视图转连续零偏移(同 _npu_inv_impl._materialize)."""
    if not x.is_contiguous():
        return x.contiguous()
    if x.storage_offset() != 0:
        return x.clone()
    return x


def solve(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """批量求解线性方程组 A·X = B (fp32, n∈[2,256]).

    支持自动微分: A/B 任一 requires_grad 时走 _SolveFn (反向调 solve_backward);
    无梯度需求时直通 solve_forward。直调 torch.ops 会绕过本层校验与物化.

    B 的 batch 可为 1 (广播语义, kernel 直接支持, 不做 expand 展开), 输出 X 的 batch 跟随 A.

    Args:
        A: 系数矩阵 [batch, n, n]
        B: 右端项 [batch|1, n, nrhs]
    Returns:
        X: 解 [batch, n, nrhs] (batch 跟随 A)
    """
    _solve_check(A, B)
    A = _solve_materialize(A)
    B = _solve_materialize(B)
    if torch.is_grad_enabled() and (A.requires_grad or B.requires_grad):
        return _SolveFn.apply(A, B)
    return torch.ops.torch_sip.solve_forward(A, B)


def solve_backward(A: torch.Tensor, X: torch.Tensor, gX: torch.Tensor) -> tuple:
    """solve 反向: gB = solve(Aᵀ, gX), gA = -(gB @ Xᵀ)"""
    return torch.ops.torch_sip.solve_backward(A, X, gX)
