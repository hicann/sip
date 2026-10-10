# torch_sip - PyTorch 扩展用于 AscendSiP

这是一个使用 Torch Library (TORCH_LIBRARY) 为 AscendSiP 操作提供 Python 绑定的 PyTorch 扩展。

## 目录结构

```txt
sip_pta/
├── torch_sip/
│   └── __init__.py          # Python 接口
├── csrc/
│   └── base
│       └── asd_mul.cpp      # C++ 实现及 TORCH_LIBRARY 注册
├── test/
│   └── base
│        └── asd_mul.py      # python接口测试脚本
├── setup.py                 # 构建脚本（被 build.sh 调用）
└── build.sh                 # 一键构建脚本（环境自检 + 自动拉取依赖 + 构建 + 可选安装）
```

## 构建方法

### 前置要求

- C++ 编译器 (g++, gcc >= 10.3.0)
- Python >= 3.8
- 支持 NPU 的 PyTorch（需安装 torch_npu）
- 昇腾 CANN 工具包 >= 8.5.0（需 source 其 set_env.sh）
- git（用于自动拉取构建依赖 ascend-boost-comm）

### 一键构建（推荐）

```bash
conda activate your_env            # 需要已安装 torch + torch_npu
source <cann安装目录>/set_env.sh   # 配置 CANN 环境
cd sip_pta
bash build.sh --install            # 构建并安装 whl 到当前环境
```

`build.sh` 会自动完成：

1. 环境自检（python3 / torch / torch_npu / CANN，`ASCEND_HOME_PATH` 未设置时自动尝试 source 常见路径）
2. 检查 MKI 头文件来源（`BOOST_COMM_PATH`，默认 `../3rdparty/ascend-boost-comm/src/include`），
   不存在时自动 `git clone` ascend-boost-comm 到仓库 `3rdparty/` 目录——**无需手动 clone sip 仓库依赖**
3. 执行 `python3 setup.py build bdist_wheel` 生成 `dist/torch_sip-<version>-<python_tag>-<platform_tag>.whl`
4. `--install` 选项：覆盖安装到当前 python 环境

其它选项：

```bash
bash build.sh                # 仅构建 whl，不安装
bash build.sh --clean        # 清理构建产物
```

### 手动构建（等价方式）

<details>
<summary>如需精细控制，可按以下步骤手动构建</summary>

配置环境（仅构建 whl 时需要，运行时见下文"脚本运行方法"）

   1. conda环境激活: conda activate your_env 需要安装torch_npu
   2. CANN配置: 如 source /opt/xxx/ascend-toolkit/set_env.sh
   3. 环境变量: export ASDSIP_HOME_PATH=asdsip安装目录，如不配置默认为/usr/local/Ascend/asdsip/latest（也可直接 source asdsip 安装目录下的 set_env.sh，效果等价）
   4. 环境变量: export BOOST_COMM_PATH=ascend-boost-comm头文件目录（MKI 头文件，asdsip 包内不带），如不配置默认为 ../3rdparty/ascend-boost-comm/src/include；若该目录不存在（未执行过 sip 仓 build.sh），需先 git clone https://gitcode.com/cann/ascend-boost-comm.git 到 ../3rdparty/
   5. 环境变量: export USE_NINJA=1，可选配置, 是否启用ninja构建

```bash
cd sip_pta
python3 setup.py build bdist_wheel
pip install --force-reinstall --no-deps ./dist/torch_sip-<version>-<python_tag>-<platform_tag>.whl
```

</details>

## 使用方法

> **注意**：当前 asdsip 版本的 `asd_mul` 仅支持 `complex64` 复数输入，不支持 float32 等实数类型，
> 实数输入会在设备侧报 `ERR99999 UNKNOWN application exception`。

```python
import torch
import torch_sip

# 在 NPU 上创建复数张量
x = torch.complex(torch.randn(10, 10).npu(), torch.randn(10, 10).npu())
y = torch.complex(torch.randn(10, 10).npu(), torch.randn(10, 10).npu())

# 调用 asd_mul 操作
result = torch_sip.asd_mul(x, y)

print(result)  # 复数逐元素乘法结果
```

### 脚本运行方法（最小环境组合，实测验证）

**前置条件**：本节脚本依赖 asdsip 动态库与 MKI 头文件目录，请先确认已完成以下两步，否则运行会报找不到库/头文件的错误（issue #201）：

1. **安装 asdsip 包**：在 sip 仓根目录执行 `bash build.sh` 并安装产物 `output/Ascend-cann-SIP_*.run`（或已有的 `/usr/local/Ascend/asdsip` 安装目录）；
2. **生成 3rdparty 目录**：执行过一次 sip 仓根目录的 `bash build.sh` 后，`3rdparty/ascend-boost-comm`（MKI 头文件来源）会被自动创建；若仍缺失，手动执行
   `git clone https://gitcode.com/cann/ascend-boost-comm.git ../3rdparty/ascend-boost-comm`（相对 sip_pta 的 `BOOST_COMM_PATH` 默认值）。

运行 whl 无需再手动 export 任何环境变量，只需三步：

```bash
# 1. source asdsip 安装目录下的 set_env.sh（自动导出 ASDSIP_HOME_PATH 并把 lib 加入 LD_LIBRARY_PATH）
source <asdsip安装目录>/set_env.sh        # 例如 /usr/local/Ascend/asdsip/set_env.sh
# 2. 配置CANN环境
source <cann安装目录>/set_env.sh          # 例如 /usr/local/Ascend/ascend-toolkit/set_env.sh
# 3. 激活环境
conda activate your_env
# 运行测试脚本
python asd_mul.py
```

说明：asdsip 的 `set_env.sh` 内容即 `export ASDSIP_HOME_PATH=<目录>/latest/` 与
`export LD_LIBRARY_PATH=$ASDSIP_HOME_PATH/lib:$LD_LIBRARY_PATH`，手动执行这两条 export 与 source 该脚本等价，二者选其一即可。
构建期需要的 `BOOST_COMM_PATH` 在运行时完全不需要；`ASCEND_HOME_PATH`/`ASCEND_TOOLKIT_HOME` 等由 CANN 的 `set_env.sh` 自动导出，无需手动设置。

## 实现细节

### Torch Library

此实现使用 **Torch Library** (TORCH_LIBRARY)：

**优势：**

- 更好地集成到 PyTorch 的算子系统
- 自动分发到 NPU 后端
- 可与 torch.compile() 和其他 PyTorch 优化一起使用
- 比 pybind11 更简单 - 无需手动 Python 绑定代码

**关键文件：**

**base类算子：**

1. **csrc/base/asd_mul.cpp** - 完整实现

   ```cpp
   namespace sip_pta {
   at::Tensor asdMul(const at::Tensor& x, const at::Tensor& y)
   {
   c10_npu::NPUGuard guard(x.device());

       at::Tensor z = at::empty_like(x);

       auto sip_stream = c10_npu::getCurrentNPUStream().stream(false);
       int64_t n = x.numel();
       uint8_t* workspace_ptr = nullptr;
       EXEC_FUNC(AsdSip::asdMul, n, x, y, z, sip_stream, workspace_ptr);
       return z;
   }
   TORCH_LIBRARY_FRAGMENT(torch_sip, m) { m.def("asd_mul(Tensor x, Tensor y) -> Tensor"); }

   TORCH_LIBRARY_IMPL(torch_sip, PrivateUse1, m) { m.impl("asd_mul", &asdMul); }
   } // namespace sip_pta
   ```

2. **torch_sip/__init__.py** - Python 接口

   ```python
   def asd_mul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.ops.torch_sip.asd_mul(x, y)
   ```

**fft类算子：** - 完整实现

1. **csrc/fft/asd_fft_c2c.cpp** - 完整实现

   ```cpp
   namespace sip_pta {
   at::Tensor asdFftC2C(const at::Tensor& in, bool isForward)
   {
       c10_npu::NPUGuard guard(in.device());

       at::Tensor input = in.is_contiguous() ? in : in.contiguous();
       at::Tensor output = at::empty_like(input);
       auto selfShape = input.sizes();
       TORCH_CHECK(selfShape.size() >= sip_pta::DIM_2 && selfShape.size() <= sip_pta::DIM_4,
                   "Input tensor must have 2, 3 or 4 dimensions, but got ", selfShape.size());
       op_api::FFTParam param; // 准备FFTParam

       if (isForward) {
           param.direction = AsdSip::ASCEND_FFT_FORWARD;
       } else {
           param.direction = AsdSip::ASCEND_FFT_INVERSE;
       }
       param.batchSize = selfShape[0];
       param.fftType = AsdSip::ASCEND_FFT_C2C;
       param.dimType = AsdSip::asdFft1dDimType::ASCEND_FFT_HORIZONTAL;
       if (selfShape.size() == sip_pta::DIM_2) {
           // 1D
           param.fftXSize = selfShape[sip_pta::DIM_1];
       } else if (selfShape.size() == sip_pta::DIM_3) {
           // 2D
           param.fftXSize = selfShape[sip_pta::DIM_1];
           param.fftYSize = selfShape[sip_pta::DIM_2];
       } else if (selfShape.size() == sip_pta::DIM_4) {
           // 3D
           param.fftXSize = selfShape[sip_pta::DIM_1];
           param.fftYSize = selfShape[sip_pta::DIM_2];
           param.fftZSize = selfShape[sip_pta::DIM_3];
       }
       EXEC_FFT_FUNC(AsdSip::asdFftExecC2C, param, input, output);
       // 使用宏EXEC_FFT_FUNC, 参数依次为算子入口函数, FFTParam, 算子入口函数的入参(handle除外, 严格按顺序排序)
       return output;
   }

   // 2. 算子注册 (Fragment)
   TORCH_LIBRARY_FRAGMENT(torch_sip, m)
   {
       m.def("asd_fft_c2c(Tensor input, bool is_forward) -> Tensor", &asdFftC2C);
   }
   TORCH_LIBRARY_IMPL(torch_sip, PrivateUse1, m) { m.impl("asd_fft_c2c", &asdFftC2C); }
   } // namespace sip_pta
   ```

2. **torch_sip/__init__.py** - Python 接口

   ```python
   def asd_fft_c2c(input_tensor: torch.Tensor, is_forward: bool = True) -> torch.Tensor:
        return torch.ops.torch_sip.asd_fft_c2c(input_tensor, is_forward)
   ```

**blas类算子：** - 完整实现

1. **csrc/blas/asd_blas_ctrmv.cpp** - 完整实现

   ```cpp
   namespace sip_pta {
   at::Tensor asdBlasCtrmv(const at::Tensor& A, at::Tensor& x, int64_t uplo, int64_t trans, int64_t diag)
   {
   // 基本数据类型校验
   // 根据算子定义，目前仅支持 ComplexFloat 单精度复数
   TORCH_CHECK(A.scalar_type() == at::kComplexFloat && x.scalar_type() == at::kComplexFloat,
   "asd_blas_ctrmv: Tensors must be ComplexFloat (Complex64).");

   // 维度校验
   // 矩阵必须是方阵，向量长度必须与矩阵维度匹配
   TORCH_CHECK(A.dim() == sip_pta::DIM_2 && A.size(0) == A.size(1),
   "asd_blas_ctrmv: Tensor A must be a 2D square matrix.");
   TORCH_CHECK(x.dim() == sip_pta::DIM_1 && x.size(0) == A.size(0),
   "asd_blas_ctrmv: Tensor x length must match matrix A dimension.");

   c10_npu::NPUGuard guard(A.device());
   // 提取张量维度和属性参数
   int64_t n = A.size(0);
   int64_t lda = n;
   int64_t incx = 1;

   // 转换枚举参数至底层类型
   AsdSip::asdBlasFillMode_t uplo_val = static_cast<AsdSip::asdBlasFillMode_t>(uplo);
   AsdSip::asdBlasOperation_t trans_val = static_cast<AsdSip::asdBlasOperation_t>(trans);
   AsdSip::asdBlasDiagType_t diag_val = static_cast<AsdSip::asdBlasDiagType_t>(diag);

   // 构建传递给 getBlasHandle 的 Plan 参数列表, 按顺序添加AsdSip::asdBlasMakeCtrmvPlan除handle外所有参数
   std::vector<int64_t> planParam = {static_cast<int64_t>(uplo_val), n};

   // 定义 Plan 构建函数
   auto makePlan = [uplo_val, n](AsdSip::asdBlasHandle handle) {
   AsdSip::asdBlasMakeCtrmvPlan(handle, uplo_val, n);
   };

   // 执行底层算子,
   EXEC_BLAS_FUNC(AsdSip::asdBlasCtrmv, makePlan, planParam,
   uplo_val, trans_val, diag_val, n, A, lda, x, incx);
   // EXEC_BLAS_FUNC, 参数依次为算子入口函数, makePlan函数, planParam, 算子入口函数的入参(handle除外, 严格按顺序排序)
   // CTRMV 为原地修改算子，直接返回被修改后的 x
   return x;
   }

   // 算子注册模块
   TORCH_LIBRARY_FRAGMENT(torch_sip, m)
   {
   // x 为原地更新，使用 Tensor(a!) 标识可变引用
   m.def("asd_blas_ctrmv(Tensor A, Tensor(a!) x, int uplo, int trans, int diag) -> Tensor(a!)");
   }

   TORCH_LIBRARY_IMPL(torch_sip, PrivateUse1, m)
   {
   m.impl("asd_blas_ctrmv", &asdBlasCtrmv);
   }
   }
   ```

2. **torch_sip/__init__.py** - Python 接口

   ```python
   def asd_blas_ctrmv(mat_a, vec_x, uplo="L", trans="N", diag="N"):
       """
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

       u_val = uplo_map.get(uplo.upper(), 0)
       t_val = trans_map.get(trans.upper(), 0)
       d_val = diag_map.get(diag.upper(), 0)

       return torch.ops.torch_sip.asd_blas_ctrmv(mat_a, vec_x, u_val, t_val, d_val)
   ```

## 添加新算子

要添加新算子（例如 `asd_add`）：

1. **添加到 csrc/asd_mul.cpp（或创建新文件）：**

   ```cpp
   at::Tensor asd_add(const at::Tensor &x, const at::Tensor &y) {
       // 实现代码
   }

   TORCH_LIBRARY(TorchSip, m) {
       m.def("asd_add(Tensor x, Tensor y) -> Tensor");
   }

   TORCH_LIBRARY_IMPL(TorchSip, PrivateUse1, m) {
       m.impl("asd_add", &asd_add);
   }
   ```

2. **添加到 torch_sip/__init__.py：**

   ```python
   def asd_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
       return torch.ops.TorchSip.asd_add(x, y)
   ```

## 故障排除

### 构建错误

- **"torch/extension.h not found"**：确保已安装支持 NPU 的 PyTorch（torch_npu）
- **"asdsip library not found"**：设置 `ASDSIP_HOME_PATH` 指向 asdsip 安装目录，或 source 其 set_env.sh
- **"g++ not found"**：安装 g++ 编译器

### 运行时错误

- **"Device mismatch"**：确保张量在 NPU 设备上
- **"Shape mismatch"**：检查输入张量形状是否兼容
- **"torch.ops.TorchSip not found"**：确保扩展已构建并加载
- **"ImportError: libasdsip.so: cannot open shared object file"**：`LD_LIBRARY_PATH` 未包含 asdsip lib 目录，
  source asdsip 安装目录下的 set_env.sh 或手动
  `export LD_LIBRARY_PATH=<asdsip安装目录>/lib:$LD_LIBRARY_PATH`
- **"call AsdSip::asdMul failed ... ERR99999"**：检查输入 dtype，当前版本 `asd_mul` 仅支持 `complex64`，
  不支持 float32 等实数类型；可通过 `ASCEND_LAUNCH_BLOCKING=1` 同步执行以获取更准确的报错栈
