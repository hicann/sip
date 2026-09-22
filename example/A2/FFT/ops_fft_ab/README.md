# ops-fft 联合构建后端 A/B 验证样例

## 介绍

独立验证样例（不依赖、不修改其他样例）：同一输入分别以 internal 与 ops-fft 后端执行 C2C 1D FFT，逐位比对输出并给出耗时，同时从 INFO 日志确认 ops-fft kernel 是否实际命中。应用代码无需任何改动，后端仅由环境变量 `SIP_FFT_BACKEND` 切换。

## 环境配置

```sh
source [CANN安装路径]/set_env.sh
export ASDSIP_HOME_PATH=[SiP安装路径]
```

> 注意：SiP 安装包需以 `bash build.sh --use_ops_fft` 构建（默认构建为纯 internal，无 ops-fft 后端）。

## 使用说明

```sh
bash build.sh            # 默认 n=8192 batch=4 (Ascend910B, fft_b 路径)
bash build.sh 15000 4    # Ascend950 混合基形状 (arch35 mix 路径)
```

## 结果判读

- `[AB] 双后端输出逐位一致`：两后端计算结果一致——能执行到对比阶段即表示 ops-fft 已生效，日志中 `run via ops-fft kernel ... success` 为直调命中证据（Ascend910B 为 `FFTCoreB run via ops-fft kernel (blocks=...)`，Ascend950 为 `FftC2CCoreArch35 run via ops-fft kernel (arch35 mix, ...)`）；
- 运行中止并报错：ops-fft 为强制请求，当前形状未命中覆盖路径（见路由条件），或安装包为 internal-only 构建（未链接存件）——`asdFftExec` 返回 `ACL_ERROR_API_NOT_SUPPORT`，请核对路由表或确认在联合构建的安装包上运行。

路由条件与更多说明参考 [ops-fft 联合构建](../../../../docs/ops_fft_joint_build.md)。
