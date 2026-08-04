# 编译与构建

## SiP编译

### 下载加速库源码

```shell
git clone https://gitcode.com/cann/sip.git
```

您可自行选择需要的分支。

### 编译

用户需进入加速库的根目录并进行编译

```shell
cd ${SiP_root_path}
bash build.sh
```

【注】：SiP库编译依赖ascend-boost-comm，因此该编译过程涉及①拉取并编译ascend-boost-comm（昇腾分布式通信加速库）仓；②编译SiP加速库两个过程。更多命令介绍可查看sip仓主目录下的[`README.md`](../README.md)和[`build.sh`](../build.sh)文件。

> 完整依赖清单请参考根目录 [`requirements.txt`](../requirements.txt) 与 [`README.md`](../README.md) 的「环境构建」章节。

### SiP编译相关说明

SiP的基本编译命令是`bash build.sh`，在默认构建模式下，该命令会生成版本信息，并创建安装包。

  ```sh
  bash build.sh
  ```

  可通过使用--help参数获取build.sh脚本功能已及对应的指令。

  ```sh
  --help                         Displays help message.
  --dev                          仅编译算子库, 若type为空，默认为dev.
  --clean                        清除缓存和依赖的三方库.
  --ut                           编译执行单元测试用例.
```

### 编译目标芯片架构配置

SiP库支持多种Ascend芯片架构，编译时通过[`configs/build_config.json`](../configs/build_config.json)配置需要编译的目标芯片。该文件内容如下：

```json
{
    "targets": {
        "ascend310b": false,
        "ascend310p": false,
        "ascend910b": true,
        "ascend950": true
    }
}
```

各字段含义说明：

| 芯片架构 | 对应产品系列 |
| --- | --- |
| `ascend310b` | Atlas 200I/500 A2 推理卡 |
| `ascend310p` | Atlas 300I 推理卡 |
| `ascend910b` | Atlas A2 训练/推理服务器 |
| `ascend950` | Ascend 950PR/950DT |

将需要编译的目标设置为`true`，不需要编译的目标设置为`false`。

> **注意**：默认配置同时启用了`ascend910b`和`ascend950`两个目标。若您的运行环境仅支持其中一种芯片（例如仅支持Ascend910），请将不支持的芯片架构设置为`false`，否则编译不支持的芯片目标时会因硬件特性不匹配而失败（典型报错如`simd_vf function 'RegCompute' must be a free function or static member function`）。

#### 自定义编译配置文件

除修改项目自带的[`configs/build_config.json`](../configs/build_config.json)外，还可通过环境变量`BUILD_CONFIG_FILE`指定自定义的配置文件路径，使项目在编译时读取该文件而非默认配置。例如：

```sh
export BUILD_CONFIG_FILE=/path/to/your/build_config.json
bash build.sh
```

> **说明**：编译系统实际读取的配置文件由[`scripts/build_util.py`](../scripts/build_util.py)中的`get_build_target_list()`函数决定。当未设置`BUILD_CONFIG_FILE`环境变量时，默认读取项目根目录下的`configs/build_config.json`；设置该环境变量后，则读取指定路径的配置文件。项目依赖的`mki`组件（位于`3rdparty/mki/`）虽也带有`configs/build_config.json`，但SiP编译流程以项目自身的配置文件为准，无需修改`3rdparty/mki/`下的配置。

### SiP关键文件介绍

1. `scripts`目录：
   - `install.sh`：安装脚本
   - `uninstall.sh`：卸载脚本
   - `release.sh`：全自动构建与打包脚本
   - `set_env.sh`：SiP的环境变量设置文件
   - `build_util.py`：编译辅助脚本，负责读取编译目标配置（`build_config.json`）及算子二进制打包
2. `configs`目录：

   - `build_config.json`：编译目标芯片架构配置文件
   - `op_list.yaml`：算子列表配置
3. `output`目录：

   - `version.info`
4. `output/lib`目录：

   - `libasdsip.so`：SiP加速库的动态链接库文件
   - `libmki.so`：MKI库的动态链接库文件

## 配置文件

### 编译文件`build.sh`

文件名：`build.sh`
加速库编译文件，文件中可设置日志存放目录、日志文件、编译器版本等，一般无需更改。

#### 环境变量设置文件`set_env.sh`

​**文件名**​：`scripts/set_env.sh`
加速库安装完成后，系统提供`set_env.sh`（进程级环境变量设置脚本），以自动完成环境变量设置，用户进程结束后自动失效。
若想查看相关变量含义说明，可访问[环境变量参考-CANN社区版-昇腾社区](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/maintenref/envvar/envref_07_0001.html)。
