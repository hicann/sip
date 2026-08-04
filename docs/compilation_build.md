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

### SiP关键文件介绍

1. `scripts`目录：
   - `install.sh`：安装脚本
   - `uninstall.sh`：卸载脚本
   - `release.sh`：全自动构建与打包脚本
   - `set_env.sh`：SiP的环境变量设置文件
2. `output`目录：

   - `version.info`
3. `output/lib`目录：

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
