# 安装部署

请参考 **[安装CANN](https://www.hiascend.com/document/detail/zh/canncommercial/900/softwareinst/instg/instg_0089.html?OS=openEuler&InstallType=netyum)**，根据如下步骤安装信号处理加速库：

- 安装Toolkit开发套件包。
- 安装ops算子包。
- 安装NNAL神经网络加速库。

运行加速库：
需确保已经正确参考如下示例命令配置环境变量，当前以非root用户安装后的默认路径“${HOME}/Ascend”为例，请用户根据set_env.sh的实际路径进行替换。

```Cpp
source ${HOME}/Ascend/nnal/asdsip/set_env.sh
```

上述环境变量配置只在当前窗口生效，用户可以按需将以上命令写入环境变量配置文件（如.bashrc文件），环境变量列表请参考环境变量参考。
