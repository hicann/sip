# Installation and Deployment

<!-- md-trans-meta sourceCommit=a6b47bb7404ddae87dcea5848180621e53ca7580 translatedAt=2026-08-12T10:59:01.293Z pushedAt=2026-08-20T11:47:59.834Z -->

See **[Install CANN](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/softwareinst/instg/instg_0000.html?OS=openEuler&InstallType=netyum)**, and install the signal processing acceleration library according to the following steps:

- Install Toolkit.

- Install the ops operator package.

- Install the NNAL neural network acceleration library.

Run the acceleration library: Ensure that the environment variables have been correctly configured by referring to the following example commands. The default path "${HOME}/Ascend" after installation as a non-root user is used as an example. Replace it with the actual path of `set_env.sh` as needed.

```Cpp
source ${HOME}/Ascend/nnal/asdsip/set_env.sh
```

The above environment variable configuration takes effect only in the current window. You can write the above commands into an environment variable configuration file (such as the `.bashrc` file) as needed. For the list of environment variables, see *Environment Variables*.
