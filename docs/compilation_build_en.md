# Compilation and Build

## SiP Compilation

### Download the Acceleration Library Source Code

```shell
git clone https://gitcode.com/cann/sip.git
```

You can select any branch as needed.

### Compile

Users need to enter the root directory of the acceleration library and compile

```shell
cd ${SiP_root_path}
bash build.sh
```

Note: SiP library compilation depends on ascend-boost-comm. Therefore, the compilation process involves two steps: ① pulling and compiling the ascend-boost-comm (Ascend Distributed Communication Acceleration Library) repository; ② compiling the SiP acceleration library. For more command introductions, refer to the [`README.md`](../README.md) and [`build.sh`](../build.sh) files in the main directory of the SiP repository.

> For the complete dependency list, refer to the [`requirements.txt`](../requirements.txt) and the "Environment Setup" section in [`README.md`](../README.md) in the root directory.

### SiP Compilation Related Notes

The basic compilation command for SiP is `bash build.sh`. In the default build mode, this command generates version information and creates an installation package.

  ```sh
  bash build.sh
  ```

  Use the --help parameter to obtain the build.sh script functions and corresponding instructions.

  ```sh
  --help                         Displays help message.
  --dev                          Compile only the operator library. If type is empty, the default is dev.
  --clean                        Clear caches and dependent third-party libraries.
  --ut                           Compile and execute unit test cases.
```

### Compilation Target Chip Architecture Configuration

The SiP library supports multiple Ascend chip architectures. During compilation, configure the target chips to compile through [`configs/build_config.json`](../configs/build_config.json). The file content is as follows:

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

The meaning of each field is as follows:

| Chip Architecture | Corresponding Product Series |
| --- | --- |
| `ascend310b` | Atlas 200I/500 A2 Inference Card |
| `ascend310p` | Atlas 300I Inference Card |
| `ascend910b` | Atlas A2 Training/Inference Server |
| `ascend950` | Ascend 950PR/950DT |

Set the targets that need compilation to `true`, and set the targets that do not need compilation to `false`.

> **Note**: The default configuration enables both `ascend910b` and `ascend950` targets simultaneously. If your runtime environment only supports one of these chips (for example, only supports Ascend910), set the unsupported chip architecture to `false`. Otherwise, compiling an unsupported chip target will fail due to hardware feature mismatch (typical error such as `simd_vf function 'RegCompute' must be a free function or static member function`).

#### Custom Compilation Configuration File

In addition to modifying the project's built-in [`configs/build_config.json`](../configs/build_config.json), you can also specify a custom configuration file path through the environment variable `BUILD_CONFIG_FILE`, so that the project reads that file instead of the default configuration during compilation. For example:

```sh
export BUILD_CONFIG_FILE=/path/to/your/build_config.json
bash build.sh
```

> **Note**: The configuration file actually read by the compilation system is determined by the `get_build_target_list()` function in [`scripts/build_util.py`](../scripts/build_util.py). When the `BUILD_CONFIG_FILE` environment variable is not set, the system reads `configs/build_config.json` in the project root directory by default. When the environment variable is set, the system reads the configuration file at the specified path. Although the `mki` component (located in `3rdparty/mki/`) that the project depends on also has a `configs/build_config.json`, the SiP compilation process uses the project's own configuration file as the standard. No modification to the configuration under `3rdparty/mki/` is required.

### SiP Key File Introduction

1. `scripts` directory:
   - `install.sh`: Installation script
   - `uninstall.sh`: Uninstallation script
   - `release.sh`: Fully automatic build and packaging script
   - `set_env.sh`: SiP environment variable setup file
   - `build_util.py`: Compilation helper script. Responsible for reading the compilation target configuration (`build_config.json`) and operator binary packaging.
2. `configs` directory:

   - `build_config.json`: Compilation target chip architecture configuration file
   - `op_list.yaml`: Operator list configuration
3. `output` directory:

   - `version.info`
4. `output/lib` directory:

   - `libasdsip.so`: Dynamic link library file of the SiP acceleration library
   - `libmki.so`: Dynamic link library file of the MKI library

## Configuration Files

### Compilation File `build.sh`

File name: `build.sh`
This is the acceleration library compilation file. You can set the log storage directory, log file, compiler version, etc. in this file. Generally, no modification is required.

#### Environment Variable Setup File `set_env.sh`

​**File name**​: `scripts/set_env.sh`
After the acceleration library is installed, the system provides `set_env.sh` (a process-level environment variable setup script) to automatically complete environment variable setup. The settings expire automatically after the user process ends.
If you want to view the descriptions of related variables, visit [Environment Variable Reference - CANN Community Edition - Ascend Community](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/maintenref/envvar/envref_07_0001.html).
