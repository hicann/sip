# Header Files and Library Files

<!-- md-trans-meta sourceCommit=a6b47bb7404ddae87dcea5848180621e53ca7580 translatedAt=2026-08-12T11:00:32.793Z pushedAt=2026-08-20T11:47:59.847Z -->

This document describes the header files and library files of the Ascend SiP Boost (SiP) library, helping users correctly reference dependencies when compiling and running applications.

## Interface Classification

AsdSip interfaces are unified under the `AsdSip` namespace and are categorized by function as follows:

| Interface Name Prefix | Module | Description |
| --- | --- | --- |
| swapLast2Axes / asdMul | Base | Basic tensor operations, including axis swapping and element-wise multiplication. |
| asdFft* | FFT (Fast Fourier Transform) | FFT/STFT/ISTFT transforms, supporting 1D/2D/3D, and C2C, C2R, and R2C types. |
| asdBlas* | BLAS (Linear Algebra) | BLAS Level 1/2/3 operations, including matrix multiplication, vector operations, and triangular solve, supporting real and complex data types. |
| asdConvolve* | Filter | One-dimensional convolution operations, supporting full, same, and valid modes. |
| asdInterp* | Interpolation | Coefficient-based interpolation operations. |
| rs* | Domain | Dedicated interfaces for specific domains such as radar signal processing, e.g., Sinc interpolation. |

## Header File Description

When compiling AsdSip interface programs, include the corresponding header files based on the interfaces actually used. Header files are located under the `include/` directory of the installation path.

| Header File | Purpose | Corresponding Library File |
| --- | --- | --- |
| asdsip.h | **Master header file** that aggregates all public API header files. Users only need `#include "asdsip.h"` to use all interfaces. Internally includes base_api.h, blas_api.h, fft_api.h, filter_api.h, interp_api.h, and domain/rs_api.h in sequence. | libasdsip.so or libasdsip_static.a |
| base_api.h | **Basic operation interfaces**. Defines basic tensor operations such as axis swap (`swapLast2Axes`) and element-wise multiplication (`asdMul`). Depends on `utils/aspb_status.h` and `utils/mem_base.h`. | libasdsip.so or libasdsip_static.a |
| fft_api.h | **FFT interfaces**. Defines FFT handle management (`asdFftCreate`/`Destroy`), stream binding (`asdFftSetStream`), 1D/2D/3D plan creation (`asdFftMakePlan1D`/`2D`/`3D`), ISTFT plan creation (`asdFftIstftMakePlan`), execution interfaces (`asdFftExecC2C`/`C2R`/`R2C`/`C2CSeparated`/`Istft`), workspace management (`asdFftGetWorkspaceSize`/`SetWorkspace`), synchronization (`asdFftSynchronize`), and more. Also defines enumeration types such as `asdFftType`, `asdFftDirection`, and `asdFft1dDimType`. | libasdsip.so or libasdsip_static.a |
| blas_api.h | **BLAS linear algebra interfaces**. Defines BLAS handle management (`asdBlasCreate`/`Destroy`), stream binding (`asdBlasSetStream`), plan creation (`MakeXxxPlan` series), workspace management, synchronization, and more. Supported BLAS operations include: matrix multiplication (`Cgemm`, `CgemmBatched`, `HCgemmBatched`), matrix-vector multiplication (`Cgemv`, `CgemvBatched`, `HCgemvBatched`), triangular matrix solve (`Strmv`, `Ctrmv`), triangular matrix multiplication (`Strmm`), rank-one update (`Cgerc`), vector dot product (`Sdot`, `Cdotu`, `Cdotc`), vector norm (`Snrm2`, `Scnrm2`), sum of absolute values (`Sasum`, `Scasum`), vector scaling (`Sscal`, `Cscal`, `Csscal`), vector copy (`Scopy`, `Ccopy`), vector swap (`Sswap`, `Cswap`), vector linear combination (`Caxpy`), Givens rotation (`Csrot`), matrix inversion (`CmatinvBatched`, `HCmatinvBatched`), column-wise multiplication (`ColwiseMul`), complex matrix dot product (`ComplexMatDot`), index of the element with the maximum absolute value (`Isamax`, `Icamax`), and more. Also defines enumeration types such as `asdBlasStatus`, `asdBlasSideMode_t`, `asdBlasOperation_t`, `asdBlasFillMode_t`, and `asdBlasDiagType_t`. Depends on `acl/acl.h` and `aclnn/opdev/fp16_t.h`. | libasdsip.so or libasdsip_static.a |
| filter_api.h | **Filter/convolution interfaces**. Defines the one-dimensional convolution operation (`asdConvolve`) and its workspace size query (`asdConvolveGetWorkspaceSize`). Supports three convolution modes (`asdConvolveMode_t` enumeration): `ASD_CONVOLVE_FULL`, `ASD_CONVOLVE_SAME`, and `ASD_CONVOLVE_VALID`. | libasdsip.so or libasdsip_static.a |
| interp_api.h | **Interpolation interfaces**. Defines the coefficient-based interpolation operation (`asdInterpWithCoeff`) and its workspace size query (`asdInterpWithCoeffGetWorkspaceSize`). | libasdsip.so or libasdsip_static.a |
| domain/rs_api.h | **Radar signal processing domain interfaces**. Defines the Sinc interpolation operation (`rsInterpolationBySinc`) and its workspace size query (`rsInterpolationBySincGetWorkspaceSize`), targeting radar signal processing scenarios. | libasdsip.so or libasdsip_static.a |

## Library File Description

After the build is complete and installation is performed, the library files are located under the `lib/` directory of the installation directory.

### AsdSip Library Files

| Library File | Type | Description |
| --- | --- | --- |
| libasdsip.so | Dynamic library (shared library) | **Main user library**. Aggregates all modules including `utils`, `base`, `blas`, `fft`, `filter`, and `interpolation`, and links `libasdsip_core.so`. User applications should link this library to use all public AsdSip APIs. |
| libasdsip_static.a | Static library | **Main user library (static version)**. Provides the same functionality as `libasdsip.so` and is suitable for scenarios requiring static linking. `libasdsip_core.so` must be additionally linked during linking. |
| libasdsip_core.so | Dynamic library (shared library) | **Operator core runtime library**. Contains operator registration, kernel loading and scheduling (Ops singleton), tiling logic, etc. Internally links the MKI static library (`libmki_static.a`) and the Ascend CANN operator compilation framework library. Automatically depended on by `libasdsip.so`; users typically do not need to reference it separately. |
| libasdsip_host.so | Dynamic library (shared library) | **Host-side utility library**. Contains host-side auxiliary functions such as operator parameter processing, and depends on the `ops_utils` module. |

### Third-Party Dependency Libraries

The following library files are provided by the CANN software package or the MKI framework. Ensure they are available in the link path during compilation and runtime.

| Library File | Source | Description |
| --- | --- | --- |
| libmki.so / libmki_static.a | MKI framework (3rdparty/mki) | **MKI kernel abstraction framework library**. Provides core abstractions such as Tensor, Kernel, and Operation, as well as runtime scheduling capabilities. `libasdsip_core.so` statically links `libmki_static.a` in release mode and dynamically links `libmki.so` in test mode. |
| libascendcl.so | CANN software package ($ASCEND_HOME_PATH/lib64) | **Ascend CL runtime library**. Provides basic data types such as `aclTensor`, as well as runtime interfaces for device management, memory management, and stream management. `acl/acl.h` is referenced in the AsdSip public header files. |
| libaclnn.so | CANN software package ($ASCEND_HOME_PATH/lib64) | **Ascend NN operator library**. Provides the aclnn series of operator interfaces. Some operations in the BLAS module depend on aclnn header files (`aclnn/opdev/fp16_t.h`, `acl/acl_meta.h`, etc.). |
