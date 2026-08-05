# Header Files and Library Files Description

This document introduces the header files and library files of the Ascend SiP Boost (SiP) library, helping users correctly reference dependencies when compiling and running application programs.

## Interface Classification

All AsdSip interfaces are located under the `AsdSip` namespace. They are divided into the following categories by function:

| Interface Name Prefix | Module | Description |
| --- | --- | --- |
| swapLast2Axes / asdMul | Base | Basic tensor operations, including axis swapping, element-wise multiplication, etc. |
| asdFft* | FFT | FFT/STFT/ISTFT transformations, supporting 1D/2D/3D, and C2C, C2R, R2C types. |
| asdBlas* | BLAS | BLAS Level 1/2/3 operations, including matrix multiplication, vector operations, triangular solving, etc., supporting real and complex types. |
| asdConvolve* | Filter | One-dimensional convolution operations, supporting full/same/valid modes. |
| asdInterp* | Interpolation | Coefficient-based interpolation operations. |
| rs* | Domain | Specialized interfaces for specific domains such as radar signal processing, including Sinc interpolation. |

## Header File Description

When compiling AsdSip interface programs, include the corresponding header files based on the interfaces actually used. Header files are located under `include/` in the installation directory.

| Header File | Usage | Corresponding Library File |
| --- | --- | --- |
| asdsip.h | **Main entry header file**. Aggregates all public API header files. Users only need `#include "asdsip.h"` to use all interfaces. It internally includes base_api.h, blas_api.h, fft_api.h, filter_api.h, interp_api.h, and domain/rs_api.h in sequence. | libasdsip.so or libasdsip_static.a |
| base_api.h | **Basic operation interfaces**. Defines basic tensor operations such as axis swapping (swapLast2Axes) and element-wise multiplication (asdMul). Depends on utils/aspb_status.h and utils/mem_base.h. | libasdsip.so or libasdsip_static.a |
| fft_api.h | **FFT fast Fourier transform interfaces**. Defines FFT Handle management (asdFftCreate/Destroy), stream binding (asdFftSetStream), 1D/2D/3D plan creation (asdFftMakePlan1D/2D/3D), ISTFT plan creation (asdFftIstftMakePlan), execution interfaces (asdFftExecC2C/C2R/R2C/C2CSeparated/Istft), Workspace management (asdFftGetWorkspaceSize/SetWorkspace), synchronization (asdFftSynchronize), etc. Also defines enumeration types such as asdFftType, asdFftDirection, and asdFft1dDimType. | libasdsip.so or libasdsip_static.a |
| blas_api.h | **BLAS linear algebra interfaces**. Defines BLAS Handle management (asdBlasCreate/Destroy), stream binding (asdBlasSetStream), plan creation (MakeXxxPlan series), Workspace management, synchronization, etc. Provided BLAS operations include: matrix multiplication (Cgemm, CgemmBatched, HCgemmBatched), matrix-vector multiplication (Cgemv, CgemvBatched, HCgemvBatched), triangular matrix solving (Strmv, Ctrmv), matrix triangular multiplication (Strmm), rank-1 update (Cgerc), vector dot product (Sdot, Cdotu, Cdotc), vector norm (Snrm2, Scnrm2), vector absolute value sum (Sasum, Scasum), vector scaling (Sscal, Cscal, Csscal), vector copy (Scopy, Ccopy), vector swap (Sswap, Cswap), vector linear combination (Caxpy), Givens rotation (Csrot), matrix inversion (CmatinvBatched, HCmatinvBatched), column-wise multiplication (ColwiseMul), complex matrix dot product (ComplexMatDot), maximum absolute value index (Isamax, Icamax), etc. Also defines enumeration types such as asdBlasStatus, asdBlasSideMode_t, asdBlasOperation_t, asdBlasFillMode_t, and asdBlasDiagType_t. Depends on acl/acl.h and aclnn/opdev/fp16_t.h. | libasdsip.so or libasdsip_static.a |
| filter_api.h | **Filter/convolution interfaces**. Defines one-dimensional convolution operations (asdConvolve) and its Workspace size query (asdConvolveGetWorkspaceSize). Supports three convolution modes: ASD_CONVOLVE_FULL, ASD_CONVOLVE_SAME, and ASD_CONVOLVE_VALID (asdConvolveMode_t enumeration). | libasdsip.so or libasdsip_static.a |
| interp_api.h | **Interpolation interfaces**. Defines coefficient-based interpolation operations (asdInterpWithCoeff) and its Workspace size query (asdInterpWithCoeffGetWorkspaceSize). | libasdsip.so or libasdsip_static.a |
| domain/rs_api.h | **Radar signal processing domain interfaces**. Defines Sinc interpolation operations (rsInterpolationBySinc) and its Workspace size query (rsInterpolationBySincGetWorkspaceSize), oriented towards radar signal processing scenarios. | libasdsip.so or libasdsip_static.a |

## Library File Description

After the build is completed and installation is executed, library files are located under `lib/` in the installation directory.

### AsdSip Library Files

| Library File | Type | Description |
| --- | --- | --- |
| libasdsip.so | Dynamic library (shared library) | **Main user library**. Aggregates all modules: utils, base, blas, fft, filter, and interpolation. Links libasdsip_core.so. User application programs should link this library to use all AsdSip public interfaces. |
| libasdsip_static.a | Static library | **Main user library (static version)**. Has the same functionality as libasdsip.so, suitable for scenarios requiring static linking. Requires additional linking of libasdsip_core.so when linking. |
| libasdsip_core.so | Dynamic library (shared library) | **Operator core runtime library**. Contains operator registration, Kernel loading and scheduling (Ops singleton), tiling logic, etc. Internally links the MKI static library (libmki_static.a) and Ascend CANN operator compilation framework library. It is automatically depended upon by libasdsip.so. Users generally do not need to reference it separately. |
| libasdsip_host.so | Dynamic library (shared library) | **Host-side tool library**. Contains host-side auxiliary functions such as operator parameter processing, depending on the ops_utils module. |

### Third-party Dependency Libraries

The following library files are provided by the CANN software package or the MKI framework. Ensure they are available in the link path during compilation and runtime.

| Library File | Source | Description |
| --- | --- | --- |
| libmki.so / libmki_static.a | MKI framework (3rdparty/mki) | **MKI kernel abstraction framework library**. Provides core abstractions such as Tensor, Kernel, and Operation, as well as runtime scheduling capability. libasdsip_core.so statically links libmki_static.a in release mode and dynamically links libmki.so in test mode. |
| libascendcl.so | CANN software package ($ASCEND_HOME_PATH/lib64) | **Ascend CL runtime library**. Provides basic data types such as aclTensor, and runtime interfaces for device management, memory management, Stream management, etc. AsdSip public header files reference acl/acl.h. |
| libaclnn.so | CANN software package ($ASCEND_HOME_PATH/lib64) | **Ascend NN operator library**. Provides aclnn series operator interfaces. Some operations in the BLAS module depend on aclnn header files (aclnn/opdev/fp16_t.h, acl/acl_meta.h, etc.). |
