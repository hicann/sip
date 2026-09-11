# SiP加速库

- [简介](overview.md)
- [安装部署](installtion_guide.md)
- [算子使用指导](operator_usage_guide.md)
- [API参考](./API_reference.md)
  - [头文件和库文件说明](../../header_files_library_files.md)
  - [base](./base.md)
    <!-- npu="A3,910b" id2 -->
    - [swapLast2Axes](../API_Reference/base/swapLast2Axes.md)
    <!-- end id2 -->
    <!-- npu="950,A3,910b" id1 -->
    - [asdMul](../API_Reference/base/asdMul.md)
    <!-- end id1 -->
  - [BLAS](./BLAS.md)
    - [BLAS公共接口](../API_Reference/BLAS/BLAS公共接口.md)
    <!-- npu="A3,910b" id3 -->
    - [Asum](../API_Reference/BLAS/Asum.md)
    <!-- end id3 -->
    <!-- npu="A3,910b" id4 -->
    - [Cal](../API_Reference/BLAS/Cal.md)
    <!-- end id4 -->
    <!-- npu="A3,910b" id5 -->
    - [Caxpy](../API_Reference/BLAS/Caxpy.md)
    <!-- end id5 -->
    <!-- npu="A3,910b" id6 -->
    - [Cgemm](../API_Reference/BLAS/Cgemm.md)
    <!-- end id6 -->
    <!-- npu="A3,910b" id7 -->
    - [Cgemv](../API_Reference/BLAS/Cgemv.md)
    <!-- end id7 -->
    <!-- npu="A3,910b" id8 -->
    - [Cgerc](../API_Reference/BLAS/Cgerc.md)
    <!-- end id8 -->
    <!-- npu="A3,910b" id9 -->
    - [Colwise_mul](../API_Reference/BLAS/Colwise_mul.md)
    <!-- end id9 -->
    <!-- npu="A3,910b" id10 -->
    - [ComplexMatDot](../API_Reference/BLAS/ComplexMatDot.md)
    <!-- end id10 -->
    <!-- npu="A3,910b" id11 -->
    - [Copy](../API_Reference/BLAS/Copy.md)
    <!-- end id11 -->
    <!-- npu="A3,910b" id12 -->
    - [Csrot](../API_Reference/BLAS/Csrot.md)
    <!-- end id12 -->
    <!-- npu="A3,910b" id13 -->
    - [Ctrmv](../API_Reference/BLAS/Ctrmv.md)
    <!-- end id13 -->
    <!-- npu="A3,910b" id14 -->
    - [Dot](../API_Reference/BLAS/Dot.md)
    <!-- end id14 -->
    <!-- npu="A3,910b" id15 -->
    - [Iamax](../API_Reference/BLAS/Iamax.md)
    <!-- end id15 -->
    <!-- npu="A3,910b" id16 -->
    - [Nrm2](../API_Reference/BLAS/Nrm2.md)
    <!-- end id16 -->
    <!-- npu="A3,910b" id17 -->
    - [Ssyr](../API_Reference/BLAS/Ssyr.md)
    <!-- end id17 -->
    <!-- npu="A3,910b" id18 -->
    - [Ssyr2](../API_Reference/BLAS/Ssyr2.md)
    <!-- end id18 -->
    <!-- npu="A3,910b" id19 -->
    - [Strmm](../API_Reference/BLAS/Strmm.md)
    <!-- end id19 -->
    <!-- npu="A3,910b" id20 -->
    - [Strmv](../API_Reference/BLAS/Strmv.md)
    <!-- end id20 -->
    <!-- npu="A3,910b" id21 -->
    - [Swap](../API_Reference/BLAS/Swap.md)
    <!-- end id21 -->
    <!-- npu="A3,910b" id22 -->
    - [HCgemmBatched](../API_Reference/BLAS/HCgemmBatched.md)
    <!-- end id22 -->
    <!-- npu="A3,910b" id23 -->
    - [CgemmBatched](../API_Reference/BLAS/CgemmBatched.md)
    <!-- end id23 -->
    <!-- npu="A3,910b" id24 -->
    - [HCgemvBatched](../API_Reference/BLAS/HCgemvBatched.md)
    <!-- end id24 -->
    <!-- npu="A3,910b" id25 -->
    - [CgemvBatched](../API_Reference/BLAS/CgemvBatched.md)
    <!-- end id25 -->
    <!-- npu="A3,910b" id26 -->
    - [HCmatinvBatched](../API_Reference/BLAS/HCmatinvBatched.md)
    <!-- end id26 -->
    <!-- npu="A3,910b" id27 -->
    - [CmatinvBatched](../API_Reference/BLAS/CmatinvBatched.md)
    <!-- end id27 -->
  - [FFT](./FFT.md)
    - [FFT公共接口](../API_Reference/FFT/FFT公共接口.md)
    <!-- npu="950,A3,910b" id28 -->
    - [FFT_1D](../API_Reference/FFT/FFT_1D.md)
    <!-- end id28 -->
    <!-- npu="950,A3,910b" id29 -->
    - [FFT_2D](../API_Reference/FFT/FFT_2D.md)
    <!-- end id29 -->
    <!-- npu="950,A3,910b" id30 -->
    - [FFT_3D](../API_Reference/FFT/FFT_3D.md)
    <!-- end id30 -->
    <!-- npu="A3,910b" id31 -->
    - [Istft](../API_Reference/FFT/Istft.md)
    <!-- end id31 -->
  - [Filter](./Filter.md)
    <!-- npu="A3,910b" id32 -->
    - [asdConvolve](../API_Reference/Filter/asdConvolve.md)
    <!-- end id32 -->
  - [Domain](./Domain.md)
    <!-- npu="A3,910b" id33 -->
    - [rsInterpolationBySinc](../API_Reference/Domain/rsInterpolationBySinc.md)
    <!-- end id33 -->
  - [Interpolation](./Interpolation.md)
    <!-- npu="A3,910b" id34 -->
    - [asdInterpWithCoeff](../API_Reference/Interpolation/asdInterpWithCoeff.md)
    <!-- end id34 -->
- [可维可测能力](./Maintainability_and_Measurability.md)
- [安全加固](./security_hardening.md)
- [附录](./appendix.md)
  - [环境变量](./environment_variable.md)
