#ifndef FFT_R2C_UNPACK_H
#define FFT_R2C_UNPACK_H

#include "kernel_operator.h"
#include "simt_api/asc_simt.h"

// ═══════════════════════════════════════════════════════════════════════════
// two-real-FFT 共轭拆分：一条复信号 z=A+jB 的完整谱 Z[0..4095]（自然频率序），
// 拆出两条实行的 RFFT 半谱 A[k]、B[k]（k=0..2048，各 2049 复）。
//
//   A[k] = ( Z[k] + conj(Z[N-k]) ) / 2
//   B[k] = ( Z[k] - conj(Z[N-k]) ) / (2j)
//
// 推导（设 Z[k]=a+jb，conj(Z[N-k])=c+jd，其中 c=zjr, d=-zji）：
//   A.re = (a+c)/2 = (zkr+zjr)/2
//   A.im = (b+d)/2 = (zki-zji)/2
//   B    = (Z[k]-conj)/(2j) = (y - jx)/2 , x=a-c, y=b-d
//   B.re = (b-d)/2 = (zki+zji)/2
//   B.im = -(a-c)/2 = (zjr-zkr)/2
//
// Z 以分离格式存于 sepRe/sepIm（16×256 展平，自然频率序 flat index=k）。
// 输出直接写到 GM 的两条相邻实行（行 2s 与 2s+1），行内连续、各元素 4B 标量写。
// 说明：输出行宽 2049 复 = 4098 floats 非 32B 对齐，故不能用 MTE3 DataCopy 连续写；
// 采用 SIMT 4B 标量 store（8B 对齐恒成立，行内 coalesce）。
// ═══════════════════════════════════════════════════════════════════════════

constexpr uint32_t RP_THREADS = 2048;

__simt_vf__ __launch_bounds__(RP_THREADS) inline void simt_unpack_two_real_store(
    __gm__ float* gmOut,          // 输出 base（整块 [batch, 2049] 复交织）
    __ubuf__ float* sepRe,        // Z.re[k]，自然频率序
    __ubuf__ float* sepIm,        // Z.im[k]
    uint32_t signal,              // 复信号号 s（对应实输出行 2s / 2s+1）
    uint32_t outRowFloats)        // 半谱行宽 = 4098
{
    constexpr uint32_t N = 4096;
    constexpr uint32_t half = N / 2 + 1;   // 2049
    const uint32_t tid = threadIdx.x;

    const uint32_t baseA = (2u * signal) * outRowFloats;
    const uint32_t baseB = (2u * signal + 1u) * outRowFloats;

    for (uint32_t k = tid; k < half; k += RP_THREADS) {
        uint32_t j = (N - k) & (N - 1);   // (N-k) mod N，k=0 -> 0

        float zkr = sepRe[k];
        float zki = sepIm[k];
        float zjr = sepRe[j];
        float zji = sepIm[j];

        float ar = 0.5f * (zkr + zjr);   // Re(A[k])
        float ai = 0.5f * (zki - zji);   // Im(A[k])
        float br = 0.5f * (zki + zji);   // Re(B[k])
        float bi = 0.5f * (zjr - zkr);   // Im(B[k])

        gmOut[baseA + 2u * k]      = ar;
        gmOut[baseA + 2u * k + 1u] = ai;
        gmOut[baseB + 2u * k]      = br;
        gmOut[baseB + 2u * k + 1u] = bi;
    }
}

// Host 侧（AIV）封装
__aicore__ inline void UnpackTwoRealStore(
    __gm__ float* gmOut, LocalTensor<float>& sepRe, LocalTensor<float>& sepIm,
    uint32_t signal, uint32_t outRowFloats)
{
    AscendC::PipeBarrier<PIPE_V>();
    __ubuf__ float* sepReAddr = (__ubuf__ float*)sepRe.GetPhyAddr();
    __ubuf__ float* sepImAddr = (__ubuf__ float*)sepIm.GetPhyAddr();
    asc_vf_call<simt_unpack_two_real_store>(
        dim3(RP_THREADS), gmOut, sepReAddr, sepImAddr, signal, outRowFloats);
    AscendC::DataSyncBarrier<AscendC::MemDsbT::UB>();
    // 融合 kernel 内 SIMT 标量 store 到 GM，必须在返回前排空 V 管线，
    // 否则下一个 stage（转置 MTE2 读）可能读到未提交的旧数据（非确定性）。
    AscendC::PipeBarrier<PIPE_V>();
}

// ═══════════════════════════════════════════════════════════════════════════
// 转置散射版（P0 实验，已回退）：直接写转置布局，实测 GM 写放大 3.7× 劣化 40%。
// 保留供参考，当前使用上面的连续行版 UnpackTwoRealStore。
// ═══════════════════════════════════════════════════════════════════════════

__simt_vf__ __launch_bounds__(RP_THREADS) inline void simt_unpack_two_real_store_transposed(
    __gm__ float* gmOut,          // z base（[padHalf, fftN] 复交织）
    __ubuf__ float* sepRe,        // Z.re[k]，自然频率序
    __ubuf__ float* sepIm,        // Z.im[k]
    uint32_t signal,              // 复信号号 s（→ 实输出行 2s / 2s+1，即 z 的列索引）
    uint32_t dstRowFloats)        // 转置后行宽 = 2*fftN = 8192
{
    constexpr uint32_t N = 4096;
    constexpr uint32_t half = N / 2 + 1;   // 2049
    const uint32_t tid = threadIdx.x;

    const uint32_t colOff = 4u * signal;   // 列索引 (2s) 的 float 偏移（2 复相邻）

    for (uint32_t k = tid; k < half; k += RP_THREADS) {
        uint32_t j = (N - k) & (N - 1);

        float zkr = sepRe[k];
        float zki = sepIm[k];
        float zjr = sepRe[j];
        float zji = sepIm[j];

        float ar = 0.5f * (zkr + zjr);
        float ai = 0.5f * (zki - zji);
        float br = 0.5f * (zki + zji);
        float bi = 0.5f * (zjr - zkr);

        uint32_t rowBase = k * dstRowFloats;
        gmOut[rowBase + colOff]     = ar;
        gmOut[rowBase + colOff + 1] = ai;
        gmOut[rowBase + colOff + 2] = br;
        gmOut[rowBase + colOff + 3] = bi;
    }
}

__aicore__ inline void UnpackTwoRealStoreTransposed(
    __gm__ float* gmOut, LocalTensor<float>& sepRe, LocalTensor<float>& sepIm,
    uint32_t signal, uint32_t dstRowFloats)
{
    AscendC::PipeBarrier<PIPE_V>();
    __ubuf__ float* sepReAddr = (__ubuf__ float*)sepRe.GetPhyAddr();
    __ubuf__ float* sepImAddr = (__ubuf__ float*)sepIm.GetPhyAddr();
    asc_vf_call<simt_unpack_two_real_store_transposed>(
        dim3(RP_THREADS), gmOut, sepReAddr, sepImAddr, signal, dstRowFloats);
    AscendC::DataSyncBarrier<AscendC::MemDsbT::UB>();
}

#endif // FFT_R2C_UNPACK_H