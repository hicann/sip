// fft_kernel_1d_4096_by_2_16_16_2_2_2.cpp — FFT1D C2C（N=4096，radix 序列 2×16×16×2×2×2）
// 流水：P1Aiv（首级 radix-2 蝶形 + T0 尾乘）→ P2Aiv（gather → B0）→ AIC GEMM#1
//       → P4Aiv（T1 复乘 + 重排 → B1）→ AIC GEMM#2 → P6Aiv（T2 复乘 + L5/L4/L3 蝶形 + interleave）→ y
// 【DIT】末级按 DIT 组织：T2 输出按 bitrev3 行写平面，L5→L4→L3 蝶形后输出自然序，末步为纯 interleave。
// AIV 计算全部为 SIMD 实现（见 fft_op/fft_simt_ops.h）；通用件在 fft_op/fft_kernel_utils.h。
#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include "catlass/arch/arch.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/catlass.hpp"
#include "catlass/status.hpp"

#include "fft_kernel.h"
#include "fft_kernel_utils.h"
#include "fft_tiling_def.h"
#include "fft_simt_ops.h"
#include "fft_kernel_transpose_stage.h"

#include "kernel_operator.h"
#include "helper.hpp"

using namespace Catlass;
using namespace AscendC;

namespace Fft {

using ArchTag = Arch::Ascend950;

// ── UB 缓冲起始地址（float 偏移）【布局峰值 58560 floats = 228.75KB ≤ 248KB】──
// [T 常驻 16384] → [S 乒乓 2×8192] → [SBO 乒乓 2×8192] → [P6 平面 8192]
//   → [P6_TW 896] → [GATHER_TABLE 64] → [P6T2 128] → [P4 128]
// T（常驻）：T0tw 紧凑表 4096（只含 k0=1 流）+ T1 4096 + T2Exp 全展开 8192；
// S：P1 蝶形 x→T0 原地 + GEMM C 半列（C0/C1 32×256），按 pair 乒乓；
// SBO：B0/B1 暂存 + y 交织输出，按 pair 乒乓（批内连续写 + DataCopy 读，单份会跨 pair 竞争）；
// P6：Re/Im 分离平面 8×512×2（SIMD LoadAlign 对齐），P6 专用独立缓冲（避免与 S/SBO 竞争）。
constexpr int64_t UB_OFFSET_T_TABLE    = 0;       // T0tw 紧凑表（4096 floats，host BuildT0TwiddleCompact）
constexpr int64_t UB_OFFSET_T1         = 4096;    // T1 起点（4096 floats）
constexpr int64_t UB_OFFSET_T2         = 8192;    // T2Exp 全展开表起点（8192 floats，host BuildT2Expanded；
                                                   //   Re 段 [0,4096) + Im 段 [4096,8192)）
constexpr int64_t UB_OFFSET_S_PING     = 16384;   // x→T0 + C 半列（8192 floats）pair0
constexpr int64_t UB_OFFSET_S_PONG     = 24576;   // 同上 pair1 乒乓
constexpr int64_t UB_OFFSET_SBO_PING   = 32768;   // B0/B1 暂存 + y 输出（8192 floats）pair0
constexpr int64_t UB_OFFSET_SBO_PONG   = 40960;   // 同上 pair1 乒乓
constexpr int64_t UB_OFFSET_P6         = 49152;   // P6 Re/Im 平面中间量（8192 floats，8×512×2，单份）
constexpr int64_t UB_OFFSET_P6_TW      = 57344;   // P6 radix-2 蝶形 twiddle 展开表（896 floats）
constexpr int64_t UB_OFFSET_GATHER_TABLE = 58240; // 步⑤ gather 偏移表段占位（64 floats；DIT 版步⑤ 零 Gather 不消费，保留以维持 coeffs 偏移口径）
constexpr int64_t UB_OFFSET_P6T2_TABLE = 58304;   // 步① T2 复乘 gather 表（128 floats；前 64 项被消费）
constexpr int64_t UB_OFFSET_P4_TABLE   = 58432;   // P4 twiddle gather 表（128 floats，P4_OFF + P4_TW）

// ── CrossCore flag 方向 slot【双流水线 × 每线 2 轮 GEMM = 4 个方向 slot】──
// 每轮 GEMM 的 B/G 分用不同 slot：AIV 各批 Set(B) 与 AIC 各 GEMM Wait(B) 时序交错，合并同 id 会错配。
constexpr uint32_t SLOT_B0 = 0;   // AIV→AIC: B0 就绪（GEMM#1，P2 后 Set / P3 前 Wait）
constexpr uint32_t SLOT_G0 = 1;   // AIC→AIV: GEMM#1 完成（P3 后 Set / P4 前 Wait）
constexpr uint32_t SLOT_B1 = 2;   // AIV→AIC: B1 就绪（GEMM#2，P4 后 Set / P5 前 Wait）
constexpr uint32_t SLOT_G1 = 3;   // AIC→AIV: GEMM#2 完成（P5 后 Set / P6 前 Wait）

// flag id 单点计算（两方向集中于此，Phase 函数内禁止私算）
CATLASS_DEVICE uint16_t FftFlagId(uint32_t pair, uint32_t slot)
{
    return static_cast<uint16_t>((pair & 1) * 4 + slot);   // 双流水线 × 每线 4 方向 slot
}

// ── 核内 EventID（Phase 内三段握手；与 fft_kernel_utils.h SYNC_ID0/1 的 MTE3_MTE2 乒乓错开）──
constexpr event_t EVENT_MTE2_V = EVENT_ID2;   // MTE2 读完成 → SIMD 计算
constexpr event_t EVENT_V_MTE3 = EVENT_ID3;   // SIMD 算完 → MTE3 搬出

// ─── Phase 函数：每个 AIV 相关 phase 一个（P1/P2/P4/P6）──
// 数据源规则：P1 输入来自 GM → MTE2_V 三段握手；P4/P6 输入来自 FIX 直写共享缓冲（C 半列）→ 无 MTE2_V。
// 跨核 Set/Wait（B/G flag）由主循环负责，Phase 内只管核内三段。

// P1（首）：读 x + radix-2 蝶形 + T0 尾乘 → T0@S 区（输入来自 GM，MTE2_V 三段；T0 留 S 供 P2，无 MTE3）
CATLASS_DEVICE void P1Aiv(
    FftKernelContext& ctx, Arch::Resource<Arch::Ascend950>& resource,
    __gm__ float* gmInput, int64_t myRow, uint32_t pair)
{
    int64_t sOff = (pair & 1) ? UB_OFFSET_S_PONG : UB_OFFSET_S_PING;

    // MTE2 读 x 行（交织复行 8192 floats）→ S 区（pair 乒乓）
    LocalTensor<float> ubS = resource.ubBuf.template GetBufferByByte<float>(sOff * sizeof(float));
    GlobalTensor<float> gmIn;
    gmIn.SetGlobalBuffer(gmInput + myRow * ctx.rowFloats);
    DataCopy(ubS, gmIn, ctx.rowFloats);
    SetFlag<HardEvent::MTE2_V>(EVENT_MTE2_V);
    WaitFlag<HardEvent::MTE2_V>(EVENT_MTE2_V);

    // x（交织复行）→ 蝶形 + T0 尾乘 → T0（复交织）原地写回 S 区
    LocalTensor<float> ubTw = resource.ubBuf.template GetBufferByByte<float>(UB_OFFSET_T_TABLE * sizeof(float));
    __ubuf__ float* sAddr  = (__ubuf__ float*)ubS.GetPhyAddr();
    __ubuf__ float* twAddr = (__ubuf__ float*)ubTw.GetPhyAddr();
    simd_p1_butterfly(sAddr, sAddr, twAddr);
}

// P2（gather）：T0@S 区 → gather 16 点组 → 写 B0 单信号列子块（输入来自 S 区，V_MTE3 三段）
CATLASS_DEVICE void P2Aiv(
    FftKernelContext& ctx, Arch::Resource<Arch::Ascend950>& resource,
    __gm__ float* gmWs, uint32_t subIdx, uint32_t pair)
{
    int64_t sOff = (pair & 1) ? UB_OFFSET_S_PONG : UB_OFFSET_S_PING;
    int64_t boOff = (pair & 1) ? UB_OFFSET_SBO_PONG : UB_OFFSET_SBO_PING;

    // T0（S 区）→ B0 单信号列子块（紧凑 32×256）写 SBO（pair 乒乓）
    LocalTensor<float> ubS  = resource.ubBuf.template GetBufferByByte<float>(sOff * sizeof(float));
    LocalTensor<float> ubBo = resource.ubBuf.template GetBufferByByte<float>(boOff * sizeof(float));
    __ubuf__ float* sAddr  = (__ubuf__ float*)ubS.GetPhyAddr();
    __ubuf__ float* boAddr = (__ubuf__ float*)ubBo.GetPhyAddr();
    simd_gather_b0(boAddr, sAddr);

    // MTE3 写 WS（B0 列子块：每行 256 列，dstGap 展开到 B0 全局 512 列；列偏移 subIdx×256）
    SetFlag<HardEvent::V_MTE3>(EVENT_V_MTE3);
    WaitFlag<HardEvent::V_MTE3>(EVENT_V_MTE3);
    int64_t bOff = static_cast<int64_t>(ctx.blockIdx) * ctx.wsRegionBytes
                 + ctx.stage[1].gemm.bOffset
                 + static_cast<int64_t>(pair & 1) * 16384
                 + static_cast<int64_t>(subIdx) * 256;
    DataCopyParams p;
    p.blockCount = 32;
    p.blockLen = 256 * sizeof(float) / 32;
    p.srcGap = 0;
    p.dstGap = (512 - 256) * sizeof(float) / 32;
    GlobalTensor<float> gmB;
    gmB.SetGlobalBuffer(gmWs + bOff);
    DataCopy(gmB, ubBo, p);
}

// P4（级间 1）：T1 复乘 + 重排 → 写 B1 单信号列子块（输入来自 FIX 直写 C0，无 MTE2_V）
CATLASS_DEVICE void P4Aiv(
    FftKernelContext& ctx, Arch::Resource<Arch::Ascend950>& resource,
    __gm__ float* gmWs, uint32_t subIdx, uint32_t pair)
{
    int64_t sOff = (pair & 1) ? UB_OFFSET_S_PONG : UB_OFFSET_S_PING;

    // C0（S 区）半列 × T1 → B1 单信号列子块（紧凑 32×256）写 SBO（pair 乒乓）
    LocalTensor<float> ubC  = resource.ubBuf.template GetBufferByByte<float>(sOff * sizeof(float));
    LocalTensor<float> ubTw = resource.ubBuf.template GetBufferByByte<float>(UB_OFFSET_T1 * sizeof(float));
    int64_t boOff = (pair & 1) ? UB_OFFSET_SBO_PONG : UB_OFFSET_SBO_PING;
    LocalTensor<float> ubBo = resource.ubBuf.template GetBufferByByte<float>(boOff * sizeof(float));
    __ubuf__ float* cAddr  = (__ubuf__ float*)ubC.GetPhyAddr();
    __ubuf__ float* twAddr = (__ubuf__ float*)ubTw.GetPhyAddr();
    __ubuf__ float* boAddr = (__ubuf__ float*)ubBo.GetPhyAddr();
    __ubuf__ uint32_t* p4Addr = (__ubuf__ uint32_t*)resource.ubBuf
        .template GetBufferByByte<float>(UB_OFFSET_P4_TABLE * sizeof(float)).GetPhyAddr();
    simd_twiddle_rotate_r4(boAddr, cAddr, twAddr, p4Addr, p4Addr + 64);

    // MTE3 写 WS（B1 列子块：每行 256 列，dstGap 展开到 B1 全局 512 列；列偏移 subIdx×256）
    SetFlag<HardEvent::V_MTE3>(EVENT_V_MTE3);
    WaitFlag<HardEvent::V_MTE3>(EVENT_V_MTE3);
    int64_t bOff = static_cast<int64_t>(ctx.blockIdx) * ctx.wsRegionBytes
                 + ctx.stage[2].gemm.bOffset
                 + static_cast<int64_t>(pair & 1) * 16384
                 + static_cast<int64_t>(subIdx) * 256;
    DataCopyParams p;
    p.blockCount = 32;
    p.blockLen = 256 * sizeof(float) / 32;
    p.srcGap = 0;
    p.dstGap = (512 - 256) * sizeof(float) / 32;
    GlobalTensor<float> gmB;
    gmB.SetGlobalBuffer(gmWs + bOff);
    DataCopy(gmB, ubBo, p);
}

// P6（末，DIT）：T2 复乘（按 bitrev3 行写平面）+ L5→L4→L3 三级 radix-2 蝶形 + 纯 interleave → y@GM
// 输入来自 FIX 直写 C1（S 区）；P6 平面中间量用独立缓冲（UB_OFFSET_P6），SBO 仅承载 y 交织输出
CATLASS_DEVICE void P6Aiv(
    FftKernelContext& ctx, Arch::Resource<Arch::Ascend950>& resource,
    __gm__ float* gmOutput, int64_t myRow, uint32_t subIdx, uint32_t pair)
{
    int64_t sOff = (pair & 1) ? UB_OFFSET_S_PONG : UB_OFFSET_S_PING;
    int64_t boOff = (pair & 1) ? UB_OFFSET_SBO_PONG : UB_OFFSET_SBO_PING;

    LocalTensor<float> ubC  = resource.ubBuf.template GetBufferByByte<float>(sOff * sizeof(float));
    LocalTensor<float> ubTw = resource.ubBuf.template GetBufferByByte<float>(UB_OFFSET_T2 * sizeof(float));
    LocalTensor<float> ubP6 = resource.ubBuf.template GetBufferByByte<float>(UB_OFFSET_P6 * sizeof(float));
    LocalTensor<float> ubBo = resource.ubBuf.template GetBufferByByte<float>(boOff * sizeof(float));
    __ubuf__ float* cAddr  = (__ubuf__ float*)ubC.GetPhyAddr();
    __ubuf__ float* twAddr = (__ubuf__ float*)ubTw.GetPhyAddr();
    __ubuf__ float* p6Addr = (__ubuf__ float*)ubP6.GetPhyAddr();
    __ubuf__ float* boAddr = (__ubuf__ float*)ubBo.GetPhyAddr();

    // ① C1 实展开 × T2Exp → Re/Im 分离平面写 P6 缓冲（8×512×2，独立缓冲，无读写竞争）
    __ubuf__ uint32_t* p6t2Addr = (__ubuf__ uint32_t*)resource.ubBuf
        .template GetBufferByByte<float>(UB_OFFSET_P6T2_TABLE * sizeof(float)).GetPhyAddr();
    simd_p6_t2_expanded(p6Addr, p6Addr + 4096, cAddr, twAddr, p6t2Addr);
    AscendC::PipeBarrier<PIPE_V>();          // 算子间同步（防下一级读旧数据）
    // ② L5 radix-2 蝶形（P6 平面 in-place，DIT 首级无 twiddle）
    simd_radix2_l5(p6Addr, p6Addr + 4096);
    AscendC::PipeBarrier<PIPE_V>();
    // ③④ L4 + L3 radix-2 蝶形（P6 平面 in-place，DIT 顺序 L4→L3，twiddle 用 P6_TW 展开表）
    __ubuf__ float* p6TwAddr = (__ubuf__ float*)resource.ubBuf
        .template GetBufferByByte<float>(UB_OFFSET_P6_TW * sizeof(float)).GetPhyAddr();
    simd_radix2_l4(p6Addr, p6Addr + 4096, p6TwAddr);
    AscendC::PipeBarrier<PIPE_V>();
    simd_radix2_l3(p6Addr, p6Addr + 4096, p6TwAddr);
    AscendC::PipeBarrier<PIPE_V>();
    // ⑤ 纯 interleave（P6 平面 → SBO 交织，DIT 输出自然序，零 bitrev 零 gather）
    simd_interleave_planes(boAddr, p6Addr, p6Addr + 4096);
    AscendC::PipeBarrier<PIPE_V>();

    // MTE3 写 y@GM（交织复行 8192 floats 连续）
    SetFlag<HardEvent::V_MTE3>(EVENT_V_MTE3);
    WaitFlag<HardEvent::V_MTE3>(EVENT_V_MTE3);
    GlobalTensor<float> gmOut;
    gmOut.SetGlobalBuffer(gmOutput + myRow * ctx.rowFloats);
    DataCopy(gmOut, ubBo, ctx.rowFloats);
}

// ═══ 流水主体（99 例程模型：双信号列拼接 + SPLIT_N + 双 AIV 全 pair）═══
// 关键（mode=2 flag 语义）：AIC 的 Wait(B, pair) 需两个 AIV 都 Set(B, pair) 才解除。
//   因此 AIV0/AIV1 都遍历所有 pair（j 循环），每 pair 各处理自己的信号（subIdx），各 Set(B) 一次。
// 数据布局：mmN = 2×单信号（B/C 列 = [信号0 组列 | 信号1 组列]）；
//   B0/B1 各 2 份乒乓槽（pair 奇偶交替，16384 floats/份）；
//   AIV0 写/读信号0 列区、AIV1 信号1 列区；C 经 SPLIT_N 拆给双 AIV；S/SBO 缓冲 pair 乒乓（(pair&1)）。
CATLASS_DEVICE void PipelineDualPingpong(
    FftKernelContext& ctx, Arch::Resource<Arch::Ascend950>& resource,
    __gm__ float* gmInput, __gm__ float* gmOutput,
    __gm__ float* gmCoeffs, __gm__ float* gmWs)
{
    uint32_t numPairs = (ctx.rowEnd - ctx.rowStart) / 2;

    // ── AIC 侧（group 循环：每 group = 2 pair，逐 GEMM 批 P3/P5）──
    if ASCEND_IS_AIC {
        FftBlockMmadToUb blockMmadToUb(resource, 0);
        uint32_t groups = (numPairs + 1) / 2;
        for (uint32_t i = 0; i < groups; i++) {
            uint32_t pairsInGroup = (i == groups - 1 && numPairs % 2 != 0) ? 1 : 2;
            // ── P3 批（GEMM #1，stage[1]）：Wait(B0) → GEMM → Set(G0) ──
            for (uint32_t j = 0; j < pairsInGroup; j++) {
                uint32_t pair = 2 * i + j;
                if (pair >= numPairs) break;
                LocalTensor<float> ubGemm = resource.ubBuf.template GetBufferByByte<float>(
                    ((pair & 1) ? UB_OFFSET_S_PONG : UB_OFFSET_S_PING) * sizeof(float));
                CrossCoreWaitFlag<FFT_SYNC_MODE, PIPE_MTE2>(FftFlagId(pair, SLOT_B0));
                int64_t bOff = static_cast<int64_t>(ctx.blockIdx) * ctx.wsRegionBytes
                             + ctx.stage[1].gemm.bOffset + static_cast<int64_t>(pair & 1) * 16384;
                RunGemmToUb(blockMmadToUb, gmCoeffs, ctx.stage[1].gemm.opOffset,
                            gmWs, bOff, ubGemm,
                            ctx.stage[1].gemm.mmM, ctx.stage[1].gemm.mmK, ctx.stage[1].gemm.mmN);
                CrossCoreSetFlag<FFT_SYNC_MODE, PIPE_FIX>(FftFlagId(pair, SLOT_G0));
            }
            // ── P5 批（GEMM #2，stage[2]）：Wait(B1) → GEMM → Set(G1) ──
            for (uint32_t j = 0; j < pairsInGroup; j++) {
                uint32_t pair = 2 * i + j;
                if (pair >= numPairs) break;
                LocalTensor<float> ubGemm = resource.ubBuf.template GetBufferByByte<float>(
                    ((pair & 1) ? UB_OFFSET_S_PONG : UB_OFFSET_S_PING) * sizeof(float));
                CrossCoreWaitFlag<FFT_SYNC_MODE, PIPE_MTE2>(FftFlagId(pair, SLOT_B1));
                int64_t bOff = static_cast<int64_t>(ctx.blockIdx) * ctx.wsRegionBytes
                             + ctx.stage[2].gemm.bOffset + static_cast<int64_t>(pair & 1) * 16384;
                RunGemmToUb(blockMmadToUb, gmCoeffs, ctx.stage[2].gemm.opOffset,
                            gmWs, bOff, ubGemm,
                            ctx.stage[2].gemm.mmM, ctx.stage[2].gemm.mmK, ctx.stage[2].gemm.mmN);
                CrossCoreSetFlag<FFT_SYNC_MODE, PIPE_FIX>(FftFlagId(pair, SLOT_G1));
            }
        }
    }

    // ── AIV 侧（bootstrap + 双 AIV 全 pair 主循环 + 收尾）──
    if ASCEND_IS_AIV {
        uint32_t subIdx = AscendC::GetSubBlockIdx();

        // ── Phase 0: T 表与 gather 表 DataCopy 常驻 ──
        LocalTensor<float> ubTw0 = resource.ubBuf.template GetBufferByByte<float>(UB_OFFSET_T_TABLE * sizeof(float));
        GlobalTensor<float> gmT0;
        gmT0.SetGlobalBuffer(gmCoeffs + ctx.stage[0].butterfly.twiddleOffset);
        DataCopy(ubTw0, gmT0, T0_COMPACT_FLOATS);   // T0tw 紧凑表（只含 k0=1 流）
        LocalTensor<float> ubTw1 = resource.ubBuf.template GetBufferByByte<float>(UB_OFFSET_T1 * sizeof(float));
        GlobalTensor<float> gmT1;
        gmT1.SetGlobalBuffer(gmCoeffs + ctx.stage[1].gemm.tOffset);
        DataCopy(ubTw1, gmT1, 4096);
        LocalTensor<float> ubTw2 = resource.ubBuf.template GetBufferByByte<float>(UB_OFFSET_T2 * sizeof(float));
        GlobalTensor<float> gmT2;
        gmT2.SetGlobalBuffer(gmCoeffs + ctx.stage[2].gemm.tOffset);
        DataCopy(ubTw2, gmT2, T2_EXPANDED_FLOATS);  // T2Exp 全展开表
        LocalTensor<float> ubP6Tw = resource.ubBuf.template GetBufferByByte<float>(UB_OFFSET_P6_TW * sizeof(float));
        GlobalTensor<float> gmP6Tw;
        gmP6Tw.SetGlobalBuffer(gmCoeffs + ctx.stage[3].butterfly.twiddleOffset);
        DataCopy(ubP6Tw, gmP6Tw, 896);              // P6 radix-2 蝶形 twiddle 展开表
        LocalTensor<float> ubP6T2 = resource.ubBuf.template GetBufferByByte<float>(UB_OFFSET_P6T2_TABLE * sizeof(float));
        GlobalTensor<float> gmP6T2;
        gmP6T2.SetGlobalBuffer(gmCoeffs + ctx.p6t2TableOffset);
        DataCopy(ubP6T2, gmP6T2, P6T2_TABLE_FLOATS); // 步① T2 复乘 gather 表
        LocalTensor<float> ubP4 = resource.ubBuf.template GetBufferByByte<float>(UB_OFFSET_P4_TABLE * sizeof(float));
        GlobalTensor<float> gmP4;
        gmP4.SetGlobalBuffer(gmCoeffs + ctx.p4TableOffset);
        DataCopy(ubP4, gmP4, P4_TABLE_FLOATS);       // P4 twiddle gather 表
        SetFlag<HardEvent::MTE2_V>(EVENT_MTE2_V);
        WaitFlag<HardEvent::MTE2_V>(EVENT_MTE2_V);

        // ── Bootstrap：纯 flag 放行首对 pair 的输入缓冲（无真实数据）──
        SetFlag<HardEvent::MTE3_MTE2>(SYNC_ID0);
        SetFlag<HardEvent::MTE3_MTE2>(SYNC_ID1);

        // ── 主循环：双 AIV 都遍历所有 pair（j 循环），每 pair 各处理自己的信号（subIdx）──
        // 7 槽周期（P1→P2→P4→P6，AIC 侧 P3/P5 两段）批式咬合；S/SBO 按 (pair&1) 乒乓
        uint32_t groups = (numPairs + 1) / 2;
        for (uint32_t i = 0; i < groups; i++) {
            uint32_t pairsInGroup = (i == groups - 1 && numPairs % 2 != 0) ? 1 : 2;

            // P1 批（首）：数据从 GM 来——MTE2 搬入，先 Wait<MTE3_MTE2>（S 区可写）
            for (uint32_t j = 0; j < pairsInGroup; j++) {
                uint32_t pair = 2 * i + j;
                if (pair >= numPairs) break;
                event_t syncId = (pair & 1) ? SYNC_ID1 : SYNC_ID0;
                int64_t gp = ctx.rowStart / 2 + pair;
                WaitFlag<HardEvent::MTE3_MTE2>(syncId);
                P1Aiv(ctx, resource, gmInput, gp * 2 + subIdx, pair);
            }

            // P2 批：T0(S 区)→B0@WS，Set(B0) 放行 GEMM#1
            for (uint32_t j = 0; j < pairsInGroup; j++) {
                uint32_t pair = 2 * i + j;
                if (pair >= numPairs) break;
                P2Aiv(ctx, resource, gmWs, subIdx, pair);
                CrossCoreSetFlag<FFT_SYNC_MODE, PIPE_MTE3>(FftFlagId(pair, SLOT_B0));
            }

            // P4 批：数据从 Fixpipe（C0 直写 S 区）——Wait(G0) 后 twiddle→B1
            for (uint32_t j = 0; j < pairsInGroup; j++) {
                uint32_t pair = 2 * i + j;
                if (pair >= numPairs) break;
                CrossCoreWaitFlag<FFT_SYNC_MODE, PIPE_V>(FftFlagId(pair, SLOT_G0));
                P4Aiv(ctx, resource, gmWs, subIdx, pair);
                CrossCoreSetFlag<FFT_SYNC_MODE, PIPE_MTE3>(FftFlagId(pair, SLOT_B1));
            }

            // P6 批：数据从 Fixpipe（C1 直写 S 区）——Wait(G1) 后 T2+蝶形+输出 y
            for (uint32_t j = 0; j < pairsInGroup; j++) {
                uint32_t pair = 2 * i + j;
                if (pair >= numPairs) break;
                event_t syncId = (pair & 1) ? SYNC_ID1 : SYNC_ID0;
                int64_t gp = ctx.rowStart / 2 + pair;
                CrossCoreWaitFlag<FFT_SYNC_MODE, PIPE_V>(FftFlagId(pair, SLOT_G1));
                P6Aiv(ctx, resource, gmOutput, gp * 2 + subIdx, subIdx, pair);
                SetFlag<HardEvent::MTE3_MTE2>(syncId);
            }
        }

        // ── 收尾：消费两组 EventID 的最后一轮 SetFlag（漏则下次 launch 死锁）──
        WaitFlag<HardEvent::MTE3_MTE2>(SYNC_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(SYNC_ID1);
    }
}

// ═══════════ 主 Kernel 入口 ═══════════
CATLASS_GLOBAL __mix__(1, 2) void FftKernel(
    GM_ADDR input, GM_ADDR output, GM_ADDR coeffs, GM_ADDR workspace, GM_ADDR tiling,
    int32_t ubMode, int32_t mixMode)
{
    FftKernelContext ctx;
    InitContext((__gm__ FftTilingData*)tiling, ctx);

    // blockIdx 越界守卫（blockDim > coreNum 时多余核直接退出）
    if (ctx.blockIdx >= static_cast<uint32_t>(
            ((__gm__ FftTilingData*)tiling)->multiCoreParams.coreNum)) {
        return;
    }

    Arch::Resource<Arch::Ascend950> resource;

    __gm__ float* gmInput = (__gm__ float*)(input);
    __gm__ float* gmOutput = (__gm__ float*)(output);
    __gm__ float* gmCoeffs = (__gm__ float*)(coeffs);
    __gm__ float* gmWs = (__gm__ float*)(workspace);

    PipelineDualPingpong(ctx, resource, gmInput, gmOutput, gmCoeffs, gmWs);
}

void FftKernelLaunch(
    GM_ADDR input, GM_ADDR output, GM_ADDR coeffs, GM_ADDR workspace, GM_ADDR tiling,
    uint32_t blockDim, aclrtStream stream, int32_t ubMode, int32_t mixMode)
{
    // UB 工作区（字节）：布局峰值 58560 floats = 234240B = 228.75KB（上限 248KB）
    constexpr uint32_t workBuff = 248 * 1024;
    FftKernel<<<blockDim, workBuff, stream>>>(
        input, output, coeffs, workspace, tiling, ubMode, mixMode);
}

// ===========================================================================
// 转置 kernel（64×64 SIMT，独立 launch；AIC 空转仅 AIV 工作）。
// 与 1D FFT kernel 分开 launch：转置用 SIMT（UB 上限 216KB），
// 1D FFT 引擎需 248KB UB——两者 UB 预算不同，不能融合成单 kernel。
// ===========================================================================
CATLASS_GLOBAL __mix__(1, 2) void FftTransposeKernel(
    GM_ADDR input, GM_ADDR output, GM_ADDR tiling, int32_t vecCoreNum)
{
    RunTransposeStage(input, output, tiling, vecCoreNum);
}

void FftTransposeKernelLaunch(
    GM_ADDR input, GM_ADDR output, GM_ADDR tiling,
    uint32_t blockDim, aclrtStream stream, int32_t vecCoreNum)
{
    // 转置 UB 峰值 ~195KB + SIMT 预留；SIMT 场景 UB 上限 216KB
    constexpr uint32_t workBuff = 216 * 1024;
    FftTransposeKernel<<<blockDim, workBuff, stream>>>(
        input, output, tiling, vecCoreNum);
}

} // namespace Fft