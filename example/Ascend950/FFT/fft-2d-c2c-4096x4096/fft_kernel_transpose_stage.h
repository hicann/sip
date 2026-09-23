// ===========================================================================
// 转置 stage 的 header 版本 —— 供多 TU 内联复用（避免跨编译单元设备符号链接）。
// 原实现见 fft_kernel_transpose.cpp；此处抽出的目的：
//   融合内核若要在一个 kernel 里调转置，而 mix 构建不支持跨 .cpp 链接普通
//   __aicore__ 函数（ld 只解析 __global__ 符号），因此把 stage 做成 inline，
//   每个 TU 各编译一份副本（CATLASS 头文件内联同款策略）。
// ===========================================================================

#ifndef FFT_KERNEL_TRANSPOSE_STAGE_H
#define FFT_KERNEL_TRANSPOSE_STAGE_H

#include "catlass/arch/arch.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/catlass.hpp"
#include "catlass/status.hpp"

#include "fft_tiling_def.h"
#include "fft_kernel_utils.h"

#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "simt_api/vector_functions.h"
#include "helper.hpp"

namespace Fft {

// 头文件内声明 AscendC 短名（原 .cpp 依赖的 using 不会自动进入头；本头被
// 多个设备 TU include，using-directive 是文件作用域，不会跨 TU 冲突）。
using namespace AscendC;
using namespace Catlass;

// S2/S4 消费读 L2 hint（profiling 轮）：转置的 GM 读上次全核写完、本次
// 整体回读（L2 写命中 54%、读命中 3-12%——128MB 工作集流式污染）。
// =1 给读侧 tensor 标 CACHE_MODE_PERSISTENT（地址高位 hint，L2 行为偏
// 保持），试图提高回读命中；读路径不影响正确性（hint 只影响缓存策略，
// SyncAll 已保证写可见）。若真机无收益可回 0。
// ★ 注意 devkit 3510 的 CacheMode 枚举只有 NORMAL/DISABLE/PERSISTENT
//   （无文档意义上的 "ENABLE"；PERSISTENT=0b010 是最强的保持提示）。
#define TR_GM_READ_L2_PERSISTENT 1

// ─── 64×64 复 tile 几何（读侧也上 512B）：读/写侧单块均 512B。───
// tile 语义：源 = TR_TILE_R 行 × TR_TILE_C 复列；转置后 = TR_TILE_C 行 ×
// TR_TILE_R 复列。square tile（C==R==64），读/写侧每行 512B。
constexpr uint32_t TR_TILE_C = 64;                          // 复列（源列数 = 目标行数）
constexpr uint32_t TR_TILE_R = 64;                          // 复行（源行数 = 目标列数）
constexpr uint32_t TR_W_IN  = TR_TILE_C * 2;                // 128 floats/源行（读块长 512B）
constexpr uint32_t TR_W_OUT = TR_TILE_R * 2;                // 128 floats/目标行（写块长 512B）
constexpr uint32_t TR_TILE_FLOATS = TR_TILE_R * TR_W_IN;    // 8192 floats/tile
// pad 修正（profiling 轮）：v2 的 pad=2 复（行距 68 floats）在相位 2 读有
// 2-way subbank 冲突——(2tx+ty) mod 4 只取 2 值。复元素 8B 对齐按 float4
// bank 交织，需行距 ≡ 2 mod 4 复（=4 mod 8 float）才使 2tx+ty mod 4 遍历
// 全部 4 值。pad=3 复 → 行距 67 复=134 floats：134 mod 8 = 6，(2tx+ty) mod 4
// 取 {0,1,2,3} 全值，相位 2 无冲突。
constexpr uint32_t TR_PAD_COMPLEX = TR_TILE_C + 3;            // 67 复行距（pad 3 复消冲突）
constexpr uint32_t TR_SHARED_STRIDE = TR_PAD_COMPLEX * 2;   // 134 floats 行距
constexpr uint32_t TR_TILES_PER_AIV = 1;                    // 每 vf_call 处理 1 个 64×64 tile
constexpr uint32_t TR_TILE_HALF = TR_TILE_R / 2;            // 32（square：行列半区同为 32）
// 线程数：1024 = 32×32（tx,ty 各为半区 0..31，每线程覆盖 2×2=4 复元素）。
constexpr uint32_t TR_THREADS = TR_TILE_HALF * TR_TILE_HALF; // 1024
constexpr uint32_t TR_SHARED_FLOATS = TR_TILE_R * TR_SHARED_STRIDE;  // 64*134 = 8576

// ─── 双深 buffer 布局（每深 = src(1 tile) + dst(1 tile) + shared）───
constexpr uint32_t TR_GROUP_FLOATS = TR_TILE_FLOATS;        // 8192
constexpr uint32_t TR_BUF_FLOATS = 2 * TR_GROUP_FLOATS + TR_SHARED_FLOATS; // 24960
// UB 容量核对（每 AIV 216KB=55296 floats，双深 2×24960=49920 ≈ 195KB ✓ 余量 21KB）

// 64×64 复 tile 两段式 padded 转置（读/写侧均 512B，square tile）。
// 1024 线程 = 32×32：线程 (tx, ty) 各为半区 0..31，每线程覆盖 2×2=4 复元素
// （行 {ty, ty+32} × 列 {tx, tx+32}），float2 成对访问，寄存器压力同旧版。
// Phase 1 读 src 行内连续、写 shared 行内连续；Phase 2 读 shared 转置（stride
// 跨行，pad 消冲突）、写 dst 行内连续（512B）。
__simt_vf__ __launch_bounds__(TR_THREADS) inline void simt_transpose_c2_64x64(
    __ubuf__ float* dst,        // 转置后：TR_TILE_C 行 × TR_W_OUT floats（8192）
    const __ubuf__ float* src,  // 源：TR_TILE_R 行 × TR_W_IN floats（8192）
    __ubuf__ float* sharedTile) // padded：TR_TILE_R 行 × TR_SHARED_STRIDE floats
{
    uint32_t ty = threadIdx.y;   // 0..31 半区行
    uint32_t tx = threadIdx.x;   // 0..31 半区列

    const __ubuf__ float2* src2 = (const __ubuf__ float2*)src;
    __ubuf__ float2* shared2 = (__ubuf__ float2*)sharedTile;
    __ubuf__ float2* dst2 = (__ubuf__ float2*)dst;

    // Phase 1: 读 src 4 元素 → 写 padded shared（行内连续）
    shared2[(ty * TR_SHARED_STRIDE + tx * 2) >> 1] =
        src2[(ty * TR_W_IN + tx * 2) >> 1];
    shared2[(ty * TR_SHARED_STRIDE + (tx + TR_TILE_HALF) * 2) >> 1] =
        src2[(ty * TR_W_IN + (tx + TR_TILE_HALF) * 2) >> 1];
    shared2[((ty + TR_TILE_HALF) * TR_SHARED_STRIDE + tx * 2) >> 1] =
        src2[((ty + TR_TILE_HALF) * TR_W_IN + tx * 2) >> 1];
    shared2[((ty + TR_TILE_HALF) * TR_SHARED_STRIDE + (tx + TR_TILE_HALF) * 2) >> 1] =
        src2[((ty + TR_TILE_HALF) * TR_W_IN + (tx + TR_TILE_HALF) * 2) >> 1];

    asc_syncthreads();

    // Phase 2: 读 shared 转置（stride 跨行，pad 消冲突）→ 写 dst（行内连续 512B）
    dst2[(ty * TR_W_OUT + tx * 2) >> 1] =
        shared2[(tx * TR_SHARED_STRIDE + ty * 2) >> 1];
    dst2[(ty * TR_W_OUT + (tx + TR_TILE_HALF) * 2) >> 1] =
        shared2[((tx + TR_TILE_HALF) * TR_SHARED_STRIDE + ty * 2) >> 1];
    dst2[((ty + TR_TILE_HALF) * TR_W_OUT + tx * 2) >> 1] =
        shared2[(tx * TR_SHARED_STRIDE + (ty + TR_TILE_HALF) * 2) >> 1];
    dst2[((ty + TR_TILE_HALF) * TR_W_OUT + (tx + TR_TILE_HALF) * 2) >> 1] =
        shared2[((tx + TR_TILE_HALF) * TR_SHARED_STRIDE + (ty + TR_TILE_HALF) * 2) >> 1];
}

// 转置 stage（原 FftTransposeKernel 主体，AIC 空转，仅 AIV 工作）。
__aicore__ inline void RunTransposeStage(
    GM_ADDR input, GM_ADDR output, GM_ADDR tiling, int32_t vecCoreNumIn)
{
    // AIC 无工作：转置是纯 AIV 任务
    if ASCEND_IS_AIC {
        AscendC::PipeBarrier<PIPE_ALL>();
        return;
    }

    if ASCEND_IS_AIV {
        __gm__ FftTilingData* td = (__gm__ FftTilingData*)tiling;
        const int64_t fftN = td->inputParams.signalLen[0];     // 4096
        const int64_t rowFloats = fftN * 2;                    // 8192
        const uint32_t vecCoreNum = static_cast<uint32_t>(vecCoreNumIn);

        uint32_t vecIdx = AscendC::GetBlockIdx();   // mix: 2*blockIdx + subIdx (0..55)
        if (vecIdx >= vecCoreNum) {
            AscendC::PipeBarrier<PIPE_ALL>();
            return;
        }

        Arch::Resource<FftArchTag> resource;

        __gm__ float* gmIn = (__gm__ float*)(input);
        __gm__ float* gmOut = (__gm__ float*)(output);

        constexpr uint32_t TPD = static_cast<uint32_t>(4096 / TR_TILE_C);  // 128 列块
        constexpr uint32_t TPR = static_cast<uint32_t>(4096 / TR_TILE_R);  // 64 行块
        const uint32_t totalTiles = TPR * TPD;                            // 8192

        uint32_t tilesPerAiv = totalTiles / vecCoreNum;
        uint32_t tilesRemain = totalTiles % vecCoreNum;
        uint32_t myStart = vecIdx * tilesPerAiv + (vecIdx < tilesRemain ? vecIdx : tilesRemain);
        uint32_t myCount = tilesPerAiv + (vecIdx < tilesRemain ? 1 : 0);

        uint32_t numGroups = (myCount + TR_TILES_PER_AIV - 1) / TR_TILES_PER_AIV;

        LocalTensor<float> srcTensor[2];
        LocalTensor<float> dstTensor[2];
        __ubuf__ float* srcAddr[2];
        __ubuf__ float* dstAddr[2];
        __ubuf__ float* shAddr[2];
        for (uint32_t b = 0; b < 2; b++) {
            srcTensor[b] = resource.ubBuf.template GetBufferByByte<float>(
                static_cast<int64_t>(b) * TR_BUF_FLOATS * sizeof(float));
            dstTensor[b] = resource.ubBuf.template GetBufferByByte<float>(
                (static_cast<int64_t>(b) * TR_BUF_FLOATS + TR_GROUP_FLOATS) * sizeof(float));
            srcAddr[b] = (__ubuf__ float*)srcTensor[b].GetPhyAddr();
            dstAddr[b] = (__ubuf__ float*)dstTensor[b].GetPhyAddr();
            shAddr[b]  = (__ubuf__ float*)resource.ubBuf.template
                GetBufferByByte<float>((static_cast<int64_t>(b) * TR_BUF_FLOATS
                    + 2 * TR_GROUP_FLOATS) * sizeof(float)).GetPhyAddr();
        }

        constexpr event_t EVT_M2_V = EVENT_ID0;
        constexpr event_t EVT_V_M3 = EVENT_ID1;
        constexpr event_t EVT_M3_M2 = EVENT_ID2;

        for (uint32_t g = 0; g < numGroups; g++) {
            uint32_t t0 = g * TR_TILES_PER_AIV;               // 组首 tile 序号
            uint32_t tilesInGroup = myCount - t0;
            if (tilesInGroup > TR_TILES_PER_AIV) {
                tilesInGroup = TR_TILES_PER_AIV;
            }
            uint32_t bufIdx = g & 1u;

            // ── MTE2 stage（前置）: 读组 g（GM → src 区，与 SIMT(g-1) 并行）──
            for (uint32_t k = 0; k < tilesInGroup; k++) {
                uint32_t tile = myStart + t0 + k;
                uint32_t tr = tile / TPD;
                uint32_t tc = tile - tr * TPD;

                GlobalTensor<float> gmSrc;
                int64_t rowBase = static_cast<int64_t>(tr * TR_TILE_R) * rowFloats
                                + static_cast<int64_t>(tc * TR_W_IN);
                gmSrc.SetGlobalBuffer(gmIn + rowBase);
#if TR_GM_READ_L2_PERSISTENT == 1
                // 消费读 L2 保持提示（见文件头 TR_GM_READ_L2_PERSISTENT）
                gmSrc.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_PERSISTENT);
#endif

                LocalTensor<float> sub = resource.ubBuf.template GetBufferByByte<float>(
                    (static_cast<int64_t>(bufIdx) * TR_BUF_FLOATS
                        + static_cast<int64_t>(k) * TR_TILE_FLOATS) * sizeof(float));
                DataCopyParams cp;
                cp.blockCount = TR_TILE_R;
                cp.blockLen = TR_W_IN * sizeof(float) / 32;   // 16 (512B)
                cp.srcGap = static_cast<uint16_t>((rowFloats - TR_W_IN) * sizeof(float) / 32);
                cp.dstGap = 0;
                DataCopy(sub, gmSrc, cp);
            }
            AscendC::SetFlag<HardEvent::MTE2_V>(EVT_M2_V);
            AscendC::WaitFlag<HardEvent::MTE2_V>(EVT_M2_V);

            // ── MTE3 stage: 排空组 g-1（dst 区 → GM）──
            if (g >= 1) {
                uint32_t p0 = (g - 1) * TR_TILES_PER_AIV;
                uint32_t pCount = myCount - p0;
                if (pCount > TR_TILES_PER_AIV) {
                    pCount = TR_TILES_PER_AIV;
                }
                uint32_t prevIdx = 1u - bufIdx;   // 组 g-1 所在 buffer
                AscendC::WaitFlag<HardEvent::V_MTE3>(EVT_V_M3);
                for (uint32_t k = 0; k < pCount; k++) {
                    uint32_t dtile = myStart + p0 + k;
                    uint32_t dtr = dtile / TPD;
                    uint32_t dtc = dtile - dtr * TPD;

                    GlobalTensor<float> gmDst;
                    int64_t rowBase = static_cast<int64_t>(dtc * TR_TILE_C) * rowFloats
                                    + static_cast<int64_t>(dtr * TR_W_OUT);
                    gmDst.SetGlobalBuffer(gmOut + rowBase);

                    LocalTensor<float> sub = resource.ubBuf.template GetBufferByByte<float>(
                        (static_cast<int64_t>(prevIdx) * TR_BUF_FLOATS + TR_GROUP_FLOATS
                            + static_cast<int64_t>(k) * TR_TILE_FLOATS) * sizeof(float));
                    DataCopyParams cp;
                    cp.blockCount = TR_TILE_C;
                    cp.blockLen = TR_W_OUT * sizeof(float) / 32;   // 16 (512B)
                    cp.srcGap = 0;
                    cp.dstGap = static_cast<uint16_t>((rowFloats - TR_W_OUT) * sizeof(float) / 32);
                    DataCopy(gmDst, sub, cp);
                }
                AscendC::SetFlag<HardEvent::MTE3_MTE2>(EVT_M3_M2);
                AscendC::WaitFlag<HardEvent::MTE3_MTE2>(EVT_M3_M2);
            }

            // ── V stage: 转置组 g（1 tile × 64×64 一次 SIMT，1024 线程）──
            {
                asc_vf_call<simt_transpose_c2_64x64>(
                    dim3(TR_TILE_HALF, TR_TILE_HALF, 1),
                    dstAddr[bufIdx], srcAddr[bufIdx], shAddr[bufIdx]);
                AscendC::DataSyncBarrier<AscendC::MemDsbT::UB>();
            }
            AscendC::SetFlag<HardEvent::V_MTE3>(EVT_V_M3);
        }

        // ── Tail: 排空最后一组 ──
        {
            uint32_t p0 = (numGroups - 1) * TR_TILES_PER_AIV;
            uint32_t pCount = myCount - p0;
            if (pCount > TR_TILES_PER_AIV) {
                pCount = TR_TILES_PER_AIV;
            }
            uint32_t pBuf = (numGroups - 1) & 1u;
            AscendC::WaitFlag<HardEvent::V_MTE3>(EVT_V_M3);
            for (uint32_t k = 0; k < pCount; k++) {
                uint32_t dtile = myStart + p0 + k;
                uint32_t dtr = dtile / TPD;
                uint32_t dtc = dtile - dtr * TPD;

                GlobalTensor<float> gmDst;
                int64_t rowBase = static_cast<int64_t>(dtc * TR_TILE_C) * rowFloats
                                + static_cast<int64_t>(dtr * TR_W_OUT);
                gmDst.SetGlobalBuffer(gmOut + rowBase);

                LocalTensor<float> sub = resource.ubBuf.template GetBufferByByte<float>(
                    (static_cast<int64_t>(pBuf) * TR_BUF_FLOATS + TR_GROUP_FLOATS
                        + static_cast<int64_t>(k) * TR_TILE_FLOATS) * sizeof(float));
                DataCopyParams cp;
                cp.blockCount = TR_TILE_C;
                cp.blockLen = TR_W_OUT * sizeof(float) / 32;
                cp.srcGap = 0;
                cp.dstGap = static_cast<uint16_t>((rowFloats - TR_W_OUT) * sizeof(float) / 32);
                DataCopy(gmDst, sub, cp);
            }
            AscendC::SetFlag<HardEvent::MTE3_MTE2>(EVT_M3_M2);
            AscendC::WaitFlag<HardEvent::MTE3_MTE2>(EVT_M3_M2);
        }

        AscendC::PipeBarrier<PIPE_MTE3>();
    }

    // 与配对核 SyncAll 前必须把本核流水排空（ex82 同款前置）
    AscendC::PipeBarrier<PIPE_ALL>();
}

} // namespace Fft

#endif // FFT_KERNEL_TRANSPOSE_STAGE_H