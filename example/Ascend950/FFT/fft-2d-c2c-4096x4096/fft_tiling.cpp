#include "fft_tiling.h"
#include <iostream>

// tiling 计算（stage 化，对齐概要设计 Part A §2.3）：按 decomp 的 stage 清单填
// FftTilingData 各族参数——mm 形状、coeffs 段偏移（按 stage 序排块）、workspace
// B 段偏移（每核区内，乒乓由 pairIdx 现算）；host 结构只收敛汇总量。
// 骨架跨序列不变；TODO 标记处为实例相关（stage 布局的具体取值）。
namespace Fft {

size_t GetCoeffsSize(const RadixDecomp& decomp)
{
    // coeffs 布局（约定：按 stage 序排块，T 系殿后）：
    //   [W 系（按 stage 去重：同基 stage 共享一份；异基各一份）]
    //   [T 系（级间 twiddle，每级间一份，大小随级间 r_i×r_{i+1} 不同）]
    //   [T_expanded（向量展开表，可选）]
    // final（目标形态）：W16 32×32=1024 floats（同基去重一条）
    //   T0 8192（Butterfly 尾随）+ T1 4096 + T2 256 floats → 合计 13568 floats = 54272 B（T3/T4/T5 固化不占）
    int64_t seenRadix[FFT_MAX_STAGE] = {0};
    int32_t seenN = 0;
    int64_t wTotal = 0;
    for (int32_t i = 0; i < decomp.stageCount; i++) {
        if (decomp.stage[i].type != FftStageType::Gemm) { continue; }
        int64_t r = decomp.stage[i].radix;
        bool dup = false;
        for (int32_t j = 0; j < seenN; j++) {
            if (seenRadix[j] == r) { dup = true; break; }
        }
        if (!dup) {
            seenRadix[seenN++] = r;
            wTotal += (2 * r) * (2 * r);   // W 实展开 (2r)² floats/份（同基去重）
        }
    }
    int64_t tTotal = 0;
    for (int32_t i = 0; i < decomp.GetTMatrixCount(); i++) {
        // T0（Butterfly 尾随）紧凑 4096（kernel 只消费 k0=1 流）；T2（最后一张 Gemm 级间 T，接 P6
        //   连续蝶形）全展开 8192（按 P6 输出平面排布，消除步① T2 侧 Gather）；T1 保持数学口径
        tTotal += (i == 0) ? T0_COMPACT_FLOATS
                : (i == decomp.stageCount - 2) ? T2_EXPANDED_FLOATS : decomp.GetTFloats(i);
    }
    int64_t tExp = 896;   // P6 radix-2 蝶形 SIMD 展开表（7 twiddle × 64 份 × Re/Im 两段 = 896 floats）
    int64_t gatherTable = GATHER_TABLE_FLOATS;   // 步⑤ SIMD gather 偏移表（coeffs 末尾；64 uint32 = 64 float 槽，256B）
    int64_t p6t2Tables = P6T2_TABLE_FLOATS;      // 步① T2 复乘 SIMD gather 表（coeffs 末尾 GATHER_TABLE 之后；128 uint32 = 128 float 槽，512B）
    int64_t p4Tables = P4_TABLE_FLOATS;          // P4 twiddle SIMD gather 表（coeffs 末尾 P6T2 之后；128 uint32 = 128 float 槽，512B）
    return static_cast<size_t>((wTotal + tTotal + tExp + gatherTable + p6t2Tables + p4Tables) * sizeof(float));
}

size_t GetWorkspaceSize(const RadixDecomp& decomp, int64_t batch, int32_t cubeCoreNum)
{
    // 每核 workspace 区（99 例程模型，双信号列拼接 SPLIT_N；乒乓复用模型）：
    //   B0 单份 = mmK0×mmN0 = 32×512 = 16384 floats（64KB，含 2 信号列）；B1 同 16384
    // 【乒乓复用】B0/B1 各只需 2 份（pair 奇偶交替槽位）——flag 时序保证跨 group 复用：
    //   AIV 写 B0[pair+2] 前已完成 P4 批 Wait(G0[pair])（AIC 读完 B0[pair]）；
    //   AIV 写 B1[pair+2] 前已完成 P6 批 Wait(G1[pair])（AIC 读完 B1[pair]）。
    //   → per core = B0×2 + B1×2 = 4 × 16384 floats = 256KB（与 pair 数、核数无关）
    (void)decomp;
    (void)batch;
    (void)cubeCoreNum;
    int64_t perCore = static_cast<int64_t>(4) * 16384 * static_cast<int64_t>(sizeof(float));
    return static_cast<size_t>(cubeCoreNum * perCore);
}

void ComputeTiling(FftHostTilingData& tiling, const RadixDecomp& decomp, int64_t fftN,
                   int64_t batch, int32_t direction, int32_t cubeCoreNum)
{
    tiling.batch = batch;
    tiling.fftN = fftN;
    tiling.direction = direction;

    tiling.cubeCoreNum = cubeCoreNum;
    tiling.totalRows = static_cast<int32_t>(batch);
    tiling.stageCount = decomp.stageCount;
    tiling.gemmCount = decomp.GetGemmCount();

    // 行划分：按 pair（2 信号）分核，余数核多 2 行（跨序列不变）
    int32_t totalPairs = tiling.totalRows / 2;
    int32_t pairsPerCore = CeilDiv(totalPairs, cubeCoreNum);

    int32_t fullPairs = pairsPerCore * cubeCoreNum;
    if (fullPairs > totalPairs) {
        tiling.rowsPerCore = (pairsPerCore - 1) * 2;
        tiling.rowRemainder = totalPairs - (pairsPerCore - 1) * cubeCoreNum;
    } else {
        tiling.rowsPerCore = pairsPerCore * 2;
        tiling.rowRemainder = 0;
    }

    tiling.coeffsSize = GetCoeffsSize(decomp);
    // wsRegionBytes：每核 B 区 = B0/B1 各 2 份乒乓槽（pair 奇偶）× 16384 floats（乒乓复用模型，与 pair 数无关）
    tiling.wsRegionBytes = static_cast<int64_t>(4) * 16384 * static_cast<int64_t>(sizeof(float));
    tiling.workspaceSize = GetWorkspaceSize(decomp, batch, cubeCoreNum);
}

// FillDeviceTiling：stage 清单 + 汇总量 → device tiling（骨架跨序列不变；
// 族参数的 mm/偏移 TODO 由实例按 stage 布局填）
void FillDeviceTiling(const FftHostTilingData& host, const RadixDecomp& decomp,
                      FftTilingData& dev)
{
    // —— 信号描述（本工程 1D 定制形态：信号轴 = 尾轴；ND/任意轴按 brief design
    //    §2.3 的 inputShape/signalAxis/signalLen 填写）——
    dev.inputParams.inputRank = 2;   // TODO: [batch, fftLen]（ND/任意轴时按实际 rank）
    dev.inputParams.inputShape[0] = host.batch;
    dev.inputParams.inputShape[1] = host.fftN;
    dev.inputParams.signalDims = 1;
    dev.inputParams.signalAxis[0] = 1;      // 尾轴
    dev.inputParams.signalLen[0] = host.fftN;
    dev.inputParams.direction = host.direction;
    dev.inputParams.ubMode = 0;             // 默认 GM roundtrip（p1 kernel 未用 ubMode 分支；host 传参扩展见 fft_main）
    dev.inputParams.tExpandedOffset = 0;    // TODO: 有展开表时填（对应 plan 的 T_expanded 段）

    // —— stage 清单：type/radix 来自 decomp；族参数（mm/偏移）按布局填 ——
    dev.inputParams.stageCount = decomp.stageCount;
    int64_t coeffFloats = 0;      // coeffs W 系段 floats 偏移（stage 序 + 同基去重，与 GetCoeffsSize 同源）
    int64_t wTotalFloats = 0;     // W 系总 floats（供 T 系定位；去重口径与 GetCoeffsSize 一致）
    {
        int64_t seenRadix[FFT_MAX_STAGE] = {0};
        int32_t seenN = 0;
        for (int32_t i = 0; i < decomp.stageCount; i++) {
            if (decomp.stage[i].type != FftStageType::Gemm) { continue; }
            int64_t r = decomp.stage[i].radix;
            bool dup = false;
            for (int32_t j = 0; j < seenN; j++) {
                if (seenRadix[j] == r) { dup = true; break; }
            }
            if (!dup) { seenRadix[seenN++] = r; wTotalFloats += (2 * r) * (2 * r); }
        }
    }
    int32_t gemmIdx = 0;
    int64_t tFloats = 0;          // T 系已排 floats（级间序，W 系之后）
    for (int32_t i = 0; i < decomp.stageCount; i++) {
        const FftStageDesc& src = decomp.stage[i];
        FftStageDesc& dst = dev.inputParams.stage[i];
        dst.type = src.type;
        dst.radix = src.radix;

        if (src.type == FftStageType::Gemm) {
            // mm 形状（99 例程口径，ref_fft_math §5）：M=K=2×radix；N = 2×N_fft/r
            //   （**双信号列拼接 + SPLIT_N**：B/C 列 = [信号0 组列 | 信号1 组列]，AIV0/AIV1 各写/读半列）
            dst.gemm.mmM = decomp.GetMmM(gemmIdx);
            dst.gemm.mmK = decomp.GetMmK(gemmIdx);
            dst.gemm.mmN = 2 * decomp.GetMmN(gemmIdx);
            // W 段偏移（floats → 字节；同基复用时指向同一块首）
            bool dupW = false;
            for (int32_t j = 0; j < i; j++) {
                if (dev.inputParams.stage[j].type == FftStageType::Gemm &&
                    dev.inputParams.stage[j].radix == src.radix) {
                    dst.gemm.opOffset = dev.inputParams.stage[j].gemm.opOffset;
                    dupW = true;
                    break;
                }
            }
            if (!dupW) {
                dst.gemm.opOffset = coeffFloats * static_cast<int64_t>(sizeof(float));
                coeffFloats += (2 * static_cast<int64_t>(src.radix)) * (2 * static_cast<int64_t>(src.radix));
            }
            // T 表偏移（级间 twiddle：T_i = W_4096^(m·k_i)，W 系之后逐级间排；末级无）
            if (i < decomp.stageCount - 1) {
                dst.gemm.tOffset = (wTotalFloats + tFloats) * static_cast<int64_t>(sizeof(float));
                // T2（最后一张 Gemm 级间 T，接 P6 连续蝶形）全展开 8192（与 GetCoeffsSize 同源）
                tFloats += (i == decomp.stageCount - 2) ? T2_EXPANDED_FLOATS : decomp.GetTFloats(i);
            } else {
                dst.gemm.tOffset = 0;
            }
            // B 段基址（每核区内，乒乓复用模型）：B0/B1 区按 gemm 序排，各留 2 份乒乓槽（pair 奇偶）× 16384 floats；
            //   pair 内信号列偏移（subIdx×单信号列宽）与 pair 槽位（(pair&1)×16384）由 kernel 现算
            dst.gemm.bOffset = static_cast<int64_t>(gemmIdx) * 2 * 16384
                             * static_cast<int64_t>(sizeof(float));
            gemmIdx++;
        } else {
            // Butterfly：非末级分配尾随 twiddle（T0，stage[0]）；末级（final stage[3] = L3/L4/L5 归并）
            //   T3/T4/T5 固化代码/寄存器，不占 coeffs（详A §2.2.6 / 详B §3.1）；末级 twiddleOffset 指向
            //   T_expanded（P6 radix-2 蝶形 SIMD 展开表，W 系 + 全部 T 系之后）
            if (i < decomp.stageCount - 1) {
                dst.butterfly.twiddleOffset = (wTotalFloats + tFloats) * static_cast<int64_t>(sizeof(float));
                tFloats += (i == 0) ? T0_COMPACT_FLOATS : decomp.GetTFloats(i);   // T0 紧凑（k0=1 流，4096）
            } else {
                dst.butterfly.twiddleOffset = (wTotalFloats + tFloats) * static_cast<int64_t>(sizeof(float));
            }
            dst.butterfly.layers = 1;
            dst.butterfly.phaseOffset = 0;
        }
    }

    // GATHER_TABLE（步⑤ SIMD gather 偏移表，coeffs 末尾）：T_expanded（896 floats）之后，
    //   与 GetCoeffsSize 的累加顺序一致（W 系 → T 系 → T_expanded → GATHER_TABLE）
    dev.inputParams.gatherTableOffset = (wTotalFloats + tFloats + 896)
                                      * static_cast<int64_t>(sizeof(float));
    // P6T2 表（步① T2 复乘 SIMD gather 表，coeffs 末尾 GATHER_TABLE 之后）：与 GetCoeffsSize 累加顺序一致
    dev.inputParams.p6t2TableOffset = dev.inputParams.gatherTableOffset
                                    + GATHER_TABLE_FLOATS * static_cast<int64_t>(sizeof(float));
    // P4 表（P4 twiddle SIMD gather 表，coeffs 末尾 P6T2 之后）：与 GetCoeffsSize 累加顺序一致
    dev.inputParams.p4TableOffset = dev.inputParams.p6t2TableOffset
                                  + P6T2_TABLE_FLOATS * static_cast<int64_t>(sizeof(float));

    // —— 多核侧 ——
    dev.multiCoreParams.coreNum = host.cubeCoreNum;
    dev.multiCoreParams.totalRows = host.totalRows;
    dev.multiCoreParams.rowsPerCore = host.rowsPerCore;
    dev.multiCoreParams.rowRemainder = host.rowRemainder;

    uint32_t offset = 0;
    for (int32_t i = 0; i < host.cubeCoreNum; i++) {
        dev.multiCoreParams.rowStartIdx[i] = offset;
        int32_t rows = host.rowsPerCore;
        if (i < host.rowRemainder) {
            rows = host.rowsPerCore + 2;
        }
        offset += static_cast<uint32_t>(rows);
    }
    dev.multiCoreParams.rowStartIdx[host.cubeCoreNum] = static_cast<uint32_t>(host.totalRows);
    dev.multiCoreParams.wsRegionBytes = host.wsRegionBytes;
}

} // namespace Fft