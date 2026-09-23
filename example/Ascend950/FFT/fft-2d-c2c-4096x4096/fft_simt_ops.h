#ifndef FFT_SIMT_OPS_H
#define FFT_SIMT_OPS_H

// AIV 侧 SIMD 算子（全 SIMD 路径，无 SIMT 兜底）：每个 AIV phase / P6 各步一个。
// 索引公式与 host 侧表构造（fft_twiddle.h）逐元素一致。
#include "kernel_operator.h"
#include "fft_tiling_def.h"

// ══ SIMD：P1 首级 radix-2 蝶形 + 尾随 twiddle T0（交织复行 → T0 复交织，S 区原地）══
//   a = x[m]、b = x[2048+m]（m∈[0,2048)，跨块同位元素）
//   T0[m]        = a + b（k0=0 流，W^0=1 免乘）
//   T0[2048+m]   = (a-b)·W_4096^m（k0=1 流，读 T0tw 紧凑表：Re tw[m]、Im tw[2048+m]）
// 每轮 64 复元素：LoadAlign 连续 128 浮点 → DeInterleave 分离 Re/Im → 复乘 → Interleave 写回
__simd_vf__ inline void simd_p1_butterfly(
    __ubuf__ float* dst, __ubuf__ float* src, __ubuf__ float* tw)
{
    constexpr uint32_t P1_VEC = 64;            // SIMD 向量宽度（float 64 lane）
    constexpr uint32_t P1_GROUPS = 32;         // 2048 m / 64 = 32 轮
    AscendC::Reg::RegTensor<float> aReg0, aReg1, bReg0, bReg1;
    AscendC::Reg::RegTensor<float> aRe, aIm, bRe, bIm;
    AscendC::Reg::RegTensor<float> twRe, twIm;
    AscendC::Reg::RegTensor<float> s0Re, s0Im, uRe, uIm, tRe1, tIm1, s1Re, s1Im;
    AscendC::Reg::RegTensor<float> out0, out1, out2, out3;
    AscendC::Reg::MaskReg mask = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();

    for (uint32_t g = 0; g < P1_GROUPS; g++) {
        // a = src 前半（复 m = g*64..g*64+63，交织 128 floats），b = src 后半（+4096 floats）
        AscendC::Reg::LoadAlign(aReg0, src + g * P1_VEC * 2);
        AscendC::Reg::LoadAlign(aReg1, src + g * P1_VEC * 2 + P1_VEC);
        AscendC::Reg::LoadAlign(bReg0, src + 4096 + g * P1_VEC * 2);
        AscendC::Reg::LoadAlign(bReg1, src + 4096 + g * P1_VEC * 2 + P1_VEC);
        AscendC::Reg::DeInterleave<float>(aRe, aIm, aReg0, aReg1);
        AscendC::Reg::DeInterleave<float>(bRe, bIm, bReg0, bReg1);
        AscendC::Reg::LoadAlign(twRe, tw + g * P1_VEC);
        AscendC::Reg::LoadAlign(twIm, tw + 2048 + g * P1_VEC);
        // k0=0 流：s0 = a + b（免乘）
        AscendC::Reg::Add(s0Re, aRe, bRe, mask);
        AscendC::Reg::Add(s0Im, aIm, bIm, mask);
        // k0=1 流：s1 = (a-b)·W_4096^m
        AscendC::Reg::Sub(uRe, aRe, bRe, mask);
        AscendC::Reg::Sub(uIm, aIm, bIm, mask);
        AscendC::Reg::Mul(tRe1, uRe, twRe, mask);
        AscendC::Reg::Mul(tIm1, uIm, twIm, mask);
        AscendC::Reg::Sub(s1Re, tRe1, tIm1, mask);          // s1Re = uRe·twRe - uIm·twIm
        AscendC::Reg::Mul(tRe1, uRe, twIm, mask);
        AscendC::Reg::Mul(tIm1, uIm, twRe, mask);
        AscendC::Reg::Add(s1Im, tRe1, tIm1, mask);          // s1Im = uRe·twIm + uIm·twRe
        // 写回复交织：k0=0 块 dst[0,2048) 复、k0=1 块 dst[2048,4096) 复
        AscendC::Reg::Interleave<float>(out0, out1, s0Re, s0Im);
        AscendC::Reg::StoreAlign(dst + g * P1_VEC * 2, out0, mask);
        AscendC::Reg::StoreAlign(dst + g * P1_VEC * 2 + P1_VEC, out1, mask);
        AscendC::Reg::Interleave<float>(out2, out3, s1Re, s1Im);
        AscendC::Reg::StoreAlign(dst + 4096 + g * P1_VEC * 2, out2, mask);
        AscendC::Reg::StoreAlign(dst + 4096 + g * P1_VEC * 2 + P1_VEC, out3, mask);
    }
}

// ══ SIMD：P2 T0 重排 gather（T0 复交织 → B0 单信号列子块 32×256 紧凑）══
// 输出 i = rb·256 + c：rb = blk>>2、c = (blk&3)·64 + lane
//   j1 = rb&15、k0 = (blk&2)>>1、m2 = (blk&1)·64 + lane → T0 复索引 idx = k0·2048 + j1·128 + m2
// 配对：blk 与 blk+64 读同一 64 复连续源段 → 一次 LoadAlign 64 复 + DeInterleave 同时得 Re/Im，
//   re 写 Re 行 dst+blk·64、im 写 Im 行 dst+4096+blk·64（源只读一遍）。
__simd_vf__ inline void simd_gather_b0(
    __ubuf__ float* dst, __ubuf__ float* src)
{
    constexpr uint32_t P2_VEC = 64;               // SIMD 向量宽度
    constexpr uint32_t P2_PAIRS = 64;             // 64 对（每对 1 源段 → Re/Im 两行），B0 共 128 行
    AscendC::Reg::RegTensor<float> r0, r1, re, im;
    AscendC::Reg::MaskReg mask = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();

    for (uint32_t blk = 0; blk < P2_PAIRS; blk++) {
        // T0 源 float 偏移 = 2·idx0（idx0 = k0·2048 + j1·128 + (blk&1)·64），64 对齐连续段
        uint32_t srcOff = ((blk & 2) >> 1) * 4096 + ((blk >> 2) & 15) * 256 + (blk & 1) * 128;
        AscendC::Reg::LoadAlign(r0, src + srcOff);
        AscendC::Reg::LoadAlign(r1, src + srcOff + P2_VEC);
        AscendC::Reg::DeInterleave<float>(re, im, r0, r1);
        AscendC::Reg::StoreAlign(dst + blk * P2_VEC, re, mask);
        AscendC::Reg::StoreAlign(dst + P2_PAIRS * P2_VEC + blk * P2_VEC, im, mask);
    }
}

// ══ SIMD：P4 T1 复乘 + 重排（C0 半列紧凑 → GEMM#1 输入 B1 单信号列子块）══
// C0 经 SPLIT_N 落本 AIV UB 的是本信号 N/2 列紧凑子块（32×256，行 k1∈[0,32) =
//   Re k1∈[0,16) + Im k1∈[16,32)；列 (k0,m2) = k0·128+m2）。
// 输出 B1 紧凑 32×256（行 rb = j2 Re + (j2+16) Im，列 c = (k0,k1,m3)），按 blk·64+lane 分块：
//   rb = blk>>2；c = (blk&3)·64 + lane；j2 = rb&15；k0 = (blk&2)>>1；k1 = (blk&1)·8 + (lane>>3)；m3 = lane&7
//   m2 = j2·8 + m3；C0 源列 col = k0·128 + m2
// C0 行内偏移：src = baseC0(blk) + M_OFF[lane]（M_OFF[lane] = (lane>>3)·256 + (lane&7)）
//   baseC0(blk) = (blk&1)·2048 + (blk&2)·64 + ((blk>>2)&15)·8
// T1 索引：twIdx = baseTw(blk) + TW_OFF[lane]（TW_OFF[lane] = (lane&7)·16 + (lane>>3)）
//   baseTw(blk) = ((blk>>2)&15)·128 + (blk&1)·8；tw + 2048 为 Im 段
// blk 与 blk+64 读同一源段 → 一次 Gather 四路 + 一次复乘同时得 vr/vi（Gather 减半）。
__simd_vf__ inline void simd_twiddle_rotate_r4(
    __ubuf__ float* dst, __ubuf__ float* srcC0, __ubuf__ float* tw,
    __ubuf__ uint32_t* offTable, __ubuf__ uint32_t* twTable)
{
    constexpr uint32_t P4_VEC = 64;               // SIMD 向量宽度
    constexpr uint32_t P4_PAIRS = 64;             // 64 对（每对 1 源段 → Re/Im 两行），B1 共 128 行
    AscendC::Reg::RegTensor<float> reA, imA, tr, ti, vr, vi, tmp;
    AscendC::Reg::RegTensor<uint32_t> offReg, twOffReg, baseReg, idxReg;
    AscendC::Reg::MaskReg mask = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();

    AscendC::Reg::LoadAlign(offReg, offTable);     // M_OFF[lane]（P4_OFF 前半），循环外一次加载
    AscendC::Reg::LoadAlign(twOffReg, twTable);    // TW_OFF[lane]（P4_TW 后半），循环外一次加载

    for (uint32_t blk = 0; blk < P4_PAIRS; blk++) {
        uint32_t rb = blk >> 2;                    // Re 输出行 0..15（Im 行 = rb+16）
        uint32_t baseC0 = ((blk & 1) * 2048) + ((blk & 2) * 64) + ((rb & 15) * 8);
        uint32_t baseTw = ((rb & 15) * 128) + ((blk & 1) * 8);
        // C0 Re 行 k1：srcOff = baseC0 + M_OFF[lane]
        AscendC::Reg::Duplicate(baseReg, baseC0, mask);
        AscendC::Reg::Add(idxReg, baseReg, offReg, mask);
        AscendC::Reg::Gather(reA, srcC0, idxReg, mask);
        // C0 Im 行 k1+16：srcC0 + 4096
        AscendC::Reg::Duplicate(baseReg, baseC0 + 4096, mask);
        AscendC::Reg::Add(idxReg, baseReg, offReg, mask);
        AscendC::Reg::Gather(imA, srcC0, idxReg, mask);
        // T1 Re/Im：twIdx = baseTw + TW_OFF[lane]
        AscendC::Reg::Duplicate(baseReg, baseTw, mask);
        AscendC::Reg::Add(idxReg, baseReg, twOffReg, mask);
        AscendC::Reg::Gather(tr, tw, idxReg, mask);
        AscendC::Reg::Duplicate(baseReg, baseTw + 2048, mask);
        AscendC::Reg::Add(idxReg, baseReg, twOffReg, mask);
        AscendC::Reg::Gather(ti, tw, idxReg, mask);
        // 复乘：vr = reA·tr - imA·ti；vi = reA·ti + imA·tr
        AscendC::Reg::Mul(vr, reA, tr, mask);
        AscendC::Reg::Mul(vi, imA, ti, mask);
        AscendC::Reg::Sub(vr, vr, vi, mask);
        AscendC::Reg::Mul(vi, reA, ti, mask);
        AscendC::Reg::Mul(tmp, imA, tr, mask);
        AscendC::Reg::Add(vi, vi, tmp, mask);
        // vr 写 Re 行 dst+blk·64、vi 写 Im 行 dst+64·64+blk·64
        AscendC::Reg::StoreAlign(dst + blk * P4_VEC, vr, mask);
        AscendC::Reg::StoreAlign(dst + P4_PAIRS * P4_VEC + blk * P4_VEC, vi, mask);
    }
}

// ══ SIMD：P6 步① T2 复乘 + 实展开 → Re/Im 分离平面（C1 半列紧凑 → P6 平面 8×512）══
// C1 经 SPLIT_N 落本 AIV UB 的是本信号 N/2 列紧凑子块（32×256，行 k2∈[0,32) =
//   Re k2∈[0,16) + Im k2∈[16,32)；列 (k0,k1,m') = k0·128+k1·8+m'）。
// 逻辑复值 idx = k0·2048 + k1·128 + k2·8 + m'；w = C1[k2][(k0,k1,m')]·W_128^(m'·k2)
// 【DIT 位序输入】输出按位序槽位排布（行 bitrev3(m')、列段搬移 rp = k0+2·k1+32·k2），
//   随后的 DIT 蝶形（L5→L4→L3）输出自然序，步⑤ 直出无需 bitrev。
// 平面布局（SIMD 对齐）：idx = r·8 + m2（r = idx>>3，m2 = m'）→ dst[bitrev3(m2)·512 + rp]
// T2 用全展开表（T2Exp 8192，按输出物理列排布）：tr = T2Exp[m2·512+rp]，零 T2 侧 Gather；
//   C1 侧 Gather：srcOff = 512·blk + m2 + M_OFF'[lane]（M_OFF' = P6T2_OFF 前半 64 项）
__simd_vf__ inline void simd_p6_t2_expanded(
    __ubuf__ float* dstRe, __ubuf__ float* dstIm,
    __ubuf__ float* srcC1, __ubuf__ float* twExp,
    __ubuf__ uint32_t* offTable)
{
    constexpr uint32_t P6_VEC = 64;               // SIMD 向量宽度
    constexpr uint32_t P6_MROWS = 8;              // 平面行 m2 ∈ [0,8)
    constexpr uint32_t P6_RBLKS = 8;              // 物理列分块 blk ∈ [0,8)
    AscendC::Reg::RegTensor<float> reA, imA, tr, ti, vr, vi, tmp;
    AscendC::Reg::RegTensor<uint32_t> offReg, baseReg, baseImReg, idxReg;
    AscendC::Reg::MaskReg mask = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();

    AscendC::Reg::LoadAlign(offReg, offTable);     // M_OFF'[lane]，循环外一次加载

    for (uint32_t m2 = 0; m2 < P6_MROWS; m2++) {
        for (uint32_t blk = 0; blk < P6_RBLKS; blk++) {
            // T2Exp 连续 LoadAlign（零 Gather）：物理列 rp = blk·64+lane 连续
            AscendC::Reg::LoadAlign(tr, twExp + m2 * 512 + blk * P6_VEC);
            AscendC::Reg::LoadAlign(ti, twExp + 4096 + m2 * 512 + blk * P6_VEC);
            // C1 源偏移（blk = 物理块 = k2>>1）：srcOff = k2·256 + k0·128 + k1·8 + m2
            //   = 512·blk + m2 + M_OFF'[lane]（lane → k0=lane&1、k1=(lane>>1)&15、k2=2·blk+(lane>>5)）
            uint32_t base = 512 * blk + m2;
            AscendC::Reg::Duplicate(baseReg, base, mask);
            AscendC::Reg::Add(idxReg, baseReg, offReg, mask);
            AscendC::Reg::Gather(reA, srcC1, idxReg, mask);      // Re 行 k2
            AscendC::Reg::Duplicate(baseImReg, base + 4096, mask);
            AscendC::Reg::Add(idxReg, baseImReg, offReg, mask);
            AscendC::Reg::Gather(imA, srcC1, idxReg, mask);      // Im 行 k2+16
            // 复乘：vr = reA·tr - imA·ti；vi = reA·ti + imA·tr
            AscendC::Reg::Mul(vr, reA, tr, mask);
            AscendC::Reg::Mul(vi, imA, ti, mask);
            AscendC::Reg::Sub(vr, vr, vi, mask);
            AscendC::Reg::Mul(vi, reA, ti, mask);
            AscendC::Reg::Mul(tmp, imA, tr, mask);
            AscendC::Reg::Add(vi, vi, tmp, mask);
            // 写平面行 = bitrev3(m2)（位序输入，供随后的 DIT 蝶形输出自然序）
            uint32_t rw = ((m2 & 1) << 2) | (m2 & 2) | ((m2 & 4) >> 2);
            AscendC::Reg::StoreAlign(dstRe + rw * 512 + blk * P6_VEC, vr, mask);
            AscendC::Reg::StoreAlign(dstIm + rw * 512 + blk * P6_VEC, vi, mask);
        }
    }
}

// ══ SIMD：P6 步② L5 radix-2 蝶形（DIT 首级，8 行平面，4 相邻对 (a,a+1) × 512 列）══
// DIT 语义：W_2^0=1 → k5=0 流 x_a+x_b、k5=1 流 x_a-x_b（无 twiddle 纯加减）
__simd_vf__ inline void simd_radix2_l5(
    __ubuf__ float* re, __ubuf__ float* im)
{
    constexpr uint32_t P6_VEC = 64;          // SIMD 向量宽度（float）
    constexpr uint32_t P6_COLBLOCKS = 512 / P6_VEC;   // 8 列块
    AscendC::Reg::RegTensor<float> reA, imA, reB, imB, tRe, tIm;
    AscendC::Reg::MaskReg mask = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();

    for (uint16_t a = 0; a < 8; a += 2) {    // 相邻对起点 {0, 2, 4, 6}
        for (uint16_t blk = 0; blk < P6_COLBLOCKS; blk++) {
            uint16_t colOff = blk * P6_VEC;
            AscendC::Reg::LoadAlign(reA, re + a * 512 + colOff);
            AscendC::Reg::LoadAlign(imA, im + a * 512 + colOff);
            AscendC::Reg::LoadAlign(reB, re + (a + 1) * 512 + colOff);
            AscendC::Reg::LoadAlign(imB, im + (a + 1) * 512 + colOff);

            AscendC::Reg::Sub(tRe, reA, reB, mask);     // t = ar - br（k5=1 流）
            AscendC::Reg::Sub(tIm, imA, imB, mask);     // t = ai - bi
            AscendC::Reg::Add(reA, reA, reB, mask);     // a' = ar + br（k5=0 流）
            AscendC::Reg::Add(imA, imA, imB, mask);     // a' = ai + bi

            AscendC::Reg::StoreAlign(re + a * 512 + colOff, reA, mask);
            AscendC::Reg::StoreAlign(im + a * 512 + colOff, imA, mask);
            AscendC::Reg::StoreAlign(re + (a + 1) * 512 + colOff, tRe, mask);
            AscendC::Reg::StoreAlign(im + (a + 1) * 512 + colOff, tIm, mask);
        }
    }
}

// ══ SIMD：P6 步③ L4 radix-2 蝶形（8 行平面，4 对 a∈{0,1,4,5}、b=a+2（对距 2））══
// DIT 语义：t = x_b·W_4^{m4}；k4=0 流 x_a + t、k4=1 流 x_a - t，m4 = a&1
// twiddle 展开表（BuildP6Radix2TwiddleExpanded）L4 段在 L3 之后：Re off = 4·64 + m4·64；Im off = 448 + 4·64 + m4·64
__simd_vf__ inline void simd_radix2_l4(
    __ubuf__ float* re, __ubuf__ float* im, __ubuf__ float* twTable)
{
    constexpr uint32_t P6_VEC = 64;          // SIMD 向量宽度（float）
    constexpr uint32_t P6_COLBLOCKS = 512 / P6_VEC;   // 8 列块
    constexpr uint32_t P6_TW_RE = 448;       // 展开表 Re 段总长
    constexpr uint32_t P6_TW_L3 = 4 * P6_VEC;         // L3 段占 4 twiddle
    AscendC::Reg::RegTensor<float> reA, imA, reB, imB;
    AscendC::Reg::RegTensor<float> twRe, twIm, tRe, tIm, tmpRe, tmpIm;
    AscendC::Reg::MaskReg mask = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();

    for (uint16_t a = 0; a < 8; a += 4) {    // 组 0/1：a 起点 {0, 4}
        for (uint16_t m4 = 0; m4 < 2; m4++) {
            uint16_t i = a + m4;
            AscendC::Reg::LoadAlign(twRe, twTable + P6_TW_L3 + m4 * P6_VEC);
            AscendC::Reg::LoadAlign(twIm, twTable + P6_TW_RE + P6_TW_L3 + m4 * P6_VEC);
            for (uint16_t blk = 0; blk < P6_COLBLOCKS; blk++) {
                uint16_t colOff = blk * P6_VEC;
                AscendC::Reg::LoadAlign(reA, re + i * 512 + colOff);
                AscendC::Reg::LoadAlign(imA, im + i * 512 + colOff);
                AscendC::Reg::LoadAlign(reB, re + (i + 2) * 512 + colOff);
                AscendC::Reg::LoadAlign(imB, im + (i + 2) * 512 + colOff);

                // DIT：t = x_b·W_4^{m4}；reA/imA 原地更新为 x_a + t、reB/imB 更新为 x_a - t
                AscendC::Reg::Mul(tmpRe, reB, twRe, mask);  // br·wr
                AscendC::Reg::Mul(tmpIm, reB, twIm, mask);  // br·wi
                AscendC::Reg::Mul(tRe, imB, twIm, mask);    // bi·wi
                AscendC::Reg::Mul(tIm, imB, twRe, mask);    // bi·wr
                AscendC::Reg::Sub(tRe, tmpRe, tRe, mask);   // tRe = br·wr - bi·wi
                AscendC::Reg::Add(tIm, tmpIm, tIm, mask);   // tIm = br·wi + bi·wr
                AscendC::Reg::Sub(reB, reA, tRe, mask);     // b'Re = ar - tRe（k4=1 流）
                AscendC::Reg::Add(reA, reA, tRe, mask);     // a'Re = ar + tRe（k4=0 流）
                AscendC::Reg::Sub(imB, imA, tIm, mask);     // b'Im = ai - tIm
                AscendC::Reg::Add(imA, imA, tIm, mask);     // a'Im = ai + tIm

                AscendC::Reg::StoreAlign(re + i * 512 + colOff, reA, mask);
                AscendC::Reg::StoreAlign(im + i * 512 + colOff, imA, mask);
                AscendC::Reg::StoreAlign(re + (i + 2) * 512 + colOff, reB, mask);
                AscendC::Reg::StoreAlign(im + (i + 2) * 512 + colOff, imB, mask);
            }
        }
    }
}

// ══ SIMD：P6 步④ L3 radix-2 蝶形（8 行平面，4 对 (m2, m2+4) × 512 列）══
// DIT 语义：t = x_b·W_8^{m2}；k3=0 流 x_a + t、k3=1 流 x_a - t
// twiddle 展开表（BuildP6Radix2TwiddleExpanded）L3 段在最前：Re off = m2·64；Im off = 448 + m2·64
__simd_vf__ inline void simd_radix2_l3(
    __ubuf__ float* re, __ubuf__ float* im, __ubuf__ float* twTable)
{
    constexpr uint32_t P6_VEC = 64;          // SIMD 向量宽度（float）
    constexpr uint32_t P6_COLBLOCKS = 512 / P6_VEC;   // 8 列块
    constexpr uint32_t P6_TW_RE = 448;       // 展开表 Re 段总长（L3+L4+L5 各 64 份）
    AscendC::Reg::RegTensor<float> reA, imA, reB, imB;
    AscendC::Reg::RegTensor<float> twRe, twIm, tRe, tIm, tmpRe, tmpIm;
    AscendC::Reg::MaskReg mask = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();

    for (uint16_t m2 = 0; m2 < 4; m2++) {
        // 本对 twiddle：W_8^{m2}（Re 段 offset m2·64、Im 段 offset 448+m2·64）
        AscendC::Reg::LoadAlign(twRe, twTable + m2 * P6_VEC);
        AscendC::Reg::LoadAlign(twIm, twTable + P6_TW_RE + m2 * P6_VEC);
        for (uint16_t blk = 0; blk < P6_COLBLOCKS; blk++) {
            uint16_t colOff = blk * P6_VEC;
            AscendC::Reg::LoadAlign(reA, re + m2 * 512 + colOff);
            AscendC::Reg::LoadAlign(imA, im + m2 * 512 + colOff);
            AscendC::Reg::LoadAlign(reB, re + (m2 + 4) * 512 + colOff);
            AscendC::Reg::LoadAlign(imB, im + (m2 + 4) * 512 + colOff);

            // DIT：t = x_b·W_8^{m2}；reA/imA 原地更新为 x_a + t、reB/imB 更新为 x_a - t
            AscendC::Reg::Mul(tmpRe, reB, twRe, mask);  // br·wr
            AscendC::Reg::Mul(tmpIm, reB, twIm, mask);  // br·wi
            AscendC::Reg::Mul(tRe, imB, twIm, mask);    // bi·wi
            AscendC::Reg::Mul(tIm, imB, twRe, mask);    // bi·wr
            AscendC::Reg::Sub(tRe, tmpRe, tRe, mask);   // tRe = br·wr - bi·wi
            AscendC::Reg::Add(tIm, tmpIm, tIm, mask);   // tIm = br·wi + bi·wr
            AscendC::Reg::Sub(reB, reA, tRe, mask);     // b'Re = ar - tRe（k3=1 流）
            AscendC::Reg::Add(reA, reA, tRe, mask);     // a'Re = ar + tRe（k3=0 流）
            AscendC::Reg::Sub(imB, imA, tIm, mask);     // b'Im = ai - tIm
            AscendC::Reg::Add(imA, imA, tIm, mask);     // a'Im = ai + tIm

            AscendC::Reg::StoreAlign(re + m2 * 512 + colOff, reA, mask);
            AscendC::Reg::StoreAlign(im + m2 * 512 + colOff, imA, mask);
            AscendC::Reg::StoreAlign(re + (m2 + 4) * 512 + colOff, reB, mask);
            AscendC::Reg::StoreAlign(im + (m2 + 4) * 512 + colOff, imB, mask);
        }
    }
}

// ══ SIMD：P6 步⑤ 纯 interleave（P6 Re/Im 平面 → y 交织复行）══
// DIT 输出自然序：y[k] 平面位置 = 行 m = (k>>9)&7、列 r = k&511（无 bitrev 重排）；
// 逐行逐列块 LoadAlign Re/Im 64 → Interleave → StoreAlign 128 floats（零 Gather 零偏移表）。
__simd_vf__ inline void simd_interleave_planes(
    __ubuf__ float* dst, __ubuf__ float* srcRe, __ubuf__ float* srcIm)
{
    constexpr uint32_t P6_VEC = 64;          // SIMD 向量宽度（float）
    constexpr uint32_t P6_COLBLOCKS = 512 / P6_VEC;   // 8 列块
    AscendC::Reg::RegTensor<float> reA, imA, out0, out1;
    AscendC::Reg::MaskReg mask = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();

    for (uint32_t m = 0; m < 8; m++) {
        for (uint16_t blk = 0; blk < P6_COLBLOCKS; blk++) {
            uint16_t colOff = blk * P6_VEC;
            AscendC::Reg::LoadAlign(reA, srcRe + m * 512 + colOff);
            AscendC::Reg::LoadAlign(imA, srcIm + m * 512 + colOff);
            AscendC::Reg::Interleave<float>(out0, out1, reA, imA);
            AscendC::Reg::StoreAlign(dst + m * 1024 + blk * 128, out0, mask);
            AscendC::Reg::StoreAlign(dst + m * 1024 + blk * 128 + P6_VEC, out1, mask);
        }
    }
}

#endif // FFT_SIMT_OPS_H