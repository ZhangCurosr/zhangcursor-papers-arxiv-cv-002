---
title: "Toward-Sub-1-kB-Identity-Preserving-Face-Compression-A-Bench"
source: https://arxiv.org/pdf/2608.22866v1.pdf
model: agnes-2.5-flash
chunks: 10
summarized_at: "2026-09-02 06:54:31"
---

# 论文速读：Toward-Sub-1-kB-Identity-Preserving-Face-Compression-A-Bench

## 一句话总结
本文系统评测了次 1 kB（512 B/1024 B）人脸压缩对身份保真度与像素重建质量的影响，提出身份验证与感知质量双轴解耦评估协议，并覆盖 14 种主流人脸匹配器与多类经典/学习型编解码器；同时提出一种带硬字节预算保证的自定义学习编解码器（Ours-FAST/Ours-ACCURATE），在极低码率下显著优于传统方案且部署适应性更优。

## 研究问题与动机
- 现有压缩标准与学习型编解码器在次 1 kB 极端低码率下缺乏统一的身份保真度基准，传统图像质量指标（PSNR/SSIM）无法可靠预测人脸识别性能。
- 子 1 kB 场景面临“身份损伤”与“像素失真”耦合评估难题，必须将验证准确率与重建质量分开报告，避免单一分数掩盖真实退化。
- 工业落地需综合权衡许可授权、硬件解码器可用性、CPU/GPU 编码速度与硬字节预算合规率，但现有文献缺乏系统性对照。
- 人口统计公平性（肤色/MST、年龄、性别）在极端压缩下的身份验证差异尚未被充分量化与切片分析。

## 核心贡献（创新点）
- **双轴评估协议**：将身份验证指标（EER/id-cos）与像素重建指标（PSNR/LPIPS/DISTS/FIQ）解耦评测，揭示次 1 kB 下二者分化规律。
- **大规模多 matcher 基准**：覆盖 14 种人脸验证模型（ArcFace/TopoFR/LVFace/EdgeFace 等）与双数据集（Color FERET/AI-Solutions-KK），证明编解码器相对排序跨 matcher 稳定。
- **硬字节预算保证的学习编解码器**：通过冻结 log-spaced gain 表与二分搜索实现严格 ≤B 字节输出，配合自描述容器格式支持身份-only fallback。
- **部署三维分析**：系统量化许可、硬件解码器、CPU/GPU 速度及操作点降级策略，为实际工程选型提供数据支撑。

## 方法详解
- **评估协议设计**：采用配对策略枚举 C(N,2) 身份对，负样本固定种子均匀采样上限 5×10⁶；存储为 compact parquet（整数索引对+标签）+ 图像索引表（路径/身份/alignment residual/pose code/detection score）。使用 4 个 anchor matcher（`arcface_antelopev2`/`lvface_l`/`topofr_r100`/`edgeface_xs`）评分，并以专有 `inno-balanced` embedder（512 维 ONNX，不在 14 模型 roster 内）计算每 cell 的 median id-cos（n=64 held-out crops）。FIQ 沿用 SER-FIQ 思路：每张图做 T=10 次轻微数据增强，取嵌入余弦相似度均值。
- **自定义编解码器架构**：共享骨干为均值-尺度超先验 + 第三层 hyper-downsample（潜变量 z 为 x/128，较标准 x/64 缩小 4 倍，gain-independent z 字节下限从 ≈524 B 降至 ≈48 B）；上采样采用 ResizeConv 替代转置卷积以消除低率棋盘伪影；每阶段注入 FiLM 层以 (gain, resolution) 条件调制特征，ACCURATE 额外以解码身份码调制。FAST 变体仅 1.35 M 参数（无注意力/侧流），ACCURATE 为 18.7 M（冻结 3.65 M EdgeFace-S 锚点，训练 15.06 M），含 4 个注意力块与身份侧流 refine head。
- **硬字节预算机制**：预置 64 级冻结 log-spaced gain 表，编码时执行 ≈6 次二分搜索并结合实测 rANS 字节数，选取满足 ≤B 的最大 gain；容器采用 7 B 固定开销自描述布局（flags / res_bucket / sc_len / 可选 side-channel payload / len(y)+y / len(z)+z / H,W 尾部），极端裁剪或极小预算下仍可输出可解析的 ≤B 流。
- **训练策略**：使用 EdgeFace identity loss 优化身份保真度；变量率增益通过二分搜索动态绑定，实现码率与质量的精细控制。

## 实验与结果
- **身份保真度（EER）**：112 px / 1024 B 下，WebP/AVIF/JPEG XL/HEIF/JPEG-AI 均保持 EER ≤ 1.0%（EdgeFace-XXS 下 JPEG-AI 1.08%、WebP 1.14%、AVIF 1.15%、JPEG XL 1.46%、HEIF 1.48%）；512 B 时 JPEG 家族崩盘至 18–33%，JPEG 2000 最劣（EER 3.23–5.57%）；Ours-FAST/Ours-ACC 在 512 B 优雅退化。强 matcher（TopoFR-R200/CVLface IR-101/EdgeFace-Base/LVFace-B）在优质编解码器下 EER ≤ 0.36%（前三者 ≤ 0.14%）。
- **预算合规率**：512 B 下块编解码器仅 112 px 合规率达 51–77%；JPEG 2000/JPEG-FzT/JPEG-AI/Ours 系列可靠达标；Ours-ACC 在 96/168 px 合规率 79%，Ours-FAST 在 168/224 px 仅 9–14%。
- **重建质量**：1024 B/112 px 全参考 PSNR 最佳为 WebP 33.45 dB > JPEG-AI 33.32 dB；LPIPS 最佳 Ours-ACC 0.012 > Ours-FAST 0.029 > WebP 0.045；DISTS 最佳 Ours-ACC 0.095。FIQ 在 1024 B 几乎所有编解码器维持 ≈0.95，512 B 时 JPEG 降至 0.80（CF）/0.85（KK），JPEG-FzT 224 px 崩溃至 0.84/0.88，Ours 系列始终 ≥0.946。
- **统计一致性**：14 个 matcher 对编解码器的 Kendall W = 0.85；PSNR 与 EER 在 1024 B 相关性不显著（ρ = −0.45, p=0.14），去除 CompressAI 后 ρ = −0.82（p=0.004）；512 B 下感知指标与 EER 强相关（DISTS ρ=+0.96，LPIPS ρ=+0.94）。
- **速度/部署**：经典编码 CPU 跨度 0.26 ms (JPEG) ~ 172 ms (HEIF)；学习类 GPU 加速显著（Ours-ACC 54/63 ms，Ours
