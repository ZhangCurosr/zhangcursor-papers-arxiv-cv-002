---
title: "NemoSplat-Feed-Forward-4D-Gaussian-Splating-for-Media-Aware"
source: https://arxiv.org/pdf/2608.22888v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 19:28:42"
---

# 论文速读：NemoSplat-Feed-Forward-4D-Gaussian-Splating-for-Media-Aware

## 一句话总结
提出 NemoSplat，首个面向强介质退化与复杂动态物体的 feed-forward 4D Gaussian Splatting 基础模型，直接从无标定海洋视频中单步前向推理，联合解算相机位姿、稠密深度、动态/静态掩码（可选文本引导）与物理水质参数，实现高保真、无伪影的水下新视角合成与介质去散射。

## 研究问题与动机
- 无约束水下环境存在严重的光吸收/散射退化及大量瞬时动态生物，传统几何与视觉重建管线难以直接适用。
- 现有 feed-forward 视觉基础模型（如 VGGT、StreamVGGT）及 3DGS 前向模型（如 AnySplat、YoNoSplat）依赖静态清晰空气假设，水下运动干扰与光学衰减会致命破坏其特征聚合，导致跟踪失效与渲染劣化。
- 优化型动态 GS-SLAM（如 WildGS-SLAM、Droid-W）严重依赖 SfM 估计的相机先验，且需迭代优化，耗时高、难以适应强动态场景。
- 现有介质去散射方法多基于静态场景假设，动态目标会与介质散射耦合，引发几何不稳定与错误色彩恢复。

## 核心贡献（创新点）
1. **提出首个介质感知动态重建的 feed-forward 4DGS 框架 NemoSplat**，在无标定水下视频中单步前向完成几何估计、动态分离与介质恢复的联合优化；与现有静态/清晰假设的 3DGS 前向方法的本质区别在于显式建模水体光学退化并同步解耦瞬态物体。
2. **设计 Promptable Dynamic Disentangler**，引入置信度感知的 logit 融合策略，将学习的动态概率图与可选 CLIP 文本语义先验结合，精准剥离大规模瞬时动态物体；与依赖低层光流或隐式时空注意力易产生运动泄漏/鬼影的方法本质不同，该策略利用高层语义显式约束掩码边界。
3. ** formulate Media-Aware Gaussian Predictor**，在单次前向中联合回归标准 3DGS 属性与物理水介质参数（衰减系数、后向散射、全局背景眩光），并引入 Jerlov 海洋光学惩罚与多视图一致性正则；与需测试时昂贵迭代优化的物理渲染方法（如 SeaSplat）的本质区别在于一次性前向推断即满足物理可解性。
4. **构建大规模动态水下数据集**（256 训练序列/15.5 万帧，覆盖 9 大地理域与 13 类海洋生物），填补该领域缺乏标准化 benchmark 的空白。

## 方法详解
- **问题形式化**：给定无标定图像序列 $\{I_i\}$ 与可选文本 $T$，学习映射 $f_\theta$ 直接输出每帧相机位姿 $p_i$、深度 $D_i$、动态掩码 $M_i$、全局背景眩光 $B_i^\infty$，以及每像素 3DGS 属性 $(\mu_g, s_g, r_g, \sigma_g, c_g)$ 与水质参数 $(\beta_g^D, \beta_g^B)$。按阈值 $\tau$ 将高斯原语划分为静态集 $G_{\text{static}}$ 与动态集 $\mathcal{G}_{\text{dyn},i}$，干净辐射度 $J_{GS,i}$ 经水下物理成像方程 $\hat{I}_{\text{obs},i} = J_{GS,i} \odot e^{-\beta_i^D D_i} + B_i^\infty \odot (1 - e^{-\beta_i^B D_i})$ 组合生成观测图像。
- **Geometry Estimator**：采用 Streaming Geometry Transformer，以 DINOv2  tokenize 图像并替换全局时序注意力为因果跨帧注意力+恒定内存 KV 缓存，适配长序列。Camera Pose Head 用 MLP 回归 9-DoF 位姿，受 StreamVGGT 教师网络 Huber 监督；Depth Head 基于 DPT 结构，结合教师置信度加权深度 Loss 与 Depth-Anything-3 提供的 SSI 结构 Loss 进行无监督训练。
- **Promptable Dynamic Disentangler**：DPT 骨干融合图像高频边缘特征，预测初始动态概率图 $\hat{M}_{\text{dyn}}$ 与水掩码 $\hat{M}_{\text{wat}}$（后者用于下游约束水质参数空间分布）。可选文本分支通过 CLIP 生成语义相似度图 $S_{\text{sem}}$，推理时采用 logit 调整策略 $\text{logit}(P_{\text{final}}) = \text{logit}(P_{\text{dyn}}) + \gamma(P_{\text{sem}} - \tau_s)$，避免硬布尔交集破坏物体完整性，实现可解释的动态-静态解耦。
- **Media-Aware Gaussian Predictor**：基于 AnySplat 高斯头设计，密集回归标准 3DGS 参数与水下物理参数。综合损失包括：物理渲染 Loss（L2 + LPIPS）、远水面 $B^\infty$ 先验 Loss、逆问题正则 Loss $\mathcal{L}_J$、Jerlov 通道衰减顺序惩罚 $\mathcal{L}_{\text{Jerlov}}$、参数空间 TV 平滑 $\mathcal{L}_{\text{TV}}$，以及渲染深度与 DPT 深度的 $L_1$ 一致性 Loss $\mathcal{L}_{\text{dc}}$。训练分两阶段：Stage 1 独立优化解算器；Stage 2 冻结视觉骨干并用 LoRA 联合优化全管线。

## 实验与结果
- **数据集与基线**：自建数据集含 6 个合成序列（BoulderShore、Coral、Deepsea）与 14 个真实水下序列。对比基线包括 VGGT、StreamVGGT、WildGS-SLAM、Droid-W、YoNoSplat、AnySplat。评估指标：ATE (m)、PSNR、SSIM、LPIPS。另在 SeaThru-NeRF 验证去散射能力。
- **主要结果**：合成数据平均 ATE 为 1.88 m，优于 VGGT (2.12) 与 StreamVGGT (2.06)；新视角合成平均 PSNR 23.98 dB、SSIM 0.56、LPIPS 0.36，显著领先 AnySplat (PSNR 21.76) 与 WildGS-SLAM。真实水下序列全面 SOTA：平均 PSNR 21.58 dB、SSIM 0.68、LPIPS 0.26，较次优 AnySplat 提升 2.36 dB / 0.08 SSIM，LPIPS 较 WildGS-SLAM 相对降低 36.6%。去散射实验中，单前向 <10 秒即可超越迭代 1k 次（~6 分钟）的 SeaSplat，恢复更自然色彩与远距离细节。
- **结论**：显式动态解耦与物理介质联合建模有效抑制了真实场景中鱼群等瞬态物体导致的鬼影与拓扑模糊，实现了无伪影、时间一致的高保真渲染。

## 相关工作脉络
- **Visual SLAM & Feed-Forward Reconstruction**：VGGT/StreamVGGT 等视觉几何基础模型统一了位姿与深度估计，但缺乏光度合成能力；PixelSplat/AnySplat 实现单步 3DGS 回归，但依赖静态清晰假设。本文将其拓展至强退化动态水下域。
- **Dynamic Scene Modeling**：Dynamic 3D Gaussians / Luiten et al. 依赖时序形变场或光流；WildGS-SLAM/Droid-W 等优化型 SLAM 依赖 SfM 先验且迭代缓慢。本文以显式文本引导的高层语义掩码替代脆弱低层线索，规避运动泄漏。
- **Media-Aware Underwater Reconstruction**：SeaSplat/WaterSplatting 等基于物理优化的方法需测试时迭代，且假设静态场景；本文单次前向联合估计介质参数与 4DGS，
