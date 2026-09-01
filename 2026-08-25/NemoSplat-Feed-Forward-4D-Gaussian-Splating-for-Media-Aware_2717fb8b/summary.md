---
title: "NemoSplat-Feed-Forward-4D-Gaussian-Splating-for-Media-Aware"
source: https://arxiv.org/pdf/2608.22888v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 19:28:26"
field: "水下三维视觉与动态场景重建"
keywords: ["4D Gaussian Splatting", "Feed-Forward Model", "Underwater Reconstruction", "Dynamic-Static Disentanglement", "Media-Aware Rendering", "Visual Geometry Foundation Model"]
innovations: ["首个面向水下动态场景的 feed-forward 4DGS 框架", "Promptable Dynamic Disentangler 实现几何与 CLIP 语义的 logit 融合解耦", "Media-Aware Gaussian Predictor 单次前向联合估计 3DGS 属性与物理介质参数"]
benchmarks: ["UE5 合成水下序列（BoulderShore/Coral/Deepsea）", "14 条真实水下序列", "SeaThru-NeRF 去散射评估"]
---

# 论文速读：NemoSplat: Feed-Forward 4D Gaussian Splatting for Media-Aware Underwater Reconstruction

## 一句话总结
本文提出 NemoSplat，首个面向水下动态场景的 feed-forward 4DGS 视觉基础模型，能从未校准水下视频中单次前向传播同步估计相机位姿、稠密深度、动态掩码和物理介质参数，实现无伪影的保真新视角合成与去散射恢复。

## 研究问题与动机
- 水下环境存在严重的光学衰减（光吸收）与体积散射效应，导致颜色失真与远处细节丢失。
- 海洋生态充满大量瞬时动态目标（鱼群等），传统几何重建难以在动态干扰下保持跟踪稳定。
- 现有 feed-forward 视觉基础模型与 3DGS 方法面向静态清晰空气场景设计，直接应用于水下视频时特征聚合会被运动干扰和光衰减破坏，引发跟踪失败与重建退化。
- 现有优化型 SLAM/GS-SLAM 依赖预先估计的相机内参或通过 SfM 初始化，在水下未校准且动态复杂的环境中难以稳定工作。

## 核心贡献（创新点）
- **首个 feed-forward 4DGS 水下框架**：NemoSplat 首次将单次前向传播的 4DGS 引入未校准水下动态场景重建，与 AnySplat、PixelSplat 等清晰静态假设的 3DGS 方法有本质区别。
- **Promptable Dynamic Disentangler**：通过置信度感知的 logit 融合策略，将学习到的动态概率与可选 CLIP 文本语义先验结合，显式解耦瞬态运动与静态背景；与传统低层光流/变形场方法相比，绕开了脆弱低层线索。
- **Media-Aware Gaussian Predictor**：在单次前向传播中联合预测标准 3DGS 属性与物理介质参数（衰减系数、背光散射、全局晕影光），并通过 Jerlov 色道约束与结构一致性正则；与 SeaSplat/Water-Splatting 等需测试时优化的方法本质不同。
- **大规模动态水下数据集**：构建含 256 训练序列（155K 帧）和 20 评估场景的数据集，覆盖 9 类地理域与 13 类海洋生物行为，填补该方向基准缺失。

## 方法详解
- **问题形式化**：输入未校准水下图像序列 `{I_i}` 与可选文本提示 `T`，学习映射 `f_θ` 一次性输出每帧相机参数 `p_i`、深度图 `D_i`、动态掩码 `M_i`、全局背景晕影光 `B^∞_i`，以及每个像素对应的 3D 高斯原语 `(μ_g, s_g, r_g, σ_g, c_g)` 和物理介质参数 `(β^D_g, β^B_g)`。
- **动态-静态解耦**：以阈值 τ 将全部高斯原语划分为静态集 `G_static` 与逐帧动态集 `{G_dyn,i}`，通过光栅化得到纯净辐射 `J_GS,i`，再经水下成像模型合成观测图像：`Î_obs,i = J_GS,i ⊙ exp(-β^D_i D_i) + B^∞_i ⊙ (1 - exp(-β^B_i D_i))`。
- **Geometry Estimator**：基于 StreamVGGT 的 Streaming Geometry Transformer，将每帧经 DINOv2  tokenize，采用空间 intra-frame 注意力与因果 cross-frame 注意力的交替结构，通过固定大小 KV cache 聚合多视图几何先验；相机头回归 9-DoF 位姿（以首帧为原点），深度头使用 DPT 架构输出参考深度。训练采用 Teacher 蒸馏 Huber 损失与基于 Depth-Anything-3 的 Scale-Shift Invariant 结构损失。
- **Promptable Dynamic Disentangler**：DPT 分支融合 CNN 提取的高频边缘线索，输出初始动态概率图与水体掩码；可选文本提示经 CLIP 编码得到语义嵌入图，与几何概率通过 logit 调整融合：`logit(P_final) = logit(P_dyn) + γ(P_sem - τ_s)`，避免简单布尔相乘造成的语义噪声破坏目标完整性。
- **Media-Aware Gaussian Predictor**：复用 AnySplat 风格的 GS head 回归标准 3DGS 属性，同时映射 per-pixel 衰减系数、背散射系数与 per-frame 全局晕影光；水掩码被传递用于空间约束介质参数学习。
- **联合损失**：渲染损失包含 L2 + LPIPS 项；物理正则包括晕影光对齐损失、还原纯净辐射的 J 损失、Jerlov 色道衰减顺序约束 `(β^D_R > β^D_G > β^D_B)`、介质参数 TV 正则；几何一致性使用 `L_dc = ||D_dpt - D_gs||_1`。整体分两阶段训练：Stage 1 优化动态解耦器；Stage 2 冻结骨干并用 LoRA 微调全管道。

## 实验与结果
- **数据集**：自建含 256 训练序列（155K 帧）、20 评估场景（6 个 UE5 合成 + 14 个真实水下）；标注由 SAM3 + SAHI 人工精修。
- **基线**：VGGT、StreamVGGT、WildGS-SLAM、Droid-W、YoNoSplat、AnySplat。
- **合成集**：NemoSplat 平均 ATE 1.88 m；PSNR 23.98 dB（最优）、LPIPS 0.36（最优），领先 DROID-W PSNR +0.47 dB、优于 AnySplat LPIPS -0.03。
- **真实水下集**：平均 PSNR 21.58 dB、SSIM 0.68、LPIPS 0.26，均超过所有基线；相较次优 AnySplat 提升 PSNR +2.36 dB、SSIM +0.08；较 WildGS-SLAM LPIPS 相对下降 36.6%。
- **去散射评估**：在 SeaThru-NeRF 上单次前向（<10s）恢复色彩与远端细节优于需 1k 次迭代（~6min）的 SeaSplat。
- **定位差异**：在 Coral（大基线、宽视场）场景中 feed-forward 方法利用学习先验优于光流 SLAM；在 Deepsea（极暗）场景中光流 SLAM 仍可利用原始像素变化维持跟踪。

## 相关工作脉络
- **VGGT / StreamVGGT**：视觉几何基础模型，提供位姿与深度蒸馏教师，但仅输出离散几何、无法直接渲染。
- **PixelSplat / MVSplat / AnySplat / YoNoSplat**：feed-forward 3DGS 系列，面向静态清晰空气假设；NemoSplat 扩展至水下动态退化场景并联合物理介质估计。
- **WildGS-SLAM / Droid-W**：优化型动态 GS-SLAM；依赖初始位姿或光流，未校准水下动态场景中容易漂移或产生重影。
- **4DGT / MoRe**：4D feed-forward 网络通过时空注意力隐式分离动态，但在密集小目标干扰下易出现运动泄漏；NemoSplat 采用高层文本语义显式解耦。
- **SeaSplat / Water-Splatting / SeaThru-NeRF**：基于优化的水下介质恢复方法，依赖外部深度先验与测试时迭代；NemoSplat 将介质参数嵌入单次前向 4DGS 预测。

## 局限性与未来方向
- **显存消耗较大**：集成重型语义掩码与物理介质模块导致训练显存占用高，限制实时部署。
- **极端低光照鲁棒性不足**：Deepsea 等极暗场景中预训练图像编码器难以提取可靠语义特征，性能下降。
- **依赖文本提示的质量**：Promptable Disentangler 在无提示时退化为纯几何判定，复杂语义场景下精度受 Prompt 质量影响。
- **未来方向**：优化架构显存效率以支持 AUV 等端侧实时应用；探索自监督/弱监督降低对 SAM3 标注依赖；增强低光照下的特征鲁棒性。

## 研究启发与可借鉴点
- **文本语义引导的动态解耦**：logit 调整融合几何概率与 CLIP 语义先验的设计可迁移到任意动态场景重建任务，尤其适合目标密集、运动复杂的室内/户外序列。
- **物理介质参数的前向联合估计**：将衰减/背散射/晕影光与 3DGS 属性在同一 Head 内回归，并配合 Jerlov 顺序与 TV 正则，为其他散射介质（雾、雨、浑浊液体）中的 3D 重建提供范式。
- **Teacher 蒸馏 + LoRA 两阶段训练**：利用 StreamVGGT/AnySplat 预训练权重初始化、分阶段微调，能在保持几何稳定性的同时快速适应新域，适合其他foundation模型下游适配。
- **水掩码的空间约束传递**：将动态解耦器输出的水体掩码直接用于约束介质参数学习，实现模块间结构化信息联动，值得推广至多任务联合学习设计。
- **Streaming Transformer + KV cache 的长序列扩展**：因果 cross-frame 注意力的流式设计可复用于长视频三维重建，降低显存开销。

## 关键术语表
**4D Gaussian Splatting (4DGS)**：在传统 3DGS 基础上引入时间维度，用时空高斯原语表示动态场景以实现时序一致渲染。
**Feed-Forward Model**：单次前向传播即输出目标结果的深度学习模型，无需逐场景迭代优化。
**Promptable Dynamic Disentangler**：支持可选文本提示的动态-静态解耦模块，通过几何概率与语义先验的 logit 融合生成高精度掩码。
**Media-Aware Gaussian Predictor**：联合预测 3DGS 属性与水下物理介质参数的网络头，使渲染结果恢复去散射纯净外观。
**Jerlov Penalty**：基于海洋光学经验的正则项，强制要求红光、绿光、蓝光的衰减系数满足递减顺序。
**StreamVGGT**：流式视觉几何 Transformer，以固定内存 KV cache 聚合长序列多视图几何先验的基础模型。
**Scale-Shift Invariant (SSI) Loss**：对深度预测施加尺度与平移不变的结构正则，消除绝对尺度歧义。
**SAM3**：Segment Anything Model 3，用于生成高质量动态与水体伪标签的分割基础模型。

## 可复现要素
- **数据集**：作者构建大规模动态水下数据集（256 训练序列、20 评估场景），其中 6 个为 UE5 合成序列；论文未明确说明是否开源，详见 supplement。
- **代码/权重**：实现基于 PyTorch 与 gsplat CUDA 库；几何估算器与高斯预测器加载 StreamVGGT 与 AnySplat 预训练权重；论文未明确开源声明。
- **关键超参**：Transformer 24 层因果交替注意力；DINOv2-ViT-L14 编码器；总参数约 1.22B；两阶段训练（Stage1 约 1 天、Stage2 约 2 天，5× RTX 4090D）；LoRA 用于 Stage 2 微调骨干；详细超参见 supplement。
- **训练细节**：Stage 2 冻结视觉骨干；动态掩码与水体掩码分别使用 BCE+Dice 损失；水掩码边界加权；logit 融合超参 γ、τ_s 等论文未列出具体数值。
