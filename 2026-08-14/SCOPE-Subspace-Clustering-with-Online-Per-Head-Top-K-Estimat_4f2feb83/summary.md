---
title: "SCOPE-Subspace-Clustering-with-Online-Per-Head-Top-K-Estimat"
source: https://arxiv.org/pdf/2608.12780v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 03:59:03"
---

# 论文速读：SCOPE: Subspace Clustering with Online Per-Head Top-K Estimation for Sparse Video Attention

## 一句话总结
提出 SCOPE，一种面向视频 DiT 推理的免训练稀疏注意力框架，通过 3D-RoPE 对齐的键子空间独立聚类与组合查表打分，结合基于当前激活的在线逐头 Top-k 动态估计，在保持与稠密注意力高保真度的同时实现最高 1.99× 的端到端加速。

## 研究问题与动机
- 现代视频 Diffusion Transformers (DiTs) 将时空 latent 展平为数万 token，自注意力 $O(N^2d)$ 计算成为高分辨率与长视频推理的核心瓶颈。
- 现有免训练稀疏方法多依赖预定义稀疏模式、块级池化或全维聚类代理分数构造掩码，导致同组 key 共享代理分数，掩盖 key 级细粒度差异，激进稀疏时易遗漏高贡献 key。
- 代理分数经 softmax 后分布可能过度集中，使 Top-p 策略保留过少 key；固定 Top-k 下限虽可保底，但单一全局值无法适配不同 attention head 与输入内容的动态差异。
- 需要同时解决“细粒度 key 判别”与“自适应保留数量估计”两个耦合问题，且不应修改预训练权重、无需离线密集校准。

## 核心贡献（创新点）
- **3D-RoPE 对齐的键子空间组合聚类**：将 post-RoPE key 按时间、高度、宽度互斥通道拆分并独立聚类，通过查询质心切片对子空间码本查表后叠加，重构 key 级代理 logits。与已有工作的本质区别在于：利用 3D RoPE 通道正交性实现乘法级组合表示空间与加法级评分开销，避免全维聚类或块级代理导致的鉴别力退化。
- **在线逐头 Top-k 估计机制**：在推理时以 query cluster 大小为权重，计算当前 head 内混合 Top-p/固定 Top-k 保留数的加权平均作为动态下限，仅对低于该下限的 cluster 沿相同降序排名扩展。与已有工作的本质区别在于：完全由当前输入与激活态实时推导，无需离线密集 profiling 或静态 head-wise schedule。
- **系统级免训练稀疏加速验证**：在 Wan2.1、Wan2.2 与 HunyuanVideo 三个视频 DiT 家族的六种 720p 配置上，SCOPE 在所有保真度指标（PSNR/SSIM
