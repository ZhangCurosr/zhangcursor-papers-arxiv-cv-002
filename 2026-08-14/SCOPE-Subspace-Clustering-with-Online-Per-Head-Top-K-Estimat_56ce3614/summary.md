---
title: "SCOPE-Subspace-Clustering-with-Online-Per-Head-Top-K-Estimat"
source: https://arxiv.org/pdf/2608.12780v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 04:00:25"
---

# 论文速读：SCOPE: Subspace Clustering with Online Per-Head Top-K Estimation for Sparse Video Attention

## 一句话总结
本文提出 SCOPE，一种免训练的稀疏注意力加速框架，通过将 Post-RoPE Key 沿 3D-RoPE 原生机理拆分为时序/高度/宽度子空间独立聚类，并在线推导每 Head 自适应 Top-K 下限，在 HunyuanVideo、Wan 等视频 DiT 上实现最高 1.99× 端到端加速且保真度全面优于现有基线。

## 研究问题与动机
- **核心问题**：视频 DiT 将时空 Token 展平后，自注意力 $O(N^2d)$ 复杂度成为高分辨率/长视频推理的主要瓶颈；如何在免训练前提下高效筛选高贡献 Key，同时保持与密集注意力一致的生成质量。
- **现有代理分数粒度不足**：基于 Block 或全维 Cluster 的稀疏方法强制同一 Representative 内的多个 Key 共享代理分数，掩盖 Key 级细粒度差异，易遗漏重要 Key。
- **Top-p 易欠选且固定 Top-k 缺乏自适应**：近似代理分布易使 Softmax 过度集中，导致纯 Top-p 保留过少 Key；单一全局固定 Top-k 下限无法捕捉不同 Attention Head 与输入内容的差异。
- **动机**：利用视频 DiT 原生 3D-RoPE 在时序、高度、宽度三个正交通道范围作用的结构先验，独立聚类各子空间可保留细粒度区分；同时从当前输入统计量在线推导每 Head 保留下限，避免昂贵的离线密集 Profile。

## 核心贡献（创新点）
1. **3D-RoPE 对齐的 Key 子空间聚类与组合打分**：按 3D-RoPE 原生通道划分将 Key 拆分为 T/H/W 三部分独立聚类，Query Centroid 切片对各子空间 Codebook 打分后索引相加重建 Key 级代理 Logit。与已有 Block/全维聚类方法的本质区别在于：乘积级组合编码打破“同 Cluster 同分数”的粗糙表达，以极低 MAC 代价实现细粒度 Key 区分。
2. **在线每 Head Top-K 估计机制**：结合 Hybrid Top-p/固定 Top-k 得到基础保留数后，以 Query Cluster 规模为权重计算 Head 级自适应下限，仅对低于该下限的 Cluster 沿同 Ranking 扩展。与已有静态 Head-wise 调度或需离线校准方案的本质区别在于：完全在线、输入自适应，无需预计算稀疏掩码或密集 Profile。
3. **免训练且 Proxy-Computation 严格解耦的稀疏推理**：代理分数仅用于 Key 选择，实际稀疏 Attention 仍使用原始 Q/K/V 计算，不修改预训练权重。与需蒸馏微调的可训练稀疏方法（如 SpargeAttn2）的本质区别在于：保持原始模型行为一致性，适配任意现成 Video DiT 部署。

## 方法详解
- **问题设定**：给定单层单 Head 的 Post-3D-RoPE 矩阵 $Q, K \in \mathbb{R}^{N \times d}$ 与 $V \in \mathbb{R}^{N \times d}$，目标是不显式计算 $N^2$ 密集分数矩阵，为每个 Query Cluster 构造保留 Key 集合 $S_c$。
- **全维度 Query 聚类**：对 $Q$ 执行 K-means 得到 $C_q \ll N$ 个 Cluster，Cluster $c$ 含 $n_c$ 个 Query 与 Centroid $\bar{q}_c$。同 Cluster 内 Query 共享代理排名与选 Key 集合，但最终 Sparse Attention 仍用原始 $q_i$ 计算注意力权重（式 2）。
- **3D-RoPE 对齐 Key 子空间聚类**：利用 3D RoPE 在 T/H/W 三个不重叠通道范围的作用特性，将每个 Key $k_j$ 切分为
