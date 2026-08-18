---
title: "SQuad-Sub-Quadratic-Attention-Distillation-for-Eficient-Vide"
source: https://arxiv.org/pdf/2608.16585v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 15:21:35"
---

# 论文速读：SQuad-Sub-Quadratic-Attention-Distillation-for-Eficient-Vide

## 一句话总结
本文提出SQuad框架，将预训练视频DiT中的$O(n^2)$ Softmax自注意力替换为保持真值Softmax的局部-全局两段式子二次注意力（$O(n\sqrt{n})$），并通过Flow-Matching SFT与DMD2两步蒸馏，在Wan 2.2 5B上以零额外参数实现VBench质量持平（83.20 vs 83.08），同时将每步注意力FLOPs降低约67倍、端到端延迟降低约2倍，并将采样步数从100压缩至6。

## 研究问题与动机
- 视频生成DiT的计算瓶颈集中在Self-Attention，其$O(n^2)$复杂度随时空token数$n$快速增长，严重限制生成分辨率、时长与吞吐量。
- 现有线性$O(n)$或低秩$O(nk)$近似虽降成本，但破坏了Softmax的非线性与输入依赖的选择性，在高质量视频生成中留下难以填补的质量鸿沟。
- 混合架构（穿插少量二次注意力与大量高效注意力）可缓解质量损失，但引入异构设计、层选择敏感，且仍无法摆脱对原始Softmax的依赖。
- 视频自注意力图呈现稀疏长尾分布，绝大部分注意力质量集中于少数关键token，这为在保留真值Softmax的前提下设计固定结构化通信模式提供了理论动机。

## 核心贡献（创新点）
- 提出SQuad-Attention算子：将标准Softmax自注意力拆解为“窗口内局部混合”与“跨窗口全局混合”两段串联，在严格保留真值softmax的同时将复杂度降至$O(n\sqrt{n})$，与线性/低秩近似形成本质区别。
- 设计两阶段蒸馏流程：先用Flow-Matching SFT将预训练二次DiT权重适配到新注意力模式，再用DMD2进行分布匹配步数蒸馏，同步恢复生成质量并实现6步快速采样。
- 理论证明局部-全局两段通信可在单层内恢复全感受野，且最优窗口大小可由$w^*=\sqrt{n}$解析导出，无需数据依赖的动态token选择或可微分mask。
- 零额外参数、硬件友好：完全由标准softmax与索引重排构成，可直接兼容`torch.compile`，无需厂商定制CUDA kernel，且在Wan 2.2 5B / 1.3B / 14B多尺度模型上泛化验证。

## 方法详解
- **SQuad-Attention结构**：输入$\mathbf{X}\in\mathbb{R}^{n\times hd}$经$W_Q,W_K,W_V$投影后，先通过无参数的`split_L`重组为局部视图$\mathbb{R}^{w\times (n/w)hd}$，执行标准Softmax注意力$\mathrm{Attn}_L$；再将输出经`split_G`重组为全局视图$\mathbb{R}^{(n/w)\times whd}$，执行$\mathrm{Attn}_G$。公式为$\mathbf{Y}_i = \mathrm{Attn}_G(\mathbf{Q}_i, \mathbf{K}_i, \mathrm{Attn}_L(\mathbf{Q}_i, \mathbf{K}_i, \mathbf{V}_i))$，Cross-Attention与FFN保持不变。
- **复杂度推导**：局部代价$C_\mathcal{L}=hdnw$，全局代价$C_\mathcal{G}=hdn^2/w$，总代价$C(w)=hd(nw+n^2/w)$。对$w$求导得极值点$w^*=\sqrt{n}$，代入后总复杂度为$O(n\sqrt{n})$。
- **全感受野证明**：任意源token $(c',j')$ 到目标token $(c,j)$ 可通过“局部 hops 到同槽位 $(c',j)$ → 全局 hops 到目标窗口 $(c,j)$”两跳路径传递，有效权重$\gamma_{(c,j)\to(c',j')}^h=\alpha_{c\to c'}^{j,h}\beta_{j\to j'}^{c',h}>0$，单层即覆盖全部$n$个token。
- **两阶段蒸馏**：
  - Stage 1（Flow-Matching SFT）：在VIPE 1M视频数据上使用原rectified-flow损失$\mathcal{L}_{\mathrm{SFT}}=\mathbb{E}\|\mathcal{D}_\theta(z_\sigma,\sigma,\mathbf{p})-(\epsilon-z_0)\|_2^2$微调8k步，使网络适应新注意力模式。
  - Stage 2（DMD2）：引入学生$G_\theta$、冻结教师$\mathcal{D}_\phi^{\mathrm{tea}}$与在线critic $\mathcal{D}_\psi^{\mathrm{fake}}$，通过梯度$\nabla_\theta\mathrm{KL}\propto\mathbb{E}[(s_{\mathrm{fake}}-s_{\mathrm{real}})\frac{\partial\hat{z}_0}{\partial\theta}]$匹配分布，训练15k–30k步将NFE压缩至6，同时将CFG蒸馏入学生。
- **窗口设计**：理论目标$\lceil\sqrt{n}\rceil\approx136$，消融表明时间维全覆盖（$w_t=T$）、空间维保持宽高比的$21\times2\times4$形状训练最稳定且质量最优。

## 实验与结果
- **数据集与基线**：Wan 2.2 5B（81×704×1280，$n=18480$）与Wan 2.1 1.3B（81×480×832，$n=32760$）；对比Original、DMD原版、VSA、Jenga、Radial Attention、Attention Surgery、ReHyAt等。
- **质量结果**：Wan 2.2 5B上SQuad 30-Block VBench Total达**83.20**，略超Original的83.08；人类偏好测试（1179对比较）中35%偏好SQuad，31%认为无差异，显著优于Radial Attention与DMD基线。
- **效率结果**：注意力FLOPs从4.205降至0.063（**~67×**），单步注意力延迟从47.10ms降至4.27ms（**~11×**）；端到端DiT前向在`torch.compile`下从667ms降至314ms（**~2.1×**）；NFE从100降至6。Wan 2.1 14B上FLOPs从41.96
