---
title: "Self-Routed-Tensor-Adapters-for-Parameter-Eficient-Universal"
source: https://arxiv.org/pdf/2608.16384v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 06:17:14"
---

# 论文速读：Self-Routed-Tensor-Adapters-for-Parameter-Eficient-Universal

## 一句话总结
本文提出自路由张量适配器（SRTA），一种面向多域视觉表征的参数高效微调框架。该方法摒弃外部MoE门控网络，直接从适配器自身的低秩投影中内生计算路由权重，并软融合共享Tucker核心张量切片，以显著更少的参数实现了具有竞争力的多域分类精度。

## 研究问题与动机
- 静态PEFT方法（如LoRA）为所有输入共享固定低秩子空间，在风格、背景、语义高度异构的多域视觉任务中难以兼顾通用因子与域特异性，易引发负迁移与优化不均。
- 现有MoE类PEFT方法（如MoLoRA、MOELoRA）虽引入多专家路径，但依赖独立的外部门控网络计算路由概率，不仅增加额外参数，还将路由决策与适配特征表示几何解耦。
- 视觉基础模型的多域泛化需要一种紧凑机制：既能保留冻结主干的预训练知识，又能按输入动态组织域感知适配路径，实现共享与特化的平衡。

## 核心贡献（创新点）
- **自路由张量适配器（SRTA）**：路由概率由低秩输入投影与可学习域坐标矩阵的内积直接推导，无需额外门控网络。与MoLoRA等工作的本质区别在于消除路由模块与适配表示空间的解耦。
- **输入条件化Tucker适配**：将适配更新空间建模为共享Tucker核心张量，利用路由权重动态混合核心切片生成样本特定的低秩适配矩阵。与静态LoRA的本质区别在于适配子空间随输入软插值而非固定不变。
- **渐进式深度加权路由监督**：提出跨适配器层的路由辅助损失，按层深分配递增权重（$w_\ell = \ell/M$）。与仅依赖顶层分类损失的方法相比，为浅层路由提供直接学习信号，缓解弱监督问题。
- **精度-参数高效权衡**：在五个多域视觉基准上，SRTA以约3.4×~4.8×更少参数达到与MoE基线相当或更优的平均精度，证明紧凑张量结构可替代大规模独立专家库。

## 方法详解
- **核心组件**：SRTA引入四个可训练部件：输入投影矩阵 $V \in \mathbb{R}^{d_{in} \times r_1}$、输出投影矩阵 $U \in \mathbb{R}^{r_2 \times d_{out}}$、共享Tucker核心张量 $\mathcal{G} \in \mathbb{R}^{r_3 \times r_1 \times r_2}$ 以及域坐标矩阵 $C \in \mathbb{R}^{r_1 \times r_3}$，其中 $r_3$ 等于数据集域数量。
- **内在自路由**：对输入序列 $x$ 投影得 $z = xV$，忽略CLS token后对patch token平均得到路由签名 $z_{route} = \frac{1}{T-1}\sum_{i=1}^{T-1}z[:,i,:]$。路由logits为 $q = z_{route}C / \tau$，经softmax得权重 $\alpha$。路由与适配共享同一低秩表示空间。
- **动态Tucker核心融合**：利用 $\alpha$ 对核心张量的 $r_3$ 个切片加权求和，构造样本特定适配矩阵 $\Sigma_b(x) = \sum_{t=1}^{r_3} \alpha_{b,t} \mathcal{G}_t$。适配输出为 $y_{adapt,b} = (z_b \Sigma_b(x))U$，层最终输出为 $y_b = x_b W_0 + s \cdot y_{adapt,b}$。
- **渐进式路由损失**：训练阶段利用真实域标签 $d^*$ 计算逐层CE损失 $\mathcal{L}_{route}^{(\ell)} = \mathcal{L}_{CE}(q^{(\ell)}, d^*)$，并赋予深度权重 $w_\ell = \ell/M$。深层路由获得更强监督，反映深层特征更具域判别性。
- **总训练目标**：$\mathcal{L}_{total} = \mathcal{L}_{CE}(\hat{y}, y^*) + \lambda_{route} \frac{1}{M} \sum_{\ell=1}^M w_\ell \mathcal{L}_{route}^{(\ell)}$。推理时仅依赖输入表征计算 $\alpha$，无需域标签输入。

## 实验与结果
- **数据集与设置**：PACS、VLCS、Office-Home、Digits-DG、NICO++五个多域视觉分类基准；冻结ViT-base-patch16-224-in21k主干，适配器注入所有Transformer层的Query与Value投影；80/20切分，训练集均匀采样2000张微调，30 epoch，AdamW，lr=5e-4，batch=64。
- **主要精度结果**：Rank 64下SRTA平均准确率达 **88.1%**，略超MoLoRA（88.0%）；在PACS（95.2%）、VLCS（84.7%）、Digits-DG（92.7%）取得最佳结果，Office-Home与NICO++表现与最强基线持平。
- **参数效率**：4域设置下SRTA仅 **2.77M** 参数（MoLoRA 9.52M，节省约3.4倍）；6域设置下仅 **3.00M** 参数（MoLoRA 14.31M，节省约4.8倍）。
- **消融验证**：加入路由监督使平均精度从87.1%提升至88.1%，VLCS、Office-Home、NICO++提升显著；Tucker秩从8增至64时精度稳定上升，无MoLoRA在高秩下出现的性能波动；路由热力图显示视觉差异大的数据集呈尖锐对角路由，语义重叠数据集呈柔和共享路由，符合设计预期。

## 相关工作脉络
- **LoRA / DoRA**：静态低秩适配基线。本文与其定位差异在于放弃固定子空间假设，通过Tucker核心与内生路由实现输入依赖的动态适配，避免多域负迁移。
- **MoLoRA / MOELoRA / MALoRA**：外部门控MoE适配器代表。本文摒弃独立路由网络与专家银行，将路由内化于低秩投影交互，在相近精度下将参数量降低3~5倍。
- **FacT**：张量分解轻量化适配方法。FacT将分解主要用于参数压缩，本文则将其作为动态路由结构，使核心张量同时承载共享适配因子与域坐标。
- **Deeply-Supervised Nets**：多层
