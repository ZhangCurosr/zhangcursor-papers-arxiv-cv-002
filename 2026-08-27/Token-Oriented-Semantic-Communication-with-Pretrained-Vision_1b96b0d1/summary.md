---
title: "Token-Oriented-Semantic-Communication-with-Pretrained-Vision"
source: https://arxiv.org/pdf/2608.25410v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 06:56:14"
field: "语义通信与边缘AI"
keywords: ["语义通信", "Token通信", "Vision Transformer", " Learned Image Compression", "边缘计算", "协同推理"]
innovations: ["提出token-aligned LIC框架，利用ViT patch与LIC latent的空间对齐实现选择性压缩", "设计layer-selective attention rollout，在单前向传播中精确估计token任务相关性", "引入surrogate token substitution，以单个可学习token实现高效的服务端输入适配"]
benchmarks: ["ImageNet-1K"]
---

# 论文速读：Token-Oriented-Semantic-Communication-with-Pretrained-Vision

## 一句话总结
本文提出了一种**模块化 token 导向语义通信框架**，通过在图像域而非 token embedding 域进行选择性压缩，实现了客户端（轻量级 ViT）与服务端（大型 ViT）之间的高效协同推理，在 ImageNet 分类任务上以 1.24 bpp 的比特率达到了 85.83% 的准确率，显著优于现有语义通信方案和传统编解码器。

## 研究问题与动机
1. **Token 通信的通信开销问题**：直接传输 token embeddings 会面临高昂的通信成本，因为每个 token 是高维嵌入向量，即使减少数量也可能产生大量数据载荷。
2. **跨模型 embedding space 互操作性不足**：不同架构、预训练过程和下游目标独立训练的模型，其 token embedding space 差异巨大，难以直接兼容。
3. **现有方案的局限性**：端到端联合训练的 VQ-based token communication 虽然高效但缺乏模块化；先前 token-oriented 方法传输像素域 image patches，无法利用 learned image compression 的统计冗余消除能力。
4. **资源受限边缘设备的实际需求**：边缘端设备内存和算力有限（如 DeiT-Tiny 仅需 22.9 MB 内存），而云端服务器可部署大规模模型（如 DeiT-III-Large 需 1217.5 MB），需要在两者间建立高效的协同推理机制。

## 核心贡献（创新点）
1. **Token-aligned Learned Image Compression**：利用 ViT patch tokens 与 LIC latent vectors 之间的一一空间对齐关系，将 token 级任务相关性直接集成到压缩过程，选择性传输任务相关 latent，而非直接传输 token embeddings——与先前方法传输像素域 patches 的本质区别在于操作于 LIC latent 域，实现了更高的率-失真效率。

2. **Layer-selective Attention Rollout**：提出从选定的 transformer 层范围（而非单层或全部层）聚合 attention weights 来估计 token 任务相关性，仅需单次前向传播即可达到与 gradient-weighted attention 相当的精度——与 prior 方法（last-layer attention、全层 rollout、gradient-weighted attention）的本质区别在于平衡了精度与计算效率，避免了资源受限客户端的梯度计算开销。

3. **Learnable Surrogate Token Substitution**：在服务端用单个可学习 token 替换未选中位置的视觉 token，在冻结预训练 backbone 的前提下实现参数高效的输入级适配——与直接重构推断和 selected-patch 推断的本质区别在于，既不需要昂贵的 backbone 微调，又能有效补偿缺失视觉信息，同时增强了不可靠信道下的推理鲁棒性。

## 方法详解

### 整体框架
框架由三个独立预训练组件模块化协同构成：
- **客户端**：轻量级 ViT（DeiT-Tiny）+ 神经压缩器
- **服务端**：大型 ViT（DeiT-III-Large）+ 神经解压缩器

推理流程（Algorithm 1）：
1. 客户端从输入图像 X 提取视觉 tokens $X^t$
2. 通过 layer-selective attention rollout 估计 token 级任务相关性 $\bar{r}$
3. 基于阈值 δ 构建 token-selection map $s \in \{0, 1\}^N$
4. Token-aligned LIC 生成仅包含选中 token 对应 latent 的压缩码流 b
5. 服务端接收后重建图像，通过 surrogate token substitution 构造输入序列
6. 服务端 ViT 完成分类并返回结果

### 关键技术 1：Token-aligned Learned Image Compression

**核心洞察**：ViT 的 patch size（P=16）与 LIC encoder 的空间下采样因子（4 层 stride-2 conv）相同，使得每个 visual token $\mathbf{x}_i^t$ 与 latent vector $\widehat{\mathbf{x}}_i^c$ 存在一一空间对应关系。

**选择性 latent 传输**：
- 基于 hyperprior-based entropy model [26, 27]，latent 元素在给定 $\widehat{Z}$ 条件下条件独立
- 算术编码器仅对选中位置 $i \in S$ 的 latent 进行熵编码，跳过未选中位置 $i \in U$
- 传输码流为 $(\widehat{\mathbf{X}}_S^c, \widehat{Z}, s)$

**通信效率理论分析**：
$$\frac{B}{B_{\text{tot}}} \leq \eta + \frac{\rho}{\rho + (1-\rho)k}$$
其中 $\rho = |S|/N$ 为 token 选择比例，$\eta$ 为 side information 开销比，$k = \bar{H}_U/\bar{H}_S$ 为信息密度比。当 $k \approx 1$ 时，$B/B_{\text{tot}} \lesssim \eta + \rho$，即速率降低与选择比例近似线性相关。

**Predicted-mean imputation**：
解码器在未选中位置用 hyper decoder 预测的均值 $\widehat{\mu}_i$ 进行填充：
$$\widetilde{\mathbf{x}}_i^c = s_i \widehat{\mathbf{x}}_i^c + (1 - s_i)\widehat{\mu}_i$$
这利用了卷积层的重叠感受野， improves 选中区域的图像重建质量。

### 关键技术 2：Layer-selective Attention Rollout

**问题动机**：
- Last-layer attention 易将高相关性错误分配给背景区域（最终层 attention 出现 "background drift"）
- Attention rollout（全层累积）会包含早期层的高噪声 attention
- Gradient-weighted attention 精度高但需反向传播，对资源受限客户端不实用

**方法**：
从第 $L_s$ 层到第 $L_e$ 层聚合 attention：
$$\widetilde{\mathbf{A}}_l = \mathbb{E}_h[\mathbf{A}_{l,h}] + \mathbf{I}_{N+1}, \quad \mathbf{R} = \widetilde{\mathbf{A}}_{L_e}\widetilde{\mathbf{A}}_{L_e-1}\cdots\widetilde{\mathbf{A}}_{L_s}$$
对 DeiT-Tiny（L=12 层），取 $(L_s, L_e) = (7, 12)$，排除噪声早期层，保留中间到后期的稳定注意力。

**Token 选择**：基于 cumulative relevance sum 与阈值 δ 确定选中集合 S。

### 关键技术 3：Surrogate Token Substitution

服务端构造输入序列时，对未选中位置用单个可学习 token 替换：
$$\widetilde{\mathbf{x}}_{i+1}^t = s_i \widehat{\mathbf{x}}_i^t + (1 - s_i)\mathbf{x}_{\text{sur}} + \mathbf{e}_i$$
其中 $\mathbf{x}_{\text{sur}} \in \mathbb{R}^{1 \times D}$ 是唯一学习的参数（D=1024 for DeiT-III-Large），backbone 保持冻结。

**训练策略**：模拟实际推理流程，用 $(\delta_{\text{sur}}, \lambda_{\text{sur}}) = (0.35, 0.03)$ 训练 surrogate token，20 epochs。

**额外优势**：在 packet-erasure channel 下，受影响的 token 同样可替换为 surrogate token，提升鲁棒性。

## 实验与结果

**实验设置**：
- 数据集：ImageNet-1K，图像 resize 至 224×224
- 客户端模型：DeiT-Tiny（5.7M 参数，72.2% 准确率）+ 超先验 LIC 模型
- 服务端模型：DeiT-III-Large（304.4M 参数，86.81% 准确率）
- 评估协议：所有 baseline 使用相同预训练服务端模型进行分类，lossless 传输时准确率为 86.81%

**主要结果**：

| 对比对象 | 关键结果 |
|---------|---------|
| **Token-oriented 基线** [33, 34] | 在 comparable bit rate 下实现更高准确率，增益主要来自 token-aligned LIC 的率效率 |
| **Hand-crafted codecs** | 在 1.24 bpp 达到 85.83% 准确率，仅低于服务器 0.98pp；WebP 需 1.57 bpp、BPG 需 1.82 bpp 才能匹配 |
| **Task-agnostic LIC** [26-28] | 在 <2.5 bpp 范围内实现更优 trade-off，验证任务相关性引导的 latent 选择优于仅优化重建质量 |

**鲁棒性实验**：在 packet-erasure channel 上（$p_e \in \{0.05, ..., 0.5\}$），surrogate token substitution 在所有 erasure probability 下均优于 direct-reconstruction 和 selected-patch inference。

**Entropy-Aware Image Transmission (EIT)**：结合 EIT 后，在 0.94 bpp 达到 85.89% 准确率，较无 EIT 节省约 0.3 bpp。

## 相关工作脉络

1. **VQ-based Token Communications** [14, 22-24]：传输 token embeddings 的 VQ codebook indices，需端到端联合训练共享 codebook，缺乏模块化；本文在 token granularity 上进行语义通信但不直接传输 embeddings，保持预训练模型独立性。

2. **Prior Token-Oriented Communications** [31-34]：利用 token 级相关性指导传输，但操作于像素域（raw/quantized patches）；本文进一步操作于 LIC latent 域，利用压缩模型的统计冗余消除能力实现更高率效率。

3. **Learned Image Compression** [25-28]：非线性变换编码框架及 hyperprior 扩展；本文在此基础上引入 token 级选择性传输机制，将任务相关性集成到压缩过程而非仅优化重建保真度。

4. **Attention-guided Importance Estimation**：Last-layer attention [31-34] 简单但背景漂移；全层 rollout [35] 累积早期噪声；gradient-weighted attention [36] 精确但需反向传播；本文的 layer-selective attention rollout 在中间层范围内聚合，兼顾精度与客户端计算约束。

5. **Learnable Input Tokens / Prompt Tuning** [37-39]：Visual prompt tuning 添加可学习 prompt tokens；MAE 使用 mask token；本文的 surrogate token 结合两者特点，但目标是补偿缺失视觉信息而非领域适配。

## 局限性与未来方向

1. **模块间独立预训练的局限**：三个组件（client ViT、LIC、server ViT）独立预训练后无需联合微调，但这可能导致 suboptimal 的全局性能；端到端联合训练可能获得更好 trade-off，但会牺牲模块化优势。

2. **对 ViT 架构的依赖**：方法核心依赖于 ViT patch size 与 LIC 下采样因子的空间对齐（均为 16 倍），对于其他架构（如 CNN-based classifiers 或不同 patch size 的 ViT）可能需要调整。

3. **Token 选择策略的静态性**：当前使用固定阈值 δ 进行 token 选择，未考虑动态信道条件或自适应调整；不同图像复杂度差异较大时可能不够灵活。

4. **仅验证图像分类任务**：实验仅在 ImageNet 分类上评估，未扩展到生成任务、目标检测或多模态场景。

5. **Surrogate token 的单一性**：仅使用一个全局 surrogate token 替换所有未选中位置，可能无法区分不同语义区域的缺失信息特征。

## 研究启发与可借鉴点

1. **跨域空间对齐思想**：ViT patch tokens 与 LIC latents 的一一映射关系是一个巧妙的设计，可迁移到其他 vision-language 或多模态场景中，利用不同模型间的结构对齐实现高效信息传递。

2. **Layer-selective aggregation 策略**：排除早期噪声层、保留关键中间层的 attention 聚合策略，可推广到任意 transformer 模型的 token importance estimation，避免全层计算或单层局限。

3. **Surrogate token 的适配范式**：单个可学习 token + frozen backbone 的轻量级适配方式，适用于任何部分输入缺失的场景（如遮挡鲁棒性、多视图融合、联邦学习中的局部更新）。

4. **模块化语义通信架构**：不直接传输模型-specific embeddings 而是在公共域（图像域）进行压缩传输的思路，可拓展到音频、视频、点云等多模态语义通信。

5. **EIT（Entropy-Aware Image Transmission）的集成**：客户端本地置信度判断 + 服务端 fallback 的两级决策机制，可作为通用组件集成到各类 edge-cloud 协同推理系统中。

## 关键术语表

**Token-oriented semantic communication**：在 transformer token 粒度上进行语义通信的框架，根据 token 级任务相关性指导传输哪些信息，而非传输完整 embeddings 或原始数据。

**Learned Image Compression (LIC)**：基于深度学习的图像压缩方法，通过学习的数据分布实现优于传统手工编解码器（如 JPEG）的率-失真性能。

**Layer-selective attention rollout**：仅在选定的 transformer 层范围内聚合 attention weights 来估计 token 任务相关性的方法，平衡了精度与计算效率。

**Surrogate token**：用于替换未选中或丢失 token 位置的可学习向量，在 frozen backbone 前提下通过输入级适配补偿信息缺失。

**Token-selection map**：二值向量 $s \in \{0, 1\}^N$，标记哪些 token 被选中传输，由 token-level 任务相关性与阈值 δ 共同决定。

**Predicted-mean imputation**：在 LIC 解码端用 hyper decoder 预测的条件均值填充未选中 latent 位置的方法，利用相邻 latent 的统计一致性改善重建质量。

**Rate-accuracy trade-off**：衡量语义通信框架性能的指标，描述传输比特率与下游任务准确率之间的权衡关系。

**Client-server collaborative inference**：资源受限客户端与算力丰富的服务端协同完成推理任务的范式，客户端负责预处理和选择性传输，服务端完成最终推理。

## 可复现要素

- **数据集**：ImageNet-1K（公开）
- **代码/权重**：论文未明确声明开源，但使用了公开预训练模型（DeiT-Tiny、DeiT-III-Large、hyperprior-based LIC models）
- **关键超参**：
  - Token 选择阈值：δ ∈ [0.5, 0.98]
  - LIC 参数：λ ∈ {0.0075, 0.03, 0.1, 0.2, 0.4, 1, 2}
  - Layer-selective attention：$(L_s, L_e) = (7, 12)$ for DeiT-Tiny
  - Surrogate token 训练：$\delta_{\text{sur}} = 0.35$, $\lambda_{\text{sur}} = 0.03$, 20 epochs
  - LIC 模型维度：latent $D_c = 192$, hyperprior 128
- **分辨率**：224×224（N=196 tokens）
