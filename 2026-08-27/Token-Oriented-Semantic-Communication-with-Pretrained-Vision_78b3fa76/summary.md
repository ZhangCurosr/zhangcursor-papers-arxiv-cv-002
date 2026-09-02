---
title: "Token-Oriented-Semantic-Communication-with-Pretrained-Vision"
source: https://arxiv.org/pdf/2608.25410v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 06:56:02"
field: "语义通信与边缘AI协同推理"
keywords: ["语义通信", "Token通信", "Learned Image Compression", "Vision Transformer", "协同推理", "边缘计算"]
innovations: ["利用ViT patch与LIC latent的空间对齐实现token级选择性压缩传输", "提出layer-selective attention rollout在单次前向传播中估计token任务相关性", "引入单个可学习surrogate token在冻结backbone下实现输入级参数高效适配"]
benchmarks: ["ImageNet-1K classification"]
---

# 论文速读：Token-Oriented-Semantic-Communication-with-Pretrained-Vision

## 一句话总结
本文提出了一种基于预训练 ViT 的 Token-导向语义通信框架，通过 Token 级任务相关性在 Learnable Image Compression (LIC) 潜空间中选择性传输图像 latent，实现了无需端到端训练的模块化客户端-服务器协同推理；在 ImageNet 分类任务上，相比现有语义通信方案和手工/任务无关编码方法取得了更有利的率-准确率权衡。

## 研究问题与动机
- **核心问题**：直接传输 Token embedding 实现语义通信面临两个实践障碍——(1) 高维 embedding 向量导致通信开销巨大；(2) 不同模型因架构、预训练和目标差异导致 embedding 空间不兼容，跨模型互操作性受限。
- **现有方案不足**：已有端到端联合训练的 Token 通信方法（如 VQ-based 方法）需共享 codebook，降低模块化、需重新训练、且与特定下游任务耦合；而先前的 Token-oriented 方法（如 [31]–[34]）仅传输像素域 image patch，未能充分利用高效压缩。
- **设计动机**：希望保留 Token 级任务相关性的优势，但以图像域压缩（LIC）而非模型特定 embedding 空间进行传输，实现模块化、低通信成本的协作推理。

## 核心贡献（创新点）
1. **Token-aligned Learned Image Compression**：利用 ViT patch tokens 与 LIC latent vectors 之间的一一对应空间关系，将 Token 级任务相关性直接映射为对 LIC latent 的选择性传输，无需修改 LIC 架构或进行端到端训练。
2. **Layer-selective Attention Rollout**：从选定的 ViT 层范围聚合注意力权重以估计 Token 任务相关性，精度优于最后一层注意力与全层 attention rollout，且仅需单次前向传播（无需梯度计算），适合资源受限客户端。
3. **Learnable Surrogate Token Substitution**：在服务端用单个可学习 surrogate token 替换未选中位置的 token，在冻结大型预训练 backbone 的前提下实现参数高效的输入级适配，提升部分视觉信息下的推理准确率与信道 erasure 鲁棒性。
4. **模块化无端到端训练框架**：协调轻量级客户端 ViT、LIC 压缩器与大型服务端 ViT 三个独立预训练组件，各组件可替换而无需重新训练其他组件。

## 方法详解
**整体流程（Algorithm 1）**：
- 客户端将输入图像 X 切分为 visual tokens $\mathbf{X}^t$，用轻量级 ViT 计算 token 级任务相关性 $\bar{\mathbf{r}}$。
- 按阈值 $\delta$ 生成 token-selection map $\mathbf{s} \in \{0,1\}^N$（累积相关性超过 $\delta$ 为止）。
- 在 LIC 潜空间中进行 selective latent transmission：仅编码选中位置 $S$ 的 latent $\widehat{\mathbf{X}}_S^c$ 与 hyperprior $\widehat{\mathbf{Z}}$，生成 bitstream $\mathbf{b}$。
- 服务端接收后，用 hyperprior-predicted mean $\widehat{\pmb{\mu}}$ 填充未选中位置（predicted-mean imputation），再经 decoder 重构图像 $\widehat{\mathbf{X}}$。
- 将未选中 token 替换为 learnable surrogate token $\mathbf{x}_{sur}$，送入服务端大 ViT 完成分类。

**关键公式与原理**：
- **Token-selection map**：按相关性降序累积选取，累积和首次超过 $\delta$ 停止，$s_i = 1$ 表示选中。
- **Selective latent transmission**：每个 latent 元素的条件分布由 hyperprior 预测的 $(\mu_{i,d}, \sigma_{i,d})$ 参数化，算术编码器仅对 $i \in S$ 编码，跳过 $i \in \mathcal{U}$。
- **率压缩上界**：$\frac{B}{B_{tot}} \leq \eta + \frac{\rho}{\rho + (1-\rho)k}$，其中 $\rho$ 为 token 选择比例，$k = \bar{H}_\mathcal{U}/\bar{H}_S$；当信息密度相近时（$k \approx 1$），压缩比近似线性于 $\rho$。
- **Predicted-mean imputation**：$\widetilde{\mathbf{x}}_i^c = s_i \widehat{\mathbf{x}}_i^c + (1-s_i)\widehat{\pmb{\mu}}_i$，利用 hyper decoder 已输出的均值进行补全，保持局部统计一致性。
- **Layer-selective attention rollout**：$\mathbf{R} = \widetilde{\mathbf{A}}_{L_e} \widetilde{\mathbf{A}}_{L_e-1} \cdots \widetilde{\mathbf{A}}_{L_s}$，其中 $\widetilde{\mathbf{A}}_l = \mathbb{E}_h[\mathbf{A}_{l,h}] + \mathbf{I}$，排除噪声大的浅层、保留尾部关键层。
- **Surrogate token 替换**：$\widetilde{\mathbf{x}}_{i+1}^t = s_i \widehat{\mathbf{x}}_i^t + (1-s_i)\mathbf{x}_{sur} + \mathbf{e}_i$，仅优化单个 $D$ 维向量，backbone 冻结。

## 实验与结果
- **数据集**：ImageNet-1K，图像中心裁剪至 224×224。
- **模型配置**：客户端使用 DeiT-Tiny（5.7M params，72.2% Top-1），服务端使用 DeiT-III-Large（304M params，86.81% Top-1）。
- **评估基线**：(1) 先验 token-oriented 语义通信方法（selected-patch transmission [33]、importance-aware quantization [34]）；(2) 手工编解码器（JPEG2000、WebP、BPG）；(3) 任务无关 LIC 模型（[26]–[28]）。
- **主要结果**：
  - 在 1.24 bpp 时达到 85.83% 准确率，仅比服务端满分辨率性能低 0.98pp；WebP/BPG 需约 1.57/1.82 bpp 才能匹敌。
  - 相比先验 token-oriented 方法，在相同码率下获得更高分类准确率（得益于 LIC 的率效率）。
  - 在 packet-erasure 信道下，surrogate token substitution 相比 direct-reconstruction 和 selected-patch inference 展现出最稳定的鲁棒性。
  - 引入 Entropy-aware Image Transmission (EIT) 后，在 0.94 bpp 即达到 85.89% 准确率。
- **最强结果**：85.92% 准确率（无信道损失，基准），在低码率（< 2 bpp）区域显著优于所有对比基线。

## 相关工作脉络
1. **VQ-based Token Communications [14], [22]–[24]**：传输 token embedding 的 VQ codebook 索引，需端到端联合训练共享 codebook，本文与之区别在于不传输 token embedding，保留模块化。
2. **Prior Token-Oriented Communications [31]–[34]**：利用 token 相关性引导传输，但传输像素域 image patch；本文与之同属 token-oriented 类别，但传输 LIC latent 域，率效率显著提升。
3. **Last-layer Attention [33]**：直接用最后一层 class-token 注意力估计相关性；本文发现其易受背景漂移影响，提出 layer-selective rollout 改进。
4. **Attention Rollout [35]**：全层注意力矩阵连乘追踪信息流；本文指出其累积浅层噪声导致选择性较低，主张仅聚合中段至尾部层。
5. **Gradient-weighted Attention [36]**：结合梯度重加权精确估计相关性，但需反向传播；本文方法在相近精度下仅需单次前向传播，更适合资源受限客户端。
6. **Learned Image Compression [25]–[28]**：超先验 LIC 模型（如 [26], [27]）作为基础压缩框架，本文创新性地将 token 相关性嵌入 LIC latent 选择过程。

## 局限性与未来方向
- **模块独立性限制**：各组件（ViT、LIC、surrogate token）独立预训练，未能联合优化以达到理论最优率-准确率边界。
- **层范围选择需调参**：$(L_s, L_e)$ 依赖 Ablation 搜索，未提出自适应或理论推导的层范围选择准则。
- **仅验证 ImageNet 分类**：未在视频、多模态或更复杂任务（检测、分割）上验证框架泛化能力。
- **EIT 非本文贡献**：Entropy-aware Image Transmission 作为外部组件集成，未深入分析其与 token 选择机制的协同最优。
- **潜在方向**：可扩展至跨模态（文本/语音）token 通信、动态调整 $(L_s, L_e)$、联合微调 LIC 与 ViT 以提升率效率上限。

## 研究启发与可借鉴点
1. **空间对齐思想可迁移**：ViT patch token 与任何空间降采样表征（如 CNN feature map、LIC latent）之间的 1:1 对应关系可复用于其他视觉编码器-压缩器组合，指导选择性传输设计。
2. **Layer-selective Aggregation 策略**：排除噪声浅层、保留尾部层的注意力聚合思路可推广至其他需要可解释性重要性估计的 Transformer 应用。
3. **Surrogate Token 机制**：单向量替换缺失输入的适配方式，可作为参数高效自适应（PEFT）的通用模板，应用于 mask、dropout 或信道 erasure 场景。
4. **模块化协议设计**：无需端到端训练的组件协调框架，为部署场景中混合不同厂商/版本模型提供了可行范式。
5. **理论压缩上界分析**：率压缩与选择比例近似线性的上界推导方法，可为其他选择性编码系统设计提供理论指导。

## 关键术语表
- **Token-oriented Semantic Communication**：以 Transformer token 粒度识别任务相关性并据此选择传输内容的语义通信范式。
- **Learned Image Compression (LIC)**：基于深度学习的图像压缩方法，通过学习自适应熵模型实现接近手工编解码器的率-失真性能。
- **Hyperprior Network**：LIC 中额外提取的辅助 latent（$\mathbf{Z}$），用于条件化预测主 latent 的概率分布参数。
- **Attention Rollout**：通过逐层连乘归一化注意力矩阵来追踪 token 间信息流动路径的可视化/重要性估计方法。
- **Layer-selective Attention Rollout**：本文提出的改进版，仅在选定的中层至尾部范围内执行 attention rollout 以避免浅层噪声累积。
- **Surrogate Token**：可学习的固定向量，用于在服务端替换未选中/缺失位置的 token，以参数高效方式适配部分信息输入。
- **Predicted-Mean Imputation**：利用 hyperprior 预测的条件均值填充未选中 latent 位置的方法，保持局部统计一致性。
- **Entropy-aware Image Transmission (EIT)**：客户端根据预测置信度（min-entropy）决定本地分类或上传服务端，进一步降低通信负载。

## 可复现要素
- **数据集**：ImageNet-1K（公开）。
- **代码/权重**：论文使用公开预训练模型（DeiT-Tiny、DeiT-III-Large、hyperprior LIC），但具体代码实现未声明开源；surrogate token 训练代码需自行实现。
- **关键超参**：
  - Token-selection threshold：$\delta \in [0.5, 0.98]$（推理），$\delta_{sur} = 0.35$（训练）。
  - LIC rate-distortion 权衡：$\lambda \in \{0.0075, 0.03, 0.1, 0.2, 0.4, 1, 2\}$，训练时 $\lambda_{sur} = 0.03$。
  - Layer range：$(L_s, L_e) = (7, 12)$ for DeiT-Tiny；$(5, 12)$ for DeiT-Small；论文未提及 DeiT-Base 的具体值但方法一致。
  - Surrogate token 训练：20 epochs，其余设置参考 MAE [39]。
- **论文未提及**：GPU 型号、训练时间、具体优化器超参数（除参考 [26][27][39]）。
