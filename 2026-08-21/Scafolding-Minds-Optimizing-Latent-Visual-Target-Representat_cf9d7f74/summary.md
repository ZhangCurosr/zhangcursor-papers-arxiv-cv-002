---
title: "Scafolding-Minds-Optimizing-Latent-Visual-Target-Representat"
source: https://arxiv.org/pdf/2608.19669v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 19:25:35"
field: "多模态推理"
keywords: ["latent visual reasoning", "multimodal reasoning", "reinforcement learning", "vision-language models", "scaffolding encoder", "Gaussian policy"]
innovations: ["可学习的scaffolding encoder替代冻结现成视觉编码器，端到端优化潜目标", "Scafolding RL通过输入自适应高斯策略采样残差潜动作实现奖励驱动的潜空间探索", "两阶段互补设计在FrozenLake上较最强潜基线提升+9.5%，9个视觉基准平均提升+5.2%"]
benchmarks: ["FrozenLake", "V★", "BLINK", "MMVP", "MMStar", "CVBench", "HRBench-4K", "HRBench-8K", "MME-RealWorld-Lite", "Jigsaw"]
---

# 论文速读：Scafolding-Minds-Optimizing-Latent-Visual-Target-Representat

## 一句话总结
本文针对多模态视觉推理中潜变量推理的两阶段训练范式，分别改进了 SFT 阶段的潜在目标表示和 RL 阶段的探索机制：提出可学习的 scaffolding encoder 以端到端优化潜目标，并设计 Scafolding RL 通过输入自适应的高斯策略对残差潜动作进行采样，二者互补地提升了 VLM 的视觉推理能力。

## 研究问题与动机
1. **SFT 阶段潜目标次优**：现有方法使用冻结的现成视觉编码器（如 SigLIP）将辅助图像编码为潜目标，该编码器面向通用视觉表征而非下游推理任务，保留了与推理无关的感知内容，导致潜在生成的表示无法充分承载推理所需的关键证据。
2. **RL 阶段缺乏潜空间探索**：现有 RL 方法要么仅优化文本轨迹（无直接潜变量梯度），要么采用固定方差的确定性高斯正则化（如 VLPO），不采样替代潜动作，奖励信号无法驱动潜空间的探索，收益小且不稳定。
3. **两阶段问题的共同根源**：潜目标未针对推理任务优化，且 RL 未在连续潜空间中进行有效探索，二者均限制了最终推理性能。

## 核心贡献（创新点）
1. **Scaffolding Encoder（Stage 1）**：用可学习的 scaffolding encoder 替代冻结现成视觉编码器，通过冻结 VLM 的下游任务损失端到端训练，使潜目标直接优化为推理敏感特征。与现有方法的本质区别：潜目标不再是通用视觉特征的简单投影，而是与下游任务联动的可学习组件。
2. **Scafolding RL（Stage 2）**：引入输入自适应的高斯策略对残差潜动作进行采样，均值和方差由两个轻量 MLP 头基于生成的潜先验预测。与 VLPO 等方法的本质区别：从确定性正则化转变为可采样的 rollout 分布，使奖励信号能直接优化潜空间探索。
3. **两阶段互补性验证**：证明 scaffolding encoder 与 Scafolding RL 高度互补，联合使用在 FrozenLake 上比最强潜推理基线提升 +9.5%（32×32 网格达 +19%），在 9 个视觉推理基准上平均提升 +5.2%。

## 方法详解
**整体框架**：两阶段训练，Stage 1 学习优化的潜目标表示，Stage 2 在潜空间进行奖励引导的探索。

**Stage 1：SFT via Scaffolding Encoder**
- **Scaffolding Phase**：将 scaffolding encoder $g_\psi^*$ 构建为 VLM 视觉编码器的可训练副本 + cross-attention pooling 模块，将辅助图像 h 压缩为 $K=4$ 个潜 token $\mathbf{z}^* = g_\psi^*(\mathbf{h}) \in \mathbb{R}^{K \times d}$，插入 VLM 后由冻结的 VLM 生成答案 y。优化目标为：$\mathcal{L}_{\text{scaffolding}}(\psi) = \mathcal{L}_{\text{CE}}(\mathbf{y} \mid \mathbf{x}, \mathbf{z}^*)$，即仅通过下游 CE 损失更新 scaffolding encoder。
- **Generation Phase**：冻结 scaffolding encoder，微调 VLM 参数 $\theta$，使其从输入 x 直接生成潜 token $\mathbf{z}$（通过可学习嵌入经 transformer 单次前向产生，非自回归）。目标函数：$\mathcal{L}_{\text{generation}}(\theta) = \lambda_{\text{latent}} \frac{1}{K}\sum_{k=1}^{K}\|\mathbf{z}_k - \mathbf{z}_k^*\|_2^2 + \lambda_{\text{task}}\mathcal{L}_{\text{CE}}(\mathbf{y} \mid \mathbf{x}, \mathbf{z})$，其中 $\lambda_{\text{latent}}=1.0, \lambda_{\text{task}}=0.5$。

**Stage 2：Scafolding RL**
- 以 Stage 1 生成的潜先验 $\mathbf{z}^\theta$ 为基动作，学习自适应调整 $\Delta_z$：$\Delta_z \sim \mathcal{N}(\mu(\mathbf{z}^\theta), \sigma(\mathbf{z}^\theta)^2)$，扰动后的潜块 $\mathbf{z}^\theta + \Delta_z$ 替换原潜块参与前向传播。
- 均值头和方差头为双层 MLP，均值头零初始化，方差头初始化为 $\sigma=0.05$，确保初始探索围绕先验分布。
- RL 目标采用 GRPO 风格 clipped 目标，同时包含潜 block 级别重要性比 $\rho_i^{\text{lat}}$ 和文本 token 级别比率 $\rho_{i,t}^{\text{text}}$，加上 KL 惩罚项 $\beta_{\text{KL}}=0.001$（公式见附录 B）。
- rollout group size $G=8$，clipping $\epsilon=0.2$，学习率 $1\times10^{-6}$。

## 实验与结果
**数据集**：
- **FrozenLake**：网格空间规划任务，8×8 至 32×32 共 4,550 个训练样本，每 level 100 个测试样本。
- **9 个视觉推理基准**：V★、BLINK、MMVP、MMStar、CVBench、HRBench-4K/8K、MME-RealWorld-Lite、Jigsaw，训练数据为开源多模态推理数据混合。

**评估基线**：
- 监督基线：SFT、SFT+GRPO
- 潜推理方法：LVR、Mirage、CoVT、Monet、VaLR、SkiLa
- 图像生成方法：VPRL、DiffThinker
- 图像思考方法：DeepEyes、Thyme

**主要结果**：
- **FrozenLake**：Scaffolding Encoder+RL 平均准确率 75.0%，较最强潜基线 VaLR（65.5%）提升 **+9.5%**；32×32 最难 level 提升 **+19%**（55.0% vs 36.0%）；较 SFT 基线提升 +23.0%。
- **9 个视觉基准**：Scaffolding Encoder+RL 平均准确率 **73.9%**，较最强潜基线 VaLR（68.7%）提升 **+5.2%**；较 base model（Qwen2.5-VL-7B，61.1%）提升 +12.8%。
- **关键单点提升**：MMVP +20.7%、MME-RealWorld-Lite +17.6%、CVBench +13.9%、V★ +14.4%。

**消融结论**：
- Stage 1：切换目标函数（CE → L2）+4.3%，微调视觉编码器 +10.0%，二者共同贡献 +14.3% 的相对增益。
- Stage 2：Scafolding RL（+3.0%）> VLPO（+1.8%）> No Sampling（+0.6%）。
- 无 scaffolding encoder 时 RL 增益更大（+5.2%）但最终精度低，说明 Stage 1 奠定基础、Stage 2 进一步优化，二者互补。
- 超参：$K=4$ 为最优潜 token 数；value-function heatmap 辅助图配合 scaffolding encoder 效果最佳。

**推理效率**：单 A100 延迟 272ms，较 base 仅增加 +4.6%，显著低于图像生成方法（+108%~+277%）。

## 相关工作脉络
1. **Think about Image**：将视觉证据转化为文本链式推理（如 Visual CoT），受限于离散 token 表达能力，难以保留细粒度空间/感知证据；本文用连续潜 token 直接插入推理链，避免信息损失。
2. **Think with Image / 图像生成方法**：如 VPRL、DiffThinker、DeepEyes 等在推理中主动生成或操作中间图像，推理成本高（3× 以上延迟）；本文在潜空间操作，以 <5% 延迟开销获得可比甚至更强的性能。
3. **Latent Reasoning（潜推理）**：LVR、Mirage、CoVT、Monet、VaLR、SkiLa 等均使用冻结现成视觉编码器的特征作为潜目标；本文的核心差异是将潜目标本身变为可学习组件，端到端优化为目标服务质量。
4. **VLPO（Wang et al., 2026）**：使用固定方差高斯对确定性潜输出做正则化，不采样潜动作；本文的 Scafolding RL 通过可采样的输入自适应高斯策略实现真正的潜空间探索。
5. **Joint-Embedding Predictive Architectures（LeCun, 2022）** 与 **Learning Using Privileged Information（Vapnik & Vashist, 2009）**：本文方法与这两个理论框架在理念上相通——利用训练时的辅助信息（helper image）学习更优的表征，推理时仅依赖主输入。

## 局限性与未来方向
1. **依赖辅助图像**：训练阶段仍需要辅助图像（helper image），在辅助图像不可用或获取成本高的场景下框架受限；如何泛化到无辅助图像 setting 是开放问题。
2. **辅助图设计未充分探索**：FrozenLake 使用 value-function heatmap 作为默认辅助图，但未系统搜索其他辅助图设计；自动为新领域生成辅助图留待未来工作。
3. **高分辨率/组合推理仍有提升空间**：HRBench-4K（+8.9%）和 Jigsaw（+9.3%）的相对增益较小，暗示极高空间分辨率或组合推理任务可能需要更强或更多样的辅助图。
4. **模型规模与微调方式**：LoRA 版本的绝对性能低于全参微调，需进一步探索高效微调与潜推理的结合。

## 研究启发与可借鉴点
1. **潜目标的可学习性**：将"冻结编码器特征"替换为"任务驱动的端到端可学习目标"是一个通用范式，可迁移至其他潜变量推理或蒸馏场景（如文本 latent reasoning、音频 latent tokens）。
2. **输入自适应探索策略**：均值/方差对潜先验条件化的 Gaussian policy 设计，相比固定方差正则化能实现更稳定的 reward-based 探索，可推广至其他连续动作空间的 RL 训练。
3. **两阶段互补的实验范式**：通过消融验证 Stage 1 与 Stage 2 的互补性（即便最弱 Stage 2 + Stage 1 也优于最强 Stage 2 无 Stage 1），为后续工作的模块化设计提供了清晰的评估思路。
4. **推理效率分析的价值**：报告了从 base VLM 到含潜推理的完整延迟预算（+4.6% overhead），为实际应用部署提供了参考基准。
5. **helper image 设计的可扩展性**：证明了非自然图像（如 value-function heatmap）配合可学习 encoder 可以大幅提升性能，启发了未来对辅助信息形式的系统性探索。

## 关键术语表
**Latent Visual Reasoning**：在 VLM 的推理链中插入连续潜视觉 token（而非文本或显式图像），使模型能以"视觉思维"方式保留细粒度感知证据。
**Scaffolding Encoder**：替代冻结现成视觉编码器的可学习模块，通过下游任务损失端到端优化，将辅助图像编码为推理敏感的潜目标表示。
**Scafolding RL**：在 Stage 1 潜先验基础上，通过输入自适应的高斯策略采样残差潜动作，使奖励信号能直接驱动潜空间的探索。
**Helper Image**：仅在训练时提供的中间图像（如标注裁剪图、热力图），用于为潜推理提供监督信号，推理时不再需要。
**GRPO（Group Relative Policy Optimization）**：基于组内相对优势的策略梯度优化方法，本文将其扩展至同时处理潜 token block 和文本 token。
**VLPO（Visual-Latent Policy Optimization）**：现有潜推理 RL 方法，使用固定方差高斯对确定性潜输出做正则化，不创建替代潜轨迹。
**FrozenLake**：网格空间规划基准，智能体需在含陷阱的网格中找到安全路径，考验多步空间推理能力。

## 可复现要素
- **数据集**：FrozenLake（VSP 套件）、V★、BLINK、MMVP、MMStar、CVBench、HRBench-4K/8K、MME-RealWorld-Lite、Jigsaw；Zebra-CoT 训练数据含配对辅助图像。论文未声明公开数据集的链接，但训练数据为开源多模态推理数据混合。
- **代码/权重**：论文未声明代码或权重是否开源。
- **关键超参**：潜 token 数 $K=4$；$\lambda_{\text{latent}}=1.0, \lambda_{\text{task}}=0.5$；AdamW lr=$2\times10^{-5}$（latent generation）/$5\times10^{-5}$（scaffolding encoder）；GRPO：$G=8, \epsilon=0.2, \text{lr}=1\times10^{-6}, \beta_{\text{KL}}=0.001$；方差头初始 $\sigma=0.05$；均值头零初始化。
- **Backbone**：Qwen2.5-VL-7B（主实验），Qwen2.5-VL-3B（规模消融）。
- **硬件**：8× NVIDIA A100（80GB），Stage 1 约 7h，Stage 2 约 3h，总计 ~10h。
