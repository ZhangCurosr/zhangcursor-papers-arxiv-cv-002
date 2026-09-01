---
title: "Scafolding-Minds-Optimizing-Latent-Visual-Target-Representat"
source: https://arxiv.org/pdf/2608.19669v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 19:25:39"
field: "多模态推理与潜视觉推理"
keywords: ["latent visual reasoning", "multimodal reasoning", "scaffolding encoder", "reinforcement learning", "visual chain-of-thought", "FrozenLake", "VLM"]
innovations: ["用端到端可学习的 scaffolding encoder 替代冻结通用视觉编码器生成任务优化的潜目标", "在潜 prior 之上学习输入自适应高斯策略对潜残差动作采样以实现奖励驱动的显式潜探索"]
benchmarks: ["FrozenLake", "V★", "BLINK", "MMVP", "MMStar", "CVBench", "HRBench-4K", "HRBench-8K", "MME-RealWorld-Lite", "Jigsaw"]
---

# 论文速读：Scaffolding Minds: Optimizing Latent Visual Target Representations for Multimodal Reasoning

## 一句话总结
本文针对潜视觉推理的两阶段训练范式（SFT + RL）中的关键缺陷，提出 **Scaffolding Minds**：用端到端可学习的 **scaffolding encoder** 替代冻结的通用视觉编码器以生成任务优化的潜目标，并用输入的自适应高斯策略对潜残差动作进行采样以实现奖励驱动的显式探索；两者互补，在 FrozenLake 空间规划上较最强基线平均提升 **+9.5%**（32×32 格网提升 +19%），在九个以视觉为核心的推理基准上平均提升 **+5.2%**。

---

## 研究问题与动机

1. **SFT 阶段的潜目标次优**：现有方法使用冻结 off-the-shelf 通用视觉编码器（如 SigLIP）对训练时辅助图像（helper image）编码作为潜目标，该目标面向通用视觉表征而非下游推理任务，可能保留无关感知内容而未能突出中间推理证据。
2. **RL 阶段缺乏潜空间的显式探索**：现有 RL 方法要么仅更新文本 token（无潜 action 的直接似然/信用分配），要么采用固定方差的高斯形式作为确定性正则（如 VLPO），仅约束潜输出漂移，并不采样替代潜轨迹用于探索。
3. **两个缺陷的同根因**：潜目标未针对推理任务优化，且 RL 无法直接在潜空间探索替代推理路径；二者缺一不可，共同导致潜视觉推理的上限受限。
4. **核心假设**：将潜目标本身变为端到端优化的系统组件，并配合输入自适应的随机潜采样策略，可使奖励信号在更丰富的潜空间中进行信用分配与探索。

---

## 核心贡献（创新点）

1. **提出 Scaffolding Encoder**：用可学习编码器替代冻结通用视觉编码器，通过下游任务 loss 端到端优化潜目标 $\mathbf{z}^*$，使目标表征服务于推理而非通用视觉特征。与已有工作本质区别：前作把潜目标视为"固定特征投影"，本文将其变为与 VLM 联合优化的端到端模块。
2. **提出 Scaffolding RL**：在潜 prior $\mathbf{z}^\theta$ 之上学习输入自适应的高斯策略 $\Delta_z \sim \mathcal{N}(\mu(\mathbf{z}^\theta), \sigma(\mathbf{z}^\theta)^2)$，对潜残差动作直接采样并在 rollout 中使用，实现显式潜探索。与 VLPO 等确定性正则的本质区别：前者采样替代潜 action 并评估，后者仅对确定输出施加固定方差的高斯惩罚。
3. **证明两阶段互补性**：Scaffolding Encoder 提供高质量起始点，Scaffolding RL 在其基础上进一步挖掘，二者叠加产生显著增益；单独使用任一模块效果劣于联合使用，且差的 Stage 1 无法被好的 Stage 2 弥补。
4. **全面的实验验证**：在结构化空间规划（FrozenLake, 8×8 至 32×32）与九个以视觉为核心的推理基准上均稳定超越最强潜推理与图像生成基线，并分析潜 token 数、辅助图像类型、模型规模与 LoRA 微调的可迁移性。

---

## 方法详解

### 总体框架：两阶段训练

- **Stage 1 (SFT via Scaffolding Encoder)**：学习优化的潜目标 $\mathbf{z}^*$，再用其监督 VLM 从输入 $x$ 中生成自身潜块 $z$。
- **Stage 2 (Scaffolding RL)**：在 Stage 1 的潜 prior $\mathbf{z}^\theta$ 之上，用学习的高斯策略采样残差 $\Delta_z$，通过 GRPO-style 目标联合优化文本与潜块。

### Stage 1.1 — Scaffolding Phase

- 架构：scaffolding encoder $g_\psi^*$ = 可训练的 VLM 视觉编码器副本 + cross-attention pooling 模块，将辅助图像 $h$ 压缩为 $K=4$ 个潜 token $\mathbf{z}^* = g_\psi^*(h) \in \mathbb{R}^{K \times d}$。
- 冻结基座 VLM，仅更新 $\psi$，通过标准 token-level 交叉熵损失端到端训练：
  $$\mathcal{L}_{\text{scaffolding}}(\psi) = \mathcal{L}_{\text{CE}}(y \mid x, \mathbf{z}^*)$$
- 直觉：让 encoder 学习恰好捕捉"VLM 用于正确作答所需的推理敏感特征"，而非泛用视觉特征。

### Stage 1.2 — Generation Phase

- 冻结 $g_\psi^*$，解冻 VLM（参数 $\theta$），从输入 $x$ 单独预测潜块 $z$（非自回归，由 $K$ 个 learnable embedding 经 VLM transformer 一次性生成）。
- 损失函数：
  $$\mathcal{L}_{\text{generation}}(\theta) = \lambda_{\text{latent}} \frac{1}{K}\sum_{k=1}^{K} \|z_k - z_k^*\|_2^2 + \lambda_{\text{task}} \mathcal{L}_{\text{CE}}(y \mid x, z)$$
- 默认 $\lambda_{\text{latent}}=1.0,\; \lambda_{\text{task}}=0.5$；纯 matching 或纯 task loss 均弱于联合。

### Stage 2 — Scaffolding RL

- 将 Stage 1 生成的潜 prior $\mathbf{z}^\theta$ 作为 base action，学习输入自适应的高斯调整：
  $$\Delta_z \sim \mathcal{N}(\mu(\mathbf{z}^\theta), \sigma(\mathbf{z}^\theta)^2)$$
- 均值头零初始化，方差头初始化为 $\sigma=0.05$，确保初始分布中心于 prior。
- 得到扰动潜块 $\mathbf{z}^\theta + \Delta_z$，用于后续 forward pass。
- 组合 GRPO-style clipped 目标（含潜重要性比值、文本重要性比值与 KL 惩罚）：
  $$\mathcal{L}_{\text{RL}} = -\frac{1}{G}\sum_{i=1}^{G}\left[\text{clip}(\rho_i^{\text{lat}} A_i) + \frac{1}{T_i}\sum_{t}\text{clip}(\rho_{i,t}^{\text{text}} A_i)\right] + \frac{\beta_{\text{KL}}}{G}\sum_{i,t} D_{\text{KL}}(\pi_\theta\|\pi_{\text{ref}})$$
- Rollout group size $G=8$，clip $\epsilon=0.2$，$\beta_{\text{KL}}=0.001$，学习率 $1\times10^{-6}$。
- 推理时：scaffolding encoder 丢弃，仅使用 VLM + 两个轻量 MLP 调整头。

---

## 实验与结果

### 数据集与设置

- **FrozenLake**：VSP 基准，8×8 至 32×32 格网迷宫，训练 4,550 样本（Level 8–32 偶数级），100 测试样本/级；所有图像 pad 至 32×32。
- **九个视觉为中心基准**：V★, BLINK, MMVP, MMStar, CVBench, HRBench-4K, HRBench-8K, MME-RealWorld-Lite, Jigsaw。
- 骨干模型：**Qwen2.5-VL-7B**；训练环境 8×A100 (80GB)。

### 主要结果（FrozenLake 空间规划，Table 1）

- Scaffolding Encoder+RL 平均精度 **75.0%**，对比最强潜基线 VaLR (65.5%) 提升 **+9.5%**；对比 SFT 提升 **+23.0%**。
- 难度越高增益越显著：Level 32 (32×32) 上相对 VaLR 提升达 **+19%**。
- 对比图像生成方法（DiffThinker 64.3%、VPRL 58.3%），本方法仍领先 **+10.7%**。

### 主要结果（视觉为中心基准，Table 2）

- Scaffolding Encoder+RL 平均精度 **73.9%**，对比 Qwen2.5-VL-7B base (+12.8%)；对比最强潜基线 VaLR (68.7%) 提升 **+5.2%**。
- 对比 think-with-images 方法（Thyme 67.2%、DeepEyes 未给出完整均值），平均提升 **+6.2%**。
- 在 MMVP 提升 **+20.7%**、MME-RealWorld-Lite 提升 **+17.6%**、CVBench 提升 **+13.9%**；在 CVBench、HRBench-4K、HRBench-8K 取得 best。
- 较小增益出现在 HRBench-4K (+8.9%)、Jigsaw (+9.3%)，暗示高分辨率/组合空间推理可能需要更强的辅助图像。

### 消融

- **Stage 1 视觉目标消融**：Downstream task loss + cross-attention pooling (+4.3%) + 进一步 tune vision encoder (+10.0%)，共 +14.3% 相对 Mean Pooling (No Opt.)。
- **Stage 2 RL 消融**：No Sampling on Latent (+0.6%) < VLPO (+1.8%) < Scaffolding RL (+3.0%)，采样学习调整是关键。
- **两阶段互补**：有 Scaffolding Encoder 时任何 RL 方法都优于无 Scaffolding Encoder 时的最佳 RL；差的 Stage 1 不能被好 Stage 2 弥补。
- **超参**：最佳潜 token 数 $K=4$；value-function heatmap 作为 helper image 配合 scaffolding encoder 效果最佳（+12.0% vs frozen encoder）。

---

## 相关工作脉络

1. **Think about Image（语言链式推理扩展）**：Kojima et al. (2022); Wei et al. (2022) 等将 CoT 扩展到 VLM（如 Qwen2.5-VL、LLaVA、Gemini），将视觉证据转写为文本推理链。本文与之区别：不依赖离散文本表征，直接插入连续潜视觉 token 保留细粒度空间/感知证据。
2. **Think with Image（图像生成/操作）**：DeepEyes、Thyme、VPRL、DiffThinker 等方法在推理过程中调用工具或生成中间图像以保留视觉证据，但推理开销 >3×。本文与之区别：全程在潜空间操作，推理开销仅 +4.6%。
3. **Latent Reasoning（潜推理前作）**：LVR、Mirage、CoVT、Monet、VaLR、SkiLa 等使用冻结 off-the-shelf 视觉编码器（如 SigLIP）从辅助图像生成潜目标。本文与之区别：将潜目标替换为端到端优化的 scaffolding encoder，直接通过下游任务 loss 驱动。
4. **连续推理空间与潜扩散**：Hao et al. (2024) 的 LaDiR 系列探索连续潜空间与 RL；本文在**多模态**设定下推进同一思路，但聚焦于潜目标的优化与显式潜采样探索。
5. **联合嵌入预测架构**：LeCun (2022) JEPAs；本文方法在方法论上与 JEPAs 理念一致——将中间表征视为可学习系统组件，而非固定蒸馏目标。
6. **Privileged Information / Fading Scaffolds**：Vapnik & Vashist (2009) 与 Vygotsky (1978) 的教育心理学思想——训练时引入额外"特权"信息帮助学习，推理时逐步消去；本文用 scaffolding encoder 实现这一理念。

---

## 局限性与未来方向

1. **依赖训练时辅助图像**：与同类潜推理方法一样，Scaffolding Minds 仍需要训练阶段的 helper image（如 value-function heatmap、标注裁剪等）；如何在无辅助图像或辅助图像稀缺/昂贵的场景下扩展仍是开放问题。
2. **辅助图像设计未充分搜索**：本文仅对 FrozenLake 使用了 value-function heatmap 与 red-arrow overlay 两种，自动为不同领域推荐/生成合适的辅助图像留待未来。
3. **高分辨率/组合推理增益有限**：HRBench-4K/8K 和 Jigsaw 上相对增益较小（+8.9% / +9.3%），暗示当前 setup 对极致高分辨率与复杂空间组合推理仍存瓶颈。
4. **模型规模与微调方式的泛化虽已验证但未穷尽**：论文验证了 3B 全参数与 LoRA (r=32) 下的有效性，但更大规模或更激进的高效微调策略下的行为仍有待探索。

---

## 研究启发与可借鉴点

1. **"潜目标端到端优化"范式可迁移**：在任意需要引入中间潜表示的多模态/文本任务中（如潜规划 token、潜 memory token、潜 memory compression），均可借鉴"下游 task loss 反向驱动潜编码器"的设计，避免固定特征的次优约束。
2. **输入自适应的高斯潜采样策略**：Scaffolding RL 的"mean/variance 均为 latent prior 的函数"设计可用于任何连续潜 action 空间的 RL 优化场景，尤其适合与 GRPO/PPO 结合；零初始化 mean + 小方差初始化的 trick 值得复用到其他潜策略研究中。
3. **Stage 1 → Stage 2 的"脚手架渐退"思想**：与认知科学中的 fading scaffold 一致——训练时借助外部辅助表征，推理时逐步消去；可类比应用于长上下文压缩、memory bank、tool-calling 决策等多阶段学习任务。
4. **隐式替代显式图像生成的性价比**：本文证明高质量潜表征可匹敌甚至超越图像生成方法的推理性能且开销低一个数量级；在资源受限部署场景下，"潜表征替代显式中间产物"是值得优先探索的路径。
5. **Loss 权重敏感性分析**：纯 matching 与纯 task loss 均劣于联合（表 5），提示在多目标潜学习框架中权重调度需精细设计；可借鉴到蒸馏、对抗训练、multi-task learning 等场景。

---

## 关键术语表

**Scaffolding Encoder**：可学习的辅助编码器，通过下游任务 loss 端到端训练，将训练时辅助图像压缩为任务优化的潜目标表示。
**Scaffolding RL**：在 Stage 1 潜 prior 之上学习输入自适应高斯策略，对潜残差动作直接采样以实现奖励驱动的显式潜探索。
**Helper Image（辅助图像）**：仅在训练时可见的中间图像（如裁剪区域、价值函数热力图、标注箭头），用于提供潜目标的监督信号，推理时不使用。
**Latent Visual Reasoning（潜视觉推理）**：在 VLM 推理链中插入连续潜视觉 token 块以保留细粒度视觉证据的推理范式。
**GRPO（Group Relative Policy Optimization）**：Shao et al. (2024b) 提出的基于组内相对优势的策略优化算法，本文用于文本与潜块的联合 RL 更新。
**VLPO（Visual-Latent Policy Optimization）**：Wang et al. (2026) 提出的潜视觉 RL 方法，使用固定方差高斯对确定式潜输出进行正则，不采样潜 action。
**FrozenLake**：格网迷宫空间规划基准，要求代理输出避开障碍到达目标的最优路径，测试多步空间推理能力。
**Cross-Attention Pooling**：scaffolding encoder 中用于将视觉 token 压缩为 $K$ 个潜 token 的模块，在 Scaffolding Phase 中与下游任务 loss 联合训练。

---

## 可复现要素

| 项目 | 详情 |
|------|------|
| 骨干模型 | Qwen2.5-VL-7B（开源） |
| 数据集 — FrozenLake | VSP 基准（训练 4,550 样本，测试 100/级）；论文未声明公开链接 |
| 数据集 — 视觉基准 | V★, BLINK, MMVP, MMStar, CVBench, HRBench-4K/8K, MME-RealWorld-Lite, Jigsaw（各自原始仓库） |
| 辅助图像 | FrozenLake 使用 value-function heatmap 与 red-arrow overlay（附录提供 Python 代码生成）；视觉基准使用 Zebra-CoT 自带的 paired helper image |
| 代码/权重开源 | 论文未明确声明开源（arXiv:2608.19669v1），截至论文发表时未附代码链接 |
| 关键超参 | $K=4$；$\lambda_{\text{latent}}=1.0,\;\lambda_{\text{task}}=0.5$；AdamW LR $2\times10^{-5}$（latent gen）/ $5\times10^{-5}$（scaffolding enc）；GRPO $G=8,\;\epsilon=0.2$，LR $1\times10^{-6}$，$\beta_{\text{KL}}=0.001$；mean head 零初始化，var head $\sigma=0.05$ |
| 训练设备 | 8×A100 (80GB)；Stage 1 ≈7h，Stage 2 ≈3h/配置 |

---
