---
title: "VA-JUDGER-REWARD-MODELING-FROM-HUMAN-PREFERENCE-FEEDBACK-FOR"
source: https://arxiv.org/pdf/2608.18607v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 16:31:19"
field: "多模态生成与对齐"
keywords: ["reward modeling", "human preference", "joint video-audio generation", "chain-of-thought", "reinforcement learning", "omni-reward model", "curriculum learning"]
innovations: ["提出首个面向联合视频-音频生成的人类偏好奖励模型VA-Judger", "构建VAPref-10K大规模人类偏好数据集并提出难度感知的三阶段课程学习训练策略", "设计维度级验证的GRPO强化学习方法以实现细粒度跨模态推理对齐"]
benchmarks: ["VA-Judger-Bench", "JavisBench"]
---

# 论文速读：VA-JUDGER: REWARD MODELING FROM HUMAN PREFERENCE FEEDBACK FOR JOINT VIDEO-AUDIO GENERATION

## 一句话总结
论文提出了 **VA-Judger**，一个面向联合视频-音频生成的链式思维全模态奖励模型，通过构建大规模人类偏好数据集 **VAPref-10K**、三阶段课程学习（简单对冷启动 → 困难对人类验证对齐 → 维度级强化学习），首次利用人类反馈为视频-音频生成任务训练出与人类判断高度对齐的 omni-reward 模型，并用其成功后训练 LTX-2 提升生成质量。

## 研究问题与动机
1. **RL 后训练缺乏高质量奖励信号**：联合视频-音频生成模型的 RL 后训练需要一个与人类感知对齐的奖励函数，现有方法多依赖多个独立专家指标（如 VideoAlign、AudioBox、CLAP、Synchformer）的组合，但这些指标评估各感知维度时相互割裂。
2. **自动指标与人类判断存在显著偏差**：这些指标能捕捉单一属性（如视觉质量或同步偏移），却无法捕捉文本-视频-音频整体的语义连贯性和跨模态时序一致性，导致"高指标分 ≠ 人类偏好"，甚至引发 reward hacking。
3. **缺乏面向联合视频-音频的人类偏好数据**：现有偏好数据集集中于图像（Pick-a-Pic、ImageReward）或纯视频（VideoReward、LiFT）领域，未涉及音频，且单模态评估模型无法捕捉跨模态交互。
4. **直接迁移已有 reward model 不可行**：图文/视频 reward model 不观测音频流，无法判断声音是否与事件匹配；纯音频评估缺乏视觉语境，无法判断听觉合理性。

## 核心贡献（创新点）
1. **首个面向联合视频-音频生成的奖励建模框架**：提出 VA-Judger，以链式思维输出五维度结构化评分，首次用人类偏好反馈训练跨模态 omni-reward 模型，区别于仅优化单一模态或拼凑多指标的方法。
2. **构建大规模人类偏好数据集 VAPref-10K**：采集 9K 真实场景 prompt 和 10.3K 细粒度成对比较，通过难度感知的数据构建策略（易/难池分离）解决训练数据稀缺问题，区别于依赖合成 prompt 或单一质量差距的现有做法。
3. **三阶段课程学习 + 维度级 GRPO**：先以 Gemini 生成 easy 对结构化 CoT 建立输出格式；再以人类标注为锚点做 rejection sampling 对齐 hard 对；最后用维度级验证的 GRPO 强化细粒度推理，区别于仅用二元偏好标签或单一 SFT 的方法。
4. **提出 VA-Judger-Bench 评测基准**：包含 in-domain 与 out-of-domain 两个子集（含 Kling、Wan、Veo、Sora 等闭源模型），首次系统评估 reward model 对跨模型人类偏好的泛化能力。

## 方法详解
- **任务形式化**：视频-音频奖励建模定义为成对偏好预测 + 结构化维度推理。输入为文本 prompt $p$ 和两个视频-音频片段 $x_1=(v_1,a_1), x_2=(v_2,a_2)$，模型输出五维度 1-10 评分及解释，最后聚合为总分并输出 `<answer>video k is better</answer>`。
- **数据集构建 VAPref-10K**：从 YouTube/Bilibili/电影/TVS 收集 10,173 个真实视频片段，按六类（多人对话、旁白、音乐、电影氛围、环境事件、讲座）组织；用 Qwen3.5-Omni 生成字幕后用 LLM 改写为无时间戳的 coherent 文本 prompt，得到 10,083 个 prompt。易池（4.4K 对）：OVI vs LTX-2（质量差距明显）；难池（9.8K 对）：同模型同 prompt 不同随机种子生成（差距微小）。
- **Stage 1：Easy Cold-Start SFT**：用 Gemini 3.1 Pro 生成结构化 CoT 答案（其在 200 对 pilot 中与人类标注一致率达 80%），训练语言骨干学习输出格式和粗粒度偏好判断。
- **Stage 2：Hard Preference Alignment（拒绝采样）**：Gemini 在 200 对 hard 对中仅与人类一致 60%，因此用人类标注的偏好和支撑维度作为锚点，保留 Gemini 生成且最终偏好与人类一致的答案，得到 4.5K 经人类验证的难对样本，训练模型关注细粒度跨模态差异。
- **Stage 3：Dimension-wise GRPO**：对每组 $G=8$ 个采样响应，奖励函数为：
  $$r_{\text{ans}}(o) = \mathbf{1}[\hat{y} = y^{\star}], \quad r_{\text{dim}}(o) = \frac{1}{|\mathcal{D}_{\mathrm{h}}|}\sum_{d \in \mathcal{D}_{\mathrm{h}}} \mathbf{1}[s_{d,y^{\star}} > s_{d,3-y^{\star}}]$$
  组合奖励 $R_i = \frac{1}{2}(r_{\text{ans}} + r_{\text{dim}})$，再用标准 GRPO clipped surrogate objective 更新策略，惩罚仅猜对答案但维度推理不一致的响应。

## 实验与结果
- **数据集与评测基准**：VAPref-10K（9K prompt，10.3K 成对比较）；VA-Judger-Bench（1150 对：400 easy + 250 in-domain + 500 out-of-domain，out-of-domain 含 Kling 2.6、Wan 2.6、Veo 3.1、Sora 2 等闭源模型输出）。
- **Reward Model 评测（Table 1）**：
  - 单维度自动指标（VideoAlign、AudioBox、CLIP、SynchFormer 等）整体准确率仅 50.43%~56.88%，在 out-of-domain 上降至 44%~48%。
  - VA-Judger（Final GRPO）在 Easy/In-domain/Out-of-domain/Overall 上分别达到 **76.25% / 66.00% / 63.40% / 68.43%** Total Acc，较 Qwen3-Omni CoT 基线提升 **10.60 pp**。
- **Post-training LTX-2（Table 2 & Figure 4）**：
  - 在 JavisBench 200 prompt 子集上，VA-Judger 后训练 LTX-2 在 11/13 个视频质量指标和 6/7 个音频/跨模态指标上排名第一。
  - 相比 LTX-2 基座：VQ +1.693、MQ +0.486、AudioBox AQ +0.843、JavisScore 0.074→0.230。
  - 相比 OmniNFT：VQ +0.214、AVHScore +0.114、DeSync 下降最多（-0.366，即同步改善最多）。
  - **人工评估（200 对三元对比）**：VA-Judger 后训练模型获得 **62.30%** 人类偏好率，远高于 OmniNFT（27.63%）和 LTX-2 基座（10.08%），约为 OmniNFT 的两倍、基座的六倍。
- **结论**：VA-Judger 在预测人类偏好上显著优于所有自动指标和已有 reward model；作为 post-training 奖励信号可有效提升生成质量且避免 reward hacking。

## 相关工作脉络
1. **OmniNFT（Zhang et al., 2026）**：当前唯一将扩散 RL 扩展到多模态生成的工作，但使用 VideoAlign、AudioBox、CLAP、Synchformer 等独立专家指标作为奖励，依赖窄目标优化而非人类判断，易导致 reward hacking。
2. **VideoReward / LiFT / InstructVideo**：面向纯视频生成的人类偏好 reward model，不观测音频流，无法评估声音与事件的匹配性，不能直接迁移至联合视频-音频场景。
3. **ImageReward / Pick-a-Pic / HPSv2 / MPS**：面向图像生成领域的 reward model，缺失音频和时序维度，不适用于视频-音频联合生成。
4. **UnifiedReward / IXC-2.5-Reward**：多模态理解 reward model，未显式建模文本-视频-音频事件的跨模态连贯性，也非作为生成模型 post-training 的 omni-reward。
5. **FlowGRPO / DanceGRPO / DiffusionNFT**：面向文本到图像/视频的单模态扩散 RL 方法，未涉及音频模态的联合优化。
6. **JavisBench（Liu et al., 2025c）**：自动评估基准，涵盖音视频生成多个维度指标，但本质仍是自动指标聚合，本文将其用于 post-training 评测并发现人类偏好与自动指标存在显著偏差。

## 局限性与未来方向
1. **数据集来源相对单一**：prompt 主要来自 YouTube/Bilibili/影视片段，类别虽有六类但场景仍偏自然视频，缺乏动画、二次元、AR/VR 等特殊风格分布。
2. **LLM 辅助标注存在偏差**：Stage 1 使用 Gemini 3.1 Pro 生成 CoT 答案（仅在 easy 对上使用），其对 hard 对的判断与人类仅 60% 一致，虽经拒绝采样缓解，但仍可能残留 LLM 偏见。
3. **推理计算开销较大**：每个 prompt 需生成 8 个候选视频-音频对并进行多维度结构化推理，inference 成本较高，限制了其在实时或资源受限场景的应用。
4. **模型仅更新 LoRA 适配器**：post-training 中 LTX-2 骨干冻结仅优化 LoRA，可能限制了奖励模型对生成架构深层改进的空间。
5. **未来可探索方向**：将维度级奖励机制扩展至其他多模态生成任务（如视频编辑、3D 生成）；探索更高效的人类偏好数据标注策略；研究如何在不开源闭源模型的情况下继续提升 reward model 的泛化性。

## 研究启发与可借鉴点
1. **难度感知的课程学习策略**：将训练数据按质量差距大小分为易/难池，先用清晰差距对建立格式和粗判能力，再用人手验证的难对细化判断，可迁移至任何需要结构化输出的多模态评估/奖励任务。
2. **拒绝采样对齐人类偏好**：在大模型生成难以完全信赖的场景下，用少量人工标注作为锚点筛选模型生成答案，兼顾效率与准确性，是一种可扩展的对齐范式。
3. **维度级验证的强化学习**：在 GRPO 中将奖励分解为答案级（binary）+ 维度级（multi-dim scoring ordering）双重验证，避免模型通过"猜对答案但推理荒谬"获得高 reward，这一设计可推广至任何需要多步结构化推理的任务。
4. **跨模态 coherency 的显式建模**：将"提示匹配、视听一致性、音频质量、视频质量、完整性"作为五维 rubric，迫使模型显式思考跨模态交互，而非输出隐式标量，可为多模态评测标准制定提供参考。
5. **post-training 的 LoRA 冻结骨干策略**：冻结基础生成模型仅训练 LoRA，在保证生成能力不被破坏的同时高效引入人类偏好对齐，适合资源受限的快速迭代场景。

## 关键术语表
**VA-Judger**：一种面向联合视频-音频生成的链式思维全模态奖励模型，输出五维度结构化评分和最终偏好判断。
**VAPref-10K**：大规模人类偏好数据集，包含 9K 真实 prompt 和 10.3K 细粒度成对比较，用于训练联合视频-音频 reward model。
**VA-Judger-Bench**：评测基准，包含 in-domain 和 out-of-domain 两个子集，用于评估 reward model 与人类偏好的对齐程度及跨模型泛化能力。
**维度级 GRPO（Dimension-wise GRPO）**：在 GRPO 框架下引入答案级和维度级双重验证奖励的强化学习方法，确保模型推理与最终偏好一致。
**OmniNFT**：当前联合视频-音频生成领域唯一的 RL post-training 方法，依赖多个独立专家指标（VideoAlign、AudioBox 等）作为奖励信号。
**CoT（Chain-of-Thought）**：链式思维推理，模型在输出最终答案前先生成逐步推理过程的结构化解释，本文用于输出五维度评分及理由。
**JavisBench**：联合视频-音频生成自动评测基准，包含视频质量、音频质量、跨模态一致性、同步等多个指标，本文取其 200 prompt 子集用于 post-training 评估。
**Reward Hacking**：优化模型对窄指标 overfitting，导致自动指标得分提升但人类感知质量下降的现象，本文认为这是现有指标化奖励的核心问题。

## 可复现要素
- **数据集**：VAPref-10K 已公开（论文声明 Code 链接）；VA-Judger-Bench 1150 对已公开。
- **代码/权重**：代码已开源（https://github.com/ShareLab-SII/VA-Judger）；论文未明确声明权重是否开源，需查看 GitHub 仓库确认。
- **关键超参**：
  - SFT 阶段：学习率 $5\times10^{-6}$，warmup 0.05，max seq len 24576，effective batch size 128，8×GPU，BF16，DeepSpeed ZeRO 3。
  - GRPO 阶段：学习率 $1\times10^{-6}$，temperature 1.0，G=8 responses/group，Adafactor 优化器。
  - Post-training LTX-2：LoRA rank=32、scaling=64，AdamW lr=$3\times10^{-5}$，weight decay=$10^{-4}$，gradient clip=1.0，6 GPU（FSDP），video loss 重加权 max weight=1.5，warmup=400 步。
- **模型初始化**：Qwen3-Omni-30B-A3B-Instruct（仅更新语言骨干，视觉 encoder 冻结）；LTX-2 19B checkpoint。
