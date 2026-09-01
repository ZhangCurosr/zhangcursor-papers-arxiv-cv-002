---
title: "VA-JUDGER-REWARD-MODELING-FROM-HUMAN-PREFERENCE-FEEDBACK-FOR"
source: https://arxiv.org/pdf/2608.18607v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 16:31:16"
field: "多模态生成与人类偏好对齐"
keywords: ["reward modeling", "joint video-audio generation", "human preference", "chain-of-thought", "GRPO", "post-training", "multimodal generation"]
innovations: ["提出VA-Judger链式思考全维奖励模型，首次将CoT Reward Modeling引入联合视频-音频生成", "构建VAPref-10K（9K提示/10.3K配对）大规模人类偏好数据集与VA-Judger-Bench基准", "设计维度级GRPO，以答案正确性+多维度排序一致性联合构建可验证奖励替代单一标量奖励"]
benchmarks: ["VA-Judger-Bench", "JavisBench (200-prompt subset)", "VAPref-10K"]
---

# 论文速读：VA-JUDGER-REWARD-MODELING-FROM-HUMAN-PREFERENCE-FEEDBACK-FOR

## 一句话总结
本文提出VA-Judger，一个面向联合视频-音频生成的链式思考全维奖励模型，通过构建9K提示词/10.3K成对比较的大规模人类偏好数据集VAPref-10K，采用"易对冷启动→难对拒斥采样→维度级GRPO强化学习"三阶段训练，实现了对人类偏好的高对齐评估；将其作为奖励信号后训练LTX-2生成模型，在11/13项JavisBench指标上取得最佳结果，人类偏好率达62.30%（远超基线的10.08%）。

## 研究问题与动机
- **现有奖励信号与人类判断严重脱节**：当前联合视频-音频生成的后训练奖励由独立评估指标（VideoAlign、AudioBox、CLAP、Synchformer等）拼接而成，但各项指标仅衡量单一维度（视频质量/音频质量/同步性），无法捕捉文本-视频-音频跨模态的整体语义与时序连贯性；论文实测表明这些指标的准确率仅50.43%~56.88%，远低于可接受水平。
- **独立指标优化导致reward hacking**：模型针对狭窄的专家指标进行优化，可刷高单项分数却产生与人类感知不一致的内容（如事件错误、声音情绪不匹配），削弱后训练的实际价值。
- **缺乏面向联合生成的优质人类偏好数据**：图像/视频领域已有Pick-a-Pic、VideoReward等偏好数据集，但联合视频-音频生成领域尚无基于整体人类判断的配对比较数据；单模态评估模型因缺失另一模态信息无法直接迁移。
- **结构化推理能力不足**：基础全模态模型（如Qwen3-Omni）未经多维权重评估训练，直接输出二元偏好标签无法引导细粒度 Reasoning，也难以提供密度足够的奖励信号以支持RL微调。

## 核心贡献（创新点）
- **构建VAPref-10K大规模人类偏好数据集**：从YouTube/Bilibili/影视等采集10,173段真实视频音频，经Qwen3.5-Omni Caption后改写为10,083条生成就绪Prompt，覆盖6大类场景，并由此构建难度感知的成对比较（4.4K易对+9.8K难对）；与既有数据集的本质区别在于聚焦联合视频-音频的全局人类偏好，而非单模态或合成提示。
- **提出VA-Judger链式思考全维奖励模型**：首次将CoT Reward Modeling引入联合视频-音频生成，要求模型在最终选择前逐维（Prompt匹配、音视频一致性、音频质量、视频质量、完整性）给出1-10分评分与解释；与既有Omni模型（如IXC-2.5-Reward、UnifiedReward）的区别在于专为跨模态一致性推理设计，而非单纯的多模态理解奖励。
- **设计易→难三阶段渐进训练范式**：Stage 1用Gemini 3.1 Pro在易对上生成结构化Rubric回答做冷启动；Stage 2对难对执行拒斥采样（仅保留Gemini偏好与人工标注一致的样本）对齐人类判断；Stage 3引入Dimension-wise GRPO，以答案正确性+维度排序一致性联合构建可验证奖励；与纯SFT或单一RL阶段的本质差异在于兼顾格式学习与人类对齐、稀疏二元奖励与密集维度信号的组合。
- **提出VA-Judger-Bench并验证后训练增益**：构建含Easy/In-domain/Out-of-domain三分割的1,150对评测基准（覆盖Kling 2.6、Wan 2.6、Veo 3.1、Sora 2等闭源模型输出）；并用其奖励信号替换OmniNFT的五项独立专家奖励后训练LTX-2，Human Preference Rate从10.08%提升至62.30%，证明人类对齐奖励可直接转化为生成质量提升。

## 方法详解
- **任务形式化**：输入为Prompt $p$ 与两个视频-音频对 $(v_1,a_1),(v_2,a_2)$，模型输出为结构化CoT比较，对每条维度给出1-10分评分与理由，最终聚合总分并以`<answer>Video k is better</answer>`输出偏好。
- **五维评估Rubric**：A) Prompt Match（文本-内容对齐）；B) Audio-Visual Consistency（口型/动作与声音协调）；C) Audio Quality（音质/失真）；D) Video Quality（清晰度/光照/细节）；E) Completeness（叙事完整与流畅度）。
- **Stage 1 — Easy Cold Start**：4.4K易对（如OVI vs LTX-2质量差距明显），用Gemini 3.1 Pro生成Rubric格式CoT回答，仅微调语言骨干、冻结视听编码器；目的是让模型学会输出格式与粗粒度偏好判别。
- **Stage 2 — Hard Preference Alignment（拒斥采样）**：难对（同模型同Prompt不同随机种子）Gemini与人类仅60%一致，因此由人工标注偏好与支撑维度；用Gemini生成Rubric回答后，仅保留最终偏好与人工一致的样本（共4.5K），使模型在保留格式的同时对齐人类细微判断。
- **Stage 3 — Dimension-wise GRPO**：对每个输入采样$G=8$个回答，定义复合奖励 $R_i = \frac{1}{2}(r_{\text{ans}}(o_i) + r_{\text{dim}}(o_i))$，其中 $r_{\text{ans}}=\mathbf{1}[\hat{y}=y^\star]$，$r_{\text{dim}}=\frac{1}{|D_h|}\sum_{d\in D_h}\mathbf{1}[s_{d,y^\star}>s_{d,3-y^\star}]$；即不仅要求最终选择正确，还要求每个被人工指定的维度上，胜出样本得分更高。优势归一化后采用Clipped Surrogate Objective + KL惩罚更新。
- **后训练配置**：以LTX-2为Policy，冻结19B骨干，仅训练LoRA（rank 32, scale 64）；每个Prompt生成8候选、28对比较，VA-Judger打分后按维度分配奖励（C→音频分支、D→视频分支、A+B+E→共享分支），Advantages裁剪至$[-5,5]$，AdamW学习率$3\times10^{-5}$，BF16精度。

## 实验与结果
- **数据集与基准**：VAPref-10K（9K Prompt、10.3K配对比较，源自LTX-2/OVI/DaVinci-MagiHuman）；VA-Judger-Bench含1,150对（400 Easy / 250 In-domain / 500 Out-of-domain，后者使用Kling 2.6、Wan 2.6、Veo 3.1、Sora 2等闭源模型输出，与训练集无重叠）。
- **Reward模型评测（表1）**：单维度指标准确率最高56.88%（Javis Score），Out-of-domain下VideoAlign仅48.35%；VA-Judger（GRPO）Overall Total Acc达68.43%，较Qwen3-Omni CoT基线提升10.60pp，Easy/In-domain/Out-of-domain分别为76.25%/66.00%/63.40%。
- **后训练LTX-2（表2）**：在JavisBench 200随机prompt子集上，VA-Judger后训练在13项指标中夺11冠、7项互补指标中夺6冠；相对基线LTX-2，VQ从2.248→3.942（+75.4%）、MQ 0.697→1.183（+69.7%）、AQ 4.767→5.610（+17.7%）、JavisScore 0.074→0.230（+210.8%）。
- **人类评估（图4）**：200 prompt三-way forced-choice，VA-Judger后训练模型获62.30%偏好率，OmniNFT为27.63%，基线LTX-2仅10.08%；VA-Judger偏好率约为OmniNFT的2.3倍、基线的6.2倍。
- **最强结果**：Out-of-domain VA-Judger-Bench Accuracy 63.40%（Total Acc），以及LTX-2 + VA-Judger后训练Human Preference Rate 62.30%。

## 相关工作脉络
- **联合视频-音频生成**：UniVerse-1、Ovi、MOVA、LTX-2、Baton、Javis系列；本文与OmniNFT定位差异在于，OmniNFT依赖五项独立专家指标作为RL奖励，本文则从人类偏好直接学习跨模态一致性奖励。
- **图像/视频Reward Model**：Pick-a-Pic、ImageReward、HPSv2、MPS、LiFT、VideoReward、VADER；本文与它们的关键区别是评测对象扩展为"文本+视频+音频"三元组，且必须同时输出多维权重评分，而不仅是二元偏好。
- **多模态理解Reward**：IXC-2.5-Reward、UnifiedReward系列；本文强调这些模型无法显式评估音视频协同事件一致性，也未作为生成后训练的omni-reward使用。
- **Diffusion RL**：FlowGRPO、DanceGRPO、DiffusionNFT；本文在联合生成领域的延伸在于将GRPO的奖励信号从单一标量升级为可验证的维度级复合信号。
- **视频评估指标**：VideoAlign、HPSv3、AudioBox、CLAP、Synchformer、Javis Score；本文实证显示这些指标与人类偏好的对齐度普遍≤57%，凸显构建人类对齐奖励的必要性。
- **LLM-as-Judge / 合成偏好数据**：早期工作多依赖LLM直接打分或生成偏好；本文采用Gemini辅助生成易对Rubric，但在难对上强制以人工标注过滤，降低LLM偏差影响。

## 局限性与未来方向
- **部分训练数据依赖LLM生成**：Stage 1全部使用Gemini 3.1 Pro生成的CoT，虽在200对小样中80%与人工一致，但仍可能引入LLM系统性偏见；论文未讨论如何进一步削弱LLM judge偏差。
- **难对拒斥采样仅保留二元一致性**：Stage 2仅检查最终偏好是否与人工一致，维度评分部分仍由Gemini生成，可能保留不准确的维度归因。
- **评估覆盖范围有限**：VAPref-10K的来源Prompt主要来自Web视频，可能缺少特定专业场景（如医学、工业检测）；Out-of-domain虽含闭源模型，但仅4个，且未见长视频/复杂对话场景。
- **推理与计算开销**：VA-Judger基于30B参数多模态模型，后训练时每步需对28对候选进行多轮推理，工程部署成本较高；论文未提供延迟/显存开销分析。
- **单模型后训练验证**：仅在LTX-2一个Policy上验证，未推广至其他架构（如Ovi、Javis），泛化性待进一步检验。

## 研究启发与可借鉴点
- **易→难课程学习+拒斥采样对齐人类**：先用LLM在高质量差距样本上完成格式学习，再在难样本上用人工标注过滤合成Reasoning，可有效兼顾规模与精度；该范式可迁移至其他需要结构化输出的多模态评估任务。
- **维度级复合奖励替代单一标量奖励**：GRPO扩展中同时验证"最终选择+各维度排序一致性"的设计，能显著缓解reward hacking；这一思路适用于任何需要细粒度优化的生成后训练场景（如3D、图像编辑）。
- **Rubric驱动的结构化CoT输出**：强制模型在决策前输出五维评分与解释，不仅提升可解释性，还使后续RL阶段能获得更丰富的可验证信号；可考虑扩展至更多维度（如情感一致性、时空因果）。
- **Out-of-domain Benchmark构建策略**：VA-Judger-Bench特意纳入闭源模型输出作为难样本，有效检验奖励模型的泛化能力；建议在同类工作中同样设计跨模型/跨域测试split。
- **开源代码与工程配置透明**：论文详细公开了LoRA配置、梯度裁剪、advantage clipping、warmup step等超参，便于复现与改进；可复用其多模态Reward建模的工程流水线。

## 关键术语表
**VA-Judger**：面向联合视频-音频生成的链式思考全维奖励模型，输出五维评分与结构化CoT以预测人类偏好。
**VAPref-10K**：本文构建的大规模人类偏好数据集，包含约9K提示词与10.3K细粒度成对比较，源自公开生成模型输出。
**VA-Judger-Bench**：用于评估视频-音频奖励模型与人类偏好对齐程度的基准，含Easy/In-domain/Out-of-domain三分割共1,150对。
**Dimension-wise GRPO**：将GRPO奖励函数扩展为"答案正确性+维度排序一致性"的复合信号，使RL阶段同时优化最终偏好与中间推理。
**OmniNFT**：现有的联合视频-音频生成RL框架，依赖VideoAlign、AudioBox、CLAP、Synchformer五项独立专家指标构造奖励；本文以其为对比基线。
**LTX-2**：Efficient joint audio-visual foundation model（HaCohen et al., 2026），本文以其为Policy进行VA-Judger后训练验证。
**JavisBench**：Joint Audio-Video generative benchmark（Liu et al., 2025c），本文随机抽取200 prompt子集用于生成质量定量评测。
**Rejection Sampling（拒斥采样）**：在Stage 2中，仅保留Gemini生成答案中最终偏好与人工标注一致样本的训练策略，用于对齐人类判断。

## 可复现要素
- **数据集**：VAPref-10K（9K prompts、10.3K paired comparisons）；论文未明确声明是否开源。
- **基准**：VA-Judger-Bench（1,150对）；论文未明确声明是否开源。
- **代码**：已开源，链接 https://github.com/ShareLab-SII/VA-Judger。
- **权重**：论文未声明开源基座以外权重（仅提及初始化自Qwen3-Omni-30B-A3B-Instruct与LTX-2 19B checkpoint）。
- **关键超参**：
  - SFT：lr $5\times10^{-6}$，warmup 0.05，max seq length 24,576，batch=1/GPU×16 accumulation=128，8×GPU（DeepSpeed ZeRO 3）。
  - GRPO：temperature 1.0，G=8，lr $1\times10^{-6}$（Adafactor），answer/dim reward等权。
  - LTX-2后训练：LoRA rank=32, scale=64，lr $3\times10^{-5}$，AdamW，20 denoising steps（随机选8个timestep），CFGS video=1.5/audio=3.0，advantage clip $[-5,5]$，KL weight $10^{-4}$，BF16，6×GPU（FSDP）。
