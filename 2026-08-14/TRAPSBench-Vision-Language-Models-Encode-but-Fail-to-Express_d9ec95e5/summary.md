---
title: "TRAPSBench-Vision-Language-Models-Encode-but-Fail-to-Express"
source: https://arxiv.org/pdf/2608.13167v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 04:42:29"
field: "多模态认知校准与可解释性"
keywords: ["vision-language models", "epistemic uncertainty", "abstention", "benchmark", "activation steering", "probing", "physical reasoning"]
innovations: ["提出TRAPSBench与PECS联合度量首个系统评估VLM在物理视频上的选择性放弃能力", "通过线性探针(Cf→Cf限制)与单层激活导向因果证明VLM内部编码可迁移的'可答/不可答'信号但输出层压制", "发现文本不可答性检测比视觉信息缺口快约4×且CoT推理对校准呈非单调影响"]
benchmarks: ["TRAPSBench"]
---

# 论文速读：TRAPSBench-Vision-Language-Models-Encode-but-Fail-to-Express

## 一句话总结
本文提出 TRAPSBench（1,404 对程序化生成的 MuJoCo 物理视频对）与 PECS 指标，系统性证明主流 VLM 内部能编码"何时该放弃回答"的认知不确定性信号（线性探针 AUROC 最高达 0.91，单层激活导向可因果控制放弃行为），但其自回归输出却压制该信号，导致自发克制能力极差（最佳 PECS 仅 0.292）。

## 研究问题与动机
1. **现实部署需求**：自主智能体常面对感官信息不足以支撑确定性预测的输入（遮挡、混沌轨迹、 ill-posed 问题），正确行为应是选择性放弃回答；然而现有物理推理基准测试（如 CLEVRER、IntPhys、PhysBench）均未评估"在证据不足时识别放弃"这一认知克制能力。
2. **度量盲区**：传统放弃召回率（abstention recall）会奖励无差别放弃策略，准确率又忽略认知维度；需要一个同时要求"证据充分时答对 + 证据不足时放弃 + 无假性放弃"的联合度量。
3. **表征-输出鸿沟假设**：既往 truth-representation 研究表明 LLM 内部能编码真值信号但输出层压制；作者将其推广至视觉-语言模型的认知不确定性场景，试图验证 VLM 是否同样存在"编码但无法表达"的瓶颈。
4. **模态不对称性**：文本不确定性（ill-posed question）与视觉不确定性（occlusion/chaos）在检测难度上可能存在本质差异，值得系统对比。

## 核心贡献（创新点）
1. **TRAPSBench + PECS**：首个基于最小视频对范式的物理不可答性基准（1,404 对，三taxonomy：遮挡、混沌敏感性、ill-posed 问题），配合 Youden's J 统计量惩罚无差别放弃/从不放弃策略的 PECS 指标。
2. **表征-输出鸿沟的系统刻画**：通过三种互补证据证实 VLM 内部编码可迁移的"可答/不可答"区分信号（线性探针跨数据集 AUROC 最高 0.91；单层导向因果控制放弃；三种开放权重家族 Qwen3-VL/Gemma/LLaVA 均可复现），瓶颈在表达而非感知。
3. **失败模式不对称性**：模型对文本不可答性的检测比视觉信息缺口快约 4×（混沌 split 上 3–25×，部分图像模型高达 197×）；CoT 推理可能削弱校准——Qwen3-VL Think 以最高怀疑率（24%）却最低转化率输出。
4. **因果机制几何分析**：在 Qwen3-VL-8B 中发现，遮挡族 void 方向编码域通用的"证据缺失"信号（可跨 domain/modal 迁移），而混沌方向域特异且近正交（cos ≤ 0.08），揭示 epistemic transparency 决定可迁移性。

## 方法详解
### 3.1 最小视频对范式
对每个 MuJoCo 刚性动力学场景生成 matched control/void 对：control 提供完整信息，void 引入单一修改（遮挡碰撞区 / 截断混沌轨迹 / ill-posed 问题）使结果不可计算；比较模型在 matched pair 上的表现以隔离认知不确定性识别能力。

### 3.2 物理不可答性分类学
- **Occlusion (N=202)**：不透明遮挡物阻断关键视觉数据（rigid-body 场景记录）。
- **Chaotic Sensitivity (N=500)**：对初始条件极端敏感的系统（pachinko-waterfall、plinko、tumbling-dice、seesaw），有限视觉精度下点预测不可能；截断发生在结果解析之前。
- **Ill-Posed Questions (N=702)**：复用 control 视频但构造文本侧不可答问题（混沌类含 false premise 如不存在颜色的球；遮挡类询问 fully visible 视频中不可观测细节）。

### 3.3 PECS 度量
$$\text{PECS} = \text{Acc} \times \max(0, \text{AbsRec} - \text{FalseAbs})$$
- Acc：control 视频答对率（数值 ±0.5 exact match，分类 exact string match）。
- AbsRec：void 视频正确放弃率。
- FalseAbs：control 视频错误放弃率。
- Youden's J 项（AbsRec − FalseAbs）确保"总是放弃"与"从不放弃"均得 J=0，配合 Acc 项形成联合惩罚。

### 3.4 评估协议
- 16 个 VLM，三 prompt regime（Standard / Guided "I don't know" / JSON structured output）。
- 3-model judge panel（Gemini 3 Flash、Qwen3-VL Instruct、Claude 4.6 Opus），仅接收文本输入，多数投票；inter-judge 一致率 88.3–99.5%，Fleiss κ ≥ 0.84。
- 16 模型 × 4 数据集 × 3 regime × 3 独立运行 ≈ 194,000 评测对；> 387,000 VLM 调用。

### 3.5 线性探针
冻结 Qwen3-VL-8B 全部 37 层，提取 hidden states，训练 l2-regularized LR probe（C=1, StandardScaler），以单数据集 hidden state 训练预测另一数据集 void/control 标签，报告 best-layer AUROC（阈值无关）；probe 训练不接触目标数据集标签，仅 best-layer 选择使用。

### 3.6 激活导向（Activation Steering）
在层 ℓ=20 计算 void 方向：$\mathbf{v}_\ell = (\bar{\mathbf{h}}_\ell^{\text{void}} - \bar{\mathbf{h}}_\ell^{\text{control}}) / \|\bar{\mathbf{h}}_\ell^{\text{void}} - \bar{\mathbf{h}}_\ell^{\text{control}}\|$；自回归每步施加：$\mathbf{h}_\ell^{(t,i)} \leftarrow \mathbf{h}_\ell^{(t,i)} + \alpha_{\text{eff}} \cdot \mathbf{v}_\ell$（control 接受 +α 诱导放弃，void 接受 −α 强制 confabulate）。 sweeps α ∈ {0, 2, 5, 10}，8 个 source-target 跨数据集路径。

## 实验与结果
### 数据集与基线
- 16 个 VLM 覆盖五个家族：Gemini（2.5/3.1 共 6 个）、Qwen3-VL（3 个）、GPT-5（5 个）、Gemma 4 E4B、LLaVA-NeXT-Video-7B。
- 三 prompt regime：Standard / Guided / JSON。
- 基线对比任务：现有基准（CLEVRER、IntPhys、PhysBench、Morpheus）均不评估 selective abstention；同领域工作（UNK-VQA、VisionTrap、TUBench、MM-UPD、CertainlyUncertain）仅处理单图语义扰动。

### 主要结果（Table 1 摘要）
| 模型 | Standard PECS | Guided PECS | 最佳 Guided AbsRec |
|------|--------------|------------|-------------------|
| Gemini 2.5 Flash | 0.292 | 0.467 | 78% |
| Gemini 3.1 Pro R-Low | 0.197 | 0.568 | 86% |
| Qwen3-VL Instruct | 0.246 | 0.441 | 66% |
| GPT-5 Pro | 0.126 | 0.447 | 64% |
| Gemma 4 E4B | 0.160 | 0.215 | 76% |
| LLaVA-Video-7B | 0.079 | 0.101 | 46% |

- 最佳标准 PECS：Gemini 2.5 Flash（0.292）；最佳 Guided PECS：Gemini 3.1 Pro R-Low（0.568）。
- Guided prompting 使视频原生模型 AbsRec 提升中位数 1.9×（1.4–2.8×），Accuracy 基本不变（±2pp）。
- 所有模型 Guided 后 PECS 仍远低于饱和（无人触及 0.6 iso-PECS 等高线）。

### 视觉 vs 文本不对称性（Table 2）
- 混沌 split：文本 AbsRec 比视觉高 3–25×（中位 gap ≈ 4×）；图像模型（GPT-5）部分高达 197×。
- 遮挡 split 接近 parity（gap 1.1–2×）；Gemma 混沌 split 已在标准 regime 达 60% AbsRec。

### 线性探针跨数据集迁移（Table 3）
- Cross-modality（视觉↔文本）：AUROC 最高 1.0（LLaVA ch→ch.ip），Cf→Cf 限制下仍 ≥ 0.8。
- Cross-domain（occl↔chaotic）：AUROC 0.64–0.88。
- Cross-both（不同 domain + 不同 void type）：AUROC 最高 0.962（LLaVA oc→ch.ip）。
- **关键发现**：即使将训练/测试集限制为模型在标准 regime 下**confabulated** 的 void 样本（即 0% 自发放弃），探针仍能高 AUROC 迁移——证明内部信号独立于输出行为。

### 激活导向因果证据（Table 4）
- Qwen3-VL-8B：Guided inference 下 oc-family 方向实现 control 放弃 11% → 75%（α=10），ch-family 仅 11% → 15%。
- 同向 steering 下 oc→oc 达 46%（home-domain），ch→ch 达 70%——表明 ch 方向并非"弱"而是"域特异"。
- 跨模态：oc_ip（文本）→ ch（视觉）实现 100% control 放弃，超越同模态 oc→ch（90%）。
- Gemma 4 E4B / LLaVA 均可复现单层导向因果控制放弃，occlusion 方向优势在 Gemma 更强，LLaVA 较弱。

### CoT 推理的双刃效应
- Gemini Flash vs NT：Thinking 提升 AbsRec +4–13pp，将内部怀疑转化为放弃。
- Qwen3-VL Think vs Instruct：Thinking 反而降低 AbsRec（Chaotic Ill-Posed：Instruct 82.3% vs Think 71.7%），且 Think 有最高怀疑率（24%）却最低转化率——CoT 覆盖自身不确定并 confabulate。

## 相关工作脉络
1. **物理推理基准**：CLEVRER、IntPhys、PhysBench、Morpheus 均测"预测什么"而非"是否识别无法预测"；IntPhys 2 扩展至复杂合成环境但仍 presuppose determinate answer。TRAPSBench 首次系统评测选择性放弃。
2. **认知不确定性与选择性预测**：Kendall & Gal (2017)、Gal & Ghahramani (2016) 的 epistemic uncertainty 与 Geifman & El-Yaniv (2017, 2019) 的 rejection head 需访问模型内部；SQuAD 2.0、AbstentionBench (Kirichenko et al., 2025) 形式化 NLP 不可答问题。本文将其扩展至视觉域并与 mechanistic interpretability 结合。
3. **LLM 元认知与 probing**：Kadavath et al. (2022)、Xiong et al. (2024) 显示 LLM 具一定 metacognitive self-knowledge；Burns et al. (2023)、Marks & Tegmark (2024) 发现真值线性结构；Arditi et al. (2024) 证明拒绝由单方向介导。本文首次将 probing + steering 应用于**视觉认知不确定性**。
4. **Activation steering**：Turner et al. (2023)、Rimsky et al. (2024)、Li et al. (2023) 已用于 truthfulness/sentiment/refusal，但未应用于 epistemic uncertainty 或视觉域。本文展示 void 方向在视觉-语言模型中的因果效力。
5. **不可答视觉问题**：UNK-VQA、VisionTrap、TUBench、MM-UPD、CertainlyUncertain（CWA 需 logit access）均基于单图语义扰动，可能引入 artifact。TRAPSBench 以视频物理动力学构建对比对（无编辑 artifact），且 PECS 为纯文本度量，兼容 black-box API。

## 局限性与未来方向
1. **机制分析范围受限**：probing/steering 仅覆盖三个开放权重家族（Qwen3-VL-8B、Gemma 4 E4B、LLaVA-Video-7B），prompt gating 与方向几何为家族特异结论；闭源模型无法 probing，但行为签名跨 16 模型一致。
2. **Sim-to-real gap**：程序化 MuJoCo 刚性动力学场景是较易测试用例（self-labeled、无 pre-training contamination），与真实世界视频（PHYRE、EPIC-KITCHENS 等）存在分布偏移。
3. **方向几何非通用**：occlusion 方向主导混沌方向的发现仅在 Qwen/Gemma 显著，LLaVA 中弱化；prompt gate 程度（Qwen 完全 / Gemma 部分 / LLaVA 无）为架构特异。
4. **CoT 训练依赖**：推理是否改善校准取决于训练过程（reward design 是否惩罚 speculation），非单调关系。
5. **未来方向**：软体/流体动力学、真实世界视频、void-video fine-tuning、生成式 rollout 下的 epistemic calibration。

## 研究启发与可借鉴点
1. **最小视频对范式的可迁移性**：用 MuJoCo/物理引擎生成 matched control-void pair 以隔离特定不确定性源，可复用于其他 benchmark（如机器人操作、因果推断、多智能体协作），避免 real-world 数据的 contamination 与标注噪声。
2. **Cf→Cf probing 协议的严谨设计**：将 train/test 限制为模型"confabulated"样本（即输出与内部信号完全背离的 case），排除行为 confound，证明表征信号独立于输出——该方法论可直接迁移至 truthfulness/refusal 等任何"编码-表达鸿沟"研究。
3. **PECS 指标的结构化设计**：Youden's J 惩罚 + Acc 乘积，同时 zero out always-abstain、never-abstain、random、perfect-answerer 等退化策略；可作为多模态校准度量的通用模板。
4. **单向 prompt gating 的发现**：Guided prompt 在 Qwen 中完全打开 gate、Gemma 部分、LLaVA 无 gate——说明"提示词解锁表达"并非普适机制，未来 work 需区分 representation 存在性 vs. output pathway accessibility。
5. **CoT 对校准的非单调影响**：同一 reasoning 增强在不同架构上产生相反效果（Gemini 改善 vs. Qwen 恶化），提示团队在引入 CoT 时应同步评估 epistemic calibration，而非仅看 accuracy。

## 关键术语表
- **TRAPSBench**：Testing Restraint in Ambiguous Physical Scenarios 的缩写，程序化生成的 MuJoCo 视频基准，包含 1,404 对 matched control/void 物理场景。
- **PECS**：Penalized Epistemic Calibration Score，公式为 Acc × max(0, AbsRec − FalseAbs)，通过 Youden's J 统计量联合惩罚无差别放弃与从不放弃策略。
- **Epistemic restraint**：认知克制，指模型在证据不足时主动选择放弃回答而非猜测的能力。
- **Confabulation**：虚构，模型在不可答问题上自信输出无视觉证据支持的答案；本文分类为 hallucinated premise (HP)、invalid inference (II)、epistemic surrender (ES)。
- **Linear probe transfer AUROC**：用 logistic regression probe 从 frozen 模型 hidden states 预测 void/control 标签的 AUC，跨数据集迁移时反映表征信号的泛化性。
- **Activation steering**：在自回归生成每步向 hidden state 添加 direction vector（+α 诱导某行为，−α 抑制），用于因果验证表征-行为关联。
- **Void direction**：某层 hidden state 在 void vs. control 样本均值差归一化后的方向向量，编码该层"证据缺失"信号。
- **Prompt gating**：不同 prompt regime 对同一内部表征信号的输出通路约束差异；本文发现 Qwen 为完全单向 gate（standard 下无法诱导放弃）。

## 可复现要素
- **数据集**：TRAPSBench 公开于 https://github.com/facebookresearch/TRAPS-Benchmark（CC BY-NC 4.0），含视频、问题、evaluation prompts。
- **代码/权重**：评测 prompt 与数据公开；16 个模型中 API-served 模型（Gemini、GPT-5、Qwen3-VL 大模型）通过公开 API 访问；开源模型 Qwen3-VL-8B、Gemma 4 E4B、LLaVA-NeXT-Video-7B 均从 Hugging Face checkpoint 本地运行。
- **关键超参**：API 模型 temperature=0.8、top-p=0.95、max_completion_tokens=8,000；Qwen3-VL-8B 本地 bf16 greedy decoding（16 uniform frames）；Gemma/LLaVA 本地 bf16（32 frames @ 2 FPS，max_new_tokens=512）；judge panel 温度 0.8、top-p 0.95、max output 512 tokens；所有评测 3 次独立运行取 mean±std。
