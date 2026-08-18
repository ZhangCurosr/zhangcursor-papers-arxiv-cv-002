---
title: "TRAPSBench-Vision-Language-Models-Encode-but-Fail-to-Express"
source: https://arxiv.org/pdf/2608.13167v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 06:14:14"
field: "多模态模型可靠性与可解释性"
keywords: ["视觉-语言模型", "认知不确定性", "选择性放弃", "激活引导", "物理推理基准", "校准评估", "表征-输出鸿沟"]
innovations: ["提出TRAPSBench程序化视频基准与PECS合取度量，首次系统评估VLM在物理不确定性下的认知克制能力", "通过线性探针(AUROC up to 0.91)和激活引导因果证实VLM内部编码可迁移的认知信号但输出抑制该信号", "揭示文本/视觉不对称性(约4×)与CoT推理对校准的双刃效应，区分不同失败模式的几何结构"]
benchmarks: ["TRAPSBench"]
---

# 论文速读：TRAPSBench-Vision-Language-Models-Encode-but-Fail-to-Express

## 一句话总结
论文提出 TRAPSBench，一个基于 MuJoCo 物理仿真生成的 1,404 对对照/无效视频基准，用于评估 VLM 在视觉证据不足时选择性放弃回答的能力；发现 VLM 内部编码了认知不确定性信号（线性探针 AUROC 高达 0.91），但自回归输出抑制了该信号，导致自发克制能力极差（最佳 PECS 仅 0.292）。

## 研究问题与动机
1. **核心问题**：当视觉证据被遮挡或混乱时，VLM 是否知道何时不应该回答？即"认知克制"（epistemic restraint）能力。
2. **现有基准不足**：既有物理推理基准（CLEVRER、IntPhys、PhysBench、Morpheus）均不评估"在证据不足时选择性放弃回答"的能力，VLM 很少被测试识别自身视觉知识的边界。
3. **度量困境**：传统 abstention recall 奖励无差别放弃，accuracy 忽略认知维度，PECS 作为合取指标要求模型在有据可依时正确回答、在不可答时选择性放弃。
4. **部署需求**：自主智能体经常面对无法支持确定性预测的输入，选择性放弃是可靠 AI 系统的关键能力。

## 核心贡献（创新点）
1. **TRAPSBench + PECS 指标**：程序化生成 1,404 对可控/无效物理视频对（三类不确定性分类学），搭配合取度量 PECS（Acc × Youden's J，两者缺一即为零），杜绝"永远放弃"和"永远回答"的退化策略。
2. **表征-输出鸿沟的系统性揭示**：通过三种独立证据（引导提示解锁潜在克制能力 1.9×、线性探针跨域迁移 AUROC 最高 0.91、单层激活引导因果操控放弃）证明瓶颈在表达能力而非感知能力，并在 Qwen、Gemma、LLaVA 三个开放权重家族中复现。
3. **失败模式不对称性**：模型检测文本不可能性的能力比检测视觉信息缺口高约 4 倍（混乱子场景中高达 197 倍）；链式思维推理可能恶化校准——Qwen3-VL Think 会覆盖自身的内在怀疑而加剧幻觉。
4. **因果机制分析**：在 Qwen3-VL-8B 中，遮挡族方向编码领域通用的"证据缺失"信号（可跨域迁移），而混沌方向是领域特定的且近似正交，揭示认知透明度决定信号可迁移性。

## 方法详解
**基准构建**：
- 使用 MuJoCo 物理引擎程序化生成最小视频对：control（确定性结果可见）与 void（单一修改使结果不可计算）。
- 三类不确定性分类学：① **Occlusion**（N=202）：不透明遮挡物遮挡关键视觉数据；② **Chaotic Sensitivity**（N=500）：对初始条件极端敏感的确定性系统，视频截断在结果解析之前；③ **Ill-Posed Questions**（文本侧基线）：在每类上方构造问题本身无答案的变体。
- 任务流程：输入（视频，问题）→ 模型返回自由文本 → 3 模型裁判小组（Gemini 3 Flash、Qwen3-VL Instruct、Claude 4.6 Opus）多数投票 → PECS 评分。

**PECS 公式**：
$$\text{PECS} = \text{Acc} \times \max(0, \text{AbsRec} - \text{FalseAbs})$$
即控制准确率 × Youden's J 统计量（抑制非选择性放弃），J 为 0 时 PECS 归零。

**探针分析**：在冻结的 Qwen3-VL-8B 全部 37 层提取隐状态，训练 l2 正则化 LR 探针预测 void vs. control，报告最佳层 AUROC（阈值无关）。Cf→Cf 限制：训练/测试集仅包含模型在标准提示下自信编造答案的 void 样本（即 0% 自发表达放弃的样本）。

**激活引导（Activation Steering）**：在 layer l=20 计算 void 方向 $\mathbf{v}_\ell = (\bar{\mathbf{h}}_\ell^{\text{void}} - \bar{\mathbf{h}}_\ell^{\text{control}}) / \|\cdot\|$，在自回归每一步修改隐状态：$\mathbf{h}_\ell^{(t,i)} \leftarrow \mathbf{h}_\li

ll + \alpha_{\text{eff}} \cdot \mathbf{v}_\ell$，其中控制样本 +α 诱导放弃，void 样本 −α 强制编造；扫描 α ∈ {0,2,5,10}。

## 实验与结果
**模型与设置**：16 个 VLM，跨越 5 个家族（Gemini 6 个、Qwen3-VL 3 个、GPT-5 5 个、Gemma 4 E4B、LLaVA-Video-7B）；三种提示制度（Standard、Guided、JSON）；3 次独立运行取均值，约 194,000 评测对。

**主要结果（Table 1，等于权重均值过四个数据集）**：
- **标准制度最佳**：Gemini 2.5 Flash，PECS=0.292（Acc 63%，AbsRec 49%，FalseAbs 1.5%）。
- **Guided 制度最佳**：Gemini 3.1 Pro RL，PECS=0.568（Acc 71%，AbsRec 86%，FalseAbs 4.9%）。
- **Guided 提示整体效果**：视频原生模型 AbsRec 提升 1.4–2.8×（中位数 1.9×），说明存在被锁定的潜在克制能力。
- **Gemma 4 E4B（guided）PECS=0.215，LLaVA-Video-7B（guided）PECS=0.101**，远低于饱和。

**视觉 vs. 文本不对称（Table 2）**：
- 混乱子场景中，模型检测文本不可能性比视觉缺口高 3–197×，中位数约 4×。
- 唯一反转：Gemini 3.1 Pro R-Low 在遮挡子场景（0.9×）。

**推理（CoT）效果分化**：
- Gemini Flash Thinking 将内在怀疑转化为放弃（AbsRec +4–13pp）。
- Qwen3-VL Think 反而覆盖自身怀疑，编造率高于非推理版本。

**探针跨域迁移（Table 3）**：
- 跨模态（视觉↔文本）：AUROC 最高 1.0（LLaVA, ch→ch_ip）。
- 跨域（遮挡↔混沌）：AUROC 最高 0.882（Qwen, ch→oc）。
- 跨双（不同域+不同类型）：AUROC 最高 0.913（Qwen, oc→ch_ip）。
- Cf→Cf 限制下 AUROC 下降 ≤0.03，证实内部信号即使在被编造的样本上依然存在。

**激活引导因果确认（Table 4, Figure 5）**：
- Qwen3-VL-8B 中，α=10 时 occlusion 方向将控制样本放弃率从 11% 提升至 75%（guided），chaos 方向仅从 11% 升至 15%。
- 方向余弦：oc、oc_ip、ch_ip 形成相关簇（cos=0.19–0.25），ch 近似正交（cos≤0.08）。
- 跨模态迁移：oc_ip→ch 达到 100% 控制放弃，强于同模态 oc→ch（90%）。

## 相关工作脉络
1. **物理推理基准**（CLEVRER, IntPhys, PhysBench, Morpheus）：评估物理推理但不评估"在证据不足时选择性放弃"；TRAPSBench 的对比对设计消除了捷径推理，测量的是模型是否认识到无答案被许可，而非预测什么。
2. **不确定性与选择性预测**（Kendall & Gal, Geifman & El-Yaniv, AbstentionBench）：前者需要访问模型内部，后者在 NLP 中形式化无答案问题；TRAPSBench 将探测和引导扩展到视觉认知不确定性，提供首个因果证据表明 VLM 编码可迁移的认知信号但其输出抑制了该信号。
3. **不可答视觉问题**（UNK-VQA, VisionTrap, TUBench, MM-UPD, CertainlyCertain）：均在单图像上操作，通过语义扰动派生不可答性；TRAPSBench 扩展到视频，不可答性源于物理动力学，对比对无编辑伪影，使用纯文本度量。
4. **激活引导与真相表示**（Turner et al., Rimsky et al., Burns et al., Marks & Tegmark）：此前应用于真相、情感和拒绝，未应用于认知不确定性或视觉域；本文扩展了该范式。
5. **Shortcut-aware 视频基准**（Krojer et al.）：引入最小视频对缓解捷径推理，但不测试放弃；TRAPSBench 在此基础上增加了无答案识别维度。

## 局限性与未来方向
1. **机制分析范围受限**：探针和引导仅覆盖三个无共享训练流水线的开放权重家族（Qwen3-VL-8B, Gemma 4 E4B, LLaVA-Video-7B）；闭源模型无法探针，行为签名跨 16 个模型成立但机制解释受限。
2. **程序化场景偏简单**：MuJoCo 刚体物理是更容易的测试用例，需与真实世界基准互补。
3. **提示门控与方向几何时家族特定**：单向门控在 Qwen 中完全存在，Gemma 部分存在，LLaVA 不存在；方向几何在三个家族中不一致。
4. **未来方向**：软体/流体物理、真实世界视频、void-video 微调、生成式 rollout。

## 研究启发与可借鉴点
1. **最小视频对范式可迁移**：程序化生成对照/无效对的方法论（单一变量修改使结果不可计算）可应用于其他视觉推理领域（如医学影像、驾驶场景）的认知可靠性评估。
2. **PECS 合取度量设计值得复用**：同时惩罚"永远回答"和"永远放弃"的 Youden's J 乘积结构，可作为多模态模型校准评估的通用模板。
3. **表征-输出鸿沟的诊断流程可复现**：线性探针 + 激活引导的三阶段验证（探测→因果确认→跨架构复现）为后续"模型已知但说不出"现象提供了完整分析框架。
4. **CoT 对校准的双刃效应**：推理不一定改善可靠性，训练奖励设计需权衡；这提示团队在设计 thinking 模型时应考虑对不确定的显式建模，而非仅奖励流畅推理。
5. **文本/视觉不对称的发现具有普遍意义**：模型对语言级不可能性的敏感度远高于视觉级不确定性，提示未来工作应加强视觉元认知训练。

## 关键术语表
**TRAPSBench**：Testing Restraint in Ambiguous Physical Scenarios 的缩写，一个基于 MuJoCo 程序化生成的 1,404 对对照/无效物理视频基准。
**PECS（Penalized Epistemic Calibration Score）**：合取校准度量，公式为 Acc × max(0, AbsRec − FalseAbs)，同时要求正确回答和选择性放弃，两者缺一不可。
**Epistemic Restraint（认知克制）**：模型在视觉证据不足以支持确定性预测时，主动选择放弃回答的能力。
**Void Direction（无效方向）**：在隐状态空间中，通过比较 void 与控制样本的平均激活差异得到的单位向量，用于因果引导模型的放弃行为。
**Cf→Cf（Confabulation-to-Confabulation）**：探针实验中训练/测试集均限制于模型在标准提示下自信编造答案的样本，用于验证内部信号独立于行为表达。
**Youden's J Statistic**：灵敏度（AbsRec）− 假阳性率（FalseAbs），衡量模型区分可答与不可答场景的判别力，取值范围 −1 至 1。
**Activation Steering（激活引导）**：在自回归生成的每一步向隐状态注入方向向量（+α 或 −α），以因果方式操控模型行为的技术。
**Confabulation（编造）**：模型对不可答问题给出自信答案的行为，87–99% 的编造包含视觉上的幻觉前提（hallucinated premise）。

## 可复现要素
- **数据集**：TRAPSBench 已在 GitHub 公开（https://github.com/facebookresearch/TRAPS-Benchmark），CC BY-NC 4.0 许可，包含 1,404 对视频、问题和评估提示。
- **代码/权重**：探针和引导代码见附录；评测使用 API 模型（Gemini、GPT-5）及开源本地模型（Qwen3-VL-8B bf16、Gemma 4 E4B bf16、LLaVA-Video-7B bf16）。
- **关键超参**：API 模型 temperature=0.8、top-p=0.95、max_completion_tokens=8000；本地模型最多生成 512–1024 tokens；探针 C=1、StandardScaler 标准化；引导 α∈{0,2,5,10}。
- **裁判设置**：3 模型裁判小组（Gemini 3 Flash、Qwen3-VL Instruct、Claude 4.6 Opus），多数投票，仅接收文本输入。
