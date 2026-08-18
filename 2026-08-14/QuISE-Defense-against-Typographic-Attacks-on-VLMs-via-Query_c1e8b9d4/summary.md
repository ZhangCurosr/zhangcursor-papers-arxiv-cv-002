---
title: "QuISE-Defense-against-Typographic-Attacks-on-VLMs-via-Query"
source: https://arxiv.org/pdf/2608.13119v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 03:56:37"
field: "多模态安全与鲁棒性"
keywords: ["typographic attack", "VLM defense", "semantic editing", "black-box defense", "multimodal security"]
innovations: ["首次提出基于查询无关语义编辑的黑盒防御框架", "影响力感知文本定位结合Q0-I0语义替换", "双重编辑+一致性验证的防御选择策略"]
benchmarks: ["SCAM", "SceneTAP", "SELF", "TextVQA", "ST-VQA"]
---

# 论文速读：QuISE-Defense-against-Typographic-Attacks-on-VLMs-via-Query

## 一句话总结
本文提出 QuISE，一种模型无关、无训练的黑色框防御方法，通过查询无关语义编辑（query-irrelevant semantic editing）来防御针对视觉语言模型（VLMs）的排版攻击（typographic attacks），实现对误导性文本的语义替换与答案一致性验证，显著提升防御准确率并保持干净图像上的性能。

## 研究问题与动机
- **排版攻击威胁**：攻击者在图像中注入误导性文本，引导 VLM 依赖对抗性文本线索而非视觉证据作答，且该类攻击在合成、真实场景和场景连贯设置下均有效。
- **现有防御局限**：Defense-Prefix、Dyslexify 等方法依赖 CLIP 架构或内部注意力头访问，难以应用于黑盒闭源 VLM；AAP 基于提示词缓解，效果有限且在强攻击下甚至降低准确率。
- **语义 vs 外观修改**：动机研究发现，对攻击文本的外观修改（模糊、对比度、旋转等）防御提升有限，而语义修改可将准确率提升至接近干净图像水平，说明攻击有效性主要来自语义内容。
- **查询相关性的核心作用**：通过 2×2 语义条件实验发现，查询相关性（query relevance）决定文本影响力，而图像相关性（image relevance）对查询无关文本的影响很小，因此 Q0-I0（同时与查询和图像无关）是最适合的替换语义。

## 核心贡献（创新点）
- **首次提出基于语义编辑的黑盒防御框架**：与现有防御依赖模型特定前缀或内部组件修改不同，QuISE 仅在输入层面通过文本定位与语义替换实现防御，无需访问目标 VLM 的内部表示。
- **影响力感知文本定位（Influence-Aware Text Localization）**：结合目标 VLM 的语义判断与辅助空间定位（PP-OCRv6），精准识别影响当前查询的文本区域，避免对全图可见文本的无差别编辑。
- **查询无关语义编辑 + 一致选择策略**：对每个定位区域生成两个语义不同的 Q0-I0 替换，仅在两个编辑图像的回答一致且合法时才采纳，否则回退到原始答案，有效兼顾防御恢复与干净图像性能保持。
- **系统级实验验证**：在 3 个攻击基准、4 种攻击设置、4 个 VLM（含闭源 GPT-4.1-mini）上验证，相比最强基线 AAP 提升 13.9–27.1 个百分点准确率，恢复率 RR 达 67.9–75.0%，伤害率 HR 仅 0.5–1.1%。

## 方法详解
QuISE 包含两个核心阶段：

### 1. 影响力感知文本定位（Influence-Aware Text Localization）
- 使用目标 VLM $M$ 识别图像中的可见文本，同时用固定辅助组件 $\mathcal{A}$（PP-OCRv6）提供空间位置，得到候选集合 $C = \{(l_j, t_j)\}_{j=1}^{N}$。
- 由目标 VLM 根据当前查询 $q$ 筛选出可能影响答案的区域：$R_q = \text{Select}_M(I, q, C) = \{(l_i, t_i)\}_{i=1}^{K} \subseteq C$。

### 2. 查询无关语义编辑防御（Query-Irrelevant Semantic Editing-based Defense）
- **文本移除**：$I^- = \text{Remove}(I, R_q)$，得到去除文本的图像用于评估图像相关性。
- **Q0-I0 替换生成**：对每个区域 $i$，由目标 VLM 基于 $(I^-, q)$ 生成候选替换，仅保留同时与查询和图像无关（Q0-I0）的候选，选择两个语义不同的替换 $w_i^{(1)}, w_i^{(2)}$，形成配对 $P_i = (w_i^{(1)}, w_i^{(2)})$。
- **配对编辑与答案选择**：
  $$
  (I^{(1)}, I^{(2)}) = \text{Edit}\left(I, R_q, \{P_i\}_{i=1}^{K}\right)
  $$
  得到两个编辑图像，分别输入 VLM 得到 $y_1 = M(I^{(1)}, q), y_2 = M(I^{(2)}, q)$。
- **一致选择规则**：
  $$
  \hat{y} = \begin{cases} y_1, & y_1, y_2 \text{ 均合法且 } y_1 = y_2 \\ y_0, & \text{否则} \end{cases}
  $$
  其中 $y_0 = M(I, q)$ 为原始答案，"合法"指可解析为任务答案空间且非空、非拒绝。

**设计要点**：
- 若未定位到任何区域或定位失败，直接保留原始答案 $y_0$，避免不必要的修改。
- Q0-I0 替换通过确定性抽象池选择，避免引入新的答案相关线索。
- 两次编辑+一致性验证在恢复率（RR）与干净图像保持率（Ret）之间取得平衡。

## 实验与结果
### 数据集与基线
- **攻击基准**：SCAM（合成+真实）、SceneTAP、SELF，共 8,558 个攻击样本。
- **干净图像测试**：TextVQA、ST-VQA（各 500 对）、FGVC-Aircraft、Food-101、ImageNet-100。
- **目标模型**：Qwen2.5-VL-7B-Instruct、LLaVA-OneVision-8B、InternVL3.5-8B、GPT-4.1-mini。
- **基线方法**：AAP（提示词缓解）、Defense-Prefix、Dyslexify、Localized Deletion。

### 主要结果（Table 1）
| 目标模型 | 攻击准确率 | QuISE 防御准确率 | Δ | RR↑ | HR↓ |
|---|---|---|---|---|---|
| Qwen2.5-VL-7B | 46.8% | 78.0% | **+31.2** | 73.9% | 0.8% |
| LLaVA-OneVision-8B | 43.8% | 73.3% | **+29.5** | 75.0% | 0.5% |
| InternVL3.5-8B | 33.2% | 63.4% | **+30.2** | 70.3% | 1.1% |
| GPT-4.1-mini | 72.7% | 85.8% | **+13.1** | 67.9% | 0.8% |

- **最强提升**：Qwen2.5-VL-7B 在 SELF 数据集上提升 46.5 个百分点（17.1% → 63.7%）。
- **对比 AAP**：QuISE 整体准确率提升 13.9–27.1 个百分点，RR 是 AAP 的 3–4 倍，HR 显著更低。
- **对比 CLIP 防御**：Defense-Prefix RR=39.3%，Dyslexify RR=57.9%，均低于 QuISE。

### 干净图像性能保持（Figure 7, Table 3）
- 在 TextVQA/ST-VQA 上，QuISE 性能保持率 Ret ≈ 80.49%，显著高于 Localized Deletion。
- 在不需图像文本的识别任务（Aircraft/Food-101/ImageNet-100）上，准确率无下降，HR=0。

### 编辑次数与一致性分析（Table 2）
- Single Edit：RR=76.48%，但 Ret 仅 40.46%。
- Three-Edit Majority：准确性略优但成本高。
- Three-Edit Unanimous：Ret=83.24%，HR=0.72%，但 RR 降至 72.43%。
- **QuISE（Two-Edit Consensus）**：RR=73.85%，HR=0.80%，Ret=80.49%，在恢复率、保持率与推理成本间取得最佳平衡。

## 相关工作脉络
- **Defense-Prefix（Azuma & Matsui, 2023）**：为 CLIP 学习防御前缀，依赖特定架构接口，无法迁移至通用 VLM；QuISE 通过输入级语义编辑实现架构无关防御。
- **Dyslexify（Hufe et al., 2026）**：定位并消融 CLIP 视觉编码器中传播文本信息的注意力头，需访问模型内部；QuISE 完全黑盒，无需内部组件访问。
- **AAP（Qraitem et al., 2025）**：提示词级缓解，要求 VLM 先检查可疑文本；QuISE 通过语义编辑直接消除误导信息，防御强度显著更高。
- **PAINT（Ilharco et al., 2022）**：通过权重插值微调开放词汇模型；QuISE 无需训练或权重修改，适用于闭源模型。
- **Typographic Attack 研究（SCAM, SceneTAP, SELF, FigStep）**：本文在多个攻击基准上统一评估防御效果，填补了系统性防御评估的空白。
- **OCR/VLM 安全研究**：现有工作主要关注攻击构造，本文首次系统探索基于语义编辑的防御路径。

## 局限性与未来方向
- **依赖 VLM 自身文本识别能力**：若目标 VLM 无法准确识别图像中的文本，定位阶段可能失败，影响防御效果。
- **Q0-I0 替换的语义多样性限制**：当前使用冻结抽象池，可能无法覆盖所有领域特定的图像内容。
- **仅评估问答任务**：未涉及指令遵循、多模态安全等其他 VLM 应用场景下的防御效果。
- **计算开销**：需两次额外推理调用，对实时性要求高的场景可能构成挑战。
- **未来方向**：可扩展至多模态安全防御、动态 Q0-I0 池构建、与其他防御策略（如输入预处理、后处理过滤）的结合。

## 研究启发与可借鉴点
- **语义编辑优于外观修改**：对对抗性文本的语义替换比视觉扰动更能恢复模型正确性，这一洞察可迁移至其他文本嵌入型攻击的防御。
- **查询相关性作为影响力度量**：利用目标模型自身判断文本与查询的相关性，是一种无需额外训练的轻量级影响力评估策略。
- **双重验证+一致性选择**：通过两次独立编辑+答案一致性验证，在保证防御效果的同时控制误杀率，可推广至其他需要鲁棒性保证的 VLM 应用。
- **两阶段框架的通用性**：先定位（localization）后编辑（editing）的设计模式可适配不同模态的安全防御需求。
- **干净图像性能保持机制**：fallback 到原始答案的策略在安全-可用性权衡中具有重要参考价值。

## 关键术语表
- **Typographic Attack（排版攻击）**：通过在图像中注入误导性文本，使 VLM 依赖文本线索而非视觉证据的对抗性攻击。
- **Query-Irrelevant Semantic Editing（查询无关语义编辑）**：将攻击文本替换为同时与查询和图像内容无关的语义内容，以消除误导影响。
- **Recovery Rate (RR)**：防御成功恢复的攻击错误比例，衡量攻击恢复能力。
- **Harm Rate (HR)**：防御导致原本正确回答变为错误的比例，衡量防御副作用。
- **Influence-Aware Text Localization（影响力感知文本定位）**：利用目标 VLM 判断哪些文本区域可能影响当前查询的答案。
- **Q0-I0 条件**：替换文本同时与查询无关（Q0）和图像无关（I0）的语义条件。
- **Consensus-based Answer Selection（基于共识的答案选择）**：仅在两次编辑图像的答案一致且合法时才采纳，否则回退原始答案。

## 可复现要素
- **数据集**：SCAM、SceneTAP、SELF、TextVQA、ST-VQA、FGVC-Aircraft、Food-101、ImageNet-100（部分为公开数据集，需查阅原链接获取访问方式）。
- **代码/权重**：论文未明确提及代码开源状态。
- **关键超参/组件**：
  - 辅助定位：PP-OCRv6（PaddleOCR 3.7.0）
  - 文本移除：LaMa（Suvorov et al., 2022）
  - 文本渲染：RS-STE（Fang et al., 2025）
  - 推理设备：单 NVIDIA vGPU-32GB
- **实现细节**：详见附录 B，包括定位失败处理、Q0-I0 候选选择策略、几何兼容性约束等。
