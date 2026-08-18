---
title: "VicEdit-Learning-to-Edit-Videos-from-Visual-In-Context-Examp"
source: https://arxiv.org/pdf/2608.16745v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 06:22:43"
field: "视频编辑与生成"
keywords: ["视频编辑", "视觉上下文学习", "多模态生成", "Diffusion Transformer", "in-context editing", "视频生成"]
innovations: ["提出视觉上下文视频编辑新范式，统一支持单图像/图像对/视频对三种异构参考模态", "设计模态自适应语义蒸馏(MASD)，通过模态条件化可学习查询自适应提取异构参考的编辑语义", "设计双上下文注入(DCI)，用视觉上下文偏移校准文本指令查询，实现视觉与文本信号的协同融合"]
benchmarks: ["VicEdit-Bench", "OpenVE-Ext", "VBench", "IVE-bench"]
---

# 论文速读：VicEdit: Learning to Edit Videos from Visual In-Context Examples

## 一句话总结
本文提出了视觉上下文编辑（Visual In-context Editing）新范式，将视频编辑从纯文本指令扩展至支持单图像、图像对和视频对三种多模态视觉参考；构建了首个大规模数据集 VicEdit-400K（40万高质量样本，覆盖3种模态与10种任务类型），并设计了包含模态自适应语义蒸馏（MASD）与双上下文注入（DCI）的统一框架，在基础指令编辑与视觉上下文编辑任务上均取得 SOTA。

## 研究问题与动机
1. **文本指令的感知鸿沟**：纯文本指令无法精确锚定细粒度纹理（如特定材质、笔触风格），现有文本基线（DiTTO、VideoCoF）常出现风格漂移。
2. **空间变换的定位缺失**：语言仅描述"改什么"而不指定"改在哪"，图像对参考能直观定义空间锚点，但现有 MLLM 基线（Kiwi-Edit 等）未能有效利用。
3. **复杂时序动态的表达瓶颈**：创意编辑中的复杂运动模式超出语言描述能力，且已有参考驱动方法（如 Kiwi-Edit）仅支持静态单图像模态，未覆盖时序维度。
4. **数据与基准的结构性匮乏**：现有数据集（OpenVE-3M、Señorita-2M）仅含文本指令或单一参考模态，缺乏覆盖多模态视觉参考的大规模训练数据与评估基准。

## 核心贡献（创新点）
1. **提出视觉上下文视频编辑新范式**：将编辑接口从"描述性 telling"升级为"示程式 showing"，统一支持单图像、图像对、视频对三种异构视觉参考模态。与 Kiwi-Edit 等仅支持单一参考模态的工作形成本质区别。
2. **构建 VicEdit-400K 数据集**：首个覆盖3种参考模态与10种任务类型的40万高质量数据集，通过自动化四阶段流水线（源数据过滤→指令合成→视觉参考生成→语义验证）确保视觉保真度与语义一致性。
3. **设计模态自适应语义蒸馏（MASD）**：通过模态条件化的可学习查询 $\mathbf{Q}_{\text{vis}} = \mathbf{Q}_{\text{base}} + \mathbf{E}_m$ 自适应提取异构参考的编辑语义，相比 Kiwi-Edit 等固定查询方案，能区分局部空间变换（图像对）与全局风格纹理（单图像）。
4. **设计双上下文注入（DCI）**：通过语义偏移机制 $\mathbf{Q}'_{\text{inst}} = \mathbf{Q}_{\text{inst}} + \text{MLP}(\mathbf{Q}_{\text{vis}})$ 用视觉上下文校准文本指令查询，使指令分支能捕捉参考中具体的色彩/纹理实例化，解决文本与视觉信号分离的歧义问题。

## 方法详解
**整体架构**：基于冻结的 MLLM（Qwen3-VL）与 Video DiT（Wan-TI2V-5B），通过两个轻量模块（MASD 与 DCI）桥接视觉与文本上下文，tokens 以 cross-attention 形式注入 Video DiT。

**Modality-Adaptive Semantic Distillation (MASD)**：
- 为三种模态（单图像 si、图像对 ip、视频对 vp）设计模态特定的可学习嵌入 $\mathbf{E}_m$，与共享基础查询 $\mathbf{Q}_{\text{base}}$ 相加：$\mathbf{Q}_{\text{vis}} = \mathbf{Q}_{\text{base}} + \mathbf{E}_m$。
- 通过 cross-attention 从 MLLM 视觉隐藏状态 $\mathbf{H}_{\text{vis}}$ 中提取 $N$ 个语义 tokens：$\mathbf{T}_{\text{vis}} = \text{CrossAttn}(\mathbf{Q}_{\text{vis}}, \mathbf{H}_{\text{vis}}, \mathbf{H}_{\text{vis}})$。
- 加性偏移将搜索空间分解为通用编辑流形与模态特定子区域，$\mathbf{Q}_{\text{base}}$ 捕获"改什么"，$\mathbf{E}_m$ 聚焦"在哪里看"。

**Dual-Context Injection (DCI)**：
- 用 MASD 的 $\mathbf{Q}_{\text{vis}}$ 经轻量 MLP 生成偏移向量：$\Delta_{\text{inst}} = \text{MLP}(\mathbf{Q}_{\text{vis}})$。
- 对指令查询进行语义校准：$\mathbf{Q}'_{\text{inst}} = \mathbf{Q}_{\text{inst}} + \Delta_{\text{inst}}$。
- 从文本 prompt 的 MLLM 隐藏状态 $\mathbf{H}_{\text{inst}}$ 中提取指令 tokens：$\mathbf{T}_{\text{inst}} = \text{CrossAttn}(\mathbf{Q}'_{\text{inst}}, \mathbf{H}_{\text{inst}}, \mathbf{H}_{\text{inst}})$。
- 最终拼接双上下文 tokens $\mathbf{T}_{\text{dual}} = [\mathbf{T}_{\text{vis}}; \mathbf{T}_{\text{inst}}]$ 作为 DiT cross-attention 的 KV。

**三阶段训练策略**：
1. **基础编辑**：在文本指令编辑任务上微调 Video DiT（LoRA，rank-64，LR=$1\times10^{-5}$，2 epoch），MLLM 冻结。
2. **上下文分支**：冻结 DiT 与 MLLM，从头训练 MASD 与 DCI（LoRA rank=64，4 epoch），每 batch 混合三种模态样本。
3. **联合微调**：解冻 DiT，联合微调全部模块（MLLM vision encoder 保持冻结，DiT LoRA rank=32，MLLM LoRA rank=64，4 epoch），优化标准去噪损失 $\mathcal{L} = \mathbb{E}[\|\epsilon - \epsilon_\theta(\mathbf{V}_e^{(t)}, t, \mathbf{T}_{\text{dual}})\|^2]$。

**数据集构建流水线**（四阶段）：
1. 源数据过滤：LAION-Aesthetic 预测器 + Qwen3-VL 双模型筛选高美学/高语义可编辑性样本。
2. 指令合成：Qwen3-VL-32B 生成模态感知的结构化指令，确保"类别一致但内容发散"。
3. 视觉参考生成：单图像/图像对使用 FLUX.2 生成（约4s/GPU）；视频对通过语义知识图谱 + Spectral Clustering 构建同义链接对。
4. 语义验证：Qwen3-VL-32B 多维度评分（指令遵循、细节保真、视觉质量），设瓶颈约束（后两项得分不超过第一项）。

## 实验与结果
**数据集与基准**：
- **VicEdit-400K**：40万样本，覆盖3种模态 × 10种任务类型（Global Style、Local Style、Background Change、Local Add、Local Change、Local Remove、Subtitle Edit、Inpainting、Outpainting、Creative Edit）。
- **VicEdit-Bench**：200个测试样本（每种任务-模态组合10个），与训练集零重叠。
- **OpenVE-Ext**：在 OpenVE-Bench 基础上扩展3个任务。

**评估指标**：MLLM-based（Overall、Instr. Comp.、Detail Fid.、Visual Qual.）+ VBench（BC、MS、TF、IQ、AQ）。

**主要结果**：
- **OpenVE-Ext（基础编辑）**：VicEdit Overall 3.28，超越次优 Kiwi-Edit（3.08）约 0.20；在 Global Style（3.48 vs 3.64）、Local Change（3.85 vs 3.83）等子任务上全面领先或持平。
- **VicEdit-Bench（视觉上下文编辑）**：VicEdit Overall **3.45**，超越次优 Kiwi-Edit（2.97）**0.48 分（相对提升 16%）**；在 Background Change（3.05 vs 2.53）、Subtitle Edit（3.62 vs 3.16）、Outpainting（3.47 vs 2.59）等任务上优势显著。
- **交叉验证**：使用 Qwen-VL-32B 替代原 MLLM 评测，VicEdit 仍保持最高分（4.40 OpenVE-Ext / 4.36 VicEdit-Bench），相对排名高度一致。
- **计算效率**：MLLM 特征提取仅额外耗时 0.5–1.0s（三种模态），占总推理时间 <0.5%，与 Kiwi-Edit（~0.4s）相当；DiT 去噪时间保持 ~4 min（81帧，480×832）。

**消融结论**：
- MASD 单独提升 Detail Fid.（3.20→3.34），DCI 单独提升 Instr. Comp.（3.40→3.56）；二者联合达最优。
- 三阶段训练中，Stage 2 上下文学习带来 VIC 指标跃升（2.97→3.40），但 Base 编辑略降（3.23→3.10）；Stage 3 联合微调实现两者平衡。

## 相关工作脉络
1. **Kiwi-Edit [13]**：首个引入参考图像的指令视频编辑方法，但仅支持单图像模态且使用固定查询提取语义；VicEdit 扩展至三种异构模态并通过 MASD 实现模态自适应。
2. **EditVerse [19]**：利用全自注意力处理 in-context tokens，但将所有参考模态视为同质输入；VicEdit 通过 MASD 的模态条件化查询实现异构语义区分。
3. **OmniTransfer [18]**：使用视频参考进行时空视频 transfer，但面向从噪声生成而非编辑现有视频；VicEdit 专注于 source video → edited video 的编辑范式。
4. **DITTO [10]**：大规模文本指令视频编辑，擅长风格迁移但在空间编辑上泛化差；VicEdit 在保持风格能力的同时补充了多模态参考增强。
5. **VideoCoF [4]**：基于时序链式推理的文本编辑，缺乏视觉参考支持；VicEdit 的 DCI 模块使时序逻辑可与视觉 exemplar 协同。
6. **Painter [21] / ImageBrush [22]**：图像领域的 in-context learning 先行者，证明视觉示例优于纯文本；VicEdit 将此范式系统性地推广至视频编辑且覆盖更丰富的模态组合。

## 局限性与未来方向
1. **语义级编辑为主**：当前设计主要面向语义层面的编辑，尚未支持像素级精度的细粒度控制（如局部纹理替换、边缘对齐）。
2. **参考序列长度限制**：虽通过 Distillation 压缩了参考信息，但长序列视频对的时序建模仍受限于 MLLM 的上下文窗口。
3. **数据集规模瓶颈**：400K 相对于纯文本编辑数据集（OpenVE-3M、Señorita-2M）仍较小，可能制约模型的泛化上限。
4. **任务-模态映射固定**：当前任务类型与支持的参考模态为预设规则（如 Inpainting/Outpainting 仅支持视频对），缺乏动态匹配机制。
5. **未来方向**：可扩展至像素级编辑、增大数据集规模、探索更少样本的 few-shot in-context 场景。

## 研究启发与可借鉴点
1. **模态条件化查询设计**：MASD 的 $\mathbf{Q}_{\text{base}} + \mathbf{E}_m$ 加性偏移策略简洁有效，可迁移至其他多模态 in-context 学习场景（如图像编辑、多视角生成），避免为每种模态设计独立分支。
2. **语义校准机制（DCI）**：用视觉上下文偏移来校准指令查询的思想，可推广至任何需要文本-视觉联合条件生成的任务（如文生视频、图像修复），解决文本歧义问题。
3. **VLM-as-a-Judge 数据管道**：四阶段流水线中引入多维度 MLLM 评分与"瓶颈约束"（Consistency/Quality ≤ Instruction Compliance），确保数据一致性，可复用于其他视觉生成数据集构建。
4. **三阶段渐进训练策略**：Base → Context Branch → Joint Fine-tuning 的 curriculum 设计有效缓解了多任务间的干扰，对训练同时支持多种条件输入的多模态模型具有参考价值。
5. **评测协议设计**：VicEdit-Bench 以"每种任务-模态组合10个零样本样本"的严格设定评估泛化能力，其 VLM 自动化评分+VBench 客观指标的组合值得借鉴。

## 关键术语表
**Visual In-Context Editing**：利用视觉示例（单图像/图像对/视频对）作为上下文指导视频编辑的新范式，弥补纯文本指令的感知鸿沟。

**Modality-Adaptive Semantic Distillation (MASD)**：通过模态条件化的可学习查询（$\mathbf{Q}_{\text{base}} + \mathbf{E}_m$）从 MLLM 隐藏状态中自适应提取异构视觉参考的语义 tokens。

**Dual-Context Injection (DCI)**：利用视觉上下文偏移量校准文本指令查询，通过 cross-attention 将视觉 tokens 与指令 tokens 协同注入 Video DiT 的机制。

**VicEdit-400K**：首个大规模视觉上下文视频编辑数据集，包含40万高质量样本，覆盖3种参考模态与10种编辑任务类型。

**VicEdit-Bench**：为评估视觉上下文编辑能力构建的标准化测试基准，含200个与训练集零重叠的样本，每种任务-模态组合10个。

**In-Context Learning (ICL)**：模型通过观察输入中的示例（demonstrations）而非显式编程来执行任务的范式，本文将其从 NLP 扩展至视频编辑。

**MLLM（多模态大语言模型）**：本文采用 Qwen3-VL-32B 作为语义理解 backbone，提取文本与视觉的跨模态对齐表征。

**Video DiT**：基于 Diffusion Transformer 的视频生成 backbone，本文继承 Wan-TI2V-5B 预训练权重，通过 cross-attention 接受双上下文条件。

## 可复现要素
- **数据集**：VicEdit-400K 与 VicEdit-Bench —— 论文提供了项目页面（https://rain152.github.io/VicEdit/），开源状态需在项目中确认。
- **代码**：论文未明确声明代码开源状态，仅提供了项目页面链接。
- **权重**：基于 Wan-TI2V-5B 与 Kiwi-Edit 预训练权重，VicEdit 自身权重开源状态论文未提及。
- **关键超参**：LoRA rank（DiT rank-32，MLLM rank-64）、学习率 $1\times10^{-5}$、训练 epoch（Stage 1: 2, Stage 2: 4, Stage 3: 4）、输入分辨率 480×832–1120×1984、帧数 81 frames（训练）/ 1280×704（推理泛化）。
- **训练硬件**：16× GPU，DeepSpeed ZeRO-3，BF16 混合精度，总耗时约 32 GPU days。
