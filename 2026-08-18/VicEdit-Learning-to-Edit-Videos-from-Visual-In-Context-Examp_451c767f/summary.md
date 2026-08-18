---
title: "VicEdit-Learning-to-Edit-Videos-from-Visual-In-Context-Examp"
source: https://arxiv.org/pdf/2608.16745v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 15:26:35"
field: "视频编辑与生成"
keywords: ["视频编辑", "视觉上下文学习", "多模态生成", "Diffusion Transformer", "in-context editing", "VicEdit"]
innovations: ["提出Visual In-context Editing统一范式，支持单图/图像对/视频对三种异构参考模态", "设计MASD模态自适应语义蒸馏机制，通过可学习模态嵌入调制查询实现异构参考的自适应提取", "设计DCI双上下文注入机制，用视觉上下文校准指令查询，解决文本-视觉语义对齐歧义"]
benchmarks: ["VicEdit-Bench", "OpenVE-Ext", "VBench"]
---

# 论文速读：VicEdit: Learning to Edit Videos from Visual In-Context Examples

## 一句话总结
本文提出**Visual In-context Editing**新范式，将视频编辑从纯文本指令扩展到多模态视觉引导（单图、图像对、视频对），构建了首个大规模数据集**VicEdit-400K**（40万高质量样本），并设计统一框架**VicEdit**，在VicEdit-Bench上以**3.45总体得分**刷新SOTA，相对次优方法提升**16%**。

---

## 研究问题与动机

1. **纯文本指令的感知鸿沟**：语言无法精确锚定细粒度纹理和复杂动态——如风格漂移、空间位置模糊、时序运动模式难以用文字描述。
2. **架构推理缺陷**：现有方法（a）依赖纯文本语义，（b）使用刚性辅助分支处理单图参考，缺乏跨异构视觉上下文灵活推理的能力。
3. **数据资源匮乏**：现有基准（如Kiwi-Edit的RefVIE-477K）仅限文本或单图驱动，缺乏覆盖多模态参考的大规模训练数据。
4. **任务覆盖不足**：现有参考类工作仅支持单一模态（如Kiwi-Edit仅单图），无法统一处理空间变换（图像对）和时序动态（视频对）。

---

## 核心贡献（创新点）

1. **提出Visual In-context Editing统一范式**：将编辑接口从"描述"转向"展示"，首次在同一框架内支持单图、图像对、视频对三种异构参考模态。
2. **构建VicEdit-400K首个大规模视觉上下文数据集**：覆盖10种任务类型、3种参考模态，通过自动化四阶段流水线（源数据筛选→指令合成→参考生成→语义验证）保障质量。
3. **设计Modality-Adaptive Semantic Distillation (MASD)**：通过模态条件化可学习查询（$Q_{vis} = Q_{base} + E_m$）自适应提取异构参考的编辑语义，解决固定查询无法区分空间局部变换与全局风格纹理的问题。
4. **设计Dual-Context Injection (DCI)**：利用视觉上下文对指令查询施加语义偏移（$\Delta_{inst}$），使文本token在视觉感知流形中校准，解决文本-视觉歧义对齐问题。
5. **建立VicEdit-Bench标准化评测基准**：200个样本（每任务-模态组合10个），严格与训练集隔离，支持zero-shot泛化评估。

---

## 方法详解

### 整体架构
VicEdit基于冻结的**MLLM**（语义理解）和**Video DiT**（生成），通过两个轻量模块连接：MASD提取视觉token $T_{vis}$，DCI生成指令token $T_{inst}$，二者拼接为 $T_{dual} = [T_{vis}; T_{inst}]$ 作为cross-attention的KV注入DiT。

### MASD：模态自适应语义蒸馏
针对异构参考结构差异，通过模态条件化shift调制基础查询：
$$Q_{vis} = Q_{base} + E_m, \quad m \in \{ip, vp, si\}$$
$$T_{vis} = \text{CrossAttn}(Q_{vis}, H_{vis}, H_{vis})$$
- $Q_{base}$：跨模态通用编辑语义（"改什么"）
- $E_m$：模态专属嵌入（"在哪里/何时看"）
- 解耦搜索空间为公共编辑流形和模态子区域，避免静态查询的语义模糊。

### DCI：双上下文注入
用视觉上下文校准文本指令提取，解决纯文本歧义：
$$\Delta_{inst} = \text{MLP}(Q_{vis})$$
$$Q'_{inst} = Q_{inst} + \Delta_{inst}$$
$$T_{inst} = \text{CrossAttn}(Q'_{inst}, H_{inst}, H_{inst})$$
- $\Delta_{inst}$将视觉感知投影到指令提取空间，使 $T_{inst}$ 在视觉上aware的流形中抽取。
- 两序列天然对齐，DiT通过标准attention层协同处理。

### 三阶段训练策略
1. **Base Editing**：微调Video DiT（LoRA rank=64），MLLM冻结，2 epoch，LR=$1\times10^{-5}$，建立基础指令遵循能力。
2. **In-Context Branch**：冻结DiT和MLLM，训练MASD（$Q_{base}$、$\{E_m\}$）、DCI MLP、$Q_{inst}$，每batch混合三模态，4 epoch。
3. **Joint Fine-Tuning**：解冻DiT，联合微调全管线（除MLLM vision encoder），LoRA rank=32，4 epoch。
- 总训练：16×GPU，DeepSpeed ZeRO-3，BF16，约32 GPU天。
- 损失函数：标准去噪损失 $\mathcal{L} = \mathbb{E}[\|\epsilon - \epsilon_\theta(V_e^{(t)}, t, T_{dual})\|^2]$。

---

## 实验与结果

### 数据集与评测
- **训练数据**：VicEdit-400K（400K样本，10任务类型）
- **评测基准**：OpenVE-Ext（基础编辑）+ VicEdit-Bench（视觉上下文编辑，200样本，零样本）
- **评估指标**：MLLM自动评分（Instruction Compliance、Detail Fidelity、Visual Quality等）+ VBench（BC、MS、TF、IQ、AQ）

### 主要结果

| 任务 | 最优方法 | VicEdit得分 | 提升幅度 |
|------|----------|-------------|----------|
| **基础编辑** (OpenVE-Ext) | Kiwi-Edit 3.08 | **VicEdit 3.28** | +0.20 |
| **视觉上下文编辑** (VicEdit-Bench) | Kiwi-Edit 2.97 | **VicEdit 3.45** | **+0.48 (16%↑)** |

- VicEdit在所有子任务（全局风格、局部风格、背景变化、局部增/删/改、字幕编辑、创意编辑、 inpainting/outpainting）均达SOTA。
- 跨模型验证（Qwen-VL-32B替代原评测模型）：VicEdit仍保持最高分4.40/4.36，相对排名一致。
- **计算效率**：MLLM提取仅增0.5–1.0s（<0.5%总推理时间），DiT去噪时间不变（~4min）。

### 消融结论
- **MASD**：Detail Fid从2.98→3.34，恢复细粒度纹理与结构保真。
- **DCI**：Instr. Comp从3.12→3.65，显著提升指令遵循精度。
- **两模块组合**：Overall达3.45，互补效应明显（MASD负责纹理/结构，DCI负责语义对齐）。
- **三阶段训练**：Stage 1强化基础能力，Stage 2跃升VIC指标，Stage 3联合微调达成平衡稳定。

---

## 相关工作脉络

1. **Kiwi-Edit [13]**：首个引入单图参考的指令视频编辑方法，使用固定可学习查询（768个ref token）提取语义。→ VicEdit扩展为三模态支持，并用MASD替代固定查询实现模态自适应。
2. **EditVerse [19]**：通过交错token和全自注意力统一图文编辑。→ VicEdit避免序列增长导致的注意力稀释，采用cross-attention轻量注入，且支持异构模态。
3. **OmniTransfer [18]**：利用视频参考进行时空视频transfer，但面向从噪声生成而非编辑已有视频。→ VicEdit聚焦于source-video-to-target-video的editing任务。
4. **TIC-FT [5]**：时序in-context微调，沿时间轴拼接条件/目标帧。→ VicEdit统一处理图像和视频参考，并引入模态条件化机制。
5. **Painter/ImageBrush**：图像生成中的visual ICL范式。→ VicEdit将其拓展至视频编辑领域，新增时序维度挑战。

---

## 局限性与未来方向

1. **语义级编辑为主**：当前设计主要面向语义层面编辑，尚未充分探索需像素级精度的细粒度场景（如边缘对齐、局部光照匹配）。
2. **数据集规模可扩展**：400K样本虽为首次大规模覆盖，但相比OpenVE-3M等纯文本数据集仍显不足，可扩展至百万级。
3. **仅支持三种参考模态**：未探索多参考组合（如同时提供单图+图像对）或更复杂的多轮交互上下文。
4. **MLLM依赖**：评测和训练均依赖Qwen3-VL等大模型，推理成本较高，未来可探索轻量化替代方案。

---

## 研究启发与可借鉴点

1. **模态条件化查询设计**：MASD的 $Q_{base} + E_m$ 加法调制思想可迁移至其他多模态适配任务（如跨模态检索、多源融合），实现"通用语义+模态专属偏置"的参数高效解耦。
2. **双上下文语义校准机制**：DCI的"用视觉上下文偏移指令查询"思路可用于图文匹配、跨模态对齐等场景，解决纯文本歧义问题。
3. **三阶段渐进式训练策略**：Base→In-Context Branch→Joint Fine-Tuning的分阶段 curriculum learning 设计，可复用于需要逐步引入新条件的多模态模型训练。
4. **自动化数据构建流水线**：四阶段 pipeline（源筛选→指令合成→参考生成→语义验证）及"Category-Consistent but Content-Divergent"参考生成策略，为大规模多模态数据集构建提供可复用模板。
5. **VBench+MLLM双重评测体系**：结合传统视频质量指标（BC/MS/TF/IQ/AQ）与MLLM语义评分，形成多维、可复现的评测协议，值得借鉴到视频生成相关工作中。

---

## 关键术语表

**Visual In-context Editing (视觉上下文编辑)**：利用单图/图像对/视频对等视觉示例作为引导，配合文本指令进行视频编辑的新范式。

**Modality-Adaptive Semantic Distillation (MASD)**：通过模态条件化可学习查询（$Q_{base}+E_m$）从异构视觉参考中自适应提取编辑语义token的模块。

**Dual-Context Injection (DCI)**：利用视觉上下文对文本指令查询施加语义偏移（$\Delta_{inst}$），使提取的指令token与视觉感知对齐的双上下文融合机制。

**VicEdit-400K**：首个大规模视觉上下文视频编辑数据集，含40万高质量样本，覆盖3种参考模态和10种任务类型。

**VicEdit-Bench**：包含200个严格与训练集隔离样本的标准化评测基准，用于zero-shot评估视觉上下文编辑性能。

**OpenVE-Ext**：在OpenVE-Bench基础上扩充3个任务类型（来自Señorita）形成的基础编辑评测集。

**Text-to-Video DiT (TI2V)**：基于Diffusion Transformer的视频生成架构，本文基于Wan-TI2V-5B进行微调。

**VLM-as-a-Judge**：利用视觉语言模型（Qwen3-VL）作为自动评分器，从Instruction Compliance、Consistency & Detail Fidelity、Visual Quality三个正交维度评估生成质量。

---

## 可复现要素

- **数据集**：VicEdit-400K（论文声称已构建，项目页面 https://rain152.github.io/VicEdit/，开源状态论文未明确声明）；VicEdit-Bench（200样本）
- **代码/权重**：项目页面链接已提供，具体开源状态论文未明确声明
- **关键超参**：
  - Stage 1：LoRA rank=64（MLLM），DiT更新，2 epoch，LR=$1\times10^{-5}$
  - Stage 2：LoRA rank=64（MLLM），MASD+DCI从头训练，4 epoch，LR=$1\times10^{-5}$
  - Stage 3：LoRA rank=32（DiT），联合微调，4 epoch，LR=$1\times10^{-5}$
  - 每样本81帧，BF16混合精度，DeepSpeed ZeRO-3，16×GPU，约32 GPU天
  - 训练分辨率：480×832–1120×1984（多桶分辨率）

---
