---
title: "SemComp-Bench-Benchmarking-Semantic-Task-Completion-in-Video"
source: https://arxiv.org/pdf/2608.17426v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 01:04:30"
field: "视频生成评测基准"
keywords: ["视频生成", "语义任务完成", "视频评估基准", "VLM评估", "语义grounding", "结果达成"]
innovations: ["提出语义任务完成视频生成新任务，联合评估结果达成与语义grounding", "构建可扩展四阶段数据流水线SemComp-Data，自动从原始视频生成评测三元组", "设计VLM驱动的双维度评估协议SemComp-Bench，支持可解释的失败诊断"]
benchmarks: ["SemComp-Bench", "SemComp-Data", "Koala-36M"]
---

# 论文速读：SemComp-Bench: Benchmarking Semantic Task Completion in Video Generation

## 一句话总结
论文提出"语义任务完成视频生成"新任务，要求生成视频不仅达成指令规定的预期结果，还需与参考图像保持任务相关的语义关联。基于此构建了 SemComp-Data 数据集与 SemComp-Bench 评估协议，揭示了当前主流视频生成模型在结果达成与语义 grounding 方面仍存在显著不足。

## 研究问题与动机
1. **现有基准侧重视觉保真度，忽视任务导向的结果达成**：视频生成基准（如 VBench、EvalCrafter 等）主要评估视觉质量、时间连贯性和提示遵循能力，但缺乏对"在给定参考图像与指令下实现预期结果"的系统性评估。
2. **语义 grounding 机制尚未被充分研究**：生成视频应在保留任务相关语义关系的同时允许无关属性变化，但现有工作未对这一细粒度语义约束进行形式化定义与评估。
3. **需要一种关注结果而非过程的新范式**：语义任务完成不要求生成中间步骤的完整序列，而是聚焦于最终状态的可见呈现与语义对应，更适合实际应用场景评估。
4. **开放与闭源模型均需统一评测框架**：当前缺乏可同时评估多类模型（开源如 Wan2.2、CogVideoX；闭源如 Seedance、HunyuanVideo）在复杂任务型生成上能力的标准化基准。

## 核心贡献（创新点）
1. **提出语义任务完成视频生成新任务**：形式化定义 Outcome Achievement 与 Semantic Grounding 联合约束，与现有注重外观一致性或时间连贯性的工作本质不同。
2. **构建 SemComp-Data 可扩展数据流水线**：从 Koala-36M 原始视频出发，通过候选过滤、状态挖掘、视频扩展、指令结构化四阶段自动生成图像-文本-视频三元组，无需人工逐任务设计。
3. **设计 VLM 驱动的 SemComp-Bench 评估协议**：采用结构化二元问题与证据支撑判断，分别评估 OA Score（结果达成）与 GR Score（生成可靠性），支持可解释的失败诊断。
4. **揭示当前模型在语义任务完成上的能力缺口**：最高 OA Score 仅 37.8%（HunyuanVideo-1.5），且 GR 与 OA 排名不一致，表明生成可靠性不等于任务完成能力。

## 方法详解
### 数据构建：四阶段 SemComp-Data 流水线
1. **候选过滤（Candidate Filtering）**：从 Koala-36M 随机采样约 20K 视频，基于标题关键词过滤排除依赖旁白的视频；使用 VLM 对视频摘要进行六领域分类（Arts、Beauty、Crafts、Food、Gardening、Sports），丢弃不确定类别。
2. **状态挖掘（State Mining）**：根据人工定义的类别状态规范，VLM 定位参考帧与结果帧的时间戳；通过 QA 质检确保结果帧可被无歧义识别，否则保守丢弃。
3. **视频扩展（Video Extension）**：以结果时间戳为锚点，按镜头检测与同景融合策略（参考 Panda-70M）提取时长约 3–4 秒的结果为中心视频片段。
4. **指令结构化（Instruction Structuring）**：
   - **归一化关系描述**：限制简要指令 ≤30 词，模板为 `[动词] + [参考主客体] + [介词] + [结果主状态]`。
   - **属性选择**：VLM 从四类对齐类型（Object Element、Person Identity、Object Appearance、Scene）中选择保留/丢弃的属性。
   - **指令组合**：基于保留属性与结果细节生成长指令，形成 brief/detailed 成对指令。

### 评估协议：SemComp-Bench
- **OA Score（结果达成）**：四项标准均为通过编码（全通过才计 1），样本级得分 $A^i = a_{or}^i \cdot a_{sg}^i \cdot a_{gec}^i \cdot a_{gvc}^i$，数据集平均即为 OA Score。
  - 结果实现（Outcome Realization）：可见达成指令指定的完成状态。
  - 语义 grounding（Semantic Grounding）：结果保留或修改与参考-指令对相关的实体/属性。
  - 锚定实体一致性（Grounded Entity Consistency）：任务相关关键实体在整个序列中保持语义可识别。
  - 全局视觉连续性（Global Visual Continuity）：无突兀场景/视角/布局切换。
- **GR Score（生成可靠性）**：五项标准的算术平均，均为失败导向（答"No"计 1）。
  - 物理合理性、视觉清晰度、无伪影渲染、场景内时空相干性、文本/界面完整性。
- **评估设置**：每视频均匀采样 27 帧，由 VLM（Doubao-Seed-1.8）独立调用三次取平均，提供视觉证据支撑二元判断。

## 实验与结果
- **数据集规模**：1,273 个结构化实例；SemComp-Core 为 60 实例（每领域 10 个）。
- **OA Score 结果**（详细指令）：
  - 最佳：**HunyuanVideo-1.5-720P-I2V** 达 **37.8%**（次佳 Wan2.2-I2V-A14B 28.3%，Seedance 2.0 20.0%）。
  - 所有模型 OA Score 均低于 40%，表明结果达成仍是挑战。
- **GR Score 结果**（详细指令）：
  - 最佳：**Seedance 2.0** 达 **91.8%**（次佳 Wan2.2-I2V-A14B 89.0%）。
  - 共同瓶颈：场景内时空相干性（$G_{wsc}$）通过率仅 0.328–0.739。
- **条件效应**：
  - I2V 显著优于 T2V（详细指令）：Wan2.2 I2V 28.3% vs T2V 4.4%；HY I2V 37.8% vs T2V 4.4%。
  - 详细指令 vs 简要指令：详细指令提升结果实现与语义 grounding，但简要指令的实体一致性与视觉连续性更高。
- **GR 与 OA 解耦**：Seedance 2.0 GR 第一但 OA 第三；HY OA 第一但 GR 中等，说明两项指标互补。

## 相关工作脉络
1. **视频生成基准**（VBench/TC-Bench/EvalCrafter）：侧重视觉质量、时间组合性与提示遵循，未评估任务导向的结果达成与语义 grounding，本文填补此空白。
2. **物理常识评估**（Videophy-2/Impossible Videos）：关注物理合理性，但不涉及参考图像驱动的语义锚定任务完成，本文将物理约束纳入 GR 维度。
3. **主体一致性基准**（OpenS2V-Nexus/VMBench）：聚焦身份保持，本文扩展至任务相关属性的细粒度保留/丢弃控制。
4. **世界模型评估**（WorldModelBench/WorldOlympiad）：评估因果与常识推理，本文聚焦单任务实例的结果可达性验证。
5. **现有视频生成模型**（Wan2.2/CogVideoX/HunyuanVideo/Seedance 等）：展示强合成能力，但本文首次系统性揭示其在语义任务完成上的局限。

## 局限性与未来方向
1. **数据集规模有限**：SemComp-Core 仅 60 实例，虽覆盖六领域，但统计效力与泛化性有待更大规模验证。
2. **VLM 评估器的潜在偏差**：依赖单一 VLM（Doubao-Seed-1.8）进行二元判断，可能存在模型特定偏见或文化/语言因素干扰。
3. **未探索微调/训练可能性**：结论仅停留在评估层面，SemComp-Data 用于任务特定训练以提升性能的效果尚未验证。
4. **结果为中心剪辑长度受限**：平均 4 秒 clips 不足以呈现复杂中间过程，可能低估模型在长程任务规划上的能力。
5. **领域覆盖不均**：当前六领域偏向手工/烹饪类任务，对叙事性、交互性场景覆盖不足。

## 研究启发与可借鉴点
1. **四阶段自动化数据构建范式可迁移**：从原始视频到结构化评测样本的流水线设计，尤其状态挖掘+QA质检的去歧义策略，可复用于其他视频理解任务的数据准备。
2. **OA/GR 双维度解耦评估思路**：将"任务完成"与"生成质量"分离评估，避免高视觉质量掩盖任务失败，此框架可扩展至图像生成、3D 生成等领域。
3. **成对指令（brief/detailed）的设计价值**：控制指令特异性变量而不改变参考/目标，可系统研究提示工程对结果达成的影响。
4. **证据支撑的二元判断协议**：要求 VLM 提供视觉证据而非仅输出分数，提升评估可解释性与失败诊断能力，值得在更多 benchmark 中推广。
5. **与团队方向的结合机会**：若团队关注视频生成的可控性/任务导向生成，可基于 SemComp-Bench 评估现有模型的语义 grounding 能力，或探索引入类似约束的微调策略。

## 关键术语表
**Semantic Task Completion Video Generation**：要求生成视频在视觉上达成指令指定结果，同时与参考图像保持任务相关语义关联的新型视频生成任务。
**Outcome Achievement (OA)**：评估维度，综合衡量结果实现、语义 grounding、实体一致性与视觉连续性四项标准。
**Generation Reliability (GR)**：评估维度，独立衡量物理合理性、清晰度、无伪影、时空相干性与文本完整性。
**SemComp-Data**：从 Koala-36M 构建的大规模图像-文本-视频三元组评测数据集，包含简要与详细成对指令。
**SemComp-Bench**：基于 VLM 的结构化二元判断评估协议，输出 OA Score 与 GR Score。
**State Mining**：数据构建第二阶段，利用 VLM 定位参考帧与结果帧时间戳并通过 QA 质检验证配对可靠性。
**Alignment Type**：属性选择中的四类语义 grounding 模式，包括 Object Element、Person Identity、Object Appearance、Scene。
**OA Score**：数据集级别结果达成得分，定义为所有样本在四项 OA 标准下全部通过的比例。

## 可复现要素
- **数据集**：SemComp-Data 共 1,273 实例；SemComp-Core 60 实例，原始数据与元信息在补充材料中提供。
- **代码**：完整流水线与阶段提示词在 Code and Data Supplementary 的 Code 文件夹；数据处理代码已开源。
- **权重**：使用各模型官方默认配置；评估 VLM 为 Doubao-Seed-1.8（通过火山引擎 API）。
- **关键超参**：采样帧数 27 帧/视频；clip 时长目标 3–4 秒；简要指令 ≤30 词；VLM 独立调用 3 次取平均。
