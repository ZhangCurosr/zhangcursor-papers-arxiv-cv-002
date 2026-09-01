---
title: "StateTrace-An-Object-Centric-Framework-for-Hidden-State-Spat"
source: https://arxiv.org/pdf/2608.18532v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 06:14:39"
field: "视频理解与时空推理"
keywords: ["Video Understanding", "VLM", "Hidden-State Reasoning", "Spatiotemporal Reasoning", "Object-Centric", "Long Video QA"]
innovations: ["提出StateTrace对象中心框架，构建可复用时空状态记忆实现隐藏状态显式推理", "构建HSR-Bench诊断基准，专门评测遮挡/不可见场景下的状态持久性推理能力"]
benchmarks: ["HSR-Bench", "MLVU", "VideoMME", "LongVideoBench"]
---

# 论文速读：StateTrace-An-Object-Centric-Framework-for-Hidden-State-Spatiotemporal-Reasoning-in-Long-Videos

## 一句话总结
StateTrace 是一个面向长视频的以对象为中心的框架，通过构建可复用的时空状态记忆，使 VideoLLM 能够显式地推理目标物体在长时间不可见期间的隐藏状态变化。

## 研究问题与动机
- **核心问题**：现有 VLMs 在处理长视频时空推理时，一旦目标物体变得不可见（遮挡、 containment、覆盖等），往往将"不可见"错误地等同于"未知"，无法推断其在不可见期间的状态变化与持久性。
- **长视频场景的特殊挑战**：物体可能因遮挡、包容或覆盖长时间不可见，且查询发生在最后一次可见之后很久，仅靠短期时序建模或片段级检索无法解决。
- **现有方法不足**：当前视频理解增强方法（如 Video-RAG、FlexSelect）多为证据驱动，聚合可见观测或压缩视频内容，未显式建模消失原因或状态持久性。
- **缺乏诊断基准**：现有基准（如 MLVU、LongVideoBench 等）对这种能力的评测严重不足。

## 核心贡献（创新点）
- **形式化隐藏状态时空推理问题**：首次系统性地定义并形式化了长视频 QA 中的隐藏状态推理任务，明确将物体不可见期间的状态持久性建模识别为现有 VideoLLM 的核心缺失能力。
- **提出 StateTrace 对象中心框架**：构建了以对象为中心的可复用时空状态记忆（图结构存储对象轨迹、关系与状态转换事件），将长视频推理从纯视觉证据驱动转变为结构化状态中心推理。
- **构建 HSR-Bench 诊断基准**：建立了首个专门针对遮挡/不可见场景下隐藏状态推理的诊断基准，包含 1,427 个视频-QA 样本（1,384 个视频），覆盖 52.35% 重度遮挡与 46.11% 中度遮挡场景。
- **多模型通用增益**：StateTrace 在多个 VideoLLM（InternVL2.5、Qwen2.5-VL、VideoLLaMA3、LLaVA-Video）上均实现稳定提升，在 HSR-Bench 上提升幅度达 15.14–28.31 个百分点。

## 方法详解
StateTrace 包含三个阶段：

**阶段 I：离线时空状态记忆构建**
- 将视频划分为连续 chunks $\{C_m\}_{m=1}^K$，构建 chunk 级有向图 $G_V$ 作为主记忆载体，辅助实体级访问图 $G_E$ 支持高效检索。
- 记忆分解为三个组件：$M = (M_{\text{sem}}, M_{\text{loc}}, M_{\text{evt}})$
  - $M_{\text{sem}}$：chunk 级语义上下文（实体、动作、场景、字幕）
  - $M_{\text{loc}}$：对象中心空间位置记忆，记录绝对几何 $p_i^{m,t}=(x_i^{m,t}, y_i^{m,t}, a_i^{m,t})$ 和成对空间关系 $r_{ij}^{m,t}=(o_i, o_j, \rho_{ij}^{m,t}, c_{ij}^{m,t})$，仅记录变化的关系以减少冗余
  - $M_{\text{evt}}$：对象中心事件记忆 $e^{m,t}=(v^{m,t}, \epsilon^{m,t}, \mu^{m,t})$，记录可见性、文本事件描述与结构化元数据
- **对象中心时空解析**：使用 VLM 提取器获取 chunk 级语义；VLM grounding 获取边界框并用 SAM2 进行 mask 传播，跨 chunk 连续性通过 IoU 匹配维护
- **消失原因推断**：对非边界消失事件，通过 VLM 推断消失原因（inside/occluded/other/unknown），并关联交互物体

**阶段 II：问题驱动的时空证据总结**
- 将问题 $q$ 转化为结构化查询 $\Psi(q) = (Q_q, \ell_q, t_q)$（关键词、控制信号、时间约束）
- 两阶段检索：先通过 $G_E$ 实体查找 + $G_V$ 语义匹配得到候选集，再经语义重排序筛选
- 生成两种互补总结：
  - 语义总结 $g_q = \Gamma(x_q^{\text{agg}}, q)$：压缩检索到的答案相关证据
  - 空间总结 $s_q = S(E_q, A_q, q)$：结构化时空证据 $E_q$ 与锚点事件 $A_q$（消失、遮挡、进入、重现转换）经多模态模型处理后生成

**阶段 III：总结增强的答案生成**
- 最终预测公式：$\hat{a} = M_{\text{ans}}(q, S_q^{\text{sub}}, V_q, g_q, s_q)$
- 文本部分整合问题、字幕上下文与两种总结，视觉部分为对应视频 chunks，模型无关，可适配不同 VideoLLM backbone

## 实验与结果
**数据集与基线**：
- 公共基准：MLVU、VideoMME（w/o sub. / w/ sub.）、LongVideoBench
- 自构基准：HSR-Bench（1,427 video-QA，1,384 videos）
- 基线模型：InternVL2.5（2B/8B）、Qwen2.5-VL（3B/7B）、VideoLLaMA3（2B/7B）、LLaVA-Video（7B）
- 对比方法：Video-RAG、FlexSelect

**主要结果**：
- 公共基准：StateTrace 在所有 backbone 上均稳定提升，LongVideoBench 提升最显著，如 VideoLLaMA3-7B 从 59.8 提升至 64.5
- HSR-Bench：提升幅度最大，区间 15.14–28.31 分；最强结果 VideoLLaMA3-7B 达到 **64.19**（基线 39.59），Qwen2.5-VL-3B 从 29.85 提升至 **57.74**
- 与对比方法相比差距显著：Qwen2.5-VL-7B 上 Video-RAG 38.20、FlexSelect 41.50，StateTrace 达 61.60
- 长度分析：视频越长增益越大，LongVideoBench 最长 bucket 中 VideoLLaMA3-7B 从 58.1 提升至 62.8
- 遮挡严重度分析：重度遮挡场景增益最大，VideoLLaMA3-7B 在 heavy bucket 从 37.28 提升至 **63.34**

**消融实验**：
- 移除消失原因推理（DispC）：VideoLLaMA3-7B LVB 从 64.5 降至 61.2
- 移除空间总结（SpS）：性能下降最大，VideoLLaMA3-7B LVB 从 64.5 降至 60.5
- 移除关键片段检索（KCR）：VideoLLaMA3-7B LVB 从 64.5 降至 62.7

## 相关工作脉络
- **Video-RAG [33]**：基于检索增强的长视频理解，通过视觉对齐检索压缩上下文；StateTrace 区别在于显式构建对象中心的状态记忆而非仅检索片段
- **FlexSelect [62]**：灵活 token 选择以实现高效长视频理解；StateTrace 不依赖 token 级选择，而是通过结构化状态记忆支持跨时段推理
- **VISA [56] / ViLLa [64]**：结合语言引导的推理与 mask 预测/track 级表示；这些方法聚焦于可见对象的推理分割，StateTrace 专注于不可见/遮挡下的状态持久性建模
- **MovieChat [45] / STORM [17]**：基于内存或 token 压缩的长视频理解；StateTrace 的 memory 是结构化对象中心而非 dense token 存储
- **TimeChat [42]**：引入时间戳感知编码；StateTrace 进一步显式建模消失原因与状态转换事件

## 局限性与未来方向
- **计算开销**：离线阶段需进行 VLM grounding、SAM2 传播与消失原因推断，对长视频的处理成本较高
- **模型依赖性**：状态记忆构建质量依赖底层 VLM 与分割模型的性能，在复杂场景下可能出现解析错误
- **事件类型有限**：当前仅覆盖 put-into、covered-by、occluded-by、removed-from 等离散事件，难以处理更细粒度的状态演变
- **未处理永久性消失**：当前框架假设物体仍在场景中仅暂时不可见，边界 exit 被单独处理，未建模物体离开场景后的推理

## 研究启发与可借鉴点
- **结构化状态记忆设计**：将对象轨迹、关系与事件分解为 sem/loc/evt 三个正交组件并存储在图结构中，这一设计可有效迁移至其他需要跨时段对象状态跟踪的任务
- **消失原因显式推断**：通过 VLM 推断不可见原因并将其结构化记录，比仅记录可见/不可见状态能显著提升后续推理准确性，这一思路可用于机器人操作理解等场景
- **摘要式证据压缩**：将低层时空记录转化为高层语义与空间总结，既保留了关键信息又降低了下游模型的输入复杂度，可作为通用接口设计参考
- **HSR-Bench 的构建范式**：基于 OVIS/MOSEv2 进行规则挖掘 + VLM 辅助问题生成 + 人工标注的组合方式，为构建专项推理基准提供了可复用的工程流程

## 关键术语表
- **Hidden-State Spatiotemporal Reasoning**：指在目标物体因遮挡/包容/覆盖等原因不可见的持续时间段内，基于 prior 交互与状态转换推断其潜在状态的能力
- **Spatiotemporal State Memory**：以图结构组织的可复用记忆，包含 chunk 级语义、对象中心空间位置与事件记录三大部分
- **Disappearance-Cause Reasoning**：对非边界消失事件推断其根本原因（放入内部/被遮挡/其他），并关联交互物体的过程
- **Boundary-Exit Filtering**：通过计算对象掩码与图像边界接触比 $\beta_i^{m,t}$ 区分对象真正消失与离开画面的机制
- **Semantic/Spatial Summary**：分别将检索到的文本证据与结构化时空记录压缩为紧凑形式，供 VideoLLM 显式推理使用
- **HSR-Bench**：针对隐藏状态时空推理构建的诊断基准，包含 1,427 个 video-QA 样本，覆盖四种任务类型（遮挡实体识别、遮挡事件总结、遮挡条件属性提取、后遮挡状态持久性）

## 可复现要素
- **数据集**：HSR-Bench 基于 OVIS [39] 与 MOSEv2 [14] 构建，论文未明确声明是否公开；公共基准（MLVU、VideoMME、LongVideoBench）均为公开数据集
- **代码/权重**：论文未提及代码开源状态
- **关键超参**：消失原因推断的上下文窗口大小（preceding/following chunks 数量）、边界过滤阈值 $\tau_b$、IoU 匹配阈值，论文未详细披露
