---
title: "THINKING-BEYOND-VIDEOS-UNIFYING-VIDEO-REASONING-AND-DEEP-RES"
source: https://arxiv.org/pdf/2608.23329v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 23:51:37"
field: "多模态大模型与视频理解"
keywords: ["视频推理", "深度研究", "多模态Agent", "强化学习", "时序定位", "检索增强生成"]
innovations: ["提出VideoRover统一框架实现视频主动感知与开放世界检索的双向迭代协调", "构建自动化数据合成管道生成26K SFT轨迹与3K难例RL实例", "引入GSPO强化学习在长视野轨迹上优化视频-检索协同策略"]
benchmarks: ["VideoDR", "VideoRover-Bench"]
---

# 论文速读：THINKING-BEYOND-VIDEOS-UNIFYING-VIDEO-REASONING-AND-DEEP-RES

## 一句话总结
本文提出 VideoRover，一个统一"视频主动观察"与"开放世界深度研究"的 Agent 框架，通过迭代协调视频裁剪、多模态搜索与网页浏览，实现从视频证据到外部知识的双向引导与验证；基于 Qwen3-VL-8B 的两阶段训练（SFT + GSPO 强化学习）在 VideoDR 与 VideoRover-Bench 上达到与最强闭源模型相当的性能。

## 研究问题与动机
- **核心问题**：开放世界视频理解需要模型同时具备"从长视频中定位稀疏视觉证据"与"获取视频/参数记忆之外的外部知识"两种能力，但现有方法将两者割裂开发。
- **现有方法不足**：
  1. Thinking-with-Videos 系列（如 VITAL、Video-o3）仅聚焦视频内推理，未将局部化视觉锚点连接到外部检索，也未根据检索结果反向修正视频观测。
  2. Deep Research 系列（如 WebThinker、Vision-DeepResearch）主要针对文本/静态图像/网页，缺乏对长视频中稀疏时序证据的显式建模。
  3. 现有训练数据缺乏覆盖"时序定位→多模态检索→证据综合→答案生成"的完整交互轨迹，简单拼凑工具无法实现真正的协同。

## 核心贡献（创新点）
1. **提出 VideoRover 统一框架**：将 Thinking-with-Videos 的时序定位与 Deep Research 的多轮检索/浏览整合进单一 Agent，通过共享研究状态实现双向引导。与既有工作的本质区别在于打破单向流水线，使检索结果可触发视频复查。
2. **自动化数据合成管道**：构建依赖"视频+外部知识"双重要求的问题，并通过 DeepSeek-V4-Flash（规划者）与 Qwen3.5-27B（视频观察员）协作生成带验证的 SFT 轨迹（26K）与难例 RL 实例（3K）。与人工标注或单步合成数据的本质区别在于轨迹覆盖完整多轮协调过程。
3. **提出 VideoRover-Bench 分层基准**：按视频时长（短/长）与研究难度（Easy/Medium/Hard）交叉分层，共 300 例。与既有视频 QA 基准的本质区别在于同时评估时序定位、开放检索与多源推理。
4. **SFT + GSPO 两阶段训练**：先用 SFT 建立可靠工具调用先验，再用 Group Sequence Policy Optimization 在难例上做长视野策略优化。与直接端到端 RL 的本质区别在于避免早期探索阶段的行为崩溃。

## 方法详解
- **任务形式化**：Video Deep Research 实例表示为 $x = (V, Q)$，研究状态 $S_t = (G_t^v, G_t^w, h_t)$ 包含累积视频证据、网页证据与交互历史；每步动作 $u_t \sim \pi_\theta(\cdot|Q, S_t)$，状态通过 Update 函数与工具观测 $z_t$ 更新。
- **四类工具**：
  1. `crop_video(V, [s_t, e_t])`：对选定区间进行密集采样（2 FPS，最多 32 帧），返回高质量片段；
  2. `image_search(v_{k_t})`：以关键帧为锚点进行视觉检索；
  3. `text_search(q_t)`：根据证据缺口生成文本查询并返回候选网页及摘要；
  4. `visit(l)`：打开网页提取详细证据。
- **迭代协调机制**：视频观察指导检索（何时搜、搜什么），检索结果更新研究状态后可触发"继续搜索/重新定位视频/重新检查证据/生成答案"。
- **SFT 损失**：$\mathcal{L}_{\text{SFT}}(\theta) = -\frac{1}{M}\sum_{i=1}^{M} \log \pi_\theta(\tau_i^* | x_i)$，工具输出作为环境上下文不参与 loss。
- **GSPO 强化学习**：每例采样 $G=8$ 条轨迹，正确/错误答案分别得奖励 1/0，格式化错误/无效工具调用被过滤；优势估计采用 leave-one-out baseline：$A_i = R_i - \frac{1}{G-1}\sum_{j\neq i}R_j$；策略梯度采用截断 importance ratio，并对轨迹内 token 先平均再跨轨迹平均，防止长轨迹主导优化。

## 实验与结果
- **数据集与基准**：VideoDR（公开 benchmark）与本文提出的 VideoRover-Bench（300 例，6 个长度-难度子集各 50 例）。
- **评估设置**：对比闭源模型直接回答 vs. 开源模型在相同工具套件下的 ReAct 式 Agentic 工具使用；初始视频输入上限 64 帧，每个 agentic 模型最多 50 次工具调用。
- **主要结果**（VideoRover-8B-RL vs. 其他开源 agentic）：
  - VideoDR：VideoRover-8B-RL = 56.00%，超越 Qwen3.5-27B（54.00%）、Qwen3.6-27B（54.00%）等更大模型；
  - VideoRover-Bench Avg.：VideoRover-8B-RL = 57.71%，最强子集 S-Easy 达 80.00%；
  - 与闭源模型相比：VideoRover-8B-RL（56.00% on VideoDR）接近 Gemini-3-Pro（55.00%）与 Gemini-3-Flash（59.00%）。
- **消融**：移除 `text_search` 导致最大降幅（VideoDR 56→24，Avg. 57.71→23.71），说明外部知识补全至关重要；移除全部 web 工具后性能暴跌至 13.00%，验证四工具互补性。
- **RL 动力学**：平滑奖励与交互轮数在 SFT cold start 后同步上升，表明更长轨迹用于持续证据获取而非冗余调用。

## 相关工作脉络
1. **Thinking-with-Images/ Videos**（OpenThinkIMG、DeepEyes、Thyme、VITAL、Video-o3、VideoTemp-o3）：本文定位差异——这些方法仅做视频内主动感知，VideoRover 将其扩展至"感知→检索→验证"闭环。
2. **Deep Research / Search Agents**（WebGPT、Search-R1、Vision-DeepResearch、MM-DeepResearch、WebThinker）：本文定位差异——既有工作以文本/静态图像为主，本文首次将长视频时序定位纳入深度研究轨迹。
3. **RAG / Multimodal RAG**（Murag、综述类工作）：本文定位差异——传统 RAG 是单向检索，VideoRover 允许检索结果反向引导视频复查，形成双向迭代。
4. **Long Video Understanding**（LongVA、LongVILA、LongVT、DynFrame）：本文定位差异——扩展上下文或自适应重采样仍属"被动观察"，本文引入工具驱动主动采样与外部知识协同。
5. **RL for Agents**（MMSearchR1、HyperEyes、Openthinkimg、Deepeyes）：本文定位差异——本文的 RL 发生在"视频-检索"跨模态联合轨迹上，奖励信号来自最终答案正确性，强调长视野协调而非单步搜索效率。

## 局限性与未来方向
- **局限**：
  1. 工具调用上限 50 次，可能在极复杂多跳任务中仍受预算约束；
  2. 数据合成依赖闭源大模型（GPT-5.4-mini、Qwen3.5-27B、Gemini-2.5-Pro）进行质量过滤与验证，存在管线成本与潜在偏差；
  3. 当前仅支持 YouTube 视频作为视觉锚点来源，未覆盖点播平台、监控录像等封闭场景；
  4. RL 阶段仅用最终答案正确性作为奖励，缺乏对中间步骤（如正确定位、有效检索）的细粒度奖励信号。
- **未来方向**：引入过程奖励模型（PRM）或检索质量奖励；扩展至更多视频源与多语言场景；探索更高效的长轨迹 RL 算法以突破 50 次工具调用上限。

## 研究启发与可借鉴点
1. **双向引导架构设计**："视觉锚点→检索→检索结果→视频复查"的闭环思路可迁移至其他"感知+知识获取"联合任务（如文档理解、科学文献挖掘）。
2. **自动化难例筛选策略**：先 SFT 冷启动，再对"需要 >10 次工具调用且在 3 次 rollout 中仍答错"的样本进行 RL，这一难例选择逻辑可有效控制 RL 阶段的训练稳定性。
3. **留一法轨迹级优势估计**：在 RL 阶段使用 $\frac{1}{G-1}\sum_{j\neq i}R_j$ 作为基线，而非全局均值，可更敏感地捕捉单条轨迹相对组内其他轨迹的差异，值得在多轨迹 RL 场景中复用。
4. **双角色协作合成**：用纯文本模型担任"规划者"、用多模态模型担任"视频观察员"，既发挥文本模型推理优势又保证视觉 grounding 一致性，可推广至其他多智能体数据合成管线。

## 关键术语表
- **Video Deep Research**：面向开放世界视频的理解任务，要求模型结合视频内时序定位与视频外多轮检索/浏览获取完整答案。
- **Thinking-with-Videos**：将视频视为主动推理工作区，通过裁剪、关键帧选择、自适应重采样等工具驱动的视频观测方法。
- **Group Sequence Policy Optimization (GSPO)**：一种序列级策略优化算法，对多条轨迹的优势进行 leave-one-out 估计并在 token 级别进行重要性采样裁剪。
- **Visual Anchor**：从视频中定位出的可被外部检索引用的关键帧或时间段，作为连接视频观察与网页搜索的桥梁。
- **Research State**：当前已累积的视频证据、网页证据与交互历史的联合表示，用于决定下一步工具调用。
- **Leave-one-out Baseline**：在 RL 中用同组其他轨迹的平均奖励作为当前轨迹优势估计的基线，避免单一样本波动。
- **SFT Cold Start**：在强化学习之前先用监督微调初始化策略，以降低早期探索阶段的行为方差。

## 可复现要素
- **数据集**：VideoDR（已有公开基准）；VideoRover-Bench（论文构建了 300 例，是否开源需查阅代码仓库声明）；训练数据 26K SFT + 3K RL 实例（作者未明确声明公开）。
- **代码/权重**：项目主页 https://liuwq-bit.github.io/VideoRover，论文未明确声明代码是否开源，需进一步确认；权重为 Qwen3-VL-8B 基础上微调。
- **关键超参**：SFT learning rate $2\times10^{-5}$，batch size 256；RL learning rate $1\times10^{-6}$，batch size 16；RL rollout 数 $G=8$，每 rollout 最多 25 次工具调用；agentic 评估最多 50 次工具调用；初始视频采样 1 FPS、最多 256 帧、每帧 224×224；crop video 采样 2 FPS、最多 32 帧、原始分辨率；硬件 4 服务器 × 8 NVIDIA H800 GPU × 2 TB 内存。
- **框架**：SFT 使用 ms-swift，RL 使用 VeRL，推理采样使用 vLLM。
