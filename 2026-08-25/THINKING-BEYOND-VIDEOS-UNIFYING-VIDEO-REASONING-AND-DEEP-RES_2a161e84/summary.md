---
title: "THINKING-BEYOND-VIDEOS-UNIFYING-VIDEO-REASONING-AND-DEEP-RES"
source: https://arxiv.org/pdf/2608.23329v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 23:52:01"
field: "多模态视频理解与Agent"
keywords: ["Video Deep Research", "Thinking-with-Videos", "Multimodal Agent", "Reinforcement Learning", "Video Reasoning", "Tool-Augmented MLLM"]
innovations: ["提出VideoRover统一框架，实现视频观察与外部检索的双向迭代协调", "构建自动化数据合成管线，生成26K SFT轨迹与3K RL困难实例", "设计GSPO强化学习策略，通过轨迹级优势估计学习长 horizon 工具组合决策"]
benchmarks: ["VideoDR", "VideoRover-Bench"]
---

# 论文速读：THINKING-BEYOND-VIDEOS-UNIFYING-VIDEO-REASONING-AND-DEEP-RESEARCH

## 一句话总结
本文提出了 VideoRover，一个统一的 Video Deep Research 框架，通过迭代协调视频裁剪、多模态搜索和网页浏览，使视频证据能够指导外部检索，而检索结果又能反向触发对视频的再验证与再观察。基于 Qwen3-VL-8B 训练的 VideoRover-8B-RL 在 VideoDR 上达到 56% 准确率，性能接近同类直接回答的闭源模型，且优于使用相同工具集的同规模开源模型。

## 研究问题与动机
1. **开放世界视频理解需要双重能力**：既要精确定位视频中稀疏的视觉证据（Temporal Grounding），又需获取视频中不存在的背景/历史信息（External Retrieval），现有 MLLM 仅靠固定输入和参数化知识无法同时满足两者。
2. **Thinking-with-Videos 与 Deep Research 各自独立发展**：前者专注于视频内局部时间定位和自适应重采样，后者专注于文本/图片检索和多轮网页浏览；两者缺乏共享状态和双向联动机制，单向流水线会放大早期感知错误。
3. **缺乏覆盖完整交互轨迹的训练数据**：现有数据极少覆盖从时间定位、多模态检索到证据修正和答案合成的完整链路，导致简单地将搜索工具附加到视频推理器上无法实现有效协调。
4. **跨模态决策与状态更新机制尚不明朗**：如何在证据缺口驱动下选择 crop video / image search / text search / visit 等工具，并保持视频证据与网页证据的统一状态表示，是一个未充分探索的问题。

## 核心贡献（创新点）
1. **提出 VideoRover 统一框架，首次实现"视频观察→外部检索→再观察"的双向迭代循环**，与既有 Thinking-with-Videos（仅内部推理）和 Deep Research（仅外部检索）的本质区别在于：每一轮工具结果更新共享研究状态，并反过来指导下一轮动作。
2. **构建自动化数据合成管线，生成 26K SFT 轨迹 + 3K RL 困难实例**：通过问答双依赖约束（问题答案必须同时依赖视频证据和外部知识）+ 轨迹过滤（仅保留最终答案正确的轨迹），确保了数据质量。
3. **提出 VideoRover-Bench 分层评测基准**：按视频时长（短/长）和研究难度（Easy/Medium/Hard）交叉划分，共 300 个验证样本，为视频 Deep Research 提供细粒度评估体系。
4. **设计 GSPO 强化学习训练策略**：基于轨迹级优势估计和序列重要性比率裁剪，使 SFT 初始化模型在真实工具反馈下学会何时继续检索、何时回看视频、何时作答。

## 方法详解
**工具集（四种）**：
- `crop_video(V, [s_t, e_t])`：对选定时间段进行密集采样，返回高分辨率片段（2 FPS，最多 32 帧原始分辨率）。
- `image_search(v_{k_t})`：以关键帧为锚点检索视觉相关信息。
- `text_search(q_t)`：以文本查询检索候选网页和摘要。
- `visit(l)`：打开网页提取详细证据。
- `answer`：终止动作。

**状态表示**：$S_t = (G_t^v, G_t^w, h_t)$，其中 $G_t^v$ 为累积视频证据，$G_t^w$ 为累积网页证据，$h_t$ 为交互历史。

**关键设计**：
- **双向引导机制**：视频观察结果决定检索类型和查询措辞；检索结果决定是否需要重新定位/复查视频片段。
- **角色分工的轨迹生成**：DeepSeek-V4-Flash 作为研究规划器，Qwen3.5-27B 作为视频观察者；观察者首先对稀疏采样帧进行分析并提出初始相关时段，规划器必须从 crop_video 开始。
- **SFT 训练目标**：$\mathcal{L}_{\mathrm{SFT}}(\theta) = -\frac{1}{M}\sum_{i=1}^{M}\log\pi_\theta(\tau_i^*|x_i)$，工具输出视为环境上下文，不参与损失计算。
- **GSPO 训练目标**（公式4）：以轨迹为单位做重要性比率裁剪，先对轨迹内 token 取平均再跨轨迹平均，优势采用 leave-one-out 估计：$A_i = R_i - \frac{1}{G-1}\sum_{j\neq i}R_j$，其中 $G=8$。正确/错误答案分别获奖励 1/0，格式错误或无效工具调用被过滤。

## 实验与结果
**数据集**：VideoDR（公开基准）、VideoRover-Bench（本文新建，300 个样本）。

**基线模型**：闭源模型直接回答（GPT-5/5.2/5.4、Gemini-2.5-Flash/Pro、Gemini-3-Flash/Pro）；开源模型直接回答与 ReAct 工具调用（Qwen3-VL 系列、Qwen3.5/3.6 系列）。

**主要结果**：
- VideoRover-8B-RL 在 VideoDR 上获得 **56%** 准确率，超越所有开源模型工具调用变体，并与 Gemini-3-Pro（55%）、Gemini-3-Flash（58.42%）等闭源模型持平。
- VideoRover-8B-RL 在 VideoRover-Bench 上平均准确率 **57.71%**，在 S-Easy（80%）、S-Medium（64%）、L-Easy（70%）等子集上均领先。
- 从 SFT 到 RL 显著提升：VideoDR 上从 39% → 56%，平均从 39.28% → 57.71%。
- ablation 显示 **text_search 移除影响最大**（平均准确率 57.71% → 23.71%），其次为 image_search（→ 49.00%）和 crop_video（→ 49.57%），移除全部 web 工具后降至 13.85%。
- RL 训练动态显示交互轮次和奖励同步上升，表明模型学会了更长的有效证据采集而非冗余调用。

## 相关工作脉络
1. **Thinking-with-Videos**（VITAL、LongVT、FrameThinker、Video-o3、VideoTemp-o3）：聚焦视频内时间定位与自适应重采样，但推理局限于视频内部；VideoRover 将定位结果作为外部检索的锚点。
2. **Deep Research 系统**（Search-R1、MMSearchR1、Vision-DeepResearch、MM-DeepResearch、HyperEyes）：处理文本/图片的多轮检索，输入为静态图像或网页，缺乏对长视频稀疏证据的处理机制。
3. **Thinking-with-Images**（OpenThinkIMG、DeepEyes、Thyme）：在静态图像上进行迭代视觉检验；VideoRover 扩展至时序维度的密集采样与跨时段复查。
4. **RAG 类方法**（MURAG、Search-RAG 综述）：提供外部知识，但无主动视频观测与动态证据状态更新机制。
5. **RL for agents**（Webthinker、WebGPT）：以文本/网页为交互环境；VideoRover 在视频+网页混合环境中进行 Group Sequence Policy Optimization。
6. **长视频理解**（LongVA、LongVILA）：通过扩展上下文窗口和 token 压缩提升容量，但观察模式在推理前固定，不具备自适应工具调用能力。

## 局限性与未来方向
1. **仅在一个基础模型（Qwen3-VL-8B）上验证**，框架是否可迁移至更大参数规模或其他架构尚待检验。
2. **工具集固定为四种**，未探讨更丰富的动作空间（如视频字幕分析、跨视频检索等）。
3. **数据规模有限**：26K SFT + 3K RL 相比大规模 agent 训练数据偏少，可能限制复杂场景泛化。
4. **RL 训练成本较高**：每个样本 8 条 rollout、每次最多 25 次工具调用，对计算资源要求较大。
5. **未来方向**：可扩展至更多工具类型（视频内跨段检索、多人物关联推理）、更大规模模型、以及真实用户提问场景下的在线评估。

## 研究启发与可借鉴点
1. **双依赖问题构建策略**：通过"问题间接提及目标 + 答案必须结合视频定位和外部知识"的设计，确保数据真正需要联合推理，这一数据筛选思路可迁移至其他多源知识融合任务。
2. **角色分工的数据生成范式**：规划器（DeepSeek-V4-Flash）+ 观察者（Qwen3.5-27B）的分工设计，保证了轨迹中视频观察的真实性和检索规划的合理性，可借鉴于其他多模态 agent 的轨迹合成。
3. **GSPO 在长 horizon 任务中的应用**：轨迹级 leave-one-out 优势估计 + 序列重要性裁剪，有效防止长轨迹主导优化，适合需要多步决策的视觉-语言 agent 训练。
4. **分层基准设计**：VideoRover-Bench 按"视频时长 × 研究难度"交叉分层，为细粒度分析模型在不同困难场景下的表现提供了可复用范式。
5. **双向迭代协调的设计哲学**："视频证据指导检索 → 检索结果触发再观察"的闭环思路，可推广至其他需要内外知识协同的场景（如科学文献解读、医学影像诊断）。

## 关键术语表
**Video Deep Research**：结合视频内部证据定位与外部网络知识检索的开放世界推理范式，要求模型在视频与网页间多次往返。
**Thinking-with-Videos**：允许 MLLM 在推理过程中主动请求视频裁剪、关键帧选择和自适应重采样的工具增强方法。
**Crop Video Tool**：对视频选定时间段进行密集采样的工具，用于将稀疏证据转化为可用于外部检索的视觉锚点。
**GSPO（Group Sequence Policy Optimization）**：一种基于轨迹级别的强化学习算法，通过 leave-one-out 优势估计和序列重要性比率裁剪优化多步决策策略。
**Leave-one-out Advantage**：在 GSPO 中，以同组其他轨迹的平均奖励作为基准，估计当前轨迹的相对优势。
**Research State**：$(G_t^v, G_t^w, h_t)$ 三元组，分别表示累积视频证据、累积网页证据和交互历史，是驱动工具选择的共享状态。
**SFT Cold Start**：先用监督微调建立稳定的工具使用先验，再进行强化学习，避免直接从随机策略开始 RL 的不稳定性。
**VideoRover-Bench**：本文构建的 Video Deep Research 评测基准，按视频时长（短/长）和研究难度（Easy/Medium/Hard）分层，共 300 个样本。

## 可复现要素
- **数据集**：VideoDR（引用已有基准）；VideoRover-Bench（本文构建，项目页面 https://liuwq-bit.github.io/VideoRover 可能有更多信息）。
- **代码/权重**：论文未明确声明开源，项目主页链接已提供。
- **关键超参**：SFT 学习率 $2\times10^{-5}$、batch size 256；RL 学习率 $1\times10^{-6}$、batch size 16；GSPO 每样本 G=8 条 rollout，每 rollout 最多 25 次工具调用；初始视频采样 1 FPS 最多 256 帧（分辨率 224×224）；crop_video 采样 2 FPS 最多 32 帧。
- **训练硬件**：4 台服务器，每台 8× NVIDIA H800 GPU，2 TB 内存。
- **框架**：SFT 使用 ms-swift，RL 使用 VeRL。
