---
title: "VOS-Agent-The-1st-Place-Solution-for-the-8th-LSVOS-Challenge"
source: https://arxiv.org/pdf/2608.12721v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 04:47:22"
field: "视频目标分割"
keywords: ["Video Object Segmentation", "Multi-Agent Framework", "SAM3", "MOSEv2", "Semi-Supervised VOS", "Visual Tracking"]
innovations: ["提出VOS-Agent协作式多智能体框架，根据目标特性路由到专门处理路径", "设计置信度感知的选择性干预机制，视觉跟踪智能体和语义智能体仅在必要时校正SAM3", "免训练的模块化解决方案，结合SU-Track和Qwen3.5增强SAM3的微小目标和语义主导目标处理"]
benchmarks: ["MOSEv2", "LSVOS Challenge"]
---

# 论文速读：VOS-Agent-The-1st-Place-Solution-for-the-8th-LSVOS-Challenge

## 一句话总结
VOS-Agent 是一个协作式多智能体框架，通过目标感知与路由机制将不同类型的目标（普通、微小、语义主导）分发到专门的处理路径，结合 SAM3 作为共享密集分割模块，在 MOSEv2 测试集上达到 69.82% 的 $\mathcal{J}\&\mathcal{F}^*$ 指标，获得第 8 届 LSVOS 挑战赛 MOSEv2 赛道第一名。

## 研究问题与动机
1. **复杂视频环境下的鲁棒分割需求**：MOSEv2 挑战包含严重遮挡、目标消失与重现、微小目标、相似干扰物等真实场景，传统 VOS 方法在突出且孤立目标上表现良好，但在非受限环境中泛化能力不足。
2. **微小目标的视觉证据不足**：微小目标仅占数个像素，难以提供稳定的长距离对应关系，易受背景杂乱、遮挡和累积定位误差的影响。
3. **语义主导目标的身份保持困难**：某些目标虽空间范围充足，但多个邻近实例共享相似的底层外观，其身份依赖于文本、logo、符号、颜色或独特服饰等显式语义属性。
4. **统一推理路径的局限性**：SAM3 等基础模型主要通过 prompt-conditioned 视觉特征和时间记忆传播目标，当目标表征严重欠解析或视觉上模糊时，追踪器缺乏足够证据进行定位和实例保持。

## 核心贡献（创新点）
1. **提出 VOS-Agent 协作式多智能体框架**：与现有工作相比，本质区别在于将任务分解为路由决策、密集分割和专项推理三个协同模块，而非设计单一端到端网络。
2. **设计目标感知与路由智能体**：基于归一化面积比 $r_{\mathrm{area}}$ 和语义独特性判断 $s_{\mathrm{sem}}$ 将目标分类，与以往"一刀切"的处理方式相比，能针对不同目标特性选择最优处理路径。
3. **构建视觉跟踪智能体（基于 SU-Track）**：通过置信度感知的边界框提示与 SAM3 协作，与纯视觉特征传播方法相比，在微小目标场景下提供额外的位置校正信号。
4. **开发基于 MLLM 的语义智能体**：利用 Qwen3.5 生成判别性描述并进行语言引导定位与候选验证，与仅依赖底层视觉对应的方法相比，能有效区分外观相似但语义不同的实例。
5. **免训练、模块化的挑战解决方案**：不需微调 SAM3 backbone，通过组合现有强模型实现 SOTA，相比需要大量训练的方案，具有更高的灵活性和可迁移性。

## 方法详解

### 整体架构
VOS-Agent 由四个核心组件构成（Fig. 2）：
- **Target Perception and Routing Agent**：分析目标的空间尺度和语义独特性，将其路由至 regular、tiny 或 semantic-dominated 三类处理路径。
- **SAM3 Segmentation Agent**：共享的密集分割与时序传播模块，负责将所有 prompt 转化为像素级 mask。
- **Visual Tracking Agent**：基于 SU-Track，为微小目标提供置信度感知的边界框校正。
- **Semantic Agent**：基于 Qwen3.5，为语义主导目标提供语言引导的定位和身份验证。

### 目标感知与路由智能体
**空间尺度分析**（公式 1）：
$$r_{\mathrm{area}} = \frac{A(M_1)}{A(I_1)}$$
当 $r_{\mathrm{area}} < \tau_{\mathrm{area}}$ 时判定为微小目标。

**语义独特性分析**（公式 2）：
$$s_{\mathrm{sem}} = \mathcal{A}_{\mathrm{sem-clns}}(I_1, M_1), \quad s_{\mathrm{sem}} \in \{0, 1\}$$
由 MLLM 判断目标是否依赖显式语义属性（如文本、logo、符号、独特颜色等）区别于相似实例。

**路由决策**（公式 3）：
$$z = \begin{cases} \text{tiny}, & r_{\mathrm{area}} < \tau_{\mathrm{area}} \\ \text{semantic}, & r_{\mathrm{area}} \geq \tau_{\mathrm{area}} \text{ 且 } s_{\mathrm{sem}} = 1 \\ \text{regular}, & \text{otherwise} \end{cases}$$

### 视觉跟踪智能体与 SAM3 协作
**初始化**（公式 4）：
$$B_1^{\mathrm{trk}} = \bar{B}(M_1), \quad Z^{\mathrm{sta}} = \mathrm{Crop}(I_1, B_1^{\mathrm{trk}})$$
其中 $Z^{\mathrm{sta}}$ 为全程保留的静态目标模板。

**搜索区域提取**（公式 5）：
$$X_t = \mathrm{Crop}(I_t, \mathrm{Expand}(B_{t-1}^{\mathrm{trk}}))$$

**跟踪估计**（公式 6）：
$$(B_t^{\mathrm{trk}}, c_t^{\mathrm{trk}}) = \mathcal{A}_{\mathrm{trk}}(Z^{\mathrm{sta}}, Z_{t-1}^{\mathrm{dyn}}, X_t)$$

**一致性度量**（公式 7）：
$$q_t^{\mathrm{trk}} = \mathrm{IoU}(B_t^{\mathrm{sam}}, B_t^{\mathrm{trk}})$$

**干预条件**（公式 8）：
$$\eta_t^{\mathrm{trk}} = \mathbb{I}[q_t^{\mathrm{trk}} < \tau_{\mathrm{iou}} \wedge c_t^{\mathrm{trk}} \geq \tau_{\mathrm{conf}}]$$
仅当两者不一致且跟踪置信度足够高时，才将 tracking box 作为额外 prompt 注入 SAM3。

### 语义智能体与 SAM3 协作
**描述生成**（公式 9）：
$$D = \mathcal{A}_{\mathrm{sem}}^{\mathrm{desc}}(I_1, M_1)$$
生成区分目标与同类干扰物的判别性描述。

**语言引导定位**（公式 10）：
$$B_t^{\mathrm{sem}} = \mathcal{A}_{\mathrm{sem}}^{\mathrm{loc}}(I_t, D)$$

**一致性度量**（公式 11）：
$$q_t^{\mathrm{sem}} = \mathrm{IoU}(B_t^{\mathrm{sam}}, B_t^{\mathrm{sem}})$$

**候选验证**（公式 12-13）：
当 $q_t^{\mathrm{sem}} < \tau_{\mathrm{sem}}$ 时，提取参考目标和两个候选区域进行语义比较：
$$y_t^{\mathrm{sem}} = \mathcal{A}_{\mathrm{sem}}^{\mathrm{judge}}(R_1^{\mathrm{ref}}, D, R_t^{\mathrm{sam}}, R_t^{\mathrm{sem}}), \quad y_t^{\mathrm{sem}} \in \{\mathrm{sam}, \mathrm{semantic}\}$$

**干预条件**（公式 14）：
$$\eta_t^{\mathrm{sem}} = \mathbb{I}[q_t^{\mathrm{sem}} < \tau_{\mathrm{sem}} \wedge y_t^{\mathrm{sem}} = \mathrm{semantic}]$$
仅当语义定位与 SAM3 冲突且语义候选被判定为更优时，才采纳语义 correction。

### 关键设计原则
- **选择性干预**：两个专项智能体仅在置信度高且与 SAM3 不一致时才介入，避免干扰 SAM3 的正常工作。
- **状态互补**：SU-Track 递归估计位置，SAM3 通过视频记忆传播 mask，两者维护互补的时序状态。
- **免训练**：所有模块均基于预训练模型，无需对 SAM3 backbone 进行微调。

## 实验与结果

### 数据集与挑战协议
- **MOSEv2 赛道**（第 8 届 LSVOS Challenge @ ECCV 2026）
- 数据规模：5,024 个视频，10,074 个标注目标（200 类），超过 701,000 个高质量 mask
- 挑战性场景：频繁消失与重现、微小/隐蔽目标、严重遮挡、拥挤场景、恶劣天气、低光照、伪装物体、需要外部知识的情况

### 主要结果（Table 1）

| 参与者 | $\mathcal{J}\&\dot{\mathcal{F}}$ | $\mathcal{J}$ | $\dot{\mathcal{F}}$ | $\mathcal{J}\&\dot{\mathcal{F}}_d$ | $\mathcal{J}\&\dot{\mathcal{F}}_r$ |
|--------|-------------------------------|-------------|-------------------|----------------------------------|----------------------------------|
| **HITsz-Dragon (VOS-Agent)** | **69.82** | 68.21 | 71.43 | 79.12 | 34.87 |
| mmm | 66.20 | 64.79 | 67.60 | 81.57 | 26.88 |
| kjeong | 64.37 | 63.16 | 65.59 | 80.26 | 27.53 |

VOS-Agent 以 **69.82%** 的 $\mathcal{J}\&\dot{\mathcal{F}}$ 获得第 1 名，较第二名 mmm 提升 **3.62** 个百分点。

### 消融实验（Table 2）

| 方法 | $\mathcal{J}\&\dot{\mathcal{F}}$ | $\mathcal{J}$ | $\dot{\mathcal{F}}$ | $\mathcal{J}\&\dot{\mathcal{F}}_d$ | $\mathcal{J}\&\dot{\mathcal{F}}_r$ |
|------|-------------------------------|-------------|-------------------|----------------------------------|----------------------------------|
| SAM3 Agent only | 62.49 | 61.38 | 63.59 | 78.93 | 25.31 |
| + Routing + Semantic Agent | 67.07 | 65.68 | 68.47 | 79.12 | 31.24 |
| **Full VOS-Agent** | **69.82** | **68.21** | **71.43** | **79.12** | **34.87** |

- 仅 SAM3：62.49%
- 加入路由 + 语义智能体：67.07%（+4.58）
- 完整 VOS-Agent：69.82%（+7.33 vs. SAM3 only）
- 视觉跟踪智能体对消失/重现场景提升显著（+3.63 on $\mathcal{J}\&\dot{\mathcal{F}}_r$）

## 相关工作脉络

1. **SAM3**（ICLR 2026）：提供 prompt-conditioned 的视频 mask 传播能力，是本文的共享密集分割基础。本文定位：将其作为通用模块，通过外部智能体增强而非修改其内部机制。
2. **XMem**（ECCV 2022）：通过互补记忆存储实现高效长时序传播。本文对比：XMem 关注记忆管理优化，本文通过路由机制针对不同目标类型采用不同策略。
3. **Cutie**（CVPR 2024）：引入 object-level memory reading 以减少 distractor-rich 场景中的匹配噪声。本文对比：Cutie 改进内部记忆机制，本文通过外部语义智能体解决身份歧义问题。
4. **SAM2**（ICLR 2025）：prompt-conditioned 图像和视频分割的通用框架。本文扩展：SAM3 进一步集成 concept-conditioned 检测，本文在此基础上构建多智能体协作。
5. **SU-Track**（AAAI 2025）：简单统一的单目标跟踪器，结合静态和动态模板。本文应用：作为微小目标的视觉跟踪后端，提供置信度感知的边界框校正。
6. **Qwen3.5**：大型多模态语言模型，支持目标描述生成、语言引导定位和候选验证。本文创新：将其用于语义主导目标的身份推理，而非传统的视觉-only 方案。

## 局限性与未来方向

1. **计算开销较高**：依赖多个独立组件（Qwen3.5-397B-A17B、SAM3、SU-Track），推理延迟和显存占用较大，不适合实时应用。
2. **阈值敏感**：路由阈值 $\tau_{\mathrm{area}}$、IoU 阈值 $\tau_{\mathrm{iou}}$ 和语义阈值 $\tau_{\mathrm{sem}}$ 需在验证集上调优，缺乏自适应机制。
3. **对极微小目标的局限**：当目标仅占数个像素且外观变化剧烈时，即使 SU-Track 也难以提供可靠定位。
4. **语义描述的生成质量依赖 MLLM 能力**：对于抽象或隐含属性的目标，生成的判别性描述可能不够准确。
5. **未来方向**：可探索端到端微调 SAM3  backbone 以提升整体性能；设计更轻量的路由和推理模块；研究动态阈值学习机制。

## 研究启发与可借鉴点

1. **路由机制设计**：将目标感知与处理路径解耦的思路可直接迁移到其他视觉任务（如视频目标检测、多目标跟踪），通过分类器选择最优处理策略。
2. **选择性干预策略**：置信度感知的协作机制（仅在专项模块置信度高且与主模块不一致时干预）是一种优雅的集成方式，避免了强行融合带来的性能下降。
3. **MLLM 辅助视觉任务**：利用语言模型生成判别性描述并进行 identity reasoning，为视觉分割中的实例歧义问题提供了新思路。
4. **免训练挑战解决方案**：组合现有 SOTA 模型而非从头训练，展示了 modular pipeline 在挑战赛中的竞争力，可作为快速原型开发的参考范式。
5. **消失/重现场景评估**：论文单独报告了 $\mathcal{J}\&\mathcal{F}_d$ 和 $\mathcal{J}\&\mathcal{F}_r$ 指标，这种细粒度评估方法值得在其他 VOS 研究中采用。

## 关键术语表

**Semi-supervised VOS**：半监督视频目标分割，给定首帧目标 mask，需在整个视频中保持一致地分割同一目标。
**MOSEv2**：More challenging Object Segmentation in Video 的升级版，包含拥挤场景、严重遮挡、微小目标等挑战。
**Promptable segmentation**：可通过点、框、mask 等 prompt 引导的分割方式，SAM3 的核心能力。
**Masklet**：目标 mask 的轻量化表示，包含 conditioning memory 用于时序传播。
**Confidence-aware**：置信度感知的，指仅在预测置信度高于阈值时才采纳干预信号。
**Semantic-dominated target**：语义主导目标，其身份主要依赖显式语义属性（如文本、logo）而非底层视觉特征。
**Conditioning memory**：条件记忆，存储初始帧信息用于后续帧的 mask 传播。
**Instance switching**：实例切换，追踪器错误地从目标切换到外观相似的干扰实例。

## 可复现要素

- **数据集**：MOSEv2（挑战赛数据集，需注册获取）
- **代码开源状态**：论文未提及代码开源
- **权重**：SAM3（官方模型）、SU-Track（官方模型）、Qwen3.5-397B-A17B（官方模型）
- **关键超参**：$\tau_{\mathrm{area}}$、$\tau_{\mathrm{iou}}$、$\tau_{\mathrm{conf}}$、$\tau_{\mathrm{sem}}$（论文未给出具体数值，仅在验证集上选定）
