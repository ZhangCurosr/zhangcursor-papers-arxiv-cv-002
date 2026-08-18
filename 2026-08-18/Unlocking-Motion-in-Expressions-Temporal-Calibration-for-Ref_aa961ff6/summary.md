---
title: "Unlocking-Motion-in-Expressions-Temporal-Calibration-for-Ref"
source: https://arxiv.org/pdf/2608.16332v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 06:22:06"
field: "视频分割与多模态理解"
keywords: ["Referring Video Object Segmentation", "Motion Calibration", "Multimodal Temporal Modeling", "Expression-Driven Segmentation"]
innovations: ["首次将表达运动语义显式建模为低维可解释控制信号指导运动校准", "提出表达条件化门控决策机制区分静态/动态表达的时序路径", "通过文本引导谱聚类构建语义时间阶段提供紧凑候选空间"]
benchmarks: ["Ref-YouTube-VOS", "Ref-DAVIS17", "MeViS", "A2D-Sentences", "JHMDB-Sentences"]
---

# 论文速读：Unlocking-Motion-in-Expressions-Temporal-Calibration-for-Ref

## 一句话总结
本文提出EMC（Expression-driven Motion Calibration）框架，通过显式建模自然语言表达中的运动语义重要性，自适应校准运动线索在时序决策中的贡献，解决现有RVOS方法统一处理运动信息而缺乏语义驱动适应性调整的问题。在Ref-YouTube-VOS、MeViS等六个基准上均达到SOTA性能。

## 研究问题与动机
- **核心问题**：现有RVOS方法将运动作为视频的固有属性统一编码进跨模态时序表示，缺乏根据表达语义动态调整运动参与度的能力，导致对含运动描述的表达适应不足、对静态表达引入冗余运动干扰。
- **现有方法局限1**：多数工作采用统一的时序建模框架，语言线索仅用于目标定位和分割，未显式区分表达中运动语义的重要性。
- **现有方法局限2**：虽有几篇工作尝试解耦静态外观与运动线索（如DMVS、DsHmp），但仍以均匀方式使用运动提示，未建立语言表达→运动需求的显式映射机制。
- **时序建模挑战**：逐帧调整运动影响计算冗余且易受噪声帧干扰，需要构建紧凑的语义时间候选空间。

## 核心贡献（创新点）
- **EMC框架**：首次提出表达驱动的运动校准范式，将表达中的运动重要性显式建模为低维可解释控制信号，而非隐式编码进高维语言表示中。
- **MSP模块**：利用LLM纯语言侧解析，同时判定表达是否包含运动语义（G∈{0,1}）并量化其强度（M∈[0,1]），建立语言理解与时序决策的清晰接口。
- **MIC模块**：基于MSP输出的运动信号，将运动线索作为校准因子而非主导信号，通过门控机制区分静态/动态表达的不同决策路径，抑制静态表达中的冗余运动干扰。
- **STSC模块**：通过文本引导的一致性相似度、时序邻近相似度和视觉相似度联合谱聚类，将视频划分为语义时间阶段，为运动校准提供紧凑且表达相关的候选时间结构。

## 方法详解
**整体流程**：输入视频V={I_t}和表达E → STSC构建语义时间阶段并抽取关键帧池K → MSP提取运动控制信号{G,M} → MIC基于信号校准关键帧得分 → 双向传播生成完整分割掩码。

**STSC（语义时间阶段构建）**：
- 文本引导一致性相似度：$S_{ij}^{text} = \exp(-|s_i - s_j|)$，s_i为帧i与表达的语义相关性分数
- 时序邻近相似度：$S_{ij}^{time} = e^{-\lambda|t_i - t_j|}$，λ=0.5控制衰减
- 视觉相似度：余弦相似度$S_{ij}^{vis}$
- 联合相似度矩阵：$S_{ij} = \frac{w_t\widetilde{S}_{ij}^{text} + w_\tau\widetilde{S}_{ij}^{time} + w_v\widetilde{S}_{ij}^{vis}}{w_t + w_\tau + w_v}$，三权值均为1
- 谱聚类划分n_cluster个时间阶段，每阶段取文本相关性最高的帧作为候选关键帧

**MSP（运动信号处理）**：
- 基于冻结的Meta-Llama-3-8B-Instruct，单次前向推理输出结构化信号
- 输出：$\text{MSP}(E) = \{\mathcal{G}, \mathcal{M}\}$，其中G为二值判断是否含运动，M为[0,1]强度的运动重要性
- 仅关注目标对象自身动作，忽略外观属性与空间描述；无运动语义时M=0

**MIC（运动影响校准）**：
- 基础可靠性得分：$S_{base}(k) = \beta S_{seg}(k) + (1-\beta)S_{align}(k)$，β=0.5平衡分割置信度与文本对齐度
- 局部运动显著性：$S_{motion}(k) = \frac{1}{2\delta}\sum_{t=k-\delta+1}^{k+\delta}\|c_t - c_{t-1}\|$，c_t为掩码质心
- 动态校准得分：$S_{dynamic}(k) = S_{base}(k) \cdot (1 + \alpha \cdot S_{motion}(k) \cdot \mathcal{M})$，α控制校准强度
- 最终门控决策：$S_{final}(k) = \begin{cases} S_{base}(k), & \text{if } \mathcal{G}=0 \\ S_{dynamic}(k), & \text{if } \mathcal{G}=1 \end{cases}$，选取得分最高帧为锚帧进行双向传播

## 实验与结果
- **基准数据集**：Ref-YouTube-VOS、Ref-DAVIS17、MeViS(valid/valid^u)、A2D-Sentences、JHMDB-Sentences（共6个）
- **Ref-YouTube-VOS**：J&F=74.7，超越MPG-SAM2（0.8%）和ReferDINO（5.4%）
- **Ref-DAVIS17**：J&F=77.7，超越FlowRVS（+4.4%）和MPG-SAM2（+5.3%）
- **MeViS valid^u**：J&F=62.5，超越DMVS（+4.2%）；valid集：J&F=54.9，超越MPG-SAM2（+1.2%）
- **A2D-Sentences**：oIoU=81.7，mIoU=72.6，全面SOTA；JHMDB-Sentences：mIoU=73.0，超越SgMg（+0.4%）
- **消融实验**：四组件全开（ID 4）在MeViS上J&F=54.9；STSC贡献最大（+0.7）；α敏感但稳定，默认值0.1最优；三相似度分量均需保留
- **效率**：MSP 16.81 FPS，STSC 20.11 FPS，EMC推理7.34 FPS，显存10.0GB

## 相关工作脉络
- **ReferFormer/MTTR/HTML**：早期DETR-based端到端查询驱动范式，统一编码时空信息，缺乏运动语义显式建模。
- **LMPM**：探索语言-时序视觉交互，但运动语义仍与外观信息耦合，未解耦。
- **DsHmp/DMVS**：分别解耦静态/分层运动感知和解耦运动表达与实例分割，但运动线索仍以均匀方式参与决策，未建立表达→运动重要性的显式映射。
- **LoSh**：从语言侧建模长短时表达，缓解对运动语义的过度依赖，但未主动根据表达需求校准运动参与程度。
- **MPG-SAM2/SAMWISE**：最新SOTA方法融合SAM2与全局上下文，但仍采用统一时序建模，缺乏表达条件化的运动调节机制。
- **Findtrack**：解耦识别与传播的范式设计，与本文在结构化建模思路上有共鸣，但本文聚焦于运动语义的显式校准。

## 局限性与未来方向
- **MSP弱运动检测不足**：对"eating"等微弱动作容易漏检，对隐含运动（如"the last sheep to appear"）识别困难，存在过检测倾向。
- **依赖LLM推理**：MSP使用8B参数模型，虽可预计算缓存，但增加端到端系统延迟。
- **参数设置数据集依赖**：α和n_cluster需针对不同数据集调优，泛化性有待验证。
- **未涉及多目标交互场景**：复杂多目标运动交互下，质心位移运动显著性的度量可能存在歧义。
- **未来方向**：增强MSP对弱/隐含运动的鲁棒性；探索轻量级运动信号提取器；扩展至多目标跟踪分割场景。

## 研究启发与可借鉴点
- **显式语义先验设计**：将隐式编码的高维特征转化为低维可解释控制信号（G,M），为跨模态任务中的语义-决策接口设计提供新思路。
- **门控校准机制**：基于语义条件的双分支决策（基础分/运动校准分），避免运动线索对静态表达的干扰，可迁移至其他视频理解任务。
- **结构化时间候选空间**：通过多源相似度联合谱聚类构建语义时间阶段，而非逐帧处理，有效降低计算冗余，适用于长视频理解。
- **评估细化**：按静态/动态表达分类分析性能分布，揭示方法对不同语义条件的稳定性，值得在后续工作中借鉴。
- **结合团队方向**：若团队研究多模态时序定位或视频问答，MSP的门控思想可迁移至"语义条件化特征选择"场景。

## 关键术语表
**Referring Video Object Segmentation (RVOS)**：基于自然语言描述在视频序列中进行像素级目标分割的任务。
**Motion Signal Processing (MSP)**：利用LLM从表达中提取运动重要性的可解释控制信号模块。
**Motion Influence Calibration (MIC)**：根据MSP信号自适应校准运动线索贡献的门控校准模块。
**Semantic Temporal Stage Construction (STSC)**：通过多源相似度谱聚类构建表达相关语义时间阶段的模块。
**Motion Strength (M)**：运动重要性量化指标，取值[0,1]，表示表达对运动线索的依赖程度。
**Base Reliability Score**：基于单帧分割置信度与文本对齐度的基础关键帧得分。
**Local Motion Salience**：目标掩码质心在局部时间窗口内的平均位移，衡量运动强度。
**Valid^u Set**：MeViS数据集的训练阶段离线评估子集，用于验证泛化能力。

## 可复现要素
- **数据集**：Ref-YouTube-VOS、Ref-DAVIS17、MeViS、A2D-Sentences、JHMDB-Sentences（均为公开基准）
- **代码开源**：是，发布于 https://github.com/Jeven7/EMC
- **权重**：使用预训练模型EVF-SAM、BEiT、SAM2、Alpha-CLIP、CLIP、Meta-Llama-3-8B-Instruct（均已公开）
- **关键超参**：λ=0.5, w_t=w_τ=w_v=1, β=0.5；α在Ref-YouTube-VOS/A2D为0.2，Ref-DAVIS17/JHMDB为0.4，MeViS为0.1；n_cluster对应为5/4/10
