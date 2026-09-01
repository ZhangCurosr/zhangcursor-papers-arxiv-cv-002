---
title: "Motion-Based-Tokenization-for-Cross-Dataset-Egocentric-Gaze"
source: https://arxiv.org/pdf/2608.22926v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 19:27:50"
field: "多模态表征学习"
keywords: ["gaze tokenization", "cross-dataset transfer", "egocentric vision", "motion vocabulary", "domain shift", "eye tracking"]
innovations: ["提出事件对齐角位移tokenization(AD)及其多维度评估协议", "证明事件接口是表示一部分而非中性预处理", "揭示SO(3)过度分段与VQ码本政策对结构保留的差异化影响"]
benchmarks: ["AEA↔RITW", "EGTEA cross-interface"]
---

# 论文速读：Motion-Based-Tokenization-for-Cross-Dataset-Egocentric-Gaze

## 一句话总结
论文提出将事件对齐的固定视距角位移（angular displacement）作为可解释的运动词汇表，用于跨数据集中心视角眼动序列建模，并通过系统对比多种token化策略，揭示了表示选择、事件接口与时间粒度共同决定跨域转移性能。

## 研究问题与动机
1. **眼动表示缺乏跨数据集共识**：原始轨迹细节丰富但含噪声且依赖设备，粗粒度事件标签易于建模但可能丢失局部运动结构，空间token化在域偏移下易失效。
2. **现有离散化方法未充分探索运动条件性**：尽管眼动移位常以振幅和方向刻画，但以角位移作为事件条件化离散词汇表用于跨数据集序列建模的研究仍不充分。
3. **低困惑度可能误导转移评估**：表征可能因目标流更容易预测而获得低域外困惑度，而接近确定性的小词汇表即使保留极少运动信息也能显得稳定。
4. **事件构建本身参与表示形成**：不同事件接口（如I-VT、原生标注、帧跨度）会显著改变同一tokenization器所暴露的转移行为和结构特性。

## 核心贡献（创新点）
1. **提出事件对齐的固定视距角位移tokenization（AD）**：将事件条件化的(∆yaw, ∆pitch)量化到固定2D网格，与事件标签、空间网格、绝对角度等基线形成系统对比。
2. **设计多维评估协议控制目标可预测性干扰**：引入目标域后悔度（target-domain regret）、低阶参考、配对自助法、顺序敏感性、motif重叠和冻结结构探针，区分真实结构保留与可预测性假象。
3. **揭示SO(3)过度分段的失败模式**：完整旋转角轴tokenization因联合量化导致局部运动结构碎片化，退化为仅角度变量后恢复强转移能力。
4. **证明事件接口是表示的一部分而非中性预处理**：在EGTEA上匹配对比三种事件构建方式，发现原生事件降至EGTEA后悔度最低（0.172 bits/token），而帧跨度事件motif重叠为零且作为源时严重失败。

## 方法详解
- **AD（Angular Delta）tokenization**：对每个有效事件，在事件后80ms插值获取yaw/pitch，计算角位移并量化到固定2D网格；无有效目标时输出事件特定零token；事件标签与表征bin联合索引单一token。
- **对比表示**：AA（anchor ang，离散绝对角度）、ASM（ang state and motion，交错的绝对状态与局部运动）、SD（SO(3) log-map连续视线间旋转）、EVT（仅事件标签）、SGE（spatial grid+event）、VQD（向量量化角位移，冻结码本）。
- **目标域后悔度**：$R_{\mathrm{raw}}(S \to T) = H_T(q_S) - H_T(q_T)$，其中$q_S$为源训练模型、$q_T$为目标训练模型，归一化后悔度除以低阶基线（unigram或Markov）与目标模型的差距。
- **结构诊断探针**：在冻结的序列模型状态上训练线性探针预测运动状态、运动象限、方向逆转可解码性。
- **时序扫描**： swept event-pair stride（40/80/120 ms）与context length（8/16/32/64），发现最优设置依赖转移方向：AEA→RITW用120ms stride+context 32，RITW→AEA用40ms stride+context 64。

## 实验与结果
- **数据集**：AEA（日常活动，Project Aria）、RITW（野外阅读，Project Aria）、EGTEA（烹饪，SMI眼动仪+BeGaze导出）。
- **AEA↔RITW主基准**：AD平均退化0.05±0.07，低于ASM（0.15）、SGE（0.19）、AA（0.21）、SD（1.30）。
- **后悔度对比（Table 1）**：AEA→RITW方向AD原始后悔度0.634 bits/token，VQD为1.620，配对差异-0.964 [CI: -1.183, -0.756]；RITW→AEA方向AD 0.503 vs VQD 0.528，差异不显著。
- **顺序敏感性**：AD为0.40±0.10，显著高于AA（0.14）和SGE（0.15）。
- **Motif重叠**：AD bigram/trigram Jaccard为0.493/0.449，远高于AA（0.031/0.003）和SD（0.034/0.005）。
- **EGTEA接口对比**：原生事件降至EGTEA后悔度0.172±0.021（AEA源）和0.297±0.036（RITW源），帧跨度事件作为源时后悔度飙升至5.806/6.048，motif重叠为零。

## 相关工作脉络
1. **Scan-Match / MultiMatch**：将注视序列转换为空间-时间字母序列或多维向量比较，侧重scanpath匹配而非序列建模的跨域可迁移性。
2. **离散眼动tokenization（如[22]）**：比较连续vs离散tokenization用于语言模型架构的眼动预测，本文从跨数据集转移结构保留角度提供补充评估。
3. **域偏移研究（Gaze360、MPIIGaze等）**：主要测量端到端估计精度下降，本文关注表示本身在shift下是否保留可预测的时序与几何结构。
4. **结构化眼动建模（HMM、recurrence patterns等）**：使用隐状态转换或重复模式，但本文强调表示选择本身就是建模问题的一部分。
5. **多 person gaze following（MTGS、Sharingan）**：使用结构化gaze或person-specific tokens提升泛化，本文聚焦单视角egocentric场景的跨数据集base representation。

## 局限性与未来方向
- 仅覆盖三个egocentric数据集，结论限于测试接口下的event-based transfer，不能推广到任意眼动仪或刺激范式。
- AEA与RITW共享Project Aria传感器，坐标约定和预处理假设一致，实际跨设备转移可能更难。
- 连续模型与离散模型使用不同度量体系，不能直接互换比较。
- 监督探针仅衡量结构保留与行为可解码性，未验证跨数据集视频理解任务的实际效用。
- 未来需更大规模平衡egocentric语料分离数据集大小、任务范式和传感器效应，并需结合egocentric视频进行活动识别、任务状态预测等应用级测试。

## 研究启发与可借鉴点
1. **多维评估协议设计**：用目标域后悔度控制目标可预测性干扰，用顺序敏感性/motif重叠/探针诊断区分真实结构保留与可预测性假象，可有效避免"低困惑度=好表示"的误判。
2. **事件接口敏感性分析**：将事件构建（boundary定义、时间单位、标签体系）视为表示设计的一部分而非中性预处理，可在同类工作中作为标准消融。
3. **时序粒度与上下文长度联合扫描**：证明最优时间尺度依赖转移方向和数据集行为特性，建议在tokenization工作中报告多尺度敏感性而非单一最优值。
4. **SO(3)过度分段的失败案例**：对3D旋转的完整角-轴量化可能导致transition结构碎片化，仅角度变量可恢复可迁移性，提示运动token化需在表达力与可用性间权衡。
5. **与团队方向结合机会**：该方法可迁移至多模态egocentric video理解预训练（如Egom2p框架），或将运动token与空间token联合用于gaze-conditioned action prediction。

## 关键术语表
- **Angular Delta (AD)**：事件对齐的固定视距角位移tokenization，将(∆yaw, ∆pitch)量化到2D网格并与事件标签联合索引。
- **Target-domain Regret**：源训练模型与目标训练模型在目标测试集上的交叉熵差值，控制目标可预测性后的转移性能度量。
- **Order Sensitivity**：训练于真实序列、测试于打乱序列时的 perplexity 相对损失，衡量模型对时序结构的依赖程度。
- **Motif Overlap**：跨数据集的bigram/trigram Jaccard重叠率，衡量局部转移结构的共享程度。
- **Structural Probe**：在冻结序列模型状态上训练的线性分类器，用于解码运动状态、象限、方向逆转等结构信息。
- **Event Interface**：眼动事件的定义与构建方式（如I-VT算法、原生标注、帧跨度），直接影响tokenization器可访问的行为单元。
- **VQD（Vector-Quantized Delta）**：将归一化角位移映射到冻结codebook质心的最近邻离散化方法。
- **SO(3) Log-map**：连续视线射线间的旋量表示，完整角-轴量化但可能导致跨域碎片化。

## 可复现要素
- 数据集：AEA [20]、RITW [24]、EGTEA [18]，均公开可获取。
- 代码：https://anonymous.4open.science/r/motionTokenizer_egocentric-BE1E/
- 关键超参：AD默认stride 80ms、context length 16/32；VQD K=128；连续基线使用Gaussian NLL。
- 训练：Transformer架构、3 seeds、session-level 70/15/15 split。
- 其他：详细超参和tokenizer实现见补充材料。
