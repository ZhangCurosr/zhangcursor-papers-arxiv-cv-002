---
title: "SCOPE-Router-Cost-Aware-Open-Set-VLM-Routing-for-Execution-O"
source: https://arxiv.org/pdf/2608.12127v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 02:22:08"
field: "多模态模型路由与调度"
keywords: ["VLM路由", "开放集路由", "成本感知", "模型选择", "执行任务", "多模态路由"]
innovations: ["首个面向执行任务的VLM路由基准VLM-ExecRouterBench，覆盖代码/Agent/搜索三大域", "混合校准策略（随机/诊断/多样性）构建查询感知模型行为画像，实现无需重训的开放集路由", "CRM+RCCR成本感知训练目标，逐对独立BCE消除多正稀释并正则化路由偏好相似查询"]
benchmarks: ["VLM-ExecRouterBench", "VL-RouterBench", "MMR-Bench"]
---

# 论文速读：SCOPE-Router: Cost-Aware Open-Set VLM Routing for Execution-Oriented Tasks

## 一句话总结
本文提出了第一个面向执行任务（代码生成、Agent工具调用、多步搜索）的VLM路由基准VLM-ExecRouterBench，并设计了SCOPE-Router——一种基于混合校准行为画像的双塔开放集路由模型，配合CRM+RCCR成本感知训练目标，在三基准上均取得Rank Score第一，且在OOD和开放集场景下显著提升。

## 研究问题与动机
- **领域覆盖不足**：现有VLM路由基准仅覆盖传统VQA任务（视觉推理、OCR、图表理解），而VLM在代码生成、工具调用Agent、多步Web检索等执行型任务中日益普及，这些任务对模型能力要求不同、互补性更强，却没有任何路由基准加以覆盖。
- **开放集能力欠缺**：新模型不断涌现，传统路由将模型视为固定类别标签，新增模型需重新训练；现有开放集方法（UniRoute、ICL-Router）缺乏系统的校准数据选择优化，在有限校准预算下导致模型画像判别力低。
- **训练目标与成本脱耦**：RouterDC的对比目标仅按正确性划分正负样本，忽略成本差异；VL-RouterBench的softmax CE损失在多正样本场景下稀释梯度；UniRoute仅在推理时作为后处理惩罚引入成本，训练过程本身不感知成本。

## 核心贡献（创新点）
- **首个面向执行任务的VLM路由基准VLM-ExecRouterBench**：覆盖Code/Agentic/Search三大域、34k样本、11个候选模型（价格跨度近两个数量级），将任务统一为Routing Input/Execution Context/Verification Rule三元组。区别于VL-RouterBench仅覆盖传统VQA，本文首次将执行型任务纳入路由评估。
- **双塔开放集路由架构SCOPE-Router**：通过混合校准（随机/诊断/多样性采样各50%/30%/20%）构建查询感知的模型行为画像，新模型仅需在小型校准集上推理一次即可生成画像加入路由池，无需重新训练路由。区别于UniRoute（随机划分验证集）和ICL-Router（单一信号过滤），本文的混合策略使画像在有限预算下更具判别力。
- **架构无关的成本感知训练目标CRM+RCCR**：CRM将成本偏好编码为连续相关性目标，采用逐对独立sigmoid BCE替代softmax CE以消除多正样本梯度稀释；RCCR通过对齐路由偏好相似的查询拉近其在路由空间的距离。该目标可无缝替换RouterDC/ZOOTER/CosineCls/VLC的损失函数，带来+1.25~6.21 Rank Score的提升，区别于此前所有方法在训练过程中均忽略或事后处理成本的局限。

## 方法详解
- **数据构造**：每个样本统一拆分为三部分——Router看到的Routing Input、模型执行的Execution Context、判定对错的Verification Rule。对所有候选模型执行后构建正确率矩阵$Y \in \{0,1\}^{N \times K}$和成本矩阵$C \in \mathbb{R}_+^{N \times K}$。
- **混合校准集构建（|S_calib|=1024）**：三种采样策略按比例分配预算——随机采样（50%保证分布覆盖）；诊断采样（30%）按公式$d_i = 0.7 \cdot 4p_i(1-p_i) + 0.3 \cdot \text{cost\_spread}_i^{\text{norm}}$选择模型分歧大且成本分散的样本；多样性采样（20%）在冻结嵌入空间上用MiniBatchKMeans选取距聚类中心最近的样本。三种策略互补，全组合最优。
- **模型行为画像$\mathbf{p}_m$**：拼接Behavioral Vector（逐样本正确率$\mathbf{y}_m$、归一化成本$\tilde{\mathbf{c}}_m$、"性价比"向量$\mathbf{y}_m \odot (1-\tilde{\mathbf{c}}_m)$、8维汇总统计）和Semantic Vector（$\bar{\mathbf{e}}_m^+$正确样本语义均值、$\bar{\mathbf{e}}_m^-$错误样本语义均值、$\bar{\mathbf{e}}_m^v$性价比向量均值），最终维度为$\mathbb{R}^{3S+8+3D}$（默认D=2048）。
- **双塔架构**：Query Encoder用BGE-M3+DINOv2-large融合后经2层MLP（隐藏128→输出64）投影至路由空间；Profile Encoder同样2层MLP将画像投影至相同64维空间；匹配分数$s_{i,m} = \mathbf{q}_i^\top \hat{\mathbf{p}}_m / \tau$，推理时取arg max。
- **CRM损失**：相关性目标$R_{i,m} = \mathbb{1}[Y_{i,m}=1] \cdot \exp(-\lambda \cdot \alpha \cdot (C_{i,m}-C_i^{\min}))$，最便宜的正确答案模型得R=1，其他正确模型得$R\in(0,1)$，错误模型得0；采用逐对独立BCE而非行方向softmax，避免多正样本稀释。
- **RCCR损失**：对L1归一化的相关性向量计算样本间相似度$w_{ij}$，在以相似度为权重的InfoNCE对比学习中拉近路由偏好相似的查询在路由空间中的距离，$\mathcal{L} = \mathcal{L}_{\text{CRM}} + 0.1 \cdot \mathcal{L}_{\text{RCCR}}$。
- **评估指标Rank Score**：$S_\beta = \frac{(1+\beta)\cdot A \cdot \widetilde{C}}{\beta \cdot A + \widetilde{C}}$，其中$\widetilde{C}$为对数归一化成本分，$\beta=0.1$偏向准确率。

## 实验与结果
- **数据集**：VLM-ExecRouterBench（34k样本，11模型，3域）、VL-RouterBench、MMR-Bench，所有路由均在对应基准的训练集上训练。
- **主要结果（VLM-ExecRouterBench）**：SCOPE-Router Rank Score **80.94±1.22**，排名第一；次优CosineCls为79.55±0.50；相对Strongest基线准确率仅下降5.21pp但成本降低85%。
- **VL-RouterBench**：SCOPE-Router **76.18±1.44**排名第一，较次优RouterDC（74.59±1.05）提升1.59点；准确率79.11%为最高，成本较Strongest降低64%。
- **MMR-Bench**：SCOPE-Router **61.23±0.75**排名第一，超次优UniRoute-KM（59.72±0.60）1.51点；成本仅为Strongest的0.8%（0.65 vs 81.30）。
- **OOD泛化**：在5个保留数据集上，SCOPE-Router Rank Score **88.14**，超次优K-means（86.30）1.84点，成本仅为其21.8%。
- **开放集（Doubly OOD）**：隐藏5个模型（含GPT-5.4、Claude Opus 4.6等），在保留数据集上SCOPE-Router Rank Score **85.35**，超UniRoute-KM（78.60）**6.75点**。
- **跨架构迁移**：CRM+RCCR作为替换损失应用于RouterDC/ZOOTER/CosineCls/VLC，Rank Score提升+1.25~6.21点，在VLM-ExecRouterBench上提升更大。
- **消融**：CRM贡献+1.0点（相对softmax）；RCCR仅在CRM基础上有效（+0.68点）；混合校准全组合（76.18）优于任一单策略（最佳Diversity 75.69）；MLP隐藏维度最优为128，对维度不敏感；冻结Query Embedder优于微调。

## 相关工作脉络
- **VL-RouterBench**（Lin et al., 2026）：首个VLM路由基准，覆盖14数据集17模型，但仅限传统VQA；本文将其扩展至执行域，首次纳入代码/Agent/搜索任务。
- **RouterDC**（Chen et al., 2024）：双对比学习路由，正负样本划分仅依赖正确性，忽略成本；本文CRM将成本编码进连续目标并消除多正稀释。
- **UniRoute**（Jitkrittum et al., 2025）：基于聚类的通用LLM路由，仅将成本作为推理后惩罚，训练不感知成本；其校准数据随机划分，判别力低；本文混合校准+开放集画像机制更优。
- **LLM-RouterBench / RouterArena**（Li et al., 2026; Lu et al., 2025）：文本路由基准和平台；本文聚焦多模态且首次覆盖执行任务。
- **MMR-Bench**（Ma et al., 2026）：多模态推理路由基准，含预算感知评估；本文在保持兼容的同时扩展至代码/Agent/搜索域并改进训练目标。
- **Routeprofile**（Xu et al., 2026）：基于GNN的冷启动LLM路由；本文面向VLM且无需图结构，直接通过校准集生成行为画像。

## 局限性与未来方向
- **开放集依赖校准集质量**：新模型仍需在一次标注校准集上推理生成画像，性能取决于校准样本能否充分暴露新模型的能力边界和失败模式。
- **单次路由决策**：当前路由器对每个query做一次性任务级选择，不支持长程Agentic/Search轨迹中的动态路由调整（如前期需精细OCR、后期需检索综合）。
- **未来方向**：从紧凑校准集学习更鲁棒的模型画像；设计跨中间执行步骤的动态路由策略；进一步探索轨迹感知的路由方法。

## 研究启发与可借鉴点
- **混合校准策略的设计思路**：将校准预算按比例分配给分布覆盖（随机）、判别力最大化（诊断/分歧+成本分散）和语义覆盖（多样性聚类）三个互补维度，可作为开放集系统构建特征画像的通用模板。
- **逐对独立BCE替代softmax CE解决多正稀释**：CRM的核心洞察——当多个模型均正确时，softmax归一化会稀释梯度信号，改用逐对独立sigmoid BCE可使每个模型的梯度独立计算，该技巧可直接迁移至任何多分类/多正样本排序任务。
- **RCCR的分布正则化思想**：通过路由偏好向量的余弦相似度加权对比损失，将"路由模式相似"的查询拉近，使路由器能从共享的模型适用性结构中泛化，这一正则化模式可适用于其他需要跨样本结构学习的排序/推荐任务。
- **实验设计借鉴**：同时报告Accuracy/Cost/Rank Score三指标、做λ扫描绘制Pareto前沿、在OOD和开放集双重设定下评估、验证目标函数对多种架构的迁移性，是一套完整且说服力强的评估范式。
- **与团队方向结合机会**：可将CRM+RCCR迁移至本团队的模型选择/任务调度模块；混合校准策略可应用于团队内部多模型集成系统的新模型快速接入；开放集画像机制可与团队已有的工具调用路由工作结合。

## 关键术语表
- **VLM-ExecRouterBench**：首个面向执行任务（代码/Agent/搜索）的VLM路由评测基准，含34k样本和11个跨价位的候选模型。
- **Query-Aware Profile**：基于校准集构建的模型行为画像，融合逐样本正确率、成本、性价比及语义方向信息，供开放集路由匹配使用。
- **Hybrid Calibration**：由随机采样（50%）、诊断采样（30%，基于模型分歧与成本分散）和多样性采样（20%，基于嵌入空间聚类）组成的校准集选择策略。
- **CRM（Cost-aware Relevance Matching）**：将成本偏好编码为连续相关性目标的损失，用逐对独立sigmoid BCE替代softmax CE以消除多正样本梯度稀释。
- **RCCR（Routing-Consistency Contrastive Regularization）**：通过在路由空间中拉近路由偏好相似的查询作为对比正则项，提升模型对共享适用性结构的泛化能力。
- **Rank Score**：以调和平均融合路由准确率和归一化成本分的综合指标（$S_\beta$），用于统一排序各路由方法。
- **Open-Set Routing**：允许新模型在无需重训路由器的情况下，仅通过校准集推理生成行为画像即可加入路由候选池的设定。

## 可复现要素
- **数据集**：VLM-ExecRouterBench已公开，HuggingFace链接：https://huggingface.co/datasets/Kirito-Lab/VLM-ExecRouterBench；包含数据生成pipeline、转换脚本、执行器prompt、工具schema、验证代码及执行轨迹。
- **代码**：已开源，GitHub链接：https://github.com/yutao1024/SCOPE-Router。
- **关键超参**：校准集大小1024；混合比例50%/30%/20%；QueryMLP/ProfileMLP隐藏维度128、输出维度64；dropout 0.5；温度τ默认未明示（文中仅提及参数名）；λ（成本敏感度）在{0,10,100,1000,10000,+∞}中 sweep，默认最优λ=10；μ=0.1；学习率0.001；weight decay 0.003；batch size 512；max 100 epochs，early stopping patience 20；Query Encoder使用BGE-M3+DINOv2-large，默认冻结。
