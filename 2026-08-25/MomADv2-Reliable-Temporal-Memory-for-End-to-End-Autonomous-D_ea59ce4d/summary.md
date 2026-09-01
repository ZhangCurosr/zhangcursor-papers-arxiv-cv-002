---
title: "MomADv2-Reliable-Temporal-Memory-for-End-to-End-Autonomous-D"
source: https://arxiv.org/pdf/2608.23405v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 19:27:37"
field: "端到端自动驾驶规划"
keywords: ["端到端自动驾驶", "长时程规划", "状态空间模型", "流匹配", "选择性记忆", "轨迹修正", "Navsim", "MomAD"]
innovations: ["提出选择性状态空间记忆查询模块SSM-Q，通过命令一致性、时序连续性和轨迹级对齐三重过滤历史规划查询", "设计流匹配轨迹残差修正器FM-Ref，以锚点规划为起点学习有界残差速度场实现精细轨迹修正", "两阶段解耦训练策略，冻结主干仅微调记忆与修正模块，避免破坏预训练表征"]
benchmarks: ["nuScenes", "Bench2Drive", "NAVSIM v1 navtest", "NAVSIM v2 navtest", "NAVSIM v2 navhard"]
---

# 论文速读：MomADv2-Reliable-Temporal-Memory-for-End-to-End-Autonomous-D

## 一句话总结
MomADv2 提出了一种可靠的状态空间记忆框架，通过**选择性过滤**历史规划查询（基于时序连续性与指令一致性）并结合**流匹配残差修正**，显著提升了端到端自动驾驶长时程规划的连续性与安全性，在 6 秒规划下较 MomAD 平均碰撞率降低 15.6%。

## 研究问题与动机
- **长时程规划的时间一致性需求**：端到端自动驾驶需要跨帧保持规划意图的连续性，避免轨迹抖动和模式不稳定。
- **历史记忆可能成为干扰源**：现有方法（如 MomAD）无条件复用历史规划查询，当驾驶指令切换或场景突变时，过时/不一致的历史信息会误导当前决策。
- **局部偏差与误差累积**：长时程规划中微小的逐帧偏差会随时间累积，导致轨迹偏离专家示范。
- **选择性记忆利用是关键挑战**：如何在保留有用历史信息的同时抑制无效或指令不一致的干扰，是当前研究的空白点。

## 核心贡献（创新点）
1. **选择性状态空间规划记忆查询模块（SSM-Q）**：通过命令一致性、时序连续性 token 匹配和轨迹级对齐三重过滤，仅检索可靠的历史规划查询，并用选择性状态空间模型（Selective SSM/Mamba）建模意图演化；与 MomAD 盲目复用历史的本质区别在于"选择性保留+不可靠抑制"。
2. **流匹配轨迹残差修正器（FM-Ref）**：从已规划轨迹出发学习一条连续残差速度场指向专家轨迹，以 Euler 积分 + 有界残差门控进行推理；与扩散模型从零生成的本质区别在于"增量修正而非从头生成"，保持锚点规划的稳定性。
3. **两阶段解耦训练策略**：先完整预训练 MomAD 基线，再冻结主干仅微调 SSM-Q 和 FM-Ref，避免大参数更新破坏已有表征；这区别于一次性联合训练的做法。
4. **多基准 SOTA**：在 nuScenes 开放环、Bench2Drive 和 NAVSIM v1/v2 闭环基准上均取得领先，6 秒规划平均碰撞率降低 15.6%，验证了方法的有效性和通用性。

## 方法详解
**整体框架**：多视图相机输入 → 稀疏场景表征 → 锚点规划解码器产出初始候选 → SSM-Q 增强规划查询 → FM-Ref 轨迹修正。

**SSM-Q（选择性状态空间记忆查询）**：
- **记忆库设计**：维护 token-keyed 记忆库 $\mathcal{M}_t = \{\chi_{t-k} \mapsto (\mathbf{Q}_{t-k}^{\mathrm{raw}}, \mathbf{Y}_{t-k}^{\mathrm{base,cmd}}, c_{t-k})\}_{k=1}^K$，存储**增强前**的原始查询和基线轨迹，防止 SSM 修正结果递归写入导致漂移。
- **可靠历史检索**：有效性指示器 $m_{t,k} = \mathbb{I}[c_t = c_{t-k}, \chi_t^{(-k)} = \chi_{t-k}, \mathbf{Q}_{t-k}^{\mathrm{raw}} \neq \mathbf{0}]$，仅在同命令、同时序 token 且查询非空时检索。
- **轨迹级对齐**：通过时移距离 $D_{\mathrm{shift}}$ 补偿帧间时序偏移，为每个当前候选匹配最相似的历史候选：$h_{t,k,j}^* = \arg\min_h D_{\mathrm{shift}}(\mathbf{Y}_{t-k,h}^{\mathrm{base,cmd}}, \mathbf{Y}_{t,j}^{\mathrm{base,cmd}})$。
- **加权聚合**：$\bar{\mathbf{Q}}_{t,j}^{\mathrm{hist}} = \sum_k w_{t,k,j} \widehat{\mathbf{Q}}_{t,k,j}^{\mathrm{hist}}$，权重 $\phi_{t,k,j} = m_{t,k} \exp(-d_{t,k,j}/\alpha) \exp(-k/\beta)$ 综合命令有效性、轨迹相似度和时间衰减。
- **可靠性门控**：$r_{t,j} = r_t^{\mathrm{cnt}} \cdot r_{t,j}^{\mathrm{dist}}$，其中 $r_t^{\mathrm{cnt}}$ 保证足够多的有效历史，$r_{t,j}^{\mathrm{dist}}$ 保证轨迹距离在阈值内，两者共同决定历史是否参与增强。
- **SSM 编码与残差注入**：将历史序列与当前查询拼接后通过 Selective SSM Encoder，得到时序增强的查询 $\mathbf{H}_{t,j}^{\mathrm{cur}}$；残差 $\Delta\mathbf{Q}$ 经 reliability-gated 裁剪后叠加到原查询：$\widetilde{\mathbf{Q}}_{t,j} = \mathbf{Q}_{t,j} + \mathrm{Clip}(g_{t,j}\Delta\mathbf{Q}_{t,j}, \rho\|\mathbf{Q}_{t,j}\|_2)$。

**FM-Ref（流匹配轨迹残差修正）**：
- **条件流场**：将 SSM 增强的规划轨迹 $\overline{\mathbf{Y}}_t^{\mathrm{ref}}$ 归一化为起点 $\mathbf{x}_0$，专家轨迹归一化为终点 $\mathbf{x}_1$，在 $[0,1]$ 流时间上采样中间态 $\mathbf{x}_\tau = (1-\tau)\mathbf{x}_0 + \tau\mathbf{x}_1$。
- **速度场估计**：融合查询嵌入 $\phi_q(\widetilde{\mathbf{Q}}_t)$、轨迹编码 $\phi_x(\mathbf{x}_\tau)$ 和时间嵌入 $\phi_\tau(\tau)$，经 MLP + Transformer Encoder 后由 MLP velocity head 预测残差速度 $\mathbf{v}_\theta$。
- **流匹配损失**：$\mathcal{L}_{\mathrm{FM}} = \|\mathbf{v}_\theta(\widetilde{\mathbf{Q}}_t, \mathbf{x}_\tau, \tau) - (\mathbf{x}_1 - \mathbf{x}_0)\|_2^2$，监督网络学习从规划轨迹到专家轨迹的线性插值速度场。
- **门控残差推理**：使用 2 步 Euler 积分得到 $\mathbf{x}_S$，通过可学习有界门控 $\eta = \eta_{\min} + \eta_{\max}\sigma(g_{\mathrm{flow}})$ 控制修正强度，最终输出 $\overline{\mathbf{Y}}_t^{\mathrm{flow}} = s[\mathbf{x}_0 + \eta(\mathbf{x}_S - \mathbf{x}_0)]$。
- **总损失**：$\mathcal{L} = \mathcal{L}_{\mathrm{plan}} + \mathcal{L}_{\mathrm{SSM}} + \lambda_{\mathrm{flow}}\mathcal{L}_{\mathrm{flow}} + \lambda_{\mathrm{FM}}\mathcal{L}_{\mathrm{FM}}$，其中 $\mathcal{L}_{\mathrm{flow}}$ 同时在位移形式和累积形式上计算回归损失。

## 实验与结果
**数据集**：
- **Open-Loop**：nuScenes（1000 场景，只用多视图图像）
- **Closed-Loop**：Bench2Drive（220 路线）、NAVSIM v1 navtest、NAVSIM v2 navtest/navhard

**主要结果**：
- **nuScenes 6秒规划**：MomADv2 平均 L2 误差 1.21m（较最优基线 DIVER 提升 11.7%），平均碰撞率 0.76%（较 DIVER 提升 11.6%），1秒碰撞率 0.00%。
- **Bench2Drive（含专家蒸馏）**：Driving Score 78.82（较 MomAD 47.91 提升 30.91），Success Rate 46.50%（较 MomAD 18.11% 提升 28.39pp），Comfortness 31.23 为最佳。
- **NAVSIM v1 navtest**：NC 99.0、DAC 97.1、TTC 95.5、PDMS 89.9，综合领先。
- **NAVSIM v2 navhard**：EPDMS 39.5（两阶段评估）。
- **NAVSIM v2 navtest**：DDC、EP、EC 三项第一，EPDMS 87.9。
- **轨迹预测一致性（TPC）**：4秒 1.04m（较 MomAD 提升 12.6%），5秒 1.32m（提升 9.0%）。

**最强结果**：Bench2Drive DS 78.82，NAVSIM v1 PDMS 89.9，nuScenes 6秒碰撞率 0.76%。

## 相关工作脉络
1. **MomAD（Song et al. 2025, CVPR'25）**：前作，通过直接复用历史规划查询维持规划动量；本文的核心改进是引入选择性过滤机制替代无条件复用。
2. **Mamba / Selective SSM（Gu & Dao 2024）**：状态空间模型的基础架构；本文将其适配于规划查询的时序建模，并引入可靠性门控机制。
3. **Flow Matching（Liu et al. 2025b, GuideFlow）**：流匹配在轨迹生成中的应用；本文将其用于"残差修正"而非从零生成，与锚点规划形成互补。
4. **TransFuser（Chitta et al. 2023, TPAMI'22）**：早期端到端融合的 Transformer 基线；本文在 NAVSIM 上与其对比，展示长时程规划优势。
5. **DiffusionDrive（Liao et al. 2025, CVPR'25）**：截断扩散模型用于生成式规划；本文与它在多基准上对比，证明选择性记忆 + 流匹配残差修正的竞争力。
6. **VAD / VADv2（Jiang et al. 2023/2026）**：向量化场景表示的端到端规划方法；本文在 Bench2Drive 上与 VAD 系列对比，DS 显著提升。

## 局限性与未来方向
- **依赖预定义驾驶指令**：记忆选择需已知命令（left/straight/right），无法处理隐式意图场景。
- **未建模未来不确定性**：记忆筛选基于确定性判断，缺乏对场景演化不确定性的显式建模。
- **未来方向**：无指令意图推断、不确定性感知记忆选择、结合世界模型的未来推理（论文自述）。

## 研究启发与可借鉴点
1. **选择性记忆机制的可迁移性**：三重过滤（命令一致性 + 时序连续性 + 轨迹对齐）的思路可迁移至其他需要历史记忆的序列决策任务（如机器人操作、对话系统）。
2. **"残差修正而非完全替代"的设计哲学**：FM-Ref 在保持锚点规划稳定性的同时进行精细修正，这种"主分支+辅助修正"的解耦设计适用于多种改进现有规划器的场景。
3. **两阶段解耦训练策略**：冻结主干微调新模块的策略既保留了预训练表征又避免了灾难性遗忘，值得在端到端系统的增量改进中借鉴。
4. **时移轨迹距离 $D_{\mathrm{shift}}$**：通过坐标变换和对齐窗口消除帧间时序偏移的轨迹匹配方法，可用于任何需要跨帧关联轨迹的场景。
5. **有界门控残差**：$\eta = \eta_{\min} + \eta_{\max}\sigma(\cdot)$ 的自适应修正强度控制，可推广到其他需要安全性约束的神经网络修正模块。

## 关键术语表
- **SSM-Q（Selective State-Space Planning Memory Query）**：选择性状态空间规划记忆查询模块，通过多重过滤检索可靠历史规划查询并用 SSM 建模意图演化。
- **FM-Ref（Flow-Matching Trajectory Residual Refiner）**：流匹配轨迹残差修正器，学习从规划轨迹到专家轨迹的连续残差速度场并进行有界修正。
- **Selective SSM / Mamba**：具有输入依赖选择扫描的状态空间模型，能够根据当前输入动态决定历史状态的保留或遗忘。
- **EPDMS（Extended Predictive Driver Model Score）**：NAVSIM v2 的综合评估指标，扩展了规则合规性（DDC、TLC、LK）和舒适度（HC、EC）项。
- **PDMS（Predictive Driver Model Score）**：NAVSIM v1 的评估指标，综合安全评分（NC、DAC）和软目标（EP、TTC、Comf）。
- **TPC（Trajectory Prediction Consistency）**：轨迹预测一致性指标，衡量跨帧预测轨迹的时域稳定性，值越低越好。
- **Anchor-based Planning**：基于锚点的规划方法，先生成离散候选轨迹再回归 refine，本文在此框架上增加时序记忆和流匹配修正。
- **Causal Stateful Inference**：因果有状态推理，每帧读取记忆库、执行规划后写入新条目，参数固定仅状态演化。

## 可复现要素
- **数据集**：nuScenes（公开）、Bench2Drive（公开）、NAVSIM v1/v2（公开）
- **代码/权重**：论文未明确声明是否开源
- **关键超参**：记忆长度 $K=4$，Euler 积分步数 $S=2$，温度参数 $\alpha, \beta$，距离阈值 $d_0, \gamma_d$，残差上限系数 $\rho$，损失权重 $\lambda_{\mathrm{flow}}, \lambda_{\mathrm{FM}}, \lambda_\Delta, \lambda_{\mathrm{cum}}$（论文未全部给出具体数值）
- **训练设备**：8× NVIDIA RTX 4090
- **训练轮数**：nuScenes 10 epochs（第二阶段），Bench2Drive 2 epochs，NAVSIM 100 epochs
