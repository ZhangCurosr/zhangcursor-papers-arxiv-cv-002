---
title: "MomADv2-Reliable-Temporal-Memory-for-End-to-End-Autonomous-D"
source: https://arxiv.org/pdf/2608.23405v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 19:27:36"
field: "端到端自动驾驶长时域规划"
keywords: ["端到端自动驾驶", "长时域规划", "状态空间模型", "Flow Matching", "时序记忆", "残差修正", "NAVSIM"]
innovations: ["提出基于指令/时序/轨迹一致性的选择性 SSM 记忆查询模块，避免无效历史干扰", "设计以规划输出为起点、向专家轨迹学习的 Flow-Matching 残差修正器", "提出先读后写且仅缓存原始查询的可靠记忆更新策略并给出多维度可靠性门控"]
benchmarks: ["nuScenes", "Bench2Drive", "NAVSIMv1", "NAVSIMv2"]
---

# 论文速读：MomADv2-Reliable-Temporal-Memory-for-End-to-End-Autonomous-D

## 一句话总结
本文提出 MomADv2，一个可靠的时空状态空间记忆框架，用于端到端自动驾驶长时域规划；通过选择性检索与当前指令一致的历史规划查询、并结合 Flow-Matching 残差修正，显著提升长时域规划的连续性与安全性，在 6 秒规划下较 MomAD 将平均碰撞率降低 15.6%。

## 研究问题与动机
- 端到端自动驾驶（E2E-AD）的长时域规划需要跨帧保持时序一致的规划意图，但现有方法通常无差别地复用历史规划状态，易引入干扰。
- 当场景发生动态变化、出现时序断层或高层导航指令切换时，历史记忆可能与当前意图不一致，从而误导规划决策。
- MomAD 表明保留规划动量可降低模式不稳定与轨迹抖动，但未考虑记忆可靠性问题，长期累积误差与错误历史仍会损害安全性。
- 因此需要在长时域规划中做到"选择性保留可靠历史、抑制与当前指令不一致的无效记忆"，同时还需缓解局部轨迹偏差与误差累积。

## 核心贡献（创新点）
- 提出 MomADv2 可靠状态空间记忆框架，与 MomAD 的"全量复用历史"策略不同，MomADv2 显式区分并过滤不可靠历史，避免负迁移。
- 设计 SSM-Q（Selective State-Space Planning Memory Query Module），按指令一致性、时序连续性、轨迹级对齐与缓冲区有效性四维规则检索历史规划查询，并与选择性状态空间机制建模意图演化；不同于以往直接拼接历史 token 的做法，本文引入可靠性门控与轨迹距离加权聚合。
- 提出 FM-Ref（Flow-Matching Trajectory Residual Refiner），以规划输出为起点学习向专家轨迹的连续残差速度场；与扩散类方法从随机噪声生成轨迹不同，该方法保留锚点规划稳定性，仅做细粒度局部修正。
- 在 NAVSIM、Bench2Drive 与 nuScenes 三个基准上系统评估，证明该方法可同步提升长时域轨迹精度、时序一致性与闭环安全评分。
- 给出可复用的"先读后写+仅缓存原始查询/基线轨迹"的记忆更新策略，避免经 SSM/Flow 修正后的输出被递归写回导致漂移放大。

## 方法详解
- **整体架构**：多视角图像经编码得到稀疏场景表示，生成初始 ego 规划查询与候选轨迹；在此基础上串联 SSM-Q 与 FM-Ref，输出经最终规划细化头处理得到分类、轨迹回归与规划状态。
- **记忆库表示**：维护按时间步倒序的条目集合 $\mathcal{M}_t = \{ \chi_{t-k} \mapsto (\mathbf{Q}_{t-k}^{\mathrm{raw}}, \mathbf{Y}_{t-k}^{\mathrm{base,cmd}}, c_{t-k}) \}_{k=1}^K$，其中 $(c_{t-k}, \chi_{t-k})$ 作为记忆键，$(\mathbf{Q}^{\mathrm{raw}}, \mathbf{Y}^{\mathrm{base,cmd}})$ 为对应值；仅写入原始查询与基线轨迹，避免 SSM 增强结果递归写入引发自我放大。
- **可靠历史检索条件**：$m_{t,k} = \mathbb{I}[c_t = c_{t-k},\; \chi_t^{(-k)} = \chi_{t-k},\; \mathbf{Q}_{t-k}^{\mathrm{raw}} \neq \mathbf{0}]$，只在一阶导航指令相同、帧级时序连续性成立且历史查询非空时保留候选。
- **轨迹级对齐**：由于不同帧对同一未来时刻的起始帧偏移不同，采用时空位移距离 $D_{\mathrm{shift}}$ 将历史轨迹的时间轴对齐到当前帧坐标下再比较，避免把"同一意图的不同起始观测"误判为不一致。
- **加权聚合**：$\bar{\mathbf{Q}}_{t,j}^{\mathrm{hist}} = \sum_k w_{t,k,j} \widehat{\mathbf{Q}}_{t,k,j}^{\mathrm{hist}}$，其中权重 $\phi_{t,k,j} = m_{t,k} \exp(-d_{t,k,j}/\alpha) \exp(-k/\beta)$ 同时编码有效性、轨迹相似度与时间衰减。
- **可靠性门控**：综合“有效历史数量”与“最小轨迹距离”，通过 $r_t^{\mathrm{cnt}}$ 与 $r_{t,j}^{\mathrm{dist}}$ 相乘形成 $\boldsymbol{r}_{t,j}$；当有效历史不足或轨迹偏离过大时，历史增强的残差将被抑制。
- **SSM 编码器**：将历史对齐查询与当前查询拼接为时序序列，经 LayerNorm、因果深度可分离卷积、输入依赖选择性扫描、SiLU 门控与残差连接，得到 $\mathbf{H}_{t,j}^{\mathrm{cur}}$；该结构利用选择性 SSM 的低复杂度优势同时保留候选语义。
- **残差注入与裁剪**：$\Delta \mathbf{Q}_{t,j} = \tanh(\mathbf{H}_{t,j}^{\mathrm{cur}} - \mathbf{Q}_{t,j})$，再通过全局/候选级门控与范数约束 $\mathbf{U}_{t,j} = \mathrm{Clip}(g_{t,j}\Delta \mathbf{Q}_{t,j},\; \rho\|\mathbf{Q}_{t,j}\|_2)$ 得到增强查询 $\widetilde{\mathbf{Q}}_{t,j}$，避免对原规划器的过度扰动。
- **Flow-Matching 残差修正**：将 SSM 增强后的规划轨迹转换为绝对空间并进行归一化得到初态 $\mathbf{x}_0$，专家轨迹对应终态 $\mathbf{x}_1$；在随机 $\tau \sim \mathcal{U}(0,1)$ 处构造线性插值 $\mathbf{x}_\tau = (1-\tau)\mathbf{x}_0 + \tau \mathbf{x}_1$，训练目标为 $\mathcal{L}_{\mathrm{FM}} = \|\mathbf{v}_\theta(\widetilde{\mathbf{Q}}_t, \mathbf{x}_\tau, \tau) - (\mathbf{x}_1 - \mathbf{x}_0)\|_2^2$。
- **推理与门控步数**：推理时仅用 Euler 积分进行 $S=2$ 步更新，并通过可学习有界残差门 $\eta = \eta_{\min} + \eta_{\max}\sigma(g_{\mathrm{flow}})$ 控制修正强度，最终输出 $\overline{\mathbf{Y}}_t^{\mathrm{flow}} = s[\mathbf{x}_0 + \eta(\mathbf{x}_S - \mathbf{x}_0)]$ 并还原为位移形式。
- **训练目标**：总损失 $\mathcal{L} = \mathcal{L}_{\mathrm{plan}} + \mathcal{L}_{\mathrm{SSM}} + \lambda_{\mathrm{flow}} \mathcal{L}_{\mathrm{flow}} + \lambda_{\mathrm{FM}} \mathcal{L}_{\mathrm{FM}}$，其中 $\mathcal{L}_{\mathrm{flow}}$ 同时在位移形式与累积形式上计算回归损失以兼顾局部精度与整体轨迹一致性。

## 实验与结果
- **开放环 nuScenes（6 秒）**：MomADv2 在 L2 与碰撞率上全面最优；平均 L2 误差为 1.21 m，平均碰撞率为 0.76%，相较最强前作平均 L2 提升 11.7%、碰撞率提升 11.6%；相对 MomAD 在 6 秒平均碰撞率上降低 15.6%。
- **Bench2Drive 闭环**：使用专家特征蒸馏时 MomADv2 达到 DS=78.82、Comf=31.23，并优于 MomAD（DS 由 47.91 提升至 52.32，SR 由 18.11% 提升至 24.24%）；在多能力评估中也达到较好均值。
- **NAVSIMv1 navtest**：取得 NC=99.0、DAC=97.1、TTC=95.5、EP=83.2、PDMS=89.9，整体安全、可行、进度与舒适度兼顾。
- **NAVSIMv2 navhard/navtest**：navhard 两阶段评估下 EPDMS 达 39.5，navtest 下 EPDMS 达 87.9 并在 DDC、EP、EC 等多项指标上位列第一，体现长时域交互下的稳定表现。
- **消融结论**：
  - SSM-Q 使 PDMS 从 84.0 提升至 86.9，FM-Ref 进一步至 89.9，二者增益互补。
  - 完全选择性记忆策略碰撞率最低（0.76%），明显优于无记忆、朴素融合与仅单维度过滤的变体。
  - 历史窗口长度 $K=4$ 在 NAVSIMv1 上取得最佳 PDMS=89.9，过长反而因噪声累积下降。
  - 时序建模上，Selective SSM/Mamba 在精度与碰撞率上优于 Concat+MLP、GRU/LSTM、Transformer 与 Vanilla SSM。
  - 求解器方面 Euler 优于 Heun，且在 FPS 上也有优势。
  - 轨迹预测一致性（TPC）在 4s/5s 分别较 MomAD 提升 12.6%/9.0%。

## 相关工作脉络
- MomAD（Song et al. 2025）：开创性利用历史规划查询维持动量，但未引入指令/时序/轨迹一致性过滤， MomADv2 在此基础上增加可靠性选择机制。
- Mamba/Selective SSM（Gu & Dao 2024；Dao & Gu 2024）：提供线性复杂度序列建模基础；本文将其适配到“带门控的历史查询时间演化”场景，而非直接用于视觉/语言表征。
- 扩散/生成规划（DiffusionDrive、DIVER、GuideFlow、GoalFlow）：这类方法从噪声出发生成轨迹；FM-Ref 与之本质不同，以可靠锚点规划为起点学习残差流，强调“稳定+微调”。
- VAD/UniAD/SparseDrive 等统一 E2E 框架：侧重于感知-规划联合表示与解码效率；本文聚焦其长时域时序记忆的可靠性缺陷，属于“记忆可靠性与校正”方向而非架构替换。
- DriveTrans/Drama/DriveDP 等时序或任务中心建模：强调高效时序或任务一致性，但仍未系统解决“历史何时应被丢弃”的问题；本文通过可微门控与显式索引匹配回答该问题。
- NAVSIM/Bench2Drive/nuScenes 评测体系：本文在这三类开放环与闭环基准上统一验证，说明可靠记忆的价值不仅体现在精度，更体现在安全与长期稳定性。

## 局限性与未来方向
- 依赖预定义高层驾驶指令（左/直/右）进行记忆筛选，无法覆盖复杂或隐式意图切换场景。
- 未显式建模未来场景演化的不确定性，记忆选择仍偏确定性规则。
- 当前仅做 2 步 Euler 积分，更复杂的流匹配求解或多步细化仍有提升空间。
- 记忆窗口较长时存在性能退化，说明极端长时历史仍需谨慎设计。
- 未来方向包括：指令无关的意图推断、不确定性感知记忆选择、以及与 world model 结合的远期推理。

## 研究启发与可借鉴点
- **“先读后写 + 仅缓存原始信号”**的记忆策略可迁移到各类时序决策系统，避免纠错结果在递归写入中被二次放大。
- 用轨迹级对齐替代固定索引匹配，能有效化解跨帧规划起点不同时造成的误判，这一思路可推广到多模态预测一致性评估。
- Flow-Matching 的残差化使用方式（以可靠基线为起点而非从噪声生成）是一种兼顾稳定性与精细修正的通用范式，适用于需要“小幅度高可靠修正”的下游任务。
- 选择性 SSM 在安全敏感场景中的表现优于普通 RNN/Transformer，提示在需要低延迟与长程依赖兼顾的系统中可优先考虑选择性状态空间结构。
- 将“可靠性门控”从单一维度扩展为指令/时序/轨迹/缓冲区多维联合条件，可作为时序记忆系统的通用设计规范。

## 关键术语表
- **端到端自动驾驶（E2E-AD）**：将感知、预测与规划统一到可微框架中，直接由传感器输入学习驾驶输出。
- **Selecttive State-Space Planning Memory Query（SSM-Q）**：按指令与轨迹一致性筛选历史规划查询，并用选择性状态空间建模意图时序演化的模块。
- **Flow-Matching Trajectory Residual Refiner（FM-Ref）**：以规划候选为起点学习朝向专家轨迹的连续残差速度场，实现局部精细化修正的模块。
- **Trajectory Prediction Consistency（TPC）**：衡量相邻时刻规划轨迹稳定性的指标，用于评估时序漂移程度。
- **Predictive Driver Model Score（PDMS）**：NAVSIM 采用的综合评分，融合安全、可行、进度与舒适度等软/硬约束。
- **Extended PDMS（EPDMS）**：NAVSIM-v2 的扩展评分，增加车道保持、信号灯遵守与历史/扩展舒适度等规则项。
- **Selective SSM / Mamba**：引入输入依赖选择性扫描的状态空间模型，能在长序列建模中动态决定信息的保留与遗忘。
- **Anchor-based planning**：以锚点轨迹为初始候选的规划范式，强调在稳定先验基础上做细粒度优化。

## 可复现要素
- **数据集与公开情况**：nuScenes（公开）、Bench2Drive（基于 CARLA 评估协议）、NAVSIM v1/v2（公开基准）。论文声明在三个基准上评估。
- **代码与权重**：论文正文与附录未明确给出仓库链接与开源声明，代码/权重情况请以下载源或作者主页为准。
- **关键超参**：记忆长度 $K=4$、Euler 积分步数 $S=2$、第二阶段训练 epoch 与学习率（nuScenes 第二阶为 10 epoch、lr=$3\times10^{-6}$）、 backbone 使用 ResNet-50、输入分辨率 256×704、检测范围 55 m、地图范围 60×30 m、运动模式数 6；Bench2Drive 使用 640×352 输入与 480 个规划查询；NAVSIM 使用 ResNet-34 backbone。具体平衡权重 $\lambda_{\mathrm{flow}}$、$\lambda_{\mathrm{FM}}$、$\alpha$、$\beta$、$\gamma_d$、$K_0$、$d_0$、$\rho$ 等在论文正文中以公式形式给出但未全部列出数值，需参考实现细节或附录补充材料。
