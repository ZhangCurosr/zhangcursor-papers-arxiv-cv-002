---
title: "S-sup-3-sup-AM-A-Single-Stream-SAM-with-Reliability-Calibrat"
source: https://arxiv.org/pdf/2608.17475v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 01:01:21"
field: "多模态视觉感知"
keywords: ["multi-modal salient object detection", "Segment Anything Model", "frequency adaptation", "reliability calibration", "single-stream architecture", "stationary wavelet transform"]
innovations: ["提出单流SAM适配框架，通过可靠性校准频率适配器控制辅助模态高频信息的早期注入", "设计频率专家混合模块(MoFE)与双门控校准机制(RCFA)，实现跨模态频域信息的自适应融合与噪声抑制"]
benchmarks: ["NJU2K", "NLPR", "SSD", "STERE", "VT5000", "VT821", "VT1000", "RGB-NIR cross-modal"]
---

# 论文速读：S³AM: A Single-Stream SAM with Reliability-Calibrated Frequency Adapter for Multi-modal Salient Object Detection

## 一句话总结
本文提出 S³AM，一种基于 Segment Anything Model (SAM) 的单流多模态显著性检测框架，通过引入可靠性校准频率适配器（RCFA）在早期融合阶段对辅助模态的高频信息进行选择性注入，在保证跨模态交互的同时避免基础模型骨干网的重复计算。

## 研究问题与动机
- **现有SAM适配方法的冗余问题**：当前基于 SAM 的 MSOD 方法多采用双流编码器或额外 prompt 生成网络，导致基础模型骨干网被重复使用，计算开销大，难以满足实时或资源受限场景需求。
- **单流早期融合的不确定性传播风险**：单流设计虽能减少架构冗余，但早期跨模态融合一旦将噪声或空间错位的高频信息注入骨干网，错误会在整个 Transformer 链式传播，尤其是辅助模态（深度图空洞、热成像模糊、NIR 纹理偏差）的高频响应可靠性不均。
- **多模态数据规模受限**：多模态对齐数据的获取成本高、标注耗时，限制了从头训练专用架构的可能性，亟需利用 SAM 等基础模型的零样本泛化能力。
- **高频信息的互补性与不一致性并存**：不同模态的高频响应在边界位置存在本地冲突（如 Fig.3 所示青/品红共存区域），直接融合会引入错误轮廓。

## 核心贡献（创新点）
1. **提出参数高效的单流 SAM 适配框架**：避免重复基础模型骨干网，同时通过可靠性校准频率适配机制控制噪声辅助高频注入，在降低架构冗余的同时保留跨模态交互能力。
2. **设计频率专家混合模块（MoFE）**：基于 stationary wavelet transform（SWT）对跨模态特征进行多频率分解，并通过图像条件路由聚合四类交叉频率对，为统一编码器提供丰富的结构先验。
3. **提出可靠性校准频率适配器（RCFA）**：通过双门控校准机制（上下文门控 + 可靠性校准门控）在 Transformer 各阶段选择性传播可靠的高频残差，抑制不可靠的辅助细节污染后续特征。
4. **构建超网络引导的语义-结构解码器（HSSD）**：联合 SAM 骨干网的超网络语义掩码先验与基于 Mamba 的结构细节恢复路径，在联合显著性-边缘监督下实现语义完整性与边界精度的平衡。

## 方法详解

### 整体架构
给定 RGB 和辅助模态（深度/热成像/NIR）配对输入，特征首先经共享 patch embedding 提取，再通过 MoFE 进行 SWT 频率分解与跨模态专家聚合得到 $\mathcal{F}^{\text{fused}}$，与 RGB 特征相加后进入冻结的 SAM 骨干网；RCFA 在各 Transformer 阶段逐级校准并传播频率先验；最终由 HSSD 输出保结构的显著性图。

### MoFE（Mixture of Frequency Experts）
- 对 RGB 和辅助模态特征 $\mathcal{F}^{\text{rgb}}, \mathcal{F}^{\text{aux}}$ 独立施加 SWT 分解，得到低频 $\mathcal{C}_L$ 和三个高频子带 $\mathcal{C}_{LH}, \mathcal{C}_{HL}, \mathcal{C}_{HH}$。
- 将三个方向高频拼接为 $\mathcal{C}_H^m$，构造四类交叉频率对：
  - $\mathcal{C}_1 = \text{Concat}(\mathcal{C}_L^{\text{rgb}}, \mathcal{C}_L^{\text{aux}})$：低频-低频粗结构
  - $\mathcal{C}_2 = \text{Concat}(\mathcal{C}_L^{\text{rgb}}, \mathcal{C}_H^{\text{aux}})$：RGB低频+辅助高频
  - $\mathcal{C}_3 = \text{Concat}(\mathcal{C}_H^{\text{rgb}}, \mathcal{C}_L^{\text{aux}})$：RGB高频+辅助低频
  - $\mathcal{C}_4 = \text{Concat}(\mathcal{C}_H^{\text{rgb}}, \mathcal{C}_H^{\text{aux}})$：联合高频精细边界
- 每对通过独立专家 $E_n(\cdot)$ 增强，由轻量级 Router 学习路由权重 $\alpha$，加权融合得到 $\mathcal{F}^{\text{fused}} = \sum \alpha_n \widehat{\mathcal{C}}_n$。
- 初始频率先验 $\mathcal{P}_0 = \alpha_4 \widehat{\mathcal{C}}_4$ 送入后续 RCFA 序列。

### RCFA（Reliability-Calibrated Frequency Adapter）
在每个 Transformer 阶段 $i$，RCFA 接收编码器特征 $\mathcal{F}_i$ 和传入频率先验 $\mathcal{P}_{i-1}$：
- **高频残差提取**：通过 3×3 平均池化低通滤波减去原始特征得到高频响应 $\mathcal{R}_i^{\text{hf}}$ 和 $\mathcal{D}_i^{\text{hf}}$。
- **双瓶颈适配器**：分别通过轻量级 $1\times1$ 卷积瓶颈产生基础残差 $B_i^{\text{res}}$ 和频率残差 $\mathcal{Q}_i^{\text{res}}$。
- **可靠性校准门控（RCG）**：计算三个有界可靠性提示：
  - 一致性 $\rho^{\text{con}} = \frac{1}{2}(1 + \text{CosSim}_c(\mathcal{R}_i^{\text{hf}}, \mathcal{D}_i^{\text{hf}}))$
  - 能量平衡 $\rho^{\text{bal}} = 1 - \frac{|\mathcal{E}^{\text{enc}} - \mathcal{E}^{\text{prior}}|}{\mathcal{E}^{\text{enc}} + \mathcal{E}^{\text{prior}} + \epsilon}$
  - 差异抑制 $\rho^{\text{diff}} = \exp\left(-\frac{\text{Mean}_c(|\mathcal{R}_i^{\text{hf}} - \mathcal{D}_i^{\text{hf}}|)}{\mathcal{E}^{\text{enc}} + \mathcal{E}^{\text{prior}} + \epsilon}\right)$
- **上下文门控**：$\psi(\mathcal{F}_i + \mathcal{P}_{i-1})$ 评估整体注入强度。
- **最终残差**：$\mathcal{F}_i^{\text{res}} = B_i^{\text{res}} + \gamma_i \cdot \psi(\cdot) \odot \mathcal{G}_i^{\text{rel}} \odot \mathcal{Q}_i^{\text{res}}$，其中 $\gamma_i$ 初始化为零，逐步激活。
- 校准残差注入 Transformer 块后作为下一阶段的频率先验 $\mathcal{P}_i$ 继续传播。

### HSSD（Hypernetwork-guided Semantic-Structural Decoder）
- 收集四个阶段的编码器特征，经 FPN-style neck 对齐后送入两条解码路径：
  - **语义路径**：沿用 SAM mask decoder 的超网络语义先验 $\mathcal{U}_{\text{SAM}}$。
  - **结构路径**：轻量级 Mamba 解码器（VSS block）逐步恢复边界细节 $\mathcal{U}_{\text{Mamba}}$。
- **自适应融合门控**：$\mathcal{G}^{\text{dec}} = \sigma(\varphi([\mathcal{U}_{\text{SAM}}, \mathcal{U}_{\text{Mamba}}]))$，$\mathcal{U}_{\text{fuse}} = \mathcal{G}^{\text{dec}} \odot \mathcal{U}_{\text{SAM}} + (1-\mathcal{G}^{\text{dec}}) \odot \mathcal{U}_{\text{Mamba}}$。
- 最终通过 SAM mask hypernetwork 输出显著性图 $S = \text{Sigmoid}(\mathcal{H}_{\text{mask}}(\mathcal{U}_{\text{fuse}}))$。

### 损失函数
- 语义分支：$\ell_{\text{mask}} = \ell_{\text{dice}} + \ell_{\text{iou}}$（Dice + IoU）
- 结构/边缘分支：$\ell_{\text{edge}} = \ell_{\text{dice}}$（对 Mamba 解码器的边缘预测监督）
- 总损失：$\mathcal{L} = \ell_{\text{mask}} + \ell_{\text{edge}}$

## 实验与结果

### 数据集与评估指标
- **RGB-D**：NJU2K（训练1,485）、NLPR（训练700）、SSD、STERE；指标：$F_\beta$、MAE、$E_m$、$S_m$
- **RGB-T**：VT5000（训练2,500对）、VT821、VT1000；指标同上
- **RGB-NIR**：跨模态零样本泛化测试（无微调）；指标：$F_{\text{avg}}$、$F_{\text{max}}$、$F_\beta$、MAE、$E_m$、$S_m$

### 主要结果
| 任务 | 数据集 | 最强提升 | 对比基线 |
|------|--------|----------|----------|
| RGB-D | NJU2K | $F_\beta$ 0.939 / MAE 0.019 | KAN-SAM ($F_\beta$ 0.935) |
| RGB-D | NLPR | $F_\beta$ 0.934 / MAE 0.014 | KAN-SAM ($F_\beta$ 0.925) |
| RGB-T | VT5000 | $F_\beta$ 0.898 / MAE 0.020 | CMDBIF ($F_\beta$ 0.846)，提升 +4.5% |
| RGB-T | VT1000 | $F_\beta$ 0.934 / MAE 0.013 | CMDBIF ($F_\beta$ 0.909)，提升 +2.4% |
| RGB-T | VT821 | $F_\beta$ 0.874 / MAE 0.025 | CMDBIF ($F_\beta$ 0.837)，提升 +3.3% |
| RGB-NIR | 跨模态零样本 | $F_\beta$ 0.921 / MAE 0.020 | SOD-8S+ ($F_\beta$ 0.745)，提升 +23.6% |

### 参数效率
- 总参数：224.36M，**可训练参数仅 12.20M（占 5.4%）**
- FLOPs：428.1G，RTX 5090 上 **45.8 FPS**；NVIDIA Orin 边缘设备约 **5 FPS**
- 对比 KAN-SAM（643.588M / 1824G FLOPs / 4 FPS）：约 2.9× 参数和 4.3× FLOPs 的开销

### 消融验证
- MoFE 引入使 NJU2K $F_\beta$ 从 0.919 提升至 0.931
- 加 RCFA 进一步提升至 0.932；全模型达 0.939
- 双门控分析：移除 Context Gate 或 RCG 均导致性能下降，NLPR 上下降更明显，说明双门控在辅助信息不稳定时提供互补约束
- 融合策略：早期融合（223.8M 参数 / 420.1G FLOPs）优于中间融合（438.7M / 816.7G）和后期融合，性价比最优

## 相关工作脉络
- **双流 SAM 适配（如 KAN-SAM、SAMSOD）**：采用辅助 prompt 生成器或 LoRA 模块注入任务知识，依赖重复骨干网或额外优化阶段，S³AM 通过单流+频率适配消除冗余。
- **SAM-Adapter / MedSAM / SAMRS**：通过轻量适配器将 SAM 适配至特定领域，但未针对多模态高频不一致性设计校准机制，S³AM 补充了跨模态频率可靠性控制。
- **HyPSAM / SSFam**：prompt-driven 方法通过文本/掩码/框提示引导 SAM，需要动态推理管线；S³AM 在早期融合阶段即完成跨模态交互，推理流程更简洁。
- **传统 MSOD（BTNet、CAVER、LESOD）**：双流/三流设计擅长模态对齐但计算昂贵；S³AM 在相近性能下大幅降低参数和 FLOPs。
- **扩散生成式 MSOD（DiMSOD）**：将 MSOD 视为条件掩码生成问题，依赖迭代去噪，推理成本高；S³AM 采用单前向通路，适合实时部署。

## 局限性与未来方向
- **仅支持双模态输入**：当前框架针对 RGB + 单一辅助模态设计，对三模态（RGB-D-T 等）的直接扩展未涉及。
- **零样本泛化依赖训练分布**：RGB-T 到 RGB-NIR 的跨模态迁移虽表现优异，但辅助模态质量差异较大时（如严重失配的深度图）可靠性门控的有效性需进一步验证。
- **边缘设备推理速度仍有优化空间**：Orin 上仅 5 FPS，对于严格实时应用（如机器人导航）仍需进一步轻量化。
- **高频分解的层数限制**：仅做一层 SWT 分解，多尺度频域信息的充分利用有待探索。

## 研究启发与可借鉴点
- **频率域跨模态融合的新范式**：将 SWT 引入多模态特征融合，以频域视角解耦结构化高频与语义低频，为其他多模态任务（分割、检测、配准）提供可迁移思路。
- **可靠性校准的门控设计**：双门控机制（注入强度控制 + 跨模态一致性评估）可推广至任何需要早期融合但存在模态噪声的场景，如遥感多光谱融合、医学多模态影像分析。
- **单流适配基础模型的高效策略**：冻结骨干 + 轻量适配模块 + 阶段级残差注入的组合，在保持基础模型泛化能力的同时实现极低参数量，适用于资源受限部署。
- **超网络+Mamba 的语义-结构解耦解码**：将语义掩码预测与结构细节恢复分离，通过自适应门控融合，为需要精细边界的分割任务提供了新的解码器设计参考。
- **联合Dice+IoU损失的多目标优化**：语义分支用 Dice+IoU、结构分支用纯 Dice 的分工策略，体现了针对不同任务特性定制损失函数的设计思想。

## 关键术语表
- **Multi-modal Salient Object Detection (MSOD)**：融合多种传感器模态（RGB、深度、热成像等）的显著性目标检测任务，提升复杂环境下的鲁棒性。
- **Stationary Wavelet Transform (SWT)**：平稳小波变换，无需下采样即可对图像进行多频段分解，保留空间位置信息，适合提取结构细节。
- **Reliability-Calibrated Frequency Adapter (RCFA)**：可靠性校准频率适配器，通过双门控机制在 Transformer 各阶段选择性传播高频残差，抑制不可靠的跨模态高频噪声。
- **Mixture of Frequency Experts (MoFE)**：频率专家混合模块，将四种交叉频率对（低频-低频、低频-高频等）送入独立专家网络，通过路由权重自适应聚合。
- **Reliability-Calibrated Gate (RCG)**：可靠性校准门控，基于高频一致性、能量平衡和差异抑制三个提示计算调制权重，评估跨模态高频响应的可信度。
- **Hypernetwork-guided Semantic-Structural Decoder (HSSD)**：超网络引导的语义-结构解码器，联合 SAM 超网络语义先验与 Mamba 结构恢复，实现完整掩码与精细边界的协同预测。
- **Context Gate**：上下文门控，通过全局平均池化和 1×1 卷积评估当前特征与频率先验的整体注入强度，与 RCG 形成双门控体系。
- **Zero-shot Cross-modal Generalization**：零样本跨模态泛化，将在某一模态组合（RGB-T）上训练的模型直接应用于未见模态（RGB-NIR）的测试。

## 可复现要素
- **数据集**：NJU2K、NLPR、SSD、STERE（RGB-D）；VT5000、VT821、VT1000（RGB-T）；RGB-NIR 基准（公开）；部分训练集划分遵循文献协议
- **代码**：论文声明代码将在 https://github.com/xuboyue1999/SSSAM 开源（截至论文发表时）
- **权重**：SAM 骨干网使用官方发布 checkpoint [5]（Frozen）
- **关键超参**：输入分辨率 512×512；训练 50 epochs；batch size 8；AdamW，lr=1e-4，weight decay=5e-4，betas=(0.9, 0.999)；梯度裁剪阈值 0.5
