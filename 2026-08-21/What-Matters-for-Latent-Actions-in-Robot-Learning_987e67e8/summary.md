---
title: "What-Matters-for-Latent-Actions-in-Robot-Learning"
source: https://arxiv.org/pdf/2608.19613v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 19:26:44"
field: "具身智能与机器人操作学习"
keywords: ["Latent Action Models", "Representation Learning", "Robot Learning", "VLM Fine-tuning", "Embodied AI", "Self-supervised Learning"]
innovations: ["首次在统一框架下系统研究LAM的41种设计选择，揭示建模范式、正则化与整合策略的关键影响", "验证FDM重建指标为最可靠的潜在动作质量代理，适用于粗粒度模型筛选", "确立潜在动作应参与下游联合优化（JAP）的设计原则，并发现scaling law"]
benchmarks: ["LIBERO", "LIBERO-Plus", "RoboTwin2.0"]
---

# 论文速读：What-Matters-for-Latent-Actions-in-Robot-Learning

## 一句话总结
本文对机器人学习中潜在动作模型（LAM）进行了首次系统性实证研究，在统一框架下对比了41种设计选择（建模范式、正则化方法、整合策略），并验证了"通过潜在动作微调VLM骨干网络可显著提升下游机器人操作性能"的核心结论。

## 研究问题与动机
- **数据瓶颈**：大规模无标签视频 abundant，但机器人动作数据稀缺且收集成本高昂，制约了具身基础模型的可扩展性。
- **现有方法碎片化**：LAM各工作孤立提出不同设计选择，在不一致的实验设置下评估，难以识别真正影响下游操作性能的关键因素。
- **评估协议缺失**：缺乏可靠方法来评估潜在动作质量；下游性能需要完整三阶段训练管线，迭代成本极高；现有代理指标（MLP探针）的有效性未经验证。
- **VLM骨干微调机制不明**：潜在动作与物理动作整合策略、维度选择、正则化超参数等设计对下游策略学习的实际影响尚不清楚。

## 核心贡献（创新点）
1. **首次系统化LAM设计空间探索**：在LIBERO、LIBERO-Plus和RoboTwin2.0三个基准上，围绕3个维度系统研究41种设计选择，填补了LAM领域缺乏统一实证理解的空缺。
2. **揭示LAPO原始方法作为强基线的价值**：发现LAPO直接在原始数据上训练仍是最强基线，且使用预训练视觉编码器（DINOv2）的简单语义特征差分即可达到竞争性性能。
3. **提出可靠的代理指标评估体系**：验证了FDM重建指标（SSIM Gain、MSE Gain）比探针指标更能可靠预测下游性能，为粗粒度模型筛选提供实用工具。
4. **确立"潜在动作参与下游联合优化"的设计原则**：证明JAP（联合潜在-动作预测）架构优于仅将潜在动作用于预训练的DAP/LAP，潜在动作应在下游策略学习中保持活跃。
5. **验证VLM骨干微调的scaling law**：在59M帧视频数据上，扩大Stage II微调数据规模可一致性地提升下游性能，LIBERO-Plus上最大增益达9.0%。

## 方法详解
**三阶段训练管线**：
- **Stage I（预训练）**：从连续帧$(o_t, o_{t+1})$学习高质量LAM，无需机器人动作数据，训练IDM（逆动力学模型）和FDM（正动力学模型）。
- **Stage II（中期微调）**：用预训练IDM作为潜在动作标注器，为语义条件视频数据生成三元组$(o_t, z_t, l)$，微调VLM骨干网络（Qwen3-VL-4B）。
- **Stage III（后训练/策略学习）**：使用有限机器人动作数据$(o_t, a_t, l)$对微调后的VLM骨干进行下游策略优化。

**建模范式（Design I）**：
- **IDM-FDM框架**（隐式）：LAPO核心公式 $z_t = \text{IDM}(o_t, o_{t+1}), \hat{o}_{t+1} = \text{FDM}(o_t, z_t), \mathcal{L}_{\text{LAPO}} = \|o_{t+1} - \hat{o}_{t+1}\|_2^2$；LAOF加入光流辅助损失$\mathcal{L}_{\text{flow}}$；CoMo用$\Delta_t$替换未来帧输入以缓解因果泄漏。
- **CFD-AE框架**（显式）：$\Delta_t$可以是$\Delta_{\text{RGB}}$、$\Delta_{\text{DINO}}$（DINOv2特征空间差分）或光流（RAFT/SEA-RAFT），$z_t = \text{Encoder}(\Delta_t), \hat{\Delta}_t = \text{Decoder}(z_t)$。

**正则化方法（Design II）**：
- VQ-VAE：向量量化正则化，强制潜在动作映射到有限码本嵌入集合。
- VAE：KL散度正则化$\mathcal{L}_{\text{reg}}^{\text{VAE}} = D_{\text{KL}}(p(z_t|o_t,o_{t+1})||\mathcal{N}(0,I))$，约束分布为标准高斯。
- Sparsity：稀疏正则化，结合$\ell_1$、$\ell_2$约束和VCM（方差-协方差-均值）正则化防止表示坍缩。
- SIGReg：各向同性高斯正则化，通过Cramér-Wold定理验证多随机方向投影匹配标准高斯分布。

**整合策略（Design III）**：
- **DAP**（直接动作预测）：$\hat{a}_t = \text{ActionHead}_a(h_t)$，潜在动作仅用于Stage II骨干微调。
- **LAP**（潜在到动作）：$\hat{z}_t = \text{ActionHead}_z(h_t), \hat{a}_t = \text{ActionHead}_a(\hat{z}_t)$，潜在动作为中间控制表示。
- **JAP**（联合潜在-动作预测）：$[\hat{z}_t, \hat{a}_t] = \text{ActionHead}_{z,a}(h_t)$，并行优化两个分支。

总体目标函数：$\mathcal{L} = \mathcal{L}_{\text{method}} + \lambda_{\text{reg}} \mathcal{L}_{\text{reg}}$。

## 实验与结果
**数据集**：Stage I/II使用约59M帧视频混合（含Open X-Embodiment子集：DROID、BC-Z、Furniture Bench等），Stage III使用LIBERO/LIBERO-Plus/RoboTwin2.0基准及Franka Panda真实机器人数据（4任务×50演示）。

**评估基线**：LAPO、LAOF、CoMo、ΔRGB、ΔDINO、RAFT、SEA-RAFT；正则化方法包括AE、VQ-VAE、VAE、Sparsity、SIGReg；5种整合策略DAP/LAP/JAP/JAP-DAP/JAP-LAP。

**关键结果**：
- **最佳建模范式**：LAPO综合平均得分最高，ΔDINO接近LAPO且LIBERO上超越（平均0.921 vs LAPO 0.910）；光流方法（RAFT/SEA-RAFT）性能最低，LAOF添加光流反而退化。
- **最佳正则化**：VAE ($\lambda_{\text{reg}}=10^{-7}$)、Sparsity ($10^{-5}$)、SIGReg ($10^{-3}$)三者下游性能相当；VQ-VAE在LIBERO-Plus零样本泛化上最优（0.517）。
- **最佳维度**：$d_z=32$在单臂和双臂平台均取得最佳平衡；LIBERO上$d_z=16$略优（0.926），RoboTwin2.0上$d_z=32$最优（0.856）。
- **最强结果**：JAP整合策略在多数设定下最优（RoboTwin2.0 JAP达0.851）；JAP-LAP在物理动作可访问设定下最强（LIBERO 0.900，RoboTwin2.0 0.856）。
- **真实机器人实验**：OpenVLA-OFT (LA-Tuned)相比基线整体成功率从64.75%提升至79.25%（+14.5pp，相对+22.4%）；10k步即超越基线40k步性能。
- **Scaling Law**：Stage II微调数据从14.5%扩至100%，LIBERO-Plus最大增益9.0%。

## 相关工作脉络
- **LAPO [28]**：原始LAM方法，IDM-FDM自监督自编码框架；本文在统一设置下验证其作为强基线的有效性。
- **LAOF [33] / CoMo [34]**：分别引入光流辅助监督和语义特征差分缓解因果泄漏；本文实验表明这些改进在机器人操作场景下未必有效。
- **StaMo [29] / Motus [10]**：分别使用DINOv2语义差分和光流作为显式运动信号；本文统一到CFD-AE框架下公平比较，证明ΔDINO可与隐式方法竞争。
- **VQ-VAE [46] / VAE [55]**：离散与连续正则化代表性方法；本文系统比较多种正则化，发现类型影响有限而超参数调节更为关键。
- **Sparsity [37] / SIGReg [58]**：新兴正则化方法；本文验证其有效范围并推荐基于任务需求的选用策略（简单性选VAE，泛化选VQ-VAE）。
- **LeWorldModel [36] / DreamDojo [38]**：世界模型方向相关工作；本文聚焦LAM对VLA骨干微调的影响，与WAM研究形成互补视角。

## 局限性与未来方向
- 潜在动作目前仅作为任务特定的微调监督信号，未来可将其提升为类似文本嵌入的基础级动作表示，融入VLM预训练阶段。
- 训练数据依赖现有开源机器人数据集，计划利用YouTube/Bilibili等网络视频扩展数据多样性和规模。
- 实验仅限于机械臂操作基准，需验证结论在灵巧手、四足机器人和人形机器人等更广泛平台上的泛化性。
- 代理指标不适用于跨维度的细粒度模型排名，仅适合同维度下的粗粒度筛选。

## 研究启发与可借鉴点
1. **"简单即有效"原则**：预训练视觉编码器（DINOv2）的语义特征差分可在不增加模型复杂度的前提下达到接近隐式IDM-FDM方法的性能，值得在资源受限场景优先尝试。
2. **正则化超参数敏感性远高于方法选择**：建议研究者投入更多精力进行正则化强度搜索（如VAE取$10^{-7}$、Sparsity取$10^{-5}$），而非频繁更换正则化方法。
3. **JAP架构的设计哲学**：潜在动作不应仅作为预训练目标，而应作为持续的正则化辅助任务参与下游联合优化；这一思路可迁移至其他多任务预训练+微调管线。
4. **代理指标的合理定位**：FDM重建指标（SSIM/MSE Gain）可作为快速模型选择的代理，但需在同一维度下使用，避免跨维度误判。
5. **Scaling Law验证范式**：保持Stage III不变、仅扩大Stage II数据规模的实验设计，为评估表征学习方法的扩展性提供了干净可靠的评估范式。

## 关键术语表
**Latent Action Models (LAMs)**：从连续视频帧中学习紧凑潜在动作表示的自监督框架，作为物理动作的代理以利用大规模无标签视频。
**Inverse Dynamics Model (IDM)**：编码器，从连续帧推断潜在动作，捕捉状态转移动力学。
**Forward Dynamics Model (FDM)**：解码器，以当前帧和潜在动作为条件预测下一帧，验证潜在动作的信息保真度。
**CFD-AE (Consecutive-Frame Differences Autoencoding)**：显式运动信号框架，通过自编码连续帧差分（像素差、语义特征差或光流）学习潜在动作。
**JAP (Joint Latent-Action Prediction)**：整合策略，从共享高层表征同时预测潜在动作和物理动作，保持潜在动作在下游训练中的持续正则化作用。
**Proxy Metrics**：用于评估潜在动作质量的代理指标，包括探针损失（Linear/MLP Probe）和FDM重建增益（SSIM/MSE Gain）。
**VLM Backbone**：视觉-语言模型骨干网络（如Qwen3-VL-4B），在Stage II中接受潜在动作监督微调，形成具身基础模型。
**Causal Leakage**：IDM在自监督训练中可直接访问目标未来帧导致的信息泄漏问题，可能引发捷径学习。

## 可复现要素
- **数据集**：Open X-Embodiment子集（DROID、BC-Z、Furniture Bench等，公开）、Robotwin（公开）、Liberoplus（公开）、LIBERO/LIBERO-Plus（公开）、RoboTwin2.0（公开）、Franka Panda真实数据（论文自建，50演示/任务）。
- **代码/权重**：项目页面 https://LAM.github.io（论文声明开源）；论文未明确说明GitHub仓库链接和预训练权重下载地址。
- **关键超参**：Stage I/II LR=$2.5\times10^{-5}$，Action Head LR=$1\times10^{-4}$，Qwen3-VL-4B LR=$1\times10^{-5}$，Batch Size=256，Steps=150k；VAE正则化强度$10^{-7}$，Sparsity $10^{-5}$，SIGReg $10^{-3}$，VQ-VAE $\beta=1$；$d_z=32$；图像分辨率224×224；LAM为700M时空Transformer（24层编码器+24层解码器）。
