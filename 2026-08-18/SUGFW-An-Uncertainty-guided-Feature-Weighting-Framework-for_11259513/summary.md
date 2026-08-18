---
title: "SUGFW-An-Uncertainty-guided-Feature-Weighting-Framework-for"
source: https://arxiv.org/pdf/2608.16110v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 15:21:46"
field: "医学图像分割中的主动学习"
keywords: ["Cold Start Active Learning", "Segment Anything Model", "Medical Image Segmentation", "Uncertainty Estimation", "Sample Selection", "Feature Weighting", "Prompted Fine-tuning"]
innovations: ["提出PFUC+PGDR结合SAM patch级特征与不确定性进行判别性全局特征表示", "设计GSCU策略同时优化特征空间代表性与不确定性空间多样性", "提出UPFT将surrogate uncertainty作为空间prompt实现极低预算下SAM自适应微调"]
benchmarks: ["Promise12", "UTAH", "MSD Liver", "ISIC 2018"]
---

# 论文速读：SUGFW-An-Uncertainty-guided-Feature-Weighting-Framework-for

## 一句话总结
论文提出 SUGFW+ 框架，利用 SAM 的 patch-level 特征与不确定性估计进行零标注冷启动样本选择，并通过不确定性引导的微调策略（UPFT）适配 SAM，在极低标注预算（0.05%–3%）下实现医学图像分割的状态最优性能。

## 研究问题与动机
- **冷启动主动学习（CSAL）** 要求在无任何标注的目标数据集上一次性选出最具信息量的样本子集进行标注，避免传统 AL 多轮查询的高成本和初始标注依赖。
- 现有 CSAL 方法多依赖数据集特定的 SSL 预训练提取特征，计算昂贵且泛化受限；而 SAM 等基础模型可提供强泛化特征嵌入，但其 patch-level 特征与不确定性如何系统用于 CSAL 尚未被探索。
- 在低标注预算下微调 SAM 等基础模型易发生严重过拟合，且 SAM 原本为交互式分割设计，如何将其转化为自动分割模型仍需有效策略。
- 医疗影像标注成本极高，如何在极致低预算（0.1%–3%）下同时保证样本选择质量和下游微调效果，是临床落地的重要挑战。

## 核心贡献（创新点）
1. **提出 SUGFW+ 统一框架**：将 SAM 用于样本选择与下游微调两个阶段，实现特征表示、不确定性估计与模型适配的一体化，相比前期 SUGFW（UNet 骨干）性能显著提升。
2. **PFUC + PGDR 不确定性感知特征表示**：利用 SAM "everything mode" 获取 patch-level 特征，并通过 K 次增强计算 surrogate uncertainty，再经不确定性加权融合得到判别性全局特征，避免传统全局平均池化的信息损失。
3. **GSCU 贪婪聚类不确定性感知的样本选择策略**：在特征空间聚类保证代表性的基础上，以贪心最大化已选样本间不确定性距离的方式选取各簇样本，兼顾多样性与不确定性覆盖。
4. **UPFT 不确定性提示微调机制**：用 surrogate uncertainty map 作为空间 prompt 替换原始 prompt encoder，引入轻量 uncertainty predictor 加速推理，结合 LoRA 在极低标注量下完成 SAM 向自动分割模型的适配。

## 方法详解
**整体流程**：对未标注训练集 D，先通过 SAM 的 PFUC 模块提取 patch-level 特征 F_i 和 surrogate uncertainty map U_i；再通过 PGDR 加权聚合为图像级特征 f_i 与不确定性分数 u_i；随后用 GSCU 策略选出 M 个查询样本；最后用 UPFT 在查询集上微调 SAM。

**PFUC 关键公式**：
- 图像编码器：$F_i = E_{img}(X_i)$
- "everything mode" 网格点提示生成 class-agnostic mask，合并小区域后得二值前景 mask $M_i$
- K 次增强后平均得 soft mask $\bar{M}_i$，下采样匹配特征图尺寸
- patch 级不确定性：$U_i(r) = -[\bar{M}_i(r)\log\bar{M}_i(r) + (1-\bar{M}_i(r))\log(1-\bar{M}_i(r))]$

**PGDR 不确定性加权聚合**：
- 全局特征：$f_i = \sum_r F_i(r) \cdot \frac{e^{\lambda \cdot U_i(r)}}{\sum_r e^{\lambda \cdot U_i(r)}}$，其中 $\lambda \geq 1$ 放大高不确定性区域贡献
- 图像级不确定性分数：$u_i = \frac{1}{H'\times W'}\sum_r U_i(r)$

**GSCU 样本选择**：
- K-means 聚类，簇数 C = 目标标注预算
- 首次随机从一簇选不确定性中位数的样本
- 后续每轮从新簇中选使最小不确定性距离最大化的样本：$X_{m+1} = \arg\max_{X_i \in C}(\min_{X_j \in \mathcal{D}^m}|u_i - u_j|)$

**UPFT 微调设计**：
- 将 SAM 原始 prompt encoder 替换为 uncertainty encoder $E_{un}$，输出 $F_i^u = E_{un}(U_i)$ 作为 decoder 中的 cross-attention query
- 引入轻量 transformer 不确定性预测器 $g(\cdot)$ 以 MSE 损失 $\mathcal{L}_u = \|\hat{F}_i^u - F_i^u\|_2^2$ 训练，推理时替代 costly 的 K 次前向
- 总损失：$\mathcal{L} = \omega \cdot \mathcal{L}_{Dice} + (1-\omega) \cdot \mathcal{L}_{CE} + \alpha_t \cdot \mathcal{L}_u$，其中 $\alpha_t$ 采用 ramp-up 策略
- LoRA（rank=32）加在冻结的 image encoder 上，更新 LoRA 层、decoder、uncertainty encoder 和 predictor

## 实验与结果
**数据集**：Promise12（前列腺 MRI）、UTAH（左心房 MRI）、MSD Liver（肝脏 CT）、ISIC 2018（皮肤镜图像）

**评估基线**：Random、ALPS、CALR、FPS、ProbCover、TypiClust、CEC，均在相同 SAM ViT-Base 编码器上公平比较

**关键结果**：
- Promise12：3% 预算下 DSC 86.38%（最优），8% 预算下 89.25%，超 FPS（84.24%）约 2.1pp；HD95 在 5%–8% 预算下最优且显著
- UTAH：0.10% 预算下 DSC 78.70%，0.75% 预算下 89.44%，HD95 在 0.75% 时达 7.30mm（最优）
- MSD Liver：0.05% 超低预算下 DSC 82.27%，显著超 TypiClust（77.07%）；0.30% 时 93.11%，HD95 稳定下降
- ISIC 2018：全预算范围 DSC 和 HD95 均最优，3.0% 预算达 89.75%
- **SUGFW+ vs 全监督 UNet**：在 Promise12 3%、UTAH 0.1%、ISIC 0.5% 预算下即超过全监督 UNet 性能
- **SUGFW+ vs 前期 SUGFW（UNet 骨干）**：Promise12 平均 DSC 提升约 35pp

**超参**：$\lambda = 6$，$\alpha_{max} = 0.02$（消融确定），K=10 次增强，LoRA rank=32，300 epochs，lr=8e-4

## 相关工作脉络
- **传统 AL 方法**（Entropy、Margin、Core-set）依赖初始标注模型和多轮查询，无法直接应用于 CSAL 单轮场景
- **早期 CSAL 方法**（ALPS、CALR、FPS、TypiClust、ProbCover）依赖 SSL 特征提取+聚类/覆盖策略，缺乏不确定性引导且需数据集特定预训练
- **CEC**（2025）利用 calibrated entropy 和 neighbor-aware uncertainty，但仍仅关注样本选择，未将不确定性融入模型微调阶段
- **SAM 适配工作**（SAM-Med2D、MedSAM、MA-SAM）聚焦全监督微调，需要大量标注；本文在极低预算下实现 SAM 自动分割适配
- **DINO-based 方法**（DINOUNet）使用冻结编码器+U-Net 解码器，但缺少任务特定的不确定性引导机制

## 局限性与未来方向
- **计算效率**：PFUC 依赖 K=10 次增强 + "everything mode" 网格点提示，在大规模 petabyte 级未标注库上可能成为瓶颈，需更高效的不确定性估计方法
- **2D 限制**：当前仅支持 2D 切片级标注与训练，因 SAM 为 2D 架构；未来可扩展至 3D SAM 变体实现真正的 3D 推理
- **Surrogate uncertainty 偏差**：SAM 的 class-agnostic 输出仅提供不确定性代理，与目标任务真实不确定性存在差距，可能影响极端低预算下的选择质量

## 研究启发与可借鉴点
- **不确定性作为空间 prompt**：将模型预测的不确定性图作为空间引导信号注入解码器，可替代人工 prompt，为其他视觉任务的自动化适应提供新思路
- **Patch-level 不确定性加权聚合**：PGDR 的 softmax 不确定性加权融合策略可迁移至其他 foundation model 的下游特征蒸馏场景
- **GSCU 双空间贪心选择**：同时优化特征空间代表性和不确定性空间多样性的贪心策略，可用于其他主动学习框架的 batch 构建
- **轻量不确定性预测器设计**：用 MSE 训练的 transformer 预测器替代多次前向估算不确定性，显著降低推理开销，该思路可推广至其他需要 test-time augmentation 的不确定性估计场景
- **UPFT 与 LoRA 结合的低预算微调范式**：在标注极少的情况下通过不确定性提示辅助权重更新，可应用于其他 foundation model 的低资源适配任务

## 关键术语表
**Cold Start Active Learning (CSAL)**：无需初始标注集、单次查询的主动学习范式，解决传统 AL 多轮查询和初始预热依赖问题
**Surrogate Uncertainty**：基于 SAM 零样本推理的多增强输出熵估计的不确定性代理，非目标任务精确不确定性但可作为有效引导信号
**Patch-level Feature and Uncertainty Calculation (PFUC)**：利用 SAM "everything mode" 提取 patch 级特征并基于 K 次增强计算不确定性图的模块
**Patch-based Global Distinct Representation (PGDR)**：不确定性加权聚合 patch 特征为全局图像级表示的策略，$\lambda$ 控制不确定性放大强度
**Greedy Selection with Cluster and Uncertainty (GSCU)**：先聚类保证特征空间代表性，再以贪心最大化不确定性距离的方式均衡选择样本的策略
**Uncertainty-Prompted Fine-Tuning (UPFT)**：用 surrogate uncertainty map 替代原始 prompt encoder，驱动 SAM 在极低标注量下自适应下游分割任务
**Everything Mode**：SAM 自动生成全局网格点提示进行 class-agnostic 分割的模式，输出所有潜在区域而非特定对象
**LoRA (Low-Rank Adaptation)**：通过在冻结权重上注入低秩矩阵微调基础模型，大幅减少可训练参数同时保留预训练知识

## 可复现要素
- **数据集**：Promise12、UTAH、MSD Liver、ISIC 2018 均为公开数据集
- **代码**：已开源，https://github.com/HiLab-git/SUGFW-plus
- **关键超参**：K=10（增强次数），$\lambda=6$（不确定性放大系数），$\alpha_{max}=0.02$（不确定性损失权重），LoRA rank=32，epochs=300，batch size=16，lr=8e-4，$\omega=0.8$，ramp-up iterations=300
- **硬件**：4× NVIDIA GeForce RTX 2080Ti
- **模型**：SAM ViT-Base checkpoint
