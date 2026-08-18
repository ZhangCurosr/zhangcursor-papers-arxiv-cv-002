---
title: "P-Fusion-Prompt-based-Progressive-Infrared-Visible-Image-Fus"
source: https://arxiv.org/pdf/2608.13045v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 02:29:51"
---

# 论文速读：P-Fusion-Prompt-based-Progressive-Infrared-Visible-Image-Fus

## 一句话总结
本文提出 P²Fusion，一种基于双重内禀先验蒸馏的提示学习红外-可见光图像融合框架。该方法将传统静态硬约束转化为可学习的动态软提示（热显著性 + 空间质量），并通过门控动态专家重校准（GDER）模块实现跨模态特征的自适应解耦与修正，在五个主流数据集上均达到 SOTA，同时显著提升下游目标检测与语义分割性能。

## 研究问题与动机
- **静态硬约束的优化冲突**：现有先验引导方法（如 SAM/显著性掩码）通常将先验作为固定正则项注入网络，迫使模型优先拟合先验分布而非像素级特征融合，导致细粒度纹理被刚性边界牺牲。
- **多目标联合优化的权衡困境**：与下游检测/分割任务联合训练的方案易引发目标冲突，任务精度提升往往以背景细节保真度为代价，视觉美学与任务性能难以兼得。
- **外部大模型语义的粒度失配**：近期利用 CLIP/DINO 等大模型外部先验的路线虽缓解了硬约束，但引入的高维抽象表征与像素级融合任务的粒度不匹配，无法提供低层纹理重建的精确指导。
- **核心动机**：回归图像数据本质，摒弃外置或刚性约束，从红外与可见光图像自身挖掘内禀属性，构建自适应提示学习框架，实现动态场景感知与像素级精准融合。

## 核心贡献（创新点）
- **提出 Teach-to-Fuse (T2F) 范式**：系统规避硬约束、多目标冲突与粒度失配三大痛点，将静态惩罚转化为过程导向的动态提示，最大化挖掘模态内禀属性。
- **设计 GDER 模块实现特征解耦修正**：不同于传统 MoE 用于容量扩展，该模块专为特征解耦与偏差校正设计，协同先验响应的模态专家与先验无关的全局注意力专家，实现模态竞争的自适应调解。
- **构建双重内禀先验蒸馏机制**：选择性提取红外热显著性与可见光空间质量作为互补先验，分别通过 SegFormer-B2 与 BRISQUE 评估器生成可学习的软提示，在无成对 GT 的无监督设置下提供隐式监督。
- **全面的基准验证与感知分析**：在五个公开数据集及十余项评测任务上验证，框架在视觉保真度、跨域泛化及下游感知任务上均展现显著优势，并开源代码。

## 方法详解
- **双教师提示蒸馏（Dual-Teacher Prompt Distillation）**：选取红外热显著性 $T_{ir}$ 与可见光空间质量 $T_{vis}$ 作为内禀先验。红外分支由预训练 SegFormer-B2 生成显著图，经后处理二值化（行人/车辆为前景）并双线性插值匹配分辨率；可见光分支由无参考评估器 BRISQUE 生成全局质量分 $s \in [0,100]$，归一化为 $T_{vis} = (100-s)/100$。教师输出投影为提示 $P_m$ 后嵌入并与特征相加：$F_m = f_m + \mathrm{Embed}(P_m)$。
- **交叉注意力特征调制**：保持双流架构，采用标准交叉注意力缓解模态混淆：
  $F'_{ir} = \mathrm{CA}(F_{ir}, F_{vis}, F_{vis}),\quad F'_{vis} = \mathrm{CA}(F_{vis}, F_{ir}, F_{ir})$
  其中 $\mathrm{CA}(Q,K,V)=\mathrm{softmax}(\frac{QK^\top}{\sqrt{d}})V$。
- **门控动态专家重校准（GDER）**：包含两个功能解耦专家：(1) **先验响应模态专家 $E_{mod}$**：显式消费动态提示，强化热目标或高对比纹理；(2) **先验无关注意力专家 $E_{att}$**：独立于提示，利用 CBAM 全局注意力挖掘长程依赖与细粒度结构。门控网络计算动态权重 $w = \mathrm{softmax}(G(F', P))$，融合公式为：
  $F'' = F' + w_1 \cdot E_{mod}(F') + w_2 \cdot E_{att}(F')$
  经 N 次迭代后拼接输入重建解码器输出 $I_{fused}$。
- **优化目标**：总损失替换传统硬约束为先验蒸馏项：
  $\mathcal{L}_{total} = \lambda_1 \mathcal{L}_{int} + \lambda_2 \mathcal{L}_{ssim} + \lambda_3 \mathcal{L}_{grad} + \lambda_4 \mathcal{L}_{ir}^{distill} + \lambda_5 \mathcal{L}_{vis}^{distill}$
  其中 $\mathcal{L}_{grad}$ 保边缘、$\mathcal{L}_{int}$ 平衡热/纹理强度、$\mathcal{L}_{ssim}$ 自适应加权保结构；蒸馏损失分别采用 MSE（可见光）与 BCE（红外显著性图）。

## 实验与结果
- **数据集与基线**：在 MSRS、M3FD、FMB、RoadScene、DroneVehicle 五个基准测试；对比 SAGE、Dit-Fuse、MRFS、FreeFusion、FreqGAN、LutFuse、LRRNet、DDFM、TarDal、ReCoNet、TIM 等 12 个 SOTA 方法。
- **融合质量**：P²Fusion 在 16 项核心指标中获得 12 项第一、3 项第二，实现 14/20 关键指标的 SOTA。MSRS 上 FQIE=0.850、VIF=0.445、Qabf=0.682、MI=3.905，
