---
title: "When-Two-Tracers-Disagree-An-Investigation-of-Multimodal-Fus"
source: https://arxiv.org/pdf/2608.19063v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 10:49:40"
---

# 论文速读：When-Two-Tracers-Disagree-An-Investigation-of-Multimodal-Fus

## 一句话总结
本文在前列腺癌全身PET/CT病灶分割任务上系统对比了早期拼接与中间层交叉注意力两种多模态融合策略，结果表明强单示踪剂nnU-Net基线（PSMA Dice=0.93，FDG Dice=0.81）已显著优于所有融合架构；融合未能稳定带来性能提升，提示在高异构示踪剂场景下，需在架构层面更好保留模态特异性表征。

## 研究问题与动机
1. **临床需求**：准确分割PSMA与FDG PET/CT病灶是前列腺癌肿瘤负荷评估与放疗规划的关键，但人工标注耗时且存在观察者间变异。
2. **单一示踪剂局限**：PSMA反映受体表达，FDG捕捉糖酵解活性；部分去分化或PSMA阴性病灶（约5-10%前列腺癌低表达PSMA）仅FDG显像，单一示踪剂易漏诊异质性表型。
3. **融合架构缺乏共识**：现有PET/CT分割模型多为单示踪剂设计，双示踪剂融合面临生物分布差异大、病灶数量/体积高度不对称等问题，简单拼接或共享编码器易引发表征冲突。
4. **亟需系统性Benchmark**：在公共DEEP-PSMA数据集上量化早/中期融合策略的实际增益与失败边界，为后续临床多模态分割研究提供实证依据。

## 核心贡献（创新点）
1. **提出DECA-UNet**：设计双编码器交叉注意力U-Net，通过零初始化的通道级门控实现PSMA与
