---
title: "Predicting-Multiple-Clinical-Outcomes-Related-to-Functional"
source: https://arxiv.org/pdf/2608.23531v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 23:51:15"
---

# 论文速读：Predicting-Multiple-Clinical-Outcomes-Related-to-Functional

## 一句话总结
本文基于 MAISON-LLF 多模态传感器数据集，将下肢骨折或髋关节置换术后老年患者的五项临床恢复指标（社交隔离、髋/膝功能、mobility、肌力）的预测建模为多输出回归问题，验证了联合预测相比单输出预测能显著提升准确性，且表格型深度学习模型（尤其是 NODE）在跨个体与跨时间验证中表现最优。

## 研究问题与动机
- 现有研究多孤立预测单一临床目标（如仅预测社交隔离风险或仅预测骨折愈合），未能充分利用功能恢复与社交参与度之间的内在关联。
- 社区康复期老年人的功能衰退与社交隔离在临床上高度耦合（如 TUG 与 Chair Stand、SIS 显著相关），但缺乏同时预测多个相关结局的建模探索。
- 传统机器学习在多输出相关性建模上表征能力有限，而新兴表格深度学习模型在该场景下的潜力尚未被充分评估。
- 需要一种端到端的联合预测框架，为临床提供更全面、上下文感知的康复轨迹评估，以辅助早期干预与照护决策。

## 核心贡献（创新点）
1. **创新点**：首次将社区康复期多项临床指标预测形式化为多输出回归问题，同步建模功能恢复与社交隔离。 **区别**：现有 MAISON 系列工作多聚焦单目标分类或聚类解释，本文直接面向连续型多变量联合预测并证明其实际收益。
2. **创新点**：系统对比单输出与多输出回归，验证联合建模可利用临床指标间的内在相关性显著降低预测误差。 **区别**：不同于传统 MTL 侧重共享表示学习，本文从回归误差与统计显著性角度直接量化多输出范式的实用增益。
3. **创新点**：确立表格深度学习模型（NODE 等）在小样本临床传感器预测任务中对传统 ML 的绝对优势。 **区别**：既往工作多在 EHR 或医学影像上验证 MTL/多输出，本文将其迁移至 wearable tabular 数据并完成严格的双验证集对比。
4. **创新点**：通过 SHAP 分析揭示“行为发生时机”与“睡眠结构”比原始活动量更具预测价值。 **区别**：不同于常规特征选择仅关注重要性阈值，本文结合跨 18 折折叠的稳定性排序，为康复监测的特征工程提供可解释的临床指引。

## 方法详解
- **数据与特征**：使用 MAISON-LLF 数据集，包含 18 名术后老年患者 8 周（1008 participant-days）的多模态传感器数据。提取 46 个每日特征，涵盖 acceleration（count, entropy, kurtosis, 时段事件数, coefficient-of-variation 等）、heartrate（max, mean, std, hours-with-data）、motion/step（count, max, ratio, max-timestamp）、position-GPS（duration, distance-travelled）与 sleep（deep, light, REM, snoring, wakeup-count）。
- **预测目标**：每两周采集一次临床量表，将 14 天内的每日传感器记录映射至同一临床得分，形成多输出标签向量 [SIS, OHS, OKS, TUG, Chair Stand]。
- **模型架构**：
  - 传统 ML：LR, ExtraTrees (ET), Random Forest (RF), KNN, SVR（通过 sklearn RegressorChain 实现多输出）。
  - 表格 DL：NODE（Neural Oblivious Decision Ensembles）、FT-Transformer、TabPFN、TabNet。
  - 所有目标变量采用 min-max 归一化，预测后反归一化计算 MSE 与 MAE。
- **验证策略**：采用两种严格交叉验证：(1) Leave-one-person-out (LOPO)
