---
title: "Self-Routed-Tensor-Adapters-for-Parameter-Eficient-Universal"
source: https://arxiv.org/pdf/2608.16384v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 15:22:21"
---

# 论文速读：Self-Routed-Tensor-Adapters-for-Parameter-Eficient-Universal

## 一句话总结
本文提出自路由张量适配器（SRTA），一种用于多域视觉适配的紧凑参数高效微调框架。该方法通过内部表示直接计算路由权重并动态混合共享的 Tucker 核心张量切片，在无需外部门控网络的情况下实现了样本特定的域自适应，以显著更少的参数量达到或略优于 MoE 基线的准确率。

## 研究问题与动机
- 多域视觉表征要求适配机制在异构域之间保持泛化，而非将知识碎片化为相互隔离的域专属模块。
- 标准 PEFT 方法（如 LoRA）使用固定低秩子空间处理所有输入，在风格、背景、语义差异显著的异构设置下容易产生负迁移与优化不均。
- 现有 MoE 类 PEFT 方法（如 MoLoRA、MOELoRA）依赖独立的外部门控网络计算路由概率，引入了额外参数，且导致路由决策与适配特征空间解耦，引发专家冗余和路由不稳定。
- 核心疑问：能否直接从适配器自身的低秩表示中派生路由权重，使路由与适配共享同一表征几何，从而实现紧凑的参数-精度权衡？

## 核心贡献（创新点）
- **内生自路由机制**。与 MoE-PEFT 依赖独立门控网络不同，SRTA 的路由概率直接由适配器低秩投影与可学习域坐标矩阵交互生成，使路由决策与适配特征几何
