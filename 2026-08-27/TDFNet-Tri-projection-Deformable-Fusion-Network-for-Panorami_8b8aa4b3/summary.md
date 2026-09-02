---
title: "TDFNet-Tri-projection-Deformable-Fusion-Network-for-Panorami"
source: https://arxiv.org/pdf/2608.25808v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 06:55:55"
---

# 论文速读：TDFNet-Tri-projection-Deformable-Fusion-Network-for-Panorami

## 一句话总结
本文提出 TDFNet，首个将 ERP、CMP 与切线投影（Tangent Projection）三投影表示与可变形注意力相融合的 panoramic salient object detection (PSOD) 框架，通过跨投影几何感知采样与纬度引导的自适应融合，有效缓解球面到平面投影带来的极区拉伸与立方体边界不连续等几何畸变，在四个主流全景显著性检测基准上持续刷新 SOTA。

## 研究问题与动机
- **核心问题**：全景图像从球面投影到 2D 平面时不可避免的几何畸变，严重制约了现有 PSOD 方法的特征判别力与目标定位精度。
- **ERP 单投影局限**：虽保持全局拓扑连续性，但非均匀采样导致高纬度（极点）区域特征严重拉伸、语义模糊，现有增强手段（如 DDS、DATFormer）仍无法弥补赤道与极区的固有几何差异。
- **ERP+CMP 双投影局限**：CMP 通过六面体分解缓解极区畸变，但立方体面边界处引入语义不连续性，破坏目标结构的连贯性（如 CSMANet、HPNet）。
- **融合策略缺失**：现有方法缺乏统一表示来同时保留全局连续性、局部几何保真度与精细边界信息；且异构投影特征在几何域与空间组织上差异巨大，直接相加/拼接易导致跨投影不一致。

## 核心贡献（创新点）
- **首个三投影可变形融合架构 (TDFNet)**：首次将切线投影引入 PSOD，构建 ERP-CMP-Tangent 三分支编码器，分别捕获全局连续性、局部保真几何与无畸变细粒度边界信息；与以往双投影工作相比，本质区别在于引入离散无畸变视口作为跨投影语义校准信号，填补了边界一致性建模的空白。
- **跨投影可变形注意力模块 (CDA)**：利用 ERP 与 CMP 之间的空间对应关系构建混合几何参考点集，引导可变形注意力进行跨投影上下文聚合；与标准可变形注意力仅依赖单一规则网格不同，CDA 的采样位置显式受跨投影几何邻域约束，显著提升对投影畸变的鲁棒性。
- **纬度引导融合模块 (LGF)**：提出基于球面纬度先验的几何置信度权重，自适应平衡 ERP 与 CMP 特征；同时以切线特征为无畸变语义参考，通过交叉投影注意力实现特征精炼；相比黑盒通道拼接或简单加权，LGF 实现了“几何先验引导+内容自适应”的异构对齐。
- **轻量可配置的双策略设计**：提供无参数的 Geometry Only 与仅增加 0.084M 参数的 Mixed 两种融合策略，在小数据集上几何规则更稳健，在大上新数据集上数据驱动策略更灵活，兼顾精度与部署效率。

## 方法详解
- **三分支编码器 (Tri-branch Encoder)**：输入 ERP 图像 $I_{\mathrm{erp}}$，并行提取分层特征：
  1. **ERP 分支**：直接输入 Hybrid-ViT-based 编码器 $\mathcal{E}_{\mathrm{erp}}$，输出 $\{f_{\mathrm{erp}}^l\}$，保留全局经纬度连续性。
  2. **CMP 分支**：经 E2C 投影转为 6 个立方体面，分别用 ResNet 编码后通过 C2E 映射回 ERP 坐标域，得到对齐的多尺度特征 $\{f_{\mathrm{cmp}}^l\}$。
  3. **Tangent 分支**：在 4 个纬度行分布 $N_t=18$ 个切视点（$\mathrm{FOV}=80^\circ$），通过逆球心投影 $\mathcal{G}^{-1}$ 映射至归一化 ERP 采样域并经双线性插值裁剪；冻结 ResNet18 编码后接 GAP 得到 $18 \times 256$ 的全局特征 $f_{\mathrm{tan}}$。
- **跨投影可变形注意力 (CDA)**：
  - **混合参考点生成 (HRPG)**：后三层特征投影至 $d=256$ 并叠加 2D 正弦位置编码。以 ERP 特征网格点为锚点，沿 8 个方向、4 个径向层级生成跨投影邻域参考点集 $R$。
  - **可变形 Transformer 聚合**：特征展平后输入可变形 Transformer 块，Query 预测偏移 $\Delta \mathbf{p}$ 与注意力权重 $\mathbf{A}$，动态调整参考点 $\hat{\mathbf{p}} = \mathbf{R} + \Delta \mathbf{p}$，经双线性采样与加权聚合后通过 FPN 恢复多尺度特征 $\hat{f}_{\mathrm{erp}}, \hat{f}_{\mathrm{cmp}}$。
- **纬度引导融合 (LGF)**：
  - 构建归一化纬度图 $M_{\mathrm{lat}} \in [-1, 1]^{H \times W}$（赤道 0，极点 ±1）。
  - **Geometry Only**：无参数，按 $W_{\mathrm{geo}} = \cos(M_{\mathrm{lat}} \pi / 2)$ 计算权重，赤道侧重 ERP、极点侧重 CMP。
  - **Mixed**：轻量卷积子网（含 1×1 Conv 与 DSConv）结合 $M_{\mathrm{lat}}$ 预测空间自适应权重 $W_{\mathrm{mix}}$，实现几何先验与图像内容的联合建模。
  - **切线引导精炼**：将 ERP-CMP 融合特征展平作为 Query，18
