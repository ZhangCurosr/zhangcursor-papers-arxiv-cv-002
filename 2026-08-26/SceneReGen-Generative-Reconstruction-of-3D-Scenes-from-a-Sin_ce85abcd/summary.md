---
title: "SceneReGen-Generative-Reconstruction-of-3D-Scenes-from-a-Sin"
source: https://arxiv.org/pdf/2608.23930v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 02:17:20"
---

# 论文速读：SceneReGen-Generative-Reconstruction-of-3D-Scenes-from-a-Sin

## 一句话总结
本文提出 SceneReGen，一种面向单图像 3D 场景的对象中心生成式重建框架。通过选择性姿态因子化（Selective Pose Factorization）策略，将物体朝向直接编码至生成的规范化网格顶点，而平移与缩放则从实例级与全局场景线索中显式估计，从而弥合对象级生成先验与场景级重建空间表示之间的根本性对齐缺口。

## 研究问题与动机
- 单图像 3D 场景重建需同时完成部分遮挡对象的几何补全，并将其一致地放置于共享的观察对齐坐标系中，单视角下的自遮挡与互遮挡导致大量几何缺失。
- 现有对象级生成模型通常输出居中且尺度归一化的规范坐标网格，与场景重建所需的真实朝向、平移与尺度存在显著的表示差距（representation gap）。
- “先补全后放置”的分阶段流水线易将几何补全误差传播至位姿估计；而端到端联合场景建模将几何与位姿强耦合，导致两类误差难以分离优化。
- 核心假设：朝向与图像条件高度耦合，可融入生成几何；平移与缩放更依赖全局布局与跨对象上下文证据，适合显式回归，据此提出选择性姿态因子化。

## 核心贡献（创新点）
- 提出对象中心范式的单图像 3D 场景生成式重建框架，将完整对象资产作为场景重建原语，打破传统规范坐标系对象生成的场景部署瓶颈。
- 首次系统界定对象生成与场景重建间的表示差距，设计选择性姿态因子化机制，将旋转解耦至生成顶点层，平移与缩放交由场景证据分支独立估计。
- 构建几何感知生成重建管线：引入 VGGT-Ω 作为密集几何编码器，设计可学习形状查询与位置查询，分别驱动旋转感知 Diffusion 形状生成与位姿组装。
- 在 3D-FUTURE 评估子集上取得最优场景级 CD、F-Score 与 3D 边界框 IoU，并在自动驾驶与具身智能场景中验证了资产级 3D 表示的跨域应用潜力。

## 方法详解
- **问题形式化**：给定场景图像 $I$ 与实例掩码 $M=\{\mathbf{M}_i\}_{i=1}^n$，生成分支输出居中、尺度归一化但已编码观测朝向的网格 $\mathcal{O}_{\mathrm{pose-aware}}^i$；位置分支估计平移向量 $\mathbf{t}_i \in \mathbb{R}^3$ 与各向同性缩放 $s_i \in \mathbb{R}_{>0}$，最终组装公式为 $\mathcal{O}_{\mathrm{final}}^i = s_i \cdot \mathcal{O}_{\mathrm{pose-aware}}^i + \mathbf{t}_i$。
- **几何编码与遮挡增强**：提取对象裁片 $\tilde{\mathbf{I}}_o^i$ 输入 VGGT-Ω 编码器，得到几何特征序列 $\mathcal{F}_{obj}^i \in \mathbb{R}^{N \times C}$。训练时采用随机放置 5 个正方形掩码覆盖前景 20% 像素的遮挡增强策略，Mask 区域填充 $[80, 180]$ 均匀随机像素，提升部分观测下的补全鲁棒性。
- **旋转感知形状生成**：可学习形状查询 $Q_{shape} \in \mathbb{R}^{M \times C}$ 通过多层交叉注意力聚合对象特征得到条件 token $C_{pose}^i = \mathrm{Attention}(Q_{shape}, \mathcal{F}_{obj}^i, \mathcal{F}_{obj}^i)$。在预训练 DiT（Hunyuan3D 基干）的每个 transformer block 中插入额外交叉注意力层，使去噪过程动态接收旋转/朝向条件，输出姿态感知网格后接多视图扩散完成 UV 纹理映射。
- **场景级位姿重建**：共享几何编码器同时提取全局场景特征 $\mathcal{F}_{scene}$ 与各对象局部特征 $\mathcal{F}_{obj}^i$。可学习位置查询 $Q_{pos}^i$ 执行 $TS_{pos}^i = \mathrm{Attention}(Q_{pos}^i, \mathcal{F}_{obj}^i, \mathcal{F}_{scene})$，直接编码目标平移与缩放参数。
- **训练损失**：生成分支采用 flow-matching 损失 $\mathcal{L}_{gen} = \mathbb{E}[\|v_\theta(z_t, t, c) - (z_0 - \epsilon)\|_2^2]$；位置分支采用 L1 损失 $\mathcal{L}_{pos} = \frac{1}{N}\sum_i \|TS_{pos}^i - \mathrm{GT}(TS_{pos}^i)\|_1$，总损失为两者之和。

## 实验与结果
- **数据集与评估设置**：基于 3D-FUTURE 数据集训练；测试子集随机采样 150 个场景。几何评估使用 FilterReg 对齐点云，度量包括场景级/对象级 CD、F-Score 与 3D 边界框 IoU ($\mathrm{IoU}_B$)，F-Score 距离阈值固定为 0.1。
- **对比基线**：单对象优化类（Gen3DSR, SAM 3D, ShapeR）与全场景联合重建类（MIDI, SceneGen, TRELLIS.2）。
- **主要结果**：SceneReGen 在场景级 CD-S (0.009↓)、F-Score-S (89.50↑) 与 $\mathrm{IoU}_B$ (0.536↑) 上全面领先；对象级 CD-O (0.031↓) 与 SceneGen 并列最优，F-Score-O (68.95↑) 排名第二。相较 Scene
