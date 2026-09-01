---
title: "Teeth2Point-A-Two-Stage-Dental-CBCT-ROI-to-Point-Segmentatio"
source: https://arxiv.org/pdf/2608.18667v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 10:47:58"
field: "3D医学图像分割"
keywords: ["CBCT segmentation", "point cloud transformer", "self-supervised learning", "dental imaging", "two-stage framework", "DINO pretraining"]
innovations: ["ROI-to-Point两阶段框架解决CBCT稀疏结构分割", "梯度引导自适应采样平衡分辨率与计算效率", "DINOv2风格点云自监督预训练提升异常案例鲁棒性"]
benchmarks: ["STS2024", "ToothFairy2", "ToothFairy3", "Internal Dataset"]
---

# 论文速读：Teeth2Point-A-Two-Stage-Dental-CBCT-ROI-to-Point-Segmentatio

## 一句话总结
本文提出 Teeth2Point，一种基于点云的 Transformer 两阶段牙科 CBCT 分割框架，先将 CNN 提取的牙齿 ROI 转换为点 token，再通过 DINOv2 风格自监督预训练的 Transformer 实现高分辨率全局上下文建模，在缺失/错位牙齿等异常案例中显著提升分割精度。

## 研究问题与动机
1. **异常牙齿标注难题**：CBCT 中缺失或错位牙齿的实例级标签是临床关键需求，但现有模型在此类结构性变异下性能显著下降。
2. **体素处理的计算-上下文权衡**：密集体素分割中，扩大上下文需增大 patch 或降低分辨率，但 CBCT 中牙齿仅占 0.5%-1.4% 的体素，背景主导导致计算浪费。
3. **点云方法的上下文局限**：现有 Volume-to-Point 方法（如 Point-UNet）依赖局部特征聚合，难以建模长距离解剖关系（如左右对称、上下颌对应）。
4. **标注效率瓶颈**：高精度分割依赖大量标注数据，自监督预训练在牙科点云领域尚未被充分探索。

## 核心贡献（创新点）
1. **ROI-to-Point 转换框架**：将两阶段 CNN ROI 裁剪与点云 Transformer 结合，避免体素级密集计算，同时保留高分辨率细节——区别于纯体素或纯点云方法。
2. **梯度引导自适应采样（GGAS）**：根据局部梯度幅度分布自适应分配采样点密度，边界区域获得更多采样点——与均匀采样或单纯随机采样本质不同。
3. **DINOv2 风格点云自监督预训练**：引入 DINO 式自蒸馏 + 随机 Token 掩码预测，学习增强不变性表示——首次将 SSL 应用于牙科 CBCT 点云建模。
4. **渐进式微调策略**：先在 GT ROI 上微调，再在预测 ROI 上微调并配合特定数据增强，缓解第一阶段误差传播——区别于单次端到端微调。
5. **异常案例显著增益**：在四个数据集上，相比最强两阶段基线平均提升 1.44 DSC 点，相比 nnU-Net 提升 1.9 点——聚焦临床高价值场景。

## 方法详解
**第一阶段：CNN ROI 提取**
- 使用 nnU-Net 对完整 CBCT 体素进行滑动窗口推理，输出 logits 后取 argmax 得到前景掩码 M。
- 对 M 进行 3D 6-connected 形态学膨胀（≈0.3mm），得到扩张 ROI 掩码 M_aug。
- 提取 M_aug 内所有体素作为点集 X，每个点包含坐标 c_i、归一化强度 I_i、梯度向量 g_i 及标签 y_i。

**自适应采样（GGAS）**
- 计算所有点梯度幅度 |g_i|，min-max 归一化后按坐标离散化到网格（ε=2）。
- 每个网格单元内构建梯度直方图（bin 宽度 α=0.125），从非空 bin 按梯度幅度概率采样一点。
- 结果：高梯度区域（边界）密集采样，低梯度区域（平坦面）稀疏采样。

**架构设计**
- 线性嵌入层将点属性映射为 point token。
- Encoder-Decoder backbone（Point Transformer v3 风格）：空间池化（max pooling within voxel）+ 反池化（nearest-neighbor interpolation）+ skip connections。

**自监督预训练（SSL）**
- 学生网络 θ_s 编码局部视图 X_l，EMA 教师网络 θ_t 编码全局视图 X_g。
- Point-level distillation：每个点特征找最近邻，最小化 KL 散度；Sinkhorn-Knopp 稳定训练。
- Masked prediction：随机掩码点（比例 r=0.3→0.7 cosine 调度），插入可学习 mask token，预测被掩码点特征。
- 牙科专用增强：对称增强（翻转+标签重映射）、伪影增强（模拟金属条状伪影）、强度扰动、旋转与弹性形变。

**渐进式微调**
- Stage 1（GT-ROI）：使用真实标注 ROI 的前景点，训练 400 epochs。
- Stage 2（Adaptation）：使用预测 ROI，标签含背景和不完整结构，训练 100 epochs，配合 CBCT 增强。
- 损失函数：Dice Loss + Cross-Entropy，等权加权。

**体素化后处理**
- 点预测通过 KNN（K=4）插值到未采样点，再体素化并合并第一阶段背景掩码。

## 实验与结果
**数据集**：STS2024（27样本，异常率 82.5%）、ToothFairy2（480样本，80.8%）、ToothFairy3（532样本，23.1%）、Internal（1170样本，9.0%）；各向同性间距 0.3mm。

**基线对比**：nnU-Net、ToothSeg、MedNeXt、VideoTeeth、U-Mamba1/2、Point-UNet。

**核心结果**：
- 平均 DSC-A（异常案例）：Teeth2Point 达 **93.70**，较 nnU-Net（91.82）提升 **+1.88 点**，较 ROI-cropped nnU-Net（92.26）提升 **+1.44 点**。
- 平均 DSC-R（常规案例）：**97.32**，较 nnU-Net（96.68）提升 **+0.64 点**。
- 常规-异常性能差距缩小约 **26%**。
- 推理速度：整体 **~13.7s/体积**，点阶段仅 **0.9s**。

**消融实验**：
- 去掉 SSL 预训练性能下降；GGAS 优于均匀采样。
- 仅在 GT ROI 微调时，内部数据集 DSC 下降 >5.6%，加入适配阶段后恢复。

**标注效率**：仅用 **10 个标注样本**微调即达 **86.7% DSC**，接近从零训练 100 样本的效果。

## 相关工作脉络
1. **nnU-Net（Isensee et al., 2021）**：通用医学图像分割基线，本文作为第一阶段 ROI 提取器，但未针对牙科稀疏结构优化。
2. **Point-UNet（Ho et al., 2021）**：最早的 ROI-to-Point 方法，使用 RandLA-Net 局部聚合，本文指出其全局上下文建模不足。
3. **VideoTeeth（Ma et al., 2025）**：全体积 Transformer，计算成本高（149 滑动窗口），本文在保持性能的同时大幅降低计算量。
4. **U-Mamba（Ma et al., 2024）**：状态空间模型，线性复杂度但异常案例性能仍低于本文方法。
5. **DINOv2（Oquab et al., 2024）**：自然图像自监督预训练标杆，本文首次将其点云版本适配到牙科 CBCT。
6. **Sonata（Wu et al., 2025）**：点云 SSL 方法，本文参考其蒸馏框架但引入牙科域增强和掩码预测。

## 局限性与未来方向
1. **非端到端架构**：依赖独立第一阶段 ROI 提取器，若牙齿完全漏检则第二阶段无法恢复。
2. **单分割实验**：仅报告一个 70/30 划分的结果，统计显著性未充分验证。
3. **硬件限制**：因 FlashAttention-2 依赖，无法在 T4 GPU 上部署，限制临床可用性。
4. **数据量有限**：预训练使用内部 3,224 个无标签扫描，大规模泛化能力待验证。
5. **未来方向**：端到端联合训练、跨中心泛化评估、实时推理优化。

## 研究启发与可借鉴点
1. **ROI-to-Point 范式**：对于稀疏结构分割（如血管、骨小梁），可复用"CNN 粗定位 + 点云精分割"的两阶段思路，平衡计算与精度。
2. **梯度引导采样策略**：GGAS 思路可迁移至其他医学点云任务（如器官表面建模），利用梯度信息自适应分配采样密度。
3. **点云 SSL + 掩码预测**：DINOv2 式蒸馏结合随机 token 掩码的 SSL 方案，适用于标注稀缺的 3D 医疗成像领域。
4. **渐进式微调缓解误差传播**：从 GT 到预测 ROIs 的两步微调策略，可作为鲁棒性训练的一般性技巧。
5. **少量样本高效微调**：10 样本达 86.7% DSC 的结果表明 SSL 预训练对低资源场景极具价值，值得在更多任务上验证。

## 关键术语表
**CBCT（Cone Beam Computed Tomography）**：锥形束 CT，牙科常用的三维成像模态，分辨率高但辐射剂量较低。
**DSC（Dice Similarity Coefficient）**：dice 系数，衡量分割结果与标注重叠度的指标，取值 0-1，越高越好。
**ROI（Region of Interest）**：感兴趣区域，本文指包含所有牙齿的裁剪体积。
**Point Token**：点云特征向量，包含坐标、强度、梯度等信息，经嵌入层后作为 Transformer 输入。
**GGAS（Gradient-Guided Adaptive Sampling）**：梯度引导自适应采样，根据局部梯度分布动态分配点采样密度的方法。
**SSL（Self-Supervised Learning）**：自监督学习，利用无标签数据通过 pretext task 学习表征的技术。
**DINOv2**：Meta 提出的自蒸馏视觉预训练框架，本文将其适配到点云领域。
**EMA Teacher**：指数移动平均教师网络，在自蒸馏中缓慢更新以提供稳定目标。

## 可复现要素
**数据集**：STS2024、ToothFairy2/3 为公开挑战数据集；Internal Dataset 未公开。
**代码/权重**：论文未提及开源声明。
**关键超参**：ε=2（网格离散化）、α=0.125（梯度直方图 bin 宽度）、K=4（KNN 插值）、mask ratio 0.3→0.7 cosine 调度、预训练 200 epochs/batch=32、微调 GT-ROI 400 epochs/batch=8、适配 100 epochs/batch=8。
