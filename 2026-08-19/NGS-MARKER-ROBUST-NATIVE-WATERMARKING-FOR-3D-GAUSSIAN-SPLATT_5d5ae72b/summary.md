---
title: "NGS-MARKER-ROBUST-NATIVE-WATERMARKING-FOR-3D-GAUSSIAN-SPLATT"
source: https://arxiv.org/pdf/2608.17447v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 06:09:07"
field: "3D内容版权保护与水印"
keywords: ["3D Gaussian Splatting", "Native Watermarking", "Partial Infringement", "Copyright Protection", "PointTransformer", "Gradient-based Optimization", "Multimodal Watermarking"]
innovations: ["首个3DGS原生局部水印框架，直接在高斯原语上嵌入水印实现任意区域可解码", "梯度驱动的渐进式嵌入策略，避免一次性注入导致的边界不一致与累积畸变", "支持位流与图像多模态水印，且与间接水印方法无缝协同的混合保护机制"]
benchmarks: ["Blender Dataset", "LLFF Dataset", "Mip-NeRF 360", "D-NeRF Dynamic Scenes", "Tanks and Temples Complex Scenes"]
---

# 论文速读：NGS-MARKER: ROBUST NATIVE WATERMARKING FOR 3D GAUSSIAN SPLATTING

## 一句话总结
本文提出了 NGS-Marker，首个针对 3D Gaussian Splatting 的原生水印框架，直接在局部高斯原语上嵌入水印信号，有效解决了现有间接水印方法无法抵御"部分侵权"（Partial Infringement）的致命缺陷，同时保持高渲染保真度和强鲁棒性。

## 研究问题与动机
1. **间接水印范式的根本缺陷**：现有 3DGS 水印方法（如 3D-GSW、GaussianMarker、GuardSplat）均依赖渲染图像作为中间载体，保护的是"渲染结果"而非"原生高斯原语"，导致底层 3D 资产本身无保护。
2. **部分侵权场景严重但未受重视**：由于 3DGS 基于瓦片光栅化和独立高斯原语，攻击者可轻易提取局部对象（如场景中的某个角色）并嵌入新场景；实验表明所有现有方法在此场景下 Bit-Acc 降至 ~50%（等同随机猜测）。
3. **原生水印面临的独特挑战**：3DGS 场景间原语数量差异巨大，标准前馈网络难以直接处理变长输入；且单一全局嵌入策略无法保证任意局部区域均可解码水印。
4. **渲染质量与水印鲁棒性的矛盾**：直接逐块注入会导致边界不一致；重复随机注入会引发累积畸变，需设计兼顾均匀分布与视觉保真的嵌入机制。

## 核心贡献（创新点）
1. **揭示并形式化"部分侵权"问题**：首次系统实验证明现有间接 3DGS 水印在局部对象提取场景下完全失效（~50% 准确率），填补了该威胁模型的研究空白。
2. **首个原生本地水印框架 NGS-Marker**：通过扰动生成器 + 跨注意力解码器实现局部 patch 级水印嵌入，可从任意 k 近邻原语子集中解码所有权信息，本质区别在于直接操作高斯参数而非渲染图像。
3. **梯度驱动的渐进式嵌入策略**：摒弃一次性注入设计，改用冻结提取器引导的梯度优化使水印均匀覆盖全场景，同时保持 PSNR > 40 dB 的视觉质量。
4. **多模态与混合保护扩展**：支持位流、图像等多类型水印消息，且与间接水印方法（如 HiDDeN、3D-GSW）完全兼容，可协同优化实现原生 + 渲染双轨保护。

## 方法详解
**整体架构**：NGS-Marker 由水印注入器（$\mathcal{P}_g + \mathcal{P}_d$）和信息提取器 $\mathcal{E}$ 两部分组成，分训练与嵌入两阶段。

**1. 局部 patch 采样与 token 化**
- 从源场景 $\mathcal{G}^s$ 中随机采样 $k$ 个最近邻高斯原语构成局部 patch $\tilde{\mathcal{G}}^s$
- 使用 FPS + KNN 将 patch 转换为 tokens，作为 PointTransformer 的 query 输入

**2. 消息编码与特征融合**
- 水印消息 $\mathcal{M}$ 映射为文本 prompt（0/1 转 False/True），经 CLIP text encoder 编码为 $f_m$
- 扰动特征生成器 $\mathcal{P}_g$（堆叠 PointTransformer 层）以 token 为 query、$f_m$ 为 key/value，输出 latent 扰动特征 $f_d$：
$$f_d = \mathcal{P}_g(\text{tokenizer}(\tilde{\mathcal{G}}^s); \text{CLIP}(\mathcal{M}))$$

**3. 逐原语扰动解码**
- 扰动解码器 $\mathcal{P}_d$（含跨注意力层）对每个原语 $\tilde{\mathcal{G}}_i^s$ 作为 query、$f_d$ 为 key/value，并行预测各原语的微扰量
- 水淹没 patch：$\tilde{\mathcal{G}}^w = \tilde{\mathcal{G}}^s + \mathcal{P}_d(\tilde{\mathcal{G}}^s; f_d)$

**4. 联合训练损失**
$$\mathcal{L}_t = \text{MSE}(\mathcal{R}(\tilde{\mathcal{G}}^s, \theta), \mathcal{R}(\tilde{\mathcal{G}}^w, \theta)) + \lambda_t \cdot \text{BCE}(\mathcal{M}, \hat{\mathcal{M}})$$
其中 $\mathcal{R}$ 为可微渲染器，$\hat{\mathcal{M}}$ 为提取器输出。

**5. 渐进式场景嵌入**
- 冻结提取器 $\mathcal{E}$，对目标场景 $\mathcal{G}^w$ 进行梯度优化
- 随机采样 $\delta$ 个邻居原语 patch，计算提取损失与渲染保真损失：
$$\mathcal{L}_w = \text{MSE}(\mathcal{R}(\mathcal{G}^s, \theta), \mathcal{R}(\mathcal{G}^w, \theta)) + \lambda_w \cdot \text{BCE}(\mathcal{M}^{id}, \hat{\mathcal{M}})$$
- 采样时引入噪声、旋转、dropout 等扰动增强鲁棒性

**6. 所有权验证**
- 可疑区域提取 k-近邻原语 → 输入 $\mathcal{E}$ → 解码消息 → 与私有 ID 比对
- 可视化：按推断所有者着色 SH 系数，直观定位侵权区域

**7. 混合保护与多模态扩展**
- 混合保护：$\mathcal{L}_{\text{cooperate}} = \mathcal{L}_w + \lambda_{\text{indirect}} \cdot \mathcal{L}_{\text{indirect}}$
- 图像水印：将 CLIP text encoder 替换为 image encoder，提取器后 MLP 替换为 transposed conv decoder

## 实验与结果
**数据集**：训练集 24 场景（Blender、LLFF、Mip-NeRF 360、3D 编辑数据集），测试集 4 场景（chair、garden、bear、person-small）；部分侵权测试集通过提取水淹没场景中目标对象并嵌入未水淹没场景构造。

**评估指标**：Bit-Acc（位级准确率）、3D-Acc（原语级分类准确率，阈值 τ=75%）、PSNR/SSIM/LPIPS（渲染质量）、多种失真鲁棒性。

**核心结果**（16-bit 消息，Table 2）：
| 方法 | Bit-Acc ↑ | 3D-Acc ↑ | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|------|-----------|----------|--------|--------|---------|
| 3D-GSW | 49.17* | N/A | 30.37 | 0.960 | 0.051 |
| GaussianMarker | 50.00* | N/A | 30.75 | 0.961 | 0.046 |
| GuardSplat | 50.83* | N/A | 39.22 | 0.994 | 0.013 |
| WI-Naive | 57.54§ | 55.90 | 27.07 | 0.962 | 0.060 |
| WI-Iterative | 69.25§ | 72.30 | 25.54 | 0.886 | 0.105 |
| **NGS-Marker** | **97.94§** | **96.60** | **40.17** | **0.995** | **0.007** |

- 现有方法在部分侵权场景下 Bit-Acc ≈ 50%（随机水平）；NGS-Marker 达 97.94%，提升近 48 个百分点
- 3D-Acc 在 8/16/24-bit 下分别达 95.20%/96.60%/96.50%，且随位数增加不下降
- PSNR 40.17 dB、SSIM 0.995、LPIPS 0.007，显著优于所有基线

**鲁棒性结果**（Table 3，16-bit）：
- 无失真：Bit-Acc 98.35%，3D-Acc 97.90%
- 高斯噪声(σ=0.015)：97.06%/96.80%
- 旋转(±π)：98.31%/97.80%
- 缩放/平移：98.35%/97.90%（因归一化天然不变）
- 密度增加(0%-50%)：97.93%/97.40%
- Dropout(0%-50%)：97.41%/97.50%
- 组合攻击：96.28%/96.30%

**混合保护结果**（Table 5）：
- 与 3D-GSW 结合：Hybrid 模式 Bit-Acc 98.68%（渲染提取）、97.59%（原生提取），PSNR 33.16
- 与 GuardSplat 结合：Bit-Acc 99.14%/97.38%，PSNR 39.46
- 证明原生与间接保护可无缝协同

**消融实验**：
- δ=2048 时仍能维持 ~95.6% Bit-Acc，支持小规模侵权检测
- 使用全部原语属性（position+SH(3)+opacity+s(r)+rotation）效果最优（Bit-Acc 98.35%）
- 对抗攻击（C&W）下 16-bit/δ=8192：NGS-Marker Bit-Acc 65.83%，远高于 3D-GSW (58.55%)、GuardSplat (65.42%)
- 3D 编辑鲁棒性："将熊改为棕熊"→91.06%；"将人改为小丑"→90.53%

## 相关工作脉络
1. **2D 图像水印（HiDDeN 及其后续）**：Zhu et al. (2018) 首次提出端到端神经网络水印，后续工作提升容量与对抗鲁棒性；本文将其思想迁移至非结构化 3D 点集领域，但面临变长输入与局部可解码的新挑战。
2. **3D 点云/网格水印**：点云方法扰动坐标或密度（Li et al., 2021; Wei et al., 2024），网格方法利用顶点偏移或谱基（Wang et al., 2022; Zhu et al., 2024）；本文面向显式参数化的 3DGS 原语，保护粒度更细（可定位至单个高斯）。
3. **NeRF/辐射场水印（MarkNeRF、CopyNeRF）**：Li et al. (2023)、Luo et al. (2023) 通过图像修改或辐射正则化保护隐式表示；Song et al. (2024) 的 NeRFProtector 通过微调嵌入二进制串；这些均属"间接保护"，本文指出其在部分侵权场景下完全失效。
4. **现有 3DGS 水印（3D-GSW、GaussianMarker、GuardSplat）**：Jang et al. (2025)、Huang et al. (2024b)、Chen et al. (2025) 均依赖渲染图像提取水印；本文通过原生操作突破这一范式，首次实现从局部原语子集直接解码。
5. **GS-Hider、WaterGS、SecureGS**：Zhang et al. (2024)、Guo et al. (2024)、Zhang et al. (2025b) 虽涉及原生保护，但 GS-Hider/WaterGS 嵌入的是"隐藏场景"而非水印消息，SecureGS 仅针对 Scaffold-GS 且水印与场景绑定不可泛化；本文方法为通用、可泛化的本地水印。
6. **3DGS 分割与编辑**：Cen et al. (2025) 的 Segment Any 3D Gaussians 等技术使组件级提取成为可能，本文将其视为威胁模型并针对性防御，二者互为应用与反制关系。

## 局限性与未来方向
1. **3D 编解码技术尚不成熟**：相比 2D 领域，3D 点集编码/解码网络性能仍有限，影响水印容量与精度上限。
2. **大规模场景的分区与遍历策略待优化**：理论上支持任意尺度场景，但当前渐进优化对所有原语逐一处理，效率随场景规模线性增长（bear 场景 418K 原语需 28.7 min）。
3. **动态场景扩展未充分验证**：虽已将方法推广至 4DGS，但仅在小规模 D-NeRF 数据集上验证，复杂动态场景的鲁棒性尚不明确。
4. **对抗攻击下的超参数敏感性**：当攻击者准确获知嵌入超参数（bit len、δ）时，NGS-Marker Bit-Acc 从 93.50%（8-bit）降至 65.83%（16-bit），存在进一步防御空间。
5. **可视化定位依赖人工选取可疑区域**：当前验证流程需用户主动指定检测区域，未实现全自动侵权扫描。

## 研究启发与可借鉴点
1. **"轻量网络训练 + 推理时梯度优化"范式**：冻结提取器引导场景优化的策略巧妙分离了"学习能力"与"适应灵活性"，可迁移至其他 3D 资产优化任务（如去噪、压缩）。
2. **局部 patch 采样 + 跨注意力融合消息的通用设计**：PointTransformer 处理变长点集、CLIP 编码消息作为条件输入的架构，可推广至 3D 点云分类、分割等下游任务的隐式条件注入。
3. **多模态水印扩展思路**：将文本 encoder 替换为图像 encoder、MLP 替换为 transposed conv decoder 的模块化改造，为 3D 内容的水印/签名提供统一框架。
4. **原生与间接保护的协同优化**：混合损失 $\mathcal{L}_w + \lambda \cdot \mathcal{L}_{\text{indirect}}$ 的设计证明了多目标一致性可行，可启发其他"底层+表层"联合保护方案。
5. **归一化不变性设计**：将所有高斯中心缩放到单位球后再处理，使方法天然抵抗全局缩放与平移，此技巧适用于任何基于距离的 3D 点集操作。

## 关键术语表
**3D Gaussian Splatting (3DGS)**：一种显式 3D 场景表示方法，用各向异性高斯原语（位置、尺度、不透明度、颜色 SH 系数、旋转）的混合来近似辐射场，支持实时可微渲染。
**Native Watermarking**：直接在 3D 资产原生表示（如高斯原语参数）上嵌入水印，不依赖渲染图像作为中间载体，可实现局部可解码。
**Partial Infringement（部分侵权）**：攻击者从受保护 3DGS 场景中提取局部对象（如单个角色/物体）并嵌入新场景的版权侵犯行为，现有间接水印在此场景下完全失效。
**PointTransformer**：基于 Transformer 架构的点云处理网络，通过 self-attention 学习点集间的空间依赖关系，本文用于编码局部高斯 patch 的几何结构。
**Bit-Acc / 3D-Acc**：Bit-Acc 为提取消息与原始消息的位级相似度；3D-Acc 为随机选取高斯原语及其 k 近邻后，经提取器分类为"受保护"的比例（阈值 τ=75%）。
**Progressive Optimization（渐进优化）**：通过冻结提取器反向传播梯度，迭代优化目标场景的所有高斯原语参数，使水印均匀分布且渲染质量不变。
**SH Coefficients（球谐函数系数）**：Spherical Harmonic 系数，编码 3DGS 原语的视图相关颜色，本文水印嵌入的目标属性之一。
**HiDDeN**：Zhu et al. (2018) 提出的端到端图像水印网络，首次用编码器-解码器对实现不可见水印嵌入与提取，本文受其启发但针对 3D 点集重新设计。

## 可复现要素
- **数据集**：Blender、LLFF、Mip-NeRF 360、3D 编辑数据集（drums/ficus/hotdog/lego/materials/mic/ship/bicycle/bonsai/counter/flowers/kitchen/room/stump/treehill/fern/fortress/horns/llff flower/llff room/orchids/trex/train/truck）公开可用；测试集为 chair/garden/bear/person-small。
- **代码**：论文声明代码将开源，匿名仓库 https://anonymous.4open.science/r/NGS-Marker/（截至论文发表时）；Demo.mp4 作为补充材料提供。
- **关键超参**：训练时 k=8192、$\lambda_t=5$、150 epochs；嵌入时 $\delta=8192$、$\lambda_w=5$；混合保护 $\lambda_{\text{indirect}}=0.1$；3D-Acc 阈值 τ=75%。
- **硬件**：训练双 A100 GPU，嵌入单 A100 GPU。
- **平行加速**：NGS-Marker-Parallel 将场景沿 x-y 平面均分为四份，四卡并行嵌入后合并再优化，平均时间从 19.3 min 降至 8.4 min。
