---
title: "MorphoCLIP-Text-Supervised-Contrastive-Learning-for-Perturba"
source: https://arxiv.org/pdf/2608.22690v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 19:27:56"
field: "生物医学图像分析 / 形态学谱型"
keywords: ["Cell Painting", "contrastive learning", "morphological profiling", "CLIP", "batch effect correction", "CPJUMP1", "DINOv3", "gene-compound matching"]
innovations: ["冻结 DINOv3+ModernBERT 骨干仅训练 CrossChannelFormer 实现单卡训练文本监督 Cell Painting 对比学习", "扰动感知批次+复本对齐损失 L_rep 显式优化实验重复性", "条件相对板校正 condition-relative plate offsets 解耦技术漂移与条件信号"]
benchmarks: ["CPJUMP1 pilot", "standard CPJUMP1 benchmark (mAP / fraction retrieved)", "RxRx3-core CORUM (baseline comparison)"]
---

# 论文速读：MorphoCLIP-Text-Supervised-Contrastive-Learning-for-Perturba

## 一句话总结
论文提出 **MorphoCLIP**，一个基于冻结 DINOv3 和 BioClinical ModernBERT 的文本监督对比学习模型，将 Cell Painting 显微图像与化合物、CRISPR 敲除和 ORF 过表达的文本描述关联到统一嵌入空间；在 CPJUMP1 验证集/测试集上，双向检索的 Top-10 召回率约为随机水平的 3-4 倍，重复实验对齐损失可显著提升可重复性 mAP，但基因-化合物跨模态匹配仍是个开放难题。

## 研究问题与动机
1. **核心问题**：如何建立 Cell Painting 图像与扰动描述（化合物/基因）之间的跨模态语义关联，支持双向检索（图→文本、文本→图）和表型匹配。
2. **技术变异掩盖生物信号**：同一扰动在不同板/天之间因光照、染色、焦距差异产生的批次效应往往强于生物学效应，导致跨实验匹配困难。
3. **现有方法局限**：CLOOME/MolPhenix 仅支持化合物；CellCLIP 虽覆盖化合物和基因但依赖大骨干网络且跨类别检索弱；CWA-MSN 无文本分支无法自然语言搜索。
4. **标签不完备与软监督需求**：两种化合物可能共享主靶点但二级效应不同；基因敲除和化合物可能通过不同机制影响同一条通路，传统的硬负样本对比学习不适用。
5. **算力约束**：CPJUMP1 包含数 TB 显微数据，单张消费级 GPU 难以端到端训练大型基础模型，需要缓存+轻量微调策略。

## 核心贡献（创新点）
1. **首个覆盖化合物+CRISPR+ORF的文本监督Cell Painting模型**：冻结 DINOv3 视觉骨干和 ModernBERT 文本骨干，仅训练 CrossChannelFormer（约14M参数）和投影层，单卡 RTX 5080 即可完成训练。
   - 与 CellCLIP 的本质区别：CellCLIP 使用更大的 DINOv2-g 骨干并端到端训练；MorphoCLIP 通过特征缓存+冻结骨干实现极低训练成本。
2. **扰动感知批次采样与复本对齐损失设计**：提出 L_rep 将同一扰动的重复井在批次内聚集，显式优化重复实验间的一致性。
   - 与 CWA-MSN 的区别：CWA-MSN 基于 Masked Siamese 做跨孔对齐且无文本分支；本文在对比学习框架内引入可微的复本对齐项，与文本-图像对比损失联合训练。
3. **条件相对板校正（condition-relative plate offsets）**：在梯度-free pass 中计算每板相对于同类条件其他板的偏移量并修正图像嵌入，既消除板间漂移又保留条件语义。
   - 与直接减板均值（global batch correction）的区别：后者会抹除与条件相关的生物学信号；本文偏移在各条件内求和为零，不对齐条件均值。
4. **完整的检索协议与分析机会基线**：定义井级/扰动级双向检索及随机基线的解析公式，支持严格的结果可比性。
   - 与 prior work 的区别：多数方法仅报告单一指标；本文提供四种检索设置及对应的 chance baseline，评估更为系统。
5. **复现工程与开源**：代码、配置、数据分割清单和评估工具链全部开源，复现门槛低。

## 方法详解
### 图像编码器
- **通道处理**：Cell Painting 五通道（MitoTracker、Phalloidin、WGA、Concanavalin A、Hoechst）各自 resized 至 224×224，复制为三通道后输入冻结的 DINOv3 ViT-L/16，每个通道输出一个 1024 维 CLS token（式2）。
- **CrossChannelFormer (CCF)**：单层四头 Transformer（式3），每个 token 做 L2 归一化后加上可学习的通道嵌入 e_c，再经一个可学习的聚合 token q 做 cross-attention，输出即为位点表示 h_site。此模块包含约 14M 参数中的大部分。
- **Site-to-well pooling**：对井内所有有效 site token 做 mask 平均，得到井级向量 h_img。
- **Image projection head**：两层 MLP（LayerNorm + GELU + Dropout p=0.3），输出 512 维 L2 归一化向量（式4）。

### 文本编码器
- **模板化提示**：为化合物、CRISPR KO、ORF 过表达、DMSO 阴性对照各设计结构化 prompt（含 SMILES、靶基因、基因功能、扰动模态等），缺失字段填"unknown"。同一扰动的所有重复井共享同一文本向量。
- **Encoding**：冻结的 BioClinical ModernBERT（150M 参数，768 维 CLS token）编码 prompt，缓存后由投影头映射至 512 维共享空间（式5）。

### 训练目标
1. **Continuously Weighted Contrastive Loss (CWCL)**（式6）：在 batch 内构建相似度矩阵 S = τ·F_img·F_txt^T，亲和矩阵 W 赋权：相同扰动为 1，共享靶基因的异扰动为 α，其余为 0。软标签对比损失比标准 InfoNCE 更宽容，适合生物相似性的灰度性。温度 τ 通过对数参数化学习，初始 ln(1/0.07)，裁剪至 [0, ln 100]。
2. **Replicate Alignment Loss L_rep**（式7）：图像-图像对比项，仅在同一扰动有 In-batch 重复井时参与梯度计算，与 CWCL 共享温度 τ，权重 λ_rep=0（基础）或 0.3（消融）。
3. **Perturbation-aware batching**：采样器将同一扰动井组成 pair，跨板打包入 batch，确保 L_rep 有效。
4. **Condition-relative plate offsets**（式8）：每 epoch 梯度-free 计算每板均值 f̄_p，偏移 δ_p = f̄_p - mean({f̄_q : q ∈ C(p)})，使校正后的嵌入去除板间漂移但保留条件均值。

### 优化设置
AdamW，lr=10^-4，β=(0.9, 0.999)，weight decay=0.2，warmup 100 步后 cosine decay 至 0，FP16 混合精度，gradient clipping=1.0，batch size=256 wells。基础模型训练 100 epoch，消融 30 epoch + early stopping。

## 实验与结果
- **数据集**：CPJUMP1 pilot，303 compound + 160 CRISPR KO + 176 ORF overexpression，U2OS/A549 细胞系，双时间点；五通道荧光（不含 brightfield）。
- **划分**：按扰动哈希 80/10/10 拆分，验证集 98 扰动/2220 井，测试集 86 扰动/1860 井，控制井不参与训练与检索评估。
- **评估协议**：四种检索设置（well/perturbation × i→t/t→i），报告 Recall@{1,5,10} 和中位排名；同时跑标准 CPJUMP1 benchmark（mAP + fraction retrieved）。
- **基础模型检索结果**（表 II）：扰动级 i→t R@10=37.2~38.8%，t→i R@10=39.8~44.2%，中位排名 ~13-14；随机基线约 10% 左右，提升约 3-4 倍。
- **消融结果**（表 III）：单因子均未在所有指标上一致优于控制；复本对齐 loss 使 Validation i→t R@10 从 37.8→44.9%，Test t→i 从 41.9→46.5%。
- **CPJUMP1 标准 benchmark**（表 IV）：复本对齐显著提升 replicability mAP：Base 0.298 → Replicate 0.343 → All three 0.369；但 target matching 和 gene-compound matching 无一致改进，gene-compound 几乎为零（fraction retrieved ~0.005-0.029）。
- **与 CellCLIP 对比**（表 V，共享 5 条 short-track）：Base 接近，+replicate loss / All three 在各 track 上多数优于 CellCLIP（mean 0.247/0.287 vs 0.195）。
- **最强结果**：Combined model (soft labels + replicate loss + plate offsets) 在 CPJUMP1 短时间点 replicability 上达到最高 fraction retrieved（Compound A549 long 0.703，mean 0.287）。

## 相关工作脉络
1. **CellProfiler [6]**：手写特征基线，仍在 CPJUMP1 compound target matching 上有竞争力（FR 4.3-25.1%）；MorphoCLIP base 覆盖更广（7.5-31.5%）且支持基因扰动。
2. **CLOOME [7]** / **MolPhenix [8]**：对比学习用于 Cell Painting，但仅支持化合物、用分子指纹或专有骨干；本文统一处理化合物+基因+文本。
3. **CellCLIP [9]**：最接近前作，用 DINOv2-g + 文本 prompt + 软标签对比损失；本文使用更新的 DINOv3 + ModernBERT，冻结骨干降低算力需求，并新增复本对齐和板校正。
4. **CWA-MSN [10]**：Masked Siamese + 跨孔对齐解决批次效应，提升 gene-gene 检索；本文借鉴其 replicate alignment 思路但加入文本分支实现跨模态检索。
5. **DINOv2/v3 [12,14]**：自然图像自监督视觉基础模型已证明可迁移至生物成像；本文直接利用 DINOv3 gram anchoring 特性获取高质量 channel-wise 特征。
6. **Channel ViT [15]**：per-channel encoding + cross-channel attention 已被证明适合多通道显微镜图像；本文 CrossChannelFormer 沿用该设计范式。

## 局限性与未来方向
1. **基因-化合物跨模态匹配失败**：所有变体在 gene-compound matching 上几乎无阳性结果，提示当前表征尚未捕获通路级共享机制。
2. **消融差异小且不稳定**：单次实验中变体间差距仅几个百分点，且排序随 split 变化，统计效力不足。
3. **板校正与条件强耦合**：CPJUMP1 中每板绑定了特定 cell line / timepoint / perturbation type，条件相对校正虽保留条件均值，但仍可能不足以解耦技术变异。
4. **标签噪声与定义不一致**：训练用的 primary-target 注释与 benchmark 的 broader target list 存在 mismatch，可能限制 gene-compound 学习。
5. **Prompt 信息贡献未充分验证**：SMILES、基因描述、功能等字段的具体作用尚待消融。
6. **泛化性待测**：仅在 U2OS/A549 和 JUMP pilot 上评估，未测试到更大 JUMP corpus 或其他细胞系。

## 研究启发与可借鉴点
1. **冻结骨干+缓存特征策略**：对"数据量大但标注有限"的生物成像任务极具参考价值；DINOv3 特征缓存可复用至下游对比/检索任务，显著降低训练成本。
2. **条件相对板校正框架**：相比全局减法，条件内相对偏移保留了生物学信号，适合"板-条件强耦合"的实验设计，可迁移至其他高内涵筛选数据集。
3. **复本对齐损失与扰动感知采样**：在对比学习中显式建模实验重复结构，可将"技术重复一致性"纳入优化目标，适用于任何含 replicate 的组学/成像数据。
4. **软标签连续加权对比学习**：基于靶基因重叠度赋权的亲和矩阵 W 能刻画生物相似度的灰度性，可推广至药物-靶点、基因-通路等带层次关系的检索任务。
5. **可复现工程实践**：完整开源代码+配置+split manifest+评估 harness，为生物 AI 社区提供低门槛复现模板。

## 关键术语表
**Cell Painting**：一种五通道荧光高内涵显微成像 assay，通过标记线粒体、肌动蛋白、高尔基体/质膜、内质网和 DNA，量化细胞形态学表型。
**CPJUMP1**：JUMP Cell Painting Consortium 发布的公共基准数据集，含 136,000+ 化学/基因扰动下的 Well 级图像，用于评估形态学分析模型。
**CrossChannelFormer (CCF)**：MorphoCLIP 中的单层四头 Transformer，负责将五个荧光通道的 DINOv3 token 融合为一个位点表示。
**Continuously Weighted Contrastive Loss (CWCL)**：基于亲和矩阵 W 的软标签对比损失，对共享靶基因的扰动赋予中间权重 α，替代二值正负样本。
**Replicate Alignment Loss (L_rep)**：在图像嵌入空间中拉近同一扰动重复井的对比项，显式优化实验重复性。
**Condition-relative plate offsets**：每板相对于同条件其他板均值的偏移量，用于校正板间技术漂移而不抹除条件信号。
**BioClinical ModernBERT**：在生物医学与临床文献上预训练的长上下文 BERT 变体，用于编码化合物/基因文本提示。
**DINOv3**：带 gram anchoring 的自监督视觉 transformer，本文使用 ViT-L/16 变体作为冻结的图像骨干。

## 可复现要素
- **数据集**：CPJUMP1 pilot（公开），JUMP Cell Painting Consortium 提供。
- **代码/权重**：代码、配置、split manifest、评估 harness 均已开源，GitHub: https://github.com/suxrobGM/morphoclip；DINOv3 ViT-L/16 和 BioClinical ModernBERT 为开源模型，需自行下载。
- **关键超参**：lr=1e-4，batch size=256，weight decay=0.2，CCF 单层 4-head，projection hidden=512，dropout=0.3，CWCL α=0（基础）/0.6（软标签），λ_rep=0（基础）/0.3（消融），τ 初始 ln(1/0.07) 裁剪至 [0, ln 100]，warmup 100 steps + cosine decay，训练 100 epoch（消融 30 epoch）。
- **硬件**：单卡 RTX 5080 (16 GB)。
- **随机种子**：seed=42。
