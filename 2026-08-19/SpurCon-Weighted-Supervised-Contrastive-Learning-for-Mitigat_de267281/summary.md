---
title: "SpurCon-Weighted-Supervised-Contrastive-Learning-for-Mitigat"
source: https://arxiv.org/pdf/2608.17598v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 06:09:58"
field: "医学影像鲁棒性与公平性学习"
keywords: ["spurious correlation mitigation", "weighted supervised contrastive learning", "medical imaging", "few-shot attribute prediction", "worst-group accuracy", "foundation model adaptation"]
innovations: ["基于预训练编码器原型的 few-shot 虚假标签预测机制", "融合元数据与预测虚假标签的加权监督对比损失 WtSupCon", "冻结主干仅训练轻量投影头的低开销训练范式"]
benchmarks: ["Waterbirds", "CheXpert - Pneumothorax", "ISIC 2020", "Synthetic Toy Dataset"]
---

# 论文速读：SpurCon-Weighted-Supervised-Contrastive-Learning-for-Mitigat

## 一句话总结
本文提出了 **SpurCon**，一种基于加权监督对比学习（WtSupCon）的轻量级框架，通过 few-shot 预测的虚假标签与可用元数据（如患者 ID）相结合，重塑图像表示几何结构，从而在医学影像等场景中有效缓解模型对虚假相关（spurious cues）的依赖，显著提升最坏群体准确率（WG）并大幅降低训练开销。

## 研究问题与动机
1. **医学影像中虚假相关严重威胁模型鲁棒性**：深度学习虽在视觉识别上进展迅速，但在高风险医疗应用中，模型常利用设备伪影、治疗标记等与病理非因果共现的虚假线索，导致 subgroup 泛化能力下降、临床信任受损。
2. **现有方法难以兼顾虚假属性识别与利用基础模型语义**：传统方法依赖领域知识标注虚假属性或训练辅助分类器，无法充分利用现代 foundation model（如 BiomedCLIP）嵌入的丰富语义；且 prior contrastive approaches（如 CA、Correct-n-Contrast）主要针对自然图像基准设计，在医学数据集上表现不佳。
3. **小样本/不平衡数据加剧 worst-group 性能退化**：医疗数据集往往规模有限且类别/群组分布极度倾斜，现有正则化或重加权方法难以在保持平均准确率的同时有效改善最坏群体的识别性能。
4. **缺乏对元数据价值的系统化利用**：医学影像常附带患者 ID、设备信息等元数据，可反映视觉相似性分组，但现有工作未将其有效融入 spurious mitigation 流程。

## 核心贡献（创新点）
1. **提出 Pick-and-Predict few-shot 虚假标签估计机制**：仅需少量（每类别×虚假标签组合 10 例）专家标注样本，利用预训练图像编码器提取原型并基于余弦相似度分配预测虚假标签，无需额外网络训练。
2. **设计加权监督对比损失 WtSupCon**：将正样本对按 `[同 ID & 同类别 & 不同虚假]`、`[同 ID & 同类别 & 同虚假]`、`[仅同类别]` 划分至 3 个集合，并分配递减权重（实验优选 `(α₁,α₂,α₃)=(4,2,1)`），显式鼓励相同病理与元数据但虚假属性不同的样本在表示空间中靠近。
3. **构建仅训练轻量投影头的低开销训练范式**：冻结预训练图像编码器（CLIP/BiomedCLIP），仅优化下游投影头与 logistic regression 分类器，在 Waterbirds、CheXpert、ISIC 2020 上实现比 JTT/CA/DFR 快约 5–9 倍的总训练时间。
4. **引入自适应分组采样器**：针对存在 patient ID 等分组信息的医学数据集设计 `ID-paired sampler` 与 `Balanced-groups sampler`，提升 mini-batch 内正样本对的覆盖率与分布均衡性。
5. **系统验证跨域虚假缓解有效性**：在合成 toy dataset、自然图像 benchmark（Waterbirds）及两个医学影像数据集（CheXpert 气胸分类、ISIC 2020 皮肤癌分类）上均取得最优或最佳平衡的 WG/Avg/AUC 性能。

## 方法详解
### 整体流程（Algorithm 1）
输入冻结编码器 E、投影头 H、少量专家标注集 Eₓ、训练集 Tr、验证集 Val、测试集 Te，输出训练好的投影头 Ĥ 与 logistic regression 分类器。

### 2.1 Pick-and-Predict：few-shot 虚假标签预测
- 对每个虚假标签 s∈{1,…,S}，从专家标注集中选取 N 个样本（均匀覆盖所有 (class, spurious) 组合），计算其编码器嵌入均值作为原型：  
  **μ⁽ˢ⁾ = (1/N) Σᵢ₌₁ᴺ zᵢ⁽ˢ⁾**
- 对任意新样本 xⱼ，编码得 zⱼ，预测虚假标签为余弦相似度最大的原型对应类别：  
  **ŝⱼ = argmaxₛ ⟨zⱼ, μ⁽ˢ⁾⟩**
- 该步骤无需梯度更新，仅依赖预训练编码器的前向传播，标注效率极高（每虚假标签仅需 10 例）。

### 2.2 Weighted-SupCon：减少虚假属性影响
- 基础 supervised contrastive loss（SupCon）：  
  **ℒᴬᶜ = Σᵢ [−(1/|P(i)|) Σₚ∈P(i) ℓᵢ,ₚ]**，其中 ℓᵢ,ₚ = log[exp(⟨zᵢ,zₚ⟩/τ) / Σₐ∈A(i) exp(⟨zᵢ,zₐ⟩/τ)]
- 改进为 **WtSupCon**：将正样本集 P(i) 划分为 K=3 个互斥子集 {Pₖ(i)}ₖ₌₁ᴷ，每个子集分配权重 αₖ>0：  
  **ℒᵂˢᶜ = Σᵢ [ Σₖ (−αₖ/|Pₖ(i)|) Σₚ∈Pₖ(i) ℓᵢ,ₚ ]**
- **三组划分定义（Tab. 1）**：
  - **Set 1**：同 ID、同类别、**不同**虚假标签 → **最高权重 α₁**（核心去偏对）
  - **Set 2**：同 ID、同类别、**相同**虚假标签 → 中等权重 α₂
  - **Set 3**：**仅同类别**（不同 ID 或不同虚假）→ 最低权重 α₃
- 权重还需补偿各组样本数量不平衡（较小集合获得较大权重），进一步凸显跨虚假属性的正样本对。
- **投影头结构**：三种配置可选（LayerNorm–GeLU MLP with residual / BatchNorm–ReLU MLP with bottleneck / 更浅变体），仅训练该头，编码器冻结。
- **采样器设计**：
  - `ID-paired sampler`：将同患者 ID 样本倾向放入同一 mini-batch，提高 Set 1/2 的共存概率（适用于 Waterbirds、toy dataset）。
  - `Balanced-groups sampler`：优先保证每个 batch 覆盖更多 (class, spurious) 群组，缓解极端不平衡（适用于 CheXpert、ISIC 2020）。
- **元数据利用**：医学影像常用 patient ID 作为 grouping key；若无元数据，可退化至仅依赖类别与预测虚假标签的两集合版本。

## 实验与结果
### 数据集与实现
- **Toy dataset**：二分类（单孔/双孔），虚假属性为背景条纹/圆点，含 100 种颜色 ID，训练 7k/验证 1k/测试 2k，训练集严重不平衡 (0,0) 与 (1,1) 各占 47.5%。
- **Waterbirds**：水鸟/陆鸟，背景为水/陆，以物种路径解析为 ID。
- **CheXpert - Pneumothorax**：气胸二分类（No Finding vs Pneumothorax），Support Devices 列作为虚假属性，共 36,679 张 X 光片，训练组内极度不平衡（≈98%/2%）。
- **ISIC 2020**：皮肤癌良恶性二分类，结合 metadata 构建 ruler presence 虚假属性与 patient ID，共 32,692 张裁剪至 256×256 图像。
- **编码器**：CLIP（d=768）用于自然图像，BiomedCLIP（d=512）用于医学图像；全部冻结。
- **分类头**：scikit-learn Logistic Regression（lbfgs solver），CheXpert/ISIC 使用基于预测虚假标签组的 inverse-frequency 样本权重。
- **权重超参**：统一使用 **(α₁,α₂,α₃)=(4,2,1)**；温度 τ、隐藏层宽 h、dropout dp 按数据集配置。

### 主要结果（Table 2）
- **Waterbirds**：SpurCon **WG=87.1±0.8**（最优）、Avg=94.5±0.4、Adj.Avg=97.5±0.0，全面超越 JTT/DFR/CA；AUC 略低于 DFR（97.4 vs 97.8）。
- **CheXpert**：SpurCon **WG=73.0±0.6**、Avg=80.7±0.3、AUC=88.9±0.2，各项均为第一；次优 Baseline-WT 的 WG 仅 71.0。
- **ISIC 2020**：SpurCon **WG=65.0±3.8**、Avg=80.3±0.9，在保持良好平均准确率的同时显著优于所有对比方法（Baseline WG=0.0，JTT Avg 虽高但 WG 仅 51.0）。
- **统计稳定性**：所有结果报告 5 次随机种子的均值±标准差；权重组合在 15 种候选中变化平缓（Fig. 5），表明对超参不敏感。

### 效率对比（Table 3，ISIC 2020）
- SpurCon 单 epoch 耗时 **7.9±0.6 s**，总训练时间 **497±36 s**；较 JTT/CA 快约 5.5 倍，较 DFR 快约 9 倍，体现“轻量投影头 + 冻结编码器”的优势。

### 关键消融与鲁棒性
- **Few-shot 标注量**（Fig. 4）：每虚假标签仅 10 例（共 20 例/标签）即可达到性能 plateau，进一步增加标注收益边际递减。
- **预测准确率**：Waterbirds AUC=95.4%、CheXpert=97.3%、ISIC=80.9%，证明原型匹配有效。
- **噪声鲁棒性**（Tab. 4）：即使在专家标注或预测标签中引入 10%–20% 噪声，WG 仍显著高于 Baseline-WT，表明方法对虚假标签估计误差具有容忍度。
- **WtSupCon 权重关系**（Tab. 5）：消融实验证实 α₁ 是驱动 WG 提升的主因；完整三权重配置（Ablation 4）在 toy dataset 上获得最高 WG=80.7±2.0。

## 相关工作脉络
1. **Just Train Twice (JTT)**：通过两阶段重加权训练提升 worst-group 性能，无需显式群组标签；SpurCon 明确利用预测虚假标签与元数据进行对比表示学习，提供更强的几何对齐能力且训练更快。
2. **Deep Feature Reweighting (DFR)**：在特征空间进行分布重加权以缓解 spurious correlation；SpurCon 不依赖特征重加权，而是通过 WtSupCon 直接塑造表示空间分布，且能整合结构化元数据。
3. **Contrastive Adapters (CA)**：针对自然图像基准设计，添加可学习适配器并进行对比训练；本文指出其在医学数据集上表现较差，而 SpurCon 针对医学虚假属性（如器械伪影、背景）重新设计权重分配与采样策略。
4. **Correct-n-Contrast**：无监督估计虚假属性并施加对比损失；SpurCon 改进为先 few-shot 预测虚假标签（更准确），再结合 patient ID 等元数据细化正负样本对划分，弥补前者在医学场景的不足。
5. **Sparsifying Spurious Attributes (Spread SA)**：基于辅助分类器自动标注虚假属性；SpurCon 摒弃训练专用分类器，转而利用 foundation model 嵌入的语义原点进行 zero/few-shot 匹配，减少额外训练开销。
6. **MedAug**：利用患者元数据进行对比学习的医学数据增强方法；SpurCon 进一步将元数据引入对比损失的权重设计，以专门缓解 spurious correlation 而非泛化增强。

## 局限性与未来方向
1. **依赖少量专家标注进行原型构建**：若虚假属性完全未知或无法提供代表性示例，few-shot 估计可能失效；未来需探索全自动虚假属性发现或自监督原型学习。
2. **当前仅适用于离散型虚假属性**：如背景类型、器械有无等；连续型或细粒度伪影（如不同强度的伪影等级）尚未涉及，需推广至序数/连续标签设定。
3. **元数据可用性假设较强**：Patient ID 在多数医院数据集中可得，但跨机构联邦场景或隐私保护限制下可能不可用；需研究无元数据时的退化策略或替代分组信号。
4. **未扩展至其他医学 AI 任务**：本文聚焦图像分类，对于目标检测、语义分割、风险预测等下游任务，spurious mitigation 的需求与方法适配性有待探索。
5. **多虚假属性交互处理未深入**：实验仅考虑单一二值虚假属性；实际医学影像中常存在多个交织伪影（设备、背景、扫描参数等），如何联合建模与去偏是开放问题。

## 研究启发与可借鉴点
1. **Few-shot 原型匹配估计隐性属性**：无需训练专用分类器，仅用少量标注样本计算嵌入均值作为原型，即可高效推断样本的隐藏属性标签；该思路可迁移至其他需要 attribute inference 的领域（如图像风格、设备型号、拍摄条件）。
2. **权重化对比损失按语义差异分层**：将正样本对按 `[共 ID][共虚假][仅共类别]` 三级粒度划分并赋权，使去偏目标显式化；这一分层思想可推广至 fairness-aware contrastive learning 或 subgroup robustness 场景。
3. **元数据驱动的正样本构造策略**：利用 patient ID、扫描时间序列等结构化信息定义“视觉相似但病理/属性不同”的样本对，为医学影像表示学习提供了新的特征工程范式。
4. **定制化分组采样器应对分布倾斜**：针对含分组键（ID）和数据倾斜的双重挑战，分别设计 `ID-paired sampler` 与 `Balanced-groups sampler`；该采样设计模式可直接复用至其他带层级结构的不平衡视觉数据集。
5. **冻结 backbone + 轻量投影头的低开销训练**：保留 foundation model 冻结状态，仅训练末尾投影层与线性分类器，可在保证预训练知识充分利用的同时将训练时间压缩一个数量级；适合资源受限的医疗 AI 部署流程。

## 关键术语表
- **Spurious cues / Spurious correlations**：输入中与目标任务统计相关但非因果的特征线索（如医疗设备、背景纹理），模型易过度依赖此类捷径而损害 subgroup 泛化。
- **Worst-group accuracy (WG)**：按 (类别, 虚假属性) 划分的各子群中最低准确率，用于量化模型对欠代表 subgroup 的鲁棒性。
- **Supervised Contrastive Learning (SupCon)**：利用类别标签将同类样本嵌入拉近、异类样本推远的对比损失，增强类别判别性表示。
- **Weighted-SupCon (WtSupCon)**：本文提出的广义 SupCon，将正样本集划分为 K 个子集并分配不同权重 αₖ，以差异化调控各类样本对对损失贡献的影响。
- **Pick-and-Predict**：一种 few-shot 虚假标签估计流程，通过专家标注样本的嵌入均值构建原型，再以余弦相似度为未标注样本预测虚假标签。
- **Patient ID**：医学影像中的患者标识元数据，可聚合同一患者的多次扫描，作为视觉上相似样本的分组键。
- **Projection head**：接在预训练图像编码器后的轻量级 MLP，用于将高维嵌入映射至对比学习所需的低维空间，仅训练该头可大幅节省计算资源。
- **ID-paired / Balanced-groups sampler**：两种面向不同数据结构的 mini-batch 采样策略，前者倾向同 ID 样本共存以提升正样本对质量，后者优先保证各 (类别,虚假) 组在 batch 中均衡出现以缓解长尾分布。

## 可复现要素
- **数据集**：Waterbirds（公开）、CheXpert（公开）、ISIC 2020（公开）；Toy dataset 由作者生成，代码/数据生成脚本未在论文中提供开源链接（论文未提及是否开源）。
- **代码/权重**：论文未声明代码开源仓库；预训练编码器使用公开模型 CLIP（openai）与 BiomedCLIP（NEJM AI）。
- **关键超参**：
  - 投影头权重：**(α₁, α₂, α₃) = (4, 2, 1)**
  - 温度系数 τ：论文未明确给出具体数值
  - 批量大小：Waterbirds=256，ISIC 2020=128，CheXpert=512
  - 投影头配置：CheXpert/ISIC 使用 LayerNorm–GeLU MLP with residual（h=256/512, dp=0.35/0.3）；Waterbirds/toy 使用 BatchNorm–ReLU MLP with bottleneck（h=256/128, dp=0.3）
  - 分类器：scikit-learn LogisticRegression（lbfgs solver, max_iter=5000），CheXpert/ISIC 使用 inverse-frequency 样本权重，C=0.01（Waterbirds C=0.001）
  - Few-shot 标注量：每虚假标签 **N=10** 例专家标注样本
  - 训练 epoch 数：Waterbirds/ISIC 未明确；CheXpert 使用 StratifiedGroupKFold 调参
  - 早停策略：以验证集 WG 最高 epoch 的中位数作为最终 refit epoch（Algorithm 1 第 14–15 行）
