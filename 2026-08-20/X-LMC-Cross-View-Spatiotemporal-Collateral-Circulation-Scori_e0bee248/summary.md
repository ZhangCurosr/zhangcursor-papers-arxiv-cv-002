---
title: "X-LMC-Cross-View-Spatiotemporal-Collateral-Circulation-Scori"
source: https://arxiv.org/pdf/2608.18986v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 10:50:08"
---

# 论文速读：X-LMC-Cross-View-Spatiotemporal-Collateral-Circulation-Scori

## 一句话总结
本文提出了X-LMC，首个基于双平面DSA序列自动预测ASITN/SIR侧支循环分级分数的时空深度学习框架；通过冻结DINOv2空间编码、Token级双向跨视角注意力融合与Bi-GRU时序动力学建模，在M1段闭塞卒中队列中实现了与临床专家观察者间一致性相当的客观评分。

## 研究问题与动机
- **临床痛点**：DSA是评估软脑膜侧支循环（LMCs）的金标准，但ASITN/SIR人工分级耗时、高度依赖专家经验，且文献报道的观察者间一致性极低（κ≈0.16~0.27），阻碍了大规模队列分析。
- **技术空白**：现有深度学习侧支评估多依赖CTA/CTP等间接模态，或在DSA上仅完成血管分割/TICI再通评分；直接以原始2D+time双平面DSA为输入进行序数侧支分级的工作尚未出现。
- **模态优势**：双平面DSA提供同步正交投影与高时间分辨率造影剂推进序列，理论上能完整捕捉侧支微血管网的三维灌注动力学，但如何有效融合跨视角与时序信息仍待探索。

## 核心贡献（创新点）
1. **首个DSA侧支循环自动化序数评分框架**：直接处理原始双平面2D+time序列输出ASITN/SIR 0–4级分数。*区别于既往CTA/CTP间接评分或单视角再通评分工作，本文首次将侧支动力学建模为多视角时空联合预测任务。*
2. **冻结DINOv2骨干 + Token级双向跨视角注意力（X-Attn）**：每帧独立提取Token特征后，在时间步级进行AP/Lat双向交叉注意力交互。*区别于传统全局池化后再做卷积/拼接融合的方案，X-Attn保留了细粒度空间Token的交互能力，更适配侧支微小血管结构的表征。*
3. **Bi-GRU时序动力学建模与序数混合损失**：融合序列经双层Bi-GRU聚合前后语境，配合CE+MAE混合损失显式利用等级间距。*区别于DeepTICI的伪灌注自监督与TICI专属损失设计，本文针对序数属性与数据规模进行了轻量化适配。*

## 方法详解
- **输入与目标**：双平面DSA序列 $\mathbf{I}=\{I_{AP}, I_{Lat}\}$，时间维度 $T$，目标为序数标签 $y \in \{0,1,...,4\}$（ASITN/SIR）。
- **多视角空间编码**：每帧 $f_t^{(v)}$ 独立输入冻结的DINOv2 ViT-B/14（$D=768$），经patch划分与Transformer层输出Token序列 $Z_t^{(v)} \in \mathbb{R}^{N \times 768}$；权重复用冻结以防止小样本过拟合。
- **跨视角特征融合**：对每个时间步 $t$ 执行双向Token级X-Attn：
  $\tilde{Z}_t^{(AP)} = \mathrm{X\text{-}Attn}(Z_t^{(AP)}, Z_t^{(Lat)})$，$\tilde{Z}_t^{(Lat)} = \mathrm{X\text{-}Attn}(Z_t^{(Lat)}, Z_t^{(AP)})$
  将两视图更新后的class token相加得融合帧表征 $h_t = \tilde{z}_{t,0}^{(AP)} + \tilde{z}_{t,0}^{(Lat)} \in \mathbb{R}^{768}$，构成序列 $\mathcal{H}=\{h_1,...,h_T\}$。
- **时序动力学建模**：$\mathcal{H}$ 输入单层Bi-GRU（隐藏维度 $H_d=256$），拼接末态前向 $\vec{h}_T$ 与后向 $\overleftarrow{h}_1$ 得到 $h_{seq}$，经MLP分类器输出等级概率。
- **损失函数**：$\mathcal{L} = (1-\alpha)\mathcal{L}_{CE} + \alpha\mathcal{L}_{MAE}$，$\alpha=0.3$；MAE项作用于预测概率向量，显式刻画序数等级间的距离惩罚。
- **预处理与训练**：序列线性插值至2 fps、截断30帧，去基线帧，resize至224×224并归一化至[0,1]；数据增强含翻转、平移、缩放、旋转、对比度扰动；Adam优化，batch size=1，早停于验证集，训练于V100 GPU。

## 实验与结果
- **数据集**：MAGIC多中心队列，134例M1段MCA闭塞AIS患者；参考标签由干预神经放射学专科医师标注，98例双医师独立评分用于计算临床观察者一致性。
- **评估设置**：5-fold交叉验证；主要对比基线为适配DeepTICI架构的时空方法（1×1卷积融合+Uni-GRU，无伪灌注监督）。
- **多分类任务（0–4级，0/1合并）**：X-LMC取得 **QWK=0.398**（最优），ACC±1=84.33%，MAE=0.746；最佳基线QWK=0.322，ACC±1=76.12%，MAE=0.866。
- **二分类任务（良好 vs 较差）**：X-LMC取得 **Kappa=0.430**，mF1=0.711；最佳基线Kappa=0.327，mF1=0.663。
- **临床对标**：该队列临床双观察者间一致性为 **QWK=0.314**；X-LMC点估计略高于人类一致性，处于“公平–中等”区间，表明性能瓶颈部分源于量表主观性而非架构缺陷。
- **可解释性**：Grad-CAM热力图显示良好侧支病例激活集中于远端皮质灌注区，较差侧支激活偏向近端血管，符合ASITN/SIR临床判读逻辑。
- **消融（80/20划分）**：DINOv2冻结骨干优于微调EfficientNet变体；Bi-GRU替换Uni-GRU使QWK从0.397升至0.47
