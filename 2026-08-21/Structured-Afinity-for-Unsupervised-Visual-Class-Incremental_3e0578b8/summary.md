---
title: "Structured-Afinity-for-Unsupervised-Visual-Class-Incremental"
source: https://arxiv.org/pdf/2608.20104v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 16:34:05"
field: "持续学习/类增量学习"
keywords: ["artificial immune networks", "class-incremental learning", "structured affinity", "ZNCC", "visual memory", "binding profile", "deep AIN"]
innovations: ["提出移位模板与ZNCC结构化免疫亲和算子使B细胞成为空间模板", "定义Feature-Map Deep AIN通过响应图谱组合实现无标签梯度-free深度免疫记忆", "证明无回放条件下免疫绑定-profile空间可被外部探针恢复类结构"]
benchmarks: ["Sklearn Digits", "MNIST", "Fashion-MNIST", "KMNIST"]
---

# 论文速读：Structured-Afinity-for-Unsupervised-Visual-Class-Incremental

## 一句话总结
本文提出将人工免疫系统（AIN）的亲和算子从展平向量距离改造为兼容图像空间结构的 ZNCC 滤镜匹配，并保留响应图谱形成 Feature-Map Deep AIN，在无回放、无标签更新、无反向传播的条件下，实现了类增量视觉记忆空间的学习与可恢复性。

## 研究问题与动机
1. 传统视觉 AIN 将图像展平为向量，亲和算子（如标准 RBF）忽略空间局部性与平移不变性，导致免疫记忆机制在视觉任务上显得乏力。
2. 类增量视觉学习的核心矛盾：新类别到来时，免疫 repertoire 需同时完成记忆获取与新表征适应，如何避免灾难性遗忘仍是开放问题。
3. 现有 AIN 研究多面向有监督分类或聚类，缺乏"以免疫绑定-profile 空间作为在线视觉表征本身"的系统性论证。
4. 需要一种可诊断的方法，分离"记忆槽增长"与"低层表征适应"的各自作用，判断何时低层适应有益、何时造成不稳定。

## 核心贡献（创新点）
1. **结构化视觉免疫亲和算子**：提出 Shifted-Template Afinity 与 ZNCC 卷积式亲和，本质区别在于将 B-cell 从"向量空间点"变为"局部空间模板"，使免疫响应与图像结构对齐。
2. **Feature-Map Deep AIN 深度定义**：深度来源于前层响应图谱（binding-profile maps）直接作为后层抗原输入，而非在免疫层外挂载分类器，本质区别于普通 RBF 堆叠。
3. **无标签的在线类增量视觉记忆协议**：免疫更新不使用类别标签，类别仅用于外部探针评估，将"表征学习"与"判别验证"完全解耦。
4. **适应性坐标重组织的记忆保持论证**：通过每步重新嵌入旧样本验证当前 repertoire 仍保留旧类结构，而非依赖静态坐标不变性。
5. **深度与层间尺度诊断**：揭示更高层性能下降往往源于亲和尺度失配，而非深度本身失效，提出 NN-distance 自适应尺度规则。

## 方法详解
- **总体架构**：两层 Feature-Map Deep AIN，第1层接收图像 $X$，用 $m_1=30$ 个 $k\times k$ 局部 B-cell 滤镜计算 ZNCC 响应图，得到 $R^{(1)}(X)\in\mathbb{R}^{m_1\times h_1\times w_1}$；第2层接收响应图堆叠，用 $m_2=30$ 个 $2\times2$ 滤镜学习响应模式记忆，输出 $R^{(2)}(X)$，最终嵌入 $z(X)=\mathrm{vec}(R^{(2)}(X))$。
- **ZNCC 亲和**：局部图像块 $P$ 与滤镜 $b$ 的零均值归一化互相关：
  $$\mathrm{ZNCC}(P,b)=\frac{\langle P-\bar{P},b-\bar{b}\rangle}{\|P-\bar{P}\|\|b-\bar{b}\|}$$
  映射到 $[0,1]$：$K(P,b)=\frac{\mathrm{ZNCC}(P,b)+1}{2}$，滑动生成响应图后取最大值得标量亲和 $K_{\mathrm{filter}}(X,b)=\max_{P} K(P,b)$。
- **深度组合公式**：$R^{(\ell)}(X)=\Phi_{\mathcal{B}^{(\ell)}}(R^{(\ell-1)}(X))$，$R^{(0)}(X)=X$，深度即免疫响应的迭代组合。
- **在线免疫更新**：采用克隆选择式 repertoires 训练（克隆、突变、抑制），不使用类别标签；新类别批次到来时更新下层视觉 repertoire，同时 binding-profile 坐标系自适应变化。
- **层间尺度自适应**：高层亲和尺度基于该层响应空间中最近邻距离估计，乘固定倍数（实验使用 $\times6$）保持跨数据集一致性。

## 实验与结果
- **数据集**：Sklearn digits（$8\times8$）、MNIST、Fashion-MNIST、KMNIST（均为 $28\times28$），均按类增量流处理（类别0初始化，1-9顺序到达）。
- **评估基线**：Static/Mini-batch/Full refit k-means、Online prototype memory、No-replay MLP、Replay MLP。
- **Sklearn digits（最终10类）**：
  - Feature-map Deep AIN + Logistic probe：**Balanced accuracy=0.939**，Initial retention=0.978，Current-class=0.933
  - Feature-map Deep AIN + 1NN probe：**Balanced accuracy=0.902**，Initial retention=0.978，Current-class=0.833
  - 自适应尺度校准后两层层 Deep AIN + Logistic：Balanced accuracy 达 **0.978**
- **跨数据集对比（同自适应尺度规则）**：
  - MNIST：Logistic=**0.862**，1NN=0.857；Replay MLP=0.858
  - Fashion-MNIST：Logistic=**0.814**，1NN=0.801；Replay MLP=0.780（Deep AIN 超越 Replay MLP）
  - KMNIST：1NN=**0.853**（最强），Logistic=0.711；Replay MLP=0.757（1NN 显著超越 Replay MLP）
- **KMNIST 探针几何诊断**：1NN（0.853）> PCA+RBF-SVM（0.818）> Logistic（0.711）> Nearest-centroid（0.612），说明绑定-profile 空间局部相干但非全局线性可分。
- **结论**：无回放、无反向传播的 Feature-map Deep AIN 在四个数据集上与 Replay MLP 相当或更强；保留响应图谱比标量亲和显著提升性能。

## 相关工作脉络
1. **传统 AIS（AIRS、aiNet、克隆选择）**：面向有监督分类或聚类，使用向量特征且非分层视觉记忆，本文将其定位为"非梯度、非监督、层级视觉表征"的新方向。
2. **原型/核方法（RBF、LVQ、SOVM、Dissimilarity Representation）**：均以参考集构建表示，但本文用免疫适应动态更新参考集，并耦合记忆与表征。
3. **结构化视觉匹配（SIFT、HOG、ZNCC）**：提供局部空间匹配的技术渊源，但本文将其植入免疫亲和框架并做在线记忆扩展，而非传统手工特征。
4. **深度卷积神经网络**：共享"局部滤镜+层叠"的操作直觉，但 CNN 靠反向传播优化权重，本文靠免疫克隆-突变-抑制驱动，目标为记忆而非判别分类。
5. **增量/持续学习（EWC、iCaRL、GEM、Replay MLP）**：依赖梯度正则化或旧样本回放，本文以免疫 repertoire 替代回放缓冲区，核心关注绑定-profile 空间的结构保持而非判别误差。
6. **Prototype-Response 表示学习**：相关思路存在已久，本文贡献在于免疫视角的实现（在线更新、无标签驱动、梯度-free）。

## 局限性与未来方向
1. **局部滤镜歧义**：小局部滤镜在不同类共享笔画/边缘时产生相似响应，导致全局判别力不足（KMNIST 线性探针较弱即为此体现）。
2. **Repertoire 膨胀与冗余**：记忆增长带来维度与计算成本上升，且缺乏有效的种群控制/冗余抑制机制。
3. **深度与尺度的敏感性**：第三层在现有配置下因过度压缩表现下降，需更精细的 receptive field 与容量匹配策略。
4. **视觉域受限**：仅在灰度基准数据集验证，未扩展至 RGB 自然图像（颜色、背景、纹理变化），属概念验证阶段。
5. **未来方向**：引入自适应种群控制；设计更强的非线性免疫读取器；探索带旋转/尺度不变性的结构化亲和扩展；与元学习/小样本场景结合。

## 研究启发与可借鉴点
1. **"响应图谱保留"原则**：在免疫/核方法中，推迟亲和标量化、保留空间响应图能显著提升后续层的信息容量，可迁移至其他模板匹配型在线表示学习。
2. **探针-模型分离的验证范式**：免疫层完全不接触标签，仅由下游探针验证表示质量，这一分离设计值得借鉴于无监督/自监督表征的可解释性评估。
3. **层间尺度自适应规则**：用 NN-distance 估计高层亲和尺度，避免手工调参，可推广至任何多层核/模板匹配的在线学习系统。
4. **稳定性-可塑性诊断框架**：将"初始保留"与"当前类习得"分开度量，并用多探针几何诊断（线性/非线性/局部）评估表示结构，为后续增量记忆研究提供标准分析工具。
5. **免疫亲和的计算机视觉化**：将 ZNCC 等经典 CV 算子嵌入免疫更新循环，为"免梯度视觉学习"提供了一种跨领域的方法融合思路。

## 关键术语表
- **Artificial Immune Network (AIN)**：受免疫系统启发的计算模型，通过抗体/ B细胞群体、亲和评估、克隆突变与抑制实现自适应记忆。
- **Class-Incremental Learning**：样本按类别顺序到达的在线学习设置，模型需在不访问旧数据的前提下持续学习新类并保留旧类能力。
- **Structured Afinity**：考虑抗原（图像）空间结构的亲和算子，如 ZNCC 局部匹配或移位模板匹配，区别于展平向量的欧氏/RBF 距离。
- **ZNCC（Zero-Normalized Cross-Correlation）**：零均值归一化互相关，对光照/对比度变化鲁棒的模板匹配度量，值域 $[-1,1]$ 后映射到 $[0,1]$ 作为亲和分数。
- **Binding Profile**：抗原相对于整个 B细胞 repertoire 的亲和响应向量（或图谱集合），构成免疫诱导的坐标化表征。
- **Feature-Map Deep AIN**：保留每层 B细胞的空间响应图谱而非压缩为标量，将前层响应图堆叠作为后层输入的深层免疫架构。
- **Adaptive Coordinate Reorganization**：新类到来后免疫系统更新导致绑定-profile 坐标系变化，但通过重嵌入验证旧类仍可恢复的结构保持机制。

## 可复现要素
- **数据集**：Sklearn digits、MNIST、Fashion-MNIST、KMNIST（均为公开标准数据集）。
- **代码/权重**：论文未提及开源代码或预训练权重。
- **关键超参**：Layer-1 滤镜数=30，滤镜尺寸 $5\times5$（$28\times28$ 数据）/对应 $8\times8$ 缩小比例；Layer-2 滤镜数=30，滤镜尺寸 $2\times2$；每批初始样本 80，新类样本 60，测试样本每类 35；自适应尺度因子 $\times6$；随机种子 3 次。
