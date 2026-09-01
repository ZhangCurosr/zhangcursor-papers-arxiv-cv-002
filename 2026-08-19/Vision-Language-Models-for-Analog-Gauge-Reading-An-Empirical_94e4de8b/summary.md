---
title: "Vision-Language-Models-for-Analog-Gauge-Reading-An-Empirical"
source: https://arxiv.org/pdf/2608.17723v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 16:31:22"
field: "工业视觉与多模态学习"
keywords: ["Analog gauge reading", "Vision-language model", "QLoRA", "Parameter-efficient fine-tuning", "Industrial automation", "Reliability assessment", "Cross-domain transfer"]
innovations: ["系统对比零样本/ICL/QLoRA三种学习策略在同一VLM基座上的直接仪表读取性能", "通过独立with-range/without-range模型实现干净的元数据消融并揭示范围信息主要抑制极端失败", "结合LODO迁移评估与确定性扰动鲁棒性分析，识别模糊为主要失效模式并刻画高置信度错误风险"]
benchmarks: ["Synthetic Data for Precision Gauge Reading", "Pressure Gauge Reader Data (video-derived)", "Siemens Energy proprietary industrial dataset"]
---

# 论文速读：Vision-Language-Models-for-Analog-Gauge-Reading-An-Empirical

## 一句话总结
本文系统评估了通用视觉语言模型 Qwen2.5-VL-7B-Instruct 通过 QLoRA 参数高效微调后，直接读取单目标模拟仪表数值的能力；在三个数据集上最佳 MPE 分别为 2.39%（合成）、2.61%（压力仪表）和 4.43%（工业数据集），但跨域迁移与模糊鲁棒性仍存显著不足。

## 研究问题与动机
- 传统模拟仪表读取依赖多阶段流水线（检测→透视校正→指针检测→几何转换），每阶段假设误差会级联放大，且对眩光、遮挡、非标准刻度布局敏感。
- 直接回归方法避免了显式指针几何，但面临数据稀缺和域偏移问题；小模型缺乏足够容量适应多样刻度。
- 通用 VLM 具备强视觉语言基础能力，但并不意味着具备计量可靠性；需要系统回答"专业化程度需多高"及"在何处不可靠"。
- 现有文献中各仪表读取系统使用不同数据集/裁剪策略/实现，难以进行可控跨论文比较，本文聚焦"同一 VLM + 不同训练策略"的系统性消融。

## 核心贡献（创新点）
- **零样本/ICL/QLoRA 的系统性策略对比**：在同一 Qwen2.5-VL-7B-Instruct 基座上，统一 20 epoch 协议对比三种学习范式，而非孤立报告单一方法。
- **干净的范围元数据消融**：独立训练 with-range 与 without-range 两套模型，消除提示设置混淆，揭示范围元数据主要降低极端失败而非显著改变均值误差。
- **LODO + 人工扰动鲁棒性联合评估**：通过留一数据集出与高斯模糊/低光照/遮挡等五类确定性扰动，刻画迁移退化与强失效模式（模糊最严重）。
- **超越 MPE 的综合可靠度指标**：引入 uncapped MPE、尾部误差（P95/Max FS）、操作容差通过率、解析失败率、校准（ECE/AURC）与高置信度错误率，暴露 capped MPE 掩盖的极端失败。
- **可视化归因分析**：结合 blurred-patch 扰动敏感图与 attention rollout，证明注意力集中并不等价于正确数值推理，警示 post-hoc 解释方法的局限。

## 方法详解
- **基座模型**：Qwen2.5-VL-7B-Instruct（总参数约 83 亿，视觉编码器 6.32 亿，projector 4457 万，语言模型 76.16 亿；LoRA 适配参数 2379 万，仅占 0.29%）。
- **QLoRA 配置**：4-bit NF4 量化 + double quantization + bfloat16 计算；LoRA rank r=8、α=16、dropout=0.05，应用于 q/k/v/o/gate/up/down proj；训练加载于双 Nvidia RTX A5500。
- **固定训练协议**：不使用验证集与早停；全部 20 epoch，最终 checkpoint 用于所有分析；batch size=1，梯度累积 8 步，AdamW 学习率 2e-4。
- **数据增强**：随机旋转（±15°, p=0.7）、颜色抖动（亮度 0.3/对比度 0.3/饱和度 0.2/色相 0.1, p=0.8）、高斯模糊（σ=0.1–1.5, p=0.3）、随机透视变换（失真尺度 0.2, p=0.5）；测试图像不加增强。
- **动态分辨率**：总像素范围 256×28×28 至 640×28×28。
- **提示模板**：Value Reading（with-range 含 min/max，without-range 不含）；Scale Probe（JSON 输出最小/最大刻度值，用于视觉端点探测）。
- **主指标**：范围归一化百分比误差 $e_i = \frac{|y_i - \hat{y}_i|}{y_{\max,i} - y_{\min,i}} \times 100\%$，MPE 为 $e_i$ 的均值；报告 capped（上限 100%）与 uncapped 两类 MPE。
- **置信度定义**：$C = \exp\left(\frac{1}{m}\sum_{k=1}^m \log p_k\right)$，即贪心生成 token 概率的几何均值；High-confidence error：$C \geq 0.99$ 且误差 > 5% FS。
- **校准评估**：10-bin ECE、MCE、reliability diagram、risk-coverage curve 与 AURC；操作正确阈值取 ±5% FS（仅为评估容忍度，非计量标准）。
- **统计方法**：10,000 次非参数 bootstrap 95% CI；配对 Wilcoxon signed-rank 检验；跨文件对齐比较。

## 实验与结果
- **数据集规模**：Synthetic（375 训/126 测，7 种量程）、Pressure Gauge（57 训/36 测，1/3 种量程，视频级分离）、SE（152 训/43 测，32/18 种量程，随机划分）。
- **主要 MPE 结果（capped）**：Synthetic with-range = **2.39%** [1.43, 3.90]；Pressure without-range = **2.61%** [1.66, 3.80]；SE with-range = **4.43%** [2.31, 7.14]。
- **范围消融**：配对差异均不显著（p>0.05）；但 uncapped 视角揭示 SE no-range 最大全量程误差达 **407.14%**（vs. with-range 41.25%），说明范围元数据主要抑制极端失败。
- **LODO 迁移退化**：排除 SE 时 MPE 上升至 13.25%（+8.82pp, p<0.001）；排除 Synthetic 时升至 14.99%（+12.60pp, p<0.001）；排除 Pressure 仅 +1.68pp（不显著）。
- **鲁棒性（SE 测试集）**：清洁基线 4.43%；S3 级别高斯模糊 MPE 升至 **19.92%**，>5% FS 失败率 **65.12%**；低光照、反射、透视、小遮挡影响相对较小。
- **解析失败率**：615 条预测全部可解析（0 失败）。
- **校准**：pooled with-range ECE=0.053、AURC=0.064；pooled no-range ECE=0.047、AURC=0.055；pooled 出现 2 例 HC error。
- **Scale Probe**：SE 测试集最小端点 100% 解析、95.3% 精确匹配（MAE=2.09）；最大端点 100% 解析、83.7% 精确匹配（MAE=44.21）。
- **其他基线（SE）**：Qwen2.5-VL-3B with-range MPE=4.98%；SmolVLM2-500M=24.14%；SmolVLM2-256M=26.22%；MobileNetV2 with-range=11.12%。
- **效率**：Qwen2.5-VL-7B 平均推理延迟 ~595ms、峰值显存 8101MB；MobileNetV2 仅 3.95ms/144MB。

## 相关工作脉络
- **多阶段 CV 管线**（Gaugetracker、子站检测、关键点测量等）：依赖几何假设与预处理，对眩光/遮挡/角度敏感；本文定位为"端到端 VLM 直读"对照。
- **直接回归紧凑型基线**（MobileNetV2、CNN regressor）：计算高效但容量有限，且不同工作使用不同目标参数化难以公平比较；本文将其作为轻量参考。
- **参考方法 [5] Howells et al. (CVPR 2021)**：任务专用多阶段手机仪表转录管线，μR=0.024 vs. 本文 Qwen no-range μR=0.026，整体接近但非严格可控比较（帧采样协议不同）。
- **通用 VLM 家族**（SmolVLM2、Qwen2.5-VL）：本文选取 7B 级因兼顾本地 QLoRA 适配可行性与足够视觉理解力，相较 SmolVLM2 显著提升。
- **QLoRA 参数高效微调**（Dettmers et al., NeurIPS 2023）：本文将其迁移至多模态 VLM 的仪表读取任务，验证 0.29% 参数可大幅改善精度。
- **合成仪表数据**（Kaggle Synthetic DS6.0）：低成本高可控基准，用于快速验证方法有效性，但与真实工业分布存在差距。

## 局限性与未来方向
- 数据集偏小（测试集 126/36/43），bootstrap 区间仅量化采样不确定性，需更大独立工业基准验证。
- Synthetic 与 SE 为随机划分，存在 Gauge 类型/环境重叠可能；LODO 仅反映域级迁移，无法诊断 gauge-family 级重叠。
- 仅处理单目标仪表图像，未评估自动多仪表检测与选择。
- Scale Probe 仅验证能否读取最大/最小刻度值，未检验刻度间隔、单位理解与全标签识别。
- 跨域迁移显著退化（Synthetic/SE 排除后 MPE 翻倍以上），模型非域不变。
- 高置信度错误仍存在，且对高斯模糊高度敏感，当前 blur 增强不足以覆盖重度模糊场景。

## 研究启发与可借鉴点
- **固定 epoch + 无早停 + 最终 checkpoint 协议**可有效消除 checkpoint selection bias，便于跨条件公平比较，值得在同类 VLM 微调研究中复用。
- **with-range / without-range 独立训练对照**是一种干净元数据消融设计，可迁移至任何需评估"外部知识注入"对 VLM 数值输出的影响场景。
- **LODO + 确定性扰动联合评估**同时刻画迁移边界与失效模式，比单一 ablation 更能指导部署风险控制。
- **扩展指标体系**（uncapped MPE、P95/Max FS、操作容差通过率、HC error、AURC）共同揭示被平均指标掩盖的长尾风险，可作为工业视觉评测的参考模板。
- **Blurred-patch + attention rollout 双诊断对比**揭示了二者空间一致性中等（Pearson 0.66–0.78）且均不能证明因果正确性，对后续 VLM 可解释性研究具有警示价值。

## 关键术语表
**Range-normalized MPE**：将绝对误差除以仪表量程（max−min）再取均值的百分比误差，用于跨不同量程仪表公平比较。
**QLoRA**：结合 4-bit NF4 量化与低秩适配的参数高效微调方法，本文仅训练 0.29% 参数即实现显著精度提升。
**LODO（Leave-one-dataset-out）**：每次保留一个完整数据集仅做测试、其余用于训练，用于评估跨域迁移退化程度。
**Uncapped MPE**：不将单样本误差上限截断为 100%，保留极端失败尾部分布以暴露严重误差。
**High-confidence error（HC error）**：序列生成置信度 C≥0.99 但绝对误差超过全量程 5% 的错误预测，反映"自信但错误"的风险。
**AURC**：风险-覆盖曲线下的面积，越小表示基于置信度阈值过滤错误预测的能力越强。
**Scale Probe**：使用 JSON 输出格式的提示，让模型直接从刻度盘图像读取最小/最大刻度值，用于验证视觉端点感知能力。
**Blurred-patch sensitivity map**：将图像划分为网格并逐格高斯模糊，测量预测误差变化量以定位对读数最敏感的区域。

## 可复现要素
- **数据集**：Synthetic 与 Pressure Gauge 来源于 Kaggle（公开）；SE 为私有工业数据不可公开分发。
- **代码/权重**：论文声明最终 adapter、精确 split manifest、gauge-range 元数据、ground-truth 读数与评估脚本将公开；基座模型 Qwen2.5-VL-7B-Instruct 为开源。
- **关键超参**：QLoRA rank=8、α=16、dropout=0.05；学习率 2e-4；batch size=1、梯度累积 8 步；20 epoch 固定训练；动态分辨率像素上限 640×28×28；bfloat16 混合精度；AdamW。
- **硬件**：双 Nvidia RTX A5500（训练）；推理评测亦在同一型号上执行。
- **统计**：10,000 次非参数 bootstrap 95% CI；配对 Wilcoxon signed-rank 检验；校准 10-bin ECE/MCE 与 AURC。
