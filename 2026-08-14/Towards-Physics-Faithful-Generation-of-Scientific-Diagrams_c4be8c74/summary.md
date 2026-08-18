---
title: "Towards-Physics-Faithful-Generation-of-Scientific-Diagrams"
source: https://arxiv.org/pdf/2608.13112v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 04:44:16"
---

# 论文速读：Towards-Physics-Faithful-Generation-of-Scientific-Diagrams

## 一句话总结
提出 Princigram 框架，通过结构化物理思维链（SP-CoT）将物理图表的视觉内容转化为可审计的五步推理 JSON 标注，在 4.3M 物理图像语料上进行两级监督训练，显著提升文本到科学图表生成的物理忠实性，并构建细粒度属性可分解评测基准 VeriphyT2IBench。

## 研究问题与动机
1. **物理忠实性缺失**：现有顶级 T2I 模型生成自然图像已达摄影级真实感，但绘制自由体受力图、P-V 图、光路图等科学图表时，常出现力方向错误、过程路径不合规、公式与图示不符等问题，在教育与科研传播中具有误导危害。
2. **数据鸿沟**：大规模网页语料中高质量物理图表稀缺，且其 alt-text caption 物理信息极浅（如仅“pendulum diagram”），无法传递力、方向、控制方程等决定性约束。
3. **监督鸿沟**：标准 caption 监督仅编码外观分布，未编码“为何如此绘制”的物理推理过程，导致模型学到表面风格而未学到生成规则。
4. **评估失效**：现有基准（GenEval、T2I-CompBench）优化 prompt 遵从度与人类审美偏好；CLIP 类对齐分数甚至可能给物理错误图打高分，开集检测器亦无法验证决定物理正确性的几何关系（方向、关联、序次）。

## 核心贡献（创新点）
1. **提出结构化物理思维链（SP-CoT）**：为六大学科定义统一五步 JSON 推理模板，严格区分“图像可见事实”与“物理推演结论”，所有数学强制符号化。与自由形式 CoT 的本质区别在于：固定架构+保真规则+机器可解析表示，使数据构建、训练与评估全程结构化且可审计。
2. **构建两级标注物理图表语料**：汇合四源数据得到 4.3M 物理图像，其中 115,037 幅经专家级人工核验。与既往工作相比：规模更大、学科覆盖更全、标注直接绑定物理推理链条而非外观描述。
3. **提出 Princigram 生成器与结构化推理管道**：在统一多模态骨干（BAGEL / DiMOO）上进行全量预训练+专家 SFT，推理时先用 LLM 将自由提示展开为填充好的 SP-CoT schema 再驱动生成。与通用 T2I 的本质区别在于：将结构化物理推理作为稠密条件输入，而非依赖简短 prompt。
4. **构建细粒度可分解评测基准 VeriphyT2IBench**：基于保留测试集的专家标注，按规则为每张图动态编译专属二元问题库，由 VLM 裁判逐条比对并拆解为命名化物理事实。与单一 holistic 分数的本质区别在于：可定位具体错误属性，且问题列表随图表复杂度动态变化。

## 方法详解
- **SP-CoT 五步模板**：(1) Scenario（场景/物体识别）、(2) Parameters（状态/参数化）、(3) Structure（相互作用/结构分析）、(4) Laws（坐标系与支配定律）、(5) Synthesis（关键物理关系与理想化假设）。每学科指定恰好两步为严格图像保真（vis），其余可物理推演（inf）。例如力学保真 Step1 与 Step3，热力学保真 Step2 与 Step3 的过程路径。
- **形式化保真规则**：字段 $f=(\kappa_f, \nu_f)$ 携带类型 $\text{type}(f)\in\{\text{ent}, \text{rel}, \text{val}\}$ 与接地标签 $g(f)\in\{\text{vis}, \text{inf}\}$。有效标注须满足 $g(f)=\text{vis} \implies \nu_f \in \text{drawn}(x)$；缺失信息强制填 `""` 或 `[]`，所有数学必须为合法 LaTeX。
- **数据构建流水线**：人工种子 + OpenDataLab 抓取 + 公开网络爬虫 + 英文物理教材采购 → 去重/分辨率过滤/子学科分类 → Qwen3-VL 批量生成 JSON 标注 → 专家人工核验专家子集（重点校对 vis 步骤防幻觉扩散）。
- **模型训练**：统一骨干优化同一子学科平衡目标 $\mathcal{L}(\theta;\mathcal{D}) = \mathbb{E}_{s\sim w}\mathbb{E}_{(x,a)\sim\mathcal{D}_s} \mathcal{L}_{\text{gen}}(\theta; x, \sigma(a))$。预训练用 4.3M 语
