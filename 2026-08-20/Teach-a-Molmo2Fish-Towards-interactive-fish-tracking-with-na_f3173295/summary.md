---
title: "Teach-a-Molmo2Fish-Towards-interactive-fish-tracking-with-na"
source: https://arxiv.org/pdf/2608.18602v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 06:14:56"
field: "多目标追踪与人机交互"
keywords: ["Multi-object tracking", "Multimodal large language model", "Human-AI interaction", "Interactive prediction correction", "Fisheries sonar video", "Parameter-efficient fine-tuning"]
innovations: ["首次将MLLM应用于渔业声呐视频追踪与交互式纠正", "提出多轮自然语言对话实现追踪预测实时修正的工作流", "构建合成轨迹生成管道使MLLM学会响应多样化纠正提示"]
benchmarks: ["Caltech Fish Counting (CFC)", "HOTA"]
---

# 论文速读：Teach a Molmo2Fish: Towards interactive fish tracking with natural language guidance

## 一句话总结
本文提出了一种基于多模态大语言模型（MLLM）的交互式多目标追踪预测纠正方法，通过多轮自然语言对话实时修正不完美的追踪预测。作者基于开源模型Molmo2微调构建了Molmo2Fish工具，在渔业声呐视频的鲑鱼追踪任务上实现了从零样本5%到高HOTA 74%的性能提升，并验证了自然语言引导的定向纠正能力。

## 研究问题与动机
- **多目标追踪在生态场景中的脆弱性**：传统两阶段跟踪-by-detection方法（如YOLO+SORT）在训练数据覆盖的环境外表现不佳，需要针对新部署点进行重新训练，而渔业利益相关者缺乏机器学习专业知识。
- **人工标注成本高昂**：多目标追踪标注需要用户在可能拥挤的场景中反复绘制和调整边界框，耗时费力，且重新训练模型需要大量标注数据和ML expertise。
- **MLLM在视觉-语言任务上的潜力尚未探索**：尽管MLLM已在一般RGB视频上展现出追踪能力，但其在声呐视频这一非自然域上的泛化能力和交互纠正能力尚不清楚。
- **传统交互式分割方法的局限**：现有交互式视频目标分割方法（如SAM3、XMEM++）依赖逐帧点击交互，而渔业工作流更关注点追踪（tracks）而非分割，且需要基于自然语言的整段视频级交互。

## 核心贡献（创新点）
1. **首次将MLLM应用于渔业声呐视频追踪任务**：证明通过LoRA参数高效微调，可以将仅在RGB视频上训练的Molmo2适配到声学声呐这一全新视觉域，性能从5%提升至74% HOTA。
2. **提出多轮交互式追踪纠正工作流**：设计了从初始错误预测到用户自然语言提示再到修正输出的多轮对话范式，使非专业用户可通过对话修正追踪结果，无需重新标注或重新训练。
3. **构建了完整的合成轨迹生成管道**：开发了一套从GT轨迹出发、应用多种腐蚀原语（如add、delete、fragment、merge等）生成训练数据的自动化工具，并采用LLM辅助生成多样化的自然语言纠正提示。
4. **实证验证了MLLM在生态追踪中的交互能力**：系统评估了Molmo2Fish在纯追踪、有引导追踪和多种纠正任务上的表现，发现模型能有效响应详细提示进行定向纠正，但对自身最优预测的进一步提升存在困难。

## 方法详解

### 模型架构
Molmo2Fish基于Molmo2多模态大语言模型构建，包含三个核心组件：
- **视觉编码器（SigLIP 2）**：接收视频帧（可能降采样），编码为潜在视觉特征空间。
- **连接器（Connector）**：将视觉特征投影到LLM嵌入空间，使其与语言token同等处理。
- **LLM（Qwen3）**：将视觉token和用户文本token堆叠到单个序列中进行处理。

输出格式：模型以文本token序列输出追踪结果，格式为 `{t track_id x_coord y_coord}`，表示在时间t秒处track_id为某个值的对象在(x, y)坐标位置。

### 指令微调策略
Molmo2Fish同时学习两个目标：
1. **纯追踪任务**：与Molmo2训练集中追踪任务格式相同，输入视频+提示"Track all fish"，输出完整追踪序列。
2. **交互式追踪纠正任务**：构建人工多轮对话示例，格式为：
   - User: ⟨video⟩ Track all fish.
   - Assistant: ⟨corrupted_point_sequence⟩（初始错误预测）
   - User: ⟨correction_prompt⟩（纠正提示）
   - Assistant: ⟨corrected_point_sequence⟩（修正后预测）

训练时仅对最后一步（纠正后输出）计算交叉熵损失。

### 合成轨迹生成管道
从Caltech Fish Counting (CFC)数据集的GT轨迹出发，应用以下腐蚀原语生成训练数据：

| 腐蚀类型 | 功能 |
|---------|------|
| add | 添加虚假轨迹 |
| delete | 删除一条轨迹 |
| fragment | 将轨迹在某帧处分裂为两条 |
| merge | 将两条轨迹合并为同一track ID |
| deviate | 施加空间位移 |
| partial_delete | 删除轨迹的一部分 |
| duplicate | 在轨迹附近添加虚假轨迹 |
| drift | 对轨迹施加方向性偏移 |
| jitter | 添加随机噪声 |
| stall | 轨迹在某帧后静止 |
| id_switch | 两条轨迹的ID在某帧交换 |
| split_gap | 轨迹在某区间内缺失 |
| swap | 两条轨迹ID整体交换 |
| subsample | 每轨迹最多保留2个点 |

提示风格分为四类：
- **No-info prompts**："Fix all mistakes in these tracks, if any exist."
- **Wrong-only prompts**："Fix all mistakes in these tracks."
- **Vague prompts**：≤20词，指定纠正内容但不含精确时间戳
- **Full prompts**：≤40词，完全指定时间戳、精确计数和track ID

此外还构建了基于真实模型预测的纠正轨迹（Molmo-low和Molmo-high）。

### 训练配置
- **微调方式**：rank 64 LoRA，学习率0.001
- **训练规模**：300步，batch size 64，最大序列长度16,384 tokens
- **上下文窗口限制**：超长视频被切片，帧降采样因子为3

## 实验与结果

### 数据集
使用 **Caltech Fish Counting (CFC)** 数据集，包含来自四个声呐部署点（Kenai left/right bank、Elwha、Nushagak）的1,500个视频的MOT标注。训练集从Elwha、Nushagak和Rightbank随机选择25%视频作为验证集，Kenai数据保持原论文划分。

### 评估指标
使用 **HOTA (Higher Order Tracking Accuracy)**，针对点预测和边界框GT的特殊适配版本（真阳性判断为点落在GT框内，不进行IoU阈值平均）。

### 主要结果

| 任务 | Molmo2（零样本） | LLM+Connector+ViT LoRA |
|------|-----------------|------------------------|
| Pure Tracking | 0.049 | **0.737** |
| Synthetic Corrections (Full) | 0.164 | **0.817** |
| Targeted Corrections (Full) | - | **0.900** |
| Molmo-low Corrections (Full) | - | **0.669** |
| Molmo-high Corrections (Full) | - | 0.747 |

- **纯追踪性能**：从零样本5%提升到LoRA微调后74% HOTA，验证了MLLM在声呐域的适配可行性。
- **纠正能力**：Molmo2Fish能从YOLO+SORT预测中显著提升追踪性能（三个河流中的四个），但面对自身最优预测（Molmo-high）时提升有限。
- **定向纠正**：在"Targeted Corrections"任务上达到90% HOTA，证明模型能准确响应用户的具体纠正指令。
- **提示信息量的影响**：对于简单轨迹，有无提示影响不大；对于中等难度轨迹，详细提示带来显著改进；对于最难轨迹，即使有详细提示也难以纠正。

### 消融实验结论
1. **微调组件**：需同时微调LLM、Connector和ViT才能达到最优性能（73.7% vs 60.8% vs 65.6%）。
2. **训练数据组成**：仅训练纯追踪任务的模型在纠正任务上表现差（0.484），必须加入Molmo-low纠正数据才能学会纠正（0.669）。
3. **分布外泛化**：对训练中未见的腐蚀类型（split_gap、swap、duplicate）泛化较差，自然语言提示无法弥补这一差距。

## 相关工作脉络

1. **两阶段MOT方法（YOLO+SORT, ByteTrack）**：传统生态追踪的首选方案，速度快、计算成本低，但跨环境泛化能力弱，需要针对新场景重新训练或域自适应。本文与之对比显示Molmo2Fish在Elwha等稀疏场景上超越YOLO+SORT，但在高密度Nushagak场景上仍落后。

2. **交互式视频分割（SAM3, XMEM++, EVA-VOS）**：允许用户通过点击/拖拽直接在帧上修正分割结果，并提出"最有用"的下一帧供标注。本文与之的本质区别是：操作对象是点追踪而非分割掩码，交互方式是全视频级别的自然语言对话而非逐帧操作。

3. **多模态大语言模型追踪（Molmo2, Gemini 3 Pro, PerceptionLM, Qwen3-VL）**：现有MLLM可在RGB视频上执行追踪任务，但均为单轮非交互形式。本文首次将其扩展到多轮交互纠正场景，并探索了域外（声呐视频）泛化能力。

4. **人-in-the-loop主动学习**：包括统计去偏、传递式预测、主动模型选择等方法。本文与之不同在于：不提供统计校正或模型选择，而是通过自然语言对话直接修改预测结果，避免了重新标注和重新训练的成本。

5. **生态追踪数据集（CFC, BUCK-TALES, SA-FARI, 3D-ZeF）**：CFC是最大的渔业声呐追踪数据集，本文在此基础上扩展了纠正轨迹标注。相比其他生态数据集，CFC的独特挑战在于声呐图像的噪声特性和鱼类密度变化大。

## 局限性与未来方向

- **自身预测纠正困难**：Molmo2Fish难以进一步提升其自身最优预测（Molmo-high仅提升约1%），表明模型对"已接近正确"的预测缺乏细粒度调整能力。
- **分布外泛化有限**：对训练中未见的腐蚀类型（如split_gap、swap）泛化能力显著下降，自然语言提示无法弥补这一差距。
- **密集场景表现不佳**：在Nushagak等高密度鱼群场景（clip中包含30-40条鱼，接近Molmo2训练上限40条）上容易混淆，出现遗忘退出帧的轨迹或水平外推等问题。
- **提示信息量的边际效益递减**：对于简单和最难轨迹，有无详细提示影响均不大，仅有中间难度轨迹能从详细提示中获益，这类轨迹在数据集中占比较低。
- **单步纠正限制**：由于上下文窗口长度限制，当前仅支持单步纠正，无法实现真正的多轮迭代交互。

未来方向包括：使用更多数据和更大规模训练提升纠正能力；理解中间难度轨迹的动态学；开发真正多步交互的模型；探索更高效利用语言先验的方法。

## 研究启发与可借鉴点

1. **合成轨迹生成管道的构建思路**：通过程序化应用多种腐蚀原语（从简单到复杂）从GT轨迹生成训练数据，配合LLM辅助生成多样化自然语言提示，是解决交互任务标注稀缺问题的有效范式，可迁移到其他需要交互式纠正的视觉任务。

2. **双目标联合微调策略**：将纯追踪任务和交互式纠正任务同时作为训练目标，使模型既具备良好基础追踪能力，又学会响应用户纠正，这一设计对多任务MLLM微调具有参考价值。

3. **HOTA指标的自适应改造**：针对点预测与边界框GT不匹配的问题，将相似性得分从IoU改为二元判断（点在框内），并优化匈牙利匹配，这一适配方法可扩展到其他预测格式与标注格式不匹配的评估场景。

4. **LLM辅助提示生成的自动化流程**：利用Opus 4.5等LLM根据结构化轨迹变更日志自动生成自然语言纠正提示，并注入描述性/处方性风格变体，为构建大规模交互数据集提供了可复用的数据生成pipeline。

5. **错误类型系统化分类框架**：将追踪错误归纳为false negative、false positive、fragmentation、merger、deviation等可组合的原语，为训练数据和评估设计提供了结构化理解，可推广到其他追踪任务的分析。

## 关键术语表

**Multi-object tracking (MOT)**：在视频中同时对多个目标进行定位并分配唯一轨迹ID的任务，是比单目标检测更复杂的视觉任务。

**Higher Order Tracking Accuracy (HOTA)**：一种综合衡量检测准确性和关联准确性的MOT评估指标，分别量化Detection Accuracy (DA)和Association Accuracy (AA)。

**Molmo2**：MIT开源的多模态大语言模型，具备视频理解、动作识别和精确定位能力，本文作为基础模型进行微调。

**LoRA (Low-Rank Adaptation)**：参数高效微调方法，通过低秩矩阵分解更新部分模型参数而非全参数微调，显著降低计算成本。

**Sonar video**：声学声呐视频，通过水下声呐摄像头获取的鱼群影像，与RGB视频在视觉特性上有显著差异（噪声多、对比度低）。

**Corruption primitives**：用于程序化生成训练数据的轨迹错误类型，包括add、delete、fragment、merge等，可组合应用以模拟真实追踪错误。

**Targeted correction**：要求模型仅纠正特定错误而非所有错误的任务设定，评估模型对用户精确指令的遵循能力。

**Trajectory**：在本文语境中指一次完整的纠正交互序列，包括初始预测、用户提示和修正输出的多轮对话。

## 可复现要素

- **数据集**：Caltech Fish Counting (CFC) 数据集公开可用（github.com/tidalove/molmo2fish），包含MOT标注的声呐视频
- **代码**：公开可用，见 github.com/tidalove/molmo2fish
- **模型权重**：Molmo2为基础模型公开可用，Molmo2Fish微调权重公开
- **关键超参**：LoRA rank=64，学习率=0.001，训练步数=300，batch size=64，最大序列长度=16,384 tokens，帧降采样因子=3
- **基础模型**：Molmo2（SigLIP 2视觉编码器 + Qwen3 LLM）
- **评估指标**：HOTA（适配点预测版本）
