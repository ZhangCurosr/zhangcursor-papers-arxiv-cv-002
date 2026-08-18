---
title: "SPARED-Reasoning-Based-AI-Generated-Image-Detection-via-Adve"
source: https://arxiv.org/pdf/2608.12876v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 04:00:07"
---

# 论文速读：SPARED-Reasoning-Based-AI-Generated-Image-Detection-via-Adve

## 一句话总结
本文提出 SPARED，一种解耦的攻击者-防御者强化学习框架，通过扩散图像编辑器将真实照片按指令编辑为语义对齐的假图，构建持续进化的对抗课程；防御者（推理 MLLM）仅以最终判罚正确性为奖励，使解释质量成为精度提升的涌现副产品，从而在三个外部基准（含两个零样本）上实现检测精度与自然语言解释同步单调提升。

## 研究问题与动机
- **数据集来源捷径（Provenance Shortcut）**：现有训练集中真实图与伪造图往往来自不同渠道，分辨率、JPEG 质量、语义分布存在系统性差异，探测器容易学到“来源线索”而非“生成痕迹”，在语义对齐的无偏测试集上性能崩塌。
- **模板化解释与静态语料天花板**：基于监督微调的 MLLM 检测器倾向于复制语料中的固定推理句式，缺乏对未知伪影的因果理解；且固定数据集永远落后于持续进化的生成器，决策边界一旦固化即成为静态靶标。
- **对抗训练中的退化风险**：纯奖励“欺骗成功”会导致攻击者跳过编辑、直接输出离分布噪声即可 fool 检测器，使防御者学到虚假规则；视觉域缺乏类似 LLM 安全自博弈中成熟的防坍塌机制。
- **部署场景对可解释性的刚性需求**：内容审核不仅需要二分类标签，还需提供可核验的自然语言证据；不可解释的误判极易引发舆情与 misinformation，且公开检测器常遭白/黑盒逃避攻击。

## 核心贡献（创新点）
1. **从数据构造端硬性关闭证明来源捷径**：通过同源配对编辑（同一张真实照片的原始版与编辑版）强制正负样本共享构图、分辨率与语义内容，使模型只能依赖编辑痕迹学习，区别于事后对齐或重采样的去偏方法。
2. **仅以最终判罚奖励驱动推理能力涌现**：防御者采用 verdict-only 的 0/1 奖励，解释文本不被直接优化；实验证明解释质量（CSemAP）随检测准确率同步单调上升，避免了多目标奖励的权重调优困境。
3. **引入 PaCo 门控的异构对抗 RL 循环**：攻击者为连续像素空间的扩散编辑器，防御者为离散 token 空间的自回归 MLLM，参数完全解耦；攻击者奖励被指令忠实度评分器门控，防止“无编辑欺骗”退化。
4. **迭代课程自动生成与跨域泛化验证**：五轮交替训练使硬负例难度动态校准至当前防御者盲区；在 DeepfakeJudge-Detect、AnomReason-Deepfake 及十种未见生成器的 Holmes-Set 上均实现单调提升，零样本 Holmes-Set 均值精度达 92.8。

## 方法详解
- **整体循环结构**：SPARED 包含解耦的防御者 $\pi_D$ 与攻击者 $\pi_A$，交替训练 5 轮。第 $t$ 轮中，冻结的 $\pi_D^{(t)}$ 评判攻击者生成的编辑对，攻击者据此更新；随后用新生成的配对池对 $\pi_D$ 进行 GRPO 训练，迭代推进。
- **推理防御者（Reasoning Defender）**：基于 Qwen3.5-9B 主干，先经 LoRA SFT 学习 `<reasoning>` / `<answer>` 标签格式与基础伪影词汇，再切换至全参数 GRPO。对输入图像 $x$ 采样 $G$ 个响应，奖励函数为：
  $$r_D(x, y) = \mathcal{H}[\hat{y} = y]$$
  仅当解析出的最终 verdict 与 ground truth 一致时得 1 分，格式、长度、推理内容均无独立奖励项；GRPO 在组内归一化优势，使解释能力只能通过“引导出正确判罚”被强化。
- **PaCo 门控扩散攻击者（PaCo-Gated Diffusion Attacker）**：基于 Qwen-Image-Edit-2511 的 LoRA，使用 DiffusionNFT 在线训练。给定真实源图 $x_{src}$ 与编辑指令 $c$，生成 $x_{edit}$。引入两个冻结评判器：指令遵循评分器 $r_{PaCo}(x_{src}, c, x_{edit}) \in [0, 1]$，以及当前防御者快照的逆检测奖励：
  $$r_{det} = \mathbb{1}\big[\neg (\pi_D^{(t)}(x_{src})=\text{real} \wedge \pi_D^{(t)}(x_{edit})=\text{fake})\big]$$
  最终门控奖励为：
  $$r_A = \begin{cases} r_{det}, & \text{if } r_{PaCo} \geq 0.7 \\ 0, & \text{otherwise} \end{cases}$$
  该设计确保攻击者必须忠实执行指令且同时欺骗防御者才能获得梯度；若仅靠离分布噪声或放弃编辑来骗过检测器，奖励恒为 0。
- **数据构造与训练调度**：每轮从混合编辑语料（ImgEdit、pico-banana-400k、MagicBrush）中抽取固定且去重的真实源图集，经感知哈希筛查排除与评测集重合；由当前攻击者重新编辑生成配对假图。采用交替而非联合优化，保证攻击者 RL 期间判决器稳定、防御者训练期间数据池稳定。调度为：SFT 初始化 → Iter1（基础编辑器）→ A1/A2（攻击者迭代）→ Iter2/Iter3（防御者迭代）。

## 实验与结果
- **数据集与基线**：DeepfakeJudge-Detect（裁判式检测，含文本生成与本地编辑子集）、AnomReason-Deepfake（联合评估判罚与语义解释质量）、Holmes-Set（10 种完全合成、零样本未见生成器）。基线覆盖开源 MLLM、闭源大模型、推理模型及专用检测器（UniGenDet、AnomReasonor、GenShield 等）。
- **主结果**：
  - DeepfakeJudge-Detect：Iter3 达 Overall Acc 79.5、Overall F1 79.3，超越参数量 26 倍的 Qwen3-VL-235B-Thinking（74.5）；仅落后于 Qwen3-VL-235B-Thinking 的 82.7。本地编辑子集精度从 44.8 跃升至 82.5，增益集中于此。
  - AnomReason-Deepfake：零样本 Acc 92.18，领先次优 UniGenDet 3.6 点、领先 AnomReasonor 9.6 点；CSemAP-Full 达
