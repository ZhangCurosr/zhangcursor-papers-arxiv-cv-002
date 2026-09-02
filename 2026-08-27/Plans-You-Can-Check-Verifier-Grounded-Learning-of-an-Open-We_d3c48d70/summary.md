---
title: "Plans-You-Can-Check-Verifier-Grounded-Learning-of-an-Open-We"
source: https://arxiv.org/pdf/2608.25622v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 06:50:04"
---

# 论文速读：Plans-You-Can-Check: Verifier-Grounded Learning of an Open-Weight Planner for Executable Video-Editing

## 一句话总结
本文提出 RefineCut，一种针对"可执行视频剪辑规划"的两阶段学习框架：首先通过确定性验证器对多教师轨迹进行重放和打分仲裁，将噪声的前沿轨迹转化为验证过的 SFT 目标和混合粒度偏好对，得到一个 8B 开放权重规划器；随后引入 RefineCut-Evo，让规划器在验证器和任务特定评分准则下对自身修复候选打分，通过高边际 DPO 进一步优化，最终在 RefineCut-Bench 上达到 0.924 VES，在与开源教师相同的闭环中匹配或超越前沿模型，且推理时完全无需调用教师。

## 研究问题与动机
1. **视频编辑本质上是受约束的规划问题**：现实编辑需要将简报、片段池、音乐元数据和硬约束（目标时长、必须保留/排除的片段、节奏等）转化为可执行的时间线补丁（RefinePatch），而非直接生成像素；现有 text-to-video 或指令编辑系统无法产生片段级计划。
2. **工作流系统锁定在封闭前沿模型**：DIRECT、GLANCE 等系统虽在决策层运行，但将规划逻辑编码在提示脚手架中并调用冻结的前沿骨干，策略从不根据硬约束优化，也无法开放部署。
3. **多教师轨迹是噪声且不一致**：编辑任务不存在单一 ground-truth 修复，三个前沿教师（GPT-5.4、Qwen3-Max、DeepSeek-V4-Pro）在 schema、分支结构和 best branch 判断上存在分歧，直接模仿会继承噪声。
4. **缺乏规划级基准**：现有视频编辑基准评估渲染像素或 agent scaffold 的输出，没有一个绑定真实素材元数据、显式约束账本、多教师轨迹和确定性验证器评估的规划层面数据集。
5. **可验证性是编辑独有的学习优势**：编辑输出可通过执行验证（验证器检查约束账本），这使得可以像数学推理和代码生成那样使用"执行基于的监督"，而像素级生成缺乏这一性质。

## 核心贡献（创新点）
1. **形式化"可执行视频剪辑规划"并发布 RefineCut-Bench**：将视频编辑决策层定义为带显式约束账本的规划问题，发布包含 3,578 个规范任务、7,971 个标注片段、499 首音乐轨的基准，首个将真实素材、约束账本、多教师轨迹和确定性验证器评估绑定在一起的规划级基准。
2. **验证器重放轨迹蒸馏（Verifier-Replayed Distillation）**：提出对多教师轨迹进行规范化、确定性重放并通过验证器打分仲裁的流程，将噪声的前沿轨迹转化为验证过的 SFT 目标和混合粒度偏好对，把"执行基于的监督"从数学/代码领域引入创意规划。
3. **RefineCut-Evo 验证器中心的学生自改进**：引入基于 Rubric 的自我改进阶段，学生用验证器分数和编辑评分准则（ER1-ER7）联合打分，通过高边际 DPO 训练，使规划器在推理时能在纯闭环验证器循环中运行而无需教师调用。
4. **跨骨架迁移与与前沿模型的对齐**：验证器重放蒸馏在 Qwen3-8B、Llama-3.1-8B 和 GLM-4-9B 上均带来显著增益；最终 8B 规划器在与教师相同的闭环协议中匹敌或超过 GPT-5.4 和 Qwen3-Max，证明确定性验证器信号的可迁移性。
5. **严格的鲁棒性分析**：通过 canonical-clean 子集、Human50 人工简报、语义干预实验、失败构成分析和人工盲评，全面验证性能增益并非来自任务重叠、指标 gaming 或渲染管道，而是真正的规划能力提升。

## 方法详解
**任务定义**：任务实例 $x = (b, C, M, s_0, L)$，其中 $b$ 是自然语言简报，$C$ 是带 caption 和视觉元数据的片段池，$M$ 是可选的音乐元数据（节拍时间和能量），$s_0$ 是初始时间线状态，$L$ 是显式约束账本（7 种约束家族：duration、transition、music_sync、clip inclusion/exclusion、repeat limit、pacing）。规划器 $\pi_\theta$ 在每个步骤 $t$ 发出一个 RefinePatch（基于 RFC 6902 的 JSON Patch），验证器应用补丁 $s_{t+1} = \text{Apply}(s_t, p_t)$ 并重新计算账本 $L_{t+1} = \text{Verify}(s_{t+1})$。

**三阶段流程**：
1. **Stage 1 - 验证器重放的教师引导启动**：对每个训练任务，从三个前沿 API 教师各收集一个细化 rollout，每步有 4 个候选分支。首先将原始输出规范化为 canonical patch trajectories（映射 JSON Pointer 路径、验证 clip 引用 against task-local aliases）。然后对每个规范化步骤保留最多 4 个具有不同 repair_operator 且 Jaccard 不同的分支，通过确定性验证器重放，计算 6 个信号分数 $V(b)$：
   - $\Delta$CSR（约束满足变化，$w_1=0.35$）
   - Targeted Repair（$w_2=0.20$）
   - Required-Clip Recall（$w_3=0.20$）
   - Patch Applicability / PASR（$w_4=0.10$）
   - No Regression（$w_5=0.10$）
   - Locality（$w_6=0.05$）
   
   验证器最佳分支作为 verified SFT 目标（标准 next-token cross-entropy，3,317 个示例，1,200 steps，lr=5e-5）。同时构建混合粒度偏好对（step-level 和 trajectory-level），用离线 DPO 训练：$\mathcal{L}_{\text{DPO}} = -\mathbb{E} \log \sigma(\beta z)$，其中 $z = \Delta_\theta(p^+, x) - \Delta_\theta(p^-, x)$，$\Delta_\theta(p,x) = \log \pi_\theta(p|x) - \log \pi_{\text{ref}}(p|x)$，$\beta=0.1$，得到 Mixed-Pref 检查点。

2. **Stage 2 - RefineCut-Evo（验证器中心的自改进）**：在每个训练状态采样 $K=4$ 个学生修复候选，过滤 JSON 解析或应用失败的候选。使用任务特定的 rubric ER1-ER7（Intent、Ledger Satisfaction、Clip Grounding、Timeline Coherence、Duration/Pacing、Music/Beat、Edit Economy）打分 $R(c) = \sum_{m=1}^7 \alpha_m r_m(c)$。联合得分 $S(c) = \lambda V(c) + (1-\lambda)R(c)$，$\lambda=0.65$。选择 $c^+ = \arg\max_{c \in \mathcal{C}(s)} S(c)$，$c^-$ 为满足 $S(c^+) - S(c) \geq \tau$ 的最强负样本（非最低分候选），构建 779 对高边际偏好对，从 Mixed-Pref 开始用 DPO 训练（lr=1e-6，$\beta=0.05$，600 steps，step-300 checkpoint 在 dev100 上选优）。

3. **Closed-loop Deployment**：推理时规划器读取 $s_t$ 和违反的账本项目，发出一个 RefinePatch，验证器验证、应用并重新计算账本，无教师调用。循环最多 $T=3$ 步，最终状态交给下游编辑工具链。

**评估指标**：VES = $0.30 \text{FCSR} + 0.15 \text{HardPass} + 0.15 \text{PASR} + 0.15 \text{ReqClipRecall} + 0.10 \text{DurationPass} + 0.
