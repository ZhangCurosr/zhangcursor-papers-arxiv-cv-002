---
title: "On-Policy-Self-Distillation-in-Diffusion-Models"
source: https://arxiv.org/pdf/2608.24646v1.pdf
model: agnes-2.5-flash
chunks: 4
summarized_at: "2026-09-01 23:54:17"
---

# 论文速读：On-Policy-Self-Distillation-in-Diffusion-Models

## 一句话总结
DiffusionOPSD 提出一种基于 on-policy self-distillation 的扩散模型强化学习对齐框架，通过将图像级奖励显式投影为 clean-output 目标并分离“目标构造”与“有限拟合”两个阶段，有效缓解端点奖励与中间去噪预测的结构性错配，在 19/20 组设置下取得最优性能，同时显著提升训练效率与优化稳定性。

## 研究问题与动机
- **监督信号错配**：扩散模型的 RL 对齐通常依赖端点奖励（endpoint reward），但该奖励不指定中间去噪预测应如何变化，导致奖励信号与预测位置存在结构性错配。
- **构造增益≠实现增益**：现有方法缺乏对“奖励引导的目标构造”与“有限拟合后的奖励变化”的独立评估，更大 target construction gain 未必转化为更大 realized gain，甚至可能反向（ordering reversal）。
- **CFG training 的 scale 依赖陷阱**：训练期引入 classifier-free guidance 会造成推理期 scale 强依赖，且不提升最佳分数；scale 错配或高 scale 训练反而显著损害性能。
- **训练效率与稳定性瓶颈**：现有 on-policy RL 方法（如 FlowGRPO、DiffusionNFT）在扩散模型上采样开销大、terminal position 不稳定，需更高效的蒸馏替代方案。

## 核心贡献（创新点）
- 提出 DiffusionOPSD 三阶段 on-policy self-distillation 循环，将 reward 信号显式投影至 clean-output 空间而非直接优化噪声预测，与常规 policy gradient 方法解耦。
- 设计有界正负目标构造机制（bounded positive/negative targets），沿 normalized reward gradient 进行有限步投影并提供 repulsive reference，区别于直接端点 loss 优化。
- 提出共享 output 的正负分支拟合策略（$y_\theta^+$ 与 $y_\theta^-$）配合动态权重 $\omega_\mathbf{c}^k$，实现 detached supervision，降低优化方差。
- 系统性揭示 target construction 与 finite realization 的非单调关系，量化 ordering reversal 比例，为对齐信号诊断提供可复用的分析范式。
- 实验证伪“训练 scale 匹配推理 scale 更优”的直觉，证明 CFG-free 训练在多 scale 下更鲁棒，为扩散模型对齐的训练配置提供反直觉但可靠的经验准则。

## 方法详解
- **Clean-output prediction**：将 velocity prediction 转换为可解码评估的 clean output，公式为 $y_\theta(s) = z_\sigma - \sigma v_\theta(s)$，使中间预测可直接与图像级奖励对齐。
- **Target construction**：从 anchor $y_0$ 出发，沿 normalized reward gradient 执行 $M_\mathrm{tgt}$ 步有界投影，构造 $\bar{y}_+$（正目标）与 $\bar{y}_-$（负目标/repulsive reference），实现奖励信号的显式目标映射。
- **Finite fitting**：通过共享 output 的正负分支 $y_\theta^+ = \beta y_\theta + (1{-}\beta)y_0$ 与 $y_\theta^- = (1{+}\beta)y_0 - \beta y_\theta$ 拟合 detached targets，使用 group-normalized endpoint reward 权重 $\omega_\mathbf{c}^k \in [0,1]$ 控制正负支相对强度，最小化构造目标与当前预测的差距。
- **Online loop**：每 outer iteration 冻结行为策略（behavior policy）采集轨迹，提供 query states 与 clean-output anchors；构造 targets 后训练可优化策略，随后通过 EMA 更新行为策略，目标集随行为策略变化而重建，保持 on-policy 性质。
- **默认超参**：$\sigma_q = 0.278$、$\rho = 0.10$、$\beta = 1$、$M_\mathrm{fit}=1$、$\eta_\mathrm{tgt}=1$。

## 实验与结果
- **实验设置**：覆盖 SD3.5-M 与 Z-Image-Turbo 两 backbone，10 个 evaluator（7 公开 + 3 内部），20 组 reward-matched settings；基线包括 ReFL、DiffusionNFT、FlowGRPO。
- **主性能**：DiffusionOPSD 在 19/20 设置中登顶；相对最强 baseline 最高提升 44.0%（SD3.5-M, VLM-Pairwise）；SD3.5-M 上 HPSv3 提升 43.0%；Z-Image-Turbo 上 Aesthetic +9.7%、ImageReward +30.7%、HPSv3 +4.9%、DeQA +3.9%、VLM-Pairwise +14.6%。
- **训练效率**：per 100 updates（8 GPU），SD3.5-M 消耗 28.2 GPU-hours，Z-Image-Turbo 消耗 149.8 GPU-hours；相比 DiffusionNFT 分别节省 40% 与 63%。
- **优化稳定性**：71 次单奖励 run 中位数 terminal position 为 98%（ReFL 97%、DiffusionNFT 90%、FlowGRPO 83%）。
- **Joint training**：300 updates 训练 PickScore/CLIPScore/HPSv2.1 三奖励，得分 25.51 / 0.333 / 0.389，相对 DiffusionNFT 提升 8.0% / 13.3% / 14.4%，且保留 specialist 模式 113.0% / 92.9% / 99.6% 增益。
- **构造 vs 实现分离分析**：正目标固定 suffix reward gain +0.03511、gradient alignment 0.7094，而 DiffusionNFT endpoint 变化 -0.03551、alignment -0.000203；HPSv2.1 上 62.3% prompts 出现 ordering reversal；CLIPScore 上 reversal rate 为 29.5%；query-state 替换为 forward-noised control 仅使 CLIPScore 从 0.3122 降至 0.3089（-1.1%）。
- **Human evaluation**：Z-Image-Turbo VLM-Pointwise（100 prompts），DiffusionOPSD 偏好比例 64%（vs base）、71%（vs FlowGRPO）、90%（vs DiffusionNFT）、61%（vs ReFL）；VLM-Pointwise 得分 0.243（ReFL 0.227）。
- **CFG training 分析**：CFG-free checkpoint 在 guidance scales 1、4.5、6 下评估得分为 0.3117、0.3108、0.3099；scale 4.5/6 训练后 conditional-only evaluation 分别下降 11.0% 与 15.5%；即使 train/eval scale 匹配，对角得分仍比 CFG-free 基线低 1.5% 与 2.3%；scale 乘积对称性不成立，3×3 网格中无单元格超过 CFG-free。

## 相关工作脉络
- 对比 ReFL、DiffusionNFT、FlowGRPO：本文方法将 reward 信号显式投影至 clean-output 空间并进行有界投影，而非直接优化 endpoint noise prediction，从根本上缓解监督错配。
- 与主流 RLHF/RLVR 在扩散模型上的变体定位差异：本文强调 on-policy self-distillation 与 detached supervision，避免 policy gradient 的高方差与不稳定，更适合连续生成任务的梯度传播特性。
- 与 CFG 训练/推理研究的关系：本文揭示 CFG training 引入的 scale 依赖陷阱，指出 CFG-free 训练配合 inference-time guidance 更优，与“训练期硬编码 guidance”路线形成对比。
