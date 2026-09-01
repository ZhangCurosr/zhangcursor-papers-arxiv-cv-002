---
title: "Optimize-Your-Sampling-Tuned-Difusion-Sampling-with-Bayesian"
source: https://arxiv.org/pdf/2608.18040v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 00:59:24"
field: "生成模型高效采样"
keywords: ["diffusion sampling", "Bayesian optimization", "sampling schedule", "text-to-image", "few-step generation", "Human Preference Score"]
innovations: ["将扩散采样时间步选择建模为黑盒优化问题并直接用贝叶斯优化目标指标而非代理目标", "提出累积积重参数化保证单调递减调度，联合优化时间步与EDM采样超参数"]
benchmarks: ["COCO Captions", "DiffusionDB", "ImageNet-512", "Prompt Diffusion Inverse Tasks"]
---

# 论文速读：Optimize-Your-Sampling-Tuned-Difusion-Sampling-with-Bayesian

## 一句话总结
论文提出 OYS（Optimize Your Sampling）框架，将扩散模型采样时间步选择建模为黑盒优化问题，直接用贝叶斯优化（Bayesian Optimization）优化目标指标本身而非代理目标；5 步 OYS 调度可保留 50 步调度的 89%–94% 质量，同时推理成本降低 10 倍。

## 研究问题与动机
- **扩散模型采样计算开销高**：迭代去噪过程需多次前向传播大网络，低步数预算下默认调度质量急剧下降。
- **时间步选择未被充分关注**：相较高效的 samplers/蒸馏模型，采样时间步的选择对质量影响显著却研究较少。
- **现有方法 AYS 的局限**：AYS 优化 KLUB（KL 散度上界）这一理论代理目标而非目标指标本身，且通过局部启发式搜索 + 早停获得，释放的调度在 log-SNR 空间中仍接近默认调度，无法在全局重新分配步骤上带来更大增益。
- **代理目标与感知质量的脱节**：ART、HSO 等新工作同样优化 Euler 离散误差或中点误差代理，绑定于代理与真实感知质量的对齐程度。

## 核心贡献（创新点）
1. **直接将目标指标作为优化对象**：OYS 用贝叶斯优化直接优化 HPS/FID/LPIPS/MSE 等目标指标，而非 KLUB、Euler 误差等代理，与 AYS/ART/HSO 形成本质区别。
2. **提出单调递减的累积积重参数化**：通过 $p_k \in [0,1]$ 的连乘构造单调递减时间步序列，适用于离散时间与连续时间两种扩散公式。
3. **联合优化采样超参数**：除时间步外，OYS 可同时优化 EDM 系列的 $\sigma_{\min}, \sigma_{\max}, \rho$、EMA 长度、引导强度等参数，扩展了可优化空间。
4. **高效收敛且泛化性强**：SDXL 约 17K 次 5 步生成（85K 次前向传播）即达到 plateau，比 AYS 上限低 1–2 个数量级；在文生图、inpainting、逆图像任务及蒸馏模型 SDXL-Turbo 上均有效。
5. **揭示被既有启发式忽略的调度规律**：短步数下应将更多步数分配到高噪声（大 timestep / 低 log-SNR）区域；EDM2 最优配置普遍压低 $\sigma_{\max}$（从 80 降至约 20）并收窄噪声范围。

## 方法详解
- **问题形式化**：给定固定预训练模型 $m$、调优输入集 $\mathcal{X}$、黑盒评估指标 $e$，求 $\mathbf{p}^* = \arg\min_{\mathbf{p} \in \mathcal{P}} e(m(\mathcal{X}; \mathbf{p}))$。
- **直接时间步优化（离散/连续）**：对 $K$-步调度，优化 $p_k \in [0,1]$ 并以累积积构造 $t_k = t_{\max} \prod_{i=1}^k p_i$，保证单调递减；离散模型最后取整 $q(\cdot)$。
- **参数化调度优化（EDM 系列）**：将 $\sigma_{\min}, \sigma_{\max}, \rho$、EMA 长度、guidance 强度等实参作为 $\mathbf{p}$ 的元素联合优化。
- **贝叶斯优化循环**：以 Sobol 采样（两倍参数数量）初始化；后续用 qLogNEI 采集函数迭代选择候选配置，高斯过程拟合后验；每个配置生成图像并计算指标，更新后验直至收敛或预算耗尽。
- **指标设计**：文生图用 HPS（基于 CLIP ViT-L/14 + Bradley-Terry 胜率）；inpainting 用 LPIPS；逆图像任务用 MSE；EDM2 用 FID。高者更优的指标取负后纳入最小化框架。

## 实验与结果
- **数据集/模型**：COCO Captions、DiffusionDB、ImageNet-512；DeepFloyd、SDv1.5、SDXL、SDXL-Turbo、FLUX.1-dev、QwenImage、EDM2 系列；inpainting 与 Prompt Diffusion 逆 HED/深度/分割任务。
- **基线**：Default 调度、AYS（log-linear 下采样至 5 步）、Karras 等网格搜索默认配置。
- **文生图（5 步，COCO）**：DeepFloyd HPS 0.252 vs Default 0.215，胜率 83.99%；SDXL HPS 0.245 vs 0.229，胜率 70.70%；SDv1.5 HPS 0.238 vs 0.230，胜率 59.29%；SDXL-Turbo 3 步 HPS 0.298 vs Default 0.287，胜率 83.35%。
- **DiffusionDB（5 步）**：FLUX.1-dev HPS 0.283（胜率 82.69%）；QwenImage HPS 0.193（胜率 77.19%）。
- **Inpainting**：SDXL PSNR 73.47 vs 66.59（+6.9 dB），HPS 胜率 93.40%；SDv1.5 PSNR 67.34 vs 64.46（+2.9 dB），胜率 85.10%。
- **Prompt Diffusion 逆任务**：Inverse HED PSNR 60.46 vs 59.79，胜率 65.43%；Inverse Depth 59.05 vs 58.51，胜率 57.62%。
- **EDM2 ImageNet-512**：FID 相对 Karras 2024 网格搜索配置提升 3.57%–7.00%。
- **人类评估**：58 名参与者，OYS 5 步 vs Default 在质量胜率 70.6%、对齐胜率 66.8%；vs AYS 质量 68.8%、对齐 62.9%，均 $p < 0.001$。
- **调优效率**：SDXL 约 17K 次 5 步生成即 plateau；AYS 上限报道约 240 万次生成，OYS 低 1–2 个数量级。

## 相关工作脉络
- **AYS (Sabour et al., 2024)**：最小化 KLUB 代理上界，局部零阶搜索 + 早停；本文直接优化目标指标并在全局空间搜索，调度在 log-SNR 空间显著偏离默认。
- **DDSS (Watson et al., 2022)**：通过采样过程反向传播优化可微代理（KID），需保存前向状态且显存随步数线性增长；OYS 无梯度、无需反向传播，适配整数时间步与非可微操作。
- **ART (Huang et al., 2026) / HSO (Zhu et al., 2026)**：同样优化 Euler 离散误差或中点误差代理；本文强调代理目标绑定于真实感知质量的风险，OYS 可直接用于无 tractable 代理的指标（如人类偏好、任务特定制约）。
- **Karras et al. (2022, 2024) EDM 系列**：网格搜索确定 $\sigma_{\min}, \sigma_{\max}, \rho$ 等；本文以 BO 替代穷举网格，发现 $\sigma_{\max}$ 大幅下调等反直觉但更优的配置。
- **Repaint (Lugmayr et al., 2022)**：针对 inpainting 的 zigzag 手动调度；OYS 自动发现适合 inpainting 的低步数调度，显著减少未掩码区域失真。

## 局限性与未来方向
- **调优成本仍为正样本生成开销**：虽远低于 AYS，但每模型/任务需一次性前向生成成本，未覆盖零生成成本的场景。
- **调度具领域依赖性**：最优噪声调度随任务/数据分布变化（作者引用 Chen 2023），难以给出通用调度；跨域迁移仍需重新调优。
- **参数搜索边界影响结论**：如 EDM2 的 $\sigma_{\max}$ 压至搜索下界 20，真实最优可能更低，需更大搜索空间验证。
- **未来方向**：扩展到视频/3D 生成、联合优化 sampler 与调度、探索更小调优预算或元学习跨任务调度。

## 研究启发与可借鉴点
- **直接优化目标指标的哲学**：在生成模型中存在非可微或无代理指标（人类偏好、任务损失）时，BO 黑盒优化是直接路线，可迁移至 LLM 采样策略、多模态排序等。
- **累积积重参数化保证单调性**：该技巧可用于任何需单调序列的超参优化（如学习率 schedule、温度 schedule），兼具连续优化与离散取整的兼容性。
- **联合优化调度与采样器超参**：OYS 对 EDM 参数的发现（$\sigma_{\max}$ 下调、guidance 按规模分化）提示可把"采样器设计"纳入自动搜索空间，而非仅依赖人工调参。
- **人类评估与自动指标双重验证**：本文以 HPS 为调优目标，但仍报告 FID/PSNR 提升与人类胜率，证明优化目标并非"刷分"，该评估范式值得复用。

## 关键术语表
- **OYS（Optimize Your Sampling）**：将扩散采样时间步选择视为黑盒优化问题并以贝叶斯优化直接求解的框架。
- **贝叶斯优化（Bayesian Optimization）**：基于高斯过程构建代理模型并利用采集函数（如 qLogNEI）序贯选择最优实验点的黑盒优化方法。
- **AYS（Align Your Steps）**：通过最小化 KLUB（KL 散度上界）代理目标来优化采样调度的 prior 方法。
- **KLUB（KL Divergence Upper Bound）**：真实生成 SDE 与其离散化之间的 KL 散度上界，AYS 的优化目标。
- **HPS（Human Preference Score）**：基于 CLIP ViT-L/14 与 Bradley-Terry 模型计算文本-图像对偏好分数的指标。
- **log-SNR**：信号-噪声比的对数，用于在不同扩散模型间统一表示噪声水平的空间。
- **EDM（Elucidating Diffusion Models）**：Karras 系列提出的扩散模型训练与采样统一框架，含 $\sigma_{\min}, \sigma_{\max}, \rho$ 等参数化调度。
- **SDXL-Turbo**：专为少步采样设计的蒸馏扩散模型，本文验证 OYS 在蒸馏模型上仍有效。

## 可复现要素
- **数据集**：COCO Captions、DiffusionDB、ImageNet-512、Wang et al. (2023) Prompt Diffusion 数据集；论文未声明新数据集。
- **代码/权重**：论文未明确声明开源仓库；使用了 DeepFloyd、SDXL、SDv1.5、FLUX.1-dev、QwenImage、EDM2 等公开模型。
- **关键超参**：Sobol 初始采样数为参数维度的 2 倍；采集函数为 qLogNEI；文生图 batch=128、inpainting batch=64；EDM2 搜索范围 $\sigma_{\min} \in [0.0001, 0.01]$、$\sigma_{\max} \in [20, 80]$ 等（详见附录表 9）。
- **采样器**：DeepFloyd/SDXL/SDv1.5 用 DPM-Solver++，其余用 Euler Discrete；inpainting 用 PNDMScheduler。
