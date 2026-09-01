---
title: "Optimize-Your-Sampling-Tuned-Difusion-Sampling-with-Bayesian"
source: https://arxiv.org/pdf/2608.18040v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 00:59:37"
field: "扩散模型高效采样"
keywords: ["diffusion sampling", "Bayesian optimization", "sampling schedule", "text-to-image", "few-step generation", "human preference score"]
innovations: ["将扩散采样时间步选择建模为黑盒优化问题，直接优化目标指标而非理论代理", "提出累积乘积重参数化保障单调调度，兼容离散/连续时间模型", "通过贝叶斯优化发现低步数调度应将更多时间步分配至高噪声区域"]
benchmarks: ["COCO Captions", "DifusionDB", "ImageNet-512", "Prompt Diffusion Inverse Tasks"]
---

# 论文速读：Optimize-Your-Sampling-Tuned-Difusion-Sampling-with-Bayesian

## 一句话总结
论文提出 OYS（Optimize Your Sampling）框架，将扩散模型采样时间步的选择问题建模为黑盒优化问题，直接使用目标质量指标（如 HPS、FID、LPIPS）通过贝叶斯优化进行搜索；该方法无需额外训练，在文本生成图像、图像修复及逆向图像任务上均显著优于默认调度与 AYS，5 步调度可保留 50 步调度 89%–94% 的质量并降低 10 倍推理成本。

## 研究问题与动机
- 扩散模型迭代去噪需要大量前向传播，计算开销大；尽管高效采样器和少步蒸馏模型研究众多，但对**采样时间步本身的选择**缺乏系统探索。
- 现有代表工作 AYS 将时间步优化转化为最小化 KL 散度上界（KLUB）等理论代理目标，而非直接优化最终感知质量指标，且在启发式初始化附近做局部调整，难以在低步数预算下全局重分配时间步。
- 默认 log-linear 调度在步数极低（如 5 步）时质量急剧退化，亟需一种不依赖可微代理、能直接优化人类偏好指标的方法。

## 核心贡献（创新点）
- **直接优化目标指标而非代理**：OYS 将采样参数选择视为黑盒优化，以 HPS/FID/LPIPS 等最终评判指标为优化目标，与 AYS/ART/HSO 等基于 KLUB 或离散化误差代理的方法形成本质区别。
- **贝叶斯优化全局搜索调度空间**：使用高斯过程与 qLogNEI 采集函数在完整配置空间中序列化搜索，无需梯度，可一次性重分配全部时间步，避免 AYS 的局部细化局限。
- **累积乘积重参数化保障单调性**：提出 $s_k = t_{\max}\prod_{i=1}^k p_i$ 的参数化形式，天然保证调度单调递减，并兼容离散时间模型的取整约束。
- **广泛适用性与效率**：无需重训练，适用于任意预训练扩散模型与采样器；SDXL 约 17K 次 5 步生成（~85K 次前向传播）即收敛，远低于 AYS 估算的百万级生成成本。
- **揭示低步数调度的新规律**：发现 OYS 最优调度在 log-SNR 空间中将更多时间步分配给**高噪声区域**，与默认/AYS 近似均匀分布截然不同；EDM2 系列中 $\sigma_{\max}$ 被压至 20（默认 80），$\sigma_{\min}$ 放大 5 倍。

## 方法详解
- **问题形式化**：给定固定预训练扩散模型 $m$、调优输入集 $\mathcal{X}$ 和黑盒评估指标 $e$，求解 $\mathbf{p}^* = \arg\min_{\mathbf{p}\in\mathcal{P}} e(m(\mathcal{X};\mathbf{p}))$，其中 $\mathbf{p}=\{p_1,\dots,p_M\}$ 为采样参数。
- **直接调度参数化（非参数化模型）**：对 $K$ 步调度，令 $p_k\in[0,1]$，定义 $s(\mathbf{p}) = [t_{\max}p_1,\; t_{\max}p_1p_2,\; \dots,\; t_{\max}\prod_{i=1}^K p_i]$；离散时间模型额外对 timestep 取整 $q(\cdot)$。该参数化使后段时间步集中在较小值，但高斯过程代理可补偿此偏差。
- **参数化调度优化（EDM 系列）**：将 $\sigma_{\min},\sigma_{\max},\rho$ 及 EMA length、guidance strength 共同作为 $\mathbf{p}$ 的元素联合优化，时间步由调度函数间接确定。
- **贝叶斯优化循环**：初始 Sobol 采样数设为参数维度的 2 倍；拟合高斯过程后切换至 qLogNEI 采集函数迭代选取候选配置；每轮生成图像并用目标指标评分，更新后验分布直至收敛或预算耗尽。

## 实验与结果
- **数据集与基线**：COCO Captions、DifusionDB、ImageNet-512、Prompt Diffusion 数据集；基线包括 Default、AYS（经 log-linear 插值适配至 5 步）。
- **文本生成图像（COCO Captions，5 步）**：
  - DeepFloyd：OYS HPS 0.252，胜率 83.99%（默认 0.215，AYS 0.225/65.39%）
  - SDv1.5：OYS 0.238，胜率 59.29%（默认 0.230，AYS 0.231/52.30%）
  - SDXL：OYS 0.245，胜率 70.70%（默认 0.229，AYS 0.210/25.29%）
  - SDXL-Turbo（3 步）：OYS 0.298，胜率 83.35%（默认 3 步 0.287）
- **文本生成图像（DifusionDB，5 步）**：FLUX.1-dev HPS 0.283/胜率 82.69%；QwenImage HPS 0.193/胜率 77.19%，均显著优于默认。
- **图像修复（5 步，PNDMScheduler，优化 LPIPS）**：SDXL PSNR 66.59→73.47（+6.9 dB），HPS 胜率 93.40%；SDv1.5 PSNR 64.46→67.34（+2.9 dB），胜率 85.10%。
- **逆向图像任务（Prompt Diffusion，5 步，优化 MSE）**：Inverse HED HPS 胜率 65.43%，Inverse Depth 57.62%，Inverse Segmentation 58.80%，HPS 平均得分全面提升。
- **EDM2 系列（ImageNet-512，优化 FID）**： Across XS–XXL，FID 相对 Karras et al. (2024) 网格搜索结果提升 3.57%–7.00%。
- **人类评估（SDXL，58 名参与者）**：OYS 在图像质量胜率 68.8%（vs AYS）、70.6%（vs Default）；文本对齐胜率 62.9%（vs AYS）、66.8%（vs Default），all $p<0.001$。
- **优化效率**：SDXL 约 17K 次 5 步生成（85K 前向传播）后性能 plateau；相比 AYS 估算上限（~2.4M 生成）低 1–2 个数量级。

## 相关工作脉络
- **AYS（Sabour et al., 2024）**：最小化 KLUB 代理目标，零阶局部调整；OYS 直接优化目标指标，全局贝叶斯搜索，定位在对立面。
- **DDSS（Watson et al., 2022）**：反向传播通过采样链优化感知分数，需可微代理且内存随步数线性增长；OYS 黑盒评估、无梯度、无状态累积问题。
- **ART（Huang et al., 2026）**：最优控制形式最小化 Euler 离散化误差；仍属代理优化，OYS 直接优化 HPS/FID 等最终指标。
- **HSO（Zhu et al., 2026）**：双层全局+局部代理优化；OYS 统一使用贝叶斯优化，无需分离搜索阶段。
- **Karras et al. (2022, 2024) EDM**：EDM 参数化调度通过昂贵网格搜索调参；OYS 以贝叶斯优化替代，效率更高且可扩展至 EMA/guidance 等超参。
- **Repaint（Lugmayr et al., 2022）**：为图像修复手工设计 zigzag 调度；OYS 自动化学习最优调度，无需人工设计。

## 局限性与未来方向
- 优化仍需数千次图像生成评估，对超大模型或高分辨率任务成本仍不可忽视。
- 最优调度具有任务/领域依赖性，非通用；每次更换模型或下游任务需重新调优。
- 实验集中于图像域（文本生成、修复、逆向），视频生成等时序任务尚未探索。
- 对非-differentiable 操作（如 timestep 取整）兼容良好，但未系统讨论与可微采样器（如 DPM-Solver++）的梯度交互。

## 研究启发与可借鉴点
- **黑盒优化思维迁移**：贝叶斯优化直接优化目标指标的范式可复用于 VAE 重建超参搜索、GAN 架构搜索、视频生成步数调度等。
- **累积乘积单调参数化**：该技巧可推广至任何需有序序列优化（如学习率 schedule、注意力 head 选择）的场景。
- **低步数高噪声优先规律**：5 步调度应将更多计算分配至高噪声区，这一发现可指导少步蒸馏采样器的设计先验。
- **与蒸馏正交互补**：OYS 在 SDXL-Turbo 等已蒸馏模型上仍有效，说明调度优化可与模型压缩结合实现双重加速。
- **多指标扩展潜力**：框架天然支持任意黑盒指标（人类偏好、任务特定重建误差），可无缝对接 LLM-as-judge 等新评估范式。

## 关键术语表
- **OYS（Optimize Your Sampling）**：将扩散采样参数选择建模为黑盒优化问题的框架，直接优化目标质量指标。
- **采样调度（Sampling Schedule）**：逆向扩散过程中选取的时间步有序序列，决定去噪轨迹。
- **log-SNR**：信号噪声比的对数，用于在不同扩散模型间统一表征噪声水平。
- **HPS（Human Preference Score）**：基于 CLIP ViT-L/14 微调的人类偏好评分，衡量文本-图像对齐与感知质量。
- **KLUB（KL Divergence Upper Bound）**：AYS 使用的理论代理目标，真实生成 SDE 与其离散近似之间的 KL 散度上界。
- **贝叶斯优化**：通过高斯过程代理与采集函数（如 qLogNEI）序列化搜索黑盒函数最优值的无梯度优化方法。
- **EDM（Elucidating Diffusion Models）**：Karras 等人提出的扩散模型训练与采样统一框架，由 $\sigma_{\min},\sigma_{\max},\rho$ 等参数化调度。
- **Prompt Diffusion**：通过上下文学习（in-context learning）在单次推理中完成多种图像变换任务的扩散模型。

## 可复现要素
- **数据集**：COCO Captions（公开）、DifusionDB（公开）、ImageNet-512（公开）、Wang et al. (2023) Prompt Diffusion 数据集（公开）。
- **代码/权重**：论文未明确声明开源仓库；模型权重使用公开预训练版本（DeepFloyd、SDXL、SDv1.5、FLUX.1-dev、QwenImage、SDXL-Turbo、EDM2 系列）。
- **关键超参**：Sobol 初始采样数 = 2×参数维度；采集函数 qLogNEI；文本生成 batch size=128，修复 batch size=64；优化收敛阈值论文未明确给出。
