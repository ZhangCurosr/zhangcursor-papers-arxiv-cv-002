---
title: "Steering-the-Flow-Inverting-Face-Recognition-Models-via-Grad"
source: https://arxiv.org/pdf/2608.16791v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 06:18:45"
---

# 论文速读：Steering-the-Flow-Inverting-Face-Recognition-Models-via-Grad

## 一句话总结
提出SFMI（Steering Flow Model Inversion），一种两阶段白盒模型反演攻击方法，将身份重建重构为基于Flow Matching的确定性ODE轨迹导向任务；通过预训练通用人脸流匹配先验约束生成流形，并在采样过程中注入时间自适应的渐进梯度信号，实现了对目标人脸识别模型的高精度、高视觉保真度身份恢复。

## 研究问题与动机
- **核心问题**：在完全白盒假设下，如何从人脸识别模型的参数中稳定、高保真地逆向重建出属于特定目标的敏感训练图像。
- **现有方法不足**：
  1. 早期像素空间优化方法直接在高维非凸景观中搜索，极易陷入局部最优，重建图像充满高频噪声且语义断裂。
  2. GAN-based方法（如GMI、PPA、PLGMI、IFGMI）将搜索空间限制在潜空间，但潜空间到图像的映射高度非线性且非光滑，梯度回传不稳定，难以精确逼近目标身份特征。
  3. Diffusion/SDE-based方法（如DiffMI、FGMIA）引入的随机噪声会持续干扰采样轨迹，且缺乏数学上严密的机制根据目标模型反馈实时修正生成路径；FGMIA仅做一次特征引导，缺乏迭代反馈。
- **动机**：Flow Matching通过连续ODE描述生成过程，速度场平滑可微；结合时间依赖的梯度引导调度器，可将反演转化为可控的轨迹导向任务，兼顾身份对齐强度与流形贴合度，系统性解决GAN非凸性与扩散随机性的优化失稳问题。

## 核心贡献（创新点）
1. **首个基于Flow Matching的白盒模型反演框架（SFMI）**：将反演重新表述为ODE速度场导向问题，以确定性轨迹替代GAN潜优化与SDE随机采样，从根本上缓解优化不稳定性。
2. **渐进式梯度引导调度器（PGS）**：设计包含预热、维持、衰减、松弛四阶段的时间依赖调制策略，动态平衡身份引导力与流形自然演化，避免恒定强引导导致的早期崩溃或晚期畸变。
3. **身份不相交交叉评估下的SOTA性能**：在CelebA严格去重协议上，对6种主流人脸模型攻击均取得最高ACC与最低LPIPS；在ArcFace上以0.9248的ACC、22.61的FID和0.3874的LPIPS刷新白盒反演记录，并验证对跨分布先验与轻量级防御的鲁棒性。

## 方法详解
SFMI分为两个紧密耦合的阶段，核心原理如下：
- **Stage I：预训练通用Flow Matching先验**
  - 在公开人脸数据集$\mathcal{D}_{\mathrm{pub}}$上训练无条件的FM模型$\mathcal{M}_\phi$，学习从标准高斯噪声$p_0=\mathcal{N}(0,\mathbf{I})$到人脸流形$p_1$的平滑映射。
  - 采用最优传输（OT）直线轨迹$x_t = (1-t)x_0 + tx_1$，预测网络直接回归干净图像$\hat{x}_1$，诱导速度场为$v_\phi(x_t,t) = (\mathcal{M}_\phi(x_t,t) - x_t)/(1-t)$。
  - 训练损失为速度匹配损失$\mathcal{L}_{\mathrm{FM}} = \mathbb{E}[\|v_\phi - u_t\|_2^2]$（$u_t=x_1-x_0$），时间步$t$按Logit-Normal分布采样以聚焦关键过渡时段。
- **Stage II：渐进梯度引导攻击（PGS）**
  - **清洁估计与身份损失**：采样时，将当前含噪状态$x_t$输入冻结的FM先验得到$\hat{x}_1$，再送入目标分类器$f_\theta$计算身份损失$\mathcal{L}_{\mathrm{id}}$；实际采用最大间隔损失（MMLoss）：$\mathcal{L}_{\mathrm{MM}} = \max_{k \neq y_{\mathrm{target}}} z_k(\hat{x}_1) - z_{y_{\mathrm{target}}}(\hat{x}_1)$，拉大目标logit与最强干扰类的差距。
  - **端到端梯度回传**：误差沿冻结的目标模型与冻结的FM先验反向传播，获得针对当前状态$x_t$的指导梯度$g(x_t) = \nabla_{x_t}\mathcal{L}_{\mathrm{inv}}$，确保梯度准确反映微小状态扰动对最终身份目标的影响。
  - **梯度归一化与速度场修正**：将梯度归一化至与当前速度场模长一致$g_{\mathrm{norm}} = g/\|g\|_2 \cdot \|v_\phi\|_2$，修正速度场为$\tilde{v} = v_\phi - \gamma(t) \cdot g_{\mathrm{norm}}$。
  - **四阶段PGS调度**：$\gamma(t)$包含线性预热($0\to t_0$)、恒强维持($t_0\to t_1$)、余
