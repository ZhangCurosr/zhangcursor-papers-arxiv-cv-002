---
title: "TINA-Probing-Residual-Visual-Knowledge-in-Unlearned-Diffusio"
source: https://arxiv.org/pdf/2608.17747v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 06:11:14"
field: "生成模型安全与鲁棒性"
keywords: ["概念擦除", "扩散模型", "机器遗忘", "对抗攻击", "逆过程", "扩散一致性"]
innovations: ["首次揭示文本中心擦除范式未消除底层视觉知识的根本缺陷", "提出优化型零文本逆过程配合扩散一致性轨迹正则化", "通过边际能量约束抑制虚假轨迹，实现可靠视觉探针"]
benchmarks: ["裸露概念擦除(I2P)", "艺术风格擦除(Van Gogh)", "物体擦除(Imagenette)", "名人身份擦除(GCD)"]
---

# 论文速读：TINA+: Probing Residual Visual Knowledge in Unlearned Diffusion Models via Diffusion-Consistent Text-Free Inversion

## 一句话总结
论文揭示了现有概念擦除方法仅切断文本-图像映射而未真正消除底层视觉知识的根本缺陷，并提出 TINA+ 作为首个基于扩散一致性的文本无关逆攻击方法，通过优化零文本逆过程配合能量正则化，可靠探测已被擦除概念的残留视觉知识。

## 研究问题与动机
- **文本中心范式的根本局限**：现有概念擦除方法（如 ESD、AdvUnlearn、STEREO 等）的操作原则是将"擦除概念"等同于"切断文本-图像映射"，但并未真正从参数空间中消除底层的视觉知识。
- **现有攻击的依赖路径**：现有对抗攻击（PEZ、UDA、CCE 等）均围绕文本条件通路设计，试图寻找新的 adversarial text prompt 或 embedding 来绕过文本防御，无法探测纯视觉维度的残留知识。
- **标准逆过程的不足**：零文本（null-text）条件下标准 DDIM 逆过程因近似误差累积导致重建质量差；且无约束逆过程可能发现虚假轨迹（spurious trajectories），即使随机初始化模型也能"重建"目标概念。
- **缺乏可靠的视觉探针**：需要一种不依赖文本引导、且能区分真实残留知识与逆过程伪影的可靠评估工具。

## 核心贡献（创新点）
- **首次揭示文本中心擦除范式的根本缺陷**：证明切断文本-图像映射不等于消除底层视觉知识，为概念擦除安全性评估提供了全新视角。
- **提出 TINA+ 文本无关逆攻击框架**：通过优化零文本逆过程寻找扩散一致性的生成轨迹，完全绕过文本条件通路，实现纯视觉维度的残留知识探测。
- **引入扩散一致性轨迹正则化（Diffusion-Consistent Trajectory Regularization）**：推导边际能量约束（marginal energy constraint）以抑制虚假的低能量坍塌轨迹，提出前向边际初始化（forward marginal initialization）稳定低噪声区域的逆过程。
- **系统性实验验证**：在 12 种擦除方法、4 类概念擦除任务（裸露、风格、物体、名人身份）、不同架构（UNet 与 DiT）下验证 TINA+ 的有效性、可靠性与泛化能力。
- **理论-实验结合的新评估范式**：通过随机 UNet 负对照实验和轨迹诊断证明，仅凭视觉相似度不足以认证模型保留了概念知识，扩散一致性是必要检验标准。

## 方法详解

### 1. 优化型零文本逆过程（Optimization-Based Text-Free Inversion）
- **问题**：标准 DDIM 逆过程使用近似公式 $z_t \approx f_\theta(z_{t-1}, t, c)$，依赖前一步噪声预测，零文本条件下误差累积导致重建失败。
- **解决**：将精确 DDIM 关系视为不动点问题，定义隐式关系 $z_t = C_1(t)z_{t-1} + C_2(t)\epsilon_\theta(z_t, t, c_{\text{null}})$，通过梯度下降最小化固定点一致性损失：
  $$\mathcal{L}_{\text{fp}}(z_t) = \|f_\theta^*(z_t, z_{t-1}, t, c_{\text{null}}) - z_t\|_2^2$$
  每步以标准逆过程结果初始化，进行 K=25 次迭代优化。

### 2. 扩散一致性轨迹正则化
- **边际能量约束**：推导前向扩散边际的能量期望 $m_t = \bar{\alpha}_t\|z_0\|_2^2 + (1-\bar{\alpha}_t)D$ 和方差 $v_t$，构造标准化能量得分 $r_t^{\text{eng}} = (\|z_t\|_2^2 - m_t)/\sqrt{v_t}$，定义单边 hinge 损失：
  $$\mathcal{L}_{\text{eng}}(z_t) = \text{ReLU}(-r_t^{\text{eng}} - \tau_{\text{eng}})^2$$
  惩罚能量远低于扩散预期的轨迹，避免虚假重建。
- **前向边际初始化（FMI）**：跳过脆弱的低噪声区域（$t < \rho T$），直接从前向边际采样初始化：$z_{t_a}^{\text{fm}} = \sqrt{\bar{\alpha}_{t_a}}z_0 + \sqrt{1-\bar{\alpha}_{t_a}}\epsilon$，其中 $\rho = 0.1$。

### 3. 整体目标函数
$$\mathcal{L}_t^{\text{TINA+}} = \mathcal{L}_{\text{fp}}(z_t) + \lambda_{\text{eng}} \mathcal{L}_{\text{eng}}(z_t)$$
其中 $\lambda_{\text{eng}} = 1.0$，$\tau_{\text{eng}} = 10$，在 $t/\bar{T} \in [0.1, 1.0]$ 区间应用能量正则化。

### 4. 攻击流程
1. 将目标图像编码为潜变量 $z_0$
2. 执行优化型零文本逆过程得到种子潜变量 $z_T^*$
3. 在已擦除模型上用 $z_T^*$ 进行零文本采样生成图像
4. 用概念分类器判断是否成功重建被擦除概念

## 实验与结果

### 数据集与任务
- **裸露概念擦除**：I2P 数据集提示词
- **艺术风格擦除**：Van Gogh 风格，ESD 评估设置
- **物体擦除**：Church、Garbage Truck、Parachute、Tench（Imagenette 类别）
- **名人身份擦除**：Taylor Swift、Elon Musk、Adam Lambert

### 评估基线
- **12 种擦除方法**：ESD、FMN、AC、UCE、SH、MACE、SPM、RECE、AdvUnlearn、SalUn、EraseDiff、STEREO
- **6 种攻击方法**：PEZ、MMA、RAB、P4D、UDA、CCE（文本逆）

### 主要结果

**裸露概念擦除（Table 2）**：
- TINA+ 在所有 8 种防御上取得最高 ASR，平均值达 **91.74%**
- 对 AdvUnlearn、SalUn、STEREO 等抗攻击防御，TINA+ 分别达到 92.37%、73.73%、97.46%，而文本攻击在该类防御上通常低于 25%

**风格擦除（Table 3）**：
- TINA+ 平均 ASR 为 **66.8%**，显著优于 UDA（46.9%）
- 对 STEREO 防御，TINA+ 达到 46.0%，而所有文本攻击为 0%

**物体擦除（Table 4）**：
- Church：平均 ASR **89.50%**
- Garbage Truck：平均 ASR **79.25%**
- Parachute：平均 ASR **88.50%**
- Tench：平均 ASR **75.00%**
- 对现代防御（SH、STEREO），TINA+ 仍保持高成功率（66%-90%）

**名人身份擦除（Table 5）**：
- ESD 防御下：Taylor Swift 96%、Elon Musk 98%、Adam Lambert 94%
- STEREO 防御下：所有文本攻击 ASR 为 0%，TINA+ 仍达 94%、46%、64%

**文本逆 vs 图像逆对比（Table 6）**：
- TINA+ 在全部 6 组设定中系统性优于 CCE（文本逆方法）
- STEREO 下 Van Gogh：CCE 4% vs TINA+ 46%；Tench：CCE 2% vs TINA+ 66%

**随机 UNet 负对照（Table 7）**：
- TINA+ 成功抑制虚假重建：Taylor Swift 基准上 ASR 从 88% 降至 0%，LPIPS 从 0.245 升至 0.749
- 证明扩散一致性正则化的必要性

**DiT 架构泛化（Fig. 12）**：
- PixArt-XL-2（DiT 架构）上 ESD 擦除后普通采样 ASR 为 0%，TINA+ 达到 **84.0%**

## 相关工作脉络
- **文本中心擦除方法（ESD、FMN、UCE、AdvUnlearn 等）**：本文指出这些方法的操作原则是切断文本-图像映射，而非消除视觉知识，与本文形成核心对比。
- **文本导向对抗攻击（PEZ、P4D、UDA、RAB）**：依赖文本条件通路，在强防御下失效，本文通过绕过文本路径实现更可靠探测。
- **文本逆攻击（CCE）**：通过 textual inversion 寻找 surrogate embedding，本质仍是文本路径，本文证明图像逆在绕过文本防御方面更有效。
- **推理时干预方法（SLD）**：不修改模型参数，本文聚焦于参数级擦除方法的评估。
- **DDIM 逆过程（Null-text inversion）**：标准方法依赖文本条件，本文扩展为优化型零文本逆并引入扩散一致性约束。
- **机器遗忘（SalUn、SH、EraseDiff）**：通用遗忘框架，本文将其应用于概念擦除场景下的攻击评估。

## 局限性与未来方向
- **擦除方法针对性**：评估的 12 种方法各有适用领域（如 nudity 专用），未来需探索更通用的擦除范式。
- **计算开销**：每步逆过程需 25 次迭代优化，计算成本高于单次前向推理，未来可探索更高效的逆过程算法。
- **评估范围**：仅测试了 Stable Diffusion v1.4 和 PixArt-XL-2，更多架构（如 SDXL、SD3）的验证有待扩展。
- **防御演进**：本文揭示的漏洞可能推动新一代"视觉级"擦除方法的发展，形成新的攻防博弈。

## 研究启发与可借鉴点
- **范式转换价值**：从文本路径转向视觉潜空间探测的思路，可用于评估其他多模态模型的"擦除完整性"。
- **扩散一致性约束的可迁移性**：边际能量约束作为一种轨迹有效性检验标准，可推广至其他逆过程应用（如编辑、插值）。
- **负对照实验设计**：随机模型对照 + 轨迹诊断的组合，为攻击方法的可靠性评估提供了严谨范式。
- **潜在空间分析**：通过 t-SNE 可视化内部激活的分离性，展示了如何从表征角度验证概念残留，可借鉴于模型可解释性研究。
- **跨架构验证**：在 UNet 和 DiT 上均验证有效性，为方法泛化性提供了完整证据链。

## 关键术语表
- **概念擦除（Concept Erasure）**：通过修改模型参数移除特定概念生成能力的机器遗忘技术。
- **DDIM 逆过程**：利用确定性采样逆过程将图像潜变量映射回高噪声种子 latent 的技术。
- **零文本条件（Null-text condition）**：文本编码器输出空条件（empty prompt），使生成完全脱离文本引导。
- **扩散一致性（Diffusion Consistency）**：逆过程发现的轨迹应符合前向扩散边际的能量演化规律。
- **固定点一致性（Fixed-point Consistency）**：优化隐式 DDIM 关系使中间潜变量满足自洽约束。
- **边际能量（Marginal Energy）**：潜变量的二阶统计量 $\|z_t\|_2^2$，用于衡量轨迹是否符合扩散能量 envelope。
- **前向边际初始化（Forward Marginal Initialization, FMI）**：跳过低噪声区直接从 $q(z_t|z_0)$ 采样初始化逆过程。
- **攻击成功率（ASR）**：生成图像被分类器识别为包含被擦除概念的比例。

## 可复现要素
- **数据集**：I2P 数据集（裸露）、ESD 评估设置（风格）、Imagenette 类别（物体）、GIPHY Celebrity Detector（名人），均基于公开资源
- **代码/权重**：项目主页 https://qianlong0502.github.io/TINA-Plus-Homepage/，Stable Diffusion v1.4 为公开 checkpoint
- **关键超参**：DDIM 步骤数 50，优化迭代 K=25，学习率 η=0.001，FMI 比例 ρ=0.1，能量权重 λ_eng=1.0，能量容忍度 τ_eng=10，采样 CFG scale=7.5
- **硬件**：单张 NVIDIA A100 GPU
