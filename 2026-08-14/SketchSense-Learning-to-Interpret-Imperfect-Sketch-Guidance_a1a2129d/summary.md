---
title: "SketchSense-Learning-to-Interpret-Imperfect-Sketch-Guidance"
source: https://arxiv.org/pdf/2608.13186v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 04:03:04"
field: "条件图像修复与生成"
keywords: ["sketch-guided inpainting", "diffusion model", "multimodal generation", "structural control", "spatial regulation", "image restoration"]
innovations: ["同步RGB-结构双分支去噪框架，联合解释不完美草图", "双向注意融合+短语级跨分支语义一致性损失对齐双分支语义grounding", "内生草图可靠性估计与可选显式符号先验的双层空间调节机制"]
benchmarks: ["DhMural1714", "ArtBench", "COCO"]
---

# 论文速读：SketchSense: Learning to Interpret Imperfect Sketch Guidance for Image Inpainting

## 一句话总结
SketchSense 提出了一个同步去噪的 RGB-结构双分支框架，通过双向注意融合、短语级语义一致性损失和可学习的草图空间调节机制，实现对局部不完美草图的智能解读——既能保留用户有意保留的结构，也能纠正局部偏差，同时输出修复后的 RGB 图像和可观测的细化结构。

## 研究问题与动机
1. **真实草图可靠性空间异质**：用户草图往往同时包含可靠的宏观布局意图和局部拥挤、错位、不完整或有意的非传统笔触，将每条笔触视为同等可靠会将局部误差传播至整个生成过程。
2. **固定草图条件方法的缺陷**：现有方法（如 PDDP、OminiControl）在去噪全程将草图作为固定外部条件，虽可通过空间加权调节影响强度，但无法随生成过程动态演化草图解释，局部误差持续存在。
3. **结构优先方法的局限**：SketchRefiner、ZITS++ 等方法先恢复干净结构再合成 RGB，必须在颜色/纹理等外观线索出现之前解决结构歧义，结构性错误会被直接继承到后续 RGB 生成中。
4. **几何信息无法区分错误与意图**：仅凭草图几何形状无法判断一条非常规笔画是绘制错误还是用户的艺术意图，需要模型将草图与 evolving 的外观、结构和语义上下文关联起来进行空间自适应调节。

## 核心贡献（创新点）
1. **同步 RGB-结构理解框架**：与先修后色或固定条件的策略本质不同，RGB 和结构分支在同一去噪步协同演化，通过双向注意融合交换证据，使外观和语义上下文实时指导结构恢复。
2. **双向注意融合（Bidirectional Attention Fusion）**：在 FLUX DiT 块中引入专用的 joint-attention QKV LoRA，使 RGB 查询能检索当前边假说与原始草图证据，边查询能检索外观和语义上下文，零初始化残差投影保持预训练行为。
3. **短语级跨分支语义一致性损失**：针对双分支可能对同一物体短语进行不同空间定位的问题，提取本地名词短语并计算其在两分支中的空间分布 JS 散度，用熵置信度加权对齐两分支的语义 grounding，而非简单统一特征表示。
4. **内在草图感知空间调节（Intrinsic Sketch-Aware Spatial Regulation）**：通过 token-state 路径（边-草图状态差异）和 pixel-space 几何路径融合生成源可靠性权重 $w_{\text{rel}}$，并独立预测每个 transformer 块的空间残差门控 $g_b^l$，实现跨位置和跨深度的草图使用自适应。
5. **显式符号草图使用建模（Explicit Signed Sketch-Use Modeling）**：引入可选的空间符号先验 $u \in [-1,1]^{H\times W}$，分别用正/负/零值表达"保留/纠正/中性"意图，通过零初始化编码器、源键偏置和独立正负 LoRA 值变换影响 token 状态、草图选择和检索内容。

## 方法详解
**整体架构**：基于 FLUX.1 Fill [dev] 构建双分支同步去噪系统，RGB 分支恢复外观，结构分支将不完美草图解释为细化结构，两分支在同一 timestep $t$ 共享 flow matching 调度。

**初始状态**：$H_x^0 = \mathcal{P}_x(z_{x,t}, x, m, p)$ 为 RGB 分支初始 token，$H_{\text{str}}^0 = [\mathcal{P}_e(z_{e,t}, e, m, p); \mathcal{P}_s(z_s)]$ 为结构分支初始 token（拼接边缘目标和草图）。

**双向注意融合**：每个 fusion 块复用原生 QKV 投影并叠加专用 joint-attention QKV LoRA：$(\bar{Q}_b^l, \bar{K}_b^l, \bar{V}_b^l) = \text{QKV}_b^l(H_b^l) + \mathcal{L}_b^l(H_b^l)$。构建 $M_{x \to \text{str}}^l = \text{Attn}(\bar{Q}_x^l, \bar{K}_{\text{str}}^l, \bar{V}_{\text{str}}^l)$ 和 $M_{e \to x}^l = \text{Attn}(\bar{Q}_e^l, \bar{K}_x^l, \bar{V}_x^l)$，经零初始化投影 $W_0^l$ 后 Split 为 $\Delta H_x^l$ 和 $\Delta H_e^l$，由空间门控 $g_x^l, g_e^l$ 调制注入。

**短语级一致性损失**：用 LLM 提取短语 $\mathcal{P}(p)$，对每分支的短语相关注意力 $A_{b,k}^l$ 聚合得空间分布 $\pi_{b,k}^l = \text{Norm}(\text{Pool}_b(A_{b,k}^l))$，计算 JS 散度 $d_{lk}$，以两分支中更高的归一化熵置信度 $c_{lk}$ 加权：$\omega_{lk} = \sigma((\text{sg}(d_{lk})-\mu)/\tau) \cdot [c_0 + c_1 \text{clip}(c_{lk}/c_{\text{ref}}, 0, 1)]$，最终 $\mathcal{L}_{\text{phrase}} = \frac{1}{|\Omega|}\sum \omega_{lk} d_{lk}$，其中 sg 项防止反向传播改变散度本身。

**源可靠性预测**：$z_{\text{state}}^l = \Phi_{\text{state}}^l(H_e^l, H_s^l, H_s^l - H_e^l, t^l)$ 度量边-草图状态一致性；$z_{\text{geo}}^l = Z_0^l(\mathcal{P}_{h_l,w_l}(\Phi_{\text{geo}}^l([S, M, S \odot M])))$ 提取像素级几何。合并后经有界指数映射：$w_{\text{rel}}^l = \exp(\log \alpha_{\text{max}} \cdot \tanh(a^l))$，值 $>1$ 增强草图检索，$<1$ 衰减。

**残差空间门控**：$g_b^l = \sigma(G_{\text{res}}^l(H_b^l, \Delta H_b^l, m_b^l, v^l))$，同时观察接收状态和候选消息，结合空洞掩码和全局条件调节融合残差的注入量。

**显式符号先验**：$u = q \odot r$，经零初始化编码器 $E_b$ 投影为分支特定 embedding，加到初始 token：$\widetilde{H}_b^0 = H_b^0 + E_b(u, r, |u|)$。源键偏置：$\ell_{b,ij}^{l,h} \leftarrow \ell_{b,ij}^{l,h} + m_{b,i}^l \beta_b^{l,h} u_j$，其中 $\beta = \beta_{\text{max}} \tanh(\bar{\beta})$ 限制幅度。值变换：$\Delta V_s^l = \omega_+(u) B_+^l A_+^l \widehat{H}_s^l + \omega_-(u) B_-^l A_-^l \widehat{H}_s^l$，正负 LoRA 分别处理保留和纠正区域。

**训练目标**：流匹配损失 $\mathcal{L}_b = \mathbb{E}[\frac{\sum_i W_{b,i}\|\hat{v}_{b,i} - v_{b,i}^\star\|_2^2}{D_b \sum_i W_{b,i}}]$ 其中 $W_{b,i} = 1 + \eta_m m_{b,i} + \eta_r r_{b,i}$；总损失 $\mathcal{L} = \frac{\lambda_x \mathcal{L}_x + \lambda_e \mathcal{L}_e}{\lambda_x + \lambda_e} + \lambda_{\text{phrase}} \mathcal{L}_{\text{phrase}}$。

**训练样本构建**：用 MuGE 提取边缘标注，通过平滑形变场生成类用户草图，并在局部区域采样三种模式：正模式（更忠实于目标的修正结构 $u>0$）、负模式（更强扭曲/笔画删除/无关补丁 $u<0$）、中性模式（保留基础草图 $u=0$）。

## 实验与结果
**数据集**：ArtBench（权重 0.3）、COCO（权重 0.6）、DhMural1714（权重 0.1），分辨率 512×512，训练约 7 天单卡 A100 BF16，batch size 12，Adafactor lr=1e-5，推理 guidance scale=25。

**对比基线**：PDDP、MaGIC（adapted）、SketchRefiner（adapted）、OminiControl。评估指标为掩膜内 PSNR、SSIM、LPIPS。

**定量结果（Hard 子集）**：
- PDDP: PSNR 10.450 / SSIM 0.5237 / LPIPS 0.3883
- SketchRefiner: PSNR 14.386 / SSIM 0.5916 / LPIPS 0.3020
- OminiControl: PSNR 13.902 / SSIM 0.5824 / LPIPS 0.2479
- **SketchSense（无符号先验）: PSNR 14.933 / SSIM 0.6121 / LPIPS 0.2040**
- **SketchSense（有符号先验）: PSNR 14.898 / SSIM 0.6128 / LPIPS 0.2061**

相比最强基线 OminiControl，在 Hard 集上 PSNR 提升约 **1.03 dB**，LPIPS 降低约 **0.042**。Easy 集上相对 OminiControl PSNR 提升约 **1.34 dB**，LPIPS 降低约 **0.020**。带/不带符号先验在 RGB 指标上几乎相当，但符号先验显著提升 Region F1@2px。

**消融实验**：
- 去除跨分支融合：Edge F1@2px 从 0.7469 降至 0.6882
- 顺序恢复（先结构后 RGB）：LPIPS 从 0.1707 升至 0.1746，Edge F1 降至 0.6882
- 去除内在空间调节：LPIPS 从 0.1707 升至 0.2242，Edge F1 降至 0.7325
- 去除符号建模：正模式下 Region F1@2px 从 0.8107 降至 0.7344，负模式从 0.6266 降至 0.5935

## 相关工作脉络
1. **固定草图条件方法**：PDDP（[16]）通过部分离散扩散过程使用草图作为条件；OminiControl（[18]）用通用 control 机制处理多模态条件。本文区别于它们的核心在于草图条件不是固定的——而是随生成过程动态演化和解读。
2. **结构优先方法**：SketchRefiner（[13]）先精炼用户草图再 inpainting；ZITS++（[2]）和 Line Drawing 引导方法（[9]）在颜色合成前恢复线/边表示。本文认为结构歧义应在外观线索出现后联合解决，而非孤立地先验结构。
3. **结构感知 inpainting**：SmartBrush（[21]）和 Structure Matters（[14]）将结构作为有效条件，但未处理"结构本身不可靠"的情形；本文明确建模草图可靠性的空间异质性。
4. **联合多模态生成**：JointDiT（[8]）和 JointNet（[22]）耦合 RGB 与 dense 模态（depth）；本文聚焦 RGB-structure 双分支，且特别解决跨分支短语 grounding 不一致的问题。
5. **空间条件控制**：S-CFG（[17]）用语义区域依赖的 guidance 替代全局 scale；Conditional Balance（[3]）研究层依赖敏感性；Dense T2I Attention Modulation（[7]）实现 region-specific 注意力调节。本文进一步引入了内生状态依赖的可靠性感知和可选的用户显式意图输入。
6. **文本-区域对齐**：Grounded T2I（[15]）通过 cross/self-attention 控制改善 text-layout 对应；MTADiffusion（[6]）结合 edge 预测与 mask-text 对齐。本文的短语级一致性损失与其精神相似，但应用于跨模态双分支而非单一生成路径。

## 局限性与未来方向
1. **双分支效率开销**：同步运行两个分支带来约两倍的计算量，作者明确承认"more efficient dual-stream modeling"是未来方向。
2. **分辨率限制**：当前仅在 512×512 下评测和训练，更高精度恢复尚未探索。
3. **符号先验的用户交互成本**：显式 signed prior 需要用户提供正/负/中性意图标注，虽然提高了可控性但增加了交互负担，简化交互方式是作者提到的未来工作。
4. **训练数据的草图合成保真度**：base sketch 通过平滑形变场合成，可能无法完全覆盖真实用户草图的全部误差模式（如手绘抖动、语义性偏离等）。
5. **FLUX 基座依赖**：方法构建在 FLUX.1 Fill 之上，在其它架构上的泛化性有待验证。

## 研究启发与可借鉴点
1. **同步双分支而非串行**：结构-外观联合去噪的思路可迁移到其它条件生成任务中（如 depth-depth-image、segmentation-image），避免串行方法中早期错误固化问题。
2. **短语级跨分支语义对齐损失**：用 LLM 提取短语后计算空间分布 JS 散度的思路，可用于任何多模态双分支模型的一致性正则化，尤其在图文-深度、图文-草图等组合场景中有直接复用价值。
3. **零初始化残差融合**：双向注意融合中零初始化 $W_0^l$ 的设计可安全地插入到任何预训练 DiT 模型中而不破坏已有能力，是一种低风险的模块升级策略。
4. **内生可靠性估计 + 显式先验的双层设计**：先用模型内生学习可靠性（无需用户标注），再提供可选的显式符号先验作为用户补充控制——这种"隐式+显式"双层设计可推广到其它受噪声/不完美条件干扰的生成任务。
5. **流式训练样本模式合成**：通过在草图上采样正/负/中性三种模式来构造带 signed prior 的训练数据，可在无真实用户标注的情况下模拟多样化的草图使用意图，这一数据合成策略值得参考。

## 关键术语表
**Synchronous RGB-Structure Understanding**：RGB 和结构两个分支在同一去噪 timestep 同步演化，相互交换证据而非串行处理。
**Bidirectional Attention Fusion**：在两个分支间建立双向注意力通道，使 RGB 查询能检索结构证据、结构查询能检索外观上下文。
**Phrase-Level Cross-Branch Semantic Consistency**：通过 LLM 提取短语、计算两分支在短语注意力上的空间分布 JS 散度，实现对齐跨模态语义 grounding。
**Intrinsic Sketch-Aware Spatial Regulation**：内生地根据边-草图状态一致性和像素几何推断草图可靠性，并据此自适应调节跨分支融合注入量。
**Explicit Signed Sketch-Use Modeling**：用户可选地提供空间符号先验（正/负/零），显式表达保留或纠正意图，通过键偏置和值 LoRA 影响生成行为。
**Flow Matching**：用于扩散模型的训练目标，预测速度场 $v^\star = \epsilon - z$ 而非噪声本身，与 rectified flow 等价。
**Sketch Refinement Trajectory**：结构分支在去噪过程中产生的中间 edge latent 序列，作为模型对原始草图解释的可观测输出。
**Region F1@2px**：在评估区域内、以 2px 容忍度计算的 F1 分数，用于衡量细化结构与 ground truth 边缘的对齐质量。

## 可复现要素
- **数据集**：ArtBench、COCO、DhMural1714（论文未明确声明各数据集是否公开可商用，COCO 和 ArtBench 已知公开，DhMural1714 来源于文献 [9]）
- **代码/权重**：论文未提及开源声明
- **关键超参**：training 7 days / single NVIDIA A100 / BF16 / batch_size=12 / lr=1e-5 (Adafactor) / guidance_scale=25 / resolution=512×512 / dataset sampling weights [0.3, 0.6, 0.1]
