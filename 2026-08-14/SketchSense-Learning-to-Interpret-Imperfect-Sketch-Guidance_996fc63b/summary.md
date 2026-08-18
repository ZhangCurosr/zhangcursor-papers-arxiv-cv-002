---
title: "SketchSense-Learning-to-Interpret-Imperfect-Sketch-Guidance"
source: https://arxiv.org/pdf/2608.13186v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 04:03:05"
---

# 论文速读：SketchSense: Learning to Interpret Imperfect Sketch Guidance for Image Inpainting

## 一句话总结
论文提出 SketchSense，通过同步去噪 RGB 外观与结构双分支，结合双向注意力融合、短语级语义对齐、内在草图可靠性预测与可选的符号先验，实现对非完美手绘草图的动态解读与空间自适应修复，显著提升图像修复的视觉质量与结构保真度。

## 研究问题与动机
- 真实用户草图具有显著的空间异质性：全局布局意图可靠，但局部 Stroke 常出现拥挤、位移、残缺或刻意非常规形状；现有固定条件方法将所有笔触等权对待，导致局部绘制错误在整个去噪过程中被持续放大。
- 主流“先精炼结构再生成 RGB”的串行方案必须在外观特征（色彩、材质、纹理）涌现前就消除结构歧义，一旦早期结构估计出错，后续 RGB 生成只能被动继承该错误。
- 仅凭草图几何无法区分“绘制误差”与“用户刻意设计”，模型必须结合 evolving appearance 与语义上下文动态调整对草图的信任度，而非依赖全局权重或固定区域掩码。
- 现有空间控制方法（如 attention modulation、classifier-free guidance 调节）假设条件语义已明确，缺乏对“保留 vs 修正”二义性的显式建模能力，难以支撑精细的人机协同交互。

## 核心贡献（创新点）
1. **同步 RGB-结构理解框架**：在共享时间步并行维护噪声 RGB 与边缘潜变量，通过双向注意力融合实现外观重建与结构精炼的协同演进；与固定条件或串行“结构优先”范式的本质区别在于条件与生成的相互解耦与动态校准。
2. **短语级跨分支语义一致性**：针对同一段 prompt 在双分支中可能产生空间定位漂移的问题，利用 LLM 提取名词短语并约束两分支在该短语上的空间注意力分布相似度；与通用多模态耦合方法的本质区别在于专门解决跨流接地偏差而非单流内部对齐。
3. **内在草图感知空间调节**：融合当前边缘-草图状态差与原始像素几何，预测局部可靠性地图 $w_{\rm rel}$ 与残差注入门控 $g_b^l$，实现空间位置与 Transformer Block 级的自适应草图利用；与全局注意力缩放方法的本质区别在于引入可学习的动态置信度估计而非静态规则。
4. **显式符号草图利用建模**：通过可选的符号空间先验 $u \in [-1, 1]$ 显式传达“保留/修正/中性”意图，并分别通过键值偏置与符号分离 LoRA 调制 token 表示与注意力行为；与纯几何驱动方法的本质区别在于引入人类可解释的意图信号以解除几何二义性。

## 方法详解
- **基础架构与初始化**：基于 FLUX.1 Fill [dev]，在共享去噪时间步 $t$ 维护噪声 RGB 潜变量 $z_{x,t}$ 与边缘潜变量 $z_{e,t}$，初始 token 分别通过投影算子 $\mathcal{P}_x, \mathcal{P}_e, \mathcal{P}_s$ 生成；草图 token 全程作为源证据保留，边缘目标 token 在每个 Transformer 块中与外观上下文共同演进。
- **双向注意力融合（Bidirectional Attention Fusion）**：每个块复用原生 QKV，额外引入联合注意力 QKV 的 LoRA 分支（$\mathcal{L}_b^l$）。边缘查询 attend 到 RGB 状态获取外观/语义上下文，RGB 查询 attend 到边缘+草图状态获取最新结构假设；零初始化投影 $W_0^l$ 将双向消息分割为 $\Delta H_x^l$ 与 $\Delta H_e^l$，后续通过空间门控注入各自分支。
- **短语级跨分支语义一致性**：用 LLM 从 prompt 中提取名词短语集合 $\mathcal{P}(p)$，将各短语与 prompt token span 对齐，聚合两分支在匹配 span 与 head 上的注意力得到空间分布 $\pi_{b,k}^l$，计算 Jensen-Shannon 散度 $d_{lk}$。权重 $\omega_{lk}$ 由软 sigmoid 与归一化熵置信度 $c_{lk}$ 共同决定，stop-gradient 防止梯度污染；仅在数值有效且匹配成功的 $(l,k) \in \Omega$ 上计算 $\mathcal{L}_{\rm phrase}$。
- **内在草图感知空间调节**：可靠性预测器 $w_{\rm rel}^l$ 融合两条路径——状态路径 $\Phi_{\rm state}^l(H_e^l, H_s^l, H_s^l - H_e^l, t^l)$ 度量当前边缘与草图的 agreement，几何路径 $Z_0^l(\mathcal{P}_{h_l,w_l}(\Phi_{\rm geo}^l([S,M,S\odot M])))$ 提取原始草图/掩码几何；两者经有界指数映射为正值，$w>1$ 促进草图检索，$w<1$ 衰减。边缘侧据此调制草图 key 得分；residual-gate 预测器 $g_b^l = \sigma(G_{\rm res}^l(H_b^l, \Delta H_b^l, m_b^l, v^l))$ 门控双向融合残差的注入强度，实现 Block 级自适应。
- **显式符号草图利用建模**：用户或下游模块提供符号先验 $u = q \odot r$（$q\in[-1,1]$ 表达正/负/中性意图，$r\in[0,1]$ 表达支持范围）。零初始化空间编码器 $E_b$ 将 $[u, r, |u|]$ 投影
