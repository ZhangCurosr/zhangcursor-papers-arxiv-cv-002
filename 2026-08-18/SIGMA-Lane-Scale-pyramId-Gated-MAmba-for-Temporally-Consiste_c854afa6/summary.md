---
title: "SIGMA-Lane-Scale-pyramId-Gated-MAmba-for-Temporally-Consiste"
source: https://arxiv.org/pdf/2608.16338v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 14:26:06"
---

# 论文速读：SIGMA-Lane-Scale-pyramId-Gated-MAmba-for-Temporally-Consiste

## 一句话总结
针对大车持续遮挡导致视频车道检测时序状态被污染的问题，提出 SIGMA-Lane，通过在 SSM 写入路径与残差融合路径部署遮挡感知双门控，结合坐标一致性仿射对齐、结构空间检索与几何感知初始化，在流式推理下实现重遮挡场景的稳定车道检测。

## 研究问题与动机
- 视频车道检测需在帧间保持预测一致，但大型车辆遮挡会切断车道视觉线索，破坏时序连贯性。
- 现有流式循环模型（如 OMR 的 ConvLSTM）将当前帧特征直接写入隐状态，遮挡污染的特征一旦进入时序记忆便会持续累积，形成“状态污染”（state contamination）。
- 已有遮挡感知方法仅将障碍物掩码作为辅助输入参与 refine，未直接约束污染特征进入 SSM 状态更新的路径。
- 特征与掩码若独立形变对齐，会导致坐标不一致，使门控信号偏离真实遮挡区域，反而引入误抑制或漏过滤。

## 核心贡献（创新点）
- **状态污染的形式化与双门控更新规则**：首次将遮挡导致的时序错误抽象为 SSM 递推方程中的噪声累积问题，在写入路径（Input Gate）与残差融合路径（Output Gate）同时引入障碍物掩码调制，与仅将掩码作旁路输入的基线方法本质不同。
- **尺度金字塔 Mamba2 时序传播**：设计高低分辨率并行分支，低分支捕获广域时序上下文，高分支保留局部车道细节，通过可学习残差融合实现多尺度稳健传播。
- **车道引导的坐标一致性仿射对齐**：利用车道掩码全局平均池化聚焦稳定几何区域预测轻量 2D 仿射参数，对特征、车道掩码与障碍物掩码同步形变，从根本上消除多源信号的对齐偏移。
- **结构空间检索（SSR）与几何感知冷启动**：SSR 通过车道掩码嵌入引导的跨帧交叉注意力从历史对齐特征中检索缺失拓扑；起点前拼接含 2D 位置编码的可学习 Start Token，为递归状态提供结构化初值。

## 方法详解
- **基础架构**：继承 OMR 的 ResNet18 编码器与 Eigenlane 解码器，仅替换时序聚合模块；基于 Mamba2 构建状态更新核心，维护 8 帧记忆库 $\mathbf{F} \in \mathbb{R}^{(BHW)\times T\times C}$、$\mathbf{M} \in \mathbb{R}^{(BHW)\times T\times 1}$ 与上一帧车道掩码 $L_{t-1}$。
- **SSM-Consistent Dual-Gating**：
  - **输入门控**：$x_t' = x_t \odot (1 - m_t)$，直接衰减 SSM 写入项 $\bar{B}_t x_t$。理论推导表明，当遮挡严重（$m_t \to 1$）时，对角衰减算子 $D_t$ 趋零，污染噪声 $\eta_t$ 在写入隐状态前被抑制，误差界由 $\beta\sigma\gamma$ 控制（公式 4-9）。
  - **输出门控**：对遮挡掩码 GAP 聚合得标量 $\bar{m}$，经 MLP+Sigmoid 生成残差保留比 $r$。融合公式为 $X_{time} = (1-M) \odot (X_{in}+X_{out}) + M \odot (r \odot X_{in}+X_{out})$，在传播后防止当前污染特征经残差通路回渗。
  - **尺度金字塔**：低分辨率支 $\tilde{x}_t = x_{out} + \alpha_{\mathrm{low}} \cdot \mathrm{Upsample}(\mathrm{Mamba}(\mathrm{Pool}_s(X)))$，$\alpha_{\mathrm{low}}$ 初始化为小值保障训练稳定。
- **Lane-Guided Affine Warp**：利用 $\mathrm{GAP}(x_{t-1}; \ell_{t-1})$ 与 $\mathrm{GAP}(x_t)$ 拼接过 MLP+tanh 预测残差仿射参数 $\Delta\theta$，与单位阵相加得 $\theta$。通过双线性采样同步变换 $\tilde{x}_{t-1}, \tilde{\ell}_{t-1}, \tilde{m}_{t-1}$，确保门控信号与调制区域坐标严格对齐。
- **Structural Spatial Retrieval (SSR)**：$Q=\mathrm{LN}(x_t)$，$K=V=\mathrm{LN}(\tilde{x}_{t-1})+\mathrm{LN}(e(\tilde{\ell}_{t-1}))$，执行标准多头交叉注意力
