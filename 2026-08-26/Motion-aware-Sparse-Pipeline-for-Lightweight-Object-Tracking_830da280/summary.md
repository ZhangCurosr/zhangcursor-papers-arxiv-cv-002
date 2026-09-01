---
title: "Motion-aware-Sparse-Pipeline-for-Lightweight-Object-Tracking"
source: https://arxiv.org/pdf/2608.24365v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 23:52:47"
---

# 论文速读：Motion-aware Sparse Pipeline for Lightweight Object Tracking

## 一句话总结
本文提出MaST（Motion-aware Sparse Tracker），一种面向边缘设备的端到端稀疏视觉跟踪框架。通过注入轻量级时序运动先验指导早期Token剪枝，并设计原生稀疏预测头，在保障跟踪精度的同时实现边缘平台上的高帧率实时推理。

## 研究问题与动机
1. **计算瓶颈限制边缘部署**：单流Transformer跟踪器凭借交叉注意力联合建模模板与搜索区取得高精度，但密集的Token序列处理带来巨大计算开销，难以在无人机、移动机器人等资源受限平台上实时运行。
2. **现有剪枝策略过早或过晚**：早期Transformer层的交叉注意力图弥散且噪声大，OSTrack、FARTrack等方法保守地将Token剪枝推迟至中间层，导致计算最昂贵的早期层仍处理全量Token。
3. **剪枝判据缺乏时序空间先验**：现有重要性评分仅依赖当前帧外观相关性，忽略了跟踪任务中目标位移平滑的强时序连续性，导致早期层直接剪枝时易误删目标Token。
4. **稀疏骨干与密集预测头不匹配**：卷积预测头要求输入为规则的2D特征网格，激进稀疏化后必须将Token填充并重排回全图，不仅浪费计算，且高压缩比下目标中心Token可能被误删，造成严重定位偏差。

## 核心贡献（创新点）
1. **运动先验引导的单层Token稀疏化**：将上一帧预测框生成的2D高斯运动窗口与交叉注意力分数融合，使Token筛选可安全提前至编码器第一层执行。*与已有工作的本质区别*：不同于引入辅助预测网络或依赖扩散注意力的方法，本文利用零成本的跨帧位置先验解决早期层Token选择不可靠的难题。
2. **原生稀疏预测头（Score-first, Regress-once）**：设计直接作用于保留的非结构化Token的轻量MLP头，先对所有Token计算置信度选出最优候选，再仅对该候选执行一次边界框回归。*与已有工作的本质区别*：彻底摒弃Padding与2D重排操作，使预测头复杂度严格跟随稀疏Token数量，消除冗余计算。
3. **端到端稀疏流水线与帕累托最优**：从早期Token筛选到最终Box解码全程保持稀疏性，在LaSOT、TrackingNet等基准上建立轻量级跟踪器的新SOTA，并在Jetson Nano/Raspberry Pi等边缘硬件上实现显著加速。*与已有工作的本质区别*：打破“稀疏骨干+密集头”的折衷惯例，实现从Tokens到Boxes的完整稀疏管线。

## 方法详解
- **整体架构**：模板图$\boldsymbol{Z}$与搜索图$\boldsymbol{X}$经Patch Embedding后送入Transformer编码器。在第一层编码器后插入稀疏化模块，仅保留Top-K搜索Token，后续所有Transformer块均在该稀疏集上运行，最终由稀疏预测头解码目标。
- **重要性分数计算**：简化交叉注意力聚合，仅使用模板中心Token作为Key代表特征，计算第$i$个搜索Token的分数：
  $$s_i = \frac{\exp(\mathbf{q}_i^\top \mathbf{k}_c / \sqrt{d})}{\sum_{k=1}^{P_x} \exp(\mathbf{q}_k^\top \mathbf{k}_c / \sqrt{d})}$$
- **运动先验注入**：基于上一帧预测框$\mathbf{b}_{t-1}=(x_{t-1}, y_{t-1}, w_{t-1}, h_{t-1})$构建2D高斯运动窗口：
  $$\mathcal{G}_t(u,v) = \exp\left(-\frac{(u-x_{t-1})^2}{2\sigma_x^2} - \frac{(v-y_{t-1})^2}{2\sigma_y^2}\right),\quad \sigma_x=\gamma w_{t-1},\ \sigma_y=\gamma h_{t-1}$$
  取$\gamma=0.5$。融合得分$\mathbf{w}_i = \mathcal{G}_t(u_i, v_i) \cdot s_i$，取Top-K个Token作为稀疏化输出。
- **原生稀疏预测头**：
  - *Score分支*：轻量MLP $g_s$ 预测标量置信度 $s_k$，目标Token由 $k^* = \arg\max_k s_k$ 选出。
  - *Regress分支*：轻量MLP $g_r$ 仅对$\mathbf{f}_{k^*}$执行一次回归，输出$\Delta_{k^*}=(\delta_x,\delta_y,w,h)$，结合Token原始网格坐标$\mathbf{p}_{k^*}$解码得最终边界框$\hat{\mathbf{b}}$。
- **训练损失**：
  $$L_{\mathrm{head}} = L_{\mathrm{cls}}(\{s_k\}) + \lambda_{\ell_1} L_{\ell_1}(\hat{\mathbf{b}}_{k^{\mathrm{gt}}}, \mathbf{b}) + \lambda_{\mathrm{GIoU}} L_{\mathrm{GIoU}}(\hat{\mathbf{b}}_{k^{\mathrm{gt}}}, \mathbf{b})$$
  其中$k^{\mathrm{gt}}$为距GT中心最近的保留Token，$\lambda_{\ell_1}=5,\ \lambda_{\mathrm{GIoU}}=2$。训练分两阶段：Stage 1训练
