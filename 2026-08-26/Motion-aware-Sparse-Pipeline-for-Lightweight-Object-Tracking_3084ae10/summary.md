---
title: "Motion-aware-Sparse-Pipeline-for-Lightweight-Object-Tracking"
source: https://arxiv.org/pdf/2608.24365v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 23:52:32"
field: "轻量级视觉目标追踪"
keywords: ["Lightweight Object Tracking", "Token Sparsification", "Motion Prior", "Vision Transformer", "Edge Deployment", "Sparse Prediction Head"]
innovations: ["在编码器第一层注入运动先验实现早期稳定 token 剪枝", "设计 score-first regress-once 原生稀疏预测头消除密集 reshape 瓶颈", "在边缘设备上实现速度-精度的帕累托最优，MaST-tiny 达 152 FPS 且精度创轻量 SOTA"]
benchmarks: ["LaSOT", "TrackingNet", "GOT-10k", "LaSOT_ext", "UAV123", "NFS", "VastTrack"]
---

# 论文速读：Motion-aware-Sparse-Pipeline-for-Lightweight-Object-Tracking

## 一句话总结
MaST 提出了一种端到端稀疏目标追踪框架，通过向 cross-attention 评分注入轻量级运动先验，实现早期稳定 token 剪枝，并设计了原生稀疏预测头（score-first, regress-once），消除密集 reshape 瓶颈，在边缘设备上达到速度-精度的帕累托最优。

## 研究问题与动机
- Transformer 类单流追踪器因二次复杂度的 self-attention 导致计算开销过高，难以部署在无人机、移动机器人等边缘设备。
- 现有 token 剪枝方法（如 OSTrack、AVTrack）大多将剪枝推迟到中间层（第 4/7/11 层），早期层仍需处理全部 token，计算节省有限；且剪枝依据（cross-attention 或辅助预测器）仅反映当前帧外观相关性，忽略了目标运动的时序连续性先验。
- 预测头仍是密集瓶颈：卷积头需要对稀疏 token 进行 padding 和 reshape 回完整 2D 特征图，浪费计算且可能导致目标中心 token 被误删。
- 实验诊断表明：若用 ground-truth 位置引导剪枝，即使在第一层剪枝精度几乎不变，说明早期剪枝并非不可行，而是缺乏可靠的空间引导信号。

## 核心贡献（创新点）
- **Motion-Aware Token Sparsification**：在高编码器第一层即执行一次性稀疏化，将前一帧预测 bbox 构造的 2D 高斯运动窗口与 cross-attention 重要性评分相融合，使早期剪枝既稳定又准确。区别于 OSTrack/FARTrack 仅依赖注意力图的浅层剪枝。
- **Natively Sparse Prediction Head（Score-First, Regress-once）**：不将稀疏 token 重排为密集 2D 网格，而是直接在保留的未结构化 token 上运行轻量 MLP，先选最高置信度 token 再单次回归 box，彻底消除 dense reshape 开销。区别于 LoRAT 等 MLP 头在密集假设下设计、未考虑高压缩比下的退化行为。
- **系统化边缘部署验证**：在三类 MACs 预算档位下均实现 Pareto 最优，MaST-tiny 在 Jetson Nano 上以 152 FPS 超越 AsymTrack-S（88 FPS），精度更高且速度快近 2 倍。

## 方法详解
- **整体流程**：输入模板图像 Z（128×128 或 192×192）和搜索图像 X（256×256 或 384×384），切分为非重叠 P×P patch 后做 patch embedding，送入 Transformer Encoder。在第一层 Encoder 后插入稀疏化模块，保留 top-K（K=30%）search token；后续所有 Encoder 块均在稀疏 token 上运行。最终由稀疏预测头解码目标框。
- **重要性评分**：简化 OSTrack 做法，只用模板中心 token k_c 作为 representative feature，search token q_i 的重要性分数为：s_i = softmax(q_i^T k_c / √d)。
- **运动先验注入**：根据上一帧预测 bbox b_{t-1}=(x,y,w,h) 构造 2D 高斯窗口：G_t(u,v)=exp(-(u-x)^2/(2σ_x^2)-(v-y)^2/(2σ_y^2))，其中 σ_x=0.5w，σ_y=0.5h。综合评分 w_i=G_t(u_i,v_i)·s_i，然后选取 top-K token。
- **稀疏预测头**：两个轻量 MLP 分支。Score 分支 g_s 对每个保留 token 预测置信度 s_k，argmax 选出最优 token k*；Regression 分支 g_r 仅对 f_{k*} 做一次预测 Δ_{k*}，再与 token 原始网格坐标 p_{k*} 结合解码最终 bbox。不使用 dense padding/reshape。
- **训练损失**：L_head = L_cls({s_k}) + λ_{l1} L_{l1}(b̂_{k_gt}, b) + λ_GIoU L_GIoU(b̂_{k_gt}, b)，其中 k_gt 为距 GT 中心最近的保留 token，λ_{l1}=5，λ_GIoU=2。

## 实验与结果
- **数据集**：LaSOT、TrackingNet、GOT-10k、LaSOT_ext、UAV123、NFS、VastTrack。
- **边缘设备评测**：Raspberry Pi 5、Apple M4、NVIDIA Jetson Orin Nano（ONNX Runtime 1.20.1）。
- **主要结果**：
  - LaSOT（~1G MACs 档）：MaST-tiny 获得 **63.8 AUC / 72.2 P_Normal**，超越 AsymTrack-S（62.8 AUC）+1.0；RPi5 上 22.6 FPS。
  - TrackingNet：MaST-tiny 获得 **80.1 SUC**，超越 AsymTrack-S（77.9）+2.2；Jetson Nano 上 **152 FPS**（AsymTrack-S 为 88 FPS）。
  - GOT-10k：MaST-tiny 获得 **66.6 AO / 62.5 SR_0.75**，超越 AsymTrack-S。
  - LaSOT_ext：MaST-small 获得最高 45.3 AUC，超越 FARTrack_tiny +0.3、AsymTrack-B +0.7。
  - UAV123：MaST-tiny 获得最高 66.6 AUC，RPi5 上 22.6 FPS。
  - NFS：MaST-tiny 获得 66.2 AUC（第二，仅次于 FARTrack_tiny 66.9）。
  - VastTrack：MaST-small 获得 35.6 SUC / 43.2 P_Normal，超越 MixFormerV2-B +0.4。
- **最强结果与提升**：MaST-tiny 在 Jetson Nano 上达 152 FPS，较 AsymTrack-S（88 FPS）提升约 73%，且 LaSOT AUC 和 TrackingNet SUC 均创轻量级追踪器新 SOTA。

## 相关工作脉络
- **OSTrack [54]**：提出利用 cross-attention 分数做 token 剪枝，但仅在中间层逐步剪枝，且无运动先验引导，早期层剪枝精度骤降。MaST 在此基础上通过运动先验实现第一层一次性剪枝。
- **FARTrack [44]**：利用 learnable location tokens 引导注意力，但仍是密集解码范式。MaST 则设计原生稀疏头，避免 dense reshape 开销。
- **AVTrack [32]**：训练辅助子网络预测 token 重要性实现 early exit，但带来变量每帧计算延迟。MaST 采用固定 top-K 保证可预测延迟。
- **LoRAT [34]**：用 MLP 替代卷积头，但未在高度稀疏 token 上评估。MaST 针对性设计了 score-first、regress-once 的稀疏 MLP 头。
- **SiamRPN++ [29] / OSTrack**：在输出端使用 Hanning/cosine 惩罚窗抑制边界响应，但仅作后处理加权，不影响编码器计算。MaST 将时序先验直接注入稀疏化决策。
- **EViT [33]**：最早提出 ViT token pruning，依赖 class token 注意力；MaST 将此思想适配到目标追踪，并结合运动先验解决追踪特有的时序一致性问题。

## 局限性与未来方向
- **高输入分辨率代价**：稀疏化生效前的早期 attention 计算仍随分辨率增大而增加，输入自适应稀疏可作为未来方向。
- **固定保留率**：当前 token retention rate 固定为 30% 以保证可预测延迟，无法根据场景难度动态调整（如高置信度时进一步稀疏）。
- **极端场景鲁棒性有限**：长时间遮挡或超大帧间位移时，局部搜索区域范式本身受限（非 MaST 独有，但稀疏化放大了这一问题）。
- **全局上下文缺失**：仅依赖局部搜索 crop，无法像两阶段或全图追踪方法那样恢复丢失目标。

## 研究启发与可借鉴点
- **运动先验驱动的稀疏选择**：将历史预测 bbox 构造高斯窗口作为空间先验，低成本且有效，可迁移到语义分割、点云处理等 ViT 稀疏化任务。
- **Score-First, Regress-once 头设计范式**：先筛选再回归的两步策略避免了密集 reshape，适合任何需要"从稀疏集合中选最优再回归"的下游任务。
- **诊断性 ablation 设计**：Fig.2(b) 用 ground-truth 位置引导剪枝的实验巧妙分离了"何时剪"和"剪什么"两个因素，这一思路可用于分析其他 sparse backbone 方法的瓶颈。
- **边缘设备公平评测协议**：统一导出 ONNX 并用相同运行时评测，对比 NAS 或压缩基线，为团队后续轻量化追踪工作提供参考范式。
- **多尺度 checkpoint 复用**：一个 dense checkpoint 通过微调即可适配 nano/tiny/small 三档变体，便于部署时根据算力动态选择。

## 关键术语表
- **Motion Prior（运动先验）**：基于上一帧预测 bbox 构造的 2D 高斯空间先验，用于引导当前帧 token 选择，捕捉目标运动的时序连续性。
- **Token Sparsification（Token 稀疏化）**：在 ViT 中仅保留 top-K 最具信息量的 patch token，丢弃其余以降低计算量。
- **Score-First, Regress-once**：先在稀疏 token 上预测置信度选出最优 token，再仅对该 token 执行一次边界框回归的策略。
- **Cross-Attention Importance Score（交叉注意力重要性评分）**：search token 与 template 中心 token 之间的 attention 权重，用于衡量 token 对目标的相关性。
- **Native Sparse Prediction Head（原生稀疏预测头）**：不经过 dense padding/reshape，直接在未结构化稀疏 token 上运行的预测模块。
- **Top-K Token Retention（Top-K Token 保留率）**：按综合评分选取比例最高的 K 个 token 保留，本文默认 30%。
- **GMACs（Giga Multiply-Accumulate Operations）**：衡量模型计算量的标准单位，1 GMAC = 10⁹ 次乘加运算。
- **P_Normal（归一化精度）**：LaSOT 等基准中以目标尺度归一化的精度指标，消除目标大小差异带来的偏差。

## 可复现要素
- **代码**：已开源，地址 github.com/TsingWei/MaST。
- **权重**：基于 MAE-lite 预训练初始化，论文提供预训练权重（见 GitHub）。
- **数据集**：COCO（训练）、LaSOT、GOT-10k、TrackingNet、VastTrack、LaSOT_ext、UAV123、NFS。
- **关键超参**：token retention rate = 30%；warmup 期 10 个 epoch 线性从 100% 降至目标率；λ_{l1}=5，λ_GIoU=2；backbone LR=4e-5，其他 LR=4e-4；weight decay=1e-4；batch size=128；300 epochs（Stage 1）+ 50 epochs（Stage 2 fine-tune）。
- **训练设备**：单卡 NVIDIA RTX 3090。
- **评测协议**：导出 ONNX OpSet 17，使用 ONNXRuntime 1.20.1 在 RPi5、Apple M4、Jetson Orin Nano 上评测。
