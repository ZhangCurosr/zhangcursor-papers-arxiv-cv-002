---
title: "STREAMTTT-RECONCILING-REAL-TIME-PERCEPTION-AND-LONG-TERM-MEM"
source: https://arxiv.org/pdf/2608.13416v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 04:01:20"
---

# 论文速读：STREAMTTT-RECONCILING-REAL-TIME-PERCEPTION-AND-LONG-TERM-MEM

## 一句话总结
StreamTTT 通过架构解耦将近期证据专注与长程历史压缩分配至不同通路，提出并行的滑窗注意力与测试时训练（TTT）快速权重记忆机制，使流式视觉语言模型在不稀释实时感知的同时有效恢复长程回忆，并在 OVO-Bench 与 StreamingBench 上取得当前 4B 规模下的最优平衡结果。

## 研究问题与动机
1. **感知-记忆固有权衡**：现有流式 VLM 面临实时感知与长程回忆的零和冲突；缩短上下文窗口可提升当前场景响应精度，但以牺牲远距离回溯为代价。
2. **历史注入的注意力稀释**：主流记忆方案（压缩、检索、Token 合并）最终仍将历史重注回注意力上下文，占用有限槽位并干扰对近期关键证据的关注（Lost-in-the-middle 效应）。
3. **训练监督信号偏差**：既有流式语料多聚焦“何时主动响应”的时机监督，缺乏针对证据初次可见时刻的实时感知训练信号。
4. **流式推理的恒定资源需求**：真实应用场景要求模型计算量与内存不随视频长度增长，传统全量上下文假设在此类设定下不可行。

## 核心贡献（创新点）
1. **感知-记忆双通路解耦架构**：将滑窗 KV 缓存（专注近期上下文）与并行 TTT 快速权重分支（存储跨窗口历史）分离，使长程记忆完全脱离注意力计算图。*本质区别在于历史不以 Token 形式争夺上下文槽位，而是以固定尺寸的参数状态隐式压缩，从根本上规避注意力稀释。*
2. **适配流式推理的 Large-Chunk TTT 机制**：基于 LaCT 的分块梯度聚合与 apply-then-update 顺序，配合输入依赖型门控动量衰减，使 TTT 状态可在多 forward pass 边界连续恢复。*区别于标准 TTT 的 token 级串行更新，该设计将序列化延迟降至最低，契合因果流式处理。*
3. **面向双能力的联合数据构建策略**：将 Streamo 主动问答的查询时间戳重定位至答案首次可见时刻（$t_q := t_a$），并融合动作、空间推理、前瞻与字幕任务构建 112.4K 实时 QA 语料，与 119K 离线长视频 QA 联合训练。*解决了既有流式数据偏重响应时机而忽视即时感知监督的结构性缺失。*
4. **匹配预算下的记忆机制公平对比**：在相同 4K 短期缓存与物理内存预算下，证明 learned fast-weight memory 显著优于启发式 KV 选择/检索存储（StreamMem、HERMES）。*表明在线学习压缩历史比静态检索策略更能保留关键长程信息，且不与短期感知产生资源竞争。*

## 方法详解
- **双分支混合层**：以 Qwen3-VL 为骨干，将每个自注意力层替换为并行结构。SWA 分支维护滑动 KV 缓存 $\mathcal{K}_t$（长度 $L$），TTT 分支维护固定尺寸循环状态 $S_t$。两路输出经可学习通道门控 $\alpha$ 融合：$\mathbf{O}_t = \mathbf{O}_t^{\mathrm{SWA}} + \tanh(\alpha) \odot \mathbf{O}_t^{\mathrm{TTT}}$，门控初始值接近零以保证训练初期模型优先依赖预训练注意力路径。
- **Fast-weight 读写与 LaCT 更新**：TTT 分支使用 gated MLP $f_{\mathbf{W}}(x) = \mathbf{W}^{(1)}(\mathrm{SiLU}(\mathbf{W}^{(0)}x) \odot (\mathbf{W}^{(2)}x))$ 生成 $(q_t, k_t, v_t)$。写操作最小化归一化重构损失 $\ell_t(\mathbf{W}) = \|\mathcal{N}(f_{\mathbf{W}}(k_t)) - \mathcal{N}(v_t - k_t)\|^2_2$，读操作为残差形式 $r_t = q_t + \mathcal{N}(f_{\mathbf{W}_t}(q_t))$。权重更新采用门控动量：$\mathbf{M}_t = \beta_t \mathbf{M}_{t-1} + \eta_t \mathbf{G}_t$，$\mathbf{W}_t = \
