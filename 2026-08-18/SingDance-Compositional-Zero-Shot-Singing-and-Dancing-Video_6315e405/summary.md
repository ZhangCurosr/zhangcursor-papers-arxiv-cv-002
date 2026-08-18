---
title: "SingDance-Compositional-Zero-Shot-Singing-and-Dancing-Video"
source: https://arxiv.org/pdf/2608.16220v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 06:17:59"
---

# 论文速读：SingDance-Compositional-Zero-Shot-Singing-and-Dancing-Video

## 一句话总结
本文提出 SingDance，一个统一的视频扩散框架，通过将可控发声 articulation 建模为显式语义角色（Source/Listener），实现了仅凭参考图、文本与音频生成“边唱边跳”视频的组合零样本能力；推理时仅切换角色状态即可在保持音乐对齐舞蹈的同时可靠地打开或关闭唇同步。

## 研究问题与动机
- **现有方法的能力割裂**：音乐驱动舞蹈生成聚焦编舞与节拍对齐，通常不建模跳舞者的唇部发音；语音驱动模型则默认输入语音由可见主体产生，二者均无法处理同一主体既唱歌又跳舞的组合设定。
- **缺乏统一的行为组织范式**：创作者希望基于同一参考图与曲目生成纯舞蹈与唱跳两种视频，但现有系统往往需要独立模型或后处理唇同步模块，难以在同一框架内无缝切换。
- **角色语义未被显式建模**：简单将 `Lip-Sync OFF` 视为“冻结嘴唇”会丧失真实聆听/反应行为；需要更语义化的角色区分来组织说话、聆听、纯舞蹈与唱跳四种行为。
- **组合零样本的可行性未知**：如何在从未见过成对唱跳数据的情况下，组合已学习的发声能力与音乐驱动舞蹈能力，仍是一个开放问题。

## 核心贡献（创新点）
1. **提出显式发声角色建模**：将可控发声定义为语义角色（Source 为音频源，Listener 为接收者），使唇同步成为角色分配的自然结果而非孤立口型指令；与现有语音驱动方法的区别在于解耦了“音频内容”与“音频归属主体”。
2. **设计角色感知的条件组合机制**：Source 与 Listener 共享同一语音表示与注入路径，通过 hard-compact routing 与 frame-wise joint audio injection 实现任务相关 token 的灵活拼接；与交互式 avatar 双流设计不同，本文保持单一路径共享，仅靠紧凑文本+学习条件切换行为。
3. **非对称分阶段训练达成组合零样本泛化**：仅使用语音对数据与纯舞蹈数据分阶段训练，Song/Source 配置在训练中完全不可见，推理时通过角色状态切换实现零样本唱跳生成；本质区别在于不依赖成对唱跳数据，而是利用条件组合性。
4. **引入文本-音频三 Pass CFG 与紧凑干预范式**：通过独立文本/音频引导尺度与条件 dropout 策略，在推理时仅修改 prompt 字段与 learned role condition 即可完成配对角色切换；相比修改模型结构或增加微调步骤，干预成本极低且行为一致性强。

## 方法详解
- **音频表示提取**：使用 Wav2Vec 2.0 提取语音特征，经可学习权重聚合层表示与非因果 1D 时序编码器，得到全局语音特征与帧对齐局部语音 token；使用 MuQ 提取音乐特征，经独立时序编码器映射为帧对齐局部音乐 token，两者均匹配预训练特征的非因果上下文。
- **发声角色控制**：提示词结构化分为五个字段（vocal role, style, action, first frame, camera），vocal role 显式指定 source（Lip-Sync ON）或 listener（Lip-Sync OFF），action 字段仅保留身体运动描述以避免文本语义干扰。同时配备全局 role embedding 与局部 role token，与文本字段保持一致更新。
- **Hard-Compact Routing**：路由由任务类型硬决定（非内容推断），仅保留任务相关 token 子集（见原表 1）：Speech/Source → $S + R_{src}$；Speech/Listener → $S + R_{lis}$；Instrumental/Listener → $R_{lis} + M$；Song/Listener → $S + R_{lis} + M$；Song/Source 仅在推理激活。歌曲类纯舞蹈仍保留语音 token，因歌词与乐句边界可能辅助编舞 timing。
- **Frame-wise Joint Audio Injection**：路由 token 与生成的每个 latent frame 时间对齐，作为 cross-attention 的 key-value；视觉 token 作为 query，cross-attention 输出残差加回更新视觉表示，参考图 latent 保持不被修改。
- **分阶段训练策略**：
  - Stage 1（语音阶段）：仅训练 source/listener 角色控制，音乐路径缺失，利用屏上说话与真实聆听/反应视频建立角色语义。
  - Stage 2（舞蹈阶段）：从 Stage 1 权重初始化，引入音乐路径，使用纯音乐与歌曲纯舞蹈视频训练；同时保留 20% 语音样本防止角色控制遗忘。全程使用标准 flow-matching 损失，无专用唇同步或节拍对齐 loss。
- **Text-Audio Classifier-Free Guidance**：推理时使用三 Pass CFG，预测公式为 $\widehat{\mathbf{v}} = \mathbf{v}_{\bar{t},0} + s_{t}(\mathbf{v}_{t,0} - \mathbf{v}_{\bar{t},0}) + s_{a}(\mathbf{v}_{t,a} - \mathbf{v}_{t,0})$。训练 dropout：文本 10%、音频 10%、双 dropout 5%，learned role embeddings 在所有 dropout 情况下均保留。

## 实验与结果
- **数据集**：语音阶段使用 HuMoSet + 私有网页视频（约 30 万条 5 秒片段，其中约 1.2 万为真实聆听/反应）；舞蹈阶段使用清洗后 MA-Data + 私有舞蹈视频（约 4.8 万条，含 1.8 万纯音乐、3 万歌曲，通过 SyncNet 过滤掉含唇同步样本）。测试集为作者自建的 SingDance-50、Dance-100 以及公开 EMTD。
- **基线**：MusicInfuser（TA2V，不支持参考图）、Wan-S2V（TIA2V，语音驱动 baseline）用于舞蹈评估；Hallo3、HY-Avatar、InfiniteTalk 用于唇同步对比。
- **主要定量结果**：
  - **Singing-and-Dancing（SingDance-50）**：SingDance 在 BeatAlign 达 0.2723、MBCR 达 0.2558，参考保真度 Subject 0.9682 / Background 0.9772 均优于 Wan-S2V；人类偏好 Rhythm 61.33%、Visual 54.67%。
  - **Dancing-Only（Dance-100）**：BeatAlign 0.3002、MBCR 0.2632，Subject 0.9705、Background 0.9766，人类偏好 Rhythm 68.00%、Visual 55.67%。
  - **配对角色切换（Table 3）**：Source 角色 LSE-C=4.9672、LSE-D=8.9363；Listener 角色 LSE-C=1.1768、LSE-D=12.4059，切换时舞蹈动作与音乐对齐保持稳定，而 Wan-S2V 仅改 prompt 几乎无行为变化。
  - **唇同步对比（EMTD，Table 4）**：SingDance 仅需 5.63B 生成时参数量，LSE-C=7.995，优于 InfiniteTalk（18.88B, 8.535）与 HY-Avatar（7.210），且美学/成像质量最高。
- **消融（Table 5）**：移除路由音乐 token 导致 BeatAlign 与 MBCR 显著下降（SingDance-50 从 0.2723→0.2462，MBCR 从 0.2558→0.2386），视觉质量基本稳定，验证音乐路径对节拍对齐的核心作用。
- **结论**：组合零样本唱跳生成可行，音乐对齐舞蹈与可控唇同步均达到竞争力水平，且推理参数效率显著。

## 相关工作脉络
- **Music-conditioned dance generation**（MusicInfuser, OmniDance, Wan-Dancer, X-Dancer）：聚焦编舞与节拍对齐，未探索可见主体是否发声的组合轴；本文在此基础上新增角色条件轴，实现同一框架下的行为切换。
- **
