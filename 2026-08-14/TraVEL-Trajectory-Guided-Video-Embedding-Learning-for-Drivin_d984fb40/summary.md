---
title: "TraVEL-Trajectory-Guided-Video-Embedding-Learning-for-Drivin"
source: https://arxiv.org/pdf/2608.13495v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 04:44:58"
field: "自动驾驶数据检索与多模态表示学习"
keywords: ["driving-video retrieval", "multimodal embedding", "trajectory-guided fine-tuning", "GRPO", "nuReasoning", "motion-centric retrieval", "video-text embedding"]
innovations: ["用 ego 轨迹相似度作为 GRPO 奖励重塑视频嵌入空间，使嵌入对纵向/横向运动敏感", "双路径（t2v + v2v）分组策略解耦跨模态对齐与视频子空间运动结构学习", "仅用单向量余弦检索完成运动中心查询，推理时不引入任何额外感知模块"]
benchmarks: ["nuReasoning driving-video retrieval benchmark", "instance-level retrieval (R@K, MdR, MnR)", "motion-centric retrieval (mAP for longitudinal and lateral motion)"]
---

# 论文速读：TraVEL-Trajectory-Guided-Video-Embedding-Learning-for-Drivin

## 一句话总结
本文提出 **TraVEL**，通过两阶段训练（InfoNCE 监督微调 + 自车轨迹奖励 GRPO）将通用多模态嵌入模型适配至自动驾驶视频检索，使视频嵌入对纵向/横向运动更敏感，同时保持单向量余弦相似度检索的高效性。

## 研究问题与动机
1. **通用嵌入模型对运动敏感不足**：如 Qwen3-VL-Embedding 等通用视觉-语言模型在驾驶视频检索中存在显著域差距，容易依赖静态场景上下文捷径，难以区分方向相反的同类动作（如左转 vs 右转、加速 vs 减速）。
2. **基于规则的检索系统工程成本高**：传统结构化检索依赖专家定义规则和多阶段感知输出（检测、跟踪、地图等），扩展到新事件词汇或新数据集需要大量工程投入。
3. **纯文本标注信号对细粒度运动稀疏**：即使使用 nuReasoning 推理链描述做 InfoNCE 监督微调，相邻速度变化、轻微车道内偏移等连续运动信号的表征仍不够精细。
4. **零样本检索性能差距悬殊**：最大 8B 参数的 Cosmos-Embed1 和 Qwen3-VL-Embed 在 nuReasoning 上 R@1 仅分别为 2.4% 和 1.6%，说明模型规模无法自动弥合驾驶领域差距。

## 核心贡献（创新点）
1. **提出 TraVEL 框架，将通用多模态嵌入模型适配到驾驶视频检索**：两阶段训练——先用 NuReasoning 推理链描述做 InfoNCE SFT，再用自车轨迹相似度作为训练时奖励进行 GRPO 优化；检索时仍只用单向量，无需 ego poses 或辅助感知模块。
2. **引入轨迹派生的 GRPO 奖励机制，显式塑造视频嵌入的运动结构**：用横向位移和速度两个运动通道经 DTW 对齐计算相似度奖励，通过视频-视频组将连续物理信号注入嵌入空间，而文本-视频组保留二元实例对齐，避免二者冲突。
3. **构建并开源基于 nuReasoning 的驾驶视频检索基准**：从 20 秒源片段提取 8 秒窗口构成 1,715 视频池，提供实例级检索（R@K、MdR、MnR）与运动中心检索（mAP）两套评测。
4. **揭示运动中心查询的性能瓶颈并量化 SFT vs TraVEL 的增益差异**：在 2B/8B 两种规模下均报告纵向 mAP 和横向 mAP 的提升，表明轨迹信号对停止/转向行为的改善最为显著，而细微车道内偏移仍是难点。

## 方法详解
**两阶段训练流程：**

- **阶段一：Driving-Domain Supervised Fine-Tuning（SFT）**
  对批次 $\mathcal{B}$ 内每对 $(q_i, x_i)$ 计算 text-to-video InfoNCE 损失：
  $$\mathcal{L}_{\mathrm{SFT}}(\theta) = -\frac{1}{|\mathcal{B}|}\sum_{i\in\mathcal{B}}\log\frac{\exp(s_\theta(q_i,x_i)/\tau_{\mathrm{SFT}})}{\sum_{j\in\mathcal{B}}\exp(s_\theta(q_i,x_j)/\tau_{\mathrm{SFT}})}$$
  此阶段缩小视觉/语言域差距，输出 checkpoint $\theta_{\mathrm{SFT}}$ 初始化阶段二。

- **阶段二：Trajectory-Aware GRPO Fine-Tuning（TraVEL）**
  - **运动表示**：每段 8 秒 clip 采样 T=80 个位姿（10 Hz），提取横向位移（单位 m）和速度（单位 m/s）两个独立通道 $u_i^c$；应用对数压缩函数 $g_c(u)=\mathrm{sign}(u)\log(1+|u|/u_{0,c})$ 增强零附近变化敏感度。
  - **轨迹相似度奖励**：用 DTW 对齐两 clip 的运动信号，归一化后加权求和得 reward $r_{ij}^{\mathrm{mot}}=R\!\left(1-\frac{\sum_c w_c\widehat{d}_c(i,j)}{\sum_c w_c}\right)\in[0,R]$，本文 $w_{\mathrm{lat}}:w_{\mathrm{speed}}=1:2$，$R=10$。
  - **On-policy 检索分组**：每个优化步从候选池 $\mathcal{P}$ 当前模型打分中构造两组：
    - $\mathcal{G}^{\mathrm{t2v}}=\{i\}\cup\mathrm{Top}_{k-1}\{s_\theta(q_i,x_j)\}$，使用二元实例奖励 $r_{ij}^{\mathrm{t2v}}=\mathbf{1}[j=i]$ 保留跨模态对齐；
    - $\mathcal{G}^{\mathrm{v2v}}=\mathrm{Top}_{k}\{s_\theta(x_i,x_j)\}$，使用 graded 轨迹奖励 $r_{ij}^{\mathrm{v2v}}=r_{ij}^{\mathrm{mot}}$ 组织视频空间。
  - **GRPO 优化**：将组内得分转为分类策略 $\pi_\theta^m(j|a_i^m)=\frac{\exp(s_\theta/\tau)}{\sum\exp(s_\theta/\tau)}$，计算群体相对优势 $A_{ij}=(r_{ij}-\mu_i)/(\sigma_i+\epsilon)$，联合优化：
    $$\mathcal{L}_{\mathrm{GRPO}}(\theta)=\mathcal{L}^{\mathrm{t2v}}(\theta)+\lambda\,\mathcal{L}^{\mathrm{v2v}}(\theta),\quad\lambda=1$$
    超参：$\tau=0.05, \varepsilon=0.2, \beta=0$，全局候选池 768 个，每组 k=24。

**关键设计原则**：ego trajectory 仅在训练时作为 privileged supervision，检索时仅需单向量，推理开销不变。

## 实验与结果
- **数据集**：nuReasoning 可用子集 2,589 个 20 秒源 clip，划分 train/test；每源 clip 提取三个 8 秒窗口（锚点 5/10/15 秒），构成 1,715 候选视频池。
- **基线模型**：CLIP4Clip（151M）、InternVideo2（6B）、Cosmos-Embed1（1.2B）、Qwen3-VL-Embed（2B/8B）。
- **实例级检索（表 1）**：
  - 2B：R@1 从 1.3 → 8.7，R@10 从 7.4 → 35.3，MdR 从 266 → 23。
  - 8B：R@1 从 1.6 → 10.1，R@10 从 7.8 → 39.8，MdR 从 264 → 17。
- **运动中心检索 mAP（表 2）**：
  - 2B 纵向 mAP：Pretrained 24.7 → SFT 45.9 → TraVEL **55.7**（+9.8 over SFT）；横向 mAP：17.7 → 31.0 → **35.7**（+4.7）。
  - 8B 纵向 mAP：26.3 → 50.3 → **57.5**（+7.2）；横向 mAP：18.2 → 38.4 → **39.9**（+1.5）。
  - 最大提升在"Gently come to a stop"（2B: 14.7→33.7→**53.8**）和"Remain stopped"（2B: 10.6→70.8→**89.4**）。
- **最强结果**：8B TraVEL 在 R@1（10.1%）、R@10（39.8%）及纵向 mAP（57.5%）上均最优，相较零样本基线 R@1 提升约 6.3 倍。

## 相关工作脉络
1. **Qwen3-VL-Embedding / VLM2Vec**：通用多模态嵌入模型，本文将其作为 base；VLM2Vec 同样采用 InfoNCE 做视频-文本微调，但本文额外引入 GRPO 轨迹奖励阶段，填补其对运动感知的不足。
2. **CLIP4Clip / VideoCLIP**：经典视频检索方法；本文在 2,589 规模驾驶数据上的 R@1 达 8.7~10.1，超过 CLIP4Clip（2.6）但仍显著低于通用视频基准，强调域适配必要性。
3. **Cap4Video / Narrating the Video**：通过辅助生成 caption 增强检索；本文不依赖推理时 caption，而是在训练时用物理轨迹信号做 dense supervision，推理时无额外计算。
4. **X-CoT / VideoComp**：通过 chain-of-thought 或时序扰动改进细粒度理解；本文路线不同——不增加推理复杂度，用连续奖励重塑嵌入空间几何。
5. **RefAV / CARIM / STRIVE-D**：结构化规则+多阶段感知的驾驶检索；本文的核心差异在于检索时不需任何规则或感知输出，用单向量替代。
6. **VL-JEPA**：在潜空间直接预测语言嵌入以支持 text-to-video；本文保持标准 cosine-similarity 检索架构，仅改变训练目标。

## 局限性与未来方向
- **评测规模有限**：当前仅在 nuReasoning 可用子集（2,589 源 clip，1,715 候选池）上验证，未在大 pool 或多个数据集上测试泛化性。
- **细微横向运动仍困难**：Slightly move right/left in the lane 等车道的微小偏移 mAP 提升很小（2B: 4.6→4.7 和 8.1→9.6），单一 ego 轨迹不足以捕捉全部语义区分。
- **未考虑周围智能体运动**：reward 仅用 ego 位姿，未纳入周围车辆/行人轨迹，而驾驶场景中交互行为对事件分类至关重要。
- **未来方向**：扩展至更大池和更多数据集；将周围智能体运动和场景结构融入 reward；进一步探索 lane-level 精细运动的表征。

## 研究启发与可借鉴点
1. **物理信号作为 privileged training reward 的思路可迁移**：在机器人/自动驾驶领域，将传感器测量值（如 IMU、里程计）作为强化学习奖励，可在不增加推理开销的前提下显著改善 embedding 的物理一致性，适用于任何"训练时有额外监督信号、推理时不可用"的场景。
2. **双路径分组策略（t2v + v2v）值得借鉴**：文本-视频组保留精确跨模态对齐，视频-视频组用连续奖励塑形视频子空间，两者解耦避免了多对多关系对配对监督的干扰，该设计可用于多模态排序/检索任务。
3. **DTW + 对数压缩的组合对时序信号对齐有效**：对数压缩放大低速区变化（如从静止到加速），DTW 容忍时间偏移，这两项预处理技巧可直接复用于其他运动建模任务（如行为识别、时序检索）。
4. **用 GRPO 而非 PPO 降低训练复杂度**：GRPO 无需单独训练 value function，仅依赖群体内相对优势，这在 embedding 微调场景下尤为合适，可推广至其他无显式 value network 的多模态微调任务。

## 关键术语表
**TraVEL**：Trajectory-Guided Video Embedding Learning，本文提出的两阶段框架，通过 ego 轨迹相似度奖励塑造视频嵌入空间。
**GRPO（Group Relative Policy Optimization）**：群体相对策略优化，不需要单独价值函数的 RL 算法，用群体内 reward 的均值和方差标准化 advantage。
**InfoNCE**：对比学习中的交叉熵损失，最大化正样本对相似度的同时对负样本施加惩罚，本文用于 SFT 阶段。
**DTW（Dynamic Time Warping）**：动态时间规整，用于对齐两个时序信号的最优非均匀映射，本文用于计算两 clip 运动轨迹的相似度。
**nuReasoning**：长尾自动驾驶推理数据集，包含 20 秒驾驶 clip 及对应推理链描述，本文检索基准的来源。
**mAP（mean Average Precision）**：多个 motion query 的 AP 均值，本文用于评估运动中心检索性能的核心指标。
**Ego trajectory**：自车在 clip 时间窗口内的位姿序列，作为训练时的 privileged supervision，不包含在检索输入中。
**t2v / v2v group**：text-to-video 组（保留跨模态实例对齐）与 video-to-video 组（用轨迹奖励组织视频空间）的训练分组策略。

## 可复现要素
- **数据集**：nuReasoning（arXiv: 2605.31572），可用子集 2,589 个源 clip 公开可获取；1,715 候选视频池及 train/test 划分需按论文实现。
- **代码/权重**：论文未提及开源代码；预训练权重 Qwen3-VL-Embedding（2B/8B）可从 HuggingFace 获取；SFT/TraVEL checkpoint 论文未声明公开。
- **关键超参**：$\tau_{\mathrm{SFT}}$（温度，论文未具体给出数值）；$\tau=0.05, \varepsilon=0.2, \beta=0, \lambda=1$；T=80 采样点（10 Hz）；参考尺度 $u_{0,\mathrm{lat}}=1\,\mathrm{m}, u_{0,\mathrm{speed}}=1\,\mathrm{m/s}$；归一化分位数 95th，$D_{\mathrm{lat}}=1.264, D_{\mathrm{speed}}=2.342$；通道权重 1:2；$R=10$；候选池 768，每组 k=24。
