---
title: "RGB-D-Video-Generation-for-Improving-Human-to-Robot-Object-H"
source: https://arxiv.org/pdf/2608.13028v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 03:56:24"
field: "人机协作与机器人感知"
keywords: ["Human-robot collaboration", "RGB-D generation", "intention prediction", "Sim-to-real", "video diffusion"]
innovations: ["PassGen生成管道：融合SVD与时序面部编码器的RGB-D生成", "形态学深度噪声模拟：基于L515传感器空洞分布的边缘侵蚀策略", "Intention Gating：注视+速度的多模态意图门控机制"]
benchmarks: ["Hand2Bot-Real", "UR5e真实H2R交接"]
---

# 论文速读：RGB-D-Video-Generation-for-Improving-Human-to-Robot-Object-H

## 一句话总结
本文提出了PassGen生成管道与Hand2Bot数据集，通过融合稳定视频扩散模型与意图感知时序面部编码器，结合形态学深度噪声模拟，有效弥合了真实传感器噪声与仿真几何之间的sim-to-real差距，实现了基于全身上下文的更早期、更准确的H2R物体交接意图预测。

## 研究问题与动机
- 现有H2R交接数据集存在**噪声缺失**：主流合成数据集（如HandoverSim、GenH2R）缺乏真实传感器噪声（边缘散射、光照变化、运动模糊），导致模型在真实部署中性能骤降。
- **上下文局限**：现有基准数据集（如DexH2R）聚焦于手部特写视角，完全忽略了人脸表情、注视方向、头部朝向和上身姿态等全局人体特征，而这些特征是预知交接时机与方式的关键前置信号。
- **数据稀缺**：真实世界多模态数据采集成本高，需算法化方案合成高保真、交互式的RGB-D序列。
- **深度生成不足**：现有深度生成方法（如DepthCrafter）过于平滑，缺乏标准传感器特有的"空洞"模式。

## 核心贡献（创新点）
1. **PassGen生成管道**：基于Stable Video Diffusion的RGB-D视频生成框架，通过姿态引导与时序面部编码器保证手-物一致性；与现有仅生成RGB的方法本质区别在于同时生成含传感器噪声的深度流。
2. **意图感知时序面部编码器（TFE）**：将面部嵌入（ArcFace）建模为高频语义流，捕捉注视与微表情；区别于传统动画模型将头部视为刚性组件的做法。
3. **形态学深度噪声模拟**：基于Intel RealSense L515传感器记录的空洞分布，实现边缘侵蚀策略；不同于DepthCrafter等生成过于平滑的深度图。
4. **Intention Gating（IG）机制**：融合时序注视置信度与物体接近速度，实现多模态意图门控；与纯反应式抓取触发方式的本质差异。
5. **Hand2Bot数据集**：5,000个含真实L515噪声模式的RGB-D视频对（2,125真实+2,875生成），包含约20%负样本与时间戳标注的意图呈现期；是目前规模最大且专门针对H2R交接的全身视角数据集。

## 方法详解
### PassGen两阶段生成框架
- **阶段一：姿态引导的RGB视频生成**
  - 主干：Stable Video Diffusion（SVD）[27]
  - 外观编码器（Appearance Encoder）：提取参考图像$I_{ref}$的身份特征
  - 姿态编码器（PoseNet）：通过DWPose提取每帧骨架$V_{ref}$
  - TFE模块：利用ArcFace提取面部嵌入，经级联时序注意力（TA）与前馈网络（FFN）生成时序一致的面部token $\tilde{F}_t$
  - 交叉注意力注入：姿态-面部引导$\tilde{P}_t$融入U-Net：
    $$U_t^l \leftarrow U_t^l + \text{CrossAttention}(Q=U_t^l, K=\tilde{P}_t, V=\tilde{P}_t)$$
  - 微调：LoRA方案在Hand2Bot-Real数据集上微调

- **阶段二：形态学深度视频生成**
  - 深度初始化：DepthCrafter生成单应不变深度图$D_{init}$
  - 高度线性对齐：基于人体掩码替换绝对深度值
  - 形态学侵蚀：记录真实噪声分布$N_0$，根据像素高度$h$映射边界宽度向量$w(h)$进行侵蚀，生成含空洞的深度图$D_{final}$

### Intention Gating（IG）机制
- 多模态融合公式：
  $$S_{intent}(t) = \sigma\left(w_g \cdot f_{gaze}(\tilde{F}_t) + w_v \cdot v_{obj}(t)\right)$$
  其中$f_{gaze}$为TFE提取的注视置信度，$v_{obj}=-\frac{\partial D_{obj}}{\partial t}$为物体接近速度
- 单调非递减约束防止状态震荡：
  $$\hat{S}_{intent}(t) = \max(\hat{S}_{intent}(t-1), S_{intent}(t))$$
- 激活阈值$\tau=0.80$时平衡响应性与误触发抑制

### Grasp Pose标注
- 使用GraspNet生成6-DoF平行夹爪候选位姿池
- 垂直对齐过滤策略：最小化夹角$\theta = \arccos(g_z \cdot v_z)$
- 时间一致性滤波确保呈现期的位姿稳定性

## 实验与结果
### 数据集与设置
- **Hand2Bot数据集**：5,000个RGB-D视频对（2,125真实/2,875生成），7个室内场景，33种日常物体
- **评估平台**：UR5e机械臂 + Intel RealSense L515相机（1.5-2.2m距离）
- **训练硬件**：NVIDIA A6000

### 动画生成结果
| 指标 | PassGen | 次优StableAnimator | 提升幅度 |
|------|---------|-------------------|---------|
| PSNR↑ | **25.12** | 23.17 | +8.4% |
| SSIM↑ | **0.909** | 0.867 | +4.9% |
| LPIPS↓ | **0.212** | 0.218 | -2.8% |
| FID-VID↓ | **99.77** | 106.73 | -6.5% |
| FVD↓ | **337.59** | 372.81 | -9.5% |

### 消融实验
- **PassGen模块**：去除TFE后FVD从337.59升至399.52（-18%）
- **IG机制**：Full Module达90.0%准确率、13.6% FPR；Baseline无意图门控时FPR高达95.5%
- **数据增强**：Real+Syn训练较Real-only提升2.5%准确率、降低9.2% FPR、 unseen目标ISR从6/10提升至7/10

### 真实世界H2R交接
- **整体ISR**：54/60（90%）
- **整体FTR**：2/30（6.7%）vs 无IG的25/30（83.3%）
- **Seen对象**：Box/Cylinder/Plate达100%，Irregular降至80%
- **Unseen对象**：7/10成功

## 相关工作脉络
1. **HandoverSim [6] & GenH2R [7]**：合成数据集，无真实噪声，手-centric视角；本文提供含L515噪声的真实+生成混合数据集。
2. **DexH2R [8]**：真实世界但仅限手部视角；本文扩展至全身上下文包括注视与姿态。
3. **Animate Anyone 2 [19] & AnchorCrafter [20]**：通用人体/交互视频生成；本文针对H2R相机视角与手-物接触一致性优化。
4. **DepthCrafter [14]**：生成平滑深度序列；本文引入形态学侵蚀模拟真实传感器空洞。
5. **Mitty [39]**：近期提出的H2R视频生成；本文强调意图感知的多模态融合与下游控制验证。
6. **HOI4ABOT [12]**：人体-物体交互前瞻；本文专门针对交接任务与并行夹爪配置。

## 局限性与未来方向
- **生成局限**：依赖静态背景重建，环境多样性受限
- **触觉对齐缺失**：缺乏用户主观舒适度和自然度的定性评估
- **不规则物体挑战**：复杂表面几何导致夹爪物理抓取失败率较高
- **未来方向**：探索用户中心触觉对齐调查、主观自然度评估、更丰富的环境上下文建模

## 研究启发与可借鉴点
1. **多模态意图融合范式**：将注视方向（视觉注意力）与物体接近速度（空间运动）融合为统一意图分数，可迁移至其他社交机器人交互场景。
2. **形态学噪声注入策略**：基于真实传感器记录的空洞分布进行经验性边缘侵蚀，为RGB-D数据增强提供了低成本、高效率的sim-to-real桥梁。
3. **高频语义流建模**：TFE将面部区域建模为高频语义流而非刚性组件，启发了其他场景中微表情/视线的前置信号利用。
4. **负样本设计**：约20%负样本包含模糊手势、弱注视到达、非注视导向工具传递等，避免门控机制在人为有利的场景上过拟合。
5. **时序一致性门控**：单调非递减约束防止瞬时跟踪输出震荡，对实时机器人安全部署有直接参考价值。

## 关键术语表
- **H2R（Human-to-Robot）**：人类到机器人的物体交接任务，人机协作的基础能力
- **PassGen**：本文提出的两阶段生成管道，融合SVD与TFE生成手-物一致的RGB-D视频
- **TFE（Intention-Aware Temporal Face Encoder）**：意图感知时序面部编码器，捕捉面部微表情与注视方向
- **Intention Gating（IG）**：意图门控机制，融合注视置信度与物体接近速度实现主动抓取触发
- **Sim-to-real gap**：仿真到现实的差距，指仿真生成的理想几何与真实传感器噪声之间的差异
- **FPR（False Trigger Rate）**：误触发率，机器人错误响应非交接意图动作的比例
- **ISR（Intention Success Rate）**：意图成功率，机器人在呈现期内正确启动到达动作的比例
- **L515**：Intel RealSense L515深度相机，本文使用的真实深度传感器

## 可复现要素
- **数据集**：Hand2Bot，5,000个RGB-D视频对，论文声明包含真实采集与生成数据（论文未明确说明开源状态）
- **代码**：论文未明确声明开源，但方法基于开源组件（SVD、DepthCrafter、GraspNet、DWPose、ArcFace）
- **权重**：使用LoRA微调SVD，论文未声明公开微调权重
- **关键超参**：
  - IG阈值$\tau = 0.80$
  - 融合权重$w_g = 1.0, w_v = 2.5 [s/m]$
  - 相机距离1.5-2.2m，L515传感器
