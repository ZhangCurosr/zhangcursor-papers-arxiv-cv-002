---
title: "Mover360-Controllable-Object-Manipulation-in-360-sup-sup-Pan"
source: https://arxiv.org/pdf/2608.23238v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 19:28:09"
field: "全景图像编辑"
keywords: ["360° panorama", "equirectangular projection", "object manipulation", "diffusion transformer", "image editing", "controllable generation", "flow matching"]
innovations: ["以Translation为中心的ERPligned三通道指令图实现点/框/mask三级控制的统一物体操作框架", "在Flux2-Klein-4B-Base上通过4D RoPE坐标分配与VAE循环padding的最小侵入式LoRA适配方案", "面向全景物体操作的UE5配对数据生成管线与双域（合成+真实）测试基准"]
benchmarks: ["UE5 synthetic test set (210 tuples)", "Real captured test set (50 tuples)"]
---

# 论文速读：Mover360: Controllable Object Manipulation in 360° Panoramic Images

## 一句话总结
Mover360 是一个面向等距柱状投影（ERP）全景图像的可控物体级操作框架，以物体**平移（Translation）**为核心任务，同时支持移除（Remove）和参考引导插入（Insert），通过三通道 ERP 对齐指令图 + 辅助深度条件实现单点点击驱动的平移交互，在合成与真实双域基准上均显著优于现有视角图像编辑基线。

## 研究问题与动机
- **全景物体级操作尚未被充分探索**：现有图像编辑/插入方法针对透视图像设计，未显式建模 ERP 的水平周期性边界、纬度相关畸变及球面场景连续性。
- **直接应用视角编辑器的结构性缺陷**：将 GPT-Image-2 等最强通用编辑器直接作用于 ERP 时，会产生畸变失真、尺度不合理；先投影到透视视图再回投的方式会丢失视锥外的投射阴影，并在回投边界处留下外观不连续。
- **缺乏配对监督数据**：真实场景中很难获得同一场景编辑前后的配对观测，限制了可控物体操作的训练。
- **交互门槛高**：非专业用户难以预估物体在不同场景深度和纬度下的投影大小，要求精确绘制目标 mask 不切实际。

## 核心贡献（创新点）
1. **统一的多任务全景物体操作框架**：以 Translation 为中心、Remove/Insert 为辅助的单一模型，而非分任务独立训练，三者共享同一 backbone 与指令接口。与视角编辑器的本质区别在于原生 ERP 处理，避免了投影-回投的结构性缺陷。
2. **三通道 ERP 对齐指令图 + 点/框/mask 三级控制粒度**：将源区域（$C_s$）、目标区域（$C_t$）和目标点（$C_p$）编码为紧凑三通道图，默认单点点击即可驱动，模型自主推理合理尺度、支撑与光照。与现有方法要求已知尺寸目标 bbox/mask 的本质区别在于更低交互成本。
3. **面向全景物体操作的 UE5 配对数据生成管线**：表面感知物体放置（surface raycasting 避免穿透+法线对齐斜平面）、尺寸依赖运动路径、全局/局部随机光照、多视角参考图采集，生成 12,150 条配对序列（每 epoch 60,750 对）。这是目前面向平移/移除/插入三任务的唯一大规模全景配对数据源。
4. **最小侵入式 LoRA 适配方案**：在 Flux2-Klein-4B-Base 上仅训练 37M 个 LoRA 参数（rank=32），通过 4D RoPE 坐标分配区分文本、噪声潜量、多条件图像槽，并沿水平轴使用 VAЕ 循环 padding 保留左右连续性，无需修改 backbone 架构。

## 方法详解
- **骨干网络**：基于 Flux2-Klein-4B-Base（rectified-flow transformer），通过 LoRA（rank=32, alpha=32）适配，训练参数量 37,355,520，其余权重冻结。
- **输入条件**：输入 ERP 全景 $I$、固定任务 prompt $P$、三通道指令图 $C=[C_s, C_t, C_p]$、辅助 $\mathrm{DA}^2$ 预测深度图 $D$（仅 Insert 时额外加入参考图像 $R$）。
- **深度条件处理**：使用冻结的 $\mathrm{DA}^2$ 模型预测 ERP 深度图，经百分位数归一化（0.00–0.98 quantile）后映射到 [−1,1] 并复制三通道输入 VAE。训练中以概率 0.1 随机丢弃，增强鲁棒性。
- **指令图设计**：$C_s$ 编码源物体（mask 或 bbox），$C_t$ 编码目标区域（bbox），$C_p$ 为高斯热图（$\sigma=5$ px，置于目标接触点）。Translation 激活 $C_s+C_t/C_p$，Remove 仅激活 $C_s$，Insert 激活 $C_t/C_p$ 并启用参考分支。
- **多条件 Token 坐标分配**：继承 Flux2 原生 4D RoPE 格式 $(T, H, W, L)$，文本 token 使用序列轴 $(0,0,0,\ell)$，噪声潜量 token 使用空间轴 $(0,h,w,0)$，各条件图像槽通过不同 T 坐标 $(10k, h, w, 0)$ 区分，不修改 RoPE 公式本身。
- **全景感知 VAE 循环 Padding**：VAE 输入/输出沿水平方向各填充 64 像素后循环拼接，保持 ERP 左右边界连续性，Transformer 层面无需改动。
- **训练目标**：纯 Flow-Matching 损失 $\mathcal{L}_{\text{FM}} = \mathbb{E}[\|\nu_\theta(z_t, t, c) - \nu\|_2^2]$，无额外 RGB 重建/感知/seam/深度辅助损失。时间步服从 sigmoid-normal 分布。
- **训练配置**：1024×512 分辨率，AdamW，初始 LR $2\times10^{-4}$，CosineAnnealingLR，8×RTX Pro6000 GPU，per-GPU batch=2，8 epochs ≈30K iter，约 20 小时。推理 50 steps，CFG scale=3，单张全景约 20 秒。

## 实验与结果
- **数据集**：UE5 合成训练集 12,150 条序列（每 epoch 60,750 对：36,450 Translation + 12,150 Remove + 12,150 Insert）；双域测试集——210 个合成元组（48 种物体）+ 50 个真实捕捉元组（15 种物体），均提供三任务的 ground truth。
- **评估协议**：两套互补协议——（1）ERP 协议：基线直接作用于全景图；（2）Perspective 协议：自适应 FOV $[35°, 150°]$ 裁剪透视视图编辑后回投，严格有利于基线。
- **评估指标**：FAED（全景分布质量）、PSNR、SSIM、LPIPS、DINOv3 相似度、FID。FAED/PSNR 在全景计算，其余在透视视口计算。
- **主要结果**（ERP 协议）：
  - **Translation（UE5）**：Mover360 (bbox) FAED=0.314，最强基线 Insert-Anything FAED=0.344；FID=95.5 vs 108.9；point 版 FAED=0.336 已超越所有基线。
  - **Insert（UE5）**：Mover360 (bbox) 在 6 项中占 5 项最优；点版本在 FAED/SSIM/LPIPS/FID 均超越全部基线。
  - **Remove（UE5）**：Mover360 (mask) 在全部 6 项指标领先；Real 集同样领先 5/6 项（FAED 仅 LaMa 更好）。
  - **Perspective 协议下**：即使基线获得优势，Mover360 在 UE5 三任务全面领先，Real 集 Remove 六项全优、Translation 除 DINOv3 外全优、Insert 三项纹理质量指标领先。
- **关键数字**：Real Translation FAED 0.185 vs 次优 0.214（Insert-Anything）；Real Remove FID 92.2 vs LaMa 122.0。

## 相关工作脉络
1. **ObjectMover [36]**：研究合成游戏引擎监督下的真实物体位移，但未处理 ERP 周期性边界和纬度畸变；Mover360 直接在 ERP 上操作并提供统一的点/框/mask 空间控制。
2. **SE360 [46]**：最接近的全景编辑系统，支持多条件物体编辑，但仅有 Remove 模型开源，不支持平移与参考引导插入；本文覆盖完整三任务且全部开源。
3. **Omni² [34] / 360PanT [24]**：前者通过全局文本指令编辑全景，后者无对象级空间控制；本文提供像素级源/目标区域定位，Granularity 从点到 mask 三级可选。
4. **World-Shaper [13]**：ERP 几何感知全景编辑，同期工作且当时无公开实现；本文提供完整开源代码与配对基准。
5. **PanoDiffusion [28] / StitchDiffusion [23] / PanFusion [38]** 等全景生成方法：从文本或部分观测生成全景，不支持对已有全景中的指定物体进行可控操作。
6. **视角物体编辑基线（AnyDoor/InstructPix2Pix/DragAnything/LaMa/OmniPaint/Insert-Anything 等）**：针对透视图像设计，本文将其适配到全景场景验证其局限性，揭示投影方法的结构性缺陷（视锥外阴影丢失、边界不连续）。

## 局限性与未来方向
- **固定任务 prompt**：当前使用三个固定文本提示，尚不支持开放式语言控制（如修改物体属性、复杂关系约束）。
- **点引导的歧义性**：当目标点附近存在多个支撑面或期望物体尺度异常时，单点引导可能产生歧义。
- **深度估计弱点**：$\mathrm{DA}^2$ 在反射面、透明区域、无纹理区域或强畸变区域可能失效。
- **复杂光影交互**：大面积光源平移时会出现光源与照明明暗脱耦（光源被移动但大范围光照残留原位）；建筑级结构平移时会幻觉出不存在的额外结构。
- **合成-真实差距**：UE5 资产和渲染光照与真实全景仍存在分布差异，尽管双域基准和光照随机化已部分缓解。
- **未来方向**：扩展数据管线覆盖场景级结构平移和全局重光照场景；探索开放语言控制接口。

## 研究启发与可借鉴点
1. **点引导作为默认交互范式**：在空间控制任务中，用高斯点替代精确 mask 可大幅降低用户负担，模型可从全景上下文自动推理尺度与支撑关系——值得迁移到其他 3D/全景编辑场景。
2. **4D RoPE 坐标分配策略**：通过 T 轴区分条件图像槽、H/W 保留空间布局、L 轴用于文本，无需修改骨干结构即可融合多模态条件——可作为多条件扩散模型的条件融合通用方案。
3. **VAE 层循环 Padding 处理周期性**：沿水平轴 64px 循环填充即可保持 ERP 边界连续性，无需修改 Transformer 层——低成本高效的全景适配技巧。
4. **随机丢弃辅助深度条件训练**：以 0.1 概率丢弃 DA² 深度图，使模型在深度不可靠时仍保持鲁棒——可推广至其他辅助条件（如 normal map、语义图）的训练策略。
5. **双协议评估设计**：ERP 直接应用 + 严格有利的透视投影协议构成互补验证，既暴露视角编辑器的固有缺陷，又提供公平的对比基准——值得在其他跨域方法对比中借鉴。

## 关键术语表
- **Equirectangular Projection (ERP)**：将球面场景展开为矩形的全景图像投影方式，水平轴周期性（左右边界相邻），垂直方向存在纬度相关畸变。
- **Flow Matching**：一种扩散模型训练目标，学习从噪声到数据的确定性流场速度，取代传统去噪分数匹配。
- **LoRA (Low-Rank Adaptation)**：通过低秩矩阵注入冻结预训练模型注意力层，仅训练少量参数实现高效微调的技术。
- **4D RoPE ( Rotary Position Embedding)**：Flux2 使用的四维旋转位置编码 $(T,H,W,L)$，可区分文本/潜量/多条件图像槽的语义与空间位置。
- **FAED (Full-area Full-resolution E2E Difference)**：专为全景图像设计的评估指标，基于全景专属 autoencoder 测量生成图像与 ground truth 的全局分布质量。
- **Target Point Guidance**：用户仅需点击一个目标接触点，模型从全景上下文自动推理物体的投影大小、支撑关系和光照的轻量交互模式。

## 可复现要素
- **数据集**：UE5 合成训练集 12,150 条序列；双域测试基准（210 合成 + 50 真实元组）——论文声明代码和基准数据集已在 https://zhonghaoyi.github.io/Mover360/ 开源。
- **代码**：开源（论文声明）。
- **权重**：模型权重开源（论文声明）。
- **关键超参**：分辨率 1024×512；LoRA rank=32, alpha=32；初始学习率 $2\times10^{-4}$；CosineAnnealingLR；batch size=2/GPU，8 GPU；8 epochs ≈ 30K iter；推理 50 steps，CFG scale=3；高斯点 $\sigma=5$ px（训练分辨率）；深度丢弃概率 0.1；文本 prompt 丢弃概率 0.01；VAE 循环 padding 64 px。
