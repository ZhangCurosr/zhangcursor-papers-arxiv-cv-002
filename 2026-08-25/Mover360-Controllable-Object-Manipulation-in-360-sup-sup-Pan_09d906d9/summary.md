---
title: "Mover360-Controllable-Object-Manipulation-in-360-sup-sup-Pan"
source: https://arxiv.org/pdf/2608.23238v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 19:28:07"
field: "全景图像编辑与对象操纵"
keywords: ["360-degree panorama", "equirectangular projection", "object manipulation", "diffusion transformer", "controllable editing", "flow matching"]
innovations: ["Translation为中心的ERP全景对象操纵统一框架（点/bbox/mask三粒度指令图）", "基于预训练Flux2-Klein的轻量化LoRA适配+4D RoPE多条件坐标分配", "UE5表面感知数据生成管道与双域（合成+真实）配对基准"]
benchmarks: ["UE5 synthetic test set (210 tuples)", "Real captured test set (50 tuples)"]
---

# 论文速读：Mover360: Controllable Object Manipulation in 360° Panoramic Images

## 一句话总结
本文提出 Mover360，一个针对 360° 等距柱状投影（ERP）全景图像的可控对象操纵框架，以**对象平移（Translation）**为核心任务，辅助支持移除（Remove）和参考引导插入（Insert）；通过三层 ERP 对齐指令图与辅助深度条件，在轻量化 LoRA 微调的预训练扩散 Transformer 上实现全局一致的对象级编辑。

## 研究问题与动机
1. **360° 全景编辑的特殊性**：ERP 图像具有水平环绕周期性、纬度依赖性畸变和全局场景连续性，现有透视图像编辑器直接应用于 ERP 会产生边界伪影、畸变的对象几何、不稳定的缩放和 inaccurate 的空间控制。
2. **透视投影方案的结构性缺陷**：将编辑区域投影到透视视口、编辑后再重投影回全景的方案，无法处理超出视锥的影响（如长投射阴影），且编辑模型仅感知视口而非整个球面，无法合成视外内容驱动的视角相关外观。
3. **缺乏成对监督数据**：真实场景中极少有同一场景在受控对象编辑前后的观测数据，导致对象级全景编辑缺乏有效的训练监督。
4. **现有全景编辑方法的局限性**：已发布的 panorama 编辑系统要么不提供对象级空间控制，要么仅通过全局文本指令编辑，要么未公开权重，无法与本工作的可控对象级编辑设置直接对比。

## 核心贡献（创新点）
1. **成对全景编辑数据集与双域基准**：构建了 UE5 数据生成管道（含表面感知对象放置、随机光照、多视角参考捕获），产出 12,150 个成对相机-对象序列（每 epoch 60,750 训练对），以及包含 210 个合成元和 50 个真实元的双域测试基准，覆盖 Translation/Remove/Insert 三个任务的地面真值。
2. **Translation 为中心的统一编辑交互设计**：提出三通道 ERP 对齐指令图，统一点、bbox 和 mask 三种控制粒度，默认单点点击即可完成对象重定位，模型自动推断合理的尺寸、支撑和光照；Translation 为主任务，Remove/Insert 作为辅助任务由同一模型支撑。
3. **最小侵入式多条件适配方案**：基于预训练 rectified-flow Transformer（Flux2-Klein-4B-Base）的 LoRA 微调，通过 4D RoPE 坐标分配区分文本 token、噪声目标全景 token 和多个条件图像 slot，结合 VAE 水平的圆形填充保持 ERP 左右连续性，无需修改骨干网络架构即可将透视图像生成器转化为统一的全景对象编辑器。
4. **双协议评估体系**：提出 ERP 直接应用协议和严格有利的透视重投影协议两个互补的评估协议，系统性地揭示了视角编辑器的结构性局限，并验证 Mover360 在两种协议下均取得最强结果。

## 方法详解
**整体架构**：Mover360 以 Flux2-Klein-4B-Base 为骨干，在 VAE 隐空间运行，输入 ERP 全景后首先用冻结的 DA² 模型估计辅助深度图，然后将输入全景、指令图、归一化 ERP 深度、可选参考图像和训练目标经 VAE 编码器编码，条件隐变量拼接并 patchify 为 token 序列输入扩散 Transformer。

**任务形式化**：编辑预测公式为 $\hat{I} = f_\theta(I, P, C, D, R)$，其中 $I$ 为输入 ERP 全景，$P$ 为固定任务提示，$C$ 为 ERP 对齐指令图（三通道 $C = [C_s, C_t, C_p]$，分别编码源区域、目标区域和目标点），$D$ 为 DA² 估计的辅助深度图，$R$ 为仅 Insert 任务使用的可选参考图像。

**指令图设计**：源区域通道 $C_s$ 用于 Translation 和 Remove 定位被操纵对象；目标区域通道 $C_t$ 支持 bbox/mask 形式目标区域；目标点通道 $C_p$ 为以期望接触位置为中心的各向同性高斯热图（σ=5 pixels）。默认点引导模式只需用户点击一个接触点，模型即可从全景上下文和辅助深度推断合理尺寸。

**辅助深度条件**：使用 DA² 估计 ERP 深度图，训练中随机以 0.1 概率 drop；深度图经分位数归一化（仅对有限正值计算 0% 和 98% 分位数作为边界）后映射至 VAE 输入范围 [-1, 1] 并复制三通道。

**多条件 Token 坐标分配**：沿用 Flux2 原生 4D RoPE，对 text token 使用序列轴 $(0,0,0,\ell)$，对噪声全景 latent token 使用空间轴 $(0,h,w,0)$，对条件图像 token 使用 $(10k,h,w,0)$ 其中 $k$ 为条件图像 slot 序号，从而在不修改 RoPE 公式和 Transformer 架构的前提下区分不同模态和 slot。

**VAE 圆形填充**：输入全景在 VAE 编码前沿水平轴每侧填充 64 像素的循环填充，解码后裁剪回原始宽度，保持 ERP 水平周期性。

**LoRA 适配**：在 Transformer 的 attention QKV 层和投影层插入 rank=32、alpha=32、dropout=0 的 LoRA adapter，可训练参数共 37,355,520 个。

**训练目标**：采用 flow-matching 损失 $\mathcal{L}_{\text{FM}} = \mathbb{E}[\|\nu_\theta(z_t, t, c) - \nu\|_2^2]$，其中 $t = \sigma(u), u \sim \mathcal{N}(0,1)$ 采用 sigmoid-normal 时间分布；无额外的 RGB 重建、感知、接缝或深度辅助损失。

## 实验与结果
**数据集**：UE5 训练集 12,150 个相机-对象序列（608 个独特对象，60 条预设运动路径，三相机/路径，10 帧轨迹），双域测试集：210 个 UE5 合成元（48 个对象）+ 50 个真实拍摄元（15 个对象），均为 Translation/Remove/Insert 提供配对 GT。

**评估基线**：
- Translation：DragAnything, AnyDoor, Paint-by-Example, OmniPaint, Insert-Anything
- Insert：Paint-by-Example, AnyDoor, OmniPaint, Insert-Anything
- Remove：LaMa, OmniPaint, SE360（仅 Release 了 Remove 模型）
- 评估协议：ERP 直接应用协议（Table 2）和严格有利的透视重投影协议（Table 3）

**主要结果（ERP 协议，UE5 测试集）**：
- Translation（bbox）：FAED 0.314↓, PSNR 25.08↑, SSIM 0.836↑, LPIPS 0.255↓, DINOv3 0.747↑, FID 95.5↓ — 六项指标全部最优
- Insert（bbox）：FAED 0.369↓, PSNR 24.52↑, SSIM 0.837↑, LPIPS 0.259↓, FID 101.5↓（DINOv3 0.736 略低于 Insert-Anything 的 0.754）
- Remove（mask）：FAED 0.262↓, PSNR 27.66↑, SSIM 0.923↑, LPIPS 0.141↓, DINOv3 0.783↑, FID 77.4↓ — 六项全部最优

**主要结果（Real 测试集，ERP 协议）**：
- Translation（bbox）：FAED 0.356↓, PSNR 26.85↑, SSIM 0.823↑, LPIPS 0.249↓, FID 131.0↓（DINOv3 0.704 低于 Insert-Anything 的 0.727）
- Remove（mask）：FAED 0.372↓, PSNR 29.99↑, SSIM 0.887↑, LPIPS 0.141↓, DINOv3 0.793↑, FID 92.2↓ — 六项全部最优

**透视协议结果**：即使基线受益于消除 ERP 畸变的有利条件，Mover360 在 UE5 上对全部三项任务仍取得 FAED/PSNR/SSIM/LPIPS/FID 全面领先；在 Real 测试集上 Remove 全指标领先，Translation 除 DINOv3 外全部领先。

**消融实验**：
- 辅助深度：去除 DA² 深度条件后所有任务指标均有下降（如 Real Translation DINOv3 从 0.669 降至 0.655）；对 Remove 的 SSIM 提升最显著（0.918 vs 0.846）
- 指令粒度：Translation 和 Insert 上 point→bbox→mask 单调提升；Remove 上 bbox 和 mask 相近，mask 在 SSIM/LPIPS/FID 略优
- 点引导轻量变体（Mover360 point）已与最强基线竞争

## 相关工作脉络
1. **ObjectMover**：研究合成游戏引擎监督下的真实感对象位移，与本工作 Translation 任务最相关，但未显式处理 ERP 周期性、纬度畸变和球面场景连续性。
2. **SE360**：构建分层编辑对用于全景中多条件对象编辑，与本工作最为接近；但其仅公开了 Remove 模型权重，Translation 和参考引导 Insert 未支持/未发布，无法作为完整基线。
3. **World-Shaper**：研究 ERP 域几何感知全景编辑（generate-then-edit 监督），与本工作同期但当时无公开实现。
4. **Perspective editors（AnyDoor, Paint-by-Example, OmniPaint, Insert-Anything, LaMa, DragAnything）**：均为透视图像编辑器，直接应用于 ERP 会暴露于训练分布外畸变；本文通过双协议评估揭示了其结构性缺陷。
5. **Panorama generation methods（PanoDiffusion, StitchDiffusion, PanFusion, DiffPano, SMGD）**：从文本或部分观测合成全景，不提供对已有对象的可控操纵接口。
6. **360PanT / Omni²**：前者做无训练文本驱动全景翻译，后者通过全局文本指令编辑，均缺乏对象级空间引导界面。

## 局限性与未来方向
1. **固定任务提示缺乏开放语言控制**：当前使用三条固定提示，无法支持改变对象属性或指定复杂关系约束的开放语言编辑。
2. **点引导的歧义性**：当目标点附近存在多个支撑面或期望对象尺寸异常时，单点引导可能产生歧义。
3. **深度估计的失败模式**：DA² 在反射、透明、无纹理或强畸变区域可能估计失败，影响编辑质量。
4. **复杂光照与阴影处理能力有限**：对于大面积发光体（如大型灯具），模型仅重定位发射体而未能重新合成其广泛投射的光照区域，导致光源与场景光照解耦；对于建筑尺度结构，可能幻觉目标区域不存在的额外结构。
5. **合成-真实域差距**：尽管随机光照和真实评估有助于缩小差距，但 UE5 资产和渲染光照仍可能与真实全景存在差异。
6. **未来方向**：扩展训练数据以覆盖场景级结构和全局重新光照编辑；探索开放语言控制和更细粒度的关系约束。

## 研究启发与可借鉴点
1. **双协议评估设计**：ERP 直接应用协议与严格有利的透视重投影协议相结合，既揭示了基线方法的结构性缺陷（视锥限制、上下文缺失），又公平地比较了方法性能，为全景相关任务的评估提供了可借鉴范式。
2. **4D RoPE 多条件坐标分配**：在不修改 Transformer 架构的前提下，通过原生 4D 位置编码的语义分配（text/noise/condition slots 分别用不同轴）实现多模态条件融合，是一种轻量且通用的多条件扩散模型适配策略。
3. **UE5 表面感知数据生成管道**：按对象大小分类的运动路径、表面射线投射放置、随机光照（全局+局部灯具）、多视角参考捕获等设计，为三维场景相关编辑任务的合成数据构建提供了系统性参考。
4. **点引导作为默认轻量交互**：利用模型从全景上下文和深度条件推断对象尺寸和支撑关系，降低用户交互负担，同时保留 bbox/mask 作为强控制变体——这种分层交互设计在空间编辑任务中具有迁移价值。
5. **辅助深度的随机 Drop 增强鲁棒性**：以 0.1 概率随机 drop 辅助深度条件，使模型在深度不可靠或缺失时仍能工作，是一种简单有效的条件鲁棒性增强 technique。

## 关键术语表
**Equirectangular Projection (ERP)**：将球面场景映射到二维矩形的投影方式，水平轴具有周期性的环绕特性，垂直方向存在纬度依赖的畸变。
**Flow Matching**：一种扩散模型训练目标，直接学习从噪声到数据的常微分方程速度场，替代传统扩散模型的噪声预测。
**DA² (Depth Anything in Any Direction)**：用于从单张图像估计深度的预训练模型，本文用作辅助几何条件。
**LoRA (Low-Rank Adaptation)**：通过低秩分解在预训练模型注意力层注入可训练参数的轻量化微调方法。
**FAED (Full-frame AutoEncoder-based Distortion)**：基于全景专用自编码器的全景级分布质量评估指标。
**RoPE (Rotary Position Embedding)**：旋转位置编码，Transformer 中用于编码 token 位置信息的机制，本文适配为 4D 形式以区分多条件输入。
**Object Translation**：将全景中指定对象从源位置移动到目标位置的任务，需同时完成源区域背景修复和目标区域的合理放置（含尺寸、支撑、光照一致性）。

## 可复现要素
- **数据集**：UE5 合成训练集（12,150 序列，60,750 对/epoch）和双域测试基准（210 合成 + 50 真实元）；论文声明代码和基准数据集已在 https://zhonghaoyi.github.io/Mover360/ 开源。
- **代码/权重**：论文声明代码和基准数据集可用（available at the project page）；基于 Flux2-Klein-4B-Base 的 LoRA 权重应随代码开源。
- **关键超参**：训练分辨率 1024×512，AdamW 初始学习率 2×10⁻⁴，CosineAnnealingLR 调度，8×RTX Pro6000 GPU，每 GPU batch size=2，训练 8 epoch（约 30K 迭代），LoRA rank=32/alpha=32/dropout=0，可训练参数 37,355,520；推理使用 50 sampling steps，text CFG scale=3。
- **其他**：辅助深度以 0.1 概率随机 drop；任务提示以 0.01 概率 drop 用于 classifier-free guidance；推理时 point/bbox/mask 变体使用相同权重，仅指令不同。
