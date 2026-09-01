---
title: "Photorealistic-Novel-View-Synthesis-of-Human-Faces-using-Nex"
source: https://arxiv.org/pdf/2608.23410v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 23:50:38"
field: "三维视觉与生成模型"
keywords: ["novel view synthesis", "human face reconstruction", "next-scale transformer", "autoregressive modeling", "3D Gaussian splatting", "multi-view consistency"]
innovations: ["将 next-scale 自回归 Transformer 扩展至多输入多输出人脸新视角合成，单次前向传播实现跨视角注意力", "提出三阶段渐进训练策略（通用高分辨率适配→人脸微调→规范视角专门化），在小规模真实数据集（2.9K）上实现 512×512 高保真生成", "证实 next-scale 自回归范式在低数据 regimes 下优于扩散模型基线，推理速度提升 3.3 倍且无需大规模 2D 预训练"]
benchmarks: ["PSGS", "Ava-256"]
---

# 论文速读：Photorealistic-Novel-View-Synthesis-of-Human-Faces-using-Nex

## 一句话总结
本文提出了一种基于 next-scale 自回归 Transformer（VAR）架构的人脸高保真新视角合成方法，通过三阶段训练策略在较小但更真实的数据集（PSGS，约 2.9K 主体）上实现 512×512 高分辨率下的多视角同时生成，无需大规模 2D 预训练，推理速度约 1.5 秒（6 个规范视角），在 PSGS 和 Ava-256 数据集上均优于 FaceLift、Splatter Image 等基线方法。

## 研究问题与动机
- **高分辨率多视角一致性难题**：现有方法在高空间分辨率和广角 viewpoints 下难以同时保持身份特征、外观细节和几何连贯性；近 frontal 流水线（如 Splatter Image、VoluMe）在非 frontal 视角质量急剧下降，而大规模预训练图像到 3D 系统（如 FaceLift）依赖大量合成数据且渲染效果偏人工化。
- **人类面部 3D 数据集稀缺**：与通用 3D 物体不同，高质量人类面部 3D 扫描数据集规模有限（PSGS 仅约 3.2K 主体），无法支撑从头训练的 next-scale 模型，也不适合直接微调已有的高分辨率通用图像生成模型。
- **扩散模型对数据与预训练的强依赖**：当前主流方法（如 FaceLift）依赖 Stable Diffusion 的大规模 2D 预训练和海量合成人脸数据，导致泛化性和真实性受限；而 next-scale 自回归范式可利用低分辨率通用预训练（如 Objaverse 训练的 ArchonView）逐步适配高分辨率目标域，降低对目的特定数据的需求。
- **多视角联合生成的效率瓶颈**：自回归模型传统上单步单视图生成，难以像扩散模型那样在多步迭代中自然实现多视角注意力；本文旨在通过 next-scale 架构在一次前向传播中实现多输出视角的跨视角注意力，提升一致性的同时加速推理。

## 核心贡献（创新点）
1. **将 next-scale 自回归 Transformer 扩展至多输入多输出人脸新视角合成**：在 ArchonView 基础上支持多个松散姿态输入和多个规范输出视角的同时生成，通过每个 scale 内跨视角注意力机制在一次前向传播中捕获多视角关联。
2. **提出三阶段渐进式训练策略以克服数据稀缺与分辨率提升的双重挑战**：第一阶段在大规模通用物体数据集（SS3D）上将 ArchonView 从 256×256 适配至 512×512（保持 scale 数量不变、增大每 scale 尺寸）；第二阶段在 PSGS 人脸数据上微调模型与 VQ-VAE；第三阶段专门化为 6 个规范视角输出以适配下游 3D 重建。
3. **证实 next-scale 自回归范式在低数据 regimes 下优于扩散模型基线**：在 PSGS 和 Ava-256 上，本文方法在 PSNR、SSIM、LPIPS、DreamSim 等指标上全面超越 FaceLift（含使用相同 PSGS 数据重训练的变体）和 Splatter Image，且推理速度提升约 3.3 倍（1.5 秒 vs 5 秒）。
4. **与 MV2Splat GS-LRM 无缝集成实现端到端人脸 3D 高斯重建**：将生成的 6 个规范视角输入像素对齐的 3D Gaussian 提升模块，获得高保真人脸 3D 模型；附录还展示该方法无需架构修改即可扩展至人体重建。

## 方法详解
- **整体两阶段管线**：第一阶段为 novel view synthesis（NVS），从 1 或 3 个输入图像生成 6 个规范视角（0°, 45°, 90°, 180°, 270°, 315°）；第二阶段使用现成的 MV2Splat（GS-LRM）将 6 个视角提升为像素对齐的 3D Gaussian 球面表示，无需输入相机位姿。
- **多尺度 VQ-VAE 编码/解码**：输入图像 $I \in \mathbb{R}^{h \times w \times 3}$ 经冻结的 VQ-VAE 编码器转换为 K 个尺度的离散潜在特征图 $\{Z_1, Z_2, \ldots, Z_K\}$，每 scale $Z_k \in \mathbb{R}^{h_k \times w_k \times C}$ 为量化后的 token 网格，codebook 大小 $V=4096$，跨所有 scale 共享；512×512 分辨率下 K=10，scale 尺寸为 (1, 2, 3, 4, 6, 9, 13, 18, 24, 32)。
- **Next-scale Transformer 多视角扩展**：传统 VAR 按 scale 顺序自回归生成（公式 1），本文扩展为同时生成 M 个输出视角（公式 2）：
  $$g(X, R, T) = \prod_{k=1}^{K} p_\theta(r_{k,1}, \ldots, r_{k,M} | r_{<k,1}, \ldots, r_{<k,M}, X, R, T)$$
  每个 scale 内所有输出视角的 token 相互注意，同时可访问之前所有 scale 的全部视角信息，实现跨视角一致性建模。
- **全局与局部条件编码**：全局条件采用 posed CLIP embedding（公式 3），将输入图像的 CLIP 特征与目标相机球面坐标 $(\theta, \phi, r)$ 拼接后经两层线性映射得到 SOS token 和 AdaLN 条件；多输出时每视角使用独立 SOS token（基于各自 $(R_m, T_m)$），AdaLN 共享来自首视角的全局条件。局部条件为各输入图像的 multi-scale latent features，多输入时沿 channel 维拼接。
- **三阶段训练策略**：
  - **Stage 1**（通用 NVS 高分辨率适配）：在 SS3D（约 2M 纹理对象，24 个随机相机视角，512×512）上训练，保持原有 scale 数量不变、增大每 scale 分辨率，验证 Tab. 4 表明直接增加 scale 数会导致收敛困难。
  - **Stage 2**（人脸 specialization）：在 PSGS（约 2.9K 训练主体）上微调，训练单输入（ frontal ）或双输入（ frontal + 两侧 ±90° 松散姿态）模型，同时微调 VQ-VAE（30 epochs，8 A100）。
  - **Stage 3**（规范视角专门化）：进一步将模型约束为输出 6 个固定规范视角以适配 GS-LRM；实验发现 classifier-free guidance（CFG）在此阶段有害（Fig. 7 右），故关闭 CFG。
- **损失函数**：论文未显式列出损失函数，但 VQ-VAE 训练采用 perceptual + discriminative loss 复合损失；Transformer 训练采用标准 next-token prediction loss（交叉熵），输出层为线性分类头预测 codebook entry logits。

## 实验与结果
- **数据集**：
  - **SS3D**（私有）：约 2M 纹理对象，24 个球面随机相机，512×512 渲染。
  - **PSGS**（私有）：3.2K 主体（288 验证）， dome 捕获校准多视角图像 + 高保真 Gaussian avatar；32 个 turntable 视角用于评估，32 种随机相机/头部姿态扰动用于多输入训练。
  - **Ava-256**（公开）：256 主体，每主体 80 个高分辨率 dome 视角；使用 FaceLift 定义的 10 主体、11 视角子集。
- **评估指标**：PSNR、SSIM、LPIPS（VGG backbone）、DreamSim、ArcFace（身份相似度；未检测到人脸记 0，均未检测到记 1）。
- **基线**：
  - **SI (PSGS)**：Splatter Image [41] 在 PSGS 上两阶段训练（256×256，受 VRAM 限制）。
  - **FaceLift-NVS**：Stable Diffusion V2-1-unCLIP 为基础的 6 视角生成（原始 + 用 PSGS 重训变体 FaceLift-NVS (PSGS)）。
  - **FaceLift / VoluMe**：完整 image-to-3D 管线。
- **主要定量结果**（PSGS 验证集，6 规范视角，Tab. 1）：
  | 方法 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | DS ↓ | ArcFace ↑ |
  |---|---|---|---|---|---|
  | SI (PSGS) II | 20.16 | 0.8614 | 0.2351 | 0.02297 | 0.7403 |
  | FaceLift-NVS (PSGS) | 16.75 | 0.7900 | 0.3206 | 0.02262 | 0.6680 |
  | **Ours MI-MO** | **23.13** | **0.8816** | **0.1762** | **0.01379** | **0.7206** |
  - Ours MI-MO 相对 SI (PSGS) II 提升：PSNR +2.97 dB，SSIM +0.0202，LPIPS -0.0589，DreamSim -0.00918；ArcFace 略低但综合考虑视觉质量更优。
  - 相对 FaceLift-NVS (PSGS) 提升：PSNR +6.38 dB，SSIM +0.0916，LPIPS -0.1444，DreamSim -0.00883。
- **完整 image-to-3D 管线结果**（PSGS，32 turntable 视角，Tab. 2）：
  - **Ours MI-MO + MV2Splat**：PSNR 21.91，SSIM 0.8865，LPIPS 0.2110，DreamSim 0.01188，ArcFace 0.7737，全面最优。
- **Ava-256 结果**（Tab. 3）：Ours MI-MO + [11] 取得 PSNR 15.67、SSIM 0.8209，在相同数据条件下优于 FaceLift (PSGS)（PSNR 14.51）。
- **推理速度**：本文方法生成 6 个视角约 1.5 秒，FaceLift 约 5 秒；Splatter Image 最快（实时）但仅适用于近 frontal 视角。
- **消融实验**（Tab. 4）：
  - 原始 256×256 ArchonView 在人脸任务上 PSNR 仅 12.02，提升分辨率至关重要。
  - 从头训练（Scratch）或从通用图像生成模型 VAR-d36 微调均无法收敛至合理性能（PSNR 16.72~19.68）。
  - 保持 scale 数量不变、仅增大每 scale 尺寸（d12, 512）优于增加 scale 数（K=12），前者 PSNR 21.43 vs 后者 15.70。
  - 更大模型（d30）需第一阶段预训练才能收敛（PSNR 21.66 vs 无 Stage 1 的 20.34）。
  - VQ-VAE 微调带来小幅提升（PSNR 22.24 → 22.24，ArcFace 67.72）。
  - Classifier-free guidance 在第三阶段有害（Fig. 7 右）。

## 相关工作脉络
- **ArchonView [48]**：本文直接扩展的基础架构，原面向通用物体 NVS（256×256，单输入单输出），本文将其适配至 512×512 高分辨率并支持多输入多输出人脸场景。
- **FaceLift [24]**：当前最强单图人脸 3D 重建基线，采用两阶段 diffusion + GS-LRM 管线，依赖大规模合成训练数据和 Stable Diffusion 预训练；本文方法在相同 PSGS 真实数据上显著超越，且推理速度快 3 倍以上。
- **Splatter Image [41] / VoluMe [20]**：实时单图 3D Gaussian 重建方法，擅长近 frontal 视角但非 frontal 视角质量骤降；本文方法在所有视角均保持高保真，弥补了其视角范围局限。
- **VAR [43]**：next-scale 自回归图像生成基础模型，本文取其 NVS 变体 ArchonView 作为起点，而非直接使用通用图像生成版本（消融证实后者无法保持身份一致性）。
- **PanoHead [1] / Rodin [46] / RodinHD [49]**：早期及近期人脸/头像生成方法，在视角一致性或真实性上逊于本文；本文方法在真实数据小样本设定下仍能达到 SOTA。
- **MV2Splat [11] / GS-LRM [50]**：像素对齐 3D Gaussian 提升模块，本文将其与 NVS 阶段解耦组合，形成端到端 image-to-3D 管线；该模块化设计使 NVS 部分可独立替换优化。

## 局限性与未来方向
- **训练数据表情与配饰覆盖不足**：PSGS 仅含中性表情和无配饰面部，模型难以泛化至多样表情（如大笑、皱眉）或配饰（帽子、眼镜），与 FaceLift 报告的限制一致。
- **背景颜色 bleed 问题**：所有方法（包括本文）在多视角生成中出现背景颜色渗入前景（如黑色背景渗入门毛），亮色背景加剧该问题；需更精细的分割或遮罩技术缓解。
- **数据规模限制**：尽管三阶段策略降低了数据需求，但 2.9K 训练主体仍属中小规模，限制了对极端视角、复杂光照和多样化身份的覆盖。
- **未来方向**：
  - 探索更多样化的训练数据（表情、配饰、人种多样性）以提升泛化能力。
  - 结合近期 next-scale VAR 改进工作（如 Infinity [13]、InfinityStar [22]）进一步增强生成质量。
  - 无需架构修改即扩展至人体全身重建（附录已验证可行性）。
  - 引入更细粒度的背景去除与前景掩码机制以抑制 color bleeding。

## 研究启发与可借鉴点
- **三阶段渐进适配策略可有效缓解小数据高分辨率训练难题**：从低分辨率通用预训练出发，先适配分辨率（Stage 1），再微调目标任务（Stage 2），最后专门化输出结构（Stage 3），该策略可迁移至其他需要高分辨率但目标域数据稀缺的生成任务。
- **保持 scale 数量不变、仅增大每 scale 分辨率比增加 scale 数更利于收敛**：消融实验（Tab. 4）表明，对于已学习固定 scale 数量的 next-scale 模型，直接扩展空间尺寸而非增加 scale 层级能保留已有表征学习能力，这一经验对类似架构的分辨率升级具有指导价值。
- **多视角跨注意力在一次前向传播中实现一致性**：与扩散模型多步迭代中累积多视角信息不同，next-scale 架构在每个 scale 内对所有输出视角进行 joint attention，推理速度快且一致性内置，该设计思路可推广至其他多视角生成任务（如人体、物体）。
- **VQ-VAE 微调以小代价换取感知质量提升**：在目标数据集上仅 30 epochs 的微调即可改善编码/解码质量（附录 Fig. A.4），这一低成本优化手段值得在其他 latent diffusion 或 VQ-based 生成管道中尝试。
- **模块化 NVS + GS-LRM 解耦设计提升灵活性与可复现性**：将视角合成与 3D 重建分离，允许独立替换或改进任一阶段；后续研究可直接复用 MV2Splat 等现成 GS-LRM，聚焦于 NVS 部分的优化。

## 关键术语表
- **Next-scale Autoregressive Transformer (VAR)**：将传统自回归模型的 token 生成顺序从 raster-scan 改为按图像 scale 顺序，每次预测整个分辨率层级的 token 网格，提升生成效率与质量。
- **VQ-VAE (Vector Quantized Variational Autoencoder)**：通过离散 codebook 将连续 latent 特征量化为 token 的自编码器，支持多尺度 latent 表示，是 VAR 架构的编码基础。
- **ArchonView**：基于 VAR 的零样本单图物体新视角合成模型，本文以其 256×256 预训练权重为起点进行分辨率适配与人脸 specialization。
- **PSGS (Capture, Canonicalize, Splat)**：Meta 发布的私有真人面部多视角捕获数据集，含 3.2K 主体的高保真 Gaussian avatar，用于本文训练与评估。
- **GS-LRM (Gaussian Large Reconstruction Model)**：将多视角图像提升为像素对齐 3D Gaussian 球面表示的大规模重建模型，本文使用 MV2Splat 变体作为第二阶段。
- **Classifier-Free Guidance (CFG)**：扩散模型中通过条件/无条件预测的差值增强生成质量的推理技术；本文发现在第三阶段训练中使用 CFG 反而有害。
- **Canonical Views**：固定角度的规范视角集合（0°, 45°, 90°, 180°, 270°, 315°），作为 NVS 阶段的标准输出以适配下游 3D 重建管线。
- **DreamSim (DS)**：基于人类感知判断微调的中层特征图像距离度量，比 LPIPS 更符合人类对图像真实性的主观评价。

## 可复现要素
- **数据集**：
  - SS3D：私有（约 2M 对象），论文未公开。
  - PSGS：私有（3.2K 主体），论文未公开；Ava-256：公开（https://github.com/nvlabs/codec-avatar-studio）。
- **代码/权重**：论文未声明开源代码或模型权重；ArchonView [48] 和 VAR [43] 有开源实现可作起点。
- **关键超参**：
  - Transformer 深度：24 blocks（约 1B 参数）。
  - VQ-VAE codebook 大小：$V = 4096$。
  - 512×512 分辨率下 scale 数量：$K = 10$，尺寸序列 (1, 2, 3, 4, 6, 9, 13, 18, 24, 32)。
  - VQ-VAE 微调：30 epochs，8 A100 GPU。
  - Stage 1 训练：50 小时，64 A100 GPU。
  - Stage 2 训练：单输入 18 小时 / 多输入 14 小时（128 A100 GPU）。
  - Stage 3 训练：2 小时（同硬件）。
  - 不使用 classifier-free guidance（CFG=0）。
