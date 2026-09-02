---
title: "Source-Face-Authenticity-Detection-for-3D-Gaussian-Heads-Rec"
source: https://arxiv.org/pdf/2608.23984v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 06:49:07"
field: "3D人脸伪造检测"
keywords: ["3D Gaussian Splatting", "Face Forgery Detection", "AI-Generated Image Detection", "Self-Supervised Learning", "Multi-View Consistency"]
innovations: ["首个3D高斯头源脸真实性检测大规模基准与数据集", "两阶段检测器结合MAE细粒度保留与多视角对比学习", "多级CLS token融合策略提升分类性能"]
benchmarks: ["IID基准", "GPT-Image-2 OOD子集", "HyperSwapper OOD子集", "PersonaLive OOD子集", "Any3DAvatar OOD子集"]
---

# 论文速读：Source-Face-Authenticity-Detection-for-3D-Gaussian-Heads-Rec

## 一句话总结
本文提出了首个针对从单张肖像重建的3D高斯头（3D Gaussian Heads）进行源脸真实性检测的大规模基准与专用检测器，通过自监督预训练保留细粒度外观信息并结合多视角对比学习实现跨视角一致性，在多种伪造类型和重建方法上均取得最优性能。

## 研究问题与动机
- **核心问题**：如何判断一个从单张肖像重建的3D高斯头的源脸是真实人脸还是AI伪造人脸，以防范身份认证攻击与隐私泄露风险。
- **现有方法不足**：
  - 3D高斯重建和渲染会压缩或改变原始2D伪造线索，仅残留微弱的局部证据，现有检测器缺乏显式机制保留细粒度信息。
  - 视角变化会导致同一3D头的伪造证据在可见性、尺度和投影上发生差异，仅使用2D监督的检测器易过拟合到视角特定的渲染捷径，难以学习视角鲁棒的真实性特征。
  - 目前尚无针对3D高斯头源脸真实性检测的大规模数据集与系统化基准。

## 核心贡献（创新点）
- **首个大规模数据集**：构建了约36万张多视角渲染的3D高斯头图像（含1.6万+唯一身份），覆盖9种2D伪造方法与5种3D重建方法；与已有2D伪造检测基准的本质区别在于引入了3D重建环节，使伪造线索被部分弱化且视角可变。
- **两阶段检测框架**：Stage I 通过掩码自编码（MAE）保留细粒度细节 + 多视角对比学习（CL）实现跨视角特征一致；Stage II 冻结主干并融合低、中、高层CLS token进行分类；与直接微调2D检测器的本质区别是引入了自监督表征学习与多视图一致性约束。
- **系统性基准评估**：在平衡的身份不相交划分下，评估了7个代表性2D伪造检测器的性能，揭示其在3DGS领域的局限性；相比单一方法评估，提供了覆盖多伪造类别与多重建方法的统一评测协议。
- **多层次CLS token融合策略**：发现不同网络深度的CLS token具有互补的空间注意力模式，融合低-中-高层token可显著提升分类性能；与仅使用最终层token的做法相比，能更充分利用各层特征。

## 方法详解
- **数据集构建**：真实肖像来自5个数据集（CelebV-HQ, Ava-256, FFHQ, NeRSemble, VFHQ）；伪造肖像通过9种方法生成，包括5种文生图模型（Z-Image, Qwen-Image, GPT-Image-2, Nano Banana Pro, FLUX.2）、2种换脸方法（HyperSwapper, BlendFace）和2种表情驱动方法（LivePortrait, PersonaLive）；使用5种单图3D高斯头重建方法（CAP4D, FaceLift, LAM, GAGAvatar, Any3DAvatar）重建后，以512×512分辨率、yaw/pitch为±45°、相机半径0.8-1.2进行多视角渲染。
- **掩码自编码（MAE）**：将输入图像划分为P个非重叠patch，随机遮盖比例ρ=0.75的patch，仅将可见patch通过ViT主干$ f_{\theta} $得到token，再由8层ViT解码器$ g_{\phi} $恢复被遮挡区域的RGB像素，损失为$ \mathcal{L}_{\mathrm{MAE}} = \frac{1}{B\rho P}\sum_{i=1}^{B}\sum_{j\in\mathcal{M}_{i}}||\hat{\mathbf{p}}_{i,j}-\mathbf{p}_{i,j}||_2^2 $；此过程迫使骨干网络关注细粒度局部外观信息。
- **多视角对比学习**：对同一3D头的K个视角（同组）提取CLS token并进行$ \ell_2 $归一化，采用多正样本对比损失$ \mathcal{L}_{\mathrm{con}} = -\frac{1}{B}\sum_{a\in\mathcal{T}}\frac{1}{|\mathcal{P}(a)|}\sum_{p\in\mathcal{P}(a)}\log q(a,p) $，其中温度τ=0.1；目标是拉近同一3D头不同视角的表征、推远不同身份的表征。
- **训练策略**：Stage I共70个epoch，前30个epoch冻结主干仅优化MAE解码器，后40个epoch解冻主干联合优化，优化$ \mathcal{L}_{\mathrm{MAE}} + 0.1\mathcal{L}_{\mathrm{con}} $；Stage II冻结主干，训练MLP分类器10个epoch。
- **多级CLS token融合**：取ViT块的block 10、18、23（零索引）的CLS token，分别代表低、中、高层表示，归一化后拼接为$ \mathbf{h}_i=[\bar{\mathbf{c}}_i^{(\ell_{\mathrm{low}})};\bar{\mathbf{c}}_i^{(\ell_{\mathrm{mid}})};\bar{\mathbf{c}}_i^{(\ell_{\mathrm{high}})}] $，经MLP分类头输出预测。

## 实验与结果
- **数据集规模**：361,469张候选渲染图，经身份不相交划分后得到147,900张（训练117,900 / 验证15,000 / 测试15,000），覆盖13,981个唯一身份，各类别均衡。
- **评估指标**：Accuracy (Acc), Real-Accuracy (R.Acc), Fake-Accuracy (F.Acc), Macro-F1, AUC, AP(fake)。
- **主要结果**：本文方法在IID基准上取得Acc=89.51%、R.Acc=90.84%、F.Acc=88.19%、AUC=95.36%、AP(fake)=96.03%，在所有分类指标和重建方法上均排名第一；最强基线Effort的Acc为85.17%，提升约4.3个百分点。
- **OOD泛化能力**：在4个未参与训练的伪造/重建方法子集（GPT-Image-2、PersonaLive、HyperSwapper、Any3DAvatar）上均取得最优结果；如在GPT-Image-2子集上Acc=96.54%，较最强基线Effort（91.87%）提升4.67个百分点。
- **消融实验**：移除MAE模块导致Acc下降至86.96%；移除对比学习导致Acc下降至87.18%；不使用Stage I预训练则Acc降至85.79%；多级CLS token融合（89.51%）显著优于单级分类器（最高85.64%）。

## 相关工作脉络
- **3D高斯头重建**：FaceLift (Lyu et al. 2025)、CAP4D (Taubner et al. 2025)、LAM (He et al. 2025)、GAGAvatar (Chu and Harada 2024)、FastAvatar (Liang et al. 2025)、Any3DAvatar (Gao et al. 2026)；本文与之区别在于关注重建头的源脸真实性而非重建质量本身。
- **Fake3DGS (Di Nucci et al. 2026)**：检测3DGS表示本身的篡改，与本文检测源脸真实性的任务定位不同，不可直接应用。
- **2D伪造检测-面部专用**：IID (Huang et al. 2023)、ForAda (Cui et al. 2025)、NPR (Tan et al. 2024a)；本文将其作为基线，证明其直接迁移到3DGS域存在局限。
- **2D伪造检测-通用AI生成图像**：AIDE (Yan et al. 2025a)、Effort (Yan et al. 2025b)、PGC (Zhou et al. 2026)、FreqNet (Tan et al. 2024b)；同样作为基线评估，揭示其在3DGS域的不足。
- **参数化/隐式3D头表示**：3DMM (Blanz and Vetter 2023)、FLAME (Li et al. 2017)、Tri-plane (An et al. 2023)、NeRF (Gafni et al. 2021)；本文聚焦更近期且保真度更高的3DGS表示。

## 局限性与未来方向
- **表情驱动伪造检测仍有挑战**：该类别准确率仅76.24%，显著低于文生图（98.28%）和换脸（90.04%），说明当前方法对这类伪造线索的捕获仍不充分。
- **训练数据规模受限于完整身份组件的保留**：为确保身份不相交，最终数据集（147,900张）远小于候选池（361,469张），可能存在数据利用不充分的问题。
- **未来方向**：扩充表情驱动伪造类别的数据覆盖、探索更细粒度的伪造线索建模、以及将方法推广至动态视频序列的3D高斯头。

## 研究启发与可借鉴点
- **自监督MAE用于伪造线索保留**：在3D重建后伪造痕迹被弱化的场景下，利用掩码自编码强制网络学习细粒度局部外观是一种有效的表征学习策略，可迁移到其他经过几何变换或渲染的伪造检测任务。
- **多视角对比学习增强视角鲁棒性**：对于输入可能从任意视角采样的3D表示检测任务，引入多视角对比损失可有效抑制视角特定捷径，该思想可推广至任意多视角3D感知任务的预训练。
- **多层CLS token融合策略**：发现ViT不同深度的CLS token具有互补的空间注意力模式并融合使用，是一种无需额外模块即可提升分类性能的轻量技巧，可在多个视觉分类任务中验证。
- **身份不相交划分确保公平评估**：构建"连接分量-全组分至同一划分"的严格划分策略，避免了身份泄漏导致的评估虚高，对任何涉及人脸/生物特征的数据集构建具有参考价值。

## 关键术语表
- **3D Gaussian Splatting (3DGS)**：一种基于3D高斯椭球显式表示的神经渲染方法，支持高效实时渲染，已被广泛应用于单人头像重建。
- **Source-Face Authenticity**：指3D高斯头所源自的原始2D肖像图像的真实性（真实人脸或AI伪造人脸）。
- **Masked Autoencoding (MAE)**：随机遮盖图像patch并通过自编码器重建的自监督预训练策略，用于学习细节保留的表征。
- **Multi-View Contrastive Learning**：利用同一3D对象的多视角图像作为正样本对进行对比学习，以学习视角不变的特征表示。
- **CLS Token**：Vision Transformer中用于全局图像表示的特殊token，本文在不同网络深度处提取并融合。
- **OOD (Out-of-Distribution)**：指测试集包含训练时未见过的伪造方法或重建方法，用于评估模型的泛化能力。
- **Identity-Disjoint Split**：按唯一身份ID划分训练/验证/测试集，确保不同划分间无共享人物，避免数据泄漏。

## 可复现要素
- **数据集**：论文构建了约361,469张候选数据集，最终基准包含147,900张；数据集来源包括CelebV-HQ、Ava-256、FFHQ、NeRSemble、VFHQ等公开数据集及9种开源/公开伪造方法。**论文未明确声明数据集是否公开**。
- **代码/权重**：**论文未明确声明代码与预训练权重是否开源**。
- **关键超参**：遮蔽比例ρ=0.75，温度τ=0.1，Stage I训练70 epoch（前30 epoch冻结主干），Stage II训练10 epoch；主干为DINOv3 ViT-H+/16，融合block 10、18、23的CLS token；输入分辨率224×224；学习率MAE解码器1×10⁻⁴、主干1×10⁻⁵、MLP 1×10⁻⁴；随机种子42。
