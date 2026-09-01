---
title: "Unwarping-the-Lens-A-Physics-Grounded-Approach-to-Video-Glas"
source: https://arxiv.org/pdf/2608.20212v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 16:36:25"
---

# 论文速读：Unwarping-the-Lens-A-Physics-Grounded-Approach-to-Video-Glas

## 一句话总结
本文提出一种基于物理光学的视频眼镜去除框架，将大规模生成先验（Nano Banana）的多视角图像知识经严格筛选与折射/反射增强后，迁移至确定性前馈网络 JFSnet，在像素空间实现高保真、身份与姿态一致的眼镜去除，推理速度达 27.68 FPS，显著优于现有扩散与 GAN 基线。

## 研究问题与动机
- 现有眼镜去除方法高度依赖随机生成先验（扩散模型/GAN），在强折射畸变与视点相关镜面反射下缺乏显式几何约束，易引发“身份漂移（identity drift）”与表情/姿态失真。
- 逐帧独立应用生成模型会导致视频时序闪烁；引入光流或多帧传播的计算成本过高，难以满足实时需求。
- 潜空间生成模型存在压缩瓶颈，易丢失皮肤纹理、眼型轮廓等高频细节，且通用视频修复方法（如 ProPainter）缺乏针对眼部光学畸变的特化先验。
- 纯生成式数据缺乏物理一致性，难以支撑高精度逆折射学习任务。

## 核心贡献（创新点）
- 提出“生成先验→确定性前馈模型”的离线迁移框架，利用物理光学仿真（Snell 定律折射+HDR反射）对生成数据进行训练时增强，使网络显式学习逆折射变换而非盲目重绘。
- 设计 JFSnet（Joint Feature-Spatial Network），采用 DINOv2 ViT 编码器捕获语义身份先验，结合 ResNet 卷积解码器与全局跳跃连接，在像素空间直接重建高频面部细节，突破潜空间瓶颈。
- 引入平移等变性约束（translation equivariance loss）作为轻量级时序一致性正则项，无需光流或跨帧内存即可有效抑制视频闪烁。
- 构建包含 1,860 个合成身份（24,180 对图像）的高质量多姿态/多表情配对数据集，并通过三阶段硬采样过滤（背景/眼部 $L_1$ 过滤、凝视/身份漂移检测、U-Net 重建阈值）保障几何对齐精度。
- 在 FFHQ 子集上取得 SOTA 指标（FID 0.379，瞳孔位移 2.249 px，地标 L2 误差 0.632 px），同时实现 27.68 FPS 实时推理，感知评测中五个维度均获最高偏好率。

## 方法详解
- **数据生成与过滤**：使用 Nano Banana 为每个合成身份生成 13 组无眼镜/戴眼镜图像（含多姿态与表情），利用 ARAP 形变网格与多尺度光度损失对齐。三阶段过滤：① 计算背景与眼部 $L_1$ 误差剔除姿态/凝视偏移样本；② 局部结构一致性分析过滤身份漂移；③ 训练轻量 U-Net 作为质量代理，丢弃高重建误差样本。
- **物理光学增强**：训练时动态合成折射与反射。折射模型通过 3DMM 拟合头部位姿 $(R, t)$，结合单目深度估计与标准成人头宽假设锚定度量尺度，利用 Vogel 规则由屈光度 $D$ 计算双球面曲率半径（$n\approx1.5$），沿视线逐像素射线追迹应用 Snell 定律生成密集形变场 $\mathcal{W}$；反射模型基于逆视角环境贴图模拟 HDR 高光与 AR 镀膜变化。合成输入公式：$I = (1-M_{lens}) \cdot G_{glasses} + M_{lens} \cdot (\mathcal{W}(G_{clean}) + \mathcal{R}) + \epsilon$。
- **JFSnet 架构**：预训练 DINOv2 (ViT-L/14) 编码器（仅解冻最后 4 层）提取语义特征，经 1×1 卷积适配通道后与 ResNet 卷积解码器多级融合；引入输入图像的直连跳跃通道保留未遮挡区域的高频纹理。
- **训练目标**：总损失 $\mathcal{L} = \lambda_{pix}\mathcal{L}_{pix} + \lambda_{perc}\mathcal{L}_{perc} + \lambda_{eye-pix}\mathcal{L}_{eye-pix} + \lambda_{eye-perc}\mathcal{L}_{eye-perc} + \lambda_{adv}\mathcal{L}_{adv} + \lambda_{temp}\mathcal{L}_{temp}$。眼部局部 $L_1$ 与 Gram 感知损失聚焦 64×64 镜片中心块恢复；时序损失强制网络与空间平移交换：$\mathcal{L}_{temp} = ||f_\theta(\mathcal{T}_{\Delta s}(I)) -
