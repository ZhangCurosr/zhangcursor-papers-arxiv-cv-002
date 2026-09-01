---
title: "NGS-MARKER-ROBUST-NATIVE-WATERMARKING-FOR-3D-GAUSSIAN-SPLATT"
source: https://arxiv.org/pdf/2608.17447v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 06:08:19"
field: "3D内容版权保护"
keywords: ["3D Gaussian Splatting", "Watermarking", "Native Protection", "Partial Infringement", "Copyright Protection"]
innovations: ["首个本地化原生水印框架，可直接从任意局部高斯基元子集中解码版权信息", "渐进式梯度注入策略，通过冻结解码器引导场景优化实现全场景均匀覆盖", "支持与间接水印方法混合保护及图像等多模态水印扩展"]
benchmarks: ["Blender", "LLFF", "Mip-NeRF 360"]
---

# 论文速读：NGS-MARKER: ROBUST NATIVE WATERMARKING FOR 3D GAUSSIAN SPLATTING

## 一句话总结
NGS-Marker 是一种针对 3D Gaussian Splatting (3DGS) 的原生水印框架，能够直接在局部高斯基元上嵌入并解码版权信息，有效防御“部分侵权”场景，同时保持渲染质量与鲁棒性。

## 研究问题与动机
- **部分侵权 (Partial Infringement) 威胁**：由于 3DGS 基元的独立性和分块光栅化特性，攻击者可轻易提取并重用受保护资产的局部组件，现有基于渲染图像的水印方法在此类场景下检测准确率降至约 50%（随机水平）。
- **间接保护的固有缺陷**：现有方法依赖渲染图像作为水印载体，不仅引入视觉退化风险，且在视角或内容分布发生偏移时失效，无法对底层 3D 基元提供直接保护。
- **原生水印的可行性待验证**：如何将神经网络有效应用于非结构化、规模任意的高斯基元集合进行不可见水印嵌入与解码，尚缺乏系统研究。

## 核心贡献（创新点）
- **首个本地化原生水印框架**：NGS-Marker 可直接从任意局部高斯基元子集中解码版权信息，与依赖渲染图像的间接方法形成本质区别。
- **渐进式梯度注入策略**：通过冻结解码器引导场景级优化，实现水印在全场景的均匀分布，避免朴素分块注入导致的边界不一致与累积失真。
- **混合保护与多模态扩展**：支持与现有间接水印方法协同优化，并可将水印消息扩展为图像等其它模态，提升实际部署的灵活性。

## 方法详解
- **联合训练注入器与解码器**：采用 PointTransformer 构成的扰动特征生成器 $\mathcal{P}_g$ 与扰动解码器 $\mathcal{P}_d$，结合 CLIP 文本编码器将水印消息 $\mathcal{M}$ 转化为提示特征 $f_m$，对采样局部 patches $\tilde{\mathcal{G}}^s$ 生成潜层扰动特征 $f_d$，并通过解码器预测每个基元的扰动量：$\tilde{\mathcal{G}}^w = \tilde{\mathcal{G}}^s + \mathcal{P}_d(\tilde{\mathcal{G}}^s; f_d)$。
- **损失函数**：训练阶段联合优化渲染保真度（MSE）与消息提取精度（BCE）：$\mathcal{L}_t = \text{MSE}(\mathcal{R}(\tilde{\mathcal{G}}^s, \theta), \mathcal{R}(\tilde{\mathcal{G}}^w, \theta)) + \lambda_t \cdot \text{BCE}(\mathcal{M}, \hat{\mathcal{M}})$。
- **渐进式嵌入**：在目标场景 $\mathcal{G}^w$ 上，通过梯度下降联合最小化渲染损失与解码损失 $\mathcal{L}_w = \text{MSE}(\mathcal{R}(\mathcal{G}^s, \theta), \mathcal{R}(\mathcal{G}^w, \theta)) + \lambda_w \cdot \text{BCE}(\mathcal{M}^{id}, \hat{\mathcal{M}})$，实现全场景均匀覆盖。
- **所有权验证**：用户可随机采样可疑区域基元输入解码器，通过可视化着色（基于 SH 系数）直观定位侵权区域。

## 实验与结果
- **数据集与基线**：使用 Blender、LLFF、Mip-NeRF 360 等公开数据集（24 训练场景，4 测试场景），对比 3D-GSW、GaussianMarker、GuardSplat 及两种本地注入变体（WI-Naive、WI-Iterative）。
- **部分侵权检测性能**：16-bit 水印下，NGS-Marker 的 Bit-Acc 达 97.94%，3D-Acc 达 96.60%；现有方法准确率均约为 50%（随机猜测）。
- **渲染质量**：16-bit 时 PSNR 为 40.17，SSIM 为 0.995，LPIPS 仅为 0.007，显著优于所有对比方法。
- **鲁棒性**：在高斯噪声、旋转、缩放、密度变化、随机删除等攻击下 Bit-Acc 保持 96% 以上；与间接方法混合保护后仍能同时保障渲染图像与基元数据的安全。
- **最强结果**：在 bear 场景 16-bit 测试中，NGS-Marker 的 Bit-Acc 与 3D-Acc 均接近 98%，较最佳基线（GuardSplat 约 51%）提升约 47 个百分点。

## 相关工作脉络
- **HiDDeN (Zhu et al. 2018)**：开创端到端图像水印网络，本文将其思想推广至非结构化 3D 基元空间。
- **3D-GSW / GaussianMarker / GuardSplat**：现有 3DGS 水印基准，均依赖渲染图像提取水印；本文指出它们在部分侵权场景下完全失效（准确率约 50%）。
- **GS-Hider / WaterGS**：将隐藏场景嵌入全局 3DGS，但未解决局部基元独立被侵权的问题；本文聚焦细粒度原生保护。
- **SecureGS**：针对 Scaffold-GS 结构的专用保护，不具备通用性；NGS-Marker 适用于任意规模与结构的 3DGS 资产。
- **3D 点云/网格水印**：点云水印常扰动坐标或密度，网格水印依赖顶点位移或拓扑变换；本文直接作用于显式高斯参数，与前序工作形成互补。

## 局限性与未来方向
- **3D 编码技术尚不成熟**：相比 2D 领域，当前 3D 特征学习网络（如 PointTransformer）在容量与鲁棒性上仍有提升空间。
- **大规模场景扩展性待优化**：虽然理论上支持任意尺度场景，但需设计更高效的分区与遍历策略以降低嵌入时间。
- **对抗攻击防御能力有限**：实验显示对黑盒 C&W 攻击有一定抵抗力，但面对白盒优化攻击（尤其超参数匹配时）性能会下降，需进一步研究。

## 研究启发与可借鉴点
- **从渲染域回归原生数据域**：将水印嵌入对象从“输出图像”转向“底层表示”，为点云、网格等其他 3D 表示的原生保护提供了可迁移范式。
- **梯度引导的渐进式注入机制**：利用冻结解码器作为监督信号进行场景级优化，该策略可应用于其他需要全局一致性的隐式表示任务（如神经辐射场编辑、压缩）。
- **多模态消息嵌入的通用框架**：通过替换编码器/解码器模块即可支持图像、文本等多模态水印，表明其可作为多模态版权管理的基础组件。

## 关键术语表
- **3D Gaussian Splatting (3DGS)**：一种基于各向异性高斯基元集合的显式辐射场表示，支持实时、高保真渲染。
- **Partial Infringement**：部分侵权，指攻击者仅提取并重用受保护 3D 资产中的局部组件而不触发整体检测的行为。
- **Native Watermarking**：原生水印，指直接作用于 3D 数据底层参数（如高斯基元属性）而非经过渲染中间产物的水印嵌入方式。
- **PointTransformer**：一种基于注意力机制的点云处理网络，用于提取非结构化 3D 几何的空间特征。
- **SH Coefficients**：球面谐波系数，用于编码高斯基元的视图相关颜色属性，本文将其用于侵权区域可视化。
- **Bit-Acc / 3D-Acc**：Bit-Acc 衡量提取比特序列与原始消息的相似度；3D-Acc 衡量单个高斯基元被正确分类为“受保护”的比例。

## 可复现要素
- **数据集**：公开 3D 数据集（Blender、LLFF、Mip-NeRF 360 等），训练集 24 场景，测试集 4 场景（chair、garden、bear、person-small）。
- **代码/权重**：作者声明将开源代码至匿名仓库（https://anonymous.4open.science/r/NGS-Marker/），模型权重未在论文中公开。
- **关键超参**：嵌入基数元数 δ=8192，训练批次 k=8192，权重系数 λ_t=5、λ_w=5，优化迭代次数依场景复杂度自适应，直至评估指标稳定。
