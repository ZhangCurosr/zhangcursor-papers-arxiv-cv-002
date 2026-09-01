---
title: "RetiWave-Mamba-A-Dual-Stream-Network-for-Retinal-Disease-Det"
source: https://arxiv.org/pdf/2608.17623v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 01:02:33"
field: "医学图像分析与视网膜疾病智能诊断"
keywords: ["Retinal disease classification", "Optical coherence tomography", "Discrete wavelet transform", "Mamba", "State space models", "dual-stream network", "speckle noise suppression"]
innovations: ["双通道频域解耦架构结合DWT与Mamba状态空间模型实现结构与纹理并行建模", "频率自适应Mamba投影器FAMP通过SE掩码分裂+Mamba长程建模+自适应权重重调制细粒度高频特征", "注意力引导门控融合AGF机制替代简单相加阻断散斑噪声跨尺度传播"]
benchmarks: ["OCT-C8", "OCT2017"]
---

# 论文速读：RetiWave-Mamba-A-Dual-Stream-Network-for-Retinal-Disease-Detection

## 一句话总结
本文提出RetiWave-Mamba双通道网络，通过离散小波变换将OCT图像解耦为低/高频流，分别融合MCLM上下文定位与AG-HRNet+FAMP频率自适应Mamba建模，在OCT-C8数据集上以98.25%准确率实现SOTA视网膜疾病分类，并展现出强抗散斑噪声鲁棒性。

## 研究问题与动机
- OCT图像固有**散斑噪声**严重模糊结构边界与细粒度纹理，传统CNN难以有效滤波
- 病变**尺度多变**且类别间形态相似（如CNV与Drusen），CNN固定感受野难以同时兼顾全局上下文理解与精准病灶定位
- 现有**多尺度融合**（简单相加/拼接）会将低频分支的背景噪声直接传递至高分辨分支，破坏高频特征质量
- 当前**长程依赖建模**方法（如Transformer）计算复杂度高，且对高频细粒度纹理的保留能力不足，而Mamba等SSM在频域的应用几乎空白

## 核心贡献（创新点）
- **双通道频域解耦架构**：首次将DWT与Mamba状态空间模型结合，低频流负责全局结构上下文，高频流负责细粒度纹理与边缘，显著优于单一空间域方法
- **多尺度上下文定位模块MCLM**：在ResNet-50低频分支后插入，融合多尺度空洞卷积（膨胀率{6,12,18}）与小波空间注意力MSW_SA，在扩展感受野的同时抑制背景噪声，定位精度超越ASPP/CBAM/SE等基线
- **注意力引导高分辨率网络AG-HRNet**：用智能门控融合AGF机制替代传统相加，仅允许高置信度纹理信息跨分辨率传播，有效阻断散斑噪声传播路径
- **频率自适应Mamba投影器FAMP**：利用SE掩码将AG-HRNet输出动态分裂为低/高频成分，经Mamba建模长程依赖后生成自适应通道权重W_L/W_H重新调制，兼顾全局序列建模与局部纹理保留
- **渐进式噪声注入训练策略**：三阶段课程学习（σ∈[0,0.1]→[0.05,0.15]→[0.1,0.2]）显著提升模型在强散斑噪声下的诊断稳定性

## 方法详解
- **DWT分解**：采用Haar小波，输入3通道OCT图像经2D DWT分解为1个低频子带$I_{LL}$（保留稳定结构）和3个高频子带$I_{LH}, I_{HL}, I_{HH}$（水平/垂直/对角纹理），三高频子带沿通道拼接为9通道输入
- **MCLM结构**：①多尺度膨胀融合MDF：$X'=σ(B(W_{proj}*X))$，并行标准卷积与3条膨胀率$r∈\{6,12,18\}$分支后求和$F_r=σ(B(W_1*X'+Σ B(W_{3,r}*_{r}X')))$$，残差融合$F=σ(B(W_{out}*F_r)+X)$；②MSW_SA：池化聚合后经4分支WTConv生成空间注意力图$M_S=δ(W_{MSW\_SA}*\sum_{k=1}^{4}B(W_{wt,k}(P_{max}(F)+P_{avg}(F))))$，最终$y=F+(F⊗M_S)$
- **AG-HRNet**：4条并行高分辨率分支，跨分支融合采用注意力门控：$G=δ(B(W_{gate}*F_l'))$，$y_{fused}=F_h+(F_l'⊗G)$，有效滤除噪声传播
- **FAMP**：SE掩码$M=δ(W_{se}*P_{avg}(X))$，频域分裂$X_L=X⊗M$、$X_H=X⊗(1-M)$，拼接后 flatten归一化为序列$X_{seq}$，经Mamba建模得权重$W=δ(P(M(X_{seq})))$，拆分$W_L,W_H$自适应调制后融合$Y=X+F_{fusion}((X_L⊗W_L)+(X_H⊗W_H))$
- **分类头**：低频分支末端GAP得嵌入向量，高频分支4个FAMP输出拼接后GAP得嵌入向量，二者Concat输入全连接分类器；损失为Weighted Cross-Entropy Loss处理类别不均衡

## 实验与结果
- **数据集**：OCT-C8（24000张，8类各3000张），训练集21200张（合并原train+val），测试集2800张；外部验证OCT2017（4类各250张，共1000张）
- **评估指标**：Accuracy、Precision、Sensitivity、F1-score
- **基线对比**：覆盖ResNet-50/VGG16/GoogLeNet/DenseNet121/InceptionV3/EfficientNet-B3/Swin_Tiny/ConvNeXtV2等主流Backbone，以及FPN-ResNet50/FPN-DenseNet121/HTC-Retina/MSLI-Net/WaveNet-SF等视网膜专项/混合模型
- **主要结果**：OCT-C8测试集准确率**98.25%**（Precision 98.27%，Sensitivity 98.25%，F1 98.25%），较ResNet-50提升**+0.64%**，全面超越所有对比方法；OCT2017外部泛化准确率**99.60%**
- **消融结论**：ResNet50(97.61%)→+DWT(97.79%)→+MCLM(98.14%)→完整(98.25%)，逐级递增；MDF以33.57M参数/6.58 GFLOPs达到与ASPP(142.63M/27.14 GFLOPs)相同的98.25%精度，参数节省**76.5%**、算力降低**75.7%**；MSW_SA(98.25%)优于SE(98.00%)和CBAM(97.96%)
- **鲁棒性**：在32.19/28.69/26.22 dB三种噪声强度下均保持SOTA，26.22 dB极强噪声时仍达**97.79%**，甚至超过ResNet-50无噪声基线(97.75%)
- **细粒度区分**：AG-HRNet与FAMP在CNV↔Drusen这对最难分类对上混淆率显著低于标准HRNet与标准Mamba

## 相关工作脉络
- **WaveNet-SF / MSLI-Net**：频域学习方法，但高频流仍依赖标准卷积+简单融合，未解决散斑噪声传播与类间细粒度纹理区分问题；本文通过AGF门控+FAMP自适应调制弥补此缺陷
- **HTC-Retina / SViT**：CNN-Transformer混合架构，凭借ViT全局感知取得较好性能，但自注意力二次复杂度制约高分辨率OCT处理；本文以Mamba替代Transformer，在保持长程建模能力的同时将计算开销控制在合理范围
- **MRVM (Zuo et al.)**：首个将Mamba引入视网膜疾病检测的工作，采用多方向选择性扫描捕捉全局上下文；本文首次将Mamba置于**高频细节分支**，聚焦细粒度纹理的长距离依赖建模，形成互补定位
- **SE / CBAM / Coordinate Attention**：经典注意力基线；本文MSW_SA创新性地将WTConv嵌入注意力图生成过程，在频域层面天然滤除散斑噪声，避免传统空间注意力易受伪影干扰的短板
- **WaveViT**：将小波变换引入Vision Transformer以实现无损下采样；本文与之不同，将DWT作为**双流解耦的前置操作**而非替换下采样，并进一步耦合Mamba与门控高分辨率网络，面向医学图像噪声鲁棒性专门设计

## 局限性与未来方向
- 代码与预训练权重**未在论文中公开**，复现依赖自行实现
- 实验仅在OCT-C8与OCT2017两个数据集验证，**缺乏更多外部多中心数据集**的泛化检验
- FAMP将2D特征展平为1D序列后送入Mamba，可能丢失部分空间局部结构信息，序列建模与空间精细性的平衡有待优化
- 渐进式噪声注入训练策略虽提升鲁棒性，但未系统分析不同噪声模型（如运动模糊、伪影）下的适应性
- 临床部署的**实时推理延迟**与显存占用未见详细统计，轻量化潜力待评估

## 研究启发与可借鉴点
- **频域解耦+双通道并行**的设计范式可有效分离结构与细节，对含噪医学图像（超声、内窥镜、X光）具有高度可迁移价值
- **智能门控融合AGF**替代简单相加/拼接的思路，适用于任何多尺度特征交互场景，值得推广至分割/检测任务
- **FAMP将SE掩码与Mamba选择性扫描结合**，实现了"频率自适应长程建模"，可迁移至遥感、病理切片等细粒度分类任务
- **渐进式噪声课程学习策略**（低σ→高σ三阶段）提升模型对设备差异与采集伪影的容忍度，可作为通用鲁棒训练范式
- MCLM中**WTConv小波注意力**与空洞卷积的组合，兼顾了感受野扩展与频域去噪，为高精度定位模块设计提供了新思路

## 关键术语表
- **Discrete Wavelet Transform (DWT)**：将图像分解为低频近似分量与三个高频细节分量（水平/垂直/对角）的频域分析方法，本文采用Haar小波
- **Multi-scale Contextual Localization Module (MCLM)**：低频分支核心模块，融合多尺度空洞卷积与WTConv小波空间注意力，扩展感受野并精准定位病灶
- **Attention-Guided High-Resolution Network (AG-HRNet)**：高频分支骨干，以智能门控融合AGF机制替代HRNet的简单相加，阻断散斑噪声跨尺度传播
- **Frequency-Adaptive Mamba Projector (FAMP)**：高频分支末端模块，利用SE掩码动态分裂低/高频成分，经Mamba建模长程依赖后自适应重加权融合
- **Speckle Noise**：OCT成像中由相干光散射产生的固有颗粒状噪声，会模糊组织边界并干扰特征提取
- **State Space Models (Mamba)**：基于选择性状态空间的序列建模架构，以线性时间复杂度捕获长程依赖，已扩展至视觉任务
- **Grad-CAM / LayerCAM**：基于梯度的类激活可视化方法，本文用于展示模型关注区域与病灶空间一致性
- **OCT-C8 / OCT2017**：公开的视网膜OCT图像分类基准数据集，前者含8类各3000张，后者含4类各250张

## 可复现要素
- **数据集**：OCT-C8公开（https://doi.org/10.5281/zenodo....）；OCT2017公开（Kermany et al.，Mendeley Data）
- **代码/权重**：论文未提及开源
- **输入尺寸**：448×448
- **优化器**：Adam（β₁=0.9，β₂=0.999）
- **学习率**：初始2×10⁻⁴，Cosine Annealing调度，5 epoch warm-up
- **Batch size**：32
- **训练轮数**：60 epochs
- **正则化**：Weight decay=1×10⁻⁴，随机旋转+水平翻转数据增强
- **混合精度**：AMP
- **损失函数**：Weighted Cross-Entropy Loss
- **硬件**：单卡NVIDIA GeForce RTX 4090 24GB
- **评估方式**：6次独立实验取平均
