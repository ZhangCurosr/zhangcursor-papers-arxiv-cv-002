---
title: "TISC-A-Text-Driven-Image-Semantic-Communication-System-for-F"
source: https://arxiv.org/pdf/2608.16100v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 15:24:12"
---

# 论文速读：TISC-A-Text-Driven-Image-Semantic-Communication-System-for-F

## 一句话总结
本文提出TISC框架，通过将图像编码为层级结构化文本语义描述，并在发射端仿真择优扩散模型的初始噪声种子，以极低带宽代价实现接收端高保真、高语义一致性的图像重建。

## 研究问题与动机
- 现有文本驱动图像语义通信在发射端I2T阶段存在语义失真：MLLM一次性生成整图描述时，常遗漏细粒度对象属性（颜色、材质、姿态）与精确空间位置信息。
- 接收端T2I阶段因扩散模型依赖随机初始噪声，相同文本描述下不同噪声种子会引导生成轨迹分化，导致重建结果的语义一致性波动较大。
- 传统特征级/像素级语义通信带宽开销仍较高，难以满足6G超低带宽需求；文本驱动范式虽可节省20倍以上比特数，但缺乏对语义提取过程与生成噪声的显式协同优化。
- 现有基线方法多直接调用现有MLLM或图像描述模型，未针对“语义保真重建”这一通信目标设计专门的信源结构化编码与噪声选择机制。

## 核心贡献（创新点）
- 提出TSASE（Tree-Structured Attribute Semantic Extraction）树状结构化属性语义提取模块，将单轮复杂描述任务解耦为全局场景、背景、目标对象及细粒度属性的层级子任务，并通过L L M压缩冗余表述。与直接提示MLLM生成整图自由文本的方法本质不同，TSASE以显式结构约束强制覆盖空间坐标与物理属性，从根本上减少信息遗漏。
- 设计INO（Initial Noise Optimization）初始噪声优化机制，在发射端利用与接收端同构的生成模型对 $m$ 个候选噪声种子进行本地仿真评估，并将表现最优的随机种子 $s^*$ 随文本下发。与常规T2I随机采样噪声的流程相比，INO将生成轨迹选择前移至发送端，实现发送-接收协同的确定性保真。
- 构建融合感知距离与结构化解耦语义匹配的综合相似度度量 $S_{\mathrm{comp}}$，其中语义项引入基于IoU的目标空间对齐、漏检/幻觉惩罚以及全局-背景-对象三级SBERT匹配。与仅依赖LPIPS或整图文本余弦相似度的评估方式相比，该度量同时刻画视觉保真与细粒度语义一致性。
- 提供基于高斯分布统计推断的噪声候选搜索预算理论推导，给出 $m_k \geq \lceil \ln(1-p) / \ln(\Phi(k)) \rceil$ 的闭式解，使系统可依据目标质量阈值与置信度自适应配置仿真开销，避免盲目网格搜索。

## 方法详解
- **TSASE发射端语义提取**：以整图为根节点，调用MLLM生成全局描述 $D_{\mathrm{global}}$ 与背景描述 $D_{\mathrm{bg}}$；YOLOv11检测 $N$ 个目标得到边界框 $\mathcal{B}=\langle b_1,\dots,b_N\rangle$，按置信度阈值过滤后裁剪出单目标子图作为第一级子节点；空间位置直接采用bbox坐标 $(x^{tl},y^{tl},x^{br},y^{br})$，其余属性（形状/姿态、颜色、材质/物理）由MLLM逐项描述；原始冗长描述经GPT-4o压缩为简洁版 $d_i^S$；最终拼接为 $D=\{D_{\mathrm{global}}, D_{\mathrm{bg}}, D_{\mathrm{obj}}\}$，其中 $D_{\mathrm{obj}}=[d_1^S,\dots,d_N^S]$。
- **LMD接收端重建适配**：采用开源LLM-grounded Diffusion (LMD) 基座。原版LMD需LLM从自由文本推理对象空间关系，本文跳过该推理步，直接将TSASE输出的结构化描述 $D$ 作为条件注入扩散生成阶段，与选定初始噪声 $z^*$ 共同驱动图像合成。
- **INO发送端噪声寻优**：发射端维护同构生成模型 $G(\cdot)$，测试候选集 $Z=[z_1,\dots,z_m]$，生成 $\hat{I}_q=G(D,z_q)$；计算 $S_{\mathrm{comp}}(I,\hat{I}_q)$ 并选取 $z^*=\arg\max_{z_q} S_{\mathrm{comp}}$；仅将映射得到的随机种子 $s^*=f(z^*)$ 以文本形式发送，开销极低。
- **综合相似度 $S_{\mathrm{comp}}$ 构造**：$S_{\mathrm{comp}}=\alpha S_{\mathrm{vis}}+(1-\alpha)S_{\mathrm{sem}}$。视觉项 $S_{\mathrm{vis}}=1-\mathrm{LPIPS}(I,\hat{I}_q)$；语义项 $S_{\mathrm{sem}}$ 对原图与候选图分别运行TSASE，SBERT计算 $D_{\mathrm{global}}$ 与 $D_{\mathrm{bg}}$ 相似度，利用IoU（阈值>0.5）建立原图与候选图目标对象的一一匹配，匹配对计算SBERT相似度，
