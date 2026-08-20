# WHEN SIMPLICITY WINS: BOTTLENECK-AWARE CONTEXT MODELING FORLIGHTWEIGHT SEMANTIC SEGMENTATION

Mian Muhammad Naeem Abid, Nancy Mehta, Zongwei Wu, Radu Timofte

Computer Vision Lab, CAIDAS & IFI, University of Wurzburg, Germany¨

## ABSTRACT

Semantic segmentation demands a careful balance between accuracy, efficiency, and scalability, which remains difficult to achieve for high-resolution imagery. Convolutional networks effectively model local patterns but struggle with long-range dependencies, whereas Vision Transformers capture global context at a high computational cost. While recent work largely focuses on encoder design, the bottleneck stage—central to contextual aggregation and information flow—has been relatively overlooked. We propose SiConMo, a lightweight yet effective framework, implemented in two variants: an RGB-only model (SiConMo) and a GME-enhanced variant (SiConMo ). We show that simplicity arises from a key design principle: at very low computational budgets, the bottleneck is the most efficient stage to integrate local and global context. SiConMo integrates three complementary components: a Token Pyramid Extraction Module for hierarchical multi-scale representation, a Transformer-Branched Depthwise Convolution block for bottleneck-aware context modeling, and a Feature Merging Module that preserves spatial structure while enhancing semantic consistency. Extensive experiments on ADE20K, PASCAL Context, Cityscapes, and COCO-Stuff demonstrate that SiConMo achieves a state-of-the-art accuracy–efficiency trade-off among lightweight semantic segmentation models, highlighting simplicity as a powerful design principle. https://github.com/miannaeem-lab/SiConMo

Index Terms— Lightweight Segmentation, Vision Transformers, Convolutional Neural Networks, Simplicity, Context Modeling, Efficiency

## 1. INTRODUCTION

Semantic segmentation is a fundamental computer vision task that requires dense, pixel-level prediction in complex visual scenes [1]. Convolutional Neural Networks (CNNs) have historically dominated this area due to their strong local feature modeling capabilities, but their limited receptive fields restrict long-range dependency modeling. Vision Transformers (ViTs) address this limitation through global self-attention [2], yet their quadratic computational complexity makes them impractical for high-resolution and real-time segmentation.

![](images/d0ebf25b58990571e6129121d9ba55830f685c7edc8698c2e070008bfb66e739.jpg)  
Fig. 1. (Top) Efficiency–accuracy comparison of lightweight semantic segmentation models on ADE20K validation set. Circle sizes denote GFLOPs. SiConMo achieves a superior trade-off, illustrating the effectiveness of a simple bottleneckaware design principle. (Bottom) Variants of the proposed SiConMo architecture.

To reduce the overhead of ViTs, prior work explores localized or window-based attention, separable attention mechanisms, and aggressive spatial downsampling [3, 4]. While computationally efficient, these strategies often degrade segmentation accuracy by limiting receptive fields or discarding fine-grained spatial information. Conversely, lightweight CNNs based on depthwise separable convolutions [5] scale efficiently but struggle to capture global context, resulting in suboptimal performance on complex datasets.

Recent hybrid CNN–Transformer models attempt to balance efficiency and expressiveness [6], yet most designs emphasize encoder complexity while overlooking the bottleneck stage, where contextual aggregation and feature redistribution naturally occur at minimal computational cost. Transformerbased bottlenecks often introduce redundant global attention, whereas CNN-based bottlenecks remain restricted to local receptive fields. This suggests that the bottleneck is a uniquely efficient point for mixing global and local context, but it remains underutilized in existing designs.

This work challenges the assumption that increasing encoder complexity is the primary path to improved performance. Instead, we show that simplicity, when placed at the right architectural location, can be more effective than overengineering. We propose SiConMo, a lightweight bottleneckaware framework built on this design principle. We further evaluate an enhanced variant, SiConMo , which incorporates additional Gradient Magnitude and Edge Maps (GME) to improve structural awareness. SiConMo integrates local and global feature processing without unnecessary architectural depth. It consists of (1) a Token Pyramid Extraction Module (TPEM) for efficient multi-scale token representation, (2) a Transformer-Branched Depthwise Convolution (Trans-BDC) block that enables lightweight context mixing at the bottleneck, and (3) a Feature Merging Module (FMM) that preserves spatial structure while enhancing semantic consistency. Together, these components form a lightweight architecture that achieves a favorable efficiency–accuracy trade-off, as shown in Figure 1.

We conduct a comprehensive and rigorous evaluation of SiConMo across multiple challenging benchmarks. Our results consistently show that SiConMo achieves a strong and reliable balance between segmentation accuracy and computational efficiency, making it well-suited for real-time and resource-constrained applications.

Our contributions are as follows:

• We introduce SiConMo, a lightweight bottleneck-aware framework that demonstrates how simplicity in design, when applied at the bottleneck stage, can rival more complex architectures.

• We propose and integrate the Transformer-Branched Depthwise Convolution (Trans-BDC) block, together with a Token Pyramid Extraction Module and a Feature Merging Module, enabling efficient multi-scale representation and lightweight context mixing at minimal computational cost.

• We extensively evaluate SiConMo on challenging segmentation benchmarks, including ADE20K [7], PASCAL Context [8], Cityscapes [9], and COCO-Stuff [10], and further demonstrate its generalization capability on COCO object detection under real-time efficiency constraints.

## 2. RELATED WORK

Efficient Convolutional Neural Networks: Convolutional Neural Networks (CNNs) have long been central to visual recognition due to their strong inductive biases [11]. However, their computational cost limits deployment in resource-constrained settings. Lightweight CNNs such as MobileNet [12, 5], ShuffleNet [13], and GhostNet [14] address this issue through depthwise separable convolutions and efficient designs. For semantic segmentation, approaches like DFANet [15] further reduce complexity via multi-scale feature aggregation. These methods, however, primarily emphasize local context modeling. In contrast, SiConMo enhances context representation by integrating multi-scale token processing and adaptive feature redistribution within a lightweight bottleneck.

Lightweight Vision Transformers: Vision Transformers (ViTs) enable global context modeling through self-attention but incur high computational overhead. To improve efficiency, models such as LeViT [16] and EfficientFormer [3] introduce architectural and training optimizations or combine convolutional and transformer components. While these designs balance efficiency and accuracy, many rely on complex attention mechanisms. In contrast, SiConMo focuses on a simple, bottleneck-aware design that captures both local and global context with minimal computational overhead.

## 3. PROPOSED METHOD

We propose SiConMo, a simple yet effective, lightweight, bottleneck-aware framework for semantic segmentation that balances representational richness with computational efficiency. Rather than relying on heavy backbones or complex multi-stage architectures, SiConMo demonstrates that carefully designed lightweight modules can effectively capture both local structural details and global contextual dependencies. As illustrated in Figure 2, the framework consists of three key components: (1) the Token Pyramid Extraction Module (TPEM), (2) the Trans-BDC bottleneck block, and (3) the Feature Merging Module (FMM).

## 3.1. Token Pyramid Extraction Module

The Token Pyramid Extraction Module (TPEM) constructs hierarchical multi-scale representations from the input image. Depending on the variant, the input is either RGB alone or RGB concatenated with gradient magnitude and edge maps to enrich structural cues. TPEM employs a sequence of inverted residual blocks from MobileNetV2 [5, 6] to generate feature maps $\{ S _ { 1 } , S _ { 2 } , S _ { 3 } , S _ { 4 } \}$ at progressively coarser resolutions, where each feature map has size $\begin{array} { r } { \frac { H } { 2 ^ { i + 1 } } \stackrel { - } { \times } \frac { W } { 2 ^ { i + 1 } } \times C _ { i } } \end{array}$ for $i \in \{ 1 , 2 , 3 , 4 \}$

To reduce computational cost while preserving contextual diversity, each feature map is pooled and concatenated along the channel dimension:

$$
X _ { f } = \langle S _ { 1 } ^ { \varphi } , S _ { 2 } ^ { \varphi } , S _ { 3 } ^ { \varphi } , S _ { 4 } ^ { \varphi } \rangle ,\tag{1}
$$

where $\varphi$ denotes average pooling. This operation efficiently combines fine-grained local details with coarse global context, producing a compact representation suitable for bottleneck processing.

![](images/cedfdedf1386d94b79067a26d1d767cae9415f3b87f12e55b22da76ba767dd00.jpg)  
Fig. 2. Overview of the proposed SiConMo framework. The architecture integrates the Token Pyramid Extraction Module (TPEM) for multi-scale representation, the Trans-BDC block for bottleneck-aware context modeling, and the Feature Merging Module (FMM) for adaptive spatial–semantic integration.

## 3.2. Trans-BDC Block: The Novel Bottleneck

The Trans-BDC block forms the core bottleneck of SiConMo, integrating local and global feature modeling within a single lightweight module. Unlike conventional bottlenecks that rely solely on convolutions or self-attention, Trans-BDC combines both paradigms through two parallel branches.

Branched Depthwise Convolution (BDC) Branch: This branch captures localized spatial patterns using three parallel operations: a $3 \times 3$ depthwise convolution $( \xi _ { 3 \times 3 } ^ { d w } )$ for neighborhood modeling, a $1 \times 1$ depthwise convolution $( \xi _ { 1 \times 1 } ^ { d w } )$ for lightweight per-channel refinement, and a depthwise separable convolution $( \xi _ { 1 \times 1 } ^ { p w } ( \xi _ { 3 \times 3 } ^ { d w } ( . ) )$ for efficient feature fusion. The aggregated output is:

$$
\delta _ { c } ^ { \prime } = \xi _ { 3 \times 3 } ^ { d w } ( X _ { f } ) + \xi _ { 1 \times 1 } ^ { d w } ( X _ { f } ) + \xi _ { 1 \times 1 } ^ { p w } ( \xi _ { 3 \times 3 } ^ { d w } ( X _ { f } ) ) + X _ { f } ,\tag{2}
$$

which is further refined via channel attention:

$$
X _ { B D C } = \Gamma _ { 2 } ( \Gamma _ { 1 } ( \varphi _ { g } ( \delta _ { c } ^ { \prime } ) ) ) \bar { \otimes } \delta _ { c } ^ { \prime } .\tag{3}
$$

Here, $\varphi _ { g }$ denotes global average pooling, $\Gamma _ { 1 }$ and $\Gamma _ { 2 }$ are fully connected layers, and ⊗<sup>¯</sup> represents the Hadamard product.

Lightweight ViT Branch: In parallel, a lightweight transformer branch models long-range dependencies using selfattention applied to pooled tokens. Efficiency is achieved through low-dimensional projections for queries, keys, and values, $1 \times 1$ convolutions in place of standard MLPs, batch normalization, and ReLU6 activations:

$$
\begin{array} { r } { X _ { V i T } = A t t e n t i o n ( X _ { f } ) + X _ { f } , } \\ { A t t e n t i o n = \mathrm { s o f t m a x } \left( \frac { Q K ^ { T } } { \sqrt { d _ { k } } } \right) V . } \end{array}\tag{4}
$$

The outputs of both branches are fused and passed through a depthwise-enhanced Feed-Forward network (FFN):

$$
X _ { f } ^ { \prime } = X _ { V i T } + X _ { B D C } , \quad X _ { f } ^ { \prime \prime } = F F N ( X _ { f } ^ { \prime } ) + X _ { f } ^ { \prime } .\tag{5}
$$

This design shows that a carefully constructed hybrid bottleneck can replace more complex architectures while maintaining strong segmentation performance.

## 3.3. Feature Merging Module and Segmentation Head

The Feature Merging Module (FMM) integrates local features from TPEM with global features from the Trans-BDC block using a gated fusion mechanism [6]. Local features are projected via a $1 \times 1$ convolution $( \xi _ { 1 \times 1 } ^ { c 1 } )$ , while global features are adaptively weighted through a sigmoid-activated $1 \times 1$ convolution, $\sigma ( \xi _ { 1 \times 1 } ^ { c 2 } ( . ) )$ . The merged feature map is:

$$
Y _ { f } = ( \xi _ { 1 \times 1 } ^ { c 1 } ( X _ { f } ) ) \bar { \otimes } \left( \sigma ( \xi _ { 1 \times 1 } ^ { c 2 } ( X _ { f } ^ { \prime \prime } ) ) \right) + \xi _ { 1 \times 1 } ^ { c 3 } ( X _ { f } ^ { \prime \prime } ) .\tag{6}
$$

The fused representation is upsampled to the target resolution and processed by a lightweight segmentation head consisting of two $1 \times 1$ convolutions to produce the final prediction $Y$

## 3.4. Variants and Design Philosophy

SiConMo is implemented in two variants: an RGB-only configuration and an enhanced version incorporating Gradient Magnitude and Edge Maps (GME), denoted as SiConMo and SiConMo . Across all components, SiConMo follows a minimalistic design philosophy: preserving multi-scale representations without deep encoders, employing lightweight hybrid bottlenecks instead of full transformers, and using simple yet effective feature fusion. These choices highlight that careful bottleneck-aware design can achieve strong accuracy and efficiency without architectural over-complexity.

## 4. EXPERIMENTS

## 4.1. Results on ADE20K

Table 1(a) reports a quantitative comparison of SiConMo against representative state-of-the-art semantic segmentation methods on the ADE20K validation set. Compared with the higher-capacity transformer-based U-MixFormer model using a MiT-B0 encoder, SiConMo achieves competitive segmentation accuracy while reducing computational cost by 90.2% and parameters by 72.1%. This efficiency is primarily enabled by its bottleneck-aware design, which combines multi-scale token representations with lightweight hybrid context modeling instead of full self-attention. Compared with CNN-based lightweight approaches such as LR-ASPP with MobileNetV3-Large, SiConMo improves mIoU by 1.9% points while operating under 70.0% lower GFLOPs, significantly reduced (70.6%) latency, and 46.9% less parameters. Unlike purely convolutional designs, SiConMo explicitly integrates global context within a compact bottleneck, allowing more effective semantic reasoning without increasing model complexity. Furthermore, under strict real-time constraints, SiConMo outperforms recent lightweight hybrid models including TopFormer and SeaFormer at comparable computational budgets. Overall, these results demonstrate that SiConMo provides a strong balance between segmentation accuracy and efficiency for real-time semantic segmentation. Details ofdatasets, evaluation protocols, and implementation settings are provided in the supplementary material.

Table 1. Quantitative evaluation on ADE20K [7] and complementary analyses. (a) Comparison with state-of-the-art representative semantic segmentation models on ADE20K dataset. (b) Input-resolution variants trained at 448 × 448. (c) Ablation study on ADE20K. (d) GME variants on ADE20K. GFLOPs are reported at 512×512 input resolution with single-scale inference; ‘-’ denotes unreported values. (e) object detection results on COCO dataset using RetinaNet (Sec. 4.3). SiConMo<sub>†</sub>: GME variant.  
(a) Results on ADE20K
<table><tr><td>Models</td><td>Encoder</td><td>mIoU</td><td>GFLOPs</td><td>Params</td><td>Latency(ms)</td></tr><tr><td colspan="6">Heavyweight Models</td></tr><tr><td>PSPNet [17]</td><td>MobileNetV2 [5]</td><td>29.6 19.7</td><td>52.2</td><td>13.7M</td><td>426</td></tr><tr><td>FCN-8s [18]</td><td>MobileNetV2 [5]</td><td>35.8</td><td>39.6 33.8</td><td>9.8M 12.8M</td><td>406</td></tr><tr><td>Semantic FPN [19]</td><td>ConvMLP-S [20]</td><td>36.2</td><td>26.9</td><td>17.1M</td><td>311</td></tr><tr><td>DeepLabV3+ [21]</td><td>EfficientNet [22]</td><td></td><td></td><td></td><td>388</td></tr><tr><td>DeepLabV3+ [21]</td><td>MobileNetV2 [5]</td><td>38.1</td><td>25.8</td><td>15.4M</td><td>414</td></tr><tr><td>Lite-ASPP [21]</td><td>ResNet18 [11]</td><td>37.5</td><td>19.2</td><td>12.5M</td><td>259</td></tr><tr><td>PEM [23]</td><td>STDC1 [23]</td><td>39.6</td><td>16.0</td><td>17.0M</td><td></td></tr><tr><td>DeepLabV3+ [21]</td><td>ShuffleNetv2-1.5x [13]</td><td>37.6</td><td>15.3</td><td>16.9M</td><td>384</td></tr><tr><td>HRÑet-Small [24]</td><td>HRNet-W18-Small [24]</td><td>33.4</td><td>10.2</td><td>4.0M</td><td>256</td></tr><tr><td>SegFormer [2]</td><td>MiT-B0 [2]</td><td>37.4</td><td>8.4</td><td>3.8M</td><td>308</td></tr><tr><td>FeedFormer-B [25]</td><td>MiT-B0 [2]</td><td>39.2</td><td>7.8</td><td>4.5M</td><td></td></tr><tr><td>SegNeXt [26]</td><td>SegNeXt-T [26]</td><td>41.1</td><td>6.6</td><td>4.3M</td><td></td></tr><tr><td>U-MixFormer [27]</td><td>MiT-B0 [2]</td><td>41.2</td><td>6.1</td><td>6.1M</td><td></td></tr><tr><td colspan="6">Lightweight Models</td></tr><tr><td>Lite-ASPP [21]</td><td>MobileNetV2 [5]</td><td>36.6</td><td>4.4</td><td>2.9M</td><td>94</td></tr><tr><td>R-ASPP [5]</td><td>MobileNetV2 [5]</td><td>32.0</td><td>2.8</td><td>2.2M</td><td>71</td></tr><tr><td>HR-NAS-B [28]</td><td>Searched [28]</td><td>34.9</td><td>2.2</td><td>3.9M</td><td></td></tr><tr><td>LR-ASPP [29]]</td><td>MobileNetV3-Large [29]</td><td>33.1</td><td>2.0</td><td>3.2M</td><td>51</td></tr><tr><td>HR-NAS-A [28]</td><td>Searched [28]</td><td>33.2</td><td>1.4</td><td>2.5M</td><td></td></tr><tr><td>LR-ASPP [29]</td><td>MobileNetV3-Large-reduce [29]</td><td>32.3</td><td>1.3</td><td>1.6M</td><td>33</td></tr><tr><td>LeMoRe [30]</td><td>LeMoRe [30]</td><td>32.7</td><td>0.8</td><td>1.6M</td><td>24</td></tr><tr><td>TopFormer [6]</td><td>TopFormer-T [6]</td><td>32.8</td><td>0.6</td><td>1.4M</td><td>16</td></tr><tr><td>SeaFormer [31]</td><td>SeaFormer-T [31]</td><td>34.6</td><td>0.6</td><td>1.7M</td><td>16</td></tr><tr><td>SiConMo (Ours)</td><td>Ours</td><td>34.8</td><td>0.6</td><td>1.7M</td><td>15</td></tr><tr><td>SiConMo+ (Ours)</td><td>Ours</td><td>35.0</td><td>0.6</td><td>1.7M</td><td>15</td></tr></table>

(f) Object detection performance of SiConMo<sub>†</sub> on COCO evaluation metrics

(b) SiConMo variants trained with 448<sup>2</sup> input size
<table><tr><td>Models</td><td>Input</td><td>mIoU</td><td>GFLOPs</td><td>Params</td><td>Latency(ms)</td></tr><tr><td>SiConMo*</td><td>448×448</td><td>34.4</td><td>0.5</td><td>1.7M</td><td>14</td></tr><tr><td>SiConMo↑*</td><td>448×448</td><td>34.7</td><td>0.5</td><td>1.7M</td><td>14</td></tr></table>

(c) Ablation Study of SiConMo on ADE20K
<table><tr><td></td><td></td><td></td><td></td><td></td><td></td><td rowspan=1 colspan=1>mloU</td><td rowspan=1 colspan=1>GFLOPs</td><td rowspan=1 colspan=1>Params</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>√√√√√</td><td rowspan=1 colspan=1>√√VV</td><td rowspan=1 colspan=1>√√√</td><td rowspan=1 colspan=1>√√</td><td rowspan=1 colspan=1>√</td><td rowspan=1 colspan=1>25.425.628.729.529.8</td><td rowspan=1 colspan=1>0.550.550.560.560.56</td><td rowspan=1 colspan=1>1.02M1.10M1.29M1.38M1.38M</td></tr><tr><td rowspan=1 colspan=1>1√√√</td><td rowspan=1 colspan=1>√√</td><td rowspan=1 colspan=1>√√</td><td rowspan=1 colspan=1>√</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>32.732.832.933.8</td><td rowspan=1 colspan=1>0.540.570.570.58</td><td rowspan=1 colspan=1>1.41M1.42M1.43M1.61M</td></tr><tr><td rowspan=1 colspan=1>V√</td><td rowspan=1 colspan=1>√S</td><td rowspan=1 colspan=1>&gt;&gt;</td><td rowspan=1 colspan=1>√√</td><td rowspan=1 colspan=1>√√</td><td rowspan=1 colspan=1>√</td><td rowspan=1 colspan=1>34.835.0</td><td rowspan=1 colspan=1>0.580.58</td><td rowspan=1 colspan=1>1.68M1.68M</td></tr></table>

(d) Comparison of SiConMo GME variants
<table><tr><td>GME (Variations)</td><td>mIoU</td><td>GFLOPs</td><td>Params</td><td>Latency(ms)</td></tr><tr><td>Canny</td><td>31.6</td><td>0.6</td><td>1.7M</td><td>18.8</td></tr><tr><td>Sobel+Canny</td><td>35.0</td><td>0.6</td><td>1.7M</td><td>19.2</td></tr><tr><td>Sobel (Ours)</td><td>35.0</td><td>0.6</td><td>1.7M</td><td>15.1</td></tr></table>

(e) Object detection results on COCO dataset
<table><tr><td rowspan=1 colspan=1>Backbone</td><td rowspan=1 colspan=1>mAP</td><td rowspan=1 colspan=1>GFLOPs</td><td rowspan=1 colspan=1>Parameters</td></tr><tr><td rowspan=2 colspan=1>ShuffleNetV2 [13]MobileNetV3 [29]TopFormer-T [6]SeaFormer-T [31]</td><td rowspan=1 colspan=1>25.9</td><td rowspan=1 colspan=1>161</td><td rowspan=1 colspan=1>10.4M</td></tr><tr><td rowspan=1 colspan=1>27.227.131.5</td><td rowspan=1 colspan=1>162160160</td><td rowspan=1 colspan=1>12.3M10.5M10.9M</td></tr><tr><td rowspan=1 colspan=1>SiConMo (Ours)</td><td rowspan=1 colspan=1>31.6</td><td rowspan=1 colspan=1>160</td><td rowspan=1 colspan=1>10.9M</td></tr></table>

<table><tr><td rowspan=1 colspan=1>Backbone</td><td rowspan=1 colspan=1>mAP</td><td rowspan=1 colspan=1>AP50</td><td rowspan=1 colspan=1>AP75</td><td rowspan=1 colspan=1>APS</td><td rowspan=1 colspan=1>APM</td><td rowspan=1 colspan=1>APL</td></tr><tr><td rowspan=1 colspan=1>SiConMot (Ours)</td><td rowspan=1 colspan=1>31.6</td><td rowspan=1 colspan=1>51.1</td><td rowspan=1 colspan=1>32.8</td><td rowspan=1 colspan=1>17.5</td><td rowspan=1 colspan=1>33.9</td><td rowspan=1 colspan=1>43.0</td></tr></table>

Visual Results: Qualitative comparisons on the ADE20K validation set are presented in Figure 3. Compared with representative lightweight baselines such as TopFormer [6], SiConMo produces segmentation maps with improved boundary delineation, finer structural details, and enhanced spatial consistency. In particular, small objects and complex scene layouts are better preserved, while over-smoothing artifacts are reduced. These visual observations are consistent with the quantitative improvements reported in Table 1(a).

## 4.1.1. Ablation Study

Effects of BDC Branch Design: The upper part of Table 1(c) evaluates the contribution of individual components in the Branched Depthwise Convolution (BDC) branch. Adding a 1 × 1 depthwise convolution to the 3 × 3 baseline provides a small accuracy gain, while introducing a parallel 3 × 3 depthwise separable branch yields a substantial improvement, highlighting the benefit of multi-scale feature modeling. Channel attention further enhances performance with negligible computational overhead.

Effect of Trans-BDC Block and Edge Guidance: The lower part of Table 1(c) analyzes the interaction between the lightweight ViT branch and the BDC components. The ViT branch alone significantly improves segmentation accuracy, and progressively adding BDC components consistently boosts performance, confirming their complementarity. Incorporating gradient magnitude and edge maps provides a modest additional gain, improving boundary localization. As shown in Table 1(d), Canny-based edge guidance increases computation without notable accuracy benefits. Overall, the ablation results validate the effectiveness of each component in the Trans-BDC design.

TopFormer  
![](images/d1e5f2637185ca4741df574e9c55553351ac4cdfb2ac92906648d48c9533ac2d.jpg)  
Fig. 3. Visual results on the ADE20K validation set. The proposed SiConMo model produces high-quality segmentation maps with improved spatial consistency and fine-detail preservation, compared to TopFormer [6].

Table 2. Results on PASCAL Context test set [8].
<table><tr><td rowspan=1 colspan=1>Methods</td><td rowspan=1 colspan=1>Backbone</td><td rowspan=1 colspan=1>mIoU59</td><td rowspan=1 colspan=1>mIoU60</td><td rowspan=1 colspan=1>GFLOPs</td></tr><tr><td rowspan=3 colspan=1>DeepLabV3+ [21]DeepLabV3+ [21]LR-ASPP [29]TopFormer [6]SeaFormer [31]</td><td rowspan=3 colspan=1>ENet-s16 [32]MobileNetV2-s16 [5]MobileNetV3-s16 [29]TopFormer-T [6]SeaFormer-T [31]</td><td rowspan=1 colspan=1>43.07</td><td rowspan=1 colspan=1>39.19</td><td rowspan=1 colspan=1>23.00</td></tr><tr><td rowspan=1 colspan=1>42.3438.0240.39</td><td rowspan=1 colspan=1>38.5935.0536.41</td><td rowspan=2 colspan=1>22.242.040.530.51</td></tr><tr><td rowspan=1 colspan=1>41.49</td><td rowspan=1 colspan=1>37.27</td></tr><tr><td rowspan=2 colspan=1>SiConMo (Ours)SiConMo+ (Ours)</td><td rowspan=2 colspan=1>OursOurs</td><td rowspan=1 colspan=1>41.84</td><td rowspan=1 colspan=1>37.49</td><td rowspan=2 colspan=1>0.470.49</td></tr><tr><td rowspan=1 colspan=1>41.85</td><td rowspan=1 colspan=1>37.78</td></tr></table>

## 4.2. Results PASCAL Context, Cityscapes, COCO-Stuff

As summarized in Table 2, Table 3, and Table 4, SiConMo demonstrates a consistent and favorable accuracy–efficiency trade-off across diverse semantic segmentation benchmarks. On the PASCAL Context dataset, SiConMo delivers strong mIoU results under both 59-class and 60-class evaluation settings, substantially outperforming efficient CNN-based baselines such as LR-ASPP while requiring 76.9% less GFLOPs. In comparison with lightweight Transformer-based models, SiConMo consistently attains higher accuracy with equal or lower computational cost, highlighting the effectiveness of its bottleneck-aware design in modeling both local structure and global context. On Cityscapes, SiConMo achieves competitive segmentation accuracy under strict real-time constraints, outperforming recent lightweight hybrid models such as Top-Former and SeaFormer while operating at the same computational budget. Compared to higher-capacity CNN- and Transformer-based approaches, SiConMo reduces computational cost by over two orders of magnitude while maintaining comparable performance. Similarly, on COCO-Stuff, SiConMo achieves accuracy comparable to high-capacity DeepLabV3+ variants while reducing computational over-

Table 3. Results on Cityscapes validation set [9].
<table><tr><td rowspan=1 colspan=1>Methods</td><td rowspan=1 colspan=1>Encoder</td><td rowspan=1 colspan=1>mIoU</td><td rowspan=1 colspan=1>GFLOPs</td></tr><tr><td rowspan=2 colspan=1>PSPNet [17]FCN [18]SegFormer [2]L-ASPP [21]LR-ASPP [29]LR-ASPP [29]TopFormer [6]SeaFormer [31]</td><td rowspan=2 colspan=1>MobileNetV2 [5]MobileNetV2 [5]MiT-B0 [2]MobileNetV2 [5]MobileNetV3-Large [29]MobileNetV3-Small [29]TopFormer-T [6]SeaFormer-T [31]</td><td rowspan=1 colspan=1>70.2</td><td rowspan=2 colspan=1>423.4317.117.712.69.72.91.21.2</td></tr><tr><td rowspan=1 colspan=1>61.571.972.772.468.466.566.8</td></tr><tr><td rowspan=2 colspan=1>SiConMo (Ours)SiConMo+ (Ours)</td><td rowspan=2 colspan=1>OursOurs</td><td rowspan=1 colspan=1>68.0</td><td rowspan=2 colspan=1>1.21.2</td></tr><tr><td rowspan=1 colspan=1>68.2</td></tr></table>

Table 4. Results on COCO-Stuff test set [10].
<table><tr><td>Methods</td><td>Encoder</td><td>mIoU</td><td>GFLOPs</td></tr><tr><td>PSPNet [17] DeepLabV3+ [21] DeepLabV3+ [21] LR-ASPP [29] TopFormer [6]</td><td>MobileNetV2-s8 [5] EfficientNet-s16 [22] MobileNetV2-s16 [5] MobileNetV3-s16 [29] TopFormer-T [6]</td><td>30.14 31.45 29.88 25.16 28.34</td><td>52.94 27.10 25.90 2.37</td></tr><tr><td>SeaFormer [31] SiConMo (Ours) SiConMo (Ours)</td><td>SeaFormer-T [31] Ours Ours</td><td>29.24 29.24 29.26</td><td>0.64 0.62 0.58 0.60</td></tr></table>

head by more than 97%. Under real-time settings, it matches or surpasses recent lightweight baselines, including Top-Former and SeaFormer, with lower GFLOPs. These results confirm that SiConMo generalizes across datasets with varying complexity and class diversity, offering a strong accuracy–efficiency trade-off.

## 4.3. Object Detection

To further assess the generalization capability of the proposed method, we conducted object detection experiments on the COCO dataset. COCO consists of 118K training images, 5K validation images, and 20K test images. For these experiments, we adopted RetinaNet, a one-stage detector, using SiConMo<sub>†</sub> as the backbone to construct a feature pyramid at multiple scales. All models were trained on the train2017 split for 12 epochs with ImageNet-pretrained weights and evaluated on the val2017 set. As summarized in Table 1(e) and detailed in Table 1(f), the proposed method demonstrates strong performance across standard detection metrics, indicating its robustness and effectiveness in downstream tasks. These results further suggest that the backbone learned via our approach generalizes well beyond semantic segmentation.

## 5. CONCLUSION

In this work, we showed that simplicity can be an effective alternative to architectural complexity for lightweight semantic segmentation. We introduced SiConMo, a lightweight and bottleneck-aware framework that integrates CNN and Transformer components through a streamlined design. By incorporating token pyramid extraction, hybrid local–global context modeling, and efficient feature merging, SiConMo balances segmentation accuracy and computational efficiency. Experiments on ADE20K, PASCAL Context, Cityscapes, and COCO-Stuff demonstrate that SiConMo achieves competitive or improved performance compared to more complex models while operating under strict efficiency constraints.

## References

[1] Y. Mo, Y. Wu, X. Yang, F. Liu, and Y. Liao, “Review the state-of-the-art technologies of semantic segmentation based on deep learning,” J. Neurcom., vol. 493, pp. 626– 646, 2022.

[2] E. Xie, W. Wang, Z. Yu, A. Anandkumar, J.M. Alvarez, and P. Luo, “Segformer: Simple and efficient design for semantic segmentation with transformers,” Adv. Neural Inf. Process. Syst., vol. 34, pp. 12077–12090, 2021.

[3] Y. Li, G. Yuan, Y. Wen, J. Hu, G. Evangelidis, S. Tulyakov, Y. Wang, and J. Ren, “Efficientformer: Vision transformers at mobilenet speed,” Adv. Neural Inf. Process. Syst., vol. 35, pp. 12934–12949, 2022.

[4] J. Pan, A. Bulat, F. Tan, X. Zhu, L. Dudziak, H. Li, G. Tzimiropoulos, and B. Martinez, “Edgevits: Competing light-weight cnns on mobile devices with vision transformers,” in ECCV, 2022.

[5] M. Sandler, A. Howard, M. Zhu, A. Zhmoginov, and L.C. Chen, “Mobilenetv2: Inverted residuals and linear bottlenecks,” in CVPR, 2018.

[6] W. Zhang, Z. Huang, G. Luo, T. Chen, X. Wang, W. Liu, G. Yu, and C. Shen, “Topformer: Token pyramid transformer for mobile semantic segmentation,” in CVPR, 2022.

[7] B. Zhou, H. Zhao, X. Puig, S. Fidler, A. Barriuso, and A. Torralba, “Scene parsing through ade20k dataset,” in CVPR, 2017.

[8] R. Mottaghi, X. Chen, X. Liu, N.G. Cho, S.W. Lee, S. Fidler, R. Urtasun, and A. Yuille, “The role of context for object detection and semantic segmentation in the wild,” in CVPR, 2014.

[9] M. Cordts, M. Omran, S. Ramos, T. Rehfeld, M. Enzweiler, R. Benenson, U. Franke, S. Roth, and B. Schiele, “The cityscapes dataset for semantic urban scene understanding,” in CVPR, 2016.

[10] H. Caesar, J. Uijlings, and V. Ferrari, “Coco-stuff: Thing and stuff classes in context,” in CVPR, 2018.

[11] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in CVPR, 2016.

[12] A.G. Howard, M. Zhu, B. Chen, D. Kalenichenko, W. Wang, T. Weyand, M. Andreetto, and H. Adam, “Mobilenets: Efficient convolutional neural networks for mobile vision applications,” arXiv preprint arXiv:1704.04861, 2017.

[13] N. Ma, X. Zhang, H.T. Zheng, and J. Sun, “Shufflenet v2: Practical guidelines for efficient cnn architecture design,” in ECCV, 2018.

[14] K. Han, Y. Wang, Q. Tian, J. Guo, C. Xu, and C. Xu, “Ghostnet: More features from cheap operations,” in CVPR, 2020.

[15] H. Li, P. Xiong, H. Fan, and J. Sun, “Dfanet: Deep feature aggregation for real-time semantic segmentation,” in CVPR, 2019.

[16] B. Graham, A. El-Nouby, H. Touvron, P. Stock, A. Joulin, H. Jegou, and M. Douze, “Levit: a vision´ transformer in convnet’s clothing for faster inference,” in ICCV, 2021.

[17] H. Zhao, J. Shi, X. Qi, X. Wang, and J. Jia, “Pyramid scene parsing network,” in CVPR, 2017.

[18] J. Long, E. Shelhamer, and T. Darrell, “Fully convolutional networks for semantic segmentation,” in CVPR, 2015.

[19] A. Kirillov, R. Girshick, K. He, and P. Dollar, “Panoptic´ feature pyramid networks,” in CVPR, 2019.

[20] J. Li, A. Hassani, S. Walton, and H. Shi, “Convmlp: Hierarchical convolutional mlps for vision,” in CVPR, 2023.

[21] L.C. Chen, Y. Zhu, G. Papandreou, F. Schroff, and H. Adam, “Encoder-decoder with atrous separable convolution for semantic image segmentation,” in ECCV, 2018.

[22] M. Tan and Q. Le, “Efficientnet: Rethinking model scaling for convolutional neural networks,” in ICML, 2019.

[23] N. Cavagnero, G. Rosi, C. Cuttano, F. Pistilli, M. Ciccone, G. Averta, and F. Cermelli, “Pem: Prototypebased efficient maskformer for image segmentation,” in CVPR, 2024.

[24] Y. Yuan, X. Chen, and J. Wang, “Object-contextual representations for semantic segmentation,” in ECCV, 2020.

[25] J.h. Shim, H. Yu, K. Kong, and S.J. Kang, “Feedformer: Revisiting transformer decoder for efficient semantic segmentation,” in AAAI, 2023.

[26] M.H. Guo, C.Z. Lu, Q. Hou, Z. Liu, M.M. Cheng, and S.M. Hu, “Segnext: Rethinking convolutional attention design for semantic segmentation,” Adv. Neural Inf. Process. Syst., vol. 35, pp. 1140–1156, 2022.

[27] S.K. Yeom and J. Von Klitzing, “U-mixformer: Unetlike transformer with mix-attention for efficient semantic segmentation,” in WACV, 2025, pp. 1–10.

[28] M. Ding, X. Lian, L. Yang, P. Wang, X. Jin, Z. Lu, and P. Luo, “Hr-nas: Searching efficient high-resolution neural architectures with lightweight transformers,” in CVPR, 2021.

[29] A. Howard, M. Sandler, G. Chu, L.C. Chen, B. Chen, M. Tan, W. Wang, Y. Zhu, R. Pang, V. Vasudevan, et al., “Searching for mobilenetv3,” in ICCV, 2019.

[30] M.M. Naeem Abid, N. Mehta, Z. Wu, and R. Timofte, “Lemore: Learn more details for lightweight semantic segmentation,” in ICIP, 2025, pp. 2163–2168.

[31] Q. Wan, Z. Huang, J. Lu, Y. Gang, and L. Zhang, “Seaformer: Squeeze-enhanced axial transformer for mobile semantic segmentation,” in ICLR, 2023.

[32] A. Paszke, A. Chaurasia, S. Kim, and E. Culurciello, “Enet: A deep neural network architecture for real-time semantic segmentation,” arXiv:1606.02147, 2016.

—— Supplementary Material ——

## 6. DATASETS AND MEASURES

We conduct the experiments on four benchmark datasets: ADE20K [7], PASCAL Context [8], CityScapes [9] and COCO-Stuff [10]. In total, ADE20K [7] dataset consists of 25000 images, containing 150 class categories. The division of dataset is as follows: 20000 images for training, 2000 images for validation and 3000 images for testing. The PASCAL Context [8] dataset contains 1 background, and 59 semantic labels. Out of 10103 images in total, training and testing scene images are split into 4998 and 5105, respectively. The CityScapes [9] dataset consists of 19 fine class annotations. 2975 images are taken into account for training and 500 images for validation/testing. Pixel-level stuff annotations were applied on COCO dataset for augmentation, resulting in COCO-Stuff [10] dataset. From COCO dataset, 10000 images are picked, where 9000 images are considered for training and 1000 images for testing. Following the recent literature [23, 6] we report the results using the standard common measures: Mean Intersection over Union (mIoU) for segmentation accuracy, Giga Floating Point Operations per Second (GFLOPs), latency and number of parameters.

## 7. IMPLEMENTATION DETAILS

Our implementation is based on PyTorch and MMSegmentation toolbox. All the models are first pretrained on ImageNet-1K dataset, including our proposed SiConMo model. Further, they are fine-tuned on semantic segmentation datasets. Batch-Normalization layers are used after almost each convolution layer except the last output layer. For the ADE20K dataset, we follow data augmentations same as in [2] for fair comparison. Moreover, for ADE20K dataset, we use batch size 16, and 160K scheduler by following [2] and [6]. Various augmentations have been applied such as random scaling, random cropping, random horizontal flip, random resize, etc. For the CityScapes dataset, the same data augmentations are followed as in [2, 6]. Images are resized and rescaled with same crop size i.e. $1 0 2 4 \times 5 1 2$ . It should be noted that for all datasets and models, we set $1 . 2 \times 1 0 ^ { - 4 }$ as the initial learning rate with the weight decay set to 0.01. However, the initial learning rate for the CityScapes dataset is set to $3 \times 1 0 ^ { - 4 }$ For the PASCAL Context and COCO-Stuff datasets, 80K training iterations are implemented. Additionally, for both datasets, the same data augmentations and training settings are incorporated as in MMSegmentation and the training images are resized and cropped to $5 1 2 \times 5 1 2$ and 480 × 480 for COCO-Stuff and PASCAL Context datasets, respectively.

## 8. IMAGENET PRE-TRAINING

To ensure a fair comparison, SiConMo is initialized with weights pre-trained on the ImageNet-1K dataset. As illustrated in Figure 4, the classification framework employs global average pooling followed by a linear projection to generate class scores, effectively utilizing high-level semantic representations from the backbone. For $2 2 4 \times 2 2 4$ inputs, the token resolution within the Trans-BDC block is configured to $\frac { 1 } { 6 4 } \times \frac { 1 } { 6 4 }$ of the image size. Quantitative results on ImageNet-1K are presented in Table 5, demonstrating that SiConMo attains competitive accuracy while maintaining an extremely lightweight computational footprint.

Table 5. SiConMo results for ImageNet classification.
<table><tr><td>Method</td><td>Input Size</td><td>Top-1 Accuracy(%)</td><td>GFLOPs</td><td>Parameters</td></tr><tr><td>SiConMo (Ours)</td><td> $2 2 4 \times 2 2 4$ </td><td>66.2</td><td>0.13</td><td>1.79M</td></tr><tr><td>SiConMo+ (Ours)</td><td> $2 2 4 \times 2 2 4$ </td><td>66.4</td><td>0.13</td><td>1.79M</td></tr></table>

## 9. DETAILED NETWORK STRUCTURE

This section presents the detailed architecture of the proposed SiConMo model, designed to achieve real-time semantic segmentation while maintaining a lightweight and bottleneckaware structure. The full architecture is summarized in Table 6. SiConMo strategically introduces and integrates simple yet effective modules to balance efficiency and representational power: the Token Pyramid Extraction Module (TPEM) captures multi-scale local features, the Trans-BDC block models long-range dependencies without excessive complexity, and the Feature Merging Module (FMM) adaptively combines spatial and contextual information.

In the table, dw and dw.sep indicate depth-wise and depth-wise separable convolutions, respectively, while N, H, and T denote the number of blocks, attention heads, and target channels. The table provides the architectural details of the network, based on an input resolution of $5 1 2 \times 5 1 2$ illustrating the layer properties and output resolutions at this size. This modular and bottleneck-conscious design highlights how simplicity, when carefully applied, can achieve robust feature extraction and context modeling.

Table 6. Architectural details of SiConMo model. For input resolution of $5 1 2 \times 5 1 2$
<table><tr><td rowspan="2">Stage</td><td colspan="5">Details</td><td rowspan="2">Output Resolution</td></tr><tr><td>layer</td><td>kernel size</td><td>expand ratio</td><td>output channels</td><td>stride</td></tr><tr><td>Stem</td><td>Conv MobileNetV2</td><td>3 3</td><td>1 1</td><td>16 16</td><td>2</td><td>256 × 256</td></tr><tr><td rowspan="7">TPEM</td><td>MobileNetV2</td><td>3</td><td>4</td><td>16</td><td>1 2</td><td></td></tr><tr><td>MobileNetV2</td><td>3</td><td>3</td><td>16</td><td>1</td><td>128 × 128</td></tr><tr><td>MobileNetV2</td><td>5</td><td></td><td>32</td><td></td><td></td></tr><tr><td></td><td>5</td><td>3</td><td></td><td>2</td><td></td></tr><tr><td>MobileNetV2 MobileNetV2</td><td></td><td>3</td><td>32</td><td>1</td><td>64 × 64</td></tr><tr><td>MobileNetV2</td><td>3</td><td>3</td><td>64</td><td>2</td><td></td></tr><tr><td>MobileNetV2</td><td>3 5</td><td>3</td><td>64</td><td>1</td><td>32 × 32</td></tr><tr><td></td><td>MobileNetV2</td><td>5</td><td>6 6</td><td>96 96</td><td>2 1</td><td>16 × 16</td></tr><tr><td rowspan="5">BDC Branch</td><td colspan="6">Trans-BDC Block</td></tr><tr><td>3 × 3dw</td><td>3</td><td></td><td>208</td><td>1</td><td></td></tr><tr><td> $1 \times 1 _ { d w }$ </td><td>1</td><td></td><td>208</td><td>1</td><td></td></tr><tr><td> $\underline { { 3 \times 3 _ { d w . s e p } } }$ </td><td>3,1</td><td></td><td>208</td><td>1</td><td></td></tr><tr><td colspan="5">N=4</td><td>8×8</td></tr><tr><td>FMM</td><td colspan="5">N=4, H=4</td><td>8× 8</td></tr><tr><td></td><td colspan="5">T=160</td><td>82, 162, 322, 642</td></tr><tr><td>GFLOPs</td><td colspan="5"></td><td>0.6</td></tr></table>

![](images/f6bdea8651992c202e1607351d92ed4206330057976937cc7f9154a63342f9d0.jpg)  
Fig. 4. The architecture of the proposed SiConMo model for the task of image classification.

## 10. DETAILS OF FEED-FORWARD NETWORK

In SiConMo, the unified features from the BDC branch and Lightweight ViT branch are processed through a lightweight Feed-Forward Network (FFN) to enhance the representation while preserving efficiency. The FFN integrates a depth-wise convolution between two 1 × 1 convolution layers, enabling effective local feature refinement and capturing contextual information without incurring substantial computational overhead. An expansion factor of two is used to balance the expressiveness of the features with the lightweight design, complementing the bottleneck-aware Trans-BDC block. This carefully designed FFN contributes to SiConMo’s ability to model global and local semantics efficiently, reinforcing the overall simplicity and robustness of the network. The architecture of the Feed-Forward Network is illustrated in Figure 5.

![](images/eb35b4171ea50499179e8daaf3ff424dcddae0286a8ca7c989b428d804b8715b.jpg)  
Fig. 5. Design of Feed-Forward Network.

## 11. GRADIENT MAGNITUDE AND EDGE MAP (GME) GENERATION

For the enhanced SiConMo variant, we augment the RGB input with Gradient Magnitude and Edge Maps (GME) to improve structural awareness.

Given an RGB input image $X \in \mathbb { R } ^ { H \times W \times 3 }$ , we first convert it to grayscale using the standard luminance formulation:

$$
X _ { g r a y } = 0 . 2 9 8 9 R + 0 . 5 8 7 0 G + 0 . 1 1 4 0 B .\tag{7}
$$

We then compute horizontal and vertical gradients using Sobel operators:

$$
S _ { x } = \left[ { \begin{array} { c c c } { - 1 } & { 0 } & { 1 } \\ { - 2 } & { 0 } & { 2 } \\ { - 1 } & { 0 } & { 1 } \end{array} } \right] , \qquad S _ { y } = \left[ { \begin{array} { c c c } { - 1 } & { - 2 } & { - 1 } \\ { 0 } & { 0 } & { 0 } \\ { 1 } & { 2 } & { 1 } \end{array} } \right] .\tag{8}
$$

The horizontal and vertical gradients are obtained via convolution:

$$
G _ { x } = X _ { g r a y } * S _ { x } , \qquad G _ { y } = X _ { g r a y } * S _ { y } ,\tag{9}
$$

where ∗ denotes convolution.

The gradient magnitude map is computed as:

$$
G _ { m } = \sqrt { G _ { x } ^ { 2 } + G _ { y } ^ { 2 } } .\tag{10}
$$

An edge map is further generated using mean-based thresholding:

$$
E = { \left\{ \begin{array} { l l } { 1 , } & { G _ { m } > \mu ( G _ { m } ) } \\ { 0 , } & { { \mathrm { o t h e r w i s e } } , } \end{array} \right. }\tag{11}
$$

where $\mu ( G _ { m } )$ denotes the mean gradient magnitude.

![](images/cdfc9d01ffb9018a40fe888f958893a0f571dc8265f3d1f18eadf7ae2184e606.jpg)  
Fig. 6. Visual results on the ADE20K validation set. Our proposed model illustrates superior consistency with the ground truth segmentation results, highlighting its robustness.

Finally, the gradient magnitude and edge maps are concatenated with the RGB image to form a five-channel input representation:

$$
X ^ { \prime } = [ R , G , B , G _ { m } , E ] .\tag{12}
$$

No additional normalization is applied to the generated GME channels beyond the standard RGB preprocessing. For SiConMo<sub>†</sub>, reported latency includes the computational cost of GME generation, since Sobel gradient magnitude and edge maps are computed within the model forward pass during inference.

## 12. VISUAL RESULTS

Figure 6 presents additional qualitative results of the proposed SiConMo model on the ADE20K validation set. For comparison, we show the original images, ground-truth annotations, predictions from TopFormer, and outputs from SiConMo<sub>†</sub>. The visualizations highlight that SiConMo produces more consistent and accurate segmentation, effectively capturing both fine-grained details and global context. These results underscore the robustness of our lightweight, bottleneck-aware design and demonstrate that careful architectural simplicity can outperform more complex models in real-world segmentation scenarios.

## 13. LIMITATIONS AND FUTURE WORK

While the proposed SiConMo model demonstrates strong performance and efficiency for lightweight semantic segmentation, several limitations remain. A primary constraint, shared with many lightweight architectures, is its reliance on ImageNet-1K pre-training; omitting this step results in a noticeable drop in segmentation accuracy. Additionally, although the model is designed for efficiency, extreme edge cases with highly complex scenes or unusual object distributions may still challenge its predictive capabilities.

Future work will focus on enhancing the robustness and adaptability of SiConMo across diverse datasets and deployment scenarios, without compromising its lightweight design. We also aim to explore techniques for further reducing pretraining dependency, improving generalization to unseen domains, and extending the model for practical real-world applications, where balancing simplicity, accuracy, and efficiency remains crucial. These directions aim to position SiConMo as a flexible and practical framework for both research and applied vision tasks.