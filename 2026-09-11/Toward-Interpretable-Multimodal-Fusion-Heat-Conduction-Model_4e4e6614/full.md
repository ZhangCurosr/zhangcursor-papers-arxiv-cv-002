# Toward Interpretable Multimodal Fusion: Heat Conduction Modeling for Hyperspectral and LiDAR Joint Classification

Kan Wei, Jiahui Cui, Jing Yao, Senior Member, IEEE, Xinyu Zhao, Lei Wang, Member, IEEE, and Pedram Ghamisi, Senior Member, IEEE

Abstract—The fusion of hyperspectral (HS) and Light Detection and Ranging (LiDAR) data plays a crucial role in enhancing land-cover classification by jointly exploiting spectral, spatial, and structural cues. However, existing multimodal fusion methods still struggle to model long-range dependencies and complex anisotropic interactions while maintaining computational efficiency. This paper introduces M2Heat, a physics-inspired frame work that investigates multimodal fusion through the lens of heat conduction. At its core, a physics-driven visual heat conduction module (vHeat) and enhanced Frequency Value Embeddings (FVEs) simulate anisotropic information flow, enabling the capture of global dependencies with sub-quadratic complexity and physical interpretability. This mechanism, combined with a hybrid spatialfrequency fusion strategy named Cross-Frequency Fusion (CFF) module, produces highly discriminative and robust feature representations. M2Heat achieves competitive overall performance on three benchmarks, i.e., Trento, Houston2013, and Augsburg, while providing an interpretable heat-conduction-guided perspective for multimodal feature fusion. These results indicate the potential of heat-conduction-guided neural operators for efficient and interpretable RS multimodal fusion. The source code is publicly available at https:/github.com/Weikan0425/M2Heat HSI LiDAR.

Index Terms—Deep learning, Hyperspectral, LiDAR, Multimodal, Remote sensing image classification, Heat conduction.

## I. INTRODUCTION

N recent years, Earth observation (EO) technologies have I played an increasingly vital role in monitoring land surface dynamics over large areas and in near real-time [1], [2]. Among the various remote sensing modalities, hyperspectral images (HSI) have emerged as a particularly powerful tool due to their exceptionally high spectral resolution [3]. Compared to conventional optical sensors such as RGB or multispectral images (MSI), HSI captures hundreds of contiguous spectral bands, enabling detailed characterization of materials based on their unique spectral signatures. This capability has been widely demonstrated to be effective in a broad range of applications, including land cover mapping [4], [5], vegetation analysis [6], [7], ecological monitoring [8], [9], and mineral exploration [10].

![](images/f31d7c77df5c84856f3185c6a8cc5a67758c29af83d1691255ed8b98908bc7e0.jpg)  
Y = Conv(X) ∼ O(HWk2)

![](images/b9c4f10c52dfd209999309eebef3bb636a4448efb69f96d7e8b0becaa88807d7.jpg)  
softmax(QKT)V ∼ O(N2d)  
(b) Self-Attention

(a) Convolution  
![](images/1730581f1b7b8cf452f36ee3179316958a90903ea4451724ecce6d245c5ec9a0.jpg)  
yt = ∑Kk · ut−k ∼ O(Nd)

![](images/e270cc908915253345ba4629587d11bcca61f6cf8a91d0f287232b1319fc1cf6.jpg)  
(c) Vision Mamba  
F−1(F(x) · e−k(ω2)t) ∼ O(N 1og N)  
(d) Heat Conduction  
Fig. 1. Comparison of mainstream modeling mechanisms. (a) Convolution. (b) Self-Attention. (c) Vision Mamba. (d) Heat Conduction.

Recent advances in deep learning (DL) have substantially improved hyperspectral image (HSI) classification by enabling more effective modeling of spatial and spectral dependencies [11], [12]. Convolutional neural networks (CNNs) first demonstrated strong capabilities in extracting local spatial–spectral patterns [13], [14], while transformer-based architectures further enhanced global context modeling through self-attention mechanism [15], [16]. More recently, statespace models such as Mamba have introduced efficient longrange dependency modeling [17], [18], further extending the representational power of DL-based HSI methods.

Nevertheless, single-modality approaches still face challenges in complex real-world environments with diverse land covers, shadow effects, or structural occlusions [19]. To address these limitations, multimodal remote sensing (RS) has emerged as a promising paradigm by integrating complementary information from heterogeneous sensors such as HSI and LiDAR, which respectively provide fine spectral and rich structural cues [20], [21].

Driven by the progress of multimodal DL, a variety of fusion-based architectures have been proposed to jointly exploit spectral and geometric representations [22]. While early works relied on simple concatenation or shared-branch designs, modern methods explore adaptive attention-based [23] and hybrid fusion frameworks [24], [25]. Despite these advances, achieving robust and well-aligned multimodal fusion remains challenging due to modality heterogeneity and scale disparities. This motivates the development of a unified, dynamically adaptive fusion framework for multimodal RS classification.

To address the aforementioned limitations, this work explores two complementary directions: developing an interpretable global modeling mechanism and designing an effective multi modal fusion framework. As shown in Fig. 1, convolutional operations are efficient for local representation but struggle to capture long-range dependencies, whereas self-attention provides global context modeling at the cost of quadratic complexity. Recently, state-space models such as Mamba [26], [27] have achieved linear-complexity global modeling, yet their interpretability remains limited for structured remote sensing data.

Motivated by this gap, we introduce a physics-inspired framework, termed M2Heat, which performs global subquadratic and interpretable modeling through a heat conduction operator (HCO) that simulates information diffusion across spatial–spectral dimensions. Building upon the vHeat paradigm [28], M2Heat extends heat-based modeling to the multimodal hyperspectral–LiDAR domain by predicting learnable thermal diffusivity rates via Frequency Value Embeddings (FVEs). Furthermore, we design a Cross-Frequency Fusion (CFF) module that aligns and fuses heterogeneous spectral and geometric representations in the frequency domain. Together, these designs establish a unified framework that jointly achieves efficient global modeling and semantically consistent multimodal fusion for remote sensing classification.

The main contributions can be summarized as follows:

1) A physics-driven multimodal framework, termed M2Heat, is established to achieve efficient HSI–LiDAR joint classification. At its core lies a cross-modal Fourier fusion mechanism that integrates modality-specific and shared representations within a unified spectral–spatial domain. By embedding diffusion dynamics into deep representations, this framework models long-range spatial–spectral dependencies with sub-quadratic complexity, thereby achieving a principled balance between interpretability and computational efficiency.

2) Central to the design is the vHeat module equipped with FVEs, which reformulates anisotropic heat propagation in the frequency domain, where FVEs act as physics-guided priors regulating the diffusion behavior of spectral–spatial features. Such formulation enhances discriminative feature learning and improves model generalization across heterogeneous modalities.

3) We introduce a hybrid cross-domain fusion mechanism, combining alignment-aware interaction in the spatial domain with spectral energy redistribution in the frequency domain. Through this dual-phase integration, complementary modality-specific cues and shared semantic information are adaptively aligned and fused, yielding a structurally consistent multimodal representation.

4) Extensive evaluations conducted on three public benchmarks, i.e., Augsburg, Houston2013, and Trento, demonstrate the competitive performance and practical effectiveness of M2Heat. The ablation studies validate the effectiveness of FVEs and the hybrid fusion strategy. This study explores the potential of interpretable heat-diffusion dynamics for HSI–LiDAR classification, bridging physical modeling and deep multimodal representation learning.

## II. RELATED WORK

A. Feature Extraction and Representation Learning: From Local to Global

CNN-based Models: Convolutional Neural Networks (CNNs) have long served as the foundation for HSI classification due to their ability to hierarchically capture local spatial–spectral structures. By stacking convolutional layers, CNNs progressively extract discriminative features that effectively characterize land-cover patterns. To extend CNNs to multimodal scenarios, various studies have explored joint exploitation of HSI spectral and LiDAR structural information. Representative examples include the 3D CNN-based H+L framework [29], the multiscale MSNetSC [30], and the tripletbased TSDN [31]. In addition, single-stream CNNs [32] have been proposed to jointly learn modality-shared features without explicit separation. More recently, DHNet [33] constructed spatial-specific and spectral-specific heterogeneous branches with feature calibration to enhance complementary spectral– spatial representation learning for HSI classification.

Despite their effectiveness in local representation, CNNbased methods are inherently constrained by fixed receptive fields, limiting their capacity to model long-range dependencies critical for multimodal understanding.

Transformer-based Models: Transformers have recently gained prominence in HSI analysis by effectively modeling global contextual dependencies [34]. Their extension to multimodal fusion has led to architectures such as DHViT [35] and MFT [36], which employ cross-attention for inter-modal correlation learning. Hybrid networks such as GLT-Net [37] further integrate CNNs and Transformers to balance local and global modeling, while efficient variants like ExViT [38] and SWFormer [39] reduce computational cost through lightweight attention designs. Nevertheless, Transformer-based models remain computationally expensive and often lack modality awareness, motivating research into more efficient and interpretable multimodal modeling approaches.

Mamba-based Models: Recently, State Space Models (SSMs) have emerged as efficient alternatives to self-attention for long-sequence modeling. The Mamba architecture, in particular, achieves linear complexity while retaining strong global modeling ability [40]. Its adaptation to multimodal RS tasks has shown promising results, such as HLMamba [41] and S<sup>2</sup>CrossMamba [42], which enhance semantic representation and cross-modal feature fusion. Subsequent works like SSFN [43] and M2FMNet [44] further unify spatial and spectral dependencies under Mamba-based frameworks. Despite their efficiency, current Mamba-based methods rely on sequential tokenization of 2D images, which weakens spatial interpretability and motivates the exploration of more physically grounded modeling paradigms.

Physics-inspired Models: Incorporating physical priors such as thermodynamics and diffusion processes into deep learning models has proven effective for improving interpretability and structural consistency. These priors introduce inductive constraints that align model behavior with real-world physical principles, which is particularly valuable for RS applications involving energy transport and material interactions.

Recent studies have explored physics-guided paradigms to enhance multimodal feature alignment and semantic consistency. For example, Yu et al. [24] integrated physically constrained embeddings into HSI–LiDAR fusion, while Li et al. [45] proposed a diffusion-based framework inspired by nonequilibrium thermodynamics. Wang et al. [28] further introduced vHeat, a heat-diffusion-based model achieving global context modeling with sub-quadratic complexity, and Hu et al. [46] extended it to RS-vHeat for spatial diffusion in optical and SAR imagery. From the broader perspective of interpretable multimodal representation, UAAFusion [47] formulated multimodal image fusion as an attribution-analysis guided deep unfolding process to improve task-aware interpretability. FAMAFuse [48] introduced a functional–anatomical multiscale attention mechanism to adaptively balance modality specific details and global contextual information in multimodal image fusion.

In diffusion-process-inspired modeling, diffusion models typically formulate diffusion as an iterative generative or refinement process, whereas vHeat and RS-vHeat discretize heat conduction as a deterministic feature-propagation operator for visual and RS representation learning. RS-vHeat further demonstrates the potential of heat-conduction modeling in optical and SAR-oriented remote sensing scenarios. For HSI–LiDAR joint classification, heterogeneous spectral signatures and LiDARderived geometric structures require coupled spectral–spatial– structural propagation and cross-modal alignment. Accordingly, M2Heat refines heat-conduction-guided modeling to the HSI– LiDAR fusion scenario by introducing modality-specific and shared HCO branches with cross-frequency fusion, aiming to achieve interpretable and efficient multimodal representation learning.

## B. Multimodal Alignment and Feature Fusion: From Shallow Interaction to Hybrid Integration

The integration of heterogeneous modalities, such as HSI and LiDAR, has proven highly effective for enhancing scene understanding in RS. However, challenges persist due to significant disparities in spatial resolution, data distribution, and semantic structure across modalities. To address these, early approaches focused on feature alignment and shallow interaction. For instance, Hong et al. [49] proposed SM-GANs, introducing an early cross-fusion strategy to capture complementary cues in urban scenes. Hang et al. [50] designed a cross-modality contrastive learning method that enables unsupervised feature alignment without labeled data. CCR-Net [51] further unified alignment and fusion via a cross-channel reconstruction module, enhancing inter-modality correlation.

Recent efforts have shifted toward hierarchical and hybrid fusion strategies to capture deeper semantic dependencies. Flex-MCFNet [52] sequentially applies information complement and global fusion modules for stage-wise integration. PID-HLfusion [53] leverages parameter sharing and geometric priors to achieve progressive modality fusion. CCEnd-Net [54] adopts a cascaded encoder–decoder design for early-to-late fusion, while DCMNet [55] employs multi-level routing spaces to dynamically align and integrate spatial–spectral features. MIViT [56] introduced an information aggregation–distribution mechanism to enhance complementary feature interaction while reducing redundant multimodal information for HSI–LiDAR classification. S3F2Net [57] developed a spatial–spectral– structural feature fusion framework to jointly exploit HSI spectral–spatial characteristics and LiDAR-derived structural cues. In parallel, Qin et al. [58] explored fractional Fourier transforms and optimal matching flows for effective crossmodal interaction.

Together, these advances underscore a clear trend toward unified, adaptive, and context-aware frameworks capable of bridging modality gaps through deeper and more flexible interactions. Nonetheless, further exploration is required to fully exploit the complementary information of heterogeneous data.

## III. METHODOLOGY

## A. Overall Framework of M2Heat

As illustrated in Fig. 2, the proposed M2Heat framework is a physically inspired multimodal architecture designed to jointly model HSI and LiDAR data. The framework aims to simulate the thermal diffusion process in a visual–spectral context, thereby achieving efficient cross-modal information propagation and structurally consistent feature alignment. Specifically, given a pair of co-registered HSI and LiDAR patches, M2Heat first extracts shallow modality representations via Stem Layers, followed by modality-specific Heat Conduction Operator (HCO) Layers that perform physics-inspired spatial–spectral diffusion. Subsequently, a shared HCO Layer is used for cross-modal alignment, ensuring that heterogeneous representations are projected into a unified latent space. A Cross-Frequency Fusion (CFF) module then facilitates multi-frequency information interaction, and a lightweight classifier produces the final pixel-wise semantic predictions. The overall framework couples physical diffusion dynamics with deep feature learning, offering an interpretable multimodal feature-learning framework for HSI–LiDAR classification.

## B. Multimodal Feature Extraction of M2Heat

1) Physical and Visual Motivation: The core design of M2Heat is grounded in the classical heat conduction equation, which describes the spatio-temporal evolution of temperature in a continuous medium. Let $u ( x , y , t )$ denote the temperature distribution at position $( x , y )$ and diffusion time t over a 2D domain $\mathbf { D } \subset \mathbb { R } ^ { 2 }$ . The fundamental governing equation can be expressed as:

$$
{ \frac { \partial u } { \partial t } } = k \left( { \frac { \partial ^ { 2 } u } { \partial x ^ { 2 } } } + { \frac { \partial ^ { 2 } u } { \partial y ^ { 2 } } } \right) ,\tag{1}
$$

where $k > 0$ represents the thermal diffusivity coefficient controlling the rate of energy transfer. The analytical solution under initial condition $u ( x , y , 0 ) = f ( x , y )$ can be derived via the 2D Fourier Transform:

$$
\tilde { u } ( \omega _ { x } , \omega _ { y } , t ) = \tilde { f } ( \omega _ { x } , \omega _ { y } ) e ^ { - k ( \omega _ { x } ^ { 2 } + \omega _ { y } ^ { 2 } ) t } ,\tag{2}
$$

which reveals that each frequency component $\tilde { f } ( \omega _ { x } , \omega _ { y } )$ is exponentially attenuated according to its spatial frequency magnitude. This behavior implies that low-frequency components representing global smooth structures are preserved, while high-frequency details gradually decay, forming a physically grounded low-pass diffusion effect. The heat conduction process thus naturally provides a mathematical model for information propagation, structural smoothing, and multi-scale feature evolution.

![](images/b1041b2f3b2561da85c3d612c936541abba657ddf94a9ce71a55ac8468023af1.jpg)  
Fig. 2. Overall framework of the proposed M2Heat, which consists of Stem Layer, Modality-specific and Modality-shared HCO Layer, CFF module, and classifier.

Motivated by this principle, we reinterpret visual feature maps as temperature fields, where each channel represents an energy distribution evolving through a learnable diffusion process. The evolution of a feature field $U ( x , y , c , t )$ can be approximated as:

$$
U _ { t } = \mathrm { I D C T } _ { 2 D } \left( \mathrm { D C T } _ { 2 D } ( U _ { 0 } ) \cdot e ^ { - k ( \omega _ { x } ^ { 2 } + \omega _ { y } ^ { 2 } ) t } \right) ,\tag{3}
$$

where $\mathrm { D C T } _ { 2 D }$ and $\mathrm { I D C T } _ { 2 D }$ denote the 2D Discrete Cosine Transform and its inverse, respectively. Compared to the traditional Fourier transform, DCT naturally satisfies Neumann (reflective) boundary conditions, avoiding boundary artifacts and making it well-suited for finite image patches. This formulation enables M2Heat to implement heat diffusion efficiently in the spectral domain, achieving a balance between local smoothness and global consistency. The computational complexity of this diffusion process is approximately $\mathcal { O } ( N ^ { 1 . 5 } )$ , ensuring scalability for large-scale RS data.

2) Heat Conduction Operator Formulation: Building upon this physical foundation, we propose the HCO as a differentiable module that embeds the heat diffusion equation into neural feature learning. Given two input modalities, $X _ { 1 }$ (HSI) and $X _ { 2 }$ (LiDAR), the features are first projected into a shared latent space through modality-specific Stem Layers:

$$
U _ { i } = \mathbf { B } \mathbf { N } ( \mathbf { C o n v } ( \mathbf { G E L U ( B N ( C o n v ( } X _ { i } ) ) ) ) ) , \quad i \in \{ 1 , 2 \} ,\tag{4}
$$

resulting in $U _ { i } \ \in \ \mathbb { R } ^ { P \times P \times 2 C }$ . To enable separate and joint modeling, each $U _ { i }$ is divided along the channel dimension:

$$
[ u _ { i } ^ { 1 } , u _ { i } ^ { 2 } ] = \mathrm { S p l i t } ( U _ { i } ) , \quad u _ { i } ^ { j } \in \mathbb { R } ^ { P \times P \times C } ,\tag{5}
$$

where $u _ { i } ^ { 1 }$ is processed by modality-specific HCO Layers to extract discriminative spatial–spectral cues, while $u _ { i } ^ { 2 }$ participates in shared HCO-based alignment.

The modality-specific and shared diffusions are formulated as:

$$
\begin{array} { r l } & { U _ { i } ^ { t } = \mathrm { H C O } _ { i } ( u _ { i } ^ { 1 } , k _ { q = i } ) , \quad i \in \{ 1 , 2 \} , } \\ & { U _ { 3 } ^ { t } = \mathrm { H C O } ( \mathrm { C o n c a t } ( u _ { 1 } ^ { 2 } , u _ { 2 } ^ { 2 } ) , k _ { q = 3 } ) , } \end{array}\tag{6}
$$

where $k _ { q }$ denotes the effective diffusion coefficient used in the q-th HCO branch. To make the heat-conduction process learnable while preserving the physical constraint on diffusivity, we introduce a latent FVE $\theta _ { q }$ and map it to $k _ { q }$ through a learnable linear transformation followed by a ReLU activation:

$$
k _ { q } = \mathrm { R e L U } ( W _ { k } \theta _ { q } + b _ { k } ) .\tag{7}
$$

Here, $\theta _ { q }$ provides a flexible learnable representation for adapting the diffusion behavior to different modalities and scenes, while the non-negative mapping ensures that the effective diffusivity used in HCO remains consistent with the forward heat-conduction formulation. Depending on the FVE configuration, $k _ { q }$ can further vary across spatial positions and channels, enabling adaptive diffusion for heterogeneous HSI and LiDAR features.

3) HCO Architecture and Theoretical Properties: Each HCO Layer adopts a “LN–Operator–LN–FFN” structure, inspired by implicit-state models such as Mamba and transformer-style visual operators, but with physically interpretable dynamics. Formally, for an input feature tensor $u \in \mathbb { R } ^ { P \times P \times C }$

$$
\begin{array} { r l } & { z _ { 1 } = \mathbf { L N } ( u ) , } \\ & { z _ { 2 } = \mathbf { D W } \mathbf { C o n v } ( z _ { 1 } ) , } \\ & { z _ { 3 } = \sigma ( \mathbf { L i n e a r } _ { 1 } ( z _ { 2 } ) ) \odot \mathbf { H C O ( L i n e a r } _ { 2 } ( z _ { 2 } ) , k _ { q } ) , } \end{array}\tag{8}
$$

where DWConv captures local spatial dependencies and $\sigma ( \cdot )$ denotes the SiLU activation. The two parallel linear projections serve as gating and conduction paths, respectively, with elementwise modulation ⊙ implementing adaptive diffusion control.

![](images/e62352cb326a9f51fc245b5c38581bb9e860122894522b4a04607ab8c5c0ceda.jpg)  
Fig. 3. Our proposed CFF module, which integrates fusion in the frequency domain and dynamic fusion with jointly learned features, enabling effective multimodal feature interaction and fusion.

The output is then normalized and updated through a residual and feed-forward structure:

$$
z _ { 4 } = \mathrm { L N } ( z _ { 1 } + z _ { 3 } ) , \quad U _ { i } ^ { t } = z _ { 4 } + \mathrm { F F N } ( z _ { 4 } ) .\tag{9}
$$

This design allows the network to explicitly learn anisotropic, directionally varying diffusion patterns that reflect the geometry and texture structure of the input. Theoretically, $\mathrm { H C O } ( \cdot , k _ { q } )$ explicitly decouples physical diffusion from learnable refinement: the DCT-domain attenuation solves the discretized heat equation via Laplacian eigenmode modulation, while the auxiliary neural components function as data-adaptive calibration units.

$$
\boldsymbol { U } ^ { ( t + 1 ) } = \boldsymbol { U } ^ { ( t ) } + k \nabla ^ { 2 } \boldsymbol { U } ^ { ( t ) } ,
$$

where the Laplacian term $\nabla ^ { 2 } U$ is implicitly realized in the frequency domain through DCT filtering. The learnable $k _ { q }$ thus plays the role of a neural thermal diffusivity tensor, dynamically controlling the propagation of information across both space and channels. Unlike conventional gated blocks that rely entirely on data-driven routing, HCO anchors its computational core in an explicit heat-diffusion frequency response; the gating branch serves strictly to modulate this physical flow for heterogeneous HSI–LiDAR features.

By embedding this physically consistent process into multimodal representation learning, the HCO effectively bridges physics-based diffusion theory and deep spectral–spatial modeling. It achieves interpretable information propagation across modalities, providing enhanced robustness, generalization, and physical plausibility compared with purely data-driven fusion mechanisms.

## C. Multimodal Feature Fusion of M2Heat

To capture complementary structures and semantic cues from hyperspectral and LiDAR modalities, M2Heat introduces a CFF module that explicitly operates in the frequency domain. This approach leverages both amplitude- and phase-based representations to achieve fine-grained cross-modal alignment and complementary feature enhancement.

1) Frequency-domain decomposition: Each input feature $U _ { i } ^ { t } \in \mathbb { R } ^ { H \times W \times \check { C } }$ is first transformed into the frequency domain using the 2D Fast Fourier Transform (FFT):

$$
\mathcal { F } _ { i } = \mathrm { F F T } ( U _ { i } ^ { t } ) , \quad i \in \{ 1 , 2 \} , \quad \mathcal { F } _ { i } \in { \mathbb C } ^ { H \times W \times C } .\tag{10}
$$

We explicitly decompose ${ \mathcal { F } } _ { i }$ into amplitude and phase components:

$$
\begin{array} { r } { \mathcal { A } _ { i } = | \mathcal { F } _ { i } | , \quad \mathcal { P } _ { i } = \angle \mathcal { F } _ { i } , } \end{array}\tag{11}
$$

where $\mathcal { A } _ { i } \in \mathbb { R } ^ { H \times W \times C }$ encodes the energy distribution across spatial frequencies, and $\mathcal { P } _ { i } \in [ - \pi , \pi ] ^ { H \times \mathbf { \breve { W } } \times C }$ preserves highfrequency structural details. This decomposition allows separate modeling of low-frequency semantic structures and high frequency geometric details.

2) Cross-modal frequency fusion: To align modalities in the spectral domain, the corresponding amplitude and phase components are concatenated and processed through a shared 2D MLP (implemented via convolution-GELU-convolution), enabling joint frequency-aware feature learning:

$$
\begin{array} { r } { \hat { \mathcal { P } } = \mathbf { M L P } _ { 2 D } \big ( \mathbf { C o n c a t } ( \mathcal { P } _ { 1 } , \mathcal { P } _ { 2 } ) \big ) , } \\ { \hat { \mathcal { A } } = \mathbf { M L P } _ { 2 D } \big ( \mathbf { C o n c a t } ( \mathcal { A } _ { 1 } , \mathcal { A } _ { 2 } ) \big ) . } \end{array}\tag{12}
$$

This operation can be interpreted as learning a frequencydomain transfer function $T : \left( { A _ { 1 } , A _ { 2 } , { \mathcal { P } } _ { 1 } , { \mathcal { P } } _ { 2 } } \right) \overset { - } { \mapsto } \left( \hat { A } , \hat { \mathcal { P } } \right)$ that maximizes inter-modality mutual information while preserving intra-modality spectral energy.

3) Spatial reconstruction: The fused frequency representations are transformed back to the spatial domain using the inverse FFT:

$$
\hat { U } _ { f } ^ { t } = \mathrm { I F F T } ( \hat { A } \odot e ^ { j \hat { \mathcal { P } } } ) ,\tag{13}
$$

where $\odot$ denotes element-wise multiplication and $j$ is the imaginary unit. This step reconstructs spatial features that integrate complementary low- and high-frequency information from both modalities.

4) Weighted integration with HCO features: To exploit additional joint correlations, the frequency-fused feature $\hat { U } _ { f } ^ { t }$ is combined with the HCO-aligned feature $U _ { 3 } ^ { t }$ via learnable weights α and $\beta \colon$

$$
\begin{array} { r } { \hat { U } ^ { t } = \alpha \cdot \mathbf { C o n v } _ { 1 \times 1 } ( U _ { 3 } ^ { t } ) + \beta \cdot \mathbf { C o n v } _ { 1 \times 1 } ( \hat { U } _ { f } ^ { t } ) , } \end{array}\tag{14}
$$

TABLE I  
DETAILED CATEGORY INFORMATION FOR THREE DATASETS
<table><tr><td rowspan="2">No.</td><td colspan="3">Trento</td><td colspan="3">Houston2013</td><td colspan="3">Augsburg</td></tr><tr><td>Class name</td><td>Train</td><td>Test</td><td>Class name</td><td>Train</td><td>Test</td><td>Class name</td><td>Train</td><td>Test</td></tr><tr><td>C1</td><td>Apple Trees</td><td>20</td><td>4014</td><td>Healthy grass</td><td>198</td><td>1053</td><td>Forest</td><td>146</td><td>13361</td></tr><tr><td>C2</td><td>Buildings</td><td>20</td><td>2883</td><td>Stressed grass</td><td>190</td><td>1064</td><td>Residential Area</td><td>264</td><td>30065</td></tr><tr><td>C3</td><td>Ground</td><td>20</td><td>459</td><td>Synthetic grass</td><td>188</td><td>505</td><td>Industrial Area</td><td>21</td><td>3830</td></tr><tr><td>C4</td><td>Woods</td><td>20</td><td>9103</td><td>Trees</td><td>196</td><td>1072</td><td>Low Plants</td><td>248</td><td>26609</td></tr><tr><td>C5</td><td>Vineyard</td><td>20</td><td>10481</td><td>Soil</td><td>186</td><td>1056</td><td>Allotment</td><td>52</td><td>523</td></tr><tr><td>C6</td><td>Roads</td><td>20</td><td>3154</td><td>Water</td><td>182</td><td>143</td><td>Commercial Area</td><td>7</td><td>1638</td></tr><tr><td>C7</td><td></td><td></td><td></td><td>Residential</td><td>196</td><td>1072</td><td>Water</td><td>23</td><td>1507</td></tr><tr><td>C8 C9</td><td></td><td></td><td></td><td>Commercial</td><td>191</td><td>1053</td><td></td><td></td><td></td></tr><tr><td>C10</td><td></td><td></td><td></td><td>Road</td><td>193</td><td>1059</td><td></td><td></td><td></td></tr><tr><td>C11</td><td></td><td></td><td></td><td>Highway</td><td>191</td><td>1036</td><td></td><td></td><td></td></tr><tr><td>C12</td><td></td><td></td><td></td><td>Railway</td><td>181</td><td>1054</td><td></td><td></td><td></td></tr><tr><td>C13</td><td></td><td></td><td></td><td>Parking Lot 1</td><td>192</td><td>1041</td><td></td><td></td><td></td></tr><tr><td>C14</td><td></td><td></td><td></td><td>Parking Lot 2</td><td>184</td><td>285</td><td></td><td></td><td></td></tr><tr><td>C15</td><td></td><td></td><td></td><td>Tennis Court</td><td>181</td><td>247</td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td>Running Track</td><td>187</td><td>473</td><td></td><td></td><td></td></tr><tr><td></td><td>Total</td><td>120</td><td>30094</td><td>Total</td><td>2832</td><td>12197</td><td>Total</td><td>761</td><td>77533</td></tr></table>

where the 1 × 1 convolution both aligns the channel dimension and introduces nonlinear refinement. The learnable parameters $\alpha , \beta \geq 0$ enforce an implicit energy-preserving constraint:  
where K denotes the number of classes.

$$
\begin{array} { r l } & { \| \hat { \boldsymbol U } ^ { t } \| _ { F } ^ { 2 } \approx \alpha ^ { 2 } \| \mathbf { C o n v } ( U _ { 3 } ^ { t } ) \| _ { F } ^ { 2 } + \beta ^ { 2 } \| \mathbf { C o n v } ( \hat { U } _ { f } ^ { t } ) \| _ { F } ^ { 2 } } \\ & { \qquad + 2 \alpha \beta \langle \mathbf { C o n v } ( U _ { 3 } ^ { t } ) , \mathbf { C o n v } ( \hat { U } _ { f } ^ { t } ) \rangle , } \end{array}\tag{15}
$$

ensuring that the fused representation balances contributions from both spatial- and frequency-domain features.

The proposed CFF module constitutes a hybrid fusion mechanism, explicitly leveraging frequency-domain amplitudephase decomposition and spatial-domain correlations. It should be noted that DCT and FFT serve different purposes in M2Heat. The HCO adopts DCT because its real-valued cosine basis naturally supports reflective boundary conditions and enables boundary-consistent heat diffusion over finite feature patches. In contrast, CFF employs FFT to obtain a complex-valued spectrum, allowing the amplitude and phase components to be explicitly separated and jointly interacted across modalities. By integrating cross-modal spectral interactions and energyaware weighting, M2Heat achieves a theoretically grounded and empirically effective representation for joint classification tasks.

## D. Classification Mapping of M2Heat

To generate semantic predictions from the fused features, M2Heat employs a lightweight yet effective classification head that leverages both spatially aligned and modality-aware information. Specifically, the fused feature $\hat { U } ^ { t } \in \mathbb { R } ^ { H \times } \mathbf { \bar { W } } \times C$ is first processed by $\mathrm { ~ \iota ~ } 1 \times 1$ convolution for channel reduction, followed by batch normalization (BN), LeakyReLU activation, and global average pooling (GAP) to capture global contextual information:

$$
z = \mathbf { G A P } \Big ( \phi \big ( \mathbf { B N } ( \mathbf { C o n v } _ { 1 \times 1 } ( \hat { U } ^ { t } ) ) \big ) \Big ) ,\tag{16}
$$

where $\phi ( \cdot )$ denotes the LeakyReLU activation function and $z \in \mathbb { R } ^ { C }$ is the compact feature vector for each input sample.

The resulting feature vector is then projected into the class space via a second $1 \times 1$ convolution followed by a softmax function to produce the normalized class probabilities:

$$
\tilde { Y } = \mathrm { s o f t m a x } ( \mathbf { C o n v } _ { 1 \times 1 } ( z ) ) \in \mathbb { R } ^ { K } ,\tag{17}
$$

For supervised training, we adopt the standard cross-entropy loss $\mathcal { L } _ { \mathrm { C E } }$ over all pixels in the training set:

$$
\mathcal { L } _ { \mathrm { C E } } = - \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \sum _ { k = 1 } ^ { K } y _ { i , k } \log ( \tilde { y } _ { i , k } ) ,\tag{18}
$$

where $y _ { i , k } \in \{ 0 , 1 \}$ is the ground-truth label indicator for pixel i and class $k , \ \tilde { y } _ { i , k }$ is the predicted probability, and N is the total number of pixels in the batch. This formulation enforces the network to maximize the likelihood of correct class assignments while maintaining the compact and modalityaware feature representation learned by M2Heat.

## IV. EXPERIMENT AND DISCUSSION

This section presents a comprehensive experimental evaluation of the proposed M2Heat model, employing three prominent HSI-LiDAR benchmark datasets: Trento, Houston2013, and Augsburg. We begin by delineating the characteristics of each dataset, followed by a detailed description of our experimental protocol and the adopted evaluation metrics. To rigorously assess its performance, M2Heat is benchmarked against nine state-of-the-art methods. Finally, to substantiate the efficacy of our design choices and demonstrate the model’s robustness, we conduct extensive ablation studies and supplementary analyses across all three datasets.

## A. Datasets Description

1) Trento: The Trento dataset was acquired over a rural area near Trento, located in the southern part of Italy. It comprises co-registered HSI and LiDAR data. The scene has a spatial size of 166 × 600 pixels with a ground sampling distance of 1 m. The hyperspectral data cover the spectral range from 0.42 to 0.99 µm, comprising 63 contiguous bands after removing noisy channels. The LiDAR measurements were collected using an Optech ALTM 3100EA sensor, providing elevation information complementary to the spectral content. The dataset contains a total of 30,214 labeled samples distributed over six land use classes, making it a widely used benchmark for multimodal RS classification.

![](images/5e1b6dba8613e7b641b05c723544cabd20e8c017a4f9e12e58e15dd89c7b0e5b.jpg)

![](images/b94380459e1d8d9c429056c48a0700f2c888b95e2b219b1ed6a452cc3a648397.jpg)

![](images/504aff9c91f004f1b8a14eeba49e7d28b777e860d784e6b9a2a4db40bcfac4a9.jpg)

![](images/cd3cfa85fd700aad9609bad333e6201ee3cce0df2a20a8dc7e779ee9971171f2.jpg)

![](images/0944f702def0e8bfa0f2af9c9323a6b3b6d3d1aa741c51a4273ab1ce81bf0839.jpg)

![](images/051f8e42585644f43f113562cccc2643d9732413eb5aa4a4550a9a22c0e21842.jpg)  
Fig. 4. Impact of hyperparameter selection on the model. (a)–(c) OA variations under different learning rates and batch sizes on the Trento, Houston2013, and Augsburg datasets, respectively; (d)–(f) OA variations under different patch sizes.

TABLE II  
CLASSIFICATION ACCURACY (%) OBTAINED BY DIFFERENT METHODS ON THE TRENTO DATASET.
<table><tr><td>No. Year</td><td>SF 2022 [34]</td><td>MambaHSI 2024 [17]</td><td>MFT 2023 [36]</td><td>HCT 2023 [13]</td><td>DSHFNet 2023 [59]</td><td>ExViT 2023 [38]</td><td>Cross-HL 2024 [60]</td><td>HLMamba 2024 [41]</td><td>S2CMamba 2024 [42]</td><td>M2Heat</td></tr><tr><td></td><td>91.75</td><td>99.28</td><td>98.03</td><td>98.93</td><td>99.05</td><td>99.48</td><td>100</td><td>99.05</td><td>98.28</td><td>99.88</td></tr><tr><td>23456</td><td>82.83</td><td>85.81</td><td>99.03</td><td>97.85</td><td>96.50</td><td>99.20</td><td>98.99</td><td>97.50</td><td>99.10</td><td>98.75</td></tr><tr><td></td><td>94.77</td><td>94.55</td><td>99.56</td><td>100</td><td>97.17</td><td>100</td><td>99.35</td><td>99.13</td><td>100</td><td>99.56</td></tr><tr><td></td><td>99.51</td><td>99.99</td><td>100</td><td>100</td><td>99.96</td><td>100</td><td>100</td><td>100</td><td>100</td><td>100</td></tr><tr><td></td><td>99.17</td><td>98.36</td><td>100</td><td>100</td><td>100</td><td>99.99</td><td>99.96</td><td>99.86</td><td>100</td><td>100</td></tr><tr><td></td><td>78.79</td><td>82.56</td><td>94.99</td><td>93.02</td><td>93.47</td><td>94.61</td><td>86.24</td><td>89.06</td><td>95.43</td><td>97.91</td></tr><tr><td>OA(%)</td><td>94.51</td><td>96.06</td><td>99.11</td><td>98.92</td><td>98.80</td><td>99.29</td><td>98.44</td><td>98.42</td><td>99.21</td><td></td></tr><tr><td>AA(%)</td><td>91.14</td><td>93.43</td><td>98.60</td><td>98.30</td><td>97.69</td><td>98.88</td><td>97.42</td><td>97.43</td><td>98.80</td><td>99.64</td></tr><tr><td>κ</td><td>0.9267</td><td>0.9475</td><td>0.9882</td><td>0.9856</td><td>0.9839</td><td>0.9905</td><td>0.9792</td><td>0.9790</td><td>0.9894</td><td>99.35 0.9952</td></tr></table>

![](images/ba22bcdde8b346d5bf295908895a608e854b2911bfe6fb1c813d6c0822a1499f.jpg)  
Fig. 5. Classification result mapping by different models on the Trento dataset.

2) Houston2013: <sup>1</sup> The Houston2013 dataset was collected over the University of Houston campus and its neighboring urban areas during the 2013 IEEE GRSS Data Fusion Contest. The HSI was acquired using the Compact Airborne Spectral Imager (CASI), with a spatial size of 345 × 1905 pixels and a ground sampling distance of 2.5 m. The spectral range spans 0.38–1.35 µm, consisting of 144 contiguous bands after removing water absorption and noisy channels. In addition, LiDAR-derived Digital Surface Model (DSM) data were acquired over the same area to provide complementary elevation information. The dataset contains 15,029 labeled samples distributed across 15 land-cover classes, offering diverse urban and vegetation categories for evaluating multimodal data fusion methods.

3) Augsburg [61]: <sup>2</sup> The Augsburg dataset covers an urban area in the vicinity of Augsburg, Germany, and includes a spaceborne hyperspectral (HSI) image and a digital surface model (DSM) derived from LiDAR measurements. The hyperspectral data were acquired by the HySpex sensor, providing 180 spectral bands spanning the 0.4–2.5 µm wavelength range. The DSM data were captured by the DLR-3K LiDAR system, providing precise elevation information for the same area. For consistency and computational efficiency, all modalities were resampled to a unified spatial resolution of 30 m ground sampling distance (GSD), resulting in an image of size 332 × 485 pixels. Each pixel is thus associated with a 180- dimensional spectral vector from the HSI and a corresponding elevation value from the DSM. Ground-truth labels were derived from OpenStreetMap, covering multiple land-cover categories suitable for multimodal classification and fusion research, including urban, vegetation, water, and impervious surfaces.

The Houston2013 and Augsburg datasets provide publicly available ground-truth annotations with standardized training–testing splits, which were directly adopted in our experiments to ensure comparability with existing works. In contrast, the Trento dataset does not include an official split. Following a class-balanced sampling strategy, we randomly selected 20 labeled pixels per class for training, with the remaining samples used for testing. Detailed category information for the three datasets, along with the division of training and test sets, is shown in Table I.

TABLE III  
CLASSIFICATION ACCURACY (%) OBTAINED BY DIFFERENT METHODS ON THE HOUSTON2013 DATASET.
<table><tr><td rowspan="2">No. Year</td><td rowspan="2">SF 2022 [34]</td><td rowspan="2">MambaHSI 2024 [17]</td><td rowspan="2">MFT 2023 [36]</td><td rowspan="2">HCT 2023 [13]</td><td rowspan="2">DSHFNet 2023 [59]</td><td rowspan="2">ExViT 2023 [38]</td><td rowspan="2">Cross-HL 2024 [60]</td><td rowspan="2">HLMamba 2024 [41]</td><td rowspan="2">S2CMamba 2024 [42]</td><td rowspan="2">M2Heat</td></tr><tr><td></td></tr><tr><td>23456789</td><td>81.29</td><td>80.44</td><td>79.68</td><td>82.15</td><td>82.72</td><td>82.72</td><td>83.10</td><td>83.00</td><td>81.77</td><td>81.39</td></tr><tr><td></td><td>75.85</td><td>84.68</td><td>96.24</td><td>84.77</td><td>84.21</td><td>84.59</td><td>84.87</td><td>85.15</td><td>83.55</td><td>84.96</td></tr><tr><td></td><td>58.61</td><td>54.46</td><td>96.63</td><td>97.31</td><td>97.03</td><td>98.41</td><td>98.42</td><td>88.32</td><td>97.82</td><td>96.24</td></tr><tr><td></td><td>85.61</td><td>88.73</td><td>95.55</td><td>97.82</td><td>89.68</td><td>92.71</td><td>93.09</td><td>100</td><td>89.77</td><td>90.53</td></tr><tr><td></td><td>98.96</td><td>97.44</td><td>100</td><td>99.91</td><td>99.91</td><td>100</td><td>99.91</td><td>100</td><td>100</td><td>100</td></tr><tr><td></td><td>95.80</td><td>100</td><td>79.02</td><td>95.80</td><td>98.60</td><td>99.30</td><td>95.80</td><td>100</td><td>81.82</td><td>100</td></tr><tr><td></td><td>84.79</td><td>81.90</td><td>88.15</td><td>92.78</td><td>74.25</td><td></td><td>86.19</td><td>87.78</td><td></td><td></td></tr><tr><td></td><td>52.90</td><td></td><td></td><td></td><td>73.03</td><td>92.68 85.16</td><td></td><td></td><td>90.02</td><td>92.82</td></tr><tr><td></td><td></td><td>64.48</td><td>83.84</td><td>92.78</td><td></td><td></td><td>83.10</td><td>77.49</td><td>88.70</td><td>95.25</td></tr><tr><td></td><td>76.96</td><td>77.53</td><td>81.40</td><td>85.55</td><td>73.09</td><td>84.99</td><td>83.29</td><td>84.42</td><td>83.66</td><td>86.97</td></tr><tr><td>10</td><td>51.16</td><td>50.97</td><td>59.17</td><td>65.83</td><td>67.86</td><td>65.57</td><td>51.93</td><td>61.29</td><td>57.53</td><td>79.25</td></tr><tr><td>11</td><td>54.74</td><td>57.69</td><td>81.69</td><td>94.02</td><td>71.73</td><td>84.25</td><td>82.64</td><td>79.51</td><td>93.07</td><td>99.43</td></tr><tr><td></td><td>83.57</td><td>80.31</td><td>88.95</td><td>86.36</td><td>92.41</td><td>90.87</td><td>92.03</td><td>87.03</td><td>96.16</td><td>93.47</td></tr><tr><td>13</td><td>66.67</td><td>87.37</td><td>84.56</td><td>88.42 93.93</td><td>56.49</td><td>91.93 99.19</td><td>91.23</td><td>90.53</td><td>92.98</td><td>84.56</td></tr><tr><td>14</td><td>91.90</td><td>80.57</td><td>100 98.52</td><td>83.93</td><td>97.17 76.53</td><td>96.41</td><td>93.12</td><td>98.79</td><td>99.60</td><td>100</td></tr><tr><td>15</td><td>69.98</td><td>69.13</td><td></td><td></td><td></td><td></td><td>100</td><td>92.18</td><td>99.37</td><td>100</td></tr><tr><td>OA(%)</td><td>74.21</td><td>75.90</td><td>87.03</td><td>88.92</td><td>81.36</td><td>89.07</td><td>85.77</td><td>85.06</td><td>87.80</td><td>91.20</td></tr><tr><td>AA(%)</td><td>75.25</td><td>77.05</td><td>87.56</td><td>89.43</td><td>82.31</td><td>89.64</td><td>87.91</td><td>87.23</td><td>89.05</td><td>92.32</td></tr><tr><td>κ</td><td>0.7214</td><td>0.7397</td><td>0.8595</td><td>0.8800</td><td>0.7982</td><td>0.8817</td><td>0.8461</td><td>0.8385</td><td>0.8678</td><td>0.9045</td></tr></table>

![](images/96eb1598169ca1658127357029f648818d9bc39ea9c8ac720a5fa7b9b1576095.jpg)  
Fig. 6. Classification result mapping by different models on the Houston dataset.

## B. Experimental Setup

All experiments in this study were implemented using PyTorch 2.0 and conducted on a Linux workstation equipped with an Intel(R) Xeon(R) Gold 6133 CPU @ 2.50 GHz and an NVIDIA GeForce RTX 4090 GPU with CUDA 11.8 support. The M2Heat model is optimized using the Adam optimizer with a weight decay of 0 and trained for 500 epochs. The learning rate was scheduled using StepLR with a step size of 20 and a decay factor of 0.9. The hyperspectral data were normalized along the spectral dimension before training. Dataset-specific learning rates, batch sizes, and patch sizes are reported in the subsequent hyperparameter selection section. For SOTA comparison methods, we strictly adhered to the hyperparameter settings and training protocols reported in their original publications. In cases where such information was unavailable, we optimized the hyperparameters under the same training regime as our method to ensure a fair and consistent comparison.

To comprehensively evaluate classification performance, three widely accepted metrics were employed: Overall Accu racy (OA), Average Accuracy (AA), and the Kappa coefficient (κ).

## C. Hyperparameter Selection

In deep learning-based classification tasks, the choice of training hyperparameters, particularly the learning rate and batch size, plays a critical role in model convergence and generalization performance.

To identify the optimal learning rate and batch size for the proposed M2Heat model, we conducted a series of grid search experiments on three benchmark datasets: Trento, Houston2013, and Augsburg. The candidate learning rates were set to {1e-4, 5e-4, 1e-3, 5e-3}, and batch sizes were selected from {16, 32, 64, 128}. As shown in Fig. 4, the performance was evaluated using Overall Accuracy (OA) on the validation sets.

For the Trento dataset, the highest OA of 99.64% was achieved with a learning rate of 5e-3 and a batch size of 16. Notably, smaller batch sizes tended to yield slightly better performance, while variations in learning rate exhibited relatively minor influence on accuracy within the tested range.

On the Houston2013 dataset, the optimal configuration was a learning rate of 1e-4 combined with a batch size of 16, which attained an OA of 91.2%. The results indicated that lower

neighborhood.

TABLE IV  
CLASSIFICATION ACCURACY (%) OBTAINED BY DIFFERENT METHODS ON THE AUGSBURG DATASET.
<table><tr><td>No. Year</td><td>SF 2022 [34]</td><td>MambaHSI 2024 [17]</td><td>MFT 2023 [36]</td><td>HCT 2023 [13]</td><td>DSHFNet 2023 [59]</td><td>ExViT 2023 [38]</td><td>Cross-HL 2024 [60]</td><td>HLMamba 2024 [41]</td><td>S2CMamba 2024 [42]</td><td>M2Heat</td></tr><tr><td>1</td><td>80.08</td><td>90.14</td><td>91.62</td><td>92.96</td><td>97.15</td><td>92.52</td><td>95.77</td><td>96.70</td><td>92.63</td><td>95.61</td></tr><tr><td>2</td><td>87.74</td><td>94.59</td><td>94.91</td><td>96.46</td><td>97.95</td><td>95.58</td><td>94.93</td><td>96.76</td><td>97.25</td><td>96.31</td></tr><tr><td>3</td><td>71.12</td><td>2.01</td><td>68.46</td><td>54.76</td><td>29.97</td><td>44.07</td><td>44.49</td><td>58.43</td><td>43.79</td><td>66.08</td></tr><tr><td>4</td><td>78.62</td><td>84.06</td><td>86.24</td><td>85.97</td><td>84.31</td><td>86.45</td><td>85.55</td><td>82.28</td><td>87.15</td><td>92.14</td></tr><tr><td>5</td><td>34.03</td><td>39.77</td><td>41.32</td><td>40.36</td><td>39.48</td><td>41.93</td><td>38.27</td><td>41.93</td><td>42.08</td><td>61.76</td></tr><tr><td>6</td><td>7.94</td><td>0.79</td><td>28.56</td><td>23.44</td><td>8.32</td><td>36.66</td><td>25.75</td><td>13.55</td><td>24.27</td><td>9.95</td></tr><tr><td>7</td><td>11.68</td><td>15.93</td><td>34.78</td><td>35.21</td><td>29.93</td><td>33.75</td><td>28.04</td><td>27.27</td><td>32.45</td><td>49.44</td></tr><tr><td>OA(%)</td><td>78.94</td><td>81.75</td><td>84.46</td><td>86.47</td><td>86.04</td><td>87.18</td><td>86.28</td><td>87.20</td><td>87.11</td><td>90.30</td></tr><tr><td>AA(%)</td><td>53.03</td><td>46.75</td><td>63.70</td><td>61.30</td><td>55.30</td><td>61.71</td><td>60.40</td><td>59.56</td><td>59.94</td><td>67.33</td></tr><tr><td>κ</td><td>0.6984</td><td>0.7318</td><td>0.7821</td><td>0.8162</td><td>0.7972</td><td>0.8164</td><td>0.8030</td><td>0.8162</td><td>0.8153</td><td>0.8607</td></tr></table>

![](images/a790edb51989607f0bde70010e4de3bfeef556e4167a8c1150e998cca33fe2f3.jpg)  
Fig. 7. Classification result mapping by different models on the Augsburg dataset.

learning rates generally facilitated more stable convergence, while increasing batch size did not significantly improve accuracy.

For the Augsburg dataset, the highest OA of 87.48% was obtained with a learning rate of 5e-4 and a batch size of 64. Unlike the other datasets, moderate batch sizes coupled with mid-range learning rates yielded better performance, suggesting a more balanced trade-off between training stability and convergence speed.

To further examine the influence of spatial context on the heat-diffusion process, we evaluated M2Heat with patch sizes of 7, 9, 11, 13, 15 while fixing the learning rate and batch size to the dataset-specific settings identified above. As illustrated in Fig. 3(d), M2Heat exhibits stable performance over a broad range of spatial neighborhoods, although the optimal patch size varies across datasets. Trento achieves its highest OA of 99.64% with an 11 × 11 patch, while Houston2013 similarly peaks at 91.20% under the same setting. For Augsburg, a smaller 7 × 7 patch yields the best OA of 90.30%, whereas larger patches provide lower performance. This result suggests that increasing the spatial context does not necessarily lead to monotonic improvements, as overly large neighborhoods may introduce heterogeneous or class-irrelevant information. The dataset-dependent optima therefore reflect differences in spatial resolution, scene composition, and local class distributions.

Based on these findings, the M2Heat model adopts datasetspecific hyperparameters for optimal performance: a learning rate of 5e-3, batch size of 16, and a patch size of 11 × 11 for Trento, 1e-4, 16, and 11 × 11 for Houston2013, and 5e-4, 64, and 7 × 7 for Augsburg. This tailored hyperparameter selection ensures robust and efficient training across diverse hyperspectral and LiDAR fusion scenarios. For a fair and controlled comparison in the ablation studies, the patch size is fixed to 11 × 11 for all datasets, so that the observed performance differences mainly reflect the effects of the investigated components rather than variations in the input

## D. Comparison Result and Analysis

To comprehensively evaluate the effectiveness of the proposed M2Heat model, we compare it against nine SOTA HSI classification methods. These include two single-source HSI classification approaches, namely SpectralFormer (SF) [34] and MambaHSI [17] which are trained on HSI only, as well as seven multisource joint classification methods integrating HSI and LiDAR data: MFT [36], HCT [13], DSHFNet [59], ExViT [38], Cross-HL [60], HLMamba [41], and S<sup>2</sup>CrossMamba (S2CMamba) [42].

1) Trento Dataset: We first present the classification performance of the competing methods on the Trento dataset, which contains complex urban and natural land cover classes captured by both HSI and LiDAR modalities. The quantitative results, including OA, AA, κ, and classwise accuracy, are summarized in Table II. The corresponding classification maps with a unified color legend are illustrated in Fig. 5.

As observed, conventional single-source HSI methods such as SF and MambaHSI achieve relatively lower OA values of 94.51% and 96.06%, respectively, reflecting the limitation of relying solely on spectral information. Multisource fusion methods generally outperform these baselines by leveraging complementary spatial and elevation cues.

Among multisource methods, MFT, HCT, and DSHFNet demonstrate strong performances with OA above 98.8%, yet still fall short compared to transformer-based and hybrid models. ExViT and Cross-HL achieve notably high accuracies, with Cross-HL reaching perfect classification on four out of six classes and an overall OA of 98.44%. The HLMamba and S2CMamba methods further improve the performance, with OA values of 98.42% and 99.21%, respectively, benefiting from enhanced feature extraction and fusion strategies. The proposed M2Heat model attains the highest OA of 99.64%, surpassing all competitors by a clear margin. It achieves the best or near-best classification accuracies across all classes, especially excelling in the challenging sixth class with 97.91% accuracy. Correspondingly, the κ reaches 0.9952, indicating near-perfect agreement.

TABLE V  
OVERALL ACCURACY (OA %) OF THE ABLATION STUDY.
<table><tr><td rowspan="3">HSI</td><td rowspan="3">LiDAR</td><td rowspan="3">Joint</td><td colspan="2">CCF</td><td colspan="3">OA (%)</td></tr><tr><td>S1</td><td>S2</td><td>Trento</td><td>Houston</td><td>Augsburg</td></tr><tr><td>√</td><td></td><td></td><td></td><td></td><td>97.63</td><td>86.55</td><td>83.69</td></tr><tr><td></td><td>√</td><td></td><td></td><td></td><td>97.92</td><td>60.31</td><td>76.98</td></tr><tr><td></td><td></td><td>√</td><td></td><td></td><td>99.18</td><td>89.80</td><td>85.90</td></tr><tr><td>√</td><td>√</td><td></td><td>√</td><td></td><td>99.26</td><td>87.82</td><td>84.95</td></tr><tr><td>√</td><td></td><td>√</td><td>√</td><td></td><td>99.23</td><td>90.05</td><td>87.23</td></tr><tr><td></td><td>√</td><td>√</td><td></td><td>√</td><td>99.26</td><td>88.61</td><td>86.57</td></tr><tr><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>99.64</td><td>91.20</td><td>87.48</td></tr></table>

TABLE VI

OA (%) COMPARISON OF DIFFERENT FREQUENCY REPRESENTATIONS INCFF AND OPERATORS IN HCO.
<table><tr><td rowspan="2">Dataset</td><td colspan="4">CFF</td><td colspan="4">HCO</td></tr><tr><td>DCT</td><td>FFT-RI</td><td>FFT-AP</td><td>FFT-L/H</td><td>FFT</td><td>DST</td><td>DCT</td><td>GFNet</td></tr><tr><td>Trento</td><td>99.51</td><td>99.27</td><td>99.64</td><td>99.44</td><td>99.24</td><td>99.17</td><td>99.64</td><td>99.56</td></tr><tr><td>Houston</td><td>89.96</td><td>90.68</td><td>91.20</td><td>90.10</td><td>91.47</td><td>90.38</td><td>91.20</td><td>90.66</td></tr><tr><td>Augsburg</td><td>87.06</td><td>86.23</td><td>87.48</td><td>86.98</td><td>87.07</td><td>86.28</td><td>87.48</td><td>85.43</td></tr></table>

Visually, the classification maps in Fig. 5 reveal that M2Heat produces spatially coherent and sharply delineated land cover regions. Compared to other methods, M2Heat better preserves fine structural details and boundaries, reducing misclassification in mixed or transitional areas. The overall classification maps generated by competing methods exhibit some degree of noise and misclassification patches, particularly in classes with fewer subtle spectral differences.

In summary, both quantitative and qualitative results validate the effectiveness of M2Heat in exploiting the complementary information of HSI and LiDAR data for accurate land cover classification on the Trento dataset.

2) Houston2013 Dataset: We further evaluate the classification performance of all competing methods on the Houston2013 dataset, which comprises diverse urban land cover classes characterized by complex spatial structures and spectral variability. Table III reports the quantitative results, while Fig. 6 presents the corresponding classification maps with a unified legend.

As shown, single-source HSI methods SF and MambaHSI yield relatively modest OA values of 74.21% and 75.90%, respectively, highlighting the limitations of using spectral information alone in this challenging scenario. Among multisource fusion techniques, CNN-based methods such as MFT and HCT achieve significantly improved results, with OA surpassing 87%, reflecting the benefit of integrating LiDAR elevation data. Transformer-based methods like ExViT and Cross-HL demonstrate competitive performances, achieving OA values of 89.07% and 85.77%, respectively. However, the recently proposed HLMamba and S2CMamba methods yield slightly lower OA values of 85.06% and 87.80%, suggesting room for improvement in their fusion strategies.Our proposed M2Heat model achieves favorable overall performance, attaining an OA of 91.20%, AA of 92.32%, and the highest Kappa coefficient of 0.9045. Notably, M2Heat achieves the best classification accuracies across several challenging classes, including classes Residential, Commercial, Highway, and Railway, which correspond to complex urban features with subtle spectral and spatial differences.

TABLE VII  
OPERATOR-LEVEL COMPARISON OF DIFFERENT MODULES. T: TRENTO; H: HOUSTON2013; A: AUGSBURG; P: PARAMETERS; F: FLOPS; M: GPU MEMORY; I: INFERENCE TIME; C : OPERATOR-LEVEL THEORETICAL COMPLEXITY. M AND I ARE MEASURED ON THE TRENTO DATASET UNDER THE SAME HARDWARE AND EVALUATION PROTOCOL.
<table><tr><td rowspan="2">Operators</td><td colspan="3">OA (%)</td><td colspan="2">Model Statistics</td><td colspan="2">Runtime Metrics</td><td rowspan="2">Ct</td></tr><tr><td>T</td><td>H</td><td>A</td><td>P(K)</td><td>F (G)</td><td>M (MB)</td><td>I (s)</td></tr><tr><td>Conv</td><td>99.26</td><td>88.92</td><td>85.60</td><td>793.71</td><td>8.444</td><td>425.84</td><td>0.62</td><td>O(N)</td></tr><tr><td>MSA</td><td>99.23</td><td>87.78</td><td>87.10</td><td>588.41</td><td>5.632</td><td>539.83</td><td>0.68</td><td>O(N2)</td></tr><tr><td>Mamba</td><td>99.22</td><td>88.14</td><td>87.21</td><td>593.16</td><td>5.189</td><td>704.63</td><td>0.75</td><td>O(N)</td></tr><tr><td>HCO</td><td>99.64</td><td>91.20</td><td>87.48</td><td>555.62</td><td>5.334</td><td>490.72</td><td>0.53</td><td> $\mathcal { O } ( \dot { N } ^ { 1 . 5 } )$ </td></tr></table>

TABLE VIII

OA (%) AND RELATIVE OA DECAY (ROD, %) UNDER DIFFERENT TRAINING-SAMPLE RATIOS. ROD IS COMPUTED RELATIVE TO THE PERFORMANCE USING 100% TRAINING SAMPLES, WHERE A LOWER VALUE INDICATES BETTER ROBUSTNESS.
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Operators</td><td colspan="3">OA (%) ↑</td><td colspan="4">ROD (%) ↓</td></tr><tr><td>25%</td><td>50%</td><td>75%</td><td>25%</td><td>50%</td><td>75%</td><td>Avg.</td></tr><tr><td rowspan="4">Trento</td><td>Conv</td><td>98.68</td><td>99.12</td><td>99.15</td><td>0.580</td><td>0.146</td><td>0.112</td><td>0.279</td></tr><tr><td>MSA</td><td>97.56</td><td>97.97</td><td>98.29</td><td>1.679</td><td>1.270</td><td>0.949</td><td>1.299</td></tr><tr><td>Mamba</td><td>97.84 99.20</td><td></td><td>98.66</td><td>1.394</td><td>0.024</td><td>0.567</td><td>0.662</td></tr><tr><td>HCO</td><td>99.18</td><td>99.45</td><td>99.54</td><td>0.465</td><td>0.191</td><td>0.102</td><td>0.252</td></tr><tr><td rowspan="4">Houston</td><td>Conv</td><td>87.62</td><td>87.69</td><td>87.98</td><td>1.462</td><td>1.379</td><td>1.056</td><td>1.299</td></tr><tr><td>MSA</td><td>86.24</td><td>86.63</td><td>87.56</td><td>1.751</td><td>1.312</td><td>0.246</td><td>1.103</td></tr><tr><td>Mamba</td><td>86.95</td><td>87.26</td><td>87.77</td><td>1.347</td><td>0.999</td><td>0.422</td><td>0.923</td></tr><tr><td>HCO</td><td>90.32</td><td>90.56</td><td>90.69</td><td>0.964</td><td>0.698</td><td>0.563</td><td>0.742</td></tr><tr><td rowspan="4">Augsburg</td><td>Conv</td><td>84.60</td><td>85.32</td><td>85.10</td><td>1.165</td><td>0.333</td><td>0.583</td><td>0.694</td></tr><tr><td>MSA</td><td>85.91</td><td>85.76</td><td>86.12</td><td>1.369</td><td>1.542</td><td>1.123</td><td>1.345</td></tr><tr><td>Mamba</td><td>86.10</td><td>86.26</td><td>85.74</td><td>1.273</td><td>1.085</td><td>1.690</td><td>1.350</td></tr><tr><td>HCO</td><td>86.80</td><td>87.10</td><td>87.26</td><td>0.779</td><td>0.438</td><td>0.254</td><td>0.490</td></tr></table>

Visual inspection of the classification maps (Fig. 6) reveals that M2Heat better handles challenging regions affected by shadows and mixed pixels, which are common in hyperspectral urban scenes. In particular, the shadowed area caused by vegetation and tall buildings, which commonly degrades spectral quality, is more accurately classified by M2Heat compared to other methods. Additionally, several competing models exhibit evident misclassifications between Water and Railway classes, likely due to spectral confusion and limited spatial context modeling. In contrast, M2Heat delineates these classes with clearer boundaries and fewer errors, demonstrating its ability to exploit complementary spectral, spatial, and elevation information.

3) Augsburg Dataset: The classification results on the Augsburg dataset are summarized in Table IV, with the corresponding maps shown in Fig. 7. Single-source HSI methods SF and MambaHSI achieve lower overall accuracies of 78.94% and 81.75%, respectively, demonstrating the challenges posed by this dataset’s diverse land cover types.

Among multisource methods, MFT, HCT, DSHFNet, ExViT, Cross-HL, HLMamba, and S2CMamba exhibit comparable performances, with OAs clustered around 86–87%. Our proposed M2Heat achieves the highest OA of 90.30% and the highest Kappa coefficient of 0.8607, confirming its robustness in integrating spectral and elevation information.

While M2Heat shows marginal gains in overall metrics, it notably improves classification accuracies in challenging classes such as classes Residential Area, Low Plants, Allotment, and Water, which correspond to heterogeneous urban features with subtle spectral differences. However, all methods struggle with classes Industrial Area and Commercial Area due to limited training samples and class imbalance, which is a known difficulty in this dataset.

![](images/d7b99aeee7c10f1292d2cb64e3c2f5d6f93ef862707d7ffdd21471cb8ee9b798.jpg)  
Fig. 8. t-SNE visualization on Trento, Houston, and Augsburg datasets. (a) M2CNN: HCO replaced by a convolution. (b) M2ViT: HCO replaced by MSA. (c) M2Mamba: HCO replaced by Mamba. (d) M2Heat: the proposed method.

TABLE IX  
OA (%) UNDER FIXED AND LEARNABLE DIFFUSIVITY CONFIGURATIONS. S DENOTES A CERTAIN NUMBER.
<table><tr><td rowspan="2">Dataset</td><td colspan="5">Fixed FVEs</td><td colspan="3">Learnable FVEs</td></tr><tr><td>0.10</td><td>0.25</td><td>0.50</td><td>0.75</td><td>1.00</td><td>S</td><td>P × P</td><td> $\overline { { P \times P \times C } }$ </td></tr><tr><td>Trento</td><td>99.44</td><td>99.31</td><td>99.27</td><td>99.24</td><td>99.21</td><td>99.27</td><td>99.42</td><td>99.64</td></tr><tr><td>Houston</td><td>88.38</td><td>87.92</td><td>88.55</td><td>88.21</td><td>88.78</td><td>87.37</td><td>88.11</td><td>91.20</td></tr><tr><td>Augsburg</td><td>86.92</td><td>86.11</td><td>86.56</td><td>87.01</td><td>86.73</td><td>86.64</td><td>87.09</td><td>87.48</td></tr></table>

Visual comparison reveals that M2Heat produces smoother and more spatially consistent classification maps, effectively reducing noise and misclassification in complex regions. This result indicates that M2Heat can effectively exploit complementary information from HSI and LiDAR data for fine-grained urban land cover classification.

## E. Ablation Study

1) Contribution of Each Component of M2Heat: To verify the effectiveness of the key components in M2Heat, we conducted a series of ablation experiments, as summarized in Table V. S1 is short for the cross-frequency fusion of U<sup>t</sup> and U<sup>t</sup>, while S2 is short for the dynamic fusion of U<sup>t</sup> and U<sup>ˆ t</sup><sub>f</sub>.

Single-modality baselines using only HSI or LiDAR yield significantly lower accuracies, especially on the Houston and Augsburg datasets, indicating that either modality alone is in sufficient for complex urban classification tasks. When features from both modalities are joint learning, performance improves across all datasets, confirming the benefit of multisource fusion.

Introducing S1 to the joint representation consistently enhances OA, with gains of +1.40%, +0.25%, and +1.33% on Trento, Houston, and Augsburg, respectively. This demonstrates the importance of capturing frequency-specific correlations between modalities. Similarly, adding S2 also boosts performance compared to the joint baseline, with notable improvements of +1.34%, +1.61%, and +0.67% on the three datasets, respectively, highlighting the role of dynamic fusion in refining high-level feature integration.

The full M2Heat model, which incorporates both S1 and S2, achieves the highest OA on all datasets: 99.64%, 91.20%, and 87.48%, surpassing all partial configurations.

Beyond the component-level ablation, we further examine the frequency representations in CFF and spectral operators in HCO. Since the two modules use frequency transforms for different purposes, module-specific alternatives are evaluated separately, as summarized in Table VI.

TABLE X  
VALUES OF THE LEARNED EFFECTIVE DIFFUSION COEFFICIENTS k UNDER THE SCALAR FVE CONFIGURATION APPLIED IN EACH DATASET.
<table><tr><td></td><td>Trento</td><td>Houston</td><td>Augsburg</td></tr><tr><td>HSI</td><td>0.7452</td><td>1.2356</td><td>0.7578</td></tr><tr><td>LiDAR</td><td>1.2769</td><td>1.0511</td><td>0.6903</td></tr><tr><td>Fuse</td><td>18.2251</td><td>15.1014</td><td>3.6354</td></tr></table>

For CFF, we compare DCT coefficients, FFT with real– imaginary decomposition (FFT-RI), the proposed amplitude– phase decomposition (FFT-AP), and FFT with low–high frequency decomposition (FFT-L/H). FFT-AP consistently achieves the best performance, outperforming the strongest alternative by 0.13%, 0.52%, and 0.42% on the three datasets. Notably, FFT-RI preserves the same Fourier information but adopts a different representation, while FFT-L/H explicitly separates frequency bands. Their lower accuracies demonstrate the advantage of amplitude–phase decomposition for crossmodal frequency interaction.

For HCO, FFT-, DST- [62], and DCT-based realizations are compared with GFNet [63], a generic Fourier filtering operator. DCT performs best on Trento and Augsburg and achieves the highest average OA across the three datasets. Although FFT slightly outperforms DCT on Houston2013, the DCT based realization shows more consistent overall performance and remains aligned with the finite-domain heat-diffusion formulation. The lower performance of DST and GFNet further suggests that the effectiveness of HCO cannot be attributed to generic frequency-domain filtering alone.

Overall, these results validate the distinct frequency designs of M2Heat: FFT-AP is more suitable for cross-modal interaction in CFF, while DCT provides an effective spectral realization of the heat-conduction process in HCO.

2) Comparison Between Different Operators: To further validate the effectiveness of the proposed HCO in M2Heat, we compare it with three representative operator designs: 3 × 3 Conv, Multihead Self Attention (MSA), and Mamba.

As summarized in Table VII, replacing HCO with any of these alternatives leads to noticeable performance degradation. Conv achieves reasonable accuracy on Trento with 99.26% due to its strong locality modeling, but underperforms on Houston2013 and Augsburg, where large-scale spatial context and inter-modal correlation are crucial. MSA improves global representation capacity, but its computational overhead is prohibitive for large spatial dimensions, and its accuracy on Houston2013 by 87.78% is notably lower than HCO. Mamba strikes a better balance between complexity and context range; however, it lacks explicit mechanisms for frequency-aware fusion, limiting its discriminative power.

In contrast, HCO attains the highest OA on all datasets: 99.64%, 91.20%, and 87.48%, while maintaining the smallest parameter count by 555.62k and moderate FLOPs with 5.334G. HCO further exhibits the lowest inference time of 0.53 s and a moderate peak GPU memory footprint of 490.72 MB, indicating that its frequency-domain computation does not introduce prohibitive runtime or memory overhead. More importantly, HCO performs two-dimensional global feature propagation in the frequency domain while preserving the spatial organization of image patches, which provides an interpretable alternative to sequence-based global modeling. This property allows M2Heat to integrate fine-grained local features and long-range dependencies across modalities, leading to favorable classification performance in both homogeneous and heterogeneous scenes.

(c) Augsburg  
(b) Houston  
(a) Trento  
High Random Low  
![](images/7469074e44e0af4397b8ac728e566c0d45ab607853ceb0fd3fc0bae4c619fa64.jpg)  
Fig. 9. Visualization of different shapes of learnable FVEs on Trento, Houston, and Augsburg datasets.

![](images/fdcd1def562e8b205fc1beaf72562b6ab59e0033ad8613e80f3692467d71b90a.jpg)

![](images/fe5a2e5dcc2ba632b8edbf0c5e9f6623c035b13bd2b702eff80e3aae53407d1f.jpg)

![](images/41d3866f38b812ee8f43e67f00aaf9eb2f91faf94dc48e50e8b23d179eec46fd.jpg)  
Fig. 10. Diffusivity-guided perturbation analysis of HCO on Trento, Houston2013, and Augsburg. High, Random, and Low represent perturbing regions with high diffusivity, random regions, and low diffusivity, respectively.

To assess robustness under limited supervision, we further evaluate the four operators using 25%, 50%, and 75% of the original training samples. For each ratio, the training samples are selected using class-balanced sampling with a fixed random seed to ensure consistent and reproducible subsets. The fullsample result is used as the reference, and the relative OA decay (ROD) is computed as $( \mathrm { O A _ { 1 0 0 } } - \mathrm { O A } _ { r } ) / \mathrm { O A _ { 1 0 0 } } \times 1 0 0 \% .$ where a lower value indicates less performance degradation.

As shown in Table VIII, HCO achieves the highest OA at all reduced training ratios and the lowest average ROD on Trento, Houston, and Augsburg, with 0.252%, 0.742%, and 0.490%, respectively. Although several competing operators obtain lower ROD at individual ratios, HCO shows the smallest overall degradation across all three datasets. This demonstrates that HCO maintains both higher classification accuracy and more stable performance as the available training samples decrease.

To further investigate the discriminative capability of different operators, we conducted a t-Distributed Stochastic Neighbor Embedding (t-SNE) analysis on the output feature representations from the last encoder layer. As shown in Fig. 8, each point corresponds to a sample, and colors indicate distinct classes, with the color scheme kept consistent with the classification map visualization in Section IV.

On the Trento dataset, the feature clusters produced by M2Heat exhibit the most distinct inter-class boundaries, with minimal overlap, indicating that the proposed HCO effectively enhances class separability in the spectral–spatial feature space. In the more challenging Houston2013 dataset, which contains a larger number of classes with substantial intra-class variability, M2Heat still maintains a clear and compact class distribution in the t-SNE space, reflecting its robustness in complex urban scenes.

For the Augsburg dataset, due to its inherent challenges, such as heterogeneous urban landscapes and subtle spectral differences, most methods, including M2Heat, show partial confusion between Residential Area and Allotment classes. Nevertheless, M2Heat achieves more coherent clustering patterns than the Conv-, MSA-, and Mamba-based baselines, demonstrating its advantage in learning modality-complementary representations with better global discrimination.

3) Analysis of Learnable Diffusion Patterns in HCO: To investigate the effect of the decay coefficient k in the HCO, we compare five fixed, non-learnable constants with three configurations of learnable FVEs: (1) Certain Number, where k is represented as a scalar; (2) $P \times P ,$ , where k varies spatially but remains spectrally invariant; and (3) $P \times P \times C ,$ , where k jointly adapts to spatial positions and spectral channels. The results in Table IX show that the optimal fixed value varies across datasets, whereas the learnable $P \times P \times C$ configuration consistently achieves the highest OA on all three benchmarks. This observation indicates that jointly modeling spatial and channel-dependent diffusion coefficients provides greater flexibility in capturing modality- and scene-specific propagation patterns, thereby motivating its adoption in the final M2Heat architecture.

To further analyze the learned diffusion behavior, Table X summarizes the scalar FVEs under the Certain Number setting. The HSI and LiDAR branches exhibit moderate and datasetdependent diffusion strengths, whereas the shared fusion branch consistently learns larger coefficients. This suggests that modality-specific HCO layers mainly preserve heterogeneous cues, while the shared branch requires stronger diffusion for cross-modal interaction and aggregation. The variation across datasets further indicates that HCO adapts its propagation behavior to different scene characteristics.

The $P \times P$ and $P \times P \times C$ FVEs are visualized in Fig. 9.

For the latter, the coefficients are decomposed into channelaveraged spatial maps and channel-wise response curves. Compared with the spatial-only configuration, $\bar { P } \times P \times C$ exhibits richer spatial and channel variations, while the HSI, LiDAR, and shared branches show distinct response patterns. These observations indicate that HCO adaptively regulates diffusion according to modality-specific and shared representations.

To further quantify whether these patterns are related to model decisions, we conduct a diffusivity-guided perturbation analysis. Spatial positions are ranked by their effective diffusivity and perturbed using three strategies: High selects the largest values, Low selects the smallest values, and Random selects the same number of positions uniformly at random. The HSI, LiDAR, and fused branches are masked independently, with each mask generated according to the distribution of its corresponding effective diffusivity map. The selected positions are replaced by the mean of the unmasked regions in each sample, and Random is averaged over five trials. Perturbation ratios of 10%, 20%, 30%, 40%, and 50% are evaluated.

As shown in Fig. 10, OA degradation generally increases with the perturbation ratio. Diffusivity-guided perturbations produce larger drops than random perturbation, with highdiffusivity regions becoming particularly sensitive at larger ratios on Houston2013 and Augsburg. Trento shows smaller overall variations but follows a similar tendency at higher perturbation levels. These results establish a quantitative association between the learned diffusivity and prediction sensitivity, complementing the qualitative visualization and supporting the mechanism-level interpretability of HCO.

## V. CONCLUSION

In this paper, we introduce M2Heat, a novel multimodal fusion framework inspired by heat conduction principles, specifically designed for HSI–LiDAR joint classification. By integrating modality-specific and shared encoders within a cross-modal conduction-inspired fusion module, M2Heat efficiently captures long-range dependencies with sub-quadratic computational complexity while providing physically interpretable insights into the fusion dynamics. The physics-driven vHeat module, complemented by enhanced FVEs, facilitates precise modeling of anisotropic spectral–spatial heat propagation, thereby enhancing discriminative feature extraction and class separability. A hybrid fusion strategy that synergizes alignment-aware spatial learning with frequency-domain interactions further ensures robust and informative feature integration. Extensive experiments across diverse benchmarks show that M2Heat achieves competitive and favorable overall performance, while the ablation studies verify the contributions of the FVE design and dual-phase fusion mechanism to classification efficacy.

While M2Heat adopts an efficient global modeling strategy through the HCO, its computational cost can still be further reduced compared with linear-complexity operators such as Mamba. This observation motivates our future efforts toward designing more computationally efficient, physics-inspired fusion architectures that retain interpretability while reducing complexity. Furthermore, we aim to enhance robustness and generalization across diverse multimodal RS tasks and extend the applicability of multimodal fusion to cross-scenario and cross-domain settings, paving the way for more practical and scalable RS solutions.

## REFERENCES

[1] Z. Chen, H. Wang, J. Yao, J. Zhang, P. Ghamisi, J. Zhou, P. M. Atkinson, and B. Zhang, “Cangling-knowflow: A unified knowledge-and-flow-fused agent for comprehensive remote sensing applications,” arXiv preprint arXiv:2512.15231, 2025.

[2] K. Wei, J. Yao, J. Cui, X. Zhao, L. Wang, G. Vivone, and P. Ghamisi, “Beyond bi-temporal and unimodal: A multimodal-temporal coupling network for change detection in conflict zones,” Inf. Fusion, p. 104402, 2026.

[3] M. Pal, “Deep learning algorithms for hyperspectral remote sensing classifications: An applied review,” Int. J. Remote Sens., vol. 45, no. 2, pp. 451–491, 2024.

[4] K. Wei, J. Dai, D. Hong, and Y. Ye, “MGFNet: An MLP-dominated gated fusion network for semantic segmentation of high-resolution multi-modal remote sensing images,” Int. J. Appl. Earth Obs. Geoinf., vol. 135, p. 104241, 2024.

[5] Z. Xie, S. Miao, Y. Jiang, Z. Zhang, J. Yao, X. Li, J. Huang, and P. Ghamisi, “FSG-Net: Frequency-spatial synergistic gated network for high-resolution remote sensing change detection,” arXiv preprint arXiv:2509.06482, 2025.

[6] A. U. G. Sankararao and P. Rajalakshmi, “UAV-based hyperspectral remote sensing and CNN for vegetation classification,” in Proc. IEEE Int. Geosci. Remote Sens. Symp. (IGARSS), 2022, pp. 7737–7740.

[7] F. Wang, X. Yao, L. Xie, J. Zheng, and T. Xu, “Rice yield estimation based on vegetation index and florescence spectral information from UAV hyperspectral remote sensing,” Remote Sens., vol. 13, no. 17, p. 3390, 2021.

[8] D. Sun, J. Yao, W. Xue, C. Zhou, P. Ghamisi, and X. Cao, “Mask approximation net: A novel diffusion model approach for remote sensing change captioning,” IEEE Trans. Geosci. Remote Sens., vol. 63, pp. 1–11, 2025.

[9] T. Xu, F. Wang, Z. Shi, L. Xie, and X. Yao, “Dynamic estimation of rice aboveground biomass based on spectral and spatial information extracted from hyperspectral remote sensing images at different combinations of growth stages,” ISPRS J. Photogramm. Remote Sens., vol. 202, pp. 169–183, 2023.

[10] Y. Wang, B. Zou, L. Chai, Z. Lin, H. Feng, Y. Tang, R. Tian, Y. Tu, B. Zhang, and H. Zou, “Monitoring of soil heavy metals based on hyperspectral remote sensing: A review,” Earth-Sci. Rev., vol. 254, p. 104814, 2024.

[11] L. Song, Z. Feng, S. Yang, X. Zhang, and L. Jiao, “Interactive spectralspatial transformer for hyperspectral image classification,” IEEE Trans. Circuits Syst. Video Technol., vol. 34, no. 9, pp. 8589–8601, 2024.

[12] K. Liu, T. Sun, H. Zeng, Y. Zhang, C.-M. Pun, and C.-M. Vong, “Spatial-aware conformal prediction for trustworthy hyperspectral image classification,” IEEE Trans. Circuits Syst. Video Technol., vol. 35, no. 9, pp. 8754–8766, 2025.

[13] G. Zhao, Q. Ye, L. Sun, Z. Wu, C. Pan, and B. Jeon, “Joint classification of hyperspectral and LiDAR data using a hierarchical CNN and transformer,” IEEE Trans. Geosci. Remote Sens., vol. 61, pp. 1–16, 2023.

[14] X. Wang, L. Song, Y. Feng, and J. Zhu, “S3F2Net: Spatial-spectralstructural feature fusion network for hyperspectral image and LiDAR data classification,” IEEE Trans. Circuits Syst. Video Technol., vol. 35, no. 5, pp. 4801–4815, 2025.

[15] J. Zhang, J. Lei, W. Xie, G. Yang, D. Li, and Y. Li, “Multimodal informative ViT: Information aggregation and distribution for hyperspectral and LiDAR classification,” IEEE Trans. Circuits Syst. Video Technol., vol. 34, no. 8, pp. 7643–7656, 2024.

[16] B. Ma, C. Mu, Y. Liu, X. He, and M. Haidarh, “RoSENet: Rotation and similarity enhancement network for multimodal remote sensing image land cover classification,” IEEE Trans. Geosci. Remote Sens., vol. 63, pp. 1–18, 2025.

[17] Y. Li, Y. Luo, L. Zhang, Z. Wang, and B. Du, “MambaHSI: Spatialspectral mamba for hyperspectral image classification,” IEEE Trans. Geosci. Remote Sens., vol. 62, pp. 1–16, 2024.

[18] Y. He, B. Tu, P. Jiang, B. Liu, J. Li, and A. Plaza, “IGroupSS-Mamba: Interval group spatial-spectral mamba for hyperspectral image classification,” IEEE Trans. Geosci. Remote Sens., vol. 62, 2024.

[19] X. Xu, W. Li, Q. Ran, Q. Du, L. Gao, and B. Zhang, “Multisource remote sensing data classification based on convolutional neural network,” IEEE Trans. Geosci. Remote Sens., vol. 56, no. 2, pp. 937–949, 2018.

[20] D. Hong, L. Gao, N. Yokoya, J. Yao, J. Chanussot, Q. Du, and B. Zhang, “More diverse means better: Multimodal deep learning meets remotesensing imagery classification,” IEEE Trans. Geosci. Remote Sens., vol. 59, no. 5, pp. 4340–4354, 2021.

[21] S. Lu, J. Jing, L. Yang, B. Nie, L. Feng, X. He, and J. Zhou, “FSDFormer: A frequency-selected differential fusion transformer for remote sensing image spatiotemporal fusion,” IEEE Trans. Geosci. Remote Sens., 2025.

[22] Q. Song, F. Mo, K. Ding, L. Xiao, R. Dian, X. Kang, and S. Li, “MCFNet: Multiscale cross-domain fusion network for HSI and LiDAR data joint

classification,” IEEE Trans. Geosci. Remote Sens., vol. 63, pp. 1–12, 2025.

[23] M. Wang, Y. Sun, J. Xiang, and Y. Zhong, “CITNet: Convolution interaction transformer network for hyperspectral and LiDAR image classification,” IEEE Trans. Geosci. Remote Sens., vol. 62, pp. 1–18, 2024.

[24] W. Yu, L. Gao, H. Huang, Y. Shen, and G. Shen, “HI<sup>2</sup>D<sup>2</sup>FNet: Hyperspectral intrinsic image decomposition guided data fusion network for hyperspectral and LiDAR classification,” IEEE Trans. Geosci. Remote Sens., vol. 61, pp. 1–15, 2023.

[25] W. Dong, T. Yang, J. Qu, T. Zhang, S. Xiao, and Y. Li, “Joint contextual representation model-informed interpretable network with dictionary aligning for hyperspectral and LiDAR classification,” IEEE Trans. Circuits Syst. Video Technol., vol. 33, no. 11, pp. 6804–6818, 2023.

[26] A. Gu and T. Dao, “Mamba: Linear-time sequence modeling with selective state spaces,” arXiv preprint arXiv:2312.00752, 2023.

[27] E. Zhu, Z. Chen, D. Wang, H. Shi, X. Liu, and L. Wang, “UNetMamba: An efficient UNet-like mamba for semantic segmentation of highresolution remote sensing images,” IEEE Geosci. Remote Sens. Lett., vol. 22, pp. 1–5, 2025.

[28] Z. Wang, Y. Liu, Y. Tian, Y. Liu, Y. Wang, and Q. Ye, “Building vision models upon heat conduction,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2025, pp. 9707–9717.

[29] S. K. Roy, A. Deria, D. Hong, M. Ahmad, A. Plaza, and J. Chanussot, “Hyperspectral and LiDAR data classification using joint CNNs and morphological feature learning,” IEEE Trans. Geosci. Remote Sens., vol. 60, pp. 1–16, 2022.

[30] Z. Xue, X. Yu, X. Tan, B. Liu, A. Yu, and X. Wei, “Multiscale deep learning network with self-calibrated convolution for hyperspectral and LiDAR data collaborative classification,” IEEE Trans. Geosci. Remote Sens., vol. 60, pp. 1–16, 2022.

[31] J. Li, Y. Ma, R. Song, B. Xi, D. Hong, and Q. Du, “A triplet semisupervised deep network for fusion classification of hyperspectral and LiDAR data,” IEEE Trans. Geosci. Remote Sens., vol. 60, pp. 1–13, 2022.

[32] Y. Yang, D. Zhu, T. Qu, Q. Wang, F. Ren, and C. Cheng, “Single-stream CNN with learnable architecture for multisource remote sensing data,” IEEE Trans. Geosci. Remote Sens., vol. 60, pp. 1–18, 2022.

[33] M. Jin, C. Wang, and Y. Yuan, “Dual heterogeneous network for hyperspectral image classification,” IEEE Trans. Circuits Syst. Video Technol., vol. 35, no. 3, pp. 2905–2917, 2025.

[34] D. Hong, Z. Han, J. Yao, L. Gao, B. Zhang, A. Plaza, and J. Chanussot, “SpectralFormer: Rethinking hyperspectral image classification with transformers,” IEEE Trans. Geosci. Remote Sens., vol. 60, pp. 1–15, 2022.

[35] Z. Xue, X. Tan, X. Yu, B. Liu, A. Yu, and P. Zhang, “Deep hierarchical vision transformer for hyperspectral and LiDAR data classification,” IEEE Trans. Image Process., vol. 31, pp. 3095–3110, 2022.

[36] S. K. Roy, A. Deria, D. Hong, B. Rasti, A. Plaza, and J. Chanussot, “Multimodal fusion transformer for remote sensing image classification,” IEEE Trans. Geosci. Remote Sens., vol. 61, pp. 1–20, 2023.

[37] K. Ding, T. Lu, W. Fu, S. Li, and F. Ma, “Global–local transformer network for HSI and LiDAR data joint classification,” IEEE Trans. Geosci. Remote Sens., vol. 60, pp. 1–13, 2022.

[38] J. Yao, B. Zhang, C. Li, D. Hong, and J. Chanussot, “Extended vision transformer (ExViT) for land use and land cover classification: A multimodal deep learning framework,” IEEE Trans. Geosci. Remote Sens., vol. 61, pp. 1–15, 2023.

[39] J. Li, Z. Zhang, Y. Liu, R. Song, Y. Li, and Q. Du, “SWFormer: Stochastic windows convolutional transformer for hybrid modality hyperspectral classification,” IEEE Trans. Image Process., vol. 33, pp. 5482–5495, 2024.

[40] Y. Wang, L. Liu, J. Xiao, D. Yu, Y. Tao, and W. Zhang, “MambaHSI+: Multidirectional state propagation for efficient hyperspectral image classification,” IEEE Trans. Geosci. Remote Sens., vol. 63, pp. 1–14, 2025.

[41] D. Liao, Q. Wang, T. Lai, and H. Huang, “Joint classification of hyperspectral and LiDAR data based on Mamba,” IEEE Trans. Geosci. Remote Sens., vol. 62, pp. 1–15, 2024.

[42] G. Zhang, Z. Zhang, J. Deng, L. Bian, and C. Yang, “S<sup>2</sup>CrossMamba: Spatial–spectral cross-mamba for multimodal remote sensing image classification,” IEEE Geosci. Remote Sens. Lett., vol. 21, pp. 1–5, 2024.

[43] L. Luo, Y. Zhang, Y. Xu, T. Yue, and Y. Wang, “A VMamba-based spatial–spectral fusion network for remote sensing image classification,” IEEE J. Sel. Topics Appl. Earth Observ. Remote Sens., vol. 18, pp. 14 115–14 131, 2025.

[44] H. Pan, R. Zhao, H. Ge, M. Liu, and Q. Zhang, “Multimodal fusion Mamba network for joint land cover classification using hyperspectral and LiDAR data,” IEEE J. Sel. Topics Appl. Earth Observ. Remote Sens., vol. 18, pp. 17 328–17 345, 2025.

[45] D. Li, W. Xie, Z. Wang, Y. Lu, Y. Li, and L. Fang, “FedDiff: Diffusion model driven federated learning for multi-modal and multi-clients,” IEEE Trans. Circuits Syst. Video Technol., vol. 34, no. 10, pp. 10 353–10 367, 2024.

[46] H. Hu, P. Wang, H. Bi, B. Tong, Z. Wang, W. Diao, H. Chang, Y. Feng, Z. Zhang, Y. Wang, Q. Ye, K. Fu, and X. Sun, “RS-vHeat: Heat conduction guided efficient remote sensing foundation model,” arXiv preprint arXiv:2411.17984, 2025.

[47] H. Bai, Z. Zhao, J. Zhang, B. Jiang, L. Deng, Y. Cui, S. Xu, and C. Zhang, “Deep unfolding multi-modal image fusion network via attribution analysis,” IEEE Trans. Circuits Syst. Video Technol., vol. 35, no. 4, pp. 3498–3511, 2025.

[48] A. A. Kamara, S. He, and A. J. Fofanah, “FAMAFuse: Functional anatomical multiscale attention for multimodal image fusion,” IEEE Trans. Circuits Syst. Video Technol., vol. 36, no. 3, pp. 3215–3230, 2026.

[49] D. Hong, J. Yao, D. Meng, Z. Xu, and J. Chanussot, “Multimodal GANs: Toward crossmodal hyperspectral–multispectral image segmentation,” IEEE Trans. Geosci. Remote Sens., vol. 59, no. 6, pp. 5103–5113, 2021.

[50] R. Hang, X. Qian, and Q. Liu, “Cross-modality contrastive learning for hyperspectral image classification,” IEEE Trans. Geosci. Remote Sens., vol. 60, pp. 1–12, 2022.

[51] X. Wu, D. Hong, and J. Chanussot, “Convolutional neural networks for multimodal remote sensing data classification,” IEEE Trans. Geosci. Remote Sens., vol. 60, pp. 1–10, 2022.

[52] J. Wang, M. Zhang, W. Li, and R. Tao, “A multistage information complementary fusion network based on flexible-mixup for HSI-X image classification,” IEEE Trans. Neural Netw. Learn. Syst., vol. 35, no. 12, pp. 17 189–17 201, 2024.

[53] W. Yu, L. Gao, H. Huang, Y. Shen, and G. Shen, “PID-HLfusion: Pluggable progressive illumination driven hyperspectral and LiDAR data fusion considering crossmodal geometric structures,” IEEE Trans. Instrum. Meas., vol. 73, pp. 1–16, 2024.

[54] S. Dai, D. Song, B. Wang, and W. Chen, “CCEnd-Net: Cross-modal cascaded encoder-decoder network for multisource data fusion classification,” IEEE Trans. Geosci. Remote Sens., vol. 63, pp. 1–23, 2025.

[55] J. Lin, F. Gao, L. Qi, J. Dong, Q. Du, and X. Gao, “Dynamic crossmodal feature interaction network for hyperspectral and LiDAR data classification,” IEEE Trans. Geosci. Remote Sens., vol. 63, pp. 1–16, 2025.

[56] J. Zhang, J. Lei, W. Xie, G. Yang, D. Li, and Y. Li, “Multimodal informative ViT: Information aggregation and distribution for hyperspectral and LiDAR classification,” IEEE Trans. Circuits Syst. Video Technol., vol. 34, no. 8, pp. 7643–7656, 2024.

[57] X. Wang, L. Song, Y. Feng, and J. Zhu, “S3F2Net: Spatial-spectralstructural feature fusion network for hyperspectral image and LiDAR data classification,” IEEE Trans. Circuits Syst. Video Technol., vol. 35, no. 5, pp. 4801–4815, 2025.

[58] B. Qin, S. Feng, C. Zhao, W. Li, and R. Tao, “Collaborative classification of hyperspectral and LiDAR data based on dynamic multiple fractional fourier domains fusion,” IEEE Trans. Geosci. Remote Sens., vol. 63, pp. 1–16, 2025.

[59] Y. Feng, L. Song, L. Wang, and X. Wang, “DSHFNet: Dynamic scale hierarchical fusion network based on multiattention for hyperspectral image and LiDAR data classification,” IEEE Trans. Geosci. Remote Sens., vol. 61, pp. 1–14, 2023.

[60] S. K. Roy, A. Sukul, A. Jamali, J. M. Haut, and P. Ghamisi, “Cross hyperspectral and LiDAR attention transformer: An extended selfattention for land use and land cover classification,” IEEE Trans. Geosci. Remote Sens., vol. 62, pp. 1–15, 2024.

[61] D. Hong, J. Hu, J. Yao, J. Chanussot, and X. X. Zhu, “Multimodal remote sensing benchmark datasets for land cover classification with a shared and specific feature learning model,” ISPRS J. Photogramm. Remote Sens., vol. 178, pp. 68–80, 2021.

[62] P. C. Yip and K. R. Rao, “A fast computational algorithm for the discrete sine transform,” IEEE Trans. Commun., vol. 28, no. 2, pp. 304–307, 1980.

[63] Y. Rao, W. Zhao, Z. Zhu, J. Lu, and J. Zhou, “Global filter networks for image classification,” in Adv. Neural Inf. Process. Syst., vol. 34, 2021, pp. 980–993.