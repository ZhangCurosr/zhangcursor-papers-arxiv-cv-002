# SeeRF: Learning to See Beyond Fields of View via Generative Image Registration and Fusion

Xunpeng Yi, Yibing Zhang, Qinglong Yan, Yifan Lu, Han Xu and Jiayi Ma

Abstract—Due to the inherent discrepancies in imaging viewpoints and field-of-view mismatches among multi-modal sensors, existing fusion methods have long struggled to achieve complete scene-level information integration beyond the limitation in sensor fields of view. Consequently, the generated fusion results often contain modality-specific regions and fusion artifacts, limiting the comprehensive representation of scene structures and semantic information. To address this challenge, we propose a generative registration–fusion framework, termed SeeRF. Specifically, it first introduces a deformation generation strategy based on the diffeomorphic stationary velocity field, combined with visual foundation model-guided feature extraction and a coarse-to-fine progressive multi-modal image registration module, to achieve accurate cross-modal flow estimation and generate valid fieldof-view masks. Subsequently, the registration module is tightly coupled with the generative diffusion module for multi-modal completion and fusion, where the cross-modal control encoding module and the control residual feature encoder–decoder mechanism provide effective guidance. By leveraging the powerful prior modeling capability of generative diffusion models, the proposed framework overcomes the limitations imposed by shared fields of view and enables effective completion and natural fusion of non-overlapping field-of-view regions. Extensive experiments on multiple public datasets and real-world scenarios demonstrate that the proposed method achieves superior performance, validating its effectiveness, superiority, and robustness. The code will be available at https://github.com/XunpengYi/SeeRF.

Index Terms—Image Fusion, Image Registration, Generative Diffusion Model

## I. INTRODUCTION

MAGE fusion is one of the fundamental problems in digital image processing [1]–[4]. Single-modal images are limited by their observation dimensions, resulting in insufficient information representation. In contrast, multi-modal images can integrate complementary information from different modalities through fusion to produce a more comprehensive representation of the scene. Therefore, it has wide applications in fields such as surveillance and security, autonomous driving, and industrial inspection [5]–[8].

Infrared and visible image fusion represents a significant subfield of multi-modal image fusion. It seeks to integrate the rich textural details present in visible images with the thermal radiation characteristics captured by infrared imagery, thereby generating a unified and comprehensive scene representation. Nevertheless, the realization of such fusion fundamentally depends on the acquisition of data from both infrared and visible imaging sensors. Owing to inherent differences in sensor configurations and imaging viewpoints, existing methods characterize the problem by noting that source images acquired by multimodal sensors often exhibit substantial parallax [9], [10]. Therefore, several studies have attempted to jointly address image registration and fusion [11]–[13] to mitigate these challenges. Despite these advances, most existing joint registration–fusion methods primarily focus on correcting spatial misalignment between cross-modal images while assuming that the source images have consistent field-of-view coverage. However, this assumption does not hold under most conditions. Consequently, existing methods must tolerate such field-ofview inconsistencies and often produce degraded fusion results with noticeable artifacts, while failing to preserve complementary cross-modal information across the entire field of view, as illustrated in Fig. 1. This also compromises the semantic integrity of the fused representation, ultimately hindering the performance of downstream tasks.

![](images/930c984b29e27e714c1e9f88624a905127c3557c4c8c9491470104aae7e1c408.jpg)  
Fig. 1. Analysis of image registration and fusion paradigms in terms of viewpoint and field of view.

On the basis of the above observations, considering differences in sensor installation positions, imaging geometry, and device intrinsic parameters, we rethink the challenges associated with data acquired by heterogeneous imaging sensors into two categories: viewpoint misalignment and field-of-view inconsistency. As illustrated in Fig. 2, viewpoint misalignment introduces spatial displacements that disrupt spatial correspondences between cross-modal images, resulting in edge misalignment and ghosting artifacts. Meanwhile, the issues caused by field-of-view inconsistency become apparent as spatial misalignment is corrected and cross-modal alignment is established. Specifically, field-of-view inconsistency leads to non-overlapping regions and the absence of information from certain modalities, making it difficult for conventional fusion strategies to reliably integrate information in regions not covered by both sensors.

![](images/773b9d935de912a486a6bb565476af3b1da0aa92b68828033c717c05bc28d5ce.jpg)  
(a) Conceptual illustration and challenge of multimodal imaging

![](images/ce538d584e47d64ea99cbf9a479e58b40a91a26e932e8b4cb1a5ab34346b6a63.jpg)  
(b) Real data of imaging viewpoint and field-of-view mismatch  
Fig. 2. Schematic illustration of the challenges and analysis of real data in the existing image registration and fusion field. (a) Conceptual illustration and challenge of multimodal imaging. (b) Real data of imaging viewpoint and field-of-view mismatch.

In this context, existing methods either fail to integrate information from the missing modality in regions outside the shared field of view, resulting in discontinuities between regions and inconsistencies in the distribution of fused information, or directly reduce the image coverage through center cropping, discarding field-of-view regions with degraded fusion quality and thereby producing fusion results with limited spatial coverage. This unresolved issue continues to hinder the full exploitation of the complementary information available in unregistered multimodal images for a long time.

To this end, we propose a unified framework for generative image registration and fusion that learns to see information beyond the shared field of view, termed SeeRF. In detail, we first develop a stationary-velocity-field driven multi-modal image registration module (MIR). It synthesizes an invertible deformation field and jointly incorporates a fine-tuned feature extractor and matcher, enabling progressive estimation of the flow field and achieving view alignment. Subsequently, we develop a generative diffusion module for multi-modal completion and fusion (MCF). By performing cross-modal guided completion and fusion, the model establishes consistent observations in missing-scene regions and thereby achieves field-of-view alignment. Importantly, MIR and MCF are intrinsically coupled, jointly breaking the limitations of image fusion beyond the fields of view. MIR establishes a geometrically consistent coordinate system and provides registration-confidence masks, thereby laying the foundation for MCF-guided generation. Conversely, MCF exploits its powerful generative and completion capabilities to compensate for the missing correspondences encountered by MIR in absent or non-overlapping regions. Extensive experimental results demonstrate that the proposed method effectively addresses image fusion beyond the field of view perspective, achieving superior performance and robustness across multiple datasets.

Overall, our contributions can be summarized as follows:

\- From the perspective of multimodal sensor imaging, we propose a unified image fusion framework that jointly addresses spatial misalignment and field-of-view inconsistency. To the best of our knowledge, this is the first image fusion framework to explicitly explore and address the challenges of field-of-view discrepancies.

\- We propose a diffeomorphic stationary velocity field deformation generation strategy, which introduces the feature representation capability guided by foundation models and combines it with a coarse-to-fine progressive flow field regression mechanism to achieve high-precision cross-modal deformation modeling and flow field estimation.

\- By fully exploiting the powerful capability of generative diffusion models and combining it with the valid fieldof-view mask and the collaborative effect of the registration module, we achieve effective generative completion and fusion of missing information outside the field of view, enabling the model to fully explore and utilize complementary information beyond the shared field of view across sensors.

\- Extensive experimental results demonstrate that SeeRF achieves excellent performance in both cross-modal image registration and fusion tasks, and effectively improves the performance of downstream semantic tasks.

## II. RELATED WORK

## A. Classical Image Fusion Methods

Early image fusion methods typically relied on hand-crafted fusion strategies, resulting in inherent limitations in both fusion effectiveness and generalization capability. With the development of deep learning, image fusion has achieved substantial advances in both methodological frameworks and technical capabilities. Initially, the autoencoder-based image fusion methods were proposed to achieve feature-level fusion by using encoders and decoders pretrained on large-scale datasets, together with hand-crafted fusion rules [14], [15]. To overcome the limitations of hand-crafted fusion rules, endto-end optimization-driven CNN-based methods were subsequently introduced [16]. These methods optimize similaritybased objectives with respect to the source images, thereby enabling effective integration of complementary source information. Recent architectural advances have provided new impetus for image fusion, leading to the emergence of Transformerbased [17]–[19], diffusion-based [20]–[22], and lookup-tablebased methods [23] that aim to improve fusion quality, model expressiveness, or efficiency. Meanwhile, considerable progress has been achieved in coupling image fusion with its upstream and downstream tasks [24]–[26], ranging from prerequisite image registration to subsequent high-level vision applications.

![](images/87196985a494dbf621d30fbb569dec031506427f181d2d3fd67460cd10e77a26.jpg)  
Fig. 3. The framework of SeeRF. It effectively couples the multi-modal image registration module with the generative diffusion module for multi-modal completion and fusion, leveraging generative modeling to enable cross-modal information fusion beyond conventional shared field-of-view constraints.

## B. Image Registration and Fusion Methods

To mitigate image misregistration arising from intrinsic discrepancies in multi-modal imaging systems, such as variations in acquisition positions, integrated registration-andfusion methods have been proposed. RFNet [9] integrates style transfer with a stage-wise registration and fusion framework to progressively compensate for parallax-induced spatial misalignment and produce geometrically aligned fused images. Also, UMF-CMGR [27] couples GAN-based crossmodal translation with a multi-level registration and fusion mechanism to facilitate reliable correspondence estimation between initially misaligned multi-modal images. Subsequently, C2RF [28] leverages commonality mining and contrastive learning to achieve image decomposition and fusion, while also adapting to subtle geometric deformations. Furthermore, MulFS-CAP [12] introduces a learnable modality dictionary and exploits a correlation matrix to jointly perform image registration and fusion in an end-to-end unified framework. FusionRegister [29] follows a strategy that first performs image fusion and subsequently refines severely misaligned regions via targeted correspondence correction, leading to enhanced fusion performance. Despite considerable progress in registration-based image fusion, existing methods remain unable to resolve field-of-view inconsistencies. It further leads to bottlenecks in both visual appearance and semantic information during fusion, thereby limiting the practical applicability of existing methods.

## III. METHODOLOGY

In this section, we first clarify the problem formulation. As illustrated in Fig. 3, we next detail the proposed framework, which comprises a multi-modal image registration module and a generative diffusion module for multi-modal completion and fusion, along with diffeomorphic deformation generation strategy. Finally, we specify the loss functions used to optimize SeeRF.

## A. Problem Formulation

Due to variations in sensor placement and imaging characteristics across multi-modal imaging devices, the acquired source images, $e . g . , I _ { v i s } , I _ { i r }$ , often suffer from geometric misalignment and field-of-view inconsistencies. Classical image fusion methods typically align the misregistered images into a common geometric space, followed by image fusion. It can be regarded as a function learning problem $\mathcal { F } _ { R F }$ , and formulated as follows:

$$
I ^ { f } = \mathcal { F } _ { R F } ( I _ { v i s } , I _ { i r } ; \theta _ { R F } ) ,\tag{1}
$$

where $\theta _ { R F }$ denotes the parameters of image registration and fusion network. $I ^ { f }$ represents the fusion image. However, this paradigm fails to fundamentally satisfy the full prerequisites of image fusion. Only registration is incapable of recovering information beyond the single-modality field of view, thereby introducing severe artifacts and leading to inconsistent fused representations. To this end, we explore a new paradigm that integrates image registration with generative completion in a collaborative manner, enabling effective utilization of information beyond the field of view. It can be expressed as:

$$
\widetilde { I } _ { i r } , C ^ { f } = \mathcal { F } _ { R } ( I _ { v i s } , I _ { i r } ; \theta _ { R } ) ,\tag{2}
$$

$$
I _ { C F } ^ { f } = \mathcal { F } _ { C F } ( I _ { v i s } , \widetilde { I } _ { i r } , C ^ { f } ; \theta _ { C F } ) ,\tag{3}
$$

where $\mathcal { F } _ { R }$ and ${ \mathcal { F } } _ { C F }$ denote the image registration and image generative completion fusion mapping function. $\theta _ { R }$ and $\theta _ { C F }$ are the parameters of the multi-modal image registration module and generative diffusion module for multi-modal completion and fusion. $C ^ { f }$ is the validity field-of-view mask.

![](images/018adcf2ee49c222bc0a843382bebf93a0dbbfbb194322a18da8e4d620fdf8ec.jpg)

![](images/0517f4ccc40db265320bd56d0d5f8e3814f52cd31f4304ab154a5060de94c153.jpg)  
(a)  
(b)  
Fig. 4. Visualization of the stationary velocity field for the diffeomorphic deformation. (a) Motion driven by the forward velocity field; (b) Motion driven by the reverse velocity field.

By coupling view registration with field-of-view completion, SeeRF maximizes the utilization of source image information for fusion, leading to improved visual appearance and enhanced performance on downstream tasks.

## B. Deformation Generation Strategy

To simulate sensor misalignment and non-rigid deformations, we construct a global–local deformation model based on diffeomorphic transformations parameterized by a stationary velocity field (SVF). Homography accounts for global projective transformations, while the diffeomorphic component models local deformations, enabling spatially varying nonrigid motion while preserving continuous invertibility. The overall deformation field generation is formulated as follows:

$$
\varphi = H \circ \phi ,\tag{4}
$$

where H corresponds to the global homography-induced deformation field, and $\phi$ corresponds to the local elastic deformation field parameterized by diffeomorphic transformations. ◦ represents the deformation field composition operation.

Homography Deformation Generation: Homography transformation models the projection process and produces globally consistent deformations, which can be formulated as follows:

$$
y \sim H ( x ) ,\tag{5}
$$

where $x ~ \in ~ \Omega ~ \subset ~ \mathbb { R } ^ { 2 }$ and $\mathbf { y } \in \Omega _ { H } \subset \mathbb { R } ^ { 2 }$ denote the original and transformed pixel coordinates, respectively. ∼ indicates equality in the sense of homogeneous coordinates. H represents a randomly sampled homography transformation.

Diffeomorphic Deformation Generation: To address the generated limitations introduced by non-invertible random elastic deformation fields in elastic deformation simulation of mainstream image registration and fusion methods, we propose a diffeomorphic deformation synthesis technique for infrared–visible image registration and fusion, which guarantees the generation of theoretically invertible deformation fields, as shown in Fig. 4. Different from methods that embed diffeomorphic modules into network outputs, we incorporate them into the construction of forward–inverse consistent training samples. Specifically, we formulate elastic deformations for infrared–visible image registration within a stationary velocity field framework. It firstly defines a time-invariant stationary velocity field, defined as follows:

$$
v ( x , t ) = v ( x ) , \quad v : \Omega \to \mathbb { R } ^ { 2 } .\tag{6}
$$

The continuous motion of points in space is governed by a flow equation, from which the relationship between motion and spatial coordinates can be derived as follows:

$$
\frac { \partial \phi _ { t } ( x ) } { \partial t } = v \big ( \phi _ { t } ( x ) \big ) , \qquad \phi _ { 0 } ( x ) = x .\tag{7}
$$

The unit-time deformation flow induced by the stationary velocity field can be defined as the exponential map. Correspondingly, the inverse unit-time deformation flow can be derived by reversing the stationary velocity field. It can be formula as:

$$
\phi ( x ) = E x p ( v ) , \qquad \phi ( x ) ^ { - 1 } = E x p ( - v ) ,\tag{8}
$$

where $E x p$ denotes the flow exponential operator rather than an element-wise exponential function. It converts the infinitesimal motion represented by the velocity field into a finite spatial transformation.

Consequently, the forward and inverse deformation fields share the same velocity-field generator. Therefore, they satisfy following equation:

$$
\phi ^ { - 1 } ( \phi ( x ) ) = x .\tag{9}
$$

By leveraging the forward and inverse deformation fields, we construct corresponding infrared–visible image pairs with elastic deformations to supervise the learning of registration transformations.

## C. Multi-Modal Image Registration Module

The multi-modal image registration module addresses viewpoint discrepancies in multi-modal source images by integrating vision foundation model guidance with a coarse-tofine progressive flow regression paradigm, enabling accurate multi-modal registration. Meanwhile, it provides customized registration-confidence masks, which serve as the foundation for completion and fusion, enabling tight coupling between the two processes.

Given unregistered infrared and visible source images, denoted as $I _ { i r }$ and $I _ { v i s }$ . Features for registration are extracted from the infrared and visible images using a pretrained pyramid feature extractor, respectively, which can be formulated as follows:

$$
\{ F _ { i r , l } ^ { p } \} _ { l = 1 } ^ { L } = \mathcal { P } _ { i } ( I _ { i r } ) , \{ F _ { v i s , l } ^ { p } \} _ { l = 1 } ^ { L } = \mathcal { P } _ { v } ( I _ { v i s } ) ,\tag{10}
$$

where $\mathcal { P } _ { i }$ and $\mathcal { P } _ { v }$ denote the pyramid feature extractors. $\{ F _ { i r , l } ^ { p } \} _ { l = 1 } ^ { L }$ and $\{ F _ { v i s , l } ^ { p } \} _ { l = 1 } ^ { L }$ are the extracted features, where $l = { 1 , . . . , L }$ indicates the feature level. Meanwhile, we leverage the vision foundation model (DINOv3) to extract visual features. Benefiting from large-scale pretraining, the model learns highly generalizable representations, which can be formulated as follows:

$$
\{ F _ { i r , l } ^ { n } \} _ { l \in S } = \mathcal { N } _ { i } ( I _ { i r } ) , \{ F _ { v i s , l } ^ { n } \} _ { l \in S } = \mathcal { N } _ { v } ( I _ { v i s } ) ,\tag{11}
$$

where ${ \mathcal { N } } _ { i }$ and $\mathcal { N } _ { v }$ denote the DINOv3 feature extractors. $\{ F _ { i r , l } ^ { n } \} _ { l \in S }$ and $\{ F _ { v i s , l } ^ { n } \} _ { l \in S }$ are the extracted foundation features. Considering that the features obtained from vision foundation models are mainly at the semantic level, we combine them with pyramid features at deep level S to obtain more robust representations, expressed as:

$$
\hat { F } _ { x , l } = F _ { x , l } ^ { p } + \mathcal { A } _ { l } ( F _ { x , l } ^ { p } , F _ { x , l } ^ { n } ) , x \in \{ i r , v i s \} , l \in S ,\tag{12}
$$

where $\mathbf { \mathcal { A } } _ { l }$ is a gated feature injection mechanism, and l denotes the feature level. It computes attention-based gating weights from the features and modulates feature injection using a temperature parameter:

$$
\mathcal { A } _ { l } ( F _ { x , l } ^ { p } , F _ { x , l } ^ { n } ) { = } \alpha { \cdot } \sigma ( \mathcal { L } _ { g a t e } ( \psi _ { c } ( F _ { x , l } ^ { p } , F _ { x , l } ^ { n } ) ) ) { \cdot } \mathcal { L } _ { p r o j } ( \psi _ { c } ( F _ { x , l } ^ { p } , F _ { x , l } ^ { n } ) ) ,\tag{13}
$$

where $\begin{array} { r } { x \in \begin{array} { r } { \{ i r , v i s \} } \end{array} } \end{array}$ and α represent the modality and temperature parameter. σ is the Sigmoid function. $\psi _ { c }$ denotes the concat operation. $\mathcal { L } _ { g a t e }$ and $\mathcal { L } _ { p r o j }$ denote the gating convolution module and the projection transformation module, respectively, which are used to produce gating coefficients and align feature channels. For the remaining levels, we adopt a feature inheritance strategy from the pyramid:

$$
{ \hat { F } } _ { x , l } = F _ { x , l } ^ { p } , x \in \{ i r , v i s \} , l \notin S .\tag{14}
$$

Following the paradigm of classical dense matching methods, we adopt a coarse-to-fine flow estimation strategy, which is decomposed into global matching and local matching stages. First, global correlation is computed on deep features to estimate the coarse flow field:

$$
{ \mathbf { u } } _ { L } = \mathcal { M } _ { g } ( C o r r _ { g l o b a l } ( \hat { F } _ { i r , L } , \hat { F } _ { v i s , L } ) ) ,\tag{15}
$$

where $C o r r _ { g l o b a l }$ denotes the global correspondence. $\mathcal { M } _ { g }$ represents the mapping operation of consistency matching and correlation-to-flow. Subsequently, the coarse flow field is progressively upsampled and refined via local correlation, which can be formulated as:

$$
\mathbf { u } _ { l } = U p ( \mathbf { u } _ { l + 1 } ) + \Delta \mathbf { u } _ { l } , l = L - 1 , \ldots , 1 ,\tag{16}
$$

$$
\Delta \mathbf { u } _ { l } = \mathcal { M } _ { l } ( C o r r _ { l o c a l } ( F _ { v i s , l } , w a r p ( F _ { i r , l } , U p ( \mathbf { u } _ { l + 1 } ) ) ) ) ,\tag{17}
$$

where $C o r r _ { l o c a l }$ denotes the local correspondence. $\mathcal { M } _ { l }$ represents the mapping operation of local correlation-to-flow. Through progressive iterative refinement, the final flow field is estimated. Therefore, we utilize the full-resolution flow field $\mathrm { \mathbf { u } } ^ { f }$ to warp the source image and obtain the registered image:

$$
\widetilde { I } _ { i r } = w a r p ( I _ { i r } , \mathbf { u } ^ { f } ) .\tag{18}
$$

To explicitly model the effective overlapping field of view after registration, we estimate a validity field-of-view mask based on the flow field. Specifically, pixel coordinates $\mathbf { p } =$ $( x , y )$ are mapped to new locations using the full-resolution flow field $\mathrm { \mathbf { u } } ^ { f }$

$$
q ( \mathbf { p } ) = \mathbf { p } + \mathbf { u } ^ { f } ( \mathbf { p } ) .\tag{19}
$$

Subsequently, the validity field-of-view mask is obtained by determining whether the mapped coordinates fall within the valid boundary of the corresponding image:

$$
C ^ { f } ( \mathbf { p } ) = \left\{ { 1 , \begin{array} { r l } { 0 \leq q _ { x } ( \mathbf { p } ) < W , \quad 0 \leq q _ { y } ( \mathbf { p } ) < H , } \\ { 0 , } & { { \mathrm { o t h e r w i s e } } , } \end{array} } \right.\tag{20}
$$

where H and $W$ are the height and width of the full-resolution image, respectively. In this way, $C ^ { f }$ can explicitly characterize the geometrically valid region defined by the estimated flow field.

![](images/454724d55438a2f0b46ac3a4530398c94c363d44d3a550015376338370748e6a.jpg)  
Fig. 5. Schematic diagram of the generative diffusion module for multi-modal completion and fusion.

## D. Generative Diffusion Module for Multi-Modal Completion and Fusion

The generative diffusion module aims to address field-ofview inconsistency, which arises from factors such as sensor characteristics and relative sensor configurations. It is tightly coupled with MIR through the registered source image and the validity field-of-view mask, and achieves field-of-view unification and fusion using a diffusion model with both generative and fusion capabilities.

We adopt a customized multi-modal latent diffusion model based on the Markov chain formulation of diffusion models [30]. By designing both the forward and reverse diffusion processes, image fusion is achieved. First, the pre-aligned source images with a complete field of view $I _ { i r } ^ { f u l l } , I _ { v i s } ^ { f u l l }$ is passed through an existing pretrained fusion network θ<sup>teacher</sup> f to obtain a full-field fused image:

$$
I _ { f } ^ { f u l l } = \theta _ { f } ^ { t e a c h e r } ( I _ { i r } ^ { f u l l } , I _ { v i s } ^ { f u l l } ) ,\tag{21}
$$

where $I _ { f } ^ { f u l l }$ is the full-field fused image. The proposed method facilitates prior knowledge transfer through distillation and integration, enabling the generation of complete fusion images covering the entire field of view, as shown in Fig. 5. In detail, it is encoded using a VAE $z _ { 0 } = \mathcal { E } _ { f u l l } ( I _ { f } ^ { f u l l } )$ , and the forward diffusion process is formulated as:

$$
\begin{array} { r } { q ( z _ { t } | z _ { 0 } ) = \mathcal { N } ( z _ { t } ; \sqrt { \overline { { \alpha _ { t } } } } z _ { 0 } , ( 1 - \overline { { \alpha } } _ { t } ) \mathbf { I } ) , } \end{array}\tag{22}
$$

where $\begin{array} { r } { \overline { { \alpha } } _ { t } = \prod _ { i = 0 } ^ { t } \alpha _ { i } , \alpha _ { t } = 1 - \beta _ { t } } \end{array}$ , which follows the scheduler of simple diffusion model. I denotes the identity covariance matrix. Given the registered source image $I _ { v i s } , \widetilde { I _ { i r } }$ and the valid field-of-view mask $C ^ { f }$ , The source images are encoded into a latent space using the encoder of VAEs:

$$
z _ { i r } = \mathcal { E } _ { i r } \big ( \widetilde { I } _ { i r } \big ) , \quad z _ { v i s } = \mathcal { E } _ { v i s } \big ( I _ { v i s } \big ) ,\tag{23}
$$

where $z _ { i r }$ and $z _ { v i s }$ denote the latent features in the compressed latent space. $\mathcal { E } _ { i r }$ and ${ \mathcal { E } } _ { v i s }$ are the VAE encoders of the latent diffusion model. Since the registered infrared images typically exhibit a smaller field of view, we treat the infrared image to be completed as the main branch of the diffusion model and fuse its feature information by concatenating it with the valid field-of-view mask. In contrast, the visible image provides consistent spatial structural guidance and is therefore introduced as a control branch through a cross-modal control encoding module into the diffusion process, enabling more accurate cross-modal constraints. The conditional control can be expressed as follows:

$$
z _ { c } = \psi _ { c } ( z _ { i r } , C ^ { f } ) ,\tag{24}
$$

where $\psi _ { c }$ denotes the concat operation. The visible image features are first processed by the cross-modal control encoder to extract multi-level representations $c _ { v i s } ^ { l }$ , which are then injected into the diffusion denoising network through zero convolutions, enabling visible-guided information generation and fusion:

$$
\widetilde { h } _ { d i f f } ^ { l } = h _ { d i f f } ^ { l } + Z e r o C o n v ( c _ { v i s } ^ { l } ) ,\tag{25}
$$

where $h _ { d i f f } ^ { l }$ denotes the features of the diffusion denoising network at the level l. Therefore, the optimization objective of the reverse diffusion process can be expressed as:

$$
\operatorname* { m i n } \| \epsilon - \epsilon _ { \theta } \big ( \sqrt { \overline { { \alpha } } _ { t } } z _ { 0 } + \sqrt { 1 - \overline { { \alpha } } _ { t } } \epsilon , z _ { c } , z _ { v i s } , t \big ) \| ,\tag{26}
$$

where ϵ is the noise added by the diffusion model. $\epsilon _ { \theta }$ denotes the denoising network. t represents the timestep. The final latent diffusion features are obtained through progressive iterative refinement $\hat { z } _ { 0 } .$ . To enable the model to learn both unimodal image completion and cross-modal control-guided completion, we first pretrain the diffusion model using full-field-of-view infrared images as the diffusion targets. Subsequently, we freeze the diffusion UNet backbone and train the cross-modal control encoder to generate the full-field-of-view fused images.

To enhance fine-grained texture details in the output fusion image, inspired by ControlNet [31], we design a control residual feature decoder (CRFD). Specifically, we duplicate the VAE encoder and make it trainable, termed the control residual feature encoder (CRFE), and inject the extracted features into a frozen pretrained decoder via residual connections and zero convolution. It can be expressed as:

$$
f _ { V A E } ^ { C R F E } = \mathcal { E } _ { C R F E } ( \psi _ { c } ( \widetilde { I } _ { i r } , I _ { v i s } , C ^ { f } ) ) ,\tag{27}
$$

$$
I _ { f } = { \mathcal { D } } _ { C R F D } ^ { * } ( \hat { z } _ { 0 } , f _ { V A E } ^ { C R F E } ) ,\tag{28}
$$

where E<sub>CRFE</sub> denotes the trainable control residual feature encoder. ${ \cal D } _ { C R F D } ^ { * }$ is the pretrained frozen control residual feature decoder. In particular, the decoder features and the residual injected features are coupled via zero convolution and addition, which can be expressed as:

$$
\widetilde { h } _ { d } ^ { l } = h _ { d } ^ { l } + Z e r o C o n v ( r _ { e } ^ { l } ) ,\tag{29}
$$

where $h _ { d } ^ { l }$ and $r _ { e } ^ { l }$ denote the features of the control residual feature decoder and $f _ { V A E } ^ { C R F E }$ at level l, respectively. ZeroConv is the zero convolution. During training, an approximate output estimation can be obtained using the results predicted by the denoising network:

$$
\widetilde { z } _ { 0 , t } = \frac { 1 } { \sqrt { \overline { { \alpha } } _ { t } } } \big ( z _ { t } - \sqrt { 1 - \overline { { \alpha } } _ { t } } \epsilon _ { \theta } \big ( z _ { t } , z _ { c } , z _ { v i s } , t \big ) \big ) .\tag{30}
$$

Subsequently, the fused image is obtained by estimating the latent representation of the approximate output $\begin{array} { r l } { \hat { I } _ { f } } & { { } = } \end{array}$ $\mathcal { D } _ { C R F D } ^ { * } ( \widetilde { z } _ { 0 , t } , \widetilde { f } _ { V A E } ^ { C R F E } )$ , and the optimization objective is defined as:

$$
\operatorname* { m i n } { \| I _ { f } ^ { f u l l } - \hat { I } _ { f } \| } .\tag{31}
$$

By jointly optimizing the latent diffusion model and the control residual encoder, we achieve both image fusion and field-of-view consistent completion.

## E. Loss Functions

To effectively optimize the network, we construct the image registration loss and the image fusion loss. The image registration loss is applied to the multi-modal image registration module and is enforced through flow-based constraints. The image fusion loss is applied to the generative diffusion module for multi-modal completion and fusion, and is supervised via the noise prediction and output consistency of the diffusion model.

For Image Registration: The flow supervision loss is applied in the image registration stage. By constraining multi-scale flow predictions with the ground-truth flow, the MIR network is optimized. It can be expressed as:

$$
L _ { r e g } = \sum _ { l = 1 } ^ { N } \| \mathbf { u } _ { l } ^ { g t } - \mathbf { u } _ { l } \| ,\tag{32}
$$

where $\mathbf { u } _ { l } ^ { g t }$ represents the ground truth flow field at the l-th level. Correspondingly, u<sub>l</sub> is the estimated flow field predicted by the network at the l-th level.

For Image Fusion: The latent-space diffusion loss and the fused image consistency loss are introduced to constrain the image fusion results.

In particular, the latent-space diffusion loss enforces supervision on the denoising network output of the diffusion model, enabling generative completion and fusion:

$$
L _ { f u s - d i f f } = \sum _ { l = 1 } ^ { N } \| \epsilon - \epsilon _ { \theta } \big ( \sqrt { \overline { { \alpha } } _ { t } } z _ { 0 } + \sqrt { 1 - \overline { { \alpha } } _ { t } } \epsilon , z _ { c } , z _ { v i s } , t \big ) \| .\tag{33}
$$

The fused image consistency loss is constructed by constraining the distillation values between the generated fused image and the full-view fused image:

$$
L _ { f u s - V A E } = \| I _ { f } ^ { f u l l } - \hat { I } _ { f } \| ,\tag{34}
$$

where $I _ { f } ^ { f u l l }$ denotes the full-view fused image, while $\hat { I } _ { f }$ represents the fused image estimated by the network as the completion result.

## IV. EXPERIMENTS

In this section, we first present the implementation details and the corresponding task settings. We then conduct comprehensive qualitative and quantitative evaluations to validate the effectiveness and superiority of the proposed method. Finally, we perform ablation studies to demonstrate the contribution of each individual module.

Implementation Details: We implement the SeeRF with the PyTorch [32] in the machine with four NVIDIA GeForce RTX 4090 GPUs. The optimizer is AdamW [33] with a learning rate of 0.0001. For the image registration task, the batch size is 8 and the number of epochs is 800. For the image fusion task, gradient accumulation is used with an effective batch size of 16, and the number of epochs is 400. For the image registration task, we use a patch size of $5 2 0 \times 5 2 0 .$ while for the image fusion task, the original image resolution is used.

Datasets: To validate the effectiveness of the proposed method, we conduct extensive experiments on widely used infrared and visible image fusion datasets, including MFNet [34], FMB [35], and LLVIP [36] datasets. For the MFNet dataset, we utilize 784 infrared–visible image pairs for training and 393 pairs for testing and evaluation. For the FMB dataset, 1220 image pairs are used for training, while 280 pairs are reserved for testing. Similarly, for the LLVIP dataset, we employ 1973 image pairs for training and 432 pairs for testing. For deformation generation, we synthesize geometric transformations using homographic transformations and elastic deformations. Specifically, the rotation angle of the homographic transformation is sampled from $[ - 4 ^ { \circ } , 4 ^ { \circ } ]$ , while the horizontal and vertical translations are constrained to 8% of the image width and height, respectively. For an image of size of $6 4 0 \times 4 8 0 .$ , these constraints correspond to maximum displacements of approximately 51 pixels horizontally and 38 pixels vertically. For elastic deformation, the displacement components along both the horizontal and vertical directions are bounded by 5 pixels. Additionally, we conduct experiments on real-world datasets to demonstrate the practicality of the proposed method. Through experiments on relatively comprehensive datasets, the performance of the method can be effectively verified.

Metric: We adopt a comprehensive set of evaluation metrics to thoroughly assess the performance of both registration and fusion. For the cross-modal image registration, we employ the end-point error (EPE) [37] metric to evaluate the accuracy of the estimated flow correspondence, which directly measures the displacement error between the predicted and ground-truth flow fields. Additionally, mean absolute error (MAE), mutual information (MI) [38], correlation coefficient (CC) [39], and structural similarity index (SSIM) [40] are used to evaluate the quality of the registered images by comparing them with the ground-truth images. Among these metrics, lower EPE and MAE values indicate better registration accuracy, whereas higher MI, CC, and SSIM values correspond to more precise image alignment. For the image fusion, we employ standard deviation (SD), visual information fidelity (VIF) [41], correlation coefficient (CC), edge information preservation metric $Q ^ { A B / F }$ [42], and structural similarity index measure (SSIM) metrics to evaluate the quality of the fused images. Note that all metrics are evaluated using the full-field-of-view source images that cover the entire scene. SD, VIF, CC, $Q ^ { A B / F }$ , and SSIM are all metrics where larger values indicate better fused image quality.

SOTA Competitors: To validate the effectiveness of the proposed method, we conduct systematic experiments on multiple public datasets and compare it with existing stateof-the-art approaches. For the image registration task, we select several representative methods as baselines, including RIFT [43], GLUNet [44], and CrossRAFT [45]. For the image fusion task, we compare our method against U2Fusion [16] and DDFM [21]. Furthermore, to comprehensively evaluate the performance of the proposed framework in joint image registration and fusion, we include a broader range of representative and advanced methods for comparison, including UMF-CMGR [27], MURF [10], SuperFusion [24], AUNet [11], MulFS-CAP [12], and CPMFusion [46].

## A. Results and Analyses

Qualitative Results on Image Registration: To provide a more comprehensive evaluation of image registration perfor mance, we visualize the gradient features of the visible images and overlay them onto the registered infrared images for qualitative analysis. The registration quality is assessed from two perspectives: first, the structural similarity between the registered infrared images and the ground-truth infrared images; second, the spatial consistency between salient infrared edges and visible-image gradient responses. A registration method is considered more accurate when the registered results preserve higher structural and detailed consistency with the ground truth while exhibiting stronger spatial correspondence between infrared edge regions and visible gradients, indicating more precise cross-modal spatial alignment. The visualization results are presented in Fig. 6. In the first scene, benefiting from the effective cross-modal registration design, SeeRF accurately aligns the human targets and road edge structures, demonstrating superior registration performance. In contrast, existing competing methods suffer from varying degrees of registration errors, leading to noticeable misalignment in target regions. Such discrepancies are particularly evident in critical structural areas, including human contours and road boundaries, where substantial spatial deviations can be observed. Similarly, in the second and third groups of scenes, CrossRAFT demonstrates relatively strong registration performance in salient regions, including street lamps, vegetation, and vehicles, and is able to establish relatively accurate structural correspondences. However, due to its limited capability in handling complex scenes and boundary regions, its results still exhibit noticeable edge mismatches and local geometric distortions. Constrained by their registration capabilities, UMF-CMGR, MulFS-CAP, and CPMFusion fail to achieve effective alignment in these critical regions, resulting in considerable spatial discrepancies. Although SuperFusion and AUNet can capture certain cross modal structural consistencies, they still suffer from local mis alignment, indicating that their registration accuracy remains suboptimal. In comparison, the proposed method achieves the best performance in both fine-grained object alignment and global spatial consistency. In the fourth scene, CrossRAFT, SuperFusion, and AUNet achieve relatively accurate alignment in the stone pillar regions; however, their spatial correction capability for pedestrian targets remains limited, leading to no ticeable deformation and structural distortion in these regions. In contrast, UMF-CMGR, MulFS-CAP, and CPMFusion fail to establish effective cross-modal spatial alignment, resulting in significant registration errors. Benefiting from the proposed registration strategy, SeeRF simultaneously preserves finegrained object details and global scene structures, achieving superior overall registration performance in these scenarios.

Quantitative Results on Image Registration: The quantitative results are summarized in Tab. I. The experimental results demonstrate that the proposed SeeRF achieves the best overall performance on the MFNet, FMB, and LLVIP datasets across multiple evaluation metrics, including the optical flow estimation error (EPE) and the MAE, MI, CC, and SSIM metrics between the registration results and reference images. These results indicate that SeeRF is capable of estimating more accurate flow fields and producing registered images with stronger structural consistency with the references. Overall, these experiments validate the superiority of the proposed method for cross-modal image registration and further demonstrate its effectiveness and robustness.

![](images/da81d6d67fb76ce39e62cc2d2b242e675469234e878e72416b2a52a2aaddc9fd.jpg)  
Fig. 6. Qualitative comparison of SeeRF with state-of-the-art image registration methods on the MFNet (the first group), FMB (the second and third groups), and LLVIP (the fourth and fifth groups) datasets for image registration. We visualize and overlay the gradient of the visible image in purple. Better registration results exhibit higher spatial consistency between the visible image gradients and infrared structures, while maintaining closer object contours and detailed structures compared with the infrared ground-truth images.

TABLE I  
QUANTITATIVE COMPARISON OF OUR SEERF WITH EXISTING STATE-OF-THE-ART METHODS OF THE IMAGE REGISTRATION TASK ON THE MFNET, FMB, AND LLVIP DATASETS. (BOLD: OPTIMAL PERFORMANCE)
<table><tr><td rowspan="2">Methods</td><td colspan="5">MFNet Dataset</td><td colspan="5">FMB Dataset</td><td colspan="5">LLVIP Dataset</td></tr><tr><td>EPE</td><td>MAE</td><td>MI</td><td>CC</td><td>SSIM</td><td>EPE</td><td>MAE</td><td>MI</td><td>CC</td><td>SSIM</td><td>EPE</td><td>MAE</td><td>MI</td><td>CC</td><td>SSIM</td></tr><tr><td>RIFT</td><td>9.9737</td><td>0.0252</td><td>1.2731</td><td>0.8014</td><td>0.8243</td><td>7.3479</td><td>0.0246</td><td>2.1123</td><td>0.9115</td><td>0.8354</td><td>6.3640</td><td>0.0456</td><td>1.3729</td><td>0.8503</td><td>0.6646</td></tr><tr><td>GLUNet</td><td>15.6251</td><td>0.0282</td><td>1.2056</td><td>0.7272</td><td>0.8156</td><td>11.8658</td><td>0.0272</td><td>2.0398</td><td>0.8933</td><td>0.8297</td><td>11.4010</td><td>0.0503</td><td>1.3036</td><td>0.8292</td><td>0.6588</td></tr><tr><td>CrossRAFT</td><td>8.5285</td><td>0.0224</td><td>1.3560</td><td>0.7977</td><td>0.8419</td><td>10.5699</td><td>0.0319</td><td>2.0789</td><td>0.8523</td><td>0.8236</td><td>6.3431</td><td>0.0414</td><td>1.4830</td><td>0.8736</td><td>0.7037</td></tr><tr><td>MURF</td><td>119.0738</td><td>0.1514</td><td>0.5892</td><td>0.3273</td><td>0.5562</td><td>93.1354</td><td>0.0958</td><td>1.2406</td><td>0.5864</td><td>0.6507</td><td>197.5209</td><td>0.1419</td><td>0.5718</td><td>0.4595</td><td>0.3734</td></tr><tr><td>UMF-CMGR</td><td>35.5214</td><td>0.0651</td><td>0.6568</td><td>0.2784</td><td>0.6911</td><td>44.9324</td><td>0.0983</td><td>1.1564</td><td>0.4800</td><td>0.6399</td><td>36.4979</td><td>0.1283</td><td>0.4790</td><td>0.4109</td><td>0.3939</td></tr><tr><td>SuperFusion</td><td>16.4511</td><td>0.0395</td><td>1.1170</td><td>0.5449</td><td>0.7844</td><td>30.2249</td><td>0.0685</td><td>1.4927</td><td>0.6466</td><td>0.7091</td><td>17.2222</td><td>0.0797</td><td>0.9676</td><td>0.6726</td><td>0.5499</td></tr><tr><td>AUNet</td><td>19.7142</td><td>0.0407</td><td>0.9784</td><td>0.5424</td><td>0.7715</td><td>30.8708</td><td>0.0666</td><td>1.4713</td><td>0.6474</td><td>0.7172</td><td>18.6910</td><td>0.0841</td><td>0.8538</td><td>0.6416</td><td>0.5244</td></tr><tr><td>MulFS-CAP</td><td>36.4609</td><td>0.0643</td><td>0.6533</td><td>0.3046</td><td>0.6903</td><td>45.7767</td><td>0.0976</td><td>1.1451</td><td>0.4893</td><td>0.6385</td><td>37.3990</td><td>0.1278</td><td>0.4761</td><td>0.4159</td><td>0.3928</td></tr><tr><td>CPMFusion</td><td>19.0854</td><td>0.0442</td><td>0.9312</td><td>0.4944</td><td>0.7564</td><td>30.0090</td><td>0.0704</td><td>1.4090</td><td>0.6463</td><td>0.6964</td><td>19.0808</td><td>0.0910</td><td>0.7762</td><td>0.6173</td><td>0.4906</td></tr><tr><td>SeeRF</td><td>3.8548</td><td>0.0119</td><td>1.6698</td><td>0.9279</td><td>0.9010</td><td>4.3088</td><td>0.0166</td><td>2.3674</td><td>0.9521</td><td>0.8809</td><td>2.0404</td><td>0.0221</td><td>2.0516</td><td>0.9507</td><td>0.8524</td></tr></table>

Qualitative Results on Image Fusion: The corresponding qualitative visualization results are presented in Fig. 7. Our method exhibits two significant advantages. First, SeeRF is able to generate more complete and natural fused scenes. Owing to the field-of-view discrepancy between infrared and visible sensors, certain target regions in the registered images may appear in only one modality, which causes existing methods to suffer from information loss, structural discontinuities, and unnatural visual transitions during the fusion process. For instance, in the first scene, part of the infrared information of the pedestrian within the purple box is missing. As most competing methods lack effective generative completion capability, they tend to produce obvious one-sided missing artifacts. Similarly, in non-overlapping field-of-view regions involving structures such as vehicle exhaust pipes, tires, road surfaces, handlebars, and feet, existing methods struggle to adequately model cross-modal field-of-view relationships, often resulting in structural omissions and fusion artifacts. In contrast, by combining an accurate cross-modal registration strategy with a diffusion-based generative fusion mechanism, SeeRF can effectively learn and recover information from non-overlapping field-of-view regions. As a result, it restores more complete target structures while maintaining spatial consistency, thereby producing fused results that are more natural, continuous, and visually coherent. Second, SeeRF can generate clearer visual representations while effectively preserving key structures and detailed information from the source images. In the fence region of the first scene and the high-voltage transmission tower region of the third scene, most comparison methods fail to sufficiently preserve texture details and color information from the visible images, while also losing structural edge features of the high-voltage transmission tower, thereby reducing the overall visual quality and scene representation capability of the fused results. In contrast, SeeRF achieves a better balance among texture details, color information, and infrared thermal radiation features during the fusion process, producing highquality fusion results with clear structures, natural colors, and prominent thermal target information.

![](images/5b33a15ef79f155bc163f412c8dd3816286ec1981bf9f0d8400151cee7cf552b.jpg)  
Fig. 7. Qualitative comparison of SeeRF with state-of-the-art image registration and fusion methods on the MFNet (the first group), FMB (the second and third groups), and LLVIP (the fourth and fifth groups) datasets for image fusion. It is worth noting that the source images are also spatially misaligned.

Quantitative Results on Image Fusion: The quantitative results are summarized in Tab. II. The experimental results demonstrate that SeeRF achieves superior overall performance across multiple image fusion evaluation metrics, validating its effectiveness and robustness under diverse scenes and complex environments. On the MFNet dataset, SeeRF obtains the best performance on four metrics, including SD, VIF, CC, and Q<sup>AB/F</sup> , while maintaining competitive performance in terms of SSIM. Compared with the overall second-best performing method, Reg.+U2Fusion, SeeRF achieves improvements of 1.8385, 0.1962, 0.0398, and 0.1154 in SD, VIF, CC, and $Q ^ { A B / F }$ , respectively. These results demonstrate that the proposed method can better preserve salient information from the source images while enhancing the structural consistency and visual quality of the fused results. On the FMB dataset, the proposed SeeRF achieves the best overall performance, obtaining the optimal results on VIF, CC, and $\dot { Q } ^ { A B / F }$ , while maintaining competitive performance on SD and SSIM. Furthermore, on the LLVIP dataset, SeeRF ranks first across all evaluation metrics, demonstrating stable and consistent performance advantages. These results comprehensively validate the superior capability of SeeRF in enhancing fused-image contrast, preserving visual information fidelity, retaining salient information from source images, and maintaining structural consistency. They further demonstrate the effectiveness and robustness of the proposed method across diverse infrared–visible image fusion scenarios.

TABLE II  
QUANTITATIVE COMPARISON OF OUR SEERF WITH EXISTING STATE-OF-THE-ART METHODS OF THE IMAGE FUSION TASK ON THE MFNET, FMB, AND LLVIP DATASETS. (BOLD: OPTIMAL PERFORMANCE)
<table><tr><td rowspan="2">Methods</td><td colspan="5">MFNet Dataset</td><td colspan="5">FMB Dataset</td><td colspan="5">LLVIP Dataset</td></tr><tr><td>SD</td><td>VIF</td><td>CC</td><td>QAB/F</td><td>SSIM</td><td>SD</td><td>VIF</td><td>CC</td><td>QAB/F</td><td>SSIM</td><td>SD</td><td>VIF</td><td>CC</td><td>QAB/F</td><td>SSIM</td></tr><tr><td>Reg.+U2Fusion</td><td>33.2232</td><td>0.6445</td><td>0.5233</td><td>0.5218</td><td>0.7022</td><td>33.9225</td><td>0.5727</td><td>0.5578</td><td>0.5358</td><td>0.7107</td><td>32.2680</td><td>0.4173</td><td>0.6164</td><td>0.3491</td><td>0.6420</td></tr><tr><td>Reg.+DDFM</td><td>31.5535</td><td>0.6845</td><td>0.5237</td><td>0.5037</td><td>0.6309</td><td>30.8141</td><td>0.4964</td><td>0.5552</td><td>0.3663</td><td>0.6299</td><td>35.3449</td><td>0.4631</td><td>0.6129</td><td>0.3391</td><td>0.6489</td></tr><tr><td>FusionReg.</td><td>34.6501</td><td>0.5219</td><td>0.4410</td><td>0.4728</td><td>0.5280</td><td>29.0892</td><td>0.6338</td><td>0.5404</td><td>0.5768</td><td>0.7611</td><td>44.9475</td><td>0.3849</td><td>0.4727</td><td>0.3962</td><td>0.5297</td></tr><tr><td>MURF</td><td>24.0411</td><td>0.2911</td><td>0.3605</td><td>0.2584</td><td>0.3905</td><td>27.7418</td><td>0.2871</td><td>0.4605</td><td>0.3143</td><td>0.5463</td><td>26.3277</td><td>0.1604</td><td>0.3801</td><td>0.1586</td><td>0.3373</td></tr><tr><td>UMF-CMGR</td><td>24.2902</td><td>0.4119</td><td>0.4493</td><td>0.2914</td><td>0.4271</td><td>32.0586</td><td>0.3703</td><td>0.4557</td><td>0.2410</td><td>0.4628</td><td>30.9089</td><td>0.2641</td><td>0.4449</td><td>0.2434</td><td>0.3849</td></tr><tr><td>SuperFusion</td><td>34.8319</td><td>0.5855</td><td>0.5064</td><td>0.4926</td><td>0.6079</td><td>31.6595</td><td>0.6746</td><td>0.5466</td><td>0.5819</td><td>0.8306</td><td>43.4156</td><td>0.4185</td><td>0.5452</td><td>0.3698</td><td>0.6057</td></tr><tr><td>AUNet</td><td>27.7622</td><td>0.5081</td><td>0.5394</td><td>0.3423</td><td>0.4896</td><td>36.1326</td><td>0.3976</td><td>0.5442</td><td>0.3635</td><td>0.4900</td><td>36.5361</td><td>0.3592</td><td>0.5465</td><td>0.2475</td><td>0.4751</td></tr><tr><td>MulFS-CAP</td><td>24.3576</td><td>0.3118</td><td>0.4902</td><td>0.3893</td><td>0.2992</td><td>27.9480</td><td>0.2654</td><td>0.4967</td><td>0.3905</td><td>0.2945</td><td>31.9229</td><td>0.2082</td><td>0.4564</td><td>0.2941</td><td>0.3153</td></tr><tr><td>CPMFusion</td><td>34.7895</td><td>0.5202</td><td>0.4926</td><td>0.4780</td><td>0.4867</td><td>34.4127</td><td>0.6928</td><td>0.5329</td><td>0.5716</td><td>0.7922</td><td>45.9738</td><td>0.3772</td><td>0.5199</td><td>0.3632</td><td>0.5029</td></tr><tr><td>SeeRF (ours)</td><td>35.0617</td><td>0.8407</td><td>0.5631</td><td>0.6372</td><td>0.7007</td><td>35.7528</td><td>0.8307</td><td>0.6312</td><td>0.6459</td><td>0.8086</td><td>48.2873</td><td>0.7312</td><td>0.6366</td><td>0.5537</td><td>0.8184</td></tr></table>

![](images/f76c84ea9f808429f67f2cde575649f234a06e33a02014361bf2561cd887465f.jpg)

Fig. 8. Qualitative comparison of SeeRF with state-of-the-art image registration and fusion methods on the MFNet dataset for semantic segmentation task.  
![](images/15e5c212ede31b6f612ada6d7b04d3d2a27ac1f9bedbb27cb666921546781097.jpg)  
Fig. 9. Qualitative comparison of SeeRF with state-of-the-art image registration and fusion methods on the FMB dataset for semantic segmentation task.

## B. Downstream Task Performance Validation

Semantic Segmentation: For the semantic segmentation task, we adopt SegFormer-B2 [47] as the backbone network and conduct experiments on the MFNet and FMB datasets. The qualitative and quantitative experimental results are presented in Figs. 8–9 and Tabs. III–IV.

From the perspective of qualitative analysis, SeeRF demonstrates superior segmentation accuracy and stronger capability in preserving complete target structures in downstream semantic segmentation tasks. In contrast, due to the limitations of existing methods, competing methods are prone to target region missing, boundary misclassification, and incorrect category predictions, thereby reducing the ability of fused results to represent high-level semantic information.

From the quantitative evaluation results, SeeRF achieves the best mIoU performance on both the FMB and MFNet datasets, demonstrating its effectiveness for downstream semantic segmentation tasks. These results suggest that, benefiting from more accurate cross-modal alignment, effective information integration, and the capability to complete non-overlapping field-of-view regions, SeeRF is able to produce fused representations with richer semantic information and stronger consistency, thereby further enhancing semantic segmentation performance.

TABLE III  
QUANTITATIVE RESULTS OF MULTI-MODAL SEMANTIC SEGMENTATION VALIDATION FOR ALL CATEGORIES IN MFNET DATASET. (BOLD: OPTIMAL PERFORMANCE)
<table><tr><td>Methods</td><td>Bac.</td><td>Car</td><td>Per.</td><td>Bike</td><td>Curve</td><td>Car_stop</td><td>Guar.</td><td>Col.</td><td>Bump</td><td>mIoU</td></tr><tr><td>Reg.+U2Fusion</td><td>97.87</td><td>86.47</td><td>63.54</td><td>62.95</td><td>31.36</td><td>28.33</td><td>4.87</td><td>52.12</td><td>50.73</td><td>53.14</td></tr><tr><td>Reg.+DDFM</td><td>97.91</td><td>87.84</td><td>64.83</td><td>63.20</td><td>29.92</td><td>27.92</td><td>4.27</td><td>49.67</td><td>50.85</td><td>52.93</td></tr><tr><td>FusionReg</td><td>97.54</td><td>84.17</td><td>57.31</td><td>59.29</td><td>23.82</td><td>25.96</td><td>5.97</td><td>43.19</td><td>39.10</td><td>48.48</td></tr><tr><td>MURF</td><td>97.52</td><td>83.35</td><td>55.72</td><td>60.88</td><td>22.48</td><td>27.74</td><td>4.88</td><td>41.60</td><td>37.24</td><td>47.93</td></tr><tr><td>UMF-CMGR</td><td>97.67</td><td>84.86</td><td>58.59</td><td>61.09</td><td>27.17</td><td>27.31</td><td>3.15</td><td>42.46</td><td>47.67</td><td>50.00</td></tr><tr><td>SuperFusion</td><td>97.76</td><td>84.68</td><td>62.28</td><td>61.26</td><td>35.96</td><td>21.00</td><td>0.44</td><td>49.00</td><td>50.37</td><td>51.42</td></tr><tr><td>AUNet</td><td>97.67</td><td>84.30</td><td>60.57</td><td>61.31</td><td>28.95</td><td>27.41</td><td>7.58</td><td>49.77</td><td>47.29</td><td>51.65</td></tr><tr><td>MulFS-CAP</td><td>97.52</td><td>84.04</td><td>54.76</td><td>58.23</td><td>25.29</td><td>22.20</td><td>4.95</td><td>43.77</td><td>37.85</td><td>47.62</td></tr><tr><td>CPMFusion</td><td>97.68</td><td>84.57</td><td>59.11</td><td>59.46</td><td>29.31</td><td>16.64</td><td>2.97</td><td>45.84</td><td>41.44</td><td>48.56</td></tr><tr><td>SeeRF (ours)</td><td>98.08</td><td>87.43</td><td>68.85</td><td>64.00</td><td>43.24</td><td>25.03</td><td>3.17</td><td>46.16</td><td>52.50</td><td>54.27</td></tr></table>

TABLE IV

QUANTITATIVE RESULTS OF MULTI-MODAL SEMANTIC SEGMENTATION VALIDATION FOR ALL CATEGORIES IN FMB DATASET. (BOLD: OPTIMAL PERFORMANCE)
<table><tr><td>Methods</td><td>Bac.</td><td>Road</td><td>Sid.</td><td>Buil.</td><td>Lamp</td><td>Sign</td><td>Veg.</td><td>Sky</td><td>Per.</td><td>Car</td><td>Trunk</td><td>Bus</td><td>Mot.</td><td>Bic.</td><td>Pole</td><td>mIoU</td></tr><tr><td>Reg.+U2Fusion</td><td>15.74</td><td>88.24</td><td>48.91</td><td>82.16</td><td>24.34</td><td>74.44</td><td>86.44</td><td>94.05</td><td>56.54</td><td>78.71</td><td>29.19</td><td>23.07</td><td>36.51</td><td></td><td>48.51</td><td>56.20</td></tr><tr><td>Reg.+DDFM</td><td>15.29</td><td>88.06</td><td>47.16</td><td>81.63</td><td>29.79</td><td>74.53</td><td>86.29</td><td>94.04</td><td>55.95</td><td>78.25</td><td>22.84</td><td>20.83</td><td>37.96</td><td></td><td>48.89</td><td>55.82</td></tr><tr><td>FusionReg</td><td>16.14</td><td>86.75</td><td>48.53</td><td>80.05</td><td>30.36</td><td>73.76</td><td>84.86</td><td>93.41</td><td>43.77</td><td>79.19</td><td>28.48</td><td>20.49</td><td>37.56</td><td></td><td>48.06</td><td>55.10</td></tr><tr><td>MURF</td><td>16.20</td><td>87.14</td><td>45.97</td><td>78.53</td><td>22.39</td><td>72.80</td><td>84.02</td><td>92.70</td><td>44.35</td><td>77.25</td><td>18.84</td><td>14.53</td><td>32.05</td><td></td><td>47.04</td><td>48.92</td></tr><tr><td>UMF-CMGR</td><td>15.70</td><td>87.46</td><td>46.30</td><td>78.76</td><td>26.56</td><td>72.41</td><td>84.47</td><td>93.09</td><td>43.57</td><td>78.11</td><td>21.60</td><td>16.99</td><td>37.76</td><td></td><td>46.61</td><td>53.53</td></tr><tr><td>SuperFusion</td><td>15.77</td><td>86.53</td><td>47.39</td><td>80.81</td><td>25.31</td><td>73.55</td><td>85.05</td><td>93.47</td><td>45.20</td><td>78.32</td><td>25.82</td><td>20.44</td><td>29.04</td><td></td><td>48.05</td><td>53.91</td></tr><tr><td>AUNet</td><td>16.19</td><td>86.88</td><td>46.07</td><td>79.36</td><td>28.32</td><td>72.67</td><td>84.89</td><td>93.30</td><td>44.71</td><td>78.90</td><td>24.67</td><td>13.79</td><td>34.11</td><td></td><td>46.84</td><td>53.62</td></tr><tr><td>MulFS-CAP</td><td>14.67</td><td>86.75</td><td>42.83</td><td>76.77</td><td>28.12</td><td>72.08</td><td>83.01</td><td>92.15</td><td>38.52</td><td>77.22</td><td>21.36</td><td>13.52</td><td>26.04</td><td></td><td>44.63</td><td>51.26</td></tr><tr><td>CPMFusion</td><td>15.46</td><td>86.40</td><td>44.17</td><td>80.45</td><td>30.25</td><td>72.75</td><td>85.32</td><td>93.68</td><td>47.98</td><td>76.10</td><td>19.90</td><td>19.20</td><td>25.96</td><td></td><td>47.33</td><td>53.21</td></tr><tr><td>SeeRF (ours)</td><td>16.67</td><td>88.99</td><td>48.51</td><td>82.52</td><td>33.94</td><td>74.80</td><td>86.99</td><td>94.32</td><td>61.55</td><td>79.90</td><td>30.28</td><td>21.13</td><td>35.66</td><td></td><td>49.57</td><td>57.49</td></tr></table>

![](images/0edb108871fbd6efa8d532d5b518530ed7ff6d692af1df95cc539100f021e809.jpg)  
Fig. 10. Qualitative comparison of SeeRF with state-of-the-art image registration and fusion methods on the LLVIP dataset for object detection task.

Object Detection: For the object detection task, we employ YOLOv11<sup>1</sup> as the evaluation backbone and initialize it with pretrained weights. For the fused images produced by each image fusion method, the object detector is retrained on the corresponding datasets to ensure a fair comparison. The qualitative and quantitative results are presented in Fig. 10 and Tab. V, respectively. From a qualitative perspective, constrained by limited viewpoint alignment and field-of-view correspondence capabilities, almost all competing methods suffer from missed detections. Moreover, U2Fusion, MURF, and SuperFusion produce duplicate detections and false positives in some sets of results. In contrast, the proposed method achieves the most comprehensive detection performance, demonstrating superior scene representation capability. From a quantitative perspective, the proposed SeeRF consistently achieves the best performance across all evaluation metrics, including Precision, Recall, mAP@50, and mAP@50:95. Compared with the second-best performing method, U2Fusion, SeeRF obtains improvements of 0.038, 0.031, 0.057, and 0.054 in Precision, Recall, mAP@50, and mAP@50:95, respectively. These results clearly demonstrate the effectiveness of the proposed method in facilitating downstream semantic segmentation tasks.

TABLE V  
QUANTITATIVE COMPARISON OF OBJECT DETECTION ON THE LLVIP DATASET. (BOLD: OPTIMAL PERFORMANCE)
<table><tr><td>Methods</td><td>Precision</td><td>Recall</td><td>mAP@50</td><td>mAP@50:95</td></tr><tr><td>Reg.+U2Fusion</td><td>0.869</td><td>0.737</td><td>0.814</td><td>0.408</td></tr><tr><td>Reg.+DDFM</td><td>0.854</td><td>0.717</td><td>0.797</td><td>0.392</td></tr><tr><td>FusionReg.</td><td>0.576</td><td>0.445</td><td>0.493</td><td>0.179</td></tr><tr><td>MURF</td><td>0.688</td><td>0.532</td><td>0.596</td><td>0.246</td></tr><tr><td>UMF-CMGR</td><td>0.596</td><td>0.490</td><td>0.531</td><td>0.207</td></tr><tr><td>SuperFusion</td><td>0.652</td><td>0.565</td><td>0.571</td><td>0.242</td></tr><tr><td>AUNet</td><td>0.707</td><td>0.574</td><td>0.633</td><td>0.276</td></tr><tr><td>MulFS-CAP</td><td>0.527</td><td>0.368</td><td>0.395</td><td>0.137</td></tr><tr><td>CPMFusion</td><td>0.530</td><td>0.484</td><td>0.487</td><td>0.177</td></tr><tr><td>SeeRF (ours)</td><td>0.907</td><td>0.768</td><td>0.871</td><td>0.462</td></tr></table>

![](images/2ca886fda68243e3d1b7671d40d379ea5a1ee2b737af801397fc580497708579.jpg)  
Fig. 11. Visualization of qualitative comparison results in the ablation study.

TABLE VI  
QUANTITATIVE COMPARISON OF THE ABLATION EXPERIMENT OF THE GENERAL COMPONENTS ON MFNET. (BOLD: OPTIMAL PERFORMANCE)
<table><tr><td>DGS</td><td>FMG</td><td>CTF</td><td>EPE</td><td>MAE</td><td>MI</td><td>CC</td><td>SSIM</td></tr><tr><td>√</td><td></td><td>√</td><td>4.0561</td><td>0.0127</td><td>1.6216</td><td>0.9220</td><td>0.8944</td></tr><tr><td></td><td>√</td><td>√</td><td>3.9055</td><td>0.0132</td><td>1.6204</td><td>0.9135</td><td>0.8910</td></tr><tr><td>√</td><td>√</td><td></td><td>10.6553</td><td>0.0248</td><td>1.1675</td><td>0.7887</td><td>0.8175</td></tr><tr><td>√</td><td>√</td><td>√</td><td>3.8548</td><td>0.0119</td><td>1.6698</td><td>0.9279</td><td>0.9010</td></tr><tr><td>VFM</td><td>CCE</td><td>CRFE</td><td>SD</td><td>VIF</td><td>CC</td><td>QAB/F</td><td>SSIM</td></tr><tr><td>√</td><td>√</td><td></td><td>33.4954</td><td>0.4360</td><td>0.5569</td><td>0.2890</td><td>0.4169</td></tr><tr><td>√</td><td></td><td>√</td><td>21.5992</td><td>0.5060</td><td>0.5091</td><td>0.4544</td><td>0.5507</td></tr><tr><td></td><td>√</td><td>√</td><td>36.1429</td><td>0.8180</td><td>0.5460</td><td>0.6261</td><td>0.6908</td></tr><tr><td>√</td><td>√</td><td>√</td><td>35.0617</td><td>0.8407</td><td>0.5631</td><td>0.6372</td><td>0.7007</td></tr></table>

## C. Ablation Study

General Experiments: To comprehensively validate the effectiveness of each proposed component, we conduct a series of ablation studies and systematically analyze the contribution of individual components from both image registration and image fusion perspectives. For the MIR, we conduct ablation experiments by removing the foundation model (DINOv3) Guidance mechanism (w/o FMG), replacing the proposed diffeomorphic deformation generation strategy with a conventional random elastic deformation generation method (w/o DGS), and substituting the coarse-to-fine progressive flow regression strategy with direct flow prediction (w/o CTF). For the MCF, we individually remove the control residual feature encoder (w/o CRFE), the cross-modal control encoding module (w/o CCE), and the valid field-of-view mask mechanism (w/o VFM). By ablating the valid field-of-view mask (VFM), we disable the guidance for generative completion, enabling a controlled comparison to evaluate the effect of out-of-field content generation. These experiments aim to systematically investigate the contribution of each component to registration accuracy and fusion quality. The results are presented in Fig. 11 and Tab. VI.

From the perspective of image registration performance, removing any individual component leads to degradation in the final registration results. Specifically, qualitative comparisons show that variants without specific modules tend to exhibit larger spatial offsets, edge misalignment, and structural inconsistencies. Quantitative evaluations further reveal increased in EPE and MAE, accompanied by noticeable declines in MI, CC, and SSIM between the registered and reference images. These results demonstrate the necessity of each proposed component for accurate displacement field modeling, effective cross-modal structural alignment, and improved registration quality.

From the perspective of joint image registration and fusion, qualitative results demonstrate that removing CRFE leads to noticeable structural distortions and loss of fine-grained details in the fused scenes. Without CCE, the weakened cross-modal guidance injection capability results in unnatural color transitions and insufficient information integration. Meanwhile, after removing VFM and preventing the diffusion model from generating content in non-overlapping field-of-view regions, we observe a significant degradation in fusion quality outside the shared field of view, with the overall results exhibiting unnatural discontinuities and incomplete scene representations. Quantitative evaluations further show that, except for the SD metric, the complete model achieves the best performance across all other evaluation metrics. These results further validate the critical contributions of the proposed components in improving fusion quality and enhancing the generative capability of the framework.

Semantic Experiments: To investigate the effects of individual components and out-of-field content generation on downstream semantic segmentation performance, we train all model variants on the MFNet dataset under the same experimental settings as those used in the semantic segmentation task. The corresponding qualitative and quantitative results are presented in Fig. 12 and Tab. VII, respectively.

From the qualitative results, the absence of modality specific information in non-overlapping field-of-view regions and the resulting unnatural fusion discontinuities negatively affect the completeness of scene semantic representation. Specifically, the bump target within the red box exhibits obvious structural truncation in the segmentation results, failing to preserve its complete semantic contour. Meanwhile, the corresponding fusion visualization also exhibits poor visual continuity and representation quality. These observations indicate that information loss caused by field-of-view inconsistency not only degrades the visual quality of fused images but also compromises the structural integrity of objects, thereby affecting the accurate recognition and complete representation of targets in downstream semantic segmentation tasks. Similarly, removing either CRFE or CCE impairs the model’s ability to capture fine-grained features and effectively integrate multimodal information, consequently degrading its performance on downstream semantic segmentation tasks. These findings further substantiate the motivation behind our approach and demonstrate the important contributions of CRFE and CCE to improving fusion quality and enhancing semantic representation for downstream tasks.

TABLE VII  
QUANTITATIVE RESULTS OF MULTI-MODAL SEMANTIC SEGMENTATION VALIDATION IN ABLATION STUDY FOR ALL CATEGORIES IN MFNET DATASET. (BOLD: OPTIMAL PERFORMANCE)
<table><tr><td>Methods</td><td>Bac.</td><td>Car</td><td>Per.</td><td>Bike</td><td>Curve</td><td>Car_stop</td><td>Guar.</td><td>Col.</td><td>Bump</td><td>mIoU</td></tr><tr><td>w/o_CRFE</td><td>97.81</td><td>83.91</td><td>64.41</td><td>59.86</td><td>38.87</td><td>25.73</td><td>4.00</td><td>40.16</td><td>48.90</td><td>51.52</td></tr><tr><td>w/o_CCE</td><td>97.89</td><td>84.91</td><td>64.61</td><td>62.20</td><td>39.40</td><td>26.91</td><td>3.42</td><td>47.75</td><td>51.65</td><td>53.19</td></tr><tr><td>w/o_VFM</td><td>97.92</td><td>85.73</td><td>67.97</td><td>62.62</td><td>38.68</td><td>24.05</td><td>3.96</td><td>47.27</td><td>54.71</td><td>53.66</td></tr><tr><td>SeeRF (ours)</td><td>98.08</td><td>87.43</td><td>68.85</td><td>64.00</td><td>43.24</td><td>25.03</td><td>3.17</td><td>46.16</td><td>52.50</td><td>54.27</td></tr></table>

![](images/813bb4a07c18fa31308de495ca6795db4482a603156808f43f68b3254c6fa9cd.jpg)  
Fig. 12. Qualitative comparison of semantic segmentation results in the ablation study.

The quantitative results show that disabling content generation beyond the shared field of view leads to a distinct decrease in segmentation mIoU, highlighting the importance of generative completion in addressing field-of-view limitations. Similarly, removing either CRFE or CCE results in degraded semantic segmentation performance, further demonstrating the contributions of these modules to downstream task effectiveness. Overall, the proposed SeeRF achieves the best performance, consistently outperforming all ablation variants by clear margins.

## D. Generative Credibility Analysis Experiment

To evaluate the feasibility of full-field-of-view image fusion, we independently assess the similarity between the generated content in non-overlapping field-of-view regions and the corresponding ground-truth images from the original modalities. Given the nature of this similarity assessment task, we adopt evaluation metrics similar to those used for image registration and additionally include peak signal-to-noise ratio (PSNR) to quantify the fidelity of the generated content. In the setting where infrared information is missing, poor metric values indicate that the generated regions either lack meaningful infrared information or fail to faithfully represent the actual infrared scene. Conversely, stronger similarity between the generated content and the ground-truth scene corresponds to better metric performance. Accordingly, higher MI, CC, PSNR, and SSIM values indicate more reliable and accurate generation, whereas a lower MAE indicates smaller reconstruction errors and better generation quality.

TABLE VIII  
QUANTITATIVE RESULTS OF GENERATIVE CREDIBILITY ANALYSIS BETWEEN GENERATED REGIONS AND THE REAL MODALITY IN THE MFNET DATASET. (BOLD: OPTIMAL PERFORMANCE)
<table><tr><td>Methods</td><td>MAE</td><td>MI</td><td>CC</td><td>PSNR</td><td>SSIM</td></tr><tr><td>w/o GEN.</td><td>0.1379</td><td>0.5440</td><td>0.0535</td><td>15.8329</td><td>0.3983</td></tr><tr><td>SeeRF (ours)</td><td>0.1067</td><td>0.6296</td><td>0.2271</td><td>18.3355</td><td>0.5915</td></tr></table>

Building on this, we disable content generation in nonoverlapping field-of-view regions (w/o GEN.) by controlling the VFM. Subsequently, we compare the generated content and the non-generative baseline against the corresponding groundtruth regions of the other modality to evaluate the reliability of the generated information. The quantitative experimental results are reported in Tab. VIII. The results demonstrate that the proposed method consistently outperforms the nongenerative variant across all quantitative evaluation metrics. This indicates that SeeRF can generate content that is more consistent with the actual scene, thereby effectively improving information representation in non-overlapping field-of-view regions.

## E. Real-World Validation Experiment

To further validate the effectiveness of the proposed method in practical scenarios, we collected real-world data using the multispectral imaging equipment of a DJI Mavic 3T and conducted experiments on the acquired dataset. The corresponding experimental results are presented in Fig. 13.

In real-world scenarios, existing methods exhibit two major limitations. (1) Limited registration capability. Some competing methods struggle to establish accurate cross-modal correspondences, leaving substantial spatial misalignment in the registered results and failing to adequately compensate for target displacements caused by temporal offsets between sensor acquisitions. Specifically, FusionReg and MulFS-CAP fail to achieve effective spatial alignment across all presented scenes, exhibiting pronounced registration failures. CPMFusion, SuperFusion, and AUNet achieve partial improvements in cross-modal alignment; however, noticeable stretching and geometric distortions persist near image boundaries, compromising the overall visual quality of the fused results. Among the competing methods, CrossRAFT, denoted as Reg., demonstrates relatively strong registration performance, achieving reasonable spatial alignment in most scenes, although noticeable misalignment remains in certain boundary regions. By comparison, SeeRF achieves more accurate alignment of both target regions and image boundaries, substantially reducing local misalignment and implausible geometric distortions, thereby delivering superior overall registration performance. (2) Field-of-view inconsistency. All competing methods fail to effectively recover missing information in non-overlapping field-of-view regions. Notably, the limitations imposed by field-of-view discrepancies become even more apparent when accurate spatial alignment has been achieved. Although some methods successfully align the overlapping regions, they fail to adequately recover targets, such as pedestrians and vehicles, located in non-overlapping regions. Consequently, the fused images exhibit pronounced structural discontinuities and unnatural transitions between observed and missing regions. These artifacts compromise the integrity of target-level semantic structures and reduce overall scene fidelity, ultimately limiting the representational capacity of the fused images. In contrast, by leveraging the powerful prior modeling and content completion capabilities of generative diffusion models, SeeRF effectively reconstructs missing content in nonoverlapping field-of-view regions and seamlessly integrates the generated content with the observed regions. This enables the generation of more complete and coherent fused images with improved scene representation capability.

![](images/9ba2eb9b90708b58f14563fc173a72929319e77ce359628b20f64d8af4823f68.jpg)  
Fig. 13. Qualitative comparison with the state-of-the-art image registration and fusion methods in the real world.

## F. Computational Overhead

To evaluate the computational efficiency of the proposed method, we report the number of model parameters, computational complexity (FLOPs), and inference time. Since our fusion process is built upon a diffusion model, we further decompose the entire framework into individual components and analyze their detailed parameter counts and computational costs, enabling a more comprehensive understanding of the resource consumption of each module. The results are reported in Tab. IX. It can be observed that, for the MIR registration component (SeeRF-Reg.), although a visual foundation model is incorporated, the overall computational cost remains acceptable, with 113.367M parameters, 437.667G FLOPs, and an inference time of 0.142s. For the MCF fusion component (SeeRF-Fus.), the overall network is built upon the diffusion model. To preserve strong generative capability, it inevitably introduces higher computational costs, with 1701.765M parameters, 1968.819G FLOPs, and an inference time of 5.293s.

TABLE IX  
QUANTITATIVE COMPUTATIONAL OVERHEAD COMPARISON OF SEERF.(PARAMS.: NUMBER OF PARAMETERS, FLOPS: FLOATING-POINTOPERATIONS, TIME: INFERENCE TIME)
<table><tr><td>Module</td><td>Params (M)</td><td>FLOPs (G)</td><td>Time (s)</td></tr><tr><td>SeeRF-Reg.</td><td>113.367</td><td>437.667</td><td>0.142</td></tr><tr><td>SeeRF-Fus.</td><td>1701.765</td><td>1968.819</td><td>5.293</td></tr></table>

## V. LIMITATION

To achieve superior registration and fusion performance, we incorporate visual foundation models with strong representation capabilities and diffusion models as the backbone components of our framework. Although these models substantially improve task performance, they inevitably introduce increased computational complexity and inference overhead. Despite employing sampling acceleration strategies to alleviate the computational burden during inference, the proposed architecture still leaves room for further lightweighting and optimization. Future research may explore more efficient generative paradigms, such as single-step generation or flow matching, to reduce computational and memory requirements while preserving the effectiveness of the proposed framework, thereby facilitating its deployment in real-world applications.

## VI. CONCLUSION

In this paper, we propose a generative registration and fusion framework, termed SeeRF, which aims to overcome the long-standing limitation of shared fields of view in image fusion. It addresses the long-standing issue in image registration and fusion, where existing methods primarily rely on information alignment and fusion within overlapping shared field-of-view regions while neglecting the complementary information contained in sensor-specific regions, resulting in missing information or residual artifacts in non-shared fieldof-view regions. Specifically, SeeRF first performs flow field estimation and generates valid field-of-view masks through a diffeomorphic stationary velocity field-based deformation generation strategy, combined with a vision foundation modelguided feature extraction mechanism and a coarse-to-fine progressive multi-modal image registration module. Subsequently, the generative diffusion module for multi-modal completion and fusion is tightly coupled with the multi-modal image registration module, leveraging powerful generative capabilities to overcome the shared field-of-view constraint and achieve effective modeling and fusion of non-shared field-of-view information. Extensive experiments on public datasets and real-world data demonstrate the effectiveness and superiority of the proposed method.

## REFERENCES

[1] J. Ma, H. Xu, J. Jiang, X. Mei, and X.-P. Zhang, “Ddcgan: A dual-discriminator conditional generative adversarial network for multiresolution image fusion,” IEEE Transactions on Image Processing, vol. 29, pp. 4980–4995, 2020.

[2] Y. Sun, B. Cao, P. Zhu, and Q. Hu, “Detfusion: A detection-driven infrared and visible image fusion network,” in Proceedings of the ACM International Conference on Multimedia, 2022, pp. 4003–4011.

[3] R. Liu, Z. Liu, J. Liu, X. Fan, and Z. Luo, “A task-guided, implicitlysearched and meta-initialized deep model for image fusion,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 46, no. 10, pp. 6594–6609, 2024.

[4] H. Bai, Z. Zhao, J. Zhang, Y. Wu, L. Deng, Y. Cui, B. Jiang, and S. Xu, “Refusion: Learning image fusion from reconstruction with learnable loss via meta-learning,” International Journal of Computer Vision, vol. 133, no. 5, pp. 2547–2567, 2025.

[5] J. Liu, X. Fan, Z. Huang, G. Wu, R. Liu, W. Zhong, and Z. Luo, “Target-aware dual adversarial learning and a multi-scenario multimodality benchmark to fuse infrared and visible for object detection,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022, pp. 5802–5811.

[6] C. He, K. Li, G. Xu, Y. Zhang, R. Hu, Z. Guo, and X. Li, “Degradationresistant unfolding network for heterogeneous image fusion,” in Proceedings ofthe IEEE/CVF International Conference on Computer Vision (ICCV), 2023, pp. 12 611–12 621.

[7] G. Wu, H. Liu, H. Fu, Y. Peng, J. Liu, X. Fan, and R. Liu, “Every sam drop counts: Embracing semantic priors for multi-modality image fusion and beyond,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025, pp. 17 882–17 891.

[8] X. Yi, Y. Ma, Y. Li, H. Xu, and J. Ma, “Artificial intelligence facilitates information fusion for perception in complex environments,” The Innovation, vol. 6, no. 4, 2025.

[9] H. Xu, J. Ma, J. Yuan, Z. Le, and W. Liu, “Rfnet: Unsupervised network for mutually reinforcing multi-modal image registration and fusion,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022, pp. 19 679–19 688.

[10] H. Xu, J. Yuan, and J. Ma, “Murf: Mutually reinforcing multi-modal image registration and fusion,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 45, no. 10, pp. 12 148–12 166, 2023.

[11] M. Lu, M. Jiang, X. Tao, and J. Kong, “Au-net: Adaptive unified network for joint multi-modal image registration and fusion,” IEEE Transactions on Image Processing, vol. 34, pp. 4721–4735, 2025.

[12] H. Li, Z. Yang, Y. Zhang, W. Jia, Z. Yu, and Y. Liu, “Mulfs-cap: Multimodal fusion-supervised cross-modality alignment perception for unregistered infrared-visible image fusion,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 47, no. 5, pp. 3673–3690, 2025.

[13] H. Xu, J. Chen, X. Yi, P. Li, and J. Ma, “Iterrf: Iterative and interpretable mutual enhancement for joint multi-modal image registration and fusion,” International Journal of Computer Vision, vol. 134, no. 10, p. 442, 2026.

[14] Y. Liu, X. Chen, R. K. Ward, and Z. J. Wang, “Image fusion with convolutional sparse representation,” IEEE Signal Processing Letters, vol. 23, no. 12, pp. 1882–1886, 2016.

[15] H. Li and X.-J. Wu, “Densefuse: A fusion approach to infrared and visible images,” IEEE Transactions on Image Processing, vol. 28, no. 5, pp. 2614–2623, 2018.

[16] H. Xu, J. Ma, J. Jiang, X. Guo, and H. Ling, “U2fusion: A unified unsupervised image fusion network,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 44, no. 1, pp. 502–518, 2022.

[17] W. Tang, F. He, Y. Liu, and Y. Duan, “Matr: Multimodal medical image fusion via multiscale adaptive transformer,” IEEE Transactions on Image Processing, vol. 31, pp. 5134–5149, 2022.

[18] Z. Zhao, H. Bai, J. Zhang, Y. Zhang, S. Xu, Z. Lin, R. Timofte, and L. Van Gool, “Cddfuse: Correlation-driven dual-branch feature decomposition for multi-modality image fusion,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023, pp. 5906–5916.

[19] X. Yi, H. Xu, H. Zhang, L. Tang, and J. Ma, “Text-if: Leveraging semantic text guidance for degradation-aware and interactive image fusion,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024, pp. 27 016–27 025.

[20] J. Yue, L. Fang, S. Xia, Y. Deng, and J. Ma, “Dif-fusion: Toward high color fidelity in infrared and visible image fusion with diffusion models,” IEEE Transactions on Image Processing, vol. 32, pp. 5705–5720, 2023.

[21] Z. Zhao, H. Bai, Y. Zhu, J. Zhang, S. Xu, Y. Zhang, K. Zhang, D. Meng, R. Timofte, and L. Van Gool, “Ddfm: denoising diffusion model for multi-modality image fusion,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2023, pp. 8082– 8093.

[22] X. Yi, L. Tang, H. Zhang, H. Xu, and J. Ma, “Diff-if: Multi-modality image fusion via diffusion model with fusion knowledge prior,” Information Fusion, vol. 110, p. 102450, 2024.

[23] X. Yi, Y. Zhang, X. Xiang, Q. Yan, H. Xu, and J. Ma, “Lut-fuse: Towards extremely fast infrared and visible image fusion via distillation to learnable look-up tables,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2025, pp. 14 559–14 568.

[24] L. Tang, Y. Deng, Y. Ma, J. Huang, and J. Ma, “Superfusion: A versatile image registration and fusion network with semantic awareness,” IEEE/CAA Journal ofAutomatica Sinica, vol. 9, no. 12, pp. 2121–2137, 2022.

[25] W. Zhao, S. Xie, F. Zhao, Y. He, and H. Lu, “Metafusion: Infrared and visible image fusion via meta-feature embedding from object detection,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023, pp. 13 955–13 965.

[26] Z. Yang, Y. Liu, J. Cheng, Z. Zhu, Y. Zhang, and H. Li, “Customized fusion: A closed-loop dynamic network for adaptive multi-task-aware infrared-visible image fusion,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2026, pp. 188–198.

[27] D. Wang, J. Liu, X. Fan, and R. Liu, “Unsupervised misaligned infrared and visible image fusion via cross-modality image generation and registration,” in Proceedings of the International Joint Conference on Artificial Intelligence (IJCAI), 2022, pp. 3508–3515.

[28] L. Tang, Q. Yan, X. Xiang, L. Fang, and J. Ma, “C2rf: Bridging multi-modal image registration and fusion via commonality mining and contrastive learning,” International Journal of Computer Vision, vol. 133, no. 8, pp. 5262–5280, 2025.

[29] C. Bian, H. Ma, H. Li, Z. Shen, X. Luo, X. Song, and X.-J. Wu, “Fusionregister: Every infrared and visible image fusion deserves registration,” arXiv preprint arXiv:2603.07667, 2026.

[30] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer, “Highresolution image synthesis with latent diffusion models,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022, pp. 10 674–10 685.

[31] L. Zhang, A. Rao, and M. Agrawala, “Adding conditional control to text-to-image diffusion models,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2023, pp. 3813– 3824.

[32] A. Paszke, S. Gross, F. Massa, A. Lerer, J. Bradbury, G. Chanan, T. Killeen, Z. Lin, N. Gimelshein, L. Antiga et al., “Pytorch: An imperative style, high-performance deep learning library,” Advances in Neural Information Processing Systems, vol. 32, 2019.

[33] I. Loshchilov and F. Hutter, “Decoupled weight decay regularization,” in Proceedings of the International Conference on Learning Representations, 2019.

[34] Q. Ha, K. Watanabe, T. Karasawa, Y. Ushiku, and T. Harada, “Mfnet: Towards real-time semantic segmentation for autonomous vehicles with multi-spectral scenes,” in Proceedings of the IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2017, pp. 5108– 5115.

[35] J. Liu, Z. Liu, G. Wu, L. Ma, R. Liu, W. Zhong, Z. Luo, and X. Fan, “Multi-interactive feature learning and a full-time multi-modality benchmark for image fusion and segmentation,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2023, pp. 8115–8124.

[36] X. Jia, C. Zhu, M. Li, W. Tang, and W. Zhou, “Llvip: A visible-infrared paired dataset for low-light vision,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2021, pp. 3496– 3504.

[37] J. L. Barron, D. J. Fleet, and S. S. Beauchemin, “Performance of optical flow techniques,” International Journal of Computer Vision, vol. 12, no. 1, pp. 43–77, 1994.

[38] G. Qu, D. Zhang, and P. Yan, “Information measure for performance of image fusion,” Electronics Letters, vol. 38, no. 7, pp. 313–315, 2002.

[39] K. Pearson, “Note on regression and inheritance in the case of two parents,” Proceedings of the Royal Society of London, vol. 58, pp. 240– 242, 1895.

[40] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, “Image quality assessment: from error visibility to structural similarity,” IEEE Transactions on Image Processing, vol. 13, no. 4, pp. 600–612, 2004.

[41] H. R. Sheikh, A. C. Bovik, and G. De Veciana, “An information fidelity criterion for image quality assessment using natural scene statistics,” IEEE Transactions on Image Processing, vol. 14, no. 12, pp. 2117– 2128, 2005.

[42] C. S. Xydeas, V. Petrovic et al., “Objective image fusion performance measure,” Electronics Letters, vol. 36, no. 4, pp. 308–309, 2000.

[43] J. Li, Q. Hu, and M. Ai, “Rift: Multi-modal image matching based on radiation-variation insensitive feature transform,” IEEE Transactions on Image Processing, vol. 29, pp. 3296–3310, 2020.

[44] P. Truong, M. Danelljan, and R. Timofte, “Glu-net: Global-local universal network for dense flow and correspondences,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020, pp. 6257–6267.

[45] S. Zhou, W. Tan, and B. Yan, “Promoting single-modal optical flow network for diverse cross-modal flow estimation,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 36, no. 3, 2022, pp. 3562–3570.

[46] Q. Xiao, H. Jin, H. Su, Y. Zhang, and G. Sheng, “Cpmfusion: a method for infrared and visible image registration and fusion using modal correlation,” Information Fusion, vol. 127, p. 103924, 2026.

[47] E. Xie, W. Wang, Z. Yu, A. Anandkumar, J. M. Alvarez, and P. Luo, “Segformer: Simple and efficient design for semantic segmentation with transformers,” Advances in Neural Information Processing Systems, vol. 34, pp. 12 077–12 090, 2021.