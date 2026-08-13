# M-Net: Integrating Spectral Features and Physical Field Operators into Deep Learning for Medical Image Segmentation

Jing Zhu<sup>1</sup> Ye Wang<sup>2</sup> Fumin Wang<sup>1,∗</sup>

<sup>1</sup>School of Physics, Xi’an Jiaotong University, Xi’an 710049, China

<sup>2</sup>University of Sydney, Sydney, NSW 2006, Australia

<sup>∗</sup>Corresponding author: fwang1991@xjtu.edu.cn

## Abstract

Purpose: Deep learning-based medical image segmentation has achieved remarkable success, yet purely data-driven approaches often fail to exploit the rich mathematical structure inherent in medical images. This work investigates whether explicit mathematical inductive biases—specifically matrix spectral analysis and vector calculus operators—can enhance segmentation performance beyond what is achievable through data-driven learning alone.

Methods: We propose M-Net (Math-Augmented Network), a segmentation framework that integrates three complementary mathematical priors into the U-Net architecture: (1) continuous spectral features derived from the condition number of centered local pixel matrices, providing a diferentiable measure of texture ill-conditioning; (2) physical field operators (divergence and a discrete curl-like boundary irregularity operator) computed from image gradient fields, capturing focal intensity extrema and edge non-smoothness; and (3) a Math-Attention Gate (MAG) mechanism that adaptively fuses mathematical features with CNN-extracted deep features at skip connections.

Results: Extensive experiments on three benchmark datasets (LiTS, KiTS, and BraTS) demonstrate that M-Net achieves Dice scores of 78.42%, 76.15%, and 83.67%, respectively, outperforming the baseline U-Net by 12.37%, 3.52%, and 5.55% on liver segmentation (LiTS), kidney segmentation (KiTS), and brain tumor segmentation (BraTS). Ablation studies reveal that the condition-number feature contributes a 2.14% improvement over binary invertibility features, while the MAG mechanism provides an additional 1.45% gain over simple concatenation fusion.

Conclusion: M-Net establishes that mathematical inductive biases provide efective complementary information for medical image segmentation. The continuous condition-number feature, computed on mean-centered local matrices, ofers superior gradient information compared to discrete alternatives, and the MAG mechanism preserves these priors throughout the network depth. This work opens avenues for integrating linear algebra and vector calculus into deep learning architectures for medical imaging.

Keywords: Medical image segmentation, U-Net, condition number, singular value decomposition, divergence, curl, attention mechanism, inductive bias, liver segmentation, kidney segmentation, brain tumor segmentation

Code and Data: https://github.com/Fumin111994/mnet-medical-seg

## Contents

1 Introduction 3   
1.1 Background and Motivation 3   
1.2 Challenges and Limitations 3   
1.3 Mathematical Priors in Medical Imaging . 4   
1.4 Contributions . 4   
1.5 Paper Organization 5   
2 Related Work 5   
2.1 Deep Learning for Medical Image Segmentation . 5   
2.2 Attention Mechanisms in Medical Segmentation 5   
2.3 Physics-Informed and Mathematical Methods in Deep Learning 6   
3 Methodology 6   
3.1 Mathematical Preliminaries 6   
3.1.1 Centered Local Pixel Matrix and the Condition Number 6   
3.1.2 Extension to Multi-Channel Images 8   
3.1.3 Physical Field Operators 8   
3.2 Architecture Overview 8   
3.3 Spectral Feature Extraction Module 9   
3.4 Physical Field Operator Module 10   
3.5 Math-Attention Gate Mechanism 10   
3.5.1 Mechanism Design 10   
3.5.2 Multi-Scale Kappa Resize Strategy 11   
3.5.3 Diference from Conventional Attention 11   
3.6 Loss Function 11   
4 Experiments 12   
4.1 Datasets . 12   
4.2 Baseline Reproduction Protocol 12   
4.3 Implementation Details 13   
4.4 Evaluation Metrics 13   
Results 14   
5.1 Comparison with State-of-the-Art Methods 14   
5.2 Ablation Study . 14   
5.3 Cross-Dataset Generalization 15   
5.4 Statistical Robustness 15   
5.5 Computational Eficiency 16   
5.6 Training Convergence 17   
5.7 Qualitative Results 17   
5.8 Condition-Number Prior Analysis . 17   
6 Discussion 17   
6.1 Analysis of the Centered Condition Number Feature 17   
6.2 Multi-Channel Condition Number Aggregation 19   
6.3 Role of Physical Field Operators 19   
6.4 Efectiveness of Math-Attention Gate 20   
6.5 Limitations and Future Work 20   
Conclusion 20

## 1 Introduction

Medical image segmentation is a cornerstone task in computational medicine, serving as the prerequisite for quantitative analysis of anatomical structures and pathological regions. Accurate delineation of organs, tumors, and tissues from computed tomography (CT) and magnetic resonance imaging (MRI) enables clinicians to assess disease progression, plan surgical interventions, and monitor treatment responses with quantitative precision <sup>1</sup>. Among the most clinically consequential applications, liver segmentation from contrast-enhanced CT scans is essential for the diagnosis and management of hepatocellular carcinoma, the sixth most common cancer worldwide<sup>2</sup>. Similarly, kidney segmentation and brain tumor segmentation represent fundamental tasks in abdominal and neuro-oncological imaging, respectively <sup>3,4</sup>.

## 1.1 Background and Motivation

Since the seminal work of Ronneberger et al. <sup>5</sup> in 2015, convolutional neural network (CNN)- based approaches have come to dominate the field of medical image segmentation. The U-Net architecture, with its symmetric encoder-decoder structure and skip connections, has proven remarkably efective for biomedical applications where training data is scarce but spatial precision is paramount. The success of U-Net stems from its ability to combine high-level semantic features from the encoder with fine-grained spatial details from the decoder, enabling precise localization even with limited annotated data.

Over the past decade, numerous extensions have been proposed to address specific limitations of the original U-Net. Çiçek et al. <sup>12</sup> extended the architecture to volumetric data with 3D U-Net; Oktay et al.<sup>6</sup> introduced attention gates in skip connections to suppress irrelevant regions; Zhou et al.<sup>7</sup> employed nested dense skip pathways to reduce the semantic gap between encoder and decoder features; and Isensee et al. <sup>8</sup> proposed nnU-Net, a self-configuring framework that automatically adapts preprocessing, network architecture, and training procedures to new tasks, achieving state-of-the-art results across numerous benchmarks.

More recently, transformer-based architectures have entered the medical imaging domain. Chen et al.<sup>9</sup> proposed TransUNet, which combines the U-Net architecture with Vision Transformers for global context modeling; Cao et al.<sup>10</sup> developed Swin-UNet based on the Swin Transformer; and Hatamizadeh et al.<sup>16</sup> introduced UNETR for 3D medical image segmentation. Despite their architectural diversity, all these methods share a common characteristic: they rely purely on data-driven feature learning, without explicitly incorporating the rich geometric and physical structure inherent in medical images.

## 1.2 Challenges and Limitations

Three fundamental challenges persist in medical image segmentation that motivate our work:

1. Ambiguous boundaries: Tumor-tissue and organ-tissue interfaces often exhibit low contrast and gradual intensity transitions, making boundary delineation inherently dificult. Standard CNN features, learned from pixel-level patterns, may lack the sensitivity to capture subtle textural discontinuities at these interfaces.

2. Irregular morphologies: Pathological regions and anatomical structures exhibit highly irregular shapes and heterogeneous textures that deviate significantly from the smooth, compact structures assumed by many segmentation algorithms. This irregularity is particularly pronounced at invasive tumor margins where tissue infiltration creates complex geometric patterns.

3. Class imbalance: The relatively small volume of lesions or thin organ boundaries compared to background tissue creates severe class imbalance, leading to models that are biased toward the dominant class unless explicitly counteracted through loss weighting or sampling strategies.

A significant limitation of existing approaches is their exclusive reliance on data-driven feature learning. While CNNs are powerful universal function approximators, their purely learned representations may fail to exploit mathematical structure that is a priori known about medical images. Medical images are not arbitrary signals—they possess rich geometric, spectral, and physical properties that arise from the underlying anatomy and imaging physics.

## 1.3 Mathematical Priors in Medical Imaging

The integration of mathematical and physical priors into deep learning has gained significant attention in recent years. Physics-informed machine learning (PIML) <sup>22</sup> embeds governing equations and physical constraints into model architectures, ensuring that predictions satisfy known mathematical relationships. In the context of medical imaging, several works have explored the use of mathematical features for enhancing segmentation:

• Spectral methods: Singular value decomposition (SVD) has been used for texture analysis in image processing <sup>26</sup>, and matrix rank has been employed as a measure of local texture complexity<sup>27</sup>. However, these approaches have typically been used as standalone texture descriptors rather than integrated into deep learning pipelines.

• Physical operators: Divergence and curl-like operators derived from vector calculus have been applied to image analysis for edge detection and boundary characterization <sup>29</sup>, but their application as inductive biases in neural networks for medical image segmentation remains largely unexplored.

• Matrix invertibility: Recent work by Zhu et al. <sup>32</sup> explored using matrix invertibility (determinant being zero or non-zero) as a binary feature for neural network input. While this demonstrated that mathematical features can accelerate convergence and improve segmentation, the binary discretization loses critical information about the degree of illconditioning.

## 1.4 Contributions

To address the limitations identified above, we propose M-Net (Math-Augmented Network), a novel segmentation framework that systematically integrates three complementary mathematical priors into the U-Net architecture. Our contributions are fourfold:

1. We establish a theoretical framework connecting matrix spectral analysis to texture characterization in medical images. We define the centered local pixel matrix and prove that its condition number provides a continuous, diferentiable measure of texture ill-conditioning that is theoretically well-motivated and empirically superior to discrete alternatives. We provide a rigorous analysis correcting a subtle but important issue in naive condition-number computation.

2. We introduce divergence and a discrete curl-like boundary irregularity operator as physicsinformed inductive biases for medical image segmentation. We demonstrate that divergence efectively detects focal intensity extrema (e.g., bright tumor cores), while the curl-like operator captures boundary non-smoothness arising from discretized mixed-derivative inconsistencies at tissue interfaces.

3. We propose the Math-Attention Gate (MAG), a novel attention mechanism at skip connections that uses mathematically-derived feature maps to generate spatial weight masks. The MAG adaptively emphasizes regions with anomalous mathematical properties, preventing the dilution of mathematical priors that occurs with simple input concatenation.

4. We conduct extensive experiments on three benchmark datasets (LiTS <sup>2</sup>, KiTS<sup>3</sup>, BraTS<sup>4</sup>) across two imaging modalities (CT and MRI) and three anatomical sites (liver, kidney, brain), demonstrating consistent improvements over baseline U-Net and competitive performance with state-of-the-art methods. We provide detailed baseline reproduction protocols to ensure experimental reproducibility.

## 1.5 Paper Organization

The remainder of this paper is organized as follows. Section 2 reviews related work in medical image segmentation, attention mechanisms, and physics-informed methods. Section 3 presents the mathematical foundations and architectural design of M-Net. Section 4 describes the experimental setup, datasets, and evaluation protocol. Section 5 reports quantitative and qualitative results. Section 6 provides analysis and interpretation. Section 7 concludes with future directions.

## 2 Related Work

## 2.1 Deep Learning for Medical Image Segmentation

The application of deep learning to medical image segmentation has undergone rapid development since the introduction of fully convolutional networks $( \mathrm { F C N s } ) ^ { 1 1 }$ . The U-Net architecture<sup>5</sup> established the dominant paradigm for biomedical segmentation through its encoder-decoder structure with skip connections, which efectively combines multi-scale contextual information with precise spatial localization.

Subsequent extensions have sought to enhance specific aspects of the U-Net design. Attention $\mathrm { U - N e t ^ { 6 } }$ introduced attention gates that learn to focus on target structures while suppressing background activations. $\mathrm { U - N e t } + + ^ { 7 }$ employed nested and dense skip connections to reduce the semantic gap between encoder and decoder feature maps. $\mathrm { V - N e t ^ { 1 3 } }$ extended the architecture to 3D volumetric data using dice loss for end-to-end training. nnU-Net <sup>8</sup> proposed a self-configuring framework that automatically determines preprocessing, network architecture, and training procedures based on dataset properties, achieving state-of-the-art results across 23 medical segmentation benchmarks.

Transformer-based architectures have recently emerged as alternatives to pure CNN approaches. TransUNet<sup>9</sup> combines CNN features with Vision Transformer $( \mathrm { V i T } ) ^ { 1 4 }$ encodings for global context modeling. Swin-UNet <sup>10</sup> adapts the hierarchical Swin Transformer <sup>15</sup> for medical segmentation. MISSFormer <sup>17</sup> and DsTransUNet <sup>18</sup> further explore transformer-based designs for medical image analysis. Despite their architectural innovations, these methods remain entirely data-driven and do not exploit mathematical structure priors.

## 2.2 Attention Mechanisms in Medical Segmentation

Attention mechanisms have proven efective for enhancing feature selectivity in medical image segmentation. Beyond the seminal Attention U-Net <sup>6</sup>, channel attention mechanisms<sup>19</sup> and spatial attention mechanisms <sup>20</sup> have been widely adopted. The concurrent use of both channel and spatial attention $\mathrm { ( C B A M ^ { 2 0 } ) }$ has shown particular efectiveness for emphasizing salient feature dimensions and regions.

More recently, gating mechanisms for adaptive feature fusion have been explored. UN-$\mathrm { E T R + + } ^ { 2 1 }$ introduced the Gated Shared Weighted Paired Attention (G-SWPA) block, which uses parallel spatial and channel attention controlled by a gating mechanism. However, all existing attention mechanisms operate purely on CNN-extracted or transformer-derived features. They do not leverage explicit external mathematical priors computed directly from the input image. Our Math-Attention Gate fundamentally difers by using mathematically-derived feature maps—condition numbers, divergence, and the curl-like boundary irregularity descriptor—as the basis for attention computation, establishing a principled connection between analytical image properties and learned deep features.

## 2.3 Physics-Informed and Mathematical Methods in Deep Learning

The integration of physical and mathematical priors into deep learning has emerged as a significant research direction. Physics-informed neural networks (PINNs) <sup>23</sup> embed governing diferential equations as soft constraints in the loss function. Physics-informed machine learning (PIML) <sup>22</sup> extends this paradigm to embed physical constraints into model architectures. In medical imaging, physics priors have been applied to enhance segmentation confidence<sup>24</sup> and improve model robustness<sup>25</sup>.

Spectral methods have a long history in image processing. SVD-based texture analysis <sup>26</sup> demonstrated that singular values provide stable texture descriptors invariant to illumination changes. More recently, matrix rank has been used as a measure of local texture complexity <sup>27</sup>, and the condition number has been explored for image quality assessment <sup>28</sup>. However, these approaches have not been systematically integrated into deep learning architectures for medical image segmentation.

In the context of vector calculus for image analysis, divergence and curl-like operators have been used for edge detection<sup>29</sup>, optical flow estimation<sup>30</sup>, and image denoising<sup>31</sup>. The application of these operators as inductive biases in neural network architectures, however, remains largely unexplored in the medical imaging literature.

The work most closely related to ours is that of Zhu et al. <sup>32</sup>, who explored using matrix invertibility (binary determinant classification) as a feature for neural network input. While this demonstrated that mathematical features can improve segmentation, the binary discretization loses critical information about the degree of ill-conditioning. Our work addresses this fundamental limitation by introducing the continuous condition number as a spectral feature, together with physical field operators and a principled attention-based fusion mechanism.

## 3 Methodology

## 3.1 Mathematical Preliminaries

## 3.1.1 Centered Local Pixel Matrix and the Condition Number

We begin by establishing the mathematical foundations of our spectral feature. A naive approach constructs local pixel matrices directly from image intensities and computes their condition numbers. However, this leads to a subtle but critical issue: in homogeneous regions where all pixels have approximately the same intensity, the local matrix is approximately a constant matrix (rank 1), yielding a very large condition number. This contradicts the intuition that homogeneous regions should have low spectral complexity. We resolve this through mean centering.

Definition 3.1 (Centered Local Pixel Matrix). Given a grayscale medical image $I : \Omega \subset \mathbb { R } ^ { 2 }  \mathbb { R }$ where Ω is the image domain, we define the local pixel matrix at position $( x , y ) \in \Omega$ as the $3 \times 3$ matrix $\mathbf { P } _ { x , y } \in \mathbb { R } ^ { 3 \times 3 }$ formed by the $3 \times 3$ neighborhood centered at $( x , y )$

$$
\mathbf { P } _ { x , y } = { \left[ \begin{array} { c c c } { I ( x - 1 , y - 1 ) } & { I ( x , y - 1 ) } & { I ( x + 1 , y - 1 ) } \\ { I ( x - 1 , y ) } & { I ( x , y ) } & { I ( x + 1 , y ) } \\ { I ( x - 1 , y + 1 ) } & { I ( x , y + 1 ) } & { I ( x + 1 , y + 1 ) } \end{array} \right] }\tag{1}
$$

where values outside the image domain are handled via reflect padding. The centered local pixel matrix is then defined as:

$$
\bar { \mathbf { P } } _ { x , y } = \mathbf { P } _ { x , y } - \mu _ { x , y } \cdot \mathbf { 1 } _ { 3 \times 3 }\tag{2}
$$

where $\begin{array} { r } { \mu _ { x , y } = \frac { 1 } { 9 } \sum _ { i , j } P _ { x , y } ( i , j ) } \end{array}$ is the mean intensity of the neighborhood and ${ \bf 1 } _ { 3 \times 3 }$ denotes the all-ones matrix.

Remark 3.1 (Necessity of Mean Centering). Without mean centering, a homogeneous region with constant intensity c yields $\mathbf { P } _ { x , y } \approx c \cdot \mathbf { 1 } _ { 3 \times 3 }$ , which has rank 1. The condition number of a rank-1 matrix is $\kappa = \sigma _ { \mathrm { m a x } } / \sigma _ { \mathrm { m i n } }  \infty$ (or numerically very large), incorrectly suggesting high textural complexity. After mean centering, $\bar { \mathbf { P } } _ { x , y } \approx \mathbf { 0 }$ , giving $\kappa \approx 0$ (with numerical stabilization), which correctly reflects the absence of local structure. Mean centering is therefore essential for the condition number to serve as a meaningful measure of local texture ill-conditioning.

Definition 3.2 (Condition Number Map). The singular value decomposition (SVD) of the centered matrix $\bar { \mathbf { P } } _ { x , y }$ is:

$$
\bar { \mathbf { P } } _ { x , y } = \mathbf { U } \boldsymbol { \Sigma } \mathbf { V } ^ { T }\tag{3}
$$

where U, $\mathbf { V } \in \mathbb { R } ^ { 3 \times 3 }$ are orthogonal matrices and $\pmb { \Sigma } = \mathrm { d i a g } ( \sigma _ { 1 } , \sigma _ { 2 } , \sigma _ { 3 } )$ with $\sigma _ { 1 } \geq \sigma _ { 2 } \geq \sigma _ { 3 } \geq 0$ The local condition number is defined as:

$$
\kappa ( x , y ) = \frac { \sigma _ { \mathrm { m a x } } ( \bar { \mathbf { P } } _ { x , y } ) } { \sigma _ { \mathrm { m i n } } ( \bar { \mathbf { P } } _ { x , y } ) + \epsilon } = \frac { \sigma _ { 1 } } { \sigma _ { 3 } + \epsilon }\tag{4}
$$

where $\epsilon = 1 0 ^ { - 6 }$ ensures numerical stability when $\sigma _ { 3 } \approx 0$

Proposition 3.1 (Properties of the Condition Number Map). The condition number map $\kappa : \Omega \to { \mathbb { R } _ { \geq 0 } }$ satisfies the following properties:

1. Continuity: κ is continuous with respect to the image intensity, i.e., small perturbations in I result in small changes in κ. This follows from the continuity of SVD with respect to matrix entries <sup>33</sup> and the linearity of the centering operation.

2. Scale invariance: $\kappa ( x , y )$ is invariant to uniform scaling of the original local matrix: $\kappa ( c \cdot \mathbf { P } _ { x , y } ) = \kappa ( \mathbf { P } _ { x , y } )$ for $c \neq 0 .$ , since scaling does not afect the mean-subtracted matrix up to a constant factor that cancels in the ratio.

3. Boundary sensitivity: At sharp intensity boundaries, where the local neighborhood spans two distinct tissue types, the centered matrix $\bar { \mathbf { P } } _ { x , y }$ captures the intensity diferences across the boundary. The rank of $\bar { \mathbf { P } } _ { x , y }$ increases (typically to $2 \mathrm { o r 3 } )$ , yielding $\sigma _ { 1 } \gg \sigma _ { 3 }$ and therefore $\kappa ( x , y ) \gg 1$

4. Smooth-region insensitivity: In homogeneous regions, $\bar { \mathbf { P } } _ { x , y } \approx \mathbf { 0 } _ { 3 \times 3 }$ (up to noise), giving $\sigma _ { 1 } \approx \sigma _ { 3 } \approx 0$ and $\kappa ( x , y ) \approx \sigma _ { 1 } / \epsilon \approx 0$ (with stabilization). This correctly reflects the absence of local textural structure.

Proof. Property 1: The SVD is continuous with respect to matrix perturbations <sup>33</sup>, and centering is a linear operation (subtraction of the mean), hence the composition is continuous.

Property 2: For $\mathbf { P } ^ { \prime } = c \mathbf { P }$ , the mean becomes $\mu ^ { \prime } = c \mu$ , so $\bar { \mathbf { P } } ^ { \prime } = c \mathbf { P } - c \mu \mathbf { 1 } = c ( \mathbf { P } - \mu \mathbf { 1 } ) = c \bar { \mathbf { P } }$ The singular values scale as $\sigma _ { i } ( c \bar { \mathbf { P } } ) = | c | \sigma _ { i } ( \bar { \mathbf { P } } )$ , so their ratio is invariant.

Property 3: At a boundary between two regions with intensities a and $b \ ( a \neq b )$ , the centered matrix contains both positive and negative entries reflecting the intensity deviation from the local mean. The matrix has at least two non-zero singular values, giving $\sigma _ { 1 } \gg \sigma _ { 3 }$

Property 4: In a homogeneous region, all entries of $\mathbf { P } _ { x , y }$ equal c (up to noise), so $\mu = c$ and $\bar { \mathbf P } _ { x , y } = \mathbf P _ { x , y } - c \cdot \mathbf { 1 } \approx \mathbf { 0 }$ . Thus $\sigma _ { 1 } \approx 0$ and $\kappa \approx 0$ (with ϵ stabilization).

For training stability, we apply log-normalization:

$$
\hat { \kappa } ( x , y ) = \log ( \kappa ( x , y ) + 1 )\tag{5}
$$

which maps $\kappa \in [ 0 , \infty )$ to $\hat { \kappa } \in [ 0 , \infty )$ while preserving monotonicity and ensuring $\hat { \kappa } ( 0 ) = 0$

## 3.1.2 Extension to Multi-Channel Images

For multi-channel images such as the four MRI sequences in BraTS (T1, T1-Gd, T2, FLAIR), the condition number is computed independently per channel and then aggregated:

$$
\hat { \kappa } _ { \mathrm { a v g } } ( x , y ) = \frac { 1 } { C } \sum _ { c = 1 } ^ { C } \hat { \kappa } ^ { ( c ) } ( x , y )\tag{6}
$$

where $C = 4$ is the number of channels and $\hat { \kappa } ^ { \left( c \right) }$ is the log-normalized condition number computed from the centered local pixel matrix of channel $c .$ This channel-averaging strategy ensures that the spectral feature captures texture information across all available imaging contrasts. The divergence and curl operators are similarly computed per-channel and averaged.

## 3.1.3 Physical Field Operators

We treat the image intensity $I ( x , y )$ as a scalar field defined over the 2D spatial domain Ω. The gradient field is $\nabla I = ( I _ { x } , I _ { y } ) ^ { T }$ , where $I _ { x }$ and $I _ { y }$ are spatial derivatives computed via Sobel operators:

$$
I _ { x } = { \left[ \begin{array} { l l l } { - 1 } & { 0 } & { 1 } \\ { - 2 } & { 0 } & { 2 } \\ { - 1 } & { 0 } & { 1 } \end{array} \right] } * I , \quad I _ { y } = { \left[ \begin{array} { l l l } { - 1 } & { - 2 } & { - 1 } \\ { 0 } & { 0 } & { 0 } \\ { 1 } & { 2 } & { 1 } \end{array} \right] } * I\tag{7}
$$

Definition 3.3 (Divergence of Gradient Field). The divergence (Laplacian) of the image scalar field is:

$$
\mathrm { d i v } ( x , y ) = \nabla \cdot \nabla I = I _ { x x } + I _ { y y }\tag{8}
$$

where $I _ { x x }$ and $I _ { y y }$ are second-order spatial derivatives computed via separable convolutions.

Definition 3.4 (Discrete Curl-like Boundary Irregularity Operator). We define a discrete curl-like operator that measures the inconsistency between mixed partial derivatives of the discrete gradient field. Given gradient components $g _ { x } = \mathrm { S o b e l } _ { x } * I$ and $g _ { y } = \mathrm { S o b e l } _ { y } * I ~ ( \mathrm { E q . ~ 7 } )$ the operator is:

$$
\mathrm { c u r l } ( x , y ) = \bigg | \frac { \partial g _ { y } } { \partial x } - \frac { \partial g _ { x } } { \partial y } \bigg |\tag{9}
$$

where the cross-derivatives $\frac { \partial g _ { y } } { \partial x }$ and $\frac { \partial g _ { x } } { \partial y }$ are computed via distinct 1D directional diferences (Eq. 11).

Remark 3.2 (Not a Strict Physical Curl). For a smooth scalar field, the curl of its gradient is identically zero by Clairaut’s theorem $\begin{array} { r } { \big ( \frac { \partial ^ { 2 } \dot { I } } { \partial y \partial x } = \frac { \partial ^ { 2 } I } { \partial x \partial y } \big ) } \end{array}$ . Our operator is not the continuous curl of a gradient field; rather, it is a discrete curl-like descriptor that captures the inconsistency between mixed partial derivatives arising from the use of diferent Sobel kernels and orthogonal 1D diference operators in the discrete setting. This inconsistency is maximized at non-smooth boundaries where the continuous diferentiability assumption breaks down, making the operator an efective feature for boundary irregularity detection.

## 3.2 Architecture Overview

The proposed M-Net architecture extends the standard U-Net by introducing three mathematical augmentation modules:

1. Spectral Feature Extraction (SFE): Computes the condition number map $\scriptstyle { \hat { \kappa } } ( x , y )$ from centered local $3 \times 3$ pixel neighborhoods via batched SVD, following Definition 3.1 and Definition 3.2.

2. Physical Field Operator (PFO): Computes divergence $\operatorname { d i v } ( x , y )$ and a discrete curl-like boundary irregularity map curl(x, y) from image gradient fields using fixed-weight Sobel convolutions and directional 1D diferences.

3. Math-Attention Gate (MAG): Adaptively fuses mathematical feature maps with CNN feature maps at skip connections, using the mathematical features to generate spatial attention weights.

Figure 1 illustrates the overall architecture.  
![](images/0c37a7e5529eb9be7bd7dfad3d66bde108862363af28290509f21c53fbafb338.jpg)  
Figure 1: Overview of the proposed M-Net architecture. The Spectral Feature Extraction (SFE) module computes condition number maps from centered local $3 \times 3$ pixel neighborhoods (Definition 3.1). The Physical Field Operator (PFO) module computes divergence and a discrete curl-like boundary irregularity descriptor from image gradients (Definition 3.4). The Math-Attention Gate (MAG) adaptively fuses mathematical features with CNN features at skip connections.

## 3.3 Spectral Feature Extraction Module

The SFE module extracts the condition number map from the input image. For a batch of images $\mathbf { X } \in \mathbb { R } ^ { B \times C \times H \times W }$ (where C = 1 for CT and $C = 4$ for BraTS MRI), the computation proceeds as follows:

1. Per-channel processing: For each channel $c \in \{ 1 , \ldots , C \}$ , extract the channel image $\mathbf { X } ^ { \left( c \right) } \in \mathbb { R } ^ { B \times 1 \times \tilde { H ^ { \times } } W }$

2. Patch extraction: Using nn.Unfold with kernel size 3 and padding 1, extract overlapping $3 \times 3$ patches: $\mathbf { P } ^ { ( c ) } \in \mathbb { R } ^ { N \times 9 \times ( H W ) }$

3. Matrix reshaping: Reshape to $\mathbf { M } ^ { ( c ) } \in \mathbb { R } ^ { ( B \cdot H W ) \times 3 \times 3 }$

4. Mean centering: Compute the mean of each $3 \times 3$ patch and subtract: $\bar { \mathbf { M } } ^ { ( c ) } = \mathbf { M } ^ { ( c ) }$ $\boldsymbol { \mu } ^ { ( c ) } \cdot \mathbf { 1 } _ { 3 \times 3 } \ : ( \mathrm { E q . ~ 2 } )$

5. Batched SVD: Compute $\mathbf { U } , \mathbf { S } , \mathbf { V } ^ { T } = \mathrm { S V D } ( \bar { \mathbf { M } } ^ { ( c ) } )$ , yielding singular values $\mathbf { S } \in \mathbb { R } ^ { ( B \cdot H W ) \times 3 }$

6. Condition number: $\kappa ^ { ( c ) } = S _ { : , 0 } / ( S _ { : , 2 } + \epsilon )$

7. Reshape and normalize: Reshape to R $B { \times } 1 { \times } H { \times } W$ and apply log-normalization $\left( \mathrm { E q . ~ 5 } \right)$

8. Channel aggregation: For multi-channel inputs, average: $\begin{array} { r } { \hat { \kappa } _ { \mathrm { a v g } } = \frac { 1 } { C } \sum _ { c } \hat { \kappa } ^ { ( c ) } \left( \mathrm { E q . ~ } 6 \right) } \end{array}$

The entire pipeline is diferentiable and GPU-accelerated, enabling end-to-end training.

Compared to the binary invertibility feature (determinant $\neq 0 )$ used in prior work<sup>32</sup>, the condition number provides three key advantages: (i) continuity—it captures the degree of ill-conditioning rather than a binary classification; (ii) robustness—small perturbations cause gradual changes rather than discrete jumps; and (iii) rich gradient information—the continuous range provides more informative gradients during backpropagation.

## 3.4 Physical Field Operator Module

The PFO module computes divergence and the curl-like boundary irregularity map using fixedweight (non-learnable) convolutional layers. The divergence operator has a direct physical interpretation (Laplacian of the scalar field). The curl-like operator, as detailed in Definition 3.4 and Remark 3.2, is a discrete descriptor that captures mixed-derivative inconsistency rather than a strict continuous curl. $\mathrm { B y }$ preventing these operators from being modified during training, we ensure that their mathematical interpretations are preserved throughout optimization. The operators serve as consistent inductive biases rather than learnable features that might converge to operators lacking clear physical meaning.

Divergence computation: The divergence (Eq. 8) is computed via separable convolutions:

$$
I _ { x x } = { \Big [ } 1 \quad - 2 \quad 1 { \Big ] } * I , \quad I _ { y y } = { \binom { 1 } { - 2 } } * I\tag{10}
$$

Curl computation: The curl $\left( \operatorname { E q . 9 } \right)$ is computed via a two-step procedure (Section ??):

1. Compute gradient components: $g _ { x } = \mathrm { S o b e l } _ { x } * I , g _ { y } = \mathrm { S o b e l } _ { y } * I \ ( \mathrm { E q . ~ 7 } )$

2. Apply directional first-order diferences:

$$
\frac { \partial g _ { y } } { \partial x } = g _ { y } \circledast \left[ - 1 \quad 0 \quad 1 \right] , \quad \frac { \partial g _ { x } } { \partial y } = g _ { x } \circledast \left[ \begin{array} { c } { - 1 } \\ { 0 } \\ { 1 } \end{array} \right]\tag{11}
$$

3. Curl magnitude: curl $\begin{array} { r } { \mathbf { \Omega } = \left| \frac { \partial g _ { y } } { \partial x } - \frac { \partial g _ { x } } { \partial y } \right| . } \end{array}$

The 1D diference kernels in Eq. 11 are distinct (horizontal $[ - 1 , 0 , 1 ]$ vs. vertical $[ - 1 ; 0 ; 1 ] )$ ensuring that the curl is not identically zero in the discrete setting. The horizontal diference extracts the x-variation of the vertical gradient $g _ { y } .$ while the vertical diference extracts the y-variation of the horizontal gradient $g _ { x }$

For multi-channel inputs, divergence and curl are computed per-channel and averaged, analogous to $\operatorname { E q . 6 }$

## 3.5 Math-Attention Gate Mechanism

A critical challenge in integrating mathematical features with deep features is preventing the dilution of mathematical priors as network depth increases. Simple input concatenation, as used in prior work <sup>32</sup>, fails to preserve mathematical features in deep layers because subsequent convolution operations treat all channels uniformly.

## 3.5.1 Mechanism Design

The Math-Attention Gate is deployed at each skip connection where mathematical features are to be integrated. Given a CNN feature map $\mathbf { \bar { F } } _ { \mathrm { c n n } } \in \mathbb { R } ^ { C _ { \mathrm { c n n } } \times H \times W }$ from the encoder and

a mathematical feature map $\mathbf { F } _ { \mathrm { m a t h } } \in \mathbb { R } ^ { C _ { \mathrm { m a t h } } \times H \times W }$ (the concatenation of condition number, divergence, and curl maps), the MAG computes:

$$
\mathbf { F } _ { \mathrm { o u t } } = \mathbf { F } _ { \mathrm { c n n } } \odot \sigma \left( \mathbf { W } _ { c } * \mathbf { F } _ { \mathrm { c n n } } + \mathbf { W } _ { m } * \mathbf { F } _ { \mathrm { m a t h } } + b \right)\tag{12}
$$

where $\mathbf { W } _ { c } \in \mathbb { R } ^ { 1 \times C _ { \mathrm { c n n } } \times 1 \times 1 }$ and $\mathbf { W } _ { m } \in \mathbb { R } ^ { 1 \times C _ { \mathrm { m a t h } } \times 1 \times 1 }$ are $1 \times 1$ convolution kernels, $b \in \mathbb { R }$ is a learnable bias, σ denotes the sigmoid activation, ∗ denotes convolution, and ⊙ represents element-wise (Hadamard) multiplication. The sigmoid output $\mathbf { G } \in \mathbb { R } ^ { 1 \times H \times W }$ acts as a spatial gate that is broadcast across all $C _ { \mathrm { c n n } }$ channels.

## 3.5.2 Multi-Scale Kappa Resize Strategy

The condition-number map is extracted at input resolution (256×256). However, encoder features at diferent skip levels have diferent spatial resolutions. For the MAG to operate correctly, the mathematical feature maps must be resized to match the CNN feature resolution at each skip level:

• Skip 1 (after E1): 256 × 256 — direct use

• Skip 2 (after E2): 128 × 128 — bilinear downsample 0.5×

• Skip 3 (after E3): 64 × 64 — bilinear downsample 0.25×

• Skip 4 (after E4): 32 × 32 — bilinear downsample 0.125×

Bilinear interpolation (rather than nearest-neighbor) is used because the condition number is a continuous-valued feature map.

## 3.5.3 Diference from Conventional Attention

The MAG difers from conventional attention mechanisms <sup>6,20</sup> in two key aspects: (i) it explicitly incorporates external mathematical features computed from the input image, rather than relying solely on internal feature statistics; and (ii) it operates at skip connections to preserve mathematical priors throughout the decoding pathway, rather than being restricted to gating encoder features.

## 3.6 Loss Function

We employ a composite loss function combining weighted cross-entropy and Dice loss to address class imbalance:

$$
\mathcal { L } = \mathcal { L } _ { \mathrm { W C E } } + \mathcal { L } _ { \mathrm { D i c e } }\tag{13}
$$

The weighted cross-entropy loss is:

$$
\mathcal { L } _ { \mathrm { W C E } } = - \sum _ { i } \left[ w _ { 0 } ( 1 - y _ { i } ) \log ( 1 - \hat { y } _ { i } ) + w _ { 1 } y _ { i } \log ( \hat { y } _ { i } ) \right]\tag{14}
$$

where $w _ { 0 } = 0 . 0 2$ and $w _ { 1 } = 1 . 0$ account for the severe imbalance between background and foreground regions<sup>5</sup>. The Dice loss is:

$$
{ \mathcal { L } } _ { \mathrm { D i c e } } = 1 - { \frac { 2 \sum _ { i } y _ { i } { \hat { y } } _ { i } + \epsilon } { \sum _ { i } y _ { i } + \sum _ { i } { \hat { y } } _ { i } + \epsilon } }\tag{15}
$$

with $\epsilon = 1 0 ^ { - 6 }$ for numerical stability. The Dice loss is computed per-sample and averaged across the batch, giving equal weight to each slice regardless of foreground pixel count.

For multi-class segmentation (BraTS with 3 tumor subregions: ET, NETC, SNFH), the Dice loss is computed independently for each class and then averaged:

$$
\mathcal { L } _ { \mathrm { D i c e } } ^ { \mathrm { m u l t i } } = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \left( 1 - \frac { 2 \sum _ { i } y _ { i } ^ { ( k ) } \hat { y } _ { i } ^ { ( k ) } + \epsilon } { \sum _ { i } y _ { i } ^ { ( k ) } + \sum _ { i } \hat { y } _ { i } ^ { ( k ) } + \epsilon } \right)\tag{16}
$$

where K is the number of classes (excluding background). The final reported DSC is the arithmetic mean of per-class Dice scores, which is standard practice in the medical segmentation community<sup>8</sup>.

## 4 Experiments

## 4.1 Datasets

We evaluate M-Net on three publicly available benchmark datasets to validate cross-organ and cross-modality generalization. Table 1 summarizes the dataset characteristics.

Table 1: Summary of experimental datasets. All datasets are publicly available benchmark collections with expert annotations.
<table><tr><td>Dataset</td><td>Modality</td><td>Anatomy</td><td>Classes</td><td>Train</td><td>Test</td><td>Task</td></tr><tr><td>LiTS</td><td>CT</td><td>Liver</td><td>2 (BG, liver)</td><td>103</td><td>28</td><td>Liver seg.</td></tr><tr><td>KiTS23</td><td>CT</td><td>Kidney</td><td>2 (BG, kidney)</td><td>489</td><td>110</td><td>Kidney seg.</td></tr><tr><td>BraTS24</td><td>MRI (4 ch.)</td><td>Brain</td><td>4 (BG, ET, NETC, SNFH)</td><td>500</td><td>70</td><td>Tumor seg.</td></tr></table>

LiTS (Liver Tumor Segmentation Challenge)<sup>2</sup>: The LiTS dataset comprises 131 contrastenhanced abdominal CT scans with expert annotations for liver segmentation. The liver is segmented as a single foreground class against background. Following the standard protocol established by the challenge organizers, we use 103 training volumes and 28 test volumes (the oficial leaderboard holdout set). The primary task is liver segmentation—delineating the liver organ from abdominal CT scans.

KiTS23 (Kidney Tumor Segmentation Challenge)<sup>3</sup>: The KiTS dataset contains 599 abdominal CT scans with voxel-level annotations for kidney, renal tumor, and renal cyst. We merge all kidney-associated annotations (kidney parenchyma, tumor, and cyst) into a single foreground class, framing the task as binary kidney segmentation (foreground = kidney + tumor + cyst vs. background). This is clinically motivated: accurate delineation of the entire kidney region (including pathological tissue) is a prerequisite for subsequent tumor-focused analysis in clinical workflows. We use the oficial train/test split (489 training, 110 testing cases). Table 1 reports 2 classes (BG, kidney) reflecting this merged foreground class.

BraTS24 (Brain Tumor Segmentation Challenge) <sup>4</sup>: The BraTS dataset provides multimodal brain MRI scans (T1, T1-Gd, T2, FLAIR) with annotations for enhancing tumor (ET), non-enhancing tumor core (NETC), and surrounding FLAIR hyperintensity (SNFH). We use 500 cases for training and evaluate on the 70-case validation set. The task is brain tumor subregion segmentation—a multi-class problem where each tumor subregion is evaluated independently and the final score is the arithmetic mean of per-class Dice scores (Eq. 16).

## 4.2 Baseline Reproduction Protocol

To ensure fair and reproducible comparisons, we retrained all baseline methods using identical experimental protocols. The following details are provided to address experimental reproducibility concerns:

U-Net Baseline: We use the original U-Net architecture <sup>5</sup> with five encoding and decoding blocks (64–128–256–512–1024 channels). The encoder applies 3 × 3 convolutions followed by

ReLU and $2 \times 2$ max-pooling. The decoder applies $2 \times 2$ transposed convolutions for upsampling.   
Skip connections concatenate encoder features with decoder features at matching resolutions.   
All baseline methods share this backbone unless otherwise noted.

Attention U-Net: We implement the attention-gated $\mathrm { U } { - } \mathrm { N e t } ^ { 6 }$ on top of the U-Net backbone, adding attention gates at each skip connection that learn to suppress irrelevant background activations.

U-Net++: We implement $\mathrm { U - N e t } + + ^ { 7 }$ with nested dense skip pathways using the standard configuration with deep supervision.

nnU-Net Comparison: We include $\mathrm { { n n U } { - } N e t } ^ { 8 }$ as a strong reference baseline. However, we note an important methodological distinction: nnU-Net is a self-configuring 3D framework that automatically determines preprocessing, patch sizes, and network topology based on dataset properties. For fair comparison with our 2D slice-based approach, we configure nnU-Net in 2D mode (nnUNetv2\_plan\_and\_preprocess -pl ExperimentPlanner\_2D), forcing it to process individual axial slices rather than volumetric patches. This ensures that performance diferences reflect architectural innovations rather than dimensionality advantages. Even in this constrained 2D setting, nnU-Net benefits from its extensive automatic preprocessing pipeline (resampling, intensity normalization, and training scheme optimization).

TransUNet: We use the oficial implementation with ViT-B/16 encoder pretrained on ImageNet, fine-tuned on each medical dataset.

All methods are trained from scratch (except TransUNet with ImageNet pretraining) using identical data augmentation, input resolution $( 2 5 6 \times 2 5 6 )$ , and training procedures to isolate the efect of each architectural contribution.

## 4.3 Implementation Details

All experiments are implemented in PyTorch 2.0. Training details:

• Optimizer: Adam, initial learning rate $1 0 ^ { - 4 }$ , weight decay $1 0 ^ { - 5 }$

• Scheduler: ReduceLROnPlateau (monitor validation loss, patience 10, factor 0.5)

• Batch size: 16 (LiTS, KiTS), 8 (BraTS)

• Sampler: WeightedRandomSampler with foreground oversampling $( 1 / 3$ positive slices)

• Augmentation: H-flip $\mathrm { ( p { = } 0 . 5 ) }$ , V-flip $\left( \mathrm { p { = } 0 . 3 } \right)$ , rotation $( \pm 1 5 \check { \mathrm { r } } , \mathrm { p } { = } 0 . 5 )$ , elastic $( \alpha = 1 . 5 , \sigma =$ $5 0 , \mathrm { p } { = } 0 . 3 )$ , brightness/contrast $( \pm 1 0 \% , \mathrm { p { = } 0 . 3 ) }$

• Input size: $2 5 6 \times 2 5 6$ pixels (axial slices)

• Max epochs: 300, early stopping patience: 50

• Checkpoint: Best validation DSC

• Hardware: Single NVIDIA RTX 4060 Ti GPU (16GB)

## 4.4 Evaluation Metrics

We report the following standard segmentation metrics:

Dice Similarity Coeficient (DSC): $\begin{array} { r } { \mathrm { D S C } = \frac { 2 | Y \cap \hat { Y } | } { | Y | + | \hat { Y } | } } \end{array}$ , measuring voxel-level overlap. LiTS and KiTS are evaluated as binary segmentation: LiTS (liver vs. background) and KiTS (kidney+tumor+cyst merged foreground vs. background). BraTS is evaluated as multi-class segmentation: the reported DSC is the arithmetic mean of per-class Dice scores across the three tumor subregions (ET, NETC, SNFH), computed via Eq. 16.

95% Hausdorf Distance (HD95): The 95th percentile of the set of minimum distances from boundary points in the prediction to boundary points in the ground truth, and vice versa:

$$
\mathrm { H D } _ { 9 5 } ( Y , \hat { Y } ) = \operatorname* { m a x } \left\{ \operatorname* { s u p } _ { y \in \partial Y } \operatorname* { i n f } _ { \hat { y } \in \partial \hat { Y } } d ( y , \hat { y } ) , \operatorname* { s u p } _ { \hat { y } \in \partial \hat { Y } } \operatorname* { i n f } _ { y \in \partial Y } d ( y , \hat { y } ) \right\} _ { \mathfrak { g } \in \partial \hat { Y } }\tag{17}
$$

where distances are computed in physical millimeters (mm) after mapping voxel coordinates to patient space using the CT/MRI scan spacing metadata. HD95 is computed on the full 3D volume by reconstructing segmentation masks from individual slices, not on a per-slice basis. The 95th percentile is used instead of the maximum to reduce sensitivity to annotation outliers.

Sensitivity (Sen): $\begin{array} { r } { \mathrm { S e n } = \frac { \mathrm { T P } } { \mathrm { T P } + \mathrm { F N } } . } \\ { \quad \mathrm { c } } \end{array}$

Specificity (Spec): $\begin{array} { r } { \mathrm { S p e c } = \frac { \mathrm { T N } } { \mathrm { T N } + \mathrm { F P } } } \end{array}$

## 5 Results

## 5.1 Comparison with State-of-the-Art Methods

Table 2 presents the quantitative comparison of M-Net against baseline U-Net and recent state-of-the-art methods. All methods were retrained using the protocol described in Section 4.2.

Table 2: Quantitative comparison on LiTS (liver segmentation), KiTS (kidney segmentation), and BraTS (brain tumor segmentation). Best results are highlighted in bold. nnU-Net is evaluated in 2D mode for fair comparison (Section 4.2).
<table><tr><td></td><td colspan="2">LiTS (Liver)</td><td colspan="2">KiTS (Kidney)</td><td colspan="2">BraTS (Tumor)</td></tr><tr><td>Method</td><td>DSC (%)</td><td>HD95 (mm)</td><td>DSC (%)</td><td>HD95 (mm)</td><td>DSC (%)</td><td>HD95 (mm)</td></tr><tr><td>U-Net</td><td>66.05</td><td>10.79</td><td>72.63</td><td>8.42</td><td>78.12</td><td>5.86</td></tr><tr><td>Attn U-Net</td><td>66.58</td><td>13.63</td><td>73.15</td><td>7.98</td><td>79.45</td><td>5.24</td></tr><tr><td>U-Net++</td><td>66.08</td><td>13.78</td><td>72.89</td><td>8.15</td><td>78.67</td><td>5.62</td></tr><tr><td>nnU-Net (2D)</td><td>74.52</td><td>7.23</td><td>75.34</td><td>6.15</td><td>82.03</td><td>4.18</td></tr><tr><td>TransUNet</td><td>72.18</td><td>8.56</td><td>74.67</td><td>6.82</td><td>81.45</td><td>4.45</td></tr><tr><td>RIS-UNet</td><td>76.84</td><td>10.03</td><td></td><td></td><td></td><td></td></tr><tr><td>M-Net (Ours)</td><td>78.42</td><td>6.87</td><td>76.15</td><td>5.78</td><td>83.67</td><td>3.92</td></tr></table>

M-Net achieves the highest Dice scores across all three datasets. The improvements over baseline U-Net are substantial: +12.37% on LiTS liver segmentation, +3.52% on KiTS kidney segmentation, and +5.55% on BraTS brain tumor segmentation. The large improvement on LiTS (12.37%) can be attributed to the challenging nature of liver boundary delineation in contrast-enhanced CT, where the condition number and the curl-like boundary irregularity descriptor excel at characterizing organ-tissue interfaces with complex texture patterns. The BraTS improvement (5.55%) demonstrates that the channel-averaged condition number (Eq. 6) efectively captures multi-modal MRI texture information.

Notably, M-Net outperforms nnU-Net even when the latter is evaluated in its optimized 2D configuration, demonstrating that explicit mathematical priors provide complementary information beyond what self-configuring pipelines can achieve through preprocessing and architecture search alone. The Hausdorf Distance improvements are particularly significant, indicating superior boundary delineation—an attribute directly attributable to the condition number and the curl-like boundary descriptor that specifically target edge characterization.

## 5.2 Ablation Study

Table 3 presents systematic ablation results on the LiTS dataset.

Table 3: Ablation study on LiTS liver segmentation dataset. CN: Condition Number; Div: Divergence; Curl: discrete curl-like boundary irregularity operator; MAG: Math-Attention Gate.
<table><tr><td>Configuration</td><td>DSC (%)</td><td>IoU (%)</td><td>HD95 (mm)</td><td>Sen (%)</td></tr><tr><td>Baseline U-Net</td><td>66.05</td><td>52.38</td><td>10.79</td><td>71.24</td></tr><tr><td>+ Binary Invertibility</td><td>68.32</td><td>54.76</td><td>9.45</td><td>73.18</td></tr><tr><td>+ Condition Number (CN)</td><td>70.46</td><td>56.82</td><td>8.23</td><td>74.65</td></tr><tr><td>+ CN + Divergence</td><td>73.18</td><td>59.47</td><td>7.56</td><td>76.32</td></tr><tr><td>+ CN + Divergence + Curl</td><td>75.82</td><td>62.14</td><td>7.12</td><td>78.45</td></tr><tr><td>+ CN + Div + Curl + Concat</td><td>76.97</td><td>63.28</td><td>7.05</td><td>79.12</td></tr><tr><td>Full M-Net (with MAG)</td><td>78.42</td><td>65.18</td><td>6.87</td><td>80.35</td></tr></table>

The ablation results reveal several key insights: (i) upgrading from binary invertibility to continuous condition numbers provides a 2.14% Dice improvement (70.46% vs. 68.32%), confirming the value of continuous spectral information; (ii) adding divergence and the curl-like operator progressively enhances performance, with the curl-like operator contributing more to boundary-sensitive metrics (HD95); (iii) the Math-Attention Gate provides an additional 1.45% Dice improvement over simple concatenation (78.42% vs. 76.97%), validating the importance of principled feature fusion.

## 5.3 Cross-Dataset Generalization

Table 4 shows cross-dataset generalization results.

Table 4: Cross-dataset generalization (DSC %). Models trained on source dataset and evaluated on target dataset without fine-tuning.
<table><tr><td rowspan="2">Train</td><td colspan="4">U-Net</td><td colspan="4">M-Net</td></tr><tr><td>LiTS</td><td>KiTS</td><td>BraTS</td><td>Avg.</td><td>LiTS</td><td>KiTS</td><td>BraTS</td><td>Avg.</td></tr><tr><td>LiTS</td><td>66.05</td><td>48.23</td><td>42.18</td><td>52.16</td><td>78.42</td><td>52.87</td><td>46.35</td><td>59.21</td></tr><tr><td>KiTS</td><td>47.82</td><td>72.63</td><td>41.56</td><td>54.00</td><td>51.24</td><td>76.15</td><td>45.82</td><td>57.74</td></tr><tr><td>BraTS</td><td>43.15</td><td>44.67</td><td>78.12</td><td>55.31</td><td>47.68</td><td>48.93</td><td>83.67</td><td>60.09</td></tr></table>

M-Net consistently outperforms U-Net in cross-dataset evaluation, with an average improvement of +4.84% across all transfer scenarios. This enhanced generalization suggests that mathematical priors provide domain-invariant features that transfer more efectively across diferent anatomical structures and imaging modalities.

## 5.4 Statistical Robustness

To address the reproducibility and statistical reliability of our results, we report mean ± standard deviation across three independent training runs with diferent random seeds (42, 123, 456). All hyperparameters are held constant across seeds; only the random initialization and data shufling difer. Table 5 presents the full statistical comparison on LiTS, and Table 6 presents the key comparisons on KiTS and BraTS.

The statistical analysis confirms that M-Net’s improvements are consistent and significant. On LiTS, M-Net achieves a mean DSC of 78.38% with a standard deviation of only 0.36%, indicating high training stability. The paired t-test against the U-Net baseline yields $p < 0 . 0 0 1$ on all three datasets, confirming that the improvements are statistically significant. Notably, even the worst-performing M-Net seed (78.02% DSC on LiTS) still outperforms the best-performing RIS-UNet seed (77.27%), demonstrating the robustness of the mathematical prior.

Table 5: Statistical comparison on LiTS liver segmentation (mean ± std over 3 seeds). Best mean results in bold. p-values from paired t-test against U-Net baseline.
<table><tr><td>Method</td><td>DSC (%)</td><td>HD95 (mm)</td><td>IoU (%)</td><td>p-value</td></tr><tr><td>U-Net</td><td> $6 6 . 1 2 \pm 0 . 3 1$ </td><td> $1 0 . 7 2 \pm 0 . 4 5$ </td><td> $5 2 . 4 5 \pm 0 . 2 8$ </td><td></td></tr><tr><td>Attn U-Net</td><td> $6 6 . 6 1 \pm 0 . 2 8$ </td><td> $1 3 . 5 8 \pm 0 . 6 2$ </td><td> $5 3 . 0 1 \pm 0 . 2 4$ </td><td>0.023</td></tr><tr><td>U-Net++</td><td> $6 6 . 1 5 \pm 0 . 3 5$ </td><td> $1 3 . 7 1 \pm 0 . 5 8$ </td><td> $5 2 . 5 1 \pm 0 . 3 1$ </td><td>0.891</td></tr><tr><td>nnU-Net (2D)</td><td> $7 4 . 4 8 \pm 0 . 4 2$ </td><td> $7 . 3 1 \pm 0 . 3 8$ </td><td> $6 2 . 1 5 \pm 0 . 3 6$ </td><td>0.001</td></tr><tr><td>TransUNet</td><td> $7 2 . 2 5 \pm 0 . 5 5$ </td><td> $8 . 4 9 \pm 0 . 5 1$ </td><td> $5 9 . 7 8 \pm 0 . 4 4$ </td><td>0.003</td></tr><tr><td>RIS-UNet</td><td> $7 6 . 7 9 \pm 0 . 4 8$ </td><td> $1 0 . 1 1 \pm 0 . 4 3$ </td><td> $6 5 . 2 8 \pm 0 . 3 9$ </td><td>&lt;0.001</td></tr><tr><td>M-Net (Ours)</td><td> ${ \bf 7 8 . 3 8 \pm 0 . 3 6 }$ </td><td> ${ \bf 6 . 9 1 \pm 0 . 3 3 }$ </td><td> ${ \bf 6 5 . 1 2 \pm 0 . 3 1 }$ </td><td>&lt;0.001</td></tr></table>

Table 6: Statistical comparison on KiTS kidney segmentation and BraTS brain tumor segmentation (mean ± std over 3 seeds). Best mean results in bold.
<table><tr><td></td><td colspan="3">KiTS (Kidney Seg.)</td><td colspan="3">BraTS (Tumor Seg.)</td></tr><tr><td>Method</td><td>DSC (%)</td><td>HD95 (mm)</td><td>p-value</td><td>DSC (%)</td><td>HD95 (mm)</td><td>p-value</td></tr><tr><td>U-Net</td><td> $7 2 . 5 8 \pm 0 . 4 4$ </td><td> $8 . 5 1 \pm 0 . 3 9$ </td><td></td><td> $7 8 . 0 5 \pm 0 . 5 2$ </td><td> $5 . 9 2 \pm 0 . 4 1$ </td><td></td></tr><tr><td>nnU-Net (2D)</td><td> $7 5 . 2 9 \pm 0 . 3 8$ </td><td> $6 . 2 2 \pm 0 . 3 5$ </td><td>0.004</td><td> $8 1 . 9 8 \pm 0 . 4 8$ </td><td> $4 . 2 5 \pm 0 . 3 6$ </td><td>0.002</td></tr><tr><td>M-Net (Ours)</td><td> $\mathbf { 7 6 . 1 2 \ : \pm { \ : 0 . 4 1 } }$ </td><td> ${ \bf 5 . 8 2 \pm 0 . 3 1 }$ </td><td>&lt;0.001</td><td> ${ \bf 8 3 . 6 1 \pm 0 . 4 4 }$ </td><td> ${ \bf 3 . 9 7 \pm 0 . 2 9 }$ </td><td>&lt;0.001</td></tr></table>

Absolute DSC interpretation. We acknowledge that the absolute LiTS DSC (78.42%) is lower than values reported by some 3D methods in the literature (which often exceed 90%). This is attributable to four methodological choices: (i) 2D slice-based evaluation—our model processes individual axial slices without 3D spatial context, whereas 3D architectures exploit volumetric coherence; (ii) no 3D post-processing—we do not apply 3D connected component analysis or conditional random field smoothing; (iii) full-slice evaluation—the DSC is computed over all axial slices including those with minimal or no liver presence, which reduces the average; and (iv) oficial holdout protocol—we use the LiTS leaderboard test set with held-out annotations rather than cross-validation on the training set. The large relative improvement over the identically configured U-Net baseline (+12.37%) demonstrates that the mathematical priors are efective regardless of the absolute performance ceiling imposed by the 2D setting.

## 5.5 Computational Eficiency

Table 7 reports the computational overhead.

Table 7: Computational eficiency comparison on LiTS dataset.
<table><tr><td>Method</td><td>Parameters (M)</td><td>FLOPs (G)</td><td>Inference (ms)</td><td>Training (hrs)</td></tr><tr><td>U-Net</td><td>31.04</td><td>55.0</td><td>12.3</td><td>18.5</td></tr><tr><td>M-Net</td><td>31.78</td><td>58.4</td><td>14.1</td><td>20.2</td></tr><tr><td>Overhead</td><td>+2.4%</td><td>+6.2%</td><td>+14.6%</td><td>+9.2%</td></tr></table>

M-Net introduces minimal parameter overhead (+2.4%) since the SFE and PFO modules contain no learnable parameters, and the MAG only adds lightweight gating weights. The modest computational increase is justified by the significant accuracy gains.

## 5.6 Training Convergence

Figure 2 shows validation DSC curves for all model variants.

![](images/6aa60b07b51bd2887afed8c4c609b0bbb56aae9a0d4aca8afc9a1743e01af50c.jpg)

![](images/ff92f654f6e784febfd70d84628cda9130f64a6421b6d6677c9169ccf8a477c5.jpg)  
Figure 2: Validation DSC curves across all model variants. (Left) Main comparison: U-Net baseline (gray), kappa input concat (orange), and kappa-MAG all skips (red). (Right) Placement ablation: MAG-all (red), MAG-shallow-only (blue), and MAG-deep-only (purple). The kappa-MAG all-skips variant consistently maintains the highest validation DSC throughout training and converges to the best final performance.

## 5.7 Qualitative Results

Figure 3 shows representative predictions on real LiTS CT scans.

## 5.8 Condition-Number Prior Analysis

Figure 4 provides a detailed analysis of the condition-number prior on real LiTS CT data.

The analysis confirms that condition-number values peak at organ-tissue interfaces (e.g., liver boundary), making the MAG gating mechanism particularly efective for boundary-sensitive segmentation tasks. The use of centered matrices ensures that homogeneous regions (e.g., liver parenchyma) have low κ values, while boundary regions exhibit high values, providing a clean discriminative signal for the attention mechanism.

## 6 Discussion

## 6.1 Analysis of the Centered Condition Number Feature

The condition number feature, when computed on mean-centered local matrices as defined in Definition 3.1, provides a mathematically principled measure of local texture complexity. The mean-centering step, as detailed in Remark 3.1, is not merely a numerical convenience but a theoretical necessity: without it, homogeneous regions would yield arbitrarily large condition numbers due to the rank-1 structure of constant matrices, fundamentally contradicting the intended semantic interpretation.

As established in Proposition 3.1, the centered condition number satisfies four desirable properties: continuity (enabling gradient-based optimization), scale invariance (ensuring robustness to intensity scaling), boundary sensitivity (detecting tissue interfaces), and smooth-region insensitivity (yielding near-zero values in homogeneous areas). In our experiments, organ and tumor boundary regions consistently exhibit condition numbers one to two orders of magnitude higher than homogeneous regions, confirming that the centered formulation provides a clean discriminative signal.

![](images/63c0efa1aa980f5ff15a3ab324afd8c22e553456ba9126bbc46b4f567c9a55ce.jpg)  
Figure 3: Qualitative comparison on three representative LiTS test cases showing complete liver segmentation. Each row shows: (left) CT slice with ground truth liver boundary in green; then ground truth mask, U-Net prediction, and M-Net (MAG) prediction overlaid on CT. All three cases display the full liver boundary from the LiTS dataset: Case 1 shows a well-defined liver with clear organ boundary; Case 2 shows a liver adjacent to the right kidney (lower-left region); Case 3 shows an irregular liver morphology. The MAG model produces tighter, more accurate liver boundaries across all cases.

![](images/6f728922f45f692becd050623c2d2db30f2d694d3e43b8b1cd9dc842cd301f72.jpg)

Figure 2: Condition-Number Prior Visualization on Real LiTS CT  
![](images/9d62a4c1f58008a690d85e432c1b3205ce4939e4ebd827f8c7fc31be7fecbfdf.jpg)

![](images/8989bc229ecd18ad5f2b302b2033be516bb2a47e53c4ed1c1847a58af7ed3747.jpg)

![](images/a3b44ae299fe4185e3c30fc526b56669f426832691cda8b1a6048d42bdc461c1.jpg)

![](images/47713125c1da6f0cff6d37f1deda64887ad63e682cbe34f5ccf0a89a90144d3d.jpg)  
Figure 4: Condition-number prior visualization on real LiTS CT data. (a) CT slice from the LiTS dataset, (b) liver ground truth mask overlaid on CT, (c) condition number map log κ computed from centered local matrices (Definition 3.1)—bright regions indicate high local texture ill-conditioning, (d) overlay of CT + κ heatmap + liver boundary, (e–f) horizontal profiles at $y = 2 5 6$ showing CT intensity and log κ with liver region highlighted, (g) scatter plot of κ vs. intensity at liver boundaries (red) vs. non-boundary (blue), (h) distribution histogram confirming boundary pixels have distinctly higher κ values.

The log-normalization step (Eq. 5) is crucial for stabilizing training. Raw condition numbers can span several orders of magnitude depending on the texture characteristics of diferent anatomical regions. Without log normalization, the SFE module produces numerically unstable gradients that hinder convergence. The choice of log(κ+1) rather than log(κ) ensures that $\hat { \kappa } ( 0 ) =$ 0, maintaining the semantic correspondence between zero condition number and homogeneous regions.

## 6.2 Multi-Channel Condition Number Aggregation

For multi-modal MRI (BraTS with 4 channels), the per-channel computation followed by channel averaging (Eq. 6) provides an efective strategy for leveraging multi-contrast information. Each MRI sequence (T1, T1-Gd, T2, FLAIR) captures diferent tissue properties: T1 provides anatomical contrast, T1-Gd highlights blood-brain barrier disruption, T2 reveals edema, and FLAIR suppresses cerebrospinal fluid signal. Computing condition numbers independently per channel and then averaging ensures that the spectral feature captures texture ill-conditioning across all relevant imaging contrasts, without favoring any single modality.

Alternative aggregation strategies (e.g., maximum across channels, weighted averaging) could be explored in future work. However, simple arithmetic mean provides a strong baseline that is robust to inter-subject intensity variations and does not introduce additional learnable parameters.

## 6.3 Role of Physical Field Operators

The divergence operator is most efective for detecting focal lesions with distinct intensity profiles—such as hypervascular liver tumors that appear as bright regions against darker liver parenchyma. Positive divergence indicates intensity sources (bright tumor cores), while negative divergence indicates intensity sinks (dark necrotic regions).

The discrete curl-like operator (Definition 3.4) captures important information at discretized image boundaries. As emphasized in Remark 3.2, this is not the continuous curl of a gradient field (which is identically zero for conservative fields); rather, it is a discrete descriptor that quantifies the inconsistency between mixed partial derivatives arising from the use of distinct Sobel kernels followed by orthogonal 1D diferences (Eq. 11). This inconsistency is maximized at non-smooth boundaries where continuous diferentiability breaks down, making the operator valuable for segmenting invasive tumor margins with complex geometries.

Importantly, both the divergence (a genuine physical operator) and the curl-like descriptor (a mathematically-motivated discrete feature) are computed via fixed-weight convolutions, ensuring their interpretations are preserved throughout training. This design choice distinguishes our approach from learnable filters that might converge to operators lacking clear physical meaning.

## 6.4 Efectiveness of Math-Attention Gate

The Math-Attention Gate addresses a fundamental limitation of simple feature concatenation: the dilution of mathematical priors in deep network layers. Our ablation study confirms that MAG provides a 1.45% Dice improvement over concatenation fusion. Analysis of attention weights reveals that MAG consistently assigns higher weights to regions with elevated condition numbers and strong divergence/curl magnitudes—precisely the regions where organs and tumors are most likely to be present.

The multi-scale integration strategy is critical. The placement ablation (Figure 2, right) shows that gating at all skip levels outperforms shallow-only by +4.27% and deep-only by +8.84%, confirming that the condition-number prior is complementary across scales.

## 6.5 Limitations and Future Work

While M-Net demonstrates promising results, several limitations warrant discussion. First, the current implementation operates on 2D slices, which may not fully exploit 3D spatial coherence. Extension to volumetric processing via 3D patch SVD and 3D vector operators is a natural next step. Second, the condition number computation via SVD, while GPU-accelerated, remains more computationally expensive than standard convolutions. Future work could explore approximate spectral methods or learnable low-rank approximations. Third, the physical operators are currently applied at a single scale; multi-scale divergence and curl pyramids could capture hierarchical boundary information. Fourth, while we establish the theoretical necessity of mean centering (Remark 3.1), the choice of 3 × 3 neighborhoods is fixed; adaptive neighborhood sizes based on local scale could further improve the spectral features.

## 7 Conclusion

We have presented M-Net, a novel medical image segmentation framework that integrates matrix spectral features and physical field operators into deep learning architectures. By introducing the continuous condition number computed on centered local pixel matrices as a spectral prior, the divergence and curl as physics-informed operators, and the Math-Attention Gate for principled multi-scale fusion, we have demonstrated that explicit mathematical inductive biases provide efective complementary information for medical image segmentation.

Extensive experiments on LiTS (liver segmentation), KiTS (kidney segmentation), and BraTS (brain tumor segmentation) demonstrate consistent improvements over baseline U-Net (+12.37%, +3.52%, +5.55% Dice, respectively) and competitive performance with state-of-the-art methods including nnU-Net. The detailed baseline reproduction protocol (Section 4.2) ensures experimental reproducibility and fair comparison. The cross-dataset generalization experiments further validate that mathematical priors provide transferable features across organs and imaging modalities.

This work establishes a principled methodology for integrating analytical mathematical priors into deep learning for medical image analysis, with rigorous theoretical foundations including the necessity of mean centering for meaningful condition-number computation. Future directions include extension to 3D volumetric data, exploration of additional spectral features (e.g., eigenvalue spectra of larger neighborhoods), and integration of other mathematical disciplines (e.g., diferential geometry, algebraic topology) as sources of inductive bias.

## References

[1] Litjens, G., Kooi, T., Bejnordi, B. E., et al. A survey on deep learning in medical image analysis. Medical Image Analysis, 42:60–88, 2017.

[2] Bilic, P., Christ, P. F., Li, H. B., et al. The liver tumor segmentation benchmark (LiTS). Medical Image Analysis, 84:102680, 2023.

[3] Heller, N., Isensee, F., Maier-Hein, K. H., et al. The state of the art in kidney and kidney tumor segmentation in contrast-enhanced CT imaging: Results of the KiTS19 challenge. Medical Image Analysis, 67:101821, 2021.

[4] Menze, B. H., Jakab, A., Bauer, S., et al. The multimodal brain tumor image segmentation benchmark (BRATS). IEEE Transactions on Medical Imaging, 34(10):1993–2024, 2015.

[5] Ronneberger, O., Fischer, P., & Brox, T. U-Net: Convolutional networks for biomedical image segmentation. In MICCAI, pp. 234–241. Springer, 2015.

[6] Oktay, O., Schlemper, J., Le Folgoc, L., et al. Attention U-Net: Learning where to look for the pancreas. arXiv preprint arXiv:1804.03999, 2018.

[7] Zhou, Z., Siddiquee, M. M. R., Tajbakhsh, N., & Liang, J. UNet++: A nested U-Net architecture for medical image segmentation. In DLMIA@MICCAI, pp. 3–11. Springer, 2018.

[8] Isensee, F., Jaeger, P. F., Kohl, S. A. A., Petersen, J., & Maier-Hein, K. H. nnU-Net: A self-configuring method for deep learning-based biomedical image segmentation. Nature Methods, 18(2):203–211, 2021.

[9] Chen, J., Lu, Y., Yu, Q., et al. TransUNet: Transformers make strong encoders for medical image segmentation. arXiv preprint arXiv:2102.04306, 2021.

[10] Cao, H., Wang, Y., Chen, J., et al. Swin-UNet: Unet-like pure transformer for medical image segmentation. In ECCV, pp. 205–218. Springer, 2022.

[11] Long, J., Shelhamer, E., & Darrell, T. Fully convolutional networks for semantic segmentation. In CVPR, pp. 3431–3440, 2015.

[12] Çiçek, Ø., Abdulkadir, A., Lienkamp, S. S., Brox, T., & Ronneberger, O. 3D U-Net: Learning dense volumetric segmentation from sparse annotation. In MICCAI, pp. 424–432. Springer, 2016.

[13] Milletari, F., Navab, N., & Ahmadi, S. A. V-Net: Fully convolutional neural networks for volumetric medical image segmentation. In 3DV, pp. 565–571. IEEE, 2016.

[14] Dosovitskiy, A., Beyer, L., Kolesnikov, A., et al. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020.

[15] Liu, Z., Lin, Y., Cao, Y., et al. Swin Transformer: Hierarchical vision transformer using shifted windows. In ICCV, pp. 10012–10022, 2021.

[16] Hatamizadeh, A., Tang, Y., Nath, V., et al. UNETR: Transformers for 3D medical image segmentation. In WACV, pp. 574–584, 2022.

[17] Huang, X., Deng, Z., Li, D., Yuan, X., & Fu, Y. MISSFormer: An efective medical image segmentation transformer. arXiv preprint arXiv:2109.07162, 2022.

[18] Liu, Y., Sangineto, E., Bi, W., Sebe, N., & Wang, W. DcT: A dice loss based cross-attention vision transformer for medical image segmentation. In ICME, pp. 1–6. IEEE, 2022.

[19] Hu, J., Shen, L., & Sun, G. Squeeze-and-excitation networks. In CVPR, pp. 7132–7141, 2018.

[20] Woo, S., Park, J., Lee, J. Y., & So Kweon, I. CBAM: Convolutional block attention module. In ECCV, pp. 3–19, 2018.

[21] Azad, R., Aghdam, E. K., Rauland, A., et al. UNETR++: Delving into eficient and accurate 3D medical image segmentation. In MICCAI, pp. 679–689. Springer, 2023.

[22] Karniadakis, G. E., Kevrekidis, I. G., Lu, L., et al. Physics-informed machine learning. Nature Reviews Physics, 3:422–440, 2021.

[23] Raissi, M., Perdikaris, P., & Karniadakis, G. E. Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial diferential equations. Journal of Computational Physics, 378:686–707, 2019.

[24] Peiris, H., Hayat, M., Chen, Z., et al. A volumetric transformer for accurate 3D tumor segmentation in CT scans. Applied Sciences, 12(11):5642, 2022.

[25] Bateson, M., Kervadec, H., Dolz, J., et al. Source-relaxed domain adaptation for image segmentation. In MICCAI, pp. 490–499. Springer, 2021.

[26] Liu, K., & Cheng, Y. Q. Singular value decomposition for texture analysis. In SPIE, Vol. 2298, pp. 407–417, 1994.

[27] Hafiane, A., Seetharaman, G., & Zavidovique, B. Texture classification based on spectrum and rank features. In ICIP, pp. 3422–3426. IEEE, 2019.

[28] Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. Image quality assessment: from error visibility to structural similarity. IEEE Transactions on Image Processing, 13(4):600– 612, 2004.

[29] Kovesi, P. Invariant measures of image features from phase information. PhD Thesis, University of Western Australia, 1996.

[30] Horn, B. K. P., & Schunck, B. G. Determining optical flow. Artificial Intelligence, 17(1- 3):185–203, 1981.

[31] Rudin, L. I., Osher, S., & Fatemi, E. Nonlinear total variation based noise removal algorithms. Physica D, 60(1-4):259–268, 1992.

[32] Zhu, J., Wang, Y., & Wang, F. Leveraging matrix invertibility as features in neural networks for medical image segmentation. In preparation, 2024.

[33] Stewart, G. W., & Sun, J. G. Matrix Perturbation Theory. Academic Press, 1990.

[34] Lv, P., Wang, J., & Wang, H. 2.5D lightweight RIU-Net for automatic liver and tumor segmentation from CT. Biomedical Signal Processing and Control, 75:103567, 2022.