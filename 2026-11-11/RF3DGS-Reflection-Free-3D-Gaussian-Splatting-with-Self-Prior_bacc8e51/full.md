# RF3DGS: Reflection-Free 3D Gaussian Splatting with Self-Priors

Qiang Wen<sup>1†</sup> Guotao Meng<sup>1†</sup> Junhong Ruan<sup>1†</sup> Hongyu Liu<sup>1</sup> Chenyang Lei<sup>2</sup> Qifeng Chen<sup>1</sup> <sup>1</sup>The Hong Kong University of Science and Technology <sup>2</sup>Huawei Research

![](images/321249f9b3804894a2e2ab77f815a5b40db54f9eb6c4bda6ecc9ada3632c1f1e.jpg)  
Monocular Video (55 Views)

![](images/0eb002b783747eedbe0ad0dfc5adaa743c78659af006c88536d3c87f6bb2b193.jpg)  
Reconstructed 3D scene

![](images/5eacb811afcdb5005be96ce520aa854cb9c1904b170d3d71646c2c17d538e8f1.jpg)

![](images/fe789e75db63573198241b315e76b354b011338d15a85fbe2d074e060e2771df.jpg)

![](images/8e911f38c86ca7a82a2b5227075349625ea715dc87f8632c0d896433a4c41c82.jpg)

![](images/cddad3fe01e3bf980c2274912beef629409f64ecff96e8adaa4d73bc9b08daa3.jpg)  
Fig. 1. Given a monocular video captured in a real-world scenario, the proposed method demonstrates its effective and efficient reflec tion removal performance. Compared with vanilla 3DGS [17], the proposed method exhibits greater robustness in real-world scenarios. Compared with the state-of-the-art 3D reflection removal method, NeRFReN [10], the proposed method not only proves effective but also excels in efficiency, enabling real-time rendering.

## Abstract

3D reflection removal is crucial in applications demanding high-fidelity visualization, such as museum exhibitions. In this work, we propose Reflection-Free 3D Gaussian Splatting (RF3DGS), a novel method for superior reflection removal. RF3DGS exclusively leverages a widely-adopted photometric priorfor reflection removal: that the minimum pixel intensity across multi-view images corresponding to the same 3D point strongly correlates with reflection-free information. However, generalizing this principle to a 3D Gaussian Splatting within a multi-view context is nontrivial. To address this, RF3DGS first optimizes a vanilla 3DGS model using a geometric prior to estimate a coarse, reflection-free depth map for each view. Leveraging these depth maps and photometric prior, a 3D warping process aggregates reflection-free information from auxiliary views to construct a clean transmission pseudo-label for a given <sup>†</sup> Joint first authors.

reference view. This pseudo-label provides a robust selfsupervised signal for optimization. We further introduce an Adaptive Twin Rendering (ATR) loss to synergistically utilize this pseudo-label while preserving high-frequency details from the original, reflection-degraded images. By adeptly fusing these internal 3D self-priors with the ATR loss, RF3DGS demonstrates state-of-the-art reflection removal performance across both synthetic and real-world scenarios. Furthermore, we establish the first 3D reflection removal benchmark, based on the Unity rendering engine, for rigorous quantitative evaluation.

## 1. Introduction

Capturing photos with undesirable reflections is common in daily scenarios, such as exhibitions shown in Fig. 1. Reflection removal seeks to restore clean transmission for better visual quality and improved downstream computer vision

performance [31].

However, removing reflections in real-world scenarios remains ill-posed. Learning-based methods improve robustness with additional priors, e.g., edge maps [6], polarization [20], textual prompts [12, 53], and flash/noflash cues [21]. Due to limited scene coverage in current datasets [20, 23, 26, 29, 35, 39, 51, 55], such methods still suffer from domain gap and often fail under strong reflections or non-ideal guidance. Multi-frame methods [1, 4, 7, 9, 24, 48] can aggregate clean information across frames, yet priors from limited neighbors (e.g., optical flow) remain too weak for strong reflections or significant motions [4, 24].

Thanks to progress in 3D neural scene representations [17, 28], 3D reflection removal methods [8, 10, 41, 47, 54] show strong robustness via multi-view consistency and crafted assumptions. Both NeRF-based (e.g., TNeRF [54], NeRFReN [10]) and 3DGS-based methods (e.g., DC-Gaussian [41], Flash-Splat [47]) perform well. However, NeRFReN’s long training and rendering times hinder practicality, while pixel-wise optimization limits global perceptual quality. Flash-Splat exploits real-time 3DGS rendering [17], but its reliance on flash/no-flash cues [21, 45, 46] in RAW images complicates capture and SfM; these cues also weaken as camera–scene distance grows.

To inherit the robustness and overcome the weaknesses of current 3D reflection removal methods, we propose a novel reflection-free 3DGS method for the 3D reflection removal task. The proposed method simplifies the capture process by requiring only a monocular sRGB video, easily obtained with any mobile phone or DSLR camera, without relying on specialized photography techniques such as capturing flash or no-flash RAW images [47]. Inspired by Li et al. [24], we leverage the Photometric Prior in realworld reflection scenarios: for a given 3D scene point, the minimum intensity value across all captured views strongly corresponds to reflection-free information. Since pixel values from the captured views must be accurately projected back into 3D space, obtaining precise depth maps is crucial to effectively apply the photometric prior. Benefiting from explicit depth estimation in 3DGS, we can efficiently generate depth maps for each captured view by optimizing a vanilla 3DGS model under reflection conditions. To ensure the depth maps contain as pure reflection-free depth information as possible, we incorporate a Geometric Prior—also used by NeRFReN [10]—within 3DGS, which encourages smoothness in the rendered depth maps. This geometric prior assumes that the depth map of the transmitted component should be locally smoother than that of the reflection-degraded one. Utilizing these refined depth maps, we project all captured views back into the 3D space and apply the photometric prior, then re-render each view with preliminary clean transmission information. Due to the flaws of depth maps and imperfect projections, current reflection-free views still have inevitable artifacts, as shown in Fig. 4. Directly optimizing the 3DGS on them leads to poor 3D consistency and color distortion. To address this issue, the proposed method goes into the comprehensive optimization where original captured views and preliminary reflection-free views are jointly treated as supervision during 3DGS optimization. We propose an Adaptive Twin Rending (ATR) loss to achieve this joint supervision, encouraging the 3DGS model to effectively capture the clean transmission information from preliminary reflectionfree views and maintain the original 3D consistency for a reflection-degraded scene.

Furthermore, recognizing that existing benchmarks provide limited scene diversity—for example, TNeRF [54] includes only four scenes—for quantitatively evaluating 3D reflection removal methods, we introduce a comprehensive 3D reflection removal benchmark aimed at significantly enhancing evaluation diversity. Since capturing videos with and without reflections along the exact same camera trajectory is challenging, we leverage the precise camera control capabilities of the modern rendering engine Unity [11] to construct our benchmark. To guarantee the diversity of our benchmark, we build 3D scenes with different settings (background & reflection scenes, glass thickness, transparency and reflection strength), resulting in 45 cases for evaluation. Besides, we additionally collect realworld cases for qualitative evaluation. Extensive experiments demonstrate that the proposed method significantly outperforms previous works on the proposed benchmark and across various real-world scenarios. This strong robustness and simplicity of our method indicate its potential for practical applications.

## 2. Related Work

## 2.1. Single-Image Reflection Removal

With the rapid development of deep learning techniques, learning-based methods [5, 6, 12–14, 23, 27, 30, 36–38, 42– 44, 49, 51–53] are chosen as the first alternative compared with early traditional methods [2, 9, 22, 25, 32, 48, 50, 52]. Fan et al. [6] propose the pioneer learning-based method, CEILNet, which first estimates an edge map as the assisted cue for the subsequent transmission layer estimation. Zhang et al. [51] propose to utilize perceptual loss [15] to train a deep network by considering misalignments, which are caused by glass refraction, between the estimated and ground-truth transmission layers. Wen et al. [44] propose the generative-based reflection synthesis and removal networks for training the removal network based on the nonlinear synthetic data. Kim et al. [18] propose a physically based rendering pipeline for data synthesis for subsequent removal network training. Hu et al. [14] propose to capture a learnable residue term during input image decomposition, which guides a further separation between transmission and reflection layers. Zhu et al. [55] propose a cascaded method, using their proposed maximum reflection filter to locate reflection for guiding subsequent removal. Most recently, Zhong et al. [53] and Hong et al. [12] propose language-guided methods by utilizing text prompts to aid reflection removal.

![](images/04db88cdf927390f8b7c25ec4a3a7aa2783feb4b362e910c90fc6342959e54ba.jpg)  
Fig. 2. An overview of the proposed method. 1) During Initialization, all captured views are processed through structure-from-motion (SfM) to generate an initial 3D scene point cloud, which is then used to initialize a 3DGS model; 2) During Geometric Prior Utilization, the 3DGS model is optimized using learnable residual maps $\hat { E } _ { n }$ combined with the depth smoothness geometric prior to generate coarse reflection-free depth maps ${ \hat { D } } _ { n } ; 3 )$ During Photometric Prior Utilization, using the reflection-free depth maps, all captured views are projected back onto the 3D scene point cloud. Subsequently, for re-rendering each specific view, a minimum value-based photometric prior is applied to extract preliminary clean transmission information; 4) During Comprehensive Optimization, three 3DGS models are jointly optimized for producing transmission image $\hat { T } _ { n }$ , reflection image $\hat { R } _ { n }$ and reflectance factor ${ \hat { \beta } } _ { n } ,$ respectively.

While learning-based methods demonstrate strong performance in benchmark evaluations, the challenge of domain adaptation limits their effectiveness in real-world daily scenarios.

## 2.2. Multi-frame Reflection Removal

Compared to single-image methods, multi-frame methods [1, 7, 9, 24, 26, 48] show their superiority with exploiting useful information from multiple frames. Alayrac et al. [1] and Liu et al. [26] propose learning-based methods which are trained on synthetic data. Such methods still suffer from domain-gap problems when they are applied on real-world scenarios. Despite learning-based methods, traditional multi-frame methods focus on exploiting image structural priors [9, 24] or motion priors [7, 48] for separating transmission and reflection layers. Most recently, Chugunov et al. [4] propose neural spline fields for burst frames reflection removal. However, these multi-frame methods face limitations when dealing with significant camera motion, particularly with large changes in viewing angles. Other works seek to enhance reflection removal performance by capturing additional data through optical priors, such as polarization-based methods [20], flash/no-flash techniques [21], and RAW-domain approaches [16]. These methods are limited by their dependence on additional data acquisition, which increases complexity and may not be practical in most everyday situations.

## 2.3. 3D Reflection Removal

There are some existing NeRF-based methods [8, 10, 54] for the 3D reflection removal task. TNeRF [54] introduces a pre-trained transmission encoder [43] and recurring edge constraints as guidance into the NeRF backbone to improve its reflection removal performance. Guo et al. [10] propose a NeRF-based method, NeRFReN, by adopting a similar image formulation as Shina et al. [33]. The effectiveness of NeRFReN can be attributed to three main aspects: the incorporation of a depth smoothness prior to within the transmission layers, the implementation of bidirectional depth consistency, and the observation that reflections exhibit simpler geometrical characteristics compared to transmission layers. Gao et al. [8] uses a similar architecture to NeR-FReN but models planar surfaces and casts reflected rays to capture reflections, relying on a pre-trained segmentation model [19] to initialize window surfaces.

There are some existing 3DGS-based methods [34, 41, 47]. However, DC-Gaussian [41] removes reflections from multi-view transmission images degraded by a single-view reflection image, while our work tackles the more complex case of multi-view transmission images degraded by multiview reflection images. The most recent and similar work to ours is Flash-Splat [47]. Flash-Splat utilizes flash/no-flash cues [21, 45, 46] for optimizing 3DGS [17] with unpaird flash/no-flash views in RAW format. However, the major limitation of flash-based methods is that the effectiveness of flash cues rapidly diminishes as the distance between the camera and the target scene increases. In this work, the proposed method does not need any additional photography techniques but multi-view captures that can be simply captured in a video mode using the sRGB format. This prevents the proposed method from being limited by any constraints.

## 3. Method

In this work, we define the 3D reflection removal task as follows: Given a set of multi-view captures ${ \mathcal { T } } _ { N } = \{ I _ { n } , n \in$ N}, where N denotes the view numbers, our objective is to achieve 3D reflection removal by reconstructing transmission and reflection scenes separately.

We follow prior work [10, 47] and model each reflectiondegraded observation as:

$$
I _ { n } = T _ { n } + \beta _ { n } \circ R _ { n } ,\tag{1}
$$

where n denotes view index, $T _ { n } \ R _ { n }$ , and $\beta _ { n }$ denote transmission image, reflection image and reflective fraction map.

We reconstruct the scene with 3D Gaussian Splatting (3DGS) [17]. Each Gaussian is parameterized by a mean $x ~ \in ~ \mathbb { R } ^ { 3 }$ , a covariance Σ factored as $\Sigma \ = \ \bar { R S } S ^ { T } R ^ { T }$ opacity $\alpha ,$ and view-dependent color c encoded by spherical harmonics. A pixel color is obtained by front-to-back α-blending of the $\mathcal { N }$ overlapping Gaussians:

$$
C = \sum _ { i \in \mathcal { N } } c _ { i } \alpha _ { i } \prod _ { j = 1 } ^ { i - 1 } ( 1 - \alpha _ { j } ) .\tag{2}
$$

Vanilla 3DGS is optimized with a combination of $\mathcal { L } _ { 1 }$ and D-SSIM:

$$
\mathcal { L } _ { \mathrm { v a n i l l a } } = \lambda \mathcal { L } _ { 1 } ( \hat { I } , I ) + ( 1 - \lambda ) \mathcal { L } _ { \mathrm { D - S S I M } } ( \hat { I } , I ) ,\tag{3}
$$

with λ = 0.2. We reuse this photometric term in later stages and refer readers to [17] for rasterization and densification details.

## 3.1. Reflection-Free 3D Gaussian Splatting

In this work, we propose a novel reflection-free 3DGS method. Unlike the recent 3DGS-based method, Flash-Splat [47], which requires flash / no-flash multi-view RAW image captures, the proposed method needs no additional photography techniques but only multi-view sRGB images captured by a monocular video. This allows the proposed method to be free from the effectiveness declining of flash / no-flash cues with distance increasing as mentioned in works [21, 45–47].

## 3.1.1. Geometric Prior Utilization

Although vanilla 3DGS achieves impressive rendering quality for 3D reconstruction, it often produces floating artifacts and residual reflections in the final outputs during reconstructing a reflection-degraded scenario. To mitigate these issues, we incorporate a geometric prior based on depth smoothness constraints [10], which helps generate cleaner and more accurate depth maps. Specifically, we apply a robust depth smoothness loss together with learnable residual maps that capture unwanted reflection components (see Fig. 3), enhancing the optimization of 3DGS on our reflection-degraded inputs:

$$
\mathcal { L } _ { \mathrm { G e o m } } = \mathcal { L } _ { \mathrm { v a n i l l a } } ( \hat { I _ { n } } + \tilde { E } _ { n } , I _ { n } ) + \mathcal { L } _ { \mathrm { s m o o t h } } ( \hat { D } _ { n } ) ,\tag{4}
$$

where $\hat { E } n$ is the learned residual map, $\hat { D } n$ is the rendered depth map, and $\mathcal { L } _ { \mathrm { { s m o o t h } } }$ is a bilateral filter loss [47] promoting local smoothness.

The key purpose of utilizing this geometric prior is to produce reflection-free depth maps with high accuracy and purity. These improved depth maps are essential for the precise application of the photometric prior, allowing more reliable projection of pixel values into 3D space and better performance of the following photometric prior utilization.

## 3.1.2. Photometric Prior Utilization

Inspired by Li et al. [24], we observe that, for a point in a 3D scene, the minimum value across all captured views is highly correlated with representing reflection-free transmission. Based on this photometric prior, for each reference view, we obtain its corresponding reflection-free version by the following operations: 1) Perform 3D warping of all auxiliary views using their corresponding over-smooth but reflection-free depth maps rendered from the geometric prior utilization; 2) Calculate the minimum value of all these warped auxiliary views and the reference view at each spatial position; 3) We treat this minimum map as the reflection-free version of the reference view, and regard it as our additional supervised signal during the following comprehensive optimization, a detailed illustration is shown in Fig. 4. These operations can be formulated as

$$
\begin{array} { r } { I _ { n } ^ { \mathrm { f r e e } } = \operatorname* { m i n i m u m } ( I _ { n } , \{ \mathcal { W } ( I _ { m } , \hat { D } _ { m } , c _ { m  n } ) \} ) , } \end{array}\tag{5}
$$

where $m \in N$ and m $\neq n , \mathcal { W } ( \cdot )$ denotes depth-based 3D warping function, $c _ { * }$ denotes the camera parameters of the captured view $I _ { * }$ . Although there are noticeable warping errors in $\mathcal { T } _ { N } ^ { \mathrm { f r e e } }$ shown in Fig. 4, we find that their reflectionfree information contributes positively to the following optimization in 3DGS. With a proposed ATR loss, we can effectively mitigate the impact of these errors, leading to improved final clean transmission results.

![](images/3691398c7a25253a5d21d2d3bd5722e3b65788353882ba470931c7201fd96762.jpg)

Fig. 3. The significance of learnable residual maps during geometric prior utilization. Introducing residual maps significantly enhances the utilization of the geometric prior, resulting in cleaner, reflection-free depth maps.  
![](images/e9cf509c1c0fce57a91a72c0339e447ea803a2943e11bc866a4cd7ef0ffdba10.jpg)  
Fig. 4. Illustration of depth-based 3D warp. $c _ { * }$ denotes the camera parameters of the captured view $I _ { * }$ . For certain meaningless areas in the warped image $I _ { m  n } ,$ , we fill these areas with pure white color.

## 3.1.3. Comprehensive Optimization

To comprehensively optimize the 3DGS model for the final clean 3D scene, we initialize three 3DGS models for transmission, reflection and reflective fraction, respectively. During optimization, we simultaneously optimize the three 3DGS models using several loss functions to achieve a balance, ensuring pure transmission and reflection with complete separation.

Firstly, the vanilla 3DGS loss based on Eq. 1 is applied:

$$
\mathcal { L } _ { \mathrm { v a n i l l a } } = \lambda _ { \mathcal { L } _ { 1 } } \mathcal { L } _ { 1 } ( \hat { I } _ { n } ^ { \cdots } , I _ { n } ) + ( 1 - \lambda _ { \mathcal { L } _ { 1 } } ) \mathcal { L } _ { \mathrm { D - S S I M } } ( \hat { I } _ { n } ^ { \cdots } , I _ { n } ) ,\tag{6}
$$

where ${ \hat { I } } _ { n } ^ { \cdots } = { \hat { T } } _ { n } + { \hat { \beta } } _ { n } \circ { \hat { R } } _ { n } , { \hat { T } } _ { n } , { \hat { \beta } } _ { n }$ and $\hat { R } _ { n }$ are rendered from their corresponding 3DGS.

Secondly, to give explicit guidance for optimizing the transmission 3DGS model, a naive L1 or MSE function can be applied to its rendered results: $\mathcal { L } _ { \mathrm { T } } = \mathcal { L } _ { \mathcal { L } _ { 1 } / \mathrm { M S E } } ( T _ { n } , I _ { n } ) +$ $\mathcal { L } _ { \mathcal { L } _ { 1 } / \mathrm { M S E } } ( T _ { n } , I _ { n } ^ { \mathrm { f r e e } } )$ . However, during optimization, we find that the rendered result tends to be collapsed into one of $I _ { n }$ (is fine-grained but suffers from reflection degradation) and $I _ { n } ^ { \mathrm { f r e e } }$ (is free from reflections but exhibits artifacts). Instead of empirically assigning reasonable weights to these two losses, we propose a more intuitive loss, Adaptive Twin Rendering (ATR) loss, that automatically balances them, enabling the transmission 3DGS model to effectively learn the optimal information from both sources:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { A T R } } = \operatorname* { m i n i m u m } ( \left\| T _ { n } - I _ { n } \right\| _ { 2 } ^ { 2 } , \left\| T _ { n } - I _ { n } ^ { \mathrm { f r e e } } \right\| _ { 2 } ^ { 2 } ) . } \end{array}\tag{7}
$$

This loss works based on two facts: 1) Compared to reconstructing transmission and reflection simultaneously, a single 3DGS model tends to only reconstruct one of them at beginning, $\mathcal { L } _ { \mathrm { A T R } }$ encourages the 3DGS model to prioritize fitting the pure transmission information in $I _ { n } ^ { \mathrm { f r e e } }$ rather than the more complex reflection-degraded information in $I _ { n } ; 2 )$ The 3DGS model tends to reconstruct 3D consistent information, the 3D inconsistent errors in $I _ { n } ^ { \mathrm { f r e e } }$ are always ignored when there is more reasonable information in $I _ { n }$

Thirdly, to encourage the reflection 3DGS model to refrain from reconstructing transmission information, we adopt the exclusion loss [51], $\Phi _ { \mathrm { e x c l } } ( I _ { 1 } , I _ { 2 } )$ , minimizing the similarity between the rendered transmission and reflection images. This loss aims to ensure that the rendered outputs remain as distinct and pure as possible:

$$
\mathcal { L } _ { \mathrm { e x c l } } = \mathcal { L } _ { \mathrm { e x c l } } ( \hat { T } _ { n } , \hat { R } _ { n } ) + \mathcal { L } _ { \mathrm { e x c l } } ( I _ { n } ^ { \mathrm { f r e e } } , \hat { R } _ { n } ) .\tag{8}
$$

Fourthly, the bilateral filter loss for encouraging local smoothness is slightly imposed on rendered depths of transmission and reflection scenes, as well as reflective fraction maps. This loss is represented as $\mathcal { L } _ { \mathrm { { s m o o t h } } }$

Finally, we combine the above losses with corresponding weights for our comprehensive optimization:

$$
\begin{array} { r } { \mathcal { L } = \lambda _ { 1 } \mathcal { L } _ { \mathrm { v a n i l l a } } + \lambda _ { 2 } \mathcal { L } _ { \mathrm { A T R } } + \lambda _ { 3 } \mathcal { L } _ { \mathrm { e x c l } } + \lambda _ { 4 } \mathcal { L } _ { \mathrm { s m o o t h } } , } \end{array}\tag{9}
$$

where λ<sub>1</sub> = 1.0, λ<sub>2</sub> = 1.0, λ<sub>3</sub> = 0.01 and $\lambda _ { 4 } = 0 . 0 1$

## 4. 3D Reflection Removal Benchmark

Table 1. Proposed 3D Reflection Removal Benchmark. Transparency: Higher values indicate clearer transmission and weaker reflection. Thickness: Affects light refraction and glass curvature. Reflection Strength: Controls the intensity of the glass’s reflection.
<table><tr><td>Configuration</td><td>Intensity Range</td><td>Total Cases</td></tr><tr><td>Transparency (%)</td><td>[88,92]</td><td></td></tr><tr><td>Thickness (mm)</td><td>[1,7]</td><td>45</td></tr><tr><td>Reflection Strength</td><td>[12, 135]</td><td></td></tr></table>

Although a real-world benchmark exists [54], it comprises only four scenes, each featuring a single setting (e.g., glass transparency, thickness), which limits its dversity for comprehensive quantitative evaluation. Besides, building a real-world 3D reflection removal benchmark with clean ground truth is challenging. Capture multiple views, but each view is captured with and without reflection before the camera movement. This necessitates multiple repositionings of the glass, resulting in spatial misalignment of the reflections. This misalignment adversely impacts the quality of 3D reflection reconstruction, ultimately degrading the performance of the final removal.

Based on the abovementioned observation and inspired by Niklaus et al. [29], we turn to build 3D reflection removal benchmark by the sophisticated rendering engine, Unity [11]. With the high-quality reflection simulation in Unity, we find that the performance of 3D reflection removal methods on our proposed benchmark is highly consistent in real-world scenarios. Precise camera movement control in Unity, combined with flexible reflection on/off switching, allows us to generate high-quality ground-truth transmission images for each view. Considering the complexity of real-world scenarios, we build our benchmark considering various real-world reflecting factors, including the transparency, thickness and reflection strength of glasses provided in Unity, as shown in Table. 1. Ultimately, we construct 5 distinct 3D scenes by Unity, each with 9 variants, resulting in 45 cases in total for our proposed benchmark.

## 5. Experiemnts

## 5.1. Implementation Details

During the initialization, we employ VGGSfM [40] to create the initial point cloud. For a fair comparison, we consistently utilize VGGSfM across all 3D methods in this study. During geometric prior utilization, the 3DGS model is optimized using the standard 3DGS settings as described by Kerbl et al. [17], with 5,000 iterations. During comprehensive optimization, three separate 3DGS models are simultaneously optimized using the same hyperparameters but with different loss functions, each undergoing 30,000 iterations.

## 5.2. Baselines

We compare the proposed method against several strong baselines:

• 3D-based method: MVSNeRF, TNeRF, NeRFReN, vanilla 3DGS;

• Burst-imaging-based method: Chugunov et al. [4];

• Learning-based image method combined with 3DGS: DSRNet [14] + vanilla 3DGS.

## 5.3. Quantitative Evaluation

We conduct a comprehensive quantitative evaluation on both our proposed benchmark and the real-world benchmark [54] of various reflection removal methods, as presented in Table 2 and Table 3.

Table 2. Quantitative evaluation on our proposed benchmark. “DSRNet + Vanilla 3DGS” indicates that degraded views are pre-processed using a state-of-the-art single-image method before 3DGS optimization. The burst-imaging-based method proposed by Chugunov et al. [4] fails to generate outputs with reasonable evaluation scores, indicating its inability to address the 3D reflection removal problem.
<table><tr><td>Method</td><td>PSNR↑</td><td>SSIM ↑</td><td>LPIPS↓</td></tr><tr><td>Chugunov et al. [4]</td><td>2.94</td><td>0.248</td><td>0.935</td></tr><tr><td>Vanilla 3DGS [17]</td><td>29.58</td><td>0.917</td><td>0.175</td></tr><tr><td>DSRNet [14] + Vanilla 3DGS</td><td>26.91</td><td>0.872</td><td>0.240</td></tr><tr><td>NeRFReN [10]</td><td>26.36</td><td>0.755</td><td>0.284</td></tr><tr><td>Ours</td><td>31.45</td><td>0.915</td><td>0.150</td></tr></table>

Table 3. Quantitative evaluation on the real-world benchmark [54]. Flash-Splat cannot work on such sRGB inputs, since it requires camera RAW images.
<table><tr><td>Method</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td></tr><tr><td>MVSNeRF [3]</td><td>17.83</td><td>0.685</td><td>0.400</td></tr><tr><td>NeRFReN [10]</td><td>18.63</td><td>0.716</td><td>0.449</td></tr><tr><td>TNeRF [54]</td><td>22.75</td><td>0.841</td><td>0.205</td></tr><tr><td>Vanilla 3DGS [17]</td><td>21.73</td><td>0.835</td><td>0.238</td></tr><tr><td>DSRNet [14] + Vanilla 3DGS</td><td>22.03</td><td>0.838</td><td>0.221</td></tr><tr><td>Flash-Splat [47]</td><td>N/A</td><td>N/A</td><td>N/A</td></tr><tr><td>Ours</td><td>25.04</td><td>0.915</td><td>0.190</td></tr></table>

In Table 2, the most notable observation is that the burstimaging-based method proposed by Chugunov et al. [4] fails to generate outputs with reasonable evaluation scores, indicating its inability to address the 3D reflection removal problem. We also observe that despite being the stateof-the-art 3D reflection removal method, NeRFReN completely fails on “Scene 3”, highlighting its lack of robustness across diverse scenarios. According to PSNR, which is highly sensitive to reflection degradation, the proposed method demonstrates strong superiority in 3D reconstruction with effective reflection removal. Furthermore, LPIPS scores indicate that the proposed method achieves state-ofthe-art perceptual quality. However, due to SSIM’s insensitivity to reflection degradation, the vanilla 3DGS achieves the highest scores. This suggests that the reflection removal priors employed by NeRFReN and the proposed method may slightly affect structural consistency.

In Table 3, the proposed method demonstrates significantly superior performance compared to previous works. Notably, Flash-Splat fails to operate when only sRGB inputs are provided, as it requires camera RAW images. This highlights that the RAW image requirement can be a burden for daily use.

## 5.4. Qualitative Evaluation

In the qualitative evaluation, the proposed method showcases its strong superiority on our proposed benchmark. Furthermore, we also collect 6 real-world scenes for the qualitative evaluation.

![](images/c2c58a95cd5124525807aff627c668f736924e40558fc1c57dfc6261553f6e47.jpg)

![](images/3a2c02488ab7ca4fe6b282c1ca8eb770bfcfa2f024c4626cd14400e2c35ea212.jpg)

![](images/cff08e3fd6a1225773405a89a57a58830a5469cb42f153f743191a9c30b36a97.jpg)

![](images/afa0efe3b2920dbb46196d3c5cc9f29f8e132ac33eff919d3322356f9b55b197.jpg)

![](images/de418a5f2bf0ed3f917d969509d3b4aba5b1a33850254e3c8977734f40b23c87.jpg)  
Captured View

![](images/1b0cd7d7fd8f11efcfe86886ca616a65232138e8dcfe19083935646325ed8dfe.jpg)  
Vanilla 3DGS

![](images/e3240aad94547531902f5841a56e4e861ee253baa56fa2dd24c42d28bac9d4e4.jpg)  
DSRNet + Vanilla 3DGS  
NeRFReN

![](images/498aeafb1eed33abadbe79a87defce9dd574c2156821193ecdb47f4615e1a11f.jpg)  
Ours (T)

![](images/c17a74306cfb2d77895cf7da506c1e0d1226cab7bc154621810e66cb778c9e4e.jpg)  
Ours (Ř)

Fig. 5. Qualitative evaluation on various real-world scenarios. “DSRNet + Vanilla 3DGS” indicates that degraded views are pre-processed using a state-of-the-art single-image method, DSRNet, before 3DGS optimization. For column 3, we show the view processed by DSRNe at the bottom right corner of each rendered result  
![](images/44215e7e2ace153a11143830cca4bf30d5d88b3300d177a6982f796193f31085.jpg)  
Captured View

![](images/e6318737c27d4c228b58d796c6d1c7cf8fc565a055f96ce87b8c5aadfbdab544.jpg)  
Vanilla 3DGS

![](images/65fe56c8fa30e04684e1d3b56fb763302f4259e400e056dd946716c60dabdd15.jpg)  
DSRNet + Vanilla 3DGS

![](images/1b817f25c40dec60041e415742e6d5257173da27b8ce1688d5e17b4cbf862650.jpg)  
NeRFReN

![](images/d9497500754bf3cf9ea94fbd64fa697d62b088b9b25b9cfb6e0398b829545710.jpg)  
Ours

![](images/da4516826d9de3d55bf578e74cd180cd34fc36d2a38bca81ea08475f0d61240e.jpg)  
GT

Fig. 6. Qualitative evaluation on the proposed benchmark. “DSRNet + Vanilla 3DGS” indicates that degraded views are pre-processed using a state-of-the-art single-image method before 3DGS optimization. For row 2, as NeRFReN incorrectly identifies the transmission and reflection information, we included its predicted “reflection layer” in the bottom-right corner of the result.  
![](images/bcf0baa9c7849af8a143141380c21874e3ac21a6c5490614aa1dc475cecbd777.jpg)  
Captured View

![](images/00a71ee19361d90d5faa61572a1d681f07799ce81e49a21ce7e467fc71d4bfa1.jpg)  
LATR W/o In  
LATR W/o Ifree  
w/o Ismooth  
w/o Iexcl

![](images/da59577de7c941eab6c29a846b6430b0d4d8c122b3b3e339c20ef399c923ea07.jpg)  
Ours

![](images/99fb7d2ce56202f5f924c313a135594f1450606250095eff99e18e6ed6061c60.jpg)  
GT  
Fig. 7. Qualitative evaluation of the ablation study. The reconstructed reflection scene and rendered depth map displayed in the smal corner images.

In Fig. 5, the proposed method demonstrates robust 3D reflection removal performance in real-world scenarios. As shown in column 2, the vanilla 3DGS approach struggles with severe floating artifacts during 3D reconstruction, heavily impacted by strong reflections. To address this, we applied a state-of-the-art image-based reflection removal method, DSRNet, prior to training the vanilla 3DGS. However, the limited generalization capability of the learningbased DSRNet results in suboptimal reflection removal for each captured view, ultimately leading to degraded 3D reconstruction performance, as illustrated in column 3. We observe that the 3D method, NeRFReN, struggles to remove strong reflections across all scenes, particularly with objects like lamps, where the proposed method excels. This clearly demonstrates the effectiveness of the photometric prior utilization in the proposed method. Additionally, due to the pixel-wise supervision in NeRFReN, the method lacks satisfactory global perceptual quality in the rendered results, leading to inconsistent pixel values across all outputs. Furthermore, the proposed method also reconstructs the underlying 3D reflection scenes, which is a potential guidance for the downstream task.

In Fig. 6, column 3, we observe that while the singleimage method, DSRNet, significantly enhances the reflection removal performance for vanilla 3DGS, it also compromises the clarity of the transmission details, particularly the high-frequency textures on the walls and objects. Due to the domain gap issue, such learning-based methods lack sufficient robustness across diverse scenes, resulting in residual reflections in the final outputs. In column 4, NeRFReN fails to effectively remove reflections while preserving clear transmission details. Additionally, it faces challenges in distinguishing between transmission and reflection information, particularly in the scene depicted in row 2. The proposed method effectively removes reflections while preserving high-frequency textures in the transmission information, a particularly challenging aspect of the reflection removal task.

## 5.5. Ablation Study

We conduct a comprehensive ablation study to assess the impact of each key component in our proposed method, as shown in Table 4 and Fig. 7. Firstly, we observe that the $^ { 6 6 } \mathcal { L } _ { \mathrm { A T R } }$ w/o $I _ { n } { } ^ { \dag }$ variant, which relies solely on $I _ { n } ^ { \mathrm { f r e e } }$ as the supervised signal, suffers from severe floating artifacts and color distortion due to the photometric prior, leading to unsatisfactory evaluation scores. Next, we analyze the $^ { 6 6 } \mathcal { L } _ { \mathrm { A T R } }$ w/o $I _ { n } ^ { \mathrm { f r e e } , , }$ variant, which undergoes full optimization but uses only $I _ { n }$ as the supervised signal, lacking supervision from the $I _ { n } ^ { \mathrm { f r e e } }$ . Although this variant achieves acceptable 3D reconstruction performance, its reflection removal capability is significantly compromised. These findings underscore the effectiveness of the proposed ATR loss, demonstrating that relying solely on $I _ { n }$ or $I _ { n } ^ { \mathrm { f r e e } }$ as the supervised signal fails to achieve a satisfactory balance between 3D reconstruction and reflection removal. Finally, we evaluate the effectiveness of the exclusion loss and smooth loss using two distinct variants, as shown in Table 4 and Fig. 7. Both losses contribute to improved overall performance. The “w/o $\mathcal { L } _ { \mathrm { { s m o o t h } } } ? $ variant demonstrates that without the local smoothness constraint on depth maps, strong reflections are more likely to persist in the rendered transmission image. The “w/o $\mathcal { L } _ { \mathrm { e x c l } } { } ^ { \mathrm { ~ } }$ variant reveals that, without the exclusion loss, the reflection scene may retain transmission information, resulting in incomplete transmission reconstruction, as shown in the bottom-left corner of the rendered result. We also investigate the effect of the number of input views on the proposed method and observe that it maintains satisfactory robustness across different input view counts.

Table 4. Quantitative evaluation of the ablation study.
<table><tr><td>Method</td><td>PSNR ↑</td><td>SSIM ↑</td><td>LPIPS↓</td></tr><tr><td rowspan="3">Ours (LATR w/o In) Ours (LATR w/o  $I _ { n } ^ { \mathrm { f r e e } } )$  Ours (w/o  $\mathcal { L } _ { \mathrm { e x c l } } )$ </td><td>19.64</td><td>0.790</td><td>0.237</td></tr><tr><td>30.45</td><td>0.888</td><td>0.176</td></tr><tr><td>31.00 31.11</td><td>0.908</td><td>0.155 0.155</td></tr><tr><td>Ours (w/o  $\mathcal { L } _ { \mathrm { { s m o o t h } } } )$  Ours</td><td>31.25</td><td>0.909 0.915</td><td>0.150</td></tr><tr><td>Ours (11 views)</td><td>31.14</td><td>0.907</td><td>0.140</td></tr><tr><td>Ours (22 views)</td><td>31.27</td><td>0.912</td><td>0.146</td></tr><tr><td>Ours (88 views)</td><td>31.25</td><td>0.915</td><td>0.150</td></tr></table>

## 5.6. Processing Time

Table 5 presents a detailed comparison of training times on the proposed benchmark with 88 views per case. Our method requires 60.74 minutes to train, which is significantly faster than NeRFReN’s 63.48 hours, demonstrating a substantial improvement in efficiency. While Vanilla 3DGS and DSRNet combined with Vanilla 3DGS complete training more quickly (5.94 and 7.85 minutes respectively), these methods do not achieve the same level of performance and robustness as our method. Overall, our method strikes a strong balance between training speed and quality, offering a practical solution for large-scale or time-sensitive applications.

Table 5. The processing-time comparison on the proposed bench mark. Time represents the training time for each case (88 views).
<table><tr><td></td><td>Vanilla 3DGS</td><td>DSRNet + Vanilla 3DGS</td><td>NeRFReN</td><td>Ours</td><td>Ours (3D Warp)</td></tr><tr><td>Time</td><td>5.94 min</td><td>7.85 min</td><td>63.48 h</td><td>60.74 min</td><td>30 sec</td></tr></table>

## 6. Conclusion

In this work, we introduce a novel reflection-free 3DGS method for the 3D reflection removal task, named RF3DGS. Unlike previous works that rely on additional photography techniques, such as flash or no-flash cues, RAW images, the proposed method simply requires a monocular sRGB video capturing multiple views of a 3D scene. By leveraging the photometric prior, we propose an adaptive twin rendering (ATR) loss that effectively enhances both reflection removal and 3D reconstruction performance. Additionally, we present a 3D benchmark with substantial diversity to enable quantitative evaluation of 3D reflection removal methods. Extensive experiments on the proposed benchmark and real-world scenes demonstrate the strong superiority of the proposed method over existing methods.

## References

[1] Jean-Baptiste Alayrac, Joao Carreira, and Andrew Zisserman. The visual centrifuge: Model-free layered video representations. In CVPR, pages 2457–2466, 2019. 2, 3

[2] Nikolaos Arvanitopoulos, Radhakrishna Achanta, and Sabine Susstrunk. Single image reflection suppression. In CVPR, pages 4498–4506, 2017. 2

[3] Anpei Chen, Zexiang Xu, Fuqiang Zhao, Xiaoshuai Zhang, Fanbo Xiang, Jingyi Yu, and Hao Su. Mvsnerf: Fast generalizable radiance field reconstruction from multi-view stereo. In ICCV, pages 14124–14133, 2021. 6

[4] Ilya Chugunov, David Shustin, Ruyu Yan, Chenyang Lei, and Felix Heide. Neural spline fields for burst image fusion and layer separation. In CVPR, pages 25763–25773, 2024. 2, 3, 6

[5] Zheng Dong, Ke Xu, Yin Yang, Hujun Bao, Weiwei Xu, and Rynson WH Lau. Location-aware single image reflection removal. In ICCV, pages 5017–5026, 2021. 2

[6] Qingnan Fan, Jiaolong Yang, Gang Hua, Baoquan Chen, and David Wipf. A generic deep architecture for single image reflection removal and image smoothing. In ICCV, pages 3238–3247, 2017. 2

[7] Kun Gai, Zhenwei Shi, and Changshui Zhang. Blind separation of superimposed moving images using image statistics. IEEE Transactions on Pattern Analysis and Machine Intelligence, 34(1):19–32, 2011. 2, 3

[8] Chen Gao, Yipeng Wang, Changil Kim, Jia-Bin Huang, and Johannes Kopf. Planar reflection-aware neural radiance fields. In ACM Transactions on Graphics (SIGGRAPHAsia), pages 1–10, 2024. 2, 3

[9] Xiaojie Guo, Xiaochun Cao, and Yi Ma. Robust separation of reflection from multiple images. In CVPR, pages 2187– 2194, 2014. 2, 3

[10] Yuan-Chen Guo, Di Kang, Linchao Bao, Yu He, and Song-Hai Zhang. Nerfren: Neural radiance fields with reflections. In CVPR, pages 18409–18418, 2022. 1, 2, 3, 4, 6

[11] John K Haas. A history of the unity game engine. 2014. 2, 6

[12] Yuchen Hong, Haofeng Zhong, Shuchen Weng, Jinxiu Liang, and Boxin Shi. L-differ: Single image reflection removal with language-based diffusion model. In ECCV, 2024. 2, 3

[13] Qiming Hu and Xiaojie Guo. Trash or treasure? an interactive dual-stream strategy for single image reflection separation. Advances in Neural Information Processing Systems, 34:24683–24694, 2021.

[14] Qiming Hu and Xiaojie Guo. Single image reflection separation via component synergy. In CVPR, pages 13138–13147, 2023. 2, 3, 6

[15] Justin Johnson, Alexandre Alahi, and Li Fei-Fei. Perceptual losses for real-time style transfer and super-resolution. In ECCV, pages 694–711. Springer, 2016. 2

[16] Eric Kee, Adam Pikielny, Blackburn-Matzen Kevin, and Marc Levoy. Removing reflections from raw photos. In ECCV, 2024. 3

[17] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuhler,¨ and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM TOG, 42(4), 2023. 1, 2, 3, 4, 6

[18] Soomin Kim, Yuchi Huo, and Sung-Eui Yoon. Single image reflection removal with physically-based training images. In CVPR, pages 5164–5173, 2020. 2

[19] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In ICCV, pages 4015–4026, 2023. 3

[20] Chenyang Lei, Xuhua Huang, Mengdi Zhang, Qiong Yan, Wenxiu Sun, and Qifeng Chen. Polarized reflection removal with perfect alignment in the wild. In CVPR, pages 1750– 1758, 2020. 2, 3

[21] Chenyang Lei, Xudong Jiang, and Qifeng Chen. Robust reflection removal with flash-only cues in the wild. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2023. 2, 3, 4

[22] Anat Levin and Yair Weiss. User assisted separation of reflections from a single image using a sparsity prior. IEEE Transactions on Pattern Analysis and Machine Intelligence, 29(9):1647–1654, 2007. 2

[23] Chao Li, Yixiao Yang, Kun He, Stephen Lin, and John E Hopcroft. Single image reflection removal through cascaded refinement. In CVPR, pages 3565–3574, 2020. 2

[24] Yu Li and Michael S Brown. Exploiting reflection change for automatic reflection removal. In ICCV, pages 2432–2439, 2013. 2, 3, 4

[25] Yu Li and Michael S Brown. Single image layer separation using relative smoothness. In CVPR, pages 2752–2759, 2014. 2

[26] Yu-Lun Liu, Wei-Sheng Lai, Ming-Hsuan Yang, Yung-Yu Chuang, and Jia-Bin Huang. Learning to see through obstructions. In CVPR, 2020. 2, 3

[27] Daiqian Ma, Renjie Wan, Boxin Shi, Alex C Kot, and Ling Yu Duan. Learning to jointly generate and separate reflections. In ICCV, pages 2444–2452, 2019. 2

[28] Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view syn thesis. In ECCV, 2020. 2

[29] Simon Niklaus, Xuaner Cecilia Zhang, Jonathan T Barron, Neal Wadhwa, Rahul Garg, Feng Liu, and Tianfan Xue. Learned dual-view reflection removal. In WACV, pages 3713–3722, 2021. 2, 6

[30] BH Prasad, Lokesh R Boregowda, Kaushik Mitra, Sanjoy Chowdhury, et al. V-desirr: Very fast deep embedded single image reflection removal. In ICCV, pages 2390–2399, 2021. 2

[31] J Redmon. You only look once: Unified, real-time object detection. In CVPR, 2016. 2

[32] YiChang Shih, Dilip Krishnan, Fredo Durand, and William T Freeman. Reflection removal using ghosting cues. In CVPR, pages 3193–3201, 2015. 2

[33] Sudipta N Sinha, Johannes Kopf, Michael Goesele, Daniel Scharstein, and Richard Szeliski. Image-based rendering for scenes with reflections. ACM TOG, 31(4):1–10, 2012. 3

[34] Jiayi Song, Zihan Ye, Qingyuan Zhou, Weidong Yang, Ben Fei, Jingyi Xu, Ying He, and Wanli Ouyang. Reflections unlock: Geometry-aware reflection disentanglement in 3d gaussian splatting for photorealistic scenes rendering. arXiv preprint arXiv:2507.06103, 2025. 3

[35] Renjie Wan, Boxin Shi, Ling-Yu Duan, Ah-Hwee Tan, and Alex C Kot. Benchmarking single-image reflection removal algorithms. In CVPR, pages 3922–3930, 2017. 2

[36] Renjie Wan, Boxin Shi, Ling-Yu Duan, Ah-Hwee Tan, and Alex C Kot. Crrn: Multi-scale guided concurrent reflection removal network. In CVPR, pages 4777–4785, 2018. 2

[37] Renjie Wan, Boxin Shi, Haoliang Li, Ling-Yu Duan, Ah-Hwee Tan, and Alex C Kot. Corrn: Cooperative reflection removal network. IEEE Transactions on Pattern Analysis and Machine Intelligence, 42(12):2969–2982, 2019.

[38] Renjie Wan, Boxin Shi, Haoliang Li, Ling-Yu Duan, and Alex C Kot. Reflection scene separation from a single im age. In CVPR, pages 2398–2406, 2020. 2

[39] Ce Wang, Dejia Xu, Renjie Wan, Bin He, Boxin Shi, and Ling-Yu Duan. Background scene recovery from an image looking through colored glass. IEEE Transactions on Multimedia, 25:2876–2887, 2022. 2

[40] Jianyuan Wang, Nikita Karaev, Christian Rupprecht, and David Novotny. Vggsfm: Visual geometry grounded deep structure from motion. In CVPR, pages 21686–21697, 2024. 6

[41] Linhan Wang, Kai Cheng, Shuo Lei, Shengkun Wang, Wei Yin, Chenyang Lei, Xiaoxiao Long, and Chang-Tien Lu. Dcgaussian: Improving 3d gaussian splatting for reflective dash cam videos. Advances in Neural Information Processing Systems, 37:99898–99920, 2024. 2, 3

[42] Mengyi Wang, Xinxin Zhang, Yongshun Gong, and Yilong Yin. Personalized single image reflection removal network through adaptive cascade refinement. In ACM MM, pages 8204–8213, 2023. 2

[43] Kaixuan Wei, Jiaolong Yang, Ying Fu, David Wipf, and Hua Huang. Single image reflection removal exploiting misaligned training data and network enhancements. In CVPR, pages 8178–8187, 2019. 3

[44] Qiang Wen, Yinjie Tan, Jing Qin, Wenxi Liu, Guoqiang Han, and Shengfeng He. Single image reflection removal beyond linearity. In CVPR, pages 3771–3779, 2019. 2

[45] Zhihao Xia, Michael Gharbi, Federico Perazzi, Kalyan¨ Sunkavalli, and Ayan Chakrabarti. Deep denoising of flash and no-flash pairs for photography in low-light environments. In CVPR, pages 2063–2072, 2021. 2, 3, 4

[46] Zhihao Xia, Jason Lawrence, and Supreeth Achar. A dark flash normal camera. In ICCV, pages 2430–2439, 2021. 2, 3

[47] Mingyang Xie, Haoming Cai, Sachin Shah, Yiran Xu, Brandon Y. Feng, Jia-bin Huang, and Christopher Metzler. Flashsplat: 3d reflection removal with flash cues and gaussian splats. In ECCV, 2024. 2, 3, 4, 6

[48] Tianfan Xue, Michael Rubinstein, Ce Liu, and William T. Freeman. A computational approach for obstruction-free photography. ACM TOG, 34(4), 2015. 2, 3

[49] Jie Yang, Dong Gong, Lingqiao Liu, and Qinfeng Shi. Seeing deeply and bidirectionally: A deep learning approach for single image reflection removal. In ECCV, pages 654–669, 2018. 2

[50] Yang Yang, Wenye Ma, Yin Zheng, Jian-Feng Cai, and Weiyu Xu. Fast single image reflection suppression via convex optimization. In CVPR, pages 8141–8149, 2019. 2

[51] Xuaner Zhang, Ren Ng, and Qifeng Chen. Single image reflection separation with perceptual losses. In CVPR, 2018. 2, 5

[52] Qian Zheng, Boxin Shi, Jinnan Chen, Xudong Jiang, Ling-Yu Duan, and Alex C Kot. Single image reflection removal with absorption effect. In CVPR, pages 13395–13404, 2021. 2

[53] Haofeng Zhong, Yuchen Hong, Shuchen Weng, Jinxiu Liang, and Boxin Shi. Language-guided image reflection separation. In CVPR, pages 24913–24922, 2024. 2, 3

[54] Chengxuan Zhu, Renjie Wan, and Boxin Shi. Neural transmitted radiance fields. Advances in Neural Information Pro cessing Systems, 35:38994–39006, 2022. 2, 3, 5, 6

[55] Yurui Zhu, Xueyang Fu, Peng-Tao Jiang, Hao Zhang, Qibin Sun, Jinwei Chen, Zheng-Jun Zha, and Bo Li. Revisiting single image reflection removal in the wild. In CVPR, pages 25468–25478, 2024. 2, 3