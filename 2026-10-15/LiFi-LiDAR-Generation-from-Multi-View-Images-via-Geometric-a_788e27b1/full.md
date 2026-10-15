# LiFi: LiDAR Generation from Multi-View Images via Geometric and Semantic Collaborative Guidance

Sizhuo Zhou<sup>1,2</sup>, Xiaosong Jia<sup>4†</sup>, Fanrui Zhang<sup>1,2</sup>, Qifeng Li<sup>3</sup>, Junjie Li<sup>1,2</sup>, Zirui Wang<sup>1,2</sup>, Yukang Feng<sup>2</sup>, Yu Hong<sup>2,3</sup>, Shaofeng Zhang<sup>3</sup>, Wenlong Liao<sup>3</sup>, Tao He, Juyong Zhang<sup>1</sup>, Junchi Yan<sup>2,3†</sup>

1. University of Science and Technology of China 2. Shanghai Innovation Institute 3. Shanghai Jiao Tong University 4. Fudan University

## Abstract

Generating LiDAR point clouds from multi-view camera images is a valuable task with applications in controllable data synthesis, cross-modal simulation and various perception tasks. However, most existing methods are not specifically designed for this task, and the quality of their generated LiDAR remains limited. Moreover, these methods typically focus only on semantic cues in images or generate LiDAR from a monocular camera image. In this work, we introduce LiFi (LiDAR From images), a dedicated framework for generating visually and geometrically aligned LiDAR scenes from multi-view camera images. LiFi guides the latent diffusion process through two complementary branches: geometry and semantics. The geometry branch constructs scene features from images through a reverse sampling algorithm and an uncertainty-aware geometric encoder combined with DepthAnything3, whereas the semantic branch provides high-level semantic representations via a cross-view interaction module. We further introduce a dual-branch balanced classifier-free guidance (CFG) strategy, which enhances the model’s conditional generation capability while preserving the independent learning and collaborative controllability of the two branches. Extensive experiments demonstrate that LiFi outperforms state-of-the-art methods in generating high-fidelity LiDAR scenes. The code will be publicly available at: https://github.com/szzhou88/LiFi.

## 1 Introduction

The rapid progress of autonomous driving (AD) has drawn widespread attention [18–20]. LiDAR sensors, which provide precise 3D structural information, have become a cornerstone of modern AD systems [30, 32]. Nevertheless, acquiring large-scale, high-quality LiDAR data could be expensive and the long-tail data captured by consumer devices might not employ LiDAR due to cost control of the product, limiting the robustness and scalability of data-driven methods. To mitigate this, pioneering LiDAR generation approaches [14, 52, 57] have been introduced, but they mainly focus on unconditional generation and suffer from limited realism because they fail to exploit the diverse contextual cues present in real-world scenes. Recently, conditional LiDAR point cloud generation [24, 28, 49, 51] has emerged, leveraging inputs such as text, maps, and semantic layouts to achieve notable gains in both the photorealism and controllability of synthesized LiDAR scenes.

In this work, we study the conditional generation of LiDAR scenes from multi-view camera images. Cameras, owing to their low cost and ease of deployment, have become a standard sensor in modern autonomous systems. Generating LiDAR point clouds from such onboard multi-view images therefore holds broad application potential, where it can complement high-quality 3D perception under cost-sensitive sensor setups, and open up new possibilities for cross-modal simulation [11], data augmentation [5], and sensor completion [6, 56]. However, this formulation remains relatively under-explored. A major challenge is the cross-modal gap between images and LiDAR. RGB images provide dense 2D appearance information, while LiDAR captures sparse yet precise 3D geometric measurements, making it difficult to establish accurate correspondences between the two heterogeneous representations. Furthermore, although multi-view cameras provide richer observations, different viewpoints also introduce variations in field of view, occlusion patterns, and semantic redundancy. As a result, existing methods [40] have only preliminarily explored images as conditioning inputs, focusing on semantic cues without explicitly enforcing geometric alignment between modalities, and are often limited to monocular inputs that lack comprehensive spatial awareness.

![](images/362c41e4f26d602a18016f4633c781e6691a07249feb058b949cb00334d4b336.jpg)  
Figure 1: Overview of the LiFi framework. The framework takes real multi-view camera images as input and is capable of generating 3D LiDAR scenes with high fidelity that are consistent with the visual information. LiFi mainly comprises the following key designs: (i) geometry–semantic decoupled conditioning branches; (ii) a dual-branch balanced classifier-free guidance module; and (iii) a latent diffusion network integrated with Transformer blocks.

To address these challenges, we propose LiFi, a dedicated framework for generating high-fidelity LiDAR scenes from multi-view camera images. LiFi guides the latent diffusion process of LiDAR scenes, represented as range images, through two decoupled yet complementary branches: geometry and semantics. In the geometry branch, we first leverage DepthAnything3 [31] to estimate perpixel depth and uncertainty maps from the input images. A range-view-centered reverse sampling algorithm then is designed to align these dense predictions, together with the original RGB images, to the LiDAR scanning pattern, producing depth, confidence, and appearance conditions. These conditions are subsequently encoded into structured geometric features via an uncertainty-modulated encoder that dynamically adjusts feature expressiveness based on local depth confidence. In the semantic branch, we extract high-level visual tokens and fuse them with patch, view, and camera extrinsic encodings. A cross-view attention module then aggregates global semantic context to capture long-range dependencies and cross-view consistency. Furthermore, we introduce a dualbranch balanced classifier-free guidance (CFG) strategy. By jointly training with full conditional, single-branch-conditional, and unconditional settings, this strategy not only enhances the model’s conditional guidance capability, but also improves the stability and effectiveness of collaborative learning between the two branches.

We primarily compare LiFi with the state-of-the-art LiDAR generation models on the nuScenes dataset. We also compare it with existing frameworks capable of performing camera-to-LiDAR generation. To further evaluate the applicability of LiFi, we also compare it with several representative baseline methods on the KITTI-360 dataset [7]. Experimental results demonstrate the superior performance of our method in generating high-fidelity LiDAR scenes with geometric consistency and semantic alignment. Furthermore, we apply LiFi to a downstream object detection task. Although this does not directly demonstrate that LiFi-generated scenes can independently support perception performance comparable to that of real LiDAR data, it nevertheless provides evidence of the effectiveness of the proposed generation framework.

In summary, our contributions are as follows:

• We propose LiFi, which, to the best of our knowledge, is the first dedicated framework capable of generating high-fidelity LiDAR scenes with geometric–visual alignment from multi-view images;

• We carefully design a range-view-centered reverse sampling algorithm, an uncertainty-aware geometric encoder, and a cross-view semantic encoder to achieve precise cross-modal alignment at the geometric level and effective global semantic guidance;

• We introduce a dual-branch balanced classifier-free guidance strategy, which effectively enhances the model’s conditional guidance capability and collaborative learning ability;

• We conduct extensive evaluations of LiFi on the nuScenes dataset for both high-fidelity LiDAR generation and downstream detection tasks, demonstrating the effectiveness of our method.

## 2 Related Work

Generative Models. Generative models synthesize realistic and diverse new samples by learning the underlying data distribution. Early generative frameworks include GANs [8] and VAEs [23]. GANs can generate realistic images but suffer from issues such as training instability and mode collapse. VAEs provide an explicit probabilistic modeling formulation, but their generated results tend to be overly smooth. Autoregressive models [48] achieve accurate generation by progressively modeling conditional distributions, yet their sequential sampling process incurs high computational overhead. Diffusion models (DMs) [12, 22, 36, 42, 43, 45–47] have demonstrated outstanding performance in image synthesis. Subsequent research has introduced latent diffusion models (LDMs) [41], which perform the diffusion process in the latent space, thereby significantly improving generation efficiency. Meanwhile, they have also facilitated the advancement of conditional generation [35, 39, 53].

LiDAR Scene Generation. LiDAR scene generation aims to synthesize 3D point cloud representations with realistic geometric structures and scanning patterns. Caccia et al. [1] were the first to adopt the range image representation and perform generation using VAEs and GANs. Subsequently, LiDARGen [57] proposed a score-based model for LiDAR generation. UltraLiDAR [52] voxelizes LiDAR point clouds and combines this representation with VQ-VAE for generation. R2DM [33], built upon DDPMs, improves the fidelity of generated point clouds. RangeLDM [14] enables fast LiDAR data generation by performing diffusion in a compact latent space. DriveLiDAR4D [3] improves LiDAR generation quality by conditioning on three modalities: road sketches, scene descriptions, and object priors. UniScene [24] introduces occupancy as an intermediate representation, thereby reducing the difficulty of generating complex LiDAR scenes. OpenDWM [34] is built upon the DiT architecture for video generation and further extends it to LiDAR scenes. LiDARCrafter [28] parses natural language inputs into scene graphs to guide generation. LaGen [55] proposes an autoregressive framework for generating long-horizon LiDAR scenes.

Camera-to-LiDAR Generation. Camera-to-LiDAR generation refers to generating 3D LiDAR scenes from camera images. Unlike other conditional LiDAR generation models, research on directly synthesizing LiDAR data from camera images remains relatively limited. LiDM [40] proposes a multimodal conditional framework for LiDAR scene generation and also provides a preliminary exploration of the camera-to-LiDAR task by extracting semantic information from camera images. However, it does not investigate how to align the image and LiDAR modalities, which differ substantially from each other, and thus its generation quality remains limited. X-Drive [51] proposes a framework for the joint generation of LiDAR point clouds and multi-view camera images, taking both LiDAR point clouds and camera images as inputs and generating the two modalities simultaneously. It can be adapted to the camera-to-LiDAR task. However, its core framework is not specifically designed for the image-only input setting, and therefore does not exhibit clear performance advantages in this scenario. Moreover, it incorporates additional text prompts and 3D bounding box information to assist the generation process. Other frameworks for the joint generation of LiDAR point clouds and camera images include HoloDrive [50] and Genesis [10], but they do not provide a camera-to-LiDAR mode. Our proposed method addresses the above challenges.

![](images/6b4e73137ac2281a91da751d0fefe2817535e0e048444c9d747c3a9f0f1b95d7.jpg)  
Figure 2: Geometry-Aware Branch extracts structured geometric features from raw images via a range-view-centered reverse sampling algorithm and an uncertainty-modulated geometric encoder, thereby guiding the latent diffusion process.

## 3 Method

## 3.1 Preliminaries

Range Image Representation. We adopt the range image as the basic representation of LiDAR data. Unlike directly processing sparse and unordered 3D point sets, it exploits the sensor’s scanning pattern to encode point clouds into a regular 2D array. For a point $p \overset { \cdot } { = } ( x , y , z ) \in \mathbb { R } ^ { 3 }$ , we first transform it from the Cartesian coordinate system to a corrected spherical coordinate system [14]:

$$
r = \sqrt { x ^ { 2 } + y ^ { 2 } + ( z - h _ { j } ) ^ { 2 } } , \quad \theta = \arctan ( y , x ) , \quad \phi = \phi _ { j } ,\tag{1}
$$

where r denotes the depth value, θ is the azimuth angle, and $\phi$ is the elevation angle; $h _ { j }$ and $\phi _ { j }$ denote the sensor height and elevation angle estimated by Hough voting. We then map this spherical representation to the range-view space and apply normalization, thereby obtaining the corresponding range image. This representation transformation is reversible.

Adopting the range image representation not only transforms the data into a 2D representation but also ensures that each pixel corresponds to a specific azimuth and elevation direction of the target LiDAR sensor, thereby preserving the sensor’s structured angular sampling topology and inherent sparsity. Furthermore, given a calibrated sensor model, the generated 2D representation can be deterministically converted back into a 3D point cloud, improving generation efficiency.

Latent Diffusion Model for LiDAR Generation. We adopt a latent diffusion model (LDM) [41] for conditional LiDAR data generation. The framework consists of a VAE and a UNet: the former compresses the range image into a latent space, while the latter performs diffusion modeling in the latent space. Specifically, given an input range image $I \in \mathbb { R } ^ { H _ { r } \times W _ { r } \times 2 }$ , the encoder first maps it to a latent representation $\dot { \hat { z } } \in \mathbb { R } ^ { h _ { r } \times w _ { r } \times \dot { c } _ { r } }$ , and the decoder then reconstructs the range image from this latent variable as $\hat { I } = G ( \hat { z } )$ . During the diffusion process, the model starts from a clean latent variable $z _ { \mathrm { 0 } }$ and progressively injects Gaussian noise into it to obtain a noisy latent variable $z _ { t }$ . During training, the UNet takes $z _ { t } ,$ , the timestep $t ,$ and the encoded conditional information c as inputs, and predicts the noise at the current step. The training objective is defined as:

$$
\mathcal { L } _ { L D M } = \mathbb { E } _ { \epsilon _ { t } \in \mathcal { N } ( 0 , 1 ) , t \in \mathcal { U } [ 0 , T ] , c } \left[ \Vert \epsilon _ { t } - \epsilon _ { \theta } \left( z _ { t } ; t , c \right) \Vert ^ { 2 } \right] ,\tag{2}
$$

where $\epsilon _ { \theta }$ denotes the learnable noise prediction network. During inference, the model starts from random Gaussian noise and iteratively denoises it to generate the latent range-view representation.

## 3.2 Overall Framework

Given the multi-view camera images of a driving scene, $V \in \mathbb { R } ^ { N _ { v } \times H _ { v } \times W _ { v } \times 3 }$ , along with the corresponding camera extrinsics $\boldsymbol { M } = \boldsymbol { \lvert { R } \rvert } t \boldsymbol { \lvert { \epsilon } } \rvert \in \mathbb { R } ^ { 3 \times 4 }$ and camera intrinsics $K \in \mathbb { R } ^ { 3 \times 3 }$ , our goal is to generate a 3D LiDAR scene that is geometrically and perceptually aligned with the input. To achieve this goal, we first extract the following three types of visual priors from multi-view camera inputs using a pre-trained vision foundation model: (i) pixel-wise relative depth estimation in the camera image space: $D \in \mathbb { R } ^ { N _ { v } \times H _ { v } \times W _ { v } }$ ; (ii) the corresponding uncertainty maps of depth estimation $U \in \mathbb { R } ^ { N _ { v } \times \breve { H } _ { v } \times \mathbf { \dot { W } } _ { v } } ;$ and (iii) camera semantic tokens $\dot { S ^ { \mathrm { ~ } } } \in \mathbb { R } ^ { N _ { v } \times l _ { s } \times c _ { s } }$ . Taking DepthAnything3 [31] as an example, it can jointly process multiple images captured from different viewpoints of the same scene and natively output corresponding confidence maps.

Based on the above conditional information, we construct an image-to-LiDAR point cloud generation framework guided by dual conditional branches with decoupled geometry and semantics. In the first branch, we construct a geometry-conditioned encoder integrated with an uncertainty modulation mechanism to fuse raw RGB images, depth maps, and uncertainty features. In the second branch, we build a pose-aware semantic encoder to provide global semantic context and capture cross-view relationships for the model. The overall pipeline of LiFi is illustrated in Figure 1.

## 3.3 Reverse Sampling Based Geometry-Aware Conditioning

In this section, we describe the construction of the geometry-aware conditional branch in detail. Thi branch is designed to extract geometric features of the scene from multi-view camera observations, thereby providing explicit spatial structural constraints for the subsequent diffusion process. To effectively encode the geometric details contained in multi-view camera images, we first design a range-view-centered reverse depth sampling algorithm.

## 3.3.1 Range-View-Centered Reverse Sampling

To fully exploit the estimated depth maps D obtained from camera images, a straightforward approach is to project these depth maps pixel by pixel into LiDAR point clouds and then convert them into range images. However, this strategy fails to ensure accurate alignment between projected depth maps and LiDAR scenes, while also introducing substantial computational overhead.

This issue arises because a single depth map contains millions of pixels (a common resolution is $1 6 0 0 \times 9 0 0 )$ ), whereas a typical LiDAR scene is inherently sparse. Direct projection is therefore not only inefficient, but also inconsistent with the distribution pattern of LiDAR data. To address this, we introduce a range-view-centered reverse depth sampling algorithm.

Specifically, to construct the relative depth map I<sub>D</sub> in the range-image representation, we first apply the inverse spherical projection transformation to a standard single-channel range image $I \in \mathbb { R } ^ { H _ { r } \times W _ { \tau } }$ <sup>r</sup> , yielding a set of canonical 3D query points $\bar { Q _ { D } } = \{ q _ { k } \}$ . We then project these query points, which follow the distribution char-

![](images/6fb5852c9454ff2177b830964f9a8fbb4aa09a68ce2bb0c5898c709d9c795a19.jpg)  
Figure 3: Schematic illustration of the reverse sampling mechanism.

acteristics of LiDAR scenes, into the camera coordinate systems of different views through a spatial coordinate transformation:

$$
[ \tilde { u } _ { n } , \tilde { v } _ { n } , \tilde { w } _ { n } ] ^ { T } = K _ { n } \cdot ( R _ { n } \cdot q _ { k } + t _ { n } ) .\tag{3}
$$

Here, $K _ { n }$ denotes the camera intrinsic matrix of the n-th image, while $R _ { n }$ and $t _ { n }$ are the rotation and translation parameters of the corresponding camera extrinsics, respectively. We then project them into the pixel space of the camera images via perspective projection to sample the corresponding depth and uncertainty values:

$$
u _ { n } = \frac { \tilde { u } _ { n } } { \tilde { w } _ { n } } , \quad v _ { n } = \frac { \tilde { v } _ { n } } { \tilde { w } _ { n } } .\tag{4}
$$

During this process, each canonical query point, together with the LiDAR center, defines a LiDAR ray with a specific direction in 3D space. For each camera image, this ray intersects at most one pixel.

![](images/d536f443a891122debc10ea346c98f156699baf627abd369f20df02229899934.jpg)  
Figure 4: Cross-View Semantic Branch first aggregates multi-view camera tokens augmented with position encodings, then extracts cross-view features through a semantic encoder with a self-attention mechanism, and finally injects them into the latent diffusion process via Transformer blocks.

We project only valid query points that fall within the cameras’ fields of view and image boundaries. Query points that cannot be projected onto any camera image are assigned null values. In cases of overlap or occlusion, we retain the depth and uncertainty values according to a nearest-depth criterion. Similarly, this algorithm can also be used to sample color information in the original RGB space. After min-max normalization, we obtain a LiDAR-aligned relative depth map $I _ { D }$ , an uncertainty map $I _ { U }$ , and an RGB image $I _ { C }$

## 3.3.2 Geometry-Aware Condition Encoding with Uncertainty Modulation

To effectively exploit these LiDAR-aligned visual priors, we design a geometry-aware conditional encoder. Specifically, we first introduce a 2D coordinate encoding for the range-view space, and concatenate it with the depth map and RGB image as the input to the encoder. Next, we use an encoder $\mathcal { E } _ { G }$ to jointly model the input features. This encoder consists of an initial convolutional layer, several residual blocks, and downsampling layers, which progressively extract local geometric and appearance information and map them to appropriate feature scales. Notably, rather than simply treating the uncertainty map $I _ { U }$ as an additional input channel, we inject it into the encoding process as an explicit modulation signal through FiLM:

$$
\tilde { H } _ { l } = \gamma _ { l } ( I _ { U } ) \odot H _ { l } + \beta _ { l } ( I _ { U } ) ,\tag{5}
$$

where $H _ { l }$ denotes the output feature of the l-th residual block, and $\gamma _ { l }$ and $\beta _ { l }$ are the scaling and shifting terms generated by the FiLM module, respectively. This mechanism enables the model to dynamically adjust the expressive strength of the conditional features according to the confidence of local depth predictions. As a result, the encoder can preserve explicit geometric structure while suppressing the propagation of unreliable conditional information in low-confidence regions.

This branch provides local, structured, and reliable geometric constraints for LiDAR scene generation. Compared with directly using raw projection results, this design more effectively integrates depth, appearance, and spatial positional information, while enhancing the consistency between the crossmodal conditions and the target LiDAR scene.

## 3.4 Cross-View Semantic Conditioning

While the geometry-aware branch provides reliable geometric constraints, it is limited in capturing long-range dependencies and cross-view consistency. To address this, we introduce a cross-view semantic branch to provide scene-level semantics and contextual information. Specifically, we design a cross-view semantic encoder $\mathcal { E } _ { S } ,$ , which takes the flattened tokens from all views as input. We further introduce three types of positional encodings to enhance its awareness of patch coordinates, camera identity, and camera extrinsic parameters, thereby obtaining enriched feature representations:

$$
\tilde { X } _ { S } = X _ { S } + E _ { p a t c h } + E _ { v i e w } + E _ { e x t } .\tag{6}
$$

Here, $E _ { p a t c h }$ denotes a set of learnable parameters of size $N L \times D$ , where $N$ is the number of cameras, L is the number of image patches per camera, and D is the feature dimension. $E _ { v i e w }$ is a learnable embedding vector associated with each camera index. $E _ { e x t }$ is obtained by flattening the extrinsic matrix of each camera into a 16-dimensional vector and passing it through a two-layer MLP. We then feed it into a cross-view interaction module based on multi-head attention. Through self-attention, this module establishes global dependencies among tokens from all views, allowing semantic information from different camera perspectives to interact within a unified feature space:

![](images/a1ba4868ba5254049fe8b6aae01c1213e41c024af0557cea4892bfe3a2380a83.jpg)  
Figure 5: Visualization results of LiFi and several baseline methods for LiDAR scene generation. It can be observed that our method generates details that are closest to the real LiDAR point cloud data.

$$
H _ { S } = \mathrm { L a y e r N o r m } ( \tilde { X } _ { S } + \mathrm { S e l f A t t n } ( \tilde { X } _ { S } ) ) .\tag{7}
$$

Subsequently, $H _ { S }$ is passed through a feed-forward network and then injected into the Transformer blocks of the UNet, where it interacts with the backbone features via cross-attention. This branch further captures long-range scene relationships and cross-view consistency, thereby improving the plausibility of the generated results in terms of both global structure and semantic coherence.

## 3.5 Dual-Branch Balanced Classifier-Free Guidance

In the previous two sections, we introduced the geometric conditional branch and the cross-view semantic branch, respectively. However, in practice, the model may become biased toward one conditional branch during training, resulting in under-utilization of the other branch.

To mitigate this issue, we construct a dual-branch balanced classifier-free guidance (CFG) module. Standard CFG typically enhances conditional guidance at inference time by randomly dropping all conditions, thereby allowing the model to learn both conditional and unconditional distributions. In contrast, we extend the condition dropout mechanism beyond the conventional “all-or-nothing” scheme by additionally introducing cases where only geometric or only semantic conditions are retained. This enables the model not only to possess the unconditional modeling capability, but also to explicitly learn the individual effects of the two types of conditions as well as their joint contributions. During inference, we adopt the same conditional guidance formulation as in standard CFG:

$$
\epsilon = \epsilon _ { \mathrm { u n c o n d } } + w \cdot ( \epsilon _ { \mathrm { c o n d } } - \epsilon _ { \mathrm { u n c o n d } } ) ,\tag{8}
$$

where w is the guidance scale, $\epsilon _ { \mathrm { u n c o n d } }$ and $\epsilon _ { \mathrm { c o n d } }$ denote the noise predictions under unconditional and conditional guidance, respectively.

This strategy preserves the advantages of standard CFG while further balancing the learning process between the geometric and semantic branches, enabling the model to leverage global semantic information to generate more plausible details when the depth information from the geometric branch is unreliable. With all the above designs, our model is able to jointly leverage local geometric conditions and global semantic conditions, thereby establishing reliable cross-modal correspondences between multi-view images and LiDAR scenes and generating high-fidelity LiDAR scenes.

![](images/16a22180782a7cd6d1ed5537586f2f5570e9dad25ddf50950a0b33426a83b8e3.jpg)  
Figure 6: Qualitative results of LiFi and several major baseline models on the KITTI-360 dataset.

## 4 Experiments

## 4.1 Experimental Setups

Dataset. We conduct experiments on the challenging nuScenes dataset [2]. It contains over a thousand autonomous driving sequences collected in two geographically distinct urban environments with complex traffic conditions. The nuScenes training set includes 297,737 LiDAR scans, while the test set contains 52,423 LiDAR scans, each captured by a 32-beam LiDAR sensor. This dataset is widely used in various autonomous driving tasks, such as 3D object detection [17, 25–27], multi-object tracking [13, 38, 54], trajectory prediction [9, 29], and end-to-end autonomous driving [15, 16, 21]. We also conduct experiments on the KITTI-360 dataset [7], as detailed in the supplementary material.

Baselines. We compare LiFi with multiple categories of baseline methods. The first category includes conventional LiDAR data generation methods, covering both unconditional and conditional generation models. The second category consists of baselines capable of generating LiDAR from camera images, including LiDM [40] and X-Drive [51]. The third category includes models for generating continuous LiDAR scenes. In addition, we apply LiFi to a downstream 3D object detection task and compare the performance of models with and without LiDAR input.

Implementation Details. We conduct distributed training on four NVIDIA H200 GPUs. Our model is trained for 200 epochs, with a total of 176,000 optimization steps, and the learning rate is set to $1 \times 1 0 ^ { - 4 }$ . We use the DDIM sampler [44] for denoising, and set the total number of diffusion timesteps T to 1024. The UNet consists of three downsampling blocks and three upsampling blocks, each containing two Transformer blocks. We employ DepthAnything3 [31] as the pretrained visual encoder, which can predict spatially consistent geometry from an arbitrary number of input views. It is an instantiation choice that matches the requirements, rather than an irreplaceable component of the framework. During inference, we perform 50 denoising steps to generate the target range image. More experimental settings and implementation details are provided in the supplementary material.

## 4.2 LiDAR Data Generation

This subsection evaluates the performance of LiFi on the task of generating high-fidelity LiDAR point clouds. We compare our method against both unconditional and conditional generation baselines. Following the evaluation protocol of LiDARGen [57], we adopt MMD, JSD, and two perceptionbased Fréchet distances as evaluation metrics. As shown in Table 1, our method achieved clear improvements across multiple metrics.

Table 1: The main quantitative results of LiFi and several baseline methods on the nuScenes dataset. The MMD values are reported in $1 0 ^ { - 5 }$ and JSD in $1 0 ^ { - 2 }$
<table><tr><td>Method</td><td></td><td>Venue</td><td>MMD↓</td><td>JSD↓</td><td>FRD↓</td><td>FPD↓</td></tr><tr><td rowspan="3">Unncon. LiDM [40]</td><td rowspan="3">LiDARGen [57] RangeLDM [14]</td><td>ECCV’22</td><td>55.14</td><td>27.69</td><td></td><td></td></tr><tr><td>ECCV&#x27;24</td><td>72.01</td><td>8.43</td><td>1172.06</td><td>441.16</td></tr><tr><td>CVPR&#x27;’24</td><td>4.57</td><td>10.61</td><td>2358.15</td><td>420.36</td></tr><tr><td rowspan="4">Cond.</td><td rowspan="4">UniScene [24] OpenDWM [34] LiDARCrafter [28] LiFi</td><td>CVPR&#x27;25</td><td>107.18</td><td>12.35</td><td>1636.60</td><td>230.89</td></tr><tr><td>CVPR&#x27;25</td><td>10.75</td><td>7.69</td><td>2607.81</td><td>146.30</td></tr><tr><td>AAAI&#x27;26</td><td>24.69</td><td>7.37</td><td>1183.11</td><td>36.17</td></tr><tr><td></td><td>2.78</td><td>3.66</td><td>906.71</td><td>126.31</td></tr></table>

![](images/8b21e062433f08499b0c830b19ec8c4c47a4e968810f3eb50e4a566da88a24ea.jpg)  
Figure 7: Qualitative results of LiDAR scene generation from camera images. The first row shows the full scenes, and the second row presents zoomed-in local details.

Table 2: The main quantitative results of LiFi and several baseline methods on the KITTI-360 dataset. The MMD values are reported in $1 0 ^ { - 4 }$ and JSD in $1 0 ^ { - 2 }$
<table><tr><td>Method</td><td>Venue</td><td>MMD JSD</td><td>FRD</td><td>FPD↓</td></tr><tr><td>LiDARGen [57] RangeLDM [14]</td><td>ECCV’22 ECCV’24</td><td>13.36 13.67 4.31 9.42</td><td>2415.23 1362.25</td><td>102.80 68.22</td></tr><tr><td>LiDM [40]</td><td>CVPR&#x27;24</td><td>9.31 19.00</td><td>2894.65</td><td>253.91</td></tr><tr><td>OpenDWM [34]</td><td>CVPR&#x27;25</td><td>12.93 14.63</td><td>1966.36</td><td>436.17</td></tr><tr><td>LiFi</td><td></td><td>1.99 8.77</td><td>1107.21</td><td>21.40</td></tr></table>

## 4.3 Camera-to-LiDAR Generation

This subsection compares LiFi with several baselines capable of generating LiDAR point clouds from camera images. We additionally introduce the CD and EMD metrics, which enable a more rigorous evaluation of point-level fidelity in the generated scenes, to assess the model’s performance in generating geometrically consistent LiDAR scenes. We also introduce the DAS metric [51] to evaluate the cross-modal consistency. As shown in Table 3, LiFi achieves substantially better point-level fidelity and cross-modal consistency than existing camera-to-LiDAR baselines.

We further evaluate the quality of LiDAR data generated by LiFi across different depth ranges to investigate how its generation performance varies with scene distance and to analyze the associated failure modes. Detailed results are provided in the supplementary material.

Table 3: Quantitative results on the nuScenes dataset. These baseline models are used for generating LiDAR scenes from camera images. The MMD values are reported in $1 0 ^ { - 5 }$ and JSD in $1 0 ^ { - \overleftarrow { 2 } }$ Inference time is evaluated in s/frame. Evaluation is conducted on a single NVIDIA H200 GPU.
<table><tr><td>Method</td><td>Venue</td><td>MMD</td><td>JSD</td><td>CD</td><td>EMD</td><td>DAS</td><td>Time</td></tr><tr><td>LiDM [40]</td><td>CVPR&#x27;24</td><td>8.02</td><td>9.97</td><td>4.57</td><td>99.71</td><td>2.17</td><td>0.34</td></tr><tr><td>X-Drive [51]</td><td>ICLR’25</td><td>48.58</td><td>10.54</td><td>4.73</td><td>179.62</td><td>1.52</td><td>1.80</td></tr><tr><td>LiFi</td><td></td><td>2.78</td><td>3.66</td><td>0.83</td><td>50.08</td><td>1.02</td><td>0.52</td></tr></table>

## 4.4 Temporal Consistency of Continuous Scene Generation

We further evaluate the temporal consistency of LiDAR scenes generated by LiFi under consecutive camera inputs. While this approach does not explicitly incorporate mechanisms for temporal generation, inputting consecutive camera frames nevertheless produces LiDAR sequences that maintain good temporal consistency. Following LiDARCrafter [28], we evaluate these methods using CTC and TTCE, two metrics designed to measure temporal consistency. The results are shown in Table 4.

Table 4: Quantitative results of LiFi for continuous LiDAR scene generation. Here, TTCE and CTC are adopted to evaluate the temporal consistency of the generated results.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Venue</td><td colspan="2">TTCE</td><td rowspan="2">1</td><td colspan="3">CTC↓</td><td rowspan="2">10</td></tr><tr><td>3</td><td>4</td><td>2</td><td>3</td><td>4</td></tr><tr><td>UniScene [24]</td><td>CVPR&#x27;25</td><td>2.74</td><td>3.69</td><td>0.90</td><td>1.84</td><td>3.64</td><td>3.90</td><td></td></tr><tr><td>OpenDWM [34]</td><td>CVPR&#x27;25</td><td>2.68</td><td>3.65</td><td>1.02</td><td>2.02</td><td>3.37</td><td>5.05</td><td></td></tr><tr><td>OpenDWM-DiT [34]</td><td>CVPR&#x27;25</td><td>2.71</td><td>3.66</td><td>0.89</td><td>1.79</td><td>3.06</td><td>4.64</td><td></td></tr><tr><td>LiDARCrafter [28]</td><td>AAAI&#x27;26</td><td>2.65</td><td>3.56</td><td>1.12</td><td>2.38</td><td>3.02</td><td>4.81</td><td></td></tr><tr><td>LiFi</td><td></td><td>2.67</td><td>3.61</td><td>0.80</td><td>1.11</td><td>1.52</td><td>2.22 3.19</td><td></td></tr></table>

## 4.5 Downstream Task: 3D Object Detection

To assess the practical value of LiFi in realworld applications, we apply it to the downstream 3D object detection task in autonomous driving. We retrained the detector using the generated LiDAR data and compared the cameraonly (BEVDet [17]) and camera-LiDAR fusion baselines (BEVFusion [30]). The experimental results in Table 5 demonstrate that LiDAR point clouds generated by LiFi have certain value in downstream tasks of autonomous driving.

Table 5: Quantitative results of applying LiFi to the downstream 3D object detection task. We compare the performance of the camera-only model and the fusion-based model.
<table><tr><td>Method</td><td>mAP↑ NDS ↑</td></tr><tr><td>BEVDet (Camera Only) [17]</td><td>0.278 0.341</td></tr><tr><td>BEVFusion (+Real LiDAR) [30]</td><td>0.529 0.533</td></tr><tr><td>BEVFusion (+Gen LiDAR) [30]</td><td>0.526 0.520</td></tr></table>

## 4.6 Ablation Study

In this subsection, we conduct ablation studies to evaluate the contribution of each component in the LiFi framework. We first conduct ablation studies on the two branches. We further perform ablations on the depth map, the original RGB image, and the uncertainty modulation mechanism in the geometric branch. The results are shown in Table 6. Finally, we perform ablation studies on the CFG module, including the dual-branch balanced design and the guidance scale parameter. The corresponding results are presented in Table 7. Furthermore, we conduct ablation studies on the number of input image views, the resolution of input range images, and the number of denoising steps during inference. Detailed results are provided in the supplementary material.

Table 6: Ablation results on the two branches and the main conditions in the LiFi framework. The MMD is reported in $1 0 ^ { - 5 }$ and JSD in $1 0 ^ { - 2 }$
<table><tr><td>LiFi</td><td>MMD↓ JSD ↓</td></tr><tr><td>w/o Geometry-Aware Branch</td><td>5.20 4.27</td></tr><tr><td>w/o Cross-View Semantic Branch</td><td>3.57 3.67</td></tr><tr><td>w/o Depth Map Condition</td><td>3.57 4.12</td></tr><tr><td>w/o RGB Image Condition</td><td>2.80 3.92</td></tr><tr><td>w/o Uncertainty Modulation</td><td>2.87 3.80</td></tr><tr><td>w/ All</td><td>2.78</td></tr></table>

Table 7: Ablation results of the dual-branch balanced CFG module. w is the CFG scale. The MMD is reported in $1 0 ^ { - 5 }$ and JSD in $1 0 ^ { - 2 }$
<table><tr><td>LiFi</td><td>MMD↓ JSD ↓</td></tr><tr><td>w/o Classifier-Free Guidance</td><td>6.32 3.02</td></tr><tr><td>w/o Dual-Branch Balanced</td><td>4.31 3.76</td></tr><tr><td>Guidance scale w = 2.0</td><td>5.71</td></tr><tr><td>Guidance scale w = 4.0</td><td>3.45 3.81</td></tr><tr><td>Guidance scale w = 8.0</td><td>2.37 4.13</td></tr><tr><td>w/ All (w = 6.0)</td><td>2.78</td></tr></table>

## 5 Conclusion

We propose LiFi, a dedicated framework for generating realistic LiDAR scenes from multi-view camera images. We introduce a dual-branch conditioning strategy to guide the latent diffusion process, effectively addressing the challenges of cross-modal geometric alignment and cross-view semantic aggregation. Experimental results show that LiFi is capable of generating high-fidelity LiDAR scenes. Moreover, the LiDAR point cloud generated by it has certain value for some downstream tasks of autonomous driving. In the future, we expect LiFi to play an important role in applications such as simulation, planning, and control in various complex environments and long-tail scenarios.

## Acknowledgments and Disclosure of Funding

This research was supported by the Scientific Research Innovation Capability Support Project for Young Faculty (U40) of the Ministry of Education of China (SRICSPYF-ZY2025019), the National Natural Science Foundation of China (No.62272433), Anhui Provincial Natural Science Foundation (No.2508085ZD011), Key Science & Technology Project of Anhui Province (No. 202523o09050004) and the Fundamental Research Funds for the Central Universities.

## References

[1] Caccia, L., Van Hoof, H., Courville, A., Pineau, J.: Deep generative modeling of lidar data. In: 2019 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). pp. 5034–5040. IEEE (2019)

[2] Caesar, H., Bankiti, V., Lang, A.H., Vora, S., Liong, V.E., Xu, Q., Krishnan, A., Pan, Y., Baldan, G., Beijbom, O.: nuscenes: A multimodal dataset for autonomous driving. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 11621–11631 (2020)

[3] Cai, K., Liu, X., Zhou, X., Hu, H., Xiang, J., Zhang, L., Zhang, X., Zhan, K., Zhan, Y., Lang, X.: Drivelidar4d: Sequential and controllable lidar scene generation for autonomous driving. arXiv preprint arXiv:2511.13309 (2025)

[4] Fan, H., Su, H., Guibas, L.J.: A point set generation network for 3d object reconstruction from a single image. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 605–613 (2017)

[5] Fang, J., Zuo, X., Zhou, D., Jin, S., Wang, S., Zhang, L.: Lidar-aug: A general rendering-based augmentation framework for 3d object detection. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 4710–4720 (2021)

[6] Fu, C., Mertz, C., Dolan, J.M.: Lidar and monocular camera fusion: On-road depth completion for autonomous driving. In: 2019 IEEE Intelligent Transportation Systems Conference (ITSC). pp. 273–278. IEEE (2019)

[7] Geiger, A., Lenz, P., Stiller, C., Urtasun, R.: Vision meets robotics: The KITTI dataset. International Journal of Robotics Research 32(11), 1231 – 1237 (Sep 2013)

[8] Goodfellow, I.J., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., Bengio, Y.: Generative adversarial nets. Advances in neural information processing systems 27 (2014)

[9] Gu, J., Hu, C., Zhang, T., Chen, X., Wang, Y., Wang, Y., Zhao, H.: Vip3d: End-to-end visual trajectory prediction via 3d agent queries. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 5496–5506 (2023)

[10] Guo, X., Wu, Z., Xiong, K., Xu, Z., Zhou, L., Xu, G., Xu, S., Sun, H., Wang, B., Chen, G., et al.: Genesis: Multimodal driving scene generation with spatio-temporal and cross-modal consistency. arXiv preprint arXiv:2506.07497 (2025)

[11] Hess, G., Lindström, C., Fatemi, M., Petersson, C., Svensson, L.: Splatad: Real-time lidar and camera rendering with 3d gaussian splatting for autonomous driving. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 11982–11992 (2025)

[12] Ho, J., Jain, A., Abbeel, P.: Denoising diffusion probabilistic models. Advances in neural information processing systems 33, 6840–6851 (2020)

[13] Hu, H.N., Yang, Y.H., Fischer, T., Darrell, T., Yu, F., Sun, M.: Monocular quasi-dense 3d object tracking. IEEE Transactions on Pattern Analysis and Machine Intelligence 45, 1992–2008 (2022)

[14] Hu, Q., Zhang, Z., Hu, W.: Rangeldm: Fast realistic lidar point cloud generation. In: European Conference on Computer Vision. pp. 115–135. Springer (2024)

[15] Hu, S., Chen, L., Wu, P., Li, H., Yan, J., Tao, D.: St-p3: End-to-end vision-based autonomous driving via spatial-temporal feature learning. In: European Conference on Computer Vision. pp. 533–549. Springer (2022)

[16] Hu, Y., Yang, J., Chen, L., Li, K., Sima, C., Zhu, X., Chai, S., Du, S., Lin, T., Wang, W., et al.: Planningoriented autonomous driving. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 17853–17862 (2023)

[17] Huang, J., Huang, G., Zhu, Z., Ye, Y., Du, D.: Bevdet: High-performance multi-camera 3d object detection in bird-eye-view. arXiv preprint arXiv:2112.11790 (2021)

[18] Jia, X., Gao, Y., Chen, L., Yan, J., Liu, P.L., Li, H.: Driveadapter: Breaking the coupling barrier of perception and planning in end-to-end autonomous driving. In: 2023 IEEE/CVF International Conference on Computer Vision (ICCV). pp. 7919–7929 (2023). https://doi.org/10.1109/ICCV51070.2023.00731

[19] Jia, X., Wu, P., Chen, L., Xie, J., He, C., Yan, J., Li, H.: Think twice before driving: Towards scalable decoders for end-to-end autonomous driving. In: 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 21983–21994. IEEE (2023)

[20] Jia, X., Yang, Z., Li, Q., Zhang, Z., Yan, J.: Bench2drive: Towards multi-ability benchmarking of closedloop end-to-end autonomous driving. Advances in Neural Information Processing Systems 37, 819–844 (2024)

[21] Jiang, B., Chen, S., Xu, Q., Liao, B., Chen, J., Zhou, H., Zhang, Q., Liu, W., Huang, C., Wang, X.: Vad: Vectorized scene representation for efficient autonomous driving. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 8340–8350 (2023)

[22] Kingma, D., Salimans, T., Poole, B., Ho, J.: Variational diffusion models. Advances in neural information processing systems 34, 21696–21707 (2021)

[23] Kingma, D.P., Welling, M.: Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114 (2013)

[24] Li, B., Guo, J., Liu, H., Zou, Y., Ding, Y., Chen, X., Zhu, H., Tan, F., Zhang, C., Wang, T., et al.: Uniscene: Unified occupancy-centric driving scene generation. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 11971–11981 (2025)

[25] Li, Y., Yu, Z., Philion, J., Anandkumar, A., Fidler, S., Jia, J., Alvarez, J.: End-to-end 3d tracking with decoupled queries. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 18302–18311 (2023)

[26] Li, Y., Ge, Z., Yu, G., Yang, J., Wang, Z., Shi, Y., Sun, J., Li, Z.: Bevdepth: Acquisition of reliable depth for multi-view 3d object detection. In: Proceedings of the AAAI conference on artificial intelligence. vol. 37, pp. 1477–1485 (2023)

[27] Li, Z., Wang, W., Li, H., Xie, E., Sima, C., Lu, T., Yu, Q., Dai, J.: Bevformer: learning bird’s-eye-view representation from lidar-camera via spatiotemporal transformers. IEEE Transactions on Pattern Analysis and Machine Intelligence (2024)

[28] Liang, A., Liu, Y., Yang, Y., Lu, D., Li, L., Kong, L., Zhao, H., Ooi, W.T.: Lidarcrafter: Dynamic 4d world modeling from lidar sequences. arXiv preprint arXiv:2508.03692 (2025)

[29] Liang, M., Yang, B., Zeng, W., Chen, Y., Hu, R., Casas, S., Urtasun, R.: Pnpnet: End-to-end perception and prediction with tracking in the loop. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 11553–11562 (2020)

[30] Liang, T., Xie, H., Yu, K., Xia, Z., Lin, Z., Wang, Y., Tang, T., Wang, B., Tang, Z.: Bevfusion: A simple and robust lidar-camera fusion framework. Advances in neural information processing systems 35, 10421–10434 (2022)

[31] Lin, H., Chen, S., Liew, J., Chen, D.Y., Li, Z., Shi, G., Feng, J., Kang, B.: Depth anything 3: Recovering the visual space from any views. arXiv preprint arXiv:2511.10647 (2025)

[32] Liu, Z., Tang, H., Amini, A., Yang, X., Mao, H., Rus, D., Han, S.: Bevfusion: Multi-task multi-sensor fusion with unified bird’s-eye view representation. In: IEEE International Conference on Robotics and Automation (ICRA) (2023)

[33] Nakashima, K., Kurazume, R.: Lidar data synthesis with denoising diffusion probabilistic models. In: 2024 IEEE International Conference on Robotics and Automation (ICRA). pp. 14724–14731. IEEE (2024)

[34] Ni, J., Guo, Y., Liu, Y., Chen, R., Lu, L., Wu, Z.: Maskgwm: A generalizable driving world model with video mask reconstruction. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 22381–22391 (2025)

[35] Nichol, A., Dhariwal, P., Ramesh, A., Shyam, P., Mishkin, P., McGrew, B., Sutskever, I., Chen, M.: Glide: Towards photorealistic image generation and editing with text-guided diffusion models. arXiv preprint arXiv:2112.10741 (2021)

[36] Nichol, A.Q., Dhariwal, P.: Improved denoising diffusion probabilistic models. In: International conference on machine learning. pp. 8162–8171. PMLR (2021)

[37] Oquab, M., Darcet, T., Moutakanni, T., Vo, H., Szafraniec, M., Khalidov, V., Fernandez, P., Haziza, D., Massa, F., El-Nouby, A., et al.: Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193 (2023)

[38] Pang, Z., Li, Z., Wang, N.: Simpletrack: Understanding and rethinking 3d multi-object tracking. In: European Conference on Computer Vision. pp. 680–696. Springer (2022)

[39] Ramesh, A., Dhariwal, P., Nichol, A., Chu, C., Chen, M.: Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125 1(2), 3 (2022)

[40] Ran, H., Guizilini, V., Wang, Y.: Towards realistic scene generation with lidar diffusion models. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 14738–14748 (2024)

[41] Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B.: High-resolution image synthesis with latent diffusion models. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 10684–10695 (2022)

[42] Saharia, C., Chan, W., Saxena, S., Li, L., Whang, J., Denton, E.L., Ghasemipour, K., Gontijo Lopes, R., Karagol Ayan, B., Salimans, T., et al.: Photorealistic text-to-image diffusion models with deep language understanding. Advances in neural information processing systems 35, 36479–36494 (2022)

[43] Sohl-Dickstein, J., Weiss, E., Maheswaranathan, N., Ganguli, S.: Deep unsupervised learning using nonequilibrium thermodynamics. In: International conference on machine learning. pp. 2256–2265. PMLR (2015)

[44] Song, J., Meng, C., Ermon, S.: Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502 (2020)

[45] Song, Y., Ermon, S.: Generative modeling by estimating gradients of the data distribution. Advances in neural information processing systems 32 (2019)

[46] Song, Y., Ermon, S.: Improved techniques for training score-based generative models. Advances in neural information processing systems 33, 12438–12448 (2020)

[47] Song, Y., Sohl-Dickstein, J., Kingma, D.P., Kumar, A., Ermon, S., Poole, B.: Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456 (2020)

[48] Van Den Oord, A., Kalchbrenner, N., Kavukcuoglu, K.: Pixel recurrent neural networks. In: International conference on machine learning. pp. 1747–1756. PMLR (2016)

[49] Wu, Y., Zhang, K., Qian, J., Xie, J., Yang, J.: Text2lidar: Text-guided lidar point cloud generation via equirectangular transformer. In: European Conference on Computer Vision. pp. 291–310. Springer (2024)

[50] Wu, Z., Ni, J., Wang, X., Guo, Y., Chen, R., Lu, L., Dai, J., Xiong, Y.: Holodrive: Holistic 2d-3d multi-modal street scene generation for autonomous driving. arXiv preprint arXiv:2412.01407 (2024)

[51] Xie, Y., Xu, C., Peng, C., Zhao, S., Ho, N., Pham, A.T., Ding, M., Tomizuka, M., Zhan, W.: X-drive: Crossmodality consistent multi-sensor data synthesis for driving scenarios. arXiv preprint arXiv:2411.01123 (2024)

[52] Xiong, Y., Ma, W.C., Wang, J., Urtasun, R.: Ultralidar: Learning compact representations for lidar completion and generation. arXiv preprint arXiv:2311.01448 (2023)

[53] Zhang, L., Rao, A., Agrawala, M.: Adding conditional control to text-to-image diffusion models. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 3836–3847 (2023)

[54] Zhang, T., Chen, X., Wang, Y., Wang, Y., Zhao, H.: Mutr3d: A multi-camera tracking framework via 3dto-2d queries. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 4537–4546 (2022)

[55] Zhou, S., Jia, X., Zhang, F., Li, J., Zhang, J., Feng, Y., Sun, J., Wong, S., You, J., Yan, J.: Lagen: Towards autoregressive lidar scene generation. arXiv preprint arXiv:2511.21256 (2025)

[56] Zhou, W., Yan, X., Liao, Y., Lin, Y., Huang, J., Zhao, G., Cui, S., Li, Z.: Bev@ dc: Bird’s-eye view assisted training for depth completion. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 9233–9242 (2023)

[57] Zyrianov, V., Zhu, X., Wang, S.: Learning to generate realistic lidar point clouds. In: European Conference on Computer Vision. pp. 17–35. Springer (2022)

## A Overview of the Appendix

The appendix is structured into the following sections to supplement the main paper. The first part provides a discussion on the limitations of this research and directions for future work. The second part presents additional experimental details. The third part provides an in-depth theoretical analysis of the proposed modules. The fourth part supplements the experimental results.

## B Limitations and Future Work

Our work presents several limitations. Primarily, relying on the range image representation is not entirely lossless, resulting in the omission of certain camera image details within the synthesized LiDAR point clouds. Refining this representation in future research holds the potential to further boost model performance. Furthermore, the overall architecture of our method is not specifically designed for explicit temporal modeling. Although this work does not focus on generating temporal LiDAR sequences, incorporating temporal consistency mechanisms in future work could extend the applicability of our method to a broader range of scenarios.

## C More Implementation Details

During training, our model is trained for 200 epochs with a total of 176,000 optimization steps. The total batch size is set to 32, and the learning rate is set to $1 \times 1 0 ^ { - 4 }$ , with a cosine annealing scheduler applied. We perform distributed training on 4 NVIDIA H200 GPUs, with a total training time of approximately 20 hours. For denoising, we employ the DDIM sampler [44], with the total number of diffusion timesteps T set to 1024. During inference, we perform 50 denoising steps to generate point clouds. The inference is conducted on a single NVIDIA H200 GPU.

We adopt DepthAnything3 [31] as the pre-trained visual encoder. For the semantic branch, we follow DepthAnything3 and employ DINOv2 [37] to extract visual features. Based on the nuScenes dataset, we use six camera images from different viewpoints as input to generate each scene.

In the geometry-aware branch, the encoder primarily consists of multiple convolutional layers, normalization layers, downsampling layers, zero-convolution layers, and FiLM modules. In the crossview semantic branch, the encoder mainly includes a multi-head self-attention layer, a feed-forward network (composed of two fully connected layers), and two Layer Normalization layers. Both the attention layer and the FFN employ residual connections.

The VAE in the LDM is initialized with pre-trained weights from RangeLDM [14]. The range image resolution is set to [1024, 32], and the VAE uses a downsampling factor of 4. The UNet consists of three downsampling blocks (the latter two incorporating Transformer layers) and three upsampling blocks (the first two incorporating Transformer layers). We employ circular convolution instead of standard convolution in both the VAE and the LDM.

## D Detailed Description of Evaluation Metrics

In the main text, we introduced the Maximum Mean Discrepancy (MMD), Jensen-Shannon Divergence (JSD), Fréchet Point cloud Distance (FPD), and Fréchet Range Image Distance (FRD) metrics to evaluate the performance of the generator. Below, we provide a detailed explanation.

Maximum Mean Discrepancy (MMD) is a non-parametric method for measuring the distance between two probability distributions. The fundamental idea is to compute the distance between the kernel means of two sample sets in a high-dimensional space; by comparing the means mapped to this space, one can determine whether the sample distributions are consistent:

$$
M M D = \frac { 1 } { N ^ { 2 } } \sum _ { i } ^ { N } \sum _ { j } ^ { N } k ( p _ { i } , p _ { j } ) + \frac { 1 } { N ^ { 2 } } \sum _ { i } ^ { N } \sum _ { j } ^ { N } k ( q _ { i } , q _ { j } ) - \frac { 2 } { N ^ { 2 } } \sum _ { i } ^ { N } \sum _ { j } ^ { N } k ( p _ { i } , q _ { j } ) .\tag{9}
$$

Jensen-Shannon Divergence (JSD) is a symmetric information-theoretic measure for quantifying the similarity between two probability distributions, derived from the Kullback-Leibler (KL) divergence

and commonly used for distribution comparison. The essence of JSD is to evaluate the information loss of each distribution with respect to their uniform mixture:

$$
J S D = { \frac { 1 } { 2 } } \sum _ { i } ^ { N } ( p _ { i } \log { \frac { p _ { i } } { m _ { i } } } + q _ { i } \log { \frac { q _ { i } } { m _ { i } } } ) .\tag{10}
$$

We compute the MMD and JSD using 160 × 100 2D histograms on the bird’s-eye view (BEV) plane.

Fréchet Point cloud Distance (FPD) is used to measure the distance between the generated point cloud distribution and the real point cloud distribution. It utilizes a pre-trained point cloud feature extraction network (such as PointNet or PointNet++) to extract global features, and then computes the Fréchet distance between two Gaussian distributions:

$$
F P D = \| \mu _ { r } - \mu _ { g } \| _ { 2 } ^ { 2 } + \operatorname { T r } ( \sigma _ { r } + \sigma _ { g } - 2 \sqrt { \sigma _ { r } \cdot \sigma _ { g } } ) .\tag{11}
$$

Fréchet Range Image Distance (FRD) is a variant specifically designed for LiDAR data. It first converts the point cloud into a Range Image and then computes the Fréchet distance. It uses a 2D CNN (such as ResNet) to extract features. The formula is identical to that of FPD, but the source of the features differs.

To evaluate the temporal consistency of the synthesized sequences, we follow LiDARCrafter [28] and adopt the TTCE (Translation and Transformation Consistency Error) and CTC (Chamfer Temporal Consistency) metrics.

Let $T _ { t } ^ { g t } = [ R _ { t } ^ { g t } \ | \ t _ { t } ^ { g t } ]$ denote the ground-truth transformation from frame t to $t + 1$ , where $R _ { t } ^ { g t }$ represents the rotation matrix and $t _ { t } ^ { g t }$ is the translation vector. Let $T _ { t } ^ { p r e d } = [ R _ { t } ^ { p r e d } \ | \ t _ { t } ^ { p r e d } ]$ be the transformation estimated from the generated point clouds via Iterative Closest Point (ICP). The TTCE is calculated as follows:

$$
T T C E = \frac { 1 } { T - 1 } \sum _ { t = 1 } ^ { T - 1 } \Vert t _ { t } ^ { p r e d } - t _ { t } ^ { g t } \Vert _ { F } ,\tag{12}
$$

where $\| \cdot \| _ { F }$ denotes the Frobenius norm and $T$ is the total number of frames in the sequence.

Furthermore, let $\mathcal { P } _ { t }$ and $\mathcal { P } _ { t + 1 }$ be the generated point clouds at frames t and $t + 1$ , respectively. Using the ground-truth transformation $T _ { t } ^ { g t }$ , we align $\mathcal { P } _ { t + 1 }$ to the coordinate system of frame t:

$$
\tilde { \mathcal { P } } _ { t + 1 } = \left( R _ { t } ^ { g t } \right) ^ { - 1 } \left( \mathcal { P } _ { t + 1 } - t _ { t } ^ { g t } \right) .\tag{13}
$$

Subsequently, the Chamfer Distance (CD) between $\mathcal { P } _ { t }$ and the aligned $\tilde { \mathcal { P } } _ { t + 1 }$ is computed:

$$
C D \left( \mathcal { P } _ { t } , \tilde { \mathcal { P } } _ { t + 1 } \right) = \frac { 1 } { \| \mathcal { P } _ { t } \| } \sum _ { x \in \mathcal { P } _ { t } } \operatorname* { m i n } _ { y \in \tilde { \mathcal { P } } _ { t + 1 } } \| x - y \| _ { 2 } ^ { 2 } + \frac { 1 } { \| \tilde { \mathcal { P } } _ { t + 1 } \| } \sum _ { y \in \tilde { \mathcal { P } } _ { t + 1 } } \operatorname* { m i n } _ { x \in \mathcal { P } _ { t } } \| y - x \| _ { 2 } ^ { 2 } .\tag{14}
$$

The final CTC score is defined as the average Chamfer Distance across all adjacent frame pairs:

$$
C T C = \frac { 1 } { T - 1 } \sum _ { t = 1 } ^ { T - 1 } C D \left( \mathcal { P } _ { t } , \tilde { \mathcal { P } } _ { t + 1 } \right) .\tag{15}
$$

Moreover, we introduce Chamfer Distance (CD) [4] and Earth Mover’s Distance (EMD) to evaluate the fidelity of the generated point clouds in the camera-to-LiDAR task.

Chamfer Distance is a commonly used metric for measuring similarity between point clouds. It quantifies the “alignment” distance between two sets of points, representing the sum of the distances from each point in one set to its nearest neighbor in the other set. By aggregating these nearest neighbor distances across all points, Chamfer Distance captures the overall distribution difference:

$$
C D ( P _ { \mathrm { g e n } } ^ { s } , P _ { \mathrm { g t } } ^ { s } ) = \frac { 1 } { M } \sum _ { x \in P _ { \mathrm { g t } } ^ { s } } \operatorname* { m i n } _ { y \in P _ { \mathrm { g e n } } ^ { s } } \| x - y \| _ { 2 } ^ { 2 } + \frac { 1 } { N } \sum _ { y \in P _ { \mathrm { g e n } } ^ { s } } \operatorname* { m i n } _ { x \in P _ { \mathrm { g t } } ^ { s } } \| y - x \| _ { 2 } ^ { 2 } .\tag{16}
$$

Earth Mover’s Distance (EMD) measures the minimum average cost required to transport the generated point cloud to the real point cloud. Compared with CD, EMD better reflects differences in the global structure and density distribution between two point clouds:

$$
E M D ( P , Q ) = \operatorname* { m i n } _ { T \geq 0 } \sum _ { i = 1 } ^ { N } \sum _ { j = 1 } ^ { M } T _ { i j } d ( p _ { i } , q _ { j } ) , \quad \mathrm { s . t . } \sum _ { j = 1 } ^ { M } T _ { i j } = a _ { i } , \sum _ { i = 1 } ^ { N } T _ { i j } = b _ { j } .\tag{17}
$$

Considering the actual requirements of autonomous driving scenarios, we measured the above indicators within the range of [-50.0, 50.0] m.

In downstream detection tasks for autonomous driving, we introduce the mean Average Precision (mAP) and nuScenes Detection Score (NDS) metrics to evaluate detection accuracy and overall perception quality.

The mAP metric measures the precision-recall performance of the detection model across different categories:

$$
m A P = \frac { 1 } { C } \sum _ { c = 1 } ^ { C } A P _ { c } ,\tag{18}
$$

where $A P$ is the area under the Precision-Recall curve:

$$
A P = \sum _ { n } ( R _ { n } - R _ { n - 1 } ) \cdot P _ { i n t e r p } ( R _ { n } ) ,\tag{19}
$$

where P and R represent precision and recall, respectively.

The NDS summarizes overall 3D detection performance:

$$
N D S = \frac { 1 } { 1 0 } \left( 5 \cdot m A P + \sum _ { k \in A T E , A S E , A V E , A A E } ( 1 - \operatorname* { m i n } ( 1 , e _ { k } ) ) \right) ,\tag{20}
$$

where ATE, ASE, AOE, AVE, and $A A E$ represent position error, size error, orientation error, velocity error, and attribute error, respectively.

## E Standard Range-View Projection and Corrected Version

Range images offer a compact representation of LiDAR data by parametrizing unstructured point clouds in 3D space into a 2D pixel space. This projection process is reversible, and after applying it, both training and inference costs for the model can be significantly reduced. Consequently, representing LiDAR data with range images has become one of the prevailing trends in the field.

The standard range image mapping method projects each point $p = ( x , y , z ) \in \mathbb { R } ^ { 3 }$ in the point cloud into the spherical coordinates $( r , \theta , \phi )$ of the range image using spherical coordinate projection:

$$
r = \sqrt { x ^ { 2 } + y ^ { 2 } + z ^ { 2 } } , \quad \theta = \arctan ( y , x ) , \quad \phi = \arctan \left( z , \sqrt { x ^ { 2 } + y ^ { 2 } + z ^ { 2 } } \right) ,\tag{21}
$$

where $r$ is the depth value, $\theta$ is the azimuth angle, and $\phi$ is the elevation angle. RangeLDM [14] points out that multiple lasers in a LiDAR system do not share the same origin during measurement, which can introduce errors during coordinate transformation and lead to incorrect range-view data distributions. Therefore, we adopt its corrected range image representation as follows:

$$
r = \sqrt { x ^ { 2 } + y ^ { 2 } + ( z - h _ { j } ) ^ { 2 } } , \quad \theta = \arctan ( y , x ) , \quad \phi = \phi _ { j } ,\tag{22}
$$

where $h _ { j }$ and $\phi _ { j }$ represent the sensor height and elevation angle estimated by Hough voting, respectively. In subsequent computations, both projection methods are processed in exactly the same way, including converting spherical coordinates $( r , \theta , \phi )$ to range image pixel coordinates $( u , v )$

$$
\begin{array} { r } { u = W - \left( ( { \theta } + { \pi } ) / { 2 \pi } \right) \cdot W , \quad v = H - j , } \end{array}\tag{23}
$$

as well as normalizing the depth values of the point cloud to obtain the range image $I .$

In the subsequent calculation of the CD metric, it is necessary to reverse-convert the range image back into a point cloud. In the corrected version, the standard LiDAR’s fifth dimension, which is the beam index $j$ for each point, is directly used during the forward projection process. However, the generated point clouds do not contain the beam index dimension. Therefore, we introduce the following formula to compute the beam index for each point:

$$
j = ( \mathrm { R a d 2 D e g } ( \arcsin ( \frac { z } { d } ) ) - f o v _ { \mathrm { d o w n } } ) / ( f o v _ { \mathrm { u p } } - f o v _ { \mathrm { d o w n } } ) \cdot ( H - 1 ) ,\tag{24}
$$

where $f o v _ { \mathrm { u p } }$ and $f o v _ { \mathrm { d o w n } }$ refer to the elevation angle range of the original LiDAR acquisition.

## F Theoretical Analysis of the Geometry-Aware Mechanism

## F.1 Computational Complexity

Let the resolution of the high-resolution camera depth map be $H _ { i m g } \times W _ { i m g } ,$ , and the target LiDAR range image resolution be $\bar { H } _ { r } \times W _ { r }$ . Conventional projection methods require iterating over each pixel of the depth map, calculating the 3D coordinates, and projecting them onto a cylindrical coordinate system, followed by rasterization. The computational complexity of this process is $\mathcal { O } ( H _ { i m g } \cdot W _ { i m g } )$ Since this process involves mapping from a high-resolution image to a low-resolution LiDAR grid, it typically results in many-to-one write conflicts. To address the occlusion problem, additional aggregation operations are required, which introduce extra computational overhead.

In contrast, the reverse sampling method starts directly from the target LiDAR grid. For each query point $q _ { k }$ , the projection via the camera is:

$$
\begin{array} { r } { [ \tilde { u } _ { n } , \tilde { v } _ { n } , \tilde { w } _ { n } ] ^ { T } = K _ { n } ( R _ { n } \cdot q _ { k } + t _ { n } ) , } \end{array}\tag{25}
$$

where $K _ { n } , R _ { n }$ , and $t _ { n }$ are the camera’s intrinsic matrix, rotation matrix, and translation vector. The overall time complexity is strictly constrained to $\mathcal { O } ( H _ { r } \cdot W _ { r } )$ . Since in physical scenes, $H _ { r } \cdot W _ { r } \ll$ $H _ { i m g } \cdot W _ { i m g }$ , the reverse sampling significantly reduces the computational overhead.

## F.2 Spatial Sampling Distribution

The forward projection can be formalized as a nonlinear mapping from the image plane $\Omega _ { \mathrm { i m g } } \subset \mathbb { R } ^ { 2 }$ to the three-dimensional space:

$$
\mathcal { F } ( u , v ) = D ( u , v ) \cdot K ^ { - 1 } [ u , v , 1 ] ^ { \top } ,\tag{26}
$$

where $D ( u , v )$ is the depth at pixel $( u , v )$ , and $K ^ { - 1 }$ is the inverse of the camera’s intrinsic matrix. Due to the nonlinearity of perspective projection, the Jacobian determinant satisfies:

$$
\left| \operatorname * { d e t } { \frac { \partial ( x , y ) } { \partial ( X , Y ) } } \right| \propto { \frac { 1 } { Z ^ { 2 } } } ,\tag{27}
$$

where $Z$ is the depth. Therefore, uniform sampling on the image induces a depth-dependent nonuniform distribution in 3D space:

$$
\rho ( X , Y , Z ) \propto \frac { 1 } { Z ^ { 2 } } .\tag{28}
$$

This results in over-sampling in nearby regions and severe under-sampling in distant regions, leading to significant aliasing and ghost artifacts when mapped to the regular LiDAR range image grid. In contrast, our proposed reverse sampling method directly constructs the set of query points $Q _ { D } = \{ q _ { k } \}$ in the target LiDAR domain, with a distribution that is consistent with the LiDAR’s spherical scanning pattern. Specifically, the query points are generated via spherical coordinate parametrization, which is equivalent to performing structured discrete sampling on the target spherical domain, rather than performing non-uniform reprojection in the image domain.

For each query point $q _ { k }$ , we project it onto the image plane to obtain continuous coordinates $( u _ { n } , v _ { n } )$ and then use nearest-neighbor sampling to extract the corresponding value from the depth map:

$$
\hat { D } ( q _ { k } ) = D _ { n } ( \lfloor u _ { n } \rceil , \lfloor v _ { n } \rceil ) .\tag{29}
$$

This process can be viewed as a zero-order hold approximation of the continuous depth field, with the error arising from pixel quantization. Assuming the underlying depth field $D ^ { * }$ is Lipschitz continuous, the sampling error satisfies:

$$
| \hat { D } ( q _ { k } ) - D ^ { * } ( u _ { n } , v _ { n } ) | \leq L \cdot \delta ,\tag{30}
$$

where L is the local Lipschitz constant, and δ is the pixel scale.

## F.3 Weighted Estimation and Optimization Dynamics

In the multi-view geometric conditioning process, due to occlusion and depth estimation errors, the intermediate features $H _ { l }$ of the encoder are often corrupted by spatially correlated and unevenly distributed noise. We model this as:

$$
H _ { l } ( x , y ) = H _ { \mathrm { t r u e } } ( x , y ) + \epsilon ( x , y ) ,\tag{31}
$$

where the noise term satisfies:

$$
\epsilon ( x , y ) \sim \mathcal { N } ( 0 , \sigma ^ { 2 } ( x , y ) ) ,\tag{32}
$$

and the noise variance $\sigma ^ { 2 } ( x , y )$ is monotonically related to the uncertainty map $I _ { U } ( x , y ) , { \mathrm { i } } . { \mathrm { e } } .$

$$
\sigma ^ { 2 } ( x , y ) \propto I _ { U } ( x , y ) .\tag{33}
$$

This model corresponds to a typical heteroscedastic noise setting, where different spatial locations exhibit varying levels of noise intensity.

In the above setup, a natural objective is to weight the features based on local noise intensity to reduce the impact of high-noise regions on the overall estimate. Consider the following weighted form:

$$
\hat { H } ( x , y ) = w ( x , y ) \cdot H _ { l } ( x , y ) ,\tag{34}
$$

where the weight $w ( x , y )$ should decrease as the noise variance increases. Classical statistical estimation theory indicates that, under certain distribution assumptions, a reasonable weight form is typically inversely proportional to the noise variance:

$$
w ( x , y ) \propto \frac { 1 } { \sigma ^ { 2 } ( x , y ) } .\tag{35}
$$

In our setup, since $\sigma ^ { 2 } ( x , y )$ is approximately characterized by $I _ { U } ( x , y )$ , the weight can be treated as a function of $I _ { U }$ :

$$
w ( x , y ) = g ( I _ { U } ( x , y ) ) ,\tag{36}
$$

where $g ( \cdot )$ is a monotonically decreasing function.

We introduce uncertainty-based FiLM modulation into the model:

$$
\tilde { H } _ { l } = \gamma _ { l } ( I _ { U } ) \odot H _ { l } + \beta _ { l } ( I _ { U } ) ,\tag{37}
$$

where $\gamma _ { l } ( \cdot )$ and $\beta _ { l } ( \cdot )$ are nonlinear mappings learned by the neural network. This can be understood as a learnable heteroscedastic weighted estimator, where the weight function is learned in a datadriven manner. In contrast, if the uncertainty $I _ { U }$ were directly used as an additional input channel to the network, the model would need to implicitly learn the above weighting relationship through high-dimensional nonlinear mappings. This approach presents challenges such as high learning difficulty, poor sample efficiency, and optimization instability.

It also has a direct impact on the backpropagation process. For the loss function ${ \mathcal { L } } ,$ we have:

$$
\frac { \partial \mathcal { L } } { \partial H _ { l } } = \gamma _ { l } ( I _ { U } ) \odot \frac { \partial \mathcal { L } } { \partial \tilde { H } _ { l } } .\tag{38}
$$

As a result, the gradient magnitude is modulated by $\gamma _ { l } ( I _ { U } )$ , and when $I _ { U } ( x , y )$ is large, $\gamma _ { l } ( I _ { U } )$ tends to decrease, causing the gradient to be attenuated. This enables the model to adaptively suppress gradient interference from unreliable regions during optimization, thereby improving training stability.

## G Theoretical Analysis of the Cross-View Semantic Branch

In multi-view systems, the 2D images from different viewpoints are projections of the same scene under different camera poses, and the imaging process can be represented as a projection mapping $\pi : \mathbb { R } ^ { 3 }  \mathbb { R } ^ { 2 }$ . Due to the lack of depth information, the 3D correspondence of any pixel $( u , v )$ is a set of points along a ray:

$$
\pi ^ { - 1 } ( u , v ) = \{ X \mid \pi ( X ) = ( u , v ) \} ,\tag{39}
$$

which means that relying solely on 2D features for cross-view interaction is theoretically ill-posed.

To address this, we introduce patch encoding $E _ { \mathrm { p a t c h } }$ , viewpoint encoding $E _ { \mathrm { v i e w } }$ , and camera extrinsic parameter encoding $E _ { \mathrm { e x t } } ( T )$ , to construct pose-conditioned features and explicitly introduce the conditional distribution into the feature space. This design reduces the feature confusion caused by projection uncertainty.

Building on this, we employ a self-attention mechanism to model the cross-view dependencies. This process can be formalized as an adaptive kernel regression:

$$
H _ { i } = \sum _ { j } \kappa ( \tilde { x } _ { i } , \tilde { x } _ { j } ) \cdot x _ { j } ,\tag{40}
$$

where the kernel function $\kappa ( \cdot , \cdot )$ is implicitly defined by feature similarity. Since the features contain pose information, this kernel function is able to learn cross-view association patterns related to 3D geometric relationships during training, thereby promoting geometric consistency in the latent space. Furthermore, if we define the cross-view feature consistency error as

$$
\mathcal { L } _ { c o n s } = \sum _ { i , j } A _ { i j } \| x _ { i } - x _ { j } \| ^ { 2 } ,\tag{41}
$$

the above attention update is equivalent to a kernel smoothing operation, which can reduce this consistency error during optimization, thus achieving cross-view feature alignment.

At the same time, our method achieves the complementary integration of local and global information through the collaborative modeling of the geometric and semantic branches. Specifically, the local geometric branch tends to capture high-frequency spatial structures (such as depth discontinuities and boundaries), while the attention-based semantic branch models long-range dependencies by globally weighted aggregation, focusing more on low-frequency structures. In summary, this design can be understood as learning a pose-conditioned kernel regression operator, which preserves local details while effectively enhancing cross-view consistency through global information interaction.

## H Theoretical Analysis of the Dual-Branch Balanced CFG

In conditional diffusion models, the introduction of a conditional variable $\boldsymbol { c } = \left( c _ { g } , c _ { s } \right)$ aims to learn the conditional distribution $p ( x \mid c )$ , ensuring that the generated results maintain a statistical dependency with the condition variable. Although the training objective of the diffusion model does not directly maximize mutual information, the maximum likelihood objective can be interpreted as implicitly encouraging the increase of mutual information between x and the condition variables. According to the chain rule for mutual information, we have:

$$
I ( x ; c _ { g } , c _ { s } ) = I ( x ; c _ { g } ) + I ( x ; c _ { s } \mid c _ { g } ) ,\tag{42}
$$

In standard joint training, if the geometric condition $c _ { g }$ provides strong structural constraints, the model may preferentially utilize $c _ { g }$ to minimize the loss during the early stages of optimization, thereby weakening its reliance on the semantic condition $c _ { s }$ . This phenomenon can be interpreted as an information shortcut, where the model tends to rely on the conditioning variable that is easier to optimize. To mitigate this issue, we sample the condition c from the following set during training:

$$
c \sim \{ \emptyset , c _ { g } , c _ { s } , ( c _ { g } , c _ { s } ) \} ,\tag{43}
$$

and the corresponding optimization objective is defined as:

$$
\mathcal { L } = \mathbb { E } _ { c } \mathbb { E } _ { x , \epsilon , t } [ \| \epsilon - \epsilon _ { \theta } ( x _ { t } , c , t ) \| ^ { 2 } ] .\tag{44}
$$

This objective is equivalent to simultaneously fitting multiple conditional distributions:

$$
p ( x ) , \quad p ( x \mid c _ { g } ) , \quad p ( x \mid c _ { s } ) , \quad p ( x \mid c _ { g } , c _ { s } ) .\tag{45}
$$

From the information-theoretic perspective, this training strategy explicitly encourages the model to maintain robust predictive capabilities across different conditional subspaces, thereby preventing the model from relying solely on a single conditioning variable and mitigating the issue of imbalanced information utilization.

## I Discussion on Pre-trained Vision Models and Metric Scale

## Replaceability of the Pretrained Vision Model.

From the perspective of the framework interface, LiFi does not depend on the specific internal architecture of DepthAnything3. The geometric branch only requires the depth foundation model to provide two types of per-pixel information for each input view: depth predictions and their corresponding reliability estimates. LiFi’s reverse sampling module then aligns these predictions with the target LiDAR scanning grid, while the uncertainty-aware encoder uses the reliability signals to modulate the geometric features. Therefore, as long as an alternative model can provide outputs in the same form, the subsequent inverse sampling, FiLM modulation, and diffusion generation pipeline can remain unchanged.

Therefore, from an architectural perspective, any large monocular depth model capable of estimating per-pixel uncertainty can technically be directly integrated into LiFi. However, if the model processes each camera view independently, the predicted depths may have independent scales across different views. In this case, we recommend introducing cross-view scale alignment before reverse sampling.

Specifically, one feasible approach is to use the known camera calibration and overlapping fields of view to perform scale-and-shift alignment across the depth maps from different views. For example, the prediction for each view can be expressed as: $\hat { D _ { n } } = a _ { n } D _ { n } + b _ { n }$ , where $a _ { n }$ and $b _ { n }$ are estimated by enforcing consistency within the overlapping regions. After cross-view alignment, a unified normalization scheme can be applied to all views before feeding them into LiFi.

Therefore, multi-view consistency is not a strict architectural requirement for LiFi, but it can positively contribute to maintaining the quality of the geometric conditions. In addition, methods with multi-view depth estimation capabilities, such as VGGT and MapAnything, can serve as direct replacements for DepthAnything3. Therefore, DepthAnything3 is an instantiation choice that matches the task requirements, rather than an irreplaceable component of the LiFi framework.

## Metric Scale of the Simulator.

We first emphasize that the final LiDAR generated by LiFi is already represented in the metric LiDAR coordinate system of the target dataset, rather than having only relative scale.

LiFi trains the diffusion model with range images constructed from real LiDAR scans as the target, and therefore learns a mapping from relative visual geometry and image semantics to real metric LiDAR distance distributions. After de-normalization, each valid cell in the generated range image represents an actual distance in the target sensor coordinate system. It can then be back-projected using the corresponding laser-beam elevation angles, azimuth sampling, and sensor calibration parameters to obtain a 3D point cloud in meters. The range image transformation adopted in the paper is itself reversible, and the generation target is constructed from the metric distances of real LiDAR scans.

Therefore, for datasets such as nuScenes and KITTI-360, there is no need to fit an additional global scale factor after generation. Metric LiDAR point clouds from datasets such as KITTI and Argoverse can likewise be directly converted into range images according to the scanning pattern of the target sensor and used as training supervision for LiFi.

If LiFi is to be adapted to a specific simulator, we recommend the following procedure:

1. If the simulator is intended to use a target LiDAR configuration consistent with those of current mainstream datasets, the LiFi architecture trained on these datasets can be directly used.

2. If the goal is to simulate a completely new virtual target LiDAR configuration, the first step is to specify the target LiDAR settings, including the beam elevation angles, horizontal angular resolution, and extrinsic parameters. Based on these settings, the target range image representation can then be constructed, and the VAE and diffusion model can be trained or fine-tuned using camera–metric LiDAR data corresponding to that sensor. At inference time, the generated results can be converted back into a metric point cloud according to the same normalization parameters and sensor model.

## J Analysis of Computational Efficiency

In this section, we provide an analysis of the inference efficiency of LiFi. As shown in Table 8, we present the performance and inference speed of our method across different denoising timesteps.

## K Ablation Study on the Number of Input Image Views

The number of input RGB images does not affect the fairness of the experimental comparisons but is merely an architectural design choice. In this section, we present an ablation study on the nuScenes dataset using 1, 2, 4, and 6 input camera images, as shown in Table 9.

Table 8: Quantitative evaluation of LiFi’s inference efficiency. The MMD values are reported in $1 0 ^ { - 5 }$ and JSD in $1 0 ^ { - 2 }$ . Inference time is evaluated in s/frame.
<table><tr><td>Denoising Step</td><td>MMD</td><td>JSD</td><td>CD</td><td>Time</td></tr><tr><td> $T _ { \mathrm { i n f e r e n c e } } = 1 0$ </td><td>180.97</td><td>18.00</td><td>1.29</td><td>0.11</td></tr><tr><td> $T _ { \mathrm { i n f e r e n c e } } = 3 0$ </td><td>0.98</td><td>4.29</td><td>1.00</td><td>0.31</td></tr><tr><td> $T _ { \mathrm { i n f e r e n c e } } = 5 0$ </td><td>2.78</td><td>3.66</td><td>0.83</td><td>0.52</td></tr><tr><td> $T _ { \mathrm { i n f e r e n c e } } = 1 0 0$ </td><td>2.73</td><td>3.84</td><td>0.83</td><td>1.04</td></tr><tr><td> $T _ { \mathrm { i n f e r e n c e } } = 2 0 0$ </td><td>2.76</td><td>3.83</td><td>0.82</td><td>2.08</td></tr></table>

Table 9: Quantitative evaluation of the impact of the number of input images on LiFi’s generation performance. The MMD values are reported in $1 0 ^ { - 5 }$ and JSD in $1 0 ^ { - 2 }$
<table><tr><td>Num.</td><td>MMD↓</td><td>JSD↓</td><td>CD↓</td><td>EMD↓</td></tr><tr><td>1</td><td>14.98</td><td>4.83</td><td>1.93</td><td>112.54</td></tr><tr><td>2</td><td>7.46</td><td>4.35</td><td>1.53</td><td>99.48</td></tr><tr><td>4</td><td>5.85</td><td>4.17</td><td>1.19</td><td>76.34</td></tr><tr><td>6</td><td>2.78</td><td>3.66</td><td>0.83</td><td>50.08</td></tr></table>

## L Ablation Study on the Range Image Resolution

In this section, we conduct an ablation study on the resolution of the range images, with the results presented in Table 10. At a resolution of [1024, 32], most pixels correspond to LiDAR points in the scene, whereas [512, 32] results in the omission of approximately half of the points in the complete LiDAR scene during representation conversion. In contrast, [2048, 32] introduces nearly an equal number of additional empty pixels, which can instead interfere with learning in valid regions.

Table 10: Quantitative evaluation of LiFi’s generation performance and inference efficiency across different range image resolutions. The MMD values are reported in $1 0 ^ { - 5 }$ and JSD in $1 0 ^ { - 2 }$
<table><tr><td>Range Image Size</td><td>MMD↓</td><td>JSD ↓</td><td>Time ↓</td></tr><tr><td>[512,32]</td><td>740.77</td><td>31.04</td><td>0.29</td></tr><tr><td>[1024,32]</td><td>2.78</td><td>3.66</td><td>0.52</td></tr><tr><td>[2048, 32]</td><td>3.32</td><td>4.03</td><td>0.97</td></tr></table>

## M Quantitative Analysis of Failure Cases at Long Range

In this section, we evaluate the generation performance of our model across different depth ranges.   
As shown in Table 11, LiFi’s performance gradually degrades as the distance increases.

Table 11: Quantitative evaluation of LiFi’s generation performance across different radial distance ranges. The MMD values are reported in $1 0 ^ { - 5 }$ and JSD in $1 0 ^ { - 2 }$
<table><tr><td>Radial Distance</td><td>Proportion of Points</td><td>MMD↓</td><td>JSD↓</td><td>CD↓</td><td>EMD↓</td></tr><tr><td>[0, 10) m</td><td>60.87%</td><td>2.08</td><td>2.06</td><td>0.11</td><td>2.05</td></tr><tr><td>[10, 20) m</td><td>23.69%</td><td>1.40</td><td>3.80</td><td>0.94</td><td>34.62</td></tr><tr><td>[20, 30) m</td><td>9.25%</td><td>2.48</td><td>5.52</td><td>4.13</td><td>189.64</td></tr><tr><td>[30, 40) m</td><td>3.92%</td><td>5.18</td><td>6.61</td><td>12.06</td><td>538.58</td></tr><tr><td>[40, +∞) m</td><td>2.27%</td><td>7.03</td><td>9.23</td><td>32.82</td><td>1264.71</td></tr></table>

## N More Qualitative Results

In this section, we first present additional visual examples comparing the generation results of our method with those of LiDAR generation baselines, as shown in Figure 8. We then provide more camera-to-LiDAR visualization results across diverse scenarios, as shown in Figure 9.

![](images/633b9ab4cbfe8d08a20202f200537dcee3975e39b982e68e017f096191a0fa2c.jpg)  
Figure 8: Additional qualitative generation examples of our method and several baseline approaches on the LiDAR generation task on the nuScenes dataset.

![](images/e01f8b3c813815c0b61efe6e5b7d37e6a62e1d199341bbdd9d1b9887b64cd27d.jpg)  
Figure 9: Qualitative comparisons of our method with several baseline approaches on the camera-to-LiDAR task across more representative scenarios on the nuScenes dataset.