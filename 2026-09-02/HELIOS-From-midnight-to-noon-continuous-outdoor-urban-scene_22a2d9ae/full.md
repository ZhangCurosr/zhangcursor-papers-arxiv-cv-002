Continuous Lighting Control

# HELIOS: From midnight to noon, continuous outdoor urban scene relighting

Hala Djeghim<sup>1,2</sup> Nathan Piasco<sup>1</sup> Luis Roldão<sup>1</sup> Moussab Bennehar<sup>1</sup> Dzmitry Tsishkou<sup>1</sup> Céline Loscos<sup>3</sup> Désiré Sidibé<sup>2</sup>

<sup>1</sup> Noah’s Ark, Huawei Paris Research Center, France 2 IBISC, Université Paris-Saclay, Univ Evry, France <sup>3</sup> L Research, France

![](images/21a12b98d0ad33d5ae11a0f694cdd6caed4cb4e1b23a5e8801f87202bbbc0e8a.jpg)  
Fig. 1: Continuous relighting control – From a single driving image under realworld lighting conditions, HELIOS produces realistic relightings for both Night-to-Day (top left), and Day-to-Night (top right) tasks while preserving the scene content. Controlled by the sun altitude, HELIOS enables smooth continuous relightings across the day-night cycle (bottom).

Abstract. Modifying the illumination of driving images is a fundamental challenge, as most datasets are captured at specific times of day. Existing methods rely on synthetic data or paired multi-illumination supervision, which limits their generalization to the diverse and challenging conditions of real-world scenarios. To address this, we propose HELIOS, a novel image relighting approach that relies on unlabeled realworld datasets without requiring any paired images for training. Our approach integrates albedo-based conditioning into a cycle-consistent difusion pipeline to prevent identity collapse and ensure accurate domain translation. To handle low-visibility nighttime conditions, we introduce a robust albedo distillation strategy that transfers structural stability from the daytime domain. Additionally, we replace traditional text prompts with a fine-grained control mechanism based on GPS-derived solar angles, enabling smooth and continuous lighting manipulation across the day-night cycle. Through extensive evaluation and a user study, we demonstrate that HELIOS produces structurally consistent and realistic results in both night-to-day and day-to-night tasks, outperforming state-of-the-art methods.

## 1 Introduction

Illumination plays a crucial role in shaping the visual appearance of a scene, as it controls the perceptual interpretation of both geometry and material properties. In the context of autonomous driving, the ability to manipulate lighting while preserving structural and semantic consistency is essential for downstream tasks, such as improving perception robustness and enabling realistic scene editing for data augmentation. Indeed, most public datasets are captured at limited times of day; simulating the same scene under diferent lighting conditions and at various times of day is highly valuable.

Realistic relighting of driving scenes remains a highly challenging problem. It requires accurate modeling of complex light transport and surface reflectance under highly dynamic and uncontrolled outdoor conditions. Although data-driven approaches have demonstrated impressive results in controlled settings, such as indoor environments or object-centric scenarios [22, 33], their generalization to real-world driving scenes remains limited. To resolve the impossibility of capturing perfectly aligned multi-illumination datasets in the wild, many frameworks rely heavily on synthetic data. However, such synthetic priors fail to capture the full complexity of real driving environments, leading to a domain gap that significantly degrades performance.

Recent eforts have attempted to mitigate this issue through multi-stage training pipelines combining synthetic data generation and auto-labeling of realworld data [5,12]. Although these methods achieve good results in outdoor daytime settings, they remain fundamentally constrained by the domain gap between synthetic and real nighttime distributions, where sensor noise, low signal-tonoise ratios, and complex illumination efects difer significantly from synthetic data.

To address both the lack of multi-illumination supervision data and the domain gap between real and synthetic data, we propose a novel single-image relighting framework trained exclusively on real-world datasets, without requiring any paired multi-illumination data. Inspired by CycleNet [27], our approach leverages the strong generative priors of difusion models to perform unpaired domain translation while preserving scene content through a cycle-consistency training strategy.

We introduce an albedo estimation module and condition the relighting process on albedo maps rather than input images. Therefore, the model can easily decompose the reflectance properties of the images from its illumination. This prevents the model from falling into identity collapse and forces the model to learn illumination transfer independently from the input image.

Furthermore, to address the ambiguity of nighttime conditions where visibility is low, we propose a robust albedo distillation strategy. This approach ensures reliable nighttime albedo prediction even when structural details are partially invisible or obscured by shadows, by transferring knowledge from a well-lit domain.

Finally, we compute solar angles associated to each image by exploiting GPS metadata publicly available in most driving datasets. The solar angles are encoded into a continuous control signal to replace the coarse text-based prompts. Our formulation enables a fine-grained control mechanism, allowing for smooth interpolation and precise, continuous lighting manipulation.

In summary, our main contributions are as follows:

– We propose an unpaired real-world training strategy specifically tailored for driving images relighting,

– We introduce albedo-based conditioning in CycleNet to prevent identity collapse and force the model to decouple reflectance from lighting during the translation process,

– We propose an albedo distillation strategy to solve nighttime failure cases,

– We replace ambiguous, discrete text prompts with a fine-grained control mechanism based on GPS-derived solar angles, enabling smooth and precise interpolation across the day-night cycle.

## 2 Related work

## 2.1 Relighting via Inverse Rendering

Relighting involves modifying the illumination of an image or scene while maintaining structural consistency. Traditional techniques rely on inverse rendering to decompose a scene into its intrinsic properties (G-Bufer), including geometry, albedo, and global illumination parameters [1]. While recent advances in rendering, such as NeRF [15] and 3D Gaussian Splatting [9], have enabled high-quality relightable assets, these methods often require per-scene optimization and are typically restricted to controlled domains such as objects [4, 13, 19], human portraits [8, 32], or indoor environments [22, 33].

In the context of large-scale outdoor driving scenarios, inverse rendering remains a highly ill-posed problem. Existing solutions for urban environments [14, 23] often rely on manually designed priors and are efective only in high-quality daytime captures. The complexity of real-world driving datasets, characterized by transient lighting and non-uniform material properties, makes it non-trivial to disentangle illumination from reflectance. Recent data-driven approaches [11,29] attempt to bridge this gap by training on synthetic data or augmenting with real datasets [12]. However, these frameworks often fail to generalize to challenging nighttime images where there is limited visibility.

## 2.2 Learning-based Relighting

An emerging alternative focuses on learning the relighting mapping directly via generative models [5, 7, 26, 28]. These methods leverage the priors of difusion models to synthesize relit images without the need for an intermediate G-bufer. While these models show impressive generalization on standard datasets, they struggle with the complexities of driving environments due to the absence of multi-illumination ground truth. Notably, frameworks such as UniRelight [5] demonstrate high-fidelity results in daytime settings, but their performance in nighttime conditions remains undocumented. They rely on pseudo-labels generated by DifusionRenderer [12], which fail to produce accurate albedo maps in nighttime images.

In contrast to these methods, our approach utilizes a cycle-consistent training and a robust albedo distillation strategy, enabling direct learning of relighting from real-world driving datasets that remain structurally accurate even in nighttime conditions.

## 2.3 Unpaired Image-to-Image Translation

Unpaired Image-to-Image (I2I) translation aims to learn a mapping between two distinct domains without pixel-aligned supervision. CycleGAN [34] established the cycle-consistency constraint to preserve structural content across domain shifts. While efective, GAN-based methods often sufer from limited diversity and instability in complex scenes. Recent works have integrated cycle-consistency into the Latent Difusion pipeline; notably, CycleNet [27] leverages the generative power of Stable Difusion to achieve high-fidelity translation.

However, CycleNet [27] is limited by a trade-of between structural consistency and domain translation, which often leads to identity collapse in complex tasks such as driving images relighting. Our work builds upon the cycleconsistent difusion training strategy but addresses its limitations by replacing input image conditioning with a robust albedo prior and introducing a continuous relighting control.

## 3 Background

## 3.1 CycleNet

Unpaired I2I translation aims to learn a mapping between domains X and Y without any aligned paired samples. CycleNet [27] adapts the cycle-consistency principle [34] to a difusion framework by enforcing that an image $x \in \mathcal { X }$ , when translated to domain Y, subsequently translated back, remains reconstructible to the original source.

This relies on the assumption that the structural condition $c _ { \mathrm { i m g } }$ remains invariant, while the target domain is controlled by the conditioning signal $c _ { \mathrm { t e x t } }$ To implement this, they use a ControlNet [30] architecture. The network $F _ { \theta }$ is trained to predict the clean sample $x _ { 0 }$ at a specific timestep t for a given noisy latent $x _ { t }$ . They first introduce a reconstruction loss that ensures the model can recover the original input when conditioned on its own domain label:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { x \to x } } = \mathbb { E } _ { { x , \epsilon } \sim \mathcal { N } ( 0 , I ) , t } \left[ \big \| x _ { 0 } - F _ { \theta } \big ( x _ { t } , c _ { x } , x _ { 0 } \big ) \big \| _ { 2 } ^ { 2 } \right] , } \end{array}\tag{1}
$$

![](images/5c76f4cf9ac5912593a186e09fe7343bcfd24ddb9860bfb03c3090db523340f8.jpg)  
Fig. 2: Overview of the HELIOS Architecture. Our framework performs unpaired relighting conditioned on the albedo a and solar altitude $\theta _ { s } .$ Starting from a daytime source sample $x _ { 0 }$ , the model first optimizes a reconstruction objective $\scriptstyle { \mathcal { L } } _ { \mathrm { r e c o } }$ conditioned on the elevation $\theta _ { s } > 0$ . Domain translation is achieved by generating a relit nighttime representation $\bar { y }$ conditioned on $\theta _ { s } ~ < ~ 0$ . To ensure structural integrity, $\bar { y }$ is mapped back to the source domain as $x _ { c }$ via the cycle-consistency loss $\mathcal { L } _ { \mathit { c y c l e } }$

They define a cycle consistency loss that ensures that translation from both domains can reconstruct the original image $x _ { 0 }$ . Let $\bar { y } = F _ { \theta } ( x _ { t } , c _ { y } , x _ { 0 } )$ be an intermediate translation where $y _ { t } = \bar { y } + \epsilon _ { y }$ is its noised version. The cycle loss is then defined as:

$$
\mathcal { L } _ { \mathrm { x }  \mathrm { y }  \mathrm { x } } = \mathbb { E } _ { \boldsymbol { x } , \epsilon \sim \mathcal { N } ( 0 , I ) , t } [ \| \boldsymbol { x } _ { 0 } - F _ { \theta } ( y _ { t } , c _ { x } , x _ { 0 } ) \| _ { 2 } ^ { 2 } ] ,\tag{2}
$$

Finally, an invariance loss is used to ensure the target domain remains stable under repeated transfers:

$$
\mathcal { L } _ { \mathrm { x \to y \to y } } = \mathbb { E } _ { z , \epsilon \sim \mathcal { N } ( 0 , I ) , t } \left[ \| F _ { \theta } ( x _ { t } , c _ { y } , x _ { 0 } ) - F _ { \theta } ( x _ { t } , c _ { y } , \bar { y } ) \| _ { 2 } ^ { 2 } \right] .\tag{3}
$$

## 4 Method

Our goal is to perform unpaired image-to-image translation to relight driving scenes from a single image. To achieve this, our method is trained exclusively on unpaired real-world datasets, without any synthetic supervision or paired captures.

We first introduce an albedo regularization strategy in Sec. 4.2, replacing image conditioning with an invariant albedo prior to prevent identity collapse. In Sec. 4.3 we present a continuous relighting control. Finally, in Sec. $5 ,$ we propose a novel distillation strategy to train a domain-invariant albedo estimator. This ensures reliable structural preservation and consistent predictions even in lowvisibility nighttime conditions.

![](images/f7dead75728f2939ac5b17ea16f94356f82a69c90088f7eb23dd4f9cda0572ef.jpg)  
Fig. 3: CycleNet [27] failure case for Day-to-Night relighting. We show the input, and predictions at diferent epochs. CycleNet fails to ensure a proper trade-of between translation and structural consistency. At the early stage, the model attempts translation to the night domain but sufers from an identity collapse, where it copies the input image without any meaningful nighttime lighting efects.

## 4.1 HELIOS

We formulate our relighting task as an unpaired translation problem using the cycle-consistency training defined in Sec. 3.1.

We aim to learn a function $F _ { \theta }$ capable of mapping an image $z _ { 0 }$ from a source distribution to a target distribution. We define X and $\mathcal { V }$ as the daytime and nighttime driving domains, respectively.

The text conditioning $c _ { \mathrm { t e x t } } \in \{ c _ { x } , c _ { y } \}$ represents the target domain ( ‘day or ‘night’), while the structural conditioning $c _ { \mathrm { i m g } } \in \{ x _ { 0 } , y _ { 0 } \}$ provides the spatial context.

A fundamental limitation of cycle-consistency, as noted in ACLGAN [31] and CycleNet [27] (in Sec ${ 5 . 4 } )$ , is the need for the translated image y¯ to retain sufficient information to allow for a perfect reconstruction of the source $x _ { 0 }$ . When the model is conditioned on the input image $( c _ { \mathrm { i m g } } = x _ { 0 } )$ , this constraint becomes overly restrictive. It prevents a proper trade-of between structural consistency and domain translation. In the complex context of driving scenes this constraint often prevents the model from performing a significant lighting shift while maintaining the scene content. We demonstrate this failure case in Fig. 3. In early training phases, the model attempts translation but fails to preserve the underlying structure. As training converges, the model eventually collapses toward an identity mapping, where $F _ { \theta } ( x _ { t } , c _ { y } , x _ { 0 } ) \approx x _ { 0 }$ . This satisfies the reconstruction objective but fails to achieve any meaningful relighting.

## 4.2 Light invariant prior as conditioning

To address the constraints in standard cycle-consistent frameworks, we introduce an Albedo Estimator A that extracts the intrinsic reflectance $a = \mathcal { A } ( x _ { 0 } )$ from the source image. The albedo represents the lighting-invariant component of the image, which captures the base color without any lighting efects.

The extracted albedo is then used as a conditioning signal to replace the input image $c _ { i m g } .$ This prevents the model from relying on the conditioning image to satisfy the cycle-consistency objective. This is possible because the albedo a remains constant across domain translations of the same image, such that the reflectance of the source $x _ { 0 }$ is identical to its translated counterpart y¯. By leveraging this invariant prior, the optimization problem is simplified; the model no longer requires an explicit invariance loss to stabilize the target domain. Instead, the network learns to map the illumination dictated by the semantic conditioning $c _ { t e x t }$ onto the structural conditioning provided by a.

The final training objectives are redefined as:

$$
\mathcal { L } _ { \mathrm { r e c o } } = \mathbb { E } _ { \boldsymbol { x } , \boldsymbol { \epsilon } \sim \mathcal { N } ( 0 , I ) , t } \left[ \left\| \boldsymbol { x } _ { 0 } - F _ { \theta } ( \boldsymbol { x } _ { t } , \boldsymbol { c } _ { x } , a ) \right\| _ { 2 } ^ { 2 } \right]\tag{4}
$$

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { c y c l e } } = \mathbb { E } _ { \boldsymbol { x } , \epsilon \sim \mathcal { N } ( 0 , I ) , t } \left[ \big \| \boldsymbol { x } _ { 0 } - F _ { \theta } \big ( \boldsymbol { y } _ { t } , \boldsymbol { c } _ { x } , a \big ) \big \| _ { 2 } ^ { 2 } \right] . } \end{array}\tag{5}
$$

## 4.3 Continuous relighting

To resolve the ambiguity of text prompts that fail to capture subtle and gradual transitions of outdoor lighting, we replace textual prompts with a continuous conditioning mechanism based on the solar altitude angle $\theta _ { s }$ . For any given sample, $\theta _ { s }$ represents the elevation of the sun relative to the observer’s local horizon:

$$
\theta _ { s } = \arcsin ( \sin ( \phi ) \sin ( \delta ) + \cos ( \phi ) \cos ( \delta ) \cos ( h ) ) ,\tag{6}
$$

where $\phi$ is the geographic latitude, δ is the solar declination, and h is the local hour angle. These parameters are extracted from GPS metadata for each training sample.

Under this convention, $\theta _ { s } > 0 ^ { \circ }$ represents daytime conditions (sun above the horizon), while $\theta _ { s } < 0 ^ { \circ }$ corresponds to nighttime states. The scalar $\theta _ { s }$ is mapped into a high-dimensional feature space via positional encoding and injected into the cross-attention layers of the UNet, replacing the text prompt. This continuous formulation enables the model to learn smooth illumination states, allowing for fine-grained control over the global lighting and the synthesis of realistic transitions along the sun’s trajectory.

## 5 Robust Night-to-Day Translation via Albedo Distillation

While we can use any state-of-the-art intrinsic decomposition method as the albedo estimator A introduced in Sec 4.2, existing approaches such as DifusionRenderer [12] often fail under nighttime driving conditions. Specifically, in areas obscured by deep shadows, these estimators struggle to maintain consistency, resulting in incomplete albedo maps, as shown in Fig. 4. These degraded reflectance maps lead to structural failures during translation, where the model fails to recover geometry obscured in the nighttime input. We address this by proposing a synthetic distillation strategy to train a domain-invariant albedo estimator A that maintains structural consistency across all illumination states.

![](images/0dfe78c4ffda97fb7b67c5f86da8619676f6ff85f8a2facad354b658c1616d05.jpg)  
Fig. 4: Failure cases of DifusionRenderer [12] in nighttime illumination. While the baseline provides accurate albedo estimates for daytime images (left), it sufers from significant structural loss and color drifting under nocturnal conditions (right).

## 5.1 Cross-Domain Albedo Distillation

Based on the observation that intrinsic decomposition is significantly more stable for daytime images X compared to nighttime images $\mathcal { V } ,$ , we introduce a multistage process to distill daytime knowledge into a robust nighttime and daytime albedo estimator.

Synthetic Illumination Sampling. We use an initial Day-to-Night (D2N) HE-LIOS model to generate a multi-illumination dataset. For each daytime image $x _ { 0 }$ , we produce a sequence of relit variants $\{ \bar { y } _ { i } \} _ { i = 1 } ^ { 3 }$ and $\{ \bar { x } _ { i } \} _ { i = 1 } ^ { 2 }$ (three nighttime and two daytime). Since the scene geometry remains static across these variants, the daytime computed albedo a serves as a pseudo-ground-truth for all generated illumination.

Robust Albedo Estimator. We fine-tune an image-conditioned Stable Difusion model to map any image $x \in \{ \mathcal { X } \cup \bar { \mathcal { Y } } \}$ to its invariant albedo a. The estimator A is optimized via a latent-space denoising objective:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { a l b e d o } } = \mathbb { E } _ { a , \epsilon _ { a } \sim \mathcal { N } ( 0 , I ) , t } \left[ \left\| a _ { 0 } - \mathcal { A } ( a _ { t } , x ) \right\| _ { 2 } ^ { 2 } \right] , } \end{array}\tag{7}
$$

where $a _ { t }$ is the noisy latent of the albedo map. By training on both real daytime and generated nighttime images, A learns to disentangle illumination from the base color, providing a robust structural prior that is domain-invariant to both daytime and nighttime conditions.

## 5.2 Day-to-Night and Night-to-Day training

The frozen robust albedo estimator is then used to re-process our entire driving dataset to generate a consistent albedo set. These maps provide a stable, light-invariant structural condition for our final unified relighting framework. An overview of our training pipeline can be found in Fig. 2.

The final HELIOS model is trained using the unified objective for both Dayto-Night and Night-to-Day tasks:

$$
\begin{array} { r } { \mathcal { L } = \mathcal { L } _ { \mathrm { r e c o } } + \mathcal { L } _ { \mathrm { c y c l e } } , } \end{array}\tag{8}
$$

where the albedo a used as condition in Eq. 4 and 5 is now estimated with our robust albedo estimator introduced in Sec. 5.1. By conditioning on these distilled reflectance priors, the model efectively bypasses the ambiguities of nighttime scenes, allowing for high-fidelity Night-to-Day and Day-to-Night translations that preserve scene geometry while accurately synthesizing global illumination.

## 6 Experiments

## 6.1 Experiments details

Implementation details. We first perform a domain-specific fine-tuning on a pretrained Stable Difusion V2.1 [17] U-Net using driving images. This stage ensures that the model learns the distinct lighting distributions of both daytime and nighttime environments. Rather than using standard text-to-image supervision, we replace text-embedding conditioning with our continuous solar altitude angle $\theta _ { s } .$ . We fine-tune the model with a batch size of 4 and for 100k steps, using a fixed learning rate of $1 \times 1 0 ^ { - 5 }$ and the Adam optimizer. This initial training phase required approximately 24 hours. We then train ControlNet [30], initializing its weights from the fine-tuned U-Net. We keep the U-Net and VAE frozen. The model is trained with a batch size of 1 and for 300k steps using a fixed learning rate of $1 \times 1 0 ^ { - 5 }$ , requiring approximately four days of training. All experiments were conducted on a GPU equivalent to the NVIDIA RTX 4090. Images were resized, preserving the original aspect ratio with a minimum height or width of 512.

Datasets. We train HELIOS on a subset of three public driving datasets: nuScenes [3], Waymo [20] and Pandaset [25]. The datasets capture a wide range of temporal conditions, providing diverse solar altitude angles $\theta _ { s }$ (we refer to the supplementary materials for data distribution). We use a balanced distribution of daytime and nighttime sequences, for a total of 41k images. To train the albedo estimator, we use a separate subset of 6798 original images. For each of these images, we generate six diferent illumination versions, resulting in a multi-illumination synthetic dataset of approximately 40k images.

Baselines. We compare our method against text-guided editing models Instruct-Pix2Pix [2] and Qwen-Image [24]. We also evaluate against DifusionRenderer [12], a relighting framework trained on paired synthetic and augmented real-world data controlled via environment maps, and CycleNet [27], a cycle-consistent diffusion framework trained on unpaired data.

Metrics. To evaluate structural preservation, we compute DINO-struct (×100) [21] by measuring feature-space distances between the source and relighted images, and report mIoU between segmentation maps of source and relit images as a complementary measure of structural consistency. Image quality and distribution alignment with the target domain are evaluated with the FID score [6]. For semantic accuracy, we calculate the CLIP-Score [16] between generated outputs and text prompts corresponding to the target lighting condition. We additionally report downstream object detection performance as an image enhancement application, with results detailed in the applications section. Finally, we conduct a human perceptual study to quantify realism and preference. We report the Win rate, defined as the percentage of trials where a method was preferred by participants, and the Fail rate, representing the frequency at which outputs were rejected due to insuficient quality.

## 6.2 Qualitative results

We show in Fig. 5 and Fig. 7 qualitative results for Night-to-Day and Day-to-Night tasks, respectively. CycleNet consistently produces an identity image without performing domain translation. InstructPix2Pix, while struggling to achieve Night-to-Day relighting, produces global night lighting but omits local lighting efects. DifusionRenderer is limited by strictly keeping the albedo and just adding an environment map, which results in synthetic-looking outputs. Qwen-Image shows very impressive and realistic results for Night-to-Day; however, while it achieves good global lighting for night translation, it often loses structural consistency. In contrast, HELIOS provides consistent results across both tasks, maintaining structural integrity while successfully translating both global and local lighting efects.

Figure 8 compares our continuous conditioning over discrete text prompts. We refer to the supplementary material for videos that better illustrate this continuous, fine-grained control. Qwen-Image [24] generates exaggerated afternoon and sunset lightings, hallucinating an intense, localized light source. In contrast, our approach generates realistic ambient illumination that preserves the natural appearance of the image. Furthermore, our continuous solar elevation parameter $\theta _ { s }$ provides fine-grained control over low-light conditions. While discrete text models struggle to diferentiate between ambiguous dark states, our method clearly isolates the distinct photometric properties of twilight, night, and deep night.

<table><tr><td></td><td colspan="4">Night→Day</td><td colspan="4">Day→Night</td><td colspan="4">Average</td></tr><tr><td>Method</td><td>DINO ↓ FID ↓</td><td></td><td>CLIP ↑mIoU ↑</td><td></td><td>DINO ↓</td><td>FID ↓</td><td></td><td></td><td></td><td>CLIP ↑ mIoU ↑ DINO ↓ FID ↓ CLIP ↑ mIoU ↑</td><td></td><td></td></tr><tr><td>CycleNet [27]</td><td>0.15</td><td>147.53</td><td>0.21</td><td>0.35</td><td>0.25</td><td>145.00</td><td>0.23</td><td>0.51</td><td>0.20</td><td>146.27</td><td>0.22</td><td>0.43</td></tr><tr><td>IP2P [2]</td><td>0.11</td><td>140.11</td><td>0.22</td><td>0.41</td><td>0.15</td><td>129.40</td><td>0.26</td><td>0.60</td><td>0.13</td><td>134.76</td><td>0.24</td><td>0.51</td></tr><tr><td>Qwen-Image [24]</td><td>0.50</td><td>110.67</td><td>0.26</td><td>0.33</td><td>1.14</td><td>114.31</td><td>0.27</td><td>0.39</td><td>0.82</td><td>112.49</td><td>0.27</td><td>0.36</td></tr><tr><td>DiffusionRenderer [12]</td><td>1.57</td><td>191.42</td><td>0.25</td><td>0.17</td><td>1.34</td><td>143.50</td><td>0.24</td><td>0.11</td><td>1.46</td><td>167.46</td><td>0.25</td><td>0.14</td></tr><tr><td>HELIOS (ours)</td><td>0.63</td><td>74.65</td><td>0.27</td><td>0.28</td><td>0.50</td><td>100.72</td><td>0.27</td><td>0.43</td><td>0.57</td><td>87.69</td><td>0.27</td><td>0.36</td></tr></table>

Table 1: Quantitative results. HELIOS consistently outperforms all methods on FID and CLIP metrics for N→D and D→N translation, showing superiority in maintaining structural integrity while generating realistic lighting efects. CycleNet and IP2P often fail to perform significant relighting, their outputs are nearly identical to the source, which explains their best DINO and mIoU scores and very low FID and CLIP scores.

Input  
CycleNet [27]  
IP2P [2]  
Qwen-Img. [24] DifRendr. [12] HELIOS (ours)  
![](images/7f2ef9c2e2c6b113cfb4ee721e0666cd0d1ff9ef26b232179d2d8408a84714e6.jpg)  
Fig. 5: Qualitative comparison for Night-to-Day relighting. HELIOS demonstrates superior performance in generating daytime illuminations, efectively recovering structural details from nighttime inputs. While Qwen-Image achieves realistic relightings as it was trained on more data with a bigger model size, it still fails to respect structure and hallucinates added objects.

![](images/db588477bb1597b6dc4c63ef4cb42e5d72db4874008d3b152f558cc06c29564d.jpg)  
(a) Preference Rate ↑

![](images/83fb697bc69ebd17b2b37533e5c5e6e6a0db998f828886792432f4e5ffb04a6d.jpg)  
Fig. 6: User Study Results. We perform a user study to evaluate the performance of our method by (a) assessing the preference rate for daytime translation and (b) assessing the failure rate (non-realistic translation). HELIOS achieves the best results for Day→Night translation and overall translation. Qwen-Image, being 20 times larger and 100 times more costly to run, achieves slightly better Day→Night translation.

## 6.3 Quantitative results

We report in Tab 1 quantitative evaluation results for Night-to-Day, Day-to-Night and average on both tasks. Baseline methods such as CycleNet [27] and InstructPix2Pix [2] achieve the best DINO and mIoU scores; however, this is primarily due to their failure to perform significant domain translation, copying the reference image. This lack of meaningful relighting is reflected in their low FID and CLIP scores. DifusionRenderer [12] reports the highest DINO and FID scores and lower mIoU, reflecting a loss of structural integrity and a shift from the real data distribution. However, the model maintains a competitive CLIP score because the environment maps provide efective global lighting cues despite the geometric failures. While Qwen-Image [24] shows competitive results for Night-to-Day translation, it struggles with Day-to-Night tasks, failing to synthesize realistic nocturnal lighting while preserving scene structure. In contrast, our approach, HELIOS, achieves the best balance across both tasks. It maintains a reasonably low DINO score, indicating high structural fidelity, while significantly outperforming baselines in FID and CLIP-Score, demonstrating superior distribution alignment and semantic relighting accuracy.

<table><tr><td rowspan="2"></td><td colspan="4">N→D</td><td colspan="4">D→N</td></tr><tr><td>DINO↓</td><td>FID↓</td><td>CLIP↑</td><td>mIoU↑</td><td>DINO↓</td><td>FID↓</td><td>CLIP↑</td><td>mIoU↑</td></tr><tr><td>w/o distill.</td><td>0.74</td><td>141.75</td><td>0.20</td><td>0.21</td><td>0.63</td><td>100.77</td><td>0.26</td><td>0.34</td></tr><tr><td>Full HELIOS</td><td>0.63</td><td>74.65</td><td>0.27</td><td>0.28</td><td>0.50</td><td>100.72</td><td>0.27</td><td>0.43</td></tr></table>

Table 2: Ablation: impact of albedo distillation.

## 6.4 User Study

The user study evaluation involved 18 participants who performed 48 pairwise comparisons of images from a randomized set. For each trial, two randomly selected methods were compared against a ground-truth reference. If both methods failed to produce a realistic result, users had the option to select “both fail”, which contributed to the failure rate score (we refer the reader to the supplementary materials for more details).

We summarize the human perceptual study results for both Night-to-Day and Day-to-Night tasks in Fig. 6. The results reveal a significant performance asymmetry for Qwen-Image [24]. While Qwen-Image shows impressive results in Night-to-Day translation, its performance collapses in the Day-to-Night task, where its win rate drops to 27.6% and its failure rate doubles. Specifically, we observe that Qwen-Image frequently sufers from semantic hallucinations, such as incorrectly inserting non-existent vehicles or failing to maintain structural consistency. These failure cases are shown in the supplementary material. In Night-to-Day translation, InstructPix2Pix [2] generally fails to produce a meaningful change. However, it performs better in Day-to-Night scenarios. This is because IP2P can adjust the global ambient lighting of a scene, but it fails to synthesize the local light sources. CycleNet [27] consistently reports high failure rates because it cannot translate any visible lighting efect. In contrast, HELIOS provides consistent and reliable results across both translation tasks. Averaged across both tasks, our method achieves a higher win rate (58.8%) and the lowest overall failure rate (19.7%), significantly outperforming Qwen-Image’s average of 48.9% and 26.0% respectively. These results confirm that HELIOS produces more stable relighting results.

## 6.5 Ablation study

We evaluate our robust albedo estimator in Fig. 9 and Tab. 2, comparing it against a baseline using standard pre-computed albedo maps from DifusionRender [12]. The results indicate that while basic albedo decomposition is suficient for Day-to-Night (D→N) translation, it performs poorly in Night-to-Day (N→D) scenarios. Standard estimators often fail to recover geometry in regions obscured by deep shadows or low nighttime visibility, leading to structural artifacts in the relit output. In contrast, our distilled cross-domain albedo estimator provides a more reliable structural prior. By leveraging knowledge from the daytime domain, the model recovers accurate scene geometry and synthesizes realistic daytime illumination even from degraded nighttime inputs. Furthermore, for D→N translation, our distillation strategy produces significantly sharper albedo maps. This increased high-frequency detail directly translates to sharper relighting results.

Input  
CycleNet [27]  
IP2P [2]  
Qwen-Img [24]  
DifRendr [12]  
![](images/75f28e5648591ef9fdddc4f4930249e7668981fbb127513c64080f0016696fbb.jpg)  
HELIOS (ours)

Fig. 7: Qualitative comparison for Day-to-Night relighting. HELIOS demonstrates superior performance in generating realistic nighttime global and local illuminations while keeping the structure of the input image.  
![](images/44b1af91d09bfeae5f4644c77accf4c3636c40cd970f9180b95af1116b4c8420.jpg)  
Fig. 8: Continuous Relighting. Our continuous solar altitude parameter (θ ) enables fine-grained, physically plausible transitions of global illumination.

## 7 Discussion

## 7.1 Applications

Color editing. HELIOS enables color editing for data augmentation. As shown in Fig. 10, modifying the albedo color within a mask and relighting it using HE-LIOS produces realistic scene edits.

Image enhancement. We evaluate the robustness of our method for perception tasks on the ACDC dataset [18]. Using YOLOv11 [10] to report mAP on night

Input  
w/o Albedo Distillation  
![](images/d346206ffd23bcde83fdf186610b53254724a80f1dbe5931af82b754a25b03d5.jpg)  
Full model

Fig. 9: Ablation Study. We compare our robust albedo estimator against a standard albedo estimator. For N →D our albedo maintains accurate structures and colors in dark areas, providing a stable conditioning for high-fidelity relighting. The albedo computed with our robust estimator predicts sharper details for D→N, which directly results in sharper relighting outputs.  
![](images/f747a7756b17de6173e90af53e0d470ccc99d6e067e40d2a60c316848b3c1011.jpg)  
Fig. 10: Application. HELIOS enables color and lighting editing for data augmentation.

scenes, we find that relighting to daytime improves detection performance by 5.3% for mAP@70 (Tab. 12). This confirms that HELIOS efectively recovers details that improve object detection in night conditions.

![](images/1d879e146f1df01fa826ae4b601b7049921731e99fcdafc03bb9c7fa13471c04.jpg)  
Input image

![](images/3742ea5dcdf8c794f17f932a54248351a2034705dbfe90a0764275a65499cd20.jpg)  
Night→Day (ours)

Fig. 11: Downstream Object Detection Performance on ACDC dataset [18].
<table><tr><td>Method</td><td>mAP@50</td><td>mAP@75</td></tr><tr><td>Night (input)</td><td>6.0</td><td>2.7</td></tr><tr><td>CycleNet [27]</td><td>5.0</td><td>2.7</td></tr><tr><td>IP2P [2]</td><td>6.1</td><td>2.8</td></tr><tr><td>Qwen-Image [24]</td><td>4.1</td><td>2.3</td></tr><tr><td>DiffusionRenderer [12]</td><td>fail</td><td>fail</td></tr><tr><td>HELIOS (ours)</td><td>10.0</td><td>8.0</td></tr></table>

Night baseline: mAP@50=6.0, mAP@75=2.7  
Fig. 12: Comparison of YOLOv11 performance on night images and relit results.

## 7.2 Limitations

While HELIOS is robust to various outdoor lighting conditions, it remains limited by extremely low-light nocturnal environments where predicting accurate albedo-like images and relighting results becomes challenging (we refer the reader to the supplementary materials).

## 7.3 Conclusion

In this work, we presented HELIOS, a method for unpaired image relighting in real-world driving scenarios. Trained without paired multi-illumination data or synthetic supervision, we introduced an albedo-based conditioning into a cycle-consistent difusion pipeline to prevent identity collapse. We addressed the challenge of low-visibility nighttime conditions through an albedo distillation strategy that ensures structural preservation across both daytime and nighttime domains. Furthermore, we replaced coarse text prompts with a finer continuous control mechanism based on GPS-derived solar angles, enabling precise and smooth lighting manipulation. Quantitative and qualitative evaluations demonstrate that HELIOS outperforms state-of-the-art methods in preserving scene structure and generating realistic lighting.

![](images/ce28055509421513449403a1f4e976d792acf463de668e2458f86ba08509abb5.jpg)  
(a) Pandaset

![](images/bb9ceddd972ec8681be181400f3a52bbbc35c5b6d568e86995adf6cba5cfb4cf.jpg)  
(b) nuScenes

![](images/4f0493065367bfa0cb31646d22b73e56965cbf39a7a24f5b8cf549a5fa2c9f32.jpg)  
(c) Waymo Open Dataset  
Fig. 13: Solar altitude distribution across training datasets. We plot the density of samples relative to the sun’s elevation.

## 8 Dataset Distribution and Illumination Analysis

Fig. 13 illustrates the solar altitude distributions for Pandaset [25], nuScenes [3], and Waymo [20]. We observe that:

– Pandaset: Solar altitudes are highly clustered with sharp peaks at specific angles. This indicates that the data was recorded during a small number of driving sessions at fixed times of day.

nuScenes: This dataset shows a wider range of daytime coverage, specifically between $2 0 ^ { \circ }$ and $6 5 ^ { \circ }$ . This variation is due to data collection in diverse locations, such as Singapore and Boston, which have diferent solar paths.

– Waymo Open Dataset: Waymo has the most continuous distribution, covering angles from deep night $( - 4 0 ^ { \circ } )$ to high noon $( + 7 0 ^ { \circ } )$ . The smoother density curve shows that data was captured over a much larger number of sessions and a broader timeframe.

By training on these diverse sequences and sun elevations, HELIOS provides continuous relighting control for outdoor driving images. We refer to the video provided in the supplementary material for examples of controllable, smooth, and continuous relighting.

## 9 User study details

We compared our model against four state-of-the-art baselines across the two translation tasks: Day-to-Night (D2N) and Night-to-Day (N2D).

## 9.1 Experimental Setup

The study was implemented as a web-based interface. Each of the 18 participants performed a total of 48 randomized pairwise comparisons (24 rounds for each task). In each trial, a reference image was displayed alongside two randomly selected results from diferent methods.

To eliminate experimental bias, we ensured a randomized protocol:

Image Selection: The reference image was randomly selected from a set of 1000 images.

– Method Pairing: For every comparison, we randomly paired two of the five methods, ensuring that over the course of the study, all methods were compared against one another in a statistically balanced manner.

– The side-by-side positioning (Left vs. Right) of the methods was randomized to prevent "side-bias" during the voting process.

Each trial presented the following instruction:

Which of the following options represents the most realistic [Night/Day] version of the reference image while preserving its original structure?

A “Both Failed” option was provided to identify cases where neither result was satisfactory, usually due to identity loss or failed relighting.

## 9.2 Evaluation Metrics

We define two primary metrics to assess the model’s performance:

– Win Rate (%): This represents the preference rate. It is the percentage of trials in which a specific method’s output was chosen as superior to its competitor. A higher win rate indicates greater visual realism and better preservation of the scene’s identity.

– Fail Rate (%): It reflects the proportion of failure case of the model, where the translation task cannot be done successfully either by producing an image with too many unrealistic artifacts or by not following the relighting instruction.

## 10 Additional results

We provide additional relighting results in Fig. 14 and 15, for the two translation tasks. The results demonstrate the robustness of HELIOS. As shown, the model consistently preserves scene geometry while accurately modulating global illumination according to the target solar altitude.

Input  
CycleNet [27]  
IP2P [2]  
Qwen-Img [24]  
DifRendr [12]  
![](images/8f5f67142e174761170959d63956a070adebf658a79519d0da3aa6855374ce6f.jpg)  
HELIOS (ours)  
Fig. 14: We qualitatively compare the re-lit images. We show that HELIOS predicts realistic Night-to-Day relighting.

Input  
CycleNet [27]  
IP2P [2]  
Qwen-Img [24]  
DifRendr [12]  
HELIOS (ours)  
![](images/c59341a1448ca6aebc95acb71a63c05aeda7c39138b3312863b5856605e58d3e.jpg)  
Fig. 15: We qualitatively compare the re-lit images. We show that HELIOS predicts realistic Day-to-Night relighting.

## 10.1 Qwen-Image failures

We show in Fig. 16 an example of a failure case of Qwen-Image [24]. The baseline often exhibits modal instability due to its multi-task training. It occasionally confuses the relighting instruction with other tasks.

## 10.2 Limitation

Fig. 17 shows a failure case for HELIOS. In images with extreme nighttime lighting, the model fails to recover structural details. This limitation could be addressed by fine-tuning a more robust albedo estimator trained on a wider distribution of extremely low-light data to better capture structural details.

![](images/c24c29178b28dfdfe2fcb0a4e11c2905f015d806282d3eeaa396a3a88e33a049.jpg)  
Fig. 16: Failure of Qwen-Image [24]

![](images/43b130b420052bec70f9fa00b0b363a755e3a8e27ed91cbe8e5b8f6231edf0f7.jpg)  
Fig. 17: Failure case of HELIOS for Night-to-Day relighting task.

## References

1. Barrow, H.G., Tenenbaum, J.M.: Recovering intrinsic scene characteristics from images. In: Computer Vision Systems. Academic Press (1978) 3

2. Brooks, T., Holynski, A., Efros, A.A.: InstructPix2Pix: Learning to follow image editing instructions. In: CVPR (2023) 9, 10, 11, 12, 13, 14, 3

3. Caesar, H., Bankiti, V., Lang, A.H., Vora, S., Liong, V.E., Xu, Q., Krishnan, A., Pan, Y., Baldan, G., Beijbom, O.: nuscenes: A multimodal dataset for autonomous driving. In: CVPR (2020) 9, 1

4. Gao, J., Gu, C., Lin, Y., Zhu, H., Cao, X., Zhang, L., Yao, Y.: Relightable 3d gaussians: Realistic point cloud relighting with brdf decomposition and ray tracing. In: ECCV (2023) 3

5. He, K., Liang, R., Munkberg, J., Hasselgren, J., Vijaykumar, N., Keller, A., Fidler, S., Gilitschenski, I., Gojcic, Z., Wang, Z.: Unirelight: Learning joint decomposition and synthesis for video relighting. arXiv:2506.15673 (2025) 2, 3, 4

6. Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B., Hochreiter, S.: Gans trained by a two time-scale update rule converge to a local nash equilibrium. In: NeurIPS (2017) 9

7. Jin, H., Li, Y., Luan, F., Xiangli, Y., Bi, S., Zhang, K., Xu, Z., Sun, J., Snavely, N.: Neural gafer: Relighting any object via difusion. In: NeurIPS (2024) 3

8. Kanamori, Y., Endo, Y.: Relighting humans: Occlusion-aware inverse rendering for full-body human images. ACM Transactions on Graphics (SIGGRAPH Asia) (2018) 3

9. Kerbl, B., Kopanas, G., Leimkuehler, T., Drettakis, G.: 3d gaussian splatting for real-time radiance field rendering. ACM Transactions on Graphics (TOG) (2023) 3

10. Khanam, R., Hussain, M.: Yolov11: An overview of the key architectural enhancements. arXiv:2410.17725 (2024) 13

11. Kocsis, P., Sitzmann, V., Nießner, M.: Intrinsic image difusion for indoor singleview material estimation. In: CVPR (2024) 3

12. Liang, R., Gojcic, Z., Ling, H., Munkberg, J., Hasselgren, J., Lin, Z.H., Gao, J., Keller, A., Vijaykumar, N., Fidler, S., Wang, Z.: Difusionrenderer: Neural inverse and forward rendering with video difusion models. In: CVPR (2025) 2, 3, 4, 7, 8, 9, 10, 11, 12, 13, 14

13. Liang, Z., Zhang, Q., Feng, Y., Shan, Y., Jia, K.: Gs-ir: 3d gaussian splatting for inverse rendering. In: CVPR (2024) 3

14. Lin, Z., Liu, B., Chen, Y.T., Forsyth, D.A., Huang, J.B., Bhattad, A., Wang, S.: Urbanir: Large-scale urban scene inverse rendering from a single video. In: 3DV (2025) 3

15. Mildenhall, B., Srinivasan, P.P., Tancik, M., Barron, J.T., Ramamoorthi, R., Ng, R.: Nerf: Representing scenes as neural radiance fields for view synthesis. In: ECCV (2020) 3

16. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., Sutskever, I.: Learning transferable visual models from natural language supervision. In: ICML (2021) 9

17. Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B.: High-resolution image synthesis with latent difusion models. In: CVPR (2022) 9

18. Sakaridis, C., Dai, D., Van Gool, L.: ACDC: The adverse conditions dataset with correspondences for semantic driving scene understanding. In: ICCV (2021) 13, 14

19. Srinivasan, P.P., Deng, B., Zhang, X., Tancik, M., Mildenhall, B., Barron, J.T.: Nerv: Neural reflectance and visibility fields for relighting and view synthesis. In: CVPR (2021) 3

20. Sun, P., Kretzschmar, H., Dotiwalla, X., Chouard, A., Patnaik, V., Tsui, P., Guo, J., Zhou, Y., Chai, Y., Caine, B., Vasudevan, V., Han, W., Ngiam, J., Zhao, H., Timofeev, A., Ettinger, S.M., Krivokon, M., Gao, A., Joshi, A., Zhang, Y., Shlens, J., Chen, Z., Anguelov, D.: Scalability in perception for autonomous driving: Waymo open dataset. In: CVPR (2020) 9, 1

21. Tumanyan, N., Bar-Tal, O., Bagon, S., Dekel, T.: Splicing vit features for semantic appearance transfer. In: CVPR (2022) 9

22. Wang, Z., Philion, J., Fidler, S., Kautz, J.: Learning indoor inverse rendering with 3d spatially-varying lighting. In: ICCV (2021) 2, 3

23. Wang, Z., Shen, T., Gao, J., Huang, S.Y., Munkberg, J., Hasselgren, J., Gojcic, Z., Chen, W., Fidler, S.: Neural fields meet explicit geometric representations for inverse rendering of urban scenes. In: CVPR (2023) 3

24. Wu, C., Li, J., Zhou, J., Lin, J., Gao, K., Yan, K., ming Yin, S., Bai, S., Xu, X., Chen, Y., Chen, Y., Tang, Z., Zhang, Z., Wang, Z., Yang, A., Yu, B., Cheng, C., Liu, D., Li, D., Zhang, H., Meng, H., Wei, H., Ni, J., Chen, K., Cao, K., Peng, L., Qu, L., Wu, M., Wang, P., Yu, S., Wen, T., Feng, W., Xu, X., Wang, Y., Zhang, Y., Zhu, Y., Wu, Y., Cai, Y., Liu, Z.: Qwen-image technical report. arXiv:2506.15673 (2025) 9, 10, 11, 12, 13, 14, 3, 4

25. Xiao, P., Shao, Z., Hao, S., Zhang, Z., Chai, X., Jiao, J., Li, Z., Wu, J., Sun, K., Jiang, K., Wang, Y., Yang, D.: Pandaset: Advanced sensor suite dataset for autonomous driving. In: ITSC (2021) 9, 1

26. Xing, X., Groh, K., Karaoglu, S., Gevers, T., Bhattad, A.: Luminet: Latent intrinsics meets difusion models for indoor scene relighting. In: CVPR (2025) 3

27. Xu, S., Ma, Z., Huang, Y., Lee, H., Chai, J.: Cyclenet: Rethinking cycle consistency in text-guided difusion for image manipulation. In: NeurIPS (2023) 2, 4, 6, 9, 10, 11, 12, 13, 14, 3

28. Zeng, C., Dong, Y., Peers, P., Kong, Y., Wu, H., Tong, X.: Dilightnet: Fine-grained lighting control for difusion-based image generation. ACM Transactions on Graphics (SIGGRAPH) (2024) 3

29. Zeng, Z., Deschaintre, V., Georgiev, I., Hold-Geofroy, Y., Hu, Y., Luan, F., Yan, L.Q., Havs.an, M.: RGB↔X: Image decomposition and synthesis using materialand lighting-aware difusion models. ACM Transactions on Graphics (SIGGRAPH) (2024) 3

30. Zhang, L., Rao, A., Agrawala, M.: Adding conditional control to text-to-image difusion models. In: ICCV (2023) 4, 9

31. Zhao, Y., Wu, R., Dong, H.: Unpaired image-to-image translation using adversarial consistency loss. In: ECCV (2020) 6

32. Zheng, Y., Chai, M., Vicini, D., Zhou, Y., Xu, Y., Guibas, L.J., Wetzstein, G., Beeler, T.: Groomlight: Hybrid inverse rendering for relightable human hair appearance modeling. In: CVPR (2025) 3

33. Zhu, J., Huo, Y., Ye, Q., Luan, F., Li, J., Xi, D., Wang, L., Tang, R., Hua, W., Bao, H., Wang, R.: I2-sdf: Intrinsic indoor scene reconstruction and editing via raytracing in neural sdfs. In: CVPR (2023) 2, 3

34. Zhu, J.Y., Park, T., Isola, P., Efros, A.A.: Unpaired image-to-image translation using cycle-consistent adversarial networks. In: ICCV (2017) 4