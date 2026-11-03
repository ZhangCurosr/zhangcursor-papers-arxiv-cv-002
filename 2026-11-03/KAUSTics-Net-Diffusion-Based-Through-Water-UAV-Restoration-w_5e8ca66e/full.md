# KAUSTics-Net: Diffusion-Based Through-Water UAV Restoration with a Semi-Synthetic Dataset

Dario Scilla<sup>1</sup>, Omar A. Lopez<sup>1</sup>, Kasper Johansen<sup>1</sup>, Brian O. Nieuwenhuis<sup>2</sup>, and Matthew F. McCabe<sup>1</sup>

Abstract—Accurate mapping of shallow-water environments using unmanned aerial vehicle (UAV) imagery is challenged by water-induced distortions, including refraction, caustics, and wavelength-dependent attenuation. The scarcity of paired distorted and reference imagery further limits the development of supervised restoration methods. To address this challenge, we introduce KAUSTics, a semi-synthetic dataset combining 1,280 reference patches derived from 32 high-resolution UAV images with 12,800 degraded observations generated through Blender-based water-surface rendering and empirical color and blur augmentations. We further propose KAUSTics-Net, a pixelspace conditional diffusion framework integrating a multi-scale conditional encoder, feature-wise linear modulation (FiLM), cross-attention, and a composite reconstruction objective to restore shallow-water imagery. We evaluated the framework against eight image-restoration methods, complemented by architectural ablations and cross-dataset experiments. On KAUS-Tics, KAUSTics-Net achieved a mean peak signal-to-noise ratio (PSNR) of 31.715 ± 0.439 dB, structural similarity index (SSIM) of 0.849 ± 0.002, and learned perceptual image patch similarity (LPIPS) of 0.145 ± 0.007 across three training seeds. On the independent Sea-Undistort dataset, sampler-aware fine-tuning improved restoration fidelity, reaching 19.469 ± 0.332 dB PSNR and 0.554 ± 0.005 SSIM under combined water-surface distortions, although perceptual trade-offs remained. Additional experiments on real UAV imagery examined feature preservation and segmentation behavior. These findings demonstrate the potential of semi-synthetic training and conditional diffusion for shallowwater image restoration while highlighting the importance of evaluating both reconstruction fidelity and downstream remotesensing utility.

Index Terms—Caustics, Conditional Diffusion, Image Restoration, Unmanned Aerial Vehicle (UAV), Remote Sensing, Shallow Water.

## I. INTRODUCTION

coral reefs, seagrass meadows, mangroves, and tidal flats, while also hosting substantial populations and infrastructure. Their monitoring is essential for ecosystem assessment, coastal management, restoration, and climate adaptation [1], [2], [3], [4]. Although in situ surveys provide detailed observations, they are spatially limited, logistically demanding, and often unable to capture rapid changes in dynamic coastal environments [5], [6], [7]. Satellite and unmanned aerial vehicle (UAV) remote sensing provide broader and more frequent coverage, enabling shallow-water mapping at scales impractical for field surveys alone [8], [9], [10].

Extracting reliable information from optical imagery over shallow coastal waters nevertheless remains challenging. Surface waves focus and defocus sunlight, producing dynamic, high-frequency caustic patterns unrelated to actual seabed variability [11], [12]. Specular reflection at the air–water interface generates sun glint that can saturate parts of the image and obscure the underlying scene [13], [14]. Refraction displaces apparent object locations and distorts submerged shapes, while wavelength-dependent absorption and scattering alter the signal returning from the seafloor. Together, these effects degrade contrast and color, distort geometry, and complicate applications such as benthic habitat mapping [15], bathymetry [16], and photogrammetry [17].

Several approaches have been proposed to mitigate waterinduced distortions. Physics-based methods invert simplified radiative-transfer models but often require estimates of water depth, optical properties, and atmospheric conditions [18], [19], [20]. Image-processing approaches apply deglinting, dehazing, contrast enhancement, or multi-view corrections [21], [16], [12]. More recently, deep learning has enabled datadriven mappings between distorted and reference imagery. GAN-based methods have been applied to underwater enhancement and coastal water removal [22], [13], [23], while diffusion-based image-to-image models have demonstrated capabilities for removing complex degradations while preserving scene structure [24], [25].

However, learning-based restoration of shallow-water UAV imagery is constrained by the scarcity of large, high-resolution paired datasets containing observations of the same scene with and without water-surface distortions [23], [26]. Repeated acquisitions under different sea states and illumination rarely provide aligned pairs and make it difficult to separate watersurface effects from genuine scene changes [10]. Purely synthetic data, in contrast, may not capture the spatial complexity and variability of real coastal environments. This motivates semi-synthetic approaches that combine real UAV imagery with physically based simulations of water-induced distortions.

To address this gap, we introduce KAUSTics, a semisynthetic dataset for high-resolution UAV water-distortion removal, together with KAUSTics-Net, a conditional diffusionbased restoration framework. We evaluated the proposed approach on KAUSTics and the independent Sea-Undistort benchmark [26], complemented by an exploratory assessment on real UAV imagery.

In summary, this work makes the following contributions:

• We introduce KAUSTics, a semi-synthetic dataset comprising 1,280 high-resolution UAV reference patches and 12,800 degraded observations generated through physically based water-surface rendering and empirical color and blur augmentations.

• We propose KAUSTics-Net, a pixel-space conditional diffusion framework combining multi-scale conditioning, FiLM-modulated residual blocks, cross-attention, and a composite restoration objective for through-water UAV imagery.

• We evaluate KAUSTics-Net against eight restoration methods using source-image-disjoint partitions, architectural ablations, cross-dataset experiments, and real-UAV assessments of downstream image-processing utility.

## II. RELATED WORK

Image restoration has been extensively studied across computer vision and remote sensing, including dehazing, deblurring, deraining, and underwater image enhancement. However, methods explicitly targeting water-surface distortions such as caustics, sun glint, and refraction in high-resolution UAV imagery remain comparatively limited. In practice, many approaches applied to this problem were originally developed for generic image restoration or underwater cameras rather than through-water aerial observations [27], [23], [28]. Existing approaches can be broadly grouped into statistical and physicsbased corrections, generative adversarial networks (GANs), and diffusion-based restoration models.

## A. Statistical and Physics-Based Approaches

In UAV-based shallow-water imaging, refraction at the air– water interface displaces apparent object locations and distorts submerged shapes, while wave-induced caustics and sun glint introduce strong spatially varying radiometric artifacts [29], [12]. Early correction methods primarily focused on geometric effects by applying empirical refraction factors derived from Snell’s law. Partama et al. proposed a correction of apparent depths from UAV structure-from-motion reconstructions, while related studies incorporated refraction correction into shallow-water bathymetric inversion schemes [30], [31]. Reviews of UAV-based shallow-water bathymetry have highlighted the effectiveness of these approaches, but also their dependence on acquisition geometry, water depth, seabed properties, and assumptions regarding the water surface [32], [33].

A complementary family of methods exploits temporal redundancy in high-frame-rate imagery. Fluid lensing uses repeated observations of the same scene to model the dynamic water surface as a time-varying optical lens and reconstruct refraction-corrected, caustic-reduced imagery [34], [35]. More recently, Scilla et al. showed that frame-wise temporal aggregation combined with color-transfer correction can reduce caustics, refraction-induced shape variations, and sun glint in UAV video while improving the consistency of submerged features [12].

Although effective, these approaches generally rely on explicit physical assumptions, hand-crafted corrections, or repeated observations of the same scene. They therefore do not directly learn a single-image mapping from a distorted observation to a clean seabed representation, motivating the use of data-driven generative models.

## B. Generative Adversarial Networks

Generative adversarial networks learn image distributions through competition between a generator and discriminator [36], and have been widely adopted for image enhancement and restoration. In image-to-image translation, CycleGAN introduced cycle consistency for learning mappings between unpaired image domains [37], a particularly useful property when paired distorted and clean observations are difficult to acquire.

Underwater enhancement models such as UGAN, FUnIE-GAN, MLFcGAN, and UWGAN use adversarial learning to improve color balance, contrast, local structure, and, in some cases, downstream robotic perception [27], [38], [39], [40]. These approaches have demonstrated substantial improvements for imagery acquired by underwater cameras. However, their target domain differs from through-water UAV imagery, where wave-induced refraction, dynamic caustics, and sun glint introduce stronger spatially varying geometric and radiometric distortions.

Only a limited number of GAN-based methods have directly addressed aerial water removal. DeWaterGAN is a physicsguided unsupervised framework for UAV coastal imagery that combines a CycleGAN backbone with a light-transportinspired attention mechanism [23]. The model can reduce water interference while preserving color and texture, but it primarily targets appearance restoration and does not rely on pixel-aligned distorted/undistorted training pairs.

Overall, GAN-based methods demonstrate the potential of learned restoration, but their application to through-water UAV imagery remains challenging, as these images belong to substantially different domains, containing stronger refraction induced distortions, and complex caustic patterns that are not fully captured in classical underwater datasets.

## C. Diffusion and Advanced Restoration Models

Denoising diffusion probabilistic models progressively corrupt training data with Gaussian noise and learn the reverse denoising process [41]. Latent diffusion models, including Stable Diffusion, further reduce computational cost by performing this process in a compressed representation space [42]. Conditional diffusion models extend this framework to imageto-image restoration by incorporating degraded observations into the denoising process. For instance, Palette introduced a general conditional diffusion framework for image-to-image translation [43].

Several diffusion-based approaches have been developed for underwater image enhancement. UW-DDPM introduced a dual-U-Net formulation with distribution-fusion sampling [44], while DiffWater incorporated degraded-image and colorcompensated guidance to improve underwater image restoration [28]. Related studies have investigated accelerated sampling strategies and alternative conditioning mechanisms [45], [46]. Beyond underwater imagery, NDR-Restore learns degradation representations to address multiple corruption types [47], while ResShift formulates diffusion in the residual space between degraded and reference images to reduce the number of sampling steps [48].

Alternative deep learning architectures have also been developed for underwater image enhancement. SS-UIE combines spatial selective scanning and spectral attention to address nonuniform image degradation [49], while U-Shape Transformer integrates multi-scale channel-wise feature fusion and spatial global modeling [50]. These methods provide complementary restoration strategies beyond conventional convolutional and diffusion-based frameworks.

More recently, the Sea-Undistort dataset introduced a paired synthetic benchmark for through-water airborne imagery and evaluated restoration models under sun glint, scattering, and wave-induced distortions [26]. These results demonstrate the potential of learning-based restoration for aerial through-water observations. However, most existing restoration methods were developed for underwater-camera imagery or general image degradation rather than through-water UAV observations. Moreover, the availability of large paired datasets preserving the texture and spatial complexity of real coastal UAV imagery remains limited.

The present work addresses this gap by introducing KAUS-Tics, a semi-synthetic dataset combining real high-resolution UAV imagery with rendered water-surface distortions and empirical appearance augmentations, together with KAUSTics-Net, a conditional diffusion framework tailored to this setting. This combination enables supervised restoration toward calmwater reference imagery and controlled evaluation under multiple caustic patterns, blur levels, and color transformations.

## III. METHODOLOGY

## A. KAUSTics Dataset

The KAUSTics dataset builds on UAV surveys conducted in June 2024 between 8 and 9 AM at the King Abdullah University of Science and Technology (KAUST), approximately 80 km north of Jeddah on the Red Sea coast of Saudi Arabia (22.34<sup>◦</sup>N, 39.09<sup>◦</sup>E). The surveys were originally collected for coastal monitoring applications, including habitat mapping. For this study, we selected 32 images acquired over a shallowwater lagoon (depth < 3 m; Fig. 1) under clear-sky and lowwind conditions to minimize visible water-surface distortions.

The images were acquired using a DJI Matrice 300 RTK flying at 35 m altitude and equipped with a Zenmuse P1 45- MP full-frame camera with a 35 mm lens. Each frame had a resolution of $8 1 9 2 \times 5 4 6 0$ pixels and a ground sampling distance of approximately $0 . 4 ~ c m / p i x e l$ . The surveyed area included sandy, rocky, seagrass, and macroalgal substrates. The selected images represent different seabed textures and visual characteristics observed in the lagoon and serve as calm-water reference targets. Their geometric and radiometric accuracy has not been independently established.

Synthetic surface distortions were generated using Blender [51] by mapping the UAV-derived reference imagery onto the bottom plane of a virtual shallow-water scene and rendering it through a simulated water surface. The rendering process introduced wave-induced illumination focusing (caustics) and refractive displacement. Sun glint was not independently parameterized or controlled in the KAUSTics rendering process. The virtual camera was configured to provide a reproducible nadir-viewing geometry at an altitude of 10 m, with a sensor size of $1 3 . 2 \times 8 . 8$ mm, a focal length of 6.201 mm, and a rendering resolution of 1000×1000 pixels. The rendering setup is illustrated in Fig. 2.

![](images/ed8ea635b6f0a0dc1cf595547c6dd28af58f09a9b3f2e91883bb8935feef2c93.jpg)  
Fig. 1. Area of interest for the KAUSTics dataset. The background shows a Sentinel-2 RGB composite of the coastal waters near Thuwal, Saudi Arabia, in the central Red Sea. The orange lines indicate the approximate area of interest of the UAV surveys used to acquire the calm-water reference imagery over shallow reef flats and sandy substrates.

![](images/00897adf17d093d34c5fd9acb5b18355be7ae079558c5f9b3fa3fdf3d49e750c.jpg)  
(a)

![](images/c42d81030297c7503524cb7c1f92d8bac95c99ecb8ddcbceac71808bc505a347.jpg)  
(b)  
Fig. 2. Rendering setup used to generate the synthetic component of KAUSTics. (a) Blender scene showing the UAV-derived seafloor texture mapped onto the bottom plane and viewed by a nadir-looking camera. (b) Mesh of the simulated water surface generated using the Ocean Modifier, with a central zoom showing its spatial sampling density.

Water-surface geometry was simulated using Blender’s Ocean Modifier with the Shallow Water spectrum, a depth parameter of 0.4 m, Spatial Size of 49 m, wave-amplitude Scale of 0.025, and wind velocity of 5 m $\mathbf { s } ^ { - 1 }$ . The geometry Size parameter was set to 0.190, with a water-object scale of approximately 1.665. Two distinct caustic patterns were generated by varying the surface alignment parameters: Shape 1 used an alignment of 0, producing predominantly celllike caustics, whereas Shape 2 used an alignment of 0.1, a direction of 90<sup>◦</sup>, and damping of 0.5, producing more parallel structures (Fig. 3). Surface realizations were varied through the Ocean Modifier’s random seed. All scenes were rendered using Blender Cycles with OptiX acceleration on an NVIDIA A100 GPU (80 GB). These parameters define the synthetic scene configuration and do not represent measured hydrodynamic or optical properties of the lagoon.

![](images/7dfdc91992e5ad6731556578533a4da6c56d6e022e63d00b7691b04755a7bda0.jpg)  
Fig. 3. Overview of the KAUSTics semi-synthetic dataset. UAV imagery is divided into 1000 × 1000 pixel reference patches and rendered in Blender to generate two caustic patterns. Gaussian blur and three empirical color-transfer configurations (Color 1–3) expand the dataset to ten degraded observations per reference patch.

The resulting dataset follows the hierarchical organization illustrated in Fig. 3. Starting from the rendered caustic patterns, Gaussian-blurred variants were generated to approximate reduced visibility and optical defocus. Three additional color transformations (Color 1–3) were applied using statistical color transfer in CIELAB space [52], matching channel means and standard deviations to manually selected target appearances. These transformations introduce variability in spectral appearance and serve as empirical proxies for absorption- and turbidity-like effects. They do not constitute calibrated simulations of water-column optical properties or measured depth-dependent attenuation.

Each of the 32 UAV images contributed 40 reference patches of 1000 × 1000 pixels, resulting in 1,280 reference patches. Each reference was associated with ten degraded observations: two caustic patterns, two Gaussian-blurred variants, and six color-transfer variants corresponding to three color configurations applied to both caustic patterns. The complete dataset therefore comprises 12,800 distorted–reference pairs. These augmentations increase the diversity of simulated degradation appearances without increasing the number of independent source images. The source-image-disjoint training, validation, and test partitioning protocol is described in

Section III-G.

Although each degraded observation is associated with its corresponding reference patch, wave-induced refraction can introduce local geometric displacement, preventing exact feature-wise pixel correspondence. The calm-water references were acquired through water and may retain residual refraction and radiometric attenuation. No independent water-free seabed images, surveyed geometric references, or calibrated reflectance measurements are available. Consequently, KAUS-Tics supports supervised restoration toward a low-distortion, calm-water reference appearance. Additional dataset visualizations and rendering specifications are provided in the supplementary material.

## B. Sea-Undistort Dataset

To assess the generalization of KAUSTics-Net beyond the KAUSTics dataset, we also evaluated it on Sea-Undistort [26], a synthetic through-water benchmark developed in Blender for airborne bathymetric mapping. The dataset comprises 1,200 scenes, originally rendered as 512 × 512 pixel RGB images. The simulated seabed includes randomized combinations of sandy flats, gravel, rocky substrates, and seagrasslike vegetation, providing a range of textures and reflectance characteristics.

For each scene, Sea-Undistort provides an original waterfree reference image and multiple distorted observations generated under different water-surface and water-column conditions, including refraction, sun glint, light absorption, and turbidity-related effects. In this study, we considered three distortion configurations: all effects, no waves, and no sunglint. The all-effects configuration combines surface-wave and water-column distortions, while the other two configurations exclude the corresponding surface effect. All three share the same water-free reference image, enabling consistent evaluation across distortion conditions. Some observations

Scene 472

exhibit strong glint and wave-induced distortions that partially or completely obscure the seabed, increasing the difficulty of restoration. Figure 4 illustrates representative scenes under the three configurations and their corresponding water-free references.

Scene 544  
Scene 782  
Scene 1183  
![](images/c8a5d689cca65a4268c351874d63f3274bbb45560300b0c0fb4e9096f969ce63.jpg)  
Fig. 4. Examples from the Sea-Undistort dataset illustrating four scenes under three distortion configurations: all effects, no waves, and no sun glint. The bottom row shows the corresponding original water-free reference images, which are identical across the three configurations.

Table I summarizes the distortions represented in both datasets. KAUSTics includes rendered caustics and refraction, while color-transfer and Gaussian-blur augmentations approximate absorption- and turbidity-related appearance changes. Sea-Undistort provides rendered refraction, sun glint, and water-column effects. Caustics were not independently controlled in the Sea-Undistort configurations considered here. The two datasets therefore provide complementary conditions for assessing restoration performance.

TABLE I  
DISTORTIONS REPRESENTED IN KAUSTICS AND SEA-UNDISTORT, DISTINGUISHING PHYSICALLY RENDERED EFFECTS FROM EMPIRICAL DATA AUGMENTATIONS.
<table><tr><td>Distortion</td><td>KAUSTics</td><td>Sea-Undistort</td></tr><tr><td>Caustics</td><td>Rendered</td><td></td></tr><tr><td>Refraction</td><td>Rendered</td><td>Rendered</td></tr><tr><td>Sun glint</td><td></td><td>Rendered</td></tr><tr><td>Light absorption</td><td>Augmented</td><td>Rendered</td></tr><tr><td>Turbidity</td><td>Augmented</td><td>Rendered</td></tr></table>

For the revised experiments, model adaptation used the alleffects observations, while testing also included the no-waves and no-sunglint configurations. The same scene identifiers and original water-free reference images were used across all three test conditions. The complete evaluation and adaptation protocols are described in the experimental setup section III-G.

## C. KAUSTics-Net Architecture

KAUSTics-Net is a pixel-space conditional diffusion framework designed to restore shallow-water UAV imagery affected by spatially varying water-induced distortions. The architecture consists of a U-Net denoiser, a multi-scale conditional encoder, feature-wise linear modulation (FiLM) residual blocks, and a cross-attention bottleneck (Fig. 5). The model operates directly on RGB images without a latent autoencoder or pretrained diffusion components and predicts diffusion velocity conditioned on the distorted observation.

Let $\mathbf { x } _ { 0 } \in \mathbb { R } ^ { H \times W \times 3 }$ denote the reference image and x<sub>cond</sub> $\in \mathbb { R } ^ { H \times W \times 3 }$ its corresponding distorted observation. During training, a noisy sample $\mathbf { x } _ { t }$ is generated from $\mathbf { x } _ { \mathrm { 0 } }$ at diffusion timestep t, as described in Section III-D. The U-Net receives the channel-wise concatenation of $\mathbf { x } _ { t }$ and $\mathbf { x } _ { \mathrm { c o n d } } .$ while a separate five-stage conditional encoder extracts multiscale features from the distorted observation. The encoder uses $3 \times 3$ convolutions, group normalization, SiLU activations, and bilinear downsampling, with channel dimensions of 16, 32, 64, 128, and 256.

The extracted features condition the U-Net residual blocks through FiLM [53]. A 256-dimensional sinusoidal timestep embedding and globally pooled conditional features are projected into channel-wise scale and shift parameters, allowing the residual blocks to incorporate both the diffusion timestep and image-dependent information. At the $1 6 \times 1 6$ bottleneck, a single-head cross-attention mechanism with 256-dimensional features further integrates the conditioning information. U-Net features serve as queries, while projected conditional features provide keys and values. The decoder combines these representations with skip connections to predict the diffusion velocity $\mathbf { v } _ { \theta } ( \mathbf { x } _ { t } , \mathbf { x } _ { \mathrm { c o n d } } , t )$

This conditioning strategy is motivated by the combination of localized caustics and refractive distortions with broader radiometric changes in through-water imagery. Multi-scale features provide image-dependent information throughout denoising, while the bottleneck attention mechanism integrates spatial information from the distorted observation. The individual architectural components follow established diffusion and conditioning techniques; their contribution within KAUSTics-Net is examined through the ablation experiments.

## D. Diffusion Training and Image Reconstruction

KAUSTics-Net follows a discrete-time diffusion process [41] with a cosine noise schedule [54]. Reference images are normalized to [−1, 1], and the forward process uses $T = 1 0 0 0$ diffusion states. At training, a timestep t is uniformly sampled from $\{ 1 , \ldots , T - 1 \}$ , and Gaussian noise $\varepsilon \sim \mathcal { N } ( { \bf 0 , I } )$ is added to the reference image x<sub>0</sub>:

$$
{ \bf x } _ { t } = a _ { t } { \bf x } _ { 0 } + b _ { t } \varepsilon , \qquad a _ { t } = \sqrt { \bar { \alpha } _ { t } } , \quad b _ { t } = \sqrt { 1 - \bar { \alpha } _ { t } } ,\tag{1}
$$

where $\begin{array} { r } { \bar { \alpha } _ { t } ~ = ~ \prod _ { s = 1 } ^ { t } ( 1 - \beta _ { s } ) } \end{array}$ denotes the cumulative noise schedule.

The network predicts the velocity target, which combines the reference image and the sampled noise. The corresponding clean-image estimate is obtained as

$$
\begin{array} { r l } & { \mathbf { v } _ { t } = a _ { t } \pmb { \varepsilon } - b _ { t } \mathbf { x } _ { 0 } , } \\ & { \hat { \mathbf { x } } _ { 0 } = a _ { t } \mathbf { x } _ { t } - b _ { t } \hat { \mathbf { v } } _ { \theta } , } \end{array}\tag{2}
$$

![](images/3f3358317f37f8b1526d3beea8ac43e9fb870f170c40e8bf17ccaadc0b63b9bd.jpg)  
Fig. 5. Architecture of KAUSTics-Net. The distorted observation $\mathbf { x } _ { \mathrm { c o n d } }$ is processed by a multi-scale conditional encoder and concatenated with the noisy target x<sub>t</sub> at the U-Net input. Time- and image-conditioned FiLM residual blocks integrate conditioning information throughout the network, while singlehead cross-attention fuses U-Net and conditional features at the bottleneck. The decoder predicts diffusion velocity $\mathbf { v } _ { \theta } ( \mathbf { x } _ { t } , \mathbf { x } _ { \mathrm { c o n d } } , t )$ for reference-image reconstruction.

where $\hat { \mathbf { v } } _ { \theta } ~ = ~ \mathbf { v } _ { \theta } ( \mathbf x _ { t } , \mathbf x _ { \mathrm { c o n d } } , t )$ is the predicted velocity. This formulation enables direct reconstruction of the referenceimage estimate during training and sampling.

At inference, the reverse diffusion process is initialized by adding Gaussian noise to the distorted observation $\mathbf { x } _ { \mathrm { c o n d } } ,$ using a strength parameter of $s ~ = ~ 0 . 9 5$ , corresponding to diffusion state $t _ { 0 } = 9 4 9$ . Restoration is performed using 25 deterministic DDIM sampling steps (η = 0) [55], progressing from the selected initial state toward the reconstructed image. The distorted observation is provided as conditioning at every step. Classifier-free guidance with a scale of 2 is applied using a conditioning dropout probability of 0.1 during training, resulting in 50 network evaluations per restored image.

The final reconstructed image is converted to the RGB intensity range for evaluation. The diffusion process and sampling configuration are shared across the KAUSTics experiments, while the Sea-Undistort adaptation protocol is described separately.

## E. Loss Functions and Optimization Strategy

KAUSTics-Net was trained using a composite objective combining diffusion velocity prediction with spatial, structural, and chromatic reconstruction losses. The total objective is defined as

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { t o t a l } } = 0 . 2 \mathcal { L } _ { v } + 0 . 8 \mathcal { L } _ { \mathrm { R G B } } + 0 . 2 \mathcal { L } _ { \mathrm { L F } } } \\ & { ~ + 0 . 3 \mathcal { L } _ { \mathrm { U V } } + 0 . 2 \mathcal { L } _ { \mathrm { S S I M } } + 0 . 2 \mathcal { L } _ { \mathrm { L a b } } , } \end{array}\tag{3}
$$

where the coefficients were selected empirically to balance reconstruction fidelity, structural consistency, and color preservation.

The velocity prediction loss $\mathcal { L } _ { v }$ minimizes the mean squared error (MSE) between the predicted velocity $\hat { \mathbf { v } } _ { \theta }$ and the target $\mathbf { v } _ { t }$ defined in Eq. (2). The RGB reconstruction loss $\mathcal { L } _ { \mathrm { { R G B } } }$ similarly applies MSE between the reconstructed image $\hat { \mathbf { x } } _ { 0 }$ and the reference $\mathbf { x } _ { 0 } .$ . To preserve large-scale appearance, the lowfrequency loss $\mathcal { L } _ { \mathrm { L F } }$ computes the $L _ { 1 }$ difference between the reconstructed and reference images after Gaussian smoothing with a $2 1 \times 2 1$ kernel and standard deviation $\sigma = 5$ . The chromatic loss ${ \mathcal { L } } _ { \mathrm { U V } }$ applies $L _ { 1 }$ differences to the U and V channels in YUV color space, encouraging consistent color reproduction.

Two additional terms constrain structural and perceptual color differences. The structural loss L<sub>SSIM</sub> is defined as one minus the structural similarity index (SSIM), computed over RGB channels using a uniform $7 \times 7$ window. The colorspace loss ${ \mathcal { L } } _ { \mathrm { L a b } }$ measures the $L _ { 1 }$ difference between Gaussiansmoothed images converted to D65/2-degree CIELAB space, with the L, a, and b channels normalized by 100, 128, and 128, respectively. Predicted RGB values are clipped to [0, 1] for the SSIM and CIELAB losses, while the RGB, low-frequency, and UV reconstruction losses use unclipped predictions. All loss terms are averaged over the image and batch. These objectives provide complementary reconstruction constraints without explicitly modeling water-column optical properties.

Optimization was performed using AdamW with a learning rate of $1 0 ^ { - 4 }$ , weight decay of $1 0 ^ { - \bar { 4 } }$ , and momentum coefficients $( \beta _ { 1 } , \beta _ { 2 } ) = ( 0 . 9 , 0 . 9 9 9 )$ . Training used a batch size of 12, FP32 precision, and gradient clipping at a global norm of 1.0. The model was trained for 1,000 epochs without learningrate decay or exponential moving average (EMA) of network weights. Conditioning dropout was applied with probability $p _ { \mathrm { c o n d } } = 0 . 1$ to support classifier-free guidance during inference. Model checkpoints were selected using validation PSNR, as detailed in the experimental setup.

## F. Benchmark Models and Evaluation Metrics

We benchmarked KAUSTics-Net against eight imagerestoration methods spanning GAN-based architectures, underwater image enhancement, and diffusion-based restoration. The GAN-based baselines include CycleGAN [37], UGAN [27], FUnIE-GAN [38], and DeWaterGAN [23]. The diffusion-based methods include DiffWater [28] and ResShift [48]. We additionally evaluated SS-UIE [49] and U-Shape Transformer[50], which incorporate spatial–spectral modeling and transformer-based feature extraction, respectively. These methods provide complementary architectural comparisons for restoring imagery affected by spatially varying distortions. All eight baselines were trained and evaluated using the revised experimental protocol described in the following subsection.

Restoration fidelity was assessed using mean peak signalto-noise ratio (mPSNR), mean structural similarity index (mSSIM), mean root mean squared error (mRMSE), and mean squared error (mMSE). We additionally included learned perceptual image patch similarity (LPIPS) [56] and CIEDE2000 $( \Delta E _ { 0 0 } )$ [57] to quantify perceptual feature differences and color discrepancies relative to the reference images. Higher values indicate better performance for mPSNR and mSSIM, while lower values are preferred for the remaining metrics. LPIPS was computed using an AlexNet backbone on 8-bit RGB outputs, and CIEDE2000 was evaluated in D65/2-degree CIELAB space.

No-reference image quality was evaluated using the Natural Image Quality Evaluator (NIQE) [58], Blind/Referenceless Image Spatial Quality Evaluator (BRISQUE) [59], Perceptionbased Image Quality Evaluator (PIQE) [60], and Integrated

Local NIQE (ILNIQE) [61]. These metrics assess deviations from statistical properties associated with natural images, with lower values indicating better predicted quality. Three learning-based perceptual metrics were also included: Learned Image Quality Evaluator (LIQE) [62], CLIP-based Image Quality assessment (CLIPIQ) [63], and Multi-scale Image Quality (MUSIQ) [64]. For MUSIQ, we report the AVAtrained variant (MUSIQ\_AVA), with higher scores indicating better predicted perceptual quality.

Sharpness and Shannon entropy were further computed as complementary indicators of image detail and intensity variability. Sharpness was defined as the variance of the discrete Laplacian, while entropy was calculated from the 256-bin luminance histogram. These metrics characterize changes in image structure and tonal distribution but do not independently establish restoration accuracy, as noise and artifacts can also increase their values.

Finally, to investigate the potential utility of restoration for downstream UAV image analysis, we evaluated real-world examples using the Segment Anything Model (SAM) [65] and scale-invariant feature transform (SIFT) [66], following the computer-vision assessment approach previously applied to shallow-water UAV imagery [12]. SAM was used to examine automatically generated image regions, while SIFT quantified local keypoint detection and descriptor correspondence. The real-UAV comparison used a temporal computer-vision reference from our previous work, which provides a low-distortion comparison image without constituting independently validated water-free ground truth. These analyses complement the image quality metrics by examining the preservation of local features and segmentation behavior.

## G. Experimental Setup

The stored KAUSTics images were converted to RGB and normalized to [−1, 1] for training. All methods used the same 256×256 image resolution, obtained by bicubic downsampling of the original 1000 × 1000 patches without additional spatial cropping.

To prevent leakage between augmented observations of the same source image, the dataset was partitioned at the UAVimage level. Using a fixed random seed of 42, the 32 source images were divided into 26 training, three validation, and three test images. Every reference patch and its ten degraded variants inherited the partition of the corresponding source image. Table II summarizes the resulting dataset composition.

The partitioning protocol ensures independence of sourceimage identifiers across training, validation, and testing. However, geographic independence cannot be guaranteed because different UAV frames may cover overlapping seabed areas. For KAUSTics-Net, each training epoch contained 2,080 patch– shape combinations, with additional degradation variants sampled with probability 0.6. Model selection used a fixed, stratified subset of 120 validation observations evaluated through the conditional sampling procedure. The complete test set contained all 1,200 degraded observations from the three heldout UAV images.

All eight benchmark models were retrained using the same source-image partitions and evaluated on identical test observations. Their original architectures and training objectives were retained, with resource-dependent adaptations to learning rates, batch sizes, and training schedules. Benchmark training included a minimum of 100 epochs, followed by validationbased early stopping with a minimum PSNR improvement of 0.05 dB, patience of ten epochs, and a maximum of 1,000 epochs. Validation was performed every ten epochs. KAUSTics-Net was trained for 1,000 epochs, with checkpoint selection based on validation PSNR.

TABLE II  
KAUSTICS SOURCE-IMAGE SPLITS. EACH UAV IMAGE CONTRIBUTES 40 REFERENCE PATCHES, WITH TEN DEGRADED OBSERVATIONS PER PATCH.
<table><tr><td>Partition</td><td>UAV images</td><td>Patches</td><td>Pairs</td></tr><tr><td>Training</td><td>26</td><td>1,040</td><td>10,400</td></tr><tr><td>Validation</td><td>3</td><td>120</td><td>1,200</td></tr><tr><td>Test</td><td>3</td><td>120</td><td>1,200</td></tr><tr><td>Total</td><td>32</td><td>1,280</td><td>12,800</td></tr></table>

Table III reports the configurations and measured training durations for the completed experiments. All models were independently trained using three random seeds (1–3), and results are reported as the mean and standard deviation across seeds. Differences in architectures, GPU hardware, and training schedules mean that the reported runtimes do not constitute a matched-compute comparison. KAUSTics-Net required approximately 0.367 s per image for sampling on an NVIDIA A100 GPU, excluding image-quality metric computation. Additional model-specific implementation details are provided in the supplementary material.

TABLE III  
TRAINING CONFIGURATIONS AND MEASURED RUNTIMES ON KAUSTICS. ALL MODELS USED 256 × 256 IMAGES. TIMES REPRESENT MEAN FULL-RUN TRAINING DURATIONS ACROSS THREE SEEDS, INCLUDING VALIDATION AND RESUME OVERHEAD.
<table><tr><td>Method</td><td>LR</td><td>Batch</td><td>GPU</td><td>Time (h)</td></tr><tr><td>CycleGAN</td><td> $2 \times 1 0 ^ { - 4 }$ </td><td>16</td><td>A100</td><td>7.455</td></tr><tr><td>UGAN</td><td> $1 \times 1 0 ^ { - 4 }$ </td><td>128</td><td>V100</td><td>1.302</td></tr><tr><td>FUnIE-GAN</td><td> $3 \times 1 0 ^ { - 4 }$ </td><td>128</td><td>V100</td><td>1.212</td></tr><tr><td>DeWaterGAN</td><td> $1 \times 1 0 ^ { - 4 }$ </td><td>8</td><td>A100</td><td>21.392</td></tr><tr><td>DiffWater</td><td> $3 \times 1 0 ^ { - 6 }$ </td><td>1</td><td>A100</td><td>20.931</td></tr><tr><td>ResShift</td><td> $5 \times 1 0 ^ { - 5 }$ </td><td>8</td><td>V100</td><td>26.470</td></tr><tr><td>SS-UIE</td><td> $1 \times 1 0 ^ { - 4 }$ </td><td>9</td><td>V100</td><td>8.451</td></tr><tr><td>U-Shape Transformer</td><td> $5 \times 1 0 ^ { - 4 }$ </td><td>8</td><td>V100</td><td>6.712</td></tr><tr><td>KAUSTics-Net (ours)</td><td> $1 \times 1 0 ^ { - 4 }$ </td><td>12</td><td>A100</td><td>4.167</td></tr></table>

## H. Sea-Undistort Adaptation Protocol

To investigate cross-dataset transfer, KAUSTics-Net was first evaluated on Sea-Undistort without additional training. We subsequently fine-tuned the KAUSTics-trained model using the existing scene-disjoint partition of 840 training, 240 validation, and 120 test scenes. Fine-tuning used the alleffects observations, while evaluation covered the all-effects, no-waves, and no-sunglint configurations. Each distorted observation was compared against its corresponding original water-free reference image, and the same scene identifiers were maintained across configurations.

All eight benchmark methods were fine-tuned from their corresponding seed-specific, validation-selected KAUSTics checkpoints, with optimizer states reinitialized. Training used only the all-effects observations from the 840 Sea-Undistort training scenes, with the original water-free references; CycleGAN and DeWaterGAN retained their unpaired objectives using independently sampled clean images from the same training pool. Each baseline completed 100 epochs at a constant learning rate of $1 0 ^ { - 5 }$ , retaining its method-specific objective and batch size. Checkpoints were selected by mean SSIM on the 240 all-effects validation scenes, with the initial checkpoint remaining eligible. DiffWater used less frequent validation after its initial checks because of the computational cost of diffusion sampling. All methods were evaluated using three corresponding training seeds on the same 120 heldout scenes under the three distortion configurations. Complete optimization and checkpoint-selection details are provided in the supplementary material.

Fine-tuning retained the original architecture and training objective, supplemented by a sampler-aware reconstruction loss. Every fourth optimizer update included an additional loss computed from the restored output of the 25-step DDIM sampling procedure:

$$
\mathcal { L } _ { \mathrm { a d a p t } } = 0 . 8 \mathcal { L } _ { \mathrm { R G B } } + 0 . 2 \mathcal { L } _ { \mathrm { S S I M } } + 0 . 2 \mathcal { L } _ { \mathrm { L a b } } ,\tag{4}
$$

where the RGB, SSIM, and CIELAB terms are defined in Section III-E. To limit memory consumption, gradients were propagated through only the final four sampling steps, with the preceding 21 steps detached. This provides a truncated gradient approximation of the inference procedure.

Optimization used AdamW with a learning rate of $3 \times 1 0 ^ { - 5 }$ weight decay of $1 0 ^ { - 4 }$ , batch size of 12, and conditioning dropout probability of 0.1. Fine-tuning was limited to 100 epochs, with checkpoints selected according to mean validation SSIM across all 240 validation scenes. Validation was performed at epochs 0 and 1 and subsequently every five epochs. The sampling configuration remained unchanged, using strength 0.95 and classifier-free guidance scale 2.

The adaptation experiments were conducted using three independent training seeds. Since the expanded evaluation followed inspection of preliminary test results, these experiments are treated as exploratory. The zero-shot results are reported separately in the supplementary material to distinguish performance without adaptation from improvements obtained through dataset-specific fine-tuning.

## IV. RESULTS

## A. Results on the KAUSTics Dataset

1) Qualitative Comparison: Figure 6 presents a qualitative comparison of the restoration methods on ten selected patches from the held-out KAUSTics test set, covering the two caustic patterns and their corresponding blur and color-transfer variants. All methods were evaluated on identical inputs, allowing direct visual comparison of distortion suppression, color reconstruction, and preservation of seabed textures.

The GAN-based methods exhibited varying restoration characteristics. CycleGAN and UGAN reduced some distortions

TABLE IV  
NO-REFERENCE IMAGE QUALITY METRICS ON THE KAUSTICS TEST SET (1,200 OBSERVATIONS). VALUES ARE REPORTED AS MEAN ± STANDARD DEVIATION ACROSS THREE TRAINING SEEDS. ALL METHODS WERE EVALUATED AT 256 × 256 RESOLUTION.
<table><tr><td>Method</td><td>NIQE↓</td><td>BRISQUE↓</td><td></td><td>PIQE↓</td><td>ILNIQE ↓</td><td>LIQE ↑</td><td></td><td>CLIPIQ ↑</td><td></td><td>MUSIQ_AVA ↑</td><td></td><td>Sharpness</td><td></td><td>Entropy</td></tr><tr><td>CycleGAN [37]</td><td> $\mathbf { \overline { { 5 . 8 7 0 \pm 0 . 1 0 5 } } }$ </td><td> $\overline { { 1 5 . 5 9 4 \pm 3 . 5 4 7 } }$ </td><td></td><td> $\overline { { 9 . 4 9 0 \pm 3 . 4 5 0 } }$ </td><td> $7 7 . 6 5 4 \pm 3 . 5 0 0$ </td><td> $\overline { { 1 . 2 3 3 \pm 0 . 0 8 1 } }$ </td><td></td><td> $\overline { { 0 . 3 1 5 \pm 0 . 0 6 2 } }$ </td><td></td><td> $\overline { { 3 . 0 7 8 \pm 0 . 0 5 1 } }$ </td><td></td><td> $\overline { { 1 3 4 . 2 6 5 \ : \pm \ : 6 0 . 2 6 9 } }$ </td><td></td><td> $\overline { { 5 . 0 4 1 \pm 0 . 1 7 6 } }$ </td></tr><tr><td>UGAN [27]</td><td> $6 . 5 1 4 \pm 0 . 1 2 9$ </td><td></td><td> $2 7 . 9 0 5 \pm 0 . 4 4 8$ </td><td> $6 . 6 8 1 \pm 0 . 4 8 6$ </td><td> $1 1 1 . 1 9 6 \pm 2 . 3 9 2$ </td><td> $1 . 0 3 3 \pm 0 . 0 0 9$ </td><td></td><td></td><td> $0 . 2 5 3 \pm 0 . 0 1 0$ </td><td> $2 . 8 0 2 \pm 0 . 0 1 5$ </td><td></td><td> $7 6 . 0 0 6 \pm 8 . 8 1 3$ </td><td></td><td> $4 . 7 3 4 \pm 0 . 0 8 4$ </td></tr><tr><td>FUnIE-GAN [38]</td><td> $7 . 3 0 9 \pm 0 . 5 5 5$ </td><td> $\mathbf { 1 0 . 7 7 9 \ : \pm { \ : 0 . 3 9 9 } }$ </td><td></td><td></td><td> $8 9 . 0 1 8 \pm 7 . 6 1 5$ </td><td> $1 . 1 7 8 \pm 0 . 0 5 8$ </td><td></td><td> $0 . 3 2 7 \pm 0 . 0 0 8$ </td><td></td><td></td><td></td><td> $1 1 7 . 2 0 3 \pm 3 3 . 8 1 9$ </td><td></td><td> $4 . 9 6 5 \pm 0 . 0 5 9$ </td></tr><tr><td></td><td> $8 . 2 5 5 \pm 0 . 7 3 6$ </td><td></td><td> $2 0 . 7 3 4 \pm 1 . 2 6 5$ </td><td> $1 0 . 9 2 2 \pm 3 . 9 6 3$ </td><td> $9 3 . 4 0 5 \pm 1 . 2 9 5$ </td><td></td><td></td><td></td><td></td><td> $3 . 0 9 3 \pm 0 . 0 7 6$ </td><td></td><td> $1 0 7 . 2 2 0 \pm 2 9 . 2 3 4$ </td><td></td><td> $4 . 7 7 1 \pm 0 . 0 8 0$ </td></tr><tr><td>DeWaterGAN [23]</td><td></td><td></td><td></td><td> $1 0 . 8 6 7 \pm 4 . 0 4 4$ </td><td></td><td> $1 . 2 8 5 \pm 0 . 0 5 1$ </td><td></td><td> $0 . 2 9 0 \pm 0 . 0 3 4$ </td><td></td><td> $3 . 0 3 7 \pm 0 . 0 8 0$ </td><td></td><td> $3 6 9 . 7 4 9 \pm 1 5 7 . 7 0 0$ </td><td></td><td> $5 . 2 1 5 \pm 0 . 2 4 3$ </td></tr><tr><td>DiffWater [28] ResShift [48]</td><td> $6 . 2 6 1 \pm 0 . 8 9 3$   $1 1 . 2 1 5 \pm 0 . 0 7 9$ </td><td></td><td> $2 3 . 6 4 1 \pm 2 . 7 6 2$   $1 9 . 7 8 1 \pm 0 . 4 4 4$ </td><td> $2 0 . 1 2 4 \pm 7 . 6 8 8$   $9 . 6 9 5 \pm 1 . 3 1 2$ </td><td> $1 1 4 . 7 1 7 \pm 1 . 9 4 7$   $9 6 . 2 1 1 \pm 1 . 9 5 0$ </td><td> $1 . 1 0 8 \pm 0 . 0 0 5$ </td><td></td><td> $0 . 3 3 4 \pm 0 . 0 0 9$   $0 . 3 4 1 \pm 0 . 0 0 6$ </td><td></td><td> $2 . 7 9 7 \pm 0 . 0 5 2$   $3 . 2 2 1 \pm 0 . 0 0 6$ </td><td></td><td> $1 7 5 . 1 5 6 \pm 1 4 . 0 1 3$ </td><td></td><td> $4 . 8 0 5 \pm 0 . 0 2 6$ </td></tr><tr><td>SS-UIE [49]</td><td> $1 0 . 1 3 0 \pm 0 . 2 5 7$ </td><td> $3 0 . 0 2 5 \pm 3 . 0 6 4$ </td><td></td><td> $3 0 . 9 3 6 \pm 3 . 7 6 6$ </td><td> $\mathbf { 7 5 . 3 1 3 \ : \pm { \ : 5 . 2 6 6 } }$ </td><td> $\mathbf { 1 . 5 6 8 \ : \pm { \ : 0 . 0 2 3 } }$   $1 . 3 4 3 \pm 0 . 0 5 9$ </td><td></td><td> $\mathbf { 0 . 4 0 3 \ : \pm { \ : 0 . 0 0 4 } }$ </td><td></td><td> $\mathbf { 3 . 3 7 3 \ : \pm { \ : 0 . 0 2 1 } }$ </td><td></td><td> $6 0 7 . 9 9 2 \pm 1 7 . 9 3 6$ </td><td></td><td> $5 . 0 8 5 \pm 0 . 0 6 7$ </td></tr><tr><td>U-Shape Transformer [50]</td><td> $7 . 6 2 6 \pm 0 . 2 7 6$ </td><td> $4 8 . 5 4 7 \pm 5 . 5 7 5$ </td><td></td><td> $3 9 . 4 3 7 \pm 7 . 6 5 7$ </td><td> $1 2 5 . 1 9 4 \pm 2 . 3 0 8$ </td><td> $1 . 1 2 7 \pm 0 . 0 3 1$ </td><td></td><td> $0 . 3 7 0 \pm 0 . 0 1 0$ </td><td></td><td> $2 . 4 9 7 \pm 0 . 0 4 4$ </td><td></td><td> $2 1 . 2 4 8 \pm 6 . 8 9 5$ </td><td></td><td> $4 . 4 2 1 \pm 0 . 1 8 1$ </td></tr><tr><td>KAUSTics-Net (ours)</td><td> $\underline { { 6 . 5 6 8 \pm 0 . 3 1 7 } }$ </td><td></td><td> $\underline { { 1 6 . 4 6 3 } } \pm 2 . 0 7 2$ </td><td> $\mathbf { 5 . 9 5 0 \pm 1 . 1 1 5 }$ </td><td> $\underline { { 8 6 . 4 1 0 \pm 3 . 9 3 7 } }$ </td><td></td><td> $\underline { { 1 . 3 6 4 \pm 0 . 0 0 3 } }$ </td><td> $\underline { { 0 . 3 3 4 } } \pm 0 . 0 4 7$ </td><td></td><td> $\underline { { 3 . 0 6 1 \pm 0 . 0 4 5 } }$ </td><td></td><td> $\underline { { 8 4 . 2 5 8 \pm 3 . 9 4 5 } }$ </td><td></td><td> $\underline { { 4 . 6 6 2 \ : \pm 0 . 0 4 9 } }$ </td></tr></table>

TABLE V

FULL-REFERENCE RESTORATION METRICS ON THE KAUSTICS TEST SET (1,200 OBSERVATIONS). VALUES ARE REPORTED AS MEAN ± STANDARD DEVIATION ACROSS THREE TRAINING SEEDS. HIGHER PSNR AND SSIM INDICATE BETTER FIDELITY, WHILE LOWER RMSE, MSE, LPIPS, AND ∆E INDICATE SMALLER RECONSTRUCTION DIFFERENCES.
<table><tr><td>Method</td><td>mPSNR ↑</td><td>mSSIM ↑</td><td>mRMSE ↓</td><td></td><td> $\overline { { { \bf { m } } { \bf { M } } { \bf { S } } { \bf { E } } \left[ { \bf { 1 0 } } ^ { - 3 } \right] \downarrow } }$ </td><td> $\overline { { { \bf { L P I P S } } \downarrow } }$ </td><td></td><td> $\overline { { \Delta E _ { 0 0 } \downarrow } }$ </td></tr><tr><td>CycleGAN [37]</td><td> $\overline { { 2 4 . 7 3 4 \pm 0 . 9 2 3 } }$ </td><td> $0 . 7 1 7 \pm 0 . 0 6 6$ </td><td> $\overline { { 0 . 0 6 5 \pm 0 . 0 0 6 } }$ </td><td></td><td> $\overline { { 5 . 1 0 6 \pm 0 . 8 7 1 } }$ </td><td></td><td> $\overline { { 0 . 3 7 2 \pm 0 . 0 5 5 } }$ </td><td> $\overline { { 8 . 1 4 6 \pm 0 . 9 7 3 } }$ </td></tr><tr><td>UGAN [27]</td><td> $2 6 . 5 4 6 \pm 0 . 3 2 1$ </td><td> $0 . 7 7 7 \pm 0 . 0 0 9$ </td><td> $0 . 0 5 0 \pm 0 . 0 0 2$ </td><td></td><td> $2 . 8 7 3 \pm 0 . 2 2 5$ </td><td></td><td> $0 . 3 4 4 \pm 0 . 0 1 3$ </td><td> $6 . 4 7 9 \pm 0 . 3 1 7$ </td></tr><tr><td>FUnIE-GAN [38]</td><td> $2 7 . 7 8 1 \pm 0 . 2 8 2$ </td><td> $0 . 7 5 8 \pm 0 . 0 2 0$ </td><td> $0 . 0 4 9 \pm 0 . 0 0 2$ </td><td></td><td> $3 . 2 9 0 \pm 0 . 5 9 0$ </td><td></td><td> $0 . 2 4 7 \pm 0 . 0 1 3$ </td><td> $5 . 2 9 6 \pm 0 . 3 1 1$ </td></tr><tr><td>DeWaterGAN [23]</td><td> $2 4 . 3 5 7 \pm 0 . 5 3 7$ </td><td> $0 . 7 6 9 \pm 0 . 0 2 3$ </td><td> $0 . 0 6 9 \pm 0 . 0 0 5$ </td><td></td><td> $5 . 7 5 1 \pm 0 . 9 3 9$ </td><td></td><td> $0 . 3 2 5 \pm 0 . 0 0 9$ </td><td> $7 . 4 9 2 \pm 0 . 3 9 7$ </td></tr><tr><td>DiffWater [28]</td><td> $2 1 . 1 6 7 \pm 1 . 4 1 9$ </td><td> $0 . 6 0 5 \pm 0 . 0 9 4$ </td><td> $0 . 0 9 5 \pm 0 . 0 1 5$ </td><td></td><td> $1 0 . 3 6 7 \pm 3 . 3 4 9$ </td><td></td><td> $0 . 4 1 2 \pm 0 . 0 7 1$ </td><td> $1 2 . 0 9 1 \pm 2 . 4 2 7$ </td></tr><tr><td>ResShift [48]</td><td> $3 0 . 3 4 6 \pm 0 . 5 1 3$ </td><td> $0 . 7 8 5 \pm 0 . 0 0 8$ </td><td> $0 . 0 3 2 \pm 0 . 0 0 2$ </td><td></td><td> $1 . 1 4 0 \pm 0 . 1 3 6$ </td><td></td><td> $0 . 1 5 1 \pm 0 . 0 0 7$ </td><td> $4 . 0 9 5 \pm 0 . 1 9 2$ </td></tr><tr><td>SS-UIE [49]</td><td> $2 8 . 3 8 7 \pm 0 . 3 2 3$ </td><td> $0 . 6 3 7 \pm 0 . 0 1 6$ </td><td> $0 . 0 4 2 \pm 0 . 0 0 1$ </td><td></td><td> $2 . 0 9 2 \pm 0 . 0 9 8$ </td><td></td><td> $0 . 4 3 8 \pm 0 . 0 1 4$ </td><td> $4 . 3 7 2 \pm 0 . 0 1 5$ </td></tr><tr><td>U-Shape Transformer [50]</td><td> $2 8 . 8 8 4 \pm 1 . 6 1 1$ </td><td> $0 . 8 4 4 \pm 0 . 0 1 6$ </td><td> $0 . 0 3 9 \pm 0 . 0 0 8$ </td><td></td><td> $1 . 7 5 5 \pm 0 . 8 0 0$ </td><td></td><td> $0 . 3 3 6 \pm 0 . 0 2 2$ </td><td> $4 . 6 3 5 \pm 1 . 3 7 8$ </td></tr><tr><td>KAUSTics-Net (ours)</td><td> $\mathbf { 3 1 . 7 1 5 \ : \pm { \ : 0 . 4 3 9 } }$ </td><td> $\mathbf { 0 . 8 4 9 \ : \pm { \ : 0 . 0 0 2 } }$ </td><td> $\mathbf { 0 . 0 2 8 \ : \pm { \ : 0 . 0 0 1 } }$ </td><td></td><td> $\mathbf { 0 . 9 8 2 \ : \pm { \ : 0 . 0 8 3 } }$ </td><td></td><td> $\mathbf { 0 . 1 4 5 \ : \pm { \ : 0 . 0 0 7 } }$ </td><td> $\mathbf { 3 . 0 8 0 \ : \pm { \ : 0 . 1 3 1 } }$ </td></tr></table>

the evaluated baselines.

but retained visible differences in color and local structure relative to the reference images. FUnIE-GAN enhanced local contrast, occasionally introducing pronounced edges and highfrequency artifacts, while DeWaterGAN produced smoother appearances in several regions. Among the remaining baselines, DiffWater and ResShift recovered some structural details, although differences in texture and radiometric appearance remained. SS-UIE and U-Shape Transformer also demonstrated distinct restoration behaviors, particularly in areas containing complex seabed textures and spatially varying distortions.

KAUSTics-Net generally produced reconstructions with reduced caustic patterns and improved correspondence to the reference appearance, while retaining recognizable sandy, rocky, and vegetated structures. Differences between methods were particularly visible in regions affected by strong illumination variations and color alterations. These examples illustrate restoration behavior across the selected degradation configurations; the quantitative results provide the broader assessment over the complete held-out test set.

Among the competing methods, ResShift achieved the second-highest mean PSNR (30.346 ± 0.513 dB) and a competitive LPIPS of 0.151 ± 0.007, while U-Shape Transformer obtained the second-highest SSIM (0.844 ± 0.016). SS-UIE reached $2 8 . 3 8 7 \pm 0 . 3 2 3$ dB PSNR but showed lower structural similarity $( 0 . 6 3 7 \pm 0 . 0 1 6 )$ , illustrating differences in reconstruction behavior among the evaluated architectures.

2) Quantitative Comparison: Quantitative results were evaluated on all 1,200 degraded observations from the sourceimage-disjoint KAUSTics test set. Tables IV and V report noreference image quality and full-reference restoration metrics, respectively, averaged across three independent training seeds.

The full-reference results in Table V show that KAUSTics-Net achieved the highest mean PSNR of $3 1 . 7 1 5 \pm 0 . 4 3 9$ dB and SSIM of 0.849 ± 0.002. It also obtained the lowest mean RMSE (0.028 ± 0.001), MSE $( 0 . 9 8 2 \pm 0 . 0 8 3 \times 1 0 ^ { - 3 } )$ , LPIPS $( 0 . 1 4 5 \pm 0 . 0 0 7 )$ , and $\Delta E _ { 0 0 } ~ ( 3 . 0 8 0 \pm 0 . 1 3 1 )$ . These results indicate improved reconstruction fidelity, structural similarity, perceptual feature agreement, and color consistency relative to

The no-reference results in Table IV show a more heterogeneous ranking. KAUSTics-Net achieved the lowest PIQE $( 5 . 9 5 0 \pm 1 . 1 1 5 )$ , while other methods obtained better scores on several naturalness-based and learned perceptual metrics. For example, FUnIE-GAN achieved the lowest BRISQUE $( 1 0 . 7 7 9 \pm 0 . 3 9 9 )$ , ResShift obtained the highest LIQE (1.568±0.023), and SS-UIE led CLIPIQ (0.403±0.004) and MUSIQ\_AVA $( 3 . 3 7 3 \pm 0 . 0 2 1 )$ . KAUSTics-Net therefore did not consistently outperform competing methods in predicted natural-image quality.

The differences between full-reference and no-reference metrics highlight the importance of evaluating restoration using complementary criteria. The full-reference results favor KAUSTics-Net in reconstructing the paired calm-water targets, whereas no-reference metrics produce different rankings based on their underlying image-quality assumptions. Sharpness and entropy provide additional information about edge content and intensity variability, but their values may also increase because of noise or restoration artifacts. Overall, KAUSTics-Net demonstrates strong reconstruction performance on the held-out KAUSTics data, with remaining differences between reference fidelity and predicted perceptual naturalness.

![](images/611d1114ec5138e4297ec4544aa5ce43af96169db7c9fe169912f4c06810557c.jpg)  
Fig. 6. Qualitative comparison of restoration methods on ten selected KAUSTics test patches. Rows represent the two caustic patterns under the base, blurred, and three color-transfer configurations (Color 1–3). Columns show the distorted input, calm-water reference, eight benchmark outputs, and KAUSTics-Net. All methods use identical 256 × 256 inputs and seed-1 checkpoints. The central zoomed regions illustrate differences in local texture and distortion suppression.

## B. Results on the Sea-Undistort Dataset

1) Qualitative Comparison: We next evaluated the restoration methods on Sea-Undistort, which includes more severe combinations of water-surface and water-column distortions. Figure 7 presents representative results from the held-out test set under the all-effects, no-waves, and no-sunglint configurations. These conditions allow the visual assessment of restoration performance under different combinations of refraction, surface reflections, and radiometric degradation.

The evaluated methods exhibited different restoration characteristics. The GAN-based models generally modified the color and contrast of distorted observations but retained visible artifacts or introduced smoothing in several regions. DiffWater, ResShift, SS-UIE, and U-Shape Transformer recovered different levels of structural detail, although reconstruction quality varied with distortion severity. Following dataset-specific adaptation, KAUSTics-Net reduced several visible surfacerelated artifacts and improved the overall correspondence with the reference appearance. However, some reconstructed regions remained smoother than the original references, particularly where strong distortions obscured fine seabed features. These examples illustrate the remaining challenges of recovering information under severe through-water degradation.

2) Quantitative Comparison: Tables VII and VI report the no-reference and full-reference results, respectively, across the three Sea-Undistort configurations. Each configuration contains 120 held-out scenes, and results are reported as mean and standard deviation across three independent training seeds. The original zero-shot evaluation without Sea-Undistort adaptation is reported separately in the supplementary material.

![](images/a6de740f346e8cda26197c16f8c432bc6874259fc158d746d7d09d79f865e7c3.jpg)  
Fig. 7. Qualitative comparison of restoration methods on the Sea-Undistort test set. Rows show three scenes under the all-effects, no-waves, and no-sunglint configurations. Columns correspond to the distorted input, original water-free reference, eight benchmark methods, and KAUSTics-Net. All methods were evaluated at 256 × 256 resolution using the corresponding seed-1 checkpoints selected through validation.

reconstruction.

The full-reference results in Table VI show that KAUSTics-Net achieved the highest mean PSNR and SSIM across all three configurations. Under all effects, it obtained $1 9 . 4 6 9 \pm$ 0.332 dB PSNR and $0 . 5 5 4 \pm 0 . 0 0 5 \mathrm { { \ S S I M } }$ , compared with 10.257 dB and 0.311 for the unprocessed input. Performance further improved under no waves, reaching 20 $. 7 7 0 \pm 0 . 0 5 1$ dB and $0 . 6 5 9 { \pm } 0 . 0 0 6$ , while the no-sunglint configuration yielded $2 0 . 1 5 7 \pm 0 . 0 5 7$ dB and $0 . 5 9 2 \pm 0 . 0 0 3$ . KAUSTics-Net also achieved the lowest mean $\Delta E _ { 0 0 }$ across the three configurations, indicating improved color agreement with the synthetic references.

ResShift provided the strongest competing full-reference results, reaching PSNR values of $1 8 . 0 2 2 \pm 0 . 0 9 5 , 1 9 . 4 4 0 \pm 0 . 0 5 7 ,$ and $1 8 . 4 1 4 \pm 0 . 1 5 5$ dB under all effects, no waves, and no sunglint, respectively. However, ResShift consistently achieved lower LPIPS values (0.288, 0.223, and 0.265) than KAUSTics-Net (0.474, 0.370, and 0.437). These findings indicate that the improvements in pixel-wise fidelity and structural similarity were accompanied by differences in perceptual feature

The no-reference results in Table VII showed more heterogeneous rankings. Under the all-effects configuration, Cycle-GAN obtained the lowest NIQE $( 6 . 0 7 1 { \scriptstyle \pm 0 . 3 3 0 } )$ , DiffWater the lowest PIQE (5 $. 1 2 0 \pm 0 . 4 3 9 )$ , and ResShift the highest LIQE $( 2 . 4 2 5 \pm 0 . 0 0 3 )$ and CLIPIQ $( 0 . 5 2 2 \pm 0 . 0 0 3 )$ . KAUSTics-Net did not consistently lead these metrics, despite its superior full-reference reconstruction scores. Similar differences were observed under the no-waves and no-sunglint conditions.

Overall, the adapted KAUSTics-Net achieved stronger agreement with the original water-free references in terms of PSNR, SSIM, and $\Delta E _ { 0 0 }$ across all three configurations. The lower LPIPS values obtained by ResShift and the mixed noreference rankings highlight remaining perceptual differences. These results demonstrate the benefits of dataset-specific adaptation while emphasizing the need to consider complementary fidelity and perceptual metrics when evaluating through-water image restoration.

TABLE VI  
FULL-REFERENCE RESTORATION METRICS ON SEA-UNDISTORT UNDER THE ALL-EFFECTS, NO-WAVES, AND NO-SUNGLINT CONFIGURATIONS. EACH CONFIGURATION COMPRISES 120 HELD-OUT SCENES, EVALUATED AGAINST THEIR ORIGINAL WATER-FREE REFERENCES. VALUES ARE REPORTED AS MEAN ± STANDARD DEVIATION ACROSS THREE TRAINING SEEDS. HIGHER PSNR AND SSIM AND LOWER RMSE, MSE, LPIPS, AND $\Delta E _ { 0 0 }$ INDICATE BETTER AGREEMENT WITH THE REFERENCES.
<table><tr><td>Method</td><td>mPSNR ↑</td><td>mSSIM ↑</td><td> $\overline { { { \bf \ m } { \bf R } { \bf M } { \bf S } { \bf E } \mathrm { \Lambda } } }$ </td><td>mMSE  $\frac { [ 1 0 ^ { - 3 } ] \downarrow } { [ 1 0 ^ { - 3 } ] \downarrow }$ </td><td> $\overline { { { \bf { L P I P S } } \downarrow } }$ </td><td> $\overline { { \Delta E _ { 0 0 } \downarrow } }$ </td><td></td></tr><tr><td colspan="6">All effects</td><td></td><td></td></tr><tr><td>Unprocessed input</td><td>10.257</td><td>0.311</td><td>0.341</td><td>136.826</td><td>0.698</td><td></td><td>32.385</td></tr><tr><td>CycleGAN [37]</td><td> $1 5 . 5 4 3 \pm 0 . 4 6 2$ </td><td> $0 . 3 8 7 \pm 0 . 0 1 9$ </td><td> $0 . 2 1 5 \pm 0 . 0 0 9$ </td><td> $7 3 . 7 5 6 \pm 4 . 4 5 9$ </td><td> $0 . 4 4 4 \pm 0 . 0 2 1$ </td><td></td><td> $1 6 . 9 5 2 \pm 0 . 9 7 8$ </td></tr><tr><td>UGAN [27]</td><td> $1 0 . 6 3 0 \pm 0 . 1 2 2$ </td><td> $0 . 3 1 8 \pm 0 . 0 0 5$ </td><td> $0 . 3 0 5 \pm 0 . 0 0 4$ </td><td> $9 9 . 8 7 5 \pm 2 . 5 5 7$ </td><td> $0 . 7 7 6 \pm 0 . 0 1 3$ </td><td></td><td> $2 6 . 4 1 1 \pm 0 . 7 3 0$ </td></tr><tr><td>FUnIE-GAN [38]</td><td> $1 6 . 8 8 6 \pm 0 . 2 7 3$ </td><td> $0 . 3 6 7 \pm 0 . 0 1 1$ </td><td> $0 . 1 8 1 \pm 0 . 0 0 6$ </td><td> $5 3 . 2 6 2 \pm 2 . 4 6 8$ </td><td> $0 . 4 9 7 \pm 0 . 0 1 4$ </td><td></td><td> $1 4 . 7 6 5 \pm 0 . 4 5 6$ </td></tr><tr><td>DeWaterGAN [23]</td><td> $1 1 . 8 8 2 \pm 1 . 0 8 8$ </td><td> $0 . 3 3 4 \pm 0 . 0 0 5$ </td><td> $0 . 3 1 8 \pm 0 . 0 1 0$ </td><td> $1 3 4 . 8 4 9 \pm 6 . 5 5 4$ </td><td> $0 . 5 9 4 \pm 0 . 0 5 1$ </td><td></td><td> $2 8 . 1 8 7 \pm 2 . 0 9 6$ </td></tr><tr><td>DiffWater [28]</td><td> $1 0 . 6 0 0 \pm 0 . 9 4 8$ </td><td> $0 . 3 2 2 \pm 0 . 0 2 9$ </td><td> $0 . 3 8 3 \pm 0 . 0 3 1$ </td><td> $2 1 4 . 0 5 5 \pm 3 0 . 1 1 8$ </td><td> $0 . 6 2 5 \pm 0 . 0 4 1$ </td><td></td><td> $3 1 . 4 8 8 \pm 2 . 8 2 1$ </td></tr><tr><td>ResShift [48]</td><td> $1 8 . 0 2 2 \pm 0 . 0 9 5$ </td><td> $0 . 4 3 5 \pm 0 . 0 0 5$ </td><td> $0 . 1 4 1 \pm 0 . 0 0 0$ </td><td> $\mathbf { 2 6 . 4 6 5 \ : \pm { \ : 0 . 3 4 2 } }$ </td><td> $\mathbf { 0 . 2 8 8 \ : \pm { \ : 0 . 0 0 2 } }$ </td><td></td><td> $1 0 . 3 2 1 \pm 0 . 0 4 4$ </td></tr><tr><td>SS-UIE [49]</td><td> $1 5 . 1 2 2 \pm 0 . 3 7 6$ </td><td> $0 . 3 5 1 \pm 0 . 0 1 7$ </td><td> $0 . 2 1 0 \pm 0 . 0 1 1$ </td><td> $6 2 . 5 6 8 \pm 7 . 4 2 3$ </td><td> $0 . 6 6 4 \pm 0 . 0 1 1$ </td><td></td><td> $1 8 . 7 7 3 \pm 1 . 1 9 6$ </td></tr><tr><td>U-Shape Transformer [50]</td><td> $1 5 . 9 7 8 \pm 0 . 1 7 9$ </td><td> $0 . 3 9 5 \pm 0 . 0 0 7$ </td><td> $0 . 2 0 3 \pm 0 . 0 0 3$ </td><td> $6 5 . 0 2 8 \pm 1 . 4 0 8$ </td><td> $0 . 7 0 0 \pm 0 . 0 0 7$ </td><td></td><td> $1 5 . 7 9 8 \pm 0 . 0 4 9$ </td></tr><tr><td>KAUSTics-Net (ours)</td><td> $\mathbf { 1 9 . 4 6 9 \ : \pm { \ : 0 . 3 3 2 } }$ </td><td> $\mathbf { 0 . 5 5 4 \ : \pm { \ : 0 . 0 0 5 } }$ </td><td> $\mathbf { 0 . 1 3 1 } \pm \mathbf { 0 . 0 1 1 }$ </td><td> $2 9 . 5 9 8 \pm 6 . 6 8 5$ </td><td> $0 . 4 7 4 \pm 0 . 0 2 0$ </td><td></td><td> $\mathbf { 9 . 8 6 5 \ : \pm { \ : 1 . 0 0 5 } }$ </td></tr><tr><td colspan="8">No waves</td></tr><tr><td>Unprocessed input</td><td>10.526</td><td>0.411</td><td>0.342</td><td>143.097</td><td>0.581</td><td></td><td></td></tr><tr><td>CycleGAN [37]</td><td> $1 7 . 2 7 7 \pm 0 . 9 5 1$ </td><td> $0 . 5 1 8 \pm 0 . 0 3 2$ </td><td> $0 . 1 7 0 \pm 0 . 0 2 0$ </td><td> $4 5 . 9 4 2 \pm 1 1 . 1 0 7$ </td><td> $0 . 3 2 4 \pm 0 . 0 3 4$ </td><td></td><td>32.815</td></tr><tr><td>UGAN [27]</td><td> $1 0 . 8 8 4 \pm 0 . 1 5 4$ </td><td> $0 . 3 7 4 \pm \ : 0 . 0 0 9$ </td><td> $0 . 2 9 9 \pm 0 . 0 0 5$ </td><td> $9 7 . 4 8 5 \pm 3 . 2 3 6$ </td><td> $0 . 7 1 1 \pm 0 . 0 1 8$ </td><td></td><td> $1 3 . 0 6 7 \pm 1 . 8 8 8$ </td></tr><tr><td>FUnIE-GAN [38]</td><td> $1 7 . 6 3 2 \pm 0 . 0 8 1$ </td><td> $0 . 4 3 7 \pm 0 . 0 1 4$ </td><td> $0 . 1 6 7 \pm 0 . 0 0 1$ </td><td> $4 7 . 8 3 4 \pm 0 . 4 6 6$ </td><td> $0 . 4 2 5 \pm 0 . 0 2 0$ </td><td></td><td> $2 5 . 9 1 4 \pm 0 . 7 9 2$ </td></tr><tr><td>DeWaterGAN [23]</td><td> $1 2 . 4 3 4 \pm 1 . 2 9 7$ </td><td> $0 . 4 4 0 \pm 0 . 0 0 6$ </td><td> $0 . 3 1 6 \pm 0 . 0 1 1$ </td><td> $1 4 0 . 9 9 4 \pm 9 . 0 6 2$ </td><td> $0 . 5 0 4 \pm 0 . 0 5 9$ </td><td></td><td> $1 4 . 0 6 6 \pm 0 . 3 5 3$ </td></tr><tr><td>DiffWater [28]</td><td> $1 0 . 7 7 4 \pm 1 . 0 7 6$ </td><td> $0 . 3 5 7 \pm 0 . 0 3 5$ </td><td> $0 . 3 8 0 \pm 0 . 0 3 5$ </td><td> $2 1 4 . 5 6 8 \pm 3 3 . 2 8 7$ </td><td> $0 . 5 8 8 \pm 0 . 0 5 2$ </td><td></td><td> $2 8 . 6 9 2 \pm 1 . 9 3 8$ </td></tr><tr><td>ResShift [48]</td><td> $1 9 . 4 4 0 \pm 0 . 0 5 7$ </td><td> $0 . 5 4 6 \pm \ : 0 . 0 0 4$ </td><td> $0 . 1 2 2 \pm 0 . 0 0 2$ </td><td> $2 1 . 9 6 4 \pm 1 . 2 1 5$ </td><td> $\mathbf { 0 . 2 2 3 \ : \pm { \ : 0 . 0 0 1 } }$ </td><td></td><td> $3 1 . 5 1 0 \pm 3 . 0 6 1$ </td></tr><tr><td>SS-UIE [49]</td><td> $1 6 . 1 2 9 \pm 0 . 2 6 7$ </td><td> $0 . 4 5 1 \pm 0 . 0 1 4$ </td><td> $0 . 1 9 1 \pm 0 . 0 1 0$ </td><td> $5 4 . 6 3 5 \pm 7 . 8 4 2$ </td><td> $0 . 5 6 8 \pm 0 . 0 0 7$ </td><td></td><td> $9 . 2 3 9 \pm 0 . 1 4 4$ </td></tr><tr><td>U-Shape Transformer [50]</td><td> $1 7 . 3 2 1 \pm 0 . 2 4 5$ </td><td> $0 . 4 7 1 \pm 0 . 0 1 5$ </td><td> $0 . 1 7 3 \pm 0 . 0 0 5$ </td><td> $4 9 . 2 0 1 \pm 3 . 4 8 0$ </td><td> $0 . 6 3 7 \pm 0 . 0 1 6$ </td><td></td><td> $1 7 . 6 5 9 \pm 1 . 3 1 4$ </td></tr><tr><td>KAUSTics-Net (ours)</td><td> $\mathbf { 2 0 . 7 7 0 \ : \pm { \ : 0 . 0 5 1 } }$ </td><td> $\mathbf { 0 . 6 5 9 \ : \pm { \ : 0 . 0 0 6 } }$ </td><td> $\mathbf { 0 . 1 0 8 \ : \pm { \ : 0 . 0 0 2 } }$ </td><td></td><td></td><td></td><td> $1 3 . 6 5 9 \pm 0 . 6 0 3$ </td></tr><tr><td>No sunglint</td><td></td><td></td><td></td><td> $\mathbf { 1 8 . 9 0 1 \ : \pm { \ : 0 . 8 0 7 } }$ </td><td> $0 . 3 7 0 \pm 0 . 0 1 2$ </td><td></td><td> $\mathbf { 8 . 1 2 4 \ : \pm { \ : 0 . 2 0 0 } }$ </td></tr><tr><td colspan="8"></td></tr><tr><td>Unprocessed input</td><td>10.492</td><td>0.374</td><td>0.342</td><td>142.779</td><td>0.643</td><td></td><td>32.785</td></tr><tr><td>CycleGAN [37]</td><td> $1 7 . 0 5 0 \pm 0 . 9 0 5$ </td><td> $0 . 4 5 0 \pm 0 . 0 2 8$ </td><td> $0 . 1 7 3 \pm 0 . 0 2 2$ </td><td> $4 7 . 0 3 0 \pm 1 3 . 9 6 5$ </td><td> $0 . 3 8 1 \pm 0 . 0 2 7$ </td><td></td><td> $1 3 . 1 9 9 \pm 2 . 1 5 3$ </td></tr><tr><td>UGAN [27]</td><td> $1 0 . 8 3 7 \pm 0 . 1 4 5$ </td><td> $0 . 3 5 1 \pm 0 . 0 0 7$ </td><td> $0 . 3 0 1 \pm 0 . 0 0 5$ </td><td> $9 8 . 1 7 0 \pm 3 . 1 1 2$ </td><td> $0 . 7 6 6 \pm 0 . 0 1 4$ </td><td></td><td> $2 5 . 9 8 3 \pm 0 . 7 9 3$ </td></tr><tr><td>FUnIE-GAN [38]</td><td> $1 7 . 4 4 4 \pm 0 . 0 2 0$ </td><td> $0 . 4 0 5 \pm 0 . 0 1 3$ </td><td> $0 . 1 6 9 \pm 0 . 0 0 0$ </td><td> $4 8 . 4 8 3 \pm 0 . 4 6 5$ </td><td> $0 . 4 7 8 \pm 0 . 0 1 9$ </td><td></td><td> $1 4 . 0 9 2 \pm 0 . 3 8 6$ </td></tr><tr><td>DeWaterGAN [23]</td><td> $1 2 . 2 6 3 \pm 1 . 1 8 0$ </td><td> $0 . 3 9 6 \pm 0 . 0 0 9$ </td><td> $0 . 3 1 8 \pm 0 . 0 0 9$ </td><td> $1 4 2 . 0 6 6 \pm 1 0 . 4 5 7$ </td><td> $0 . 5 6 5 \pm 0 . 0 5 9$ </td><td></td><td> $2 8 . 9 0 7 \pm 1 . 7 4 6$ </td></tr><tr><td>DiffWater [28]</td><td> $1 0 . 6 5 0 \pm 1 . 0 0 2$ </td><td> $0 . 3 3 9 \pm 0 . 0 3 0$ </td><td> $0 . 3 8 4 \pm 0 . 0 3 4$ </td><td> $2 1 6 . 6 8 4 \pm 3 3 . 1 1 2$ </td><td> $0 . 6 1 6 \pm 0 . 0 4 3$ </td><td></td><td> $3 1 . 6 7 6 \pm 2 . 9 8 7$ </td></tr><tr><td>ResShift [48]</td><td> $1 8 . 4 1 4 \pm 0 . 1 5 5$ </td><td> $0 . 4 6 4 \pm 0 . 0 0 2$ </td><td> $0 . 1 3 4 \pm 0 . 0 0 4$ </td><td> $2 3 . 5 2 3 \pm 3 . 2 8 8$ </td><td> $\mathbf { 0 . 2 6 5 \ : \pm { \ : 0 . 0 0 2 } }$ </td><td></td><td> $9 . 8 4 2 \pm 0 . 3 9 3$ </td></tr><tr><td>SS-UIE [49]</td><td> $1 5 . 9 3 5 \pm 0 . 2 8 9$ </td><td> $0 . 4 1 8 \pm 0 . 0 1 4$ </td><td> $0 . 1 9 4 \pm 0 . 0 0 9$ </td><td> $5 5 . 6 3 8 \pm 7 . 2 5 0$ </td><td> $0 . 6 2 3 \pm 0 . 0 0 3$ </td><td></td><td> $1 7 . 7 8 4 \pm 1 . 2 4 7$ </td></tr><tr><td>U-Shape Transformer [50]</td><td> $1 7 . 1 4 8 \pm 0 . 1 6 4$ </td><td> $0 . 4 4 9 \pm 0 . 0 1 1$ </td><td> $0 . 1 7 4 \pm 0 . 0 0 4$ </td><td> $4 9 . 3 3 1 \pm 2 . 5 4 1$ </td><td> $0 . 6 7 9 \pm 0 . 0 1 4$ </td><td></td><td> $1 3 . 6 5 7 \pm 0 . 4 8 3$ </td></tr><tr><td>KAUSTics-Net (ours)</td><td> ${ \bf 2 0 . 1 5 7 \pm 0 . 0 5 7 }$ </td><td> $\mathbf { 0 . 5 9 2 \ : \pm { \ : 0 . 0 0 3 } }$ </td><td> $\mathbf { 0 . 1 1 5 \ : \pm { \ : 0 . 0 0 2 } }$ </td><td> $\mathbf { 2 0 . 4 7 5 \ : \pm { \ : 2 . 0 4 3 } }$ </td><td> $0 . 4 3 7 \pm 0 . 0 1 1$ </td><td></td><td> $\mathbf { 8 . 4 7 7 \pm 0 . 3 1 7 }$ </td></tr></table>

## C. Ablation Study

To assess the contribution of individual architectural components and training objectives, we conducted an ablation study on the KAUSTics dataset. Each configuration was evaluated on the same 1,200 held-out observations across three independent training seeds. Table VIII reports the effects of removing FiLM, cross-attention, the conditional encoder, and individual reconstruction losses, using the full KAUSTics-Net configuration as the reference.

indicates that structural supervision plays an important role in the reconstruction objective. In contrast, removing the lowfrequency, UV, RGB reconstruction, or CIELAB loss produced mixed results, with some configurations improving individual metrics. For example, removing the UV loss increased PSNR to 32.181 dB, while removing the CIELAB loss slightly improved SSIM to 0.851. These findings indicate that the complete loss formulation does not optimize every metric individually.

Among the architectural components, removing FiLM or the conditional encoder slightly reduced the mean PSNR and SSIM, indicating a modest benefit from multi-scale conditioning. Removing cross-attention produced a marginally higher PSNR (31.813 versus 31.715 dB) but lower SSIM (0.846 versus 0.849). These differences were small relative to the variability across seeds, suggesting that the contributions of individual conditioning mechanisms are not consistently reflected across all metrics.

Conditioning dropout was evaluated separately using a matched classifier-free guidance scale of 1 for both configurations. Removing dropout preserved the mean SSIM (0.849) while improving PSNR, LPIPS, and $\Delta E _ { 0 0 }$ . Thus, the ablation does not establish an independent advantage from conditioning dropout under this sampling configuration.

The most pronounced degradation occurred when removing the SSIM loss, which reduced PSNR from 31.715 to 29.994 dB and SSIM from 0.849 to 0.821, while increasing LPIPS from 0.145 to 0.166 and $\Delta E _ { 0 0 }$ from 3.080 to 3.660. This

Overall, the ablation results identify the SSIM loss as the component with the clearest contribution to reconstruction performance. The remaining architectural and loss components exhibit smaller or metric-dependent effects, highlighting the importance of assessing the proposed configuration across multiple restoration criteria.

NO-REFERENCE IMAGE QUALITY METRICS ON SEA-UNDISTORT UNDER THE ALL-EFFECTS, NO-WAVES, AND NO-SUNGLINT CONFIGURATIONS. EACH CONFIGURATION COMPRISES 120 HELD-OUT SCENES. VALUES ARE REPORTED AS MEAN ± STANDARD DEVIATION ACROSS THREE TRAINING SEEDS. SHARPNESS AND ENTROPY ARE DESCRIPTIVE INDICATORS WITHOUT PERFORMANCE RANKING.

TABLE VII

$$
\mathbf { 6 . 0 7 1 \pm 0 . 3 3 0 }
$$

$$
7 . 9 0 5 \pm 0 . 3 1 8
$$

$$
1 6 . 7 6 8 \pm 3 . 2 8 8
$$

$$
2 3 . 9 3 9 \pm 4 . 0 9 2
$$

$$
1 6 . 3 8 2 \pm 2 . 8 9 2
$$

$$
4 7 . 9 9 9 \pm 2 . 6 2 3
$$

$$
2 1 . 0 4 2 \pm 3 . 5 9 2
$$

$$
1 0 0 . 1 1 6 \pm 2 . 5 5 8
$$

$$
1 . 8 0 8 \pm 0 . 1 5 2
$$

$$
0 . 4 5 6 \pm 0 . 0 2 8
$$

$$
\mathbf { 9 . 9 3 9 } \pm \mathbf { 6 . 3 0 9 }
$$

$$
1 . 0 1 6 \pm 0 . 0 0 3
$$

$$
3 . 1 5 5 \pm 0 . 0 8 3
$$

$$
1 4 1 4 . 6 3 6 \pm 5 0 7 . 0 4 0
$$

$$
0 . 1 6 8 \pm 0 . 0 1 5
$$

$$
7 4 . 3 4 8 \pm 4 . 0 0 0
$$

$$
2 . 8 2 1 \pm 0 . 1 0 5
$$

$$
1 3 6 . 4 3 5 \pm 1 3 . 9 3 6
$$

$$
7 . 8 7 0 \pm 0 . 2 0 7
$$

$$
6 . 0 3 5 \pm 0 . 0 6 1
$$

$$
3 4 . 2 3 2 \pm 0 . 6 9 7
$$

$$
3 1 . 1 9 6 \pm 0 . 5 7 6
$$

$$
8 . 6 9 3 \pm 1 . 5 2 2
$$

$$
1 2 . 6 6 8 \pm 2 . 3 5 9
$$

$$
6 7 . 7 5 6 \pm 1 . 0 0 8
$$

$$
1 . 7 0 4 \pm 0 . 0 3 8
$$

$$
9 3 7 . 3 0 2 \pm 6 8 . 3 3 9
$$

$$
2 . 9 5 8 \pm 0 . 0 1 2
$$

$$
9 . 0 3 2 \pm 0 . 0 1 8
$$

$$
5 9 . 5 4 9 \pm 1 3 . 8 3 9
$$

$$
3 4 . 8 7 3 \pm 0 . 1 8 6
$$

$$
0 . 3 2 5 \pm 0 . 0 1 9
$$

$$
1 8 . 6 4 5 \pm 0 . 5 8 4
$$

$$
5 . 6 0 9 \pm 0 . 0 7 8
$$

$$
1 . 5 6 1 \pm 0 . 3 4 5
$$

$$
3 . 0 1 9 \pm 0 . 1 1 2
$$

$$
3 3 . 3 0 5 \pm 2 . 1 6 7
$$

$$
{ \bf 4 0 . 7 4 6 \pm 0 . 2 9 8 }
$$

$$
0 . 4 7 7 \pm 0 . 0 6 6
$$

$$
7 . 4 6 5 \pm 0 . 6 2 6
$$

$$
5 4 4 . 3 8 5 \pm 1 8 7 . 1 6 3
$$

$$
3 2 . 4 2 5 \pm 2 . 8 3 2
$$

$$
4 . 8 0 7 \pm 0 . 6 1 3
$$

$$
\mathbf { 2 . 4 2 5 \ : \pm { 0 . 0 0 3 } }
$$

$$
7 7 . 1 0 5 \pm 1 . 0 6 5
$$

$$
\mathbf { 0 . 5 2 2 \ : \pm { \ : 0 . 0 0 3 } }
$$

$$
3 . 1 1 4 \pm 0 . 0 0 6
$$

$$
5 7 . 0 2 2 \pm 0 . 7 7 2
$$

$$
1 0 . 4 0 9 \pm 0 . 1 7 6
$$

$$
3 1 0 4 . 0 4 8 \pm 1 6 9 . 8 5 5
$$

$$
7 4 . 1 4 4 \pm 3 . 5 7 8
$$

$$
1 . 1 3 2 \pm 0 . 0 4 3
$$

$$
0 . 2 6 0 \pm 0 . 0 0 9
$$

$$
6 . 3 2 0 \pm 0 . 0 1 7
$$

$$
2 . 8 3 0 \pm 0 . 0 5 2
$$

$$
1 0 4 . 4 9 0 \pm 1 . 7 3 1
$$

$$
5 3 8 . 7 0 1 \pm 1 4 3 . 1 8 0
$$

$$
3 3 . 9 5 9 \pm 3 . 7 4 9
$$

$$
1 . 2 5 4 \pm 0 . 0 1 2
$$

$$
3 3 . 7 0 7 \pm 1 . 6 7 0
$$

$$
7 . 8 8 9 \pm 0 . 3 6 0
$$

$$
0 . 2 9 8 \pm 0 . 0 1 3
$$

$$
7 9 . 9 9 9 \pm 2 . 7 1 7
$$

$$
5 . 7 9 5 \pm 0 . 0 4 7
$$

$$
2 . 5 9 3 \pm 0 . 0 4 0
$$

$$
1 . 3 4 6 \pm 0 . 0 2 4
$$

$$
0 . 3 5 2 \pm 0 . 0 2 7
$$

$$
7 9 . 1 9 5 \pm 1 0 . 8 1 4
$$

$$
5 . 4 5 2 \pm 0 . 1 4 0
$$

$$
5 . 7 0 6 \pm 0 . 1 2 4
$$

$$
\mathbf { 5 . 6 8 9 \ : \pm { \ : 0 . 1 8 0 } }
$$

$$
1 1 . 3 2 5 \pm 2 . 2 3 1
$$

$$
1 5 . 4 1 9 \pm 2 . 2 3 9
$$

$$
8 . 2 8 5 \pm 0 . 4 9 7
$$

$$
4 6 . 0 0 9 \pm 2 . 1 1 2
$$

$$
1 8 . 5 6 9 \pm 4 . 0 1 3
$$

$$
0 . 4 7 9 \pm 0 . 0 3 4
$$

$$
1 . 7 0 5 \pm 0 . 1 3 5
$$

$$
3 . 2 3 5 \pm 0 . 1 0 1
$$

$$
2 1 . 2 2 9 \pm 4 . 2 1 4
$$

$$
1 1 9 3 . 1 3 4 \pm 4 1 9 . 7 5 2
$$

$$
9 5 . 9 8 2 \pm 2 . 7 2 5
$$

$$
8 . 5 8 2 \pm 0 . 6 7 6
$$

$$
\mathbf { 7 . 8 8 0 \ : \pm { \ : 7 . 2 8 9 } }
$$

$$
1 . 0 1 2 \pm 0 . 0 0 2
$$

$$
1 2 . 1 0 3 \pm 2 . 4 3 5
$$

$$
0 . 1 6 0 \pm 0 . 0 1 0
$$

$$
5 . 9 9 1 \pm 0 . 0 9 8
$$

$$
1 3 8 . 8 1 1 \pm 1 3 . 1 3 7
$$

$$
7 2 . 5 6 0 \pm 4 . 9 7 0
$$

$$
7 . 1 6 7 \pm 0 . 3 6 1
$$

$$
\mathbf { 3 . 4 3 9 } \pm \mathbf { 0 . 0 1 8 }
$$

$$
0 . 3 1 1 \pm 0 . 0 2 0
$$

$$
2 9 . 6 3 5 \pm 2 . 3 7 3
$$

$$
3 0 . 1 7 3 \pm 0 . 7 0 9
$$

$$
5 . 8 7 4 \pm 0 . 0 4 1
$$

$$
5 4 8 . 0 5 2 \pm 5 7 . 4 6 8
$$

$$
6 5 . 0 6 0 \pm 2 . 0 4 3
$$

$$
1 . 4 4 0 \pm 0 . 0 7 9
$$

$$
0 . 3 1 2 \pm 0 . 0 1 3
$$

$$
2 . 8 8 1 \pm 0 . 0 2 4
$$

$$
8 . 7 6 2 \pm 1 . 5 6 7
$$

$$
1 2 . 8 3 9 \pm 2 . 4 9 9
$$

$$
6 . 0 7 3 \pm 0 . 1 0 1
$$

$$
2 8 8 . 0 5 7 \pm 5 4 . 9 7 9
$$

$$
5 . 1 2 4 \pm 0 . 0 7 5
$$

$$
9 . 5 1 6 \pm 0 . 1 3 4
$$

$$
1 . 5 7 1 \pm 0 . 3 6 2
$$

$$
3 4 . 5 4 5 \pm 0 . 7 7 2
$$

$$
2 0 . 0 7 9 \pm 0 . 8 1 4
$$

$$
6 . 8 5 9 \pm 0 . 7 6 3
$$

$$
2 7 . 8 3 4 \pm 2 . 6 0 4
$$

$$
\mathbf { 4 2 . 7 3 5 \ : \pm { \ : 0 . 6 8 2 } }
$$

$$
4 . 8 1 3 \pm 0 . 6 2 2
$$

$$
\mathbf { 2 . 4 6 4 \ : \pm { 0 . 0 0 9 } }
$$

$$
\mathbf { 0 . 5 4 0 \ : \pm { \ : 0 . 0 0 5 } }
$$

$$
3 0 . 5 6 4 \pm 2 . 6 2 9
$$

$$
3 . 2 5 4 \pm 0 . 0 2 2
$$

$$
1 1 . 3 8 9 \pm 0 . 2 6 1
$$

$$
7 1 . 3 6 9 \pm 2 . 0 9 6
$$

$$
5 6 . 5 3 5 \pm 0 . 1 3 8
$$

$$
3 0 0 0 . 8 4 7 \pm 1 5 7 . 6 1 2
$$

$$
7 8 . 8 8 2 \pm 3 . 6 2 9
$$

$$
1 . 1 0 5 \pm 0 . 0 2 5
$$

$$
0 . 2 7 6 \pm 0 . 0 1 4
$$

$$
1 0 1 . 9 1 8 \pm 0 . 9 2 0
$$

$$
1 . 2 2 1 \pm 0 . 0 2 4
$$

$$
2 . 9 5 5 \pm 0 . 0 6 5
$$

$$
0 . 2 7 1 \pm 0 . 0 0 9
$$

$$
2 . 6 3 7 \pm 0 . 0 4 3
$$

$$
7 3 . 0 9 4 \pm 1 2 . 9 8 9
$$

$$
5 . 5 5 8 \pm 0 . 0 3 2
$$

$$
5 . 2 1 3 \pm 0 . 1 3 4
$$

$$
\mathbf { 5 . 6 8 3 \ : \pm { \ : 0 . 3 5 0 } }
$$

$$
1 4 . 0 6 8 \pm 2 . 5 6 8
$$

$$
8 . 1 6 7 \pm 0 . 4 1 0
$$

$$
1 8 . 0 4 7 \pm 4 . 3 0 8
$$

$$
1 5 . 7 8 9 \pm 3 . 1 9 5
$$

$$
8 . 8 0 0 \pm 0 . 6 9 9
$$

$$
\mathbf { 7 . 6 3 0 \ : \pm { 7 . 3 2 2 } }
$$

$$
6 . 7 7 4 \pm 0 . 2 2 6
$$

$$
6 . 0 4 1 \pm 0 . 1 3 3
$$

$$
1 1 . 8 6 8 \pm 2 . 1 7 4
$$

$$
2 8 . 3 7 4 \pm 2 . 4 1 7
$$

$$
1 1 9 2 . 9 5 2 \pm 4 6 3 . 4 5 1
$$

$$
1 . 7 4 6 \pm 0 . 1 2 2
$$

$$
0 . 4 6 8 \pm 0 . 0 3 2
$$

$$
3 . 1 8 3 \pm 0 . 1 0 5
$$

$$
9 . 0 0 7 \pm 1 . 6 8 1
$$

$$
2 4 . 9 8 4 \pm 1 . 2 6 5
$$

$$
7 5 . 2 6 8 \pm 4 . 8 1 4
$$

$$
1 1 . 0 2 3 \pm 0 . 3 4 9
$$

$$
1 . 0 1 7 \pm 0 . 0 0 2
$$

$$
1 2 . 4 5 8 \pm 2 . 7 8 2
$$

$$
9 . 1 0 0 \pm 0 . 0 6 8
$$

$$
6 . 8 1 9 \pm 0 . 4 1 4
$$

$$
0 . 1 5 8 \pm 0 . 0 1 3
$$

$$
{ \bf 5 . 1 8 8 \pm 0 . 8 1 2 }
$$

$$
7 8 . 5 3 0 \pm 2 . 6 3 6
$$

$$
3 6 . 3 1 2 \pm 4 . 3 8 6
$$

$$
8 . 3 5 5 \pm 0 . 4 6 1
$$

$$
7 0 . 3 9 4 \pm 1 . 6 1 6
$$

$$
1 . 0 8 2 \pm 0 . 0 2 4
$$

$$
2 9 . 1 4 9 \pm 2 . 6 3 9
$$

$$
1 9 . 7 7 5 \pm 0 . 3 6 7
$$

$$
3 5 . 0 2 2 \pm 0 . 4 6 0
$$

$$
5 8 . 7 1 4 \pm 0 . 3 3 7
$$

$$
1 0 4 . 8 5 5 \pm 1 . 7 1 9
$$

$$
1 3 4 . 4 5 8 \pm 1 3 . 8 2 3
$$

$$
2 7 . 8 0 9 \pm 2 . 0 4 3
$$

$$
7 6 . 9 6 9 \pm 2 . 4 0 4
$$

$$
0 . 3 1 3 \pm 0 . 0 2 4
$$

$$
1 . 3 7 8 \pm 0 . 0 3 1
$$

$$
\mathbf { 4 0 . 9 9 1 \pm 0 . 1 9 9 }
$$

$$
5 . 9 1 9 \pm 0 . 0 4 0
$$

$$
3 4 . 8 1 7 \pm 1 . 7 2 5
$$

$$
1 . 5 4 4 \pm 0 . 3 3 9
$$

$$
\mathbf { 3 . 5 1 7 \ : \pm { \ : 0 . 0 2 7 } }
$$

$$
4 9 6 . 5 3 3 \pm 5 4 . 7 3 4
$$

$$
\mathbf { 2 . 4 6 0 \ : \pm { \ : 0 . 0 3 2 } }
$$

$$
1 . 1 0 0 \pm 0 . 0 2 2
$$

$$
0 . 3 0 6 \pm 0 . 0 0 8
$$

$$
1 . 2 9 0 \pm 0 . 0 0 9
$$

$$
0 . 4 7 6 \pm 0 . 0 6 8
$$

$$
1 . 3 6 9 \pm 0 . 0 1 4
$$

$$
\mathbf { 0 . 5 2 2 \ : \pm { \ : 0 . 0 0 4 } }
$$

$$
0 . 2 6 6 \pm 0 . 0 2 1
$$

$$
0 . 2 8 7 \pm 0 . 0 1 1
$$

$$
2 1 9 . 4 8 8 \pm 4 3 . 9 4 7
$$

$$
5 . 9 8 4 \pm 0 . 1 0 9
$$

$$
3 . 0 4 7 \pm 0 . 1 1 7
$$

$$
3 . 1 7 2 \pm 0 . 0 1 7
$$

$$
5 1 9 . 5 4 6 \pm 1 8 4 . 0 6 3
$$

$$
2 . 8 2 7 \pm 0 . 0 4 9
$$

$$
2 . 5 9 0 \pm 0 . 0 4 6
$$

$$
5 . 1 9 2 \pm 0 . 1 0 2
$$

$$
2 . 7 9 5 \pm 0 . 0 2 5
$$

$$
3 1 0 2 . 7 8 9 \pm 1 6 7 . 1 6 3
$$

$$
4 . 7 2 7 \pm 0 . 6 3 5
$$

$$
2 0 7 . 2 1 7 \pm 3 3 . 0 6 4
$$

$$
6 . 2 7 7 \pm 0 . 0 1 6
$$

$$
6 7 . 3 4 8 \pm 1 4 . 1 2 4
$$

$$
5 . 5 3 5 \pm 0 . 0 1 5
$$

$$
1 9 5 . 4 7 0 \pm 2 9 . 1 7 1
$$

$$
5 . 1 5 9 \pm 0 . 1 7 9
$$

$$
5 . 6 7 6 \pm 0 . 1 1 1
$$

TABLE VIII  
ABLATION STUDY OF KAUSTICS-NET ON 1,200 HELD-OUT KAUSTICS OBSERVATIONS. RESULTS ARE REPORTED AS MEAN ± STANDARD DEVIATION ACROSS THREE TRAINING SEEDS. INDIVIDUAL ARCHITECTURAL COMPONENTS AND LOSS TERMS ARE REMOVED WHILE RETAINING THE REMAINING CONFIGURATION. THE COMPONENT ABLATIONS USE GUIDANCE SCALE 2, WHILE THE MATCHED CONDITIONING-DROPOUT COMPARISON USES GUIDANCE SCALE 1 FOR BOTH CONFIGURATIONS. BOLD VALUES INDICATE THE BEST MEAN WITHIN EACH COMPARISON BLOCK.
<table><tr><td>Method</td><td>mPSNR ↑</td><td>mSSIM ↑</td><td>LPIPS↓</td><td>∆E00 ↓</td></tr><tr><td colspan="5">Component removals: guidance 2</td></tr><tr><td>Ours</td><td> $\overline { { 3 1 . 7 1 5 \pm 0 . 4 3 9 } }$ </td><td> $\overline { { 0 . 8 4 9 \ : \pm \ : 0 . 0 0 2 } }$ </td><td> $\overline { { 0 . 1 4 5 \ : \pm \ : 0 . 0 0 7 } }$ </td><td> $\overline { { 3 . 0 8 0 \pm 0 . 1 3 1 } }$ </td></tr><tr><td>Without FiLM</td><td> $3 1 . 0 8 9 \pm 0 . 2 1 6$ </td><td> $0 . 8 4 5 \pm 0 . 0 0 5$ </td><td>0.149 ± 0.003</td><td>3.131 ± 0.072</td></tr><tr><td>Without cross-attention</td><td> $3 1 . 8 1 3 \pm 0 . 7 6 0$ </td><td>0.846 ± 0.006</td><td> $0 . 1 4 6 \pm 0 . 0 0 4$ </td><td> $3 . 0 3 7 \pm 0 . 2 8 9$ </td></tr><tr><td>Without conditional encoder</td><td> $3 1 . 2 4 0 \pm 0 . 4 0 6$ </td><td> $0 . 8 4 0 \pm 0 . 0 1 3$ </td><td> $0 . 1 4 6 \pm 0 . 0 0 7$ </td><td> $3 . 1 2 8 \pm 0 . 1 2 2$ </td></tr><tr><td>Without low-frequency loss</td><td>31.838 ± 0.240</td><td>0.847 ± 0.006</td><td>0.146 ± 0.003</td><td>3.007 ± 0.143</td></tr><tr><td>Without UV loss</td><td> $\mathbf { 3 2 . 1 8 1 \ : \pm { \ : 0 . 1 0 3 } }$ </td><td> $0 . 8 4 4 \pm \ : 0 . 0 0 9$ </td><td> $0 . 1 4 2 \pm 0 . 0 0 2$ </td><td> $3 . 0 4 8 \pm 0 . 1 2 6$ </td></tr><tr><td>Without RGB reconstruction</td><td> $3 2 . 0 6 2 \pm 0 . 2 3 4$ </td><td> $0 . 8 4 8 \pm 0 . 0 0 8$ </td><td> $0 . 1 4 8 \pm 0 . 0 0 6$ </td><td> $\mathbf { 2 . 9 9 1 \pm 0 . 0 8 2 }$ </td></tr><tr><td>Without SSIM loss</td><td>29.994 ± 0.247</td><td>0.821 ± 0.004</td><td>0.166 ± 0.004</td><td> $3 . 6 6 0 \pm 0 . 2 5 7$ </td></tr><tr><td>Without Lab loss</td><td> $3 1 . 9 1 6 \pm 0 . 6 1 0$ </td><td> $\mathbf { 0 . 8 5 1 \ : \pm { \ : 0 . 0 0 2 } }$ </td><td> $\mathbf { 0 . 1 4 2 \ : \pm { \ : 0 . 0 0 6 } }$ </td><td> $2 . 9 9 1 \pm 0 . 1 5 8$ </td></tr><tr><td colspan="5">Matched dropout control: guidance 1</td></tr><tr><td>Ours, CFG 1</td><td> $\overline { { 2 8 . 9 2 4 \pm 0 . 5 9 2 } }$ </td><td> $\overline { { 0 . 8 4 9 \ : \pm \ : 0 . 0 0 4 } }$ </td><td> $\overline { { 0 . 1 4 7 \ : \pm \ : 0 . 0 0 3 } }$ </td><td> $\overline { { 4 . 1 4 3 \ : \pm \ : 0 . 1 3 7 } }$ </td></tr><tr><td>Without conditioning dropout, CFG 1</td><td> $\mathbf { 3 0 . 7 8 9 \ : \pm { \ : 0 . 2 6 0 } }$ </td><td> $\mathbf { 0 . 8 4 9 \ : \pm { \ : 0 . 0 0 1 } }$ </td><td> $\mathbf { 0 . 1 3 4 } \pm \mathbf { 0 . 0 0 1 }$ </td><td> $\mathbf { 3 . 3 4 0 \ : \pm { \ : 0 . 0 6 3 } }$ </td></tr></table>

## D. Real-UAV Downstream Evaluation

All restoration methods were applied to identical 1000 × 1000 pixel image contexts resized to $2 5 6 \times 2 5 6$ pixels, using their Sea-Undistort-adapted seed-1 checkpoints without further training. To reduce differences in large-scale color appearance, the model outputs received the same low-frequency color harmonization based on the distorted input, using a Gaussian smoothing scale of $\sigma = 5$ pixels and correction strength α = 1. The input and temporal reference remained unchanged, and no geometric registration was applied. This processing was specific to the real-UAV case study and was not used in the quantitative benchmark evaluations.

To examine the behavior of the restoration methods on real through-water imagery, we conducted an exploratory case study using a UAV frame from the controlled shallow-water survey of Scilla et al. [12]. The comparison reference was generated using the temporal computer-vision correction approach developed in that study, which combines information from 60 consecutive frames. This image provides a low-distortion visual reference but does not constitute independently validated water-free ground truth. Two regions containing a ground control point (GCP) and a bucket, located at approximately 0.600 and 0.900 m water depth, respectively, were selected for evaluation.

Figure 8 compares automatic segmentation regions produced by the Segment Anything Model (SAM) and local features detected using the scale-invariant feature transform (SIFT). SAM ViT-B was applied with a 32 × 32 point grid, predicted-IoU threshold of 0.88, and stability threshold of 0.95. SIFT used a contrast threshold of 0.040, with identical detector settings across all images. These diagnostics characterize segmentation behavior and local feature detection without requiring manually annotated object masks.

![](images/3743e26a4a2251d5ebfecf2b715afb224ae4d9b2589782c19e85898f8ceb6155.jpg)  
Fig. 8. Real-UAV structural diagnostics for the GCP (0.600 m depth) and bucket (0.900 m depth). Columns show the temporal computer-vision reference, distorted input, and eight benchmark restorations alongside KAUSTics-Net. Each object is evaluated using SAM segmentation regions and SIFT keypoints. All model outputs receive identical low-frequency color harmonization (α = 1, σ = 5 pixels), while the input and reference remain unchanged. Red boxes identify corresponding object-centered regions, and insets enlarge these regions for visualization. Histograms show the spatial distribution of SIFT keypoints using 20 bins and a common count scale within each object. SAM colors distinguish automatically generated regions and do not represent semantic classes. All methods use identical SAM and SIFT settings.

For the GCP region, SIFT detected 92 keypoints in the distorted input, 219 in the KAUSTics-Net reconstruction, and 874 in the temporal reference. The corresponding counts for the bucket were 19, 152, and 347. Although KAUSTics-Net increased the number of detected features relative to the distorted input, substantial differences remained compared with the reference. Other restoration methods exhibited different feature densities, demonstrating that the effect of image restoration on keypoint detection depends on the model and scene content.

To further examine feature correspondence, Fig. 9 compares SIFT descriptor matches between the restored images and the temporal reference. Using mutual nearest-neighbor matching with a bidirectional ratio threshold of 0.75, KAUSTics-Net produced five putative matches for the GCP, compared with three for the distorted input, and one for the bucket, compared with none for the input. UGAN produced ten matches in each region, while DiffWater produced no matches despite detecting local features. These results indicate that increased keypoint density does not necessarily correspond to increased descriptor agreement with the reference.

The SAM results also varied among methods, with changes in the number and spatial distribution of automatically generated regions (Fig. 8). However, these regions do not represent validated semantic classes, and no independent segmentation annotations were available to quantify accuracy. Similarly, the

![](images/b21426c4c51bea6fcf11fe21843e30e4e8373014c7acc8ec5ed320d3f697828d.jpg)  
Fig. 9. SIFT feature correspondence between the distorted input or restored GCP images and the temporal computer-vision reference. Each panel shows the evaluated image on the left and the common reference on the right. Orange lines indicate mutual nearest-neighbor descriptor matches obtained using a bidirectional ratio threshold of 0.75. SIFT features were extracted with a contrast threshold of 0.040. Match counts represent putative correspondences without RANSAC-based geometric verification. No spatial registration was applied to the restored images.

SIFT correspondences were not geometrically verified using RANSAC or an equivalent consistency test. Consequently, the observed changes cannot establish improvements in object localization, geometric reconstruction, or habitat mapping accuracy.

Overall, this case study illustrates how restoration can alter the outputs of subsequent computer-vision operations on real UAV imagery. The results are descriptive and limited to two selected regions from one survey frame. A broader evaluation with independent geometric measurements and annotated seabed features is required to determine whether these changes translate into improved remote-sensing products.

## V. DISCUSSION

## A. Generalization and Limitations

Obtaining paired distorted and reference UAV imagery of the same shallow-water scene remains challenging because water-surface conditions can change rapidly, while repeated acquisitions under comparable illumination, tide, and viewing geometry are difficult to achieve. Consequently, learning-based restoration methods frequently rely on synthetic data, unpaired training, or domain adaptation [67], [68], [69], [70]. KAUS-Tics addresses this data limitation by combining real UAV imagery with rendered caustics and refraction, complemented by empirical color-transfer and blur augmentations. This approach enables controlled restoration experiments while retaining the textures and spatial characteristics of real shallow-water UAV observations [16], [12].

The revised evaluation demonstrates improved reconstruction fidelity on held-out KAUSTics images and substantial gains following adaptation to Sea-Undistort. However, these findings have distinct implications for generalization. The KAUSTics test set contains 1,200 degraded observations derived from only three source UAV images, and the sourceimage-disjoint partition does not guarantee geographic independence between potentially overlapping survey footprints. Similarly, the Sea-Undistort experiments evaluate one independently generated synthetic dataset, with the main improvements obtained after dataset-specific fine-tuning. These results establish performance under the evaluated conditions but do not demonstrate reliable zero-shot transfer to arbitrary coastal environments, sensors, or acquisition geometries.

Additional limitations arise from the dataset construction. The KAUSTics reference images were acquired through water under relatively calm conditions and may retain residual geometric and radiometric distortions. Furthermore, the Blender simulations use planar seabed textures and selected watersurface configurations, while blur and color transfer provide empirical approximations of water-column degradation. The resulting observations do not reproduce the complete interaction between three-dimensional bathymetry, water constituents, multiple scattering, and sensor radiometry. Consequently, reconstruction accuracy is measured relative to the available reference appearance, and the recovery of absolute seabed geometry or intrinsic reflectance remains unverified.

Severe distortions introduce a further limitation because sun glint, scattering, and wave-induced displacement can obscure information needed for faithful reconstruction [26]. Learning-based models may generate plausible structures in regions where the observations provide insufficient evidence, potentially reflecting correlations learned from the training distribution [71], [72]. This risk reinforces the importance of evaluating restored imagery against independent geometric and ecological measurements before using it for quantitative remote-sensing applications.

## B. Perceptual Metrics, Pixel Fidelity, and Task Utility

The revised experiments highlight differences between reconstruction fidelity and predicted perceptual quality. On KAUSTics, KAUSTics-Net achieved the highest mean PSNR and SSIM and the lowest LPIPS and $\Delta E _ { 0 0 }$ among the evaluated methods. Following Sea-Undistort adaptation, it also obtained the highest PSNR and SSIM across all three distortion configurations. However, the no-reference rankings remained heterogeneous, and ResShift consistently achieved lower LPIPS on Sea-Undistort. These findings indicate that improvements in pixel-level similarity, perceptual feature agreement, and natural-image statistics do not necessarily coincide, consistent with limitations reported in underwater image quality assessment [73].

The component ablations further clarify these results. Removing the SSIM loss produced the most pronounced degradation across the evaluated fidelity and perceptual metrics, supporting its contribution to the reconstruction objective. In contrast, removing individual conditioning mechanisms or other loss terms produced smaller and sometimes favorable changes in specific metrics. The complete architecture therefore represents one experimentally evaluated configuration, without establishing that every component independently contributes to all aspects of restoration quality.

For shallow-water remote sensing, improved image quality is valuable only when relevant scene information is preserved. Accurate feature correspondence, object localization, habitat interpretation, and bathymetric reconstruction depend on geometric consistency and reliable seabed information [16], [12]. The real-UAV case study illustrates this distinction: restoration altered SIFT keypoint density and SAM segmentation behavior, but the number of repeatable descriptor matches with the temporal computer-vision reference remained limited. Moreover, the SAM regions lacked independent semantic annotations, and the feature matches were not geometrically veri fied. The results therefore demonstrate changes in downstream computer-vision outputs without establishing improvements in mapping accuracy. Future task-oriented evaluations should incorporate annotated habitat boundaries, independently surveyed features, or validated bathymetric measurements.

## C. Practical Considerations and Future Directions

KAUSTics-Net integrates established conditional diffusion components within a restoration framework designed for through-water UAV imagery. Its contribution lies in the combination of semi-synthetic data construction, multi-scale conditioning, and a composite reconstruction objective evaluated under controlled degradation settings. The ablation study provides partial support for these design choices, with the clearest benefit associated with structural supervision. Additional experiments are required to isolate the effects of individual conditioning mechanisms and determine their contributions under different training conditions.

Computational requirements also affect practical applicability. Although all methods were evaluated at a common 256 × 256 resolution, differences in training schedules, optimization procedures, and GPU hardware prevent a strictly matchedcompute comparison. KAUSTics-Net required approximately 0.367 s per image for inference on an NVIDIA A100 GPU, excluding metric computation. Its iterative sampling procedure introduces additional considerations for processing large UAV surveys, particularly when preserving the native spatial resolution of high-resolution acquisitions. Comparisons under matched computational budgets and evaluation at larger image resolutions would help establish the trade-offs between restoration quality and operational efficiency.

Future research should prioritize spatially independent UAV surveys covering additional water types, depths, illumination conditions, and sensors, together with physically informed simulations incorporating measured water-column properties and seabed geometry. Limited real-data adaptation and uncertainty-aware restoration may further help address domain shifts and regions where seabed information is severely obscured [69], [70]. Ultimately, independently validated downstream assessments are needed to establish whether the observed restoration improvements support more accurate coastal mapping and environmental monitoring.

## VI. CONCLUSION

This study introduced KAUSTics, a semi-synthetic dataset comprising 1,280 reference patches derived from 32 shallowwater UAV images and 12,800 paired degraded observations, together with KAUSTics-Net, a pixel-space conditional diffusion framework for through-water image restoration. The framework integrates multi-scale conditioning, FiLMmodulated residual blocks, cross-attention, and a composite training objective combining velocity prediction with structural and chromatic reconstruction losses.

The revised evaluation used source-image-disjoint partitions, a common 256×256 image resolution, eight benchmark methods, and three independent training seeds. On KAUSTics, KAUSTics-Net achieved 31.715 ± 0.439 dB PSNR, 0.849 ± 0.002 SSIM, and 0.145 ± 0.007 LPIPS. Following datasetspecific adaptation, it also achieved the highest PSNR and SSIM across the three evaluated Sea-Undistort configurations, although ResShift obtained better LPIPS and several noreference metrics favored competing methods. Architectural ablations identified structural supervision as a particularly important component, while the real-UAV case study documented changes in feature detection and segmentation behavior.

The principal limitations concern the small number of independent source scenes, the synthetic-to-real domain gap, the absence of independently validated water-free references for KAUSTics, and the limited evidence of downstream mapping improvements. Further validation across geographically independent UAV surveys, measured water-column conditions, and quantitative remote-sensing tasks is required. Within these constraints, the study demonstrates the potential of semisynthetic data and conditional diffusion for restoring shallowwater UAV imagery toward low-distortion reference appearances.

## DATA AND CODE AVAILABILITY

The KAUSTics dataset, pretrained KAUSTics-Net models, and source code will be made publicly available upon publication at https://huggingface.co/datasets/dar96io/KAUSTics, https://huggingface.co/dar96io/KAUSTics-Net, and https://github.com/Dark3850/KAUSTics-Net, respectively. The fixed Sea-Undistort experimental partition will be available at https://huggingface.co/datasets/dar96io/ Sea-Undistort-evaluation, retaining the original dataset’s attribution and CC BY-NC-SA 4.0 license.

## REFERENCES

[1] C. J. Cruz-Ramírez, V. Chávez, R. Silva, J. J. Muñoz-Perez, and E. Rivera-Arriaga, “Coastal management: A review of key elements for vulnerability assessment,” Journal of Marine Science and Engineering, vol. 12, no. 3, 2024. [Online]. Available: https: //www.mdpi.com/2077-1312/12/3/386

[2] G. Motta Zanin, S. P. Muwafu, and M. Máñez Costa, “Nature-based solutions for coastal risk management in the mediterranean basin: A literature review,” Journal of Environmental Management, vol. 356, p. 120667, 2024. [Online]. Available: https://www.sciencedirect.com/ science/article/pii/S0301479724006534

[3] S. D. D. M. Déguénon, R. Adade, O. Teka, D. W. Aheto, and B. Sinsin, “Sea-level rise and flood mapping: a review of models for coastal management,” Natural Hazards, vol. 120, no. 3, pp. 2155–2178, Feb. 2024. [Online]. Available: https://doi.org/10.1007/s11069-023-06225-1

[4] K. J. Chowdhury, M. R. Ali, M. A. Chowdhury, and S. L. U. Islam, “Climate change induced risks assessment of a coastal area: A “socioeconomic and livelihood vulnerability index” based study in coastal bangladesh,” Natural Hazards Research, vol. 5, no. 1, pp. 75–87, 2025. [Online]. Available: https://www.sciencedirect.com/ science/article/pii/S2666592124000544

[5] J. Holzner, G. Strunz, S. Martinis, and S. Plank, “Analyzing coastal dynamics by means of multi-sensor satellite imagery at the East Frisian Island of Langeoog, Germany,” Scientific Reports, vol. 15, no. 1, p. 7372, Mar. 2025, publisher: Nature Publishing Group. [Online]. Available: https://www.nature.com/articles/s41598-025-91306-3

[6] E. Vandebroek, R. Lindenbergh, F. Van Leijen, M. De Schipper, S. De Vries, and R. Hanssen, “Semi-automated monitoring of a mega-scale beach nourishment using high-resolution terrasar-x satellite data,” Remote Sensing, vol. 9, no. 7, 2017. [Online]. Available: https://www.mdpi.com/2072-4292/9/7/653

[7] Y.-L. S. Tsai, “Nation-scale multidecadal shoreline extraction and coastal spatio-temporal change monitoring using cross-mission remote sensing data,” Ocean & Coastal Management, vol. 253, p. 107136, 2024. [Online]. Available: https://www.sciencedirect.com/science/article/pii/ S0964569124001212

[8] D. Christofi, C. Mettas, E. Evagorou, N. Stylianou, M. Eliades, C. Theocharidis, A. Chatzipavlis, T. Hasiotis, and D. Hadjimitsis, “A review of open remote sensing data with gis, ai, and uav support for shoreline detection and coastal erosion monitoring,” Applied Sciences, vol. 15, no. 9, 2025. [Online]. Available: https: //www.mdpi.com/2076-3417/15/9/4771

[9] Z. Yang, X. Yu, S. Dedman, M. Rosso, J. Zhu, J. Yang, Y. Xia, Y. Tian, G. Zhang, and J. Wang, “Uav remote sensing applications in marine monitoring: Knowledge visualization and review,” Science of The Total Environment, vol. 838, p. 155939, 2022. [Online]. Available: https://www.sciencedirect.com/science/article/pii/S0048969722030364

[10] P. Kujawa, J. Wajs, and K. Plesniak, “The approach to´ uav image acquisition and processing for very shallow water mapping,” International Journal of Applied Earth Observation and Geoinformation, vol. 141, p. 104604, 2025. [Online]. Available: https://www.sciencedirect.com/science/article/pii/S1569843225002511

[11] T. Forbes, M. Goldsmith, S. Mudur, and C. Poullis, “Deepcaustics: Classification and removal of caustics from underwater imagery,” IEEE Journal of Oceanic Engineering, vol. 44, no. 3, pp. 728–738, 2019.

[12] D. Scilla, O. A. Lopez, B. O. Nieuwenhuis, K. Johansen, M. Elías-Lara, V. Angulo, J. Rodríguez, B. H. Jones, and M. McCabe, “Computer vision corrections enhance uav-based retrievals in shallow waters,” IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 2025.

[13] J. Qin, M. Li, J. Zhao, D. Li, H. Zhang, and J. Zhong, “Advancing sun glint correction in high-resolution marine uav rgb imagery for coral reef monitoring,” ISPRS Journal of Photogrammetry and Remote Sensing, vol. 207, pp. 298–311, 2024. [Online]. Available: https://www.sciencedirect.com/science/article/pii/S0924271623003428

[14] A. M. Muslim, W. S. Chong, C. D. M. Safuan, I. Khalil, and M. S. Hossain, “Coral reef mapping of uav: A comparison of sun glint correction methods,” Remote Sensing, vol. 11, no. 20, 2019. [Online]. Available: https://www.mdpi.com/2072-4292/11/20/2422

[15] B. Nababan, L. O. K. Mastu, N. H. Idris, and J. P. Panjaitan, “Shallow-water benthic habitat mapping using drone with object based image analyses,” Remote Sensing, vol. 13, no. 21, 2021. [Online]. Available: https://www.mdpi.com/2072-4292/13/21/4452

[16] P. Agrafiotis, K. Karantzalos, A. Georgopoulos, and D. Skarlatos, “Correcting image refraction: Towards accurate aerial image-based bathymetry mapping in shallow waters,” Remote Sensing, vol. 12, no. 2, 2020. [Online]. Available: https://www.mdpi.com/2072-4292/12/2/322

[17] I. G. Y. Partama, A. Kanno, M. Ueda, Y. Akamatsu, R. Inui, M. Sekine, K. Yamamoto, T. Imai, and T. Higuchi, “Removal of water-surface reflection effects with a temporal minimum filter for uav-based shallow-water photogrammetry,” Earth Surface Processes and Landforms, vol. 43, no. 12, pp. 2673–2682, 2018. [Online]. Available: https://onlinelibrary.wiley.com/doi/abs/10.1002/esp.4399

[18] S. Maritorena, A. Morel, and B. Gentili, “Diffuse reflectance of oceanic shallow waters: Influence of water depth and bottom albedo,” Limnology and Oceanography, vol. 39, no. 7, pp. 1689–1703, 1994. [Online]. Available: https://onlinelibrary.wiley.com/doi/abs/10.4319/lo. 1994.39.7.1689

[19] H. M. Dierssen, R. C. Zimmerman, R. A. Leathers, T. V. Downes, and C. O. Davis, “Ocean color remote sensing of seagrass and bathymetry in the Bahamas Banks by high-resolution airborne imagery,” Limnology and Oceanography, vol. 48, no. 1part2, pp. 444–455, 2003. [Online]. Available: https://onlinelibrary.wiley.com/doi/abs/10.4319/lo.2003.48.1\_ part\_2.0444

[20] E. Vahtmäe, T. Kutser, and B. Paavel, “Performance and applicability of water column correction models in optically complex coastal waters,” Remote Sensing, vol. 12, no. 11, 2020. [Online]. Available: https://www.mdpi.com/2072-4292/12/11/1861

[21] J. D. Hedley, A. R. Harborne, and P. J. Mumby, “Technical note: Simple and robust removal of sun glint for mapping shallow-water benthos,” International Journal ofRemote Sensing, vol. 26, no. 10, pp. 2107–2112, 2005. [Online]. Available: https://doi.org/10.1080/01431160500034086

[22] P. Hambarde, S. Murala, and A. Dhall, “Uw-gan: Single-image depth estimation and image enhancement for underwater images,” IEEE Transactions on Instrumentation and Measurement, vol. 70, pp. 1–12, 2021.

[23] F. Yao, F. Tang, X. Wang, Y. Wang, S. Wang, G. Zhong, and J. Zhang, “Dewatergan: A physics-guided unsupervised image water removal for uav in coastal zone,” IEEE Transactions on Geoscience and Remote Sensing, vol. 62, pp. 1–15, 2024.

[24] G. Parmar, K. Kumar Singh, R. Zhang, Y. Li, J. Lu, and J.-Y. Zhu, “Zero-shot image-to-image translation,” in ACM SIGGRAPH 2023 Conference Proceedings, ser. SIGGRAPH ’23. New York, NY, USA: Association for Computing Machinery, 2023. [Online]. Available: https://doi.org/10.1145/3588432.3591513

[25] B. Xia, Y. Zhang, S. Wang, Y. Wang, X. Wu, Y. Tian, W. Yang, and L. Van Gool, “Diffir: Efficient diffusion model for image restoration,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), October 2023, pp. 13 095–13 105.

[26] M. Kromer, P. Agrafiotis, and B. Demir, “Sea-undistort: A dataset for through-water image restoration in high resolution airborne bathymetric mapping,” IEEE Geoscience and Remote Sensing Letters, 2025.

[27] C. Fabbri, M. J. Islam, and J. Sattar, “Enhancing underwater imagery using generative adversarial networks,” in 2018 IEEE International Conference on Robotics and Automation (ICRA), 2018, pp. 7159–7165.

[28] M. Guan, H. Xu, G. Jiang, M. Yu, Y. Chen, T. Luo, and X. Zhang, “Diffwater: Underwater image enhancement based on conditional denoising diffusion probabilistic model,” IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, vol. 17, pp. 2319– 2335, 2024.

[29] W. S. Chong, N. H. M. Zaki, M. S. Hossain, A. M. Muslim, and A. B. Pour, “Introducing theil-sen estimator for sun glint correction of uav data for coral mapping,” Geocarto International, vol. 37, no. 15, pp. 4527–4556, 2022. [Online]. Available: https: //doi.org/10.1080/10106049.2021.1892206

[30] I. Partama, A. Kanno, Y. Akamatsu, R. Inui, M. Goto, and M. Sekine, “A simple and empirical refraction correction method for uav-based shallow-water photogrammetry,” Int. J. Environ. Chem. Ecol. Geol. Geophys. Eng, vol. 11, pp. 254–261, 2017.

[31] S. E. Lambert and C. E. Parrish, “Refraction correction for spectrally derived bathymetry using uas imagery,” Remote Sensing, vol. 15, no. 14, p. 3635, 2023.

[32] A. Luis, “A synoptic review on deriving bathymetry information using remote sensing technologies: models, methods and comparisons,” Advances in remote Sensing, 2015.

[33] M. Specht, M. Wisniewska, A. Stateczny, C. Specht, B. Szostak,´ O. Lewicka, M. Stateczny, S. Wid´zgowski, and A. Halicki, “Analysis of methods for determining shallow waterbody depths based on images taken by unmanned aerial vehicles,” Sensors, vol. 22, no. 5, p. 1844, 2022.

[34] V. Chirayath and S. A. Earle, “Drones that see through waves– preliminary results from airborne fluid lensing for centimetre-scale aquatic conservation,” Aquatic Conservation: Marine and Freshwater Ecosystems, vol. 26, pp. 237–250, 2016.

[35] V. Chirayath and R. Instrella, “Fluid lensing and machine learning for centimeter-resolution airborne assessment of coral reefs in american samoa,” Remote sensing of environment, vol. 235, p. 111475, 2019.

[36] I. J. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville, and Y. Bengio, “Generative adversarial nets,” Advances in neural information processing systems, vol. 27, 2014.

[37] J.-Y. Zhu, T. Park, P. Isola, and A. A. Efros, “Unpaired image-to-image translation using cycle-consistent adversarial networks,” in Proceedings of the IEEE international conference on computer vision, 2017, pp. 2223–2232.

[38] M. J. Islam, Y. Xia, and J. Sattar, “Fast underwater image enhancement for improved visual perception,” IEEE Robotics and Automation Letters, vol. 5, no. 2, pp. 3227–3234, 2020.

[39] X. Liu, Z. Gao, and B. M. Chen, “Mlfcgan: Multilevel feature fusionbased conditional gan for underwater image color correction,” IEEE Geoscience and Remote Sensing Letters, vol. 17, no. 9, pp. 1488–1492, 2020.

[40] N. Wang, Y. Zhou, F. Han, H. Zhu, and J. Yao, “Uwgan: Underwater gan for real-world underwater color restoration and dehazing,” arXiv preprint arXiv:1912.10269, 2019.

[41] J. Ho, A. Jain, and P. Abbeel, “Denoising diffusion probabilistic models,” Advances in neural information processing systems, vol. 33, pp. 6840– 6851, 2020.

[42] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer, “Highresolution image synthesis with latent diffusion models,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 10 684–10 695.

[43] C. Saharia, W. Chan, H. Chang, C. Lee, J. Ho, T. Salimans, D. Fleet, and M. Norouzi, “Palette: Image-to-image diffusion models,” in ACM SIGGRAPH 2022 Conference Proceedings, ser. SIGGRAPH ’22. New York, NY, USA: Association for Computing Machinery, 2022. [Online]. Available: https://doi.org/10.1145/3528233.3530757

[44] S. Lu, F. Guan, H. Zhang, and H. Lai, “Underwater image enhancement method based on denoising diffusion probabilistic model,” Journal of Visual Communication and Image Representation, vol. 96, p. 103926, 2023. [Online]. Available: https://www.sciencedirect.com/ science/article/pii/S1047320323001761

[45] ——, “Speed-up ddpm for real-time underwater image enhancement,” IEEE Transactions on Circuits and Systems for Video Technology, vol. 34, no. 5, pp. 3576–3588, 2024.

[46] X. Shi and Y.-G. Wang, “Cpdm: Content-preserving diffusion model for underwater image enhancement,” Scientific Reports, vol. 14, no. 1, p. 31309, 2024.

[47] M. Yao, R. Xu, Y. Guan, J. Huang, and Z. Xiong, “Neural degradation representation learning for all-in-one image restoration,” IEEE Transactions on Image Processing, vol. 33, pp. 5408–5423, 2024.

[48] Z. Yue, J. Wang, and C. C. Loy, “Resshift: Efficient diffusion model for image super-resolution by residual shifting,” Advances in Neural Information Processing Systems, vol. 36, pp. 13 294–13 307, 2023.

[49] L. Peng and L. Bian, “Adaptive Dual-domain Learning for Underwater Image Enhancement,” Proceedings of the AAAI Conference on Artificial Intelligence, vol. 39, no. 6, pp. 6461–6469, Apr. 2025. [Online]. Available: https://ojs.aaai.org/index.php/AAAI/article/view/32692

[50] L. Peng, C. Zhu, and L. Bian, “U-shape transformer for underwater image enhancement,” IEEE Transactions on Image Processing, vol. 32, pp. 3066–3079, 2023.

[51] Blender Foundation, “Blender: Free and open source 3d creation suite,” 2024, version 4.3.2. [Online]. Available: https://www.blender.org

[73] X. Li, H. Xu, G. Jiang, M. Yu, T. Luo, X. Zhang, and H. Ying, “Underwater image quality assessment from synthetic to real-world: Dataset and objective method,” ACM Transactions on Multimedia Computing, Communications and Applications, vol. 20, no. 3, pp. 1–23, 2023.

[52] E. Reinhard, M. Adhikhmin, B. Gooch, and P. Shirley, “Color transfer between images,” IEEE Computer Graphics and Applications, vol. 21, no. 5, pp. 34–41, 2001.

[53] E. Perez, F. Strub, H. De Vries, V. Dumoulin, and A. Courville, “Film: Visual reasoning with a general conditioning layer,” in Proceedings of the AAAI conference on artificial intelligence, vol. 32, no. 1, 2018.

[54] A. Q. Nichol and P. Dhariwal, “Improved denoising diffusion probabilistic models,” in International conference on machine learning. PMLR, 2021, pp. 8162–8171.

[55] J. Song, C. Meng, and S. Ermon, “Denoising diffusion implicit models,” arXiv preprint arXiv:2010.02502, 2020.

[56] R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang, “The unreasonable effectiveness of deep features as a perceptual metric,” in 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2018, pp. 586–595.

[57] G. Sharma, W. Wu, and E. N. Dalal, “The CIEDE2000 color-difference formula: Implementation notes, supplementary test data, and mathematical observations,” Color Research & Application, vol. 30, no. 1, pp. 21–30, 2005.

[58] A. Mittal, R. Soundararajan, and A. C. Bovik, “Making a “completely blind” image quality analyzer,” IEEE Signal Processing Letters, vol. 20, no. 3, pp. 209–212, 2013.

[59] A. Mittal, A. K. Moorthy, and A. C. Bovik, “Blind/referenceless image spatial quality evaluator,” in 2011 Conference Record of the Forty Fifth Asilomar Conference on Signals, Systems and Computers (ASILOMAR), 2011, pp. 723–727.

[60] V. N, P. D, M. C. Bh, S. S. Channappayya, and S. S. Medasani, “Blind image quality evaluation using perception based features,” in 2015 Twenty First National Conference on Communications (NCC), 2015, pp. 1–6.

[61] L. Zhang, L. Zhang, and A. C. Bovik, “A feature-enriched completely blind image quality evaluator,” IEEE Transactions on Image Processing, vol. 24, no. 8, pp. 2579–2591, 2015.

[62] W. Zhang, G. Zhai, Y. Wei, X. Yang, and K. Ma, “Blind image quality assessment via vision-language correspondence: A multitask learning perspective,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2023, pp. 14 071–14 081.

[63] L. Agnolucci, L. Galteri, and M. Bertini, “Quality-aware imagetext alignment for opinion-unaware image quality assessment,” arXiv preprint arXiv:2403.11176, 2024.

[64] J. Ke, Q. Wang, Y. Wang, P. Milanfar, and F. Yang, “Musiq: Multiscale image quality transformer,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), October 2021, pp. 5148–5157.

[65] A. Kirillov, E. Mintun, N. Ravi, H. Mao, C. Rolland, L. Gustafson, T. Xiao, S. Whitehead, A. C. Berg, W.-Y. Lo, P. Dollár, and R. Girshick, “Segment anything,” arXiv preprint arXiv:2304.02643, 2023. [Online]. Available: https://arxiv.org/abs/2304.02643

[66] D. G. Lowe, “Distinctive image features from scale-invariant keypoints,” International Journal of Computer Vision, vol. 60, no. 2, pp. 91–110, 2004. [Online]. Available: https://www.cs.ubc.ca/\~lowe/papers/ijcv04. pdf

[67] K. Purnima and C. S. Kumar, “Csuid – comprehensive synthetic underwater image dataset,” Data in Brief, vol. 55, p. 110723, 2024. [Online]. Available: https://www.sciencedirect.com/science/article/pii/ S2352340924006905

[68] L. A. Barbosa and A. L. Apolinario, “From physically based to generative models: A survey on underwater image synthesis techniques,” Journal of Imaging, vol. 11, no. 5, 2025. [Online]. Available: https://www.mdpi.com/2313-433X/11/5/161

[69] Z. Wang, L. Shen, M. Xu, M. Yu, K. Wang, and Y. Lin, “Domain adaptation for underwater image enhancement,” IEEE Transactions on Image Processing, vol. 32, pp. 1442–1457, 2023.

[70] Q. Jiang, Y. Zhang, F. Bao, X. Zhao, C. Zhang, and P. Liu, “Two-step domain adaptation for underwater image enhancement,” Pattern Recognition, vol. 122, p. 108324, 2022. [Online]. Available: https://www.sciencedirect.com/science/article/pii/S0031320321005045

[71] W. Ye, G. Zheng, X. Cao, Y. Ma, and A. Zhang, “Spurious correlations in machine learning: A survey,” arXiv preprint arXiv:2402.12715, 2024.

[72] V. Mulchandani and J.-E. Kim, “Severing spurious correlations with data pruning,” arXiv preprint arXiv:2503.18258, 2025.

## SUPPLEMENTARY MATERIAL

This supplement documents the sampler-aware adaptation procedure, provides full zero-shot Sea-Undistort comparisons, and illustrates the reference-patch inventory and appearance variation in KAUSTics. The zero-shot and fine-tuned evaluations are distinct. Sea-Undistort full-reference metrics are computed against the original synthetic water-free targets on the same 120 held-out scene identifiers per input configuration at $2 5 6 \times 2 5 6$ RGB resolution.

## APPENDIX A

## ADDITIONAL ADAPTATION IMPLEMENTATION DETAILS

Sampler-aware fine-tuning was performed in FP32 on one NVIDIA A100 GPU per seed, with gradient clipping at a global norm of 1. Each epoch consisted of 70 optimizer updates. The extra sampled-output objective was evaluated on updates 0, 4, . . ., 68 (18 of 70 updates), using microbatches of four to cover the effective batch of 12. The sampling procedure ran in evaluation mode. Gradients from the standard training objective and the sampler-aware objective were combined before each optimizer step. Sampled RGB mean squared error (MSE) used unclipped outputs, while the SSIM and blurred-CIELAB terms used the clipped RGB range. The first 21 DDIM sampling steps were detached, and differentiation was restricted to the last four steps. Early stopping required at least 60 epochs and used a patience of five validation checks, with a minimum SSIM improvement of $1 0 ^ { - 4 } .$ . Each seed was initialized from its corresponding KAUSTics checkpoint, with a newly initialized AdamW optimizer. All three runs reached 100 epochs; validation-selected checkpoints were obtained at epochs 95, 100, and 100. Selection used the mean SSIM for all-effects validation scenes; the no-waves validation results were secondary. Test scene identifiers were excluded from optimization and checkpoint selection. However, preliminary Sea-Undistort test results had been inspected during the revision process, so the expanded adaptation experiment is exploratory and is not a newly blind external validation. An initial LPIPS guard permitting an increase of at most 0.02 was not satisfied by any seed. This criterion was not retrospectively relaxed in the experimental record. Differences in model-specific schedules and the additional KAUSTics-Net adaptation experiments also mean that training and tuning effort were not matched across methods.

## A. Sea-Undistort Fine-Tuning of the Benchmark Methods

All eight baselines: CycleGAN, UGAN, FUnIE-GAN, De-WaterGAN, DiffWater, ResShift, SS-UIE, and U-Shape Transformer, were initialized from their corresponding seed-specific, validation-selected KAUSTics-trained restoration checkpoints. These were the same checkpoints used for the zero-shot Sea-Undistort evaluation. Restoration networks were therefore adapted from KAUSTics, rather than trained from scratch on Sea-Undistort. Optimizer states were reinitialized when adaptation began.

Every baseline used the same fixed partition of 840 training, 240 validation, and 120 test scene IDs, at RGB 256 × 256 resolution. Training used only the all-effects distorted inputs and the original water-free reference images from the 840 training scenes. No no-waves or no-sunglint configurations were added to the baseline training sets. CycleGAN and DeWaterGAN retained their unpaired training formulations: clean targets were sampled independently from the same 840- scene training pool. The other methods used the corresponding paired references.

All baselines completed 100 fine-tuning epochs with a constant learning rate of $1 0 ^ { - 5 }$ and no early stopping. Each epoch traversed all 840 distorted training inputs. Methodspecific optimization objectives were retained; U-Shape Transformer continued its final L1-based training stage. Batch sizes were 8 for CycleGAN, DeWaterGAN, ResShift, and U-Shape Transformer; 128 for UGAN and FUnIE-GAN; 9 for SS-UIE; and 1 for DiffWater.

Checkpoint selection used the highest mean SSIM over all 240 all-effects validation scenes. The initial KAUSTics checkpoint, evaluated at epoch 0, remained eligible. For seven baselines, validation was performed at epochs 0, 1, 5, and every five epochs thereafter. For DiffWater, validation checks were performed every 20 epochs to reduce the cost of its native diffusion sampler. Consequently, validation frequency, and therefore checkpoint-selection opportunities, were not identical across methods.

Every baseline was fine-tuned using seeds 1, 2, and 3, starting from the matching KAUSTics training seed. The 120 test scene IDs were excluded from optimization and per-run checkpoint selection. Each selected checkpoint was evaluated on all three test-input variants, all-effects, no-waves, and nosunglint, against the same original water-free reference.

## APPENDIX B ZERO-SHOT SEA-UNDISTORT EVALUATION

## A. Protocol and Full-Reference Results

Zero-shot transfer was evaluated using the KAUSTicstrained checkpoints without any Sea-Undistort optimization. This term refers to the source-trained experimental checkpoints, not externally released model weights. All eight benchmark methods used their completed KAUSTics training runs under the 100-epoch warm-up and patience-10 rule, with validation-selected checkpoints. KAUSTics-Net used its longer-trained, validation-selected checkpoint. These training schedules were not matched for total computation. The comparison uses identical held-out scenes under all-effects, nowaves, and no-sunglint inputs, each evaluated against the original water-free reference image.

On all-effects observations, KAUSTics-Net achieved $9 . 4 1 8 \pm 0 . 0 9 4$ dB PSNR and 0.233 ± 0.012 SSIM, below the unprocessed input (10.257 dB and 0.311, respectively). The corresponding zero-shot scores were also limited under no $\mathrm { w a v e s ~ ( 9 . 6 5 2 \pm 0 . 1 3 9 ~ d B , 0 . 2 9 1 \pm 0 . 0 2 2 ) }$ and no sunglint $( 9 . 6 3 2 \pm 0 . 1 4 7 ~ \mathrm { d B } , 0 . 2 6 8 \pm 0 . 0 1 9 )$ . These results document a substantial synthetic-domain shift and should not be conflated with the gains achieved after Sea-Undistort fine-tuning in the main manuscript. Table S1 presents the complete full-reference comparison.

![](images/9223379ca606564a5c78a97c1208f5a29d5d7973c84aa098dd4e1423bfab0db1.jpg)  
Fig. S1. Sea-Undistort zero-shot comparison using seed-1 checkpoints at 256 x 256 RGB resolution. Row groups show all effects, no waves, and no sunglint; each group contains scenes 944, 143, and 646 in that order. Columns show the distorted input, original water-free reference, and eight benchmark methods alongside KAUSTics-Net. Each panel uses the same central 26 x 26 pixel inset. No panel-specific colour correction was applied. All methods use the completed KAUSTics-trained checkpoints from Tables S1 and S2.

## B. No-Reference Image Quality Results

Table S2 reports nine no-reference metrics for the same restored outputs and checkpoints used in the full-reference evaluation. The metrics assess statistical or learned perceptual image properties without access to the reference image and cannot establish recovery of obscured seabed content. Sharpness and entropy describe changes in spatial intensity structure and are not indicators of mapping accuracy.

## APPENDIX C

## KAUSTICS DATASET INVENTORY AND APPEARANCE EXAMPLES

The dataset inventory contains 32 original UAV-image identifiers and 1,280 reference patches, with 40 patches per source image. Ten distorted variants per reference yield 12,800 input– reference pairs. Each reference is counted once in the spatialpixel inventory. The original reference patches are 1000×1000 pixels, and the experiments resize both references and distorted observations to 256 × 256 pixels. The original-file audit identified one degraded training image stored at 1344 × 1344 pixels; the other 12,799 degraded images are 1000 × 1000 pixels. The source audit inspected 28,160 original and resized image-file headers without modifying the original files.

The 1,280 original reference patches contain 1,280,000,000 spatial pixels. After resizing, the references contain 83,886,080 spatial pixels. The training, validation, and test groups account for 1,040, 120, and 120 reference patches, respectively, corresponding to 1,040,000,000, 120,000,000, and 120,000,000 original spatial pixels. The resized totals are 68,157,440, 7,864,320, and 7,864,320 spatial pixels. These figures represent image sampling volume, not unique geographic coverage, habitat area, or independent scene diversity.

The following image groups describe representative patch appearances from a single shallow-water lagoon: hard-bottom or rock-like texture, sand-dominated appearance, vegetationlike or mixed benthic texture, and dark or low-visibility appearance. These descriptions are visual categories and were not field-validated as substrate classes or species identities. In particular, image darkness and blue tint do not independently establish water depth. Habitat-classification frameworks such as Reef Cover can inform descriptive terminology, but the KAUSTics patches do not include a verified patch-level habitat label manifest. Consequently, category proportions and classspecific pixel counts are not reported.

TABLE S1  
SEA-UNDISTORT ZERO-SHOT FULL-REFERENCE METRICS ON 120 HELD-OUT SCENES PER CONFIGURATION. RESULTS ARE MEAN ± SAMPLE STANDARD DEVIATION OVER THREE SOURCE-TRAINING SEEDS; ALL RESULTS ARE EVALUATED AGAINST THE ORIGINAL WATER-FREE REFERENCE. MSE IS SHOWN IN UNITS OF $1 0 ^ { - 3 }$ , AND RMSE USES [0, 1] RGB UNITS. THE UNPROCESSED INPUT IS PROVIDED FOR CONTEXT.
<table><tr><td>Method</td><td>mPSNR ↑</td><td>mSSIM ↑</td><td>mRMSE↓</td><td>mMSE  $[ \mathbf { 1 0 ^ { - 3 } } ] \downarrow$ </td><td> $\overline { { { \bf { L P I P S } } \downarrow } }$ </td><td> $\overline { { \Delta E _ { 0 0 } \downarrow } }$ </td></tr><tr><td colspan="7">All effects</td></tr><tr><td>Unprocessed input</td><td>10.257</td><td>0.311</td><td>0.341</td><td>136.826</td><td>0.698</td><td>32.385</td></tr><tr><td>CycleGAN [37]</td><td> $\mathbf { 1 0 . 1 5 7 \pm 0 . 0 6 4 }$ </td><td> $\mathbf { 0 . 3 2 4 \pm 0 . 0 0 7 }$ </td><td> $\mathbf { 0 . 3 1 7 \pm 0 . 0 0 2 }$ </td><td> $\mathbf { 1 0 5 . 0 5 9 \pm 1 . 2 3 6 }$ </td><td> $0 . 7 1 4 \pm 0 . 0 2 9$ </td><td> $2 8 . 9 9 6 \pm 0 . 2 8 2$ </td></tr><tr><td>UGAN [27]</td><td> $1 0 . 1 4 0 \pm 0 . 0 9 5$ </td><td> $0 . 3 0 5 \pm 0 . 0 0 5$ </td><td> $0 . 3 2 5 \pm 0 . 0 0 3$ </td><td> $1 1 4 . 5 1 2 \pm 2 . 0 4 3$ </td><td> $0 . 7 7 1 \pm 0 . 0 1 4$ </td><td> $\mathbf { 2 8 . 3 7 6 \pm 0 . 5 1 7 }$ </td></tr><tr><td>FUnIE-GAN [38]</td><td> $9 . 2 4 9 \pm 0 . 1 5 6$ </td><td> $0 . 2 0 4 \pm 0 . 0 1 1$ </td><td> $0 . 3 6 9 \pm 0 . 0 1 5$ </td><td> $1 5 2 . 3 9 2 \pm 1 6 . 0 8 7$ </td><td> $0 . 7 4 4 \pm 0 . 0 1 1$ </td><td> $3 4 . 0 1 1 \pm 1 . 4 8 7$ </td></tr><tr><td>DeWaterGAN [23]</td><td> $1 0 . 0 8 1 \pm 0 . 1 8 5$ </td><td> $0 . 3 0 9 \pm 0 . 0 0 9$ </td><td> $0 . 3 4 3 \pm 0 . 0 0 9$ </td><td> $1 3 4 . 1 3 1 \pm 7 . 1 9 8$ </td><td> $0 . 7 1 9 \pm 0 . 0 2 8$ </td><td> $3 1 . 5 0 0 \pm 1 . 0 4 6$ </td></tr><tr><td>DiffWater [28]</td><td> $9 . 8 0 0 \pm 0 . 2 8 6$ </td><td> $0 . 2 2 7 \pm 0 . 0 3 2$ </td><td> $0 . 3 3 4 \pm 0 . 0 0 9$ </td><td> $1 1 8 . 4 1 7 \pm 5 . 6 9 0$ </td><td> $0 . 7 4 5 \pm 0 . 0 5 2$ </td><td> $2 9 . 7 5 8 \pm 2 . 6 6 8$ </td></tr><tr><td>ResShift [48]</td><td> $9 . 6 3 9 \pm 0 . 2 0 8$ </td><td> $0 . 2 5 3 \pm 0 . 0 1 3$ </td><td> $0 . 3 5 5 \pm 0 . 0 0 9$ </td><td> $1 4 1 . 4 9 7 \pm 8 . 1 8 0$ </td><td> $\mathbf { 0 . 6 6 4 \pm 0 . 0 1 3 }$ </td><td> $3 1 . 9 6 1 \pm 0 . 7 0 2$ </td></tr><tr><td>SS-UIE [49]</td><td> $9 . 7 5 0 \pm 0 . 1 2 7$ </td><td> $0 . 2 8 8 \pm 0 . 0 0 0$ </td><td> $0 . 3 5 6 \pm 0 . 0 0 9$ </td><td> $1 4 5 . 1 4 1 \pm 7 . 9 2 8$ </td><td> $0 . 6 9 8 \pm 0 . 0 1 2$ </td><td> $3 2 . 2 1 3 \pm 1 . 4 0 6$ </td></tr><tr><td>U-Shape Transformer [50]</td><td> $9 . 0 8 2 \pm 0 . 0 6 5$ </td><td> $0 . 2 5 0 \pm 0 . 0 0 9$ </td><td> $0 . 3 7 9 \pm 0 . 0 0 4$ </td><td> $1 6 0 . 6 1 2 \pm 4 . 4 0 0$ </td><td> $0 . 8 4 1 \pm 0 . 0 0 9$ </td><td> $3 5 . 0 8 3 \pm 1 . 0 7 3$ </td></tr><tr><td>KAUSTics-Net (ours)</td><td> $9 . 4 1 8 \pm 0 . 0 9 4$ </td><td> $0 . 2 3 3 \pm 0 . 0 1 2$ </td><td> $0 . 3 7 3 \pm 0 . 0 0 4$ </td><td> $1 6 0 . 5 3 2 \pm 6 . 3 2 8$ </td><td> $0 . 7 3 2 \pm 0 . 0 5 4$ </td><td> $3 3 . 6 3 2 \pm 0 . 0 5 3$ </td></tr></table>

<table><tr><td colspan="7">No waves</td></tr><tr><td>Unprocessed input</td><td>10.526</td><td>0.411</td><td>0.342</td><td>143.097</td><td>0.581</td><td>32.815</td></tr><tr><td>CycleGAN [37]</td><td> $1 0 . 0 9 1 \pm 0 . 2 5 7$ </td><td> $\mathbf { 0 . 4 0 1 \pm 0 . 0 1 1 }$ </td><td> $\mathbf { 0 . 3 2 0 \pm 0 . 0 1 0 }$ </td><td> $\mathbf { 1 0 6 . 6 6 0 \pm 6 . 5 8 8 }$ </td><td> $0 . 6 3 3 \pm 0 . 0 2 8$ </td><td> $2 9 . 3 0 2 \pm 0 . 5 2 4$ </td></tr><tr><td>UGAN [27]</td><td> ${ \bf 1 0 . 3 4 3 \pm 0 . 1 3 4 }$ </td><td> $0 . 3 5 3 \pm 0 . 0 0 7$ </td><td> $0 . 3 2 2 \pm 0 . 0 0 4$ </td><td> $1 1 3 . 9 0 9 \pm 2 . 6 9 4$ </td><td> $0 . 7 0 7 \pm 0 . 0 1 9$ </td><td> $\mathbf { 2 8 . 0 6 4 \pm 0 . 5 6 7 }$ </td></tr><tr><td>FUnIE-GAN [38]</td><td> $9 . 5 8 1 \pm 0 . 1 8 3$ </td><td> $0 . 2 3 8 \pm 0 . 0 1 0$ </td><td> $0 . 3 6 4 \pm 0 . 0 1 9$ </td><td> $1 5 3 . 0 4 3 \pm 2 0 . 1 8 3$ </td><td> $0 . 6 9 7 \pm 0 . 0 0 7$ </td><td> $3 3 . 7 3 7 \pm 1 . 8 6 3$ </td></tr><tr><td>DeWaterGAN [23]</td><td> $1 0 . 3 1 3 \pm 0 . 2 5 0$ </td><td> $0 . 3 8 2 \pm 0 . 0 1 8$ </td><td> $0 . 3 4 3 \pm 0 . 0 1 0$ </td><td> $1 3 8 . 3 6 5 \pm 7 . 5 4 1$ </td><td> $0 . 6 4 0 \pm 0 . 0 5 1$ </td><td> $3 1 . 8 0 2 \pm 1 . 0 9 8$ </td></tr><tr><td>DiffWater [28]</td><td> $9 . 8 6 2 \pm 0 . 3 3 9$ </td><td> $0 . 2 5 7 \pm 0 . 0 3 3$ </td><td> $0 . 3 3 2 \pm 0 . 0 1 1$ </td><td> $1 1 7 . 0 3 0 \pm 7 . 2 3 5$ </td><td> $0 . 7 2 2 \pm 0 . 0 5 5$ </td><td> $2 9 . 6 5 2 \pm 3 . 0 6 3$ </td></tr><tr><td>ResShift [48]</td><td> $9 . 8 4 4 \pm 0 . 3 3 5$ </td><td> $0 . 3 1 3 \pm 0 . 0 1 7$ </td><td> $0 . 3 5 3 \pm 0 . 0 1 2$ </td><td> $1 4 2 . 8 2 9 \pm 1 0 . 5 3 3$ </td><td> $0 . 6 0 1 \pm 0 . 0 2 4$ </td><td> $3 1 . 7 9 8 \pm 1 . 0 4 0$ </td></tr><tr><td>SS-UIE [49]</td><td> $1 0 . 0 4 4 \pm 0 . 0 8 4$ </td><td> $0 . 3 7 5 \pm 0 . 0 0 0$ </td><td> $0 . 3 4 8 \pm 0 . 0 1 2$ </td><td> $1 4 1 . 3 4 1 \pm 1 1 . 4 0 6$ </td><td> $\mathbf { 0 . 5 8 2 \pm 0 . 0 1 5 }$ </td><td> $3 1 . 5 6 8 \pm 1 . 6 6 8$ </td></tr><tr><td>U-Shape Transformer [50]</td><td> $9 . 2 8 2 \pm 0 . 0 9 7$ </td><td> $0 . 2 7 9 \pm 0 . 0 1 3$ </td><td> $0 . 3 8 0 \pm 0 . 0 0 7$ </td><td> $1 6 6 . 6 4 2 \pm 6 . 6 5 2$ </td><td> $0 . 7 9 4 \pm 0 . 0 1 1$ </td><td> $3 5 . 5 2 0 \pm 1 . 4 6 3$ </td></tr><tr><td>KAUSTics-Net (ours)</td><td> $9 . 6 5 2 \pm 0 . 1 3 9$ </td><td> $0 . 2 9 1 \pm 0 . 0 2 2$ </td><td> $0 . 3 7 4 \pm 0 . 0 0 6$ </td><td> $1 6 5 . 3 4 8 \pm 8 . 4 7 7$ </td><td> $0 . 6 7 1 \pm 0 . 0 6 7$ </td><td> $3 3 . 9 4 4 \pm 0 . 3 8 8$ </td></tr></table>

<table><tr><td>No sungiint</td><td></td><td>0.374</td><td>0.342</td><td>142.779</td><td>0.643</td><td>32.785</td></tr><tr><td>Unprocessed input</td><td>10.492</td><td></td><td></td><td></td><td></td><td> $2 9 . 0 6 1 \pm 0 . 0 8 1$ </td></tr><tr><td>CycleGAN [37]</td><td> $1 0 . 1 8 5 \pm 0 . 1 4 2$ </td><td> $\mathbf { 0 . 3 6 0 \pm 0 . 0 1 3 }$ </td><td> $\mathbf { 0 . 3 1 6 \pm 0 . 0 0 5 }$ </td><td> $\mathbf { 1 0 4 . 0 4 0 \pm 3 . 5 2 1 }$ </td><td> $0 . 6 9 0 \pm 0 . 0 3 1$ </td><td></td></tr><tr><td>UGAN [27]</td><td> ${ \bf 1 0 . 3 1 0 \pm 0 . 1 2 5 }$ </td><td> $0 . 3 3 4 \pm 0 . 0 0 6$ </td><td> $0 . 3 2 2 \pm 0 . 0 0 4$ </td><td> $1 1 4 . 3 3 2 \pm 2 . 5 2 1$ </td><td> $0 . 7 5 8 \pm 0 . 0 1 6$ </td><td> $\mathbf { 2 8 . 1 1 0 \pm 0 . 5 6 5 }$ </td></tr><tr><td>FUnIE-GAN [38]</td><td> $9 . 4 0 5 \pm 0 . 1 5 3$ </td><td> $0 . 2 1 7 \pm 0 . 0 1 2$ </td><td> $0 . 3 7 2 \pm 0 . 0 1 8$ </td><td> $1 6 0 . 1 2 0 \pm 1 9 . 9 0 5$ </td><td> $0 . 7 3 6 \pm 0 . 0 1 3$ </td><td> $3 4 . 5 5 3 \pm 1 . 8 4 1$ </td></tr><tr><td>DeWaterGAN [23]</td><td> $1 0 . 2 7 8 \pm 0 . 2 3 2$ </td><td> $0 . 3 5 4 \pm 0 . 0 1 3$ </td><td> $0 . 3 4 3 \pm 0 . 0 0 9$ </td><td> $1 3 8 . 3 0 0 \pm 7 . 3 4 6$ </td><td> $0 . 6 9 9 \pm 0 . 0 4 3$ </td><td> $3 1 . 7 7 5 \pm 1 . 0 7 6$ </td></tr><tr><td>DiffWater [28]</td><td> $9 . 8 3 2 \pm 0 . 3 2 9$ </td><td> $0 . 2 4 4 \pm 0 . 0 3 7$ </td><td> $0 . 3 3 3 \pm 0 . 0 1 1$ </td><td> $1 1 7 . 7 9 2 \pm 6 . 8 9 6$ </td><td> $0 . 7 3 9 \pm 0 . 0 5 6$ </td><td> $2 9 . 7 2 6 \pm 3 . 0 0 1$ </td></tr><tr><td>ResShift [48]</td><td> $9 . 8 0 4 \pm 0 . 2 6 0$ </td><td> $0 . 2 9 1 \pm 0 . 0 1 3$ </td><td> $0 . 3 5 4 \pm 0 . 0 0 9$ </td><td> $1 4 3 . 1 1 1 \pm 8 . 1 2 9$ </td><td> $\mathbf { 0 . 6 4 3 \pm 0 . 0 2 2 }$ </td><td> $3 1 . 9 3 8 \pm 0 . 7 0 3$ </td></tr><tr><td>SS-UIE [49]</td><td> $1 0 . 0 7 3 \pm 0 . 0 7 3$   $9 . 2 2 0 \pm 0 . 0 8 4$ </td><td> $0 . 3 4 3 \pm 0 . 0 0 2$   $0 . 2 6 9 \pm 0 . 0 1 1$ </td><td> $0 . 3 4 6 \pm 0 . 0 1 1$ </td><td> $1 3 9 . 5 3 5 \pm 1 0 . 7 2 7$ </td><td> $0 . 6 4 7 \pm 0 . 0 1 3$ </td><td> $3 1 . 3 3 9 \pm 1 . 5 8 2$ </td></tr><tr><td>U-Shape Transformer [50]</td><td></td><td></td><td> $0 . 3 8 2 \pm 0 . 0 0 6$ </td><td> $1 6 8 . 4 9 5 \pm 6 . 2 5 1$ </td><td> $0 . 8 3 0 \pm 0 . 0 1 1$ </td><td> $3 5 . 7 6 0 \pm 1 . 4 3 6$ </td></tr><tr><td>KAUSTics-Net (ours)</td><td> $9 . 6 3 2 \pm 0 . 1 4 7$ </td><td> $0 . 2 6 8 \pm 0 . 0 1 9$ </td><td> $0 . 3 7 4 \pm 0 . 0 0 4$ </td><td> $1 6 5 . 0 1 6 \pm 7 . 4 2 3$ </td><td> $0 . 7 2 9 \pm 0 . 0 6 2$ </td><td> $3 3 . 9 4 8 \pm 0 . 2 7 0$ </td></tr></table>

The corresponding augmented observations illustrate how the same base textures appear under two rendered caustic patterns, two Gaussian-blurred variants, and three empirical color-transfer presets. These transformations introduce different local illumination and radiometric appearances but do not expand the number of independent source scenes. Color presets represent neither measured depth levels nor calibrated light-attenuation coefficients. Visual variation within this survey is not evidence of generalization across independent coastal environments.

TABLE S2  
SEA-UNDISTORT ZERO-SHOT NO-REFERENCE METRICS ON THE SAME 120 HELD-OUT SCENES PER CONFIGURATION AND THE SAME CHECKPOINTS AS TABLE S1. RESULTS ARE MEAN ± SAMPLE STANDARD DEVIATION OVER THREE SOURCE-TRAINING SEEDS. SHARPNESS AND ENTROPY ARE DESCRIPTIVE DIAGNOSTICS AND ARE NOT RANKED.
<table><tr><td>Method</td><td>NIQE↓</td><td>BRISQUE↓</td><td>PIQE↓</td><td>ILNIQE ↓</td><td>LIQE ↑</td><td>CLIPIQ ↑</td><td>MUSIQ_AVA ↑</td><td>Sharpness</td><td>Entropy</td></tr><tr><td colspan="10">All effects</td></tr><tr><td>Unprocessed input</td><td> $\overline { { 8 . 3 4 8 } }$ </td><td>30.378</td><td>23.864</td><td>64.283</td><td>1.935</td><td>0.388</td><td>3.205</td><td>1380.087</td><td>5.211</td></tr><tr><td>CycleGAN [37]</td><td> $\mathbf { 6 . 7 1 0 \pm 0 . 2 8 6 }$ </td><td> $2 0 . 1 6 6 \pm 6 . 8 4 4$ </td><td> $\mathbf { 1 0 . 6 2 1 \pm 1 . 9 4 5 }$ </td><td> $7 4 . 5 9 0 \pm 6 . 2 2 0$ </td><td> $1 . 5 2 5 \pm 0 . 1 2 0$ </td><td> $0 . 3 2 5 \pm 0 . 0 4 3$ </td><td> $3 . 0 7 2 \pm 0 . 0 3 3$ </td><td>191.879 ± 47.371</td><td> $5 . 2 5 7 \pm 0 . 2 9 4$ </td></tr><tr><td>UGAN [27]</td><td> $7 . 3 8 9 \pm 0 . 1 6 6$ </td><td> $2 6 . 3 5 8 \pm 2 . 4 4 9$ </td><td> $1 6 . 1 5 4 \pm 0 . 4 9 5$ </td><td> $9 8 . 2 3 3 \pm 0 . 3 3 9$ </td><td> $1 . 0 3 3 \pm 0 . 0 0 4$ </td><td> $0 . 1 9 3 \pm 0 . 0 0 4$ </td><td> $2 . 6 8 8 \pm 0 . 0 2 8$ </td><td> $1 0 6 . 6 4 0 \pm 1 4 . 3 1 0$ </td><td> $5 . 3 7 9 \pm 0 . 1 0 6$ </td></tr><tr><td>FUnIE-GAN [38]</td><td> $1 0 . 5 8 0 \pm 0 . 1 4 2$ </td><td> $\mathbf { 2 0 . 0 0 7 \pm 1 2 . 7 5 5 }$ </td><td> $3 7 . 9 4 0 \pm 7 . 2 4 9$ </td><td> $1 1 2 . 4 4 1 \pm 3 . 8 6 8$ </td><td> $1 . 1 6 5 \pm 0 . 0 6 6$ </td><td> $0 . 3 0 1 \pm 0 . 0 3 6$ </td><td> $3 . 0 7 5 \pm 0 . 0 6 5$ </td><td> $7 0 8 . 7 8 0 \pm 4 3 2 . 1 4 9$ </td><td> $5 . 7 3 2 \pm 0 . 2 9 7$ </td></tr><tr><td>DeWaterGAN [23]</td><td> $9 . 3 9 2 \pm 0 . 6 2 6$ </td><td> $4 2 . 3 9 0 \pm 3 . 0 8 9$ </td><td> $3 9 . 8 2 2 \pm 4 . 8 4 1$ </td><td> $7 9 . 8 5 3 \pm 3 . 5 5 6$ </td><td> $1 . 6 2 9 \pm 0 . 0 7 1$ </td><td> $0 . 2 9 6 \pm 0 . 0 2 6$ </td><td> $2 . 9 0 4 \pm 0 . 0 1 9$ </td><td> $4 2 8 . 5 0 6 \pm 5 1 . 4 5 6$ </td><td> $4 . 7 8 2 \pm 0 . 1 2 6$ </td></tr><tr><td>DiffWater [28]</td><td> $6 . 7 4 6 \pm 0 . 8 4 3$ </td><td> $2 4 . 2 9 0 \pm 3 . 7 2 0$ </td><td> $2 6 . 8 6 7 \pm 7 . 6 7 9$ </td><td> $1 0 8 . 6 6 7 \pm 3 . 9 0 7$ </td><td> $1 . 1 3 2 \pm 0 . 0 8 1$ </td><td> $0 . 3 5 2 \pm 0 . 0 2 8$ </td><td> $2 . 9 0 0 \pm 0 . 1 1 3$ </td><td> $9 0 2 . 1 2 7 \pm 4 0 9 . 3 5 3$ </td><td> $5 . 8 4 5 \pm 0 . 3 1 0$ </td></tr><tr><td>ResShift [48]</td><td> $1 1 . 8 4 7 \pm 0 . 1 4 7$ </td><td> $3 6 . 2 8 7 \pm 1 . 0 3 4$ </td><td> $4 1 . 1 7 1 \pm 3 . 7 5 4$ </td><td> $7 7 . 3 4 3 \pm 7 . 1 2 3$ </td><td> $\mathbf { 2 . 0 1 2 \pm 0 . 0 9 4 }$ </td><td> $\mathbf { 0 . 4 4 3 \pm 0 . 0 2 2 }$ </td><td> $3 . 0 5 7 \pm 0 . 0 4 9$ </td><td> $1 0 0 4 . 8 0 3 \pm 1 9 4 . 4 8 2$ </td><td> $5 . 5 7 9 \pm 0 . 1 3 7$ </td></tr><tr><td>SS-UIE [49]</td><td> $9 . 4 8 2 \pm 0 . 1 4 2$ </td><td> $3 3 . 1 1 4 \pm 0 . 2 9 9$ </td><td> $2 7 . 9 1 2 \pm 2 . 0 8 2$ </td><td> $\mathbf { 6 7 . 0 4 6 \pm 0 . 4 9 7 }$ </td><td> $1 . 6 1 8 \pm 0 . 0 3 3$ </td><td> $0 . 3 5 0 \pm 0 . 0 1 6$ </td><td> $3 . 1 0 6 \pm 0 . 0 5 7$ </td><td> $9 1 7 . 3 7 1 \pm 6 6 . 6 7 7$ </td><td> $5 . 3 4 1 \pm 0 . 0 3 0$ </td></tr><tr><td>U-Shape Transformer [50]</td><td> $1 0 . 3 7 2 \pm 0 . 3 1 8$ </td><td> $5 5 . 1 5 2 \pm 2 . 1 7 0$ </td><td> $5 6 . 6 5 0 \pm 3 . 7 3 1$ </td><td> $1 1 0 . 7 6 3 \pm 2 . 4 2 9$ </td><td> $1 . 5 9 4 \pm 0 . 0 5 8$ </td><td> $0 . 3 2 3 \pm 0 . 0 2 3$ </td><td> $2 . 5 1 7 \pm 0 . 0 0 8$ </td><td> $5 4 . 8 0 6 \pm 1 7 . 3 7 2$ </td><td> $4 . 5 7 3 \pm 0 . 2 4 7$ </td></tr><tr><td>KAUSTics-Net (ours)</td><td> $8 . 8 5 1 \pm 1 . 3 5 8$ </td><td> $3 3 . 0 2 2 \pm 1 1 . 3 7 9$ </td><td> $3 7 . 9 9 2 \pm 7 . 2 0 7$ </td><td> $6 7 . 4 7 7 \pm 1 . 6 0 2$ </td><td> $1 . 6 4 9 \pm 0 . 0 8 7$ </td><td> $0 . 3 8 9 \pm 0 . 0 3 3$ </td><td> $\mathbf { 3 . 1 1 9 \pm 0 . 0 3 7 }$ </td><td> $1 1 7 0 . 9 8 5 \pm 4 3 7 . 0 5 3$ </td><td> ${ \underline { { 5 . 8 9 8 \pm 0 . 1 4 4 } } }$ </td></tr></table>

$$
7 . 3 3 3 \pm 0 . 3 7 7
$$

$$
7 . 6 5 1 \pm 0 . 3 8 2
$$

$$
\mathbf { 1 5 . 9 5 1 \pm 5 . 5 7 7 }
$$

$$
2 0 . 1 4 1 \pm 2 . 6 4 7
$$

$$
\mathbf { 1 0 . 1 0 6 \pm 1 . 2 5 2 }
$$

$$
1 6 . 3 0 6 \pm 0 . 6 0 6
$$

$$
7 1 . 7 0 4 \pm 5 . 7 2 3
$$

$$
1 1 . 4 8 2 \pm 0 . 2 8 1
$$

$$
2 1 . 2 2 1 \pm 1 4 . 9 6 8
$$

$$
1 . 4 3 0 \pm 0 . 0 5 7
$$

$$
9 2 . 6 0 3 \pm 0 . 2 8 0
$$

$$
4 0 . 7 6 3 \pm 7 . 2 0 8
$$

$$
0 . 3 2 7 \pm 0 . 0 4 7
$$

$$
1 . 0 2 3 \pm 0 . 0 0 3
$$

$$
3 . 1 3 9 \pm 0 . 0 5 7
$$

$$
1 1 4 . 3 4 9 \pm 3 . 1 9 7
$$

$$
0 . 1 8 1 \pm 0 . 0 0 6
$$

$$
4 0 . 8 5 0 \pm 4 . 5 6 1
$$

$$
9 . 5 6 8 \pm 0 . 7 5 2
$$

$$
4 . 9 8 2 \pm 0 . 3 2 9
$$

$$
1 . 1 4 3 \pm 0 . 0 6 9
$$

$$
4 2 . 8 0 2 \pm 6 . 5 5 1
$$

$$
2 . 7 2 7 \pm 0 . 0 3 2
$$

$$
1 0 9 . 7 0 4 \pm 1 5 . 5 4 2
$$

$$
0 . 2 9 6 \pm 0 . 0 4 9
$$

$$
8 0 . 9 1 6 \pm 5 . 3 2 5
$$

$$
5 . 1 8 8 \pm 0 . 1 3 2
$$

$$
3 . 1 1 2 \pm 0 . 0 6 4
$$

$$
2 3 . 6 8 3 \pm 4 . 0 4 8
$$

$$
\mathbf { 6 . 6 8 5 \pm 0 . 8 9 7 }
$$

$$
1 . 4 0 7 \pm 0 . 1 0 2
$$

$$
7 2 6 . 0 0 3 \pm 4 9 1 . 6 7 6
$$

$$
2 8 . 2 8 1 \pm 8 . 7 2 9
$$

$$
0 . 2 7 4 \pm 0 . 0 2 7
$$

$$
1 0 8 . 2 3 5 \pm 4 . 5 3 1
$$

$$
5 . 6 5 0 \pm 0 . 3 6 3
$$

$$
3 2 . 0 9 9 \pm 1 . 5 0 2
$$

$$
1 2 . 2 0 6 \pm 0 . 1 4 2
$$

$$
2 . 8 5 4 \pm 0 . 0 1 3
$$

$$
4 3 . 7 8 8 \pm 4 . 3 6 2
$$

$$
1 . 1 3 4 \pm 0 . 0 8 4
$$

$$
1 2 0 . 3 4 5 \pm 2 2 . 2 5 1
$$

$$
7 2 . 4 2 0 \pm 9 . 1 3 4
$$

$$
7 . 0 1 2 \pm 0 . 4 0 8
$$

$$
1 9 . 5 7 4 \pm 0 . 2 1 2
$$

$$
2 1 . 8 8 4 \pm 2 . 9 5 1
$$

$$
4 . 3 3 8 \pm 0 . 1 1 4
$$

$$
0 . 3 5 3 \pm 0 . 0 2 8
$$

$$
2 . 8 8 7 \pm 0 . 1 0 6
$$

$$
\mathbf { 1 . 8 4 9 \pm 0 . 1 6 4 }
$$

$$
1 1 . 0 7 2 \pm 0 . 4 9 6
$$

$$
8 7 4 . 1 4 5 \pm 4 4 7 . 9 4 0
$$

$$
5 3 . 3 4 4 \pm 3 . 0 6 0
$$

$$
\mathbf { 6 3 . 0 5 0 \pm 0 . 9 1 5 }
$$

$$
5 9 . 1 4 5 \pm 5 . 7 2 4
$$

$$
\mathbf { 0 . 4 4 0 \pm 0 . 0 3 4 }
$$

$$
4 3 . 2 1 7 \pm 7 . 3 9 7
$$

$$
8 . 8 4 0 \pm 1 . 4 4 9
$$

$$
1 0 6 . 9 2 1 \pm 2 . 5 1 5
$$

$$
1 . 4 9 2 \pm 0 . 0 2 9
$$

$$
3 2 . 6 8 2 \pm 1 3 . 8 9 4
$$

$$
3 . 0 7 1 \pm 0 . 0 6 2
$$

$$
5 . 7 4 8 \pm 0 . 3 5 8
$$

$$
6 4 . 2 7 9 \pm 2 . 2 9 7
$$

$$
8 6 8 . 9 3 9 \pm 2 6 2 . 1 4 6
$$

$$
0 . 3 5 6 \pm 0 . 0 1 2
$$

$$
1 . 6 5 9 \pm 0 . 0 5 3
$$

$$
0 . 3 1 2 \pm 0 . 0 3 8
$$

$$
3 . 1 3 7 \pm 0 . 0 8 4
$$

$$
0 . 3 6 5 \pm 0 . 0 4 0
$$

$$
2 . 5 7 1 \pm 0 . 0 1 9
$$

$$
2 7 6 . 9 3 1 \pm 1 1 . 2 8 0
$$

$$
5 . 3 7 1 \pm 0 . 1 8 2
$$

$$
\mathbf { 3 . 1 6 2 \pm 0 . 0 1 4 }
$$

$$
5 1 . 8 0 2 \pm 1 7 . 0 0 2
$$

$$
4 . 9 9 2 \pm 0 . 0 7 1
$$

$$
4 . 1 7 9 \pm 0 . 2 6 2
$$

$$
5 . 7 4 4 \pm 0 . 1 8 2
$$

$$
6 . 7 5 1 \pm 0 . 3 5 9
$$

$$
7 . 4 4 6 \pm 0 . 3 3 5
$$

$$
1 8 . 6 8 0 \pm 6 . 9 0 1
$$

$$
\mathbf { 9 . 7 9 7 \pm 1 . 1 0 4 }
$$

$$
1 9 . 8 2 1 \pm 2 . 9 4 0
$$

$$
1 6 . 3 2 5 \pm 0 . 9 5 5
$$

$$
1 1 . 9 3 0 \pm 0 . 2 8 5
$$

$$
9 3 . 2 7 3 \pm 0 . 1 0 8
$$

$$
2 2 . 4 2 6 \pm 1 4 . 7 2 9
$$

$$
4 2 . 4 0 0 \pm 6 . 7 4 3
$$

$$
1 . 4 1 9 \pm 0 . 1 1 3
$$

$$
1 3 2 . 7 0 3 \pm 6 9 . 0 3 6
$$

$$
1 . 0 3 1 \pm 0 . 0 0 1
$$

$$
1 2 0 . 5 5 2 \pm 4 . 3 6 3
$$

$$
0 . 3 2 5 \pm 0 . 0 5 4
$$

$$
5 . 1 1 8 \pm 0 . 3 5 4
$$

$$
3 . 0 9 7 \pm 0 . 0 4 9
$$

$$
9 . 0 3 7 \pm 0 . 9 1 4
$$

$$
3 2 . 5 0 8 \pm 6 . 9 5 7
$$

$$
1 . 1 4 8 \pm 0 . 0 6 5
$$

$$
0 . 1 7 7 \pm 0 . 0 0 6
$$

$$
1 0 5 . 5 0 9 \pm 1 5 . 5 4 0
$$

$$
8 7 . 9 0 7 \pm 5 . 5 9 8
$$

$$
2 . 7 3 2 \pm 0 . 0 2 0
$$

$$
5 . 2 2 3 \pm 0 . 1 3 9
$$

$$
6 7 9 . 3 3 8 \pm 4 6 2 . 6 5 2
$$

$$
0 . 3 0 4 \pm 0 . 0 5 0
$$

$$
2 3 . 2 8 9 \pm 3 . 9 9 4
$$

$$
2 8 . 3 6 1 \pm 9 . 3 1 9
$$

$$
6 . 7 0 1 \pm 0 . 9 1 5
$$

$$
3 . 1 2 2 \pm 0 . 0 7 2
$$

$$
1 . 3 9 0 \pm 0 . 1 1 9
$$

$$
5 . 5 9 4 \pm 0 . 3 6 7
$$

$$
1 2 . 0 2 9 \pm 0 . 1 2 2
$$

$$
1 0 8 . 7 8 5 \pm 4 . 3 8 5
$$

$$
8 8 . 0 2 7 \pm 1 6 . 3 6 3
$$

$$
3 0 . 7 9 1 \pm 1 . 6 4 5
$$

$$
4 2 . 4 5 5 \pm 4 . 5 0 5
$$

$$
2 . 7 9 5 \pm 0 . 0 3 5
$$

$$
\mathbf { 6 . 6 7 4 \pm 0 . 1 7 6 }
$$

$$
5 . 7 1 6 \pm 0 . 3 6 2
$$

$$
\mathbf { 1 8 . 6 0 1 \pm 0 . 3 2 2 }
$$

$$
8 3 7 . 9 0 7 \pm 4 4 5 . 4 8 3
$$

$$
7 6 . 9 5 3 \pm 8 . 8 8 0
$$

$$
0 . 3 5 8 \pm 0 . 0 3 0
$$

$$
1 9 . 7 3 8 \pm 2 . 3 1 5
$$

$$
\mathbf { 1 . 8 2 0 \pm 0 . 1 3 0 }
$$

$$
1 0 . 4 9 2 \pm 0 . 4 1 9
$$

$$
5 4 . 2 3 6 \pm 2 . 3 7 7
$$

$$
5 8 . 8 8 1 \pm 4 . 9 8 9
$$

$$
\mathbf { 6 7 . 0 2 9 \pm 0 . 6 4 9 }
$$

$$
2 . 9 0 3 \pm 0 . 1 1 0
$$

$$
4 2 . 7 1 2 \pm 7 . 6 4 2
$$

$$
1 1 0 . 0 9 2 \pm 3 . 0 2 3
$$

$$
3 2 . 1 5 3 \pm 1 3 . 4 6 5
$$

$$
1 . 3 8 5 \pm 0 . 0 3 2
$$

$$
9 . 0 9 5 \pm 1 . 4 5 5
$$

$$
\mathbf { 0 . 4 3 8 \pm 0 . 0 3 0 }
$$

$$
6 7 . 4 5 8 \pm 2 . 5 0 1
$$

$$
5 . 3 3 6 \pm 0 . 1 8 1
$$

$$
1 . 8 0 3 \pm 0 . 0 4 7
$$

$$
1 . 5 0 0 \pm 0 . 1 1 6
$$

$$
3 . 0 9 8 \pm 0 . 0 6 1
$$

$$
0 . 3 3 9 \pm 0 . 0 2 3
$$

$$
0 . 3 2 9 \pm 0 . 0 3 8
$$

$$
3 . 0 6 7 \pm 0 . 0 8 1
$$

$$
0 . 3 6 3 \pm 0 . 0 4 1
$$

$$
2 1 2 . 2 3 5 \pm 7 . 7 2 8
$$

$$
2 . 5 5 7 \pm 0 . 0 1 0
$$

$$
\mathbf { 3 . 1 7 4 \pm 0 . 0 2 5 }
$$

$$
5 . 0 1 4 \pm 0 . 0 5 8
$$

$$
1 0 4 7 . 9 3 4 \pm 4 2 2 . 9 8 5
$$

$$
4 . 1 7 3 \pm 0 . 2 6 9
$$

$$
5 . 6 6 0 \pm 0 . 1 8 6
$$

Total: 1,280.000 million  
![](images/6a7e3ccc890abd2fb2d1b7eaed6d958292ef1d381c23330547bee3982204eef6.jpg)

![](images/59af14f786270d1c542893e6aa4857db81605eb637b3a51a6560b5ec74b86ce7.jpg)  
Fig. S2. Verified reference-image pixel inventory. Training, validation, and test contain 1,040, 120, and 120 reference patches. Original 1000 x 1000 references total 1,280.000 million spatial pixels; resized 256 x 256 references total 83.886 million. References are counted once without their ten degraded observations. Axes have different vertical scales; pixel totals do not measure unique geographic or habitat coverage.

![](images/f0bbb2640e6acb5c88ed8bfea1c0318e1bc9d7f52c4ef1922b1248e2617e0657.jpg)

![](images/868d971d798a01de83481ac7d2c251a21b14a84af7da231c4e45f92a0bea1853.jpg)

Fig. S3. Representative calm-water KAUSTics reference patches grouped by descriptive appearance: (a) hard-bottom or rock-like, (b) sand-dominated, (c) vegetation-like or mixed benthic, and (d) dark or low-visibility. These labels are qualitative descriptions and do not indicate field-validated habitat classes species, or measured water depths.  
![](images/77ab1c1eaf41b613d6b084d6e4281fd9437573f4a98f5816b0ef2fe16edb6cb1.jpg)

![](images/ce9be291cb6bc874bcef3399d115691e8f363b226254aa7c2510e58361195f66.jpg)

![](images/0d4363ed354aa80a27f11e2275715d1183239e87316c934753552fcdecc96e3c.jpg)

![](images/900a2cbf09fff84911fa2765ec362dee0fdd03ebb7b17fb64f70e1213d97c9ce.jpg)

![](images/adf32526b9f9666f6c86cebc72d1cea44e39faa63bc4da0cb460ae11cf77b12f.jpg)

![](images/cd07b3f33e0d3874485840db8fad306c375d9db0cde0cb5240c36bb1d899043b.jpg)

![](images/7eb9953171be8acf84fa5ab314cb161b186304a8a9bee1594622883fb3a6a33e.jpg)  
(a)

![](images/67e21be768c2fba42c576196a695d9a071c77735446f57f904568246051a1ef6.jpg)  
(b)

![](images/c941a7f5a50847dbf9ebf078616415bafa578da928ff919486531b7e7e10533c.jpg)  
(c)

![](images/f9c6eb3005693c692d539a7505b9589e159685d6aab050358ce87a94ab52cee5.jpg)

![](images/5b3ffd376c5db7ed9702b669d02895cd3fa897368b128836ac2f4a18dfe1a09e.jpg)  
(d)

![](images/31a2caa2260eab6df73fb711c5fef774413b97652b4af09d2d0d57aa4c2b9f4a.jpg)  
(e)

![](images/1e03fae3c6fe2c630d01ea0bc73563c2466aaa5d17de104883b74120dcbe2701.jpg)  
(f)

![](images/668df100b0940a2670226438e6e99ad0faef7831117deec4cf59f6ee9c9fd423.jpg)  
(g)

![](images/3fefc93305b14a9b242bfebc3c36ded219fea7962eab9dfa8df7605f6b45632a.jpg)  
(h)  
Fig. S4. KAUSTics references and their rendered or augmented counterparts. Rows follow the descriptive appearance groups of Supplementary Fig. S3 Columns show (a) calm-water references, (b–c) two rendered caustic patterns, (d–e) their Gaussian-blurred variants, and (f–h) empirical color-transfer presets Color presets are not physical depth classes.