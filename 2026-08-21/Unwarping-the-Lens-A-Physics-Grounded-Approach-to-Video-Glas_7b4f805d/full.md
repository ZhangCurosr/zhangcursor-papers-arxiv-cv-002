# Unwarping the Lens: A Physics-Grounded Approach to Video Glasses Removal

Radim Spetlik<sup>1∗</sup> , David Futschik<sup>2</sup> , Radek Danecek<sup>2</sup> , Feitong Tan<sup>2</sup>, Ziqian Bai<sup>2</sup>, Rohit Pandey<sup>2</sup>, and Yinda Zhang<sup>2</sup>

<sup>1</sup> Czech Technical University in Prague, Faculty of Electrical Engineering <sup>2</sup> Google, <sup>∗</sup> work done while at Google

Abstract. High-fidelity removal of eyeglasses from video is a major challenge in facial attribute editing, as the underlying facial geometry is often obscured by complex refractive distortions and view-dependent specular reflections. While large-scale generative priors have shown promise in eyeglasses removal via static image inpainting, they often lack the structural constraints necessary to maintain identity, expression, and pose, leading to visible “identity drift” in both static images and dynamic sequences. In this paper, we propose a novel transfer framework that addresses the stochastic nature of generative priors. Our pipeline first extracts highfidelity synthetic face images from a commercial-grade generative model (Nano Banana, Gemini 3 Pro Image), regularizes them via a three-stage structural filtering process to preserve identity, expression, and pose, and finally applies physically-based simulation of lens optics during training to provide diverse, paired data. This process transfers Nano Banana’s photo-realistic, multi-view knowledge into a specialized restoration architecture, JFSnet (Joint Feature-Spatial network). JFSnet integrates DINOv2-based semantic features with a convolutional decoder for spatial reconstruction, leveraging translation equivariance constraints to improve temporal consistency and high-frequency detail preservation. Evaluations on the curated Flickr-Faces-HQ (FFHQ) subset (12,163 images) show that our approach achieves high fidelity and structural accuracy, while maintaining inference speed of 27.68 FPS. In perceptual studies on CelebV-Text video sequences, our results are consistently preferred over difusion and GAN-based baselines for ocular consistency, temporal stability, and overall restoration quality.

Keywords: Image Editing · Video Editing · Glasses Removal

## 1 Introduction

The growth of high-resolution video content has increased the demand for sophisticated facial attribute editing. While significant progress has been made in global style transfer and large-scale object manipulation [2, 11, 25], precise local editing remains a dificult task. Eyeglasses, specifically, are not just blocking layers; they are complex optical elements that introduce depth-dependent refractive distortions and view-dependent specular reflections. Standard generative inpainting models [29, 40] tend to struggle to separate these rare optical efects from the underlying facial geometry, which can lead to “identity drift” and structural inconsistencies in both static images and videos.

![](images/4990370f22b9b44785e55de9d399d74749225cb3548edadb04e4f9d7123abca2.jpg)  
(a) Input  
Frame

![](images/6618ebdc8dc55558b61f4db4136689cfe2c03cc697ab2ada3da6cae6096b5a4b.jpg)  
(b) Our  
Approach

![](images/9ed228e619c02c4947be6f3576ee5f891aff9689b7cc1a6bda051bd953c0f421.jpg)  
(c) Runway Gen-4.5 [30]

![](images/c163ed5adf3ecb05bf6c67e7d23d6b1337fe7791841e367fc8a3264b4dd809c7.jpg)  
(d) Nano  
Banana [10]

![](images/3f0c9c0dae60e14c0c9685e741e0f940ac32d1f0887de1544a430cae3ddb2761.jpg)  
(e) Pro-  
Painter [47]  
Fig. 1: Joint Feature-Spatial network removes glasses in videos producing results that are both temporally consistent and structurally faithful. Given an input video frame (a), our approach (b) efectively erases the glasses while preserving facial identity. In comparison, the commercial video editing model Runway Gen-4.5 [30] alters the subject’s expression and introduces blurriness (c). Similarly, the image-based editor Nano Banana [10] fails to preserve the expression when applied to video frames (d). The video inpainting method ProPainter [47] faces diferent limitations, resulting in incomplete glasses removal (e). In contrast, our method (b) ensures coherent removal and maintains high-fidelity structural integrity throughout the sequence. Source: CelebV-Text [39].

Existing approaches to glasses removal [19, 32] primarily rely on stochastic generative priors to hallucinate the occluded regions. However, in the presence of strong refractive lenses, the human visual system relies on stable geometric cues, such as the continuity of the head contour and the undistorted shape of the eyes, to maintain identity. When a model relies on black-box priors to reconstruct the eye area, it often produces results that look realistic in isolation but can fail to preserve the subject’s unique structural features, such as the exact expression, pose, and identity. Furthermore, applying these methods frame-byframe in a naive manner leads to temporal flickering, as the generative process is unconstrained across the temporal domain.

In this work, we propose a novel transfer framework that addresses the stochastic nature of these generative priors. Our approach leverages large-scale models like Nano Banana (Gemini 3 Pro Image) [10], which provide photorealistic results with multi-view consistency, but often struggle to preserve finegrained identity, expression, and pose. To address these limitations, we propose improvements to both data curation and model architecture. On the side of data generation, we introduce a structured curation and augmentation pipeline: (i) we perform multi-view generative synthesis of eyeglass-free (“clean”) and glasseswearing identities in sets of 13 poses and expressions using Nano Banana, (ii) we apply a structural filtering process — first identifying pose shifts via background $L _ { 1 }$ error, then filtering gaze and identity drift via localized structural consistency analysis, and finally utilizing a learned reconstruction threshold — to ensure a high-fidelity curated source, and (iii) we finally apply physically-based substitution of lens optics during training to provide diverse and realistic pairs. This process allows us to transfer the knowledge of large-scale generative models into a restoration architecture, JFSnet, to achieve inference speeds suitable for real-time video applications.

In contrast to latent difusion approaches that are constrained by the manifold of a fixed pre-trained autoencoder, our method operates directly in pixel space. This allows JFSnet to preserve facial details that might be lost during latent reconstruction. By leveraging a self-supervised DINOv2-based encoder, we capture stable semantic priors that are invariant to local distortions, while a precision-focused convolutional decoder reconstructs fine facial features. To address the challenge of temporal instability without the high computational cost of multi-frame propagation or flow estimation, we enforce translation equivariance with a training constraint. By encouraging the network to commute with spatial shifts, we improve the coherence of restored features across frames, helping to suppress temporal flickering.

## In summary, our contributions are as follows. We

(i) formulate video glasses removal as the ofline transfer of a generative prior into a deterministic, feed-forward model, and introduce a pipeline that combines curated generative face sets with physically-based augmentation of lens optics during training;

(ii) present a framework utilizing JFSnet, a ViT-CNN architecture enabling identity preservation and temporal consistency;

(iii) evaluate on the curated FFHQ subset, outperforming state-of-the-art generative and video inpainting baselines with inference speed of 27.68 FPS;

(iv) demonstrate through perceptual studies that our approach is preferred across gaze preservation, temporal stability, removal quality, identity preservation, and sharpness.

Project site is available at radimspetlik.github.io/unwarpingthelens/.

## 2 Related Work

## 2.1 Eyeglasses Removal

Early work in glasses removal primarily focused on static portrait editing using Generative Adversarial Networks (GANs). ByeglassesGAN [19] and subsequent work [23] leveraged GAN-based architectures to synthesize eyeglass-free ocular regions, with the latter utilizing 3D synthetic data to improve the restoration of cast shadows. More recently, video-specific approaches have emerged to address temporal consistency. V-LASIK [32] introduces identity-preserving constraints and color normalization to align restored regions with the rest of the face. Similarly, IP-FaceDif [1] utilizes difusion priors to handle the removal of eyewear in dynamic sequences. While these methods show promise, they often exhibit identity drift and local blurring because they rely on stochastic generative priors that lack explicit geometric constraints. Our work difers by utilizing a physicsbased simulation of lens optics during training, allowing the model to learn to invert the specific refractive distortions introduced by eyewear.

## 2.2 Generative Image and Video Editing

Advances in difusion models have enabled broad image editing capabilities [5, 12, 37]. InstructPix2Pix [5] and LEDITS [37] allow for instruction-based and semantic editing, while video-to-video translation models [8,15,27] propagate edits across frames to maintain coherence. However, many of these methods operate in the latent space of pre-trained autoencoders, which introduce a bottleneck that inevitably sacrifices fine-grained spatial details like skin texture and eye geometry. Furthermore, while general-purpose models excel at global style transfer, they often struggle with precise local attribute removal, frequently failing to preserve the underlying facial structure.

Video inpainting and object removal methods [41, 44, 47] address occlusions by propagating information from adjacent frames. However, in the context of glasses removal, the “background” is the subject’s own eyes — critical features for identity that are often refracted or partially occluded rather than entirely missing. Generic inpainting approaches lack the specialized facial priors needed to accurately reconstruct these details, often leading to hallucinatory results that do not meet the bar of perceived geometric fidelity.

## 2.3 Identity Preservation and 3D-Aware Editing

Preserving identity is a core challenge in facial editing. Recent works utilize pretrained Vision Transformers (ViTs) as semantic identity priors [26, 34], while 3D-aware methods leveraging Morphable Models (3DMMs) [3] or Neural Radiance Fields (NeRFs) [24] maintain consistency under varying viewpoints. Recent methods integrate physical optical transport — refraction, shadows, and specularities — into 3D morphable models and neural radiance fields to accurately model eyewear [20,21,42]. Similarly, 3D Gaussian Splatting enables high-fidelity dynamic avatars [22, 28, 31, 43], though these representations rarely support removal or refraction. While some explicitly regress 3D facial topology to handle occlusion [46], these representations often demand multi-view calibration or expensive optimization. Our approach bridges the gap between 2D translation and 3D geometric reasoning by incorporating physics-based refraction models, ensuring identity preservation without the need for per-scene 3D reconstruction or expensive overhead.

## 3 Method

We propose a novel transfer framework for consistent glasses removal that addresses the tendency of generative priors toward identity divergence and the introduction of structural inconsistencies (see Fig. 1). Our approach utilizes

![](images/f2d4be1082dc8c11409b456f712254bc0f6addb5db9aa57b95e9ecac28cdd684.jpg)  
Fig. 2: Dataset creation pipeline. Raw multi-pose face sets are generated from the Nano Banana prior and enriched with segmentation masks, monocular depth, and parametric face model fits. These sets undergo multi-stage filtering (joint background/ocular L1 Filtering, followed by U-Net reconstruction) and an As-Rigid-As-Possible (ARAP) warp to align clean and glasses-wearing pairs, giving the final dataset.

a restoration network, JFSnet, which learns to invert complex optical efects while preserving identity, expression, and pose.

## 3.1 Generative Data Creation and Filtering

To support our training pipeline, we constructed a dataset of eyeglass-free (“clean”) and glasses-wearing face pairs using a generative workflow powered by Nano Banana (Gemini 3 Pro Image) [10] shown in Fig. 2. For each synthetic identity, we prompted the model to generate a set of 13 high-fidelity clean images capturing a diverse range of head poses and facial expressions. We further generated a standalone image of eyeglasses and a corresponding set of composed images where the subject wears the glasses in each pose, along with precise segmentation masks for the left lens, right lens, and the frames for every sample in the sequence. See prompts in Supplementary Sec. S1.2. This provided us with a comprehensive set of multi-pose and multi-expression pairs with ground-truth frame textures and pixel-accurate lens regions. A challenge in generative data synthesis is maintaining spatial alignment between the eyeglass-free and glasses-wearing pairs due to a number of efects, mainly the propensity of ViTs to overcompress spatial information and stochasticity in latent generative models. To address this, we used the composed Nano Banana images and their associated masks to define the precise geometric boundaries of the frames and lenses in each pose. Furthermore, we employ an As-Rigid-As-Possible (ARAP) [13] deformation mesh optimized with landmark constraints and a multi-scale photometric loss to align the clean face to the exact pose and facial configuration found in the glasses-wearing image (see Supplementary Sec. S1.9).

To ensure high data quality, we performed a three-stage hard-sampling-based filtering process. In the first two stages, we perform background and ocular $L _ { 1 }$ filtering (L1 Filtering in Fig. 2) to filter out identities with significant shifts in head pose, gaze, or eye identity. This process efectively eliminates samples where the generative model introduced unwanted expression drift; such expression drift is exemplified by Nano Banana in Fig. 1 (d). This stage remains crucial because we down-weight the physics-based lens substitution for extreme head poses specifically rotations approaching a profile view (beyond roughly three-quarters of the way from frontal to full profile) — where monocular depth-based simulation becomes less reliable and we rely on the generative model’s native outputs. In the third stage, we trained a lightweight U-Net on the remaining pairs to perform a preliminary restoration task, which serves as a quality proxy; identities with high reconstruction error were discarded as they typically exhibited residual misalignments or generative artifacts. This elaborate filtering resulted in a final training dataset of 1,860 identities exhibiting high structural and photometric consistency across the 13-image sets (see the Supplementary Sec. S1.1).

![](images/4b5def5898a2eec81244504c126ae845d1d8a81b370659424065d28e1da5a87d.jpg)  
−6

![](images/f4b0cf66f774013a5afed82f2a5537d96be1bfb88c03c5e8030e9dd4dc4412f3.jpg)  
−4

![](images/f076c9feb175e229ad85469ffc8322deec1ab7b87f43dcb8d19efc08a8b9f291.jpg)  
−2

![](images/cfa1491aa24bc1356f78fd9499d3340fac2939247bd758939d215b4779db482b.jpg)  
+1  
Fig. 3: Refraction simulation. Physics-based lens distortion for $- 6 , - 4 , - 2$ , and +1 diopters on a sample from our dataset. Our refraction model captures realistic magnification and minification while preserving facial identity and structure.

## 3.2 Physically-Grounded Training Augmentation

Although this dataset is rich in identities, the optical efects represented in generated data are far more constrained, likely following the generative model’s simplicity bias. To address the lack of large-scale paired data, we introduce a physics-based augmentation pipeline that simulates the optical properties of lenses on the curated generative face sets (see Fig. 3). This pipeline operates during training to provide a source of consistent and diverse training pairs.

Refraction Simulation (Fig. 3). To ensure realistic placement and distortion, we first identify the lens boundaries using precise segmentation masks provided by Nano Banana. The relative 3D pose of the head is estimated by fitting a 3D face parametric model [7] to the input frame to extract the global rotation R and translation t relative to the camera. We define the optical center of each lens by back-projecting the detected pupil landmarks using the depth map from a singleframe depth estimation model [38]. To resolve the scale ambiguity inherent in monocular depth estimation, we assume a standard adult head width to anchor the metric scale. The lenses are then positioned by projecting these metric 3D pupil coordinates 15mm along the head pose direction (the normal vector derived from R). We assume a standard perspective camera model with a typical portrait focal length to initialize the 3D projection. The corrective lens is represented as a physical 3D object defined by two spherical surfaces with radii $R _ { 1 }$ and $R _ { 2 }$ positioned relative to the eye centers. These radii are derived from the target diopter strength D using Vogel’s Rule [4]: $D _ { 1 } = D / 2 + 6 . 0$ and $D _ { 2 } = D - D _ { 1 }$ where $D = ( n - 1 ) ( 1 / R _ { 1 } - 1 / R _ { 2 } )$ . For each pixel within the lens boundary, a ray is cast and intersected with these surfaces. At each interface, the ray direction is updated via Snell’s law $( n \approx 1 . 5 )$ . The resulting intersection with the facial surface defines the source coordinates for the dense warp field W, simulating the depth-dependent magnification of prescription eyewear. Our rendering pipeline operates on high-resolution grids to preserve detail.

Reflection Simulation. Specular reflections R are simulated by leveraging Nano Banana to generate an inverse-view environment map, conditioned on the subject’s portrait, that approximates the scene behind the camera, sampled along the reflected view ray $\mathbf { r } = \mathbf { i } - 2 ( \mathbf { n } ^ { \top } \mathbf { i } ) \mathbf { n }$ (i: incident view direction, n: lens-surface normal). To achieve realistic highlights, we simulate reflections in High Dynamic Range (HDR), allowing for strong specularities of glass surfaces. We further augment these reflections by simulating a variable reflection tint and opacity, accounting for the diverse range of lens coatings and lighting conditions.

Training Input Formation. To provide a physically grounded yet photo-realistic training source, we define the observed image $\bar { I } \stackrel { \cdot } { \in } \bar { \mathbb { R } } ^ { H \times W \times 3 }$ as a composition of generative templates and optical simulations. Let $G _ { c l e a n }$ and $G _ { g l a s s e s }$ be the multi-pose, multi-expression sets from our curated dataset. During training, we synthesize the final input I by selecting a pose-expression pair and substituting the lens regions of $G _ { g l a s s e s }$ with a physically simulated rendering of the corresponding clean face $G _ { c l e a n } \colon$

$$
I = ( 1 - M _ { l e n s } ) \cdot G _ { g l a s s e s } + M _ { l e n s } \cdot ( \mathcal { W } ( G _ { c l e a n } ) + \mathcal { R } ) + \epsilon\tag{1}
$$

where $M _ { l e n s }$ is a binary mask for the lens regions (provided by the left and right lens masks from the generative model), $\mathcal { W } ( \cdot )$ represents the refractive warping operator, and R is the additive HDR specular reflection layer. This formulation ensures that the network learns to invert the refractive transformation W and handle additive reflections R while benefiting from the realistic frame textures and global illumination present in the generative model $G _ { g l a s s e s }$ . The network’s goal is to map this input I back to the original clean identity $G _ { c l e a n }$

## 3.3 JFSnet Architecture

We utilize a restoration network $\left( f _ { \theta } \right)$ , referred to as JFSnet, designed to map the input image to the clean RGB face. As detailed in Fig. S2, JFSnet is a combination of standard architectural blocks: a pre-trained DINOv2 $\left( \mathrm { V i T  – L } / 1 4 \right)$ encoder [26] and a ResNet-based convolutional decoder. To preserve representational power while adapting to the restoration task, we selectively fine-tune the deeper layers of the ViT backbone and fuse multi-level features into the decoder. Crucially, by incorporating a direct skip connection from the input image, we bypass the information bottleneck typical of latent models. This design allows

JFSnet to reconstruct high-frequency details that might fall outside standard generative manifolds, ensuring that the restored facial features remain faithful to the original input. As discussed in Sec. 4.9, this configuration was found to significantly outperform alternative architectures.

## 3.4 Training Objectives

The network $f _ { \theta }$ is trained using a weighted combination of reconstruction and perceptual losses. Let $f _ { \theta } ( I )$ be the predicted clean face and $G _ { c l e a n }$ be the ground truth clean face. The total loss $\mathcal { L }$ is defined as:

$$
\begin{array} { r l } & { \mathcal { L } = \lambda _ { p i x } \mathcal { L } _ { p i x } + \lambda _ { p e r c } \mathcal { L } _ { p e r c } + \lambda _ { e y e - p i x } \mathcal { L } _ { e y e - p i x } } \\ & { \qquad + \lambda _ { e y e - p e r c } \mathcal { L } _ { e y e - p e r c } + \lambda _ { a d v } \mathcal { L } _ { a d v } + \lambda _ { t e m p } \mathcal { L } _ { t e m p } } \end{array}\tag{2}
$$

Global Reconstruction and Perceptual. $\mathcal { L } _ { p i x }$ is the global $L _ { 1 }$ distance between the predicted output and the ground truth. $\mathcal { L } _ { p e r c }$ ensures high-level texture consistency across the full image, defined by the distance between Gram matrices of VGG-19 features.

Eye-specific Objectives. To restore features behind lenses, we utilize localized objectives centered on the center of gravity (CoG) of each lens mask within 64 × 64 pixel blocks. $\mathcal { L } _ { e y e - p i x }$ is the localized $L _ { 1 }$ reconstruction loss, while $\mathcal { L } _ { \it e y e - p e r c }$ is the corresponding Gram-matrix based perceptual loss.

Adversarial Loss. $\mathcal { L } _ { a d v }$ incorporates feedback from a PatchGAN-based critic [14] to ensure realistic texture generation, encouraging the network to produce sharp, high-frequency details. The critic is trained in the standard GAN min-max setting [9], though we omit this formulation for brevity.

Temporal Consistency. To ensure temporally stable results on video data, we enforce translation equivariance during training. For each training sample, we simulate motion by applying a random spatial shift $\varDelta s$ to the input image I. The temporal consistency loss $\mathcal { L } _ { t e m p }$ is defined as:

$$
\mathcal { L } _ { t e m p } = | | f _ { \theta } ( \mathcal { T } _ { \Delta s } ( I ) ) - \mathcal { T } _ { \Delta s } ( f _ { \theta } ( I ) ) | | _ { 1 }\tag{3}
$$

where ${ \tau _ { \varDelta s } ( \cdot ) }$ represents a translation by $\varDelta s .$ . By forcing $f _ { \theta }$ to commute with small spatial translations, we ensure that high-frequency artifacts and identitydefining features move coherently with the subject’s motion. This constraint encourages the model to learn a mapping that is robust to spatial misalignments and resists flickering across frames.

## 4 Experiments

In order to assess the eficacy of our proposed approach, we evaluate the task of glasses removal from images and consistent removal from videos. Our experiments are designed to validate the efectiveness of the physics-based synthesis pipeline and the JFSnet architecture in handling optical distortions and maintaining temporal stability.

## 4.1 Experimental Setup

Implementation Details. We implement our framework using JAX/Flax, training the models on high-resolution images resized to $5 1 8 \times 5 1 8$ pixels. We use the AdamW optimizer with a base learning rate of $3 \times 1 0 ^ { - 4 }$ for the JFSnet and Adversarial Critic. The pre-trained DINOv2 encoder is fine-tuned with a lower learning rate of $1 \times 1 0 ^ { - 5 }$ for its last four layers, while the rest of the backbone is kept frozen. Training is performed on TPU/GPU accelerators for approximately 400,000 batch iterations. For the physics-based augmentation, we sample diopter powers ranging from −6.0 to +1.0 with an assumed refractive index of $n \approx 1 . 5$ . Since negative diopters (minification) introduce visible background artifacts into the eye area while positive diopters (magnification) are harder to distinguish, we focus on minification in our qualitative analysis (see Fig. 4) and simulation. Specular reflections are simulated via sphere mapping with highdynamic-range (HDR) emulation and randomized anti-reflective (AR) coating efects. To mitigate aliasing, we compute a Level of Detail (LOD) map based on the Jacobian of the ray divergence, enabling interpolation between sharp and pre-blurred background images.

Data Augmentation Pipeline. We use our curated dataset of 1,860 synthetic identities (24,180 image pairs) with on-the-fly augmentation applied during training. By randomizing specular reflection intensity and tint to simulate diverse lens coatings, the model encounters a wide variety of optical distortions while remaining anchored to the eyeglass-free ground truth.

## 4.2 Dataset Generation

We constructed our training dataset using the generative workflow and filtering process detailed in Sec. 3.1. This process ensures high structural and photometric consistency across the multi-view generative face sets, providing a robust foundation for the subsequent physics-based augmentation. See Supplementary Sec. S1.1 for a qualitative sample of a single face set.

## 4.3 Datasets

We utilize the following datasets exclusively for qualitative and quantitative evaluation; neither dataset was used during the training phase of our models. Flickr-Faces-HQ (FFHQ) [16] consists of 70,000 high-resolution portrait images with significant variation in demographic factors and backgrounds. For our evaluation, we manually curated images containing eyewear from this dataset, classifying them into two distinct categories based on ocular visibility: (i) individuals wearing corrective glasses with visible eye regions, and (ii) individuals wearing sunglasses or eyewear with significant darkening and specular reflections. This process yielded a refined evaluation set containing 12,163 images with clear glasses and 2,176 images in the sunglasses category. The remaining images in the dataset (≈ 55,000) are utilized as the “without eyewear” reference category. CelebV-Text [39] is a large-scale video dataset capturing dynamic facial expressions. For our qualitative evaluation and perceptual study (Sec. 4.7), we manually curated a subset of 60 video clips featuring subjects with clearly visible eyeglasses.

## 4.4 Baselines

We evaluate our method against representative approaches across three categories. For general video editing, we compare against TokenFlow [8], RAVE [15], and IP-FaceDif [1]. In the domain of video inpainting, we include ProPainter [47], Flow-Guided Transformer (FGT) [44], Inpaint Anything [40], and STTN [41]. We also evaluate image-based facial editing methods including Take-Of-Eyeglasses (TOE) [23], LEDITS [37], and InstructPix2Pix [5].

Additionally, Runway Gen-4.5 is a private, commercial video generation model with significant inference costs, leading us to restrict its evaluation to the human preference study. For difusion-based methods such as TokenFlow, RAVE, LEDITS, and IP-FaceDif, achieving high-fidelity results is often sensitive to parameter tuning and the stability of the underlying inversion process (e.g., DDIM inversion [33]). As finding viable configurations exceeded the practical scope of our evaluation, we focused our human preference study on the examples where baseline methods achieved competitive scores on the FFHQ dataset.

## 4.5 Flickr-Faces-HQ Qualitative Results

We present a qualitative comparison on the FFHQ dataset in Fig. S4. The generative models, such as Nano Banana [10], often reconstruct the eye region with plausible but inaccurate details, leading to subtle identity shifts or gaze changes.

LEDITS [37] struggles to fully remove frame artifacts, whereas our method preserves facial textures and gaze and compensates for the spatial minification of strong refractive lenses (see Fig. 3). See the Supplementary Sec. S1.11 for more results and Sec. S1.12 for failure cases.

## 4.6 CelebV-Text Qualitative Results

The qualitative results for video sequences are shown in Fig. 5. Large-scale videoto-video model Runway Gen-4.5 [30] removes reflections, but often fails to preserve the subject’s gaze and eye identity. Video inpainting methods ProPainter [47] and FGT [44] are primarily designed for object removal and lack the facial priors needed to handle the complex shadows and reflections cast by eyeglasses. These methods often leave behind ghosting artifacts. By contrast, our method leverages its physics-based training, providing temporally stable results that maintain the geometric integrity of the face. Full videos are available in the supplement.

ProPainter [47]

Input Frame

TOE [23]

LEDITS [37]  
![](images/f6226516dd3b45f6959566f5351648f7b0c9fb5e012acd0dc2b4200b7ea6ba52.jpg)  
Fig. 4: Qualitative comparison on high-refraction samples from FFHQ [17]. Close-up eye regions are shown for each subject. While Nano Banana [10] removes reflections, it alters structural details (e.g., gaze shift in row 1, eye closure in row 2). Difusion-based LEDITS [37] exhibits identity drift or fails to remove glasses (row 1). Video inpainting methods [41, 44, 47] often fail to recover underlying facial structures. Our method efectively removes reflections and corrects refraction while preserving fine facial details and maintaining high structural integrity. See Sec. 4.5 for more details.

## 4.7 Perceptual Study

To further evaluate our method, we conducted a perceptual study consisting of two surveys focused on video and image glasses removal. For the video survey, we utilized 60 clips from the CelebV-Text dataset, while the image survey used 60 high-resolution portraits from the FFHQ dataset. We invited 37 participants to compare our results against 4 other representative methods, resulting in a total of 5 methods per question. In each survey, participants were asked to rank the methods based on specific criteria derived from the qualitative characteristics of the task. For the video survey, the evaluation axes included gaze consistency, temporal stability, and overall restoration quality. The image survey focused on gaze preservation, removal completeness, identity preservation, sharpness, and overall removal quality. These criteria allow for a granular assessment of how well each method handles both the physical restoration of ocular details and the generative consistency of the face. The collected preferences are visualized in the heatmap of Fig. 6. Our method is consistently preferred across both temporal and spatial dimensions. While Nano Banana achieves high scores for completeness due to its powerful generative prior, it is consistently outperformed in gaze and identity preservation where its unconstrained outputs diverge from the subject’s original features. Temporal metrics are in the Supplementary Sec. S1.4.

![](images/6b588aaa37e5c96725996f51e7992cfcb27b136856dece97b379b9aed6bc473e.jpg)  
Fig. 5: Qualitative comparison on CelebV-Text [39]. Runway Gen-4.5 removes reflections but changes the gaze and eye identity. V-LASIK causes blur and changes eye shape. TOE struggles with strong reflections. ProPainter, FGT, and STTN only perform inpainting, leaving behind shadows, reflections, or parts of the glasses. Our Joint Feature-Spatial network preserves gaze and fine facial details. See Sec. 4.6.

## 4.8 Quantitative Evaluation

We assess the performance of the methods across several dimensions — gaze preservation (head pose, pupil position, and ocular structure), visual realism, structural fidelity, and computational eficiency.

Gaze preservation is quantified via the Pupil Displacement error (in px), which measures the spatial shift of iris centers between the input and restored output. To avoid evaluation bias, we detect pupils using the MediaPipe FaceMesh model [18], which is distinct from the models used in our data curation pipeline. We calculate the Euclidean distance in normalized coordinates and scale the average displacement by 512 for readability. Note that while physical refraction correction necessitates a minor spatial shift of the ocular features, a low displacement indicates that the model avoids the significant gaze drift common in generative baselines. Inpaint Anything [40] achieves a competitive pupil error as it ideally avoids ocular modification, but is prone to generating artifacts when segmentation fails. Conversely, Nano Banana [10] sufers from significant gaze drift despite serving as our primary data source.

![](images/7e1c8c61596ac5b3c02884ccc226a374b63156c0e626cce618b1dd2381d28c4b.jpg)  
Fig. 6: Preference study on CelebV-Text [39] and FFHQ [17]. Our approach is consistently preferred across both video and image surveys. The heatmap shows the percentage of participants who preferred our result over the baseline for each evaluation criterion. See Sec. 4.7 for more details.

To evaluate the visual realism and quality of the generated results, we employ the Fréchet Inception Distance (FID), which measures the statistical distance between the feature distributions of real and synthesized images to capture distribution-level fidelity.

Structural and geometric stability is measured by the Landmark L2 Error (L2), the Euclidean distance between facial landmarks in the input images (with glasses) and the corresponding landmarks in the restored outputs. This metric ensures that the restoration process does not introduce unwanted spatial distortions or identity-altering pose shifts. We empirically found that ArcFace [6] does not perform well for identity preservation analysis in our case (see Supp. Sec. S1.6). Finally, we report the Inference Speed in frames per second (FPS) to evaluate the practical utility of each method for video applications. We use the results of our perceptual study (Sec. 4.7) to measure temporal stability.

Results on FFHQ. We evaluate our method on the curated FFHQ evaluation set, primarily focusing on the clear glasses category where ocular details remain partially visible. As shown in Table 1, our approach achieves an FID of 0.379 and a landmark L2 error of 0.632. For the FID evaluation, we take the real images from the curated FFHQ “clear glasses” subset (12,163 images), resize them to $5 1 8 \times 5 1 8$ , run our removal pipeline, and compare the resulting distribution of restored images against a size-matched subset of the FFHQ “without eyewear” category. This distribution-level comparison validates that our method correctly restores facial regions to a state statistically indistinguishable from natural portraits without eyewear. The sunglasses category is in the Supplementary Sec. S1.3. Paired metrics and retrained baselines in the Supplementary Sec. S1.5.

Table 1: Quantitative evaluation on the Flickr-Faces-HQ (FFHQ) dataset [16]. We compare our proposed approach against several state-of-the-art image-based, video editing, and video inpainting methods using the Fréchet Inception Distance (FID) ↓, Pupil Displacement ↓, landmark L2 error (L2) ↓, and inference speed (FPS) ↑. Best results in bold. <sup>†</sup>
<table><tr><td>Method</td><td>FID ↓</td><td> $\mathrm { { P u p i l } \left( p x \right) \downarrow }$ </td><td>L2 (px) ↓</td><td>FPS ↑</td></tr><tr><td>Nano Banana [10]</td><td>0.389</td><td> $2 . 8 5 4 \pm 2 . 3 8 7$ </td><td> $0 . 6 9 9 \pm 0 . 4 6 2$ </td><td>1.29</td></tr><tr><td>Inpaint Anything [40]</td><td>0.391</td><td> $2 . 2 5 9 \pm 2 . 2 9 5$ </td><td> $0 . 6 6 4 \pm 0 . 9 0 2$ </td><td>1.84</td></tr><tr><td>InstructPix2Pix [5]</td><td>1.089</td><td> $5 . 5 4 1 \pm 6 . 5 2 2$ </td><td> $8 . 0 3 8 \pm 1 8 . 2 9 3$ </td><td>0.05</td></tr><tr><td>LEDITS [37]</td><td>1.953</td><td> $3 . 9 0 5 \pm 2 . 9 9 9$ </td><td> $1 . 5 2 7 \pm 2 . 4 3 7$ </td><td>0.08</td></tr><tr><td>Take-Off-Eyeglasses [23]</td><td>3.256</td><td> $2 . 5 9 3 \pm 1 . 6 9 6$ </td><td> $0 . 8 7 7 \pm 0 . 7 7 3$ </td><td>13.31</td></tr><tr><td>IP-FaceDiff [1]</td><td>3.832</td><td> $1 7 . 6 0 7 \pm 8 . 7 1 8$ </td><td> $2 1 . 2 5 7 \pm 2 7 . 1 7 1$ </td><td>0.04</td></tr><tr><td>RAVE [15]</td><td>11.559</td><td> $3 . 5 7 7 \pm 2 . 3 7 3$ </td><td> $2 . 3 5 6 \pm 2 . 7 7 3$ </td><td>0.16</td></tr><tr><td>TokenFlow [8]</td><td>2.430</td><td> $3 . 6 8 9 \pm 2 . 8 1 2$ </td><td> $1 . 9 0 5 \pm 1 . 6 7 8$ </td><td>0.04</td></tr><tr><td>Flow-Guided Transformer [44]</td><td>0.569</td><td> $3 . 0 7 8 \pm 2 . 7 1 4$ </td><td> $0 . 9 6 5 \pm 0 . 9 5 2$ </td><td>1.28</td></tr><tr><td>ProPainter [47]</td><td>0.408</td><td> $2 . 6 8 4 \pm 2 . 8 3 8$ </td><td> $0 . 9 4 7 \pm 1 . 0 9 5$ </td><td>8.77</td></tr><tr><td>Ours</td><td>0.379</td><td> $\mathbf { 2 . 2 4 9 \pm 1 . 9 5 7 }$ </td><td> $\mathbf { 0 . 6 3 2 \pm 0 . 2 7 1 }$ </td><td>27.68</td></tr></table>

## 4.9 Architectural Ablation

We compare the JFSnet configuration against standard baseline architectures to validate our choice of combining semantic and spatial blocks. Our evaluation includes a pure Vision Transformer $\left( \mathrm { V i T - L } / 1 4 \right)$ , a standard U-Net with a ResNet-34 backbone, and the proposed JFSnet (see the Supplementary Sec. S1.7).

## 4.10 Ablation Study

To validate our design choices, we investigate the impact of the core components of our pipeline, as summarized in the Supplementary Table S8.

Our results demonstrate that both reflection and refraction simulations are critical for achieving low FID scores; removing either component leads to a degradation. While the temporal consistency loss $\left( \lambda _ { \mathrm { t e m p } } \right)$ yields the same FID score for static images, it is included for its role in temporal stability; the combination of $\lambda _ { \mathrm { t e m p } } ,$ rigorous dataset filtering, and ARAP warping results in an FID of 0.379. The dataset filtering process—which includes $L _ { 1 }$ thresholding and hard-sample mining—ensures the quality of the generative pairs, while the ARAP warping (see Supplementary Sec. S1.9) accounts for residual pose shifts to maintain pixelperfect alignment. Unfreezing the last four layers of the DINOv2 encoder provides a measurable boost in performance. Finally, the localized eye-specific losses $( \lambda _ { \mathrm { e y e - p i x } }$ and $\lambda _ { \mathrm { e y e - p e r c } } )$ enforce recovering details in the eye region.

## 5 Conclusion

In this paper, we presented a physics-grounded generative-to-deterministic transfer framework for consistent eyeglasses removal from video, bridging the gap between large-scale generative priors and physical restoration. By training our JFSnet architecture on a generative-to-physical synthesis pipeline, we achieved state-of-the-art visual fidelity (FID 0.379) while maintaining real-time performance at 27.68 FPS. Our approach leverages multi-level feature fusion and enforced translation equivariance to preserve gaze and identity across dynamic sequences, significantly outperforming existing baselines in perceptual studies.

## Acknowledgments

The research reported in this paper has been partly funded by BMK, BMAW, and the State of Upper Austria in the frame of the SCCH competence center INTEGRATE [Project 1.6 TFI (Transferable Intelligence)] part of the FFG COMET Competence Centers for Excellent Technologies Programme, and by the Czech Technical University in Prague grant No. SGS26/074/OHK3/1T/13.

## References

1. Anand, T., Garg, A., Mitra, K.: IP-FaceDif: Identity-Preserving Facial Video Editing with Difusion. pp. 248–258 (2025)

2. Bar-Tal, O., Chefer, H., Tov, O., Herrmann, C., Paiss, R., Zada, S., Ephrat, A., Hur, J., Li, Y., Michaeli, T., others: Lumiere: A Space-Time Difusion Model for Video Generation. In: SIGGRAPH Asia 2024 Conference Papers (2024)

3. Blanz, V., Vetter, T.: A morphable model for the synthesis of 3D faces. In: Proceedings of the 26th annual conference on Computer graphics and interactive techniques. pp. 187–194. ACM Press/Addison-Wesley Publishing Co. (1999). https://doi.org/10.1145/311535.311556

4. Brooks, C.W., Borish, I.M.: System for Ophthalmic Dispensing. Butterworth-Heinemann (2007). https://doi.org/10.1016/B978-0-7506-7480-5.X5001-1

5. Brooks, T., Holynski, A., Efros, A.A.: InstructPix2Pix: Learning to follow image editing instructions. In: Proceedings of IEEE Conference on Computer Vision and Pattern Recognition. pp. 18392–18402 (2023)

6. Deng, J., Guo, J., Xue, N., Zafeiriou, S.: ArcFace: Additive Angular Margin Loss for Deep Face Recognition. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 4690–4699 (2019)

7. Egger, B., Smith, W.A.P., Tewari, A., Wuhrer, S., Zollhoefer, M., Beeler, T., Bernard, F., Bolkart, T., Kortylewski, A., Romdhani, S., Theobalt, C., Blanz, V., Vetter, T.: 3D Morphable Face Models—Past, Present, and Future. ACM Trans. Graph. 39(5), 157:1–157:38 (Jun 2020). https://doi.org/10.1145/3395208

8. Geyer, M., Bar-Tal, O., Bagon, S., Dekel, T.: TokenFlow: Consistent Difusion Features for Consistent Video Editing (2023)

9. Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., Bengio, Y.: Generative adversarial nets. In: Advances in Neural Information Processing Systems. pp. 2672–2680 (2014)

10. Google: Gemini 3 Pro Image - Model Card. Model Card, Google (2025), https:// storage.googleapis.com/deepmind-media/Model-Cards/Gemini-3-Pro-Image-Model-Card.pdf

11. Google: Veo2 (2025), https://veo2.ai/

12. Hertz, A., Mokady, R., Tenenbaum, J., Aberman, K., Pritch, Y., Cohen-Or, D.: Prompt-to-Prompt Image Editing with Cross-Attention Control. In: International Conference on Learning Representations (2023)

13. Igarashi, T., Moscovich, T., Hughes, J.F.: As-rigid-as-possible shape manipulation. ACM Trans. Graph. 24(3), 1134–1141 (2005). https://doi.org/10.1145/ 1073204.1073323

14. Isola, P., Zhu, J.Y., Zhou, T., Efros, A.A.: Image-to-image translation with conditional adversarial networks. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 1125–1134 (2017)

15. Kara, O., Kurtkaya, B., Yesiltepe, H., Rehg, J.M., Yanardag, P.: RAVE: Randomized Noise Shufling for Fast and Consistent Video Editing with Difusion Models. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (2024)

16. Karras, T., Laine, S., Aila, T.: A style-based generator architecture for generative adversarial networks. In: Proceedings of IEEE Conference on Computer Vision and Pattern Recognition. pp. 4401–4410 (2019)

17. Karras, T., Laine, S., Aila, T.: A Style-Based Generator Architecture for Generative Adversarial Networks. IEEE Transactions on Pattern Analysis and Machine Intelligence 43(12), 4217–4228 (2021). https://doi.org/10.1109/TPAMI.2020. 2970919

18. Kartynnik, Y., Ablavatski, A., Grishchenko, I., Grundmann, M.: Real-time Facial Surface Geometry from Monocular Video on Mobile GPUs. In: CVPR Workshop on Computer Vision for Augmented and Virtual Reality 2019 (2019)

19. Lee, Y.H., Lai, S.H.: ByeGlassesGAN: Identity Preserving Eyeglasses Removal for Face Images (Aug 2020). https://doi.org/10.48550/arXiv.2008.11042, http: //arxiv.org/abs/2008.11042, arXiv:2008.11042 [cs]

20. Li, G., Meka, A., Mueller, F., Buehler, M.C., Hilliges, O., Beeler, T.: EyeNeRF: a hybrid representation for photorealistic synthesis, animation and relighting of human eyes. ACM Trans. Graph. 41(4), 166:1–166:16 (2022). https://doi.org/ 10.1145/3528223.3530130

21. Li, J., Saito, S., Simon, T., Lombardi, S., Li, H., Saragih, J.: MEGANE: Morphable Eyeglass and Avatar Network. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 12769–12779 (2023). https://doi.org/10. 1109/CVPR52729.2023.01228

22. Liang, H., Ge, Z., Majee, S., Tiwari, A., Godaliyadda, G.M.D., Veeraraghavan, A., Balakrishnan, G.: FastAvatar: Instant 3D Gaussian Splatting for Faces from Single Unconstrained Poses (2025). https://doi.org/10.48550/arXiv.2508.18389

23. Lyu, J., Wang, Z., Xu, F.: Portrait eyeglasses and shadow removal by leveraging 3d synthetic data. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 3429–3439 (2022)

24. Mildenhall, B., Srinivasan, P.P., Tancik, M., Barron, J.T., Ramamoorthi, R., Ng, R.: NeRF: representing scenes as neural radiance fields for view synthesis. Communications of the ACM 65(1), 99–106 (2021). https://doi.org/10.1145/3503250 25. OpenAI: SORA (2025), https://sora.com

26. Oquab, M., Darcet, T., Moutakanni, T., Vo, H.V., Szafraniec, M., Khalidov, V., Fernandez, P., Haziza, D., Massa, F., El-Nouby, A., Assran, M., Ballas, N., Galuba, W., Howes, R., Huang, P.Y., Li, S.W., Misra, I., Rabbat, M., Sharma, V., Synnaeve, G., Xu, H., Jegou, H., Mairal, J., Labatut, P., Joulin, A., Bojanowski, P.: DINOv2: Learning Robust Visual Features without Supervision. Transactions on Machine Learning Research (2023)

27. Qi, C., Cun, X., Zhang, Y., Lei, C., Wang, X., Shan, Y., Chen, Q.: FateZero: Fusing attentions for zero-shot text-based video editing. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 15932–15942 (2023)

28. Qian, S., Kirschstein, T., Schoneveld, L., Davoli, D., Giebenhain, S., Nießner, M.: GaussianAvatars: Photorealistic Head Avatars with Rigged 3D Gaussians. In: 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 20299–20309 (2024). https://doi.org/10.1109/CVPR52733.2024.01919

29. Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B.: High-resolution image synthesis with latent difusion models. In: Proceedings of IEEE Conference on Computer Vision and Pattern Recognition. pp. 10684–10695 (2022)

30. Runway AI, Inc.: Runway Gen-4.5: High-Fidelity Text-to-Video and In-Context Editing (2025), https://runwayml.com/research/introducing-runway-gen-4.5

31. Serifi, G., Buehler, M.C.: HyperGaussians: High-Dimensional Gaussian Splatting for High-Fidelity Animatable Face Avatars. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2026)

32. Shalev-Arkushin, R., Azulay, A., Halperin, T., Richardson, E., Bermano, A.H., Fried, O.: V-LASIK: Consistent Glasses-Removal from Videos Using Synthetic Data. In: ICLR 2025 Workshop on Will Synthetic Data Finally Solve the Data Access Problem? (2025). https://doi.org/10.48550/arXiv.2406.14510

33. Song, J., Meng, C., Ermon, S.: Denoising Difusion Implicit Models (2021)

34. Spetlik, R., Futschik, D., Sýkora, D.: StructuReiser: A Structure-preserving Video Stylization Method. Computer Graphics Forum 44(4), e70161 (2025). https:// doi.org/10.1111/cgf.70161

35. Teed, Z., Deng, J.: RAFT: Recurrent all-pairs field transforms for optical flow. In: Proceedings of the European Conference on Computer Vision (ECCV). pp. 402–419 (2020)

36. Telea, A.: An Image Inpainting Technique Based on the Fast Marching Method. Journal of Graphics Tools 9(1), 23–34 (2004). https://doi.org/10.1080/ 10867651.2004.10487596

37. Tsaban, L., Passos, A.: LEDITS: Real image editing with DDPM inversion and semantic guidance. arXiv preprint arXiv:2307.00522 (2023)

38. Yang, L., Kang, B., Huang, Z., Xu, X., Feng, J., Zhao, H.: Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data. pp. 10371–10381 (2024)

39. Yu, J., Zhu, H., Jiang, L., Loy, C.C., Cai, W., Wu, W.: CelebV-Text: A Large-Scale Facial Text-Video Dataset. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 14805–14814 (2023)

40. Yu, T., Feng, R., Feng, R., Liu, J., Jin, X., Zeng, W., Chen, Z.: Inpaint anything: Segment anything meets image inpainting. arXiv preprint arXiv:2304.06790 (2023)

41. Zeng, Y., Fu, J., Chao, H.: Learning Joint Spatial-Temporal Transformations for Video Inpainting. In: Vedaldi, A., Bischof, H., Brox, T., Frahm, J.M. (eds.) Com-

puter Vision – ECCV 2020. pp. 528–543. Springer International Publishing, Cham (2020). https://doi.org/10.1007/978-3-030-58517-4\_31

42. Zhan, Y., Nobuhara, S., Nishino, K., Zheng, Y.: NeRFrac: Neural Radiance Fields through Refractive Surface. In: 2023 IEEE/CVF International Conference on Computer Vision. pp. 18356–18366 (2023). https://doi.org/10.1109/ICCV51070. 2023.01687

43. Zhang, D., Liu, Y., Lin, L., Zhu, Y., Chen, K., Qin, M., Li, Y., Wang, H.: HRAvatar: High-Quality and Relightable Gaussian Head Avatar. In: Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR). pp. 26285–26296 (2025)

44. Zhang, K., Fu, J., Liu, D.: Flow-guided transformer for video inpainting. In: Proceedings of the European Conference on Computer Vision (ECCV). pp. 74–90 (2022)

45. Zhang, R., Isola, P., Efros, A.A., Shechtman, E., Wang, O.: The Unreasonable Efectiveness of Deep Features as a Perceptual Metric. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 586–595 (2018)

46. Zhao, D., Qi, Y.: Generative Landmarks Guided Eyeglasses Removal 3D Face Reconstruction. In: MultiMedia Modeling: 28th International Conference, MMM 2022, Phu Quoc, Vietnam, June 6–10, 2022, Proceedings, Part II. pp. 109–120. Springer-Verlag, Berlin, Heidelberg (2022). https://doi.org/10.1007/978-3- 030-98355-0\_10

47. Zhou, S., Li, C., Chan, K.C., Loy, C.C.: ProPainter: Improving Propagation and Transformer for Video Inpainting. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 10477–10486 (2023)

## S1 Supplementary

## S1.1 Dataset Overview

In Fig. S1, we illustrate the multi-modal nature of our generated dataset. Each sample identity in our collection is represented by 13 clean and glasses-wearing pairs across a diverse range of head poses and expressions.

![](images/4c09aba853ae8ff6db5b8a35e43253ddd7bdcc63d259ffef50ac02132afde7ef.jpg)  
Fig. S1: Visualization of our generated dataset. Row 1: Clean images with warp grids (green points), dense facial landmarks (green: certain, purple: uncertain), and gaze vectors derived from rotation matrices. Row 2: Glasses compositions. Row 3: Clean images with precise lens and frame segmentations. Row 4: Estimated depth maps. Row 5: environment map (left) and the glasses asset (right) used for composition.

Geometric and Semantic Ground-Truth. Our dataset goes beyond simple image pairs by providing dense geometric and semantic annotations. As shown in the first row of Fig. S1, each clean face is associated with precise warp grids, dense facial landmarks, and gaze vectors derived from the rotation matrices of a fitted 3D face parametric model [7]. The third row highlights the high-fidelity segmentation masks for both the lenses and the frames, which are generated directly by the Nano Banana (NB) prior. These masks provide the pixel-level guidance necessary for both our physics-based simulation and our restoration network.

Depth and Environmental Context. To facilitate realistic optical simulation, we incorporate depth information and environmental context for every identity. The fourth row of Fig. S1 displays estimated depth maps used to compute refractive warps. Furthermore, the final row shows the generative environment maps and standalone eyewear assets. By conditioning reflections on these inverse-view scenes, our simulation captures the lighting conditions of the original portrait, bridging the gap between purely synthetic rendering and generative composition. This comprehensive set of modalities ensures that our restoration model is trained on data that is both photorealistic and structurally grounded.

## S1.2 Data Generation Prompts

The generation of our multi-pose synthetic dataset follows a structured prompting workflow. These instructions are designed to enforce geometric consistency while maintaining demographic diversity.

## Identity and Pose Generation (3×3 Grid).

“CRITICAL INSTRUCTION: Create 9 poses of[PERSON\_DESCRIPTION]. The person making the poses MUST NOT wear glasses. The camera view is the same. Only the head poses and expressions change. It is CRITICAL that you succeed. If not, you will be TERMINATED! Random background that is the same in all 9 poses, talking head framing. Poses/expressions: frontal/neutral, looking halfway towards left shoulder/neutral, looking halfway towards right shoulder/neutral, head tilted back/neutral, frontal/smile without teeth, frontal/smile showing teeth, frontal/raised eyebrows, frontal/frowning, frontal/neutral. Ifyou will not succeed, we will be fired. You MUST succeed! IMPORTANT: No gutter, seamless, images touching, no borders, no frames, zero padding, edge-to-edge layout! CRITICAL: Lighting conditions MUST be realistic, not photo studiolike, amateur photography-like, selfie-like!!!”

## Extreme Poses (2×2 Grid).

“Make a grid of 2x2 images showing the same person in the following poses: extreme left head turn, extreme right turn, extreme chin up (look up), extreme chin down (look down). It is CRITICAL that you succeed.”

Eyewear Reference Generation.

“Generate a single top-right view of common dioptric eyeglasses.” or “Generate a single top-right view of dioptric glasses that look completely diferent to the ones attached.”

Table S1: Quantitative evaluation on the Flickr-Faces-HQ (FFHQ) dataset [16]. We compare our proposed approach against several state-of-the-art image-based, video editing, and video inpainting methods using the Fréchet Inception Distance (FID) ↓ and inference speed (FPS) ↑. We report results for regular glasses, sunglasses, and the full combined test set. Best results are highlighted in bold.
<table><tr><td rowspan="2">Method</td><td colspan="3">Fréchet Inception Distance (FID) ↓</td><td rowspan="2">FPS ↑</td></tr><tr><td>glasses</td><td>sunglasses</td><td>combined</td></tr><tr><td>Inpaint Anything [40]</td><td>0.391</td><td>1.796</td><td>0.337</td><td>1.84</td></tr><tr><td>InstructPix2Pix [5]</td><td>1.089</td><td>1.608</td><td>0.927</td><td>0.05</td></tr><tr><td>LEDITS [37]</td><td>1.953</td><td></td><td>1.953</td><td>0.08</td></tr><tr><td>Take-Off-Eyeglasses [23]</td><td>3.256</td><td>2.741</td><td>2.839</td><td>13.31</td></tr><tr><td>RAVE [15]</td><td>11.559</td><td>16.741</td><td>12.065</td><td>0.16</td></tr><tr><td>TokenFlow [8]</td><td>2.430</td><td></td><td>2.430</td><td>0.04</td></tr><tr><td>Flow-Guided Transformer [44]</td><td>0.569</td><td>1.432</td><td>0.453</td><td>1.28</td></tr><tr><td>ProPainter [47]</td><td>0.408</td><td>1.653</td><td>0.335</td><td>8.77</td></tr><tr><td>Ours</td><td>0.379</td><td>1.979</td><td>0.364</td><td>27.68</td></tr></table>

## Synthetic Eyewear Fitting.

“Have the subject wear the attached glasses. CRITICAL: Do not change the skin, eyes, or anything else. It is CRITICAL that you succeed. If the image will change significantly, you will be TERMINATED!”

Inverse-View Scene Generation.

“Create a person-free picture of the environment on the mirror side (the reverse view) of the attached picture. Location and lighting must match the original picture.”

Semantic Segmentation Masks.

“Create a glasses mask - left lens in blue, right lens in green, frame in red.”

Prompt Parameterization. The [PERSON\_DESCRIPTION] placeholder is dynamically populated with demographic attributes (e.g., hispanic real-world woman in her thirties) and appearance constraints (average-looking, no photo model) to ensure validity and avoid bias toward professional studio photography.

## S1.3 Complete Quantitative Results on FFHQ

The task of sunglass removal points toward pure inpainting as the eyes must be entirely hallucinated. By contrast, our work emphasizes the restoration of facial features through the compensation of optical distortions where ground-truth guidance from the underlying geometry is available.

Table S2: Temporal consistency metrics on CelebV-Text clips. RAFT-L1 is the flow-warping error (previous output warped into the current frame by RAFT-Sintel [35] flow estimated on the inputs); temporal LPIPS (tLPIPS) [45] is an adjacentframe perceptual distance. Both measure stability, not restoration quality. Laplacian variance is a sharpness/detail proxy: lower-detail outputs are smoother and score better on stability metrics, so the comparison is conservative for our method. Best per column in bold.
<table><tr><td>Approach</td><td>RAFT-L1  $\times 1 0 ^ { 3 } \downarrow$ </td><td>tLPIPS  $\times 1 0 ^ { 3 } \downarrow$ </td><td>Laplacian variance↑</td></tr><tr><td>Nano Banana [10]</td><td>9.763</td><td>21.215</td><td>83.9</td></tr><tr><td>LEDITS [37]</td><td>14.455</td><td>27.904</td><td>73.9</td></tr><tr><td>FGT [44]</td><td>6.529</td><td>17.722</td><td>37.0</td></tr><tr><td>TOE [23]</td><td>6.517</td><td>16.569</td><td>19.9</td></tr><tr><td>V-LAŠIK [32]</td><td>7.760</td><td>16.803</td><td>54.9</td></tr><tr><td>STTN [41]</td><td>7.152</td><td>19.658</td><td>66.7</td></tr><tr><td>ProPainter [47]</td><td>6.569</td><td>16.756</td><td>57.8</td></tr><tr><td>Runway Gen-4.5 [30]</td><td>7.970</td><td>16.844</td><td>56.8</td></tr><tr><td>Ours (JFSnet)</td><td>7.682</td><td>18.528</td><td>162.6</td></tr></table>

While our approach achieves the lowest FID in the corrective glasses category (0.379), we note that some methods in Table S1 exhibit lower FID scores on the combined dataset than on the glasses subset alone. This phenomenon is a direct consequence of the Fréchet Inception Distance being a biased estimator, where the bias is approximately proportional to $1 / N$ for N samples. Because the combined set utilizes the full evaluation pool (∼14,000 images), the significant reduction in estimator bias outweighs the quality degradation introduced by the sunglasses subset (∼2,000 images). This results in a “suspicious but correct” statistical outcome where the larger sample size dominates the metric calculation. Our method remains specifically optimized for the high-fidelity restoration of visible ocular details through physical compensation, which is most accurately captured by the glasses-only metric.

## S1.4 Temporal Consistency Evaluation

To complement the perceptual study (Sec. 4.7) with objective measurements, we evaluate temporal stability on the 60 CelebV-Text clips used in the video survey, using common, method-agnostic metrics that require no masks or learned task-specific features. For RAFT-L1, we estimate dense RAFT-Sintel [35] optical flow on the (repeated) input frames, warp the previous output into the current frame, and measure the L1 residual, which compensates for input motion. Temporal LPIPS (tLPIPS) measures the perceptual distance between adjacent output frames. Both quantify stability rather than restoration quality. Laplacian variance is a sharpness/detail proxy: smoother, lower-detail outputs incur lower temporal error, which is why a strongly smoothing method such as TOE [23] ranks well temporally while retaining the least detail (lowest sharpness, 19.9). The comparison is therefore conservative for JFSnet, which produces by far the sharpest outputs (162.6).

Table S3: Temporal-loss ablation. Efect of the translation-equivariance loss $\lambda _ { \mathrm { t e m p } } .$ Restoration metrics use a freshly generated, identity-disjoint paired split (300 identities, 3,900 paired samples); temporal metrics use the CelebV-Text video protocol of Table S2. Despite no paired-video supervision, the temporal loss reduces temporal errors with no spatial-restoration trade-of. Best per column in bold.
<table><tr><td rowspan="2">JFSnet λtemp</td><td colspan="3">Global</td><td colspan="2">Lens region</td><td colspan="2">Temporal</td></tr><tr><td>SSIM↑</td><td>LPIPS↓</td><td> $\begin{array} { c } { { \mathrm { L 1 } } } \\ { { \times 1 0 ^ { 2 } \downarrow } } \end{array}$ </td><td>PSNR↑</td><td>LPIPS↓</td><td>RAFT-L1  $\times 1 0 ^ { 3 } \downarrow$ </td><td>tLPIPS  $\times 1 0 ^ { 3 } \downarrow$ </td></tr><tr><td>x</td><td>0.947</td><td>0.025</td><td>1.01</td><td>27.44</td><td>0.00307</td><td>8.023</td><td>19.520</td></tr><tr><td>V</td><td>0.950</td><td>0.024</td><td>1.00</td><td>27.65</td><td>0.00300</td><td>7.682</td><td>18.528</td></tr></table>

Table S4: Prior baselines retrained on our data. We retrain a video inpainter (ProPainter [47], run on the input image with our lens masks) and a glasses-specific method (TOE [23]) on JFSnet’s training data, and evaluate on a 300-identity paired set (identity-disjoint, freshly generated, not used in any prior training). : as originally reported; \: retrained on our data. Global metrics cover the whole image; lens-region metrics isolate the actual restoration task. Best in bold.
<table><tr><td></td><td colspan="3">Global</td><td colspan="2">Lens region</td></tr><tr><td>Approach</td><td>SSIM↑</td><td>LPIPS↓</td><td>L1↓</td><td>PSNR↑</td><td>LPIPS↓</td></tr><tr><td>ProPainter [47] </td><td>0.907</td><td>0.054</td><td>0.021</td><td>15.28</td><td>0.037</td></tr><tr><td>ProPainter [47] </td><td>0.913</td><td>0.044</td><td>0.020</td><td>16.71</td><td>0.027</td></tr><tr><td>TOE [23] *</td><td>0.916</td><td>0.050</td><td>0.022</td><td>18.98</td><td>0.012</td></tr><tr><td>TOE [23] </td><td>0.940</td><td>0.028</td><td>0.014</td><td>24.19</td><td>0.004</td></tr><tr><td>Ours (JFSnet)</td><td>0.950</td><td>0.024</td><td>0.010</td><td>27.65</td><td>0.003</td></tr></table>

As Table S2 shows, JFSnet is materially more temporally stable than the framewise generative and editing baselines (Nano Banana, LEDITS, Runway Gen-4.5) while preserving the high-frequency detail that conservative inpainters discard.

Table S3 ablates the translation-equivariance loss $\lambda _ { \mathrm { t e m p } }$ on the same protocol. Despite using no paired-video supervision, it reduces both temporal metrics (RAFT-L1 8.023 → 7.682, tLPIPS 19.520 → 18.528) without degrading any spatial-restoration metric, confirming its role in suppressing flicker.

## S1.5 Baselines Retrained on Our Data

To separate the contribution of our data from that of our architecture, we retrain two prior baselines on JFSnet’s training data: a video inpainter (ProPainter [47], applied to the input image with our lens masks) and a glasses-specific method (TOE [23]). All methods are evaluated on a 300-identity paired set (identitydisjoint, freshly generated, not used in any prior training). ProPainter copies non-masked pixels from the input, while TOE processes the full image, so the lens-region metrics in Table S4 isolate the actual restoration task. Even with same-data fine-tuning, both baselines remain well below JFSnet in the lens region $- \ \mathrm { b y \ + 3 . 5 d B }$ (TOE) and +10.9 dB (ProPainter) PSNR — although TOE is glasses-specific and ProPainter is a recent video inpainter. This supports our claim that inverting refraction exploits the distorted pixels, which mask-based pipelines and TOE’s de-shadow/de-glass cascade are less able to use.

Table S5: Establishing the “Expected Biometric Drop” for glass removal. To model the biometric penalty of removing glasses, we treat an image with synthetically heavy sunglasses (70% dark, 15% reflection) as the baseline identity. We then compute the identity shift when analytically “removing” these artifacts: restoring the true clear eyes (“Un-Lensing”) and inpainting the frames (“Un-Framing”) with OpenCV’s implementation of [36]. Combining both establishes a theoretical ceiling for identity preservation during complete glass removal.
<table><tr><td>Modification</td><td>ArcFace Sim. ↑</td></tr><tr><td>Analytic Un-Lensing (Darkening &amp; Reflection Removal)</td><td> $0 . 9 6 8 \pm 0 . 0 1 6$ </td></tr><tr><td>Analytic Un-Framing (Frame Removal)</td><td> $0 . 9 8 2 \pm 0 . 0 3 2$ </td></tr><tr><td>Combined Analytic Removal</td><td> $0 . 9 4 6 \pm 0 . 0 3 7$ </td></tr></table>

## S1.6 ArcFace Identity Similarity Under Eyewear Removal

In the main text (Sec. 4.8) we note that ArcFace [6] similarity is unreliable for identity preservation under eyewear removal. Here we quantify why: successful removal necessarily shifts the biometric embedding, so a lower ArcFace score reflects the task rather than a loss of identity. We compute ArcFace on MediaPipe-aligned 112 × 112 crops with the InsightFace w600k\_r50 model and cosine similarity.

Analytic expected drop. Treating an image with synthetically heavy sunglasses as the baseline identity, we measure the embedding shift induced by analytically removing the eyewear: restoring the clear eyes (“Un-Lensing”) and inpainting the frames (“Un-Framing” [36]). As shown in Table S5, each operation lowers similarity, and their combination establishes a ceiling of 0.946 ± 0.037 for nongenerative removal — the frame geometry itself acts as an identity anchor.

Real-world reference. As an empirical floor, we captured 72 real before/after pairs ( the same subject photographed wearing eyeglasses and again after they were physically removed). ArcFace similarity between these genuine sameidentity pairs is only $0 . 8 7 \pm 0 . 0 2$ . On FFHQ, JFSnet scores $0 . 8 9 \pm 0 . 0 4 - \mathrm { a t }$ or above this real-world reference — whereas a conservative inpainter (Inpaint Anything [40]) scores $0 . 9 5 \pm 0 . 0 3$ by leaving ocular content largely unmodified.

Table S6: Component Analysis of Generative Glass Removal across State-of-the-Art Methods. By selectively substituting generated content into specific geometric regions of the original image, we identify the primary sources of identity shift. The results demonstrate that models with high overall ArcFace scores frequently leave structural frame artifacts (as seen in higher Gen. Frame Only scores) and fail to deeply reconstruct the ocular region (as seen in near-perfect Gen. Lenses Only scores). Our pipeline provides the most structurally complete removal, incurring the expected biometric penalty.
<table><tr><td>Method</td><td>Frame Only (Orig. Lenses)</td><td>Lenses Only  $\left( \mathrm { O r i g . \ F r a m e } \right)$ </td><td> $( { \mathrm { F r a m e s ~ } } + { \mathrm { ~ L e n s e s } } )$ </td><td>Glasses Area Full Generated Output</td></tr><tr><td>Nano Banana</td><td> $0 . 9 5 4 \pm 0 . 0 2 3$ </td><td> $0 . 9 6 2 \pm 0 . 0 2 2$ </td><td> $0 . 9 1 9 \pm 0 . 0 3 1$ </td><td> $0 . 8 8 3 \pm 0 . 0 3 6$ </td></tr><tr><td>Inpaint Anything</td><td> $0 . 9 6 1 \pm 0 . 0 2 0$ </td><td> $0 . 9 8 5 \pm 0 . 0 2 4$ </td><td> $0 . 9 5 0 \pm 0 . 0 2 9$ </td><td> $0 . 9 4 8 \pm 0 . 0 3 0$ </td></tr><tr><td>InstructPix2Pix</td><td> $0 . 9 6 1 \pm 0 . 0 2 6$ </td><td> $0 . 9 2 9 \pm 0 . 0 5 5$ </td><td> $0 . 8 9 3 \pm 0 . 0 8 3$ </td><td> $0 . 6 2 5 \pm 0 . 3 0 4$ </td></tr><tr><td>LEDITS</td><td> $0 . 9 6 5 \pm 0 . 0 2 8$ </td><td> $0 . 9 3 8 \pm 0 . 0 4 5$ </td><td> $0 . 9 0 0 \pm 0 . 0 6 8$ </td><td> $0 . 7 5 5 \pm 0 . 1 2 6$ </td></tr><tr><td>TOE</td><td> $0 . 9 5 4 \pm 0 . 0 2 6$ </td><td> $0 . 9 7 5 \pm 0 . 0 1 4$ </td><td> $0 . 9 3 4 \pm 0 . 0 2 9$ </td><td> $0 . 9 0 8 \pm 0 . 0 3 0$ </td></tr><tr><td>IP-FaceDiff</td><td> $0 . 9 7 9 \pm 0 . 0 1 2$ </td><td> $0 . 9 2 3 \pm 0 . 0 3 9$ </td><td> $0 . 9 0 7 \pm 0 . 0 4 1$ </td><td> $0 . 2 6 6 \pm 0 . 1 2 6$ </td></tr><tr><td>RAVE</td><td> $0 . 9 3 4 \pm 0 . 0 4 5$ </td><td> $0 . 8 6 1 \pm 0 . 0 6 0$ </td><td> $0 . 7 8 6 \pm 0 . 0 8 1$ </td><td> $0 . 3 1 6 \pm 0 . 1 0 9$ </td></tr><tr><td>TokenFlow</td><td> $0 . 9 7 4 \pm 0 . 0 1 9$ </td><td> $0 . 9 4 7 \pm 0 . 0 3 4$ </td><td> $0 . 9 2 3 \pm 0 . 0 4 7$ </td><td> $0 . 7 0 5 \pm 0 . 0 9 6$ </td></tr><tr><td>FGT</td><td> $0 . 9 4 4 \pm 0 . 0 3 0$ </td><td> $0 . 9 7 5 \pm 0 . 0 2 7$ </td><td> $0 . 9 2 0 \pm 0 . 0 3 7$ </td><td> $0 . 8 9 7 \pm 0 . 0 3 7$ </td></tr><tr><td>ProPainter</td><td> $0 . 9 4 0 \pm 0 . 0 3 4$ </td><td> $0 . 9 7 5 \pm 0 . 0 2 7$ </td><td> $0 . 9 1 6 \pm 0 . 0 4 0$ </td><td> $0 . 8 9 0 \pm 0 . 0 4 6$ </td></tr><tr><td>Ours</td><td> $0 . 9 5 1 \pm 0 . 0 2 7$ </td><td> $0 . 9 7 0 \pm 0 . 0 2 3$ </td><td> $0 . 9 2 2 \pm 0 . 0 3 1$ </td><td> $0 . 8 9 4 \pm 0 . 0 3 7$ </td></tr></table>

The reduced similarity is therefore intrinsic to removing the eyewear, not a failure to preserve identity.

Component analysis across methods. Using our precise masks, we selectively substitute only the generated frames, only the generated lenses, or the entire glasses area back into the original clean images (Table S6). Across all methods, replacing only the frame region lowers similarity even when the ocular features are untouched, confirming that frames act as biometric anchors; methods that retain residual frame silhouettes (e.g., IP-FaceDif, TokenFlow) artificially stabilize the score rather than improving restoration. JFSnet’s full-output similarity (0.894) is consistent with the analytic ceiling once frame-cast shadow removal is accounted for, while difusion-based resampling (RAVE, IP-FaceDif) collapses identity $( < 0 . 3 2 )$

## S1.7 Architectural Ablation

In Table S7, we explore various architectural configurations to justify our choice of JFSnet. Purely convolutional architectures (CNN-only) maintain spatial structure but struggle with the global identity consistency required for high-fidelity restoration, resulting in higher FID scores (0.439 vs. 0.379). Conversely, all-Transformer models (ViT-ViT) exhibit significant artifacts without extensive fine-tuning and global skip connections, often failing to recover high-frequency facial textures previously occluded by eyewear.

Table S7: Architectural Ablation. We evaluate the impact of diferent encoderdecoder backbones, global skip connections, and encoder fine-tuning. Our hybrid JFSnet (ViT encoder, CNN decoder, and global skip) achieves the best performance, efectively balancing high-level semantic consistency with low-level spatial detail. Beyond FID, we report paired restoration metrics (SSIM/LPIPS/L1) for each configuration on the 300-identity paired set.
<table><tr><td colspan="4">Global</td><td colspan="4"></td></tr><tr><td>Encoder</td><td>Decoder</td><td>Skip</td><td>Encoder </td><td>SSIM ↑</td><td>LPIPS ↓</td><td>L1 ↓</td><td>FID ↓</td></tr><tr><td>ViT</td><td>ViT</td><td></td><td></td><td>0.941</td><td>0.070</td><td>0.024</td><td>1.131</td></tr><tr><td>ViT</td><td>ViT</td><td></td><td>√</td><td>0.940</td><td>0.170</td><td>0.025</td><td>0.572</td></tr><tr><td>CNN</td><td>CNN</td><td></td><td></td><td>0.895</td><td>0.110</td><td>0.020</td><td>0.493</td></tr><tr><td>CNN</td><td>CNN</td><td>√</td><td></td><td>0.911</td><td>0.113</td><td>0.017</td><td>0.439</td></tr><tr><td>ViT</td><td>ViT</td><td>√</td><td>√</td><td>0.943</td><td>0.171</td><td>0.026</td><td>0.392</td></tr><tr><td>ViT</td><td>CNN</td><td></td><td>√</td><td>0.917</td><td>0.170</td><td>0.025</td><td>0.387</td></tr><tr><td>ViT</td><td>ViT</td><td>√</td><td></td><td>0.928</td><td>0.172</td><td>0.026</td><td>0.384</td></tr><tr><td>ViT</td><td>CNN</td><td>√</td><td>√</td><td>0.932</td><td>0.070</td><td>0.013</td><td>0.379</td></tr></table>

Our hybrid approach (JFSnet) leverages the powerful semantic representation of DINOv2 while utilizing a CNN decoder to recover high-frequency spatial details. The addition of a global skip connection is critical for preserving pixel-level fidelity, particularly for facial features not occluded by the eyeglasses, yielding a significant reduction in FID (0.493 vs. 0.439 for CNN and 0.387 vs. 0.379 for JFSnet). Finally, fine-tuning the last four layers of the ViT backbone (unlocked) provides the necessary adaptation to the restoration task, enabling the model to better handle the complex optical distortions of reflections and refractions.

## S1.8 JFSnet Architecture

The JFSnet restoration network (f<sub>θ</sub>) is designed to map the input image to the clean RGB face by combining the semantic robustness of Vision Transformers with the spatial precision of convolutional networks. It utilizes a DINOv2 (ViT-$\mathrm { L } / 1 4 )$ encoder to capture high-level facial structure and identity features. The output tokens from the ViT backbone are reshaped into a 37 × 37 feature map and processed via a 1 × 1 convolution to match the channel dimension of the decoder. To adapt the pre-trained encoder to the task while maintaining its representational power, we fine-tune only the last four layers of the ViT backbone. The decoder is a custom U-Net-style CNN that aggregates these multi-level features, fusing them with a high-resolution skip connection from the input image to maintain pixel-level fidelity.

## S1.9 As-Rigid-As-Possible (ARAP) Warping

To ensure precise spatial alignment between the Nano Banana clean and glasseswearing pairs, we apply an As-Rigid-As-Possible (ARAP) deformation mesh.

![](images/c40afc59eaa4ac105e6146032ed2d4b5c3e69ee9f83f304f10215aa40b24497a.jpg)  
Fig. S2: Overview of the proposed Joint Feature-Spatial network (JFSnet). Our architecture integrates a pre-trained DINOv2 encoder with a convolutional decoder to restore facial details behind eyeglasses. By leveraging multi-level feature integration and selective fine-tuning, the network bridges high-level semantic identity preservation with high-resolution spatial precision, ensuring consistent restoration across diverse facial poses and optical conditions.

This step accounts for the subtle pose and expression drifts that may persist even in high-fidelity generative sequences.

We construct a 2D mesh $M = ( V , E , T )$ using facial landmarks and a coarse grid of background points. We then optimize the deformed vertex positions $V ^ { \prime }$ by minimizing an energy function:

$$
E ( V ^ { \prime } ) = \lambda _ { r i g i d } E _ { r i g i d } ( V ^ { \prime } ) + \lambda _ { d a t a } E _ { d a t a } ( V ^ { \prime } ) + \lambda _ { p h o t o } E _ { p h o t o } ( V ^ { \prime } )\tag{S1}
$$

where $E _ { r i g i d }$ preserves local facial geometry by finding optimal per-triangle rotations $R _ { T } . ~ E _ { d a t a }$ pulls vertices toward detected landmarks in the glasses image, and $E _ { p h o t o }$ is a multi-scale photometric loss that aligns the appearance of the clean image with the glasses image, particularly in the regions surrounding the frames. For the lens regions, we utilize a robust Geman-McClure-like loss to handle the deliberate occlusions and reflections introduced by the eyewear. This optimization is implemented in JAX and performed for 300 iterations per pair, ensuring that the final synthetic training data is strictly aligned with the underlying facial ground truth.

## S1.10 Full Ablation Study

In Table S8, we provide the full results of our ablation study, examining the impact of physics-based data augmentation (reflections and refractions), our dataset filtering process, as-rigid-as-possible (ARAP) warping, specific loss components, and the fine-tuning strategy for the Vision Transformer encoder. These results validate that each component of our pipeline contributes to the final restoration quality, with the combination of all elements achieving the best performance.

Table S8: Ablation Study. Impact of diferent augmentations, loss components, and encoder freezing strategies on FID scores. Best results are highlighted in bold.
<table><tr><td></td><td>Refr. Refl.</td><td> $\lambda _ { \mathrm { p e r c } }$ </td><td> $\lambda _ { \mathrm { e y e - p i x } }$ </td><td> $\lambda _ { \mathrm { e y e - p e r c } }$ </td><td>λadv</td><td> $\lambda _ { \mathrm { t e m p } }$ </td><td>Enc. </td><td>dataset filtering</td><td>ARAP warping</td><td>FID↓</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.403</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>ーノ</td><td>0.402</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.398</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.398</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.394</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.394</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.391</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.383</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.382</td></tr><tr><td>ー√</td><td></td><td>&lt;ー—√&gt;&gt;</td><td>&lt;ー√—&gt;&gt;&gt;</td><td>ーー</td><td></td><td></td><td>&gt;&gt;ー√&gt;</td><td>——</td><td></td><td>0.380</td></tr><tr><td></td><td>ー</td><td></td><td></td><td></td><td>-ー</td><td>ーー</td><td></td><td></td><td></td><td>0.379</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.379</td></tr></table>

S1.11 Qualitative Results for Additional Methods on FFHQ

In Fig. S3, we provide a broader qualitative comparison against four additional methods: RAVE [15], IP-FaceDif [1], TokenFlow [8], and InstructPix2Pix [5]. While these approaches demonstrate the ability to remove eyeglass frames, they often exhibit diferent modes of failure in maintaining facial fidelity. In Fig. S4 more results are presented.

Stylization and Structural Shift. As shown in Fig. S3, difusion-based video translation methods such as RAVE and TokenFlow tend to prioritize generative consistency over pixel-perfect identity preservation. RAVE often performs an unintended stylization of the face, resulting in an appearance that resembles a smooth, synthetic 3D model rather than a natural portrait. TokenFlow, while maintaining temporal coherence, frequently experiences identity drift where the subject’s original features are largely replaced by the generative prior. Similarly, IP-FaceDif can introduce unintended structural distortions that alter the global facial geometry.

Artifacts and Texture Fidelity. InstructPix2Pix, while ofering a computationally eficient image-to-image editing path, often introduces local artifacts around the ocular region and struggles to fully synthesize the skin texture previously occluded by the frames. These challenges are reflected in the higher FID scores reported in Table S1, where the stochastic nature of the denoising process in these baselines leads to a greater distribution shift from natural face portraits. In contrast, our JFSnet leverages a physically-grounded synthesis pipeline to maintain high-frequency facial details and ensure that the restoration remains faithful to the original identity.

Input Frame  
RAVE [15]  
IP-FaceDif [1]  
TokenFlow [8]  
InstructPix2Pix [5]  
![](images/68c73338eb28b5ac46cc516b233f292ab01ad66db1cb47789b2c726fc5a34650.jpg)  
Fig. S3: Additional qualitative comparison on FFHQ. We show results from image-to-image and video-to-video translation methods that showed lower quantitative performance. While these methods can remove glasses, they often fail to preserve the original facial features or introduce artifacts.

## S1.12 Failure Cases

Typical failure cases of our approach are presented in Fig. S5. These can be classified into four primary categories: (i) incomplete removal of eyewear, (ii) residual shadows cast by the frames on the subject’s face, (iii) minor iris deformation, and (iv) refraction undercorrection. We attribute the first three categories to artifacts or misalignments in the synthetic training data, which could be mitigated through more rigorous dataset cleaning. The fourth category arises from the simplified refraction model used in our simulation, which may not fully capture the complex lens geometries of certain high-power prescriptions.

Input Frame  
Our Approach  
Nano Banana [10]  
TOE [23]  
![](images/90bbb4059421fa63950f1d3011a8dd8151ef99d73b60760b0f0942817933eb3b.jpg)  
ProPainter [47]  
FGT [44]  
STTN [41]  
LEDITS [37]  
Fig. S4: Qualitative comparison on Flickr-Faces-HQ (FFHQ) [17]. Generative models like LEDITS and Nano Banana struggle to maintain gaze consistency and preserve fine facial details. Video completion methods like ProPainter and FGT, being designed for inpainting, often fail to recover underlying facial structures accurately. Our Joint Feature-Spatial network preserves gaze and fine facial details while maintaining high-fidelity structural integrity. See Sec. 4.5 for more details.

![](images/c6d01b7617aa9acf15c07807cec964efdf8ed6eb5821fb1205a94e4c253aafd0.jpg)  
Fig. S5: Failure cases of our approach. We identify several failure modes, shown in row pairs (input and our result) from top to bottom: (i) incomplete removal of eyewear, (ii) incomplete removal of shades cast by glasses on person’s face, (iii) iris deformation, and (iv) refraction undercorrection.