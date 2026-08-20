# USR-Drive: Unified Driving Scene Representation via Joint Denoising of 3D Gaussians and Boxes

Li-Heng Chen<sup>1,2</sup> Haokai Pang<sup>2</sup> Chengye Su<sup>1</sup> Jiarun Liu<sup>1</sup> Qifeng Chen<sup>1</sup> Ziqian Ni<sup>1</sup> Jianxin Huang<sup>2</sup> Shi-Sheng Huang<sup>3</sup> Hongbo Fu<sup>2B</sup> Sheng Yang<sup>1B</sup> <sup>1</sup>NIO <sup>2</sup>HKUST <sup>3</sup>Beijing Normal University

![](images/0d45c272fbb5ab14f1529c52d21d68e1ee34188cd9542a9d527da351d96c6235.jpg)  
Figure 1: We present a Unified Generative Framework for 3D Driving Scenes, addressing Reconstruction and Detection as two mutually beneficial representations. (a) Conventional methods isolate reconstruction from detection, causing temporal smearing and ungrounded boxes. (b) We insteadjointly denoise 3D geometry and object-centric layouts as co-evolving latents via a Multi-Modal Difusion Transformer (MMDiT). (c) Our unified generation mitigates temporal smearing and yields physically consistent, tightly grounded 3D bounding boxes.

## Abstract

Spatial representation learning for autonomous driving aims to map raw visual signals into structured 3D scene representations, where object-centric bounding boxes and renderingoriented 3D primitives (e.g., 3D Gaussians) serve as two distinct yet highly complementary levels for scene understanding. Existing methods typically treat dynamic reconstruction and instance-level perception as separate tasks, despite their shared goal of estimating the underlying 3D world state. As a result, dynamic reconstruction is under-constrained while 3D detection lacks geometric grounding. To address this gap, we propose USR-Drive, a unified conditional generative framework that, given only posed multi-view driving videos, jointly recovers dense dynamic geometry and instance-level object layouts within a shared scene representation. Specifically, USR-Drive represents dense Gaussian primitives and sparse 3D bounding boxes as two aligned latent token streams and jointly denoises them with a unified multi-modal difusion Transformer. Unlike prior paradigms that use boxes as external conditions or predict them with detached modules, USR-Drive treats them as mutually constrained state variables with a Unified Positional Encoding (UPE) that aligns heterogeneous tokens within a shared metric spatiotemporal coordinate. Via such unified representation and generative framework, the two modalities reinforce each other: geometry supplies dense metric evidence for box prediction, while

boxes provide instance-level structural priors that help preserve spatial consistency and reduce ambiguity in sequential 3D geometric representation. Our approach successfully delivers state-of-the-art results for both dynamic reconstruction and 3D detection on the nuScenes and VKitti datasets.

## 1 Introduction

Understanding dynamic 3D scenes from driving video fundamentally requires two tightly coupled capabilities: reconstructing geometrically consistent sequential 3D scene states (Yang et al. 2024; Luiten et al. 2024; Wang et al. 2025a; Yang et al. 2025; He et al. 2026; Wang et al. 2025b; Zhang et al. 2026), and recovering instance-level 3D detection and tracking (Li et al. 2024; Hu et al. 2023b; Zhang et al. 2022). While both areas have advanced rapidly in recent years, they have largely evolved along separate research tracks.

Traditional pipelines treat scene reconstruction and instance-level perception as separate tasks, e.g., for ofline rendering and online perception, respectively. However, advances in visual foundation models (Kirillov et al. 2023; Oquab et al. 2023; Wang et al. 2025a, 2024a; Leroy, Cabon, and Revaud 2024) and generative world models (Hu et al. 2023a; Wang et al. 2024b; Li et al. 2025b; Zuo et al. 2025; Bogdoll et al. 2025) increasingly call for shared representations that support both scene generation and understanding. Despite this trend, driving-scene methods still lack a unified representation that jointly models and mutually constrains dense geometry and object-centric structure.

This gap is particularly limiting for feed-forward reconstruction from driving videos, which is under-constrained by occlusions, large motions, and sparse viewpoints. Without object-level structure, continuous geometry may produce temporally inconsistent geometry and smearing artifacts in dynamic regions (Fig. 1); conversely, standalone 3D perception lacks explicit grounding in dense reconstructed geometry. Existing unified approaches still model the two asymmetrically: Gaussian-based methods often lack instance-level semantics (Zhu et al. 2026; Mao et al. 2025), while occupancybased methods use coarse voxel grids that limit rendering fidelity (Bogdoll et al. 2025; Li et al. 2025a). We therefore represent dense geometry and object layouts as spatially aligned, jointly generated components of a dynamic scene state: geometry provides metric evidence for object localization, while object layouts regularize ambiguous dynamic and occluded regions.

In this paper, we propose USR-Drive, a unified representation for dynamic driving scene reconstruction and instancelevel perception from posed multi-view videos. Our key insight is to learn a shared multimodal latent space as a unified spatiotemporal scene state, in which dense, renderable 3D Gaussians and ego-frame 3D bounding boxes are jointly modeled as co-evolving latent variables. USR-Drive first encodes dense spatiotemporal geometry using a pretrained geometric prior (Lin et al. 2026) and represents 3D bounding boxes as compact object-centric tokens, then maps both streams into a shared metric spatiotemporal coordinate system through our Unified Positional Encoding (UPE), establishing explicit correspondence between dense Gaussian tokens and sparse box tokens. Conditioned on the posed multiview input video, a unified Multi-Modal Difusion Transformer (MMDiT) jointly recovers both token streams from Gaussian noise through conditional rectified-flow denoising. This joint process allows dense geometry to provide local metric evidence for object localization, while object-centric tokens supply structural priors that regularize the temporal evolution of reconstructed Gaussians. At inference time, both dynamic 3D Gaussians and 3D boxes are produced directly by the unified denoising process without ground-truth boxes, object tracks, or layout annotations.

Our contributions are summarized as follows:

• We propose a unified representation for 3D driving scenes that models object-centric structure and the corresponding dense geometry of the scene as co-evolving state variables, moving beyond detached perception and unconstrained reconstruction paradigms.

• We design a dual-branch autoencoding framework to represent heterogeneous scene modalities, coupled with UPE that spatially aligns dense rendering primitives (3D Gaussians) and sparse object-centric structures (3D BBox) within a shared metric spatiotemporal space.

• We develop a generative framework with MMDiT that jointly denoises geometry and layout tokens, achieving state-of-the-art performance in both high-fidelity dynamic scene synthesis and temporally coherent 3D detection on various datasets.

## 2 Related Work

## 2.1 Feed-Forward Scene Reconstruction

Traditional dynamic scene reconstruction relies on per-scene optimization such as Neural Radiance Fields (NeRFs) or 3D Gaussian Splatting (3DGS), which is computationally intensive and ill-suited for real-time driving applications. Recent work has shifted toward feed-forward architectures that directly infer dense 3D attributes from sparse observations. For static scenes, VGGT (Wang et al. 2025a) and Depth Anything 3 (Lin et al. 2026) leverage strong visual priors to infer multi-view consistent geometry. For dynamic environments, 4RC (Luo et al. 2026),D4RT (Zhang et al. 2025) and STORM (Yang et al. 2025) capture spatiotemporal evolution in a feed-forward manner, while DGGT (Chen et al. 2025) achieves pose-free 4D reconstruction with a dynamic head that disentangles moving agents. However, these methods model appearance and geometry purely as low-level rendering primitives. Without explicit instance-level structure such as 3D bounding boxes, the reconstructed dynamic geometry remains weakly constrained and prone to temporal ambiguity and smearing in highly dynamic or occluded trafic scenes.

## 2.2 Generative World Models and 3D Perception

Difusion models have driven a paradigm shift in 3D perception and scene simulation. Early generative perception methods model 3D bounding boxes directly as denoising trajectories (e.g., 3DifFusionDet (Xiang, Dräger, and Zhang 2026) and MonoDif (Ranasinghe, Hegde, and Patel 2024)), enabling detection without external geometric priors. More recently, world models built on large-scale Difusion Transformers (DiT) (Wan et al. 2025) have made layout controllability a central theme: MagicDrive-V2 (Gao et al. 2025) and DrivingDifusion (Li, Zhang, and Ye 2024) introduce spatio-temporal conditional encodings to guide multi-view video generation with road semantics and 3D boxes. However, these methods use 3D boxes strictly as extrinsic conditioning signals — the layout serves as an immutable input prior rather than an inferable state variable, and thus cannot be refined from the synthesized geometry.

Pushing toward unified state generation, UniScene (Li et al. 2025a) jointly generates 3D semantic occupancy and video, DriveLaW (Xia et al. 2026) fuses video generation with trajectory planning in a shared latent space, and World-Splat (Zhu et al. 2026) predicts feed-forward 4D Gaussians with a 4D-aware latent difusion model. MagicDrive3D (Gao et al. 2024) couples layout-conditioned video generation with deformable Gaussian reconstruction, while ReconViaGen (Chang et al. 2025) and Gen3R (Huang et al. 2026) combine feed-forward reconstruction priors with difusionbased generative priors to improve geometric completeness. Crucially, none of these methods treat sparse 3D bounding boxes and dense continuous geometry as co-predicted, mutually regularized variables within the same denoising process.

In contrast, our method co-denoises geometry and layout tokens within a shared world state, establishing bidirectional regularization: dense geometry resolves scale and depth ambiguity for object localization, while object-level structure improves instance-level coherence and rendering rigidity.

## 3 Method

## 3.1 Overview

USR-Drive aims to unify dense geometric recovery and instance-level 3D object detection within a joint multi-modal framework. As illustrated in Fig. 2, the overall framework consists of two tightly coupled stages. In the first stage, we learn a dual-branch autoencoder system that compresses two heterogeneous yet complementary modalities into compact latent representations: a geometric branch that captures dense scene structure and temporal dynamics from the raw video, and a layout branch that encodes object-level spatial information from bounding box sequences. These two branches yield a disentangled yet aligned latent space prepared for downstream co-generation.

In the second stage, we build a unified multi-modal $d i f f u -$ sion architecture over the learned latent spaces. Instead of performing difusion in a low-dimensional, patch-aligned latent space, our model operates on the joint latent tokens from both the geometry and layout branches, which preserve richer scene information. The difusion model is trained to perform joint denoising for scene-level geometric reconstruction and object-level layout grounding.

## 3.2 Dual-branch Scene Autoencoder

Dynamic driving scenes involve heterogeneous representations ranging from dense geometry to sparse object-centric structures. To model these complementary modalities, we adopt a dual-branch autoencoder framework that separately encodes dense Gaussian geometry and 3D bounding box layouts before aligning them in a unified metric spatiotemporal latent space for joint difusion modeling.

Dense Geometry Autoencoder Instead of learning a latent space directly from raw 3D Gaussian parameters, our geometry encoder ${ \mathcal E } _ { \mathrm { g e o } }$ leverages a frozen, pretrained foundational geometry encoder, Depth Anything V3 Base (DA3- Base) (Lin et al. 2026), to extract high-fidelity geometric features as the continuous dense geometry representation $z _ { \mathrm { g e o } } ^ { c } .$ To ensure the resulting feature space forms a wellregularized and smooth manifold suitable for latent diffusion, we formulate this bottleneck as a Representation Autoencoder (RAE) (Zheng et al. 2026), which projects $z _ { \mathrm { g e o } } ^ { c }$ into the MMDiT hidden space through a lightweight 3D convolutional encoder, yielding the geometry latent $z _ { \mathrm { g e o } } \in \mathbb { R } ^ { B \times T \times N _ { \mathrm { g e o } } \times D _ { \mathrm { g } } }$ . For decoding, we symmetrically employ the Dense Prediction Transformer (DPT) decoder (Ranftl, Bochkovskiy, and Koltun 2021) from DA3 to reconstruct the dense scene geometry from the latents. The geometry autoencoder is trained end-to-end with a composite objective that supervises both photometric rendering fidelity $( \mathcal { L } _ { \mathrm { r g b } } )$ and geometric structure via sparse depth regularization from ground-truth LiDAR maps $( { \mathcal { L } } _ { \mathrm { d e p t h } } )$ . Detailed layer configurations and loss formulations are provided in the supplementary materials.

Sparse Bounding Box Autoencoder Parallel to the dense geometry branch, discrete object-level constraints are modeled via a bounding box path. Our BBox Autoencoder $\mathcal { E } _ { \mathrm { b o x } }$ encodes slot-aligned 3D bounding box sequences into latents $z _ { \mathrm { b o x } } \in \mathbb { R } ^ { B \times T \times \breve { N } _ { \mathrm { m a x } } \times D _ { \mathrm { b } } }$ , where $\breve { N } _ { \mathrm { m a x } }$ is the maximum number of boxes. Specifically, each box is converted into its eight 3D corners, embedded with a Fourier Positional Embedding (FPE) and a learnable class embedding, and processed by spatiotemporal transformer blocks that alternately perform frame-local spatial attention across object slots and temporal attention along each slot’s trajectory. A symmetric decoder then reconstructs the explicit 3D bounding box parameters from $z _ { \mathrm { b o x } }$

The BBox Autoencoder is trained independently following established 3D layout representation paradigms (Gao et al. 2025), where invalid slots are handled by a binary mask. The reconstruction objective combines an $\ell _ { 1 }$ corner regression loss, a sine-cosine yaw regression loss in the SO(2) embedding space to circumvent periodicity artifacts, a crossentropy category loss, a slot-existence loss, and a KL regularizer over the latent distribution. Please refer to the supplementary materials for detailed formulations and training strategies.

## 3.3 Unified Difusion Architecture

Unified Positional Encoding In the second stage, we perform denoising process over two heterogeneous latent streams: dense Gaussian Splatting (GS) geometry tokens and sparse 3D bounding-box layout tokens. Since the two modalities difer in token density, semantics, and ordering, independent index-based positional embeddings cannot provide meaningful cross-modal spatial correspondence. As shown in Fig. 3, we introduce a Unified Positional Encoding (UPE), which represents both modalities in a shared metric spatiotemporal scene coordinate system.

Given a token with metric 3D anchor $ { \mathbf { p } } \in \mathbb { R } ^ { 3 }$ at frame t, UPE encodes its spatial location and temporal index jointly through Fourier features:

$$
\begin{array} { c } { \Phi _ { \mathrm { u p e } } ( { \bf p } , t ) = \mathrm { M L P } _ { \mathrm { u p e } } \left( \left[ \gamma _ { \mathrm { 3 D } } ( { \bf p } ) , \gamma _ { \mathrm { t i m e } } ( \tau _ { t } ) \right] \right) , } \\ { \displaystyle \tau _ { t } = \frac { t } { T - 1 } . } \end{array}\tag{1}
$$

Here, $\tau _ { t } \in [ 0 , 1 ]$ denotes the normalized frame index, independent of the difusion timestep. The spatial and temporal Fourier features are defined as:

$$
\begin{array} { r l } & { \gamma _ { \mathrm { 3 D } } ( \mathbf { p } ) = \left[ \sin ( 2 ^ { k } \mathbf { p } ) , \cos ( 2 ^ { k } \mathbf { p } ) \right] _ { k = 0 } ^ { K _ { s } - 1 } , } \\ & { \gamma _ { \mathrm { t i m e } } ( \tau _ { t } ) = \left[ \sin ( 2 ^ { k } \pi \tau _ { t } ) , \cos ( 2 ^ { k } \pi \tau _ { t } ) \right] _ { k = 0 } ^ { K _ { t } - 1 } , } \end{array}\tag{2}
$$

where $K _ { s }$ and $K _ { t }$ denote the number of Fourier frequency bands used for spatial coordinates and temporal indices, respectively. This shared encoding maps both geometry and layout tokens into a unified geometry-centric positional space.

![](images/9d8fed9d7e2c60b426621f7847773725e26fd6203de0c5f2989fbd19d5a22581.jpg)  
Figure 2: Overview of USR-Drive. Stage I (Top): A dual-branch autoencoder compresses dense geometry and slot-aligned 3D boxes into latents $z _ { \mathrm { g e o } }$ and $z _ { \mathrm { b o x } } .$ respectively. Stage II (Bottom): A unified MMDiT jointly denoises both latent streams, where the Unified Positional Encoding (UPE) anchors heterogeneous tokens in a common 3D metric space, enabling bidirectional regularization between geometry and layout through shared self-attention. Inference (Right): Starting from pure noise, the MMDiT iteratively denoises the joint latent streams conditioned on reference video latents, and the denoised latents are decoded by the DPT geometry head and the box decoder into explicit dense scene geometry and 3D object layouts.

Specifically, during the training phase, we construct metric anchors for GS tokens using only the reference video, its calibrated camera poses, and a frozen geometry prior. Before denoising, the geometry prior is evaluated once to obtain a coarse set of Gaussian primitives, including their preliminary centers $\widetilde { \mu } _ { t , q } ^ { \mathrm { p r i o r } }$ and opacities $\widetilde { \alpha } _ { t , q } ^ { \mathrm { p r i o n } }$ <sup>r</sup>. Using the known camera poses, we transform each preliminary center into the ego coordinate system of the first frame and denote the transformed center by $\overline { { \mu } } _ { t , q } ^ { \mathrm { p r i o r } }$ . For each GS token u, we then define its anchor as the opacity-weighted centroid of the prior primitives associated with its corresponding image-space patch $\mathcal { P } ( u )$ . Importantly, this anchor provides only a coarse metric reference for the token. It should not be interpreted as either the clean difusion target or the Gaussian center predicted by the difusion model.

For box tokens, instead of using ground-truth box centers during training, we assign each slot to a deterministic BEV grid anchor:

$$
{ \bf p } _ { t , u } ^ { \mathrm { G S } } = \frac { \sum _ { \boldsymbol { q } \in \mathcal { P } ( u ) } \widetilde { \alpha } _ { t , \boldsymbol { q } } ^ { \mathrm { p r i o r } } \overline { { \mu } } _ { t , \boldsymbol { q } } ^ { \mathrm { p r i o r } } } { \sum _ { \boldsymbol { q } \in \mathcal { P } ( u ) } \widetilde { \alpha } _ { t , \boldsymbol { q } } ^ { \mathrm { p r i o r } } + \epsilon } , \qquad { \bf e } _ { t , u } ^ { \mathrm { G S } } = \boldsymbol { \Phi } _ { \mathrm { u p e } } \left( { \bf p } _ { t , u } ^ { \mathrm { G S } } , t \right) ,\tag{3}
$$

$$
{ \bf p } _ { t , n } ^ { \mathrm { B o x } } = { \bf p } _ { n } ^ { \mathrm { B E V } } , \qquad { \bf e } _ { t , n } ^ { \mathrm { B o x } } = \Phi _ { \mathrm { u p e } } \left( { \bf p } _ { t , n } ^ { \mathrm { B o x } } , t \right) .\tag{4}
$$

where $\overline { { \mu } } _ { t , q } ^ { \mathrm { p r i o r } }$ denotes the preliminary center of Gaussian q after transformation into the $t = 0$ ego coordinate system, and $\widetilde \alpha _ { t , q } ^ { \mathrm { p r i o r } }$ denotes its opacity. $\mathcal { P } ( u )$ denotes the set of prior Gaussian primitives associated with geometry token u. The geometry anchors and the fixed BEV anchors are therefore expressed in the same metric reference coordinate system. $\mathbf { p } _ { n } ^ { \mathrm { { B E V } } }$ denotes the metric center ofthe n-th cell in a fixed BEV grid and is shared across all frames; its temporal instances are distinguished by the frame index t in $\Phi _ { \mathrm { u p e } } ( \mathbf { p } , t )$

Neither type of anchor depends on the clean difusion targets. The GS anchors are computed once during preprocessing, whereas the BEV anchors are predefined; both remain fixed throughout the denoising process.

Furthermore, to improve detection accuracy and motion perception, we augment each box latent with a confidence scalar c, an anchor-relative position bias δ, a velocity vector $v ,$ and an attribute vector a for dynamic judgment. We concatenate $[ z _ { \mathrm { u n i } } , c , \delta , v , a ]$ as the augmented unified latent and send it to the DiT, which predicts the corresponding flows jointly with the geometry and box latents. During decoding, the predicted $\hat { \delta }$ is mapped back to metric space to refine the final box center, while vˆ and aˆ serve as the box velocity and dynamic status predictions. Please refer to our supplementary materials for detailed designs and ablations.

Multi-Modal Difusion Transformer Inspired by recent geometry RAE based difusion models (Jang et al. 2026), we design a unified multi-modal difusion model to directly denoise the joint noisy latent space. Specifically, we adopt the structure of DiT blocks from Wan 2.1-1.3B (Wan et al. 2025). The spatially grounded geometry and bounding box tokens are concatenated into a singular 1D sequence and processed through a unified Multi-Modal Difusion Transformer (MMDiT) attention trunk. To guide the denoising process, the network is conditioned on reference video latents extracted from a visual VAE, which are linearly projected and injected as the cross-attention context. The denoised latents are subsequently processed by their respective autoencoder decode heads — the DA3-based DPT head and the symmetric bounding box decoder — to yield the final explicit 3D geometry and discrete object layouts.

![](images/ab4e173980a0dcbb661884d90864db0e02e59724a4ed89dfb25a6532d7601c4c.jpg)  
Figure 3: Unified Positional Encoding. Dense GS tokens $\mathbf { p } _ { t , u } ^ { \breve { \mathrm { G S } } }$ and sparse box tokens $\mathbf { p } _ { t , n } ^ { \mathrm { B o x } }$ are assigned metric 3D anchors and a normalized frame index, which are encoded by shared spatial and temporal Fourier features followed by an MLP. The resulting UPE embeddings are combined with projected geometry and box latents together with learnable modality-aware tokens, aligning heterogeneous modalities into a unified latent space for joint DiT modeling.

Training Strategy We formulate the generative training objective within the continuous-time rectified flow framework (Lipman et al. 2022), which predicts the target flow velocity pointing from the data sample toward the noise sample.

Given the predicted velocities from the geometry and layout heads, denoted by $v _ { \mathrm { g e o } }$ and $v _ { \mathrm { b o x } } .$ , respectively, we optimize the two branches with separate objectives to account for their diferent structures. For the geometry branch, we apply a standard Mean Squared Error (MSE) loss $\mathcal { L } _ { \mathrm { g e o } } =$ $\mathrm { M S E } ( \mathbf { v } _ { \mathrm { g e o } } , \tilde { \mathbf { v } } _ { \mathrm { g e o } } )$ between the predicted and target velocities.

For the layout branch, supervision is applied only to valid bounding box slots. Let $\boldsymbol { M } ^ { \star } \in \{ 0 , 1 \} ^ { N _ { \operatorname* { m a x } } }$ denote the binary occupancy mask for the box tokens, where $M _ { i } = 1$ indicates a valid object slot. The layout loss is defined as:

$$
\mathcal { L } _ { \mathrm { b o x } } = \frac { \sum _ { i = 1 } ^ { N _ { \mathrm { m a x } } } M _ { i } \left. \left. \mathbf { v } _ { \mathrm { b o x } } ^ { i } - \tilde { \mathbf { v } } _ { \mathrm { b o x } } ^ { i } \right. \right. _ { 2 } ^ { 2 } } { \sum _ { i = 1 } ^ { N _ { \mathrm { m a x } } } M _ { i } } ,\tag{5}
$$

where $\tilde { \mathbf { v } } _ { \mathrm { b o x } } ^ { i }$ is the target velocity for the i-th box slot. This masking ensures that the layout objective is evaluated only on populated entities, which is particularly important for stable training under slot masking augmentation.

In addition to the latent layout velocity, we supervise auxiliary layout targets on the fixed-anchor slots, including object confidence, anchor-relative center ofset, ground-plane velocity, and object attribute. We group these terms into an auxiliary layout loss:

$$
{ \mathcal { L } } _ { \mathrm { a u x } } = \lambda _ { c } { \mathcal { L } } _ { c } + \lambda _ { \delta } { \mathcal { L } } _ { \delta } + \lambda _ { v } { \mathcal { L } } _ { v } + \lambda _ { a } { \mathcal { L } } _ { a } .\tag{6}
$$

The overall training objective is:

$$
\mathcal { L } _ { \mathrm { t o t a l } } = \mathcal { L } _ { \mathrm { g e o } } + \lambda _ { \mathrm { b o x } } \mathcal { L } _ { \mathrm { b o x } } + \mathcal { L } _ { \mathrm { a u x } } .\tag{7}
$$

Please refer to our supplementary materials for detailed loss function.

## 4 Experiments

## 4.1 Experimental Setup

Dataset We conduct experiments on the nuScenes (Caesar et al. 2020) and VKitti (Gaidon et al. 2016) datasets, which provide synchronized camera streams and 3D annotations for autonomous driving. We use the nuScenes oficial split for training and evaluation. We take all 6 surround-view cameras as input with temporal clips of $T { = } 8$ frames, and all images are resized to $1 1 2 \times 1 6 8 ;$ camera poses are used only for metric grounding of the UPE anchors. As for VKitti dataset, we sampled 400 cases for zero-shot evaluation. Please refer to supplementary materials for detailed dataset preprocessing.

Training and Inference All models are trained on the nuScenes dataset solely. During inference, the only scenecontent input is the posed multi-view video: no ground-truth boxes, object tracks, or layout annotations are used. We encode the video with the frozen Wan-VAE as the conditioning signal and initialize all latents $( z _ { \mathrm { g e o } } , z _ { \mathrm { b o x } } , c , \delta , v , a )$ from Gaussian noise, denoising them with 50 rectified-flow steps under classifier-free guidance between the video-conditioned and zero-conditioned predictions. The sparse layout tokens are tied to a deterministic BEV anchor grid rather than ground-truth box centers, so boxes are pure outputs of the unified denoising process. The denoised geometry and layout latents are then decoded by the DPT decoder and the frozen BBox-AE into 3D Gaussians and 3D bounding boxes, respectively; longer sequences are handled with overlapping 8-frame sliding windows. Detailed decoding and postprocessing procedures are provided in the supplementary materials.

Baselines We evaluate our method on two complementary tasks: (1) dynamic scene reconstruction, and (2) 3D object detection. This dual evaluation directly reflects our goal of unifying geometry and perception.

We compare against two categories of methods:

(1) Reconstruction methods. We compare our method with the recent geometry foundation models including VGGT (Wang et al. 2025a), Depth Anything 3 (Lin et al. 2026) and Pi-3 (Wang et al. 2026). In addition, we also compare with the state-of-the-art feed-forward Gaussian reconstruction methods including AnySplat (Jiang et al. 2025), STORM (Yang et al. 2025) and DGGT (Chen et al. 2025).

(2) 3D detection methods. We report the 3D detection results of nuScenes on their val set, including the best performed methods reported in the oficial benchmark (e.g., HoP (Zong et al. 2023), StreamPETR (Wang et al. 2023), RayDN (Liu et al. 2024), etc.). As for VKitti dataset, we calculate mAP and mATE metrics with the ground truth 3D annotations.

The detailed implementations of the geometry and bounding box autoencoders and the unified difusion model are provided in the supplementary materials.

## 4.2 Experimental Results

Reconstruction Quality We first evaluate dynamic scene reconstruction quality against recent geometry foundation models and dynamic Gaussian reconstruction methods. As reported in Tab. 1, our method achieves the best performance across all image-level metrics and 3D depth metrics. In particular, the gain is most pronounced on foreground dynamic objects, where reconstruction is most under-constrained: USR-Drive improves object-level PSNR and SSIM by a clear margin over all baselines. Since dynamic regions benefit the least from appearance cues alone, this improvement directly evidences that the sparse box tokens serve as instance-leve structural priors that regularize dynamic geometry — a concrete source of the mutual benefit enabled by joint denoising. We attribute this improvement to the unified latent modeling of dense geometry and sparse object layout, which helps preserve scene structure while reducing local visual artifacts. Qualitative results on nuScenes further show that our method recovers better scene details, cleaner object boundaries, and more coherent dynamic content under challenging driving scenarios. Please refer to Fig. 4 and supplementary materials for visualization results.

![](images/c78066fd6ed9530d5b82a7386dab58eb69def98ee42d9a2cb3af4251441a8033.jpg)  
Figure 4: Visualization of reconstruction and detection results. Compared to baseline methods, our approach reveals better quality on both reconstruction task (top 2 rows) and detection task (bottom 2 rows).

3D Detection Quality We follow the oficial nuScenes detection benchmark protocol for the vision track. The results on the nuScenes benchmark reveal the high precision and recall of our method for 3D detection. As shown in Tab. 2, our method surpasses dedicated state-of-the-art detectors on both NDS and mAP. Visualization results in Fig.4 and the supplementary materials further demonstrate that our joint generation paradigm benefits from the information of the geometric branch, achieving more accurate and complete object-level detection. Moreover, the zero-shot detection results on the VKitti datasets in Tab. 3 also demonstrate the strong generalizability of our method, while baseline methods fail to generalize to the unseen synthetic domain.

## 4.3 Ablation Studies

Joint Denoising vs. Decoupled Pipelines To verify that our gains stem from the unified co-denoising design rather than the pretrained DA3 backbone alone, we compare against (i) the Stage-I geometry autoencoder without difusion refinement, (ii) a decoupled two-stage pipeline that first reconstructs geometry and then predicts boxes from it, and (iii) a cascaded DA3 + StreamPETR (Wang et al. 2023) pipeline for detection. As shown in Tab. 4, the decoupled variants degrade both tasks: errors in the two-stage pipeline compound across stages, and the box branch can no longer feed structural priors back into the geometry. In contrast, UPE places noisy geometry and box tokens in a shared metric space, where MMDiT self-attention allows them to exchange dense geometric evidence and object-level layout cues during denoising, yielding mutual correction.

Component and UPE Ablation We further ablate the key components of our framework, as reported in the bottom block of Tab. 4. Removing the layout branch, which reconstructs scenes using only finetuned DA3 geometry features(Geom. Only), degrades all reconstruction metrics, demonstrating that sparse layout tokens provide complementary structural cues that help resolve scene ambiguities during joint denoising. Conversely, relying exclusively on the discrete bounding box branch collapses 3D detection to a mere 0.012 mAP: without dense geometric features providing structural context and spatial grounding, predicting object layouts in isolation is highly ill-posed. Finally, removing the metric spatio-temporal positional encoding leaves heterogeneous tokens coordinated only by sequence order, causing clear degradation in both layout accuracy and reconstruction quality. These results confirm that the two modalities mutually reinforce each other and that UPE provides a crucial shared spatial reference for aligning them. Visualized ablation results are provided in the supplementary materials.

<table><tr><td>Method</td><td>PSNR↑ SSIM↑</td><td>LPIPS↓</td><td>D-RMSE↓</td></tr><tr><td colspan="4">Scene-level Reconstruction</td></tr><tr><td>VGGT DA3</td><td>21.75 0.617</td><td>0.183</td><td>5.07 12.16</td></tr><tr><td>Pi-3</td><td></td><td></td><td>6.93</td></tr><tr><td>AnySplat STORM</td><td>25.53 0.803</td><td>0.157</td><td>17.65</td></tr><tr><td></td><td>24.54 0.784</td><td>0.267</td><td>6.48</td></tr><tr><td>DGGT</td><td>26.63 0.813</td><td>0.122</td><td>5.08</td></tr><tr><td>USR-Drive</td><td>27.55 0.853</td><td>0.076</td><td>4.59</td></tr><tr><td colspan="4">Object-level Foreground Reconstruction</td></tr><tr><td>AnySplat</td><td>14.78 0.395</td><td>0.209</td><td>19.93</td></tr><tr><td>DA3</td><td>14.34 0.408</td><td>0.297</td><td>15.32</td></tr><tr><td>STORM</td><td>20.97 0.532</td><td>0.515</td><td>12.56</td></tr><tr><td>DGGT</td><td>19.73 0.791</td><td>0.150</td><td>19.78</td></tr><tr><td>USR-Drive</td><td>24.45</td><td>0.083</td><td>9.98</td></tr><tr><td></td><td>0.833</td><td></td><td></td></tr></table>

Table 1: Scene-level and object-level reconstruction results on the nuScenes dataset. The object-level evaluation is conducted on foreground dynamic objects. We highlight the first , second , and third best results within each reconstruction block. VGGT and Pi-3 only produce 3D point clouds, so rendered-image quality metrics are omitted.

<table><tr><td>Method</td><td>NDS↑ mAP↑ mATE↓ mASE↓ mAOE↓ mAVE↓ mAAE↓</td></tr><tr><td>BEVDepth 0.475 0.3510.629</td><td>0.267 0.479 0.428 0.198</td></tr><tr><td>BEVFormer 0.517 0.416 0.673</td><td>0.274 0.372 0.394 0.198</td></tr><tr><td>PETRv2 0.456 0.349</td><td>0.700 0.275 0.580 0.437 0.187</td></tr><tr><td>HoP 0.558 0.454</td><td>0.565 0.265 0.327 0.337 0.194</td></tr><tr><td>StreamPETR 0.550 0.4500.613</td><td>0.267 0.413 0.265 0.198</td></tr><tr><td>RayDN 0.563 0.469</td><td>0.579 0.264 0.433 0.256 0.187</td></tr><tr><td>RoPETR 0.614 0.529</td><td>0.537 0.255 0.289 0.229 0.195</td></tr><tr><td>USR-Drive 0.625 0.552 0.525</td><td>0.211 0.303 0.251 0.177</td></tr></table>

Table 2: Results of 3D detection tasks on the nuScenes val set. We inherit the reported results of baseline methods from their oficial publications (Zong et al. 2023; Liu et al. 2024; Ji et al. 2025).

More Ablation Studies We also conduct ablation studies on the auxiliary layout tokens, training parameters, and MMDiT hidden dimension. Please refer to our supplementary materials for detailed discussion.

## 5 Conclusion

In this paper, we propose USR-Drive, a unified generative framework for autonomous driving that jointly models dynamic 3D geometry and instance-level 3D bounding boxes as a shared physical world state. By introducing a multimodal latent space, a Unified Positional Encoding (UPE), and a joint Difusion Transformer (DiT) for simultaneous denoising, our method enables dense geometry and object structure to mutually constrain each other. This unified design improves both the fidelity and appearance consistency of dynamic reconstruction and the geometric grounding of 3D detection. Experiments on nuScenes and VKitti demonstrate state-of-the-art performance on both tasks, highlighting the promise of unified scene representations for future autonomous driving world models.

<table><tr><td rowspan="2">Method</td><td>Reconstruction</td><td colspan="2">Detection</td></tr><tr><td>PSNR↑ SSIM↑</td><td>mAP↑</td><td>mATE↓</td></tr><tr><td colspan="4">Detection-Only Baselines</td></tr><tr><td>RayDN</td><td></td><td>0.015</td><td>1.024</td></tr><tr><td>HoP</td><td></td><td>0.022</td><td>1.258</td></tr><tr><td>StreamPETR</td><td>一</td><td>0.008</td><td>1.340</td></tr><tr><td colspan="4">Reconstruction-Only Baselines</td></tr><tr><td>DA3</td><td>17.12 0.465</td><td>一</td><td></td></tr><tr><td>DGGT</td><td>25.38 0.712</td><td>一</td><td>一</td></tr><tr><td>AnySplat</td><td>23.80 0.625</td><td></td><td></td></tr><tr><td>USR-Drive</td><td>26.45 0.743</td><td>0.518</td><td>0.812</td></tr></table>

Table 3: Zero-shot reconstruction and 3D detection results on VKitti. Our unified model performs both tasks, while baselines specialize in only one.

<table><tr><td colspan="3">Method PSNR↑ SSIM↑ LPIPS↓ | mAP↑ mATE↓</td></tr><tr><td colspan="3">Joint Denoising and Pipeline Comparison</td></tr><tr><td>Stage-I Geo-AE Decoupled two-stage DA3 + StreamPETR</td><td>23.430.709 0.177 21.470.655 0.179</td><td>0.4730.701 0.4910.648</td></tr><tr><td colspan="3">Component and UPE Ablation</td></tr><tr><td>Geom. Only Box Only</td><td>26.370.833 0.127</td><td>0.012 0.901</td></tr><tr><td>Ours w/o UPE</td><td>25.32 0.799 0.134</td><td>0.214 0.602</td></tr><tr><td>Ours</td><td>27.550.853 0.076</td><td>|0.552 0.525</td></tr></table>

Table 4: Ablation results on nuScenes val set. Top: joint denoising vs. decoupled pipelines. Bottom: component and UPE ablations. Our full model performs best on both tasks.

Limitations and Future Work Our patch- and framealigned geometry representation captures local consistency but lacks a compact global scene state and explicit long-term identities, limiting dynamic modeling and direct support for 4D tracking. Future work will explore global representations that unify reconstruction, detection, and tracking. Moreover, iterative denoising introduces non-trivial latency, restricting the current model to ofline estimation. Few-step distillation or causal autoregressive generation may enable real-time deployment and long-term scene representation.

## References

Bogdoll, D.; Yang, Y.; Joseph, T.; Yazgan, M.; and Zollner, J. M. 2025. MUVO: A Multimodal Generative World Model for Autonomous Driving with Geometric Representations. In IEEE Intelligent Vehicles Symposium (IV), 2243–2250.

Caesar, H.; Bankiti, V.; Lang, A. H.; Vora, S.; Liong, V. E.; Xu, Q.; Krishnan, A.; Pan, Y.; Baldan, G.; and Beijbom, O. 2020. nuScenes: A Multimodal Dataset for Autonomous Driving. In IEEE Conf. Comput. Vis. Pattern Recog., 11621– 11631.

Chang, J.; Ye, C.; Wu, Y.; Chen, Y.; Zhang, Y.; Luo, Z.; Li, C.; Zhi, Y.; and Han, X. 2025. ReconViaGen: Towards Accurate Multi-view 3D Object Reconstruction via Generation. arXiv preprint arXiv:2510.23306.

Chen, X.; Xiong, Z.; Chen, Y.; Li, G.; Wang, N.; Luo, H.; Chen, L.; Sun, H.; Wang, B.; Chen, G.; et al. 2025. DGGT: Feedforward 4D Reconstruction of Dynamic Driving Scenes using Unposed Images. arXiv preprint arXiv:2512.03004.

Gaidon, A.; Wang, Q.; Cabon, Y.; and Vig, E. 2016. Virtual worlds as proxy for multi-object tracking analysis. In Proceedings of the IEEE conference on computer vision and pattern recognition, 4340–4349.

Gao, R.; Chen, K.; Li, Z.; Hong, L.; Li, Z.; and Xu, Q. 2024. MagicDrive3D: Controllable 3D Generation for Any-View Rendering in Street Scenes. In European Conference on Computer Vision (ECCV).

Gao, R.; Chen, K.; Xiao, B.; Hong, L.; Li, Z.; and Xu, Q. 2025. MagicDrive-V2: High-Resolution Long Video Generation for Autonomous Driving with Adaptive Control. In Int. Conf. Comput. Vis., 28135–28144.

He, Z.; Li, J.; Li, G.; Chen, X.; Tang, J.; Zhang, S.; Jin, Z.; Cai, F.; Li, B.; Pu, J.; et al. 2026. DynamicVGGT: Learning Dynamic Point Maps for 4D Scene Reconstruction in Autonomous Driving. In IEEE Conf. Comput. Vis. Pattern Recog.

Hu, A.; Russell, L.; Yeo, H.; Murez, Z.; Fedoseev, G.; Kendall, A.; Shotton, J.; and Corrado, G. 2023a. GAIA-1: A Generative World Model for Autonomous Driving. arXiv preprint arXiv:2309.17080.

Hu, Y.; Yang, J.; Chen, L.; Li, K.; Sima, C.; Zhu, X.; Chai, S.; Du, S.; Lin, T.; Wang, W.; et al. 2023b. Planning-Oriented Autonomous Driving. In IEEE Conf. Comput. Vis. Pattern Recog., 17853–17862.

Huang, J.; Yang, Y.; Yang, B.; Ma, L.; Ma, Y.; and Liao, Y. 2026. Gen3R: 3D Scene Generation Meets Feed-Forward Reconstruction. In Proceedings ofthe IEEE/CVFConference on Computer Vision and Pattern Recognition (CVPR).

Jang, W.; Jeon, S.; Han, J.; Choi, J.; Kwon, M.; Kim, S.; Xie, S.; and Liu, S. 2026. Repurposing Geometric Foundation Models for Multi-view Difusion. arXiv preprint arXiv:2603.22275.

Ji, H.; Ni, T.; Huang, X.; Shi, Z.; Luo, T.; Zhan, X.; and Chen, J. 2025. RoPETR: Improving Temporal Camera-Only 3D Detection by Integrating Enhanced Rotary Position Embedding. arXiv preprint arXiv:2504.12643.

Jiang, L.; Mao, Y.; Xu, L.; Lu, T.; Ren, K.; Jin, Y.; Xu, X.; Yu, M.; Pang, J.; Zhao, F.; et al. 2025. AnySplat: Feedforward 3D Gaussian Splatting from Unconstrained Views. ACM Trans. Graph., 44(6): 1–16.

Kirillov, A.; Mintun, E.; Ravi, N.; Mao, H.; Rolland, C.; Gustafson, L.; Xiao, T.; Whitehead, S.; Berg, A. C.; Lo, W.- Y.; et al. 2023. Segment Anything. In Int. Conf. Comput. Vis., 4015–4026.

Leroy, V.; Cabon, Y.; and Revaud, J. 2024. Grounding Image Matching in 3D with MASt3R. In Eur. Conf. Comput. Vis., 71–91.

Li, B.; Guo, J.; Liu, H.; Zou, Y.; Ding, Y.; Chen, X.; Zhu, H.; Tan, F.; Zhang, C.; Wang, T.; et al. 2025a. UniScene: Unified Occupancy-centric Driving Scene Generation. In IEEE Conf. Comput. Vis. Pattern Recog., 11971–11981.

Li, X.; Zhang, Y.; and Ye, X. 2024. DrivingDifusion: Layout-Guided Multi-view Driving Scene Video Generation with Latent Difusion Model. In Eur. Conf. Comput. Vis., 469–485.

Li, Y.; Wang, Y.; Liu, Y.; He, J.; Fan, L.; and Zhang, Z. 2025b. End-to-End Driving with Online Trajectory Evaluation via BEV World Model. In Int. Conf. Comput. Vis., 27137–27146.

Li, Z.; Wang, W.; Li, H.; Xie, E.; Sima, C.; Lu, T.; Yu, Q.; and Dai, J. 2024. BEVFormer: Learning Bird’s-Eye-View Representation from Multi-Camera Images via Spatiotemporal Transformers. IEEE Trans. Pattern Anal. Mach. Intell., 47(3): 2020–2036.

Lin, H.; Chen, S.; Liew, J.; Chen, D. Y.; Li, Z.; Shi, G.; Feng, J.; and Kang, B. 2026. Depth Anything 3: Recovering the Visual Space from Any Views. In Int. Conf. Learn. Represent.

Lipman, Y.; Chen, R. T.; Ben-Hamu, H.; Nickel, M.; and Le, M. 2022. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747.

Liu, F.; Huang, T.; Zhang, Q.; Yao, H.; Zhang, C.; Wan, F.; Ye, Q.; and Zhou, Y. 2024. Ray Denoising: Depth-aware Hard Negative Sampling for Multi-view 3D Object Detection. In Eur. Conf. Comput. Vis., 200–217.

Luiten, J.; Kopanas, G.; Leibe, B.; and Ramanan, D. 2024. Dynamic 3D Gaussians: Tracking by Persistent Dynamic View Synthesis. In Int. Conf. 3D Vis., 800–809.

Luo, Y.; Zhou, S.; Lan, Y.; Pan, X.; and Loy, C. C. 2026. 4RC: 4D Reconstruction via Conditional Querying Anytime and Anywhere. arXiv preprint arXiv:2602.10094.

Mao, J.; Li, B.; Ivanovic, B.; Chen, Y.; Wang, Y.; You, Y.; Xiao, C.; Xu, D.; Pavone, M.; and Wang, Y. 2025. Dream-Drive: Generative 4D Scene Modeling from Street View Images. In IEEE International Conference on Robotics and Automation (ICRA), 367–374.

Oquab, M.; Darcet, T.; Moutakanni, T.; Vo, H.; Szafraniec, M.; Khalidov, V.; Fernandez, P.; Haziza, D.; Massa, F.; El-Nouby, A.; et al. 2023. DINOv2: Learning Robust Visual Features without Supervision. arXiv preprint arXiv:2304.07193.

Ranasinghe, Y.; Hegde, D.; and Patel, V. M. 2024. MonoDif: Monocular 3D Object Detection and Pose Estimation

with Difusion Models. In IEEE Conf. Comput. Vis. Pattern Recog., 10659–10670.

Ranftl, R.; Bochkovskiy, A.; and Koltun, V. 2021. Vision transformers for dense prediction. In Int. Conf. Comput. Vis., 12179–12188.

Wan, T.; Wang, A.; Ai, B.; Wen, B.; Mao, C.; Xie, C.-W.; Chen, D.; et al. 2025. Wan: Open and Advanced Large-Scale Video Generative Models. arXiv preprint arXiv:2503.20314.

Wang, J.; Chen, M.; Karaev, N.; Vedaldi, A.; Rupprecht, C.; and Novotny, D. 2025a. VGGT: Visual Geometry Grounded Transformer. In IEEE Conf. Comput. Vis. Pattern Recog., 5294–5306.

Wang, Q.; Ye, V.; Gao, H.; Zeng, W.; Austin, J.; Li, Z.; and Kanazawa, A. 2025b. Shape of Motion: 4D Reconstruction from a Single Video. In Int. Conf. Comput. Vis., 9660–9672.

Wang, S.; Leroy, V.; Cabon, Y.; Chidlovskii, B.; and Revaud, J. 2024a. DUSt3R: Geometric 3D Vision Made Easy. In IEEE Conf. Comput. Vis. Pattern Recog., 20697–20709.

Wang, S.; Liu, Y.; Wang, T.; Li, Y.; and Zhang, X. 2023. Exploring Object-Centric Temporal Modeling for Eficient Multi-View 3D Object Detection. In Int. Conf. Comput. Vis., 3621–3631.

Wang, X.; Zhu, Z.; Huang, G.; Chen, X.; Zhu, J.; and Lu, J. 2024b. DriveDreamer: Towards Real-world-driven World Models for Autonomous Driving. In Eur. Conf. Comput. Vis., 55–72.

Wang, Y.; Zhou, J.; Zhu, H.; Chang, W.; Zhou, Y.; Li, Z.; Chen, J.; Pang, J.; Shen, C.; and He, T. 2026. π<sup>3</sup>: Permutation-Equivariant Visual Geometry Learning. In Int. Conf. Learn. Represent.

Xia, T.; Li, Y.; Zhou, L.; Yao, J.; Xiong, K.; Sun, H.; Wang, B.; Ma, K.; Chen, G.; Ye, H.; et al. 2026. DriveLaW: Unifying Planning and Video Generation in a Latent Driving World. In IEEE Conf. Comput. Vis. Pattern Recog.

Xiang, X.; Dräger, S.; and Zhang, J. 2026. 3DifFusion-Det: Difusion Model for 3D Object Detection with Robust LiDAR-Camera Fusion. In ICASSP, 10157–10161.

Yang, J.; Huang, J.; Chen, Y.; Wang, Y.; Li, B.; You, Y.; Sharma, A.; Igl, M.; Karkus, P.; Xu, D.; Ivanovic, B.; Wang, Y.; and Pavone, M. 2025. STORM: Spatio-Temporal Reconstruction Model for Large-scale Outdoor Scenes. In Int. Conf. Learn. Represent.

Yang, J.; Ivanovic, B.; Litany, O.; Weng, X.; Kim, S. W.; Li, B.; Che, T.; Xu, D.; Fidler, S.; Pavone, M.; and Wang, Y. 2024. EmerNeRF: Emergent Spatial-Temporal Scene Decomposition via Self-Supervision. In Int. Conf. Learn. Represent.

Zhang, C.; Moing, G. L.; Koppula, S.; Rocco, I.; Momeni, L.; Xie, J.; Sun, S.; Sukthankar, R.; Barral, J. K.; Hadsell, R.; et al. 2025. Eficiently reconstructing dynamic scenes one d4rt at a time. arXiv preprint arXiv:2512.08924.

Zhang, C.; Moing, G. L.; Koppula, S.; Rocco, I.; Momeni, L.; Xie, J.; Sun, S.; Sukthankar, R.; Barral, J. K.; Hadsell, R.; et al. 2026. Eficiently Reconstructing Dynamic Scenes One D4RT at a Time. In IEEE Conf. Comput. Vis. Pattern Recog.

Zhang, T.; Chen, X.; Wang, Y.; Wang, Y.; and Zhao, H. 2022. MUTR3D: A Multi-camera Tracking Framework via 3D-to-2D Queries. In IEEE Conf. Comput. Vis. Pattern Recog., 4537–4546.

Zheng, B.; Ma, N.; Tong, S.; and Xie, S. 2026. Difusion Transformers with Representation Autoencoders. In Int. Conf. Learn. Represent.

Zhu, Z.; Wu, Z.; Zhu, Z.; Zhou, L.; Sun, H.; Wang, B.; Ma, K.; Chen, G.; Ye, H.; Xie, J.; et al. 2026. WorldSplat: Gaussian-Centric Feed-Forward 4D Scene Generation for Autonomous Driving. In Int. Conf. Learn. Represent.

Zong, Z.; Jiang, D.; Song, G.; Xue, Z.; Su, J.; Li, H.; and Liu, Y. 2023. Temporal enhanced training of multi-view 3d object detector via historical object prediction. In Int. Conf. Comput. Vis., 3781–3790.

Zuo, S.; Zheng, W.; Huang, Y.; Zhou, J.; and Lu, J. 2025. GaussianWorld: Gaussian World Model for Streaming 3D Occupancy Prediction. In IEEE Conf. Comput. Vis. Pattern Recog., 6772–6781.

## A Dual-branch Scene Autoencoder Details

## A.1 Dense Geometry Autoencoder

The geometry encoder ${ \mathcal E } _ { \mathrm { g e o } }$ builds on the frozen Depth Anything V3 Base (DA3-Base) encoder (Lin et al. 2026). To reduce the token space and alleviate GPU memory consumption, we extract the token representations from the $5 ^ { \mathrm { t h } }$ layer of the DA3-Base ViT backbone as the continuous dense geometry representation $z _ { \mathrm { g e o } } ^ { c }$ . On top of these features, the RAE bottleneck applies a lightweight 3D convolutional encoder with a $1 \times 1 \times 1$ kernel, which projects each spatio-temporal feature location into the MMDiT hidden space, producing the final geometry latent $z _ { \mathrm { g e o } } ~ \in ~ \mathbb { R } ^ { B \times T \times N _ { \mathrm { g e o } } \times \hat { D } _ { \mathrm { g } } }$ , where $N _ { \mathrm { g e o } } = H _ { \mathrm { g e o } } \times W _ { \mathrm { g e o } }$ . Unlike standard VAEs that often induce high-frequency detail loss via strict KL-divergence regularization, the RAE formulation maintains a well-regularized and smooth feature manifold. The geometry autoencoder is trained end-to-end with a composite objective:

$$
{ \mathcal { L } } _ { \mathrm { G e o - A E } } = \lambda _ { \mathrm { r g b } } { \mathcal { L } } _ { \mathrm { r g b } } + \lambda _ { \mathrm { d e p t h } } { \mathcal { L } } _ { \mathrm { d e p t h } } ,\tag{8}
$$

where $\mathcal { L } _ { \mathrm { r g b } }$ applies an $\mathcal { L } _ { 1 }$ penalty between the rendered and ground-truth RGB images, and ${ \mathcal { L } } _ { \mathrm { d e p t h } }$ provides sparse depth regularization from ground-truth LiDAR maps to maintain structural accuracy. The geometry autoencoder is fine-tuned from the pretrained DA3 weights, and temporal interactions among the flattened geometry tokens are modeled by the subsequent MMDiT attention blocks.

## A.2 Sparse Bounding Box Autoencoder

The BBox Autoencoder is trained independently before joint difusion training. For each frame, object tracks are padded or truncated to a fixed number of slots, where each valid slot contains a 3D box $( x , y , z , w , l , h , \theta )$ and a semantic category. The BBox Autoencoder $\mathcal { E } _ { \mathrm { b o x } }$ processes slot-aligned 3D bounding box sequences by first converting each box into its eight 3D corners, applying a Fourier Positional Embedding (FPE), and concatenating a learnable class embedding. The resulting per-slot features are linearly projected into a hidden dimension and processed by a stack of spatiotemporal transformer blocks (ST-Transformer). Within these blocks, the model first performs frame-local spatial attention across object slots to capture cross-entity relationships, followed by temporal attention along the trajectory of each individual object slot to model temporal dynamics. Finally, a linear latent head predicts the posterior mean and log-variance for each slot, producing the bounding box latents $z _ { \mathrm { b o x } }$

To accommodate scenes containing a variable number of objects, we allocate at most $N _ { \mathrm { m a x } }$ slots and define a binary mask $\bar { M } \in \{ 0 , 1 \} ^ { N _ { \operatorname* { m a x } } }$ to pad invalid slots. The overall reconstruction objective is formulated as:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { B B o x - A E } } = \mathcal { L } _ { \mathrm { c o r n e r } } + \lambda _ { \mathrm { y a w } } \mathcal { L } _ { \mathrm { y a w } } + \lambda _ { \mathrm { c l s } } \mathcal { L } _ { \mathrm { c l s } } } \\ { + \lambda _ { \mathrm { e x i s t } } \mathcal { L } _ { \mathrm { e x i s t } } + \lambda _ { \mathrm { K L } } \mathcal { L } _ { \mathrm { K L } } . \quad } \end{array}\tag{9}
$$

The geometric and semantic losses are strictly evaluated over valid slots $( M _ { i } ~ = ~ 1 )$ , which include an $\ell _ { 1 }$ penalty on the 3D bounding box corners $( \mathcal { L } _ { \mathrm { c o r n e r } } = \Vert P _ { i } - \hat { P } _ { i } \Vert _ { 1 }$ where $P _ { i }$ is the corner point), a continuous Euclidean loss in the SO(2) embedding space for sine-cosine yaw regression $( \mathcal { L } _ { \mathrm { y a w } } = \Vert [ \sin \theta _ { i } , \cos \theta _ { i } ] ^ { \top } - [ \sin \hat { \theta } _ { i } , \cos \hat { \theta } _ { i } ] ^ { \top } \Vert _ { 2 } ^ { 2 }$ , where θ is the yaw angle) to circumvent periodicity artifacts, and a cross-entropy loss for semantic categories $( \mathcal { L } _ { \mathrm { c l s } } )$ . Across all slots, $\mathcal { L } _ { \mathrm { e x i s t } }$ computes a binary cross-entropy loss for slot existence probability, and ${ \mathcal { L } } _ { \mathrm { K L } }$ provides the standard Kullback-Leibler divergence to regularize the latent distribution.

## B Training and Implementation Details

This section details the configuration and optimization strategies for our full 6-camera surround-view training pipeline on the nuScenes dataset (Caesar et al. 2020). We emphasize the spatial-temporal formulation, cross-view attention mechanism, and complex training curriculum.

## B.1 Data Preprocessing and Geometry Encoding

Our model processes a temporal clip of $T = 8$ frames across $N _ { C } = 6$ surrounding cameras per sample. The input images are resized to a resolution of $1 1 2 \times 1 6 8 .$ . Given an input shape of $[ B \times T , N _ { C } , 3 , H , W ]$ , the encoder downsamples the spatial resolution by a factor of 14, yielding geometric latent tokens $z _ { \mathrm { g e o } } \in \mathbb { R } ^ { \check { B } \times N _ { C } \times C \times T \times h \times w }$ , where $\bar { C } = 1 5 3 6$ $h = 8 ,$ , and $w = 1 2$

## B.2 Condition Encoding with Wan-VAE

The $6 \times 8$ input frames are not concatenated into a single 48- frame video. Each camera stream is encoded independently as an 8-frame video: the input $\left[ B , T , N _ { C } , 3 , H , \mathrm { \bar { \it W } } \right]$ is reshaped into $B \times N _ { C }$ videos of shape $[ 3 , T , H , W ]$ , which the frozen Wan-VAE maps to condition latents of shape $[ B , N _ { C } , 1 6 , 2 , H / 8 , W / 8 ]$

## B.3 Camera Pose Encoding and Unified Spatial Grounding

To spatially ground the geometric features, we inject explicit camera pose tokens into the latent space. For the i-th camera at the t-th frame, the continuous pose token $\mathcal { P } _ { t } ^ { i }$ is constructed from both intrinsics (normalized by image width/height) and extrinsics (transformed relative to the ego coordinate system at $t = 0 )$ . The translation components are further normalized by a predefined scene radius factor $\rho = 8 0$

To facilitate high-frequency spatial sensitivity, we apply a Fourier positional embedding $\gamma ( \cdot )$ with 4 frequency bands to the 6-DoF pose vector:

$$
f _ { \mathrm { p o s e } } = \mathrm { M L P } \big ( \gamma ( \mathcal { P } _ { t } ^ { i } ) \big ) .\tag{10}
$$

This camera pose embedding $f _ { \mathrm { p o s e } }$ is subsequently added to the corresponding geometric tokens $z _ { \mathrm { g e o } }$ and concatenated as part of the conditional context for the difusion backbone.

## B.4 Ring-Structured Cross-View Attention

Instead of relying on computationally prohibitive fullattention across all cameras, we implement a memoryeficient ring-structured cross-view attention mechanism (Gao et al. 2025). The 6 cameras are modeled as a closed topological ring $( C _ { 0 }  C _ { 1 }  C _ { 2 }  C _ { 3 }  C _ { 4 } $ $C _ { 5 }  \bar { C _ { 0 } } )$ . During the forward pass within each MMDiT block, the geometric tokens from camera $C _ { i }$ only attend to their immediate spatial neighbors $C _ { ( i - 1 ) \% 6 }$ and $C _ { ( i + 1 ) \% 6 } .$ It is important to note that this cross-view interaction is strictly applied to the dense geometry tokens $z _ { \mathrm { g e o } } ;$ the discrete bounding box tokens bypass the cross-view attention, are replicated across camera branches during inference, and are eventually averaged to produce the final layout predictions.

## B.5 Training Loss

The total training loss combines the geometry difusion objective, the layout latent flow-matching objective, and auxiliary layout supervision:

$$
\mathcal { L } _ { \mathrm { t o t a l } } = \mathcal { L } _ { \mathrm { g e o } } + \lambda _ { \mathrm { b o x } } \mathcal { L } _ { \mathrm { b o x } } + \mathcal { L } _ { \mathrm { a u x } } ,\tag{11}
$$

where $\mathcal { L } _ { \mathrm { g e o } }$ is the difusion objective for the geometry branch, and $\mathcal { L } _ { \mathrm { b o x } } ^ { \mathrm { ~ \tiny ~ - ~ } }$ is the velocity-matching loss for the BBox Autoencoder latent $z _ { \mathrm { b o x } } ,$ evaluated only on valid object slots. We set $\lambda _ { \mathrm { b o x } } = 5 . 0$ by default.

In addition, we supervise four auxiliary layout variables: object confidence $c ,$ anchor-relative center ofset $\delta ,$ groundplane velocity $v ,$ and object attribute $a .$ For each auxiliary target $y _ { j } ,$ , where $j \in \{ c , \delta , v , a \}$ , we apply the same rectifiedflow objective. Given Gaussian noise $\epsilon _ { j }$ and noise level $\sigma _ { : }$ we construct

$$
y _ { j , \sigma } = ( 1 - \sigma ) y _ { j } + \sigma \epsilon _ { j } , \qquad \tilde { v } _ { j } = \epsilon _ { j } - y _ { j } ,\tag{12}
$$

and train the corresponding prediction head $v _ { j }$ with an MSE loss.

The confidence target is defined over all anchor slots, with $c = 1$ for occupied slots and $c = - 1$ for empty slots. For occupied slots, the center ofset target is computed relative to the fixed anchor center:

$$
\begin{array} { r } { \delta ^ { t , i } = \mathrm { c l i p } \left( \mathbf { x } _ { \mathrm { b o x } } ^ { t , i } - \mathbf { x } _ { \mathrm { a n c h o r } } ^ { i } , - 1 , 1 \right) , } \end{array}\tag{13}
$$

The velocity target is the normalized ground-plane velocity, clipped to a bounded range:

$$
v ^ { t , i } = \mathrm { c l i p } \left( { \mathbf { u } } ^ { t , i } , - 3 , 3 \right) .\tag{14}
$$

Finally, the attribute target $\boldsymbol { a } ^ { t , i }$ is represented as a one-hot vector, and is supervised only when a valid attribute annotation is available.

The auxiliary loss is therefore:

$$
\mathcal { L } _ { \mathrm { a u x } } = \lambda _ { c } \mathcal { L } _ { c } + \lambda _ { \delta } \mathcal { L } _ { \delta } + \lambda _ { v } \mathcal { L } _ { v } + \lambda _ { a } \mathcal { L } _ { a } .\tag{15}
$$

In practice, we set $\lambda _ { c } = \lambda _ { v } = \lambda _ { a } = 1 . 0$ and $\lambda _ { \delta } = 2 . 0$

## B.6 Training Curriculum and Flow Matching Hyperparameters

Our unified MMDiT is initialized with the pretrained Wan2.1-1.3B backbone weights (Wan et al. 2025), leaving the DA3 encoder (Lin et al. 2026) and the VAE components strictly frozen. We train the entire MMDiT trunk from the first epoch using AdamW optimizer. The flow-matching objective is simulated with 1000 integration timesteps and a flow shift factor of 5.0.

To stabilize the optimization of high-frequency geometric details, we design a progressive three-stage noise curriculum:

• Phase I (0%–10% progress): High-noise sampling is disabled $( \sigma \le 0 . 8 )$

• Phase II (10%–20% progress): Samples are drawn from a high-noise range of [0.65, 0.9] with a 50% probability.

• Phase III (20%–100% progress): The high-noise sampling range is expanded to [0.7, 1.0] with a 50% probability.

Furthermore, to enhance robust unconditional generation capabilities, we apply a classifier-free guidance dropout of $p _ { \mathrm { c f g } } = 0 . 1$ to the visual conditions.

## B.7 Model Configuration and Optimization Hyperparameters

For the geometry autoencoder, the projected features are flattened into geometry tokens with embedding dimension $D _ { \mathrm { g } } = 1 5 3 6$ . The geometry autoencoder is fine-tuned for 150k iterations before joint training. For the BBox Autoencoder, object tracks are padded or truncated to at most $N _ { \mathrm { m a x } } ^ { \mathrm { A E } } = 1 0 0$ tracks per frame, and each track is mapped into a latent token of dimension $D _ { \mathrm { b } } = 6 4$ . The BBox Autoencoder is trained for 16k iterations. We set the loss hyperparameters $\lambda _ { \mathrm { e x i s t } } = 1 . 0$ $\lambda _ { \mathrm { c l s } } = 0 . 5 , \lambda _ { \mathrm { y a w } } = 1 . 0$ , and $\lambda _ { \mathrm { K L } } = \bar { 1 } 0 ^ { - 4 }$

For joint difusion, we instantiate $N _ { \mathrm { a n c h o r } } = 1 2 0 0$ fixed layout queries on a 40 × 30 BEV grid. The grid spans $x \in \ [ - 4 5 , 9 0 ]$ m and $y ~ \in ~ \left[ - 6 0 , 6 0 \right] \mathrm { m }$ , corresponding to a cell resolution of $3 . 3 7 5 \times 4 . 0 \mathrm { m }$ . Each query is anchored at the center of its corresponding BEV cell with $z = 0 ,$ , and the anchors are shared across all frames. The joint latent sequence is processed by a Multi-Modal Difusion Transformer (MMDiT) (Wan et al. 2025), where the geometry and layout tokens are embedded by the proposed UPE module into a unified latent space for difusion.

The unified MMDiT consists of 30 layers with hidden dimension $D = 1 5 3 6$ and 12 attention heads. For joint difusion training, the model is trained on 8 NVIDIA H800 GPUs for 1500k iterations using the AdamW optimizer with an initial learning rate of $1 \times \bar { 1 0 ^ { - 4 } }$ , followed by cosine decay to $1 \times 1 0 ^ { - 5 }$ . We use a total batch size of $^ { 8 , }$ with one video clip per GPU and no gradient accumulation. Gradient clipping with a maximum norm of 1.0 is applied for stability.

## B.8 Decoding and Post-Processing at Inference

We use 50 rectified-flow denoising steps at inference. Since the learned frame embeddings and box tensors are fixed to $T { = } 8 ,$ , sequences longer than 8 frames are processed with overlapping 8-frame sliding windows. At inference time, the generated geometry latent is denormalized and propagated through the DPT decoder to obtain 3D Gaussian representations, while layout tokens are denormalized and decoded by the frozen BBox-AE into 3D bounding boxes, class logits, and existence logits. Box centers are recovered from the predicted anchor-relative ofsets $\delta ,$ and final detections are selected by confidence thresholding and per-frame 3D nonmaximum suppression.

## C Eficiency

On a single H800 GPU, USR-Drive processes one 6-camera × 8-frame clip in 45.2 s with 58.5 GB peak memory, dominated by the 50-step iterative denoising, while the underlying DA3 encoder alone takes 1.3 s and 46.6 GB. Unlike per-scene optimization methods that require hours of test-time optimization per scene, our model is fully feed-forward apart from the denoising loop, and the training cost is amortized across all scenes. USR-Drive targets ofline unified scenestate reconstruction and perception rather than real-time onboard deployment; distilling the multi-step denoiser into a few-step causal model is a promising direction for future work.

<table><tr><td>Auxiliary variable placement</td><td>mAP↑</td><td>mATE↓</td><td>PSNR↑</td></tr><tr><td>No auxiliary</td><td>0.223</td><td>0.801</td><td>27.49</td></tr><tr><td>Inside BBox Autoencoder</td><td>0.445</td><td>0.679</td><td>27.31</td></tr><tr><td>Separate Learnable Tokens (Ours)</td><td>0.552</td><td>0.525</td><td>27.55</td></tr></table>

Table 5: Ablation on auxiliary layout tokens.

## D More Ablation Studies

## D.1 Ablation on Auxiliary Layout Variables

We ablate how to incorporate auxiliary layout variables, including object confidence c, anchor-relative center ofset δ, velocity v, and dynamic/static attribute a. One alternative is to encode these variables directly into the BBox Autoencoder and reconstruct them together with the box representation. However, this couples the autoencoder to the fixed-anchor denoising design: confidence is defined over both occupied and empty anchor slots, and δ depends on the anchor centers used by the difusion model. Any change to these targets or the anchor design would then require retraining the autoencoder. In addition, velocity and attribute labels can be sparse or unavailable for some boxes, which is more naturally handled with masked auxiliary difusion heads. Therefore, we keep the BBox Autoencoder focused on the core box latent and add $c , \delta , v ,$ and a as separate denoising branches in the layout difusion model. This preserves the pretrained autoencoder, keeps the auxiliary variables explicit and controllable, and enables clean ablations. Results are reported in Tab. 5.

## D.2 Ablation on Training Parameters

We further ablate two key training hyperparameters: the bounding-box loss weight $\lambda _ { \mathrm { b o x } }$ and the high-noise curriculum, with all other settings kept unchanged.

As shown in Tab. $6 , \lambda _ { \mathrm { b o x } }$ balances dense geometry reconstruction and sparse layout prediction. A small weight improves reconstruction but hurts detection, while an overly large weight over-emphasizes boxes and degrades PSNR/SSIM. We use $\lambda _ { \mathrm { b o x } } ~ \stackrel { - } { = } ~ 5 . 0$ by default, which gives the best overall trade-of.

For the high-noise curriculum, our default setting disables high-noise sampling in the first 10% of training, enables it with probability 0.5 from noise range [0.65, 0.9] during 10%– 20%, and expands the range to [0.7, 1.0] afterwards. Removing this curriculum destabilizes early training and degrades both reconstruction and detection, indicating that gradually increasing denoising dificulty helps the model learn local geometry before global layout.

<table><tr><td>Configuration</td><td>PSNR↑</td><td>SSIM↑</td><td>mAP↑</td><td>mATE↓</td></tr><tr><td> $\lambda _ { \mathrm { b o x } } = 1 . 0$ </td><td>27.42</td><td>0.849</td><td>0.449</td><td>0.624</td></tr><tr><td> $\lambda _ { \mathrm { b o x } } = 3 . 0$ </td><td>27.50</td><td>0.851</td><td>0.491</td><td>0.575</td></tr><tr><td> $\lambda _ { \mathrm { b o x } } = 7 . 0$ </td><td>27.20</td><td>0.841</td><td>0.521</td><td>0.540</td></tr><tr><td> $\lambda _ { \mathrm { b o x } } = 1 0 . 0$ </td><td>26.38</td><td>0.812</td><td>0.516</td><td>0.548</td></tr><tr><td> $\lambda _ { \mathrm { b o x } } = 5 . 0 , \mathrm { w / o } \mathrm { C u r r } .$ </td><td>26.95</td><td>0.831</td><td>0.473</td><td>0.585</td></tr><tr><td> $\lambda _ { \mathbf { b o x } } = 5 . 0 ,$  w/ Curr.</td><td>27.55</td><td>0.853</td><td>0.552</td><td>0.525</td></tr></table>

Table 6: Ablation on training hyperparameters. We evaluate the bounding-box loss weight $\lambda _ { \mathrm { b o x } }$ and the high-noise curriculum. The default setting achieves the best balance between geometric fidelity and layout accuracy.

<table><tr><td>Feature Dim.</td><td>|Params (B)|</td><td>|PSNR↑</td><td>SSIM↑ LPIPS↓</td><td></td></tr><tr><td>768</td><td>0.356</td><td>25.84</td><td>0.792</td><td>0.138</td></tr><tr><td>1024</td><td>0.631</td><td>26.68</td><td>0.825</td><td>0.104</td></tr><tr><td>1536 (Default)</td><td>1.418</td><td>27.55</td><td>0.853</td><td>0.076</td></tr><tr><td>2048</td><td>2.517</td><td>27.48</td><td>0.849</td><td>0.080</td></tr></table>

Table 7: Ablation on MMDiT hidden dimension. The default 1536-dimensional setting achieves the best trade-of between reconstruction quality and model size.

## D.3 Ablation on DiT Hidden Dimension

We ablate the MMDiT hidden dimension D while keeping all other settings unchanged. As shown in Tab. 7, smaller D reduces cost but creates a bottleneck for 1536-channel geometry features and breaks compatibility with pretrained Wan2.1- 1.3B weights. Larger D (e.g., 2048) greatly increases parameters and training cost with only marginal gains, while also losing pretrained initialization. Thus, we use $D = 1 5 3 6$ as the best trade-of between capacity, eficiency, and architectural compatibility.

## E Qualitative Results

## E.1 Reconstruction Results on nuScenes

Fig. 5 presents qualitative reconstruction results on the nuScenes dataset. Compared to baseline methods, our approach reveals better appearance quality on both static scenes (top 2 rows) and dynamic objects (bottom 2 rows). In particular, our method recovers sharper scene details, cleaner object boundaries, and more coherent dynamic content under challenging driving scenarios, benefiting from the unified latent modeling of dense geometry and sparse object layout.

## E.2 3D Detection Results on nuScenes

Fig. 6 visualizes 3D detection results on the nuScenes dataset. While previous methods sufer from missed or inaccurate detection, our method achieves better results in terms of both completeness and accuracy. This indicates that the joint generation paradigm benefits from the information of the geometric branch, which provides structural context and spatial grounding for object-level detection.

GT  
Ours  
DA3  
DGGT  
AnySplat  
![](images/3ce8c1d90751534c0a6d45a68e531ed038bbaf2fccbad802da2a19d734fd3138.jpg)  
Figure 5: Visualization of reconstruction results. Compared to baseline methods, our approach reveals better appearance quality on both static scenes (top 2 rows) and dynamic objects (bottom 2 rows).

## E.3 Visualization of Ablation Studies

Fig. 7 visualizes the ablation variants reported in the main paper. The geometry-only variant produces reasonable appearances but lacks object-level structure, while removing UPE leads to visible misaligned objects and implausible box placements. The complete model provides the best reconstruction quality and detection accuracy, confirming the necessity of both the co-denoising design and the unified positional encoding.

## E.4 Zero-Shot Results on VKitti

Fig. 8 and Fig. 9 show zero-shot 3D detection and reconstruction results on the VKitti dataset, respectively. Without any fine-tuning, our method transfers well to the unseen synthetic domain, producing tightly grounded bounding boxes and high-fidelity reconstructions, whereas baseline methods sufer from missed detections or degraded appearance quality.

## E.5 Depth Estimation

Fig. 10 compares the depth maps derived from the reconstructed 3D Gaussians. Our method achieves the best depth estimation in both qualitative and quantitative evaluations, further verifying that the unified latent space preserves accurate metric scene structure.

GT  
Ours  
RayDN  
HoP  
StreamPETR  
![](images/14fc9883978a5b49db26133d0809cada91aaa420e688508ac7bb278dd9767e6b.jpg)  
Figure 6: Visualization of 3D detection results. While previous methods sufer from missed or inaccurate detection, our method achieves better results considering completeness and accuracy.

Ground Truth  
Box Only  
Ours w/o UPE  
Ours  
![](images/1f7ba7a780a75fedb90a38540cc87db574a18248747aabf04cebdcd19773ab71.jpg)  
Figure 7: Visualization of Ablation Studies. The complete model provides the best reconstruction efect and detection quality.

DA3  
DGGT  
Ours  
GT  
AnySplat  
![](images/d6f9a8054212789a6d8e5326f5382c018411f6ac71a982cb454151c11c53c0ca.jpg)  
Figure 8: Visualization of 3D detection results on VKitti dataset. While previous methods sufer from missed or inaccurate detection, our method achieves better results considering completeness and accuracy.

![](images/93fd2b0d5b48cb395798da10b79507d65f708afafcf2f2c774517699980132db.jpg)  
Figure 9: Visualization of reconstruction results on VKitti dataset. Compared to baseline methods, our approach reveal better appearance quality.

![](images/a75b9c68d44c1840a8459d7f9f05bdd1f57eee127534b2b2270b6752db6697fc.jpg)  
Figure 10: Visualization of Depth results. Our method achieves best depth estimation on both qualitative and quantitative results.