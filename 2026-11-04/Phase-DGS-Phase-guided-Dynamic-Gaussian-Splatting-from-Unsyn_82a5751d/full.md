# Phase-DGS: Phase-guided Dynamic Gaussian Splatting from Unsynchronized Multi-view Video

Hosung Jeon<sup>†1</sup> Jun Young Jeong<sup>1</sup> Sangwoon Kwak<sup>†1</sup> Joonsoo Kim<sup>1</sup> Sangmin Kim<sup>2</sup> Jaesik Park<sup>2</sup> Won-Sik Cheong<sup>1</sup> Hyon-Gon Choo<sup>1</sup>

<sup>1</sup>Electronics and Telecommunications Research Institute (ETRI), Daejeon, Republic of Korea <sup>2</sup>Seoul National University, Seoul, Republic of Korea

{h.jeon, jyj0120, s.kwak, joonsoo.kim, wscheong, hgchoo}@etri.re.kr {sm.kim, jaesik.park}@snu.ac.kr

## Abstract

Spatiotemporal scene reconstruction with Dynamic Gaussian Splatting (DGS) fundamentally depends on perfect temporal synchronization across all cameras, an assumption rarely achievable in practice. Industry-standard synchronization requires expensive specialized hardware and complex setup, while data-driven alternatives including audio-based alignment, geometry matching, and pose tracking remain limited by restrictive environmental or scene-specific assumptions. To overcome these limitations, we propose Phase-DGS, a framework that replaces conventional timestamps with a semantic phase representing each frame’s inherent state within the underlying motion cycle. Frames capturing the same motion state share the same phase regardless of which camera records them or when, naturally establishing spatiotemporal correspondence across unsynchronized cameras. We validate Phase-DGS under temporal offsets, frame drops, and camera freezes, achieving up to +8.5 dB PSNR improvement over baselines while enabling seamless integration with existing DGS backbones including RealTime-4DGS and FreeTimeGS.

## 1 Introduction

In recent years, radiance field representations such as Neural Radiance Fields (NeRF) [6, 21, 37] and 3D Gaussian Splatting (3DGS) [11, 18, 42] have been extended to dynamic scenes, leading to Dynamic NeRF (DNeRF) and Dynamic Gaussian Splatting (DGS), which have emerged as a leading paradigm for spatiotemporal scene reconstruction. The latest of these methods [15, 24, 34, 35] reconstruct complex real-world scenes with striking realism, enabling applications spanning from autonomous navigation to immersive content creation. However, such results come with a prerequisite: perfect temporal synchronization across all cameras, with each camera capturing the scene at exactly the same instant, frame-by-frame [12, 14, 39]. Without such synchronization, observations from different viewpoints no longer correspond to the same scene instant, making it ill-posed to learn a consistent radiance field under a shared timestamp. This spatiotemporal incoherence inevitably produces floaters, severe ghosting, and geometric distortion in rendered views.

A straightforward approach is to enforce camera synchronization during the capture stage through standardized solutions, providing the radiance field with already-aligned frames that guarantee spatiotemporal coherence. Trigger-based methods such as Genlock [27, 30, 31] synchronize all camera shutters via a master sync signal, achieving high precision. However, depending on the capture environment, trigger signals can be degraded by attenuation and noise, and maintaining highquality synchronization requires expensive specialized hardware and complex wiring. Protocol-based methods such as the Precision Time Protocol (PTP) [9] and Network Time Protocol (NTP) [1] avoid dedicated wiring and offer synchronization at reasonable cost, but rely heavily on network bandwidth and stability. When transmission delays or packet losses occur, frames can be dropped, or cameras may suffer temporal drift in which their time axes gradually diverge from one another.

![](images/c0a4e606253f34bf6b319d42ba380f5c3555c0fca1ccb1623557c16dc0e9f124.jpg)  
Figure 1: Conceptual diagram of Phase-DGS. (a) In an unsynchronized multi-view dataset, images sharing the frame index are actually captured at different time instants. The proposed phase extraction module estimates an inherent semantic phase from each input image, which is then used to train 4DGS. (b) The graph shows the estimated phases across cameras. When a sequence is offset, the phase shifts along the x-axis direction; when frozen, the phase is predicted parallel to the x-axis; and when dropped, the phase is predicted parallel to the y-axis.

Beyond capture-stage synchronization, recent works have explored training radiance fields with improved spatiotemporal coherence by using cues extracted from the captured data itself. Audiobased alignment methods [2, 8, 28, 29, 33] synchronize cameras by matching audio signals across microphones. However, they rely on a carefully controlled acoustic environment, a condition that rarely holds in open-capture settings. Geometry- and feature-matching methods [5, 7, 17, 36] estimate the degree of unsynchrony by finding spatiotemporal correspondences across views, but they struggle with weak scene texture or large viewpoint differences. Human- and pose-tracking approaches [3, 10, 25, 32, 40] align frames by tracking human body poses across cameras, but such reliance restricts their use to human-centric scenarios where a person remains visible across all frames. Reconstruction-coupled methods [12, 14, 39] jointly optimize sub-frame-level temporal misalignment together with the radiance field while most assign only a single offset per camera. Furthermore, these methods require camera pose information, which is typically obtained from synchronized data or provided as known calibration in the dataset. In unsynchronized recordings, however, temporal misalignment makes accurate camera pose estimation difficult.

Instead of estimating absolute temporal offsets to synchronize videos, we infer an inherent semantic phase for each image that represents its position within the dynamic instance. Based on this idea, we propose Phase-DGS, a framework that integrates a phase extraction module into 4D Gaussian Splatting (4DGS) to learn a radiance field with high spatiotemporal coherence, where images sharing the same phase naturally establish spatiotemporal correspondence regardless of which camera captures them or when. As shown in Fig. 1, even when cameras are unsynchronized and their frame indices no longer correspond to the same instant, our phase extraction module assigns the same inherent semantic phase to images capturing the same motion state, such as the "jumping" pose, across different cameras. The estimated phases then replace absolute timestamps as input to 4DGS, allowing it to reconstruct spatiotemporally coherent rendered images. Its modular design further enables plugand-play integration with state-of-the-art 4DGS backbones, as validated on RealTime-4DGS [35] and FreeTimeGS [34]. Experimentally, we validate that the proposed method optimizes 4DGS with enhanced spatiotemporal coherence under various frame-level errors, including temporal offsets, frame drops, and camera freezes. Our method achieves up to +8.5 dB improvement in PSNR over baseline methods under unsynchronized conditions.

## 2 Related Works

Existing synchronization approaches fall into several categories, each with its own limitations. Hardware- and software-based methods such as Genlock, Timecode, and PTP offer high precision but provide no post-hoc correction for timing errors induced by signal drift, and their cost remains a barrier to practical deployment. Audio-based methods [2, 8, 29, 28, 33] align videos through shared acoustic events but rely on a controlled acoustic environment and suffer from non-negligible audio-visual latency in large-scale setups, where the finite speed of sound introduces meaningful error. Geometry and feature matching methods [5, 7, 16, 20] estimate the amount of unsynchrony by finding spatiotemporal correspondences across views, and they are highly dependent on the quality of feature extraction and degrade considerably under the occlusion of tracked features. Approaches based on human or pose tracking [3, 10, 25, 32, 40] align frames by tracking human body poses across cameras, and strictly require the presence of humans in the scene. It substantially limits their applicability to common real-world environments. Learning-based methods such as Sync-NeRF [12, 14, 39] jointly optimize temporal offsets as learnable parameters by using rendering quality as a supervisory signal, yet they assume continuous frame availability across all views and fundamentally cannot recover from per-camera frame drops and freezes that arise independently across cameras.

## 3 Method

## 3.1 Overview

Phase-DGS replaces the absolute timestamp used in conventional 4DGS with a per-frame phase derived from the image content. We observe that each frame possesses an inherent semantic phase within video sequences, independent of its absolute timestamp. For instance, in a sit-to-stand motion, the "fully seated" pose corresponds to the same phase regardless of which camera captures it or at what moment. Based on this observation, Phase-DGS extracts a phase $p _ { i } ^ { f }$ from each camera image $u _ { i } ^ { f }$ and uses it instead of the timestamp t to supervise 4DGS training. Note that $f$ and i represent the frame and camera indices, respectively. Since the phase is derived from the image content, it naturally accommodates unsynchronized capture conditions such as frame drops, frame freezes, and temporal offsets between cameras.

## 3.2 Phase Extraction Module

We introduce the phase projector module, which extracts a phase $p _ { i } ^ { f }$ from each camera image $u _ { i } ^ { f }$ using a pretrained Vision Transformer (ViT) encoder [4, 23] followed by a lightweight MLP.

Feature Generation. The input image $u _ { i } ^ { f }$ of the f-th frame of the i-th camera is coarsely cropped to a motion-intensive region of interest (ROI) and resized to match the input resolution of the ViT encoder. Then, the cropped image is passed through a pretrained ViT encoder, which partitions it into $H \times W$ patches and generates a token for each patch. We spatially aggregate all patch tokens by averaging to obtain a d-dimensional global feature vector $z _ { i } ^ { f } \in \mathbb { R } ^ { d }$ that summarizes the visual content of the image:

$$
z _ { i } ^ { f } = \frac { 1 } { H W } \sum _ { n = 1 } ^ { H \times W } \left[ \mathrm { V i T } _ { \mathrm { e n c } } \Big ( u _ { i } ^ { f } \Big ) \right] _ { n } .\tag{1}
$$

Phase Projector. The global feature vector $z _ { i } ^ { f }$ is subsequently passed through a lightweight MLP that consists of three fully connected layers, which produces a normalized phase value $p _ { i } ^ { f } \in [ 0 , 1 ]$

$$
p _ { i } ^ { f } = \mathbf { M } \mathbf { L } \mathbf { P } \left( z _ { i } ^ { f } \right) .\tag{2}
$$

Notably, the phase projector is trained without any explicit phase supervision; instead, it is optimized through the rendering loss together with the phase consistency and monotonic losses (see Sections 3.4 and 3.6), enabling unsupervised phase extraction directly from visual observations.

## 3.3 Feature Bank

The feature bank B serves as a learnable reference repository that enforces cross-camera phase coherence, ensuring that identical motion states share the same phase across all views by enabling the phase projector to map each input feature to a phase through comparison against stored references. The feature bank is defined as $\dot { \boldsymbol { B } } \in \mathbb { R } ^ { I \times ( F \times k ) \times \dot { d } }$ , where I is the number of cameras, F is the video length, k is an upsampling factor, and d is the dimension of the feature. $k > 1$ is designed to handle videos with varying frame rates or staggered frame timestamps on some cameras, providing finer temporal resolution in the feature bank. Since the DINOv2 features are extracted independently from each image and do not contain temporal information, we make the feature bank learnable rather than fine-tuning the entire ViT encoder for every scene. Inspired by temporal prototype learning [22], which aligns multiple videos through learned prototypes, the feature bank is optimized jointly with the phase projector to adapt to the scene, so that frames capturing the same motion state are assigned the same phase across all views. When the feature bank is fixed, which corresponds to a fixed-memory alternative, the accuracy of phase estimation drops substantially (see Section 4.5).

## 3.4 Loss Function

Phase-DGS is trained with a combination of three loss functions,

$$
\mathcal { L } = \lambda _ { p h a s e } \mathcal { L } _ { p h a s e } + \lambda _ { m o n o } \mathcal { L } _ { m o n o } + \lambda _ { r e n d e r } \mathcal { L } _ { r e n d e r } .\tag{3}
$$

Phase Consistency Loss. $\mathcal { L } _ { p h a s e }$ supervises the phase projector so that it retrieves features with matching phases from the feature bank. Given an input image $u _ { i } ^ { f }$ , the extracted feature $z _ { i } ^ { f }$ is passed through the phase projector to obtain the phase $p _ { i } ^ { f }$ . The entire i-th camera array of the feature bank, $B _ { i } \in \mathbb { R } ^ { ( F \times k ) \times d }$ , is then passed through the phase projector to generate a phase map $\mathcal { P } _ { i } \in \mathbb { R } ^ { ( F \times k ) \times 1 }$ An attention weight w is computed by applying softmax to the distance between the phase map and $p _ { i } ^ { f }$

$$
w = \mathrm { s o f t m a x } \left( - \left| \mathcal { P } _ { i } - p _ { i } ^ { f } \right| \right) .\tag{4}
$$

The reference feature $\hat { z } _ { i } ^ { f }$ that shares the same phase as $p _ { i } ^ { f }$ is then retrieved through a weighted sum over the feature bank,

$$
\hat { z } _ { i } ^ { f } = \sum w \cdot B _ { i } .\tag{5}
$$

The loss is defined as the MSE between $z _ { i } ^ { f }$ and $\hat { z } _ { i } ^ { f }$ ,

$$
\mathcal { L } _ { p h a s e } = \mathbf { M } \mathbf { S } \mathrm { E } \left( z _ { i } ^ { f } , \hat { z } _ { i } ^ { f } \right) .\tag{6}
$$

Monotonic Loss. ${ \mathcal { L } } _ { m o n o }$ promotes the phase map generated from the feature bank to increase monotonically along the temporal axis. The entire feature bank is passed through the phase projector to produce $\dot { \mathcal { P } } \in \mathbb { R } ^ { \mathbf { \breve { I } } \times ( F \times k ) \times 1 }$ , and the difference between adjacent phases is computed for each camera,

$$
\Delta _ { i } ^ { f } = \mathcal { P } _ { i } ^ { f + 1 } - \mathcal { P } _ { i } ^ { f } , \quad f = 0 , 1 , \ldots , F k - 2 .\tag{7}
$$

A penalty is applied whenever the phase difference falls below the minimum increment $\begin{array} { r } { a = \frac { 1 } { F \times k } } \end{array}$

$$
\mathcal { L } _ { m o n o } = \sum _ { i } \sum _ { f } \mathrm { r e l u } \left( - \Delta _ { i } ^ { f } + \frac { 1 } { F k } \right) .\tag{8}
$$

This term encourages the phase projector to learn temporally consistent phases across the feature bank.

Rendering Loss. $\mathcal { L } _ { \boldsymbol { r e n d e r } }$ optimizes the rendering quality of 4DGS by minimizing the visual discrepancy between the rendered images and the ground-truth observations.

![](images/ae5e85476ea18970d693bf7fa2ba1d888a8cf466084dbb8ed12122773cb67709.jpg)  
Phase-guided feature matching  
Figure 2: Training pipeline of Phase-DGS. Multi-view data provides camera parameters (view matrix, focal length, resolution) and an RGB image. The phase projector module predicts a phase value from the RGB image, replacing the timestamp input of the 4DGS backbone. The feature bank serves as a learnable reference repository that enforces cross-camera phase coherence, so that identical motion states share the same phase across all views. Training is guided by three losses: a Rendering loss between the rendered and input images, a monotonic loss, and a phase consistency loss.

## 3.5 Integration with Dynamic Gaussian Splatting

Existing 4DGS methods model the deformation of 3D Gaussians over time by taking the absolute timestamp t of each frame as input [35, 34]. However, as discussed above, absolute timestamps are inherently unreliable in real-world capture environments because of inter-camera synchronization errors, frame drops, and frame freezes.

To address this issue, we replace the timestamp t in 4DGS model $\mathcal { G }$ with the phase $p _ { i } ^ { f }$ extracted by Phase-DGS, as shown in Fig. 1. Since $p _ { i } ^ { f }$ is extracted independently from the image content of each camera, it naturally compensates for inter-camera synchronization errors, frame drops, and frame freezes without requiring any explicit timestamp correction. In other words, whereas conventional 4DGS optimizes Gaussians along the temporal axis, Phase-DGS optimizes Gaussians along the phase axis,

$$
\mathcal { G } ( \boldsymbol { p } _ { i } ^ { f } )  \mathcal { G } ( t ) .\tag{9}
$$

Note that the phase is designed as a one-dimensional scalar because it plays the same role as the timestamp t. Since existing 4DGS methods take a scalar timestamp as input [35, 34], the phase can be integrated in a plug-and-play manner without modifying the backbone. Furthermore, all objects within a frame are captured at the same instant, so a single phase per frame is sufficient to represent the temporal state of the scene regardless of the number of moving objects. The motion of each object, including complex motions and multi-object interactions, is still modeled by the deformation of the 4DGS backbone, as in the timestamp-based setting.

## 3.6 Training

Phase-DGS is trained end-to-end by jointly optimizing a plug-and-play 4DGS backbone as shown in Fig. 2 (e.g., RealTime-4DGS [35], FreeTimeGS [34]), the phase projector, and the feature bank under the combined loss in Eq. (3). The ViT encoder remains frozen throughout the training, while all other modules are updated by gradient descent. In each iteration, a frame $u _ { i } ^ { f }$ is sampled from the camera i and passed through the ViT encoder and the phase projector to produce the phase $p _ { i } ^ { f }$ . The estimated phase replaces the absolute timestamp t as the temporal coordinate for the 4DGS backbone, and all three losses are back-propagated simultaneously.

## 4 Experimental results

## 4.1 Implementation Details.

Phase Extraction Module. For the ViT encoder, we adopt DINOv2 [23] with the dinov2\_vitb14 configuration. The phase projector (see Section 3.2) MLP follows a $d  2 5 6  1 2 8  6 4  1$ architecture, in which each hidden layer is activated by GELU and the final output is normalized to [0, 1] by a Sigmoid activation.

4DGS Backbones. We primarily adopt FreeTimeGS [34], which we re-implement based on the original paper, and we additionally validate plug-and-play compatibility with RealTime-4DGS [35]. These two methods constitute the state of the art in 4DGS and represent the explicit and implicit paradigms of modeling Gaussian motion, respectively.

Training Details. Phase-DGS is trained with the Adam optimizer and a batch size of one camera per iteration. The shared hyperparameters across both backbones are a learning rate of $1 \times 1 0 ^ { - 4 }$ for the phase projector, a learning rate of $1 \times 1 0 ^ { - 3 }$ for the feature bank, $\lambda _ { r e n d e r } = 1 . 0$ , and $\lambda _ { p h a s e } = 0 . 1$ The backbone-specific configurations are as follows. FreeTimeGS is trained for 500,000 iterations with an upsampling factor $\bar { k } = 2$ and $\lambda _ { m o n o } = 0 . 1$ , while RealTime-4DGS is trained for 60,000 iterations with $k = 3$ and $\lambda _ { m o n o } = 0 . 0 1$ . Compared with the FreeTimeGS backbone alone, Phase-DGS requires approximately 30% more GPU memory and 2.1× longer training time per iteration, while the rendering cost at inference remains the same (see Section A).

Datasets. We evaluated Phase-DGS on two representative datasets for dynamic scene reconstruction, each featuring distinct camera configurations to ensure a comprehensive assessment. The Nvidia Dynamic Scene Dataset [41] consists of 12 front-facing Full HD cameras, and SelfCap [38] provides 24 convergent 4K cameras, with sequences of approximately 100 and 300 frames, respectively. For the Nvidia Dynamic Scene Dataset, we use three scenes: Jumping, Playground, and Umbrella. For SelfCap, we use Corgi, Yoga, and Bike. For both datasets, all images are downsampled to $9 6 0 \times 5 4 0$ . To validate robustness under various unsynchronized conditions, we evaluated on three data configurations. The Synced condition uses the provided dataset as is and assumes synchronized cameras. The Shifted condition introduces inter-camera temporal offsets of about 20–30 frames by randomly shifting the starting frame of each camera. The Dropped condition simulates severe realworld asynchrony with 10% random frame drops and 5–10 frames freeze, which vary independently across cameras.

Train and Test View Selection. Since Phase-DGS improves the optimization quality of the radiance field, we select a single camera located at the outermost spatial position in each scene as the test view. This view has the largest viewpoint difference from the train views, where artifacts such as spatial incoherence caused by inaccurate timestamps appear more prominently.

## 4.2 Baselines

We compare Phase-DGS to the following baselines. 4DGS w/o Sync trains the 4DGS backbone directly using absolute timestamps without any synchronization and serves as a lower-bound reference. VisualSync [16] performs video synchronization as a preprocessing step, after which the 4DGS backbone is trained on the aligned sequences. Sync-NeRF [12] jointly optimizes temporal offsets together with neural radiance field reconstruction in an end-to-end manner.

## 4.3 Quantitative Results

As shown in Tab. 1, the Synced condition achieves the highest performance in all datasets (all experimental results are shown in Tab. A4 and A5, and see evaluation detail in Section B). In contrast, the Shifted and Dropped conditions exhibit different tendencies depending on the dataset. A noteworthy observation is that RealTime-4DGS [35], which implicitly represents motion, is relatively more robust to unsynchronized conditions than FreeTimeGS [34], which explicitly represents motion. Due to this property, the proposed Phase-DGS shows only a marginal improvement in PSNR when combined with the RealTime-4DGS backbone. In contrast, when combined with the FreeTimeGS backbone, which uses an explicit motion representation, Phase-DGS achieves a substantial improvement in quantitative metrics. FreeTimeGS is therefore particularly sensitive to timestamp inaccuracy, and under such conditions the phase-based supervision of Phase-DGS achieves the greatest effect.

Table 1: Average quantitative comparison on the SelfCap and Nvidia Dynamic Scene datasets under synchronized (Synced) and unsynchronized (Shifted, Dropped) conditions. ↑: higher is better, ↓: lower is better. Green : best among methods sharing the same backbone and condition. Red: best across all methods under the same condition.
<table><tr><td rowspan="2">Condition</td><td rowspan="2">Backbone</td><td rowspan="2">Method</td><td colspan="3">SelfCap</td><td colspan="3">Nvidia</td></tr><tr><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td></tr><tr><td rowspan="5">Synced</td><td>K-Planes</td><td>Sync-NeRF</td><td></td><td>=</td><td>1</td><td>28.889</td><td>0.942</td><td>0.048</td></tr><tr><td>RealTime-4DGS</td><td></td><td>30.858</td><td>0.970</td><td>0.030</td><td>29.735</td><td>0.952</td><td>0.045</td></tr><tr><td rowspan="2"></td><td>Ours</td><td>32.145</td><td>0.973</td><td>0.026</td><td>28.938</td><td>0.946</td><td>0.061</td></tr><tr><td></td><td>34.181</td><td>0.984</td><td>0.010</td><td>31.167</td><td>0.963</td><td>0.022</td></tr><tr><td rowspan="2">FreeTimeGS</td><td>VisualSync</td><td>33.446</td><td>0.983</td><td>0.013</td><td>30.429</td><td>0.958</td><td>0.025</td></tr><tr><td>Ours</td><td>34.727</td><td>0.985</td><td>0.012</td><td>30.416</td><td>0.959</td><td>0.030</td></tr><tr><td rowspan="5">Shifted</td><td>K-Planes</td><td>Sync-NeRF</td><td>=</td><td>=</td><td>1</td><td>28.492</td><td>0.940</td><td>0.049</td></tr><tr><td rowspan="2">RealTime-4DGS</td><td></td><td>27.614</td><td>0.956</td><td>0.039</td><td>23.083</td><td>0.915</td><td>0.084</td></tr><tr><td>Ours</td><td>28.288</td><td>0.961</td><td>0.037</td><td>26.655</td><td>0.937</td><td>0.064</td></tr><tr><td rowspan="2">FreeTimeGS</td><td></td><td>27.170</td><td>0.955</td><td>0.026</td><td>23.785</td><td>0.918</td><td>0.068</td></tr><tr><td>VisualSync</td><td>29.611</td><td>0.967</td><td>0.018</td><td>27.958</td><td>0.950</td><td>0.033</td></tr><tr><td rowspan="6">Dropped</td><td>K-Planes</td><td>Ours</td><td>32.947</td><td>0.981</td><td>0.014</td><td>29.836</td><td>0.956</td><td>0.037</td></tr><tr><td>RealTime-4DGS</td><td>Sync-NeRF</td><td></td><td></td><td></td><td>26.683</td><td>0.929</td><td>0.064</td></tr><tr><td rowspan="2"></td><td>Ours</td><td>26.679 29.638</td><td>0.959 0.966</td><td>0.036 0.032</td><td>25.547 28.263</td><td>0.931 0.944</td><td>0.068</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td>0.056</td></tr><tr><td rowspan="2">FreeTimeGS</td><td></td><td>27.894 29.739</td><td>0.962</td><td>0.025 0.019</td><td>25.313 28.057</td><td>0.929 0.943</td><td>0.055</td></tr><tr><td>VisualSync Ours</td><td>34.379</td><td>0.972 0.984</td><td>0.012</td><td>30.665</td><td>0.961</td><td>0.039 0.030</td></tr></table>

Since Sync-NeRF [12] is retrained under capture conditions that differ from those in the original paper, both the COLMAP-based pose initialization and hyperparameter settings additionally affect the reconstruction quality under all conditions. In particular, the SelfCap dataset [38] proves incompatible with the default configuration of Sync-NeRF, which was originally designed for forward-facing captures such as the Nvidia Dynamic Scene dataset [41], resulting in notably degraded reconstruction quality. Furthermore, the Dropped condition falls outside the design scope of Sync-NeRF, which assumes only real-valued sub-frame temporal offsets. Consequently, Sync-NeRF also yields poor performance under the Dropped condition.

VisualSync [16] successfully corrects synchronization errors before training in the Shifted conditions, yielding reconstruction quality comparable to that of the backbone under the Synced condition. However, the Dropped condition disrupts the temporal consistency of feature correspondences, preventing VisualSync from reliably estimating epipolar geometry. In particular, even under the Synced condition, VisualSync introduces an unexpected frame offset of approximately 3% of the total frame count after synchronization, which decreases the reconstruction quality.

Under the Synced condition, Phase-DGS improves the average PSNR on SelfCap by 0.55 dB and 1.29 dB with the FreeTimeGS and RealTime-4DGS backbones, respectively, while it shows a slightly lower PSNR on the Nvidia Dynamic Scene dataset. In SelfCap, the convergent multi-camera setup introduces greater viewpoint diversity, making spatiotemporal coherence a more critical factor for reconstruction quality. As a result, accurate synchronization leads to a more pronounced improvement in rendering quality. In contrast, in the Nvidia Dynamic Scene dataset, the simpler forward-facing configuration allows the backbone to converge quickly in terms of image quality, whereas Phase-DGS requires additional iterations for phase learning within the same training budget, resulting in a slightly lower PSNR than the backbone under the Synced condition. We note that Phase-DGS is designed to provide robustness to synchronization errors rather than to improve reconstruction under perfect synchronization. Since the synchronization status of real-world captures is often unknown, it is noteworthy that Phase-DGS remains within 0.8 dB of the backbone under the Synced condition, while improving the PSNR by up to 6.5 dB under the Dropped condition (SelfCap, FreeTimeGS).

+FreeTimeGS

Ours(RealTime-4DGS)

Sync-NeRF  
FreeTimeGS  
(FreeTimeGS)  
GT  
![](images/07d7425c4a602a5930192e808a2c0507564f24ba3d8a58708cff5ed8b5c02949.jpg)  
Figure 3: Qualitative comparison under the synchronized condition (Synced) and the unsynchronized condition (Shifted and Dropped) on Nvidia Dynamic Scene dataset and SelfCap datasets. Phase-DGS consistently produces cleaner renderings while baselines exhibit ghosting and geometric distortion.

![](images/381b70566929a686e35b07c914eb94c4e96c5c78f9617dfc8650cf70ad28fb0a.jpg)  
Figure 4: Qualitative comparison with enlarged views under the Synced condition. While baseline models exhibit ghosting and geometric distortion, Phase-DGS consistently reduces artifacts in rendered images. →: visual artifact.

## 4.4 Qualitative Results

As shown in Fig. 3, Phase-DGS with the FreeTimeGS [34] backbone achieves the best visual quality among all the methods compared. This qualitative comparison reveals perceptual differences that quantitative metrics alone cannot fully capture. For the RealTime-4DGS [35] backbone, as an additional discussion , the difference in PSNR before and after applying Phase-DGS is small. However, the rendering results show that the Gaussians in dynamic regions are accurately synchronized and converge to a single spatial point, and as a result the shape of dynamic objects appears much sharper than in the baseline. Since the phase estimated by the phase projector is gradually reflected throughout the training process, the overall PSNR of the scene shows only a limited change, while the Gaussian alignment in dynamic regions steadily improves. These results demonstrate that Phase-DGS effectively learns accurate phase information for each frame, precisely defining the three-dimensional position of each Gaussian and enabling cleaner renderings across both backbones.

Fig. 4 presents enlarged views for a more detailed comparison, where all outputs are rendered from models trained under the Synced condition. Comparing Phase-DGS against the FreeTimeGS backbone, we show pairs of frames from the same scene to highlight inter-frame differences. Although Phase-DGS under the Synced condition yields a lower PSNR than the FreeTimeGS baseline in the Jumping scene, the rendered images reveal noticeably fewer visual artifacts, a trend that is consistently observed in the Playground scene as well. We attribute this to the fact that Phase-DGS learns a radiance field with higher spatiotemporal coherence, enabling more geometrically consistent reconstructions even when quantitative metrics do not fully reflect this improvement. A notable observation is that even on datasets provided as synchronized, the backbone models without Phase-DGS still exhibit subtle rendering artifacts, empirically supporting the claim that even sub-frame-level timing disparities can measurably affect reconstruction quality.

## 4.5 Ablation Study

We conduct ablation studies on the Jumping scene of the Nvidia Dynamic Scene Dataset [41] and the Bike scene of SelfCap [38] under the Dropped condition to analyze the contribution of each component in Phase-DGS with FreeTimeGS [34] backbone.

Monotonic Loss. Without this loss, the phase is estimated in a temporally disordered manner, causing the beginning and end of the sequence to be assigned phases without consistent ordering. (see Section 3.4)

Feature Bank. When the feature bank is initialized but not updated during training, the stored features remain fixed and the accuracy of phase estimation drops substantially (see Section 3.3). This shows that the learnable feature bank plays an essential role by providing a sufficient reference over the entire sequence of each camera, which stabilizes the phase estimation process.

Upsampling Factor k. We vary the upsampling factor k of the feature bank and analyze its effect on performance (see Section 3.3). When k = 1, the feature bank operates at the original frame rate and cannot capture sub-frame asynchrony. At k = 2, the finer temporal resolution yields the best trade-off between phase accuracy and computational cost. Increasing k to 4 does not improve performance, as the enlarged feature bank requires more training iterations to converge within the fixed training budget, leading to a slight PSNR degradation.

ViT Backbone. We compare DINOv2-S, -B, and -L [23] depending on the number of parameters of the ViT encoder. The self-supervised DINOv2 learns detailed visual representations, and the performance gap between different sizes of DINOv2 models is small. This suggests that phase estimation captures the overall progression of motion, for which the global semantic representation learned by the base model is already sufficient.

ROI Crop. We compare two input configurations. The first crops the input image to the motionintensive ROI, while the second forcibly resizes the full image to 224 × 224 for the visual transformer (see Sections 3.2 and 3.6) [23]. The ROI-based configuration achieves slightly better performance, but the difference is less than 0.2 dB in both scenes, which shows that Phase-DGS also operates on full images without the ROI crop. The ROI crop is expected to be more effective when the dynamic objects occupy only a small region of the image, since resizing the full image to 224 × 224 can remove fine motion cues.

Table 2: Ablation study on the Jumping scene (Nvidia Dynamic Scene dataset) and Bike scene (SelfCap dataset) under the Dropped condition. FreeTimeGS is used as the DGS backbone.
<table><tr><td rowspan="2">Method</td><td colspan="3">Jumping</td><td colspan="3">Bike</td></tr><tr><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td></tr><tr><td>Ours (default)</td><td>29.386</td><td>0.942</td><td>0.037</td><td>31.212</td><td>0.977</td><td>0.016</td></tr><tr><td>w/o Monotonic Loss</td><td>24.421</td><td>0.887</td><td>0.073</td><td>25.903</td><td>0.944</td><td>0.039</td></tr><tr><td>w/o Bank Learning</td><td>24.794</td><td>0.897</td><td>0.070</td><td>27.642</td><td>0.958</td><td>0.026</td></tr><tr><td>k = 1</td><td>29.425</td><td>0.942</td><td>0.038</td><td>31.130</td><td>0.977</td><td>0.015</td></tr><tr><td>k = 4</td><td>29.006</td><td>0.938</td><td>0.038</td><td>31.101</td><td>0.976</td><td>0.015</td></tr><tr><td>DINOv2-S</td><td>29.034</td><td>0.939</td><td>0.039</td><td>30.932</td><td>0.976</td><td>0.016</td></tr><tr><td>DINOv2-L</td><td>29.648</td><td>0.943</td><td>0.037</td><td>31.411</td><td>0.978</td><td>0.015</td></tr><tr><td>w/o ROI Crop</td><td>29.231</td><td>0.939</td><td>0.039</td><td>31.16</td><td>0.977</td><td>0.017</td></tr></table>

## 5 Conclusion

We presented Phase-DGS, a self-supervised framework for robust dynamic scene reconstruction under unsynchronized multi-camera conditions. By replacing absolute timestamps with per-frame phases extracted directly from images, Phase-DGS eliminates the reliance on hardware synchronization and enables robust 4DGS training even under severe frame drops and temporal offsets. Experiments on Nvidia Dynamic Scene Dataset and SelfCap demonstrate that Phase-DGS consistently outperforms existing synchronization baselines under unsynchronized conditions, achieving competitive performance even under the synchronized setting. Furthermore, plug-and-play compatibility with multiple 4DGS backbones highlights the generalizability of our approach. The ability to train without precise timestamps suggests the potential to extend Phase-DGS to archival or previously captured footage, where hardware synchronization cannot be applied retroactively. We hope Phase-DGS serves as a practical and scalable solution for real-world multi-camera capture systems.

Limitations and Future Work. The reconstruction quality of Phase-DGS depends on the underlying 4DGS backbone, and the way timestamps are processed inside the network also affects performance. In addition, Phase-DGS currently struggles with scenes containing periodic motions, such as the Balloon1 sequence, where visually similar frames make accurate phase assignment difficult. We plan to address these issues through backbone-agnostic training strategies and inter-frame constraints that encourage smooth and consistent phase transitions.

## Acknowledgements

This work was supported by the IITP grant funded by the Korea government (MSIT): Development of immersive video spatial computing technology for ultra-realistic metaverse services (No.2022-0- 00022, RS-2022-II220022), and by the IITP grant (RS-2021-II211343, AI Graduate School Program (Seoul National University)), funded by the Korean government (MSIT).

## Disclosures

The authors declare no competing interests.

## References

[1] Sameer Ansari, Neal Wadhwa, Rahul Garg, and Jiawen Chen. Wireless software synchronization of multiple distributed cameras. In 2019 IEEE International Conference on Computational Photography (ICCP), pages 1–9. IEEE, 2019.

[2] Nicholas J Bryan, Paris Smaragdis, and Gautham J Mysore. Clustering and synchronizing multi-camera video via landmark cross-correlation. In 2012 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 2389–2392. IEEE, 2012.

[3] Changwoon Choi, Jeongjun Kim, Geonho Cha, Minkwan Kim, Dongyoon Wee, and Young Min Kim. Humans as a calibration pattern: Dynamic 3d scene reconstruction from unsynchro-

nized and uncalibrated videos. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 6598–6608, 2025.

[4] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020.

[5] Ahmed Elhayek, Carsten Stoll, Kwang In Kim, H-P Seidel, and Christian Theobalt. Featurebased multi-video synchronization with subframe accuracy. In Joint DAGM (German Associationfor Pattern Recognition) and OAGM Symposium, pages 266–275. Springer, 2012.

[6] Sara Fridovich-Keil, Alex Yu, Matthew Tancik, Qinhong Chen, Benjamin Recht, and Angjoo Kanazawa. Plenoxels: Radiance fields without neural networks. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pages 5501–5510, 2022.

[7] Huayu Huang, Yang Shang, Banglei Guan, and Qifeng Yu. 3d trajectory reconstruction of moving points based on asynchronous cameras. Acta Mechanica Sinica, 41(7):425322, 2025.

[8] Vladimir Iashin, Weidi Xie, Esa Rahtu, and Andrew Zisserman. Synchformer: Efficient synchronization from sparse cues. In ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 5325–5329. IEEE, 2024.

[9] IEEE. IEEE standard for a precision clock synchronization protocol for networked measurement and control systems, 2019.

[10] Rishit Javia, Falak Shah, and Shivam Dave. Posesync: Robust pose based video synchronization. arXiv preprint arXiv:2308.12600, 2023.

[11] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, George Drettakis, et al. 3d gaussian splatting for real-time radiance field rendering. ACM Trans. Graph., 42(4):139–1, 2023.

[12] Seoha Kim, Jeongmin Bae, Youngsik Yun, Hahyun Lee, Gun Bang, and Youngjung Uh. Syncnerf: Generalizing dynamic nerfs to unsynchronized videos. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pages 2777–2785, 2024.

[13] Jonas Kulhanek, Songyou Peng, Zuzana Kukelova, Marc Pollefeys, and Torsten Sattler. Wildgaussians: 3d gaussian splatting in the wild. arXiv preprint arXiv:2407.08447, 2024.

[14] Yonghan Lee, Tsung-Wei Huang, Shiv Gehlot, Jaehoon Choi, Guan-Ming Su, and Dinesh Manocha. Synctrack4d: Cross-video motion alignment and video synchronization for multivideo 4d gaussian splatting. arXiv preprint arXiv:2512.04315, 2025.

[15] Tianye Li, Mira Slavcheva, Michael Zollhoefer, Simon Green, Christoph Lassner, Changil Kim, Tanner Schmidt, Steven Lovegrove, Michael Goesele, Richard Newcombe, et al. Neural 3d video synthesis from multi-view video. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5521–5531, 2022.

[16] Shaowei Liu, David Yifan Yao, Saurabh Gupta, and Shenlong Wang. Visual sync: Multi-camera synchronization via cross-view object motion. arXiv preprint arXiv:2512.02017, 2025.

[17] Cheng Lu and Mrinal Mandal. A robust technique for motion-based video sequences temporal alignment. IEEE Transactions on Multimedia, 15(1):70–82, 2012.

[18] Tao Lu, Mulin Yu, Linning Xu, Yuanbo Xiangli, Limin Wang, Dahua Lin, and Bo Dai. Scaffoldgs: Structured 3d gaussians for view-adaptive rendering. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 20654–20664, 2024.

[19] Ricardo Martin-Brualla, Noha Radwan, Mehdi SM Sajjadi, Jonathan T Barron, Alexey Dosovitskiy, and Daniel Duckworth. Nerf in the wild: Neural radiance fields for unconstrained photo collections. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 7210–7219, 2021.

[20] Jaro Meyer, Frédéric Giraud, Joschua Wüthrich, Marc Pollefeys, Philipp Fürnstahl, and Lilian Calvet. Rocsync: Millisecond-accurate temporal synchronization for heterogeneous camera systems. Sensors, 26(3):1036, 2026.

[21] Thomas Müller, Alex Evans, Christoph Schied, and Alexander Keller. Instant neural graphics primitives with a multiresolution hash encoding. ACM transactions on graphics (TOG), 41(4): 1–15, 2022.

[22] Avihai Naaman, Ron Shapira Weber, and Oren Freifeld. Synchronization of multiple videos. In Proceedings of the IEEE/CVF international conference on computer vision, pages 12514–12523, 2025.

[23] Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023.

[24] Albert Pumarola, Enric Corona, Gerard Pons-Moll, and Francesc Moreno-Noguer. D-nerf: Neural radiance fields for dynamic scenes. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10318–10327, 2021.

[25] Dmitry Pundik and Yael Moses. Video synchronization using temporal signals from epipolar lines. In European conference on computer vision, pages 15–28. Springer, 2010.

[26] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman Rädle, Chloe Rolland, Laura Gustafson, et al. Sam 2: Segment anything in images and videos. arXiv preprint arXiv:2408.00714, 2024.

[27] RED.COM, LLC. Genlock. https://docs.red.com/955-0156/ EPICSCARLETOperationGuide/en-us/Content/8\_TC\_Gen\_Multi/Genlock.htm, 2019. EPIC/SCARLET Operation Guide, 955-0156\_V6.3, REV-N.

[28] Prarthana Shrestha, Mauro Barbieri, Hans Weda, and Dragan Sekulovski. Synchronization of multiple camera videos using audio-visual features. IEEE Transactions on Multimedia, 12(1): 79–92, 2009.

[29] Prarthana Shrstha, Mauro Barbieri, and Hans Weda. Synchronization of multi-camera video recordings based on audio. In Proceedings of the 15th ACM international conference on Multimedia, pages 545–548, 2007.

[30] Society of Motion Picture and Television Engineers. SMPTE ST 274:2008 – Television – 1920 × 1080 Image Sample Structure, Digital Representation and Digital Timing Reference Sequences for Multiple Picture Rates. SMPTE Standard, jan 2008. Revision of SMPTE 274M-2005.

[31] Society of Motion Picture and Television Engineers. ST 296:2011 – SMPTE Standard – 1280 × 720 Progressive Image Sample Structure — Analog and Digital Representation and Analog Interface. SMPTE Standard, oct 2011. Revision of SMPTE 296M-2001.

[32] Minh Vo, Srinivasa G Narasimhan, and Yaser Sheikh. Spatiotemporal bundle adjustment for dynamic 3d reconstruction. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 1710–1718, 2016.

[33] Oliver Wang, Christopher Schroers, Henning Zimmer, Markus Gross, and Alexander Sorkine-Hornung. Videosnapping: Interactive synchronization of multiple videos. ACM Transactions on Graphics (TOG), 33(4):1–10, 2014.

[34] Yifan Wang, Peishan Yang, Zhen Xu, Jiaming Sun, Zhanhua Zhang, Yong Chen, Hujun Bao, Sida Peng, and Xiaowei Zhou. Freetimegs: Free gaussian primitives at anytime anywhere for dynamic scene reconstruction. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21750–21760, 2025.

[35] Guanjun Wu, Taoran Yi, Jiemin Fang, Lingxi Xie, Xiaopeng Zhang, Wei Wei, Wenyu Liu, Qi Tian, and Xinggang Wang. 4d gaussian splatting for real-time dynamic scene rendering. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 20310–20320, 2024.

[36] Xinyi Wu, Zhenyao Wu, Yujun Zhang, Lili Ju, and Song Wang. Multi-video temporal synchronization by matching pose features of shared moving subjects. In Proceedings of the IEEE/CVF International Conference on Computer Vision Workshops, pages 0–0, 2019.

[37] Qiangeng Xu, Zexiang Xu, Julien Philip, Sai Bi, Zhixin Shu, Kalyan Sunkavalli, and Ulrich Neumann. Point-nerf: Point-based neural radiance fields. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5438–5448, 2022.

[38] Zhen Xu, Yinghao Xu, Zhiyuan Yu, Sida Peng, Jiaming Sun, Hujun Bao, and Xiaowei Zhou. Representing long volumetric video with temporal gaussian hierarchy. ACM Transactions on Graphics (TOG), 43(6):1–18, 2024.

[39] Zhixin Xu, Hengyu Zhou, Yuan Liu, Wenhan Xue, Hao Pan, Wenping Wang, and Bin Wang. Dynamic gaussian scene reconstruction from unsynchronized videos. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, pages 11469–11477, 2026.

[40] Liqiang Yin, Ruize Han, Wei Feng, and Song Wang. Self-supervised human pose based multicamera video synchronization. In Proceedings of the 30th ACM International Conference on Multimedia, pages 1739–1748, 2022.

[41] Jae Shin Yoon, Kihwan Kim, Orazio Gallo, Hyun Soo Park, and Jan Kautz. Novel view synthesis of dynamic scenes with globally coherent depths from a monocular camera. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5336–5345, 2020.

[42] Zehao Yu, Anpei Chen, Binbin Huang, Torsten Sattler, and Andreas Geiger. Mip-splatting: Alias-free 3d gaussian splatting. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 19447–19456, 2024.

![](images/4a4f2e7aaa8a576ff9e6fb88268ae910dedf761e405a44ee539e661f57e0c52a.jpg)  
Figure A1: Multi-camera frames at the same frame index from the Jumping sequence (Nvidia Dynamic Scene Dataset) under the Dropped condition. Despite sharing the same frame index, each camera captures a distinct motion state, illustrating the severity of temporal misalignment introduced by frame drops and freezes.

## A Additional Implementation Details

Data Generation. To assess robustness under various real-world capture conditions, we construct three data configurations with increasing levels of temporal asynchrony. The Synced condition uses the original dataset without modification, serving as the upper-bound reference under ideal synchronization. The Shifted condition introduces inter-camera temporal offsets of 20–30 frames by randomly shifting the starting frame of each camera, simulating clock drift or delayed recording onset. The Dropped condition simulates severe asynchrony by applying 10% random frame drops and 5–10 frame freezes that vary independently across cameras, reflecting realistic packet loss and buffer failures in large-scale capture systems. As illustrated in Fig. A1, frames captured at the same frame index across cameras under the Dropped condition exhibit pronounced motion discrepancies, highlighting the severity of temporal misalignment introduced by our simulation. Additional visualizations are provided in the supplementary video.

ROI Cropping. Since this work focuses on the synchronization problem, we implement a manual ROI extraction tool throughout the experiments with the help of an LLM. The ROI extraction tool used in our experiments is implemented as a simple interface that supports manual bounding box selection as shown in Fig. A2. For each scene, the ROI is specified to cover the region where motion is most concentrated, based on a single representative frame from the scene. The tool also provides options to use the central square region of the frame or to keep the full frame without cropping. Different cameras in the same scene can have bounding boxes of different sizes and positions, and the location of each bounding box can be specified freely. Note that the ROI crop is an optional preprocessing step and is not required by Phase-DGS (see Section 4.5). The ROI extraction process can be further automated using algorithms such as optical flow, motion detection, or video segmentation models such as SAM2 [26]. For reproducibility, we will release the ROI coordinates and segmentation masks used in our experiments together with the code.

Feature bank Initialization. The feature bank is initialized with features extracted from all frames of all cameras. When the upsampling factor is k > 1, the feature entries at upsampled positions that do not correspond to original frames are filled by linear interpolation between neighboring features. Initializing the feature bank in this way provides meaningful feature values from the beginning of training, which shortens the convergence time.

Compute Resources. All experiments in this work were conducted on a single NVIDIA RTX 5090 GPU with Python 3.12, PyTorch 2.8.0, and CUDA 12.8. For the RealTime-4DGS [35] backbone, 100K iterations take approximately one hour. For the FreeTimeGS [34] backbone, 500K iterations take approximately 2.5 hours. The GPU memory usage depends on the scene, with the RealTime-4DGS backbone using approximately 12.4 GB and the FreeTimeGS backbone using approximately 4 GB. The full set of reported experiments covers 6 scenes (3 from the Nvidia Dynamic Scene

![](images/0e6bdf2e70855bbdaa32c1fe0666cdf195790949ba1a79ee52bc53fda1e57bea.jpg)  
Figure A2: Interface of the ROI extraction tool used in our experiments. The tool supports manual bounding box selection, with options to use the central square region or the full frame. Different cameras in the same scene can have independent bounding boxes.

Dataset and 3 from SelfCap) across 3 synchronization conditions (Synced, Shifted, Dropped) and two 4DGS backbones (RealTime-4DGS and FreeTimeGS). The reported computation does not include preliminary experiments and failed runs.

Training Overhead. To quantify the additional cost of phase learning, we measure the GPU memory usage and training throughput with the FreeTimeGS [34] backbone on a multi-view sequence captured with 23 cameras (50 frames), as shown in Tab. A1. Note that the absolute values differ from those reported above because the dataset is different. Phase-DGS requires an additional 514 MB (+30.5%) of GPU memory, which mainly comes from the feature bank, the phase projector, and the ViT encoder. Since the ViT encoder remains frozen, back-propagation through the encoder is not required, and only the lightweight phase projector and the feature bank are additionally optimized. The training throughput decreases from 148 to 70 iterations per second (2.11× slower). This overhead applies only to training, since the ViT encoder, the phase projector, and the feature bank are not used at inference, where the backbone is rendered directly at a given phase.

Table A1: Training overhead of Phase-DGS with the FreeTimeGS backbone on a 23-camera, 50- frame sequence, measured on a single NVIDIA RTX 5090 GPU.
<table><tr><td>Configuration</td><td>GPU Memory</td><td>Training Throughput</td></tr><tr><td>FreeTimeGS (backbone only)</td><td>1686 MB</td><td>148 it/s</td></tr><tr><td>Phase-DGS (FreeTimeGS)</td><td>2200 MB (+514 MB)</td><td>70 it/s (2.11× slower)</td></tr></table>

![](images/8939c30db46179156307b83621bff79a1a881007f19c591d0807162271310fdb.jpg)  
Figure A3: Phase estimation results on the Jumping scene under the Shifted (top) and Dropped (bottom) conditions. (Left) Ground-truth frame correspondences between each camera and the source video. (Center, Right) Per-frame phase estimates produced by Phase-DGS with the RealTime-4DGS and FreeTimeGS backbones, respectively.

## B Evaluation Protocol

Phase Reference Computation. Phase-DGS does not estimate an absolute timestamp from an image but instead extracts a relative phase value. As shown in Fig. A3, the phase values produced by Phase-DGS for frames within a single scene follow a non-linear curve. It is noted that the phase curves of the train views are regularized by the monotonic loss so that they remain monotonic, while the test view (Cam 0) is not subject to this loss and thus exhibits a slightly different phase curve, although it follows a similar monotonic trend. As a result, the phase stamp of the Phase-DGS, which refers to the ordering of frames along the phase axis and plays a role similar to a time stamp, does not provide absolute temporal information. To quantitatively evaluate Phase-DGS, we therefore need to define a reference phase stamp for each scene. We should construct the reference phase stamp for the test view. The unsynchronized dataset used in our experiments provides the correspondence between each view’s frame index and the index in the original source video. Using this correspondence, we can identify which frame of the source video each test view frame is sampled from, and then find the train view frames that are sampled from the same source index. The phase reference for each test view frame is obtained by averaging the phases of the corresponding train view frames. The overall procedure is summarized in Algorithm 1. Since this protocol does not recover the exact absolute phase stamp but only estimates a reference phase in an indirect manner, the actual time of a test view frame may not perfectly align with the time implied by the estimated reference phase. This misalignment can introduce minor inaccuracies in the reported quantitative metrics. We note that this protocol is used only for evaluation purposes, and the exact procedure for constructing the phase-line may vary depending on the application or the dataset. However, this correspondence is a form of ground-truth temporal alignment that is available only in our simulated datasets, and it is not used for the baselines. The reported metrics therefore reflect the reconstruction quality of Phase-DGS given an accurate temporal anchor, which is stronger than the anchor available in practice, where the global phase stamp is defined without ground truth (see Section C).

Evaluation. To evaluate the performance of the trained GS model, we first compute the phase reference $\tilde { p } _ { t e s t } ^ { f }$ for each test view frame using Algorithm 1. Here, the reference phase $\tilde { p } _ { t e s t } ^ { f }$ from

Algorithm 1 contains estimation noise, which can lower the reported metrics even when the underlying reconstruction is accurate. To account for this, we adopt a strategy similar to the test-time optimization used in NeRF-W [19] and WildGaussians [13]. With the trained GS model fully frozen, we search within ±δ of $\tilde { p } _ { t e s t } ^ { f }$ for the phase $\tilde { p ^ { \prime } } _ { t e s t } ^ { f }$ that yields the highest PSNR against the ground-truth frame. Because the search window is small and phase is a single scalar, this procedure corrects for estimator noise without introducing meaningful bias toward the ground truth. In the experiments, we set $\delta = 0 . 0 5$ , which corresponds to a window size of approximately 10 to 30 frames relative to the ground truth. The final metrics are computed from images rendered at these refined phases. To further assess the effect of synchronization on rendering quality, we compute metrics exclusively over dynamic regions. As shown in Fig. A4, segmentation masks are extracted from the ground-truth frames, and PSNR, SSIM, and LPIPS are evaluated only within these masked areas. Since dynamic regions are most sensitive to temporal misalignment, restricting the evaluation to these areas allows for a more precise examination of whether the phase reference is accurately synchronized with the ground truth.

Search Window Sensitivity. To verify that the test-time phase search only compensates for the estimation noise, we vary the search window W and report the PSNR with the FreeTimeGS [34] backbone in Tab. A2. $W { = } 0$ denotes rendering directly at the reference phase $\tilde { p } _ { \mathrm { t e s t } } ^ { f }$ without any search, and $W { = } 1 0$ is the default setting used for all reported results. Increasing the window from 10 to 50 does not change the PSNR in any setting, which indicates that the search converges within a small neighborhood of the reference phase. Without the search $( W { = } 0 )$ , the PSNR changes by less than 0.2 dB in five of the six settings, and the largest drop of 0.75 dB is observed in the Shifted condition of the Jumping scene. Even at $W { = } 0$ , Phase-DGS outperforms the FreeTimeGS baseline by 4.8–7.0 dB under all Shifted and Dropped settings (see Tab. A4 and A5), which shows that the performance gains do not result from the test-time search.

Table A2: Sensitivity of PSNR to the search window W of the test-time phase search with the FreeTimeGS backbone. W=0 denotes rendering at the reference phase without search, and $W { = } 1 0$ is the default setting.
<table><tr><td>Condition</td><td>Scene</td><td> $W { = } 0$ </td><td> $W { = } 5$ </td><td> $W { = } 1 0$  (default)</td><td> $W { = } 5 0$ </td></tr><tr><td>Synced</td><td>Bike Jumping</td><td>31.355 28.851</td><td>31.376 28.851</td><td>31.376 28.851</td><td>31.376 28.851</td></tr><tr><td>Shifted</td><td>Bike Jumping</td><td>31.110 27.354</td><td>31.299 28.090</td><td>31.299 28.103</td><td>31.299 28.103</td></tr><tr><td>Dropped</td><td>Bike Jumping</td><td>31.131 29.386</td><td>31.217 29.386</td><td>31.217 29.386</td><td>31.217 29.386</td></tr></table>

![](images/82208f49e2ce1051d315609225faf3d471c23ea8c39206d8116a1e9916908415.jpg)  
GT image

![](images/7eee57a80b3301d171b073fff04d9cc6c5e2ab9885d631bcbe466d4b93e04c2b.jpg)  
Mask based on GT

![](images/3ba0c966959de9ab7f894f2c8cf2e2d265f93a8111ba120e5b5967b80627df99.jpg)  
Masked GT

![](images/710ace92b83edb2b1cb1bfebf8867ebec595e77668b3e607693aa696bff1a709.jpg)  
Rendered image(RealTime-4DGS)

![](images/396cd6680d0f991949e0beea2f40f6e520b115ac8dc71498fc995b93530da00a.jpg)  
Mask based on GT

![](images/6809bab25548c3f0b3ca1cfe726a5736e4855a6250616735506e0e9b8cdc45ff.jpg)  
Masked render image

Figure A4: To assess rendering quality in dynamic regions, we generate segmentation masks from ground-truth images and compute metrics exclusively over the masked areas. The mask is generated by object segmentation using SAM2 [26], and a small padding margin is applied to the mask boundaries to assess the reconstruction quality of regions surrounding dynamic objects.  
Algorithm 1 Phase Reference Computation for Test Views   
Require: Train view images $\{ u _ { i } ^ { f } \}$ with source indices $\{ s _ { i } ( f ) \}$ , test view source indices $\{ s _ { \mathrm { t e s t } } ( f ) \}$   
Trained Phase projector Φ   
Ensure: Reference phase $\tilde { p } _ { \mathrm { t e s t } } ^ { f }$ for each test view frame   
1: for each test view frame index $f = 1 , \ldots , F$ do   
2: $\tau  \emptyset$ ▷ Collect matching train view frames   
3: for each train view $i = 1 , \ldots , N$ do   
4: for each frame index $f ^ { \prime } = 1 , \ldots , F$ do   
5: if $s _ { i } ( f ^ { \prime } ) = s _ { \mathrm { t e s t } } ( f )$ then   
6: $\ddot { \mathcal { T } }  \mathcal { T } \cup \{ ( i , f ^ { \prime } ) \}$   
7: end if   
8: end for   
9: end for   
10: $p _ { \mathrm { s u m } } \gets 0$   
11: for each $( i , f ^ { \prime } ) \in \mathcal { T }$ do   
12: $p _ { \mathrm { s u m } }  p _ { \mathrm { s u m } } + \Phi ( u _ { i } ^ { f ^ { \prime } } )$ ▷ Extract phase from each matching train frame   
13: end for   
14: $\tilde { p } _ { \mathrm { t e s t } } ^ { f }  p _ { \mathrm { s u m } } / | \mathcal { T } |$   
15: end for   
16: return $\{ \tilde { p } _ { \mathrm { t e s t } } ^ { f } \} _ { f = 1 } ^ { F }$

![](images/f6b4d93f1d4b4fb57b75f456fea9d2608ce2020ea34758e079d57417734b0056.jpg)  
Figure A5: Per-frame phase estimates on the Bike scene (SelfCap) under the Dropped condition. (Top left) Ground-truth frame correspondences. Phase estimation results under (Top right) default setting, (Bottom left) without monotonic loss. (Bottom right) without feature bank learning.

## C Additional Discussion

Backbone-Dependent Reconstruction Quality. As briefly noted in Section 5, the performance of Phase-DGS is influenced by the visual representation capacity of the underlying 4DGS backbone. We attribute this to the relatively large weight assigned to the rendering loss during training, which makes the rendering quality of the backbone an important factor for stable phase learning. We provide a more detailed observation here. The RealTime-4DGS [35] backbone began to overfit to the training scene after approximately 60,000 iterations, where the test-view PSNR dropped sharply and, at the same time, the consistency of the estimated phase values also deteriorated. In contrast, the FreeTimeGS [34] backbone maintained stable visual quality up to 500,000 iterations, and the corresponding phase estimates remained close to the ground-truth phase profile throughout training.

Global phase estimation in Practice. The phase projector module of Phase-DGS estimates the phase independently for each frame of each camera view. As shown in our experiments, it is not possible to define an accurate global time stamp under the Dropped condition. Since Phase-DGS can only extract relative phase stamps, the global phase stamp must be defined externally. We propose two approaches for this. The first uses the average of the phase stamps estimated across all camera views as the global phase stamp. The second uses the phase estimated from a single reference camera view as the global phase stamp.

## D Ablation Detail

As shown in the ablation study, the learnable feature bank and Monotonic Loss are the two most critical components for training Phase-DGS. Removing either of them prevents accurate phase estimation, making the Gaussian Splatting training infeasible. Figure A5 shows the per-frame predicted phase of each camera view, produced by the fully trained phase projector module. In the default setting, the relative phase values across frames follow the ground truth labels. Without Monotonic Loss, phase differences across views appear, but the phase within a single view barely changes, making training collapse. Without feature bank learning, all train views produce the same linear phase output, which means the phase projector module provides no useful information.

## E Numerical Stability

To verify the stability of Phase-DGS across different runs, we train the model four times from scratch, including COLMAP initialization, on the Corgi scene under the Dropped condition. As reported in Tab. A3, Phase-DGS achieves consistent performance across all runs, with a PSNR standard deviation of 0.466 dB, demonstrating reliable convergence regardless of initialization.

Table A3: Numerical stability of Phase-DGS across four independent runs on the Corgi scene (SelfCap dataset) under the Dropped condition.
<table><tr><td>Run</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td></tr><tr><td>1</td><td>35.81</td><td>0.989</td><td>0.008</td></tr><tr><td>2</td><td>36.86</td><td>0.990</td><td>0.007</td></tr><tr><td>3</td><td>36.04</td><td>0.990</td><td>0.008</td></tr><tr><td>4</td><td>35.99</td><td>0.990</td><td>0.008</td></tr><tr><td>Mean</td><td>36.175</td><td>0.990</td><td>0.008</td></tr><tr><td>Std</td><td>0.466489</td><td>0.000450</td><td>0.000455</td></tr></table>

## F Failure Case

Highly Periodic Content. Since Phase-DGS estimates the phase from visual patterns extracted from image content, it has a limitation when a scene contains highly repetitive patterns along the temporal axis. We observed this behavior in the Balloon scene of the Nvidia Dynamic Scene Dataset [41]. The scene consists of a person repeatedly moving a balloon up and down against a simple background, and the visual patterns of frames at different times remain nearly identical. As a result, the phase projector assigns similar phase values to frames that are temporally distant, which provides ambiguous supervision to the 4DGS training and prevents accurate reconstruction. Fig. A6 shows the estimated phase values and the rendered results on the Balloon scene. This limitation arises from the fact that Phase-DGS uses only the visual pattern of a single frame as input, and extending the phase estimation module to incorporate temporal causality across frames (e.g., frame indices or global frame-to-frame similarity) or cross-camera information is expected to mitigate this ambiguity.

![](images/91d26b26dc16d32d59390f8f2ad61e9b18d08cadf63c1241d24f8dbd991a568e.jpg)

![](images/c1ff516862358ea0830f80c0db13451541c2a42240d17ecc7272f612568ebdc7.jpg)

![](images/d0fcc3cb657a07513f61a7b313f2183d1340b3b148f4464fe60514e9d0727d43.jpg)

![](images/75d9d15ecad36d79f086ea40bdd3d08f38c6b1e1ff653d2484e58a7bca662d93.jpg)  
Figure A6: Failure Case of Phase-DGS on Balloon1, a Highly Periodic Scene.

## G Per-scene quantitative results

Tables A4 and A5 provide per-scene results corresponding to the averages reported in Tab. 1.

Table A4: Quantitative comparison on the SelfCap dataset under synchronized (Synced) and unsynchronized (Shifted, Dropped) conditions. ↑: higher is better, ↓: lower is better. Green : best among methods sharing the same backbone and condition. Red: best across all methods under the same condition.
<table><tr><td colspan="3"></td><td colspan="3">Bike</td><td colspan="3">Corgi</td><td colspan="3">Yoga</td></tr><tr><td>Condition</td><td>Backbone</td><td>Method</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td></tr><tr><td rowspan="6">Synced</td><td>K-Planes</td><td>Sync-NeRF</td><td>一</td><td>-</td><td>=</td><td></td><td>1</td><td>-</td><td></td><td>-</td><td>-</td></tr><tr><td>RealTime-4DGS</td><td> $\mathrm { O u r s }$ </td><td>26.430 28.565</td><td>0.950 0.960</td><td>0.040 0.032</td><td>35.552 34.907</td><td>0.985 0.982</td><td>0.013</td><td>30.593 32.963</td><td>0.974 0.978</td><td>0.036</td></tr><tr><td rowspan="2"></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.019</td><td></td><td></td><td>0.028</td></tr><tr><td></td><td>30.888</td><td>0.977</td><td>0.014</td><td>35.677</td><td>0.990</td><td>0.006</td><td>35.977</td><td>0.987</td><td>0.011</td></tr><tr><td rowspan="2">FreeTimeGS</td><td>VisualSync</td><td>30.490</td><td>0.974 0.978</td><td>0.016</td><td>36.646</td><td>0.990</td><td>0.006</td><td>33.203</td><td>0.984</td><td>0.016</td></tr><tr><td>Ours</td><td>31.376</td><td></td><td>0.015</td><td>37.681</td><td>0.991</td><td>0.007</td><td>35.124</td><td>0.987</td><td>0.013</td></tr><tr><td rowspan="6">Shifted</td><td>K-Planes</td><td>Sync-NeRF</td><td>=</td><td>-</td><td>-</td><td></td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>RealTime-4DGS</td><td></td><td>24.594</td><td>0.932</td><td>0.049</td><td>29.100</td><td>0.971</td><td>0.025</td><td>29.149</td><td>0.966</td><td>0.043</td></tr><tr><td rowspan="2"></td><td> $\mathrm { \dot { o } u r s }$ </td><td>26.753</td><td>0.948</td><td>0.036</td><td>30.812</td><td>0.974</td><td>0.021</td><td>27.298</td><td>0.961</td><td>0.054</td></tr><tr><td></td><td>24.139</td><td>0.935</td><td>0.035</td><td>27.678</td><td>0.965</td><td>0.018</td><td>29.693</td><td>0.965</td><td>0.026</td></tr><tr><td rowspan="2">FreeTimeGS</td><td>VisualSync</td><td>26.227</td><td>0.949</td><td>0.024</td><td>29.975</td><td>0.974</td><td>0.012</td><td>32.632</td><td>0.978</td><td>0.018</td></tr><tr><td>Ours</td><td>31.299</td><td>0.977</td><td>0.016</td><td>34.767</td><td>0.988</td><td>0.010</td><td>32.775</td><td>0.979</td><td>0.017</td></tr><tr><td rowspan="6">Dropped</td><td>K-Planes</td><td>Sync-NeRF</td><td></td><td>-</td><td>-</td><td></td><td>-</td><td>-</td><td></td><td>=</td><td>1</td></tr><tr><td rowspan="2">RealTime-4DGS</td><td></td><td>26.339</td><td>0.946</td><td>0.037</td><td>27.149</td><td>0.967</td><td>0.025</td><td>26.548</td><td>0.963</td><td></td><td>0.045</td></tr><tr><td> $\mathrm { \dot { \ o u r s } }$ </td><td>28.608</td><td>0.961</td><td>0.029</td><td>33.664</td><td>0.981</td><td>0.016</td><td>26.642</td><td>0.956</td><td>0.052</td></tr><tr><td rowspan="3">FreeTimeGS</td><td></td><td>26.304</td><td>0.947</td><td>0.027</td><td>26.505</td><td>0.964</td><td>0.020</td><td>30.872</td><td>0.976</td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td>27.491</td><td></td><td></td><td></td><td></td><td></td><td>0.028</td></tr><tr><td>VisualSync Ours</td><td>29.746 31.217</td><td>0.970 0.977</td><td>0.019 0.016</td><td>36.859</td><td>0.968 0.990</td><td>0.017 0.007</td><td>31.981 35.061</td><td>0.978 0.985</td><td></td><td>0.021 0.014</td></tr></table>

Table A5: Quantitative comparison on the Nvidia Dynamic Scene dataset under synchronized (Synced) and unsynchronized (Shifted, Dropped) conditions. ↑: higher is better, ↓: lower is better. Green : best among methods sharing the same backbone and condition. Red: best across all methods under the same condition.
<table><tr><td colspan="3"></td><td colspan="3">Umbrella</td><td colspan="3">Jumping</td><td colspan="3">Playground</td></tr><tr><td>Condition</td><td>Backbone</td><td>Method</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td></tr><tr><td rowspan="6">Synced</td><td>K-Planes</td><td>Sync-NeRF</td><td>31.467</td><td>0.968</td><td>0.028</td><td>24.228</td><td>0.889</td><td>0.088</td><td>30.972</td><td>0.969</td><td>0.027</td></tr><tr><td rowspan="2">RealTime-4DGS</td><td></td><td>32.494</td><td>0.971</td><td>0.026</td><td>26.891</td><td>0.917</td><td>0.073</td><td>29.820</td><td>0.967</td><td>0.035</td></tr><tr><td> $\mathrm { O u r s }$ </td><td>30.328</td><td>0.961</td><td>0.049</td><td>26.724</td><td>0.912</td><td>0.093</td><td>29.763</td><td>0.965</td><td>0.040</td></tr><tr><td rowspan="2">FreeTimeGS</td><td></td><td>33.839</td><td>0.978</td><td>0.019</td><td>29.759</td><td>0.942</td><td>0.032</td><td>29.904</td><td>0.969</td><td>0.016</td></tr><tr><td>VisualSync</td><td>33.998</td><td>0.978</td><td>0.019</td><td>27.904</td><td>0.929</td><td>0.039</td><td>29.384</td><td>0.967</td><td>0.018</td></tr><tr><td rowspan="2">K-Planes</td><td>Ours</td><td>32.549</td><td>0.972</td><td>0.032</td><td>28.851</td><td>0.936</td><td>0.038</td><td>29.849</td><td>0.969</td><td>0.021</td></tr><tr><td>Sync-NeRF</td><td>30.947</td><td>0.967</td><td>0.030</td><td>25.088</td><td>0.891</td><td>0.087</td><td>29.442</td><td>0.963</td><td>0.031</td></tr><tr><td rowspan="6">Shifted</td><td>RealTime-4DGS</td><td></td><td>24.432</td><td>0.942</td><td>0.067</td><td>19.365</td><td>0.852</td><td>0.135</td><td>25.452</td><td>0.950</td><td>0.050</td></tr><tr><td rowspan="2"></td><td> $\mathrm { \dot { \ o u r s } }$ </td><td>29.215</td><td>0.958</td><td>0.047</td><td>25.124</td><td>0.901</td><td>0.097</td><td>25.627</td><td>0.951</td><td>0.047</td></tr><tr><td></td><td>24.169</td><td>0.945</td><td>0.068</td><td>20.985</td><td>0.860</td><td>0.100</td><td>26.202</td><td>0.950</td><td>0.036</td></tr><tr><td rowspan="2">FreeTimeGS</td><td>VisualSync</td><td>26.042</td><td>0.951</td><td>0.044</td><td>28.071</td><td>0.931</td><td>0.038</td><td>29.761</td><td>0.967</td><td>0.018</td></tr><tr><td>Ours</td><td>32.316</td><td>0.971</td><td>0.035</td><td>28.103</td><td>0.931</td><td>0.049</td><td>29.090</td><td>0.965</td><td>0.026</td></tr><tr><td>K-Planes</td><td>Sync-NeRF</td><td>29.708</td><td>0.955</td><td>0.053</td><td>22.175</td><td>0.873</td><td>0.104</td><td>28.167</td><td>0.960</td><td>0.035</td></tr><tr><td rowspan="4">Dropped</td><td>RealTime-4DGS</td><td></td><td>23.990</td><td>0.942</td><td>0.067</td><td>23.901</td><td>0.888</td><td>0.100</td><td>28.750</td><td>0.964</td><td>0.038</td></tr><tr><td rowspan="2"></td><td> $\mathrm { O u r s }$ </td><td>29.410</td><td>0.958</td><td>0.047</td><td>26.551</td><td>0.913</td><td>0.081</td><td>28.827</td><td>0.962</td><td>0.039</td></tr><tr><td></td><td>23.948</td><td>0.942</td><td>0.065</td><td>23.509</td><td>0.881</td><td>0.076</td><td>28.482</td><td>0.963</td><td>0.023</td></tr><tr><td rowspan="2">FreeTimeGS</td><td>VisualSync</td><td>30.069</td><td>0.965</td><td>0.036</td><td>25.605</td><td>0.903</td><td>0.058</td><td>28.497</td><td>0.962</td><td></td><td>0.023</td></tr><tr><td>Ours</td><td>32.357</td><td>0.970</td><td>0.033</td><td>29.386</td><td>0.942</td><td>0.037</td><td>30.251</td><td></td><td>0.971</td><td>0.021</td></tr></table>

## H License Information

All datasets, pretrained models, and baseline implementations used in this work are used in accordance with their respective licenses and terms of use.

Datasets. We use the Nvidia Dynamic Scene Dataset [41] and SelfCap [38] for evaluation. Both datasets are publicly available for academic research and are used in accordance with the terms provided by their respective authors.

Pretrained Models. We use DINOv2 [23] as the ViT encoder, which is released under the Apache 2.0 license.

Baseline Methods. Our re-implementation of FreeTimeGS [34] is based on the original paper. We also compare with RealTime-4DGS [35], VisualSync [16], and Sync-NeRF [12]. We cite the original papers for all methods and use them in accordance with their stated conditions.