# SURE-Map: Self-Correcting Streaming Geometric Foundation Models

Mingkai Liu<sup>1,2</sup>, Hao Zhao<sup>3,∗</sup>, Xingxing Zuo<sup>1,∗</sup>

Abstract— Streaming geometric foundation models are emerging as a compelling alternative to SLAM systems. Yet this streaming nature introduces a fundamental issue: each prediction is made from limited context, which is vulnerable to dynamic objects and weak textures. Small local errors accumulate into severe geometric distortion and long-horizon scale drift. We argue that reliable streaming reconstruction requires geometric foundation models to be not only predictive, but also self-correcting. We introduce SURE-Map, a self-correcting framework built upon two complementary principles. First, we explicitly model cross-view geometric uncertainty. Unlike conventional depth or point confidence, which primarily reflects the reliability of individual-view prediction, our uncertainty directly measures whether the jointly predicted pose and depth induce geometrically consistent cross-view pixel correspondences. Second, because local correction alone cannot eliminate slowly accumulating scale errors, we introduce multi-timescale selfcorrection: fast consecutive-frame inference preserves streaming efficiency, while sparse keyframe-window inference provides longer-range geometric evidence to periodically recalibrate the scale of recent trajectories. SURE-Map establishes new stateof-the-art performance for online feed-forward reconstruction across long-horizon benchmarks, reducing ATE-RMSE from 24.00 to 17.24 m on KITTI, 5.11 to 4.74 m on Oxford Spires, and 31.37 to 28.58 m on VBR, with further improvements to 15.17, 4.63, and 22.12 m when incorporating loop-closure refinement. Project page: https://mingkai-liu.github. io/projects/sure-map/.

## I. INTRODUCTION

3D reconstruction from video streams is a fundamental capability for augmented reality (AR) [1] and embodied intelligence [2, 3]. Recent feed-forward geometric foundation models [4, 5, 6, 7, 8] have demonstrated a promising alternative to conventional SLAM pipelines by directly predicting camera poses and dense scene geometry from images, largely bypassing explicit feature matching, triangulation, and costly back-end optimization. However, extending these models to streaming reconstruction introduces a fundamental challenge. Unlike offline reconstruction, where predictions can leverage broad temporal context, an online system must operate causally with bounded memory and low latency, forcing each prediction to rely on only limited recent observations. Such restricted context makes the reconstruction particularly vulnerable to ambiguous local evidence, such as dynamic objects and weakly textured regions, where small errors in pose or geometry can progressively accumulate into severe geometric distortion and scale drift.

Existing approaches to long-horizon reconstruction largely trade off global geometric consistency against streaming efficiency. SLAM-hybrid systems [9, 10] integrate feed forward geometric priors into classical mapping pipelines, using keyframe management, bundle adjustment, and posegraph optimization to repeatedly enforce global consistency. While effective, such iterative back-end optimization introduces substantial computational overhead and weakens the simplicity and low latency of feed-forward inference. At the other extreme, offline feed-forward methods [11, 12, 13, 14] obtain longer-range geometric context by jointly processing complete sequences or reconstructing overlapping temporal chunks followed by global alignment. Their strong performance, however, relies on non-causal computation, large temporal context, or access to future frames, making them unsuitable for strict online reconstruction.

Recent state-of-the-art streaming reconstruction models LingBot-Map [19] and HorizonStream [20] maintain compact geometric context for efficient online inference. Despite their practical memory-efficient design, these systems still inherit limitations from incremental pose estimation with limited context and remain vulnerable to observation ambi guities induced by dynamic objects, geometric degeneracy, weak textures, and repetitive structures. These ambiguities introduce local geometric inconsistencies that degrade dense point-cloud quality. Over long trajectories, the resulting local pose errors gradually accumulate into severe scale drift.

In this work, we propose SURE-Map (Scale- and Uncertainty-aware REconstruction), a self-correcting framework that equips streaming geometric foundation models with the ability to detect and rectify their own geometric failures. Our first key idea is to model cross-view geometric uncertainty. Instead of asking whether an individual depth or 3D prediction is reliable in isolation, SURE-Map estimates whether the jointly predicted pose and depth induce geometrically consistent pixel correspondences across consecutive views. This provides a direct measure of the reliability of the implicit data association underlying streaming reconstruction. However, such frame-to-frame correction alone cannot prevent small scale errors from accumulating over long trajectories. We therefore introduce multi-timescale selfcorrection, coupling fast consecutive-frame inference with sparse keyframe-window inference over a longer temporal horizon. The former preserves the low latency of streaming reconstruction, while the latter provides more stable longrange geometric evidence to periodically recalibrate the scale of recently estimated trajectories. Fig. 1 illustrates these complementary effects on indoor and outdoor scenes.

Our contributions are summarized as follows:

(a) Outdoor long-horizon reconstruction  
![](images/b767599636646e557b8f6a9a3a5d836696a02bad5958bf44ea618dcc50ac1748.jpg)  
Fig. 1: Comparison of SURE-Map with online feed-forward reconstruction methods [15, 17, 19, 20, 36] across multiple benchmarks [23, 40, 21, 22, 41]. (a) On long outdoor sequences, SURE-Map produces a more complete and accurate map; LingBot-Map exhibits severe drift along the gravity direction, while HorizonStream suffers from occlusion by floating artifacts. (b) SURE-Map reconstructs indoor windows and lamps with cleaner surfaces and sharper geometry, whereas LingBot-Map and HorizonStream exhibit visible artifacts. (c) Quantitative comparison of reconstruction accuracy (F1) and pose accuracy (ATE) across five benchmarks.

• We introduce SURE-Map, a self-correcting framework for streaming geometric foundation models, addressing the challenge that local geometric errors caused by limited temporal context can accumulate into severe reconstruction distortion and long-horizon scale drift.

• We introduce cross-view geometric uncertainty, which reflects whether jointly predicted pose and depth from streaming models induce geometrically consistent pixel correspondences across views. Unlike conventional depth confidence that evaluates individual-view prediction, our formulation explicitly captures the reliability of cross-view geometry and enables uncertainty-aware geometric optimization.

• We propose multi-timescale self-correction, combining efficient consecutive-frame inference (geometrycontext-attention) with sparse keyframe-window inference (full-attention) to provide longer-range geometric evidence for periodic scale recalibration. This suppresses accumulated scale drift while preserving streaming efficiency, establishing new state-of-the-art performance on long-horizon benchmarks.

## II. RELATED WORK

Traditional 3D Reconstruction and Learned Backends. Classical 3D reconstruction pipelines include Structure-from-Motion (SfM) [24], Simultaneous Localization and Mapping (SLAM) [25, 26, 27], and

Multi-View Stereo (MVS) [28, 29]. SfM and SLAM estimate camera motion and sparse structure through feature matching, keyframes, bundle adjustment, and pose-graph optimization, while MVS recovers dense geometry from posed images. These systems are accurate and interpretable, but their explicit matching and intensive optimization are costly for long-horizon streams. Recent methods incorporate learning through features and matchers [30, 31], learned SfM/VO/SLAM modules [32, 33, 34], and learned-prior back-ends [9, 10], highlighting the growing role of learned 3D priors in reconstruction pipelines.

3D Foundation Models. Feed-forward 3D foundation models [4, 5, 6, 7, 8] have recently emerged as a strong alternative to optimization-heavy reconstruction pipelines. DUSt3R [7] directly regresses dense point maps from unposed image pairs, but requires additional alignment to handle multiple views. VGGT [4] extends this paradigm to multi-view inputs, jointly predicting camera poses and dense 3D attributes in a single forward pass. Despite their strong accuracy and generalization, these models are designed for offline processing with full input access, and their full attention becomes costly on long-horizon streams.

Feed-Forward Streaming 3D Reconstruction. Feedforward streaming 3D reconstruction extends geometric foundation models to causal video inputs. Recent methods [15, 17, 18, 35, 36, 37] transfer information across time through recurrent states, causal attention, sliding windows, keyframe memory, cache pruning, or test-time updates. Long-stream systems further improve context retention: LongStream [16] analyzes degradation from attention sink and state saturation, HorizonStream [20] mitigates scale drift via channel-wise geometric-evidence propagation, and LingBot-Map [19] improves long-range consistency with compact geometric context and large-scale longsequence training. Despite improved scalability, bounded context leaves incremental pose estimation vulnerable to ambiguities that degrade dense reconstruction and induce local pose errors, which accumulate into long-trajectory scale drift. SURE-Map addresses this error propagation by using cross-view geometric uncertainty to identify unreliable observations and multi-timescale self-correction to correct trajectory errors across temporal scales.

Confidence in 3D Reconstruction. Per-pixel confidence is widely adopted in point-map prediction and depth estimation [4, 5, 6, 7], while per-correspondence confidence is used in SLAM [9, 33, 34]. Feed-forward 3D foundation models typically learn depth or point confidence via negative log-likelihood (NLL) regression losses and use it to filter unreliable geometry. These scores primarily reflect the reliability of individual-view geometry prediction, and their raw magnitudes need not be comparable across scenes. SURE-Map instead models cross-view geometric uncertainty by assessing whether jointly predicted pose and depth induce geometrically consistent pixel correspondences across consecutive views.

## III. METHODOLOGY

## A. Preliminary and Problem Definition

Causal Streaming Reconstruction. Given an image stream $\left\{ \mathbf { I } _ { 0 } , \mathbf { I } _ { 1 } , \ldots \right\}$ , a streaming geometric foundation model processes each new frame upon arrival. At time instant $t _ { i }$ capturing image $\mathbf { I } _ { i } ~ \in ~ \mathbb { R } ^ { \bar { H } \times W \times 3 }$ , the model estimates the current camera pose and dense depth using the image and a compact memory summarizing historical observations:

$$
f _ { \theta } ( \mathbf { I } _ { i } , \mathcal { M } _ { i - 1 } )  ( \mathbf { T } _ { i } , \mathbf { D } _ { i } , \mathbf { K } _ { i } , \mathcal { M } _ { i } ) ,\tag{1}
$$

where $f _ { \theta }$ denotes the streaming geometric foundation model, $\mathbf { T } _ { i } = [ \mathbf { R } _ { i } | \mathbf { t } _ { i } ] \in S E ( 3 )$ is the camera-to-world pose, ${ \bf R } _ { i } \in  { \bf \Psi }$ $S O ( 3 ) \bar { , } \mathbf { t } _ { i } \in \mathbb { R } ^ { 3 } , \mathbf { D } _ { i } \in \mathbb { R } ^ { H \times W }$ is the dense depth map, $\mathbf { K } _ { i }$ is the camera intrinsic matrix, and $\mathcal { M } _ { i - 1 }$ denotes the compact token memory summarizing historical observations until image $\mathbf { I } _ { i - 1 }$

From Ambiguous Observations to Local Errors. Recent streaming geometric foundation models [16, 18, 19, 20] directly predict pose and dense depth without explicit 2D– 2D correspondences, relying instead on implicit data association encoded by learned scene priors. Neural networks favor smooth embeddings [38], so visually similar but geometrically distinct observations may receive similar representations. Non-causal offline systems [4, 11, 12, 13, 14] can use future multi-view evidence to reject such spurious associations, whereas causal streaming inference is restricted to limited historical context. It is therefore more vulnerable to ambiguities caused by dynamic objects, geometric degeneracy, weak textures, and repetitive structures, which degrade dense reconstruction and induce local pose errors.

From Local Errors to Scale Drift. Unlike classical monocular SLAM, streaming geometric foundation models [16, 18, 19, 20] benefit from learned scene priors and often preserve an approximately stable scale within short segments. However, causal streaming inference lacks explicit global cross-segment constraints to align the scales of different segments. Consequently, local pose errors accumulate into segment-wise scale drift over long trajectories, even when individual segments remain locally plausible. In contrast, dense depth estimation is largely driven by image appearance and tends to maintain a more stable local scale under the same limited context. SURE-Map leverages this favorable property through a dedicated inference procedure over a selected sparse keyframe window, periodically recalibrating the trajectory scale.

## B. Overview of SURE-Map

SURE-Map is a self-correcting framework for streaming geometric foundation models that addresses visual artifacts and segment-wise scale drift (Fig. 2). First, crossview geometric uncertainty assesses whether jointly predicted pose and depth induce geometrically consistent pixel correspondences across consecutive views, guiding densepoint filtering and local translation optimization. Second, multi-timescale self-correction combines fast consecutiveframe inference (geometry-context-attention [19]) with sparse keyframe-window inference (full-attention), providing longer-range geometric evidence to periodically recalibrate the scale of recent trajectories. Together, they improve dense geometry and long-horizon trajectory consistency while preserving streaming efficiency.

## C. Cross-View Geometric Uncertainty Modeling

Pose-Depth-Induced Optical Flow. As shown in Fig. 2(a), given the pose and depth predicted by a streaming model, we can analytically derive the optical flow between two images. To quantify the cross-view geometric uncertainty in the predicted pose and depth, we augment the streaming model with a cross-view geometric uncertainty head. This head estimates uncertainty from geometric tokens and is trained under the supervision of the discrepancy between the ground-truth optical flow and the flow induced by the predicted pose and depth. During streaming inference, it directly estimates the uncertainty of cross-view correspondences without explicitly predicting an optical-flow field.

For consecutive frames $\left( \mathbf { I } _ { i - 1 } , \mathbf { I } _ { i } \right)$ with resolution $H \times W$ we predict a cross-view geometric uncertainty map $\mathbf { S } _ { i } .$ . For any pixel $\mathbf { u } = [ u , v ] ^ { \top }$ in the current frame $\mathbf { I } _ { i }$ , with homogeneous coordinate $\bar { \mathbf { u } } = [ u , v , 1 ] ^ { \top }$ , we warp it into the previous frame $\mathbf { I } _ { i - 1 }$ using the predicted depth and relative pose. The resulting backward optical flow ${ \bf f } _ { i \to i - 1 } ( { \bf u } )$ represents the pose-depth-induced correspondence from $\mathbf { I } _ { i }$ to $\mathbf { I } _ { i - 1 } $

$$
\begin{array} { r l } & { \mathbf { p } ^ { \mathrm { c u r } } \qquad = \mathbf { D } _ { i } ( \mathbf { u } ) \mathbf { K } _ { i } ^ { - 1 } \bar { \mathbf { u } } , } \\ & { \mathbf { p } ^ { \mathrm { p r e v } } \qquad = \mathbf { R } _ { i - 1 } ^ { \top } ( \mathbf { R } _ { i } \mathbf { p } ^ { \mathrm { c u r } } + \mathbf { t } _ { i } - \mathbf { t } _ { i - 1 } ) , } \\ & { \mathbf { f } _ { i  i - 1 } ( \mathbf { u } ) = \pi ( \mathbf { K } _ { i - 1 } \mathbf { p } ^ { \mathrm { p r e v } } ) - \mathbf { u } . } \end{array}\tag{2}
$$

![](images/802ad362cd6cf2bed7c088dcd5a7aae431336846db4bf0909cc8f33b43c59125.jpg)  
Fig. 2: SURE-Map methodology. (a) The cross-view geometric uncertainty head is supervised by discrepancies between pose-depth-induced and ground-truth optical flow. (b) During streaming reconstruction, the uncertainty guides dense-point filtering and local translation optimization. Multi-timescale self-correction couples fast consecutive-frame inference with sparse keyframe-window inference, whose longer-range geometric evidence periodically recalibrates trajectory scale.

where $\mathbf { p } ^ { \mathrm { c u r } }$ is the 3D point back-projected from pixel u in the current camera frame, $\mathbf { p } ^ { \mathrm { p r e v } }$ is the same point expressed in the previous camera frame, and $\pi ( \cdot )$ denotes perspective division that projects 3D points onto the image plane.

Cross-View Geometric Uncertainty Prediction. The head $h _ { \psi }$ takes the geometric token maps of two consecutive frames, $\left( \mathbf { G } _ { i - 1 } , \mathbf { G } _ { i } \right)$ , as input. These tokens encode local geometry and cross-view correspondence cues, enabling the head to predict the uncertainty associated with the pose-anddepth-induced backward optical flow. Specifically, the head outputs $\mathbf { S } _ { i } = h _ { \psi } ( \mathbf { G } _ { i - 1 } , \bar { \mathbf { G } _ { i } } ) = [ \mathbf { S } _ { i } ^ { u } ; \mathbf { S } _ { i } ^ { v } ] ^ { \top } \in \mathbb { R } ^ { H \times \bar { W } \times 2 }$ , where $\mathbf { S } _ { i } ^ { u } , \mathbf { S } _ { i } ^ { v } \ \in \ \mathbb { R } ^ { H \times \dot { W } }$ denote the horizontal and vertical logvariance maps, respectively. For simplicity, $\mathbf { S } _ { i } ( \mathbf { u } )$ parameterizes only a diagonal covariance matrix:

$$
\Sigma _ { i } ( \mathbf { u } ) = \mathrm { d i a g } \left( e ^ { \mathbf { S } _ { i } ^ { u } ( \mathbf { u } ) } , e ^ { \mathbf { S } _ { i } ^ { v } ( \mathbf { u } ) } \right) .\tag{3}
$$

Residual-Based Supervision. Given the ground-truth flow $\mathbf { f } _ { i \to i - 1 } ^ { \mathrm { g t } }$ , the flow residual can be computed by $\begin{array} { r l } { \epsilon _ { i } ( \mathbf { u } ) } & { { } = } \end{array}$ $\mathbf { f } _ { i \to i - 1 } ( \mathbf { u } ) - \mathbf { f } _ { i \to i - 1 } ^ { \mathrm { g t } } ( \mathbf { u } )$ . We train $h _ { \psi }$ over the set of pixels with valid optical flow, denoted by $\Omega _ { i } ,$ , using the negative log-likelihood (NLL) loss:

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { f l o w } } = \displaystyle \frac { 1 } { | \Omega _ { i } | } \sum _ { \mathbf { u } \in \Omega _ { i } } \ell _ { i } ( \mathbf { u } ) , } \\ & { \ell _ { i } ( \mathbf { u } ) = \epsilon _ { i } ( \mathbf { u } ) ^ { \top } \Sigma _ { i } ( \mathbf { u } ) ^ { - 1 } \epsilon _ { i } ( \mathbf { u } ) + \log \left( \operatorname* { d e t } \Sigma _ { i } ( \mathbf { u } ) \right) , } \end{array}\tag{4}
$$

In the NLL loss, the Mahalanobis term encourages the network to assign higher uncertainty to large residuals in the induced optical flow, whereas the log-determinant term penalizes trivially large uncertainty estimates. The learned uncertainty thus measures the reliability of cross-view pixel correspondences.

## D. Applications of Predicted Cross-View Uncertainty

Dense-Point Filtering. For each pixel, we convert the diagonal covariance in Eq. (3) into a scalar uncertainty

$$
\sigma _ { i } ^ { \mathrm { g e o } } ( \mathbf { u } ) = \sqrt { \mathrm { t r } \big ( \Sigma _ { i } ( \mathbf { u } ) \big ) } = \sqrt { e ^ { \mathbf { S } _ { i } ^ { u } ( \mathbf { u } ) } + e ^ { \mathbf { S } _ { i } ^ { v } ( \mathbf { u } ) } } .\tag{5}
$$

To recover the dense point cloud, SURE-Map back-projects only pixels with valid predicted depths and sufficiently low uncertainty, $\sigma _ { i } ^ { \mathrm { g e o } } ( \mathbf { u } )$ This filtering removes noisy points originating from pixels with high pose-depth-induced uncertainty. Although the native streaming backbone also predicts depth confidence, this confidence does not reliably reflect the quality of the back-projected points, as demonstrated by the filtering ablation in Table V and Fig. 5.

Uncertainty-Weighted Optimization. The same uncertainty weights point-to-plane residuals in local translation optimization to reduce pose errors from ambiguous observations. We keep predicted rotations fixed to prevent scale or correspondence residuals from inducing rotation errors that accumulate multiplicatively over long sequences.

For each pair of consecutive images $\left( \mathbf { I } _ { i - 1 } , \mathbf { I } _ { i } \right)$ , let $\mathbf { t } _ { i \to i - 1 }$ denote the optimizable relative translation from $\mathbf { I } _ { i }$ to $\mathbf { I } _ { i - 1 }$ For notational simplicity, we write $\mathbf { t } _ { i \to i - 1 }$ as t throughout this section when no ambiguity arises.

Using the warping operation in Eq. (2), a pixel u with valid depth ${ { \bf D } _ { i } } ( { \bf u } )$ can be warped from frame $\mathbf { I } _ { i }$ to the previous frame $\mathbf { I } _ { i - 1 }$ . We denote the corresponding point in the previous frame by $\mathbf { p } ^ { \mathrm { p r e v } } ( \mathbf { u } ; \mathbf { t } )$ . The residual $r _ { i } ( \mathbf { u } ; \mathbf { t } )$ denotes the point-to-plane distance between $\mathbf { p } ^ { \mathrm { p r e v } } ( \mathbf { u } ; \mathbf { t } )$ and its corresponding local surface induced from the depth map $\mathbf { D } _ { i - 1 }$

$$
r _ { i } ( { \bf { u } } ; { \bf { t } } ) = { \bf { n } } ^ { \top } ( { \bf { p } } ^ { \mathrm { { p r e v } } } ( { \bf { u } } ; { \bf { t } } ) - { \bf { q } } ) ,\tag{6}
$$

where $\mathbf { q }$ is the surface point obtained by back-projecting the target pixel $\mathbf { u } ^ { \prime } \ = \ \pi ( \mathbf { K } _ { i - 1 } \mathbf { p } ^ { \mathrm { p r e v } } ( \mathbf { u } ; \mathbf { t } ) )$ in $\mathbf { I } _ { i - 1 }$ using $\mathbf { D } _ { i - 1 } ( \mathbf { u } ^ { \prime } )$ and $\mathbf { K } _ { i - 1 }$ , and n is its unit surface normal.

We further derive the uncertainty of the point-to-plane constraint from the predicted cross-view uncertainty, represented by the covariance $\pmb { \Sigma } _ { i } ( \mathbf { u } )$ defined in Eq. (3). Treating the predicted depth $\mathbf { D } _ { i - 1 }$ as locally fixed, we propagate ${ \pmb { \Sigma } } _ { i } ( { \bf u } )$ to the point-to-plane constraint via first-order Jacobian-based uncertainty propagation:

$$
\begin{array} { r l } & { a _ { i } ^ { u } ( \mathbf { u } ) = \mathbf { n } ^ { \top } \mathbf { R } _ { i - 1 } ^ { \top } \mathbf { R } _ { i } [ \mathbf { D } _ { i } ( \mathbf { u } ) / f _ { x , i } , 0 , 0 ] ^ { \top } , } \\ & { a _ { i } ^ { v } ( \mathbf { u } ) = \mathbf { n } ^ { \top } \mathbf { R } _ { i - 1 } ^ { \top } \mathbf { R } _ { i } [ 0 , \mathbf { D } _ { i } ( \mathbf { u } ) / f _ { y , i } , 0 ] ^ { \top } , } \\ & { \mathbf { J } _ { i } ^ { r } ( \mathbf { u } ) = [ a _ { i } ^ { u } ( \mathbf { u } ) , a _ { i } ^ { v } ( \mathbf { u } ) ] , } \\ & { \sigma _ { r , i } ^ { 2 } ( \mathbf { u } ) = \mathbf { J } _ { i } ^ { r } ( \mathbf { u } ) \Sigma _ { i } ( \mathbf { u } ) \mathbf { J } _ { i } ^ { r } ( \mathbf { u } ) ^ { \top } + \sigma _ { 0 } ^ { 2 } . } \end{array}\tag{7}
$$

where $a _ { i } ^ { u } ( \mathbf { u } )$ and $a _ { i } ^ { v } ( { \mathbf { u } } )$ are partial derivatives of the point-toplane residual with respect to horizontal and vertical currentframe pixel coordinates, and $\mathbf { J } _ { i } ^ { r } ( \mathbf { u } ) = [ a _ { i } ^ { u } ( \mathbf { u } ) , a _ { i } ^ { v } ( \mathbf { u } ) ]$ is the corresponding residual-pixel Jacobian. Here, $\sigma _ { r , i } ^ { 2 } ( \mathbf { u } )$ is the point-to-plane residual variance, $f _ { x , i }$ and $f _ { y , i }$ denote the camera focal lengths for $\mathbf { I } _ { i } ,$ and $\sigma _ { 0 } ^ { 2 }$ is a predefined residualvariance floor.

With the point-to-plane residual $r _ { i } ( \mathbf { u } ; \mathbf { t } )$ and its uncertainty $\sigma _ { r , i } ^ { 2 } ( \mathbf { u } )$ , the objective function for optimizing the local translation is:

$$
\mathbf { t } _ { i } ^ { \mathrm { G N } } = \arg \operatorname* { m i n } _ { \mathbf { t } } \sum _ { \mathbf { u } \in \Omega _ { i } ^ { g } } \frac { r _ { i } ( \mathbf { u } ; \mathbf { t } ) ^ { 2 } } { \sigma _ { r , i } ^ { 2 } ( \mathbf { u } ) } ,\tag{8}
$$

where $\Omega _ { i } ^ { g }$ denotes the set of pixels with valid projected ${ \mathrm { g e } } -$ ometry. We solve the above nonlinear optimization problem using a small, fixed number of Gauss–Newton iterations. The initial value of the optimizable local translation t can be computed from the pose predictions of the streaming model.

## E. Multi-Timescale Self-Correction

Consecutive-frame inference uses geometry-contextattention [19] for efficient causal updates, whereas full-attention enables all input frames to interact jointly and capture richer cross-frame geometric dependencies.

Keyframe-Window Inference. To preserve online efficiency, we maintain a fixed-size sliding keyframe window $\kappa .$ Keyframes are selected and inserted at a fixed stride, with a new keyframe inserted when the magnitude of pose-depthinduced optical flow across the latest keyframe exceeds a predefined threshold. We periodically run joint full-attention inference with the streaming model $f _ { \theta }$ over the current window:

$$
\{ ( \mathbf { D } _ { c } ^ { k } , \mathbf { T } _ { c } ^ { k } ) \} _ { c \in \mathcal { K } } = f _ { \theta } ( \{ \mathbf { I } _ { c } \} _ { c \in \mathcal { K } } ) .\tag{9}
$$

where $\mathbf { D } _ { c } ^ { k }$ and $\mathbf { T } _ { c } ^ { k }$ denote the depth and pose for the frame c predicted by the full-attention inference. The depth map $\mathbf { D } _ { c } ^ { k }$ obtained from keyframe-window inference (fullattention) will be aligned with its corresponding depth map $\mathbf { D } _ { c }$ predicted by consecutive-frame inference (geometrycontext-attention) via inverse-depth fitting:

$$
\eta = \arg \operatorname* { m i n } _ { \eta > 0 } \sum _ { c \in { \cal K } } \sum _ { \mathbf { u } \in \Omega _ { c } } \left| { \bf D } _ { c } ( \mathbf { u } ) ^ { - 1 } - \eta { \bf D } _ { c } ^ { k } ( \mathbf { u } ) ^ { - 1 } \right| ^ { 2 } ,\tag{10}
$$

where $\Omega _ { c }$ denotes the set of pixels with valid depth values in both depth maps.

Segment-Scale Estimation. The scale cue $\phi$ is obtained by comparing keyframe-window relative translation lengths with the corresponding streaming relative translation lengths. The keyframe-window relative translation lengths are divided by η to match the streaming-depth scale:

$$
\phi = \mathrm { m e d i a n } _ { ( m , n ) \in \mathcal { P } } \frac { \left\| \mathrm { t r a n s } \left( ( \mathbf { T } _ { m } ^ { k } ) ^ { - 1 } \mathbf { T } _ { n } ^ { k } \right) \right\| / \eta } { \left\| \mathrm { t r a n s } \left( \mathbf { T } _ { m } ^ { - 1 } \mathbf { T } _ { n } \right) \right\| } ,\tag{11}
$$

where P denotes valid pairs in the keyframe window and trans(·) extracts the translation vector.

Multi-Timescale Fusion. For each frame-to-frame relative edge in the recent trajectory segment covered by the current recalibration step, SURE-Map keeps the predicted rotation and updates only the translation. We define the scalerecalibrated translation estimate as

$$
\mathbf { t } _ { i } ^ { \mathrm { s c a l e } } = { \phi } \operatorname { t r a n s } ( \mathbf { T } _ { i - 1 } ^ { - 1 } \mathbf { T } _ { i } ) .\tag{12}
$$

The final translation fuses the scale-recalibrated estimate with the local translation estimate from Eq. (8):

$$
\mathbf { t } _ { i } ^ { * } = \arg \operatorname* { m i n } _ { \mathbf { t } } \left[ \left( 1 - \mu \right) \left\| \mathbf { t } - \mathbf { t } _ { i } ^ { \mathrm { s c a l e } } \right\| _ { 2 } ^ { 2 } + \mu \left\| \mathbf { t } - \mathbf { t } _ { i } ^ { \mathrm { G N } } \right\| _ { 2 } ^ { 2 } \right] ,\tag{13}
$$

where $\mu$ is a predefined hyperparameter that balances the scale-recalibrated translation estimate from keyframewindow inference and the local translation estimate from consecutive frames.

## IV. EXPERIMENTS

## A. Datasets and Implementation Details

We train only the cross-view geometric uncertainty head on TartanAir [39], using ground-truth optical flow and validity masks for residual-based supervision (see Eq. (4)). We use a frozen LingBot-Map [19] backbone and train the uncertainty head for 20k iterations with the NLL loss in Sec. III-C. Training uses 8–24-frame clips at $5 1 8 \times 3 9 2$ and AdamW with a learning rate of $1 0 ^ { - 4 }$

For evaluation, we assess long-horizon trajectory accuracy on KITTI [22], VBR [41], and Oxford Spires [21], and evaluate dense reconstruction with geometric-uncertainty filtering on Neural RGB-D [23] and 7-Scenes [40].

TABLE I: ATE-RMSE (m) on KITTI [22]. Red/blue: best/second-best within Streaming Fwd. LoGeR<sup>\*</sup> [12] denotes optimization-based LoGeR; CUT3R [17] and TTT3R [15] report both refresh settings. LC denotes loop closure.
<table><tr><td colspan="2">Methods</td><td colspan="10">KITTI ATE ↓ 03</td><td rowspan="2">Avg. 10</td></tr><tr><td></td><td></td><td>00 4542 fr. 1101 fr. 3.7 km 2.5 km</td><td>01 4661 fr. 5.1 km</td><td>02 801 fr. 0.6 km</td><td>04 271 fr. 0.4 km</td><td>05 2761 fr. 2.2 km</td><td>06 1101 fr. 1.2 km</td><td>07 1101 fr.</td><td>4071 fr.</td><td>08 1591 fr.</td><td>09 1201 fr.</td></tr><tr><td></td><td>MASt3R-SLAM [9]</td><td></td><td>530.37</td><td></td><td>18.87</td><td>88.98</td><td>159.43</td><td>92.00</td><td>3.2 km</td><td>263.75</td><td>153.07</td><td>186.64</td></tr><tr><td>Opt-nn-tic</td><td>VGGT-SLAM 2.0 [10]</td><td></td><td>163.65</td><td></td><td>50.04</td><td>19.38</td><td>159.58</td><td>46.35</td><td>57.80</td><td>167.96</td><td>76.99</td><td>92.72</td></tr><tr><td></td><td>COLMAP [24]</td><td>139.12</td><td>3.83</td><td>71.99</td><td>1.46</td><td>112.77</td><td>20.37</td><td>10.95</td><td>7.80 21.72</td><td>21.19</td><td>4.52</td><td>37.79</td></tr><tr><td></td><td>MASt3R-SfM [32]</td><td></td><td>463.52</td><td></td><td>15.80</td><td>41.44</td><td>150.39</td><td>136.14</td><td>71.69</td><td>176.36</td><td>69.50</td><td>140.60</td></tr><tr><td></td><td>DPVO [34]</td><td>113.11</td><td>16.60</td><td>113.01</td><td>2.46</td><td>0.98</td><td>59.34</td><td>55.91</td><td>19.30</td><td>110.63</td><td>74.55 13.71</td><td>52.69</td></tr><tr><td></td><td>DROID-SLAM [33]</td><td></td><td>82.81</td><td></td><td>3.20</td><td>1.47</td><td>73.50</td><td>61.10</td><td>18.41</td><td>104.22</td><td>89.49 22.19</td><td>50.71</td></tr><tr><td></td><td>VGGT-Long [11]</td><td>8.64</td><td>61.21</td><td>52.72</td><td>8.78</td><td>4.20</td><td>9.88</td><td>4.67</td><td>2.66</td><td>72.98</td><td>31.84 27.71</td><td>25.94</td></tr><tr><td></td><td>FastVGGT [14]</td><td></td><td>639.39</td><td></td><td>21.53</td><td>9.51</td><td></td><td>40.56</td><td>51.35</td><td>201.54</td><td>196.22</td><td>165.73</td></tr><tr><td></td><td>LoGeR [12]</td><td>62.34</td><td>41.64</td><td>39.64</td><td>4.89</td><td>1.82</td><td>41.27</td><td>13.99</td><td>16.24</td><td>26.46</td><td>22.71 8.84</td><td>25.44</td></tr><tr><td></td><td>LoGeR* [12]</td><td>30.47</td><td>47.91</td><td>36.32</td><td>5.38</td><td>1.95</td><td>26.34</td><td>6.60</td><td>5.55</td><td>24.41</td><td>10.12 10.11</td><td>18.65</td></tr><tr><td>Oie Fv.</td><td>Scal3R [13]</td><td>4.30</td><td>45.29</td><td>42.06</td><td>3.36</td><td>1.74</td><td>3.30</td><td>2.49</td><td>2.03</td><td>36.69</td><td>12.32 6.46</td><td>14.55</td></tr><tr><td></td><td>CUT3R [17] w/o refresh</td><td>185.89</td><td>651.52</td><td>296.98</td><td>148.06</td><td>22.17</td><td>155.61</td><td>132.54</td><td>77.03</td><td>238.39 205.94</td><td>193.39</td><td>209.78</td></tr><tr><td></td><td>CUT3R [17] w/ refresh</td><td>190.38</td><td>90.59</td><td>264.39</td><td>20.40</td><td>7.31</td><td>92.25</td><td>67.54 22.48</td><td>145.08</td><td>67.42</td><td>40.00</td><td>91.62</td></tr><tr><td></td><td>TTT3R [15] w/o refresh</td><td>190.93</td><td>546.84</td><td>218.77</td><td>105.28</td><td>11.62</td><td>153.12</td><td>132.94</td><td>70.95</td><td>180.57 211.01</td><td>133.00</td><td>177.73</td></tr><tr><td></td><td>TTT3R [15] w/ refresh</td><td>119.94</td><td>99.59</td><td>238.07</td><td>16.83</td><td>3.98</td><td>36.38</td><td>47.20</td><td>11.62</td><td>107.33</td><td>86.96 33.58</td><td>72.86</td></tr><tr><td>d.</td><td>Stream3R [18]</td><td>190.98</td><td>681.95</td><td>301.40</td><td>158.25</td><td>102.73</td><td>159.85</td><td>135.03</td><td>90.37</td><td>261.15 216.31</td><td>207.49</td><td>227.77</td></tr><tr><td>Samng</td><td>StreamVGGT [35]</td><td>191.93</td><td>653.06</td><td>303.35</td><td>157.50</td><td>108.24</td><td>160.46</td><td>133.71</td><td>89.00</td><td>263.95 216.69</td><td>209.80</td><td>226.15</td></tr><tr><td></td><td></td><td>186.46</td><td>623.62</td><td>289.16</td><td>166.74</td><td>68.00</td><td>143.84</td><td>117.57</td><td>85.33</td><td>221.56</td><td>215.41 156.92</td><td>206.78</td></tr><tr><td></td><td>InfiniteVGGT [36]</td><td>92.55</td><td>46.01</td><td>134.70</td><td>3.81</td><td>1.95</td><td>84.69</td><td>23.12</td><td>14.93</td><td></td><td>85.61</td><td>51.90</td></tr><tr><td></td><td>LongStream [16]</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>62.07</td><td>21.48</td><td></td></tr><tr><td></td><td>LingBot-Map [19]</td><td>23.09</td><td>88.05</td><td>70.57</td><td>2.80</td><td>0.95</td><td>18.48</td><td>4.90</td><td>5.03</td><td>13.83</td><td>22.17 14.20</td><td>24.00</td></tr><tr><td></td><td>HorizonStream [20]</td><td>29.20</td><td>23.90</td><td>96.98</td><td>5.56</td><td>0.71</td><td>13.97</td><td>5.41</td><td>6.19</td><td>22.32</td><td>29.58 12.29</td><td>22.38</td></tr><tr><td></td><td></td><td>15.37</td><td>23.90</td><td>70.89</td><td>5.56</td><td>0.71</td><td>7.93</td><td>7.72</td><td>6.19</td><td>22.32</td><td>27.18 12.29</td><td>18.19</td></tr><tr><td></td><td>HorizonStream [20] w/LC</td><td></td><td>39.54</td><td>53.47</td><td>2.28</td><td>1.33</td><td>18.44</td><td></td><td></td><td></td><td></td><td>17.24</td></tr><tr><td></td><td>SURE-Map</td><td>20.81</td><td></td><td></td><td></td><td></td><td></td><td>4.00</td><td>3.99</td><td>14.38</td><td>21.15 10.20</td><td></td></tr><tr><td></td><td>SURE-Map w/LC</td><td>20.59</td><td>39.54</td><td>35.85</td><td>2.28</td><td>1.33</td><td>17.48</td><td>3.36</td><td>3.99</td><td>14.38</td><td>17.87 10.20</td><td>15.17</td></tr></table>

TABLE II: Average ATE-RMSE (m) on all sequences of the longhorizon benchmarks.
<table><tr><td>Method</td><td>KITTI [22]</td><td>Oxford [21]</td><td>VBR [41]</td></tr><tr><td>CUT3R [17]</td><td>91.62</td><td>32.47</td><td>66.25</td></tr><tr><td>TTT3R [15]</td><td>72.86</td><td>25.05</td><td>64.99</td></tr><tr><td>InfiniteVGĠT [36]</td><td>206.78</td><td>31.75</td><td>91.60</td></tr><tr><td>LongStream [16]</td><td>51.90</td><td>15.92</td><td>77.93</td></tr><tr><td>LingBot-Map [19]</td><td>24.00</td><td>5.11</td><td>31.37</td></tr><tr><td>HorizonStreâm [20]</td><td>22.38</td><td>8.87</td><td>29.68</td></tr><tr><td>HorizonStream [20] w/LC</td><td>18.19</td><td>8.12</td><td>22.87</td></tr><tr><td>SURE-Map</td><td>17.24</td><td>4.74</td><td>28.58</td></tr><tr><td>SURE-Map w/LC</td><td>15.17</td><td>4.63</td><td>22.12</td></tr></table>

## B. Baselines

For long-horizon trajectory evaluation, we group baselines by their dominant estimation paradigm: optimization-centric SfM/SLAM methods [9, 10, 24, 32, 33, 34], offline feedforward methods [11, 12, 13, 14], and streaming feedforward methods [15, 16, 17, 18, 19, 20, 35, 36], as summarized in Tables I and II. Optimization-centric methods may operate online but rely primarily on iterative pose estimation. For loop closure (LC), we retrieve revisited frame pairs with cached early-layer DINOv2 [42] features and use the resulting geometric corrections as pose graph optimization (PGO) constraints. For point-cloud filtering evaluation, we compare against streaming feed-forward reconstruction methods [15, 17, 18, 19, 20, 35, 36, 37] on Neural RGB-D and 7-Scenes.

## C. Evaluation Metrics

For trajectory evaluation, we report ATE-RMSE [43] in meters after standard Sim(3) trajectory alignment. For dense reconstruction, we report Accuracy, Completeness, Chamfer Distance (CD), and F1 score after point-cloud alignment [44]; F1 uses a 0.05 m distance threshold.

TABLE III: Dense reconstruction results for streaming feedforward methods. Acc./Comp.: m; F1: %.
<table><tr><td rowspan="2">Method</td><td colspan="3">NRGBD [23]</td><td colspan="3">7-Scenes [40]</td></tr><tr><td>Acc. ↓</td><td>Comp. ↓</td><td>F1 ↑</td><td>Acc. ↓</td><td>Comp. ↓</td><td>F1 ↑</td></tr><tr><td>StreamVGGT [35]</td><td>0.13</td><td>0.05</td><td>45.08</td><td>0.04</td><td>0.11</td><td>69.44</td></tr><tr><td>InfiniteVGGT [36]</td><td>0.13</td><td>0.05</td><td>42.27</td><td>0.04</td><td>0.11</td><td>68.53</td></tr><tr><td>CUT3R [17]</td><td>0.25</td><td>0.15</td><td>32.22</td><td>0.07</td><td>0.10</td><td>58.98</td></tr><tr><td>TTT3R [15]</td><td>0.16</td><td>0.06</td><td>53.55</td><td>0.03</td><td>0.08</td><td>77.25</td></tr><tr><td>Wint3R [37]</td><td>0.09</td><td>0.04</td><td>56.96</td><td>0.03</td><td>0.07</td><td>78.81</td></tr><tr><td>Stream3R [18]</td><td>0.21</td><td>0.07</td><td>54.07</td><td>0.02</td><td>0.09</td><td>78.79</td></tr><tr><td>HorizonStream [20]</td><td>0.14</td><td>0.07</td><td>49.13</td><td>0.79</td><td>0.30</td><td>51.09</td></tr><tr><td>LingBot-Map [19]</td><td>0.074</td><td>0.030</td><td>65.10</td><td>0.035</td><td>0.043</td><td>81.77</td></tr><tr><td>SURE-Map</td><td>0.067</td><td>0.035</td><td>66.20</td><td>0.033</td><td>0.045</td><td>81.93</td></tr></table>

## D. Experimental Results

Fig. 3 compares our learned cross-view geometric uncertainty with the depth confidence predicted from the streaming 3D backbone [19]. Our cross-view uncertainty highlights mismatches between pose-depth-induced optical flow and the cross-frame pixel correspondences, often in dynamic or ambiguous regions. Depth confidence instead primarily reflects the range-dependent reliability of individual-view geometry rather than cross-view correspondence quality.

Camera Pose Estimation. Fig. 4 compares representative long-horizon trajectories without PGO post-processing, showing that SURE-Map reduces segment-wise drift through online correction alone. Table I reports per-sequence KITTI [22] results, while Table II summarizes average ATE-RMSE on KITTI, Oxford Spires [21], and VBR [41]. SURE-Map performs strongly across all benchmarks. Optional PGO incorporates LC constraints between revisited frames, further improving global consistency.

3D Reconstruction with Geometric-Uncertainty Filtering. Neural RGB-D [23] features cluttered indoor scenes, and 7-Scenes [40] provides clean indoor environments. Table III shows that geometric-uncertainty filtering improves accuracy and F-score while largely preserving completeness. Runtime Analysis. Under the same streaming setting, SURE-Map adds 15–30 ms per frame over LingBot-Map [19]. On KITTI sequence 04 [22], throughput decreases from 11.68 to 9.17 FPS. Despite the slight FPS reduction, SURE-Map still preserves streaming efficiency.

![](images/0ed70db5b7b494d7906b5f34bafb0357d2374d6fc0006192591adb0c9262165c.jpg)  
Fig. 3: Comparison of our learned cross-view uncertainty and the depth confidence from the streaming model [19]. Warmer colors indicate higher uncertainty. Our geometric uncertainty map highlights dynamic or ambiguous regions where posedepth-induced optical flow disagrees with consistent pixel correspondences while preserving static vehicles; depth confidence mainly reflects range-dependent reliability.

TABLE V: Point-cloud filtering ablation. CD/Comp.: m; F1: %.
<table><tr><td>Dataset</td><td>Variant</td><td>CD ↓</td><td>Comp. ↓</td><td>F1 ↑</td></tr><tr><td rowspan="3">NRGBD [23]</td><td>SURE-Map</td><td>0.051</td><td>0.035</td><td>66.20</td></tr><tr><td>SURE-Map w/o uncertainty filtering</td><td>0.052</td><td>0.030</td><td>65.10</td></tr><tr><td>SURE-Map w/ depth conf.&quot; filtering</td><td>0.084</td><td>0.113</td><td>65.00</td></tr><tr><td rowspan="3"></td><td>SURE-Map</td><td>0.039</td><td>0.045</td><td>81.93</td></tr><tr><td>7-Scenes [40] SURE-Map w/o uncertainty filtering</td><td>0.039</td><td>0.043</td><td>81.77</td></tr><tr><td>SURE-Map w/ depth conf. filtering</td><td>0.051</td><td>0.079</td><td>80.90</td></tr></table>

Ablation Studies. Table IV reports progressive ablations of scale recalibration (Sec. III-E), uncertainty-weighted optimization (Sec. III-D), and loop closure (LC). Scale recalibration reduces accumulated drift, while uncertainty-weighted optimization provides complementary local pose corrections. At a fixed point-removal ratio, Table V and Fig. 5 show that uncertainty filtering improves geometric quality while preserving valid structures, whereas depth-confidence filtering discards more valid points and degrades reconstruction.

![](images/c7ed2ef39416782f6c8192ea86c48aec7b23c138addd8b69f5967321635c8f29.jpg)

Fig. 4: Qualitative trajectory comparison on long-horizon sequences. SURE-Map better preserves global trajectory shape compared to LingBot-Map [19], HorizonStream [20], and LongStream [16]. Ground truth is shown in blue, and predictions are shown in orange.  
![](images/9b78390ce94c0428dee674acabef6beacec69ca6ed16713e949a7c8849c50cfd.jpg)  
Fig. 5: Qualitative point-cloud filtering ablation. Learned geometric uncertainty rejects unreliable points while preserving valid scene structures.

TABLE IV: Pose-estimation ablation with ATE-RMSE (m).
<table><tr><td>Variant</td><td>Oxford [21]</td><td>KITTI [22]</td><td>VBR [41]</td></tr><tr><td>SURE-Map w/LC</td><td>4.63</td><td>15.17</td><td>22.12</td></tr><tr><td>LC</td><td>4.74</td><td>17.24</td><td>28.58</td></tr><tr><td>uncertainty-weighted opt.</td><td>4.91</td><td>17.57</td><td>29.96</td></tr><tr><td>scale recalibration</td><td>5.11</td><td>24.00</td><td>31.37</td></tr></table>

## V. CONCLUSION

We presented SURE-Map, a self-correcting framework for streaming geometric foundation models. It uses cross-view geometric uncertainty to identify unreliable pixel correspondences for indoor point-cloud filtering, while multi-timescale self-correction couples fast consecutive-frame inference with sparse keyframe-window inference to correct long-horizon trajectory errors. Experiments demonstrate state-of-the-art long-horizon trajectory accuracy and improved dense geometry while preserving streaming efficiency.

## REFERENCES

[1] T. Whelan, S. Leutenegger, R. F. Salas-Moreno, B. Glocker, and A. J. Davison, “ElasticFusion: Dense SLAM without a pose graph,” in Robotics: Science and Systems, vol. 11, Rome, Italy, 2015.

[2] C. Huang, O. Mees, A. Zeng, and W. Burgard, “Visual language maps for robot navigation,” in 2023 IEEE International Conference on Robotics and Automation (ICRA), pp. 10608–10615, IEEE, 2023.

[3] N. Hughes, Y. Chang, and L. Carlone, “Hydra: A real-time spatial perception system for 3d scene graph construction and optimization,” in Robotics: Science and Systems, 2022.

[4] J. Wang, M. Chen, N. Karaev, A. Vedaldi, C. Rupprecht, and D. Novotny, “Vggt: Visual geometry grounded transformer,” in 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 5294–5306, IEEE, 2025.

[5] H. Lin, S. Chen, J. Liew, D. Y. Chen, Z. Li, G. Shi, J. Feng, and B. Kang, “Depth anything 3: Recovering the visual space from any views,” arXiv preprint arXiv:2511.10647, 2025.

[6] Y. Wang, J. Zhou, H. Zhu, W. Chang, Y. Zhou, Z. Li, J. Chen, J. Pang, C. Shen, and T. He, “π<sup>3</sup>: Permutation-equivariant visual geometry learning,” in International Conference on Learning Representations, vol. 2026, pp. 10481–10497, 2026.

[7] S. Wang, V. Leroy, Y. Cabon, B. Chidlovskii, and J. Revaud, “Dust3r: Geometric 3d vision made easy,” in 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 20697–20709, IEEE, 2024.

[8] V. Leroy, Y. Cabon, and J. Revaud, “Grounding image matching in 3d with mast3r,” in European conference on computer vision, pp. 71–91, Springer, 2024.

[9] R. Murai, E. Dexheimer, and A. J. Davison, “Mast3r-slam: Realtime dense slam with 3d reconstruction priors,” in 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 16695–16705, IEEE, 2025.

[10] D. Maggio and L. Carlone, “Vggt-slam 2.0: Real-time dense feedforward scene reconstruction,” arXiv preprint arXiv:2601.19887, 2026.

[11] K. Deng, Z. Ti, J. Xu, J. Yang, and J. Xie, “Vggt-long: Chunk it, loop it, align it–pushing vggt’s limits on kilometer-scale long rgb sequences,” in 2026 IEEE International Conference on Robotics and Automation (ICRA), IEEE, 2026.

[12] J. Zhang, C. Herrmann, J. Hur, C. Sun, M.-H. Yang, F. Cole, T. Darrell, and D. Sun, “Loger: Long-context geometric reconstruction with hybrid memory,” arXiv preprint arXiv:2603.03269, 2026.

[13] T. Xie, P. Yang, Y. Jin, Y. Cai, W. Yin, W. Ren, Q. Zhang, W. Hua, S. Peng, X. Guo, et al., “Scal3r: Scalable test-time training for largescale 3d reconstruction,” arXiv preprint arXiv:2604.08542, 2026.

[14] Y. Shen, Z. Zhang, Y. Qu, X. Zheng, J. Ji, S. Zhang, and L. Cao, “Fastvggt: Training-free acceleration of visual geometry transformer,” arXiv preprint arXiv:2509.02560, 2025.

[15] X. Chen, Y. Chen, Y. Xiu, A. Geiger, and A. Chen, “Ttt3r: 3d reconstruction as test-time training,” in International Conference on Learning Representations, vol. 2026, pp. 50694–50718, 2026.

[16] C. Cheng, X. Chen, T. Xie, W. Yin, W. Ren, Q. Zhang, X. Guo, and H. Wang, “Longstream: Long-sequence streaming autoregressive visual geometry,” arXiv preprint arXiv:2602.13172, 2026.

[17] Q. Wang, Y. Zhang, A. Holynski, A. A. Efros, and A. Kanazawa, “Continuous 3d perception model with persistent state,” in 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 10510–10522, IEEE, 2025.

[18] Y. Lan, Y. Luo, F. Hong, S. Zhou, H. Chen, Z. Lyu, B. Dai, S. Yang, C. C. Loy, and X. Pan, “Stream3r: Scalable sequential 3d reconstruction with causal transformer,” in International Conference on Learning Representations, vol. 2026, pp. 42746–42768, 2026.

[19] L.-Z. Chen, J. Gao, Y. Chen, K. L. Cheng, Y. Sun, L. Hu, N. Xue, X. Zhu, Y. Shen, Y. Yao, et al., “Geometric context transformer for streaming 3d reconstruction,” arXiv preprint arXiv:2604.14141, 2026.

[20] C. Cheng, P. Tao, N. Yao, G. Ding, X. Chen, Y. Du, X. Guo, W. Yin, W. Ren, Q. Zhang, et al., “Horizonstream: Long-horizon attention for streaming 3d reconstruction,” arXiv preprint arXiv:2605.23889, 2026.

[21] Y. Tao, M. A. Mu<sup>´</sup> noz-Ba˜ n˜on, L. Zhang, J. Wang, L. F. T. Fu, and´ M. Fallon, “The oxford spires dataset: Benchmarking large-scale lidarvisual localisation, reconstruction and radiance field methods,” The International Journal of Robotics Research, vol. 45, no. 6, pp. 839– 857, 2026.

[22] A. Geiger, P. Lenz, and R. Urtasun, “Are we ready for autonomous driving? the kitti vision benchmark suite,” in 2012 IEEE conference on computer vision and pattern recognition, pp. 3354–3361, IEEE, 2012.

[23] D. Azinovic, R. Martin-Brualla, D. B. Goldman, M. Nießner, and´ J. Thies, “Neural rgb-d surface reconstruction,” in 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 6280–6291, IEEE, 2022.

[24] J. L. Schonberger and J.-M. Frahm, “Structure-from-motion revisited,” in Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 4104–4113, 2016.

[25] R. Mur-Artal, J. M. M. Montiel, and J. D. Tardos, “Orb-slam: A versatile and accurate monocular slam system,” IEEE transactions on robotics, vol. 31, no. 5, pp. 1147–1163, 2015.

[26] L. Von Stumberg, V. Usenko, and D. Cremers, “Direct sparse visualinertial odometry using dynamic marginalization,” in 2018 IEEE International Conference on Robotics and Automation (ICRA), pp. 2510– 2517, IEEE, 2018.

[27] C. Forster, M. Pizzoli, and D. Scaramuzza, “Svo: Fast semi-direct monocular visual odometry,” in 2014 IEEE international conference on robotics and automation (ICRA), pp. 15–22, IEEE, 2014.

[28] Y. Furukawa and J. Ponce, “Accurate, dense, and robust multiview stereopsis,” IEEE transactions on pattern analysis and machine intelligence, vol. 32, no. 8, pp. 1362–1376, 2009.

[29] J. L. Schonberger, E. Zheng, J.-M. Frahm, and M. Pollefeys, “Pixel-¨ wise view selection for unstructured multi-view stereo,” in European conference on computer vision, pp. 501–518, Springer, 2016.

[30] M. Tyszkiewicz, P. Fua, and E. Trulls, “Disk: Learning local features with policy gradient,” Advances in neural information processing systems, vol. 33, pp. 14254–14265, 2020.

[31] P. Lindenberger, P.-E. Sarlin, and M. Pollefeys, “Lightglue: Local feature matching at light speed,” in 2023 IEEE/CVF international conference on computer vision (ICCV), pp. 17581–17592, IEEE, 2023.

[32] B. P. Duisterhof, L. Zust, P. Weinzaepfel, V. Leroy, Y. Cabon, and J. Revaud, “Mast3r-sfm: a fully-integrated solution for unconstrained structure-from-motion,” in 2025 International Conference on 3D Vision (3DV), pp. 1–10, IEEE, 2025.

[33] Z. Teed and J. Deng, “Droid-slam: Deep visual slam for monocular, stereo, and rgb-d cameras,” Advances in neural information processing systems, vol. 34, pp. 16558–16569, 2021.

[34] Z. Teed, L. Lipson, and J. Deng, “Deep patch visual odometry,” Advances in neural information processing systems, vol. 36, pp. 39033– 39051, 2023.

[35] D. Zhuo, W. Zheng, J. Guo, Y. Wu, J. Zhou, and J. Lu, “Streaming visual geometry transformer,” in International Conference on Learning Representations, vol. 2026, pp. 88055–88072, 2026.

[36] S. Yuan, Y. Yang, X. Yang, X. Zhang, Z. Zhao, L. Zhang, and Z. Zhang, “Infinitevggt: Visual geometry grounded transformer for endless streams,” arXiv preprint arXiv:2601.02281, 2026.

[37] Z. Li, J. Zhou, Y. Wang, H. Guo, W. Chang, Y. Zhou, H. Zhu, J. Chen, C. Shen, and T. He, “Wint3r: Window-based streaming reconstruction with camera token pool,” in International Conference on Learning Representations, vol. 2026, pp. 59686–59702, 2026.

[38] N. Rahaman, A. Baratin, D. Arpit, F. Draxler, M. Lin, F. Hamprecht, Y. Bengio, and A. Courville, “On the spectral bias of neural networks,” in International Conference on Machine Learning, pp. 5301–5310, PMLR, 2019.

[39] W. Wang, D. Zhu, X. Wang, Y. Hu, Y. Qiu, C. Wang, Y. Hu, A. Kapoor, and S. Scherer, “Tartanair: A dataset to push the limits of visual slam,” in 2020 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pp. 4909–4916, IEEE, 2020.

[40] J. Shotton, B. Glocker, C. Zach, S. Izadi, A. Criminisi, and A. Fitzgibbon, “Scene coordinate regression forests for camera relocalization in rgb-d images,” in Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 2930–2937, 2013.

[41] L. Brizi, E. Giacomini, L. Di Giammarino, S. Ferrari, O. Salem, L. De Rebotti, and G. Grisetti, “Vbr: A vision benchmark in rome,”

in 2024 IEEE International Conference on Robotics and Automation (ICRA), pp. 15868–15874, IEEE, 2024.

[42] M. Oquab, T. Darcet, T. Moutakanni, H. Vo, M. Szafraniec, V. Khalidov, P. Fernandez, D. Haziza, F. Massa, A. El-Nouby, et al., “Dinov2: Learning robust visual features without supervision,” arXiv preprint arXiv:2304.07193, 2023.

[43] J. Sturm, N. Engelhard, F. Endres, W. Burgard, and D. Cremers, “A benchmark for the evaluation of RGB-D SLAM systems,” in 2012 IEEE/RSJ International Conference on Intelligent Robots and Systems, pp. 573–580, IEEE, 2012.

[44] X. Zuo, N. Yang, N. Merrill, B. Xu, and S. Leutenegger, “Incremental dense reconstruction from monocular video with guided sparse feature volume fusion,” IEEE Robotics and Automation Letters, vol. 8, no. 6, pp. 3875–3882, 2023.