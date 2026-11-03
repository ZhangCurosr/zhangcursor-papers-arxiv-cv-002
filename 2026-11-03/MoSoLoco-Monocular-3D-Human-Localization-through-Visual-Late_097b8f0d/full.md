# MoSoLoco: Monocular 3D Human Localization through Visual Latent Features and Social Context

Yasamin Borhani<sup>1</sup>, Javad Khoramdel<sup>2</sup>, Yang Gao<sup>1</sup>, Po-Chien Luan<sup>1</sup>, Alexandre Alahi<sup>1</sup> <sup>1</sup>Visual Intelligence for Transportation (VITA) Laboratory, EPFL, Switzerland <sup>2</sup>Faculty of Computer Science, University of Vienna, Vienna, Austria

![](images/9073a5e83ce96ccd1e517537da20801630e7ae1056489a0999a803fe9924b9ef.jpg)  
Fig. 1: Overview of the proposed system architecture. The pipeline processes raw monocular video by extracting 2D human skeletons and joint-aligned latent features from the pose detector. These inputs are fused within our MoSoLoco module, alongside within-frame multi-agent spatial context, to estimate per-frame 3D global coordinates. The coordinates are then evaluated in a downstream trajectory forecasting study.

Abstract—Estimating the metric 3D position of people from a monocular camera is important for robot navigation and motion prediction. Lightweight pose-based localizers are attractive for robotic hardware with limited computational resources because they operate on compact 2D keypoints. However, they typically discard the richer visual representations computed by the pose estimator. We introduce MoSoLoco, a per-frame monocular 3D human localizer that combines the pose detector’s internal visual features with the detected 2D skeleton, without requiring an additional RGB or depth encoder. MoSoLoco can further incorporate sparse same-frame multi-person context to refine individual position estimates. Experiments on JRDB and NuScenes show that reusing pose-detector representations improves localization, reducing the best JRDB error by approximately 23% relative to the strongest evaluated keypoint-based baseline. A sparse social-interaction module incorporates nearby detections from the same image for further refinement. Controlled trajectoryforecasting experiments show downstream benefits, while the complete pipeline remains suitable for efficient Jetson AGX Xavier inference.

Index Terms—Monocular vision, 3D human localization, human pose estimation, trajectory forecasting

## I. INTRODUCTION

Robots operating around people can benefit from real-world 3D human positions for navigation, interaction, and motion prediction [14, 19, 23]. Monocular cameras are attractive for compact systems, but perspective projection provides no direct metric depth. One solution is to add a depth sensor such as LiDAR, which increases hardware cost and system complexity; another is to use large monocular depth models such as UniDepth V2 [15], which provide depth understanding at a considerable computational cost. A pose estimator already provides a compact representation of each person and can support tasks beyond pose estimation, such as pedestrian intention prediction [17]. This motivates us to ask: can computation already performed for human pose estimation also support metric 3D localization?

Pose-based localizers estimate 3D person positions from 2D pose detections. MonoLoco [2] uses camera-normalized skeletons, while MonoTransmotion [12] adds temporal pose observations. In these approaches, localization relies mainly on the final keypoints, even though the detector computes richer visual representations before producing them. Similarly, trajectory-forecasting methods model interactions among nearby people [1, 18], whereas pose-based localizers have generally focused on the target person and temporal observations rather than same-frame spatial context. These observations point to two underused sources of localization information: intermediate detector features and neighboring detections.

The study proceeds in two stages. First, a controlled feature probe tests whether frozen pose-detector representations retain camera-depth information beyond coarse skeleton-derived cues. These observations motivate MoSoLoco, a per-frame monocular 3D human localizer that estimates each person’s 3D position from the detector’s visual representations and the detected 2D skeleton expressed in camera coordinates, as illustrated in Fig. 1. The pose detector remains frozen, so no additional RGB or depth encoder is required. Two localizer designs are considered: a lightweight MLP-based model and a more expressive Transformer-based model. For multi-person scenes, the model can optionally use sparse attention over nearby detections from the same image. This module uses neither identities nor observations from previous frames, allowing the contributions of detector representations and same-frame spatial context to be evaluated independently.

![](images/938f8deb091f8bc497d52a672f7e979d05bc9d7f2542e166ca5e8147c72638e0.jpg)  
Fig. 2: Overview of MoSoLoco architecture. (Left) Given RGB images and camera intrinsic parameters, the framework produces per-frame 3D world coordinates using three key localization components: (1) DETRPose [10] for multi-person 2D pose detection, (2) a non-linear probe that maps each joint-specific decoder embedding to a latent feature, and (3) the MoSoLoco network. Fixed trajectory forecasters use the estimated coordinates for downstream evaluation. (Right) MoSoLoco processes explicit 2D skeletons and the joint-aligned latent features via dual streams. Both streams are linearly projected and processed independently before bidirectional cross-attention exchanges geometric and latent information. Attentive pooling produces one token per person hypothesis. An optional sparse k-nearest-neighbor attention block then incorporates within-frame context before the regression head recovers the center of the pedestrian’s 3D bounding box.

We evaluate the method on JRDB [13] and NuScenes [4]. The experiments separate the effects of skeleton geometry, pose-detector representations, fusion architecture, detector capacity, and same-frame multi-person context. We also evaluate the estimated coordinates as observations for fixed trajectory forecasters, following the view that errors in real perception inputs can propagate to downstream motion prediction [23]. Finally, we measure detector–localizer inference on an embedded robotic platform.

Our contributions are as follows:

• Visual cues for monocular 3D localization: a controlled analysis shows that visual features computed inside a frozen 2D pose detector retain distance-related information beyond coarse skeleton geometry. MoSoLoco incorporates these features with the detected 2D skeleton to estimate per-person 3D positions, without adding a separate image or depth encoder.

• Multi-person spatial context: inspired by the value of inter-person context in human motion modeling, we investigate whether neighboring detections can also refine per-person monocular localization and introduce a sparse attention module that remains frame-based.

• Empirical validation across settings: across JRDB and NuScenes with two detector capacities, MoSoLoco reduces localization error by up to 23.9% relative to the evaluated keypoint-based baselines. On JRDB, it matches or exceeds the accuracy of the much larger UniDepth and FCOS3D reference systems, while downstream forecasting and embedded-deployment studies assess its practical utility.

## II. RELATED WORK

Monocular 3D localization is commonly addressed through 3D object detection [21, 20, 8, 22] or metric-depth estimation [15]. These approaches provide general image-to-3D solutions but require a dedicated visual or depth network. The setting considered here is different: a human-centric perception stack already contains a 2D pose estimator, and the question is whether representations computed inside that estimator can also provide metric localization cues without introducing another image encoder.

Pose-based methods offer a lightweight alternative by exploiting human geometry. MonoLoco [2] maps cameranormalized 2D skeletons to 3D locations, while MonStereo [3] and pose-aware localization methods [16] further exploit human priors and visibility cues. Chen et al. [5] complement skeleton-based localization with pseudo-LiDAR depth from an auxiliary image network, whereas MonoTransmotion [12] uses temporal pose sequences to improve localization and trajectory prediction. MoSoLoco instead reuses joint-aligned features already computed by the pose detector and fuses them with skeleton geometry. It requires neither an additional depth/image network nor temporal observations, and it additionally investigates sparse same-frame multi-person context.

Human trajectory forecasting has evolved from recurrent social models [1] to generative and attention-based approaches [7, 9, 18, 6]. The contribution here is not a new forecasting model. Forecasting serves as a controlled downstream evaluation of localization quality: fixed predictors and ground-truth association isolate whether improved per-frame 3D observations produce more useful observed trajectories.

## III. METHODOLOGY

MoSoLoco estimates the metric 3D center of each detected person from a single frame while reusing representations already computed by the upstream pose estimator. As shown in Fig. 2, each DETRPose [10] detection provides two complementary representations: explicit 2D skeleton geometry and joint-aligned decoder features available before keypoint regression. MoSoLoco fuses these representations into a personlevel embedding and optionally refines it using sparse context from other people in the same frame. DETRPose remains frozen during localizer training, so no additional RGB or depth encoder is introduced.

## A. Skeleton Geometry and Pose-Feature Reuse

For person i and joint $j ,$ , let $( u _ { i j } , v _ { i j } )$ denote the detected image coordinates. Camera-dependent pixel scale is removed using intrinsics $( f _ { x } , f _ { y } , c _ { x } , c _ { y } ) ;$

$$
\mathbf { g } _ { i j } = \left[ \frac { u _ { i j } - c _ { x } } { f _ { x } } , \frac { v _ { i j } - c _ { y } } { f _ { y } } \right] ^ { T } .\tag{1}
$$

The resulting $J = 1 7$ joint tokens retain the explicit geometry used by conventional pose-based localizers.

Rather than discarding the detector representation after keypoint prediction, the corresponding 256-dimensional DE-TRPose embedding $\mathbf { h } _ { i j }$ is reused for each joint. A three-layer non-linear probe $\phi ( \cdot )$ maps it to $\mathbf { l } _ { i j } = \bar { \phi ( \mathbf { h } _ { i j } ) } \in \mathbb { R } ^ { 1 2 8 }$ . The probe is trained jointly with the localizer while DETRPose remains frozen, allowing the visual representation to be reused without a second image backbone.

In MoSoLoco-Trans, geometry and latent features are projected to 256-dimensional joint tokens and processed independently before bidirectional cross-attention exchanges information between the two streams. Residual fusion followed by attentive pooling produces a 128-dimensional person token. This preserves the joint-level correspondence between explicit geometry and detector features while allowing either representation to complement ambiguities in the other.

MoSoLoco-MLP tests whether the benefit of feature reuse depends on Transformer fusion. The 34-dimensional flattened skeleton is concatenated with the mean-pooled 128-dimensional latent representation, projected to a 256- dimensional embedding, and processed by three residual MLP stages. Both variants therefore use the same geometric and detector information but differ in how those representations are fused.

## B. Sparse Same-Frame Multi-Person Context

Monocular depth can remain ambiguous for an isolated person hypothesis. Each person representation can therefore be refined using other detections visible in the same frame. For each hypothesis, the center of its 2D skeleton is computed in the normalized camera plane and the $k _ { i } = \operatorname* { m i n } ( K , Q - 1 )$ nearest external hypotheses are retained, where $Q$ is the number of detected people. The person’s own token is included in the neighborhood.

A single multi-head attention and feed-forward block updates the representation from these sparse neighbors. Restricting attention to at most K external hypotheses reduces person-level attention from $O ( Q ^ { 2 } D )$ to $O ( Q ( K + 1 ) D )$ for token dimension $D .$ The module is strictly per-frame: it uses neither identities nor observations from previous frames and therefore does not turn MoSoLoco into a temporal localization or tracking method.

## C. 3D Localization and Training Objective

The prediction target is the center $\mathbf { c } = ( x _ { c } , y _ { c } , z _ { c } )$ of the annotated 3D pedestrian bounding box in the camera frame. Following MonoLoco [2], its position is parameterized by azimuth $\theta ,$ polar angle $\psi ,$ and radial distance $r = \| \mathbf { c } \| _ { 2 }$ . The regression head predicts $[ \theta , \psi , r , s , h , w , l ,$ sin α, cos α], where s parameterizes Laplacian uncertainty in distance, $( h , w , l )$ are normalized box dimensions, and α is the observation angle. Dimensions and orientation provide auxiliary supervision; only $( \theta , \psi , r )$ determine the evaluated center.

For matched prediction–target pairs $\mathcal { M } .$ , the localizer is trained with

$$
\mathcal { L } _ { \mathrm { l o c } } = \Big \langle \left| 1 - \frac { \hat { r } } { r } \right| e ^ { - \hat { s } } + \hat { s } + \sum _ { q \in \{ \theta , \psi , h , w , l \} } \left| \hat { q } - q \right|\tag{2}
$$

where $\hat { a }$ and $\hat { b }$ are the predicted sine and cosine components of the observation angle. All task terms have unit weight. The pose estimator remains frozen, and no additional posedetection or localization-ADE loss is used.

At inference, the camera-frame center is reconstructed as

$$
\hat { \mathbf { c } } = \left[ \hat { r } \sin \hat { \psi } \cos \hat { \theta } \quad \hat { r } \cos \hat { \psi } \quad \hat { r } \sin \hat { \psi } \sin \hat { \theta } \right] ^ { T } ,\tag{3}
$$

and transformed to the global frame using the known camera and robot extrinsics. These global ground-plane centers are used for localization evaluation and, without changing the localizer, as observations in the controlled forecasting study of Section IV-B.

## IV. EXPERIMENTS

The experiments are organized around the central claim that a pose detector contains useful localization cues beyond its final joint coordinates. JRDB provides the main localization comparison and component analyses, NuScenes evaluates the approach in a second camera and scene domain, and controlled forecasting and embedded-deployment studies examine downstream value and practical use.

## A. Localization Protocols

JRDB: The evaluation split contains 1,658 images and 3,422 annotated person-frame targets before filtering. MoSoLoco, MonoLoco [2], and MonoTransmotion [12] are trained for 100 epochs with batch size 16 and learning rate $5 \times 1 0 ^ { - 4 }$ . Detector hypotheses are filtered at confidence 0.20 and pose-box NMS IoU 0.50, then matched to annotations using Hungarian assignment with minimum OKS 0.50. ADE is the mean Euclidean error between predicted and annotated pedestrian centers in the global ground plane over accepted matches. Coverage is the fraction of annotated targets retained by this policy. DETRPose-S retains 3,162/3,422 targets (92.40%), while DETRPose-N retains 2,794/3,422 (81.65%).

NuScenes: NuScenes does not provide human keypoints, so preprocessing follows MonoTransmotion [12]. OpenPif-Paf [11] produces pseudo poses in COCO-17 order that are associated with pedestrian annotations. DETRPose-S is initialized from its official COCO weights and its Transformer pose head is adapted for 10 epochs on pseudo poses from the official training scenes. Each localizer is then trained for 100 epochs on the same training scenes with fresh optimizer and scheduler states. Evaluation uses the official validation scenes, comprising 2,496 valid person-frame targets with mean and median camera-to-person distances of 16.13 m and 14.95 m. Confidence, NMS, and OKS thresholds match JRDB.

External systems: UniDepth V2 ViT-L [15] and FCOS3D R101 [21] provide broader reference points rather than controlled localizer ablations. On JRDB, frozen UniDepth uses DETRPose-S hypotheses and the same matching policy. Its calibrated variant applies a fixed per-camera ray correction estimated on the training split and a +0.14 m validationselected depth offset; evaluation annotations are not used for calibration. FCOS3D is initialized from its official NuScenespretrained checkpoint and selected after one epoch of JRDB fine-tuning. On NuScenes, UniDepth again uses DETRPose-S hypotheses, whereas FCOS3D performs its own detection. Because detection, training, and depth-processing pipelines differ, these systems are reported as contextual accuracy–resource comparisons rather than parameter-matched ablations.

Resource measurements: FP32 inference costs are measured on an NVIDIA A100-SXM4-80GB at batch size one. Peak memory is the maximum torch.cuda.max\_memory\_allocated over three forward passes after two warm-up passes and includes parameters, inputs, activations, outputs, and temporary PyTorch allocations. GFLOPs are measured from the executed graph with CalFlops. Learned localizers use Q = 5 people per image, 17 joints, 128-dimensional detector latents, and K = 4 unless stated otherwise. DETRPose uses 640×640 input, and the shared detector cost is listed separately from each localizer.

## B. Controlled Forecasting Protocol

Trajectories are sampled at 2.5 Hz. Each example contains eight observation slots (3.2 s), of which two to eight may contain accepted localizer predictions, followed by 12 prediction frames (4.8 s). The Vanilla Transformer and OmniTraj [6] are each trained once using only ground-truth locations and identities, with batch size 16 and learning rate $1 0 ^ { - 4 }$ . At evaluation time, every localizer is paired with the same trained forecaster and the same ground-truth future trajectories.

Ground-truth identities are used only as an offline association control. In each observed frame, post-filtering pose detections are assigned one-to-one to ground-truth people by

TABLE I: Frozen DETRPose feature association with cameraaxis depth. PC is the strongest absolute Pearson-correlated component among the first 32; EV is explained variance. Partial r controls the coarse cues described in text.
<table><tr><td>Dataset</td><td></td><td>Matched PC (EV)</td><td></td><td>Pearson r Spearman ρ Partial r</td><td></td></tr><tr><td>JRDB</td><td>2,964</td><td>3 (11.2%)</td><td>0.652</td><td>0.643</td><td>0.168</td></tr><tr><td>NuScenes</td><td>2,252</td><td>1 (15.0%)</td><td>0.846</td><td>0.852</td><td>0.369</td></tr></table>

Hungarian keypoint matching, and only matches satisfying the OKS threshold are retained. Accepted detections inherit the corresponding identity for this experiment; rejected detections remain missing and are represented with a zero temporal mask. A window is scored when the final observed frame is valid and at least two of the eight observation slots are valid. Methods are compared on the intersection of retained windows within each detector condition.

Forecasting ADE is the mean ground-plane error over the 12 future steps and FDE is the error at the final future step. OmniTraj uses best-of-20 minimum ADE and the FDE from the same ADE-selected mode. The DETRPose-S common cohort retains 1,277 of 1,349 forecast-scored ground-truth-eligible trajectories (94.66%) across 631 cache scenes. DETRPose-N retains 1,054 trajectories across 568 cache scenes (78.13%). Coverage is reported with ADE/FDE because missed or rejected observations are not assigned zero displacement error.

For the filtered-history condition, only the observed prefix is processed with a second-order forward–backward Butterworth filter $( W _ { n } ~ = ~ 1 / 6 )$ . Missing slots are interpolated only for filtering and restored to missing afterward; masks and future coordinates remain unchanged. The filter therefore uses no information after the forecast origin.

## C. Embedded Deployment Protocol

DETRPose-S with the Transformer and MLP social localizers $( K = 4 )$ is evaluated on an NVIDIA Jetson AGX Xavier using JetPack 5.1.2, TensorRT 8.5.2, batch size one, and 640 × 640 inputs. The board runs in 30 W all-core mode with fixed CPU, GPU, and memory clocks. The baseline uses PyTorch FP32. The optimized implementation uses a mixed FP16/FP32 TensorRT detector, FP32 TensorRT localizers, reused TensorRT buffers, and an exact stable-greedy CUDA NMS kernel. Both runtimes use identical checkpoint weights and the same 1,658-image JRDB evaluation payload.

Latency is measured with CUDA synchronization around each forward pass; the first 10 images are excluded as warm-up. The timed region contains DETRPose inference, confidence filtering, NMS, and localization. Image loading and resizing, host-to-device transfer, and final metric postprocessing are excluded. Reported FPS therefore measures detector–localizer model inference rather than full sensor-tooutput system throughput.

## V. RESULTS

The results progress from analyzing the information used by MoSoLoco to evaluating the complete detector–localizer pipeline. The first analysis characterizes the depth-related structure retained in frozen pose-detector features. The JRDB study then quantifies the resulting localization gains, with component ablations separating the contributions of detector representations and multi-person context. NuScenes extends the evaluation to a different camera and scene domain, while the final experiments examine downstream trajectory forecasting and embedded inference efficiency.

TABLE II: JRDB localization. Params/Size report localizer parameters and FP32 storage; Peak/GFLOPs follow the resource protocol. Detector headers give shared front-end cost. Bold denotes best ADE.
<table><tr><td>Localization Model</td><td>Skeleton</td><td>Latent</td><td>Social</td><td>Params (M)</td><td>Size (MB)</td><td>Peak (MB)</td><td>GFLOPs</td><td>ADE↓</td></tr><tr><td colspan="9">End-to-end monocular 3D detector</td></tr><tr><td>FCOS3D R101 [21]</td><td>x</td><td>x</td><td>x</td><td>55.0</td><td>219.8</td><td>430.9</td><td>169.46</td><td>0.550</td></tr><tr><td colspan="9">Detector: DETRPose-S [10] (11.5 M, 46.9 MB; peak: 146.7 MB; 32.42 GFLOPs; COCO AP: 67.0; coverage: 92.40%)</td></tr><tr><td>MonoLoco [2]</td><td>√</td><td>x</td><td>x</td><td>0.41</td><td>1.6</td><td>11.3</td><td>0.004</td><td>0.326</td></tr><tr><td>MonoTransmotion [12]</td><td>√</td><td>x</td><td>x</td><td>5.29</td><td>21.2</td><td>30.8</td><td>0.053</td><td>0.322</td></tr><tr><td>UniDepth V2 ViT-L (uncalibrated) [15]</td><td>√</td><td>x</td><td>x</td><td>353.8</td><td>1415.3</td><td>2173.2</td><td>1612.09</td><td>0.765</td></tr><tr><td>UniDepth V2 ViT-L (calibrated) [15]</td><td>√</td><td>x</td><td>x</td><td>353.8</td><td>1415.3</td><td>2173.2</td><td>1612.09</td><td>0.622</td></tr><tr><td>MoSoLoco-MLP (latent-only)</td><td>x</td><td>√</td><td>x</td><td>0.43</td><td>1.7</td><td>11.4</td><td>0.004</td><td>0.289</td></tr><tr><td>MoSoLoco-MLP</td><td>√</td><td>√</td><td>x</td><td>0.44</td><td>1.8</td><td>11.4</td><td>0.004</td><td>0.266</td></tr><tr><td>MoSoLoco-MLP</td><td>√</td><td>√</td><td>√</td><td>0.67</td><td>2.7</td><td>12.4</td><td>0.008</td><td>0.254</td></tr><tr><td>MoSoLoco-Trans (latent-only)</td><td>x</td><td>√</td><td>x</td><td>2.05</td><td>8.2</td><td>18.6</td><td>0.107</td><td>0.267</td></tr><tr><td>MoSoLoco-Trans</td><td>√</td><td>√</td><td>x</td><td>4.42</td><td>17.7</td><td>29.0</td><td>0.465</td><td>0.262</td></tr><tr><td>MoSoLoco-Trans</td><td>√</td><td>√</td><td>√</td><td>4.62</td><td>18.5</td><td>29.8</td><td>0.468</td><td>0.248</td></tr><tr><td colspan="9">Detector: DETRPose-N [10] (4.1 M, 16.8 MB; peak: 73.3 MB; 9.14 GFLOPs; COCO AP: 57.2; coverage: 81.65%)</td></tr><tr><td>MonoLoco [2]</td><td>√</td><td>x</td><td>x</td><td>0.41</td><td>1.6</td><td>11.3</td><td>0.004</td><td>0.333</td></tr><tr><td>MonoTransmotion [12]</td><td>√</td><td>x</td><td>x</td><td>5.29</td><td>21.2</td><td>30.8</td><td>0.053</td><td>0.343</td></tr><tr><td colspan="9">MoSoLoco-MLP (latent-only)</td></tr><tr><td>MoSoLoco-MLP</td><td>x √</td><td>V √</td><td>x x</td><td>0.43 0.44</td><td>1.7 1.8</td><td>11.4 11.4</td><td>0.004 0.004</td><td>0.351 0.297</td></tr><tr><td>MoSoLoco-MLP</td><td>√</td><td>√</td><td>√</td><td>0.67</td><td>2.7</td><td>12.4</td><td>0.008</td><td>0.276</td></tr><tr><td>MoSoLoco-Trans (latent-only)</td><td>x</td><td>√</td><td>x</td><td>2.05</td><td>8.2</td><td>18.6</td><td>0.107</td><td>0.347</td></tr><tr><td>MoSoLoco-Trans</td><td>√</td><td>√</td><td>x</td><td>4.42</td><td>17.7</td><td>29.0</td><td>0.465</td><td>0.273</td></tr><tr><td>MoSoLoco-Trans</td><td>√</td><td>√</td><td>√</td><td>4.62</td><td>18.5</td><td>29.8</td><td>0.468</td><td>0.265</td></tr></table>

## A. Depth-Related Structure in Frozen Pose-Detector Features

A descriptive probe tests whether frozen DETRPose features retain distance-related information before localization. The 17 decoder embeddings of each matched detection are meanpooled to 256 dimensions; PCA is applied and correlations with camera-axis depth are evaluated over the first 32 components. Table I reports the component with the strongest absolute Pearson correlation. Partial r additionally regresses both depth and that component on coarse skeleton/camera cues (size, spread, position, aspect ratio, confidence, normalized focal length, and scale-derived terms).

The raw correlations are substantial on JRDB (r = 0.652, $\rho = 0 . 6 4 3 )$ and NuScenes (r = 0.846, ρ = 0.852), but partial r falls to 0.168 and 0.369. Thus, geometry explains much of the association, while a positive residual association remains beyond the controlled coarse cues. No localizer outputs enter this probe; subsequent ablations test task utility.

## B. JRDB Localization

Table II gives the main JRDB comparison. With DETRPose-S, social MoSoLoco-Trans reaches 0.248 m ADE, a 23.9% reduction from MonoLoco (0.326 m) and 23.0% from Mono-Transmotion (0.322 m); with DETRPose-N, it reaches 0.265 m versus 0.333/0.343 m. The social MLP obtains 0.254/0.276 m for S/N. FCOS3D and UniDepth use different pipelines and are contextual references: FCOS3D obtains 0.550 m over 3,271/3,422 targets (95.59%), while UniDepth obtains 0.765 m uncalibrated and 0.622 m calibrated on 2,964/3,422 targets (86.62%).

TABLE III: Maximum external-neighbor ablation for MoSoLoco-Trans.
<table><tr><td colspan="2">Detector: DETRPose-S JRDB coverage: 92.40%; NuScenes coverage: ~90.1%</td></tr><tr><td>Social setting JRDB ADE↓</td><td>NuScenes ADE↓</td></tr><tr><td>Without social 0.262</td><td>1.427</td></tr><tr><td>K = 1 0.252</td><td>1.146</td></tr><tr><td>K = 2 0.261</td><td>1.185</td></tr><tr><td>K = 4 0.248 K = 6</td><td>1.130</td></tr><tr><td>0.252</td><td>1.181</td></tr></table>

## C. Component Analysis

Pose-detector features: Without social context, geometry+latent MLP reduces ADE from MonoLoco’s 0.326 to 0.266 m with DETRPose-S and from 0.333 to 0.297 m with DETRPose-N. Latent-only MLP/Transformer models reach 0.289/0.267 m (S) and 0.351/0.347 m (N); restoring skeleton geometry gives 0.266/0.262 m and 0.297/0.273 m, respectively.

Same-frame context: Social attention improves all four detector–architecture point estimates: MLP changes from 0.266 to 0.254 m (S) and 0.297 to 0.276 m (N), while Transformer changes from 0.262 to 0.248 m and 0.273 to 0.265 m.

Table III shows a non-monotonic effect of neighborhood size; every tested K improves on no social context, and

![](images/51b8da6fa11d3fb43e114075bd1ba49e18b2cc1922b58561c7a871049b7bb6c9.jpg)  
(a) Social refinement degrades the estimate.

![](images/e2fdeb47f2d69529ec8ac52e4f57dc34aea9900ccc0b7dd41f083773fa92108e.jpg)  
(b) Social refinement improves the estimate.  
<sup>Atention-colored</sup> <sup>overlay</sup>Fig. 3: Qualitative social refinement. The anchor is shown with attended neighbors; color denotes relative attention weight. (a) A compact, strongly articulated anchor overlaps a bicycle while neighbors are upright pedestrians, and the social variant <sup>Weights</sup> <sup>are</sup> <sup>social-at ention</sup> <sup>softmax</sup> <sup>probabilities</sup> <sup>averaged</sup> <sup>over</sup> <sup>heads;</sup> <sup>they</sup> <sup>describe</sup> <sup>at ention</sup> <sup>al ocation,</sup> <sup>not</sup> <sup>causal</sup> <sup>importance.</sup>is less accurate. (b) For an anchor truncated by the image boundary, social context improves localization. Attention weights indicate allocation, not causal importance.

TABLE IV: NuScenes 3D pedestrian-center localization. Params/Size report localizer FP32 cost; Peak/GFLOPs follow the resource protocol. FCOS3D performs its own detection; social models use K = 4.
<table><tr><td>Localization Model</td><td>Social</td><td>Params (M)</td><td>Size (MB)</td><td>Peak (MB)</td><td>GFLOPs</td><td>ADE↓ (m)</td></tr><tr><td colspan="7">Detector: DETRPose-S (11.5 M params; 46.9 MB; 32.42 GFLOPs) Peak: 146.7 MB; coverage: 90.14%; FCOS3D uses its own detector</td></tr><tr><td>MonoLoco [2]</td><td>x</td><td>0.41</td><td>1.6</td><td>11.3</td><td>0.004</td><td>1.450</td></tr><tr><td>MonoTransmotion [12]</td><td>x</td><td>5.29</td><td>21.2</td><td>30.8</td><td>0.053</td><td>1.407</td></tr><tr><td>UniDepth V2 ViT-L [15]</td><td>x</td><td>353.8</td><td>1415.3</td><td>2654.1</td><td>2650.56</td><td>0.935</td></tr><tr><td>FCOS3D R101 [21]</td><td>x</td><td>55.0</td><td>219.8</td><td>621.3</td><td>682.50</td><td>0.901</td></tr><tr><td>MoSoLoco-MLP</td><td>√</td><td>0.67</td><td>2.7</td><td>12.4</td><td>0.008</td><td>1.219</td></tr><tr><td>MoSoLoco-Trans</td><td>√</td><td>4.62</td><td>18.5</td><td>29.8</td><td>0.468</td><td>1.130</td></tr></table>

K = 4 is best on both datasets. Because K is an upper bound, each frame uses min(K, Q − 1) external neighbors. Figure 3 illustrates complementary cases: context is less reliable for a strongly articulated anchor that differs from upright neighbors, but helps when the anchor is heavily truncated. These examples motivate compatibility-aware aggregation while avoiding a causal interpretation of attention weights.

## D. NuScenes Localization

On NuScenes (Table IV), social MoSoLoco-MLP/Trans reach 1.219/1.130 m ADE versus 1.450 m for MonoLoco and 1.407 m for MonoTransmotion, preserving the posebased advantage after dataset-specific pose adaptation and localizer training. UniDepth and FCOS3D obtain lower ADE (0.935/0.901 m) over 2,253/2,372 targets (90.26/95.03% coverage), but at much larger compute: including DETRPose-S, complete MoSoLoco-MLP/Trans contain 12.17/16.12 M parameters versus 365.3 M for UniDepth+DETRPose-S and 55.0 M for FCOS3D. Their different pipelines make this an accuracy–resource comparison rather than a controlled ablation.

![](images/55aeb454390033b2f9c6991b4228805c3b03811194ae18f62a2d1deae09f3f23.jpg)  
Fig. 4: JRDB forecasting example. Dashed trajectories are raw localization observations; dotted trajectories apply the futureblind filter only to observed history.

## E. Downstream Trajectory Forecasting

Table V shows lower forecasting ADE from MoSoLoco observations than from both pose-based baselines for both detectors and forecasters. For DETRPose-S raw histories, MoSoLoco-Trans gives 0.250 m Vanilla ADE and 0.219 m OmniTraj ADE versus 0.283/0.250 m for MonoLoco and 0.280/0.240 m for MonoTransmotion. With DETRPose-N, the corresponding MoSoLoco-Trans values are 0.275/0.223 m versus 0.310/0.264 and 0.337/0.272 m. Filtering reduces ADE for every predicted-history row, while the GT-history references remain substantially lower.

## F. Embedded Deployment

TensorRT raises synchronized model-inference throughput from 6.31 to 23.00 FPS for MoSoLoco-Trans and from 6.97 to 24.15 FPS for MoSoLoco-MLP $( 3 . 6 5 { \times } / 3 . 4 7 { \times } )$ , with ADE unchanged at 0.248/0.254 m. Optimized mean latencies are 43.48/41.40 ms (95th percentile: 50.21/48.04 ms); CUDA NMS preserves query-selection masks and coverage on all 1,658 images.

TABLE V: Controlled JRDB trajectory forecasting. MoSoLoco uses K = 4. OmniTraj reports best-of-20 minADE with FDE from the same mode. Each detector block uses its common retained cohort; filtering uses only observed history.
<table><tr><td rowspan="3">Localization Model</td><td colspan="4">Vanilla Transformer</td><td colspan="4">OmniTraj [6]</td></tr><tr><td colspan="2">Raw Forecast</td><td colspan="2">Filtered</td><td colspan="2">Raw Forecast</td><td colspan="2">Filtered</td></tr><tr><td>ADE↓</td><td>FDE↓</td><td>ADE↓</td><td>FDE↓</td><td>ADE↓</td><td>FDE↓</td><td>ADE↓</td><td>FDE↓</td></tr><tr><td colspan="9">Detector: DETRPose-S (Small)</td></tr><tr><td>GT-History Input</td><td>0.082</td><td>0.144</td><td></td><td>一</td><td>0.059</td><td>0.116</td><td></td><td></td></tr><tr><td>MonoLoco [2]</td><td>0.283</td><td>0.364</td><td>0.244</td><td>0.304</td><td>0.250</td><td>0.271</td><td>0.218</td><td>0.226</td></tr><tr><td>MonoTransmotion [12]</td><td>0.280</td><td>0.359</td><td>0.243</td><td>0.300</td><td>0.240</td><td>0.260</td><td>0.207</td><td>0.208</td></tr><tr><td>MoSoLoco-MLP  $( K = 4 )$ </td><td>0.259</td><td>0.347</td><td>0.215</td><td>0.278</td><td>0.226</td><td>0.265</td><td>0.182</td><td>0.201</td></tr><tr><td>MoSoLoco-Trans  $( K = 4 )$ </td><td>0.250</td><td>0.336</td><td>0.206</td><td>0.267</td><td>0.219</td><td>0.251</td><td>0.181</td><td>0.207</td></tr><tr><td colspan="9">Detector: DETRPose-N (Nano)</td></tr><tr><td>GT-History Input</td><td>0.082</td><td>0.146</td><td>一</td><td>一</td><td>0.062</td><td>0.122</td><td>一</td><td>一</td></tr><tr><td>MonoLoco [2]</td><td>0.310</td><td>0.411</td><td>0.256</td><td>0.318</td><td>0.264</td><td>0.282</td><td>0.221</td><td>0.224</td></tr><tr><td>MonoTransmotion [12]</td><td>0.337</td><td>0.448</td><td>0.276</td><td>0.346</td><td>0.272</td><td>0.281</td><td>0.238</td><td>0.243</td></tr><tr><td>MoSoLoco-MLP  $( K = 4 )$ </td><td>0.280</td><td>0.377</td><td>0.230</td><td>0.295</td><td>0.238</td><td>0.267</td><td>0.199</td><td>0.222</td></tr><tr><td>MoSoLoco-Trans  $( K = 4 )$ </td><td>0.275</td><td>0.382</td><td>0.216</td><td>0.286</td><td>0.223</td><td>0.259</td><td>0.187</td><td>0.215</td></tr></table>

TABLE VI: Jetson AGX Xavier model-inference results on JRDB.
<table><tr><td>Model</td><td>Runtime</td><td>ADE↓</td><td>FPS ↑</td></tr><tr><td>MoSoLoco-Trans</td><td>Baseline Optimized</td><td>0.248 0.248</td><td>6.31 23.00</td></tr><tr><td></td><td>Baseline</td><td>0.254</td><td>6.97</td></tr><tr><td>MoSoLoco-MLP</td><td>Optimized</td><td>0.254</td><td>24.15</td></tr></table>

## VI. DISCUSSION

The central result is that intermediate pose-detector representations improve a lightweight pose-based monocular localizer when fused with camera-normalized skeleton geometry. The gain appears with both DETRPose-S and DETRPose-N on JRDB and remains present on NuScenes after dataset-specific pose adaptation and localizer training. The latent-only ablation is also informative: detector features can support localization by themselves with DETRPose-S, but not consistently with DETRPose-N, while the combined models outperform their latent-only counterparts for both detector capacities. Together with the feature probe, these results are consistent with the reused representations carrying localization-relevant information that complements the final 2D keypoints; they do not identify which visual cues are responsible.

Same-frame multi-person attention provides an additional but smaller refinement. Its point estimate improves for both MLP and Transformer localizers and for every tested value of K relative to the no-social setting. The non-monotonic neighborhood ablation and the qualitative cases in Fig. 3 indicate, however, that neighboring hypotheses are not uniformly useful.

The failure example is consistent with context becoming less reliable when the anchor differs strongly from surrounding upright pedestrians, whereas context can help when the anchor itself is heavily truncated. This motivates compatibility-aware aggregation using pose, visibility, or activity-related cues rather than assuming that more neighbors will always help.

The forecasting experiment evaluates downstream utility under deliberately controlled association. MoSoLoco observations yield lower forecasting ADE than the pose-based localization baselines for both a deterministic Transformer and OmniTraj, while the ground-truth-history rows remain substantially below every predicted-history condition. The result is consistent with localization quality being an important source of downstream forecasting error in the evaluated setting. It is not an end-to-end tracking result: identities are supplied through ground-truth association, failed detections remain masked, and forecasting metrics are computed on a common retained cohort. Per-frame and forecast-window coverage therefore remain necessary companion metrics when interpreting ADE and FDE.

Several limitations bound the claims. First, NuScenes is not a zero-shot transfer experiment because both the pose front end and localizers use the NuScenes training scenes before validation. Second, UniDepth and FCOS3D follow different detection and processing pipelines and obtain lower ADE than MoSoLoco on NuScenes; those rows therefore quantify an accuracy–compute trade-off rather than a controlled accuracy comparison. Third, the tables report point estimates without repeated-seed uncertainty, so small differences, including the few-centimeter social gains and the gap between the two MoSoLoco backbones, should be interpreted cautiously. Finally, Jetson measurements exclude image preprocessing, hostto-device transfer, and final post-processing. The reported 23– 24 FPS is detector–localizer model-inference throughput, not full sensor-to-output latency.

## VII. CONCLUSION

This work introduced MoSoLoco, a lightweight per-frame approach to monocular 3D human localization that reuses intermediate visual representations from a frozen 2D pose detector. MoSoLoco combines these representations with skeleton geometry, while a sparse same-frame interaction module uses nearby detections to further refine each estimate. The feature analysis showed that the detector representations retain depth-related structure beyond coarse skeleton cues, providing empirical support for their use in localization.

Across JRDB and NuScenes, this design consistently improved over the evaluated keypoint-based localizers across detector capacities and fusion architectures. The best JRDB configuration reduced ADE by 23.9% and achieved accuracy comparable to much larger monocular depth and 3D detection systems. More accurate locations also reduced error in the controlled trajectory-forecasting study, while TensorRT deployment reached 23.0–24.2 FPS on a Jetson AGX Xavier without changing localization accuracy. Together, these results show that representations already computed for pose estimation can improve the accuracy–efficiency balance of monocular human localization. Extending the approach with tracking, studying cross-domain transfer, and measuring complete sensor-tooutput latency remain important directions.

## VIII. ACKNOWLEDGMENT

The authors gratefully acknowledge the financial support provided by the Swiss National Science Foundation (SNSF) through grant no. 10003100.

## REFERENCES

[1] Alexandre Alahi et al. “Social lstm: Human trajectory prediction in crowded spaces”. In: CVPR. 2016.

[2] Lorenzo Bertoni, Sven Kreiss, and Alexandre Alahi. “Monoloco: Monocular 3d pedestrian localization and uncertainty estimation”. In: ICCV. 2019.

[3] Lorenzo Bertoni et al. “MonStereo: When Monocular and Stereo Meet at the Tail of 3D Human Localization”. In: IEEE International Conference on Robotics and Automation (ICRA). 2021, pp. 5126–5132.

[4] Holger Caesar et al. “nuscenes: A multimodal dataset for autonomous driving”. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2020, pp. 11621–11631.

[5] Hao Chen et al. “3D Pedestrian Localization Fusing via Monocular Camera”. In: Journal of Visual Communication and Image Representation 95 (2023), p. 103871. DOI: 10.1016/j.jvcir.2023.103871.

[6] Yang Gao et al. “OmniTraj: Pre-Training on Heterogeneous Data for Adaptive and Zero-Shot Human Trajectory Prediction”. In: arXiv:2507.23657 (2025).

[7] Agrim Gupta et al. “Social gan: Socially acceptable trajectories with generative adversarial networks”. In: CVPR. 2018.

[8] Kuan-Chih Huang et al. “Monodtr: Monocular 3d object detection with depth-aware transformer”. In: CVPR. 2022.

[9] Boris Ivanovic et al. “Multimodal deep generative models for trajectory prediction: A conditional variational autoencoder approach”. In: RAL (2020).

[10] Sebastian Janampa and Marios Pattichis. “DETRPose: Real-time end-to-end transformer model for multiperson pose estimation”. In: arXiv:2506.13027 (2025).

[11] Sven Kreiss, Lorenzo Bertoni, and Alexandre Alahi. “Pifpaf: Composite fields for human pose estimation”. In: CVPR. 2019.

[12] Po-Chien Luan et al. “Unified Human Localization and Trajectory Prediction with Monocular Vision”. In: arXiv:2503.03535 (2025).

[13] Roberto Martin-Martin et al. “Jrdb: A dataset and benchmark of egocentric robot visual perception of humans in built environments”. In: TPAMI (2021).

[14] Diego Paez-Granados et al. “3d point cloud and rgbd of pedestrians in robot crowd navigation: Detection and tracking”. In: IEEE DataPort 12 (2021).

[15] Luigi Piccinelli et al. “Unidepthv2: Universal monocular metric depth estimation made simpler”. In: IEEE Transactions on Pattern Analysis and Machine Intelligence (2025).

[16] Mohammad Masoud Rahimi et al. “Pose-Aware Monocular Localization of Occluded Pedestrians in 3D Scene Space”. In: ISPRS Open Journal of Photogrammetry and Remote Sensing 2 (2021), p. 100006. DOI: 10.1016/ j.ophoto.2021.100006.

[17] Haziq Razali, Taylor Mordan, and Alexandre Alahi. “Pedestrian intention prediction: A convolutional bottom-up multi-task approach”. In: Transportation research part C: emerging technologies 130 (2021), p. 103259.

[18] Saeed Saadatnejad et al. “Social-Transmotion: Promptable Human Trajectory Prediction”. In: ICLR. 2024.

[19] Edward Schmerling et al. “Multimodal probabilistic model-based planning for human-robot interaction”. In: ICRA. 2018.

[20] Li Wang et al. “Depth-conditioned dynamic message propagation for monocular 3d object detection”. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2021, pp. 454–463.

[21] Tai Wang et al. “Fcos3d: Fully convolutional one-stage monocular 3d object detection”. In: ICCV. 2021.

[22] Zizhang Wu et al. “Monopgc: Monocular 3d object detection with pixel geometry contexts”. In: 2023 IEEE International Conference on Robotics and Automation (ICRA). IEEE. 2023, pp. 4842–4849.

[23] Yihong Xu et al. “Towards motion forecasting with real-world perception inputs: are end-to-end approaches competitive?” In: ICRA. 2024.