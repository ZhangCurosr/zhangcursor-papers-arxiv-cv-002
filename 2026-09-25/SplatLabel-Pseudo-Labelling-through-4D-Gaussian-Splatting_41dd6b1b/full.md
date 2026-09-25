# SplatLabel: Pseudo-Labelling through 4D Gaussian Splatting

Nitya Nanvani<sup>1,2</sup>

Andras Palffy<sup>1</sup>

Holger Caesar<sup>2</sup>

![](images/a1c974e677404f621472da0ade1e1154253be6adfb3650bd85e01334cbb00eea.jpg)  
Fig. 1. SplatLabel Architecture Overview. SplatLabel optimizes a 4D Gaussian representation governed by an explicit temporal manifold. (Left) 2D images and LiDAR point clouds are distilled to structurally ground the scene. (Middle) Dynamics are explicitly modeled by parameterizing the trajectories and lifespans of individual primitives. (Right) The unified representation yields high-confidence 3D pseudo-labels for multiple downstream tasks.

Abstract— While 2D Vision Foundation Models offer a pathway to automate 3D semantic pseudo-labelling, translating these priors into robust 3D representations typically requires complex heuristics or multi-model ensembles. We introduce SplatLabel, an automated pipeline that leverages a 4D Gaussian representation to extract LiDAR segmentation with predictive confidence, as well as semantic occupancy grids at arbitrary voxel resolutions. At its core, SplatLabel handles dynamic environments through an explicit temporal manifold that models the trajectories and lifespans of individual 3D primitives. This allows the system to accurately track moving actors and strictly define when objects appear and disappear, completely eliminating the need for pre-annotated 3D bounding boxes. To robustly support this dynamic tracking, the representation is grounded by structural and semantic priors: we guide scene geometry in unobserved regions by integrating 360-degree LiDAR via virtual depth maps, and rather than relying on domain-specific prompt engineering, we directly distill continuous soft probabilities from 2D models to inherently resolve semantic ambiguities over time and space. Finally, to accurately reflect the realworld trade-off between precision and recall, we reframe pseudo-label evaluation as a selective classification task using a generalized risk-recall metric. Experiments on SemanticKITTI demonstrate that SplatLabel consistently outperforms state-ofthe-art baselines across multiple recall levels, establishing a highly robust framework for both 3D LiDAR segmentation and occupancy prediction.

## I. INTRODUCTION

Comprehensive 3D semantic perception is a fundamental prerequisite for autonomous systems [1], [2]. However, the manual annotation required for training robust models is prohibitively expensive, prompting the use of 2D Vision Foundation Models (VFMs) to automatically generate 3D pseudolabels [3], [4], [5], [6]. Concurrently, 3D Gaussian Splatting (3DGS) has emerged as a powerful explicit radiance field representation [7]. Yet, despite its rapid adoption, 3DGS optimization remains overwhelmingly focused on novel view synthesis, leaving its potential as a continuous backbone for spatial pseudo-labelling underutilized.

Current automated 3D pseudo-labelling pipelines face critical bottlenecks when moving beyond static environments. Most notably, modeling dynamic scenes typically requires pre-annotated 3D bounding boxes to cleanly separate moving actors from static backgrounds [8], [9], creating a circular dependency that defeats the purpose of annotation-free pseudo-labelling. Furthermore, extracting consistent semantic features and integrating 360-degree LiDAR geometry to support these scenes demands complex heuristics, multimodel ensembles, or computationally heavy spherical rasterization to counteract temporal inconsistencies and visual occlusion [4], [6], [10], [11]. Finally, existing baselines evaluate pseudo-label quality strictly on successfully labeled points, fundamentally ignoring the crucial trade-off between arbitrary recall levels and overall label accuracy.

To solve these challenges, we propose SplatLabel, an automated 4D Gaussian Splatting pipeline designed specifically for 3D semantic pseudo-labelling and volumetric occupancy prediction. Rather than treating motion as a post-processing step or relying on implicit global deformations, SplatLabel unifies explicit spatial representation, structural densification, and kinematic tracking into a single continuous optimization framework driven by a fully parameterized temporal manifold. By doing so, we address the core limitations of existing pipelines and establish a robust, annotation-free foundation for dynamic scene understanding. Our main contributions are summarized as follows:

![](images/b282b0d3eafd9dfa8cedc1e5753e96086583bd5ae88df8d1b8bc6099a5484f05.jpg)  
Fig. 2. Qualitative Results on SemanticKITTI [1]. Top two rows: Bird’s-Eye View (BEV) LiDAR segmentation. Bottom two rows: Semantic occupancy grids. Left: Reference RGB images. Middle: Ground truth annotations. Right: Our complete 360-degree predictions. Crucially, our predictions encompass all points outside the camera’s Field of View (FOV). Furthermore, our explicit temporal tracking resolves the artifactual moving-object “trails” present in the occupancy ground truth (visible in row 3), cleanly separating dynamic actors from the static background.

Explicit Temporal Manifold for Dynamic Scenes: We parameterize motion and temporal existence boundaries as intrinsic properties of individual 3D Gaussians. By utilizing a Dual Deformation Kernel [12] coupled with a Gated Deformation Mask and explicit temporal lifespans, SplatLabel achieves robust, object-centric tracking. This naturally separates moving actors from the static background across time without the need for pre-annotated 3D bounding boxes [8], [9] or unpredictable black-box motion models [13].

Spatiotemporal Distillation and Geometric Priors: We fundamentally advance 3DGS optimization by dynamically regulating primitive capacity through LiDAR priors. By backpropagating continuous soft semantics directly from 2D VFMs [14], our continuous optimization inherently filters out single-frame noise and resolves categorical ambiguities through spatiotemporal consensus, eliminating the need for domain-specific prompt engineering [4], [6].

Pseudo-Labelling as Selective Classification: We establish a mathematically grounded framework to evaluate pseudolabelling systems. By reformulating evaluation as a selective classification task using the class-averaged Area Under the Generalized Risk-Coverage (AUGRC) [15] metric, we directly capture the risk of injecting noisy labels into downstream training, revealing the true performance-recall tradeoffs ignored by standard baselines [4], [6].

## II. RELATED WORK

## A. Spatiotemporal Semantic Distillation in 3DGS

3DGS models scenes using an explicit radiance field of anisotropic Gaussian primitives [7]. Typically initialized from Structure-from-Motion (SfM) point clouds [16], each discrete primitive is parameterized by a 3D center, covariance, opacity, and spherical harmonics [7]. While 3DGS achieves highly efficient rendering, its original photometric gradient-based densification heuristics are extremely fragile [7], [17]. Crucially, introducing semantic and depth supervision fundamentally alters these loss scales, which vary significantly depending on the total number of classes or the specific foundation model utilized, causing standard densification heuristics to struggle. To address this, recent work reformulates 3DGS optimization as a Markov Chain Monte Carlo (MCMC) sampling process via Stochastic Gradient Langevin Dynamics (SGLD) [17], [18]. This approach naturally explores the scene landscape and manages primitive capacity irrespective of disrupted gradient scales [17].

Beyond geometry, Gaussians explicitly store highdimensional semantic features without structural modifications [19], [7], enabling memory-efficient semantic occupancy prediction [20], [21] and sensor fusion [22], [23]. Concurrently, 3D semantic pseudo-labelling has advanced by transferring knowledge from 2D VFMs to 3D spaces [3], [4], [5], [6]. However, extracting consistent semantics typically demands complex prompt formulation or multi-model ensembles to counteract temporal inconsistencies and singleframe label flickering [4], [6]. SplatLabel overcomes this by directly distilling continuous soft logits into a 4D Gaussian representation, natively converging on a unified semantic identity across time and space.

## B. LiDAR-Guided Geometric Optimization

While VFMs provide rich 2D semantic priors, accurately lifting these labels into 3D requires robust underlying geometry. The integration of LiDAR into 3DGS typically enforces structural constraints through pseudo-depth projection onto camera planes [24], [25], but frequently discards LiDAR measurements outside the visual frustum [26], [11], [24]. To utilize 360-degree sweeps, recent methods develop explicit LiDAR sensor models utilizing specialized spherical rasterization [26], [11], which introduce significant computational complexity. Crucially, prior approaches treat LiDAR strictly as a post-rendering supervision signal rather than an explicit densification trigger. To fully utilize 360-degree sweeps without heavy spherical rasterizers, SplatLabel introduces Virtual Depth Maps and a Geometry-Guided sampler that actively spawns primitives in structurally inconsistent regions.

## C. Dynamic Scene Modeling and the Temporal Manifold

Despite the geometric fidelity achieved in static scenes, real-world environments are inherently dynamic. Importantly, the vast majority of continuous 4DGS frameworks are designed exclusively for novel view synthesis [8], [9], [12], [13], [24], [26], [27], [28], largely overlooking how these rich spatiotemporal representations can be repurposed for dense scene annotation.

Supervised frameworks achieve high geometric fidelity by separating moving foregrounds from static backgrounds [8], [9]. Annotation-free approaches utilize either implicit global environmental deformations [29], [13], or explicit representations that treat motion as an intrinsic property of the 3D primitives [28], [12], [27]. However, while implicit methods like CODA-4DGS [13] effectively model dynamic context, they allow semantic features to deform and alter across time. This continuous semantic morphing contradicts physical reality for label extraction, where an object’s core semantic identity must remain firmly grounded and constant. Conversely, while explicit models provide the object-centric tracking required for pseudo-labelling, isolating dynamic actors typically requires pre-annotated 3D bounding boxes, creating a circular dependency for self-supervised systems. SplatLabel resolves this through a bounding-box-free explicit Dual Deformation Kernel [12] and a fully parameterized temporal manifold, cleanly separating moving actors from static backgrounds while strictly preserving their semantic identities.

## D. Selective Classification for Pseudo-Label Evaluation

Even with robust explicit tracking, self-supervised pipelines inevitably generate noisy predictions. Consequently, evaluating pseudo-label quality is uniquely challenging. Existing baselines conventionally measure accuracy only on the fraction of points they manage to annotate, failing to account for the widely varying and arbitrary recall levels across different methods. To address this, we draw from the broader machine learning literature, where selective classification is utilized to explicitly evaluate the trade-off between coverage (recall) and accuracy [30], [31]. However, simply adopting standard selective metrics like the Area Under the Risk-Coverage curve (AURC) [31] introduces new issues: they calculate risk strictly over accepted predictions, which disproportionately penalizes high-confidence failures [15] and frequently mask bias by disproportionately rejecting minority classes under severe dataset imbalance [32]. To overcome both the limitations of standard pseudo-label evaluation and the flaws of naive selective metrics, SplatLabel reformulates LiDAR pseudo-labelling as a selective classification task utilizing a class-averaged Area Under the Generalized Risk-Coverage (AUGRC) metric [15], providing a holistic, bias-aware measure of label reliability.

![](images/5e8fc2590ccd36c35df974e11b727f18aadb5aed9e983ee97d9ea78c185dcfde.jpg)  
Fig. 3. Performance-recall trade-off on SemanticKITTI [1]. SplatLabel provides a continuous performance-recall profile, displaying the mean mIoU and standard deviation across five random seeds. Our method’s tight variance and consistent upper bound strictly dominate the single-operating-point baselines, LeAP [4] and UniLiPs [6].

## III. METHODOLOGY

## A. Problem Formulation and Overview

Given a continuous sequence of LiDAR point clouds and 2D images, our objective is to generate dense 3D semantic pseudo-labels and volumetric semantic occupancy grids. The fundamental challenge in continuous dynamic scene reconstruction is accurately tracking moving actors while ensuring the static environment remains stable over time.

To resolve this, we formulate the scene as a continuous 4D representation strictly governed by an explicit temporal manifold. As illustrated in Figure 1, our pipeline centers on this manifold by treating motion and temporal lifespan as intrinsic properties of individual Gaussian primitives. While multimodal inputs (LiDAR and 2D semantics) are distilled to manage primitive capacity, the core of our approach lies in explicit kinematic tracking and temporal existence parameterization. This formulation allows the system to accurately define when and where objects appear, seamlessly separating dynamic actors from static backgrounds to achieve a globally consistent 4D reconstruction.

## B. Spatiotemporal Distillation and Geometric Priors

To build a robust foundation for our temporal manifold, we must first accurately ground our Gaussians in both semantic and geometric reality. Standard photometric gradient-based densification heuristics fail under the altered loss scales introduced by semantic and depth supervision. Therefore, we abandon fragile densification heuristics and build upon stochastic state transitions (MCMC via SGLD) [17] to dynamically regulate primitive capacity. We extend this framework to natively incorporate multimodal 2D and 3D priors.

1) Continuous Semantic Distillation: To overcome domain-specific prompt limitations, we directly extract raw logits from Segment Anything Model (SAM) 3 [14] by querying base dataset labels. This bypasses complex prompt manipulation, yielding dense, continuous soft labels. By backpropagating the semantic loss directly into the Gaussian parameters and aggregating gradients across the entire spatiotemporal sequence, our continuous optimization intrinsically smooths frame-to-frame label flickering and resolves categorical ambiguities (e.g., ‘car’ versus ‘vehicle’).

2) Omnidirectional Geometric Supervision: Standard camera-frame depth supervision discards crucial 360-degree LiDAR geometry. To supervise visually occluded regions without abandoning the efficient pinhole rasterizer, we introduce Virtual Depth Maps. By explicitly placing artificial cameras in unobserved areas, we project the full LiDAR point cloud to generate virtual depth targets. Furthermore, we integrate these raw LiDAR depth cues directly into our density controller as structural sampling priors. When the system detects regions exhibiting high structural inconsistency between the Gaussian distribution and LiDAR data, it actively prioritizes spawning and adjusting primitives directly within those areas.

## C. Temporal Manifold Parameterization

To accurately model dynamic environments without decoupling moving actors from global environmental deformations, we parameterize motion and temporal existence as an explicit temporal manifold intrinsic to individual Gaussian primitives. This unifies spatial tracking and temporal boundaries into a single continuous representation.

1) Dual Deformation Kernel: Trajectory changes are captured via a Dual Deformation Kernel [12], combining an ndegree polynomial framework with an n-degree Fourier noise network to model both smooth linear paths and complex, high-frequency variations:

$$
\Delta ( t ) = \sum _ { k = 1 } ^ { n } \alpha _ { k } t ^ { k } + \sum _ { m = 1 } ^ { n } \beta _ { m } \sin ( 2 \pi m t ) + \gamma _ { m } \cos ( 2 \pi m t )\tag{1}
$$

2) Gated Deformation Mask: Uniformly applying explicit deformation destabilizes static backgrounds. To isolate static geometry, we introduce a learned Sigmoid-Gated Deformation Mask [13], which clamps deformation kernels to zero for background structures. Furthermore, moving objects with poorly converged trajectories often suffer low opacity and are consequently pruned. To address this, we integrate the mask directly into our sampling prior as a dynamicweighted relocalization mechanism. By scaling primitive respawn probabilities with this motion mask, the state transition backend actively prioritizes sampling along dynamic trajectories, ensuring robust reconstruction of moving actors.

3) Deformation-Coupled Lifespan: To enforce distinct temporal windows for moving actors while preserving static backgrounds indefinitely, we assign each Gaussian a base lifespan $( L _ { \mathrm { b a s e } } )$ and couple it to the Gated Deformation Mask $( M _ { \mathrm { d e f o r m } } )$ to compute an effective lifespan $( L _ { \mathrm { e f f } } ) _ { \mathrm { - } }$

$$
{ \cal L } _ { \mathrm { e f f } } = ( 1 - M _ { \mathrm { d e f o r m } } ) + M _ { \mathrm { d e f o r m } } \cdot { \cal L } _ { \mathrm { b a s e } }\tag{2}
$$

This formulation ensures that static primitives $( M _ { \mathrm { d e f o r m } }  0 )$ retain a full, sequence-wide footprint of 1.0, while dynamic elements $( M _ { \mathrm { d e f o r m } }  1 )$ are tightly bounded by their localized lifespan. Primitives whose $L _ { \mathrm { e f f } }$ falls below a minimum temporal threshold are actively culled and respawned, preventing the waste of computational capacity on transient artifacts.

4) Temporal Existence Activation: To bridge continuous gradient optimization and discrete physical boundaries, we introduce the Temporal Existence Activation:

$$
\alpha _ { t } = \mathrm { s i g m o i d } \left( k \cdot \left( 1 - \frac { | t - \tau | } { L _ { \mathrm { e f f } } } \right) \right)\tag{3}
$$

This activation value acts as a multiplier to the Gaussian’s base spatial opacity. It is evaluated symmetrically around the primitive’s learned temporal center $( \tau \in [ 0 , 1 ] )$ within its effective lifespan $L _ { \mathrm { e f f } } .$ , where t represents the normalized global sequence time. Crucially, k acts as a progressive sharpness parameter. During initial training, a low k creates a smooth distribution for free temporal gradient propagation. As optimization converges, k is systematically scaled upward, sharpening into a rigid box-car distribution that strictly enforces temporal existence boundaries.

5) Trajectory-Preserving Temporal Decoupling: To capture highly complex, non-linear motions, the system performs temporal exploration (perturbing τ to discover optimal temporal alignments) and spatiotemporal cloning (splitting a dynamic primitive’s lifespan into sequential segments to accurately represent piecewise motion). However, actively shifting this reference center within a localized kernel fundamentally misaligns the learned polynomial and Fourier coefficients of Equation 1.

To enable these temporal shifts without corrupting trajectory convergence, we propose a decoupled, global-time based deformation kernel, $\Delta _ { \mathrm { e f f } } ( t , \tau ) \colon$

$$
\Delta _ { \mathrm { e f f } } ( t , \tau ) = \Delta ( t ) - \Delta ( \tau )\tag{4}
$$

Under this parameterization, the underlying trajectory $\Delta ( t )$ operates strictly on global sequence time. When the system updates τ to a new value $\tau ^ { \prime } ,$ the Gaussian’s spatial parameters are exactly and efficiently compensated by applying the analytic offset $\Delta ( \tau ^ { \prime } ) - \Delta ( \tau )$ to the base reference position, guaranteeing that temporal exploration and cloning can freely optimize the existence window without destroying established kinematics.

## D. Deformation and Trajectory Regularization

While the explicit temporal manifold effectively captures 4D kinematics, it requires regularization to converge to physically plausible states. To ensure the Gated Deformation Mask functions as a strict binary classifier rather than a continuous scaling factor, we apply a combined L<sub>1</sub> sparsity penalty and a self-entropy minimization loss. This exerts downward pressure on static geometry while driving mask values strictly toward bistable states (0 or 1). Simultaneously, to prevent overfitting in the polynomial motion framework, we apply a small L<sub>1</sub> penalty directly to the trajectory coefficients, which inherently favors smooth paths unless complex motion is heavily supported by structural gradients.

TABLE I  
DETAILED LIDAR SEGMENTATION PERFORMANCE. COMPREHENSIVE PER-CLASS IOU EVALUATED ON THE SEMANTICKITTI VALIDATION SET [1] AT THE SPECIFIC RECALL LEVELS OF BASELINE METHODS. BEST RESULTS ARE HIGHLIGHTED IN BOLD.
<table><tr><td> %</td><td>  lu mo</td><td colspan="7"></td><td>Person Road</td><td>Sdalk</td><td>Othor-und</td><td>Mammade</td><td>Vetton Teran</td><td></td><td>Recall: 47.32%</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>↓ % </td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>↓ % o ml</td><td></td><td></td><td>86.6</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>68.3</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>71.6</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>87.6</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></table>

## E. Unified Downstream Extraction

To demonstrate the versatility of our continuous 4D representation, we extract discrete pseudo-labels for two distinct downstream tasks: LiDAR point cloud segmentation and semantic occupancy prediction. Because 3DGS inherently prioritizes 2D camera coverage, it often generates artifacts in regions lacking depth priors. To ensure high fidelity during this continuous-to-discrete translation, we first apply a Voxel-Based Geometric Filter using accumulated LiDAR point clouds, strictly grounding the extraction in physical scene geometry by removing unconstrained floaters.

For point-level LiDAR segmentation, we avoid naive nearest-neighbor assignments by employing an inversedistance weighted voting scheme among the local Gaussian neighborhood. This spatial consensus out-votes isolated outliers and provides a native uncertainty metric via Shannon Entropy. Similarly, our continuous formulation enables resolution-agnostic semantic occupancy extraction. A target voxel is classified as occupied if it falls within a structurally filtered Gaussian. Its semantic class is derived from the mean consensus of the continuous semantic field queried at its eight bounding vertices, elegantly resolving boundary ambiguities while maintaining spatial coherence.

## F. Pseudo-Labelling as Selective Classification

Existing pseudo-labelling baselines frequently evaluate performance strictly on the subset of points they label [4], [6]. This evaluation protocol fails to provide a holistic understanding of a pipeline’s efficacy, as it ignores the widely varying and arbitrary recall levels across different methods.

To resolve this, we propose formulating 3D pseudolabelling as a selective classification task [30]. By treating unassigned pseudo-labels as “rejected” predictions, we can evaluate disparate pipelines using robust risk-coverage tradeoffs. We utilize Generalized Risk, which quantifies the joint probability that a model accepts a point for pseudo-labelling and that the assigned label is incorrect [15]. This acts as a direct, practical measure of the risk of injecting noisy labels into downstream training.

For a given class k at a global coverage/recall threshold $^ { c , }$ the selective risk is defined as $1 - \mathrm { I o U } _ { k , c } ,$ , where IoU represents the Intersection over Union. Under severe class imbalance, standard selective global metrics often appear deceptively optimistic by hiding the disproportionate rejection of minority classes [32]. To ensure a fair, bias-aware evaluation across the entire dataset, we formulate our overall performance metric as the class-averaged Area Under the Generalized Risk-Coverage (AUGRC) curve:

$$
A U G R C = \int _ { 0 } ^ { 1 } ( 1 - \mathrm { m I o U } _ { c } ) \cdot c d c\tag{5}
$$

Bounded strictly between 0 and 0.5, a lower AUGRC indicates superior performance. By deriving this overall metric from the mean of the class-specific generalized risks, we provide a robust mathematical foundation to evaluate the true precision-recall reliability of any pseudo-labelling system.

## IV. EXPERIMENTS

To validate the efficacy of our proposed representation, we evaluate its performance on two downstream tasks: 3D LiDAR segmentation and semantic occupancy prediction.

## A. Experimental Setup

We evaluate our method on the SemanticKITTI [1] validation set (Sequence 08). Semantic features are extracted zero-shot using the SAM 3 [14] foundation model. Crucially, we eschew complex prompt engineering, opting instead to directly prompt the dataset class names to evaluate the native robustness of our differentiable learning backend. Due to the inherent VRAM constraints associated with optimizing dense 3D Gaussian Splatting over long durations, we divide the sequence into independent, approximately 20- second continuous scenes. These localized scenes are trained independently on NVIDIA A40 GPUs and subsequently stitched together using a primitive alignment strategy during inference.

## B. LiDAR Segmentation and Recall Evaluation

To ensure fair comparisons with existing baselines, we evaluate our results using the 11 common merged classes (car, bicycle, motorcycle, other-vehicle, person, road, sidewalk, other-ground, manmade, vegetation, and terrain) and the six categories (flat, construction, object, nature, human, and vehicle) defined by the KITTI-360 benchmark [37]. Qualitative segmentation results are visualized in the top rows of Fig. 2.

TABLE II  
DETAILED SEMANTIC OCCUPANCY PERFORMANCE ON THE SEMANTICKITTI VALIDATION SET [1]. FOLLOWING THE EVALUATION PROTOCOL INTRODUCED BY [5], THE EVALUATED BASELINES ARE TRAINED EXCLUSIVELY ON OCC3D-NUSCENES [33] AND TESTED IN A ZERO-SHOT SETTING ON SEMANTICKITTI [1] TO ASSESS THEIR CROSS-DATASET TRANSFERABILITY. IN CONTRAST, AUTOOCC [5] OPERATES AS AN OPEN-ENDED LABEL GENERATION PIPELINE ANNOTATING THE TARGET DATASET DIRECTLY. COMPARED AGAINST BOTH ZERO-SHOT TRANSFERRED BASELINES AND AUTOOCC [5] UNDER THIS PROTOCOL, OUR METHOD DEMONSTRATES SUPERIOR PERFORMANCE. INPUT MODALITIES ARE DENOTED AS C (CAMERA) AND L (LIDAR), WITH C+L INDICATING THE USE OF BOTH. BEST RESULTS ARE HIGHLIGHTED IN BOLD AND SECOND-BEST ARE UNDERLINED.
<table><tr><td>Method</td><td>Input</td><td>G   </td><td>↓  . mo Car ■</td><td colspan="10">Oth-icle Motorcist Motorce Biylist Parking SIidwalk Bieyycle Person Truck Road</td><td>Buiding</td><td>Fence</td><td>Vettion</td><td></td><td>Trun</td><td>Lerain</td><td>Trassgn Pole</td></tr><tr><td>GaussianOcc [34]</td><td>C</td><td>22.42</td><td>4.18</td><td>7.10</td><td>1.33</td><td>3.06 3.42</td><td>2.81</td><td>2.91</td><td>0.00 0.00</td><td>15.80</td><td>0.00</td><td>10.43</td><td>2.55</td><td>0.00</td><td>22.11</td><td>0.00</td><td>3.78</td><td>0.00</td><td>0.00</td></tr><tr><td>OVO [35]</td><td>C</td><td>20.94</td><td>5.83</td><td>12.70</td><td>0.40</td><td>0.20 0.70</td><td>3.50</td><td>0.74</td><td>0.90 0.00</td><td>19.44</td><td>0.68</td><td>24.81</td><td>11.70</td><td>3.50</td><td>15.62</td><td>2.31</td><td>4.86</td><td>0.60</td><td>2.20</td></tr><tr><td>SurroundOcc [36]</td><td>L</td><td>27.83</td><td>6.39</td><td>23.19</td><td>1.52</td><td>6.71 3.16</td><td>4.81</td><td>4.37 0.00</td><td>0.00</td><td>24.32</td><td>0.00</td><td>11.98</td><td>5.79</td><td>0.00</td><td>19.14</td><td>0.00</td><td>9.95</td><td>0.00</td><td>0.00</td></tr><tr><td>VLM-LiDAR [5]</td><td>C+L</td><td>28.12</td><td>5.32</td><td>19.17</td><td>2.04</td><td>2.13 5.89</td><td>3.31</td><td>2.64 0.00</td><td>0.00</td><td>19.02</td><td>0.00</td><td>16.58</td><td>3.59</td><td>0.00</td><td>14.98</td><td>0.00</td><td>6.31</td><td>0.00</td><td>0.00</td></tr><tr><td>AutoOcc-V [5]</td><td>C</td><td>35.64 9.36</td><td></td><td>22.29</td><td>4.71</td><td>10.35 8.78</td><td>3.89</td><td>7.54 1.38</td><td>3.60</td><td>26.14</td><td>0.59</td><td>15.66</td><td>4.14</td><td>4.34</td><td>18.87</td><td>5.36</td><td>9.84</td><td>14.32</td><td>6.62</td></tr><tr><td>AutoOcc-M [5]</td><td>C+L</td><td>41.23 12.76</td><td></td><td>24.60</td><td>7.83</td><td>9.30 8.39</td><td>4.92</td><td>11.18 1.27</td><td>5.23</td><td>44.74</td><td>0.33</td><td>24.43</td><td>17.01</td><td>5.71</td><td>29.12</td><td>5.97</td><td>5.85</td><td>15.17</td><td>8.72</td></tr><tr><td>SplatLabel (Ours)</td><td>C+L</td><td>47.36 15.56</td><td></td><td>28.04</td><td>10.91</td><td>16.74 9.65</td><td>0.28</td><td>7.65 0.26</td><td>0.01</td><td>48.58</td><td>11.30</td><td>30.94</td><td>22.58</td><td>11.79</td><td>32.96</td><td>19.15</td><td>7.37</td><td>12.10</td><td>9.71</td></tr></table>

Through direct correspondence with the respective authors, we determined that LeAP [4] generates labels for 65.3% of the points, while UniLiPs [6] labels an average of 47.32% of the points on the training set. Because our methodology inherently supports continuous evaluation across all recall thresholds via our selective classification formulation, we benchmark our pipeline directly against these baselines at their exact recall levels to ensure absolute fairness. As illustrated in Figure 3, SplatLabel maintains a superior performance profile across all recall levels, consistently outperforming baseline operating points even when accounting for variance. Global quantitative results at these specific recall thresholds are summarized in Table I. As detailed in Table I, our method struggles specifically with the ‘terrain’ class. This is an expected consequence of our zero-shot VFM distillation without domain-specific prompt engineering, as 2D VFM could conflate terrain with vegetation. However, because both classes belong to the same broader category, our category-level mIoU (cat. mIoU) natively captures this structural understanding without penalty.

## C. Semantic Occupancy Prediction

Beyond point-level segmentation, we evaluate our performance on semantic occupancy prediction against AutoOcc [5], the current state-of-the-art pseudo-labelling method. Following their evaluation protocol, we assess performance across 18 semantic classes, excluding the ambiguous “other flat/ground” class to ensure a fair comparison [5]. As qualitatively demonstrated in the bottom rows of Fig. 2, our explicit kinematic parameterization successfully prevents the temporal smearing artifacts often present in aggregated ground truth data.

As shown in Table II, our method demonstrates superior Geometric and Semantic IoU. To ensure the statistical significance of these improvements, we evaluated our method across five random seeds. We observed a minimal standard deviation of just 0.066 and 0.041 percentage points for Geometric IoU and Semantic mIoU respectively. These microscopic variances are orders of magnitude smaller than the absolute performance margins we achieve over previous baselines, thoroughly solidifying the stability and superiority of our approach.

## D. Spatial Domain and Generalized Risk Evaluation

To establish standardized baselines for pseudo-label reliability under varying geometric constraints, we evaluate the Area Under the Generalized Risk-Coverage curve (AUGRC) across three distinct spatial visibility subsets (see Table III):

• All Points (100% recall): The complete 360-degree LiDAR frame, evaluated irrespective of camera overlap.

• Camera (92.0% recall): Points projecting into any camera frustum across the temporal sequence, inclusive of physically occluded points.

• Unoccluded (62.3% recall): Points within the frustum that are physically unobstructed. We determine this by comparing a LiDAR point’s true depth to the rendered depth map; a point is classified as unoccluded if its true depth does not exceed the rendered depth by more than a 0.5-meter tolerance.

Lower AUGRC values indicate a more favorable accuracycoverage trade-off, demonstrating our method’s robustness across diverse semantic classes without restricting evaluation to heavily curated domains.

## E. Computational Resources

All experiments are executed on a cluster utilizing 4 NVIDIA A40 GPUs concurrently. Because our pipeline divides the dataset into independent, approximately 20- second continuous scene splits, the optimization process is highly parallelizable, requiring a total end-to-end training time of 15.28 GPU hours for 3D LiDAR segmentation and 33.67 GPU hours for semantic occupancy prediction on the complete SemanticKITTI validation set. During the inference phase, the label generation process operates efficiently, taking an average of 0.28 and 4.62 seconds per frame for LiDAR segmentation and semantic occupancy prediction respectively, as measured on a single NVIDIA A40 GPU.

TABLE III  
DETAILED SPATIAL DOMAIN EVALUATION. PER-CLASS AND PER-CATEGORY AUGRC EVALUATED ACROSS VARYING SPATIAL VISIBILITY SUBSETS ON THE SEMANTICKITTI VALIDATION SET [1]. LOWER VALUES INDICATE BETTER RELIABILITY.
<table><tr><td></td><td>e%</td><td>↑  AGRC</td><td>Car ■</td><td>Biyce ■</td><td>Motoryce</td><td>Ooh.--ile ■</td><td>Person 一</td><td>Road 1</td><td>SIidalk</td><td>Othor-und</td><td>Mammade</td><td>Vetton</td><td>Tan</td><td>↑  G </td><td>Hat</td><td>Constuction</td><td>Obet</td><td>Natue</td><td>Human</td><td>Vehiecle</td></tr><tr><td>All Points</td><td>100</td><td>19.4</td><td>10.2</td><td>27.2</td><td>17.3</td><td>10.1</td><td>17.0</td><td>3.4</td><td>10.2</td><td>46.5</td><td>11.0</td><td>14.4</td><td>45.9</td><td>11.3</td><td>4.2</td><td>11.2</td><td>17.9</td><td>5.9</td><td>18.5</td><td>10.4</td></tr><tr><td>Camera</td><td>92.0</td><td>19.2</td><td>9.8</td><td>27.5</td><td>19.5</td><td>9.8</td><td>14.7</td><td>3.1</td><td>9.3</td><td>46.4</td><td>10.7</td><td>14.2</td><td>45.9</td><td>10.1</td><td>3.7</td><td>10.8</td><td>16.0</td><td>5.5</td><td>14.7</td><td>9.9</td></tr><tr><td>Unoccluded</td><td>62.3</td><td>16.1</td><td>5.2</td><td>23.3</td><td>15.1</td><td>5.4</td><td>10.3</td><td>1.9</td><td>5.9</td><td>43.7</td><td>7.2</td><td>13.3</td><td>45.5</td><td>6.4</td><td>1.9</td><td>6.6</td><td>12.5</td><td>3.4</td><td>8.8</td><td>5.1</td></tr></table>

TABLE IV

## V. ABLATION STUDY

To isolate the individual contributions of our pipeline’s spatial, kinematic, and temporal mechanisms, we perform a leave-one-out ablation study, summarized in Table IV.

1) Semantic and Geometric Priors: Replacing our continuous soft labels with traditional hard semantic assignments (w/ Hard Semantic Labels) severely degrades overall performance. This validates that backpropagating continuous soft semantics is critical for natively resolving categorical ambiguities through spatiotemporal consensus. Geometric priors are equally vital; removing Virtual Depth Maps, which integrate 360-degree LiDAR into unobserved regions, causes a massive drop in Geometric IoU. Finally, bypassing Geometry Filtering allows unconstrained floating artifacts and 2D-optimized noise to corrupt 3D pseudo-label generation, uniformly harming all metrics.

2) Spatiotemporal Kinematics: Explicit kinematic parameterization is essential for accurately separating dynamic actors from static environments. Disabling the Gated Deformation Mask allows static structures to improperly track micro-motions rather than remaining firmly clamped to zero, introducing widespread geometric jitter that sharply reduces geometric and semantic fidelity. Furthermore, decoupling the lifespan from the deformation mask (− Coupled Lifespan) causes a consistent drop across all metrics. This indicates that ensuring static primitives retain a full, sequence-wide footprint is vital for maintaining global scene stability.

## VI. CONCLUSION

We presented SplatLabel, an automated 4D Gaussian Splatting pipeline designed specifically for 3D semantic pseudo-labelling and volumetric occupancy prediction. At its core, SplatLabel overcomes the bottlenecks of dynamic scene reconstruction by introducing an explicit temporal manifold. By parameterizing continuous motion, temporal lifespans, and physical existence boundaries as intrinsic primitive properties, our framework successfully tracks dynamic actors while cleanly preserving static backgrounds. This approach entirely eliminates the need for pre-annotated 3D bounding boxes.

THE (−) DENOTES THE REMOVAL OF A PROPOSED COMPONENT, WHILE (W/) DENOTES THE SUBSTITUTION OF A BASELINE METHODOLOGY. VALUES REPRESENT THE PERFORMANCE DELTA RELATIVE TO THE FULL PIPELINE. GREEN VALUES INDICATE THAT REMOVING THE COMPONENT DEGRADES OVERALL PERFORMANCE (A POSITIVE DELTA FOR AUGRC OR A NEGATIVE DELTA FOR IOU), THEREBY VALIDATING ITS NECESSITY.
<table><tr><td>Ablation Variant</td><td>AUGRC</td><td>Geo. IoU</td><td>Sem. mIoU</td></tr><tr><td>SplatLabel: Full Pipeline</td><td>19.4</td><td>47.4</td><td>15.6</td></tr><tr><td>w/ Hard Semantic Labels</td><td>+6.5</td><td>-1.9</td><td>-1.2</td></tr><tr><td>– Virtual Depth Maps</td><td>-0.5</td><td>-15.7</td><td>-3.4</td></tr><tr><td>— Gated Deformation Mask</td><td>+2.0</td><td>-12.7</td><td>-4.3</td></tr><tr><td>Coupled Lifespan</td><td>+1.1</td><td>-5.9</td><td>-1.5</td></tr><tr><td>Geometry Filtering</td><td>+3.7</td><td>-3.9</td><td>-3.2</td></tr></table>

To robustly ground this temporal representation, we utilize a Spatiotemporal Sampling-Based Density Controller built on [17] that directly distills continuous soft semantics from 2D foundation models [14] and injects 360-degree LiDAR priors via virtual depth maps. Furthermore, to address systemic flaws in standard pseudo-label evaluation, we reformulated the task as selective classification using the AUGRC [15] metric, providing a mathematically grounded, bias-aware measure of precision-recall reliability. Experiments on SemanticKITTI demonstrate that SplatLabel consistently outperforms state-of-the-art baselines across arbitrary recall levels, establishing a robust, annotation-free foundation for dynamic scene understanding.

## A. Limitations

Our approach is constrained by the high VRAM requirements inherent to dense 3DGS [7] optimization and relies on precise camera-LiDAR synchronization. Additionally, because the pipeline depends heavily on camera imagery for dense semantics and LiDAR for structural geometry, its robustness inherently degrades in adverse lighting or weather conditions where sensor inputs are compromised.

## B. Future Work

For future work, we plan to validate the practical efficacy of our generated labels directly on downstream tasks. This will involve training downstream semantic segmentation and occupancy models purely on our generated annotations and benchmarking their final performance against models trained on human-annotated ground truth. Finally, because our pipeline natively outputs predictive confidence scores via spatial consensus, future research will explore integrating these uncertainty metrics directly into downstream training processes to dynamically weight loss functions.

## REFERENCES

[1] J. Behley, et al., “SemanticKITTI: A Dataset for Semantic Scene Understanding of LiDAR Sequences,” in 2019 IEEE/CVF International Conference on Computer Vision (ICCV). IEEE, Oct. 2019, pp. 9296– 9306.

[2] H. Caesar, et al., “nuScenes: A Multimodal Dataset for Autonomous Driving,” in 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, June 2020, pp. 11 618–11 628.

[3] S. Peng, et al., “OpenScene: 3D Scene Understanding with Open Vocabularies,” in 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, June 2023, pp. 815–824.

[4] S. Gebraad, A. Palffy, and H. Caesar, “LeAP: Consistent multi-domain 3D labeling using Foundation Models,” in 2025 IEEE International Conference on Robotics and Automation (ICRA). IEEE, May 2025, pp. 12 671–12 679.

[5] X. Zhou, et al., “AutoOcc: Automatic Open-Ended Semantic Occupancy Annotation via Vision-Language Guided Gaussian Splatting,” in 2025 IEEE/CVF International Conference on Computer Vision (ICCV). IEEE, Oct. 2025, pp. 3367–3377.

[6] F. Ghilotti, et al., “UniLiPs: Unified LiDAR Pseudo-Labeling with Geometry-Grounded Dynamic Scene Decomposition,” in 2026 International Conference on 3D Vision (3DV). IEEE, Mar. 2026, pp. 1680–1690.

[7] B. Kerbl, et al., “3D Gaussian Splatting for Real-Time Radiance Field Rendering,” ACM Transactions on Graphics, vol. 42, pp. 1–14, Aug. 2023.

[8] X. Zhou, et al., “DrivingGaussian: Composite Gaussian Splatting for Surrounding Dynamic Autonomous Driving Scenes,” in 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, June 2024, pp. 21 634–21 643.

[9] Z. Chen, et al., “OmniRe: Omni Urban Scene Reconstruction,” in International Conference on Learning Representations, Y. Yue, et al., Eds., vol. 2025, 2025, pp. 85 508–85 527.

[10] E. Giacomini, et al., “Splat-LOAM: Gaussian Splatting LiDAR Odometry and Mapping,” in 2025 IEEE/CVF International Conference on Computer Vision (ICCV). IEEE, Oct. 2025, pp. 27 630–27 639.

[11] P.-C. Kung, et al., “LiHi-GS: Li DAR-Supervised Gaussian Splatting for Hi ghway Driving Scene Reconstruction,” IEEE Robotics and Automation Letters, vol. 10, pp. 13 272–13 279, Dec. 2025.

[12] Y. Lin, et al., “Gaussian-Flow: 4D Reconstruction with Dynamic 3D Gaussian Particle,” in 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, June 2024, pp. 21 136– 21 145.

[13] R. Song, et al., “CoDa-4DGS: Dynamic Gaussian Splatting with Context and Deformation Awareness for Autonomous Driving,” in 2025 IEEE/CVF International Conference on Computer Vision (ICCV). IEEE, Oct. 2025, pp. 28 031–28 041.

[14] N. Carion, et al., “SAM 3: Segment Anything with Concepts,” International Conference on Learning Representations, vol. 2026, pp. 138 846–138 923, Apr. 2026.

[15] J. Traub, et al., “Overcoming Common Flaws in the Evaluation of Selective Classification Systems,” in Advances in Neural Information Processing Systems 37. Neural Information Processing Systems Foundation, Inc. (NeurIPS), 2024, pp. 2323–2347.

[16] J. L. Schonberger and J.-M. Frahm, “Structure-from-Motion Revis-¨ ited,” in 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), June 2016, pp. 4104–4113.

[17] S. Kheradmand, et al., “3D Gaussian Splatting as Markov Chain Monte Carlo,” in Advances in Neural Information Processing Systems 37. Neural Information Processing Systems Foundation, Inc. (NeurIPS), 2024, pp. 80 965–80 986.

[18] N. Brosse, A. Durmus, and E. Moulines, “The promises and pitfalls of Stochastic Gradient Langevin Dynamics,” in Advances in Neural Information Processing Systems, S. Bengio, et al., Eds., vol. 31. Curran Associates, Inc., 2018.

[19] S. Zhu, et al., “3D Gaussian Splatting in Robotics: A Survey,” 2024. [Online]. Available: https://arxiv.org/abs/2410.12262

[20] Y. Huang, et al., “GaussianFormer: Scene as Gaussians for Vision-Based 3D Semantic Occupancy Prediction,” in Computer Vision – ECCV 2024, A. Leonardis, et al., Eds. Springer Nature Switzerland, 2025, pp. 376–393.

[21] T. Pavkovic,´ et al., “GaussianFusionOcc: A Seamless Sensor Fusion Approach for 3D Occupancy Prediction Using 3D Gaussians,” 2025. [Online]. Available: https://arxiv.org/abs/2507.18522

[22] X. Bai, et al., “RaGS: Unleashing 3D Gaussian Splatting from 4D Radar and Monocular Cue for 3D Object Detection,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2026, pp. 4983–4992.

[23] S. Montiel-Mar´ın, et al., “GaussianCaR: Gaussian Splatting for Efficient Camera-Radar Fusion,” 2026. [Online]. Available: https: //arxiv.org/abs/2602.08784

[24] Y. Yan, et al., “Street Gaussians: Modeling Dynamic Urban Scenes with Gaussian Splatting,” in Computer Vision – ECCV 2024, A. Leonardis, et al., Eds. Springer Nature Switzerland, 2025, pp. 156–173.

[25] N. Huang, et al., “S<sup>3</sup>Gaussian: Self-Supervised Street Gaussians for Autonomous Driving,” 2024. [Online]. Available: https://arxiv.org/ abs/2405.20323

[26] G. Hess, et al., “SplatAD: Real-Time Lidar and Camera Rendering with 3D Gaussian Splatting for Autonomous Driving,” in 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, June 2025, pp. 11 982–11 992.

[27] Z. Li, et al., “Spacetime Gaussian Feature Splatting for Real-Time Dynamic View Synthesis,” in 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, June 2024, pp. 8508–8520.

[28] Y. Chen, et al., “Periodic Vibration Gaussian: Dynamic Urban Scene Reconstruction and Real-time Rendering,” International Journal of Computer Vision, vol. 134, p. 83, Mar. 2026.

[29] G. Wu, et al., “4D Gaussian Splatting for Real-Time Dynamic Scene Rendering,” in 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, June 2024, pp. 20 310–20 320.

[30] R. El-Yaniv and Y. Wiener, “On the Foundations of Noise-free Selective Classification,” Journal of Machine Learning Research, vol. 11, pp. 1605–1641, 2010.

[31] Y. Geifman and R. El-Yaniv, “Selective Classification for Deep Neural Networks,” in Advances in Neural Information Processing Systems, I. Guyon, et al., Eds., vol. 30. Curran Associates, Inc., 2017.

[32] F. Saglam,˘ et al., “Selective classification under imbalance in multiclass settings: A novel metric for bias-aware risk–coverage evaluation,” Journal of Biomedical Informatics, vol. 181, p. 105084, Sept. 2026.

[33] X. Tian, et al., “Occ3D: A Large-Scale 3D Occupancy Prediction Benchmark for Autonomous Driving,” in Advances in Neural Information Processing Systems 36. Neural Information Processing Systems Foundation, Inc. (NeurIPS), 2023, pp. 64 318–64 330.

[34] W. Gan, et al., “GaussianOcc: Fully Self-Supervised and Efficient 3D Occupancy Estimation with Gaussian Splatting,” in 2025 IEEE/CVF International Conference on Computer Vision (ICCV). IEEE, Oct. 2025, pp. 28 980–28 990.

[35] Z. Tan, et al., “OVO: Open-Vocabulary Occupancy,” 2023. [Online]. Available: https://arxiv.org/abs/2305.16133

[36] Y. Wei, et al., “SurroundOcc: Multi-Camera 3D Occupancy Prediction for Autonomous Driving,” in 2023 IEEE/CVF International Conference on Computer Vision (ICCV). IEEE, Oct. 2023, pp. 21 672– 21 683.

[37] Y. Liao, J. Xie, and A. Geiger, “KITTI-360: A Novel Dataset and Benchmarks for Urban Scene Understanding in 2D and 3D,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 45, pp. 3292–3310, Mar. 2023.