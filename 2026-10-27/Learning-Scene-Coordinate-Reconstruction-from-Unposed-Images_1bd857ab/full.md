# Learning Scene Coordinate Reconstruction from Unposed Images via Pose Graph Optimization

Tze Ho Elden Tse<sup>1,2</sup>, Jizong Peng<sup>1</sup>, Angela Yao<sup>2</sup> <sup>1</sup>dConstruct Robotics <sup>2</sup>National University of Singapore

https://eldentse.github.io/visual\_localization\_pgo\_public/

## Abstract

Learning-based structure-from-motion methods such as ACE-Zero have demonstrated strong performance in estimating cameraposes and scene coordinatesfrom unordered image collections without requiring ground truth supervision. However, the lack of global and multi-view consistency constraints in ACE-Zero can lead to pose drift and misalignment, particularly in complex or ambiguous scenes. In this work, we propose a hybrid framework that integrates pose graph optimization (PGO) into ACE-Zero to refine camera poses and suppress incorrect refinements. We construct pose graphs directly from ACE-Zero outputs by extracting relative pose constraints from predicted scene coordinates. Furthermore, we introduce an uncertainty-aware optimization strategy by estimating confidence scores using geometric priors, including epipolar and optical flow consistencies across views. Our approach improves the robustness and accuracy of pose estimation, demonstrating that global geometric reasoning can effectively complement learning-based inference in structure-from-motion.

## 1. Introduction

Structure-from-Motion (SfM) and visual localization are fundamental problems in computer vision and robotics, enabling applications such as 3D reconstruction, autonomous navigation, and augmented reality [16, 32, 41, 45, 53]. Recent advances [8, 49, 50] have introduced learningbased approaches that bypass traditional feature matching and bundle adjustment. Among these, ACE-Zero [8] has emerged as a promising visual localization framework that estimates camera poses and scene coordinates directly from unordered image collections, without requiring ground truth or explicit geometric supervision.

As shown in Fig. 1, ACE-Zero’s reliance on local neural refinement can lead to locally optimal but globally inconsistent poses, particularly in large-scale or ambiguous scenes. The absence of global consistency constraints makes the framework susceptible to drift and misalignment, especially when the initial pose estimates are noisy or when scene coordinate predictions are uncertain.

![](images/a77144a1aac6c226abe031151b55ea334015b836b5949b963d7ed65a57ef1fc4.jpg)  
Figure 1. Despite being initialized from (a) COLMAP generated camera poses, local refinements in (b) often introduce global inconsistencies. In contrast, (c) our hybrid-framework, which integrates pose graph optimization into ACE-Zero, enforces multiview consistency and suppress incorrect refinements.

To address these limitations, we propose a hybrid framework that integrates pose graph optimization (PGO) into the ACE-Zero pipeline. PGO is a well-established technique in SLAM and SfM that enforces global consistency by optimizing camera poses over a graph of relative constraints [11, 12]. Unlike purely local refinement, PGO leverages pairwise pose relationships across the entire scene, making it well-suited for correcting accumulated drift and resolving ambiguities. Compared to alternatives such as global bundle adjustment or learning-based refinement, PGO is generally more scalable due to its reduced parameter space as it optimizes only camera poses rather than both poses and 3D structure. Moreover, its graph-based formulation offers clearer interpretability by explicitly modeling pairwise pose constraints, which can be visualized and debugged more easily than opaque learned representations.

However, integrating PGO into ACE-Zero is non-trivial. ACE-Zero does not produce explicit pairwise pose constraints or co-visibility graphs, which are typically required for graph construction. Moreover, the scene coordinate predictions are inherently noisy and lack uncertainty estimates, making naive optimization prone to over-fitting or instability. To overcome these challenges, we introduce a geometry-driven approach to derive reliable relative pose constraints from ACE-Zero outputs and construct a robust pose graph for optimization. Specifically, we identify image pairs with consistent scene coordinate predictions and use them to construct the graph edges for PGO.

Furthermore, we extend our framework with uncertaintyaware optimization, which incorporates confidence estimates into the graph refinement process. This allows the optimizer to weigh constraints based on their reliability, suppressing the influence of uncertain predictions and improving robustness. While this requires additional computation to estimate and propagate uncertainty, it significantly enhances the quality of the final pose estimates, especially in scenes with occlusions, repetitive structures, or low texture. These conditions often lead to ambiguous or unreliable 2D-3D correspondences, which can severely degrade pose accuracy. By integrating uncertainty modeling, our approach reduces the impact of such ambiguities and improves robustness in these challenging environments.

This work bridges the gap between learning-based and optimization-based SfM, showing that global geometric reasoning can complement neural inference to produce more reliable and consistent camera pose estimates.

We summarize our main contributions as follows:

• A hybrid framework that integrates pose graph optimization into ACE-Zero to enforce multi-view and global consistency.

• A method for constructing pose graphs from ACE-Zero outputs without explicit geometric supervision.

• An uncertainty-aware optimization strategy that improves robustness by accounting for prediction confidence.

• Our framework achieves more reliable camera poses estimates while requiring on average half the number of iterations compared to ACE-Zero alone.

## 2. Related Work

Traditional SfM pipelines [19, 30, 31, 39] rely on feature matching, triangulation, and bundle adjustment to estimate camera poses and reconstruct 3D scenes. While effective, these methods are sensitive to textureless regions, occlusions, and large viewpoint changes.

Learning-based SfM approaches [8, 49, 50] aim to overcome these limitations by directly regressing scene coordinates or camera poses from the images. Notably, ACE-Zero [8] estimates camera parameters from unordered image collections using a neural scene coordinate regression model. By predicting 3D coordinates directly from images and using reprojection-based losses instead of groundtruth supervision, ACE-Zero enables self-supervised learning without relying on explicit feature matching or pose pri ors. However, we observe that ACE-Zero can suffer from pose drift and misalignment in complex scenes. We hypothesize that this is due to the absence of global consistency constraints, as its refinement is purely local and does not leverage sequential or pairwise structure. To address this, we propose a hybrid framework that integrates pose graph optimization to enforce multi-view consistency and improve robustness.

Other learning-based SfM methods such as DeepSFM [51] and GraphSfM [10] incorporate neural architectures to replace traditional submodules like triangulation and pose averaging. These methods often assume structured inputs (i.e. image sequences or reliable initialization) which inherently provide stronger geometric priors and facilitate global consistency. In contrast, ACE-Zero operates on unordered image sets without pose supervision, making it more susceptible to global misalignment.

Pose Graph Optimization (PGO) is a core technique in SLAM and global SfM. It refines camera poses by minimizing the inconsistency between relative transformations. Each node in the graph represents a camera pose, and each edge encodes a relative constraint derived from image pairs. Optimization is typically performed over the Lie group SE(3) or its variants, such as Sim(3) when scale ambiguity is present [2, 15]. PGO has been widely adopted in SLAM systems like ORB-SLAM [29] and GTSAMbased pipelines [14], where it helps enforce loop closure and correct accumulated drift. Recent works have explored efficient initialization [2], robust estimators, and graphtheoretic principles to improve scalability and convergence. However, integrating PGO into learning-based SfM frameworks remains underexplored, especially in settings without ground truth or sequential data.

Uncertainty-aware optimization incorporates uncertainty into the pose estimation for both object-centric scenarios (e.g., single-instance localization) and scene-level reconstruction (e.g., multi-view SfM). Methods like UA-Pose [25] and UnPose [21] leverage uncertainty estimates to guide optimization and improve robustness to noisy inputs. These approaches typically derive uncertainty from neural networks or diffusion models. Uncertainty-aware formulations of PGO have been explored in domains such as robotics and object pose estimation, where edge weights are adjusted based on measurement confidence [26, 28]. However, their application to learning-based SfM remains underexplored, primarily due to the lack of explicit uncertainty estimates and structured pairwise constraints in neural models like ACE-Zero.

Visual localization aims to estimate the camera pose of a query image within a previously reconstructed scene. Classical approaches [34–36] rely on matching 2D features from the query to 3D points in the scene, followed by pose estimation via PnP. For large-scale environments, retrievalbased methods [33, 37] improve efficiency by narrowing down relevant database images. Learning-based methods offer alternatives by encoding scene geometry directly into neural networks. PoseNet [22] regresses absolute poses directly from images, but struggles with generalization beyond training poses [38]. Relative pose regression [1, 46] and scene coordinate regression [5, 6, 8, 9, 13] improve flexibility by estimating geometric relationships or 3D coordinates, enabling pose estimation via PnP and RANSAC. These methods are typically trained in a supervised manner, relying on ground-truth poses or 3D annotations. In contrast, ACE-Zero [8] trains a visual localizer from unposed images. Our work builds on ACE-Zero by extending it with global consistency via pose graph optimization and uncertainty-aware refinement.

Hybrid approaches that combine learning-based inference with optimization-based refinement are emerging as a powerful paradigm in visual localization. HERO-SLAM [52] and DeepSFM [51] demonstrate that neural representations can be enhanced by geometric reasoning, leading to improved accuracy and robustness. These methods highlight the complementary strengths of data-driven and modelbased techniques. In this work, we build on this insight by integrating pose graph optimization into ACE-Zero, using learned scene coordinates to construct relative constraints and geometric priors to estimate confidence. This hybrid framework bridges the gap between implicit neural representations and explicit geometric consistency.

## 3. Methodology

## 3.1. Preliminary

Pose graph optimization (PGO). PGO is a widely used technique in structure-from-motion (SfM) and simultaneous localization and mapping (SLAM) to refine camera poses by minimizing inconsistencies across pairwise relative transformations. A pose graph is defined as a set of nodes and edges:

• Nodes: Each node i represents a camera pose $\mathbf { T } _ { i } \ =$ $[ \mathbf { R } _ { i } | \mathbf { t } _ { i } ] \in S E ( 3 )$ , where $\mathbf { R } _ { i } \in S O ( 3 )$ is the rotation and $\mathbf { t } _ { i } \in \mathbb { R } ^ { 3 }$ is the translation.

• Edges: Each edge $( i , j ) \in \mathcal { E }$ represents a relative pose constraint between nodes i and j, where $\mathcal { E }$ is the set of all such connected pairs. Each edge is associated with a relative transformation measurement $\mathbf { Z } _ { i j } \in S E ( 3 )$ , typically obtained from pairwise pose estimation.

The goal of PGO is to find the set of absolute camera poses {T<sub>i</sub>} that best explain the observed relative transformations, while ensuring global consistency among all poses throughout the scene. This is formulated as the following optimization problem:

$$
\underset { \{ \mathbf { T } _ { i } \} } { \operatorname* { m i n } } \sum _ { ( i , j ) \in \mathcal { E } } \left. \mathrm { L o g } \left( \mathbf { Z } _ { i j } ^ { - 1 } \mathbf { T } _ { i } ^ { - 1 } \mathbf { T } _ { j } \right) \right. _ { \Omega _ { i j } } ^ { 2 } .\tag{1}
$$

Here, $\operatorname { L o g } ( { \mathord { \cdot } } )$ denotes the logarithmic map from the Lie group $S E ( 3 )$ to its corresponding Lie algebra se(3), producing a 6D vector that concatenates rotational and translational residuals. The matrix $\Omega _ { i j } \in \mathbb { R } ^ { 6 \times 6 }$ is the information matrix, representing the confidence in the measurement $\mathbf { Z } _ { i j }$

This equation jointly optimizes both rotation and translation by aligning the estimated relative transformation $\mathbf { T } _ { i } ^ { - 1 } \mathbf { T } _ { j }$ with the measured one $\mathbf { Z } _ { i j }$ . The residual becomes small when the estimated and measured transformations are close in the $S E ( 3 )$ manifold, thereby promoting global consistency across the entire pose graph.

Scene coordinate regression. Given a query image $I _ { i }$ where i refers to image index, this family of localization methods regresses for a 2D pixel $\mathbf { x } _ { i } \in \mathbb { R } ^ { 2 }$ the corresponding 3D coordinates $\mathbf { X } _ { i } \in \mathbb { R } ^ { 3 }$ in the global scene coordinate frame [7, 40, 48]. This mapping enables the establishment of 2D–3D correspondences, which can then be used to estimate the camera pose $\mathbf { T } _ { i } = [ \mathbf { R } _ { i } | \mathbf { t } _ { i } ] \in S E ( 3 )$ via a PnP [18] solver in a RANSAC [17] loop.

Formally, given an image $I _ { i }$ , the scene coordinate regression network $\Phi _ { \theta }$ predicts $\mathbf { X } _ { i } ^ { k } = \Phi _ { \theta } ( \mathbf { x } _ { i } , I _ { i } )$ . The predicted 3D point $\mathbf { X } _ { i } ^ { k }$ is assumed to lie in a global coordinate frame shared across all images. Given a set of $K$ such 2D– 3D correspondences $\{ ( \mathbf { x } _ { i } , \mathbf { X } _ { i } ) \} _ { k = 1 } ^ { K }$ and ground truth camera poses $\begin{array} { r } { \hat { \mathbf { T } } _ { \mathbf { i } } , } \end{array}$ the network can be trained by minimizing the reprojection error:

$$
\sum _ { k = 1 } ^ { K } \left\| \pi ( \hat { \mathbf { T } } _ { i } \mathbf { X } _ { i } ^ { k } ) - \mathbf { x } _ { i } ^ { k } \right\| ^ { 2 } ,\tag{2}
$$

where $\pi ( \cdot )$ denotes the camera projection function.

Learning-based SfM frameworks typically rely on supervised training, where ground truth poses or depth maps are used to guide the learning of scene geometry and camera motion. ACE-Zero [8] stands out in that it is unsupervised. It recovers camera poses from unordered image collections by iteratively refining pseudo-labels derived from initial pose estimates. The method begins by selecting five random seed images and uses ZoeDepth [4] to predict depth maps for each. The seed with the highest relocalization rate across 1000 mapping images is chosen to initialize the reconstruction. These depth maps are used to generate initial pseudo-labels, which supervise the training of a scene coordinate regression network. This network is then used to register new images and refine their focal lengths and poses using MLPs. What distinguishes ACE-Zero is its ability to bootstrap accurate scene geometry and camera poses without any external supervision, relying solely on internally generated signals. The system jointly learns scene coordinates and camera poses in a closed loop, where each component improves the other over time. As more images are registered, the scene coordinate regressor is updated and pose estimates are refined, gradually improving the reconstruction without requiring ground truth poses, depth maps, or known 3D geometry. The reconstruction process stops either when 99% of the images have been successfully registered, or when fewer than 1% of new images are registered compared to the previous iteration.

ACE-Zero refines camera poses using a learned optimizer that operates locally. Specifically, it updates each pose based only on image-specific information, without enforcing consistency across the entire scene. As a result, the method lacks global constraints and is prone to accumulating drift, leading to inconsistent camera trajectories. To address this limitation, we propose integrating global optimization techniques, such as PGO, which enforce consistency across all estimated poses and improve the overall accuracy and coherence of the reconstructed scene.

## 3.2. Pose graph construction

To enable global pose refinement within the ACE-Zero framework, we construct a pose graph using the predicted scene coordinates and estimated camera poses. The process begins with identifying reliable relative pose constraints between image pairs based on scene coordinate predictions.

At iteration t, ACE-Zero predicts dense scene coordinates $\mathbf { X } _ { i } ^ { t } ~ \in ~ \mathbb { R } ^ { 3 }$ for each pixel $\mathbf { x } _ { i } ^ { t } \in \mathbb { R } ^ { 2 }$ in every image $I _ { i } \in \mathbf { I } .$ , where i refers to the image index. To establish correspondences between two images $I _ { i }$ and $I _ { j }$ , we identify pixel pairs whose predicted scene coordinates are geometrically consistent. Specifically, for each 3D point $\mathbf { X } _ { i } ^ { t }$ predicted in $I _ { i } ,$ we project it into image $I _ { j }$ by first transforming it into the coordinate frame of camera j using the inverse of its absolute pose $\mathbf { T } _ { j } ^ { - 1 }$ . The projected pixel location is then computed as $\hat { \mathbf { x } } _ { j } ^ { t } = \pi ( \mathbf { T } _ { j } ^ { - 1 } \mathbf { X } _ { i } ^ { t } )$ , where $\pi ( \cdot )$ denotes the camera projection function. At this location in $I _ { j }$ , we retrieve the predicted scene coordinate $\mathbf { X } _ { i } ^ { t }$ and compute the 3D consistency error $e _ { \mathrm { m a t c h } } ^ { t } = \| \mathbf { X } _ { j } ^ { t } - \mathbf { X } _ { i } ^ { t } \|$ . If this error is below a predefined threshold $\tau ,$ the pixel pair $( \mathbf { x } _ { i } ^ { t } , \hat { \mathbf { x } } _ { j } ^ { t } )$ is considered a spatially consistent match. Optionally, a symmetric check can be performed by projecting $\mathbf { X } _ { j } ^ { t }$ back into $I _ { i }$ and verifying consistency with $\mathbf { X } _ { i } ^ { t } .$ These geometrically verified correspondences are then used to estimate the relative transformation between the two camera poses.

To compute the relative pose $\mathbf { Z } _ { i j } = [ \mathbf { R } _ { i j } | \mathbf { t } _ { i j } ] \in S E ( 3 )$ between images $I _ { i }$ and $I _ { j }$ , we use RANSAC with PnP to find the transformation that best aligns the matched 3D points by minimizing the reprojection error:

$$
\mathbf { X } _ { j } ^ { t } \approx \mathbf { R } _ { i j } \mathbf { X } _ { i } ^ { t } + \mathbf { t } _ { i j } .\tag{3}
$$

Once a sufficient number of reliable relative pose estimates are obtained, we construct the pose graph. Each node in the graph corresponds to a camera pose $\mathbf { T } _ { i } .$ , and each edge $( i , j )$ represents a relative transformation $\mathbf { Z } _ { i j }$ between two images. The resulting graph captures the spatial relationships between camera views and serves as the foundation for global pose optimization.

While pose graph construction enables global refinement, it introduces several challenges. The accuracy of the relative pose estimates is sensitive to the quality of the scene coordinate predictions and the geometric consistency threshold τ. A low threshold may reject valid correspondences, while a high threshold risks incorporating noisy matches. Additionally, although RANSAC with PnP is used to robustly estimate relative transformations from matched points, its success depends heavily on the presence of a sufficient number of inliers. In cases where the correspondences are sparse or noisy, it may fail to produce a reliable estimate, even when using only geometrically verified matches. These factors can lead to erroneous or missing edges in the pose graph, which in turn affect the quality of global optimization. To address these limitations, we incorporate uncertainty-aware mechanisms in the subsequent optimization stage to improve robustness and reliability.

## 3.3. Uncertainty-aware pose graph optimization

Incorporating uncertainty into pose graph optimization has been shown to improve robustness and accuracy, particularly in scenarios where input data may be noisy or ambiguous. By weighting constraints according to their confidence, uncertainty-aware methods can down-weight unreliable measurements and prevent them from disproportionately influencing the global solution. In traditional SfM or SLAM pipelines, uncertainty estimates are often derived from sensor noise models, reprojection errors, or covariance propagation during bundle adjustment. However, ACE-Zero does not provide explicit uncertainty estimates for its predicted scene coordinates or camera poses. This presents a challenge for integrating uncertainty into the optimization process. To address this, we propose two unsupervised strategies for estimating confidence in the predicted scene coordinates based on geometric priors. These confidence scores are then used to weight the relative pose constraints in the pose graph, allowing the optimizer to account for the varying reliability of different correspondences.

Epipolar-based uncertainty metric. We use epipolar geometry as a geometric prior to estimate the uncertainty of predicted scene coordinates. Epipolar geometry leverages the fundamental constraint that a 3D point observed in one image must project onto the corresponding epipolar line in another image, given the relative camera poses.

We first project $\mathbf { X } _ { i } ^ { t }$ into image $I _ { j }$ to obtain the pixel location $\begin{array} { r } { \hat { \mathbf { x } } _ { j } ^ { t } = \bar { \pi } ( \bar { \mathbf { T } _ { j } ^ { - 1 } } \mathbf { X } _ { i } ^ { t } ) } \end{array}$ . Using the relative pose, we compute the essential matrix:

$$
{ \bf E } _ { i j } = [ { \bf t } _ { i j } ] { \bf { \times R } } _ { i j } ,\tag{4}
$$

where $\left[ \mathbf { t } _ { i j } \right]$ <sub>×</sub> is the skew-symmetric matrix of the relative translation vector. Using the camera intrinsic matrix $\mathbf { K } _ { j }$ the epipolar line $\mathrm { l } _ { j } ^ { t }$ in image $I _ { j }$ corresponding to pixel $\mathbf { x } _ { i } ^ { t }$ in image $I _ { i }$ is given by:

$$
\mathbf { l } _ { j } ^ { t } = \mathbf { K } _ { j } ^ { - \top } \mathbf { E } _ { i j } \mathbf { K } _ { j } ^ { - 1 } \mathbf { x } _ { i } ^ { t } .\tag{5}
$$

![](images/581dbc47217cd6d6ede86e99c604bcc249c70cbc3f65250e0bf2a9211e966910.jpg)  
Figure 2. Qualitative example of the epipolar-based uncertainty metric. We overlay the reprojected points and their corresponding epipolar lines which are color-coded by error (green=low, red=high), indicating geometric alignment quality under perturbed pose. In (a), errors remain low with no perturbation applied. Introducing pose perturbations in (b) and (c) naturally increases the error. Interestingly, adding translation perturbations in (d) and (e) can sometimes reduce the error, indicating that epipolar error is sensitive to interactions between pose components and does not always correlate directly with pose accuracy. Consequently, we interpret it as a heuristic confidence indicato rather than a strict geometric validator.

We then compute the epipolar uncertainty as the distance between the projected point $\hat { \mathbf { x } } _ { j } ^ { k }$ and the epipolar line $\mathbf { l } _ { j } ^ { k }$ :

$$
e _ { \mathrm { e p i } } ^ { t } = \frac { \vert \mathbf { l } _ { j } ^ { t \top } \hat { \mathbf { x } } _ { j } ^ { t } \vert } { \sqrt { l _ { j , 1 } ^ { 2 } + l _ { j , 2 } ^ { 2 } } } ,\tag{6}
$$

where $l _ { j , 1 } , l _ { j , 2 }$ are the first two components of the epipolar line $\mathbf { l } _ { j } ^ { k } = [ a , b , c ] ^ { \top }$ . The numerator computes the algebraic distance from the point to the line, and the denominator normalizes the line direction to yield the Euclidean (perpendicular) distance from the point to the epipolar line.

This error measures how well the predicted 3D point aligns with the epipolar constraint across views. A lower error indicates higher geometric consistency and thus greater confidence in the scene coordinate prediction. Unlike depth consistency, which compares predicted 3D coordinates directly, epipolar geometry provides a view-invariant constraint that is particularly useful when depth predictions are noisy or ambiguous.

Optical flow-based uncertainty metric. While epipolar geometry provides global geometric prior for assessing scene coordinate confidence, its effectiveness depends on the accuracy of both the predicted 3D coordinates and the relative camera poses. Errors in scene coordinates can lead to unstable reprojections, and pose inaccuracies distort the essential matrix, resulting in misleading epipolar lines– particularly during early training stage.

As shown in Fig. 2, the epipolar-based uncertainty metric is often weakly sensitive to small pose perturbations. Even when camera poses deviate from ground truth, the resulting epipolar residuals may remain low. This suggests that epipolar metric alone may not be sufficiently discriminative for uncertainty estimation. While it can indicate inconsistency, it may also underrepresent uncertainty in ambiguous or noisy conditions.

To address these limitations, we introduce a complementary metric based on optical flow, which provides local and image-driven evidence that is more robust to pose inaccuracies and complements the global nature of epipolar geometry. As shown in Fig. 3, dense optical flow provides pixellevel correspondences across views that are robust to pose errors. Unlike epipolar geometry, which depends on global pose accuracy, optical flow captures local motion patterns directly from image content. Specifically, we obtain the dense optical flow field $\mathbf { F } _ { i  j }$ from RAFT [43], which estimates the pixel displacement between $I _ { i }$ and $I _ { j }$ . The flowpredicted correspondence of $\mathbf { x } _ { i } ^ { t }$ in $I _ { j }$ is:

![](images/2749cc7ff9e07d68045e3f6fdab550eed56583bd7574a09ed56785e80d4e3183.jpg)  
Figure 3. Qualitative example of optical flow predictions. (a) One image from the input pair. (b) Predicted flow visualization where similar colours indicate similar motion directions. (c) UV correspondences overlaid on the image pair, demonstrating that optical flow provides robust pixel-level matches between views.

$$
\begin{array} { r } { \tilde { \mathbf { x } } _ { j } ^ { t } = \mathbf { x } _ { i } ^ { t } + \mathbf { F } _ { i  j } ( \mathbf { x } _ { i } ^ { t } ) . } \end{array}\tag{7}
$$

We then define the optical flow consistency error as the Euclidean distance between the projected point $\hat { \mathbf { x } } _ { j } ^ { t }$ and the flow-predicted correspondence $\tilde { \mathbf { x } } _ { j } ^ { t }$

$$
e _ { \mathrm { f l o w } } ^ { t } = \left\| \hat { \mathbf { x } } _ { j } ^ { t } - \tilde { \mathbf { x } } _ { j } ^ { t } \right\| _ { 2 } .\tag{8}
$$

This error measures how well the predicted 3D point aligns with the pixel-level motion observed in the image pair. $\mathbf { A }$ lower error indicates that the reprojection is consistent with the observed optical flow, suggesting higher confidence in the scene coordinate prediction.

By incorporating both epipolar and optical flow uncertainty metrics, we obtain a more robust confidence estimation framework. Epipolar constraints provide global geometric validation, while optical flow offers local, imagedriven evidence. Together, they improve the reliability of uncertainty-aware pose graph optimization in ACE-Zero.

Combined confidence score. To model the uncertainty of each relative pose constraint in the pose graph, we compute a confidence score for every matched pixel pair between two images. For a pixel $\mathbf { x } _ { i } ^ { t }$ in image $I _ { i } ,$ , its confidence score $\sigma _ { i j } ^ { t }$ is defined as a weighted sum of two geometric error metrics:

$$
\sigma _ { i j } ^ { t } = \alpha _ { 1 } e _ { \mathrm { e p i } } ^ { t } + \alpha _ { 2 } e _ { \mathrm { f l o w } } ^ { t } ,\tag{9}
$$

where $\alpha _ { 1 } , \alpha _ { 2 }$ are hyperparameters. These scores reflect the geometric reliability of each match and are then aggregated over matched points to compute edge weights in the pose graph optimization.

For each image pair $( i , j )$ , we compute the edge-level uncertainty $\sigma _ { i j }$ as the mean of all valid per-match confidence scores. This aggregated uncertainty is then used to define the standard deviation of the noise model for the relative pose constraint in PGO. Specifically, we construct the information matrix $\Omega _ { i j }$ as:

$$
\Omega _ { i j } = \frac { 1 } { \sigma _ { i j } ^ { 2 } + \epsilon } \cdot I ,\tag{10}
$$

where ϵ is a small constant added for numerical stability, and I is the identity matrix. This diagonal information matrix allows the optimizer to treat constraints with lower confidence as noisier, thereby reducing their influence during global refinement.

## 3.4. Global optimization

Once the pose graph has been constructed and the edge weights have been adjusted based on uncertainty estimates, we perform global optimization to refine the camera poses. Specifically, we apply pose graph optimization at every iteration of the ACE-Zero framework to continuously improve the global consistency of the estimated trajectory. This iterative refinement ensures that local updates to scene coordinates and camera poses are propagated throughout the entire graph, mitigating drift and reinforcing spatial coherence.

By integrating pose graph optimization into each iteration of ACE-Zero, we enable consistent global refinement that complements local updates, resulting in improved accuracy and stability of the reconstructed camera trajectory.

## 4. Experiments

Implementation details. Our pipeline is based on the public implementation of ACE-Zero [8] and all hyperparameters are kept in their default settings. All experiments are initialized using ZoeDepth [4] and executed on a single NVIDIA 3090 GPU. We initialize the geometric consistency threshold at $\tau = 0$ .2m and progressively reduce it to 0.05m over training iterations to enforce stricter correspondence checks as predictions become more accurate. We set $\alpha _ { 1 } = 0 . 4$ and $\alpha _ { 2 } = 0 . 6$ , and scale the information matrix using the median confidence value for numerical stability during optimization. During pose graph optimization, a tight prior is applied to the first pose using a noise model with a standard deviation of $1 0 ^ { - 6 }$ for all components. Between factors are modeled with standard deviation of 0.1 for rotational and translational components. To improve robustness against outliers, we employ a Huber loss function with a scale parameter of 1.0, which reduces the influence of large residuals while preserving sensitivity to small errors. For nodes that remain unconnected after edge filtering, we add loose priors using a standard deviation of 10 for all components to prevent optimization failure without introducing strong constraints. The final optimization is performed using Levenberg–Marquardt optimizer.

Baselines. We treat COLMAP-generated poses as pseudo ground truth. Besides default settings, we also report results using the recommended parameter settings for large image collections, which we denote as COLMAP (fast). We compare against our most relevant method, ACE-Zero (ACE0) [8] as well as two neural SLAM approaches, i.e. DRIOD-SLAM [44] and VGGT-SLAM [27].

Datasets. We follow the evaluation protocol from [8] and experiment on three benchmarks: 7-Scenes [40] for relocalization, Mip-NeRF 360 [3] for view synthesis, and Tanks and Temples [24] for 3D reconstruction.

Evaluation metrics. As obtaining ground truth camera poses in real-world settings is challenging, pseudo ground truth camera poses are estimated rather than measured. Consequently, some methods may outperform COLMAP in certain scenes which aligns with prior findings that pose error computed against COLMAP can be misleading. Following established methods [8, 23], we assess pose quality in a self-supervised manner using novel view synthesis [47]. Specifically, each method estimates poses for all images in a scene. We then split the images into training and test sets, train a Nerfacto [42] model on the training set, and synthesize views for the test poses. The synthesized images are compared to the test images using peak signal-to-noise ratio (PSNR). Pose quality influences PSNR in two ways: accurate training poses enable the NeRF model to learn a consistent scene representation, while accurate test poses ensure alignment between synthesized and original views.

## 4.1. Experimental results

7-Scenes (Tab. 1). This dataset consists seven indoor scenes. It provides pseudo ground-truth camera poses estimated by KinectFusion [20], a depth-based SLAM system. Although KinectFusion operates in real time, its poses yield lower PSNR; thus, we omit these results from our tables and refer readers to [8] for reported numbers. Here, we use COLMAP poses as the reference pseudo ground truth. We show that our approach matches or surpass the performance of both COLMAP and ACE-Zero while requiring significantly less computation time.

<table><tr><td rowspan="2"></td><td rowspan="2">Frames</td><td colspan="2">Pseudo Ground Truth</td><td colspan="4">All Frames</td></tr><tr><td>COLMAP (default)</td><td>COLMAP (fast)</td><td>ACE0 [8]</td><td>Ours (vanilla pgo)</td><td>Ours (w/o uncertainty)</td><td>Ours (full)</td></tr><tr><td>Chess</td><td>6k</td><td>23.6</td><td>23.5</td><td>23.3 (23.1)</td><td>11.2</td><td>19.2</td><td>23.5</td></tr><tr><td>Fire</td><td>4k</td><td>22.6</td><td>22.6</td><td>22.3 (22.0)</td><td>13.3</td><td>19.0</td><td>22.6</td></tr><tr><td>Heads</td><td>2k</td><td>18.8</td><td>18.9</td><td>18.8 (18.6)</td><td>11.9</td><td>17.7</td><td>19.2</td></tr><tr><td>Office</td><td>10k</td><td>21.4</td><td>21.6</td><td>21.1 (21.0)</td><td>12.4</td><td>18.5</td><td>21.6</td></tr><tr><td>Pumpkin</td><td>6k</td><td>24.1</td><td>23.8</td><td>24.1 (24.0)</td><td>14.5</td><td>19.3</td><td>24.3</td></tr><tr><td>RedKitchen</td><td>12k</td><td>21.4</td><td>21.4</td><td>20.8 (20.9)</td><td>11.9</td><td>18.5</td><td>21.2</td></tr><tr><td>Stairs</td><td>3k</td><td>16.7</td><td>21.0</td><td>17.7 (17.3)</td><td>12.7</td><td>17.5</td><td>20.1</td></tr><tr><td>Average</td><td></td><td>21.2</td><td>21.8</td><td>21.2 (21.0)</td><td>12.6</td><td>18.5</td><td>21.7</td></tr><tr><td>Avg. Time</td><td></td><td>38h</td><td>13h</td><td>1h</td><td>1h</td><td>1h</td><td>30min</td></tr></table>

Table 1. Quantitative comparisons on 7-Scenes dataset. We report the pose accuracy via view synthesis as PSNR in dB and the reconstruction time. Dark green indicate the best results. Values shown in parentheses correspond to our self-reproduction results.
<table><tr><td></td><td>COLMAP (default)</td><td>VGGT- SLAM [27]</td><td>ACE0 [8]</td><td>Ours (vanilla pgo)</td><td>Ours (w/o uncertainty)</td><td>Ours (full)</td></tr><tr><td>Bicycle</td><td>21.5</td><td>11.8</td><td>18.7 (18.3)</td><td>12.5</td><td>19.0</td><td>21.9</td></tr><tr><td>Bonsai</td><td>27.6</td><td>16.7</td><td>25.8 (25.5)</td><td>13.9</td><td>22.1</td><td>26.6</td></tr><tr><td>Counter</td><td>25.5</td><td>12.8</td><td>24.5 (24.8)</td><td>14.5</td><td>21.2</td><td>25.2</td></tr><tr><td>Garden</td><td>26.3</td><td>17.9</td><td>25.0 (24.5)</td><td>14.7</td><td>22.5</td><td>26.6</td></tr><tr><td>Kitchen</td><td>27.4</td><td>14.3</td><td>26.1 (26.3)</td><td>14.3</td><td>23.1</td><td>26.8</td></tr><tr><td>Room</td><td>28.0</td><td>12.6</td><td>19.8 (19.5)</td><td>14.3</td><td>19.3</td><td>21.8</td></tr><tr><td>Stump</td><td>16.8</td><td>13.7</td><td>20.5 (16.1)</td><td>12.7</td><td>20.1</td><td>21.4</td></tr><tr><td>Average</td><td>24.7</td><td>14.3</td><td>22.9 (22.14)</td><td>13.8</td><td>21.0</td><td>24.3</td></tr></table>

Table 2. Quantitative comparisons on Mip-NeRF 360 dataset. We report the pose accuracy via view synthesis as PSNR in dB and the reconstruction time. Dark green indicate the best results. Values shown in parentheses correspond to our self-reproduction results.

Mip-NeRF 360 (Tab. 2). This dataset consists of seven small-scale scenes which covers both indoor and outdoor. Similarly, we use COLMAP generated poses as the reference pseudo ground truth. VGGT-SLAM [27] is a recent dense RGB SLAM system constructed by incrementally and globally aligning submaps generated from VGGT [50]. We observe that VGGT-SLAM struggles in this dataset due to the large differences between sequential images. In contrast, ACE-Zero successfully reconstructs the scenes and achieves relatively better performance. Nevertheless, our approach demonstrates improvements over ACE-Zero while maintaining competitive results against COLMAP.

Tanks and Temples (Tab. 3). This dataset comprises 21 diverse indoor and outdoor scenes, covering a substantially larger spatial extent compared to the previous two datasets. We present results for both the short version of the dataset (150–500 images) and the long version (4k–22k images). Across both versions, our proposed method generally outperforms Ace-Zero and achieves performance comparable to COLMAP. Notably, on the Advanced split, VGGT-SLAM delivers similar accuracy to ACE-Zero (13.6 vs.

<table><tr><td rowspan="2" colspan="2"></td><td rowspan="2">Frames</td><td colspan="3">VGGT- SLAM</td><td rowspan="2">FTaes</td><td rowspan="2"></td><td colspan="4">DROID-</td></tr><tr><td>COLMAP (default)</td><td>ACE0 [27] [8]</td><td>Ours (full)</td><td>COLMAP (fast)</td><td>SLAM [44]</td><td>ACE0 [8]</td><td>Ours (full)</td></tr><tr><td rowspan="9">Training</td><td>Barn</td><td>410</td><td>24.0</td><td>17.2</td><td>16.4</td><td>18.5</td><td>12.2k</td><td>24.4</td><td>13.5</td><td>18.4</td><td>20.1</td></tr><tr><td>Caterpillar</td><td>383</td><td>17.1</td><td>14.1</td><td>16.9</td><td>17.0</td><td>11.4k</td><td>18.6</td><td>18.9</td><td>18.6</td><td>18.6</td></tr><tr><td>Church</td><td>507</td><td>18.3</td><td>13.5</td><td>17.1</td><td>17.1</td><td>19.3k</td><td>12.1</td><td>11.5</td><td>16.8</td><td>17.3</td></tr><tr><td>Ignatius</td><td>264</td><td>20.1</td><td>15.1</td><td>19.6</td><td>19.6</td><td>7.8k</td><td>20.8</td><td>19.1</td><td>20.6</td><td>20.2</td></tr><tr><td>Meetingroom 371</td><td></td><td>18.6</td><td>14.5</td><td>16.6</td><td>17.2</td><td>11.1k</td><td>19.4</td><td>17.1</td><td>18.4</td><td>19.3</td></tr><tr><td>Truck</td><td>251</td><td>21.1</td><td>16.0</td><td>20.1</td><td>21.0</td><td>7.5k</td><td>23.6</td><td>20.6</td><td>22.7</td><td>22.7</td></tr><tr><td>Average</td><td>364</td><td>19.9</td><td>15.1</td><td>17.8</td><td>18.4</td><td>14.6k</td><td>19.8</td><td>16.8</td><td>19.3</td><td>19.7</td></tr><tr><td>Avg Time</td><td></td><td>1h</td><td>5min</td><td>1.1h</td><td>30min</td><td></td><td>74h</td><td>18min</td><td>2.2h</td><td>1h</td></tr><tr><td rowspan="9"></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Family</td><td>152</td><td>19.5</td><td>16.6</td><td>18.9</td><td>19.1</td><td>4.4k</td><td>21.2</td><td>19.8</td><td>17.9</td><td>20.2</td></tr><tr><td>Francis</td><td>302</td><td>21.6</td><td>17.4</td><td>20.0</td><td>21.0</td><td>7.8k</td><td>19.9</td><td>21.8</td><td>21.8</td><td>22.1</td></tr><tr><td>Horse</td><td>151</td><td>19.2</td><td>15.9</td><td>19.3</td><td>19.2</td><td>6.0k</td><td>21.6</td><td>19.2</td><td>21.5</td><td>21.5</td></tr><tr><td>Lighthouse</td><td>309</td><td>16.6</td><td>12.0</td><td>17.0</td><td>17.8</td><td>8.3k</td><td>19.0</td><td>18.9</td><td>18.0</td><td>18.8</td></tr><tr><td>Playground</td><td>307</td><td>19.1</td><td>14.1</td><td>18.5</td><td>19.0</td><td>7.7k</td><td>17.9</td><td>11.3</td><td>20.6</td><td>21.0</td></tr><tr><td>Train</td><td>301</td><td>16.8</td><td>12.4</td><td>16.0</td><td>16.6</td><td>12.6k</td><td>19.6</td><td>15.6</td><td>17.6</td><td>18.3</td></tr><tr><td>Average</td><td>254</td><td>18.8</td><td>14.7</td><td>18.3</td><td>18.8</td><td>7.8k</td><td>19.9</td><td>17.8</td><td>19.6</td><td>20.3</td></tr><tr><td>Avg Time</td><td></td><td>32min</td><td>5min</td><td>1.3h</td><td>30min</td><td></td><td>48h</td><td></td><td>14min 2.2h</td><td></td><td>1.3h</td></tr><tr><td rowspan="7"></td><td>Auditorium</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Ballroom</td><td>302 324</td><td>19.6</td><td>14.7</td><td>18.0</td><td>19.1</td><td>13.6k</td><td>13.7</td><td>16.6</td><td>20.0</td><td>20.9</td></tr><tr><td></td><td></td><td>16.3</td><td>12.7</td><td>17.9</td><td>18.3</td><td>10.8k</td><td>17.2</td><td>10.4</td><td>18.7</td><td>18.9</td></tr><tr><td>Courtroom</td><td>301</td><td>18.2</td><td>13.5</td><td>16.9</td><td>17.4</td><td>12.6k</td><td>14.6</td><td>10.2</td><td>16.5</td><td>17.9</td></tr><tr><td>Palace</td><td>509</td><td>14.2</td><td>13.0</td><td>10.0</td><td>13.6</td><td>21.9k</td><td>13.8</td><td>8.6</td><td>11.0</td><td>13.8</td></tr><tr><td>Temple</td><td>302</td><td>18.1</td><td>14.0</td><td>9.9</td><td>15.7</td><td>17.5k</td><td>13.3</td><td>11.9</td><td>14.5</td><td>16.7</td></tr><tr><td>Average</td><td>348</td><td>17.3</td><td>13.6</td><td>14.5</td><td>16.8</td><td>15.6k</td><td>14.5</td><td>11.5</td><td>16.1</td><td>17.6</td></tr><tr><td>Avg Time</td><td></td><td></td><td>1h</td><td>5min</td><td>1h</td><td>30min</td><td></td><td>71h</td><td>27min</td><td>2.8h</td><td>1.5h</td></tr></table>

Table 3. Quantitative comparisons on Tanks and Temples dataset. We report the pose accuracy via view synthesis as PSNR in dB and the reconstruction time. Dark green indicate the best results, excluding COLMAP.

14.5 PSNR on average) while requiring significantly less reconstruction time. For the long version, we attempt to downsample sequence to around 4k frames for VGGT-SLAM in order fit GPU memory. However, we observe fracture scenes for all sequences, so their performance is not reported. We provide qualitative comparisons in Fig. 4.

## 4.2. Ablations

Pose graph optimization (Fig. 5(a)). We ablate key components of our pipeline on 7-Scenes and Mip-NeRF 360 datasets. Vanilla PGO without graph construction performs poorly due to unreliable edges (12.6dB, 13.8dB). Adding graph construction significantly improves accuracy (>5dB), and incorporating epipolar-based uncertainty yields moderate gains. Optical flow further boosts robustness by providing local consistency. Applying PGO only once at the end underperforms compared to progressive integration (20.5dB vs. 21.7dB; 23.5dB vs. 24.3dB), highlighting the importance of iterative global refinement.

Pose graph construction (Fig. 5(b)). We evaluate the impact of fixing the geometric consistency threshold τ at 0.05m, 0.1m and 0.2m. We show that τ = 0.1m achieves the best performance among fixed settings, balancing accuracy and graph connectivity, while τ = 0.2m admits noisy matches that degrade global consistency. We also tested enforcing bidirectional consistency when constructing pose graph edges. Symmetric checks slightly improve PSNR in dense indoor scenes with rich texture, as they yield more reliable edges. However, the gains are modest (< 5%) overall and performance can degrade in sparser settings because stricter filtering reduces graph connectivity, leading to fewer edges and weaker global consistency. Given the limited benefit and added complexity, we adopt one-way checks in our default pipeline.

![](images/5fc172f08483d1f4b458635f2d1dc5624869baec1d980b2de3773e11038b593c.jpg)  
Ground truth  
VGGT-SLAM  
ACE-Zero

Ours  
Figure 4. Qualitative comparison on Mip-NeRF 360 (top two rows) and Tanks and Temples datasets (bottom two rows) with VGGT SLAM [27] and ACE-Zero [8]. While ACE-Zero performs well on small-scale scenes such as Mip-NeRF 360, it struggles with large, complex environments like Tanks and Temples, where our method maintains high rendering quality across diverse scenes.  
![](images/b71abeecb3b92b97a0d458fa0be7e34472ce1b04e046597c5366e677ea2f4919.jpg)

(b) Sensitivity to Threshold τ Across Two Datasets (c) Mip-NeRF 360 dataset (evaluate every 8th image  
![](images/f661abb9356ef103ff59590186c90badc1986b94b63994c4162adfb5c366700b.jpg)  
Figure 5. Ablations on (a) design components, and sensitivity to (b) hyperparameter and (c) sparse training data.

Impact on less input images (Fig. 5(c)). Our method demonstrates better robustness to limited training data.

Runtime breakdown. Our added PGO step takes about 30s per iteration, compared to 150s for vanilla ACE-Zero. Despite the extra computation, PGO reduces the total number of iterations by roughly half. We provide detailed breakdown in the supplementary material.

## 5. Conclusion

We presented a hybrid framework that integrates pose graph optimization into ACE-Zero for learning scene coordinate reconstruction from unposed images. By progressively enforcing global consistency and incorporating uncertaintyaware weighting, our approach significantly improves pose accuracy and robustness across diverse datasets. This work demonstrates that combining neural inference with geometric reasoning yields scalable and reliable reconstructions. Despite strong results, our method has several limitations. First, uncertainty estimation relies on heuristic metrics, which may not generalize well to highly dynamic or textureless scenes. Second, the hyperparameters requires manual tunning. Third, while our approach improves global consistency, it does not jointly optimize scene structure, leaving room for future work on full bundle adjustment.

## References

[1] Vassileios Balntas, Shuda Li, and Victor Prisacariu. Relocnet: Continuous metric learning relocalisation using neural nets. In ECCV, 2018. 3

[2] Daniel Barath, Dmytro Mishkin, Ivan Eichhardt, Ilia Shipachev, and Jiri Matas. Efficient initial pose-graph generation for global sfm. In CVPR, 2021. 2

[3] Jonathan T Barron, Ben Mildenhall, Dor Verbin, Pratul P Srinivasan, and Peter Hedman. Mip-nerf 360: Unbounded anti-aliased neural radiance fields. In CVPR, 2022. 6

[4] Shariq Farooq Bhat, Reiner Birkl, Diana Wofk, Peter Wonka, and Matthias Muller. Zoedepth: Zero-shot trans-¨ fer by combining relative and metric depth. arXiv preprint arXiv:2302.12288, 2023. 3, 6

[5] Wenjing Bian, Axel Barroso-Laguna, Tommaso Cavallari, Victor Adrian Prisacariu, and Eric Brachmann. Scene coordinate reconstruction priors. In ICCV, 2025. 3

[6] Eric Brachmann, Alexander Krull, Sebastian Nowozin, Jamie Shotton, Frank Michel, Stefan Gumhold, and Carsten Rother. Dsac-differentiable ransac for camera localization. In CVPR, 2017. 3

[7] Eric Brachmann, Tommaso Cavallari, and Victor Adrian Prisacariu. Accelerated coordinate encoding: Learning to relocalize in minutes using rgb and poses. In CVPR, 2023. 3, 12, 13

[8] Eric Brachmann, Jamie Wynn, Shuai Chen, Tommaso Cavallari, Aron Monszpart, Daniyar Turmukhambetov, and Vic-<sup>´</sup> tor Adrian Prisacariu. Scene coordinate reconstruction: Posing of image collections via incremental learning of a relocalizer. ECCV, 2024. 1, 2, 3, 6, 7, 8, 12, 13, 17

[9] Leonard Bruns, Axel Barroso-Laguna, Tommaso Cavallari, Aron Monszpart, Sowmya Munukutla, Victor Prisacariu, and <sup>´</sup> Eric Brachmann. ACE-G: Improving generalization of scene coordinate regression through query pre-training. In ICCV, 2025. 3

[10] Lucas Brynte, Jose Pedro Iglesias, Carl Olsson, and Fredrik´ Kahl. Learning structure-from-motion with graph attention networks. In CVPR, 2024. 2

[11] Luca Carlone, Rosario Aragues, Jose A Castellanos, and´ Basilio Bona. A fast and accurate approximation for planar pose graph optimization. The International Journal of Robotics Research, 2014. 1

[12] Luca Carlone, Roberto Tron, Kostas Daniilidis, and Frank Dellaert. Initialization techniques for 3d slam: A survey on rotation estimation and its use in pose graph optimization. In ICRA, 2015. 1

[13] Tommaso Cavallari, Stuart Golodetz, Nicholas A Lord, Julien Valentin, Luigi Di Stefano, and Philip HS Torr. Onthe-fly adaptation of regression forests for online camera relocalisation. In CVPR, 2017. 3

[14] Frank Dellaert. Factor graphs and gtsam: A hands-on introduction. Georgia Institute of Technology, Tech. Rep, 2012. 2

[15] EmergentMind. Sim(3) pose graphs in slam & sfm. EmergentMind, 2025. 2

[16] Runyang Feng, Yixing Gao, Tze Ho Elden Tse, Xueqing Ma, and Hyung Jin Chang. Diffpose: Spatiotemporal dif-

fusion model for video-based human pose estimation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023. 1

[17] Martin A Fischler and Robert C Bolles. Random sample consensus: a paradigm for model fitting with applications to image analysis and automated cartography. Communications ofthe ACM, 1981. 3

[18] Xiao-Shan Gao, Xiao-Rong Hou, Jianliang Tang, and Hang-Fei Cheng. Complete solution classification for the perspective-three-point problem. TPAMI, 2003. 3

[19] Richard Hartley and Andrew Zisserman. Multiple view geometry in computer vision. Cambridge university press, 2003. 2

[20] Shahram Izadi, David Kim, Otmar Hilliges, David Molyneaux, Richard Newcombe, Pushmeet Kohli, Jamie Shotton, Steve Hodges, Dustin Freeman, Andrew Davison, et al. Kinectfusion: real-time 3d reconstruction and interaction using a moving depth camera. In UIST, 2011. 6

[21] Zhaodong Jiang, Ashish Sinha, Tongtong Cao, Yuan Ren, Bingbing Liu, and Binbin Xu. Unpose: Uncertainty-guided diffusion priors for zero-shot pose estimation. CoRL, 2025. 2

[22] Alex Kendall, Matthew Grimes, and Roberto Cipolla. Posenet: A convolutional network for real-time 6-dof camera relocalization. In ICCV, 2015. 2

[23] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuhler,¨ and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Transactions on Graphics (TOG), 2023. 6

[24] Arno Knapitsch, Jaesik Park, Qian-Yi Zhou, and Vladlen Koltun. Tanks and temples: Benchmarking large-scale scene reconstruction. ACM Transactions on Graphics (ToG), 2017. 6

[25] Ming-Feng Li, Xin Yang, Fu-En Wang, Hritam Basak, Yuyin Sun, Shreekant Gayaka, Min Sun, and Cheng-Hao Kuo. Uapose: Uncertainty-aware 6d object pose estimation and on line object completion. CVPR, 2025. 2

[26] Xinyi Li and Haibin Ling. Pogo-net: Pose graph optimiza tion with graph neural networks. In ICCV, 2021. 2

[27] Dominic Maggio, Hyungtae Lim, and Luca Carlone. Vggtslam: Dense rgb slam optimized on the sl (4) manifold. arXiv preprint arXiv:2505.12549, 2025. 6, 7, 8, 12, 13, 17

[28] Nathaniel Merrill, Yuliang Guo, Xingxing Zuo, Xinyu Huang, Stefan Leutenegger, Xi Peng, Liu Ren, and Guo quan Huang. Symmetry and uncertainty-aware object slam for 6dof object pose estimation. In CVPR, 2022. 2

[29] Raul Mur-Artal, Jose Maria Martinez Montiel, and Juan D Tardos. Orb-slam: A versatile and accurate monocular slam system. IEEE transactions on robotics, 2015. 2

[30] John Oliensis. A critique of structure-from-motion algo rithms. Computer Vision and Image Understanding, 2000. 2

[31] Onur Ozyes¸il, Vladislav Voroninski, Ronen Basri, and Amit <sup>¨</sup> Singer. A survey of structure from motion\*. Acta Numerica, 2017. 2

[32] Jizong Peng, Tze Ho Elden Tse, Kai Xu, Wenchao Gao, and Angela Yao. A constrained optimization approach for gaus-

sian splatting from coarsely-posed images and noisy lidar point clouds. In ICCV, 2025. 1

[33] Paul-Edouard Sarlin, Cesar Cadena, Roland Siegwart, and Marcin Dymczyk. From coarse to fine: Robust hierarchical localization at large scale. In CVPR, 2019. 2

[34] Torsten Sattler, Bastian Leibe, and Leif Kobbelt. Fast imagebased localization using direct 2d-to-3d matching. In ICCV, 2011. 2

[35] Torsten Sattler, Bastian Leibe, and Leif Kobbelt. Improving image-based localization by active correspondence search. In ECCV, 2012.

[36] Torsten Sattler, Bastian Leibe, and Leif Kobbelt. Efficient & effective prioritized matching for large-scale image-based localization. TPAMI, 2016. 2

[37] Torsten Sattler, Akihiko Torii, Josef Sivic, Marc Pollefeys, Hajime Taira, Masatoshi Okutomi, and Tomas Pajdla. Are large-scale 3d models really necessary for accurate visual localization? In CVPR, 2017. 2

[38] Torsten Sattler, Qunjie Zhou, Marc Pollefeys, and Laura Leal-Taixe. Understanding the limitations of cnn-based absolute camera pose regression. In CVPR, 2019. 3

[39] Johannes Lutz Schonberger and Jan-Michael Frahm.¨ Structure-from-motion revisited. In CVPR, 2016. 2

[40] Jamie Shotton, Ben Glocker, Christopher Zach, Shahram Izadi, Antonio Criminisi, and Andrew Fitzgibbon. Scene coordinate regression forests for camera relocalization in rgb-d images. In CVPR, 2013. 3, 6

[41] Pengzhan Sun, Junbin Xiao, Tze Ho Elden Tse, Yicong Li, Arjun Akula, and Angela Yao. Visual intention grounding for egocentric assistants. In ICCV, 2025. 1

[42] Matthew Tancik, Ethan Weber, Evonne Ng, Ruilong Li, Brent Yi, Terrance Wang, Alexander Kristoffersen, Jake Austin, Kamyar Salahi, Abhik Ahuja, et al. Nerfstudio: A modular framework for neural radiance field development. In ACM SIGGRAPH, 2023. 6

[43] Zachary Teed and Jia Deng. Raft: Recurrent all-pairs field transforms for optical flow. In ECCV, 2020. 5

[44] Zachary Teed and Jia Deng. Droid-slam: Deep visual slam for monocular, stereo, and rgb-d cameras. NeurIPS, 2021. 6, 7, 12, 13

[45] Tze Ho Elden Tse, Daniele De Martini, and Letizia Marchegiani. No need to scream: Robust sound-based speaker localisation in challenging scenarios. In International Conference on Social Robotics, 2019. 1

[46] Mehmet Ozgur Turkoglu, Eric Brachmann, Konrad Schindler, Gabriel J Brostow, and Aron Monszpart. Visual camera re-localization using graph neural networks and relative pose supervision. In 3DV, 2021. 3

[47] Michael Waechter, Mate Beljan, Simon Fuhrmann, Nils Moehrle, Johannes Kopf, and Michael Goesele. Virtual rephotography: Novel view prediction error for 3d reconstruction. ACM Transactions on Graphics (TOG), 2017. 6

[48] Fangjinhua Wang, Xudong Jiang, Silvano Galliani, Christoph Vogel, and Marc Pollefeys. Glace: Global local accelerated coordinate encoding. In CVPR, 2024. 3

[49] Jianyuan Wang, Nikita Karaev, Christian Rupprecht, and David Novotny. Vggsfm: Visual geometry grounded deep structure from motion. In CVPR, 2024. 1, 2

[50] Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, and David Novotny. Vggt: Visual geometry grounded transformer. In CVPR, 2025. 1, 2, 7

[51] Xingkui Wei, Yinda Zhang, Zhuwen Li, Yanwei Fu, and Xiangyang Xue. Deepsfm: Structure from motion via deep bundle adjustment. In ECCV, 2020. 2, 3

[52] Zhe Xin, Yufeng Yue, Liangjun Zhang, and Chenming Wu. Hero-slam: Hybrid enhanced robust optimization of neural slam. arXiv preprint arXiv:2407.18813, 2024. 3

[53] Kai Xu, Tze Ho Elden Tse, Jizong Peng, and Angela Yao. Das3r: Dynamics-aware gaussian splatting for static scene reconstruction. arXiv preprint arXiv:2412.19584, 2024. 1

## 6. Appendix

In this supplemental document, we provide:

• extended implementation details and discussion (Sec $\mathrm { A } ) ;$

• additional analysis (Sec B);

• additional qualitative examples (Sec C).

## A. Extended implementation details and discussion

Overall algorithm. We present the complete pseudocode algorithm in Algorithm 1. Our key contribution is to suppress incorrectly refined camera poses, enable a more accurate and globally consistent set of poses. This improved alignment allows the scene coordinate regression network to be trained on a more reliable pose set in subsequent iterations. Consequently, the overall reconstruction process converges faster by reaching accurate camera poses earlier, supported by pose graph optimization.

Impact of graph sparsity or connectivity on optimization quality. Pose graph connectivity plays a critical role in the success of global optimization. A sparse graph, where only a few reliable edges exist, limits the propagation of global constraints and can lead to fragmented or weakly consistent pose estimates. Conversely, an overly dense graph may include noisy or incorrect edges, which can degrade optimization quality by introducing conflicting constraints. We observed that:

• Sparse Graphs: When the geometric consistency threshold τ is set too low, many valid correspondences are rejected, resulting in fewer edges. This reduces the optimizer’s ability to enforce global consistency, especially in large-scale scenes with repetitive structures.

• Dense Graphs: A high τ admits more matches, increasing connectivity but also introducing outliers. These noisy edges can bias the optimization, even when robust loss functions are applied.

• Balanced Connectivity: Our experiments (see Fig. 5(b) in the main paper) show that τ = 0.1 m achieves the best trade-off between accuracy and connectivity. This setting provides sufficient edges for global consistency while minimizing the inclusion of unreliable matches.

Can the uncertainty-aware metrics provide supervision signal to model training? Our current framework uses uncertainty-aware metrics exclusively for weighting constraints during pose graph optimization. These metrics— epipolar error and optical flow consistency—quantify geometric reliability between image pairs without requiring ground truth poses. This raises an interesting question: can these metrics also serve as supervision signals for training the ACE model? We summarize the potential benefits:

```latex
Algorithm 1: Overall algorithm
Input: Unordered image set $T = \{ I _ { 1 } , I _ { 2 } , \ldots , I _ { n } \}$
Output: Refined camera poses $\{ T _ { i } \}$ and trained
scene coordinate regression network $\Phi _ { \theta }$
1 Initialization:
Select seed images and predict depth maps using
ZoeDepth.
3 Generate initial pseudo-labels for scene
coordinates.
4 Iterative Reconstruction Loop:
5 for each iteration t do
6 (1) Model training
7 Train scene coordinate regression network $\Phi _ { \theta }$
on current set of images with pseudo-poses.
8 (2) Pose estimation with previously trained
model
9 Use trained scene coordinate regression
network $\Phi _ { \theta }$ to estimate poses for all images.
10 Update pseudo-poses for unregistered images.
11 (3) Scene coordinate prediction
12 Predict dense scene coordinates $X _ { i } = \Phi _ { \theta } ( I _ { i } )$
for each image.
13 (4) Pose Graph Construction
14 for each image pair $( I _ { i } , I _ { j } )$ do
15 Find geometrically consistent matches:
project $X _ { i }$ into $I _ { j } ,$ compute
$e _ { \mathrm { m a t c h } } = \| X _ { j } - X _ { i } \| .$ , accept if $e _ { \mathrm { m a t c h } } < \tau .$
16 Estimate relative pose $Z _ { i j }$ using PnP +
RANSAC.
17 Add edge $( i , j )$ with $Z _ { i j }$ to pose graph.
18 (5) Uncertainty estimation
19 Compute epipolar error $e _ { \mathrm { e p i } }$ and optical flow
error $e _ { \mathrm { f l o w } } .$
20 Confidence score $\sigma = \alpha _ { 1 } e _ { \mathrm { e p i } } + \alpha _ { 2 } e _ { \mathrm { f l o w } }$
21 Aggregate σ to edge-level uncertainty $\sigma _ { i j }$ and
compute $\Omega _ { i j } = \mathrm { d i a } \bar { \bf g } ( 1 / ( \sigma _ { i j } ^ { 2 } + \epsilon ) )$
22 (6) Pose Graph Optimization
23 Solve
min $\{ T _ { i } \} \sum _ { - } ( { \it i } , { \it j } ) \in E { \Vert } \mathrm { L o g } ( Z _ { i j } ^ { - 1 } T _ { i } ^ { - 1 } T _ { j } ) { \Vert } _ { \Omega _ { i j } } ^ { 2 }$
using LM optimizer with Huber loss.
24 (7) Check stopping criteria
25 Stop if 99% images registered $0 \mathrm { r } < 1 \%$ new.
```

• Self-Supervised Learning: Epipolar and flow-based consistency could act as additional loss terms, encouraging the ACE model to predict scene coordinates that satisfy multi-view geometric constraints.

• Improved Robustness: Incorporating these metrics during training may reduce reliance on pseudo-labels and improve performance in ambiguous or textureless regions.

• Dynamic Weighting: Confidence scores could guide curriculum learning, where reliable correspondences are emphasized early in training.

One major challenge is non-differentiability: epipolar distance and optical flow errors involve discrete matching and RANSAC steps, making direct backpropagation difficult. Additionally, these metrics depend on estimated poses, which may be inaccurate during early iterations, potentially introducing bias into the training process. Finally, computing optical flow and epipolar constraints for all image pairs during training would significantly increase computational overhead, making the approach less practical for large-scale datasets.

Future work could explore differentiable approximations of epipolar and flow consistency, enabling their integration into end-to-end training. Another promising direction is to incorporate confidence-based weighting into ACE’s reprojection loss, allowing uncertainty-aware metrics to influence optimization directly. Finally, curriculum strategies could be investigated, where uncertainty-aware metrics gradually gain importance as pose estimates become more reliable, balancing computational cost and training stability.

## B. Additional analysis

Details of the teaser example. For the teaser figure, we use the the example video office loop from [27] which consists of 473 frames capturing movement around an office environment. We provide the full qualitative comparison in Fig. 6 with sample image frames in Fig. 7. As illustrated in Fig. 7, ACE-Zero exhibits suboptimal performance, achieving PSNR values of 19.1 and 15.6 with and without COLMAP initialization, respectively. This highlights that the vanilla ACE-Zero configuration can incorrectly refine camera poses and fails to recover, even with additional iterations and accurate initialization from COLMAP. Furthermore, VGGT-SLAM, despite correctly identifying loop closures, does not recover sufficiently accurate camera poses, yielding a PSNR of 17.1. In contrast, our method achieves a PSNR of 25.6, which is comparable to COLMAP’s PSNR of 28.1. In addition, we provide qualitative comparisons in Fig. 8.

Full experimental results. We present both SSIM and LPIPS scores for 7-Scenes and Mip-NeRF 360 datasets in Table 4 and 5, respectively. We also present SSIM and LPIPS scores for Tanks and Temples dataset in Table 6 and 7, respectively. We observe that both SSIM and LPIPS scores behave similarly to the PSNR results presented in the main script.

Reolcalization performance on 7-Scenes (Table 8). We compare with the supervised relocalizer ACE [7] by training with both KinectFusion and COLMAP mapping poses. As pointed out in [8], ACE achieves almost perfect relocalization rate under the 5cm, 5<sup>◦</sup> error threshold as the evaluation is performed against COLMAP query poses. We show that our approach can further improve the relocalization performance by providing a more globally aligned poses.

<table><tr><td rowspan="2"></td><td rowspan="2">Frames</td><td colspan="2">Pseudo Ground Truth</td><td colspan="4">All Frames</td></tr><tr><td>COLMAP SSIM (↑)</td><td>COLMAP LPIPS (↓)</td><td>ACE0 SSIM (↑)</td><td>ACE0 LPIPS (↓)</td><td>Ours SSIM (↑)</td><td>Ours LPIPS (↓)</td></tr><tr><td>Chess</td><td>6k</td><td>0.86</td><td>0.17</td><td>0.84</td><td>0.19</td><td>0.85</td><td>0.18</td></tr><tr><td>Fire</td><td>4k</td><td>0.73</td><td>0.27</td><td>0.68</td><td>0.30</td><td>0.73</td><td>0.26</td></tr><tr><td>Heads</td><td>2k</td><td>0.82</td><td>0.26</td><td>0.80</td><td>0.28</td><td>0.84</td><td>0.25</td></tr><tr><td>Office</td><td>10k</td><td>0.85</td><td>0.25</td><td>0.82</td><td>0.27</td><td>0.86</td><td>0.24</td></tr><tr><td>Pumpkin</td><td>6k</td><td>0.84</td><td>0.23</td><td>0.83</td><td>0.24</td><td>0.87</td><td>0.22</td></tr><tr><td>RedKitchen</td><td>12k</td><td>0.80</td><td>0.25</td><td>0.77</td><td>0.29</td><td>0.78</td><td>0.27</td></tr><tr><td>Stairs</td><td>3k</td><td>0.59</td><td>0.50</td><td>0.62</td><td>0.45</td><td>0.76</td><td>0.29</td></tr><tr><td>Average</td><td></td><td>0.78</td><td>0.28</td><td>0.76</td><td>0.29</td><td>0.81</td><td>0.24</td></tr></table>

Table 4. Quantitative comparisons on 7-Scenes dataset. We report the pose accuracy via view synthesis as SSIM (↑) and LPIPS (↓). Dark green indicate the best results. We report results for COLMAP (default).

<table><tr><td colspan="5">SSIM (↑)</td><td colspan="4">LPIPS (↓)</td></tr><tr><td></td><td>COLMAP (default)</td><td>VGGT- SLAM [27]</td><td>ACE0 [8]</td><td>Ours (full)</td><td>COLMAP (default)</td><td>VGGT- SLAM [27]</td><td>ACE0 [8]</td><td>Ours (full)</td></tr><tr><td>Bicycle</td><td>0.68</td><td>0.15</td><td>0.46</td><td>0.70</td><td>0.20</td><td>0.71</td><td>0.36</td><td>0.18</td></tr><tr><td>Bonsai</td><td>0.89</td><td>0.37</td><td>0.84</td><td>0.85</td><td>0.06</td><td>0.59</td><td>0.10</td><td>0.08</td></tr><tr><td>Counter</td><td>0.82</td><td>0.22</td><td>0.77</td><td>0.80</td><td>0.10</td><td>0.70</td><td>0.13</td><td>0.12</td></tr><tr><td>Garden</td><td>0.86</td><td>0.27</td><td>0.77</td><td>0.87</td><td>0.08</td><td>0.51</td><td>0.12</td><td>0.07</td></tr><tr><td>Kitchen</td><td>0.87</td><td>0.29</td><td>0.80</td><td>0.84</td><td>0.06</td><td>0.77</td><td>0.10</td><td>0.08</td></tr><tr><td>Room</td><td>0.91</td><td>0.30</td><td>0.58</td><td>0.68</td><td>0.05</td><td>0.73</td><td>0.43</td><td>0.22</td></tr><tr><td>Stump</td><td>0.40</td><td>0.15</td><td>0.20</td><td>0.65</td><td>0.45</td><td>0.61</td><td>0.54</td><td>0.29</td></tr><tr><td>Average</td><td>0.78</td><td>0.25</td><td>0.63</td><td>0.77</td><td>0.14</td><td>0.66</td><td>0.25</td><td>0.15</td></tr></table>

Table 5. Quantitative comparisons on Mip-NeRF 360 dataset. We report the pose accuracy via view synthesis as SSIM (↑) and LPIPS (↓). Dark green indicate the best results. We report results for COLMAP (default).

<table><tr><td rowspan="2" colspan="2"></td><td rowspan="2">Fraes</td><td colspan="3">VGGT- SLAM</td><td rowspan="2">Frames</td><td rowspan="2">COLMAP</td><td colspan="4">DROID-</td></tr><tr><td>COLMAP (default) [27]</td><td>ACE0 [8]</td><td>Ours (full)</td><td>(fast)</td><td>SLAM [44]</td><td>ACE0 [8]</td><td>Ours (full)</td></tr><tr><td rowspan="9">Training</td><td>Barn</td><td>410</td><td>0.77</td><td>0.50</td><td>0.54</td><td>0.68</td><td>12.2k</td><td>0.77</td><td>0.49</td><td>0.60</td><td>0.68</td></tr><tr><td>Caterpillar</td><td>383</td><td>0.63</td><td>0.34</td><td>0.56</td><td>0.62</td><td>11.4k</td><td>0.72</td><td>0.62</td><td>0.67</td><td>0.67</td></tr><tr><td>Church</td><td>507</td><td>0.64</td><td>0.36</td><td>0.57</td><td>0.57</td><td>19.3k</td><td>0.29</td><td>0.34</td><td>0.56</td><td>0.63</td></tr><tr><td>Ignatius</td><td>264</td><td>0.69</td><td>0.32</td><td>0.65</td><td>0.65</td><td>7.8k</td><td>0.77</td><td>0.61</td><td>0.74</td><td>0.62</td></tr><tr><td>Meetingroom 371</td><td></td><td>0.67</td><td>0.41</td><td>0.55</td><td>0.59</td><td>11.1k</td><td>0.67</td><td>0.55</td><td>0.62</td><td>0.67</td></tr><tr><td>Truck</td><td>251</td><td>0.76</td><td>0.44</td><td>0.71</td><td>0.73</td><td>7.5k</td><td>0.83</td><td>0.69</td><td>0.79</td><td>0.80</td></tr><tr><td>Average</td><td>364</td><td>0.69</td><td>0.40</td><td>0.60</td><td>0.64</td><td>14.6k</td><td>0.68</td><td>0.55</td><td>0.66</td><td></td></tr><tr><td>Avg Time</td><td></td><td>1h</td><td>5min</td><td>1.1h</td><td>30min</td><td></td><td>74h</td><td>18min</td><td>2.2h</td><td>0.68 1h</td></tr><tr><td colspan="9"></td><td></td><td></td></tr><tr><td rowspan="9">Inuteiate</td><td>Family</td><td>152</td><td>0.77</td><td>0.53</td><td>0.69</td><td></td><td>4.4k</td><td>0.83</td><td>0.67</td><td>0.69</td><td>0.71</td></tr><tr><td>Francis</td><td>302</td><td>0.84</td><td>0.62</td><td>0.75</td><td>0.78</td><td>7.8k</td><td>0.77</td><td>0.80</td><td>0.75</td><td>0.81</td></tr><tr><td>Horse</td><td>151</td><td>0.77</td><td>0.51</td><td>0.73</td><td>0.71</td><td>6.0k</td><td>0.79</td><td>0.69</td><td>0.73</td><td>0.78</td></tr><tr><td>Lighthouse</td><td>309</td><td>0.65</td><td>0.47</td><td>0.62</td><td>0.69</td><td>8.3k</td><td>0.70</td><td>0.68</td><td>0.62</td><td>0.67</td></tr><tr><td>Playground</td><td>307</td><td>0.54</td><td>0.33</td><td>0.48</td><td>0.52</td><td>7.7k</td><td>0.53</td><td>0.32</td><td>0.48</td><td>0.63</td></tr><tr><td>Train</td><td>301</td><td>0.62</td><td>0.30</td><td>0.52</td><td>0.59</td><td>12.6k</td><td>0.73</td><td>0.51</td><td>0.52</td><td>0.62</td></tr><tr><td>Average</td><td>254</td><td>0.70</td><td>0.46</td><td>0.63</td><td>0.66</td><td>7.8k</td><td>0.73</td><td></td><td></td><td></td></tr><tr><td>Avg Time</td><td></td><td>32min</td><td>5min</td><td>1.3h</td><td>30min</td><td></td><td>48h</td><td>0.61 14min</td><td>0.63 2.2h</td><td>0.70 1.3h</td></tr><tr><td colspan="9"></td><td></td><td></td></tr><tr><td rowspan="9">Adavanned</td><td>Auditorium</td><td>302</td><td>0.74</td><td>0.54</td><td>0.63</td><td>0.69</td><td>13.6k</td><td>0.50</td><td>0.58</td><td>0.63</td><td>0.66</td></tr><tr><td>Ballroom</td><td>324</td><td>0.46</td><td>0.24</td><td>0.49</td><td>0.65</td><td>10.8k</td><td>0.48</td><td>0.25</td><td>0.49</td><td>0.57</td></tr><tr><td>Courtroom</td><td>301</td><td>0.61</td><td>0.36</td><td>0.51</td><td>0.57</td><td>12.6k</td><td>0.43</td><td>0.32</td><td>0.51</td><td>0.59</td></tr><tr><td>Palace</td><td>509</td><td>0.54</td><td>0.45</td><td>0.24</td><td>0.49</td><td>21.9k</td><td>0.50</td><td>0.38</td><td>0.24</td><td>0.50</td></tr><tr><td>Temple</td><td>302</td><td>0.68</td><td>0.45</td><td>0.25</td><td>0.53</td><td>17.5k</td><td>0.44</td><td>0.45</td><td>0.25</td><td>0.63</td></tr><tr><td>Average</td><td>348</td><td>0.61</td><td>0.41</td><td>0.42</td><td>0.59</td><td>15.6k</td><td>0.47</td><td>0.39</td><td>0.42</td><td>0.59</td></tr><tr><td>Avg Time</td><td></td><td>1h</td><td>5min</td><td>1h</td><td>30min</td><td></td><td>71h</td><td>27min</td><td>2.8h</td><td>1.5h</td></tr></table>

Table 6. Quantitative comparisons on Tanks and Temples dataset. We report the pose accuracy via view synthesis as SSIM (↑). Dark green indicate the best results, excluding COLMAP.

![](images/bb17186408d502e28cbc7ad37adc4de232969c37f8bf0c7b14d1fef608532a20.jpg)  
Figure 6. Despite being initialized from (a) COLMAP generated camera poses, local refinements in (b) often introduce global inconsis tencies. In contrast, (c) our hybrid-framework, which integrates pose graph optimization into ACE-Zero, enforces multi-view consistency and suppress incorrect refinements. Furthermore, we demonstrate that (d) ACE-Zero fails without COLMAP initialization, and (e) the low PSNR indicates that VGGT-SLAM lacks accuracy despite its ability to perform loop closure.

![](images/a0fabeb15ed46b7ee0f510f5781978c144a26f0b5586c694d10aae039094be51.jpg)  
Figure 7. We provide several sample image frames from the office loop video that were used for the teaser example.

<table><tr><td rowspan="2" colspan="2"></td><td rowspan="2">FTames</td><td colspan="4">VGGT-</td><td rowspan="2"></td><td colspan="4">DROID-</td></tr><tr><td>COLMAP (default)</td><td>SLAM [27]</td><td>ACE0 [8]</td><td>Ours (full)</td><td>COLMAP (fast)</td><td>SLAM [44]</td><td>ACE0 [8]</td><td>Ours (full)</td></tr><tr><td rowspan="9">Training</td><td>Barn</td><td>410</td><td>0.21</td><td>0.57</td><td>0.51</td><td>0.34</td><td>12.2k</td><td>0.19</td><td>0.63</td><td>0.39</td><td>0.30</td></tr><tr><td>Caterpillar</td><td>383</td><td>0.29</td><td>0.61</td><td>0.34</td><td>0.32</td><td>11.4k</td><td>0.18</td><td>0.26</td><td>0.23</td><td>0.23</td></tr><tr><td>Church</td><td>507</td><td>0.31</td><td>0.73</td><td>0.39</td><td>0.40</td><td>19.3k</td><td>0.79</td><td>0.94</td><td>0.38</td><td>0.36</td></tr><tr><td>Ignatius</td><td>264</td><td>0.21</td><td>0.54</td><td>0.25</td><td>0.25</td><td>7.8k</td><td>0.15</td><td>0.25</td><td>0.17</td><td>0.20</td></tr><tr><td>Meetingroom 371</td><td></td><td>0.33</td><td>0.71</td><td>0.48</td><td>0.42</td><td>11.1k</td><td>0.31</td><td>0.50</td><td>0.35</td><td>0.31</td></tr><tr><td>Truck</td><td>251</td><td>0.18</td><td>0.52</td><td>0.22</td><td>0.20</td><td>7.5k</td><td>0.11</td><td>0.24</td><td>0.14</td><td>0.15</td></tr><tr><td>Average</td><td>364</td><td>0.26</td><td>0.61</td><td>0.37</td><td>0.32</td><td>14.6k</td><td>0.29</td><td>0.47</td><td>0.28</td><td>0.26</td></tr><tr><td>Avg Time</td><td></td><td>1h</td><td>5min</td><td>1.1h</td><td>30min</td><td></td><td>74h</td><td>18min</td><td>2.2h</td><td>1h</td></tr><tr><td colspan="9">Family</td><td></td><td></td></tr><tr><td rowspan="9">Inuteiate</td><td></td><td>152</td><td>0.17</td><td>0.36</td><td>0.24</td><td>0. 20</td><td>4.4k</td><td>0.12</td><td>0.23</td><td>0.24</td><td>0.22</td></tr><tr><td>Francis</td><td>302</td><td>0.15</td><td>0.43</td><td>0.25</td><td></td><td>7.8k</td><td>0.24</td><td>0.17</td><td>0.25</td><td>0.23</td></tr><tr><td>Horse</td><td>151</td><td>0.21</td><td>0.49</td><td>0.24</td><td>0.25</td><td>6.0k</td><td>0.17</td><td>0.27</td><td>0.24</td><td>0.24</td></tr><tr><td>Lighthouse</td><td>309</td><td>0.36</td><td>0.67</td><td>0.38</td><td>0.32</td><td>8.3k</td><td>0.29</td><td>0.32</td><td>0.38</td><td>0.33</td></tr><tr><td>Playground</td><td>307</td><td>0.45</td><td>0.68</td><td>0.53</td><td>0.46</td><td>7.7k</td><td>0.45</td><td>0.82</td><td>0.53</td><td>0.33</td></tr><tr><td>Train</td><td>301</td><td>0.29</td><td>0.70</td><td>0.39</td><td>0.32</td><td>12.6k</td><td>0.20</td><td>0.45</td><td>0.39</td><td>0.29</td></tr><tr><td>Average</td><td>254</td><td>0.27</td><td>0.55</td><td>0.34</td><td>0.30</td><td>7.8k</td><td>0.24</td><td>0.38</td><td></td><td></td></tr><tr><td>Avg Time</td><td></td><td>32min</td><td>5min</td><td>1.3h</td><td>30min</td><td></td><td>48h</td><td>14min</td><td>0.34 2.2h</td><td>0.27 1.3h</td></tr><tr><td colspan="9"></td><td></td><td></td></tr><tr><td rowspan="9">Advanced</td><td>Auditorium</td><td>302</td><td>0.31</td><td>0.69</td><td>0.50</td><td>0.39</td><td>13.6k</td><td>0.75</td><td>0.64</td><td>0.50</td><td>0.34</td></tr><tr><td>Ballroom</td><td>324</td><td>0.43</td><td>0.68</td><td>0.40</td><td>0.37</td><td>10.8k</td><td>0.44</td><td>0.92</td><td>0.40</td><td>0.33</td></tr><tr><td>Courtroom</td><td>301</td><td>0.34</td><td>0.67</td><td>0.46</td><td>0.41</td><td>12.6k</td><td>0.61</td><td>0.79</td><td>0.46</td><td>0.34</td></tr><tr><td>Palace</td><td>509</td><td>0.56</td><td>0.67</td><td>0.83</td><td>0.67</td><td>21.9k</td><td>0.60</td><td>0.84</td><td>0.83</td><td>0.62</td></tr><tr><td>Temple</td><td>302</td><td>0.30</td><td>0.62</td><td>0.86</td><td>0.51</td><td>17.5k</td><td>0.65</td><td>0.70</td><td>0.86</td><td>0.44</td></tr><tr><td>Average</td><td>348</td><td>0.39</td><td>0.56</td><td>0.51</td><td>0.47</td><td>15.6k</td><td>0.61</td><td>0.78</td><td>0.51</td><td>0.41</td></tr><tr><td>Avg Time</td><td></td><td>1h</td><td>5min</td><td>1h</td><td>30min</td><td></td><td>71h</td><td>27min</td><td>2.8h</td><td>1.5h</td></tr></table>

Table 7. Quantitative comparisons on Tanks and Temples dataset. We report the pose accuracy via view synthesis as LPIPS (↓). Dark green indicate the best results, excluding COLMAP.

Uncertainty metrics weight sensitivity (Table 9). We evaluate the impact of varying the weighting parameters $\alpha _ { 1 }$ and $\alpha _ { 2 }$ in Eq. (9), which combine epipolar and optical flow errors into the confidence score. Our best-performing configuration is $\alpha _ { 1 } = 0 . 4$ and $\alpha _ { 2 } = 0 . 6$ , achieving 21.7 dB PSNR on the 7-Scenes dataset and 24.3 dB PSNR on the Mip-NeRF 360 dataset. Table 9 reports performance for different weight ratios. We observe that moderate variations in α (e.g., 0.3:0.7 or 0.5:0.5) have minimal impact on performance, indicating robustness to weight changes. However, extreme imbalance significantly degrades accuracy. When epipolar weighting dominates $( \mathbf { e . g . } , \alpha _ { 1 } \ge 0 . 7 ) .$ performance drops due to sensitivity to pose initialization errors and lack of local consistency. Conversely, optical flow only configurations $( \alpha _ { 2 } = 1 . 0 )$ improve robustness in indoor scenes but fail to enforce global consistency, leading to drift in large-scale or mixed environments. The best trade-off is achieved with $\alpha _ { 1 } ~ = ~ 0 . 4$ and $\alpha _ { 2 } \ = \ 0 . 6 ,$ which balances global and local geometric cues effectively across both datasets. This suggests that optical flow provides strong local evidence for texture-rich indoor scenes, while epipolar constraints remain essential for maintaining global structure in complex environments.

<table><tr><td>Supervision</td><td>ACE [7] KinextFusion</td><td>ACE [7] COLMAP</td><td>ACE0[8] self-supervised</td><td>Ours self-supervised</td></tr><tr><td>Chess</td><td>96.0%</td><td>100.0%</td><td>100.0%</td><td>100.0%</td></tr><tr><td>Fire</td><td>98.4%</td><td>99.5%</td><td>98.8%</td><td>99.0%</td></tr><tr><td>Heads</td><td>100.0%</td><td>100.0%</td><td>100.0%</td><td>100.0%</td></tr><tr><td>Office</td><td>36.9%</td><td>100.0%</td><td>99.1%</td><td>99.1%</td></tr><tr><td>Pumpkin</td><td>47.3%</td><td>100.0%</td><td>99.9%</td><td>100.0%</td></tr><tr><td>Redkitchen</td><td>47.8%</td><td>98.9%</td><td>98.1%</td><td>98.5%</td></tr><tr><td>Stairs</td><td>74.1%</td><td>85.0%</td><td>61.0%</td><td>75.1%</td></tr><tr><td>Average</td><td>71.5%</td><td>97.6%</td><td>93.8%</td><td>96.0%</td></tr></table>

Table 8. Relocalization performance comparisons on 7-Scenes dataset. We follow [8] and report percentage below 5cm, 5 error which are computed with respect to COLMAP pseudo ground truth.

Ground truth

Ours  
![](images/64031936bdddf77517ce345f8c00e3b40e1fd52b1c53dbaac78a433ce47c0a20.jpg)  
ACE-Zero w/o warm start

Figure 8. Qualitative comparison from the office loop video that were used for the teaser example. Although ACE-Zero, even when initialized with COLMAP-generated camera poses, tends to exhibit artifacts and blurriness in synthesized novel views, our proposed method consistently delivers high-fidelity renderings. This improvement reflects a more globally consistent and structurally coherent scene reconstruction.
<table><tr><td rowspan=1 colspan=1>α1:α2</td><td rowspan=1 colspan=1>7-Scenes</td><td rowspan=1 colspan=1>Mip-NeRF 360</td></tr><tr><td rowspan=2 colspan=1>0.4 : 0.6 (Best)0.5 : 0.5</td><td rowspan=1 colspan=1>21.7</td><td rowspan=1 colspan=1>24.3</td></tr><tr><td rowspan=1 colspan=1>21.5</td><td rowspan=1 colspan=1>24.1</td></tr><tr><td rowspan=1 colspan=1>0.3 : 0.7</td><td rowspan=1 colspan=1>21.6</td><td rowspan=1 colspan=1>24.2</td></tr><tr><td rowspan=2 colspan=1>0.6 : 0.40.7 : 0.3</td><td rowspan=1 colspan=1>21.3</td><td rowspan=1 colspan=1>23.8</td></tr><tr><td rowspan=1 colspan=1>21.0</td><td rowspan=1 colspan=1>23.5</td></tr><tr><td rowspan=1 colspan=1>1.0 : 0.0</td><td rowspan=1 colspan=1>20.5</td><td rowspan=1 colspan=1>22.8</td></tr><tr><td rowspan=1 colspan=1>0.0 : 1.0</td><td rowspan=1 colspan=1>21.2</td><td rowspan=1 colspan=1>23.6</td></tr></table>

Table 9. Ablations on different α ratios on 7-Scenes and Mip-NeRF 360 datasets. We report the pose accuracy via view synthesis as PSNR in dB.

Full runtime breakdown. For a 200 images frames dataset, our additional pose graph optimization step requires approximately 30 seconds per reconstruction iteration, compared to 150 seconds for vanilla ACE-Zero. Of this 30-second overhead, the actual optimization accounts for only 0.016 seconds, while the remaining time is spent on graph construction and uncertainty metric computation.

The scalability of this step depends primarily on graph construction rather than optimization. The optimization itself is highly efficient, as it operates on a sparse graph and optimizes only camera poses, resulting in near-linear complexity with respect to the number of nodes and edges. In contrast, graph construction involves pairwise consistency checks and uncertainty metric computation, which can scale quadratically with the number of images if all pairs are considered. This explains why graph construction dominates the runtime: it requires dense correspondence verification and optical flow estimation across multiple image pairs, whereas the optimization step leverages sparse factor graphs and converges rapidly. For larger datasets, pruning strategies or retrieval-based filtering will be essential to maintain scalability.

Effect of removing optical flow or epipolar metric individually (Table 10). We analyze the effect of incorporating uncertainty-aware metrics into pose graph optimization. Starting from vanilla PGO with graph construction, we progressively add epipolar and optical flow metrics. The full configuration uses both metrics. Table 10 summarizes the results for the 7-Scenes and Mip-NeRF 360 datasets. We observe that adding uncertainty-aware metrics significantly improves pose graph optimization over vanilla PGO. Epipolar constraints provide global geometric consistency, yielding a 1.0 dB improvement on both datasets. Optical flow offers dense local correspondences, which are particularly beneficial in texture-rich indoor scenes, leading to 1.6 dB improvement over vanilla PGO. Combining both metrics achieves the highest gains (2.3 dB and 2.2 dB): epipolar constraints enforce global structure, while optical flow enhances local robustness. This complementary effect high-

lights the importance of integrating multiple geometric priors for reliable pose refinement.
<table><tr><td>Configuration</td><td>7-Scenes</td><td>Mip 360</td></tr><tr><td>Vanilla PGO (with construction)</td><td>18.5</td><td>21.0</td></tr><tr><td>+ Epipolar metric</td><td>19.5</td><td>22.0</td></tr><tr><td>+ Optical flow metric</td><td>20.1</td><td>22.3</td></tr><tr><td>+ Both metrics</td><td>20.8</td><td>23.2</td></tr></table>

Table 10. Ablations on different uncertainty-aware configurations. We report the pose accuracy via view synthesis as PSNR in dB.

Full impact of spare data (Fig. 9). After including the results from the Tanks and Temples dataset, we identified an error in the previously generated figure. The figure has now been corrected, and the version in the main text should be disregarded. The updated figure presented here demonstrates that our method consistently exhibits superior robustness under limited training data conditions.

Mip-NeRF 360 and Tanks and Temples datasets (evaluate every 8th image)  
![](images/33ab680860d9050fd7f2493ebbd508cddc0c3d6a7bae4d732cc8068575ad777c.jpg)  
Figure 9. Ablations on sparse training data for both Mip-NeRF 360 and Tanks and Temples (T2) datasets. We show that our method consistently demonstrates better robustness to limited training data.

## C. Additional qualitative examples

Epipolar-based uncertainty metric (Fig. 10). To further analyze the behavior of the epipolar-based uncertainty metric, we apply synthetic perturbations to estimated camera poses. Specifically, we introduced rotational perturbations around the z-axis and translational perturbations along the z-axis to simulate varying degrees of pose deviation. For each pose, we randomly sampled 100 predicted 3D scene coordinates and reprojected them into the paired image. We then drew the corresponding epipolar lines and discarded any second-view projections that fell outside the image boundaries to ensure valid comparisons. Fig. 10 illustrates qualitative results on two datasets (7-Scenes and Tanks and Temples), where reprojected points and their

epipolar lines are color-coded by error magnitude (green = low, red = high). Interestingly, in some cases, the highest epipolar error occurs when no perturbation is applied, suggesting that the original pose estimate is inaccurate. Introducing small rotational perturbations can reduce this error, while translation perturbations occasionally lead to similar effects. These observations indicate that epipolar error is sensitive to interactions between pose components and does not consistently correlate with absolute pose accuracy. Consequently, we interpret this metric as a heuristic confidence indicator rather than a strict geometric validator.

Optical flow-based uncertainty metric. We provide additional qualitative examples in Fig. 11.

Additional qualitative examples. We provide additional qualitative examples in Fig. 12.

![](images/1ec95892572668b36295263264c6791e1e938e261e4f035fdf4c4bcca1e3fd61.jpg)  
Figure 10. Qualitative example of the epipolar-based uncertainty metric (top row: 7-Scenes dataset; bottom row: Tanks and Temples dataset). We overlay reprojected points with their corresponding epipolar lines, color-coded by error (green = low, red = high), to indicate geometric alignment quality under perturbed poses. In the top row, the highest error occurs when no perturbation is applied, suggesting the original camera pose is inaccurate; introducing rotation perturbations reduces the error. In the bottom row, from left to right, we observe a similar trend as in Fig. 2 of the main text: errors remain low without perturbation, increase with pose perturbations, and occasionall decrease when translation perturbations are added. This behavior highlights that epipolar error is sensitive to interactions between pose components and does not always correlate directly with pose accuracy. Consequently, we treat it as a heuristic confidence indicator rathe than a strict geometric validator.

![](images/2b141f786f5445ab087107d21faa57ac035d885ddb4f7776472dc500c285ba44.jpg)

![](images/9bf760225bab422a7dd77750af8dc54c49bcfe4b9ec034a1af1bc94f30d1ade3.jpg)  
Input

![](images/da2a06cb495ab6acf943009ebeb78715510d0c8d1077d17b96986425610d50ff.jpg)

![](images/37d4ffc8d6ede7637deddfdbfda9219253d80719b1917caaf5a9363c49b1721c.jpg)  
Predicted flow

![](images/f710aaf72b075a9fa005fa3486fba06478c6f7eb581338d46f833dbc4cc80e10.jpg)  
UV correspondences  
Figure 11. Qualitative examples of optical flow predictions are shown (top row: 7-Scenes dataset; bottom row: Tanks and Temples dataset). From left to right: (i) an image from the input pair, (ii) predicted flow visualization where similar colors indicate similar motion directions, and (iii) UV correspondences overlaid on the image pair, illustrating that optical flow provides robust pixel-level matches across views.

Ours  
ACE-Zero  
![](images/7a10b163300ac32a89ccca32e4cf5deeba937859c42fe02d833f8e5d653d8d49.jpg)  
Ground truth  
VGGT-SLAM

Figure 12. Qualitative comparison on Mip-NeRF 360 (top two rows) and Tanks and Temples datasets (bottom two rows) with VGGT SLAM [27] and ACE-Zero [8]. While ACE-Zero performs well on small-scale scenes such as Mip-NeRF 360, it struggles with large, complex environments like Tanks and Temples, where our method maintains high rendering quality across diverse scenes.