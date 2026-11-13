# ReFineR: Single image to dense point cloud registration at pixel level with point splatting

Ugo Insalaco<sup>1,2</sup>

ugo.insalaco@minesparis.psl.eu

Jean-Emmanuel Deschaud<sup>1</sup>

jean-emmanuel.deschaud@minesparis.psl.eu

<sup>0</sup>Pierre Dellenbach<sup>2</sup> <sup>2</sup>pdellenbach@chapsvision.com

<sup>1</sup> Mines Paris, PSL University Centre for robotics (CAOR) 75006 Paris, France

<sup>2</sup> ChapsVision 4 Rue du Port aux Vins, 92150 Suresnes, France

## Abstract

Cameras and LiDARs (Light Detection and Ranging) provide complementary observations of the same scene: cameras capture rich appearance cues while LiDAR sensors provide accurate geometric measurements. In this work, we address the problem of fine image-to-point cloud registration from a single image and a roughly estimated camera pose. On robotic platforms, residual reprojection errors of several pixels are commonly observed even after calibration due to temporal desynchronization, mechanical vibrations, or inaccuracies in point cloud reconstruction. Such errors can lead to decimeterlevel geometric discrepancies at moderate distances and significantly affect multiple downstream tasks, such as dense cross-modal supervision, texture mapping, and semantic label transfer. To address this challenge, we introduce ReFineR: an offline framework to finely register individual images to pre-reconstructed point clouds. ReFineR builds upon differentiable point splatting rasterization and a smooth Gradient Orientation Measure (GOM) to iteratively refine the camera pose through analysis-by-synthesis optimization. We evaluated the method on a curated dataset of manually calibrated images collected from heterogeneous data sources. Furthermore, we show that an automatic point cloud rescale factor can be estimated prior to optimization, enabling robust refinement across a wide range of scenarios using a single scale-invariant formulation. The project page can be found at https://refiner-project.github.io/

## Introduction

Accurate camera-LiDAR alignment is often treated as a solved pre-processing step, assuming a fixed extrinsic calibration between sensors across an entire sequence. However, in practice, even well-calibrated multi-sensor platforms exhibit residual reprojection errors that vary from frame to frame due to temporal misalignment, mechanical vibrations, rolling-shutter effects, and SLAM drift. In large-scale autonomous driving datasets such as nuScenes [6], reprojection discrepancies of 7–10 pixels are commonly observed despite official calibration parameters (details in Supplementary Section B.8). Although such errors may appear negligible, they can lead to significant metric misalignments: a 10-pixel error can induce up to 40 cm of lateral misalignment at moderate distances. This magnitude critically affects downstream applications like dense cross-modal supervision, semantic label transfer from image to LiDAR, HD map construction, and fine-grained scene understanding, where centimeter-level consistency is required. Importantly, these errors are not constant over time and cannot be corrected by a single global calibration step, making per-image fine alignment essential. Therefore, beyond coarse calibration, sub-pixel single-camera-LiDAR registration is a fundamental problem for high-precision multi-sensor perception.

![](images/1c66c9c82b033996b752bbf100e20baed0a0a75810108e4c49ddcc18f9b81c62.jpg)  
Figure 1: ReFineR optimization loop. A. During initialization, points are spatially extended to prepare the differentiable rendering step. B. Differentiable point splatting creates a view of the point cloud at the initial camera position (Section 4.2). C. Gradients are computed using a Sobel filter. D. Both modalities are compared using the GOM loss and selected comparison channels (Section 4.3 and Section 4.4). E. The camera pose is optimized by backpropagation through the differentiable image, and the new pose iteratively renders new views (Section 4.5).

Advances in performant SLAM-LiDAR algorithms [12] allow the reconstruction of dense and precise point clouds from the aggregation of multiple LiDAR scans. When used for rasterizing new images, these dense point clouds create high-quality images containing rich visual information that can be leveraged for image-to-point cloud registration. Previous work [9, 10, 19, 30, 52, 59] has shown that analysis-by-synthesis – the principle of iteratively rendering the point cloud in artificial images and comparing each render with the real image in a loop – performs significantly well for the task of fine-level registration. While earlier methods used non-differentiable algorithms for optimization [19, 52], our work takes advantage of the recent development of differentiable rendering techniques [28, 32, 56, 61] to propose a fully differentiable pipeline, from the camera pose parameters to the loss value.

This paper presents a threefold contribution. 1) As pixel-level registration is in fact absent from many LiDAR-image datasets [6, 7, 51], we propose an open dataset, created from six heterogeneous sources of point cloud and image data (ETH3D [50], CARLA [14], nuScenes [6], Tanks and Temples [29], RGBDScenes [31], and data from a Livox sensor [62]) with manually calibrated images on reconstructed dense point clouds for three of these sets. 2) We propose an automatic rescaling method, derived from first-order pinhole camera model analysis, to provide a principled solution to current scene-dependent hyperparameter tuning. 3) We introduce ReFineR for Registration at Fine level using differentiable Rendering, between single images and any pre-reconstructed point clouds. ReFineR is a training-free method that handles heterogeneous point channels at once. Prioritizing registration accuracy over speed, ReFineR was created for offline processing (HD mapping, semantic transfer...) rather than being a localization brick in real-time systems.

## 2 Related work

Knowing correspondences between 3D points and 2D pixels, closed-form solutions exist to estimate the camera’s 6 degrees of freedom [33]. For more robustness to matching noise, most common approaches use either the EPnP algorithm [33] combined with the RANSAC [15] scheme, or a residual minimization of the reprojection error using the Levenberg-Marquardt optimizer [30, 34, 40]. For the relaxed problem of camera alignment without known matches, a recent survey from An et al. [1] classifies automatic registration methods as explicit or implicit from their estimation of point-pixel pairs or not.

## 2.1 Explicit registration

In order to remove the manual task of selecting matches, some work has been focusing on automatically estimating such correspondences, either via feature matching algorithms in 2D, such as SuperGlue [30, 49] or MatchAnything [20], or with direct estimation of 2D-3D matches [25, 35, 44, 48, 54, 55]. CoFiI2P [25] trains a transformer model to generate matches between image and point features at different scales. CorrI2P [48] adds global descriptors on both modalities, given to an overlapping region detector network that produces frustum-consistent matches. I2PNet [54] estimates soft matches between points from the LiDAR scan and pixels, themselves fed to a pose regression network. Diff2I2P [44] uses a deformable correspondence tuning layer to estimate matching offsets and a diffusion loss to compare RGB images and depth maps. UniCorrn [18] embeds 2D and 3D features in a common space to predict correspondences between any modality pair, regrouping 2D-2D, 2D-3D, and 3D-3D matching into a single algorithm. A key limitation of most 2D-3D matching algorithms [18, 25, 35, 44, 48, 54, 55] is their generalization ability to various point clouds. Since the models were generally trained on individual frames (LiDAR or RGBD), they tend to show weaker results given denser point clouds as input (for instance, acquired from a dense static LiDAR scanner or aggregated scans from a mobile LiDAR). Other approaches try to directly predict pointmaps (a Width × Height × 3 tensor representing the position in space of each pixel): Brachmann et al. [4, 5, 58] trained a deep neural network with a differentiable version of RANSAC [4] for this task. SANet [58] estimated a pointmap from deep matching between a query image and the point cloud and reference views’ features. Similarly, both these methods are typically trained on specific datasets or scenes, which limits their ability to generalize across different environments or scenes. In contrast, our approach does not require any training and can be applied directly to a wide variety of image–LiDAR datasets without parameter tuning.

## 2.2 Implicit registration

A parallel line of work focuses on synthesizing views from the point cloud using coarse initialization, then comparing the rendered and real images to refine the position and orientation, with the goal of overlapping both images (a.k.a. analysis-by-synthesis). Iterations can either be carried out using neural network comparison of images [24, 39] – Calib-Net [24] concatenates image and depth map features before feeding them to a fully connected neural network, LCCNet [39] introduced a cost volume layer instead, and Ou et al. [46] added a trained diffusion model as a denoiser to the pose predicted by LCCNet – or optimization of the pose’s parameters from handcrafted loss functions between images [8, 19, 30, 37, 47, 52, 62]. Early work from Taylor et al. [52] compared a rasterized view of the point cloud to the image using Gradient Orientation Measure which minimizes the angle between each pixel’s gradient. This loss was then optimized using the non-differentiable Nelder-Mead optimizer [45]. Koide et al.’s method [30] minimized the normalized information distance, a comparison between the distributions of RGB values and synthesized LiDAR reflectivity to optimize the pose. Guislain et al. [19] refined the camera position using a mix of entropic and gradient information thanks to Mutual Information and Histogram of Oriented Gradients (HOG) features [11]. Yuan et al. [62] first extracted depth continuous edges from the point cloud and image, and minimized a line reprojection error over the Lie algebra of rigid transforms. Our approach (ReFineR) falls into the category of implicit registration.

## 2.3 Differentiable rendering

Differentiable rendering creates images whose pixel values are differentiable with respect to the 3D scene’s parameters (point positions, camera extrinsics and intrinsics, etc). These methods arose jointly with smooth 3D representation and reconstruction engines like NeRF [43] or Gaussian Splatting [28], and involve mechanisms like volumetric rendering [3, 43] or Gaussian/surfel splatting [28, 32, 56, 61]. For instance, DSS [61] projects isotropic flat Gaussians centered on each point onto the image plane, and averages for each pixel the contribution of each Gaussian. To account for the discontinuous point occlusion, they computed artificial gradients based on the relative movement between the point and the deprojected pixel. 3D Gaussian Splatting [28] took a completely differentiable path by rasterizing Gaussians onto the image plane and blending them using an opacity value.

## 2.4 Gaussian Splatting registration

Setting aside image-to-point-cloud registration, some studies have been carried out on registering new images on pre-reconstructed Gaussian Splatting models [10, 36, 41, 42, 57] using either feature matching on rendered views [10, 36], reprojection error minimization from rendered depth maps [57], or reverse rendering to find rays intersecting the image plane [42]. Matsuki et al. [41] proposed to directly integrate Lie gradients over the camera pose in the rendering process to achieve end-to-end differentiability from camera to pixel values. These methods do not directly apply to our problem as they require preliminary optimization of a Gaussian Splatting model from multiple images of the same scene (which in our single-image case is not available) and are often based on photometric error minimization [10, 41], which cannot be achieved with geometry- or reflectivity-valued LiDAR point clouds. They, however, prove that differentiable rendering can be a valuable asset for estimating fine camera positions.

## 3 Preliminaries

Reprojection error and parallax: Pixel error $p _ { e }$ (a.k.a. reprojection error) can be defined as the root-mean-square value of the displacement of pixels between two projection models $\pi _ { 1 }$ and $\pi _ { 2 }$ , given a point cloud $\left( X _ { i } \right) _ { 1 \leq i \leq N } \in \left( \mathbb { R } ^ { 3 } \right) ^ { N }$

$$
p _ { e } = { \sqrt { { \frac { 1 } { N } } \sum _ { i = 1 } ^ { N } \| \pi _ { 1 } ( X _ { i } ) - \pi _ { 2 } ( X _ { i } ) \| ^ { 2 } } }\tag{1}
$$

In our problem, pixel error is interesting because it defines the registration difficulty independently of the latent point cloud scale (which would not be the case when comparing distances using the translation error). For a small perturbation $\delta \tau \in \mathfrak { s e } ( 3 )$ of the camera extrinsics around 0, we obtain:

$$
\pi ( X _ { i } ) - \pi _ { 0 } ( X _ { i } ) = J ( X _ { i } ) \delta \tau + o ( \lVert \delta \tau \rVert )\tag{2}
$$

with $\begin{array} { r } { J = \frac { D \pi } { D \tau } } \end{array}$ (borrowing notation from [13]) computed explicitly in supplementary. In particular, the terms in J associated with rotation only depend on point projections $\left( x _ { c } , y _ { c } \right) =$ $\left( f _ { x } X _ { i X } / X _ { i Z } , f _ { y } X _ { i Y } / X _ { i Z } \right)$ on the image, with $f _ { x } , f _ { y }$ the camera intrinsics. In contrast, the ones for translation also depend on their depth $X _ { i Z }$ : the farther a point is, the less impact a translation will have on its reprojection error. This is the well-known phenomenon of parallax.

## 4 Method

We assume that a coarse initialization of the camera pose is available prior to optimization. Such an initialization may originate from a standard camera-LiDAR calibration performed on the robotic platform, combined with LiDAR-Odometry [6] or from Structurefrom-Motion point clouds registered on a LiDAR point cloud [29]. Importantly, estimating this coarse pose differs fundamentally from the fine alignment addressed in this work, as it is typically solved through geometric calibration or feature matching rather than analysis-bysynthesis optimization.

The ReFineR algorithm is structured into four main stages (see Fig. 1). 1) The input point cloud scale is normalized following Section 4.1, allowing our method to be scale independent. 2) A rendered view of the point cloud is created with differentiable point splatting (based on Gaussian Splatting’s [28] framework), as described in Section 4.2. 3) Handcrafted features, based on gradient orientation, are computed on the synthetic and real images and compared according to the rendered channels (Section 4.3 and Section 4.4). 4) Backpropagation is carried out with computation of Lie gradients with respect to the camera parameter τ (Section 4.5), and the process continues from step 2. This pipeline is entirely differentiable.

## 4.1 Point cloud rescale estimation

When handling heterogeneous point clouds (e.g., indoor, outdoor, or RGB-D data) and accounting for parallax effects, optimizing the camera pose using gradient-based algorithms requires prior knowledge of the point cloud’s scale in order to appropriately adjust the learning rate. In this section, we propose an expression for automatically estimating such a prior, allowing fixed learning rates for rotation and translation across all scenes.

Following Eq. (2), we can approximate the pixel error using:

$$
p _ { e } ^ { 2 } = \delta \tau ^ { T } \underbrace { \left( \frac { 1 } { N } \sum _ { i = 1 } ^ { N } J ( X _ { i } ) ^ { T } J ( X _ { i } ) \right) } _ { S } \delta \tau + o ( \Vert \delta \tau \Vert ^ { 2 } )\tag{3}
$$

Intuitively, the symmetric positive matrix S integrates the contributions and correlations to the pixel error of each transform parameter in τ. Separating contributions from rotation and translation, and introducing the unknown scale coefficient α such that $X _ { i } = \alpha X _ { i } ^ { \prime }$ , this matrix can be rewritten as (details in Supplementary Section A.3):

$$
S = \binom { S _ { r } } { S _ { r t } ^ { T } } \quad S _ { r t } \bigg ) = \binom { S _ { r } } { \alpha } \quad \frac { 1 } { \alpha } S _ { r t } ^ { \prime } \bigg )\tag{4}
$$

To select an appropriate $\alpha ,$ our criterion is to balance the effects of translation and rotation on the pixel error. Concretely, we would expect a translation of 1 (written without a unit on purpose), to have approximately the same effect on the pixel error as a rotation of 1. To do so, we choose to equalize the Hilbert-Schmidt norms between $S _ { r }$ and $S _ { t } ^ { \prime } .$ . We obtain:

$$
\alpha = \left( \sum _ { i = 1 } ^ { 3 } \lambda _ { r , i } ^ { 2 } / \sum _ { i = 1 } ^ { 3 } \lambda _ { t , i } ^ { 2 } \right) ^ { 1 / 4 }\tag{5}
$$

with $( \lambda _ { r , i } , \lambda _ { t , i } ) _ { i }$ the respective eigenvalues of $S _ { r }$ and $S _ { t }$ . An example of rescaled scenes is shown in Supplementary Fig. 2. Generally speaking, this rescaling procedure is a form of reparameterization to obtain better variable conditioning, which is a major issue in numerical optimization [53].

## 4.2 Synthesis

After rescaling the point cloud, we initialize a Splatting model by creating an isotropic Gaussian for each 3D point (with standard deviation proportional to the distance to each point’s nearest neighbor by a factor $p _ { s } )$ , and setting a fixed opacity $^ { o . }$ To limit the visual impact of isolated points (which would therefore be initialized with a large size), we limit the size of each Gaussian such that its projection does not exceed 4 pixels in diameter. We add each point channel value as a Gaussian feature (normals calculated from the point cloud, LiDAR reflectivity, etc.).

## 4.3 Analysis

Edge-based cues have been explored in prior work for cross-modal alignment [19, 52]. Building on the observation that image gradients correlate with both point cloud geometry and object texture, we follow Taylor et al.’s work [52] and adapt the Gradient Orientation Measure (GOM). From two images $\left( I _ { p } \right) _ { p } , \left( I _ { p } ^ { \prime } \right) _ { p }$ indexed by their pixels and their corresponding gradients $\left( G _ { p } \right) _ { p }$ and $\left( G _ { p } ^ { \prime } \right) _ { p }$ , computed using (differentiable) Sobel filters, the GOM is defined as

$$
G O M ( I , I ^ { \prime } ) = \frac { \sum _ { p } \vert \left( G _ { p } \vert G _ { p } ^ { \prime } \right) \vert } { \sum _ { p } \Vert G _ { p } \Vert _ { 2 } \Vert G _ { p } ^ { \prime } \Vert _ { 2 } }\tag{6}
$$

GOM aims at maximizing the dot product between gradients, therefore aligning edges from two images.

## 4.4 Comparison

Comparison between modalities is achieved by creating correspondences between point clouds and image channels, and computing the GOM loss function for each correspondence.

For the point cloud, the different usable channels are: average normal direction $( p ^ { n } )$ , individual normal coordinates $( p ^ { n x } , p ^ { n y } , p ^ { n z } )$ , normalized projected depth $( p ^ { d } )$ , LiDAR reflectivity $( p ^ { r } )$ , or grayscale value per point $( p ^ { g } )$ acquired from an embedded camera in the LiDAR sensor (for example the fixed terrestrial FARO scanner). For images, we compute their grayscale values $( i ^ { g } )$ from the RGB channels. In our ablation study, we experiment with adding generated channels for images using the Marigold diffusion model [27], adding the artificial normal channels $i ^ { n x } , i ^ { n y } , i ^ { n z }$ and normalized depth $i ^ { d }$ (see Section C.4).

The final multimodal loss function between the multichannel rendered image $( I ^ { c } ) _ { c \in { \mathcal { C } } _ { 3 \mathrm { D } } }$ and real image $( I ^ { \prime c } ) _ { c \in { \mathcal { C } } _ { 2 \mathrm { D } } }$ is then computed as:

$$
\mathcal { L } ( I , I ^ { \prime } ) = - \frac { 1 } { \vert \mathcal { C } _ {  } \vert } \sum _ { ( c , c ^ { \prime } ) \in \mathcal { C } _ {  } } G O M ( I ^ { c } , I ^ { \prime c ^ { \prime } } )\tag{7}
$$

With $\mathcal { C } _ { 3 \mathrm { D } }$ and $\mathcal { C } _ { \mathrm { 2 D } }$ denoting the sets of available point cloud and image channels and $\mathcal { C } _ {  } \subset \mathcal { C } _ { 3 D } \times \mathcal { C } _ { 2 D }$ a set of correspondences between 3D and 2D channels. In our experiments, we consider two common setups: the first, where the only available information is the point cloud geometry, and the second, with LiDAR reflectivity or grayscale value. For the first scenario, we therefore use ReFineR with $\mathcal { C } _ {  } = \{ ( p ^ { n x } , i ^ { g } ) , ( p ^ { n y } , i ^ { g } ) , ( p ^ { n z } , i ^ { g } ) \}$ and for the second, we use $\mathcal { C } _ {  } = \{ ( p ^ { r } , i ^ { g } ) \}$ or $\mathcal { C } _ {  } = \{ ( p ^ { g } , i ^ { g } ) \}$

## 4.5 Optimization

Our pipeline is entirely differentiable from the camera parameter $\tau \in \mathfrak { s e } ( 3 )$ to the loss value $\mathcal { L } ( I , I ^ { \prime } )$ and uses Gaussian Splatting SLAM’s [41] codebase as the differentiable point splatting rasterizer. Optimization is carried out using the Lie version of gradient descent, with a fixed learning rate. Other optimization algorithms could be possible, as previous work experimented with integrating the Levenberg-Marquardt optimizer [23, 34] through the Gaussian Splatting rasterizer, but their method additionally requires a large amount of GPU memory (∼50GB), which would reduce the applicability of our method. The number of iterations is bounded between 50 and 1000, and a minimum step magnitude $s _ { m }$ is fixed to dynamically stop optimization upon convergence.

## 5 Experiments

## 5.1 Setup

Data: For the task of image-to-point cloud registration with a pixel-level goal, there is, to our knowledge, no unified evaluation framework. Generally, state-of-the-art methods either choose to run their experiments on their own data [19, 30, 62], only on RGB-D images [35, 55] or on odometry-based calibration like KITTI Odometry or nuScenes [16, 25, 39, 48] which do not have pixel-precision calibration (see Table 6 from Supplementary Section B.8) with registration on individual frames rather than the fully reconstructed point clouds. As part of our open contributions, we curated a dataset of point clouds and finely registered color images from six different sources: ETH3D [50], Tanks and Temples [29], nuScenes [6], Livox [62], RGBDScenes [31], and synthetic data from the CARLA simulator [14]. ETH3D is a multiview stereo benchmark with high-resolution images and highquality point clouds acquired with a FARO scanner, used for evaluation. The ETH3D dataset provides subpixel camera registration, thanks to photometric error minimization across multiple views at once. This ideal scenario (RGB point cloud, high-precision scanner, multiple views) does not, however, reflect the full diversity of point clouds, and we therefore choose to enrich it from other sources. Tanks and Temples is closely related to ETH3D, as each point cloud was acquired using a similar FARO scanner, but we show (see Supplementary Section B) that this dataset has coarser image registration of the order of 5 to 10 pixels on average. Indeed, the ground truth camera poses were obtained through manual registration of an SfM model to the point cloud, and errors originating from the sparse reconstruction directly impact the final camera poses. To improve on this error, we estimated the camera parameters for 20 images of this dataset by selecting points and pixel matches and minimizing the reprojection error with Gauss-Newton optimization. We obtained reprojection errors of 2px on average after this refinement. More information on our manual registration procedure is given in Section B. nuScenes data are outdoor sequences obtained from a Velodyne HDL32E LiDAR mounted on a car, alongside cameras, radars, and an IMU. We obtained dense point clouds by running CT-ICP [12], a LiDAR odometry method that provides a precise pose for each LiDAR scan. After reconstruction and using the official sensor platform calibration, we still noticed residual errors and chose to manually register 20 images from this dataset, improving the registration from 7–10px to 3–5px. Livox data presented in Yuan et al.’s paper [62] consist of four scans obtained from a Livox sensor and a single image per scan. Each image was similarly manually registered. RGBDScenes consists of color and depth images recorded from a Kinect sensor, where point clouds are created by aligning frames using Patch Volumes Mapping [21]. CARLA is an autonomous driving simulator that benefits from perfect registration, while still producing realistic point clouds and images. For the rest ofthis paper, exceptfor the RGBDScenes dataset, we will consider images with a resized height of1024px. For RGBDScenes, height is kept at 480px. Every pixel error will be expressed accordingly. In total, this dataset is composed of 50 images from ETH3D, 50 images from CARLA, 20 images from Tanks and Temples, 20 images from nuScenes, 20 images from RGBDScenes, and 4 images from Livox, for a total of 164 finely registered images across 29 heterogeneous point cloud scenes.

Methods: ReFineR is evaluated against Koide et al.’s method [30] and the intensityfree method from Yuan et al. [62]. It is known that synthesized views of a point cloud can be used to find meaningful correspondences with real images in 2D [22, 50]. We thus compare ReFineR with the MatchAnything model [20], which extracts 2D matches between heterogeneous modality images. After matching in 2D using MatchAnything, we retrieve the 3D coordinates of non-empty pixels and estimate the pose using PoseLib’s RANSAC. MatchAnything’s default model configuration was kept. We added GS-SLAM, which uses photometric loss for comparison with the base work from Matsuki et al. [41]. The first two methods (Koide et al. [30] and Yuan et al. [62]) require hyperparameter tuning based on the scene type and scale, which we detail in Supplementary Section C.1. Conversely, ReFineR is a scale-invariant approach, for which we fix $s _ { m } = 2 \times 1 0 ^ { - 5 }$ , generally stopping our registration loop after 100 to 200 iterations. Gradient descent is handled by the Adam optimizer in PyTorch with learning rate $l _ { r } = 2 \times 1 0 ^ { - 4 }$ . The splatting model is initialized using $o = 0 . 2$ and $p _ { s } = 0 . 5$ . All our experiments ran on a GeForce RTX 5090 graphics card with 32GB VRAM, but for most sets, 8GB VRAM would suffice. Indeed, ReFineR’s computational cost is directly related to the number of points to be registered to, which in general doesn’t exceed 10M (including frustum culling). We experimented with pretrained deep learningbased approaches [18, 24, 55, 63], but noticed that they tend to predict abnormal 2D-3D correspondences for our task of fine-level camera calibration to dense point clouds. These methods were pretrained on individual nuScenes scans or RGBD depth maps, which we suspect introduced a distribution gap with inference on dense depth maps. A more detailed reflection is shared in Section C.9.

Randomized Pose Initialization: Since point clouds are acquired at different scales and to keep consistency with our scale-invariant approach, we perturb our camera poses such that the resulting average pixel error has a fixed value. This can be done using a first-order approximation of the pixel error (see Eq. (2)) and selecting a random transform on the ellipsoid of constant error. Computation details are given in Supplementary Section A.4. For the sets that don’t have manual matches between points and pixels (e.g., ETH3D, CARLA, RGBD-Scenes), we sample 100 random points from each image’s depth map. This initialization technique has the advantage of avoiding a fixed translation perturbation, which might have a varying effect on the reprojection error due to parallax. We chose an initial perturbation of 10 px as it was, on average, our observed residual reprojection errors for the nuScenes and Tanks and Temples datasets. In practice, an error of about 10 pixels can induce translation discrepancies of several tens of centimeters in outdoor scenes (e.g. CARLA). Each camera pose was perturbed from random starting positions twice for the CARLA and ETH3D sets, and 5 times for the others, adding up to a common 100 registrations per set (except Livox, which has 20).

Metrics: We report results in terms of percentage of registrations below a given threshold $t _ { h }$ in pixels, denoted pix $ @ t _ { h }$ with pix $\begin{array} { r } { @ t _ { h } = \frac { 1 } { N } \sum _ { i } \mathbb { 1 } ( p _ { i } < t _ { h } ) } \end{array}$ , N the number of evaluated pairs, and $p _ { i }$ the resulting pixel error of sample i. We also give median pixel errors across all registered scenes per set. Because translation error is scale-dependent, we only report it along with rotation error in supplementary Section C.2. Both metrics are respectively defined for two rigid transform matrices $\begin{array}{c} T = { \binom { R } { 0 } } \qquad { 1 } \end{array}$ and $\begin{array} { r } { T ^ { \prime } = \binom { R ^ { \prime } } { 0 } \qquad 1 } \end{array}$ as $\mathrm { t r a n s l } = \| t ^ { \prime } - t \|$ and $\mathrm { r o t } = \operatorname { a r c c o s } ( ( \operatorname { t r a c e } ( R ^ { \prime T } R ) - 1 ) / 2 )$

## 5.2 Registration with raw point cloud

In this first setup, we assume that no other information than the point cloud geometry is available. Indeed, this is a rather common setup, as for low-cost LiDARs, the point intensity is strongly correlated with the laser range or incidence angle, and not the underlying material, which makes this channel unusable when fusing multiple scans in a dense point cloud [26]. From the raw point cloud, we compute normal directions using 50-nearest-neighbor PCA. Since the benchmarking methods [20, 30, 41, 62] only take as input mono-intensity images, we synthesize the mean normal value $p ^ { n }$ and compare it with the image grayscale intensity $i ^ { g }$ . For our method, which is capable of handling multiple channels at once, we use each normal’s coordinate individually, compared with the image’s grayscale value $i _ { g }$ :

$$
{ \mathcal C } _ {  } = \{ ( p ^ { n x } , i ^ { g } ) , ( p ^ { n y } , i ^ { g } ) , ( p ^ { n z } , i ^ { g } ) \} .
$$

Results from Table 1 (one stage) show that for the more geometrically accurate datasets, ReFineR greatly improves on MatchAnything’s baseline (41% subpixel registration on ETH3D versus 8%, 44% below 2px on CARLA against 14% for MatchAnything) but lacks robustness for noisier datasets (Livox and RGBDScenes) and underperforms in the 5–10px ranges in which MatchAnything excels. It is unsurprising that this experimental setup is a failing case for GS-SLAM and Koide et al. as both these methods directly compare point intensities with the image using photometric loss and Mutual Information, but here normal orientation values are very weakly correlated with real image color. This explains the large error values for GS-SLAM, since it completely diverges in most cases. Yuan et al.’s method does not seem robust for this fine registration task for most datasets. Their method extracts edges from the point cloud and matches them with 2D edges after projection. This procedure may be efficient for the specific Livox scans, which were taken at fixed positions and therefore do not contain occluded edges. The occluded edges are not filtered by the algorithm and could create unwanted supervision and wrong alignment. Additionally, 3D edges might be more difficult to extract in the first place, using plane intersection for sparser point clouds, as in CARLA and nuScenes. Taking advantage of MatchAnything’s and ReFineR’s strengths, we combine the two algorithms into our complete registration pipeline (two stages). A multistage approach was also considered by Koide et al. using SuperGlue [49]. To do so, we propose to use our point splatting rendering of the point cloud as input to MatchAnything instead of point rasterization in order to create a denser image (referred to as MatchAnything\*). This choice is closely related to the reconstruction procedure of ETH3D, which fills the empty pixels using nearest neighbors before extracting SIFT [38] features, and we show in the ablation that our design slightly increases the reliability of MatchAnything alone. This combination appears fruitful as performance is increased under all registration thresholds (58% under 1px on ETH3D, 68% under 2px on CARLA). This also proves that ReFineR is particularly performant as a "refinement" tool capable of improving registration even at a few pixels of displacement. In comparison, two successive inferences of MatchAnything are not able to reach as fine results. As for computation times, MatchAnything is generally constant, under 2s per image, while the other methods vary between 10s and 40s because of the variable number of iterations and point cloud size. On CARLA, ReFineR and Koide et al.’s methods took about 10s, while Yuan et al.’s took 45s per image. We provide more insight on computation times in supplementary Section C.6.

<table><tr><td rowspan="2"></td><td colspan="3">ETH3D</td><td colspan="3">Tanks and Temples</td><td colspan="3">CARLA</td><td colspan="3">Livox</td><td colspan="3">nuScenes</td><td colspan="3">RGBDScenes</td></tr><tr><td>pix@ (%↑)</td><td>5</td><td>pe(px↓)</td><td>2</td><td>pix@ (%↑) 5</td><td></td><td>Pe(px↓) 1</td><td>pix@ (%↑) 2 5</td><td></td><td>Pe(px↓)</td><td>pix@ (%↑) 10</td><td>Pe(px↓)</td><td>3</td><td>pix@ (%↑) 10</td><td>Pe(px↓)</td><td>pix@ (%↑) 3</td><td>5 10</td><td>Pe(px↓)</td></tr><tr><td>One stage</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GS-SLAM []</td><td>0</td><td>0</td><td>0 123.23±64.59</td><td>0</td><td>0</td><td>264.12±181.23</td><td>0</td><td>0</td><td></td><td>371.52±214.75 0</td><td>0</td><td>113.85±34.13</td><td>0</td><td>0</td><td>309.77±110.45</td><td>0 0</td><td>0</td><td>90.32±32.97</td></tr><tr><td>Yuan et al. 口</td><td>0</td><td>2 20</td><td>10.29±9.74</td><td>1</td><td>17 52</td><td></td><td>4.39±7.71 0</td><td>2</td><td>4</td><td>15.20±41.70 25</td><td>100</td><td>5.83±1.21</td><td>0</td><td>5</td><td>21.10±37.53</td><td>0 0</td><td>5</td><td>26.95±26.01</td></tr><tr><td>Koide et al. []</td><td>0</td><td>25</td><td>11.10±30.86</td><td>0</td><td>0</td><td>4</td><td>29.00±67.16 2</td><td>8</td><td>28</td><td>8.82±10.73 25</td><td>75</td><td>6.61±2.60</td><td>0000 0</td><td>40</td><td>12.03±12.21</td><td>0 0</td><td>5</td><td>36.19±25.14</td></tr><tr><td>Taylor et al. 口</td><td>0</td><td>27</td><td>7.13±3.00</td><td>2</td><td>5 18</td><td></td><td>7.79±2.47 0</td><td>0</td><td>10</td><td>8.87±3.75 0</td><td>75</td><td>8.35±1.00</td><td>0</td><td>25</td><td>14.61±26.77</td><td>0 10</td><td>25</td><td>12.77±7.73</td></tr><tr><td>MatchAnything []</td><td>8 65</td><td>100</td><td>1.80±0.83</td><td>1</td><td>46</td><td>95</td><td>3.05±0.94 0</td><td>14</td><td>65</td><td>4.12±3.88 65</td><td>100</td><td>4.44±0.86</td><td>0 28</td><td>88</td><td>5.71±2.56</td><td>0 28</td><td>95</td><td>6.07±1.90</td></tr><tr><td>ReFineR</td><td>41 73</td><td>79</td><td>1.10±3.17</td><td>64</td><td>88 94</td><td></td><td>1.86±1.87 12</td><td>44 74</td><td></td><td>2.36±3.45 30</td><td>90</td><td>5.50±2.13</td><td>9 41</td><td>87</td><td>6.29±3.08</td><td>8</td><td>72</td><td>8.96±2.08</td></tr><tr><td>Two stages</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>MatchAnything + GS-SLAM []</td><td>0 0</td><td>0</td><td>122.55±64.26</td><td>0</td><td>0</td><td>219.32±204.55</td><td>0</td><td>0</td><td></td><td>356.62±492.77 0</td><td>0</td><td>117.12±33.88</td><td>0 0</td><td>0</td><td>320.79±110.81</td><td>0 0</td><td>0</td><td>86.55±43.01</td></tr><tr><td>MatchAnything + Yuan et al. 口</td><td>0 1</td><td>30 45</td><td>8.99±48.80</td><td>0</td><td>11 55</td><td>7</td><td>4.56±7.63 0</td><td>2 13</td><td></td><td>12.32±17.25 50</td><td>100</td><td>4.94±1.30</td><td>0 2</td><td>14</td><td>21.78±435</td><td>0 0</td><td>5</td><td>32.08±20.41</td></tr><tr><td>MatchAnything + Koide et al. []</td><td>1 14</td><td></td><td>5.90±17.58</td><td>0</td><td>1 65</td><td></td><td>22.25±60.57 0</td><td>5 29</td><td></td><td>7.70±22.69 0</td><td>85</td><td>7.16±1.92</td><td>0 13</td><td>54</td><td>8.38±46.37</td><td>0 0</td><td>12</td><td>37.68±26.45</td></tr><tr><td>MatchAnything + Taylor et al. []</td><td>21 74 70</td><td>96 99</td><td>1.48±1.85 1.58±0.85</td><td>12 8</td><td>98 99</td><td></td><td>2.72±0.81 0 2.63±0.77 3</td><td>9 51 24 66</td><td></td><td>4.97±5.77 25 3.38±4.31 50</td><td>100 100</td><td>5.69±1.17 5.13±0.%</td><td>0 31 0 32</td><td>80</td><td>6.62±42.65</td><td>1 27 9 37</td><td>63</td><td>7.31±6.01</td></tr><tr><td>MatchAnything + MatchAnything []</td><td>14 96</td><td>100</td><td>0.90±0.40</td><td>70</td><td>71 100</td><td></td><td>16</td><td></td><td></td><td>55</td><td>100</td><td>4.94±0.94</td><td>54</td><td>98</td><td>5.70±1.93 4.90±2.75</td><td>43</td><td>91</td><td>5.58±2.51</td></tr><tr><td>MatchAnything* + ReFineR</td><td>58</td><td></td><td></td><td></td><td>95</td><td></td><td>1.76±0.51</td><td>68</td><td>91</td><td>1.54±3.04</td><td></td><td></td><td>6</td><td>97</td><td></td><td>13</td><td>92</td><td>5.36±4.25</td></tr></table>

Table 1: Fine image to raw point cloud registration results (Section 5.2). Each image was registered using rendered point cloud normals.

## 5.3 Registration with intensity point clouds

In this second setup, we consider the point cloud intensity when available. This intensity may either be a reflectivity value returned by each emitted light ray, which is related to the measured range or material absorption (Livox, nuScenes), or it may be, in some cases, a color value. Color is specific to the Faro Focus scanner of ETH3D and Tanks and Temples, and the RGBD camera of RGBDScenes, as color comes from an embedded, pre-calibrated camera.

<table><tr><td rowspan="3"></td><td colspan="4">ETH3D</td><td colspan="3">Tanks and Temples</td><td colspan="3">Livox</td><td colspan="3">nuScenes</td><td colspan="3">RGBDScenes</td></tr><tr><td>pix@ (%↑)</td><td></td><td>5</td><td> $p _ { e } ( \mathrm { p x } \downarrow )$ </td><td> $\operatorname { p i x } ( \underline { { \omega } } \ ( \mathcal { V } \delta \uparrow )$  2 3</td><td></td><td> $p _ { e } ( \mathrm { p x } \downarrow )$ </td><td>pix@ (%↑)</td><td></td><td> $p _ { e } ( \mathfrak { p x } \downarrow )$ </td><td> $\operatorname { p i x } ( \underline { { \varpi } } ) \ ( \mathcal { p } _ { o } \uparrow )$ </td><td></td><td> $p _ { e } ( \mathbf { p x } \downarrow )$ </td><td> $\operatorname { p i x } ( \underline { { \alpha } } ) \ ( \mathcal { R } \gamma )$ </td><td></td><td> $p _ { e } ( \mathrm { p x } \downarrow )$ </td></tr><tr><td>One stage</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>10</td><td></td><td></td><td></td><td></td><td></td><td></td><td>10</td><td></td></tr><tr><td>GS-SLAM []</td><td>19</td><td>40</td><td>47</td><td> $7 . 5 4 \div 3 9 . 0 0$ </td><td>45 62</td><td>67</td><td>2.06±59.53</td><td>20</td><td>25</td><td> $5 6 . 4 1 { \scriptstyle \pm 3 2 . 7 9 }$ </td><td>0</td><td>8</td><td> $1 3 1 . 0 7 \substack { \pm 9 4 . 6 8 }$ </td><td>1</td><td>5</td><td>11</td><td>37.54±29.24</td></tr><tr><td>Koide et al. []</td><td>57</td><td>98</td><td>100</td><td>0.93±0.29</td><td>51 95</td><td>100</td><td>1.99±0.50</td><td></td><td>100</td><td>4.66±0.28</td><td>0</td><td>40</td><td></td><td>11.14±743 25</td><td>45</td><td>85</td><td>5.32±22.24</td></tr><tr><td>Taylor et al. []</td><td>1</td><td>1</td><td>29 6.76±5.05</td><td></td><td>7</td><td>23</td><td>7.25±3.48</td><td>75 0</td><td>75</td><td>9.09±6.69</td><td>10 0</td><td>15</td><td>22.44±46.66</td><td>0</td><td>5</td><td>55</td><td>9.07±6.65</td></tr><tr><td>MatchAnything []</td><td>35</td><td>87</td><td>100 1.12±0.61</td><td></td><td>3 38 87</td><td>100</td><td>2.13±0.63</td><td>75</td><td>100</td><td>4.17±0.92</td><td>4 64</td><td>100</td><td></td><td>4.72±1.19 81</td><td>100</td><td>100</td><td>1.81±0.93</td></tr><tr><td>ReFineR</td><td>61</td><td>70</td><td>73 0.71±3.62</td><td>71</td><td>91</td><td>98</td><td>1.93±0.94</td><td>45</td><td>95</td><td>5.17±1.68</td><td>6 35</td><td>84</td><td>6.99±2.88</td><td>12</td><td>22</td><td>81</td><td>8.07±3.00</td></tr><tr><td>Two stages</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>MatchAnything + GS-SLAM []</td><td>32</td><td>66</td><td>73</td><td>1.30±38.55</td><td>36 74</td><td>83</td><td>2.12±38.45</td><td>25</td><td>25</td><td>65.64±30.08</td><td>0 11</td><td>17</td><td>139.64±101.83</td><td>5</td><td>5</td><td>10</td><td>35.67±31.37</td></tr><tr><td>MatchAnything + Koide et al. []</td><td>57</td><td>100</td><td>100 0.95±0.25</td><td></td><td>48 96</td><td>100</td><td>2.02±0.48</td><td>50</td><td>100</td><td>4.99±0.71</td><td>0 18</td><td>68</td><td>7.99±10.92</td><td>38</td><td>60</td><td>85</td><td>3.96±27.30</td></tr><tr><td>MatchAnything + Taylor et al. []</td><td>68</td><td>89</td><td>98 0.81±1.45</td><td>36</td><td>91</td><td>97</td><td>2.23±0.95</td><td>70</td><td>100</td><td>4.48±0.63</td><td>4 45</td><td>80</td><td>5.35±26.49</td><td>50</td><td>78</td><td>93</td><td>2.99±3.53</td></tr><tr><td>MatchAnything + MatchAnything []</td><td>62</td><td>96 100</td><td>0.90±0.43</td><td>52</td><td>96</td><td>99</td><td>1.98±0.64</td><td>75</td><td>100</td><td>4.12±0.57</td><td>2 54</td><td>100</td><td>4.83±1.24</td><td>87</td><td>100</td><td>100</td><td>2.00±0.64</td></tr><tr><td>MatchAnything* + ReFineR</td><td>86</td><td>100</td><td>100</td><td>0.57±0.29</td><td>74 95</td><td>100</td><td>1.90±0.55</td><td>75</td><td>100</td><td>4.17±0.63</td><td>10</td><td>62 100</td><td></td><td>4.29±1.40 99</td><td>100</td><td>100</td><td>1.39±0.54</td></tr></table>

Table 2: Fine image to intensity point cloud registration results (Section 5.3). Each image was registered using rendered point cloud reflectivity or grayscale value.

We observe that intensity provides, in general, richer information for registration. This is naturally the case for the color channel, but also for reflectivity, as drawn patterns (white arrows on roads...) will now appear on reflectivity images, thanks to the uneven absorption strength of colored surfaces. Examples of rendered images are shown in supplementary Fig. 4. Thanks to the color information, photometric losses become a relevant supervision channel, which allows GS-SLAM to perform well on ETH3D and Tanks and Temples and to obtain competitive results on RGBDScenes, but it fails on the remaining reflectivity sets. Koide et al.’s method is a little more resilient as it compares intensity histograms to extract correlation information between image color and LiDAR reflectivity. Our 2-stage variant with the MatchAnything\* + ReFineR combination obtained near 100% subpixel registration for ETH3D and best results at lower thresholds for all sets. We show qualitative results of both experiments in supplementary Fig. 7 and Fig. 8.

## 5.4 Ablation studies

Our design choices are evaluated in Table 3 on all datasets with raw point clouds. As before, each experiment is initialized at a 10px mean error. Experiments (I-IV) incorporate the MatchAnything\* first stage with variations of ReFineR. Experiment (V) shows the performance gain from MatchAnything\* alone. Experiment (VI) compares two stages of MatchAnything and MatchAnything\*.

Automatic rescaling (I): First, we test the impact of not using our automatic point cloud rescaling factor, keeping the point cloud at its acquired scale. In practice, this implies that since the learning rate remains fixed, translation is less updated at each iteration, and the optimization is more prone to fall into local translation minima. This is why we observe a more significant impact on the CARLA set, as this set has the largest point clouds, being simulated outdoors and on large streets and with large depth ranges. Practically, adding an automatic rescale factor also serves the purpose of fixing the hyperparameters independently of the point cloud scale (learning rate, point size).

Splatting parameters (II) & (III): We change the values of the opacity and point size parameters used for splatting. These parameters are directly linked to the loss landscape of the optimization, as a larger point size creates larger splats, which can reduce the fine registration ability of the method by having too smooth images, but a smaller point size creates rendering holes, which can bias the gradient computations. We found that using a size 0.5 gave reasonable results. Similarly, opacity controls a form of smoothness in the optimization, as low opacity increases the individual contribution of each Gaussian, making the loss more robust, but also enables occluded structures to appear in the render. We chose to fix the opacity at 0.2.

![](images/d44f39dcece62872204007156f2b0ea8787c0e47368b30973451d782fdb7a25f.jpg)  
Table 3: Ablation study (Section 5.4).

Loss functions (IV): We experimented with the idea of using intermediate MatchAnything features as a replacement for the GOM loss. The intuition was to leverage the model’s cross-modality features at a dense and multiscale level. We compared features at scales 2, 4, and 8, computed for the initial RGB and rendered images, using an L2 loss. Due to memory constraints, rendered images at 1024px were differentiably resized to 664px before giving them as input to MatchAnything, which we suspect had an impact on the comparatively weak results for this experiment. Additionally, we guess that the presence of pooling layers in the CNN encoder reduces the fine-registration abilities, as these layers divide the image resolution by 2, 4, and 8. We believe that a less naive approach could certainly be proposed to better leverage the MatchAnything features, which may be investigated in future work. Inspired by Guislain et al.’s work [19], we propose to implement a differentiable version of the HOG features [11]. To compute the HOG features, one should first estimate the image gradients, then apply a smooth histogram convolution kernel onto the image, creating cell histograms, followed by a local block normalization of each histogram bin along blocks of cells (an explicit formulation of differentiable HOGs can be found in Supplementary Section A.5). The idea behind this loss is quite similar to the GOM, which locally compares images’ gradients. This is probably why we observe, on average, comparable results between GOM and HOG features. On average, GOM loss is a little faster than HOG, as the former took on average 0.05s per iteration, whereas the latter took 0.2s.

MatchAnything (V) & (VI): Comparing performance between MatchAnything inference on image rasters and Gaussian splatting renders, we observe a net gain in the final pixel error for the CARLA and Tanks and Temples datasets, hinting that a denser image is a better quality input for MatchAnything. A very small improvement can be observed on the ETH3D dataset, which is coherent, as this dataset already has very dense point clouds that completely fill the image, even with simple rasterization. When comparing two consecutive inferences of MatchAnything and MatchAnything\*, this gap closes, demonstrating the limits of the

MatchAnything model itself. The similar performances when adding ReFineR as a second stage tend to show that our method is capable of handling some of MatchAnything’s weaker cases, but given results from (VI), we chose to keep MatchAnything\* for our final pipeline.

## 5.5 Registration at scale

To capture the performance of the MatchAnything\* + ReFineR combination, we vary the initialization of our registration experiment from 0 to 100 pixels (Table 4). Starting from a 0px perturbation, we observe that ReFineR alone keeps the registration stable under 2px 76% of the time, which is about three times as often as MatchAnything alone, but when combined, the fine registration capabilities of ReFineR mitigate MatchAnything’s instabilities. As the perturbation increases, our pipeline proves to stay relatively robust even at the large 100px initialization, keeping its median final reprojection error below 2px. Another interesting result is the net reduction in translation error, probably caused by the automatic rescaling factor, which allows better conditioning for the joint optimization of rotation and translation. A similar experiment was carried out on the ETH3D set in supplementary Table 10.

<table><tr><td></td><td colspan="10">CARLA</td></tr><tr><td></td><td>2 3</td><td></td><td>5</td><td>pix@ (%↑)</td><td>10</td><td>20</td><td>50 100</td><td>pe(px↓)</td><td>rot (°↓)</td><td>trans1 (cm↓)</td></tr><tr><td>0px initialization</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>MatchAnything</td><td>3 26</td><td></td><td>44 85</td><td>98</td><td>100</td><td>100</td><td>100</td><td>3.20±2.60</td><td>0.14±0.08</td><td>33.94±27.77</td></tr><tr><td>ReFineR</td><td>20</td><td>76</td><td>92 100</td><td>100</td><td>100</td><td>100</td><td>100</td><td>1.47±0.79</td><td>0.06±0.04</td><td>2.99±1.74</td></tr><tr><td>MatchAnything* + ReFineR</td><td>16</td><td>71</td><td>83 95</td><td>96</td><td>99</td><td>100</td><td>100</td><td>1.49±2.72</td><td>0.05±0.09</td><td>3.03±6.32</td></tr><tr><td>5px initialization</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>MatchAnything</td><td></td><td>2 17</td><td>39 78</td><td>95</td><td>99</td><td>100</td><td>100</td><td>3.35±3.20</td><td>0.13±0.12</td><td>32.50±32.13</td></tr><tr><td>ReFineR</td><td>11</td><td>63</td><td>82 95</td><td>99</td><td>100</td><td>100</td><td>100</td><td>1.68±1.72</td><td>0.06±0.07</td><td>3.29±3.82</td></tr><tr><td>MatchAnything* + ReFineR</td><td>16</td><td>66</td><td>84 96</td><td>99</td><td>100</td><td>100</td><td>100</td><td>1.49±1.59</td><td>0.06±0.06</td><td>3.05±3.72</td></tr><tr><td>10px initialization</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>MatchAnything</td><td>0</td><td>14</td><td>30 65</td><td>90</td><td>100</td><td>100</td><td>100</td><td>4.12±3.88</td><td>0.15±0.17</td><td>37.95±53.56</td></tr><tr><td>ReFineR</td><td>12</td><td>44</td><td>57 74</td><td>92</td><td>100</td><td>100</td><td>100</td><td>2.36±3.45</td><td>0.08±0.25</td><td>4.44±7.52</td></tr><tr><td>MatchAnything* + ReFineR</td><td>16</td><td>68</td><td>81 91</td><td>97</td><td>99</td><td>100</td><td>100</td><td>1.54±3.04</td><td>0.06±0.09</td><td>3.05±6.39</td></tr><tr><td>20px initialization</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>MatchAnything</td><td>0</td><td>11</td><td>19 49</td><td>85</td><td>96</td><td>100</td><td>100</td><td>5.13±4.94</td><td>0.19±0.24</td><td>43.47±66.34</td></tr><tr><td>ReFineR</td><td>2</td><td>14</td><td>18 21</td><td>26</td><td>82</td><td>100</td><td>100</td><td>16.67±7.70</td><td>0.94±0.64</td><td>26.50±16.88</td></tr><tr><td>MatchAnything* + ReFineR</td><td>18</td><td>66</td><td>82 90</td><td>94</td><td>98</td><td>100</td><td>100</td><td>1.54±4.19</td><td>0.06±0.22</td><td>3.06±8.75</td></tr><tr><td>50px initialization</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>MatchAnything</td><td>0</td><td>8</td><td>18 34</td><td>65</td><td>93</td><td>98</td><td>99</td><td>6.79±14.99</td><td>0.23±0.32</td><td>50.53±79.29</td></tr><tr><td>ReFineR</td><td>0</td><td>0</td><td>0</td><td>0</td><td></td><td>0 56</td><td>100</td><td>49.60±6.50</td><td>3.63±1.19</td><td>82.69±29.06</td></tr><tr><td>MatchAnything* + ReFineR</td><td>17</td><td>54</td><td>73 80</td><td>92</td><td></td><td>96</td><td>99 100</td><td>1.64±9.36</td><td>0.06±0.29</td><td>3.35±20.37</td></tr><tr><td>100px initialization</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>MatchAnything</td><td>1</td><td>3</td><td>9 31</td><td>55</td><td>77</td><td>95</td><td>100</td><td>8.32±15.98</td><td>0.28±0.87</td><td>81.66±202.21</td></tr><tr><td>ReFineR</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0 52</td><td>99.66±13.05</td><td>7.43±2.28</td><td>170.25±56.97</td></tr><tr><td>MatchAnything* + ReFineR</td><td>12</td><td>52</td><td>60</td><td>69</td><td>80</td><td>89</td><td>95 99</td><td>1.98±20.24</td><td>0.07±0.65</td><td>4.29±41.36</td></tr></table>

Table 4: Registration at scale experiment, each camera pose was initialized at increasing pixel errors (Section 5.5).

## 6 Limitations of this work

Several limitations were not addressed in this paper and still represent challenges for the fine image-to-point cloud registration task, which we will discuss here. First, we show in this paper that there is a strong dependence between registration capabilities and the point cloud quality. In particular, floating individual points in the 3D model and in front of the camera might be rendered, biasing the optimization. This is also the case for dynamic objects in the 3D scene, which create discrepancies between the real image and the rendered view. This issue was mitigated by manually removing such points in preprocessing, but results on nuScenes vs ETH3D show that noisier point clouds represent a greater challenge. In general, for any analysis-by-synthesis method, the correct rendering of occlusion is a major challenge: by standing close to walls or having a sparse point cloud, one might render points that are supposedly concealed by others, which, in the end, creates discrepancies with the real image once again.

For RGBDScenes, ETH3D, and Tanks and Temples datasets, color is different from reflectivity as it does not correspond to real frequency information associated with a given light ray/point. Instead, color is captured by a separate camera sensor and synchronized in post-processing with the 3D data and precomputed calibration. This implies that some error in color projection can be caused by bad sensor registration, which is precisely the problem we are aiming to solve. For our current resolution of 1024 px height, however, this error seems to be negligible (or less than a pixel), but it will surely appear at finer resolutions.

Finally, our manual dataset and evaluation present their limitations. On the one hand, as we saw for the scale estimation method, pixel error is entirely dependent on the chosen point’s depth: because of parallax, farther points will be less displaced than closer points for a fixed translation. This means that pixel error is simply a function of the depth distribution, which can be manipulated by selecting far points only. During our data acquisition, we ensured not to fall into this loophole by selecting evenly distributed points in depth. Second, camera estimation through point-pixel selection is itself a biased process toward certain kinds of rigid transforms. Indeed, because of the uncertainty matrix S defined in Eq. (3), some transform directions have less observability than others (e.g., the z axis in camera coordinates), which would not occur in multi-sensor setups with LiDAR-odometry registration. Third, due to uncertainty in selecting "the right pixel" for our manual registration, we were unable to achieve sub-pixel registration on our three manual sets. In Table 6 of the Supplementary, we observe final reprojection errors of 2px for Tanks and Temples and 3.5px for nuScenes on average. Additionally, the 1px threshold column in the Tanks and Temples ablation in Table 3 displays a complete column of 0%, while our method achieves subpixel registration on the similar ETH3D set. This evidence suggests that, with ReFineR, we reached the limits of manual registration, and that the true capabilities of our method are probably slightly better at the pixel level than the results we obtained from manual registration.

## 7 Conclusion

Through the paradigm of analysis-by-synthesis, we present ReFineR: a generalist, differentiable, and training-free approach for fine-level image-to-point cloud registration. Combined with MatchAnything, ReFineR achieves state-of-the-art results on heterogeneous image and point cloud data and is capable of fine registration even at large camera perturbations up to 100px. Furthermore, we have shown that thanks to a first-order analysis of the pinhole camera model, it is possible to remove the point cloud scale ambiguity from the optimization parameters, creating a unified method for both indoor and outdoor scenes.

## References

[1] Pei An, Junfeng Ding, Siwen Quan, Jiaqi Yang, You Yang, Qiong Liu, and Jie Ma. Survey of extrinsic calibration on lidar-camera system for intelligent vehicle: Challenges, approaches, and trends. IEEE Transactions on Intelligent Transportation Systems, 25 (11):15342–15366, 2024. doi: 10.1109/TITS.2024.3419758.

[2] Gilad Baruch, Zhuoyuan Chen, Afshin Dehghan, Tal Dimry, Yuri Feigin, Peter Fu, Thomas Gebauer, Brandon Joffe, Daniel Kurz, Arik Schwartz, and Elad Shulman. ARKitScenes - a diverse real-world dataset for 3D indoor scene understanding using mobile RGB-d data. In Thirty-Fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 1). URL https://openreview. net/forum?id=tjZjv\_qh\_CE.

[3] Hugo Blanc, Jean-Emmanuel Deschaud, and Alexis Paljic. Raygauss: Volumetric gaussian-based ray casting for photorealistic novel view synthesis. In IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), pages 1808–1817, 2025. doi: 10.1109/WACV61041.2025.00183.

[4] Eric Brachmann and Carsten Rother. Visual camera re-localization from rgb and rgb-d images using dsac. IEEE Transactions on Pattern Analysis and Machine Intelligence, 44(9):5847–5865, 2022. doi: 10.1109/TPAMI.2021.3070754.

[5] Eric Brachmann, Tommaso Cavallari, and Victor Adrian Prisacariu. Accelerated coordinate encoding: Learning to relocalize in minutes using rgb and poses. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 5044–5053, 2023. doi: 10.1109/CVPR52729.2023.00488.

[6] Holger Caesar, Varun Bankiti, Alex H. Lang, Sourabh Vora, Venice Erin Liong, Qiang Xu, Anush Krishnan, Yu Pan, Giancarlo Baldan, and Oscar Beijbom. nuscenes: A multimodal dataset for autonomous driving. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 11618–11628, 2020. doi: 10.1109/ CVPR42600.2020.01164.

[7] Nicholas Carlevaris-Bianco, Arash K Ushani, and Ryan M Eustice. University of michigan north campus long-term vision and lidar dataset. The International Journal ofRobotics Research, 35(9):1023–1035, 2016. doi: 10.1177/0278364915614638.

[8] Juan Castorena, Ulugbek S. Kamilov, and Petros T. Boufounos. Autocalibration of lidar and optical cameras via edge alignment. In IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 2862–2866, 2016. doi: 10. 1109/ICASSP.2016.7472200.

[9] Xu Chen, Zijian Dong, Jie Song, Andreas Geiger, and Otmar Hilliges. Category level object pose estimation via neural analysis-by-synthesis. In European Conference on Computer Vision (ECCV), 2020.

[10] Yuzhou Cheng, Jianhao Jiao, Yue Wang, and Dimitrios Kanoulas. LoGS: Visual localization via gaussian splatting with fewer training images. 2025 IEEE International Conference on Robotics and Automation (ICRA), pages 15029–15036.

[11] N. Dalal and B. Triggs. Histograms of oriented gradients for human detection. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), volume 1, pages 886–893 vol. 1, 2005. doi: 10.1109/CVPR.2005.177.

[12] Pierre Dellenbach, Jean-Emmanuel Deschaud, Bastien Jacquet, and François Goulette. Ct-icp: Real-time elastic lidar odometry with loop closure. In International Conference on Robotics and Automation (ICRA), pages 5580–5586, 2022. doi: 10.1109/ ICRA46639.2022.9811849.

[13] Jérémie Deray and Joan Solà. Manif: A micro Lie theory library for state estimation in robotics applications. Journal of Open Source Software, 5(46):1371, 2020. doi: 10.21105/joss.01371.

[14] Alexey Dosovitskiy, Germán Ros, Felipe Codevilla, Antonio M. López, and Vladlen Koltun. CARLA: An open urban driving simulator. In Conference on Robot Learning, 2017. URL https://carla.org/.

[15] Martin A. Fischler and Robert C. Bolles. Random sample consensus: a paradigm for model fitting with applications to image analysis and automated cartography. Commun. ACM, 24(6):381–395, June 1981. ISSN 0001-0782. doi: 10.1145/358669.358692. URL https://doi.org/10.1145/358669.358692.

[16] A Geiger, P Lenz, C Stiller, and R Urtasun. Vision meets robotics: The kitti dataset. The International Journal of Robotics Research, 32(11):1231–1237, 2013. doi: 10. 1177/0278364913491297.

[17] Ben Glocker, Shahram Izadi, Jamie Shotton, and Antonio Criminisi. Real-time RGB-d camera relocalization. In International Symposium on Mixed and Augmented Reality (ISMAR). IEEE. URL https://www.microsoft.com/en-us/research/ publication/real-time-rgb-d-camera-relocalization/.

[18] Prajnan Goswami, Tianye Ding, Feng Liu, and Huaizu Jiang. UniCorrn: Unified correspondence transformer across 2D and 3D. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2026.

[19] Maximilien Guislain, Julie Digne, Raphaëlle Chaine, and Gilles Monnier. Fine scale image registration in large-scale urban lidar point sets. Computer Vision and Image Understanding, 157:90–102, 2017. ISSN 1077-3142. doi: https://doi. org/10.1016/j.cviu.2016.12.004. URL https://www.sciencedirect.com/ science/article/pii/S1077314216302016.

[20] Xingyi He, Hao Yu, Sida Peng, Dongli Tan, Zehong Shen, Hujun Bao, and Xiaowei Zhou. MatchAnything: Universal Cross-Modality Image Matching with Large-Scale Pre-Training. arXiv e-prints, art. arXiv:2501.07556, January 2025. doi: 10.48550/ arXiv.2501.07556.

[21] Peter Henry, Dieter Fox, Achintya K. Bhowmik, and Rajiv Mongia. Patch volumes: Segmentation-based consistent mapping with rgb-d cameras. 2013 International Conference on 3D Vision, pages 398–405, 2013.

[22] Qi-Xing Huang, Hai Wang, and Vladlen Koltun. Single-view reconstruction via joint analysis of image and shape collections. ACM Transactions on Graphics (TOG), 34:1 – 10, 2015.

[23] Lukas Höllein, Aljavz Bovzivc, Michael Zollhofer, and Matthias Nießner. 3DGS-LM: Faster Gaussian-Splatting Optimization with Levenberg-Marquardt. Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), abs/2409.12892, 2024. URL https://github.com/lukasHoel/3DGS-LM? tab=readme-ov-file.

[24] Ganesh Iyer, R. Karnik Ram, J. Krishna Murthy, and K. Madhava Krishna. Calibnet: Geometrically supervised extrinsic calibration using 3d spatial transformer networks. In IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 1110–1117, 2018. doi: 10.1109/IROS.2018.8593693.

[25] Shuhao Kang, Youqi Liao, Jianping Li, Fuxun Liang, Yuhao Li, Xianghong Zou, Fangning Li, Xieyuanli Chen, Zhen Dong, and Bisheng Yang. Cofii2p: Coarse-to-fine correspondences-based image to point cloud registration. IEEE Robotics and Automation Letters, 9(11):10264–10271, 2024. doi: 10.1109/LRA.2024.3466068.

[26] Alireza G. Kashani, Michael J. Olsen, Christopher E. Parrish, and Nicholas Wilson. A review of lidar radiometric processing: From ad hoc intensity correction to rigorous radiometric calibration. Sensors, 15(11):28099–28128, 2015.

[27] Bingxin Ke, Kevin Qu, Tianfu Wang, Nando Metzger, Shengyu Huang, Bo Li, Anton Obukhov, and Konrad Schindler. Marigold: Affordable adaptation of diffusion-based image generators for image analysis. IEEE Transactions on Pattern Analysis and Machine Intelligence, pages 1–18, 2025. doi: 10.1109/TPAMI.2025.3591076.

[28] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuehler, and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Transactions on Graphics, 42(4), July 2023. ISSN 0730-0301. doi: 10.1145/3592433. URL https: //doi.org/10.1145/3592433.

[29] Arno Knapitsch, Jaesik Park, Qian-Yi Zhou, and Vladlen Koltun. Tanks and temples: benchmarking large-scale scene reconstruction. ACM Transactions on Graphics, 36 (4), July 2017. ISSN 0730-0301. doi: 10.1145/3072959.3073599. URL https: //doi.org/10.1145/3072959.3073599.

[30] Kenji Koide, Shuji Oishi, Masashi Yokozuka, and Atsuhiko Banno. General, singleshot, target-less, and automatic lidar-camera extrinsic calibration toolbox. In IEEE International Conference on Robotics and Automation (ICRA), pages 11301–11307, 2023. doi: 10.1109/ICRA48891.2023.10160691.

[31] Kevin Lai, Liefeng Bo, and Dieter Fox. Unsupervised feature learning for 3d scene labeling. In IEEE International Conference on Robotics and Automation (ICRA), pages 3050–3057, 2014. doi: 10.1109/ICRA.2014.6907298.

[32] Christoph Lassner and Michael Zollhöfer. Pulsar: Efficient sphere-based neural rendering. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 1440–1449, 2021.

[33] Vincent Lepetit, Francesc Moreno-Noguer, and Pascal Fua. EPnP: An Accurate O(n) Solution to the PnP Problem. International Journal of Computer Vision, 81:155–166, 2008. doi: 10.1007/s11263-008-0152-6. URL https://www.tugraz.at/fileadmin/user\_upload/Institute/ ICG/Images/team\_lepetit/publications/lepetit\_ijcv08.pdf.

[34] Kenneth Levenberg. A method for the solution of certain non-linear problems in least squares. Quarterly of Applied Mathematics, 2(2):164–168, 1944. ISSN 0033569X, 15524485. URL http://www.jstor.org/stable/43633451.

[35] Minhao Li, Zheng Qin, Zhirui Gao, Renjiao Yi, Chenyang Zhu, Yulan Guo, and Kai Xu. 2d3d-matr: 2d-3d matching transformer for detection-free registration between images and point clouds. In IEEE/CVF International Conference on Computer Vision (ICCV), pages 14082–14092, 2023. doi: 10.1109/ICCV51070.2023.01299.

[36] Changkun Liu, Shuai Chen, Yash Bhalgat, Siyan Hu, Zirui Wang, Ming Cheng, Victor Adrian Prisacariu, and Tristan Braud. Gs-cpr: Efficient camera pose refinement via 3d gaussian splatting. In Proceedings of the Thirteenth International Conference on Learning Representations (ICLR), 2025.

[37] Xiyuan Liu, Chongjian Yuan, and Fu Zhang. Targetless extrinsic calibration of multiple small fov lidars and cameras using adaptive voxelization. IEEE Transactions on Instrumentation and Measurement, 71:1–12, 2022. doi: 10.1109/TIM.2022.3176889.

[38] David G. Lowe. Object recognition from local scale-invariant features. Proceedings of the Seventh IEEE International Conference on Computer Vision, 2:1150–1157 vol.2.

[39] Xudong Lv, Boya Wang, Ziwen Dou, Dong Ye, and Shuo Wang. Lccnet: Lidar and camera self-calibration using cost volume network. In IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), pages 2888–2895, 2021. doi: 10.1109/CVPRW53098.2021.00324.

[40] Donald W. Marquardt. An Algorithm for Least-Squares Estimation of Nonlinear Parameters. Journal of The Society for Industrial and Applied Mathematics, 11:431–441, 1963.

[41] Hidenobu Matsuki, Riku Murai, Paul H. J. Kelly, and Andrew J. Davison. Gaussian splatting slam. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 18039–18048, 2024. doi: 10.1109/CVPR52733.2024.01708.

[42] Bortolon Matteo, Theodore Tsesmelis, Stuart James, Fabio Poiesi, and Alessio Del Bue. 6dgs: 6d pose estimation from a single image and a 3d gaussian splatting model. In European Conference on Computer Vision (ECCV), page 420–436, Berlin, Heidelberg, 2024. ISBN 978-3-031-72942-3. doi: 10.1007/978-3-031-72943-0\_24. URL https://doi.org/10.1007/978-3-031-72943-0\_24.

[43] Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: representing scenes as neural radiance fields for view synthesis. Communications of the ACM, 65(1):99–106, December 2021. ISSN 0001- 0782. doi: 10.1145/3503250. URL https://doi.org/10.1145/3503250.

[44] Juncheng Mu, Chengwei Ren, Weixiang Zhang, Liang Pan, Xiao-Ping Zhang, and Yue Gao. Diff2I2P: Differentiable image-to-point cloud registration with diffusion prior. 2025 IEEE/CVF International Conference on Computer Vision (ICCV), pages 25777– 25787, 2025.

[45] John A. Nelder and Roger Mead. A Simplex Method for Function Minimization. Comput. J., 7:308–313, 1965.

[46] Ni Ou, Zhuo Chen, Xinru Zhang, and Junzheng Wang. Iterative camera-LiDAR extrinsic optimization via surrogate diffusion. In 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 3069–3075, 2025. doi: 10.1109/IROS60139.2025.11247343.

[47] Gaurav Pandey, James R. McBride, Silvio Savarese, and Ryan M. Eustice. Automatic targetless extrinsic calibration of a 3d lidar and camera by maximizing mutual information. In Proceedings ofthe Twenty-Sixth AAAI Conference on Artificial Intelligence, AAAI’12, page 2053–2059. AAAI Press, 2012.

[48] Siyu Ren, Yiming Zeng, Junhui Hou, and Xiaodong Chen. Corri2p: Deep image-topoint cloud registration via dense correspondence. IEEE Transactions on Circuits and Systems for Video Technology, 33(3):1198–1208, 2023. doi: 10.1109/TCSVT.2022. 3208859.

[49] Paul-Edouard Sarlin, Daniel DeTone, Tomasz Malisiewicz, and Andrew Rabinovich. Superglue: Learning feature matching with graph neural networks. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 4937–4946, 2020. doi: 10.1109/CVPR42600.2020.00499.

[50] Thomas Schöps, Johannes L. Schönberger, Silvano Galliani, Torsten Sattler, Konrad Schindler, Marc Pollefeys, and Andreas Geiger. A multi-view stereo benchmark with high-resolution images and multi-camera videos. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2017.

[51] Pei Sun, Henrik Kretzschmar, Xerxes Dotiwalla, Aurélien Chouard, Vijaysai Patnaik, Paul Tsui, James Guo, Yin Zhou, Yuning Chai, Benjamin Caine, Vijay Vasudevan, Wei Han, Jiquan Ngiam, Hang Zhao, Aleksei Timofeev, Scott Ettinger, Maxim Krivokon, Amy Gao, Aditya Joshi, Yu Zhang, Jonathon Shlens, Zhifeng Chen, and Dragomir Anguelov. Scalability in perception for autonomous driving: Waymo open dataset. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 2443–2451, 2020. doi: 10.1109/CVPR42600.2020.00252.

[52] Zachary Taylor, Juan Nieto, and David Johnson. Multi-modal sensor calibration using a gradient orientation measure. Journal of Field Robotics, 32(5):675–695, 2015. doi: https://doi.org/10.1002/rob.21523. URL https://onlinelibrary. wiley.com/doi/abs/10.1002/rob.21523.

[53] Bill Triggs, Philip F. McLauchlan, Richard I. Hartley, and Andrew W. Fitzgibbon. Bundle adjustment - a modern synthesis. In Workshop on Vision Algorithms, 1999.

[54] Guangming Wang, Yu Zheng, Yuxuan Wu, Yanfeng Guo, Zhe Liu, Yixiang Zhu, Wolfram Burgard, and Hesheng Wang. End-to-end 2d-3d registration between image and

lidar point cloud for vehicle localization. IEEE Transactions on Robotics, 41:4643– 4662, 2025. doi: 10.1109/TRO.2025.3588454.

[55] Haiping Wang, Yuan Liu, Bing Wang, Yujing Sun, Zhen Dong, Wenping Wang, and Bisheng Yang. Freereg: Image-to-point cloud registration leveraging pretrained diffusion models and monocular depth estimators. In Proceedings of the Twelfth International Conference on Learning Representations (ICLR), 2024.

[56] Olivia Wiles, Georgia Gkioxari, Richard Szeliski, and Justin Johnson. Synsin: End-toend view synthesis from a single image. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 7465–7475, 2020. doi: 10.1109/CVPR42600. 2020.00749.

[57] Yueming Xu, Haochen Jiang, Zhongyang Xiao, Jianfeng Feng, and Li Zhang. Dg-slam: robust dynamic gaussian splatting slam with hybrid pose optimization. In Proceedings ofthe 38th International Conference on Neural Information Processing Systems, NIPS ’24, Red Hook, NY, USA, 2025. ISBN 9798331314385.

[58] Luwei Yang, Ziqian Bai, Chengzhou Tang, Honghua Li, Yasutaka Furukawa, and Ping Tan. Sanet: Scene agnostic network for camera localization. In IEEE/CVF International Conference on Computer Vision (ICCV), pages 42–51, 2019. doi: 10.1109/ICCV.2019.00013.

[59] Lin Yen-Chen, Pete Florence, Jonathan T. Barron, Alberto Rodriguez, Phillip Isola, and Tsung-Yi Lin. inerf: Inverting neural radiance fields for pose estimation. In IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 1323–1330, 2021. doi: 10.1109/IROS51168.2021.9636708.

[60] Chandan Yeshwanth, Yueh-Cheng Liu, Matthias Nießner, and Angela Dai. ScanNet++: A high-fidelity dataset of 3D indoor scenes. 2023 IEEE/CVF International Conference on Computer Vision (ICCV), pages 12–22. URL https://github.com/ scannetpp/scannetpp.

[61] Wang Yifan, Felice Serena, Shihao Wu, Cengiz Öztireli, and Olga Sorkine-Hornung. Differentiable surface splatting for point-based geometry processing. ACM Transactions on Graphics (proceedings ofACM SIGGRAPH ASIA), 38(6), 2019.

[62] Chongjian Yuan, Xiyuan Liu, Xiaoping Hong, and Fu Zhang. Pixel-level extrinsic self calibration of high resolution lidar and camera in targetless environments. IEEE Robotics and Automation Letters, 6(4):7517–7524, 2021. doi: 10.1109/LRA.2021. 3098923.

[63] Kaiwen Yuan, Zhenyu Guo, and Z. Jane Wang. Rggnet: Tolerance aware lidar-camera online calibration with geometric deep learning and generative model. IEEE Robotics and Automation Letters, 5(4):6956–6963, 2020. doi: 10.1109/LRA.2020.3026958.

## A Theory and implementation

## A.1 Pinhole camera model

The pinhole camera model computes, for each point $X \in \mathbb { R } ^ { 3 }$ in space, the projection $x = \pi ( X )$ onto the camera plane, in pixel units. Each camera is defined by its respective intrinsics and extrinsic parameters $K _ { i } \in \mathbb { R } ^ { 3 \times 3 }$ and $K _ { e } \in S E ( 3 )$ such that:

$$
\pi ( X ) = \varphi ( K _ { i } \varphi \left( K _ { e } \varphi ^ { - 1 } ( X ) \right) )\tag{8}
$$

with $\varphi$ and $\varphi ^ { - 1 }$ defined by

$$
\varphi ( x _ { 1 } , \ldots , x _ { n } ) = ( x _ { 1 } / x _ { n } , \ldots , x _ { n - 1 } / x _ { n } )\tag{9}
$$

$$
\varphi ^ { - 1 } ( x _ { 1 } , \dots , x _ { n - 1 } ) = ( x _ { 1 } , \dots , x _ { n - 1 } , 1 )\tag{10}
$$

## A.2 Reprojection error asymptotic development

Writing $K _ { i } = { \left( \begin{array} { l l l } { f _ { x } } & { 0 } & { c _ { x } } \\ { 0 } & { f _ { y } } & { c _ { y } } \\ { 0 } & { 0 } & { 1 } \end{array} \right) }$ and $A ( \tau , X ) = \exp ( \theta _ { \times } ) X + V ( \theta ) \rho = \left( X _ { X } \quad X _ { Y } \quad X _ { Z } \right) ^ { T }$ , with

$$
V ( \theta ) = I _ { 3 } + \frac { 1 - \cos ( \| \theta \| ) } { \| \theta \| ^ { 2 } } \theta _ { \times } + \frac { \| \theta \| - \sin ( \| \theta \| ) } { \| \theta \| ^ { 3 } } \theta _ { \times } ^ { 2 }\tag{11}
$$

and $\mathbf { \nabla } \cdot \times$ the antisymmetric operator, we then have from Eq. (8):

$$
\begin{array} { r } { \pi ( \tau , X ) = \varphi ( K _ { i } A ( \tau , X ) ) } \\ { = \left( f _ { x } \frac { X _ { X } } { X _ { Z } } + c _ { x } \right) } \\ { f _ { y } \frac { X _ { Y } } { X _ { Z } } + c _ { y } \vphantom { \bigg ( } } \end{array}\tag{12}
$$

(13)

Therefore

$$
{ \begin{array} { c c c } { { \overline { { \mathcal { D } } } } \pi } & { \displaystyle { ( 0 , X ) = { \frac { \partial \varphi } { \partial X } } ( K _ { i } X ) K _ { i } { \frac { \mathcal { D } A } { \mathcal { D } \tau } } ( 0 , X ) } } & { \displaystyle { ( 1 4 ) } } \\ { \displaystyle } & { \displaystyle { = \left( { \begin{array} { c c c c } { { \frac { 1 } { X _ { Z } } } } & { 0 } & { - { \frac { f _ { x } X _ { Y } + c _ { x } X _ { Z } } { X _ { Z } ^ { 2 } } } } \\ { 0 } & { { \frac { 1 } { X _ { Z } } } } & { - { \frac { f _ { y } X _ { Y } + c _ { y } X _ { Z } } { X _ { Z } ^ { 2 } } } } \end{array} } \right) \left( { \begin{array} { c c c c } { f _ { x } } & { 0 } & { c _ { x } } \\ { 0 } & { f _ { y } } & { c _ { y } } \\ { 0 } & { 0 } & { 1 } \end{array} } \right) \left( { \begin{array} { c c c c c } { 0 } & { X _ { Z } } & { - X _ { Y } } & { 1 } & { 0 } & { 0 } \\ { - X _ { Z } } & { 0 } & { X _ { X } } & { 0 } & { 1 } & { 0 } \\ { X _ { Y } } & { - X _ { X } } & { 0 } & { 0 } & { 0 } & { 1 } \end{array} } \right) } } \end{array} }\tag{15}
$$

$$
\begin{array} { r } { = \left( \begin{array} { c c c c } { \frac { f _ { x } } { X _ { Z } } } & { 0 } & { - \frac { f _ { x } X _ { X } } { X _ { Z } ^ { 2 } } } \\ { 0 } & { \frac { f _ { y } } { X _ { Z } } } & { - \frac { f _ { y } X _ { Y } } { X _ { Z } ^ { 2 } } } \end{array} \right) \left( \begin{array} { c c c c c } { 0 } & { X _ { Z } } & { - X _ { Y } } & { 1 } & { 0 } & { 0 } \\ { - X _ { Z } } & { 0 } & { X _ { X } } & { 0 } & { 1 } & { 0 } \\ { X _ { Y } } & { - X _ { X } } & { 0 } & { 0 } & { 0 } & { 1 } \end{array} \right) } \end{array}\tag{16}
$$

$$
= \left( \begin{array} { c c c c c c } { - \frac { x _ { c } y _ { c } } { f _ { y } } } & { \frac { x _ { c } ^ { 2 } + f _ { x } ^ { 2 } } { f _ { x } } } & { - \frac { f _ { x } y c } { f _ { y } } } & { \frac { f _ { x } } { X _ { Z } } } & { 0 } & { - \frac { x _ { c } } { X _ { Z } } } \\ { - \frac { y _ { c } ^ { 2 } + f _ { y } ^ { 2 } } { f _ { y } } } & { \frac { x _ { c } y _ { c } } { f _ { x } } } & { \frac { f _ { y } x _ { c } } { f _ { x } } } & { 0 } & { \frac { f _ { y } } { X _ { Z } } } & { - \frac { y _ { c } } { X _ { Z } } } \end{array} \right)\tag{17}
$$

with $\left( x _ { c } , y _ { c } \right) = \left( f _ { x } X _ { X } / X _ { Z } , f _ { y } X _ { Y } / X _ { Z } \right)$

## A.3 Scale estimation

Writing $X = \alpha X ^ { \prime }$ and

$$
J ( X ) = { \frac { \mathcal { D } \pi } { \mathcal { D } \tau } } ( 0 , X )\tag{18}
$$

$$
\begin{array} { r l } { = \left( J _ { r } ( X ) } & { { } J _ { t } ( X ) \right) } \end{array}\tag{19}
$$

$$
\begin{array} { r l } { = \left( J _ { r } ( X ) } & { { } { \frac { 1 } { \alpha } } J _ { t ^ { \prime } } ( X ) \right) } \end{array}\tag{20}
$$

then

$$
S = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } J ( X _ { i } ) ^ { T } J ( X _ { i } )\tag{21}
$$

$$
= \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \left( \begin{array} { l l } { J _ { r } ( X _ { i } ) ^ { T } J _ { r } ( X _ { i } ) } & { \frac { 1 } { \alpha } J _ { r } ( X _ { i } ) ^ { T } J _ { t ^ { \prime } } ( X _ { i } ) } \\ { \frac { 1 } { \alpha } J _ { t ^ { \prime } } ( X _ { i } ) ^ { T } J _ { r } ( X _ { i } ) } & { \frac { 1 } { \alpha ^ { 2 } } J _ { t ^ { \prime } } ( X _ { i } ) ^ { T } J _ { t ^ { \prime } } ( X _ { i } ) } \end{array} \right)\tag{22}
$$

$$
\mathbf { \Sigma } = \left( \begin{array} { l l } { S _ { r } } & { \frac { 1 } { \alpha } S _ { r t ^ { \prime } } } \\ { \frac { 1 } { \alpha } S _ { r t ^ { \prime } } ^ { T } } & { \frac { 1 } { \alpha ^ { 2 } } S _ { t ^ { \prime } } } \end{array} \right)\tag{23}
$$

$$
= \left( \begin{array} { c c } { S _ { r } } & { S _ { r t } } \\ { S _ { r t } ^ { T } } & { S _ { t } } \end{array} \right)\tag{24}
$$

The eigenvalues between $S _ { t }$ and $S _ { t ^ { \prime } }$ are related by

$$
\lambda _ { t ^ { \prime } , i } = \alpha ^ { 2 } \lambda _ { t , i }\tag{25}
$$

Therefore, equalizing the Hilbert-Schmidt norms:

$$
\sum _ { i = 1 } ^ { 3 } \lambda _ { t ^ { \prime } , i } ^ { 2 } = \sum _ { i = 1 } ^ { 3 } \lambda _ { r , i } ^ { 2 }\tag{26}
$$

$$
\Leftrightarrow \alpha ^ { 4 } \sum _ { i = 1 } ^ { 3 } \lambda _ { t , i } ^ { 2 } = \sum _ { i = 1 } ^ { 3 } \lambda _ { r , i } ^ { 2 }\tag{27}
$$

$$
\iff \alpha = \left( \sum _ { i = 1 } ^ { 3 } \lambda _ { r , i } ^ { 2 } \Big / \sum _ { i = 1 } ^ { 3 } \lambda _ { t , i } ^ { 2 } \right) ^ { 1 / 4 }\tag{28}
$$

Fig. 2 shows examples of 3 point clouds rescaled using the above equations for α.

## A.4 Initialization at a target pixel error

We explain here how to perturb the ground truth camera pose to obtain a fixed pixel error. Starting back from Eq. (3) around the identity, and using the first-order approximation, we wish to find a transform $\delta \tau$ such that for a given pixel error $\Delta$ we have:

$$
\delta \tau ^ { T } S \delta \tau = \Delta ^ { 2 }\tag{29}
$$

which by diagonalizing S reads:

$$
\Delta ^ { 2 } = ( P \delta \tau ) ^ { T } D ( P \delta \tau )\tag{30}
$$

$$
= \sum _ { i = 1 } ^ { 6 } \lambda _ { i } ( P \delta \tau ) _ { i } ^ { 2 }\tag{31}
$$

![](images/336769041f96344084919be3bb03b92163e5b0b2ac032d7f004023f7de0e6009.jpg)  
Figure 2: Three scenes from heterogeneous sources (nuScenes and Tanks and Temples), transformed to a unified scale using the method of Section 4.1.

with P an orthogonal matrix and $D = \mathrm { d i a g } ( \lambda _ { 1 } , . . . \lambda _ { 6 } )$ . Therefore, such a $\delta \tau$ can be found by randomly selecting a unit vector $\tau = ( \tau _ { 1 } , . . . , \tau _ { 6 } ) ^ { T }$ and computing:

$$
\delta \tau = \Delta P ^ { T } \left( \begin{array} { c } { { \frac { \tau _ { 1 } } { \sqrt { \lambda _ { 1 } } } } } \\ { { \vdots } } \\ { { \frac { \tau _ { 6 } } { \sqrt { \lambda _ { 6 } } } } } \end{array} \right)\tag{32}
$$

Note that this formulation remains an approximation of the pixel error due to the 1st order asymptotic development, which, in practice, might not produce exactly a pixel error of magnitude $\Delta$ for large perturbations.

## A.5 Implementation of differentiable HOG features

As a comparison to Guislain et al. [19], who proposed using HOG features [11], we implement a differentiable version of this metric to fit our pipeline. In four steps:

1. Gradient computation: Starting from an image $( I _ { p } ) _ { p }$ indexed by its pixel $p ,$ we compute the gradient magnitude and orientation images $( G _ { p } , \Theta _ { p } ) _ { p }$ using Sobel filters.

2. Cell histogram: A cell C is defined as a set of locally adjacent pixels: $C = \{ p _ { 1 } ^ { C } , \ldots , p _ { | C | } ^ { C } \}$ In practice, we used a kernel with cell size $k c = 4$ , with padding and stride hyperparameters. The cell size is then $| C | = k _ { C } ^ { 2 }$ . For each cell, a differentiable histogram $F _ { m } ^ { C }$ of unsigned

orientation is computed with M evenly spaced bin centers $( m _ { i } ) _ { 1 \leq i \leq M } \in [ 0 , \pi ] ^ { M }$

$$
F _ { i } ^ { C } = \sum _ { p \in C } G _ { p } f _ { \sigma } ( \Theta _ { p } , m _ { i } )\tag{33}
$$

with $f _ { \boldsymbol { \sigma } } ( \boldsymbol { \theta } , m )$ the Gaussian’s density with mean m and variance $\sigma ^ { 2 }$ , and σ fixed to $0 . 5 * \left( m _ { 2 } - m _ { 1 } \right)$

3. Block normalization: Similarly, a block B is defined as a set of adjacent cells: $\boldsymbol { B } = \{ C _ { 1 } ^ { B } \ldots C _ { | B | } ^ { B } \}$ , and the block size is defined using block kernel size $| B | = \bar { k _ { B } ^ { 2 } }$ . We used $k _ { B } = 4$ . The normalized features $\left( F _ { i } ^ { B , C } \right) _ { B , C , i }$ are computed as:

$$
N ^ { B } = \sqrt { \frac { 1 } { M | B | } \sum _ { C \in B , 1 \leq i \leq M } \left( F _ { i } ^ { C } \right) ^ { 2 } }
$$

$$
F _ { i } ^ { B , C } = \frac { F _ { i } ^ { C } } { N _ { B } }\tag{34}
$$

4. Loss function: Given two images I and I<sup>′</sup>, and the features $\left( F _ { i } ^ { B , C } \right) _ { B , C , i } , \left( F _ { i } ^ { \prime B , C } \right) _ { B , C , i } ,$ the full feature loss is:

$$
H O G ( I , I ^ { \prime } ) = \frac { \underset { B , C , i } { \sum } \left( F _ { i } ^ { B , C } - F _ { i } ^ { \prime B , C } \right) ^ { 2 } } { | B | | C | M }\tag{35}
$$

## B Dataset preparation

## B.1 ETH3D

From the ETH3D dataset [50], we reconstructed the global point clouds using ground truth alignment files and subsampled them using a 5 mm voxel grid. Then the normals’ orientations were computed using CloudCompare. Fifty images spanning ten scenes were then randomly chosen to form our dataset, along with their ground truth camera positions and intrinsics. Thanks to its pixel-level refinement scheme, this dataset presents sub-pixel camera registration for resolutions up to $6 0 4 8 \times 4 0 3 2$ . Additionally, each point cloud was acquired using a Faro Focus X 330, providing centimeter-level point precision.

## B.2 Tanks and Temples

We chose the preprocessed Church and Meetingroom LiDAR point clouds from Tanks and Temples [29] and ten images from each scene. The available LiDAR point clouds were cleanly preprocessed by the original authors, and because of their high number of points, we randomly subsampled them, keeping 40% of the points during our experiments. Since camera positions and intrinsics were not given by the authors, we used a manual point-pixel matching protocol to estimate the extrinsic parameters as well as the shared intrinsics and 5 distortion parameters following the OpenCV distortion model. The final reprojection errors were around 2px on average.

## B.3 nuScenes

We chose four scenes (1, 3, 7, and 12) from the nuScenes [6] dataset. For each scene, the LiDAR scans were aggregated into a global point cloud using CT-ICP [12], and we removed dynamic objects using the ground truth annotations and manual selection. We also applied histogram equalization on the LiDAR intensity values to enhance contrast. Five images were selected for each scene, all from the same front-right camera, taking care to choose images where the point cloud would be dense enough to fill most of the image when projected. Ground truth camera poses were similarly obtained using our manual matching.

## B.4 Livox

We took 4 Livox scans with their associated images from Yuan et al.’s work [62], consisting of outdoor scenes. Despite its relatively small size, we chose to include this set for a fair comparison with Yuan et al.’s work [62], whose authors built a method for the specific case of Livox sensors.

## B.5 RGBDScenes

We selected 5 images from scenes 01, 07, 09, and 13 of the RGBDScenes dataset [31]. Calibration was kept unchanged, and we used $( f _ { x } , f _ { y } , c _ { x } , c _ { y } ) = ( 5 2 5 , 5 2 5 , 3 2 0 , 2 4 0 )$ as our intrinsics values. This set represents a drastic shift in point cloud quality from the other sets because of the sensor’s resolution, but has the advantage of intrinsically having color intensity, which can be leveraged for our task.

## B.6 CARLA

We used the CARLA simulator [14] to generate five sequences of synthetic data. The sequences were recorded in Towns 2, 3, 4, 5, and 10. We placed a rotating LiDAR and two cameras on the main vehicle, one facing forward and the other facing right; the sensor parameters are given in Table 5. Parameters that are not mentioned are kept at their default values (see here<sup>1</sup>). We also provide for each modality the associated semantic information, which was used in Section C.8 to assess a 2D to 3D feature transfer quality in regard to a perturbed camera position. We noticed that the CARLA LiDAR had inaccuracies (missing objects or buildings); therefore, we manually selected 10 images from each sequence to mitigate the discrepancies between images and point clouds. Being synthetically generated, this dataset has the benefit of having floating-point-perfect image registration.

## B.7 Common preprocessing

For these different sets, we computed normal directions using a local PCA of 50 neighbors. All images were resized to a fixed height of 1024px except for the RGBDScenes set, which was kept at a resolution of $6 4 0 \times 4 8 0$ . Example point clouds are shown in Fig. 3, and images with point splatting renders are shown in Fig. 4.

![](images/55d69cb441e90a80420be32ed7506f9f098cfb851ac858f89d001a442b7b2978.jpg)  
(a) ETH3D

![](images/3bf7e9029da4cc0df915b5ca3acb789782a4ac3ba86d0867d244745660e7cda9.jpg)  
(b) Tanks And Temples

![](images/7839fa2bcc642fa1b3dcf80e8d4947d871a59c52c143a56ce7eef7c0ca87bed9.jpg)  
(c) CARLA

![](images/079171b6abf68387183c31b9cf5b42a4e85fac521f16bd6c543b0753157b8282.jpg)

![](images/3e8dddae21252ed2481ab4392dc3e1588b48d4a45023a2b5b38c9e06e20f0f09.jpg)  
(e) nuScenes

(d) Livox  
![](images/91d6e0666c469bea146911f2b86ac13bf99ea853ca49da6a34dbb076d2bd513c.jpg)  
(f) RGBDScenes  
Figure 3: Example point clouds from our benchmark dataset.

![](images/c4e32d2a0cb5a9ebef6daf25430b8356f2843705247437ca61496f846e567cea.jpg)

![](images/c52fbb67f3f7c68a4b6dadebc875b0745d1dc2b7f31b07b983f9c982df4c7d89.jpg)  
(a) i<sup>g</sup>  
(b) p<sup>nx</sup>

![](images/2597ffdd414f62b7df3e77cfbc674d0bf1f9c28fa567bbab4f6605783262ce41.jpg)

![](images/5533f48d004f7a77bd576c05181d5acf170e97c69015cbb90782e32f480da959.jpg)

(c) p<sup>g</sup>  
![](images/4ee3fe749c9a2eace8b5d257152cf747922de4f2e99e57337be8a7a8158a8161.jpg)

![](images/c84629607c223659f9e6df074d7380655932aa164fdc741a71d6700498ded75e.jpg)  
(f) $p ^ { g }$

(d) i<sup>g</sup>  
(e) p<sup>nx</sup>  
![](images/e0e864196035196479bf65f7426b2af46ecfbe874430fbf904aa17094518f202.jpg)

![](images/64d375568dd7f7d04063ee52cbff538f465111782c34f2fd28a43f55f468a99a.jpg)  
(h) $p ^ { n x }$

(g) i<sup>g</sup>  
![](images/f41a484e507b47cac7d7b3031e76aabb4426345509edb77eb6f32227c8b50b48.jpg)

![](images/da1a4ce492dd08acc715f3db1cbcb86eb8cfacd52fc436fd17a1f48acc4d6dc4.jpg)

![](images/bf2f629609ed9aa6fd7413c4fd685f20bdc8df160d0d8b236158cedf10f3eea9.jpg)

(i) i<sup>g</sup>  
![](images/10e06cafe888844ddb877e3598e41bddd0cea734e43da237adc994b67134e266.jpg)

(j) $p ^ { n x }$  
![](images/d58e016a70a9a0ec3603591df5cea31ce018389b0c5293de5ee27a30a4b26cfc.jpg)

(k) p<sup>r</sup>  
![](images/a870ae7dda40e6c37f54cdd6f15c2d52cfef7fcb2b6f9c169a6aa3bdaa900abe.jpg)  
(m) p<sup>nx</sup>

(l) i<sup>g</sup>  
![](images/b615476e7a368a7388beacf83970905256c7c23204ec1ac2127e37e2f1308465.jpg)  
(o) i<sup>g</sup>

(n) p<sup>r</sup>  
![](images/345718abf6c5988e9ed0902110c38b9d5ba7e557a7bc482ccd5d772097084eba.jpg)  
(p) $p ^ { n x }$

![](images/324c96d2ace6dede806f93dc30562445bc29bc9ae32d437d58d6ec66245f1052.jpg)  
(q) $p ^ { g }$  
Figure 4: Example of images and rendered views from our dataset using point normals or intensity. From top to bottom: ETH3D, Tanks and Temples, CARLA, Livox, nuScenes, RGBDScenes.

<table><tr><td>Sensor / Parameter</td><td>value</td></tr><tr><td>Camera</td><td></td></tr><tr><td>image_size_x</td><td>1920</td></tr><tr><td>image_size_y</td><td>1080</td></tr><tr><td>fov</td><td>90</td></tr><tr><td>LiDAR</td><td></td></tr><tr><td>points_per_second</td><td>500,000</td></tr><tr><td>rotation_frequency</td><td>5</td></tr><tr><td>dropoff_general_rate</td><td>0</td></tr><tr><td>dropoff_intensity_limit</td><td>0</td></tr><tr><td>range</td><td>200</td></tr><tr><td>channels</td><td>64</td></tr></table>

Table 5: CARLA sensor setup.

## B.8 Manual registration

For the nuScenes, Tanks and Temples, and Livox datasets, we manually computed our ground truth from 2D-3D correspondences and least-squares minimization. For each image, we manually selected around 15 point-pixel pairs, which we used to find the camera position.

For Tanks and Temples, we also noticed the presence of camera distortion and approximate intrinsic values, which could not be solved by simply estimating the camera position. For this set, we therefore jointly optimized the camera extrinsics, intrinsics, and distortion through least-squares minimization, keeping a single camera model (intrinsics and distortion) among all the images of the same scene. Final reprojection errors are given in Table 6.

This manual process may itself present biases as the final given errors are computed with respect to the selected matches, which may themselves be inaccurate. This was especially the case on the nuScenes set, for which the image edges were often blurry because of the video acquisition. RGBDScenes samples were kept with their official ground truth, as the point cloud noise prevented us from selecting accurate enough correspondences. For most images, we were able to significantly reduce the reprojection error down to a few pixels compared to the official registration for each dataset. Our approach is all the more motivated by finding a method to avoid this manual labor. We also observe from Table 6 that the initial error given by the official registration is on the order of 7-10 pixels, based on our manual correspondences on Tanks and Temples and nuScenes. This supports our assumption that fine registration should be studied from an initialization with a reprojection error of about 10 pixels. Qualitative results of our registration procedure are shown for nuScenes and Tanks and Temples in Fig. 5 and Fig. 6.

## C Benchmark information and additional experiments

## C.1 Benchmark methods

We compared ReFineR with 4 different state-of-the-art methods for fine-level registration. MatchAnything [20] is a transformer-based feature-matching model trained on multimodal images. We used their code and pretrained models available here<sup>2</sup>. For our benchmark, we used the RoMa model as it yielded better matches for our task.

![](images/462c15e7a55dbf8cd0868ecc10003be7fb633845263dbe85187548ca78e09f06.jpg)  
Figure 5: Manual registration results on nuScenes. Left: before manual registration, right: after.

![](images/c9460df3b4f917d2a8d1d5bfbbf5d2f6d6a7fe8c8fb5ab3ba79f6a6d02ace617.jpg)  
Figure 6: Manual registration results on Tanks and Temples. Left: before manual registration, right: after.

We ran Yuan et al.’s method [62] from their repository code<sup>3</sup>. Depending on the scene modality, we used their predefined configurations in order to adapt the edge extraction to the point cloud scale. For instance, the voxel size parameter is set to 0.5 for indoor scenes and 1.0 for outdoor scenarios. All parameters can be found in their configuration file.

For Koide et al.’s method [30]<sup>4</sup>, we only changed the "delta\_trans\_thresh" parameter, which defines the translation threshold used to claim algorithm convergence. For indoor and outdoor scenes, we respectively set it to 1 and 10 cm.

GS-SLAM [41] is the least robust method for our benchmark as it was designed to jointly optimize camera poses and Gaussians’ parameters in a Gaussian Splatting model, which only leverages photometric information between the 3D and 2D modalities, and is difficult to apply to broader ranges of point clouds (LiDAR, Livox, RGBD).

We reimplemented the method from Taylor et al. [52] with SciPy’s version of the Nelder-Mead [45] optimizer.

Contrary to Yuan et al. and Koide et al.’s methods, ReFineR does not need any adjustment to the point cloud scale and scene modality.

## C.2 Rotation and translation errors

Results from Section 5.2 and Section 5.3 are expressed in terms of rotation and translation errors in Table 7 and Table 8, respectively.

## C.3 Qualitative results

Fig. 7 and Fig. 8 show qualitative results for registration on the raw point cloud and using point cloud intensity (grayscale value or LiDAR reflectivity).

## C.4 Supervision channel

Apart from point cloud normals and intensity, as well as image RGB color, we experimented with different comparison modalities. Supervising the registration with the rendered depth $( \mathcal { C } _ {  } = \{ ( p ^ { d } , i ^ { g } ) \} )$ , we generally observe a decrease in registration performance in all benchmarking datasets. We suspect that this is the case because the main source of gradients in depth images arises from discontinuities between objects and gradients converging towards vanishing points, whereas rendered normal images from point clouds provide a richer set of gradients, including regions where the surface orientation changes (e.g., wall edges).

Another source of supervision could come from the monocular estimation of depth and normals with the Marigold diffusion model [27]. From a single RGB image, Marigold uses its trained geometric priors to estimate the normal orientation and normalized depth from a single image. For the normals, we kept the point splatting rendering of normals and reoriented them using the rendered view such that both the normal frames from the render and the generation coincide, and the rendered depth was normalized. We compared these true rendered channels with the monocular estimated normals and depth $( { \mathcal { C } } _ {  } =$ $\{ ( p ^ { n x } , i ^ { n x } ) , ( p ^ { n y } , i ^ { n y } ) , ( p ^ { n z } , i ^ { n z } ) \}$ and $\mathcal { C } _ {  } = \{ ( p ^ { d } , i ^ { d } ) \} )$ . Table 9 shows that using the monocular normals produces better registration than the monocular depth, and sometimes matches the performance of the ReFineR baseline. Both synthetic modalities, however, generally struggle at finer registration levels. We may explain this because of the coarse resemblance between the image’s gradient and its diffusion-generated counterpart, which allows for some edge supervision, but this supervision is limited by the current performance of diffusion models, which still create blurry edges, removing any pixel-level registration ability.

## C.5 Registration at scale on ETH3D

We ran a similar experiment to the one in Section 5.5 on the ETH3D dataset. Results in Table 10 show strong performance of the MatchAnything\* + ReFineR pipeline even at large perturbations such as 100px.

## C.6 Computation times

The experiments were carried out on Ubuntu 24.04.2 LTS with an Intel Core Ultra 9 285K CPU and NVIDIA GeForce RTX 5090 GPU. We provide in Table 11 average computation times on CARLA with normals and ETH3D with RGB color. ReFineR on its own is on average as fast as Koide et al.’s method on CARLA, but a little slower when adding MatchAnything\* as a first stage. Two main factors impact ReFineR’s runtime: first, the quality of the point cloud: a more precise point cloud will require fewer iterations to be registered; second, the number of points, which directly correlates with the Gaussian splatting forward and backward passes as shown in Table 12. However, this second point is mitigated by the tile-based Gaussian Splatting rendering CUDA engine, which parallelizes computations. By contrast, Koide et al.’s method does not have such an implementation and scales more slowly with the number of points. Large computation times for GS-SLAM on CARLA are caused by its complete divergence, forcing the algorithm to run until the maximum number of iterations (arbitrarily fixed) is reached. On the contrary, on the ETH3D set with color registration, GS-SLAM is able to converge in roughly the same amount of time as ReFineR. From our experiments with ReFineR alone on the ETH3D set, our method converges on average after 200 iterations, while using GS-SLAM’s photometric loss, the algorithm settled after 330 iterations. This result explains the time gap between GS-SLAM (10s / image) and ReFineR (7s / image), and tends to prove that GOM is also a more efficient supervision metric for this registration task than photometric loss. Adding MatchAnything as a first stage provides more accurate initialization, which consequently reduces the number of required iterations. For ETH3D, the algorithm looped only 80 times on average, therefore reducing the total amount of time compared to ReFineR alone.

## C.7 Point downsampling

In Table 13 we show a controlled robustness experiment on CARLA and ETH3D by randomly downsizing the point cloud (from 75% to 5%) to better characterize the density boundary of the method. We observe that starting from a 10-pixel perturbation, the final error monotonically decreases when the density increases, proving that ReFineR performs better with denser point clouds.

## C.8 Impact of pixel-level misalignment on 3D semantic label transfer

To assess the practical importance of fine image-LiDAR registration, we conducted an additional experiment measuring how semantic labels transferred from a single segmented image to a dense point cloud degrade as the camera pose is increasingly perturbed. Starting from the reference pose, we introduced controlled perturbations on 100 cameras of our CARLA dataset corresponding to reprojection errors ranging from 0 to 20 pixels, projected the point cloud into the image, and assigned semantic labels only to geometrically visible points using occlusion estimation from a splatting rendering. We then evaluated the consistency of the resulting 3D labels with the reference assignment using mIoU on 20 semantic classes.

The results in Fig. 9 show that semantic transfer is highly sensitive to small pose errors. In particular, the average mIoU drops sharply from 1.0 at 0 pixels to about 0.4 at 5 pixels, and to about 0.33 at 10 pixels. This strong degradation is mainly caused by thin structures and object boundaries (such as poles, barriers, lamp posts, and contour regions), which are the first to be incorrectly reprojected under small pose perturbations. In contrast, the degradation becomes much slower beyond 10 pixels, with mIoU decreasing only marginally from about 0.33 to 0.27 between 10 and 20 pixels. This observation supports the importance of pixellevel image-LiDAR registration.

## C.9 Qualitative registration experiment using UniCorrn

In preliminary experiments, we ran different inferences of the state-of-the-art image-topoint cloud registration model UniCorrn [18]. This model trains several layers of self- and cross-attention, embedding 2D and 3D features in a common space in order to generate dimensionality-agnostic descriptors and correspondences. In the 2D-3D case, the model was trained using the 7scenes [17] and RGB-D Scenes v2 [31] datasets, supplemented with synthetic depth maps from ScanNet++ [60] and ARKitScenes [2]. From Fig. 10, we suspect that the training data specialized the model for indoor depth images, which have particular geometry and density. Running the same model on general LiDAR point clouds may then represent a distribution shift for the model, which explains the weak results. We believe that this claim should be studied in more depth as it may represent a limitation for other similar methods [35, 55].

<table><tr><td>Set / Scene / Image</td><td>| Official registration (px)</td><td>I Manual registration (px)</td></tr><tr><td>Tanks and Temples</td><td></td><td></td></tr><tr><td>Church</td><td></td><td></td></tr><tr><td>21.png</td><td>5.13</td><td>1.37</td></tr><tr><td>40.png</td><td>4.35</td><td>1.89</td></tr><tr><td>83.png</td><td>7.27</td><td>1.99</td></tr><tr><td>110.png</td><td>4.4</td><td>1.28</td></tr><tr><td>190.png</td><td>4.0</td><td>1.68</td></tr><tr><td>281.png</td><td>7.6</td><td>1.24</td></tr><tr><td>332.png</td><td>6.44</td><td>1.07</td></tr><tr><td>358.png</td><td>7.66</td><td>1.37</td></tr><tr><td>408.png</td><td>5.11</td><td>1.19</td></tr><tr><td>438.png</td><td>364.66</td><td>1.31</td></tr><tr><td>Meetingroom</td><td></td><td></td></tr><tr><td>2.png</td><td>7.24</td><td>1.83</td></tr><tr><td>28.png</td><td>8.1</td><td>1.37</td></tr><tr><td>43.png</td><td>7.88</td><td>1.31</td></tr><tr><td>75.png</td><td>7.47</td><td>1.55</td></tr><tr><td>117.png</td><td>7.01</td><td>1.49</td></tr><tr><td>171.png</td><td>5.09</td><td>2.15</td></tr><tr><td>183.png</td><td>10.04</td><td>2.16</td></tr><tr><td>281.png</td><td>7.48</td><td>1.15</td></tr><tr><td>301.png</td><td>11.34</td><td>1.55</td></tr><tr><td>338.png</td><td>9.21</td><td>1.44</td></tr><tr><td>nuScenes</td><td></td><td></td></tr><tr><td>scene 1</td><td></td><td></td></tr><tr><td>12.png</td><td>11.26</td><td>3.67</td></tr><tr><td>18.png</td><td>10.98</td><td>4.69</td></tr><tr><td>23.png</td><td>8.3</td><td>2.36</td></tr><tr><td>32.png</td><td>8.35</td><td>4.6</td></tr><tr><td>38.png</td><td>7.53</td><td>4.7</td></tr><tr><td>scene 12</td><td></td><td></td></tr><tr><td>7.png</td><td>9.03</td><td>3.17</td></tr><tr><td>26.png</td><td>8.54</td><td>2.55</td></tr><tr><td>31.png</td><td>7.53</td><td>2.62</td></tr><tr><td>34.png</td><td>8.37</td><td>2.97</td></tr><tr><td>45.png</td><td>7.89</td><td>3.19</td></tr><tr><td>scene 3</td><td></td><td></td></tr><tr><td>6.png</td><td>8.76</td><td>2.77</td></tr><tr><td>11.png</td><td>9.32</td><td>3.1</td></tr><tr><td>18.png</td><td>8.55</td><td>2.47</td></tr><tr><td>32.png</td><td>9.62</td><td>2.91</td></tr><tr><td>39.png</td><td>9.2</td><td>2.17</td></tr><tr><td>scene 7</td><td></td><td></td></tr><tr><td>1.png</td><td>8.46</td><td>4.74</td></tr><tr><td>9.png 23.png</td><td>8.24</td><td>2.72</td></tr><tr><td>27.png</td><td>9.05</td><td>2.54</td></tr><tr><td></td><td>9.58</td><td>2.71</td></tr><tr><td>33.png</td><td>8.18</td><td>5.19</td></tr><tr><td>Livox</td><td></td><td></td></tr><tr><td>0.png</td><td>N/A</td><td>3.6</td></tr><tr><td>1.png</td><td>N/A</td><td>3.84</td></tr><tr><td>2.png</td><td>N/A</td><td>3.56</td></tr><tr><td>3.png</td><td>N/A</td><td>3.65</td></tr></table>

Table 6: Reprojection error before and after manual registration of our images from Tanks and Temples, nuScenes, and Livox. Initial registration was obtained from the official dataset’s ground truth for Tanks and Temples and using a state-of-the-art odometry algorithm for nuScenes.

<table><tr><td></td><td colspan="2">ETH3D</td><td colspan="2">Tanks and Temples</td><td colspan="2">CARLA</td><td colspan="2">Livox</td><td colspan="2">nuScenes</td><td colspan="2">RGBDScenes</td></tr><tr><td></td><td colspan="2">rot (°↓) transl (cm↓)</td><td colspan="2">rot (°↓) transl (cm↓)</td><td colspan="2">rot (°↓) transl (cm↓)</td><td colspan="2">rot (°↓) transl (cm↓)</td><td colspan="2">rot (°↓) transl (cm↓)</td><td colspan="2">rot (°↓) transl (cm↓)</td></tr><tr><td>One stage</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GS-SLAM []</td><td> $8 . 1 9 { \scriptstyle \pm 2 . 3 0 }$ </td><td> $5 7 . 0 8 \substack { \pm 3 7 . 9 8 }$ </td><td> $8 . 0 5 { \scriptstyle \pm 1 . 3 2 }$ </td><td> $1 0 1 . 6 9 _ { \pm 6 9 . 4 2 }$ </td><td> $9 . 0 0 { \scriptstyle \pm 1 . 2 3 }$ </td><td>333.86±101.20</td><td> $7 . 5 8 { \scriptstyle \pm 1 . 1 5 }$ </td><td> $7 1 . 9 5 \substack { \pm 3 8 . 5 2 }$ </td><td> $6 . 9 2 \div 2 . 0 8$ </td><td> $1 8 0 . 3 4 \div 3 7 . 0 2$ </td><td> $7 . 9 3 { \scriptstyle \pm 2 . 5 7 }$ </td><td> $2 2 . 3 5 { \scriptstyle \pm 5 . 4 9 }$ </td></tr><tr><td>Yuan et al. []</td><td> $0 . 7 5 { \scriptstyle \pm 0 . 9 2 }$ </td><td> $7 . 9 9 \pm 2 2 . 4 7$ </td><td> $0 . 3 0 { \scriptstyle \pm 0 . 4 7 }$ </td><td> $6 . 1 2 \pm 1 1 . 0 4$ </td><td> $0 . 7 9 { \scriptstyle \pm 1 . 5 1 }$ </td><td> $1 9 9 . 6 8 \substack { \pm 4 9 7 . 1 8 }$ </td><td> $0 . 4 6 \div 0 . 1 2$ </td><td> $6 . 7 9 { \scriptstyle \pm 6 . 4 6 }$ </td><td> $1 . 1 7 { \scriptstyle \pm 2 7 6 }$ </td><td> $9 0 . 9 3 { \scriptstyle \pm 1 7 9 . 3 2 }$ </td><td> $4 . 9 2 \substack { \pm 4 . 1 1 }$ </td><td> $1 2 . 7 8 \substack { \pm 2 0 . 3 5 }$ </td></tr><tr><td>Koide et al. []</td><td> $0 . 7 9 { \scriptstyle \pm 2 5 3 }$ </td><td> $1 1 . 5 2 \substack { \pm 1 7 . 0 4 }$ </td><td> $1 . 3 3 \div 3 . 6 8$ </td><td> $3 2 . 3 8 \substack { \pm 6 7 . 3 1 }$ </td><td> $0 . 4 8 \substack { \pm 0 . 7 9 }$ </td><td> $9 4 . 4 6 \substack { \div 3 5 1 . 1 9 }$ </td><td> $0 . 3 5 { \scriptstyle \pm 0 . 4 7 }$ </td><td>9.78±8.92</td><td> $0 . 4 9 { \scriptstyle \pm 0 . 6 0 }$ </td><td> $4 6 . 3 0 { \scriptstyle \div } 6 7 . 0 5$ </td><td> $6 . 6 0 { \scriptstyle \pm 3 . 7 1 }$ </td><td> $2 3 . 3 8 \pm 1 4 . 4 6$ </td></tr><tr><td>Taylor et al. [] MatchAnything []</td><td> $0 . 5 5 { \scriptstyle \pm 0 . 4 8 }$ </td><td> $6 . 2 8 { \scriptstyle \pm 1 0 . 3 0 }$ </td><td> $0 . 3 8 { \scriptstyle \pm 0 . 3  s }$ </td><td> $7 . 1 9 { \scriptstyle \pm 1 2 . 5 9 }$ </td><td> $0 . 3 7 { \scriptstyle \pm 0 . 4 1 }$ </td><td> $8 4 . 8 6 \div 1 7 2 . 5 5$ </td><td> $0 . 5 7 a g$ </td><td> $8 . 1 7 { \scriptstyle \pm 7 . 1 4 }$ </td><td> $0 . 7 6 { \scriptstyle \pm 1 . 0 9 }$  0.15±0.11</td><td> $6 5 . 9 7 \div 1 6 5 . 0 4$ </td><td> $2 . 2 6 { \scriptstyle \pm 1 . 0 2 }$ </td><td> $5 . 6 4 \pm 2 . 8 6$  2.48±1.28</td></tr><tr><td>ReFineR</td><td> $0 . 1 3 { \scriptstyle \pm 0 . 1 1 }$ </td><td> $1 . 3 2 \div 1 . 6 2$   $1 . 0 2 \substack { \pm 4 6 9 }$ </td><td> $0 . 1 2 \substack { \pm 0 . 0 8 }$  0.08±0.22</td><td>2.78±3.75 1.24±4.97</td><td>0.15±0.17 0.08±0.25</td><td>37.95±53.56 4.44±7.52</td><td>0.28±0.11</td><td>4.84±4.99</td><td>0.17±0.32</td><td> $1 4 . 9 3 \substack { \pm 2 0 . 8 4 }$  5.63±9.76</td><td> $1 . 1 3 { \scriptstyle \pm 0 . 5 5 }$ </td><td></td></tr><tr><td></td><td> $0 . 1 0 z o 3 6$ </td><td></td><td></td><td></td><td></td><td></td><td>0.33±0.13</td><td>6.86±5.93</td><td></td><td></td><td> $2 . 2 0 { \scriptstyle \pm 1 . 0 0 }$ </td><td> $6 . 1 6 { \scriptstyle \pm 2 . 5 8 }$ </td></tr><tr><td>Two stages</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>MatchAnything + GS-SLAM []</td><td> $7 . 7 0 { \scriptstyle \pm 2 . 2 }$ </td><td> $5 8 . 4 5 { \scriptstyle \pm 3 3 0 6 }$ </td><td> $7 . 9 8 \div 0 . 8 1$ </td><td> $1 4 2 . 2 2 \substack { \pm 6 7 . 1 7 }$ </td><td> $8 . 7 5 { \scriptstyle \pm 1 . 3 2 }$ </td><td>331.93±105.16</td><td> $7 . 7 5 { \scriptstyle \pm 0 . 9 4 }$ </td><td> $8 3 . 0 7 { \scriptstyle \pm 3 3 . 7 9 }$ </td><td> $7 . 4 1 { \scriptstyle \pm 1 . 8 3 }$ </td><td> $1 8 4 . 2 4 \div 3 6 . 2 5$ </td><td> $\Pi 2 \Pi { \pm } \ m a _ { n }$ </td><td> $2 2 . 1 1 { \scriptstyle \pm 1 1 . 1 7 }$ </td></tr><tr><td>MatchAnything + Yuan et al. []</td><td> $0 . 7 7 \pm \ast 7 7$ </td><td> $7 . 4 5 \pm 1 6 . 2 6$ </td><td> $0 . 2 6 { \scriptstyle \pm 0 . 2 9 }$ </td><td> $6 . 7 9 { \scriptstyle \pm 8 . 1 9 }$ </td><td>0.58±1.31</td><td>117.48±455.80</td><td>0.34±0.20</td><td>2.77±6.56</td><td> $1 . 2 3 { \scriptstyle \pm 3 . 2 0 }$ </td><td> $9 4 . 8 0 \substack { \pm \infty 2 . 1 7 }$ </td><td> $5 . 0 8 { \scriptstyle \pm 3 . 9 4 }$ </td><td> $1 7 . 7 3 { \scriptstyle \pm 1 6 6 4 }$ </td></tr><tr><td>MatchAnything + Koide et al. []</td><td> $0 . 3 9 { \scriptstyle \pm 1 , 3 3 }$ </td><td> $4 . 4 0 { \scriptstyle \pm 1 2 4 1 }$ </td><td> $1 . 2 1 \pm 3 . 3 0$ </td><td> $2 5 . 6 8 \substack { \pm 8 1 . 5 6 }$ </td><td>0.40±1.52 0.18±0.45</td><td>87.48±258.70</td><td>0.39±0.14 0.31±0.17</td><td>10.80±6.77 4.02±4.82</td><td> $0 . 3 4 \pm 1 . 6 5$   $0 . 2 4 { \scriptstyle \pm 1 . 6 6 }$ </td><td> $3 3 . 7 9 \substack { \pm 1 8 6 . 6 1 }$ </td><td> $5 . 3 0 { \scriptstyle \pm 4 . 2 3 }$   $1 . 3 9 { \scriptstyle \pm 0 . 7 2 }$ </td><td>26.55±19.88 3.39±1.63</td></tr><tr><td>MatchAnything + Taylor et al. []</td><td> $0 . 1 3 { \scriptstyle \pm 0 . 1 5 }$ </td><td>1.38±1.46</td><td> $0 . 1 2 \substack { \pm 0 . 0 8 }$ </td><td>2.22±2.76</td><td>0.19±0.17</td><td>44.04±72.88 36.51±56.88</td><td></td><td> $4 . 2 2 \substack { \pm 4 . 5 9 }$ </td><td>0.16±0.13</td><td> $1 7 . 3 1 { \scriptstyle \pm 2 4 1 . 2 6 }$   $1 9 . 3 4 \div 1 6 . 0 4$ </td><td> $0 . 9 9 { \scriptstyle \pm 0 . 5 4 }$ </td><td> $2 . 7 4 \scriptstyle \pm 1 . 4 9$ </td></tr><tr><td>MatchAnything + MatchAnything [] MatchAnything* + ReFineR</td><td> $0 . 1 1 { \scriptstyle \pm 0 . 0 8 }$   $0 . 0 6 { \scriptstyle \pm 0 . 0 5 }$ </td><td>1.03±1.65 0.70±0.62</td><td>0.10±0.08 0.06±0.04</td><td>2.80±2.38 1.24±0.65</td><td>0.06±0.09</td><td>3.05±6.39</td><td>0.31±0.09 0.38±0.09</td><td> $5 . 8 1 { \scriptstyle \pm 3 . 2 5 }$ </td><td>0.13±0.14</td><td>4.92±4.80</td><td> $1 . 0 9 { \scriptstyle \pm 0 . 8 9 }$ </td><td>2.78±2.14</td></tr></table>

Table 7: Fine image to raw point cloud registration results (Section 5.2). Each image was registered using rendered point cloud normals.

<table><tr><td></td><td colspan="2">ETH3D</td><td colspan="2">Tanks and Temples</td><td colspan="2">Livox</td><td colspan="2">nuScenes</td><td colspan="2">RGBDScenes</td></tr><tr><td></td><td>rot (°↓) transl (cm↓)</td><td></td><td>rot (°↓) transl (cm↓)</td><td></td><td></td><td>rot (°↓) transl (cm↓)</td><td></td><td>rot (°↓) trans1 (cm↓)</td><td>rot (°↓)</td><td>transl (cm↓)</td></tr><tr><td>One stage</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GS-SLAM []</td><td>0.47±1.87</td><td>4.31±20.80</td><td> $0 . 1 0 { \scriptstyle \pm 1 . 3 6 }$ </td><td>1.62±67.28</td><td>2.57±2.03</td><td>74.66±39.92</td><td>4.37±2.66</td><td>131.13±57.91</td><td>2.81±2.22</td><td> $1 7 . 6 1 { \scriptstyle \pm 7 . 1 3 }$ </td></tr><tr><td>Koide et al. []</td><td>0.05±0.03</td><td></td><td>0.38±0.56 0.06±0.04</td><td>1.82±0.73</td><td>0.26±0.09</td><td>8.13±3.55</td><td>0.48±0.53</td><td>30.81±61.62</td><td>0.77±3.38</td><td>2.06±11.74</td></tr><tr><td>Taylor et al. []</td><td> $0 . 5 5 { \scriptstyle \pm 0 . 5 3 }$ </td><td> $6 . 0 3 { \scriptstyle \pm 1 0 . 2 0 }$ </td><td> $0 . 3 3 { \scriptstyle \pm 0 . 4 3 }$ </td><td> $6 . 3 0 { \scriptstyle \pm 1 4 . 7 2 }$ </td><td>0.93±0.29</td><td>9.58±7.64</td><td> $0 . 9 2 { \scriptstyle \pm 1 . 7 4 }$ </td><td>50.40±183.95</td><td>2.13±0.82</td><td>5.92±2.40</td></tr><tr><td>MatchAnything [</td><td> $0 . 0 7 _ { \pm 0 . 1 0 }$ </td><td> $0 . 6 6 \substack { \pm 0 . 7 9 }$ </td><td> $0 . 0 7 { \scriptstyle \pm 0 . 0 5 }$ </td><td> $1 . 6 4 \pm 1 . 5 2$ </td><td>0.26±0.15</td><td>3.14±5.77</td><td>0.11±0.06</td><td>11.24±10.67</td><td>0.40±0.24</td><td>0.85±0.64</td></tr><tr><td>ReFineR</td><td> $0 . 0 8 { \scriptstyle \pm 0 . 5 0 }$ </td><td>0.57±7.09 0.07±0.10</td><td></td><td>1.20±1.56</td><td> $\overline { { 0 . 2 8 { \scriptstyle \pm 0 . 1 6 } } }$ </td><td>4.83±6.31</td><td>0.23±0.28</td><td>7.43±9.85</td><td>2.12±1.19</td><td>5.77±2.96</td></tr><tr><td>Two stages</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>MatchAnything + GS-SLAM []</td><td> $0 . 1 3 { \scriptstyle \pm 2 0 0 }$ </td><td>1.15±18.81</td><td> $0 . 0 8 { \scriptstyle \pm 1 . 7 4 }$ </td><td>1.51±24.68</td><td> $3 . 0 0 { \scriptstyle \pm 2 . 0 9 }$ </td><td>82.11±35.87</td><td> $4 . 9 1 { \scriptstyle \pm 3 . 0 9 }$ </td><td>141.23±68.28</td><td> $2 . 7 3 { \scriptstyle \pm 3 . 6 7 }$ </td><td>16.21±8.91</td></tr><tr><td>MatchAnything + Koide et al. []</td><td> $0 . 0 5 { \scriptstyle \pm 0 . 0 3 }$ </td><td>0.37±0.54</td><td> $0 . 0 6 { \scriptstyle \pm 0 . 0 4 }$ </td><td>1.76±0.75</td><td> $0 . 2 5 { \scriptstyle \pm 0 . 0 9 }$ </td><td>10.38±4.60</td><td> $0 . 3 4 { \scriptstyle \pm 0 . 5 2 }$ </td><td>32.70±96.20</td><td> $0 . 6 8 { \scriptstyle \pm 3 . 5 6 }$ </td><td>1.75±15.71</td></tr><tr><td>MatchAnything + Taylor et al. []</td><td> $0 . 0 7 { \scriptstyle \pm 0 . 1 1 }$ </td><td> $0 . 6 6 \pm 0 . 8 5$ </td><td> $0 . 0 8 { \scriptstyle \pm 0 . 0 6 }$ </td><td> $1 . 6 3 { \scriptstyle \pm 1 . 3 7 }$ </td><td> $0 . 2 8 \substack { \pm 0 . 1 4 }$ </td><td>2.52±6.02</td><td> $0 . 1 6 { \scriptstyle \pm 1 . 0 0 }$ </td><td>18.14±55.96</td><td> $0 . 4 5 { \scriptstyle \pm 0 . 3 7 }$ </td><td>1.38±1.06</td></tr><tr><td>MatchAnything + MatchAnything []</td><td> $0 . 0 5 { \scriptstyle \pm 0 . 0 5 }$ </td><td> $0 . 4 9 { \scriptstyle \pm 0 . 6 6 }$ </td><td>0.07±0.05</td><td>1.51±1.03</td><td>0.24±0.16</td><td>3.85±5.85</td><td>0.12±0.06</td><td>10.63±9.80</td><td>0.35±0.16</td><td>0.88±0.46</td></tr><tr><td>MatchAnything* + ReFineR</td><td>0.04±0.04</td><td>0.40±0.60</td><td>0.06±0.04</td><td>1.17±0.65</td><td>0.22±0.16</td><td>3.55±6.21</td><td>0.11±0.08</td><td>3.88±3.02</td><td>0.24±0.13</td><td>0.50±0.62</td></tr></table>

Table 8: Fine image to intensity point cloud registration results (Section 5.3). Each image was registered using rendered point cloud reflectivity or grayscale value.

<table><tr><td></td><td colspan="3">ETH3D</td><td colspan="3">Tanks and Temples</td><td colspan="3">CARLA</td><td colspan="3">Livox</td><td colspan="3">nuScenes</td><td colspan="3">RGBDScenes</td></tr><tr><td></td><td>pix@ (%↑) 2</td><td>5</td><td> $p _ { e } ( \mathrm { p x } \downarrow )$ </td><td>1</td><td> $\operatorname { p i x } ( \underline { { a } } ) \ ( \mathcal { R } \uparrow )$  2 3</td><td>5</td><td>pe(px↓)</td><td>pix@ (%↑) 1 2</td><td>5</td><td>pe(px↓)</td><td>pix@ (%↑) 5 10</td><td>Pe(px↓)</td><td>3</td><td>pix@(%↑) 5 10</td><td>pe(px↓)</td><td>3</td><td>pix@ (%↑) 5 10</td><td>pe(px↓)</td></tr><tr><td>(VII)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Rendered depth</td><td>39 93</td><td>100</td><td>1.04±0.55</td><td>0</td><td>58</td><td>92 100</td><td>1.95±0.76</td><td>8</td><td>36 78</td><td>2.45±7.23</td><td>25</td><td>95</td><td>6.48±2.32 3</td><td>36 92</td><td></td><td>5.66±8.57</td><td>6 22</td><td>80</td><td>6.61±4.58</td></tr><tr><td>Monocular normals</td><td>0 2</td><td>9</td><td>8.34±226</td><td>0</td><td>28 90</td><td>100</td><td>2.43±0.58</td><td>2 23</td><td>71</td><td>3.29±6.49</td><td>65 100</td><td>4.94±0.27</td><td>0</td><td>12 71</td><td>7.15±13.75</td><td></td><td>18 57</td><td>94</td><td>4.47±2.83</td></tr><tr><td>Monocular depth</td><td>0 0</td><td>24</td><td>7.20±5.17</td><td>0</td><td>5</td><td>32 83</td><td>3.62±2.67</td><td>2</td><td>7 41</td><td>5.76±559.08</td><td>5</td><td>60</td><td>9.77±2.87 0</td><td>2 37</td><td>11.88±34.59</td><td></td><td>5 61</td><td>90</td><td>4.59±3.03</td></tr><tr><td>MatchAnything* + ReFineR (RGB image)</td><td>58 96</td><td>100</td><td></td><td>0.90±0.40 0</td><td>70</td><td>95 100</td><td>1.76±0.51</td><td>16</td><td>68 91</td><td>1.54±3.04</td><td>55</td><td>100</td><td>4.94±0.94 6</td><td>54</td><td>97</td><td>4.90±2.75</td><td>13 43</td><td>92</td><td>5.36±4.25</td></tr></table>

Table 9: Ablation study on ReFineR’s supervision channel (Section C.4).

![](images/fb045f9c228c5f1cd0d153d3c476b3e2e8b77c32a891b1f59f24fb7e063de40c.jpg)  
Figure 7: Qualitative results from experiment Section 5.2.

![](images/4b9909353e255c12738d14bff373524b3a98f35642523f1cfd0cf2221076bbc4.jpg)  
Figure 8: Qualitative results from experiment Section 5.3.

<table><tr><td></td><td colspan="10">ETH3D</td></tr><tr><td></td><td colspan="5">pix@ (%↑)</td><td colspan="3"></td><td> $p _ { e } ( \mathfrak { p x } \downarrow )$   $\cot { ( \sqrt [ ] { \sqrt { 3 } } } )$ </td><td></td><td>transl (cm↓)</td></tr><tr><td></td><td>1</td><td>2</td><td>3</td><td>5</td><td>10</td><td>20</td><td>50</td><td>100</td><td></td><td></td><td></td></tr><tr><td>0px initialization</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>MatchAnything</td><td>14</td><td>67</td><td>98</td><td>100</td><td>100</td><td>100</td><td>100</td><td>100</td><td> $1 . 6 6 \substack { \pm 0 . 6 6 }$ </td><td> $0 . 1 0 { \scriptstyle \pm 0 . 0 5 }$ </td><td> $0 . 9 4 { \scriptstyle \pm 1 . 5 5 }$ </td></tr><tr><td>ReFineR</td><td>64</td><td>100</td><td>100</td><td>100</td><td>100</td><td>100</td><td>100</td><td>100</td><td> $0 . 8 9 { \scriptstyle \pm 0 . 3 5 }$ </td><td> $0 . 0 6 { \scriptstyle \pm 0 . 0 4 }$ </td><td> $0 . 7 0 { \scriptstyle \pm 0 . 6 2 }$ </td></tr><tr><td>MatchAnything* + ReFineR</td><td>65</td><td>97</td><td>100</td><td>100</td><td>100</td><td>100</td><td>100</td><td>100</td><td> $0 . 8 7 { \scriptstyle \pm 0 . 3 8 }$ </td><td>0.06±0.05</td><td>0.68±0.74</td></tr><tr><td>5px initialization</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>MatchAnything</td><td>8</td><td>62</td><td>91</td><td>100</td><td>100</td><td>100</td><td>100</td><td>100</td><td> $1 . 7 2 { \scriptstyle \pm 0 . 8 4 }$ </td><td> $0 . 1 3 { \scriptstyle \pm 0 . 0 9 }$ </td><td> $1 . 2 0 { \scriptstyle \pm 1 . 7 2 }$ </td></tr><tr><td>ReFineR</td><td>52</td><td>92</td><td>96</td><td>98</td><td>100</td><td>100</td><td>100</td><td>100</td><td> $0 . 9 5 { \scriptstyle \pm 0 . 8 6 }$ </td><td> $0 . 0 7 { \scriptstyle \pm 0 . 1 2 }$ </td><td> $0 . 7 5 { \scriptstyle \pm 1 . 9 5 }$ </td></tr><tr><td>MatchAnything* + ReFineR</td><td>58</td><td>97</td><td>100</td><td>100</td><td>100</td><td>100</td><td>100</td><td>100</td><td> $0 . 8 7 { \scriptstyle \pm 0 . 3 8 }$ </td><td> $0 . 0 6 { \scriptstyle \pm 0 . 0 5 }$ </td><td> $0 . 6 9 { \scriptstyle \pm 0 . 6 6 }$ </td></tr><tr><td>10px initialization</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>MatchAnything</td><td>8</td><td>65</td><td>90</td><td>100</td><td>100</td><td>100</td><td>100</td><td>100</td><td> $1 . 8 0 { \scriptstyle \pm 0 . 8 3 }$ </td><td> $0 . 1 3 { \scriptstyle \pm 0 . 1 1 }$ </td><td> $1 . 3 2 { \scriptstyle \pm 1 . 6 2 }$ </td></tr><tr><td>ReFineR</td><td>41</td><td>73</td><td>78</td><td>79</td><td>96</td><td>100</td><td>100</td><td>100</td><td> $1 . 1 0 { \scriptstyle \pm 3 . 1 7 }$ </td><td>0.10±0.36</td><td> $1 . 0 2 \pm 4 . 6 9$ </td></tr><tr><td>MatchAnything* + ReFineR</td><td>58</td><td>96</td><td>100</td><td>100</td><td>100</td><td>100</td><td>100</td><td>100</td><td> $0 . 9 0 { \scriptstyle \pm 0 . 4 0 }$ </td><td> $0 . 0 6 { \scriptstyle \pm 0 . 0 5 }$ </td><td> $0 . 7 0 { \scriptstyle \pm 0 . 6 2 }$ </td></tr><tr><td>20px initialization</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>MatchAnything</td><td>7</td><td>52</td><td>87</td><td>96</td><td>100</td><td>100</td><td>100</td><td>100</td><td> $1 . 9 2 \pm 1 . 1 6$ </td><td> $0 . 1 5 { \scriptstyle \pm 0 . 1 1 }$ </td><td> $1 . 2 8 { \scriptstyle \pm 1 . 8 2 }$ </td></tr><tr><td>ReFineR</td><td>20</td><td>35</td><td>39</td><td>41</td><td>48</td><td>87</td><td>100</td><td>100</td><td> $1 0 . 7 4 \substack { \pm 8 . 1 1 }$ </td><td> $0 . 7 9 { \scriptstyle \pm 1 . 1 0 }$ </td><td> $1 0 . 2 1 \pm 1 4 . 0 1$ </td></tr><tr><td>MatchAnything  $\ u _ { \mathrm { p } } ^ { \ast } + \mathrm { R e F i n e R }$ </td><td>60</td><td>96</td><td>100</td><td>100</td><td>100</td><td>100</td><td>100</td><td>100</td><td> $0 . 8 9 { \scriptstyle \pm 0 . 3 9 }$ </td><td> $0 . 0 6 { \scriptstyle \pm 0 . 0 5 }$ </td><td> $0 . 7 0 { \scriptstyle \pm 0 . 6 7 }$ </td></tr><tr><td>50px initialization</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>MatchAnything</td><td>10</td><td>54</td><td>77</td><td>96</td><td>97</td><td>100</td><td>100</td><td>100</td><td> $1 . 9 6 { \scriptstyle \pm 2 . 1 7 }$ </td><td> $0 . 1 3 { \scriptstyle \pm 0 . 1 5 }$ </td><td> $1 . 4 3 { \scriptstyle \pm 1 . 8 5 }$ </td></tr><tr><td>ReFineR</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>2</td><td>60</td><td>100</td><td> $4 7 . 8 5 _ { \pm 9 . 0 6 }$ </td><td> $4 . 9 0 { \scriptstyle \pm 2 . 2 3 }$ </td><td> $4 1 . 2 6 { \scriptstyle \pm 4 5 . 4 4 }$ </td></tr><tr><td>MatchAnythin  $\mathbf { g } ^ { * } + \mathbf { R e } \mathrm { F i n e R }$ </td><td>58</td><td>97</td><td>100</td><td>100</td><td>100</td><td>100</td><td>100</td><td>100</td><td> $0 . 9 0 { \scriptstyle \pm 0 . 4 0 }$ </td><td> $0 . 0 6 { \scriptstyle \pm 0 . 0 5 }$ </td><td> $0 . 7 0 z 0 . 6 5$ </td></tr><tr><td>100px initialization</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>MatchAnything</td><td>3</td><td>39</td><td>71</td><td>88</td><td>97</td><td>100</td><td>100</td><td>100</td><td> $2 . 1 4 \substack { \pm 2 . 3 7 }$ </td><td> $0 . 1 4 { \scriptstyle \pm 0 . 2 5 }$ </td><td> $1 . 9 4 \pm 2 . 4 7$ </td></tr><tr><td>ReFineR</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>36</td><td> $1 0 7 . 2 0 { \scriptstyle \pm 1 7 . 0 5 }$ </td><td> $1 0 . 1 7 { \scriptstyle \pm 4 . 3 6 }$ </td><td> $8 4 . 2 9 { \scriptstyle \pm 9 2 . 1 7 }$ </td></tr><tr><td>MatchAnything* + ReFineR</td><td>60</td><td>97</td><td>100</td><td>100</td><td>100</td><td>100</td><td>100</td><td>100</td><td> $0 . 8 7 _ { \pm 0 . 4 1 }$ </td><td> $0 . 0 6 { \scriptstyle \pm 0 . 0 5 }$ </td><td> $0 . 7 0 z 0 . 7 0$ </td></tr></table>

Table 10: Registration at scale experiment, each camera pose was initialized at increasing pixel errors (Section C.5).
<table><tr><td rowspan="2"></td><td>CARLA (normals)</td><td>ETH3D (intensity)</td></tr><tr><td>Time (s)</td><td>Time (s)</td></tr><tr><td>GS-SLAM []</td><td>36.13</td><td>9.68</td></tr><tr><td>Render</td><td>11.31</td><td>3.48</td></tr><tr><td>Loss</td><td>0.12</td><td>0.62</td></tr><tr><td>Backward</td><td>22.24</td><td>4.62</td></tr><tr><td>Yuan et al. []</td><td>47.03</td><td>N/A</td></tr><tr><td>Koide et al. []</td><td>10.89</td><td>26.80</td></tr><tr><td>Taylor et al. []</td><td>80.01</td><td>87.79</td></tr><tr><td>MatchAnything []</td><td>4.84</td><td>4.90</td></tr><tr><td>Weight Loading</td><td>3.12</td><td>3.11</td></tr><tr><td>Inference</td><td>1.69</td><td>1.75</td></tr><tr><td>RANSAC</td><td>0.02</td><td>0.03</td></tr><tr><td>ReFineR</td><td>11.62</td><td>6.83</td></tr><tr><td>Render</td><td>3.69</td><td>2.89</td></tr><tr><td>Loss</td><td>0.08</td><td>0.10</td></tr><tr><td>Backward</td><td>7.31</td><td>3.14</td></tr><tr><td>MatchAnything* + ReFineR</td><td>13.81</td><td>6.97</td></tr></table>

Table 11: Detailed computation times for the benchmarking methods on the CARLA and ETH3D sets.

<table><tr><td></td><td>time / iter (s)</td><td>Num Points (M)</td></tr><tr><td>ETH3D</td><td> $0 . 0 3 { \scriptstyle \pm 0 . 0 2 }$ </td><td> $3 . 2 1 { \scriptstyle \pm 2 . 8 2 }$ </td></tr><tr><td>Tanks and Temples</td><td> $0 . 1 4 { \scriptstyle \pm 0 . 0 7 }$ </td><td> $1 9 . 9 2 { \scriptstyle \pm 1 1 . 4 0 }$ </td></tr><tr><td>CARLA</td><td> $0 . 0 8 { \scriptstyle \pm 0 . 0 5 }$ </td><td> $3 . 7 4 \pm 2 . 7 2$ </td></tr><tr><td>Livox</td><td> $0 . 0 7 { \scriptstyle \pm 0 . 0 2 }$ </td><td> $5 . 0 1 { \scriptstyle \pm 1 . 6 8 }$ </td></tr><tr><td>nuScenes</td><td> $0 . 0 2 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $1 . 0 2 \substack { \pm 0 . 3 5 }$ </td></tr><tr><td>RGBDScenes</td><td> $0 . 0 3 { \scriptstyle \pm 0 . 0 1 }$ </td><td> $0 . 6 5 { \scriptstyle \pm 0 . 1 7 }$ </td></tr></table>

Table 12: Average time per ReFineR iteration and average number of rendered points for each set.

<table><tr><td></td><td colspan="7">ETH3D</td><td colspan="7">CARLA</td></tr><tr><td></td><td> $\mathrm { p i x } @ \ ( \% \uparrow )$ </td><td colspan="3"></td><td></td><td> $p _ { e } ( \mathfrak { p x } \downarrow )$ </td><td> $\mathrm { t i m e } \left( \mathrm { s } \right)$ </td><td colspan="5"> $\mathrm { p i x } @ \ ( \% \uparrow )$ </td><td> $p _ { e } ( \mathfrak { p x } \downarrow )$ </td><td>time (s)</td></tr><tr><td></td><td>1</td><td>2</td><td>3</td><td>5</td><td>10</td><td></td><td></td><td>1</td><td>2</td><td>3</td><td>5</td><td>10</td><td></td><td></td></tr><tr><td>ReFineR  $( \mathrm { s } { = } 1 0 0 \% )$ </td><td>41</td><td>73</td><td>78</td><td>79</td><td>96</td><td> $1 . 1 0 { \scriptstyle \pm 3 . 1 7 }$ </td><td> $6 . 8 3 { \scriptstyle \pm 5 . 0 3 }$ </td><td>12</td><td>44</td><td>57</td><td>74</td><td>92</td><td> $2 . 3 6 \pm 3 . 4 5$ </td><td> $1 1 . 6 2 \substack { \pm 7 . 9 8 }$ </td></tr><tr><td>ReFineR  $( \mathrm { s } { = } 7 5 \% )$ </td><td>31</td><td>64</td><td>68</td><td>73</td><td>96</td><td> $1 . 3 5 { \scriptstyle \pm 3 . 2 6 }$ </td><td> $6 . 7 3 { \scriptstyle \pm 4 . 1 5 }$ </td><td>2</td><td>25</td><td>48</td><td>60</td><td>92</td><td> $3 . 5 8 { \scriptstyle \pm 3 . 3 6 }$ </td><td> $1 2 . 3 8 \pm 1 0 . 8 5$ </td></tr><tr><td>ReFineR  $( \mathrm { s } { = } 5 0 \% )$ </td><td>29</td><td>58</td><td>61</td><td>69</td><td>92</td><td> $1 . 4 5 { \scriptstyle \pm 3 . 5 2 }$ </td><td> $5 . 3 5 { \scriptstyle \pm 3 . 0 9 }$ </td><td>2</td><td>22</td><td>36</td><td>54</td><td>88</td><td> $4 . 2 7 { \scriptstyle \pm 4 . 0 1 }$ </td><td> $9 . 6 3 { \scriptstyle \pm 8 . 0 6 }$ </td></tr><tr><td>ReFineR  $( \mathrm { s } { = } 2 5 \% )$ </td><td>19</td><td>43</td><td>46</td><td>51</td><td>92</td><td> $4 . 2 0 { \scriptstyle \pm 3 . 7 1 }$ </td><td> $4 . 0 0 { \scriptstyle \pm 2 . 1 7 }$ </td><td>0</td><td>8</td><td>29</td><td>48</td><td>80</td><td> $5 . 1 7 { \scriptstyle \pm 3 . 9 2 }$ </td><td> $5 . 9 7 { \scriptstyle \pm 4 . 0 5 }$ </td></tr><tr><td>ReFineR  $( \mathrm { s } { = } 1 0 \% )$ </td><td>8</td><td>34</td><td>42</td><td>50</td><td>90</td><td> $4 . 9 4 \pm 3 . 7 5$ </td><td> $3 . 1 8 { \scriptstyle \pm 1 . 5 7 }$ </td><td>0</td><td>0</td><td>9</td><td>26</td><td>71</td><td> $7 . 5 2 { \scriptstyle \pm 3 . 6 6 }$ </td><td> $3 . 7 7 { \scriptstyle \pm 1 . 9 8 }$ </td></tr><tr><td>ReFineR (s=5%)</td><td>4</td><td>14</td><td>20</td><td>24</td><td>78</td><td> $8 . 0 5 { \scriptstyle \pm 3 . 6 0 }$ </td><td> $2 . 6 3 { \scriptstyle \pm 1 . 5 2 }$ </td><td>0</td><td>0</td><td>0</td><td>15</td><td>69</td><td> $9 . 2 3 { \scriptstyle \pm 3 . 0 6 }$ </td><td> $2 . 8 6 { \scriptstyle \pm 1 . 5 6 }$ </td></tr></table>

Table 13: Downsampling experiment with raw point clouds. Each registration was initially perturbed by 10px and carried out with random downsampling of the point cloud by a varying amount.

![](images/76bbaf28eac1b009cf52d99232916e467592709f6dce54ff5000a01e335e205a.jpg)  
Figure 9: Average impact of camera pose error on projected 3D semantic labels.

![](images/01a779fd028721b376cc748458c1874bf5b2603582e7203dd768152951637692.jpg)  
Figure 10: Left to right: UniCorrn inference on a single RGBD frame (accurate predictions), on densely aggregated scans, and on synthetic depth from the dense point cloud (large errors).