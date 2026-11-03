# OctreeSplat: Hierarchical Octree-Guided Feed-Forward Gaussian Splatting

Yibin Zhao<sup>\*</sup>, Yihan Pan<sup>\*</sup>, Wenli Yang, Jun Nan, and Jianjun Yi<sup>†</sup>

![](images/321c6784824372f57c06b3337be825a7a3390dc4ee3e075495912d3ba4771009.jpg)  
Fig. 1. Given a variable number of images and optional camera parameters, OctreeSplat reconstructs a geometrically accurate 3DGS scene through feedforward inference. Predicted densification scores guide the spatial allocation of Gaussian primitives. Compared with pixel-aligned [1] and voxel-aligned [2] methods, OctreeSplat improves RGB synthesis, depth synthesis, and surface reconstruction while using fewer primitives. Optional camera-conditioned refinement is performed through test-time optimization.

Abstract—3D Gaussian Splatting (3DGS) has shown considerable potential for novel view synthesis (NVS) and 3D reconstruction. Feed-forward methods avoid costly per-scene optimization, but fixed image or spatial resolutions limit their ability to allocate primitives according to local complexity. We present OctreeSplat, a hierarchical octree-guided feed-forward Gaussian splatting. OctreeSplat extracts multi-view features and geometric priors from a frozen visual foundation model (VFM), constructs a hierarchical octree, and predicts densification scores. The resulting representation allocates Gaussian primitives at spatially varying densities according to local complexity. Stochastic solids volume rendering improves depth rendering, strengthens geometric supervision, and supports surface reconstruction. When camera parameters are available, an optional test-time optimization (TTO) module improves agreement between predicted and ground-truth parameters. Training updates only 76 million additional parameters and requires 35 GPU-hours. Experiments on Replica, TUM, Tanks and Temples, and DTU demonstrate competitive RGB view synthesis, depth synthesis, and surface reconstruction. OctreeSplat reduces the number of Gaussian primitives while maintaining high reconstruction quality.

Index Terms—3D Gaussian splatting, visual foundation model, hierarchical octree, novel view synthesis.

## I. INTRODUCTION

Image-based 3D reconstruction has traditionally relied on structure-from-motion (SfM) pipelines, with iterative techniques represented by bundle adjustment (BA) [3] for multiview pose estimation and sparse reconstruction. Common scene representations include point clouds, voxels, and signed distance fields. More recently, 3D Gaussian Splatting (3DGS) [4] has combined photorealistic novel view synthesis with high rendering efficiency. Given images and camera parameters, 3DGS iteratively renders the scene and optimizes the parameters of Gaussian primitives. Adaptive density control (ADC) periodically adds and removes primitives during optimization, which typically takes tens of minutes to an hour.

Visual foundation models (VFMs) [5]–[7] learn strong priors from large-scale datasets and directly regress 3D information from image collections. Outputs include depth maps, 3D point maps, and camera parameters. These models reduce accumulated errors and iterative computation in conventional reconstruction pipelines while maintaining strong generalization. Feed-forward 3DGS methods [1], [2], [8] similarly predict Gaussian scenes in a single feed-forward pass. However, commonly used pixel-aligned representations [8], [9] can introduce overlapping primitives and floating artifacts. Voxelaligned methods [2], [10] and methods that regress primitives from fixed token sets [11], [12] produce more compact representations. Nevertheless, their primitive allocation does not explicitly adapt to local scene complexity, limiting density flexibility and geometric accuracy.

To address the above problems, we propose OctreeSplat, a hierarchical octree-guided feed-forward Gaussian splatting. Given uncalibrated images and optional camera parameters, OctreeSplat predicts a geometrically accurate 3DGS scene with spatially adaptive primitive allocation. Depth maps and camera parameters predicted by a VFM backbone support voxel aggregation at multiple resolutions. A dense prediction transformer (DPT) upsamples the backbone’s patch tokens to construct aligned multi-resolution voxel features. A sparseconvolutional U-Net (SPUNet) processes these features and predicts Gaussian parameters at each level, together with densification scores between adjacent levels. During inference, subdivision starts from the coarsest voxels and selects a fixed fraction for refinement according to the predicted scores. Gaussian primitives associated with the selected octree nodes form the final octree-aligned scene.

We initialize the backbone with pretrained VFM weights and introduce an optional test-time optimization (TTO) module. When camera parameters are supplied, TTO improves alignment between the predicted parameters and the groundtruth. Stochastic-solid volume rendering provides more accurate geometric supervision by representing Gaussian primitives as equivalent stochastic solids and volume rendering along viewing rays. This formulation improves depth rendering and geometric accuracy while supporting surface reconstruction. The backbone remains frozen during training. Only the feature-upsampling DPT, SPUNet, densification-score heads, and Gaussian prediction heads are optimized, totaling 76 million parameters. This design allows OctreeSplat to be trained in 35 GPU-hours on RTX PRO 6000.

Our main contributions are summarized as follows:

• We introduce OctreeSplat, a hierarchical octree-guided method that reconstructs geometrically accurate Gaussian scenes from uncalibrated images and optional camera parameters.

• Predicted densification scores guide adaptive octree subdivision and allocate Gaussian primitives according to local scene complexity.

• A frozen VFM backbone limits training to 76 million additional parameters and 35 GPU-hours. Optional TTO aligns camera predictions with ground-truth without updating backbone weights.

• Extensive experiments demonstrate competitive RGB view synthesis and geometric reconstruction across multiple datasets and settings.

## II. RELATED WORK

## A. 3D Scene Representations

Conventional reconstruction methods represent scenes using point clouds [13], voxels [14], signed distance fields (SDFs) [15], or triangle meshes [16]. Neural radiance fields (NeRFs) [17] instead use multilayer perceptrons (MLPs) to represent scenes as continuous implicit radiance fields, which map 3D positions and viewing directions to volume densities and colors. Differentiable image rendering optimizes radiance fields’ parameters to produce photorealistic NVS. Subsequent work has improved reconstruction speed [18], [19], rendering quality [20], [21], and surface reconstruction [22], [23]. Despite NeRF’s reconstruction accuracy, implicit representations require repeated field inferences along each ray during volume rendering. This computational cost limits rendering efficiency and transferability.

3DGS [4] addresses this bottleneck by explicitly storing scenes as 3D Gaussian primitives. These primitives are initialized from sparse point clouds, typically obtained through SfM [13], and optimized by iteratively rendering the input images. ADC splits, clones, and prunes primitives using photometric gradients [4] or alternative criteria [24], [25]. The optimization process typically takes tens of minutes to an hour. Later methods accelerate optimization [26]–[28] by controlling the primitive count. However, they still require precomputed camera poses and sparse point clouds, and optimization generally takes at least several minutes. ADC strategies have also received sustained attention [29], [30].

Vanilla 3DGS primarily optimizes RGB rendering and lacks explicit surface supervision. Subsequent studies improve geometric accuracy through surface-alignment regularization [31]. Depth-distortion, normal-consistency, and multi-viewconsistency constraints further reduce geometric ambiguities arising from photometric supervision [32], [33]. GOF [34] constructs an opacity field from Gaussian primitives and extracts its isosurface for surface reconstruction. GGGS [35] and GW [36] further improve surface reconstruction by representing Gaussian primitives as equivalent stochastic solids.

## B. Feed-Forward Reconstruction

Another line of research predicts Gaussian scenes through a single feed-forward based on neural-network. Feed-forward methods such as pixelSplat [37] take images and camera parameters as input. They aggregate multi-view features, estimate per-pixel depth, back-project pixels into world coordinates, and regress Gaussian parameters. Multi-view cost volumes [9] and pretrained depth priors [38] improve reconstruction quality in challenging scenes. VolSplat [10] replaces pixel alignment with voxel alignment, reducing the dependence of primitives on individual input views.

VFMs extend reconstruction to uncalibrated images without requiring camera parameters as input. DUSt3R [39] formulates image-pair reconstruction as 3D point-map regression, allowing dense geometry prediction without camera parameters. MUSt3R [40] extends DUSt3R to multiple views and predicts geometry directly in a common coordinate system. VGGT [5] alternates single-view and multi-view self-attention to aggregate information across a variable number of views. A shared backbone and separate prediction heads jointly estimate camera parameters, depth maps, 3D point maps, and point tracks. Subsequent work improves robustness [6], longsequence processing [41], and model scalability [7].

VFMs have also advanced feed-forward 3DGS.Some approaches build directly upon VFM backbones by adding prediction heads for pixel-aligned Gaussian attributes [1], [42]. As more views are added, overlapping observations and the one-to-one pixel-to-primitive correspondence introduce redundant, overlapping primitives that can blur the rendered images. AnySplat [2] uses pretrained VGGT geometry for pseudo-geometric supervision and differentiable voxelization for multi-view aggregation, reducing redundancy and overlap artifacts. Other methods [11], [12] regress Gaussian primitives directly from backbone tokens, removing the one-to-one correspondence with pixels. Nevertheless, increasing the number of input views can still introduce blur, and geometric accuracy remains limited. Some methods have explored Gaussian selfsplitting during the feed-forward process [43]. However, they remain focused on the pixel coordinate system, which leads to the overlapping of primitives.

![](images/9518359cb5a2efa215540b210d571d8d881a94be9bdfce46a4210899fc8d7052.jpg)  
Fig. 2. Overview of the OctreeSplat pipeline. A frozen visual foundation model extracts multi-view features and predicts geometry. DPT and SPUNet construct multi-resolution voxel features and predict Gaussian parameters and densification scores, which guide adaptive octree subdivision to construct an octree-aligned Gaussian representation that adapts to local scene complexity. Stochastic-solid volume rendering provides RGB and geometric supervision.

## III. METHOD

OctreeSplat reconstructs a geometrically accurate 3DGS scene from N uncalibrated images and optional camera parameters, as shown in Figure 2. The network predicts Gaussian primitives at multiple octree levels and uses densification scores to guide subdivision. The hierarchical representation improves geometric accuracy and NVS quality while controlling the total number of Gaussian primitives.

## A. Preliminaries

Given uncalibrated images $\mathcal { T } = \{ I _ { i } \} _ { i = 1 } ^ { N }$ , a VFM uses a large transformer $\mathcal { T } _ { \theta }$ to aggregate multi-view features. Separate prediction heads $\mathcal { H }$ estimate depth maps D<sup>ˆ</sup> , camera parameters ${ \hat { \mathbf { C } } } ,$ and other 3D attributes, as shown in Equation 1.

$$
( \hat { \bf D } , \hat { \bf C } ) = \mathcal { H } \left( \mathcal { T } _ { \theta } ( \mathcal { T } ) \right)\tag{1}
$$

For NVS, VFM-based 3DGS methods further predict a Gaussian scene in a single forward pass, represented as $\mathcal { G } =$ $\{ ( \pmb { \mu } _ { i } , \mathbf { R } _ { i } , \mathbf { s } _ { i } , o _ { i } , \mathbf { h } _ { i } ) \} _ { i = 1 } ^ { M }$ . Each primitive has a position $\pmb { \mu } _ { i }$ rotation $\mathbf { R } _ { i } ,$ , scale ${ \bf s } _ { i } ,$ opacity $o _ { i } ,$ , and spherical-harmonic color coefficients $\mathbf { h } _ { i \cdot }$ Gaussian splatting and α-blending produce novel views as shown in Equation 2, where $\pmb { \mu } _ { i } ^ { \prime }$ and $\Sigma _ { i } ^ { \prime }$ denote the projected pixel coordinates and covariance matrix, respectively.

$$
\begin{array} { l } { { \displaystyle { \hat { \mathbf { C } } } ( \mathbf { p } ) = \sum _ { i = 1 } ^ { M } \left( \prod _ { j = 1 } ^ { i - 1 } ( 1 - \alpha _ { j } ) \right) \alpha _ { i } \mathbf { c } _ { i } } , } \\ { { \displaystyle \alpha _ { i } = o _ { i } \exp \left( - \frac { 1 } { 2 } ( \mathbf { p } - { \boldsymbol \mu } _ { i } ^ { \prime } ) ^ { \top } \sum _ { i } ^ { \prime - 1 } ( \mathbf { p } - { \boldsymbol \mu } _ { i } ^ { \prime } ) \right) } } \end{array}\tag{2}
$$

A single forward pass cannot readily reproduce the iterative splitting and cloning of the original 3DGS [4]. Consequently, many feed-forward methods use pixel-aligned representations with fixed primitive counts. AnySplat [2] introduces differentiable voxelization for voxel-aligned prediction, allowing the primitive count to be controlled through voxel size. However, assigning only one Gaussian to each voxel can reduce representation accuracy [43]. Methods without explicit pixel or voxel alignment [11], [12] remove this correspondence constraint, but do not explicitly adapt primitive density to local scene complexity.

## B. Visual Foundation Model Backbone

OctreeSplat’s VFM backbone follows the DA3 architecture [1]. A pretrained DINOv2 encoder [44] extracts patch tokens $\mathbf { t } _ { \mathrm { p a t c h } , i }$ from each input image $I _ { i } .$ . These tokens are concatenated with learnable camera tokens $\mathbf { t } _ { \mathrm { c a m } }$ and register tokens $\mathbf { t } _ { \mathrm { r e g } } ,$ , then processed by 40 attention layers. The first 12 layers apply single-view self-attention; subsequent layers alternate single-view self-attention with global multi-view selfattention. A DPT decoder [45] upsamples the resulting patch tokens into pixel-aligned depth maps. An MLP decodes the camera tokens into camera parameters, including poses and fields of view.

When camera parameters are available, lightweight lowrank adaptation (LoRA) modules support TTO to align predictions with ground-truth. LoRA matrices are inserted into the alternating attention layers, starting from layer 13, and applied to the query, key, and value projections. Within each attention layer, these projections share a down-projection matrix A and use separate up-projection matrices $B _ { Q } , \ B _ { K }$ , and $B _ { V }$ , as shown in Equation 3.

$$
y = W x + b + { \frac { \alpha } { r } } B A x\tag{3}
$$

Here, $x \in \mathbb { R } ^ { d }$ and $\boldsymbol { y } \in \mathbb { R } ^ { d }$ are the input and output matrixs. The original projection parameters are $\bar { W } \in \mathbb { R } ^ { d \times d }$ and $b \in \mathbb { R } ^ { d }$ The low-rank dimension is $r = 4$ by default, and the scaling factor $\alpha / r$ is set to 4.

The LoRA modules are activated when camera parameters corresponding to the input images are supplied. A TTO loss compares the backbone’s camera predictions with groundtruth. Back-propagation updates only the LoRA parameters for a fixed number of iterations. VGGT-style VFMs typically anchor the world coordinate system to the first frame, while image-based reconstruction retains an inherent scale ambiguity. We therefore align predicted and reference poses before computing the TTO loss. All poses are expressed relative to the first frame, and scale is normalized using the mean distance of the cameras.

After alignment, we compute separate rotation and translation losses. The rotation loss uses the relative rotation between predicted and reference orientations, while the translation loss applies SmoothL1 to the aligned translation vectors, as shown in Equation 4.

$$
\begin{array} { r l } & { { \mathcal { L } _ { R } } = \displaystyle \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \frac { 3 - \mathrm { t r } \left[ { { { \left( \widetilde { \bf { R } } _ { i } ^ { \mathrm { p r e d } } \right) } ^ { \top } } \widetilde { \bf { R } } _ { i } ^ { \mathrm { g t } } } \right] } { 2 } , } \\ & { { \mathcal { L } _ { t } } = \displaystyle \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathrm { S m o o t h L 1 } { \left( \widetilde { \bf { t } } _ { i } ^ { \mathrm { p r e d } } , \widetilde { \bf { t } } _ { i } ^ { \mathrm { g t } } \right) } } \end{array}\tag{4}
$$

Here, $N$ is the number of input frames. The matrices $\widetilde { \bf R } _ { i } ^ { \mathrm { p r e d } } , \widetilde { \bf R } _ { i } ^ { \mathrm { g t } } \in \mathrm { S O } ( 3 )$ denote the aligned predicted and groundtruth rotations for frame i. The vectors $\widetilde { \mathbf { t } } _ { i } ^ { \mathrm { p r e d } } , \widetilde { \mathbf { t } } _ { i } ^ { \mathrm { g t } } \ \in \ \mathbb { R } ^ { 3 }$ denote the corresponding translations after scale alignment. We additionally supervise camera intrinsics. For frame i, the intrinsic-parameter vector is $k _ { i } ~ = ~ ( f _ { x , i } , f _ { y , i } , c _ { x , i } , c _ { y , i } )$ . An L1 loss compares the predicted and ground-truth intrinsic parameters:

$$
\mathcal { L } _ { K } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \left. k _ { i } ^ { \mathrm { p r e d } } - k _ { i } ^ { \mathrm { g t } } \right. _ { 1 }\tag{5}
$$

The complete TTO objective is $\mathcal { L } _ { \mathrm { T T O } } = \lambda _ { p } \left( \mathcal { L } _ { R } + \beta _ { t } \mathcal { L } _ { t } \right) +$ $\lambda _ { k } \mathcal { L } _ { K }$

As shown in Figure 3, TTO improves the accuracy of camera parameters predicted by the VFM backbone.

## C. Hierarchical Voxel Learning

Given the VFM’s depth maps D<sup>ˆ</sup> , patch tokens $\{ \mathbf { t } _ { \mathrm { p a t c h } , i } \} _ { i = 1 } ^ { N }$ and camera parameters C<sup>ˆ</sup> , we construct hierarchical voxel features. The voxel sets $\{ \mathcal { V } ^ { ( \ell ) } \} _ { \ell = 1 } ^ { L }$ support prediction of Gaussian scenes $\{ \mathcal { G } ^ { ( \ell ) } \} _ { \ell = 1 } ^ { L }$ at the corresponding levels.

DINOv2 patch tokens $\mathbf { t } _ { \mathrm { p a t c h } , i }$ have a spatial resolution downsampled by a factor of 14. To obtain dense pixel features, we introduce a feature DPT $\mathcal { D } _ { \mathrm { f e a t } }$ , that upsamples these tokens into pixel-aligned feature maps, as shown in Equation 6.

![](images/f9ab4ef00172c50c8006480628544562555fbe24bca5213c25ca77a4ae72015e.jpg)  
Fig. 3. Improvement in pose estimation accuracy with TTO.

$$
\mathbf { F } _ { i } = \mathcal { D } _ { \mathrm { f e a t } } ( \mathbf { t } _ { \mathrm { p a t c h } , i } )\tag{6}
$$

The corresponding depth map $\hat { \bf D } _ { i }$ and camera parameters $\hat { \mathbf { C } } _ { i }$ are used to back-project pixels into a global 3D point cloud $\mathbf { X } _ { i } .$ . Pairing each 3D point with its pixel feature produces the global point cloud feature ${ \mathcal { P } } = \{ \mathbf { X } _ { i } , \mathbf { F } _ { i } \}$

We aggregate $\mathcal { P }$ into global voxel features to obtain a compact representation. L voxel levels are constructed $( L = 3 )$ with voxel size halved at each successive level. All grids share a common origin, establishing the octree hierarchy. For each voxel, we aggregate pixel features and 3D positions using depth-confidence weighted averaging over the included pixels ${ \mathcal { S } } _ { v } ^ { ( { \dot { \ell } } ) }$ . As shown in Equation $7 , c _ { i } ( \mathbf { p } )$ denotes the confidence of the depth estimate. The aggregated 3D position replaces the voxel center to represent more accurate scene geometry.

$$
\left\{ \mathbf { f } _ { v } ^ { ( \ell ) } , \mathbf { x } _ { v } ^ { ( \ell ) } \right\} = \frac { \displaystyle \sum _ { ( i , \mathbf { p } ) \in S _ { v } ^ { ( \ell ) } } c _ { i } ( \mathbf { p } ) \left\{ \mathbf { F } _ { i } ( \mathbf { p } ) , \mathbf { X } _ { i } ( \mathbf { p } ) \right\} } { \displaystyle \sum _ { ( i , \mathbf { p } ) \in S _ { v } ^ { ( \ell ) } } c _ { i } ( \mathbf { p } ) }\tag{7}
$$

A sparse 3D convolutional U-Net further aggregates and enhances the scene features using SpConv [46]. The first L encoder stages correspond to the input voxel levels. At coarser stages, we inject the aggregated features from the matching voxel levels alongside features downsampled from finer levels. The final L decoder stages predict Gaussian scenes $\{ \mathcal { G } ^ { ( \ell ) } \} _ { \ell = 1 } ^ { L }$ whose voxel sizes match the corresponding input levels.

The network also predicts densification scores at the $L - 1$ decoder levels preceding the finest level. At each such level, a separate sparse convolution, MLP, and sigmoid function produce a score $d _ { v } ^ { ( \ell ) } \in [ 0 , 1 ]$ for each voxel $v \in \mathcal { V } ^ { ( \ell ) }$

## D. Octree-Aligned Reconstruction

The predicted Gaussian scenes $\{ \mathcal { G } ^ { ( \ell ) } \} _ { \ell = 1 } ^ { L }$ and densification scores $d _ { v } ^ { ( \ell ) } \in [ 0 , 1 ]$ guide octree construction and primitive allocation. Reconstruction starts from the coarsest Gaussian scene. Previous work [47] measures voxel complexity using the maximum multi-view sampling rate, defined as the largest number of pixels covered by a voxel across views. However, this sampling rate reflects projected coverage rather than local detail. Nearby regions with simple textures can therefore receive high sampling rates, which may not indicate a need for Gaussian densification. We instead predict densification scores that are higher near edges and complex textures, as shown in Figure 4. These scores guide subdivision toward regions requiring finer representations.

![](images/f51b01e6e9cd763b6537a683e2e1d2779271bfd068b5f521ffca36e468f2af94.jpg)

![](images/493ac8ffb350352c3eb07079cfa67a4adf676dffb093812e9ccf14433c3f6229.jpg)  
Max Sample Rate

![](images/537b5a772d8a9a9991a83117c5a4a426a62534efd56f9089c59def1f986a698c.jpg)  
Densification Score  
RGB Pointcloud  
Fig. 4. Comparison of the maximum multi-view sampling rate and predicted densification scores.

At each of the L − 1 non-finest levels, we rank voxels by their predicted scores and subdivide a fixed fraction into eight children. The corresponding finer voxels are retrieved from the next level. Children absent from that level are skipped, preserving spatial sparsity and reducing computational cost. Repeating this procedure (2 subdivision rounds by default) constructs a sparse octree that allocates finer voxels to more complex regions. Each selected octree node is associated with a previously predicted Gaussian primitive at this level. The final scene is thus assembled from a single feed-forward prediction.

## E. Rendering and Training

1) Stochastic-Solid Volume Rendering: Vanilla 3DGS projects Gaussian primitives onto the image plane and uses α-blending for photorealistic RGB synthesis. However, this formulation has limited accuracy for depth synthesis. We adopt stochastic-solid volume rendering [48], representing Gaussian primitives as equivalent spatial stochastic solids and integrating along rays. For the same Gaussian scene, this renderer improves depth accuracy and suppresses floating artifacts and erroneous depths, as shown in Figure 5.

![](images/11836ed466d25991748e0d574b5fae2d5cacc5930b033d648e61689ba640150f.jpg)  
Fig. 5. Depth maps rendered using standard 3DGS and stochastic-solid volume rendering for the same Gaussian scene.

Following the equivalence established in prior work [35], each Gaussian primitive is replaced by an equivalent stochastic solid to preserve its RGB contribution. The associated vacancy v is computed as shown in Equation 8, where $G ( t ^ { * } )$ is the maximum Gaussian value along the ray.

$$
\mathbf { v } = { \sqrt { 1 - G ( t ^ { * } ) } }\tag{8}
$$

Both RGB images and depth maps are generated through volume rendering. The RGB rendering formulation is shwon in Equation 9.

$$
\begin{array} { l } { { \displaystyle { \hat { \mathbf { C } } } ( \mathbf { p } ) = \int _ { t _ { n } } ^ { t _ { f } } p ( t ) \mathbf { c } ( \mathbf { x } ( t ) , \omega ) \mathrm { d } t = \mathbf { c } \left( 1 - \mathbf { v } ( t ) ^ { 2 } \right) } , } \\ { { \displaystyle p ( t ) = T ( t ) \sigma ( \mathbf { x } ( t ) , \omega ) } , } \\ { { \displaystyle T ( t ) = \exp \left( - \int _ { t _ { n } } ^ { t } \sigma ( \mathbf { x } ( s ) , \omega ) \mathrm { d } s \right) } } \end{array}\tag{9}
$$

Depth rendering uses a fixed number of bisection iterations to locate the median depth, where accumulated transmittance $T ( t )$ reaches 0.5. RGB rendering remains differentiable throughout. For depth back-propagation, we use the gradient at the median depth, with $\begin{array} { r } { \frac { \partial \hat { T } } { \partial t } = - \overline { { T } } ( t ) \sum _ { i } \sigma _ { i } ( t ) } \end{array}$

2) Training Strategy: OctreeSplat’s training pipeline is differentiable and trained end-to-end. We use pretrained DA3- GIANT weights [1] for the VFM backbone. Excluding the camera-parameter encoder and Gaussian head, the backbone contains 1.190 billion parameters. OctreeSplat adds 76 million parameters: 36 million in the feature DPT and 40 million in the sparse U-Net and decoding MLPs. The backbone remains fully frozen, and only the additional parameters are optimized, substantially reducing training time.

a) Training datasets: We train on a mixture of Infinigen [49], ScanNet++ [50], ARKitScenes [51], and DL3DV [52]. Infinigen provides synthetic RGB-D data, ScanNet++ and ARKitScenes provide real-captured RGB-D data, and DL3DV provides real-captured RGB data. The combined dataset contains 6,810,309 images from 12,419 scenes, covering indoor and outdoor environments as well as synthetic and realcaptured images.

b) Implementation details: We train the additional parameters for 100,000 iterations using AdamW. The learning rate starts at $3 \times 1 0 ^ { - 6 }$ and decays to $1 0 ^ { - 8 }$ with cosine annealing. We use $L = 3$ voxel levels, with a coarsest voxel size of 8 mm. Training images have a resolution of $2 4 4 \times 4 4 8$ Each iteration randomly samples 3–18 images per scene, with at most 18 images and four scenes in a batch. Training takes 35 hours on a single NVIDIA RTX PRO 6000 GPU.

c) Loss functions: To improve training efficiency, we render the Gaussian scene at each of the L levels and supervise it directly, without octree subdivision. Densification scores are supervised using gradients with respect to Gaussian positions. The objective comprises ground-truth supervision, VFM alignment, and score prediction losses. Stochastic-solid volume rendering produces RGB images, depth maps, and normal maps for each level. Ground-truth supervision includes RGB L1 loss $\mathcal { L } _ { \mathrm { R G B } } ^ { L 1 }$ , RGB SSIM loss $\bar { \mathcal { L } } _ { \mathrm { R G B } } ^ { \mathrm { S S I M } }$ , depth L1 loss $\mathcal { L } _ { D }$ , and normal cosine-consistency loss $\mathcal { L } _ { N }$ . Groundtruth normals are computed from depth-map gradients. VFM alignment comprises depth loss $\mathcal { L } _ { D } ^ { \mathrm { V F M } }$ and normal loss $\mathcal { L } _ { N } ^ { \mathrm { V F M } }$ These compare rendered depth and normals with the VFM’s depth predictions and the derived normals.

The score loss follows the densification criterion of the original 3DGS [4]. During back-propagation, we accumulate the absolute gradients of projected Gaussian centers over multiple views at each level [29], as showin in Equation 10.

$$
g _ { v } ^ { ( \ell ) } = \sum _ { i = 1 } ^ { N } \left( \left| \frac { \partial \mathcal { L } _ { i } ^ { ( \ell ) } } { \partial \mu _ { i , v , x } ^ { \prime ( \ell ) } } \right| + \left| \frac { \partial \mathcal { L } _ { i } ^ { ( \ell ) } } { \partial \mu _ { i , v , y } ^ { \prime ( \ell ) } } \right| \right)\tag{10}
$$

Here, $v \in \mathcal { V } ^ { ( \ell ) }$ denotes a voxel at level ℓ. The coordinates $\mu _ { i , v , x } ^ { \prime ( \ell ) }$ and $\mu _ { i , v , y } ^ { \prime ( \ell ) }$ denote its Gaussian’s projected center in view i. To stabilize optimization and reduce the influence of outliers, we transform the accumulated gradient using $\mathrm { n } ( 1 + \alpha _ { g } g _ { v } ^ { ( \ell ) } )$ where $\alpha _ { g }$ is a hyperparameter. We then apply the same sigmoid normalization as the score prediction head and compute an L1 score loss $\mathcal { L } _ { S }$ . The complete training objective is shown in Equation 11.

$$
\begin{array} { r l } & { \mathcal { L } = ( \lambda _ { \mathrm { R G B } } ^ { L 1 } \mathcal { L } _ { \mathrm { R G B } } ^ { L 1 } + \lambda _ { \mathrm { R G B } } ^ { \mathrm { S S I M } } \mathcal { L } _ { \mathrm { R G B } } ^ { \mathrm { S S I M } } + \lambda _ { D } \mathcal { L } _ { D } + \lambda _ { N } \mathcal { L } _ { N } ) } \\ & { \qquad + ( \lambda _ { D } ^ { \mathrm { V F M } } \mathcal { L } _ { D } ^ { \mathrm { V F M } } + \lambda _ { N } ^ { \mathrm { V F M } } \mathcal { L } _ { N } ^ { \mathrm { V F M } } ) + ( \lambda _ { S } \mathcal { L } _ { S } ) } \end{array}\tag{11}
$$

The default loss weights are $\lambda _ { \mathrm { R G B } } ^ { L 1 } = 7 , \lambda _ { \mathrm { R G B } } ^ { \mathrm { S S I M } } = 3 , \lambda _ { D } = 2 5$ ${ { \lambda } _ { N } } = 0 . 2 , { { \lambda } _ { D } ^ { \mathrm { V F M } } } = 2 5 , { { \lambda } _ { N } ^ { \mathrm { V F M } } } = 0 . 2 ,$ and $\lambda _ { S } = 1$

## IV. EXPERIMENTS

## A. RGB View Synthesis

We evaluate NVS on Replica [53], TUM [54], and Tanks and Temples [55]. These datasets provide synthetic indoor, real-captured indoor, and real-captured outdoor scenes, respectively. We conducted qualitative and quantitative experiments between OctreeSplat and several SOTA baselines, which can be categorized into two groups: the first follows pixelSplat [37] and requires camera parameters, including MVSplat [9] and VolSplat [10]; the second uses VGGT-style [5] reconstruction, including AnySplat [2], YoNoSplat [8], DA3 [1], and QuerySplat [11]. AnySplat and VolSplat use voxel alignment, whereas QuerySplat uses implicit query tokens. MVSplat, YoNoSplat, and DA3 use pixel alignment. We measure RGB synthesis quality using PSNR, SSIM [56], and LPIPS [57]. These metrics characterize pixel-level fidelity, structural similarity, and perceptual similarity between synthesized and groundtruth views.

As shown in Table I and Figure 6, OctreeSplat produces clear RGB renderings and suppresses overlap artifacts. Octree-Splat (Finest) directly uses Gaussian primitives predicted from the finest voxel level. OctreeSplat efficiently controls the primitive count, which reduces the count by approximately half or more in most settings, with a quality–compactness trade-off relative to the finest-level variant.

## B. Depth Map Synthesis

We evaluate depth synthesis using the same datasets and baselines as for RGB synthesis. Metrics include root mean square error (RMSE), absolute relative error (AbsRel), and threshold accuracy $( \delta _ { m } )$ . RMSE measures the root mean squared difference between predicted depth <sup>ˆ</sup>d and groundtruth depth d. For $N _ { \mathrm { v a l i d } }$ valid pixels, AbsRel is AbsRel = $\begin{array} { r } { \frac { 1 } { N _ { \mathrm { v a l i d } } } \sum _ { i = 1 } ^ { \infty } { \frac { | \hat { d } _ { i } - d _ { i } | } { d _ { i } } } } \end{array}$ . Threshold accuracy is the fraction of pixels satisfying max $\cdot ( \hat { d } _ { i } / d _ { i } , d _ { i } / \hat { d } _ { i } ) < m$ , with $m = 1 . 0 5$ and $m = 1 . 1 0$

As shown in Table II and Figure 7, OctreeSplat shows strong depth synthesis performance across the evaluated datasets.

## C. Pose Estimation

OctreeSplat inherits the pretrained DA3 backbone and therefore produces the same pose predictions before TTO. When ground-truth camera parameters are available, the additional TTO module improves alignment between predictions and the ground-truth parameters. TTO uses 20 iterations with a lr of $5 \times 1 0 ^ { - 4 }$ . As shown in Table III, OctreeSplat and DA3 have identical pose accuracy before adaptation. Conditioning DA3 on camera parameters, denoted DA3 (Pose), improves poses accuracy. OctreeSplat does not encode camera parameters into the backbone. Instead, TTO uses them as supervision and achieves lower translation and rotation errors, with particularly clear gains in rotation accuracy.

TABLE III  
QUANTITATIVE POSE ESTIMATION RESULTS WITH 20 INPUT VIEWS.
<table><tr><td rowspan="2">Method</td><td colspan="2">Trans. (mm) ↓</td><td colspan="2">Rot. (°) ↓</td></tr><tr><td>Mean</td><td>Median</td><td>Mean</td><td>Median</td></tr><tr><td colspan="5">DTU</td></tr><tr><td>YoNoSplat</td><td>16.383</td><td>11.687</td><td>5.562</td><td>5.210</td></tr><tr><td>AnySplat</td><td>5.846</td><td>4.581</td><td>2.105</td><td>2.036</td></tr><tr><td>QuerySplat</td><td>3.043</td><td>2.773</td><td>0.971</td><td>0.969</td></tr><tr><td>ÖctreeSplat &amp; DA3</td><td>2.007</td><td>1.535</td><td>0.530</td><td>0.511</td></tr><tr><td>DA3 (Pose)</td><td>1.250</td><td>1.049</td><td>0.503</td><td>0.477</td></tr><tr><td>OctreeSplat (TTO)</td><td>1.134</td><td>0.903</td><td>0.254</td><td>0.238</td></tr><tr><td colspan="5">Replica</td></tr><tr><td>YoNoSplat</td><td>118.385</td><td>70.258</td><td>5.195</td><td>4.415</td></tr><tr><td>AnySplat</td><td>109.801</td><td>95.020</td><td>13.474</td><td>13.754</td></tr><tr><td>QuerySplat</td><td>15.357</td><td>12.439</td><td>0.825</td><td>0.786</td></tr><tr><td>OctreeSplat &amp; DA3</td><td>8.434</td><td>5.981</td><td>0.444</td><td>0.400</td></tr><tr><td>DA3 (Pose)</td><td>6.008</td><td>4.019</td><td>0.499</td><td>0.450</td></tr><tr><td>OctreeSplat (TTO)</td><td>4.045</td><td>3.172</td><td>0.205</td><td>0.192</td></tr></table>

## D. Surface Reconstruction

Following the DTU evaluation protocol [58], we measure surface reconstruction quality using the symmetric Chamfer distance (CD). Let $P$ and Q denote points sampled from the reconstructed and ground-truth surfaces. Accuracy (Acc.) and completeness (Com.) are the symmetric mean nearestneighbor distances, and CD is their arithmetic mean, as shown in Equation 12.

$$
\begin{array} { r l r } {  { \mathrm { C D } ( P , Q ) = \frac { \operatorname { A c c } . ( P , Q ) + \mathrm { C o m . } ( P , Q ) } { 2 } } } \\ & { } & { = \frac { 1 } { 2 } \Bigg ( \frac { 1 } { | P | } \sum _ { \mathbf { p } \in P } \underset { \mathbf { q } \in Q } { \operatorname* { m i n } } \| \mathbf { p } - \mathbf { q } \| _ { 2 } } \\ & { } & { +  \frac { 1 } { | Q | } \sum _ { \mathbf { q } \in Q } \underset { \mathbf { p } \in P } { \operatorname* { m i n } } \| \mathbf { q } - \mathbf { p } \| _ { 2 } ) . } \end{array}\tag{12}
$$

Because OctreeSplat and the evaluated baselines reconstruct scenes without input poses, we first align predicted and ground-truth camera poses using the Umeyama algorithm [59].

OctreeSplat (Finest)

MVSplat  
AnySplat  
DA3  
QuerySplat  
OctreeSplat  
![](images/61aeec1227bade7cf6484408f88c04474094daa27a51482b41375d5b9dd9933f.jpg)  
Fig. 6. Qualitative comparison of RGB view synthesis.

TABLE I  
QUANTITATIVE RGB VIEW SYNTHESIS RESULTS. BOLD, UNDERLINED, AND ITALIC VALUES INDICATE THE BEST, SECOND-BEST, AND THIRD-BEST RESULTS.
<table><tr><td rowspan="2">Method</td><td colspan="4">3 Views</td><td colspan="4">10 Views</td><td colspan="4">20 Views</td></tr><tr><td>PSNR (dB) ↑</td><td>SSIM ↑</td><td>LPIPS ↓</td><td>Count (100K)</td><td>PSNR (dB) ↑</td><td>SSIM ↑</td><td>LPIPS ↓</td><td>Count (100K)</td><td>PSNR (dB) ↑</td><td>SSIM ↑</td><td>LPIPS ↓</td><td>Count (100K)</td></tr><tr><td colspan="10">Replica</td></tr><tr><td>MVSplat</td><td>17.02</td><td>0.757</td><td>0.303</td><td>1.97</td><td>19.97</td><td>0.759</td><td>0.318</td><td>6.55</td><td>10.69</td><td>0.324</td><td>0.610</td><td>6.55</td></tr><tr><td>VolSplat</td><td>19.83</td><td>0.754</td><td>0.450</td><td>1.97</td><td>16.15</td><td>0.667</td><td>0.471</td><td>6.54</td><td>14.77</td><td>0.615</td><td>0.567</td><td>6.53</td></tr><tr><td>AnyŠplat</td><td>25.82</td><td>0.886</td><td>0.159</td><td>6.74</td><td>23.25</td><td>0.825</td><td>0.200</td><td>20.80</td><td>22.93</td><td>0.810</td><td>0.286</td><td>42.18</td></tr><tr><td>YoNoSplat</td><td>25.68</td><td>0.833</td><td>0.279</td><td>1.51</td><td>24.52</td><td>0.809</td><td>0.262</td><td>5.02</td><td>23.65</td><td>0.806</td><td>0.331</td><td>10.04</td></tr><tr><td>DA3</td><td>21.42</td><td>0.782</td><td>0.335</td><td>8.04</td><td>23.64</td><td>0.837</td><td>0.208</td><td>26.83</td><td>26.00</td><td>0.860</td><td>0.212</td><td>53.66</td></tr><tr><td>QuerySplat</td><td>33.14</td><td>0.924</td><td>0.199</td><td>2.00</td><td>30.03</td><td>0.884</td><td>0.270</td><td>1.87</td><td>28.32</td><td>0.859</td><td>0.316</td><td>1.75</td></tr><tr><td>OctreeSplat</td><td>34.27</td><td>0.944</td><td>0.198</td><td>2.81</td><td>32.10</td><td>0.923</td><td>0.228</td><td>8.32</td><td>31.24</td><td>0.910</td><td>0.243</td><td>17.61</td></tr><tr><td>OctreeSplat(Finest)</td><td>36.22</td><td>0.966</td><td>0.128</td><td>7.32</td><td>33.47</td><td>0.947</td><td>0.158</td><td>22.83</td><td>32.40</td><td>0.934</td><td>0.185</td><td>46.08</td></tr><tr><td colspan="10">Tanks and Temples</td><td colspan="3"></td></tr><tr><td>MVSplat</td><td>14.18</td><td>0.537</td><td>0.401</td><td>1.97</td><td>15.65</td><td>0.541</td><td>0.420</td><td>6.55</td><td>13.76</td><td>0.431</td><td>0.544</td><td>6.55</td></tr><tr><td>VolSplat</td><td>15.28</td><td>0.520</td><td>0.506</td><td>1.96</td><td>14.98</td><td>0.544</td><td>0.439</td><td>6.55</td><td>12.87</td><td>0.356</td><td>0.595</td><td>6.53</td></tr><tr><td>AnyŠplat</td><td>19.24</td><td>0.715</td><td>0.263</td><td>6.14</td><td>16.89</td><td>0.615</td><td>0.318</td><td>19.71</td><td>16.67</td><td>0.536</td><td>0.394</td><td>29.35</td></tr><tr><td>YoNoSplat</td><td>12.30</td><td>0.372</td><td>0.588</td><td>1.51</td><td>13.17</td><td>0.420</td><td>0.533</td><td>5.02</td><td>13.62</td><td>0.374</td><td>0.587</td><td>10.04</td></tr><tr><td>DA3</td><td>17.63</td><td>0.551</td><td>0.398</td><td>8.04</td><td>18.26</td><td>0.567</td><td>0.373</td><td>26.83</td><td>18.48</td><td>0.533</td><td>0.428</td><td>53.66</td></tr><tr><td>QuerySplat</td><td>25.43</td><td>0.745</td><td>0.286</td><td>2.34</td><td>22.31</td><td>0.612</td><td>0.397</td><td>2.00</td><td>22.17</td><td>0.647</td><td>0.397</td><td>2.01</td></tr><tr><td>OctreeSplat</td><td>25.10</td><td>0.826</td><td>0.253</td><td>4.73</td><td>22.28</td><td>0.746</td><td>0.306</td><td>13.97</td><td>20.52</td><td>0.662</td><td>0.385</td><td>20.58</td></tr><tr><td>OctreeSplat(Finest)</td><td>26.51</td><td>0.876</td><td>0.170</td><td>7.22</td><td>22.77</td><td>0.784</td><td>0.253</td><td>22.79</td><td>21.32</td><td>0.697</td><td>0.336</td><td>39.68</td></tr><tr><td colspan="10">TUM</td><td colspan="3"></td></tr><tr><td>MVSplat</td><td>11.03</td><td>0.412</td><td>0.551</td><td>1.97</td><td>12.01</td><td>0.407</td><td>0.585</td><td>6.55</td><td>11.70</td><td>0.386</td><td>0.617</td><td>6.55</td></tr><tr><td>VolSplat</td><td>14.61</td><td>0.466</td><td>0.572</td><td>1.97</td><td>13.61</td><td>0.447</td><td>0.582</td><td>6.55</td><td>13.16</td><td>0.423</td><td>0.610</td><td>6.54</td></tr><tr><td>AnyŠplat</td><td>24.33</td><td>0.825</td><td>0.203</td><td>5.38</td><td>19.20</td><td>0.707</td><td>0.315</td><td>14.00</td><td>18.00</td><td>0.655</td><td>0.365</td><td>28.33</td></tr><tr><td>YoNoSplat</td><td>16.55</td><td>0.552</td><td>0.471</td><td>1.51</td><td>16.36</td><td>0.534</td><td>0.474</td><td>5.02</td><td>16.01</td><td>0.530</td><td>0.504</td><td>10.04</td></tr><tr><td>DA3</td><td>19.62</td><td>0.692</td><td>0.363</td><td>8.04</td><td>16.51</td><td>0.626</td><td>0.439</td><td>26.83</td><td>17.63</td><td>0.639</td><td>0.423</td><td>53.66</td></tr><tr><td>QuerySplat</td><td>24.61</td><td>0.791</td><td>0.299</td><td>1.66</td><td>20.53</td><td>0.682</td><td>0.414</td><td>1.30</td><td>19.71</td><td>0.650</td><td>0.459</td><td>1.13</td></tr><tr><td>OctreeSplat</td><td>27.86</td><td>0.858</td><td>0.282</td><td>3.48</td><td>21.86</td><td>0.767</td><td>0.364</td><td>10.40</td><td>21.00</td><td>0.723</td><td>0.394</td><td>17.46</td></tr><tr><td>OctreeSplat(Finest)</td><td>29.40</td><td>0.893</td><td>0.202</td><td>7.33</td><td>21.83</td><td>0.775</td><td>0.307</td><td>23.98</td><td>21.38</td><td>0.733</td><td>0.354</td><td>43.49</td></tr></table>

![](images/64bcb8c2f9b901512392cb2c764789389020247af136f03584d0577a5b50bffe.jpg)  
Fig. 7. Qualitative comparison of depth map synthesis.

TABLE II  
QUANTITATIVE DEPTH SYNTHESIS RESULTS.
<table><tr><td>Method</td><td colspan="4">3 Views</td><td colspan="4">10 Views</td><td colspan="4">20 Views</td></tr><tr><td></td><td>RMSE (cm) ↓</td><td>AbsRel (%) ↓</td><td> $\delta _ { 1 . 0 5 }$  ↑</td><td> $\delta _ { 1 . 1 0 }$  ↑</td><td>RMSE (cm) ↓</td><td>AbsRel (%) ↓</td><td> $\delta _ { 1 . 0 5 }$  ↑</td><td> $\delta _ { 1 . 1 0 }$  ↑</td><td>RMSE (cm) ↓</td><td>AbsRel (%) ↓</td><td> $\delta _ { 1 . 0 5 }$  ↑</td><td> $\delta _ { 1 . 1 0 }$  ↑</td></tr><tr><td colspan="9">Replica</td><td></td><td></td><td></td></tr><tr><td>MVSplat</td><td>114.92</td><td>30.90</td><td>12.78</td><td>24.15</td><td>216.08</td><td>69.27</td><td>1.98</td><td>3.89</td><td>202.49</td><td>66.09</td><td>2.26</td><td>4.37</td></tr><tr><td>VolSplat</td><td>72.77</td><td>20.89</td><td>16.19</td><td>30.36</td><td>102.87</td><td>30.43</td><td>11.88</td><td>22.56</td><td>100.54</td><td>31.53</td><td>12.40</td><td>23.16</td></tr><tr><td>AnyŠplat</td><td>15.39</td><td>3.51</td><td>76.32</td><td>95.70</td><td>33.73</td><td>9.58</td><td>59.05</td><td>76.88</td><td>21.63</td><td>5.52</td><td>65.52</td><td>86.30</td></tr><tr><td>YoNoSplat</td><td>27.04</td><td>7.59</td><td>44.76</td><td>73.48</td><td>23.47</td><td>7.28</td><td>51.08</td><td>75.58</td><td>22.64</td><td>7.34</td><td>51.25</td><td>76.38</td></tr><tr><td>DA3</td><td>105.30</td><td>32.62</td><td>13.49</td><td>25.20</td><td>125.53</td><td>43.28</td><td>28.31</td><td>46.40</td><td>21.39</td><td>5.65</td><td>73.96</td><td>88.62</td></tr><tr><td>QuerySplat</td><td>7.25</td><td>1.66</td><td>95.30</td><td>98.98</td><td>8.88</td><td>2.36</td><td>89.25</td><td>96.18</td><td>10.93</td><td>2.94</td><td>84.26</td><td>95.20</td></tr><tr><td>OctreeSplat</td><td>4.50</td><td>0.38</td><td>99.50</td><td>99.71</td><td>5.35</td><td>0.48</td><td>99.31</td><td>99.58</td><td>6.25</td><td>0.59</td><td>99.04</td><td>99.40</td></tr><tr><td>OctreeSplat(Finest)</td><td>4.37</td><td>0.36</td><td>99.50</td><td>99.71</td><td>5.19</td><td>0.45</td><td>99.34</td><td>99.60</td><td>6.00</td><td>0.54</td><td>99.17</td><td>99.45</td></tr><tr><td colspan="9">Tanks and Temples</td><td></td><td></td><td></td><td></td></tr><tr><td>MVSplat</td><td>486.96</td><td>35.15</td><td>12.96</td><td>24.89</td><td>708.32</td><td>64.97</td><td>0.97</td><td>1.95</td><td>880.57</td><td>63.64</td><td>10.82</td><td>20.12</td></tr><tr><td>VolSplat</td><td>402.38</td><td>35.34</td><td>8.76</td><td>16.85</td><td>404.29</td><td>37.41</td><td>8.04</td><td>15.44</td><td>409.49</td><td>40.08</td><td>10.61</td><td>21.12</td></tr><tr><td>AnyŠplat</td><td>111.23</td><td>5.13</td><td>67.28</td><td>89.96</td><td>91.41</td><td>4.19</td><td>74.78</td><td>92.55</td><td>83.96</td><td>8.76</td><td>81.63</td><td>91.20</td></tr><tr><td>YoNoSplat</td><td>416.84</td><td>27.57</td><td>8.02</td><td>17.22</td><td>423.07</td><td>40.87</td><td>6.45</td><td>12.69</td><td>197.36</td><td>27.17</td><td>28.12</td><td>41.91</td></tr><tr><td>DA3</td><td>416.05</td><td>31.97</td><td>15.11</td><td>27.21</td><td>368.92</td><td>38.28</td><td>15.65</td><td>30.66</td><td>255.79</td><td>26.38</td><td>30.98</td><td>42.56</td></tr><tr><td>QuerySplat</td><td>135.41</td><td>4.53</td><td>78.10</td><td>90.39</td><td>190.17</td><td>4.99</td><td>82.93</td><td>92.11</td><td>325.92</td><td>24.42</td><td>59.00</td><td>75.41</td></tr><tr><td>OctreeSplat</td><td>91.48</td><td>4.81</td><td>74.09</td><td>85.56</td><td>65.89</td><td>2.22</td><td>93.01</td><td>97.38</td><td>114.33</td><td>8.74</td><td>92.14</td><td>94.61</td></tr><tr><td>OctreeSplat(Finest)</td><td>87.77</td><td>3.56</td><td>76.88</td><td>89.34</td><td>54.22</td><td>1.78</td><td>95.36</td><td>98.45</td><td>104.48</td><td>8.33</td><td>93.37</td><td>94.73</td></tr><tr><td colspan="9">TUM</td><td rowspan="3"></td><td rowspan="3"></td><td rowspan="3"></td><td rowspan="3"></td><td rowspan="3">4.63</td></tr><tr><td>MVSplat</td><td>182.10</td><td>16.35</td><td>24.51</td><td>45.02</td><td>1388.26</td><td>105.96</td><td>6.51</td></tr><tr><td>VolSplat</td><td>162.56</td><td>13.74</td><td>25.96</td><td>48.93</td><td>146.04</td><td>13.05</td><td>30.59 50.66</td><td>187.98</td><td>13.59</td></tr><tr><td>AnyŠplat</td><td>70.27</td><td>5.00</td><td>65.57</td><td>83.67</td><td>77.69</td><td>5.44</td><td>66.31</td><td>85.03</td><td>113.17</td><td>6.32</td><td>69.34</td><td>86.30</td></tr><tr><td>YoNoSplat</td><td>117.50</td><td>8.71</td><td>42.38</td><td>66.10</td><td>104.08</td><td>7.91</td><td>57.47</td><td>76.95</td><td>120.41</td><td>8.65</td><td>54.82</td><td>74.42</td></tr><tr><td>DA3</td><td>150.90</td><td>16.08</td><td>19.67</td><td>39.67</td><td>166.85</td><td>15.20</td><td>30.89</td><td>55.73</td><td>173.82</td><td>11.60</td><td>47.10</td><td>68.50</td></tr><tr><td>QuerySplat</td><td>91.88</td><td>7.18</td><td>62.08</td><td>81.05</td><td>97.82</td><td>8.49</td><td>68.53</td><td>81.99</td><td>114.56</td><td>9.23</td><td>60.39</td><td>74.53</td></tr><tr><td>OctreeSplat</td><td>51.35</td><td>2.28</td><td>92.99</td><td>96.62</td><td>61.73</td><td>2.77</td><td>91.95</td><td>96.48</td><td>76.98</td><td>3.11</td><td>90.41</td><td>95.67</td></tr><tr><td>OctreeSplat(Finest)</td><td>51.67</td><td>2.29</td><td>92.95</td><td>96.60</td><td>64.89</td><td>2.87</td><td>91.96</td><td>96.35</td><td>79.54</td><td>3.20</td><td>90.31</td><td>95.62</td></tr></table>

Joint errors in poses, intrinsics, and depth can leave residual surface misalignment after camera alignment. We therefore also apply iterative closest point (ICP) alignment between reconstructed and ground-truth surfaces before computing CD. Results before and after ICP are reported separately to distinguish alignment errors from residual surface discrepancies.

As shown in Table IV and Figure 8, OctreeSplat improves surface reconstruction. Adding TTO reduces surface misalignment caused by pose errors and further improves the reconstruction accuracy after ICP.

TABLE IV  
SURFACE RECONSTRUCTION RESULTS BEFORE AND AFTER ICP ALIGNMENT. ALL DISTANCES ARE MEASURED IN MM.
<table><tr><td rowspan="2">Method</td><td colspan="3">Before ICP</td><td colspan="3">After ICP</td></tr><tr><td>Acc.</td><td>Com.</td><td>CD</td><td>Acc.</td><td>Com.</td><td>CD</td></tr><tr><td colspan="7">Replica</td></tr><tr><td>AnySplat</td><td>51.905</td><td>49.643</td><td>50.774</td><td>41.793</td><td>38.600</td><td>40.196</td></tr><tr><td>DA3</td><td>65.833</td><td>64.426</td><td>65.129</td><td>46.325</td><td>43.459</td><td>44.892</td></tr><tr><td>YoNoSplat</td><td>48.781</td><td>46.121</td><td>47.451</td><td>43.176</td><td>40.751</td><td>41.963</td></tr><tr><td>QuerySplat</td><td>36.569</td><td>34.523</td><td>35.546</td><td>33.769</td><td>31.730</td><td>32.750</td></tr><tr><td>OctreeSplat</td><td>15.108</td><td>14.504</td><td>14.806</td><td>12.105</td><td>11.686</td><td>11.895</td></tr><tr><td>OctreeSplat (TTO)</td><td>9.221</td><td>8.831</td><td>9.026</td><td>8.495</td><td>8.172</td><td>8.333</td></tr><tr><td colspan="7">DTU</td></tr><tr><td>AnySplat</td><td>4.408</td><td>11.186</td><td>7.797</td><td>3.544</td><td>9.976</td><td>6.760</td></tr><tr><td>DA3</td><td></td><td>Fail</td><td></td><td></td><td>Fail</td><td></td></tr><tr><td>YoNoSplat</td><td>7.740</td><td>12.830</td><td>10.285</td><td>7.287</td><td>12.020</td><td>9.654</td></tr><tr><td>QuerySplat</td><td>5.194</td><td>9.382</td><td>7.288</td><td>2.740</td><td>5.757</td><td>4.249</td></tr><tr><td>OctreeSplat</td><td>7.703</td><td>9.337</td><td>8.520</td><td>3.035</td><td>3.344</td><td>3.190</td></tr><tr><td>OctreeSplat (TTO)</td><td>3.187</td><td>5.700</td><td>4.443</td><td>1.105</td><td>1.914</td><td>1.510</td></tr></table>

![](images/433a001dbbdce088d45e54f3063792a2f68b7725be0540305042693292f03899.jpg)  
Fig. 8. Qualitative comparison of reconstructed surfaces.

## E. Efficiency

The inference time of OctreeSplat is shown in Figure 9. The total inference time is 2.66 s, compared with 2.33 s for DA3 using the same backbone, an increase of 14.2%. Most additional computation comes from the multi-scale SPUNet.

When TTO is enabled, OctreeSplat runs the VFM backbone in bf16 to improve efficiency, with little effect on accuracy. A backbone forward pass takes 287.9 ms, and back-propagation takes 426.0 ms. The total runtime with TTO remains below 15 s.

![](images/f8bb8518952134cc3affb1330633a276a92abc9ad175508325341854119145f7.jpg)  
Fig. 9. Model inference time.

## F. Ablation Studies

1) Stochastic-Solid Volume Rendering: We first replace stochastic-solid volume rendering with the vanilla 3DGS renderer [4]. As shown in Table V(a), after replacement, RGB synthesis accuracy shows a slight decline, while the depth synthesis accuracy exhibits a significant degradation. This confirms the effectiveness of the stochastic solid-volume rendering module.

2) Depth Supervision: We ablate depth supervision to assess the contribution of RGB-D training data. First, we remove the ground-truth depth and normal losses, $\mathcal { L } _ { D }$ and $\mathcal { L } _ { N }$ denoted OctreeSplat (w/o data depth). We then additionally remove the VFM depth and normal alignment losses, $\mathcal { L } _ { D } ^ { \mathrm { V F M } }$ and $\mathcal { L } _ { N } ^ { \mathrm { V F M } }$ , denoted OctreeSplat (w/o all depth losses). As shown in Table V(b) shows that removing depth supervision generally slightly improves RGB metrics and slightly degrades depth metrics. The model retains useful geometric accuracy without explicit depth losses, supporting its applicability to RGB-only training.

3) Voxel Size: We further ablate the coarsest voxel size over a range of 6–64 mm. As shown in Table V(c), increasing the voxel size substantially degrades RGB synthesis quality and reduces the number of Gaussian primitives, while its impact on depth synthesis is less pronounced. Balancing reconstruction quality and computational cost, and following prior work, we set the default coarsest voxel size to 8 mm.

![](images/5678343f09b3fe3507b3f2aa16ff319fabe9eea3443f4793a07b44d85be89d6a.jpg)  
Fig. 10. Spatial density of Gaussian primitives. Redder regions indicate higher primitive density.

4) Octree Subdivision: We compare four primitiveallocation strategies to assess score-guided octree subdivision. OctreeSplat uses predicted densification scores. OctreeSplat (Grad) renders multiple views and computes gradients directly to guide subdivision. OctreeSplat (Point) uses the maximum multi-view sampling rate, while OctreeSplat (Finest) uses Gaussian primitives from the finest voxel level. All subdivision-based variants use two rounds, splitting 30% of the voxels per round and discarding children without spatial points.

TABLE V  
ABLATION STUDIES. (A) STOCHASTIC-SOLID VOLUME RENDERING. (B) DEPTH SUPERVISION. (C) VOXEL SIZE. (D) OCTREE SUBDIVISION.  
(a) Stochastic-solid volume rendering
<table><tr><td>Dataset</td><td>Method</td><td>PSNR ↑</td><td>SSIM ↑</td><td>LPIPS↓</td><td>RMSE↓</td><td>AbsRel ↓</td><td>δ1.05 ↑</td><td>δ1.10 ↑</td></tr><tr><td rowspan="2">Replica</td><td>OctreeSplat</td><td>32.10</td><td>0.923</td><td>0.228</td><td>5.35</td><td>0.48</td><td>99.31</td><td>99.58</td></tr><tr><td>OctreeSplat(3DGS)</td><td>29.11</td><td>0.889</td><td>0.213</td><td>9.58</td><td>1.01</td><td>96.09</td><td>97.92</td></tr><tr><td rowspan="2">TUM</td><td>OctreeSplat</td><td>21.86</td><td>0.767</td><td>0.364</td><td>61.73</td><td>2.77</td><td>91.95</td><td>96.48</td></tr><tr><td>OctreeSplat(3DGS)</td><td>20.68</td><td>0.727</td><td>0.357</td><td>286.14</td><td>3.59</td><td>91.16</td><td>96.16</td></tr></table>

(b) Depth supervision
<table><tr><td>Dataset</td><td>Method</td><td>PSNR ↑</td><td>SSIM ↑</td><td>LPIPS ↓</td><td>RMSE↓</td><td>AbsRel ↓</td><td>δ1.05 ↑</td><td>δ1.10 ↑</td></tr><tr><td rowspan="3">Replica</td><td>OctreeSplat</td><td>32.10</td><td>0.923</td><td>0.228</td><td>5.35</td><td>0.48</td><td>99.31</td><td>99.58</td></tr><tr><td>OctreeSplat (w/o data depth)</td><td>32.56</td><td>0.928</td><td>0.219</td><td>5.39</td><td>0.48</td><td>99.29</td><td>99.58</td></tr><tr><td>OctreeSplat (w/o depth losses)</td><td>33.42</td><td>0.935</td><td>0.216</td><td>5.26</td><td>0.57</td><td>99.18</td><td>99.60</td></tr><tr><td rowspan="3">TUM</td><td>OctreeSplat</td><td>21.86</td><td>0.767</td><td>0.364</td><td>61.73</td><td>2.77</td><td>91.95</td><td>96.48</td></tr><tr><td>OctreeSplat (w/o data depth)</td><td>21.80</td><td>0.772</td><td>0.353</td><td>62.70</td><td>2.83</td><td>91.94</td><td>96.38</td></tr><tr><td>OctreeSplat (w/o depth losses)</td><td>23.19</td><td>0.807</td><td>0.344</td><td>64.82</td><td>2.86</td><td>92.17</td><td>96.65</td></tr></table>

(c) Voxel size
<table><tr><td>Dataset</td><td>Method</td><td>PSNR ↑</td><td>SSIM ↑</td><td>LPIPS ↓</td><td>RMSE↓</td><td>AbsRel ↓</td><td>δ1.05 ↑</td><td>δ1.10 ↑</td><td>Count(100K) ↓</td></tr><tr><td rowspan="6">Replica</td><td>OctreeSplat (6mm)</td><td>32.73</td><td>0.931</td><td>0.206</td><td>5.21</td><td>0.48</td><td>99.33</td><td>99.60</td><td>11.85</td></tr><tr><td>OctreeSplat (8mm)</td><td>32.10</td><td>0.923</td><td>0.228</td><td>5.35</td><td>0.48</td><td>99.31</td><td>99.58</td><td>8.32</td></tr><tr><td>OctreeSplat (12mm)</td><td>30.57</td><td>0.904</td><td>0.275</td><td>5.74</td><td>0.49</td><td>99.22</td><td>99.52</td><td>4.64</td></tr><tr><td>OctreeSplat (16mm)</td><td>28.98</td><td>0.883</td><td>0.319</td><td>6.11</td><td>0.52</td><td>99.12</td><td>99.44</td><td>2.86</td></tr><tr><td>OctreeSplat (32mm)</td><td>21.16</td><td>0.784</td><td>0.465</td><td>9.31</td><td>0.98</td><td>98.26</td><td>98.81</td><td>0.74</td></tr><tr><td>OctreeSplat (64mm)</td><td>15.70</td><td>0.701</td><td>0.582</td><td>15.79</td><td>2.74</td><td>94.88</td><td>96.30</td><td>0.17</td></tr><tr><td rowspan="6">TUM</td><td>OctreeSplat (6mm)</td><td>22.12</td><td>0.778</td><td>0.349</td><td>63.42</td><td>2.84</td><td>91.94</td><td>96.45</td><td>13.69</td></tr><tr><td>OctreeSplat (8mm)</td><td>21.86</td><td>0.767</td><td>0.364</td><td>61.73</td><td>2.77</td><td>91.95</td><td>96.48</td><td>10.40</td></tr><tr><td>OctreeSplat (12mm)</td><td>21.30</td><td>0.746</td><td>0.388</td><td>58.23</td><td>2.67</td><td>91.94</td><td>96.48</td><td>6.43</td></tr><tr><td>OctreeSplat (16mm)</td><td>20.62</td><td>0.724</td><td>0.417</td><td>56.32</td><td>2.63</td><td>91.83</td><td>96.44</td><td>4.25</td></tr><tr><td>OctreeSplat (32mm)</td><td>17.51</td><td>0.645</td><td>0.508</td><td>55.17</td><td>2.67</td><td>90.82</td><td>95.88</td><td>1.29</td></tr><tr><td>OctreeSplat (64mm)</td><td>13.27</td><td>0.560</td><td>0.608</td><td>64.50</td><td>3.18</td><td>87.52</td><td>93.85</td><td>0.35</td></tr></table>

(d) Octree subdivision
<table><tr><td>Dataset</td><td>Method</td><td>PSNR ↑</td><td>SSIM ↑</td><td>LPIPS↓</td><td>RMSE↓</td><td>AbsRel↓</td><td>δ1.05 ↑</td><td>δ1.10 ↑</td><td>Count(100K) ↓</td></tr><tr><td rowspan="4">Replica</td><td>OctreeSplat</td><td>32.10</td><td>0.923</td><td>0.228</td><td>5.35</td><td>0.48</td><td>99.31</td><td>99.58</td><td>8.32</td></tr><tr><td>OctreeSplat (Grad)</td><td>32.40</td><td>0.926</td><td>0.227</td><td>5.29</td><td>0.48</td><td>99.32</td><td>99.59</td><td>8.39</td></tr><tr><td>OctreeSplat (Point)</td><td>30.89</td><td>0.913</td><td>0.214</td><td>5.57</td><td>0.48</td><td>99.25</td><td>99.54</td><td>12.22</td></tr><tr><td>OctreeSplat (Finest)</td><td>33.47</td><td>0.947</td><td>0.158</td><td>5.19</td><td>0.45</td><td>99.34</td><td>99.60</td><td>22.83</td></tr><tr><td rowspan="4">TUM</td><td>OctreeSplat</td><td>21.86</td><td>0.767</td><td>0.364</td><td>61.73</td><td>2.77</td><td>91.95</td><td>96.48</td><td>10.40</td></tr><tr><td>OctreeSplat (Grad)</td><td>21.69</td><td>0.768</td><td>0.358</td><td>65.33</td><td>2.90</td><td>91.90</td><td>96.31</td><td>10.83</td></tr><tr><td>OctreeSplat (Point)</td><td>21.55</td><td>0.756</td><td>0.344</td><td>59.08</td><td>2.67</td><td>91.95</td><td>96.53</td><td>18.16</td></tr><tr><td>OctreeSplat (Finest)</td><td>21.83</td><td>0.775</td><td>0.307</td><td>64.89</td><td>2.87</td><td>91.96</td><td>96.35</td><td>23.98</td></tr></table>

As shown in Table V(d), OctreeSplat (Finest) achieves the highest RGB and depth synthesis accuracy but uses the most Gaussian primitives. OctreeSplat (Point) exhibits a substantial performance drop without a notable reduction in primitive count. OctreeSplat and OctreeSplat (Grad) achieve comparable performance and primitive counts, while OctreeSplat avoids the multi-view rendering and gradient computation required by OctreeSplat (Grad). These results demonstrate the effectiveness of the densification score guided octree subdivision strategy. As shown in Figure 10, OctreeSplat appropriately allocates more primitives to complex regions.

## V. CONCLUSION AND FUTURE WORK

We presented OctreeSplat, a hierarchical octree-guided method for feed-forward 3DGS reconstruction. Given images and optional camera parameters, OctreeSplat constructs hierarchical voxel features from multi-view features and geometric priors provided by a frozen VFM. A sparse-convolutional U-Net and separate heads jointly predict Gaussian parameters and inter-level densification scores. These scores guide octree subdivision and allocate primitives according to local scene complexity. Training updates only 76 million parameters in the feature DPT, SPUNet, densification-score heads, and Gaussian prediction heads, and takes 35 GPU-hours. Stochastic-solid volume rendering improves depth rendering and geometric supervision. Optional TTO optimizes LoRA modules inserted into the frozen backbone to align predicted and groundtruth camera parameters. Experiments across multiple datasets demonstrate strong RGB synthesis and geometric accuracy, while octree-guided allocation produces more compact scene representations. When camera parameters are supplied, TTO improves pose accuracy and alignment between reconstructed and ground-truth surfaces.

Future work will investigate more flexible subdivision strategies and dynamic scene reconstruction. The current octree uses predefined levels and fixed subdivision ratios. Learning adaptive subdivision and merging from local geometric complexity may offer greater flexibility in balancing compactness and reconstruction quality. The present method primarily targets static scenes and does not explicitly model object motion or nonrigid deformation. Incorporating Gaussian motion models could extend feed-forward reconstruction to dynamic scenes while preserving geometric consistency across frames.

## ACKNOWLEDGMENT

The authors gratefully acknowledge the financial support from the Zhejiang Provincial “Pioneer” and “Leading Goose + X” Science and Technology Program (Grant No. 2026C02A1220), the Shanghai Science and Technology Action Plan (Grant No. 21JM0010300), the Shanghai Aerospace Science and Technology Innovation Fund (SAST) (Grant No. 2021-037), the Special Fund for Technology Innovation Support Projects of Shanghai (Grant Nos. 2021-cyxt-kj1, XTCX-KJ-2022-37, and HCXBCY-2023-046), and the National Defense Basic Scientific Research Program of China (Grant No. JCKY2021606B002).

## REFERENCES

[1] H. Lin, S. Chen, J. H. Liew, D. Y. Chen, Z. Li, Y. Zhao, S. Peng, H. Guo, X. Zhou, G. Shi, J. Feng, and B. Kang, “Depth anything 3: Recovering the visual space from any views,” in International Conference on Learning Representations, C. Vondrick, B. Hariharan, C. Raffel, L. Pinto, D. Yang, and A. Faust, Eds., vol. 2026, 2026, pp. 141 261–141 285.

[2] L. Jiang, Y. Mao, L. Xu, T. Lu, K. Ren, Y. Jin, X. Xu, M. Yu, J. Pang, F. Zhao et al., “Anysplat: Feed-forward 3d gaussian splatting from unconstrained views,” ACM Transactions on Graphics (TOG), vol. 44, no. 6, pp. 1–16, 2025.

[3] R. Hartley, A. Zisserman et al., Multiple view geometry in computer vision. Cambridge university press, Cambridge, 2003, vol. 2, no. 3.

[4] B. Kerbl, G. Kopanas, T. Leimkuhler, and G. Drettakis, “3d gaussian¨ splatting for real-time radiance field rendering,” ACM Transactions on Graphics, vol. 42, no. 4, July 2023. [Online]. Available: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/

[5] J. Wang, M. Chen, N. Karaev, A. Vedaldi, C. Rupprecht, and D. Novotny, “Vggt: Visual geometry grounded transformer,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025.

[6] Y. Wang, J. Zhou, H. Zhu, W. Chang, Y. Zhou, Z. Li, J. Chen, J. Pang, C. Shen, and T. He, “π<sup>3</sup>: Permutation-equivariant visual geometry learning,” in International Conference on Learning Representations, vol. 2026, 2026, pp. 10 481–10 497.

[7] J. Wang, M. Chen, S. Zhang, N. Karaev, J. Schonberger, P. Labatut,¨ P. Bojanowski, D. Novotny, A. Vedaldi, and C. Rupprecht, “Vggt-ω,” 2026.

[8] B. Ye, B. Chen, H. Xu, D. Barath, and M. Pollefeys, “Yonosplat: You only need one model for feedforward 3d gaussian splatting,” in International Conference on Learning Representations (ICLR), 2026.

[9] Y. Chen, H. Xu, C. Zheng, B. Zhuang, M. Pollefeys, A. Geiger, T.- J. Cham, and J. Cai, “Mvsplat: Efficient 3d gaussian splatting from sparse multi-view images,” in European conference on computer vision. Springer, 2024, pp. 370–386.

[10] W. Wang, Y. Chen, Z. Zhang, H. Liu, H. Wang, Z. Feng, W. Qin, F. Chen, J.-W. Bian, Z. Zhu et al., “Volsplat: Rethinking feed-forward 3d gaussian splatting with voxel-aligned prediction,” in European Conference on Computer Vision. Springer, 2026, pp. 1–19.

[11] Y. Li, D. Shen, X. Zhang, Z. Ye, H. Wu, A. Hao, G. Zhang, and H. Liu, “Querysplat: Decoupling geometry and appearance representations in 3dgs prediction,” arXiv preprint arXiv:2608.01186, 2026. [Online]. Available: https://arxiv.org/abs/2608.01186

[12] Y. Li, C. Lv, Z. Tang, H. Yang, and D. Huang, “Tokensplat: Tokenaligned 3d gaussian splatting for feed-forward pose-free reconstruction,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2026, pp. 40 886–40 895.

[13] J. L. Schonberger and J.-M. Frahm, “Structure-from-motion revisited,”¨ in Conference on Computer Vision and Pattern Recognition (CVPR), 2016.

[14] S. M. Seitz and C. R. Dyer, “Photorealistic scene reconstruction by voxel coloring,” International journal of computer vision, vol. 35, no. 2, pp. 151–173, 1999.

[15] B. Curless and M. Levoy, “A volumetric method for building complex models from range images,” in Proceedings of the 23rd annual conference on Computer graphics and interactive techniques, 1996, pp. 303– 312.

[16] P. Labatut, J.-P. Pons, and R. Keriven, “Robust and efficient surface reconstruction from range data,” in Computer graphics forum, vol. 28, no. 8. Wiley Online Library, 2009, pp. 2275–2290.

[17] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and R. Ng, “Nerf: Representing scenes as neural radiance fields for view synthesis,” in ECCV, 2020.

[18] T. Muller, A. Evans, C. Schied, and A. Keller, “Instant neural graphics¨ primitives with a multiresolution hash encoding,” ACM Trans. Graph., vol. 41, no. 4, pp. 102:1–102:15, Jul. 2022. [Online]. Available: https://doi.org/10.1145/3528223.3530127

[19] S. Fridovich-Keil, A. Yu, M. Tancik, Q. Chen, B. Recht, and A. Kanazawa, “Plenoxels: Radiance fields without neural networks,” in CVPR, 2022.

[20] J. T. Barron, B. Mildenhall, D. Verbin, P. P. Srinivasan, and P. Hedman, “Mip-nerf 360: Unbounded anti-aliased neural radiance fields,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 5470–5479.

[21] J. T. Barron, B. Mildenhall, M. Tancik, P. Hedman, R. Martin-Brualla, and P. P. Srinivasan, “Mip-nerf: A multiscale representation for antialiasing neural radiance fields,” 2021.

[22] P. Wang, L. Liu, Y. Liu, C. Theobalt, T. Komura, and W. Wang, “Neus: Learning neural implicit surfaces by volume rendering for multi-view reconstruction,” arXiv preprint arXiv:2106.10689, 2021.

[23] Y. Wang, Q. Han, M. Habermann, K. Daniilidis, C. Theobalt, and L. Liu, “Neus2: Fast learning of neural implicit surfaces for multiview reconstruction,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2023.

[24] S. Kim, K. Lee, and Y. Lee, “Color-cued efficient densification method for 3d gaussian splatting,” in 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW). IEEE, 2024, pp. 775–783.

[25] Y. Lyu, K. Cheng, X. Kang, and X. Chen, “Resgs: Residual densification of 3d gaussian for efficient detail recovery,” in 2025 IEEE/CVF

International Conference on Computer Vision (ICCV). IEEE, 2025, pp. 28 093–28 102.

[26] Y. Chen, J. Jiang, K. Jiang, X. Tang, Z. Li, X. Liu, and Y. Nie, “Dashgaussian: Optimizing 3d gaussian splatting in 200 seconds,” in 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2025, pp. 11 146–11 155.

[27] S. S. Mallick, R. Goel, B. Kerbl, M. Steinberger, F. V. Carrasco, and F. De La Torre, “Taming 3dgs: High-quality radiance fields with limited resources,” in SIGGRAPH Asia 2024 Conference Papers, 2024, pp. 1– 11.

[28] A. Meuleman, I. Shah, A. Lanvin, B. Kerbl, and G. Drettakis, “Onthe-fly reconstruction for large-scale novel view synthesis from unposed images,” ACM Transactions on Graphics (TOG), vol. 44, no. 4, pp. 1–14, 2025.

[29] Z. Ye, W. Li, S. Liu, P. Qiao, and Y. Dou, “Absgs: Recovering fine details in 3d gaussian splatting,” in Proceedings of the 32nd ACM international conference on multimedia, 2024, pp. 1053–1061.

[30] S. Ren, T. Wen, Y. Fang, and B. Lu, “Fastgs: Training 3d gaussian splatting in 100 seconds,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2026, pp. 26 094–26 103.

[31] A. Guedon and V. Lepetit, “Sugar: Surface-aligned gaussian splatting´ for efficient 3d mesh reconstruction and high-quality mesh rendering,” CVPR, 2024.

[32] B. Huang, Z. Yu, A. Chen, A. Geiger, and S. Gao, “2d gaussian splatting for geometrically accurate radiance fields,” in SIGGRAPH 2024 Conference Papers. Association for Computing Machinery, 2024.

[33] D. Chen, H. Li, W. Ye, Y. Wang, W. Xie, S. Zhai, N. Wang, H. Liu, H. Bao, and G. Zhang, “Pgsr: Planar-based gaussian splatting for efficient and high-fidelity surface reconstruction,” arXiv preprint arXiv:2406.06521, 2024.

[34] Z. Yu, T. Sattler, and A. Geiger, “Gaussian opacity fields: Efficient adaptive surface reconstruction in unbounded scenes,” ACM Transactions on Graphics, 2024.

[35] B. Zhang, C. Jiang, H. Li, S. Shen, and P. Tan, “Geometry-grounded gaussian splatting,” arXiv preprint arXiv:2601.17835, 2026.

[36] D. Gomez, A. Guedon, N. Maruani, B. Gong, and M. Ovsjanikov,´ “From blobs to spokes: High-fidelity surface reconstruction via oriented gaussians,” 2026. [Online]. Available: https://arxiv.org/abs/2604.07337

[37] D. Charatan, S. L. Li, A. Tagliasacchi, and V. Sitzmann, “pixelsplat: 3d gaussian splats from image pairs for scalable generalizable 3d reconstruction,” in 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2024, pp. 19 457–19 467.

[38] H. Xu, S. Peng, F. Wang, H. Blum, D. Barath, A. Geiger, and M. Pollefeys, “Depthsplat: Connecting gaussian splatting and depth,” in CVPR, 2025.

[39] S. Wang, V. Leroy, Y. Cabon, B. Chidlovskii, and J. Revaud, “Dust3r: Geometric 3d vision made easy,” in CVPR, 2024.

[40] Y. Cabon, L. Stoffl, L. Antsfeld, G. Csurka, B. Chidlovskii, J. Revaud, and V. Leroy, “Must3r: Multi-view network for stereo 3d reconstruction,” in 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2025, pp. 1050–1060.

[41] Y. Liu, C. Luo, Z. Tang, J. Peng, and Z. Zhang, “Vggt-x: When vggt meets dense novel view synthesis,” arXiv preprint arXiv:2509.25191, 2025.

[42] B. Smart, C. Zheng, I. Laina, and V. A. Prisacariu, “Splatt3r: Zero-shot gaussian splatting from uncalibrated image pairs,” arXiv preprint arXiv:2408.13912, 2024. [Online]. Available: https://arxiv.org/abs/2408. 13912

[43] I. Kim, C. Kim, M. Bae, M. Joo, and H. J. Kim, “F4splat: Feed-forward predictive densification for feed-forward 3d gaussian splatting,” 2026. [Online]. Available: https://arxiv.org/abs/2603.21304

[44] M. Oquab, T. Darcet, T. Moutakanni, H. V. Vo, M. Szafraniec, V. Khalidov, P. Fernandez, D. Haziza, F. Massa, A. El-Nouby, R. Howes, P.-Y. Huang, H. Xu, V. Sharma, S.-W. Li, W. Galuba, M. Rabbat, M. Assran, N. Ballas, G. Synnaeve, I. Misra, H. Jegou, J. Mairal, P. Labatut, A. Joulin, and P. Bojanowski, “Dinov2: Learning robust visual features without supervision,” 2023.

[45] R. Ranftl, A. Bochkovskiy, and V. Koltun, “Vision transformers for dense prediction,” ArXiv preprint, 2021.

[46] S. Contributors, “Spconv: Spatially sparse convolution library,” https: //github.com/traveller59/spconv, 2022.

[47] C. Sun, J. Choe, C. Loop, W. Ma, and Y. F. Wang, “Sparse voxels rasterization: Real-time high-fidelity radiance field rendering,” in CVPR, 2025.

[48] B. Miller, H. Chen, A. Lai, and I. Gkioulekas, “Objects as volumes: A stochastic geometry view of opaque solids,” IEEE, 2023.

[49] A. Raistrick, L. Lipson, Z. Ma, L. Mei, M. Wang, Y. Zuo, K. Kayan, H. Wen, B. Han, Y. Wang, A. Newell, H. Law, A. Goyal, K. Yang, and J. Deng, “Infinite photorealistic worlds using procedural generation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 12 630–12 641.

[50] C. Yeshwanth, Y.-C. Liu, M. Nießner, and A. Dai, “Scannet++: A highfidelity dataset of 3d indoor scenes,” in 2023 IEEE/CVF International Conference on Computer Vision (ICCV). IEEE, 2023, pp. 12–22.

[51] G. Baruch, Z. Chen, A. Dehghan, T. Dimry, Y. Feigin, P. Fu, T. Gebauer, B. Joffe, D. Kurz, A. Schwartz et al., “Arkitscenes: A diverse real-world dataset for 3d indoor scene understanding using mobile rgb-d data,” arXiv preprint arXiv:2111.08897, 2021.

[52] L. Ling, Y. Sheng, Z. Tu, W. Zhao, C. Xin, K. Wan, L. Yu, Q. Guo, Z. Yu, Y. Lu et al., “Dl3dv-10k: A large-scale scene dataset for deep learningbased 3d vision,” in 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2024, pp. 22 160–22 169.

[53] J. Straub, T. Whelan, L. Ma, Y. Chen, E. Wijmans, S. Green, J. J. Engel, R. Mur-Artal, C. Ren, S. Verma, A. Clarkson, M. Yan, B. Budge, Y. Yan, X. Pan, J. Yon, Y. Zou, K. Leon, N. Carter, J. Briales, T. Gillingham, E. Mueggler, L. Pesqueira, M. Savva, D. Batra, H. M. Strasdat, R. D. Nardi, M. Goesele, S. Lovegrove, and R. Newcombe, “The Replica dataset: A digital replica of indoor spaces,” arXiv preprint arXiv:1906.05797, 2019.

[54] J. Sturm, N. Engelhard, F. Endres, W. Burgard, and D. Cremers, “A benchmark for the evaluation of rgb-d slam systems,” in 2012 IEEE/RSJ international conference on intelligent robots and systems. IEEE, 2012, pp. 573–580.

[55] A. Knapitsch, J. Park, Q.-Y. Zhou, and V. Koltun, “Tanks and temples: Benchmarking large-scale scene reconstruction,” ACM Transactions on Graphics, vol. 36, no. 4, 2017.

[56] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, “Image quality assessment: from error visibility to structural similarity,” IEEE Trans Image Process, vol. 13, no. 4, 2004.

[57] R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang, “The unreasonable effectiveness of deep features as a perceptual metric,” IEEE, 2018.

[58] Aanaes, HenrikJensen, R. RamsbolVogiatzis, GeorgeTola, EnginDahl, and A. Bjorholm, “Large-scale data for multiple-view stereopsis,” International Journal of Computer Vision, vol. 120, no. 2, 2016.

[59] S. Umeyama, “Least-squares estimation of transformation parameters between two point patterns,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 13, no. 4, pp. 376–380, 1991.