# SG-Splat: Stable Geometry Learning under Pose Uncertainty for Pose-Free 3D Gaussian Splatting

Xiaohan Lu, Jiaqi Zhang, Weijia Jiang, Siwei Ma, Fellow, IEEE

Abstract—3D Gaussian Splatting (3DGS) has become a promising representation for real-time novel view synthesis and 3D scene reconstruction. For practical deployment, recent studies have explored pose-free feed-forward 3DGS, which reconstructs Gaussian representations directly from unposed input images. However, these methods remain sensitive to residual pose errors, which perturb canonical-space Gaussian transformation, introducing noisy supervision for geometry and attribute learning. To address this problem, we present SG-Splat, a pose-free feedforward 3DGS framework that improves robustness to pose uncertainty and stabilizes the learning of Gaussian primitives. SG-Splat decouples geometry estimation and Gaussian attribute prediction into two dedicated branches, reducing error coupling between pose-sensitive geometry and Gaussian attributes. We further introduce a coarse-to-fine refinement pipeline, consisting of Flow-Guided Pose Refinement (FGPR) and Residual-Guided Geometry Correction (RGC), to improve geometric alignment before Gaussian attribute reconstruction. Specifically, FGPR refines camera poses with flow-based correlation features and differentiable bundle adjustment. RGC then corrects initial point maps using depth residuals derived from FGPR, producing more reliable Gaussian centers. To stabilize training, we introduce a dual-path strategy that mixes ground-truth and predicted poses for canonical transformation. It provides clean supervision for Gaussian attributes while improving geometry robustness to residual pose errors. Extensive experiments on DL3DV and RealEstate10K show that SG-Splat achieves state-of-the-art performance in both novel view synthesis and pose estimation.

Index Terms—Feed-forward 3D Gaussian Splatting, Pose-free, Novel View synthesis, Pose Estimation.

## I. INTRODUCTION

3D Gaussian Splatting (3DGS) [4] has become a powerful representation for real-time photorealistic rendering and highfidelity scene reconstruction, with growing potential in world modeling [5], autonomous driving [6], and virtual reality [7]. However, conventional 3DGS is limited by its reliance on per-scene optimization and calibrated poses, which limits scalable and efficient 3D reconstruction. This has motivated recent interest in pose-free feed-forward 3DGS, which seeks to predict camera poses and Gaussian parameters directly from input images in a single forward pass. Early pose-free feed-forward 3DGS methods [8], [9] predict 3D Gaussian primitives directly in a canonical space, thereby avoiding the explicit propagation of pose estimation errors during

![](images/4c7e8ebb157ebceac4ba0826bac007f0a874a3f6019449771b12b75d94dabf94.jpg)  
Fig. 1. Comparison of pose usage strategies in pose-free feed-forward 3DGS training. (a) AnySplat [1] uses predicted poses with pose supervision, maintaining consistency between training and inference but suffering from perturbations caused by pose errors. (b) WorldMirror [2] uses ground-truth poses with pose supervision, stabilizing training but introducing inconsistency between training and inference. (c) E-RayZer [3] relies on predicted poses without pose supervision, making optimization sensitive to pose errors. (d) Our method uses mixed poses with pose supervision, balancing stable learning and robustness to residual pose errors.

Gaussian aggregation. These methods have shown promising performance under sparse-view settings, typically with two to four input views. However, directly constructing a canonical Gaussian representation becomes increasingly challenging as the number of input views grows [10], due to the difficulty of maintaining cross-view geometric consistency without explicit pose-guided aggregation.

To scale to multi-view settings, recent methods such as AnySplat [1] and WorldMirror [2] predict per-view Gaussian primitives in camera coordinate systems and transform them into a canonical space using camera poses of each view, as illustrated in Fig. 1(a) and (b). At inference time, this transformation must rely on predicted poses, whose errors directly affect the alignment of Gaussian primitives across views. During training, this transformation becomes nontrivial, as the poses used for transformation affect both optimization stability and the consistency between training and inference. AnySplat uses predicted poses for transformation during training, improving consistency with inference but introducing scene-dependent pose perturbations that destabilize the learning of geometry and Gaussian attributes. In contrast, WorldMirror uses ground-truth poses for canonical transformation, which stabilizes Gaussian parameter learning but introduces a discrepancy between training and inference. In addition, both methods directly couple pose supervision with Gaussian reconstruction, introducing competing optimization objectives. This coupling becomes especially detrimental when residual pose errors perturb the canonical transformation used for rendering supervision. To remove this coupling, recent methods [3], [11] adopt a self-supervised formulation. As shown in Fig. 1(c), they discard explicit pose supervision and optimize Gaussian reconstruction solely through rendering supervision. Although this design removes the need for pose supervision, it still relies on predicted poses for canonical transformation during training. Consequently, pose errors still perturb the optimization of Gaussian geometry and attributes. In addition, without direct 3D constraints, these methods may produce visually plausible but geometrically unreliable reconstructions [12], restricting their applicability in geometrysensitive tasks.

These observations reveal a key challenge: how can posefree feed-forward 3DGS effectively exploit pose supervision during training while maintaining stable learning under residual pose errors? Although YoNoSplat [13] mixes predicted and ground-truth poses during training, predicted poses are sampled with a low probability. As a result, this strategy only weakly exposes the model to inference-time pose errors and still allows noisy supervision to affect Gaussian attributes when predicted poses are selected.

To address this challenge, we propose SG-Splat, a pose-free feed-forward 3DGS framework with decoupled geometry and attribute branches, as shown in Fig. 1(d). This design separates geometry and camera estimation from Gaussian attribute prediction, reducing the interference between pose errors and attribute learning. We further introduce a progressive refinement pipeline with Flow-Guided Pose Refinement (FGPR) and Residual-Guided Geometry Correction (RGC). FGPR refines the initial camera poses using flow-based correlation features and differentiable bundle adjustment (DBA). RGC corrects the initial point maps using depth residuals produced by FGPR and intermediate geometry features. The refined poses stabilize canonical transformations, and the corrected point maps improve Gaussian center accuracy. Together, they reduce the discrepancy between training and inference and improve the consistency between the refined camera poses and the Gaussian centers.

In addition, we design a pose-aware dual-path training strategy that prevents the attribute branch from overfitting to alignments corrupted by residual pose errors. Specifically, either ground-truth poses or predicted poses are selected to transform Gaussian primitives into a canonical coordinate system during training. When ground-truth poses are used, the model learns geometry reconstruction and Gaussian attribute prediction under reliable alignment. When predicted poses are used, we mainly optimize the geometry branch and pose estimation modules, preventing the attribute branch from overfitting to residual pose errors. This asymmetric optimization stabilizes Gaussian attribute learning and improves geometry robustness under pose uncertainty. Our contributions are summarized as follows:

• We propose SG-Splat, a pose-free feed-forward 3DGS framework with a geometry-attribute decoupled architecture. By separating geometry and camera estimation from Gaussian attribute prediction, the framework suppresses error propagation from inaccurate pose estimates and improves the reliability of Gaussian parameter generation.

• We introduce a coarse-to-fine refinement pipeline consisting of FGPR and RGC to enhance pose and Gaussian alignment. FGPR iteratively refines camera poses by combining flow-based correlation features with differentiable bundle adjustment. RGC then corrects local point maps using depth residuals from FGPR, improving the consistency between refined camera poses and Gaussian centers.

• We develop a pose-aware dual-path training strategy that integrates mixed-pose canonical transformation with asymmetric branch optimization. This strategy uses ground-truth poses to provide reliable attribute supervision, while using predicted poses to regularize geometry learning against scene-dependent pose errors.

## II. RELATED WORK

A. Novel View Synthesis with Feed-Forward 3D Gaussian Splatting

Early image-based rendering methods [14]–[16] synthesize novel views by resampling or blending images captured from nearby viewpoints. These methods demonstrated the possibility of photorealistic view synthesis without fully explicit 3D reconstruction, but they often require dense image capture, calibrated cameras, or approximate geometry to handle large viewpoint changes and occlusions. With the advent of deep learning, learning-based NVS methods [17], [18]learn the geometry of scenes and appearance priors from data, improving flexibility under sparse-view inputs. NeRF [19] represents a scene as a continuous radiance field optimized from posed images and renders novel views via volumetric ray marching, which incurs substantial computational cost. To improve efficiency, follow-up works [20]–[22] introduced compact encodings or hybrid scene representations to accelerate rendering, but they still struggle to achieve real-time performance. Motivated by this limitation, 3DGS [4] and its variants [23]–[25] adopt explicit Gaussian primitives for scene representation, achieving high-fidelity novel view synthesis with real-time rendering efficiency. However, both NeRF and 3DGS pipelines typically rely on per-scene optimization. Recent methods extend 3DGS to sparse input views by introducing diffusion priors [26] or geometric constraints [27]. This has motivated recent feed-forward methods for NeRF [28]–[30] as well as feed-forward 3DGS methods [31]–[34]. More recently, DepthSplat [35] and ProSplat [36] extend this paradigm to sparse multi-view or wide-baseline settings. In addition, recent works [37]–[39] decouple Gaussian primitives from pixelaligned locations, resulting in a more flexible representation.

![](images/9145127a6d68be556f50d2cec17014f2d8a50c24f04fdae89fe25771812dabfe.jpg)  
Fig. 2. Overview of the pipeline. The input images are processed by separate geometry and attribute branches. The geometry branch predicts initial local point maps and camera poses through the point head and camera head, while the attribute branch predicts the remaining Gaussian attributes. Flow-Guided Pose Refinement (FGPR) refines the initial poses and produces depth residuals, which are used by Residual-Guided Geometry Correction (RGC) to correct the local Gaussian centers $\mu ^ { \mathrm { l o c a l } }$ . The corrected centers are transformed into the canonical space with the refined poses and combined with the predicted attributes to form the final Gaussian representation.

Nevertheless, most of them still assume known camera poses or rely on pose-conditioned geometric cues, which limits their applicability to casually captured, unposed image collections. In contrast, our work targets pose-free feed-forward 3DGS, where camera poses, geometry, and Gaussian attributes must be jointly predicted under residual pose uncertainty.

## B. 3D Foundation Models for Geometry and Pose Priors

Recent 3D foundation models have substantially advanced feed-forward geometry and pose estimation from unposed images. DUSt3R [40] reformulates multi-view reconstruction as point map regression from unconstrained image pairs. MASt3R [41] and its variants [42], [43] further improve dense matching accuracy and geometric consistency. Building on this line of work, VGGT [44], $\pi ^ { 3 }$ [45], and Depth Anything 3 (DA3) [46] directly predict camera poses, point maps, depth maps, and geometric tracks from a few to hundreds of views in a single forward pass, offering scalable geometry priors for downstream applications.

These foundation models have also been increasingly adopted in feed-forward 3DGS. Splatt3R [9] and NoPoSplat [8] build upon MASt3R to obtain geometric priors and predict Gaussians in a global coordinate system from unposed images. AnySplat [1] uses VGGT as a teacher to supervise pose and geometry prediction during training. YoNoSplat [13] extends this paradigm with a $\pi ^ { 3 } .$ based architecture and introduces an intrinsic condition embedding to handle unknown camera intrinsics. AirSplat [47] leverages DA3 as a frozen geometry foundation model and trains a Gaussian generation module on top of its priors. Collectively, these methods demonstrate that 3D foundation models have become a powerful source of geometric priors for pose-free feedforward 3DGS.

## C. Pose-Free Feed-Forward 3D Gaussian Splatting

Pose-free feed-forward 3DGS reconstructs Gaussian representations directly from unposed images, avoiding camera calibration at inference time. Existing methods mainly fall into two categories. The first line exploits 3D foundation models to provide geometry and camera priors. Methods built on DUSt3R [48] or MASt3R [8]–[10], [49], [50] recover geometry and camera poses before Gaussian prediction. However, these methods are mostly tailored to sparse input views, which limits their scalability to denser multi-view reconstruction. Recent methods [1], [13], [47], [51] improve scalability by using visual geometry foundation models that jointly predict point maps and camera poses in a single forward pass [44]– [46]. However, their Gaussian reconstruction remains sensitive to pose accuracy, because residual pose errors can perturb the transformation of Gaussian centers and introduce noisy supervision into the optimization of Gaussian geometry and attributes.

Another line of work removes 3D supervision and trains pose-free models with self-supervised rendering objectives [3], [11], [52], [53]. Although these methods avoid explicit pose supervision, pose estimation, geometry prediction, and Gaussian attribute generation are optimized in a coupled manner through rendering losses. Consequently, pose and geometry errors may be absorbed by Gaussian attributes, leading to visually plausible but geometrically unreliable reconstructions [12].

In contrast, our work focuses on the practical setting where explicit 3D supervision is available. We study how to effectively exploit 3D foundation priors for dense-view posefree feed-forward 3DGS while preventing pose-corrupted supervision from destabilizing Gaussian geometry and attribute learning.

## III. METHOD

## A. Problem Formulation

Given N unposed input views $\boldsymbol { \mathcal { T } } = \{ \mathbf { I } _ { i } \} _ { i = 1 } ^ { N }$ , where $\mathbf { I } _ { i } \in$ $\mathbb { R } ^ { H \times W \times 3 }$ , the goal is to reconstruct a set of 3D Gaussian

![](images/29a398d0a5907907ca7598e842e6dd9309827761b87487ad331074051f23cf57.jpg)  
Fig. 3. Flow-Guided Pose Refinement (FGPR). At each iteration, FGPR uses the current pose and geometry estimates to build flow-based correlation features and optical flow of camera motion. Together with context features, these cues are fed into an update operator to predict correspondence corrections $\Delta \mathbf { p } _ { i j }$ . The corrected correspondences $\mathbf { p } _ { i j } ^ { * }$ are optimized by differentiable bundle adjustment (DBA) to jointly update camera poses and depths. Over T iterations, FGPR progressively improves pose accuracy and produces depth residuals for subsequent geometry correction.

primitives for novel view synthesis. Each primitive is defined as

$$
\mathbf { g } _ { i , j } = ( \pmb { \mu } _ { i , j } , \mathbf { c } _ { i , j } , \mathbf { r } _ { i , j } , \mathbf { s } _ { i , j } , \alpha _ { i , j } ) ,\tag{1}
$$

and the complete Gaussian representation is written as

$$
\mathcal { G } = \{ \mathbf { g } _ { i , j } \} _ { i = 1 , \dots , N ; j = 1 , \dots , H W } .\tag{2}
$$

The components $\pmb { \mu } \in \mathbb { R } ^ { 3 } , \mathbf { r } \in \mathbb { R } ^ { 4 } , \mathbf { s } \in \mathbb { R } ^ { 3 } , \alpha \in \mathbb { R } _ { + }$ , and $\mathbf { c } ~ \in ~ \mathbb { R } ^ { 3 \times ( k + 1 ) ^ { 2 } }$ correspond to the 3D position, quaternion rotation, scaling factor, opacity, and spherical harmonics (SH) coefficients of degree $k ,$ respectively. We assume known camera intrinsics, which are directly obtained from the capture device. In addition, the model predicts the camera poses $\{ \mathbf { P } _ { i } \} _ { i = 1 } ^ { N }$ for all input views. Formally, the feed-forward model $f _ { \theta }$ parameterized by θ learns the mapping

$$
f _ { \theta } : \{ \mathbf { I } _ { i } \} _ { i = 1 } ^ { N } \mapsto \left( \mathcal { G } , \{ \mathbf { P } _ { i } \} _ { i = 1 } ^ { N } \right) .\tag{3}
$$

## B. Pipeline

As illustrated in Fig. 2, SG-Splat decouples geometry and attribute modeling into two independent branches, each consisting of an encoder and a decoder. Given the input images, the two branches produce a geometry token $\mathbf { z } _ { i } ^ { g } ,$ , a camera token ${ \bf z } _ { i } ^ { c } ,$ , and an attribute token $\mathbf { z } _ { i } ^ { a }$ for the i-th input view. The geometry token $\mathbf { z } _ { i } ^ { g }$ and attribute token $\mathbf { z } _ { i } ^ { a }$ are both of size $\overline { { \mathbb { R } } } ^ { ( l + 5 ) \times \check { d } }$ , while the camera token $\mathbf { z } _ { i } ^ { c }$ is of size $\mathbb { R } ^ { ( l / 4 + 5 ) \times d / 4 }$ where $\begin{array} { r } { l = \frac { 4 H W } { p ^ { 2 } } } \end{array}$ denotes the number of image tokens, and the additional five tokens are register tokens following $\pi ^ { 3 }$ [45]. Here, $p = 1 4$ is the patch size and $d = 1 0 2 4$ is the token dimension.

The geometry token and the camera token are then fed into a point head and a camera head to predict an initial local point map $\hat { \mathbf { M } } _ { i } ^ { l o c a l }$ and an initial camera pose $\hat { { \bf P } } _ { i }$ for each input view, respectively. The predicted pose is further refined by the proposed Flow-Guided Pose Refinement (FGPR) module, which iteratively updates the pose for $T$ rounds and produces the refined pose $\tilde { \mathbf { P } } _ { i }$ together with depth residual cues for geometric correction. The initial local point map $\hat { \mathbf { M } } _ { i } ^ { l o c a l }$ is then refined by the Residual-Guided Geometry Correction module to obtain the refined local point map $\tilde { \mathbf { M } } _ { i } ^ { l o c a l }$ . Subsequently, $\tilde { \mathbf { M } } _ { i } ^ { l o c a l }$ is transformed from the local camera coordinate system into the shared canonical space, yielding the global Gaussian centers $\{ \mu _ { i , j } ^ { g l o b a l } \} _ { j = 1 } ^ { H W }$

In parallel, the attribute token is fed into a Gaussian head to predict the remaining Gaussian attributes, which are combined with the transformed Gaussian centers to form the complete Gaussian representation:

$$
\mathcal { G } = \left\{ \left( \mu _ { i , j } ^ { g l o b a l } , \mathbf { c } _ { i , j } , \mathbf { r } _ { i , j } , \mathbf { s } _ { i , j } , \alpha _ { i , j } \right) \right\} _ { i = 1 , \dots , N ; j = 1 , \dots , H W }\tag{4}
$$

## C. Flow-Guided Pose Refinement (FGPR)

Inspired by DROID-SLAM [54], we adopt an iterative pose refinement module to further correct the initial camera parameters $\hat { \mathbf { P } } _ { i } ,$ as shown in Fig. 4. Given the input images $\mathbf { I } _ { i } ,$ initial camera parameters $\hat { \mathbf { P } } _ { i } ,$ and initial depth estimates $\hat { \bf D } _ { i }$ from the z-axis $\begin{array} { r } { \hat { \bf M } _ { i } , } \end{array}$ the module constructs low-based correlation features and optical flow of camera motion to refine poses iteratively.

a) Inverse Depth Normalization: To improve robustness to scene-scale variation, we use normalized inverse depth. First, we convert depth to inverse depth:

$$
\hat { \mathbf { d } } _ { i } ^ { r a w } = \frac { 1 } { \hat { \mathbf { D } } _ { i } } , \qquad \hat { \mathbf { d } } _ { i } ^ { r a w } \in \mathbb { R } ^ { H ^ { \prime } \times W ^ { \prime } } ,\tag{5}
$$

where $H ^ { \prime } = H / 8$ and $W ^ { \prime } = W / 8 .$ . We then normalize the inverse depth using only valid values,

$$
\hat { \mathbf { d } } _ { i } = \frac { \hat { \mathbf { d } } _ { i } ^ { r a w } } { \mathrm { m e a n } \big ( \{ \hat { \mathbf { d } } _ { i } ^ { r a w } [ \hat { \mathbf { d } } _ { i } ^ { r a w } > \tau ] \} _ { i = 1 } ^ { N } \big ) } ,\tag{6}
$$

where $\tau ~ = ~ 0 . 0 5$ filters out extremely small inverse-depth values corresponding to very distant points. This normalization maintains a consistent numerical scale for the geometric input, thereby improving the robustness of pose refinement.

b) Feature Extraction: For feature extraction, input images are processed by a feature encoder and a context encoder, producing two dense feature maps at $1 / 8$ image resolution. The feature encoder is used to compute 4D correlation volumes [55] $C _ { i j } \in \mathbb { R } ^ { H ^ { \prime } \times W ^ { \prime } \times H ^ { \prime } \times W ^ { \prime } }$ for each pair of overlapping images $( i , j )$ . Then, a correlation lookup operator takes a $H ^ { \prime } \times W ^ { \prime }$ grid of query coordinates and retrieves the values within a local square neighborhood of radius $r$ from $C _ { i j } ,$ producing $\mathbf { C o r } _ { i j } \stackrel { \cdot } { \in } \mathbb { R } ^ { H ^ { \prime } \times W ^ { \prime } \times ( 2 r + 1 ) ^ { 2 } }$ . The context features are injected into the update network during iterative refinement.

c) Input: Under the current pose estimates $\hat { { \bf P } } _ { i }$ and $\hat { \mathbf { P } } _ { j }$ and the inverse depth $\hat { \mathbf { d } } _ { i }$ , we warp the pixel grid $\mathbf { p } _ { i }$ from view i to view $j$ to obtain the correspondence field:

$$
{ \bf p } _ { i j } = \Pi \Bigl ( \hat { \bf P } _ { j } ^ { - 1 } \circ \hat { \bf P } _ { i } \circ \Pi ^ { - 1 } ( { \bf p } _ { i } , \hat { \bf d } _ { i } ) \Bigr ) ,\tag{7}
$$

where $\mathbf { p } _ { i }$ is the pixel grid of view $i ,$ and $\Pi ^ { - 1 }$ and Π denote back-projection and projection, respectively. Based on the correspondence field $\mathbf { p } _ { i }$ , we derive the optical flow of camera motion $\mathbf { F } _ { i }$ as the displacement between the projected and original pixel locations. Consequently, the update module takes the context feature $\mathbf { h } _ { \mathrm { c t x } } .$ the correlation feature $\mathbf { C o r } _ { i j }$ , and the optical flow $\mathbf { F } _ { i }$ as inputs.

d) Update: The update operator is implemented as a $3 { \times } 3$ convolutional GRU initialized with the context feature $\mathbf { h } _ { \mathrm { c t x } } .$ . To improve feature fusion, the correlation feature $\mathbf { C o r } _ { i j }$ and the optical flow $\mathbf { F } _ { i }$ are first encoded by two convolutional layers before being injected into the GRU, while the context feature $\mathbf { h } _ { \mathrm { c t x } }$ is incorporated through element-wise addition.

In iteration t, the operator updates the hidden state and predicts a correction term $\Delta \mathbf { p } _ { i j } ^ { ( t ) }$ for the current correspondence field $\mathbf { p } _ { i j } ^ { ( t ) }$

$$
\Delta \mathbf { p } _ { i j } ^ { ( t ) } = \mathrm { U p d a t e } \left( \mathbf { h } _ { \mathrm { c t x } } , \mathbf { C o r } _ { i j } ^ { ( t ) } , \mathbf { F } _ { i } ^ { ( t ) } \right) ,\tag{8}
$$

$$
\mathbf { p } _ { i j } ^ { * ( t ) } = \mathbf { p } _ { i j } ^ { ( t ) } + \Delta \mathbf { p } _ { i j } ^ { ( t ) } .\tag{9}
$$

The corrected correspondence field is then fed into the DBA [56] module together with the current pose and inverse depth estimates $\tilde { \mathbf { P } } _ { i } ^ { ( t ) }$ and $\tilde { \mathbf { d } } _ { i } ^ { ( t ) }$ , to jointly update them as $\tilde { \mathbf { P } } _ { i } ^ { ( t + 1 ) }$ and $\tilde { \mathbf { d } } _ { i } ^ { ( t + 1 ) }$ . After $T$ iterations, the final refined pose and depth are given by

$$
\tilde { { \bf P } } _ { i } = \tilde { { \bf P } } _ { i } ^ { ( T ) } , \qquad \tilde { { \bf D } } _ { i } = \frac { 1 } { \tilde { \bf d } _ { i } ^ { ( T ) } } .\tag{10}
$$

The depth residual used for subsequent geometry correction is defined as

$$
\Delta \hat { \mathbf { D } } _ { i } = \tilde { \mathbf { D } } _ { i } - \hat { \mathbf { D } } _ { i } ,\tag{11}
$$

where $\hat { \bf D } _ { i }$ denotes the initial depth estimate.

## D. Residual-Guided Geometry Correction (RGC).

To further improve the coarse local geometry, we introduce a lightweight Residual-Guided Geometry Correction (RGC) module, as shown in Fig. 4. RGC takes the initial local point map together with the depth residual from FGPR as input, so that local geometry correction is explicitly guided by residual geometric cues after pose refinement.

For each input view, the initial point map $\hat { \mathbf { M } } _ { i } \in \mathbb { R } ^ { H \times W \times 3 } \ \mathrm { i s }$ concatenated with the FGPR depth residual $\Delta \bar { \mathbf { D } } _ { i } \in \mathbb { R } ^ { H \times W \times 1 }$ and converted into patch tokens $\mathbf { z } _ { i } ^ { r } \in \mathbb { R } ^ { \frac { 4 H W } { p ^ { 2 } } \times D }$ through a patch embedding layer:

$$
\mathbf { z } _ { i } ^ { r } = \mathrm { P a t c h E m b e d } \left( \mathrm { C o n c a t } ( \hat { \mathbf { M } } _ { i } , \Delta \hat { \mathbf { D } } _ { i } ) \right) ,\tag{12}
$$

where $p = 1 4$ denotes the patch size and $D = 1 0 2 4$ is the token dimension. The residual tokens are then combined with the geometry tokens $\mathbf { z } _ { i } ^ { g }$ from the geometry branch, which share the same spatial resolution. We concatenate them along the channel dimension and use an MLP to obtain the correction tokens:

![](images/3a37b6560e78ed571b2229c40f9ebbc94d66883a69caa6307bd9c82f57614c2f.jpg)  
Fig. 4. Residual-Guided Geometry Correction (RGC). RGC uses the residual depth map ∆D<sup>ˆ</sup> <sub>i</sub> from FGPR to correct the initial local point map $\hat { { \bf M } } _ { i }$ . The residual depth map and the initial point map are embedded and fused with geometry tokens to predict a bounded multiplicative correction, producing the corrected point map $\tilde { \mathbf { M } } _ { i }$

$$
\begin{array} { r } { \mathbf { z } _ { i } ^ { \delta } = \mathrm { M L P } ( \mathrm { C o n c a t } _ { c } ( \mathbf { z } _ { i } ^ { r } , \mathbf { z } _ { i } ^ { g } ) ) , } \end{array}\tag{13}
$$

where $\mathbf { z } _ { i } ^ { \delta } \in \mathbb { R } ^ { \frac { 4 H W } { p ^ { 2 } } \times D }$ . The correction tokens are decoded into a dense point-wise correction field $\Delta \hat { \mathbf { M } } _ { i } \in \mathbb { R } ^ { H \times W \times 3 }$ through linear projection and pixel-shuffle upsampling:

$$
\Delta \hat { \mathbf { M } } _ { i } = \mathrm { P S } \big ( \mathrm { L i n e a r } \big ( \mathbf { z } _ { i } ^ { \delta } \big ) \big ) ,\tag{14}
$$

where PS(·) denotes pixel-shuffle. Rather than directly predicting additive offsets, the module estimates a bounded multiplicative correction,

$$
\mathbf { S } _ { i } = 1 + \beta \operatorname { t a n h } ( \Delta \hat { \mathbf { M } } _ { i } ) ,\tag{15}
$$

which is applied element-wise to the initial point map:

$$
\begin{array} { r } { \tilde { \mathbf { M } } _ { i } = \hat { \mathbf { M } } _ { i } \odot \mathbf { S } _ { i } , } \end{array}\tag{16}
$$

where ⊙ denotes element-wise multiplication, and $\beta = 0 . 2$ constrains the correction magnitude to avoid excessive corrections that could disrupt the coarse geometric structure.

This design preserves the coarse geometric prior while enabling spatially adaptive geometry correction. Compared with direct offset regression, the multiplicative formulation yields more stable updates and is less prone to over-correction. By conditioning the refinement on FGPR depth residuals, the refined point map becomes better aligned with the corrected camera poses, providing more reliable Gaussian centers for subsequent 3D Gaussian prediction.

![](images/adb5a957c29ec0bc68383bd3b79b951ef823c4f1430bc9e15f946057dc0a4759.jpg)  
Fig. 5. Dual-Path Training Strategy. The refined predicted pose and the ground-truth (GT) pose are sampled with equal probability. The selected pose is used to transform local Gaussian centers into the canonical space for rendering. When the ground-truth pose is selected, reliable rendering supervision updates both the geometry and attribute branches. When the predicted pose is selected, rendering gradients are detached from the attribute branch and only the optimize geometry branch. This asymmetric optimization stabilizes attribute learning while improving geometry robustness under residual pose errors.

## E. Dual-Path Training Strategy

We adopt a dual-branch design for SG-Splat, where the geometry branch estimates camera poses and local Gaussian centers, and the attribute branch predicts Gaussian attributes. This separation reduces error propagation from inaccurate poses and improves the reliability of Gaussian parameter generation. However, 3DGS rendering requires transforming local Gaussian centers into a shared canonical space, making cross-view alignment sensitive to pose errors. To mitigate this issue, we introduce a mixed-pose canonical transformation with asymmetric optimization, as shown in Fig. 5.

a) Mixed-pose canonical transformation: The refined point map $\tilde { \mathbf { M } } _ { i } \in \mathbb { R } ^ { H \times W \times 3 }$ provides local Gaussian centers for the i-th input view. After flattening the spatial dimensions, we obtain $\{ \grave { \mu } _ { i , j } ^ { \mathrm { l o c a l } } \} _ { j = 1 } ^ { H W }$ . During training, the pose used for canonical transformation is sampled from either the groundtruth pose or the refined predicted pose:

$$
\bar { \mathbf { P } } _ { i } = \left\{ \begin{array} { l l } { \mathbf { P } _ { i } ^ { * } , } & { z = 1 , } \\ { \tilde { \mathbf { P } } _ { i } , } & { z = 0 , } \end{array} \right. \quad z \sim \mathrm { B e r n o u l l i } ( 0 . 5 ) ,\tag{17}
$$

where $\mathbf { P } _ { i } ^ { * }$ and $\tilde { \mathbf { P } } _ { i }$ denote the ground-truth pose and refined predicted pose, respectively. The binary variable z is sampled once per training scene and shared across all input views. Each local center is transformed into the canonical space by

$$
\pmb { \mu } _ { i , j } ^ { \mathrm { g l o b a l } } = \bar { \bf P } _ { i } \circ \pmb { \mu } _ { i , j } ^ { \mathrm { l o c a l } } .\tag{18}
$$

The attribute branch predicts the remaining Gaussian attributes for each center:

$$
\left\{ ( \mathbf { c } _ { i , j } , \mathbf { r } _ { i , j } , \mathbf { s } _ { i , j } , \alpha _ { i , j } ) \right\} _ { j = 1 } ^ { H W } .\tag{19}
$$

Together with the transformed centers $\{ \mu _ { i , j } ^ { \mathrm { g l o b a l } } \} _ { j = 1 } ^ { H W }$ , these predicted parameters form the Gaussian primitives for rendering. For N input views, the model generates $N \times H \times W$ primitives in total.

b) Asymmetric optimization under mixed poses: The optimization rule depends on the sampled pose in Eq. (17). When $z = 1$ , ground-truth poses provide reliable canonical transformation and clean rendering supervision. We therefore update the full model, including the geometry branch, pose refinement module, and attribute branch. This path stabilizes Gaussian attribute learning under well-aligned transformations. When z = 0, refined predicted poses are used, matching the inference setting where ground-truth poses are unavailable. However, residual pose errors may misalign Gaussian centers across views and introduce noisy rendering supervision. To avoid corrupting attribute learning, we detach the rendering gradients from the attribute branch and use them only to optimize the geometry branch.

This asymmetric design combines the advantages of both paths. The ground-truth-pose path provides reliable supervision for Gaussian attribute prediction, while the predicted-pose path exposes the geometry branch to the pose distribution used during inference. By alternating between them, the proposed strategy reduces the discrepancy between training and inference and prevents the attribute branch from absorbing poseinduced noise.

## F. Training Objectives

a) Progressive Decoupled Training: Directly integrating FGPR and RGC into the SG-Splat pipeline leads to unstable optimization, as pose refinement, geometry correction, and Gaussian attribute prediction are optimized simultaneously. The resulting entanglement makes it difficult for the refinement modules to learn reliable corrections from noisy pose and geometry estimates. To address this issue, we adopt a progressive training strategy that decouples geometry and appearance optimization across four stages, including coupled pretraining, pose refinement, point refinement, and final joint optimization.

Stage I: Coupled pretraining. We first train a coupled base model without FGPR and RGC, including the encoder, decoder, point head, camera head, and Gaussian head. This stage provides a stable initialization for coarse geometry reconstruction and Gaussian attribute prediction.

Stage II: Pose refinement via FGPR. Based on the pretrained coupled model, we duplicate the encoder and decoder to initialize separate geometry and appearance branches. The point and camera heads are assigned to the geometry branch, while the Gaussian head is assigned to the appearance branch. All other modules are frozen, and only FGPR is optimized to learn stable pose refinement.

TABLE I  
QUANTITATIVE COMPARISONS ON DL3DV [57] UNDER DIFFERENT INPUT-VIEW SETTINGS. GT POSE INDICATES WHETHER CAMERA POSES ARE PROVIDED. OUR METHOD CONSISTENTLY ACHIEVES THE BEST PERFORMANCE FROM 12 TO 32 INPUT VIEWS IN BOTH POSED AND POSE-FREE SETTINGS.
<table><tr><td rowspan="2">Method</td><td rowspan="2">GT Pose</td><td colspan="3"> $1 2 \mathrm { v }$ </td><td colspan="3"> $1 8 \mathrm { v }$ </td><td colspan="3"> $2 4 \mathrm { v }$ </td><td colspan="3"> $3 2 \mathrm { v }$ </td></tr><tr><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td></tr><tr><td>MVSplat</td><td>V</td><td>21.149</td><td>0.696</td><td>0.236</td><td>19.452</td><td>0.632</td><td>0.281</td><td>19.761</td><td>0.641</td><td>0.272</td><td>17.912</td><td>0.492</td><td>0.403</td></tr><tr><td>DepthSplat</td><td>V</td><td>22.027</td><td>0.763</td><td>0.206</td><td>19.712</td><td>0.673</td><td>0.294</td><td>20.110</td><td>0.695</td><td>0.275</td><td>18.240</td><td>0.601</td><td>0.365</td></tr><tr><td>YoNoSplat</td><td>√</td><td>23.285</td><td>0.773</td><td>0.177</td><td>22.428</td><td>0.747</td><td>0.199</td><td>22.664</td><td>0.758</td><td>0.192</td><td>21.728</td><td>0.727</td><td>0.217</td></tr><tr><td>Ours</td><td>√</td><td>23.487</td><td>0.781</td><td>0.168</td><td>22.615</td><td>0.764</td><td>0.179</td><td>23.163</td><td>0.775</td><td>0.174</td><td>22.255</td><td>0.745</td><td>0.187</td></tr><tr><td>NoPoSplat</td><td></td><td>19.182</td><td>0.543</td><td>0.338</td><td>17.325</td><td>0.501</td><td>0.401</td><td>17.641</td><td>0.498</td><td>0.399</td><td>17.024</td><td>0.481</td><td>0.410</td></tr><tr><td>AnySplat</td><td></td><td>19.493</td><td>0.608</td><td>0.314</td><td>19.241</td><td>0.614</td><td>0.324</td><td>19.914</td><td>0.635</td><td>0.312</td><td>19.503</td><td>0.620</td><td>0.325</td></tr><tr><td>WorldMirror</td><td></td><td>21.629</td><td>0.673</td><td>0.184</td><td>20.190</td><td>0.684</td><td>0.248</td><td>20.664</td><td>0.697</td><td>0.241</td><td>20.369</td><td>0.689</td><td>0.248</td></tr><tr><td>YoNoSplat</td><td></td><td>23.151</td><td>0.759</td><td>0.182</td><td>22.334</td><td>0.733</td><td>0.207</td><td>22.433</td><td>0.735</td><td>0.205</td><td>21.525</td><td>0.702</td><td>0.231</td></tr><tr><td>Ours</td><td></td><td>23.791</td><td>0.790</td><td>0.156</td><td>22.982</td><td>0.764</td><td>0.178</td><td>23.190</td><td>0.772</td><td>0.171</td><td>22.430</td><td>0.744</td><td>0.193</td></tr></table>

Stage III: Point refinement via RGC. We then freeze the remaining components and optimize only the RGC, enabling geometry correction under the refined poses without interference from appearance optimization. During the refinement stages, local points are transformed into the canonical space using ground-truth poses, which removes stochastic perturbations from inaccurate pose predictions and provides stable supervision for pose and geometry refinement.

Stage IV: Joint optimization. Finally, we fine-tune the full dual-path architecture in an end-to-end manner. With stabilized geometry and pose estimates as initialization, this stage jointly optimizes geometry and appearance, improving reconstruction fidelity, view consistency, and robustness to residual pose errors.

b) Training Loss: Our model is a joint learning framework and trained with a multi-task loss function comprising two components: camera pose loss, and 3DGS rendering loss. As for the pose loss, we only explicitly supervise the refined poses produced by the FGPR module. Given the groundtruth poses $\{ \mathbf { P } _ { i } ^ { * } \} _ { i = 1 } ^ { N }$ , we supervise the refined pose estimates $\{ \tilde { \mathbf { P } } _ { i } ^ { ( t ) } \} _ { i = 1 } ^ { N }$ over T iterations by

$$
\mathcal { L } _ { \mathrm { p o s e } } = \sum _ { t = 1 } ^ { T } \gamma ^ { T - t } \sum _ { i = 1 } ^ { N } \left. \log \left( \mathbf { P } _ { i } ^ { * } { } ^ { - 1 } \tilde { \mathbf { P } } _ { i } ^ { ( t ) } \right) \right. _ { 2 } ,\tag{20}
$$

where log(·) denotes the SE(3) logarithm map and $\gamma = 0 . 9$ emphasizes later refinement iterations. We do not impose direct depth supervision because 3DGS renders both images and depths via α-blending, such that accurate pixel-level depth does not necessarily imply accurate rendered depth. In addition, the predicted depths are defined in the camera coordinate system, making it difficult to enforce cross-view consistency.

The rendering loss is defined as the sum of the MSE and LPIPS [58] losses over all target views rendered in a single

TABLE II  
QUANTITATIVE COMPARISONS ON REALESTATE10K [59] UNDER THE 6-VIEW SETTING. GT POSE INDICATES WHETHER CAMERA POSES ARE PROVIDED.
<table><tr><td>Method</td><td>GT Pose</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td></tr><tr><td>MVSplat</td><td>√</td><td>22.456</td><td>0.761</td><td>0.223</td></tr><tr><td>DepthSplat</td><td>√</td><td>24.172</td><td>0.848</td><td>0.141</td></tr><tr><td>Ours</td><td>√</td><td>25.041</td><td>0.849</td><td>0.132</td></tr><tr><td rowspan="3">NoPoSplat AnySplat</td><td></td><td>22.044</td><td>0.731</td><td>0.210</td></tr><tr><td></td><td>23.712</td><td>0.748</td><td>0.186</td></tr><tr><td></td><td>23.795</td><td>0.844</td><td>0.152</td></tr><tr><td>WorldMirror YoNoSplat</td><td></td><td>25.334</td><td>0.848</td><td>0.135</td></tr><tr><td>Ours</td><td></td><td>25.478</td><td>0.863</td><td>0.130</td></tr></table>

forward pass:

$$
\mathcal { L } _ { \mathrm { r g b } } = \sum _ { n = 1 } ^ { N _ { \mathrm { t g t } } } \left( \mathrm { M S E } ( \tilde { I } ^ { n } , I _ { \mathrm { g t } } ^ { n } ) + \lambda \mathrm { L P I P S } ( \tilde { I } ^ { n } , I _ { \mathrm { g t } } ^ { n } ) \right) ,\tag{21}
$$

where $N _ { \mathrm { t g t } }$ denotes the number of target views in each forward pass, $\tilde { I } ^ { n }$ is the rendered image, and $I _ { \mathrm { g t } } ^ { n }$ is the corresponding ground-truth image. The overall training objective is then defined as

$$
\begin{array} { r } { \mathcal { L } = \mathcal { L } _ { \mathrm { r g b } } + \beta \mathcal { L } _ { \mathrm { p o s e } } , } \end{array}\tag{22}
$$

where $\beta$ controls the relative importance of pose supervision.

## IV. EXPERIMENTS

## A. Experimental Setup

a) Implementation Details: For both geometry and appearance branches, the encoder employs the DINOv2 Large model with 24 attention layers, and the decoder consists of 18 alternating-attention layers. The backbones, encoders, decoders, point head, camera head, and Gaussian head are initialized from the pretrained YoNoSplat [13] weights. For FGPR, the feature encoder, context encoder, and update operator are initialized from the pretrained DROID-SLAM model [54]. The newly introduced layers are initialized to zero, so that they do not perturb the pretrained representations at the beginning of training. During training, we randomly select the number of input views between 2 and 32 and sample 4 target views. All experiments are conducted at a resolution of 224 × 224. We train the model on a single NVIDIA L20 GPU with a batch size of 1 for 100k steps. In the FGPR module, the number of pose refinement iterations T is set to 2. For the training objective, we set λ = 0.05 for the LPIPS term and β = 0.02 for the pose supervision term.

YoNoSplat  
NoPoSplat  
AnySplat  
WorldMirror  
Ours  
Ground Truth  
![](images/e57691678406c08d51211342cc977bc3eb1766ab91ea0718346a7a1f5d8435b3.jpg)  
Fig. 6. Qualitative comparisons on the DL3DV dataset with 24 input views. With more accurate camera pose estimation, our model produces geometrically consistent Gaussian primitives, yielding sharper boundaries, finer details, and fewer artifacts in novel views.

b) Datasets: We train and evaluate our model on the DL3DV [57] and RealEstate10K [59] datasets. DL3DV contains 51.3 million frames captured from 10,510 real-world scenes, spanning 65 point-of-interest categories. We use the full training set for model training, while excluding the 140 benchmark scenes to avoid potential evaluation leakage. We evaluate novel view synthesis under different input-view settings, including 12, 18, 24, and 32 views, with 8 target views used for evaluation. RealEstate10K consists of real-estate videos collected from YouTube, where diverse views of a scene are obtained by sampling frames at different timestamps. We use this dataset to evaluate sparse-view reconstruction with 6 input views.

c) Evaluation Metrics: We evaluate novel view synthesis using the widely adopted metrics PSNR, SSIM [60], and LPIPS [58]. For pose estimation, we follow prior works [1], [13] and report the area under the cumulative angular pose error curve (AUC) at angular thresholds of 5<sup>◦</sup>, 10<sup>◦</sup>, and 20<sup>◦</sup>.

d) Baselines: We compare our method with representative state-of-the-art feed-forward 3DGS methods under both pose-free and pose-dependent settings. For pose-free reconstruction, we consider the following methods:

YoNoSplat [13] is a unified feed-forward 3DGS model designed for both pose-dependent and pose-free reconstruction. It predicts per-view Gaussian primitives and camera poses and aggregates them into a global representation using the predicted camera poses. In our experiments, we use the publicly released checkpoints trained on DL3DV and RealEstate10K at 224 × 224 resolution, respectively.

• NoPoSplat [8] reconstructs 3D Gaussians from sparse unposed images by anchoring one input view as the canonical coordinate system, thereby avoiding explicit local-to-global Gaussian transformation with estimated poses. For a fair comparison, we retrain NoPoSplat under the corresponding multi-view settings on each dataset for 150K steps. On RealEstate10K, we initialize the model from the official pretrained 3-view weights, while keeping the remaining training configurations consistent with the original paper.

![](images/fb071ad240de0eb92399c636ff467be3998c9e3b574d073700707bc803e8b30f.jpg)  
Fig. 7. Qualitative comparisons on the RealEstate10K dataset with sparse-view inputs (6 views). Even with only a few input views, our model reconstructs more coherent scene geometry and more realistic novel views, with clearer details and reduced artifacts.

• AnySplat [1] targets feed-forward 3DGS from unconstrained image collections. It jointly predicts Gaussian parameters, depth maps, and camera poses from uncalibrated inputs, with geometry supervision provided by pretrained VGGT priors.

• WorldMirror [2] is a general-purpose feed-forward 3D geometric prediction model that can incorporate different geometric priors and output multiple 3D representations, including camera parameters, depth maps, point clouds, and 3D Gaussians. We use its publicly released checkpoint trained on multiple large-scale datasets, including DL3DV and RealEstate10K.

We further compare with pose-dependent feed-forward 3DGS methods, where ground-truth camera poses are provided as input:

• MVSplat [61] constructs plane-sweeping cost volumes from posed sparse-view images and uses the estimated depth maps to localize Gaussian centers. For RealEstate10K, we evaluate the official checkpoint released by the authors. For DL3DV, no official checkpoint is available for our multi-view setting. Therefore, we retrain MVSplat for 300K iterations following the original training protocol and extend the number of input views to 32 views.

• DepthSplat [35] enhances cost-volume depth estimation with pretrained monocular depth features, leading to more robust Gaussian center prediction under challenging matching conditions. We use the official checkpoints trained on DL3DV and RealEstate10K.

For camera pose estimation, we additionally report results from recent feed-forward 3D Foundation Models, including MASt3R [41], VGGT [44] and $\pi ^ { 3 }$ [45].

## B. Novel View Synthesis

Tab. I and Tab. II show the quantitative results for novel view synthesis on DL3DV and RealEstate10K. The symbol p indicates whether ground-truth camera poses are provided as input. On DL3DV, our method achieves the best results across all evaluated view settings in both pose-dependent and posefree configurations. In the posed setting, SG-Splat achieves the best quantitative results across all view settings. This demonstrates that the proposed framework remains effective even when accurate camera poses are available. In the posefree setting, SG-Splat also achieves the best quantitative results across all view settings, with a maximum PSNR gain of 0.91 dB over the second-best result. Although the absolute difficulty varies across view settings, the relative advantage of our method becomes more pronounced under denser input settings. This property is particularly important for feed-forward 3DGS, where dense inputs introduce more view-dependent Gaussian primitives, making accurate geometric correction crucial for mitigating pose-induced errors and ensuring reliable Gaussian reconstruction.

On RealEstate10K, SG-Splat further demonstrates its effectiveness under the pose-free sparse-view setting. It achieves the best overall quantitative performance across the evaluation metrics, outperforming all baselines. This confirms that our method remains robust in sparse real-world scenes, rather than being limited to dense-view reconstruction.

TABLE III  
POSE ESTIMATION RESULTS ON DL3DV AND REALESTATE10K. THE UPPER BLOCK REPORTS 3D FOUNDATION MODELS, AND THE LOWER BLOCK REPORTS POSE-FREE FEED-FORWARD 3DGS METHODS. OUR METHOD ACHIEVES THE BEST AUC ACROSS DENSE-VIEW DL3DV SETTINGS AND - R E 10K.
<table><tr><td rowspan="3">Method</td><td colspan="11">DL3DV</td><td colspan="3">RealEstate10K</td></tr><tr><td colspan="3"> $1 2 \mathrm { v }$ </td><td colspan="3"> $1 8 \mathrm { v }$ </td><td colspan="3"> $2 4 \mathrm { v }$ </td><td colspan="3"> $3 2 \mathrm { v }$ </td><td colspan="3"> $_ { 6 \mathrm { v } }$ </td></tr><tr><td> $5 ^ { \circ }$ </td><td> $1 0 ^ { \circ }$ </td><td> $2 0 ^ { \circ }$ </td><td> $5 ^ { \circ }$ </td><td> $1 0 ^ { \circ }$ </td><td> $2 0 ^ { \circ }$ </td><td> $5 ^ { \circ }$ </td><td> $1 0 ^ { \circ }$ </td><td> $2 0 ^ { \circ }$ </td><td> $5 ^ { \circ }$ </td><td> $1 0 ^ { \circ }$ </td><td> $2 0 ^ { \circ }$ </td><td> $5 ^ { \circ }$ </td><td> $1 0 ^ { \circ }$ </td><td> $2 0 ^ { \circ }$ </td></tr><tr><td>MASt3R</td><td>0.626</td><td>0.773</td><td>0.874</td><td>0.599</td><td>0.756</td><td>0.866</td><td>0.638</td><td>0.784</td><td>0.883</td><td>0.653</td><td>0.788</td><td>0.881</td><td>0.521</td><td>0.727</td><td>0.770</td></tr><tr><td>VGGT</td><td>0.680</td><td>0.838</td><td>0.919</td><td>0.654</td><td>0.821</td><td>0.908</td><td>0.653</td><td>0.824</td><td>0.911</td><td>0.629</td><td>0.809</td><td>0.903</td><td>0.590</td><td>0.763</td><td>0.870</td></tr><tr><td> $\pi ^ { 3 }$ </td><td>0.837</td><td>0.918</td><td>0.959</td><td>0.792</td><td>0.895</td><td>0.947</td><td>0.829</td><td>0.914</td><td>0.957</td><td>0.797</td><td>0.899</td><td>0.949</td><td>0.725</td><td>0.852</td><td>0.921</td></tr><tr><td>NoPoSplat</td><td>0.412</td><td>0.625</td><td>0.744</td><td>0.393</td><td>0.612</td><td>0.735</td><td>0.401</td><td>0.635</td><td>0.751</td><td>0.390</td><td>0.625</td><td>0.739</td><td>0.431</td><td>0.619</td><td>0.749</td></tr><tr><td>AnySplat</td><td>0.606</td><td>0.794</td><td>0.896</td><td>0.576</td><td>0.768</td><td>0.879</td><td>0.593</td><td>0.782</td><td>0.888</td><td>0.570</td><td>0.770</td><td>0.880</td><td>0.512</td><td>0.700</td><td>0.830</td></tr><tr><td>WorldMirror</td><td>0.840</td><td>0.920</td><td>0.960</td><td>0.774</td><td>0.885</td><td>0.941</td><td>0.792</td><td>0.895</td><td>0.947</td><td>0.772</td><td>0.883</td><td>0.941</td><td>0.762</td><td>0.871</td><td>0.931</td></tr><tr><td>YoNoSplat</td><td>0.833</td><td>0.917</td><td>0.958</td><td>0.743</td><td>0.870</td><td>0.935</td><td>0.799</td><td>0.895</td><td>0.947</td><td>0.759</td><td>0.875</td><td>0.937</td><td>0.745</td><td>0.878</td><td>0.932</td></tr><tr><td>Ours</td><td>0.868</td><td>0.934</td><td>0.967</td><td>0.792</td><td>0.896</td><td>0.948</td><td>0.849</td><td>0.920</td><td>0.960</td><td>0.809</td><td>0.901</td><td>0.950</td><td>0.801</td><td>0.896</td><td>0.946</td></tr></table>

TABLE IV

ABLATION STUDY ON DL3DV WITH 24 INPUT VIEWS. FGPR\* INDICATES THAT FGPR IS TRAINED, WITH ALL OTHERS FROZEN. THE BASE MODEL ADOPTS THE SAME ARCHITECTURE BUT EXCLUDES BOTH FGPR AND RGC, AND IS TRAINED WITH GROUND-TRUTH (GT) POSES.
<table><tr><td>Variant</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td></tr><tr><td>Base</td><td>22.361</td><td>0.735</td><td>0.203</td></tr><tr><td>FGPR*</td><td>22.499</td><td>0.740</td><td>0.201</td></tr><tr><td>FGPR + Pred Pose</td><td>21.155</td><td>0.647</td><td>0.213</td></tr><tr><td>FGPR + GT Pose</td><td>22.912</td><td>0.751</td><td>0.192</td></tr><tr><td>FGPR + Mix Pose</td><td>23.078</td><td>0.763</td><td>0.181</td></tr><tr><td>FGPR + Mix Pose + RGC</td><td>23.190</td><td>0.772</td><td>0.171</td></tr></table>

TABLE V

EFFECT OF THE NUMBER OF FGPR ITERATIONS T ON DL3DV WITH 24 INPUT VIEWS.
<table><tr><td> $T$ </td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td> $\mathbf { A U C @ 5 ^ { \circ } } \uparrow$ </td></tr><tr><td>1</td><td>23.079</td><td>0.761</td><td>0.179</td><td>0.794</td></tr><tr><td>2</td><td>23.190</td><td>0.772</td><td>0.171</td><td>0.849</td></tr><tr><td>3</td><td>23.183</td><td>0.770</td><td>0.170</td><td>0.851</td></tr></table>

Qualitative comparisons are shown in Fig. 6 and Fig. 7. Our method produces sharper textures, cleaner object boundaries, and more complete scene structures. In contrast, AnySplat tends to generate blurred details because inaccurate predicted poses perturb the transformed Gaussian centers and degrade the supervision of Gaussian attributes. Other methods often suffer from distorted geometry or missing structures due to inaccurate pose estimation and geometric reconstruction. With the proposed FGPR module, our method improves pose accuracy before Gaussian reconstruction, leading to more reliable geometric alignment and more stable Gaussian parameter prediction. As a result, the rendered novel views are visually sharper and geometrically more consistent.

## C. Relative Pose Estimation

We further evaluate the relative pose estimation on DL3DV and RealEstate10K, as reported in Tab. III. DL3DV is evaluated under dense-view settings with 12, 18, 24, and 32 input views, while RealEstate10K is evaluated under the sparse-view setting with 6 input views. The comparison includes posefree feed-forward 3DGS methods and 3D foundation models, including MASt3R, VGGT, and $\pi ^ { 3 }$

Our method achieves the best or comparable-best pose estimation accuracy across the evaluated settings. Our method achieves the highest AUC values on DL3DV under most evaluated view settings and consistently outperforms the compared methods on RealEstate10K across all AUC thresholds. The consistent performance across dense-view DL3DV and sparse-view RealEstate10K demonstrates the robustness of the proposed pose estimation framework.

The comparison with $\pi ^ { 3 }$ and VGGT is particularly informative. These models are trained on large-scale multi-dataset corpora that include DL3DV and RealEstate10K, whereas our method is designed and trained for the feed-forward 3DGS reconstruction task. Despite this strong comparison setting, our method still achieves superior or comparable pose accuracy on most evaluated protocols. These results indicate that the proposed FGPR module effectively improves both fine-grained pose accuracy and coarse relative pose estimation, which further contributes to the superior rendering quality reported in Tab. I and Tab. II.

## D. Ablation Study

We conduct ablation experiments on DL3DV under the 24- view setting. The base model adopts the same architecture but excludes the proposed Flow-Guided Pose Refinement (FGPR) and Residual-Guided Geometry Correction (RGC) modules, and is trained with ground-truth (GT) poses. We progressively incorporate FGPR, different pose selection strategies, and RGC to evaluate their individual contributions to rendering performance.

a) Effect of FGPR: As shown in Tab. IV, introducing FGPR alone (FGPR\*) leads to only marginal improvements over the base model. This suggests that while pose refinement is beneficial, its effectiveness is limited when optimized in isolation with all other components frozen, and cannot fully compensate for pose inaccuracies during training.

b) Effect of Dual-Branch Pose Selection Strategy: We further investigate different pose selection strategies for training FGPR. Training with predicted poses (Pred Pose) leads to a noticeable performance degradation, indicating that noisy pose estimates introduce instability and hinder effective optimization. In contrast, using only GT poses achieves strong performance but suffers from exposure bias, as the model is not exposed to pose perturbations during training. Our proposed dual-branch strategy (Mix Pose), which adaptively combines predicted and GT poses, achieves the best results. This demonstrates that jointly leveraging accurate poses and realistic pose perturbations effectively reduces the discrepancy between training and inference, resulting in more robust rendering quality.

c) Effect of RGC: Building upon FGPR with the proposed dual-path training strategy, further incorporating RGC consistently improves all metrics. This suggests that point-map correction driven by depth residuals improves the reliability of Gaussian centers, thereby enhancing rendering fidelity.

d) Effect of the Number of Iterations: Tab. V evaluates the effect of the number of FGPR iterations T. A single iteration (T = 1) yields suboptimal performance, indicating that one-step refinement is insufficient to fully correct pose errors. Increasing T to 2 significantly improves all metrics, demonstrating that iterative refinement is effective in enhancing pose quality and rendering accuracy. Increasing T from 2 to 3 slightly improves LPIPS and AUC@5<sup>◦</sup>, but reduces PSNR and SSIM while increasing computational cost. We therefore set T = 2 as a practical trade-off between reconstruction quality, pose accuracy, and efficiency.

## V. CONCLUSION

In this paper, we presented SG-Splat, a pose-free feedforward 3D Gaussian Splatting framework for robust reconstruction under inaccurate pose estimation. To reduce the impact of residual pose errors, SG-Splat decouples geometry estimation from Gaussian attribute prediction and introduces a coarse-to-fine refinement pipeline with Flow-Guided Pose Refinement and Residual-Guided Geometry Correction. We further proposed a dual-path training strategy that mixes ground-truth and predicted poses with asymmetric optimization, improving robustness to pose errors while maintaining reliable pose supervision. Extensive experiments on DL3DV and RealEstate10K show that SG-Splat achieves improved novel view synthesis and relative pose estimation under both sparse and dense input-view settings.

Despite these improvements, several limitations remain. First, our method assumes known camera intrinsics, limiting its applicability to in-the-wild images with unavailable or unreliable intrinsics. Second, pixel-aligned feed-forward reconstruction tends to produce view-dependent and redundant Gaussian primitives, resulting in less compact representations. Third, high-resolution reconstruction still incurs substantial training and inference costs. Future work will explore intrinsicfree reconstruction, more efficient high-resolution architectures, and non-pixel-aligned Gaussian prediction to obtain more compact and effective 3DGS representations.

## REFERENCES

[1] L. Jiang, Y. Mao, L. Xu, T. Lu, K. Ren, Y. Jin, X. Xu, M. Yu, J. Pang, F. Zhao et al., “Anysplat: Feed-forward 3d gaussian splatting from unconstrained views,” ACM Transactions on Graphics (TOG), vol. 44, no. 6, pp. 1–16, 2025.

[2] Y. Liu, Z. Min, Z. Wang, J. Wu, T. Wang, Y. Yuan, Y. Luo, and C. Guo, “Worldmirror: Universal 3d world reconstruction with any-prior prompting,” arXiv preprint arXiv:2510.10726, 2025.

[3] Q. Zhao, H. Tan, Q. Wang, S. Bi, K. Zhang, K. Sunkavalli, S. Tulsiani, and H. Jiang, “E-rayzer: Self-supervised 3d reconstruction as spatial visual pre-training,” arXiv preprint arXiv:2512.10950, 2025.

[4] B. Kerbl, G. Kopanas, T. Leimkuhler, and G. Drettakis, “3d gaussian¨ splatting for real-time radiance field rendering.” ACM Transactions on Graphics (TOG), vol. 42, no. 4, pp. 139–1, 2023.

[5] L. Kong, W. Yang, J. Mei, Y. Liu, A. Liang, D. Zhu, D. Lu, W. Yin, X. Hu, M. Jia et al., “3d and 4d world modeling: A survey,” arXiv preprint arXiv:2509.07996, 2025.

[6] S. Fan, L. Zeng, W. He, L. Zhang, W. Chen, and Z. Pan, “A review of recent advances in gaussian splatting,” Applied Intelligence, vol. 56, no. 5, p. 184, 2026.

[7] H. Song, B. Yoon, S. Yang, S. Kang, H. Kim, H. Metzmacher, and W. Woo, “Vrgaussianavatar: Integrating 3d gaussian avatars into vr,” arXiv preprint arXiv:2602.01674, 2026.

[8] B. Ye, S. Liu, H. Xu, X. Li, M. Pollefeys, M.-H. Yang, and S. Peng, “No pose, no problem: Surprisingly simple 3d gaussian splats from sparse unposed images,” in The Thirteenth International Conference on Learning Representations, 2025.

[9] B. Smart, C. Zheng, I. Laina, and V. A. Prisacariu, “Splatt3r: Zeroshot gaussian splatting from uncalibrated image pairs,” arXiv preprint arXiv:2408.13912, 2024.

[10] S. Zhang, J. Wang, Y. Xu, N. Xue, C. Rupprecht, X. Zhou, Y. Shen, and G. Wetzstein, “Flare: Feed-forward geometry, appearance and camera estimation from uncalibrated sparse views,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025, pp. 21 936–21 947.

[11] R. Huang and K. Mikolajczyk, “Spfsplatv2: Efficient self-supervised pose-free 3d gaussian splatting from sparse views,” arXiv preprint arXiv:2509.17246, 2025.

[12] T. W. Mitchel, H. Ryu, and V. Sitzmann, “True self-supervised novel view synthesis is transferable,” 2025. [Online]. Available: https://arxiv.org/abs/2510.13063

[13] B. Ye, B. Chen, H. Xu, D. Barath, and M. Pollefeys, “Yonosplat: You only need one model for feedforward 3d gaussian splatting,” arXiv preprint arXiv:2511.07321, 2025.

[14] M. Levoy and P. Hanrahan, “Light field rendering,” in Seminal Graphics Papers: Pushing the Boundaries, Volume 2, 2023, pp. 441–452.

[15] S. J. Gortler, R. Grzeszczuk, R. Szeliski, and M. F. Cohen, “The lumigraph,” in Seminal Graphics Papers: Pushing the Boundaries, Volume 2, 2023, pp. 453–464.

[16] C. Buehler, M. Bosse, L. McMillan, S. Gortler, and M. Cohen, “Unstructured lumigraph rendering,” in Proceedings of the 28th annual conference on Computer graphics and interactive techniques, 2001, pp. 425–432.

[17] J. Flynn, I. Neulander, J. Philbin, and N. Snavely, “Deepstereo: Learning to predict new views from the world’s imagery,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2016, pp. 5515–5524.

[18] N. K. Kalantari, T.-C. Wang, and R. Ramamoorthi, “Learning-based view synthesis for light field cameras,” ACM Transactions on Graphics (TOG), vol. 35, no. 6, Dec. 2016.

[19] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and R. Ng, “Nerf: Representing scenes as neural radiance fields for view synthesis,” Communications of the ACM, vol. 65, no. 1, pp. 99–106, 2021.

[20] T. Muller, A. Evans, C. Schied, and A. Keller, “Instant neural graphics¨ primitives with a multiresolution hash encoding,” ACM Transactions on Graphics (TOG), vol. 41, no. 4, pp. 1–15, 2022.

[21] S. Fridovich-Keil, A. Yu, M. Tancik, Q. Chen, B. Recht, and A. Kanazawa, “Plenoxels: Radiance fields without neural networks,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 5501–5510.

[22] A. Chen, Z. Xu, A. Geiger, J. Yu, and H. Su, “Tensorf: Tensorial radiance fields,” in European conference on computer vision. Springer, 2022, pp. 333–350.

[23] H. Jiang, X. Xiang, H. Sun, H. Li, L. Zhou, X. Zhang, and G. Zhang, “Geotexdensifier: Geometry-texture-aware densification for high-quality photorealistic 3d gaussian splatting,” IEEE Transactions on Visualization and Computer Graphics, 2026.

[24] A. Zhou, L. Wang, J. Li, J. Huang, L. Li, and J. Yao, “Sggs: Semanticguided 3d gaussian splatting with adaptive rendering,” IEEE Transactions on Visualization and Computer Graphics, 2026.

[25] X. Cai, Y. Wang, Z. Fan, D. Haoran, S. Wang, W. Li, D. Li, L. Luo, M. Wang, and J. Xu, “Dust to tower: Coarse-to-fine photo-realistic scene reconstruction from sparse uncalibrated images,” arXiv preprint arXiv:2412.19518, 2024.

[26] F. Liu, W. Sun, H. Wang, Y. Wang, H. Sun, J. Ye, J. Zhang, and Y. Duan, “Reconx: Reconstruct any scene from sparse views with video diffusion model,” IEEE Transactions on Image Processing, 2026.

[27] Z. Bao, G. Liao, K. Zhou, K. Liu, Q. Li, and G. Qiu, “Loopsparsegs: Loop based sparse-view friendly gaussian splatting,” IEEE Transactions on Image Processing, 2025.

[28] A. Yu, V. Ye, M. Tancik, and A. Kanazawa, “pixelnerf: Neural radiance fields from one or few images,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2021, pp. 4578– 4587.

[29] A. Chen, Z. Xu, F. Zhao, X. Zhang, F. Xiang, J. Yu, and H. Su, “Mvsnerf: Fast generalizable radiance field reconstruction from multi-view stereo,” in Proceedings of the IEEE/CVF international conference on computer vision, 2021, pp. 14 124–14 133.

[30] Y. Chen, P. An, X. Huang, and Q. Wu, “Volume feature aware viewepipolar transformers for generalizable nerf,” IEEE Transactions on Visualization and Computer Graphics, 2025.

[31] D. Charatan, S. L. Li, A. Tagliasacchi, and V. Sitzmann, “pixelsplat: 3d gaussian splats from image pairs for scalable generalizable 3d reconstruction,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 19 457–19 467.

[32] Y. Bao, T. Ding, J. Huo, W. Li, and Y. Gao, “Fisn: Finding spatial neighborhoods for generalizable novel view synthesis,” IEEE Transactions on Visualization and Computer Graphics, 2026.

[33] K. Zhang, S. Bi, H. Tan, Y. Xiangli, N. Zhao, K. Sunkavalli, and Z. Xu, “Gs-lrm: Large reconstruction model for 3d gaussian splatting,” in European Conference on Computer Vision. Springer, 2024, pp. 1–19.

[34] Z. Song, J. Fu, J. Zhang, X. Lu, C. Jia, S. Ma, and W. Gao, “Tinysplat: Feedforward approach for generating compact 3d scene representation,” IEEE Transactions on Circuits and Systems for Video Technology, 2026.

[35] H. Xu, S. Peng, F. Wang, H. Blum, D. Barath, A. Geiger, and M. Pollefeys, “Depthsplat: Connecting gaussian splatting and depth,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025.

[36] X. Lu, J. Fu, J. Zhang, Z. Song, C. Jia, and S. Ma, “Cross-viewconsistent feed-forward 3d gaussian splatting for novel view synthesis,” Pattern Recognition, p. 113474, 2026.

[37] A. Moreau, R. Shaw, M. Nazarczuk, J. Shin, T. Tanay, Z. Zhang, S. Xu, and E. Perez-Pellitero, “Off the grid: Detection of primitives for feed-´ forward 3d gaussian splatting,” arXiv preprint arXiv:2512.15508, 2025.

[38] G. Kang, S. Nam, S. Yang, X. Sun, S. Khamis, A. Mohamed, and E. Park, “ilrm: An iterative large 3d reconstruction model,” arXiv preprint arXiv:2507.23277, 2025.

[39] J. Ren, M. J. Tyszkiewicz, J. Huang, and Z. Gojcic, “Tokengs: Decoupling 3d gaussian prediction from pixels with learnable tokens,” arXiv preprint arXiv:2604.15239, 2026.

[40] S. Wang, V. Leroy, Y. Cabon, B. Chidlovskii, and J. Revaud, “Dust3r: Geometric 3d vision made easy,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 20 697–20 709.

[41] V. Leroy, Y. Cabon, and J. Revaud, “Grounding image matching in 3d with mast3r,” in European conference on computer vision. Springer, 2024, pp. 71–91.

[42] J. Yang, A. Sax, K. J. Liang, M. Henaff, H. Tang, A. Cao, J. Chai, F. Meier, and M. Feiszli, “Fast3r: Towards 3d reconstruction of 1000+ images in one forward pass,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 21 924–21 935.

[43] L.-Z. Chen, J. Gao, Y. Chen, K. L. Cheng, Y. Sun, L. Hu, N. Xue, X. Zhu, Y. Shen, Y. Yao et al., “Geometric context transformer for streaming 3d reconstruction,” arXiv preprint arXiv:2604.14141, 2026.

[44] J. Wang, M. Chen, N. Karaev, A. Vedaldi, C. Rupprecht, and D. Novotny, “Vggt: Visual geometry grounded transformer,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 5294– 5306.

[45] Y. Wang, J. Zhou, H. Zhu, W. Chang, Y. Zhou, Z. Li, J. Chen, J. Pang, C. Shen, and T. He, “pi<sup>3</sup>: Permutation-equivariant visual geometry learning,” arXiv preprint arXiv:2507.13347, 2025.

[46] H. Lin, S. Chen, J. Liew, D. Y. Chen, Z. Li, G. Shi, J. Feng, and B. Kang, “Depth anything 3: Recovering the visual space from any views,” arXiv preprint arXiv:2511.10647, 2025.

[47] M.-Q. V. Bui, J. Moon, and M. Kim, “Airsplat: Alignment and rating for robust feed-forward 3d gaussian splatting,” arXiv preprint arXiv:2603.25129, 2026.

[48] J. Xu, S. Gao, and Y. Shan, “Freesplatter: Pose-free gaussian splatting for sparse-view 3d reconstruction,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025, pp. 25 442–25 452.

[49] Z. Li, C. Dong, Y. Chen, Z. Huang, and P. Liu, “Vicasplat: A single run is all you need for 3d gaussian splatting and camera estimation from unposed video frames,” arXiv preprint arXiv:2503.10286, 2025.

[50] J. Li, J. He, W. Liu, T. Huang, S. Zhou, J. Ma, H. Wang, and H. Li, “Scsv: Spatial-temporal consistent dynamic 3d scene generation from sparse views,” IEEE Transactions on Image Processing, 2026.

[51] Y. Li, C. Lv, Z. Tang, H. Yang, and D. Huang, “Tokensplat: Tokenaligned 3d gaussian splatting for feed-forward pose-free reconstruction,” arXiv preprint arXiv:2603.00697, 2026.

[52] H. Jiang, H. Tan, P. Wang, H. Jin, Y. Zhao, S. Bi, K. Zhang, F. Luan, K. Sunkavalli, Q. Huang et al., “Rayzer: A self-supervised large view synthesis model,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025, pp. 4918–4929.

[53] G. Kang, J. Yoo, J. Park, S. Nam, H. Im, S. Shin, S. Kim, and E. Park, “Selfsplat: Pose-free and 3d prior-free generalizable 3d gaussian splatting,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 22 012–22 022.

[54] Z. Teed and J. Deng, “Droid-slam: Deep visual slam for monocular, stereo, and rgb-d cameras,” Advances in neural information processing systems, vol. 34, pp. 16 558–16 569, 2021.

[55] Z. Teed and J. Deng, “Raft: Recurrent all-pairs field transforms for optical flow,” in European conference on computer vision. Springer, 2020, pp. 402–419.

[56] C. Tang and P. Tan, “Ba-net: Dense bundle adjustment network,” arXiv preprint arXiv:1806.04807, 2018.

[57] L. Ling, Y. Sheng, Z. Tu, W. Zhao, C. Xin, K. Wan, L. Yu, Q. Guo, Z. Yu, Y. Lu et al., “Dl3dv-10k: A large-scale scene dataset for deep learning-based 3d vision,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 22 160–22 169.

[58] R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang, “The unreasonable effectiveness of deep features as a perceptual metric,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2018, pp. 586–595.

[59] T. Zhou, R. Tucker, J. Flynn, G. Fyffe, and N. Snavely, “Stereo magnification: Learning view synthesis using multiplane images,” in SIGGRAPH, 2018.

[60] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, “Image quality assessment: from error visibility to structural similarity,” IEEE transactions on image processing, vol. 13, no. 4, pp. 600–612, 2004.

[61] Y. Chen, H. Xu, C. Zheng, B. Zhuang, M. Pollefeys, A. Geiger, T.- J. Cham, and J. Cai, “Mvsplat: Efficient 3d gaussian splatting from sparse multi-view images,” in European Conference on Computer Vision. Springer, 2024, pp. 370–386.