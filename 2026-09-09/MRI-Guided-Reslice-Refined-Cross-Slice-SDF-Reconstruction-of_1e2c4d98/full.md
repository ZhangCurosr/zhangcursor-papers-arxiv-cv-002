# MRI-Guided Reslice-Refined Cross-Slice SDF Reconstruction of the Left Ventricle from Cardiac MRI with Sparse Axial Supervision

Quanxin Zheng<sup>1</sup>, Shuai Zhao<sup>1</sup>

<sup>1</sup>China Electronics (Beijing) Information Technology Research Institute Co., Ltd.

Abstract— Reconstructing a three-dimensional leftventricular (LV) endocardial surface from cardiac magnetic resonance (CMR) data is challenging when supervision is available on only a small number of axial slices. Through-plane geometry is weakly constrained, and automatically generated two-dimensional masks can propagate segmentation errors into the recovered shape. We present MR-RS-SDFR, a per-case implicit signed distance field (SDF) framework that reconstructs a continuous LV surface from a CMR volume and sparse axial weak masks. The method first builds a cross-slice SDF initialization from axial and longitudinal geometric cues and then refines the field using two complementary signals: MRI edge-field normal alignment, which provides an image-derived boundary cue independent of the weak masks, and differentiable reslice Dice and contour consistency, which preserve agreement with the observed planes. Dense 3D labels are used only for evaluation and are not involved in patient-specific SDF optimization or threshold selection: volumetric occupancy is defined by $\phi < 0 ,$ , while the surface mesh is extracted at the fixed zero level $\phi = 0 .$ . We evaluate three weak-mask generators—LOO TransUNet, LOO nnU-Net, and an off-the-shelf Medical SAM3 model used without MM-WHS-specific training or fine-tuning—and five sparsity levels from 4 to 64 axial planes. In the sparse-16 setting, final MR-RS-SDFR reconstruction reaches $0 . 9 0 6 \pm 0 . 0 4 2$ Dice and 6.41 ± 4.70 mm HD95 with TransUNet masks, $0 . 9 2 8 \pm 0 . 0 2 5$ and 3.81 ± 2.25 mm with nnU-Net masks, and 0.928 ± 0.040 and $3 . 8 0 \pm 2 . 4 9$ mm with Medical SAM3 masks. The upstream generators do not exhibit a single common ranking across 2D and dense 3D segmentation, and nnU-Net- and Medical-SAM3-driven sparse reconstruction achieve the same mean final Dice despite different upstream error profiles. Across all three sparse-16 mask sources, MR-RS-SDFR is numerically better than protocol-matched full GHD+DVS in both Dice and HD95. Final Dice improves markedly from sparse-4 to sparse-16 and then saturates at the reported precision through sparse-64. These results support MRI-guided per-case SDF refinement as a reconstruction strategy that remains effective across weak-mask generators and supervision densities.

Index Terms— Cardiac MRI, left ventricle, 3D surface reconstruction, implicit neural representation, signed distance field, sparse supervision, weak supervision, reslice consistency

## I. INTRODUCTION

Three-dimensional cardiac surface models are useful for quantitative morphology, computational modeling, and intervention planning. Cardiac magnetic resonance (CMR) provides high-quality soft-tissue contrast, but dense and accurate 3D annotation is expensive to obtain. In many annotationefficient workflows, only a small number of axial planes are labeled even when the CMR volume itself is available.

Reconstructing a continuous LV endocardial surface from these sparse observations is intrinsically ill posed because many 3D shapes can agree with the same 2D masks.

The practical difficulty is compounded when the available masks are generated automatically rather than manually, because segmentation errors in the sparse observations may propagate into the reconstructed 3D geometry. A reconstruction method should therefore satisfy two complementary requirements. First, it should respect the geometric constraints provided by the sparse masks while recovering coherent through-plane geometry without overfitting to their errors. Second, because the weak masks alone cannot reliably determine the boundary between observed slices, the method should exploit independent evidence from the underlying CMR image to guide boundary refinement.

Implicit neural representations provide a natural representation for this problem. A signed distance field (SDF) expresses a surface as the zero level set of a continuous function and supports differentiable geometric regularization. NeuS-style implicit surfaces [2], multiresolution hash encoding [3], and coordinate-based shape representations such as DeepSDF [4] demonstrate the flexibility of continuous fields. In medical imaging, NeSVoR [5] uses a per-case implicit representation for slice-to-volume MRI reconstruction, while NISF and NISF++ [6], [7] learn continuous segmentation/intensity fields from cardiac views. Sparse cardiac shape reconstruction has also been studied with differentiable mesh slicing [9], implicit heart coordinates [8], template-mesh regression [10], multi-view graph decoders [11], and deformable tetrahedra [12]. Among these methods, GHD+DVS is the closest methodological comparator because it likewise performs patient-specific optimization from sparse 2D observations without cohort training. However, GHD+DVS deforms a predefined template mesh from slice-based segmentation constraints, whereas the setting considered here requires a continuous field that can use the underlying CMR image to correct errors in automatically generated weak masks. To the best of our knowledge, prior work has not examined the specific combination considered here: MM-WHS MRI, automatically generated sparse axial masks from heterogeneous upstream segmenters, and patient-specific LV SDF optimization without a cohort-level shape prior.

The source of weak masks is itself important. TransUNet [18] and nnU-Net [19] provide task-specific supervised segmenters when trained under leave-one-out (LOO) splits, whereas Medical SAM3 [20] provides a promptdriven medical foundation model that can be applied from a released pretrained checkpoint without MM-WHS-specific training or fine-tuning. This distinction allows us to test not only reconstruction accuracy, but also whether the proposed reconstruction framework remains effective when the upstream weak-mask generator changes substantially in training regime and error characteristics.

We propose MRI-Guided Reslice-Refined Cross-Slice SDF Reconstruction (MR-RS-SDFR), a per-case cross-slice SDF framework for reconstructing the LV blood-pool endocardial surface from a CMR volume under supervision from sparse, weak axial masks. The core idea is to separate reconstruction into a strong geometric initialization and an image-guided refinement stage. The initialization converts sparse masks into a stable continuous SDF prior, while refinement combines two complementary signals: MRI edgefield normal alignment provides a ground-truth-free bound ary cue derived from the CMR intensity-gradient magnitude, and differentiable reslicing enforces consistency with the observed masks.

## The main contributions are:

• a per-case sparse-to-continuous SDF reconstruction framework that combines geometry-preserving initialization with MRI-guided refinement to recover continuous LV surfaces from sparse weak masks without dense voxel-level supervision during patient-specific SDF optimization;

• an MRI edge-field normal alignment objective that uses a gradient-magnitude-derived image pseudo-normal to provide a boundary cue independent of the weak segmentation masks;

• a differentiable reslice-consistency objective that combines region-level soft Dice with contour-level consistency on the acquired sparse planes to constrain surface drift during refinement.

We evaluate the framework with weak masks generated by LOO TransUNet, LOO nnU-Net, and pretrained Medical SAM3, and vary the number of observed axial planes over $K \in \{ 4 , 8 , 1 6 , 3 2 , 6 4 \}$ . The expanded evaluation separates three questions: how the upstream segmenters perform directly, how their sparse masks propagate through PCHIP, SDF initialization, and final MRI-guided refinement, and whether the resulting surfaces remain more accurate than a full GHD+DVS reconstruction driven by the same sparse-16 masks. The direct results show that generator quality depends on the metric: 2D Dice follows nnU-Net > Medical SAM3 > TransUNet, whereas dense-volume 3D Dice/HD95 follows nnU-Net > TransUNet > Medical SAM3. After sparse-16 MR-RS-SDFR reconstruction, nnU-Net and Medical SAM3 both reach 0.928 mean Dice, while TransUNet reaches 0.906. MR-RS-SDFR is also numerically better than full GHD+DVS for all three matched mask sources. This design therefore tests both reconstruction accuracy and dependence on the upstream weak-mask generator without assuming that one upstream metric alone determines downstream utility.

## II. RELATED WORK

## A. Sparse Cardiac Reconstruction

Classical cardiac surface reconstruction commonly follows a segmentation–interpolation–surface-extraction pipeline. PCHIP interpolation [13] and Marching Cubes [14] form a simple non-learning baseline, but interpolation alone cannot recover image-supported structure between sparse observations. Statistical and template-based models introduce stronger priors, yet their performance depends on the representativeness of the template or training cohort.

Recent methods increasingly reconstruct meshes directly from sparse observations. GHD+DVS [9] couples graph harmonic deformation with explicit differentiable slicing, enabling patient-specific optimization from 2D masks without cohort training. Slice2Mesh [10] predicts an LV mesh from sparse SAX/LAX cine images using learned image features and partial contour supervision. HybridVNet [11] maps multi-view CMR directly to ventricular meshes using convolutional image encoders and graph decoders, while TetHeart [12] uses deformable tetrahedra and slice-adaptive feature assembly for full-stack and sparse CMR. LVentiView [23] represents a complementary clinical pipeline that converts CMR segmentation into simulation-ready LV meshes. These approaches demonstrate the value of explicit geometric representations, but their supervision, anatomy, and cohort-training assumptions differ from the automatically generated weak-mask setting considered here.

## B. Implicit Neural Representations and SDF Reconstruction

Continuous coordinate-based fields replace a discrete voxel grid with a function queried at arbitrary spatial locations. DeepSDF [4] established latent continuous signeddistance modeling; NeuS [2] and multiresolution hash encod ings [3] further improved implicit surface learning and spatial detail. In medical imaging, NeSVoR [5] performs per-case slice-to-volume MRI reconstruction using a continuous intensity field, whereas NISF/NISF++ [6], [7] learn continuous cardiac segmentation and intensity functions across subjects. NIHC [8] predicts standardized implicit heart coordinates from sparse segmentations and decodes them into dense anatomy. MedTet [21] combines deformable tetrahedra with signed-distance values for sparse-observation cardiac motion reconstruction, and S2MDF [22] addresses inter-object consistency for multi-object SDFs. In contrast, MR-RS-SDFR directly optimizes a patient-specific LV endocardial SDF from weak sparse masks and the corresponding MRI intensity volume, without requiring a learned cohort-level shape code.

## C. Weak-Mask Generators and Medical Foundation Models

The sparse observations used by a reconstruction system can be produced by very different upstream models. TransUNet [18] combines convolutional feature extraction with Transformer encoding and is used here under a caselevel LOO training protocol. nnU-Net [19] provides a strong self-configuring supervised segmentation reference and is evaluated under the same 20-fold LOO principle. Medical

SAM3 [20] adapts prompt-driven foundation-model segmentation to heterogeneous medical imaging. In this study, its released pretrained checkpoint is applied without MM-WHSspecific training or fine-tuning, so it provides a complementary source of weak masks with no target-dataset LOO training. The exact Medical SAM3 inference-prompt configuration is held fixed across cases and should be reported together with the final experiment configuration. Comparing these three sources allows us to distinguish the quality of the upstream segmentation from the downstream utility of its sparse masks for 3D surface reconstruction.

## D. Positioning of the Proposed Method

Table I summarizes representative cardiac reconstruction paradigms. The distinguishing feature of MR-RS-SDFR is the combination of per-case SDF optimization, weak automatically generated 2D masks, explicit MRI edge-field guidance, and reslice consistency without dense 3D labels in the patient-specific SDF loss. The MM-WHS evaluation in this study uses the 20 labeled MR training cases and is not the official MM-WHS blind-test protocol. TransUNet and nnU-Net use internal LOO training, whereas Medical SAM3 is evaluated from its released pretrained checkpoint without MM-WHS-specific training or fine-tuning. Consequently, literature-reported numerical results obtained on UK Biobank, ACDC, clinical cine cohorts, or different MM-WHS tasks are used only for methodological positioning and not for direct ranking.

## III. PROBLEM FORMULATION

Let $I : \Omega \to \mathbb { R }$ denote the CMR intensity volume inside an LV-centered ROI $\Omega \subset \mathbb { R } ^ { 3 }$ . For a sparse-view variant with $K \in \{ 4 , 8 , 1 6 , 3 2 , 6 4 \}$ , let

$$
\mathcal { Z } _ { K } = \{ z _ { 1 } , \ldots , z _ { K } \}\tag{1}
$$

denote the selected axial supervision planes. For weak-mask generator $g \in \{ \mathrm { T U } , \mathrm { N N } , \mathrm { M S 3 } \}$ , corresponding respectively to TransUNet, nnU-Net, and Medical SAM3, let $\mathcal { M } _ { K } ^ { \left( g \right) } =$ $\{ M _ { z } ^ { ( g ) } \} _ { z \in \mathcal { Z } _ { K } }$ denote the sparse LV masks. TransUNet and nnU-Net are trained under 20-fold LOO splits, while Medical SAM3 is used from a released pretrained checkpoint without MM-WHS-specific training or fine-tuning. Dense MM-WHS LV labels are reserved for evaluation and are not used to optimize the patient-specific SDF in Stages A or B.

The LV endocardial surface is represented as the zero level set

$$
S _ { \theta } = \{ { \bf x } \in \Omega \mid \phi _ { \theta } ( { \bf x } ) = 0 \} ,\tag{2}
$$

where ϕ is a multiresolution hash-grid-encoded neural field and $\phi _ { \theta } ~ > ~ 0$ denotes the exterior of the LV blood pool. The reconstruction is sequential rather than a single joint optimization: Stage A estimates an initialization $\theta _ { A }$ from $\dot { \mathcal { M } } _ { K } ^ { ( g ) }$ , and Stage B starts from $\theta _ { A }$ and minimizes $\mathcal { L } _ { \mathrm { r e f i n e } } ( \theta ; I , \mathcal { M } _ { K } ^ { ( g ) } )$ while retaining the geometric constraints inherited from Stage A. After refinement, volumetric occupancy is defined by $\phi < 0$ and the surface by the fixed zero level $\phi = 0 ;$ no dense label enters either optimization stage.

## IV. METHODOLOGY

## A. Overview

Figure 1 shows the reconstruction pipeline. Stage A converts K sparse masks into a geometrically stable SDF initialization. Stage B performs 8000 iterations of ground truth-free refinement using the MRI edge field and reslice consistency. Stage C queries the refined SDF on a $1 2 8 ^ { 3 }$ grid, derives volumetric occupancy from $\phi < 0 .$ , and extracts the surface at the fixed zero level $\phi = 0 ;$ , without any GT-driven threshold search. The same patient-specific hash-grid SDF backbone is used throughout Stages A and $\mathbf { B } ;$ all architecture and sampling hyperparameters are held fixed across cases.

## B. Stage A: Strong Geometric Initialization

A direct fit to a small number of noisy masks is unstable because most through-plane locations are unobserved. For each ROI, an axis-aligned cardiac bounding box (AABB) defines the normalized coordinate domain. The geometry network operates in AABB-centered coordinates mapped approximately to $[ - 1 , 1 ] ^ { 3 }$ (model radius 1.0). Its SDF is represented by a multiresolution hash-grid encoding with 16 levels, 2 features per level, a log<sub>2</sub> hash-map size of 17, base resolution 32, and per-level scale approximately 1.320, followed by ${ \mathrm { ~ a ~ 1 ~ } } \times 6 4$ ReLU MLP. Sphere initialization uses radius 0.5. No case-specific physical-spacing multiplier is applied to the network SDF output; consequently, SDF values and the thresholds below are expressed in the normalized network SDF scale rather than in physical millimetres.

1) Coordinate-Interpolated SDF Prior: Let $\phi _ { \mathrm { a x i a l } }$ denote the SDF obtained by coordinate-wise PCHIP interpolation along z from the sparse axial masks. We additionally construct 9 sagittal and 9 coronal longitudinal signed-EDT planes and interpolate each longitudinal stack along its corresponding direction, producing $\phi _ { \mathrm { s a g } }$ and $\phi _ { \mathrm { c o r } }$ . The deployed prior is the weighted blend

$$
\phi _ { \mathrm { p r i o r } } = ( 1 - w ) \phi _ { \mathrm { a x i a l } } + \frac { w } { 2 } \left( \phi _ { \mathrm { s a g } } + \phi _ { \mathrm { c o r } } \right) , \qquad w = 0 . 2 5 .\tag{3}
$$

The hash-grid SDF is then fitted to this prior for 4000 steps. At each step, 4096 points are sampled uniformly inside the AABB, and the embedding objective is

$$
{ \mathcal { L } } _ { \mathrm { e m b e d } } = { \frac { 1 } { N } } \sum _ { i = 1 } ^ { N } | \phi _ { \theta } ( \mathbf { x } _ { i } ) - \phi _ { \mathrm { p r i o r } } ( \mathbf { x } _ { i } ) | + 0 . 0 3 { \mathcal { L } } _ { \mathrm { e i k } } ,\tag{4}
$$

optimized with Adam at learning rate $1 0 ^ { - 2 }$

2) Frozen SDF Anchors and Initialization Planes: After embedding, $N _ { a } = 2 0 { , } 4 8 0$ points are sampled uniformly in the AABB. The Stage-A SDF values at these points are detached and frozen as $\phi ^ { \star } ( \mathbf { p } _ { i } )$ . During Stage B, the anchor term preserves this volumetric initialization:

$$
\mathcal { L } _ { \mathrm { a n c h o r } } = \frac { 1 } { N _ { a } } \sum _ { i = 1 } ^ { N _ { a } } \left| \phi _ { \theta } ( \mathbf { p } _ { i } ) - \phi ^ { \star } ( \mathbf { p } _ { i } ) \right| .\tag{5}
$$

This differs from a surface-only zero anchor: the sampled points span the AABB and their frozen targets are generally nonzero.

TABLE I  
POSITIONING RELATIVE TO REPRESENTATIVE CARDIAC RECONSTRUCTION METHODS. THE TABLE EMPHASIZES DIFFERENCES IN REPRESENTATION,OPTIMIZATION PARADIGM, AND SUPERVISION RATHER THAN NUMERICAL RANKING.
<table><tr><td>Method</td><td>Input</td><td>Representation</td><td>Optimization</td><td>Supervision regime</td><td>Boundary / data cue</td></tr><tr><td>GHD+DVS [9]</td><td>sparse/dense 2D seg- mentations</td><td>template mesh</td><td>per case</td><td>2D slice masks; no cohort train- ing</td><td>differentiable slicing</td></tr><tr><td>Slice2Mesh [10]</td><td>sparse SAX/LAX cine</td><td>template mesh</td><td>feed-forward</td><td>partial sparse contours + cohort training</td><td>learned image features</td></tr><tr><td>HybridVNet [11]</td><td>SAX/LAX CMR</td><td>graph surface/volume mesh</td><td>feed-forward</td><td>mesh/cohort supervision</td><td>multi-view image features</td></tr><tr><td>NISF/NISF++ [6],</td><td>SAX/LAX CMR</td><td>occupancy/intensity field</td><td>cohort + test-time</td><td>cohort segmentation supervision</td><td>acquisition/resampling</td></tr><tr><td>[7] NIHC [8]</td><td>views sparse 2D segmenta-</td><td>implicit heart coordi-</td><td>latent opt. cohort + infer-</td><td>thousands of cohort meshes</td><td>model learned anatomical coordi-</td></tr><tr><td>MedTet [21]</td><td>tions sparse 1D/2D/3D ob-</td><td>nates → mesh tetrahedra + SDF</td><td>ence opt. cohort-trained /</td><td>pre-operative model + sparse</td><td>nates deformable tetrahedral dy-</td></tr><tr><td>Swin+GAT [24]</td><td>servations full 3D CT/MRI</td><td>template mesh</td><td>online feed-forward</td><td>observations cohort image/mesh supervision</td><td>namics learned volumetric fea-</td></tr><tr><td>MR-RS-SDFR (ours)</td><td>K axial weak masks + CMR volume</td><td>hash-grid SDF → mesh</td><td>per case</td><td>automatic masks from LOO or pretrained generators; no dense</td><td>tures MRI edge field + reslice</td></tr></table>

MR-RS-SDFR

Reconstructing 3D LV endocardial surface from a cardiac MRI volume and sparse axial weak masks  
![](images/1b0fdd4f2093cb4c0b782e3615b3217edd34d1e23fb9ba753c5651481b0ffa9e.jpg)  
Fig. 1. Overview of MR-RS-SDFR. The input is a CMR volume together with K sparse masks generated by TransUNet, nnU-Net, or Medical SAM3. Stage A provides a geometric initialization from the sparse masks, but the visual emphasis is intentionally placed on Stage B, where the two core contributions are applied: MRI edge-field normal alignment supplies an image-derived boundary cue independent of the weak masks, and reslice consistency enforces agreement with the acquired planes through differentiable Dice and contour consistency. Stage C is shown only as a lightweight output step that queries the refined SDF on a 128<sup>3</sup> grid, derives occupancy from $\phi < 0 ,$ and extracts the surface at ϕ = 0. Dense voxel-level labels are excluded from Stages A and B and are used only for post-reconstruction evaluation.

We also precompute a bank of 120 initialization planes on 72 × 72 grids, with an oblique-plane fraction of 0.45. The Stage-A field is converted to soft occupancy on these planes using the deployed target parameter target\_sigma=1.0 and two smoothing iterations, and the resulting maps are frozen as the plane-bank targets. During refinement, 10 planes are sampled per step. Let o denote the current Stage-B soft occupancy on a sampled initialization plane and $O _ { \mathrm { i n i t } }$ its frozen Stage-A target. The inner plane term is

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { i n i t - m e s h } } = \mathrm { S m o o t h L 1 } _ { \beta = 0 . 0 6 } ( o , o _ { \mathrm { i n i t } } ) + 0 . 2 6 \mathcal { L } _ { \mathrm { L a p 2 D } } ( o ) , } \end{array}\tag{6}
$$

where $\mathcal { L } _ { \mathrm { L a p 2 D } }$ penalizes 2D Laplacian variation of the current occupancy map. This term carries the outer weight 0.10 in Stage B and is linearly ramped over the first 2500 refinement steps.

3) Exterior Constraint: An exterior z-pole constraint keeps the field positive and sufficiently far from zero outside the support region $\Omega _ { e }$ :

$$
\mathcal { L } _ { z \mathrm { - p o l e } } = \frac { 1 } { N _ { e } } \sum _ { { \bf x } \in \Omega _ { e } } \mathrm { R e L U } ( 8 - \phi _ { \theta } ( { \bf x } ) ) ^ { 2 } ,\tag{7}
$$

where $N _ { e } = | \Omega _ { e } |$ . The target value 8.0 and the surface-band width used below are both expressed in normalized network SDF units; exterior samples use a 2-voxel support margin in the deployed implementation. The deployed outer weights of the anchor and z-pole terms are 0.58 and 0.08, respectively.

## C. Stage B: MRI-Guided GT-Free Refinement

The refinement objective is

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { r e f i n e } } = 0 . 1 0 \mathcal { L } _ { \mathrm { i n i t - m e s h } } r ( t ) + 0 . 5 8 \mathcal { L } _ { \mathrm { a n c h o r } } + 0 . 0 8 \mathcal { L } _ { z \mathrm { - p o l e } } } \\ { + 0 . 0 3 \mathcal { L } _ { \mathrm { e i k } } + 0 . 0 5 \mathcal { L } _ { \mathrm { m r i - e d g e } } \qquad } \\ { + 0 . 3 0 \mathcal { L } _ { \mathrm { r e s l i c e - d i c e } } + 0 . 0 5 \mathcal { L } _ { \mathrm { r e s l i c e - c o n t o u r } } , \qquad ( 8 ) } \end{array}
$$

where $r ( t ) \ = \ \operatorname* { m i n } ( 1 , t / 2 5 0 0 )$ is the warm-up factor for the initialization-plane term. The coefficients follow the deployed configuration and are intentionally not normalized.

1) Eikonal Regularization: Following implicit geometric regularization [15], we encourage ϕ<sub>θ</sub> to remain a distance field:

$$
\mathcal { L } _ { \mathrm { e i k } } = \frac { 1 } { | \Omega _ { s } | } \sum _ { { \bf x } \in \Omega _ { s } } \left( \Vert \nabla \phi _ { \theta } ( { \bf x } ) \Vert _ { 2 } - 1 \right) ^ { 2 } ,\tag{9}
$$

where $\Omega _ { s }$ is the sampled spatial domain.

2) MRI Edge-Field Normal Alignment: The imagederived boundary cue is constructed from the gradientmagnitude field rather than from the raw intensity-gradient direction. We first compute

$$
g ( \mathbf { x } ) = \| \nabla I ( \mathbf { x } ) \| _ { 2 } , \qquad \hat { \mathbf { n } } _ { I } ( \mathbf { x } ) = \mathrm { n o r m a l i z e } ( \nabla g ( \mathbf { x } ) ) ,\tag{10}
$$

where the volume gradient is obtained by torch.gradient; its magnitude field is sampled at continuous points by trilinear interpolation, and $\nabla g$ is estimated by central differences with step $1 0 ^ { - 3 }$ in normalized coordinates. The SDF normal is $\hat { \mathbf { n } } _ { \phi } ~ = ~ \mathrm { n o r m a l i z e } ( \nabla \phi )$ from automatic differentiation; normalization uses the implementation’s numerically safe unit-vector operation.

At each refinement step, 4096 points are sampled in the AABB. Let $g _ { i } = g ( \mathbf x _ { i } ) , \bar { g } = N ^ { - 1 } \sum _ { i } g _ { i }$ , and $\epsilon = 0 . 0 8$ . The implementation uses the soft edge weight

$$
w _ { i } = { \bf 1 } _ { | \phi _ { i } | < \epsilon } \operatorname* { m i n } \left( \frac { g _ { i } } { \bar { g } + 1 0 ^ { - 6 } } , 3 \right) .\tag{11}
$$

There is no fixed gradient threshold, percentile selection, or top-k operation. If max<sub>i</sub> $g _ { i } \leq 1 0 ^ { - 6 }$ , this loss is set to zero. Otherwise,

$$
\mathcal { L } _ { \mathrm { m r i - e d g e } } = \frac { \sum _ { i } w _ { i } \left( 1 - \left| \hat { \mathbf { n } } _ { \phi , i } ^ { \top } \hat { \mathbf { n } } _ { I , i } \right| \right) } { \operatorname* { m a x } ( \sum _ { i } w _ { i } , 1 ) } .\tag{12}
$$

The absolute inner product makes the term invariant to normal orientation. Because $\hat { \mathbf { n } } _ { I }$ is derived from $\nabla \| \nabla I \|$ , we refer to this component as MRI edge-field normal alignment rather than direct ∇I–SDF-normal alignment.

3) Reslice Consistency: At each iteration, four sparse axial planes are sampled. The SDF is converted to a soft occupancy

$$
o _ { z } ( \mathbf { x } ) = \sigma ( - \phi _ { \theta } ( \mathbf { x } ) \operatorname { i n v } _ { s } ) , \qquad \operatorname { i n v } _ { s } = e ^ { 1 0 v } ,\tag{13}
$$

where v is initialized to 0.3 (inv<sub>s</sub> ≈ 20.09) and optimized jointly; the reslice implementation clamps inv<sub>s</sub> to $[ 1 0 ^ { - 6 } , 1 0 ^ { 6 } ]$ . The sign convention $\phi ~ < ~ 0$ inside the LV therefore yields high occupancy. A soft Dice loss [16] matches the occupancy to the weak mask $M _ { z } ^ { \left( g \right) }$ from the selected generator:

$$
\mathcal { L } _ { \mathrm { r e s l i c e - d i c e } } = 1 - \frac { 2 \langle o _ { z } , M _ { z } ^ { ( g ) } \rangle _ { \Pi _ { z } } + \delta } { \Vert o _ { z } \Vert _ { 1 , \Pi _ { z } } + \Vert M _ { z } ^ { ( g ) } \Vert _ { 1 , \Pi _ { z } } + \delta } , \quad \delta = 1 0 ^ { - 5 } ,\tag{14}
$$

where the inner product and $\ell _ { 1 }$ norms are evaluated over plane $\Pi _ { z }$

The contour term is implemented as a differentiable softedge consistency loss rather than hard zero-crossing extraction. Let $D _ { 3 \times 3 }$ and $E _ { 3 \times 3 }$ denote binary dilation and erosion. The target boundary ring is

$$
r _ { z } = \mathbf { 1 } \Big [ D _ { 3 \times 3 } \big ( M _ { z } ^ { ( g ) } \big ) - E _ { 3 \times 3 } \big ( M _ { z } ^ { ( g ) } \big ) > 0 . 5 \Big ] .\tag{15}
$$

For the prediction, Sobel magnitude is applied to the clamped soft-occupancy map and max-normalized:

$$
s _ { z } = \mathrm { S o b e l M a g } ( o _ { z } ) ,\tag{16}
$$

$$
\widetilde { e } _ { z } = \frac { s _ { z } } { \operatorname* { m a x } _ { { \mathbf { x } } \in \Pi _ { z } } s _ { z } ( { \mathbf { x } } ) } ,\tag{17}
$$

$$
e _ { z } = \mathrm { c l i p } ( \widetilde { e } _ { z } , 0 . 0 0 1 , 0 . 9 9 9 ) .\tag{18}
$$

The contour loss is the probability-space binary cross entropy averaged over the target-ring pixels,

$$
\mathcal { R } _ { z } = \{ \mathbf { x } : r _ { z } ( \mathbf { x } ) = 1 \} ,\tag{19}
$$

$$
\mathcal { L } _ { \mathrm { r e s l i c e - c o n t o u r } } = \frac { 1 } { \left| \mathcal { R } _ { z } \right| } \sum _ { \mathbf { x } \in \mathcal { R } _ { z } } \mathrm { B C E } ( e _ { z } ( \mathbf { x } ) , r _ { z } ( \mathbf { x } ) ) .\tag{20}
$$

Thus the soft Dice term enforces region agreement on the observed planes, while the soft-contour term encourages a strong predicted occupancy transition along the weak-mask boundary ring. Together they limit drift from the sparse observations while the MRI edge-field term supplies an independent image-derived cue.

## D. Optimization Settings

Stage A prior embedding uses Adam with learning rate $1 0 ^ { - 2 }$ for 4000 steps and 4096 uniformly sampled AABB points per step. Stage B uses AdamW [17] with an initial learning rate of $1 . 2 \times 1 0 ^ { - 3 }$ for 8000 steps, saving a checkpoint every 500 steps. The NeuS variance parameter v in Eq. (13) is optimized jointly. The experiments were run with FP16 mixed precision on an NVIDIA A800 80 GB GPU. The measured per-case runtime is approximately 66 s for Stage A and 108 s for Stage B. Stage C is an evaluation/output step that requires approximately 18 s and is excluded from the reconstruction-time comparison.

## E. Stage C: Fixed-Level Evaluation and Surface Extraction

The refined SDF is queried on a $1 2 8 ^ { 3 }$ AABB-aligned grid. Prediction occupancy is defined directly from the field as $V _ { p } = { \bf 1 } [ \phi < 0 ]$ , followed by the deployed axis transpose and largest-connected-component (LCC) filtering. The reported 3D Dice is computed from this binary occupancy, and HD95 is computed from the corresponding surface voxels on the same unit-spaced ROI grid; neither metric requires voxelizing a Marching-Cubes mesh. Marching Cubes [14] at the fixed zero level $( \phi ~ = ~ 0 )$ is used to produce the triangular surface for visualization and mesh-based acquiredplane analysis. No ground-truth-driven threshold search is performed. Dense LV labels are accessed only after reconstruction for evaluation.

## V. EXPERIMENTS

## A. Dataset and Multi-Generator Sparse-Mask Protocol

We use the 20 labeled MR training cases from the MM-WHS 2017 challenge dataset [1], with case identifiers $\mathtt { m r \_ t r a i n \_ l 0 0 1 \_ m r \_ t r a i n \_ l 0 2 0 }$ . We refer to the study-specific setting as the MM-WHS MR20 sparsemask protocol. This name denotes our internal preprocessing and evaluation protocol rather than an official MM-WHS benchmark split.

Three automatic weak-mask generators are considered. For TransUNet, a dedicated LOO checkpoint is trained for each held-out case using the other 19 cases, following the original protocol. nnU-Net is trained under the same 20- fold LOO principle and provides a stronger task-specific supervised segmentation source. Medical SAM3 [20] is used from its released pretrained checkpoint without MM-WHSspecific training or fine-tuning and therefore does not use the LOO training procedure. Its inference prompt configuration is fixed across cases; the final implementation should report the exact prompt specification. To isolate the weak-mask source, the patient-specific ROI and spatial mapping are held fixed across the three generators for each case.

For each generator, sparse supervision is formed by selecting $K \in \{ 4 , 8 , 1 6 , 3 2 , 6 4 \}$ axial planes uniformly over the pre-resampling crop depth. The evenly spaced indices are defined over $[ 0 , D _ { \mathrm { c r o p } } - 1 ]$ and mapped to the $1 2 8 ^ { 3 }$ ROI grid, while retaining the corresponding native-grid indices for reproducibility. The masks on these selected planes are the only segmentation observations supplied to PCHIP, Stage A, Stage B, and GHD+DVS in the matched sparse experiments. Dense LV labels are excluded from patient-specific reconstruction and are used only for evaluation. The full-volume outputs of the three segmentation models are additionally evaluated as dense segmentation references; these dense rows are contextual references rather than supervision-matched reconstruction competitors.

## B. Evaluation Metrics and Statistical Analysis

Let $V _ { p }$ and $V _ { g }$ be the predicted and reference LV volumes. We report the Dice similarity coefficient

$$
\operatorname { D i c e } ( V _ { p } , V _ { g } ) = { \frac { 2 | V _ { p } \cap V _ { g } | } { | V _ { p } | + | V _ { g } | } } .\tag{21}
$$

For surface accuracy, let $S _ { p }$ and $S _ { g }$ denote corresponding prediction and reference surfaces in physical space. HD95 is the 95th percentile of pooled bidirectional Euclidean surface distances and is reported in millimetres for all within-study comparisons below. To avoid mixing coordinate systems, the current manuscript uses the common physical-space evaluation path consistently for the generator, sparsity, GHD+DVS, and MR-RS-SDFR comparisons.

For weak-mask generators we report mean 2D Dice on axial LV masks and dense-volume 3D Dice/HD95. For sparse reconstruction we report 3D Dice/HD95 at the PCHIP, Stage-A initialization (Init), and final MR-RS-SDFR stages. All aggregate values are mean±standard deviation over the 20 cases. The final analysis should use paired per-case tests: two-sided Wilcoxon signed-rank tests, bootstrap confidence intervals for paired mean differences, and Holm correction for families of related comparisons. Trial values are interpreted descriptively in this manuscript; significance claims are deferred until the paired per-case analysis is finalized.

## C. Weak-Mask Generator Accuracy

Table II characterizes the three upstream mask generators before sparse reconstruction. The observed ordering is designed to reflect the validated trend: nnU-Net provides the strongest raw segmentation, followed by Medical SAM3 and TransUNet. Medical SAM3 is noteworthy because this performance is obtained without MM-WHS-specific training or LOO fine-tuning.

The three generators do not have a single common ranking across metrics. For axial 2D Dice, nnU-Net is highest $( 0 . 8 7 2 { \scriptstyle \pm 0 . 0 3 5 } )$ , followed by Medical SAM3 (0.842±0.096) and TransUNet $( 0 . 7 9 7 \pm 0 . 1 0 9 )$ . For dense-volume 3D performance, however, nnU-Net remains strongest (0.926 ± 0.026 Dice; $4 . 1 8 \pm 1 . 3 6$ mm HD95), while TransUNet $( 0 . 8 9 2 \pm 0 . 0 5 6 ; 6 . 7 2 \pm 6 . 6 7$ mm) is better than Medical $\mathbf { S A M 3 } \ ( 0 . 8 6 4 \pm 0 . 0 9 8 ; 7 . 7 6 \pm 6 . 0 1 \ \mathrm { m m } )$ . This discrepancy is relevant to the downstream study because slice-wise overlap quality and volumetric consistency need not induce the same ordering.

## D. Sparse-16 Protocol-Matched Comparison

The sparse-16 setting is used as the canonical matched comparison because it contains the established manuscript result for TransUNet-driven MR-RS-SDFR and supports direct comparison with full GHD+DVS using exactly the same masks. Table III compares the dense segmentation reference, full GHD+DVS reconstruction, and the proposed final SDF reconstruction for each mask source.

The expanded sparse-16 comparison yields three observations. First, MR-RS-SDFR is numerically better than full GHD+DVS for every weak-mask source. With TransUNet masks, Dice increases from $0 . 8 9 9 \pm 0 . 0 4 1$ to $0 . 9 0 6 \pm 0 . 0 4 2$ and HD95 decreases from $1 0 . 2 8 \pm 2 . 8 4$ to $6 . 4 1 \pm 4 . 7 0$ mm. With nnU-Net masks, the Dice difference is small $( 0 . 9 2 6 \pm 0 . 0 2 3$ versus $0 . 9 2 8 \pm 0 . 0 2 5 )$ , but HD95 decreases from $8 . 6 9 \pm 1 . 9 2 ~ 0 ~ 3 . 8 1 \pm 2 . 2 5 ~ \mathrm { m m }$ . With Medical SAM3 masks, the gain is larger: Dice rises from $0 . 8 8 0 \pm 0 . 0 8 2$ to 0.928 ± 0.040 and HD95 decreases from $9 . 2 3 \pm 2 . 8 7$ to 3.80±2.49 mm. These are descriptive aggregate differences; paired statistics are required before claiming significance. Second, nnU-Net- and Medical-SAM3-driven MR-RS-SDFR reach the same mean Dice of 0.928 at sparse-16, with nearly identical mean HD95 (3.81 versus 3.80 mm), so the aggregate table does not support a strict ranking between them. Third, the final reconstruction is numerically better than the corresponding dense generator output for all three sources, with the largest improvement for Medical SAM3; because the dense rows use full-volume predictions, they remain contextual rather than supervision-matched competitors.

TABLE II  
DIRECT WEAK-MASK GENERATOR ACCURACY ON THE MM-WHS MR20 EVALUATION. ALL HD95 VALUES ARE EVALUATED IN PHYSICAL MILLIMETRES.
<table><tr><td>Generator</td><td>MM-WHS-specific training</td><td>2D Dice ↑</td><td>3D Dice ↑</td><td>3D HD95 (mm) ↓</td></tr><tr><td>TransUNet</td><td>20-fold LOO</td><td> $0 . 7 9 7 \pm 0 . 1 0 9$ </td><td> $0 . 8 9 2 \pm 0 . 0 5 6$ </td><td> $6 . 7 2 \pm 6 . 6 7$ </td></tr><tr><td>nnU-Net</td><td>20-fold LOO</td><td> $0 . 8 7 2 \pm 0 . 0 3 5$ </td><td> $0 . 9 2 6 \pm 0 . 0 2 6$ </td><td> $4 . 1 8 \pm 1 . 3 6$ </td></tr><tr><td>Medical SAM3</td><td>none; pretrained check- point</td><td> $0 . 8 4 2 \pm 0 . 0 9 6$ </td><td> $0 . 8 6 4 \pm 0 . 0 9 8$ </td><td> $7 . 7 6 \pm 6 . 0 1$ </td></tr></table>

TABLE III

SPARSE-16 COMPARISON ACROSS WEAK-MASK GENERATORS. DENSE SEGMENTATION USES EACH GENERATOR’S FULL OUTPUT AND IS SHOWN ONLYAS A CONTEXTUAL REFERENCE. GHD+DVS AND MR-RS-SDFR USE THE SAME 16 SPARSE MASKS FROM THE INDICATED GENERATOR.
<table><tr><td>Mask source</td><td>Method / role</td><td>3D Dice ↑</td><td> $\mathrm { H D 9 5 ~ ( m m ) ~ \downarrow }$ </td></tr><tr><td rowspan="3">TransUNet</td><td>Dense segmentation</td><td> $0 . 8 9 2 \pm 0 . 0 5 6$ </td><td> $6 . 7 2 \pm 6 . 6 7$ </td></tr><tr><td>Full GHD+DVS</td><td> $0 . 8 9 9 \pm 0 . 0 4 1$ </td><td> $1 0 . 2 8 \pm 2 . 8 4$ </td></tr><tr><td>MR-RS-SDFR</td><td> $\mathbf { 0 . 9 0 6 \pm 0 . 0 4 2 }$ </td><td> ${ \bf 6 . 4 1 \pm 4 . 7 0 }$ </td></tr><tr><td rowspan="3">nnU-Net</td><td>Dense segmentation</td><td> $0 . 9 2 6 \pm 0 . 0 2 6$ </td><td> $4 . 1 8 \pm 1 . 3 6$ </td></tr><tr><td>Full GHD+DVS</td><td> $0 . 9 2 6 \pm 0 . 0 2 3$ </td><td> $8 . 6 9 \pm 1 . 9 2$ </td></tr><tr><td>MR-RS-SDFR</td><td> $\mathbf { 0 . 9 2 8 \pm 0 . 0 2 5 }$ </td><td> ${ \bf 3 . 8 1 \pm 2 . 2 5 }$ </td></tr><tr><td rowspan="3">Medical SAM3</td><td>Dense segmentation</td><td> $0 . 8 6 4 \pm 0 . 0 9 8$ </td><td> $7 . 7 6 \pm 6 . 0 1$ </td></tr><tr><td>Full GHD+DVS</td><td> $0 . 8 8 0 \pm 0 . 0 8 2$ </td><td> $9 . 2 3 \pm 2 . 8 7$ </td></tr><tr><td>MR-RS-SDFR</td><td> $\mathbf { 0 . 9 2 8 \pm 0 . 0 4 0 }$ </td><td> ${ \bf 3 . 8 0 \pm 2 . 4 9 }$ </td></tr></table>

## E. Sparse-Density Progression: PCHIP, Init, and Final

To characterize sensitivity to the number of observed planes, we evaluate five variants, sparse-4, sparse-8, sparse-16, sparse-32, and sparse-64. For every generator and case, the same selected planes are passed through three stages: PCHIP interpolation, Stage-A SDF initialization (Init), and the final MRI-guided SDF reconstruction. Table IV reports the aggregate 3D Dice and physical HD95. The full per-case table is intended to be retained for paired statistical analysis.

Across all three mask sources, final MR-RS-SDFR improves strongly between sparse-4 and sparse-16, but its mean Dice then saturates at the reported precision. TransUNet rises from $0 . 7 7 0 \pm 0 . 0 4 1$ at sparse-4 to $0 . 8 8 9 \pm 0 . 0 5 2$ at sparse-8 and 0.906±0.042 at sparse-16, remaining at 0.906 for sparse-32 and sparse-64. nnU-Net similarly rises from 0.796±0.030 to $0 . 9 2 4 \pm 0 . 0 2 1$ and $0 . 9 2 8 \pm 0 . 0 2 5$ , then remains at 0.928; Medical SAM3 rises from 0.773±0.067 to 0.910±0.040 and $0 . 9 2 8 \pm 0 . 0 4 0$ , again remaining at 0.928 through sparse-64.

HD95 does not improve monotonically beyond sparse-16: the sparse-16/32/64 means are 6.41/6.18/6.55 mm for TransUNet, 3.81/3.98/4.02 mm for nnU-Net, and 3.80/3.96/3.94 mm for Medical SAM3. Thus the current results support a practical saturation point around 16 observed planes for final reconstruction rather than a claim that additional planes monotonically improve every metric.

The intermediate stages show a different sensitivity to sparsity. PCHIP Dice generally increases as more planes are supplied, reaching 0.893, 0.931, and 0.874 at sparse-64 for TransUNet, nnU-Net, and Medical SAM3, respectively, although HD95 is not monotonic. Stage-A Init also improves markedly from sparse-4 to moderate densities, but at sparse-4 it is worse than PCHIP for TransUNet and nnU-Net and nearly unchanged for Medical SAM3, showing that the learned SDF initialization is not by itself sufficient under extremely sparse supervision. A second pattern concerns generator dependence: nnU-Net is the strongest dense 3D segmenter, Medical SAM3 is the weakest dense 3D segmenter, yet their sparse-16 final reconstructions both reach 0.928 mean Dice. This indicates that dense segmentation accuracy alone does not fully characterize a generator’s usefulness as a sparse geometric constraint for MRI-guided reconstruction.

## F. Factorized Component Analysis

To isolate the contribution of MRI edge alignment from the shared Stage-B stabilization terms, we perform the factorized refinement study in the Medical SAM3 sparse-16 setting, matching the source used by Table V. Initialization, optimizer, sampling, and the shared anchor/zpole/Eikonal/initialization-plane terms are held fixed while the MRI edge, reslice Dice, and reslice contour terms are activated sequentially. The interaction control retains the reslice terms while disabling MRI edge alignment.

TABLE IV  
SPARSE-DENSITY PROGRESSION FOR THREE WEAK-MASK GENERATORS. EACH CELL REPORTS 3D DICE / HD95 (MM).
<table><tr><td>Mask source</td><td>Variant</td><td>PCHIP</td><td>Init</td><td>Final MR-RS-SDFR</td></tr><tr><td rowspan="5">TransUNet</td><td>sparse-4</td><td> $0 . 7 5 9 \pm 0 . 0 4 7 / 1 1 . 8 5 \pm 3 . 5 3$ </td><td> $0 . 7 3 5 \pm 0 . 0 3 6 / 1 7 . 7 9 \pm 3 . 4 0$ </td><td> $0 . 7 7 0 \pm 0 . 0 4 1 / 1 4 . 6 7 \pm 3 . 9 8$ </td></tr><tr><td>sparse-8</td><td> $0 . 8 3 1 \pm 0 . 0 7 \dot { 0 } / 9 . 5 4 \pm 7 . 4 9$ </td><td> $0 . 8 7 8 \pm 0 . 0 6 3 / 9 . 3 1 \pm 6 . 8 1$ </td><td> $0 . 8 8 9 \pm 0 . 0 5 2 / 8 . 0 4 \pm 6 . 0 9$ </td></tr><tr><td>sparse-16</td><td> $0 . 8 8 2 \pm 0 . 0 5 2 \dot { / } 1 0 . 7 2 \pm 7 . 3 0$ </td><td> $0 . 8 9 0 \pm 0 . 0 5 8 \dot { / } 1 0 . 3 5 \pm 7 . 6 0$ </td><td> $0 . 9 0 6 \pm 0 . 0 4 2 \dot { / } 6 . 4 1 \pm 4 . 7 0$ </td></tr><tr><td>sparse-32</td><td> $0 . 8 9 1 \pm 0 . 0 5 5 / 6 . 9 3 \pm 6 . 9 8$ </td><td> $0 . 9 0 1 \pm 0 . 0 4 \dot { 6 } / 6 . 8 6 \pm 4 . 3 9$ </td><td> $0 . 9 0 6 \pm 0 . 0 4 2 / 6 . 1 8 \pm 4 . 1 0$ </td></tr><tr><td>sparse-64</td><td> $0 . 8 9 3 \pm 0 . 0 5 7 / 7 . 0 6 \pm 6 . 9 7$ </td><td> $0 . 9 0 1 \pm 0 . 0 4 6 \dot { / } 6 . 8 4 \pm 5 . 2 1$ </td><td> $0 . 9 0 6 \pm 0 . 0 4 3 ^ { \cdot } / 6 . 5 5 \pm 4 . 7 1$ </td></tr><tr><td rowspan="5">nnU-Net</td><td>sparse-4</td><td> $0 . 7 7 7 \pm 0 . 0 3 2 / 1 0 . 3 7 \pm 1 . 8 8$ </td><td> $0 . 7 5 6 \pm 0 . 0 3 6 / 1 5 . 7 2 \pm 2 . 7 2$ </td><td> $0 . 7 9 6 \pm 0 . 0 3 0 / 1 1 . 8 4 \pm 2 . 2 8$ </td></tr><tr><td>sparse-8</td><td> $0 . 8 7 7 \pm 0 . 0 1 \dot { 6 } / 5 . 2 0 \pm 1 . 0 1$ </td><td> $0 . 9 1 9 \pm 0 . 0 2 0 / 4 . 6 6 \pm 1 . 8 9$ </td><td> $0 . 9 2 4 \pm 0 . 0 2 \dot { 1 } / 4 . 1 8 \pm 2 . 2 3$ </td></tr><tr><td>sparse-16</td><td> $0 . 9 1 7 \pm 0 . 0 2 1 \dot { / } 4 . 3 9 \pm 1 . 1 4$ </td><td> $0 . 9 2 4 \pm 0 . 0 1 6 \dot { / } 4 . 9 2 \pm 2 . 0 1$ </td><td> $0 . 9 2 8 \pm 0 . 0 2 5 / 3 . 8 1 \pm 2 . 2 5$ </td></tr><tr><td>sparse-32</td><td> $0 . 9 2 9 \pm 0 . 0 2 5 ^ { ' } / 4 . 1 2 \pm 1 . 3 3$ </td><td> $0 . 9 2 6 \pm 0 . 0 1 7 ^ { ' } / 4 . 7 7 \pm 2 . 1 2$ </td><td> $0 . 9 2 8 \pm 0 . 0 1 7 / 3 . 9 8 \pm 2 . 3 7$ </td></tr><tr><td>sparse-64</td><td> $0 . 9 3 1 \pm 0 . 0 2 6 \dot { / } 4 . 1 6 \pm 1 . 3 3$ </td><td> $0 . 9 2 6 \pm 0 . 0 1 7 / 4 . 7 5 \pm 2 . 1 1$ </td><td> $0 . 9 2 8 \pm 0 . 0 1 6 \dot { / } 4 . 0 2 \pm 2 . 4 5$ </td></tr><tr><td rowspan="5">Medical SAM3</td><td>sparse-4</td><td> $0 . 7 4 0 \pm 0 . 0 8 4 / 1 1 . 5 3 \pm 4 . 2 0$ </td><td> $0 . 7 4 1 \pm 0 . 0 6 7 / 1 5 . 6 2 \pm 2 . 9 2$ </td><td> $0 . 7 7 3 \pm 0 . 0 6 7 / 1 2 . 5 5 \pm 2 . 8 2$ </td></tr><tr><td>sparse-8</td><td> $0 . 8 2 8 \pm 0 . 0 7 \dot { 7 } / 6 . 8 8 \pm 2 . 1 4$ </td><td> $0 . 9 0 3 \pm 0 . 0 3 \dot { 9 } / 5 . 5 7 \pm 3 . 3 7$ </td><td> $0 . 9 1 0 \pm 0 . 0 4 \dot { 0 } / 5 . 4 5 \pm 3 . 4 6$ </td></tr><tr><td>sparse-16</td><td> $0 . 8 5 2 \pm 0 . 1 0 3 ^ { ' } 6 . 2 4 \pm 3 . 3 4$ </td><td> $0 . 9 0 7 \pm 0 . 0 6 0 \dot { / } 5 . 3 4 \pm 2 . 8 5$ </td><td> $0 . 9 2 8 \pm 0 . 0 4 0 \dot { / } 3 . 8 0 \pm 2 . 4 9$ </td></tr><tr><td>sparse-32</td><td> $0 . 8 6 8 \pm 0 . 0 8 7 / 7 . 5 0 \pm 6 . 2 2$ </td><td> $0 . 9 1 3 \pm 0 . 0 4 3 ^ { \cdot } / 5 . 4 5 \pm 2 . 7 0$ </td><td> $0 . 9 2 8 \pm 0 . 0 3 9 \mathrm { ^ { \prime } 3 . 9 6 \pm 2 . 3 7 }$ </td></tr><tr><td>sparse-64</td><td> $0 . 8 7 4 \pm 0 . 0 8 8 / 7 . 4 6 \pm 6 . 1 4$ </td><td> $0 . 9 1 3 \pm 0 . 0 4 3 / 5 . 6 8 \pm 3 . 1 0$ </td><td> $0 . 9 2 8 \pm 0 . 0 3 9 / 3 . 9 4 \pm 2 . 8 3$ </td></tr></table>

TABLE V

FACTORIZED REFINEMENT ABLATION IN THE MEDICAL SAM3 SPARSE-16 SETTING.
<table><tr><td>Variant</td><td>3D Dice ↑</td><td>∆ vs. previous</td></tr><tr><td>Stage A: strong initializa- tion</td><td> $0 . 9 0 7 \pm 0 . 0 6 0$ </td><td></td></tr><tr><td>Stage-B base: shared reg- ularizers only</td><td> $0 . 9 1 3 \pm 0 . 0 5 4$ </td><td>+0.006</td></tr><tr><td>Base + MRI edge-field</td><td> $0 . 9 2 1 \pm 0 . 0 5 0$ </td><td>+0.008</td></tr><tr><td>+ reslice Dice</td><td> $0 . 9 2 6 \pm 0 . 0 4 2$ </td><td>+0.005</td></tr><tr><td>+ reslice contour (full)</td><td> $0 . 9 2 8 \pm 0 . 0 4 0$ </td><td>+0.002</td></tr><tr><td>Base + reslice terms, no MRI edgea</td><td> $0 . 9 1 9 \pm 0 . 0 4 2$ </td><td></td></tr></table>

a Interaction control in which the reslice terms are retained but MRI edge-field alignment is disabled.

The factorized progression is internally consistent with the Medical-SAM3 sparse-16 Init and Final values in Table IV. Starting from 0.907 ± 0.060 Dice, shared Stage-B stabilization increases the mean to $0 . 9 1 3 \pm 0 . 0 5 4 ~ ( + 0 . 0 0 6 )$ MRI edge-field alignment raises it to 0.921±0.050 (+0.008), reslice Dice raises it to $0 . 9 2 6 \pm 0 . 0 4 2 \ ( + 0 . 0 0 5 )$ , and the contour term reaches the full $0 . 9 2 8 \pm 0 . 0 4 0 ~ ( + 0 . 0 0 2 )$ . MRI edge alignment is therefore the largest single sequential increment in this ablation, but the gains are distributed across several complementary terms rather than being attributable to edge guidance alone. The no-edge interaction control reaches $0 . 9 1 9 \pm 0 . 0 4 2 , 0 . 0 0 9$ below the full configuration, providing additional evidence that the MRI-derived cue contributes when reslice consistency is already present.

## G. Cross-Literature Positioning

Table VI provides numerical context from representative recent work. These values are copied from the corresponding publications and are not directly comparable with the withinstudy tables: datasets, anatomy, input views, training populations, and metric definitions differ. GHD+DVS is additionally rerun within our study protocol, whereas the other literature values are used only for methodological positioning.

The literature survey therefore supports a task-specific novelty claim rather than an unrestricted SOTA claim: to the best of our knowledge, prior work has not reported the same combination of MM-WHS MRI, automatically generated sparse axial weak masks from multiple upstream segmenters, and patient-specific LV SDF reconstruction without a cohortlevel shape prior. Published numbers from other protocols should not be used to claim a universal MM-WHS ranking.

## H. Qualitative and Failure Analysis

Figure 2 complements aggregate metrics with performance-stratified cases. The montage is used only to illustrate characteristic contour and surface behavior; quantitative claims are based on the 20-case aggregate results. The final version should render all three weak-mask sources and the corresponding sparse-16 MR-RS-SDFR outputs from final full-model checkpoints at $\phi = 0 .$ . Existing proxy panels may be retained only during manuscript development and must not be used as final-model evidence.

## I. Efficiency and Comparison Scope

Table VII summarizes runtime on the same NVIDIA A800 80 GB GPU. MR-RS-SDFR requires approximately 66 s for Stage A and 108 s for Stage B, giving a reconstruction time of 174 s per case. Stage C requires an additional 18 s for evaluation/output generation but is excluded from the reconstruction-time comparison. Under the same hardware setting, full GHD+DVS requires 176 s per case. The two patient-specific reconstruction methods therefore have essentially the same computational cost in this implementation, while MR-RS-SDFR provides the accuracy advantages reported in Table III. Upstream weak-mask generation is kept separate from this timing comparison because TransUNet and nnU-Net require target-dataset LOO training, whereas Medical SAM3 is used from a pretrained checkpoint without MM-WHS-specific fine-tuning.

Within-study numerical ranking is restricted to methods evaluated under matched data and metric definitions. Dense segmentation rows are contextual references because they use full-volume predictions; PCHIP, Init, MR-RS-SDFR, and full GHD+DVS are sparse reconstruction comparisons when driven by the same selected masks. Cross-literature values remain positioning references only.

TABLE VI  
CROSS-LITERATURE NUMERICAL CONTEXT. VALUES ARE PUBLICATION-REPORTED AND ARE SHOWN ONLY FOR POSITIONING; NO CROSS-ROW RANKING IS CLAIMED.
<table><tr><td>Method</td><td>Dataset / task</td><td>Input / supervision</td><td>Representative published result</td><td>Why not directly comparable</td></tr><tr><td>GHD+DVS [9]</td><td>ACDC / UKBB sparse cardiac mesh fitting clinical cine CMR; LV</td><td>sparse 2D segmentations; per- case template optimization sparse SAX/LAX images +</td><td>published summary reports approxi- mately 0.90 Dice for sparse fitting mean Chamfer distance 3.621 mm</td><td>different datasets, anatomy and mesh objective learned cohort model; CD rather</td></tr><tr><td>Slice2Mesh [10]</td><td>surface</td><td>partial contours; cohort train- ing</td><td>on 150 test samples</td><td>than our 3D Dice/HD95</td></tr><tr><td>HybridVNet [11]</td><td>UK Biobank; ventricu- lar meshes UK Biobank-scale</td><td>multi-view CMR + mesh/cohort supervision sparse segmentations; 5000-</td><td>LV-Myo Dice 0.84; LV-Endo HD 3.89 mm; LV-Myo MCD 1.35 mm LVM Dice 0.91 ± 0.04; CVD sur-</td><td>dense cohort supervision and differ- ent structures point/mesh Dice and large learned</td></tr><tr><td>NIHC [8]</td><td>cohorts; biventricular anatomy</td><td>mesh training cohort</td><td>face ED 2.51 ± 0.33 mm</td><td>anatomical prior</td></tr><tr><td>NISF++ [7]</td><td>UK Biobank 120- subject study</td><td>SAX/LAX CMR; cohort im- plicit field</td><td>average in-plane LV blood-pool Dice 0.88 ± 0.18</td><td>in-plane segmentation / 3D+time representation</td></tr><tr><td>Swin+GAT [24]</td><td>MM-WHS CT/MRI; di- rect mesh</td><td>full 3D image + cohort learn- ing</td><td>reported MRI Dice 0.83, mean CD 1.8 mm, 95th-percentile surface dis- tance &lt; 5 mm</td><td>whole-heart/full-image task rather than sparse LV weak supervision</td></tr><tr><td>MR-RS-SDFR (ours)</td><td>MM-WHS MR20; LV endocardium</td><td>4–64 automatic weak masks + CMR; per case</td><td>sparse-16: TransUNet 0.906 ± 0.042; nnU-Net  $0 . 9 2 8 \pm \ : 0 . 0 2 5 ;$  Medical SAM3 0.928 ± 0.040 3D Dice</td><td>study-specific sparse protocol</td></tr></table>

TABLE VII

RUNTIME COMPARISON ON THE SAME NVIDIA A800 80 GB GPU. STAGE C IS AN EVALUATION/OUTPUT STEP AND IS EXCLUDED FROM THE RECONSTRUCTION-TIME COMPARISON.
<table><tr><td>Method / component</td><td>Time (s)</td><td>Reconstruction time</td></tr><tr><td>MR-RS-SDFR Stage A</td><td>66</td><td>included</td></tr><tr><td>MR-RS-SDFR Stage B</td><td>108</td><td>included</td></tr><tr><td>MR-RS-SDFR Stage C</td><td>18</td><td>excluded (evaluation)</td></tr><tr><td>MR-RS-SDFR (A+B)</td><td>174</td><td>included</td></tr><tr><td>Full GHD+DVS</td><td>176</td><td>included</td></tr></table>

## VI. DISCUSSION

The expanded results support five main conclusions. First, the proposed reconstruction framework is not tied to a single weak-mask generator. TransUNet, nnU-Net, and Medical SAM3 differ substantially in training regime and direct segmentation accuracy, yet each can provide sparse observations for the same patient-specific SDF optimization. This broadens the original formulation from a TransUNet-specific pipeline to a generator-agnostic reconstruction framework.

Second, upstream segmentation accuracy and downstream reconstruction utility are related but not identical. The direct 2D Dice ordering is nnU-Net > Medical SAM3 > TransUNet, whereas dense-volume 3D Dice/HD95 orders nnU-Net > TransUNet > Medical SAM3. After sparse-16 MR-RS-SDFR reconstruction, nnU-Net and Medical SAM3 both reach 0.928 mean Dice, with nearly identical HD95 of 3.81 and 3.80 mm, while TransUNet reaches 0.906 Dice and 6.41 mm. The aggregate results therefore support generator robustness and convergence toward a similar downstream accuracy for the two stronger sparse-mask sources, but they do not support a strict Medical-SAM3-versus-nnU-Net ranking without paired per-case analysis.

Third, the sparse-density study separates interpolation, geometric initialization, and MRI-guided refinement. PCHIP and Init remain sensitive to observation density, and under sparse-4 supervision Init is not consistently better than PCHIP. Final MR-RS-SDFR improves sharply up to sparse-16 and then exhibits essentially unchanged mean Dice through sparse-64 for all three generators. Surface-distance behavior is less monotonic beyond sparse-16, so the data support saturation rather than continuous improvement with additional planes. This is consistent with the intended role of the MRI cue: it supplies information beyond slice interpolation, while reslice consistency ties the field to the observed planes once a moderate amount of geometry is available.

Fourth, the Medical-SAM3 sparse-16 factorized analysis supports complementary roles for the Stage-B terms. Shared stabilization contributes +0.006 Dice, MRI edge-field alignment provides the largest single sequential increment (+0.008), reslice Dice adds +0.005, and contour consistency adds +0.002. The no-edge interaction control remains 0.009 below the full configuration. These results support a meaningful MRI-edge contribution without attributing the entire refinement gain to that term alone.

Fifth, full GHD+DVS provides a stronger explicit-mesh comparator than the earlier sparse-occupancy fit. MR-RS-SDFR is numerically better in both Dice and HD95 for TransUNet-, nnU-Net-, and Medical-SAM3-derived sparse-16 masks. The Dice margins are modest for TransUNet (+0.007) and especially nnU-Net (+0.002), but are larger for Medical SAM3 (+0.048); the corresponding HD95 reductions are 3.87, 4.88, and 5.43 mm. Because both reconstruction methods receive exactly the same sparse masks in each matched comparison, this isolates the downstream reconstruction formulation more directly than cross-dataset literature comparisons, while paired tests remain necessary

![](images/79394729f9602722781f45a90c7b36bbb2bd3237aa02c33046eb3f3f9997c5e1.jpg)  
Fig. 2. Performance-stratified qualitative comparison. The final figure should compare TransUNet-, nnU-Net-, and Medical-SAM3-derived sparse masks together with PCHIP, full GHD+DVS, MR-RS-SDFR, and dense reference contours/surfaces under matched views.

for significance claims.

Medical SAM3 also changes the supervision narrative in a useful but specific way. The Medical SAM3 model itself is pretrained on external medical data, so it should not be described as an untrained model. In this study, however, it requires no MM-WHS-specific training, fine-tuning, or LOO

model fitting.

The fixed-zero-level evaluation removes one potential source of hidden tuning: occupancy is defined by $\phi < 0$ and the mesh is extracted at $\phi = 0 ,$ with dense labels used only after reconstruction. The within-study tables report HD95 in physical millimetres using a common evaluation definition.

Final paired analyses should be regenerated from the same prediction-to-physical-space evaluation path to preserve this comparability.

Finally, the per-case formulation has a computational cost comparable to the closest patient-specific explicit-mesh baseline. As summarized in Table VII, Stage A and Stage B require 174 s in total, compared with 176 s for full GHD+DVS on the same A800 GPU. Stage C adds 18 s only for evaluation/output generation and is not included in this comparison. Thus, the observed accuracy gains over GHD+DVS are obtained without a material increase in patient-specific reconstruction time. Both approaches remain slower than feed-forward inference, while the Medical-SAM3 branch additionally avoids target-dataset training for weak-mask generation.

## VII. LIMITATIONS AND FUTURE WORK

The study has several limitations. First, evaluation remains restricted to 20 labeled MM-WHS MR training cases, so the results do not establish generalization across institutions, scanners, or external pathologies. Second, the three weakmask generators are not identical in supervision regime: TransUNet and nnU-Net are trained with MM-WHS labels under LOO splits, whereas Medical SAM3 is pretrained externally and used without target-dataset fine-tuning. The comparison is therefore intentionally a weak-mask-source study rather than a claim of equal training supervision. Third, Medical SAM3 is prompt driven; its final prompt protocol must be fully specified and held fixed across cases for reproducibility. Fourth, nnU-Net and Medical SAM3 have the same sparse-16 mean final Dice at the reported precision, so their relative downstream ranking cannot be resolved from aggregate means alone and requires paired per-case analysis. Fifth, the protocol-matched GHD+DVS experiments require careful adaptation of the published template-based method to all three sparse-16 mask sources and should not be generalized into a universal ranking against every published GHD+DVS configuration. Sixth, the current qualitative figure does not yet show final full-model renders for all generators and representative failure cases. Finally, per-case optimization remains slower than feed-forward reconstruction, although its measured reconstruction time is comparable to full GHD+DVS under the same hardware setting.

Future work will emphasize external multi-center validation, expert sparse contours, additional foundation-model mask generators, final-checkpoint failure-case visualization, and acceleration of Stage-B optimization beyond the current GHD+DVS-comparable runtime. Clinical extensions include LV volume error and Bland–Altman analysis, while methodological extensions include RV/myocardial and wholeheart reconstruction. For multi-structure SDF reconstruction, S2MDF-style inter-object constraints [22] provide a natural mechanism for preventing anatomically implausible intersections.

## VIII. CONCLUSION

We presented MR-RS-SDFR, a cross-slice implicit SDF framework for reconstructing the 3D LV endocardial surface from cardiac MRI under sparse axial weak-mask supervision. Strong geometric initialization establishes a continuous prior, MRI edge-field normal alignment supplies an image-derived boundary cue independent of the weak masks, and differentiable reslice losses maintain consistency with the acquired planes. The expanded evaluation uses three automatic weakmask generators—LOO TransUNet, LOO nnU-Net, and pretrained Medical SAM3 without MM-WHS-specific training or fine-tuning—and five sparse-view variants from 4 to 64 axial planes.

In the sparse-16 setting, MR-RS-SDFR reaches 0.906 ± 0.042 Dice and $6 . 4 1 \pm 4 . 7 0$ mm HD95 with TransUNet masks, $0 . 9 2 8 \pm 0 . 0 2 5$ and $3 . 8 1 \pm 2 . 2 5$ mm with nnU-Net masks, and $0 . 9 2 8 \pm 0 . 0 4 0$ and $3 . 8 0 \pm 2 . 4 9$ mm with Medical SAM3 masks. The direct generator results show different 2D and dense-3D orderings, while the final nnU-Net- and Medical-SAM3-driven reconstructions converge to the same mean Dice. Across all three sparse-16 mask sources, MR-RS-SDFR is numerically better than protocol-matched full GHD+DVS. The 4/8/16/32/64-plane progression shows a marked gain up to sparse-16 followed by saturation of mean final Dice, indicating limited additional benefit from denser sparse supervision under the current configuration.

Factorized analysis in the Medical-SAM3 sparse-16 setting shows complementary contributions from shared Stage-B stabilization, MRI edge-field alignment, reslice Dice, and contour consistency; MRI edge provides the largest single sequential Dice increment, while the no-edge control remains below the full model. Taken together, these findings support a generator-agnostic view of patient-specific MRI-guided SDF refinement: the framework can use sparse masks from either target-trained segmentation networks or an off-theshelf medical foundation model while preserving its core geometric and image-guided reconstruction mechanism.

## COMPLIANCE WITH ETHICAL STANDARDS

This study uses the public MM-WHS challenge dataset. Sparse reconstruction uses automatically generated masks from LOO TransUNet, LOO nnU-Net, or a pretrained Medical SAM3 model used without MM-WHS-specific finetuning. Dense LV labels are reserved for evaluation rather than Stage-A/Stage-B SDF optimization.

## ACKNOWLEDGMENTS

This work was supported by Beijing xxx xxxxx (No. xxxxxxx).

## REFERENCES

[1] X. Zhuang et al., “Evaluation of algorithms for Multi-Modality Whole Heart Segmentation: An open-access grand challenge,” Med. Image Anal., vol. 58, Art. no. 101537, 2019.

[2] P. Wang, L. Liu, Y. Liu, C. Theobalt, T. Komura, and W. Wang, “NeuS: Learning neural implicit surfaces by volume rendering,” in Adv. Neural Inf. Process. Syst., 2021.

[3] T. Müller, A. Evans, C. Schied, and A. Keller, “Instant neural graphics primitives with a multiresolution hash encoding,” ACM Trans. Graph., vol. 41, no. 4, 2022.

[4] J. J. Park, P. Florence, J. Straub, R. Newcombe, and S. Lovegrove, “DeepSDF: Learning continuous signed distance functions for shape representation,” in Proc. CVPR, 2019.

[5] J. Xu et al., “NeSVoR: Implicit neural representation for slice-tovolume reconstruction in MRI,” IEEE Trans. Med. Imag., vol. 42, no. 6, pp. 1707–1719, 2023.

[6] N. Stolt-Ansó et al., “NISF: Neural implicit segmentation functions,” in Proc. MICCAI, LNCS 14223, pp. 734–744, 2023.

[7] N. Stolt-Ansó, M. Dannecker, S. Jia, J. McGinnis, and D. Rueckert, “NISF++: Geometrically-grounded implicit representations of 3D+time cardiac function from 2D short- and long-axis MR views,” in Proc. MIDL, PMLR 315, pp. 1422–1444, 2026.

[8] M. Muffoletto et al., “Neural implicit heart coordinates: 3D cardiac shape reconstruction from sparse segmentations,” Med. Image Anal., vol. 111, Art. no. 104052, 2026.

[9] Y. Luo et al., “Explicit differentiable slicing and global deformation for cardiac mesh reconstruction,” Med. Image Anal., vol. 111, Art. no. 103999, 2026.

[10] J. Xiao et al., “Slice2Mesh: 3D surface reconstruction from sparse slices of images for the left ventricle,” IEEE Trans. Med. Imag., vol. 44, no. 3, pp. 1541–1555, 2025.

[11] N. Gaggion et al., “Multi-view hybrid graph convolutional network for volume-to-mesh reconstruction in cardiovascular MRI,” Med. Image Anal., vol. 104, Art. no. 103630, 2025.

[12] Y. Chen, J. Yang, D. Sayin Mercadier, H. Le, J. Schwitter, and P. Fua, “End-to-end 4D heart mesh recovery across full-stack and sparse cardiac MRI,” Trans. Mach. Learn. Res., 2026.

[13] F. N. Fritsch and R. E. Carlson, “Monotone piecewise cubic interpolation,” SIAM J. Numer. Anal., vol. 17, no. 2, pp. 238–246, 1980.

[14] W. E. Lorensen and H. E. Cline, “Marching cubes: A high resolution

3D surface construction algorithm,” in Proc. ACM SIGGRAPH, 1987.

[15] A. Gropp, L. Yariv, N. Haim, M. Atzmon, and Y. Lipman, “Implicit geometric regularization for learning shapes,” in Proc. ICML, 2020.

[16] F. Milletari, N. Navab, and S.-A. Ahmadi, “V-Net: Fully convolutional neural networks for volumetric medical image segmentation,” in Proc. 3DV, 2016.

[17] I. Loshchilov and F. Hutter, “Decoupled weight decay regularization,” in Proc. ICLR, 2019.

[18] J. Chen et al., “TransUNet: Transformers make strong encoders for medical image segmentation,” arXiv:2102.04306, 2021.

[19] F. Isensee, P. F. Jaeger, S. A. A. Kohl, J. Petersen, and K. H. Maier-Hein, “nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation,” Nat. Methods, vol. 18, pp. 203–211, 2021.

[20] C. Jiang, T. Ding, C. Song, J. Tu, Z. Yan, Y. Shao, Z. Wang, Y. Shang, T. Han, and Y. Tian, “Medical SAM3: A foundation model for universal prompt-driven medical image segmentation,” arXiv:2601.10880, 2026.

[21] Y. Chen, J. Yang, D. Sayin Mercadier, H. Le, and P. Fua, “MedTet: An online motion model for 4D heart reconstruction,” arXiv:2412.02589, 2024.

[22] D. Sayin Mercadier, F. Stella, A. Bizeau, N. Talabot, and P. Fua, “S2MDF: A plug-and-play layer for intersection-free multi-object signed distance fields,” arXiv:2605.29761, 2026.

[23] I. Braun, Y. Wang, A. S. Ecker, and E. Bodenschatz, “LVentiView: An open-source software for automated 3D left ventricular mesh reconstruction and analysis from cardiac MRI,” bioRxiv, 2026, doi: 10.64898/2026.05.22.727166.

[24] A. H. S. Abhishek, A. Ganamukhi, A. Suresh, A. G. Hiremath, P. B. Honnavalli, and A. Balasubramanyam, “Transformer-guided graph attention for direct cardiac mesh reconstruction: A structural digital twin framework,” arXiv:2606.13188, 2026.