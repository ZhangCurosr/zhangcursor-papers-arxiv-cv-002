# Reconstructing Humans and Objects in Interaction using Large Reconstruction Models

Agniv Chatterjee and Georgios Pavlakos

University of Texas at Austin

agniv.chatterjee@utexas.edu, pavlakos@cs.utexas.edu

Abstract. Estimation of Human-Object Interactions in 3D (3D HOI) is a fundamental problem in 3D computer vision with applications in AR/VR, robotics, and embodied AI. However, reconstructing these interactions in 3D remains challenging due to depth ambiguities, occlusions, and object shape variability. Existing approaches are primarily concerned with reprojection and contact constraints, fitting parametric human models and object templates to 2D images. In this paper, we explore a diferent avenue. We present MILO, a framework that leverages the visual capabilities of Large Reconstruction Models (LRMs) to recover detailed 3D human-object interactions from a single image. Our key observation is that LRMs provide a powerful geometric scafold that preserves relative human-object arrangement and proximity cues. This significantly simplifies the reconstruction procedure, reframing the problem as interpreting the LRM mesh: we segment it into human and object components, fit a parametric body model to the human part, and optionally align an object template to the object part (if such a template is available). MILO achieves strong reconstruction accuracy and outperforms existing baselines across multiple benchmarks and interaction scenarios. Our code is available at https://ac5113.github.io/MILO.

Keywords: 3D human-object interactions · Large Reconstruction Models

## 1 Introduction

Daily human life is characterized by rich 3D interactions with a wide variety of objects, from utensils and smartphones to furniture and tools. Modeling these human-object interactions (HOI) is central to many vision applications, including immersive AR/VR, teleoperation, and robot manipulation. However, HOI reconstruction from a single image is dificult due to diverse object shapes and affordances, multiple interaction modes, occlusions, and depth ambiguities, which make the recovery of human pose and object geometry challenging.

Human parametric models such as SMPL [37], MANO [51], and SMPL-X [45] provide standardized representations which have led to successes for 3D human reconstruction. In contrast, the absence of analogous canonical models for generic objects makes object reconstruction significantly more challenging. Moreover, the generalization capabilities of methods [43, 66] trained on existing HOI datasets [3, 20] are hindered by the characteristics of these datasets, which are captured in controlled environments with a limited set of objects and scripted interactions. Additionally, approaches [6, 68] that require the retrieval of object meshes further depend on the quality and coverage of available shape repositories. More importantly, accurate placement of retrieved meshes often requires privileged cues such as ground-truth contact or depth [6, 43, 76].

![](images/8fff16eaa092df510794289acf97318b62c7723a7222fff584f55b93ea09c5e7.jpg)  
Fig. 1: We present MILO, an approach for reconstructing 3D human-object interactions from a single image. Given an input photo (left), MILO uses a Large Reconstruction Model to predict a combined human-object mesh (center). This mesh is interpreted (right) by identifying human and object parts, fitting a SMPL-H body to the human part, and optionally an object template to the object part. The examples illustrate accurate 3D human mesh together with consistent object pose and shape. Human meshes are shown in blue and objects in red.

Our insight is that we can circumvent many of these limitations by leveraging the rich visual capabilities of Large Reconstruction Models (LRMs) [18, 22, 59, 72]. LRMs are trained on large-scale data [7,8] and are capable of reconstructing complex scenes, outputting an image-specific joint reconstruction of the human and the object. This efectively provides a geometric scafold that encodes relative arrangement and proximity cues. Building on this success, our method proposes to replace the ambiguous image fitting and the contact assumptions with the interpretation of an LRM-generated human-object reconstruction.

Our goal is to systematically assess the extent to which LRMs can be used for HOI reconstruction. While previous work has used LRMs primarily to reconstruct the object [36], our insight is that LRMs can be very powerful for capturing the relative positioning and proximity cues between a human and an object. To this end, we propose MILO; Modeling Interactions using Large Reconstruction MOdels. MILO first applies an of-the-shelf LRM [59] to the input RGB image to obtain a holistic human-object mesh. This reframes our problem as explaining the LRM interaction mesh. Through multi-view segmentation of the mesh, we can extract the human and the object point cloud. For the human part, we fit the parametric SMPL-H body model to the LRM mesh via predicted 3D keypoints [46, 73]. For the object, we compose the raw LRM reconstruction with the fitted human, or optionally align a template mesh if one is available. In that case, alignment is supported by establishing semantic correspondences [77] between the template and the LRM mesh. We present example results in Fig. 1.

We quantitatively compare MILO with state-of-the-art methods on multiple datasets, i.e., InterCap [20], HODome [78] and IMHD [80], while also showing qualitative results on in-the-wild images [6]. Importantly, MILO outperforms previous work across the board, without using ground-truth contact information.

Our main contributions are as follows:

1. We demonstrate the power of Large Reconstruction Models for HOI reconstruction. Instead of operating with the high ambiguity of reprojection objectives, we reframe the problem as explaining the LRM mesh results.

2. We design an approach that robustly fits a parametric human model to the LRM mesh and optionally aligns an object template to it.

3. We achieve state-of-the-art performance across multiple benchmarks, all while using weaker information than previous work, i.e., without relying on contact information.

## 2 Related Works

## 2.1 Monocular 3D Human Reconstruction

Single-image 3D human reconstruction methods can be broadly categorized into parametric and non-parametric approaches. Parametric methods [9,26,29,30,52] estimate the parameters of a body model [37,45,51,71], whereas non-parametric methods [27, 32, 33] predict explicit meshes or point clouds. Optimization-based pipelines [4,28,45,71] recover body pose and shape by fitting a parametric body model to 2D cues, e.g., keypoints, silhouettes, or segmentation, via diferentiable rendering. In contrast, learning-based approaches directly regress body-model parameters from inputs [10,23,30,31,41] using deep networks. While these methods have led to robust monocular human reconstruction, they typically focus on the person in isolation and do not explicitly reason about surrounding objects or human-object contact. In our method, we use the HMR2.0 network [15] to obtain initial SMPL parameters from the input RGB image.

## 2.2 Monocular 3D Object Reconstruction

Monocular 3D object reconstruction has a long history with a wide range of 3D representations explored. Early work employed explicit volumetric [5,14], pointbased [11], or mesh-based [11, 63] representations, while later methods [40, 44] adopted implicit signed-distance or radiance-field representations. More recently, 3D Gaussian splatting [25] has emerged as an alternative to NeRF-style radiance fields, enabling eficient object reconstruction. Difusion-based image-to-3D approaches [39, 48, 57] and view-conditioned difusion models [34] further improve visual fidelity and handle complex appearance, but often rely on per-instance optimization or sufer from multi-view inconsistencies across generated views.

Recent work on Large Reconstruction Models [18, 22, 56, 60, 64] has enabled eficient and view-consistent reconstruction of diverse object categories. By leveraging the LRM output, MILO obtains an image-specific joint human-object mesh, which captures accurate geometry and proximity information. For our implementation, we adopt Hunyuan3D-2.0 [59], although we show that our algorithm is not specific to the choice of LRM.

## 2.3 Monocular 3D Human-Object Reconstruction

Joint reconstruction of humans and objects from a single image has been explored with both learning- and optimization-based approaches. HOI-TG [66] employs a transformer to jointly reason about pose and interaction. LEMON [75] learns 3D human-object interaction relations from 2D images by jointly predicting human contact, object afordance, and spatial relations. CONTHO [43] predicts vertex-level contact maps and refines the human-object mesh poses accordingly. StackFLOW [21] regresses dense human-object ofsets and refines both the human and object reconstructions using stacked normalizing flows. CHORE [70] jointly learns a part-correspondence field that links human and object geometry. TriDi [47] supports multiple conditioning modes by introducing a trilateral difusion model over humans, objects, and interactions.

A common strategy involves reconstructing the object by retrieving the closest object mesh from existing shape repositories [8]. PICO [6] retrieves object candidates, while Wu et al. [68] uses text-to-3D retrieval to obtain an object template. Related to our work, LRMs have also been used to reconstruct objects of interaction directly from images, without relying on explicit CAD templates. For instance, EasyHOI [36] employs InstantMesh [72] to obtain object meshes in hand-object interaction scenarios. Our approach also leverages an LRM, but we propose to use these powerful models for joint human-object reconstruction, simplifying the relative positioning of humans and objects.

Most of the previous methods assume access to object templates and/or GT contact annotations, which constrains scalability and generalization to novel objects and in-the-wild scenes. For our approach, the use of an object template is optional, while we do not rely on explicit contact information.

## 2.4 Human-Object Contact Estimation

Human-object contact provides a powerful geometric cue that tightly constrains the arrangement of humans and objects. Early work modeled contact at a coarse level [1,17,24], over patches [12,42], or only on the human body surface [50,54,79]. More recent approaches [16,19,55, 61,76] estimate dense, vertex-level contact or correspondence maps and exploit them to regularize HOI reconstruction or scene understanding tasks. Several HOI methods integrate contact signals directly into their pipelines. CONTHO [43] uses vertex-level human-object contact maps for mesh refinement. CHORE [70] learns part-to-body correspondence fields that implicitly encode contact patterns. PICO [6] uses paired human-object contact maps to guide joint human-object fitting. A related efort, HULC [53], exploits body-scene contact to stabilize monocular motion capture. In contrast to them, our method does not require contact annotations. Instead, we show that we can extract competitive contact estimates from our HOI reconstructions.

## 3 Technical Approach

In this section, we present MILO, our approach that reconstructs detailed humanobject interactions from a single RGB image by leveraging an LRM. Sections 3.1 and 3.2 discuss preliminaries, while Sections 3.3 and 3.4 describe how we can fit a parametric human model to the LRM mesh. In Section 3.5, we describe how we can segment the reconstructed mesh into a human and object part. Finally, if an object template is available, Section 3.6 describes how we can align it to the LRM reconstruction. An overview of our pipeline is shown in Fig. 2.

## 3.1 Human Body Model Preliminaries

We represent the human body with the SMPL-H model [51], parameterized by global orientation $\varPhi \in \mathbb { R } ^ { 3 }$ , body pose $\theta \in \mathbb { R } ^ { 2 1 \times 3 }$ , hand pose $\theta _ { h } \ \in \ \mathbb { R } ^ { 3 0 \times 3 }$ shape $\beta \in \mathbb { R } ^ { 1 0 }$ , and root translation $T \in \mathbb { R } ^ { 3 }$ . Given these parameters, SMPL-H outputs body vertices and kinematic joints as: $[ V , J ] = \mathcal { M } ( \varPhi , \varTheta , \varTheta _ { h } , \beta , T )$ where $V \in \mathbb { R } ^ { 3 \times 6 8 9 0 }$ are the mesh vertices and $J \in \mathbb { R } ^ { 3 \times 5 2 }$ are the joint locations.

## 3.2 Human-Object Mesh Reconstruction

One of our key ideas is to use an LRM to directly reconstruct a combined humanobject mesh from the input image. Concretely, we employ Hunyuan3D-2.0 [59], which produces a combined mesh capturing both the person and the interacted object. This choice is driven by two key advantages: (1) LRMs can generate a high-quality 3D approximation of the object and interaction present in the image. (2) The combined mesh encodes rich information about human-object spatial relations and proximity cues without requiring explicit contact annotations. To this end, we pass an RGBA image to the LRM to obtain a holistic human-object mesh. The alpha channel represents the combined segmentation maps of the human and the object. We interpret the LRM output mesh as a non-parametric interaction scafold, segment it into human and object components, and fit the SMPL-H model to the human part to explain how the person interacts with the object. If an object template is available, we can also align it to the LRM mesh.

## 3.3 3D Keypoint Estimation

The goal of this step is to estimate a sparse set of 3D keypoints on the LRM mesh and use them for subsequent SMPL-H fitting. Specifically, we render the LRM mesh from a set of virtual views V (set to 60) and run ViTPose [73] and HaMeR [46] on each rendered image to obtain body and hand keypoints, respectively. For each view $v \in \mathcal V$ , ViTPose predicts the COCO-25 body keypoints $x _ { b } ^ { ( v ) } \in \mathbb { R } ^ { 2 5 \times 2 }$ with confidence scores $c _ { b } ^ { ( v ) } \in [ 0 , 1 ] ^ { 2 5 }$ . Simultaneously, HaMeR predicts hand keypoints $x _ { h , s } ^ { ( v ) } \in \mathbb { R } ^ { 2 1 \times 2 }$ for $s \in \{ \mathrm { L } , \mathrm { R } \}$ (with confidence set to 1).

![](images/2d0f4f4db07e4ac60f6017b75b11c1320d3a92a937363fe4394fc8ae8c8a4d09.jpg)  
Fig. 2: Overview of MILO, our method for reconstructing 3D humanobject interactions from a single image. Given an input RGB image, we first use Hunyuan3D-2.0 [59] to predict a combined human-object mesh (Sec. 3.2). We then render the mesh from multiple virtual views and estimate 2D body and hand keypoints in each view (Sec. 3.3), which are robustly triangulated to obtain 3D keypoints on the LRM reconstruction. Starting from an initial SMPL-H estimate from HMR2.0 [15] and HaMeR [46], we optimize SMPL-H in two stages (Sec. 3.4): (i) root fitting, optimizing only global orientation and translation, and (ii) pose fitting, additionally optimizing body/hand pose, and shape. Finally, we perform multi-view segmentation of the LRM mesh to extract the object component and compose it with the parametric human mesh for evaluation and visualization (Sec. 3.5). If an object template is available, we optionally align it to the segmented object using semantic correspondences (Sec. 3.6).

For each keypoint, we discard low-confidence detections, triangulate the surviving observations across all pairs of observed views, and select the hypothesis with the largest consensus set under a reprojection-error threshold, followed by nonlinear refinement over the inlier views. This returns the final set of body and hand 3D keypoints (X, c¯), with $\boldsymbol { X } \in \mathbb { R } ^ { 6 7 \times 3 }$ . The confidence c¯ comes from aggregating the per-keypoint confidence over its inlier views. These keypoints are used in the next stage to fit SMPL-H to the LRM mesh. We provide details about the confidence thresholds, triangulation, and aggregation in the SuppMat.

## 3.4 Human Optimization

We get initial estimates of the SMPL-H parameters using HMR2.0 [15], which predicts human pose and shape in the camera frame. We also initialize the hand pose parameters using HaMeR [46]. We then refine these estimates in two stages: the root fitting, which optimizes for the global orientation and the root translation, followed by the pose fitting stage, which additionally optimizes for the body shape, body, and hand pose. Both stages optimize the SMPL-H mesh, keeping the LRM mesh fixed. Additionally, in the first fitting stage, we optimize a human scale parameter. This helps align the human to the LRM mesh without devolving to degenerate scales, since the LRM mesh is not guaranteed to produce the combined mesh at the correct metric scale.

Root Fitting. Let $\hat { P } = \{ \tilde { \Phi } , \tilde { \Theta } , \tilde { \Theta ^ { h } } , \tilde { \beta } , \tilde { \Gamma } \}$ denote the initial SMPL-H prediction for image I as given by HMR2.0 [15] and HaMeR [46], with the parameters expressed in the camera frame. We initialize the body pose (Θ), hand pose $( \Theta ^ { h } )$ and body shape (β) with these values. In this first stage, we keep $\Theta , \bar { \Theta } ^ { h } , \beta$ fixed and only optimize Φ and $\varGamma$ to align the SMPL-H joints $\bar { X } ^ { S }$ to their corresponding 3D keypoints X estimated in the previous section. The objective is expressed as: $\begin{array} { r } { \mathscr { L } _ { r f } = \sum _ { i } \rho ( X _ { i } ^ { S } - X _ { i } ) } \end{array}$ where $\rho$ is the Geman-McClure robustifier [13]. This stage runs for 30 iterations and coarsely aligns SMPL-H with the LRM mesh.

Pose Fitting. Next, we optimize the full SMPL-H parameter set, i.e., shape $\beta _ { ; }$ body pose $\theta ,$ , and hand pose $\theta ^ { h }$ . To keep the solution feasible, we regularize pose and shape with the VPoser prior [45], the MANO prior [51] and an $L _ { 2 }$ shape prior. Let $\zeta _ { b p } \in \mathbb { R } ^ { 3 2 }$ be the representation of the body pose parameters Θ in the latent space of VPoser [45], and $\zeta _ { h p } \in \mathbb { R } ^ { 3 0 }$ be the representation of the hand pose parameters $\theta ^ { h }$ in the latent space of the MANO model [51]. The regularization terms can be expressed as: $\mathcal { L } _ { b p } = \| \zeta _ { b p } \| ^ { 2 } , \mathcal { L } _ { h p } = \| \zeta _ { h p } \| ^ { 2 }$ and $\mathcal { L } _ { \beta } = \| \beta \| ^ { 2 }$

Under severe self-occlusion or object occlusion, the LRM may hallucinate missing body parts, which can lead to image-inconsistent pose estimates. While $\mathcal { L } _ { b p }$ and $\mathcal { L } _ { h p }$ encourage feasible poses, they do not explicitly preserve consistency with the image. To improve robustness in such cases, we add an anchoring loss based on the HMR2.0 [15] pose initialization. We use the initialized latent body pose $\zeta _ { b p } ^ { \mathrm { h m r } }$ and penalize large deviations from it during pose fitting:

$$
\mathcal { L } _ { h a } = \sum _ { j } \left[ \left( \zeta _ { b p , j } - \zeta _ { b p , j } ^ { \mathrm { h m r } } \right) ^ { 2 } - \tau \right] _ { + } ,\tag{1}
$$

where $\tau$ is a tolerance margin, set to 2.5. This acts as a soft anchor: small latent refinements are not penalized, while larger deviations are discouraged.

To further improve alignment in occluded regions, we introduce a visiblevertex 3D consistency loss with the LRM reconstruction. Let $P ^ { \mathrm { o b s } } \subset \mathbb { R } ^ { 3 }$ denote a set of 3D points sampled from the observed (human) LRM reconstruction, and let $P ^ { \mathrm { p r e d } } \subset \mathbb { R } ^ { 3 }$ denote the set of predicted SMPL-H mesh vertices. Because the reconstruction can be unreliable in occluded areas, we restrict the comparison to a subset $P _ { \nu } ^ { \mathrm { p r e d } } \subseteq P ^ { \mathrm { p r e d } }$ containing only those predicted vertices associated with visible joints, where V is the vertex-visibility set estimated from the rendered views. We then define a one-way robust Chamfer loss from the observed points to these visible predicted vertices:

$$
\mathcal { L } _ { 3 D } = \frac { 1 } { | P ^ { \mathrm { o b s } } | } \sum _ { p \in P ^ { \mathrm { o b s } } } \rho \binom { \operatorname* { m i n } _ { m \in \mathrm { } } } { q \in P _ { \nu } ^ { \mathrm { p r e d } } } \| p - q \| _ { 2 } ) ,\tag{2}
$$

where $p \in P ^ { \mathrm { o b s } }$ is an observed 3D point, $q \in P _ { \nu } ^ { \mathrm { p r e d } }$ is a visible predicted vertex, |P<sup>obs</sup>| is the number of observed points, and $\rho ( \cdot )$ is a robust penalty (Geman– McClure) to reduce sensitivity to outliers. This term improves fitting in both self-occlusion and object-occlusion cases.

We combine these with the root loss to obtain the final fitting objective:

$$
\mathcal { L } _ { p f } = \lambda _ { r f } \mathcal { L } _ { r f } + \lambda _ { b p } \mathcal { L } _ { b p } + \lambda _ { h p } \mathcal { L } _ { h p } + \lambda _ { \beta } \mathcal { L } _ { \beta } + \lambda _ { h a } \mathcal { L } _ { h a } + \lambda _ { 3 D } \mathcal { L } _ { 3 D } ,\tag{3}
$$

where $\lambda _ { r f } = 1 0 , \lambda _ { b p } = \lambda _ { h p } = 0 . 0 4 , \lambda _ { \beta } = 0 . 0 5 , \lambda _ { h a } = 1 0$ and $\lambda _ { 3 D } = 5 0$ . We optimize this objective for 60 iterations, yielding a human mesh that is tightly aligned with the LRM reconstruction while remaining anatomically plausible.

## 3.5 Point Cloud Segmentation

The next step is to extract the object geometry from the LRM mesh to obtain an object point cloud. We use the same multi-view renders from Sec. 3.3 and apply a segmentation model [35,49] to obtain human and object masks. The prompt used is “person.<obj\_name>”, indicating the return of one mask for the person and one for the object. The visible vertices in each view are projected onto the image plane and labeled using these masks, followed by multi-view aggregation and filtering. For each vertex, we aggregate the per-viewpoint scores that account for both viewpoint quality and a boundary-closeness weight. The viewpoint quality score balances geometric coverage and segmentation reliability. The boundarycloseness weight handles human-object contact regions. Next, we threshold the aggregated scores to obtain final labels. The vertices labeled as non-human form the raw object point cloud, which we clean via geometric filtering. This procedure is described in detail in the SuppMat. The resulting clean object point cloud is composed with the fitted SMPL-H mesh.

## 3.6 Template Alignment

MILO can operate without a template at test time. When a dataset provides a template mesh, we can optionally align it to the segmented object component of the LRM reconstruction. Let $M ^ { \mathrm { t m p } } ~ = ~ \{ u _ { i } \} _ { i = 1 } ^ { N }$ be the template mesh vertices and let $P ^ { \mathrm { l r m } } = \{ v _ { j } \} _ { j = 1 } ^ { M }$ denote the segmented LRM point cloud. We seek a similarity transform: $T ( u ) = s \mathbf { R } u + \mathbf { t }$ with $s > 0 , { \bf R } \in S O ( 3 )$ and $\mathbf { t } \in \mathbb { R } ^ { 3 }$ align-

![](images/f8c812233ecc4373e4c93a58778d8f28bd273c6ac93f7a6ffbbe15c453175efe.jpg)  
Fig. 3: Template Alignment. Top row: Visualization of semantic correspondences between a render of the template mesh and the LRM mesh. Bottom row: Alignment of the template mesh with the LRM mesh.

ing the template to the reconstructed object.

Semantic correspondences. Our first step is to establish correspondences between the LRM and the template mesh. We do this by considering multi-view renders of both meshes. For this, we first restrict matching to LRM renders that are most consistent with the input image. Using the geometry-aware semantic correspondence model of Zhang et al. [77], we compute a global mutual-nearest-neighbor similarity $S ( \cdot , \cdot )$ between the input image and every LRM render (considering only the object foreground), and retain the top- $K _ { \mathrm { i n } }$ views with $K _ { \mathrm { i n } } = 5 $

Next, for each template render $I _ { q } ^ { \mathrm { t m p } }$ , we evaluate its similarity $S ( I _ { q } ^ { \mathrm { t m p } } , I _ { r } ^ { \mathrm { l r m } } )$ against the retained LRM renders $I _ { r } ^ { \mathrm { { l r m } } }$ and keep the top- $\cdot K _ { \mathrm { p a i r } }$ matches $( K _ { \mathrm { p a i r } } =$ ). For each selected pair $( q , r )$ , we compute dense bidirectional pixel correspondences in descriptor space. Let $\Pi _ { q  r } ( x )$ denote the target pixel matched to source pixel $x ,$ and let $C _ { q  r } ( x )$ be its cosine similarity.

We lift these 2D matches to 3D correspondences by unprojection. A template vertex $u _ { i }$ is considered only if it is visible in view $q ,$ projects inside the foreground mask, and its matched pixel $\Pi _ { q \to r } ( \pi _ { q } ( u _ { i } ) )$ lies within a radius $\tau _ { \mathrm { 2 D } }$ of some visible LRM vertex projection in view r. The matched LRM vertex is:

$$
j ^ { \ast } ( i , q , r ) = \arg \operatorname* { m i n } _ { j \in \mathcal { V } _ { r } ^ { \mathrm { l r m } } } \left\| \pi _ { r } ( v _ { j } ) - \pi _ { q \to r } ( \pi _ { q } ( u _ { i } ) ) \right\| _ { 2 } ,\tag{4}
$$

subject to: $C _ { q \to r } ( \pi _ { q } ( u _ { i } ) ) > \tau _ { \mathrm { l o c } }$ and $\| \pi _ { r } ( v _ { j ^ { * } } ) - \pi _ { q \to r } ( \pi _ { q } ( u _ { i } ) ) \| _ { 2 } < \tau _ { \mathrm { 2 D } }$ We use $\tau _ { \mathrm { l o c } } = 0 . 6$ and $\tau _ { \mathrm { 2 D } } = 1 5$ pixels. A template vertex may obtain multiple candidate matches across view pairs. We retain only the highest-scoring one, with score:

$$
\omega _ { i , q , r } = C _ { q \to r } ( \pi _ { q } ( u _ { i } ) ) \cdot S ( I _ { q } ^ { \mathrm { t m p } } , I _ { r } ^ { \mathrm { l r m } } ) .\tag{5}
$$

This gives a sparse forward map $i \mapsto j _ { i }$ with weight $w _ { i } = \mathrm { m a x } _ { q , r } \omega _ { i , q , r }$ . To suppress unstable matches, we also compute the reverse map \nameCOLR {\mbox {LRM}\xspace \ rightarow \ template and apply neighborhood-based cycle consistency. A forward match $u _ { i } \mapsto v _ { j _ { i } }$ is retained only if the reverse map sends $v _ { j _ { i } }$ back to a template vertex within a local neighborhood of $u _ { i } .$ defined relative to the local template mesh scale. This yields the final sparse correspondence set: $\mathcal { C } = \{ ( i _ { k } , j _ { k } , w _ { k } ) \} _ { k = 1 } ^ { K } , ( \mathrm { F i g . 3 } )$

Template alignment. Given sparse weighted correspondences $\bar { \mathcal { C } } = \{ ( i _ { k } , j _ { k } , w _ { k } ) \} _ { k = 1 } ^ { K }$ we first estimate an initial similarity transform $T ( u ) = s \mathbf { R } u + \mathbf { t }$ by solving a weighted Sim(3) Procrustes problem with a weighted Kabsch fit. We then refine $T$ with an iterative weighted Sim(3) re-estimation that jointly uses $\mathcal { C }$ and nearest-neighbor constraints to the segmented LRM object point cloud $P ^ { \mathrm { l r m } }$

At iteration t, we transform template vertices as $\tilde { u } _ { i } ^ { ( t ) } = T ^ { ( t ) } ( u _ { i } )$ , assign each $\tilde { u } _ { i } ^ { ( t ) }$ its closest point $p _ { i } ^ { ( t ) } = \mathrm { N N } ( \tilde { u } _ { i } ^ { ( t ) } , P ^ { \mathrm { l r m } } )$ , and form an inlier set $\nu _ { t } ~ = ~ \{ i ~ \mid$ $\| \tilde { u } _ { i } ^ { ( t ) } - p _ { i } ^ { ( t ) } \| _ { 2 } < \tau _ { t } \}$ using a threshold $\tau _ { t }$ that is progressively tightened across iterations. We then compute $T ^ { ( t + 1 ) }$ by a single weighted Kabsch-with-scale fit on the combined pairs $\{ ( u _ { i _ { k } } , v _ { j _ { k } } ) \} _ { k = 1 } ^ { K } \cup \{ ( u _ { i } , p _ { i } ^ { ( t ) } ) \} _ { i \in \mathcal { V } _ { t } }$ , using weights $w _ { k }$ for semantic pairs and a constant weight of 10 for each nearest-neighbor pair. This procedure retains the semantic alignment captured by $\mathcal { C }$ while improving surfacelevel agreement with the reconstructed object geometry.

## 4 Experiments

This section presents our empirical evaluation. Section 4.1 compares MILO with the state-of-the-art on the InterCap [20], HODome [78] and IMHD [80] datasets.

<table><tr><td>Methods</td><td>Type</td><td>Needs Template?</td><td>Needs Contact?</td><td>PA-CDh PA-CD。 PA-CDh+o (cm) ↓ (cm) ↓</td><td>(cm) ↓</td></tr><tr><td>CONTHO [43]</td><td>Reg.</td><td>√</td><td>√</td><td>8.36 24.30</td><td>13.14</td></tr><tr><td>HOI-TG [66]</td><td>Reg.</td><td>√</td><td>X</td><td>8.22 25.05</td><td>14.63</td></tr><tr><td>PHOSA [76]</td><td>Opt.</td><td>√</td><td>√</td><td>10.07 23.36</td><td>13.38</td></tr><tr><td>Open3DHOI [67]</td><td>Opt.</td><td>X</td><td>X</td><td>6.88 31.18</td><td>10.17</td></tr><tr><td>PICO [6]</td><td>Opt.</td><td>√</td><td>√</td><td>7.43 21.85</td><td>10.33</td></tr><tr><td>Ours (w/ template)</td><td>Opt.</td><td>√</td><td>X</td><td>6.96 18.97</td><td>7.45</td></tr><tr><td>Ours (w/o template) Opt.</td><td></td><td>X</td><td>X</td><td>6.85 20.74</td><td>9.36</td></tr></table>

Table 1: Quantitative comparison with the state of the art on the Inter-Cap dataset [20]. We report Procrustes-Aligned Chamfer Distance (PA-CD) for the human mesh $\left( \mathrm { P A - C D } _ { h } \right)$ , object mesh $\left( \mathrm { P A - C D } _ { o } \right)$ , and the combined human-object pair $( \mathrm { P A } \mathrm { - C D } _ { h + o } )$ ; lower is better. Regression (Reg.) and optimization (Opt.) baselines are evaluated under the training protocol from Cseke et al. [6]. Our method, MILO, uses no contact information, yet achieves the best overall reconstruction accuracy.

We evaluate settings with and without access to a template. Sec. 4.2 evaluates the efect of MILO on the downstream task of estimating vertex-level contact on the human body. Section 4.3 studies the efect of the underlying LRM mesh geometry and the dependence of MILO on the choice of LRM. Section 4.4 compares our joint reconstruction scafold against object-only and individual-reconstruction baselines. Section 4.5 analyzes and ablates the key design choices. Finally, Sec. 4.6 presents qualitative in-the-wild reconstructions on PICO-db [6].

<table><tr><td>Methods</td><td> $\left| \mathrm { P A } \mathrm { - C D } _ { h } \mathrm { P A } \mathrm { - C D } _ { o } \mathrm { P A } \mathrm { - C D } _ { h + o } \right.$  (cm) ↓ (cm) ↓</td><td>(cm) ↓</td></tr><tr><td>PICO [6]</td><td>13.12 14.44</td><td>10.07</td></tr><tr><td>Ours (w/ template)</td><td>9.50 12.97</td><td>6.68</td></tr><tr><td>Ours (w/o template)</td><td>9.71 13.75</td><td>6.38</td></tr></table>

<table><tr><td>Methods</td><td>[PA-CDh PA-CD0 PA-CDh+o (cm) ↓ (cm) ↓</td><td>(cm) ↓</td></tr><tr><td>PICO [6]</td><td>15.81 17.70</td><td>13.24</td></tr><tr><td>Ours (w/ template)</td><td>11.76 13.39</td><td>10.10</td></tr><tr><td>Ours (w/o template)</td><td>8.99 9.71</td><td>6.98</td></tr></table>

Table 2: Quantitative evaluation on the HODome dataset [78].  
Table 3: Quantitative evaluation on the IMHD dataset [80].

## 4.1 Comparison with the state-of-the-art

InterCap [20] is an indoor dataset with ground-truth human and object meshes, including ten human subjects and ten everyday objects in various interaction scenarios. We evaluate MILO for HOI reconstruction in this controlled setting and compare it with both regression-based [43,66] and optimization-based methods [6, 67, 76] following the protocol of Cseke et al. [6]. For the learning-based baselines, we use checkpoints trained only on BEHAVE [3], i.e., enforcing an out-of-domain evaluation.

We report Procrustes-Aligned Chamfer Distance (PA-CD) for the human mesh, the object mesh, and the combined human-object pair. The rigid alignment is performed on the combined mesh; after alignment, PA-CD is computed separately for the human and object components as well as for the joint reconstruction. Quantitative results are presented in Tab. 1. MILO outperforms all previous work (even without access to a template), while relying on less privileged information than competing methods.

CONTHO

HOI-TG

PICO

Ours

Ours

GT

Table 4: Contact evaluation. Metrics are reported on InterCap. Our 3D geometryinferred contact outperforms the image-regressed contact by the DECO baseline [61].  
![](images/482e76bb158bcd7107298562c13e571f135787e24467c0c3dcafb0ec6fa4710c.jpg)

Fig. 4: Qualitative comparison on the InterCap dataset [20]. From left to right: input RGB image, CONTHO [43], HOI-TG [66], PICO-fit [6], MILO reconstructions (with and without template meshes, respectively), and ground-truth human-object meshes. Our method achieves better human-object alignment in 3D. Humans are rendered in blue, template objects in red, and segmented object meshes in orange.

<table><tr><td>Method</td><td>F1↑</td><td>Precision↑</td><td>Recall↑</td><td>geo. (cm)↓</td></tr><tr><td>DECO [61]</td><td>9.37</td><td>12.57</td><td>10.28</td><td>126.70</td></tr><tr><td>Ours</td><td>30.00</td><td>36.57</td><td>39.49</td><td>39.57</td></tr></table>

Additionally, we evaluate MILO on HODome [78] and IMHD [80] datasets, comparing against the best-performing baseline PICO. HODome and IMHD are also indoor datasets with ground-truth human and object meshes. HODome consists of 10 human subjects and 20 object classes, while IMHD consists of 8 human subjects and 10 object categories with diferent interactions with each object. As can be seen from Tab. 2 and Tab. 3, MILO performs better than PICO across the board, in both modes.

The improvements from MILO are also evident in the qualitative results on InterCap [20] (Fig. 4), HODome [78], and IMHD [80] (Fig. 5), where MILO achieves better 3D alignment between the human and the object, with more accurate contact and fewer penetrations than prior approaches.

## 4.2 Contact estimation

MILO does not predict contact, but we can infer contact based on the proximity of the SMPL-H mesh and the aligned object template. We report the results

PICO

Ours (w/ template)

Ours (w/o template)

GT

![](images/d73f4cce0beb6ca84e2b155aac1b0c66369b506045d9823c33da3b5e2c981038.jpg)  
Fig. 5: Qualitative evaluation of PICO-fit [6] and MILO on images from the HODome [78] and IMHD [80] datasets. Our outputs, with or without templates, are more physically plausible and closer to the GT than PICO. Humans are rendered in blue, template objects in red, and segmented object meshes in orange.

of this strategy in Tab. 4. Compared to the DECO RICH checkpoint, MILO achieves higher F1, Precision, and Recall, and lower geometric error on InterCap.

## 4.3 Efect of mesh geometry

Next, we evaluate the efect of the initial mesh geometry in the MILO pipeline. Table 5 compares our default Hunyuan3D-2.0 choice with alternative LRM backbones, and an oracle mesh obtained from the ground-truth human-object mesh. Replacing Hunyuan3D-2.0 [59] with SAM3D (joint) [58] gives comparable performance, suggesting that MILO is not tied to a specific LRM choice. InstantMesh [72] performs worse, but we acknowledge that it corresponds to an older method. Finally, the oracle result indicates that the remaining error is largely bounded by upstream reconstruction quality.

<table><tr><td>Mesh Geometry</td><td></td><td>PA-CDh (cm) ↓ PA-CD。 (cm)↓</td><td> $\mathrm { P A } { \mathrm { - C D } } _ { h + o } \ ( \mathrm { c m } ) \ \downarrow$ </td></tr><tr><td>InstantMesh [72]</td><td>7.51</td><td>33.21</td><td>10.84</td></tr><tr><td>SAM3D (joint) [58]</td><td>7.00</td><td>21.99</td><td>9.80</td></tr><tr><td>Hunyuan3D-2.0 [59]</td><td>6.85</td><td>20.74</td><td>9.36</td></tr><tr><td>Oracle</td><td>3.28</td><td>4.93</td><td>3.12</td></tr></table>

Table 5: Efect of mesh geometry. Results presented on InterCap. We replace the mesh reconstructed by Hunyuan3D-2.0 [59] with meshes produced by SAM3D [58] and InstantMesh [72], as well as the ground-truth geometry (oracle). We observe that SAM3D performs comparably to Hunyuan3D-2.0, which indicates that MILO is not tied to a specific LRM choice.

<table><tr><td>Baseline</td><td></td><td>Contact Loss PA-CDh (cm) ↓ PA-CD。 (cm) ↓</td><td> $\overline { { \mathrm { P A - C D } _ { h + o } \left( \mathrm { c m } \right) } } \downarrow$ </td></tr><tr><td>EasyHOI* (full-body)</td><td>X</td><td>41.57 77.06</td><td>37.21</td></tr><tr><td>EasyHOI* (full-body)</td><td>√</td><td>9.48 27.93</td><td>12.08</td></tr><tr><td>SAM3D (independent)</td><td>x</td><td>18.18 31.07</td><td>15.89</td></tr><tr><td>Ours</td><td>X</td><td>6.85 20.74</td><td>9.36</td></tr></table>

Table 6: Efect of joint human & object reconstruction scafold. Results are presented on InterCap. The two EasyHOI-style variants follow an object-only pipeline: the object is reconstructed by the LRM, and its pose is optimized by silhouette reprojection, either alone (first row) or with an additional contact term (second row). SAM3D (independent) instead reconstructs the human and object independently and composes them through MoGe [65] pointmap alignment. The use of a joint humanobject scafold proposed by MILO (last row) outperforms all baselines.

## 4.4 Comparison with individual reconstruction

MILO leverages the LRM reconstruction as a shared geometric scafold for HOI reconstruction: it fits a parametric human model to the human region of the reconstructed mesh and extracts the interacting object from the same scafold. In contrast, other recent methods [36] follow an object-only pipeline, in which an LRM is used to reconstruct the object mesh and the resulting shape is subsequently fitted using reprojection and contact constraints. For example, Easy-HOI [36] adopts this strategy for hand-object interaction reconstruction.

Although EasyHOI is not directly comparable to our setting because it considers hand-object rather than full-body interactions, we design a baseline that follows the same object-only paradigm. Specifically, we follow the same pipeline with EasyHOI [36]; first (1) we perform inpainting, then (2) we reconstruct the object with InstantMesh [72] and post-process the output mesh to make it watertight, and finally (3) we optimize the object pose using a silhouette reprojection loss. Additionally, we consider a version where a contact loss is added. We discuss details of contact estimation in the SuppMat.

As shown in Tab. 6, MILO outperforms both versions, with and without contact loss. This result indicates that using a joint reconstruction scafold is more efective for HOI reconstruction than an object-only reconstruction followed by fitting to the image. Moreover, we observe that contact cues are essential for these image-fitting approaches, since otherwise the location and scale of the object are mostly unconstrained (first row of Tab. 6). Removing the reliance on explicit contact estimation is one of the key advantages of MILO.

<table><tr><td>Human Fitting</td><td>Uses Template</td><td>(cm) ↓</td><td>PA-CDh PA-CD。 PA-CDh+o (cm) ↓</td><td>(cm) ↓</td><td>Human Fitting</td><td>Uses Template</td><td>(cm) ↓</td><td>PA-CDh PA-CD。 PA-CDh+o (cm) ↓</td><td>(cm) ↓</td></tr><tr><td>No fitting</td><td>X</td><td>30.58</td><td>23.53</td><td>14.10</td><td>No fitting</td><td>√</td><td>14.97</td><td>37.02</td><td>14.57</td></tr><tr><td>Root Fitting</td><td>x</td><td>7.98</td><td>22.53</td><td>10.15</td><td>Root Fitting</td><td>√</td><td>7.79</td><td>18.63</td><td>7.99</td></tr><tr><td>Pose Fitting</td><td>x</td><td>6.85</td><td>20.74</td><td>9.36</td><td>Pose Fitting</td><td>√</td><td>6.96</td><td>18.97</td><td>7.45</td></tr></table>

Table 7: Efect of human fitting. Results are presented on the InterCap dataset [20]. No fitting computes the CD metrics using the raw LRM mesh (adopting our pointcloud segmentation), without SMPL-H optimization. Root fitting optimizes only the global orientation and translation of the human. Pose fitting additionally optimizes body/hand pose and shape for SMPL-H. We consider both template and templatefree settings. The progressive reductions across metrics highlight the importance of the fitting stages and the necessity of a parametric model for precise 3D HOI reconstruction.

To further identify whether MILO’s gains come from the joint reconstruction or from the downstream fitting pipeline, we add a controlled baseline that reconstructs the human and the object separately. We follow the composition method of SAM3D [58,74] and run SAM3D independently on the human and the object regions of the same image. Then, we align the two resulting meshes in a common frame via MoGe [65] pointmap alignment, yielding the “SAM3D (independent)” baseline in Tab. 6. This baseline difers from the “SAM3D (joint)” baseline of Table 5 in that the geometry comes from two independent reconstructions instead of a single joint scafold, so we can isolate the value of joint reconstruction. As reported in Tab. 6, MILO clearly outperforms this baseline across all metrics. This substantial improvement confirms that the joint scafold, rather than the fitting procedure, is the main driver of MILO’s accuracy.

## 4.5 Ablation Experiments

We also evaluate the significance of the two fitting stages of the optimization pipeline. We perform this ablation using both the segmented LRM mesh and the aligned template (Tab. 7). As can be seen, removing the pose fitting or both fitting stages consistently degrades performance, confirming that both stages of our optimization are necessary to achieve the best overall reconstruction quality. Additional ablation experiments have been included in the SuppMat.

## 4.6 Qualitative Evaluation on In-The-Wild Images

To assess MILO’s robustness, we further evaluate it qualitatively on in-the-wild images from PICO-db [6]. As illustrated in Fig. 6, MILO recovers human and object meshes that are better aligned in 3D and exhibit more coherent contact patterns and relative positioning compared to prior work [6].

![](images/8eda231924660fd7c497eed4abf42f130b135784689b5e7674987ef6f23b67ab.jpg)  
Fig. 6: Qualitative reconstructions on in-the-wild images from PICO-db [6]. MILO recovers 3D human-object geometry from a single image, with coherent arrangement and fewer interpenetrations than the PICO-fit [6] baseline. Humans are rendered in blue, template objects in red, and the segmented object mesh in orange.

## 5 Conclusion

We introduced MILO, an approach for reconstructing human-object interactions from a single image. MILO leverages the visual capabilities of Large Reconstruction Models to predict a holistic human-object mesh. It then segments the mesh into human and object parts, fits the SMPL-H body to the human part, and optionally the object template to the object part, if such a template is available. By treating the LRM output as a geometric scafold, our method captures arrangement and proximity cues while remaining agnostic to object category and shape. Quantitative experiments on multiple benchmarks show that MILO surpasses existing HOI reconstruction methods despite using significantly less privileged information. Moreover, qualitative results on in-the-wild images demonstrate the generalization to diverse objects, poses, and interaction contexts. These results highlight the promise of LRMs for scalable, label-eficient HOI estimation.

Limitations and Future Work. One limitation of our approach is that the performance of MILO is bounded by the quality of the underlying LRM. However, given the quick progress in this space [58,60,62], we expect that the performance of LRMs will continue to improve. Based on our results, we also observe that our performance is hindered by the accuracy of the point cloud segmentation. We anticipate that with better methods for point-cloud segmentation, our approach will also benefit. Future work includes the refinement of LRM outputs, especially for smaller objects, integrating stronger instance-level reasoning and segmentation to extract a better object mesh from the LRM output, and extending the framework to video and multi-person/multi-object interaction settings.

## Acknowledgement

This research was supported by NSF-2504906, and 2544200; gifts from Adobe, Google, and Nvidia; and computing support on the Vista GPU Cluster through the Center for Generative AI (CGAI) and the Texas Advanced Computing Center (TACC) at the University of Texas at Austin.

## References

1. Bernardin, K., Ogawara, K., Ikeuchi, K., Dillmann, R.: A sensor fusion approach for recognizing continuous human grasping sequences using hidden markov models. Transactions on Robotics (T-RO) 21(1), 47–57 (2005)

2. Bhat, S.F., Birkl, R., Wofk, D., Wonka, P., Müller, M.: Zoedepth: Zero-shot transfer by combining relative and metric depth. arXiv preprint arXiv:2302.12288 (2023)

3. Bhatnagar, B.L., Xie, X., Petrov, I.A., Sminchisescu, C., Theobalt, C., Pons-Moll, G.: BEHAVE: Dataset and method for tracking human object interactions. In: Computer Vision and Pattern Recognition (CVPR). pp. 15935–15946 (2022)

4. Bogo, F., Kanazawa, A., Lassner, C., Gehler, P., Romero, J., Black, M.J.: Keep it SMPL: Automatic estimation of 3D human pose and shape from a single image. In: European Conference on Computer Vision (ECCV). vol. 9909, pp. 561–578 (2016)

5. Choy, C.B., Xu, D., Gwak, J., Chen, K., Savarese, S.: 3d-r2n2: A unified approach for single and multi-view 3d object reconstruction. In: European Conference on Computer Vision (ECCV). pp. 628–644. Springer (2016)

6. Cseke, A., Tripathi, S., Dwivedi, S.K., Lakshmipathy, A., Chatterjee, A., Black, M.J., Tzionas, D.: PICO: Reconstructing 3D people in contact with objects. In: Computer Vision and Pattern Recognition (CVPR). pp. 1783–1794 (June 2025)

7. Deitke, M., Liu, R., Wallingford, M., Ngo, H., Michel, O., Kusupati, A., Fan, A., Laforte, C., Voleti, V., Gadre, S.Y., VanderBilt, E., Kembhavi, A., Vondrick, C., Gkioxari, G., Ehsani, K., Schmidt, L., Farhadi, A.: Objaverse-XL: A universe of 10m+ 3D objects. arXiv:2307.05663 (2023)

8. Deitke, M., Schwenk, D., Salvador, J., Weihs, L., Michel, O., VanderBilt, E., Schmidt, L., Ehsani, K., Kembhavi, A., Farhadi, A.: Objaverse: A universe of annotated 3D objects. In: Computer Vision and Pattern Recognition (CVPR). pp. 13142–13153 (2023)

9. Dwivedi, S.K., Schmid, C., Yi, H., Black, M.J., Tzionas, D.: POCO: 3D pose and shape estimation using confidence. In: International Conference on 3D Vision (3DV). pp. 85–95 (2024)

10. Dwivedi, S.K., Sun, Y., Patel, P., Feng, Y., Black, M.J.: TokenHMR: Advancing human mesh recovery with a tokenized pose representation. In: Computer Vision and Pattern Recognition (CVPR). pp. 1323–1333 (2024)

11. Fan, H., Su, H., Guibas, L.J.: A point set generation network for 3d object reconstruction from a single image. In: 2017 IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2017, Honolulu, HI, USA, July 21-26, 2017. pp. 2463–2471. IEEE Computer Society (2017)

12. Fieraru, M., Zanfir, M., Oneata, E., Popa, A.I., Olaru, V., Sminchisescu, C.: Learning complex 3D human self-contact. In: AAAI Conference on Artificial Intelligence. pp. 1343–1351 (2021)

13. Geman, S., Mcclure, D.E.: Statistical methods for tomographic image reconstruction. Bulletin of the International Statistical Institute 52(4), 5–21 (1987)

14. Girdhar, R., Fouhey, D., Rodriguez, M.D., Gupta, A.: Learning a predictable and generative vector representation for objects. In: European Conference on Computer Vision (ECCV). vol. 9910, pp. 484–499 (2016)

15. Goel, S., Pavlakos, G., Rajasegaran, J., Kanazawa\*, A., Malik\*, J.: Humans in 4D: Reconstructing and tracking humans with transformers. In: International Conference on Computer Vision (ICCV). pp. 14737–14748 (2023)

16. Hassan, M., Ghosh, P., Tesch, J., Tzionas, D., Black, M.J.: Populating 3D scenes by learning human-scene interaction. In: Computer Vision and Pattern Recognition (CVPR). pp. 14708–14718 (2021)

17. Hasson, Y., Varol, G., Tzionas, D., Kalevatykh, I., Black, M.J., Laptev, I., Schmid, C.: Learning joint reconstruction of hands and manipulated objects. In: Computer Vision and Pattern Recognition (CVPR). pp. 11807–11816 (2019)

18. Hong, Y., Zhang, K., Gu, J., Bi, S., Zhou, Y., Liu, D., Liu, F., Sunkavalli, K., Bui, T., Tan, H.: LRM: Large reconstruction model for single image to 3D. arXiv: 2311.04400 (2023)

19. Huang, C.H., Yi, H., Höschle, M., Safroshkin, M., Alexiadis, T., Polikovsky, S., Scharstein, D., Black, M.J.: Capturing and inferring dense full-body human-scene contact. In: Computer Vision and Pattern Recognition (CVPR). pp. 13264–13275 (2022)

20. Huang, Y., Taheri, O., Black, M.J., Tzionas, D.: InterCap: Joint markerless 3D tracking of humans and objects in interaction from multi-view RGB-D images. International Journal of Computer Vision (IJCV) 132(7), 2551–2566 (2024)

21. Huo, C., Shi, Y., Ma, Y., Xu, L., Yu, J., Wang, J.: Stackflow: Monocular humanobject reconstruction by stacked normalizing flow with ofset. In: Proceedings of the Thirty-Second International Joint Conference on Artificial Intelligence, IJCAI 2023, 19th-25th August 2023, Macao, SAR, China. pp. 902–910. ijcai.org (2023)

22. Jiang, H., Huang, Q., Pavlakos, G.: Real3d: Towards scaling large reconstruction models with real images. In: International Conference on Computer Vision (ICCV). pp. 5821–5833 (October 2025)

23. Joo, H., Neverova, N., Vedaldi, A.: Exemplar fine-tuning for 3D human pose fitting towards in-the-wild 3D human pose estimation. In: International Conference on 3D Vision (3DV). pp. 42–52 (2021)

24. Kamakura, N., Matsuo, M., Ishii, H., Mitsuboshi, F., Miura, Y.: Patterns of static prehension in normal hands. American Journal of Occupational Therapy 34(7), 437–445 (1980)

25. Kerbl, B., Kopanas, G., Leimkühler, T., Drettakis, G.: 3d gaussian splatting for real-time radiance field rendering. ACM Trans. Graph. 42(4), 139:1–139:14 (2023)

26. Kocabas, M., Huang, C.H.P., Hilliges, O., Black, M.J.: PARE: Part attention regressor for 3D human body estimation. In: International Conference on Computer Vision (ICCV). pp. 11127–11137 (2021)

27. Kolotouros, N., Pavlakos, G., Daniilidis, K.: Convolutional mesh regression for single-image human shape reconstruction. In: Computer Vision and Pattern Recognition (CVPR). pp. 4496–4505 (2019)

28. Lassner, C., Romero, J., Kiefel, M., Bogo, F., Black, M.J., Gehler, P.: Unite the people: Closing the loop between 3D and 2D human representations. In: Computer Vision and Pattern Recognition (CVPR) (2017)

29. Li, J., Xu, C., Chen, Z., Bian, S., Yang, L., Lu, C.: HybrIK: A hybrid analyticalneural inverse kinematics solution for 3D human pose and shape estimation. In: Computer Vision and Pattern Recognition (CVPR). pp. 3383–3393 (2021)

30. Li, Z., Liu, J., Zhang, Z., Xu, S., Yan, Y.: CLIFF: Carrying location information in full frames into human pose and shape estimation. In: European Conference on Computer Vision (ECCV). vol. 13665, pp. 590–606 (2022)

31. Lin, J., Zeng, A., Wang, H., Zhang, L., Li, Y.: One-stage 3D whole-body mesh recovery with component aware transformer. In: Computer Vision and Pattern Recognition (CVPR). pp. 21159–21168 (2023)

32. Lin, K., Wang, L., Liu, Z.: End-to-end human pose and mesh reconstruction with transformers. In: Computer Vision and Pattern Recognition (CVPR). pp. 1954– 1963 (2021)

33. Lin, K., Wang, L., Liu, Z.: Mesh graphormer. In: International Conference on Computer Vision (ICCV) (2021)

34. Liu, R., Wu, R., Hoorick, B.V., Tokmakov, P., Zakharov, S., Vondrick, C.: Zero-1-to-3: Zero-shot one image to 3D object. International Conference on Computer Vision (ICCV) (2023)

35. Liu, S., Zeng, Z., Ren, T., Li, F., Zhang, H., Yang, J., Li, C., Yang, J., Su, H., Zhu, J., et al.: Grounding dino: Marrying dino with grounded pre-training for open-set object detection. arXiv preprint arXiv:2303.05499 (2023)

36. Liu, Y., Long, X., Yang, Z., Liu, Y., Habermann, M., Theobalt, C., Ma, Y., Wang, W.: Easyhoi: Unleashing the power of large models for reconstructing hand-object interactions in the wild. In: Computer Vision and Pattern Recognition (CVPR). pp. 7037–7047. Computer Vision Foundation / IEEE (2025)

37. Loper, M., Mahmood, N., Romero, J., Pons-Moll, G., Black, M.J.: SMPL: A skinned multi-person linear model. Transactions on Graphics (TOG) 34(6), 248:1– 248:16 (2015)

38. Ma, C., Li, Y., Yan, X., Xu, J., Yang, Y., Wang, C., Zhao, Z., Guo, Y., Chen, Z., Guo, C.: P3-sam: Native 3d part segmentation. arXiv preprint arXiv:2509.06784 (2025)

39. Melas-Kyriazi, L., Laina, I., Rupprecht, C., Vedaldi, A.: Realfusion 360° reconstruction of any object from a single image. In: Computer Vision and Pattern Recognition (CVPR). pp. 8446–8455. IEEE (2023)

40. Mildenhall, B., Srinivasan, P.P., Tancik, M., Barron, J.T., Ramamoorthi, R., Ng, R.: Nerf: Representing scenes as neural radiance fields for view synthesis. In: Vedaldi, A., Bischof, H., Brox, T., Frahm, J. (eds.) Computer Vision - ECCV 2020 - 16th European Conference, Glasgow, UK, August 23-28, 2020, Proceedings, Part I. Lecture Notes in Computer Science, vol. 12346, pp. 405–421. Springer (2020)

41. Moon, G., Choi, H., Lee, K.M.: Accurate 3d hand pose estimation for whole-body 3d human mesh estimation. In: Computer Vision and Pattern Recognition Workshops (CVPRw). pp. 2307–2316. IEEE (2022)

42. Müller, L., Osman, A.A.A., Tang, S., Huang, C.H.P., Black, M.J.: On self-contact and human pose. In: Computer Vision and Pattern Recognition (CVPR). pp. 9990– 9999 (2021)

43. Nam, H., Jung, D.S., Moon, G., Lee, K.M.: Joint reconstruction of 3D human and object via contact-based refinement transformer. In: Computer Vision and Pattern Recognition (CVPR) (2024)

44. Park, J.J., Florence, P., Straub, J., Newcombe, R., Lovegrove, S.: Deepsdf: Learning continuous signed distance functions for shape representation. In: Computer Vision and Pattern Recognition (CVPR). pp. 165–174 (2019)

45. Pavlakos, G., Choutas, V., Ghorbani, N., Bolkart, T., Osman, A.A.A., Tzionas, D., Black, M.J.: Expressive body capture: 3D hands, face, and body from a single image. In: Computer Vision and Pattern Recognition (CVPR). pp. 10975–10985 (2019)

46. Pavlakos, G., Shan, D., Radosavovic, I., Kanazawa, A., Fouhey, D., Malik, J.: Reconstructing hands in 3d with transformers. In: Computer Vision and Pattern Recognition (CVPR). pp. 9826–9836. IEEE (2024)

47. Petrov, I.A., Marin, R., Chibane, J., Pons-Moll, G.: Tridi: Trilateral difusion of 3d humans, objects, and interactions. In: International Conference on Computer Vision (ICCV). pp. 5523–5535 (October 2025)

48. Qian, G., Mai, J., Hamdi, A., Ren, J., Siarohin, A., Li, B., Lee, H., Skorokhodov, I., Wonka, P., Tulyakov, S., Ghanem, B.: Magic123: One image to high-quality 3d object generation using both 2d and 3d difusion priors. In: The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net (2024)

49. Ravi, N., Gabeur, V., Hu, Y.T., Hu, R., Ryali, C., Ma, T., Khedr, H., Rädle, R., Rolland, C., Gustafson, L., Mintun, E., Pan, J., Alwala, K.V., Carion, N., Wu, C.Y., Girshick, R., Dollár, P., Feichtenhofer, C.: Sam 2: Segment anything in images and videos (2024)

50. Rempe, D., Birdal, T., Hertzmann, A., Yang, J., Sridhar, S., Guibas, L.J.: HuMoR: 3D human motion model for robust pose estimation. In: International Conference on Computer Vision (ICCV). pp. 11468–11479 (2021)

51. Romero, J., Tzionas, D., Black, M.J.: Embodied hands: Modeling and capturing hands and bodies together. Transactions on Graphics (TOG) 36(6) (Nov 2017)

52. Rong, Y., Shiratori, T., Joo, H.: FrankMocap: A monocular 3D whole-body pose estimation system via regression and integration. In: International Conference on Computer Vision Workshops (ICCVw). pp. 1749–1759 (2021)

53. Shimada, S., Golyanik, V., Li, Z., Pérez, P., Xu, W., Theobalt, C.: HULC: 3D human motion capture with pose manifold sampling and dense contact guidance. In: European Conference on Computer Vision (ECCV). pp. 516–533 (2022)

54. Shimada, S., Golyanik, V., Xu, W., Theobalt, C.: PhysCap: Physically plausible monocular 3D motion capture in real time. Transactions on Graphics (TOG) 39(6), 235:1–235:16 (2020)

55. Taheri, O., Ghorbani, N., Black, M.J., Tzionas, D.: GRAB: A dataset of wholebody human grasping of objects. In: European Conference on Computer Vision (ECCV) (2020)

56. Tang, J., Chen, Z., Chen, X., Wang, T., Zeng, G., Liu, Z.: LGM: large multi-view gaussian model for high-resolution 3d content creation. In: Leonardis, A., Ricci, E., Roth, S., Russakovsky, O., Sattler, T., Varol, G. (eds.) Computer Vision - ECCV 2024 - 18th European Conference, Milan, Italy, September 29-October 4, 2024, Proceedings, Part IV. Lecture Notes in Computer Science, vol. 15062, pp. 1–18. Springer (2024)

57. Tang, J., Wang, T., Zhang, B., Zhang, T., Yi, R., Ma, L., Chen, D.: Make-it-3d: High-fidelity 3d creation from A single image with difusion prior. In: Computer Vision and Pattern Recognition (CVPR). pp. 22762–22772. IEEE (2023)

58. Team, S.D., Chen, X., Chu, F.J., Gleize, P., Liang, K.J., Sax, A., Tang, H., Wang, W., Guo, M., Hardin, T., Li, X., Lin, A., Liu, J., Ma, Z., Sagar, A., Song, B., Wang, X., Yang, J., Zhang, B., Dollár, P., Gkioxari, G., Feiszli, M., Malik, J.: Sam 3d: 3dfy anything in images (2025)

59. Team, T.H.: Hunyuan3d 2.0: Scaling difusion models for high resolution textured 3d assets generation (2025)

60. Tochilkin, D., Pankratz, D., Liu, Z., Huang, Z., Letts, A., Li, Y., Liang, D., Laforte, C., Jampani, V., Cao, Y.: Triposr: Fast 3d object reconstruction from a single image. CoRR abs/2403.02151 (2024)

61. Tripathi, S., Chatterjee, A., Passy, J.C., Yi, H., Tzionas, D., Black, M.J.: DECO: Dense estimation of 3D human-scene contact in the wild. In: International Conference on Computer Vision (ICCV). pp. 8001–8013 (2023)

62. Voleti, V., Yao, C., Boss, M., Letts, A., Pankratz, D., Tochilkin, D., Laforte, C., Rombach, R., Jampani, V.: SV3D: novel multi-view synthesis and 3d generation from a single image using latent video difusion. In: Leonardis, A., Ricci, E., Roth, S., Russakovsky, O., Sattler, T., Varol, G. (eds.) Computer Vision - ECCV 2024 - 18th European Conference, Milan, Italy, September 29-October 4, 2024, Proceedings, Part I. Lecture Notes in Computer Science, vol. 15059, pp. 439–457. Springer (2024)

63. Wang, N., Zhang, Y., Li, Z., Fu, Y., Liu, W., Jiang, Y.: Pixel2mesh: Generating 3d mesh models from single RGB images. In: Ferrari, V., Hebert, M., Sminchisescu, C., Weiss, Y. (eds.) Computer Vision - ECCV 2018 - 15th European Conference, Munich, Germany, September 8-14, 2018, Proceedings, Part XI. Lecture Notes in Computer Science, vol. 11215, pp. 55–71. Springer (2018)

64. Wang, P., Tan, H., Bi, S., Xu, Y., Luan, F., Sunkavalli, K., Wang, W., Xu, Z., Zhang, K.: PF-LRM: pose-free large reconstruction model for joint pose and shape prediction. In: The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net (2024)

65. Wang, R., Xu, S., Dai, C., Xiang, J., Deng, Y., Tong, X., Yang, J.: Moge: Unlocking accurate monocular geometry estimation for open-domain images with optimal training supervision. In: Computer Vision and Pattern Recognition (CVPR). pp. 5261–5271. Computer Vision Foundation / IEEE (2025)

66. Wang, Z., Zheng, Q., Ma, S., Ye, M., Zhan, Y., Li, D.: End-to-end HOI reconstruction transformer with graph-based encoding. In: Computer Vision and Pattern Recognition (CVPR). pp. 27706–27715. Computer Vision Foundation / IEEE (2025)

67. Wen, B., Huang, D., Zhang, Z., Zhou, J., Deng, J., Gong, J., Chen, Y., Ma, L., Li, Y.L.: Reconstructing in-the-wild open-vocabulary human-object interactions. In: Computer Vision and Pattern Recognition (CVPR). pp. 17426–17436 (2025)

68. Wu, J., Pavlakos, G., Gkioxari, G., Malik, J.: Reconstructing hand-held objects in 3d. arXiv preprint arXiv:2404.06507 (2024)

69. Xie, X., Bhatnagar, B.L., Lenssen, J.E., Pons-Moll, G.: Template free reconstruction of human-object interaction with procedural interaction generation. In: Computer Vision and Pattern Recognition (CVPR) (2024)

70. Xie, X., Bhatnagar, B.L., Pons-Moll, G.: CHORE: contact, human and object reconstruction from a single RGB image. In: European Conference on Computer Vision (ECCV). vol. 13662, pp. 125–145 (2022)

71. Xu, H., Bazavan, E.G., Zanfir, A., Freeman, W.T., Sukthankar, R., Sminchisescu, C.: GHUM & GHUML: Generative 3D human shape and articulated pose models. In: Computer Vision and Pattern Recognition (CVPR). pp. 6183–6192 (2020)

72. Xu, J., Cheng, W., Gao, Y., Wang, X., Gao, S., Shan, Y.: Instantmesh: Eficient 3d mesh generation from a single image with sparse-view large reconstruction models. CoRR abs/2404.07191 (2024)

73. Xu, Y., Zhang, J., Zhang, Q., Tao, D.: Vitpose: Simple vision transformer baselines for human pose estimation. In: Koyejo, S., Mohamed, S., Agarwal, A., Belgrave, D., Cho, K., Oh, A. (eds.) Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022 (2022)

74. Yang, X., Kukreja, D., Pinkus, D., Fan, T., Park, J., Shin, S., Cao, J., Liu, J.W., Ugrinovic, N., Sagar, A., et al.: Sam 3d body: Robust full-body human mesh recovery. In: Computer Vision and Pattern Recognition (CVPR). pp. 7209–7219 (2026)

75. Yang, Y., Zhai, W., Luo, H., Cao, Y., Zha, Z.J.: LEMON: Learning 3D humanobject interaction relation from 2D images. In: Computer Vision and Pattern Recognition (CVPR) (2024)

76. Zhang, J.Y., Pepose, S., Joo, H., Ramanan, D., Malik, J., Kanazawa, A.: Perceiving 3D human-object spatial arrangements from a single image in the wild. In: European Conference on Computer Vision (ECCV). vol. 12357, pp. 34–51 (2020)

77. Zhang, J., Herrmann, C., Hur, J., Chen, E., Jampani, V., Sun, D., Yang, M.H.: Telling left from right: Identifying geometry-aware semantic correspondence. In: Computer Vision and Pattern Recognition (CVPR) (2024)

78. Zhang, J., Luo, H., Yang, H., Xu, X., Wu, Q., Shi, Y., Yu, J., Xu, L., Wang, J.: Neuraldome: A neural modeling pipeline on multi-view human-object interactions. In: Computer Vision and Pattern Recognition (CVPR) (2023)

79. Zhang, S., Zhang, Y., Bogo, F., Pollefeys, M., Tang, S.: Learning motion priors for 4D human body capture in 3D scenes. In: International Conference on Computer Vision (ICCV). pp. 11343–11353 (2021)

80. Zhao, C., Zhang, J., Du, J., Shan, Z., Wang, J., Yu, J., Wang, J., Xu, L.: I’m hoi: Inertia-aware monocular capture of 3d human-object interactions. In: Computer Vision and Pattern Recognition (CVPR). pp. 729–741 (2024)

# Supplementary Material for: Reconstructing Humans and Objects in Interaction using Large Reconstruction Models

Agniv Chatterjee Georgios Pavlakos

University of Texas at Austin agniv.chatterjee@utexas.edu, pavlakos@cs.utexas.edu

In this Supplementary Material, we provide implementation details, additional analysis, and extended results that were omitted from the main paper due to space constraints. We first expand on the $\mathrm { M L O }$ pipeline in Sec. S.1, including details about multiview rendering of the LRM mesh (which is used for keypoint detection, pointcloud segmentation and correspondence estimation), keypoint triangulation, object template alignment and pointcloud segmentation. We then present further qualitative results in Sec. S.2, including discussion (Sec. S.2.1) and representative failure cases (Sec. S.2.2). Section S.3 provides contact evaluation details, and Sec. S.4 discusses the interaction cues captured by our method as well as its limitations. In Sec. S.5, we provide additional quantitative results. Finally, Sec. S.6 presents a runtime breakdown of the full pipeline for recovering human-object geometry from a single RGB image.

## S.1 MILO implementation details

Multi-view rendering: For keypoint detection, correspondence estimation, and mesh segmentation, we render the LRM mesh from multiple viewpoints, using various values for the elevation and azimuth angles. Specifically, we pick angles at intervals of 30 degrees for the azimuth angle, while for the elevation we consider the angles of $0 , \pm 3 0$ , and ±60 degrees. This leads to a total count of 60 viewpoints.

3D keypoint estimation: For hand keypoint triangulation, hand detection may return multiple candidates for the same hand in one view. We therefore retain a single candidate per side before triangulation. If the corresponding body wrist (i.e., wrist keypoint in the body keypoints list) is available, we choose the candidate whose centroid is closest to the body wrist in the image plane:

$$
m ^ { \star } = \arg \operatorname* { m i n } _ { m } \left\| \mu { \left( x _ { h , s } ^ { ( v , m ) } \right) } - x _ { b , \mathrm { w r i s t } ( s ) } ^ { ( v ) } \right\| _ { 2 } ,
$$

where $\boldsymbol { x } _ { h , s } ^ { ( v , m ) } \in \mathbb { R } ^ { 2 1 \times 2 }$ is the m-th candidate and $\mu ( \cdot )$ denotes the centroid over the 21 hand joints. We use the centroid rather than the wrist keypoint of the candidate’s hand because the wrist lies at the arm-hand boundary and is often poorly localized. Thus, averaging over all 21 keypoints yields a more stable spatial summary of the detection bounding box. If the body wrist is unavailable, we fall back to the candidate with the largest image-plane extent. This produces at most one left-hand and one right-hand observation per view.

Next, we provide details about the confidence filtering, triangulation, and aggregation, summarized in Sec. 3.3 of the main paper.

Confidence filtering. For each keypoint index $k ,$ we keep only views whose confidence exceeds a threshold $\gamma$ (empirically set to 0.6): $\mathcal { O } _ { k } = \Big \{ v \in \mathcal { V } \Big | c _ { k } ^ { ( v ) } > \gamma \Big \}$ $\mathrm { I f } \ | \mathcal { O } _ { k } | < N _ { \mathrm { m i n } } = 3 .$ , the keypoint is marked as invalid and excluded from fitting. Robust triangulation. For each keypoint, we triangulate candidate 3D points using all pairs of observed views. We select the hypothesis with the largest consensus set, i.e., the most inliers under reprojection-error threshold \tau (set to 5.0). The final 3D point is obtained by nonlinear refinement over the inlier views $( \mathcal { T } _ { k } ^ { \star } )$

$$
X _ { k } ^ { \star } = \arg \operatorname* { m i n } _ { X \in \mathbb { R } ^ { 3 } } \sum _ { v \in \mathcal { T } _ { k } ^ { \star } } \left\| \pi _ { v } ( X ) - x _ { k } ^ { ( v ) } \right\| _ { 2 } ^ { 2 } .\tag{S.1}
$$

Confidence aggregation and output. For each valid keypoint, we compute an aggregated reliability score as the mean confidence over its inlier views: $\bar { c } _ { k } =$ $\begin{array} { r } { \frac { 1 } { \left| \mathcal { T } _ { k } ^ { \star } \right| } \sum _ { v \in \mathcal { T } _ { k } ^ { \star } } c _ { k } ^ { ( v ) } } \end{array}$ Applying this to body and hand joints yields $X _ { b } \in \mathbb { R } ^ { 2 5 \times 3 } , X _ { h , \mathrm { L } } \in$ $\mathbb { R } ^ { 2 1 \times 3 }$ and $X _ { h , \mathrm { R } } \in \mathbb { R } ^ { 2 1 \times 3 }$ which we concatenate into $X = [ X _ { b } ; X _ { h , \mathrm { L } } ; X _ { h , \mathrm { R } } ] \in$ $\mathbb { R } ^ { 6 7 \times 3 }$ and $\bar { c } \in \mathbb { R } ^ { 6 7 }$ The pair $( X , { \bar { c } } )$ defines a sparse, confidence-weighted set of 3D keypoints that are used in the next stage to fit SMPL-H to the LRM mesh. Object template alignment: For the template alignment, if reliable semantic correspondences between the template mesh and the segmented object point cloud cannot be established, we initialize alignment using centroid matching and scale normalization, and then refine the result using ICP only. This typically occurs under severe occlusion of the object by the human or for very small objects, where the LRM reconstruction does not recover the object well.

Pointcloud segmentation: We expand here on the segmentation procedure summarized in Sec. 3.5, in the main paper. For each viewpoint k, we first compute a viewpoint quality score $Q _ { k }$ that reflects both geometric coverage and segmentation reliability:

$$
Q _ { k } = \alpha \cdot { \frac { | V _ { k } ^ { \mathrm { v i s i b l e } } | } { | V | } } + ( 1 - \alpha ) \cdot { \frac { | M _ { k } ^ { \mathrm { h u m a n } } | + | M _ { k } ^ { \mathrm { o b j e c t } } | } { | I _ { k } | } } ,\tag{S.2}
$$

where $V _ { k } ^ { \mathrm { v i s i b l e } }$   
and $M _ { k } ^ { \mathrm { o b j e c t } }$ are the human and object masks respectively, $I _ { k }$ is the set of image pixels, and $\alpha \in [ 0 , 1 ]$ balances visibility and mask quality. The viewpoint quality takes into account both the ratio of vertices visible in that viewpoint and the quality of the human and object masks.

To better handle contact regions, we construct a multi-scale human-object boundary by dilating both masks with kernel sizes of 3, 5, and 7 and intersecting the dilated regions. We compute a boundary closeness weight using Euclidean distance transform from the boundary, normalized by a maximum distance threshold, and transformed through a non-linear function with an optional base score boost for highly overlapped regions. This penalizes vertices near the boundary more heavily, while still allowing those farther away to be segmented. The optional boost is applied when the size of the object is much smaller compared to the human - the boundary region covers most of the object. For a projected vertex $p _ { i }$ , with distance $d ( p _ { i } , B )$ to this boundary $B ,$ we define a boundary-closeness weight:

$$
w _ { i } ^ { \mathrm { b o u n d a r y } } = \left( \operatorname* { m i n } \left( \frac { d ( p _ { i } , B ) } { d _ { \operatorname* { m a x } } } , 1 \right) \right) ^ { \gamma } + b ,\tag{S.3}
$$

where $d _ { \mathrm { m a x } }$ is a distance threshold for normalization, $\gamma \in [ 0 . 3 , 0 . 8 ]$ is an adaptive exponent determined by the human-object overlap ratio, and $b \geq 0$ is an optional base boost for highly overlapped regions. This encourages accurate labeling near the human-object contact regions. Finally, for each vertex $V _ { i }$ , we aggregate scores over all viewpoints $K _ { i }$ where it is visible:

$$
w _ { i } = \sum _ { k \in { \cal K } _ { i } } Q _ { k } \cdot w _ { i } ^ { \mathrm { b o u n d a r y } } ( V _ { i } ) ,\tag{S.4}
$$

and apply a threshold on $\{ w _ { i } \}$ to obtain final human and object labels. Vertices labeled as non-human form the raw object point cloud. We then apply geometric filtering: statistical outlier removal, pruning vertices with too few neighbors within a local radius, and keeping only the largest DBSCAN cluster. The resulting clean object point cloud is composed with the fitted SMPL-H mesh.

## S.2 Further Qualitative Results

We provide further qualitative results on PICO-db [6] (Fig. S.1), HODome [78] (Fig. S.2) and IMHD [80] (Fig. S.3) datasets. In these cases, we use PICO [6] as a baseline for comparison. For PICO-db, we visualize only the non-template variant of MILO, since the dataset only provides retrieved object meshes, which may not be representative of the true object geometry. Below, we discuss the qualitative trends observed across these datasets.

As can be seen from the qualitative results, our reconstructions better preserve the global object geometry, while accurately positioning it relative to the body. This is particularly visible for large carried or supported objects such as backpacks, suitcases, chairs, tables, and box-like objects, where PICO often produces implausible orientation or placement, whereas our method yields configurations that are more consistent with the image evidence.

On PICO-db (Fig. S.1), the non-template variant is visually more accurate than PICO in everyday in-the-wild scenes, despite the large variation in appearance, viewpoint, and object category. On HODome and IMHD (Fig. S.2 and Fig. S.3 respectively), where template models are available, the template-based variant further improves geometric regularity and alignment. This is especially noticeable for rigid objects with canonical shapes, such as chairs, tables, and suitcases, where template alignment leverages a more precise object geometry while preserving the overall interaction. In contrast, the gap between the template and

PICO

Ours

PICO

Ours

![](images/528e0fc144b5525a055c993b272f9af80b796ab6277cb752f31ab60ebdc415ea.jpg)  
Fig. S.1: Qualitative evaluation of PICO [6] and MILO on images from PICO-db [6]. We visually compare the non-template version of MILO with PICO, and output more accurate and physically plausible interactions, compared to PICO, without using ground-truth contact annotations.

non-template variants is smaller for slender objects, such as bats, for which the main challenge is pose estimation rather than detailed surface recovery.

Finally, as can be seen from the visual results, even when the recovered object geometry is not very detailed, the reconstructed interaction is often more physically plausible than that of PICO, which is critical for downstream reasoning about contact, support, and afordances.

## S.2.1 Results Discussion

For our quantitative evaluation on InterCap, we do not include HDM [69] in Tab. 1 of the main manuscript, since HDM is trained with ProciGen [69], whose construction draws on both BEHAVE [3] and InterCap [20], so it has a competitive advantage over methods not trained on InterCap.

![](images/a2d36b668c2415883a710b9ceaf2a37c241ac88e550d9d53ef9a30a96c7f2b0b.jpg)  
Fig. S.2: Qualitative evaluation of PICO [6] and MILO on images from the HODome dataset [78]. Our outputs, with or without templates, are more physically plausible and closer to the GT than PICO.

For the evaluation of PICO [6] on HODome [78] and IMHD [80], we found that the retrieval-based strategy they proposed for the non-ground-truth-contact setting is often unsuccessful in these datasets. In particular, several object categories in HODome, namely flower, box, pan, pillow, pink, and trashcan, and in IMHD, namely dumbbell, kettlebell, pan, and broom, do not have suficiently similar counterparts in PICO-db, where the template retrieval is happening from. In addition, even when corresponding categories exist, diferences in object geometry and interaction characteristics often afect retrieval quality. Accordingly, to obtain a more representative evaluation of PICO in this setting, we use the ground-truth template object meshes instead. To obtain paired human-object contact labels, we first use DECO [61] to predict human vertices that are in contact. From the predicted human vertices, we only consider the ones that are indeed in contact according to the ground-truth contact maps. For these human vertices, we obtain the human-object vertex contact pairs from the ground truth human-object contact annotations.

Regarding our qualitative results, we use LRM meshes as a geometric scafold to pose the parametric human meshes without optimizing using any reprojection objectives. Instead, we opt for 3D consistency and accuracy of the interaction between humans and objects. Our observation is that the LRM meshes generated by Hunyuan3D-2.0 [59] and other related approaches [22, 60, 62, 72] are generated in a coordinate frame diferent from the camera coordinate frame of the input image. Thus, our reconstruction output would require an additional transformation to align with the input RGB image.

![](images/ce1fac1a48923b92ec96dd92bec4dae624aae155e5402279c68a4e5d0696fb5e.jpg)  
Fig. S.3: Qualitative evaluation of PICO [6] and MILO on images from the IMHD dataset [80].

## S.2.2 Failure Cases

We visualize some failures of MILO in Fig. S.4. Failures typically occur either due to the quality of the LRM reconstruction or due to the quality of the point cloud segmentation stage. For larger, more complicated objects, the LRM may fail to reconstruct the entirety of the object. This also occurs in cases where the object is truncated. Also, the LRM might miss the boundary between the object and the background, causing elements of the background to seep into the reconstruction. Finally, for instances where the human is interacting with multiple objects, the LRM might reconstruct these additional objects. Regarding the point cloud segmentation stage, this is reliant on the quality of the multiview segmentation maps. Thus, if the segmentation maps are noisy, it might be unable to filter out certain elements of the scene.

We also show representative failures of the template-alignment module in Fig. S.5. The quality of the recovered alignment is determined by the semantic correspondences established between template renders and LRM renders. This can be sensitive to appearance quality, lighting, occlusion, and object symmetry. We observe two common failure modes. First, for nearly symmetric objects with less distinctive parts, the recovered alignment may be flipped. Second, when the object is only partially visible in the input image, the selected LRM views tend to emphasize the visible portion, and the resulting correspondences may align the template only to that partial geometry rather than to the full object.

Quantitatively, template alignment is most beneficial for thin or small objects, where the LRM tends to produce incomplete geometry. For example, PA-CD<sub>o</sub> improves substantially after alignment for the bottle (34.23 → 9.97 cm) and cup (42.91 → 18.64 cm) categories. For larger objects whose raw LRM mesh is already reasonable, it can instead degrade the result, as observed for the trolley case (17.46 → 21.79 cm), chair (12.22 → 18.77 cm), and umbrella (17.88 → 26.76 cm). The dominant error source is the symmetry-induced orientation flips noted above: mean rotation error reaches 90<sup>◦</sup>-120<sup>◦</sup> for symmetric objects, compared to

![](images/1e365603513c007d1873b5294bd7e9c78b46412e813a08c6fd55848a239be768.jpg)  
Fig. S.4: Representative failure cases of MILO. Typical errors arise from incorrect LRM reconstructions or imperfect point-cloud segmentation, leading to incomplete/warped geometry for large objects (mid-left), ambiguity when multiple objects fall into a single segmentation (top-right), objects being truncated in the image (topright, bottom-left), leakage of non-object parts (e.g., shoes) into the object mask (topleft), and unfiltered scattered points around the object (mid-right, bottom row).

21.71<sup>◦</sup> for the chair, which provides stronger orientation cues. Scale recovery is generally reliable, with the umbrella being the main outlier (scale error 0.918, versus 0.27 on average for the remaining categories). The box example in Fig. S.5 illustrates this efect.

## S.3 Contact Evaluation

For contact evaluation, we use the RICH checkpoint of DECO as our baseline. This choice yields a fair comparison, as the best DECO checkpoint is obtained from training with InterCap in its training set. We do not evaluate contact metrics for the other HOI reconstruction baselines because, except for HOI-TG, these methods explicitly rely on contact information as an input to reconstruction. By contrast, our method does not use contact annotations or contact priors at test time. We therefore compare against a state-of-the-art contact estimation network rather than against reconstruction methods that are given contact supervision.

To derive contact labels from our output, we first align the object template to the segmented object geometry. We then classify a human vertex as being in contact with the object if its Euclidean distance to the aligned object surface is smaller than 5 mm.

![](images/89dd0f815c7d3fb334c7442520a4742294deced8a4513ebf82e903593155dc60.jpg)  
Fig. S.5: Representative failure cases of the template alignment module. Errors typically occur from incorrect semantic correspondences obtained due to the symmetrical nature of some objects, missing regions in the segmented object mesh, and occlusions, which cause correspondences only to visible regions.

## S.4 Interaction Cues and Limitations

MILO does not explicitly encode interaction using contact annotations, interaction labels, or object-specific priors. Instead, it leverages the geometric scafold provided by the LRM reconstruction, which implicitly captures relative arrangement and proximity cues between the human and the object. Based on this scafold, we recover the human mesh, segment the object component, and optionally align an object template when one is available.

This formulation enables reconstruction of plausible human-object configurations without requiring explicit contact supervision. As a result, the recovered geometry is also informative for downstream contact estimation, where our method performs competitively despite not predicting contact directly.

Our method can fail for small objects under severe occlusion. For larger objects, partial occlusion is typically handled well via LRM priors (bottom-left example, Fig. S.1; second example, Fig. S.3), while occluded body regions are plausibly completed by SMPL-H. Purely 2D-grounded baselines are equally, if not more, brittle under heavy occlusions. More broadly, without access to explicit object-shape information, the problem is particularly challenging.

## S.5 Additional Experiments

## S.5.1 Ablation Experiments

Here, we provide additional ablation experiments. First, we consider the use of alternative methods for point cloud segmentation. Specifically, in Table S.1, we compare our proposed segmentation module with the recent P3-SAM method [38]. As can be seen, our strategy performs substantially better, which justifies the design of our approach.

<table><tr><td>Segmentation Used</td><td> $\overline { { { \mathrm { P A - C D } _ { h } \ \mathrm { P A - C D } _ { o } \ \mathrm { P A - C D } _ { h + o } } } }$  (cm) ↓ (cm) ↓</td><td>(cm) ↓</td></tr><tr><td>P3-SAM [38]</td><td>7.30 55.86</td><td>14.68</td></tr><tr><td>Ours</td><td>6.85 20.74</td><td>9.36</td></tr></table>

Table S.1: Comparison with P3- SAM. Results presented on Inter-Cap [20].

<table><tr><td>Method Used</td><td> $\sqrt { \mathrm { P A } \mathrm { - } \mathrm { C D } _ { h } }$  PA-CD (cm) ↓ (cm) ↓</td><td> $\overline { { \mathrm { P A - C D } _ { h + o } } }$  (cm) ↓</td></tr><tr><td>Kabsch</td><td>7.16 19.98</td><td>7.53</td></tr><tr><td>Kabsch + ICP</td><td>6.96 18.97</td><td>7.45</td></tr></table>

Table S.2: Template alignment ablation. Results presented on InterCap [20].

Next, in Tab. S.2, we ablate the stages of our template alignment pipeline. Using ICP on top of the initial weighted Kabsch fit consistently improves alignment across all metrics. The geometry-aware correspondences provide a reliable coarse similarity transform, while ICP further refines the local geometric fit to the segmented LRM object point cloud.

Finally, we ablate the point cloud segmentation pipeline. Specifically, we analyze the efect of (i) visibility coverage and mask quality, which together define the viewpoint quality score Q, and (ii) the boundary-closeness weight w<sup>boundary</sup>, which prevents confusion in vertices near human-object contact regions. Dropping either visibility coverage or mask quality would mean the quality score is wholly computed from the value of the other term, while the boundary-closeness term can be dropped by setting it to unity. The results in Tab. S.3 indicate that dropping any of these components leads to worse PA-CD. This efectively highlights that all elements of our segmentation design are important for obtaining clean object point clouds from the combined LRM mesh and, consequently, high-quality HOI reconstructions.

## S.5.2 Individual object reconstruction

Here, we provide more details about the EasyHOI-like experiment we present in the main manuscript. The contact loss used in EasyHOI [36] is not directly applicable to our setting because it enforces contact based on hand-object overlap. In contrast, our notion of contact is not limited to the hands, and occlusion cannot be assumed to be a reliable indicator of contact. Instead, we implement a contact-constrained variant that incorporates a distance-based contact term using ground-truth human contact vertices. Concretely, we first extract the set of human vertices annotated as being in contact, and then penalize the distance from these vertices to the object. For each contact vertex, we compute the Euclidean distance to its nearest vertex on the transformed object mesh and define the penalty as the average of these minimum distances over all contact vertices.

## S.5.3 Open3DHOI baseline

In this subsection, we provide more details about our implementation of the Open3DHOI [67] baseline reported in Tab. 1 of the main paper. As the preprocessing code for Open3DHOI (object extraction and coarse alignment) has not been publicly released, we reimplement the coarse-reconstruction stage from the Open3DHOI manuscript, which reconstructs the human and object independently and places them in a common camera frame using monocular depth estimates.

<table><tr><td></td><td>vis cover mask_quality boundary_dist</td><td>(cm) ↓</td><td>PA-CDh PA-CD。 PA-CDh+o (cm) ↓</td><td>(cm) ↓</td></tr><tr><td></td><td></td><td></td><td>6.85 20.74</td><td>9.36</td></tr><tr><td>X</td><td></td><td></td><td>7.62 22.81</td><td>9.82</td></tr><tr><td></td><td>X</td><td></td><td>7.61 22.55</td><td>9.83</td></tr><tr><td></td><td></td><td>X</td><td>7.57 21.75</td><td>9.77</td></tr></table>

Table S.3: Ablation of point-cloud segmentation. Results are reported on InterCap [20]. The columns represent the visibility coverage (vis\_cover) and the mask quality (mask\_quality), together defining the viewpoint quality score Q, and the boundary-closeness weight (boundary\_dist). Removing any component degrades performance, confirming each term’s contribution to clean point cloud segmentation.

We estimate per-frame SMPL-X parameters with OSX [31] and obtain a watertight object mesh with InstantMesh [72], the latter expressed in a normalized canonical frame with unknown scale, orientation, and translation. To recover the scene layout, we predict monocular depth with ZoeDepth [2]. Since this depth is defined only up to a global scale, we calibrate it against the human: projecting the SMPL-X vertices into the image, we take the median ratio between cameraspace and predicted depth over pixels within the person mask as the metric scale. Back-projecting the calibrated depth under the human and object masks yields a person and an object point cloud, with a median-absolute-deviation filter applied to the latter to suppress depth spikes at mask boundaries.

We align the human by translation only, shifting the SMPL-X root so that the mesh centroid coincides with the person point-cloud centroid, while keeping the OSX pose and scale fixed. For the object, scale and translation are taken from the corresponding InterCap ground-truth mesh (scale from the ratio of median radii, translation from the centroid), and orientation is recovered by ICP between the InstantMesh mesh and the object point cloud. The resulting human and object meshes, expressed in a shared camera frame, form the coarse initialization subsequently refined by the HOI-Gaussian optimizer.

## S.6 Runtime Analysis

All experiments are performed on an NVIDIA RTX 6000 Ada Generation GPU. Table S.4 reports the runtime of each stage in our pipeline. For the core pipeline, the total runtime is 344.40 s per image.

We highlight that our core optimization stages require a total runtime of 32.3s, which is significantly lower compared to the three stages of our primary baseline PICO, which requires 475.84s (i.e., roughly half a minute for us, vs 8 minutes for PICO). The 2D keypoint prediction, 3D keypoint triangulation, root fitting, pose fitting, and mesh segmentation stages together require 66.0s, with 2D keypoint prediction (15.95s) and pose fitting (21.60s) the largest contributors among them.

<table><tr><td>Stage</td><td>Time taken (s)</td></tr><tr><td>LRM reconstruction</td><td>189.00</td></tr><tr><td>Multi-view rendering</td><td>89.40</td></tr><tr><td>2D keypoint prediction</td><td>15.95</td></tr><tr><td>3D keypoint triangulation</td><td>7.95</td></tr><tr><td>Root fitting</td><td>10.70</td></tr><tr><td>Pose fitting</td><td>21.60</td></tr><tr><td>Mesh segmentation</td><td>9.80</td></tr><tr><td colspan="2">Optional template-alignment steps</td></tr><tr><td>Semantic correspondences</td><td>375.75</td></tr><tr><td>Template alignment</td><td>0.21</td></tr></table>

Table S.4: Runtime breakdown of each stage in our pipeline on a single NVIDIA RTX 6000 Ada GPU. The first seven rows correspond to the core pipeline, whose total runtime is 344.40 s per image. The last two rows report optional templatealignment steps, which are only used when template-based object alignment is enabled.

The final two rows in Table S.4 correspond to the template-alignment steps that are optional and are only used when a template is available. In this setting, the additional cost is dominated by semantic correspondence estimation (375.75 s), while the final template alignment itself is negligible (0.21 s). The runtime of semantic correspondence estimation is primarily incurred by running DINO, Stable Difusion, and an aggregation network on all renders of the LRM mesh and object template, and then performing dense vertex-level correspondence estimation. Overall, these results show that the runtime of the core pipeline is primarily bottlenecked by the upstream LRM reconstruction and rendering stages, whereas the optional template-based refinement is dominated by correspondence computation.