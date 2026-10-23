# H-2DGS: Height-Field Gaussian Splatting for Sharp and Deformable Neurosurgical Surface Reconstruction

Zhang Zichuan<sup>1</sup>, Bart Brouwers<sup>2</sup>, Rembrandt Bakker<sup>1,2</sup>, Maud Landers<sup>2</sup>, Geert-Jan Rutten<sup>1,2</sup>, and Maxime Chamberland<sup>1</sup>

<sup>1</sup>Department of Computer Science & Mathematics, Eindhoven University of Technology, Eindhoven, The Netherlands <sup>2</sup>Department of Neurosurgery, Elisabeth-Tweesteden Hospital Tilburg, The Netherlands

## Abstract

Neurosurgical navigation commonly presents magnetic resonance imaging (MRI) as 2D slices, making it difficult to relate the anatomy in those slices to the cortical surface observed through the surgical microscope. We seek a common 3D visualization that combines a photorealistic reconstruction of the exposed cortex during tumor surgery with whole-brain MRI context and diffusion tractography pathways. This requires the reconstructed cortex to form a clear boundary around the brain anatomy rather than a diffuse volume that obscures internal structures. Conventional 3D Gaussian Splatting (3DGS) can reproduce observed views but, under limited surgical view coverage, may distribute opacity through a thick and geometrically ambiguous band. We introduce height-field 2D Gaussian Splatting (H-2DGS), a three-stage pipeline that concentrates a robust 3DGS reconstruction, fuses its depth into a shared truncated signed distance field (TSDF) surface, and represents that surface with curved, surface-oriented Gaussian primitives. Separate height, color, and opacity bases provide local geometry and appearance, while a neighbor-transition objective coordinates overlapping patches. The explicit surface further supports postoperative alignment: MRI-derived depth deforms the reconstruction, and tracked-color supervision with alternating optimization preserves its appearance. Across five real neurosurgical cases, H-2DGS gives low multi-view depth inconsistency, a compact 7.97 MB representation, and interactive rendering at 29–42 ms per frame. The resulting sharp, deformable surface integrates clearly with tractography and MRI anatomy, including a postoperative resection cavity.

## 1 Introduction

Medical visualization plays a central role in the interpretation, communication and education of neuroanatomy and tractography. Interactive visualization is important in, e.g., neurosurgical planning, as it enables users to explore anatomical structures and fiber pathways. Modern neurosurgery uses dedicated navigation systems for surgical planning, intraoperative localization, and postoperative assessment. These systems commonly present registered magnetic resonance imaging (MRI) as orthogonal 2D slices [6, 13]. The slices contain rich tissue information, but they do not resemble the exposed cortical surface seen through the surgical microscope. Consequently, understanding where a visible vessel, sulcus, or resection boundary lies within the MRI volume requires a mental correspondence between two very different views. This separation increases cognitive load, and the missing 3D spatial context makes the relationship less intuitive [3].

We instead aim to place the complementary information in one 3D scene. Microscope video supplies the current color and fine surface landmarks of the exposed cortex. MRI supplies the surrounding whole-brain tissue context, while diffusion MRI tractography exposes subcortical pathways that are not visible at the surface. A photorealistic 3D reconstruction can connect what the surgeon sees to these internal structures under continuous viewpoint changes: the cortex provides the visible outer layer, the MRI-derived model locates that layer within the brain, and tractography shows the structures beneath it. The fused views in Fig. 1 illustrate this intended relationship.

Creating this visualization is not only a registration problem; it first requires an appropriate representation of the visible layer. 3D Gaussian Splatting (3DGS) [11] is attractive because it reconstructs realistic appearance, renders rapidly, and exposes primitives that can be manipulated. It has consequently been adopted for static and deforming surgical scenes [7, 9, 10, 16, 17, 22]. However, each conventional 3D Gaussian distributes opacity through a volume. With the narrow camera trajectories typical of microscope video, Gaussians at several depths can explain the same training pixels. The model may therefore look realistic from observed views while forming a thick, blurry, or fragmented density band elsewhere. When tractography is rendered beneath this band, fibers can be obscured by or visually embedded in the reconstructed cortex, so the boundary between visible surface and subcortical anatomy becomes ambiguous.

![](images/5b3e7c104f7f767ab6a76cd2a04461e4484e878e8e6969cfdfe387e83cdfd1df.jpg)  
(a) Fusion of reconstructed cortex and tractography (streamlines) derived from diffusion MRI data.

![](images/f42c07455fb009e77b0da06f9376caf8d267889ef2be408752187417c3e1df9d.jpg)  
(b) Pre-operative fusion with an MRI-derived surface (T1-weighted contrastenhanced).  
Figure 1. A geometrically sharp, deformable surgical surface. (a) H-2DGS reconstructs a localized photorealistic cortical layer tha remains clear when combined with subcortical fiber tracts. (b) Preoperative alignment between the deformed surgical surface and MRI.

We address this representation gap with H-2DGS, which progressively converts a robust volumetric reconstruction into a geometrically sharp yet photorealistic surface. Stage I first uses conventional 3DGS to establish appearance without prematurely imposing a fragile surface. Stage II then measures the spread of density along camera rays and contracts it toward the visible boundary. Multi-view depth from this concentrated model is fused into a truncated signed distance field (TSDF), providing a shared surface rather than unrelated per-view estimates. Stage III discards the volumetric Gaussians and places curved height-field primitives on the TSDF zero level set. The height field lets each patch follow local cortical curvature, separate color and opacity bases retain spatially varying texture, and neighbor-transition optimization coordinates overlapping patches.

The common 3D view must also remain meaningful when anatomy changes. Brain motion and tissue removal alter the relationship between the reconstructed surface and MRI [6]; after resection, the cavity can also invalidate vessel-based correspondence. For the postoperative case, we therefore render depth from the MRI-derived mesh and deform the explicit height-field surface toward that geometry. Depth determines the target shape, while tracked color supervision, overlap regularization, and alternating geometry–appearance updates prevent the alignment from erasing the photorealistic cortical texture. Preoperative vessel-guided alignment is retained only as a downstream use of the reconstructed surface.

Our contributions are<sup>1</sup>:

• a three-stage conversion from volumetric 3DGS to a thin TSDF-initialized cortical surface, including a differentiable ray-wise thickness objective;

• a curved height-field Gaussian primitive with spatially varying geometry, color, and opacity, together with overlap constraints for locally continuous geometry;

• a depth-supervised deformation procedure that aligns the photorealistic surface to postoperative MRI geometry while preserving appearance, enabling unified 3D visualization with whole-brain anatomy and tractography.

## 2 Related Work

Gaussian surface representations. Standard 3DGS represents a scene with anisotropic volumetric Gaussians optimized through differentiable alpha compositing [11]. Several alternatives strengthen its geometric prior. 2D Gaussian Splatting (2DGS) replaces ellipsoids by oriented disks and uses ray–splat intersection to obtain a more surface-like representation [8]. 3D Half-Gaussian Splatting (3D-HGS) splits primitives with learned planes to sharpen discontinuities [14], while Quadratic Gaussian Splatting (QGS) models local geometry with second-order surfaces [15]. These approaches initialize their designed primitives directly from an input point cloud. Under limited view coverage, the source geometry can itself be weakly constrained. H-2DGS instead first concentrates a robust volumetric reconstruction, fuses its multi-view depth into a shared TSDF, and only then initializes curved surface charts. Per-primitive appearance bases are related to textured Gaussian representations [19–21, 23], but our height basis additionally makes each primitive a deformable local surface.

Surgical reconstruction and anatomical fusion. Neural and Gaussian representations have been applied to static and deforming surgical scenes [17, 18, 22]. Most of these methods prioritize view synthesis or temporal tissue motion. Cross-modal methods connect intraoperative imagery to preoperative anatomy through inverse rendering or coupled meshes [4, 5]. Our focus is complementary: we construct a sharp, compact cortical boundary from limited microscope views, use it as a substrate for tractography, and directly deform this renderable surface toward postoperative geometry.

## 3 Sharp Height-Field Surface Reconstruction

## 3.1 Problem formulation and design rationale

The input is a set of calibrated microscope images and camera projections

$$
\mathcal { D } = \{ ( I _ { v } , \Pi _ { v } ) \} _ { v = 1 } ^ { V } .\tag{1}
$$

Our output is a renderable set of surface patches that must satisfy three requirements. First, it should reproduce cortical color and vessel landmarks from the input views. Second, opacity should be confined to a well-localized surface so that MRI anatomy and tractography remain visible on the correct side of the cortical boundary. Third, the representation should expose depth and local geometric parameters for subsequent alignment.

Directly fitting surface-oriented splats from the sparse input point cloud is brittle in this setting: the same limited camera coverage that creates 3DGS thickness also leaves the initial geometry weakly constrained. We therefore separate appearance acquisition from surface construction. Stage I obtains a robust photometric reconstruction with flexible volumetric Gaussians. Stage II makes this reconstruction a reliable geometric proxy by contracting its density along observed rays without changing the primitive type. Stage III fuses the resulting multi-view depths into one surface and replaces the volumetric model with curved height-field patches. Thus each stage has a distinct purpose: bootstrap appearance, localize geometry, and encode an explicit surface.

## 3.2 Stages I–II: from appearance to localized geometry

Photometric bootstrap. Stage I trains conventional 3DGS with adaptive densification. Starting from this flexible representation avoids forcing an inaccurate initial point cloud directly into a surface. The model is optimized with the standard photometric objective

$$
\mathcal { L } _ { \mathrm { p h o t o } } = ( 1 - \lambda ) \| \hat { I } _ { v } - I _ { v } \| _ { 1 } + \lambda \big ( 1 - \mathrm { S S I M } ( \hat { I } _ { v } , I _ { v } ) \big ) .\tag{2}
$$

to establish the visible tissue texture and overall scene support, where we set $\lambda = 0 . 2$ . This objective determines how the training views should look, but it does not determine where along a camera ray the explaining opacity should lie. The resulting model is therefore an appearance bootstrap rather than our final surface.

Ray-wise density concentration. The purpose of Stage II is to remove this depth ambiguity while retaining the successful image reconstruction. A sharp visible boundary should generate a narrow density distribution along each foreground ray; a thick 3DGS band generates a broad or multimodal distribution. We measure that spread with differentiable quantiles, which are less sensitive to isolated low-density tails than the extreme occupied samples. For camera ray $\mathbf { r } ( t ) = \mathbf { o } + t \mathbf { d }$ , substituting the ray into candidate 3D Gaussians gives one-dimensional contributions

![](images/507dcda89f07bafac5e71651c572249b1d1f451ab5d16a04fe83b32dde711756.jpg)  
Figure 2. Ray-wise density concentration. Stage II contracts the density interval between soft 1% and 99% quantiles.

$$
\sigma _ { i } ( t ) = \beta _ { i } \exp \bigl [ - { \textstyle \frac { 1 } { 2 } } a _ { i } ( t - m _ { i } ) ^ { 2 } \bigr ] , \qquad \sigma ( t ) = \sum _ { i } \sigma _ { i } ( t ) .
$$

At samples $\lbrace t _ { m } \rbrace _ { m = 1 } ^ { M }$ , let

(3)

$$
p _ { m } = \frac { \sigma ( t _ { m } ) } { \sum _ { n } \sigma ( t _ { n } ) + \epsilon } , \qquad F _ { m } = \sum _ { n = 1 } ^ { m } p _ { n } .\tag{4}
$$

We approximate quantile q by

$$
w _ { m } ( q ) = \frac { \exp ( - \tau _ { q } | F _ { m } - q | ) } { \sum _ { n } \exp ( - \tau _ { q } | F _ { n } - q | ) } , \qquad t _ { q } = \sum _ { m } w _ { m } ( q ) t _ { m } .\tag{5}
$$

The effective thickness is $\Delta t = t _ { q _ { \mathrm { h i g h } } } - t _ { q _ { \mathrm { l o w } } }$ , producing

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { t h i c k } } ( \mathbf { r } ) = \displaystyle \frac { \operatorname* { m a x } ( 0 , \Delta t - \Delta t ^ { \star } ) } { E } , } \\ & { \quad \mathcal { L } _ { \mathrm { s t a g e 2 } } = \mathcal { L } _ { \mathrm { p h o t o } } + \lambda _ { \mathrm { t h i c k } } \mathbb { E } _ { \mathbf { r } } [ \mathcal { L } _ { \mathrm { t h i c k } } ] . } \end{array}\tag{6}
$$

where E is the scene extent and $\lambda _ { \mathrm { t h i c k } }$ is set to 5.0. We use $q _ { \mathrm { l o w } } = 0 . 0 1 , q _ { \mathrm { h i g h } } = 0 . 9 9$ , and zero normalized target thickness. Stage II runs for 1,000 iterations after a 5,000- iteration warm start. The photometric term remains active, so contraction cannot simply collapse density at an arbitrary depth: it must preserve the observed cortical images while selecting a much thinner explanation.

## 3.3 Stage III: TSDF-guided surface conversion

Although Stage II provides concentrated depth, it is still a collection of volumetric Gaussians and may contain residual floating density. We therefore do not convert its individual ellipsoids directly into final primitives. Instead, we render depth from all training cameras and fuse those observations into a truncated signed distance field (TSDF). This intermediate field serves two purposes: it aggregates evidence across views into a single geometric reference, and its zero level set supplies both surface positions and normals for consistently orienting the new patches. For voxel center x, view v compares its projected depth $d _ { v } ( { \bf x } )$ with the Stage II depth $\hat { D } _ { v } ( \pi _ { v } ( \mathbf { x } ) )$ :

![](images/618e7c3f1d059c5bae6f497e94eb0c8fd95310721030ad1561243a9fbb9ea340.jpg)  
Figure 3. Multi-view TSDF fusion. Rendered depths define multiview signed-distance observations whose fused zero level set initializes Stage III.

$$
\begin{array} { l } { { \displaystyle \phi _ { v } ( { \bf x } ) = \mathrm { c l i p } \left( \frac { \hat { D } _ { v } ( \pi _ { v } ( { \bf x } ) ) - d _ { v } ( { \bf x } ) } { \mu } , - 1 , 1 \right) , } } \\ { { \displaystyle \Phi ( { \bf x } ) = \frac { \sum _ { v } w _ { v } ( { \bf x } ) \phi _ { v } ( { \bf x } ) } { \sum _ { v } w _ { v } ( { \bf x } ) + \epsilon } . } } \end{array}\tag{7}
$$

Only observations in the truncation band (we set $\mu$ to be three times of the voxel size) are fused. Grid cells with a sign change provide zero-level candidates, which are projected toward $\Phi ( \mathbf { x } ) = 0$ . The normalized gradient $\nabla \Phi / | | \nabla \Phi | |$ estimates surface normals. After view-consistency filtering and subsampling, a local quadric fit supplies each patch center, tangent frame, curvature estimate, and support size. The result is a sparse set of overlapping local charts that covers the shared TSDF surface rather than a noisy set of independently projected depth samples.

## 3.4 Height-field Gaussian primitive

The TSDF could be represented with planar surfels, but a fixed plane can only approximate cortical curvature by using many small elements. Our primitive keeps the thin support of a 2D surfel while allowing one patch to bend away from its tangent plane. It also stores spatially varying appearance on that same local domain, so a relatively small number of patches can represent both smooth geometry and detailed cortical texture. Primitive i has center $\mathbf { c } _ { i }$ , tangent frame $R _ { i } = [ \mathbf { t } _ { u , i } , \mathbf { t } _ { v , i } , \mathbf { n } _ { i } ]$ , and tangent scales $s _ { x , i } , s _ { y , i }$ . Its surface is

$$
\mathbf { S } _ { i } ( u , v ) = \mathbf { c } _ { i } + R _ { i } \left[ s _ { x , i } u \quad s _ { y , i } v \quad h _ { i } ( u , v ) \right] ^ { \top } ,\tag{8}
$$

![](images/8e45bd994170c9a34b03bfdcd5f7dca61136373f2f449f125696bf9cc9aec6c9.jpg)  
Figure 4. Concept of the proposed H-2DGS primitive. Each primitive is a curved local chart with independent basis fields over one tangent domain.

with a Gaussian-basis height field

$$
\begin{array} { r l } & { \displaystyle h _ { i } ( u , v ) = \sum _ { k = 1 } ^ { K _ { h } } b _ { i , k } ^ { h } \phi _ { i , k } ^ { h } ( u , v ) , } \\ & { \displaystyle \phi _ { i , k } ^ { h } ( u , v ) = \exp \left( - \frac { \| ( u , v ) - \pmb { \mu } _ { i , k } ^ { h } \| ^ { 2 } } { 2 ( \sigma _ { i , k } ^ { h } ) ^ { 2 } } \right) . } \end{array}\tag{9}
$$

Color $\mathbf { C } _ { i } ( u , v )$ and opacity $O _ { i } ( u , v )$ use analogous, independent bases plus constant terms. For each primitive, we use 5 height bases $( K _ { h } )$ , 20 opacity bases and 50 color bases. Geometry is intentionally lower-dimensional than appearance because surface shape varies more smoothly than cortical texture. The color bases act as a local texture map, whereas the opacity bases control support and boundary strength inside the patch. This separation lets appearance vary rapidly without introducing corresponding high-frequency oscillations in geometry.

To render, a world-space ray is transformed as $\mathbf { y } _ { i } ( t ) =$ $R _ { i } ^ { \top } ( { \bf r } ( t ) - { \bf c } _ { i } )$ Normalized coordinates are $u _ { i } ( t ) \ =$ $y _ { x , i } ( t ) / s _ { x , i }$ and $v _ { i } ( t ) = y _ { y , i } ( t ) / s _ { y , i }$ . The intersection is the root

$$
f _ { i } ( t ) = y _ { z , i } ( t ) - h _ { i } ( u _ { i } ( t ) , v _ { i } ( t ) ) = 0 ,\tag{10}
$$

found with a small fixed number of Newton iterations. Color and opacity are evaluated at the accepted intersection and composited in depth order. Unlike volumetric accumulation through an ellipsoid, this operation provides an explicit surface intersection and camera-linear depth, which are later reused by the postoperative alignment.

## 3.5 Initialization, continuity, and refinement

Surface positions alone are not enough to initialize the richer primitive. For each patch, nearby TSDF zero-level samples are expressed in its tangent frame, and the height coefficients are fitted to their signed offsets from the tangent plane. This converts the locally planar initialization into a curved chart before image refinement. Nearby Stage II Gaussians transfer color and opacity to the new patch; the volumetric Gaussians are then discarded completely.

Because each chart is fitted locally, two individually accurate patches can still disagree where they overlap, producing cracks or abrupt normal changes. We therefore sample corresponding locations in neighboring charts and coordinate them with

$$
{ \mathcal { L } } _ { \mathrm { t r a n s } } = \lambda _ { \mathrm { c o n t } } { \mathcal { L } } _ { \mathrm { c o n t } } + \lambda _ { \mathrm { p r e s } } { \mathcal { L } } _ { \mathrm { p r e s } } + \lambda _ { \mathrm { c u r v } } { \mathcal { L } } _ { \mathrm { c u r v } } .\tag{11}
$$

where $\lambda _ { \mathrm { c o n t } } = 0 . 5 , \lambda _ { \mathrm { p r e s } } = 2 . 0$ and $\lambda _ { \mathrm { c u r v } } = 0 . 0 2 . ~ \mathcal { L } _ { \mathrm { c o n t } }$ penalizes world-space position and tangent mismatch at paired overlap samples; $\mathcal { L } _ { \mathrm { p r e s } }$ retains the TSDF-fitted heights and gradients; and ${ \mathcal { L } } _ { \mathrm { c u r v } }$ suppresses oscillation. The three terms have complementary roles: continuity closes patch interfaces, preservation prevents this pairwise adjustment from drifting away from the multi-view TSDF, and curvature regularization avoids fitting the correspondences with ripples. After this geometric transition, final photometric refinement uses Eq. (2) to recover cortical appearance on the already localized surface. Shared basis layouts, precomputed inverse squared bandwidths, and CUDA-side indexing reduce rasterization overhead.

## 4 Postoperative Surface Alignment

## 4.1 Problem setup and optimization strategy

We use postoperative alignment as a representative scenario to test the geometric and rendering flexibilty of the proposed H-2DGS. In the postoperative scenario, the microscope reconstruction contains the post-resection photorealistic cortex, and the postoperative MRI supply information after the surgery. These two data modalities typically disagree geometrically, due to the resection cavity that significantly changes the surface morphology. We use the depth map as the proxy of the surface geometry to supervise the deformation. The inputs are an initial height-field model $\mathcal { G } _ { 0 }$ reconstructed from microscope video and a depth dataset

$$
\mathcal { D } _ { \mathrm { d e p t h } } = \{ ( D _ { v } , \Pi _ { v } ) \} _ { v = 1 } ^ { V }\tag{12}
$$

rendered from the postoperative MRI-derived mesh. For the same cameras, the current height-field model produces differentiable depths $\hat { D } _ { v }$ . The desired output is a deformed version of $\mathcal { G } _ { 0 }$ whose surface follows the MRI depth while retaining the initial cortical texture. This gives a direct imagespace comparison without establishing sparse cross-modal point correspondences.

Naive depth fitting, however, admits undesirable solutions: neighboring patches can move independently and tear, opacity can change which surface is observed, and texture can smear while geometry moves. Our optimization therefore combines three mechanisms with distinct roles. Depth loss supplies the target shape, the transition objective couples adjacent charts, and tracked self-supervision transports the original appearance. Alternating geometry and appearance windows keep the latter two goals from competing through the same updates.

## 4.2 Depth-supervised surface deformation

Centers, rotations, tangent scales, appearance coefficients, opacity, and height coefficients are trainable, allowing both global displacement and local reshaping. Basis centers and bandwidths remain fixed so that every primitive retains a stable local coordinate system during deformation. We reuse the explicit ray–surface intersection in Eq. (10) to obtain camera-linear depth. For this pass, opacity is overridden to 0.999; otherwise, the model could reduce the loss by making a mismatched foreground patch transparent rather than moving its geometry. With rendered and target validity masks $M _ { v }$ and $M _ { v } ^ { \star }$

$$
\mathcal { L } _ { \mathrm { d e p t h } } = \frac { \sum _ { \mathbf { p } } M _ { v } M _ { v } ^ { \star } | \hat { D } _ { v } ( \mathbf { p } ) - D _ { v } ( \mathbf { p } ) | } { \sum _ { \mathbf { p } } M _ { v } M _ { v } ^ { \star } + \epsilon } .\tag{13}
$$

Centers use an exponentially decayed learning rate from $1 . 6 { \times } 1 0 ^ { - 3 } \ \mathrm { t o } \ 1 . 6 { \times } \mathrm { \bar { 1 } } 0 ^ { - 5 }$ ; tangent scale uses $1 0 ^ { - 3 }$ , rotation and height coefficients use $1 0 ^ { - 4 }$ , and opacity uses $1 0 ^ { - 8 }$ This hierarchy makes center and scale the dominant deformation variables, uses rotation and height for smaller local corrections, and effectively stabilizes visibility. The transition objective in Eq. (11) remains active to prevent independently moving charts from tearing the surface.

## 4.3 Tracked appearance preservation

The MRI mesh constrains shape but contains no information about the color that should remain attached to each cortical location. We therefore use the initial reconstruction as its own appearance reference. Before deformation, valid pixels from reference renderings are back-projected and bound to local coordinates on candidate primitives. Each tracked sample stores its initial rendered color:

$$
\begin{array} { r } { \mathcal { Q } = \{ ( i _ { m } , u _ { m } , v _ { m } , h _ { m } , \mathbf { c } _ { m } ^ { 0 } ) \} . } \end{array}\tag{14}
$$

where $i _ { m }$ indexes the associated height-field primitive, $( u _ { m } , v _ { m } , h _ { m } )$ are its local surface coordinates, and $\mathbf { \bar { c } } _ { m } ^ { 0 } \in \mathbb { R } ^ { 3 }$ is its initial rendered RGB color.

As the primitives deform, samples move with their charts and are reprojected. We then encourage the reprojected color to be consistent with the stored initial color using a self-preservation term:

$$
\mathcal { L } _ { \mathrm { t r a c k } } = \mathbb { E } _ { m \in \mathcal { Q } } \left[ \left\| \mathbf { C } _ { \mathrm { c u r } } ( \pi _ { v } ( \mathbf { x } _ { m } ( \mathcal { G } ) ) ) - \mathbf { c } _ { m } ^ { 0 } \right\| _ { 1 } \right] .\tag{15}
$$

![](images/b43b4b7e42c3c353af0cd8169440231d29fa0513e3163253d9e48fe42fe759f8.jpg)  
2DGS

![](images/bcf2336b7694bf240b15cb0a591f9b1053fdd79d6a1acfd86d59facc4546e97b.jpg)  
3D-HGS

![](images/4d0d7da56709ccfb212b5e22240d9e319219f87e24245dc31fa5c9dd25a33f59.jpg)  
QGS

![](images/d3b885f9e6943785cdb6508b181467ee0f033c296f8623dde6d4f27e2e017a2a.jpg)  
H-2DGS

Figure 5. Fusion of reconstructed cortex and tractography derived from diffusion MRI. The top row shows the results of case preop 2, the second and the bottom rows show the results of case preop 3. columns are 2DGS, 3D-HGS, QGS, and H-2DGS.  
![](images/d8527690e9d5db854f62d9dc4924693b64fe95403c4945a0fed8a77afa0d47ee.jpg)

![](images/2907e5ad11991cb0e3249ca5196b79302820cb47aba2aa15db4e3c90f7b0537f.jpg)  
2DGS

![](images/23e661909dae060c000c287b7141cecc8d865cf3458fdd65a53108e06cd5356a.jpg)

![](images/b3ba9c9d6ad69641c8a4361c0249ebcb35e651dc802577eadb3c0037c71f2506.jpg)  
3D-HGS  
QGS  
H-2DGS  
Figure 6. Uncovered-view comparison over three cases and four methods. Each row corresponds to the results of one of the four case. From top to bottom rows: results of case preop 1, preop 2 and preop 3 respectively. H-2DGS remains localized, whereas baseline geometry streaks, fragments, or blurs.

Each sample is bound to eight candidate primitives, and targets are aggregated in image space. By enforcing the pixel-wise color consistency before and after deformation, the overall texture pattern of the surface is preserved.

![](images/65fedaeced14f4c34d703c11ea00842c9cad38dd5636bf37af6c64babe2f444e.jpg)

![](images/d7c700796a0114ee886041206aefe88789dcc8c129f2e1411699f94e5b1e33de.jpg)

![](images/ef9c513a119a206358e672f65c79ecb5ee371a114a9957bf9a0dfaa0075ee1b2.jpg)  
Figure 7. Result of deforming the post-resection reconstruction of the case postop 1. Initial (top) and deformed (bottom) reconstruction fitted to postoperative MRI morphology.

## 4.4 Alternating geometry and appearance updates

Applying depth and appearance updates simultaneously can yield an unstable compromise: the former must change geometry, while the latter resists changes to the rendered image. We instead separate their dominant effects. Optimization alternates a geometry-active window

$$
\mathcal { L } _ { \mathrm { d e p t h - w i n } } = \mathcal { L } _ { \mathrm { d e p t h } } + \lambda _ { \mathrm { t r a n s } } \mathcal { L } _ { \mathrm { t r a n s } } + \lambda _ { \mathrm { t r a c k } } \mathcal { L } _ { \mathrm { t r a c k } }\tag{16}
$$

![](images/4cca04ba3a57ba62d8b07a29738688a6cb42ca028d630039f4585fefa6907223.jpg)  
Figure 8. Depth-supervised postoperative alignment of the case postop 1. From left to right: initial, optimized, and target postoperative-MRI depths.

with an appearance window that freezes geometry and minimizes $\mathcal { L } _ { \mathrm { t r a c k } }$ using multi-view augmentation. We set $\lambda _ { \mathrm { t r a n s } }$ to $_ { 6 . 0 }$ and $\lambda _ { \mathrm { t r a c k } }$ to 10.0. The first window moves and reshapes the surface under MRI supervision while maintaining local continuity and a weak appearance anchor. The second cannot undo the acquired geometry; it only restores the texture fields around their deformed locations. Together, the two windows separate MRI-driven shape changes from restoration of the original photorealistic appearance.

## 5 Experiments

## 5.1 Setup and metrics

We evaluate five MRI cases using real neurosurgical videos: three preoperative cases and two postoperative cases. For each case, we select the continuous segment of the surgical video with the most comprehensive viewpoint coverage and uniformly sample frames from the original segment, resulting in a continuous sequence of 60–80 images for training. The data were shared under the approved data-sharing agreement between the participating institutions. All patients provided written informed consent for the use of their data for research purposes. All experiments run on an NVIDIA RTX 3050 GPU with 4 GB memory, 32 GB system memory, and an Intel i7-12700H CPU. H-2DGS uses 2,400 primitives per case, with 50 color and 20 opacity bases per primitive. We compare against 2DGS [8], 3D-HGS [14], and QGS [15], trained with their released default settings.

PSNR, SSIM, and LPIPS evaluate appearance. PSNR and SSIM use the intersection of valid rendered pixels across methods. Model size and per-frame time measure efficiency. For geometry, multi-view depth consistency (MVDC) backprojects valid depth from one evaluation view, reprojects it into a neighboring view, and reports the median absolute depth discrepancy normalized by scene extent.

![](images/f457113562928de48179fceee354971c200c4016c5fdc5a2638243a6a1f4ed7d.jpg)  
Figure 9. Components of postoperative deformation, case postop 1. Top: appearance-window iterations recover texture after depth fitting. Bottom: stronger geometry-continuity weighting coordinates neighboring patches.

## 5.2 Sharpness, consistency, and fusion

Table 1 exposes the intended trade-off. H-2DGS has lower training-view PSNR and SSIM than the baselines, mainly losing high-frequency highlights. Its geometric behavior is substantially more stable: MVDC lies between $1 \times 1 0 ^ { - 5 }$ and $4 \times 1 0 ^ { - 5 }$ . The representation is consistently 8 MB and renders in 30–42 ms. QGS attains the strongest photometric scores but uses 79–142 MB and 120–188 ms.

Differences that are muted in covered views become pronounced away from the training trajectory. Figure 6 shows that volumetric or weakly constrained representations can extend into streaks and fragments. H-2DGS remains a recognizable cortical layer. This matters for anatomical fusion: in Fig. 5, fibers appear attached to a defined surface instead of intersecting a diffuse cloud. The tractography therefore serves both as a target application and as a visual stress test of geometric localization [1].

## 5.3 Postoperative deformation

On the resection case, depth fitting moves the post-resection reconstruction toward postoperative MRI-rendered depth. The optimized map follows the target, while the surface stretches laterally and the cavity becomes shallower without losing major tissue or vessel patterns (Figs. 7 and 8).

Depth supervision alone destroys high-frequency appearance. Appearance-window updates restore cortical texture with geometry frozen, while stronger transition regularization improves local geometric and texture coherence (Fig. 9). The final overlay places the photorealistic cavity at the

Table 1. Quantitative comparison on five surgical cases. MVDC is normalized by scene extent. Lower is better for LPIPS, MVDC, size, and time. Missing LPIPS values were not reported by the corresponding baseline runs.
<table><tr><td>Case</td><td>Method</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>MVDC↓</td><td>Size (MB)↓</td><td>Time (ms)↓</td></tr><tr><td>Preop 1</td><td>H-2DGS 2DGS 3D-HGS QGS</td><td>23.84 28.54 28.00 31.46</td><td>0.776 0.901 0.887 0.952</td><td>0.218 0.109</td><td>4e-5 7e-5 1.18 9e-5</td><td>7.97 40.87 32.24 86.04</td><td>39.13 25.42 152.81 120.68</td></tr><tr><td>Preop 2</td><td>H-2DGS 2DGS 3D-HGS QGS</td><td>23.29 27.35 27.46 30.64</td><td>0.810 0.906 0.900 0.956</td><td>0.212 0.109</td><td>2e-5 6e-5 1.57 1e-4</td><td>7.97 45.72 36.39 107.03</td><td>29.68 29.69 178.32 164.09</td></tr><tr><td>Preop 3</td><td>H-2DGS 2DGS 3D-HGS QGS H-2DGS</td><td>25.20 33.33 31.69 40.07 21.54</td><td>0.839 0.956 0.941 0.986 0.745</td><td>0.161 0.048 0.210</td><td>1e-5 3e-5 1.41 5e-5 4e-5</td><td>7.97 51.69 36.88 93.64 7.97</td><td>31.98 28.21 6.16 152.35</td></tr><tr><td>Postop 1</td><td>2DGS 3D-HGS QGS</td><td>29.23 28.26 32.03</td><td>0.915 0.895 0.964</td><td>0.075</td><td>1e-4 1.0 1.4e-4</td><td>70.14 51.95 142.01</td><td>40.13 33.38 8.94 188.04</td></tr><tr><td>Postop 2</td><td>H-2DGS 2DGS 3D-HGS QGS</td><td>23.53 30.56 29.51 33.92</td><td>0.797 0.933 0.921 0.968</td><td>0.180 0.073</td><td>3.2e-5 9.58e-5 1.42 1.18e-4</td><td>7.97 46.26 40.78 78.93</td><td>42.47 24.78 7.52 135.80</td></tr></table>

corresponding postoperative MRI region across viewpoints (Fig. 1b).

## 6 Discussion and Conclusion

Together with the proposed three-stage pipeline, H-2DGS produces a photorealistic reconstruction of the cortical surface with a sharp geometry and well-defined boundary. This explicit surface enables direct integration with subcortical tractography and whole-brain MRI context without the internal anatomy being obscured by diffuse volumetric opacity. Compared with the baseline approaches, H-2DGS achieves substantially better geometric generalization across viewpoints: it consistently maintains a localized cortical surface in uncovered views and obtains lower multi-view depth inconsistency across the five cases (Tab. 1 and Fig. 6).

The clear geometry of H-2DGS also provides a suitable foundation for deformation. Its surface primitives retain sufficient geometric flexibility to represent substantial morphological changes, allowing deformation objectives to be defined directly on the primitive parameters and fitted to a target geometry. The geometric and rendering flexibility of H-2DGS is verified in the postoperative case, where the fitting process is separated into depth-fitting and appearancepreservation stages. The postoperative experiment demonstrates that this strategy can accommodate a pronounced geometric discrepancy while preserving the photorealistic appearance of the original reconstruction.

The compact size of H-2DGS follows from its fixed 2,400- primitive budget with per-primitive texture on an explicit surface. We do not match primitive counts to the baselines, since volumetric primitives lacking local texture generally need more of them to fit the same images. Overall, the smaller footprint reflects an advantage of the height-field primitive, especially given its stronger geometric localization.

A remaining limitation is rendering quality. As indicated by PSNR and SSIM, H-2DGS does not reproduce the observed images as accurately as the baseline methods. Replacing volumetric support with a thin explicit surface removes geometric ambiguity, but also reduces the rendering expressiveness available for reproducing high-frequency appearance variations. We introduce spatially varying texture through the color bases to compensate for this loss, but a clear photometric gap remains. Future work should therefore investigate richer geometry-preserving appearance models that improve rendering fidelity without reintroducing diffuse volumetric opacity.

The current pipeline also depends on camera poses recovered by structure from motion (SfM), whose static-scene assumption can be violated in surgical videos. One additional sequence failed because physiological cortical pulsation introduced non-rigid inter-frame motion, causing SfM to confound tissue motion with camera motion and produce inaccurate poses. This error subsequently degraded all reconstruction stages. Potential solutions include obtaining calibrated or externally tracked camera poses from specialized surgical equipment, representing each primitive’s geometric attributes as continuous functions of time, and estimating the dominant pulsation frequency for phase-aware frame selection or motion correction. Addressing both appearance expressiveness and dynamic tissue motion will be important for extending H-2DGS to more diverse surgical videos. In the future, combining this approach with cinematic rendering [2, 12] should offer a unique opportunity to improve surgical navigation.

## References

[1] Maxime Chamberland, Kevin Whittingstall, David Fortin, David Mathieu, and Maxime Descoteaux. Real-time multipeak tractography for instantaneous connectivity display. Frontiers in Neuroinformatics, 8:59, 2014.

[2] Dorin Comaniciu, Klaus Engel, Bogdan Georgescu, and Tommaso Mansi. Shaping the future through innovations: From medical imaging to precision medicine, 2016.

[3] Moustafa Elshafei, Johannes Binder, Justus Baecker, Maximilian Brunner, Michael Uder, Georg F. Weber, Robert Grutzmann, and Christian Krautz. Comparison of cinematic¨ rendering and computed tomography for speed and comprehension of surgical anatomy. JAMA Surgery, 154(8):738–744, 2019.

[4] Maximilian Fehrentz, Mohammad Farid Azampour, Reuben Dorent, Hassan Rasheed, Colin Galvin, Alexandra Golby, William M. Wells III, Sarah Frisken, Nassir Navab, and Nazim Haouchine. Intraoperative registration by cross-modal inverse neural rendering. In Medical Image Computing and Computer Assisted Intervention – MICCAI 2024, pages 317–327. Springer Nature Switzerland, 2024.

[5] Maximilian Fehrentz, Alexander Winkler, Thomas Heiliger, Nazim Haouchine, Christian Heiliger, and Nassir Navab. Bridgesplat: Bidirectionally coupled ct and non-rigid gaussian splatting for deformable intraoperative surgical navigation. In International Conference on Medical Image Computing and Computer-Assisted Intervention, pages 44–53. Springer, 2025.

[6] Ian J. Gerard, Marta Kersten-Oertel, Kevin Petrecca, Denis Sirhan, Jeffery A. Hall, and D. Louis Collins. Brain shift in neuronavigation of brain tumors: A review. Medical Image Analysis, 35:403–420, 2017.

[7] Jiaxin Guo, Jiangliu Wang, Di Kang, Wenzhen Dong, Wenting Wang, and Yun-hui Liu. Free-surgs: Sfm-free 3d gaussian splatting for surgical scene reconstruction. 2024.

[8] Binbin Huang, Zehao Yu, Anpei Chen, Andreas Geiger, and Shenghua Gao. 2d gaussian splatting for geometrically accurate radiance fields. In ACM SIGGRAPH 2024 Conference Proceedings, pages 1–11. ACM, 2024.

[9] Yiming Huang, Beilei Cui, Long Bai, Ziqi Guo, Mengya Xu, Mobarakol Islam, and Hongliang Ren. Endo-4dgs: Endoscopic monocular scene reconstruction with 4d gaussian splatting. In International Conference on Medical Image Computing and Computer-Assisted Intervention, pages 197–207. Springer, 2024.

[10] Yiming Huang, Long Bai, Beilei Cui, Yanheng Li, Tong Chen, Jie Wang, Jinlin Wu, Zhen Lei, Hongbin Liu, and Hongliang Ren. Endo-4dgx: Robust endoscopic scene reconstruction and illumination correction with gaussian splatting. In International Conference on Medical Image Computing and Computer-Assisted Intervention, pages 181–191. Springer, 2025.

[11] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuhler, and¨ George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Transactions on Graphics, 42(4):1–14, 2023.

[12] Bram Kraaijeveld, Andrei C Jalba, Anna Vilanova, and Maxime Chamberland. Real-time rendering of dynamic line sets using voxel ray tracing. In Computer Graphics Forum, page e70372. Wiley Online Library, 2026.

[13] Jose Pedro Lavrador, V´ ´ıctor Garc´ıa-Milan, Francesco Marchi,´ Kapil Rajwani, Nida Kalyal, Andrea Perera, Sankhya Prakashvel, Yasir A. Chowdhury, Ali Elhag, Laura Ferrari, Alba Diaz-Baamonde, Richard Gullan, Keyoumars Ashkan, Ranjeev Bhangoo, Ana Mirallave Pescador, and Francesco Vergani. Anatomo-functional approach to multimodal motor mapping in diffuse glioma surgery: Hierarchical networks. Brain Structure and Function, 230(6):103, 2025.

[14] Haolin Li, Jinyang Liu, Mario Sznaier, and Octavia Camps. 3d-hgs: 3d half-gaussian splatting. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10996–11005. IEEE/CVF, 2025.

[15] Hao Li, Shaofei Wang, Yinghao Xu, Qianyi Wu, Yujun Shen, and Xin Tao. Quadratic gaussian splatting. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. IEEE/CVF, 2025.

[16] Qian Li, Shuojue Yang, Daiyun Shen, Jimmy Bok Yan So, Jing Qin, and Yueming Jin. Free-dygs: Camera-pose-free scene reconstruction for dynamic surgical videos with gaussian splatting. arXiv preprint arXiv:2409.01003, 2025.

[17] Yifan Liu, Chenxin Li, Chen Yang, and Yixuan Yuan. Endogaussian: Real-time gaussian splatting for dynamic endoscopic scene reconstruction. arXiv preprint arXiv:2401.12561, 2024.

[18] Alberto Neri, Maximilian Fehrentz, Veronica Penza, Leonardo S. Mattos, and Nazim Haouchine. Surgical neural radiance fields from one image. International Journal ofComputer Assisted Radiology and Surgery, 20(11):2329– 2333, 2025.

[19] Panagiotis Papantonakis, Georgios Kopanas, Fredo Durand,´ and George Drettakis. Content-aware texturing for gaussian splatting. In Eurographics Symposium on Rendering. The Eurographics Association, 2025.

[20] Victor Rong, Jingxiang Chen, Sherwin Bahmani, Kiriakos N. Kutulakos, and David B. Lindell. Gstex: Per-primitive texturing of 2d gaussian splatting for decoupled appearance and geometry modeling. In IEEE/CVF Winter Conference on Applications ofComputer Vision, 2025.

[21] Sebastian Weiss and Derek Bradley. Gaussian billboards: Expressive 2d gaussian splatting with textures, 2024.

[22] Shuojue Yang, Qian Li, Daiyun Shen, Bingchen Gong, Qi Dou, and Yueming Jin. Deform3dgs: Flexible deformation for fast surgical scene reconstruction with gaussian splatting. In International Conference on Medical Image Computing and Computer-Assisted Intervention, pages 132–142. Springer, 2024.

[23] Mae Younes and Adnane Boukhayma. Texturesplat: Perprimitive texture mapping for reflective gaussian splatting. In 2026 International Conference on 3D Vision (3DV), pages 1–12, 2026.