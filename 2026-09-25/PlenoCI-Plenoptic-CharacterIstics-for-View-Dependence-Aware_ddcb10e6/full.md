# PlenoCI: Plenoptic CharacterIstics for View Dependence Aware Change Classification

Jason Lai<sup>1</sup> Chamuditha Jayanga Galappaththige<sup>2,3</sup>

Niko Sunderhauf¨ <sup>2,3</sup> Dimity Miller<sup>2,3</sup> Donald G. Dansereau<sup>1,3</sup>

<sup>1</sup>Australian Centre for Robotics, School of Aerospace, Mechanical and Mechatronic Engineering, The University of Sydney <sup>2</sup>QUT Centre for Robotics <sup>3</sup>ARIAM Hub

{jason.lai, donald.dansereau}@sydney.edu.au

## Abstract

Radiance field representations such as 3D Gaussian Splatting (3DGS) natively encode complex visual phenomena such as occlusions and view dependence, but they are inherently underconstrained. Independently optimized reconstructions converge to different primitive configurations, even in unchanged regions. We introduce Plenoptic CharacterIstics (PlenoCI), a novel feature built from the plenoptic field these representations approximate. PlenoCI directly captures rich visual behaviors while ignoring Lambertian textures. By deriving closed-form analytic plenoptic derivatives from a 3DGS representation, we efficiently detect these 5D structures. Our approach is robust to underconstrained representations by construction, reporting two orders of magnitude fewer false positives between independent reconstructions of unchanged scenes than concurrent work. We demonstrate PlenoCI’s utility on change classification. First, we detect changes with an instanceaware 3DGS pipeline, achieving state-of-the-art results on CL-Splats with a 25.7% mIoU gain over the strongest competitor, while remaining competitive on the more challenging PASLCD benchmark. Leveraging PlenoCI, we classify changes as geometric or appearance-based with a balanced accuracy of 0.735, comparable to the best performing baseline. We believe plenoptic derivatives and PlenoCI open new directions for view dependence aware understanding in visually complex environments. Code and data are available at https://js0n-lai.github.io/plenoci.

## 1. Introduction

Robust scene change detection (SCD) and change classification (CC) are vital for autonomous systems in unconstrained environments, with applications spanning damage assessment [62], environmental monitoring [67], and map updating [59]. For robotic platforms tasked with long-term monitoring, such as routine operator rounds in industrial facilities, reliably distinguishing genuine geometric and appearance changes from superficial lighting fluctuations, as our approach does in Figure 1, enables richer scene understanding and drives advanced autonomous reasoning.

Radiance field representations such as neural radiance fields (NeRF) [52] and 3D Gaussian Splatting (3DGS) [33] natively capture complex visual phenomena such as occlusion edges, specularity and reflections common in unconstrained settings. However, recent SCD works [20, 22, 23, 47, 78] building on these representations treat viewdependent effects purely as inconsistencies to suppress.

Our approach uses three insights: view dependence can signal complex visual behaviors; these phenomena manifest as unique structures in the 5D plenoptic field, captured by our novel Plenoptic CharacterIstics (PlenoCI) feature; and regressing the plenoptic field with 3DGS yields closedform, analytic derivatives that make these structures efficient to detect. Notably, PlenoCI extraction requires no perscene retuning across all scenes and datasets we evaluate on. By construction, PlenoCI also bypasses a fundamental challenge: radiance field representations are underconstrained. As Figure 1 shows, two independent optimizations given the same training images converge to different primitive configurations. Concurrent work [22] addresses this by explicitly modeling reconstruction drift to enable primitive comparisons. We take an orthogonal approach, deriving PlenoCI from the plenoptic field these representations approximate. We report a false positive rate two orders of magnitude lower when comparing independently optimized 3DGS models of unchanged scenes.

We demonstrate these tools on offline CC, where captures follow unconstrained trajectories through a scene. We detect changes with an instance-aware 3DGS pipeline that aggregates change cues across views, then use PlenoCI as a proxy for scene geometry to distinguish geometric from appearance-based changes.

![](images/2489b64a5fce571a31b7787ea4ab7f25ee43a56b9bb0722459a782b52e82ef89.jpg)  
Figure 1. Given two 3DGS reconstructions from different times, our method first detects scene changes. To classify changes, we draw on view-dependent plenoptic structures encoded in our novel Plenoptic CharacterIstics (PlenoCI) feature. For example, we correctly detect the sheet of paper and book as changes, and classify them using PlenoCI as occlusion edge evidence (left). Our approach is robust to the inherently underconstrained nature of 3DGS representations; two independent optimizations given the same training images yield different primitive configurations. Primitive-space comparisons [22] report false positives in static scenes while our method does not (right).

Our key contributions are:

• We derive closed-form analytic plenoptic derivatives from a 3DGS representation, and show they can be rendered efficiently using standard 3DGS rasterization.

• We propose PlenoCI as a novel 5D feature for summarizing informative plenoptic structures manifesting as complex visual behaviors, while ignoring Lambertian textures. These features build on generalized structure tensor analysis enabled by our plenoptic derivatives.

• We introduce a 3DGS-based SCD pipeline using off-theshelf instance segmentation to detect changed objects, and combine this with PlenoCI to distinguish between geometric and appearance-based changes.

We believe this work advances robust scene understanding. PlenoCI has possible applications in tasks such as localization, navigation and map stitching, while plenoptic derivatives can be used for visual odometry and uncertainty estimation. Code will be released upon acceptance.

## 2. Related Work

## 2.1. 2D Scene Change Detection

Early SCD works compare 2D images, relying on fully or weakly supervised training on closely aligned image pairs [3, 8, 9, 39, 40, 44, 60, 61, 63, 69–71]. However, this paradigm is significantly hindered by costly change annotations and real-world domain shifts [25]. Vision foundation models [36, 56] have introduced a new paradigm for zeroshot SCD [4, 10, 32, 35], circumventing the drawbacks of supervised models and outperforming them on real-world datasets. These approaches however inherit the failure modes of their underlying backbones while remaining brittle against viewpoint discrepancies and view-dependent inconsistencies [23]. In contrast, our approach aggregates change cues across multiple views, is agnostic to viewpoint variations, and can additionally classify changes.

## 2.2. 2D-3D Scene Change Detection

Rich 3D representations can be constructed from multiview images for SCD. Geometric foundation models [41, 72, 73] directly infer 3D scene properties using attention mechanisms. However, SCD works [19, 46, 75] operating on these representations inherently struggle in scenes with strongly view-dependent behavior. NeRF [52] and 3DGS [33] have been widely established as high-fidelity, photorealistic representations capable of capturing view dependence. Consequently, recent methods in SCD use NeRF [28, 50, 77] and 3DGS [20, 23, 27, 30, 37, 45, 47, 78] to model the prechange scene and synthesize views from the post-change viewpoint, enabling SCD in a pose-agnostic setting. These methods use learned semantic features [27, 37, 45, 47, 78] from vision backbones, or a combination with pixel data [20, 23] to generate zero-shot change cues.

Recent works [20, 21, 23, 47, 78] move beyond simple pairwise comparisons, aggregating change information across views to achieve state-of-the-art results. However, these approaches treat view-dependent cues purely as inconsistencies to suppress, discarding possibly genuine changes. Concurrent work [22] detects changes in primitive space with explicit drift modeling to overcome underconstrained optimizations, but discards higher order spherical harmonics (SH) encoding view dependence. We take an orthogonal approach, explicitly uncovering view-dependent structures by operating in the plenoptic field.

## 2.3. Leveraging the Plenoptic Field

The plenoptic field [2] describes light rays as a highdimensional phenomenon, mapping position, direction, wavelength and time to intensity. Light fields [42, 66] represent 4D slices with fixed wavelength and time, while assuming rays are constant-valued along their propagation. They are applicable in tasks such as depth estimation [11], structure from motion (SfM) [31, 68], scene flow [49] and visual odometry [12, 55]. Dansereau et al. [13] propose closedform change detection as regions that violate plenoptic flow, a linear system relating light field derivatives to camera motion. However, these methods derive analytic derivatives assuming a Lambertian scene, or densely sample for numerical estimation. By leveraging 3DGS as a plenoptic regression, we derive closed-form analytic expressions with no such assumptions, and generalize beyond light fields.

Radiance field representations including NeRF [52] and 3DGS [33] regress the 5D plenoptic function [16–18, 20, 58]. Recent works use the plenoptic field as a unifying space, such as benchmarking representations [16], and geometry regularization with surface light field constraints [54]. Building on this perspective, we efficiently detect 5D plenoptic structures from a 3DGS representation.

## 3. PlenoCI: A New Plenoptic Feature

We introduce PlenoCI, a feature capturing rays that experience color change as they shift along multiple axes of position and direction, signaling complex behaviors such as occlusion edges and specularity. We build intuition in a 2D world (Sec. 3.1) before generalizing to 3D scenes (Sec. 3.2).

## 3.1. Toy Example: Flatland

We begin exploring the plenoptic field $\mathcal { L } ( \mathbf { r } ) = \mathcal { L } ( x , y , \theta )$ in flatland, a world with two spatial dimensions (x, y) and in-plane rotation θ. Figure 2a shows an example scene. The problem is stated as follows: a set of 1D images $\mathcal { T }$ captured at known camera poses $\mathcal { P }$ is used to learn a radiance field representation $\mathcal { G }$ of the scene. G contains 2D Gaussian primitives parameterized by center ${ \pmb { \mu } } _ { i } \in \mathbb { R } ^ { 2 }$ , covariance $\Sigma _ { i } ,$ base opacity $\alpha _ { i } \in [ 0 , 1 ]$ and view-dependent color $\mathbf { c } _ { i }$ denoted by Fourier transform coefficients operating on a query direction $\mathrm { C H } _ { i } ( \theta ) . \ \Sigma _ { i } = R _ { i } S _ { i } S _ { i } ^ { T } R _ { i } ^ { T }$ is decomposed into a rotation matrix $R _ { i }$ and scaling matrix $S _ { i }$

## 3.1.1. Extracting 3D Plenoptic Derivatives

The plenoptic derivatives measure sensitivity of a ray’s intensity to shifts in its origin or direction. PlenoCI builds on $\nabla \mathcal { L } = \left[ \mathcal { L } _ { x } , \mathcal { L } _ { y } , \mathcal { L } _ { \theta } \right]$ , where $\mathcal { L } _ { * } = \partial \mathcal { L } /$ ∂∗ is the partial derivative for scalar ∗, or the Jacobian for vector ∗. Figures 2b to 2c show examples of $\mathcal { L } _ { * }$ taken at a fixed θ slice. We show that $\nabla \mathcal { L }$ is closed-form and analytic when $\mathcal { L }$ is regressed using 3DGS.

![](images/14a6cf6a2dcef04486bf90c273b4f72ed1cc3f749a49694a829d62b250e31e44.jpg)

![](images/382a216dfdd357907f95201b991add04157359674447edc3f098af92996b90d0.jpg)  
(b) |L<sub>y</sub>| at θ = 180<sup>◦</sup>

![](images/3dcb8240a19c8adf5f2e4033d77876ca9664aa181cf28d54000e198327a785fc.jpg)  
(c) |L<sub>θ</sub>| at θ = 180<sup>◦</sup>

![](images/71291f453cd4034d52c96c0803ca97a4ca4b49c7e0481a54247e660d61c2da1a.jpg)  
(d) Occlusion edges  
Figure 2. Example scenario in flatland to motivate our novel PlenoCI feature. Given a 2D world (Fig. 2a), we build the 3D plenoptic field $\mathcal { L }$ and compute partial derivatives $\mathcal { L } _ { * }$ (Figs. 2b to 2c). These are useful signals for detecting visually complex phenomena such as occlusion edges (Fig. 2d). Brightness represents magnitude corresponding to each color channel’s plenoptic field. While we show ground truth geometry and $\mathcal { L }$ here, this also generalizes to radiance field representations.

Radiance field representations can be queried to determine the color of a ray using volumetric rendering [52],

$$
\mathcal { L } ( \mathbf { r } ) = \sum _ { i } c _ { i } \overline { { \alpha } } _ { i } \prod _ { j } ^ { i - 1 } ( 1 - \overline { { \alpha } } _ { j } ) ,\tag{1}
$$

where α is the effective opacity.

Prior 3DGS works show that closed-form analytic derivatives with respect to their parameters [33] and camera pose [51] can be computed. Similarly, we can write derivatives with respect to r. Differentiating (1), then using $\overline { { \alpha } } _ { i } = \alpha _ { i } \exp ( \beta _ { i } )$ and transmittance $\begin{array} { r } { T _ { i } = \bar { \prod _ { j } ^ { i - 1 } } ( 1 - \overline { { \alpha } } _ { j } ) } \end{array}$

$$
L _ { * } = \sum _ { i } \overline { { \alpha } } _ { i } T _ { i } \bigg ( ( c _ { i } ) _ { * } + c _ { i } \bigg ( ( \beta _ { i } ) _ { * } - \sum _ { j } ^ { i - 1 } \frac { \overline { { \alpha } } _ { j } ( \beta _ { j } ) _ { * } } { 1 - \overline { { \alpha } } _ { j } } \bigg ) \bigg ) .\tag{2}
$$

Intuitively, (2) contains a view-dependent color derivative term $\left( c _ { i } \right)$ ∗ describing the change in CH<sub>i</sub>(θ) with r, and a geometry term describing the change in fall-off $\beta _ { i }$ . The former is trivial to compute, noting that $( c _ { i } ) _ { x } = ( c _ { i } ) _ { y } = 0$

For the latter, we follow [53, 74] by ray-tracing, i.e., for each ray, we evaluate intersecting Gaussians at the point of maximum response $\mathbf { p } = \mathbf { x } + \tau _ { \mathrm { m a x } } \mathbf { d }$ , where x is the ray’s origin and $\mathbf { d } = \left[ \cos \theta , \right.$ sin $\theta \rfloor ^ { T }$ is the direction. Dropping the Gaussian subscripts for brevity, it can be shown that the Gaussian fall-off based on Mahalanobis distance $d _ { \perp }$

$$
\beta = - \frac 1 2 d _ { \perp } ^ { 2 } = - \frac 1 2 ( { \bf p } - { \pmb \mu } ) ^ { T } \Sigma ^ { - 1 } ( { \bf p } - { \pmb \mu } ) ,\tag{3}
$$

is minimized at the point of closest approach:

$$
\tau _ { \mathrm { m a x } } = \frac { ( \pmb { \mu } - \mathbf { x } ) ^ { T } \Sigma ^ { - 1 } \mathbf { d } } { \mathbf { d } ^ { T } \Sigma ^ { - 1 } \mathbf { d } } = \frac { - \mathbf { x } _ { g } ^ { T } \mathbf { d } _ { g } } { \mathbf { d } _ { g } ^ { T } \mathbf { d } _ { g } } .\tag{4}
$$

${ \bf x } _ { g } = S ^ { - 1 } R ^ { T } ( { \bf x } - { \bf \nabla } { \pmb { \mu } } )$ and $\mathbf { d } _ { q } = { \cal S } ^ { - 1 } { \cal R } ^ { T }$ d transforms r into the Gaussian’s canonical frame. We can apply the chain rule to derive analytic, closed-form expressions for $\beta _ { * }$

$$
\beta _ { \mathbf { x } } = ( \beta ) _ { d _ { \perp } } ( d _ { \perp } ) _ { \mathbf { x } _ { g } } ( \mathbf { x } _ { g } ) _ { \mathbf { x } } ,\tag{5}
$$

$$
\beta _ { \boldsymbol { \theta } } = ( \beta ) _ { d _ { \perp } } ( d _ { \perp } ) _ { \mathbf { d } _ { g , \perp } } ( \mathbf { d } _ { g , \perp } ) _ { \mathbf { d } _ { g } } ( \mathbf { d } _ { g } ) _ { \mathbf { d } } ( \mathbf { d } ) _ { \boldsymbol { \theta } } .\tag{6}
$$

$d _ { \perp }$ is the orthogonal projection of $\mathbf { r } _ { g }$ onto $\pmb { \mu }$ in the canonical frame and $\mathbf { d } _ { g , \perp }$ is the unit vector orthogonal to ${ \bf d } _ { g }$

## 3.1.2. The Plenoptic Structure Tensor

Edges and corners provide information about scene geometry and textures robust to illumination changes [38, 65], making them desirable for image matching [6, 48] and landmark detection [24, 29]. Harris [26] and Shi-Tomasi [65] detect corners as points in an image I with large intensity changes in two orthogonal directions. We treat the plenoptic field in the same way, asking how a ray’s color changes in $( x , y , \theta )$ to uncover occlusion edges and specularity.

The structure tensor smooths gradients over a window w,

$$
M = \sum _ { ( x , y ) } w ( x , y ) \left[ \begin{array} { c c } { { I _ { x } ^ { 2 } } } & { { I _ { x } I _ { y } } } \\ { { I _ { y } I _ { x } } } & { { I _ { y } ^ { 2 } } } \end{array} \right] ,\tag{7}
$$

with corners detected where M has two large eigenvalues. We reformulate (7) with the angular dimension,

$$
M = \sum _ { ( x , y , \theta ) } w _ { s } ( x , y ) w _ { d } ( \theta ) \big ( \nabla \mathcal { L } ^ { * } \big ) \big ( \nabla \mathcal { L } ^ { * } \big ) ^ { T } .\tag{8}
$$

To control sensitivity to scene details, we smooth the analytic $\nabla \mathcal { L }$ using Gaussian windows $\omega _ { s }$ and $\omega _ { d }$ to construct $\nabla \mathcal { L } ^ { \ast }$ , with 2π periodicity for $\omega _ { d }$ . We use similar Gaussian windows $w _ { s }$ and $w _ { d }$ to aggregate outer products. M has 3 eigenvalues $\lambda _ { 0 } \geq \lambda _ { 1 } \geq \lambda _ { 2 } \geq 0$ , giving us 4 cases:

• No large λ: a flat region.

• 1 large λ: a Lambertian texture or geometric edge where $\mathcal { L }$ is sensitive to shifts along one $( x , y , \theta )$ direction.

• 2 large λ: visually complex behaviors such as occlusion edges (Fig. 2d) and specularity where $\mathcal { L }$ is sensitive to shifts along two orthogonal $( x , y , \theta )$ directions.

• 3 large $\lambda \colon \mathcal { L }$ is sensitive to any shift. This is highly unlikely in free space where rays do not change color along the direction of propagation.

For RGB, we write $\nabla { \mathcal { L } } ^ { * } = \left\lceil \nabla { \mathcal { L } } _ { R } ^ { * } \quad \nabla { \mathcal { L } } _ { G } ^ { * } \quad \nabla { \mathcal { L } } _ { B } ^ { * } \right\rceil$ to combine gradients across channels [14] and apply (8).

## 3.1.3. PlenoCI in Flatland

We propose PlenoCI as rays R where M has two large eigenvalues, i.e., where $\lambda _ { 1 } ( \mathbf { r } ) \ge \tau _ { 1 }$ . Figure 2d shows regions with PlenoCI $i . e . ,$ , rays observing an occlusion edge.

## 3.2. Generalizing to 5D

Real scenes are 3D, calling for a 5D plenoptic field: $( x , y , z )$ for position and $( \phi , \theta )$ for direction. Figure 3 shows our pipeline to extract 5D PlenoCI from a 3DGS representation. We explain key deviations from Section 3.1 here and provide full details in Section 7.

In (2), we highlighted two terms: the view-dependent color derivative $( c _ { i } ) _ { \mathbf { r } }$ <sub>r</sub> and geometry term $( \beta _ { i } ) _ { \mathbf { r } }$ Following [33], we model $c _ { i }$ with SH, queried using the unit vector vˆ from the camera to Gaussian’s center, $i . e . , c _ { i } = \mathrm { S H } _ { i } ( \hat { \mathbf { v } } )$ This leads to the surprising result that ray direction does not affect $c _ { i } . \left( c _ { i } \right) _ { \mathbf { x } }$ can be computed with the chain rule. For the geometry term, the Mahalanobis distance can be written as:

$$
d _ { \perp } ^ { 2 } = \lVert \hat { \mathbf { d } } _ { g } \times \mathbf { x } _ { g } \rVert ^ { 2 } ,\tag{9}
$$

where $\hat { \mathbf { d } } _ { g }$ is the unit transformed direction. Applying the chain rule in (5)–(6) yields closed-form derivatives.

Following (8), M extends to a $5 \times 5$ matrix. To avoid sampling 5D volumes, we sample a local light field of analytic $\nabla \mathcal { L } \colon$ a single pinhole camera provides $( \phi , \theta )$ samples while a $3 \times 3$ camera array with fixed spacing also provides $( x , y , z )$ samples. We use 2D Gaussian kernels for spatial and angular windows, and additionally scale spatial $\mathcal { L } _ { * }$ by the median rendered depth before constructing $\nabla \mathcal { L } ^ { \ast }$

The third eigenvalue $\lambda _ { 2 }$ is sensitive to occlusion edges and specularity as Figure 3 shows. Hence, it forms the basis of PlenoCI: we threshold $\lambda _ { 2 } ~ \geq ~ \tau _ { 2 }$ , and mask this with a Canny edge map [7] to form R. We also find $\lambda _ { 0 }$ is sensitive to edges, $\lambda _ { 1 }$ to corners as Section 8 shows.

## 4. Change Detection and Classification

Figure 4 shows our overall approach. Following prior work [20], we use SfM [64] to register $\mathcal { T } _ { 0 } , \mathcal { T } _ { 1 }$ and find camera poses $\mathcal { P } _ { 0 } , \mathcal { P } _ { 1 }$ in a common coordinate frame. We then construct instance-aware 3DGS [43] representations $\mathcal { G } _ { 0 } , \mathcal { G } _ { 1 }$ Our pipeline aggregates ray differences at the instance level to detect and classify changes. Given a set of rays, we accumulate change evidence in the corresponding primitives and threshold to yield sparse anchors $\Delta \mathcal { G } _ { t } ^ { \prime } .$ . We propagate this information to related primitives using baked instance cues, yielding change representations $\Delta \mathcal { G } _ { t }$ useful for rendering dense change masks. For change detection, we use rays $\mathcal { R } ^ { e }$ observing image edges. For change classification, we leverage PlenoCI R to build geometric change representations $\Delta \mathcal { G } _ { t } ^ { \mathrm { g e o } }$ from which geometric change masks are rendered. Appearance change representations $\overline { { \Delta \mathcal { G } _ { t } ^ { \mathrm { a p p } } } } = \Delta \mathcal { G } _ { t } ^ { \mathrm { a l l } } \backslash \Delta \mathcal { G } _ { t } ^ { \mathrm { g e o } }$ can be used to render appearance change masks.

![](images/73ad8d83875b7572da869dc0b16e80bfea82104ade6c38dec0e5d57f7ad7baea.jpg)  
Figure 3. Our PlenoCI feature extraction pipeline. Given a 3DGS representation ${ \mathcal { G } } ,$ we sample analytic plenoptic derivatives $\nabla \mathcal { L }$ over a local light field to compute the structure tensor. The third eigenvalue $\lambda _ { 2 }$ provides an informative signal for complex visual behaviors where existing features are unreliable, while ignoring Lambertian textures. We mask pixelwise $\lambda _ { 2 }$ with an edge image (dilated for clarity) to yield PlenoCI $\mathcal { R } ^ { k }$ from view k. Repeating this over selected views yields R.

![](images/cb44db998999b02a7a5bd9b51d6bc621fe12d7d1d7c6a9e393857b01464e7736.jpg)  
Figure 4. Our overall scene change detection and classification approach. Given instance-aware 3DGS scene representations $\mathcal { G } _ { 0 }$ and $\mathcal { G } _ { 1 } .$ we aggregate per-ray change scores δ in the corresponding primitives. This yields a sparse set of anchors $\Delta \mathcal { G } _ { t } ^ { \prime }$ to propagate on changed instances. For change detection, we sample edge rays $\mathcal { R } ^ { e }$ to find changed Gaussians $\Delta \mathcal { G } _ { t } ^ { \mathrm { a l l } }$ . For change classification, we leverage PlenoCI R to find geometrically changed Gaussians $\Delta \dot { \mathcal { G } } _ { t } ^ { \mathrm { g e c } }$ . These explicit representations can be used to render novel change masks with disambiguation between geometric and appearance changes.

## 4.1. Finding Changes

We seek representations $\Delta \mathcal { G } _ { t } ^ { \mathrm { a l l } }$ denoting changed primitives. For pose k, we select rays $\mathcal { R } _ { t } ^ { e , k }$ on Canny edge [7] segments in the RGB render $I _ { t } ^ { k }$ . To obtain reliable RGB and DINOv2 features [56], we displace these boundary pixels towards the foreground using rendered depth gradients. We follow [20,

23] to form per-ray change,

$$
\delta = w _ { \mathrm { r g b } } \delta _ { \mathrm { r g b } } + w _ { \mathrm { f e a t } } \delta _ { \mathrm { f e a t } } ,\tag{10}
$$

for all edge rays $\mathcal { R } _ { 0 } ^ { e , k } \cup \mathcal { R } _ { 1 } ^ { e , k } . \mathrm { ~ } \delta _ { \mathrm { r g b } }$ is the L1 error and $\delta _ { \mathrm { f e a t } }$ is the cosine dissimilarity between feature embeddings. We repeat this across all poses k with fixed $w _ { \mathrm { r g b } } = w _ { \mathrm { f e a t } } = 0 . 5$

To map rays to primitives, we consider the most dominant primitive along the ray, $i . e . ,$ the primitive $i _ { t } ^ { * }$ at timestep t satisfying $i _ { t } ^ { * } = \arg \operatorname* { m a x } _ { i } \overline { { \alpha } } _ { i } T _ { i }$ . Mean-pooling δ in the frontmost of primitives $i _ { 0 } ^ { * }$ and $i _ { 1 } ^ { * }$ addresses two ambiguities: 1) primitives on the side of an edge of the changed object receive high δ, and 2) whether the change is attributable to the frontmost primitive or one it occludes. The latter resolves disocclusion cases (1) alone cannot distinguish. This yields sparse anchors $\Delta \mathcal { G } _ { t } ^ { \prime }$ with change information.

To segment dense representations $\Delta \mathcal { G } _ { t } ,$ , we leverage learned per-Gaussian instance IDs. Let $\mathcal { G } _ { t } ^ { m }$ denote the set of primitives from time t with instance ID m, and $\Delta { \mathcal { G } _ { t } ^ { ' m } }$ the set of anchor primitives on this instance. $\mathcal { G } _ { t } ^ { m }$ is marked as changed if the mean δ over $\Delta { \mathcal { G } _ { t } ^ { ' m } }$ exceeds $\tau _ { \mathrm { i n s t } }$ . We also impose an occupancy gate, where changed instances require an anchor fraction $\psi = | \Delta \mathcal { G } _ { t } ^ { ' m } | / | \Delta \mathcal { G } ^ { m } | \geq \tau _ { \mathrm { a n c } }$

Following [20, 23], we render change views $M _ { k , t } ^ { \mathrm { a l l } }$ representing the change associated with time t from novel view $k .$ Combining the two timesteps $M _ { k } ^ { \mathrm { a l l } } ~ =$ max $( M _ { k , 0 } ^ { \mathrm { a l l } } , M _ { k , 1 } ^ { \mathrm { a l l } } ) \ge \tau _ { \mathrm { m a s k } } = 0 . 5$ yields the final mask.

## 4.2. Classifying Changes with PlenoCI

We use PlenoCI as a proxy for geometry; occlusion edges shift when scene structure changes. However, PlenoCI also captures appearance properties such as specularity; we leave further disambiguation as future work. Given PlenoCI from both timesteps $\mathcal { R } _ { 0 } \cup \mathcal { R } _ { 1 }$ , we compute change scores using (10), and aggregate in the corresponding primitives,

$$
\delta ( G _ { i } ) = \frac { \sum _ { \mathbf { r } } w _ { \mathrm { g e o } } ( \mathbf { r } ) \delta ( \mathbf { r } ) } { \sum _ { \mathbf { r } } w _ { \mathrm { g e o } } ( \mathbf { r } ) } .\tag{11}
$$

$w _ { \mathrm { g e o } }$ represents confidence of an occlusion edge change,

$$
w _ { \mathrm { g e o } } = | e _ { 0 } - e _ { 1 } | , \quad e _ { t } = \sigma \Big ( \frac { \lambda _ { 2 } - \tau _ { 2 } } { T } \Big ) ,\tag{12}
$$

where $T$ is a temperature hyperparameter, and $\sigma ( \cdot )$ is the sigmoid activation function.

We similarly mark instances as a geometric change if the mean δ over its anchors exceeds $\tau _ { \mathrm { i n s t } } ^ { \mathrm { g } \overline { { \mathrm { e } } } 0 }$ , and anchor fraction $\psi \geq \tau _ { \mathrm { a n c } }$ . We also impose a minimum mass $\begin{array} { r } { \sum _ { \mathbf { r } } w _ { \mathbf { g e o } } ( \mathbf { r } ) \ge } \end{array}$ $\tau _ { \mathrm { g e o } } ~ = ~ 1$ to ensure sufficient geometric change support, and exclude instances not in $\Delta \mathcal { G } _ { t } ^ { \mathrm { a l l } }$ This yields a set of geometric change primitives $\Delta \mathcal { G } _ { t } ^ { \mathrm { g e o } }$ All other changed primitives are marked as appearance-based, i.e., $\Delta \mathcal { G } _ { t } ^ { \mathrm { a p p } }$ = $\overline { { \Delta \mathcal { G } _ { t } ^ { \mathrm { a l l } } } } \setminus \Delta \mathcal { G } _ { t } ^ { \mathrm { g e o } }$ . These representations can be used to render geometric and appearance change masks, respectively.

## 5. Experiments

We build instance-aware 3DGS representations with InstanceGaussian [43] while maintaining explicit ray-traced primitives [74] and densification [34]. We use gsplat [76] with CUDA-accelerated kernels for real-time plenoptic derivatives. All experiments were conducted on a single RTX PRO 6000 Blackwell Max-Q Workstation Edition

GPU. On average, training each scene takes 45 minutes while our SCD and CC pipeline takes 5 minutes.

We evaluate our method on CL-Splats [1] and PASLCD [20], datasets featuring real-world scenes captured in-the-wild. CL-Splats contains five scenes with simple geometric changes. PASLCD has ten scenes with concurrent geometric and appearance changes, alongside two lighting variants each. Ground truth labels for CL-Splats were annotated using the SceneDiff annotation tool [75] and human-verified, while we use provided labels for PASLCD.

In our evaluations, we fix PlenoCI extraction hyperparameters including smoothing windows, eigenvalue threshold $\tau _ { 2 }$ and temperature $T .$ . All pre-change images are used to train $\mathcal { G } _ { 0 }$ . For $\mathcal { G } _ { 1 }$ , we hold out every $5 ^ { \mathrm { t h } }$ image from training and evaluate SCD on these unseen frames. To build PlenoCI, we sample 20 post-change training views—this corresponds to all training views for PASLCD, and a subset for CL-Splats selected using farthest point sampling [15]. We use a grid search to select per-scene instance thresholds τ , $\tau _ { \mathrm { i n s t } } ^ { \mathrm { g e o } }$ , and minimum anchor fraction $\tau _ { \mathrm { a n c } } .$

## 5.1. Binary Change Detection Results

We compare our approach to state-of-the-art methods: CYWS [60], GeSCF [35], SceneDiff [75], 3DGS-CD [47], MV3DCD [20], OSCD [23] and GS-Diff [22]. For fair comparison, we benchmark OSCD [23] in the offline setting and evaluate pairwise methods [35, 61] by rendering aligned views using vanilla 3DGS [33]. To assess change masks, we follow prior SCD work [20, 44, 47, 57] and report mean intersection over union (mIoU) and F1 score for the changed class.

Table 1 summarizes binary SCD results. Our approach surpasses all baselines on CL-Splats, making a 25.7% mIoU improvement over the strongest competitor, while we closely match the strongest performance on PASLCD. This stems from leveraging instance-aware 3D representations. As Figure 5 shows, our approach produces complete segmentations for singular changed objects in CL-Splats, while methods lacking this information [20, 23, 75] provide incomplete masks. PASLCD contains more cluttered scenes where instance segmentation quality can limit our performance; we perform oracle experiments in Section 5.2.

## 5.2. Ablations

In Table 2, we ablate our change scoring using identical reconstructions and rays $\mathcal { R } ^ { e }$ . As [20, 23] find, RGB and semantics play complementary roles. Occupancy gating inductively biases our SCD pipeline towards localized changes, preventing a small number of false positive anchors from propagating through large instances. Large changed instances with sufficient evidence are preserved.

Since instance segmentation is upstream and orthogonal to PlenoCI, our SCD pipeline directly benefits from improvements to the segmenter. PASLCD contains small changes which may be localized by the anchors but undersegmented in the 3DGS representation, with qualitative examples in Section 8.1. We validate this by substituting learned instance IDs [43] with ground truth oracles. Concretely, we run connected components analysis with 8- connectivity on ground truth masks. We lift blob pixels to 3D using the most dominant primitive (Sec. 4.1), and group primitives by spatial proximity. The lifted groups are merged by intersection over union to form the oracles. Since unchanged objects do not receive an oracle, change recall is the main metric. As Table 3 shows, making this substitution with fixed anchors increases recall by 20% while leaving precision stable, confirming our hypothesis.

![](images/9237be9cef3f3a35ac56315aa8d6d82deb2147504068ca27508b11daceb39c7f.jpg)  
Figure 5. Qualitative comparison of binary change masks for CL-Splats [1] (rows 1–2) and PASLCD [20] (rows 3–4). Pairwise approaches such as GeSCF [35] are susceptible to spurious detections. Our instance-aware 3DGS approach outperforms all baselines on CL-Splats, and is competitive to state-of-the-art on the challenging PASLCD benchmark.

Table 1. Binary SCD results averaged over CL-Splats [1] and PASLCD [20]. PASLCD baselines are sourced from [23, 75]. Our method achieves state-of-the-art performance on CL-Splats, and is competitive with the strongest PASLCD baselines. The first , second , and third best performances are highlighted.
<table><tr><td rowspan="2">Method</td><td colspan="2">CL-Splats [1]</td><td colspan="2">PASLCD [20]</td></tr><tr><td>mIoU↑</td><td>F1↑</td><td>mIoU ↑</td><td>F1 ↑</td></tr><tr><td>CYWS [60]</td><td>0.495</td><td>0.640</td><td>0.273</td><td>0.398</td></tr><tr><td>GeSCF [35]</td><td>0.692</td><td>0.793</td><td>0.477</td><td>0.611</td></tr><tr><td>SceneDiff [75]</td><td>0.334</td><td>0.433</td><td>0.473</td><td></td></tr><tr><td>3DGS-CD [47]</td><td>0.700</td><td>0.792</td><td>0.209</td><td>0.339</td></tr><tr><td>MV3DCD [20]</td><td>0.634</td><td>0.752</td><td>0.478</td><td>0.628</td></tr><tr><td>OSCD [23]</td><td>0.756</td><td>0.845</td><td>0.552</td><td>0.694</td></tr><tr><td>GS-Diff [22]</td><td>0.724</td><td>0.829</td><td>0.630</td><td>0.745</td></tr><tr><td>Ours</td><td>0.951</td><td>0.975</td><td>0.538</td><td>0.688</td></tr></table>

Table 2. Change scoring ablation averaged over PASLCD [20]. Incorporating all elements provides the best performance.
<table><tr><td>Method</td><td>mIoU↑ F1 ↑</td></tr><tr><td>No RGB</td><td>0.530 0.671</td></tr><tr><td>No Semantics</td><td>0.469 0.614</td></tr><tr><td>No Occupancy 0.514</td><td>0.658</td></tr><tr><td>Ours</td><td>0.538 0.688</td></tr></table>

Table 3. SCD results averaged over PASLCD [20] varying instance segmentations. Oracle derives instances from ground truth masks while we use learned instances [43]. Recall is the main metric, improving by 20% with oracle instances as the learned instances undersegment small changed objects.
<table><tr><td>Method</td><td>mIoU↑</td><td>F1↑</td><td>Precision ↑</td><td>Recall ↑</td></tr><tr><td>Ours</td><td>0.538</td><td>0.688</td><td>0.712</td><td>0.691</td></tr><tr><td>Oracle</td><td>0.596</td><td>0.731</td><td>0.667</td><td>0.839</td></tr></table>

We also compare querying only edge rays with all rays from the sample views in Section 8.2. Querying all rays yields comparable SCD performance, dropping mIoU by 0.021 while considerably increasing compute costs.

## 5.3. Multi-class Change Classification Results

To physically ground changes, we classify them as geometric (Geo) or appearance-based (App). We evaluate multiclass performance on PASLCD [20] using human-annotated Geo or App classifications from [22]. We report balanced accuracy [5] and per-class precision and recall.

![](images/5c6457a7e612aaefe9ea40d9410d64d6d01bb0170726f8fffcf0d126db185b1f.jpg)  
Figure 6. Qualitative change classification results for PASLCD [20]. Top to bottom: Porch, Lounge, Meeting Room, Zen. Our PlenoCI feature enables building explicit geometry and appearance-based change representations for rendering multi-class change masks.

Table 4. CC performance averaged over PASLCD [20]. The best performance excluding Oracle is bolded. Oracle uses instances derived from ground truth masks (Sec. 5.2), driving improvements in recall of both classes. Ground truth class imbalance favoring Geo amplifies App classification errors.
<table><tr><td>Metric</td><td>GS-Diff [22]</td><td>Ours</td><td>Oracle</td></tr><tr><td>Balanced Accuracy ↑</td><td>0.792</td><td>0.735</td><td>0.851</td></tr><tr><td>Geo Precision ↑</td><td>0.970</td><td>0.985</td><td>0.970</td></tr><tr><td>Geo Recall ↑</td><td>0.896</td><td>0.745</td><td>0.871</td></tr><tr><td>App Precision ↑</td><td>0.478</td><td>0.431</td><td>0.463</td></tr><tr><td>App Recall ↑</td><td>0.687</td><td>0.725</td><td>0.830</td></tr></table>

Table 4 and Figure 6 present our CC results, using PlenoCI as a proxy for geometry. Our approach is competitive with GS-Diff. App precision is low across methods due to class imbalance; 84% of change pixels are Geo, amplifying App errors [22]. PlenoCI conflates occlusion edges with specularity and reflections. Conversely, object replacements (e.g., the mug and pen in Porch) can create overlapping PlenoCI, misclassifying these Geo changes as App. Upstream segmentation also affects CC; using oracle instances (Sec. 5.2) improves recall for both classes.

## 5.4. Robustness to Reconstruction Ambiguity

We take an orthogonal approach to concurrent work. GS-Diff [22] directly compares primitives, mitigating reconstruction ambiguity with explicit drift modeling while assuming geometrically accurate representations. We derive

Table 5. False positive rate from comparing two independently optimized 3DGS models of an unchanged scene, averaged over PASLCD [20]. Scenes with real changes contain 3.51% change pixels on average, ranging from 0.17–20.12%.
<table><tr><td>Method</td><td>Binary (%) ↓</td><td>Geo (%) ↓</td><td>App (%) ↓</td></tr><tr><td>GS-Diff [22]</td><td>0.389</td><td>0.064</td><td>0.325</td></tr><tr><td>Ours</td><td>0.004</td><td>0.001</td><td>0.003</td></tr></table>

PlenoCI from the plenoptic field, remaining robust to underconstrained primitives by construction. To evaluate this, we independently optimize two 3DGS models of a static scene using all pre-change images, reporting all detections as false positives. For our method, we use the same $( \tau _ { \mathrm { i n s t } } , \tau _ { \mathrm { a n c } } , \tau _ { \mathrm { i n s t } } ^ { \mathrm { g e o } } )$ from Sections 5.1 and 5.3. As Table 5 shows, our false positive rates are consistently lower. Our approach is also robust to reconstruction ambiguity induced by varying training view overlaps and coverage levels (Sec. 8.3).

## 6. Conclusion

We introduced PlenoCI, a novel feature that explicitly captures complex visual behaviors discarded by existing methods, while being robust to underconstrained representations. We showed that using 3DGS to regress the plenoptic field yields closed-form analytic derivatives. In this work, we demonstrated the utility of PlenoCI for distinguishing between geometric and appearance-based changes. Future work could also conduct instance segmentation in plenoptic space, propagating change evidence along related rays. Beyond SCD, we believe this work lays a foundation for robust autonomous perception in visually complex environments. PlenoCI could be applicable in tasks such as localization, navigation and map stitching, while the underlying plenoptic derivatives could be informative for visual odometry and uncertainty estimation.

## Acknowledgements

This research was supported in part through the NVIDIA Academic Grant Program and by the ARC Research Hub in Intelligent Robotic Systems for Real-Time Asset Management (IH210100030). J.L. acknowledges ongoing support by the Australian Government Research Training Program (RTP) Scholarship. C.J., N.S., and D.M. acknowledge ongoing support from the QUT Centre for Robotics.

## References

[1] Jan Ackermann, Jonas Kulhanek, Shengqu Cai, Xu Haofei, Marc Pollefeys, Gordon Wetzstein, Leonidas Guibas, and Songyou Peng. CL-Splats: Continual learning of Gaussian splatting with local optimization. In Int. Conf. Comput. Vis., 2025. 6, 7

[2] Edward H. Adelson and James R. Bergen. The plenoptic function and the elements of early vision. In Computational Models of Visual Processing. The MIT Press, 1991. 3

[3] Pablo F Alcantarilla, Simon Stent, German Ros, Roberto Arroyo, and Riccardo Gherardi. Street-view change detection with deconvolutional networks. Auton. Robots, 42(7):1301– 1322, 2018. 2

[4] Tim Alpherts, Sennay Ghebreab, and Nanne van Noord. EMPLACE: Self-supervised urban scene change detection. AAAI, 39(2):1737–1745, 2025. 2

[5] Kay Henning Brodersen, Cheng Soon Ong, Klaas Enno Stephan, and Joachim M. Buhmann. The balanced accuracy and its posterior distribution. In Int. Conf. Pattern Recog., pages 3121–3124. IEEE, 2010. 7

[6] M. Brown, R. Szeliski, and S. Winder. Multi-image matching using multi-scale oriented patches. In IEEE Conf. Comput. Vis. Pattern Recog., pages 510–517 vol. 1, 2005. 4

[7] John Canny. A computational approach to edge detection. IEEE Trans. Pattern Anal. Mach. Intell., PAMI-8(6):679– 698, 1986. 4, 5

[8] Rodrigo Caye Daudt, Bertr Le Saux, and Alexandre Boulch. Fully convolutional siamese networks for change detection. In IEEE Int. Conf. Image Process., pages 4063–4067, Athens, 2018. IEEE. 2

[9] Shuo Chen, Kailun Yang, and Rainer Stiefelhagen. Drtanet: Dynamic receptive temporal attention network for street scene change detection. In IEEE Intell. Veh. Symp., pages 502–509. IEEE, 2021. 2

[10] Kyusik Cho, Dong Yeop Kim, and Euntai Kim. Zero-shot scene change detection. AAAI, 39(3):2509–2517, 2025. 2

[11] Donald G. Dansereau and L. Bruton. Gradient-based depth estimation from 4D light fields. In IEEE Int. Symp. Circuit Sys., pages III–549, 2004. 3

[12] Donald G. Dansereau, Ian Mahon, Oscar Pizarro, and Stefan B. Williams. Plenoptic flow: Closed-form visual odom etry for light field cameras. In IEEE/RSJ Int. Conf. Intell. Robot. Syst., pages 4455–4462. IEEE, 2011. 3

[13] Donald G. Dansereau, Stefan B. Williams, and Peter I. Corke. Simple change detection from mobile light field cam eras. Comput. Vis. Image Underst., 145:160–171, 2016. 3

[14] Silvano Di Zenzo. A note on the gradient of a multi-image. Comput. Vis. Graph. Image Process., 33(1):116–125, 1986. 4

[15] Y. Eldar, M. Lindenbaum, M. Porat, and Y.Y. Zeevi. The farthest point strategy for progressive image sampling. IEEE Trans. Image Process., 6(9):1305–1315, 1997. 6

[16] Davi R. Freitas, Ricardo L. De Queiroz, Ioan Tabus, and Christine Guillemot. A comparative assessment of implicit and explicit plenoptic scene representations. In IEEE Int. Worksh. Multi. Sig. Proc., pages 1–6, West Lafayette, IN, USA, 2024. IEEE. 3

[17] Davi R. Freitas, Ioan Tabus, and Christine Guillemot. Visibility-based geometry pruning of neural plenoptic scene representations. IEEE Trans. Multimedia, 28:1–16, 2026.

[18] Sara Fridovich-Keil, Alex Yu, Matthew Tancik, Qinhong Chen, Benjamin Recht, and Angjoo Kanazawa. Plenoxels: Radiance fields without neural networks. In IEEE Conf. Comput. Vis. Pattern Recog., pages 5491–5500, 2022. 3

[19] Almog Friedlander, Ariel Shamir, and Ohad Fried. GOLDILOCS: General object-level detection and labeling of changes in scenes. In Int. Conf. Learn. Represent., 2026. 2

[20] Chamuditha Jayanga Galappaththige, Jason Lai, Lloyd Win drim, Donald Dansereau, Niko Sunderhauf, and Dimity¨ Miller. Multi-view pose-agnostic change localization with zero labels. In IEEE Conf. Comput. Vis. Pattern Recog., pages 11600–11610, 2025. 1, 2, 3, 4, 5, 6, 7, 8

[21] Chamuditha Jayanga Galappaththige, Thomas Gottwald, Peter Stehr, Edgar Heinert, Niko Suenderhauf, Dimity Miller, and Matthias Rottmann. Predictive photometric uncertainty in gaussian splatting for novel view synthesis. In Eur. Conf. Comput. Vis., 2026. 2

[22] Chamuditha Jayanga Galappaththige, Jason Lai, Timothy Patten, Donald G. Dansereau, Niko Suenderhauf, and Dim ity Miller. From pixels to primitives: Scene change detection in 3D Gaussian splatting. arXiv preprint arXiv:2605.07203, 2026. 1, 2, 6, 7, 8, 3, 4

[23] Chamuditha Jayanga Galappaththige, Jason Lai, Lloyd Windrim, Donald G. Dansereau, Niko Suenderhauf, and Dimity Miller. Changes in real time: Online scene change detection with multi-view fusion. In IEEE Conf. Comput. Vis. Pattern Recog., pages 32246–32256, 2026. 1, 2, 5, 6, 7

[24] Arturo Gil, Oscar Martinez Mozos, Monica Ballesta, and Oscar Reinoso. A comparative evaluation of interest point detectors and local descriptors for visual SLAM. Mach. Vis. Appl., 21(6):905–920, 2010. 4

[25] Ishaan Gulrajani and David Lopez-Paz. In search of lost domain generalization. In Int. Conf. Learn. Represent., 2021. 2

[26] Christopher G. Harris and M. J. Stephens. A combined cor ner and edge detector. In Alvey Vision Conference, 1988. 4

[27] Satoko Hattori-Nagao, Kazuo Oda, Tomoaki Eguchi, Takanobu Nagao, and Satomi Kakuta. Render-to-real imagebased change detection of outdoor infrastructure using 3D Gaussian splatting. Int. Arch. Photogramm. Remote Sens. Spatial Inf. Sci., XLIX-B2-2026:771–777, 2026. 2

[28] Rui Huang, Binbin Jiang, Qingyi Zhao, William Wang, Yuxiang Zhang, and Qing Guo. C-NeRF: Representing scene changes as directional consistency difference-based NeRF. arXiv preprint arXiv:2312.02751, 2023. 2

[29] Woo Yeon Jeong and Kyoung Mu Lee. Visual SLAM with line and corner features. In IEEE/RSJ Int. Conf. Intell. Robot. Syst., pages 2570–2575, 2006. 4

[30] Binbin Jiang, Rui Huang, Qingyi Zhao, and Yuxiang Zhang. Gaussian difference: Find any change instance in 3D scenes. In ICASSP, pages 1–5, Kothaguda, India, 2025. IEEE. 2

[31] Ole Johannsen, Antonin Sulc, and Bastian Goldluecke. On linear structure from motion for light field cameras. In Int. Conf. Comput. Vis., pages 720–728, 2015. 3

[32] Shyam Sundar Kannan and Byung-Cheol Min. ZeroSCD: Zero-shot street scene change detection. In IEEE Int. Conf. Rob. Autom., pages 4665–4671, 2025. 2

[33] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuehler, and George Drettakis. 3D Gaussian splatting for real-time radiance field rendering. ACM Trans. Graph., 42(4):1–14, 2023. 1, 2, 3, 4, 6

[34] Shakiba Kheradmand, Daniel Rebain, Gopal Sharma, Weiwei Sun, Yang-Che Tseng, Hossam Isack, Abhishek Kar, Andrea Tagliasacchi, and Kwang Moo Yi. 3D Gaussian splatting as markov chain monte carlo. In Adv. Neural Inform. Process. Syst., 2024. Spotlight Presentation. 6

[35] Jae-Woo Kim and Ue-Hwan Kim. Towards generalizable scene change detection. In IEEE Conf. Comput. Vis. Pattern Recog., pages 24463–24473, 2025. 2, 6, 7

[36] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Dollar, and Ross Girshick. Segment anything. In Int. Conf. Comput. Vis., pages 4015–4026, 2023. 2

[37] Mathis Kruse, Marco Rudolph, Dominik Woiwode, and Bodo Rosenhahn. SplatPose & detect: Pose-agnostic 3D anomaly detection. In IEEE Conf. Comput. Vis. Pattern Recog. Worksh., pages 3950–3960, 2024. 2

[38] Ivan Laptev. On space-time interest points. Int. J. Comput. Vis., 64(2):107–123, 2005. 4

[39] Seonhoon Lee and Jong-Hwan Kim. Semi-supervised scene change detection by distillation from feature-metric alignment. In IEEE Winter Conf. App. Comput. Vis., pages 1226– 1235, 2024. 2

[40] Yinjie Lei, Duo Peng, Pingping Zhang, Qiuhong Ke, and Haifeng Li. Hierarchical paired channel fusion network for street scene change detection. IEEE Trans. Image Process., 30:55–67, 2020. 2

[41] Vincent Leroy, Yohann Cabon, and Jerome Revaud. Grounding image matching in 3D with MASt3R. In Eur. Conf. Comput. Vis., page 71–91, Berlin, Heidelberg, 2024. Springer-Verlag. 2

[42] Marc Levoy and Pat Hanrahan. Light field rendering. In ACM SIGGRAPH, pages 31–42, New Orleans, USA, 1996. Association for Computing Machinery. 3

[43] Haijie Li, Yanmin Wu, Jiarui Meng, Qiankun Gao, Zhiyao Zhang, Ronggang Wang, and Jian Zhang. Instancegaussian: Appearance-semantic joint Gaussian representation for 3D instance-level perception. In IEEE Conf. Comput. Vis. Pat tern Recog., 2025. 4, 6, 7, 1

[44] Chun-Jung Lin, Sourav Garg, Tat-Jun Chin, and Feras Dayoub. Robust scene change detection using visual foundation models and cross-attention mechanisms. In IEEE Int. Conf. Rob. Autom., pages 8337–8343, Atlanta, USA, 2025. IEEE. 2, 6

[45] Yizhe Liu, Yan Song Hu, Yuhao Chen, and John Zelek. SplatPose+: Real-time image-based pose-agnostic 3D anomaly detection. In Eur. Conf. Comput. Vis. Worksh., pages 378–391, 2024. 2

[46] Ziling Liu, Ziwei Chen, Mingqi Gao, Jinyu Yang, and Feng Zheng. Leveraging geometric priors for unaligned scene change detection. arXiv preprint arXiv:2509.11292, 2025. 2

[47] Ziqi Lu, Jianbo Ye, and John Leonard. 3DGS-CD: 3D Gaussian splatting-based change detection for physical object rearrangement. IEEE Robot. Autom. Letters, 10(3):2662–2669, 2025. 1, 2, 6, 7

[48] Jiayi Ma, Xingyu Jiang, Aoxiang Fan, Junjun Jiang, and Junchi Yan. Image matching from handcrafted to deep fea tures: A survey. Int. J. Comput. Vis., 129(1):23–79, 2021. 4

[49] Sizhuo Ma, Brandon M. Smith, and Mohit Gupta. 3D scene flow from 4D light field gradients. In Eur. Conf. Comput. Vis., 2018. 3

[50] E. Martinson and P. Lauren. Meaningful Change Detection in indoor environments using CLIP models and NeRF-based image synthesis. In Int. Conf. Ubiq. Rob., pages 603–610, 2024. 2

[51] Hidenobu Matsuki, Riku Murai, Paul HJ Kelly, and Andrew J Davison. Gaussian splatting SLAM. In IEEE Conf. Comput. Vis. Pattern Recog., pages 18039–18048, 2024. 3

[52] Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, and Ren Ng. NeRF: Representing scenes as neural radiance fields for view syn thesis. In Eur. Conf. Comput. Vis., pages 405–421, Cham, Switzerland, 2020. Springer International Publishing. 1, 2, 3

[53] Nicolas Moenne-Loccoz, Ashkan Mirzaei, Or Perel, Riccardo de Lutio, Janick Martinez Esturo, Gavriel State, Sanja Fidler, Nicholas Sharp, and Zan Gojcic. 3D Gaussian ray tracing: Fast tracing of particle scenes. ACM Trans. Graph., 43(6):1–19, 2024. 3, 1

[54] Jack Naylor, Viorela Ila, and Donald G. Dansereau. Surf-NeRF: Surface regularised neural radiance fields. IEEE Conf. Comput. Vis. Pattern Recog. Worksh., pages 258–268, 2026. 3

[55] J. Neumann, C. Fermuller, and Y. Aloimonos. A hierarchy of cameras for 3D photography. In Int. Symp. 3D Data Process., Vis. Transm., pages 2–11, 2002. 3

[56] Maxime Oquab, Timothee Darcet, Th ´ eo Moutakanni, Huy´ Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez,

Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. DINOv2: Learning robust visual features without supervision. Trans. Mach. Learn. Res., 2024. 2, 5

[57] Jin-Man Park, Jae-hyuk Jang, Sahng-Min Yoo, Sun-Kyung Lee, Ue-hwan Kim, and Jong-Hwan Kim. ChangeSim: Towards end-to-end online scene change detection in industrial indoor environments. In IEEE/RSJ Int. Conf. Intell. Robot. Syst. IEEE, 2021. 6

[58] I. Petrovska and B. Jutzi. 3D Gaussian splatting methods for real-world scenarios. ISPRS Ann. Photogramm. Remote Sens. Spatial Inf. Sci., X-G-2025:641–648, 2025. 3

[59] Richard J Radke, Srinivas Andra, Omar Al-Kofahi, and Badrinath Roysam. Image change detection algorithms: A systematic survey. IEEE Trans. Image Process., 14(3):294– 307, 2005. 1

[60] Ragav Sachdeva and Andrew Zisserman. The change you want to see. In IEEE Winter Conf. App. Comput. Vis., pages 3993–4002, 2023. 2, 6, 7

[61] Ragav Sachdeva and Andrew Zisserman. The change you want to see (now in 3D). In Int. Conf. Comput. Vis. Worksh., pages 2052–2061, Paris, France, 2023. IEEE. 2, 6

[62] Ken Sakurada and Takayuki Okatani. Change detection from a street image pair using CNN features and superpixel segmentation. In Brit. Mach. Vis. Conf., pages 61.1–61.12, Swansea, 2015. British Machine Vision Association. 1

[63] Ken Sakurada, Mikiya Shibuya, and Weimin Wang. Weakly supervised silhouette-based semantic scene change detection. In IEEE Int. Conf. Rob. Autom., pages 6861–6867, Paris, France, 2020. IEEE. 2

[64] Johannes L. Schonberger and Jan-Michael Frahm. Structure-¨ from-motion revisited. In IEEE Conf. Comput. Vis. Pattern Recog., pages 4104–4113, 2016. 4

[65] Jianbo Shi and Tomasi. Good features to track. In IEEE Conf. Comput. Vis. Pattern Recog., pages 593–600, 1994. 4

[66] Richard Szeliski and Michael F Cohen. The lumigraph. In ACM SIGGRAPH, pages 43–54, New Orleans, USA, 1996. Association for Computing Machinery. 3

[67] Aparna Taneja, Luca Ballan, and Marc Pollefeys. Image based detection of geometric changes in urban environments. In 2011 International Conference on Computer Vision, pages 2336–2343. IEEE, 2011. 1

[68] Dorian Tsai, Donald G. Dansereau, Thierry Peynot, and Peter Corke. Distinguishing refracted features using light field cameras with application to structure from motion. IEEE Robot. Autom. Letters, 4(2):177–184, 2019. 3

[69] Ashley Varghese, Jayavardhana Gubbi, Akshaya Ramaswamy, and P Balamuralidhar. ChangeNet: A deep learning architecture for visual change detection. In Eur. Conf. Comput. Vis. Worksh., pages 0–0, 2018. 2

[70] Ashley Varghese, Jayavardhana Gubbi, Akshaya Ramaswamy, and P. Balamuralidhar. ChangeNet: A deep learning architecture for visual change detection. In Eur. Conf. Comput. Vis. Worksh., pages 129–145. Springer International Publishing, Munich, Germany, 2018.

[71] Guo-Hua Wang, Bin-Bin Gao, and Chengjie Wang. How to reduce change detection to semantic segmentation. Pattern Recognition, 138:109384, 2023. 2

[72] Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, and David Novotny. Vggt: Visual geometry grounded transformer. In IEEE Conf. Comput. Vis. Pattern Recog., 2025. 2

[73] Yifan Wang, Jianjun Zhou, Haoyi Zhu, Wenzheng Chang, Yang Zhou, Zizun Li, Junyi Chen, Jiangmiao Pang, Chunhua Shen, and Tong He. \$\piˆ3\$: Permutation-equivariant visual geometry learning. In Int. Conf. Learn. Represent., 2026. 2

[74] Qi Wu, Janick Martinez Esturo, Ashkan Mirzaei, Nicolas Moenne-Loccoz, and Zan Gojcic. 3DGUT: Enabling distorted cameras and secondary rays in Gaussian splatting. In IEEE Conf. Comput. Vis. Pattern Recog., pages 26036– 26046, 2025. 3, 6

[75] Yuqun Wu, Chih-hao Lin, Henry Che, Aditi Tiwari, Chuhang Zou, Shenlong Wang, and Derek Hoiem. SceneDiff: A benchmark and method for multiview object change detection. In Eur. Conf. Comput. Vis., 2026. 2, 6, 7

[76] Vickie Ye, Ruilong Li, Justin Kerr, Matias Turkulainen, Brent Yi, Zhuoyang Pan, Otto Seiskari, Jianbo Ye, Jeffrey Hu, Matthew Tancik, et al. gsplat: An open-source library for Gaussian splatting. J. Mach. Learn. Res., 26(34):1–17, 2025. 6

[77] Qiang Zhou, Weize Li, Lihan Jiang, Guoliang Wang, Guyue Zhou, Shanghang Zhang, and Hao Zhao. PAD: A dataset and benchmark for pose-agnostic anomaly detection. In Adv. Neural Inform. Process. Syst., pages 44558–44571, New Or leans, USA, 2023. 2

[78] Zirui Zhou, Junfeng Ni, Shujie Zhang, Yixin Chen, and Siyuan Huang. 3D scene change modeling with consistent multi-view aggregation. In Int. Conf. 3D Vis., 2025. 1, 2

# PlenoCI: Plenoptic CharacterIstics for View Dependence Aware Change Classification

Supplementary Material

## 7. Plenoptic Gradients Derivation

## 7.1. Volumetric Rendering

The color of a ray r can be computed using volumetric rendering as per (1),

$$
\mathcal { L } ( \mathbf { r } ) = \sum _ { i } c _ { i } \overline { { \alpha } } _ { i } \prod _ { j } ^ { i - 1 } ( 1 - \overline { { \alpha } } _ { j } ) .\tag{S1}
$$

Our goal is to differentiate this with respect to r. Let $C _ { i } =$ $\begin{array} { r } { c _ { i } \overline { { \alpha } } _ { i } , T _ { i } = \prod _ { j } ^ { i - 1 } ( 1 - \overline { { \alpha } } _ { j } ) } \end{array}$ and $\mathcal { L } _ { i } = C _ { i } T _ { i }$ . Differentiating with the product rule yields:

$$
( \mathcal { L } _ { i } ) _ { * } = ( C _ { i } ) _ { * } T _ { i } + C _ { i } ( T _ { i } ) _ { * } .\tag{S2}
$$

Taking the natural logarithm of $T _ { i } ,$ , differentiating and rearranging gives:

$$
( T _ { i } ) _ { * } = - T _ { i } \sum _ { j } ^ { i - 1 } \frac { ( \overline { { \alpha } } _ { j } ) _ { * } } { 1 - \overline { { \alpha } } _ { j } } .\tag{S3}
$$

Substituting (S3) and $( C _ { i } ) _ { * } = ( c _ { i } ) _ { * } { \overline { { \alpha } } } _ { i } + c _ { i } ( { \overline { { \alpha } } } _ { i } )$ <sub>∗</sub> into (S1) and (S2) yields (2).

## 7.2. Ray-Gaussian Intersection

We now consider derivatives of the geometric fall-off term $( \beta _ { i } )$ <sub>∗</sub>. Dropping the Gaussian subscripts for brevity, we can write from (3):

$$
\beta _ { * } = - \frac { 1 } { 2 } ( d _ { \perp } ^ { 2 } ) _ { * } ,\tag{S4}
$$

where $d _ { \perp }$ is the Mahalanobis distance given by (9):

$$
d _ { \perp } ^ { 2 } = \lVert \hat { \mathbf { d } } _ { g } \times \mathbf { x } _ { g } \rVert ^ { 2 } .\tag{S5}
$$

$\mathbf { x } _ { g }$ and $\hat { \mathbf { d } } _ { g }$ represent the ray transformed into canonical Gaussian space. Using $\| \hat { \mathbf { d } } _ { g } \| = 1$ , we can write:

$$
( d _ { \perp } ^ { 2 } ) _ { { \bf x } _ { g } } = 2 \big ( \mathbf { x } _ { g } - ( \mathbf { x } _ { g } \cdot \hat { \mathbf { d } } _ { g } ) \hat { \mathbf { d } } _ { g } \big ) ,\tag{S6}
$$

$$
( d _ { \perp } ^ { 2 } ) _ { \hat { \mathbf { d } } _ { g } } = 2 \big ( \| \mathbf { x } _ { g } \| ^ { 2 } \hat { \mathbf { d } } _ { g } - ( \mathbf { x } _ { g } \cdot \hat { \mathbf { d } } _ { g } ) \mathbf { x } _ { g } \big ) .\tag{S7}
$$

Given a primitive with center ${ \pmb \mu } ,$ scale matrix $S$ and rotation matrix $R ,$ the world-to-canonical projection of a ray with origin x and direction d is given by:

$$
{ \bf x } _ { g } = S ^ { - 1 } R ^ { T } ( { \bf x } - { \bf \nabla } { \pmb { \mu } } ) ,\tag{S8}
$$

$$
\mathbf { d } _ { g } = S ^ { - 1 } R ^ { T } \mathbf { d } ,\tag{S9}
$$

The spatial derivatives can be evaluated as:

$$
\left[ \beta _ { x } \beta _ { y } \beta _ { z } \right] = - \frac { 1 } { 2 } ( d _ { \perp } ^ { 2 } ) _ { \mathbf { x } _ { g } } S ^ { - 1 } R ^ { T } I _ { 3 } ,\tag{S10}
$$

where $I _ { 3 }$ is the identity matrix. The angular term requires the chain rule for normalization, world-to-canonical projection and the Jacobian $J _ { \mathrm { a n g } }$ . We parameterize direction with spherical coordinates: let $\phi \in [ - \pi , \pi ]$ represent the ray’s azimuth in the reference xy plane, and $\theta \in [ - \pi / 2 , \pi / 2 ]$ be the elevation from this plane. Then $J _ { \mathrm { a n g } }$ is

$$
\begin{array} { r } { J _ { \mathrm { a n g } } = \left[ \mathbf { d } _ { \phi } \right] ^ { T } = \left[ \begin{array} { c c } { - \sin \phi \cos \theta } & { - \cos \phi \sin \theta } \\ { \cos \phi \cos \theta } & { - \sin \phi \sin \theta } \\ { 0 } & { \cos \theta } \end{array} \right] . } \end{array}\tag{S11}
$$

Therefore, the angular derivatives can be evaluated as:

$$
\big [ \beta _ { \phi } \quad \beta _ { \theta } \big ] = - \frac 1 2 ( d _ { \perp } ^ { 2 } ) _ { \hat { \mathbf { d } } _ { g } } ( \hat { \mathbf { d } } _ { g } ) _ { \mathbf { d } _ { g } } ( \mathbf { d } _ { g } ) _ { \mathbf { d } } J _ { \mathrm { a n g } } .\tag{S12}
$$

Each of these terms can be accumulated per-Gaussian and alpha-blended following standard ray-traced 3DGS rasterization [53]. To prevent numerical instability of the derivatives for flat primitives, we add $\varepsilon = 1 0 ^ { - 3 }$ to the scaling matrix during all forward and backward passes,

$$
S _ { i , i } ^ { - 1 } = \frac { 1 } { s _ { i } + \varepsilon } .\tag{S13}
$$

We find this causes a small drop in render quality as it imposes a lower bound on a primitive’s scale.

## 8. Additional Visualizations

Figure 7 presents additional visualizations of per-pixel eigenvalues for the plenoptic structure tensor M developed in Section 3.2. In a 3D world, M has 5 eigenvalues $\lambda _ { 0 } \ge \lambda _ { 1 } \ge . . . \ge \lambda _ { 4 } \ge 0$ . We find that $\lambda _ { 0 }$ is sensitive to edges, $\lambda _ { 1 }$ to corners, and $\lambda _ { 2 }$ to occlusion edges, specularity and reflections.

## 8.1. Instance Segmentation

Our approach uses InstanceGaussian [43] to build an instance-aware 3DGS representation. These instance cues enable change information from sparse anchors $\Delta \mathcal { G } _ { t } ^ { \prime }$ to propagate to all other primitives on the changed object. Thus, our SCD and CC performance depends on the instance segmentation quality. As noted in Section 5.1, PASLCD contains numerous small changed objects. Figure 8 shows two examples of undersegmentation (rows 1– 2) where the anchors (column 2) correctly localize changes, but are prevented from propagating by our instance gating due to an undersegmented representation (column 3), producing false negatives (columns 4–5). Substituting with oracle instances (rows 3–4) mitigates this, greatly reducing false negative predictions.

![](images/5b16c06fb56d8c96e7eecf82cd0fc28e4f787a2381020b1fc43eff6b30c53ebf.jpg)  
Figure 7. Pixelwise eigenvalue signals derived from the plenoptic structure tensor. Given an RGB view (column 1), we sample analytic plenoptic derivatives over a local light field to build pixelwise structure tensors M. The eigenvalues λ of M provide informative signals of scene structures, such as edges (column 2), corners (column 3) and occlusion edges as well as specular highlights (column 4).

## 8.2. Ray Selection Comparison

We compare ray selection approaches in Table 6, using the same 3DGS reconstructions for fair comparison. Querying all rays in the sample views yields comparable SCD performance. During ray extraction, most floating point operations (FLOPs) occur during DINOv2 inference, requiring entire frames regardless of the ray selection approach. Substantial compute savings occur during downstream change scoring: with edge rays occupying on average 3.6% of the frame, a proportional drop in FLOPs follows.

## 8.3. Robustness to Reconstruction Ambiguity

We follow the protocol in Section 5.4, reporting all changes detected between the 3DGS reconstructions as false positives. We investigate two conditions: 1) varying the fraction of overlapping views in each model’s training views, and 2)

![](images/2da20eb339768d617774f468c734cdd8306ed22ea6ca0e326d6d477b5d927a6b.jpg)  
Figure 8. Propagation of change information from anchors ∆G<sup>′</sup> (column 1, highlighted in green) to produce dense change representations ∆G<sub>t</sub> (column 3, highlighted in green) depends on the quality of the instance-aware 3DGS (column 2). While our anchors correctly identify changed objects, upstream undersegmentation (rows 1–2) can cause our instance gating to attenuate these anchors, creating false negative predictions (column 4). With oracle segmentations (rows 3–4), our predicted changes are considerably more accurate.

Table 6. Comparison of ray selection approaches averaged over all scenes in PASLCD [20] using 20 sample views each. Floating point operations reported for Extraction (Ext) do not include rasterization, which is shared across both methods. Retaining only edge rays (Ours) marginally improves SCD performance, but substantially reduces compute costs for change scoring (Sco).
<table><tr><td>Metric</td><td>All Rays</td><td>Edge Rays (Ours)</td></tr><tr><td>mIoU↑</td><td>0.517</td><td>0.538</td></tr><tr><td>F1↑</td><td>0.661</td><td>0.688</td></tr><tr><td>Ext TFLOPs ↓</td><td>35.0</td><td>34.9</td></tr><tr><td>Ext VRAM (GB) ↓</td><td>10.3</td><td>5.13</td></tr><tr><td>Sco GFLOPs ↓</td><td>1.11</td><td>0.043</td></tr><tr><td>Sco VRAM (GB) ↓</td><td>3.64</td><td>1.92</td></tr></table>

reducing the coverage of views provided to one model. For our method, we use the same $( \tau _ { \mathrm { i n s t } } , \tau _ { \mathrm { a n c } } , \tau _ { \mathrm { i n s t } } ^ { \mathrm { g e o } } )$ from the main results (Sec. 5.1, Sec. 5.3).

Table 7 summarizes our results for Cantina from PASLCD [20], a challenging scene with fine details and view-dependent objects prone to reconstruction ambiguity. As Figure 9 shows, our approach consistently reports fewer false positive changes than GS-Diff, stemming from our comparisons in plenoptic rather than primitive space.

Table 7. False positive rate from comparing two independently optimized 3DGS reconstructions of the Cantina scene from PASLCD [20] under no changes. Under varying training view overlaps and levels of coverage, our method consistently reports fewer false positives than GS-Diff [22].
<table><tr><td>Condition</td><td>Ours (%) ↓</td><td>GS-Diff (%) ↓</td></tr><tr><td>View Overlap</td><td></td><td></td></tr><tr><td>100% (full vs. full)</td><td>0.000</td><td>0.487</td></tr><tr><td>50% (half vs. half)</td><td>0.003</td><td>0.397</td></tr><tr><td>0% (half vs. half)</td><td>0.020</td><td>0.446</td></tr><tr><td>Coverage</td><td></td><td></td></tr><tr><td>Full vs. full</td><td>0.000</td><td>0.487</td></tr><tr><td>Full vs. half</td><td>0.002</td><td></td></tr><tr><td>Full vs. quarter</td><td>0.142</td><td>0.446 0.442</td></tr></table>

Our false positive rate increases under severe coverage asymmetry (Fig. 9 rows 3–4), but remains below GS-Diff. With very few views to constrain the optimization, reconstruction quality drops as fine details are lost, causing spurious change cues from ray differencing to leak into our pipeline. GS-Diff uses Fisher information to model observability as a way to absorb this reconstruction ambiguity, allowing a consistent false positive rate to be maintained.

100% Overlap  
Model A (rendered)  
Model B (rendered)  
Model A Inset  
Model B Inset  
![](images/d4e4aab0fe8f0c1b7f79de237d84265d9d765445a01a39c95306777a9c198130.jpg)  
Figure 9. Comparison of independently optimized 3DGS models of the Cantina scene from PASLCD without changes. At 100% view overlap, both models receive all training views. GS-Diff [22] reports a false positive rate of 0.487% as it directly compares primitives (inset), while we correctly report no changes. At 25% view coverage, Model B receives 5 out of 20 training views. Poor reconstruction quality causes our approach to detect false positives, albeit at a lower rate than GS-Diff.