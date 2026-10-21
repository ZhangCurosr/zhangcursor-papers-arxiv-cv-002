# Geometry Regularization for Vanilla NeRF via Self-Supervised Cross-Ray Consistency

ZONGJI WANG

Neural Radiance Fields (NeRF) achieve high-quality novel-view synthesis by representing a scene as a continuous volumetric radiance field. However, the geometry encoded by the density field of vanilla NeRF can be noisy and inaccurate, since diferent combinations of shape and radiance may explain the same input images. To reduce this ambiguity, we augment vanilla NeRF with a geometry-optimization branch based on selfsupervised cross-ray consistency. Specifically, this branch samples candidate surface points from the density field, generates virtual rays that intersect at each point from diferent directions, and renders their appearance using the NeRF network. Based on the visibility-weighted color consistency across these rays, we constrain the implicit geometry. We further regularize the implicit geometry by penalizing opacity at candidates with inconsistent appearance predictions and encouraging local normal smoothness. These geometric constraints are derived from the NeRF predictions and require no ground-truth depth or surface-normal supervision. Experiments on eight Replica scenes show that the proposed method reduces the mean Chamfer distance by 30.8% compared with vanilla NeRF.

Additional Key Words and Phrases: neural radiance fields, implicit geometry, 3D reconstruction, cross-ray consistency

## 1 INTRODUCTION

Recovering 3D shape from 2D observations is a cornerstone problem of computer vision and graphics. This long-standing research topic supports various applications in AR, VR, games, movies, etc. [1, 3] With the advances of inverse graphics, diferentiable renderer can be integrated in neural network pipelines, making it feasible to directly optimize the 3D geometric parameters using observed 2D images. Among various 3D geometry representations, implicit function represents 3D content continuously and without discretization. With its small memory footprint, implicit neural representation is popular in recent studies. One of the most successful approaches is NeRF (Neural Radiance Fields) [16].

NeRF is a scene representation method presented by Mildenhall et al. [16]. In that work, a deep neural network (typically, a multilayer perceptron, MLP) is applied to parameterize a continuous implicit function, which takes the position $\bar { \boldsymbol { x } } \in \mathbb { R } ^ { 3 }$ and viewing direction $d \in \mathbb { R } ^ { 3 }$ as input, and predicts the density $\sigma ( x ) \in \mathbb { R } ^ { + }$ and view-dependent color $c ( x , d ) \in \mathbb { R } ^ { 3 }$ . Through diferentiable volume rendering, dense colors from multiple 2D images can be used directly to optimize the neural radiance field. Encoding the information from multi-view 2D images, this powerful representation shows promising results in capturing complex scene structures and synthesizing 3D-consistent images with fine details. The superior rendering efects ignite research enthusiasm towards NeRFlike methods. Subsequent works include modification for dynamic scene modelling [9, 14], largescale scene representation [22, 23, 28], baking approaches for real-time rendering [5, 12, 31], encoding extra layers for scene understanding (e.g. intrinsic [33] and semantic [37] layers), and 3D-aware content or scene generation [6, 8, 10]. Although these approaches have shown impressive results for novel view synthesis, the 3D geometry extracted from the underlying volume density is far from satisfactory, usually sufering from noise and inaccuracy [30, 31].

Neural implicit functions help circumvent the limitations about complex pipelines in classic depth-based multi-view stereo (MVS) [21] and expensive memory consumption in voxel-based approaches [24, 26]. However, it is still a fundamentally under-constrained problem to recover accurate geometry from 2D images, even for approaches based on diferentiable rendering [2].

One of the main observations is the ambiguity between 3D shape and radiance, dubbed shaperadiance ambiguity [36]. As illustrated in Figure 1, diferent surface hypotheses can reproduce similar training-view colors through view-dependent radiance. This ambiguity makes it dificult to recover the scene’s actual geometry from image agreement alone [2, 13, 36].

![](images/2261a07bb5909c642723912973796f05264dec51a4f1b422ad8688c5309e0d14.jpg)  
Fig. 1. Schematic illustration of geometry–appearance ambiguity. (a) Two captured rays observe the same physical surface point � on a coherent 3D object. (b) An inflated, distorted surface intercepts the same rays at diferent points $Q _ { 1 }$ and $Q _ { 2 }$ . Based on view-dependent radiance, the incorrect geometry can nevertheless reproduce similar rendered RGB values along each captured ray, as indicated by the paired color swatches. Thus, image agreement alone does not uniquely determine the geometry. The dashed contour denotes the reference geometry, and the dashed ray segments extend toward � behind the incorrect surface.

A lot of efort has been dedicated to improve the 3D reconstruction quality of implicit neural representations. To put it briefly, there are works applying signed distance function (SDF) to model the 3D surface [18, 25, 30–32], combining neural implicit field with explicit 3D representations such as voxel [15, 29] and polygon mesh [5, 17], optimizing sampling approach of ray marching [15, 19, 20, 25], applying regularizers about smoothness and sparsity to the implicit field [16, 19, 34], adding priors about depth and structure to constrain the geometry [11, 19, 27, 35], and using multi-view consistency as MVS-based reconstruction [4, 7, 27].

In this paper, we directly refine the implicit geometry of vanilla NeRF to reduce its shape-radiance ambiguity. We choose vanilla NeRF [16] because its simple radiance-field representation supports complex scenes and arbitrary camera poses. However, as its geometry is represented only by volume density, the extracted surfaces often contain noise and inaccuracies. To improve the geometric fidelity, we build on the observation that the appearance of a surface point cannot vary arbitrarily across nearby viewpoints [13]. Existing neural reconstruction methods use this photo-consistency cue through feature matching or image warping between captured views [4, 7, 27]. We instead extend the NeRF architecture with a branch that samples candidate surface points, generates virtual viewpoints around them, and renders the corresponding cross-ray appearances. Based on these predictions, we obtain a self-supervised signal for optimizing the density field.

Contributions. We summarize the contributions as follows:

• We design a geometry-optimization branch for vanilla NeRF that derives geometric cues from the density field and integrates multi-view appearance constraints into the diferentiable rendering pipeline.

• We propose visibility-weighted, self-supervised cross-ray consistency and penalize opacity at candidates whose rendered appearances disagree across virtual views.

• Experiments on eight Replica scenes show a 30.8% reduction in mean Chamfer distance relative to vanilla NeRF, demonstrating the efectiveness of the proposed framework in improving implicit geometry.

## 2 RELATED WORK

## 2.1 3D Geometric Representations

3D geometry is commonly represented by volumes, point clouds, meshes, or implicit functions. Volumetric representations encode spatial structure directly [26], but their memory consumption often limits resolution. Point clouds are compact and flexible, although they lack an explicit neighborhood structure. Meshes are compatible with conventional graphics pipelines [31], but changes in topology require modifying their connectivity. Implicit representations instead describe geometry through continuous functions, such as occupancy fields or signed distance fields (SDFs) [18, 30]. Neural networks parameterize these functions without maintaining a dense spatial grid. We focus on the geometry encoded by implicit neural representations.

## 2.2 Implicit Neural Representations

NeRF [16] represents a scene with a neural network that maps 3D positions and viewing directions to density and view-dependent color. Through diferentiable volume rendering, it learns the radiance field directly from posed images. Its continuous representation supports novel-view synthesis without explicitly reconstructing a surface during training.

Subsequent work extends this representation to dynamic scenes [9, 14], large-scale environments [22, 23, 28], and scene understanding [33, 37]. Specifically, Li et al. [14] model scene motion through forward and backward flow fields, while Fang et al. [9] use time-aware voxel features. For large environments, Mega-NeRF and Block-NeRF decompose scenes into local radiance fields, whereas BungeeNeRF progressively grows its representation. Semantic extensions jointly encode scene labels and appearance, and intrinsic models separate reflectance, shading, and residual components. Neural fields also support 3D-aware content generation [6, 8, 10]. However, high-quality rendered images do not ensure accurate geometry: the surfaces extracted from the underlying density field can still contain noise and inaccuracies [30, 31]. We therefore focus on improving implicit geometry through neural rendering.

## 2.3 Reconstruction through Neural Rendering

Recovering geometry from images remains under-constrained because diferent combinations of shape and radiance can explain the same observations [2, 36]. Existing methods address this ambiguity through surface representations, sampling strategies, geometric priors, and multi-view constraints.

One approach is to make the surface representation more explicit. Niemeyer et al. [18] proposed Diferentiable Volumetric Rendering (DVR), which models shape with an occupancy network and diferentiates the surface projection through implicit diferentiation. VolSDF [30] and NeuS [25] instead optimize SDF-based surfaces through volume rendering. DVR uses object masks during training, while SDF-based methods such as VolSDF and NeuS can reconstruct surfaces without mask supervision. In NeuS, the volume-rendering formulation reduces the geometric bias in the recovered zero-level set. Appropriate initialization and a bounding-sphere prior also guide its surface optimization [25]. UNISURF [19] unifies implicit surfaces and radiance fields and progressively narrows the sampling region around the estimated surface, allowing reconstruction without input masks.

Hybrid representations combine neural fields with explicit spatial structures. Neural Sparse Voxel Fields (NSVF) [15] organizes local implicit fields in sparse voxels, which also restrict the sampling region during ray marching. Its sparse voxel structure avoids evaluating empty regions, and progressive optimization supports the recovery of finer scene geometry. Munkberg et al. [17] optimize a discrete SDF on a deformable tetrahedral grid and render the resulting mesh during training. Their framework estimates geometry, spatially varying materials, and lighting from posed images and object masks. For eficient rendering, MobileNeRF [5] encodes appearance and opacity on polygon faces and bakes the result into textures, while BakedSDF [31] optimizes a hybrid surfacevolume representation for unbounded scenes and bakes it for rasterization. Such approaches modify the representation or rendering pipeline to connect neural reconstruction with explicit geometry.

Besides the representation, geometric regularization and additional cues can improve reconstruction. NeRF [16] injects noise into density predictions to regularize ambiguous regions. UNISURF [19] encourages local normal smoothness, and PlenOctrees [34] uses a sparsity prior to favor empty space in ambiguous regions. MonoSDF [35] incorporates monocular depth and normal estimates to constrain the surface. Based on predictions from general-purpose monocular estimators, these cues improve reconstruction quality and reduce optimization time. Guo et al. [11] apply planar constraints to wall and floor regions based on the Manhattan-world assumption. Semantic segmentation identifies these regions, and the constraints encourage their surface normals to follow parallel or orthogonal directions. Other approaches use direct geometric observations: Rematas et al. [20] use lidar supervision to constrain reconstruction and limit ray-marching samples to a bounded region. These constraints reduce ambiguity by imposing smoothness, sparsity, or prior geometric information.

Multi-view methods obtain geometric cues from agreement between captured images. MVS-NeRF [4] builds plane-swept cost volumes for geometry-aware scene reasoning and combines them with volume rendering to reconstruct a radiance field from a small set of nearby views. Nerfing-MVS [27] uses learned depth priors, adapted with sparse structure-from-motion observations, to guide NeRF optimization. NeuralWarp [7] combines image warping with volume rendering to optimize photo-consistency across source views. These methods use captured images or depth priors to constrain the geometry through multi-view observations. Direct photo-consistency methods must also account for source-view selection and visibility when comparing projections of the same 3D point.

In this paper, we constrain the density field of vanilla NeRF through agreement among virtual rays rendered by the same network. Specifically, the cross-ray constraint compares appearance predictions around each candidate point, with visibility weights reducing the influence of occluded rays. This provides a self-supervised geometric constraint without image warping or additional depth, normal, or correspondence supervision in the training losses.

## 3 CROSS-RAY CONSISTENCY FOR IMPLICIT GEOMETRY OPTIMIZATION

Our framework extends vanilla NeRF with a geometry-optimization branch, as shown in Figure 2. The NeRF network still predicts density and view-dependent color from 3D positions and viewing directions. The added branch samples candidate surface points from the current density field, generates virtual viewpoints around them, and performs multi-view appearance rendering for rays passing through each point. Based on these predictions, we impose visibility-aware cross-ray consistency and geometric regularization to optimize the field.

![](images/fd3b589a83579495e7b45ce2726d920000aadece2f0ee5994cb1d64e6472abdb.jpg)  
Fig. 2. Overview of the proposed framework. The shared NeRF renders training views for RGB reconstruction and supplies the density field for candidate surface sampling. The geometry-optimization branch generates virtual cross-rays through each candidate, renders their appearances, and estimates visibility weights to reduce the influence of occluded rays. Cross-ray consistency, occupancy suppression, and local normal smoothness jointly regularize the field.

## 3.1 Overview and Cross-Ray Consistency

A vanilla NeRF represents a scene with a function $F _ { \theta }$ that maps a position $\mathbf { x } \in \mathbb { R } ^ { 3 }$ and viewing direction $\mathbf { d } \in \mathbb { R } ^ { 3 }$ to density $\sigma _ { \boldsymbol { \theta } } ( \mathbf { x } )$ and color $\mathbf { c } _ { \boldsymbol { \theta } } ( \mathbf { x } , \mathbf { d } )$ :

$$
F _ { \theta } ( \mathbf { x } , \mathbf { d } ) = ( \sigma _ { \theta } ( \mathbf { x } ) , \mathbf { c } _ { \theta } ( \mathbf { x } , \mathbf { d } ) ) .\tag{1}
$$

For a camera ray $\mathbf { r } ( t ) = \mathbf { o } +$ �d with ordered samples $\{ t _ { k } \} _ { k = 1 } ^ { K }$ , volume rendering gives

$$
\widehat { \mathbf { C } } ( \mathbf { r } ) = \sum _ { k = 1 } ^ { K } T _ { k } \alpha _ { k } \mathbf { c } _ { \theta } ( \mathbf { r } ( t _ { k } ) , \mathbf { d } ) ,\tag{2}
$$

$$
\alpha _ { k } = 1 - \exp \bigl ( - \sigma _ { \theta } ( { \mathbf { r } ( t _ { k } ) } ) \delta _ { k } \bigr ) , \qquad T _ { k } = \prod _ { l < k } ( 1 - \alpha _ { l } ) ,\tag{3}
$$

where $\delta _ { k } = t _ { k + 1 } - t _ { k }$ . The NeRF rendering branch minimizes the image reconstruction loss

$$
L _ { \mathrm { r g b } } = \sum _ { \mathbf { r } \in \mathcal { R } } \left\| \widehat { \mathbf { C } } ( \mathbf { r } ) - \mathbf { C } ( \mathbf { r } ) \right\| _ { 2 } ^ { 2 } .\tag{4}
$$

Multi-view reconstruction commonly compares the projections of the same 3D point in captured images. Such methods must select source views, establish correspondences, and handle visibility [7]. Our added branch instead generates a local bundle of virtual rays for each candidate surface point and renders every ray with the NeRF network. These rays intersect at the same point but difer in origin and direction. If the candidate lies on a stable surface, its appearance should remain similar across the visible rays in this local bundle. We therefore use their color disagreement as a self-supervised constraint, without image warping or ground-truth depth.

## 3.2 Candidate Surface Sampling

As illustrated in Figure 3(a), we combine random proposals with rendered-depth anchors to obtain candidate surface points. We first sample random points in a bounded scene volume V and query the current density field. With a fixed interval �, we convert density to opacity,

$$
\alpha ( \mathbf { x } ) = 1 - \exp \bigl ( - \sigma _ { \theta } ( \mathbf { x } ) \delta \bigr ) .\tag{5}
$$

We then select candidate points from an intermediate opacity band:

$$
S _ { \mathrm { r a n d } } = \left\{ \mathbf { x } \in \mathcal { V } \mid \alpha _ { \mathrm { m i n } } < \alpha ( \mathbf { x } ) < \alpha _ { \mathrm { m a x } } \right\} , \qquad \alpha _ { \mathrm { m i n } } = 0 . 5 , \quad \alpha _ { \mathrm { m a x } } = 0 . 9 5 .\tag{6}
$$

We retain intermediate-opacity samples as surface candidates, excluding low-density samples and those with nearly saturated opacity.

To complement the random candidates, we also obtain surface anchors from rendered training rays. Specifically, using the rendering weights in Equation (2), we define the unnormalized rendereddepth estimate $\begin{array} { r } { \widehat { t } ( \mathbf { r } ) = \sum _ { k } T _ { k } \alpha _ { k } t _ { k } } \end{array}$ . The corresponding anchor is

$$
\mathbf { x } _ { \mathrm { r a y } } = \mathbf { o } + \widehat { t } ( \mathbf { r } ) \mathbf { d } , \qquad S = S _ { \mathrm { r a n d } } \cup S _ { \mathrm { r a y } } .\tag{7}
$$

We apply the same opacity-band test to the rendered-depth anchors before adding them to $S _ { \mathrm { r a y } }$ Based on the combined set S, we randomly retain a candidate batch for the geometry branch. The uniform proposals explore the current field, while the rendered-depth anchors provide candidates at surface locations estimated along the training rays.

![](images/1d970c8d7a19eeaf0af21e698ce4aef5069f14bc152e4bc1b2a5f5f3c22880fe.jpg)

![](images/2b511ee5f249a3b406448984619bc76eadc3b9d5566b0449e589e4ec87286ee9.jpg)

![](images/abb598437a3c94cbf33fe78f20f67e7481828245d1e75ab6510d59929fbe6195.jpg)  
Fig. 3. Details of candidate sampling and visibility-aware cross-ray construction. (a) Uniform proposals and rendered-depth anchors are filtered by an intermediate opacity band to obtain candidate points. The curved patch schematically represents the current density boundary. (b) For a candidate $\mathbf { x } _ { i } ,$ the auxiliary sphere is centered at $\mathbf { x } _ { i } + l _ { 0 } \mathbf { n } _ { i }$ with radius �. Its sampled locations $\widetilde { \mathbf { o } } _ { i j }$ determine ray directions, while the actual origins $\mathbf { o } _ { i j }$ are placed independently at distances $l _ { i j }$ . All generated rays pass through the same candidate. (c) Opacity samples along each origin-to-candidate segment determine a continuous visibility weight: an intervening occupied region increases the opacity sum and reduces the $\mathrm { \ r a y { ' } } s$ influence. This segment is used for visibility estimation; appearance is obtained by standard volume rendering along the generated ray.

## 3.3 Cross-Ray Generation and Visibility Estimation

Figure 3(b) illustrates the cross-ray construction. For each candidate $\mathbf { x } _ { i } \in S$ , we first estimate a local surface direction from the opacity gradient:

$$
\mathbf { n } _ { i } = - \frac { \nabla _ { \mathbf { x } } \alpha ( \mathbf { x } _ { i } ) } { \| \nabla _ { \mathbf { x } } \alpha ( \mathbf { x } _ { i } ) \| _ { 2 } + \epsilon } .\tag{8}
$$

This normal is computed on the fly from the radiance field rather than supplied as ground truth. We keep its computation in the automatic-diferentiation graph so that cross-ray construction and normal smoothness can update the density field. Next, we sample $\mathbf { u } _ { i j } \sim \mathcal { U } ( \mathbb { S } ^ { 2 } )$ and form an auxiliary location

$$
\widetilde { \mathbf { o } } _ { i j } = \mathbf { x } _ { i } + l _ { 0 } \mathbf { n } _ { i } + r \mathbf { u } _ { i j } .\tag{9}
$$

Here $l _ { 0 }$ is a normal-direction bias: it shifts the isotropic perturbation $r \mathbf { u } _ { i j }$ toward the estimated outward side of the candidate surface. The auxiliary location only determines the ray direction. Defining the normal-bias strength $\lambda _ { \mathrm { n } } = l _ { 0 } / r$ for $r > 0$ , we obtain

$$
\mathbf { d } _ { i j } = \frac { \mathbf { x } _ { i } - \widetilde { \mathbf { o } } _ { i j } } { \left\| \mathbf { x } _ { i } - \widetilde { \mathbf { o } } _ { i j } \right\| _ { 2 } } = - \frac { \lambda _ { \mathrm { n } } \mathbf { n } _ { i } + \mathbf { u } _ { i j } } { \left\| \lambda _ { \mathrm { n } } \mathbf { n } _ { i } + \mathbf { u } _ { i j } \right\| _ { 2 } } .\tag{10}
$$

Thus, the virtual-viewpoint directions depend on $l _ { 0 }$ and � through their ratio. When $\lambda _ { \mathrm { n } } = 0$ , they are isotropic around the candidate; increasing $\lambda _ { \mathrm { n } }$ continuously favors the estimated normal side without explicitly rejecting sampled directions. We independently draw $l _ { i j } \sim \mathcal { U } ( l _ { \mathrm { m i n } } , l _ { \mathrm { m a x } } )$ to place the final ray origin:

$$
\mathbf { o } _ { i j } = \mathbf { x } _ { i } - l _ { i j } \mathbf { d } _ { i j } , \qquad \mathbf { r } _ { i j } ( t ) = \mathbf { o } _ { i j } + t \mathbf { d } _ { i j } .\tag{11}
$$

The ratio $\lambda _ { \mathrm { n } }$ therefore controls the angular normal bias, whereas $l _ { i j }$ controls the actual ray-origin distance. By construction, all rays indexed by � intersect at $\mathbf { x } _ { i } .$ . Standard volume rendering in Equation (2) gives their colors $\mathbf { c } _ { i j } = \widehat { \mathbf { C } } ( \mathbf { r } _ { i j } )$

However, a generated ray may pass through another occupied region before reaching the candidate, as shown in Figure 3(c). To reduce the influence of such rays, we sample � points $\{ \mathbf { p } _ { i j k } \} _ { k = 1 } ^ { M }$ on the segment from $\mathbf { o } _ { i j }$ to $\mathbf { x } _ { i }$ and compute

$$
\alpha _ { i j k } = 1 - \exp \mathopen { } \mathclose \bgroup \mathopen { } \mathclose \bgroup \mathopen { } \mathclose \bgroup \mathopen { } \mathclose \bgroup \mathopen { } \mathclose \bgroup \mathopen { } \mathclose \bgroup \mathopen { } \mathclose \bgroup \mathopen { } \mathclose \bgroup \mathopen { } \mathclose \bgroup \mathopen { } \mathclose \bgroup \mathopen { } \mathclose \bgroup \left( - \sigma _ { \theta } \mathopen { } \mathclose \bgroup \left( \mathbf { p } _ { i j k } \aftergroup \egroup \aftergroup \egroup \aftergroup \egroup \aftergroup \egroup \aftergroup \egroup \aftergroup \egroup \aftergroup \egroup \aftergroup \egroup \aftergroup \egroup \aftergroup \egroup \aftergroup \egroup \aftergroup \egroup \aftergroup \egroup \aftergroup \egroup \aftergroup \egroup \right) \Delta _ { i j k } \aftergroup \egroup \aftergroup \egroup \aftergroup \egroup \aftergroup \egroup \aftergroup \egroup \aftergroup \egroup \aftergroup \egroup \aftergroup \egroup \right) ,\tag{12}
$$

$$
m _ { i j } = \sum _ { k = 1 } ^ { M } \alpha _ { i j k } ,\tag{13}
$$

$$
w _ { i j } = \frac { 1 } { 1 + \exp \big ( \beta ( m _ { i j } - \gamma ) \big ) } , \qquad \beta = 2 0 , \quad \gamma = 1 . 1 0 .\tag{14}
$$

Here $m _ { i j }$ is an accumulated opacity proxy, defined as the sum of per-sample opacities rather than physical transmittance. Based on this proxy, the sigmoid assigns a smaller weight to rays whose paths contain substantial density. We treat the estimated visibility weights as constants during backpropagation (stop gradients through $w _ { i j } )$ , preventing the consistency loss from directly adjusting them to downweight rays with inconsistent color predictions.

## 3.4 Geometry Regularization

To measure the appearance agreement of the visible rays, let $\begin{array} { r } { D _ { i } = \operatorname* { m a x } ( \sum _ { j } w _ { i j } , \eta ) } \end{array}$ with $\eta = 0 . 1$ for candidate �. We then compute the visibility-weighted mean color and cross-ray variance as

$$
\pmb { \mu } _ { i } = \frac { \sum _ { j } w _ { i j } \mathbf { c } _ { i j } } { D _ { i } } ,\tag{15}
$$

$$
V _ { i } = \frac { \sum _ { j } w _ { i j } \left\| \mathbf { c } _ { i j } - \pmb { \mu } _ { i } \right\| _ { 2 } ^ { 2 } } { D _ { i } } .\tag{16}
$$

The cross-ray consistency loss is the average variance over � candidates:

$$
L _ { \mathrm { C R C } } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } V _ { i } .\tag{17}
$$

We assume that a valid surface hypothesis should exhibit similar appearance across visible virtual views within the sampled local angular range. A large cross-view rendering discrepancy therefore provides negative evidence for occupancy at the candidate point. Minimizing color variance alone may change the appearance predictions while leaving unreliable density in place. We therefore use the visibility-weighted variance to penalize the candidate point’s opacity:

$$
L _ { \mathrm { o c c } } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathbf { s g } ( V _ { i } ) \alpha ( \mathbf { x } _ { i } ) ,\tag{18}
$$

where sg(·) denotes stop-gradient. The variance acts as a fixed weight in this term: a larger $V _ { i }$ gives a stronger penalty on $\alpha ( \mathbf { x } _ { i } )$ . Stop-gradient prevents the penalty from being reduced by changing $V _ { i }$ through this path.

Although �<sub>CRC</sub> is expressed through rendered colors, it also afects geometry. Specifically, each rendered color depends on density through opacity and transmittance in Equation (2), and the cross-ray directions depend on diferentiable opacity gradients. In addition, Equation (18) directly penalizes the opacity of inconsistent candidates. The opacity-band test itself is discrete and provides no gradient; instead, it updates the candidate distribution as the field changes. Meanwhile, $L _ { \mathrm { r g b } }$ keeps the predictions anchored to the captured images during refinement. This reasoning assumes that surface appearance varies modestly across the sampled local directions; strong specular efects may therefore penalize valid geometry.

To further reduce high-frequency noise in the density field, we apply local normal smoothness. Given a random perturbation $\mathbf { \bar { \varepsilon } } _ { \varepsilon _ { i } } \sim \mathcal { U } ( - s / 2 , s / 2 ) ^ { 3 }$ and $\mathbf { x } _ { i } ^ { \prime } = \mathbf { x } _ { i } + \pmb { \varepsilon } _ { i }$

$$
L _ { \mathrm { s m o o t h } } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \left\| \mathbf { n } ( \mathbf { x } _ { i } ) - \mathbf { n } ( \mathbf { x } _ { i } ^ { \prime } ) \right\| _ { 2 } .\tag{19}
$$

Here, $s > 0$ denotes the side length of the cubic neighborhood used to sample local perturbations.

## 3.5 Training Objective

We combine RGB reconstruction from the NeRF path with the constraints from the geometry branch in the training objective

$$
L = L _ { \mathrm { r g b } } + \lambda _ { \mathrm { c r } } \left( L _ { \mathrm { C R C } } + \lambda _ { \mathrm { o c c } } L _ { \mathrm { o c c } } \right) + \lambda _ { \mathrm { s m } } L _ { \mathrm { s m o o t h } } .\tag{20}
$$

Here $L _ { \mathrm { r g b } }$ maintains agreement with the captured images, while �<sub>CRC</sub> supplies the cross-ray constraint. The occupancy and smoothness terms act on the geometry in complementary ways: one suppresses inconsistent density, and the other reduces local noise.

Algorithm 1 summarizes one optimization update with a nonempty candidate batch. Specifically, the geometry branch uses the current field to select candidates, construct cross-rays, and render their colors. Based on these predictions, it evaluates the visibility-weighted losses and updates the shared network together with RGB reconstruction. The normal computation and appearance rendering remain diferentiable, whereas the visibility weights and the variance used in the occupancy penalty are detached as described above.

Algorithm 1 One update with visibility-aware cross-ray consistency   
Input: Shared NeRF $F _ { \theta ; }$ , training-ray batch ${ \mathcal { R } } ,$ sampling volume $\boldsymbol { \mathcal { V } } ,$ , candidate budget $N _ { \mathrm { m a x } }$ , cross  
ray count �   
1: Render R to obtain $L _ { \mathrm { r g b } }$ and rendered-depth anchors (Equations (4) and (7)).   
2: Draw uniform proposals from $_ \mathrm { ~ \textit ~ { ~ N ~ } ~ }$ and append the rendered-depth anchors.   
3: Apply the opacity-band test and randomly retain up to $N _ { \mathrm { m a x } }$ candidates $\{ \mathbf { x } _ { i } \} _ { i = 1 } ^ { N } .$   
4: for each candidate $i = 1 , \ldots , N$ do   
5: Compute the diferentiable opacity normal ${ \bf n } _ { i }$ (Equation (8)).   
6: for each virtual ray $j = 1 , \dots , J$ do   
7: Draw $\mathbf { u } _ { i j } \sim \mathcal { U } ( \mathbb { S } ^ { 2 } )$ and $l _ { i j } \sim \mathcal { U } ( l _ { \mathrm { m i n } } , l _ { \mathrm { m a x } } )$   
8: Construct the normal-biased direction ${ \bf { d } } _ { i j }$ (Equation (10)).   
9: Set $\mathbf { o } _ { i j } = \mathbf { x } _ { i } - l _ { i j } \mathbf { d } _ { i j }$ and $\mathbf { r } _ { i j } ( t ) = \mathbf { o } _ { i j } + t \mathbf { d } _ { i j }$   
10: Render $\mathbf { c } _ { i j } = \widehat { \mathbf { C } } ( \mathbf { r } _ { i j } )$ by Equation (2).   
11: Sum sampled opacities on the origin-to-candidate segment to obtain $m _ { i j }$   
12: Set $w _ { i j } = \mathrm { s g } \big ( [ 1 + \exp ( \beta ( m _ { i j } - \gamma ) ) ] ^ { - 1 } \big )$   
13: end for   
14: Compute $\pmb { \mu } _ { i }$ and $V _ { i }$ using $w _ { i j }$ (Equation (16)).   
15: end for   
16: Compute �<sub>CRC</sub>, $\begin{array} { r } { L _ { \mathrm { o c c } } = N ^ { - 1 } \sum _ { i } \mathbf { s g } ( V _ { i } ) \alpha ( \mathbf { x } _ { i } ) } \end{array}$ , and �<sub>smooth</sub>.   
17: Form the training objective in Equation (20) and update $\theta .$   
Output: Updated shared NeRF $F _ { \theta }$

## 4 EXPERIMENTS

## 4.1 Implementation Details

For the reported experiments, we initialize the NeRF network from a vanilla-NeRF checkpoint and continue optimization with the objective in Equation (20). In each geometry-optimization step, we draw 20,000 uniform proposals from the sampling volume, append rendered-depth anchors, apply the opacity-band filter, and randomly retain up to 100 candidates. We generate 15 cross-rays for each candidate. For the opacity and normal computations, we use $\delta = ( 1 0 . 0 - 0 . 1 ) / 1 2 8$ and $\epsilon = 1 0 ^ { - 5 }$ respectively. Unless noted below, we set $r = 0 . 1 5$ and $l _ { 0 } = 0 . 2 8$ , giving a default normal-bias strength of $\lambda _ { \mathrm { n } } = l _ { 0 } / r \approx 1 . 8 7$ . We also set $( l _ { \mathrm { m i n } } , l _ { \mathrm { m a x } } ) = ( 0 . 1 5 , 0 . 4 5 ) , s = 0 . 0 8 , \lambda _ { \mathrm { c r } } = 1 . 5 \times 1 0 ^ { - 3 } , \lambda _ { \mathrm { o c c } } = 1 . 1 5$ , and $\lambda _ { \mathrm { s m } } = 2 . 5 \times 1 0 ^ { - 4 }$

For visibility estimation, we place 32 uniform samples between $0 . 0 1 l _ { i j }$ and $l _ { i j }$ along each originto-candidate segment, including the endpoint, and set $\Delta _ { i j k } = 0 . 9 9 l _ { i j } / 3 2 .$ Cross-ray rendering uses near and far bounds of 0.1 and 10.0 meters. The continuation with the geometry branch starts at a learning rate of $1 . 1 5 \times 1 0 ^ { - 4 }$

Room 1 uses $( l _ { \mathrm { m i n } } , l _ { \mathrm { m a x } } ) = ( 0 . 1 2 5 , 0 . 3 0 ) , \lambda _ { \mathrm { o c c } } = 0 . 8$ , and $\lambda _ { \mathrm { s m } } = 2 . 5 \times 1 0 ^ { - 3 }$ . Ofice 2 uses $r = 0 . 2 5$ $( l _ { \mathrm { m i n } } , l _ { \mathrm { m a x } } ) = ( 0 . 0 8 , 0 . 7 0 ) , s = 0 . 1 0 , \lambda _ { \mathrm { c r } } = 1 . 0 5 \times 1 0 ^ { - 3 }$ , and $\lambda _ { \mathrm { o c c } } = 3 . 2 5 ;$ its scene-specific continuation starts at a learning rate of $1 . 2 5 \times 1 0 ^ { - 4 }$ . These settings are fixed in the reported runs.

For Ofice 2, we additionally suppress residual floating density with an auxiliary global sparsity penalty:

$$
L _ { \mathrm { s p a r s e } } = 1 - \mathbb { E } _ { \mathbf { x } \sim \mathcal { V } } \left[ \exp \left( - \delta _ { s } \mathrm { R e L U } ( \sigma _ { \theta } ( \mathbf { x } ) ) \right) \right] .\tag{21}
$$

The optimization objective for this scene is $L + \lambda _ { \mathrm { s p } } L _ { \mathrm { s p a r s e } }$ , where � is defined in Equation (20), $\lambda _ { \mathrm { s p } } = 1 0 ^ { - 2 }$ , and $\delta _ { s } = 0 . 0 1$ . This auxiliary term is not used for the other seven scenes.

For the comparison in Table 1, both vanilla NeRF and our method are evaluated after 300,000 cumulative training iterations. When our run is initialized from a vanilla-NeRF checkpoint, its continuation iterations are counted within this total. Thus, both methods have the same iteration budget.

All surface normals used during refinement are computed on the fly from opacity gradients; ground-truth normals are not used. For Replica scenes, the reference mesh determines an oriented 3D sampling volume, but does not enter the training losses. Ground-truth depth is reserved for evaluation. Thus, the losses use no ground-truth depth or normal supervision, although the sampling bounds still depend on reference geometry in the current experiments.

## 4.2 Geometry Reconstruction Results

Table 1 reports the Chamfer distance on eight Replica scenes at the matched 300,000-iteration budget. Compared with vanilla NeRF, our method improves six scenes and reduces the mean distance by 30.8%. Room 1 and Ofice 0 are the two scenes for which the distance increases.

Table 1. Chamfer distance on Replica scenes after 300,000 cumulative training iterations for both methods multiplied by 100 and rounded to two decimal places (lower is beter). Relative changes are computed from the unrounded distances with respect to vanilla NeRF; downward arrows indicate reductions. The lower distance in each scene and the entire mean row are highlighted in bold.
<table><tr><td>Scene</td><td>Baseline (Vanilla NeRF)</td><td>Ours</td><td>Relative change</td></tr><tr><td>Room 0</td><td>0.49</td><td>0.29</td><td>↓40.0%</td></tr><tr><td>Room 1</td><td>0.54</td><td>0.65</td><td>↑21.2%</td></tr><tr><td>Room 2</td><td>0.67</td><td>0.22</td><td>↓67.4%</td></tr><tr><td>Office 0</td><td>0.41</td><td>0.48</td><td>↑16.2%</td></tr><tr><td>Office 1</td><td>2.78</td><td>2.43</td><td>↓12.5%</td></tr><tr><td>Office 2</td><td>1.33</td><td>0.48</td><td>↓64.2%</td></tr><tr><td>Office 3</td><td>1.42</td><td>0.85</td><td>↓40.4%</td></tr><tr><td>Office 4</td><td>0.58</td><td>0.29</td><td>↓50.5%</td></tr><tr><td>Mean</td><td>1.03</td><td>0.71</td><td>↓30.8%</td></tr></table>

Figure 4 qualitatively compares the geometry extracted from the density fields with the groundtruth scene geometry. Although vanilla NeRF recovers the overall scene layout, its surfaces often contain pronounced noise and irregular deformations. After geometry optimization, several wall and furniture surfaces become smoother and more closely resemble the reference geometry. Specifically, the wall in Room 1 and the tabletops in Room 2 and Ofice 2 exhibit reduced surface irregularities. Together with the quantitative results, these comparisons demonstrate that the proposed geometry optimization can improve the geometric fidelity of NeRF’s underlying density field. The smoother wall in Room 1 is a local improvement, despite the higher scene-level Chamfer distance reported in Table 1.

wo Geo Loss

Ground Truth  
w Geo Loss  
![](images/27edc3a82de8c08df6ee78dc63c5237ee198c7b8e136df5e4f0be08e91389b65.jpg)  
Fig. 4. Qualitative geometry comparison on Replica scenes. From left to right, each row shows the ground-truth geometry, vanilla NeRF, and the reconstruction after optimization with the proposed geometry-refinement objective.

## 5 CONCLUSION AND DISCUSSION

We proposed a framework that augments vanilla NeRF with a geometry-optimization branch. Specifically, the branch samples candidate surface points, generates virtual cross-rays through them, and renders their appearance from multiple viewpoints. Based on the visibility-weighted disagreement of these rendered colors, cross-ray consistency constrains the density field, while occupancy suppression and local normal smoothness reduce geometric noise. On eight Replica scenes, the method reduces the mean Chamfer distance by 30.8% relative to vanilla NeRF.

The current experiments use a reference mesh to define sampling bounds and scene-specific settings for Room 1 and Ofice 2. Future work should estimate the bounds from training cameras, isolate the efect of the added branch through an RGB-only continuation from the same checkpoint, and investigate appearance models that are less sensitive to specular surfaces.

## REFERENCES

[1] Dejan Azinović, Ricardo Martin-Brualla, Dan B Goldman, Matthias Nießner, and Justus Thies. 2022. Neural RGB-D Surface Reconstruction. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 6290–6301.

[2] Jonathan T Barron, Ben Mildenhall, Dor Verbin, Pratul P Srinivasan, and Peter Hedman. 2022. Mip-nerf360: Unbounded anti-aliased neural radiance fields. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 5470–5479.

[3] Mark Boss, Andreas Engelhardt, Abhishek Kar, Yuanzhen Li, Deqing Sun, Jonathan T. Barron, Hendrik P.A. Lensch, and Varun Jampani. 2022. SAMURAI: Shape And Material from Unconstrained Real-world Arbitrary Image collections. In Advances in Neural Information Processing Systems (NeurIPS).

[4] Anpei Chen, Zexiang Xu, Fuqiang Zhao, Xiaoshuai Zhang, Fanbo Xiang, Jingyi Yu, and Hao Su. 2021. Mvsnerf: Fast generalizable radiance field reconstruction from multi-view stereo. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision. 14124–14133.

[5] Zhiqin Chen, Thomas Funkhouser, Peter Hedman, and Andrea Tagliasacchi. 2023. MobileNeRF: Exploiting the Polygon Rasterization Pipeline for Eficient Neural Field Rendering on Mobile Architectures. In The Conference on Compute Vision and Pattern Recognition (CVPR).

[6] Zhaoxi Chen, Guangcong Wang, and Ziwei Liu. 2023. SceneDreamer: Unbounded 3D Scene Generation from 2D Image Collections. In arXiv.

[7] François Darmon, Bénédicte Bascle, Jean-Clément Devaux, Pascal Monasse, and Mathieu Aubry. 2022. Improving neural implicit surfaces geometry with patch warping. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition. 6260–6269.

[8] Yu Deng, Jiaolong Yang, Jianfeng Xiang, and Xin Tong. 2022. GRAM: Generative Radiance Manifolds for 3D-Aware Image Generation. In IEEE/CVF Conference on Computer Vision and Pattern Recognition

[9] Jiemin Fang, Taoran Yi, Xinggang Wang, Lingxi Xie, Xiaopeng Zhang, Wenyu Liu, Matthias Nießner, and Qi Tian. 2022. Fast dynamic radiance fields with time-aware neural voxels. In SIGGRAPH Asia 2022 Conference Papers. 1–9.

[10] Jiatao Gu, Lingjie Liu, Peng Wang, and Christian Theobalt. 2022. StyleNeRF: A Style-based 3D Aware Generator for High-resolution Image Synthesis. In International Conference on Learning Representations.

[11] Haoyu Guo, Sida Peng, Haotong Lin, Qianqian Wang, Guofeng Zhang, Hujun Bao, and Xiaowei Zhou. 2022. Neural 3d scene reconstruction with the manhattan-world assumption. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition. 5511–5520.

[12] Peter Hedman, Pratul P. Srinivasan, Ben Mildenhall, Jonathan T. Barron, and Paul Debevec. 2021. Baking Neura Radiance Fields for Real-Time View Synthesis. ICCV (2021).

[13] Kiriakos N Kutulakos and Steven M Seitz. 2000. A theory of shape by space carving. International journal of compute vision 38 (2000), 199–218.

[14] Zhengqi Li, Simon Niklaus, Noah Snavely, and Oliver Wang. 2021. Neural scene flow fields for space-time view synthesis of dynamic scenes. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 6498–6508.

[15] Lingjie Liu, Jiatao Gu, Kyaw Zaw Lin, Tat-Seng Chua, and Christian Theobalt. 2020. Neural sparse voxel fields. Advances in Neural Information Processing Systems 33 (2020), 15651–15663.

[16] Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, and Ren Ng. 2021. NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis. Commun. ACM 65, 1 (dec 2021), 99–106. https://doi.org/10.1145/3503250

[17] Jacob Munkberg, Jon Hasselgren, Tianchang Shen, Jun Gao, Wenzheng Chen, Alex Evans, Thomas Müller, and Sanja Fidler. 2022. Extracting triangular 3d models, materials, and lighting from images. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 8280–8290.

[18] Michael Niemeyer, Lars Mescheder, Michael Oechsle, and Andreas Geiger. 2020. Diferentiable Volumetric Rendering: Learning Implicit 3D Representations without 3D Supervision. In Proc. IEEE Conf. on Computer Vision and Pattern

Recognition (CVPR).

[19] Michael Oechsle, Songyou Peng, and Andreas Geiger. 2021. UNISURF: Unifying Neural Implicit Surfaces and Radiance Fields for Multi-View Reconstruction. In International Conference on Computer Vision (ICCV).

[20] Konstantinos Rematas, Andrew Liu, Pratul P Srinivasan, Jonathan T Barron, Andrea Tagliasacchi, Thomas Funkhouser, and Vittorio Ferrari. 2022. Urban radiance fields. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 12932–12942.

[21] Johannes L Schönberger, Enliang Zheng, Jan-Michael Frahm, and Marc Pollefeys. 2016. Pixelwise view selection for unstructured multi-view stereo. In Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11-14, 2016, Proceedings, Part III 14. Springer, 501–518.

[22] Matthew Tancik, Vincent Casser, Xinchen Yan, Sabeek Pradhan, Ben Mildenhall, Pratul P Srinivasan, Jonathan T Barron, and Henrik Kretzschmar. 2022. Block-nerf: Scalable large scene neural view synthesis. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition. 8248–8258.

[23] Haithem Turki, Deva Ramanan, and Mahadev Satyanarayanan. 2022. Mega-nerf: Scalable construction of large-scale nerfs for virtual fly-throughs. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 12922–12931.

[24] Ali Osman Ulusoy, Andreas Geiger, and Michael J Black. 2015. Towards probabilistic volumetric reconstruction using ray potentials. In 2015 International Conference on 3D Vision. IEEE, 10–18.

[25] Peng Wang, Lingjie Liu, Yuan Liu, Christian Theobalt, Taku Komura, and Wenping Wang. 2021. NeuS: Learning Neural Implicit Surfaces by Volume Rendering for Multi-view Reconstruction. NeurIPS (2021).

[26] Zongji Wang and Feng Lu. 2019. Voxsegnet: Volumetric cnns for semantic part segmentation of 3d shapes. IEEE transactions on visualization and computer graphics 26, 9 (2019), 2919–2930.

[27] Yi Wei, Shaohui Liu, Yongming Rao, Wang Zhao, Jiwen Lu, and Jie Zhou. 2021. Nerfingmvs: Guided optimization of neural radiance fields for indoor multi-view stereo. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision. 5610–5619.

[28] Yuanbo Xiangli, Linning Xu, Xingang Pan, Nanxuan Zhao, Anyi Rao, Christian Theobalt, Bo Dai, and Dahua Lin. 2022. BungeeNeRF: Progressive Neural Radiance Field for Extreme Multi-scale Scene Rendering. In The European Conference on Computer Vision (ECCV).

[29] Bangbang Yang, Yinda Zhang, Yinghao Xu, Yijin Li, Han Zhou, Hujun Bao, Guofeng Zhang, and Zhaopeng Cui. 2021. Learning object-compositional neural radiance field for editable scene rendering. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision. 13779–13788.

[30] Lior Yariv, Jiatao Gu, Yoni Kasten, and Yaron Lipman. 2021. Volume rendering of neural implicit surfaces. In NeurIPS.

[31] Lior Yariv, Peter Hedman, Christian Reiser, Dor Verbin, Pratul P Srinivasan, Richard Szeliski, Jonathan T Barron, and Ben Mildenhall. 2023. BakedSDF: Meshing Neural SDFs for Real-Time View Synthesis. arXiv preprint arXiv:2302.14859 (2023).

[32] Lior Yariv, Yoni Kasten, Dror Moran, Meirav Galun, Matan Atzmon, Basri Ronen, and Yaron Lipman. 2020. Multiview neural surface reconstruction by disentangling geometry and appearance. Advances in Neural Information Processing Systems 33 (2020), 2492–2502.

[33] Weicai Ye, Shuo Chen, Chong Bao, Hujun Bao, Marc Pollefeys, Zhaopeng Cui, and Guofeng Zhang. 2022. IntrinsicNeRF: Learning Intrinsic Neural Radiance Fields for Editable Novel View Synthesis. (2022).

[34] Alex Yu, Ruilong Li, Matthew Tancik, Hao Li, Ren Ng, and Angjoo Kanazawa. 2021. Plenoctrees for real-time rendering of neural radiance fields. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision. 5752–5761.

[35] Zehao Yu, Songyou Peng, Michael Niemeyer, Torsten Sattler, and Andreas Geiger. 2022. MonoSDF: Exploring Monocular Geometric Cues for Neural Implicit Surface Reconstruction. Advances in Neural Information Processing Systems (NeurIPS) (2022).

[36] Kai Zhang, Gernot Riegler, Noah Snavely, and Vladlen Koltun. 2020. NeRF++: Analyzing and Improving Neural Radiance Fields. arXiv:2010.07492 (2020).

[37] Shuaifeng Zhi, Tristan Laidlow, Stefan Leutenegger, and Andrew Davison. 2021. In-Place Scene Labelling and Understanding with Implicit Scene Representation. In Proceedings ofthe International Conference on Computer Vision (ICCV).