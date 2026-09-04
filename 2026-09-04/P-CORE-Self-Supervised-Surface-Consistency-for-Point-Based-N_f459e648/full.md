# P-CORE: Self-Supervised Surface Consistency for Point-Based Neural Editing

Yanshu Zhang<sup>1⋆</sup> , Shichong Peng<sup>1</sup> , Mehran Aghabozorgi<sup>1</sup> , Alireza Moazeni<sup>1</sup> , and Ke Li<sup>1,2,3</sup>

<sup>1</sup> Simon Fraser University, Canada

<sup>2</sup> Alberta Machine Intelligence Institute (Amii), Canada

3 Canadian Institute for Advanced Research (CIFAR), Canada {yanshu\_zhang, shichong\_peng, maa143, sam62, keli}@sfu.ca

Abstract. Advances in neural rendering have enabled high-fidelity multiview reconstruction of 3D scenes. However, free-form non-rigid shape editing remains a significant challenge. Point-based neural representations are highly desirable for multi-view reconstruction because they lack fixed connectivity, which does not constrain the learned surface topology to that of the initialization. Yet this same property causes point-based representations to struggle with holes and surface discontinuities under large deformations. To address this, we propose a novel self-supervised method to enable point-based representations to adapt to large deformations without requiring ground truth multi-view images of deformed geometry. The key idea is to generate random deformations and to ensure consistency in the predicted surface before and after deformation. In particular, the surface prediction from the deformed point cloud should be the same as the deformation applied to the surface prediction from the original point cloud. We incorporate our approach into attention-based point representations, which difer from splatting-based point representations in their use of a learned interpolation kernel between points as opposed to a Gaussian kernel around each point. This learned interpolation kernel can learn to adapt to large deformations, without requiring addition or removal of points. We show that our framework significantly enhances its robustness to large deformations. Experiments on synthetic geometry editing benchmarks (Neural Editor, Objaverse) demonstrate that our approach outperforms existing point-based methods in zero-shot editing and significantly reduces artifacts. Furthermore, qualitative results on the DTU and Mip-NeRF 360 datasets demonstrate our method’s efectiveness on real-world scenes. Project page: https://zvict.github.io/p-core/.

## 1 Introduction

Achieving free-form non-rigid editing of photo-realistic neural scenes is highly desirable for applications in animation, design, and gaming. Recent advances in neural rendering [42, 53, 56, 78, 86] have enabled high-fidelity multi-view reconstructions, but performing zero-shot deformation of these reconstructed scenes remains an open challenge. Point-based neural renderers have emerged as a highly desirable representation for reconstruction; by lacking fixed connectivity, they do not constrain the learned surface topology to that of the initialization, ofering exceptional flexibility in capturing high-fidelity details of complex structures.

However, when transitioning from static reconstruction to free-form non-rigid editing, this lack of connectivity becomes a double-edged sword. When the point cloud is deformed, the points drift apart because there are no explicit edges or polygons holding them together. Without explicit topological connections, point-based methods severely struggle with holes, see-through artifacts, and surface discontinuities under large deformations. To tackle these issues, existing approaches often rely on explicit spatial regularizations [29, 34, 68] or reintroduce proxy geometry such as cages [33,35,59,79] or meshes [26–28,37,51,71,72,80,83,89]. While these proxies can enforce structural integrity, they introduce a host of new limitations. Constructing an artifact-free proxy that tightly encloses the geometry without self-intersections is complex and often requires manual intervention. Furthermore, proxy-based methods are restricted to edits that the rigid topology can support, making part-level manipulations extremely challenging.

![](images/fed46d6bf49c39ca55148b05d310a07616a28101dd4a28ae978e7563ba3cb17e.jpg)  
Fig. 1: Overview. While point-based representations can learn a good surface prediction from multi-view images in its canonical space, they often struggle with holes and surface discontinuities during unseen deformations. To address this, we propose a self-supervised surface consistency framework. Given a point cloud representation learned in the canonical space, we predict the surface points from it. We then apply an identical deformation field to both the canonical point cloud and these extracted surface points. The resulting deformed surface points serve as geometric pseudo-ground truth to supervise the model’s new surface predictions on the deformed point cloud. A stop-gradient operation is applied to the target points to prevent model collapse.

To overcome these limitations without sacrificing the flexibility of point-based representations, we introduce a novel self-supervised surface consistency framework, as illustrated in Figure 1. Instead of relying on ground truth de formed meshes or manually crafted spatial regularizations, our pipeline generates random deformations and enforces a surface consistency constraint: the surface predicted from the deformed point cloud should equal the deformation applied to the surface predicted from the original point cloud. By generating such ge ometric “pseudo-ground truth” for the deformed state and jointly enforcing photometric anchoring on the original state, our method significantly improves the generalizability of attention-based point renderers to unseen edits, ensuring hole-free, smooth surface reconstruction and high-fidelity rendering under non-rigid deformations.

We deliberately apply this framework to Proximity Attention Point Rendering (PAPR) [86], an attention-based neural point renderer. Unlike 3D Gaussian Splatting (3DGS) [42], where discrete, anisotropic Gaussians must be coherently transformed (e.g., adjusting covariances) to avoid gaps and tears after deformation, PAPR treats each point as infinitesimal and fills gaps via a learned attentionbased interpolation kernel between nearby points. This design is advantageous for two reasons: first, it eliminates the need to transform spatial extents under deformation, inherently supporting continuous surface prediction; second, the learned interpolation kernel can adapt to large deformations without requiring addition or removal of points, ofering the capacity to generalize to unseen edits. We demonstrate that when coupled with our self-supervised framework, PAPR’s robustness to large deformations is significantly enhanced, preventing topological breakage even under drastic non-rigid edits.

## Contributions Below are our key contributions:

– We propose a novel self-supervised fine-tuning framework that substantially reduces holes and surface discontinuities in attention-based point renderers without requiring ground-truth dynamic data or geometry proxies.

– We demonstrate the successful coupling of this framework with an attentionbased neural point renderer (PAPR), highlighting its inherent advantages over spatial-extent-based methods (e.g., 3DGS) for continuous surface prediction during large deformations.

– We provide results on synthetic geometry editing benchmarks (Neural Editor, Objaverse) demonstrating that our method outperforms both proxy-based and proxy-free point-based methods.

## 2 Related Work

## 2.1 3D Shape Deformation

There is a long line of work in geometry processing on editing 3D shapes [24, 82]. Many surface-based methods use parametric patches or surface meshes as proxies to manipulate shapes [5,18,22,23,38,39,43,64,70], utilizing approaches like Laplacian [25, 50, 65–67] and cage-based [39, 81, 87] deformations. However, designing an appropriate proxy (e.g., mesh or cage) that closely fits the target model is challenging. Manual creation can be time-consuming and may require extensive expertise, while automatic generation methods might not always produce optimal proxies, especially for complex or highly detailed models.

To bypass proxy construction, traditional point-based deformation methods directly manipulate the shape using a set of freely positioned points or Moving Least Squares (MLS) handles [3, 6, 8–10, 12–14, 30, 31, 46, 54, 55, 60, 63, 91]. While ofering greater flexibility, directly applying these traditional point editing algorithms to modern neural point renderers introduces a critical failure mode. Traditional algorithms only relocate the discrete 3D coordinates. Because point clouds lack connectivity, deformations that stretch or bend the shape cause the points to spread apart.

## 2.2 Point-based Neural Rendering

Point-based representations [42,78,86] have gained significant attention in neural rendering. Some methods [4, 44, 57, 61] assume a given point cloud from Multi-View Stereo (MVS) or LiDAR, while others learn it from an initialization. Since point clouds natively lack connectivity, a fundamental challenge is how to fill the gaps between discrete points. One line of work focuses on predicting continuous surfaces from point clouds, employing classical techniques like Moving Least Squares (MLS) [2, 48] and Poisson Surface Reconstruction [40, 41], or more recent learning-based implicit surface prediction methods [20, 37, 47, 52]. Another line of work bypasses explicit meshing and directly renders the points. To fill the gaps during rendering, most methods associate a spatial extent with each point, either using 2D splats [45, 62, 76, 84, 90] or 3D Gaussian splats [42]. While these spatial extents enable high-quality rendering, they introduce parameters (e.g., covariances) that are dificult to correctly adjust during deformation. Instead, methods like PAPR [86] and Pointersect [15] represent points without spatial extents, using an attention mechanism [69] to interpolate between nearby points and fill gaps.

## 2.3 Editing of Neural Scene Representations

To edit neural scene representations while preserving surface continuity, one family of approaches [27,35,59,72,79,80,83,89] extracts and edits a proxy geometry (mesh or cage), then maps the edits back to the neural scene. However, this conversion introduces approximation errors and additional points of failure, because some geometric features easily represented implicitly cannot be represented in a proxy, and vice versa. Moreover, the edits are limited to what the rigid proxy connectivity supports—part-level editing (e.g., spinning wheels) becomes severely challenging.

Another line of work [11,21,34,77,88] trains neural representations on dynamic scene observations to learn shape variations over time, associating each shape with keypoints so that the shape changes when the keypoints are edited. However, these methods fail to generalize to out-of-distribution shapes and edits not observed in the dynamic scenes.

![](images/8c06e758c79417e73272b5c86eb83bbe0cd4a35b1eff71d04fc9487714021a6b.jpg)  
Fig. 2: As shown on the left, in 3DGS editing the points requires adjusting each Gaussian splat’s covariance to eliminate gaps. However, this adjustment may introduce distortions, as circled. Instead, attention-based methods model only the center of each point and learn a renderer that interpolates between points using an attention mechanism. This interpolation is based on the local point configurations, for example, the relative distances between points, with many variations observed from diferent rays during training. This variability enables attention-based methods to efectively generalize to unseen point configurations and find a promising interpolation of the points after editing, as shown on the right.

More recently, a line of work [27, 34, 49, 73–75] edits scenes by directly manipulating neural point primitives. RIP-NeRF [73] and RISE-Editing [74] learn rotation-invariant point features for fine-grained editing and cross-scene compositing. SPIDR [49] couples a neural point field with an SDF to support geometry deformation and relighting. IRIS [75] attaches latent features to explicit Gaussian anchors for eficient part-level editing. The key distinction from our work lies in the rendering paradigm. First, all these methods use volume rendering, whereas P-CORE builds on a surface rendering point representation that produces a single attention-weighted ray–surface intersection (PAPR). For the editing task, volume rendering must populate the object interior with primitives, making it far less compact than a surface representation and requiring every edit to displace the interior primitives consistently with the surface. Second, 3DGS-based methods [27, 34, 75] represent the scene with Gaussian primitives with spatial extent, so deformation must transform each primitive’s covariance, which may cause distortion and surface tearing as shown in Fig. 2.

In contrast to these approaches, P-CORE operates directly on points, requiring neither a proxy geometry nor dynamic-scene observations. By rendering only the surface with extent-free points, P-CORE represents an object compactly and requires no interior or primitive extent transformation when editing.

## 3 Preliminaries

PAPR learns a point-based representation of a 3D scene from multi-view RGB images and corresponding camera parameters. The point-based representation ${ \mathcal { P } } = \{ ( { \bf p } _ { i } , \mu _ { i } ) \}$ , where $i \in \{ 1 , 2 , \ldots , N \}$ represents a 3D scene with N neural points. Each neural point has a learnable 3D position $ { \mathbf { p } } \in \mathbb { R } ^ { 3 }$ and a learnable feature vector $\mu \in \mathbb { R } ^ { d }$ . Given a ray $\mathbf { r } _ { j }$ represented by a camera center ${ \mathbf o } _ { j } \in \mathbb { R } ^ { 3 }$ and a ray direction $\mathbf { d } _ { j } \in \mathbb { R } ^ { 3 }$ as the query, PAPR learns $K \ll N$ attention weights for each of the $K$ nearest points around the ray. It then uses the attention weights to aggregate the value embedding vectors for the $K$ points to get a feature vector $\mathbf { f } _ { j }$ that captures the color of the ray. To map the query and key into the same feature space, PAPR uses MLPs to learn embedding vectors:

$$
\mathbf { q } _ { j } = f _ { \theta _ { Q } } \left( \gamma \left( \mathbf { d } _ { j } \right) \right) , \mathbf { k } _ { i j } = f _ { \theta _ { K } } \left( \left[ \gamma \left( \mathbf { h } _ { i , j } \right) , \gamma \left( \mathbf { t } _ { i , j } \right) , \gamma \left( \mathbf { p } _ { i } \right) \right] \right) ,\tag{1}
$$

where $f _ { \theta }$ are the embedding $\mathrm { M L P s } , \gamma$ is the positional encoding function. $\mathbf { h } _ { i j }$ and $\mathbf { t } _ { i j }$ are two ray-dependent point feature vectors, computed as:

$$
\mathbf { p } _ { i j } ^ { \prime } = \mathbf { o } _ { j } + \langle \mathbf { p } _ { i } - \mathbf { o } _ { j } , \mathbf { d } _ { j } \rangle \cdot \mathbf { d } _ { j } , ~ \mathbf { h } _ { i j } = \mathbf { p } _ { i j } ^ { \prime } - \mathbf { o } _ { j } , ~ \mathbf { t } _ { i j } = \mathbf { p } _ { i } - \mathbf { p } _ { i j } ^ { \prime } ,\tag{2}
$$

where $\mathbf { p } _ { i j } ^ { \prime }$ is the position of the point’s projection on ray $\mathbf { r } _ { j }$ . These features capture the point configuration of the neighborhood by incorporating the features from all K points around a ray. To model ray-dependent appearance, PAPR learns a ray-dependent feature vector $\mathbf { v } _ { i j }$ for each point:

$$
\mathbf { v } _ { i j } = f _ { \theta _ { V } } \left( \left[ \gamma \left( \mathbf { h } _ { i , j } \right) , \gamma \left( \mathbf { t } _ { i , j } \right) , \gamma \left( \mu _ { i } \right) \right] \right) ,\tag{3}
$$

The ray’s feature $\mathbf { f } _ { j }$ can then be computed by $\begin{array} { r } { \mathbf { f } _ { j } = \sum _ { i = 1 } ^ { K } w _ { i j } \mathbf { v } _ { i j } } \end{array}$ , where $w _ { i j } =$ softmax $( \langle \mathbf { q } _ { j } , \mathbf { k } _ { i j } \rangle / \sqrt { d _ { \mathbf { k } } } )$ are the attention weights, $d _ { \mathbf { k } }$ is the dimension of $\mathbf { k } _ { i j } . \mathrm { B y }$ spatially aggregating the feature vectors of all the rays from the same camera view, PAPR produces a feature map of the view, which is then passed through a UNet-based renderer to predict RGB image <sup>ˆ</sup>I.

## 4 Method

## 4.1 3D Gaussian Splatting vs. PAPR

Compared to 3DGS, a key diference is that in PAPR, points have no spatial extent. Rather than using splats to fill gaps between points, PAPR’s attention mechanism outputs a set of weights over the K nearest points for every ray. As attention weights are non-negative and sum up to 1, it essentially learns a convex interpolation of those points. Such interpolations are based on the local point configurations around the rays, which vary significantly across diferent rays observed during training. This variability enables PAPR to generalize efectively to unseen point sets after deformation.

With 3DGS, it is crucial to adjust the covariance of each afected Gaussian when editing, otherwise it may cause surface discontinuities. As shown in Fig. 2, fixing covariances can result in gaps between splats after editing. Moreover, even with adjusted covariances, it’s challenging to get them right, as adjustments vary between splats. In contrast, PAPR lacks spatial extent around points, so no spatial adjustment is needed—only the point locations must be changed.

Furthermore, even if the covariances were adjusted optimally, as shown in the left of Fig. 2, the shape surface after editing can still be not as smooth as it was originally. Neither shrinking nor rotating the splats would work since the former causes gaps, while the latter makes splats stick out. This is caused by the fact that non-rigid deformations of Gaussians are not necessarily Gaussian anymore. As a result, more Gaussian splats would need to be added at locations where the non-rigidity is the greatest. In contrast, since non-rigid deformations of a point set simply produce another point set, in PAPR, it sufices to move the points without adding new ones. The interpolator learned by PAPR should adapt to the changes in point positions, providing smooth and reliable interpolation for the updated point set, thereby preserving surface continuity.

## 4.2 Canonical Initialization

Our framework begins by capturing the canonical geometry and appearance of the target scene. We pre-train the PAPR model on a set of static multi-view images to extract a learned dense point cloud representation P. This canonical initialization establishes a high-fidelity reference state, provides accurate implicit surface intersections and their associated photometric properties before any non-rigid edits are applied.

## 4.3 Surface Point Extraction

A key component of our framework is the extraction of diferentiable surface intersection estimates from PAPR’s attention mechanism. Given a camera ray $\mathbf { r } _ { j } = ( \mathbf { o } _ { j } , \mathbf { d } _ { j } )$ and its associated top-K selected points $\{ \mathbf { p } _ { k } \} _ { k = 1 } ^ { K }$ with attention weights $\{ w _ { k j } \}$ , we define the surface intersection point for ray $\mathbf { r } _ { j }$ as the attentionweighted average of the selected point positions:

$$
\hat { \mathbf { s } } _ { j } = \sum _ { k = 1 } ^ { K } w _ { k j } \cdot \mathbf { p } _ { k }\tag{4}
$$

As the model is only trained on the observations at the canonical state of the scene, the attention model often fails to generalize to unseen edits, resulting in holes in the surface prediction and artifacts in the renderings under deformation.

## 4.4 Self-Supervised Fine-Tuning Objective

To ensure consistent surface prediction under deformation, we need to enhance the robustness of PAPR’s attention model to unseen edits. A naïve solution is to finetune the learned PAPR model on the captures of the deformed scene, for example, in the case of dynamic scene reconstruction. However, such ground truth data are not often available. To bypass the need for ground-truth dynamic sequences, we introduce a self-supervised fine-tuning objective that utilizes analytical 3D deformation fields and creates “pseudo” supervision in the fine-tuning stage.

At each fine-tuning step, we first perform a standard forward pass on the canonical point cloud $\mathcal { P }$ with a batch of training rays, extracting both rendered

RGB values and canonical surface points ˆs from the attention weights for each ray. We then apply an analytically defined 3D deformation field $\mathcal { D } : \mathbb { R } ^ { 3 }  \mathbb { R } ^ { 3 }$ to the point cloud, yielding the deformed point cloud $\mathcal { P } _ { \mathrm { d e f } } = \mathcal { D } ( \mathcal { P } )$

We then re-query the model under the deformed point cloud with deformed rays (Sec. 4.5) to extract predicted deformed surface points $\hat { \bf s } _ { \mathrm { p r e d } }$ , and compute a geometric loss between $\hat { \bf s } _ { \mathrm { p r e d } }$ , and the deformed canonical surface points $\mathcal { D } ( \hat { \mathbf { s } } _ { j } )$ , using the identical deformation field D:

$$
\mathcal { L } _ { \mathrm { g e o } } = \frac { 1 } { | \mathcal { F } | } \sum _ { j \in \mathcal { F } } \| \hat { \mathbf { s } } _ { \mathrm { p r e d } , j } - \mathcal { D } ( \mathrm { s g } [ \hat { \mathbf { s } } _ { j } ] ) \| ^ { 2 }\tag{5}
$$

where $\mathcal { F }$ denotes the set of ray indices and $\mathrm { s g } [ \cdot ]$ is the stop-gradient operator, which detaches the canonical surface points ˆs from the computation graph before applying the deformation $\mathcal { D } .$ This prevents gradients from flowing back through the canonical extraction and destabilizing the already-converged representation, ensuring only the deformed-state predictions $\hat { \bf s } _ { \mathrm { p r e d } }$ are optimized. Additionally, we enforce photometric consistency on the deformed rendering through a texture loss $\mathcal { L } _ { \mathrm { t e x } } = \Vert \hat { I } _ { \mathrm { d e f } } - \hat { I } _ { \mathrm { c a n } } \Vert ^ { 2 }$ , which anchors the appearance of the deformed state to the canonical rendering without requiring ground-truth deformed images.

## 4.5 Deformation Modes

We introduce two complementary deformation modes that determine how camera rays interact with the deformed geometry, each addressing a distinct failure mode:

Points-only mode. The point cloud and surface points are deformed, but the camera ray origins remain at their original positions. Foreground ray directions are recomputed to point toward the deformed ray termination points. This mode preserves the original camera viewpoint structure, ensuring robust rendering quality from distant views where the deformation is observed globally.

Rays-and-points mode. In addition to deforming the point cloud and surface points, the ray origins are repositioned near the surface and then deformed by the same field. Ray directions are recomputed from the deformed origins toward the deformed termination points. This mode addresses the occlusion problem: when the point cloud deforms, the original ray directions may no longer correctly query the deformed surface due to self-occlusion from displaced geometry. By diversifying the set of ray directions the model encounters during training, this mode significantly enhances robustness to complex deformations.

In practice, we use both modes simultaneously.

## 4.6 Joint Loss Formulation

The total training objective combines geometric alignment on the deformed state with appearance anchoring on the canonical state:

$$
{ \mathcal { L } } _ { \mathrm { t o t a l } } = { \mathcal { L } } _ { \mathrm { r g b } } + \lambda _ { \mathrm { g e o } } \cdot { \mathcal { L } } _ { \mathrm { g e o } } + \lambda _ { \mathrm { t e x } } \cdot { \mathcal { L } } _ { \mathrm { t e x } }\tag{6}
$$

where $\mathcal { L } _ { \mathrm { r g b } }$ is the canonical photometric rendering loss used in PAPR [86], $\mathcal { L } _ { \mathrm { g e o } }$ is the self-supervised geometric loss $\left( \operatorname { E q . 5 } \right)$ , and $\mathcal { L } _ { \mathrm { t e x } }$ is the texture consistency loss, all defined in Sec. 4.4.

## 5 Experiments

## 5.1 Datasets and Baselines

To evaluate the structural resilience and rendering fidelity of our self-supervised framework, we test against both proxy-based and proxy-free point editing methods on the Neural Editor dataset [16] and dynamic scene subsets from Objaverse [19] introduced by PAPR-in-Motion [58], a recent method that extends PAPR to synthesize smooth point-level 3D scene interpolations between diferent scene states without intermediate supervision. For these synthetic datasets, ground truth deformed states are available, enabling both qualitative and quantitative benchmark comparisons. We follow the scene selection and evaluation protocols from [16] for Neural Editor and PAPR-in-Motion [58] for Objaverse, evaluating on scenes containing complex non-rigid transformations and part-level motions. Additionally, we qualitatively evaluate our method on real-world scenes from the DTU and Mip-NeRF 360 datasets as we lack ground truth deformation fields and renderings of the deformed scenes for these real-world datasets. We utilize a variety of analytical 3D deformation fields during our self-supervised training across all datasets, such as global twisting, scaling, dilating, and bending along random directions and varying degrees.

For proxy-based baselines, we compare with Deforming-NeRF [79], Neural Editor [16] (cage-based), and Mani-GS [27] (mesh-based). For proxy-free baselines, we compare with SC-GS [34] and the vanilla PAPR [86] model to explicitly demonstrate the robustness provided by our framework. SC-GS and Mani-GS represent two common ways to deform the covariance of the Gaussians (which must be deformed, otherwise Gaussians will easily extrude out of the surface); SC-GS uses the ARAP geometry regularizer, and Mani-GS uses a mesh as proxy geometry. In contrast, our method inherently treats each point as infinitesimal, so we can directly update the point positions without worrying about the shape of the points. For a fair comparison we apply identical deformation fields across all methods and evaluate the resulting visual and structural fidelity. In our method, the fine-tuning requires only ∼500 steps (< 1min)—roughly 1/500 of the pre-training duration—using the same learning rates and batch size, making it a lightweight post-processing step.

## 5.2 Qualitative Results

Fig. 3 and Fig. 5 show the qualitative comparison between P-CORE and the baselines on the Neural Editor and Objaverse datasets. To align the deformations across diferent methods for fairness, we first manually deform the cage from Deforming-NeRF, and then use the cage to deform the mesh or the control points of other methods through cage-based deformation.

![](images/0c2d818df9479330367cafc43c4a30071e74b79c8bb242c603426e970709e301.jpg)  
Fig. 3: Qualitative comparison on the Neural Editor dataset.

As shown in the figures, proxy-based approaches like Mani-GS often struggle with part-level manipulations and complex topological changes. Because they are bound by a rigid mesh, they fail to separate distinct parts or introduce blocky artifacts when the underlying topology is severely stretched. In contrast, proxy-free methods like SC-GS do not rely on geometry proxies, making them more adaptable to various types of deformations. However, SC-GS severely fails to preserve surface continuity after editing, leading to pervasive holes and “extruded Gaussians” around the boundaries of the edited shapes. Furthermore, while the base PAPR model handles minor interpolations well, it still exhibits noticeable tearing and structural collapse under large non-rigid edits.

In contrast, our P-CORE framework enhances structural robustness. It signifi cantly reduces holes, tearing, and other artifacts, consistently preserving perfectly continuous surfaces across all scenes even under severe deformation.

Additionally, we demonstrate the versatility of our method on complex realworld captures. As shown in Fig. 4, P-CORE extends robustly to the DTU and Mip-NeRF 360 datasets, supporting multiple sequential edits without degrading texture quality or introducing surface tearing.

![](images/8fdc2726ec298c6ae98a23945cfd0c2bac15865ead0a12ee988a187e97001096.jpg)

![](images/e7c1f8ae557a1308128ce212fdca43a8fd4afcc9ceb14635a21d108ca624ba53.jpg)  
(b) Mip-NeRF 360 dataset

Fig. 4: Editing results on real-world scenes from the DTU and Mip-NeRF 360 datasets.  
Table 1: Average quantitative performance on the Neural Editor and Objaverse datasets across all scenes. Bold indicates the best and underline indicates the second best.
<table><tr><td rowspan="2">Method</td><td colspan="4">Neural Editor [16]</td><td colspan="4">Objaverse [19]</td></tr><tr><td>PSNR ↑</td><td>SSIM ↑</td><td>LPIPS ↓</td><td># Primitives</td><td>PSNR ↑</td><td>SSIM ↑</td><td>LPIPS ↓</td><td># Primitives</td></tr><tr><td>Deforming-NeRF [79]</td><td>14.71</td><td>0.722</td><td>0.197</td><td>一</td><td>15.89</td><td>0.773</td><td>0.180</td><td></td></tr><tr><td>Neural Editor [16]</td><td>25.62</td><td>0.939</td><td>0.078</td><td>~1M</td><td>28.02</td><td>0.956</td><td>0.048</td><td>~1M</td></tr><tr><td>SC-GS [34]</td><td>18.55</td><td>0.832</td><td>0.123</td><td>~236K</td><td>25.29</td><td>0.929</td><td>0.057</td><td>~228K</td></tr><tr><td>Mani-GS [27]</td><td>20.59</td><td>0.875</td><td>0.116</td><td>~1.2M</td><td>26.04</td><td>0.946</td><td>0.069</td><td>~362K</td></tr><tr><td>PAPR [86]</td><td>24.22</td><td>0.930</td><td>0.066</td><td>30K</td><td>23.15</td><td>0.940</td><td>0.058</td><td>30K</td></tr><tr><td>Ours</td><td>25.31</td><td>0.942</td><td>0.054</td><td>30K</td><td>28.30</td><td>0.965</td><td>0.039</td><td>30K</td></tr></table>

## 5.3 Quantitative Results

To quantitatively evaluate our zero-shot editing performance, we compare our method against all baselines on scenes from the Neural Editor [16] and Objaverse [19] datasets where ground truth deformed states are available. By applying identical deformation fields to the points or proxies, we ensure a fair comparison across all methods.

Table 1 summarizes the average quantitative performance. P-CORE achieves state-of-the-art rendering quality across all metrics (PSNR, SSIM, and LPIPS) on the Objaverse dataset, and either matches or outperforms the baselines on the Neural Editor dataset, with a much smaller number of primitives (30K vs. 1M). Although Neural Editor attains strong metrics, it relies on ∼1M points and is far more expensive, requiring ∼7 days of training per scene versus ∼1 minute of fine-tuning for our method; we provide side-by-side qualitative comparisons with Neural Editor in the supplementary material.

## 5.4 User Study

To directly assess the geometric artifacts that pixel-level metrics (PSNR/SSIM/LPIPS) cannot fully capture, we conduct a two-alternative forced-choice (2AFC) user study. We collected 525 responses from 25 participants, who compared pairs of deformed renderings across all scenes against the three strongest baselines (PAPR [86], Mani-GS [27], and SC-GS [34]) and were asked to select the result

![](images/f1fe30ca38dbf4b1c9316d0eb2643d6b5cca54d62ebec0e6f35f3fc7a55dd24e.jpg)  
Fig. 5: Qualitative comparison on scenes from Objaverse.

with fewer geometric artifacts such as holes, tearing, and broken topology. As shown in Table 2, our method was preferred in 94.7% of trials overall and in > 89% of every baseline–dataset pair, confirming that P-CORE is consistently judged to produce fewer geometric artifacts than all baselines.

## 5.5 Ablation Study

To evaluate the contribution of each component in our self-supervised fine-tuning framework, we conduct an ablation study on the Lego scene under a 2× uniform scaling deformation. Fig. 6 provides a qualitative comparison of the rendered novel views and depth maps across diferent configurations.

Table 2: User-study win rate of our method (%, with raw counts). On each trial, participants selected the deformed rendering with fewer geometric artifacts (holes, tearing, broken topology).
<table><tr><td>Pairing</td><td>Neural Editor</td><td>Objaverse</td><td>All</td></tr><tr><td>Ours vs. PAPR</td><td>90.2% (83/92)</td><td>95.3% (82/86)</td><td>92.7% (165/178)</td></tr><tr><td>Ours vs. Mani-GS</td><td>98.0% (100/102)</td><td>94.9% (75/79)</td><td>96.7% (175/181)</td></tr><tr><td>Ours vs. SC-GS</td><td>98.9% (88/89)</td><td>89.6% (69/77)</td><td>94.6% (157/166)</td></tr><tr><td>Overall</td><td>95.8% (271/283)</td><td>93.4% (226/242)</td><td>94.7% (497/525)</td></tr></table>

![](images/701b50d443e3f99cba069d0d610251e3bf6967d9bd4d8884552c6f341bc60191.jpg)  
Fig. 6: Ablation study on the Lego scene under 2× scaling. From left to right: canonical (undeformed), vanilla PAPR, without rays-and-points (R&P) mode, without points-only mode, without texture loss $\mathcal { L } _ { \mathrm { t e x } } ,$ , and our full pipeline.

Vanilla PAPR (without fine-tuning) exhibits obvious holes in both the novel views and depth maps, confirming the base representation’s inability to maintain surface continuity under large deformations. Without the rays-and-points (R&P) mode, many holes still remain in both the depth and rendering, as the ray directions are not diverse enough for the attention model to properly generalize to the deformed geometry. Without the points-only mode, the depth around the boundaries of the object becomes inaccurate, and there are visible artifacts around the boundaries in the rendering as well, since the fine-tuning overfits to the close viewpoints sampled in the rays-and-points mode. Without the texture consistency loss $\left( \mathcal { L } _ { \mathrm { t e x } } \right)$ , the depth map has far fewer holes, but the rendering is noticeably blurred. Our full method with all components produces a hole-free depth map and a high-fidelity rendering with preserved details, demonstrating that all proposed components are essential for robust deformation.

## 5.6 Application: Skinning Weights Optimization

A downstream application of our method is to optimize the skinning weights of the parameterized scenes without a ground truth motion sequence.

Specifically, we extract M(= 128) sparse control anchors $\{ \mathbf { a } _ { m } \} _ { m = 1 } ^ { M }$ from the canonical dense point cloud using Farthest Point Sampling (FPS). We parameterize the dense point cloud using a Neural Blend Skinning (NBS) approach, where an anchor skinning MLP $f _ { \phi }$ with positional encoding takes a normalized 3D point coordinate and outputs logits over all M anchors: $\ell ( \mathbf { p } ) = f _ { \phi } ( \gamma ( \mathbf { p } ) ) \in \mathbb { R } ^ { M }$ . For each point p to be deformed, we find its $K _ { \mathrm { n n } } ( = 8 )$ nearest anchors by Euclidean distance, gather the corresponding logits, and apply softmax to obtain sparse skinning weights: $w _ { k } = \operatorname { s o f t m a x } ( \ell _ { m _ { k } } ( { \bf p } ) )$

To derive per-anchor rigid transformations from an analytical deformation ${ \mathcal { D } } ,$ we estimate the Jacobian $\mathbf { J } _ { m } = \nabla \mathcal { D } ( \mathbf { a } _ { m } )$ via finite diferences, extract the closest rotation matrix $\mathbf { R } _ { m }$ via SVD projection, and recover the translation $\mathbf { t } _ { m } = { \mathcal { D } } ( \mathbf { a } _ { m } ) - \mathbf { R } _ { m } \mathbf { a } _ { m }$ . The deformed position is then predicted via Linear Blend Skinning:

$$
\mathbf { p } ^ { \prime } = \sum _ { k = 1 } ^ { K _ { \mathrm { n n } } } w _ { k } \cdot ( \mathbf { R } _ { m _ { k } } \mathbf { p } + \mathbf { t } _ { m _ { k } } )\tag{7}
$$

This NBS parameterization replaces the direct analytical deformation for the point cloud and surface points within our self-supervised framework, while camera rays are still deformed analytically. As the pipeline is end-to-end diferentiable, during the self-supervised fine-tuning, we jointly optimize these skinning MLP weights alongside the neural renderer. At inference time, users perform edits by simply repositioning the sparse anchor points.

Table 3 compares our optimized NBS weights against distance-based Gaussiankernel RBF weights [34], where $\hat { w } _ { j k } = \exp ( - d _ { j k } ^ { 2 } / \bar { 2 } \sigma ^ { 2 } )$ with $\sigma = 1$ , normalized to sum to one. For the geometry evaluation, we compare the deformed full point cloud derived from the deformed anchor points with the ground truth deformed mesh. For a fair comparison we use the same set of anchor points for both methods. As shown, our NBS consistently outperforms the RBF baseline across all rendering and geometry metrics, confirming that self-supervised deformation augmentation yields spatially discriminative skinning weights that generalize and better preserve geometry detail during editing. Fig. 7 shows qualitative editing results on two Objaverse scenes.

Table 3: Ablation on skinning weight estimation, averaged over 6 Objaverse scenes. Both methods use the same FPS anchors; the dense point cloud is deformed via anchordriven RBF interpolation and compared against ground-truth end-state views and meshes (normalized to $[ - 1 , 1 ] ^ { 3 } )$ .
<table><tr><td rowspan="2">Skinning Weights</td><td colspan="3">Rendering Quality</td><td colspan="3">Geometry Quality</td></tr><tr><td>PSNR ↑</td><td>SSIM ↑</td><td>LPIPS X</td><td>CD↓ EMD ↓</td><td> $\mathrm { F _ { 0 . 0 1 } }$ </td><td>↑ F0.02 ↑</td></tr><tr><td>RBF [34]</td><td>25.32</td><td>0.945</td><td>0.054</td><td>0.077</td><td>0.307 0.409</td><td>0.626</td></tr><tr><td>NBS (Ours)</td><td>26.46</td><td>0.952</td><td>0.050</td><td>0.077</td><td>0.303 0.438</td><td>0.640</td></tr></table>

![](images/933f52bc598890f8108a6dc75aaeff4003ee57dd1ee1b69bb2f96a71bbbd86e3.jpg)  
Fig. 7: Qualitative comparison of proxy-free sparse control editing on the Lego and Butterfly scenes. Without self-supervised augmentation, the RBF-deformed renderings exhibit unfaithful deformation and surface discontinuities. With our self-supervised fine-tuning, the edited geometry is better aligned with the ground-truth target.

## 6 Conclusion and Future Work

In this paper, we presented a self-supervised surface-consistency framework that addresses a fundamental tension in point-based neural rendering: the absence of fixed connectivity provides flexibility for reconstruction, but can also lead to holes and surface discontinuities under large deformations. Our approach generates random deformations and enforces that surface prediction commutes with deformation, eliminating the need for ground-truth dynamic data or explicit geometry proxies. We integrated this framework into PAPR, an attention-based neural point renderer whose learned interpolation kernel can adapt to large deformations without adding or removing points. Experiments on synthetic benchmarks and real-world scenes demonstrate the efectiveness of our approach, and we further show its applicability to downstream tasks such as Neural Blend Skinning (NBS). A limitation of the current method is that excessive fine-tuning can lead to over-smoothed or blurry renderings, suggesting a trade-of between deformation consistency and appearance fidelity. Future work could explore improved regularization or adaptive stopping criteria, as well as integration with physically based simulation, more sophisticated multi-level editing controls, and extensions to other point-based architectures.

## 7 Acknowledgments

This research was enabled in part by support provided by NSERC, the BC DRI Group, the Digital Research Alliance of Canada and the Canada CIFAR AI Chairs Program.

## References

1. Adobe Mixamo. https://www.mixamo.com (2026)

2. Alexa, M., Behr, J., Cohen-Or, D., Fleishman, S., Levin, D., Silva, C.T.: Computing and rendering point set surfaces. IEEE Trans. Vis. Comput. Graph. 9, 3–15 (2003)

3. Alexa, M., Cohen-Or, D., Levin, D.: As-rigid-as-possible shape interpolation. Proceedings of the 27th annual conference on Computer graphics and interactive techniques (2000)

4. Aliev, K.A., Ulyanov, D., Lempitsky, V.S.: Neural point-based graphics. In: European Conference on Computer Vision (2019)

5. Angelidis, A., Cani, M.P., Wyvill, G., King, S.: Swirling-sweepers: constant volume modeling. In: ACM SIGGRAPH 2004 Sketches, p. 40 (2004)

6. Aubert, F., Bechmann, D.: Volume-preserving space deformation. Computers & Graphics 21(5), 625–639 (1997)

7. Barron, J.T., Mildenhall, B., Verbin, D., Srinivasan, P.P., Hedman, P.: Mip-nerf 360: Unbounded anti-aliased neural radiance fields. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 5470–5479 (2022)

8. Bechmann, D., Dubreuil, N.: Animation through space and time based on a space deformation model. The Journal of Visualization and Computer Animation 4(3), 165–184 (1993)

9. Bechmann, D., Dubreuil, N.: Order-controlled free-form animation. The Journal of Visualization and computer animation 6(1), 11–32 (1995)

10. Bechmann, D., Gerber, D.: Arbitrary shaped deformations with dogme. The Visual Computer 19(2), 175–186 (2003)

11. Bian, W., Huang, Z., Shi, X., Li, Y., Wang, F.Y., Li, H.: Gs-dit: Advancing video generation with pseudo 4d gaussian fields through eficient dense 3d point tracking. ArXiv abs/2501.02690 (2025)

12. Borrel, P., Bechmann, D.: Deformation of n-dimensional objects. In: Proceedings of the first ACM symposium on Solid modeling foundations and CAD/CAM applications. pp. 351–369 (1991)

13. Borrel, P., Rappoport, A.: Simple constrained deformations for geometric modeling and interactive design. ACM Transactions on Graphics (TOG) 13(2), 137–155 (1994)

14. Botsch, M., Kobbelt, L.: Real-time shape editing using radial basis functions. Computer Graphics Forum 24 (2005)

15. Chang, J.H.R., Chen, W.Y., Ranjan, A., Yi, K.M., Tuzel, O.: Pointersect: Neural rendering with cloud-ray intersection. 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) pp. 8359–8369 (2023)

16. Chen, J.K., Lyu, J., Wang, Y.X.: Neuraleditor: Editing neural radiance fields via manipulating point clouds. 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) pp. 12439–12448 (2023)

17. Chen, Y., Chen, Z., Zhang, C., Wang, F., Yang, X., Wang, Y., Cai, Z., Yang, L., Liu, H., Lin, G.: GaussianEditor: Swift and controllable 3d editing with gaussian splatting. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 21476–21485 (2024)

18. Decaudin, P.: Geometric deformation by merging a 3d-object with a simple shape. In: Graphics Interface. vol. 96, pp. 55–60 (1996)

19. Deitke, M., Schwenk, D., Salvador, J., Weihs, L., Michel, O., VanderBilt, E., Schmidt, L., Ehsani, K., Kembhavi, A., Farhadi, A.: Objaverse: A universe of annotated 3d objects. 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) pp. 13142–13153 (2022)

20. Erler, P., Guerrero, P., Ohrhallinger, S., Mitra, N.J., Wimmer, M.: Points2surf learning implicit surfaces from point clouds. In: European Conference on Computer Vision (2020)

21. Feng, H., Zhang, J., Wang, Q., Ye, Y., Yu, P., Black, M.J., Darrell, T., Kanazawa, A.: St4rtrack: Simultaneous 4d reconstruction and tracking in the world. ArXiv abs/2504.13152 (2025)

22. Feng, J., Ma, L., Peng, Q.: A new free-form deformation through the control of parametric surfaces. Computers & Graphics 20(4), 531–539 (1996)

23. Feng, J., Shao, J., Jin, X., Peng, Q., Forrest, A.R.: Multiresolution free-form deformation with subdivision surface of arbitrary topology. The Visual Computer 22, 28–42 (2006)

24. Gain, J.E., Bechmann, D.: A survey of spatial deformation from a user-centered perspective. ACM Trans. Graph. 27, 107:1–107:21 (2008)

25. Gao, L., Lai, Y.K., Yang, J., Zhang, L.X., Xia, S., Kobbelt, L.: Sparse data driven mesh deformation. IEEE transactions on visualization and computer graphics 27(3), 2085–2100 (2019)

26. Gao, L., Yang, J., Zhang, B.T., Sun, J., Yuan, Y.J., Fu, H., Lai, Y.K.: Mesh-based gaussian splatting for real-time large-scale deformation. ArXiv abs/2402.04796 (2024)

27. Gao, X., Li, X., Zhuang, Y., Zhang, Q., Hu, W., Zhang, C., Yao, Y., Shan, Y., Quan, L.: Mani-gs: Gaussian splatting manipulation with triangular mesh. ArXiv abs/2405.17811 (2024)

28. Gu’edon, A., Lepetit, V.: Gaussian frosting: Editable complex radiance fields with real-time rendering. ArXiv abs/2403.14554 (2024)

29. Han, X., Tian, R., Tong, Y., Yu, F., Liu, D., Zhang, Y.: Arap-gs: Drag-driven as-rigid-as-possible 3d gaussian splatting editing with difusion prior (2025), https: //arxiv.org/abs/2504.12788

30. Hsu, W.M., Hughes, J.F., Kaufman, H.: Direct manipulation of free-form deformations. ACM Siggraph Computer Graphics 26(2), 177–184 (1992)

31. Hu, S.M., Zhang, H., Tai, C.L., Sun, J.G.: Direct manipulation of fd: eficient explicit solutions and decomposible multiple point constraints. Visual Computer 17(6), 370–379 (2001)

32. Huang, B., Yu, Z., Chen, A., Geiger, A., Gao, S.: 2d gaussian splatting for geometrically accurate radiance fields. ACM SIGGRAPH 2024 Conference Papers (2024)

33. Huang, J., Yu, H.: Gsdeformer: Direct cage-based deformation for 3d gaussian splatting. ArXiv abs/2405.15491 (2024)

34. Huang, Y.H., tian Sun, Y., Yang, Z., Lyu, X., Cao, Y.P., Qi, X.: Sc-gs: Sparsecontrolled gaussian splatting for editable dynamic scenes. ArXiv abs/2312.14937 (2023)

35. Jambon, C., Kerbl, B., Kopanas, G., Diolatzis, S., Leimkühler, T., Drettakis, G.: Nerfshop. Proceedings of the ACM on Computer Graphics and Interactive Techniques 6, 1 – 21 (2023)

36. Jensen, R.R., Dahl, A., Vogiatzis, G., Tola, E., Aanæs, H.: Large scale multi-view stereopsis evaluation. 2014 IEEE Conference on Computer Vision and Pattern Recognition pp. 406–413 (2014)

37. Jiang, Y., Yu, C., Xie, T., Li, X., Feng, Y., Wang, H., Li, M., Lau, H., Gao, F., Yang, Y., Jiang, C.: Vr-gs: A physical dynamics-aware interactive gaussian splatting system in virtual reality. ArXiv abs/2401.16663 (2024)

38. Jin, X., Li, V.: Three-dimensional deformation using directional polar coordinates. Journal of Graphics Tools 5(2), 15–24 (2000)

39. Ju, T., Schaefer, S., Warren, J.D.: Mean value coordinates for closed triangular meshes. ACM SIGGRAPH 2005 Papers (2005)

40. Kazhdan, M.M., Bolitho, M., Hoppe, H.: Poisson surface reconstruction. In: Euro graphics Symposium on Geometry Processing (2006)

41. Kazhdan, M.M., Hoppe, H.: Screened poisson surface reconstruction. ACM Trans. Graph. 32, 29:1–29:13 (2013)

42. Kerbl, B., Kopanas, G., Leimkuehler, T., Drettakis, G.: 3d gaussian splatting for real-time radiance field rendering. ACM Transactions on Graphics (TOG) 42, 1 – 14 (2023)

43. Kobayashi, K.G., Ootsubo, K.: t-fd: free-form deformation by using triangular mesh. In: Proceedings of the eighth ACM symposium on Solid modeling and applications. pp. 226–234 (2003)

44. Kopanas, G., Philip, J., Leimkühler, T., Drettakis, G.: Point-based neural rendering with per-view optimization. Computer Graphics Forum 40 (2021)

45. Lassner, C., Zollhöfer, M.: Pulsar: Eficient sphere-based neural rendering. 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) pp. 1440–1449 (2021)

46. Lee, S.Y., Chwa, K.Y., Shin, S.Y.: Image metamorphosis using snakes and free-form deformations. In: Proceedings of the 22nd annual conference on Computer graphics and interactive techniques. pp. 439–448 (1995)

47. Lei, H.: Ofsetopt: Explicit surface reconstruction without normals. 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) pp. 11729–11738 (2025)

48. Levin, D.: Mesh-independent surface interpolation (2004)

49. Liang, R., Zhang, J., Li, H., Yang, C., Vijaykumar, N.: Spidr: Sdf-based neural point fields for illumination and deformation. ArXiv abs/2210.08398 (2022)

50. Lipman, Y., Sorkine, O., Alexa, M., Cohen-Or, D., Levin, D., Rössl, C., Seidel, H.P.: Laplacian framework for interactive mesh editing. International Journal of Shape Modeling 11(01), 43–61 (2005)

51. Liu, R., Xiang, J., Zhao, B., Zhang, R., Yu, J., Zheng, C.: Neural impostor: Editing neural radiance fields with explicit shape manipulation. Computer Graphics Forum 42 (2023)

52. Ma, B., Han, Z., Liu, Y.S., Zwicker, M.: Neural-pull: Learning signed distance functions from point clouds by learning to pull space onto surfaces. ArXiv abs/2011.13495 (2020)

53. Mildenhall, B., Srinivasan, P.P., Tancik, M., Barron, J.T., Ramamoorthi, R., Ng, R.: Nerf: Representing scenes as neural radiance fields for view synthesis. Commun. ACM 65, 99–106 (2020)

54. Moccozet, L., Magnenat-Thalmann, N.: Dirichlet free-form deformations and their application to hand simulation. Proceedings. Computer Animation ’97 (Cat. No.97TB100120) pp. 93–102 (1997)

55. Müller, M., Heidelberger, B., Teschner, M., Gross, M.H.: Meshless deformations based on shape matching. ACM SIGGRAPH 2005 Papers (2005)

56. Müller, T., Evans, A., Schied, C., Keller, A.: Instant neural graphics primitives with a multiresolution hash encoding. ACM transactions on graphics (TOG) 41(4), 1–15 (2022)

57. Ost, J., Laradji, I.H., Newell, A., Bahat, Y., Heide, F.: Neural point light fields. 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) pp. 18398–18408 (2021)

58. Peng, S., Zhang, Y., Li, K.: Papr in motion: Seamless point-level 3d scene interpolation. 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) pp. 21007–21016 (2024)

59. Peng, Y., Yan, Y., Liu, S., Cheng, Y., Guan, S., Pan, B., Zhai, G., Yang, X.: Cagenerf: Cage-based neural radiance field for generalized 3d deformation and animation. In: Neural Information Processing Systems (2022)

60. Rafin, R., Neveu, M., Jaar, F.: Curvilinear displacement of free-form-based deformation. The Visual Computer 16(1), 38–46 (2000)

61. Rakhimov, R., Ardelean, A.T., Lempitsky, V.S., Burnaev, E.: Npbg++: Accelerating neural point-based graphics. ArXiv abs/2203.13318 (2022)

62. Rückert, D., Franke, L., Stamminger, M.: Adop: Approximate diferentiable onepixel point rendering. ACM Trans. Graph. 41, 99:1–99:14 (2021)

63. Ruprecht, D., Müller, H.: Free form deformation with scattered data interpolation methods. In: Geometric Modeling (1993)

64. Singh, K., Fiume, E.: Wires: a geometric deformation technique. In: Proceedings of the 25th annual conference on Computer graphics and interactive techniques. pp. 405–414 (1998)

65. Sorkine, O.: Laplacian mesh processing. Eurographics (State of the Art Reports) 4(4), 1 (2005)

66. Sorkine, O., Alexa, M.: As-rigid-as-possible surface modeling. In: Symposium on Geometry processing. vol. 4, pp. 109–116. Citeseer (2007)

67. Sorkine, O., Cohen-Or, D., Lipman, Y., Alexa, M., Rössl, C., Seidel, H.P.: Lapla cian surface editing. In: Proceedings of the 2004 Eurographics/ACM SIGGRAPH symposium on Geometry processing. pp. 175–184 (2004)

68. Sorkine-Hornung, O., Alexa, M.: As-rigid-as-possible surface modeling. In: Euro graphics Symposium on Geometry Processing (2007)

69. Vaswani, A., Shazeer, N.M., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, L., Polosukhin, I.: Attention is all you need. In: Neural Information Pro cessing Systems (2017)

70. Von Funck, W., Theisel, H., Seidel, H.P.: Vector field based shape deformations. ACM Transactions on Graphics (ToG) 25(3), 1118–1125 (2006)

71. Waczy’nska, J., Borycki, P., Tadeja, S.K., Tabor, J., Spurek, P.: Games: Mesh-based adapting and modification of gaussian splatting. ArXiv abs/2402.01459 (2024)

72. Wang, C., He, M., Chai, M., Chen, D., Liao, J.: Mesh-guided neural implicit field editing. ArXiv abs/2312.02157 (2023)

73. Wang, Y., Wang, J., Qu, Y., Qi, Y.: Rip-nerf: Learning rotation-invariant pointbased neural radiance field for fine-grained editing and compositing. Proceedings of the 2023 ACM International Conference on Multimedia Retrieval (2023)

74. Wang, Y., Wang, J., Wang, C., Qi, Y.: Rise-editing: Rotation-invariant neural point fields with interactive segmentation for fine-grained and eficient editing. Neural networks : the oficial journal of the International Neural Network Society 187, 107304 (2025)

75. Wilczy’nski, G., Zielinski, M., Byrski, K., Waczy’nska, J., Belter, D., Spurek, P.: Iris: Intersection-aware ray-based implicit editable scenes. ArXiv abs/2603.15368 (2026)

76. Wiles, O., Gkioxari, G., Szeliski, R., Johnson, J.: Synsin: End-to-end view synthesis from a single image. 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) pp. 7465–7475 (2019)

77. Wu, G., Yi, T., Fang, J., Xie, L., Zhang, X., Wei, W., Liu, W., Tian, Q., Wang, X.: 4d gaussian splatting for real-time dynamic scene rendering. 2024 IEEE/CVF

Conference on Computer Vision and Pattern Recognition (CVPR) pp. 20310–20320 (2023)

78. Xu, Q., Xu, Z., Philip, J., Bi, S., Shu, Z., Sunkavalli, K., Neumann, U.: Point-nerf: Point-based neural radiance fields. 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) pp. 5428–5438 (2022)

79. Xu, T., Harada, T.: Deforming radiance fields with cages. ArXiv abs/2207.12298 (2022)

80. Yang, B., Bao, C., Zeng, J., Bao, H., Zhang, Y., Cui, Z., Zhang, G.: Neumesh: Learning disentangled neural mesh-based implicit field for geometry and texture editing. ArXiv abs/2207.11911 (2022)

81. Yifan, W., Aigerman, N., Kim, V.G., Chaudhuri, S., Sorkine-Hornung, O.: Neural cages for detail-preserving 3d deformations. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 75–83 (2020)

82. Yuan, Y.J., Lai, Y.K., Wu, T., Gao, L., Liu, L.: A revisit of shape editing techniques: From the geometric to the neural viewpoint. Journal of Computer Science and Technology 36, 520 – 554 (2021)

83. Yuan, Y.J., Sun, Y.T., Lai, Y.K., Ma, Y., Jia, R., Gao, L.: Nerf-editing: Geometry editing of neural radiance fields. 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) pp. 18332–18343 (2022)

84. Zhang, Q., Baek, S.H., Rusinkiewicz, S., Heide, F.: Diferentiable point-based radiance fields for eficient view synthesis. SIGGRAPH Asia 2022 Conference Papers (2022)

85. Zhang, X., Chen, A., Xiong, J., Dai, P., Shen, Y., Xu, W.: Neural shell texture splatting: More details and fewer primitives. ArXiv abs/2507.20200 (2025)

86. Zhang, Y., Peng, S., Moazenipourasil, S.A., Li, K.: PAPR: Proximity attention point rendering. In: Thirty-seventh Conference on Neural Information Processing Systems (2023)

87. Zhang, Y., Zheng, J., Cai, Y.: Proxy-driven free-form deformation by topologyadjustable control lattice. Computers & Graphics 89, 167–177 (2020)

88. Zheng, C., chang Lin, W., Xu, F.: Editablenerf: Editing topologically varying neural radiance fields by key points. 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) pp. 8317–8327 (2022)

89. Zhou, K., Hong, L., Xie, E., Yang, Y., Li, Z., Zhang, W.: Serf: Fine-grained interactive 3d segmentation and editing with radiance fields. ArXiv abs/2312.15856 (2023)

90. Zuo, Y., Deng, J.: View synthesis with sculpted neural points. ArXiv abs/2205.05869 (2022)

91. Zwicker, M., Pauly, M., Knoll, O., Gross, M.H.: Pointshop 3d: an interactive system for point-based surface editing. Proceedings of the 29th annual conference on Computer graphics and interactive techniques (2002)

## A Qualitative Comparison with Neural Editor

Figures 8 and 9 present side-by-side qualitative comparisons between our method and Neural Editor [16] on the Neural Editor and Objaverse datasets, respectively. Both methods produce visually faithful deformations; however, Neural Editor occasionally exhibits holes or color bleeding in fine-detail regions, whereas our method preserves surface continuity. Beyond rendering quality, Neural Editor is also far more expensive: it requires ∼7 days of training per scene and a further ∼23 minutes of preprocessing per frame at inference on an NVIDIA RTX 3090— including KD-tree construction, mesh normal recomputation, and infinitesimal surface transformation—whereas our method needs only ∼1 minute of fine-tuning and no preprocessing of any kind.

![](images/a899dda5224d12874a7682af0936eb33381481411316acd4ecf89bdc48608108.jpg)  
Fig. 8: Qualitative comparison between our method and Neural Editor on the Neural Editor dataset [16].

## B Comparison with PAPR-in-Motion

PAPR-in-Motion [58] uses a supervised two-stage pipeline: a geometry fine-tuning stage that moves the start-state point cloud to match the end-state geometry, followed by an appearance fine-tuning stage that captures deformation-induced appearance changes. Table 4 compares our zero-shot result against the renderings from both stages over the six Objaverse scenes. Without any target-state rendering or geometry supervision, our method substantially outperforms the geometry-only stage. The remaining gap to the appearance-finetuned result is expected, since that variant is directly supervised by target-state ground-truth images, including shadows, highlights, and reflections that are dificult to infer zero-shot.

![](images/db523676f90c0861139b8b9573e39eb177e5237d357a989d54703f3fb4c1040a.jpg)  
Fig. 9: Qualitative comparison between our method and Neural Editor on scenes from Objaverse [19, 58].

Table 4: Ours vs. PAPR-in-Motion over the six Objaverse scenes.
<table><tr><td colspan="2">Method</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td></tr><tr><td>PAPR-in-Motion (geometry finetune only)</td><td></td><td>26.31</td><td>0.946</td><td>0.044</td></tr><tr><td></td><td>PAPR-in-Motion (geometry + appearance finetune)</td><td>33.46</td><td>0.986</td><td>0.021</td></tr><tr><td>Ours (zero-shot, no end-state supervision)</td><td></td><td>28.84</td><td>0.968</td><td>0.030</td></tr></table>

## C Range of Deformation

During fine-tuning, we use analytic deformations following Blender’s Simple Deform operations, including stretch, twist, bend, and uniform scaling. Stretch allows up to a 3× factor, twist and bend allow up to $6 0 ^ { \circ }$ , and uniform scaling allows up to 2× dilation or 0.5× shrinkage. The deformation type, axis/direction, and magnitude are uniformly sampled at each iteration, without composing multiple deformations.

## D Additional Realistic Deformation Results

In addition to the Objaverse scenes with realistic motions evaluated in the main paper (e.g., the flying butterfly and walking girafe), we evaluate two additional dynamic scenes from Objaverse [19] and Adobe Mixamo [1] with realistic motions: making a fist and running. We train our method on the first time step and evaluate multi-view renderings at three uniformly sampled held-out time steps. As shown in Fig. 10 and Table $5 ,$ our method efectively improves surface continuity for these realistic deformations, despite being fine-tuned only with random analytical deformations.

Train (t<sub>0</sub>)  
![](images/b5dd3cf668a6fe5de6d249126c6050f824766c1b95d9de89634623fd06e30f68.jpg)

PAPR (t<sub>1</sub>)  
![](images/392783eb54ebb14d3c663171358aa310762340782c1103b48a222f4fe4d8dbb4.jpg)

Ours (t<sub>1</sub>)  
![](images/d714a8cd6b74a099ebdf65caad730660f0af9c0b2aafceba0147e9a49a075a64.jpg)

PAPR (t<sub>2</sub>)  
![](images/deb5e728e9f2a6310c07b301e0bf8de623b6bcf953061610b91a5a6579e8f7ee.jpg)

Ours (t<sub>2</sub>)  
![](images/3ad794c1b0484b9b93b166b1e88f4838bcd3838bd22520ba3ceddc14359c1afb.jpg)  
Fig. 10: Results for realistic deformation. $t _ { 1 , 2 }$ are zero-shot edits.

Table 5: Quantitative results for the realistic deformations.
<table><tr><td></td><td colspan="3">Objaverse hand</td><td colspan="3">Mixamo running</td></tr><tr><td>Method</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td></tr><tr><td>PAPR</td><td>28.55</td><td>0.964</td><td>0.040</td><td>24.40</td><td>0.956</td><td>0.046</td></tr><tr><td>Ours</td><td>29.66</td><td>0.972</td><td>0.028</td><td>29.95</td><td>0.979</td><td>0.021</td></tr></table>

"remove the artifacts"

## E Comparison with GaussianEditor

GaussianEditor [17] mainly focuses on text-driven appearance editing and object addition/removal, rather than free-form deformation. Nevertheless, we evaluate whether it can refine deformed Gaussians and reduce the resulting artifacts using diferent prompts, as shown in Fig. 11. As shown, it fails to reduce the artifacts while preserving identity or sharp details.

![](images/e5a8fcfc7d899d3874cf0d8889bcd0cee9122b10baab6f40b0e8fb2d40bfcfa4.jpg)  
Fig. 11: GaussianEditor refinement (please zoom in for prompts).

## F Reconstruction Quality on Real-World Scenes

We evaluate the novel-view synthesis quality of our method on the real-world DTU [36] and Mip-NeRF 360 [7] datasets before any editing. Table 6 reports the mean PSNR, SSIM, and LPIPS against 3DGS [42], 2DGS [32] and Mani-GS [27]. As full resolution training on the Mip-NeRF 360 dataset exceeds the memory of our GPUs, we evaluate our method at half the resolution of the Mip-NeRF 360 dataset. The numbers for the Gaussian splatting-based baselines are from their original papers and NeST-Splatting [85].

Table 6: Mean novel-view synthesis quality on real-world DTU [36] and Mip-NeRF 360 [7] scenes (no editing). ‡ For Mip-NeRF 360 dataset P-CORE is trained and rendered at the half-resolution of the dataset, as full resolution evaluation exceeds our GPU memory.
<table><tr><td rowspan="3">Method</td><td colspan="3">DTU [36]</td><td colspan="3">Mip-NeRF 360 [7]</td></tr><tr><td>PSNR ↑</td><td>SSIM ↑</td><td>LPIPS √</td><td>PSNR ↑</td><td>SSIM ↑</td><td>LPIPS ↓</td></tr><tr><td>3DGS [42]</td><td>33.77</td><td>0.965</td><td>0.044</td><td>26.54</td><td>0.776</td><td>0.266</td></tr><tr><td>2DGS [32]</td><td>33.89</td><td>0.966</td><td>0.048</td><td>27.03</td><td>0.805</td><td>0.223</td></tr><tr><td>Mani-GS [27]</td><td>31.50</td><td>0.943</td><td>0.088</td><td></td><td></td><td></td></tr><tr><td>P-CORE (Ours)‡</td><td>33.48</td><td>0.973</td><td>0.023</td><td>28.04</td><td>0.808</td><td>0.183</td></tr></table>