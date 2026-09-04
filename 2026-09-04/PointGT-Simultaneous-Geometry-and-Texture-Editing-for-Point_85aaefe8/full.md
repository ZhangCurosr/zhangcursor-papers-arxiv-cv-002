# PointGT: Simultaneous Geometry and Texture Editing for Point-Based Representations

Yanshu Zhang<sup>1⋆</sup> , George Shramko<sup>1</sup> , Pratul P. Srinivasan<sup>4</sup> , and Ke Li<sup>1,2,3</sup>

<sup>1</sup> Simon Fraser University, Canada

<sup>2</sup> Alberta Machine Intelligence Institute (Amii), Canada

Canadian Institute for Advanced Research (CIFAR), Canada

Google DeepMind, USA

{yanshu\_zhang, george\_shramko, keli}@sfu.ca pratul.srinivasan@gmail.com

Abstract. We present PointGT, a point-based 3D representation that enables simultaneous editing of object geometry and appearance. Existing reconstruction and view synthesis techniques produce volumetric 3D representations that are high-quality and photorealistic, but are difficult to edit. In particular, recent eforts to enable texture editing for 3D Gaussian Splatting representations are not compatible with geometry edits and deformations. Our method combines a point-based representation that is well-suited for geometry deformations with a learned UV mapping technique that enables high-resolution texture editing. We show that PointGT enables fine-grained editing of both geometry and texture in point-based neural representations with high rendering quality. Project page: https://zvict.github.io/pointgt/.

Keywords: Point neural representation · Geometry and texture editing

## 1 Introduction

Point-based radiance field representations, most notably 3D Gaussian Splatting (3DGS) [10], have emerged as a powerful approach to novel view synthesis and 3D reconstruction. These methods represent scenes as point clouds, where each point is associated with a volumetric primitive (e.g. a 3D Gaussian distribution) that has an opacity and a single view-dependent color. Rasterizing these volumetric primitives is eficient, so techniques such as 3DGS can quickly optimize large collections of primitives to represent detailed scenes.

However, since each Gaussian is assigned a single (view-dependent) color, 3DGS can only represent detailed appearance and textures by using many (often millions) of Gaussian primitives. This coupling of geometry and appearance resolution prevents scenes reconstructed by 3DGS from being easily edited. Editing appearance is dificult because the textures of reconstructed scenes cannot be modified at resolutions finer than the sizes of the reconstructed Gaussians, and editing geometry is dificult because it is not clear how to non-rigidly transform Gaussian primitives based on user manipulations.

![](images/1f4893c1de0d8126c76258c735bf418da25f755354691c53fa7fc600c8cd0963.jpg)  
Fig. 1: Overview Given multi-view captures (left), PointGT reconstructs a pointbased 3D representation and learns a global UV mapping onto 2D charts (middle left). By maintaining a deformation-aware correspondence between canonical and deformed space, PointGT ensures persistent texture edits under non-rigid geometry changes. Users can simultaneously deform and retexture a reconstructed dress (top) or a realworld table (bottom), with geometry and appearance fully decoupled and editable.

Recent eforts to decouple appearance and geometry in Gaussian splatting have extended 3DGS to include either global [27] or per-primitive [3,16,18,22,23, 30] UV maps. However, these techniques are still unable to enable both geometry deformations and appearance editing. Learned global UV maps from 3D surface points to 2D UV coordinates are no longer valid after geometry deformations, and per-primitive UV maps have dificulties handling deformations that are finer than the size of the Gaussian primitives.

We focus on another point-based representation that uses attention to render a point cloud [2, 14, 31], which efectively avoids those limitations by modeling scenes as a collection of infinitesimal points with learned features. These methods render rays with a cross-attention mechanism between the ray and the K points closest to the ray. This defines the aggregated feature of a ray as the interpolation of the features from those K nearest points, which enables editing scene geometry since these interpolation weights are still valid after deformations of the scene points. While attention-based point representation is better suited for geometry editing than 3DGS [31], appearance editing is still challenging as it is not possible to locally edit textures by modifying the optimized point features.

In this paper, we propose PointGT, a representation that extends the attentionbased point renderer PAPR [31] to enable simultaneous editing of geometry and appearance. Our key contributions are:

– We extend PAPR to include a learned UV mapping from 3D points on objects to a learned 2D texture map. (Sec. 4.2)

– We introduce two novel geometry regularizers that significantly improve the fidelity of PAPR’s geometry representation and demonstrate that these are crucial for efective UV mapping. (Sec. 4.1)

– We propose a deformation-aware correspondence mechanism that explicitly maps deformed-space ray intersections back to canonical space via displacement fusion, preserving texture attachment under non-rigid edits. (Sec. 4.3)

## 2 Related Work

Editing the geometry and appearance of 3D assets is a core part of the computer graphics workflow for digital content creation. While there are many established techniques and tools for editing widely-used mesh-based representations, editing the 3D representations produced by recent view synthesis and 3D reconstruction methods is much more challenging.

The geometry of neural scene representations such as Neural Radiance Fields (NeRFs) [13] cannot be directly edited, so many works [8,15,24,26,28,29,32,34] have focused on associating NeRFs with proxy geometry which can be edited, and then converting this edited proxy geometry back into the NeRF representation. However, converting NeRFs to proxy geometry and back can introduce approximation errors, and the choice of proxy geometry can limit the types of edits that are possible.

On the other hand, the appearance of reconstructed NeRFs can be edited by optimizing UV mapping functions that map from 3D coordinates on the object to 2D texture coordinates [20, 25]. In this work, we use ideas from Nuvo [20] to develop a UV mapping technique for point-based reconstruction techniques.

## 2.1 3D Shape Deformation and Correspondence

Classical geometry processing has a rich history of editing 3D shapes, leveraging methods like As-Rigid-As-Possible (ARAP) deformation [19], Moving Least Squares (MLS), and embedded deformation graphs [21] to manipulate surface meshes or cages flexibly. While these approaches efectively manipulate geometry and preserve local structural rigidity, applying appearance edits to the deformed surfaces is relatively straightforward for meshes because texture coordinates (UVs) are explicitly attached to vertices and interpolate naturally across triangles.

However, projecting and interpolating high-resolution 2D textures onto unstructured, dynamically-weighted point clouds is fundamentally dificult. As point-based neural representations (like PAPR or 3DGS) lack explicit connectivity, the set of neighbors and attention weights used to define a surface point can shift non-rigidly under deformation. PointGT directly targets this novel problem of establishing canonical correspondence within dynamic attention fields to enable robust texture persistence.

More recent view synthesis techniques are based on 3D Gaussian Splatting (3DGS), which is a point-based representation that optimizes parameters of a 3D Gaussian along with a view-dependent color (parameterized using spherical harmonics) for each point. 3DGS is more suitable for direct geometry editing, since it is straightforward to move the position, orientation, and scale of any particle in the representation.

While Gaussian splatting has proved to be a powerful representation for numerous tasks, this paradigm has several limitations: (1) it often requires millions of Gaussian primitives to enable high rendering quality and accurate scene reconstructions, (2) it’s challenging to make appearance edits that are finer than the size of an individual Gaussian, and (3) free-form geometry deformation remains inconvenient without introducing geometry proxies.

Recent works tackle the first two issues by decoupling the appearance and geometry of Gaussian splatting. Texture-GS [27] learns a global UV mapping for all the ray-Gaussian intersections and stores features of these intersections on a 2D texture map. Instead of learning a global UV mapping with surface parameterization constraints (e.g., simple topologies), another line of works [3,16,18,22,23,30] learn a per-primitive texture map for each Gaussian, where the Gaussians are often restricted to be 2D surfels [5] so texture for each point on the surfel can easily be queried from the texture map with their 2D UV coordinates. Since the frequency of surface textures is often much higher than that of geometry, decoupling the resolutions of geometry and texture representations allows these methods to render scenes photo-realistically with only 5K points [3,16,22,27,30] and enables manipulation of the textures with fine-grained details [16, 23, 27].

However, none of these methods can perform simultaneous geometry deformation and texture editing. This limitation is rooted in the fundamental representation of Gaussian splatting, where each point is represented as a volume of space. In the case where per-primitive texture maps are learned, preserving textures after deformation requires not only the correct transformation (rotation, scaling) of each Gaussian but also accurate splitting or cloning of primitives to represent deformations that are finer than a Gaussian (as Gaussians are rigid and cannot be bent). Such a process is often intractable in practice. In the case where a global UV mapping is learned, in addition to the dificulties described above, the UV mapping must be correctly found for the new intersection points after deformation. The combination of these factors makes the simultaneous editing of texture and geometry for Gaussian splatting extremely dificult.

## 3 Preliminaries

We briefly introduce the relevant background on attention-based point representations and neural UV mappings, and establish the notation used throughout.

## 3.1 Attention-Based Point Renderers

Attention-based point renderers (like PAPR [31] and Pointersect [2]) represent scenes as unstructured point clouds coupled with an attention interpolation mechanism. For instance, PAPR represents a scene as N learnable points $\mathcal { P } = \{ ( \mathbf { p } _ { j } , \mathbf { f } _ { j } ) \} _ { j = 1 } ^ { N }$ , where $\mathbf { p } _ { j } \in \mathbb { R } ^ { 3 }$ is the spatial position and $\mathbf { f } _ { j } \in \mathbb { R } ^ { h }$ is a feature vector. To render ray $\mathbf { r } _ { i } .$ , PAPR selects the K points nearest to it by perpendicular distance as the local neighborhood. A learned cross-attention mechanism then assigns a softmax weight $a _ { i j }$ to each neighbor point $j ,$ and the ray’s feature ${ \bf f } _ { i } ^ { \mathrm { r a y } }$ is computed as the weighted sum of their features:

$$
\mathbf { f } _ { i } ^ { \mathrm { r a y } } = \sum _ { j = 1 } ^ { K } a _ { i j } \ : \mathbf { f } _ { i j } ,\tag{1}
$$

where $\mathbf { f } _ { i j }$ denotes the feature associated with the $j \cdot$ -th selected neighbor point $\mathbf { p } _ { i j }$ for ray i. The aggregated feature map is then decoded by a UNet [17] to predict the RGB colors. Given the same attention weights, these methods typically predict the intersection point $\mathbf { x } _ { i } \in \mathbb { R } ^ { 3 }$ where the ray hits the surface by interpolating the point positions:

$$
\mathbf { x } _ { i } = \sum _ { j = 1 } ^ { K } a _ { i j } \mathbf { p } _ { i j } .\tag{2}
$$

Note that $\mathbf { x } _ { i }$ is an attention-weighted average of point positions, not a true geometric intersection: its quality depends on how concentrated the weights are and how well-placed the supporting points are. This sensitivity motivates the geometry regularizers introduced in Sec. 4.1.

Canonical and deformed space. We call the point positions after training the canonical positions $\mathbf { p } _ { j } ^ { \mathrm { c a n } }$ . The corresponding canonical intersection

$$
\mathbf { x } _ { i } ^ { \mathrm { { c a n } } } = \sum _ { j = 1 } ^ { K } a _ { i j } ^ { \mathrm { { c a n } } } \mathbf { p } _ { i j } ^ { \mathrm { { c a n } } }\tag{3}
$$

is what the UV mapping is trained on. At edit time, a non-rigid deformation displaces each point to $\mathbf { p } _ { j } ^ { \mathrm { d e f } }$ , producing a deformed intersection

$$
\mathbf { x } _ { i } ^ { \mathrm { { d e f } } } = \sum _ { j = 1 } ^ { K } a _ { i j } ^ { \mathrm { { d e f } } } \mathbf { p } _ { i j } ^ { \mathrm { { d e f } } }\tag{4}
$$

in deformed space, where direct UV lookup is no longer valid. Bridging this gap is the central problem addressed in Sec. 4.3.

## 3.2 Neural UV Mapping

Neural UV mapping methods (like Nuvo [20] and NeuTex [25]) parameterize surface textures by learning continuous mappings from 3D space to 2D texture atlases. For example, Nuvo learns a multi-chart UV atlas and texture map from a set of surface points $\mathcal { G }$ (in our setting, $\mathcal { G } = \{ \mathbf { x } _ { i } ^ { \mathrm { c a n } } \} ,$ . It uses cycle-consistency (bijectivity) losses and additional regularizers to learn the UV mappings; we refer readers to [20] for full details.

To enable texture editing and to encourage an even allocation of texture resolution over the scene, Nuvo also penalizes mapping distortion using conformal and stretch regularizers that operate on the diferential of each chart’s 3D→ 2D UV mapping [20]. These regularizers require the surface normal at each surface point to construct tangent-space directions, which are not explicitly available for raw attention-interpolated intersection points ${ \bf x } _ { i } ^ { \mathrm { c a n } }$ . We therefore use a normalfree distortion loss in Sec. 4.2 that removes the need for surface normals.

![](images/5a013355c5a80f557ce1715a734a1a006e2a07e4c7af646a415e5b84f9eb9247.jpg)  
Fig. 2: Comparison of geometry deformations of textured Gaussian splatting methods and PointGT. Left: Since each Gaussian is a linear primitive, these methods struggle to preserve surface smoothness under large non-rigid deformations. Right: PointGT maintains surface continuity after deformation by predicting ray–surface points via attention-based interpolation on a point cloud.

## 4 Method

The goal of PointGT is to enable persistent texture edits under non-rigid deformation for point-based neural representations. While applying texture edits to deformed surfaces is relatively straightforward for meshes—where texture coordinates (UVs) are explicitly attached to vertices and interpolate naturally across triangles—point clouds lack this explicit connectivity. Consequently, maintaining a consistent mapping between the deformed geometry and a 2D texture map becomes fundamentally dificult, often causing texture edits to drift or break when the underlying points move non-rigidly.

To tackle this challenge, the choice of the underlying point-based representation is critical. As discussed, Gaussian Splatting-based methods struggle to preserve surface continuity under non-rigid deformations (Fig. 2, left), and may break continuous texture mapping. In contrast, attention-based renderers like PAPR [31] maintain surface continuity via smooth point interpolation (Fig. 2, right), providing the stable surface points required for consistent UV mapping.

Building upon this insight, PointGT introduces a framework for simultaneous geometry and texture editing. To ensure the attention-based surface points are suitable for learning a multi-chart UV atlas and texture map [20], we first propose geometry regularization losses that condition the ray–surface intersections to accurately reflect the true geometry (Sec. 4.1). Then, to preserve texture attachment during edits, we introduce a deformation-aware canonical correspondence mechanism (Sec. 4.3). This explicitly maps the deformedspace intersections reliably back to the original canonical space for consistent UV lookup, ensuring that texture edits persist seamlessly regardless of the applied geometric deformation.

## 4.1 Learning PAPR with Geometry Regularization

To enable reliable UV learning and stable editing, we require accurate and wellconditioned ray–surface intersection points. In PAPR, the predicted surface point for ray i is an attention-weighted average of K ray-nearest neighbors (Eq. 2), which can be sensitive to sparse attention and imperfect point geometry. Without additional constraints, predicted intersection points may collapse toward discrete support points and/or be supported by noisy of-surface neighborhoods (Fig. 6 and 7a), which harms the sampled surface point set used to train the UV atlas. We therefore add two geometry regularizers during PAPR pre-training.

On-ray regularizer (close-to-ray). As shown in Fig. 6, sparse attention weights can cause predicted intersection points to cluster around discrete point supports. This degrades the efective geometric resolution of $\mathbf { x } _ { i }$ to the resolution of the point cloud and yields unstable surface samples for UV supervision. Since a valid ray–surface intersection point should lie on its ray by definition, we minimize the distance between each predicted point $\mathbf { x } _ { i }$ and its orthogonal projection $\mathbf { x } _ { i } ^ { \prime }$ onto the corresponding ray $\mathbf { r } _ { i } { : }$

$$
\mathcal { L } _ { \mathrm { c l o s e 2 r a y } } = \frac { 1 } { M } \sum _ { i = 1 } ^ { M } \Vert \mathbf { x } _ { i } - \mathbf { x } _ { i } ^ { \prime } \Vert _ { 2 } ,\tag{5}
$$

where M is the number of rays in a batch.

On-surface neighborhood regularizer (close-to-surface). Even when intersections can be predicted, the underlying point cloud may contain of-surface points. This is problematic when starting from sparse points and densifying: new points may gather around an of-surface location, and a ray may then select a neighborhood whose convex hull does not cover the true surface. To regularize neighborhood geometry, we encourage each selected neighbor point $\mathbf { p } _ { i j }$ to move toward the predicted surface point $\mathbf { x } _ { i } \mathbf { \cdot }$

$$
\mathcal { L } _ { \mathrm { c l o s e 2 s u r f a c e } } = \frac { 1 } { M K } \sum _ { i = 1 } ^ { M } \sum _ { j = 1 } ^ { K } \| \mathbf { p } _ { i j } - \mathrm { s g } ( \mathbf { x } _ { i } ) \| _ { 2 }\tag{6}
$$

that encourages all the nearby points to move closer to the predicted intersection point. Note that we stop the gradients backpropagating to $\mathbf { x } _ { i }$ from this regularizer for training stability. To prevent early geometry collapse, we apply $\mathcal { L } _ { \mathrm { c l o s e 2 s u r f a c e } }$ in the middle of training after a coarse geometry is learned.

Objective. We combine the regularizers with PAPR’s rendering loss for pretraining:

$$
\mathcal { L } _ { \mathrm { r e n d e r i n g } } + \gamma \mathcal { L } _ { \mathrm { c l o s e 2 r a y } } + \eta \mathcal { L } _ { \mathrm { c l o s e 2 s u r f a c e } } ,\tag{7}
$$

where $\mathcal { L } _ { \mathrm { r e n d e r i n g } }$ is PAPR’s rendering loss, γ and $\eta$ are the coeficients of the regularizers.

## 4.2 Learning UV Mapping

Building upon the regularized point cloud geometry, we extract ray–surface intersections to learn a multi-chart UV atlas and a 2D texture map. We sample foreground rays from the training views, compute their canonical-space intersection points ${ \bf x } _ { i } ^ { \mathrm { c a n } }$ , and optimize a continuous neural UV parameterization over these surface samples.

Distortion loss without normals (Jacobian regularization). To enable consistent texture editing and encourage an even allocation of texture resolution across the scene, it is crucial to minimize the distortion of the learned UV mappings. While existing surface parameterization methods [20] typically rely on ground-truth geometry normals to compute conformal and stretch penalties, point-based representations natively lack this explicit geometric connectivity and normal information. To overcome this, we define a distortion loss directly over the Jacobians of the texture coordinate mappings.

Specifically, for each chart k in our multi-chart texture maps, we compute the Jacobian matrix $\mathbf { J } _ { i k } \in \mathbb { R } ^ { 2 \times 3 }$ of the UV mapping by taking the gradients of the texture coordinates ${ \bf u } _ { i k } \in \mathbb { R } ^ { 2 }$ w.r.t. the 3D point coordinates $\mathbf { x } _ { i } .$ The Jacobian’s singular values $\sigma _ { i k } ^ { 1 }$ and $\sigma _ { i k } ^ { 2 }$ characterize the local scaling behavior of the mapping. To minimize anisotropic scaling distortion, we minimize the squared diference of the singular values across all charts:

$$
\mathcal { L } _ { \mathrm { s c a l i n g } } = \frac { 1 } { G n } \sum _ { i = 1 } ^ { G } \sum _ { k = 1 } ^ { n } \left( \sigma _ { i k } ^ { 1 } - \sigma _ { i k } ^ { 2 } \right) ^ { 2 } ,\tag{8}
$$

where $G$ is the total number of 3D points. This encourages the texture coordinate network to learn mappings with isotropic scaling (where $\sigma ^ { 1 } = \sigma ^ { 2 } )$ While $\mathcal { L } _ { \mathrm { s c a l i n g } }$ minimizes anisotropy, it does not explicitly prevent area shrinkage or blow-up. To address this, we also define an area distortion penalty, $\begin{array} { r } { \bar { \mathcal { L } } _ { \mathrm { a r e a } } = \frac { 1 } { G n } \sum _ { i = 1 } ^ { G } \sum _ { k = 1 } ^ { n } ( \log ( \sigma _ { i k } ^ { 1 } \sigma _ { i k } ^ { 2 } ) ) ^ { 2 } } \end{array}$ , which regularizes the local scale to remain close to 1 and heavily penalizes near-zero areas, preventing degenerate mappings. We define the distortion loss as the sum of these two terms:

$$
\mathcal { L } _ { \mathrm { d i s t o r t i o n } } = \mathcal { L } _ { \mathrm { s c a l i n g } } + \mathcal { L } _ { \mathrm { a r e a } } .\tag{9}
$$

In practice, we optimize the UV atlas by minimizing a weighted sum of Nuvo $[ 2 0 ] \mathrm { { s } }$ losses (excluding the normal-based distortion terms) and the proposed distortion loss $\mathcal { L } _ { \mathrm { d i s t o r t i o n } }$ . We jointly learn a 2D RGB texture map by supervising rays with PAPR-predicted colors using bilinear sampling on the atlas; we use the learned texture map for texture editing.

## 4.3 Geometry and Texture Editing

Texture editing in UV space. PointGT supports texture editing by directly modifying the 2D texture atlas. Given an edited texture map $\mathbf { T } ^ { \prime }$ , we generate a mask map $\mathbf { \bar { M } } \in [ 0 , 1 ] ^ { n \times h \times w }$ by comparing the original atlas T and the edited atlas $\mathbf { T } ^ { \prime }$ where the pixel values are marked as 1 in edited regions and 0 otherwise. During rendering, for each ray we compute the UV coordinate $\mathbf { u } _ { i }$ of its intersection and replace the sampled color with the edited texture when $m _ { i } = \mathbf { M } ( \mathbf { u } _ { i } ) > 0 . 9 5$ (both using bilinear interpolation on the atlas).

Geometry editing and the correspondence problem. Geometry editing applies a non-rigid deformation to the learned point cloud, producing deformed point positions $\mathbf { p } _ { j } ^ { \mathrm { d e f } }$ from canonical positions $\mathbf { p } _ { j } ^ { \mathrm { c a n } }$ (Sec. 3.1). Rendering after deformation produces a deformed-space ray–surface point by interpolating deformed neighbor positions with deformed-space attention weights:

$$
\hat { \mathbf { x } } _ { i } ^ { \mathrm { d e f } } = \sum _ { j = 1 } ^ { K } a _ { i j } ^ { \mathrm { d e f } } \mathbf { p } _ { i j } ^ { \mathrm { d e f } } ,\tag{10}
$$

where the top-K neighbors $\mathbf { p } _ { i j } ^ { \mathrm { d e f } }$ are selected by ray proximity in the deformed point cloud, and the weights $a _ { i j } ^ { \mathrm { { \check { d e f } } } }$ are re-inferred in deformed space (i.e., both the neighbor set and weights are recomputed after deformation). However, the UV atlas is trained in canonical space, so we must transfer $\hat { \mathbf { x } } _ { i } ^ { \mathrm { d e f } }$ back to a canonical coordinate before UV lookup.

Why a naïve transfer fails. A natural approach is to reuse the deformed attention weights to directly interpolate canonical neighbor positions:

$$
\mathbf { x } _ { i , \mathrm { n a i v e } } ^ { \mathrm { c a n } } = \sum _ { j = 1 } ^ { K } a _ { i j } ^ { \mathrm { d e f } } \mathbf { p } _ { i j } ^ { \mathrm { c a n } } .\tag{11}
$$

In practice, attention weights in PAPR are often sparse; $\operatorname { E q }$ . 11 therefore collapses many rays’ canonical lookup points onto a small subset of support points, causing discontinuous UV jumps across neighboring rays and visible texture popping/drift under deformation. While the on-ray regularizer (Sec. 4.1) improves training stability, it does not guarantee that intersections remain exactly on rays under extreme non-rigid, non-training deformations.

Deformation-aware canonical correspondence. To address this, we propose an edit-time transfer rule that maps each deformed-space ray–surface point back to canonical space while keeping samples well-distributed. Step 1 (point-to-ray projection). We strictly enforce the on-ray constraint by orthogonally projecting $\hat { \mathbf { x } } _ { i } ^ { \mathrm { d e f } }$ onto its ray $\mathbf { r } _ { i } { : }$

$$
\mathbf { x } _ { i } ^ { \mathrm { d e f } } = { \cal I } _ { \mathbf { r } _ { i } } ( \hat { \mathbf { x } } _ { i } ^ { \mathrm { d e f } } ) ,\tag{12}
$$

![](images/6108ef52dd472444289652a44231cd7ba16879d6c377b2a6267827f15b3a9f75.jpg)  
Fig. 3: Illustration of the deformation-aware canonical correspondence. In deformed space (left), a ray intersects the deformed point cloud to predict a surface point (red). Each deformed neighbor (dark blue) has a stored canonical counterpart (light blue); the per-neighbor displacements (dashed gray arrow) are fused with the deformed-space attention weights to transfer the intersection back to canonical space (middle), where UV lookup retrieves the texture (right).

Step 2 (attention-weighted displacement fusion). Since point identity is preserved under deformation, each deformed neighbor $\mathbf { p } _ { i j } ^ { \mathrm { d e f } }$ has a corresponding canonical position ${ \bf p } _ { i j } ^ { \mathrm { c a n } }$ . We compute per-neighbor displacements and fuse them using the same deformed-space attention weights, then transfer the ray-projected deformed intersection back to canonical space:

$$
\mathbf { t } _ { i } = \sum _ { j = 1 } ^ { K } a _ { i j } ^ { \mathrm { d e f } } \left( \mathbf { p } _ { i j } ^ { \mathrm { c a n } } - \mathbf { p } _ { i j } ^ { \mathrm { d e f } } \right) , \quad \mathbf { x } _ { i } ^ { \mathrm { c a n } } = \mathbf { x } _ { i } ^ { \mathrm { d e f } } + \mathbf { t } _ { i } .\tag{13}
$$

We then query the UV atlas at ${ \bf x } _ { i } ^ { \mathrm { c a n } }$ and sample the (edited) texture map to render appearance. Intuitively, even if attention weights are sparse, ray projection keeps $\mathbf { x } _ { i } ^ { \mathrm { { d e f } } }$ distributed along rays, and the fused displacement provides a stable canonical transfer for UV lookup.

Edit-order interchangeability. PointGT supports flexible and simultaneous geometry and texture editing; the order of applying geometry and texture edits is interchangeable, which is important for applications such as 3D character design where edits are applied iteratively in varying order.

## 5 Experiments

We evaluate PointGT along two axes: (i) simultaneous geometry and texture editing under non-rigid deformation, where texture edits should persist without drifting (Sec. 4.3), and (ii) novel view synthesis quality compared to recent textured Gaussian splatting baselines.

## 5.1 Implementation Details

When pre-training PAPR with geometry regularization (Sec. 4.1), we use γ = 0.002 and η = 0.01 for the close-to-ray and close-to-surface losses. Since geometry is unreliable early in training, we start applying the regularizers at step 25,000 (i.e., after 1/10 of total steps). When learning the UV atlas and 2D texture map (Sec. 4.2), we follow Nuvo’s optimization settings and use weight 0.4 for the proposed $\mathcal { L } _ { \mathrm { d i s t o r t i o n } }$ and 0.04 for $\mathcal { L } _ { \mathrm { t e x t u r e } }$ . We use $n = 4$ or 8 charts depending on the complexity of the geometry, and fix memory usage by using a texture resolution of $2 5 6 { \sqrt { 2 / n } } \times 2 5 6 { \sqrt { 2 / n } }$ for each chart. For editing evaluation, we use Objaverse assets with artist-created motion sequences to evaluate persistent editing under realistic deformation scenarios (Sec. 4.3).

## 5.2 Simultaneous Geometry and Texture Editing

PointGT supports texture editing by modifying the 2D texture atlas, and geometry editing by applying non-rigid deformations to the point cloud while preserving canonical UV lookup via our edit-time canonical correspondence (Sec. 4.3). We evaluate this capability under non-rigid deformations and both local edits (e.g., paste an icon into a localized region) and global edits (e.g., recolor or material-style changes).

Non-rigid deformation with local and global edits. Figure 4 shows qualitative results on Objaverse assets with artist-created motion. PointGT preserves surface continuity under deformation and maintains edit attachment: the edited texture remains aligned with the deforming surface without noticeable drifting or popping.

Comparison to GSTex. We choose GSTex [16] as the baseline for simultaneous geometry and texture editing. While Texture-GS [27] learns a texture map, the quality of its learned texture map is insuficient for reliable texture-space editing in practice, with evidence in the supplementary. Textured Gaussians [3] is architecturally close to GSTex, they both learn a per-Gaussian texture map which is queried by ray–Gaussian intersection. Since texture editing is not its focus and its released code does not provide an editing pipeline, we treat GSTex as the representative method of this family. NeST-Splatting [30] stores texture in a global shell texture anchored in world space rather than to individual Gaussians, so deforming the geometry causes ray–Gaussian intersections to sample the texture at the wrong locations. It thus cannot edit geometry without external proxies such as cages [26,29], and it also does not release an editing pipeline. We provide the full analysis of both in the supplementary.

In Figure 5 we add a “rocket” icon to a butterfly scene from Objaverse [4]. We edit PointGT by pasting the icon into the desired region on the 2D texture map. GSTex performs texture editing by editing a rendered view and projecting the edited pixels back to Gaussians. To apply the same local edit to the same region for a fair comparison, we use the edited pixels from our method’s rendered view as the editing input for GSTex. As shown, our method maintains sharp edit details across diferent point budgets (1,000 to 30,000), while GSTex renders blurry textures, especially with 1,000 or 60,000 Gaussians. Moreover, as shown in Figure 4 our method maintains surface continuity after non-rigid deformation, while GSTex exhibits extruded Gaussians and surface artifacts under deformations.

![](images/0526941039b85943d1fb5de7a9ee87acf23dda6603a7bedc4b7a9ef3cfb89ac7.jpg)  
Fig. 4: Simultaneous geometry and texture editing under non-rigid deformation with both local and global texture edits.

Quantitative Evaluation. Due to the lack of ground truth, we use VBench [6] to quantitatively evaluate the editing quality of our method and GSTex. VBench is a benchmark suite that assesses generated videos along several dimensions, and is now widely used for video quality evaluation in general [7, 33]. Specifically, for each of the five scenes in Fig. 4, we render a 360<sup>◦</sup> orbit video in which the edited texture is shown while the geometry is animated over the video’s time span using their artist-created motions. As shown in Table 1, our method either outperforms or matches GSTex across all five dimensions.

Table 1: VBench evaluation for Figure 4. ∆% is the relative improvement.
<table><tr><td></td><td>Method subject consistency bkg consistency motion smooth aesthetic quality imaging quality</td><td></td><td></td><td></td><td></td></tr><tr><td>GSTex</td><td>0.827</td><td>0.923</td><td>0.985</td><td>0.434</td><td>0.520</td></tr><tr><td>Ours</td><td>0.844</td><td>0.925</td><td>0.982</td><td>0.437</td><td>0.585</td></tr><tr><td>Δ%</td><td>+2.16</td><td>+0.24</td><td>-0.40</td><td>+0.55</td><td>+12.48</td></tr></table>

![](images/fdc39425487ffe753f36c0f187d0a0dcdcdf0e2fd863ef897b6e741e9c9abd78.jpg)  
Fig. 5: Texture editing compared with GSTex [16] with diferent number of primitives.

Table 2: Quantitative comparison of Novel-view Synthesis.
<table><tr><td></td><td colspan="4">DTU [9]</td><td colspan="4">NeRF Synthetic [13]</td></tr><tr><td>Method</td><td>PSNR ↑</td><td></td><td>SSIM ↑ LPIPS</td><td></td><td>↓ # Pts PSNR ↑</td><td></td><td>SSIM ↑ LPIPS ↓ # Pts</td><td></td></tr><tr><td colspan="9">Authors&#x27; recommended primitive counts</td></tr><tr><td>Texture-GS [27]</td><td>30.53</td><td>0.920</td><td>0.083</td><td>90k</td><td>28.97</td><td>0.938</td><td>0.055</td><td>90k</td></tr><tr><td>GSTex [16]</td><td>32.87</td><td>0.956</td><td>0.038</td><td>186k</td><td>33.25</td><td>0.969</td><td>0.024</td><td>100k</td></tr><tr><td>Textured-Ġaussian [3]</td><td>33.61</td><td>0.970</td><td>0.056</td><td>240k</td><td>33.24</td><td>0.967</td><td>0.043</td><td>190k</td></tr><tr><td>NeST Splatting [30]</td><td>33.65</td><td>0.964</td><td>0.042</td><td>80k</td><td>33.37</td><td>0.967</td><td>0.032</td><td>73k</td></tr><tr><td>PAPR [31]</td><td>29.34</td><td>0.952</td><td>0.073</td><td>30k</td><td>32.07</td><td>0.971</td><td>0.038</td><td>30k</td></tr><tr><td>Ours</td><td>33.48</td><td>0.973</td><td>0.023</td><td>30k</td><td>33.57</td><td>0.982</td><td>0.021</td><td>30k</td></tr><tr><td colspan="9">Fixed primitive count (5k)</td></tr><tr><td>Texture-GS [27]</td><td>26.81</td><td>0.833</td><td>0.206</td><td>5k</td><td>19.29</td><td>0.795</td><td>0.213</td><td>5k</td></tr><tr><td>GSTex [16]</td><td>28.92</td><td>0.932</td><td>0.072</td><td>5k</td><td>30.20</td><td>0.897</td><td>0.149</td><td>5k</td></tr><tr><td>Textured-Ġaussian [3]</td><td>29.58</td><td>0.912</td><td>0.080</td><td>5k</td><td>26.21</td><td>0.919</td><td>0.086</td><td>5k</td></tr><tr><td>NeST Splatting [30]</td><td>32.68</td><td>0.966</td><td>0.056</td><td>5k</td><td>30.48</td><td>0.958</td><td>0.057</td><td>5k</td></tr><tr><td>PAPR [31]</td><td>24.87</td><td>0.846</td><td>0.040</td><td>5k</td><td>30.38</td><td>0.963</td><td>0.048</td><td>5k</td></tr><tr><td>Ours</td><td>33.25</td><td>0.970</td><td>0.024</td><td>5k</td><td>31.01</td><td>0.976</td><td>0.039</td><td>5k</td></tr></table>

## 5.3 Novel View Synthesis

We compare PointGT for novel view synthesis to PAPR [31], Texture-GS [27], GSTex [16], NeST-Splatting [30], and Textured Gaussians [3] on the Blender [13] dataset and the DTU dataset [5, 9]. Table 2 shows quantitative results for novel view synthesis on the Blender and DTU datasets. We compare against baselines in two settings: (1) the original settings reported in prior works (no cap on the number of primitives), and (2) capping all methods at 5,000 primitives. As shown in Table 2, our method outperforms baselines in both settings.

## 5.4 Ablation Study

Evaluation of the Geometry Regularizers We ablate the two regularizers described in Sec. 4.1.

On-ray regularizer. Figure 6 shows the ablation of the on-ray loss (close-to-ray). We visualize the predicted surface points from a camera view, the same points in

UV space (across 4 atlas maps), and crops of the final renderings using varying loss weights. Without the on-ray loss, the predicted surface points collapse and cluster around discrete support points. Consequently, querying the texture map with these points fails to smoothly interpolate the texture, degrading the efective texture resolution. With the on-ray loss, the surface points are continuous and smoothly interpolate the surface, yielding much clearer details in the renderings.

![](images/96d9964c18cb1ad4a5213433a09bde40253526282a493b31502a602ca6a2fbb2.jpg)  
λ<sub>on-ray</sub> = 0

![](images/9b4c9bb44215b450a293c4415d7fa26879c734a5669471582cf34e5595ae3325.jpg)  
λ<sub>on-ray</sub> = 0.001

![](images/97f29356cf113e50e9cf42bc5a8c0b3d8cb7e53b99ab04b16a68373d954ef6b2.jpg)  
λ<sub>on-ray</sub> = 0.005

![](images/c8e0489125eb1a362e01fcab2996f236c59282cdddfa21b3b0d419b15f64969a.jpg)  
λ<sub>on-ray</sub> = 0.01  
Fig. 6: Ablation of the on-ray regularizer with diferent weights. For each weight, we show the predicted surface points in 3D (left), the corresponding UV coordinates in the 2D atlas (middle), and the final rendered texture from texture maps (right). As shown, the on-ray loss ensures a better coverage of the surface, which leads to more uniform UV coordinates and gives a sharper texture in the renderings.

![](images/7c0b170dda216caf669c2e0a8eddc299684fa470f71cd7bb3ab5970a8de07c2d.jpg)  
w/o on-surface loss

![](images/b4dd09ac229f2d98e04d3d0446bb891a91b41a4242672436e34978b206eafa92.jpg)  
(a) The on-surface regularizer reduces noisy points and produces a smoother surface as shown in the depth maps.  
w/ on-surface loss

![](images/ffc33045d5e945e942b07268e5cefcfd50cd69ceac9dd95266ce4e74b18c2a99.jpg)  
baseline

![](images/430a7d391ba36af4b5890cf79215e634100a26eae1e4ab8af46990c41d9f38da.jpg)  
on-ray regularizer  
(b) Ablation of canonical correspondence. Naïve correspondence (Eq. 11) exhibits texture drift while our deformationaware correspondence (Eq. 13) improves stability.

![](images/15c6cbb2c147cf79947eeb03c128abc851f29b7ca8ab5868aade49ade30bcd3a.jpg)  
+ deform-aware corr.

Fig. 7: Ablation studies on geometry regularization and canonical correspondence.

On-surface regularizer. Figure 7a shows the ablation of the on-surface loss (closeto-surface). We visualize a depth map from a selected view alongside a side-view of the learned point cloud. Without the on-surface loss, the point cloud contains many noisy, of-surface points. Adding the on-surface loss significantly reduces these noisy points, resulting in a much smoother depth map with fewer holes.

Evaluation of Distortion Loss Figure 8 shows how the learned texture map depends on the weight of the proposed distortion loss in Sec. 4.2. Increasing the distortion-loss weight reduces UV distortion, which is crucial for preserving the shape of the edited texture.

Correspondence Stability Under Deformation We ablate the edit-time canonical correspondence used for UV lookup after deformation (Sec. 4.3). Specifically, we compare our deformation-aware canonical transfer (Eq. 13) against the naïve alternative that directly interpolates canonical neighbor positions using deformed-space attention weights (Eq. 11). Figure 7b shows that the naïve transfer leads to discontinuous UV jumps and visible texture drift/popping under non-rigid deformations, while our transfer better preserves correspondence and keeps texture edits attached.

![](images/699adf7efb6ad00c12827fc438e266db81d943f5d64f2f0e663bc03850529de4.jpg)  
Fig. 8: Learned UV mapping under diferent weights of the proposed distortion loss.

## 6 Conclusion and Limitation

We presented PointGT, a representation based on PAPR that enables simultaneous editing of geometry and texture for point-based representations. PointGT successfully achieves this by learning a UV mapping which maps 3D points to a learned 2D texture map. To improve geometry accuracy, we introduced two novel geometry regularizers, which have been demonstrated to improve the geometry significantly. We also introduced a novel correspondence mechanism to maintain UV mapping under deformation, which has been demonstrated to be highly efective. Comparison with baselines showed that PointGT preserves texture edits under large deformations while the baseline exhibits artifacts and surface discontinuities. Furthermore, we showed that PointGT outperforms competitive baselines on novel view synthesis with a much smaller number of primitives. Our method currently relies on an optimization-based UV parameterization [20], which can struggle to produce clean and low-distortion charts for objects with complex topology, which we believe could be alleviated by learning-based UVmapping methods [11, 12] in the future.

## 7 Acknowledgments

This research was enabled in part by support provided by NSERC, the BC DRI Group, the Digital Research Alliance of Canada and the Canada CIFAR AI Chairs Program.

## References

1. Barron, J.T., Mildenhall, B., Verbin, D., Srinivasan, P.P., Hedman, P.: Mip-nerf 360: Unbounded anti-aliased neural radiance fields. 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) pp. 5460–5469 (2021)

2. Chang, J.H.R., Chen, W.Y., Ranjan, A., Yi, K.M., Tuzel, O.: Pointersect: Neural rendering with cloud-ray intersection. 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) pp. 8359–8369 (2023)

3. Chao, B., Tseng, H.Y., Porzi, L., Gao, C., Li, T., Li, Q., Saraf, A., Huang, J.B., Kopf, J., Wetzstein, G., Kim, C.: Textured gaussians for enhanced 3d scene appearance modeling. 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) pp. 8964–8974 (2024)

4. Deitke, M., Schwenk, D., Salvador, J., Weihs, L., Michel, O., VanderBilt, E., Schmidt, L., Ehsani, K., Kembhavi, A., Farhadi, A.: Objaverse: A universe of annotated 3d objects. 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) pp. 13142–13153 (2022)

5. Huang, B., Yu, Z., Chen, A., Geiger, A., Gao, S.: 2d gaussian splatting for geometrically accurate radiance fields. ACM SIGGRAPH 2024 Conference Papers (2024)

6. Huang, Z., He, Y., Yu, J., Zhang, F., Si, C., Jiang, Y., Zhang, Y., Wu, T., Jin, Q., Chanpaisit, N., Wang, Y., Chen, X., Wang, L., Lin, D., Qiao, Y., Liu, Z.: VBench: Comprehensive benchmark suite for video generative models. In: CVPR (2024)

7. Huang, Z., Zhang, F., Xu, X., He, Y., Yu, J., Dong, Z., Ma, Q., Chanpaisit, N., Si, C., Jiang, Y., Wang, Y., Chen, X., Chen, Y.C., Wang, L., Lin, D., Qiao, Y., Liu, Z.: VBench++: Comprehensive and versatile benchmark suite for video generative models. IEEE Transactions on Pattern Analysis and Machine Intelligence (2025)

8. Jambon, C., Kerbl, B., Kopanas, G., Diolatzis, S., Leimkühler, T., Drettakis, G.: Nerfshop. Proceedings of the ACM on Computer Graphics and Interactive Techniques 6, 1 – 21 (2023)

9. Jensen, R.R., Dahl, A., Vogiatzis, G., Tola, E., Aanæs, H.: Large scale multi-view stereopsis evaluation. 2014 IEEE Conference on Computer Vision and Pattern Recognition pp. 406–413 (2014)

10. Kerbl, B., Kopanas, G., Leimkuehler, T., Drettakis, G.: 3d gaussian splatting for real-time radiance field rendering. ACM Transactions on Graphics (TOG) 42, 1 – 14 (2023)

11. Li, Y., Cheung, V., Liu, X., Chen, Y., Luo, Z., Lei, B., Weng, H., Zhao, Z., Huang, J., Chen, Z., et al.: Auto-regressive surface cutting. arXiv preprint arXiv:2506.18017 (2025)

12. Ma, X., Yan, X., Zhang, C., Huang, H.: Meshtailor: Cutting seams via generative mesh traversal (2026)

13. Mildenhall, B., Srinivasan, P.P., Tancik, M., Barron, J.T., Ramamoorthi, R., Ng, R.: NeRF: Representing scenes as neural radiance fields for view synthesis. Commun. ACM 65, 99–106 (2020)

14. Ost, J., Laradji, I., Newell, A., Bahat, Y., Heide, F.: Neural point light fields. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 18419–18429 (2022)

15. Peng, Y., Yan, Y., Liu, S., Cheng, Y., Guan, S., Pan, B., Zhai, G., Yang, X.: Cagenerf: Cage-based neural radiance field for generalized 3d deformation and animation. In: Neural Information Processing Systems (2022)

16. Rong, V., Chen, J., Bahmani, S., Kutulakos, K.N., Lindell, D.B.: Gstex: Perprimitive texturing of 2d gaussian splatting for decoupled appearance and geometry modeling. 2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) pp. 3508–3518 (2024)

17. Ronneberger, O., Fischer, P., Brox, T.: U-net: Convolutional networks for biomedical image segmentation. ArXiv abs/1505.04597 (2015)

18. Song, Y., Lin, H., Lei, J., Liu, L., Daniilidis, K.: Hdgs: Textured 2d gaussian splatting for enhanced scene rendering. ArXiv abs/2412.01823 (2024)

19. Sorkine, O., Alexa, M.: As-rigid-as-possible surface modeling. In: Symposium on Geometry processing. vol. 4, pp. 109–116. Citeseer (2007)

20. Srinivasan, P.P., Garbin, S.J., Verbin, D., Barron, J.T., Mildenhall, B.: Nuvo: Neural uv mapping for unruly 3d representations. ArXiv abs/2312.05283 (2023)

21. Sumner, R.W., Schmid, J., Pauly, M.: Embedded deformation for shape manipulation. ACM SIGGRAPH 2007 papers (2007)

22. Svitov, D., Morerio, P., de Agapito, L., Bue, A.D.: Billboard splatting (bbsplat): Learnable textured primitives for novel view synthesis. ArXiv abs/2411.08508 (2024)

23. Tang, K., Ai, K., Han, J., Wang, C.: Texgs-volvis: Expressive scene editing for volume visualization via textured gaussian splatting. ArXiv abs/2507.13586 (2025)

24. Wang, C., He, M., Chai, M., Chen, D., Liao, J.: Mesh-guided neural implicit field editing. ArXiv abs/2312.02157 (2023)

25. Xiang, F., Xu, Z., Havsan, M., Hold-Geofroy, Y., Sunkavalli, K., Su, H.: Neutex: Neural texture mapping for volumetric neural rendering. 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) pp. 7115–7124 (2021)

26. Xu, T., Harada, T.: Deforming radiance fields with cages. ArXiv abs/2207.12298 (2022)

27. Xu, T., Hu, W., Lai, Y.K., Shan, Y., Zhang, S.: Texture-gs: Disentangling the geometry and texture for 3d gaussian splatting editing. In: European Conference on Computer Vision (2024)

28. Yang, B., Bao, C., Zeng, J., Bao, H., Zhang, Y., Cui, Z., Zhang, G.: Neumesh: Learning disentangled neural mesh-based implicit field for geometry and texture editing. ArXiv abs/2207.11911 (2022)

29. Yuan, Y.J., Sun, Y.T., Lai, Y.K., Ma, Y., Jia, R., Gao, L.: Nerf-editing: Geometry editing of neural radiance fields. 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) pp. 18332–18343 (2022)

30. Zhang, X., Chen, A., Xiong, J., Dai, P., Shen, Y., Xu, W.: Neural shell texture splatting: More details and fewer primitives. ArXiv abs/2507.20200 (2025)

31. Zhang, Y., Peng, S., Moazenipourasil, S.A., Li, K.: PAPR: Proximity attention point rendering. In: Thirty-seventh Conference on Neural Information Processing Systems (2023)

32. Zheng, C., chang Lin, W., Xu, F.: Editablenerf: Editing topologically varying neural radiance fields by key points. 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) pp. 8317–8327 (2022)

33. Zheng, D., Huang, Z., Liu, H., Zou, K., He, Y., Zhang, F., Zhang, Y., He, J., Zheng, W.S., Qiao, Y., Liu, Z.: VBench-2.0: Advancing video generation benchmark suite for intrinsic faithfulness. arXiv preprint arXiv:2503.21755 (2025)

34. Zhou, K., Hong, L., Xie, E., Yang, Y., Li, Z., Zhang, W.: Serf: Fine-grained interactive 3d segmentation and editing with radiance fields. ArXiv abs/2312.15856 (2023)

## A Multi-View Editing Results

We show our simultaneous geometry and texture editing results rendered from multiple viewpoints. For each of the five Objaverse scenes in Figure 4, we render a 360<sup>◦</sup> orbit of the edited and non-rigidly deformed object—the same orbit renderings used for the VBench evaluation in Table 1—and show five evenly spaced views in Figure 9. Across all viewpoints, the edited texture stays sharp and remains firmly attached to the deformed surface, with no visible seams, drifting, or loss of resolution, confirming that PointGT preserves both view consistency and edit attachment under large non-rigid deformation.

![](images/bf930d0e0e0f5e427cecfa95bce86ad12d50e4c9503d3d6e64995bdbf8e9c236.jpg)  
Fig. 9: Novel views of our simultaneous geometry and texture editing results. For each scene, we render a 360<sup>◦</sup> orbit of the edited and non-rigidly deformed object and show five evenly spaced viewpoints. The edited texture stays sharp and remains consistently attached to the deforming surface across all views.

## A.1 Part Removal and Duplication

Because PointGT decouples geometry and appearance through a canonical UV mapping, it also supports editing operations that change the object’s topology, while keeping the edited texture correctly attached to the modified geometry. We demonstrate this on the Blossom scene with two operations. Figure 10 shows part removal: petals are sequentially detached from the flower (top to bottom). The learned UV mapping (left, visualized as a checkerboard with one color per chart) and the rendered result together with its 2D texture map (right) stay consistent throughout; since appearance is decoupled, the geometry can be retextured at the same time, and we switch the applied texture across the sequence. Figure 11 shows part duplication, where the duplicated geometry is rendered under two diferent texture edits, with the texture remaining correctly mapped onto the duplicated parts through the canonical UV mapping.

![](images/b45eaacce24a11bdb7a703b0e312b0733b2a7d633c96c254fc38d775fa0d1d31.jpg)  
Fig. 10: Part removal on the Blossom scene. From top to bottom, petals are sequentially removed. UV mapping: checkerboard visualization of the learned charts on the deforming geometry. Rendering and texture map: the rendered result (left) and its 2D texture map (right); the applied texture is switched across the sequence to show that the remaining geometry stays editable.

![](images/b71b7149d2a1b6f9dae02462a51c3c2386a255b2fbe846aaf6761472d5f5c8fa.jpg)  
Fig. 11: Part duplication on the Blossom scene. The duplicated geometry is shown under two diferent texture edits. UV mapping: checkerboard visualization of the learned charts; Edited texture: the rendered result; Texture map: the corresponding 2D texture map. The texture stays correctly attached to the duplicated parts through the canonical UV mapping.

## B Additional Ablation Results

We show ablation results for the proposed on-ray regularizer 5 and the deformationaware correspondence 13 in Figure 12, in addition to Figure 7b. As shown, the proposed on-ray regularizer significantly reduces the artifacts of the texture editing, and the deformation-aware correspondence efectively boosts the editing fidelity.

## C Discussion of Other Gaussian Splatting Baselines

As noted in the main paper, we choose GSTex [16] as the primary baseline for simultaneous geometry and texture editing. Here we provide additional discussion and evidence for why the other baselines are limited in their ability to perform simultaneous geometry and texture editing.

Texture-GS [27] supports texture editing through a single-chart global texture map. However, this representation often struggles with complex geometries, resulting in texture maps that are insuficient for reliable texture-space editing in practice (see Figure 13). For a fair comparison, Figure 14 shows the comparison of simultaneous geometry and texture editing results on the ‘Butterfly’ scene from Objaverse, where Texture-GS successfully generates a valid map. As shown, unlike Texture-GS, which sufers from broken surface continuity and loss of resolution after deformation, our method ensures that texture edits remain sharp and continuous.

edited scene  
(1) baseline  
(2) + on-ray regularizer  
(3) + deform-aware corr.  
![](images/875e04990dc210c9bab39cdd1a30c21b5541339c04c72ea38c9d313c3adc2010.jpg)  
Fig. 12: Additional texture editing results. We show both edited scene and close-up views of two diferent regions of the scene. As shown, the proposed on-ray regularizer and deformation-aware correspondence maintain the high-fidelity.

Textured Gaussians [3] adopts the same core idea as GSTex [16]: both methods attach a learnable per-Gaussian texture map to each Gaussian primitive and query it via ray–Gaussian intersection with UV mapping defined by the Gaussian’s local coordinate axes. Both also decompose appearance into a spatially varying texture and spatially constant spherical harmonics, and both follow a two-stage optimization that first trains a standard Gaussian model before adding textures. The main diferences are that Textured Gaussians builds on 3D Gaussians rather than 2D, uses a fixed texture resolution across all primitives (instead of adapting the resolution to each Gaussian’s scale), and optionally includes an alpha channel in the texture map. Given the strong architectural similarity and the fact that texture editing is not the main focus of Textured Gaussians (their released codebase does not include an editing pipeline), we compare against GSTex as a representative method from this family.

NeST-Splatting [30] decouples geometry and appearance by replacing per-Gaussian spherical harmonics with a global shell texture, parameterized as a multi-resolution hash grid encoding. During rendering, each ray–Gaussian intersection is queried against the hash grid in world space to obtain textures. While this design yields a compact representation that achieves high-fidelity rendering with fewer primitives, the hash grid remains anchored in world space rather than attached to individual Gaussians. Consequently, displacing or deforming the Gaussian geometry causes the ray-Gaussian intersections to sample the hash grid at the wrong locations. NeST-Splatting also does not provide an editable texture pipeline in their released codebase. As a result, NeST-Splatting is unable to perform geometry editing without the use of additional proxies, such as cages [26] or explicit meshes [29], to establish a correspondence between the deformed and canonical space.

Learned Texture Map from Texture-GS  
![](images/c141bb829fee926a496ea894b8f686f025c00c318f2131fe2dcb6b9883a2312b.jpg)  
Fig. 13: Texture-GS [27] may fail to learn a reasonable texture map that is valid for texture editing.

## D Additional Novel View Synthesis Results

We show qualitative novel view synthesis comparison using either 5,000 primitives or their originally suggested number of primitives for each method in Figure 15 and Figure 16. As shown, our method achieves rendering quality comparable to or better than the baselines while using only 30,000 primitives.

We further evaluate PointGT for novel view synthesis on the real-world scenes of the MipNeRF360 dataset [1]. Table 3 reports per-scene PSNR, SSIM, and LPIPS. We compare against the 3D Gaussian Splatting [10], the textured Gaussian baselines, and the PAPR [31]. As full resolution training exceeds the memory of our GPUs, we evaluate both PAPR and PointGT at half the resolution of the MipNeRF360 dataset. The numbers for the Gaussian splatting-based baselines are from their original papers.

![](images/fc5e29f63ebd57db98526f10fd3bf6b73d5c013a3536c7a1540f5cc674e69492.jpg)  
Fig. 14: Simultaneous geometry and texture editing comparisons with Texture-GS [27] on the Butterfly from Objaverse [4]. We use the original settings of Texture-GS without capping the number of Gaussians.

![](images/fb9b02272804dff16b964480f40866ea2250fa2539348059e5aaf193ea2eb713.jpg)  
Fig. 15: Qualitative comparison of novel view synthesis with 5,000 primitives. As shown, our method achieves better rendering fidelity with the same number of points.

![](images/fb33828aa8a484951d07b1522e3a948ed6c784d19b09c38c425d7da046bf0c55.jpg)  
Fig. 16: Qualitative comparison of novel view synthesis with the original number of primitives proposed by the authors.

Table 3: Quantitative novel view synthesis on the MipNeRF360 [1] scenes. ‡ Both PAPR and PointGT were trained and rendered at the half-resolution of the dataset, as full resolution evaluation exceeds our GPU memory.
<table><tr><td>Method</td><td>bicycle</td><td>counter</td><td>flowers</td><td>room</td><td>stump</td><td>Mean</td></tr><tr><td>PSNR ↑</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>3DGS [10]</td><td>24.71</td><td>28.96</td><td>21.09</td><td>31.50</td><td>26.45</td><td>26.54</td></tr><tr><td>GSTex [16]</td><td>24.68</td><td>28.50</td><td>21.17</td><td>31.15</td><td>26.24</td><td>26.35</td></tr><tr><td>Textured Gaussians [3]</td><td></td><td></td><td></td><td></td><td></td><td>27.35</td></tr><tr><td>NeST-Splatting [30]</td><td>24.49</td><td>28.45</td><td>20.05</td><td>31.30</td><td>25.87</td><td>26.03</td></tr><tr><td>PAPR [31]</td><td>24.21</td><td>27.13</td><td></td><td>29.36</td><td>26.23</td><td>26.73</td></tr><tr><td>PointGT (Ours)‡</td><td>26.61</td><td>28.58</td><td>22.40</td><td>34.66</td><td>27.96</td><td>28.04</td></tr><tr><td>SSIM ↑</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>3DGS [10]</td><td>0.729</td><td>0.905</td><td>0.571</td><td>0.915</td><td>0.762</td><td>0.776</td></tr><tr><td>GSTex [16]</td><td>0.730</td><td>0.896</td><td>0.582</td><td>0.910</td><td>0.758</td><td>0.775</td></tr><tr><td>Textured Ġaussians [3]</td><td></td><td></td><td></td><td></td><td></td><td>0.827</td></tr><tr><td>NeST-Splatting [30]</td><td>0.729</td><td>0.888</td><td>0.521</td><td>0.909</td><td>0.737</td><td>0.757</td></tr><tr><td>PAPR [31]</td><td>0.736</td><td>0.838</td><td></td><td>0.822</td><td>0.818</td><td>0.804</td></tr><tr><td>PointGT (Ours)‡</td><td>0.757</td><td>0.850</td><td>0.640</td><td>0.962</td><td>0.832</td><td>0.808</td></tr><tr><td>LPIPS ↓</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>3DGS [10]]</td><td>0.265</td><td>0.201</td><td>0.377</td><td>0.219</td><td>0.266</td><td>0.266</td></tr><tr><td>GSTex [16]</td><td>0.265</td><td>0.221</td><td>0.365</td><td>0.237</td><td>0.248</td><td>0.267</td></tr><tr><td>Textured Ġaussians [3]</td><td></td><td></td><td></td><td></td><td></td><td>0.186</td></tr><tr><td>NeST-Splatting [30]</td><td>0.236</td><td>0.202</td><td>0.358</td><td>0.194</td><td>0.246</td><td>0.247</td></tr><tr><td>PAPR [31]</td><td>0.201</td><td>0.226</td><td></td><td>0.209</td><td>0.202</td><td>0.210</td></tr><tr><td>PointGT (Ours)‡</td><td>0.199</td><td>0.184</td><td>0.303</td><td>0.049</td><td>0.182</td><td>0.183</td></tr></table>