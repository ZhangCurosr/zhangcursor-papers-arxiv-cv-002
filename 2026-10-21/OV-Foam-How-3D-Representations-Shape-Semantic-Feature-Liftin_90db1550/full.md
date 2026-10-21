# OV-Foam: How 3D Representations Shape Semantic Feature Lifting

Yousef Rajeh<sup>1,∗</sup> Alejandra Esparza Pelcastre<sup>2</sup> Ivan Viola<sup>1</sup> Dominik Engel<sup>1RGB</sup> <sup>appearance</sup>

<sup>1</sup>King Abdullah University of Science and Technology (KAUST) <sup>2</sup>Universidad Panamericana {yousef.rajeh, ivan.viola, dominik.engel}@kaust.edu.sa 0263483@up.edu.mx <sup>∗</sup>Corresponding author

## Abstract

Existing feature lifting approaches for open-vocabulary segmentation rely on simple backprojection to assign features to primitives. In the case of 3D Gaussian splatting reconstruction, the primitives overlap heavily, leading to suboptimal reconstructions, governed by many primitives per ray. Backprojection, however, reduces to a least-squares solvable problem, exactly when no two primitives share a ray.Feature PCA (shared basis) Motivated by this insight, we propose OV-Foam, which uses PowerFoam as the underlying representation. This representation uses a power-diagram to partition the scene, resulting in zero overlap between primitives and a substantial reduction in primitives per ray, which ultimately simplifies the lifting problembasis) and allows for closer-to-optimal feature assignment. We show that the space partitioning primitives of OV-Foam reach near-optimal assignment, and our experiments demonstrate improved open-vocabulary segmentation over Gaussian baselines.<sup>OV-Foam</sup>

![](images/c531b283259c14dd8d182689b79986dc23c70fa38a2be1a50a710b944bb2dfa7.jpg)

![](images/02264b604e0fbc6b5839bd052735db699c3c6703be81c3743b330bf8bf6ff603.jpg)

![](images/6a71c2ac7a6f4568237aaeeedc18dd6fd854cf2028050a611fef9b1c286e3fef.jpg)

![](images/517b1b4a8dbd7291f24802107d0b71213c57362e1ca4b05882b1619017ded276.jpg)

![](images/0b4f5b753e6b7836a4f34a2389a4cd51d88bb37fb2ad416ffb2e77caa76a1925.jpg)

![](images/edf3c64e769b7fffe27eeb50cfa9a3a9bebfdb56137a5805f357612841238a3b.jpg)

![](images/24b75d4aa6c8e88b33575a7d103ded3476c549407719c8298385dc1ebd73db4a.jpg)

![](images/c7f239687b0e541517e53c1d89eb6d3b3408a964a76a2620c78ce6df857b928b.jpg)

![](images/32e56ae02587c62965ff61c3fc2b07e675d961732d23d943b98e98a8e99bdda9.jpg)  
Figure 1: Pipeline overview and qualitative example. Starting from the input views, we create SAM masks, which are fed to CLIP to create image features B. The views also reconstruct our space-partitioning PowerFoam scene, of which we then use the rendering weights A for the back projection and assign each primitive a queryable feature x<sup>′</sup> that is compared to the text features.

## 1 Introduction

Open-vocabulary 3D segmentation assigns vision-language aligned semantic features, such as those produced by CLIP (Radford et al., 2021b), to reconstructed scenes, allowing objects and spatial regions to be identified using free-form language queries. This capability supports applications such as robotic manipulation (Gu et al., 2024), complex scene understanding (Kerr et al., 2023) and medical-imaging (Wang et al., 2025). Existing approaches include grouping-based methods, training-based methods, and heuristic-based methods. We focus on the latter family, which directly lifts features from 2D images onto a 3D representation.

The process of assigning these features to scene geometry is known as feature lifting. It associates features observed in rendered images with the primitives contributing to each pixel, using their rendering weights to aggregate information across views. This provides an efficient way to obtain semantic representations without training a 3D foundation model from scratch, but introduces a fundamental ambiguity: a 2D observation may be distributed across multiple primitives, while a primitive may collect observations from different objects. As a result, feature lifting can entangle semantic evidence before a classifier makes a prediction.

This ambiguity is particularly important near object boundaries. Repeating the lifting process across views can produce feature entanglement and degraded semantic boundaries, even when the underlying image features are accurate. Recent work has begun to formalize this problem by treating feature lifting as a linear inverse problem and quantifying the ambiguity introduced by multiple primitive contributions (Xiong et al., 2026).

Existing feature-lifting methods largely operate on Gaussian-based reconstructions, where overlap leads to multiple primitives contributing to the same ray. This raises a more fundamental question: to what extent is feature-lifting ambiguity a property of the lifting procedure itself, and to what extent is it induced by the underlying 3D representation?

In this work, we investigate feature lifting as a representation-dependent problem. We ask whether changing the geometric primitives used to represent the scene can reduce the mixing of semantic evidence, while keeping the image features and lifting procedure fixed. We consider PowerFoam (Govindarajan et al., 2026), which represents a scene using bounded power diagrams whose cells partition 3D space. This non-overlapping spatial support reduces the number of primitives that can contribute simultaneously to a ray, reducing semantic feature mixing.

We isolate this effect, by evaluating feature lifting independently of errors introduced by the upstream 2D semantic model. We replace predicted image semantics with ground-truth segmentation and text features, allowing us to measure the extent of semantic feature mixing caused by a feature lifting onto a chosen underlying 3D representation. Under this controlled setting, we compare Gaussian-based and PowerFoam reconstructions using the same lifting procedure and find that the reduced dispersion in PowerFoam leads to substantially more class-consistency in lifted features. This demonstrates that the choice of 3D primitive can itself impose a significant limitation on semantic feature lifting. Applied to real datasets, we observe that all our tested lifting methods achieve improved 3D open-vocabulary segmentation results using PowerFoam instead of 3DGS as underlying representation.

Our contributions are:

• A controlled study of representation-dependent feature lifting. We isolate the feature-lifting stage from upstream semantic errors and systematically compare 3D primitives under the same image features and lifting procedure.

• An analysis of representation-induced feature dispersion. We show that space-partitioning primitives can substantially reduce semantic feature mixing compared with overlapping Gaussian primitives.

• An open-vocabulary 3D segmentation system based on PowerFoam. We incorporate these insights into a PowerFoam-based semantic representation and demonstrate improved performance on openvocabulary 3D segmentation benchmarks.

## 2 Related work

Existing approaches for lifting 2D features to 3D primitives differ primarily in how the correspondence between image observations and 3D primitives is established: through training-based, grouping-based, and heuristic-forward methods.

Early approaches fit continuous 3D semantic fields via scene-specific optimization. LERF, for example, distills CLIP features into a radiance field (Kerr et al., 2023; Radford et al., 2021b), while LangSplat trains a scene-specific autoencoder to represent lifted features in a latent space (Qin et al., 2024). OpenGaussian similarly learns instance-level features together with a codebook for point-level understanding (Wu et al., 2024). These methods can produce expressive semantic representations, but require computationally expensive per-scene optimization and may inherit inconsistencies from the underlying multi-view 2D predictions.

An alternative is replacing continuous optimization with discrete associations derived from 2D masks (e.g., SAM (Kirillov et al., 2023)). LAGA decomposes scenes into object entities by clustering SAM mask descriptors across views (Cen et al., 2025), while THGS constructs Gaussian adjacency graphs with contrastive edge reweighting for hierarchical superpoint partitioning (Dai et al., 2025). LUDVIG avoids training altogether by combining multi-view weighted feature averaging with graph diffusion across adjacent primitives (Marrie et al., 2025). Although these approaches eliminate dense feature optimization, they introduce multi-stage grouping heuristics and remain sensitive to primitive spatial overlap.

More recent methods avoid additional optimization by directly inverting the image-to-primitive aggregation used during rendering. Occam’s LGS (Cheng et al., 2025), CosegGaussians (Dou et al., 2025), and DrSplat (Jun-Seong et al., 2025) independently use row-normalized splatting equations as zero-shot feature lifter.

Splat Feature Solver (SFS) (Xiong et al., 2026) formulates feature lifting as a sparse linear inverse problem and provides a closed-form row-sum estimator with a provable approximation bound. Specifically, SFS defines β as a measure of the dispersion of the optimal lifted feature along a viewing ray. In the case where each ray intersects only a single primitive, each row of the system matrix contains a single non-zero entry of value one, yielding this zerodispersion regime. Standard 3D Gaussians, however, have substantial spatial overlap, causing multiple primitives to contribute to the same ray and increasing feature dispersion (Figure 2). SFS therefore introduces Tikhonov Guidance, which modifies the opacity weights during lifting to increase diagonal dominance and reduce this coupling. Thus, while SFS provides a principled formulation and efficient solver, its lifting accuracy remains tied to the overlap structure of the underlying Gaussian primitives.

An alternative to mitigating primitive overlap through the lifting procedure is to eliminate overlap at the representation level. Space-partitioning representations divide the scene into disjoint spatial cells, so each 3D location is associated with a single primitive by construction. Radiant Foam (Govindarajan et al., 2025) represents a scene using

![](images/c607f95e4cc13bdc29285fcc655ff822f97326cff0f9dd3840d55749554d4639.jpg)  
Figure 2: Illustration of ray contributions. Disjoint cells with concentrated opacity and overlapping Gaussians. Middle plots show cumulative contributor counts.

non-overlapping volumetric cells based on an unbounded Voronoi construction. However, its ray-tracing procedure relies on explicit cell stepping rather than tile-based rasterization, making it difficult to evaluate under the same rendering pipeline as standard 3D Gaussian Splatting. PowerFoam (Govindarajan et al., 2026)

instead generalizes foam representations to bounded power diagrams, defining finite spatial cells through seed sites and per-cell radii. These bounded cells support tile-based rasterization while preserving disjoint spatial boundaries, making PowerFoam a suitable representation for studying feature lifting under otherwise comparable rendering conditions.

Semantic Foam (Sharafeldin et al., 2026) jointly learns reconstruction and object-identity features supervised by segmentation masks. These features encode scene-specific identities rather than text-aligned semantics, so they cannot be directly compared with text embeddings for open-vocabulary queries. In contrast, we study feature transfer from fixed 2D observations onto an already trained reconstruction, allowing the effect of the underlying representation on lifting to be isolated from semantic optimization. Building on this setting, we introduce OV-Foam, an open-vocabulary 3D segmentation framework that lifts 2D vision-language features onto PowerFoam reconstructions.

## 3 Method

OV-Foam lifts image features onto a PowerFoam reconstruction. We first describe how the cells define rendering contributions, then use those contributions in the SFS feature estimator. We then examine how image observations are distributed across primitives and whether each primitive receives consistent features from one class or conflicting features from several classes.

## 3.1 PowerFoam

PowerFoam represents a scene with a power diagram (Govindarajan et al., 2026). Each primitive $j$ is associated with a center $\mathbf { p } _ { j }$ and radius $a _ { j }$ . For a 3D point x, its power cell $V _ { j }$ is the region where primitive j has the smallest power distance among all primitives k. The rendering support $C _ { j }$ further restricts this cell to the primitive’s bounding sphere:

$$
\begin{array} { r l } & { V _ { j } = \{ x : \| x - \mathbf { p } _ { j } \| ^ { 2 } - a _ { j } ^ { 2 } \leq \| x - \mathbf { p } _ { k } \| ^ { 2 } - a _ { k } ^ { 2 } \mathrm { f o r } \mathrm { a l l } k \} , } \\ & { C _ { j } = V _ { j } \cap \mathrm { S p h e r e } ( \mathbf { p } _ { j } , a _ { j } ) . } \end{array}\tag{1}
$$

The power cells partition space, so their interiors do not overlap, although a ray can traverse several cells. Each primitive is bounded to $C _ { j }$ , so only the portion of its cell inside the bounding sphere can contribute to rendering.

Within each support $C _ { j }$ , a dipole defines an oriented boundary separating occupied space from empty space. The dipole provides a planar reference boundary, where occupied space satisfies $\left( \mathbf { x } - \mathbf { p } _ { j } \right) \cdot \mathbf { n } _ { j } < 0$ with $\mathbf { n } _ { j }$ denoting the outward unit normal and $\mathbf { p } _ { j }$ the primitive center. A displacement field can deform this boundary away from the reference plane. For a given ray i, PowerFoam determines the resulting intersections with both the support $C _ { j }$ and the occupied region. These intersections define the occupied ray segment and its length $\ell _ { i j }$ . Density is constant within the occupied region and zero elsewhere. During reconstruction, primitive centers, radii, densities, dipoles, and enabled displacement parameters are optimized.

For a ray $r _ { i } ( t )$ parameterized by distance, volume rendering of a continuous 3D feature field $f : \mathbb { R } ^ { 3 } \to \mathbb { R } ^ { F }$ between near and far limits $t _ { n }$ and $t _ { f }$ gives:

$$
\mathcal F _ { i } = \int _ { t _ { n } } ^ { t _ { f } } T _ { i } ( t ) \sigma ( r _ { i } ( t ) ) f ( r _ { i } ( t ) ) d t , \qquad T _ { i } ( t ) = \exp \left( - \int _ { t _ { n } } ^ { t } \sigma ( r _ { i } ( s ) ) d s \right) ,\tag{2}
$$

where ${ \mathcal { F } } _ { i }$ is the rendered feature for ray i, and $T _ { i } ( t )$ is the transmittance accumulated before distance t. PowerFoam discretizes $f ( \mathbf { x } )$ as a piecewise-constant field across primitives, where $f ( \mathbf { x } ) = x _ { j } \in \mathbb { R } ^ { F }$ within the occupied region of primitive j and zero elsewhere. Because each primitive has constant density within its occupied region and carries a single feature $x _ { j }$ , the contribution of each occupied ray segment can be expressed through its length $\ell _ { i j }$ . For primitive $j ,$ , a segment of length $\ell _ { i j }$ and density $\sigma _ { j }$ has opacity $\alpha _ { i j } = 1 - \exp ( - \sigma _ { j } \ell _ { i j } )$ . For the occupied segment of each primitive, the corresponding transmittance and opacity give the front-to-back weights:

$$
A _ { i j } = T _ { i j } \alpha _ { i j } , \qquad T _ { i j } = \prod _ { k \prec _ { i j } } ( 1 - \alpha _ { i k } ) , \qquad { \mathcal { F } } _ { i } = \sum _ { j } A _ { i j } x _ { j } , \quad s _ { i } = \sum _ { j } A _ { i j } \leq 1 .\tag{3}
$$

Here $k \prec _ { i } j$ denotes a primitive encountered before $j$ along ray i, and the background feature is zero. The total scene contribution can equivalently be written as $s _ { i } = 1 - \prod _ { j } ( 1 - \alpha _ { i j } )$ , with the remaining $1 - s _ { i }$ reaching the background. Thus, $A _ { i j }$ measures the contribution of primitive j to pixel $i ,$ after absorption by earlier primitives. A shorter occupied segment reduces $\alpha _ { i j }$ , while opacity accumulated earlier along the ray reduces $T _ { i j }$

Gaussian splatting uses the same front-to-back compositing formulation, but obtains $\alpha _ { i j }$ from the opacity and projected footprint of each Gaussian (Kerbl et al., 2023). PowerFoam instead integrates density through bounded occupied segments. Both renderers therefore provide weights $A _ { i j }$ for feature lifting, despite their different geometry. Spatial partitioning prevents overlapping occupied interiors, but contributions from different primitives can still accumulate along the same ray, depending on the reconstructed surface and density.

## 3.2 Feature lifting

Let $B \in \mathbb { R } ^ { R \times F }$ contain F feature values for each of the $R$ rays, and let $X \in \mathbb { R } ^ { P \times F }$ contain the unknown features of $P$ primitives. With the reconstruction fixed, the renderer defines a nonnegative contribution matrix $A \in \mathbb { R } ^ { R \times P }$ , whose entries $A _ { i j }$ describe how much primitive $j$ contributes to ray i. Rendering the primitive features X with these contributions produces the image features $A X$ , which can be compared to the observed features B. SFS therefore formulates feature recovery as a least-squares inverse problem (Xiong et al., 2026):

$$
{ \widehat { X } } = \underset { X } { \arg \operatorname* { m i n } } \mathcal { L } ( X ) , \qquad \mathcal { L } ( X ) = \frac { 1 } { 2 } \| A X - B \| ^ { 2 } .\tag{4}
$$

The minimum-norm least-squares is $A ^ { \dagger } B .$ , where $A ^ { \dagger }$ is the Moore–Penrose pseudoinverse. This solution minimizes the residual, but does not necessarily reproduce B exactly. We use $\hat { \boldsymbol { X } } = \boldsymbol { A } ^ { \dagger } \boldsymbol { B }$ to define dispersion in Section 3.3. The SFS one-shot approximation instead estimates each primitive feature by averaging the observations it receives:

$$
x _ { j } ^ { \prime } = \frac { \sum _ { i } A _ { i j } B _ { i } } { \sum _ { i } A _ { i j } } , \qquad X ^ { \prime } = D ^ { - 1 } A ^ { \top } B , \qquad D = \mathrm { d i a g } ( A ^ { \top } { \bf 1 } ) .\tag{5}
$$

This expression applies to primitives with positive total contribution. Here $\begin{array} { r } { D _ { j j } = \sum _ { i } A _ { i j } } \end{array}$ is that total and $\mathbf { 1 } \in \mathbb { R } ^ { R }$ is the all-ones vector. The average minimizes the separable objective

$$
\begin{array} { r } { J ( \boldsymbol { X } ) = \frac { 1 } { 2 } \displaystyle \sum _ { i , j } A _ { i j } \| \boldsymbol { x } _ { j } - \boldsymbol { B } _ { i } \| _ { 2 } ^ { 2 } . } \end{array}\tag{6}
$$

Unlike $\mathcal { L } , J$ compares each primitive directly with each observation, rather than comparing their rendered mixture with that observation. SFS also gives a squared-weight variant after opacity adjustment in its Appendix B, while Equation 5 corresponds to its base estimator. Within the OV-Foam lifting framework, we instantiate this estimator on PowerFoam reconstructions by accumulating the numerator and denominator using PowerFoam’s rendering weights. Real image observations use SAM regions encoded with CLIP (Kirillov et al., 2023; Radford et al., 2021a).

For this estimator, the relationship to the least-squares solution depends on the number and strength of contributors along each ray. When every ray has one fully absorbing contributor, Equation 5 is an exact least-squares solution. With multiple contributors, the unknown features are generally coupled, making the independent average an approximation. Even with a single contributor, however, weak opacity does not guarantee equivalence to least squares, since incomplete absorption changes the feature scale required to reproduce the observation. Equation 5 normalizes by $\textstyle \sum _ { i } A _ { i j }$ , whereas the normal equations normalize by $\textstyle \sum _ { i } A _ { i j } ^ { 2 }$ ; these agree only when every $A _ { i j } \in \{ 0 , 1 \}$

## 3.3 Dispersion and class purity

SFS introduces $\beta$ as a measure of feature dispersion along each viewing ray, characterizing how closely the one-shot estimator approximates the least-squares solution. Under complete absorption, ray opacity $s _ { i } = 1$ For each contributing primitive, define the feature discrepancy with respect to the observed pixel feature as $\Delta _ { i j } = \| \widehat { \boldsymbol { x } } _ { j } - B _ { i } \| _ { 2 }$ . Its weighted mean and relative dispersion are

$$
\mu _ { i } = \sum _ { j } A _ { i j } \Delta _ { i j } , \qquad \beta _ { i } = \frac { \sum _ { j } A _ { i j } ( \Delta _ { i j } - \mu _ { i } ) ^ { 2 } } { \mu _ { i } ^ { 2 } } , \qquad \mu _ { i } > 0 .\tag{7}
$$

The condition $s _ { i } = 1$ turns $A _ { i j }$ into a probability distribution over contributors along ray i. Low $\beta _ { i }$ indicates that the primitives contributing to ray i share similar error distances.

In addition, we introduce purity $P _ { j }$ , which addresses the complementary question of which classes contribute evidence to a primitive. Given labeled image rays with classes $y _ { i }$ , let

$$
\pi _ { j c } = { \frac { \sum _ { i : y _ { i } = c } A _ { i j } } { \sum _ { i : y _ { i } { \mathrm { i s } } \mathrm { l a b e l e d } } A _ { i j } } } , \qquad P _ { j } = \operatorname* { m a x } _ { c } \pi _ { j c } .\tag{8}
$$

Where $\pi _ { j c }$ is the share of the rendering weight that primitive $j$ receives from class c, not the share of rays: one ray striking $j$ with a large $A _ { i j }$ outweighs many that graze it, so the dominant class by weight can differ from the dominant class by count (Figure 3). Let $t _ { c }$ be the unit CLIP text embedding of class c. With $B _ { i } = t _ { y _ { i } }$ and the same labeled rays in both sums of Equation 5, the lifted feature becomes $\begin{array} { r } { x ^ { \prime } j = \sum _ { c } \pi j c t _ { c } } \end{array}$ , a convex combination of class embeddings in which $P _ { j }$ is the weight on the dominant class.

Inconsistent image observations arise from two distinct sources: geometric mixing, where rays sweep through multiple primitives across object boundaries, and upstream feature noise, where 2D models like CLIP produce different features across different viewpoints. These inconsistencies reveal a fundamental limitation of feature lifting: because each primitive $j$ stores a single feature vector $x _ { j } \in \mathbb { R } ^ { F }$ , any primitive receiving conflicting observations across views is forced to compromise. Even the exact least-squares solution $\widehat { X } = A ^ { \dagger } B$ returns a blended feature vector (e.g., a centroid between competing class embeddings). While this centroid minimizes the quadratic rendering loss from Equation 4, its diluted feature vector fails to align cleanly with any single class embedding during text-prompt classification.

We therefore distinguish the solver approximation error from the unavoidable semantic error of the least-squares target. By the orthogonality of least-squares residuals $( A ^ { \top } ( A { \widehat { X } } - B ) = 0 )$ , the additional reconstruction error incurred by the one-shot estimator $X ^ { \prime }$ is defined by thefitting gap:

$$
\mathcal { L } ( X ^ { \prime } ) - \mathcal { L } ( \widehat { X } ) = \frac { 1 } { 2 } \| A ( X ^ { \prime } - \widehat { X } ) \| ^ { 2 } \leq \frac { 1 } { 2 } \sum _ { i = 1 } ^ { R } \sum _ { j , k } A _ { i j } A _ { i k } \| \widehat { x } _ { j } - \widehat { x } _ { k } \| _ { 2 } ^ { 2 } .\tag{9}
$$

The inequality additionally assumes complete absorption, $s _ { i } = 1$ , the same row-stochastic condition assumed by $\mathbf { S F S } ^ { \prime } \mathbf { s }$ own approximation bound (Xiong et al., 2026). Here, $A _ { i j } A _ { i k }$ weights the co-contribution of primitives $j$ and k along ray i, while $\| \widehat { x } _ { j } - \widehat { x } _ { k } \| _ { 2 } ^ { 2 }$ captures their pairwise feature disagreement. This inequality shows that the fitting gap is bounded by the shared-ray feature disagreement between co-contributing primitives. Ray dispersion $\beta \ ( { \mathrm { E q . } } \ 7 )$ provides a complementary ray-level measure of this contributor disagreement. Concentrating ray weights onto fewer primitives reduces these cross-contributor terms, hence reducing the upper bound on the fitting gap $\hat { \mathcal { L } } \big ( X ^ { \prime } \big ) ^ { - } - \hat { \mathcal { L } } ( \widehat { X } )$ and in extension allowing for better feature assignments.

![](images/84fde648e30925a682eff0278c277b1475d4edb3ffd19ba97bc3fe917e3caf07.jpg)  
Figure 3: How semantic evidence reaches a primitive. (a) A cell can contain multiple classes. (b) Observation weights can reverse the majority suggested by ray counts. (c) Partial transparency lets foreground observations influence cells behind the surface.

However, because a small fitting gap cannot fix semantic ambiguity in the least-squares optimum $\widehat { X }$ we evaluate primitive purity $P _ { j }$ to measure whether the evidence aggregated at primitive $j$ is dominated by a single class. Together, $\beta ,$ the fitting gap, and purity allow us to trace how the 3D reconstruction governs feature lifting: from how closely $X ^ { \prime }$ approximates ${ \widehat { X } } .$ , to the class consistency of the evidence being aggregated, and ultimately to the accuracy of the recovered 3D labels. Figure 3 illustrates how spatial ownership, observation weights, and shared rays govern this interplay.

To disentangle these effects, we separate reconstruction-dependent errors from errors in the semantic observations serving as input to the lifting. We first isolate the geometric quality of the reconstruction by lifting ground-truth class masks encoded as text embeddings. Because both reconstructions receive perfect upstream labels, any difference in the resulting 3D labels proceeds from reconstruction and lifting process, unaffected by CLIP noise. We then evaluate open-vocabulary segmentation using real SAM/CLIP image features, testing whether the reconstruction-dependent effects persist under practical, noisy observation conditions.

## 4 Experiments

We evaluate the effect of the reconstruction on feature lifting at three levels. First, we compare the dispersion $\beta$ and the fitting gap $\mathcal { L } ( X ^ { \prime } ) { - } \mathcal { L } ( \widehat { X } )$ to determine whether the observed differences in $\beta$ correspond to differences in the quality of the one-shot lifting solution. We then examine purity and primitive-level prediction accuracy to determine whether the evidence accumulated by each reconstruction is more class-consistent.

To isolate the effect of the reconstruction from errors in the input image features, we next lift ground-truth image labels encoded as class text embeddings. This controlled experiment gives both reconstructions the same correct semantic observations and therefore tests whether differences in the recovered 3D labels arise from the reconstruction and lifting process itself. Finally, we evaluate CLIP image features, measuring whether the same differences persist when the lifted observations also contain errors.

## 4.1 Dispersion and class purity

We compare PowerFoam and 3DGS on ten ScanNet scenes (Dai et al., 2017), initializing both from the same GT vertices with one primitive per vertex, following the OpenGaussian (Wu et al., 2024) protocol. Centers and primitive counts remain fixed during reconstruction, and all reconstruction parameters remain fixed during lifting. We measure the number of nonzero contributors per ray, $k _ { i }$ , and dispersion $\beta _ { i }$ , using a least-squares reference computed on twelve matched views per scene. PSNR uses all views at native resolution. Table 1: Dispersion and rav contribution. Ten

Table 1 shows that PowerFoam uses fewer contributors per ray $k _ { i }$ and exhibits lower dispersion $\beta _ { i }$ while retaining higher average RGB reconstruction quality. To establish whether its one-shot estimate lies closer to least squares, we measure the fitting gap using GT image labels encoded as unit class text embeddings on twelve matched views. We compare raw, unnormalized feature fields against numerically converged, unregularized least-

Table 1: Dispersion and ray contribution. Tenscene averages. Dispersion is the mean of scene medians. Per-scene results in Table 5.
<table><tr><td colspan="2">Representation PSNR ↑</td><td> $k _ { i } \downarrow$ </td><td> $\beta _ { i } \downarrow$ </td></tr><tr><td>PowerFoam</td><td>27.01</td><td>2.32</td><td>0.0080</td></tr><tr><td>3DGS</td><td>25.22</td><td>33.63</td><td>1.5554</td></tr></table>

squares references. Averaging the per-ray gap $[ \mathcal { L } ( X ^ { \prime } ) - \mathcal { L } ( \widehat { X } ) ] / R$ equally across scenes gives $4 . 6 6 3 \times 1 0 ^ { - 3 }$ for 3DGS and $8 . 3 5 2 \times 1 0 ^ { - 4 }$ for PowerFoam, an 82.1% reduction. PowerFoam has the smaller gap and lower one-shot loss in all ten scenes, although 3DGS reaches the lower least-squares reference loss. Thus, PowerFoam supports a more accurate one-shot approximation on these reconstructions, rather than a lower attainable least-squares loss.

Closer agreement with least squares does not establish whether a primitive receives consistent class evidence. We therefore measure purity from the evidence accumulated by the renderer and compare it with prediction accuracy when lifting CLIP features. Each observation contributes according to its rendering weight $A _ { i j }$ , and purity measures the fraction of accumulated class evidence belonging to the dominant class.

![](images/b31bb9e430614659cba0f3bea68cb580c73115b1f4b16237cb7fee421f0c4d54.jpg)

Figure 4 shows that accuracy increases with purity across all five bands for both representations. PowerFoam places more evaluated points in the highest-purity band, where predictions are most accurate. Its advantage therefore includes a more favorable distribution of incoming evidence, rather than consistently higher accuracy within every band. Purity tracks prediction accuracy even with real image features, but does not guarantee correct labels.

![](images/60ef401a99ed64daec5a40f2a7190022c6a47145805a39f0366cc700a1b8303a.jpg)  
Figure 4: CLIP feature purity. Foam has more highpurity points. Higher purity tracks higher accuracy.

Even points assigned to the highest-purity primitives remain frequently misclassified with real CLIP features. Their contributing rays mostly observe the same GT class, but the image features need not identify that class correctly. We therefore replace the image features with GT class embeddings to measure the segmentation errors that remain when the upstream labels are correct.

## 4.2 Lifting ground-truth semantics

To separate errors in the image features from errors introduced by reconstruction and lifting, we replace each labeled pixel’s feature with the unit CLIP text embedding of its GT class, obtained from ScanNet’s label masks. Unlabeled pixels contribute no evidence. With each reconstruction fixed, we apply Equation 5 over all available lifting views and classify the resulting primitive features by cosine similarity to the class text embeddings.

Table 2 shows that PowerFoam recovers more accurate point labels from the same correct image observations, with higher mIoU in all ten scenes. This advantage is not explained by a higher geometric upper bound. The geometric upper bound pinpoints the limit introduced by mixed labels, as described in Figure 3. Therefore, we conclude that Power-Foam recovers more of the label accuracy available because of improved assignment from the image-based lifting.

These results complement the fitting-gap measurement.

Table 2: Lifting ground-truth image semantics. The geometric upper bound (Geom UB) assigns each primitive the majority GT class of its evaluated points.
<table><tr><td rowspan="2">Representation</td><td colspan="2">GT feature lifting</td><td rowspan="2">Geom UB</td></tr><tr><td>Point accuracy</td><td>mIoU</td></tr><tr><td>PowerFoam</td><td>96.37</td><td>90.59</td><td>mIoU 98.13</td></tr><tr><td>3DGS</td><td>94.25</td><td>85.40</td><td>98.92</td></tr></table>

PowerFoam’s one-shot estimate is closer to its least-squares reference, and, when supplied with correct image labels, it also produces more accurate 3D segmentation. The remaining question is whether this advantage persists with imperfect image features and different aggregation methods. We evaluate that setting next by replacing the Gaussian representation in existing open-vocabulary segmentation methods with PowerFoam.

## 4.3 Open-vocabulary segmentation

We adapt existing Gaussian-based methods’ feature lifting approaches to PowerFoam. Each method uses the same SAM/CLIP image features for both representations, allowing us to compare their segmentation results while holding the upstream observations fixed.

For ScanNet, we follow OpenGaussian’s ten-scene point-level evaluation setup (Wu et al., 2024), with the changes to point assignment and opacity filtering described below. We assign each evaluated point its primitive’s predicted class and report mIoU (mean intersection over union) and mAcc (mean class accuracy). Predictions use three different class sets. For PowerFoam, we assign points using the reconstructed surface inside each cell, represented by its dipole, rather than the cell center alone. This surface can shift during reconstruction even when the center remains fixed, so the center need not lie in the primitive’s occupied region (Appendix B.1).

OpenGaussian’s protocol excludes ground-truth points associated with Gaussians whose opacity is below 0.1. However, low-opacity primitives can still contribute to rendering and receive features during lifting. Excluding their associated points removes part of the reconstruction from assessment and allows the evaluated point set to depend on the representation. It also means that a higher score on retained points need not imply better performance on the full scene. We therefore retain all valid labeled ground-truth points for both representations, so weakly represented regions remain part of the segmentation task. Under this common point set, PowerFoam improves both metrics for every tested method across all three class sets (Table 3). For the opacity-filtered evaluation, see Table 6.

We also evaluate rendered text-query segmentation on Figurines, Ramen, Teatime, and Waldo kitchen from LERF (Kerr et al., 2023), following the SFS evaluation protocol (Xiong et al., 2026). For each query, we render the lifted features from the evaluation camera and compare them with the query’s text embedding to obtain a segmentation mask. Both representations use the same query evaluator, reference masks, and threshold rules. Table 4 reports mean IoU and mean accuracy across the four scenes.

![](images/e80af663a3c3381dfcc8a04bc22c41991561d8946167202ff3fd70a6ffb40b1e.jpg)

OV-Foam improves both metrics for all four methods,

extending the advantage from 3D point labels to rendered segmentation masks. Together, these results show that PowerFoam’s advantage extends beyond correct upstream labels and a single lifting method. The earlier measurements identify differences in how the representations distribute and combine image evidence. The CLIP feature benchmarks establish that replacing Gaussians with PowerFoam can turn those representation differences into better open-vocabulary segmentation, even when the input features remain imperfect.

Table 3: Open-vocabulary 3D segmentation on ScanNet. We report mIoU and mAcc (%) for three class sets. Bold marks the higher score within each pair. Green parentheses show OV-Foam’s improvement over 3DGS in percentage points. Appendix B details the protocol.
<table><tr><td colspan="2"></td><td colspan="2">19 classes</td><td colspan="2">15 classes</td><td colspan="2">10 classes</td></tr><tr><td>Method</td><td>Representation mIoU</td><td></td><td>mAcc</td><td>mIoU</td><td>mAcc</td><td>mIoU</td><td>mAcc</td></tr><tr><td>VALA (Wang et al., 2026)</td><td>3DGS</td><td>30.08</td><td>47.57</td><td>32.70</td><td>52.00</td><td>39.56</td><td>60.35</td></tr><tr><td rowspan="3"></td><td>OV-Foam</td><td>33.68</td><td>51.18</td><td>37.15</td><td>56.08</td><td>45.68</td><td>65.39</td></tr><tr><td> $\Delta$ </td><td>+3.60</td><td>+3.61</td><td>+4.45</td><td>+4.08</td><td>+6.12</td><td>+5.04</td></tr><tr><td>Occam&#x27;s LGS (Cheng et al., 2025) 3DGS</td><td>32.76</td><td>50.40</td><td>35.62</td><td>55.24</td><td>43.94</td><td>64.66</td></tr><tr><td></td><td>OV-Foam</td><td>34.19</td><td>52.34</td><td>37.58</td><td>57.46</td><td>46.46</td><td>66.64</td></tr><tr><td rowspan="3">SFS (Xiong et al., 2026)</td><td> $\Delta$ </td><td>+1.43</td><td>+1.94</td><td>+1.96</td><td>+2.22</td><td>+2.52</td><td>+1.98</td></tr><tr><td>3DGS</td><td>32.03</td><td>49.29</td><td>35.41</td><td>54.78</td><td>44.34</td><td>64.24</td></tr><tr><td>OV-Foam</td><td>34.00</td><td>51.75</td><td>37.59</td><td>57.43</td><td>46.91</td><td>66.64</td></tr><tr><td rowspan="4">LUDVIG (Marrie et al., 2025)</td><td> $\Delta$ </td><td>+1.97</td><td></td><td>+2.46+2.18</td><td>+2.65</td><td>+2.57</td><td>+2.40</td></tr><tr><td>3DGS</td><td>32.94</td><td>50.36</td><td>36.58</td><td>56.22</td><td>45.64</td><td>65.43</td></tr><tr><td>OV-Foam</td><td>34.58</td><td>+1.78 +1.53</td><td>52.14 38.11</td><td>57.84</td><td>47.59</td><td>67.04</td></tr><tr><td> $\Delta$ </td><td>+1.64</td><td></td><td></td><td></td><td>+1.62 +1.95</td><td>+1.61</td></tr></table>

Table 4: LERF text-query evaluation. Mean IoU and mean accuracy across four scenes under a query and mask protocol. Bold marks the better representation. $\Delta$ denotes OV-Foam minus 3DGS.
<table><tr><td></td><td colspan="3">mIoU</td><td colspan="3">mAcc</td></tr><tr><td>Method</td><td>3DGS</td><td>OV-Foam</td><td> $\Delta$ </td><td>3DGS</td><td>OV-Foam</td><td> $\Delta$ </td></tr><tr><td>VALA (Wang et al., 2026)</td><td>0.555</td><td>0.570</td><td>+0.015</td><td>0.707</td><td>0.730</td><td>+0.023</td></tr><tr><td>Occam&#x27;s LGS (Cheng et al., 2025)</td><td>0.522</td><td>0.529</td><td>+0.007</td><td>0.699</td><td>0.749</td><td>+0.050</td></tr><tr><td>SFS (Xiong et al., 2026)</td><td>0.502</td><td>0.570</td><td>+0.068</td><td>0.641</td><td>0.659</td><td>+0.018</td></tr><tr><td>LUDVIG (Marrie et al., 2025)</td><td>0.438</td><td>0.543</td><td>+0.105</td><td>0.735</td><td>0.746</td><td>+0.011</td></tr></table>

## 5 Limitations and conclusion

Our ScanNet experiments cover ten scenes and initialize both representations from the same point cloud through the SfM initialization pipeline. For this benchmark, the supplied point cloud is the ground-truth geometry provided by ScanNet rather than an image-derived SfM reconstruction. The ground-truth lifting experiment isolates errors that remain with correct image labels, but does not establish an upper bound over all lifting methods. Our fitting-gap analysis concerns feature reconstruction loss rather than segmentation accuracy, and high purity can still reflect consistently incorrect observations. The observed advantages therefore depend on the reconstruction, input features, and evaluation setting.

We investigated how the choice of reconstruction primitives affects feature lifting. Under matched reconstruction budgets, PowerFoam has fewer contributors per ray and lower measured dispersion than 3DGS, while placing more evaluated points on primitives with high-purity evidence. Its one-shot estimate is closer to its least-squares reference, and lifting correct image labels produces more accurate 3D segmentation. These advantages extend to real image features, adapting existing lifting methods to PowerFoam improves the tested ScanNet and LERF segmentation results. Together, these findings establish representation choice as a practical way to improve semantic lifting without requiring more accurate upstream image features.

## AI use statement

In this work, we used generative AI tools for drafting and editing sections of the paper. We also used AI tools to find related works beyond what our manual research found. We used AI models to review and critique our paper and used its feedback to improve our work. AI coding agents were used for parts of the implementation and debugging. All generated text, code and suggested papers were reviewed thoroughly by the authors. In all cases we used AI tools as tools to assist our work, never to replace or automate full aspects of the research process.

## Reproducibility statement

With our paper, we release the full source code required to reproduce our results. We will upload the reconstructed 3DGS and Powerfoam scenes and the lifted feature sets used as input for our segmentation. We will supply a simple Python script that reproduces the segmentations that lead to the results.

## References

Jiazhong Cen, Xudong Zhou, Jiemin Fang, Changsong Wen, Lingxi Xie, Xiaopeng Zhang, Wei Shen, and Qi Tian. Tackling view-dependent semantics in 3D language gaussian splatting. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 7013–7034. PMLR, 2025. URL https://proceedings.mlr.press/v267/ cen25a.html.

Jiahuan Cheng, Jan-Nico Zaech, Luc Van Gool, and Danda Pani Paudel. Occam’s LGS: An efficient approach for language gaussian splatting. In Proceedings ofthe 36th British Machine Vision Conference. BMVA, 2025. URL https://bmvc2025.bmva.org/proceedings/694/.

Angela Dai, Angel X. Chang, Manolis Savva, Maciej Halber, Thomas Funkhouser, and Matthias Nießner. ScanNet: Richly-annotated 3D reconstructions of indoor scenes. In IEEE Conference on Computer Vision and Pattern Recognition, pages 5828–5839, 2017. URL https://openaccess.thecvf.com/content\_cvpr\_2017/html/Dai\_ScanNet\_ Richly-Annotated\_3D\_CVPR\_2017\_paper.html.

Shaohui Dai, Yansong Qu, Zheyan Li, Xinyang Li, Shengchuan Zhang, and Liujuan Cao. Training-free hierarchical scene understanding for gaussian splatting with superpoint graphs. In Proceedings ofthe 33rd ACM International Conference on Multimedia, pages 3673–3682. Association for Computing Machinery, 2025. doi: 10.1145/3746027.3755146. URL https://doi.org/10.1145/3746027.3755146.

Bin Dou, Tianyu Zhang, Zhaohui Wang, Yongjia Ma, Zejian Yuan, and Nanning Zheng. Learning segmented 3D gaussians via efficient feature unprojection for zero-shot neural scene segmentation. In Neural Information Processing – ICONIP 2024, volume 15293 of Lecture Notes in Computer Science, pages 398–412. Springer, 2025. doi: 10.1007/978-981-96-6596-9\_28. URL https://link.springer. com/chapter/10.1007/978-981-96-6596-9\_28.

Shrisudhan Govindarajan, Daniel Rebain, Kwang Moo Yi, and Andrea Tagliasacchi. Radiant foam: Real-time differentiable ray tracing. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pages 4135–4145, 2025. URL https://openaccess.thecvf.com/content/ICCV2025/ html/Govindarajan\_Radiant\_Foam\_Real-Time\_Differentiable\_Ray\_Tracing\_ ICCV\_2025\_paper.html.

Shrisudhan Govindarajan, Daniel Rebain, Dor Verbin, Kwang Moo Yi, Anish Prabhu, and Andrea Tagliasacchi. Power foam: Unifying real-time differentiable ray tracing and rasterization. arXiv preprint arXiv:2604.24994, 2026. URL https://arxiv.org/abs/2604.24994.

Qiao Gu, Ali Kuwajerwala, Sacha Morin, Krishna Murthy Jatavallabhula, Bipasha Sen, Aditya Agarwal, Corban Rivera, William Paul, Kirsty Ellis, Rama Chellappa, Chuang Gan, Celso Miguel de Melo, Joshua B. Tenenbaum, Antonio Torralba, Florian Shkurti, and Liam Paull. ConceptGraphs: Open-vocabulary 3D scene graphs for perception and planning. In 2024 IEEE International Conference on Robotics and Automation (ICRA), pages 5021–5028. IEEE, 2024. doi: 10.1109/ICRA57147.2024.10610243. URL https://doi.org/10.1109/ICRA57147.2024.10610243.

Kim Jun-Seong, GeonU Kim, Kim Yu-Ji, Yu-Chiang Frank Wang, Jaesung Choe, and Tae-Hyun Oh. Dr. Splat: Directly referring 3D gaussian splatting via direct language embedding registration. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 14137–14146. IEEE, 2025. doi: 10.1109/CVPR52734.2025.01319. URL https://doi.org/10.1109/CVPR52734. 2025.01319.

Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, and George Drettakis. 3D gaussian splatting for realtime radiance field rendering. ACM Transactions on Graphics, 42(4):1–14, 2023. doi: 10.1145/3592433. URL https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/.

Justin Kerr, Chung Min Kim, Ken Goldberg, Angjoo Kanazawa, and Matthew Tancik. LERF: Language embedded radiance fields. In IEEE/CVF International Conference on Computer Vision, pages 19729– 19739, 2023. URL https://openaccess.thecvf.com/content/ICCV2023/html/Kerr\_ LERF\_Language\_Embedded\_Radiance\_Fields\_ICCV\_2023\_paper.html.

Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Dollár, and Ross Girshick. Segment anything. In IEEE/CVF International Conference on Computer Vision, pages 4015–4026, 2023. URL https://openaccess.thecvf.com/content/ICCV2023/html/Kirillov\_ Segment\_Anything\_ICCV\_2023\_paper.html.

Juliette Marrie, Romain Menegaux, Michael Arbel, Diane Larlus, and Julien Mairal. LUDVIG: Learning-free uplifting of 2D visual features to gaussian splatting scenes. In IEEE/CVF International Conference on Computer Vision, pages 7440–7450, 2025. URL https://openaccess.thecvf.com/ content/ICCV2025/html/Marrie\_LUDVIG\_Learning-Free\_Uplifting\_of\_2D\_ Visual\_Features\_to\_Gaussian\_Splatting\_ICCV\_2025\_paper.html.

Minghan Qin, Wanhua Li, Jiawei Zhou, Haoqian Wang, and Hanspeter Pfister. LangSplat: 3D language gaussian splatting. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20051– 20060, 2024. URL https://openaccess.thecvf.com/content/CVPR2024/html/Qin\_ LangSplat\_3D\_Language\_Gaussian\_Splatting\_CVPR\_2024\_paper.html.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In Proceedings of the 38th International Conference on Machine Learning, volume 139 of Proceedings ofMachine Learning Research, pages 8748–8763. PMLR, 2021a. URL https://proceedings.mlr.press/v139/radford21a.html.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In Proceedings of the 38th International Conference on Machine Learning, volume 139 of Proceedings of Machine Learning Research, pages 8748–8763. PMLR, 2021b. URL https://proceedings.mlr.press/v139/radford21a.html.

Amr Sharafeldin, Aryan Mikaeili, Thomas Walker, Shrisudhan Govindarajan, Daniel Rebain, Kwang Moo Yi, and Andrea Tagliasacchi. Semantic foam: Unifying spatial and semantic scene decomposition. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 29814–29823, 2026. URL https://openaccess.thecvf.com/content/CVPR2026/html/Sharafeldin\_ Semantic\_Foam\_Unifying\_Spatial\_and\_Semantic\_Scene\_Decomposition\_CVPR\_ 2026\_paper.html.

Haoyu Wang, Sizheng Guo, Jin Ye, Zhongying Deng, Junlong Cheng, Tianbin Li, Jianpin Chen, Yanzhou Su, Ziyan Huang, Yiqing Shen, Bin Fu, Shaoting Zhang, and Junjun He. SAM-Med3D: A vision foundation model for general-purpose segmentation on volumetric medical images. IEEE Transactions on Neural Networks and Learning Systems, 36(10):17599–17612, 2025. doi: 10.1109/TNNLS.2025.3586694. URL https://doi.org/10.1109/TNNLS.2025.3586694.

Sen Wang, Kunyi Li, Siyun Liang, Elena Alegret, Jing Ma, Nassir Navab, and Stefano Gasperini. Visibilityaware language aggregation for open-vocabulary segmentation in 3D gaussian splatting. In 2026 International Conference on 3D Vision (3DV), pages 1791–1800, 2026. doi: 10.1109/3DV69130.2026.00170. URL https://doi.org/10.1109/3DV69130.2026.00170.

Yanmin Wu, Jiarui Meng, Haijie Li, Chenming Wu, Yahao Shi, Xinhua Cheng, Chen Zhao, Haocheng Feng, Errui Ding, Jingdong Wang, and Jian Zhang. OpenGaussian: Towards point-level 3D gaussian-based open vocabulary understanding. In Advances in Neural Information Processing Systems, volume 37, 2024. doi: 10.52202/079017-0604. URL https://proceedings.neurips.cc/paper\_files/paper/ 2024/hash/21f7b745f73ce0d1f9bcea7f40b1388e-Abstract-Conference.html.

Butian Xiong, Rong Liu, Kenneth Xu, Meida Chen, and Andrew Feng. Splat feature solver. In International Conference on Learning Representations, 2026. URL https://arxiv.org/abs/2508.12216.

## A Dispersion and Contribution count

Table 5: Per-scene matched-budget results on ScanNet. Both representations use the same primitive count P within each scene. PSNR uses all views. Contributor count $k _ { i }$ and dispersion $\beta _ { i }$ use twelve views per scene. The final row averages the scene summaries, including the scene medians of $\beta _ { i }$ . Bold indicates higher PSNR or lower contributor count and dispersion.
<table><tr><td colspan="2" rowspan="2">Scene</td><td colspan="2">PSNR (dB)</td><td colspan="2">Mean  $k _ { i }$ </td><td colspan="2">Median  $\beta _ { i }$ </td></tr><tr><td>P Foam</td><td>3DGS</td><td>Foam</td><td>3DGS</td><td>Foam</td><td>3DGS</td></tr><tr><td>scene0000_00</td><td>81,369</td><td>21.84</td><td>19.74</td><td>2.32</td><td>31.52</td><td>0.0053</td><td>1.8443</td></tr><tr><td>scene0062_00</td><td>51,610</td><td>27.31</td><td>23.16</td><td>2.02</td><td>33.90</td><td>0.0000</td><td>1.0508</td></tr><tr><td>scene0070_00</td><td>109,380</td><td>28.13</td><td>29.46</td><td>2.69</td><td>34.11</td><td>0.0318</td><td>2.1095</td></tr><tr><td>scene0097_00</td><td>72,007</td><td>31.94</td><td>34.44</td><td>1.91</td><td>33.26</td><td>0.0000</td><td>1.1392</td></tr><tr><td>scene0140_00</td><td>372,941</td><td>27.25</td><td>26.62</td><td>2.76</td><td>34.24</td><td>0.0092</td><td>1.6057</td></tr><tr><td>scene0200_00</td><td>83,291</td><td>26.06</td><td>21.09</td><td>2.00</td><td>34.82</td><td>0.0000</td><td>1.1613</td></tr><tr><td>scene0347_00</td><td>67,984</td><td>25.24</td><td>20.56</td><td>2.13</td><td>31.35</td><td>0.0001</td><td>1.2667</td></tr><tr><td>scene0400_00</td><td>155,959</td><td>26.60</td><td>21.48</td><td>1.96</td><td>36.26</td><td>0.0000</td><td>1.5230</td></tr><tr><td>scene0590_00</td><td>222,957</td><td>31.67</td><td>32.50</td><td>2.73</td><td>33.52</td><td>0.0035</td><td>1.7131</td></tr><tr><td>scene0645_00</td><td>352,477</td><td>24.04</td><td>23.15</td><td>2.66</td><td>33.36</td><td>0.0302</td><td>2.1408</td></tr><tr><td>Mean</td><td>156,998</td><td>27.01</td><td>25.22</td><td>2.32</td><td>33.63</td><td>0.0080</td><td>1.5554</td></tr></table>

## B ScanNet evaluation protocol

We follow OpenGaussian’s ten-scene point-level evaluation setup (Wu et al., 2024). For our matched reconstructions, both representations start from the same ground-truth vertices, with one primitive per vertex. Primitive centers and counts remain fixed during reconstruction, while the remaining reconstruction parameters are optimized. We then hold the reconstruction fixed during feature lifting.

We assign semantic labels by maximum cosine similarity to the class text embeddings and report mIoU and mAcc for the 19-, 15-, and 10-class evaluations. Within each scene, these metrics average class IoU and class recall over the valid ground-truth classes present. Final scores average the scene results.

## B.1 Dipole-aware point assignment

For OV-Foam, point ownership uses the reconstructed dipole geometry. A fixed initialization center need not lie on the occupied side of its primitive’s displaced surface. For each evaluation point $q ,$ we take the 64 Euclidean-nearest supported primitive centers and rank them by power distance $\lVert q - p _ { j } \rVert ^ { 2 } - a _ { j } ^ { 2 }$ . We select the first candidate satisfying $( q - p _ { j } ) \cdot n _ { j } \leq h _ { j } ( q )$ , where $h _ { j } ( q )$ is the interpolated dipole height at the point’s tangent-plane coordinates. This test checks the occupied side of the dipole surface.

If no candidate passes, we use the minimum power distance in the same pool. The fallback preserves assignment coverage, so the dipole test does not discard points. Power distances are evaluated in double precision. The finite candidate search approximates global power-distance ranking.

## B.2 Evaluation without opacity culling

OpenGaussian excludes GT points associated with Gaussians whose opacity is below 0.1. The Del.% column in Table 6 reports the fraction of labeled points excluded before scoring. Rates marked <sup>‡</sup> come from our matched reproductions. This exclusion changes which points contribute to the metrics. To compare both representations on the same evaluation support, Table 3 retains all valid labeled ground-truth points, keeping the reconstructions and lifted features fixed. Removing opacity culling widens OV-Foam’s advantage over the matched Gaussian baselines. OV-Foam also achieves higher mIoU and mAcc for every method across the class set evaluations. These results show that its advantage persists and increases when low-opacity points are included in scoring.

Table 6: ScanNet results under opacity-filtered evaluation. Published Gaussian scores are paired with our OV-Foam adaptations. Bold marks the better score within each pair. Green parentheses show OV-Foam minus 3DGS in percentage points. Del.% denotes excluded labeled points. <sup>‡</sup>Rates are measured on our matched reproductions, not reported by the cited papers. Table 3 reports results without filtering.
<table><tr><td colspan="2"></td><td colspan="2">19 classes</td><td colspan="2">15 classes</td><td colspan="2">10 classes</td><td rowspan="2">Del.%</td></tr><tr><td>Method</td><td>Representation</td><td>mIoU</td><td>mAcc</td><td>mIoU</td><td>mAcc</td><td>mIoU</td><td>mAcc</td></tr><tr><td rowspan="2">VALA (Wang et al., 2026)</td><td>3DGS</td><td>32.11</td><td>50.05</td><td>35.10</td><td>54.77</td><td>46.21</td><td>65.61</td><td>14.16‡</td></tr><tr><td>OV-Foam</td><td>(+1.57) 33.68</td><td>(+1.13)51.18</td><td>(+2.05)37.15</td><td>(+1.31)56.08</td><td>(-0.53)45.68</td><td>(-0.22)65.39</td><td>0.00</td></tr><tr><td rowspan="2">Occam&#x27;s LGS (Cheng et al., 2025)</td><td>3DGS</td><td>31.93</td><td>48.93</td><td>34.25</td><td>53.71</td><td>45.16</td><td>64.39</td><td>19.28‡</td></tr><tr><td>OV-Foam</td><td>(+2.26)34.19</td><td>(+3.42) 52.35</td><td>(+3.34) 37.59</td><td>(+3.76)57.47</td><td>(+1.31)46.47</td><td>(+2.26) 66.65</td><td>0.10</td></tr><tr><td rowspan="2">SFS (Xiong et al., 2026)</td><td>3DGS</td><td>33.33</td><td>51.35</td><td>36.43</td><td>55.38</td><td>44.74</td><td>63.53</td><td>12.66‡</td></tr><tr><td>OV-Foam</td><td>(+0.68)34.01</td><td>(+0.41)51.76</td><td>(+1.18)37.61</td><td>(+2.07)57.45</td><td>(+2.18)46.92</td><td>(+3.13) 66.66</td><td>0.10</td></tr><tr><td rowspan="2">LUDVIG (Marrie et al., 2025)</td><td>3DGS</td><td>33.90</td><td>51.40</td><td>37.40</td><td>57.20</td><td>46.40</td><td>66.20</td><td>15.92‡</td></tr><tr><td>OV-Foam</td><td>(+0.69)34.59</td><td>(+0.75) 52.15</td><td>(+0.72)38.12</td><td>(+0.66)57.86</td><td>(+1.20)47.60</td><td>(+0.85)67.05</td><td>0.11</td></tr></table>

## C Full Oracle table

Table 7: Per-scene lifting of ground-truth image semantics on ScanNet. Point accuracy and mIoU (%) when each valid pixel carries its ground-truth class text embedding, using ScanNet’s released 2D label masks. Geom UB is the geometric upper bound: each primitive is assigned the majority ground-truth class of its evaluated points. Bold marks the higher score within each pair. C is the number of ground-truth classes present, and N is the number of scored points. This shows 3DGS doesn’t lose due to lack of capacity but due to the solve errors. The Mean row is the equal-weight average over scenes and matches Table 2.
<table><tr><td colspan="3"></td><td colspan="3">Point accuracy</td><td colspan="3">mIoU</td><td colspan="3">Geom UB (mIoU)</td></tr><tr><td>Scene</td><td>C</td><td>N</td><td>3DGS</td><td>OV-Foam</td><td>∆</td><td>3DGS</td><td>OV-Foam</td><td>∆</td><td>3DGS</td><td>OV-Foam</td><td>∆</td></tr><tr><td>scene0000_00</td><td>14</td><td>54,480</td><td>92.98</td><td>96.10</td><td>(+3.12)</td><td>83.61</td><td>91.27</td><td>(+7.66)</td><td>97.00</td><td>97.81</td><td>(+0.81)</td></tr><tr><td>scene0062_00</td><td>6</td><td>41,873</td><td>95.85</td><td>96.50</td><td>(+0.65)</td><td>86.53</td><td>87.82</td><td>(+1.29)</td><td>98.94</td><td>97.50</td><td>(-1.44)</td></tr><tr><td>scene0070_00</td><td>9</td><td>69,922</td><td>94.10</td><td>96.77</td><td>(+2.67)</td><td>83.30</td><td>89.89</td><td>(+6.59)</td><td>98.69</td><td>97.92</td><td>(-0.77)</td></tr><tr><td>scene0097_00</td><td>7</td><td>52,656</td><td>95.41</td><td>97.34</td><td>(+1.93)</td><td>88.37</td><td>93.06</td><td>(+4.69)</td><td>99.33</td><td>97.81</td><td>(-1.52)</td></tr><tr><td>scene0140_00</td><td>8</td><td>279,359</td><td>91.82</td><td>94.11</td><td>(+2.29)</td><td>86.03</td><td>90.80</td><td>(+4.77)</td><td>99.19</td><td>98.95</td><td>(-0.24)</td></tr><tr><td>scene0200_00</td><td>8</td><td>65,082</td><td>93.65</td><td>96.07</td><td>(+2.42)</td><td>83.11</td><td>89.59</td><td>(+6.48)</td><td>99.20</td><td>97.69</td><td>(-1.51)</td></tr><tr><td>scene0347_00</td><td>7</td><td>48,063</td><td>95.06</td><td>97.07</td><td>(+2.01)</td><td>88.44</td><td>93.10</td><td>(+4.66)</td><td>99.34</td><td>99.05</td><td>(-0.29)</td></tr><tr><td>scene0400_00</td><td>7</td><td>93,930</td><td>94.94</td><td>97.26</td><td>(+2.32)</td><td>85.36</td><td>91.23</td><td>(+5.87)</td><td>99.57</td><td>98.87</td><td>(-0.70)</td></tr><tr><td>scene0590_00</td><td></td><td>14 153,151</td><td>93.91</td><td>95.78</td><td>(+1.87)</td><td>84.74</td><td>88.85</td><td>(+4.11)</td><td>98.99</td><td>97.28</td><td>(-1.71)</td></tr><tr><td>scene0645_00</td><td>16</td><td>253,034</td><td>94.76</td><td>96.71</td><td>(+1.95)</td><td>84.55</td><td>90.24</td><td>(+5.69)</td><td>98.96</td><td>98.46</td><td>(-0.50)</td></tr><tr><td>Mean</td><td></td><td></td><td>94.25</td><td>96.37</td><td>(+2.12)</td><td>85.40</td><td>90.59</td><td>(+5.19)</td><td>98.92</td><td>98.13</td><td>(-0.79)</td></tr></table>

## D Proof of the fitting-gap identity

Let A contain the rendering weights, B the observed image features, and X the primitive features. The rendering loss is

$$
\mathcal { L } ( \boldsymbol { X } ) = \frac { 1 } { 2 } \| \boldsymbol { A } \boldsymbol { X } - \boldsymbol { B } \| _ { F } ^ { 2 } = \frac { 1 } { 2 } \sum _ { i = 1 } ^ { R } \sum _ { f = 1 } ^ { d } \bigl ( ( \boldsymbol { A } \boldsymbol { X } ) _ { i f } - B _ { i f } \bigr ) ^ { 2 } ,\tag{10}
$$

where $R$ is the number of rays and d is the feature dimension. The subscript F denotes the Frobenius norm, which sums squared entries of a matrix as shown above.

Let $\widehat { X }$ be an exact, unconstrained least-squares minimizer and $X ^ { \prime }$ the one-shot estimate, both using the same A and $B .$ . Then

$$
\boxed { \mathcal { L } ( X ^ { \prime } ) - \mathcal { L } ( \widehat { X } ) = \frac { 1 } { 2 } \| A ( X ^ { \prime } - \widehat { X } ) \| _ { F } ^ { 2 } . }\tag{11}
$$

To prove this, write the one-shot rendering residual as

$$
A X ^ { \prime } - B = ( A \widehat { X } - B ) + A ( X ^ { \prime } - \widehat { X } ) .
$$

Expanding its squared norm gives

$$
\mathcal { L } ( X ^ { \prime } ) - \mathcal { L } ( \widehat { X } ) = \frac { 1 } { 2 } \| A ( X ^ { \prime } - \widehat { X } ) \| _ { F } ^ { 2 }\tag{12}
$$

$$
+ \sum _ { i = 1 } ^ { R } \sum _ { f = 1 } ^ { d } ( A \widehat { X } - B ) _ { i f } \big [ A ( X ^ { \prime } - \widehat { X } ) \big ] _ { i f } .\tag{13}
$$

The final term is zero because the least-squares normal equations give $A ^ { \top } ( A { \widehat { X } } - B ) = 0$ . Thus the fitting gap equals half the squared difference between the image features rendered by the two estimates.

The identity allows incomplete absorption and does not require a unique least-squares solution. It measures agreement between rendered features, not correctness of the resulting semantic labels. For a numerically computed reference, the equality holds up to its remaining optimization error.

## E Qualitative figures

OV-Foam

Query: rubber duck with hat | frame 00105

![](images/28b72d364bf4a34f956649b2bdf6315a8a29c9781534897a47e62c9b78d5f877.jpg)

3DGS + SFS  
![](images/551c1fc341a8055572a3485937dada820fceafeb6de0b4f9ac559eb91477c132.jpg)

Occam's LGS  
![](images/a1e9081bf35af7edff87f605f38ff3867cc15628f3a63e7945e91c1c33ba205d.jpg)  
Query: rubber duck with hat | frame 00152

![](images/fe663aa596fdcdb95661bb0a07f853fa808307c5acd91e1871770e05cadf5d09.jpg)

![](images/3a7d230f8f9722b163e3d7069254be6c2fc6e701e5e684e6fab57607633f8cf8.jpg)

![](images/1e3029b9ce1e6683bafbac27fd840cd20730fc4a554aa2f5e7c77adab95a99f6.jpg)  
Query: rubber duck with hat | frame 00195

![](images/9488fbc916643991cc8ee9dcdb3afd7622d4026ab9a48847a6aa14dd76a4c2b7.jpg)

![](images/c88cc614a508a33f47f455c2a989fe103b814e4a2f21ea39ce7ce0e82e3b6aa9.jpg)

![](images/c4e2fbe6827700c7a27d2c17d35802155141787dbf55e672c77660e658b15fa0.jpg)  
Figure 5: Object extraction for “rubber duck with hat, frames 00105, 00152 and 00195. OV-Foam uses SFS here

![](images/255b163a495551644c5f51c51dd8ba29e4fd77b99a65dd8a89088891e111a913.jpg)  
Figure 6: Object extraction for “rubics cube, frames 00105, 00152 and 00195. OV-Foam uses SFS here

OV-Foam

Query: toy elephant | frame 00041

![](images/62101ec159b59aba306343e64db9c05768f02b4631ea009f6d42f27454dad4df.jpg)  
Query: toy elephant | frame 00152

3DGS + SFS  
![](images/a9d80ec8a97afae0671b1f1845f09ff8c7b765bab402bf92047d63ec5dccc148.jpg)  
Occam's LGS

![](images/9605a8185cd99917fa7242532a88530b34507048d2e881e5c3363127bd276bea.jpg)

![](images/1689321aa9dee2f0ed18098d480cf2826da96e3795277a9a8130fcd7f3d95654.jpg)

![](images/015d6ed5330143bd69179781015357eb914866cbeeae7feec1aa030adbafa7b3.jpg)

![](images/6a7c3f2921a21506706a8d7de6cd2bed31b2575999b912905d732c6184c45038.jpg)  
Query: toy elephant | frame 00195

![](images/ad9988d0f5970222d2a153a4d41311f0b49859e98f7f2f6ed4c3a846dffd7f5a.jpg)

![](images/5e07d024ec352f9100523caf0daecde8e06f5ac8974d71a986e4bade5a3a6ce5.jpg)

![](images/3e28611e1708adf8467e0b4e5825eb163e10a9c96436b49c4c314ec65ebcfe13.jpg)  
Figure 7: Object extraction for “toy elephant, frames 00041, 00152 and 00195. OV-Foam uses SFS here

OV-Foam  
3DGS + SFS  
Occam's LGS  
![](images/265c29649aa16f9492c0c710d17b899688f2401856efdb66a6db687266d1b587.jpg)  
Figure 8: RGB appearance, shared-basis feature PCA and cosine similarity to “pikachu, frame 00041. OV-Foam uses SFS here

OV-Foam  
3DGS + SFS  
Occam's LGS  
![](images/73098fdccf0c1fae1989e35cc86d887e5e736b275677ee787e195097e148518e.jpg)  
Figure 9: RGB appearance, shared-basis feature PCA and cosine similarity to “rubber duck with hat, frame 00152. The supplied overlay uses alpha 0.55. OV-Foam uses SFS here

RGB appearance  
OV-Foam  
3DGS + SFS  
Occam's LGS  
![](images/ab31ec116b77d61b24955bf93a6945168aaca5242b7989ebb75fbb762ca305bf.jpg)  
Figure 10: RGB appearance, shared-basis feature PCA and cosine similarity to “rubics cube, frame 00152. The supplied overlay uses alpha 0.55. OV-Foam uses SFS here