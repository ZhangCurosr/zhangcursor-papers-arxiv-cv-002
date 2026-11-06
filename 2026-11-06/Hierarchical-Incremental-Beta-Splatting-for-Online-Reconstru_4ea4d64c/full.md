# Hierarchical Incremental Beta Splatting for Online Reconstruction

Linus Franke

University of Wurzburg¨

https://linus-franke.com/hibs

![](images/41afd347a38d51773bc5665c7f83881a6bc84fbdf38ca4013b0a674b4f4557a1.jpg)  
Figure 1. We present an online method for incremental reconstruction of primitive-based radiance fields from an unordered image stream. Building on beta kernels, we use kernel shape as a confidence dial: each keyframe spawns primitives with opacity and shape initialized from measured multi-view confidence. Short optimization bursts, fused rasterization, error-driven relocation, and closed-form LoD merging enable efficient reconstruction. Compared to i3DGS [42], we achieve higher quality with about 20% faster optimization.

## Abstract

Online 3D Gaussian Splatting reconstructs a scene during capture: each keyframe spawns primitives once and refines themfor only afew dozen iterations, leaving little chance to fix poor initialization or missing detail. Gaussians are also inherently soft, needing many overlapsfor solid surfaces.

We present Hierarchical Incremental Beta Splatting (HIBS), which brings beta kernels to incremental reconstruction. Their shape parameter interpolates between a soft, Gaussian-like profile and a flat-topped, near-opaque surfel, acting as a confidence dial for geometry. Our confidence-aware spawning uses geometric observability and multi-view depth agreement to initialize both opacity and shape, producing solid primitives where geometry is verifiable and soft, correctable primitives where it is not. As online densification, error-driven relocation recycles stale primitives into photometric-error hotspots. Bounded beta kernel support enables exact tile culling and faster rasterization, and a closed-form, shape-preserving merging formulation provides a level-of-detail (LOD) hierarchy for fast optimization and real-time rendering. On Mip-NeRF 360 in the on-the-fly protocol, HIBS improves PSNR by up to 0.8 dB while reducing optimization time by up to 40%.

## 1. Introduction

3D Gaussian Splatting (3DGS) [26] has made high-quality radiance-field capture practical, but the standard pipeline is offline: images are collected, poses are recovered by structure-from-motion [51], and the scene is optimized for tens of thousands of iterations. Recent work instead reconstructs during capture [25, 39, 41], providing immediate feedback and reducing the delay from capture to observable reconstruction to seconds. We target this online regime by reconsidering the primitive itself.

Online reconstruction inverts the assumptions of offline 3DGS. Each incoming image is registered as a keyframe, spawns new primitives once, and is optimized only for a short burst before the next keyframe arrives. This leaves little room for gradient-driven clone/split densification, opacity resets, or long optimization schedules to repair poorly initialized primitives. In practice, what spawning produces is largely what the model keeps. The critical decisions— what to place and how much to trust it—must therefore be made at initialization from measurable evidence.

We argue that under these constraints the primitive itself can be improved, and that the Gaussian is suboptimal. A Gaussian is fully opaque only at its center and translucent elsewhere; representing a solid surface therefore requires overlapping many Gaussians, together with the densification and long optimization that offline 3DGS can afford.

Without this machinery, online reconstruction must spawn translucent Gaussians and hope a short optimization burst can arrange them correctly.

We instead use the beta kernel [35], whose bounded profile has a learnable shape parameter b that continuously interpolates between a Gaussian-like volume and a flattopped, near-opaque surfel. A single beta primitive can thus be a flat-topped surface patch while retaining a soft representation for uncertain geometry, thin structures, or volume.

While Deformable Beta Splatting [35] uses this parameter offline as a compactness degree of freedom learned over long optimization schedules, we use it from geometric evidence as a confidence axis. The key question is therefore not only how to optimize the shape, but whether the available evidence justifies treating a primitive as a solid surface.

That evidence must be geometric, not merely photometric. Naively spawning solid kernels is harmful: a compact, near-opaque primitive at the wrong depth blocks transmittance and suppresses gradients behind it. Conversely, a low photometric error does not guarantee correct geometry. When the triangulation baseline is small relative to depth, many depths explain the observed views similarly well, yet only a narrow range remains consistent under parallax.

We therefore use the beta shape parameter as a confidence dial. Confidence-aware spawning combines two independent geometric cues, Multi-view stereo (MVS) agreement with an aligned monocular prior and depth observability, to initialize each primitive along this soft-to-solid axis: well-constrained candidates are spawned solid, while ambiguous ones remain soft and correctable. As additional views arrive, a lifetime controller updates the same parameter from accumulated multi-view evidence, progressively turning confirmed geometric confidence into solidity.

To compensate for the absence of offline densification, we introduce error-driven relocation. Near-dead primitives are recycled into photometric-error hotspots, where render-preserving copies provide additional capacity precisely where the current model is deficient.

The beta representation is also performant. Its bounded support yields an exact tile-culling bound and faster rasterization, while its closed-form moments enable a shapeaware level-of-detail hierarchy for large captures.

Together, these components yield an online reconstruction system in which the primitive is adapted to the short optimization horizon rather than relying on offline densification to compensate for it. Our experiments show improved reconstruction quality and efficiency across online, benchmark, and large-scale captures.

In summary, our contributions are:

• We identify the Gaussian primitive as suboptimal in the spawn-once, optimize-briefly regime and adopt the beta kernel as a confidence-adaptive online primitive whose shape ranges from soft to solid.

• We introduce confidence-aware spawning, which initializes opacity and shape from independent depth agreement and geometric observability.

• We introduce error-driven relocation, a render-preserving online alternative to clone/split densification that recycles near-dead primitives into photometric-error hotspots.

• We derive a closed-form merge for beta primitives, preserving their shape statistics and enabling a level-of-detail hierarchy for large-scale reconstruction.

• We develop a fast beta rasterizer with a shape-correct culling bound that exploits the kernel’s compact support.

The project page and source code can be found at: https://linus-franke.com/hibs.

## 2. Related Work

Novel View Synthesis. Neural Radiance Fields [3, 43] synthesize photorealistic novel views from posed images, but volumetric rendering remains computationally expensive. 3D Gaussian Splatting (3DGS) [26] instead enables efficient rasterization and fast optimization. Subsequent work improves compositing and rendering fidelity [12, 47, 56] and scales to large scenes with hierarchical representations such as Hierarchical 3DGS [27] and Octree-GS [48]. These methods are predominantly offline, relying on precomputed poses and long optimization schedules, whereas we target reconstruction during capture.

A range of systems reconstruct 3DGS maps incrementally from video, including SLAM-based methods [4, 19, 21, 25, 39, 45, 58] and joint pose-and-reconstruction approaches [10, 32]. SLAM methods spend far more per keyframe, e.g., hundreds of mapping iterations, densification, and long final refinement [39, 58]. Feed-forward and hybrid methods [23, 30, 31, 33] trade scene-specific optimization for speed, typically at reduced fidelity. Closest to our setting, On-the-Fly NVS [41] and i3DGS [42] reconstruct image streams with short optimization bursts and no densification, using learned matching and hierarchical structures. However, these methods retain the Gaussian and spawn it once per view. Our work instead revisits the primitive itself, introducing a controllable notion of solidity specifically for this short-burst regime.

Primitive Kernels and Densification. Alternative splatting primitives have been proposed to improve compactness, surface accuracy, or geometric fidelity, including generalized and radial kernels [14, 22], planar Gaussians [20], point sprites [2, 16, 50], linear and convex primitives [18, 53], volumetric ellipsoids [37], and trilinear splats [8]. Deformable radial kernels [22] are flexible, but planar and without closed-form moments, which our spawning and merge rely on. Most closely related, Deformable Beta Splatting (DBS) [35, 36] introduces a bounded beta kernel with a learnable shape parameter spanning Gaussianlike and surfel-like profiles. DBS learns this shape offline together with densification; we instead initialize it from geometric evidence and update it online throughout.

![](images/813e88129cbee13d1ed4c6598e8c8a5e6501b7b2c55589fbc6b4cfa7f66dcb6e.jpg)  
Figure 2. Overview of our pipeline. An RGB stream is registered with camera parameters estimated online [42]. Each keyframe spawns beta primitives using geometric observability and depth agreement, keeping uncertain primitives soft. Near-dead primitives are relocated to high-error regions, followed by a short 30-iteration optimization burst with our fused rasterizer. Periodically, small-footprint primitives are merged hierarchically using closed-form moment matching that preserves their solidity.

3DGS densification has been refined through errorand gradient-based splitting, budget-aware pruning, and MCMC-based relocation [7, 28, 49, 57]. In particular, 3DGS-MCMC [28] reuses low-opacity primitives by relocating them onto live ones with render-preserving corrections. These techniques assume long offline optimization and periodic densification, whereas our online setting has neither. We adapt relocation to this regime by recycling near-dead primitives and targeting them using attributed photometric error.

Efficient Rasterization. Recent work accelerates 3DGS through improved backward passes, gradient reduction, tighter tile bounds, and fused optimization [6, 13, 15, 38, 47]. We build on these ideas for the online burst, but derive the rasterization bounds for the bounded beta kernel. In particular, the Gaussian cutoff is not valid for low-opacity solid kernels, making the beta-specific bound a correctness requirement in addition to an efficiency improvement.

Depth Priors and Uncertainty. Monocular and multiview depth predictors are commonly used to initialize or regularize radiance-field reconstruction [33, 54, 55]; for example, Meuleman et al. [41] combine aligned monocular depth with guided MVS. Uncertainty is often estimated post-hoc or used for view selection [11, 24]. We instead incorporate uncertainty directly into the online primitive lifecycle: agreement between independent depth estimates and geometric observability determine the primitive’s opacity and shape, while the evolving shape field provides a live confidence representation.

## 3. Preliminaries: Online 3DGS

Online 3DGS systems [30, 41] reconstruct an RGB stream incrementally: each incoming frame is registered, used to spawn new primitives, and followed by a short optimization burst before the next keyframe. In i3DGS [42], registration matches each frame against previously registered keyframes using place recognition and learned feature matching [1, 34, 46], initializes the pose with PnP/RANSAC, and performs local bundle adjustment over a covisibility graph, all running in tens of milliseconds per frame.

After registration, pixels with local image detail are sampled and lifted to 3D using available multi-view and monocular depth cues [30, 41, 42]. The samples are instantiated as low-opacity, translucent isotropic Gaussians, followed by a short optimization burst of only a few dozen iterations. Crucially, these online systems do not perform the clone/split densification or opacity resets used by offline 3DGS, nor the long per-keyframe mapping and final refinement of Gaussian SLAM [39]: primitives are spawned once and then retained, making their initialization critical.

## 4. Method

Our method reconstructs a primitive-based radiance field online from an unordered stream of RGB images. We adop the pose-estimation backbone of Meuleman et al. [42], but redesign the primitive and its lifecycle. In the spawn-perview, short-burst regime, positions receive too little optimization to recover from poor initialization, while longer optimization overfits already-seen keyframes. We therefore make confidence explicit in the primitive representation: each primitive can be either a soft probe or a selfsupporting surface patch, and primitives are later relocated from unused capacity to image-space error hotspots. An overview of our method can be found in Fig. 2.

## 4.1. The Beta Kernel as a Confidence Dial

Following Deformable Beta Splatting [35], we replace the Gaussian falloff with a bounded beta kernel. With $x = r ^ { 2 } / 9$ in the primitive’s local frame (support $r \leq 3 .$ , using the common 3σ extent), a primitive of opacity o and shape parameter $b \in \mathbb { R }$ contributes

$$
\alpha = o \mathcal { B } ( x ; b ) , \qquad \mathcal { B } ( x ; b ) = ( 1 - x ) ^ { \beta ( b ) } , \qquad \beta ( b ) = 4 e ^ { b } .\tag{1}
$$

At $b ~ = ~ 0 ~ ( \beta ~ = ~ 4 )$ the profile closely matches a Gaussian; decreasing $\beta$ flattens it toward a solid plateau, while increasing $\beta$ produces a soft, rapidly vanishing blob. Thus opacity o controls contribution strength and $\beta$ controls the primitive’s degree of geometric commitment.

We also use the spherical-beta color basis of DBS,

$$
\mathbf { C } ( \mathbf { v } ) = \mathbf { c } _ { 0 } + \sum _ { k } \mathbf { a } _ { k } \operatorname* { m a x } ( 0 , \mathbf { d } _ { k } { \cdot } \mathbf { v } ) ^ { 4 e ^ { b _ { k } } } .\tag{2}
$$

Unlike offline DBS, where nonzero lobe amplitude initializations are used and optimized away over thousands of iterations, online primitives receive only a few dozen updates. We therefore initialize all lobe amplitudes at zero with random orientations. This removes the persistent viewdependent color bias of the offline initialization, and works as the amplitude gradient remains nonzero at zero.

## 4.2. Observability-Aware Spawning

Each registered keyframe spawns primitives where the current model under-explains image detail. With the short optimization burst following and no guarantee of repeated observations, opacity and shape must be initialized well from observable geometry. We combine geometric depth observability with cross-validation between two independent depth estimators, as also depicted in Fig. 3.

Depth observability. For a maximum covisible baseline B and candidate depth z, the triangulation angle is approximately $B / z$ . We therefore use

$$
c _ { \mathrm { z } } = \mathrm { c l a m p } \left( { \frac { B / z } { \theta _ { 0 } } } , 0 , 1 \right) , \qquad \theta _ { 0 } = 2 ^ { \circ } .\tag{3}
$$

Large $c _ { \mathrm { z } }$ indicates that depth is geometrically constrained; small $c _ { \mathrm { z } }$ indicates that many depths explain the training views: the estimate is unreliable under novel-view parallax.

Confidence-aware spawning. Pixel p is sampled according to image detail of image I not already explained by the render of the current model R, following [41],

$$
p ( \mathbf { p } ) = \mathrm { m a x } \Big ( \phi s _ { I } ( \mathbf { p } ) , p _ { \mathrm { m i n } } \Big ) - \phi s _ { R } ( \mathbf { p } ) ,\tag{4}
$$

![](images/335b6c297daf6b7a95abf7b244002d63e77731616234380f18bbb01dbf90a3e9.jpg)  
Figure 3. Spawning/parameter initialization. Based on depth observability $c _ { \mathbf { z } }$ and agreement of depths (orange window) $c _ { \mathrm { m v s } } ,$ the beta primitives are initialized as hard patches or soft probes.

where $s _ { I } , s _ { R }$ are blurred-Laplacian magnitudes, $p _ { \mathrm { m i n } }$ the lower probability bound, and the scalar ϕ the base sampling scale that sets the spawn density (Tab. 18).

Candidate depths $i _ { \mathrm { m v s } }$ are obtained with guided MVS in an inverse-depth window of radius $r _ { \mathrm { m v s } }$ around the metrically aligned monocular prior $i _ { \mathrm { m o n o } } .$ . We introduce their agreement as an independent confidence measure:

$$
c _ { \mathrm { m v s } } = \mathrm { c l a m p } \big ( 1 - | i _ { \mathrm { m v s } } - i _ { \mathrm { m o n o } } | / r _ { \mathrm { m v s } } , 0 , 1 \big ) ,\tag{5}
$$

set to zero for flat matching-cost profiles. We combine the two signals as $c = ( c _ { \mathrm { m v s } } \cdot c _ { \mathrm { z } } ) ^ { \gamma }$ and initialize

$$
\begin{array} { r l } & { o _ { \mathrm { s p a w n } } = o _ { 0 } + ( o _ { 1 } - o _ { 0 } ) c , } \\ & { \beta _ { \mathrm { s p a w n } } = \beta _ { \mathrm { s o f t } } - ( \beta _ { \mathrm { s o f t } } - \beta _ { \mathrm { s o l i d } } ) c , } \end{array}\tag{6}
$$

with $\gamma = 4 , o _ { 0 } = 0 . 1 , o _ { 1 } = 0 . 3 , \beta _ { \mathrm { s o l i d } } = 1$ , and $\beta _ { \mathrm { { s o f t } } } = 5$ When MVS provides no reliable correspondence $( c _ { \mathrm { m v s } } ~ =$ 0), we use the conservative opacity floor $o _ { \mathrm { f l o o r } } = 0 . 0 2$

The primitive scale is calibrated for $\beta = 4$ and compensated to preserve the half-maximum footprint:

$$
s  s \sqrt { \frac { u _ { 1 / 2 } ( \beta _ { 1 } ) } { u _ { 1 / 2 } ( \beta _ { 2 } ) } } , \qquad u _ { 1 / 2 } ( \beta ) = 1 - 2 ^ { - 1 / \beta } ,\tag{7}
$$

with $\beta _ { 1 } = 4 \mathrm { a n d } \beta _ { 2 } = \beta _ { \mathrm { s p a w n } }$ 1

## 4.3. Controlling Primitive Solidity

The spawn decision is extended over the primitive’s lifetime by a deterministic controller applied once per keyframe. For each visible primitive, we track exponential moving averages $( \eta = 0 . 3 )$ of the mean e¯ and variance $\sigma _ { e } ^ { 2 }$ of the photometric error over the pixels it dominates, across views. The shape parameter is updated as

$$
b  b \pm \Delta b , \qquad \Delta b = 0 . 4 , \qquad \beta = 4 e ^ { b } ,
$$

with $\beta$ decreased (hardened) when the primitive is old $( \geq 1 0$ keyframes), opaque $( o > 0 . 7 )$ , well observed (>20 dominated pixels, statistics from ${ \ge } 3$ views), and has low mean error $( \bar { e } ~ < ~ 0 . 0 3 )$ , and increased (softened) toward $\beta ~ = ~ 5$ only when well observed high error is strongly view-inconsistent $( \bar { e } > 0 . 0 3 , \bar { e } ^ { 2 } / ( \bar { e } ^ { 2 } + \sigma _ { e } ^ { 2 } ) < 0 . 5 )$ . Viewconsistent error is left to relocation (see next section), since it indicates geometric error rather than view-dependent appearance. Updates are clamped to $\beta ~ \in ~ [ 1 , 5 ]$ and use the scale compensation from Eq. (7) with $\beta _ { 1 } ~ = ~ \beta _ { \mathrm { o l d } }$ and $\beta _ { 2 } = \beta _ { \mathrm { n e w } }$ . Note that through gradient descent, $\beta$ is free to be optimized outside the clamped range.

![](images/5c0e3da7d27df0de1be33a50713c9a63d5d127436fea8615669d43928e878f39.jpg)  
Figure 4. Online confidence field. After the first orbit (left), many background primitives remain uncertain, and the table reconstruction is fuzzy. After the second orbit (middle), accumulated multiview evidence solidifies most primitives. In the final reconstruction (right, novel view rendering), these areas are largely solidified, while uncertainty remains around the reflective table center.

Thus $\beta$ acts as an evidence-driven confidence state: low $\beta$ denotes earned geometric commitment, while high $\beta$ denotes unresolved or view-dependent evidence. The resulting shape field provides an immediate confidence map of the reconstruction, as illustrated in Fig. 4.

## 4.4. Error-Driven Incremental Relocation

The overall amount of new detail that can be added per keyframe is limited, and offline clone/split heuristics require longer optimization. Instead, inspired by the relocation mechanism of 3DGS-MCMC [28], we introduce an error-targeted split: near-dead primitives are recycled by teleporting them to poorly reconstructed live primitives.

Once per keyframe, dead primitives, being below the recycling-opacity threshold, outside the hierarchy, and past the spawn-protection window, serve as sources. Live destinations are sampled according to their per-primitive error attribution, weighted by error consistency and restricted to the top 10%, so high-error regions receive more capacity. Each selected source is teleported to be a copy of the destination’s position, shape, and color, producing stacks of relocated primitives whose size grows with residual error.

Before optimization, copy stacks are made render preserving, by adjusting their combined opacity and scale [28]. Small positional jitter and optimizer resets allow the subsequent burst to separate the copies and represent more detail.

![](images/de62169868d4ea4bbde0f272cc4166a5da9ff923ec92bf61191a33a4fb9bb0d1.jpg)  
Figure 5. Merging beta primitives. KL-style merging of beta primitives (middle) similar to merging of Gaussians (left) leads to oversized and solid primitives. Instead, we use moment merging (right), keeping the footprint.

## 4.5. Per-keyframe Optimization Burst

Each burst minimizes the photometric objective [26],

$$
\mathcal { L } = ( 1 - \lambda ) \mathcal { L } _ { 1 } + \lambda ( 1 - \mathcal { L } _ { \mathrm { S S I M } } ) + \lambda _ { \mathcal { R } } \mathcal { R } _ { \mathrm { o } } + \lambda _ { \mathcal { R } } \mathcal { R } _ { \Sigma } , \quad \lambda = 0 . 1 5 ,\tag{8}
$$

with learned exposure/ISP [5] correction applied before comparison. We additionally use the opacity $( \mathcal { R } _ { \mathrm { o } } )$ and scale $( \mathcal { R } _ { \Sigma } )$ regularizers of 3DGS-MCMC [28], also used by DBS [35], both with weight $\lambda _ { \mathcal { R } } = 1 0 ^ { - 2 }$ . The opacity penalty drives unsuccessful primitives toward the recycling band rather than pruning them outright.

## 4.6. Fast Beta Rasterization

The bounded support of the beta kernel permits tighter rasterization bounds than the Gaussian. For opacity o, the normalized squared radius at the rasterizer’s 1/255 cutoff is $x _ { \mathrm { c u t } } = \bar { \operatorname* { m a x } } ( 0 , 1 - ( 2 5 5 o ) ^ { - 1 / \beta ( b ) } )$ . We use this bound for tile culling. The standard 3DGS Gaussian threshold is insufficient for solid, low-opacity beta kernels and can discard tiles containing nonzero contributions; the betaspecific bound is therefore both a correctness fix and a speedup.

The optimization burst is further accelerated with a FasterGS-style fused rasterizer [13]: backward recomputes forward quantities per primitive, keeps intermediate gradients in registers, and applies sparse Adam directly to the optimizer state. Primitive gradients therefore never materialize in memory. The regularizers are also applied analytically inside the fused backward/optimizer update. This rasterizer substantially reduces computation time.

## 5. Hierarchical Beta Primitives

Kilometre-scale captures exceed VRAM, so we maintain a screen-space level-of-detail hierarchy [27, 42]. Groups of four neighbouring primitives that each project below half a pixel are replaced by a coarse parent; inactive fine levels are offloaded to host memory. For sub-pixel groups, the parent only needs to preserve the integrated footprint, which we approximate by conserving 3D mass. Unlike a Gaussian, however, our primitive also carries shape, so the merge must preserve solidity (Fig. 5).

Table 1. Results for novel view synthesis on pose-free methods. The best and second best are color coded for pose-free methods. Times for G+Beta-Splatting and G+Faster-GS (offline baselines at 7k as a reference) include GLOMAP pose estimation (about 50s for TUM, 174s for MipNeRF360 (otf) and 58min for StaticHikes). <sup>†</sup> numbers taken from i3DGS [42], with times roughly scaled to our setup.
<table><tr><td rowspan="2"></td><td colspan="4">TUM</td><td colspan="4">MIPNERF360 (otf)</td><td colspan="4">STATICHIKES</td></tr><tr><td>PSNR↑</td><td></td><td>SSIM↑ LPIPS↓</td><td>Time↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>Time↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td> $\mathrm { T i m e ^ { \downarrow } }$ </td></tr><tr><td>G+Beta-Splatting (7k)</td><td>22.73</td><td>0.842</td><td>0.214</td><td>0:01:50</td><td>27.75</td><td>0.863</td><td>0.232</td><td>0:04:34</td><td>20.36</td><td>0.538</td><td>0.470</td><td>0:59:22</td></tr><tr><td>G+Faster-GS (7k)</td><td>23.08</td><td>0.845</td><td>0.207</td><td>0:01:42</td><td>27.81</td><td>0.876</td><td>0.210</td><td>0:04:22</td><td>20.57</td><td>0.547</td><td>0.459</td><td>0:59:38</td></tr><tr><td>MonoGS†</td><td>16.60</td><td>0.682</td><td>0.381</td><td>~15min</td><td>14.46</td><td>0.436</td><td>0.663</td><td>~4min</td><td>15.46</td><td>0.301</td><td>0.659</td><td>~9min</td></tr><tr><td>S3PO-GS with refinement†</td><td>21.88</td><td>0.777</td><td>0.309</td><td>~1h</td><td>19.82</td><td>0.553</td><td>0.576</td><td>~20min</td><td>18.09</td><td>0.368</td><td>0.611</td><td>~50min</td></tr><tr><td>LongSplat†</td><td>26.51</td><td>0.875</td><td>0.176</td><td>~1.5h</td><td>23.63</td><td>0.657</td><td>0.415</td><td>~2h</td><td>20.05</td><td>0.461</td><td>0.481</td><td>~20h</td></tr><tr><td>On-The-Fly NVS</td><td>23.37</td><td>0.824</td><td>0.241</td><td>0:01:00</td><td>22.51</td><td>0.671</td><td>0.370</td><td>0:01:45</td><td>21.14</td><td>0.619</td><td>0.338</td><td>0:02:08</td></tr><tr><td>i3DGS</td><td>25.36</td><td>0.858</td><td>0.188</td><td>0:01:24</td><td>26.66</td><td>0.835</td><td>0.224</td><td>0:02:09</td><td>22.17</td><td>0.699</td><td>0.269</td><td>0:02:53</td></tr><tr><td>Ours</td><td>25.94</td><td>0.868</td><td>0.175</td><td>0:01:44</td><td>27.47</td><td>0.852</td><td>0.204</td><td>0:01:18</td><td>22.49</td><td>0.710</td><td>0.259</td><td>0:02:02</td></tr></table>

Closed-form merging by moment matching. Each primitive carries an opacity $^ { O , }$ center $\mu ,$ shape matrix $\Sigma ,$ and scalar shape parameter b, with $\beta = 4 e ^ { b }$ . Its compactly supported kernel is $\rho ( \mathbf { x } ) = o ( 1 - \| \mathbf { w } \| ^ { 2 } / 9 ) ^ { \beta }$ , where w is the whitened offset.

Two properties make closed-form merging possible. First, the mass $Z ,$ covariance $\mathrm { C o v } = \xi ( \beta ) \Sigma$ with shrink factor $\xi ( \beta ) = 4 . 5 / ( \beta + 2 . 5 )$ , and Mardia kurtosis $\kappa ( \beta ) =$ $1 5 ( \beta + 2 . 5 ) / ( \beta + 3 . 5 )$ all have closed forms (Sec. A). Here Σ parameterizes the support ellipsoid rather than the spread, since peakier profiles concentrate mass near the center. Second, κ is strictly monotone in $\beta$ with $1 5 - \kappa ( \beta ) =$ $1 5 / ( \beta + 3 . 5 )$ , so on the band $( 7 5 / 7 , 1 5 )$ it has the closedform inverse $\beta = 1 5 / ( 1 5 - \kappa ) - 3 . 5$ and uniquely identifies the profile within the family.

We merge primitives by moment matching rather than KL minimization. For compactly supported kernels, KL is unsuitable because any nonzero mass outside the parent’s support makes the divergence infinite.

With mass weights $w _ { i } \propto Z _ { i }$ , offsets $\delta _ { i } = \mu _ { i } - \mu ^ { \star }$ , and mixture kurtosis $\kappa _ { \mathrm { m i x } }$ (a closed-form mass-weighted sum of per-child trace terms; see Sec. A), the parent is

$$
\mu ^ { \star } = \sum _ { i } w _ { i } \mu _ { i } , \qquad \Sigma _ { \operatorname* { m i x } } = \sum _ { i } w _ { i } \big ( \xi \big ( \beta _ { i } \big ) \Sigma _ { i } + \delta _ { i } \delta _ { i } ^ { \top } \big ) ,\tag{9}
$$

$$
\beta ^ { \star } = \frac { 1 5 } { 1 5 - \hat { \kappa } } - 3 . 5 , \quad \Sigma ^ { \star } = \frac { \Sigma _ { \operatorname* { m i x } } } { \xi ( \beta ^ { \star } ) } , \quad \sigma ^ { \star } = \frac { \sum _ { i } Z _ { i } } { Z ( \beta ^ { \star } , \Sigma ^ { \star } ) } ,\tag{10}
$$

with $b ^ { \star } = \ln ( \beta ^ { \star } / 4 )$ and κˆ the mixture kurtosis $\kappa _ { \mathrm { m i x } }$ clipped to the representable band of the beta family (Sec. A). The covariance follows from the law of total covariance, dividing out the shrink factor turns it back into a shape matrix, and mass conservation determines the opacity.

Large-scale training. The hierarchy is rebuilt periodically rather than per keyframe, with dynamic-object and sky masks applied when available. To prevent redundant spawning, views observing primitives below one-quarter of their finest observed resolution are excluded from spawning and receive reduced update weights. Finally, a targeted repair pass allocates 1000 iterations to the most deteriorated training views, identified by photometric errors exceeding (1.5×) the median, mitigating recency bias.

## 6. Results and Experiments

We evaluate using the online protocols of Meuleman et al. [41] on three MipNeRF360 scenes (garden, counter, bonsai, marked with otf) [3], three TUM scenes [52], and three StaticHikes captures [40], with the same heldout views for evaluation. We additionally evaluate on the 3DGS benchmark [3, 17, 26, 29] and two large-scale captures, SMALLCITY [27] and CITYWALK [41]. We compare to methods including MonoGS [39], S3PO-GS [4], LongSplat [32], On-The-Fly NVS [41], i3DGS [42], DROID-Splat [19], CF-3DGS [10], DepthAnything3 [33], ARTDECO [30], H3DGS [27] and Octree-GS [48], as well as GLOMAP-based [44, 51] offline references using Beta Splatting [35] and Faster-GS [13]. We report PSNR, SSIM, LPIPS, and total time, averaging ours over five runs.

Unless noted otherwise, experiments use an RTX 5090 (32 GB) and i9-14900KS with 64GB DDR4 RAM on Windows 11. For methods requiring reduced input resolution, we follow [41, 42]. These results are reported in Sec. D.

## 6.1. Results

Tab. 1 evaluates on the MipNeRF360 (otf), TUM, and StaticHikes scenes against pose-free and online baselines. In this benchmark, our method outperforms our closest competitor i3DGS (except time on TUM), and is consistently among the strongest methods overall. On MipNeRF360 (otf) and StaticHikes, we surpass competing posefree methods on all three metrics while reducing computation time by up to 40%. On TUM, LongSplat has slightly better PSNR and SSIM, but takes roughly 50× longer.

BetaSplatting (7K) [35]  
FasterGS (7K) [13]  
i3DGS [42]  
Ours  
GT  
![](images/3c64e3d87c5c199eacebbf123c8196c2124e0f32484ff81e4621649ea7939439.jpg)  
Figure 6. Qualitative comparison against baselines on the 3DGS benchmark: TRUCK, and FLOWERS scenes.

Table 2. Results on the full 3DGS benchmark. We compare with online method i3DGS as well as FasterGS and BetaSplatting with GLOMAP sequential mapping (requiring 1:52 on MipNeRF360, 1:42 on Tanks&Temples and failing on Deep Blending). More expensive exhaustive mapping is in Tab. 16. The best and second best are color coded.
<table><tr><td rowspan="2" colspan="2"></td><td colspan="4">MIPNERF360</td><td colspan="4">TANKS AND TEMPLES</td><td colspan="4">DEEP BLENDING</td></tr><tr><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>Time↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>Time↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>Time↓</td></tr><tr><td></td><td>BetaSplatting (30K)</td><td>25.90</td><td>0.703</td><td>0.333</td><td>0:09:07</td><td>23.31</td><td>0.831</td><td>0.190</td><td>0:08:27</td><td>×</td><td>×</td><td>×</td><td>×</td></tr><tr><td>Geg) ,)</td><td>FasterGS (30K)</td><td>25.53</td><td>0.702</td><td>0.356</td><td>0:05:29</td><td>22.17</td><td>0.802</td><td>0.250</td><td>0:04:21</td><td>X</td><td>X</td><td>X</td><td>X</td></tr><tr><td></td><td>BetaSplatting (7K)</td><td>24.05</td><td>0.656</td><td>0.396</td><td>0:03:17</td><td>20.81</td><td>0.733</td><td>0.321</td><td>0:02:55</td><td>X</td><td>X</td><td>X</td><td>X</td></tr><tr><td></td><td>FasterGS (7K)</td><td>24.57</td><td>0.678</td><td>0.411</td><td>0:02:29</td><td>20.79</td><td>0.742</td><td>0.343</td><td>0:02:10</td><td>X</td><td>×</td><td>×</td><td>×</td></tr><tr><td colspan="2">i3DGS</td><td>24.75</td><td>0.721</td><td>0.280</td><td>0:01:59</td><td>22.08</td><td>0.783</td><td>0.226</td><td>0:01:25</td><td>27.16</td><td>0.856</td><td>0.285</td><td>0:02:03</td></tr><tr><td colspan="2">Ours</td><td>25.19</td><td>0.731</td><td>0.264</td><td>0:01:17</td><td>23.00</td><td>0.803</td><td>0.200</td><td>0:01:18</td><td>26.49</td><td>0.847</td><td>0.295</td><td>0:01:22</td></tr></table>

On the standard 3DGS benchmark (Tab. 2 and Fig. 6), we compare against BetaSplatting and FasterGS with sequential (fast) GLOMAP pose estimation (more expensive exhaustive setting in Tab. 16). Our method is consistently the fastest, outperforming in SSIM and LPIPS on MipNeRF360 while remaining competitive on Tanks&Temples. Deep-Blending is more challenging; however replacing sphericalbeta colors with SH improves our result by ∼ 0.3 dB (Sec. E). On MipNeRF360, sequential matching can drop frames (e.g., bicycle), and metrics are therefore computed only on successfully registered views. Against exhaustive-GLOMAP references (Tab. 16), we remain competitive at 7k iterations, while 30k iterations achieve higher quality at ∼ 6–12× our runtime and require the complete capture to be available beforehand.

Tab. 3 evaluates the large scale SmallCity and CityWalk captures. Our method achieves the best metric on both scenes except for SSIM and LPIPS on SmallCity, where Octree-GS is marginally better but requires about one hour versus roughly 4 minutes for us. Compared with On-The-Fly NVS, we are ∼ 50% slower while improving PSNR by 0.86 dB on SmallCity and 4.1 dB on CityWalk.

Table 3. Results for large scale scenes. Offline methods (H3DGS, Octree-GS) include COLMAP times. Comparing image metrics are from [42], non-approximate times measured on our machine. S3PO-GS includes refinement. The best and second best are color coded.
<table><tr><td></td><td colspan="3">SMALLCITY PSNR↑ SSIM↑ LPIPS↓ Time↓</td><td colspan="4">CITYWALK PSNR↑</td></tr><tr><td rowspan="2">H3DGS</td><td colspan="3"></td><td colspan="2"></td><td colspan="2">SSIM↑ LPIPS↓ Time↓</td></tr><tr><td>21.17</td><td>0.679</td><td>0.285</td><td>~2.5h|</td><td>11.78 0.557</td><td>0.560</td><td>~20h</td></tr><tr><td rowspan="3">Octree-GS S3PO-GS</td><td>24.18</td><td>0.831 0.273</td><td>~1h</td><td>17.33</td><td>0.638</td><td>0.520</td><td>~2.5h</td></tr><tr><td>19.71</td><td>0.652</td><td>0.462 ~0.7h</td><td>10.72</td><td>0.439</td><td>0.579</td><td>~5.5h</td></tr><tr><td>23.59</td><td>0.789 0.323</td><td>02:21</td><td>21.71</td><td>0.712</td><td>0.395</td><td>29:35</td></tr><tr><td rowspan="2">i3DGS Ours</td><td>23.56</td><td>0.800</td><td>0.302</td><td>02:54</td><td>22.03</td><td>0.772</td><td>0.336</td><td>45:20</td></tr><tr><td>24.45</td><td>0.808</td><td>0.285</td><td>03:47</td><td>25.81</td><td>0.857</td><td>0.239</td><td>42:43</td></tr></table>

## 6.2. Ablation Studies

Tab. 4 (top) ablates the components of our final configuration across the nine otf scenes. Spherical-beta lobes (SB) provide a small quality and runtime improvement over SH, although SH is preferable on scenes such as Deep Blending (Sec. E). Confidence-aware spawning, solidification, and zero-initialized lobes each improve reconstruction quality.

Table 4. Component ablation (leave-one-out from full method). Top table: Metrics are averaged over all nine OtF scenes (Tab. 1).
<table><tr><td colspan="2">Configuration</td><td>|PSNR↑</td><td>SSIM↑</td><td>LPIPS↓ t(s)</td><td></td></tr><tr><td colspan="2">Full (ours)</td><td>25.30</td><td>0.810</td><td>0.213</td><td>101</td></tr><tr><td colspan="2">w/o SB lobes (→ SH) [Sec. 4.1] DBS lobe init prior [Sec. 4.1]</td><td>25.12 25.19</td><td>0.807 0.809</td><td>0.214</td><td>106</td></tr><tr><td colspan="2">w/o confidence</td><td colspan="2">[Sec. 4.2] 25.14</td><td>0.214</td><td>100</td></tr><tr><td colspan="2">w/o solidification</td><td colspan="2">[Sec. 4.3] 25.15</td><td>0.808 0.215 0.807</td><td>99</td></tr><tr><td colspan="2">w/o error relocation [Sec. 4.4]</td><td colspan="2">25.09</td><td>0.216 0.804 0.219</td><td>100 95</td></tr><tr><td colspan="2">w/o regularizers [Sec. 4.5]</td><td colspan="2">24.98</td><td>0.216</td><td>108</td></tr><tr><td colspan="2">w/o fused rasterizer [Sec. 4.6]</td><td colspan="2">25.24</td><td>0.803 0.809 0.214</td><td>120</td></tr><tr><td colspan="2">Large-scene confs</td><td colspan="2">SMALLCITY PSNR↑ SSIM↑ LPIPS↓</td><td colspan="2">CITYWALK</td></tr><tr><td colspan="2">[Sec. 5] Full (ours) 24.45</td><td colspan="2">0.808 0.285 25.81</td><td colspan="2">PSNR↑ SSIM↑ LPIPS↓</td></tr><tr><td colspan="2">23.78</td><td colspan="2"></td><td>0.857</td><td>0.239</td></tr><tr><td colspan="2">w/o moment merge 23.60</td><td colspan="2">0.800 0.299 0.798 0.295</td><td>23.54 0.833 23.44</td><td>0.262</td></tr><tr><td colspan="2">w/o error repair w/o res-weighting</td><td colspan="2">0.805 0.294</td><td>0.843</td><td>0.244 0.294</td></tr><tr><td colspan="2">24.12</td><td colspan="2"></td><td>22.93 0.812</td></tr></table>

Table 5. Kernel ablation. Metrics averaged over all OtF scenes.
<table><tr><td>Configuration</td><td>Confidence</td><td>PSNR↑</td><td>SSIM↑ LPIPS↓ t (s)</td><td></td><td></td></tr><tr><td>i3DGS [42]</td><td>none</td><td>24.73</td><td>0.797</td><td>0.227</td><td>129</td></tr><tr><td>Gaussian</td><td>Ospawn</td><td>25.09</td><td>0.808</td><td>0.214</td><td>130</td></tr><tr><td>β frozen at 4</td><td>Ospawn</td><td>25.06</td><td>0.806</td><td>0.217</td><td>100</td></tr><tr><td>learnable β</td><td>Ospawn</td><td>25.11</td><td>0.807</td><td>0.215</td><td>100</td></tr><tr><td>learnable β</td><td> $\sigma _ { \mathrm { s p a w n } } , \beta _ { \mathrm { s p a w n } }$ </td><td>25.15</td><td>0.807</td><td>0.216</td><td>100</td></tr><tr><td>Full (ours)</td><td> $o _ { \mathrm { s p a w n } } , \beta _ { \mathrm { s p a w n } , \mathrm { s o l i d i f y } }$ </td><td>25.30</td><td>0.810</td><td>0.213</td><td>101</td></tr></table>

The fused rasterizer reduces runtime by about 16%; the 30- iteration optimization burst reduces from 287ms to 170ms (GARDEN). Error-driven relocation yields a small runtime penalty of ∼ 6% but improves quality. The opacity and scale regularizers further improve both runtime and quality, by feeding ineffective primitives into relocation.

Tab. 4 (bottom) evaluates the large-scale hierarchical setting (Sec. 5). Our moment-based merge consistently outperforms the opacity-weighted baseline by avoiding overly large parent primitives, with the largest gain on CITYWALK (+2.27 dB) where multiple hierarchy levels are required. Resolution-aware weighting and error-based repair further improve quality by preventing late, low-resolution observations from degrading previously well-reconstructed regions.

Tab. 5 evaluates the effect of different kernel choices on the nine otf scenes. Beta kernels provide a small quality improvement over Gaussians, while also being faster due to their bounded support. Together with confidence-aware spawning, they provide a +0.2 dB improvement.

## 6.3. Quality During Reconstruction

An online system should improve throughout reconstruction, not only in the final model. Fig. 7 (top) reports mean PSNR over the test views registered so far as reconstruction progresses. Our method consistently outperforms i3DGS on MipNeRF360 (otf), TUM, and StaticHikes, with the gap emerging shortly after bootstrap and persisting throughout the capture. The bottom row measures how much subsequent keyframes improve a test view after it enters the map: each point compares its PSNR immediately after registration with its PSNR just before the next held-out view is registered. Nearly all points lie above the diagonal, indicating continued improvement without forgetting.

![](images/682b71c5d12c7acd20eb45ca48e834a9128998b91441580ee0da191b8c828ae1.jpg)  
Figure 7. Quality during reconstruction. Top: mean PSNR over test views registered so far versus reconstruction progress, for our method and i3DGS. Bottom: PSNR immediately after each view’s registration versus just before the next test view arrives. Points above the diagonal show improvement after registration.

## 7. Conclusion

Limitations and Future Work. On the 3DGS benchmark, experiments show a remaining quality gap to fully optimized offline methods, suggesting that further improvements in online pose refinement [42]—orthogonal to our work—are a promising direction for closing the remaining gap. The merge approximates out-of-band groups and the rendered contribution, and shuffled input can reduce quality. Further promising directions include integration with VR [9] and using the confidence map for refinement [11].

Conclusion. We present an online reconstruction framework that redesigns the primitive lifecycle around the short optimization horizon of incremental capture. Our bounded beta kernel encodes a continuous notion of solidity, observability-aware spawning initializes this confidence from geometric evidence, and error-driven relocation reallocates unused capacity to poorly explained regions. Closedform moment matching enables a shape-aware hierarchy, while bounded-support rasterization and fused optimization make the representation efficient for large-scale online reconstruction. Together, these components yield faster and higher-quality reconstruction throughout capture, while turning the primitive shape field into a live confidence visualization.

## Acknowledgments

We thank the anonymous reviewers for their constructive feedback.

## References

[1] Amar Ali-bey, Brahim Chaib-draa, and Philippe Giguere.\` MixVPR: Feature mixing for visual place recognition. In IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 2023. 3, 13

[2] Kara-Ali Aliev, Artem Sevastopolsky, Maria Kolos, Dmitry Ulyanov, and Victor Lempitsky. Neural point-based graphics. In European Conference on Computer Vision (ECCV), 2020. 2

[3] Jonathan T. Barron, Ben Mildenhall, Dor Verbin, Pratul P. Srinivasan, and Peter Hedman. Mip-NeRF 360: Unbounded anti-aliased neural radiance fields. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022. 2, 6

[4] Chong Cheng, Sicheng Yu, Zijian Wang, Yifan Zhou, and Hao Wang. Outdoor monocular SLAM with global scaleconsistent 3D Gaussian pointmaps. In IEEE/CVF International Conference on Computer Vision (ICCV), 2025. 2, 6

[5] Isaac Deutsch, Nicolas Moenne-Loccoz, Gavriel State, and¨ Zan Gojcic. PPISP: Physically-plausible compensation and control of photometric variations in radiance field reconstruction. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 7289–7298, 2026. 5

[6] Sankeerth Durvasula, Adrian Zhao, Fan Chen, Ruofan Liang, Pawan Kumar Sanjaya, and Nandita Vijaykumar. DISTWAR: Fast differentiable rendering on raster-based rendering pipelines. arXiv preprint arXiv:2401.05345, 2024. 3

[7] Guangchi Fang and Bing Wang. Mini-Splatting: Representing scenes with a constrained number of Gaussians. In European Conference on Computer Vision (ECCV), 2024. 3

[8] Linus Franke, Darius Ruckert, Laura Fink, and Marc Stam-¨ minger. TRIPS: Trilinear point splatting for real-time radiance field rendering. Computer Graphics Forum, 43(2), 2024. 2

[9] Linus Franke, Laura Fink, and Marc Stamminger. VR-Splatting: Foveated radiance field rendering via 3D Gaussian splatting and neural points. Proc. ACM Comput. Graph. Interact. Tech., 8(1), 2025. 8

[10] Yang Fu, Sifei Liu, Amey Kulkarni, Jan Kautz, Alexei A. Efros, and Xiaolong Wang. COLMAP-free 3D Gaussian splatting. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024. 2, 6

[11] Lily Goli, Cody Reading, Silvia Sellan, Alec Jacobson, and´ Andrea Tagliasacchi. Bayes’ rays: Uncertainty quantification for neural radiance fields. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024. 3, 8

[12] Florian Hahlbohm, Fabian Friederichs, Tim Weyrich, Linus Franke, Moritz Kappel, Susana Castillo, Marc Stamminger, Martin Eisemann, and Marcus Magnor. Efficient perspective-correct 3D Gaussian splatting using hybrid transparency. Computer Graphics Forum, 44(2), 2025. 2

[13] Florian Hahlbohm, Linus Franke, Martin Eisemann, and Marcus Magnor. Faster-GS: Analyzing and improving Gaussian splatting optimization. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2026. 3, 5, 6, 7, 13, 22, 23

[14] Abdullah Hamdi, Luke Melas-Kyriazi, Jinjie Mai, Guocheng Qian, Ruoshi Liu, Carl Vondrick, Bernard Ghanem, and An drea Vedaldi. GES: Generalized exponential splatting for efficient radiance field rendering. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024. 2

[15] Alex Hanson, Allen Tu, Geng Lin, Vasu Singla, Matthias Zwicker, and Tom Goldstein. Speedy-Splat: Fast 3D Gaussian splatting with sparse pixels and sparse primitives. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025. 3

[16] Mathias Harrer, Linus Franke, Laura Fink, Marc Stamminger, and Tim Weyrich. Inovis: Instant novel-view synthesis. In SIGGRAPH Asia 2023 Conference Papers, 2023. 2

[17] Peter Hedman, Julien Philip, True Price, Jan-Michael Frahm, George Drettakis, and Gabriel Brostow. Deep blending for free-viewpoint image-based rendering. ACM Transactions on Graphics, 37(6), 2018. 6

[18] Jan Held, Renaud Vandeghen, Abdullah Hamdi, Adrien Deliege, Anthony Cioppa, Silvio Giancola, Andrea Vedaldi, Bernard Ghanem, and Marc Van Droogenbroeck. 3D convex splatting: Radiance field rendering with 3D smooth convexes. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025. 2

[19] Christian Homeyer, Leon Begiristain, and Christoph Schnorr. DROID-Splat: Combining end-to-end SLAM with¨ 3D Gaussian splatting. arXiv preprint arXiv:2411.17660, 2024. 2, 6

[20] Binbin Huang, Zehao Yu, Anpei Chen, Andreas Geiger, and Shenghua Gao. 2D Gaussian splatting for geometrically accurate radiance fields. In ACM SIGGRAPH 2024 Conference Papers, 2024. 2

[21] Huajian Huang, Longwei Li, Hui Cheng, and Sai-Kit Yeung. Photo-SLAM: Real-time simultaneous localization and photorealistic mapping for monocular, stereo, and RGB-D cameras. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024. 2

[22] Yi-Hua Huang, Ming-Xian Lin, Yang-Tian Sun, Ziyi Yang, Xiaoyang Lyu, Yan-Pei Cao, and Xiaojuan Qi. Deformable radial kernel splatting. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025. 2

[23] Lihan Jiang, Yucheng Mao, Linning Xu, Tao Lu, Kerui Ren, Yichen Jin, Xudong Xu, Mulin Yu, Jiangmiao Pang, Feng Zhao, et al. AnySplat: Feed-forward 3D Gaussian splatting from unconstrained views. ACM Transactions on Graphics, 44(6), 2025. 2, 14, 15

[24] Wen Jiang, Boshu Lei, and Kostas Daniilidis. FisherRF: Active view selection and uncertainty quantification for radiance fields using Fisher information. In European Confer ence on Computer Vision (ECCV), 2024. 3

[25] Nikhil Keetha, Jay Karhade, Krishna Murthy Jatavallabhula, Gengshan Yang, Sebastian Scherer, Deva Ramanan, and

Jonathon Luiten. SplaTAM: Splat, track & map 3D Gaussians for dense RGB-D SLAM. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024. 1, 2

[26] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuhler,¨ and George Drettakis. 3D Gaussian splatting for real-time radiance field rendering. ACM Transactions on Graphics, 42 (4), 2023. 1, 2, 5, 6, 18

[27] Bernhard Kerbl, Andreas Meuleman, Georgios Kopanas, Michael Wimmer, Alexandre Lanvin, and George Drettakis. A hierarchical 3D Gaussian representation for real-time rendering of very large datasets. ACM Transactions on Graphics, 43(4), 2024. 2, 5, 6

[28] Shakiba Kheradmand, Daniel Rebain, Gopal Sharma, Weiwei Sun, Yang-Che Tseng, Hossam Isack, Abhishek Kar, Andrea Tagliasacchi, and Kwang Moo Yi. 3D Gaussian splatting as Markov Chain Monte Carlo. In Advances in Neu ral Information Processing Systems (NeurIPS), 2024. 3, 5, 16, 18

[29] Arno Knapitsch, Jaesik Park, Qian-Yi Zhou, and Vladlen Koltun. Tanks and Temples: Benchmarking large-scale scene reconstruction. ACM Transactions on Graphics, 36 (4), 2017. 6

[30] Guanghao Li, Kerui Ren, Linning Xu, Zhewen Zheng, Changjian Jiang, Xin Gao, Bo DAI, Jian Pu, Mulin Yu, and Jiangmiao Pang. Artdeco: Toward high-fidelity on-the-fly reconstruction with hierarchical gaussian structure and feedforward guidance. In International Conference on Learning Representations, pages 102997–103029, 2026. 2, 3, 6

[31] Yang Li, Jinglu Wang, Lei Chu, Xiao Li, Shiu-hong Kao, Ying-Cong Chen, and Yan Lu. StreamGS: Online generalizable Gaussian splatting reconstruction for unposed image streams. In IEEE/CVF International Conference on Computer Vision (ICCV), 2025. 2

[32] Chin-Yang Lin, Cheng Sun, Fu-En Yang, Min-Hung Chen, Yen-Yu Lin, and Yu-Lun Liu. LongSplat: Robust unposed 3D Gaussian splatting for casual long videos. In IEEE/CVF International Conference on Computer Vision (ICCV), 2025. 2, 6

[33] Haotong Lin, Sili Chen, Jun Hao Liew, Donny Y. Chen, Zhenyu Li, Yang Zhao, Sida Peng, Hengkai Guo, Xiaowei Zhou, Guang Shi, Jiashi Feng, and Bingyi Kang. Depth Anything 3: Recovering the visual space from any views. In International Conference on Learning Representations (ICLR), 2026. 2, 3, 6, 14

[34] Philipp Lindenberger, Paul-Edouard Sarlin, and Marc Pollefeys. LightGlue: Local feature matching at light speed. In IEEE/CVF International Conference on Computer Vision (ICCV), 2023. 3, 13

[35] Rong Liu, Dylan Sun, Meida Chen, Yue Wang, and Andrew Feng. Deformable Beta splatting. In ACM SIGGRAPH 2025 Conference Papers, 2025. 2, 4, 5, 6, 7, 18, 22, 23

[36] Rong Liu, Zhongpai Gao, Benjamin Planche, Meida Chen, Van Nguyen Nguyen, Meng Zheng, Anwesa Choudhuri, Terrence Chen, Yue Wang, Andrew Feng, et al. Universal Beta splatting. In International Conference on Learning Representations (ICLR), 2026. 2

[37] Alexander Mai, Peter Hedman, Georgios Kopanas, Dor Verbin, David Futschik, Qiangeng Xu, Falko Kuester, Jonathan T. Barron, and Yinda Zhang. EVER: Exact volumetric ellipsoid rendering for real-time view synthesis. In IEEE/CVF International Conference on Computer Vision (ICCV), 2025. 2

[38] Saswat Subhajyoti Mallick, Rahul Goel, Bernhard Kerbl, Markus Steinberger, Francisco Vicente Carrasco, and Fernando De La Torre. Taming 3DGS: High-quality radiance fields with limited resources. In SIGGRAPH Asia 2024 Con ference Papers, 2024. 3

[39] Hidenobu Matsuki, Riku Murai, Paul H. J. Kelly, and Andrew J. Davison. Gaussian splatting SLAM. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024. 1, 2, 3, 6

[40] Andreas Meuleman, Yu-Lun Liu, Chen Gao, Jia-Bin Huang, Changil Kim, Min H Kim, and Johannes Kopf. Progressively optimized local radiance fields for robust view synthesis. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023. 6

[41] Andreas Meuleman, Ishaan Shah, Alexandre Lanvin, Bernhard Kerbl, and George Drettakis. On-the-fly reconstruction for large-scale novel view synthesis from unposed images. ACM Transactions on Graphics, 44(4), 2025. 1, 2, 3, 4, 6, 18, 20, 21

[42] Andreas Meuleman, Linus Franke, Boris Zhestiankin, Camille Montemagni, and George Drettakis. Immediate 3D Gaussian splat reconstruction of unordered input with global consistency. In ACM SIGGRAPH 2026 Conference Papers, 2026. 1, 2, 3, 5, 6, 7, 8, 13, 15, 20, 21, 22, 23

[43] Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, and Ren Ng. NeRF: Representing scenes as neural radiance fields for view synthesis. In European Conference on Computer Vision (ECCV), 2020. 2

[44] Linfei Pan, Daniel Bar ´ ath, Marc Pollefeys, and Jo-´ hannes Lutz Schonberger. Global structure-from-motion¨ revisited. In European Conference on Computer Vision (ECCV), 2024. 6

[45] Zhexi Peng, Tianjia Shao, Yong Liu, Jingke Zhou, Yin Yang, Jingdong Wang, and Kun Zhou. RTG-SLAM: Real-time 3D reconstruction at scale using Gaussian splatting. In ACM SIGGRAPH 2024 Conference Papers, 2024. 2

[46] Guilherme Potje, Felipe Cadar, Andre Araujo, Renato Mar-´ tins, and Erickson R. Nascimento. XFeat: Accelerated features for lightweight image matching. In IEEE/CVF Confer ence on Computer Vision and Pattern Recognition (CVPR), 2024. 3, 13

[47] Lukas Radl, Michael Steiner, Mathias Parger, Alexander Weinrauch, Bernhard Kerbl, and Markus Steinberger. StopThePop: Sorted Gaussian splatting for view-consistent real-time rendering. ACM Transactions on Graphics, 43(4), 2024. 2, 3

[48] Kerui Ren, Lihan Jiang, Tao Lu, Mulin Yu, Linning Xu, Zhangkai Ni, and Bo Dai. Octree-GS: Towards consistent real-time rendering with LOD-structured 3D Gaussians. IEEE Transactions on Pattern Analysis and Machine Intelli gence, 2025. 2, 6

[49] Samuel Rota Bulo, Lorenzo Porzi, and Peter Kontschieder.\` Revising densification in Gaussian splatting. In European Conference on Computer Vision (ECCV), 2024. 3

[50] Darius Ruckert, Linus Franke, and Marc Stamminger.¨ ADOP: Approximate differentiable one-pixel point rendering. ACM Transactions on Graphics, 41(4), 2022. 2

[51] Johannes L. Schonberger and Jan-Michael Frahm. Structure-¨ from-motion revisited. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2016. 1, 6

[52] Jurgen Sturm, Nikolas Engelhard, Felix Endres, Wolfram¨ Burgard, and Daniel Cremers. A benchmark for the evaluation of RGB-D SLAM systems. In IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2012. 6

[53] Nicolas von Lutzow and Matthias Nießner. LinPrim: Lin-¨ ear primitives for differentiable volumetric rendering. In Advances in Neural Information Processing Systems (NeurIPS), 2025. 2

[54] Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, and David Novotny. VGGT: Visual geometry grounded transformer. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025. 3

[55] Lihe Yang, Bingyi Kang, Zilong Huang, Zhen Zhao, Xiaogang Xu, Jiashi Feng, and Hengshuang Zhao. Depth Anything V2. In Advances in Neural Information Processing Systems (NeurIPS), 2024. 3, 13

[56] Zehao Yu, Anpei Chen, Binbin Huang, Torsten Sattler, and Andreas Geiger. Mip-Splatting: Alias-free 3D Gaussian splatting. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024. 2

[57] Zehao Yu, Torsten Sattler, and Andreas Geiger. Gaussian opacity fields: Efficient adaptive surface reconstruction in unbounded scenes. ACM Transactions on Graphics (ToG), 43(6):1–13, 2024. 3

[58] Liyuan Zhu, Yue Li, Erik Sandstrom, Shengyu Huang, Kon-¨ rad Schindler, and Iro Armeni. LoopSplat: Loop closure by registering 3D Gaussian splats. In International Conference on 3D Vision (3DV), 2025. 2

## A. Derivation of the closed-form merge

(a) Problem formulation. Given children $\left( o _ { i } , \mu _ { i } , \Sigma _ { i } , b _ { i } \right)$ we seek one parent $( o ^ { \star } , \mu ^ { \star } , \Sigma ^ { \star } , b ^ { \star } )$ that summarizes their combined density. We use an encode–decode construction: represent the mixture by its mass, mean, covariance, and kurtosis, then invert these statistics back into the beta family. The required ingredients are: closed-form moments for one primitive (b–c), the motivation for moment matching (d), closed-form mixture moments (e–f), and the resulting merge algorithm (g), followed by validation (h).

(b) A primitive in whitened coordinates. In whitened coordinates $\mathbf { w } = \Sigma ^ { - 1 / 2 } ( \mathbf { x } - \boldsymbol { \mu } )$ , the kernel is spherical: $o ( 1 - \| \mathbf { w } \| ^ { 2 } / 9 ) ^ { \beta }$ . Its radial distribution therefore contains all required information. Since shell probability is proportional to density × the shell area, $p ( r ) \overset { \cdot } { \propto } r ^ { 2 } ( 1 - \overset { \cdot } { r } ^ { 2 } / \overset { \cdot } { 9 } ) ^ { \beta }$ on $[ 0 , 3 ]$ With $x = r ^ { 2 } / 9$ , as in Eq. (1), this becomes a Beta $( \frac { 3 } { 2 } , \beta { + } 1 )$ distribution.

Hence all required radial moments are available in closed form:

$$
\mathbb { E } [ x ] = { \frac { 1 . 5 } { \beta + 2 . 5 } } , \qquad \mathbb { E } [ x ^ { 2 } ] = { \frac { 1 5 / 4 } { ( \beta + 2 . 5 ) ( \beta + 3 . 5 ) } } ,\tag{11}
$$

with no numerical integration.

(c) Moments of one primitive. Mass. Integrating the kernel gives

$$
Z = 5 4 \pi \mathrm { B } ( { \textstyle { \frac { 3 } { 2 } } } , \beta + 1 ) \sqrt { \operatorname * { d e t } \Sigma } o .\tag{12}
$$

This is opacity × support volume $\times \textbf { a }$ β-dependent fill factor.

Covariance. Spherical symmetry gives

$$
\xi ( \beta ) = 3 \mathbb { E } [ x ] = { \frac { 4 . 5 } { \beta + 2 . 5 } } , \qquad \operatorname { C o v } = \xi ( \beta ) \Sigma .\tag{13}
$$

Thus Σ parameterizes the support, while ξ(β) converts it to statistical spread. In particular, $\xi ( 0 ) = 1 .$ 8 for the uniform radius-3 ball and $\xi ( \beta ) \to 0$ as $\beta \to \infty$

Kurtosis. The Mardia kurtosis $\boldsymbol { \kappa } ~ = ~ \mathbb { E } [ ( \mathbf { z } ^ { \top } \mathbf { z } ) ^ { 2 } ]$ , after standardization by the true covariance, is invariant to position, scale, and orientation. For this family it therefore depends only on $\beta \colon$

$$
\kappa ( \beta ) = 1 5 \frac { \beta + 2 . 5 } { \beta + 3 . 5 } .\tag{14}
$$

It ranges from $\kappa ( 0 ) = 7 5 / 7$ for the uniform ball to $\kappa ( \infty ) =$ 15 for the Gaussian limit. Since it is strictly increasing, it admits the inverse

$$
\beta = { \frac { 7 \kappa - 7 5 } { 2 \left( 1 5 - \kappa \right) } } , \qquad b = \ln ( \beta / 4 ) .\tag{15}
$$

Thus kurtosis uniquely decodes the profile shape within the family.

(d) Why moment matching. For Gaussians, the Mprojection arg $\operatorname* { m i n } _ { q } \mathrm { K L } ( p \parallel q )$ reduces to moment matching. For compact beta kernels, however, KL is unsuitable because it is infinite wherever $p > 0$ but $q = 0$ . A child extending even infinitesimally beyond the candidate parent’s support therefore dominates the divergence despite carrying negligible mass. Moment matching avoids this support pathology while retaining the Gaussian-limit interpretation.

(e) Mixture mass, mean, and covariance. Let $w _ { i } \propto Z _ { i }$ with $\textstyle \sum _ { i } w _ { i } = 1$ . The mixture mean is

$$
\mu ^ { \star } = \sum _ { i } w _ { i } \mu _ { i } .
$$

By the law of total covariance,

$$
\Sigma _ { \mathrm { m i x } } = \sum _ { i } w _ { i } \big ( \xi ( \beta _ { i } ) \Sigma _ { i } + \delta _ { i } \delta _ { i } ^ { \top } \big ) , \qquad \delta _ { i } = \mu _ { i } - \mu ^ { \star } .\tag{16}
$$

Crucially, each child contributes its true covariance $\xi ( \beta _ { i } ) \Sigma _ { i }$ , not its support matrix $\Sigma _ { i }$

(f) Mixture kurtosis. The mixture kurtosis is evaluated with respect to the mixture covariance $\Sigma _ { \mathrm { m i x } } .$

$$
\kappa _ { \mathrm { m i x } } = \sum _ { i } w _ { i } \mathbb { E } _ { i } [ ( \mathbf { z } ^ { \top } \mathbf { z } ) ^ { 2 } ] ,
$$

where z $\bar { \mathbf { \Omega } } ^ { \top } \mathbf { z } = ( \mathbf { x } - { \boldsymbol { \mu } } ^ { \star } ) ^ { \top } { \boldsymbol { \Sigma } } _ { \operatorname* { m i x } } ^ { - 1 } ( \mathbf { x } - { \boldsymbol { \mu } } ^ { \star } )$

Write a sample from child i $\mathrm { a s } \ \mathbf { x } = \mu _ { i } + \Sigma _ { i } ^ { 1 / 2 } \mathbf { s } ,$ where s is isotropic with radial moments $m _ { 2 } = 9 \mathbb { E } _ { i } [ x ]$ and $m _ { 4 } =$ $8 1 \mathbb { E } _ { i } [ x ^ { 2 } ]$ . Define

$$
a _ { i } = \delta _ { i } ^ { \top } \Sigma _ { \operatorname* { m i x } } ^ { - 1 } \delta _ { i } ,\tag{17}
$$

$$
v _ { i } = \Sigma _ { i } ^ { 1 / 2 } \Sigma _ { \operatorname* { m i x } } ^ { - 1 } \delta _ { i } ,\tag{18}
$$

$$
M _ { i } = \Sigma _ { i } ^ { 1 / 2 } \Sigma _ { \mathrm { m i x } } ^ { - 1 } \Sigma _ { i } ^ { 1 / 2 } ,\tag{19}
$$

so that

$$
\begin{array} { r } { \mathbf { z } ^ { \top } \mathbf { z } = a _ { i } + 2 v _ { i } ^ { \top } \mathbf { s } + \mathbf { s } ^ { \top } M _ { i } \mathbf { s } . } \end{array}\tag{20}
$$

Odd terms vanish by symmetry. Isotropy gives $\mathbb { E } [ { \mathbf { s } } { \mathbf { s } } ^ { \top } ] =$ $\frac { m _ { 2 } } { 3 } I$ and

$$
\begin{array} { r } { \mathbb E \left[ ( \mathbf s ^ { \top } M \mathbf s ) ^ { 2 } \right] = \frac { m _ { 4 } } { 1 5 } \big ( ( \mathrm { t r } M ) ^ { 2 } + 2 \mathrm { t r } M ^ { 2 } \big ) . } \end{array}\tag{21}
$$

Therefore

$$
\begin{array} { r l r } & { } & { { \mathbb E } _ { i } \big [ ( { \mathbf { z } ^ { \top } } { \mathbf { z } } ) ^ { 2 } \big ] = a _ { i } ^ { 2 } + \frac { 4 m _ { 2 } } { 3 } \| v _ { i } \| ^ { 2 } } \\ & { } & { ~ + \frac { m _ { 4 } } { 1 5 } \big ( ( \mathrm { t r } M _ { i } ) ^ { 2 } + 2 \mathrm { t r } M _ { i } ^ { 2 } \big ) } \\ & { } & { ~ + \frac { 2 a _ { i } m _ { 2 } } { 3 } \mathrm { t r } M _ { i } . } \end{array}\tag{22}
$$

Thus each contribution requires only traces of products of 3×3 matrices and is readily batched on the GPU.

Table 6. Per-stage runtime, in milliseconds (CUDA-synchronized timers, so stage costs are exact but totals are overestimated). Image load and feature extraction run once per streamed frame; all other stages run per registered keyframe (loop closure rarely $1 - 3 \times )$ .
<table><tr><td>Stage (ms/call)</td><td>GARDEN</td><td>FR1/DESK</td><td>TRUCK</td><td>UNIVERSITY2</td><td>SMALLCITY</td></tr><tr><td>Image load, features</td><td>13.4</td><td>11.8</td><td>13.4</td><td>11.4</td><td>13.2</td></tr><tr><td>Pose estimation</td><td>119.2</td><td>149.2</td><td>117.6</td><td>114.6</td><td>119.7</td></tr><tr><td>Loop closure</td><td>101.3</td><td></td><td>164.1</td><td></td><td></td></tr><tr><td>Primitive spawning</td><td>47.1</td><td>49.3</td><td>46.0</td><td>44.6</td><td>36.4</td></tr><tr><td>Optimization burst</td><td>170.6</td><td>146.5</td><td>129.2</td><td>174.9</td><td>329.6</td></tr><tr><td>Hierarchy update</td><td>2.7</td><td>2.7</td><td>3.0</td><td>41.4</td><td>56.5</td></tr><tr><td>Registered keyframes</td><td>180</td><td>159</td><td>228</td><td>347</td><td>369</td></tr><tr><td>Primitives (final)</td><td>2.24M</td><td>0.33M</td><td>1.23M</td><td>2.44M</td><td>1.32M</td></tr><tr><td>Total time (m:ss)</td><td>1:02</td><td>1:05</td><td>1:09</td><td>2:07</td><td>3:47</td></tr></table>

Out-of-band mixtures. A mixture of beta kernels can fall outside the family’s kurtosis band. This happens on both sides: well-separated children give lighter tails than any beta kernel, while heavy-tailed groups exceed 15. Since κ is scale-invariant, shrinking a group to sub-pixel size does not change this. We restrict the parent to $b ^ { \star } \in \left[ - 4 , 4 \right]$ and clip $\kappa _ { \mathrm { m i x } }$ to κˆ in the corresponding band $[ \kappa ( 4 e ^ { - 4 } ) , \kappa ( 4 e ^ { 4 } ) ]$ . Because κ is monotone, this selects the closest representable profile and keeps $\beta ^ { \star }$ finite. Parent opacity is capped at the rasterizer’s limit of 0.99. Clipping is frequent on large scenes, yet the merge still clearly outperforms the opacityweighted baseline (Tab. 4).

(g) End-to-end merge. The merge proceeds as follows:

1. Compute $Z _ { i }$ from (12) and normalize to $w _ { i }$

2. Compute $\begin{array} { r } { \mu ^ { \star } = \sum _ { i } w _ { i } \mu _ { i } } \end{array}$

3. Compute $\Sigma _ { \mathrm { m i x } }$ from (16).

4. Evaluate (22) and accumulate $\kappa _ { \mathrm { m i x } } .$

5. Clip $\kappa _ { \mathrm { m i x } }$ to the band of $b ^ { \star } \in [ - 4 , 4 ]$

6. Decode $\beta ^ { \star }$ and $b ^ { \star }$ using (15).

7. Recover $\Sigma ^ { \star } = \Sigma _ { \mathrm { m i x } } / \xi ( \beta ^ { \star } )$ and conserve mass through $\begin{array} { r } { O ^ { \star } = \sum _ { i } Z _ { i } / Z ( \beta ^ { \star } , \Sigma ^ { \star } ) } \end{array}$ , capped at 0.99.

The procedure requires no sampling or iterative fitting. Its only nontrivial linear algebra consists of symmetric 3×3 square roots, inverses, and the final decomposition of $\Sigma ^ { \star }$ all handled by batched $3 { \times } 3$ eigendecompositions.

(h) Stress tests. Coincident children. For N identical children, $\delta _ { i } = 0$ and the mixture moments reduce to those of the child itself. Consequently the decoder recovers $\beta ^ { \star } = \beta _ { \star }$ $\Sigma ^ { \star } = \Sigma$ , and $o ^ { \star } = N \cdot o ,$ , which can exceed 1 and is then capped.

Out-of-band groups. The kurtosis range also acts as a validity certificate. For four near-point children at the vertices of a regular tetrahedron, $\Sigma _ { i } \  \ 0$ leaves only the centeroffset terms. Symmetry gives $\begin{array} { r } { \Sigma _ { \mathrm { m i x } } = \frac { R ^ { 2 } } { 3 } I } \end{array}$ and $a _ { i } \ = \ 3 .$ yielding $\kappa _ { \mathrm { m i x } } ~ = ~ 9 ~ < ~ 7 5 / 7$ . Since this lies outside the beta family’s kurtosis range, the merge is clipped to the flattest admissible profile. At the opposite boundary, mixtures with sufficiently heavy tails satisfy $\kappa _ { \mathrm { m i x } } \geq 1 5$ and clamp to $b ^ { \star } = 4$

Numerical validation. Across 200 randomized fourchild merges, Monte-Carlo estimates using 1.5M samples per merge agree with the closed forms to 0.19% in kurtosis and 0.13% in covariance.

## B. Implementation Details & Startup

The system is implemented in PyTorch 2.11/CUDA 12.8, with performance-critical stages implemented as custom GPU kernels. Feature extraction uses XFeat [46], matching combines mutual nearest neighbors with LighterGlue [34], retrieval uses MixVPR [1], and monocular depth priors use Depth Anything V2 [55]. The pose-estimation and mapping stages follow the backbone of i3DGS [42], including P3P/RANSAC, triangulation, and local bundle adjustment.

Rasterization uses our beta-kernel implementation of a FasterGS-style fused rasterizer [13]. Forward quantities are recomputed during the backward pass, and the sparse Adam update and analytic regularizer gradients are applied directly to the optimizer state without materializing primitive gradients. Spherical-beta colour parameters are stored compactly for the fused implementation, while exported models retain the standard SH-width layout for compatibility.

Each keyframe performs one 30-iteration optimization burst. The bootstrap uses 8 keyframes, local BA operates on the 5 most similar keyframes, and the LoD hierarchy is rebuilt every 10 keyframes. Inactive fine levels and keyframes beyond the active window are offloaded to host memory. All reported timings include the online pose estimation, spawning, optimization, and hierarchy stages.

## C. Runtime and Latency

Per-keyframe runtime. Tab. 6 reports the runtime of the main online stages. Optimization is the dominant cost (129– 330 ms per keyframe), followed by pose estimation and primitive spawning. Hierarchy updates remain small on the bounded scenes and become more significant for UNIVER-SITY2 and SMALLCITY. Frames that are not registered as keyframes incur only image processing and matching, resulting in lower average latency on TUM and StaticHikes. Using the fused rasterizer reduces the optimization cost on GARDEN from 287 ms to 170 ms per keyframe and total runtime from 84 s to 62 s.

Table 7. Time to first image. Time from the first streamed frame to the first renderable reconstruction, including bootstrap, mini-BA, and the first optimization burst. “Frames used” denotes the number of streamed frames consumed by bootstrap.
<table><tr><td>Scene</td><td>1st image (s)</td><td>Frames used</td><td>KFs</td><td>Boot. Mean latency/ frame (ms)</td></tr><tr><td>GARDEN</td><td>1.2</td><td>8</td><td>8</td><td>336</td></tr><tr><td>FR1/DESK</td><td>1.5</td><td>17</td><td>8</td><td>106</td></tr><tr><td>TRUCK</td><td>1.0</td><td>10</td><td>8</td><td>276</td></tr><tr><td>UNIVERSITY2</td><td>1.2</td><td>9</td><td>8</td><td>176</td></tr><tr><td>SMALLCITY</td><td>1.4</td><td>9</td><td>8</td><td>237</td></tr></table>

Table 8. Fused rasterization step profile on a converged GAR-DEN reconstruction (2.2M primitives, 1297×840). The fused backward integrates the sparse Adam update in-kernel, so the perprimitive stages carry the optimizer’s memory traffic: running the preprocess backward with plain gradient stores instead of the fused Adam update halves it (0.66 vs. 1.31 ms) — i.e. half that kernel is the optimizer. End-to-end, the fused path reduces the 30-iteration burst from 287 to 170 ms per keyframe (Tab. 6).
<table><tr><td>Stage</td><td>ms</td><td>share</td></tr><tr><td>blend backward</td><td>1.67</td><td>37%</td></tr><tr><td>preprocess backward (fused Adam)</td><td>1.31</td><td>29%</td></tr><tr><td>blend forward</td><td>0.85</td><td>19%</td></tr><tr><td>preprocess forward</td><td>0.36</td><td>8%</td></tr><tr><td>sort / instancing</td><td>0.37</td><td>8%</td></tr><tr><td>preprocess backward (no fused Adam)</td><td>0.66</td><td></td></tr></table>

Time to first image. Tab. 7 measures the time from the first streamed frame to the first renderable reconstruction. Across the evaluated scenes, the first image is available within 1.0–1.5 s, after an 8-keyframe bootstrap, mini-BA, and first 30-iteration optimization burst.

Fused optimization. Tab. 8 profiles one fused optimization iteration. Integrating the sparse Adam update removes the separate gradient write and optimizer read, substantially reducing the optimization cost. On GARDEN, this reduces the optimization burst from 287 to 170 ms per keyframe (Tab. 6).

![](images/2d4759b9044eb48df76925d46b1b125ea37be0f50c98f636e7e5f4107bad6da0.jpg)  
Figure 8. Half-maximum scale compensation. Without compensation, lowering β broadens the kernel; with the proposed compensation, all profiles share the same half-maximum radius, isolating shape changes from footprint changes.

## D. Additional Results

Reduced-Resolution Protocols. Tab. 9 evaluates Mip-NeRF360 (otf), TUM, and StaticHikes at reduced resolution. We remain strongest on the image metrics for TUM and MipNeRF360, although the reduced-detail setting negates the speed advantage and degrades performance on StaticHikes scenes, where there is less sharp detail to explain. We also surpass ARTDECO, which becomes substantially more expensive at higher resolution (StaticHikes and MipNeRF360).

Vision Transformer Protocol. In Tab. 10, we evaluate AnySplat [23] and DepthAnything3 [33] (Giant-Large-1.1, Gaussian head) under the 448×448 protocol. Despite their feed-forward reconstruction, our method has increased reconstruction quality by at least 5 dB, including over the post-trained AnySplat variant.

## E. Colour Basis on DeepBlending

On DeepBlending, degree-3 SH outperforms the sphericalbeta (SB) lobes used in our default configuration. Tab. 11 shows that SH improves PSNR by 0.2 − 0.3 dB. The per-scene comparison shows that this behavior is largely specific to this dataset; with OFFICE, COUNTER, and TUM/XYZ showing strong improvements with SB (Tab. 12).

## F. Kernel Shape vs. Footprint

Fig. 8 verifies the scale compensation used whenever β changes. Without it, a β = 1 kernel is about 1.8× wider at half maximum than the β = 4 reference, also increasing its footprint area by roughly 3×. The proposed compensation preserves the half-maximum radius, allowing spawning, solidification, and merging to change solidity without changing apparent size.

Table 9. Low-resolution protocol (TUM resized to 448×336, MipNeRF360 and StaticHikes to 640 px width). <sup>†</sup> from i3DGS [42], with times roughly scaled to our setup. Bold: best.
<table><tr><td></td><td colspan="3">TUM PSNR↑SSIM↑LPIPS↓ Time↓</td><td colspan="3">MIPNERF360 (otf) PSNR↑SSIM↑LPIPS↓ Time↓</td><td colspan="3">STATICHIKES PSNR↑SSIM↑LPIPS↓ Time↓</td><td></td></tr><tr><td>DROID-Splat†</td><td>19.49 0.721</td><td>0.325</td><td>~5min</td><td>25.87</td><td>0.776</td><td>0.258</td><td>~9min 19.65</td><td>0.470</td><td>0.506</td><td>~8min</td></tr><tr><td>CF-3DGS†</td><td>15.05 0.578</td><td>0.405</td><td>~1h</td><td>13.52</td><td>0.295</td><td>0.621 ~1h</td><td>15.21</td><td>0.301</td><td>0.560</td><td>~7h</td></tr><tr><td>OtF NVS†</td><td>22.45 0.815</td><td>0.225</td><td>~50s</td><td>25.80</td><td>0.834</td><td>0.182</td><td>~40s 21.93</td><td>0.673</td><td>0.270</td><td>~1min</td></tr><tr><td>ARTDECO</td><td>23.03 0.662</td><td>0.343</td><td>04:56</td><td>22.97</td><td>0.841</td><td>0.210 14:07</td><td>21.28</td><td>0.555</td><td>0.380</td><td>13:05</td></tr><tr><td>i3DGS</td><td>22.51 0.836</td><td>0.189</td><td>00:58</td><td>27.16</td><td>0.849</td><td>0.165 00:43</td><td>22.24</td><td>0.683</td><td>0.297</td><td>01:04</td></tr><tr><td>Ours</td><td>25.14 0.872</td><td>0.148</td><td>01:26</td><td>27.73</td><td>0.867</td><td>0.144</td><td>00:49</td><td>22.17 0.679</td><td>0.309</td><td>01:12</td></tr></table>

Table 10. Tiny resolution (448×448) input. <sup>†</sup> from i3DGS [42]. R indicates refinement following [23]. Bold: best.
<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>MIPNERF360 (otf)PSNR↑SSIM↑LPIPS↓ Time↓</td></tr><tr><td rowspan=3 colspan=1>AnySplat†AnySplatR†i3DGS</td><td rowspan=2 colspan=1>18.050.4300.390 ~20s21.53</td></tr><tr><td rowspan=1 colspan=1>53 0.600 0.300 ~2min</td></tr><tr><td rowspan=1 colspan=1>26.510.8450.173  0:38</td></tr><tr><td rowspan=1 colspan=1>DA3-Gauss</td><td rowspan=1 colspan=1>19.880.4870.338  0:17</td></tr><tr><td rowspan=1 colspan=1>Ours</td><td rowspan=1 colspan=1>26.560.8420.166  0:46</td></tr></table>

Table 11. Colour basis on DeepBlending. Comparison of the default five zero-initialized spherical-beta lobes with degree-3 SH using the identical reconstruction pipeline.
<table><tr><td>Scene</td><td>Colour</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td></tr><tr><td>drjohnson</td><td>SB lobes</td><td>25.67</td><td>0.828</td><td>0.315</td></tr><tr><td></td><td>SH</td><td>25.88</td><td>0.832</td><td>0.309</td></tr><tr><td>playroom</td><td>SB lobes</td><td>27.28</td><td>0.865</td><td>0.276</td></tr><tr><td></td><td>SH</td><td>27.58</td><td>0.868</td><td>0.268</td></tr></table>

Table 12. Spherical-beta lobes vs. SH colour, per scene. ∆ denotes lobes minus SH for PSNR and LPIPS. Both variants use the same reconstruction configuration except color representation.
<table><tr><td>Dataset</td><td>Scene</td><td>∆PSNR</td><td>∆SSIM</td><td>∆LPIPS</td></tr><tr><td>db db</td><td>drjohnson playroom</td><td>-0.21 -0.29</td><td>-0.004 -0.003</td><td>+0.007 +0.008</td></tr><tr><td>MipNeRF360</td><td>garden</td><td>+0.18</td><td>+0.005</td><td>-0.000</td></tr><tr><td>MipNeRF360</td><td>bonsai</td><td>+0.23</td><td>+0.003</td><td>-0.002</td></tr><tr><td>MipNeRF360</td><td>counter</td><td>+0.41</td><td>+0.007</td><td>-0.006</td></tr><tr><td>MipNeRF360</td><td>bicycle</td><td>+0.16</td><td>+0.013</td><td>-0.001</td></tr><tr><td>MipNeRF360</td><td>kitchen</td><td>+0.25</td><td>+0.007</td><td>-0.004</td></tr><tr><td>MipNeRF360</td><td>room</td><td>-0.03</td><td>-0.004</td><td>+0.002</td></tr><tr><td>MipNeRF360</td><td>stump</td><td>-0.16</td><td>-0.009</td><td>+0.014</td></tr><tr><td>MipNeRF360</td><td>flowers</td><td>+0.03</td><td>+0.005</td><td>+0.005</td></tr><tr><td>MipNeRF360</td><td>treehill</td><td>-0.08</td><td>-0.001</td><td>+0.002</td></tr><tr><td>TUM</td><td>TUM desk</td><td>+0.19</td><td>+0.004</td><td>-0.003</td></tr><tr><td>TUM</td><td>TUM xyz</td><td>+0.27</td><td>+0.002</td><td>+0.001</td></tr><tr><td>TUM</td><td>TUM office</td><td>+0.62</td><td>+0.014</td><td>-0.016</td></tr><tr><td>T&amp;T</td><td></td><td></td><td></td><td></td></tr><tr><td>T&amp;T</td><td>train truck</td><td>+0.04 +0.16</td><td>+0.002 +0.005</td><td>+0.001 -0.001</td></tr><tr><td>StaticHikes</td><td></td><td></td><td></td><td></td></tr><tr><td>StaticHikes</td><td>forest1</td><td>+0.02</td><td>+0.003</td><td>+0.001</td></tr><tr><td></td><td>forest2</td><td>-0.03</td><td>+0.001</td><td>+0.003</td></tr><tr><td>StaticHikes</td><td>university2</td><td>-0.08</td><td>-0.002</td><td>+0.003</td></tr></table>

## G. Pose Accuracy on TUM

Tab. 13 compares our pose estimation with i3DGS on the three TUM sequences using the same trajectories and COLMAP pseudo-ground truth. On FR2/XYZ and FR3/OFFICE, our trajectories are slightly more accurate than those of i3DGS on all four metrics. On FR1/DESK, i3DGS remains marginally ahead (ATE 0.55 vs. 0.47%, rotation APE 1.90<sup>◦</sup> vs. 1.42<sup>◦</sup>). This sequence is the hardest of the three for our bootstrap: it is strongly motion-blurred, and the first eight frames, which we use to reach the first image quickly, provide little parallax.

Table 13. TUM pose accuracy. Our method and i3DGS evaluated against the same COLMAP pseudo-ground truth using aligned trajectories. Translation errors are reported as a percentage of trajectory extent and rotation errors in degrees; results are averaged over multiple runs.
<table><tr><td>Scene</td><td colspan="4">Method ATE(%)↓ R.APE(°)↓ T.RPE(%)↓ R.RPE(°)↓</td></tr><tr><td>fr1/desk</td><td>Ours</td><td>0.55</td><td>1.90</td><td>0.344</td><td>0.644</td></tr><tr><td></td><td>i3DGS</td><td>0.47</td><td>1.42</td><td>0.314</td><td>0.548</td></tr><tr><td rowspan="2">fr2/xyz</td><td>Ours</td><td>0.33</td><td>0.41</td><td>0.178</td><td>0.128</td></tr><tr><td>i3DGS</td><td>0.37</td><td>0.42</td><td>0.185</td><td>0.133</td></tr><tr><td>fr3/office</td><td>Ours</td><td>0.32</td><td>0.94</td><td>0.127</td><td>0.326</td></tr><tr><td></td><td>i3DGS</td><td>0.38</td><td>1.02</td><td>0.150</td><td>0.347</td></tr></table>

Table 14. Conditioning the FR1/DESK bootstrap. Three bootstrap variants under the same scoring protocol. The spread selector improves pose accuracy and PSNR at the cost of 1.1 s additional latency; the parallax floor degrades accuracy.
<table><tr><td>Bootstrap</td><td colspan="4">ATE (%)↓ R. APE (°)↓ PSNR↑ f.i. (s) f</td></tr><tr><td>ordered first-8</td><td>0.55</td><td>1.90</td><td>23.93 1.5</td><td>519</td></tr><tr><td>ordered + parallax</td><td>1.78</td><td>3.17</td><td>23.67 1.5</td><td>527</td></tr><tr><td>spread, pool 30</td><td>0.51</td><td>1.21</td><td>24.20</td><td>2.6511</td></tr></table>

Tab. 14 shows how sensitive FR1/DESK is to the choice of bootstrap frames. All three variants converge to a focal length close to the calibrated value (511–527 vs. ≈517), so the differences come from the trajectory rather than the intrinsics. Enforcing a parallax floor on the ordered bootstrap is counterproductive (ATE 1.78%). Selecting a spreadmaximizing subset from the first 30 frames reduces ATE to 0.51% and rotation APE to 1.21<sup>◦</sup>, on par with or better than i3DGS, and also improves PSNR by 0.27 dB. Its only cost is 1.1 s of additional time to first image. Our default ordered bootstrap therefore trades a small amount of pose accuracy on FR1/DESK for lower latency; the underlying pose pipeline is on par with i3DGS, and a wider bootstrap pool closes the remaining gap when the extra second of latency is acceptable.

end of online pass (keyframe 40)  
![](images/4dd09e05e54ce65ab326bb37c89b5edcbe07f295c42a2d14c85ab6d15296a1f8.jpg)  
Figure 9. Online confidence field on a walking capture (UNIVERSITY2), analogous to Fig. 4: renders over their perprimitive shape field (β, warm = soft). As the walk advances, accumulated multi-view evidence solidifies the traversed corridor of the scene behind the camera, while freshly entered and peripheral regions remain soft probes — the field marks the reconstruction frontier of an unbounded capture live, at no quality cost.

## H. Confidence

## H.1. Solidity Controller Sweep and Sensitivity

Tab. 19 evaluates the controller step ∆b on GARDEN. Performance is stable for steps from 0.1 to 0.8. Although the first two slightly degrade quality, the optimization is stable with regards to this hyperparameter. The number of solidification events remains similar across settings, indicating that the evidence gates primarily determine which primitives are solidified, while ∆b controls the update rate. We therefore use ∆b = 0.4 as a stable operating point. The resulting shape field provides the confidence visualization shown in Fig. 4; Fig. 9 shows the same behavior on UNIVERSITY2.

## H.2. Spawn Confidence

Fig. 10 visualizes the two spawn-confidence signals on UNIVERSITY2. Geometric observability suppresses poorly constrained far-field depth, while MVS–monocular agreement rejects ambiguous, occlusion-dominated regions. Their product therefore assigns high confidence to nearby, well-verified surfaces and low confidence to uncertain geometry.

The correlation is also quantitative. Among 12.7k MVS-supported GARDEN spawn candidates with held-out COLMAP depth $z _ { \mathrm { G T } }$ , we label a candidate depth-accurate when $| \log ( z / z _ { \mathrm { G T } } ) | < 0 . 0 5$ (5% relative error). The MVS confidence ranks an accurate candidate above an inaccurate one with AUC 0.92, whereas $c _ { z }$ is largely uninformative on GARDEN because the baseline saturates its observability score. On UNIVERSITY2, where far-field content is present, $c _ { z }$ provides the complementary geometric signal.

Table 15. Relocation design choices on FR2/XYZ. Each row removes one component of the error-driven relocation.
<table><tr><td>Variant</td><td></td><td>PSNR↑ SSIM↑ LPIPS↓</td><td></td></tr><tr><td>Full (error-weighted + consistency gate)</td><td>28.45</td><td>0.893</td><td>0.133</td></tr><tr><td>w/o error weighting (opacity-sampled)</td><td>27.89</td><td>0.885</td><td>0.148</td></tr><tr><td>w/o relocation</td><td>28.21</td><td>0.890</td><td>0.136</td></tr></table>

## I. Relocation Details

Fig. 11 illustrates one relocation step on GARDEN. Neardead primitives are recycled onto persistent high-error regions, keeping the total primitive count unchanged while redistributing capacity.

Tab. 15 shows error-based destination sampling of Sec. 4.4 on FR2/XYZ. Both PSNR and LPIPS favor the full method (28.45 dB and 0.133). Replacing error weighting with the opacity-based sampling of 3DGS-MCMC [28] is worse than disabling relocation (27.89 dB vs. 28.21 dB and 0.148 vs. 0.136), indicating that opacity sampling primarily reinforces already strong regions.

## J. Constants

Tab. 18 summarizes the constants used by the method and their provenance. Several quantities are derived directly from the kernel or image resolution, while the loss weights are inherited from prior work. The remaining thresholds are chosen from simple heuristics and evaluated with one-at-atime sensitivity tests in Tab. 20 and Tab. 19. Across these probes, quality varies by at most 0.17 dB, indicating low sensitivity to the tested operating points.

## K. Full 3DGS Benchmark Table

See Tab. 16 for the full benchmark table including GLOMAP with exhaustive mapping, discussed in Sec. 6.1. Sequential mapping fails to register 2 test frames on bicycle and flowers, and 7 on stump. In Tab. 2, they are excluded from the average, but they are included in the full table here for completeness as black frames.

## L. Memory Footprint Over Time

The online pass remains well within the 32 GB GPU budget: peak memory is 8.6 GB on GARDEN (2.2M primitives),

![](images/5d8c43482b1e06ba461f3a7d5587686d780d97a0d7ad7898a632a93ceef0ae8e.jpg)  
Figure 10. Spawn-confidence signals on UNIVERSITY2. Primitive locations are colored by geometric observability $c _ { z }$ , MVS–monocular agreement $c _ { \mathrm { m v s } } ,$ and their product c. The two signals fail in different regions: distant trees have low observability, while dense foliage has poor depth agreement despite sufficient baseline. Their product therefore assigns solid, opaque initialization to well-constrained surfaces and soft initialization to ambiguous regions.

Table 16. Results on the full 3DGS benchmark. Nine MipNeRF360, two Tanks&Temples, and two Deep Blending scenes. We evaluate BetaSplatting and FasterGS with GLOMAP poses using sequential and exhaustive matching, and compare against i3DGS and our method GLOMAP sequential requires 0:01:52 on MipNeRF360, 0:01:42 on Tanks&Temples and fails on Deep Blending. GLOMAP exhaustive takes respectively 0:04:53, 0:04:22 and 0:03:42. The best and second best are color coded for non-exhaustive matching methods.
<table><tr><td rowspan="2" colspan="2"></td><td colspan="4">MIPNERF360</td><td colspan="4">TANKS AND TEMPLES</td><td colspan="4">DEEP BLENDING</td></tr><tr><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>Time↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>Time↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>Time↓</td></tr><tr><td></td><td>BetaSplatting (30K)</td><td>28.09</td><td>0.822</td><td>0.222</td><td>0:12:34</td><td>22.18</td><td>0.818</td><td>0.209</td><td>0:11:14</td><td>30.49</td><td>0.915</td><td>0.250</td><td>0:10:11</td></tr><tr><td></td><td>FasterGS (30K)</td><td>27.66</td><td>0.818</td><td>0.251</td><td>0:09:01</td><td>20.45</td><td>0.781</td><td>0.274</td><td>0:07:03</td><td>29.78</td><td>0.908</td><td>0.306</td><td>0:07:14</td></tr><tr><td>G(xh,.)</td><td>BetaSplatting (7K)</td><td>25.92</td><td>0.750</td><td>0.308</td><td>0:06:21</td><td>19.72</td><td>0.710</td><td>0.347</td><td>0:05:36</td><td>26.85</td><td>0.861</td><td>0.350</td><td>0:04:56</td></tr><tr><td></td><td>FasterGS (7K)</td><td>25.36</td><td>0.740</td><td>0.347</td><td>0:05:34</td><td>19.07</td><td>0.712</td><td>0.373</td><td>0:04:48</td><td>28.91</td><td>0.891</td><td>0.358</td><td>0:04:17</td></tr><tr><td></td><td>BetaSplatting (30K)</td><td>25.34</td><td>0.679</td><td>0.341</td><td>0:09:07</td><td>23.31</td><td>0.831</td><td>0.190</td><td>0:08:27</td><td>×</td><td>X</td><td>X</td><td>×</td></tr><tr><td></td><td>FasterGS (30K)</td><td>24.96</td><td>0.679</td><td>0.362</td><td>0:05:29</td><td>22.17</td><td>0.802</td><td>0.250</td><td>0:04:21</td><td>X</td><td>X</td><td>X</td><td>X</td></tr><tr><td>Gee) )</td><td>BetaSplatting (7K)</td><td>23.62</td><td>0.636</td><td>0.402</td><td>0:03:17</td><td>20.81</td><td>0.733</td><td>0.321</td><td>0:02:55</td><td>X</td><td>X</td><td>X</td><td></td></tr><tr><td></td><td>FasterGS (7K)</td><td>23.96</td><td>0.654</td><td>0.413</td><td>0:02:29</td><td>20.79</td><td>0.742</td><td>0.343</td><td>0:02:10</td><td>X</td><td>X</td><td>X</td><td>× ×</td></tr><tr><td></td><td>i3DGS</td><td>24.75</td><td>0.721</td><td>0.280</td><td>0:01:59</td><td>22.08</td><td>0.783</td><td>0.226</td><td>0:01:25</td><td>27.16</td><td>0.856</td><td>0.285</td><td>0:02:03</td></tr><tr><td></td><td>Ours</td><td>25.19</td><td>0.731</td><td>0.264</td><td>0:01:17</td><td>23.00</td><td>0.803</td><td>0.200</td><td>0:01:18</td><td>26.49</td><td>0.847</td><td>0.295</td><td>0:01:22</td></tr></table>

Table 17. Unordered input. Identical configuration.
<table><tr><td>Scene</td><td>Order</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td></tr><tr><td>GARDEN</td><td>ordered</td><td>25.82</td><td>0.786</td><td>0.187</td></tr><tr><td></td><td>shuffled</td><td>25.34</td><td>0.774</td><td>0.191</td></tr><tr><td>COUNTER</td><td>ordered</td><td>26.98</td><td>0.860</td><td>0.222</td></tr><tr><td></td><td>shuffled</td><td>27.54</td><td>0.877</td><td>0.204</td></tr><tr><td>BONSAI</td><td>ordered</td><td>29.61</td><td>0.909</td><td>0.204</td></tr><tr><td></td><td>shuffled</td><td>27.11</td><td>0.850</td><td>0.261</td></tr><tr><td>FR1/DESK</td><td>ordered</td><td>23.93</td><td>0.844</td><td>0.209</td></tr><tr><td></td><td>shuffled</td><td>23.09</td><td>0.870</td><td>0.179</td></tr></table>

9.3 GB on FR1/DESK, 6.1 GB on TRUCK, 9.6 GB on UNI-VERSITY2 (2.4M), and 10.3 GB on SMALLCITY. Fig. 12 shows that memory saturates while reconstruction continues. On GARDEN, the active count reaches about 2.2M and changes by only 12% over the second half of the capture, while 1.45M relocations recycle existing capacity. On SMALLCITY, the model grows from 0.85M to 1.32M active primitives while peak GPU memory remains near 10 GB, with inactive hierarchy levels stored in host memory.

## M. Unordered input

Our method is designed for unordered image streams, so we evaluate the identical configuration on fixed random permutations of the input images (Tab. 17). On GAR-

render at keyframe arrival (22.6 dB)  
![](images/0d3f4a65e7f3924d39485e22072541d57635d48ab24b44e0458e3a8e2d3a1c99.jpg)

error + relocation targets (9980 moved)  
![](images/e92831fc4b751bfc6ae2cef48a2bd683a044f83b3e022726f23db83d010eb268.jpg)

![](images/220e2555ea73424eb80c2fdccd183b37e0daf785e74aa21a91592663c5f1f8dd.jpg)  
Figure 11. Error-driven relocation on GARDEN. Left: incoming keyframe and per-pixel error at 22.6 dB. Middle: dead primitives relocated to high-error contributors, with render-preserving opacity and scale correction. Right: the subsequent 30-iteration burst increases quality to 24.5 dB.

Table 18. Method constants and their provenance. Values are grouped by spawning, solidity control, relocation, optimization/rasterization, and the hierarchy. Derived quantities follow from other parameters, structural values are fixed by the kernel construction, and inherited values are taken unchanged from prior work.
<table><tr><td>Constant</td><td>Value</td><td>Role</td><td>Provenance</td></tr><tr><td colspan="4">Spawning (Sec. 4.2)</td></tr><tr><td> $\theta _ { 0 }$ </td><td>2°</td><td rowspan="4">full-observability angle spawn opacity, unverified</td><td rowspan="4">probed  $( \times / \div 2 )$  fixed heuristic probed (0.2/0.5)</td></tr><tr><td> $o _ { 0 }$ </td><td>0.1</td></tr><tr><td>01</td><td>0.3</td></tr><tr><td> ${ \cal O } _ { \mathrm { H o o r } }$ </td><td>0.02</td></tr><tr><td colspan="2"> $\beta _ { \mathrm { s o l i d , s o f t } }$  1,5</td><td rowspan="4">shape band around init β=4 coverage floor of sampling base sampling scale</td><td rowspan="4">structural derived inherited [41]</td></tr><tr><td> $p _ { \mathrm { m i n } }$ </td><td> $1 / ( 0 . 5 s _ { \mathrm { m a x } } ) ^ { 2 }$ </td></tr><tr><td>φ</td><td>2</td></tr><tr><td> $r _ { \mathrm { m v s } }$ </td><td>0.2</td></tr><tr><td colspan="4">Solidity controller (Sec. 4.3)</td></tr><tr><td>∆b</td><td>0.4</td><td rowspan="4">per-keyframe shape step opacity gate for hardening min. owned-pixel support</td><td rowspan="4">swept (0.1–0.8) probed (0.5/0.9)</td></tr><tr><td>Osolid</td><td>0.7</td></tr><tr><td></td><td>20 px, 3 views</td></tr><tr><td>esolid</td><td>0.03 0.5</td></tr><tr><td colspan="4"></td></tr><tr><td>Relocation (Sec. 4.4)</td><td></td><td></td><td></td></tr><tr><td colspan="4">Odead 0.10</td></tr><tr><td></td><td>10 keyframes</td><td rowspan="2">spawn-protection window</td><td rowspan="2">fixed heuristic fixed heuristic</td></tr><tr><td></td><td>0.5× mean scale positional jitter of relocations</td></tr><tr><td colspan="4">Optimization / rasterization (Secs. 4.5 and 4.6)</td></tr><tr><td>K</td><td>5</td><td>spherical-beta colour lobes</td><td rowspan="4">fixed heuristic inherited [41]</td></tr><tr><td> $\lambda$ </td><td> $_ { 0 . 1 5 }$ </td><td>SSIM weight</td></tr><tr><td> $\lambda _ { \mathcal { R } }$ </td><td> $1 0 ^ { - 2 }$ </td><td>regularizer weight</td></tr><tr><td></td><td> $1 / 2 5 5$ </td><td>inherited [28, 35] rasterizer contribution cutoff convention [26]</td></tr><tr><td colspan="4">5% of diag.  $s _ { \mathrm { m a x } }$ </td></tr><tr><td></td><td>Hierarchy (Sec. 5)</td><td></td><td></td></tr><tr><td colspan="4"> $0 . 5 \mathrm { p x }$ </td></tr></table>

DEN, COUNTER, and BONSAI, shuffling changes PSNR by −0.48, +0.56, and −2.50 dB, respectively, with comparable registration rates. It does, however, increase run-to-run variance, e.g. $\sigma ~ = ~ 0 . 7 4 \mathrm { d B }$ for shuffled BONSAI versus 0.16 dB for the ordered stream.

On FR1/DESK, shuffling reduces PSNR by 0.84 dB but improves SSIM and LPIPS (0.844 → 0.870 and 0.209 →

Table 19. Solidity-controller step sweep on GARDEN. ∆b denotes the per-keyframe shape update. “Solid.” and “Soft.” report the corresponding update counts.
<table><tr><td>∆b</td><td>PSNR↑</td><td>LPIPS↓</td><td>Solid.</td><td>Soft.</td></tr><tr><td>0.1</td><td>25.77</td><td>0.188</td><td>229k</td><td>81k</td></tr><tr><td>0.2</td><td>25.65</td><td>0.190</td><td>226k</td><td>79k</td></tr><tr><td>0.4 (ours)</td><td>25.82</td><td>0.187</td><td>224k</td><td>81k</td></tr><tr><td>0.8</td><td>25.80</td><td>0.188</td><td>220k</td><td>82k</td></tr></table>

![](images/d40c688c0015719eb63eab0f10ca27074ab2a82e43a6cb1d4438cd9968c83df3.jpg)

![](images/6ccf2e27cf2c3946c458c05772a40b401fce0d475fc91c03f09fde2c4faa5852.jpg)  
Figure 12. Memory and capacity over reconstruction. GPU memory remains bounded while active capacity saturates on GAR-DEN. Relocation redistributes this capacity over time, whereas hierarchy offloading allows the model to grow on SMALLCITY without increasing the GPU footprint.

0.179), with essentially unchanged registration. Runtime increases by about 1.5× as temporal adjacency is lost and more frames require shelving and retrieval. At the same time, random ordering provides a more diverse bootstrap, avoiding the poorly conditioned ordered start observed on this sequence.

Overall, the results confirm that the reconstruction does not require temporal ordering: failed sequential matches fall back to shelf-based bootstrap selection and retrieval over previously registered keyframes.

## LLM Usage Statement

We used Claude Fable 5.0 and 5.1, Claude Opus 5 and 5.5 (Anthropic), as well as GPT-5.6 Luna (OpenAI) to help draft and edit text and figures, generate and debug code, and develop and check the mathematical formulation in Sec. 5 (Sec. A). All outputs were reviewed and verified for accuracy and relevance.

Table 20. One-at-a-time sensitivity study. Each row varies one constant around its default while keeping all other settings fixed. Results are averaged over three runs; ∆ reports the PSNR change relative to the default configuration.
<table><tr><td>Scene</td><td>Constant</td><td>shipped → probe</td><td>PSNR↑</td><td> $\Delta$ </td></tr><tr><td>GARDEN</td><td> $O _ { \mathrm { s o l i d } }$  (solidify opacity gate)</td><td> $0 . 7  0 . 5$ </td><td> $2 5 . 7 8 { \pm } 0 . 0 5$ </td><td> $- 0 . 0 4 \pm 0 . 0 6$ </td></tr><tr><td>GARDEN</td><td> $O _ { \mathrm { s o l i d } }$  (solidify opacity gate)</td><td> $0 . 7  0 . 9$ </td><td>25.75±0.08</td><td> $- 0 . 0 7 \pm 0 . 0 9$ </td></tr><tr><td>GARDEN</td><td> $\theta _ { 0 } ( \mathrm { o b s e r v a b i l i t y a n g l e } )$ </td><td> $2 ^ { \circ }  1 ^ { \circ }$ </td><td>25.78±0.02</td><td> $- 0 . 0 4 \pm 0 . 0 4$ </td></tr><tr><td>GARDEN</td><td> $\theta _ { 0 }$  (observability angle)</td><td> $2 ^ { \circ }  4 ^ { \circ }$ </td><td>25.77±0.09</td><td> $- 0 . 0 5 \pm 0 . 1 0$ </td></tr><tr><td>GARDEN</td><td> $o _ { 1 }$  (verified spawn opacity)</td><td> $0 . 3  0 . 2$ </td><td>25.83±0.03</td><td> $+ 0 . 0 1 \pm 0 . 0 5$ </td></tr><tr><td>GARDEN</td><td> $o _ { 1 }$  (verified spawn opacity)</td><td> $0 . 3  0 . 5$ </td><td>25.81±0.08</td><td> $- 0 . 0 1 \pm 0 . 0 9$ </td></tr><tr><td>UNIVERSITY2</td><td> $\theta _ { 0 }$  (observability angle)</td><td> $2 ^ { \circ }  1 ^ { \circ }$ </td><td>23.61±0.04</td><td> $- 0 . 0 6 \pm 0 . 0 6$ </td></tr><tr><td>UNIVERSITY2</td><td> $\theta _ { 0 }$  (observability angle)</td><td> $2 ^ { \circ }  4 ^ { \circ }$ </td><td>23.58±0.06</td><td> $- 0 . 0 9 \pm 0 . 0 8$ </td></tr></table>

i3DGS [42]  
OtF-NVS [41]  
![](images/debfe933e7dcabfc3572438d3b4bb3cfedb0297ad9d19c738f3388bc6f42e3a8.jpg)  
Figure 13. Qualitative comparison against online reconstruction methods on GARDEN, FOREST1, UNIVERSITY2, FR1/DESK, and SMALL-CITY (top to bottom).

![](images/3e578391a413e5c9528d1b918d8470954a1ed538179f4ab65e42be18cfcf9bc0.jpg)  
Figure 14. Additional qualitative comparisons against online reconstruction methods: BONSAI, COUNTER, FOREST2, FR2/XYZ, and FR3/OFFICE (top to bottom).

![](images/e1dbdb00518c9d16ab4032ed2ba1f71af3391d57aa0163262f737784da96fe32.jpg)  
Figure 15. Additional qualitative comparisons against offline baselines: BICYCLE, KITCHEN, ROOM, TREEHILL, and GARDEN (top to bottom).

![](images/f83fdfa81857494f586b6afd8b3ce66abaa9611f630782677fe73342c0e31418.jpg)  
Figure 16. Additional qualitative comparisons against offline baselines: BONSAI, COUNTER, DR. JOHNSON, TRAIN, STUMP, and PLAY-ROOM (top to bottom).