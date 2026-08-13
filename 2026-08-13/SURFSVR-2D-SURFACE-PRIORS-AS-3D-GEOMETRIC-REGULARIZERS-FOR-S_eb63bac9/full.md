# SURFSVR: 2D SURFACE PRIORS AS 3D GEOMETRIC REGULARIZERS FOR SPARSE VOXEL RECONSTRUC-TION

Yan Di<sup>1</sup> Chengxi Li<sup>2</sup> Yaoxing Wang<sup>3</sup> Mengge Liu<sup>2</sup> Zhigang Li<sup>2</sup>

Ruida Zhang<sup>2</sup> Mingyang Li<sup>4</sup> Pengyuan Wang<sup>5</sup> Shan Gao<sup>3</sup> Xiangyang Ji<sup>2</sup>

<sup>1</sup>HIT (Shenzhen) <sup>2</sup>Tsinghua University <sup>3</sup>NWPU

<sup>4</sup>Beijing Institute of Control and Engineering <sup>5</sup>Technical University of Munich

## ABSTRACT

Sparse voxel reconstruction offers an efficient representation for high-fidelity 3D modeling, yet its geometry is commonly optimized from local photometric evidence and discrete visibility statistics. This often leads to fragmented surfaces, excessive subdivision, and floating artifacts, particularly in weakly textured or sparsely observed regions. We introduce SurfSVR, a novel sparse voxel reconstruction paradigm that treats 2D surface priors as explicit 3D geometric regularizers. Instead of directly lifting noisy pixel-wise depth predictions, SurfSVR first organizes each image into coherent surface regions by jointly reasoning over appearance, monocular depth, normals and cross-view geometry. Each region is then represented by an adaptively selected planar or quadratic surface model based on fitting reliability and geometric complexity, while cross-model agreement distinguishes reliable geometry from ambiguous predictions. These structured 2D priors are lifted into 3D and integrated throughout the reconstruction pipeline. They guide surface-adaptive voxel subdivision, provide region-level depth and normal supervision during optimization, enhance geometrically reliable sparse-observed surfaces in voxel pruning, and suppress off-surface floaters during post-refinement training. This unified design converts semantic and geometric coherence in image space into persistent structural constraints in 3D. Extensive experiments on 3 public benchmarks demonstrate that SurfSVR consistently improves sparse voxel reconstruction across scenes with substantially different visibility and geometry characteristics, achieving state-of-the-art reconstruction quality. Codes and models will be released soon.

## 1 INTRODUCTION

High-fidelity 3D reconstruction from multi-view images is fundamental to robotics, autonomous systems, and digital content creation. Sparse voxel representations are particularly attractive since they combine explicit spatial organization, efficient rendering, and direct surface extraction Fridovich-Keil et al. (2022); Sun et al. (2022); Li et al. (2025). However, their geometry is still primarily optimized using local photometric evidence, rendering weights, and discrete visibility statistics. Such signals become unreliable in weakly textured, occluded, or sparsely observed regions, often resulting in fragmented surfaces, unnecessary subdivision, and floating artifacts. Moreover, visibility-based pruning introduces an inherent trade-off: conservative rules retain floaters, whereas aggressive rules remove valid thin or sparsely observed surfaces.

Recent foundation models provide informative depth, normal, and semantic cues Yang et al. (2024); Wang et al. (2025). Nevertheless, directly lifting pixel-wise predictions into 3D is unreliable since predicted geometry can be scale-ambiguous, inconsistent across views, and inaccurate near object boundaries. More importantly, independent pixels do not explicitly describe the extent, continuity, or complexity of the physical surface to which they belong. Consequently, conventional pixel-wise regularization Li et al. (2025); Sun et al. (2025) cannot reliably distinguish a coherent sparsely observed surface from isolated prediction errors.

Pixel-wise Regularization  
![](images/b06662afbd9c458662f71990925ecb772b005cb3817d45c05a65fb7161234519.jpg)

![](images/ebd746ca719d0a439213454040de26c90609c3056758d8fca7b9b5efad454cc0.jpg)  
Figure 1: SurfSVR achieves more complete and accurate geometry by treating 2D surface priors as explicit 3D regularizers. Compared to state-of-the-art method Li et al. (2025), our approach better reconstructs weakly textured surfaces, preserves thin structures, and suppresses floating artifacts.

We argue that coherent 2D surface regions provide a better interface between image-space priors and dense 3D geometry. Based on this insight, we introduce SurfSVR, a novel sparse voxel reconstruction paradigm that treats 2D surface priors as explicit 3D geometric regularizers. SurfSVR first constructs surface regions by refining appearance-based superpixels with depth, normals, semantic boundaries, and cross-view geometric consistency. Each region is then represented by an adaptively selected planar or quadratic inverse-depth model according to its fitting reliability and geometric complexity, as shown in Figure 1. Regions that cannot be reliably explained by either low-order model remain available as complex or unknown regions rather than being forced into an unreliable fit.

The resulting structured priors are lifted into 3D and integrated throughout sparse voxel reconstruction based on GeoSVR Li et al. (2025). First, surface classes guide voxel subdivision and pruning: smooth planar regions stop at coarser octree levels, while complex surfaces and boundaries retain finer capacity. Second, the priors provide confidence-weighted depth, continuous subpixel depth, normal, and coverage supervision, enabling reliable surfaces to be optimized even when their initial opacity is low. Third, a late-stage consensus filter removes only weak voxels that several reliable views place in free space, preserving supported geometry.

Unlike methods that use geometric predictions only as auxiliary pixel-wise losses, SurfSVR transforms heterogeneous 2D predictions into persistent surface-level constraints for voxel optimization and topology control. As shown in Figure 1, SurfSVR produces more complete surfaces on weakly textured regions and better preserves fine geometric details than state-of-the-art methods. Extensive experiments on three public benchmarks demonstrate consistent improvements across diverse geometry and visibility conditions, achieving state-of-the-art reconstruction quality.

Our contributions are summarized as follows:

• We introduce a 2D-to-3D surface regularization paradigm that represents coherent image regions as explicit geometric constraints for sparse voxel reconstruction.

• We develop a geometry-aware surface construction method that combines appearance, depth, normals, semantic boundaries, and cross-view consistency with adaptive planar-orquadratic inverse-depth fitting.

• We integrate the resulting priors into confidence-weighted dense and subpixel supervision, normal alignment, coverage control, and surface-aware topology refinement.

• SurfSVR achieves state-of-the-art reconstruction quality across three public benchmarks.

## 2 RELATED WORK

## 2.1 RADIANCE FIELD REPRESENTATIONS

NeRF establishes differentiable volume rendering for learning radiance fields from posed images Mildenhall et al. (2021). Explicit and hybrid successors accelerate it with hash grids, tensor factorization, anti-aliased grids, and directly optimized sparse voxels Muller et al. (2022); Chen¨ et al. (2022); Hu et al. (2023); Liu et al. (2020); Yu et al. (2021); Fridovich-Keil et al. (2022); Sun et al. (2022). In parallel, 3DGS enables efficient splatting with explicit Gaussian primitives, including structured and level-of-detail variants Kerbl et al. (2023); Lu et al. (2024); Ren et al. (2025); Yu et al. (2024b). SVRaster instead combines adaptive sparse voxels with rasterization Sun et al. (2025), and GeoSVR adapts it to surface reconstruction using depth constraints and voxel regularization Li et al. (2025). These representations improve efficiency and scalability, but their primitives do not explicitly encode coherent physical surfaces. Consequently, voxel allocation and topology still rely mainly on local photometric and visibility evidence. SurfSVR uses coherent surface regions to regulate both geometry optimization and topology.

## 2.2 SURFACE RECONSTRUCTION WITH GEOMETRIC PRIORS

Classical methods reconstruct surfaces from multi-view correspondence and visibility Hartley & Zisserman (2004); Furukawa & Ponce (2010); Shen (2013); Zheng et al. (2014); Schonberger &¨ Frahm (2016); Schonberger et al. (2016); Yao et al. (2018). Neural approaches learn continuous¨ SDFs through differentiable rendering, with grids or staged optimization improving detail and efficiency Park et al. (2019); Yariv et al. (2020); Oechsle et al. (2021); Yariv et al. (2021); Wang et al. (2021); Wu et al. (2023); Li et al. (2023); Wang et al. (2024). Gaussian methods instead align primitives with surfaces, adopt surfel parameterizations, derive opacity fields, or couple Gaussians with SDFs Guedon & Lepetit (2024); Huang et al. (2024); Dai et al. (2024); Yu et al. (2024c;a);´ Lyu et al. (2024); Chen et al. (2023b); Xu et al. (2024); Zhang et al. (2024). Planar constraints, novel-view stereo, and local structural hints further improve mesh recovery Chen et al. (2025); Wolf et al. (2024); Wu et al. (2024). Despite these advances, geometry remains optimized at the sample or primitive level without persistent region support.

Geometric priors mitigate photometric ambiguity through patch warping, multi-view features, or predicted depth and normals Darmon et al. (2022); Chen et al. (2023a); Ren et al. (2024); Yu et al. (2022); Wang et al. (2022); Fu et al. (2022). Gaussian and voxel methods similarly use normalized depth, depth-normal consistency, or uncertainty-aware monocular constraints Li et al. (2024a); Chen et al. (2024); Turkulainen et al. (2025); Li et al. (2024b; 2025). Such per-pixel supervision remains vulnerable to scale ambiguity, boundary noise, and cross-view inconsistency. SurfSVR instead forms confidence-aware surface regions and lifts their extent, complexity, and support into persistent 3D constraints.

## 3 METHOD

## 3.1 PRELIMINARIES:

Our SurfSVR framework is built upon SVRaster Sun et al. (2025) and GeoSVR Li et al. (2025). We first briefly review the basic steps in these two sparse voxel reconstruction methods.

Scene Representation. Scenes are represented by Octree. A voxel at Octree level $l \in [ 1 , L ]$ with grid index $v = \{ i , j , k \}$ has size $v _ { s } = w _ { s } \cdot 2 ^ { - l }$ and center $v _ { c } = w _ { c } - 0 . 5 \cdot w _ { s } + v _ { s } \cdot v$ . Each voxel stores Spherical Harmonics (SH) coefficients $v _ { \mathrm { s h } }$ for view-dependent color $c _ { i } .$ , and 8 corner density values $\dot { v _ { \mathrm { g e } 0 } } \in [ 0 , + \infty ) ^ { 2 \times 2 \times 2 }$ defining a continuous density field via trilinear interpolation interp(·). Octree Voxels are dynamically pruned (low opacity) or subdivided (high gradient) during training.

Volume Rendering. For a ray intersecting N ordered voxels, pixel color C is rendered via front-toback α-blending:

$$
C = \sum _ { i = 1 } ^ { N } T _ { i } \alpha _ { i } c _ { i } , \quad T _ { i } = \prod _ { j = 1 } ^ { i - 1 } ( 1 - \alpha _ { j } ) ,\tag{1}
$$

![](images/4576e8b2ccfebeae9059440e7fe6cfb22ce2e04f119b0f230797b7b22f5617a3.jpg)  
Figure 2: Overview of the SurfSVR pipeline. Given calibrated multi-view images and geometric priors, we refine appearance-based superpixels with depth, normals, semantic boundaries, and cross-view evidence, then fit each region with the simplest reliable planar or quadratic inverse-depth model. The fitted priors provide continuous geometric supervision and guide voxel subdivision, pruning, and conservative floater removal.

where voxel opacity $\begin{array} { r } { \alpha _ { i } = 1 - \exp { \left( - \frac { \Delta t } { K } \sum _ { k = 1 } ^ { K } \mathrm { i n t e r p } ( v _ { \mathrm { g e o } } , q _ { k } ) \right) } } \end{array}$ for K samples along segment ∆t inside the voxel. Depth D is rendered similarly, while voxel surface normal n is derived from the density gradient at center $q _ { c } .$

$$
n = { \mathrm { n o r m a l i z e } } \left( \nabla _ { q } { \mathrm { i n t e r p } } ( v _ { \mathrm { g e o } } , q _ { c } ) \right) .\tag{2}
$$

## 3.2 OVERVIEW OF SURFSVR

Given COLMAP-calibrated multi-view images $\{ \mathbf { I } _ { i } \} _ { i = 1 } ^ { N }$ and their depth/normal priors, SurfSVR reconstructs a sparse voxel field V whose voxels encode geometry, opacity, and view-dependent appearance. Our core idea is to construct coherent surface regions in 2D and lift them into persistent 3D regularizers. Unlike pixel-wise supervision, a SurfSVR prior describes the spatial extent, geometric complexity, reliability, and cross-view support of a surface.

As shown in Figure 2, we first construct 2D priors by refining appearance-based superpixels using depth, normal and cross-view geometric cues, then fit each region with an automatically selected surface model. The fitted priors are used directly in the optimization losses, while a voxel-adaptation module projects them into 3D to control subdivision, pruning, and conservative floater suppression.

## 3.3 STRUCTURED 2D SURFACE PRIORS

Geometry-aware region refinement. We initialize each image with appearance-based superpixels Gauen & Chan (2025), then construct a four-neighbor pixel graph whose edges are interrupted by relative-depth jumps, normal discontinuities, semantic boundaries, and depth-validity transitions. Connected components form the initial surface candidates.

Large candidates that cannot be explained by a low-order surface are recursively divided using spatial position, inverse depth, and normal features. Adjacent fragments are merged if their shared boundary is geometrically weak and the union remains representable by a reliable surface model.

To enhance cross-view consistency, we unproject depth samples from nearby calibrated views and project them into the target image with per-source z-buffers. Samples inconsistent with the target depth are rejected, and the remaining samples are confidence-weighted and fused into a cross-view depth field. This field provides sub-pixel evidence for the current image and is used to refine region topology without copying every projected sample as a hard boundary.

Adaptive surface model selection. Given a refined surface region R, SurfSVR models the inverse depth $s _ { \mathcal { R } } ( \mathbf { p } ) = z ^ { - 1 }$ of every pixel $\mathbf { p } = [ u , v ] ^ { \top }$ as a low-order polynomial defined on normalized image coordinates:

$$
s _ { \mathcal { R } } ( \mathbf { p } ) = \pmb { \theta } _ { \mathcal { R } } ^ { \top } \phi _ { k } ( \tilde { \mathbf { p } } ) , \qquad \tilde { \mathbf { p } } = \frac { \mathbf { p } - \pmb { \mu } _ { \mathcal { R } } } { \pmb { \sigma } _ { \mathcal { R } } } ,\tag{3}
$$

where $\mu _ { \mathcal { R } }$ is the coordinate-wise median and $\sigma _ { \mathcal { R } }$ is a robust coordinate scale computed from the 90th percentile absolute deviation within region $\mathcal { R }$ , clamped to at least one pixel, and $\bar { \tilde { \bf p } } = [ \tilde { u } , \tilde { v } ] ^ { \top }$ denotes the normalized coordinate. The vector $\scriptstyle \theta _ { \mathcal { R } }$ contains the polynomial coefficients to be estimated for region R, while $\phi _ { k } ( \cdot )$ is the basis function of a polynomial with degree k. We consider two models: $\phi _ { 1 } ( x , y ) = [ 1 , \stackrel { . } { x } , y ] ^ { \top }$ , which represents a planar surface, and $\bar { \phi _ { 2 } ( x , y ) } = [ 1 , x , y , x ^ { 2 } , x y , y ^ { 2 } ] ^ { \top }$ which models smoothly curved surfaces. Under the pinhole camera model, the first-order polynomial corresponds to a 3D plane in image space, whereas the second-order polynomial captures local quadratic surface variations.

SurfSVR first attempts the planar model and introduces the quadratic model only when the former cannot satisfy the fitting criteria, thereby selecting the simplest model that adequately explains the geometry of the region.

We initialize each fit with RANSAC and refine using Tukey-biweight IRLS:

$$
\pmb { \theta } _ { \mathcal { R } } ^ { \ast } = \arg \operatorname* { m i n } _ { \pmb { \theta } } \sum _ { j \in \mathcal { R } } w _ { j \rho } \left( \pmb { \theta } ^ { \top } \phi _ { k } ( \tilde { \mathbf { p } } _ { j } ) - z _ { j } ^ { - 1 } \right) .\tag{4}
$$

SurfSVR selects the simplest model satisfying fitting-error, inlier-ratio, support, positivity, and conditioning requirements. A planar model is tested first; the quadratic model is introduced only when necessary. Regions that cannot be reliably represented by either are retained as complex or unknown rather than being assigned a misleading low-order surface.

Region confidence $c _ { \mathcal { R } }$ combines inlier ratio $r _ { \mathcal { R } }$ , fitting error $e _ { \mathcal { R } }$ , and support $n _ { \mathcal { R } }$

$$
c _ { \mathcal { R } } = r _ { \mathcal { R } } \exp \left( - \frac { e _ { \mathcal { R } } } { \tau _ { e } } \right) \operatorname* { m i n } \left( 1 , \frac { n _ { \mathcal { R } } } { \tau _ { n } } \right) ,\tag{5}
$$

where $\tau _ { n }$ is the support reference (128 samples in our configuration).

## 3.4 SURFACE-ADAPTIVE SPARSE VOXELS

We lift 2D priors into 3D by projecting each voxel center into all available views. A voxel receives a smooth-surface vote when it lies within a voxel-scaled distance of a reliable planar or quadratic fit. Projections onto unreliable regions or boundaries produce complex-surface votes. Cross-view voting assigns each voxel a persistent class:

$$
\ell ( \mathbf { x } ) \in \{ \mathrm { p l a n e , q u a d r a t i c , c o m p l e x , u n k n o w n } \} .\tag{6}
$$

The surface class regulates maximum octree level, subdivision priority, and conditional pruning protection. Planar regions stop at coarser levels, quadratic regions receive intermediate resolution, and complex or unknown regions retain the full voxel budget. A voxel can subdivide only if $l ( \mathbf { x } ) <$ $L _ { \ell ( \mathbf { x } ) }$ , where $L _ { \ell }$ is the class-dependent level cap. Complex voxels receive higher subdivision priority, while reliable multi-view observations lower the risk of premature pruning. During late refinement, the topology is frozen and the surface losses continue to optimize geometry.

## 3.5 SURFACE-REGULARIZED OPTIMIZATION

Let $\alpha _ { i } ( \mathbf { p } ) = 1 - T _ { i } ( \mathbf { p } )$ be the accumulated opacity of a rendered ray and $D _ { i } ( \mathbf { p } )$ its accumulated depth numerator. We compute expected depth as

$$
\bar { z } _ { i } ( \mathbf { p } ) = \frac { D _ { i } ( \mathbf { p } ) } { \mathrm { s g } ( \mathrm { m a x } ( \alpha _ { i } ( \mathbf { p } ) , \epsilon ) ) } ,\tag{7}
$$

where $\operatorname { s g } ( \cdot )$ denotes stop-gradient. This prevents the model from reducing geometric loss merely by changing opacity. The complete surface regularization is

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { s u r f } } = \eta ( t ) \left( \lambda _ { d } \mathcal { L } _ { d } + \lambda _ { s } \mathcal { L } _ { \mathrm { s u b } } + \lambda _ { n } \mathcal { L } _ { n } \right) } \\ { + \eta _ { c } ( t ) \lambda _ { c } \mathcal { L } _ { \mathrm { c o v } } , \qquad } \end{array}\tag{8}
$$

where $\eta ( t )$ gradually adds geometric supervision and $\eta _ { c } ( t )$ decays coverage supervision near the end of optimization. For reliable pixel centers with sufficient rendered opacity, we employ a confidenceweighted robust log-depth loss:

$$
\mathcal { L } _ { d } = \frac { \sum _ { \mathbf { p } } c _ { \mathcal { R } ( \mathbf { p } ) } \rho _ { \epsilon } \big ( \log \bar { z } ( \mathbf { p } ) - \log \hat { z } ( \mathbf { p } ) \big ) } { \sum _ { \mathbf { p } } c _ { \mathcal { R } ( \mathbf { p } ) } } ,\tag{9}
$$

where $\rho _ { \epsilon } ( x ) = \sqrt { x ^ { 2 } + \epsilon ^ { 2 } }$ denotes a robust distance function. Pixels are selected only when the fitted region passes the reliability gates and the rendered opacity exceeds a small depth-supervision floor; the depth denominator is detached from the gradient so that opacity alone cannot reduce the loss.

The fitted priors provide supervision at arbitrary continuous coordinates. We sample a continuous image coordinate q inside a reliable region:

$$
\begin{array} { r } { \mathbf { q } = \mathbf { p } + \delta , \qquad \delta \sim \mathcal { U } ( - 0 . 5 , 0 . 5 ) ^ { 2 } . } \end{array}\tag{10}
$$

We normalize q using the same region-specific transform used during surface fitting:

$$
\tilde { \mathbf { q } } = \frac { \mathbf { q } - \pmb { \mu } _ { \mathcal { R } } } { \pmb { \sigma } _ { \mathcal { R } } } .\tag{11}
$$

The target depth at q is evaluated as

$$
z _ { \mathcal { R } } ( \mathbf { q } ) = \left[ \pmb { \theta } _ { \mathcal { R } } ^ { \top } \phi _ { k _ { \mathcal { R } } } \left( \mathbf { \tilde { q } } \right) \right] ^ { - 1 } .\tag{12}
$$

We bilinearly sample the rendered depth at q to compute $\mathcal { L } _ { \mathrm { s u b } }$ . Regions are sampled with inversearea weighting so that small surfaces are not overwhelmed. The normal loss aligns rendered cameraspace normals with the region prior:

$$
\mathcal { L } _ { n } = \frac { \sum _ { \mathbf { p } } c _ { \mathcal { R } ( \mathbf { p } ) } \left( 1 - \left| \mathbf { n } ( \mathbf { p } ) ^ { \top } \hat { \mathbf { n } } ( \mathbf { p } ) \right| \right) } { \sum _ { \mathbf { p } } c _ { \mathcal { R } ( \mathbf { p } ) } } .\tag{13}
$$

Depth supervision alone provides little gradient for rays with very low accumulated opacity, making it difficult to recover valid yet transparent surfaces. To encourage reliable surface regions to acquire sufficient opacity, we introduce a surface coverage loss:

$$
\mathcal { L } _ { \mathrm { c o v } } = \frac { \sum _ { \mathbf { p } } \tilde { c } ( \mathbf { p } ) \left[ \mathrm { m a x } ( 0 , \alpha ^ { * } - \alpha ( \mathbf { p } ) ) \right] ^ { 2 } } { \sum _ { \mathbf { p } } \tilde { c } ( \mathbf { p } ) } ,\tag{14}
$$

where $\alpha ^ { * }$ is the target minimum opacity threshold, and $\tilde { c } ( \mathbf { p } )$ is the confidence weight used for coverage supervision. Specifically, $\tilde { c } ( \mathbf { p } ) = c _ { \mathcal { R } } ( \mathbf { p } )$ only when pixel p belongs to a reliable planar or quadratic surface region, or when its geometry is independently verified by projected observations from neighboring views; otherwise, $\tilde { c } ( \mathbf { p } ) = 0$

During optimization, the coverage sampler explicitly includes the lowest-opacity reliable pixels so that valid but under-reconstructed surfaces receive stronger opacity gradients. Since excessive opacity may lead to artificially thick surfaces, the weight of ${ \mathcal { L } } _ { \mathrm { c o v } }$ is gradually reduced during the late training stage, allowing the optimization to focus on accurate surface positions rather than surface thickness.

## 3.6 SURFACE-AWARE PRUNING AND FLOATER SUPPRESSION

Visibility alone cannot determine whether a weak voxel belongs to a valid sparsely observed surface. SurfSVR therefore lowers the pruning threshold for voxels supported by reliable planar or quadratic priors. This protection is deliberately conditional: a child with no rendering evidence can still be removed.

Table 1: Quantitative comparison on DTU in terms of Chamfer distance (↓). Green and light-yellow cells denote the best and second-best results, respectively.
<table><tr><td>Method</td><td>24</td><td>37</td><td>40</td><td>55</td><td>63</td><td>65</td><td>69</td><td>83</td><td>97 105</td><td></td><td>106 110</td><td>114</td><td>118</td><td>122</td><td>Mean</td></tr><tr><td>VolSDF Yariv et al. (2021)</td><td>1.14</td><td>1.26</td><td>0.81</td><td>0.49</td><td>1.25</td><td>0.70</td><td>0.72</td><td>1.29</td><td>1.18</td><td>0.70</td><td>0.66</td><td>1.08 0.42</td><td>0.61</td><td>0.55</td><td>0.86</td></tr><tr><td>NeuS Wang et al. (2021)</td><td>1.00</td><td>1.37</td><td>0.93</td><td>0.43</td><td>1.10</td><td>0.65</td><td>0.57</td><td>1.48</td><td>1.09</td><td>0.83 0.52</td><td>1.20</td><td>0.35</td><td>0.49</td><td>0.54</td><td>0.84</td></tr><tr><td>Neuralangelo Li et al. (2023)</td><td>0.37</td><td>0.72</td><td>0.35</td><td>0.35</td><td>0.87</td><td>0.54</td><td>0.53</td><td>1.29</td><td>0.97</td><td>0.73 0.47</td><td>0.74</td><td>0.32</td><td>0.41</td><td>0.43</td><td>0.61</td></tr><tr><td>Geo-NeuS Fu et al. (2022)</td><td>0.38</td><td>0.54</td><td>0.34</td><td>0.36</td><td>0.80</td><td>0.45</td><td>0.41</td><td>1.03</td><td>0.84</td><td>0.55</td><td>0.46 0.47</td><td>0.29</td><td>0.36</td><td>0.35</td><td>0.51</td></tr><tr><td>MonoSDF Yu et al. (2022)</td><td>0.66</td><td>0.88</td><td>0.43</td><td>0.40</td><td>0.87</td><td>0.78</td><td>0.81</td><td>1.23</td><td>1.18</td><td>0.66</td><td>0.66 0.96</td><td>0.41</td><td>0.57</td><td>0.51</td><td>0.73</td></tr><tr><td>2DGS Huang et al. (2024)</td><td>0.48</td><td>0.91</td><td>0.39</td><td>0.39</td><td>1.01</td><td>0.83</td><td>0.81</td><td>1.36</td><td>1.27</td><td>0.76</td><td>0.70 1.40</td><td>0.40</td><td>0.76</td><td>0.52</td><td>0.80</td></tr><tr><td>GOF Yu et al. (2024c)</td><td>0.50</td><td>0.82</td><td>0.37</td><td>0.37</td><td>1.12</td><td>0.74</td><td>0.73</td><td>1.18</td><td>1.29</td><td>0.68</td><td>0.77</td><td>0.90 0.42</td><td>0.66</td><td>0.49</td><td>0.74</td></tr><tr><td>SVRaster Sun et al. (2025)</td><td>0.61</td><td>0.74</td><td>0.41</td><td>0.36</td><td>0.93</td><td>0.75</td><td>0.94</td><td>1.33</td><td>1.40</td><td>0.61</td><td>0.63</td><td>1.19 0.43</td><td>0.57</td><td>0.44</td><td>0.76</td></tr><tr><td>GS2Mesh Wolf et al. (2024)</td><td>0.59</td><td>0.79</td><td>0.70</td><td>0.38</td><td>0.78</td><td>1.00</td><td>0.69</td><td>1.25</td><td>0.96</td><td>0.59</td><td>0.50</td><td>0.68 0.37</td><td>0.50</td><td>0.46</td><td>0.68</td></tr><tr><td>VCR-GauS Chen et al. (2024)</td><td>0.55</td><td>0.91</td><td>0.40</td><td>0.43</td><td>0.97</td><td>0.95</td><td>0.84</td><td>1.39</td><td>1.30</td><td>0.90</td><td>0.76</td><td>0.92 0.44</td><td>0.75</td><td>0.54</td><td>0.80</td></tr><tr><td>MonoGSDF Li et al. (2024b)</td><td>0.45</td><td>0.65</td><td>0.36</td><td>0.36</td><td>0.94</td><td>0.70</td><td>0.67</td><td>1.27</td><td>0.99</td><td>0.63</td><td>0.49</td><td>0.84 0.39</td><td>0.53</td><td>0.47</td><td>0.65</td></tr><tr><td>PGSR Chen et al. (2025)</td><td>0.36</td><td>0.57</td><td>0.38</td><td>0.33</td><td>0.78</td><td>0.58</td><td>0.50</td><td>1.08</td><td>0.63</td><td>0.59</td><td>0.46</td><td>0.54 0.30</td><td>0.38</td><td>0.34</td><td>0.52</td></tr><tr><td>AmbiSuR Li et al. (2026)</td><td>0.32</td><td>0.48</td><td>0.31</td><td>0.33</td><td>0.65</td><td>0.48</td><td>0.42</td><td>0.97</td><td>0.61</td><td>0.54</td><td>0.36</td><td>0.44 0.29</td><td>0.34</td><td>0.34</td><td>0.46</td></tr><tr><td>GeoSVR Li et al. (2025)</td><td>0.32</td><td>0.51</td><td>0.30</td><td>0.33</td><td>0.71</td><td>0.48</td><td>0.42</td><td>1.03</td><td>0.62</td><td>0.56</td><td>0.33</td><td>0.46 0.30</td><td>0.34</td><td>0.32</td><td>0.47</td></tr><tr><td>SurfSVR (Ours)</td><td>0.31</td><td>0.49</td><td>0.29</td><td>0.30</td><td>0.65</td><td>0.45</td><td>0.43</td><td>0.97</td><td>0.62</td><td>0.54</td><td>0.35</td><td>0.47 0.30</td><td>0.33</td><td>0.31</td><td>0.45</td></tr></table>

After the standard subdivision and pruning operations, we perform a separate conservative floatersuppression step. For a voxel x, let $\bar { \mathcal { V } } ( \mathbf { x } )$ denote the set of reliable views in which x projects inside a valid fitted surface region and is visible. For each $i \in \mathcal { V } ( { \mathbf { x } } )$ , we define

$$
\Delta _ { i } ( \mathbf { x } ) = z _ { i } ( \mathbf { x } ) - z _ { \mathcal { R } _ { i } } ( \pi _ { i } ( \mathbf { x } ) ) ,\tag{15}
$$

which measures the camera-space displacement between the voxel center and the fitted surface in view i. We then accumulate surface-support and free-space votes as

$$
S ( \mathbf { x } ) = \sum _ { i \in \mathcal { V } ( \mathbf { x } ) } \mathbb { I } [ | \Delta _ { i } ( \mathbf { x } ) | \leq \tau _ { s } h _ { \mathbf { x } } ] ,\tag{16}
$$

$$
F ( \mathbf { x } ) = \sum _ { i \in \mathcal { V } ( \mathbf { x } ) } \mathbb { I } [ \Delta _ { i } ( \mathbf { x } ) < - \tau _ { f } h _ { \mathbf { x } } ] .\tag{17}
$$

Under the camera-space depth convention, $\Delta _ { i } ( { \bf x } ) \ < \ 0$ indicates that the voxel lies between the camera and the fitted surface and is therefore supported as free-space evidence. We remove a voxel only when it receives no surface-support votes, receives sufficient free-space votes, and has weak rendering evidence:

$$
\mathrm { r e m o v e } ( \mathbf { x } ) = \mathbb { I } [ S ( \mathbf { x } ) = 0 \mathrm { ~ } \land \ F ( \mathbf { x } ) \ge n _ { f } \mathrm { ~ } \land \ e _ { \mathrm { r e n d e r } } ( \mathbf { x } ) < \tau _ { r } ] .\tag{18}
$$

Here, $e _ { \mathrm { r e n d e r } } ( \mathbf { x } )$ denotes the rendering-evidence score used by the underlying pruning procedure, and lower values indicate weaker support. We cap the number of removed voxels per pass. The topology is then fixed during late refinement while continuous surface supervision remains active.

## 4 EXPERIMENTS

## 4.1 EXPERIMENTAL SETUP

Datasets. Following GeoSVR Li et al. (2025), we evaluate SurfSVR on three public benchmarks: DTU Jensen et al. (2014), Tanks and Temples (TnT) Knapitsch et al. (2017), and Mip-NeRF 360 Barron et al. (2022). For DTU, we use the 15 commonly evaluated scans and report Chamfer Distance (lower is better). For TnT, we evaluate on 5 scenes from the training split and report F1-Score (higher is better). Courthouse is removed due to inaccurate ground truth. For Mip-NeRF 360, we use all nine scenes (five outdoor and four indoor scenes) and evaluate novel-view synthesis quality. We follow the same image downsampling, camera preprocessing, scene selection, and evaluation protocols as GeoSVR.

![](images/31e1ade7b75bfe47d82599134186075932d57a6367f1ee0c3d54e94eef7f99f6.jpg)  
Figure 3: Visualization results on DTU dataset.

Implementation details. SurfSVR is implemented upon the GeoSVR. On DTU, we first optimize the complete SurfSVR using Equation 8 and apperance loss from GeoSVR for 20,000 iterations, then conduct 2,000 iterations of topology-frozen surface-guided late-refinement. On TnT and Mip-NeRF, the training step is 20,000 for complete training and 10,000 for late-refinement. Structured priors are precomputed offline using BiST superpixels, neighboring 48-view depth projection, and robust surface fitting. Training uses reliable regions selected by degree, confidence, fitting-error, inlier, support, and opacity gates. For mesh extraction, we adopt TSDF fusion with a voxel size of 0.002 on DTU and follow PGSR Chen et al. (2025) to determine extraction resolution for TnT. Surface losses are gradually activated after initialization. Coverage supervision ${ \mathcal { L } } _ { \mathrm { c o v } }$ is decayed linearly to 0 during late-refinement to prevent surface thickening. Other details are provided in supplementary material. All experiments are conducted on a single NVIDIA A100 GPU.

## 4.2 COMPARISON WITH THE STATE OF THE ART

DTU. Table 1 compares SurfSVR with baseline methods across all 15 DTU scans. SurfSVR achieves the best overall performance with a mean Chamfer distance of 0.45, outperforming both GeoSVR (0.47) and AmbiSuR (0.46). Notably, SurfSVR obtains the best results, including ties, on 9 out of 15 scenes and the marked second-best results on 3 scenes. It matches the best baseline on scan 63 and achieves a tied best result on scan 65, two challenging scenes with weakly textured surfaces. Figure 3 visualizes the reconstructed meshes, demonstrating that SurfSVR produces more complete surfaces on smooth regions while preserving fine geometric details better than baseline methods.

Tanks and Temples. Table 2 reports per-scene F1-scores on TnT. SurfSVR achieves the best mean F1-score of 0.62, surpassing GeoSVR (0.60) and mono-AmbiSuR (0.61). We achieve best performance on Caterpillar, Meetingroom and Truck scenes. Figure 4 demonstrates the visulization comparison results. SurfSVR reliably recovers thin structures in Caterpillar as highlighted in rectangle. For Meetingroom, our surface fitting strategy helps recover the reflective ground.

Mip-NeRF 360. Since Mip-NeRF 360 does not provide ground-truth geometry, we follow previous work and evaluate novel-view synthesis quality, reporting results separately for outdoor and indoor scenes. Table 3 shows that SurfSVR achieves competitive rendering quality while maintaining accurate geometry. For outdoor scenes, SurfSVR obtains the best PSNR (24.90) and second-best

Table 2: Quantitative comparison on Tanks and Temples. We report per-scene F1-score (↑) and mean F1-score. \* denotes results of our own re-implementation under the same experimental conditions.
<table><tr><td>Method</td><td>Barn</td><td>Caterpillar</td><td>Ignatius</td><td>Meetingroom</td><td>Truck</td><td>Mean</td></tr><tr><td>NeuS Wang et al. (2021)</td><td>0.29</td><td>0.29</td><td>0.83</td><td>0.24</td><td>0.45</td><td>0.42</td></tr><tr><td>Neuralangelo Li et al. (2023)</td><td>0.70</td><td>0.36</td><td>0.89</td><td>0.32</td><td>0.48</td><td>0.55</td></tr><tr><td>Geo-NeuS Fu et al. (2022)</td><td>0.33</td><td>0.26</td><td>0.72</td><td>0.20</td><td>0.45</td><td>0.39</td></tr><tr><td>MonoSDF Yu et al. (2022)</td><td>0.49</td><td>0.31</td><td>0.78</td><td>0.23</td><td>0.42</td><td>0.45</td></tr><tr><td>2DGS Huang et al. (2024)</td><td>0.41</td><td>0.23</td><td>0.51</td><td>0.17</td><td>0.45</td><td>0.35</td></tr><tr><td>GOF Yu et al. (2024c)</td><td>0.51</td><td>0.41</td><td>0.68</td><td>0.28</td><td>0.59</td><td>0.49</td></tr><tr><td>SVRaster Sun et al. (2025)</td><td>0.35</td><td>0.33</td><td>0.69</td><td>0.19</td><td>0.54</td><td>0.42</td></tr><tr><td>VCR-GauS Chen et al. (2024)</td><td>0.62</td><td>0.26</td><td>0.61</td><td>0.19</td><td>0.52</td><td>0.44</td></tr><tr><td>MonoGSDF Li et al. (2024b)</td><td>0.56</td><td>0.38</td><td>0.72</td><td>0.25</td><td>0.62</td><td>0.51</td></tr><tr><td>PGSR Chen et al. (2025)</td><td>0.66</td><td>0.44</td><td>0.81</td><td>0.33</td><td>0.66</td><td>0.58</td></tr><tr><td>GeoSVR* Li et al. (2025)</td><td>0.68</td><td>0.49</td><td>0.83</td><td>0.37</td><td>0.65</td><td>0.60</td></tr><tr><td>AmbiSuRM Li et al. (2026)</td><td>0.67</td><td>0.49</td><td>0.83</td><td>0.39</td><td>0.68</td><td>0.61</td></tr><tr><td>SurfSVR (Ours)</td><td>0.67</td><td>0.52</td><td>0.83</td><td>0.40</td><td>0.69</td><td>0.62</td></tr></table>

![](images/50d03db86c9cd5e1c5544bdaba0d0dcba1b040b410ade59a6c0583719d39f71d.jpg)  
Figure 4: Visualization results on TnT dataset. Our results preserve more reliable geometry than baselines.

SSIM (0.750), comparable to PGSR and GOF. For indoor scenes, our method achieves second-best SSIM (0.929) and competitive LPIPS (0.166). These results demonstrate that enforcing geometric accuracy through surface priors does not compromise appearance reconstruction quality, validating the effectiveness of our surface-regularized optimization framework.

## 4.3 ABLATION STUDY

We perform an incremental ablation study to evaluate both the effectiveness of the proposed 2D-to-3D surface regularization pipeline and the quality of the constructed surface priors in Table 5.

Stage I: Effectiveness of 2D-to-3D surface regularization. Starting from the GeoSVR baseline (Row A), we progressively introduce each component of the proposed surface-guided optimization pipeline.

Adding continuous surface supervision (Row B) reduces the DTU Chamfer distance from 0.477 to 0.465 and improves the TnT F1-score from 0.606 to 0.617. This demonstrates that region-level depth, continuous subpixel supervision, normal alignment, and coverage constraints provide substantially denser and more reliable geometric guidance than pixel-wise supervision alone.

Row C further incorporates surface-adaptive subdivision and surface-aware pruning. Although the numerical improvement is relatively modest (0.465→0.464 on DTU and 0.617→0.619 on TnT), these operations improve voxel allocation efficiency by assigning finer resolution only to geomet rically complex regions while protecting reliable but sparsely observed surfaces from premature pruning.

<table><tr><td></td><td colspan="3">Outdoor</td><td colspan="3">Indoor</td></tr><tr><td>Method</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td></tr><tr><td>NeRF</td><td>21.46</td><td>0.458</td><td>0.515</td><td>26.84</td><td>0.790</td><td>0.370</td></tr><tr><td>Deep Blending</td><td>21.54</td><td>0.524</td><td>0.364</td><td>26.40</td><td>0.844</td><td>0.261</td></tr><tr><td>Instant NGP</td><td>22.90</td><td>0.566</td><td>0.371</td><td>29.15</td><td>0.880</td><td>0.216</td></tr><tr><td>Mip-NeRF 360</td><td>24.47</td><td>0.691</td><td>0.283</td><td>31.72</td><td>0.917</td><td>0.180</td></tr><tr><td>3DGS</td><td>24.67</td><td>0.728</td><td>0.240</td><td>30.96</td><td>0.924</td><td>0.187</td></tr><tr><td>SVRaster</td><td>24.68</td><td>0.738</td><td>0.206</td><td>30.65</td><td>0.927</td><td>0.161</td></tr><tr><td>BakedSDF</td><td>22.47</td><td>0.585</td><td>0.349</td><td>27.06</td><td>0.836</td><td>0.258</td></tr><tr><td>SuGaR</td><td>22.93</td><td>0.629</td><td>0.356</td><td>29.43</td><td>0.906</td><td>0.225</td></tr><tr><td>2DGS</td><td>24.34</td><td>0.717</td><td>0.246</td><td>30.40</td><td>0.916</td><td>0.195</td></tr><tr><td>GOF</td><td>24.82</td><td>0.750</td><td>0.202</td><td>30.79</td><td>0.924</td><td>0.184</td></tr><tr><td>VCR-GauS</td><td>24.31</td><td>0.707</td><td>0.280</td><td>30.53</td><td>0.921</td><td>0.184</td></tr><tr><td>PGSR</td><td>24.76</td><td>0.752</td><td>0.203</td><td>30.36</td><td>0.934</td><td>0.147</td></tr><tr><td>GeoSVR</td><td>24.83</td><td>0.738</td><td>0.218</td><td>30.46</td><td>0.921</td><td>0.172</td></tr><tr><td>SurfSVR (Ours)</td><td>24.90</td><td>0.750</td><td>0.203</td><td>30.76</td><td>0.929</td><td>0.166</td></tr></table>

Table 4: Runtime comparison of different surface reconstruction methods on DTU dataset.

<table><tr><td>Method</td><td>Time (h)</td></tr><tr><td>NeuS Neuralangelo Geo-NeuS</td><td>&gt; 12h &gt; 128h &gt; 12h</td></tr><tr><td>MonoSDF</td><td>6h</td></tr><tr><td>2DGS GOF</td><td>0.2h 1h</td></tr><tr><td>GS2Mesh</td><td>0.3h</td></tr><tr><td>VCR-GauS PGSR</td><td>1h</td></tr><tr><td></td><td></td></tr><tr><td></td><td>0.5h</td></tr><tr><td>GeoSVR</td><td>0.8h</td></tr><tr><td>AmbiSuR</td><td>0.6h</td></tr><tr><td></td><td></td></tr><tr><td>SurfSVR (Ours)</td><td>0.9h</td></tr></table>

Table 5: Ablation study of SurfSVR. Stage I ablates 2D-to-3D constraints; Stage II replaces the 2D surface construction under the full pipeline.
<table><tr><td>ID Variant</td><td>DTU Chamfer TnT F1 ↓</td><td>↑</td></tr><tr><td>Stage I: 2D-to-3D constraints</td><td></td><td></td></tr><tr><td>A GeoSVR baseline*</td><td>0.477</td><td>0.606</td></tr><tr><td>B A + continuous surface supervision</td><td>0.465</td><td>0.617</td></tr><tr><td>C B + surface-adaptive</td><td>0.464</td><td>0.619</td></tr><tr><td>subdivision and pruning D C + Floater</td><td>0.454</td><td>0.622</td></tr><tr><td>suppression</td><td></td><td></td></tr><tr><td>Stage II: 2D surface construction E Full SurfSVR w/o</td><td>0.461</td><td>0.615</td></tr><tr><td>appearance-region fitting</td><td></td><td></td></tr><tr><td>F Full SurfSVR w/o</td><td>0.458</td><td>0.620</td></tr><tr><td>boundary-refined surfaces</td><td>0.456</td><td></td></tr><tr><td>G Full SurfSVR w/</td><td></td><td>0.620</td></tr><tr><td>only planar surface</td><td></td><td></td></tr><tr><td>H Full SurfSVR w/o cross-view</td><td>0.457</td><td>0.621</td></tr><tr><td>surface evidence (Ours)</td><td></td><td></td></tr></table>

Finally, Row D enables the proposed post-stage floater suppression strategy, yielding the largest gain in the second stage of optimization. DTU Chamfer further decreases from 0.464 to 0.454, while the TnT F1-score increases from 0.619 to 0.622. Since voxels are removed only when multiple reliable surface observations consistently identify them as free space, the proposed consensus mechanism effectively suppresses floating artifacts without sacrificing valid thin structures or reconstruction completeness.

Stage II: Quality of structured surface construction. Under the complete reconstruction pipeline, we further analyze how different surface construction strategies affect the final reconstruction quality.

Removing appearance-region fitting (Row E) leads to a noticeable performance degradation (0.454→0.461 on DTU and 0.622→0.615 on TnT), indicating that initializing surface candidates from coherent appearance regions provides an important structural prior before geometric refinement.

Row F restores appearance-region fitting but removes the proposed boundary refinement, resulting in partial recovery (0.458 on DTU and 0.620 on TnT). This shows that refining region topology using depth discontinuities, normal changes, and geometric boundaries substantially improves the quality of fitted surfaces by preventing different physical surfaces from being merged together.

Row G indicates that using planar 2D surface priors alone results in no performance degradation on the large-scale TnT and only a little drop on object-centric dataset DTU, since 67% of the surface priors on DTU and 89% on TnT are classified as planar surfaces in SurfSVR.

Finally, Row H further evaluates the effectiveness of cross-view geometry in surface prior construction. Without this term, the performance drops from 0.454 to 0.457 on DTU and from 0.622 to 0.621 on TnT, which is a small but consistent degradation in the reported ablation. The experiment verifies its importance.

## 4.4 EFFICIENCY AND LIMITATIONS

SurfSVR requires approximately 54 minutes to process a DTU scene on a single NVIDIA A100 GPU, including superpixel extraction and robust cross-view surface fitting. Table 4 compares SurfSVR against baselines. The additional cost of SurfSVR comes from offline surface-prior construction and the added geometric supervision during voxel optimization. SurfSVR trades part of the efficiency advantage of GeoSVR for more accurate geometry and more reliable surface topology. Future work may accelerate the pipeline through parallel surface fitting, incremental cross-view projection, and tighter coupling between the optimization of 2D priors and the 3D voxel representation.

## 5 CONCLUSION

We presented SurfSVR, a surface-regularized sparse voxel reconstruction framework that transforms structured 2D surface priors into persistent 3D geometric constraints. SurfSVR constructs reliable regions from appearance, depth, normals and cross-view consistency, and represents them with adaptively selected planar-quadratic models. These priors directly supervise continuous geometry and regulate voxel subdivision, pruning, and floater suppression, allowing surface structure to influence both geometry optimization and topology. Experiments demonstrate state-of-the-art reconstruction accuracy on 3 public benchmarks. Future work will focus on more efficient surface construction and tighter joint optimization.

## REFERENCES

Jonathan T. Barron, Ben Mildenhall, Dor Verbin, Pratul P. Srinivasan, and Peter Hedman. Mip-NeRF 360: Unbounded anti-aliased neural radiance fields. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 5470–5479, 2022.

Anpei Chen, Zexiang Xu, Andreas Geiger, Jingyi Yu, and Hao Su. TensoRF: Tensorial radiance fields. In European Conference on Computer Vision, pp. 333–350. Springer, 2022.

Danpeng Chen, Hai Li, Weicai Ye, Yifan Wang, Weijian Xie, Shangjin Zhai, Nan Wang, Haomin Liu, Hujun Bao, and Guofeng Zhang. PGSR: Planar-based gaussian splatting for efficient and high-fidelity surface reconstruction. IEEE Transactions on Visualization and Computer Graphics, 31(9):6100–6111, 2025. doi: 10.1109/TVCG.2024.3494046.

Decai Chen, Peng Zhang, Ingo Feldmann, Oliver Schreer, and Peter Eisert. Recovering fine details for neural implicit surface reconstruction. In Proceedings of the IEEE/CVF Winter Conference on Applications ofComputer Vision, pp. 4330–4339, 2023a.

Hanlin Chen, Chen Li, Yunsong Wang, and Gim Hee Lee. NeuSG: Neural implicit surface reconstruction with 3D gaussian splatting guidance. arXiv preprint arXiv:2312.00846, 2023b.

Hanlin Chen, Fangyin Wei, Chen Li, Tianxin Huang, Yunsong Wang, and Gim Hee Lee. VCR-GauS: View consistent depth-normal regularizer for gaussian surface reconstruction. In Advances in Neural Information Processing Systems, volume 37, pp. 139725–139750, 2024.

Pinxuan Dai, Jiamin Xu, Wenxiang Xie, Xinguo Liu, Huamin Wang, and Weiwei Xu. High-quality surface reconstruction using gaussian surfels. In ACM SIGGRAPH 2024 Conference Papers, pp. 1–11, 2024.

Franc¸ois Darmon, Ben´ edicte Bascle, Jean-Cl´ ement Devaux, Pascal Monasse, and Mathieu Aubry.´ Improving neural implicit surfaces geometry with patch warping. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 6260–6269, 2022.

Sara Fridovich-Keil, Alex Yu, Matthew Tancik, Qinhong Chen, Benjamin Recht, and Angjoo Kanazawa. Plenoxels: Radiance fields without neural networks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 5501–5510, 2022.

Qiancheng Fu, Qingshan Xu, Yew-Soon Ong, and Wenbing Tao. Geo-NeuS: Geometry-consistent neural implicit surfaces learning for multi-view reconstruction. In Advances in Neural Information Processing Systems, volume 35, pp. 3403–3416, 2022.

Yasutaka Furukawa and Jean Ponce. Accurate, dense, and robust multiview stereopsis. IEEE Transactions on Pattern Analysis and Machine Intelligence, 32(8):1362–1376, 2010.

Kent W Gauen and Stanley H Chan. Bayesian-inspired space-time superpixels. prepint, 2025.

Antoine Guedon and Vincent Lepetit. SuGaR: Surface-aligned gaussian splatting for efficient 3D´ mesh reconstruction and high-quality mesh rendering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 5354–5363, 2024.

Richard Hartley and Andrew Zisserman. Multiple View Geometry in Computer Vision. Cambridge University Press, Cambridge, UK, 2 edition, 2004.

Wenbo Hu, Yuling Wang, Lin Ma, Bangbang Yang, Lin Gao, Xiao Liu, and Yuewen Ma. Tri-miprf: Tri-mip representation for efficient anti-aliasing neural radiance fields. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 19774–19783, 2023.

Binbin Huang, Zehao Yu, Anpei Chen, Andreas Geiger, and Shenghua Gao. 2D gaussian splatting for geometrically accurate radiance fields. In ACM SIGGRAPH 2024 Conference Papers, pp. 1–11, 2024.

Rasmus Jensen, Anders Dahl, George Vogiatzis, Engin Tola, and Henrik Aanæs. Large scale multiview stereopsis evaluation. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 406–413, 2014.

Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuhler, and George Drettakis. 3D gaussian splat-¨ ting for real-time radiance field rendering. ACM Transactions on Graphics, 42(4):1–14, 2023.

Arno Knapitsch, Jaesik Park, Qian-Yi Zhou, and Vladlen Koltun. Tanks and temples: Benchmarking large-scale scene reconstruction. ACM Transactions on Graphics, 36(4):78:1–78:13, 2017.

Jiahe Li, Jiawei Zhang, Xiao Bai, Jin Zheng, Xin Ning, Jun Zhou, and Lin Gu. DNGaussian: Optimizing sparse-view 3D gaussian radiance fields with global-local depth normalization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 20775– 20785, 2024a.

Jiahe Li, Jiawei Zhang, Youmin Zhang, Xiao Bai, Jin Zheng, Xiaohan Yu, and Lin Gu. GeoSVR: Taming sparse voxels for geometrically accurate surface reconstruction. In Advances in Neural Information Processing Systems, volume 38, 2025.

Jiahe Li, Jiawei Zhang, Xiao Bai, Jin Zheng, Xiaohan Yu, Lin Gu, and Gim Hee Lee. Revisiting photometric ambiguity for accurate gaussian-splatting surface reconstruction. In ICML, 2026.

Kunyi Li, Michael Niemeyer, Zeyu Chen, Nassir Navab, and Federico Tombari. MonoGSDF: Exploring monocular geometric cues for gaussian splatting-guided implicit surface reconstruction. arXiv preprint arXiv:2411.16898, 2024b.

Zhaoshuo Li, Thomas Muller, Alex Evans, Russell H. Taylor, Mathias Unberath, Ming-Yu Liu, and¨ Chen-Hsuan Lin. Neuralangelo: High-fidelity neural surface reconstruction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 8456–8465, 2023.

Lingjie Liu, Jiatao Gu, Kyaw Zaw Lin, Tat-Seng Chua, and Christian Theobalt. Neural sparse voxel fields. In Advances in Neural Information Processing Systems, volume 33, pp. 15651–15663, 2020.

Tao Lu, Mulin Yu, Linning Xu, Yuanbo Xiangli, Limin Wang, Dahua Lin, and Bo Dai. Scaffold-gs: Structured 3D gaussians for view-adaptive rendering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 20654–20664, 2024.

Xiaoyang Lyu, Yang-Tian Sun, Yi-Hua Huang, Xiuzhe Wu, Ziyi Yang, Yilun Chen, Jiangmiao Pang, and Xiaojuan Qi. 3DGSR: Implicit surface reconstruction with 3D gaussian splatting. ACM Transactions on Graphics, 43(6):1–12, 2024.

Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, and Ren Ng. NeRF: Representing scenes as neural radiance fields for view synthesis. Communications ofthe ACM, 65(1):99–106, 2021.

Thomas Muller, Alex Evans, Christoph Schied, and Alexander Keller. Instant neural graphics prim-¨ itives with a multiresolution hash encoding. ACM Transactions on Graphics, 41(4):1–15, 2022.

Michael Oechsle, Songyou Peng, and Andreas Geiger. UNISURF: Unifying neural implicit surfaces and radiance fields for multi-view reconstruction. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pp. 5589–5599, 2021.

Jeong Joon Park, Peter Florence, Julian Straub, Richard Newcombe, and Steven Lovegrove. DeepSDF: Learning continuous signed distance functions for shape representation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 165–174, 2019.

Kerui Ren, Lihan Jiang, Tao Lu, Mulin Yu, Linning Xu, Zhangkai Ni, and Bo Dai. Octree-gs: Towards consistent real-time rendering with LOD-structured 3D gaussians. IEEE Transactions on Pattern Analysis and Machine Intelligence, pp. 1–15, 2025. doi: 10.1109/TPAMI.2025.3568201.

Xinlin Ren, Chenjie Cao, Yanwei Fu, and Xiangyang Xue. Improving neural surface reconstruction with feature priors from multi-view images. In European Conference on Computer Vision, pp. 445–463. Springer, 2024.

Johannes L. Schonberger and Jan-Michael Frahm. Structure-from-motion revisited. In¨ Proceedings ofthe IEEE Conference on Computer Vision and Pattern Recognition, pp. 4104–4113, 2016.

Johannes L. Schonberger, Enliang Zheng, Marc Pollefeys, and Jan-Michael Frahm. Pixelwise view ¨ selection for unstructured multi-view stereo. In European Conference on Computer Vision, pp. 501–518. Springer, 2016.

Shuhan Shen. Accurate multiple view 3D reconstruction using patch-based stereo for large-scale scenes. IEEE Transactions on Image Processing, 22(5):1901–1914, 2013.

Cheng Sun, Min Sun, and Hwann-Tzong Chen. Direct voxel grid optimization: Super-fast convergence for radiance fields reconstruction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 5459–5469, 2022.

Cheng Sun, Jaesung Choe, Charles Loop, Wei-Chiu Ma, and Yu-Chiang Frank Wang. Sparse voxels rasterization: Real-time high-fidelity radiance field rendering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 16187–16196, 2025. doi: 10.1109/ CVPR52734.2025.01509.

Matias Turkulainen, Xuqian Ren, Iaroslav Melekhov, Otto Seiskari, Esa Rahtu, and Juho Kannala. DN-Splatter: Depth and normal priors for gaussian splatting and meshing. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pp. 2421–2431, 2025.

Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, and David Novotny. VGGT: Visual geometry grounded transformer. In´ Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 5294–5306, 2025. doi: 10.1109/ CVPR52734.2025.00499.

Jiepeng Wang, Peng Wang, Xiaoxiao Long, Christian Theobalt, Taku Komura, Lingjie Liu, and Wenping Wang. NeuRIS: Neural reconstruction of indoor scenes using normal priors. In European Conference on Computer Vision, pp. 139–155. Springer, 2022.

Peng Wang, Lingjie Liu, Yuan Liu, Christian Theobalt, Taku Komura, and Wenping Wang. NeuS: Learning neural implicit surfaces by volume rendering for multi-view reconstruction. In Advances in Neural Information Processing Systems, volume 34, pp. 27171–27183, 2021.

Yifan Wang, Di Huang, Weicai Ye, Guofeng Zhang, Wanli Ouyang, and Tong He. NeuRodin: A two-stage framework for high-fidelity neural surface reconstruction. In Advances in Neural Information Processing Systems, volume 37, 2024.

Yaniv Wolf, Amit Bracha, and Ron Kimmel. GS2Mesh: Surface reconstruction from gaussian splatting via novel stereo views. In European Conference on Computer Vision, pp. 207–224. Springer, 2024. doi: 10.1007/978-3-031-73024-5\ 13.

Qianyi Wu, Jianmin Zheng, and Jianfei Cai. Surface reconstruction from 3D gaussian splatting via local structural hints. In European Conference on Computer Vision, pp. 441–458. Springer, 2024.

Tong Wu, Jiaqi Wang, Xingang Pan, Xudong Xu, Christian Theobalt, Ziwei Liu, and Dahua Lin. Voxurf: Voxel-based efficient and accurate neural surface reconstruction. In International Conference on Learning Representations, 2023.

Baixin Xu, Jiangbei Hu, Jiaze Li, and Ying He. GSurf: 3D reconstruction via signed distance fields with direct gaussian supervision. arXiv preprint arXiv:2411.15723, 2024.

Lihe Yang, Bingyi Kang, Zilong Huang, Zhen Zhao, Xiaogang Xu, Jiashi Feng, and Hengshuang Zhao. Depth anything V2. In Advances in Neural Information Processing Systems, volume 37, pp. 21875–21911, 2024.

Yao Yao, Zixin Luo, Shiwei Li, Tian Fang, and Long Quan. MVSNet: Depth inference for unstructured multi-view stereo. In Proceedings of the European Conference on Computer Vision, pp. 767–783, 2018.

Lior Yariv, Yoni Kasten, Dror Moran, Meirav Galun, Matan Atzmon, Basri Ronen, and Yaron Lipman. Multiview neural surface reconstruction by disentangling geometry and appearance. In Advances in Neural Information Processing Systems, volume 33, pp. 2492–2502, 2020.

Lior Yariv, Jiatao Gu, Yoni Kasten, and Yaron Lipman. Volume rendering of neural implicit surfaces. In Advances in Neural Information Processing Systems, volume 34, pp. 4805–4815, 2021.

Alex Yu, Ruilong Li, Matthew Tancik, Hao Li, Ren Ng, and Angjoo Kanazawa. Plenoctrees for real-time rendering of neural radiance fields. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 5752–5761, 2021.

Mulin Yu, Tao Lu, Linning Xu, Lihan Jiang, Yuanbo Xiangli, and Bo Dai. GSDF: 3DGS meets SDF for improved neural rendering and reconstruction. In Advances in Neural Information Processing Systems, volume 37, pp. 129507–129530, 2024a.

Zehao Yu, Songyou Peng, Michael Niemeyer, Torsten Sattler, and Andreas Geiger. MonoSDF: Exploring monocular geometric cues for neural implicit surface reconstruction. In Advances in Neural Information Processing Systems, volume 35, pp. 25018–25032, 2022.

Zehao Yu, Anpei Chen, Binbin Huang, Torsten Sattler, and Andreas Geiger. Mip-splatting: Aliasfree 3D gaussian splatting. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 19447–19456, 2024b.

Zehao Yu, Torsten Sattler, and Andreas Geiger. Gaussian opacity fields: Efficient adaptive surface reconstruction in unbounded scenes. ACM Transactions on Graphics, 43(6):1–13, 2024c.

Wenyuan Zhang, Yu-Shen Liu, and Zhizhong Han. Neural signed distance function inference through splatting 3D gaussians pulled on zero-level set. In Advances in Neural Information Processing Systems, volume 37, pp. 101856–101879, 2024.

Enliang Zheng, Enrique Dunn, Vladimir Jojic, and Jan-Michael Frahm. PatchMatch based joint view selection and depthmap estimation. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 1510–1517, 2014.