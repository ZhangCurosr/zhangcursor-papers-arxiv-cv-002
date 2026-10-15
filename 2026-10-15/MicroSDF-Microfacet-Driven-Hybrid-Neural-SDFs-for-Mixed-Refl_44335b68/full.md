# MicroSDF: Microfacet-Driven Hybrid Neural SDFs for Mixed-Reflectance Surface Reconstruction

Lejia Ye , Yuhua Xu, Yulan Guo , Senior Member, IEEE, Lian Xu , Xiaohu Zhang

Abstract—Accurate 3D reconstruction in real-world environments remains a significant challenge due to the coexistence of reflective and non-reflective surfaces, which pose distinct modeling demands. Existing methods often treat these surface types separately, limiting their generalizability and physical plausibility. To bridge this gap, we propose MicroSDF, a novel neural implicit framework that facilitates geometry and reflectance modeling through microfacet theory. Our approach incorporates three core innovations: (i) a microfacet-guided geometry model that extracts multi-scale surface normals (macroscopic and microfacet) from a signed distance field (SDF), regularized by a proposed microfacet normal consistency loss to enforce physically plausible surface orientations; (ii) an enhanced dual-branch color model, where the specular branch leverages the microfacet normals to model highfrequency reflectance, and the vanilla branch, unlike prior works, uses reflection direction (instead of viewing direction) to better model diffuse and low-frequency specular components; and (iii) a detection-guided color blending strategy that adaptively fuses the color outputs based on reflection priors, providing more physically intuitive blending than implicitly learned blending weights. Combined with a tailored multi-stage optimization scheme, the proposed MicroSDF achieves robust and high-fidelity reconstruction across reflective and non-reflective surfaces. Extensive experiments on DTU, Shiny Blender, Ref-NeRF, and DeepVoxels datasets demonstrate state-of-the-art performance, establishing a new direction for physically grounded neural reconstruction.

Index Terms—Neural surface reconstruction, Surface rendering, Microfacet theory.

## I. INTRODUCTION

3 <sup>D</sup> <sup>surface</sup> <sup>reconstruction</sup> <sup>aims</sup> <sup>to</sup> <sup>recover</sup> <sup>the</sup> <sup>geometry</sup>of a scene from 2D images, enabling downstream tasks such as novel view synthesis and scene understanding. It is a fundamental task in computer vision [1], with widespread applications in robotics, autonomous navigation, and augmented reality. Real-world scenes often contain a mixture of reflective and non-reflective surfaces, coexisting within the same environment. However, most existing methods are designed to handle either diffuse (non-reflective) [2], [3], or highly reflective surfaces [4], independently. This limits their generalization in mixed scenes, reducing their robustness in practical applications. To achieve high-fidelity and reliable reconstruction in diverse real-world scenarios, a unified framework that adaptively models both surface types is essential.

![](images/6f9b61f34e5eb5a4793aafe34c16804da1014a35e865cd24e6fe868eb5722ffb.jpg)  
Fig. 1. Comparison between prior works and ours for implicit 3D reconstruction. (a) Prior dual-branch color modeling methods [5], [6]. (b) The proposed dual-branch color modeling method. Our approach differs from prior works in the following three aspects: (i) Using reflection direction d<sup>ˆ</sup> as input to the vanilla branch to better model diffuse and low-frequency specular reflections, (ii) leveraging microfacet normals as input to the specular branch to capture high-frequency specular highlights, and (iii) using detection-based weights for physically intuitive color blending. These designs lead to significantly better geometric reconstruction on complex reflective surfaces than prior works.

Current reconstruction paradigms face significant limitations when applied to reflective surfaces. For example, traditional Multi-View Stereo (MVS) methods rely on the assumption of multi-view consistency, which is commonly violated in reflective regions due to their strong view-dependent appearance. While recent neural implicit representations, especially neural Signed Distance Field (SDF) methods [3], [7], [8], have significantly advanced 3D reconstruction, their underlying rendering formulations often neglect the physical properties of highly reflective surfaces. Consequently, these methods struggle to disentangle view-dependent appearance from true geometry in reflective regions, leading to inaccurate reconstructions.

The difficulty of handling reflective surfaces is also encountered in inverse rendering, which aims to infer the scene’s geometry, material properties, and lighting from observed images. This is, however, fundamentally ill-posed: multiple combinations of material parameters, lighting, and shape can produce the same observed color, making decomposition from color supervision solely inherently ambiguous [4], [9], [10]. To address this, recent inverse rendering approaches [11]– [13] have incorporated physically grounded reflectance models, enabling more plausible and interpretable rendering results. While our goal differs, these advances raise a natural question: can physical principles be similarly integrated into neural implicit reconstruction frameworks to better handle reflective surfaces? Inspired by this, we turn to microfacet theory [14], a foundational model in graphics that captures surface reflectance behavior through statistical distributions of microscopic surface normals. A key insight from this theory is the distinction between reflective and non-reflective regions: specular regions exhibit small variance in microfacet normals, while non-specular regions show large variance.

Building upon this insight, we propose MicroSDF, a unified neural SDF framework that explicitly embeds microfacetinspired priors into the reconstruction process, for handling both reflective and non reflective surfaces. It inherits the coarse-to-fine learning paradigm of recent neural SDF methods [3], [4], [7], [8], [15]–[18] , with sequential geometry and color networks. More specifically, MicroSDF introduces three key innovations: (i) Microfacet-guided geometry modeling. We sample local surface normals from a finer-scale geometry hierarchy around each query point and concatenate them with the numerical normal at the current level. To ensure physically consistent reflectance behavior, we introduce a microfacet normal consistency loss that encourages alignment between the microfacet normals and local surface normal in reflective regions, enabling coherent reflection directions. (ii) Microfacet-driven color modeling. We employ a dual-branch color network, as illustrated in Fig. 1, in contrast to prior methods [5], [6], our specular branch incorporates microfacet normals as additional inputs to model high-frequency reflective behavior, and the vanilla branch uses the reflection direction as input to better capture diffuse and low-frequency reflective effects. This design ensures that these two branches focus on complementary aspects of color modeling. (iii) Detectionguided color blending. While the specular branch enables reflectance modeling, correctly identifying where to apply it remains challenging. Prior works [5] have used implicitly learned blending weights from an auxiliary MLP, which often fail to reliably identify specular highlights from true geometry. In contrast, we propose a detection-guided blending module that leverages per-pixel specular likelihoods from a pretrained detector (e.g., StableDelight [19]) to guide the fusion of color predictions from the two branches. This enables physically intuitive color blending, reducing artifacts and improving reconstruction robustness across mixed-material scenes.

Through these designs, the proposed MicroSDF achieves robust and high-fidelity reconstruction of reflective and nonreflective surfaces within a unified framework. The proposed method consistently improves performance in challenging reflective regions while preserving accuracy on standard scenes. In summary, our main contributions are fourfold:

1) We propose MicroSDF, the first neural implicit surface reconstruction framework that explicitly integrates microfacet theory in a principled manner. The proposed MicroSDF offers a unified solution to robustly handle complex real-world scenes with both reflective and nonreflective surfaces.

2) We propose a microfacet normal consistency loss that aligns microfacet and local surface normals in specular regions, enhancing physically meaningful geometry that supports accurate view-dependent appearance.

3) We propose an enhanced dual-branch color model that incorporates microfacet normals into the specular branch to capture high-frequency reflections, and uses reflection direction in the vanilla branch to model diffuse and lowfrequency reflections. Guided by a detection-based reflectance prior, this design enables robust and physically intuitive color blending.

4) The proposed MicroSDF achieves the state-of-the-art performance across diverse benchmarks, including DTU, Shiny Blender, Ref-NeRF, and DeepVoxels, showing the effectiveness on both reflective and non-reflective surfaces.

## II. RELATED WORKS

## A. Multi-view Stereo

Multi-view stereo (MVS) aims to reconstruct fine-grained scene geometry from multi-view images. Traditional MVS methods can be classified into four categories based on the output scene representation: volumetric-based methods, meshbased methods, point cloud-based methods, and depth mapbased methods [20]. Among them, depth map-based methods [21]–[24] are the most flexible, estimating depth maps for each view by utilizing photometric consistency between reference and neighboring images, then fusing all the depth maps into dense point clouds. A typical example is COLMAP [21], [22], which employs a robust patch-based matching strategy to estimate depth maps and then fuses them into a complete point cloud representation. Surface reconstruction methods, such as screened Poisson Surface Reconstruction [25], are then employed to reconstruct surfaces from the point clouds.

The advent of deep learning has transformed MVS by leveraging data-driven techniques for feature extraction and depth estimation. MVSNet [26] pioneered deep learning in MVS by using CNNs to construct 3D cost volumes and regress depth maps. Inspired by MVSNet, a series of outstanding works have emerged [27]–[31]. MVSNet++ [27] improved upon MVSNet by introducing feature pyramid structures and instance normalization, fusing multi-level information for accurate 3D point localization and generating smooth depth maps through tailored loss functions and a curriculum learning framework. Subsequent works like RA-MVSNet [28] and ET-MVSNet [29] enhanced accuracy through recurrent aggregation and epipolar constraints, while GeoMVSNet [32] incorporated geometric priors for better depth prediction. NR-MVSNet [30] further refined the coarse-to-fine strategy with a normal consistency-based depth hypotheses module and a depth refinement module with reliable attention, producing smoother and more accurate depth maps, particularly in texture-less and repetitive-texture regions. More recently, Transformer-based methods, such as MVSFormer [33], MVS-Former++ [34], and EI-MVSNet [31], have improved performance by modeling long-range view dependencies, with EI-MVSNet leveraging epipolar-guided cost volume construction and an interval-aware depth estimation module to enhance depth estimation accuracy. However, these deep learning-based MVS methods typically require large training datasets.

While the above traditional and learning-based MVS methods have shown promising results in general scenarios, they face fundamental limitations when dealing with reflective surfaces. The core photometric consistency assumption underlying these methods breaks down for glossy materials, making accurate reconstruction challenging.

## B. Neural Implicit Surface Based Reconstruction

In recent years, representation methods such as Neural Radiance Fields (NeRF) [35] and 3D Gaussian Splatting (3DGS) [36] have revolutionized traditional 3D reconstruction, advancing tasks such as novel view rendering [37], [38], human performance capture [39], human pose and shape estimation [40], object segmentation [41] and Multi-View 3D detection [42].

3D Gaussian Splatting (3DGS) [36] offers high rendering quality and efficiency and has been widely adopted for novel view synthesis. However, achieving accurate geometry remains challenging because it lacks an explicit surface representation, which makes reliable surface extraction and geometrydependent downstream tasks difficult. To improve geometric fidelity, several recent methods align Gaussian primitives with the underlying surface to enable surface extraction, including 2DGS [43], BBSplat [44], SuGaR [45], GOF [46] and Rade-GS [47]. However, their geometric accuracy in practice remains generally inferior to that of neural SDF based methods. Several other variants of 3D Gaussian splatting [48]– [50] have integrated domain-specific physical principles into the 3DGS framework for different objectives. For example, WaterSplatting [48] combines 3DGS with an underwater light transport model, explicitly accounting for absorption and scattering to enable fast underwater scene reconstruction. RainyGS [49] incorporates physically based raindrop and shallow-water simulations to synthesize dynamic rain effects such as water accumulation and surface waves in reconstructed scenes. LITA-GS [50] leverages Kubelka–Munk theory to introduce illumination-invariant physical priors, thereby facilitating illumination-agnostic novel view synthesis under challenging lighting conditions. In contrast to these 3DGS based methods that are tailored to specific tasks such as underwater reconstruction, rain synthesis and illuminationrobust rendering, our work focuses on surface reconstruction and geometric accuracy by adopting an implicit SDF representation and employing microfacet theory as a geometric regularizer, with the explicit goal of achieving high-fidelity surfaces and accurate geometry.

Native NeRF similarly faces geometric limitations: its reconstruction capabilities are constrained by inherent geometric-color ambiguity [18], [51] and the limitations of volumetric density representations, resulting in imprecise surface details. To address these limitations, subsequent research [3], [7], [8], [16]–[18], [52], [53] has explored explicit surface representations (e.g., SDF-based formulations) to achieve high-precision geometry in NeRF-based methods. Among them, VolSDF [16] introduced a significant advancement by incorporating SDF with a Laplace transform, establishing a mathematical relationship between SDF and volumetric density. This enables precise mesh extraction through

Marching Cubes [54]. Building upon this, NeuralAngelo [7] has further refined the reconstruction pipeline, achieving enhanced surface fidelity and overall reconstruction quality.

Numerous studies [18], [55]–[59] have demonstrated the potential for mutual enhancement through the integration of NeRF with traditional MVS approaches. Among these, several methods [18], [58], [59] have incorporated external geometric priors, such as depth maps, point clouds, and normal maps, to provide intermediate supervision for SDF-based networks, thereby improving geometric reconstruction quality. Notably, [18] identified a fundamental bias between rendered colors and geometry, and theoretically established that direct supervision of the zero-level set of SDF networks using point cloud information offers unbiased advantages and facilitates neural implicit surface learning. However, the effectiveness of these approaches heavily relies on the quality of external supervision. Both externally supervised geometric information and native NeRF techniques face challenges in reconstructing reflective surfaces, as NeRF’s view-dependent color prediction mechanism, while effective for diffuse materials, fails to accurately model the complex light interactions and viewdependent effects of reflective surfaces. Furthermore, external supervision often relies on multi-view consistency assumptions, which may not hold in real-world scenarios.

## C. Reflective Surface Reconstruction

Recent research in inverse rendering [13], [60]–[64] has increasingly embraced physics-based rendering models to address the challenge of reflectance interpretation. While these methods employ various basis functions to approximate the rendering equation, they primarily focus on improving rendering quality, without explicit investigation and evaluation of the geometry. Notably, Neural Microfacet Fields (NMF) [64] shares a similar high-level motivation with our work: both explicitly leverage microfacet-based physical modeling as priors to reduce the ambiguity of learning from images, particularly for reflective surfaces. However, NMF and our MicroSDF are designed for different primary tasks: NMF targets volumetric inverse rendering with explicit material–illumination decomposition, whereas MicroSDF focuses on high-precision geometric reconstruction. In terms of how microfacets are incorporated, NMF embeds microfacet modeling into a volumetric representation by assigning local normals and microfacet BRDF parameters to every volumetric sample along the ray, and performs surface-style Monte Carlo reflection integration to jointly recover materials and lighting and enable relighting, with geometry represented implicitly by a density field rather than an explicit SDF surface. In contrast, MicroSDF introduces microfacets only on the explicit SDF surface, defining microfacet normals at the surface that are used both as inputs to a specular branch to capture reflective appearance and as a consistency prior that regularizes surface orientation, thereby prioritizing accurate geometry recovery in scenes with mixed reflective and non-reflective regions without performing explicit material–illumination decomposition.

Ref-NeRF [65] achieved a notable breakthrough by introducing an innovative reparametrization technique that effectively disentangles reflectance from appearance. Nevertheless, its geometric reconstructions remain susceptible to the complex nature of view-dependent effects. Building upon the reparameterization strategy of Ref-NeRF, Ref-NeuS [4] integrated an anomaly detection module to effectively identify and handle specular regions. While this solution demonstrates effectiveness for reflective surfaces, it often underperforms when dealing with ordinary objects.

![](images/ff9ddb86b177998688f1594bc0d30b5dfeaeb7eff96bf41db25571a78aef2c52.jpg)  
Fig. 2. Overview of the proposed MicroSDF. It comprises a geometry network $\Theta _ { \mathrm { G } }$ and a color network $\Theta _ { \mathrm { { C } } } .$ . The geometry network samples spatial points x, encodes them via an instant Neural Graphics Primitive(iNGP) [66] to obtain features $F ,$ and passes these to the SDF MLP, which is initialized using the external point cloud $P .$ The color network Θ<sub>C</sub> consists of Vanilla MLP and Specular MLP, both receiving the same inputs: geometric features F , points x, reflection direction d<sup>ˆ</sup> (computed from normal n and viewing direction $^ { d ) , }$ , and normal n. The Specular MLP additionally takes microfacet normals m as input. The outputs from the two branches are fused via the proposed detection-guided blending module, where the blending weights are predicted by an additional specular detector, yielding the final color.

UniSDF [5] proposed a more universal solution by introducing a dual-branch architecture to handle both reflective and non-reflective regions. However, the core idea of UniSDF remains constrained by complex lighting phenomena in realworld scenarios, such as near-field reflection, which often leads to geometric reconstructions with significant artifacts. Through three modifications illustrated in Fig. 1, we significantly enhance its performance in such settings. More recently, AniSDF [6] attempted to develop a universal reconstruction method by leveraging a dual-granularity geometric network and physics-based encoding. In contrast, our approach maintains a single geometric network by sampling normals at different scales. Furthermore, our modifications adopt a “loosely coupled” design with existing neural implicit reconstruction paradigms, facilitating easier practical application.

## III. THE PROPOSED MICROSDF

In this section, we introduce the proposed MicroSDF, a unified neural surface reconstruction framework that integrates microfacet theory to jointly model geometry and reflectance for both reflective and non-reflective surfaces. An overview of the proposed framework is illustrated in Fig. 2. This section elaborates the overall design of our method, beginning with rendering preliminaries (Section III-A), followed by two core components: (i) microfacet-guided geometry modeling (Section III-B), which extracts physically consistent surface normals from a signed distance field, and (ii) microfacetdriven color modeling (Section III-C), which employs a dualbranch network with a proposed detection-guided blending process. We also describe our multi-stage optimization strategy (Section III-D), which enables progressive refinement and ensures physically faithful reconstruction.

## A. Preliminaries

1) Volume Rendering in Neural Radiance Fields: Neural Radiance Fields (NeRF) represent a scene by mapping a 3D position x and view direction d to density σ and color c via MLPs. For each camera ray, points are sampled along its path, and the final pixel color is computed by volume rendering:

$$
C = \sum _ { i } w _ { i } \pmb { c } _ { i } , \quad w _ { i } = T _ { i } \alpha _ { i } ,\tag{1}
$$

where $\alpha _ { i } = 1 - \exp ( - \sigma _ { i } \delta _ { i } )$ is the opacity, $\delta _ { i }$ is the interval between samples, and $\begin{array} { r } { T _ { i } = \prod _ { i = 1 } ^ { i - 1 } ( 1 - \alpha _ { j } ) } \end{array}$ is the accumulated transmittance. The network is trained by minimizing the color reconstruction loss:

$$
\mathcal { L } _ { \mathrm { c o l o r } } = \sum \| C - C _ { \mathrm { g t } } \| ^ { 2 } ,\tag{2}
$$

where $C _ { \mathrm { g t } }$ denotes the ground-truth pixel color.

2) Surface Rendering: The traditional rendering equation describes the outgoing radiance $L _ { o }$ from a surface point x in direction v:

$$
L _ { o } ( { \pmb x } , { \pmb v } ) = \int _ { \Omega } L _ { i } ( { \pmb x } , { \pmb v } , l ) f _ { r } ( { \pmb x } , { \pmb v } , l ) d l + L _ { e } ( { \pmb x } , { \pmb v } ) ,\tag{3}
$$

where $L _ { i } ( { \pmb x } , { \pmb v } , l )$ is the incoming radiance from direction l, $f _ { r } ( { \pmb x } , { \pmb v } , l )$ is the Bidirectional Reflectance Distribution Function (BRDF), $L _ { e } ( { \pmb x } , { \pmb v } )$ is the emitted light, and the integral is over the hemisphere Ω.

Cook-Torrance [67] model was established based on microfacet theory, assuming surfaces consist of tiny mirror-like facets. Its specular BRDF at point x is:

$$
f _ { \mathrm { c o o k - t o r r a n c e } } ( \pmb { x } , \omega _ { o } , \omega _ { i } ) = \frac { D ( \pmb { x } ) \cdot F ( \pmb { x } , \omega _ { o } ) \cdot G ( \pmb { x } , \omega _ { o } , \omega _ { i } ) } { 4 \cdot ( \omega _ { o } \cdot \pmb { n } ) \cdot ( \omega _ { i } \cdot \pmb { n } ) } ,\tag{4}
$$

where $D ( { \pmb x } )$ is the normal distribution function, $F ( \pmb { x } , \pmb { \omega } _ { o } )$ is the Fresnel function, $G ( \pmb { x } , \omega _ { o } , \omega _ { i } )$ is the geometry function, $\omega _ { o }$ and $\omega _ { i }$ are the observation and incident directions, and n is the surface normal at x.

## B. Microfacet-Guided Geometry Modeling

1) Geometry Network: The geometry network, parameterized by $\Theta _ { G } ,$ is designed for capturing the underlying 3D structure of the scene. $\Theta _ { G }$ takes as input the sampled 3D coordinates x and the external point cloud P, and outputs the corresponding surface normal vectors n, geometric features F, and SDF values. To encode the spatial properties of the scene, the 3D coordinates x are first fed into an iNGP, which produces geometric features F. These features are then further passed through an MLP to predict the SDF value. To facilitate effective training, especially in the early stages, the external point cloud P is used to directly supervise the SDF predictions. The resulting outputs provide essential geometric context for the subsequent color modeling stage.

2) Dual-Scale Normal Extraction: Accurately modeling surface reflectance, especially specular components, requires not only capturing the overall surface shape but also accounting for fine-grained normal variations. To this end, we propose to extract dual-scale normals that compute both macroscopic and microfacet normals for each sampled 3D point. This dualscale formulation enables our model to capture both global surface structure and fine-grained normal variations, which are critical for consistent geometry reconstruction and physically plausible specular reflection modeling.

Given a sampled 3D point x<sub>i</sub>, two types of normals are computed as follows: (i) Macroscopic normals n are estimated via central differences along coordinate axes with a decreasing step size ϵ during training. For example, the $\mathbf { X } ^ { - }$ directional gradient is computed as:

$$
\nabla _ { \boldsymbol { x } } f ( \mathbf { x } _ { i } ) = \frac { \Phi ( \mathbf { x } _ { i } + \epsilon _ { \pmb { x } } ) - \Phi ( \mathbf { x } _ { i } - \epsilon _ { \pmb { x } } ) } { 2 \epsilon } ,\tag{5}
$$

where $\epsilon _ { x } ~ = ~ [ \epsilon , 0 , 0 ]$ , and $\Phi ( { \pmb x } )$ is the SDF learned by the geometry network, representing the distance from point x to the surface. (ii) Microfacet normals $m _ { i }$ are computed at a finer scale (eg. level $L + 4$ when training at level L), using the first-order derivative of $\Phi ( { \pmb x } )$ as an analytical solution. The sampled macroscopic normal n and microfacet normals $\mathbf { \nabla } m _ { i }$ are concatenated to form a multi-scale normal representation, which is later used to predict the specular color component in our color modeling network (Section III-C1).

![](images/34b4317fbdc582e35f3f9512799ca199d6e80af0e4bf2963a6178bd566b4010a.jpg)  
Fig. 3. Microfacet normal characteristics. $\pmb { p _ { 1 } }$ and $\mathbf { \Delta } \mathbf { p _ { 2 } }$ illustrate representative points on specular and diffuse surfaces, respectively. According to microfacet theory, the normal variance at specular surfaces (p<sub>1</sub>) is smaller than that at diffuse surfaces (p<sub>2</sub>).

This hierarchical sampling strategy is grounded in microfacet theory, where surfaces are modeled as collections of tiny mirror-like facets. By leveraging this representation, we can better capture the directional behavior of specular reflections. Note that, unlike AniSDF [6], which requires maintaining two separate geometric networks at different granularities, our approach achieves multi-scale learning within a single geometry network, enabling more efficient and physically consistent training.

3) Microfacet Normal Consistency: As illustrated in Fig. 3, specular regions $( e . g . , p _ { 1 } )$ tend to exhibit low variance among microfacet normals, while rough regions $( e . g . , \ p _ { 2 } )$ show higher variance, consistent with classical microfacet theory. To enforce alignment between macroscopic and microscopic surface orientations, we define a consistency metric based on the angular variance:

$$
\tau ^ { 2 } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \left( 1 - \pmb { n } \cdot \pmb { m } _ { i } \right) ,\tag{6}
$$

where N denotes the number of microfacet normals. This measures the average angular discrepancy between the coarse surface normal n and each microfacet normal $m _ { i } .$ The microfacet consistency loss is then defined as:

$$
\mathcal { L } _ { \mathrm { m i c r o f a c e t } } = \frac { 1 } { M } \sum _ { i = 1 } ^ { M } \left( \tau _ { i } ^ { 2 } \cdot m a s k _ { i } \cdot s p e c _ { i } \right) ,\tag{7}
$$

where M denotes the number of sampled points, mask<sub>i</sub> indicates valid surface regions, spec<sub>i</sub> is the specular score from the detection network (Section III-C2). This loss enforces the physically grounded constraint that specular regions should exhibit consistent microfacet orientations, thereby regularizing the geometry network to produce normals that are directionally coherent with specular reflection behavior.

## C. Microfacet-Driven Color Modeling

1) Dual-branch color network: We decompose the color modeling process into two branches: (i) Vanilla Branch: Following Ref-NeRF [65], we first compute the reflection

direction $\hat { \ b { d } }$ from the normal vector n and the ray direction $\mathbf { \delta } d .$ The Vanilla MLP takes as inputs x, d<sup>ˆ</sup>, n, and $F ,$ , and outputs the vanilla color $C _ { v a n i l l a }$ , formulated as:

$$
\begin{array} { r } { C _ { v a n i l l a } = \mathrm { M L P } _ { \mathrm { v a n i l l a } } ( \pmb { x } , \hat { \pmb { d } } , \pmb { n } , \pmb { F } ) . } \end{array}\tag{8}
$$

As evidenced by Ref-NeRF [65], the reflection direction $\hat { \ b { d } }$ is adequate for modeling most real-world scenarios, allowing the vanilla branch to absorb low-frequency specular cues and reducing the burden on the specular branch. Our ablation studies (see Fig. 8 and Table IV) further demonstrate the advantages of this design. (ii) Specular Branch: To more accurately model specular reflections, we augment the input of the Specular MLP with microfacet normals m, sampled as described in Section III-B2. This branch also receives (x, $\hat { \boldsymbol { d } } ,$ n, and F), leveraging both macroscopic and fine-grained geometric cues, and outputs the specular color component for each sampled point, formulated as:

$$
C _ { \mathrm { s p e c u l a r } } = \mathrm { M L P } _ { \mathrm { s p e c u l a r } } ( \boldsymbol { x } , \hat { \boldsymbol { d } } , n , m , \boldsymbol { F } ) .\tag{9}
$$

Introducing microfacet normals m enables us to empirically estimate $D ( \pmb { x } )$ in Eq. 4, improving the network’s ability to reconstruct specular regions in a physically grounded manner.

2) Detection-guided blending: Handling reflective and nonreflective surfaces within a unified framework requires not only accurate modeling of each component but also precise spatial allocation of where these components should dominate. A key challenge in dual-branch color networks lies in ambiguity when decomposing color into low-frequency and high-frequency specular components.

To address this, we propose a detection-guided color blending mechanism that leverages external, physically interpretable priors to control the blending between the vanilla branch (responsible for low-frequency specular effects) and the specular branch (capturing high-frequency specular highlights). Unlike UniSDF [5] and AniSDF [6], which infer blending weights via an auxiliary MLP, making the fusion process susceptible to dataset bias and prone to misidentifying specular cues, we employ StableDelight [19], a diffusion-based specular detector, to per-pixel specular confidence maps directly from the input images. These detection scores encode the likelihood of each pixel exhibiting high-frequency specular reflection, thus providing explicit, spatially varying blending weights grounded in physical intuition. The final pixel color is computed as:

$$
C = ( 1 - \mathrm { S p e c } ) \cdot C _ { v a n i l l a } + \mathrm { S p e c } \cdot C _ { s p e c u l a r } .\tag{10}
$$

where Spec denotes specular score.

## D. Optimization

Our optimization process consists of three key components: geometry initialization, a multi-stage training strategy, and our loss function design.

1) Geometry Initialization: To provide a reliable starting point for the geometry network $\Theta _ { G }$ , we initialize the SDF $\Phi ( { \pmb x } )$ using dense point cloud information obtained from traditional 3D reconstruction algorithms (e.g., OpenMVS [68]). These point clouds help constrain surface positions and identify empty regions in 3D space.

Inspired by the regularization approach in SIREN [69], we encourage the SDF values of surface points to approach 0. However, unlike SIREN [69], which incorporates normal constraints, our microfacet-based rendering method is particularly sensitive to inaccuracies in surface normals. To avoid the detrimental effects of the initial normal errors, we exclude normal constraints during the geometry initialization stage. Besides, we randomly sample off-surface points and enforce that their SDF gradients are close to zero. This encourages flatness in empty regions and contributes to a more stable and robust initialization. The specific loss function is defined as:

$$
\begin{array} { l } { \displaystyle \mathcal { L } _ { \mathrm { s d f } } = \int _ { \Omega } \left. \left. \nabla _ { x } \Phi ( x ) \right. \right. - 1 \vert ~ d  { \boldsymbol { x } } } \\ { \displaystyle ~ + \int _ { \Omega _ { 0 } } \left. \left. \Phi (  { \boldsymbol { { x } } } ) \right. \right. ~ d  { \boldsymbol { x } } + \int _ { \Omega \backslash \Omega _ { 0 } } \psi \left( \Phi (  { \boldsymbol { { x } } } ) \right) ~ d  { \boldsymbol { x } } , } \end{array}\tag{11}
$$

where Ω denotes the entire 3D space, $\Omega _ { 0 }$ is the surface region of the point cloud, and ψ() is a penalty function applied to off-surface SDF values. The first term ensures that the SDF satisfies the eikonal equation, the second term enforces consistency of surface positions, and the third term penalizes deviations in off-surface points.

To further distinguish occupied from empty regions, we utilize foreground masks of input images by leveraging established fast matting algorithms, such as [70]. We enforce the accumulated weight of sampled points along each background ray path to be minimized. This mask loss is defined as:

$$
\mathcal { L } _ { \mathrm { m a s k } } = \left. \boldsymbol { G } \boldsymbol { T } _ { \mathrm { m a s k } } - \int _ { t _ { n } } ^ { t _ { f } } \boldsymbol { T } ( t ) \cdot \boldsymbol { \sigma } ( t ) d t \right. ^ { 2 } ,\tag{12}
$$

where $G T _ { \mathrm { m a s k } }$ is the ground-truth mask extracted from the input images, $\pmb { T } ( t )$ and $\sigma ( t )$ are the transmittance and density along the ray t, respectively, and $t _ { n }$ and $t _ { f }$ represent the near and far boundaries of the ray.

2) Four-stage Training Strategy: Although the effectiveness of coarse-to-fine training has been well demonstrated, our experiments show that, with sufficient training at the initial level, certain intermediate levels can be skipped. This enables a more direct transition to the finest level, where analytical gradients can replace numerical ones for improved efficiency. Based on this insight, we propose a four-stage training strategy comprising: (i) geometry initialization (Geometry Init.), (ii) thorough training at the first level (1st Level), (iii) progressive coarse-to-fine processing (Coarse-to-Fine), and (iv) final training at the finest level (Finest Level).

This training strategy significantly reduces the total number of training iterations compared to conventional coarse-to-fine training approaches, while yielding superior quantitative results (see Table VI in Supplementary Material). Our microfacet sampling process is mainly performed during the thorough training at the first level and the progressive coarse-to-fine processing stages.

3) Loss function design: Our loss function design is closely aligned with the four-stage training strategy, with specific loss terms activated at different training stages. A summary of the loss terms at each stage is provided in Table I.

TABLE I  
LOSS TERMS USED AT DIFFERENT TRAINING STAGES.
<table><tr><td></td><td>Geometry Init. 1st Level Coarse-to-Fine Finest Level</td><td></td><td></td><td></td></tr><tr><td> $\mathcal { L } _ { \mathrm { s d f } }$ </td><td>√</td><td></td><td></td><td></td></tr><tr><td> $\mathcal { L } _ { \mathrm { m a s k } }$ </td><td>√</td><td></td><td></td><td></td></tr><tr><td> $\scriptstyle { \mathcal { L } } _ { \mathrm { c o l o r } }$ </td><td></td><td>√</td><td>√</td><td>√</td></tr><tr><td> $\mathcal { L } _ { \mathrm { e i k } }$ </td><td></td><td>√</td><td>√</td><td>√</td></tr><tr><td> ${ \mathcal { L } } _ { \mathrm { c u r v } }$ </td><td></td><td>√</td><td>√</td><td></td></tr><tr><td>Lmicrofacet</td><td></td><td>√</td><td>√</td><td></td></tr></table>

As the geometry initialization stage (detailed in Section III-D1) focuses on constraining surface geometry and identifying empty space, we define a composite loss function:

$$
{ \mathcal { L } } _ { \mathrm { G e o m e t r y ~ I n i t . } } = \lambda _ { \mathrm { s d f } } { \mathcal { L } } _ { \mathrm { s d f } } + \lambda _ { \mathrm { m a s k } } { \mathcal { L } } _ { \mathrm { m a s k } }\tag{13}
$$

where $\lambda _ { \mathrm { s d f } }$ and $\lambda _ { \mathrm { m a s k } }$ are weighting coefficients balancing geometric fidelity and empty region constraints. In the 1st Level and Coarse-to-Fine stages, the primary objective shifts to jointly refining geometry and color. The core loss components include: (i) Color loss: This loss minimizes the discrepancy between the rendered color and the input image pixel color:

$$
\mathcal { L } _ { \mathrm { c o l o r } } = \sum \left. C - C _ { \mathrm { g t } } \right. ^ { 2 }\tag{14}
$$

where $C$ denotes the rendered color, and $C _ { \mathrm { g t } }$ represents the ground-truth color. (ii) Eikonal loss [71]: To maintain the validity of the SDF features, we enforce the Eikonal constraint:

$$
\mathcal { L } _ { \mathrm { { e i k } } } = \sum \left( \| \nabla \Phi ( \pmb { x } ) \| - 1 \right) ^ { 2 }\tag{15}
$$

(iii) Curvature loss: Following Neuralangelo [7], we introduce a curvature regularization based on the second-order derivatives of the SDF, which promotes smooth surface reconstructions:

$$
\mathcal { L } _ { \mathrm { c u r v } } = \sum \left( \left| \left| \nabla ^ { 2 } \Phi ( x _ { i } ) \right| \right| \right)\tag{16}
$$

where $\nabla ^ { 2 } \Phi ( { \pmb x } _ { i } )$ denotes the Laplacian of the SDF value at the sampled point ${ \mathbf { } } x _ { i } ,$ , reflecting local curvature. (iv) Microfacet consistency loss: To enhance the fidelity of reflectance modeling, we introduce a microfacet consistency loss that aligns predicted surface orientations with the microfacet-based reflectance model (Detailed in Eq. 7 in Section III-B3). The combined loss during the second and third stages is:

$$
{ \begin{array} { r l } { { \mathcal { L } } _ { \mathrm { 1 s t ~ L e v e l ~ \& ~ C o a r s e \mathrm { - } t o - F i n e } } = { \mathcal { L } } _ { \mathrm { c o l o r } } + \lambda _ { \mathrm { e i k } } { \mathcal { L } } _ { \mathrm { e i k } } + \lambda _ { \mathrm { c u r v } } { \mathcal { L } } _ { \mathrm { c u r v } } } & { } \\ { + \lambda _ { \mathrm { m i c r o f a c e t } } { \mathcal { L } } _ { \mathrm { m i c r o f a c e t } } \quad } & { ( 1 7 ) } \end{array} }
$$

where $\lambda _ { \mathrm { e i k } } , \lambda _ { \mathrm { c u r v } }$ , and $\lambda _ { \mathrm { { m i c r o f a c e t } } }$ are weighting factors of their respective loss terms. In the final stage, analytical gradients are used, making curvature and microfacet losses unnecessary. The objective loss function is:

$$
{ \mathcal { L } } _ { \mathrm { F i n e s t \ L e v e l } } = { \mathcal { L } } _ { \mathrm { c o l o r } } + \lambda _ { \mathrm { e i k } } { \mathcal { L } } _ { \mathrm { e i k } } .\tag{18}
$$

## IV. EXPERIMENTS

## A. Implementation Settings

Datasets. Following prior works [3], [5], [7], [10], [16], we evaluate the proposed method on three widely used benchmarks: (i) DTU [75]: The DTU multi-view dataset contains

15 real-world scenes, each captured from 49 or 64 viewpoints with high-resolution images. Each scene contains objects with diverse materials and complex geometries, including both reflectance and non-reflectance surfaces. DTU provides ground-truth 3D models and camera trajectories, making it a standard benchmark for quantitative evaluation of point cloud reconstruction.(ii) Shiny Blender [65]: The Shiny Blender dataset is a synthetic benchmark consisting of 6 reflective objects rendered under controlled settings. We use it to provide additional quantitative evaluation on reflective surfaces. (iii) Ref-NeRF [65]: The Ref-NeRF dataset is designed to evaluate reconstruction quality for reflective objects in uncontrolled real-world scenarios. It includes three challenging scenes, $i . e . ,$ , “sedan”, “garden spheres”, and “toycar”, each featuring multi-view images of highly reflective surfaces. Ref-NeRF specifically tests an algorithm’s robustness to specular reflections. (iv) DeepVoxels [76]: To evaluate performance in more realistic settings, we select two scenes (a glossy sphere and a coffee cup) from this dataset. Both scenes were captured with handheld cameras and processed via COLMAP [21], [22], featuring reflective surfaces under non-ideal conditions.

Evaluation Metrics. For scenes with available ground-truth geometry (e.g., DTU dataset), we employ the Chamfer Distance (CD) as the primary quantitative metric to assess the accuracy of the reconstructed point cloud. Following the official evaluation protocol, CD is defined as:

$$
C D _ { ( { \pmb A } , { \pmb B } ) } = \frac { 1 } { | { \pmb A } | } \sum _ { { \pmb a } \in { \pmb A } } \operatorname* { m i n } _ { { \pmb b } \in { \pmb B } } \| { \pmb a } - { \pmb b } \| _ { 2 } ^ { 2 } + \frac { 1 } { | { \pmb B } | } \sum _ { { \pmb b } \in { \pmb B } } \operatorname* { m i n } _ { { \pmb a } \in { \pmb A } } \| { \pmb b } - { \pmb a } \| _ { 2 } ^ { 2 } ,\tag{19}
$$

where A and B are the predicted and ground-truth point sets, respectively. The first term measures how close the prediction is to the ground truth (accuracy), and the second term measures coverage of the ground truth by the prediction (completeness). Lower CD values indicate higher geometry fidelity and are reported in millimeters for DTU following the official benchmark. For the Shiny Blender dataset, we follow UniSDF [5] and Ref-NeuS [4] to report mean angular error (MAE) of normals and accuracy (ACC) of mesh. The ground-truth meshes are double-layered; the inner layer is not visible in multi-view images and cannot be reconstructed, so the completeness term of the Chamfer Distance would be biased by these invisible points. We therefore report only ACC, which corresponds to the accuracy term (first term) in the CD definition above and reflects how well the predicted surface aligns with the visible ground truth (lower is better). We also evaluate surface orientation with the MAE between rendered normals and ground-truth normals over the valid foreground mask. Consistent with Ref-NeuS, ACC is not reported for the ball and teapot scenes. For datasets lacking ground-truth geometry (e.g., Ref-NeRF and DeepVoxels), we conduct a qualitative evaluation. We focus on the preservation of fine geometric details, especially for reflective surfaces, such as the accurate reconstruction of depressions, as well as the overall reconstruction plausibility in both reflective and non-reflective areas.

Implementation Details. We implement our algorithm in PyTorch. For efficiency, we adopt the approach from [70] to obtain image mask information. The training process follows our four-stage training strategy described in Section III-D2, with the following iteration schedule: we start with geometric initialization for 2,500 iterations. This is followed by the first level of training for 5,000 iterations, then a coarse-tofine stage for 10,000 iterations divided into 4 levels with 2,500 iterations per level, and finally, the finest level training for 20,000 iterations. In total, the training comprises 37,500 iterations. All experiments are conducted on an NVIDIA A100 GPU (40GB), taking approximately 40 minutes for training (excluding pose estimation and data preprocessing time). After training, we extract the final model using the Marching Cubes algorithm [54] with a resolution of 1024.

![](images/73473e0f32dc29f981d6fb2786403747a717a557d1f76648f9e819b0064dc43c.jpg)  
Scan 40  
Scan 69

![](images/d3e185890d4a9d1585e6ccdb61673ebde41f0c7cc84476f01a04ffd541802450.jpg)

![](images/67d0e1f2756bae33784dd409376430d44784c8286e3b297cb43869b1f8ca1f23.jpg)

![](images/e97e78021b6dfcb63d7030d44f96da112b96c871b0093fa4da3af32935fe24f8.jpg)

![](images/e1f42406fd2746937171569d98fdbfbc5eb567327c287c7e7b1e35259c95cb13.jpg)

![](images/093764f1e76f9fa1fea1f55da886de255841eded584672a8044d08deb06a43e6.jpg)

![](images/c83df04d852cd00af5dca00bf0a8973a03addbde34c494ada1ef658b34a66921.jpg)

![](images/088924b91c0318455135275e570eb1aae846162a12e925856e096924dd7674c3.jpg)  
Scan 65

![](images/2fc0803d9709600a2e1ded2404ff07865dce7405267dd301350044ebab8ccecb.jpg)

![](images/fbbefb6e6a24702d54b5ae4f35cf977e8b2c93cedbbc6a7fee85925b9e8ab88d.jpg)  
Scan 118

![](images/6c5e207ba755da2b106e22b7583cb251690d5f0c37a5a9fef73715bfd5906356.jpg)  
Reference Image

![](images/53b4716b81cebfbfeb88cb120e1d6d365eff3d71da95e807aab196eabbfc5d2c.jpg)  
Ours (Normal)

![](images/72ff92a7e8edbdff6499f5908796a0c7d77094be23e1d5ac76714ad232a2412a.jpg)  
Ours

![](images/c79c7bf76589a71d1ed5a96dc84ae64856844647b1a7d112eb86eb18fb6d987a.jpg)  
NeuS  
UniSDF

Fig. 4. Qualitative comparison results on the DTU benchmark. Our approach achieves reconstruction that preserves finer geometric details compared to the state-of-the-art neural implicit method, NeuS [3], while significantly outperforming the general-purpose reconstruction method UniSDF [5].  
TABLE II  
RECONSTRUCTION PERFORMANCE ON THE DTU DATASET IN TERMS OF CHAMFER DISTANCE(MM) ↓ (LOWER IS BETTER). RED AND ORANGE DENOTE THE BEST AND THE SECOND-BEST RESULTS, RESPECTIVELY. RESULTS MARKED WITH \* ARE SOURCED FROM THE UNISDF [5] PAPER, WHILE OTHERS ARE REPORTED FROM THEIR RESPECTIVE PAPERS.
<table><tr><td>Methods</td><td>24</td><td>37</td><td>40</td><td>55</td><td>63</td><td>65</td><td>69</td><td>83</td><td>97</td><td>105</td><td>106</td><td>110</td><td>114</td><td>118</td><td>122</td><td>Mean</td><td>Runtime</td></tr><tr><td>VolSDF* [16]</td><td>1.14</td><td>1.26</td><td>0.81</td><td>0.49</td><td>1.25</td><td>0.70</td><td>0.72</td><td>1.29</td><td>1.18</td><td>0.70</td><td>0.66</td><td>1.08</td><td>0.42</td><td>0.61</td><td>0.55</td><td>0.86</td><td>≈8h</td></tr><tr><td>NeuS* [3]</td><td>1.37</td><td>1.21</td><td>0.73</td><td>0.40</td><td>1.20</td><td>0.70</td><td>0.72</td><td>1.01</td><td>1.16</td><td>0.82</td><td>0.66</td><td>1.69</td><td>0.39</td><td>0.49</td><td>0.51</td><td>0.87</td><td>≈8h</td></tr><tr><td>NeuralWarp* [72]</td><td>0.49</td><td>0.71</td><td>0.38</td><td>0.38</td><td>0.79</td><td>0.81</td><td>0.82</td><td>1.20</td><td>1.06</td><td>0.68</td><td>0.66</td><td>0.74</td><td>0.41</td><td>0.63</td><td>0.51</td><td>0.68</td><td>≈8h</td></tr><tr><td>Neuralangelo* [7]</td><td>0.49</td><td>1.05</td><td>0.95</td><td>0.38</td><td>1.22</td><td>1.10</td><td>2.16</td><td>1.68</td><td>1.78</td><td>0.93</td><td>0.44</td><td>1.46</td><td>0.41</td><td>1.13</td><td>0.97</td><td>1.08</td><td>≈16h</td></tr><tr><td>PermutoSDF* [73]</td><td>0.52</td><td>0.75</td><td>0.41</td><td>0.37</td><td>0.90</td><td>0.66</td><td>0.59</td><td>1.37</td><td>1.07</td><td>0.85</td><td>0.46</td><td>0.98</td><td>0.33</td><td>0.39</td><td>0.50</td><td>0.68</td><td>≈1h</td></tr><tr><td>UniSDF* [5]</td><td>0.54</td><td>0.84</td><td>0.66</td><td>0.51</td><td>0.76</td><td>0.64</td><td>0.71</td><td>0.70</td><td>0.86</td><td>0.57</td><td>0.69</td><td>0.65</td><td>0.45</td><td>0.56</td><td>0.50</td><td>0.64</td><td>≈1h</td></tr><tr><td>MonoSDF [59]</td><td>0.66</td><td>0.88</td><td>0.43</td><td>0.40</td><td>0.87</td><td>0.78</td><td>0.81</td><td>1.23</td><td>1.18</td><td>0.66</td><td>0.66</td><td>0.96</td><td>0.41</td><td>0.57</td><td>0.51</td><td>0.73</td><td>≈8h</td></tr><tr><td>AniSDF [6]</td><td>0.52</td><td>0.82</td><td>0.65</td><td>0.43</td><td>0.76</td><td>0.64</td><td>0.71</td><td>0.97</td><td>0.86</td><td>0.64</td><td>0.52</td><td>0.67</td><td>0.42</td><td>0.67</td><td>0.50</td><td>0.65</td><td>≈1h</td></tr><tr><td>GSDF [74]</td><td>0.59</td><td>0.94</td><td>0.46</td><td>0.38</td><td>1.30</td><td>0.77</td><td>0.73</td><td>1.59</td><td>1.29</td><td>0.76</td><td>0.59</td><td>1.22</td><td>0.38</td><td>0.52</td><td>0.51</td><td>0.80</td><td>≈8h</td></tr><tr><td>SuGaR [45]</td><td>1.47</td><td>1.33</td><td>1.13</td><td>0.61</td><td>2.25</td><td>1.71</td><td>1.15</td><td>1.63</td><td>1.62</td><td>1.07</td><td>0.79</td><td>2.45</td><td>0.98</td><td>0.88</td><td>0.79</td><td>1.33</td><td>≈1h</td></tr><tr><td>2DGS [43]</td><td>0.48</td><td>0.91</td><td>0.39</td><td>0.39</td><td>1.01</td><td>0.83</td><td>0.81</td><td>1.36</td><td>1.27</td><td>0.76</td><td>0.70</td><td>1.40</td><td>0.40</td><td>0.76</td><td>0.52</td><td>0.80</td><td>≈ 0.5h</td></tr><tr><td>GOF [46]</td><td>0.50</td><td>0.82</td><td>0.37</td><td>0.37</td><td>1.12</td><td>0.74</td><td>0.73</td><td>1.18</td><td>1.29</td><td>0.68</td><td>0.77</td><td>0.90</td><td>0.42</td><td>0.66</td><td>0.49</td><td>0.74</td><td>≈1h</td></tr><tr><td>Rade-GS [47]</td><td>0.46</td><td>0.73</td><td>0.33</td><td>0.38</td><td>0.79</td><td>0.75</td><td>0.76</td><td>1.19</td><td>1.22</td><td>0.62</td><td>0.70</td><td>0.78</td><td>0.36</td><td>0.68</td><td>0.47</td><td>0.68</td><td>≈ 0.5h</td></tr><tr><td>Ours</td><td>0.53</td><td>0.64</td><td>0.31</td><td>0.44</td><td>0.88</td><td>0.62</td><td>0.56</td><td>1.34</td><td>1.00</td><td>0.52</td><td>0.48</td><td>0.66</td><td>0.39</td><td>0.39</td><td>0.40</td><td>0.61</td><td>≈1h</td></tr></table>

Sampling Strategy. Following the sampling strategy proposed in [77], we employ two rounds of proposal sampling and one NeRF sampling, with sample counts of 256, 96, and 24, respectively. More specifically, the final rendering is determined by the 24 sampling points from the NeRF sampling. Notably, to focus on object-level reconstruction, we do not perform sampling along the ray path from its origin until it intersects with the axis-aligned bounding box.

## B. Comparison with the State-of-the-Art

DTU Dataset. As shown in Table II, our proposed method achieves the best performance on most scenes and the best average Chamfer distance among all compared methods, demon-

AniSDF

Ours

![](images/f4941d18a9b9e57958f7d69e5c13a38785bde4cd60b7183d5e7c899c5d81d1d7.jpg)  
Reference Image  
Ref-NeuS  
UniSDF

Fig. 5. Qualitative comparison of the reflective surface reconstruction between the state-of-the-art methods and ours on the DeepVoxels (top two scenes) and Ref-NeRF (bottom three scenes) datasets. Our method outperforms UniSDF [5] and AniSDF [6] with higher success rates and better geometric fidelity.

strating strong generalization across scenes with both reflective and non-reflective surfaces. The comparison comprises neural implicit methods (e.g., NeuS [3], UniSDF [5]) and recent 3D Gaussian Splatting based methods (SuGaR [45], 2DGS [43], GOF [46], Rade-GS [47], GSDF [74]). Optimization time in the table ranges from ∼0.5 h to 16 h per scene: several 3DGS methods require less time (e.g., ∼0.5 h), while some neural implicit methods require over 16 h. Our method takes ∼1 h per scene and attains the best mean Chamfer distance, offering a favorable trade-off between efficiency and accuracy. Fig. 4 presents qualitative comparisons, which further illustrate the superiority of our reconstructions. Our method consistently recovers finer geometry details, such as the facial features of the DTU scan118 statue, more accurately than NeuS [3] (which requires over ten hours of optimization) and significantly surpasses UniSDF [5], a method designed for general object reconstruction.

Shiny Blender Dataset. We compare our method with recent rendering-oriented work (Ref-NeRF [65]), inverse rendering (NMF [64]), conventional neural implicit methods (Geo-NeuS [18], Neuralangelo [7]), and methods designed for reflective surfaces (Ref-NeuS [4], UniSDF [5]). As shown in Table III, our method achieves the best mean MAE (4.58<sup>◦</sup>) and the best mean ACC (0.61) among compared methods. This demonstrates the advantages of the proposed microfacetregularized SDF representation for high-precision geometry reconstruction in reflective scenes.

TABLE III  
QUANTITATIVE RESULTS ON SHINY BLENDER DATASET. LOWER IS BETTER FOR BOTH MAE<sup>◦</sup> AND ACC. RED AND ORANGE DENOTE THE BEST AND THE SECOND-BEST RESULTS, RESPECTIVELY. RESULTS MARKED WITH \* ARE SOURCED FROM THE UNISDF [5] PAPER, WHILE OTHERS ARE REPORTED FROM THEIR RESPECTIVE PAPERS.
<table><tr><td rowspan="2">Method</td><td colspan="7"> $\mathbf { M A E } ^ { \circ } \downarrow$ </td><td colspan="5">ACC↓</td></tr><tr><td>car</td><td>ball</td><td>helmet</td><td>teapot</td><td>toaster</td><td>coffee</td><td>mean</td><td>car</td><td>helmet</td><td>toaster</td><td>coffee</td><td>mean</td></tr><tr><td>Geo-NeuS* [18]</td><td>12.15</td><td>1.04</td><td>4.12</td><td>19.77</td><td>16.23</td><td>9.79</td><td>10.52</td><td>0.72</td><td>0.74</td><td>4.14</td><td>0.90</td><td>1.63</td></tr><tr><td>Neuralangelo*[7]</td><td>12.34</td><td>35.63</td><td>9.23</td><td>4.98</td><td>14.14</td><td>8.61</td><td>14.16</td><td>1.89</td><td>1.71</td><td>2.99</td><td>0.63</td><td>1.81</td></tr><tr><td>Ref-NeRF*[65]</td><td>14.93</td><td>1.55</td><td>29.48</td><td>9.23</td><td>42.87</td><td>12.24</td><td>18.38</td><td></td><td></td><td></td><td>一</td><td></td></tr><tr><td>NMF [64]</td><td>7.74</td><td>0.72</td><td>2.40</td><td>5.67</td><td>6.66</td><td>13.17</td><td>6.06</td><td></td><td></td><td></td><td>一</td><td></td></tr><tr><td>Ref-NeuS* [4]</td><td>7.68</td><td>0.50</td><td>1.94</td><td>10.25</td><td>5.95</td><td>5.73</td><td>5.34</td><td>0.50</td><td>0.53</td><td>0.54</td><td>1.83</td><td>0.85</td></tr><tr><td>UniSDF* [5]</td><td>6.88</td><td>0.45</td><td>1.72</td><td>2.80</td><td>8.71</td><td>8.00</td><td>4.76</td><td>0.58</td><td>0.46</td><td>2.02</td><td>1.17</td><td>1.06</td></tr><tr><td>Ours</td><td>7.15</td><td>0.51</td><td>1.77</td><td>3.60</td><td>7.92</td><td>6.55</td><td>4.58</td><td>0.35</td><td>0.24</td><td>0.90</td><td>0.94</td><td>0.61</td></tr></table>

TABLE IV

ABLATION ON MICROFACET NORMALS SAMPLING (MS), REFLECTIVE DIRECTION OF VANILLA MLP (RD), AND DETECTION-BASED COLOR BLEND WEIGHTING STRATEGY (DC). ALL METRICS ARE REPORTED IN CHAMFER DISTANCE (MM) ↓ (LOWER IS BETTER). RED AND ORANGE DENOTE TOP AND SECOND-BEST RESULTS. WHEN DC=×, BLENDING WEIGHTS ARE LEARNED VIA MLP AS IN UNISDF [5].
<table><tr><td>MS</td><td>RD</td><td>DC</td><td>24</td><td>37</td><td>40</td><td>55</td><td>63</td><td>65</td><td>69</td><td>83</td><td>97</td><td>105</td><td>106</td><td>110</td><td>114</td><td>118</td><td>122</td><td>Mean</td></tr><tr><td>X</td><td>√</td><td>√</td><td>0.95</td><td>0.62</td><td>0.32</td><td>0.51</td><td>0.92</td><td>0.58</td><td>0.52</td><td>1.37</td><td>1.01</td><td>0.51</td><td>0.50</td><td>0.69</td><td>0.35</td><td>0.43</td><td>0.39</td><td>0.65</td></tr><tr><td>√</td><td>X</td><td>√</td><td>0.74</td><td>0.75</td><td>0.35</td><td>0.54</td><td>0.84</td><td>0.70</td><td>0.58</td><td>1.33</td><td>0.99</td><td>0.50</td><td>0.50</td><td>0.68</td><td>0.34</td><td>0.41</td><td>0.39</td><td>0.64</td></tr><tr><td>√</td><td>√</td><td>X</td><td>0.60</td><td>0.73</td><td>0.31</td><td>0.53</td><td>0.89</td><td>0.51</td><td>0.52</td><td>1.35</td><td>0.92</td><td>0.50</td><td>0.47</td><td>0.75</td><td>0.38</td><td>0.37</td><td>0.39</td><td>0.615</td></tr><tr><td>√</td><td>√</td><td>√</td><td>0.53</td><td>0.64</td><td>0.31</td><td>0.44</td><td>0.88</td><td>0.62</td><td>0.56</td><td>1.34</td><td>1.00</td><td>0.52</td><td>0.48</td><td>0.66</td><td>0.33</td><td>0.39</td><td>0.40</td><td>0.607</td></tr></table>

TABLE V

ABLATION STUDY ON DETECTION-BASED COLOR BLENDING (DC) STRATEGIES ON SHINY BLENDER DATASET. LOWER IS BETTER FOR BOTH MAE<sup>◦</sup> AND ACC. RED AND ORANGE DENOTE THE BEST AND THE SECOND-BEST RESULTS, RESPECTIVELY. STABLE: STABLEDELIGHT DETECTION SCORES; ALL ONES: ALL BLENDING WEIGHTS 1 (FULLY SPECULAR BRANCH); ALL ZEROS: ALL 0 (FULLY DIFFUSE BRANCH); MLP: WEIGHTS LEARNED VIA AUXILIARY MLP AS IN UNISDF [5].
<table><tr><td rowspan="2">DC Config</td><td colspan="7">MAE° ↓</td><td colspan="5">ACC↓</td></tr><tr><td>car</td><td>ball</td><td>helmet</td><td>teapot</td><td>toaster</td><td>coffee</td><td>mean</td><td>car</td><td>helmet</td><td>toaster</td><td>coffee</td><td>mean</td></tr><tr><td>Stable</td><td>7.60</td><td>0.54</td><td>1.84</td><td>3.44</td><td>9.88</td><td>8.92</td><td>5.37</td><td>0.44</td><td>0.29</td><td>1.37</td><td>1.10</td><td>0.80</td></tr><tr><td>All Zeros</td><td>7.44</td><td>0.54</td><td>1.84</td><td>3.30</td><td>8.38</td><td>7.28</td><td>4.80</td><td>0.42</td><td>0.29</td><td>1.26</td><td>0.85</td><td>0.71</td></tr><tr><td>MLP</td><td>7.09</td><td>0.59</td><td>1.71</td><td>3.16</td><td>8.25</td><td>6.76</td><td>4.59</td><td>0.35</td><td>0.23</td><td>1.16</td><td>0.86</td><td>0.65</td></tr><tr><td>All Ones</td><td>7.15</td><td>0.51</td><td>1.77</td><td>3.60</td><td>7.92</td><td>6.55</td><td>4.58</td><td>0.35</td><td>0.24</td><td>0.90</td><td>0.94</td><td>0.61</td></tr></table>

TABLE VI

ABLATION STUDY ON DETECTION-BASED COLOR BLENDING (DC) STRATEGIES ON DTU DATASET (CHAMFER DISTANCE (MM) ↓, LOWER IS BETTER). RED AND ORANGE DENOTE THE BEST AND SECOND-BEST PER COLUMN. STABLE: STABLEDELIGHT DETECTION SCORES; ALL ONES: ALL BLENDING WEIGHTS SET TO 1 (FULLY USING THE SPECULAR BRANCH); ALL ZEROS: ALL BLENDING WEIGHTS SET TO 0 (FULLY USING THE DIFFUSE BRANCH); MLP: WEIGHTS LEARNED VIA AUXILIARY MLP AS IN UNISDF [5].
<table><tr><td>DC Config</td><td>24</td><td>37</td><td>40</td><td>55</td><td>63</td><td>65</td><td>69</td><td>83</td><td>97</td><td>105</td><td>106</td><td>110</td><td>114</td><td>118</td><td>122</td><td>Mean</td></tr><tr><td>All Ones</td><td>0.60</td><td>0.69</td><td>0.37</td><td>0.59</td><td>0.87</td><td>0.58</td><td>0.61</td><td>1.46</td><td>1.06</td><td>0.53</td><td>0.48</td><td>0.72</td><td>0.34</td><td>0.38</td><td>0.40</td><td>0.65</td></tr><tr><td>All Zeros</td><td>0.64</td><td>0.73</td><td>0.31</td><td>0.51</td><td>0.83</td><td>0.54</td><td>0.55</td><td>1.37</td><td>1.05</td><td>0.52</td><td>0.46</td><td>0.76</td><td>0.38</td><td>0.41</td><td>0.39</td><td>0.63</td></tr><tr><td>MLP</td><td>0.60</td><td>0.73</td><td>0.31</td><td>0.53</td><td>0.89</td><td>0.51</td><td>0.52</td><td>1.35</td><td>0.92</td><td>0.50</td><td>0.47</td><td>0.75</td><td>0.38</td><td>0.37</td><td>0.39</td><td>0.615</td></tr><tr><td>Stable</td><td>0.53</td><td>0.64</td><td>0.31</td><td>0.44</td><td>0.88</td><td>0.62</td><td>0.56</td><td>1.34</td><td>1.00</td><td>0.52</td><td>0.48</td><td>0.66</td><td>0.39</td><td>0.39</td><td>0.40</td><td>0.607</td></tr></table>

Ref-NeRF and DeepVoxels Datasets. As shown in Fig. 5. We compare the qualitative reconstruction results of our method against those of Ref-NeuS [4], UniSDF [5] (via InstantAngelo [15]), and AniSDF [6]. These datasets pose significant challenges due to uncontrolled lighting, real-world reflections, and specular highlights. Our method demonstrates robust performance under these conditions, consistently preserving complex geometric structures and accurately reconstructing specular regions. Specifically, Ref-NeuS struggles on realworld reflective surfaces, such as in the “toycar” scene where it incorrectly reconstructs depressed regions as flat surfaces. AniSDF and UniSDF show improved performance over Ref-

NeuS in handling reflective surfaces but still fall short in preserving fine geometric details. AniSDF particularly suffers from texture replication artifacts, erroneously interpreting printed patterns as geometric features and exhibiting a “wrapped” appearance of the target object, as seen in the first two scenes (from top to bottom) in Fig. 5. In contrast, our approach maintains geometric details while avoiding texture replication artifacts, achieving more accurate and physically plausible reconstructions.

## C. Ablation Studies

Component-wise Analysis of Color Modeling. Compared to the color network configuration in UniSDF [5], our ap-

Blend Weight

Reference Image

Specular Color

![](images/8edab4ca9d2c50742a63e1c5f12b83c4d341739e0ed0c9859eae77cd1249bff8.jpg)

![](images/410aeedfc036bd860080571a1c63d62c5aab3071d1450b702856da6936198988.jpg)

![](images/abf8b37dda764a3959d98f617cd0bbe51f1b45ce050c35f492fdefe8f0c4c6d6.jpg)

![](images/ddde1f94a4b989085ae43c6e3cac97bf8233a95c6dae6e3aa0d02dc160bc5f9b.jpg)

![](images/10c7e1721c678ebbb283bfb71d364acf1453e8434764aee51c46f42e47a79aa1.jpg)

![](images/1a5c3d57bcbd8068b429dbd9aae318aa4596ecfa1e6ebc42d318cb6fc47f3116.jpg)  
Color  
Fig. 6. Qualitative results of the proposed method under different configurations: (1) Baseline; (2) Baseline + RD (reflection direction for vanilla MLP input); (3) Baseline + RD + MS (microfacet sampling); (4) Full model: baseline + RD + MS + DC (detection-based color blending).

proach introduces three key modifications: (i) Incorporating the reflection direction (RD) as input to the Vanilla MLP; (ii) Introducing microfacet sampling (MS) during training, and (iii) Utilizing a detection-based specular score (DC) as the color blending weight between the vanilla and specular branches. To investigate their individual and combined effects of these components, we start from a baseline configuration that excludes all three components and gradually add each one.

Qualitative results in Fig. 6 illustrate the individual contributions of each component in our method. The baseline model, without RD, MS, and DC, fails to reconstruct highfrequency, nearby reflective surfaces accurately, resulting in distorted geometry, such as the region highlighted in red in Fig. 6 , which reflects the nearby sphere like a mirror. Introducing RD improves surface continuity by incorporating view-dependent cues. Adding MS further enhances geometric fidelity, particularly in reflective regions. Finally, the full model incorporating RD, MS, and DC achieves a complete and accurate reconstruction of the spherical object, demonstrating the effectiveness of the proposed design.

Quantitative results on the DTU dataset are presented in Table IV, showing the contributions of each component via Chamfer Distance (CD, lower is better). Combining all modifications (MS, RD, DC) yields an average CD of 0.607. Removing MS, RD, or DC individually degrades performance by 7.08% (CD 0.65), 5.44% (CD 0.64), or 1.32% (CD 0.615), respectively. While the DC component exhibits relatively smaller performance gains on the DTU dataset when combined with the MS and RD components, compared to the baseline that learns color blending weights automatically via an MLP, it remains valuable from a physical intuition perspective. Besides, its removal significantly worsens the CD on metallic objects like DTU 110 by 13.64% (from 0.66 to 0.75).

Specifically, explicitly introducing detection-based blending weights provides more interpretable and physically grounded guidance for the color branches. In contrast, MLP-learned weights may overfit to dataset-specific cues, leading to less physically meaningful blending behaviors. As illustrated in the two DTU scenes shown in Fig. 7, the MLP-learned approach erroneously predicts predominantly specular reflections for surfaces with diffuse characteristics, deviating from human visual perception. Fig. 8 provides a qualitative visualization for the DTU scene 65 results in Table IV, revealing that while RD has minimal impact on color appearance, it significantly improves geometric reconstruction, particularly in the bright white region on the skull’s crown.

Effect of Specular Blending Weights. To further quantify the impact of specular blending weights on reconstruction performance, we conduct an ablation study on the Shiny Blender dataset and DTU dataset by varying how these weights are defined. We compare four specular blending weight configurations: “Stable” (StableDelight detection scores directly), “All Ones” (all pixels treated as specular), “All Zeros” (all pixels treated as diffuse), and “MLP” (blending weights learned via an auxiliary MLP as in UniSDF [5]).

As shown in Table V, on Shiny Blender, which predominantly contains reflective materials, “All Ones” achieves the best performance (mean MAE 4.58<sup>◦</sup>, mean ACC 0.61), consistent with the dataset statistics. “Stable” performs relatively worse (mean MAE 5.37<sup>◦</sup>). This is due to the large domain gap between the pretrained detector and this synthetic reflective domain. However, even under this unsuitable prior, the model remains stable and competitive with other state-of-the-art methods (MAE ranks the top three, ACC remains the best as shown in Table III), indicating that the framework does not collapse under imperfect specular guidance. Qualitatively (Fig. 9),

mlp-learned

mlp-learned

![](images/17be54d7a3e7165617c5c64620a5d07cc2dcf1307eaf8c48d3da1245a279a04e.jpg)

Vanilla Color  
![](images/7a10f94ca858697fddcf64e71cd1aff70e20ed587a445bd21526e80cc012613d.jpg)

![](images/1409b9ae553f2f568758ccd0a8929d24b7b22d1515e16408397ac2d4fd65ebfa.jpg)  
Reference Image  
Normal

![](images/4023c06c0dbba5764484fb82fe20e3959d74e73908c474ca45c372137f0f4daa.jpg)  
Vanilla Color

![](images/a11c03651026bf26a6b5f1b204fdf420e0993133f0777affecb23c997c2c6e32.jpg)  
Specular Color

![](images/6dcd1270d90ecc9f01ca3e1f31588018ca87e4077c85fe85457701e5a19beda5.jpg)  
Blend Weight

![](images/ce3ebfcd65684b4170a8bf75e9476826222c91eb94513f804dfbd7693cd13480.jpg)  
Color

Fig. 7. Qualitative comparison between the MLP-learned and the proposed detection-guided color blending methods on the DTU dataset.  
![](images/0ca4913ccda6a9bf68a3cbe3a57e2bde054d466a87a755a50be2686775c779c4.jpg)  
Reference Image

![](images/3c84502f851c3377fe5505c2af1d708ced583d6999d22088b3c8018e7d3adbca.jpg)  
Normal

![](images/2de53137cfba666f1435a8ef7e182d11d962460b65153344e9d24760e0a09e0d.jpg)

![](images/108beac6dda6e8cec7b309a9d96b2cb939d2be196dc3ce36f29c6a37a47a731d.jpg)  
Specular Color

![](images/2fd592c006221e6b48008a3c7281e737c3c6e6c76f115ff05059c8a5fb9259af.jpg)  
Blend Weight

![](images/04a47f88cc2b376cb32c60937a263c0da7e6e40c4fc1948805864f1c01e90737.jpg)  
Color

Fig. 8. Qualitative comparison between using view direction and reflective direction as input to the vanilla MLP of the color network.

TABLE VII

IMPACT OF THE MICROFACET NORMAL CONSISTENCY LOSS (MNCL) ON THE DTU DATASET (CHAMFER DISTANCE (MM) ↓). WE ABLATE MNCL WHILE KEEPING MS, RD, AND DC ENABLED. RED DENOTES THE BEST RESULTS.

<table><tr><td>MS</td><td>24</td><td>37</td><td>40</td><td>55</td><td>63</td><td>65</td><td>69</td><td>83</td><td>97</td><td>105</td><td>106</td><td>110</td><td>114</td><td>118</td><td>122</td><td>Mean</td></tr><tr><td>X</td><td>0.95</td><td>0.62</td><td>0.32</td><td>0.51</td><td>0.92</td><td>0.58</td><td>0.52</td><td>1.37</td><td>1.01</td><td>0.51</td><td>0.50</td><td>0.69</td><td>0.35</td><td>0.43</td><td>0.39</td><td>0.65</td></tr><tr><td>√(wo MNCL)</td><td>0.82</td><td>0.67</td><td>0.38</td><td>0.50</td><td>0.83</td><td>0.63</td><td>0.60</td><td>1.36</td><td>0.96</td><td>0.53</td><td>0.50</td><td>0.67</td><td>0.38</td><td>0.44</td><td>0.40</td><td>0.65</td></tr><tr><td>√(w MNCL)</td><td>0.53</td><td>0.64</td><td>0.31</td><td>0.44</td><td>0.88</td><td>0.62</td><td>0.56</td><td>1.34</td><td>1.00</td><td>0.52</td><td>0.48</td><td>0.66</td><td>0.33</td><td>0.39</td><td>0.40</td><td>0.607</td></tr></table>

the advantage of stronger specular emphasis (“All Ones”) is particularly evident in near-field reflection regions (e.g., Toaster as shown in the 7th row 7th column of Fig. 9), whereas “Stable” underestimates reflective confidence in certain scenes (e.g., Coffee, as shown in 7th row 4th column of Fig. 9), leading to less accurate normal reconstruction. The “MLP” configuration yields balanced performance but does not fully match the reflective-only prior in highly specular regions. As shown in Table VI, on DTU, which contains mixed reflective and non-reflective materials, “Stable” achieves the best mean Chamfer Distance (0.607), outperforming fixed priors (“All Ones”: 0.65; “All Zeros”: 0.63) and learned blending (“MLP”: 0.615). This indicates that detector-derived blending weights are beneficial when reflective and diffuse regions coexist.

These results show that performance is affected by the suitability of the specular prior to the scene statistics rather than a hard dependency on detector accuracy. The detector serves as a soft, replaceable guidance signal, and the framework remains robust under extreme blending configurations.

Effect of Microfacet Normal Consistency Loss. The microfacet normal consistency loss (MNCL) enforces the alignment between microfacet normals and local surface normals in specular regions (as detailed in Section III-B3 and Eq. 7.), reflecting the core principle of microfacet theory. To isolate the impact of this loss term, we ablate it while keeping the full configuration (MS, RD, DC as shown in Table IV). As shown in Table VII, removing MNCL degrades performance by 7.08% (CD 0.65). This demonstrates that the consistency

#

Reference Image & Norma

All Ones

All Zeros

Stable

MLP

Reference Image & Norma

All Ones

All Zeros

Stable

MLP

Fig. 9. Ablation of specular blending weights on the Shiny Blender dataset. Columns: “Reference Image and Normal”;“All Ones” (weights 1, specular branch only); “All Zeros” (weights 0, diffuse branch only); “Stable” (StableDelight detector); “MLP” (learned weights as in UniSDF). For each object, the first row shows the blending weight map, the second row shows the reconstructed normal map.

TABLE VIII  
COMPARISON BETWEEN DENSE INITIALIZATION AND FINAL RECONSTRUCTION RESULTS ON DTU. ALL METRICS ARE REPORTED IN CHAMFER DISTANCE (MM) ↓ (LOWER IS BETTER). RED DENOTES THE BEST RESULTS.
<table><tr><td>Method</td><td>24</td><td>37</td><td>40</td><td>55</td><td>63</td><td>65</td><td>69</td><td>83</td><td>97</td><td>105</td><td>106</td><td>110</td><td>114</td><td>118</td><td>122</td><td>Mean</td></tr><tr><td>OpenMVS [68]</td><td>0.53</td><td>0.87</td><td>0.34</td><td>0.38</td><td>1.12</td><td>1.01</td><td>0.65</td><td>1.48</td><td>1.10</td><td>0.72</td><td>0.51</td><td>0.93</td><td>0.35</td><td>0.48</td><td>0.51</td><td>0.73</td></tr><tr><td>Ours</td><td>0.53</td><td>0.64</td><td>0.31</td><td>0.44</td><td>0.88</td><td>0.62</td><td>0.56</td><td>1.34</td><td>1.00</td><td>0.52</td><td>0.48</td><td>0.66</td><td>0.33</td><td>0.39</td><td>0.40</td><td>0.61</td></tr></table>

![](images/64996de9108f501252a973982e4635d4c7f5eeabc04c42b0bc3e44c62792a2e5.jpg)  
Reference Image <sub>Reference image</sub>

![](images/2a736d224b2b3d8ad136f796d4b628fea17c0f86030b9913052b767cae257b1e.jpg)  
W/O Diffuse Images <sub>W/O diffuse images</sub>

![](images/cea419b75c2f996b854d51328a7786477284b4f5295dffe27cdcd5075d079c27.jpg)  
W Diffuse Images <sub>W diffuse images</sub>

Fig. 10. Ablation result on diffuse images, showcasing reconstruction of reflective surfaces.

loss plays a crucial role in maintaining physically plausible surface orientations, which directly benefits the reconstruction quality.

On the Use of Diffuse v.s. Original Images in the Vanilla branch. To evaluate the suitability of using diffuse images as input for the vanilla branch of our dual-branch color network, we conducted a comparative experiment with two types of input: diffuse images and original images. As shown in Fig. 10, although directly employing diffuse images for reconstruction may intuitively seem appropriate for modeling diffuse reflectance, our results reveal that they significantly limit reconstruction quality. Specifically, using diffuse images leads to a substantial loss of detail, which in turn compromises the geometric fidelity of the reconstructed models. This results in degraded reconstructions, particularly in fine structural features, such as the blade edges of the scissors and the helical patterns of the screwdriver in DTU scan 37, as well as the intricate details of the buddha statue in the “gardensphere” scene. In these cases, the loss of surface detail significantly impacts reconstruction quality. Quantitatively, as shown in Table IX, the CD metric drops from 0.81 to 0.65 between (b) and (d), and from 0.72 to 0.61 between (f) and Ours, further confirming the adverse effect of using diffuse images directly.

Geometry Initialization Strategies. To evaluate the impact of initialization points on reconstruction results, we consider using initialization points from different sources. Specifically, we conduct experiments on the DTU dataset, comparing two initialization strategies: using sparse initialization points and using dense initialization points. The sparse initialization points are byproducts obtained during COLMAP’s camera pose estimation, requiring no additional processing. However, the resulting point cloud is unevenly distributed and of limited quality. In contrast, dense initialization points are generated using downsampled image sets by OpenMVS [68], which, while requiring more computational overhead compared to COLMAP, still represents a relatively small portion of the overall reconstruction process (approximately a few minutes). We utilize traditional 3D reconstruction results as the starting point for first-stage geometric initialization. To minimize the potential influence of initialization results on subsequent performance, we release the constraints of dense input points during the following three stages. The quantitative comparison results with OpenMVS are shown in Table VIII. Our method consistently outperforms OpenMVS across all scenes except for DTU scan 55, where OpenMVS reports a slightly better metric score. However, as illustrated in Fig. 11, our method produces visibly superior results on this scene, with significantly better preservation of fine details and surface quality. As shown in Fig. 12, the reconstruction results using dense initialization appear significantly smoother compared to those using sparse initialization, especially in regions with darker colors. This conclusion is further validated in Table IX; for example, the CD metric improved from 0.67 (d) to 0.61 (Ours).

![](images/5d45e8b0295f3244f0fc0c4033b762e6b7b9df379e26a2419c6fdc556e0e860d.jpg)

![](images/c99650f1b66fb661ef191f5eac322c5cc9923da1a86cb53b0f54e7587dd97fe1.jpg)

![](images/6fe31595323d0ea74f4ad84fe3a4ae10701ba5625cc2265af84bc72059152668.jpg)

![](images/32b02b94109a10fdf22bad159582825fe2f23e0489b7f86580e88ff424f36a0d.jpg)

![](images/9a2daba711ad1affaa8fe4e74345f8d4ee3b84031efc4bd72cebdbe8174900ee.jpg)

![](images/c8d9929f18dd5d279255fe79378deb5a33860ccfa4d691ce82be8013b84d77de.jpg)

![](images/8772b2453bc623e3c3e898f2ab85f79d26ec1ed1f784e44a85015eb9a9d79987.jpg)

![](images/d8956d460095b66bf67cfd340fdaa04c694c656dc27149726a6812e5603a4b68.jpg)

![](images/779a39a3a6087952394fe0c5808fc8d52de0d609d756e618a387e76bfdf1a743.jpg)

![](images/19d14f2ca038231f7c4fca420e87bd1bfd441157ce8309c4f564cc5020a908df.jpg)

![](images/b905ef5b0a3ce623c1ad037c82c9ec03ecc760bf1743e3a433bfcbe166aa1ae2.jpg)

![](images/e61119ca56070dae9000241f940c4c933daf7b64d340038ae73a29146574da19.jpg)

![](images/eef9d84f8ab338b6a0a49fdbdd6660be07a5d05f4f63139090d09e671e99b209.jpg)

![](images/c01d9797cf14e6a57393c14ea8c76e95666c752cfb84f0007bdd7bd2c8d1c10b.jpg)

![](images/ddf825297e308c9731e4196e87499ffc7e41ffe8e12c54587d21cf93ac731a48.jpg)

![](images/f32301fef02876147bff39c1613aaa5b631822cc8a0628d0a264797f951e1e70.jpg)

![](images/f90d546f23cbcfdada59caa51e34bee757bbcff56fa2edce72bc17ee8baffde3.jpg)

![](images/1d4e427270cfb14ce0ee6b35c06bd5d7103b8e1014fd911ada79312d04a2b157.jpg)

![](images/003b28dca34dd0389fd64f95ebc0ef0b79055a8964cfb4b552f021abe3e98896.jpg)

![](images/502c06e434e1f81aae5ef55de6f3ceaafd2d12b66919f2f5dd6470eb2f7e4032.jpg)  
Reference Image

![](images/d208f349d4a573f38c87439b700e2a1d4a995ab074b5d780d8e7b1d3e1579cef.jpg)

![](images/22bc5fbd1a9352a47e322c9af179f701637ba61d769797170cda02184c478ff8.jpg)

![](images/c5070c5379bfd7e769b07c663039c44280c82c3d7843575434d56854796ed8e6.jpg)  
Ours (Mesh)

![](images/18f429321210f468f9c464aeb548cba542a3db90f091d75b3851ffe50d00ad96.jpg)

![](images/f9ae13ed685b1f1eb0482bab77cb2d73a7290f39d196b4b9f09fbba946914887.jpg)  
Ours (Normal)

![](images/f6540e8669bcb96e89b504d3a3472224fd48d8527af4d8a1c777e57af91964c2.jpg)

![](images/410ad2ea6f99c094ff1149afd22d9cb2b6cd4e346d41375ee79820efa58da4c7.jpg)

![](images/82e18e8fba45408539ea4281e8699e4289858eb1997cb09b2574b996aef60e92.jpg)  
OpenMVS (Mesh)

![](images/3fc4f2e99f93fc2807f5f8a36151c5b76f951e61c28e5d2bb241661f1ab782a5.jpg)  
OpenMVS (Normal)

![](images/835a2fe8b1c2889f85f9b1c1a1b2357a3643ee6a3d003a12922351bc538e79f0.jpg)  
Fig. 11. Qualitative comparison between OpenMVS [68] and ours on the scan 55 and scan 114 from the DTU dataset.  
TABLE IX  
ABLATION STUDY ON GEOMETRY INITIALIZATION (GI), MICROFACET SAMPLING SCALE (MSS) AND DIFFUSE IMAGES (DI). ALL METRICS ARE REPORTED IN CHAMFER DISTANCE (MM) ↓ (LOWER IS BETTER). RED AND ORANGE DENOTE THE BEST AND SECOND-BEST PER COLUMN. “CUR+4” DENOTES A SCALE THAT IS 4 LEVELS FINER THAN THE CURRENT LEVEL, AND “CUR+8” DENOTES A SCALE THAT IS 8 LEVELS FINER THAN THE CURRENT LEVEL (I.E., THE FINEST GRANULARITY LEVEL IN OUR SETTING).

<table><tr><td></td><td>GI</td><td>MSS</td><td>DI</td><td>24</td><td>37</td><td>40</td><td>55</td><td>63</td><td>65</td><td>69</td><td>83</td><td>97</td><td>105</td><td>106</td><td>110</td><td>114</td><td>118</td><td>122</td><td>Mean</td></tr><tr><td>(a)</td><td>Sparse</td><td>cur+8</td><td>X</td><td>0.80</td><td>0.69</td><td>0.64</td><td>1.13</td><td>0.55</td><td>0.64</td><td>0.61</td><td>0.50</td><td>1.15</td><td>0.59</td><td>0.59</td><td>1.34</td><td>0.83</td><td>0.59</td><td>1.23</td><td>0.79</td></tr><tr><td>(b)</td><td>Sparse</td><td>cur+4</td><td>√</td><td>1.26</td><td>0.71</td><td>0.89</td><td>0.51</td><td>1.37</td><td>0.99</td><td>0.65</td><td>1.23</td><td>1.25</td><td>0.50</td><td>0.56</td><td>0.97</td><td>0.43</td><td>0.48</td><td>0.41</td><td>0.81</td></tr><tr><td>(c)</td><td>Sparse</td><td>X</td><td>X</td><td>0.60</td><td>0.73</td><td>0.52</td><td>0.51</td><td>1.23</td><td>0.96</td><td>0.53</td><td>0.92</td><td>1.17</td><td>0.51</td><td>0.54</td><td>0.70</td><td>0.36</td><td>0.39</td><td>0.37</td><td>0.67</td></tr><tr><td>(d)</td><td>Sparse</td><td>cur+4</td><td>X</td><td>0.67</td><td>0.64</td><td>0.50</td><td>0.50</td><td>1.19</td><td>0.69</td><td>0.58</td><td>0.87</td><td>1.23</td><td>0.50</td><td>0.48</td><td>0.69</td><td>0.35</td><td>0.45</td><td>0.38</td><td>0.65</td></tr><tr><td>(e)</td><td>Dense</td><td>cur+8</td><td>X</td><td>0.48</td><td>0.61</td><td>0.35</td><td>0.46</td><td>0.88</td><td>0.64</td><td>0.55</td><td>1.34</td><td>1.03</td><td>0.57</td><td>0.52</td><td>0.69</td><td>0.37</td><td>0.39</td><td>0.40</td><td>0.62</td></tr><tr><td>(f)</td><td>Dense</td><td>cur+4</td><td>√</td><td>0.65</td><td>0.64</td><td>0.41</td><td>0.50</td><td>1.02</td><td>0.95</td><td>0.65</td><td>1.53</td><td>1.10</td><td>0.52</td><td>0.57</td><td>0.93</td><td>0.41</td><td>0.50</td><td>0.44</td><td>0.72</td></tr><tr><td>(g)</td><td>Dense</td><td>X</td><td>X</td><td>0.95</td><td>0.62</td><td>0.32</td><td>0.51</td><td>0.92</td><td>0.58</td><td>0.52</td><td>1.37</td><td>1.01</td><td>0.51</td><td>0.50</td><td>0.69</td><td>0.35</td><td>0.43</td><td>0.39</td><td>0.64</td></tr><tr><td>Ours</td><td>Dense</td><td>cur+4</td><td>X</td><td>0.53</td><td>0.64</td><td>0.31</td><td>0.44</td><td>0.88</td><td>0.62</td><td>0.56</td><td>1.34</td><td>1.00</td><td>0.52</td><td>0.48</td><td>0.66</td><td>0.33</td><td>0.39</td><td>0.40</td><td>0.61</td></tr></table>

![](images/6b5fcf7893e1f1a3851e1394e64be66df4d366a81b0bdf76651f3bf74df5494f.jpg)  
Reference

![](images/fc6c4d245cfed6106d561a1ef50cb66c178e83a70ae47776736692f33188bec7.jpg)

![](images/bb9810f14033bbd9cc9099b3bc1e9f6896f513aeec549e8997a37fd569cdc454.jpg)  
(a)  
(b)  
Fig. 12. Comparison of reconstruction results using different initialization strategies: (a) dense point cloud initialization from OpenMVS, and (b) sparse point cloud initialization from COLMAP.

Effect of Microfacet Normal Scale. We evaluate the performance of microfacet normals at different scales under both sparse and dense input conditions, as shown in Table IX. In our setting, the initial level is 8 and the finest level is 16. We compare two scale choices. “cur+4” denotes a scale 4 levels finer than the current level, and “cur+8” denotes 8 levels finer, i.e. the finest granularity, corresponding to directly sampling microfacet normals at level 16. Under dense input, switching from the finest scale to “cur+4” improves CD from 0.62 in row (e) to 0.61 in Ours. Under sparse input, the same switch from “cur+8” to “cur+4” improves CD from 0.79 in row (a) to 0.65 in row (d), a larger gain than in the dense case. In both conditions, “cur+4” outperforms the finest scale “cur+8”. This suggests that using the finest grid scale is not necessarily optimal, which is particularly evident when the geometry initialization is sparse, where overly fine scales may amplify noise or lead to unstable normal estimation.

## V. CONCLUSION

In this work, we presented MicroSDF, a novel framework for 3D reconstruction of scenes with both reflective and nonreflective surfaces. Grounded in physical principles, MicroSDF integrates microfacet theory to guide both geometric and color modeling. Specifically, we introduce a microfacet consistency loss to enforce physically meaningful surface geometry for reflective regions, while leveraging microfacet normals to facilitate specular color learning. This is further enhanced by a detection-guided color blending strategy, which provides external supervision for physically plausible color modeling. Together, these components enable our framework to effectively handle both reflective and non-reflective surfaces in a unified manner.

Extensive experiments across diverse benchmark datasets validate the effectiveness and robustness of our framework. On the DTU dataset, which mixes reflective and non-reflective surfaces, MicroSDF achieves state-of-the-art performance with an average Chamfer Distance of 0.61, outperforming existing methods. On the synthetic reflective Shiny Blender benchmark, our method also attains state-of-the-art quantitative results, demonstrating its strong capability in handling highly specular scenes. For real-world reflective datasets such as Ref-NeRF and DeepVoxels, we provide qualitative evaluations that show high-fidelity reconstruction of reflective regions and improved surface consistency. These results indicate that MicroSDF generalizes well across both controlled synthetic environments and real-world scenarios. Moreover, the proposed integration of microfacet-driven priors offers a principled way to incorporate physically grounded reflectance modeling into neural reconstruction frameworks. This idea is potentially compatible with recent 3D Gaussian Splatting approaches for improved reconstruction of reflective and specular surfaces, which we consider a promising direction for future work.

## REFERENCES

[1] R. Hartley and A. Zisserman, Multiple view geometry in computer vision. Cambridge Univ. Press, 2003.

[2] C. Barnes, E. Shechtman, A. Finkelstein, and D. B. Goldman, “Patch-Match: A randomized correspondence algorithm for structural image editing,” ACM Trans. Graph., vol. 28, no. 3, p. 24, 2009.

[3] P. Wang, L. Liu, Y. Liu, C. Theobalt, T. Komura, and W. Wang, “NeuS: Learning neural implicit surfaces by volume rendering for multi-view reconstruction,” in Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), 2021.

[4] W. Ge, T. Hu, H. Zhao, S. Liu, and Y.-C. Chen, “Ref-NeuS: Ambiguityreduced neural implicit surface learning for multi-view reconstruction with reflection,” in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), 2023.

[5] F. Wang, M.-J. Rakotosaona, M. Niemeyer, R. Szeliski, M. Pollefeys, and F. Tombari, “UniSDF: Unifying neural representations for highfidelity 3D reconstruction of complex scenes with reflections,” in Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), 2024.

[6] J. Gao, Z. Chen, X. Yang, and Y. Yan, “AniSDF: Fused-Granularity neural surfaces with anisotropic encoding for high-fidelity 3d reconstruction,” in Proc. IEEE Int. Conf. Learn. Represent. (ICLR), 2025.

[7] Z. Li, T. Muller, A. Evans, R. H. Taylor, M. Unberath, M.-Y. Liu, and¨ C.-H. Lin, “Neuralangelo: High-fidelity neural surface reconstruction,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2023.

[8] Y. Wang, Q. Han, M. Habermann, K. Daniilidis, C. Theobalt, and L. Liu, “NeuS2: Fast learning of neural implicit surfaces for multi-view reconstruction,” in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), 2023.

[9] X. Guo, X. Cao, and Y. Ma, “Robust separation of reflection from multiple images,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2014.

[10] K. Zhang, F. Luan, Q. Wang, K. Bala, and N. Snavely, “PhySG: Inverse rendering with spherical gaussians for physics-based material editing and relighting,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2021.

[11] M. Boss, R. Braun, V. Jampani, J. T. Barron, C. Liu, and H. Lensch, “NeRD: Neural reflectance decomposition from image collections,” in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), 2021.

[12] P. P. Srinivasan, B. Deng, X. Zhang, M. Tancik, B. Mildenhall, and J. T. Barron, “NeRV: Neural reflectance and visibility fields for relighting and view synthesis,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2021.

[13] X. Zhang, P. P. Srinivasan, B. Deng, P. Debevec, W. T. Freeman, and J. T. Barron, “NeRFactor: Neural factorization of shape and reflectance under an unknown illumination,” ACM Trans. Graph., vol. 40, pp. 1–18, 2021.

[14] M. Pharr, W. Jakob, and G. Humphreys, Physically based rendering: From theory to implementation. MIT Press, 2023.

[15] hugoycj, “Instant-angelo: Build high-fidelity Digital Twin within 20 Minutes!” https://github.com/hugoycj/Instant-angelo, 2023.

[16] L. Yariv, J. Gu, Y. Kasten, and Y. Lipman, “Volume rendering of neural implicit surfaces,” in Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), 2021.

[17] L. Yariv, P. Hedman, C. Reiser, D. Verbin, P. P. Srinivasan, R. Szeliski, J. T. Barron, and B. Mildenhall, “BakedSDF: Meshing neural sdfs for real-time view synthesis,” in Proc. ACM SIGGRAPH Conf. (SIG-GRAPH), 2023.

[18] Q. Fu, Q. Xu, Y. S. Ong, and W. Tao, “Geo-NeuS: Geometry-consistent neural implicit surfaces learning for multi-view reconstruction,” in Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), 2022.

[19] Stable-X, “StableDelight: Revealing hidden textures by removing specular reflections,” https://github.com/Stable-X/StableDelight/tree/master, 2025.

[20] S. M. Seitz, B. Curless, J. Diebel, D. Scharstein, and R. Szeliski, “A comparison and evaluation of multi-view stereo reconstruction algorithms,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2006.

[21] J. L. Schonberger and J.-M. Frahm, “Structure-from-motion revisited,”¨ in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2016.

[22] J. L. Schonberger, E. Zheng, M. Pollefeys, and J.-M. Frahm, “Pixelwise¨ view selection for unstructured multi-view stereo,” in Proc. Eur. Conf. Comput. Vis. (ECCV), 2016.

[23] S. Shen, “Accurate multiple view 3D reconstruction using patch-based stereo for large-scale scenes,” IEEE Trans. Image Process., vol. 22, pp. 1901–1914, 2013.

[24] Z. Li, W. Zuo, Z. Wang, and L. Zhang, “Confidence-based large-scale dense multi-view stereo,” IEEE Trans. Image Process., vol. 29, pp. 7176–7191, 2020.

[25] M. Kazhdan and H. Hoppe, “Screened poisson surface reconstruction,” ACM Trans. on Graph., vol. 32, pp. 1–13, 2013.

[26] Y. Yao, Z. Luo, S. Li, T. Fang, and L. Quan, “MVSNet: Depth inference for unstructured multi-view stereo,” in Proc. Eur. Conf. Comput. Vis. (ECCV), 2018.

[27] P.-H. Chen, H.-C. Yang, K.-W. Chen, and Y.-S. Chen, “MVSNet++: Learning depth-based attention pyramid features for multi-view stereo,” IEEE Trans. Image Process., vol. 29, pp. 7261–7273, 2020.

[28] Y. Zhang, J. Zhu, and L. Lin, “Multi-view stereo representation revist: Region-aware mvsnet,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2023.

[29] T. Liu, X. Ye, W. Zhao, Z. Pan, M. Shi, and Z. Cao, “When epipolar constraint meets non-local operators in multi-view stereo,” in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), 2023.

[30] J. Li, Z. Lu, Y. Wang, J. Xiao, and Y. Wang, “NR-MVSNet: Learning multi-view stereo based on normal consistency and depth refinement,” IEEE Trans. Image Process., vol. 32, pp. 2649–2662, 2023.

[31] J. Chang, J. He, T. Zhang, J. Yu, and F. Wu, “EI-MVSNet: Epipolar-Guided multi-view stereo network with interval-aware label,” IEEE Trans. Image Process., vol. 33, pp. 753–766, 2024.

[32] Z. Zhang, R. Peng, Y. Hu, and R. Wang, “GeoMVSNet: Learning multiview stereo with geometry perception,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2023.

[33] C. Cao, X. Ren, and Y. Fu, “MVSFormer: Multi-view stereo by learning robust image features and temperature-based depth,” [Online]. Available: https://arxiv.org/abs/2208.02541, 2022, arXiv preprint.

[34] X. R. Chenjie Cao and Y. Fu, “MVSFormer++: Revealing the devil in transformer’s details for multi-view stereo,” in Proc. IEEE Int. Conf. Learn. Represent. (ICLR), 2024.

[35] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and R. Ng, “NeRF: Representing scenes as neural radiance fields for view synthesis,” in Proc. Eur. Conf. Comput. Vis. (ECCV), 2020.

[36] B. Kerbl, G. Kopanas, T. Leimkuhler, and G. Drettakis, “3D Gaussian¨ splatting for real-time radiance field rendering.” ACM Trans. Graph., vol. 42, pp. 139–1, 2023.

[37] Z. Bao, G. Liao, K. Zhou, K. Liu, Q. Li, and G. Qiu, “LoopSparseGS: Loop-Based sparse-view friendly gaussian splatting,” IEEE Trans. Image Process., vol. 34, pp. 3889–3902, 2025.

[38] L. Han, D. Zhong, L. Li, K. Zheng, and L. Fang, “Learning residual color for novel view synthesis,” IEEE Trans. Image Process., vol. 31, pp. 2257–2267, 2022.

[39] K. Wang, C. Wang, J. Yang, and G. Zhang, “CloCap-GS: Clothed human performance capture with 3D gaussian splatting,” IEEE Trans. Image Process., pp. 1–1, 2025.

[40] K. Yang, R. Gu, M. Wang, M. Toyoura, and G. Xu, “LASOR: Learning accurate 3D human pose and shape via synthetic occlusion-aware data and neural mesh rendering,” IEEE Trans. Image Process., vol. 31, pp. 1938–1948, 2022.

[41] X. Zheng, L. Liao, J. Jiao, F. Gao, and R. Wang, “Surface-SOS: Self-Supervised object segmentation via neural surface representation,” IEEE Trans. Image Process., vol. 33, pp. 2018–2031, 2024.

[42] C. Huang, Y. Hou, W. Ye, D. Huang, X. Huang, B. Lin, and D. Cai, “NeRF-Det++: Incorporating semantic cues and perspective-aware depth supervision for indoor multi-view 3d detection,” IEEE Trans. Image Process., vol. 34, pp. 2575–2587, 2025.

[43] B. Huang, Z. Yu, A. Chen, A. Geiger, and S. Gao, “2D gaussian splatting for geometrically accurate radiance fields,” in ACM SIGGRAPH Comput. Graph., 2024.

[44] D. Svitov, P. Morerio, L. Agapito, and A. Del Bue, “Billboard Splatting (BBSplat): Learnable textured primitives for novel view synthesis,” in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), 2025.

[45] A. Guedon and V. Lepetit, “SuGaR: Surface-aligned gaussian splatting´ for efficient 3d mesh reconstruction and high-quality mesh rendering,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2024.

[46] Z. Yu, T. Sattler, and A. Geiger, “Gaussian Opacity Fields: Efficient adaptive surface reconstruction in unbounded scenes,” ACM Trans. on Graph., 2024.

[47] B. Zhang, C. Fang, R. Shrestha, Y. Liang, X. Long, and P. Tan, “Rade-GS: Rasterizing depth in gaussian splatting,” [Online]. Available: https: //arxiv.org/abs/2406.01467, 2024, arXiv preprint.

[48] H. Li, W. Song, T. Xu, A. Elsig, and J. Kulhanek, “WaterSplatting: Fast underwater 3d scene reconstruction using gaussian splatting,” in Proc. Int. Conf. 3D Vision (3DV). IEEE, 2025.

[49] Q. Dai, X. Ni, Q. Shen, W. Chen, B. Chen, and M. Chu, “RainyGS: Efficient rain synthesis with physically-based gaussian splatting,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2025.

[50] H. Zhou, W. Dong, and J. Chen, “LITA-GS: Illumination-Agnostic novel view synthesis via reference-free 3d gaussian splatting and physical priors,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2025.

[51] K. Zhang, G. Riegler, N. Snavely, and V. Koltun, “NeRF++: Analyzing and improving neural radiance fields,” [Online]. Available: https://arxiv. org/abs/2010.07492, 2020, arXiv preprint.

[52] X. Long, C. Lin, P. Wang, T. Komura, and W. Wang, “SparseNeuS: Fast generalizable neural surface reconstruction from sparse views,” in Proc. Eur. Conf. Comput. Vis. (ECCV), 2022.

[53] M. Oechsle, S. Peng, and A. Geiger, “UNISURF: Unifying neural implicit surfaces and radiance fields for multi-view reconstruction,” in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), 2021.

[54] W. E. Lorensen, “History of the marching cubes algorithm,” IEEE Comput. Graph. Appl., vol. 40, pp. 8–15, 2020.

[55] K. Deng, A. Liu, J.-Y. Zhu, and D. Ramanan, “Depth-supervised nerf: Fewer views and faster training for free,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2022.

[56] B. Roessle, J. T. Barron, B. Mildenhall, P. P. Srinivasan, and M. Nießner, “Dense depth priors for neural radiance fields from sparse input views,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2022.

[57] Y. Wei, S. Liu, Y. Rao, W. Zhao, J. Lu, and J. Zhou, “NerfingMVS: Guided optimization of neural radiance fields for indoor multi-view stereo,” in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), 2021.

[58] C. Zhang, W. Su, Q. Xu, X. Liao, and W. Tao, “PG-NeuS: Robust and efficient point guidance for multi-view neural surface reconstruction,” IEEE Trans. Vis. Comput. Graph., vol. 31, pp. 6410–6424, 2025.

[59] Z. Yu, S. Peng, M. Niemeyer, T. Sattler, and A. Geiger, “MonoSDF: Exploring monocular geometric cues for neural implicit surface recon struction,” in Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), 2022.

[60] R. Liang, H. Chen, C. Li, F. Chen, S. Panneer, and N. Vijaykumar, “Envidr: Implicit differentiable renderer with neural environment lighting,” in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), 2023.

[61] Y.-C. Guo, D. Kang, L. Bao, Y. He, and S.-H. Zhang, “NerfReN: Neural radiance fields with reflections,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2022.

[62] M. Boss, V. Jampani, R. Braun, C. Liu, J. Barron, and H. Lensch, “Neural-PIL: Neural pre-integrated lighting for reflectance decomposition,” in Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), 2021.

[63] H. Jin, I. Liu, P. Xu, X. Zhang, S. Han, S. Bi, X. Zhou, Z. Xu, and H. Su, “Tensoir: Tensorial inverse rendering,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2023.

[64] A. Mai, D. Verbin, F. Kuester, and S. Fridovich-Keil, “Neural microfacet fields for inverse rendering,” in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), 2023.

[65] D. Verbin, P. Hedman, B. Mildenhall, T. Zickler, J. T. Barron, and P. P. Srinivasan, “Ref-NeRF: Structured view-dependent appearance for neural radiance fields,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2022.

[66] T. Muller, A. Evans, C. Schied, and A. Keller, “Instant neural graph-¨ ics primitives with a multiresolution hash encoding,” ACM Trans. on Graph., vol. 41, pp. 1–15, 2022.

[67] R. L. Cook and K. E. Torrance, “A reflectance model for computer graphics,” ACM Trans. on Graph., vol. 1, pp. 7–24, 1982.

[68] cdcseacave, “OpenMVS: Open multi-view stereo reconstruction library,” [Online]. Available: https://github.com/cdcseacave/openMVS, 2023, gitHub Repository.

[69] V. Sitzmann, J. Martel, A. Bergman, D. Lindell, and G. Wetzstein, “Implicit neural representations with periodic activation functions,” in Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), 2020.

[70] M. S. Lee, W. Shin, and S. W. Han, “TRACER: Extreme attention guided salient object tracing network (student abstract),” in Proc. AAAI Conf. Artif. Intell. (AAAI), 2022.

[71] A. Gropp, L. Yariv, N. Haim, M. Atzmon, and Y. Lipman, “Implicit geometric regularization for learning shapes,” in Proc. Mach. Learn. Syst. (MLSys), 2020.

[72] F. Darmon, B. Bascle, J.-C. Devaux, P. Monasse, and M. Aubry, “Improving neural implicit surfaces geometry with patch warping,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2022.

[73] R. A. Rosu and S. Behnke, “PermutoSDF: Fast multi-view reconstruction with implicit surfaces using permutohedral lattices,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2023.

[74] M. Yu, T. Lu, L. Xu, L. Jiang, Y. Xiangli, and B. Dai, “GSDF: 3DGS meets SDF for improved rendering and reconstruction,” in Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), 2024.

[75] H. Aanæs, R. R. Jensen, G. Vogiatzis, E. Tola, and A. B. Dahl, “Large-scale data for multiple-view stereopsis,” International Journal of Computer Vision, 2016.

[76] V. Sitzmann, J. Thies, F. Heide, M. NieBner, G. Wetzstein, and M. Zollhofer, “DeepVoxels: Learning persistent 3D feature embeddings,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2019.

[77] J. T. Barron, B. Mildenhall, D. Verbin, P. P. Srinivasan, and P. Hedman, “Mip-NeRF 360: Unbounded anti-aliased neural radiance fields,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2022.