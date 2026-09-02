# MEROPE: METRIC ROTARY POSITION EMBEDDING FOR CAMERA-CONTROLLED VIDEO GENERATION

Zhijian Qiao<sup>1</sup>, Xinjiang Wang<sup>2</sup>, Jiajie Chen<sup>2</sup>, Haoming Huang<sup>2</sup>, Meng Li<sup>2</sup>, Chih-Chung Chou<sup>2</sup>, Jing Wang<sup>2</sup>, and Shaojie Shen<sup>1,†</sup>

<sup>1</sup>The Hong Kong University of Science and Technology <sup>2</sup>Zhuoyu Technology

<sup>†</sup>Corresponding author: eeshaojie@ust.hk

Project page: qiaozhijian.github.io/merope

## ABSTRACT

In camera-controlled video generation, geometry-aware positional encodings condition tokens on camera extrinsics and per-token viewing rays. Existing schemes, however, have a scale-dependent failure mode on real-world metric camera trajectories: homogeneous projective encodings cause attention logits and feature norms to grow unbounded with physical translation baselines. We propose MeRoPE (Metric Rotary Position Embedding), a norm-preserving relative camera encoding for attention. MeRoPE encodes relative orientations between calibrated viewing rays with orthogonal rotation blocks, maps raw metric displacements into multi-frequency rotary phases, and adds a disparity-anchored correspondence prior along the epipolar arc. This design strictly preserves feature norms, bounds pre-softmax attention logits regardless of the physical translation scale, and maintains exact invariance to global rigid coordinate changes. Across nuScenes and PanShot, which cover large-baseline trajectories and diverse camera optics, respectively, MeRoPE achieves stronger camera control than prior encodings, with the best consistency between generated camera motion and conditioning poses in both rotation and translation. Code will be made publicly available.

## 1 INTRODUCTION

Geometry-aware positional encodings (PEs) enable camera-controlled video generation by making attention sensitive to camera poses and token-level viewing geometry. However, representing realworld metric translation within these encodings remains challenging: raw physical displacements can destabilize attention, whereas normalization discards physical scale.

Existing relative camera encodings based on homogeneous projective matrices (Miyato et al., 2024; Li et al., 2025; Zhang et al., 2026a) jointly encode rotation and translation in non-orthogonal attention operators. In these formulations, physical translation enters as an unnormalized inner product, causing attention logits and feature norms to grow unbounded with the baseline length. Over large displacements, translational terms artificially dominate attention scores and suppress visual content, degrading trajectory control.

This tension yields three task-driven desiderata for camera positional encodings: (A) depthindependent full metric relative-pose dependence on $T _ { i } ^ { - 1 } T _ { j }$ , which retains both motion direction and physical translation scale without coupling translation to an assumed scene depth; (B) strict pertoken factorization, which permits independent query/key transformations compatible with standard attention; and (C) norm-preserving actions, which prevent baseline-dependent amplification of logits and features. However, Theorem 1 shows that no continuous finite-dimensional encoding can satisfy all three while remaining sensitive to metric translation. Because camera control requires both metric fidelity (A) and scale stability (C), we prioritize (A+C) and relax (B) through query-camera grouping.

We therefore introduce MeRoPE (Metric Rotary Position Embedding), a norm-preserving encoding with full metric relative-pose dependence. MeRoPE constructs a minimum-rotation (MinRot)

local frame for each calibrated 3D ray, represents relative ray orientation with orthogonal blocks, and maps query-frame metric displacements into multi-frequency rotary phases. Together, these operators retain raw metric camera motion while bounding attention logits across arbitrary physical baselines.

Because this pose operator is scene-independent, it does not identify potential static-scene correspondences across views. MeRoPE therefore encodes correspondence hypotheses as disparityanchored rotations along spherical epipolar arcs. Unlike CamCo (Xu et al., 2024), this design uses epipolar geometry to express relative cross-view relations without restricting which token pairs can interact. Its angular disparity anchors span a bounded spherical arc, avoiding the dataset-specific metric-depth intervals of URoPE (Xie et al., 2026a) and the learned depth estimation of Ray-RoPE (Wu et al., 2026).

## Contributions.

• We identify baseline-dependent logit and feature-norm growth in homogeneous projective camera encodings and formalize a trilemma among full metric relative-pose dependence, strict per-token factorization, and norm preservation.

• We propose MeRoPE, which resolves this trade-off through query-camera grouping and combines a norm-preserving metric-pose operator with disparity-anchored spherical rotations for cross-view relations.

• We demonstrate improved camera control on large-baseline nuScenes and camera-diverse PanShot, with MeRoPE achieving the best joint rotation–translation consistency on both benchmarks.

## 2 RELATED WORK

Camera conditioning pathways in video generation. Camera trajectories control viewpoint in video synthesis for applications such as autonomous driving and robotic simulation. Existing approaches implement this control through several broad pathways. Feature fusion and adapters integrate compact pose or ray representations into generative backbones: MotionCtrl (Wang et al., 2024) concatenates camera extrinsics with temporal features, CameraCtrl (He et al., 2025) adds multi-scale features encoded from per-pixel Plucker rays, ReCamMaster (Bai et al., 2025) addi-¨ tively embeds target extrinsics into transformer features, and CamCo (Xu et al., 2024) combines Plucker-conditioned feature fusion with epipolar attention. These mechanisms provide lightweight¨ control but do not by themselves encode relative viewpoint relations in attention. Explicit 3D guidance and rendering instead derives structural conditions from estimated geometry: Gen3C (Ren et al., 2025) renders point-cloud caches, Lyra 2.0 (Shen et al., 2026) combines depth-based warping with geometry-aware memory, and ViewCrafter (Yu et al., 2025b) and TrajectoryCrafter (Yu et al., 2025a) condition generation on point-cloud renderings. They provide direct spatial guidance, but errors in estimated depth, pose, or scene motion can produce holes, distortions, or misaligned conditions. Adaptive modulation injects camera-dependent affine transformations rather than directly fusing conditions: BulletTime (Wang et al., 2026b) combines Camera-AdaLN with camera-aware positional encoding, while CompoSIA (Zhan et al., 2026) couples AdaLN-style residual ego-motion modulation with global projective attention. Geometry-aware positional encodings incorporate camera relations directly into attention without constructing an external 3D proxy at inference time.

Evolution of geometry-aware positional encodings. Within this family, early formulations operated on camera-level extrinsic transformations (Kong et al., 2024; Miyato et al., 2024) and projective viewing frustums (Li et al., 2025). Specifically, CaPE (Kong et al., 2024) applies block-diagonal relative pose matrices to queries and keys in multi-view attention, GTA (Miyato et al., 2024) gen eralizes this to group representations of $S E ( 3 ) \times S O ( 2 ) \times S O ( 2 )$ acting on queries, keys, and values, and PRoPE (Li et al., 2025) incorporates intrinsics to model projective viewing frustums. Subsequent methods construct per-token 3D ray representations: UCPE (Zhang et al., 2026a) builds token-level ray-local rigid frames from calibrated pixel-to-ray mappings to support diverse camera optics, and RayNova (Xie et al., 2026b) applies multi-axis RoPE to Plucker coordinates in a fixed¨ world frame. Recent efforts further incorporate depth cues: URoPE (Xie et al., 2026a) lifts tokens to discrete depth anchors and reprojects them onto the query pixel plane, RayRoPE (Wu et al.,

2026) computes expected rotary phases from predicted depth segments, and CRePE (Jin et al., 2026) models curved rays with pseudo-depth expectation distributions.

Taxonomy under the trilemma and MeRoPE. Table 1 organizes camera positional encodings by three properties: (A) depth-independent full metric relative-pose dependence on $\stackrel { \cdot } { T } _ { i } ^ { - 1 } T _ { j }$ , (B) strict per-token factorization, and (C) norm-preserving attention actions. Homogeneous and projective encodings (CaPE, GTA, PRoPE, UCPE) preserve (A+B) but violate (C), as non-orthogonal translation blocks cause attention logits to scale unbounded with physical baselines. World-frame ray models such as RayNova and direction-only encodings such as ViewRope (Xiang et al., 2026) preserve (B+C) but sacrifice (A): RayNova depends on the chosen global coordinate frame, whereas ViewRope omits metric translation. Query-frame reprojection approaches (Xie et al., 2026a; Wu et al., 2026; Jin et al., 2026) relax (B) and do not satisfy (A), because their attention relations are defined in pixel coordinates. Under

Table 1: Taxonomy under three properties: (A) depth-independent full metric relative pose, (B) strict per-token factorization, and (C) norm-preserving attention action.
<table><tr><td>Method</td><td>(A) (B) (C)</td></tr><tr><td>CaPE (Kong et al., 2024) GTA (Miyato et al., 2024) PRoPE (Li et al., 2025) UCPE (Żhang et al., 2026a)</td><td>Yes Yes s No Yes Yes No Yes Yes No</td></tr><tr><td>RayNova (Xie et al., 2026b) ViewRope (Xiang et al., 2026)</td><td>Yes Yes No No Yes Yes No Yes Yes</td></tr><tr><td>URoPE (Xie et al., 2026a) RayRoPE (Wu et al., 2026) CRePE (Jin et al., 2026)</td><td>No No Yes No No No No No No</td></tr><tr><td>MeRoPE (Ours)</td><td>Yes No Yes</td></tr></table>

central projection, jointly scaling camera translation and scene depth leaves projected coordinates unchanged, so pixel-space relations identify translation only up to a depth scale—the same projective ambiguity underlying monocular scale ambiguity. Fixed or predicted depths select a scale but do not make the pixel-space relation intrinsically metric. URoPE preserves (C), while the expected phasors used by RayRoPE and CRePE are not strictly norm-preserving. MeRoPE retains (A+C) by relaxing (B) via query-camera grouping; Section 3 verifies $\mathrm { ( A + C ) }$ , and Theorem 1 establishes why (B) must be relaxed.

## 3 METHOD

This section presents MeRoPE (Metric Rotary Position Embedding), a norm-preserving relative camera positional encoding for video generation. Section 3.1 formalizes geometry-aware attention and analyzes the scale-dependent instability of homogeneous relative-pose operators. Section 3.2 encodes ray-frame rotation and raw metric translation with orthogonal blocks, while Section 3.3 introduces disparity-anchored spherical rotations for potential static-scene correspondences.

## 3.1 GEOMETRY-AWARE ATTENTION AND METRIC-SCALE LIMITATION

We formalize how camera positional encodings act within a geometry-aware attention layer. Let $a = ( i , p )$ denote patch $p$ from query camera i, and let $b = ( j , q )$ denote patch $q$ from key camera $j ,$ with corresponding query, key, and value vectors $\mathbf q _ { a } , \mathbf k _ { b } , \mathbf v _ { b } \ \in \mathbb R ^ { d }$ . Geometry-aware attention modulates query-key comparison and value aggregation through a pairwise transformation operator $\mathcal { U } _ { a b } \in \mathbb { R } ^ { d \times \bar { d } }$

$$
s _ { a b } = \frac { \mathbf { q } _ { a } ^ { \top } \mathcal { U } _ { a b } \mathbf { k } _ { b } } { \sqrt { d } } , \qquad \alpha _ { a b } = \frac { \exp \left( s _ { a b } \right) } { \sum _ { b ^ { \prime } } \exp \left( s _ { a b ^ { \prime } } \right) } , \qquad \mathbf { y } _ { a } = \sum _ { b } \alpha _ { a b } \mathcal { U } _ { a b } \mathbf { v } _ { b } .\tag{1}
$$

In GTA (Miyato et al., 2024), the camera component of $\mathcal { U } _ { a b }$ is a direct sum $\oplus u _ { i j } ^ { \mathrm { c a m } }$ of $4 \times 4$ homogeneous relative-pose blocks:

$$
R _ { i j } : = R _ { i } ^ { \top } R _ { j } , \qquad \Delta \mathbf { o } _ { j \mid i } : = R _ { i } ^ { \top } ( \mathbf { o } _ { j } - \mathbf { o } _ { i } ) , \qquad \mathcal { U } _ { i j } ^ { \mathrm { c a m } } : = \left[ { R _ { i j } \quad \Delta \mathbf { o } _ { j \mid i } } \right] ,\tag{2}
$$

where $R _ { i } \in \mathrm { S O ( 3 ) }$ and ${ \mathbf o } _ { i } \in \mathbb { R } ^ { 3 }$ denote the rotation and optical center of camera i. Decomposing the 4D feature segments of ${ \bf q } _ { a }$ and $\mathbf { k } _ { b }$ into 3D vectors and scalar coordinates, $[ \mathbf { u } ^ { q } ; h _ { q } ]$ and $[ \tilde { \mathbf { u } ^ { k } } ; h _ { k } ]$ the resulting attention score contribution is:

$$
\begin{array} { r } { \left[ \mathbf { u } _ { q } ^ { q } \right] ^ { \top } \mathcal { U } _ { i j } ^ { \mathrm { c a m } } \left[ \mathbf { u } _ { k } ^ { k } \right] = ( \mathbf { u } ^ { q } ) ^ { \top } R _ { i j } \mathbf { u } ^ { k } + h _ { k } ( \mathbf { u } ^ { q } ) ^ { \top } \Delta \mathbf { o } _ { j | i } + h _ { q } h _ { k } . } \end{array}\tag{3}
$$

Because the second term $h _ { k } ( { \mathbf { u } } ^ { q } ) ^ { \top } \Delta { \mathbf { o } } _ { j \mid i }$ is an unnormalized linear inner product, it scales linearly with the physical translation baseline $\lVert  { \mathbf { o } } _ { j } -  { \mathbf { o } } _ { i } \rVert _ { 2 }$ . In large-baseline settings, this translational term numerically dominates the attention logits, suppressing both rotational geometry and semantic visual content. Similarly, because $\mathcal { U } _ { i j } ^ { \mathrm { c a m } }$ is non-orthogonal, the aligned value features also scale unbounded with the baseline:

$$
\left\| \mathcal { U } _ { i j } ^ { \mathrm { c a m } } \left[ \mathbf { u } ^ { v } \right] \right\| _ { 2 } ^ { 2 } = \left\| R _ { i j } \mathbf { u } ^ { v } + h _ { v } \Delta \mathbf { o } _ { j | i } \right\| _ { 2 } ^ { 2 } + h _ { v } ^ { 2 } .\tag{4}
$$

Figure 1 empirically illustrates this limitation on a trained baseline model. Evaluating temporal attention allocations along a forward-driving trajectory, we measure the cross-frame attention mass from the center query token of the final frame across all key frames. UCPE assigns an increasing share of attention to distant early frames as the translation baseline grows, reducing the mass on adjacent frames despite their greater crossview co-visibility. MeRoPE bounds the positional contribution and avoids this baseline-driven shift.

![](images/250051e249e6de4be52c77bfbc6dec75f8dfc789985220fd31eeee7996355f36.jpg)  
Figure 1: Temporal attention under large baselines. Cross-frame attention mass from the center query token of the final frame across all preceding key frames along a straight trajectory. As the baseline grows, UCPE (red) allocates more mass to distant early frames, whereas MeRoPE (blue) avoids this shift.

## 3.2 ORTHOGONAL METRIC POSE ENCODING

The intrinsic relative camera pose $( R _ { i } ^ { \top } R _ { j } , \Delta \mathbf { o } _ { j \mid i } )$ consists of relative rotation and metric displacement. To prevent the norm and logit explosion of homogeneous matrices, we decompose relative pose into two decoupled orthogonal operators: a ray-local minimumrotation (MinRot) block $\mathcal { U } _ { a b } ^ { \mathrm { r o t } }$ and a query-frame metric translation RoPE block $\mathcal { U } _ { i j } ^ { \mathrm { t r a n s } }$

Ray-local coordinate framing via MinRot. Each image patch $p$ from camera i is mapped to a calibrated unit viewing ray $\mathbf { d } _ { i , p } ^ { c ^ { \scriptscriptstyle - } } \in \mathbb { S } ^ { 2 }$ in the camera coordinate frame, providing a camera-modelagnostic representation across heterogeneous optics. To make attention sensitive to individual viewing directions rather than whole-camera poses alone, each token ray is assigned a local coordinate frame $( A _ { i , p }$ and $A _ { j , q }$ in Figure 2). We construct this ray-local rotation matrix $A _ { i , p } \in \mathrm { S O } ( 3 )$ via the Rodrigues minimum-rotation formula, carrying the optical axis $\mathbf { e } _ { z }$ directly onto $\mathbf { d } _ { i , p } ^ { c } \mathrm { : \_ }$

$$
A _ { i , p } = I + [ \mathbf { v } _ { i , p } ] _ { \times } + \frac { [ \mathbf { v } _ { i , p } ] _ { \times } ^ { 2 } } { 1 + c _ { i , p } } , \mathrm { ~ w h e r e ~ } \mathbf { v } _ { i , p } = \mathbf { e } _ { z } \times \mathbf { d } _ { i , p } ^ { c } , c _ { i , p } = \mathbf { e } _ { z } ^ { \top } \mathbf { d } _ { i , p } ^ { c } .\tag{5}
$$

Unlike UCPE (Zhang et al., 2026a), whose cross-product frame degenerates when a token’s vertical ray angle reaches $\pm 9 0 ^ { \circ }$ (equivalently, as the vertical field of view approaches $1 8 0 ^ { \circ } ) , A _ { i , p }$ is singular only at the backward direction $\mathbf { d } _ { i , p } ^ { c } = - \mathbf { e } _ { z } \left( - 1 8 0 ^ { \circ } \right)$ . Transforming ray p from camera i into world coordinates yields the ray-local frame $R _ { i , p } ^ { \mathrm { r a y } } = R _ { i } A _ { i , p } .$ . The pairwise relative rotation between query ray $( i , p )$ and key ray $( j , q )$ is then $A _ { i , p } ^ { \top } R _ { i } ^ { \top } R _ { j } A _ { j , q }$ . In attention, this relative rotation is embedded as a direct sum over m repeated 3D rotation blocks:

$$
\mathcal { U } _ { a b } ^ { \mathrm { r o t } } : = \bigoplus _ { k = 1 } ^ { m } \left( A _ { i , p } ^ { \top } R _ { i } ^ { \top } R _ { j } A _ { j , q } \right) ,\tag{6}
$$

which strictly preserves feature norms while capturing the 3D relative orientation between rays.

Query-frame metric translation RoPE. To encode the query-frame metric displacement $\Delta \mathbf { o } _ { j | i } = R _ { i } ^ { \top } ( \mathbf { o } _ { j } - \mathbf { o } _ { i } )$ illustrated in Figure 2 without baseline-dependent logit growth, we map each translation coordinate into multi-frequency rotary phases. Following RoPE frequency construction and position-interpolation principles (Su et al., 2024; Chen et al., 2023), given predefined minimum and maximum wavelengths $[ \lambda _ { \operatorname* { m i n } } , \lambda _ { \operatorname* { m a x } } ]$ , we construct K logarithmically spaced wavelengths and their corresponding angular frequencies:

$$
\omega _ { k } = \frac { 2 \pi } { \lambda _ { k } } , \qquad \lambda _ { k } = \lambda _ { \mathrm { m i n } } \left( \frac { \lambda _ { \mathrm { m a x } } } { \lambda _ { \mathrm { m i n } } } \right) ^ { k / ( K - 1 ) } , \qquad k = 0 , \dots , K - 1 .\tag{7}
$$

For each Cartesian axis $c \in \{ x , y , z \}$ , the 1D displacement $[ \Delta \mathbf { o } _ { j | i } ] _ { c }$ rotates a 2D feature subspace by angle $\omega _ { k } [ \Delta \mathbf { o } _ { j | i } ] _ { c }$ . The translation operator is then the direct sum over all 3 axes and K frequencies:

$$
\mathcal { U } _ { i j } ^ { \mathrm { t r a n s } } : = \bigoplus _ { c \in \{ x , y , z \} } \bigoplus _ { k = 0 } ^ { K - 1 } \mathrm { R o t } \left( \omega _ { k } [ \Delta \mathbf { o } _ { j | i } ] _ { c } \right) , \qquad \mathrm { w h e r e ~ R o t } ( \theta ) = \left[ \cos \theta _ { \hbar } \quad - \sin \theta \right] ,\tag{8}
$$

which is strictly orthogonal, preserving feature norms and preventing attention logits from scaling with the physical translation baseline.

## 3.3 DISPARITY-ANCHORED SPHERICAL ENCODING

Potential static-scene correspondences add geometric guidance beyond the scene-independent relative camera framing provided by the orthogonal metric pose operator. Existing methods either restrict attention strictly to 2D pixel epipolar lines (Xu et al., 2024) or project metric depth anchors back into the query image plane (Xie et al., 2026a). However, pixel-plane reprojections break the camera-model-agnostic ray interface, and unbounded metric depth ranges require heuristic, dataset-specific anchor choices.

![](images/44ccdaddcc0066f880ecaee81d63597b86816bad815c3f7977d0b240a525263f.jpg)

We instead construct correspondence anchors in spherical ray space using bounded angular disparity fractions $( { \mathrm { F i g } } -$ ure 2). Let ${ \bf u } _ { \infty } \ = \ R _ { i } ^ { \top } R _ { j } { \bf d } _ { j , q } ^ { c }$ be the key ray direction in the query camera frame; for a nonzero baseline, let $\mathbf e = \Delta \bar { \mathbf o } _ { j | i } \bar { \langle \| \Delta \mathbf o _ { j | i } \| _ { 2 } }$ be the epipole unit vector. As 3D scene depth varies from ∞ to 0, points along key ray $( j , q )$ sweep a great-circle arc on the unit sphere connecting $\mathbf { u } _ { \infty }$ and e. We sample L discrete disparity anchors along this epipolar arc at angular fractions $\rho _ { \ell } \in [ 0 , 1 ]$

$$
\mathbf { u } _ { i , \beta _ { \ell } } = \cos ( \rho _ { \ell } \beta _ { \mathrm { m a x } } ) \mathbf { u } _ { \infty } + \sin ( \rho _ { \ell } \beta _ { \mathrm { m a x } } ) \hat { \mathbf { t } } ,
$$

where $\hat { \mathbf { t } } = \frac { \mathbf { e } - ( \mathbf { e } ^ { \top } \mathbf { u } _ { \infty } ) \mathbf { u } _ { \infty } } { \| \mathbf { e } - ( \mathbf { e } ^ { \top } \mathbf { u } _ { \infty } ) \mathbf { u } _ { \infty } \| _ { 2 } } , \quad \beta _ { \operatorname* { m a x } } = \operatorname { a r c c o s } ( \mathbf { e } ^ { \top } \mathbf { u } _ { \infty } )$

(9)

For each anchor $\ell ,$ the Rodrigues rotation $A _ { i , \beta _ { \ell } }$ carries the query-camera optical axis $\mathbf { e } _ { z }$ onto the query-frame direction $\mathbf { u } _ { i , \beta _ { \ell } }$ . The disparity encoding computes the relative

Figure 2: Geometric construction. Blue marks the query/key ray-local frames $A _ { i , q }$ and $A _ { j , k }$ used by rotation PE; green marks the query-frame metric displacement $\Delta \mathbf { o } _ { j \mid i }$ used by translation PE; purple marks disparity PE, whose great-circle arc runs from the transformed key-ray direction $\mathbf { u } _ { \infty }$ to the epipole $\mathbf { e } ,$ with endpoint frames $A _ { i , { \bf u } _ { \scriptscriptstyle \mathrm { o } } }$ ∞ and $A _ { i , \mathrm { e } }$ and sampled anchor directions $\mathbf { u } _ { i , \beta _ { \ell } }$ in between.

rotation $A _ { i , p } ^ { \top } A _ { i , \beta _ { \ell } }$ from query ray ${ \bf d } _ { i , p } ^ { c }$ to candidate direction $\mathbf { u } _ { i , \beta _ { \ell } }$ , repeated across n coordinate triplets:

$$
\mathcal { U } _ { ( i , p ) , ( j , q ) } ^ { \mathrm { d i s p } } : = \bigoplus _ { \ell = 1 } ^ { L } \bigoplus _ { \underbrace { k = 1 } _ { n { \mathrm { t i m e s } } } } ^ { n } \left( A _ { i , p } ^ { \top } A _ { i , \beta _ { \ell } } \right) .\tag{10}
$$

These anchors act as fixed geometric knots rather than predictions of mutually exclusive correspondences. Together, their rotation blocks encode the query ray relative to several fixed directions along the continuous arc, while leaving the network to determine from visual content whether a valid 3D correspondence exists.

Degenerate configurations make the construction in Eq. (9) undefined: the epipole e is undefined when the camera centers coincide, while the great-circle direction <sup>ˆ</sup>t is undefined when e is parallel or antiparallel to $\mathbf { u } _ { \infty }$ . In either case, we collapse all anchor directions to $\mathbf { u } _ { \infty }$ to avoid division by zero.

The complete pairwise MeRoPE operator $\mathcal { U } _ { a b }$ is assembled by the block-diagonal direct sum:

$$
\mathcal { U } _ { a b } = \mathcal { U } _ { ( i , p ) , ( j , q ) } ^ { \mathrm { d i s p } } \oplus \mathcal { U } _ { ( i , p ) , ( j , q ) } ^ { \mathrm { r o t } } \oplus \mathcal { U } _ { i j } ^ { \mathrm { t r a n s } } \oplus \mathcal { U } ^ { \mathrm { n a t i v e } } ,\tag{11}
$$

where $\boldsymbol { \mathcal { U } } ^ { \mathrm { n a t i v e } }$ retains the backbone’s native spatio-temporal RoPE over the temporal and imageplane $( x , y )$ coordinate bands (Figure 3). Thus, MeRoPE satisfies (A+C): it uses the full metric

$T _ { i } ^ { - 1 } T _ { j }$ without scale normalization, while $\mathcal { U } _ { a b } ^ { \top } \mathcal { U } _ { a b } = I$ ensures norm preservation and baselineindependent logits.

Integration into Diffusion Transformers. Following the adapter scheme in UCPE (Zhang et al., 2026a), we integrate MeRoPE into pretrained video Diffusion Transformers through a lightweight parallel attention branch. The base model retains its original selfattention and spatio-temporal representations, while the parallel branch evaluates camera-conditioned attention modulated by $\mathcal { U } _ { a b }$ . The resulting features are mapped back through a zero-initialized linear projection and added residually to the backbone stream, so the pretrained backbone features remain unchanged at initialization.

## 4 EXPERIMENTS

![](images/fe74a8340374b58be838747b8cc818b398c91a4274cbdf8642e2c3c999143af1.jpg)  
Figure 3: Block-diagonal structure of $\mathcal { U } _ { a b } .$ Pairwise attention query-key modulation $\mathbf { q } _ { a } ^ { \top } \mathcal { U } _ { a b } \mathbf { k } _ { b }$ , assembled as the orthogonal direct sum of disparity MinRot $\mathcal { U } ^ { \mathrm { d i s p } }$ , ray-local MinRot $\mathcal { U } ^ { \mathrm { r o t } }$ , translation RoPE $\mathcal { U } ^ { \mathrm { t r a n s } }$ , and native RoPE $\boldsymbol { \mathcal { U } } ^ { \mathrm { n a t i v e } }$

We evaluate MeRoPE on two benchmarks: largebaseline driving video generation on nuScenes (Caesar et al., 2020) (Section 4.2) and multi-lens camera control on PanShot (Zhang et al., 2026a;b) (Section 4.3).

## 4.1 EXPERIMENTAL SETUP

Datasets and training recipe. For driving video generation, training data combines real clips from the nuScenes training set (103.6k clips) with synthetic trajectories rendered in 3D Gaussian environments by World Engine (Li et al., 2026b) (244.1k clips), totaling 347.7k unique training sequences (Figure 4). Candidate trajectories are grouped using k-means and filtered by non-maximum suppression (NMS) with a Frechet distance threshold of´ 1 m to obtain representative motion prototypes. During training, classbalanced sampling draws mini-batches uniformly across trajectory clusters. All models are trained at 288 × 512 resolution with 49 frames, conditioned on the first 9 RGB frames, commanded camera trajectory parameters, and text prompts. Training uses the AdamW optimizer in bfloat16 mixed precision on $6 4 \times \mathrm { { N V I D I A } }$ H20 GPUs for 20k iterations with a global batch size of 64 (details in $\mathsf { A p - }$ pendix B.1).

![](images/fe35fb837911ea0deabc383f7a5f0a152ed79596d6572056ad5af4ab39457ff8.jpg)  
Figure 4: Training trajectory distribution. Bird’s-eye view (BEV) of motion patterns in the training mixture: forward-dominant corridors from nuScenes (blue, 103.6k) and wide lateral/turning maneuvers from World Engine (orange, 244.1k).

Evaluation protocols. We evaluate driving scenes on 128 representative sequences selected from the nuScenes validation and test sets to cover diverse trajectory geometries. Each base trajectory is augmented with left-turn and rightturn variations, yielding 384 test trajectories in total. Following standard visual odometry evaluation, camera poses are extracted from generated frames using VGGT-Ω (Wang et al., 2026a). Quantitative metrics report geodesic rotation error $( \mathrm { r o t } ^ { \circ } )$ , scale-aligned relative translation error $( \mathrm { t r \% } )$ , pairwise joint-pose AUC (AUC@3, AUC@10), and camera motion consistency (CamMC). Visual quality is evaluated separately on 512 nuScenes clips using FID over 20,480 predicted frames and FVD over 512 generated videos after excluding the 9-frame conditioning prefix. Appendix B.2 defines all metrics mathematically.

Matched-backbone baselines. For a matched-backbone comparison, all methods share the same Wan2.2 TI2V-5B backbone (Team Wan et al., 2025; Wan Team, 2025), initialization, training mixture, optimization schedule, and attention head dimension $( d = 1 2 8 )$ . All camera-conditioned methods are retrained with the same 15-block alternating injection schedule. GTA, PRoPE, and UCPE use a common fixed global translation scale estimated solely from training-set motion statistics. We compare against: (i) the backbone without camera PE, (ii) GTA (Miyato et al., 2024), (iii) PRoPE (Li et al., 2025), (iv) UCPE (Zhang et al., 2026a), (v) RayNova (Xie et al., 2026b), (vi) RayRoPE (Wu et al., 2026), (vii) URoPE (Xie et al., 2026a), and (viii) a CameraCtrl-style baseline (He et al., 2025), adapted to the DiT backbone by adding ray features before the query and key projections following Lyra 2.0 (Shen et al., 2026).

Table 2: Camera pose controllability on the nuScenes driving benchmark. Camera-conditioned methods are ordered by CamMC from highest to lowest. Best in bold; second-best is underlined.
<table><tr><td>Method</td><td> $\mathrm { r o t } ^ { \circ } \downarrow$ </td><td> $\operatorname { t r } \% \downarrow$ </td><td>AUC@3↑</td><td>AUC@10↑</td><td>CamMC↓</td><td>FID↓</td><td>FVD↓</td></tr><tr><td>No camera PE</td><td>5.57</td><td>5.94</td><td>17.21</td><td>52.05</td><td>7.96</td><td>21.06</td><td>153.31</td></tr><tr><td>RayRoPE (Wu et al., 2026)</td><td>1.83</td><td>2.97</td><td>36.56</td><td>76.62</td><td>2.78</td><td>21.21</td><td>150.26</td></tr><tr><td>PRoPE (Li et al., 2025)</td><td>1.53</td><td>3.10</td><td>42.11</td><td>78.58</td><td>2.67</td><td>20.40</td><td>142.16</td></tr><tr><td>RayNova (Xie et al., 2026b)</td><td>1.56</td><td>3.02</td><td>41.19</td><td>78.42</td><td>2.66</td><td>20.15</td><td>144.91</td></tr><tr><td>CameraCtrl-style (He et al., 2025)</td><td>1.54</td><td>3.07</td><td>42.57</td><td>78.90</td><td>2.63</td><td>20.13</td><td>142.23</td></tr><tr><td>URoPE (Xie et al., 2026a)</td><td>1.66</td><td>2.67</td><td>44.41</td><td>79.92</td><td>2.53</td><td>19.98</td><td>135.63</td></tr><tr><td>GTA (Miyato et al., 2024)</td><td>1.44</td><td>3.03</td><td>40.53</td><td>77.72</td><td>2.49</td><td>20.00</td><td>141.84</td></tr><tr><td>UCPE (Zhang et al., 2026a)</td><td>1.39</td><td>2.95</td><td>41.21</td><td>78.38</td><td>2.45</td><td>20.20</td><td>136.40</td></tr><tr><td>MeRoPE</td><td>1.37</td><td>2.84</td><td>42.11</td><td>79.47</td><td>2.32</td><td>19.82</td><td>134.15</td></tr></table>

Table 3: Ablation of camera-encoding components on nuScenes. Best in bold; second-best is underlined.
<table><tr><td></td><td colspan="3">Components</td><td colspan="5">Overall Metrics</td></tr><tr><td>Method</td><td>camera rot.</td><td></td><td>metric trans. disparity anchor</td><td> $\mathrm { r o t } ^ { \circ } \downarrow$ </td><td>tr%↓</td><td></td><td>AUC@3↑ AUC@10↑</td><td>CamMC↓</td></tr><tr><td>MeRoPE</td><td>√</td><td>√</td><td>√</td><td>1.37</td><td>2.84</td><td>42.11</td><td>79.47</td><td>2.32</td></tr><tr><td>Rot. + trans.</td><td>√</td><td>√</td><td></td><td>1.36</td><td>3.09</td><td>38.82</td><td>77.50</td><td>2.40</td></tr><tr><td>Disparity + trans.</td><td></td><td>√</td><td>√</td><td>1.66</td><td>2.88</td><td>40.60</td><td>78.82</td><td>2.56</td></tr><tr><td>Disparity + rot.</td><td>√</td><td></td><td>√</td><td>1.48</td><td>4.31</td><td>44.42</td><td>79.78</td><td>3.66</td></tr><tr><td>Disparity only</td><td></td><td></td><td>√</td><td>1.59</td><td>4.01</td><td>43.19</td><td>79.53</td><td>3.56</td></tr><tr><td>Rotation only</td><td>√</td><td></td><td></td><td>1.41</td><td>4.01</td><td>42.52</td><td>78.33</td><td>3.35</td></tr><tr><td>Translation only</td><td></td><td>√</td><td></td><td>2.59</td><td>3.09</td><td>30.64</td><td>73.50</td><td>3.50</td></tr></table>

## 4.2 POSE-CONTROLLED VIDEO GENERATION ON DRIVING BENCHMARKS

Main comparison on driving benchmarks. Table 2 compares camera control across methods. We use CamMC as the primary metric because it jointly evaluates rotation and translation trajectories, whereas thresholded AUC is less stable and less sensitive to translation magnitude. MeRoPE records the lowest CamMC (2.32); UCPE is the strongest baseline at 2.45. It also achieves the best rotation accuracy and ranks second on both tr% and AUC@10. URoPE leads tr% and the two AUC metrics, while the CameraCtrl-style baseline ranks second on AUC@3. The weaker pose results without camera PE show the benefit of explicit camera geometry for controllable generation. MeRoPE also attains the lowest FID and FVD point estimates, so its control gains do not coincide with worse measured visual quality. Figure 5 compares the commanded and recovered paths under three camera controls together with the corresponding generated frames. Appendices D.1 and D.2 provide matched-backbone comparisons and additional nuScenes scenes, respectively.

Ablation of encoding components. Table 3 shows that MeRoPE’s three sub-blocks play distinct, complementary roles while keeping d = 128 fixed through identity padding. Removing metric translation degrades translation fidelity: all variants without it have substantially worse tr% and CamMC, even though Disparity + rot. attains the highest AUC because AUC evaluates translation direction but not magnitude. Ray-local rotation primarily controls orientation: adding it to Disparity + trans. reduces rotation error from 1.66<sup>◦</sup> to 1.37<sup>◦</sup>. Disparity anchors instead refine cross-view tracking: adding them to Rot. + trans. improves both AUC metrics and lowers tr% and CamMC while leaving rotation essentially unchanged. Together, these complementary gains yield the ful model’s lowest tr% and CamMC.

![](images/2219dfddf6dc23b8393435a2478fb5e1a02ca396a30c9317ffb1d343f484e8a8.jpg)  
Figure 5: Qualitative video generation under camera pose control. Visual results on two representative nuScenes scenes under three commanded trajectories (original, left turn, right turn). In each row, the leftmost plot displays the bird’s-eye view of the commanded path (solid) and the path recovered from the generated video by VGGT-Ω (dashed), followed by generated video frames sampled at t ∈ {0, 16, 32, 48}.

Appendices C.1 and C.3 analyze adapter capacity, injection strategies, layer-wise modulation, and real-world grounding with retrieved history.

## 4.3 CAMERA-DIVERSE EVALUATION ON PANSHOT

To complement the fixed-intrinsic nuScenes setting, we evaluate MeRoPE on the PanShot benchmark (Zhang et al., 2026a;b), which spans pinhole, wide-angle, and fisheye lenses under the unified camera model. Following the matched protocol, we retrain every compared method from the same Wan2.1-T2V-1.3B backbone for 10k steps on 8 × NVIDIA H20 GPUs with identical spatial attention adapters and evaluate them on the official jitter-filtered test split (272 clips). We compare camera-conditioning poses with near-depth normalization (a) enabled or (b) disabled; the latter preserves raw physical displacements during training and inference, while evaluation still applies the per-trajectory translation normalization in Appendix B.2. Poses are extracted using VGGT-Ω (Wang et al., 2026a).

Table 4 shows that MeRoPE’s advantage on PanShot lies in translation control rather than rotation. Across both conditioning-pose settings, it achieves the lowest TransErr and CamMC, while UCPE and MeRoPE remain close on RotErr. The wider separation between UCPE and PRoPE than on fixed-intrinsic nuScenes is consistent with the benefit of calibrated ray representations when camera optics vary. Using raw conditioning displacements during training and generation increases TransErr and CamMC for all methods, but MeRoPE retains the lowest errors and the smallest increase. Figure 6 shows generated sequences across varying fields of view and lens distortions.

Table 4: Camera pose control on PanShot across diverse camera optics. Evaluated on 272 test clips under the unified camera model (C = 8, 1 × 192, VGGT-Ω). (a) Conditioning-pose near-depth normalization enabled. (b) Conditioning-pose near-depth normalization disabled. GTA and PRoPE retain the latitude-up map; UCPE and MeRoPE do not. Lower is better. Best in each panel in bold; second-best is underlined.  
(a) Conditioning norm. on
<table><tr><td>Method</td><td>RotErr↓</td><td>TransErr↓</td><td>CamMC↓</td></tr><tr><td>GTA</td><td>5.14</td><td>19.85</td><td>22.59</td></tr><tr><td>PRoPE</td><td>5.10</td><td>20.38</td><td>23.02</td></tr><tr><td>UCPE</td><td>4.22</td><td>15.15</td><td>17.50</td></tr><tr><td>MeRoPE</td><td>4.57</td><td>10.99</td><td>13.88</td></tr></table>

(b) Conditioning norm. off
<table><tr><td>Method</td><td>RotErr↓</td><td>TransErr↓</td><td>CamMC↓</td></tr><tr><td>GTA</td><td>5.30</td><td>22.70</td><td>25.34</td></tr><tr><td>PRoPE</td><td>5.11</td><td>23.04</td><td>25.58</td></tr><tr><td>UCPE</td><td>4.40</td><td>17.91</td><td>20.13</td></tr><tr><td>MeRoPE</td><td>4.82</td><td>12.66</td><td>15.56</td></tr></table>

![](images/4ae0506a20b384a0e55d0ac81aa2199c23812db90de393a9d5e7d2ea82da7bbd.jpg)  
Figure 6: PanShot generations across camera optics. Rows 1–4 span (FoV, ξ) from (97<sup>◦</sup>, 0.00) to (173<sup>◦</sup>, 1.66), covering pinhole through fisheye cameras.

## 5 CONCLUSION

We introduced MeRoPE, a norm-preserving relative camera encoding for camera-controlled video generation under real-world metric motion. Our analysis identifies baseline-dependent logit and feature-norm growth in homogeneous projective encodings and formalizes a three-way trade-off: retaining the complete metric relative pose, factorizing strictly per token, and preserving feature norms cannot be achieved simultaneously. MeRoPE resolves this trade-off by relaxing per-token factorization through query-camera grouping, representing relative ray orientation and raw metric translation with orthogonal blocks, and adding disparity-anchored spherical rotations as geometric knots for cross-view relations.

Across large-baseline nuScenes and camera-diverse PanShot, MeRoPE achieves the lowest CamMC among the compared encodings; it also attains the lowest nuScenes FID/FVD point estimates and the lowest PanShot TransErr under both conditioning settings. These results support norm-preserving metric camera encoding as a stable basis for camera control across physical baselines and camera optics. Query-camera grouping introduces additional local camera-attention cost; reducing this overhead while retaining metric sensitivity is an important direction for future work.

## AI USE STATEMENT

Generative AI tools assisted code implementation and language editing. All outputs were reviewed and verified by the authors, who take full responsibility for this paper.

## REFERENCES

Jianhong Bai, Menghan Xia, Xiao Fu, Xintao Wang, Lianrui Mu, Jinwen Cao, Zuozhu Liu, Haoji Hu, Xiang Bai, Pengfei Wan, and Di Zhang. ReCamMaster: Camera-controlled generative rendering from a single video. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 14834–14844, 2025. doi: 10.1109/ICCV51701.2025.01376. URL https://arxiv.org/abs/2503.11647.

Holger Caesar, Varun Bankiti, Alex H. Lang, Sourabh Vora, Venice Erin Liong, Qiang Xu, Anush Krishnan, Yu Pan, Giancarlo Baldan, and Oscar Beijbom. nuscenes: A multimodal dataset for autonomous driving. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 11621–11631, 2020. doi: 10.1109/CVPR42600.2020.01164. URL https://arxiv.org/abs/1903.11027.

Shouyuan Chen, Sherman Wong, Liangjian Chen, and Yuandong Tian. Extending context window of large language models via positional interpolation. arXiv preprint arXiv:2306.15595, 2023. URL https://arxiv.org/abs/2306.15595.

Gene Chou, Charles Herrmann, Kyle Genova, Boyang Deng, Songyou Peng, Bharath Hariharan, Jason Y. Zhang, Noah Snavely, and Philipp Henzler. CityRAG: Stepping into a city via spatiallygrounded video generation. arXiv preprint arXiv:2604.19741, 2026. URL https://arxiv. org/abs/2604.19741.

Gerald B. Folland. A Course in Abstract Harmonic Analysis. CRC Press, 2nd edition, 2016.

Hao He, Yinghao Xu, Yuwei Guo, Gordon Wetzstein, Bo Dai, Hongsheng Li, and Ceyuan Yang. CameraCtrl: Enabling camera control for video diffusion models. In International Conference on Learning Representations, 2025. URL https://arxiv.org/abs/2404.02101.

Roger A. Horn and Charles R. Johnson. Matrix Analysis. Cambridge University Press, 2nd edition, 2012.

Seonghyun Jin, Youngmin Kim, Sunwoo Park, and Jong Chul Ye. CRePE: Curved ray expectation positional encoding for unified-camera-controlled video generation. arXiv preprint arXiv:2605.12938, 2026. URL https://arxiv.org/abs/2605.12938.

Xin Kong, Shikun Liu, Xiaoyang Lyu, Marwan Taher, Xiaojuan Qi, and Andrew J. Davison. Escher-Net: A generative model for scalable view synthesis. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 9503–9513, 2024. doi: 10.1109/CVPR52733. 2024.00908. URL https://arxiv.org/abs/2402.03908.

Chunyang Li, Yuanbo Yang, Jiahao Shao, Hongyu Zhou, Katja Schwarz, and Yiyi Liao. ReRoPE: Repurposing RoPE for relative camera control. In ACM SIGGRAPH 2026 Conference Papers, pp. 1–11, 2026a. doi: 10.1145/3799902.3811226. URL https://arxiv.org/abs/2602. 08068.

Ruilong Li, Brent Yi, Junchen Liu, Hang Gao, Yi Ma, and Angjoo Kanazawa. Cameras as relative positional encoding. In Advances in Neural Information Processing Systems, pp. 18020–18045, 2025. doi: 10.52202/085713-0540. URL https://arxiv.org/abs/2507.10496.

Tianyu Li, Li Chen, Caojun Wang, Haochen Liu, Kashyap Chitta, Zhenjie Yang, Yuhang Lu, Naisheng Ye, Yihang Qiu, Yufei Wang, Luoxi Zou, Jiaxin Peng, Jin Pan, Zhaoyu Su, Andrei Bursuc, Shengbo Eben Li, Andreas Geiger, Peng Su, and Hongyang Li. World Engine: Towards the era of post-training for autonomous driving. arXiv preprint arXiv:2606.19836, 2026b. URL https://arxiv.org/abs/2606.19836.

Takeru Miyato, Bernhard Jaeger, Max Welling, and Andreas Geiger. GTA: A geometry-aware attention mechanism for multi-view transformers. In International Conference on Learning Representations, 2024. URL https://arxiv.org/abs/2310.10375.

Xuanchi Ren, Tianchang Shen, Jiahui Huang, Huan Ling, Yifan Lu, Merlin Nimier-David, Thomas Muller, Alexander Keller, Sanja Fidler, and Jun Gao. Gen3C: 3d-informed world-consistent video¨ generation with precise camera control. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 6121–6132, 2025. doi: 10.1109/CVPR52734.2025. 00574. URL https://arxiv.org/abs/2503.03751.

Junyoung Seo, Hyunwook Choi, Minkyung Kwon, Jinhyeok Choi, Siyoon Jin, Gayoung Lee, Junho Kim, JoungBin Lee, Geonmo Gu, Dongyoon Han, Sangdoo Yun, Seungryong Kim, and Jin-Hwa Kim. Grounding world simulation models in a real-world metropolis. In European Conference on Computer Vision, 2026. URL https://arxiv.org/abs/2603.15583. Spotlight; Seoul World Model.

Tianchang Shen, Sherwin Bahmani, Kai He, Sangeetha Grama Srinivasan, Tianshi Cao, Jiawei Ren, Ruilong Li, Zian Wang, Nicholas Sharp, Zan Gojcic, Sanja Fidler, Jiahui Huang, Huan Ling, Jun Gao, and Xuanchi Ren. Lyra 2.0: Explorable generative 3D worlds. arXiv preprint arXiv:2604.13036, 2026. URL https://arxiv.org/abs/2604.13036.

Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. RoFormer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063, 2024. doi: 10.1016/j.neucom.2023.127063. URL https://arxiv.org/abs/2104.09864.

Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025. URL https://arxiv.org/abs/2503.20314.

Wan Team. Wan2.2: Open-source video generation models. Official GitHub repository, 2025. URL https://github.com/Wan-Video/Wan2.2.

Jianyuan Wang, Minghao Chen, Shangzhan Zhang, Nikita Karaev, Johannes L. Schonberger, Patrick¨ Labatut, Piotr Bojanowski, David Novotny, Andrea Vedaldi, and Christian Rupprecht. VGGT-Ω. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 21486–21499, 2026a.

Yiming Wang, Qihang Zhang, Shengqu Cai, Tong Wu, Jan Ackermann, Zhengfei Kuang, Yang Zheng, Frano Rajic, Siyu Tang, and Gordon Wetzstein. BulletTime: Decoupled control of timeˇ and camera pose for video generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 18319–18330, 2026b.

Zhouxia Wang, Ziyang Yuan, Xintao Wang, Yaowei Li, Tianshui Chen, Menghan Xia, Ping Luo, and Ying Shan. MotionCtrl: A unified and flexible motion controller for video generation. In ACM SIGGRAPH Conference Papers, pp. 1–11, 2024. doi: 10.1145/3641519.3657518. URL https://arxiv.org/abs/2312.03641.

Yu Wu, Minsik Jeon, Jen-Hao Rick Chang, Oncel Tuzel, and Shubham Tulsiani. RayRoPE: Projective ray positional encoding for multi-view attention. In European Conference on Computer Vision, 2026. URL https://arxiv.org/abs/2601.15275. Spotlight.

Chendong Xiang, Jiajun Liu, Jintao Zhang, Xiao Yang, Zhengwei Fang, Shizun Wang, Zijun Wang, Yingtian Zou, Hang Su, and Jun Zhu. Geometry-aware rotary position embedding for consistent video world model. arXiv preprint arXiv:2602.07854, 2026. URL https://arxiv.org/ abs/2602.07854.

Yichen Xie, Depu Meng, Chensheng Peng, Yihan Hu, Quentin Herau, Masayoshi Tomizuka, and Wei Zhan. URoPE: Universal relative position embedding across geometric spaces. In European Conference on Computer Vision, 2026a. URL https://arxiv.org/abs/2604.18747. Spotlight.

Yichen Xie, Chensheng Peng, Mazen Abdelfattah, Yihan Hu, Jiezhi Yang, Eric Higgins, Ryan Brigden, Masayoshi Tomizuka, and Wei Zhan. RayNova: Scale-temporal autoregressive world modeling in ray space. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 25426–25437, 2026b.

Dejia Xu, Weili Nie, Chao Liu, Sifei Liu, Jan Kautz, Zhangyang Wang, and Arash Vahdat. CamCo: Camera-controllable 3D-consistent image-to-video generation. arXiv preprint arXiv:2406.02509, 2024. URL https://arxiv.org/abs/2406.02509.

Mark Yu, Wenbo Hu, Jinbo Xing, and Ying Shan. TrajectoryCrafter: Redirecting camera trajectory for monocular videos via diffusion models. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pp. 100–111, 2025a. doi: 10.1109/ICCV51701.2025.00017. URL https://arxiv.org/abs/2503.05638.

Wangbo Yu, Jinbo Xing, Li Yuan, Wenbo Hu, Xiaoyu Li, Zhipeng Huang, Xiangjun Gao, Tien-Tsin Wong, Ying Shan, and Yonghong Tian. ViewCrafter: Taming video diffusion models for high-fidelity novel view synthesis. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025b. doi: 10.1109/TPAMI.2025.3613256. URL https://arxiv.org/abs/ 2409.02048.

Yifan Zhan, Zhengqing Chen, Qingjie Wang, Zhuo He, Muyao Niu, Xiaoyang Guo, Wei Yin, Weiqiang Ren, Qian Zhang, and Yinqiang Zheng. Composing driving worlds through disentangled control for adversarial scenario generation. In European Conference on Computer Vision, 2026. URL https://arxiv.org/abs/2603.12864.

Cheng Zhang, Boying Li, Meng Wei, Yan-Pei Cao, Camilo Cruz Gambardella, Dinh Phung, and Jianfei Cai. Unified camera positional encoding for controlled video generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 38027–38037, 2026a.

Cheng Zhang, Boying Li, Meng Wei, Yan-Pei Cao, Camilo Cruz Gambardella, Dinh Phung, and Jianfei Cai. UCPE: Official implementation and PanShot dataset. GitHub repository and Hugging Face dataset, 2026b. URL https://github.com/chengzhag/UCPE. Commit d992f1807803ba99331e807e8f018ed552886afd.

## A IMPOSSIBILITY OF UNITARY RELATIVE PER-TOKEN ENCODINGS FOR METRIC TRANSLATION

We show that no strictly per-token, unitary (or orthogonal) positional operator can completely capture metric relative translation. This result formalizes the trilemma introduced in Section 1 and motivates MeRoPE’s use of query-camera grouping.

## A.1 PROBLEM FORMULATION AND HYPOTHESES

Let $T _ { i } = ( R _ { i } , \mathbf { o } _ { i } ) \in S E ( 3 )$ denote the camera-to-world pose of view i, where $R _ { i } \in \mathrm { S O } ( 3 )$ and ${ \bf o } _ { i } ~ \in ~ \mathbb { R } ^ { 3 }$ . The intrinsic relative pose between views i and $j ,$ invariant to any global coordinate change $G \in S E ( 3 )$ , is given by

$$
T _ { i } ^ { - 1 } T _ { j } = \left( R _ { i } ^ { \top } R _ { j } , R _ { i } ^ { \top } ( \mathbf { o } _ { j } - \mathbf { o } _ { i } ) \right) .\tag{12}
$$

We examine the existence of a matrix-valued positional embedding operator $U : S E ( 3 ) \to \mathbb { C } ^ { d \times d }$ (or $\mathbb { R } ^ { d \times d } )$ satisfying four standard axioms of relative positional attention:

(H1) Multiplicative per-token factorization: The camera positional operator acts independently on each query and key token via its own pose:

$$
\widetilde { \mathbf { q } } _ { i } = U ( T _ { i } ) \mathbf { q } _ { i } , \qquad \widetilde { \mathbf { k } } _ { j } = U ( T _ { j } ) \mathbf { k } _ { j } , \qquad s _ { i j } = \frac { \mathbf { q } _ { i } ^ { * } U ( T _ { i } ) ^ { * } U ( T _ { j } ) \mathbf { k } _ { j } } { \sqrt { d } } .\tag{13}
$$

(H2) Exact relative-pose dependence: The pairwise interaction depends strictly and exclusively on the relative pose $T _ { i } ^ { - 1 } T _ { j }$

$$
U ( T _ { i } ) ^ { * } U ( T _ { j } ) = U ( T _ { i } ^ { - 1 } T _ { j } ) , \qquad \forall T _ { i } , T _ { j } \in S E ( 3 ) .\tag{14}
$$

(H3) Norm preservation (unitarity): $U ( T )$ is unitary (or orthogonal in the real case), so that $\| U ( T ) \mathbf { x } \| _ { 2 } = \| \mathbf { x } \| _ { 2 }$ for all $\mathbf { x } \in \mathbb { C } ^ { d } , \mathrm { i . e . }$ 9

$$
U ( T ) ^ { * } U ( T ) = I _ { d } , \qquad \forall T \in S E ( 3 ) .\tag{15}
$$

(H4) Finite-dimensional continuity: The feature dimension is finite $( d < \infty )$ , and the mapping $T \mapsto U ( T )$ is continuous with respect to the standard Lie group topology on $S E ( 3 )$ .

## A.2 THE IMPOSSIBILITY THEOREM

Theorem 1 (Translation Blindness of Unitary Relative Per-Token Encodings). Let $U : S E ( 3 ) $ $\mathrm { U } ( d )$ be any continuous map satisfying hypotheses (H1)–(H4). Then for every pure metric translation $\mathbf { t } \in \mathbb { R } ^ { 3 }$

$$
U ( I _ { 3 } , \mathbf { t } ) = I _ { d } .\tag{16}
$$

Consequently, for any arbitrary camera pose (R, t) ∈ SE(3),

$$
U ( R , { \bf t } ) = U ( R , { \bf 0 } ) .\tag{17}
$$

The positional operator U is identically trivial on metric translations, failing to distinguish any non-zero relative translation $\Delta \mathbf { o } _ { j | i } = R _ { i } ^ { \top } ( \mathbf { o } _ { j } - \mathbf { o } _ { i } ) \neq \mathbf { 0 }$

Proof. Step 1: Homomorphism deduction. First, we show that (H2) and (H3) enforce U to be a group representation of $S \bar { E } ( 3 )$ , i.e., $U ( g h ) = U ( g ) U ( h )$ for all $g , h \in S E ( 3 )$

• Setting $T _ { i } = T _ { i } = I$ in Eq. (14) yields $U ( I ) ^ { * } U ( I ) = U ( I )$ . On the other hand, Eq. (15) with $T = I$ gives $U ( I ) ^ { * } U ( \bar { I } ) = \bar { I _ { d } }$ . The two right-hand sides must coincide, so $U ( I ) \bar { = } I _ { d }$

• Setting $T _ { j } = I$ in Eq. (14) yields $U ( T _ { i } ) ^ { * } U ( I ) = U ( T _ { i } ^ { - 1 } )$ . Since $U ( I ) = I _ { d }$ from the first item, the left-hand side reduces to $U ( T _ { i } ) ^ { * }$ , and by unitarity $( U ( T _ { i } ) ^ { - 1 } = U ( T _ { i } ) ^ { * } )$ we conclude $U ( T _ { i } ^ { - 1 } ) = U ( T _ { i } ) ^ { * } = U ( T _ { i } ) ^ { - 1 }$

• For any $g , h \in S E ( 3 )$ , set $T _ { i } = g$ and $T _ { j } = g h$ . Substituting into Eq. (14) gives:

$$
U ( g ) ^ { * } U ( g h ) = U ( g ^ { - 1 } ( g h ) ) = U ( h ) .\tag{18}
$$

Multiplying on the left by $U ( g )$ and using $U ( g ) U ( g ) ^ { * } = I _ { d }$ , we conclude that

$$
U ( g h ) = U ( g ) U ( h ) , \qquad \forall g , h \in S E ( 3 ) .\tag{19}
$$

Step 2: Commutativity of the translation subgroup. Let $U _ { \mathrm { t r } } ( \mathbf { t } ) : = U ( I _ { 3 } , \mathbf { t } )$ denote the pure translation operator for $\mathbf { \dot { t } } \in \mathbb { R } ^ { 3 }$ . In $S E ( 3 )$ , pure translations commute: $( I _ { 3 } , { \bf s } ) ( I _ { 3 } , { \bf t } ) = ( I _ { 3 } , { \bf s } + { \bf t } ) =$ $( I _ { 3 } , \mathbf { t } ) ( I _ { 3 } , \mathbf { s } )$ . Applying the homomorphism in Eq. (19):

$$
U _ { \mathrm { t r } } ( \mathbf { s } ) U _ { \mathrm { t r } } ( \mathbf { t } ) = U { \big ( } ( I _ { 3 } , \mathbf { s } ) ( I _ { 3 } , \mathbf { t } ) { \big ) } = U ( I _ { 3 } , \mathbf { s } + \mathbf { t } ) = U { \big ( } ( I _ { 3 } , \mathbf { t } ) ( I _ { 3 } , \mathbf { s } ) { \big ) } = U _ { \mathrm { t r } } ( \mathbf { t } ) U _ { \mathrm { t r } } ( \mathbf { s } ) = U _ { \mathrm { t r } } ( \mathbf { s } + \mathbf { t } ) .
$$

Thus, $\{ U _ { \mathrm { t r } } ( \mathbf { t } ) : \mathbf { t } \in \mathbb { R } ^ { 3 } \}$ is a family of mutually commuting unitary matrices.

(20)

Step 3: Simultaneous unitary diagonalization. Every unitary matrix is normal $\begin{array} { r } { ( U ^ { * } U = U U ^ { * } = } \end{array}$ I). Step 2 shows that the operators $\{ U _ { \mathrm { t r } } ( \mathbf { t } ) : \mathbf { t } \in \bar { \mathbb { R } } ^ { 3 } \}$ pairwise commute, and each is unitary by (H3), hence normal. The spectral theorem for commuting families of normal operators (Horn & Johnson, 2012, Theorem 2.5.5), which applies to families of arbitrary cardinality, then guarantees a single, t-independent unitary change-of-basis matrix $V \in \operatorname { U } ( d )$ that simultaneously diagonalizes all $U _ { \mathrm { t r } } ( \mathbf { t } )$

$$
V ^ { * } U _ { \mathrm { t r } } ( \mathbf { t } ) V = \mathrm { d i a g } \left( \chi _ { 1 } ( \mathbf { t } ) , \chi _ { 2 } ( \mathbf { t } ) , \dots , \chi _ { d } ( \mathbf { t } ) \right) .\tag{21}
$$

By (H4), for each $a \in \{ 1 , \ldots , d \}$ , the diagonal entry $\chi _ { a } : \mathbb { R } ^ { 3 } \to \mathbb { C }$ is continuous. By (H3), $| \chi _ { a } ( \mathbf { t } ) | = 1$ . It remains to show that each entry is multiplicative in t. Conjugating Eq. (20) by V and inserting $V V ^ { * } = I _ { d }$ between $U _ { \mathrm { t r } } ( \mathbf { s } )$ and $U _ { \mathrm { t r } } ( \mathbf { t } )$

$$
V ^ { * } U _ { \mathrm { t r } } ( \mathbf { s } + \mathbf { t } ) V = \left( V ^ { * } U _ { \mathrm { t r } } ( \mathbf { s } ) V \right) \left( V ^ { * } U _ { \mathrm { t r } } ( \mathbf { t } ) V \right) .\tag{22}
$$

By Eq. (21), the left-hand side equals diag $\big ( \chi _ { 1 } ( \mathbf { s + t } ) , \ldots , \chi _ { d } ( \mathbf { s + t } ) \big )$ , and the right-hand side equals diag $\big ( \chi _ { 1 } ( \mathbf { s } ) \chi _ { 1 } ( \mathbf { t } ) , \dots , \chi _ { d } ( \mathbf { s } ) \chi _ { d } ( \mathbf { t } ) \big )$ Comparing diagonal entries, each entry satisfies the multiplicative Cauchy character equation:

$$
\chi _ { a } ( \mathbf { s } + \mathbf { t } ) = \chi _ { a } ( \mathbf { s } ) \chi _ { a } ( \mathbf { t } ) , \qquad \forall \mathbf { s } , \mathbf { t } \in \mathbb { R } ^ { 3 } .\tag{23}
$$

Step 4: Character form and finite spatial spectrum. Fixing an index a and dropping it, we classify the continuous maps $\chi : ( \mathbb { R } ^ { 3 } , + ) \to { \overline { { \mathbb { T } } } }$ , where $\mathbb { T } = \{ z \in \mathbb { C } : | z | = 1 \}$ }, satisfying the multiplicative character equation

$$
\chi ( \mathbf { s } + \mathbf { t } ) = \chi ( \mathbf { s } ) \chi ( \mathbf { t } ) , \qquad | \chi | = 1 ,\tag{24}
$$

whose only solutions are spatial plane waves; this is the classical classification of continuous characters of $( \mathbb { R } ^ { 3 } , + )$ (Folland, 2016, Theorem 4.6 and Corollary 4.8):

$$
\chi _ { a } ( \mathbf { t } ) = \exp ( i \omega _ { a } ^ { \top } \mathbf { t } ) , \qquad \mathrm { f o r ~ a ~ u n i q u e ~ f r e q u e n c y ~ v e c t o r } ~ \omega _ { a } \in \mathbb { R } ^ { 3 } .\tag{25}
$$

This classification enables the spherical-orbit contradiction below. Consequently, in the eigenbasis of $V$ , every translation operator takes the standard multi-frequency phase form:

$$
V ^ { * } U _ { \mathrm { t r } } ( \mathbf { t } ) V = \mathrm { d i a g } \big ( \exp ( i \omega _ { 1 } ^ { \top } \mathbf { t } ) , \dots , \exp ( i \omega _ { d } ^ { \top } \mathbf { t } ) \big ) .\tag{26}
$$

Because the feature dimension is finite $( d < \infty )$ , the active frequency spectrum

$$
\Omega : = \{ \omega _ { 1 } , \omega _ { 2 } , \ldots , \omega _ { d } \} \subset \mathbb { R } ^ { 3 }\tag{27}
$$

contains at most d discrete frequency vectors $( | \Omega | \leq d )$

## Step 5: Semi-direct product conjugation and spherical orbit contradiction.

A lemma on rotation invariance. If a finite set $\Omega \subset \mathbb { R } ^ { 3 }$ is $\mathrm { S O ( 3 ) }$ -invariant, i.e. $R \Omega = \Omega$ for all $R \in \mathrm { S O } ( 3 )$ , then $\boldsymbol { \Omega } = \{ \mathbf { 0 } \}$ . Indeed, the orbit of any non-zero $\omega \in \Omega$ under $\mathrm { S O ( 3 ) }$ is the entire sphere $\{ \mathbf { u } \in \mathbb { R } ^ { 3 } : \| \mathbf { u } \| _ { 2 } = \| \pmb { \omega } \| _ { 2 } \}$ , an uncountably infinite set; invariance forces this whole sphere into Ω, which finiteness forbids.

Let $U _ { \mathrm { r o t } } ( R ) : = U ( R , { \bf 0 } )$ denote the pure rotation operator for $R \in \mathrm { S O ( 3 ) }$ . In $S E ( 3 )$ , rotations and translations obey the semi-direct product conjugation:

$$
( I _ { 3 } , R \mathbf { t } ) = ( R , \mathbf { 0 } ) ( I _ { 3 } , \mathbf { t } ) ( R , \mathbf { 0 } ) ^ { - 1 } , \qquad \forall R \in \mathrm { S O } ( 3 ) , \mathbf { t } \in \mathbb { R } ^ { 3 } .\tag{28}
$$

Applying the homomorphism U to Eq. (28) gives the operator-level conjugation

$$
U _ { \mathrm { t r } } ( R \mathbf { t } ) = U _ { \mathrm { r o t } } ( R ) U _ { \mathrm { t r } } ( \mathbf { t } ) U _ { \mathrm { r o t } } ( R ) ^ { - 1 } .\tag{29}
$$

Since t is a free variable ranging over $\mathbb { R } ^ { 3 }$ , replacing it by $R ^ { - 1 } { \bf t }$ and then multiplying both sides on the right by $U _ { \mathrm { r o t } } ( R )$ (the factors $U _ { \mathrm { r o t } } ( R ) ^ { - 1 } U _ { \mathrm { r o t } } \mathbf { \bar { ( } } R ) \stackrel { . } { = } I _ { d }$ cancel) recasts the same fact in the commutation form

$$
U _ { \mathrm { t r } } ( \mathbf { t } ) U _ { \mathrm { r o t } } ( R ) = U _ { \mathrm { r o t } } ( R ) U _ { \mathrm { t r } } ( R ^ { - 1 } \mathbf { t } ) ,\tag{30}
$$

which is the form we use below to push $U _ { \mathrm { r o t } } ( R )$ past $U _ { \mathrm { t r } } ( \mathbf { t } )$ when acting on an eigenvector. For any frequency $\omega \in \Omega$ , define its joint eigenspace, and set $E _ { \omega } = \{ \mathbf { 0 } \}$ for $\omega \not \in \Omega$

$$
\begin{array} { r } { E _ { \omega } = \big \{ \mathbf { v } \in \mathbb { C } ^ { d } : U _ { \mathrm { t r } } ( \mathbf { t } ) \mathbf { v } = \exp ( i \omega ^ { \top } \mathbf { t } ) \mathbf { v } , \ \forall \mathbf { t } \in \mathbb { R } ^ { 3 } \big \} , \quad E _ { \omega } = \{ \mathbf { 0 } \} \mathrm { ~ i f ~ } \omega \not \in \Omega . } \end{array}\tag{31}
$$

For any non-zero $\mathbf { v } \in E _ { \omega }$ , we apply Eq. (30) to evaluate the translation action on $U _ { \mathrm { r o t } } ( R ) \mathbf { v } \colon$

$$
\begin{array} { r l } & { U _ { \mathrm { t r } } ( \mathbf { t } ) \big ( U _ { \mathrm { r o t } } ( R ) \mathbf { v } \big ) = U _ { \mathrm { r o t } } ( R ) U _ { \mathrm { t r } } ( R ^ { - 1 } \mathbf { t } ) \mathbf { v } } \\ & { ~ = U _ { \mathrm { r o t } } ( R ) \exp ( i \omega ^ { \top } R ^ { - 1 } \mathbf { t } ) \mathbf { v } } \\ & { ~ = \exp ( i ( R \omega ) ^ { \top } \mathbf { t } ) \big ( U _ { \mathrm { r o t } } ( R ) \mathbf { v } \big ) . } \end{array}\tag{32}
$$

Here the second equality uses that $\textbf { v } \in \ E _ { \omega }$ is an eigenvector of $U _ { \mathrm { t r } } ( R ^ { - 1 } { \bf t } )$ ; the last uses that the (unit-modulus) eigenvalue $\exp ( i \omega ^ { \top } R ^ { - 1 } \mathbf { t } )$ is a complex scalar, hence commutes with $U _ { \mathrm { r o t } } ( R )$ and can be pulled out of the product, together with $\boldsymbol { \omega } ^ { \top } \boldsymbol { R } ^ { - 1 } \mathbf { t } = ( \boldsymbol { R } \boldsymbol { \omega } ) ^ { \top } \mathbf { t } .$ . Thus, $U _ { \mathrm { r o t } } ( R ) \mathbf { v }$ is an eigenvector of $U _ { \mathrm { t r } } ( \mathbf { t } )$ with spatial frequency $R \omega$ , meaning $U _ { \mathrm { r o t } } ( R ) E _ { \omega } = E _ { R \omega }$ . This necessitates that the frequency spectrum Ω must be invariant under all continuous 3D rotations:

$$
R \Omega = \Omega , \qquad \forall R \in \mathrm { S O } ( 3 ) .\tag{33}
$$

By Eq. (33) and the finiteness $| \Omega | \leq d < \infty$ from Step 4, Ω is a finite $\mathrm { S O ( 3 ) }$ -invariant subset of $\mathbb { R } ^ { 3 } ;$ the lemma therefore forces $\Omega = \{ \mathbf { 0 } \}$

Substituting $\omega _ { a } = \mathbf { 0 }$ back into the diagonalized form yields

$$
U _ { \mathrm { t r } } ( \mathbf { t } ) = V I _ { d } V ^ { * } = I _ { d } , \qquad \forall \mathbf { t } \in \mathbb { R } ^ { 3 } .\tag{34}
$$

Finally, decomposing any arbitrary pose as $( R , { \bf t } ) = ( I _ { 3 } , { \bf t } ) ( R , { \bf 0 } )$ and applying the homomorphism yields $U ( R , \mathbf { t } ) { \dot { = } } U _ { \mathrm { t r } } ( \mathbf { t } ) { \dot { U } } _ { \mathrm { r o t } } ( R ) { \dot { = } } U ( R , \mathbf { 0 } )$ . This completes the proof. □

## B IMPLEMENTATION AND EVALUATION DETAILS

## B.1 ARCHITECTURE AND TRAINING CONFIGURATIONS

Head partition. In our primary implementation based on the Wan2.2 TI2V-5B backbone $( d =$ 128), the channel budget is partitioned into four orthogonal sub-blocks: $d = d _ { \mathrm { d i s p } } + d _ { \mathrm { r o t } } + d _ { \mathrm { t r a n s } } +$ $d _ { \mathrm { n a t i v e } } = 3 6 + 3 6 + 2 4 + 3 2$ . Specifically, $d _ { \mathrm { d i s p } } = 3 6$ encodes $L = 6$ non-uniform angular disparity fractions $\rho _ { \ell } \in \{ 0 , 0 . 0 5 , 0 . 1 5 , \bar { 0 . 4 } , 0 . 7 , 1 \}$ with $n = 2$ triplets each $( 6 \times 2 \times 3 ) ; d _ { \mathrm { r o t } } \stackrel { \cdot } { = } 3 6$ allocates $m = 1 2$ coordinate triplets for ray-local MinRot; $d _ { \mathrm { t r a n s } } = 2 4$ covers 3 spatial axes over $K = 4$ logspaced metric wavelengths $( \lambda \in [ 0 . 5 , 2 0 0 ] \mathrm { { m } ) ; }$ ; and $d _ { \mathrm { n a t i v e } } = 3 2$ applies standard RoPE to temporal (16D) and image-plane x/y (8D each) coordinate bands.

Optimization details. All driving models are trained with AdamW $( \beta _ { 1 } = 0 . 9 , \beta _ { 2 } = 0 . 9 9 9 , \epsilon =$ $1 \bar { 0 ^ { - 8 } }$ , weight decay 0.01) using a constant learning rate of $1 \times 1 0 ^ { - 5 }$ and no warmup. Training uses bfloat16 mixed precision on 64 × NVIDIA H20 GPUs for 20k iterations with a global batch size of 64. PanShot models are trained on 8 × NVIDIA H20 GPUs for 10k iterations.

## B.2 EVALUATION METRICS AND MATHEMATICAL FORMULATIONS

Estimated camera poses are recovered from generated frames by VGGT-Ω (Wang et al., 2026a). Let T denote the number of frames in an evaluated sequence, and let $T _ { t } = ( R _ { t } , \mathbf { o } _ { t } ) \in S E ( 3 )$ and $\widehat { T } _ { t } \ = \ ( \widehat { R } _ { t } , \widehat { \mathbf { o } } _ { t } ) \in S E ( 3 )$ denote the commanded and estimated camera-to-world poses at frame $t \in \{ 1 , \ldots , T \}$ . To remove the absolute $S E ( 3 )$ gauge, all poses are expressed relative to the first frame,

$$
T _ { t } ^ { \mathrm { r e l } } = T _ { 1 } ^ { - 1 } T _ { t } = \left( R _ { 1 } ^ { \top } R _ { t } , R _ { 1 } ^ { \top } ( \mathbf { o } _ { t } - \mathbf { o } _ { 1 } ) \right) = : \left( R _ { t } ^ { \mathrm { r e l } } , \mathbf { o } _ { t } ^ { \mathrm { r e l } } \right) ,\tag{35}
$$

with $\widehat { T } _ { t } ^ { \mathrm { r e l } } = ( \widehat { R } _ { t } ^ { \mathrm { r e l } } , \widehat { \mathbf { o } } _ { t } ^ { \mathrm { r e l } } )$ constructed analogously. The geodesic rotation error in degrees is

$$
\angle ( R , { \widehat { R } } ) = \operatorname { a r c c o s } \left( \operatorname { c l a m p } \left( { \frac { \operatorname { t r } ( R ^ { \mathsf { T } } { \widehat { R } } ) - 1 } { 2 } } , - 1 , 1 \right) \right) \times { \frac { 1 8 0 ^ { \circ } } { \pi } } .\tag{36}
$$

Reported driving scores average equally over the 384 test clips; PanShot scores average over the 272 jitter-filtered clips. The pairwise AUC below is a micro-average over all within-clip frame pairs.

Monocular pose recovery leaves a global scale ambiguity. On driving evaluations we therefore apply a single least-squares scale to the estimated camera centers, leaving rotations unchanged:

$$
s ^ { \star } = \frac { \sum _ { t = 1 } ^ { T } \widehat { \mathbf { o } } _ { t } ^ { \mathrm { r e l } \top } \mathbf { o } _ { t } ^ { \mathrm { r e l } } } { \sum _ { t = 1 } ^ { T } \Vert \widehat { \mathbf { o } } _ { t } ^ { \mathrm { r e l } } \Vert _ { 2 } ^ { 2 } } , \qquad \widetilde { \mathbf { o } } _ { t } = s ^ { \star } \widehat { \mathbf { o } } _ { t } ^ { \mathrm { r e l } } .\tag{37}
$$

Following the official PanShot evaluation code (Zhang et al., 2026b), CamMC and PanShot translation metrics instead normalize the ground-truth and recovered trajectories independently by their respective maximum translation norms,

$$
\bar { \mathbf { o } } _ { t } = \frac { \mathbf { o } _ { t } ^ { \mathrm { r e l } } } { \operatorname* { m a x } _ { 1 \leq \tau \leq T } \| \mathbf { o } _ { \tau } ^ { \mathrm { r e l } } \| _ { 2 } + \epsilon _ { 0 } } , \qquad \bar { \hat { \mathbf { o } } } _ { t } = \frac { \widehat { \mathbf { o } } _ { t } ^ { \mathrm { r e l } } } { \operatorname* { m a x } _ { 1 \leq \tau \leq T } \| \widehat { \mathbf { o } } _ { \tau } ^ { \mathrm { r e l } } \| _ { 2 } + \epsilon _ { 0 } } ,\tag{38}
$$

with $\epsilon _ { 0 } = 1 0 ^ { - 9 }$ . The two scale treatments are not interchangeable: $\mathrm { t r } \%$ uses $s ^ { \star }$ , whereas CamMC and TransErr use Eq. (38); both rotation metrics are invariant to translation scaling.

Rotation Error (rot<sup>◦</sup>). Used on driving evaluations. The first-frame relative pose is the identity, so the mean excludes $t = 1$

$$
\mathrm { r o t } ^ { \circ } = \frac { 1 } { T - 1 } \sum _ { t = 2 } ^ { T } \mathcal { L } \big ( R _ { t } ^ { \mathrm { r e l } } , \widehat { R } _ { t } ^ { \mathrm { r e l } } \big ) .\tag{39}
$$

Relative Translation Error $( \mathbf { t r \% } )$ . Used on driving evaluations. After the scale alignment in Eq. (37), the mean center error is normalized by the ground-truth endpoint displacement (chord length):

$$
\begin{array} { r } { \mathrm { t r } \% = \frac { \frac { 1 } { T - 1 } \sum _ { t = 2 } ^ { T } \| { \bf o } _ { t } ^ { \mathrm { r e l } } - \widetilde { \bf o } _ { t } \| _ { 2 } } { \| { \bf o } _ { T } ^ { \mathrm { r e l } } \| _ { 2 } } \times 1 0 0 \% . } \end{array}\tag{40}
$$

Pairwise Joint Pose AUC (AUC@τ). Used on driving evaluations. Let $\mathcal { P } = \{ ( i , j ) : 1 \leq i < j \leq$ $T \}$ denote within-clip frame pairs, pooled over the evaluation set. For each pair, form the relative motions $T _ { i  j } = ( T _ { i } ^ { \mathrm { r e l } } ) ^ { - 1 } T _ { j } ^ { \mathrm { r e l } }$ and $\widehat { T } _ { i  j } = ( \widehat { T } _ { i } ^ { \mathrm { r e l } } ) ^ { - 1 } \widehat { T } _ { j } ^ { \mathrm { r e l } }$ , and define the unoriented translationdirection error

$$
e _ { i j } ^ { \mathrm { t } } = \operatorname { a r c c o s } ( \mathrm { c l a m p } ( | \frac { \mathbf { o } _ { i  j } ^ { \top } \widehat { \mathbf { o } } _ { i  j } } { \| \mathbf { o } _ { i  j } \| _ { 2 } \| \widehat { \mathbf { o } } _ { i  j } \| _ { 2 } } | , 0 , 1 ) ) \times \frac { 1 8 0 ^ { \circ } } { \pi } ,\tag{41}
$$

with $e _ { i j } ^ { \mathrm { t } }$ set to a large sentinel when either translation vanishes. The joint pair error is $\begin{array} { r l } { e _ { i j } } & { { } = } \end{array}$ max $\cdot ( \angle ( R _ { i  j } , \widehat { R } _ { i  j } ) , e _ { i j } ^ { \mathrm { t } } )$ . Because $e _ { i j } ^ { \mathrm { t } }$ depends only on direction, this metric does not apply $s ^ { \star }$ With integer thresholds $\bar { \tau } \in \{ 3 , 1 0 \}$ (degrees), the discrete threshold average is

$$
\operatorname { A U C @ } \tau = \frac { 1 } { \tau } \sum _ { k = 1 } ^ { \tau } \frac { 1 } { | \mathscr { P } | } \sum _ { ( i , j ) \in \mathscr { P } } \mathbb { I } ( e _ { i j } < k ^ { \circ } ) \times 1 0 0 \% .\tag{42}
$$

Camera Motion Consistency (CamMC). Let $P _ { t } = [ R _ { t } ^ { \mathrm { r e l } } \ | \ \bar { \mathbf { o } } _ { t } ]$ and $\widehat { P } _ { t } = [ \widehat { R } _ { t } ^ { \mathrm { r e l } } \ | \ \bar { \widehat { \mathbf { o } } } _ { t } ]$ be the $3 \times 4$ relative pose matrices after Eq. (38). Following UCPE, CamMC is the sum of per-frame Frobenius errors, then averaged over videos:

$$
\mathrm { C a m M C } = \sum _ { t = 1 } ^ { T } \| P _ { t } - \widehat { P } _ { t } \| _ { F } .\tag{43}
$$

PanShot Rotation Error (RotErr). PanShot sums per-frame geodesic errors in radians, following its official evaluation code (Zhang et al., 2026b):

$$
\mathrm { R o t E r r } = \sum _ { t = 1 } ^ { T } \operatorname { a r c c o s } \left( \mathrm { c l a m p } \left( \frac { \mathrm { t r } \left( ( R _ { t } ^ { \mathrm { r e l } } ) ^ { \top } \widehat { R } _ { t } ^ { \mathrm { r e l } } \right) - 1 } { 2 } , - 1 , 1 \right) \right) \mathrm { . }\tag{44}
$$

Translation Error (TransErr). Used on PanShot evaluations, after the same per-trajectory maxnorm as CamMC (not the driving scale $s ^ { \star } )$

$$
\mathrm { T r a n s E r r } = \sum _ { t = 1 } ^ { T } \| \bar { \mathbf { o } } _ { t } - \bar { \hat { \mathbf { o } } } _ { t } \| _ { 2 } .\tag{45}
$$

## C ADDITIONAL EXPERIMENTAL ANALYSES

## C.1 ARCHITECTURAL DESIGN AND LAYER-WISE MODULATION

Table 5 varies four design axes around the default MeRoPE configuration. The Comp. rows vary only the adapter compression ratio C. The Design rows fix $C = 4$ and vary one design factor at a time: translation frequency, injection schedule, or integration style; check marks denote active mechanisms and blank cells denote inactive ones.

Adapter compression. The Wan2.2 backbone has width 3072 with 24 attention heads, and the ratio C sets the parallel camera-attention branch to width $3 0 7 2 / C$ with $2 4 / C$ heads while preserving the 128-dimensional head size. Thus, $C = 2 ,$ 4, and 8 use widths 1536, 768, and 384, with 12, 6, and 3 heads, respectively. Although $C = 2$ gives the strongest pose scores, the default $C = 4$ retains similar CamMC with half the adapter capacity.

Translation frequencies. The Multi-f column indicates whether metric translation uses multiple wavelength bands; the Single-f variant keeps the same 128-dimensional channel allocation and changes only the translation frequency. It leaves directional AUC nearly unchanged but degrades tr% and CamMC, showing that one wavelength does not provide reliable translation control across the evaluated trajectories.

Table 5: Architecture and injection ablations on nuScenes. Best in bold; second-best is underlined within each group.
<table><tr><td></td><td></td><td colspan="5">Configuration</td><td colspan="5">Overall Metrics</td></tr><tr><td></td><td>Variant</td><td>C</td><td>Injection</td><td>CamSA</td><td>ReRoPE</td><td>Multi-f</td><td>rot°↓</td><td>tr%↓</td><td>AUC@3↑</td><td>AUC@10↑</td><td>CamMC↓</td></tr><tr><td rowspan="3">Comp.</td><td>C=2</td><td>2</td><td>15 blks</td><td>√</td><td></td><td>√</td><td>1.37</td><td>2.68</td><td>45.28</td><td>80.62</td><td>2.26</td></tr><tr><td>C=4 (ours)</td><td>4</td><td>15 blks</td><td>√</td><td></td><td>√</td><td>1.37</td><td>2.84</td><td>42.11</td><td>79.47</td><td>2.32</td></tr><tr><td>C=8</td><td>8</td><td>15 blks</td><td>√</td><td></td><td>√</td><td>1.47</td><td>2.83</td><td>41.85</td><td>79.11</td><td>2.42</td></tr><tr><td rowspan="5">Design</td><td>Single f</td><td>4</td><td>15 blks</td><td>√</td><td></td><td></td><td>1.49</td><td>3.58</td><td>43.34</td><td>79.61</td><td>3.01</td></tr><tr><td>Every block</td><td>4</td><td>30 blks</td><td>√</td><td></td><td>√</td><td>1.38</td><td>2.65</td><td>43.60</td><td>79.90</td><td>2.27</td></tr><tr><td>Peak 11, 13</td><td>4</td><td>2 blks</td><td>√</td><td></td><td>√</td><td>1.37</td><td>2.97</td><td>45.00</td><td>80.27</td><td>2.43</td></tr><tr><td>ReRoPE (Li et al., 2026a)</td><td>4</td><td>15 blks</td><td></td><td>√</td><td>√</td><td>1.67</td><td>2.67</td><td>42.98</td><td>79.71</td><td>2.53</td></tr><tr><td>Default (ours)</td><td>4</td><td>15 blks</td><td>√</td><td></td><td>√</td><td>1.37</td><td>2.84</td><td>42.11</td><td>79.47</td><td>2.32</td></tr></table>

Injection schedule. The Injection column compares three compute levels: Peak 11,13 uses two blocks, the default uses 15 alternating blocks, and Every block uses all 30. Figure 7 shows that camera modulation concentrates in the middle layers, with Blocks 11 and 13 contributing over half of the residual amplitude, motivating the Peak 11,13 variant. This variant preserves rotation accuracy and strong directional AUC while increasing CamMC only modestly from 2.32 to 2.43, and reduces camera-branch executions by 87% relative to the default. The default uses half as many camera-branch executions as Every block and has a similar CamMC (2.32 versus 2.27).

![](images/b6afc8b9c279a4d520848ec12737cfae5e9812109d3f738f808081b8f907de3c.jpg)  
Figure 7: Per-block camera modulation amplitude. Relative camera-attention residual norm across 30 DiT blocks; red marks the two largest blocks and blue marks alternating-block injection.

Integration style. A check in CamSA denotes the dedicated parallel camera self-attention branch; a check in ReRoPE (Li et al., 2026a) denotes carving low-frequency camera rotary channels into the backbone’s native self-attention without that branch. ReRoPE improves tr% but weakens rotation and CamMC, so the parallel CamSA design provides the better overall balance.

## C.2 COST OF RELAXING PER-TOKEN FACTORIZATION

Strictly per-token encodings transform each token once, whereas MeRoPE expresses metric translation and disparity anchors in each query-camera frame and therefore constructs query-dependent key/value representations. Table 6 measures the resulting cost at both the operator and full-model levels.

Evaluation protocol. The Single CamSA Block columns time one complete camera self-attention (CamSA) block, including positional modulation, attention, and output projection, while fixing the block dimensions and input tokens across methods. This isolated benchmark uses one H20 because it excludes the backbone and distributed model sharding. The Training and Latent Denoising columns instead measure full training steps and the complete denoising loop on four H20 GPUs under the production four-way HSDP setup.

Matched comparison. For the full-model benchmark, GTA, UCPE, and MeRoPE inject CamSA into the same 15 alternating transformer blocks (Blocks 1, 3, . . . , 29), yielding 5.14B parameters for each camera-conditioned model; the backbone-only baseline contains 5.00B parameters. This controls both the injection schedule and model size across the three camera encodings.

Results. Query-camera grouping is more expensive within a single CamSA block, but the full backbone amortizes much of this local overhead. Relative to UCPE, MeRoPE requires about 2.5× the isolated-block latency and 3× the peak memory, whereas full-model throughput decreases by

Table 6: Computational cost of query-camera grouping. Single-block results use one H20; fullmodel results use four H20 GPUs with a matched 15-block injection schedule.
<table><tr><td></td><td></td><td colspan="2">Single CamSA Block</td><td colspan="2">Training</td><td colspan="2">Latent Denoising</td></tr><tr><td>Method</td><td>Factorization</td><td>Latency (ms)↓</td><td>Peak Mem. (GiB)↓</td><td>Throughput (clips/s)↑</td><td>Peak Mem. (GiB/GPU)↓</td><td>Throughput (clips/s)↑</td><td>Peak Mem. (GiB/GPU)↓</td></tr><tr><td>Wan2.2 base</td><td>No CamSA</td><td></td><td>N/A</td><td>5.30</td><td>35.80</td><td>0.197</td><td>29.65</td></tr><tr><td>GTA (Miyato et al., 2024)</td><td>Per-token</td><td>4.97</td><td>0.38</td><td>4.89</td><td>36.58</td><td>0.188</td><td>31.03</td></tr><tr><td>UCPE (Zhang et al., 2026a)</td><td>Per-token</td><td>5.03</td><td>0.40</td><td>4.86</td><td>36.58</td><td>0.188</td><td>31.03</td></tr><tr><td>MeRoPE</td><td>Query-grouped</td><td>12.47</td><td>1.21</td><td>4.70</td><td>37.57</td><td>0.177</td><td>31.13</td></tr></table>

Table 7: Real-world grounding with retrieved history on nuScenes. Best in bold; second-best is underlined.
<table><tr><td>Method</td><td>rot°↓</td><td>tr%↓</td><td>AUC@3↑</td><td>AUC@10↑</td><td>CamMC↓</td><td>FID↓</td><td>FVD↓</td></tr><tr><td>MeRoPE (no history)</td><td>0.82</td><td>2.21</td><td>48.76</td><td>83.24</td><td>1.60</td><td>19.82</td><td>134.15</td></tr><tr><td>MeRoPE + History CA</td><td>0.78</td><td>2.13</td><td>52.01</td><td>84.17</td><td>1.52</td><td>19.59</td><td>130.27</td></tr><tr><td>MeRoPE + History SA</td><td>0.74</td><td>2.11</td><td>51.65</td><td>84.22</td><td>1.48</td><td>18.21</td><td>104.41</td></tr></table>

3.3% for training and 5.9% for denoising, with peak-memory increases of 2.7% and 0.3%, respectively. GTA and UCPE have nearly identical system costs under the matched schedule.

Source of overhead and future work. The extra local cost comes from constructing and storing separate transformed K/V tensors for each query-camera group rather than from additional attention dot products. Processing query-camera groups in smaller batches can reduce peak memory, while a fused grouped-attention kernel could avoid expanded K/V tensors and reduce both memory and latency.

## C.3 REAL-WORLD GROUNDING WITH RETRIEVED HISTORY

To test grounding beyond the current field of view, we encode calibrated same-location images with the video VAE and inject their patch tokens through two pathways. History CA follows CityRAG (Chou et al., 2026): video tokens attend to retrieved-reference K/V through standard crossattention. History SA follows Seoul World Model (Seo et al., 2026): retrieved reference tokens are concatenated with the current video tokens and jointly processed by self-attention across successive DiT blocks; only the generated video tokens are decoded.

Training signals. Current-image dropout randomly removes the entire clean conditioning prefix when usable history is available, so the model must predict from retrieved observations rather than copy the current clip; evaluation retains the full prefix. Standard training retrieves history close to the ground-truth camera path, but left/right test commands often have only history with a larger relative pose, creating a train–test mismatch. Lateral-conflict training simulates this case by substituting a spatially overlapping donor from a neighboring lane while leaving the target video and camera command unchanged, teaching the model to use off-trajectory history without weakening camera control.

Temporal tagging. Both pathways add 100 to the matched history-frame time coordinates, placing retrieved tokens in a dedicated temporal region so the model can distinguish historical context from current video tokens.

Both history variants provide modest pose-control gains, while History SA achieves the best CamMC, FID, and FVD in Table 7. The VAE latents are optimized for pixel reconstruction rather than semantic alignment, so they provide no explicit high-level interface for retrieval fusion. History SA repeatedly mixes retrieved and future tokens across successive DiT blocks, whereas History CA exposes the retrieved latents only as cross-attention K/V; this difference may explain History SA’s stronger visual anchoring. Figure 8 illustrates this pattern: History CA follows the no-history layout at late timesteps, while History SA preserves retrieval-supported geometry in both selected scenes.

(a) Tree-lined corridor  
![](images/d21f625f93cbc15dfe3a6083d0e5d0b1dfedb4d9e9f9f5f00d9f267044034b70.jpg)

(b) Cross-weather street corner  
![](images/23311422ad021ce59619a10cf25eee1bde5c535aa316920ae845f38ad1f9ee5b.jpg)  
Figure 8: Qualitative comparison of history injection. Across the same scenes and timesteps, History SA preserves retrieval-supported late-frame geometry, whereas History CA follows the nohistory layout.

## D ADDITIONAL QUALITATIVE RESULTS

## D.1 MATCHED-BACKBONE COMPARISON WITH PRIOR CAMERA ENCODINGS

Figure 9 presents three fixed clips separately for readability; Table 2 reports the aggregate evaluation.   
In these clips, MeRoPE preserves more coherent late-frame geometry.

![](images/ae6afd0e8a0041cb8eabd8e4a9fc4457c9d412519d398777c9c53816e8fa1c84.jpg)  
Figure 9: Matched-backbone qualitative comparison (1/3). Each row shows commanded and VGGT-Ω-recovered paths with frames at t ∈ {0, 16, 32, 48}.

![](images/43cb9143bb207469014bd5834e18a50052430be4fbed81bde995bed139970aac.jpg)  
Figure 9: Matched-backbone qualitative comparison (2/3). The same protocol is shown for the second scene.

![](images/5fcef93b549821ead091d58d1fc86d0ee9e7f5f98f7b8d1abc1af48a745a21a3.jpg)  
Figure 9: Matched-backbone qualitative comparison (3/3). The same protocol is shown for the third scene.

## D.2 ADDITIONAL CAMERA-POSE-CONTROLLED GENERATIONS

Figure 10 extends the main qualitative comparison to three additional nuScenes scenes.

![](images/dd4fa1ac01d471f42c451cf3aabb697a56f151bc489dde03672c0caa7d9eef1f.jpg)  
Figure 10: Additional nuScenes camera-control results. For each scene, the left plot compares commanded (yellow solid) and VGGT-Ω-recovered (cyan dashed) paths; the remaining columns show frames at t ∈ {0, 16, 32, 48}.