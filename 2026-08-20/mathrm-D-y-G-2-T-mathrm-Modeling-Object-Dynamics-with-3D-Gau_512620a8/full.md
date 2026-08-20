# $\mathrm { D y G ^ { 2 } T } \mathrm { : }$ Modeling Object Dynamics with 3D Gaussian Temporal-Spatial Particle Graph Transformer

Yansong Wang, Zhaobo Qi, Xinyan Liu, Beichen Zhang, Shuhui Wang, Weigang Zhang, and Qingming Huang, Fellow, IEEE

Abstract—Modeling object dynamics from limited visual observations is a fundamental problem for enabling accurate motion trajectory prediction in embodied interaction scenarios. Existing dynamics modeling methods first compress reconstructed particle representations into sparse Key Points and model their evolution using locally constrained interactions, thereby discarding finegrained local details and obscuring discriminative interaction modeling across spatial and temporal scales, leading to drifting trajectories and inaccurate appearance prediction. To tackle these issues, we propose DyG<sup>2</sup>T, a dynamics modeling framework that infers object motion trajectories by spatially completing and temporally discriminating Key Point representations and modeling multi-scale interaction over particle graphs. Spatially, $\mathbf { D y G ^ { 2 } T }$ enriches each Key Point by aggregating neighboring raw particle positions to recover fine-grained local details, while explicitly encoding relative offsets among Key Points to enhance geometric structure perception. Temporally, we introduce a Temporal Disentangling Network (TDN) to identify dominant cross-frame variations in latent space and amplify inter-frame differences, yielding temporally discriminative representations that are subsequently aggregated via Temporal Attention to capture frame-wise temporal evolution cues. For comprehensive interaction modeling, a Particle Graph Transformer leverages global attention to preserve discriminative long-range dependencies among Key Points, mitigating representation homogenization induced by locality-constrained modeling and providing a robust basis for accurate trajectory prediction. Experiments on both synthetic and real-world datasets demonstrate that DyG<sup>2</sup>T achieves accurate dynamics modeling and reasoning, and exhibits strong cross-object and real-world generalization.

Index Terms—Object Dynamics Modeling, Spatial Completion, Temporal Aggregation, Particle Graph Transformer.

## I. INTRODUCTION

R <sup>ECENTLY,</sup> <sup>artificial</sup> <sup>intelligence</sup> <sup>has</sup> <sup>evolved</sup> <sup>from</sup> <sup>dis-</sup>embodied intelligence operating independently of the embodied intelligence operating independently of the physical world toward embodied systems that require interaction with more realistic, complex real-world environments [1].

Under this paradigm shift, dynamics modeling and reasoning for dynamic objects have become increasingly important, as applications such as embodied intelligence, digital twins, and interactive simulation require not only understanding the current state of objects but also inferring their future motion from limited visual observations. Interaction with elastic objects represents one particularly challenging scenario, since such objects exhibit complex material-dependent dynamic behaviors, while their internal motion patterns are difficult to fully observe from sparse viewpoints. Consequently, object dynamics modeling has emerged as a fundamental technical demand for capturing motion patterns in continuous dynamic spaces by modeling complex interactions among internal components, such as cross-particle influence propagation [2], ultimately enabling precise future trajectory prediction to support seamless agent–environment interaction.

To acquire object dynamics, most existing approaches [3, 4] first reconstruct 3D particle representations of motion objects from 2D visual observations [5, 6], then downsample them via Farthest Point Sampling (FPS) [7] to obtain sparse Key Points for dynamic modeling. Current methods largely fall into two categories. Strong physics-prior approaches [8, 9] assume a priori of material properties [3] and select corresponding differential equations to evolve Key Points trajectories [10]. However, such reliance on constitutive models restricts their applicability to objects with unknown or heterogeneous materials [11]. In contrast, differentiable network-based methods [12, 13] learn dynamics in a data-driven manner, avoiding material priors and handcrafted equations. They typically use multilayer perceptrons (MLPs) to infer cross-frame deformations from Key Point spatial attributes, enabling evolution from a canonical representation to arbitrary time steps [14, 15], or employ graph neural networks (GNNs) [2, 4] to learn Key Point motion patterns within local neighborhoods for future trajectory inference. Yet, relying solely on Key Point-level attributes discards the rich local details of the raw particles, while locality-constrained message passing often leads to homogenized features. These limitations ultimately result in inaccurate appearance predictions and drifting trajectories.

Specifically, in one respect, researchers adopt the coordinates of downsampled Key Points as the initial particle representation for dynamics modeling. As shown in Figure 1, FPS drops a significant amount of raw particles (i.e., Raw Point Cloud), consequently losing fine-grained local details carried by those discarded points; moreover, relying on coordinates as initial features further weakens the model’s perception of object geometry and structure. This dual loss of local details and geometric cues compromises the integrity of the feature basis. Such incomplete particle representations from different frames become temporally entangled, causing semantic collapse that severely reduces temporal discriminability and hinders downstream dynamics modeling, leading to errors in appearance and trajectory decoding. Therefore, retaining the particle representations that are both complete and temporally discriminative is crucial to robust dynamics modeling.

![](images/9b4470d9bc6de91a7358ca557aa3aebc051939c362b3a0923403834470e887d7.jpg)  
Fig. 1. DyG<sup>2</sup>T vs. differentiable network-based methods for dynamics modeling of a falling Banana. Colored curves denote motion trajectories. Differentiable network-based methods rely on Key Point-level spatial attributes and propagate features node-wise, leading to appearance artifacts and trajectory drift. In contrast, $\mathrm { { D y G ^ { 2 } T } }$ constructs discriminative embeddings by modeling multi-scale interaction based on completed particle features, enabling accurate trajectory and appearance decoding.

In another respect, locality-constrained message passing limits particle representation learning by inducing feature homogenization. As shown in Figure 1 top, non-local information is obtained only through iterative neighbor aggregation, leading to diffusive updates that progressively blur representational distinctions [16]. This homogenization obscures interaction patterns across different spatial extents, producing homogenization particle embeddings and resulting in positional decoding errors and trajectory deviations. Recent studies [17] suggest that capturing dependencies across multiple spatial ranges explicitly provides an effective means to preserve discriminative particle representations. By aggregating interaction cues spanning different spatial ranges, multi-scale modeling preserves informative variations in particle embeddings. This, in turn, provides a more reliable representational basis for accurate dynamics modeling and trajectory prediction.

To address these issues, we propose $\mathrm { \Delta D y G ^ { 2 } T , }$ a dynamics modeling framework that enables accurate motion inference by spatially completing and temporally disentangling enriched Key Point representations, while explicitly preserving particle feature diversity through the capture of discriminative cross-scale dependencies. As shown in Figure 1, $\mathrm { \Delta D y G ^ { 2 } T }$ first leverages dynamic reconstruction to extract trackable particle representations from sparse-view videos. To ensure complete spatiotemporal representations, we enhance particle features from two complementary perspectives. Spatially, each Key Point aggregates neighboring raw particle positions that encode local appearance, enriching fine-grained local details. Relative offsets capturing directed layout relations between

Key Points are further used to enhance geometric structure perception explicitly. Temporally, we feed adjacent-frame Key Point representations into a Temporal Disentangling Network (TDN) to estimate offsets along dominant disentangling directions, amplifying inter-frame differences and yielding temporally discriminative features. These disentangled embeddings are then aggregated via Temporal Attention to enrich Key Points’ frame-wise temporal evolution cues. To further model multi-scale interaction from local contacts to global structure, a Particle Graph Transformer leverages global attention to capture multi-scale dependencies among interaction-relevant Key Points directly, preserving discriminative features essential for accurate dynamics modeling.

We evaluate $\mathrm { \Delta D y G ^ { 2 } T }$ on the Spring-Gaus synthetic & realworld [2] and our Unity3D-Heterogeneous datasets, demonstrating its strong cross-object generalization and real-world scalability in dynamics modeling and reasoning. The contributions of this work are as follows:

• We propose $\mathrm { \Delta D y G ^ { 2 } T }$ , a dynamics modeling framework that mitigates trajectory prediction bias by capturing differentiated multi-scale dependencies within particle representations.

$\mathrm { \Delta D y G ^ { 2 } T }$ enriches particle representations with fine-grained local details and geometry construction through spatial completion, and captures discriminative temporal evolution cues via TDN and temporal aggregation.

• Extensive experiments on both synthetic and real-world datasets demonstrate that $\mathrm { \Delta D y G ^ { 2 } T }$ accurately predicts object motion trajectories across diverse geometries and scales effectively in real-world scenarios.

## II. RELATED WORK

## A. Dynamic 3D Scene Reconstruction

Reconstructing 3D dynamic scenes from multi-view images has become a central approach for acquiring 3D representations of dynamic scenes [18]. Existing methods can be broadly grouped into dynamic extensions of neural radiance fields (NeRFs) [19, 20] or 3D Gaussians [13, 21]. Early efforts extend NeRF representations [22, 23], originally designed for static scenes, to dynamic settings by either fitting frame-wise residual motion [24, 25] or learning deformation fields that map a canonical template to each observed frame [26, 27]. While these approaches can reconstruct 3D geometry, their ray-sampling nature leads to slow training [28], and the purely data-driven implicit representations exhibit limited generalization and poor robustness to noise, ultimately constraining downstream dynamics modeling and prediction.

In contrast, 3D Gaussian-based methods build upon 3D Gaussian Splatting (3DGS) [5], representing dynamic objects as explicit Gaussian spheres or ellipsoids. This explicit parameterization enhances controllability and provides a solid foundation for dynamic extensions. Dyn3DGS [29], the first dynamic extension of 3DGS, optimizes non-visual Gaussian parameters per frame to enable trackable dynamic reconstruction. Similar to NeRF-based deformation fields, subsequent work [15, 30, 31] introduces MLP-based deformation networks to query temporally varying Gaussian attributes, enabling continuous warping from a canonical representation to arbitrary observations. Follow-up methods expand this direction: Scale-GS [32] and SCas4D [33] integrate hierarchical clustering into the deformation network to achieve layered fitting of dynamic Gaussians, while 3DGStream [14] decodes Gaussian rotations and translations from hashed voxel features. Distinct from these reconstruction-focused approaches that do not account for physical behavior, our work incorporates dynamics modeling to support future motion inference.

## B. Physics-based Dynamics Modeling and Reasoning

Inferring object dynamics from only sparse initial observations is highly challenging due to the complexity of continuous state spaces [3]. Existing dynamics modeling and reasoning methods can be broadly divided into 2 groups, including strong physics-prior approaches, such as physics-engine-based and PDE-regularized methods, and differentiable network-based methods. Physics-engine-based approaches [9] treat material properties as priors and evolve trajectories using constitutive equations corresponding to each material. Recent studies [3, 11] estimate physical parameters in a reverse manner to reduce reliance on predefined material settings, yet they remain constrained by the limited generalizability of specific constitutive formulations. Inspired by Physics-Informed Neural Networks (PINNs) [34], PDE-regularized methods [35, 36, 37] incorporate various PDEs as soft constraints to regularize the learning of physical properties of 3D objects, but they often suffer from low training efficiency and difficulty in modeling complex boundary conditions [38].

In contrast, differentiable-network-based approaches shift the burden of modeling complex interactions to nonlinear architectures such as MLPs [38, 39], GNNs [4, 40], or springmass [2], achieving strong results on simulated or real entities, including animals [38], cloth [41], fluids [40], and elastic materials [2]. FreeGave [38] focuses on joint novel view synthesis and future prediction from free-view video, but its divergencefree deformation field, based on MLPs, neglects discontinuous state changes induced by collisions and other dynamic interactions. Methods such as GaussianPrediction [12] instead treat downsampled Key Points as nodes and model dynamics via local message passing, which inevitably discards fine-grained geometric details and overlooks interaction modeling across different spatial scales.

In comparison, our method lies between these directions: it combines physics-regularized dynamic reconstruction with differentiable dynamics prediction, and constructs a Spatial-Temporal Graph Transformer that directly captures objectinternal multi-scale interaction patterns induced by object motion, striking a balance between physical plausibility and generalization to unknown or heterogeneous materials.

## III. METHOD

## A. Preliminary: 3D Gaussian Spaltting

3D Gaussian Splatting (3DGS) [5] optimizes a set of learnable Gaussian kernels as an explicit dynamic object representation through a differentiable rasterizer. Each kernel is parameterized by spatial and appearance descriptors. The spatial descriptors include the 3D position $\mu = ( x , y , z )$ and the 3D rotation $r = ( q w , q x , q y , q z )$ represented as a quaternion, where qw is the quaternion real part, and $( q x , q y , q z )$ is the quaternion imaginary components. The appearance descriptor includes the 3D scaling factor $s ~ \in ~ \mathbb { R } ^ { 3 }$ , m-DoF spherical harmonics coefficients $\bar { \mathbf { h } } \in \mathbb { R } ^ { 3 ( m + 1 ) ^ { 2 } }$ , and opacity $\sigma \in [ 0 , 1 ]$ The color C of each 2D pixel is computed using a depth-sorted Max Volume Rendering [29]:

$$
\mathbf { C } = \sum _ { i \in N } c _ { i } \phi _ { i } ^ { 2 D } \prod _ { j = 1 } ^ { i - 1 } ( 1 - \phi _ { j } ^ { 2 D } ) ,\tag{1}
$$

where N denotes the set of the Gaussian kernels, and $c _ { i }$ is the RGB color of kernel i obtained from spherical harmonics based on the view direction and coefficients h. $\phi _ { i }$ is the weighted opacity of kernel i,

$$
\phi _ { i } = \sigma _ { i } \exp \left( - \frac { 1 } { 2 } ( { \bf x } - { \boldsymbol { \mu } } _ { i } ) ^ { \mathsf { T } } { \boldsymbol { \Sigma } } _ { i } ^ { - 1 } ( { \bf x } - { \boldsymbol { \mu } } _ { i } ) \right) ,\tag{2}
$$

where $\Sigma _ { i } = R _ { i } s _ { i } s _ { i } ^ { \mathsf { T } } R _ { i } ^ { \mathsf { T } }$ denotes the covariance matrix of Gaussian i composed of a scaling matrix $s _ { i } = \mathrm { d i a g } ( s x _ { i } , s y _ { i } , s z _ { i } )$ and a rotation matrix $R _ { i } = { \mathrm { q 2 R } } ( q w _ { i } , q x _ { i } , q y _ { i } , q z _ { i } )$ . q2R(·) converts a quaternion into a rotation matrix. $\phi _ { i } ^ { 2 D }$ is the 2D version of Equation 2. The pixel position is obtained by approximating a perspective projection of the 3D Gaussian’s position $\mu _ { i } ^ { 2 D }$ and covariance matrix $\Sigma _ { i } ^ { 2 D }$

$$
\begin{array} { r l } & { \mu _ { i } ^ { 2 D } = ( P ( ( E \mu _ { i } ) / ( E \mu _ { i } ) _ { z } ) ) _ { 1 : 2 } } \\ & { \Sigma _ { i } ^ { 2 D } = ( J E \Sigma _ { i } E ^ { \mathsf { T } } J ^ { \mathsf { T } } ) _ { 1 : 2 , 1 : 2 } } \end{array} ,\tag{3}
$$

where $E$ and P represent the extrinsic and intrinsic parameters of the view camera, respectively, and J is the Jacobian matrix of the projection transformation.

## B. Overivew

For each dynamic object, given a set of 2D observations $\{ I _ { o , 1 } , \dots , I _ { o , t } \} _ { o = 1 } ^ { O }$ captured from O views over t timesteps, our objective is to predict the object’s motion over the subsequent ϵ timesteps via dynamics modeling. For clarity, we capture one frame per timestep and refer to the $O \times t$ frames from 1 to t as observation frames, where t = 1 denotes the initial frame. The subsequent ϵ frames are referred to as prediction frames.

![](images/d930bfcf584096f18b87aed6aa3214b88cf417b7e9b42bec3e1a279e4f8599a3.jpg)  
Fig. 2. Overview of the $\mathrm { D y G ^ { 2 } T . }$ (a) $\mathrm { \Delta D y G ^ { 2 } T }$ utilizes dynamic reconstruction to extract trackable particle representations. (b) The Spatial-Temporal Feature Completion and Aggregation performs semantic completion and temporal aggregation at both particle and object levels, which spatiotemporally enhance particle representations. (c) With the global attention of the Particle Graph Transformer, $\mathrm { \Delta D y G ^ { 2 } T }$ enables discriminative multi-scale dependencies modeling.

As the data foundation for dynamics modeling, we reconstruct the trackable raw particle sequences from 2D visual observations for the current object (as shown in Figure 2(a)). Specifically, we first apply vanilla 3DGS [5] on the initial frame $\{ I _ { o , 1 } \} _ { o = 1 } ^ { O }$ to obtain the initial 3D Gaussian parameters. For subsequent observation frames, we follow Dyn3DGS [29] to perform frame-wise reconstruction with trackable Gaussians. Appearance descriptors are frozen and reused across frames, as they encode view-dependent object appearance that remains approximately consistent over short temporal intervals. Spatial descriptors are iteratively optimized under multi-view constraints $\mathbf { \bar { \{ } }  I _ { o , 2 } , \dots , I _ { o , t } \mathbf  \} _ { o = 1 } ^ { O }$ . Since Dyn3DGS initializes each frame from the previous one, this process naturally yields temporally trackable spatial descriptors, forming raw particle 3D position sequences $\{ G _ { 1 } , \ldots , G _ { t } \}$ , where $G _ { t } = \{ \mu _ { i } ^ { t } | i \in [ 1 , N ] \}$ , along with corresponding 3D rotation sequences. For notational convenience, we also refer to parameterized 3D Gaussians as particles.

To predict the future motion of objects parameterized by 3D Gaussians, we design a Spatial-Temporal Feature Completion and Aggregation module and a Particle Graph Transformerbased Dynamics Modeling module. As shown in Figure 2(b), we first apply FPS to sample Key Points from the raw particle sequence (i.e., the Raw Point Cloud). Since subsequent dynamics modeling mainly operates on 3D positions, we represent the downsampled Key Points simply by their 3D position sequences $\{ G _ { 1 } ^ { * } , \ldots , G _ { t } ^ { * } \}$ , where $G _ { t } ^ { * } \stackrel { \cdot } { = } \{ \mu _ { i } ^ { * , t } \mathrm { ~ } | \mathrm { ~ } i \in [ \bar { 1 } , N ^ { * } ] \}$ The Spatial-Temporal Feature Completion and Aggregation mechanism subsequently operates on Raw & Key Points to perform particle-level spatial semantic completion and objectlevel dynamic temporal aggregation, resulting in enriched Key Point embeddings $X _ { \mathrm { A g } }$ . We then employ the Particle Graph Transformer to construct direct interaction pathways between Key Points, enabling accurate trajectory modeling and precise displacement prediction $M ^ { * , t }$ . Afterwards, the positions of the Key Points $\hat { G } _ { t + 1 } ^ { * }$ at frame t+1 is updated by adding predicted displacement $M ^ { * , t }$ to $G _ { t } ^ { * }$ (Figure 2(c)). Finally, using Linear Blend Skinning (LBS) [42, 43], we interpolate the predicted Key Point positions $\hat { G } _ { t + 1 } ^ { * }$ to estimate the 3D positions $\hat { G } _ { t + 1 }$ and 3D rotations of raw particles for the next frame. Appearance descriptors are directly reused from the previous frame. This yields the complete 3D Gaussian parameters for future frames. By iteratively incorporating predicted frames into the observation frames, we enable motion prediction over the subsequent ϵ frames. In practice, the range of observation frames is set to 3 (i.e., from t − 2 to t).

## C. Spatial-Temporal Feature Completion and Aggregation

In this section, we enhance particle representations from two complementary perspectives: spatial semantic completion and dynamic temporal aggregation.

As a prerequisite, we repeatedly apply Farthest Point Sampling (FPS) [44] to extract a sequence of Key Points $\{ G _ { 1 } ^ { * } , \ldots , G _ { t } ^ { * } \}$ from the Raw Point Cloud, where FPS iteratively selects points that are farthest from the already sampled set to ensure uniform spatial coverage. Based on the downsampled Key Point sequence, as illustrated in Figure 2(b), we first introduce a Particle-level Spatial Semantic Completion module, which enriches each Key Point with fine-grained local details and geometric structure by jointly integrating neighborhood positional information from the Raw Point

Cloud and relative offset relationships among Key Points. Subsequently, an Object-level Dynamic Temporal Aggregation module is employed to model temporal dynamics, in which a Temporal Disentangling Net (TDN) estimates dominant crossframe disentangling offsets in the latent space to amplify inter-frame differences, followed by a Temporal Attention mechanism that selectively aggregates discriminative features to capture frame-wise temporal evolution cues.

1) Particle-level Spatial Semantic Completion: This module leverages a Position-Aware Attention mechanism to enrich the initial coordinate-level representations by aggregating positional information from neighboring raw particles and the relative offsets among Key Points.

As shown in Figure 2(b), to obtain the coordinate-level initial representation, we first employ Coord Net, which is the 2-layer MLPs with ReLU, to map the Key Point 3D position $\mu _ { i } ^ { * , t } \in G _ { t } ^ { * }$ into coordinate features $X _ { \mathrm { C o } } ^ { t } \in \mathbf { \bar { \mathbb { R } } } ^ { N ^ { * } \times H _ { \mathrm { C o } } }$

Then, we employ PointNet [45] to encode the positions of the k-nearest Raw Points for each Key Point, map them into a unified feature space using a shared MLP, and apply max pooling over the k neighbors to obtain the neighborhood feature $X _ { \mathrm { P o } , i } ^ { t } ~ \in ~ \mathbb { R } ^ { H _ { \mathrm { P o } } }$ . Here, max-pooling serves as a neighborhood aggregation mechanism that preserves the most salient local responses while suppressing potentially noisy information, thereby aggregating permutation-invariant fine-grained local appearance information from the unordered Raw-Point positional neighborhood, ultimately enhancing the spatial representation capability of each Key Point.

Subsequently, we employ a PosDiff Encoder that models the pairwise relative offset between Key Points to capture the directed layout relations of Key Points. In detail, we encode the coordinate differences between Key Points using a 2- layer MLP with ReLU. The pairwise encodings are reduced to nodewise via neighbor-wise mean aggregation and get relative geometric embeddings for Key Points.

Afterwards, we add these relative geometric embeddings to the coordinate features $X _ { \mathrm { C o } } ^ { t }$ and neighborhood features $X _ { \mathrm { P o } } ^ { t } ,$ yielding enhanced coordinate and neighborhood embeddings, $\mathbf { \bar { \mathbf { X } } } _ { \mathrm { C o P } } ^ { t } \in \mathbb { R } ^ { N ^ { * } \times H _ { \mathrm { C o } } }$ and $X _ { \mathrm { P o P } } ^ { t } ~ \in ~ \mathbb { R } ^ { N ^ { * } \times H _ { \mathrm { P o } } }$ , respectively. By injecting identical relative offset cues into both feature streams, the PosDiff Encoder establishes a shared relational reference between complementary semantics that enables consistent reasoning under directed Key Point interactions. In addition, this symmetric enhancement is particularly important for subsequent interaction modeling, where feature updates between Key Points are inherently direction-dependent [46] (e.g., information propagated A→B may differ from B→A). Encoding relative geometry beforehand ensures that such bidirectional interactions are grounded in explicit geometric cues, thereby improving both local detail and structural preservation.

Finally, we design a Multi-head Position-Aware Attention mechanism that enriches the Key Point enhanced coordinate features $X _ { \mathrm { C o P } } ^ { t }$ by aggregating the neighborhood representations $X _ { \mathrm { P o } } ^ { t }$ and the PosDiff-enhanced neighborhood features $X _ { \mathrm { P o P } } ^ { t } ,$ allowing the coordinate embeddings to incorporate fine-grained local detail and relative layout information. Meanwhile, we incorporate a Gaussian-weighted relation prior ω to regularize the attention distribution [47], ensuring that it adheres to plausible geometric correlations among Key Points. For clarity, we illustrate the procedure using the Single-Head formulation,

$$
Q = X _ { \mathrm { C o P } } ^ { t } \mathbf { W } ^ { Q } , K = X _ { \mathrm { P o P } } ^ { t } \mathbf { W } ^ { K } , V = X _ { \mathrm { P o } } ^ { t } \mathbf { W } ^ { V } ,\tag{4}
$$

$$
X _ { \mathrm { I n } } ^ { t } = \mathrm { S o f t M a x } \left( \frac { Q \cdot K ^ { \mathsf { T } } } { \sqrt { H _ { K } } } + \log \left( \omega \right) \right) V ,\tag{5}
$$

where $\mathbf { W } ^ { Q } , \mathbf { W } ^ { K } \ \in \ \mathbb { R } ^ { H _ { \mathrm { C o } } \times H _ { K } }$ and $\mathbf { W } ^ { V } ~ \in ~ \mathbb { R } ^ { H _ { \mathrm { P } _ { 0 } } \times H _ { V } }$ are learnable matrices, $\begin{array} { r } { \omega = \exp \left( - \frac { d ^ { 2 } } { 2 \rho ^ { 2 } } \right) } \end{array}$ is Gaussian-weighted relation, d is the Euclidean distance between Key Points and $\rho$ is a sharpness coefficient. $X _ { \mathrm { I n } } ^ { t } ~ \in ~ \mathbb { R } ^ { N ^ { * } \times H _ { \mathrm { I n } } }$ is the spatial semantic features at frame t. Applying the same procedure to Key Points at frame $t - 2$ and $t - 1$ , we obtain the spatial semantic feature sequence $\{ X _ { \mathrm { I n } } ^ { t - 2 } , X _ { \mathrm { I n } } ^ { t - 1 } , X _ { \mathrm { I n } } ^ { t } \}$

2) Object-level Dynamic Temporal Aggregation: To enrich the particle embeddings with meaningful dynamic evolutions, we aggregate spatial semantic features across frames. However, directly aggregating multi-frame features without explicit temporal disentanglement often leads to severe semantic collapse: features from different frames become highly overlapped in distribution. This phenomenon is observed in our visualizations (see Figure 7) and is aligned with theoretical observations that [48, 49], in the absence of explicit diversity or correspondence constraints, the optimal solution of a representation learner tends to degenerate into an almost constant embedding that maps all inputs to the same region of the latent space. For dynamic modeling, such a collapse drastically impairs temporal discriminability and prevents downstream modules from capturing meaningful temporal information.

To address this issue, building on the strong feature foundation provided by spatial completion, we design a Temporal Disentangling Network (TDN) to reconstruct inter-frame differences in cross-frame representations before temporal aggregation, yielding temporally discriminative embeddings. We intentionally adopt a 3-frame observation window to encourage TDN to disentangle short-range inter-frame variations. This local design constrains the disentangling process within a relatively stable temporal neighborhood, thereby avoiding the large inter-frame distribution discrepancies and instability introduced by long-range temporal alignment. To provide a stable canonical anchor for disentangling offset estimation and temporal difference reconstruction, we first designate frame t − 1 as the reference frame and enforce zero offset, i.e., $X _ { \mathrm { I n D } } ^ { t - 1 } = X _ { \mathrm { I n } } ^ { t - 1 }$ . This symmetric center-frame reference design further alleviates the potential bias that may arise when using a one-sided endpoint frame as the temporal reference. We then concatenate the observed frames’ spatial semantic features, including the reference frame, to form a temporal joint representation $X _ { \mathrm { I n C } }$

$$
\begin{array} { r } { X _ { \mathrm { I n C } } = \mathrm { C o n c a t } \left( X _ { \mathrm { I n } } ^ { t - 2 } , X _ { \mathrm { I n } } ^ { t - 1 } , X _ { \mathrm { I n } } ^ { t } \right) \in \mathbb { R } ^ { N ^ { * } \times 3 H _ { \mathrm { I n } } } , } \end{array}\tag{6}
$$

and passed through a linear layer with tanh activation to estimate the disentangling offsets $\delta ^ { t - 2 } , \delta ^ { t }$ of the non-reference frames $t - 2 .$ , t relative to the reference frame t − 1,

$$
\delta ^ { t - 2 } , \delta ^ { t } = \operatorname { t a n h } \big ( \operatorname { L i n e a r } \big ( X _ { \mathrm { I n C } } \big ) \big ) \in \mathbb { R } ^ { N ^ { * } \times H _ { \mathrm { I n } } } .\tag{7}
$$

These offsets represent the dominant disentangling directions of adjacent-frame spatial semantic representations. By adding the offsets to the spatial features, we pull nonreference representations along these directions, separating them from the entangled spatial features and amplifying interframe differences, thereby obtaining temporally discriminative disentangled features $X _ { \mathrm { I n D } } ^ { t - 2 } , ~ X _ { \mathrm { I n D } } ^ { t }$

$$
X _ { \mathrm { I n D } } ^ { t - 2 } = X _ { \mathrm { I n } } ^ { t - 2 } + \delta ^ { t - 2 } , \ X _ { \mathrm { I n D } } ^ { t } = X _ { \mathrm { I n } } ^ { t } + \delta ^ { t } .\tag{8}
$$

Subsequently, we apply a Temporal Attention module to aggregate the disentangled Key Point features across frames adaptively, capturing frame-wise temporal evolution cues. This mechanism assigns frame-wise importance scores $s _ { i }$ based on a nonlinear transformation, enabling the model to emphasize temporally informative cues while suppressing redundant ones,

$$
X _ { \mathrm { A g } } = \sum _ { i = t - 2 } ^ { t } \frac { \exp \left( s _ { i } \right) } { \sum _ { j = t - 2 } ^ { t } \exp \left( s _ { j } \right) } X _ { \mathrm { I n D } } ^ { i } ,\tag{9}
$$

$$
s _ { i } = \mathrm { L i n e a r } \left( \operatorname { t a n h } \left( \mathrm { L i n e a r } \left( X _ { \mathrm { I n D } } ^ { i } \right) \right) \right) ,\tag{10}
$$

where the scalar scores $s _ { i }$ encode the temporal saliency of each frame, and the resulting aggregated representation $X _ { \mathrm { A g } } ~ \in ~ \mathbb { R } ^ { N ^ { * } \times H _ { \mathrm { A g } } }$ captures the dynamically weighted fusion of the temporally disentangled features.

## D. Dynamics Modeling Based on Particle Graph Transformer

In this section, we describe how the Particle Graph Transformer is utilized to capture multi-scale interaction patterns for preserving discriminative particle-wise information, and to predict the translation vectors $M ^ { * , t } \in \mathbb { R } ^ { N ^ { * } \times 3 }$ of Key Points at frame t, estimating the position $\hat { G } _ { t + 1 } ^ { * }$ at frame $t + 1$

As shown in Figure 2(c), inspired by [50], we introduce a Particle Graph Transformer. At each iteration step, we construct a Particle Graph from the current Key Point positions by connecting each Key Point to its $\mathrm { t o p } { - } k _ { G }$ nearest neighbors within a distance threshold $d _ { e } .$ . The resulting binary adjacency is vectorized and encoded as the learnable edge embedding $e _ { i j }$ between nodes i and $j .$ After each Key Point position update, the graph is reconstructed using the newly predicted positions. The pseudocode for the Particle Graph construction and evolution process is provided in the Supplementary Material. We then perform global interaction modeling through Graph Attention. Specifically, the aggregated Key Point features $X _ { \mathrm { A g } }$ are projected into the key $\boldsymbol { k } ^ { ( 0 ) } \in \mathbb { R } ^ { N ^ { * } \times d _ { k } }$ , query $\boldsymbol { q } ^ { ( 0 ) } ~ \in ~ \bar { \mathbb { R } } ^ { N ^ { * } \times \bar { d _ { q } } }$ , and value $\boldsymbol { v } ^ { ( 0 ) } \in \overline { { \mathbb { R } ^ { N ^ { * } \times d _ { v } } } }$ using separate linear layers. The attention scores α are computed using a scaled dot-product function:

$$
\alpha _ { i j } ^ { ( l ) } = \frac { \langle q _ { i } ^ { ( l ) } , k _ { j } ^ { ( l ) } + e _ { i j } \rangle } { \sum _ { u \in \mathcal { N } ( i ) } \langle q _ { i } ^ { ( l ) } , k _ { u } ^ { ( l ) } + e _ { i u } \rangle } ,\tag{11}
$$

where $\begin{array} { r } { \langle \boldsymbol { q } , \boldsymbol { k } \rangle ~ = ~ \exp \left( \frac { \boldsymbol { q } ^ { \intercal } \boldsymbol { k } } { \sqrt { d _ { k } } } \right) } \end{array}$ , i and $j$ are the endpoints of an edge, $i \in N ^ { * } , \dot { N }$ represents the set of neighbors of node i, and $l = 1 , 2 , \ldots , L$ denotes the index of the Particle Graph Transformer layer. Finally, we selectively aggregate node features over the entire graph to obtain $\mathbf { \Psi } _ { X } ( l ) \mathbf { \Psi } \dot { \in } \mathbb { R } ^ { N ^ { * } \times H _ { \mathrm { G } } } ;$

$$
X ^ { ( l ) } = \sum _ { i \in N ^ { * } } \sum _ { j \in N ( i ) } \alpha _ { i , j } ^ { ( l ) } ( v _ { j } ^ { ( l ) } + e _ { i , j } ) .\tag{12}
$$

The Particle Graph Transformer leverages global attention over the particle graph to directly model long-range dependencies that are inaccessible to purely local message passing. By doing ${ \bf { S O } } ,$ it mitigates representation homogenization and preserves discriminative particle-wise features, providing a robust representational basis for accurately decoding complex motion trajectories. In addition, to alleviate feature homogenization that may arise during multi-scale message passing, we introduce Gated Residual Connections between successive transformer layers to retain informative signals selectively while preventing over-smoothing. Further architectural details are provided in the Supplementary Material.

Finally, the output representation $X ^ { ( L ) }$ from the Particle Graph Transformer is fed into a 3-layer MLP with ReLU to decode the Key Point displacement field $M ^ { * , t } \in \mathbb { R } ^ { N ^ { * } \times 3 }$ . This displacement field provides a direct parametric estimate of the particle-wise motion induced by the underlying dynamics. The predicted Key Point positions at frame $t + 1$ are then obtained via simple forward integration:

$$
\begin{array} { r l } & { \hat { G } _ { t + 1 } ^ { * } = \left\{ \hat { \mu } _ { i } ^ { * , t + 1 } \Big | 1 \leq i \leq N ^ { * } \right\} , } \\ & { \hat { \mu } _ { i } ^ { * , t + 1 } = { \mu } _ { i } ^ { * , t } + { M } _ { i } ^ { * , t } . } \end{array}\tag{13}
$$

This decoding step translates the learned latent dynamic representations into explicit motion trajectories. The details of how the 3D positions $\hat { G } _ { t + 1 }$ and 3D rotations of raw particles for the next frame are interpolated from the Key Points $\hat { G } _ { t + 1 } ^ { * }$ using LBS are provided in the Supplementary Material. Here, LBS serves as a dense motion propagation mechanism that smoothly transfers predicted Key Point motions to raw particles while preserving coherent object deformation and rendering consistency.

## E. Training Strategy

We minimize the Mean Squared Error (MSE) between the final next-frame raw particle positions $\hat { G } _ { t + i }$ and the ground truth $G _ { t + i }$ over the subsequent ϵ frames to train $\mathrm { { D y G ^ { 2 } T } }$

$$
\mathcal { L } _ { \mathrm { p r e d } } = \sum _ { i = 1 } ^ { \epsilon } \left\| \hat { G } _ { t + i } - G _ { t + i } \right\| ^ { 2 } .\tag{14}
$$

In practice, we set $\epsilon = 5$ to provide a short-horizon yet sufficiently informative temporal window. Supervising multiple future frames encourages the $\mathrm { \Delta D y G ^ { 2 } T }$ to generate dynamically stable and drift-resistant dynamics reasoning, rather than overfitting to a single-step displacement.

## IV. EXPERIMENT

## A. Experiment Settings

1) Dataset: We evaluate $\mathrm { \Delta D y G ^ { 2 } T }$ on the Spring-Gaus dataset [2] and our Unity3D-Heterogeneous (Unity3D-H) dataset. Spring-Gaus (synthetic) contains elastic objects with diverse materials and appearances, recorded as 30-frame 512× 512 videos from 10 views, with MPM-simulated trajectories as 3D ground truth. Its real-world subset provides 20-frame 1920 × 1080 3 views videos of five toys dropped from rest, together with 50∼70 static multiview images for appearance reconstruction. Unity3D-H is collected using the Unity3D engine [51], featuring a polyhedral object composed of two rigidly connected elastic materials. Released from random upper-hemisphere positions without initial velocity, the object exhibits material-dependent bouncing trajectories. The dynamic sequence is rendered as a 30-frame $2 0 9 8 \times 1 3 2 7$ video from 10 views. All cameras across the above datasets are fixed and uniformly distributed over the upper hemisphere surrounding the dynamic scene.

Following the Spring-Gaus [2], $\mathrm { \Delta D y G ^ { 2 } T }$ learns dynamics from the first 20 frames (10 for real-world Spring-Gaus) of each observation, while the final 10 frames, which are unseen during training, are reserved for dynamic modeling and reasoning evaluation. Additionally, following prior work [2, 52], we use GroundingDINO [53] and SAM [54] to extract dynamicobject masks and remove background interference.

2) Baselines: Following prior work, we evaluate $\mathrm { \Delta D y G ^ { 2 } T }$ from two perspectives. For dynamic reconstruction, we compare against Spring-Gaus [2], which performs per-frame geometry optimization using a spring-mass model initialized from the first frame. For dynamics modeling, we adopt GS-Dynamics [4], a differentiable network-based pipeline combining Dyn3DGS with GNN-based dynamics prediction, as well as Spring-Gaus as baselines. Furthermore, for the challenging heterogeneous-material dynamics scenario, we additionally introduce 2 strong physics-prior baselines. PAC-NeRF [8] characterizes dynamic scenes by integrating neural radiance fields with continuum mechanics modeling, incorporating explicit physical parameters and differentiable simulation constraints. GIC [3], in contrast, assists object dynamics modeling through physical parameter inversion.

All baselines are re-trained and re-evaluated using the authors’ recommended hyperparameters, and we report their best-performing results. To ensure fairness, both GS-Dynamic and $\mathrm { \Delta D y G ^ { 2 } T }$ perform dynamics modeling using the same Dyn3DGS reconstruction results.

3) Metrics: Following prior work [4], we adopt CD, the bidirectional L2 distance between predicted and ground-truth point clouds, and EMD, the minimal transport cost between two point sets, as our 3D trajectory evaluation metrics. For 2D appearance assessment, we measure the similarity between rendered and ground-truth images using PSNR, SSIM [55], and LPIPS [56] across multiple views. PSNR and SSIM quantify pixel-level discrepancies, whereas LPIPS leverages AlexNet-based perceptual features to capture human-aligned differences in overall visual appearance. Because baseline methods do not release metric-evaluation code, we reimplement all metrics in a unified framework to ensure fair comparison. 3D metrics are averaged over all evaluation frames, while 2D metrics are averaged across views per frame and then across all evaluation frames.

4) Implementation Details: We train $\mathrm { \Delta D y G ^ { 2 } T }$ for 1000 epochs, with each epoch consisting of 100 iterations. The PointNet neighborhood size k is 16 $( k = 8$ for Apple and Toothpaste), and the sharpness parameter $\rho$ is chosen as half of the minimum pairwise distance between Key Points. To increase data diversity and improve robustness to reconstruction noise, we augment each dynamic reconstruction trajectory into 30 instances by adding a uniformly sampled perturbation from $[ - 0 . 3 , 0 . 3 ]$ to the Raw Point Cloud of every frame. The test split uses only the clean, unaugmented reconstructions to ensure unbiased evaluation. All augmented trajectories are divided into training and test sets at a 4:1 ratio. All experiments are conducted with a single NVIDIA RTX 4090 GPU.

TABLE I  
QUANTITATIVE RESULTS OF DYNAMIC RECONSTRUCTION FOR $\mathrm { D Y G ^ { 2 } T }$ AND BASELINES ON SPRING-GAUS SYNTHETIC DATASET.
<table><tr><td rowspan="2"></td><td colspan="2">CD↓</td><td colspan="2">EMD↓</td></tr><tr><td>Objects Spring-Gaus</td><td> $\mathrm { D y G ^ { 2 } T ( O u r s ) }$ </td><td>Spring-Gaus</td><td> $\mathrm { D y G ^ { 2 } T ( O u r s ) }$ </td></tr><tr><td>Torus</td><td>0.012</td><td>0.008</td><td>0.003</td><td>0.001</td></tr><tr><td>Cross</td><td>0.016</td><td>0.010</td><td>0.005</td><td>0.002</td></tr><tr><td>Cream</td><td>0.014</td><td>0.012</td><td>0.007</td><td>0.005</td></tr><tr><td>Apple</td><td>0.014</td><td>0.011</td><td>0.006</td><td>0.003</td></tr><tr><td>Paste</td><td>0.011</td><td>0.008</td><td>0.003</td><td>0.002</td></tr><tr><td>Chess</td><td>0.017</td><td>0.010</td><td>0.007</td><td>0.002</td></tr><tr><td>Banana</td><td>0.049</td><td>0.007</td><td>0.027</td><td>0.002</td></tr><tr><td>Mean</td><td>0.019</td><td>0.010</td><td>0.008</td><td>0.003</td></tr></table>

## B. Dynamic Reconstruction of Dynamic Objects

To evaluate the dynamic reconstruction quality in $\mathrm { \Delta D y G ^ { 2 } T , }$ we follow standard practice and report CD and EMD on the recovered 3D trajectories. As shown in Table I, our reconstruction module, built upon a strong dynamic 3D Gaussian representation backbone Dyn3DGS [29], produces accurate and stable 3D trajectories across various objects.

Although dynamic reconstruction is not the primary focus of this work, its reliability is essential because these trajectories serve as the sole supervision signal for the downstream dynamics modeling. The quantitative results verify that the reconstructed dynamic sequences are sufficiently precise to support the downstream modules. Additional qualitative results are provided in the Supplementary Material.

## C. Dynamics Modeling and Reasoning of Dynamic Objects

1) Synthetic Dynamic Scenes: To assess dynamics modeling and reasoning, we perform 3D trajectory forecasting on the Spring-Gaus synthetic datasets. As summarized in Table II, $\mathrm { \Delta D y G ^ { 2 } T }$ delivers strong performance on this benchmark, especially in 3D trajectory metrics (CD and EMD), demonstrating its ability to capture physically faithful $\mathrm { { d y } \mathrm { { - } } }$ namics. While $\mathrm { \Delta D y G ^ { 2 } T }$ ranks second on Cross in PSNR and SSIM, it surpasses all baselines in LPIPS, which more closely reflects human perceptual similarity. Qualitative results in Figure 3 further show that predictions from Spring-Gaus [2] and GS-Dynamics [4] suffer from significant positional drift, whereas $\mathrm { \Delta D y G ^ { 2 } T }$ produces more accurate dynamic trajectories and sharper appearance details, demonstrating strong generalization across diverse object geometries.

2) Real-world and Cross-benchmark Generalization: Furthermore, as shown in Table III, the evaluation on the Spring-Gaus real-world and our Unity3D-H datasets demonstrates that $\mathrm { \Delta D y G ^ { 2 } T }$ exhibits strong real-world generalization and can be readily transferred to other benchmarks. Notably, while GS-Dynamics achieves leading performance on the relatively simple Potato, it falls behind on toys with more complex geometries, such as Burger, where $\mathrm { D y G ^ { 2 } T ^ { * } s }$ spatiotemporally completed particle features and multi-scale propagation paths provide a clear advantage.

TABLE II  
QUANTITATIVE RESULTS OF DYNAMICS MODELING AND REASONING ON SPRING-GAUS SYNTHETIC DATASET.
<table><tr><td>Metrics</td><td>Methods</td><td>Torus</td><td>Cross</td><td>Cream</td><td>Apple</td><td>Paste</td><td>Chess</td><td>Banana</td><td>Mean</td></tr><tr><td rowspan="3">CD↓</td><td>Spring-Gaus</td><td>0.033</td><td>0.046</td><td>0.032</td><td>0.047</td><td>0.068</td><td>0.053</td><td>0.184</td><td>0.066</td></tr><tr><td> $\mathrm { \bar { G S - D y n a m i c s } }$ </td><td>0.073</td><td>0.122</td><td>0.154</td><td>0.043</td><td>0.227</td><td>0.284</td><td>0.328</td><td>0.176</td></tr><tr><td> $\mathrm { D y G ^ { 2 } T ( O u r s ) }$ </td><td>0.029</td><td>0.038</td><td>0.027</td><td>0.028</td><td>0.038</td><td>0.050</td><td>0.055</td><td>0.039</td></tr><tr><td rowspan="3">EMD↓</td><td>Spring-Gaus</td><td>0.014</td><td>0.024</td><td>0.023</td><td>0.029</td><td>0.035</td><td>0.027</td><td>0.091</td><td>0.035</td></tr><tr><td> $\mathrm { \bar { G S - D y n a m i c s } }$ </td><td>0.033</td><td>0.062</td><td>0.097</td><td>0.021</td><td>0.164</td><td>0.200</td><td>0.171</td><td>0.107</td></tr><tr><td> $\mathrm { D y G ^ { 2 } T ( O u r s ) }$ </td><td>0.013</td><td>0.018</td><td>0.013</td><td>0.015</td><td>0.020</td><td>0.021</td><td>0.029</td><td>0.019</td></tr><tr><td rowspan="3">PSNR↑</td><td>Spring-Gaus</td><td>12.220</td><td>11.993</td><td>11.267</td><td>17.443</td><td>11.016</td><td>11.305</td><td>15.949</td><td>13.028</td></tr><tr><td>GS-Dynamics</td><td>13.450</td><td>10.621</td><td>12.647</td><td>19.632</td><td>11.506</td><td>11.758</td><td>16.622</td><td>13.748</td></tr><tr><td> $\mathrm { D y G ^ { 2 } T ( O u r s ) }$ </td><td>14.048</td><td>11.632</td><td>14.765</td><td>20.477</td><td>14.698</td><td>15.653</td><td>17.904</td><td>15.587</td></tr><tr><td rowspan="3">SSIM↑</td><td>Spring-Gaus</td><td>0.850</td><td>0.876</td><td>0.709</td><td>0.828</td><td>0.775</td><td>0.755</td><td>0.865</td><td>0.808</td></tr><tr><td>G$-Dynamics</td><td>0.876</td><td>0.842</td><td>0.763</td><td>0.887</td><td>0.802</td><td>0.749</td><td>0.880</td><td>0.828</td></tr><tr><td> $\mathrm { D y G ^ { 2 } T ( O u r s ) }$ </td><td>0.895</td><td>0.871</td><td>0.875</td><td>0.907</td><td>0.887</td><td>0.873</td><td>0.919</td><td>0.889</td></tr><tr><td rowspan="3">LPIPS↓</td><td>Spring-Gaus</td><td>0.349</td><td>0.303</td><td>0.370</td><td>0.230</td><td>0.332</td><td>0.335</td><td>0.250</td><td>0.310</td></tr><tr><td>G$-Dynamics</td><td>0.197</td><td>0.280</td><td>0.324</td><td>0.163</td><td>0.317</td><td>0.306</td><td>0.210</td><td>0.257</td></tr><tr><td> $\mathrm { D y G ^ { 2 } T ( O u r s ) }$ </td><td>0.139</td><td>0.220</td><td>0.189</td><td>0.131</td><td>0.178</td><td>0.207</td><td>0.122</td><td>0.171</td></tr></table>

TABLE III

QUANTITATIVE RESULTS OF DYNAMICS MODELING AND REASONING ON SPRING-GAUS REAL-WORLD DATASET AND OUR UNITY3D-H DATASET.
<table><tr><td rowspan="2">Metrics</td><td rowspan="2">Methods</td><td colspan="5">Spring-Gaus real-world</td><td>Unity3D-H</td></tr><tr><td>Dog</td><td>Potato</td><td>Pig</td><td>Burger</td><td>Bun</td><td>Polyhedron</td></tr><tr><td rowspan="3">PSNR↑</td><td>Spring-Gaus</td><td>21.499</td><td>20.881</td><td>21.136</td><td>21.026</td><td>20.456</td><td>31.027</td></tr><tr><td>G$-Dynamics</td><td>26.141</td><td>28.623</td><td>27.114</td><td>27.969</td><td>26.929</td><td>31.352</td></tr><tr><td> $\mathrm { D y G ^ { 2 } T ( O u r s ) }$ </td><td>27.676</td><td>27.933</td><td>27.750</td><td>30.645</td><td>27.197</td><td>31.768</td></tr><tr><td rowspan="3">SSIM↑</td><td>Spring-Gaus</td><td>0.987</td><td>0.985</td><td>0.986</td><td>0.985</td><td>0.984</td><td>0.985</td></tr><tr><td>GS-Dynamics</td><td>0.988</td><td>0.989</td><td>0.989</td><td>0.988</td><td>0.988</td><td>0.987</td></tr><tr><td> $\mathrm { D y G ^ { 2 } T ( O u r s ) }$ </td><td>0.991</td><td>0.987</td><td>0.989</td><td>0.994</td><td>0.988</td><td>0.990</td></tr><tr><td rowspan="3">LPIPS↓</td><td>Spring-Gaus</td><td>0.030</td><td>0.032</td><td>0.031</td><td>0.031</td><td>0.032</td><td>0.020</td></tr><tr><td>G$-Dynamics</td><td>0.023</td><td>0.020</td><td>0.020</td><td>0.022</td><td>0.020</td><td>0.018</td></tr><tr><td> $\mathrm { D y G ^ { 2 } T ( O u r s ) }$ </td><td>0.019</td><td>0.021</td><td>0.017</td><td>0.012</td><td>0.018</td><td>0.015</td></tr></table>

TABLE IV

QUANTITATIVE RESULTS OF DYG<sup>2</sup>T, ITS VARIANTS, AND BASELINES FOR HETEROGENEOUS TORUS DYNAMICS MODELING AND REASONING ON THE SPRING-GAUS SYNTHETIC DATASET.
<table><tr><td>Methods</td><td>CD↓</td><td>EMD↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td></tr><tr><td>Spring-Gaus</td><td>0.030</td><td>0.012</td><td>12.197</td><td>0.850</td><td>0.350</td></tr><tr><td>GS-Dynamics</td><td>0.116</td><td>0.049</td><td>13.342</td><td>0.861</td><td>0.235</td></tr><tr><td>PAC-NeRF</td><td>0.284</td><td>0.203</td><td>15.188</td><td>0.900</td><td>0.187</td></tr><tr><td>GIC</td><td>0.116</td><td>0.132</td><td>14.009</td><td>0.882</td><td>0.217</td></tr><tr><td> $\mathrm { D y G ^ { 2 } T _ { n o i s y } }$ </td><td>0.035</td><td>0.016</td><td>13.647</td><td>0.878</td><td>0.188</td></tr><tr><td> $\mathrm { D y G ^ { 2 } T ( O u i s ) }$ </td><td>0.015</td><td>0.000</td><td>14.080</td><td>0.893</td><td>0.129</td></tr></table>

Compared with the synthetic and real subsets of Spring-Gaus, the performance gain of $\mathrm { \Delta D y G ^ { 2 } T }$ on Unity3D-H is smaller due to differences in their simulation and observation pipelines. Spring-Gaus synthetic subset, produced via a staged simulate-then-render process, introduces additional noise, while the real subset further suffers from deformation noise and non-ideal physical interactions. These perturbations amplify semantic collapse, thereby accentuating $\mathrm { D y G ^ { 2 } T ^ { \dag } s }$ advantage in modeling multi-scale interaction using temporally discriminative, complete particle features. By contrast, Unity3D-H is generated through a physically consistent onestep simulator, producing cleaner motion and weaker temporal collapse, so the relative improvement appears smaller. Nevertheless, $\mathrm { \Delta D y G ^ { 2 } T }$ achieves stable gains, and Unity3D-H further demonstrates its cross-benchmark generalization.

TABLE V  
QUANTITATIVE RESULTS OF DYNAMICS MODELING AND REASONING PHYSICAL CONSISTENCY EVALUATION ON THE HETEROGENEOUS TORUS.
<table><tr><td>Methods</td><td>LSE↓</td><td>SPC↑</td><td>ACE↓</td></tr><tr><td>GS-Dynamics</td><td>0.499</td><td>-0.248</td><td>0.017</td></tr><tr><td> $\mathrm { D y G ^ { 2 } T ( O u r s ) }$ </td><td>0.225</td><td>0.089</td><td>0.016</td></tr></table>

3) Robustness to Heterogeneous Objects and Noise Input: We conduct experiments on a Heterogeneous Torus with complex physical properties. As shown in Table IV and Figure $4 , \mathrm { D y G ^ { 2 } T }$ delivers more accurate trajectory predictions by leveraging spatiotemporally completed particle representations and multi-scale interaction modeling, enabling the model to capture subtle variations in material response. Relative to strong physics-prior approaches such as PAC-NeRF and GIC, $\mathrm { \Delta D y G ^ { 2 } T }$ is still able to preserve 3D geometric distribution quality more accurately, particularly exhibiting more stable dynamics modeling capability in terms of particle trajectories and deformation structures. The results on the Unity3D-H dataset in Table III also validate $\mathrm { D y G ^ { 2 } T ^ { 2 } s }$ scalability to heterogeneous objects across different benchmarks.

Additionally, we introduce 3 physics-consistency metrics

Time  
Time  
Time  
![](images/0601780e3ea7bf0c85398c100b54182aba4adf1eb78ab58638dd82ded9ad347d.jpg)  
(a) Banana

![](images/ebef5ea90dc9416ca0e3e3c29c04924b000e4537a1719a7677b869d236d7cbd0.jpg)  
(b) Toothpaste

Time  
![](images/fd0e94f95d3cd6b63e798e6b05cc7180d68fdae9f91f70e1eafc310bb5d4903a.jpg)  
(c) Apple

Time

![](images/6ba9e10d0f425e44d563c3d8b215bed2a74632c6371251d1cd240203a75969f7.jpg)  
(d) Cross  
Fig. 3. Qualitative results of dynamics reasoning. (a), (b), (c), and (d) correspond to Banana, Toothpaste, Apple, and Cross, respectively.<sub>u</sub>t<sup>h</sup>

![](images/d3fefc4440bd28bc9d7f159d53b01abaf934b7152187714ca715fe0632d16182.jpg)  
Fig. 4. Qualitative results of dynamics modeling and reasoning on thee Heterogeneous Torus. The spatial positions of dynamic objects are calibrated/<sup>o</sup> using a consistent coordinate system.<sub>e</sub><sup>r</sup>

sensitive to heterogeneous objects to further evaluate dynamicst<sup>r</sup> reasoning on the Heterogeneous Torus, including Local Stretch<sup>w</sup> Error (LSE), Strain Pattern Correlation (SPC), and Acceleration Consistency Error (ACE). Specifically, LSE measures the consistency of local stretch with Ground Truth, SPC evaluates the spatial correlation of high/low strain regions with Ground Truth, and ACE assesses second-order temporal acceleration consistency with Ground Truth. Detailed formulations are provided in the supplementary material. As shown in Table V, DyG<sup>2</sup>T achieves superior performance on these metrics, indicating its ability to better preserve local deformation behaviors, accurately capture the spatial distribution of strain intensity, and maintain consistent temporal acceleration.

As shown in Table IV, we additionally test robustness to imperfect observations by injecting noise into central trajectories and applying non-rigid distortions to the reconstructed point clouds. Thanks to the disentangling and temporal discriminability Key Point features, $\mathrm { D y G ^ { 2 } T _ { n o i s y } }$ exhibits only minor degradation in 3D trajectory prediction, while its 2D appearance evaluation still surpasses all baselines.

## D. Ablation Studies

To validate the effectiveness of $\mathrm { D y G ^ { 2 } T ^ { 2 } s }$ core components and assess optimal parameter settings, we conduct ablation studies on the Spring-Gaus synthetic dataset.

![](images/b0ae8a6348ba6c43ac75ba4fb79a5c535766423ee9290c3fab2ce20428b026d1.jpg)  
Fig. 5. Qualitative results of module ablation study. The vertical axis shows the object motion’s temporal evolution.

TABLE VI  
QUANTITATIVE RESULTS OF MODULE ABLATION STUDY.
<table><tr><td>Methods</td><td>CD↓</td><td>EMD↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td></tr><tr><td>w/o spatial</td><td>0.354</td><td>0.203</td><td>16.174</td><td>0.875</td><td>0.220</td></tr><tr><td>w/o temporal</td><td>0.227</td><td>0.120</td><td>16.493</td><td>0.882</td><td>0.198</td></tr><tr><td>w/o transformer</td><td>0.197</td><td>0.110</td><td>16.264</td><td>0.875</td><td>0.201</td></tr><tr><td> $\mathrm { D y G ^ { 2 } T ( O u r s ) }$ </td><td>0.055</td><td>0.029</td><td>17.904</td><td>0.919</td><td>0.122</td></tr></table>

1) The Modules of $D y G ^ { 2 } T$ : The ablation study in Table VI and Figure 5 reveals the contribution of each core component in $\mathrm { \Delta D y G ^ { 2 } T }$ Removing the spatial completion module (w/o spatial), which prevents Key Points from aggregating positional cues from neighboring raw particles and pairwise Key Points, leads to a substantial drop in accuracy. This validates the efficacy of our completed particle representations in restoring fine-grained local and geometric structural cues that sparse Key Point sampling alone cannot preserve.

Eliminating temporal disentangling (TDN) and Temporal Attention (w/o temporal), and instead relying on naive feature concatenation, also results in notable degradation. Without disentangling and temporal discriminability Key Point features, the model fails to maintain coherent cross-frame temporal information, highlighting the critical role of TDN followed by Temporal Attention aggregation.

Finally, replacing the Particle Graph Transformer with a vanilla GNN (w/o transformer) severely limits performance despite having access to the same spatiotemporally completed features. This demonstrates that accurate dynamics modeling hinges on establishing multi-scale interaction paths and capturing long-range dependencies, capabilities that standard local GNNs cannot provide. The qualitative results in Figure 6 further reinforce this conclusion. Due to the heterogeneous material properties of the Torus, the w/o transformer variant accumulates noticeable errors in the later stages of dynamics reasoning (Figure 6(a)), leading to visible artifacts in the inferred appearance (Figure 6(b)).

![](images/5e7bc1dc0e5743cf0be7fc9373edd73c189361046706b758ce32918c2345358f.jpg)  
Fig. 6. Qualitative results of the Particle Graph Transformer ablation study. (a) $\mathrm { { D y G ^ { 2 } T } }$ and its w/o transformer variants on the dynamics modeling and reasoning task of the Heterogeneous Torus. (b) Local zoom-in views for evaluating the fine-grained local details.

2) Particle-level Spatial Semantic Completion: To assess the role of spatial particle completion, we vary the neighborhood size k used to aggregate Raw Point features $X _ { \mathrm { P o } } ^ { t } .$ As shown in Table VII (Row 1&2), using too small a neighborhood $\mathrm { ( D y G ^ { 2 } T } _ { k = 8 } )$ prevents each Key Point from capturing sufficient fine-grained local details from neighboring raw particles, weakening the spatial completion process and causing notable degradation in trajectory prediction. Conversely, an overly large neighborhood $( \mathrm { D y } \mathrm { G } ^ { 2 } \mathrm { T } _ { k = 3 2 } )$ introduces redundant or non-informative points, diluting the fine-grained local signals that the spatial module aims to preserve. Empirically, $k ~ = ~ 1 6$ provides the best balance (k = 8 for Apple and Toothpaste), enabling each Key Point to reconstruct finegrained local details effectively.

To validate the local Raw Point neighborhood aggregation strategy, we compare max pooling with average pooling (PN Avg) and attention-based pooling (PN Att) while keeping all other settings unchanged. As shown in Table VII (Row 3&4), max pooling (Row 11) achieves the best performance across all metrics. This indicates that max pooling provides a more discriminative and noise-robust summary of local Raw Point responses for $\mathrm { { D y G ^ { 2 } T } }$

We further ablate the components within Particle-level Spatial Semantic Completion. As shown in Table VII (Row 5&6), removing the PosDiff Encoder (w/o PosDiff) leads to the largest degradation, confirming that explicitly encoding pairwise Key Point offsets is essential for reconstructing accurate geometric structure. Without these relative positional cues, the model loses its ability to anchor local detail features to the correct spatial layout, severely impairing interaction modeling and trajectory prediction. Similarly, the w/o PosAware variant demonstrates that omitting the Multi-head Position-Aware Attention weakens the alignment between coordinate embeddings and neighborhood features.

TABLE VII  
ABLATION STUDY ON NEIGHBORHOOD RANGES k (ROW 1&2), RAW POINTS NEIGHBORHOOD AGGREGATION STRATEGY (ROW 3&4), THE PARTICLE-LEVEL SPATIAL SEMANTIC COMPLETION (ROW 5&6), AND THE OBJECT-LEVEL DYNAMIC TEMPORAL AGGREGATION (ROW 7∼10).
<table><tr><td></td><td>Methods</td><td>CD↓</td><td>EMD↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td></tr><tr><td>1</td><td> $\mathrm { D y G ^ { 2 } T } _ { k = 8 }$ </td><td>0.244</td><td>0.125</td><td>16.368</td><td>0.878</td><td>0.207</td></tr><tr><td>2</td><td> $\mathrm { D y G ^ { 2 } T } _ { k = 3 2 }$ </td><td>0.169</td><td>0.092</td><td>16.365</td><td>0.879</td><td>0.190</td></tr><tr><td>3</td><td>PN Avg</td><td>0.107</td><td>0.240</td><td>16.071</td><td>0.871</td><td>0.242</td></tr><tr><td>4</td><td>PN Att</td><td>0.116</td><td>0.246</td><td>16.012</td><td>0.869</td><td>0.246</td></tr><tr><td>5</td><td>w/o PosDiff</td><td>0.372</td><td>0.178</td><td>16.742</td><td>0.886</td><td>0.195</td></tr><tr><td>6</td><td>w/o PosAware</td><td>0.107</td><td>0.056</td><td>17.613</td><td>0.912</td><td>0.139</td></tr><tr><td>7</td><td>w/o TDN</td><td>0.107</td><td>0.240</td><td>16.071</td><td>0.871</td><td>0.242</td></tr><tr><td>8</td><td>LSTM</td><td>0.117</td><td>0.055</td><td>17.002</td><td>0.893</td><td>0.166</td></tr><tr><td>9</td><td>avg pool</td><td>0.096</td><td>0.052</td><td>17.344</td><td>0.906</td><td>0.145</td></tr><tr><td>10</td><td>max pool</td><td>0.365</td><td>0.195</td><td>16.290</td><td>0.874</td><td>0.221</td></tr><tr><td>11</td><td> $\mathrm { D y G ^ { 2 } T ( O u r s ) }$ </td><td>0.055</td><td>0.029</td><td>17.904</td><td>0.919</td><td>0.122</td></tr></table>

TABLE VIII

QUANTITATIVE RESULTS OF DYNAMICS MODELING AND REASONING UNDER DIFFERENT OBSERVATION WINDOW SIZES.
<table><tr><td>Obs. Windows</td><td>CD↓</td><td>EMD↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td></tr><tr><td>5</td><td>0.018</td><td>0.090</td><td>16.709</td><td>0.887</td><td>0.186</td></tr><tr><td>3</td><td>0.055</td><td>0.029</td><td>17.904</td><td>0.919</td><td>0.122</td></tr></table>

3) Object-level Dynamic Temporal Aggregation: We ablate the Temporal Disentangling Net (TDN) to examine its role in mitigating cross-frame feature collapse and producing temporally discriminative Key Point representations. As shown in Table VII (Row 7∼10), removing TDN (w/o TDN) leads to a drop in prediction accuracy. Figure 7 provides a deeper analysis. The t-SNE visualizations in (a) and (b) show that before TDN, Key Point embeddings from frames t − 2, t − 1, and t collapse into entangled clusters, evidencing the temporal semantic drift. In contrast, TDN effectively canonicalizes these representations into 3 clearly separated, temporally ordered manifolds, demonstrating successful disentangling. Similarly, the cosine similarity heatmaps in (c) and (d) reveal that TDN substantially reduces spurious cross-frame correlations while strengthening intra-frame consistency, precisely matching the goal of temporally discriminative features. We also compare our Temporal Attention aggregation with LSTM and poolingbased baselines. While these alternatives either blur temporal distinctions (pooling) or accumulate redundant correlations (LSTM), Temporal Attention leverages disentangled TDN features to focus on motion-relevant temporal cues, achieving the best performance.

To further examine the stability of TDN, we extend the ob servation window from 3 to 5 frames while still using the temporal center as the reference frame to preserve symmetric local disentangling; the ablation study on selecting the center frame as the reference is provided in the Supplementary Material. As shown in Table VIII, the results indicate that the default 3-frame setting remains more effective, indicating that TDN is better suited to local temporal canonicalization, whereas a longer window introduces larger cross-frame discrepancy and makes disentangling less stable. We further partition the evaluation frames into collision and non-collision subsets and report 2D rendering metrics separately. The results in Table IX show that $\mathrm { \Delta D y G ^ { 2 } T }$ maintains strong predictive quality even on collision frames, suggesting that the proposed temporal disentangling remains effective under abrupt contact-induced motion changes. The slightly better performance on collision frames is likely due to their stronger and more coherent shortterm motion cues, whereas non-collision frames often contain weaker residual deformations that are more susceptible to rollout accumulation errors.

![](images/fd67abf9010e60fe1b9eb8722eb83682afd711d0bd5e10176efe99abeaa0289e.jpg)

![](images/d87f7ffdef2d92eb0eb411f3472598a55e437fbc7969dfafd9964423ddba1f98.jpg)  
(a)

![](images/e04c2954a16d9150ea1008430fe6cbfaf087991f769ce9998075020a1d262fe2.jpg)

(b)  
![](images/14a71c635d5a430abef906db70b7aa2b53afa58cf638fc7d4e6b7cb1b25fbb97.jpg)  
Fig. 7. Visualization of the Temporal Disentangling Net (TDN) ablation. The Key Point feature embeddings before (a) and after (b) passing through TDN from frames $t - 2 , t - 1$ , and t are projected into 2D using t-SNE and plotted as scatter plots, with colors indicating frame indices. Cosine similarities among the 3 frames’ Key Point embeddings are also computed before (c) and after (d) TDN and visualized as heatmaps.

TABLE IX  
QUANTITATIVE RESULTS OF DYNAMICS MODELING AND REASONING ON COLLISION AND NON-COLLISION FRAMES FOR $\mathrm { D Y G ^ { 2 } T }$
<table><tr><td>Frames</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td></tr><tr><td>Collision Frame</td><td>18.410</td><td>0.927</td><td>0.109</td></tr><tr><td>Rest Frame</td><td>17.687</td><td>0.916</td><td>0.128</td></tr></table>

TABLE X

ABLATION STUDY OF THE NUMBER OF KEY POINTS $N ^ { * }$ IN THE PARTICLE GRAPH.
<table><tr><td>Variants</td><td>CD↓</td><td>EMD↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td></tr><tr><td> $N ^ { * } = 5 0$ </td><td>0.470</td><td>0.247</td><td>16.003</td><td>0.869</td><td>0.248</td></tr><tr><td> $N ^ { * } = 1 0 0$ </td><td>0.055</td><td>0.029</td><td>17.904</td><td>0.919</td><td>0.122</td></tr><tr><td> $N ^ { * } = 1 5 0$ </td><td>0.098</td><td>0.050</td><td>17.509</td><td>0.909</td><td>0.151</td></tr></table>

4) Dynamics Modeling Based on Particle Graph Transformer: Table X presents the impact of the number of Key Points $N ^ { * }$ on dynamics modeling. A small Graph $( N ^ { * } = 5 0 )$ including sparse Key Points, which fail to provide sufficient support for dynamics modeling. Moreover, a large Graph $( N ^ { * } ~ = ~ 1 5 0 )$ introduces excessive redundancy, which can hinder effective modeling of dynamics. Table XI investigates the effect of different edge sparsity levels in the Particle Graph. A sparse Particle Graph $( k _ { G } = 2 )$ lacks sufficient alternative paths for interaction modeling, while a dense graph $( k _ { G } = 1 0 )$

TABLE XI  
ABLATION STUDY OF THE PRESENCE OF EDGES BETWEEN KEY POINTS AND THEIR $\mathrm { T O P } { \cdot } k _ { G }$ NEAREST NEIGHBORS.
<table><tr><td>Variants</td><td>CD↓</td><td>EMD↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td></tr><tr><td> $k _ { G } = 2$ </td><td>0.564</td><td>0.255</td><td>18.436</td><td>0.910</td><td>0.183</td></tr><tr><td> $k _ { G } = 5$ </td><td>0.055</td><td>0.029</td><td>17.904</td><td>0.919</td><td>0.122</td></tr><tr><td> $k _ { G } = 1 0$ </td><td>0.186</td><td>0.107</td><td>16.398</td><td>0.881</td><td>0.193</td></tr></table>

TABLE XII

ABLATION STUDY OF THE SIZE OF THE TRAINING TEMPORAL WINDOW ϵ.
<table><tr><td>€</td><td> $\mathrm { C D \downarrow }$ </td><td>EMD↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td></tr><tr><td> $\epsilon = 2$ </td><td>0.106</td><td>0.056</td><td>17.047</td><td>0.897</td><td>0.161</td></tr><tr><td> $\epsilon = 5$ </td><td>0.055</td><td>0.029</td><td>17.904</td><td>0.919</td><td>0.122</td></tr><tr><td> $\epsilon = 8$ </td><td>4.844</td><td>2.406</td><td>19.062</td><td>0.933</td><td>0.143</td></tr></table>

increases the difficulty of identifying optimal interaction. Notably, the $k _ { G } = 2$ variant exhibits poor trajectory prediction, causing the rendering images to be filled with a large number of invalid pixels and resulting in abnormally high PSNR; the LPIPS and SSIM still demonstrate the superiority of our method. Accordingly, we set $N ^ { * } ~ = ~ 1 0 0 , ~ k _ { G } ~ = ~ 5$ for the Spring-Gaus synthetic subset as the best practice, while using $N ^ { * } ~ = ~ 1 2 0$ $k _ { G } ~ = ~ 7$ for the Spring-Gaus real-world and Unity3D-H datasets due to their differing data distributions.

5) Training Strategy: We analyze the impact of the prediction window ϵ in Table XII. A short horizon $( \epsilon ~ = ~ 2 )$ provides limited temporal context and yields unstable longrange dynamics. Extending to $\epsilon = 5$ achieves the best overall performance, as supervising multiple future frames encourages $\mathrm { \Delta D y G ^ { 2 } T }$ to capture coherent interaction modeling while keeping gradients stable. Interestingly, $\epsilon = 8$ leads to much worse CD/EMD but the highest PSNR/SSIM. Upon inspection, longhorizon predictions cause the object to drift outside the camera frustum, resulting in near-empty frames. Pixel-based metrics are thus inflated by large background regions, whereas 3Dbased metrics correctly reveal the physical errors. The optimal setting $\epsilon = 5$ is shared across all datasets.

TABLE XIII  
INFERENCE-PHASE COMPUTATIONAL RESOURCE STATISTICS.
<table><tr><td>Methods</td><td>Time(s)↓</td><td>FPS↑</td><td>Params(M)↓</td><td>FLOPs(M)↓</td></tr><tr><td>GS-Dynamics</td><td>6.852</td><td>4.378</td><td>2.900</td><td>1707.674</td></tr><tr><td>w/o spatial</td><td>2.823</td><td>10.627</td><td>2.902</td><td>2884.713</td></tr><tr><td>w/o temporal</td><td>2.887</td><td>10.391</td><td>4.878</td><td>4410.182</td></tr><tr><td>w/o transformer</td><td>2.862</td><td>10.482</td><td>4.647</td><td>4107.910</td></tr><tr><td> $\mathrm { \Delta D y G ^ { 2 } T }$  (Ours)</td><td>3.074</td><td>9.761</td><td>4.650</td><td>4393.837</td></tr></table>

6) Efficiency Analysis: To analyze computational cost, we report inference time (Time), throughput (FPS), parameter count (Params), and FLOPs of $\mathrm { { D y G ^ { 2 } T } }$ and its variants on a NVIDIA RTX 4090 (24GB) GPU. As shown in Table XIII, although $\mathrm { \Delta D y G ^ { 2 } T }$ has higher Params and FLOPs than GS-Dynamics, it achieves significantly better runtime efficiency. This is because $\mathrm { \Delta D y G ^ { 2 } T }$ performs particle representation enrichment only on sparse Key Points, using highly parallelizable and GPU-friendly operations such as batched MLPs and attention [57, 58]. Ablation results further identify the source of overhead. Removing the Spatial Semantic Completion module (w/o spatial) leads to substantial reductions in both Params and FLOPs, indicating that most computational cost arises from raw particle aggregation and Key Point directed relation modeling. In contrast, removing the Dynamic Temporal Aggregation (w/o temporal) and Particle Graph Transformer (w/o transformer) results in minor reductions, suggesting a relatively small contribution to overall complexity.

## V. LIMITATION

Despite $\mathrm { \Delta D y G ^ { 2 } T }$ performing effectively in modeling foreground object dynamics and predicting dense Gaussian trajectories from sparse-view observations under fixed point-light illumination, several issues remain.

$\mathrm { \Delta D y G ^ { 2 } T }$ adopts the LBS-based densification that propagates Key Point motions to raw particles through smooth weighted interpolation, which may attenuate deformations at spatial scales smaller than the Key Point spacing, such as localized wrinkles or fine surface details.

$\mathrm { \Delta D y G ^ { 2 } T }$ reuses SH-based appearance descriptors across future frames without explicitly modeling time-varying or view-dependent appearance changes, which may reduce visual fidelity under strong specular reflections or dynamic illumination.

## VI. CONCLUSION

This paper presents $\mathrm { \Delta D y G ^ { 2 } T , }$ a dynamics modeling framework that combines spatiotemporally completed particle representations with multi-scale interaction modeling. $\mathrm { \Delta D y G ^ { 2 } T }$ enriches Key Point features with local details from neighboring raw particles and relative offsets among Key Points, while a Temporal Disentangling Net and Temporal Attention capture frame-wise temporal evolution cues. A Particle Graph Transformer further captures long-range interactions via multi-scale interaction modeling, enabling accurate dynamics prediction. Extensive experiments on synthetic and real-world datasets demonstrate precise trajectory prediction and strong generalization, with ablations validating each component. Future work includes augmenting the current LBS-based densification with residual deformation modeling or particle-wise refinement to better capture localized high-frequency surface variations, and extending the framework with time-varying or view-conditioned appearance residual SH modeling to enhance fidelity under dynamic lighting or specular highlights.

## REFERENCES

[1] Y. Zhang, T. Liang, Z. Chen, Y. Ze, and H. Xu, “Catch it! learning to catch in flight with mobile dexterous hands,” in 2025 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2025, pp. 14 385–14 391.

[2] L. Zhong, H.-X. Yu, J. Wu, and Y. Li, “Reconstruction and simulation of elastic objects with spring-mass 3d gaussians,” in Proceedings of the European Conference on Computer Vision. Springer, 2024, pp. 407–423.

[3] J. Cai, Y. Yang, W. Yuan, Y. He, Z. Dong, L. Bo, H. Cheng, and Q. Chen, “Gic: Gaussian-informed continuum for physical property identification and simulation,” in Proceedings of the 38th International Conference on Neural Information Processing Systems, vol. 37, 2024, pp. 75 035–75 063.

[4] M. Zhang, K. Zhang, and Y. Li, “Dynamic 3d gaussian tracking for graph-based neural dynamics modeling,” in Proceedings of the Conference on Robot Learning. PMLR, 2025, pp. 1851–1862.

[5] B. Kerbl, G. Kopanas, T. Leimkuhler, and G. Drettakis,¨ “3d gaussian splatting for real-time radiance field rendering,” ACM Transactions on Graphics, vol. 42, no. 4, pp. 1–14, 2023.

[6] C. Li, L. Zhou, H. Jiang, Z. Zhang, X. Xiang, H. Sun, Q. Luan, H. Bao, and G. Zhang, “Hybrid-mvs: Robust multi-view reconstruction with hybrid optimization of visual and depth cues,” IEEE Transactions on Circuits and Systems for Video Technology, vol. 33, no. 12, pp. 7630–7644, 2023.

[7] C. R. Qi, L. Yi, H. Su, and L. J. Guibas, “Pointnet++: Deep hierarchical feature learning on point sets in a metric space,” in Proceedings of the 31th International Conference on Neural Information Processing Systems, 2017.

[8] X. Li, Y.-L. Qiao, P. Y. Chen, K. M. Jatavallabhula, M. Lin, C. Jiang, and C. Gan, “Pac-nerf: Physics augmented continuum neural radiance fields for geometryagnostic system identification,” in Proceedings of the International Conference on Learning Representations, 2023.

[9] Z. Liu, W. Ye, Y. Luximon, P. Wan, and D. Zhang, “Unleashing the potential of multi-modal foundation models and video diffusion for 4d dynamic physical scene simulation,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 11 016– 11 025.

[10] T. Xie, Z. Zong, Y. Qiu, X. Li, Y. Feng, Y. Yang, and C. Jiang, “Physgaussian: Physics-integrated 3d gaussians for generative dynamics,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 4389–4398.

[11] H. Mittal, P. Zhuang, H.-Y. Lee, and S. Tulsiani, “Uniphy: Learning a unified constitutive model for inverse physics simulation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025, pp. 16 208–16 218.

[12] B. Zhao, Y. Li, Z. Sun, L. Zeng, Y. Shen, R. Ma, Y. Zhang, H. Bao, and Z. Cui, “Gaussianprediction: Dynamic 3d gaussian prediction for motion extrapolation and free view synthesis,” in ACM SIGGRAPH 2024 Conference Papers, 2024, pp. 1–12.

[13] J. Fu, Q. Gao, C. Wen, Y. Wu, S. Ma, J. Zhang, and J. Zhang, “Recon-gs: Continuum-preserved guassian streaming for fast and compact reconstruction of dynamic scenes,” in Proceedings of the 39th International Conference on Neural Information Processing Systems, 2025.

[14] J. Sun, H. Jiao, G. Li, Z. Zhang, L. Zhao, and W. Xing, “3dgstream: On-the-fly training of 3d gaussians for efficient streaming of photo-realistic free-viewpoint videos,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 20 675– 20 685.

[15] Z. Yang, X. Gao, W. Zhou, S. Jiao, Y. Zhang, and X. Jin, “Deformable 3d gaussians for high-fidelity monocular dynamic scene reconstruction,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 20 331–20 341.

[16] Y. Sun, D. Zhu, H. Du, and Z. Tian, “Mhnf: Multi-hop heterogeneous neighborhood information fusion graph representation learning,” IEEE Transactions on Knowledge and Data Engineering, vol. 35, no. 7, pp. 7192– 7205, 2022.

[17] Y. Sun, D. Zhu, Y. Wang, Y. Fu, and Z. Tian, “Gtc: Gnntransformer co-contrastive learning for self-supervised heterogeneous graph representation,” Neural Networks, vol. 181, p. 106645, 2025.

[18] T. Chen, Y. Mu, Z. Liang, Z. Chen, S. Peng, Q. Chen, M. Xu, R. Hu, H. Zhang, X. Li et al., “G3flow: Generative 3d semantic flow for pose-aware and generalizable object manipulation,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 1735–1744.

[19] A. Pumarola, E. Corona, G. Pons-Moll, and F. Moreno-Noguer, “D-nerf: Neural radiance fields for dynamic scenes,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021, pp. 10 318–10 327.

[20] S. Li, Z. Xia, and Q. Zhao, “Representing boundaryambiguous scene online with scale-encoded cascaded grids and radiance field deblurring,” IEEE Transactions on Circuits and Systems for Video Technology, vol. 34, no. 4, pp. 2026–2040, 2023.

[21] Y. Liang, T. Xu, and Y. Kikuchi, “Himor: Monocular deformable gaussian reconstruction with hierarchical motion representation,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 886–895.

[22] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and R. Ng, “Nerf: Representing scenes

as neural radiance fields for view synthesis,” Communications of the ACM, vol. 65, no. 1, pp. 99–106, 2021.

[23] P. Wang, L. Liu, Y. Liu, C. Theobalt, T. Komura, and W. Wang, “Neus: learning neural implicit surfaces by volume rendering for multi-view reconstruction,” in Proceedings of the 35th International Conference on Neural Information Processing Systems, 2021, pp. 27 171– 27 183.

[24] T. Li, M. Slavcheva, M. Zollhoefer, S. Green, C. Lassner, C. Kim, T. Schmidt, S. Lovegrove, M. Goesele, R. Newcombe et al., “Neural 3d video synthesis from multiview video,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 5521–5531.

[25] Y. Wang, Q. Han, M. Habermann, K. Daniilidis, C. Theobalt, and L. Liu, “Neus2: Fast learning of neural implicit surfaces for multi-view reconstruction,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 3295–3306.

[26] K. Park, U. Sinha, J. T. Barron, S. Bouaziz, D. B. Goldman, S. M. Seitz, and R. Martin-Brualla, “Nerfies: Deformable neural radiance fields,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2021, pp. 5865–5874.

[27] H. Gao, R. Li, S. Tulsiani, B. Russell, and A. Kanazawa, “Monocular dynamic view synthesis: a reality check,” in Proceedings of the 36th International Conference on Neural Information Processing Systems, 2022, pp. 33 768–33 780.

[28] J. Ding, Y. He, B. Yuan, Z. Yuan, P. Zhou, J. Yu, and X. Lou, “Ray reordering for hardware-accelerated neural volume rendering,” IEEE Transactions on Circuits and Systems for Video Technology, vol. 34, no. 11, pp. 11 413–11 422, 2024.

[29] J. Luiten, G. Kopanas, B. Leibe, and D. Ramanan, “Dynamic 3d gaussians: Tracking by persistent dynamic view synthesis,” in Proceedings of the 2024 International Conference on 3D Vision. IEEE, 2024, pp. 800–809.

[30] W. Li, X. Pan, J. Lin, P. Lu, D. Feng, and W. Shi, “Frpgs: Fast, robust, and photorealistic monocular dynamic scene reconstruction with deformable 3d gaussians,” IEEE Transactions on Circuits and Systems for Video Technology, 2025.

[31] Z. Guo, W. Zhou, L. Li, M. Wang, and H. Li, “Motionaware 3d gaussian splatting for efficient dynamic scene reconstruction,” IEEE Transactions on Circuits and Systems for Video Technology, 2024.

[32] J. Yang, W. Su, S. Zhang, Y. Han, J. Suo, and Q. Zhang, “Scale-gs: Efficient scalable gaussian splatting via redundancy-filtering training on streaming content,” arXiv preprint arXiv:2508.21444, 2025.

[33] J. Lyu, J. Dong, and Y.-X. Wang, “Scas4d: Structural cascaded optimization for boosting persistent 4d novel view synthesis,” arXiv preprint arXiv:2510.06694, 2025.

[34] M. Raissi, P. Perdikaris, and G. E. Karniadakis, “Physicsinformed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations,” Journal of Compu-

tational physics, vol. 378, pp. 686–707, 2019.

[35] W. Im, G. Cha, S. Lee, J. Lee, J. Seon, D. Wee, and S.-E. Yoon, “Regularizing dynamic radiance fields with kinematic fields,” in Proceedings of the European Conference on Computer Vision, 2024, pp. 312–328.

[36] M. Raissi, A. Yazdani, and G. E. Karniadakis, “Hidden fluid mechanics: Learning velocity and pressure fields from flow visualizations,” Science, vol. 367, no. 6481, pp. 1026–1030, 2020.

[37] Y. Wang, S. Tang, and M. Chu, “Physics-informed learning of characteristic trajectories for smoke reconstruction,” in ACM SIGGRAPH 2024 Conference Papers, 2024, pp. 1–11.

[38] J. Li, Z. Song, S. Zhou, and B. Yang, “Freegave: 3d physics learning from dynamic videos by gaussian velocity,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025, pp. 12 433–12 443.

[39] B. Hu, Y. Li, R. Xie, B. Xu, H. Dong, J. Yao, and G. H. Lee, “Learnable infinite taylor gaussian for dynamic view rendering,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 26 844– 26 854.

[40] Y. Li, J. Wu, R. Tedrake, J. B. Tenenbaum, and A. Torralba, “Learning particle dynamics for manipulating rigid bodies, deformable objects, and fluids,” in Proceedings of the International Conference on Learning Representations, 2019.

[41] X. Lin, Y. Wang, Z. Huang, and D. Held, “Learning visible connectivity dynamics for cloth smoothing,” in Proceedings of the Conference on Robot Learning. PMLR, 2022, pp. 256–266.

[42] R. W. Sumner, J. Schmid, and M. Pauly, “Embedded deformation for shape manipulation,” ACM Transactions on Graphics, vol. 26, no. 3, p. 80, 2007.

[43] Y.-H. Huang, Y.-T. Sun, Z. Yang, X. Lyu, Y.-P. Cao, and X. Qi, “Sc-gs: Sparse-controlled gaussian splatting for editable dynamic scenes,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 4220–4230.

[44] Y. Eldar, M. Lindenbaum, M. Porat, and Y. Y. Zeevi, “The farthest point strategy for progressive image sampling,” IEEE transactions on image processing, vol. 6, no. 9, pp. 1305–1315, 1997.

[45] C. R. Qi, H. Su, K. Mo, and L. J. Guibas, “Pointnet: Deep learning on point sets for 3d classification and segmentation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2017, pp. 652–660.

[46] H. Zhao, L. Jiang, J. Jia, P. H. Torr, and V. Koltun, “Point transformer,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2021, pp. 16 259–16 268.

[47] M. Yang, H. Ren, and S. Velipasalar, “Trans<sup>2</sup>-cbct: A dual-transformer framework for sparse-view cbct reconstruction,” arXiv preprint arXiv:2506.17425, 2025.

[48] C. Zhang, K. Zhang, C. Zhang, T. X. Pham, C. D. Yoo, and I. S. Kweon, “How does simsiam avoid collapse

without negative samples? a unified understanding with self-supervised contrastive learning,” in Proceedings of the International Conference on Learning Representations, 2022.

[49] Y. Tian, X. Chen, and S. Ganguli, “Understanding selfsupervised learning dynamics without contrastive pairs,” in Proceedings of the International Conference on Machine Learning. PMLR, 2021, pp. 10 268–10 278.

[50] Y. Shi, Z. Huang, S. Feng, H. Zhong, W. Wang, and Y. Sun, “Masked label prediction: Unified message passing model for semi-supervised classification,” in Proceedings of the Thirtieth International Joint Conference on Artificial Intelligence, 2021, pp. 1548–1554.

[51] S. Wang, Z. Mao, C. Zeng, H. Gong, S. Li, and B. Chen, “A new method of virtual reality based on unity3d,” in 2010 18th International Conference on Geoinformatics. IEEE, 2010, pp. 1–5.

[52] H. Xue, A. Torralba, J. Tenenbaum, D. Yamins, Y. Li, and H.-Y. Tung, “3d-intphys: towards more generalized 3d-grounded visual intuitive physics under challenging scenes,” in Proceedings of the 37th International Conference on Neural Information Processing Systems, 2023, pp. 7116–7136.

[53] S. Liu, Z. Zeng, T. Ren, F. Li, H. Zhang, J. Yang, Q. Jiang, C. Li, J. Yang, H. Su et al., “Grounding dino: Marrying dino with grounded pre-training for openset object detection,” in Proceedings of the European Conference on Computer Vision. Springer, 2024, pp. 38–55.

[54] A. Kirillov, E. Mintun, N. Ravi, H. Mao, C. Rolland, L. Gustafson, T. Xiao, S. Whitehead, A. C. Berg, W.- Y. Lo et al., “Segment anything,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 4015–4026.

[55] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, “Image quality assessment: from error visibility to structural similarity,” IEEE transactions on image processing, vol. 13, no. 4, pp. 600–612, 2004.

[56] R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang, “The unreasonable effectiveness of deep features as a perceptual metric,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2018, pp. 586–595.

[57] Z. Wang, Y. Wang, C. Yuan, R. Gu, and Y. Huang, “Empirical analysis of performance bottlenecks in graph neural network training and inference with gpus,” Neurocomputing, vol. 446, pp. 165–191, 2021.

[58] A. Segura, J. M. Arnau, and A. Gonzalez, “Irregular accesses reorder unit: improving gpgpu memory coalescing for graph-based workloads: A. segura,” The Journal of Supercomputing, vol. 79, no. 1, pp. 762–787, 2023.

![](images/5cbd046da522149c8f157b257b39313be70352d12dbb73391cff88b90d64b64a.jpg)

Yansong Wang received the B.S. degree from Harbin Institute of Technology, Weihai, China, in 2022. He is currently pursuing the Ph.D. degree in computer science and technology at the Harbin Institute of Technology, China. His research interests include computer vision, physical scene understanding, and video understanding.

![](images/b1e66766a204468b52cd3bb0e951986e87d8aa34c1c4ed26bf1d3a8bc6a5c570.jpg)  
IEEE MIPR 2018.

Weigang Zhang received the B.S. degree in Computer Science and Technology in 2003, the M.S. and Ph.D. degree in Computer Applied Technology in 2005 and 2016, respectively, from Harbin Institute of Technology, Harbin, China. He is currently a Full Professor with the School of Computer Science and Technology, Harbin Institute of Technology, Weihai, China. His research interests include multimedia analysis, computer vision, and pattern recognition. He has published more than 80 academic papers and is the recipient of the Best Student Paper Award at

![](images/f034643a944d0dd4b29f4ff29072ef53a2cf9b90f946913915ce760973c65bfb.jpg)

Zhaobo Qi received the B.S. degree from Harbin Institute of Technology, Weihai, China, in 2016, and the Ph.D. degree from the School of Computer Science and Technology, University of Chinese Academy of Sciences, Beijing, China, in 2022. He is currently an associate professor at Harbin Institute of Technology, Weihai, China. His current research interests include video understanding, knowledge engineering, and computer vision.

![](images/a27826a561acb0b6857ab84ae7bcacf22786c1d4f11ae19c8ae356072e557a38.jpg)

Xinyan Liu received the Ph.D. degree from the University of Chinese Academy of Sciences, Beijing, China, in 2025. He is currently an assistant professor with the School of Computer Science and Technology, Harbin Institute of Technology, Weihai, China. His research interests include crowd counting and localization, video language tracking.

![](images/a2683ed378df0a4d3acddb630b93b059e52c8bb63675552e16f5b1c675ab5d9f.jpg)

Beichen Zhang received the B.S. degree and the Ph.D. degree from the University of Chinese Academy of Sciences, China, in 2018 and 2024, respectively. He is currently an associate professor at Harbin Institute of Technology, Weihai. His research interests include active learning, federated learning, and multimedia analysis.

![](images/32bf269e274df3f875c4f7568a0b770f5ad008c122a3342d0d5c369259951d69.jpg)

Qingming Huang received the B.S. degree in computer science and Ph.D. degree in computer engineering from the Harbin Institute of Technology, Harbin, China, in 1988 and 1994, respectively. He is currently a Chair Professor with the School of Computer Science and Technology, University of Chinese Academy of Sciences. He has authored over 400 academic papers in international journals, such as IEEE Transactions on Pattern Analysis and Machine Intelligence, IEEE Transactions on Image Processing, IEEE Transactions on Multimedia, IEEE

Transactions on Circuits and Systems for Video Technology, and top level international conferences, including the ACM Multimedia, ICCV, CVPR, ECCV, VLDB, and IJCAI. His research interests include multimedia computing, image/video processing, pattern recognition, and computer vision.

![](images/fdf3b9206a108a7399460cafa2cffb709d39ebe21873006aaa6720dfaacf4f89.jpg)  
knowledge extraction.

Shuhui Wang received the B.S. degree in electronics engineering from Tsinghua University, Beijing, China, in 2006, and the Ph.D. degree from the Institute of Computing Technology, Chinese Academy of Sciences, Beijing, China, in 2012. He is currently a Full Professor with the Institute of Computing Technology, Chinese Academy of Sciences. He is also with the Key Laboratory of Intelligent Information Processing, Chinese Academy of Sciences. His research interests include image/video understanding/retrieval, cross-media analysis, and visual-textual