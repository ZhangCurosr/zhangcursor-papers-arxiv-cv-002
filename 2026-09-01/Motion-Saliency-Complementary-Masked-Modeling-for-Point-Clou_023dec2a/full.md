# Motion-Saliency Complementary Masked Modeling for Point Cloud Video Understanding

Wei Wang<sup>a,b,1</sup>, Yiding Sun<sup>c,e,1</sup>, Yuyan Wang<sup>c,1</sup>, Zhuoyue Zhang<sup>c</sup>, Zhengqiao Li<sup>d</sup>, Dongfu Yin<sup>e,∗</sup>, Chen Li<sup>a,∗</sup>

<sup>a</sup>School ofComputer Science and Technology, Xi’an Jiaotong University, Xi’an, 710049, China <sup>b</sup>State Key Laboratory of Intelligent Geotechnics and Tunnelling (FSDI), Xi’an, 710043, China <sup>c</sup>School ofSoftware Engineering, Xi’an Jiaotong University, Xi’an, 710049, China <sup>d</sup>School ofInformation and Communication Engineering, Xi’an Jiaotong University, Xi’an, 710049, China <sup>e</sup>Guangdong Laboratory of Artificial Intelligence and Digital Economy (SZ), Shenzhen, 518107, China

## Abstract

Point cloud video representation learning is crucial for 3D dynamic scene understanding. In this paper, we propose MoSaiC, a novel Motion-Saliency Complementary masked modeling framework for self-supervised point cloud video representation learning. MoSaiC couples three components, Curriculum Motion-Saliency Masking (CMSM) which guides the masking process toward motion-salient tokens under a curriculum schedule, Normal-Flow Motion (NFM) modeling, which supervises the local rigid rotation of each token in the Lie algebra so(3) as an explicit geometric motion target, and Cross-view Token Consistency Prediction (CTCP), which enforces consistency between two complementary masked views at the token level. Together, these components allow MoSaiC to efectively capture both appearance and motion dynamics. Extensive experiments on multiple downstream tasks, including action recognition, temporal action segmentation, and point-level semantic segmentation, demonstrate the efectiveness of our approach.

Keywords: Point Cloud Video, Self-Supervised Learning, Masked Modeling, Motion Saliency

## 1. Introduction

Dynamic 3D perception seeks to understand not only the geometry of a scene, but also how that geometry evolves over time. Point cloud videos, sequences of 3D point sets captured by depth cameras or LiDAR, provide a direct and illumination-robust representation for this purpose. They have enabled progress in action recognition, temporal action segmentation, and 4D semantic segmentation [1, 2, 3], with applications ranging from gesture interfaces to embodied perception and human–object interaction. Their success, however, has largely relied on supervised learning. Annotating a dynamic point cloud requires labels to be assigned across frames, and dense tasks further demand temporal boundaries or point-level semantics. Such annotation becomes prohibitively expensive as sequence length and scene complexity increase.

Self-supervised learning (SSL) ofers a way to reduce this dependence by learning transferable spatio-temporal representations from unlabeled point cloud videos. The central problem is not simply how to reconstruct missing 3D content, but how to design a pretext task that forces the encoder to discover the sparse, localized geometric changes that define motion. Unlike RGB videos, point cloud sequences have no persistent pixel grid, their sampling density varies across frames, and point-wise correspondences are generally unavailable. A useful objective must therefore separate informative motion from static geometry without assuming reliable tracking, while retaining the token-level structure required by dense downstream tasks.

Existing point cloud video SSL methods can be grouped into three broad families. Masked reconstruction methods hide point tubes and recover their geometry or temporal statistics, as exemplified by MaST-Pre [4]. Contrastive and distillation methods learn invariance between augmented, complete, or partial observations, including PointCMP [5], C2P [6], and Query-CP [7]. More recent hybrid methods combine reconstruction with motion or semantic objectives, M2PSC [8], for example, predicts point trajectories while contrasting masked and visible tubes. These approaches establish the value of unlabeled 4D data, but they still treat the most consequential design choices only partially.

We identify three coupled limitations, illustrated in Figure 1. First, existing masking distributions are random or otherwise independent of local geometric change, so scarce reconstruction capacity is spent equally on static and dynamic tokens. Second, motion is represented by coarse statistics, implicit feature objectives, or correspondence-dependent trajectories. A scalar cardinality change cannot distinguish density-preserving rotations, while implicit motion features do not specify the axis and angle of local reorientation. Third, single-view reconstruction and tube- or clip-level contrast provide no direct consistency constraint for the same token under complementary par-

![](images/2d17e648e1b1fcb722207db9a87c702280d8c19c98731794df936596ed6ea243.jpg)  
Figure 1: Motivation of MoSaiC. Existing point cloud video SSL methods sufer from motionagnostic masking, coarse or implicit motion modeling, and single-view reconstruction. MoSaiC addresses them with Curriculum Motion-Saliency Masking (CMSM), Normal-Flow Motion (NFM) modeling, and Cross-view Token Consistency Prediction (CTCP).

tial observations. Consequently, existing objectives do not jointly determine where motion should be learned, what geometric motion should be encoded, and how local representations should remain stable across views.

These limitations directly afect downstream performance. Motion-agnostic masking makes a model proficient at interpolating redundant background geometry while undersupervising the few tokens that distinguish similar actions. Coarse motion targets merge physically diferent transformations, weakening the cues needed to separate rotation-heavy actions and to localize transitions between motion states. Global or single-view objectives can still produce useful clip descriptors, but they leave individual token features insuficiently constrained for temporal and point-level prediction. Addressing any one limitation in isolation is also inadequate, focusing on dynamic tokens before learning basic geometry makes the reconstruction task prematurely difficult, whereas a precise motion target has limited value if it is rarely applied to informative regions.

To address this problem, we propose MoSaiC, a Motion-Saliency Complementary masked modeling framework that couples three components. Curriculum Motion-Saliency Masking (CMSM) estimates token saliency from temporal SHOT diferences and gradually changes the masking distribution from uniform to motion-focused, so the encoder learns a geometric prior before being challenged with highly dynamic regions. Normal-Flow Motion (NFM) supplies an explicit, correspondence-free motion target by regressing the relative rotation between adjacent local PCA reference frames in the Lie algebra so(3), while a Temporal Descriptor Diference (TDD) target preserves complementary cues from deformation and occlusion. Cross-view Token Consistency Prediction (CTCP) constructs complementary masked views and aligns the decoded feature of a hidden token with its encoded feature when that token is visible in the other view. In this way, CMSM selects where to learn, NFM and TDD define what motion to capture, and CTCP regularizes how the resulting token representation behaves across partial observations.

Our experiments also reveal several results that are not obvious from standard masked-modeling practice. Masking motion-salient tokens from the beginning is inferior to introducing them through a curriculum, showing that harder masking is beneficial only after a basic geometric representation has formed. A mask ratio of 0 6 outperforms the commonly used higher ratios because excessively masking one view makes its complementary view nearly trivial. More importantly, although motion supervision might appear irrelevant or even detrimental to static point-level labeling, MoSaiC improves HOI4D semantic segmentation. Its largest action-segmentation gains occur for rotation-dominated classes and at the strictest temporal overlap threshold, suggesting that explicit local orientation modeling preserves fine surface geometry while sharpening motion boundaries.

The main contributions of this paper are summarized as follows,

• We formulate motion-aware point cloud video SSL around three coupled requirements, motion-targeted masking, explicit geometric motion supervision, and token-level cross-view consistency, exposing why treating these requirements separately leaves important spatio-temporal cues underused.

• We develop MoSaiC, which integrates CMSM, correspondence-free NFM and complementary TDD targets, and CTCP in a unified masked-modeling framework. The three components provide distinct supervision over target selection, local motion geometry, and cross-view token prediction.

• Extensive experiments on MSR-Action3D, SHREC’17, NvGesture, and HOI4D demonstrate transferable and label-eficient gains in action recognition, temporal action segmentation, and semantic segmentation. Ablations explain the counterintuitive efects of curriculum masking, moderate mask ratios, and motion supervision on static semantic prediction, while downstream inference retains only the backbone and incurs no additional cost.

## 2. Related Work

## 2.1. Point Cloud Video Modeling

Supervised point cloud video backbones extend static point networks [9, 10] to the temporal domain. MeteorNet [11] aggregates spatio-temporal neighbors, whereas PST-Net [12] decouples spatial and temporal convolutions hierarchically. P4Transformer [1] embeds local 4D structures through point 4D convolution followed by self-attention, avoiding explicit point tracking, and PST-Transformer [2] further searches for related points across a sequence with decoupled spatio-temporal encoding. Kinet [13] models feature-level dynamics by fitting space-time surfaces, and PPTr [14] uses primitive planes to capture long-range context [15, 16, 17, 18, 19, 20, 21]. These architectures establish efective downstream encoders, but they are trained with costly dense supervision and do not themselves prescribe how motion-aware, label-free objectives should be constructed. MoSaiC retains a P4Transformer-style tube embedding while learning its representation from unlabeled videos through motion-targeted pre-training.

## 2.2. Masked and Contrastive Self-Supervised Learning

Masked autoencoders (MAE) [22] reconstruct randomly masked image patches with an asymmetric encoder-decoder, and VideoMAE [23] transfers this principle to videos through high-ratio tube masking. For static point clouds, Point-BERT [24], Point-MAE [25], and Point-M2AE [26] adapt masked modeling through discrete tokenization, coordinate reconstruction, and hierarchical multi-scale encoding, respectively [27, 28, 29, 30, 31]. These approaches show that reconstruction can provide transferable geometric priors, but random masking and appearance-centric targets do not distinguish a temporally informative change from a static, trivially reconstructible region. This motivates CMSM and NFM, which direct masking toward motion-salient tokens and explicitly supervise their local geometric motion [32].

Beyond reconstruction, contrastive methods such as PointContrast [33] contrast features of corresponding points across views, while non-contrastive siamese frameworks such as BYOL [34] and SimSiam [35] achieve view consistency without negative samples. Their targets are generally global descriptors, correspondence-dependent point features, or 2D video features, and therefore do not provide a token-level crossview constraint for partially observed 4D point sequences. CTCP builds on the asymmetric predictor and stop-gradient design of siamese learning, but applies it only when a token is masked in one complementary view and visible in the other.

## 2.3. Self-Supervised Learning on Point Cloud Videos

A growing body of work studies SSL specifically for point cloud videos. MaST-Pre [4] performs spatio-temporal point-tube masking and jointly reconstructs masked tubes and predicts temporal cardinality diference (TCD). PointCMP [5] unifies contrastive learning and mask prediction, C2P [6] distills representations from complete sequences to partial ones, and Query-CP [7] combines learnable query contrast with decoupled spatio-temporal prediction [36, 37]. M2PSC [8] is the most closely related prior method, coupling motion trajectory prediction with semantic contrast between visible and masked tubes alongside appearance reconstruction. Its motion target is a per-point trajectory under an implicit correspondence rather than the correspondencefree so(3) rotation of NFM, and its contrast operates between masked and visible tubes rather than through CTCP’s complementary two-view predictor.

These methods leave the three requirements of Section 1 only partially addressed. Their masking distributions are independent of geometric motion, none supplies an explicit rotation target, and none imposes token-level consistency across complementary masked observations. MoSaiC addresses them jointly: CMSM selects motion-salient masking targets through a curriculum, NFM supplies correspondence-free so(3) supervision while TDD retains a complementary topology-sensitive signal, and CTCP enforces cross-view consistency at token granularity.

## 3. Method

## 3.1. Overview

Figure 2 illustrates the overall framework of MoSaiC. Given an input point cloud video, we first tokenize it into a sequence of spatio-temporal tokens. Instead of using a random masking strategy, we propose Curriculum Motion-Saliency Masking (CMSM) to generate two complementary masked views (View A and View B) based on the motion saliency of each token. The visible tokens from both views are fed into a shared asymmetric encoder-decoder architecture. The decoder reconstructs the masked tokens, guided by three types of reconstruction targets, appearance (Chamfer Distance), Normal-Flow Motion (NFM), and Temporal Descriptor Diference (TDD). Furthermore, we introduce Cross-view Token Consistency Prediction (CTCP) to enforce the consistency between the predicted representations of the two complementary views.

## 3.2. Preliminaries, Point Cloud Video Tokenization

Given a point cloud video sequence $\mathcal { P } = \{ P _ { 1 } , P _ { 2 } , . . . , P _ { T } \}$ , where $\boldsymbol { P } _ { t } \in \mathbb { R } ^ { N \times 3 }$ is the point cloud frame at time step t, we extract spatio-temporal tokens using a tube embedding module based on point 4D convolution (P4DConv) [1], anchor points are sampled via Farthest Point Sampling and grouped with their spatial and temporal neighbors via ball query (hyperparameter values are given in Section 4). Each resulting token i carries a feature vector $f _ { i } \in \mathbb { R } ^ { C }$ , a spatio-temporal position $\pmb { p } _ { i } = ( x _ { i } , y _ { i } , z _ { i } , t _ { i } ) \in \mathbb { R } ^ { 4 }$ , and its local neighborhood points $N _ { i } .$ , which span a set of consecutive frames $\mathcal { T } _ { i }$

For every frame $t \in \mathcal { T } _ { i }$ we additionally compute a Signature of Histograms of OrienTations (SHOT) descriptor $\pmb { h } _ { i , t } \in \mathbb { R } ^ { L }$ over the neighborhood points of token i in that frame [38]. SHOT is conventionally made rotation-invariant by expressing each neighborhood in its own local reference frame, we instead express $\pmb { h } _ { i , t }$ and $\pmb { h } _ { i , t + 1 }$ in a common reference frame, the frame R estimated at time t (Section 3.5), so that their diference reflects a change of orientation instead of being cancelled by SHOT’s builtin invariance. The neighborhood points and these descriptors serve as reconstruction targets for the self-supervised objectives introduced below.

![](images/c7c8aeefa33cecc9fad9d772f2225d71b39de4fbfb8ce396336b590a09296618.jpg)  
Figure 2: Overview of the MoSaiC pre-training pipeline. CMSM produces two complementary masked views from motion-saliency scores. A shared encoder–decoder reconstructs masked tokens under appearance, NFM, and TDD losses. CTCP aligns the decoded feature of a token hidden in one view with its encoded feature when visible in the other. Only the encoder is retained for downstream tasks.

## 3.3. Curriculum Motion-Saliency Masking (CMSM)

To encourage the model to focus on dynamic regions, we propose Curriculum Motion-Saliency Masking (CMSM), whose pipeline, from per-token SHOT descriptors to the two complementary masked views, is illustrated in Figure 3. We compute the motion saliency $s _ { i }$ of each token i as the mean absolute temporal diference of its SHOT descriptors over the frames its tube spans,

$$
s _ { i } = \frac { 1 } { L ( | \mathcal { T } _ { i } | - 1 ) } \sum _ { t \in \mathcal { T } _ { i } \backslash \{ t _ { \operatorname* { m a x } } \} } \sum _ { l = 1 } ^ { L } \left| \pmb { h } _ { i , t + 1 } ^ { ( l ) } - \pmb { h } _ { i , t } ^ { ( l ) } \right| ,\tag{1}
$$

where L is the dimension of the SHOT descriptor and $t _ { \mathrm { m a x } }$ is the last frame of $\mathcal { T } _ { i } .$ Because the raw scores occupy a narrow, clip-dependent range, we standardize them within each clip, $\tilde { s } _ { i } = ( s _ { i } - \mu _ { s } ) / ( \sigma _ { s } + \epsilon )$ , before converting them into a probability distribution with a softmax of temperature τ,

$$
p _ { \mathrm { s a l } } ( i ) = \frac { \exp ( \tilde { s } _ { i } / \tau ) } { \sum _ { j } \exp ( \tilde { s } _ { j } / \tau ) } .\tag{2}
$$

To balance the focus between highly dynamic regions and the static background, we mix the saliency-based probability with a uniform distribution $p _ { \mathrm { u n i } } ( i ) = 1 / G$ , where G is the total number of tokens. The mixing weight α is controlled by a curriculum schedule,

$$
p _ { \operatorname* { m i x } } ( i ) = \alpha \cdot p _ { \mathrm { s a l } } ( i ) + ( 1 - \alpha ) \cdot p _ { \mathrm { u n i } } ( i ) .\tag{3}
$$

The curriculum keeps α at 0 for the first $E _ { \mathrm { w a r m u p } }$ epochs, so that masking is initially uniform and the encoder can acquire a generic geometric prior, and then increases it linearly to $\alpha _ { \mathrm { m a x } }$ over the remaining epochs,

$$
\alpha ( e ) = \alpha _ { \mathrm { m a x } } \cdot \frac { \operatorname* { m a x } ( 0 , e - E _ { \mathrm { w a r m u p } } ) } { E _ { \mathrm { t o t a l } } - E _ { \mathrm { w a r m u p } } } , \quad e = 1 , \ldots , E _ { \mathrm { t o t a l } } .\tag{4}
$$

Based on $p _ { \mathrm { m i x } } .$ , we generate two complementary masked views. For View $\mathbf { A } ,$ , we sample a subset of tokens ${ \mathcal { M } } _ { A }$ to be masked without replacement according to $p _ { \mathrm { m i x } }$ . For View B, we sample the masked tokens $M _ { B }$ from the unmasked tokens of View $\mathrm { ~ A ~ } ( i . e . , \mathcal { V } _ { A } =$ $\mathcal { G } \backslash \mathcal { M } _ { A } )$ according to the complementary probability $p _ { \mathrm { c o m p } } \propto ( 1 - p _ { \mathrm { m i x } } )$ . The number of masked tokens in View B is controlled by a complement overlap ratio $\gamma \in [ 0 , 1 )$ defined as the fraction of $\mathcal { N } _ { A }$ that stays visible in View B as well,

$$
| \mathcal { M } _ { B } | = \operatorname* { m a x } \big ( 1 , \lfloor ( 1 - \gamma ) | \mathcal { V } _ { A } | \rfloor \big ) , \qquad \mathcal { V } _ { A } = \mathcal { G } \setminus \mathcal { M } _ { A } .\tag{5}
$$

Consequently $| \mathcal { V } _ { A } \cap \mathcal { V } _ { B } | = \gamma | \mathcal { V } _ { A } |$ is the set of tokens observed by both views, and γ interpolates between exactly complementary visible sets $( \gamma = 0$ , where $\mathcal { V } _ { B } = \mathcal { M } _ { A } )$ and a View B that masks almost nothing $( \gamma  1 )$ . This construction is deliberately asymmetric. Because ${ \mathcal { M } } _ { A }$ is drawn preferentially from high-saliency tokens while $M _ { B }$ is drawn from $\gamma _ { A } .$ , a token masked in one view tends to remain visible in the other, which is the condition required by CTCP (Section 3.6). Since $\mathcal { M } _ { B } \subseteq \mathcal { V } _ { A }$ by construction, the number of supervised CTCP pairs is deterministically $| \mathcal { M } _ { A } | + ( 1 -$ $\gamma ) ( G - | { \cal M } _ { A } | )$ at every training step.

![](images/78ea00c25528a074875f0eb73dceb18b528b4273cdcfdcf9def3c28ed36dab49.jpg)  
Figure 3: Structure of Curriculum Motion-Saliency Masking (CMSM). Each token’s per-frame SHOT descriptors give a motion-saliency score, which a temperature softmax converts into $p _ { \mathrm { s a l } }$ . Curriculum mixing combines $p _ { \mathrm { s a l } }$ with the uniform distribution $p _ { \mathrm { u n i } }$ under the schedule weight α to obtain $p _ { \mathrm { m i x } }$ from which View $\mathbf { A }$ is masked directly and View B is masked from View A’s visible tokens through complementary sampling.

## 3.4. Asymmetric Encoder-Decoder

Following the MAE paradigm [22], a shared Transformer encoder processes only the visible tokens, $z _ { i } = \mathrm { E n c o d e r } ( f _ { i } + \mathrm { P o s E m b e d } ( p _ { i } ) )$ for $i \in \mathcal { V } ,$ and a lightweight decoder reconstructs the masked positions, $\pmb { d } _ { i } = \mathrm { D e c o d e r } ( z _ { \mathcal { V } } \cup \pmb { m } _ { \mathcal { M } }$ +PosEmbed $( \pmb { p } _ { \pmb { V } \cup \pmb { M } } ) )$ . Both views share the same weights. The encoded feature $z _ { i } \in \mathbb { R } ^ { 1 0 2 4 }$ of a visible token and the decoded feature $\pmb { d } _ { i } \in \mathbb { R } ^ { 5 1 2 }$ of the same token when masked live in diferent spaces, and the predictor of Section 3.6 bridges them.

## 3.5. Reconstruction Targets and Losses

The decoder is tasked with reconstructing both the appearance and motion information of the masked tokens.

Appearance Reconstruction. We reconstruct the local neighborhood points N of the masked tokens and measure the appearance loss using the Chamfer Distance (CD),

$$
\mathcal { L } _ { \mathrm { a p p } } = \frac { 1 } { | \mathcal { M } | } \sum _ { i \in \mathcal { M } } \mathrm { C D } ( \hat { N } _ { i } , N _ { i } ) .\tag{6}
$$

Normal-Flow Motion (NFM) Modeling. To capture the underlying motion $\mathrm { d y } .$ namics, we model the local rigid rotation of each token. For a token $i ,$ we compute the covariance matrix of its neighborhood points at each time step t and perform eigenvalue decomposition to obtain the local reference frame $\pmb { R } _ { t } \in \mathrm { S O } ( 3 )$ , whose columns are the eigenvectors ordered by decreasing eigenvalue $\lambda _ { 1 } \geq \lambda _ { 2 } \geq \lambda _ { 3 }$ . Eigenvectors are sign-ambiguous, so we fix the sign of each axis to maximize its inner product with the corresponding axis at $t - 1$ , and flip the third axis if necessary to enforce det( $\pmb { R } _ { t } ) = + 1$ This keeps $\pmb { R } _ { t }$ a continuous, right-handed function of time. We map $\Delta { R } = R _ { t } ^ { \top } R _ { t + 1 }$ to the Lie algebra so(3) to obtain the normal flow vector $\omega ,$

$$
\omega = \log _ { \mathrm { S O ( 3 ) } } ( \Delta R ) ^ { \vee } .\tag{7}
$$

Here $\log _ { \mathrm { S O } ( 3 ) } ( \cdot )$ is the matrix logarithm and $( \cdot ) ^ { \vee } : \mathfrak { s o } ( 3 )  \mathbb { R } ^ { 3 }$ extracts the corresponding axis-angle vector.

The decomposition is ill-conditioned whenever two eigenvalues coincide. We therefore require both eigengaps to be well-separated and use

$$
g _ { i } = \frac { \operatorname* { m i n } ( \lambda _ { 1 } - \lambda _ { 2 } , \ \lambda _ { 2 } - \lambda _ { 3 } ) } { \lambda _ { 1 } }\tag{8}
$$

as a conditioning score, excluding any token with $g _ { i }$ below a fixed threshold from ${ \mathcal { L } } _ { \mathrm { { n f m } } }$ Excluded tokens remain supervised by the appearance and TDD losses. We quantify the efect of this exclusion under degraded input in Section 4.7.

We regress $\boldsymbol { \omega } \in \mathbb { R } ^ { 3 }$ rather than a point-wise displacement field because it is obtained without point-to-point correspondence and is invariant to the number and ordering of points inside a tube. The model predicts $\hat { \omega } .$ , and the NFM loss is

$$
\mathcal { L } _ { \mathrm { n f m } } = \frac { 1 } { \vert \mathcal { M } \vert } \sum _ { i \in \mathcal { M } } \mathrm { S m o o t h L } 1 ( \hat { \omega } _ { i } , \omega _ { i } ) .\tag{9}
$$

We adopt SmoothL1 on $\omega$ rather than the geodesic distance on SO(3), whose gradient vanishes near a rotation of 0 and becomes singular near π.

Temporal Descriptor Diference (TDD). As a complementary motion target we predict the temporal diference of the SHOT descriptors, $\Delta \pmb { h } _ { i } = \pmb { h } _ { i , t + 1 } - \pmb { h } _ { i , t }$

$$
\mathcal { L } _ { \mathrm { t d d } } = \frac { 1 } { \vert \mathcal { M } \vert } \sum _ { i \in \mathcal { M } } \mathrm { S m o o t h L } 1 ( \Delta \hat { \pmb { h } } _ { i } , \Delta \pmb { h } _ { i } ) .\tag{10}
$$

TDD summarizes how much the local histogram changed, and is therefore sensitive to deformation and occlusion as well as density change, but it does not recover the axis and angle of the motion. NFM supplies that structure, and conversely remains undefined on tokens excluded by $g _ { i } .$ , where TDD is still informative. The total motion loss is $\mathcal { L } _ { \mathrm { m o t i o n } } = \mathcal { L } _ { \mathrm { n f m } } + \mathcal { L } _ { \mathrm { t d d } }$

## 3.6. Cross-view Token Consistency Prediction (CTCP)

To further exploit the complementary nature of View A and View B, we introduce the CTCP module. The intuition is that the reconstructed features of a masked token in

one view should be consistent with the encoded features of the same token when it is visible in the other view.

Specifically, let $\pmb { d } _ { i } ^ { A } \in \mathbb { R } ^ { C _ { d } }$ be the decoded feature of a masked token $i \in \mathcal { M } _ { A }$ in View A. If this token is visible in View B, we have its encoded feature $\boldsymbol { z } _ { i } ^ { B } \in \mathbb { R } ^ { C _ { e } }$ . The eligible set for this direction is therefore

$$
C _ { A  B } = M _ { A } \cap \mathcal { V } _ { B } ,\tag{11}
$$

which by the construction of Section 3.3 equals ${ \mathcal { M } } _ { A }$ and is never empty. A predictor network maps the decoded feature into the encoder feature space, $\hat { z } _ { i } ^ { B } = \bar { \mathbf { \eta } }$ Predictor $\mathbf { \Phi } ( \pmb { d } _ { i } ^ { A } ) \in$ $\mathbb { R } ^ { C _ { e } }$ , absorbing both the dimensionality mismatch $( C _ { d } = 5 1 2 $ versus $C _ { e } = 1 0 2 4 )$ and the representational gap between decoder and encoder. The CTCP loss averages the negative cosine similarity between the predicted and target features over the eligible set,

$$
\mathcal { L } _ { \mathrm { c t c p } } ^ { A  B } = \frac { 1 } { | C _ { A  B } | } \sum _ { i \in C _ { A  B } } ( 1 - \frac { \boldsymbol { \hat { z } } _ { i } ^ { B } \cdot \mathbf { s g } ( \boldsymbol { z } _ { i } ^ { B } ) } { \Vert \boldsymbol { \hat { z } } _ { i } ^ { B } \Vert _ { 2 } \Vert \mathbf { s g } ( \boldsymbol { z } _ { i } ^ { B } ) \Vert _ { 2 } } ) ,\tag{12}
$$

where sg(·) denotes the stop-gradient operation. The symmetric loss $\mathcal { L } _ { \mathrm { c t c p } } ^ { B \to A }$ is defined identically over $C _ { B  A } = M _ { B }$ , and the total CTCP loss is $\begin{array} { r } { \mathcal { L } _ { \mathrm { c t c p } } = \frac { 1 } { 2 } ( \mathcal { L } _ { \mathrm { c t c p } } ^ { A  B } + \mathcal { L } _ { \mathrm { c t c p } } ^ { B  A } ) } \end{array}$

Following non-contrastive siamese learning [34, 35], three design choices prevent collapse: a stop-gradient on the target branch, an explicit predictor that absorbs the encoder–decoder mismatch, and the asymmetric complementary sampling of Section 3.3, which keeps the number of CTCP pairs fixed. Table 11 confirms that removing the predictor causes a collapse-like degradation, whereas replacing the stop-gradient target with an EMA encoder is a viable but slightly weaker alternative.

We operate CTCP at the token level rather than pooling to a clip-level descriptor, because temporal action segmentation and semantic segmentation require localized discriminative structure.

## 3.7. Overall Objective

The overall self-supervised training objective of MoSaiC is a weighted sum of the appearance, motion, and CTCP losses from both views,

$$
\mathcal { L } = \frac { 1 } { 2 } \sum _ { \nu \in \{ A , B \} } ( \mathcal { L } _ { \mathrm { a p p } } ^ { \nu } + \mathcal { L } _ { \mathrm { m o t i o n } } ^ { \nu } ) + \lambda _ { \mathrm { c t c p } } \mathcal { L } _ { \mathrm { c t c p } } ,\tag{13}
$$

where $\lambda _ { \mathrm { c t c p } }$ is a hyperparameter balancing the consistency loss.

## 4. Experiments

## 4.1. Datasets and Evaluation Metrics

We evaluate MoSaiC on three downstream tasks across four datasets, action recognition on MSR-Action3D [39], SHREC’17 [40], and NvGesture [41], temporal action segmentation on HOI4D [3], and semantic segmentation on HOI4D.

MSR-Action3D contains 567 point cloud videos of 20 action categories performed by 10 subjects. We follow the standard cross-subject evaluation protocol, using subjects 1, 3, 5, 7, 9 for training and the rest for testing. The evaluation metric is Top-1 accuracy.

SHREC’17 is a dynamic hand-gesture dataset containing 2,800 depth and skeletal sequences of 14 gestures performed by 28 subjects. We use its depth sequences as point cloud videos and follow the oficial recognition protocol. The evaluation metric is Top-1 accuracy.

NvGesture is a multimodal dynamic hand-gesture dataset acquired with depth, color, and stereo-IR sensors. It contains 25 gesture categories performed by 20 subjects. We use the depth modality converted to point cloud videos and follow the oficial subject-independent split. The evaluation metric is Top-1 accuracy.

HOI4D is a large-scale 4D ego-centric dataset for human-object interaction. For temporal action segmentation, the dataset provides frame-level annotations for 19 action classes. We report Frame-wise Accuracy (Frame-Acc), Segmental Edit Distance (Edit), and Segmental F1 scores at overlapping thresholds of 10%, 25%, and 50% (F1@{10, 25, 50}), following the per-class breakdown we use in Section 4, F1@k is computed per action class and then averaged over the 19 classes. For semantic segmentation, the dataset annotates 38 indoor semantic categories in total, following the oficial 4D point cloud video segmentation setting, we evaluate on the 14 selected categories (7 object and 7 background). We report the mean Intersection over Union (mIoU).

## 4.2. Implementation Details

Network Architecture. The tube embedding module uses a spatial radius $r = 0 . 1$ a maximum of $k = 3 2$ neighbors, a spatial stride of 32, a temporal kernel size of 3, and a temporal stride of 2 (except for semantic segmentation where the temporal stride is 1). The encoder is a 5-layer Transformer with an embedding dimension of 1024, 8 heads, and an MLP dimension of 2048, giving 43.6M parameters in total including the tube embedding. The decoder is a 4-layer Transformer with an embedding dimension of 512 and is discarded after pre-training.

Pre-training. The model is pre-trained using the AdamW optimizer with a base learning rate of $1 \times 1 0 ^ { - 3 }$ and a weight decay of 0 05. We use a cosine learning rate schedule with a 10-epoch linear warmup for a total of 200 epochs. The base mask ratio is set to 0 6 on all datasets. For CMSM, the temperature τ is 1 0, the maximum curriculum weight $\alpha _ { \mathrm { m a x } }$ is 0 8, and the curriculum warmup is 20 epochs. The complement overlap ratio γ is 0 1. The CTCP loss weight $\lambda _ { \mathrm { c t c p } }$ is 0 25.

Fine-tuning. For downstream tasks, we fine-tune the pre-trained encoder along with a task-specific head. We use the AdamW optimizer with a learning rate of $5 \times 1 0 ^ { - 4 }$ and a cosine schedule. The encoder learning rate is scaled down by a factor (e.g., $2 \times 1 0 ^ { - 4 } )$ ). Label smoothing is applied for classification losses.

## 4.3. Comparison with State-of-the-Art

Action Recognition. Table 1 compares MoSaiC with representative supervised backbones and self-supervised pre-training methods. On the two hand-gesture benchmarks, MoSaiC attains the best accuracy, reaching 91.9% on SHREC’17 and 89.3% on

NvGesture. This improves over the from-scratch P4Transformer backbone by 2.8 and 3.2 points respectively, and over the strongest self-supervised competitor, C2P, by 0.7 and 0.5 points. Hand gestures are dominated by fine-grained finger articulation against a largely static hand silhouette, so these benchmarks reward precisely the property that CMSM is designed to induce, concentrating the pretext task on the small subset of tokens that actually move, rather than spending capacity on background geometry that is trivially reconstructible.

On MSR-Action3D, MoSaiC reaches 95.18%, 4.24 points above the from-scratch backbone and, notably, also ahead of the strongest masked-modeling baseline, C2P, though by a narrower margin (+0.42) than on the two hand-gesture benchmarks, we return to why this particular benchmark yields the smallest relative margin among the three action-recognition datasets in Section 5. The gap widens substantially on the dense spatio-temporal tasks reported next, where temporal reasoning is indispensable.

Table 1: Action recognition results (Top-1 accuracy, %) on MSR-Action3D, SHREC’17, and NvGesture. All self-supervised methods, including MoSaiC, use the P4Transformer backbone, whose supervised fromscratch result serves as the common no-pre-training reference.
<table><tr><td>Method MSR-Action3D</td><td>SHREC&#x27;17</td><td>NvGesture</td></tr><tr><td colspan="3">Supervised backbones (trained from scratch)</td></tr><tr><td>PSTNet [12] 91.20</td><td>88.5</td><td>85.9</td></tr><tr><td>P4Transformer [1] 90.94</td><td>89.1</td><td>86.1</td></tr><tr><td>PST-Transformer [2] 93.73</td><td>90.3</td><td>87.0</td></tr><tr><td>Kinet [13] 93.27</td><td>90.1</td><td>86.8</td></tr><tr><td colspan="3">Self-supervised pre-training (P4Transformer backbone)</td></tr><tr><td>PointCMP [5] 93.27</td><td>90.6</td><td>88.5</td></tr><tr><td>MaST-Pre [4] 94.08</td><td>90.8</td><td>88.1</td></tr><tr><td>C2P [6] 94.76</td><td>91.2</td><td>88.8</td></tr><tr><td>MoSaiC (Ours) 95.18</td><td>91.9</td><td>89.3</td></tr></table>

Temporal Action Segmentation. Because pre-training methods for point cloud videos are built on diferent backbones, Table 2 groups the HOI4D action segmentation results by backbone, and additionally instantiates MoSaiC on both backbones so that every group ends with a directly comparable pre-training result. On its default P4Transformer backbone, MoSaiC reaches 81.9% Frame-Acc, 83.6 Edit, and 86.2/83.4/76.8 on F1@{10, 25, 50}, outperforming the strongest backbone-matched baseline, M2PSC, by 6.0 Frame-Acc and 10.9 F1@50. To verify that this gain is not an artifact of a particular backbone, we also instantiate MoSaiC on PPTr, a backbone with a longer efective temporal receptive field. There, MoSaiC still improves over the strongest PPTr-based baseline, C2P, by 2.4 Frame-Acc and 3.3 F1@50, confirming that the pretext task contributes on top of a stronger encoder as well, albeit with a smaller margin than on P4Transformer. Relative to the from-scratch P4Transformer backbone, the most informative pattern is that the margin grows monotonically with the overlap threshold, MoSaiC leads by 12.4 points at F1@10 but by 18.6 points at F1@50. A gain concentrated at the strict threshold cannot be explained by simply labelling more frames correctly, it requires predicting segments whose temporal extents align tightly with the ground-truth boundaries.

This is exactly the behaviour that NFM is designed to produce. By supervising the relative rotation of each token’s local reference frame in so(3), the pretext task forces the encoder to represent when a local part changes its motion state, which is the cue that delimits an action boundary. An unstructured change-magnitude target, such as MaST-Pre’s cardinality diference or our own TDD, registers that something changed but not how the local frame reoriented, and therefore leaves boundary transitions between two motion states of similar magnitude largely unsupervised. The comparatively modest scores of MaST-Pre and M2PSC under the strict F1@50 metric (63.8 and 65.9, both on the same P4Transformer backbone as MoSaiC) are consistent with this interpretation.

Table 2: Temporal action segmentation results on HOI4D. Methods are grouped by backbone, and MoSaiC is instantiated on both PPTr and P4Transformer (its default backbone) to verify that its gains generalize across backbones. Baseline numbers for STRL, VideoMAE, C2P, and M2PSC are as reported in their respective papers.
<table><tr><td>Method</td><td>Frame-Acc (%)</td><td>Edit</td><td>F1@10</td><td>F1@25</td><td>F1@50</td></tr><tr><td>PPTr [14] (scratch)</td><td>77.4</td><td>80.1</td><td>81.7</td><td>78.5</td><td>69.5</td></tr><tr><td>+ STRL [42]</td><td>78.4</td><td>79.1</td><td>81.8</td><td>78.6</td><td>69.7</td></tr><tr><td>+ VideoMAE [23]</td><td>78.6</td><td>80.2</td><td>81.9</td><td>78.7</td><td>69.9</td></tr><tr><td>+ C2P [6]</td><td>81.1</td><td>84.0</td><td>85.4</td><td>82.5</td><td>74.1</td></tr><tr><td>+ MoSaiC (Ours)</td><td>83.5</td><td>84.9</td><td>88.1</td><td>84.1</td><td>77.4</td></tr><tr><td>P4Transformer [1] (scratch)</td><td>71.2</td><td>73.1</td><td>73.8</td><td>69.2</td><td>58.2</td></tr><tr><td>+ C2P [6]</td><td>73.5</td><td>76.8</td><td>77.2</td><td>72.9</td><td>62.4</td></tr><tr><td>+ MaST-Pre [4]</td><td>74.1</td><td>75.4</td><td>76.6</td><td>73.4</td><td>63.8</td></tr><tr><td>+ M2PSC [8]</td><td>75.9</td><td>77.1</td><td>78.4</td><td>74.1</td><td>65.9</td></tr><tr><td>+ MoSaiC (Ours)</td><td>81.9</td><td>83.6</td><td>86.2</td><td>83.4</td><td>76.8</td></tr></table>

Where the Action Segmentation Gain Comes From. The aggregate 13.0-point F1@50 margin over MaST-Pre in Table 2 could in principle come from a uniform improvement across all 19 HOI4D action classes, or from a concentrated improvement on a subset of them, these two scenarios support very diferent explanations of why MoSaiC helps. Table 3 breaks down the per-class F1@50 for the four classes with the largest gain and the three with the smallest gain (the remaining 12 classes fall between these extremes and average +13 2, close to the overall margin). The gain is clearly concentrated, the four largest-gain classes, Turnon, Papercut, Dump, and Open, all involve a sustained rotation, either of a hand-held object about the wrist, as in tipping a bucket or kettle, or of an articulated part about its joint axis, as in turning a knob, closing the blades of a pair of scissors, or swinging a lid or a laptop screen, these classes improve by 16.9 to 22.4 points. The three smallest-gain classes, Carry, Reachout, and Rest, improve by only 2.1 to 4.8 points, each is characterised either by translation-dominated hand motion with a nearly constant object orientation or by the absence of motion altogether, so the orientation-change information that NFM supplies adds little beyond what appearance and TDD cues already provide.

This pattern is precisely what we would expect if the aggregate gain is driven by

![](images/8e1e1b9eabda060b3026c533cf45091180d4bef6b676a23371bfb52c6fb8beb2.jpg)  
Figure 4: Qualitative comparison of frame-level action segmentation on a representative ∼150-frame HOI4D clip. From top to bottom, Ground Truth, the P4Transformer baseline, and MoSaiC. The horizontal axis is the frame index, and each color denotes a distinct action class, so a correct prediction requires both the colorblock boundaries to align with those of Ground Truth (accurate temporal localization) and the color identity to match (correct class prediction).

NFM rather than by a generic efect of more pre-training or more parameters, classes whose boundaries are defined by rotation dynamics benefit most from a pretext task that explicitly supervises rotation, while classes that are translation-dominated or static benefit comparatively little. It also clarifies a point left implicit in the discussion of Table 2, namely why the margin over MaST-Pre grows with the overlap threshold, rotation-heavy classes require precise boundary localization to be counted as correct at F1@50, so a method that captures orientation dynamics should disproportionately gain at the strict threshold, which is exactly the trend observed.

Table 3: Per-class F1@50 on HOI4D temporal action segmentation, showing the four classes with the largest MoSaiC gain over MaST-Pre and the three with the smallest. The overall row is the class-wise mean reported in Table 2, the 12 classes omitted here have gains between +4 8 and +16 9 and average +13 2.
<table><tr><td>Action Class</td><td>MaST-Pre</td><td>MoSaiC Δ</td></tr><tr><td>Turnon</td><td>41.2 63.6</td><td>+22.4</td></tr><tr><td>Papercut</td><td>47.5</td><td>67.6 +20.1</td></tr><tr><td>Dump</td><td>52.1 70.7</td><td>+18.6</td></tr><tr><td>Open</td><td>58.3 75.2</td><td>+16.9</td></tr><tr><td>Carry</td><td>74.6 79.4</td><td>+4.8</td></tr><tr><td>Reachout</td><td>73.1 76.5</td><td>+3.4</td></tr><tr><td>Rest</td><td>81.9 84.0</td><td>+2.1</td></tr><tr><td>All 19 classes (mean)</td><td>63.8 76.8</td><td>+13.0</td></tr></table>

Figure 4 complements this per-class quantitative breakdown with a qualitative view. The P4Transformer baseline tends to oversegment, splitting several ground-truth segments into shorter, inconsistently colored fragments, and misassigns the class of some segments altogether. MoSaiC’s predicted color bands track Ground Truth much more closely, both in where segment boundaries fall and in which class label each segment receives, visually corroborating the more precise motion-boundary localization that we

attribute to NFM above.

Semantic Segmentation. As with action segmentation, Table 4 groups the HOI4D semantic segmentation results by backbone. On its default P4Transformer backbone, MoSaiC achieves 43.7% mIoU, exceeding its strongest backbone-matched competitor, M2PSC, by 1.4 mIoU, and MaST-Pre by 3.4 mIoU. Instantiated on the PPTr backbone, MoSaiC reaches 44.3% mIoU, exceeding the strongest PPTr-based baseline, C2P, by 2.0 mIoU, confirming that the benefit of the pretext task is not confined to a single backbone. This result deserves emphasis because semantic segmentation is a static per-point labelling task in which motion is not directly required at inference time. A motion-guided pretext task could plausibly bias the encoder away from the fine geometric detail that per-point labelling needs, yet we observe the opposite.

We attribute this to the geometric formulation of our motion target. Because NFM is computed from a local PCA reference frame, predicting it requires the encoder to retain an accurate description of the local surface orientation of each token, not merely a coarse notion of displacement. Surface orientation is simultaneously the primary cue for distinguishing object parts, so the motion objective and the semantic objective reinforce rather than compete with each other. The fact that MaST-Pre improves only marginally over its backbone on this task (40.3 versus 40.1 mIoU) supports the view that the benefit comes from the geometric nature of the target rather than from motion supervision in general.

Table 4: Semantic segmentation results (mIoU, %) on HOI4D. Methods are grouped by backbone, and MoSaiC is instantiated on both PPTr and P4Transformer (its default backbone) to verify that its gains generalize across backbones. Baseline numbers for STRL, VideoMAE, C2P, and M2PSC are as reported in their respective papers.
<table><tr><td>Method</td><td>mIoU (%)</td></tr><tr><td>PPTr [14] (scratch)</td><td>41.0</td></tr><tr><td>+ STRL [42]</td><td>41.2</td></tr><tr><td>+ VideoMAE [23]</td><td>41.3</td></tr><tr><td>+ C2P [6]</td><td>42.3</td></tr><tr><td>+ MoSaiC (Ours)</td><td>44.3</td></tr><tr><td>P4Transformer [1] (scratch)</td><td>40.1</td></tr><tr><td>+ MaST-Pre [4]</td><td>40.3</td></tr><tr><td>+ C2P [6]</td><td>41.4</td></tr><tr><td>+ M2PSC [8]</td><td>42.3</td></tr><tr><td>+ MoSaiC (Ours)</td><td>43.7</td></tr></table>

## 4.4. Label Eficiency and Linear Probing

A key motivation of self-supervised pre-training is to reduce the dependence on costly 4D annotations. We therefore evaluate MoSaiC under two additional protocols. (i) Label eficiency, we fine-tune the pre-trained encoder using only a fraction (10% and 50%) of the labeled training data, and compare against training from scratch and against representative self-supervised counterparts. (ii) Linear probing, we freeze the pre-trained encoder and train only a linear classifier on top of it, which directly measures the linear separability and quality of the learned representations. Table 5 reports the results on MSR-Action3D action recognition. With only 10% of the labels, MoSaiC attains 77.8% accuracy against 62.4% for the from-scratch baseline, a margin of 15.4 points that is more than three times the 4.24-point margin observed under full supervision. The advantage narrows to 8.3 points at 50% labels, showing that the value of pre-training is largest exactly where annotation is scarcest. MoSaiC also leads the strongest self-supervised counterpart by 3.2 points in the 10% regime, indicating that the gain is not merely an artifact of pre-training in general but of the specific pretext design.

Linear probing isolates this further. Because the encoder is frozen, the resulting 86.9% accuracy measures how much action-discriminative structure is already linearly accessible in the representation before any task-specific adaptation. MoSaiC exceeds MaST-Pre by 2.6 points and PointCMP by 3.3 points under this protocol. Since the three methods share the same backbone and the same 200-epoch pre-training schedule, the diference is attributable to the objectives themselves, cross-view token consistency supplies an instance-discriminative signal that pure reconstruction lacks, and it operates at token granularity, so the resulting structure survives the pooling step used by the linear classifier.

Table 5: Label-eficiency and linear-probing results for action recognition on MSR-Action3D (Top-1 accu racy, %). FT denotes fine-tuning with the indicated fraction of labels. LP denotes linear probing. The FT (100%) column repeats the fully-supervised results of Table 1.
<table><tr><td>Method</td><td>FT (10%)</td><td>FT (50%)</td><td>FT (100%)</td><td>LP (100%)</td></tr><tr><td>Scratch (no pre-training)</td><td>62.4</td><td>82.1</td><td>90.94</td><td>N/A</td></tr><tr><td>MaST-Pre [4]</td><td>74.6</td><td>88.7</td><td>94.08</td><td>84.3</td></tr><tr><td>PointCMP [5]</td><td>73.9</td><td>88.2</td><td>93.27</td><td>83.6</td></tr><tr><td>MoSaiC (Ours)</td><td>77.8</td><td>90.4</td><td>95.18</td><td>86.9</td></tr></table>

## 4.5. Transfer Learning

Cross-dataset transfer. To assess generalization beyond the pre-training domain, we transfer an encoder pre-trained on HOI4D to the SHREC’17 and NvGesture handgesture benchmarks, and fine-tune it using their labeled training sets. Table 6 reports these transfer-learning results. MoSaiC reaches 91.4% on SHREC’17 and 87.9% on NvGesture, improving over the from-scratch baseline by 2.3 and 1.8 points and over MaST-Pre by 1.3 and 1.2 points. This setting is a demanding test of generality, because HOI4D consists of egocentric human-object interactions at room scale whereas the two targets consist of close-range hand gestures, the two domains share neither object categories, spatial scale, nor capture viewpoint. That the pre-trained encoder still transfers positively indicates that what MoSaiC learns is a generic prior over how local geometry deforms over time, rather than a memorized inventory of HOI4D-specific shapes. Such a prior remains valid under a change of domain, which is consistent with the fact that the transfer margins are of the same order as the in-domain margins rather than collapsing.

Table 6: Cross-dataset transfer for action recognition (Top-1 accuracy, %). Models are pre-trained on the source and fine-tuned on each target.
<table><tr><td>Method</td><td>Source → Target</td><td>Acc (%)</td></tr><tr><td>Scratch (no pre-training)</td><td>N/A → SHREC&#x27;17 / NvGesture</td><td>89.1 / 86.1</td></tr><tr><td>MaST-Pre [4]</td><td>HOI4D → SHREC&#x27;17 / NvGesture</td><td>90.1 / 86.7</td></tr><tr><td>MoSaiC (Ours)</td><td>HOI4D → SHREC&#x27;17 / NvGesture</td><td>91.4/87.9</td></tr></table>

## 4.6. Eficiency Analysis

Table 7 compares model complexity and pre-training cost with representative selfsupervised methods. Because MoSaiC discards the decoder, the predictor, and all motion-target computations after pre-training, its fine-tuning and inference cost is identical to the underlying backbone, all three methods in Table 7 share the same 43.6M encoder parameters and 78.4G inference FLOPs.

The only overhead is therefore confined to pre-training, where MoSaiC requires 30.2 hours against 21.5 hours for MaST-Pre (1 4×), almost entirely because two complementary views are encoded instead of one, the saliency scores and NFM targets reuse quantities (SHOT descriptors, local PCA statistics) that the tokenizer already computes for the single-view baselines, so they add negligible cost on top of the dualview forward pass.

Table 7: Eficiency comparison. Params (M) and FLOPs (G) are measured on the encoder used for downstream inference, Pre-train is the wall-clock pre-training time in hours.
<table><tr><td>Method</td><td>Params</td><td>FLOPs</td><td>Pre-train</td></tr><tr><td>MaST-Pre [4]</td><td>43.6</td><td>78.4</td><td>21.5</td></tr><tr><td>PointCMP [5]</td><td>43.6</td><td>78.4</td><td>19.8</td></tr><tr><td>MoSaiC (Ours)</td><td>43.6</td><td>78.4</td><td>30.2</td></tr></table>

## 4.7. Ablation Studies

We conduct ablation studies to analyze the contribution of each component in Mo-SaiC. Unless otherwise specified, action recognition is evaluated on MSR-Action3D (Top-1 accuracy) and the dense prediction tasks on HOI4D (action segmentation F1@10 and semantic segmentation mIoU).

Component Analysis. Table 8 reports a cumulative ablation that progressively adds each proposed component to the baseline. The baseline adopts random masking with appearance (Chamfer) reconstruction plus the TDD target, which plays the role of the change-magnitude target used by prior work, so that the NFM column isolates the efect of our own motion target rather than of motion supervision in general. Adding Curriculum Motion-Saliency Masking (CMSM) directs the pretext task toward dynamic regions, further adding Normal-Flow Motion (NFM) modeling injects explicit geometric motion supervision, and finally Cross-view Token Consistency Prediction (CTCP) enforces cross-view consistency. Each component brings consistent improvements across all three tasks, and the full MoSaiC achieves the best results, validating that the three components are complementary.

The magnitude of each increment is informative. CMSM alone contributes the single largest gain on action segmentation $( 7 8 . 6  8 2 . 1 \mathrm { F } 1 @ 1 0 , + 3 . 5 )$ , which confirms that where the model is asked to reconstruct matters as much as what it is asked to reconstruct, redirecting the same reconstruction budget toward motion-salient tokens improves temporal understanding without adding any new loss term. NFM adds a further +2 6 F1@10 and +1 1 mIoU, and is the only component that improves the two dense tasks by comparable amounts, consistent with its target being geometric rather than purely temporal. CTCP contributes the smallest increment on the dense tasks (+1 5 F1@10) but the largest single increment on recognition (+2 0), matching its role as an instance-level discriminative signal rather than a localization signal. The dense-task increments do shrink as components accumulate $( + 3 . 5 , + 2 . 6 , + 1 . 5 \mathrm { F } 1 @ 1 0 )$ , as expected once the earlier components have already absorbed the easiest headroom, but the ordering across metrics is not the same for the three components, CMSM is strongest on action segmentation, NFM is the only component that lifts the two dense tasks by comparable amounts, and CTCP is strongest on recognition. This dissociation, rather than the raw size of the increments, is what indicates that the three objectives supervise largely non-overlapping aspects of the representation.

Table 8: Cumulative component ablation. We progressively add each proposed component to the baseline (random masking + appearance reconstruction + TDD). Action Rec., MSR-Action3D Top-1 accuracy, Action Seg., HOI4D F1@10, Sem. Seg., HOI4D mIoU.
<table><tr><td>Configuration</td><td>CMSM</td><td>NFM</td><td>CTCP Action Rec. Action Seg. Sem. Seg.</td><td></td><td></td></tr><tr><td>Baseline</td><td></td><td></td><td>91.3</td><td>78.6</td><td>40.4</td></tr><tr><td>+ CMSM</td><td>√</td><td></td><td>92.4</td><td>82.1</td><td>41.8</td></tr><tr><td> $\mathbf { \Gamma } + \mathbf { C } \mathbf { M } \mathbf { S } \mathbf { M } + \mathbf { N } \mathbf { F } \mathbf { M }$ </td><td>V</td><td>V</td><td>93.2</td><td>84.7</td><td>42.9</td></tr><tr><td>MoSaiC (full)</td><td></td><td></td><td>95.18</td><td>86.2</td><td>43.7</td></tr></table>

Masking Strategy. Table 9 compares diferent masking strategies. Replacing random masking with pure motion-saliency masking already yields +1 2 F1@10, and adding the curriculum schedule contributes a further +1 1, for a total of +2 3 over the random baseline. The second increment is the one that validates our design choice. Motion-saliency masking without a curriculum aggressively hides the most dynamic tokens from the very first epoch, at which point the encoder has not yet learned a usable geometric prior and the reconstruction target is close to unpredictable, the resulting gradients are dominated by noise. By annealing α from 0 to $\alpha _ { \mathrm { m a x } }$ , CMSM lets the model first acquire generic structure under an essentially uniform masking distribution and only then shifts the dificulty toward dynamic regions. The consistent ordering across both dense tasks indicates that this is a genuine optimization efect rather than a task-specific artifact.

Motion Targets. Table 10 evaluates the impact of diferent motion reconstruction targets. Any form of motion supervision improves over appearance-only reconstruction, but the two targets are not interchangeable, NFM alone outperforms TDD alone by 1.1 F1@10 and 0.7 mIoU. This gap supports our central claim that an unstructured change-magnitude target is too coarse. TDD records how much each histogram bin of a token changed, but two motions with the same descriptor perturbation, for instance a rotation about one axis and a rotation of similar magnitude about another, are assigned indistinguishable targets, whereas NFM assigns each a distinct element of so(3) with an explicit axis and angle.

Table 9: Ablation on masking strategies. Action Seg., HOI4D F1@10, Sem. Seg., HOI4D mIoU.
<table><tr><td>Masking Strategy</td><td>Action Seg. Sem. Seg.</td></tr><tr><td>Random 83.9</td><td>42.1</td></tr><tr><td>Motion Saliency (MS)</td><td>85.1 42.8</td></tr><tr><td>MS + Curriculum (CMSM)</td><td>86.2 43.7</td></tr></table>

Combining both targets is nonetheless better than either alone (+0 8 F1@10 over NFM), which indicates that they are complementary rather than competing. The two cover disjoint failure modes, TDD remains informative under occlusion, dis-occlusion, and deformation, precisely the conditions in which the eigengap criterion $g _ { i }$ of Section 3.5 withdraws NFM supervision because the local frame is ill-conditioned, conversely NFM resolves the direction of a rigid motion that TDD can only report as a magnitude. Supervising both therefore yields motion signals over a broader range of conditions than either target can cover in isolation.

Table 10: Ablation on motion reconstruction targets. Action Seg., HOI4D F1@10, Sem. Seg., HOI4D mIoU.
<table><tr><td>Motion Target</td><td>Action Seg.</td><td>Sem. Seg.</td></tr><tr><td>None (Appearance only)</td><td>82.4</td><td>41.5</td></tr><tr><td>TDD</td><td>84.3</td><td>42.4</td></tr><tr><td>NFM</td><td>85.4</td><td>43.1</td></tr><tr><td>TDD + NFM</td><td>86.2</td><td>43.7</td></tr></table>

CTCP Design. The preceding ablations validate CMSM and NFM but leave CTCP, our third component, supported only by its contribution to Table 8. Table 11 isolates the design choices identified in Section 3.6. The row without CTCP reproduces the CMSM + NFM configuration of Table 8. Removing the predictor while keeping the stop-gradient target collapses performance far below even that configuration (78.1 F1@10, 39.1 mIoU, versus 84.7/42.9 without CTCP at all), confirming that an explicit predictor is not an optional refinement but a necessary condition for the objective to avoid the degenerate solution discussed above, without it, the shared pressure toward a trivial constant representation actively harms the features learned by the appearance and motion losses rather than merely failing to help. Replacing the stop-gradient online target with an EMA target is a viable alternative (86.0/43.5) but is marginally worse and adds a second set of momentum-updated weights, so we retain the simpler stopgradient design. The complement overlap ratio γ has a clear optimum at 0 1, and the two directions in which it degrades have diferent causes. $\mathbf { A } \mathbf { t } \gamma = 0$ the number of CTCP pairs is maximal, but View B then observes exactly $M _ { A } , i . e .$ , only the high-saliency tokens, and loses the static context that anchors its own reconstruction losses, since those losses are what keep the CTCP target branch from drifting, the targets themselves become less reliable. $\mathrm { A t } \gamma \ge 0 . 3$ the opposite happens, View B masks progressively fewer tokens, its reconstruction task becomes easy, and the number of B → A CTCP pairs falls by $( 1 - \gamma ) ( G - | M _ { A } | )$

Table 11: Ablation on CTCP design. Action Rec., MSR-Action3D Top-1 accuracy, Action Seg., HOI4D F1@10, Sem. Seg., HOI4D mIoU.
<table><tr><td>Configuration</td><td>Action Rec.</td><td>Action Seg.</td><td>Sem. Seg.</td></tr><tr><td>w/o CTCP (dual view only)</td><td>93.2</td><td>84.7</td><td>42.9</td></tr><tr><td>w/o predictor</td><td>90.9</td><td>78.1</td><td>39.1</td></tr><tr><td>EMA target instead of stop-gradient</td><td>94.9</td><td>86.0</td><td>43.5</td></tr><tr><td> $\gamma = 0$ </td><td>94.6</td><td>85.0</td><td>43.0</td></tr><tr><td> $\gamma = 0 . 3$ </td><td>94.9</td><td>85.8</td><td>43.4</td></tr><tr><td> $\gamma = 0 . 5$ </td><td>94.7</td><td>85.3</td><td>43.2</td></tr><tr><td> $\gamma = 0 . 1 \ : ( \mathrm { M o S a i C ) }$ </td><td>95.18</td><td>86.2</td><td>43.7</td></tr></table>

Robustness to Sparsity, Noise, and Frame Drop. Section 3 noted that NFM’s local reference frame becomes ill-conditioned under sparse or degraded input. Table 12 tests this directly by comparing MoSaiC with an NFM-ablated variant (CMSM + TDD + CTCP, i.e., the full model minus the NFM loss) under three perturbations applied consistently at pre-training, fine-tuning, and test time, reducing the number of points sampled per frame, adding Gaussian coordinate noise with σ expressed in units of the unit-normalized scene extent, and randomly dropping frames. Applying the perturbation during pre-training is essential here, because the local PCA reference frame that defines the NFM target is computed only at pre-training time, a perturbation confined to the downstream stages could not test the conditioning of that frame at all. At the default operating point (2048 points, no noise, no dropped frames), NFM contributes +1 9 F1@10 and +1 3 mIoU on top of the ablated variant, matching the TDD-versus-TDD+NFM comparison in Table 10. This contribution shrinks steadily as points are removed, down to +0 2 F1@10 at 256 points per frame, and similarly shrinks under frame dropping, down to +0 6 F1@10 at a 20% drop rate, both perturbations directly degrade the point count and temporal adjacency that the local PCA reference frame depends on, exactly the failure mode anticipated in Section 3. In contrast, NFM’s contribution is far less afected by Gaussian coordinate noise, retaining +1 3 to +1 4 F1@10 across $\sigma \in \{ 0 . 0 1 , 0 . 0 2 \}$ against the near-total loss observed under sparsity, since covariance-based PCA on a fixed, suficiently large point set is inherently a smoothing operation and is far less sensitive to per-point jitter than to a reduction in the number of points available to estimate the covariance in the first place. Even where NFM’s own contribution nearly vanishes, the full MoSaiC pipeline still degrades gracefully rather than catastrophically, because CMSM and CTCP do not share NFM’s dependence on a well-conditioned local frame.

Qualitative Results. Figure 5 visualizes the complementary masking and reconstruction results of MoSaiC on representative action sequences, showing its ability to recover both the static structure and dynamic movements from the two masked views. Feature Visualization. To further assess the discriminativeness of the learned representations, Figure 6 shows a t-SNE [43] projection of the encoder features on the action recognition test set. Compared with the from-scratch baseline, the features produced by MoSaiC form more compact and better-separated clusters per action category, qualitatively confirming that our pre-training learns more semantically structured representations.

Table 12: Robustness of the NFM contribution under input degradation. w/o NFM denotes CMSM + TDD + CTCP without the NFM loss, Full denotes MoSaiC. ∆ is the F1@10 gain attributable to adding NFM under each condition.
<table><tr><td>Perturbation</td><td>Level</td><td>w/o NFM F1@10</td><td>w/o NFM mIoU</td><td>Full F1@10</td><td>Full mIoU</td><td>Δ F1@10</td></tr><tr><td rowspan="4">Points per frame</td><td>2048 (default)</td><td>84.3</td><td>42.4</td><td>86.2</td><td>43.7</td><td>+1.9</td></tr><tr><td>1024</td><td>83.4</td><td>41.5</td><td>84.6</td><td>42.8</td><td>+1.2</td></tr><tr><td>512</td><td>80.9</td><td>39.8</td><td>81.3</td><td>40.9</td><td>+0.4</td></tr><tr><td>256</td><td>74.0</td><td>36.5</td><td>74.2</td><td>36.8</td><td>+0.2</td></tr><tr><td rowspan="2">Gaussian noise σ</td><td>0.01</td><td>83.8</td><td>41.9</td><td>85.1</td><td>43.0</td><td>+1.3</td></tr><tr><td>0.02</td><td>81.0</td><td>40.0</td><td>82.4</td><td>41.2</td><td>+1.4</td></tr><tr><td rowspan="2">Frame drop rate</td><td>10%</td><td>83.0</td><td>41.2</td><td>84.3</td><td>42.6</td><td>+1.3</td></tr><tr><td>20%</td><td>79.5</td><td>39.0</td><td>80.1</td><td>40.3</td><td>+0.6</td></tr></table>

## 5. Discussion

Robustness of the Motion Target. NFM’s supervision signal is derived from a local PCA reference frame, and Section 3 anticipated that this frame becomes ill-conditioned under sparse or degraded input. Table 12 confirms this is a real, quantifiable limitation rather than a theoretical concern, NFM’s own contribution to F1@10 shrinks from +1 9 at the default 2048 points per frame to +0 2 at 256 points, and from +1 9 to +0 6 under a 20% frame-drop rate, since both perturbations directly reduce the point count and temporal adjacency the local frame depends on. Encouragingly, the eigengap-based exclusion criterion we use to mask out unreliable tokens (Section 3) keeps this degra-

![](images/76fb76f96d548df9c4aa202b3fba96d4d1792534d57e925dbf3aa47003e3a909.jpg)  
Figure 5: Qualitative visualization of complementary masking and reconstruction. For each sample we show the input, the two complementary masked views (Mask-A / Mask-B), and the corresponding reconstructions (Recon-A / Recon-B).

dation graceful rather than catastrophic, because CMSM and CTCP do not share this dependency and continue to contribute under the same conditions. A confidenceweighted or fully learned formulation of the exclusion threshold, rather than our fixed one, is a natural next step for deployment on sensors with lower point density than the datasets studied here.

Why the Margin on MSR-Action3D Is Smallest. MoSaiC leads every baseline on all three action-recognition datasets, temporal action segmentation, semantic segmentation, label eficiency, linear probing, and cross-dataset transfer, but the margin it achieves on MSR-Action3D is the narrowest of the three action-recognition benchmarks, +0 42 over the strongest baseline, C2P (95.18 vs. 94.76, Table 1), compared with +0 7 on SHREC’17 and +0 5 on NvGesture. We attribute this to a mismatch between what NFM supervises and what the benchmark rewards, MSR-Action3D consists of 20 coarse, whole-body actions (e.g., high arm wave, forward kick) that are largely identifiable from posture in a handful of frames, so the fine-grained local rotation signal NFM provides has comparatively little additional discriminative value once appearance is well modeled. C2P’s complete-to-partial distillation, in contrast, directly supervises whole-sequence completeness, which aligns more closely with a benchmark where the relevant motion is coarse and global rather than local and rotational. This is consistent with the pattern in Table 8, where NFM’s increment is smaller on action recognition (+0 8) than on the two dense tasks (+2 6 F1@10, +1 1 mIoU), it is CTCP’s instance-discriminative signal, not NFM’s motion signal, that contributes most of MoSaiC’s advantage on MSR-Action3D. We view this as an informative pattern that clarifies NFM’s intended regime (local, dense, rotation-bearing motion) rather than as evidence against the general design.

![](images/13adf917796bee483144b83e98e2b5e48f3f91be243ee186ef7152be319411ca.jpg)  
Figure 6: t-SNE visualization of the encoder features (each color denotes an action category). Left, training from scratch. Right, MoSaiC pre-training.

## 6. Conclusion

In this paper, we presented MoSaiC, a self-supervised masked-modeling framework that learns transferable point cloud video representations by jointly exploiting motion-saliency-guided masking, explicit local geometric motion supervision, and complementary cross-view consistency. Experiments across four datasets and three downstream task families demonstrate its consistent efectiveness, particularly for dense spatio-temporal tasks and low-label settings, while highlighting robustness under severe input degradation as an important direction for future work.

## CRediT authorship contribution statement

Wei Wang: Conceptualization, Methodology, Writing – original draft. Yiding Sun: Methodology, Software, Validation, Writing – original draft. Yuyan Wang: Resources, Visualization. Zhuoyue Zhang: Investigation. Zhengqiao Li: Investigation. Dongfu Yin: Writing – review & editing. Chen Li: Writing – review & editing.

## Declaration of competing interests

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## Acknowledgments

This work was supported in part by the Technology Innovation Leading Program of Shaanxi (Program No. 2024QY-SZX-23), the Shenzhen Science and Technology Program under Grant KJZD20240903104400001, and the Research Task Assignment Project from Guangdong Laboratory of Artificial Intelligence and Digital Economy (SZ) under Grant No. GML-26420004, and by the High-performance Computing Platform of Peking University.

## References

[1] H. Fan, Y. Yang, M. Kankanhalli, Point 4d transformer networks for spatiotemporal modeling in point cloud videos, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2021.

[2] H. Fan, X. Yu, Y. Ding, Y. Yang, M. Kankanhalli, Point spatio-temporal transformer networks for point cloud video modeling, IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 2023.

[3] Y. Liu, Y. Liu, C. Jiang, K. Lyu, W. Wan, H. Shen, B. Liang, Z. Fu, H. Wang, L. Yi, HOI4D: A 4d egocentric dataset for category-level human-object interaction, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022.

[4] Z. Shen, X. Sheng, H. Fan, L. Wang, Y. Guo, Q. Liu, H. Wen, X. Zhou, Masked spatio-temporal structure prediction for self-supervised learning on point cloud videos, in: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2023.

[5] Z. Shen, X. Sheng, L. Wang, Y. Guo, Q. Liu, X. Zhou, PointCMP: Contrastive mask prediction for self-supervised learning on point cloud videos, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023.

[6] Z. Zhang, Y. Dong, Y. Liu, L. Yi, Complete-to-partial 4d distillation for selfsupervised point cloud sequence representation learning, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023.

[7] X. Sheng, Z. Shen, G. Xiao, Learnable query contrast and spatio-temporal prediction on point cloud video pre-training, IEEE Latin America Transactions, 2024.

[8] Y. Han, C. Xu, R. Xu, J. Qian, J. Xie, Masked motion prediction with semantic contrast for point cloud sequence learning, in: European Conference on Computer Vision (ECCV), 2024.

[9] C. R. Qi, H. Su, K. Mo, L. J. Guibas, PointNet: Deep learning on point sets for 3d classification and segmentation, in: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017.

[10] C. R. Qi, L. Yi, H. Su, L. J. Guibas, PointNet++: Deep hierarchical feature learning on point sets in a metric space, in: Advances in Neural Information Processing Systems (NeurIPS), 2017.

[11] X. Liu, M. Yan, J. Bohg, Meteornet: Deep learning on dynamic 3d point cloud sequences, in: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2019.

[12] H. Fan, X. Yu, Y. Ding, Y. Yang, M. Kankanhalli, PSTNet: Point spatio-temporal convolution on point cloud sequences, in: International Conference on Learning Representations (ICLR), 2021.

[13] J.-X. Zhong, K. Zhou, Q. Hu, B. Wang, N. Trigoni, A. Markham, No pain, big gain: Classify dynamic point cloud sequences with static models by fitting feature-level space-time surfaces, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022.

[14] H. Wen, Y. Liu, J. Huang, B. Duan, L. Yi, Point primitive transformer for longterm 4d point cloud video understanding, in: European Conference on Computer Vision (ECCV), 2022.

[15] D. Zhang, Y. Wang, Y. Sun, H. Xu, P. Fan, J. Zhu, CMHANet: A cross-modal hybrid attention network for point cloud registration, Neurocomputing, 2026.

[16] D. Zhang, Y. Sun, P. Li, Y. Liu, H. Lin, H. Xu, X. Mu, L. Lin, W. Yan, N. Yang, C. Fang, J. Zhao, J. Zhu, C. He, C. Tan, PointCoT: A multi-modal benchmark for explicit 3d geometric reasoning, arXiv preprint arXiv:2602.23945, 2026.

[17] Y. Sun, J. Zhu, H. Cheng, C. Lu, Z. Yang, L. Chen, Y. Wang, Align then adapt: Rethinking parameter-eficient transfer learning in 4D perception, IEEE Transactions on Multimedia (TMM), 2026.

[18] Z. Guo, J. Zhu, J. Liu, A. S. Mian, Mantis: Mamba-native tuning is eficient for 3d point cloud foundation models, arXiv preprint arXiv:2605.03438, 2026.

[19] Y. Wang, Y. Sun, Q. Wang, P. Li, C. Lu, D. Zhang, PointRFT: Explicit reinforcement fine-tuning for point cloud few-shot learning, arXiv preprint arXiv:2603.23957, 2026.

[20] Z. Guo, J. Zhu, Y. Sun, L. Chen, D. Wang, Parameter-eficient fine-tuning for spiking point cloud models, arXiv preprint arXiv:2607.29048, 2026.

[21] Y. Sun, X. Yang, D. Zhang, Q. Wang, Z. Xu, W. Liu, S. Li, J. Zhu, Z. Yu, T. Huang, SpikingMOT: A spike-driven multi-object tracker, arXiv preprint arXiv:2607.19875, 2026.

[22] K. He, X. Chen, S. Xie, Y. Li, P. Dollár, R. Girshick, Masked autoencoders are scalable vision learners, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022.

[23] Z. Tong, Y. Song, J. Wang, L. Wang, VideoMAE: Masked autoencoders are dataeficient learners for self-supervised video pre-training, in: Advances in Neural Information Processing Systems (NeurIPS), 2022.

[24] X. Yu, L. Tang, Y. Rao, T. Huang, J. Zhou, J. Lu, Point-BERT: Pre-training 3d point cloud transformers with masked point modeling, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022.

[25] Y. Pang, W. Wang, F. E. H. Tay, W. Liu, Y. Tian, L. Yuan, Masked autoencoders for point cloud self-supervised learning, in: European Conference on Computer Vision (ECCV), 2022.

[26] R. Zhang, Z. Guo, P. Gao, R. Fang, B. Zhao, D. Wang, Y. Qiao, H. Li, Point-M2AE: Multi-scale masked autoencoders for hierarchical point cloud pretraining, in: Advances in Neural Information Processing Systems (NeurIPS), 2022.

[27] Y. Sun, H. Cheng, C. Lu, Z. Li, M. Wu, H. Lu, J. Zhu, HyperPoint: Multimodal 3d foundation model in hyperbolic space, Pattern Recognition (PR), 2026.

[28] P. Li, Y. Sun, H. Cheng, PointDico: Contrastive 3d representation learning guided by difusion models, in: Proceedings of the International Joint Conference on Neural Networks (IJCNN), 2025.

[29] Y. Sun, C. Lu, H. Cheng, J. Wang, H. Lu, L. Chen, J. Zhu, Curve3D: Curvatureaware masked autoencoder for self-supervised point cloud understanding, Pattern Recognition (PR), 2026.

[30] X. Han, Y. Sun, C. Lu, Rethinking regressor in 3d gaussian pretraining, in: Proceedings of the Chinese Conference on Pattern Recognition and Computer Vision (PRCV), 2025.

[31] Z. You, J. Zhu, Y. Sun, Z. Guo, H. Cheng, D. Zhang, L. Chen, H. Luo, GaussFusion: Towards multimodal 3d gaussian pretraining, arXiv preprint arXiv:2607.05906, 2026.

[32] D. Zhang, Y. Sun, C. Tan, W. Yan, N. Yang, J. Zhu, H. Zhang, Chain-of-thought compression should not be blind: V-skip for eficient multimodal reasoning via dual-path anchoring, in: Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (ACL), 2026.

[33] S. Xie, J. Gu, D. Guo, C. R. Qi, L. Guibas, O. Litany, PointContrast: Unsupervised pre-training for 3d point cloud understanding, in: European Conference on Computer Vision (ECCV), 2020.

[34] J.-B. Grill, F. Strub, F. Altché, C. Tallec, P. H. Richemond, E. Buchatskaya, C. Doersch, B. A. Pires, Z. D. Guo, M. G. Azar, et al., Bootstrap your own latent: A new approach to self-supervised learning, in: Advances in Neural Information Processing Systems (NeurIPS), 2020.

[35] X. Chen, K. He, Exploring simple siamese representation learning, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2021.

[36] Z. Zhang, J. Zhu, C. Fang, J. Liu, A. S. Mian, Difusion masked pretraining for dynamic point cloud, arXiv preprint arXiv:2605.03639, 2026.

[37] Y. Sun, D. Zhang, J. Zhu, H. Cheng, Z. Li, P. Li, C. Fang, Y. Dong, L. Chen, Tri-eficient transfer learning for point cloud videos, arXiv preprint arXiv:2606.24175, 2026.

[38] F. Tombari, S. Salti, L. Di Stefano, Unique signatures of histograms for local surface description, in: European Conference on Computer Vision (ECCV), 2010.

[39] W. Li, Z. Zhang, Z. Liu, Action recognition based on a bag of 3d points, in: IEEE Computer Society Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), 2010.

[40] Q. De Smedt, H. Wannous, J.-P. Vandeborre, J. Guerry, B. Le Saux, D. Filliat, SHREC’17 track: 3d hand gesture recognition using a depth and skeletal dataset, in: Eurographics Workshop on 3D Object Retrieval (3DOR), 2017.

[41] P. Molchanov, X. Yang, S. Gupta, K. Kim, S. Tyree, J. Kautz, Online detection and classification of dynamic hand gestures with recurrent 3d convolutional neural networks, in: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016.

[42] S. Huang, Y. Xie, S.-C. Zhu, Y. Zhu, Spatio-temporal self-supervised representation learning for 3d point clouds, in: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2021.

[43] L. van der Maaten, G. Hinton, Visualizing data using t-SNE, Journal of Machine Learning Research (JMLR), 2008.