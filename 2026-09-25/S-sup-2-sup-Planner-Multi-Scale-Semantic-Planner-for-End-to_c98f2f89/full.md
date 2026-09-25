# S<sup>2</sup>Planner: Multi-Scale Semantic Planner for End-to-End

Autonomous Driving

Zhaowei Lu, Liguo Zhou, Yujie Guo, Lei Yu, Alois Knoll Fellow, IEEE,

Abstract—We present S<sup>2</sup>Planner, a trajectory planner that combines three front-facing cameras with ego-motion history and the current driving command. A fine-tuned DINOv3 backbone and a Spatial Tuning Adapter produce multi-scale image features; a coarse-to-fine decoder then uses trajectory self-attention and camera-projected cross-attention to refine candidate waypoints. The contribution is the integration of ego-conditioned trajectory initialization with iterative, geometry-guided sampling of multiscale image features, rather than a new visual backbone or attention operator. On the NAVSIM v1 non-reactive evaluation, the previously reported navtest run obtained 88.03 PDMS. Because that run was selected using navtest performance, this number is exploratory and cannot be interpreted as an unbiased test estimate. Validation-selected evaluation on unexposed data, repeated runs, and computational measurements are needed to establish generalization and efficiency.

Index Terms—Autonomous Driving, End-to-End Framework, Path Planning, Computer Vision, Multi-scale Semantic Feature.

## I. INTRODUCTION

OTION planning in autonomous driving requires a motion, and a navigation command to choose a feasible future trajectory. End-to-end planners can learn this mapping directly, but differences in sensing, state inputs, evaluation protocols, and checkpoint selection complicate comparisons. NAVSIM v1 [1] evaluates proposed trajectories in a four-second nonreactive simulation: other actors follow recorded motion, while the ego vehicle is simulated along the proposed plan. Its Predictive Driver Model Score (PDMS) measures several aspects of plan quality, but does not establish performance in interactive traffic or a deployed closed-loop system.

Image-based planners have used BEV representations, large trajectory vocabularies, and iterative generative models [2], [3], [4]. These are useful design choices, each with different modeling and computational trade-offs. We investigate whether multi-scale image features can instead be sampled directly around candidate trajectory points. Our model uses three front-facing cameras together with ego velocity, longitudinal acceleration, pose history, and a driving command. We therefore describe it as a camera-and-ego-state planner; the absence of LiDAR does not make it an image-only method.

S<sup>2</sup>Planner combines a fine-tuned DINOv3 encoder [5] and the Spatial Tuning Adapter from DEIMv2 [6] with a trajectory decoder. Ego-conditioned tokens first generate coarse candidates. At each refinement stage, trajectory selfattention exchanges information among candidate waypoints, and spatial cross-attention samples multi-scale image features at calibrated projections of those waypoints. The candidate coordinates are updated between stages. DINOv3, the adapter, conditional transformer blocks, and deformable attention are established components; the proposed contribution is their particular trajectory-centered coupling and its evaluation in this planning setting.

An earlier navtest experiment yielded 88.03 PDMS. The checkpoint was selected by navtest score, so that result and the corresponding ablations are exploratory. We retain them to document the existing experiments while identifying the validation-selected rerun, unexposed evaluation data, and controlled comparisons required for a confirmatory result. In particular, the small difference from DiffusionDrive cannot support a superiority claim without repeated runs and uncertainty estimates.

The paper makes three contributions:

• A trajectory-centered decoder that updates coarse candidates through ego-conditioned initialization and calibrated sampling of multi-scale camera features.

• An explicit account of how features from the concatenated camera input are associated with individual camera calibration matrices, together with the limitations introduced by attention across image seams.

• An exploratory NAVSIM v1 component study, with the effect of test-guided selection stated explicitly and a protocol for validation-selected follow-up evaluation.

## II. RELATED WORKS

## A. Foundation Models for Visual Feature Learning

Self-supervised foundation models have become a powerful means of obtaining robust visual encoders without manual labels. The DINO family exemplifies this trend: DINOv2 [7] demonstrates strong transfer with self-distilled ViT backbones, and DINOv3 [5] scales training to 1.7 billion images, producing high-resolution features with strong semantic grouping. Such properties are valuable for autonomous driving, where lane markings, traffic signs, and vulnerable road users must be recognized under varied conditions.

Vision–language models such as CLIP [8], ALIGN [9], BLIP and BLIP-2 [10], [11] further show that large-scale pretraining can enrich contextual understanding. Although not designed for driving, these models demonstrate how broad semantic priors can enhance downstream visuomotor tasks.

Foundation-model backbones have also been adapted to dense prediction. DEIM and DEIMv2 [12], [6] integrate DI-NOv3 features with multi-scale matching for object detection, motivating the use of multi-scale feature hierarchies in planning. Other self-supervised approaches, including MAE [13], BEiT [14], SimMIM [15], and data2vec [16], aim to learn transferable visual tokens. We study DINOv3 with a multiscale adapter in the NAVSIM v1 non-reactive planning setting.

## B. Deformable Attention and Multi-Scale Transformers

Transformers provide global context but become costly at high resolution due to dense attention. Deformable attention [17] restricts each query to a sparse set of learned sampling locations, including across multi-scale feature maps. Deformable DETR [18] and its variants [19], [20] use this mechanism for object detection; BEVFormer [2] and DETR3D [21] use calibrated multi-camera image features for 3D perception.

These ideas have recently influenced sequence modeling and planning. Our approach applies stacked deformable selfand cross-attention to trajectory queries and multi-scale image features. Sparse sampling reduces the number of sampled feature locations, although end-to-end runtime must be measured before an efficiency or real-time claim can be made.

## C. Coarse-to-Fine and Generative Trajectory Planning

Coarse-to-fine refinement is a long-standing strategy in vision and planning. Two-stage detectors [22], [23] refine coarse proposals, and trajectory forecasting methods such as ThinkTwice [24] and TrajFine [25] apply similar cascades to improve waypoint quality. Diffusion-based planners such as DiffRefiner [26] model multimodal futures through iterative denoising. S<sup>2</sup>Planner instead produces candidate paths and refines them through a fixed stack of attention layers. The effect of this design on latency and prediction quality remains an empirical question.

## D. Vision-Based End-to-End Autonomous Driving

End-to-end driving aims to map raw sensory input directly to future trajectories or controls. Early CNN-based steering models [27] handled only simple scenarios, motivating later works that introduced high-level commands [28], multi-camera architectures, and attention mechanisms. Transformer-based planners, including cascaded decoders in ThinkTwice [24], and BEV-based systems [2], [29], [30], integrate scene context through cross-attention or multi-sensor fusion.

Multi-task frameworks such as YOLOP [31] and UniAD [32] jointly optimize perception and planning but increase model complexity and latency. Vectorized representations, as used in VAD [33], offer flexible trajectory modelling but often rely on large anchor sets or predefined motion vocabularies.

Our method combines established components in a specific way: ego-conditioned coarse trajectory tokens are updated using direct, calibrated sampling from multi-scale image features, without an explicit BEV prediction head or a fixed trajectory vocabulary. Like other NAVSIM agents, it also uses ego-state and command information. The present experiments compare selected components, but do not isolate every architectural choice or establish lower computational cost than the closest prior designs.

## III. METHOD

## A. Initialization & Tokenization

We learn trajectory tokens $Q _ { \mathrm { t r a j } } \in \mathbb { R } ^ { M \times L \times D }$ , where M is the number of candidate modes, L is the number of future waypoints, and D is the token dimension. Unlike methods that initialize candidates from a fixed cluster vocabulary, this representation is learned with the planner. The coarse waypoints subsequently predicted from these tokens are still used as reference points for attention; “anchor-free” here refers only to the absence of a predefined trajectory vocabulary.

## B. Semantic-Aware Multi-Scale Scene Encoder

1) Semantic Feature Extraction: Traditional convolutionbased backbones such as ResNet architectures [34] combined with Feature Pyramid Networks (FPN) [35] have long underpinned object detection and instance segmentation. These networks build hierarchical representations via stacked convolutions and multi-scale pyramids to mitigate the inherent single-scale nature of CNNs. Although effective in capturing local spatial structures, they often exhibit limited generalization ability and require extensive task-specific fine-tuning, increasing training cost. Their locality bias further constrains global reasoning: even with dilated convolutions or deeper designs, long-range dependencies and globally coherent semantics remain difficult to model, restricting performance in complex scene understanding tasks.

Recent advances in the DINO family have redefined feature extraction by leveraging Vision Transformers (ViT) [36] and self-supervised learning. Through global self-attention, DINObased models learn robust, semantically aligned, and objectcentric representations that generalize well across diverse tasks with minimal fine-tuning. This cross-task transferability significantly reduces dependence on dataset-specific training and has established the DINO series as a dominant choice for modern visual backbones.

The DINO family applies self-supervised learning to vision transformers. The original DINO work [37] demonstrated emergent semantic structure in self-supervised ViT features; DINOv2 [7] and DINOv3 [5] extended this line of work with larger-scale pretraining. These pretrained features motivate their use as a planning backbone, but their effect in this application must be established by controlled evaluation.

We use the front-left, front, and front-right cameras. After resizing each image to $H \times W$ , we concatenate them in that order to form $I _ { \mathrm { a l l } } \in \mathbb { R } ^ { H \times 3 W \times 3 }$ . The DINOv3 backbone is initialized from DINOv3 pretraining and fine-tuned for planning. Joint self-attention lets tokens on one view access tokens from other views; it also permits attention across artificial seams between images with different camera perspectives. We do not assume that the stitched image is a single pinhole view. The downstream projection uses the calibration of each original camera. The effect of joint processing, compared with separate per-camera encoding, has not yet been isolated experimentally.

![](images/c1476582973bc73f86e6703b37a939e660dac551531c02a132cbeab4ac0b8acb.jpg)  
Fig. 1. S<sup>2</sup>Planner architecture. A fine-tuned DINOv3 backbone with a Spatial Tuning Adapter extracts multi-scale features from three concatenated camera images. Ego history and status condition coarse trajectory tokens and waypoints. Stacked trajectory self-attention and camera-projected cross-attention refin the candidates; the highest-scoring mode is returned.

For patch size $P$ and camera dimensions divisible by $P ,$ the stitched input has $N \ = \ 3 H W / P ^ { 2 }$ spatial patches. We collect spatial outputs from selected DINOv3 blocks as $F ^ { ( \ell ) } = B _ { \ell } \bar { ( I _ { \mathrm { a l l } } ) }$ , where $B _ { \ell }$ denotes the backbone through block ℓ. The decoder uses these spatial features, not an imagelevel class output. We do not specify the internal token types or positional encoding here because the exact DINOv3 variant and checkpoint have not been recovered from the original configuration.

2) Multi-Scale Representation Module: DINOv3 patch tokens are produced on a native spatial grid. Planning can benefit from features at additional resolutions, particularly when a projected waypoint lies near a small object or boundary. Figure 2 shows one example of the backbone’s spatial features.

![](images/6cb578a73cc0da1c4c2f05671f1f9efa157f4b974775074b91a23d065f42c7c3.jpg)  
Fig. 2. Concatenated front-facing images and a PCA visualization of DINOv3 patch features. This example illustrates the feature structure; a single visualization cannot establish a general loss of boundary or small-object information.

The visualization motivates multi-scale sampling, but it does not by itself measure localization or detection accuracy.

We adopt the Spatial Tuning Adapter (STA) from DEIMv2 [6]. It converts intermediate DINOv3 features into a multi-scale hierarchy, as illustrated in fig. 3. The adapter is an existing component; its contribution to this planner is assessed through the component ablation, subject to the test-selection caveat in section IV.

![](images/3ce199b0c8b34222ba12f672984c76918713695e75e3f60aae7eb67434b2eb32.jpg)  
Fig. 3. DINOv3 and the Spatial Tuning Adapter architecture adopted from DEIMv2 [6] to form multi-scale features.

Let $F _ { \mathrm { a l l } } ^ { ( s ) } \ \in \ \mathbb { R } ^ { H _ { s } \times 3 W _ { s } \times D _ { s } }$ denote a stitched feature map at scale s, with its width partitioned into three equal camera bands. The spatial features used by cross-attention are $F _ { i } ^ { ( s ) } =$ $F _ { \mathrm { a l l } } ^ { ( s ) } [ : , ( i - 1 ) W _ { s } : i W _ { s } , : ]$ for $i \in \{ 1 , 2 , 3 \}$ , ordered frontleft, front, front-right. A 3D point is projected with camera i’s own resized-image calibration; sampling coordinates are then normalized within $F _ { i } ^ { ( s ) }$ , not within the full $3 W _ { s }$ width. This is a spatial split of jointly encoded features, so each band can still contain information from other cameras through self-attention. Correct band boundaries require patch-aligned resizing or equivalent padding; the preprocessing implementation and feature-grid shapes must be checked against the training code before a reproducibility claim is made.

Algorithm 1: Initial Trajectory Encoding with Condi  
tional DiT Block   
Input: Trajectory tokens $Q _ { \mathrm { t r a j } }$   
Input: Ego history $H _ { \mathrm { e g o } } ,$ Status $s _ { \mathrm { e g o } }$   
Output: $\mathrm { \bar { \it T } _ { c o a r s e } } ,$ , Q<sub>traj-coarse</sub>   
$Q _ { \mathrm { h i s t } } $ HistEncoder(H<sub>ego</sub>) ; // Encode ego history   
$Q _ { \mathrm { c o a r s e } }  \mathrm { C r o s s A t t n } ( Q _ { \mathrm { t r a j } } , Q _ { \mathrm { h i s t } } ) ;$   
$/ /$ Inject priors   
$( \gamma , \beta )$ ← CondML $\mathrm { \bf \nabla } { \cal P } ( s _ { \mathrm { e g o } } )$ ; // Compute modulation   
$\hat { Q } \gets \mathrm { I }$ LayerNorm(Q<sub>coarse</sub>);   
$\hat { Q }  \gamma \odot \hat { Q } + \beta ;$ ; // Apply condition   
Q<sub>DiT</sub> ← Q<sub>coarse</sub> + TransformerBlock(Q<sup>ˆ</sup>);   
$/ / \quad \mathtt { S e l f } \ \mathtt { A t t { t n } }$   
$Q _ { \mathrm { t r a j - c o a r s e } }  \mathrm { F F N } ( Q _ { \mathrm { D i T } } ) + Q _ { \mathrm { D i T } } \ ; / /$ Final projection   
$T _ { \mathrm { c o a r s e } }  \mathrm { M L P } ( Q _ { \mathrm { t r a j - c o a r s e } } ) \ ; \quad \ / /$ Output coarse traj   
return $T _ { \mathrm { c o a r s e } } , Q _ { \mathrm { t r a j - c o a r s e } } ;$

## C. Coarse-to-Fine Trajectory Decoder

The trajectory decoder refines an ego-conditioned coarse proposal in three stages: initial trajectory encoding, a spatial transformer, and trajectory updating and scoring (fig. 1). Its distinguishing connection is that predicted metric waypoints determine where each refinement block samples calibrated image features.

1) Initial Trajectory Encoding: The decoder starts from the initial trajectory tokens produced in III-A. These tokens represent different candidate motion modes and are used as queries, while the ego history acts as the source of contextual information. Concretely, the ego history (e.g., past positions, headings) is first embedded by an MLP into a sequence of ego history features, which serve as keys and values in a crossattention layer. This Initial Trajectory Encoding step injects temporal motion priors from the ego’s past behavior into each trajectory mode: tokens corresponding to different modes can attend to different segments of the ego history, enabling the model to initialize each mode with a plausible global motion trend rather than a purely data-agnostic guess.

We then apply a transformer block with scale-and-shift conditioning inspired by DiT [38]. The trajectory tokens supply queries, keys, and values, while instantaneous ego speed, longitudinal acceleration, and the driving command are embedded into modulation parameters. This block does not perform diffusion sampling. The modulation gives the block access to current vehicle state; the present experiments do not isolate its contribution.

A feed-forward network with a residual connection updates the tokens, and an MLP predicts coarse future waypoints. This produces $\pmb { T } _ { \mathrm { c o a r s e } } \in \mathbb { R } ^ { M \times L \times 3 }$ and $Q _ { \mathrm { t r a j , c o a r s e } } \in \dot { \mathbb { R } } ^ { \hat { M } \times L \times D }$ . At this point the candidates are conditioned on ego history and status; image information enters later through spatial crossattention. Algorithm 1 summarizes this initialization.

2) Spatial Transformer: The spatial transformer contains trajectory self-attention (TSA) and spatial cross-attention (SCA). TSA exchanges information among candidate trajectory tokens; SCA samples the multi-scale camera features around calibrated projections of the current trajectory points.

Both attention operators adapt deformable sampling from prior work [2], [18]. We use the current predicted waypoints as reference locations rather than a fixed library of trajectory anchors. Sparse sampling reduces attention locations, but whole-model speed and memory have not been measured.

a) Trajectory Self Attention: In Trajectory Self Attention (TSA), coarse trajectory points $\pmb { T } _ { \mathrm { c o a r s e } } \in \mathbb { R } ^ { M \times }$ L×3 provide reference coordinates in the current ego frame. The manuscript specifies a forward region $x ~ \in ~ [ 0 , 3 2 ]$ m and lateral region $y \in [ - 3 2 , 3 2 ]$ m, with positive x forward and positive y to the left. The center waypoints and four egofootprint corners are normalized to this region as proposed reference locations [39]. The number and distribution of any additional sampled locations remain to be verified against the implementation.

For TSA, queries and values originate from the coarse trajectory tokens $Q _ { \mathrm { t r a j , c o a r s e } }$ generated in section III-C1. We summarize the intended operation abstractly as

$$
\mathrm { T S A } ( Q _ { m } ^ { p } , V _ { T S A } ) = \sum \mathrm { D e f o r m A t t n } ( Q _ { m } ^ { p } , p , V _ { T S A } ) |\tag{1}
$$

where $Q _ { m } ^ { p }$ is the query for mode m at waypoint p. Learned offsets can shift reference locations and permit information exchange between candidate modes. The $M \times L \times D$ token tensor is indexed by mode and time, however, not by a regular $( x , y )$ image grid. Therefore, the physical meaning of bilinear sampling in this tensor, its spatial layout, and its offset parameterization cannot be established from the manuscript alone and must be specified from the implementation. Figure 4 is a conceptual illustration of reference-point sampling, not evidence of a particular interpolation rule.

![](images/02f930822adfcb9977c5256fdede550424b120bf49f493617ec90e965cb3077f.jpg)  
Fig. 4. Visualization of the anchor-based feature sampling. The interaction of the queries in the trajectory feature space is activated while the dense attention computation is prevented. For the sake of simplicity, the offset is not presented here.

b) Spatial Cross Attention: The updated trajectory feature tokens $Q _ { t r a j - T S A } ~ \in ~ \mathbb { R } ^ { M \times L \times D }$ from Trajectory Self Attention module are fed into the Spatial Cross Attention (SCA) layer, where they act as queries, while the multi-scale semantic features from the visual backbone serve as keys and values. Instead of applying standard multi-head attention directly over all views and all feature tokens with different scale, which leads to prohibitive computational and memory costs, the SCA implements the deformable attention to 3D space. In particular, each trajectory query token q attends only to a small set of spatially relevant locations across the camera views, rather than densely interacting with all image features. Also, since the deformable attention was originally proposed for purely 2D perception tasks, where queries and sampled features lie in the same image plane, several modifications are necessary when extending it to 3D scenes and trajectory representations.

The coarse trajectory point $p = ( x , y )$ remains in the current ego vehicle’s metric frame. Following the pillar-style construction in [2], $N _ { \mathrm { r e f } }$ heights $z _ { j }$ produce 3D reference points. Each point is projected independently into each calibrated camera $i \colon$

$$
\begin{array} { r } { \tilde { { \mathbf { u } } } _ { i j } = P _ { i } [ x , y , z _ { j } , 1 ] ^ { \mathsf { T } } , } \end{array}
$$

$$
\mathcal { P } _ { i } ( p , z _ { j } ) = ( \tilde { u } _ { i j , 1 } / \tilde { u } _ { i j , 3 } , \tilde { u } _ { i j , 2 } / \tilde { u } _ { i j , 3 } )\tag{2}
$$

(3)

where $P _ { i } \in \mathbb { R } ^ { 3 \times 4 }$ maps ego-frame 3D points to camera i’s resized pixel grid. A projection is valid only if its depth $\tilde { u } _ { i j , 3 }$ is positive and its pixel coordinate is inside that camera’s $W \times H$ image. For scale s, valid pixel coordinates are transformed to the local coordinate system of $F _ { i } ^ { ( s ) }$ before deformable sampling. In particular, the front-camera coordinate is never offset by the width of the front-left image during calibration projection; the offset is used only to extract its band from the stitched feature tensor.

Finally, all sampled features would be aggregated based on a set of softmax-normalized attention weights generated by a MLP with $Q _ { t r a j - T S A }$ as input. The overall process could be summarized as:

$$
\frac { 1 } { | \mathcal { N } _ { h i t } | } \sum _ { i \in \mathcal { N } _ { h i t } } \sum _ { j = 1 } ^ { N _ { \mathrm { r e f } } } \sum _ { s } \mathrm { D e f o r m A t t n } \big ( Q _ { m } ^ { p } , \mathcal { P } _ { i } ( p , z _ { j } ) , \pmb { F } _ { i } ^ { ( s ) } \big )\tag{4}
$$

Here $i , j ,$ , and s index the camera, pillar height, and feature scale. $\mathcal { N } _ { h i t }$ is the set of cameras with valid projections, and the expression is evaluated only when that set is nonempty; the implementation’s fallback for an empty set must be verified. $Q _ { m } ^ { p }$ is the query for mode m at waypoint $p .$ As illustrated in fig. 5, invalid projections should be masked before attention weights are normalized; the exact masking behavior requires code verification. The per-view tensors remain jointly encoded because DINOv3 attended across the concatenated input; a separate-view encoder is needed to measure whether crossseam attention helps or harms.

![](images/81e0633f24ae69e4d059d8990913d8e0a3116c16eeeb0d3874215c44522a76ce.jpg)  
Fig. 5. Illustration of the principle of Spatial Cross Attention. The reference anchors are directly derived from the actual trajectory coordinates, making sure that the metric consistency is preserved. For each trajectory query, only the feature from region where the projected 2D point locates inside the image plane would be seen as valid. The new trajectory feature is computed based on a weighted aggregation of the valid image features

3) Trajectory Updating & Scoring: The Trajectory Self Attention module and the Spatial Cross Attention module together form the core of a Spatial Transformer block. This block is stacked K times to obtain a hierarchical refinement architecture for trajectory prediction. After each iteration, the refined trajectory tokens are passed through an MLP decoder to predict trajectory coordinates directly. These trajectory coordinates are the updated trajectory points, yielding progressively more accurate trajectory estimates. The updated points then serve as new reference points for the next refinement stage. The final fine-grained trajectory tokens would be also processed by another MLP decoder, to get scores for each of the mode of trajectories. The whole process could be summarized in following:

$$
Q ^ { k } = S F _ { k } ( Q ^ { k - 1 } ) ,
$$

$$
T ^ { k } = M L P ( Q ^ { k } ) ,\tag{5}
$$

(6)

$$
S _ { p r e d } = M L P ( Q ^ { K } ) .\tag{7}
$$

where $k = 1 , \ldots , K$ indexes the refinement stage. $Q ^ { k }$ illustrates the output trajectory queries of k-th Spatial Transformer layer $S F _ { k } , \pmb { T } ^ { k }$ denotes the corresponding trajectory predictions decoded from $Q ^ { k }$ , and $S _ { p r e d }$ denotes the final set of scores associated with the trajectories derived from the laststage tokens $Q ^ { K }$

The decoder thus uses updated trajectory coordinates to choose the image sampling locations at the next stage. Its effect on final planning scores is examined through the exploratory component ablation in section IV-E.

## D. Loss

The overall training objective is composed of two main components: a trajectory regression loss and a score-based classification loss. The trajectory loss supervises the predicted future motion at multiple refinement stages, while the score loss supervises the mode probabilities associated with the final trajectory set. The first part of the trajectory loss is the intermediate trajectory loss applied to multiple stages of the refinement pipeline. Inspired by [39], We apply a simple Minimum over N (MoN) loss [40], which is defined as:

$$
\mathcal { L } _ { i n t e r m e d i a t e } = \sum _ { k = 0 } ^ { K } \alpha ^ { K - k } \operatorname* { m i n } _ { m \in [ 0 , \ldots , M - 1 ] } \| \hat { \pmb { T } } _ { k } ^ { m } - \pmb { T } _ { g t } \|\tag{8}
$$

As written, $k = 0$ denotes the coarse prediction and $k =$ $1 , \ldots , K$ denotes the K spatial-transformer outputs. Thus the sum includes the final stage, which also receives the separate final loss below. The intended supervision stages and decay factor α must be checked against the training code. Here $\hat { \pmb { T } } _ { k } ^ { m }$ is mode m at stage $k ,$ and $\bar { \pmb { T } } _ { q t } \in \mathbb { R } ^ { L \times 3 }$ is the expert trajectory.

The second part of the trajectory loss is applied at the final refinement stage and directly supervises the multi-modal set of trajectories and their associated scores. For the final layer output $\hat { \pmb { T } } _ { f i n a l } \ \in \ \mathbb { R } ^ { M \times L \times 3 }$ and the predicted scores $S _ { p r e d } \in \mathbb { R } ^ { M }$ , we construct a one-hot mode label $S _ { l a b e l } \in \mathbb { R } ^ { M }$ as follows:

$$
S _ { l a b e l } ^ { m } = \left\{ { 1 , \mathrm { i f } m = m ^ { * } , } \atop { 0 , \mathrm { o t h e r w i s e . } }  \right.
$$

Specifically, the trajectory points $\hat { T } _ { f i n a l } ^ { m ^ { * } } \in \mathbb { R } ^ { L \times 3 }$ that is closest to the expert trajectory points $\pmb { T } _ { g t } \doteq \mathbb { R } ^ { L \times 3 }$ (in terms of the $\ell _ { 1 }$ distance measure) is identified and labeled as positive, while all other modes are labeled as negative.

Given the predicted scores $S _ { p r e d }$ and the label vector $S _ { l a b e l }$ two particular trajectories are then extracted from the multimodal set. First, a predicted-best trajectory points $\hat { \pmb { T } } _ { p r e d } \in$ $\mathbb { R } ^ { L \times 3 }$ is obtained by selecting the trajectory mode with the highest predicted score:

$$
\hat { \mathbf { T } } _ { p r e d } = \hat { \pmb { T } } _ { f i n a l } ^ { \mathrm { a r g m a x } _ { m } } \pmb { S } _ { p r e d } ^ { m }\tag{9}
$$

Second, a label-best trajectory points $\hat { \pmb { T } } _ { l a b e l } \in \mathbb { R } ^ { L \times 3 }$ is obtained by selecting the trajectory mode with the highest label value, which is equivalent to the distance-based best mode:

$$
\hat { \pmb { T } } _ { l a b e l } = \hat { \pmb { T } } _ { f i n a l } ^ { \mathrm { a r g m a x } _ { m } } \hat { S } _ { l a b e l } ^ { m } = \hat { \pmb { T } } _ { f i n a l } ^ { m ^ { * } }\tag{10}
$$

For the score supervision, the score loss is formulated as:

$$
\mathcal { L } _ { l a b e l } = \mathcal { L } _ { B C E } ( \sigma ( S _ { p r e d } ) , S _ { l a b e l } )\tag{11}
$$

where σ denotes the sigmoid function applied element-wise to the predicted scores. The final loss could then be defined as:

$$
\mathcal { L } _ { f } = \frac { \mathcal { L } _ { p r e d - g t } + \mathcal { L } _ { l a b e l - g t } + \mathcal { L } _ { p r e d - l a b e l } } { 3 } + \lambda \mathcal { L } _ { l a b e l }\tag{12}
$$

where each $\mathcal { L } _ { t r a j _ { 1 } - t r a j _ { 2 } }$ denotes an $\ell _ { 1 }$ loss between two sets of trajectory points, and λ is a weighting factor that balances the trajectory regression terms and the score supervision term $\mathcal { L } _ { l a b e l }$ . By averaging $\mathcal { L } _ { p r e d - g t } , \mathcal { L } _ { l a b e l - g t }$ and $\mathcal { L } _ { p r e d - l a b e l }$ , the final-stage trajectory loss ensures that the model learns to produce at least one accurate trajectory in the multi-modal set, to assign high scores to accurate trajectories, and to maintain consistency between the scoring mechanism and the regression quality. The total loss is $\mathcal { L } _ { t o t a l } = \mathcal { L } _ { f } + \mathcal { L } _ { i n t e r m e d i a t e } .$

## IV. EXPERIMENTS

## A. Benchmark and Evaluation Protocol

We study NAVSIM v1 [1], whose official filtered splits are navtrain and navtest; there is no standard navval filter in the v1.1 split documentation. A validation set for model selection must therefore be constructed from navtrain with driving logs kept disjoint between training and validation, and its scene identifiers published. NAVSIM evaluates an agentproposed trajectory in a four-second non-reactive simulation: background agents follow recorded future motion, while an LQR controller tracks the proposed ego trajectory. The planner is not queried repeatedly in an interactive rollout. PDMS therefore measures the proposed trajectory under this fixedfuture evaluation, not long-horizon closed-loop behavior with agents reacting to the ego vehicle.

The NAVSIM v1.1 evaluator combines no-at-fault collision (NC) and drivable-area compliance (DAC) as multiplicative factors with ego progress (EP), time-to-collision (TTC), and comfort (C) as weighted terms [1]:

$$
\mathrm { P D M S } = \mathrm { N C D A C } \left( \frac { \mathrm { 5 E P + 5 T T C + 2 C } } { 1 2 } \right) .\tag{13}
$$

The reference implementation is the NAVSIM v1.1 scorer at https://github.com/autonomousvision/navsim/blob/v1. 1/navsim/planning/simulation/planner/pdm planner/scoring/ pdm scorer.py. The official per-scenario scores are aggregated by the evaluator. Equation (13) must not be applied to rounded, dataset-averaged columns to reconstruct dataset PDMS.

## B. Implementation Details and Reproducibility

1) Inputs and Optimization: S<sup>2</sup>Planner uses front-left, front, and front-right RGB cameras and four categories of nonimage input: current ego velocity, longitudinal acceleration, recent pose history in the ego frame, and a driving command. Thus the camera entry in the sensor column of table I indicates the absence of LiDAR; it does not indicate image-only input. Images are resized and calibration matrices must be adjusted to the resized pixels. The backbone is initialized from DINOv3 pretrained weights and fine-tuned; it is not initialized from an ImageNet-supervised classification checkpoint. The trajectory decoder is trained from scratch.

The available manuscript records Adam with $\beta _ { 1 } ~ = ~ 0 . 9 .$ $\beta _ { 2 } = 0 . 9 9 9$ , initial learning rate $1 0 ^ { - 4 } ,$ polynomial decay to $1 0 ^ { - 5 }$ , weight decay $1 0 ^ { - 2 } ,$ gradient clipping at norm 1, global mini-batch size 128, mixed precision, and eight NVIDIA A100 GPUs. The archived text does not specify the DINOv3 variant, per-camera image resolution, patch handling at camera seams, number of trajectory modes, waypoint interval, deformableattention sampling count, exact loss weights, training epochs or steps, or random seeds. These values must be recovered from the experiment configuration before the study can be reproduced. The notation M, L, $N _ { \mathrm { r e f } }$ , K, α, and λ in section III does not substitute for configuration values. The main architecture figure also depicts a different order of history attention and conditional transformer processing from Algorithm 1; the executed order must be checked in code.

TABLE I  
PREVIOUSLY REPORTED NAVSIM V1 NAVTEST RESULTS (PERCENT). OUR ROW IS TEST-SELECTED AND EXPLORATORY; THE OTHER ROWS ARE LITERATURE RESULTS REPRODUCED FROM THE ORIGINAL COMPARISON, WITH THEIR SELECTION PROTOCOLS AND NON-IMAGE INPUTS NOT AUDITED HERE. CAMERA DESCRIBES THE SENSOR CATEGORY ONLY. NC, DAC, TTC, COMFORT (C), AND EP ARE DATASET SUMMARIES; DATASET PDMS IS AGGREGATED FROM SCENARIO SCORES. THESE ROWS DO NOT ESTABLISH A STATISTICALLY SIGNIFICANT RANKING.
<table><tr><td>Method</td><td>Sensors</td><td>Anchors</td><td>NC</td><td>DAC</td><td>TTC</td><td>C</td><td>EP</td><td>PDMS</td></tr><tr><td>Transfuser [41]</td><td> $\mathrm { C a m e r a } + \mathrm { L i D A R }$ </td><td>0</td><td>97.7</td><td>92.8</td><td>92.8</td><td>100</td><td>79.2</td><td>84.0</td></tr><tr><td>DRAMA [42]</td><td> $\mathrm { C a m e r a } + \mathrm { L i D A R }$ </td><td>0</td><td>98.0</td><td>93.1</td><td>94.8</td><td>100</td><td>80.1</td><td>85.5</td></tr><tr><td> $\mathrm { V A D v { 2 - } V _ { 8 1 9 2 } } \ [ 4 3 ]$ </td><td> $\mathrm { C a m e r a } + \mathrm { L i D A R }$ </td><td>8192</td><td>97.2</td><td>89.1</td><td>91.6</td><td>100</td><td>76.0</td><td>80.9</td></tr><tr><td>Hydra  $. \mathrm { M D P - V _ { 8 1 9 2 } } \ [ 3 ]$ </td><td> $\mathrm { C a m e r a } + \mathrm { L i D A R }$ </td><td>8192</td><td>97.9</td><td>91.7</td><td>92.9</td><td>100</td><td>77.6</td><td>83.0</td></tr><tr><td> $\mathrm { H y d r a \mathrm { - } M D P \mathrm { - } V _ { 8 1 9 2 } \mathrm { - } W \mathrm { - } E P \ [ 3 ] }$ </td><td> $\mathrm { C a m e r a } + \mathrm { L i D A R }$ </td><td>8192</td><td>98.3</td><td>96.0</td><td>94.6</td><td>100</td><td>78.7</td><td>86.5</td></tr><tr><td>DiffusionDrive [4]</td><td> $\mathrm { C a m e r a } + \mathrm { L i D A R }$ </td><td>20</td><td>98.2</td><td>96.2</td><td>94.7</td><td>100</td><td>82.2</td><td>88.1</td></tr><tr><td>UniAD [32]</td><td>Camera</td><td>0</td><td>97.8</td><td>91.9</td><td>92.9</td><td>100</td><td>78.8</td><td>83.4</td></tr><tr><td>PARA-Drive [44]</td><td>Camera</td><td>0</td><td>97.9</td><td>92.4</td><td>93.0</td><td>99.8</td><td>79.3</td><td>84.0</td></tr><tr><td> $\mathrm { L T F } \ [ 4 1 ]$ </td><td>Camera</td><td>0</td><td>97.4</td><td>92.8</td><td>92.4</td><td>100</td><td>79.0</td><td>83.8</td></tr><tr><td>S2Planner (test-selected)</td><td>Camera + ego state</td><td>0</td><td>98.2</td><td>96.5</td><td>94.3</td><td>100</td><td>82.2</td><td>88.03</td></tr></table>

2) Checkpoint Selection and Result Status: In the original experiments, the checkpoint was selected by the highest navtest PDMS. This exposes the test set to model selection and can bias both the main result and navtest ablations upward. All S<sup>2</sup>Planner values retained below are consequently exploratory, test-selected observations, including the reported 88.03 PDMS. They are not independent estimates of generalization. A corrected development protocol would select the checkpoint and all hyperparameters on a log-disjoint validation subset of navtrain, freeze the configuration, and then evaluate navtest once per selected checkpoint. Because navtest has already influenced this project, such a rerun on the same split cannot restore its status as untouched evidence. An independent generalization claim requires a genuinely unexposed evaluation set. The ablations also need to be rerun under the corrected selection protocol.

3) Required Statistical and Computational Reporting: The original records provide one score per model and no scenario-level score files, repeated-seed results, parameter counts, FLOPs, peak inference memory, batch-one latency, or throughput. Accordingly, this paper does not infer statistical significance, low latency, or real-time suitability from the existing table. A complete comparison should report multiple training seeds as mean and standard deviation, a paired scenario-level analysis on the same navtest scenes, and timing on the same hardware with synchronized inference and a specified precision, input resolution, batch size, and warm-up. Baseline resource figures should be measured under the same conditions or clearly identified as numbers reported by their authors.

## C. Exploratory Quantitative Results

The previously reported $\mathrm { S ^ { 2 } P l a n n e r }$ PDMS of 88.03 is close to DiffusionDrive’s reported 88.1. Their sensors, state channels, compute budgets, and selection procedures have not been matched, and the S<sup>2</sup>Planner score is test-selected. We therefore cannot conclude that either method is better from this comparison. An ego-state-only baseline and a camera-plusstate baseline would help quantify the contribution of visual information beyond the state signals.

TABLE II  
PREVIOUSLY REPORTED COMPONENT ABLATIONS ON NAVTEST. VALUES ARE EXPLORATORY AND TEST-SELECTED; REPEATED SEEDS AND A VALIDATION-SELECTED RERUN ARE PENDING.
<table><tr><td>Variant</td><td>DINOv3</td><td>STA</td><td>Deform.</td><td>PDMS</td></tr><tr><td>A0: ResNet</td><td>x</td><td>x</td><td>x</td><td>72.4</td></tr><tr><td>A1: DINOv3</td><td>√</td><td>x</td><td>x</td><td>84.9</td></tr><tr><td> ${ \mathsf { A } } 2 \colon { \mathsf { D I N O v } } 3 + { \mathsf { S T A } }$ </td><td>√</td><td>√</td><td>x</td><td>86.3</td></tr><tr><td>A3: full model</td><td>√</td><td>√</td><td>√</td><td>88.0</td></tr></table>

## D. Qualitative Examples

Figure 6 displays selected predicted trajectories against expert paths. In these images, several predictions visually follow the road corridor and avoid visible obstacles. Such overlays are illustrative; they do not establish collision freedom, anticipation of another agent’s future action, comfort, or humanlike behavior. Those properties require quantitative scenariolevel metrics or reactive evaluation. The plotted expert and predicted paths also need matched temporal horizons for a direct geometric comparison.

## E. Exploratory Ablations

The earlier component experiments used navtest in a testguided development cycle. Their differences are descriptive and need replication with checkpoints selected on a logdisjoint validation subset of navtrain. The results in table II do not identify the causal effect of one component unless all other training choices, input channels, and compute budgets were held fixed.

Within the reported runs, A1 exceeds A0 by 12.5 points, A2 exceeds A1 by 1.4 points, and A3 exceeds A2 by 1.7 points at the displayed precision. Because these comparisons were developed on navtest and have no uncertainty estimates, they cannot establish which component reliably improves held-out performance. A controlled architecture study should additionally compare separate versus concatenated camera encoding, matched state inputs, and the closest trajectory-refinement decoder under equal training and compute budgets.

![](images/6e66dbd06362855b82d6e8bd24dffdb6c31db1be8d78e0cd6c00d5913974fa9b.jpg)  
Fig. 6. Selected navtest examples. Green shows the displayed expert path and red the predicted path. The displayed paths may cover different time horizons; their apparent lengths should not be interpreted as a planning error or a progress score. These static overlays do not show how other agents would react to the plan.

TABLE III  
PREVIOUSLY REPORTED BEV AUXILIARY-HEAD COMPARISON ON NAVTEST. THIS TESTS ONE BEV-HEAD DESIGN AND LOSS SETTING ONLY; VALUES ARE EXPLORATORY.
<table><tr><td>Variant</td><td>BEV head</td><td>PDMS</td></tr><tr><td>A1: direct regression</td><td>x</td><td>84.9</td></tr><tr><td>A1.1: A1 + BEV auxiliary head</td><td>√</td><td>83.5</td></tr></table>

The single BEV comparison in table III shows a lower score for this particular auxiliary head. It does not show that BEV representations are generally unnecessary: head capacity, loss weighting, gradient interference, and training variance could all affect the result. A stronger test would vary the BEV loss weight and head design under the same validation-selected protocol.

## V. CONCLUSIONS AND FUTURE WORK

S<sup>2</sup>Planner combines a fine-tuned DINOv3 and Spatial Tuning Adapter with an ego-conditioned, coarse-to-fine trajectory decoder. Its specific design samples multi-scale camera features at calibrated projections of candidate waypoints and updates the waypoints between refinement stages. The model uses three RGB cameras, ego velocity, acceleration, pose history, and a driving command.

The previously reported NAVSIM v1 result of 88.03 PDMS was obtained after selecting a checkpoint using navtest. It is therefore exploratory, and the available single-run ablations and qualitative examples cannot establish generalization, a statistically meaningful advantage over close baselines, realtime performance, or interactive driving competence. The BEV experiment applies to one auxiliary head and does not support a general conclusion about BEV representations.

The next experimental step is to create a log-disjoint validation subset from navtrain, select all settings there, and freeze the configuration before further evaluation. Since this project has already used navtest for development, a fresh independent generalization claim requires an unexposed evaluation set. Repeated seeds, paired scenario-level analyses, an ego-stateonly control, a separate-camera-encoding control, and matched comparisons with the closest refinement methods are needed. Parameter count, FLOPs, latency, memory, and throughput should be measured under a specified hardware and software setup. Finally, interactive simulation or real-world testing is necessary to evaluate reactions to other agents and behavior beyond the four-second non-reactive NAVSIM v1 horizon.

## REFERENCES

[1] D. Dauner, M. Hallgarten, T. Li, X. Weng, Z. Huang, Z. Yang, H. Li, I. Gilitschenski, B. Ivanovic, M. Pavone et al., “Navsim: Data-driven non-reactive autonomous vehicle simulation and benchmarking,” Advances in Neural Information Processing Systems, vol. 37, pp. 28 706– 28 719, 2024.

[2] Z. Li, W. Wang, H. Li, E. Xie, C. Sima, T. Lu, Q. Yu, and J. Dai, “Bevformer: Learning bird’s-eye-view representation from multi-camera images via spatiotemporal transformers,” in European Conference on Computer Vision, 2022.

[3] Z. Li, K. Li, S. Wang, S. Lan, Z. Yu, Y. Ji, Z. Li, Z. Zhu, J. Kautz, Z. Wu et al., “Hydra-mdp: End-to-end multimodal planning with multi-target hydra-distillation,” arXiv preprint arXiv:2406.06978, 2024.

[4] B. Liao, S. Chen, H. Yin, B. Jiang, C. Wang, S. Yan, X. Zhang, X. Li, Y. Zhang, Q. Zhang et al., “Diffusiondrive: Truncated diffusion model for end-to-end autonomous driving,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 12 037–12 047.

[5] O. Simeoni, H. V. Vo, M. Seitzer, F. Baldassarre, M. Oquab, C. Jose,´ V. Khalidov, M. Szafraniec, S. Yi, M. Ramamonjisoa et al., “Dinov3,” arXiv preprint arXiv:2508.10104, 2025.

[6] S. Huang, Y. Hou, L. Liu, X. Yu, and X. Shen, “Real-time object detection meets dinov3,” arXiv preprint arXiv:2509.20787, 2025.

[7] M. Oquab, T. Darcet, T. Moutakanni, H. Vo, M. Szafraniec, V. Khalidov, P. Fernandez, D. Haziza, F. Massa, A. El-Nouby et al., “Dinov2: Learning robust visual features without supervision,” arXiv preprint arXiv:2304.07193, 2023.

[8] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark et al., “Learning transferable visual models from natural language supervision,” in International conference on machine learning. PmLR, 2021, pp. 8748–8763.

[9] C. Jia, Y. Yang, Y. Xia, Y.-T. Chen, Z. Parekh, H. Pham, Q. Le, Y.-H. Sung, Z. Li, and T. Duerig, “Scaling up visual and vision-language representation learning with noisy text supervision,” in International conference on machine learning. PMLR, 2021, pp. 4904–4916.

[10] J. Li, D. Li, C. Xiong, and S. Hoi, “Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation,” in International conference on machine learning. PMLR, 2022, pp. 12 888–12 900.

[11] J. Li, D. Li, S. Savarese, and S. Hoi, “Blip-2: Bootstrapping languageimage pre-training with frozen image encoders and large language models,” in International conference on machine learning. PMLR, 2023, pp. 19 730–19 742.

[12] S. Huang, Z. Lu, X. Cun, Y. Yu, X. Zhou, and X. Shen, “Deim: Detr with improved matching for fast convergence,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 15 162– 15 171.

[13] K. He, X. Chen, S. Xie, Y. Li, P. Dollar, and R. Girshick, “Masked au-´ toencoders are scalable vision learners,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 16 000–16 009.

[14] H. Bao, L. Dong, S. Piao, and F. Wei, “Beit: Bert pre-training of image transformers,” arXiv preprint arXiv:2106.08254, 2021.

[15] Z. Xie, Z. Zhang, Y. Cao, Y. Lin, J. Bao, Z. Yao, Q. Dai, and H. Hu, “Simmim: A simple framework for masked image modeling,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 9653–9663.

[16] A. Baevski, W.-N. Hsu, Q. Xu, A. Babu, J. Gu, and M. Auli, “Data2vec: A general framework for self-supervised learning in speech, vision and language,” in International conference on machine learning. PMLR, 2022, pp. 1298–1312.

[17] Z. Xia, X. Pan, S. Song, L. E. Li, and G. Huang, “Vision transformer with deformable attention,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 4794–4803.

[18] X. Zhu, W. Su, L. Lu, B. Li, X. Wang, and J. Dai, “Deformable detr: Deformable transformers for end-to-end object detection,” arXiv preprint arXiv:2010.04159, 2020.

[19] S. Liu, F. Li, H. Zhang, X. Yang, X. Qi, H. Su, J. Zhu, and L. Zhang, “Dab-detr: Dynamic anchor boxes are better queries for detr,” arXiv preprint arXiv:2201.12329, 2022.

[20] D. Meng, X. Chen, Z. Fan, G. Zeng, H. Li, Y. Yuan, L. Sun, and J. Wang, “Conditional detr for fast training convergence,” in Proceedings of the IEEE/CVF international conference on computer vision, 2021, pp. 3651– 3660.

[21] Y. Wang, V. C. Guizilini, T. Zhang, Y. Wang, H. Zhao, and J. Solomon, “Detr3d: 3d object detection from multi-view images via 3d-to-2d queries,” in Conference on robot learning. PMLR, 2022, pp. 180–191.

[22] R. Girshick, J. Donahue, T. Darrell, and J. Malik, “Rich feature hierarchies for accurate object detection and semantic segmentation,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2014, pp. 580–587.

[23] S. Ren, K. He, R. Girshick, and J. Sun, “Faster r-cnn: Towards real-time object detection with region proposal networks,” Advances in neural information processing systems, vol. 28, 2015.

[24] X. Jia, P. Wu, L. Chen, J. Xie, C. He, J. Yan, and H. Li, “Think twice before driving: Towards scalable decoders for end-to-end autonomous driving,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 21 983–21 994.

[25] K.-L. Wang, L.-W. Tsao, J.-C. Wu, H.-H. Shuai, and W.-H. Cheng, “Trajfine: Predicted trajectory refinement for pedestrian trajectory forecasting,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 4483–4492.

[26] L. Yin, R. Ju, G. Guo, and E. Cheng, “Diffrefiner: Coarse to fine trajectory planning via diffusion refinement with semantic interaction for end to end autonomous driving,” arXiv preprint arXiv:2511.17150, 2025.

[27] M. Bojarski, D. Del Testa, D. Dworakowski, B. Firner, B. Flepp, P. Goyal, L. D. Jackel, M. Monfort, U. Muller, J. Zhang et al., “End to end learning for self-driving cars,” arXiv preprint arXiv:1604.07316, 2016.

[28] F. Codevilla, M. Muller, A. L¨ opez, V. Koltun, and A. Dosovitskiy,´ “End-to-end driving via conditional imitation learning,” in 2018 IEEE international conference on robotics and automation (ICRA). IEEE, 2018, pp. 4693–4700.

[29] J. Huang, G. Huang, Z. Zhu, Y. Ye, and D. Du, “Bevdet: Highperformance multi-camera 3d object detection in bird-eye-view,” arXiv preprint arXiv:2112.11790, 2021.

[30] J. Huang and G. Huang, “Bevdet4d: Exploit temporal cues in multicamera 3d object detection,” arXiv preprint arXiv:2203.17054, 2022.

[31] D. Wu, M.-W. Liao, W.-T. Zhang, X.-G. Wang, X. Bai, W.-Q. Cheng, and W.-Y. Liu, “Yolop: You only look once for panoptic driving perception,” Machine Intelligence Research, vol. 19, no. 6, pp. 550–562, 2022.

[32] Y. Hu, J. Yang, L. Chen, K. Li, C. Sima, X. Zhu, S. Chai, S. Du, T. Lin, W. Wang et al., “Planning-oriented autonomous driving,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2023, pp. 17 853–17 862.

[33] B. Jiang, S. Chen, Q. Xu, B. Liao, J. Chen, H. Zhou, Q. Zhang, W. Liu, C. Huang, and X. Wang, “Vad: Vectorized scene representation for efficient autonomous driving,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 8340–8350.

[34] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2016, pp. 770–778.

[35] T.-Y. Lin, P. Dollar, R. Girshick, K. He, B. Hariharan, and S. Belongie,´ “Feature pyramid networks for object detection,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2017, pp. 2117–2125.

[36] A. Dosovitskiy, “An image is worth 16x16 words: Transformers for image recognition at scale,” arXiv preprint arXiv:2010.11929, 2020.

[37] M. Caron, H. Touvron, I. Misra, H. Jegou, J. Mairal, P. Bojanowski, and´ A. Joulin, “Emerging properties in self-supervised vision transformers,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2021, pp. 9650–9660.

[38] W. Peebles and S. Xie, “Scalable diffusion models with transformers,” in Proceedings of the IEEE/CVF international conference on computer vision, 2023, pp. 4195–4205.

[39] K. Guo, H. Liu, X. Wu, J. Pan, and C. Lv, “ipad: Iterative proposal-centric end-to-end autonomous driving,” arXiv preprint arXiv:2505.15111, 2025.

[40] A. Gupta, J. Johnson, L. Fei-Fei, S. Savarese, and A. Alahi, “Social gan: Socially acceptable trajectories with generative adversarial networks,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2018, pp. 2255–2264.

[41] K. Chitta, A. Prakash, B. Jaeger, Z. Yu, K. Renz, and A. Geiger, “Transfuser: Imitation with transformer-based sensor fusion for autonomous driving,” IEEE transactions on pattern analysis and machine intelligence, vol. 45, no. 11, pp. 12 878–12 895, 2022.

[42] C. Yuan, Z. Zhang, J. Sun, S. Sun, Z. Huang, C. D. W. Lee, D. Li, Y. Han, A. Wong, K. P. Tee et al., “Drama: An efficient end-to-end motion planner for autonomous driving with mamba,” arXiv preprint arXiv:2408.03601, 2024.

[43] S. Chen, B. Jiang, H. Gao, B. Liao, Q. Xu, Q. Zhang, C. Huang, W. Liu, and X. Wang, “Vadv2: End-to-end vectorized autonomous driving via probabilistic planning,” arXiv preprint arXiv:2402.13243, 2024.

[44] X. Weng, B. Ivanovic, Y. Wang, Y. Wang, and M. Pavone, “Paradrive: Parallelized architecture for real-time autonomous driving,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 15 449–15 458.

![](images/6916cb20db588bafbf32c73bf8fde9ae3fde40b4b60c76f6c54f8248b9ad2b58.jpg)

Zhaowei Lu received the Bachelor of Science degree from Tongji University and is pursuing a Master of Science degree in Robotics, Cognition, and Intelligence at the Technical University of Munich. His research interests include 3D computer vision, autonomous vehicles, generative AI, and embodied systems.

![](images/75abd26d44d9e33f88d0d17441054f433b8862cd7164c76516767a359efcee46.jpg)

Liguo Zhou received the Bachelor of Science degree in software engineering from Suzhou University, the Master of Science degree in pattern recognition and intelligent systems from Wuhan University, and the doctoral degree in computer science from the Technical University of Munich. His research interests include computer vision, deep learning, autonomous driving, and embodied intelligence.

![](images/4ca9a40686f36f1758a867257c9ba4eb25e4b627ae11978431692eaedff2767b.jpg)

Yujie Guo received the Bachelor of Science degree from Tongji University and the Master of Science degree in Robotics, Cognition, and Intelligence from the Technical University of Munich. His research interests include deep learning, computer vision, and autonomous driving.

![](images/c744a1897ccf220a09236365ae8d3334335b3ac39c7c4e50cc732024a2a2fad5.jpg)

Lei Yu is a professor in the School of Computer Science and Technology at Huaibei Normal University. His research interests include artificial intelligence, information security, and security protocols.

![](images/7ef9e7e3c487eb2da58bc2b520eded42aa794773afc96c31b805ad63aaedbfbe.jpg)

Alois Knoll (Fellow, IEEE) earned a diploma in electrical engineering from the University of Stuttgart in 1985 and a doctorate in computer science from the Technical University of Berlin in 1988. He was a professor at Bielefeld University from 1993 to 2001 and has been a professor at the Technical University of Munich since 2001. His research focuses on autonomous systems, robotics, and artificial intelligence.