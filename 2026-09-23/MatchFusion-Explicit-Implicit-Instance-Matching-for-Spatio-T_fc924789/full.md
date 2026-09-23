# MatchFusion: Explicit–Implicit Instance Matching for Spatio-Temporal Multimodal Autonomous Driving

Xiaoyu Li<sup>1,†</sup>, Jiajia Fu<sup>1,†</sup>, Long Shi<sup>1</sup>, Tianyu Du<sup>1</sup>, Ruihang Li<sup>2</sup>, Xian Wu<sup>1</sup>, Lijun Zhao<sup>1,∗</sup>, Yingtao Zhang<sup>1</sup>, Lining Sun<sup>1</sup>, and Ruifeng Li<sup>1</sup>

Abstract— Sparse instance representations provide a compact interface for spatial LiDAR–camera and temporal past–current interaction in multimodal perception and end-to-end autonomous driving (E2EAD). Effective interaction requires reliable instance correspondences despite geometric discrepancies and heterogeneous semantic representations. Attentionbased methods exploit contextual semantics but often require specialized representation alignment, increasing computational overhead. In contrast, association based on structured object states is efficient and interpretable but lacks contextual evidence to resolve ambiguous matches. To combine these complementary strengths, we propose MatchFusion, a learnable instance matching and fusion module for spatio-temporal multimodal autonomous driving. MatchFusion initializes pairwise affinities using geometric similarity and category consistency, then selectively refines structurally plausible associations using instance embeddings. The resulting soft matchmap guides a common residual aggregation operator for adaptive information exchange. This unified matching–fusion formulation supports spatial LiDAR–camera and temporal past–current interaction, using multi-view image-plane geometry and motioncompensated Bird’s-Eye-View (BEV) geometry as the respective structural priors. Experiments on nuScenes demonstrate consistent perception gains across diverse front-end configurations. Compared with a prior instance-centric fusion method, the MatchFusion-equipped system achieves higher perception accuracy while reducing FLOPs by 55.3% and GPU memory usage by 39.3%, with the matching–fusion module accounting for only 3.7% of total perception latency. Integrating temporal MatchFusion into SparseDrive further improves perception within an end-to-end driving framework without additional supervision. These results establish explicit–implicit matching as an effective and efficient mechanism for spatio-temporal instance interaction. Code will be released.

## I. INTRODUCTION

Reliable multimodal autonomous driving benefits from integrating complementary observations across sensors [1], [2] and time [3], [4]. LiDAR provides accurate spatial geometry, cameras capture rich appearance and semantics, and historical observations supply temporal context beyond individual frames. Recent advances in sparse 3D perception [5], [6] and E2EAD [3] increasingly represent scene entities as instance queries paired with structured object states. These representations provide a compact interface for instance-level reasoning, reducing the need for dense interaction with background features. Spatial LiDAR–camera and temporal past–current interaction can thus be formulated as information exchange between sparse instance sets. Effective exchange requires reliable correspondences and efficient aggregation despite geometric discrepancies and heterogeneous semantic representations across modalities and time.

![](images/4e21da5898b976f652830c92b44edd9da56f79823db15b606fd0471ac51fd7f8.jpg)  
Fig. 1: Various object-level multimodal fusion paradigms.

As illustrated in Fig. 1, existing instance-level interaction methods primarily adopt implicit attention-based interaction or explicit structural association. Query-centric methods [5], [7] extract modality-specific instances and perform crossmodal interaction to integrate their features through learned transformations and attention. These methods exploit rich contextual semantics and support adaptive information exchange, but aligning heterogeneous representations often requires specialized components that increase architectural complexity and computational overhead. Temporal instance aggregation similarly relies on query propagation, feature compensation, and attention [3], [8], [9] to accommodate object motion and viewpoint changes [10]. In contrast, boxcentric association methods [11], [12] construct costs from structural cues, such as geometry and category, and solve discrete assignments for state updates or candidate fusion. Matching in a shared geometric space enables efficient and interpretable association without elaborate feature alignment. However, hard assignments restrict fusion to selected correspondences, while structural costs alone cannot exploit contextual semantics to resolve ambiguous matches. These complementary strengths motivate a matching and fusion mechanism that combines the efficiency of structural association with the adaptability of learned semantic interaction.

To this end, we propose MatchFusion, a learnable instance matching and fusion module for spatio-temporal multimodal autonomous driving. Given sparse instances comprising structured object states and query embeddings, Match-Fusion initializes coarse correspondences by quantifying geometric similarity and category consistency. A lightweight semantic compatibility module then selectively refines structurally plausible associations using instance embeddings. This coarse-to-fine process produces a soft matchmap that preserves many-to-many affinities and guides a common residual aggregation operator. Spatial and temporal interaction instantiate this unified matching–fusion formulation with setting-specific geometric priors. For LiDAR–camera interaction, 3D boxes from both modalities are projected onto image planes, and geometric similarities based on relative positions and sizes are aggregated across jointly visible views. For past–current matching, past instances are aligned through object-motion compensation and ego-motion transformation, and geometric affinities are computed in BEV space. Both settings use the same semantic refinement and residual aggregation mechanisms. The aggregation operator is applied bidirectionally for cross-modal exchange and unidirectionally to incorporate past evidence into current candidates. Temporal interaction can operate independently on upstream instances or incorporate spatially fused representations.

Operating solely on instance representations, MatchFusion is decoupled from front-end feature extraction and provides enhanced queries for downstream perception and planning. Experiments on nuScenes [13] demonstrate consistent perception gains across diverse front-end configurations. Compared with SparseFusion [7], the MatchFusion-equipped system achieves higher perception accuracy while reducing FLOPs by 55.3% and GPU memory usage by 39.3%, with the matching–fusion module accounting for only 3.7% of total perception latency. Integrating temporal MatchFusion into SparseDrive [3] further improves perception within an end-to-end driving framework without additional supervi sion. Our contributions are:

• We propose an explicit–implicit instance matching mechanism that combines structural affinity initialization with selective semantic refinement to establish efficient and adaptive soft correspondences.

• We develop a unified matching–fusion formulation for spatial LiDAR–camera and temporal past–current interaction, combining setting-specific geometric priors with common semantic refinement and a single residual aggregation operator.

• We demonstrate consistent perception gains across diverse front-ends, improved computational and memory efficiency, and compatibility with E2EAD frameworks.

Spatio-Temporal Multimodal 3D Perception. Integrating complementary observations across sensors and time supports reliable scene understanding in autonomous driving. For spatial fusion, LiDAR provides accurate geometry, while cameras capture rich appearance and semantics. Pointlevel methods [14], [15] augment point clouds with imagederived information. BEVFusion [16] integrates modalityspecific features in a shared BEV space. SparseFusion [7] and MV2DFusion [5] adopt sparse instance representations for object-level cross-modal interaction. For temporal fusion, past information provides persistent context beyond individual frames. StreamPETR [9] propagates object queries with motion-aware feature compensation, Sparse4D [8] combines anchor-guided feature aggregation with recurrent instance propagation, and StreamCMT [6] incorporates temporal query interaction into multimodal perception. Spatiotemporal information further contributes to E2EAD by enriching the representations used for perception and planning. LEAD [17] highlights the value of multimodal information for driving. BridgeAD [18] uses sparse instance representations to connect temporal perception with motion prediction and planning. These developments motivate a common interaction mechanism for integrating cross-modal and historical evidence. MatchFusion provides a unified matching–fusion formulation with geometric priors adapted to spatial Li-DAR–camera and temporal past–current interaction.

Instance Matching and Fusion. Instance-level interaction requires establishing correspondences and aggregating complementary information. Implicit attention-based approaches infer association weights from instance features and positional representations to guide feature aggregation. Sparse-Fusion [7] combines modality-specific transformations with attention-based fusion. MV2DFusion [5] employs modalityspecific query generators and uncertainty-aware positional representations for sparse interaction. Temporal methods similarly integrate past information through query propagation, feature compensation, and attention [3], [9]. These methods exploit contextual semantics and support adaptive information exchange, but specialized representation processing can increase architectural complexity, computation, and memory consumption. Explicit box-centric approaches [11], [12] instead construct association costs from geometric, categorical, or motion cues and solve discrete assignments for candidate fusion or state updates. Matching compact structured states in a shared geometric space enables efficient and interpretable association without elaborate feature alignment. However, hard assignments restrict information exchange to selected correspondences, and structural costs alone cannot exploit contextual evidence in instance embeddings to resolve ambiguous matches. MatchFusion combines the efficiency of structural association with the adaptability of semantic interaction through structural affinity initialization, selective semantic refinement, and soft matchmap-guided residual aggregation.

![](images/a79ab9a7f32b3e4249c8a6570dc99c6d1558ed151828aa3f10b970a0cbf59b65.jpg)  
Fig. 2: Architecture of MatchFusion for spatio-temporal interaction through unified instance matching and fusion.

## III. MATCHFUSION

## A. Overview and Instance Formulation

As illustrated in Fig. 2, MatchFusion is an instance-level matching and fusion module for spatio-temporal multimodal interaction. It operates on sparse instance representations from upstream query-based perception models, decoupling matching and fusion from front-end feature extraction. Each input source provides structured object states $\boldsymbol { B } = \{ B _ { i } \} _ { i = 1 } ^ { N }$ and corresponding query embeddings $\mathcal { Q } = \{ Q _ { i } \} _ { i = 1 } ^ { N } ,$ forming an instance set $\mathcal { T } = \{ I _ { i } = ( B _ { i } , Q _ { i } ) \} _ { i = 1 } ^ { N }$ , where N denotes the number of instances. Each state B<sub>i</sub> comprises the 3D center $( x _ { i } , y _ { i } , z _ { i } )$ , dimensions $( w _ { i } , l _ { i } , h _ { i } )$ , yaw angle $\theta _ { i }$ , and predicted category $c _ { i } .$ The query embedding $\bar { Q } _ { i } ~ \in ~ \mathbb { R } ^ { D }$ captures contextual semantics associated with the instance.

Given two instance sets, MatchFusion initializes coarse correspondences from structural cues and selectively refines them using query semantics. The resulting soft matchmap guides residual feature aggregation to produce enhanced instance representations for downstream perception and planning. Sec. III-B introduces this explicit–implicit matching and fusion mechanism. Sec. III-C instantiates it for spatial Li-DAR–camera and temporal past–current interaction. Match-Fusion separates instance interaction from task-specific prediction. Adapting it to spatial or temporal interaction requires specifying the geometric prior and aggregation direction, while the enhanced queries are optimized through the prediction objectives of the corresponding system.

## B. Explicit–Implicit Matching-Fusion

Structured object states provide compact and interpretable evidence for instance association, while query embeddings contain contextual information that helps resolve ambiguous candidates. MatchFusion combines these complementary cues through a coarse-to-fine design: structural initialization establishes matching priors, and semantic refinement adjusts promising associations. Given two instance sets, $\scriptstyle { \mathcal { T } } ^ { A } = \{ I _ { i } ^ { A } =$ $\stackrel { \cdot } { ( { \cal B } _ { i } ^ { A } , { \cal Q } _ { i } ^ { A } ) } \} _ { i = 1 } ^ { N ^ { A } }$ and $\mathcal { T } ^ { B } = \{ I _ { j } ^ { B } = ( B _ { j } ^ { B } , Q _ { j } ^ { B } ) \} _ { j = 1 } ^ { N ^ { B } }$ , the module performs explicit structural initialization, implicit semantic refinement, and matchmap-guided residual fusion.

Explicit Structural Initialization. Geometric similarity provides an efficient starting point for association by comparing compact object states in a shared geometric space. Category consistency is enforced as a hard constraint that excludes pairs with different predicted categories. For each instance pair $I _ { i } ^ { A }$ and $I _ { j } ^ { B }$ , the initial affinity is:

$$
M _ { i j } = G _ { i j } + H _ { i j } ,\tag{1}
$$

where $G _ { i j }$ represents geometric similarity and $H _ { i j }$ is an additive category-consistency mask. Geometric similarity is derived from relative object positions and sizes, with settingspecific formulations for spatial and temporal interaction detailed in Sec. III-C. Motivated by category-consistent association in prior 3D perception methods [19], we define the mask as:

$$
H _ { i j } = \left\{ \begin{array} { l l } { 0 , } & { c _ { i } ^ { A } = c _ { j } ^ { B } , } \\ { - \infty , } & { c _ { i } ^ { A } \neq c _ { j } ^ { B } , } \end{array} \right.\tag{2}
$$

where $c _ { i } ^ { A }$ and $c _ { j } ^ { B }$ denote the predicted categories. Categoryconsistent pairs retain their geometric affinities, whereas category-inconsistent pairs are excluded from subsequent semantic refinement and matchmap-guided aggregation. These cues yield the initial affinity matrix $M \in \mathbb { R } ^ { N ^ { A } \times N ^ { B } }$ without elaborate transformations of query embeddings.

Implicit Semantic Refinement. Structural similarity can leave ambiguities among geometrically similar candidates. Query embeddings provide contextual evidence to refine these associations. We condition semantic refinement on the initial masked affinity, concentrating learned pairwise corrections on category-consistent pairs.

$$
\tilde { M } _ { i j } = \left\{ \begin{array} { c } { { M _ { i j } , \quad M _ { i j } \leq \epsilon , } } \\ { { M _ { i j } + R _ { i j } , \quad M _ { i j } > \epsilon , } } \end{array} \right.\tag{3}
$$

Here, ϵ is a finite threshold controlling semantic refinement. Category-inconsistent pairs receive no semantic correction. For selected pairs, the semantic correction is computed as:

$$
R _ { i j } = \mathrm { M L P } \left( \psi _ { A } ( Q _ { i } ^ { A } ) + \psi _ { B } ( Q _ { j } ^ { B } ) \right) ,\tag{4}
$$

where $\psi _ { A }$ and $\psi _ { B }$ are branch-specific linear mappings. The multi-layer perceptron (MLP) maps their summed embeddings to a scalar correction. An element-wise sigmoid

function converts the resulting affinities into matching scores:

$$
\hat { M } _ { i j } = \mathrm { S i g m o i d } ( \tilde { M } _ { i j } ) .\tag{5}
$$

The resulting soft matchmap $\hat { M }$ preserves graded manyto-many associations, allowing contextual semantics to refine promising correspondences without imposing discrete assignments [20].

Matchmap-Guided Residual Fusion. Ambiguous observations can support multiple plausible correspondences, making early discrete assignment restrictive. The soft matchmap M<sup>ˆ</sup> instead guides many-to-many feature aggregation among category-consistent instances, incorporating complementary evidence through residual feature updates. For information flow from $\mathcal { T } ^ { B }$ to $\mathcal { T } ^ { A }$ , we define a common residual aggregation operator:

$$
{ \mathcal { F } } ( Q ^ { A } , Q ^ { B } , { \hat { M } } ) = Q ^ { A } + { \hat { M } } \otimes \mathrm { M L P } _ { B \to A } ( Q ^ { B } ) ,\tag{6}
$$

where $\otimes$ denotes matrix multiplication and $\mathrm { M L P } _ { B  A }$ maps source embeddings into the receiving feature space.

The same operator supports both spatial and temporal fusion. Spatial interaction applies $\mathcal { F }$ in both directions using M<sup>ˆ</sup> and $\hat { M } ^ { \top }$ , with both updates operating on the original input queries. Temporal interaction applies $\mathcal { F }$ once to aggregate aligned historical features into augmented current candidates. The enhanced queries are subsequently processed by the corresponding prediction heads, separating the common aggregation mechanism from task-specific prediction.

## C. Spatio-Temporal Instantiation

MatchFusion supports spatio-temporal interaction through a common explicit–implicit matching and residual aggregation principle. Setting-specific geometric priors and candidate selection establish plausible correspondences, while the aggregation direction determines how complementary evidence updates instance representations.

Spatial LiDAR–Camera Interaction. Camera depth errors can separate corresponding instances in 3D while preserving their projected overlap in relevant views [21]. We therefore initialize cross-modal affinities using multi-view image-plane geometry to reduce sensitivity to depth discrep ancies. Since distinct objects can also exhibit similar projections, these affinities provide coarse priors for subsequent semantic refinement. For each camera $k = 1 , \ldots , N _ { \mathrm { v i e w } }$ , we project 3D boxes from both modalities onto the image plane:

$$
\begin{array} { r } { b _ { i } ^ { C , k }  \mathrm { P r o j } ( B _ { i } ^ { C } , P ^ { k } ) , b _ { j } ^ { L , k }  \mathrm { P r o j } ( B _ { j } ^ { L } , P ^ { k } ) , } \end{array}\tag{7}
$$

where Proj(·) denotes 3D-to-2D projection and $P ^ { k }$ is the corresponding projection matrix. Each projected box is represented by its center and extent, $\boldsymbol { b } = ( b x , b y , b w , b h )$ . Relative geometry is encoded using normalized positional offsets and logarithmic size ratios:

$$
\begin{array} { r } { g _ { i j } ^ { k } = [ \log ( \frac { | b x _ { i } ^ { C , k } - b x _ { j } ^ { L , k } | } { b w _ { i } ^ { C , k } } + 1 ) , \log ( \frac { | b y _ { i } ^ { C , k } - b y _ { j } ^ { L , k } | } { b h _ { i } ^ { C , k } } + 1 ) , } \end{array}
$$

$$
\begin{array} { r } { \log \left( \frac { b w _ { i } ^ { C , k } } { b w _ { j } ^ { L , k } } \right) , \log \left( \frac { b h _ { i } ^ { C , k } } { b h _ { j } ^ { L , k } } \right) \bigg ] . } \end{array}\tag{8}
$$

Sinusoidal positional encoding followed by an MLP maps this descriptor to a view-specific geometric affinity:

$$
G _ { i j } ^ { k } = \mathrm { M L P } \big ( \mathrm { F l a t t e n } ( \mathrm { P o s E m b } ( g _ { i j } ^ { k } ) ) \big ) ,\tag{9}
$$

where PosEmb(·) denotes sinusoidal positional encoding. For pairs with at least one jointly visible view, we aggregate affinities through visibility-aware averaging:

$$
G _ { i j } = \frac { \sum _ { k = 1 } ^ { N _ { \mathrm { v i e w } } } O _ { i j } ^ { k } \cdot G _ { i j } ^ { k } } { \sum _ { k = 1 } ^ { N _ { \mathrm { v i e w } } } O _ { i j } ^ { k } } ,\tag{10}
$$

where $O _ { i j } ^ { k } = 1$ if both projected boxes are visible in view k, and zero otherwise. The aggregated affinity is combined with category consistency through Eq. 1 and refined using query semantics as described in Sec. III-B. The resulting matchmap M<sup>ˆ</sup> guides bidirectional application of the residual aggregation operator in Eq. 6:

$$
\begin{array} { r l } & { \hat { Q } ^ { C } = \mathcal { F } ( Q ^ { C } , Q ^ { L } , \hat { M } ) , } \\ & { \hat { Q } ^ { L } = \mathcal { F } ( Q ^ { L } , Q ^ { C } , \hat { M } ^ { \top } ) . } \end{array}\tag{11}
$$

Both updates use the original input queries. The enhanced queries are concatenated along the instance dimension and passed to an added refinement decoder. Its regression branch predicts box offsets:

$$
\Delta { \cal B } = \mathrm { D e c o d e r } \left( \mathrm { C o n c a t } \left( \hat { \mathcal { Q } } ^ { C } , \hat { \mathcal { Q } } ^ { L } \right) \right) ,\tag{12}
$$

which update the corresponding input boxes:

$$
\begin{array} { r } { \hat { \boldsymbol { B } } = \operatorname { C o n c a t } \left( \boldsymbol { B } ^ { C } , \boldsymbol { B } ^ { L } \right) + \Delta \boldsymbol { B } . } \end{array}\tag{13}
$$

Retaining candidates from both modalities preserves complementary object hypotheses and semantic evidence. The resulting queries and updated states form the spatially fused instance set ${ \hat { \mathcal { I } } } .$ Classification and box-regression objectives supervise the added decoder.

Temporal Past–Current Interaction. Historical instances provide persistent object information under weak or incomplete current observations. Object-motion compensation and ego-motion transformation align historical states $\mathcal { T } ^ { t - 1 }$ to the current frame, yielding $\bar { \mathcal { T } } ^ { t - 1 , t }$ . These aligned instances are combined with the current set ${ \mathcal { T } } ^ { t }$ to form an augmented candidate set ${ \bar { \mathcal { T } } } ,$ preserving historical hypotheses within the available query budget. Notably, current instances can originate from an upstream perception branch or from spatially fused set ${ \hat { \mathcal { I } } } ,$ allowing temporal interaction to operate independently or alongside spatial fusion.

Temporal matching associates augmented current candidates $\dot { \bar { \tau } }$ with aligned historical instances $\mathcal { T } ^ { t - 1 , t }$ . After motion compensation, relative BEV positions and sizes provide geometric priors for association. We therefore construct geometric affinities directly in the BEV space, while semantic refinement helps resolve ambiguities caused by residual localization and motion estimation errors. For augmented candidate ${ \bar { I } } _ { i }$ and aligned historical instance $I _ { j } ^ { t - 1 , t }$ , we define:

$$
\begin{array} { r } { g _ { i j } = [ \log ( \frac { | \bar { x } _ { i } - x _ { j } ^ { t - 1 , t } | } { \bar { w } _ { i } } + 1 ) , \log ( \frac { | \bar { y } _ { i } - y _ { j } ^ { t - 1 , t } | } { \bar { l } _ { i } } + 1 ) , } \end{array}
$$

$$
\begin{array} { r } { \log \left( \frac { \bar { w } _ { i } } { w _ { j } ^ { t - 1 , t } } \right) , \log \left( \frac { \bar { l } _ { i } } { l _ { j } ^ { t - 1 , t } } \right) \bigg ] . } \end{array}\tag{14}
$$

As in spatial interaction, sinusoidal positional encoding followed by an MLP maps the descriptor to a scalar affinity:

$$
G _ { i j } = \mathrm { M L P } \big ( \mathrm { F l a t t e n } ( \mathrm { P o s E m b } ( g _ { i j } ) ) \big ) .\tag{15}
$$

Category consistency and semantic refinement then produce the temporal matchmap $\bar { M }$ . The residual aggregation operator incorporates historical evidence into current queries:

$$
\bar { Q } ^ { \prime } = \mathcal { F } ( \bar { Q } , Q ^ { t - 1 , t } , \bar { M } ) .\tag{16}
$$

This unidirectional update enriches current candidates with past context. Enhanced queries $\bar { Q } ^ { \prime }$ continue through the host decoder to refine the associated object states, yielding $\bar { \mathcal { T } } ^ { \prime }$

## IV. EXPERIMENTS

## A. Datasets and Implementation Details

Datasets and Metrics. We evaluate MatchFusion on nuScenes for perception and planning. For nuScenes detection, we report the nuScenes detection score (NDS) and mean average precision (mAP). Tracking is evaluated using average multi-object tracking accuracy (AMOTA) and multiobject tracking accuracy (MOTA). Planning is assessed using trajectory L2 error and collision rate (CR). Computational efficiency is measured by FLOPs, GPU memory consumption, and inference latency.

Implementation Details. We denote the spatial, temporal, and combined spatio-temporal configurations as MatchFusion-S, MatchFusion-T, and MatchFusion-ST, respectively. MatchFusion-S enables LiDAR–camera interaction, while MatchFusion-T aggregates historical evidence into current instance representations. Category consistency is enforced across all settings. Each component of the 4- D relative geometric descriptor is encoded using a 256-D sinusoidal embedding, yielding a flattened 1024-D feature that a single-layer MLP maps to geometric affinity. Semantic refinement uses a single-layer MLP. The residualfusion and modality-projection MLPs each comprise two layers. All hidden representations are 256-D. For perception experiments, we freeze the pretrained front-ends to isolate the contribution of MatchFusion and evaluate its ability to enhance existing instance representations without updating the front-ends. We evaluate multiple front-end configurations and train MatchFusion for 16 epochs with a batch size of 32 on four NVIDIA RTX 3090 GPUs, requiring approximately 12 hours. We use AdamW [22] with an initial learning rate of $1 . 0 \times 1 0 ^ { - 3 }$ and cosine annealing schedule. All perception results are reported on the nuScenes validation set without testtime augmentation or model ensembling. In MatchFusion-S, enhanced camera and LiDAR queries are concatenated and processed by an added refinement decoder, supervised using the classification and box-regression objectives adopted by Sparse4Dv3 [8]. MatchFusion-T uses existing downstream objectives without additional losses. Tracking follows the query propagation and update mechanism of Sparse4Dv3 without explicit association module. For E2EAD, we integrate MatchFusion-T into SparseDrive and jointly train the entire system from scratch, following the baseline training protocol. This setting evaluates its compatibility with endto-end optimization under the original perception, motion, and planning objectives. MatchFusion-T operates directly on the instance representations of framework without spatial Li-DAR–camera fusion or additional supervision. The baseline and MatchFusion-equipped systems follow matched training and evaluation protocols.

## B. Comparative Evaluation

Perception Gains Across Front-Ends. The instance-level formulation of MatchFusion separates matching and fusion from front-end feature extraction. To validate the compatibility of MatchFusion with diverse instance representations, we retain TransFusion-L as LiDAR front-end and pair it with DETR3D, StreamPETR, Sparse4Dv3, and SimPB. As shown in Table. II, MatchFusion-ST consistently improves over the stronger unimodal front-end, yielding gains of 1.9–3.1% NDS and 4.3–5.5% mAP. With the weak DETR3D camera detector, fusion improves NDS from 70.1 to 72.0 and mAP from 65.1 to 69.4. This demonstrates that a less accurate front-end can still provide complementary evidence through effective instance matching and aggregation. These consistent improvements support the reuse of MatchFusion across different camera architectures through a common instance interface. For the Sparse4Dv3-based configuration, MatchFusion-ST also improves AMOTA and MOTA through the existing query propagation mechanism, extending its benefits from detection to tracking.

Accuracy–Efficiency Trade-off. To demonstrate the efficiency of MatchFusion for integrated perception, we evaluate system-level computational cost and module-level inference latency. With Sparse4Dv3 and TransFusion-L as front-ends, MatchFusion-ST-equipped system achieves 73.0 NDS and 62.3 AMOTA with 254.2 GFLOPs and 3.7 GB memory. As shown in Table III, it achieves higher NDS than SparseFusion while reducing FLOPs by 55.3% and memory consumption by 39.3%. These results demonstrate that combining compact structural priors with lightweight semantic refinement enables effective instance interaction without introducing substantial computational overhead. Furthermore, under the same inference setting, the camera front-end, LiDAR frontend, and MatchFusion-ST require 47.3, 283.3, and 12.9 ms, respectively. Including tracking operations, MatchFusion-ST accounts for only 3.7% of the total latency, supporting integrated detection and tracking with limited additional runtime. This further demonstrates that the proposed instance interaction mechanism enables accurate integrated perception with only marginal runtime overhead.

Integration into End-to-End Driving. To evaluate the independent applicability of temporal interaction within an end-to-end driving framework, we integrate MatchFusion-T into SparseDrive using its original training objectives without additional supervision. As shown in Table I, MatchFusion-T improves NDS and mAP by 0.8 and 0.5 points, respectively, and AMOTA and MOTA by 2.7 and 1.7 points. Planning performance remains comparable, with CR decreasing from 0.123% to 0.119% and L2 error increasing slightly from 0.63 to 0.64 m. These results support the applicability of temporal MatchFusion to end-to-end driving, with the clearest gains observed in perception.

TABLE I: Comparison on nuScenes for the E2EAD task. †: Our reproduced results by official code.
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Method</td><td colspan="2">Detection</td><td colspan="2">Tracking</td><td colspan="2">Planning</td></tr><tr><td>NDS↑</td><td>mAP↑</td><td>AMOTA↑</td><td>MOTA↑</td><td>L2 (m)↓</td><td>CR (%)↓</td></tr><tr><td rowspan="2">nuScenes</td><td>SparseDrive†</td><td>52.2</td><td>41.2</td><td>36.9</td><td>34.2</td><td>0.63</td><td>0.123</td></tr><tr><td>SparseDrive + MatchFusion-T</td><td>53.0 (+0.8) 41.7 (+0.5)</td><td></td><td></td><td>39.6 (+2.7) 35.9 (+1.7)</td><td>0.64</td><td>0.119</td></tr></table>

TABLE II: Perception enhancement with MatchFusion-ST across front-end configurations on the nuScenes validation set. LiDAR denotes the TransFusion-L front-end.
<table><tr><td>Camera</td><td>LiDAR</td><td>MatchFusion-ST</td><td>NDS↑</td><td>mAP↑</td></tr><tr><td></td><td>√</td><td>一</td><td>70.1</td><td>65.1</td></tr><tr><td rowspan="2">DETR3D [23]</td><td>一</td><td>1</td><td>42.2</td><td>34.7</td></tr><tr><td>√</td><td>√</td><td>72.0 (+1.9)69.4 (+4.3)</td><td></td></tr><tr><td rowspan="2">StreamPETR [9]</td><td></td><td></td><td>53.7</td><td>43.2</td></tr><tr><td>V</td><td>√</td><td>72.7 (+2.6)69.9 (+4.8)</td><td></td></tr><tr><td rowspan="2">Sparse4Dv3 [8]</td><td></td><td>一</td><td>56.1</td><td>46.9</td></tr><tr><td>√</td><td>√</td><td>73.0 (+2.9)70.4 (+5.3)</td><td></td></tr><tr><td rowspan="2">SimPB [24]</td><td></td><td>一</td><td>59.0</td><td>48.7</td></tr><tr><td> $\checkmark$ </td><td>√</td><td>73.2 (+3.1)70.6 (+5.5)</td><td></td></tr></table>

TABLE III: Detection accuracy and computational cost on the nuScenes validation set. All methods use VoxelNet [25] and ResNet-50 [26] as the LiDAR and camera backbones, respectively. FLOPs and GPU memory are measured for the complete system with a batch size of one. †: Our reproduced results by official code.
<table><tr><td rowspan="3">Method</td><td colspan="2">Accuracy</td><td colspan="2">Efficiency</td></tr><tr><td>NDS↑</td><td>mAP↑</td><td>FLOPs (G)↓</td><td>Mem. (GB)↓</td></tr><tr><td>TransFusion† [27]</td><td>71.3</td><td>67.5</td><td>449.8</td><td>10.5</td></tr><tr><td>DeepInteraction† [28]</td><td>72.6</td><td>69.9</td><td>513.1</td><td>22.1</td></tr><tr><td>SparseFusion [7]</td><td>72.8</td><td>70.4</td><td>569.1</td><td>6.1</td></tr><tr><td>DeepInteraction++ [29]</td><td>72.9</td><td>70.1</td><td></td><td>11.4</td></tr><tr><td>StreamCMT [6]</td><td>71.8</td><td>69.0</td><td></td><td>一</td></tr><tr><td>MatchFusion-ST (Ours)</td><td>73.0</td><td>70.4</td><td>254.2</td><td>3.7</td></tr></table>

## C. Ablation Studies

Unless otherwise specified, ablation studies use Sparse4Dv3 with ResNet-50 and TransFusion-L with VoxelNet as the camera and LiDAR front-ends, respectively. We examine the contributions of spatial and temporal interaction, the design of the matching–fusion mechanism, and robustness to spatial and temporal misalignment.

Spatio-Temporal Fusion. To validate the complementary contributions of cross-modal and historical evidence, we compare MatchFusion-S and MatchFusion-ST in Table. IV. MatchFusion-S improves NDS from 70.1 for the LiDARonly baseline to 72.5. Adding temporal interaction further increases NDS to 73.0, while improving AMOTA from 49.0 to 62.3 and MOTA from 43.6 to 60.8. These results demonstrate that temporal aggregation complements spatial fusion, benefiting broad 3D perception tasks.

TABLE IV: Contributions of spatial and temporal interaction.
<table><tr><td colspan="2">Spatial</td><td rowspan="2">Temporal</td><td colspan="2">Detection</td><td colspan="2">Tracking</td></tr><tr><td>Camera</td><td>LiDAR</td><td>NDS↑</td><td>mAP↑</td><td>AMOTA↑</td><td>MOTA↑</td></tr><tr><td>一</td><td>√</td><td></td><td>70.1</td><td>65.1</td><td></td><td></td></tr><tr><td>√</td><td>1</td><td></td><td>56.1</td><td>46.9</td><td>49.0</td><td>43.6</td></tr><tr><td>√</td><td>√</td><td></td><td>72.5</td><td>69.6</td><td>49.0</td><td>43.6</td></tr><tr><td>√</td><td>√</td><td>√</td><td>73.0</td><td>70.4</td><td>62.3</td><td>60.8</td></tr></table>

TABLE V: Ablation of matchmap construction and fusion strategies on MatchFusion-S.
<table><tr><td rowspan="2">Setting</td><td rowspan="2">Variant</td><td colspan="2">Detection</td></tr><tr><td>| NDS↑</td><td>mAP↑</td></tr><tr><td colspan="4">Matching</td></tr><tr><td>3D</td><td>Structural Structural + Semantic</td><td>72.2 72.4</td><td>69.1 69.2</td></tr><tr><td>2D</td><td>Structural + Semantic</td><td>72.5</td><td></td></tr><tr><td></td><td></td><td></td><td>69.6</td></tr><tr><td colspan="4">Fusion</td></tr><tr><td rowspan="3">Implicit</td><td>Concat</td><td>69.4</td><td>64.8</td></tr><tr><td>Self-Attention</td><td>71.3</td><td>67.3</td></tr><tr><td>Cross-Attention</td><td>71.8</td><td>67.9</td></tr><tr><td>Explicit-Implicit</td><td>MatchFusion-S</td><td>72.5</td><td>69.6</td></tr></table>

The independent applicability of MatchFusion-T is evaluated through its integration into SparseDrive without spatial LiDAR–camera fusion. As shown in Table. I, MatchFusion-T improves detection and tracking and reduces planning CR. These results support the utility of MatchFusion for shared instance representations beyond standalone perception.

Matching–Fusion Construction. Table. V evaluates matchmap construction and feature aggregation within MatchFusion-S. With direct 3D geometry, structural matching achieves 72.2 NDS and 69.1 mAP. Adding semantic refinement further improves detection performance, supporting the contribution of contextual query information to correspondence estimation. Replacing direct 3D geometry with multi-view image-plane geometry further improves performance to 72.5 NDS and 69.6 mAP. These results favor combining projection-based structural priors with semantic refinement for cross-modal matching. The fusion comparison in Table V evaluates alternative instance aggregation strategies. Direct concatenation achieves 69.4 NDS and 64.8 mAP, below the LiDAR-only baseline. Self-attention and cross-attention improve NDS to 71.3 and 71.8, respectively, demonstrating the benefit of information exchange between instances. MatchFusion-S further improves over cross-attention by 0.7 NDS and 1.7 mAP points, supporting residual aggregation guided by structurally initialized and semantically refined correspondences. Table VI evaluates the effect of category consistency. Enforcing category consistency improves NDS from 72.1 to 73.0 and mAP from 69.0 to 70.4. These gains indicate that excluding crosscategory associations provides more reliable correspondences for subsequent semantic refinement and feature aggregation.

TABLE VI: Ablation of the category consistency mechanism in MatchFusion-ST.
<table><tr><td>Category Consistency</td><td>NDS↑</td><td>mAP↑</td></tr><tr><td></td><td>72.1</td><td>69.0</td></tr><tr><td>× &gt;</td><td>73.0</td><td>70.4</td></tr></table>

TABLE VII: Training efficiency of MatchFusion-S matchmap variants in the 2D setting. Time and GPU memory are normalized to the structural baseline.
<table><tr><td>Matchmap Construction</td><td>Mem.↓</td><td>Time↓</td></tr><tr><td>Structural</td><td>1.0×</td><td>1.0×</td></tr><tr><td>Structural + Semantic</td><td>1.3×</td><td>1.7×</td></tr><tr><td>Semantic</td><td>2.2×</td><td>1.9×</td></tr></table>

TABLE VIII: Effect of front-end optimization on the training efficiency and detection performance of MatchFusion-ST on the nuScenes val set. Training time and GPU memory usage are normalized to the configuration with frozen front-ends.
<table><tr><td>Front-ends</td><td>Mem.↓</td><td>Time↓</td><td>NDS↑</td><td>mAP↑</td></tr><tr><td>Unfrozen</td><td>2.1×</td><td>5.5×</td><td>71.7</td><td>68.5</td></tr><tr><td>Frozen</td><td>1.0×</td><td>1.0×</td><td>73.0</td><td>70.4</td></tr></table>

Efficiency of Instance-Level Interaction. Tables. VII and VIII complement the system-level accuracy–efficiency comparison by evaluating matchmap construction and frontend optimization. For MatchFusion-S, structural initialization with selective semantic refinement requires 1.7× training time and 1.3× memory relative to structural-only matching, versus 1.9× and 2.2× for semantic-only matching. For perception, MatchFusion-ST with frozen front-ends achieves 73.0 NDS and 70.4 mAP, outperforming the unfrozen configuration while avoiding its 2.1× memory and 5.5× training time requirements. These results demonstrate the training efficiency of structural guidance and front-end decoupling, while the jointly trained SparseDrive results in Table I further establish compatibility with end-to-end optimization.

Spatio-Temporal Fusion Robustness. We evaluate the robustness of MatchFusion-ST by perturbing spatial and temporal priors at inference, as summarized in Table. IX. For spatial interaction, translation perturbations of 5–20 cm to the projection extrinsics leave NDS unchanged at the reported precision. Rotation perturbations of 1°, 2°, and 4° reduce NDS from 73.0 to 71.9, 71.8, and 71.7, respectively. Even at 4°, the system retains 71.7 NDS and 68.0 mAP, exceeding the LiDAR-only baseline by 1.6 and 2.9 points. These results demonstrate that cross-modal fusion remains beneficial under the tested calibration errors, with greater sensitivity to rotational perturbations. For temporal interaction, we independently perturb the object motion, ego motion, and inter-frame interval used for historical state alignment. As shown in Table. IX, MatchFusion-ST maintains at least 72.3 NDS, 69.2 mAP, 61.2 AMOTA, and 60.1 MOTA across the tested perturbations. Detection performance remains above both unimodal baselines, while tracking performance consistently exceeds the camera-only baseline. These results demonstrate that the benefits of temporal aggregation persist under imperfect motion compensation.

TABLE IX: Robustness to spatio-temporal perturbations on MatchFusion-ST.
<table><tr><td rowspan="2">Perturbation</td><td rowspan="2">Magnitude</td><td colspan="2">Detection</td><td colspan="2">Tracking</td></tr><tr><td></td><td>|NDS↑ mAP↑</td><td>AMOTA↑</td><td>MOTA↑</td></tr><tr><td colspan="7">Baseline</td></tr><tr><td>MatchFusion-ST</td><td rowspan="2"></td><td>73.0</td><td>70.4</td><td>62.3</td><td>60.8</td></tr><tr><td>LiDAR-only detector</td><td>70.1</td><td>65.1</td><td></td><td></td></tr><tr><td>Camera-only detector</td><td></td><td>56.1</td><td>46.9</td><td>49.0</td><td>43.6</td></tr><tr><td colspan="6">Spatial</td></tr><tr><td rowspan="3">Rotation (°)</td><td>1</td><td>71.9</td><td>68.4</td><td>59.2</td><td>56.9</td></tr><tr><td>2</td><td>71.8</td><td>68.3</td><td>59.0</td><td>56.7</td></tr><tr><td>4</td><td>71.7</td><td>68.0</td><td>58.8</td><td>56.5</td></tr><tr><td rowspan="3">Translation (cm)</td><td>5</td><td>73.0</td><td>70.4</td><td>62.5</td><td>60.6</td></tr><tr><td>10</td><td>73.0</td><td>70.4</td><td>62.5</td><td>60.4</td></tr><tr><td>20</td><td>73.0</td><td>70.4</td><td>62.6</td><td>60.8</td></tr><tr><td colspan="6">Temporal</td></tr><tr><td rowspan="4">∆t (s)</td><td>0.1</td><td>72.9</td><td>70.3</td><td>62.4</td><td>60.2</td></tr><tr><td>0.2</td><td>73.0</td><td>70.4</td><td>62.5</td><td>60.6</td></tr><tr><td>0.3</td><td>72.9</td><td>70.3</td><td>62.3</td><td>60.6</td></tr><tr><td>0.2</td><td>72.9</td><td>70.3</td><td>62.3</td><td>60.4</td></tr><tr><td rowspan="4">Ego Motion (m)</td><td>0.5</td><td>72.7</td><td>69.9</td><td>62.0</td><td>60.2</td></tr><tr><td>1.0</td><td>72.3</td><td>69.2</td><td>61.3</td><td>60.3</td></tr><tr><td>0.5</td><td>72.8</td><td>70.2</td><td>62.1</td><td>60.6</td></tr><tr><td>1.0</td><td>72.6</td><td>69.9</td><td>61.6</td><td>60.1</td></tr><tr><td rowspan="2"></td><td>2.0</td><td>72.3</td><td>69.4</td><td>61.2</td><td>60.2</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr></table>

![](images/5feb390dd24b86bc437168b8d68b0643bd6a67a831d388c988c2ba88948d7a0f.jpg)  
w/ MatchFusion-S

![](images/6c12a140dfd8f8e07408379bdd93b6b8f77803463566c7af7b977be412b4eefc.jpg)

![](images/7dc66cff153e0118b9b6ed2e83b1e8c5f18654e100d6c0e6556df32d90cdbe8e.jpg)

GT  
![](images/5edf03af827856ce58b76ab11f9406be3cb1aa6142d36996f9298c415e51e027.jpg)  
Fig. 3: Qualitative BEV detection results of MatchFusion-S on the nuScenes validation set. Results are shown for the camera front-end Sparse4Dv3, the LiDAR front-end TransFusion-L, and the MatchFusion-equipped system.

## D. Qualitative Results

Fig. 3 illustrates the complementary advantages of multimodal fusion. MatchFusion corrects individual front-end errors and recovers objects missed by a single modality, illustrating the benefit of aggregating complementary evidence through the instance interface. Fig. 4 highlights the complementary roles of structural initialization and semantic refinement. Structural priors identify plausible associations, while query semantics distinguish corresponding instances from geometrically ambiguous candidates. The refinement gate concentrates semantic corrections on structurally plausible pairs for efficient correspondence refinement.

![](images/4a333cbeb25ee7517950d77d127bab71f43a7df0aa1cda8e1235847bc7e27374.jpg)  
Fig. 4: Qualitative visualization of explicit–implicit crossmodal matching on the nuScenes validation set. Box thickness represents detection confidence, and line thickness represents matching score.

## V. CONCLUSIONS

We present MatchFusion, a learnable instance matching and fusion module for spatio-temporal multimodal autonomous driving. By combining explicit geometric and categorical priors with selective semantic refinement, Match-Fusion establishes soft correspondences for residual feature aggregation. A common formulation supports spatial Li-DAR–camera and temporal past–current interaction while remaining decoupled from front-end feature extraction. Experiments on nuScenes demonstrate consistent perception gains across diverse front-ends, computational and memory efficiency, and robustness to spatial and temporal perturbations. Integrating MatchFusion-T into SparseDrive further improves perception under the original training objectives, supporting its compatibility with end-to-end optimization. These findings support explicit–implicit matching as an efficient and reusable mechanism for instance-level interaction.

Future work will explicitly model uncertainty in geometric priors to adapt their influence and extend MatchFusion to multi-source ensembling for offline perception.

## REFERENCES

[1] B. Liao, S. Chen, H. Yin, B. Jiang, C. Wang, S. Yan, X. Zhang, X. Li, Y. Zhang, Q. Zhang et al., “Diffusiondrive: Truncated diffusion model for end-to-end autonomous driving,” in CVPR. IEEE, 2025, pp. 12 037–12 047.

[2] K. Chitta, A. Prakash, B. Jaeger, Z. Yu, K. Renz, and A. Geiger, “Transfuser: Imitation with transformer-based sensor fusion for autonomous driving,” TPAMI, vol. 45, no. 11, pp. 12 878–12 895, 2022.

[3] W. Sun, X. Lin, Y. Shi, C. Zhang, H. Wu, and S. Zheng, “Sparsedrive: End-to-end autonomous driving via sparse scene representation,” in ICRA. IEEE, 2025, pp. 8795–8801.

[4] Z. Song, C. Jia, L. Liu, H. Pan, Y. Zhang, J. Wang, X. Zhang, S. Xu, L. Yang, and Y. Luo, “Don’t shake the wheel: Momentum-aware planning in end-to-end autonomous driving,” in CVPR. IEEE, 2025, pp. 22 432–22 441.

[5] Z. Wang, Z. Huang, Y. Gao, N. Wang, and S. Liu, “Mv2dfusion: Leveraging modality-specific object semantics for multi-modal 3d detection,” TPAMI, 2025.

[6] Y. Huang and Y. Liu, “Streamcmt: Prior-guided multimodal temporal fusion for sparse 3d object detection,” RA-L, vol. 11, no. 5, pp. 5358– 5365, 2026.

[7] Y. Xie, C. Xu, M.-J. Rakotosaona, P. Rim, F. Tombari, K. Keutzer, M. Tomizuka, and W. Zhan, “Sparsefusion: Fusing multi-modal sparse representations for multi-sensor 3d object detection,” in ICCV, 2023, pp. 17 591–17 602.

[8] X. Lin, Z. Pei, T. Lin, L. Huang, and Z. Su, “Sparse4d v3: Advancing end-to-end 3d detection and tracking,” arXiv preprint arXiv:2311.11722, 2023.

[9] S. Wang, Y. Liu, T. Wang, Y. Li, and X. Zhang, “Exploring objectcentric temporal modeling for efficient multi-view 3d object detection,” in ICCV, 2023, pp. 3621–3631.

[10] X. Li, P. Li, X. Wu, L. Shi, D. Liu, Y. Wu, J. Fu, D. Cui, L. Zhao, and L. Sun, “Rethinking the spatio-temporal alignment of end-to-end 3d perception,” in AAAI, vol. 40, no. 8, 2026, pp. 6513–6520.

[11] X. Wu, Y. Wu, X. Li, Z. Li, L. Zhao, and L. Sun, “Fusion-poly: A polyhedral framework based on spatial-temporal fusion for 3d multiobject tracking,” arXiv preprint arXiv:2603.08199, 2026.

[12] Y. Zhang, X. Wang, X. Ye, W. Zhang, J. Lu, X. Tan, E. Ding, P. Sun, and J. Wang, “Bytetrackv2: 2d and 3d multi-object tracking by associating every detection box,” 2023.

[13] H. Caesar, V. Bankiti, A. H. Lang, S. Vora, V. E. Liong, Q. Xu, A. Krishnan, Y. Pan, G. Baldan, and O. Beijbom, “nuscenes: A multimodal dataset for autonomous driving,” in CVPR, 2020, pp. 11 621–11 631.

[14] S. Vora, A. H. Lang, B. Helou, and O. Beijbom, “Pointpainting: Sequential fusion for 3d object detection,” in CVPR, 2020, pp. 4604– 4612.

[15] C. Wang, C. Ma, M. Zhu, and X. Yang, “Pointaugmenting: Crossmodal augmentation for 3d object detection,” in CVPR, 2021, pp. 11 794–11 803.

[16] Z. Liu, H. Tang, A. Amini, X. Yang, H. Mao, D. L. Rus, and S. Han, “Bevfusion: Multi-task multi-sensor fusion with unified bird’s-eye view representation,” in ICRA. IEEE, 2023, pp. 2774–2781.

[17] L. Nguyen, M. Fauth, B. Jaeger, D. Dauner, M. Igl, A. Geiger, and K. Chitta, “Lead: Minimizing learner-expert asymmetry in end-to-end driving,” in CVPR, 2026, pp. 39 775–39 785.

[18] B. Zhang, N. Song, X. Jin, and L. Zhang, “Bridging past and future: End-to-end autonomous driving with historical prediction and planning,” in CVPR. IEEE, 2025, pp. 6854–6863.

[19] X. Li, T. Xie, D. Liu, J. Gao, K. Dai, Z. Jiang, L. Zhao, and K. Wang, “Poly-mot: A polyhedral framework for 3d multi-object tracking,” in IROS. IEEE, 2023, pp. 9391–9398.

[20] A. Kim, A. Osep, and L. Leal-Taix ˇ e, “Eagermot: 3d multi-object ´ tracking via sensor fusion,” in ICRA. IEEE, 2021, pp. 11 315–11 321.

[21] J. Philion and S. Fidler, “Lift, splat, shoot: Encoding images from arbitrary camera rigs by implicitly unprojecting to 3d,” in ECCV. Springer, 2020, pp. 194–210.

[22] I. Loshchilov and F. Hutter, “Decoupled weight decay regularization,” arXiv preprint arXiv:1711.05101, 2017.

[23] Y. Wang, V. C. Guizilini, T. Zhang, Y. Wang, H. Zhao, and J. Solomon, “Detr3d: 3d object detection from multi-view images via 3d-to-2d queries,” in CoRL. PMLR, 2022, pp. 180–191.

[24] Y. Tang, Z. Meng, G. Chen, and E. Cheng, “Simpb: A single model for 2d and 3d object detection from multiple cameras,” in ECCV. Springer, 2024, pp. 1–17.

[25] Y. Zhou and O. Tuzel, “Voxelnet: End-to-end learning for point cloud based 3d object detection,” in CVPR, 2018, pp. 4490–4499.

[26] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in CVPR, 2016, pp. 770–778.

[27] X. Bai, Z. Hu, X. Zhu, Q. Huang, Y. Chen, H. Fu, and C.-L. Tai, “Transfusion: Robust lidar-camera fusion for 3d object detection with transformers,” in CVPR, 2022, pp. 1090–1099.

[28] Z. Yang, J. Chen, Z. Miao, W. Li, X. Zhu, and L. Zhang, “Deepinteraction: 3d object detection via modality interaction,” NeurIPS, vol. 35, pp. 1992–2005, 2022.

[29] Z. Yang, N. Song, W. Li, X. Zhu, L. Zhang, and P. H. Torr, “Deepinteraction++: Multi-modality interaction for autonomous driving,” TPAMI, 2025.