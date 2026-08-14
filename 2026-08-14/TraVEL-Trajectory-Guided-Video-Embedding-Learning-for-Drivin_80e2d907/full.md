# TraVEL: Trajectory-Guided Video Embedding Learning for Driving-Video Retrieval

Yi-Chung Chen<sup>1,2</sup>, Philip Jacobson<sup>1†</sup>, Tom Lampo<sup>1</sup>, Yiren Lu<sup>1</sup>, Jin Yao<sup>1</sup>, David I. Inouye<sup>2</sup>, Jing Gao<sup>2</sup>, Danhua Guo<sup>1</sup>, Burhan Yaman<sup>1‡</sup>

<sup>1</sup>Uber AV Labs <sup>2</sup>Purdue University

† Corresponding author, ‡ Project Lead

Eficiently retrieving relevant clips from large-scale driving logs is essential for data curation, model development, and safety analysis. Structured and rule-based retrieval systems can explicitly target driving events, but typically require expert-defined rules, auxiliary data, and multi-stage perception pipelines. Multimodal embedding models ofer a simpler and more eficient alternative by representing each video with a single searchable vector. However, general-purpose models often rely on shortcuts from static scene context and struggle to distinguish motion-centric events, such as turning left versus right or accelerating versus decelerating. In this work, we study how to adapt a general-purpose multimodal embedding model to driving-video retrieval. We first fine-tune Qwen3-VL-Embedding on paired clips and reasoning traces from nuReasoning using an InfoNCE objective. While this stage substantially improves overall retrieval, caption supervision alone remains insuficient for finegrained motion understanding. We therefore introduce TraVEL (Trajectory-Guided Video Embedding Learning), a motion-aware fine-tuning framework that uses ego-trajectory similarity as a reward within Group Relative Policy Optimization. Trajectories serve only as privileged training supervision; retrieval still operates on single-vector video embeddings without ego poses, expert rules, or auxiliary perception outputs. We further construct a driving-video retrieval benchmark from nuReasoning. Experiments show that TraVEL improves motion-centric retrieval across model scales: relative to SFT, it raises longitudinal and lateral mAP by 9.8 and 4.7 points at 2B, with corresponding gains of 7.2 and 1.5 points at 8B. TraVEL thus combines physically grounded supervision with eficient embedding-based search.

Date: August 14, 2026

UberPURDSUE UNIVERSITY

## 1 Introduction

Large-scale autonomous-driving fleets continuously collect vast amounts of video, yet safety-critical interactions and long-tail events constitute only a small fraction of these logs. Retrieving scenarios of interest is therefore a critical component of autonomous-driving data engines: retrieved examples can be curated for targeted annotation and model improvement [12], selected for simulation and safety validation [2, 3], or used as precedents for counterfactual assessment of candidate driving actions [19]. Such targeted data curation can support the development and evaluation of downstream driving systems [14, 29]. The value of these downstream workflows depends on whether operationally meaningful events can be reliably surfaced from large-scale driving data.

In practice, a common and dependable way to retrieve well-specified driving events is to use structured or rule-based systems that translate expert knowledge into functions over detected events and motion cues [30]. Such systems provide explicit control over the target scenario, but the relevant event definitions and decision rules must be designed in advance and calibrated to the domain. Their execution also commonly depends on auxiliary data or multi-stage perception outputs, such as object detections, tracks, map elements, and ego-motion signals. Extending them to new event vocabularies, datasets, or operating conditions can therefore incur substantial engineering cost.

In contrast, multimodal embedding models ofer a direct and eficient retrieval interface. They map text and video into a shared representation space, allowing every clip to be precomputed as a single vector and searched through fast similarity matching without a structured event detector or reasoning pipeline at retrieval time. Beginning with CLIP [21], recent methods have adapted increasingly capable vision-language models into general-purpose embedding models; VLM2Vec, VLM2Vec-V2 [8, 17], and Qwen3-VL-Embedding [11] achieve strong performance across a broad range of multimodal retrieval tasks. However, this simplicity does not guarantee sensitivity to the distinctions that matter in driving. Video–text models often exploit shortcuts based on objects or static scene context and struggle with counterfactual descriptions that difer primarily in their actions [16]. Consequently, operationally opposite events, such as turning left versus right or accelerating versus decelerating, can remain dificult to distinguish when they occur in visually similar scenes.

![](images/61fd9ac10f61f10c7fe4056f378fc5253dfca569630eb1f4c4c3e6994f8f9132.jpg)  
Figure 1 Conceptual efect of TraVEL fine-tuning on the video embedding space. Before trajectory-aware fine-tuning (left), two daytime clips receive relatively high cosine similarity (0.420) despite exhibiting diferent ego motion: an accelerating left turn versus a decelerating right turn. Meanwhile, daytime and nighttime clips that share the same accelerating left-turn trajectory have low similarity (0.123). TraVEL reverses this context bias (right), reducing the similarity of the motion-incompatible pair to 0.094 and increasing the similarity of the matched-trajectory pair to 0.462. Thus, the learned geometry becomes more sensitive to vehicle motion and less dominated by visual context.

Motivated by this tension between eficient retrieval and motion sensitivity, we ask: Can a general-purpose multimodal embedding model be adapted into an efective driving-video retriever? We first evaluate Qwen3-VL-Embedding on driving videos and observe a substantial domain gap: zero-shot retrieval varies widely across actions and is particularly weak for directional turns, lane changes, and subtle within-lane motion. We then fine-tune the model on video clips paired with reasoning-trace captions from the nuReasoning dataset [6] using the InfoNCE objective [23]. Although this supervised fine-tuning stage substantially improves overall retrieval, several motion-centric queries, including acceleration and deceleration, remain challenging even when the corresponding terms appear in the training captions. Caption supervision alone therefore does not provide a suficiently dense signal for learning fine-grained vehicle motion.

To address this limitation, we introduce TraVEL (Trajectory-Guided Video Embedding Learning), a motionaware fine-tuning framework that uses ego-trajectory similarity as a privileged training signal. Ego trajectories are commonly available from onboard localization and inertial sensors; when they are not, they can also be estimated directly from camera observations using visual geometry models such as DVGT [32]. TraVEL incorporates this signal through Group Relative Policy Optimization (GRPO) [4], encouraging the video embedding space to reflect the structure of vehicle motion. As illustrated in Fig. 1, TraVEL increases similarity between clips that share the same ego trajectory despite diferent visual contexts, while decreasing similarity between visually similar clips with incompatible motion. After training, each clip is still indexed and retrieved using a single embedding; ego poses, expert-defined rules, and auxiliary perception outputs are not required at retrieval time. TraVEL thus combines physically grounded motion supervision with the simplicity and eficiency of embedding-based search, yielding particularly clear gains on motion-centric queries.

Our contributions are threefold:

• We present TraVEL, a simple and efective framework for adapting general-purpose multimodal embedding models to driving-video retrieval.

• We introduce a trajectory-derived reward and a GRPO-based fine-tuning stage that make video embeddings more sensitive to ego motion and improve downstream retrieval performance.

• We introduce a driving-video retrieval benchmark derived from nuReasoning and demonstrate the potential of embedding-based retrieval for motion-centric driving queries.

## 2 Related Work

General-purpose multimodal embeddings. Contrastive vision–language pretraining established shared embedding spaces as an efective interface for large-scale cross-modal retrieval. CLIP [21] learns such a space from image–text pairs, while SigLIP [31] replaces softmax normalization over in-batch pairs with independent sigmoid losses. More recent work repurposes pretrained vision-language models as broadly applicable embedding models. VLM2Vec and VLM2Vec-V2 [8, 17] unify heterogeneous multimodal tasks under an embedding formulation and extend evaluation from images to videos and visual documents; Qwen3-VL-Embedding [11] further couples a general-purpose embedding model with a reranker. Alternative objectives and interaction mechanisms improve the expressiveness of these representations: VL-JEPA [1] predicts continuous language embeddings and supports text-to-video retrieval directly in its latent space, whereas MetaEmbed [27] exposes a compact set of vectors for flexible late interaction at test time. These models demonstrate strong general semantic retrieval, but their ability to distinguish visually similar driving clips by fine-grained ego motion remains underexplored. We retain the eficiency of a single-vector index and adapt its geometry using physical motion available during training.

Fine-grained video–text retrieval. Video retrieval methods extend image–text contrastive learning with temporal modeling and video-specific supervision. VideoCLIP [28] jointly pretrains video and text transformers, while CLIP4Clip [15] transfers CLIP representations to video retrieval with temporal aggregation. More recently, VeRVE [5] uses a shared multimodal language-model backbone to support corpus-, moment-, and composed-query retrieval through unified embeddings. A complementary line augments the visual representation with generated language. Cap4Video [26] incorporates auxiliary captions into training, feature interaction, and scoring; Narrating the Video [7] aggregates frame-level captions while filtering irrelevant or incorrect narration; and ELIOT [13] combines of-the-shelf captioners, language models, and text retrieval for zero-shot search. X-CoT [20] instead replaces similarity-only ranking with language-model chain-of-thought comparisons to produce interpretable rankings.

Despite this progress, fine-grained temporal distinctions remain dificult. Counterfactually augmented evaluation reveals that video–text models often rely on objects and scene context rather than the queried action [16]. VideoComp [10] addresses a related limitation with temporally disrupted negatives and a preference objective for compositional alignment in multi-event videos. Our work shares the goal of motion-sensitive retrieval but takes a diferent route: rather than adding generated captions, inference-time reasoning, or discrete temporal perturbations, we use continuous ego trajectories to shape the video embedding space during training. Retrieval itself remains a single cosine-similarity search.

Driving-scenario retrieval and mining. Driving-log search introduces domain-specific requirements because relevant events are often defined by multi-agent motion and planning context. RefAV [2] formulates planningcentric scenario mining as detecting and spatiotemporally localizing natural-language descriptions of multi-agent interactions, and evaluates referential multi-object tracking systems on Argoverse 2 logs. CARIM [9] generates compressed scene captions and converts driving-video retrieval into inclusive text-to-text matching so that all requested conditions are satisfied. For complex dynamic events, STRIVE-D [30] calibrates structured query rules from weakly labeled in-domain videos and combines them with vision-language and keyword retrieval signals. These approaches provide expressive control through tracking outputs, intermediate captions, or calibrated rules. In contrast, TraVEL requires no structured event library or auxiliary reasoning pipeline at retrieval time: ego poses act only as privileged training supervision, after which every video is indexed by one embedding and retrieved directly from natural-language queries.

## 3 Preliminaries

## 3.1 Multimodal Embedding-Based Retrieval

Let $\mathcal { D } = \{ ( q _ { i } , x _ { i } ) \} _ { i = 1 } ^ { N }$ denote paired natural-language descriptions $q _ { i }$ and videos $x _ { i }$ . Multimodal embedding models map variable-length text and visual inputs into a shared fixed-dimensional space. Models such as Qwen3-VL-Embedding [11] tokenize the input content together with a dedicated pooling token; after contextualization by the transformer, the pooling token’s final hidden state summarizes the full input and is normalized to form a single embedding vector.

At retrieval time, a text query $q$ ranks a video corpus $\mathcal { X } = \{ x _ { j } \} _ { j = 1 } ^ { M }$ . Let $f _ { \theta }$ denote the embedding model, whose outputs are unit-normalized. Each candidate is scored by cosine similarity,

$$
s _ { \theta } ( q , x _ { j } ) = \langle f _ { \theta } ( q ) , f _ { \theta } ( x _ { j } ) \rangle .\tag{1}
$$

The candidates are sorted in descending order of $s _ { \theta } ( q , x _ { j } )$ . Because every corpus video is represented by one precomputed vector, ranking reduces to eficient nearest-neighbor search over the dense index.

## 3.2 Group Relative Policy Optimization

Group Relative Policy Optimization (GRPO) [4] optimizes a policy using rewards normalized within a group of sampled candidates, avoiding a separately learned value function. For a group $\mathcal { G } _ { i }$ of k candidates with rewards $r _ { i j }$ , GRPO computes the group-relative advantage

$$
\begin{array} { l l } { \displaystyle { A _ { i j } = \frac { r _ { i j } - \mu _ { i } } { \sigma _ { i } + \epsilon } } , } \\ { \displaystyle { \mu _ { i } = \frac { 1 } { k } \sum _ { j \in { \mathcal { G } _ { i } } } r _ { i j } } , \qquad } & { \displaystyle { \sigma _ { i } ^ { 2 } = \frac { 1 } { k } \sum _ { j \in { \mathcal { G } _ { i } } } ( r _ { i j } - \mu _ { i } ) ^ { 2 } } . } \end{array}\tag{2}
$$

When $\sigma _ { i } = 0$ , all candidates carry the same reward and therefore provide no relative ranking signal; we set their advantages to zero.

Let $\pi _ { \boldsymbol { \theta } } , \pi _ { \mathrm { o l d } }$ , and $\pi _ { \mathrm { r e f } }$ denote the current, rollout, and reference policies, respectively. The clipped GRPO objective with reference-policy regularization is

$$
\begin{array} { l } { \displaystyle \mathcal { L } _ { \mathrm { G R P O } } ( \theta ) = \mathrm { ~ - ~ } \mathbb { E } _ { i } \left[ \frac { 1 } { k } \sum _ { j \in \mathcal { G } _ { i } } \operatorname* { m i n } ( \rho _ { i j } A _ { i j } , \mathrm { c l i p } ( \rho _ { i j } , 1 - \varepsilon , 1 + \varepsilon ) A _ { i j } ) \right] } \\ { \displaystyle + \beta D _ { \mathrm { K L } } ( \pi _ { \theta } | | \pi _ { \mathrm { r e f } } ) , } \end{array}\tag{3}
$$

where $\rho _ { i j } = \pi _ { \theta } ( j ) / \pi _ { \mathrm { o l d } } ( j )$ is the importance ratio, $\varepsilon$ is the clipping threshold, and $\beta$ controls an optional KL penalty toward the reference policy. TraVEL instantiates the policy, candidate groups, and rewards for driving-video retrieval in Sec. 4.3.

## 4 TraVEL: Trajectory-Guided Video Embedding Learning

The single-vector retrieval formulation in Sec. 3.1 keeps search eficient, but does not by itself ensure that the embedding space preserves fine-grained vehicle motion. During training, TraVEL augments each text–video pair $( q _ { i } , x _ { i } )$ with a synchronized ego-pose sequence $\mathbf { P } _ { i }$ . These poses provide privileged training supervision only: they are never provided as inputs to the embedding model and are not required when indexing or retrieving videos. In the following sections, we first diagnose the motion-related failure modes of a general-purpose embedding model, adapt it to the driving domain with paired caption supervision, and finally reshape its video representation using a trajectory-derived GRPO objective.

![](images/34b3f0882b4c8318b7470df49497a95b19e87d9ccf1ac05e723f07de276fd804.jpg)  
(a) Caption–video similarity distributions.

![](images/eb4700c27e650362870c7e699d417a1b348fba478443f653b2d274734b89cd9d.jpg)  
(b) Joint caption–video embedding visualization.

Figure 2 Diagnosis of the pretrained driving-video embedding space, with the representation after the complete TraVEL pipeline shown for comparison. (a) Density of cosine similarity between each reasoning-trace caption and its paired ground-truth video (blue) or non-ground-truth videos (orange). Their heavy overlap under Pretrained demonstrates weak retrieval discriminability; the TraVEL panel previews the separation obtained after fine-tuning. (b) Two-dimensional t-SNE projection of video embeddings (filled markers) and caption embeddings (hollow markers), with colors denoting longitudinal motion categories. Pretrained exhibits a large caption–video modality gap and strongly intermixed video motions, while the TraVEL panel shows the resulting cross-modal alignment and clearer motion-specific structure.

## 4.1 Diagnosing Driving-Video Embedding Gaps

We begin with Qwen3-VL-Embedding [11] and directly evaluate its pretrained representations on driving videos. As shown in Fig. 2(a), using the pretrained model for retrieval yields poor performance. The figure compares the cosine-similarity distribution between each reasoning-trace caption and its paired ground-truth video with that between the same caption and all other videos. For the pretrained model, the two distributions are concentrated at similarly high values and overlap substantially. Consequently, paired and non-paired videos receive nearly indistinguishable similarity scores, leaving little discriminative margin for retrieval.

We further examine the structure of the embedding space. Fig. 2(b) presents a t-distributed stochastic neighbor embedding (t-SNE) visualization [24] of caption and video embeddings, colored by longitudinal-motion category. A clear modality gap separates caption embeddings from video embeddings, making cross-modal matching dificult. Moreover, video embeddings associated with diferent motion categories are densely intermixed, indicating that the pretrained representation does not organize driving clips according to ego-vehicle motion.

We attribute these limitations to two factors: the domain gap between general-purpose pretraining data and driving videos, and insuficient motion sensitivity because textual supervision provides only an indirect signal for fine-grained vehicle dynamics. To address both issues, we propose a two-stage training pipeline. The first stage uses supervised fine-tuning to adapt the embedding space to the driving domain, while the second stage applies trajectory-aware GRPO to explicitly improve motion sensitivity.

## 4.2 Driving-Domain Supervised Fine-Tuning

We first reduce the visual and linguistic domain gap through supervised contrastive fine-tuning. Given a mini-batch B of videos paired with reasoning-trace captions, every non-matching video in the batch serves as a negative for the corresponding caption. The text-to-video InfoNCE objective [23] is

$$
\mathcal { L } _ { \mathrm { S F T } } ( \theta ) = - \frac { 1 } { \vert \mathcal { B } \vert } \sum _ { i \in \mathcal { B } } \log \frac { \exp ( s _ { \theta } ( q _ { i } , x _ { i } ) / \tau _ { \mathrm { S F T } } ) } { \sum _ { j \in \mathcal { B } } \exp ( s _ { \theta } ( q _ { i } , x _ { j } ) / \tau _ { \mathrm { S F T } } ) } ,\tag{4}
$$

where $\tau _ { \mathrm { S F T } }$ is a temperature. This stage aligns driving-specific language with the corresponding visual content and yields the checkpoint $\theta _ { \mathrm { S F T } }$ used to initialize the reinforcement stage.

Caption supervision substantially improves overall retrieval, but it provides only a sparse description of the motion contained in each clip. Moreover, InfoNCE treats every non-matching video as a negative: two clips

![](images/ad98d08fef10f77e02ad201cbcae09284f26c35301f3aebc20576ec0f16afd47.jpg)  
Figure 3 Overview of TraVEL’s trajectory-aware GRPO fine-tuning. Given a video or text query and the in-batch video candidates, the current embedding policy performs online retrieval to form a group of high-scoring candidates. The text-to-video group combines the paired video with online hard negatives and uses binary instance rewards, whereas the video-to-video group uses graded ego-trajectory similarity rewards. GRPO standardizes rewards within each group and updates the shared embedding model, preserving instance-level retrieval while organizing the video space by motion.

with nearly identical ego motion receive no more afinity than two physically unrelated clips. This binary supervision is poorly matched to the continuous structure of vehicle dynamics. The remaining errors on motion-centric queries motivate a second stage that uses the full ego trajectory as dense supervision.

## 4.3 Trajectory-Aware GRPO Fine-Tuning

Our reinforcement stage uses Group Relative Policy Optimization (GRPO) [4] and physical ego motion to organize the video embedding space while retaining a text-to-video retrieval objective. Fig. 3 provides an overview of this training stage.

Motion representation. Each clip spans the same temporal window as an ego-pose sequence, resampled to T steps. From this sequence, we extract scalar motion channels

$$
\mathbf { u } _ { i } ^ { c } = \big [ u _ { i } ^ { c } ( 1 ) , \ldots , u _ { i } ^ { c } ( T ) \big ] , \qquad c \in \mathcal { C } .\tag{5}
$$

Our implementation uses ego-frame lateral displacement and speed. We keep the two channels separate because their physical units and numerical ranges difer; directly concatenating them would allow the largest-scale channel to dominate the reward. To better capture behavior changes in the lower-value region, such as the transition from stopping to accelerating, we apply the following warping function to each signal:

$$
g _ { c } ( u ) = \mathrm { s i g n } ( u ) \log ! \left( 1 + \frac { | u | } { u _ { 0 , c } } \right) ,\tag{6}
$$

where $u _ { 0 , c } > 0$ is a channel-specific reference scale. The transformation preserves the signal direction while compressing large values, thereby placing greater emphasis on changes near zero during temporal alignment.

To compare two motion signals, we use dynamic time warping (DTW) [22], which aligns similar maneuvers even when they occur at slightly diferent times within a clip. For clips i and j, the DTW distance for channel c is

$$
\begin{array} { r } { d _ { c } ( i , j ) = \displaystyle \frac { 1 } { T } \mathrm { D T W } \big ( g _ { c } ( \mathbf u _ { i } ^ { c } ) , g _ { c } ( \mathbf u _ { j } ^ { c } ) \big ) \ : , } \\ { \mathrm { D T W } ( \mathbf a , \mathbf b ) = \displaystyle \operatorname* { m i n } _ { \gamma \in \Gamma } \sum _ { ( t , t ^ { \prime } ) \in \gamma } \vert a ( t ) - b ( t ^ { \prime } ) \vert , } \end{array}\tag{7}
$$

where Γ denotes the set of monotone alignment paths. The pointwise cost captures diferences in motion direction and magnitude, while DTW provides tolerance to temporal ofsets.

Because motion channels may have diferent numerical scales, we normalize each channel using a robust corpus-level scale $D _ { c } ,$ defined as the q-th percentile of its pairwise distances. The normalized distance is $\widehat { d } _ { c } ( \bar { i } , j ) = \operatorname* { m i n } ( d _ { c } ( i , j ) / D _ { c } , 1 )$ , where clipping prevents extreme trajectories from dominating the reward. We then combine the channels using non-negative weights $w _ { c } \colon$

$$
r _ { i j } ^ { \mathrm { m o t } } = R \left( 1 - \frac { \sum _ { c \in \mathcal { C } } w _ { c } \widehat { d } _ { c } ( i , j ) } { \sum _ { c \in \mathcal { C } } w _ { c } } \right) .\tag{8}
$$

The resulting reward lies in [0, R], with larger values assigned to clips exhibiting more similar ego motion.   
Normalizing by $\textstyle \sum _ { c } w _ { c }$ keeps the reward range unchanged when the channel weights are adjusted.

On-policy retrieval groups. At each optimization step, we construct candidate groups from the current model scores over a candidate pool P. For each paired caption–video example $( q _ { i } , x _ { i } )$ , we define

$$
\begin{array} { r l } & { \mathcal { G } _ { i } ^ { \mathsf { t 2 v } } = \{ i \} \cup \mathrm { T o p } _ { k - 1 } \{ s _ { \theta } ( q _ { i } , x _ { j } ) : j \in \mathcal { P } , j \neq i \} , } \\ & { \mathcal { G } _ { i } ^ { \mathrm { v 2 v } } = \mathrm { T o p } _ { k } \left\{ s _ { \theta } ( x _ { i } , x _ { j } ) : j \in \mathcal { P } , j \neq i \right\} . } \end{array}\tag{9}
$$

(10)

The text-to-video group contains the paired video and the highest-scoring non-paired videos under the current model. We assign the binary reward $\bar { r } _ { i j } ^ { \mathrm { t 2 v } } = { \bf 1 } [ j = i ]$ , encouraging the model to preserve the paired text–video alignment learned during supervised fine-tuning. The video-to-video group excludes the query video itself and instead uses the trajectory-based reward from Eq. (8), such that $r _ { i j } ^ { \mathrm { v 2 v } } = r _ { i j } ^ { \mathrm { m o t } }$

We apply the graded trajectory reward to video-to-video similarity rather than directly to text-to-video retrieval. A motion description can correspond to multiple valid clips, and imposing this many-to-many relationship on the text-to-video pathway may conflict with paired-caption supervision. Instead, the video-tovideo objective organizes the video embedding space according to ego motion, while the text-to-video objective maintains precise cross-modal alignment.

Group-relative optimization. For modality $m \in \{ \mathrm { t } 2 \mathrm { v } , \mathrm { v } 2 \mathrm { v } \}$ , let $a _ { i } ^ { \mathrm { t 2 v } } = q _ { i }$ and $a _ { i } ^ { \mathrm { v 2 v } } = x _ { i }$ . We convert the retrieval scores within each group into a categorical policy:

$$
\pi _ { \theta } ^ { m } ( j \mid a _ { i } ^ { m } ) = \frac { \exp ( s _ { \theta } ( a _ { i } ^ { m } , x _ { j } ) / \tau ) } { \sum _ { j ^ { \prime } \in \mathcal { G } _ { i } ^ { m } } \exp ( s _ { \theta } ( a _ { i } ^ { m } , x _ { j ^ { \prime } } ) / \tau ) } , \qquad j \in \mathcal { G } _ { i } ^ { m } .\tag{11}
$$

For each group, rewards are standardized using Eq. (2), and the resulting advantages are optimized with the GRPO objective in Eq. (3). This yields separate text-to-video and video-to-video objectives, denoted by $\mathcal { L } ^ { \mathrm { t 2 v } }$ and $\mathcal { L } ^ { \mathrm { v 2 v } }$ , respectively. The final objective combines cross-modal retrieval alignment with motion-aware video-space learning:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { G R P O } } ( \boldsymbol { \theta } ) = \mathcal { L } ^ { \mathrm { t 2 v } } ( \boldsymbol { \theta } ) + \lambda \mathcal { L } ^ { \mathrm { v 2 v } } ( \boldsymbol { \theta } ) . } \end{array}\tag{12}
$$

## 5 Experiments

## 5.1 Experimental Setup

Dataset. nuReasoning [6] contains 20,000 long-tail driving clips, each spanning 20 seconds. At the time of our experiments, the released subset contains 2,589 source clips, which we partition into disjoint training and test sets. From each 20-second source clip, we extract three 8-second windows anchored at 5, 10, and 15 seconds. Each window contains 3 seconds of history and 5 seconds of future context relative to its anchor. We use the nuReasoning reasoning trace associated with each clip as its natural-language caption for supervised fine-tuning and instance-level retrieval. Synchronized ego poses are used to construct the trajectory reward during TraVEL’s GRPO stage.

Table 1 Instance-level text-to-video retrieval on the held-out nuReasoning pool. Recall is reported as a percentage, while MdR and MnR are raw rank positions. The chance-level reference is the analytical expectation under a uniformly random ranking of the 1,715 candidate videos. The best result in each column is bold.
<table><tr><td>Model</td><td>#Params</td><td>R@1 ↑</td><td>R@5↑</td><td>R@10 ↑</td><td>MdR↓</td><td>MnR↓</td></tr><tr><td>Chance-level reference Uniform random ranking</td><td></td><td>0.1</td><td>0.3</td><td>0.6</td><td>858</td><td>858.0</td></tr><tr><td>Off-the-shelf models</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>CLIP4Clip</td><td>151M</td><td>0.6</td><td>2.6</td><td>4.7</td><td>415</td><td>545.0</td></tr><tr><td>InternVideo2</td><td>6B</td><td>0.1</td><td>0.6</td><td>1.0</td><td>860</td><td>858.6</td></tr><tr><td>Cosmos-Embed1</td><td>1.2B</td><td>2.4</td><td>9.2</td><td>14.8</td><td>106</td><td>231.7</td></tr><tr><td>Qwen3-VL-Embed</td><td>2B</td><td>1.3</td><td>4.5</td><td>7.4</td><td>266</td><td>400.5</td></tr><tr><td>Qwen3-VL-Embed</td><td>8B</td><td>1.6</td><td>5.2</td><td>7.8</td><td>264</td><td>371.7</td></tr><tr><td>Fine-tuned models</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td> $\mathrm { Q w e n 3 – V L - E m b e d + T r a V E L }$ </td><td>2B</td><td>8.7</td><td>24.5</td><td>35.3</td><td>23</td><td>87.8</td></tr><tr><td> $\mathrm { Q w e n 3 – V L - E m b e d + T r a V E L }$ </td><td>8B</td><td>10.1</td><td>28.3</td><td>39.8</td><td>17</td><td>70.6</td></tr></table>

Evaluation metrics. We evaluate both instance-level retrieval and motion-centric retrieval. For instance-level retrieval, each reasoning-trace caption is used to rank all 1,715 videos in the held-out pool by cosine similarity. We report recall at rank K (R@K), the percentage of queries whose paired video appears among the top K results, for $K \in \{ 1 , 5 , 1 0 \}$ . We additionally report the median rank (MdR) and mean rank (MnR) of the paired video across queries. Higher R@K and lower MdR and MnR indicate better retrieval. To measure motion understanding independently of exact caption matching, we additionally use each longitudinal or lateral motion phrase as a text query and rank the same video pool using its similarity to the video embeddings. We compute average precision (AP) against the videos associated with that motion and report mean average precision (mAP) by averaging AP across all phrases in each motion family. R@K, AP, and mAP are reported as percentages, whereas MdR and MnR are raw rank positions.

Implementation details. We compare against four of-the-shelf video–text embedding models spanning a range of architectures and scales: CLIP4Clip [15] with a ViT-B/32 backbone and mean pooling, InternVideo2- Stage2 [25], the 224p variant of Cosmos-Embed1 [18], and Qwen3-VL-Embed [11]. We evaluate their publicly released pretrained checkpoints without adaptation to nuReasoning and use each model’s native preprocessing. In all cases, the reasoning trace and its paired 8-second clip are encoded independently, and candidate videos are ranked by cosine similarity between the resulting query and video embeddings.

To study adaptation at diferent model scales, we use the 2B and 8B Qwen3-VL-Embed checkpoints. We refer to the original checkpoints as Pretrained. SFT applies the contrastive objective in Eq. (4), while TraVEL combines this supervised fine-tuning stage with trajectory-aware GRPO. We follow the VLM2Vec recipe [8] for multimodal preprocessing and supervised fine-tuning. Tables 1 and 2 report instance-level and motion-centric retrieval performance, respectively, at both model scales.

For trajectory-aware GRPO, each 8-second clip is represented by T = 80 poses sampled at 10 Hz. We apply logarithmic warping with reference scales of 1 m for lateral displacement and 1 m/s for speed. Using the 95th percentile for normalization gives channel scales of D = 1.264 and 2.342, respectively. The lateral-displacement and speed channels are weighted by 1:2, and the maximum trajectory reward is R = 10. Each optimization step uses a global candidate pool of 768 examples, from which we construct text-to-video and video-to-video groups with k = 24 candidates each. We set $\tau = 0 . 0 5 , \varepsilon = 0 . 2 , \beta = 0 .$ , and λ = 1. The trajectory reward is computed only for the selected video pairs after group construction, keeping the DTW overhead small relative to the encoder forward pass and avoiding the need for a corpus-scale reward table.

## 5.2 Results

Overall retrieval. As shown in Tab. 1, all of-the-shelf models exhibit a substantial domain gap on drivingvideo retrieval: their R@1 values are at most 2.4, despite ranging from 151M to 8B parameters. InternVideo2 remains close to the uniform-random reference, whereas Cosmos-Embed1 is the strongest of-the-shelf baseline, reaching 2.4, 9.2, and 14.8 at R@1, R@5, and R@10, respectively, with a median rank of 106. Scaling Qwen3-VL-Embed from 2B to 8B yields only marginal zero-shot gains, increasing R@1 from 1.3 to 1.6 and R@10 from 7.4 to 7.8. These results indicate that model scale alone does not resolve the driving-domain gap.

Table 2 Motion-centric text-to-video retrieval at two model scales. Each motion phrase is embedded as a text query and evaluated against the held-out video pool. Per-query AP is reported for individual motions, and mAP averages AP within each motion family. All values are percentages; higher is better. The best result within each model scale is bold.
<table><tr><td rowspan="3">Motion query</td><td colspan="3">2B</td><td colspan="3">8B</td></tr><tr><td>Pretrained</td><td>SFT</td><td>TraVEL</td><td>Pretrained</td><td>SFT</td><td>TraVEL</td></tr><tr><td colspan="7">Longitudinal motion</td></tr><tr><td>Gently accelerate</td><td>34.1</td><td>49.9</td><td>55.7</td><td>29.2</td><td>49.9</td><td>56.6</td></tr><tr><td>Maintain speed</td><td>39.0</td><td>44.2</td><td>49.2</td><td>30.0</td><td>44.1</td><td>49.7</td></tr><tr><td>Slow down gently</td><td>25.2</td><td>30.7</td><td>30.4</td><td>32.3</td><td>32.0</td><td>34.2</td></tr><tr><td>Gently come to a stop</td><td>14.7</td><td>33.7</td><td>53.8</td><td>18.4</td><td>43.3</td><td>57.4</td></tr><tr><td>Remain stopped</td><td>10.6</td><td>70.8</td><td>89.4</td><td>21.8</td><td>82.0</td><td>89.4</td></tr><tr><td>mAP</td><td>24.7</td><td>45.9</td><td>55.7</td><td>26.3</td><td>50.3</td><td>57.5</td></tr><tr><td colspan="7">Lateral motion</td></tr><tr><td>No lateral action</td><td>66.2</td><td>77.8</td><td>80.5</td><td>63.4</td><td>77.1</td><td>78.9</td></tr><tr><td>Turn right</td><td>13.1</td><td>56.0</td><td>69.8</td><td>19.1</td><td>70.5</td><td>72.0</td></tr><tr><td>Turn left</td><td>16.5</td><td>49.7</td><td>63.6</td><td>14.4</td><td>65.6</td><td>68.3</td></tr><tr><td>Right lane change</td><td>9.8</td><td>9.0</td><td>10.2</td><td>10.5</td><td>19.4</td><td>20.2</td></tr><tr><td>Left lane change</td><td>5.7</td><td>11.1</td><td>11.7</td><td>7.6</td><td>19.3</td><td>22.5</td></tr><tr><td>Slightly move right in the lane</td><td>4.6</td><td>4.2</td><td>4.7</td><td>7.7</td><td>6.4</td><td>6.8</td></tr><tr><td>Slightly move left in the lane</td><td>8.1</td><td>9.1</td><td>9.6</td><td>4.9</td><td>10.2</td><td>10.6</td></tr><tr><td>mAP</td><td>17.7</td><td>31.0</td><td>35.7</td><td>18.2</td><td>38.4</td><td>39.9</td></tr></table>

Fine-tuning with TraVEL substantially improves instance retrieval at both model scales. At 2B, TraVEL increases R@1, R@5, and R@10 from 1.3, 4.5, and 7.4 to 8.7, 24.5, and 35.3, respectively, while reducing the median rank from 266 to 23. At 8B, it raises the corresponding recall values from 1.6, 5.2, and 7.8 to 10.1, 28.3, and 39.8, and reduces the median rank from 264 to 17. These consistent gains show that the TraVEL pipeline efectively adapts a general-purpose embedding model to driving-video retrieval.

Motion-centric retrieval. Table 2 separates the efects of supervised fine-tuning and trajectory-aware optimization. At 2B, SFT improves longitudinal and lateral mAP from 24.7 and 17.7 to 45.9 and 31.0, respectively. TraVEL further raises them to 55.7 and 35.7, corresponding to gains of 9.8 and 4.7 points over SFT. The same trend holds at 8B: SFT reaches 50.3 longitudinal and 38.4 lateral mAP, while TraVEL improves them to 57.5 and 39.9, for additional gains of 7.2 and 1.5 points. The largest improvements appear in stopping and turning behaviors, indicating that trajectory supervision provides motion structure beyond what is learned from captions alone. Smaller gains on subtle within-lane shifts suggest that these fine-grained lateral distinctions remain challenging.

Qualitative retrieval. Fig. 4 contrasts Pretrained and TraVEL on a compositional reasoning-trace query. Pretrained fails to recover the requested longitudinal behavior and returns a nighttime clip in which the ego vehicle accelerates. TraVEL instead ranks the paired clip first, capturing both the gentle slowdown and the pedestrian interaction described by the query. This example suggests that trajectory-aware optimization can improve motion sensitivity while retaining contextual alignment with the reasoning trace.

## 6 Conclusion

We presented TraVEL, a simple and eficient framework for adapting a general-purpose multimodal embedding model to driving-video retrieval. Across 2B and 8B Qwen3-VL-Embedding checkpoints, supervised fine-tuning

Slow down gently to yield to the pedestrian crossing the crosswalk directly ahead, despite having a green traffic light.

![](images/755b48883bd06c11e5e4e5208c0956fc3e4018a2a73ecfc863bbe4856a4e4089.jpg)  
Figure 4 Qualitative top-1 text-to-video retrieval for a nuReasoning reasoning-trace query describing gentle deceleration to yield to a pedestrian despite a green trafic light. TraVEL retrieves the paired ground-truth clip (green), matching both the slowing behavior and pedestrian interaction, whereas Pretrained retrieves an incorrect nighttime clip in which the ego vehicle accelerates (red).

on paired clips and reasoning-trace captions first bridges the driving-domain gap. We then use ego-trajectory similarity as a training-only reward within GRPO to organize the embedding space according to vehicle motion, without introducing additional perception modules or changing the single-vector cosine-similarity retrieval pipeline at inference time. TraVEL preserves the instance-level gains of SFT at both model scales. At 2B, it improves longitudinal mAP from 45.9 to 55.7 and lateral mAP from 31.0 to 35.7 over SFT; at 8B, the corresponding improvements are from 50.3 to 57.5 and from 38.4 to 39.9. These results show that readily available physical motion signals can complement language supervision and produce representations that better capture fine-grained driving behavior.

Our current evaluation uses the available subset of nuReasoning and a retrieval pool of 1,715 videos. Evaluating on larger pools, additional datasets, and a broader range of driving conditions will be important for establishing scalability and generalization. Fine-grained lane changes and within-lane shifts also remain challenging, suggesting that ego motion alone does not capture every distinction relevant to driving-video retrieval. Future work could incorporate the motion of surrounding agents and scene structure into the reward while retaining the eficiency of embedding-based search.

## References

[1] Delong Chen, Mustafa Shukor, Theo Moutakanni, Willy Chung, Jade Yu, Tejaswi Kasarla, Yejin Bang, Allen Bolourchi, Yann LeCun, and Pascale Fung. VL-JEPA: Joint embedding predictive architecture for vision-language. arXiv preprint arXiv:2512.10942, 2025.

[2] Cainan Davidson, Deva Ramanan, and Neehar Peri. RefAV: Towards planning-centric scenario mining. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21537–21548, 2026.

[3] Wenhao Ding, Yulong Cao, Ding Zhao, Chaowei Xiao, and Marco Pavone. RealGen: Retrieval augmented generation for controllable trafic scenarios. In European Conference on Computer Vision, 2024.

[4] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

[5] Shaunak Halbe, Bhagyashree Puranik, Jayakrishnan Unnikrishnan, Kushan Thakkar, Vimal Bhat, and Toufiq Parag. VeRVE: Versatile retrieval for videos via unified embeddings. arXiv preprint arXiv:2601.12193, 2026.

[6] Zhiyu Huang, Johnson Liu, Rui Song, Zewei Zhou, Ruining Yang, Yun Zhang, Tianhui Cai, Hanyin Zhang, Mingxuan Gao, Valeria Xu, Jiali Chen, Yishan Shen, Yiluan Guo, Tony Qi, and Jiaqi Ma. nuReasoning: A reasoning-centric dataset and benchmark for long-tail autonomous driving. arXiv preprint arXiv:2605.31572, 2026.

[7] Chan Hur, Jeong-hun Hong, Dong-hun Lee, Dabin Kang, Semin Myeong, Sang-hyo Park, and Hyeyoung Park.

Narrating the video: Boosting text-video retrieval via comprehensive utilization of frame-level captions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 24077–24086, 2025.

[8] Ziyan Jiang, Rui Meng, Xinyi Yang, Semih Yavuz, Yingbo Zhou, and Wenhu Chen. VLM2Vec: Training vision-language models for massive multimodal embedding tasks. In International Conference on Learning Representations, 2025.

[9] Minjoo Ki, Daejung Kim, Kisung Kim, Seon Joo Kim, and Jinhan Lee. CARIM: Caption-based autonomous driving scene retrieval via inclusive text matching. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 22036–22045, 2025.

[10] Dahun Kim, AJ Piergiovanni, Ganesh Mallya, and Anelia Angelova. VideoComp: Advancing fine-grained compositional and temporal alignment in video-text models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 29060–29070, 2025.

[11] Mingxin Li, Yanzhao Zhang, Dingkun Long, Keqin Chen, Sibo Song, Shuai Bai, Zhibo Yang, Pengjun Xie, An Yang, Dayiheng Liu, Jingren Zhou, and Junyang Lin. Qwen3-VL-Embedding and Qwen3-VL-Reranker: A unified framework for state-of-the-art multimodal retrieval and ranking. arXiv preprint arXiv:2601.04720, 2026.

[12] Mingfu Liang, Jong-Chyi Su, Samuel Schulter, Sparsh Garg, Shiyu Zhao, Ying Wu, and Manmohan Chandraker. AIDE: An automatic data engine for object detection in autonomous driving. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14695–14706, 2024.

[13] Xuye Liu, Yimu Wang, and Jian Zhao. ELIOT: Zero-shot video-text retrieval through relevance-boosted captioning and structural information extraction. In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 4: Student Research Workshop, pages 381–391, 2025. doi: 10.18653/v1/2025.naacl-srw.37.

[14] Yiren Lu, Xin Ye, Burhaneddin Yaman, Jingru Luo, Zhexiao Xiong, Liu Ren, and Yu Yin. Reconstruction matters: Learning geometry-aligned BEV representation through 3d gaussian splatting. arXiv preprint arXiv:2603.19193, 2026.

[15] Huaishao Luo, Lei Ji, Ming Zhong, Yang Chen, Wen Lei, Nan Duan, and Tianrui Li. CLIP4Clip: An empirical study of CLIP for end-to-end video clip retrieval and captioning. Neurocomputing, 508:293–304, 2022. doi: 10.1016/j.neucom.2022.07.028.

[16] Wufei Ma, Kai Li, Zhongshi Jiang, Moustafa Meshry, Qihao Liu, Huiyu Wang, Christian Häne, and Alan Yuille. Rethinking video-text understanding: Retrieval from counterfactually augmented data. In European Conference on Computer Vision, pages 254–269. Springer, 2024. doi: 10.1007/978-3-031-72624-8\_15.

[17] Rui Meng, Ziyan Jiang, Ye Liu, Mingyi Su, Xinyi Yang, Yuepeng Fu, Can Qin, Zeyuan Chen, Ran Xu, Caiming Xiong, Yingbo Zhou, Wenhu Chen, and Semih Yavuz. VLM2Vec-V2: Advancing multimodal embedding for videos, images, and visual documents. arXiv preprint arXiv:2507.04590, 2025.

[18] NVIDIA, Francesco Ferroni, Prithvijit Chattopadhyay, Greg Heinrich, Mike Ranzinger, Roberto Amoroso, Alice Luo, Andrew Wang, and Ming-Yu Liu. Cosmos-Embed1: A joint video-text embedder for physical ai, 2025. URL https://research.nvidia.com/labs/cosmos-lab/cosmos-embed1/.

[19] Jay Patrikar, Apoorva Sharma, Sushant Veer, Boyi Li, Sebastian Scherer, and Marco Pavone. The case for negative data: From crash reports to counterfactuals for reasonable driving. arXiv preprint arXiv:2509.18626, 2025.

[20] Prasanna Reddy Pulakurthi, Jiamian Wang, Majid Rabbani, Sohail Dianat, Raghuveer Rao, and Zhiqiang Tao. X-CoT: Explainable text-to-video retrieval via LLM-based chain-of-thought reasoning. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 31184–31195, 2025. doi: 10.18653/v1/2025.emnlp-main.1588.

[21] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR, 2021.

[22] Hiroaki Sakoe and Seibi Chiba. Dynamic programming algorithm optimization for spoken word recognition. IEEE Transactions on Acoustics, Speech, and Signal Processing, 26(1):43–49, 1978. doi: 10.1109/TASSP.1978.1163055.

[23] Aaron van den Oord, Yazhe Li, and Oriol Vinyals. Representation learning with contrastive predictive coding. arXiv preprint arXiv:1807.03748, 2018.

[24] Laurens Van der Maaten and Geofrey Hinton. Visualizing data using t-SNE. Journal of Machine Learning Research, 9:2579–2605, 2008.

[25] Yi Wang, Kunchang Li, Xinhao Li, Jiashuo Yu, Yinan He, Chenting Wang, Guo Chen, Baoqi Pei, Ziang Yan, Rongkun Zheng, Jilan Xu, Zun Wang, Yansong Shi, Tianxiang Jiang, Songze Li, Hongjie Zhang, Yifei Huang, Yu Qiao, Yali Wang, and Limin Wang. InternVideo2: Scaling foundation models for multimodal video understanding. In European Conference on Computer Vision, 2024.

[26] Wenhao Wu, Haipeng Luo, Bo Fang, Jingdong Wang, and Wanli Ouyang. Cap4Video: What can auxiliary captions do for text-video retrieval? In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023.

[27] Zilin Xiao, Qi Ma, Mengting Gu, Chun-cheng Jason Chen, Xintao Chen, Vicente Ordonez, and Vijai Mohan. MetaEmbed: Scaling multimodal retrieval at test-time with flexible late interaction. In International Conference on Learning Representations, 2026.

[28] Hu Xu, Gargi Ghosh, Po-Yao Huang, Dmytro Okhonko, Armen Aghajanyan, Florian Metze, Luke Zettlemoyer, and Christoph Feichtenhofer. VideoCLIP: Contrastive pre-training for zero-shot video-text understanding. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 6787–6800. Association for Computational Linguistics, 2021. doi: 10.18653/v1/2021.emnlp-main.544.

[29] Jin Yao, Dhruva Dixith Kurra, Tom Lampo, Zezhou Cheng, Danhua Guo, and Burhan Yaman. VLGA: Visionlanguage-geometry-action models for autonomous driving. arXiv preprint arXiv:2606.12396, 2026.

[30] Manyi Yao, Sparsh Garg, Christian Shelton, Amit Roy-Chowdhury, and Abhishek Aich. Driving video retrieval for complex queries with structured grounding. arXiv preprint arXiv:2606.09109, 2026.

[31] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 11975–11986, 2023.

[32] Sicheng Zuo, Zixun Xie, Wenzhao Zheng, Shaoqing Xu, Fang Li, Shengyin Jiang, Long Chen, Zhi-Xin Yang, and Jiwen Lu. DVGT: Driving visual geometry transformer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14658–14668, 2026.