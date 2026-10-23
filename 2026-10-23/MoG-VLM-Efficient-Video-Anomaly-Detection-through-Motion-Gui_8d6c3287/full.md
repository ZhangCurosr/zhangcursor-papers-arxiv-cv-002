# MoG-VLM: Efficient Video Anomaly Detection through Motion-Guided Vision-Language Models

Khaled Waleed Dawoud <sup>1</sup>   
6khaled.dawoud@mbzuai.ac.ae   
<sup>2</sup>Muhammad Zaigham Zaheer<sup>1</sup>   
Zaigham.Zaheer@mbzuai.ac.ae   
Khan Muhammad<sup>2</sup>   
<sup>c</sup>khanmuhammad@skku.edu   
Muhammad Haris Khan<sup>1</sup>   
2muhammad.haris@mbzuai.ac.ae   
2   
<sup>1</sup> Mohamed bin Zayed University of   
Artificial Intelligence, Abu Dhabi, United Arab Emirates   
<sup>2</sup> Sungkyunkwan University,   
Seoul 03063,   
South Korea

## Abstract

Vision-Language Models (VLMs) have shown strong potential for video anomaly detection, but dense frame-level inference makes them impractical for real-time deployment. Existing acceleration commonly relies on uniform sampling, which ignores content dynamics and can either waste computation on static segments or miss localized anomalies. We propose MoG-VLM a motion-adaptive anomaly detection framework that leverages motion purely as a lightweight control signal to decide when VLM inference is necessary. To this end, we introduce Temporal Gradient Magnitude (TGM) which performs content-aware sampling to concentrate computation on informative regions while avoiding redundant processing. We further introduce a hierarchical dual-stream design in which dense multi-frame VLM reasoning is activated only around frames with elevated anomaly likelihood, via anchor-triggered context-window expansion with early termination. A fusion module then reconciles these instantaneous semantic cues with temporal action dynamics. Across four benchmarks, our method achieves 91.91%, 84.30%, 87.88%, and 78.96% AUC-ROC on XD-Violence, UCF-Crime, MSAD, and ShanghaiTech, respectively, establishing new state of the art on three benchmarks under the zero-shot, training-free setting. On XD-Violence, we improve over the previous best result by +0.57% while delivering a 1.31× inference speedup. These results demonstrate that content-aware allocation of VLM capacity, processing fewer frames more intelligently rather than all frames uniformly, is a more effective axis of progress than uniformly densifying VLM inference across frames. Code is available on GitHub: https://github.com/kdawoud91/MoG-VLM.

## 1 INTRODUCTION

The proliferation of camera systems has generated unprecedented volumes of video data, creating an urgent need for automated anomaly detection systems capable of identifying rare and unpredictable events in real-time [6, 21]. Traditional approaches [6, 12, 36] that process every frame with equal computational intensity or uniform stride are fundamentally misaligned with the nature of video footage, where anomalous events are sparse and localized. This inefficiency becomes particularly pronounced when deploying computationally expensive Vision-Language Models (VLMs), which offer unprecedented semantic understanding but impose prohibitive inference costs for continuous real-time processing [3, 11]. Modern video-capable architectures such as Video-LLaMA [37] and the Qwen-VL family [3] demonstrate remarkable improvements in temporal coherence and semantic depth compared to earlier contrastive models [2, 18], making them particularly well-suited for applications where abnormality is often defined by context. However, this representational power comes at a substantial computational cost. Without mechanisms to adapt inference effort to content dynamics in video anomaly detection (VAD), this representational power remains largely inaccessible under the computational and memory constraints of real-time deployment [36].

![](images/6599f417e4a89692b89ffbebc1ba35f8da32afb98c8c03837a06478681cefe26.jpg)  
Figure 1: Comparison of frame sampling strategies for video anomaly detection. (a) Dense sampling, incurring prohibitively high computational cost [20]. (b) Uniform sampling, which can waste computation on static segments [12, 36]. (c) Our motion-adaptive sampling method based on Temporal Gradient Magnitude (TGM) dynamically selects frames according to motion patterns, reducing processed frames. (d) Our introduced anchortriggered processing selectively expands time windows around high-confidence detections.

Current VLM-based anomaly detection methods overwhelmingly adopt uniform processing paradigms, sampling frames at fixed strides [12, 35, 36] or analyzing entire videos exhaustively [20] regardless of content dynamics. This is wasteful: many video intervals are static or low-activity yet incur the same per-frame VLM cost as dynamic, anomalyrich segments. Recent zero-shot extensions [12, 36] and parameter-efficient fine-tuning approaches [34, 35] have advanced reasoning quality, but inherit this uniform-sampling assumption and overlook the spatial and temporal sparsity inherent to surveillance footage.

In contrast, our framework operates from a deliberately coarse baseline sampling rate and selectively densifies only where motion dynamics indicate informative content. As illustrated in Fig. 1, dense (Fig. 1a) and uniform (Fig. 1b) sampling process frames indiscriminately, while our motion-adaptive approach (Fig. 1c-d) concentrates computation on temporally dynamic segments intervals exhibiting high or sudden motion changes characteristic of fights, explosions, or vehicle collisions and suppresses redundant inference on static footage.

This gap between semantic capability and computational tractability motivates a fundamental question: Starting from a coarse uniform sampling baseline, can we adaptively allocate additional computation only when video dynamics demand it to achieve real-time VLM-based anomaly detection? Addressing this question requires moving beyond accuracycentric evaluation toward a holistic framework that explicitly balances detection performance with computational efficiency. Specifically, there exists no prior work that systematically investigates adaptive mechanisms for content-aware resource allocation in vision-languagebased anomaly detection, nor sufficient analysis of the trade-offs between computational cost and video anomaly detection performance.

In this work, we introduce a motion-adaptive anomaly detection framework that fundamentally reimagines how computational resources are allocated across video content. Unlike classical approaches that employ motion as a direct classification feature [7, 8, 17, 19], we take advantage of motion purely as a lightweight control mechanism, a semantic activation signal that determines when VLM inference is necessary. We introduce TGM analysis to establish a motion quantification mechanism, thereby dynamically adjusting the sampling density based on detected temporal dynamics. We further introduce an anchor-triggered hierarchical refinement mechanism that preserves computational efficiency while recovering fine-grained temporal context. Adaptively sampled frames that exhibit high anomaly confidence serve as anchor points, triggering progressive multi-window analysis, ensuring that dense coverage is applied exclusively where anomalous patterns emerge. In parallel, we introduce a complementary temporal stream that captures action dynamics through videolanguage modeling. This branch processes non-overlapping temporal windows, generating video-level semantic descriptions that encode motion trajectories and behavioral sequences. The two pathways converge through adaptive scoring, combining frame-level instantaneous detections with temporal-level behavioral context into unified predictions.

Contributions. We summarize our key contributions as follows: Motion-Adaptive Sampling Framework: A novel motion quantification mechanism using multiscale TGM analysis and adaptive z-score that dynamically adjusts sampling density based on temporal dynamics, achieving substantial computational savings. Anchor-Triggered Hierarchical Refinement: A progressive multi-window expansion mechanism where adaptively sampled frames exhibiting high anomaly trigger conditional retrospective analysis, examining preceding context with early termination to prevent unnecessary computation on low-confidence detections. Tier-Aware Adaptive Fusion: A fusion mechanism in which the mixing weight λ between frame-level and temporal-level streams is not learned but inherited from the TGM motion tier of each frame, with λ taking a piecewise-constant value determined by whether the frame is unflagged, motion-flagged, or anchor-triggered. This tier-driven weighting is agnostic to dataset specifics and scene dynamics. Comprehensive Efficiency-Accuracy Analysis: A systematic study of the compute-accuracy trade-off in zero-shot VLM-based anomaly detection, a dimension largely absent from prior work. MoG-VLM attains 84.30%, 91.91%, 87.88% and 78.96% AUC-ROC on UCF-Crime, XD-Violence, MSAD and Shang setting a new zero-shot training-free state of the art on three benchmarks.

## 2 Related Work

Video anomaly detection (VAD) aims to temporally localise frames containing events that deviate from expected behaviour. Early supervised methods required frame-level annotations [21], while weakly-supervised formulations reduced the annotation burden to videolevel labels through multiple-instance learning (MIL) [4, 21, 25, 26, 27]. The emergence of large vision–language models (VLMs) and large language models (LLMs) has opened a new paradigm for VAD by inferring anomaly scores from natural-language scene descriptions at test time, without gradient-based training. LAVAD [36] pioneered this direction by captioning each frame with a frozen VLM, summarising temporal windows with an LLM, and refin ing anomaly scores through cross-modal similarity. Despite strong results, its uniform pro cessing of every sampled frame regardless of scene content leads to 71.4 GPU-hours on the XD-Violence test set alone, allocating identical computation to static backgrounds and fastevolving anomalous events alike. Subsequent work has focused on reasoning quality rather than efficiency. Unified-reasoning [12] introduces a score-based gate that triggers a second LLM pass with automatically extracted anomaly-tag priors for ambiguous videos, raising AUC-ROC to 84.28% on UCF-Crime and 91.34% on XD. Yet the underlying sampling stays content-blind, a fixed stride of 16 frames is applied uniformly, expending VLM inference on static footage regardless of scene dynamics. Other VLM-based methods relax the strictly zero-shot constraint. VERA [35] steers a frozen VLM with verbalized guiding questions optimised from labelled data, without modifying model weights, though the prompt optimisation itself requires an annotated training set. Holmes-VAU [38] and Hawk [22] instructiontune multimodal LLMs on anomaly clips to produce rich explanations, but they remain tied to seen anomaly types and incur high annotation costs. STPrompt [30] learns spatio-temporal prompts under weak class-level supervision. AnyAnomaly [1] proposes customisable VAD by casting detection as context-aware visual question answering with a frozen LVLM, and EventVAD [20] combines graph architectures with multimodal LLMs through temporal rea soning. The literature has overwhelmingly prioritised detection accuracy [1, 20, 39] over computational efficiency, leaving the deployment feasibility of VLM-based VAD unclear. Among the methods discussed above, only Unified [12] reports timing information. This gap is significant: real-world video systems must process hours of continuous footage, and the practical feasibility of VLM-based anomaly detection depends critically on throughput, not solely on AUC-ROC. The principle of allocating computation only where content demands it has a long history in video understanding. In action recognition, AdaFrame [32] trains a policy network to adaptively select informative frames, SCSampler [10] routes only salient clips to a heavy 3D backbone, AR-Net [15] learns per-frame resolution selection, FrameExit [9] conditions early-exit decisions on intermediate features, and LiteEval [31] alternates between a coarse and fine network via a gating module. The shared design pattern is a two-stage architecture: a cheap first pass produces a saliency signal, and a more expensive pass is activated selectively. Our framework inherits this two-stage philosophy but differs in three respects. First, our gating mechanism is entirely training-free, we threshold a multiscale Temporal Gradient Magnitude (TGM) via adaptive z-scores, requiring no learned parameters, whereas the methods above learn sampling policies from labelled data, an impractical requirement in anomaly detection where positive examples are scarce. Second, we target untrimmed footage spanning minutes to hours rather than trimmed classification clips, where the informative-to-static ratio is far more extreme. Third, motion serves purely as a control signal that decides when VLM inference is invoked, it does not enter the anomaly score itself, avoiding the conflation of motion salience with anomaly likelihood.

## 3 Methodology

Our framework introduces motion-adaptive sampling with hierarchical VLM-based processing through a dual-stream architecture that balances computational efficiency with detection

![](images/0321f531429c4cef0dde05ac8e0279ac419ed9efe853f8b2db39a8d27f529380.jpg)  
Figure 2: Overview of the MoG-VLM framework. Motion-adaptive sampling via TGM categorizes frames into three tiers: baseline uniform (orange), TGM-triggered densification (green), and anchor-triggered (magenta). Baseline stride frames and TGM-triggered frames with high anomaly scores become anchors, initiating hierarchical lookback windows $( W _ { 1 } , W _ { 2 } , W _ { 3 } )$ . Stream B conducts temporal-level Video-VLM multi-frame reasoning and LLM scoring. Tier-aware adaptive fusion combines both streams using motion-tier-inherited weights $( \lambda _ { \mathrm { b a s e } } , \lambda _ { \mathrm { m o t i o n } } , \lambda _ { \mathrm { a n c h o r } } )$ without training.

accuracy. Fig. 2 illustrates the complete pipeline.

## 3.1 Frame-Level Anomaly Detection via Motion-Adaptive Sampling

We propose a motion-adaptive sampling strategy that dynamically adjusts processing density based on temporal dynamics, enabling efficient allocation of computational resources to informative regions while maintaining detection accuracy.

## 3.1.1 Temporal Gradient Magnitude Analysis

Given a video sequence $V = \{ I _ { t } \} _ { t = 1 } ^ { T }$ , we extract frames at a baseline sampling rate to establish coarse temporal coverage. For each sampled frame I<sub>t</sub>, we compute the Temporal Gradient Magnitude (TGM) at multiple temporal scales $\mathcal { W } = \{ w _ { 1 } , . . . , w _ { K } \}$ to capture motion patterns ranging from instantaneous changes to sustained activities. At each scale w, we form per-pixel velocity $\nu _ { w } = ( I _ { t } - I _ { t - w } ) / w$ and acceleration $a _ { w } = ( I _ { t } - 2 I _ { t - w } + I _ { t - 2 w } ) / w ^ { 2 }$ and reduce the resulting TGM map to a scalar via its $9 0 ^ { \mathrm { t h } }$ percentile:

$$
m _ { w } ( t ) = \mathrm { P } _ { 9 0 } \biggl [ \sqrt { \nu _ { w } ^ { 2 } + \beta a _ { w } ^ { 2 } } \biggr ] ,\tag{1}
$$

where $\beta = 0 . 3 5$ controls the relative contribution of acceleration to the combined motion signal. The per-scale summaries are aggregated by magnitude-weighted averaging,

$$
\mathrm { T G M } ( t ) = \sum _ { w \in \mathcal { W } } \tilde { \alpha } _ { w } ( t ) m _ { w } ( t ) , \quad \tilde { \alpha } _ { w } ( t ) = \frac { m _ { w } ( t ) } { \sum _ { w ^ { \prime } \in \mathcal { W } } m _ { w ^ { \prime } } ( t ) } .\tag{2}
$$

Scales exhibiting stronger response contribute proportionally more to the aggregated signal.

## 3.1.2 Adaptive Z-Score Anomaly Detection

To identify temporal regions exhibiting anomalous motion patterns, we employ adaptive z-score normalization over a sliding exponential moving average (EMA) of TGM statistics. This approach enables context-aware anomaly detection that adapts to scene-specific activity levels while remaining sensitive to deviations from normal behavior.

For each frame t, we compute the adaptive z-score:

$$
z ( t ) = { \frac { \mathrm { T G M } ( t ) - \mu _ { t } } { \sigma _ { t } + \varepsilon } }\tag{3}
$$

where $\mu _ { t }$ and $\sigma _ { t }$ are the running mean and standard deviation maintained via EMA updates with decay factor α. Critically, we implement a freezing mechanism during anomaly detection windows: when $z ( t ) > \theta _ { \mathrm { a n o m } }$ (where $\theta _ { \mathrm { a n o m } }$ is the anomaly threshold), we freeze updates to $\mu _ { t }$ and $\sigma _ { t }$ to prevent contamination of the reference distribution by anomalous behavior. This ensures that detected anomalies do not erroneously recalibrate, which would reduce sensitivity to sustained anomalous activities.

## 3.1.3 Motion-Guided Adaptive Sampling

Upon detecting motion anomalies via z-score thresholding, we trigger adaptive refinement of the temporal sampling rate in the identified region. Specifically, when $z ( t ) > \theta _ { \mathrm { a n o m } }$ , we extend the detection window to $\left[ t - \Delta t , t + \Delta t \right]$ and increase the sampling rate to $r _ { \mathrm { h i g h } }$ within this temporal neighborhood, while maintaining the baseline rate $r _ { \mathrm { b a s e } }$ elsewhere. This produces a non-uniform sampling schedule:

$$
r ( t ) = \left\{ { \begin{array} { l l } { r _ { \mathrm { h i g h } } } & { { \mathrm { i f } } \exists \tau : z ( \tau ) > \theta _ { \mathrm { a n o m } } { \mathrm { ~ a n d ~ } } | t - \tau | \leq \Delta t } \\ { r _ { \mathrm { b a s e } } } & { { \mathrm { o t h e r w i s e } } . } \end{array} } \right.\tag{4}
$$

This adaptive sampling strategy concentrates computational resources on temporally dynamic regions while avoiding redundant processing of static or slowly-changing segments.

## 3.2 Vision Language Frame-Level Captioning

For each adaptively sampled frame, we employ a vision–language model to generate descriptions of the visual content. Given a sampled frame $I _ { t } .$ , frame-level captioning is formulated as:

$$
c _ { t } = \mathcal { F } _ { \mathrm { V L M } } ( I _ { t } ; \mathcal { P } _ { \mathrm { f r a m e } } ) .\tag{5}
$$

$\mathcal { F } _ { \mathrm { V L M } }$ denotes the vision–language model and ${ \mathcal { P } } _ { \mathrm { f r a m e } }$ is a task-specific prompt designed to emphasize anomaly-relevant attributes such as objects, persons, and spatial relationships. For exact prompts please see Supplementary Section C.3.1

## 3.3 Language-Based Anomaly Scoring

Given a frame caption $c _ { t }$ , we prompt the language model to assess the anomaly likelihood through structured reasoning. The prompt template $\mathcal { P } _ { \mathrm { s c o r e } } ^ { \mathrm { f r a m e } }$ is designed to emphasize instantaneous anomaly indicators, instructing the model to identify unusual actions, dangerous situations, suspicious behaviors, violence, accidents, or any visual patterns that deviate from typical video footage. (see Supplementary C.3.2 for the full prompts) The frame-level anomaly score at time t is computed as:

$$
\begin{array} { r } { s _ { t } ^ { \mathrm { f r a m e } } = \mathcal { F } _ { \mathrm { L M } } ( c _ { t } ; \mathcal { P } _ { \mathrm { s c o r e } } ^ { \mathrm { f r a m e } } ) . } \end{array}\tag{6}
$$

$\mathcal { F } _ { \mathrm { L M } }$ denotes the language model and $s _ { t } ^ { \mathrm { f r a m e } } \in [ 0 , 1 ]$ represents the frame-level anomaly score.

## 3.4 Anchor-Triggered Multi-Frame Analysis

While the frame-level analysis provides efficient anomaly detection through motion-adaptive sampling at an elevated rate $r _ { \mathrm { h i g h } }$ in dynamic regions, certain anomalous activities require even denser temporal coverage to fully capture their fine-grained dynamics. Rather than indiscriminately escalating the sampling rate across all dynamic segments, we introduce an anchor-triggered multi-window mechanism that selectively activates dense multi-frame processing only around adaptively sampled frames that exhibit high anomaly likelihood.

## 3.4.1 Anchor-Based Processing

From the motion-adaptive sampling pipeline (Section 3.1), we obtain frame-level anomaly scores $s _ { t } ^ { \mathrm { f r a m e } }$ at non-uniform positions determined by the adaptive sampling rate $r ( t ) \left( \mathrm { E q . } 4 \right)$ The full set of adaptively sampled frames is $\mathcal { T } _ { \mathrm { s a m p l e } } = \mathcal { T } _ { \mathrm { b a s e } } \cup \mathcal { T } _ { \mathrm { m o t i o n } }$ , where $\mathcal { T } _ { \mathrm { b a s e } }$ contains frames sampled at the baseline stride rate $r _ { \mathrm { b a s e } }$ and $\mathcal { T } _ { \mathrm { m o t i o n } }$ contains additional frames densified at $r _ { \mathrm { h i g h } }$ within motion-gated regions. Each frame $t _ { i } \in \mathcal { T } _ { \mathrm { s a m p l e } }$ serves as a potential anchor for triggering fine-grained window analysis.

A frame $t _ { i } \in \mathcal { T } _ { \mathrm { s a m p l e } }$ qualifies as a trigger anchor when its frame-level anomaly score exceeds an activation threshold $\theta _ { \mathrm { t r i g g e r } }$ . We define the set of trigger anchors as:

$$
\mathcal { T } _ { \mathrm { a n c h o r } } = \left\{ t _ { i } \in \mathcal { T } _ { \mathrm { s a m p l e } } ~ \middle | ~ s _ { t _ { i } } ^ { \mathrm { f r a m e } } > \theta _ { \mathrm { t r i g g e r } } \right\} .\tag{7}
$$

Since $\tau _ { \mathrm { s a m p l e } }$ inherits the non-uniform structure of $r ( t )$ , anchor candidates emerge naturally from motion dynamics rather than fixed intervals. Dynamic regions (where $r ( t ) = r _ { \mathrm { h i g h } } )$ yield denser anchor candidates, while static regions (where $r ( t ) = r _ { \mathrm { b a s e } } )$ yield anchor candidates at the baseline uniform sampling density. For each $t _ { i } \in \mathcal { T } _ { \mathrm { a n c h o r } } ,$ we initiate a hierarchical temporal analysis that progressively examines preceding frames at the native video frame rate to capture fine-grained dynamics.

## 3.4.2 Hierarchical Window Processing

For each anchor frame $t _ { i } \in \mathcal { T } _ { \mathrm { a n c h o r } }$ , we apply a three-stage hierarchical processing strategy that progressively expands the amount of pre-anchor context examined at full resolution. We define three non-overlapping lookback windows of increasing duration, parameterized by $0 < \Delta _ { 1 } < \Delta _ { 2 } < \Delta _ { 3 } \colon W _ { 1 } ( t _ { i } ) = [ t _ { i } - \Delta _ { 1 } , t _ { i } ) , W _ { 2 } ( t _ { i } ) = [ t _ { i } - \Delta _ { 2 } , t _ { i } - \Delta _ { 1 } )$ , and $W _ { 3 } ( t _ { i } ) = [ t _ { i } - \Delta _ { 3 } , t _ { i } -$ $\Delta _ { 2 } )$ . At each stage $k \in \{ 1 , 2 , 3 \}$ , we run the same frame-level pipeline on frames in $W _ { k } ( t _ { i } )$ to obtain scores $s _ { \tau } ^ { \mathrm { f r a m e } }$ , and compute the mean score over that window:

$$
\bar { s } _ { k } ( t _ { i } ) = \frac { 1 } { | W _ { k } ( t _ { i } ) | } \sum _ { \tau \in W _ { k } ( t _ { i } ) } s _ { \tau } ^ { \mathrm { f r a m e } } .\tag{8}
$$

Stage progression is conditioned on $\bar { s } _ { k } ( t _ { i } ) > \theta _ { \mathrm { t r i g g e r } }$ for $k \in \{ 1 , 2 \}$ , so additional highresolution is only processed when earlier evidence suggests sustained anomalous activity.

## 3.5 Temporal-Level Anomaly Detection via Multi-Frame Context

While frame-level analysis captures fine-grained semantic understanding of individual moments, many anomalous activities exhibit temporal dependencies spanning multiple frames. To capture these dynamics, we introduce a complementary temporal-level stream that processes multi-frame sequences through a video-language model.

## 3.5.1 Temporal Window Construction

We construct temporal windows at regular intervals along the video to capture local motion patterns and activity evolution. For a video sequence $V = \{ I _ { t } \} _ { t = 1 } ^ { T }$ , target frames are selected at a stride s, and around each target $t _ { \mathrm { t a r g e t } } ^ { ( i ) }$ we form a window $\mathcal { W } _ { i }$ of duration $2 \omega$ seconds by sampling $n _ { \mathrm { f r a m e s } }$ frames:

$$
\begin{array} { r } { \mathcal { W } _ { i } = \left\{ I _ { t _ { j } ^ { ( i ) } } \right\} _ { j = 1 } ^ { n _ { \mathrm { f r a m e s } } } , \quad t _ { j } ^ { ( i ) } \in \left[ t _ { \mathrm { t a r g e t } } ^ { ( i ) } - \omega , t _ { \mathrm { t a r g e t } } ^ { ( i ) } + \omega \right] , } \end{array}\tag{9}
$$

where the $n _ { \mathrm { f r a m e s } }$ sample positions are spaced across the interval. This ensures the videolanguage model receives a fixed-size input that spans the full window duration.

## 3.5.2 Video-Language Temporal Captioning and Scoring

For each window $\mathcal { W } _ { i }$ , we employ a video-language model that incorporates spatio-temporal attention to capture motion patterns, action progressions, and inter-frame dependencies across the sequence. The temporal caption is generated as:

$$
c _ { i } ^ { \mathrm { t e m p o r a l } } = \mathcal { F } _ { \mathrm { V i d e o L L M } } ( \mathcal { W } _ { i } ; \mathcal { P } _ { \mathrm { t e m p o r a l } } ) ,\tag{10}
$$

where $\mathcal { F } _ { \mathrm { V i d e o L L M } }$ denotes the video-language model and $\mathcal { P } _ { \mathrm { t e m p o r a l } }$ is a prompt template designed to elicit descriptions of activities, motion dynamics, and behavioral patterns. (see Supplementary C.3.3 for the full prompts). Following the frame-level scoring procedure, the temporal anomaly score for window i is then obtained via the same language model:

$$
s _ { i } ^ { \mathrm { t e m p o r a l } } = \mathcal { F } _ { \mathrm { L M } } ( c _ { i } ^ { \mathrm { t e m p o r a l } } ; \mathcal { P } _ { \mathrm { s c o r e } } ^ { \mathrm { t e m p o r a l } } ) ,\tag{11}
$$

with $s _ { i } ^ { \mathrm { t e m p o r a l } } \in [ 0 , 1 ]$ . Window-level scores are propagated forward in time until the next window, so every frame in the evaluation timeline inherits the most recent window-level score, mirroring the forward-fill rule used for frame-level scores.

## 3.6 Tier-Aware Adaptive Fusion

The frame and temporal streams provide complementary evidence: $s _ { t } ^ { \mathrm { f r a m e } }$ captures instantaneous semantics from adaptively sampled frames, while $s _ { t } ^ { \mathrm { t e m p o r a l } }$ encodes action dynamics over multi-frame windows. We fuse them per frame through a combination with a timevarying weight $\lambda _ { t } \in [ 0 , 1 ]$

$$
\begin{array} { r } { \widetilde { s } _ { t } = \lambda _ { t } s _ { t } ^ { \mathrm { f r a m e } } + \left( 1 - \lambda _ { t } \right) s _ { t } ^ { \mathrm { t e m p o r a l } } . } \end{array}\tag{12}
$$

Rather than treating $\lambda _ { t }$ as a global hyperparameter, we condition it on the sampling tier $\tau _ { t } \in \{ 0 , 1 , 2 \}$ inherited from the motion-adaptive pipeline: base $( \tau _ { t } { = } 0$ , baseline rate $r _ { \mathrm { b a s e } } )$

motion $( \tau _ { t } = 1$ , TGM-densified to $r _ { \mathrm { h i g h } } )$ , and anchor $\displaystyle ( \tau _ { t } = 2$ , hierarchical refinement at full resolution). Tier membership maps to $\lambda _ { t }$ through three scalars:

$$
\lambda _ { t } = \left\{ \begin{array} { l l } { \lambda _ { \mathrm { b a s e } } , } & { \tau _ { t } = 0 , } \\ { \lambda _ { \mathrm { m o t i o n } } , } & { \tau _ { t } = 1 , \quad \lambda _ { \mathrm { b a s e } } < \lambda _ { \mathrm { m o t i o n } } < \lambda _ { \mathrm { a n c h o r } } . } \\ { \lambda _ { \mathrm { a n c h o r } } , } & { \tau _ { t } = 2 , } \end{array} \right.\tag{13}
$$

Each frame’s fusion coefficient $\lambda _ { t }$ corresponds to its tier, and trust in $s _ { t } ^ { \mathrm { f r a m e } }$ increases monotonically with tier level. Base-tier frames are uniformly sampled and receive the smallest coefficient, which shifts emphasis to the temporal stream. Motion-tier frames are densified by TGM and receive an intermediate coefficient. Anchor-tier frames are refined hierarchically, receive the largest coefficient, and dominate the fused score. A frame’s tier reflects the highest level of processing it receives, baseline-stride and TGM-densified frames are upgraded to $\tau _ { t } = 2$ when they qualify as anchors (Eq. 7). We smooth $\lambda _ { t }$ with a 1-D Gaussian and clip to $[ 0 , 1 ]$ before applying Eq. 12 to avoid discontinuities at tier boundaries, then apply a second Gaussian to $\widetilde { s } _ { t }$ for evaluation stability.

## 4 Experiments

## 4.1 Datasets & Evaluation Metrics

XD-Violence Dataset [28]. XD-Violence (XD) is a large-scale anomaly detection benchmark that contains 800 videos for testing, covering six violence categories. UCF-Crime Dataset [21]. UCF is a large-scale anomaly detection benchmark with 290 untrimmed videos for testing across 13 categories. MSAD [40] is a recent multi-scenario benchmark offering 360 test videos across 14 scene types with varied camera perspectives. ShanghaiTech Dataset [14]. ShanghaiTech contains 107 testing videos with abnormal events captured from fixed cameras. Given its scale and category diversity, we anchor our ablation and efficiency analysis on XD-Violence and report supporting numbers on the remaining benchmarks. Evaluation Metrics. We report frame-level AUC-ROC, the standard thresholdindependent metric for video anomaly detection.

## 4.2 Implementation Details

Motion-Adaptive Sampling. We extract frames at a baseline rate of 1 FPS to establish coarse temporal coverage. For TGM computation, we employ multiscale temporal analysis with window sizes $\mathcal { W } = \{ 1 , 2 , 4 \}$ frames to capture motion patterns at different temporal scales. Adaptive z-score anomaly detection uses an exponential moving average (EMA) decay factor of $\alpha = 0 . 9 5$ for baseline statistics tracking, with anomaly threshold $\theta _ { \mathrm { a n o m } } = 2 . 3$ standard deviations. When motion anomalies are detected, we increase sampling density to $r _ { \mathrm { h i g h } } = 3$ FPS within a temporal window of $\Delta t = 1$ seconds around the detected frame. Vision-Language Models. We employ Qwen3-VL-8B-Instruct for frame-level captioning and VideoLLaMA3-7B for temporal-level processing, where each input consists of $n _ { \mathrm { f r a m e s } } =$ 10 frames uniformly sampled from a non-overlapping 64-frame window at the video frame rate. Language-Based Anomaly Scoring. For both frame-level and temporal-level, we employ Qwen3-4B-Instruct-2507. The scoring prompts employ chain-of-thought reasoning, instructing the model to first analyze the caption content, then assess anomaly likelihood on a scale from 0 (completely normal) to 1 (highly anomalous). Run-to-run AUC variation across three reruns remains below 0.20% on all benchmarks (variance attributable to LLM temperature sampling). Anchor-Triggered Multi-Frame Analysis. When a frame’s anomaly score exceeds $\theta _ { \mathrm { t r i g g e r } } { = } 0 . 5$ , we initiate hierarchical lookback windows of $[ t _ { i } - 5 , t _ { i } ) , [ t _ { i } - 1 5 , t _ { i } - 5 )$ and $[ t _ { i } - 3 0 , \stackrel { \_ } { t _ { i } } - 1 5 )$ frames at dense FPS. Progressive expansion to each subsequent window is conditioned on the preceding window’s mean score exceeding $\theta _ { \mathrm { t r i g g e r } }$ . For tier-aware fusion (Eq. 13), we set $\lambda _ { \mathrm { b a s e } } = 0 . 3 , \lambda _ { \mathrm { m o t i o n } } = 0 . 4$ , and $\lambda _ { \mathrm { a n c h o r } } = 0 . 7$ across all datasets. These values reflect a monotonically increasing trust in the frame-level stream as the sampling tier rises. We smooth $\lambda _ { t }$ with a 1-D Gaussian $( \sigma _ { \lambda } = 0 . 2 5 \mathrm { s ) }$ and clip to [0,1] before applying Eq. 12 to avoid discontinuities at tier boundaries. The fused score $\widetilde { s } _ { t }$ is then post-processed with a second 1-D Gaussian filter $( \sigma = 1 . 0 \mathrm { s } )$ for evaluation stability.

Table 1: VAD Performance comparison across XD-Violence, UCF-Crime, MSAD and ShanghaiTech. ✓ / ✗ indicate whether a method is zero-shot and training-free. Our MoG-VLM results are mean ± std over 3 runs.
<table><tr><td rowspan="2">Method</td><td rowspan="2"></td><td rowspan="2">Zero-shot Training-free</td><td>XD-Violence</td><td>UCF-Crime</td><td>MSAD</td><td>ShanghaiTech</td></tr><tr><td>AUC(%)</td><td>AUC(%)</td><td>AUC(%)</td><td>AUC(%)</td></tr><tr><td>FusedVision [日]</td><td>X</td><td>X</td><td>一</td><td>一</td><td>一</td><td>83.58</td></tr><tr><td>RTFM []</td><td>X</td><td>X</td><td>61.56</td><td>一</td><td>85.67</td><td></td></tr><tr><td>Jigsaw []</td><td>X</td><td>X</td><td></td><td></td><td></td><td>84.30</td></tr><tr><td>MGFN []</td><td>X</td><td>X</td><td></td><td>86.98</td><td>85.0</td><td></td></tr><tr><td>STPrompt []</td><td>X</td><td>X</td><td></td><td>88.08</td><td></td><td></td></tr><tr><td>OVVAD [四]</td><td>X</td><td>X</td><td>一</td><td>86.40</td><td></td><td>一</td></tr><tr><td>Holmes-VAU []</td><td>X</td><td>X</td><td>一</td><td>88.96</td><td></td><td></td></tr><tr><td>MULDE []</td><td>X</td><td>X</td><td></td><td></td><td></td><td>86.70</td></tr><tr><td>LEC-VAD []</td><td>X</td><td>X</td><td>88.47</td><td>89.97</td><td></td><td></td></tr><tr><td>Yu Wang et al. []</td><td>X</td><td>X</td><td>88.53</td><td>89.75</td><td></td><td>98.13</td></tr><tr><td>LAS-VAD []</td><td>X</td><td>X</td><td>89.96</td><td>91.05</td><td></td><td></td></tr><tr><td>AnomalyRuler [3]</td><td>X</td><td>√</td><td></td><td></td><td></td><td>85.20</td></tr><tr><td>VERA []</td><td>X</td><td>√</td><td>88.26</td><td>86.55</td><td>一</td><td>一</td></tr><tr><td>LLAVA-1.5 [] (ZS)</td><td>√</td><td></td><td>79.62</td><td>72.84</td><td></td><td>1</td></tr><tr><td>UR-DMU [9]</td><td>√</td><td></td><td></td><td></td><td>74.30</td><td></td></tr><tr><td>Eventvad [四]</td><td>√</td><td>V</td><td>87.51</td><td>82.30</td><td></td><td></td></tr><tr><td>AnyAnomaly [0]</td><td>√</td><td></td><td></td><td>80.70</td><td>1</td><td>79.70</td></tr><tr><td>LAVAD []</td><td>√</td><td></td><td>85.36</td><td>80.28</td><td></td><td></td></tr><tr><td>Unified-reasoning []</td><td>√</td><td>√</td><td>91.34</td><td>84.28</td><td>85.90</td><td></td></tr><tr><td>MoG-VLM (adaptive)</td><td>J</td><td>1</td><td> ${ \bf 9 1 . 9 1 \pm . 1 0 }$ </td><td> $\mathbf { 8 4 . 3 0 \pm . 0 8 }$ </td><td> ${ \bf 8 7 . 8 8 \pm . 1 7 }$ </td><td> $7 8 . 9 6 \pm . 1 4$ </td></tr></table>

## 4.3 Results

## 4.3.1 Comparison to State-of-the-Art Methods and Discussion

Table 1 compares MoG-VLM against representative weakly-supervised, one-class, and zeroshot training-free baselines on four standard benchmarks. Among methods that are simultaneously zero-shot and training-free the regime in which our method operates, MoG-VLM establishes new state of the art on three of the four benchmarks. On XD-Violence, we reach 91.91% AUC-ROC, improving over the previous best zero-shot training-free method, Unified-reasoning [12], by +0.57% while simultaneously delivering a 1.31× end-to-end inference speedup (Sec. 4.5). On MSAD, we obtain 87.88%, a +1.98% absolute gain over Unified-reasoning (85.90%) and a +2.88% gain over the strongest non-zero-shot method, MGFN [5]. On UCF-Crime, MoG-VLM reaches 84.30%, marginally surpassing Unifiedreasoning (84.28%). It also exceeds other zero-shot, training-free baselines such as LAVAD (80.28%), AnyAnomaly (80.70%), and EventVAD (82.30%) by a clear margin.

Table 2: Breakdown of AUC-ROC (%) across datasets under different components.
<table><tr><td rowspan=1 colspan=4>Uniformbaseline   +   +FrameTemporalrbase  (TGM) anchor   only  FusionDataset (Eq. 4) (Eq. 4) (Eq.8) (Eq.11)(Eq. 12)</td><td rowspan=1 colspan=1>Refine-ment</td></tr><tr><td rowspan=1 colspan=1>XD</td><td rowspan=1 colspan=1>84.00   84.69   87.29</td><td rowspan=1 colspan=1>85.17</td><td rowspan=1 colspan=1>88.60</td><td rowspan=1 colspan=1>91.91</td></tr><tr><td rowspan=1 colspan=1>UCF</td><td rowspan=1 colspan=1>65.02   68.31  70.07</td><td rowspan=1 colspan=1>74.48</td><td rowspan=1 colspan=1>77.99</td><td rowspan=1 colspan=1>84.30</td></tr><tr><td rowspan=1 colspan=1>MSAD</td><td rowspan=1 colspan=1>78.41   81.51   82.31</td><td rowspan=1 colspan=1>79.06</td><td rowspan=1 colspan=1>84.84</td><td rowspan=1 colspan=1>87.88</td></tr><tr><td rowspan=1 colspan=1>Shang</td><td rowspan=1 colspan=1>71.54   72.38  73.10</td><td rowspan=1 colspan=1>61.64</td><td rowspan=1 colspan=1>75.87</td><td rowspan=1 colspan=1>78.96</td></tr></table>

Table 3: Per-component computational cost.
<table><tr><td>Component</td><td>Bat</td><td>s/b</td></tr><tr><td>Stream A — Frame-Level</td><td></td><td></td></tr><tr><td>VLM Cap. (Qwen3VL-8B) LLM Score (Qwen3-4B)</td><td>24 24</td><td>8.19 0.64</td></tr><tr><td>Stream B — Temporal-Level</td><td></td><td></td></tr><tr><td>Vid-VLM (VideoLLaMA3)</td><td>4</td><td>2.04</td></tr><tr><td>LLM Score (Qwen3-4B)</td><td>24</td><td>0.64</td></tr></table>

These gains are particularly notable given the constraints of our setting. Methods such as Holmes-VAU [38] (88.96% on UCF-Crime) and STPrompt [30] (88.08%) rely on weak supervision with video-level labels and dataset-specific tuning, while VERA [35] performs verbalized learning that adapts prompts to each benchmark. On XD-Violence, our zeroshot training-free result of 91.91% in fact exceeds all supervised baselines we are aware of, suggesting that motion-adaptive sampling and anchor-triggered refinement provide a complementary inductive bias that learned representations alone do not capture.

Fig.3 disaggregates detection performance by anomaly category. On MSAD and XD, most categories exceed 90% AUC-ROC, with near-perfect scores on high-motion events such as Fighting and Abuse. UCF exhibits greater variance, visually distinctive events like Assault (94.8%) are detected reliably, while categories involving minimal distinguishing motion (e.g., Shoplifting, 59.8%) remain challenging, as the anomalous action is often brief or subtle. The results suggest that detection is most reliable when anomalies manifest as clearly dynamic events. Shang is omitted as its abnormal events are not annotated by category.

## 4.3.2 Progressive Pipeline Component Analysis

Table 2 decomposes the contribution of each component by progressively activating them on top of a uniform-sampling baseline, tracing the full path to the final MoG-VLM.

Motion-adaptive sampling (TGM). The TGM gate improves AUC-ROC by +0.69% to +3.29% across all datasets relative to the uniform baseline. These consistent gains confirm that motion gating successfully concentrates VLM computation on informative intervals without sacrificing detection quality. Anchor-triggered hierarchical refinement. Adding anchor refinement produces strongly dataset-dependent effects. On UCF (+1.76%), XD (+2.6%) and MSAD (+0.80%), hierarchical lookback contributes substantially, as many anomalies in these benchmarks (assault, accidents) unfold over multi-second pre-event contexts that benefit from denser temporal evidence. Temporal stream. The temporal column reveals notably different behavior across datasets. On UCF, the video stream alone (74.48%) surpasses the frame stream (70.07%) by +4.41%, suggesting that action-level dynamics carry strong discriminative signal for the diverse activity categories in this benchmark. On Shang, in contrast, the temporal stream (61.64%) trails the uniform baseline (71.54%), consistent with a domain gap between VideoLLaMA3’s pretraining distribution and the subtle pedestrian-scene anomalies (skating, wrong-direction walking) that characterize this dataset. Tier-aware fusion is designed to accommodate precisely this kind of cross-dataset variation, allowing each stream to contribute where it is most reliable. Tier-aware dual-stream fusion. Fusing the two streams via the tier-conditioned coefficient $\lambda _ { t }$ (Eq. 12) yields the largest single improvement at any stage on two of four datasets: +7.92% on UCF, and +2.77% on Shang, with a smaller +1.31% on XD where anchor refinement contributes more. The same mechanism, with identical scalars $( \lambda _ { \mathrm { b a s e } } , \lambda _ { \mathrm { m o t i o n } } , \lambda _ { \mathrm { a n c h o r } } )$ , accommodates both directions of complementarity. On UCF the temporal stream rescues the frame stream, while on Shang the frame stream rescues a temporal stream that performed below baseline. Because $\lambda _ { t }$ inherits from the motion sampling tier rather than being tuned per dataset, anchor-tier frames automatically receive the largest frame-stream weight while base-tier frames defer to the temporal stream. Score refinement. Following standard VAD post-processing practice [8, 12, 35], the fused score $\widetilde { s } _ { t }$ and the tier coefficient $\lambda _ { t }$ are passed through a 1D Gaussian filter to suppress high-frequency noise from VLM caption variability and tier transitions. This step introduces no new evidence and is applied identically by our zero-shot baselines, but is required for fair comparison.

Table 4: Cumulative timing breakdown (hours) across datasets.
<table><tr><td>Configuration</td><td>XD</td><td>UCF</td><td>MSAD</td><td>Shang</td></tr><tr><td>Uniform (1 FPS)</td><td>7.96</td><td>3.51</td><td>0.78</td><td>0.13</td></tr><tr><td>+ (TGM)</td><td>8.42</td><td>4.12</td><td>0.85</td><td>0.14</td></tr><tr><td>+ Frame anchor</td><td>9.60</td><td>5.21</td><td>0.96</td><td>0.17</td></tr><tr><td>Temporal only</td><td>5.20</td><td>2.59</td><td>0.66</td><td>0.11</td></tr><tr><td>Final</td><td>14.80</td><td>7.80</td><td>1.62</td><td>0.28</td></tr></table>

Table 5: Throughput across datasets.
<table><tr><td>Dataset</td><td>Method</td><td>Time (h)</td><td>FPS</td><td>Speedup</td></tr><tr><td rowspan="3">XD</td><td>LAVAD [0]</td><td>71.40</td><td>9.1</td><td>1.0×</td></tr><tr><td>Unified []</td><td>19.40</td><td>33.4</td><td>3.7×</td></tr><tr><td>Ours</td><td>14.80</td><td>43.84</td><td>4.82×</td></tr><tr><td rowspan="3">UCF</td><td>LAVAD [0]</td><td>33.93</td><td>9.1</td><td>1.0×</td></tr><tr><td>Unified []</td><td>9.24</td><td>33.40</td><td>3.7×</td></tr><tr><td>Ours</td><td>7.80</td><td>39.25</td><td>4.35×</td></tr><tr><td rowspan="3">MSAD</td><td>LAVAD [0]</td><td>6.95</td><td>9.1</td><td>1.0×</td></tr><tr><td>Unified []</td><td>1.90</td><td>33.40</td><td>3.7×</td></tr><tr><td>Ours</td><td>1.62</td><td>39.01</td><td>4.30×</td></tr><tr><td rowspan="3">Shang</td><td>LAVAD []</td><td>1.24</td><td>9.1</td><td>1.0×</td></tr><tr><td>Unified []</td><td>0.34</td><td>33.40</td><td>3.7×</td></tr><tr><td>Ours</td><td>0.28</td><td>40.46</td><td>4.45×</td></tr></table>

## 4.4 Qualitative Results

Figure 4 presents per-frame anomaly score trajectories for six videos spanning distinct anomaly regimes. MSAD—Assault (1). Both streams correctly elevate scores during the annotated anomaly window. Notably, the fused output sustains elevated scores beyond the groundtruth boundary, capturing the post-impact interval in which the victim remains prone on the ground after frame #150, a behaviorally meaningful detection that extends beyond the ground truth. XD-Violence—Multi-category violence (2). Across six concatenated anomaly types interspersed with normal intervals, both streams suppress their outputs during the normal segment (frames 300–1000) and re-elevate in close temporal alignment with each subsequent event, confirming that tier-aware fusion generalizes across heterogeneous anomaly semantics without category-specific tuning. UCF–Normal Videos-453-264 (3) The score remains consistently near zero throughout, demonstrating stable false-positive suppression under normal conditions. MSAD—Fire09 (4) TGM gating remains robust to photometric perturbations: a flickering television and ambient illumination shifts in the normal interval stay below the z-score threshold, while the propagating flame and smoke sustain a coherent multiscale gradient signal that recruits additional VLM inference, improving AUC-ROC from 0.648 (1 FPS) to 0.675 (Sec. 3.1.3). ShanghaiTech—08\_0080 (5) The anomalous skating behavior blends visually with surrounding pedestrian activity, and TGM fails to intensify

![](images/12ca93af6d2efe6e22757fe7f4afffefe5a033a3437c09258c5cce33d8072db1.jpg)

![](images/ed1122b7c19a8e97db45ba5bfa97cbde522b0effff3cf59ee67a87b0b1f93dac.jpg)

![](images/00dbda69bfebda2a1a2d5804d8b8b5fa89554ce8c63cd942baa3f5f31afb7ae6.jpg)

Figure 3: Per-category AUC-ROC breakdown across three benchmarks.  
![](images/0e4a9e874a1f483e87333f04bf11b571ea612d60ed77b49c56156767e906bf3a.jpg)  
Figure 4: Qualitative results. Pink regions denote ground-truth anomalous intervals.

on a motion pattern that is kinematically indistinguishable from normal walking, with AUC-ROC plateauing at the 1 FPS baseline of 72.9% and showing that brief, subtle ShanghaiTech anomalies remain challenging. UCF—RoadAccidents001 (6). The brief hit is missed, but the model sustains an elevated score throughout the prolonged post-impact interval in which the victim remains prone, indicating that sudden short-lived anomalies can evade detection while their behavioral post-event manifestation is more reliably captured.

## 4.5 Computational Efficiency Analysis

We report timing across XD, UCF, MSAD and Shang on a single NVIDIA A100 40 GB GPU. Table 3 gives per-component latency, Table 4 traces cumulative timing cost as each stage is activated, and Table 5 compares end-to-end throughput against prior zero-shot training-free methods. Where time is spent. Frame-level VLM captioning dominates at 8.19 s/batch (Qwen3-VL-8B), followed by VideoLLaMA3 at 2.04 s/batch; LLM scoring is negligible (0.64 s/batch). Pipeline cost is therefore governed by how often the visual encoders are invoked. Cost attribution per stage. The 1 FPS uniform baseline consumes 7.96, 3.51, 0.78 and .13 h on XD, UCF, MSAD and Shang. TGM gating adds 6–17%, anchor refinement adds a further 13–26%, and the temporal stream contributes roughly half the frame-stream cost. The final dual configuration totals 14.80, 7.80, 1.62 and .28 h. Throughput. Following the convention of stride-based methods [12], we report amortized effective throughput $\mathrm { F P S } _ { \mathrm { e f f } } = N _ { \mathrm { f r a m e s } } / T _ { \mathrm { p i p e l i n e } }$ . MoG-VLM reaches 43.84, 36.25, 39.01 and 40.46 FPS, yielding 4.30–4.82× speedups over LAVAD and 1.17–1.31× over Unified-reasoning. On XD-Violence, it saves 4.60 h of GPU inference time while improving AUC-ROC by +0.57%. Where the savings originate. Three mechanisms: a coarser 1 FPS baseline stride (vs. stride-

Table 6: Ablation of multiscale TGM window sizes W. AUC-ROC (%) is reported at the frame-level (motion-adaptive sampling only). The default configuration is shaded.
<table><tr><td>W</td><td>XD</td><td>UCF</td><td>MSAD</td><td>Shang</td></tr><tr><td>{1,2,4}</td><td>84.69</td><td>68.31</td><td>81.51</td><td>72.38</td></tr><tr><td>{1,3,5}</td><td>84.53</td><td>66.76</td><td>79.07</td><td>71.38</td></tr><tr><td>{2,4,8}</td><td>85.36</td><td>68.61</td><td>79.38</td><td>70.67</td></tr></table>

Table 7: Effect of $\theta _ { \mathrm { t r i g g e r } }$ on AUC-ROC (%) across datasets. Default conf. is shaded.
<table><tr><td> $\theta _ { \mathrm { t r i g g e r } }$ </td><td>XD</td><td>UCF</td><td>MSAD</td><td>Shang</td></tr><tr><td>0.4</td><td>87.56</td><td>70.21</td><td>82.92</td><td>73.44</td></tr><tr><td>0.5</td><td>87.29</td><td>70.07</td><td>82.31</td><td>73.10</td></tr><tr><td>0.6</td><td>85.84</td><td>68.46</td><td>82.22</td><td>73.09</td></tr><tr><td>0.7</td><td>85.29</td><td>67.95</td><td>82.14</td><td>73.16</td></tr><tr><td>0.8</td><td>84.80</td><td>67.61</td><td>82.07</td><td>73.16</td></tr></table>

16 in prior work) nearly halves the candidate pool before any content-aware filtering; TGM gating suppresses VLM inference on static intervals; and anchor expansion triggers denser lookback only when frame-level scores exceed $\theta _ { \mathrm { t r i g g e r } } ,$ , with early termination.

Crucially, motion-adaptive allocation improves both axes of the efficiency and accuracy trade-off jointly: a 1.17–1.31× speedup over Unified-reasoning is achieved alongside AUC-ROC gains on every benchmark where direct comparison is reported, indicating that the savings originate from where compute is spent rather than how much. The resulting 36– 44 FPS throughput on a single A100 enters the operating regime of standard surveillance camera framerates, suggesting that content-aware allocation is a more deployable axis of scaling for VLM-based VAD than uniformly densifying inference.

## 4.6 Ablation

Multiscale Window Size Sensitivity. We ablate the temporal window sizes used for multiscale TGM computation by comparing three configurations: $\mathcal { W } = \{ 1 , 2 , 4 \} , \{ 1 , 3 , 5 \}$ , and {2,4,8}. Our default $\mathcal { W } = \{ 1 , 2 , 4 \}$ was fixed before this sensitivity study. Table 6 reports frame-level AUC-ROC on all benchmarks (TGM only). Performance is stable across configurations, with the largest gap being 2.48% on MSAD. The wider $\mathcal { W } = \{ 2 , 4 , 8 \}$ performs best on UCF and XD, while MSAD and Shang favors $\mathcal { W } = \{ 1 , 2 , 4 \}$ , suggesting that aggregating motion over longer horizons benefits scene-diverse benchmarks while finer granularity captures short pedestrian-scene events. The robustness across configurations indicates that the framework is not sensitive to this hyperparameter choice. Anchor Trigger Threshold Sensitivity. Table 7 shows that performance peaks at lower thresholds $( \theta _ { \mathrm { t r i g g e r } } = 0 . 4 { - } 0 . 5 )$ across all datasets, with AUC-ROC degrading as $\theta _ { \mathrm { t r i g g e r } }$ increases beyond 0.5. Most notably a drop of 2.76% on XD and 2.60% on UCF when raising the threshold from 0.4 to 0.8. This indicates that permissive anchor activation is preferable, as overly conservative thresholds suppress hierarchical refinement. See Supplementary for more ablation studies. We include ablations on tier-aware fusion weights, EMA freezing during anomaly windows, $\theta _ { \mathrm { a n o m } }$ sensitivity, and Gaussian smoothing strength in Supplementary Section A.

## 5 Conclusion

We presented MoG-VLM, a zero-shot training-free framework that treats motion as a lightweight control signal governing when VLM inference is invoked. MoG-VLM sets state of the art on UCF-Crime, XD-Violence, and MSAD, remains competitive on ShanghaiTech, and achieves a speedup over the strongest zero-shot baseline, demonstrating that content-aware allocation of model capacity is a more productive axis of progress than scaling uniform VLM inference.

## Acknowledgement

This work is supported by MBZUAI Base funds BF0008 and by the BK21 FOUR Project (Bigdata Research and Education Group for Enhancing Social Connectedness Through Advanced Data Technology and Interaction Science Research, 5199990913845), funded by the Ministry of Education (MOE, Korea) and the National Research Foundation of Korea (NRF).

## References

[1] Sunghyun Ahn, Youngwan Jo, Kijung Lee, Sein Kwon, Inpyo Hong, and Sanghyun Park. Anyanomaly: Zero-shot customizable video anomaly detection with lvlm. In Proceedings ofthe IEEE/CVF Winter Conference on Applications ofComputer Vision, pages 3026–3035, 2026.

[2] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. Advances in neural information processing systems, 35:23716–23736, 2022.

[3] Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, et al. Qwen technical report. arXiv preprint arXiv:2309.16609, 2023.

[4] Weiling Chen, Keng Teck Ma, Zi Jian Yew, Minhoe Hur, and David Aik-Aun Khoo. Tevad: Improved video anomaly detection with captions. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5549–5559, 2023.

[5] Yingxian Chen, Zhengzhe Liu, Baoheng Zhang, Wilton Fok, Xiaojuan Qi, and Yik-Chung Wu. Mgfn: Magnitude-contrastive glance-and-focus network for weaklysupervised video anomaly detection. In Proceedings of the AAAI conference on artificial intelligence, volume 37, pages 387–395, 2023.

[6] Khaled Dawoud, Zaigham Zaheer, Mustaqeem Khan, Karthik Nandakumar, Abdulmotaleb Elsaddik, and Muhammad Haris Khan. Fusedvision: A knowledge-infusing approach for practical anomaly detection in real-world surveillance videos. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 4036–4046, 2025.

[7] Christoph Feichtenhofer, Axel Pinz, and Andrew Zisserman. Convolutional two-stream network fusion for video action recognition. In Proceedings ofthe IEEE conference on computer vision and pattern recognition, pages 1933–1941, 2016.

[8] Mariana-Iuliana Georgescu, Antonio Barbalau, Radu Tudor Ionescu, Fahad Shahbaz Khan, Marius Popescu, and Mubarak Shah. Anomaly detection in video via selfsupervised and multi-task learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 12742–12752, 2021.

[9] Amir Ghodrati, Babak Ehteshami Bejnordi, and Amirhossein Habibian. Frameexit: Conditional early exiting for efficient video recognition. In Proceedings of the

IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 15608– 15618, 2021.

[10] Bruno Korbar, Du Tran, and Lorenzo Torresani. Scsampler: Sampling salient clips from video for efficient action recognition. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 6232–6242, 2019.

[11] Bin Lin, Yang Ye, Bin Zhu, Jiaxi Cui, Munan Ning, Peng Jin, and Li Yuan. Video-llava: Learning united visual representation by alignment before projection. In Proceedings of the 2024 conference on empirical methods in natural language processing, pages 5971–5984, 2024.

[12] Dongheng Lin, Mengxue Qu, Kunyang Han, Jianbo Jiao, Xiaojie Jin, and Yunchao Wei. A unified reasoning framework for holistic zero-shot video anomaly analysis. Advances in Neural Information Processing Systems, 38:29713–29744, 2026.

[13] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 26296–26306, 2024.

[14] Wen Liu, Weixin Luo, Dongze Lian, and Shenghua Gao. Future frame prediction for anomaly detection–a new baseline. In Proceedings ofthe IEEE conference on computer vision and pattern recognition, pages 6536–6545, 2018.

[15] Yue Meng, Chung-Ching Lin, Rameswar Panda, Prasanna Sattigeri, Leonid Karlinsky, Aude Oliva, Kate Saenko, and Rogerio Feris. Ar-net: Adaptive frame resolution for efficient action recognition. In European conference on computer vision, pages 86–104. Springer, 2020.

[16] Jakub Micorek, Horst Possegger, Dominik Narnhofer, Horst Bischof, and Mateusz Kozinski. Mulde: Multiscale log-density estimation via denoising score matching for video anomaly detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18868–18877, 2024.

[17] Hyunjong Park, Jongyoun Noh, and Bumsub Ham. Learning memory-guided normality for anomaly detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14372–14381, 2020.

[18] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR, 2021.

[19] Paul Scovanner, Saad Ali, and Mubarak Shah. A 3-dimensional sift descriptor and its application to action recognition. In Proceedings of the 15th ACM international conference on Multimedia, pages 357–360, 2007.

[20] Yihua Shao, Haojin He, Sijie Li, Siyu Chen, Xinwei Long, Fanhu Zeng, Yuxuan Fan, Muyang Zhang, Ziyang Yan, Ao Ma, et al. Eventvad: Training-free event-aware video anomaly detection. In Proceedings ofthe 33rd ACM International Conference on Multimedia, pages 2586–2595, 2025.

[21] Waqas Sultani, Chen Chen, and Mubarak Shah. Real-world anomaly detection in surveillance videos. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 6479–6488, 2018.

[22] Jiaqi Tang, Hao Lu, Ruizheng Wu, Xiaogang Xu, Ke Ma, Cheng Fang, Bin Guo, Jiangbo Lu, Qifeng Chen, and Ying-Cong Chen. Hawk: Learning to understand open-world video anomalies. Advances in Neural Information Processing Systems, 37:139751–139785, 2024.

[23] Yu Tian, Guansong Pang, Yuanhong Chen, Rajvinder Singh, Johan W Verjans, and Gustavo Carneiro. Weakly-supervised video anomaly detection with robust temporal feature magnitude learning. In Proceedings ofthe IEEE/CVF international conference on computer vision, pages 4975–4986, 2021.

[24] Guodong Wang, Yunhong Wang, Jie Qin, Dongming Zhang, Xiuguo Bao, and Di Huang. Video anomaly detection by solving decoupled spatio-temporal jigsaw puzzles. In European Conference on Computer Vision, pages 494–511. Springer, 2022.

[25] Yu Wang and Shiwei Chen. Learning event completeness for weakly supervised video anomaly detection. arXiv preprint arXiv:2506.13095, 2025.

[26] Yu Wang and Shengjie Zhao. Weakly supervised video anomaly detection with anomaly-connected components and intention reasoning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 28545–28556, June 2026.

[27] Yu Wang, Shengjie Zhao, Jianyu Wang, and Xutao Chu. Learning local semantic signals and inter-class discrepancy for weakly supervised video anomaly detection. IEEE Transactions on Multimedia, 28:6088–6099, 2026. doi: 10.1109/TMM.2026.3668562.

[28] Peng Wu, Jing Liu, Yujia Shi, Yujia Sun, Fangtao Shao, Zhaoyang Wu, and Zhiwei Yang. Not only look, but also listen: Learning multimodal violence detection under weak supervision. In European conference on computer vision, pages 322–339. Springer, 2020.

[29] Peng Wu, Xuerong Zhou, Guansong Pang, Yujia Sun, Jing Liu, Peng Wang, and Yanning Zhang. Open-vocabulary video anomaly detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18297– 18307, 2024.

[30] Peng Wu, Xuerong Zhou, Guansong Pang, Zhiwei Yang, Qingsen Yan, Peng Wang, and Yanning Zhang. Weakly supervised video anomaly detection and localization with spatio-temporal prompts. In Proceedings of the 32nd ACM International Conference on Multimedia, pages 9301–9310, 2024.

[31] Zuxuan Wu, Caiming Xiong, Yu-Gang Jiang, and Larry S Davis. Liteeval: A coarse-tofine framework for resource efficient video recognition. Advances in neural information processing systems, 32, 2019.

[32] Zuxuan Wu, Caiming Xiong, Chih-Yao Ma, Richard Socher, and Larry S Davis. Adaframe: Adaptive frame selection for fast video recognition. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1278– 1287, 2019.

[33] Yuchen Yang, Kwonjoon Lee, Behzad Dariush, Yinzhi Cao, and Shao-Yuan Lo. Follow the rules: Reasoning for video anomaly detection with large language models. arXiv preprint arXiv:2407.10299, 2024.

[34] Zhiwei Yang, Jing Liu, and Peng Wu. Text prompt with normality guidance for weakly supervised video anomaly detection. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 18899–18908, 2024.

[35] Muchao Ye, Weiyang Liu, and Pan He. Vera: Explainable video anomaly detection via verbalized learning of vision-language models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 8679–8688, 2025.

[36] Luca Zanella, Willi Menapace, Massimiliano Mancini, Yiming Wang, and Elisa Ricci. Harnessing large language models for training-free video anomaly detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18527–18536, 2024.

[37] Hang Zhang, Xin Li, and Lidong Bing. Video-llama: An instruction-tuned audio-visual language model for video understanding. arXiv preprint arXiv:2306.02858, 2023.

[38] Huaxin Zhang, Xiaohao Xu, Xiang Wang, Jialong Zuo, Xiaonan Huang, Changxin Gao, Shanjun Zhang, Li Yu, and Nong Sang. Holmes-vau: Towards long-term video anomaly understanding at any granularity. In Proceedings of the computer vision and pattern recognition conference, pages 13843–13853, 2025.

[39] Hang Zhou, Junqing Yu, and Wei Yang. Dual memory units with uncertainty regulation for weakly supervised video anomaly detection. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 37, pages 3769–3777, 2023.

[40] Liyun Zhu, Lei Wang, Arjun Raj, Tom Gedeon, and Chen Chen. Advancing video anomaly detection: A concise review and a new dataset. Advances in Neural Informa tion Processing Systems, 37:89943–89977, 2024.

## Appendix Roadmap

In this appendix, we cover the following materials:

A Extended ablations   
A.1 Tier-aware fusion weight sensitivity . S1   
A.2 EMA freezing during anomaly windows . S2   
A.3 Anomaly threshold $\theta _ { \mathrm { a n o m } }$ sensitivity . S2   
A.4 Gaussian smoothing strength . S2   
A.5 Prompt sensitivity . . . S2   
A.6 Isolating the sampling contribution . S2   
B Additional qualitative results . S3   
C Prompts . . S4   
C.1 Frame-level captioning prompt . S4   
C.2 Frame-level anomaly scoring prompt . S5   
C.3 Temporal multi-frame captioning prompt . S5   
D Limitations . . . S6

## A Extended Ablations

## A.1 Tier-Aware Fusion Weight Sensitivity

We sweep each tier weight in $( \lambda _ { \mathrm { b a s e } } , \lambda _ { \mathrm { m o t i o n } } , \lambda _ { \mathrm { a n c h o r } } )$ independently while holding the other two at the paper configuration (0.3, 0.4,0.7). Results are reported in Tables 8–9.

$\lambda _ { \mathbf { a n c h o r } }$ dominates the trade-off. Raising it from 0.2 to 0.7 lifts mean AUC by 0.63% driven by a 2.31% gain on ShanghaiTech against a 0.28% drop on UCF-Crime, XD-Violence is invariant and MSAD gains 0.50%. Anchor-triggered frames thus carry stronger localization signal on sparse-anomaly scenes, but over-commit on the denser, more ambiguous UCF-Crime distribution. $\lambda _ { \mathbf { b a s e } }$ exhibits the same asymmetry more sharply. Increasing it from 0.3 to 0.6 raises mean AUC to 86.31% (+0.46%), with ShanghaiTech gaining 1.54% and UCF-Crime losing 0.55%. We retain $\lambda _ { \mathrm { b a s e } } = 0 . 3$ because higher values inflate the mean at UCF-Crime’s expense, where motion-gated triggering is the primary contributor.

$\lambda _ { \mathbf { m o t i o n } }$ is the least sensitive: the full sweep spans only 0.19% in mean AUC, with perdataset variation under 1.2%. We keep $\lambda _ { \mathrm { m o t i o n } } = 0 . 4$ as the inflection point where ShanghaiTech and UCF-Crime trends cross.

## A.2 Ablation: EMA Freezing during Anomaly Windows

The adaptive z-score in manuscript Eq. 3 maintains running statistics $\mu _ { t }$ and $\sigma _ { t }$ via EMA updates. Updating these at every frame allows sustained anomalous motion to be absorbed into the baseline: as the EMA tracks elevated TGM values during an ongoing event, the z-score decays below $\theta _ { \mathrm { a n o m } }$ and suppresses motion-gated densification mid-event. We prevent this self-suppression by freezing $\mu _ { t }$ and $\sigma _ { t }$ whenever $z ( t ) > \theta _ { \mathrm { a n o m } }$ , so the reference distribution reflects only normal background dynamics. Table 11 reports frame-level AUC-ROC with and without freezing. The mechanism yields consistent gains, largest on MSAD (+2.01 pp) and UCF-Crime (+1.95 pp) where anomalies unfold over multi-second intervals, and smaller on XD-Violence (+0.44 pp) where punctate events (gunshots, explosions) complete before the EMA can adapt. The mechanism adds no inference cost and no hyperparameters.

<table><tr><td colspan="5">Table 8:  $\lambda _ { \mathrm { a n c h o r } }$  sweep  $( \lambda _ { \mathrm { b a s e } } { = } 0 . 3 , \lambda _ { \mathrm { m o t i o n } } { = } 0 . 4 ) .$ </td></tr><tr><td> $\lambda _ { \mathrm { a n c h o r } }$ </td><td>MSAD</td><td>UCF</td><td>XD</td><td>Shang.</td></tr><tr><td>0.2</td><td>87.43</td><td>84.76</td><td>91.97</td><td>76.71</td></tr><tr><td>0.3</td><td>87.59</td><td>84.71</td><td>91.97</td><td>77.58</td></tr><tr><td>0.4</td><td>87.72</td><td>84.66</td><td>91.97</td><td>78.16</td></tr><tr><td>0.5</td><td>87.82</td><td>84.60</td><td>91.97</td><td>78.58</td></tr><tr><td>0.6</td><td>87.89</td><td>84.54</td><td>91.97</td><td>78.86</td></tr><tr><td>0.7*</td><td>87.93</td><td>84.48</td><td>91.97</td><td>79.02</td></tr></table>

<table><tr><td colspan="5">Table 9:  $\lambda _ { \mathrm { m o t i o n } }$  sweep  $( \lambda _ { \mathrm { b a s e } } { = } 0 . 3 , \lambda _ { \mathrm { a n c h o r } } { = } 0 . 7 ) .$ </td></tr><tr><td> $\lambda _ { \mathrm { m o t i o n } }$ </td><td>MSAD</td><td>UCF</td><td>XD</td><td>Shang.</td></tr><tr><td>0.2</td><td>87.92</td><td>84.61</td><td>91.96</td><td>78.25</td></tr><tr><td>0.3</td><td>87.94</td><td>84.55</td><td>91.97</td><td>78.64</td></tr><tr><td>0.4*</td><td>87.93</td><td>84.48</td><td>91.97</td><td>79.02</td></tr><tr><td>0.5</td><td>87.90</td><td>84.38</td><td>91.97</td><td>79.26</td></tr><tr><td>0.6</td><td>87.85</td><td>84.23</td><td>91.97</td><td>79.42</td></tr></table>

Table 10: Sensitivity to $\lambda _ { \mathrm { b a s e } } ~ ( \lambda _ { \mathrm { m o t i o n } } { = } 0 . 4 , \lambda _ { \mathrm { a n c h o r } } { = } 0 . 7 )$ . AUC (%). ⋆: paper configuration.
<table><tr><td> $\lambda _ { \mathrm { b a s e } }$ </td><td>MSAD</td><td>UCF</td><td>XD</td><td>Shang.</td></tr><tr><td>0.2</td><td>87.57</td><td>84.47</td><td>91.79</td><td>77.99</td></tr><tr><td>0.3*</td><td>87.93</td><td>84.48</td><td>91.97</td><td>79.02</td></tr><tr><td>0.4</td><td>88.20</td><td>84.39</td><td>92.10</td><td>79.79</td></tr><tr><td>0.5</td><td>88.40</td><td>84.20</td><td>92.17</td><td>80.35</td></tr><tr><td>0.6</td><td>88.56</td><td>83.90</td><td>92.18</td><td>80.56</td></tr></table>

## A.3 Ablation: Anomaly Threshold $\theta _ { \mathbf { a n o m } }$ Sensitivity

The motion gate in manuscript Eq. 3 triggers when the adaptive z-score exceeds $\theta _ { \mathrm { a n o m } }$ . We fix $\theta _ { \mathrm { a n o m } } = 2 . 3$ across all datasets without per-dataset tuning. This value was selected prior to running the sensitivity analysis below and was not chosen to maximize performance on any dataset. Table 12 reports sensitivity over the range [1.5,2.3]. Frame-level AUC-ROC varies by less than 1% across this range on every benchmark, confirming that the framework is insensitive to this hyperparameter. All four datasets favor lower thresholds that admit more candidates.

## A.4 Ablation: Gaussian Smoothing Strength

The fused score $\widetilde { s } _ { t }$ is passed through a 1-D Gaussian filter to suppress high-frequency noise from caption variability and tier transitions. We sweep the filter width $\sigma$ over [0.5, 3.0] s with all other settings fixed; Fig. 5 reports frame-level AUC-ROC. The optimal width is anomaly-density dependent. Sparse-anomaly ShanghaiTech degrades monotonically beyond $\sigma = 1 . 0$ , falling 2.32% to 76.24% at $\sigma = 3 . 0$ as brief events are blurred into the normal baseline. Dense-anomaly benchmarks are robust and mildly benefit from wider kernels: XD-Violence plateaus near 92.0%, MSAD peaks at 88.17%, and UCF-Crime at 84.73%, their multi-second intervals surviving while single-frame noise is removed. We fix $\sigma { = } 1 . 0$ across all datasets, as it preserves the temporal precision sparse-anomaly detection requires.

## A.5 Ablation: Prompt Sensitivity

The framework is insensitive to prompt wording. We evaluate four variants of the framelevel pipeline prompts on XD dataset (Table 13): the paper default (full structured prompt),

Table 11: Effect of EMA freezing during anomaly windows on frame-level AUC-ROC (%). Default config (Sec. 4.2).
<table><tr><td rowspan=1 colspan=1>Dataset</td><td rowspan=1 colspan=1>With freeze</td><td rowspan=1 colspan=1>No freeze</td><td rowspan=1 colspan=1>∆(%)</td></tr><tr><td rowspan=1 colspan=1>XD-Violence</td><td rowspan=1 colspan=1>84.69</td><td rowspan=1 colspan=1>84.25</td><td rowspan=1 colspan=1>+0.44</td></tr><tr><td rowspan=1 colspan=1>UCF-Crime</td><td rowspan=1 colspan=1>68.31</td><td rowspan=1 colspan=1>66.36</td><td rowspan=1 colspan=1>+1.95</td></tr><tr><td rowspan=1 colspan=1>MSAD</td><td rowspan=1 colspan=1>81.51</td><td rowspan=1 colspan=1>79.50</td><td rowspan=1 colspan=1>+2.01</td></tr></table>

Table 12: Sensitivity to $\theta _ { \mathrm { a n o m } }$ (Eq. 3). Frame-level AUC-ROC (%). Default shaded.
<table><tr><td rowspan=1 colspan=1> $\theta _ { \mathrm { a n o m } }$ </td><td rowspan=1 colspan=1>XD.</td><td rowspan=1 colspan=1>UCF</td><td rowspan=1 colspan=1>MSAD</td><td rowspan=1 colspan=1>Shang.</td></tr><tr><td rowspan=1 colspan=1>1.5</td><td rowspan=1 colspan=1>85.64</td><td rowspan=1 colspan=1>68.41</td><td rowspan=1 colspan=1>81.68</td><td rowspan=1 colspan=1>72.50</td></tr><tr><td rowspan=1 colspan=1>1.8</td><td rowspan=1 colspan=1>85.14</td><td rowspan=1 colspan=1>68.35</td><td rowspan=1 colspan=1>81.67</td><td rowspan=1 colspan=1>72.50</td></tr><tr><td rowspan=1 colspan=1>2.0</td><td rowspan=1 colspan=1>85.01</td><td rowspan=1 colspan=1>68.26</td><td rowspan=1 colspan=1>81.61</td><td rowspan=1 colspan=1>72.50</td></tr><tr><td rowspan=1 colspan=1>2.3 (default)</td><td rowspan=1 colspan=1>84.69</td><td rowspan=1 colspan=1>68.31</td><td rowspan=1 colspan=1>81.51</td><td rowspan=1 colspan=1>72.38</td></tr></table>

![](images/8f633d4eadcebf0d88333f9dc06390333a606c694162a89d7e5190ed5af8dcf1.jpg)  
Figure 5: Sensitivity of frame-level AUC-ROC to the Gaussian smoothing width $\sigma$ across datasets.

V1 (indicator lists removed), V2 (lists and rules cut, no structure), and V3 (default content reworded). AUC-ROC spans only 0.71% across all four prompts, confirming that results do not depend on a hand-tuned prompt.
<table><tr><td>Prompt variant</td><td>Final</td></tr><tr><td>Default (paper — full prompt)</td><td>91.91</td></tr><tr><td>V1 (minimal — indicator lists removed)</td><td>91.98</td></tr><tr><td>V2 (no structure — lists + rules cut)</td><td>91.34</td></tr><tr><td>V3 (rephrased — same content, reworded) 92.05</td><td></td></tr></table>

Table 13: Prompt sensitivity on XD (AUC-ROC, %).

## A.6 Ablation: Isolating the Sampling Contribution

To isolate motion-adaptive allocation, we hold the pipeline fixed and vary only the sampling schedule, replacing it with uniform sampling at 3 FPS $( r _ { \mathrm { h i g h } }$ , MoG-VLM’s densest rate). Under this setting (Table 14), MoG-VLM is 2.55–2.86× faster. Accuracy does not suffer: frame-level AUC-ROC improves on XD-Violence, UCF-Crime, and MSAD, with a small ShanghaiTech drop the full pipeline recovers. The gains thus stem from where compute is spent.

<table><tr><td>Sampling (AUC%/h)</td><td>XD</td><td>UCF</td><td>MSAD</td><td>Shang</td></tr><tr><td>Uniform 1 FPS</td><td>84.00/7.96</td><td>65.02/3.51</td><td>77.57/0.78</td><td>71.54/0.13</td></tr><tr><td>Uniform 3 FPS</td><td>84.44/23.9</td><td>65.18/10.5</td><td>77.99/2.3</td><td>73.67/0.4</td></tr><tr><td>Ours (adaptive 1→3 FPS)</td><td>84.69/8.42</td><td>68.31/4.12</td><td>81.51/0.85</td><td>72.38/0.14</td></tr></table>

Table 14: Sampling Schedule Ablation.

## B Additional Qualitative Results

Successful Cases: Fig. 6 illustrates two successful detections on MSAD dataset. In Robbery\_016, forced entry through a glass door generates sustained high-magnitude motion that continuously identifies the weapon, structural damage, and intruder context, yielding a score of 0.96 that closely tracks the ground-truth anomaly boundary. Traffic\_accident\_12 presents a semantically anomalous event with comparatively modest motion, a vehicle driven into a retail interior. The score ramps gradually before stabilizing at a high-confidence plateau. Despite the absence of conventional violence cues, the scene-level caption resolves the anomaly through contextual incongruity, a vehicle inside a non-vehicular space, producing a score of 0.92. Together, these cases demonstrate MoG capacity to handle both kinematically and semantically driven anomalies.

Failure Cases. Fig. 6 shows two failure cases. In Shang\_03\_0036, the anomaly is a person walking in circles, a subtle behavioral irregularity that generates indistinguishable from normal pedestrian activity the VLM caption interprets the scene as routine behavior, producing a score of only 0.1. This reflects a known limitation of VLMs in distinguishing trajectory-level semantics from appearance alone, rather than a failure of MOG. In Shang\_01\_0140, the VLM correctly perceives a motorcycle in a pedestrian plaza yet assigns a score of only 0.2, as the LLM fails to flag it as anomalous without explicit pedestrianonly context, a limitation of zero-shot semantic grounding rather than visual recognition.

![](images/9f13d84d7a03c80bbb749815028b46f9cbbaaaf00eeda07a94e8d3adf79d7ddb.jpg)

![](images/0356e7ac8fbe7eb3c37289e23f039a7498ad054345ad2f4cfbb1fa3cf4219dc7.jpg)

![](images/2b01fc8fbad30dca1c1c546b916c60c2f6a8a00b043fe359d68152d0555face8.jpg)

![](images/cfa03c5556db0bf7c0ec1c1dd8a9d17ebdf3716e931602a485516415529ac058.jpg)

![](images/81afe7ae54be54cbd50ed428c74eee6e53c2402fa5a5cc06c9e18b0f7e75aa36.jpg)

![](images/13f21b8089a97a62441b03d10317449de18c5ec8d52dd9ff41b0ace43d925b90.jpg)

![](images/e2f1a869837e4901b92ffb6758f4eb84e5003d775a8238c8b28b3f776a85949a.jpg)

![](images/67b31d207d7faad2b86bee691062757bba7bcaf98705842a231ec10a48dff33e.jpg)

Caption: The surveillance footage captures the interior of a jewelry store during a break-in. A single individual, appearing as a blurred silhouette due to motion, is seen entering through a shattered glass door. The person is holding a long, dark objec resembling a crowbar or similar tool, which they are using to break the glass. The scene is chaotic: glass shards are scattered across the floor, and display cases are overturned or broken, with jewelry visibly spilled or damaged. The presence of broken glass, damaged displays, and the armed intruder poses a significant safety and security concern. # Frame 210

Anomaly\_score": 0.96

Caption: The scene is an indoor retail store with bright orange and white decor, featuring product shelves, a display of portable generators, and a white car parked inside the store near the right side. In the background, several other people are visible near the far end of the store, some standing and others walking. The white car is parked in an unusual location, inside the store, with its front end near a display area that appears to have been damaged or disrupted, with debris scattered on the floor. There are no visible weapons, bags, or suspicious items. The environment is a retail store, not a pedestrian area or parking lot, and the crowd density is low. The presence of a vehicle inside a retail store is highly unusual and poses a significant safety Framed250."

Anomaly\_score": 0.92

Caption: The scene depicts five young males walking across a paved pedestrian crosswalk. The individuals are casually dressed in t-shirts and shorts, and one carries a green backpack. They are walking in different directions, not interacting with each other, and appear to be crossing the street independently. No vehicles are visible in the frame, and the area is not pedestrian-only as it includes a crosswalk and adjacent road. There are no weapons, abandoned objects, or signs of distress or unusual behavior. The environment includes a fence and some greenery, suggesting a suburban or institutional setting. Crowd density is low, with only five individuals visible.",

Anomaly\_score": 0.1

Caption: "original\_caption": "The scene depicts a wide, paved pedestrian plaza or walkway with light-colored stone tiles. There are five individuals visible: three are walking in a loose group. The pedestrians are casually dressed, carrying backpacks, and appear to be walking normally without interaction or unusual behavior. The scooter is on the paved area, which appears to be designated for pedestrians and vehicles alike (no clear barrier or signage indicating pedestrian-only). No weapons, abandoned objects, or suspicious items are visible. Crowd density is low. No vehicles other than the scooter are present, and it is not in a clearly restricted pedestrian zone.

Anomaly\_score": 0.2

Figure 6: Qualitative examples of successful detections and failure cases.

## C Prompts

The effectiveness of our motion-adaptive video anomaly detection framework relies on designed prompts that guide the Vision-Language Models (VLMs) to extract relevant information. This section details the specific prompts used at different stages of our pipeline and the rationale behind their design.

## C.1 Frame-Level Captioning Prompt

For the frame-level analysis using Qwen/Qwen3-VL-8B-Instruct, we employ a structured prompt designed to extract comprehensive scene understanding while maintaining focus on security-relevant elements. The complete prompt is:

## Frame Captioning Prompt

\- All people present: their count, positions, activities, and interactions

\- Objects of interest: bags, weapons, vehicles, or suspicious items

\- Environmental context: location type (street, pedestrian area,

parking lot, etc.), crowd density

\- Vehicles in pedestrian areas: note any cars, bicycles, skateboards,

or carts in pedestrian-only zones

Design Rationale. This prompt is structured to elicit detailed, security-focused descriptions from the VLM through several key design principles:

• Explicit Entity Enumeration: By explicitly requesting "count, positions, activities, and interactions" for people, we ensure quantitative information that is crucial for anomaly detection (e.g., crowd formation, group behaviors).

• Open-Ended Safety Assessment: The final request for "safety or security concerns" allows the model to identify potential anomalies that may not fit predefined categories, leveraging the VLM’s general knowledge of unsafe situations.

## Prompt Considerations.

Zero-Shot Generalization: The prompt avoids dataset-specific terminology, enabling the same prompt to work across different surveillance scenarios (street scenes, campus footage, traffic monitoring) without modification.

## C.2 Frame-Level Anomaly Scoring Prompt

Following the frame captioning stage, we employ Qwen/Qwen3-4B-Instruct as a specialized text-based classifier to determine whether the generated descriptions indicate normal or anomalous behavior. The system prompt is designed to provide clear classification criteria while ensuring robust, independent assessments:

![](images/d63346776370c7c4a53c7740dca957747f78e4dca1e122c4a8ea76e43bd18ad2.jpg)  
Design Rationale. We provide exhaustive lists of both anomalous and normal behaviors to reduce ambiguity and ensure consistent classification across diverse scenarios, rather than relying on the model’s implicit understanding of "anomaly."

Integration with Motion-Adaptive Sampling. This frame-level classification prompt operates on selectively sampled frames identified by our TGM-based motion detection. The design ensures that when expensive VLM inference is triggered by motion, the subsequent classification is both accurate and informative, maximizing the value extracted from each processed frame.

## C.3 Temporal-Level Multi-Frame Captioning Prompt

We employ Video-LLaMA 8B to process non-overlapping temporal windows for capturing action dynamics across multiple frames. The temporal captioning prompt is designed to elicit video-level descriptions that encode motion trajectories and sequential activities.

![](images/cd95d8b952b036d25d60bdadbae345cf8026df55591b74d894e3d868e8c634c6.jpg)

Design Rationale. This prompt architecture addresses the unique requirements of temporal video understanding:

• Motion-Centric Language: The emphasis on "actions, movements, and events" guides the model to focus on temporal dynamics rather than static scene elements, ensuring descriptions capture the behavioral evolution across frames.

• Anomaly-Focused: While maintaining open-ended description capabilities, the explicit listing of suspicious behaviors (theft, violence, fighting, etc.) ensures the model remains vigilant for security-relevant patterns that might be subtle or easily overlooked.

These carefully designed prompts enable our framework to extract security-relevant information at both frame and temporal levels while maintaining zero-shot generalization across diverse surveillance scenarios.

## D Limitations

Because TGM operates on raw temporal gradients, abrupt illumination changes (e.g., lights switching on/off) and camera jitter can produce spurious activations that erode efficiency gains. The adaptive EMA baseline $( \alpha = 0 . 9 5 )$ and z-score normalization absorb gradual drift but cannot distinguish a true motion burst from a global photometric spike within a single window. Our multiscale design $( \mathcal { W } = \{ 1 , 2 , 4 \} )$ partially mitigates this, a brief photometric spike elevates the w = 1 response but is diluted at $w = 2$ and $w = 4$ , so the aggregated signal is less likely to exceed the z-score threshold than a sustained physical motion event would.