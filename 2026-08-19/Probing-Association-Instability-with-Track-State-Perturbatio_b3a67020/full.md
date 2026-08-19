# Probing Association Instability with Track-State Perturbations for Clip-Level Active Learning in Query-Propagation Multi-Object Tracking

2Riku Inoue   
<sup>0</sup>riku.inoue@ntt.com   
<sup>2</sup>Shogo Sato   
<sup>g</sup>shg.sato@ntt.com   
Kazuhiko Murasaki   
kazuhiko.murasaki@ntt.com   
<sub>1</sub><sup>8</sup>Tomoyasu Shimada   
tomoyasu.shimada@ntt.com   
<sup>]</sup>Toshihiko Nishimura   
toshihiko.nishimura@ntt.com   
Ryuichi Tanida   
<sup>s</sup>ryuichi.tanida@ntt.com Human Informatics Laboratories NTT, Inc.   
Kanagawa, Japan

## Abstract

Training query-propagation end-to-end multi-object tracking (MOT) models requires dense bounding-box and identity annotations across video sequences, making dataset construction expensive. Clip-level active learning reduces this cost by selecting video clips for annotation, but prior acquisition criteria based on output-level temporal uncertainty may miss clips whose informativeness comes from association instability in propagated track states. We propose QPID (Query-Propagation Instability and Diversity), a clip acquisition method for query-propagation MOT that targets association instability in propagated track states. QPID estimates this instability by applying two-sided perturbations to internal track states and measuring prediction differences from a clean reference branch. The key idea is that, in stable clips, each propagated track should continue to follow the same target under small perturbations, whereas in ambiguous clips, small changes in the track state can alter which target the track follows, leading to changes in localization or confidence. QPID measures these perturbation-induced prediction differences with two metrics: Localization Drift and Entropy-Weighted Confidence Discrepancy. These metrics are aggregated into a clip-level association-instability score. To avoid redundant uncertainty-only selection, QPID selects a representative annotation batch from high-instability clips using Uncertainty-Weighted Visual Coverage with track-level visual prototypes. Experiments on DanceTrack and SportsMOT with MeMOTR and SambaMOTR show that QPID achieves strong performance compared with active learning baselines under the same annotation budget.

![](images/0f42293e2946910a7d8460436f6fa2703c97f4730a8ac0ec1849bae1db6f77f9.jpg)  
Figure 1: Association instability via two-sided perturbations to internal track states. QPID applies small two-sided perturbations (±ε) to internal track states, consisting of track query embeddings and reference points, and compares the resulting predictions with a clean branch. Stable overlap between the clean and perturbed boxes indicates low association instability, whereas localization drift or association switches indicate high association instability. Green denotes the clean box, and red and blue dashed boxes denote the perturbed boxes for the track prediction visualized in each example.

## 1 Introduction

Multi-Object Tracking (MOT) aims to localize multiple objects in a video and maintain their identities over time. For many years, MOT has been dominated by the tracking-by-detection paradigm [1, 10, 47], where objects are detected in each frame and associated across frames using motion or appearance cues. While this paradigm is highly effective on conventional pedestrian benchmarks [17, 29], it can struggle in benchmarks with irregular motion and many appearance-similar targets, such as DanceTrack [39] and SportsMOT [16]. End-to-end trackers have therefore emerged as a strong direction by maintaining identities through propagated track queries [19, 28, 36, 45]. These trackers have substantially improved tracking performance in such challenging scenes by learning to maintain identities among multiple plausible association candidates.

Despite these advances, training such trackers remains annotation-intensive. MOT supervision requires dense bounding boxes together with temporally consistent identity labels across video sequences. This cost becomes especially severe in crowded scenes, where annotators must resolve ambiguous associations over time. Moreover, videos contain strong temporal redundancy, so naive annotation sampling can waste the budget on near-duplicate temporal segments. Therefore, under a limited annotation budget, it is important to select clips that expose difficult association cases and are informative for improving identity maintenance in the target tracker.

Active Learning (AL) reduces annotation cost by prioritizing informative samples for labeling. In computer vision, AL has been widely studied for image classification [15, 37, 40] and object detection [24, 42, 44], where acquisition criteria are often based on uncertainty, representativeness, or their combination. Video AL [21, 31, 32, 38] and MOT-oriented AL [25] have also explored sample selection under limited annotation budgets, often through frame-level acquisition. However, selecting individual frames is not fully aligned with endto-end trackers, whose training samples are multi-frame clips with temporally consistent identities. Since identity maintenance in these trackers relies on temporal propagation across frames, evidence of association difficulty can be distributed across multiple frames within a clip rather than captured by a single frame. Thus, acquisition should be performed at the clip level, allowing such evidence to be aggregated across multiple frames.

CUTAL [23] addresses the acquisition-unit mismatch by formulating clip-level active learning for end-to-end MOT. By selecting fixed-length clips, CUTAL better aligns acquisition with the multi-frame training structure of end-to-end trackers than frame-level acquisition. However, its output-level temporal uncertainty mainly reflects variations in final predictions, such as entropy variation or confidence inconsistency, and does not directly probe instability in propagated track states. In query-propagation trackers, each propagated track state carries information about the target followed by the track across frames. When appearance-similar targets approach or cross, final predictions may remain confident and temporally smooth, while small changes in the propagated state can change localization or alter which target the track follows. Such instability exposes clips where the tracker must learn to maintain identities among competing targets, which may not be captured by outputlevel temporal uncertainty alone.

In this work, we propose QPID, Query-Propagation Instability and Diversity, a clip-level active learning framework for query-propagation MOT. QPID estimates association instability by applying two-sided perturbations to internal track states and measuring prediction differences from a clean reference branch. As illustrated in Fig. 1, in stable clips, perturbed predictions remain consistent with the clean prediction under small perturbations, whereas in ambiguous clips, a perturbed track state can lead to localization drift or an association change. QPID aggregates these perturbation-induced prediction differences using Localization Drift and Entropy-Weighted Confidence Discrepancy into a clip-level associationinstability score. To avoid selecting redundant high-instability clips, QPID selects a representative annotation batch using Uncertainty-Weighted Visual Coverage with track-level visual prototypes.

Our contributions are as follows.

• We propose QPID, a clip-level active learning framework for query-propagation MOT that selects informative and non-redundant clips by estimating association instability in propagated track states and improving visual coverage.

• We introduce a two-sided perturbation scheme that applies paired perturbations to internal track states and quantifies perturbation-induced association instability using Localization Drift and Entropy-Weighted Confidence Discrepancy.

• We demonstrate the effectiveness of QPID on DanceTrack and SportsMOT with MeM-OTR and SambaMOTR, showing strong performance against active learning baselines under the same annotation budget.

## 2 Related Work

## 2.1 Multi-Object Tracking

MOT has long been studied under the tracking-by-detection paradigm, where objects are detected in each frame and then associated over time. Early studies were largely centered on pedestrian benchmarks [17, 29], where pipelines combining strong detectors with post-hoc association became standard. These methods typically combine object detectors [20, 49] with association based on IoU matching [8, 43] or Re-ID features [46].

Recent benchmarks such as DanceTrack [39] and SportsMOT [16] emphasize crowded scenes, frequent occlusions, complex motion, and appearance-homogeneous targets. These settings expose limitations of frame-wise processing with heuristic association and motivate models that learn temporal association within the network. Transformer-based trackers such as TrackFormer [28] and MOTR [45] propagate track queries across frames and update them through attention to frame features. Later methods improve query-based tracking with detector priors in MOTRv2 [48], long-term memory in MeMOTR [19], and efficient long-sequence modeling in SambaMOTR [36] based on selective state-space models [22].

Overall, MOT has evolved from post-hoc association on per-frame detections to endto-end frameworks that maintain internal track states across frames. QPID leverages these propagated track states to estimate association instability for clip selection, rather than modifying the tracker architecture itself.

## 2.2 Active Learning

Active Learning aims to reduce annotation cost by selecting informative samples from an unlabeled pool. In computer vision, active learning has been studied for image classification [7, 18, 40], object detection [9, 35, 42], semantic segmentation [11, 30], and visionlanguage settings [6, 34]. Common acquisition strategies include uncertainty-based sampling [18, 40], representative selection such as core-set methods [37], and hybrid strategies that combine uncertainty and diversity [3, 15]. Recent coverage-based methods further formalize batch selection through generalized coverage and uncertainty coverage [4, 5]. QPID is related to this coverage-based perspective, but adapts it to clip-level MOT by selecting a visually representative subset from high-instability clips using track-level visual prototypes.

Perturbation- and consistency-based AL. Several methods estimate informativeness by measuring prediction changes under input perturbations or augmentations. In object detection, Localization Stability [24] measures how predicted object locations vary when input images are corrupted by noise, while CALD [44] compares predictions between original and augmented images using box and class-score consistency. These methods probe detector robustness to input-space transformations, whereas QPID perturbs internal track states of query-propagation trackers to estimate association instability.

Active learning for videos. Extending active learning to videos is challenging because samples are temporally correlated and annotations are structured across frames. Several video AL methods reduce annotation cost by selecting informative videos or frames for action detection, video classification, object detection, and video segmentation [21, 31, 32, 38, 41, 50]. However, frame-level acquisition is not fully aligned with end-to-end MOT trackers, whose training samples are multi-frame clips with temporally consistent identities. A single frame may not reveal whether the tracker can maintain identity through temporal interactions such as crossing, occlusion, and reappearance.

Active learning for MOT. Active learning for MOT has been less explored. HD-AMOT [25] formulates informative frame selection as a Markov Decision Process using heterogeneous geometric and semantic cues. It performs frame-level sampling, whereas QPID selects multiframe clips aligned with the training structure of end-to-end MOT trackers. SPAM [12] improves tracking annotation efficiency by combining pseudo labels, human correction, and a graph-based labeling model. Unlike SPAM, which is designed as a video label engine for producing tracking annotations, QPID focuses on selecting which clips should be annotated to train end-to-end trackers.

CUTAL [23] is the most directly related work, formulating clip-level active learning for end-to-end MOT with output-level temporal uncertainty and temporal diversity sampling. QPID differs from CUTAL by probing internal track states through two-sided perturbations to estimate association instability beyond final-prediction variation. It further uses tracklevel visual coverage to select informative and non-redundant clips.

## 3 Problem Definition

We follow the clip-level active learning setting introduced by CUTAL for end-to-end MOT. Given a set of training videos V, a clip pool C is constructed by sampling fixed-length clips from each video. Each clip $c \in { \mathcal { C } }$ is associated with a source video $\nu ( c ) \in \mathcal { V }$ , starts at frame $t _ { c } ,$ , and consists of T sampled frames with an intra-clip interval $\Delta \mathrm { : }$

$$
{ \mathcal { K } } ( c ) = \{ t _ { c } + j \Delta | j = 0 , . . . , T - 1 \} .\tag{1}
$$

Here, $T$ is chosen to match the standard training clip length of the target end-to-end tracker. Annotating a clip requires bounding boxes and temporally consistent identity labels for all sampled frames in $\kappa ( c )$ , so the annotation cost is proportional to $T$

Active learning proceeds over rounds. At round $r ,$ the labeled set is $\mathcal { L } _ { r } \subset \mathcal { C }$ , and the unlabeled pool is $\mathcal { U } _ { r } = \mathcal { C } \backslash \mathcal { L } _ { r }$ . The process starts from a small random seed set $\mathcal { L } _ { 0 }$ . At each round, the tracker is trained on $\mathcal { L } _ { r }$ , acquisition scores are computed for clips in $\mathcal { U } _ { r }$ , and a subset $S _ { r } \subset \mathcal { U } _ { r }$ is selected for annotation.

Given a frame-level annotation budget $b _ { r }$ , the corresponding clip budget is $B _ { r } = \lfloor b _ { r } / T \rfloor$ The selected batch satisfies $S _ { r } \subset \mathcal { U } _ { r }$ and $\left| { \cal S } _ { r } \right| = B _ { r }$ . After annotation, the labeled set and unlabeled pool are updated as $\mathcal { L } _ { r + 1 } = \mathcal { L } _ { r } \cup \mathcal { S } _ { r }$ and $\mathcal { U } _ { r + 1 } = \mathcal { U } _ { r } \backslash \mathcal { S } _ { r }$

Following CUTAL, clips that temporally overlap with already labeled clips from the same video are excluded from selection. We also enforce temporal disjointness among clips selected within the same round to avoid duplicate annotation.

## 4 Method

We propose QPID, a clip acquisition method for query-propagation MOT under the cliplevel active learning setting. While CUTAL estimates clip informativeness from output-level temporal uncertainty, QPID targets association instability in propagated track states. QPID applies two-sided perturbations to internal track states and measures perturbation-induced prediction differences from a clean reference branch. This enables QPID to prioritize clips where small changes to propagated track states produce large prediction differences, even when final predictions appear temporally smooth.

As shown in Fig. 2(b), QPID consists of two stages. First, Two-Sided Track-State Perturbation computes a clip-level association-instability score $U ( c )$ from Localization Drif and Entropy-Weighted Confidence Discrepancy. Second, QPID constructs a high-instability candidate pool and selects a representative annotation batch by Uncertainty-Weighted Visual Coverage using track-level visual prototypes.

## 4.1 Two-Sided Track-State Perturbation

In query-propagation MOT, propagated track states interact with current visual features at each frame and are updated to produce track predictions. If association is stable, small changes to the track state should produce consistent predictions for the same propagated track. In contrast, when appearance-similar targets approach or cross each other, small changes to the track state can lead to localization drift or confidence changes.

![](images/523a2f8d312d44dc5a8e5961918a0ab376d760ccd9834cc4028d30f5857bd9c2.jpg)  
Figure 2: Overview of QPID. (a) Two-sided track-state perturbation. QPID applies positive and negative perturbations along the same sampled direction to internal track states and compares the resulting predictions with a clean reference branch. Perturbed states are not propagated to subsequent frames, so the score reflects local perturbation-induced prediction differences. (b) Clip acquisition pipeline. QPID aggregates Localization Drift and Entropy-Weighted Confidence Discrepancy into a clip-level association-instability score, forms a high-instability candidate pool, and selects a representative annotation batch by Uncertainty-Weighted Visual Coverage with track-level visual prototypes.

QPID measures these changes by comparing a clean reference branch with positive and negative perturbed branches at each frame t. The clean branch uses the track state propagated from frame t −1. For track instance i, we denote the clean track query embedding by $\bar { q _ { i , t } ^ { 0 } } \in \mathbb { R } ^ { d }$ and the normalized reference point by:

$$
\tilde { r } _ { i , t } ^ { 0 } = ( \tilde { c } _ { x , i , t } ^ { 0 } , \tilde { c } _ { y , i , t } ^ { 0 } , \tilde { w } _ { i , t } ^ { 0 } , \tilde { h } _ { i , t } ^ { 0 } ) \in [ 0 , 1 ] ^ { 4 } .\tag{2}
$$

Here, $( \tilde { c } _ { x } , \tilde { c } _ { y } )$ denotes the reference-point center and $( \tilde { w } , \tilde { h } )$ denotes the reference-point width and height [26]. We refer to the pair of the track query embedding and the reference point as an internal track state. The track query embedding is a propagated track-specific representation, while the reference point provides a spatial anchor for localization. By perturbing both components, QPID probes whether the propagated track state remains stable in ambiguous association cases.

We use two-sided perturbations by applying the same sampled perturbation direction with opposite signs to probe the local response around the clean state. From the clean state $( q _ { i , t } ^ { 0 } , \tilde { r } _ { i , t } ^ { 0 } )$ , QPID constructs a positive perturbed state $( q _ { i , t } ^ { + } , \tilde { r } _ { i , t } ^ { + } )$ and a negative perturbed state $( q _ { i , t } ^ { - } , \tilde { r } _ { i , t } ^ { - } )$ The clean, positive, and negative branches are initialized from the same clean track instance i at frame t, so QPID compares predictions with the same track index across branches without additional matching (Fig. 2(a)). After scoring frame t, the perturbed branches are discarded, and only the clean state is propagated to frame t + 1. Thus, perturbations do not accumulate over time, and QPID measures local perturbation-induced prediction differences at each sampled frame.

Query-embedding perturbation. For each clip and track instance, QPID samples one perturbation direction and reuses the same direction across the sampled frames. To control the perturbation magnitude across tracks and clips, we sample $z _ { q , i } \sim \mathcal { N } ( 0 , \mathbf { I } _ { d } )$ and project it onto the $\ell _ { 2 }$ sphere with radius $\alpha _ { q }$

$$
\varepsilon _ { q , i } = \alpha _ { q } \frac { z _ { q , i } } { | | z _ { q , i } | | _ { 2 } } .\tag{3}
$$

The positive and negative perturbed queries are:

$$
q _ { i , t } ^ { + } = q _ { i , t } ^ { 0 } + \varepsilon _ { q , i } , \qquad q _ { i , t } ^ { - } = q _ { i , t } ^ { 0 } - \varepsilon _ { q , i } .\tag{4}
$$

Here, $\mathbf { I } _ { d }$ is the $d \times d$ identity matrix.

Scale-calibrated reference-point perturbation. Adding a fixed offset to the normalized reference point can introduce scale bias. The same absolute displacement corresponds to a larger relative change for small objects than for large objects. To mitigate this bias, QPID scales the reference-point perturbation by the width and height of the clean propagated reference point. We sample $z _ { \tilde { r } , i } \sim \mathcal { N } ( 0 , \mathbf { I } _ { 4 } )$ and define:

$$
\varepsilon _ { \tilde { r } , i } = \alpha _ { \tilde { r } } \frac { z _ { \tilde { r } , i } } { \lvert \lvert z _ { \tilde { r } , i } \rvert \rvert _ { 2 } } .\tag{5}
$$

The perturbed reference points are:

$$
\begin{array} { r } { \widetilde { r } _ { i , t } ^ { + } = \Pi _ { [ 0 , 1 ] ^ { 4 } } \left( \widetilde { r } _ { i , t } ^ { 0 } + \left( \tilde { w } _ { i , t } ^ { 0 } , \tilde { h } _ { i , t } ^ { 0 } , \tilde { w } _ { i , t } ^ { 0 } , \tilde { h } _ { i , t } ^ { 0 } \right) \odot \varepsilon _ { \widetilde { r } , i } \right) , } \end{array}\tag{6}
$$

$$
\begin{array} { r } { \widetilde { r } _ { i , t } ^ { - } = \Pi _ { [ 0 , 1 ] ^ { 4 } } \left( \widetilde { r } _ { i , t } ^ { 0 } - \left( \tilde { w } _ { i , t } ^ { 0 } , \widetilde { h } _ { i , t } ^ { 0 } , \tilde { w } _ { i , t } ^ { 0 } , \widetilde { h } _ { i , t } ^ { 0 } \right) \odot \varepsilon _ { \widetilde { r } , i } \right) , } \end{array}\tag{7}
$$

where $\Pi _ { [ 0 , 1 ] ^ { 4 } }$ denotes element-wise clipping to the range $[ 0 , 1 ] ^ { 4 }$ , and $\odot$ denotes element-wise multiplication.

## 4.2 Association-Aware Instability Metrics

QPID quantifies perturbation-induced prediction differences using the tracker outputs. For branch $k \in \{ 0 , + , - \}$ , let $b _ { i , t } ^ { k } = ( c _ { x , i , t } ^ { k } , c _ { y , i , t } ^ { k } , w _ { i , t } ^ { k } , h _ { i , t } ^ { k } )$ be the predicted bounding box of track instance i at frame t, and let $s _ { i , t } ^ { k }$ be its confidence score. The metrics are computed only for active tracks in the clean branch:

$$
\mathcal { T } _ { t } = \{ i \mid s _ { i , t } ^ { 0 } > \gamma _ { \mathrm { s c o r e } } \} .\tag{8}
$$

Here, $\gamma _ { \mathrm { s c o r e } }$ is the confidence threshold for selecting active clean-branch tracks, and we use $\gamma _ { \mathrm { s c o r e } } = 0 . 5$ in all experiments.

Localization Drift. Localization Drift measures how much the predicted box changes under two-sided perturbations. It is defined as the average IoU drop between the clean prediction and the perturbed predictions:

$$
D _ { i , t } = \frac { 1 } { 2 } \sum _ { k \in \{ + , - \} } \left( 1 - \mathrm { I o U } \left( b _ { i , t } ^ { 0 } , b _ { i , t } ^ { k } \right) \right) .\tag{9}
$$

A large $D _ { i , t }$ indicates that a small perturbation to the internal track state causes a large localization change.

Entropy-Weighted Confidence Discrepancy. Perturbation-induced instability can also appear as changes in confidence scores. However, raw confidence differences alone do not distinguish informative uncertainty from small fluctuations in already confident predictions.

QPID therefore weights the confidence discrepancy by the binary entropy of the clean-branch confidence:

$$
\begin{array} { r } { \mathcal { H } ( s ) = - s \log s - ( 1 - s ) \log ( 1 - s ) . } \end{array}\tag{10}
$$

The Entropy-Weighted Confidence Discrepancy is defined as:

$$
E _ { i , t } = \mathcal { H } ( s _ { i , t } ^ { 0 } ) \cdot \frac { 1 } { 2 } \sum _ { k \in \{ + , - \} } \left| s _ { i , t } ^ { 0 } - s _ { i , t } ^ { k } \right| .\tag{11}
$$

This weighting emphasizes perturbation-induced confidence changes when the clean prediction is uncertain and suppresses fluctuations for highly confident predictions.

## 4.3 Score Aggregation and Normalization

Since annotation is performed at the clip level, instance-level metrics must be aggregated into a clip-level score. QPID emphasizes hard cases within a clip by taking the maximum over active track instances at each sampled frame and averaging these values over the clip. For metric $X \in \{ D , E \}$ , let $X _ { i , t }$ be its value for track instance i at frame t. The aggregated score for clip c is:

$$
\bar { X } ( c ) = \frac { 1 } { \lvert \mathcal { K } ( c ) \rvert } \sum _ { t \in \mathcal { K } ( c ) } \operatorname* { m a x } _ { i \in \mathcal { T } _ { t } } X _ { i , t } .\tag{12}
$$

If ${ \mathcal { T } } _ { t } = \emptyset .$ , we set $\mathrm { m a x } _ { i \in \mathscr { T } _ { t } } X _ { i , t } = 0$

Following CUTAL [23], QPID normalizes each component using statistics over the unlabeled pool $\mathcal { U } _ { r }$ at each round. This normalization makes the two components comparable, since localization drift and confidence discrepancy have different ranges and distributions. For metric $X$ , let $\mu _ { X }$ and $\sigma _ { X }$ be the mean and standard deviation of $\{ \bar { X } ( c ) \} _ { c \in \mathcal { U } _ { r } }$ . We define a clipped normalization function:

$$
\phi _ { X } ( x ) = \operatorname* { m a x } \left( 0 , \operatorname* { m i n } \left( 1 , \frac { x - ( \mu _ { X } - \kappa \sigma _ { X } ) } { 2 \kappa \sigma _ { X } } \right) \right) .\tag{13}
$$

Here, κ controls the clipping range.

The final clip-level association-instability score is:

$$
U ( c ) = \phi _ { D } ( \bar { D } ( c ) ) \cdot \phi _ { E } ( \bar { E } ( c ) ) .\tag{14}
$$

We use multiplicative aggregation to prioritize clips where perturbations jointly affect localization and confidence, and analyze an additive alternative in the ablation study.

## 4.4 Track-Prototype Visual Distance

The association-instability score $U ( c )$ ranks clips by estimated association instability, but batch selection also needs to avoid visual redundancy. Since a MOT clip contains multiple targets, a single global clip feature can obscure which tracked objects are redundant. QPID therefore represents each clip by a set of track-level visual prototypes and compares clips with a symmetric weighted Chamfer distance.

We use the final-layer decoder output embedding $e _ { i , t } ^ { 0 } \in \mathbb { R } ^ { d }$ of track instance i at frame t in the clean branch. Let norm $( x ) = x / \| x \| _ { 2 }$ denote $\ell _ { 2 }$ normalization. For clip $^ { c , }$ the active frames of track instance i are:

$$
\begin{array} { r } { \mathcal { K } _ { i } ( c ) = \{ t \in \mathcal { K } ( c ) \mid s _ { i , t } ^ { 0 } > \gamma _ { \mathrm { s c o r e } } \} . } \end{array}\tag{15}
$$

If $\mathcal { K } _ { i } ( c ) \neq \emptyset$ , the track prototype is:

$$
\bar { e } _ { i } ( c ) = \mathrm { n o r m } \left( \frac { 1 } { | \mathcal { K } _ { i } ( c ) | } \sum _ { t \in \mathcal { K } _ { i } ( c ) } \mathrm { n o r m } ( e _ { i , t } ^ { 0 } ) \right) .\tag{16}
$$

To reflect prototype reliability, QPID weights each prototype by the average clean-branch confidence over its active frames:

$$
\pmb { \omega } _ { i } ( c ) = \frac { 1 } { | \mathcal { K } _ { i } ( c ) | } \sum _ { t \in \mathcal { K } _ { i } ( c ) } \boldsymbol { s } _ { i , t } ^ { 0 } .\tag{17}
$$

After normalizing weights within each clip, we obtain the weighted prototype set $\mathcal { P } ( c ) =$ $\{ ( \bar { e } _ { u } ( c ) , \bar { \omega } _ { u } ( c ) ) \} _ { u = 1 } ^ { N _ { c } }$ . Here, $N _ { c }$ denotes the number of valid prototypes, and the normalized weights satisfy $\begin{array} { r } { \sum _ { u = 1 } ^ { N _ { c } } \bar { \omega } _ { u } ( c ) = 1 } \end{array}$

For two clips $c _ { m }$ and $c _ { n }$ , the Track-Prototype Visual Distance is defined as:

$$
\begin{array} { r l r } {  { \mathrm { D i s t } _ { \mathrm { v i s } } \big ( c _ { m } , c _ { n } \big ) = \sum _ { u = 1 } ^ { N _ { c _ { m } } } \bar { \omega } _ { u } \big ( c _ { m } \big ) \operatorname* { m i n } _ { \nu \in \{ 1 , \dots , N _ { c _ { n } } \} } d _ { \cos } \big ( \bar { e } _ { u } \big ( c _ { m } \big ) , \bar { e } _ { \nu } \big ( c _ { n } \big ) \big ) } }  \\ & { } & { + \sum _ { \nu = 1 } ^ { N _ { c _ { n } } } \bar { \omega } _ { \nu } \big ( c _ { n } \big ) \operatorname* { m i n } _ { u \in \{ 1 , \dots , N _ { c _ { m } } \} } d _ { \cos } \big ( \bar { e } _ { \nu } \big ( c _ { n } \big ) , \bar { e } _ { u } \big ( c _ { m } \big ) \big ) , } \end{array}\tag{18}
$$

where $d _ { \mathrm { c o s } } ( a , b ) = 1 - a ^ { \top } b$ denotes cosine distance for normalized feature vectors. This distance measures visual redundancy at the tracked-object level, making it more suitable for multi-target MOT clips than a single global clip representation.

## 4.5 Uncertainty-Weighted Visual Coverage

Using the Track-Prototype Visual Distance, QPID selects a representative batch from clips with high association instability. The selection proceeds in two steps. First, QPID restricts the search space to high-instability clips. At round $r ,$ given the clip budget $B _ { r }$ and the candidate expansion ratio $\rho > 1$ , we form the high-instability candidate pool ${ \mathcal { C } } _ { r } ^ { \mathrm { c a n d } }$ by selecting up to $\rho B _ { r }$ clips in $\mathcal { U } _ { r }$ with the largest association-instability scores $U ( c )$

Second, QPID greedily selects clips to improve visual coverage of the high-instability candidate pool. Let $s$ denote the clips already selected in the current round, initialized as $\varnothing .$ The current reference set is defined as:

$$
\begin{array} { r } { \boldsymbol { \mathcal { A } } ( \boldsymbol { \mathcal { S } } ) = \mathcal { L } _ { \boldsymbol { r } } \cup \boldsymbol { \mathcal { S } } . } \end{array}\tag{19}
$$

For each clip $c \in \mathcal { C } _ { r } ^ { \mathrm { c a n d } }$ , its distance to the current reference set is:

$$
d _ { \cal A } ( c ) = \operatorname* { m i n } _ { a \in { \cal A } ( { \cal S } ) } \mathrm { D i s t } _ { \mathrm { v i s } } ( c , a ) .\tag{20}
$$

A larger $d _ { \boldsymbol { A } } ( \boldsymbol { c } )$ means that c is less covered by the labeled or already selected clips.

If a clip $c ^ { \prime }$ is selected, the nearest distance of each clip $c \in \mathcal { C } _ { r } ^ { \mathrm { c a n d } }$ is updated as:

$$
d _ { \mathcal { A } \cup \{ c ^ { \prime } \} } ( c ) = \operatorname* { m i n } \left\{ d _ { \mathcal { A } } ( c ) , \operatorname { D i s t } _ { \mathrm { v i s } } ( c , c ^ { \prime } ) \right\} .\tag{21}
$$

QPID measures the benefit of selecting $c ^ { \prime }$ by the uncertainty-weighted reduction of these distances:

$$
G ( c ^ { \prime } ) = \sum _ { c \in { \mathcal { C } } _ { r } ^ { \mathrm { c a n d } } } U ( c ) \left[ d _ { \cal A } ( c ) - d _ { \cal A \cup \{ c ^ { \prime } \} } ( c ) \right] .\tag{22}
$$

Thus, a clip is favored when it improves coverage for many high-instability clips.

At each greedy step, QPID selects:

$$
c ^ { * } = \underset { c ^ { \prime } \in \mathcal { C } _ { r } ^ { \mathrm { c a n d } } \backslash \mathcal { S } } { \mathrm { a r g m a x ~ } } G ( c ^ { \prime } ) .\tag{23}
$$

Clips that temporally overlap with labeled clips or already selected clips from the same video are excluded from this maximization. The selected clip $c ^ { * }$ is added to $s ,$ , and the procedure is repeated until $| { \cal S } | = B _ { r }$ . The final selected set is used as the annotation batch $S _ { r }$

## 5 Experiments

We evaluate QPID in a clip-level active learning setting for end-to-end MOT. Additional round-wise results, ablations, statistical analyses, qualitative examples, and implementation details are provided in the Supplementary Material.

## 5.1 Datasets and Evaluation Metrics

We evaluate QPID on DanceTrack [39] and SportsMOT [16]. For both datasets, the training split is used as the unlabeled pool for active learning, and all methods are evaluated on the validation split. Official test-set evaluation servers are available, but their submission limits make them impractical for multi-round active learning, which requires repeated evaluations across rounds, methods, trackers, and seeds. Therefore, we report validation results for all methods under the same splits, budgets, and seeds.

We use HOTA [27] as the primary metric and additionally report AssA [27] and IDF1 [33] to assess association quality and identity consistency. All results are averaged over three seeds.

## 5.2 Implementation Details

Trackers. We evaluate MeMOTR [19] and SambaMOTR [36]. The network architecture, loss functions, data augmentation, optimization, and learning-rate schedules follow the original papers and official configurations.

Active learning protocol. Following the clip-level protocol of CUTAL [23], we construct a clip pool C by sampling T frames with an intra-clip interval ∆ from each training video (Sec. 3). We set $T = 4$ for MeMOTR and $T = 1 0$ for SambaMOTR, with $\Delta = 5$ on both datasets. The labeling budget is defined as a percentage of the total number of training frames. At round r, a frame budget $b _ { r }$ is allocated and $B _ { r } = \lfloor b _ { r } / T \rfloor$ clips are selected for annotation. We evaluate two schedules, $5 \% + 5 \%$ and $20 \% + 1 0 \%$ , which denote the initial labeled percentage and the active learning step size, respectively.

Implementation of QPID. We set the query-embedding perturbation magnitude to $\alpha _ { q } = 0 . 5$ and the reference-point perturbation magnitude to $\alpha _ { \tilde { r } } = 0 . 0 1$ . For the normalization, we use $\kappa = 3$ for clipping. For Uncertainty-Weighted Visual Coverage, we set the candidate expansion ratio to $\rho = 4$ . All hyperparameters are fixed across rounds, datasets, and trackers.

![](images/388075b152d543989b1c90435243079b9717ff1a07d3b862ee58f19fc181aec8.jpg)  
Figure 3: Round-wise performance on DanceTrack under the 5%+5% schedule. The upper and lower rows show results using MeMOTR and SambaMOTR, respectively. From left to right, we report HOTA, AssA, and IDF1 of QPID and baseline methods.

## 5.3 Baselines

We compare QPID with representative random, uncertainty-based, representativeness-based, gradient-based, and clip-level MOT acquisition strategies under the same protocol. For methods originally designed for image- or frame-level acquisition, frame-level scores or embeddings are aggregated within each clip. Further implementation details are provided in the Supplementary Material.

Random. Clips are sampled uniformly at random from the unlabeled pool.

Entropy. Entropy is an uncertainty-based baseline. For each frame, we compute the entropy of the predicted confidence scores of active tracks, take the maximum entropy over active tracks, and average the frame scores over the clip.

Core-set [37]. Core-set is a representativeness-based baseline. We construct a clip embedding by averaging decoder last-layer track-query embeddings over active tracks and sampled frames, and apply k-Center Greedy in this embedding space.

BADGE [3]. BADGE combines uncertainty and diversity through gradient embeddings. We compute query-level gradient embeddings using the model’s current predictions as hypothetical labels, aggregate them over active tracks and sampled frames to form a clip representation, and select a diverse batch via k-means++ [2].

CUTAL [23]. CUTAL is the most directly related clip-level active learning baseline for endto-end MOT. It selects clips using output-level temporal uncertainty and temporal diversity under the same clip-level protocol.

![](images/f55f201986e4f2c793588c2317a910832a7cf057a701074f21d32f41dd998584.jpg)  
Figure 4: Round-wise performance on SportsMOT under the 5%+5% schedule. The upper and lower rows show results using MeMOTR and SambaMOTR, respectively.

## 5.4 Quantitative Results

DanceTrack. Fig. 3 summarizes the round-wise learning curves under the 5%+5% schedule for MeMOTR (upper) and SambaMOTR (lower). For MeMOTR, QPID performs strongly across budgets and improves association-oriented metrics, although CUTAL remains competitive in several rounds. Under the 20%+10% schedule in Tab. 1, QPID is the best on all metrics from 40% onward. At 50%, QPID outperforms the second-best method by +0.68 HOTA, +1.61 AssA, and +1.30 IDF1, narrowing the gap to full supervision to 0.72 HOTA, 0.91 AssA, and 0.52 IDF1.

For SambaMOTR, QPID is less dominant in the earliest 5%+5% rounds, where Entropy and CUTAL remain competitive. Nevertheless, QPID becomes the best from 25% onward and achieves the best final performance at 30%, improving over the best baseline by +0.28 HOTA, +1.64 AssA, and +1.21 IDF1. Under the 20%+10% schedule in Tab. 1, QPID is again the best on all metrics from 40% onward. At 50%, it improves over the second-best method by +0.93 HOTA, +0.88 AssA, and +0.73 IDF1, while remaining below full supervision. A post-hoc GT-based analysis further confirms that the QPID score is positively associated with clip-level association difficulty, with detailed definitions and results provided in the Supplementary Material.

SportsMOT. Fig. 4 summarizes the round-wise learning curves under the 5%+5% schedule for MeMOTR (upper) and SambaMOTR (lower). For MeMOTR, QPID achieves the best HOTA and IDF1 across all annotation budgets, and obtains the best AssA from 10% to 25% while remaining second best at 30%. Tab. 2 shows that at 50%, QPID improves over the best baseline by +0.43 HOTA, +0.57 AssA, and +0.04 IDF1, achieving performance comparable

Table 1: Round-wise results on DanceTrack under the 20%+10% schedule. Best results are marked in bold and second best are underlined.
<table><tr><td rowspan="2">Method</td><td colspan="3">HOTA</td><td colspan="3">AssA</td><td colspan="3">IDF1</td></tr><tr><td>30%</td><td>40%</td><td>50%</td><td>30%</td><td>40%</td><td>50%</td><td>30%</td><td>40%</td><td>50%</td></tr><tr><td colspan="10">MeMOTR</td></tr><tr><td>Random</td><td>57.94</td><td>58.79</td><td>59.24</td><td>46.03</td><td>47.62</td><td>48.04</td><td>60.21</td><td>61.10</td><td>61.58</td></tr><tr><td>Entropy</td><td>59.21</td><td>60.67</td><td>60.52</td><td>48.41</td><td>49.89</td><td>49.86</td><td>61.68</td><td>63.45</td><td>62.81</td></tr><tr><td>Core-set</td><td>58.03</td><td>59.09</td><td>59.83</td><td>46.53</td><td>47.93</td><td>48.93</td><td>60.57</td><td>61.37</td><td>62.21</td></tr><tr><td>BADGE</td><td>57.91</td><td>59.86</td><td>60.55</td><td>46.18</td><td>48.86</td><td>49.97</td><td>60.32</td><td>61.72</td><td>62.79</td></tr><tr><td>CUTAL</td><td>58.87</td><td>60.05</td><td>61.23</td><td>48.55</td><td>49.17</td><td>50.60</td><td>62.11</td><td>62.21</td><td>63.49</td></tr><tr><td>QPID</td><td>59.91</td><td>61.01</td><td>61.91</td><td>49.09</td><td>50.54</td><td>52.21</td><td>62.03</td><td>63.98</td><td>64.79</td></tr><tr><td>Full supervision</td><td></td><td>62.63</td><td></td><td></td><td>53.12</td><td></td><td></td><td>65.31</td><td></td></tr><tr><td colspan="10">SambaMOTR</td></tr><tr><td>Random</td><td>48.99</td><td>51.69</td><td>50.41</td><td>38.91</td><td>41.44</td><td>39.99</td><td>52.05</td><td>54.68</td><td>53.39</td></tr><tr><td>Entropy</td><td>51.51</td><td>51.81</td><td>53.48</td><td>41.89</td><td>42.48</td><td>43.33</td><td>54.74</td><td>55.30</td><td>56.64</td></tr><tr><td>Core-set</td><td>49.57</td><td>51.62</td><td>53.84</td><td>41.07</td><td>42.16</td><td>44.91</td><td>53.79</td><td>55.86</td><td>58.24</td></tr><tr><td>BADGE</td><td>50.27</td><td>51.65</td><td>52.63</td><td>40.90</td><td>42.06</td><td>43.16</td><td>54.06</td><td>55.31</td><td>56.44</td></tr><tr><td>CUTAL</td><td>50.42</td><td>52.33</td><td>54.08</td><td>41.20</td><td>42.19</td><td>44.49</td><td>54.17</td><td>55.68</td><td>57.84</td></tr><tr><td>QPID</td><td>50.82</td><td>52.55</td><td>55.01</td><td>43.19</td><td>43.65</td><td>45.79</td><td>55.38</td><td>56.75</td><td>58.97</td></tr><tr><td>Full supervision</td><td></td><td>61.86</td><td></td><td></td><td>52.23</td><td></td><td></td><td>65.80</td><td></td></tr></table>

to the full-supervision reference.

For SambaMOTR, QPID is not the best in the earliest 5%+5% rounds, where Entropy is best at 10% and CUTAL remains strong at 15% and 20%. QPID becomes the best in HOTA from 25% onward and achieves the best final performance at 30%, improving over the best baseline by +0.03 HOTA, +0.35 AssA, and +0.42 IDF1. This early-round behavior is consistent with the cold-start issue in active learning [13, 14] and is further analyzed in the Supplementary Material. Since the annotation budget is defined at the frame level, SambaMOTR’s longer clips (T=10) yield fewer selected clips than MeMOTR’s shorter clips (T=4) under the same budget. This can reduce early coverage of diverse scenes and interactions, making the learned internal track states less reliable for perturbation-based acquisition. Under the 20%+10% schedule in Tab. 2, QPID is the best on all metrics from 40% onward for SambaMOTR. At 50%, it improves over the second-best method by +1.23 HOTA, +1.13 AssA, and +1.10 IDF1, showing stronger association performance than the baselines under the same frame budget.

## 5.5 Ablation Study

Tab. 3 ablates the two instability metrics and Uncertainty-Weighted Visual Coverage, and compares internal track-state perturbation with image-space perturbation and multiplicative with additive aggregation. Image Gaussian Noise adds Gaussian noise to the input images instead of perturbing internal track states.

Removing Localization Drift decreases HOTA, AssA, and IDF1 by 0.96, 1.16, and 1.03 points, respectively, while removing Entropy-Weighted Confidence Discrepancy decreases them by 0.65, 0.82, and 0.43 points. Removing Uncertainty-Weighted Visual Coverage also reduces all three metrics, confirming the benefit of visual coverage. Image Gaussian Noise performs substantially worse than QPID, supporting internal track-state perturbations over generic input-space noise. Finally, additive aggregation performs worse than multiplicative aggregation, supporting the joint use of localization and confidence responses.

Table 2: Round-wise results on SportsMOT under the 20%+10% schedule.
<table><tr><td rowspan="2">Method</td><td colspan="3">HOTA</td><td colspan="3">AssA</td><td colspan="3">IDF1</td></tr><tr><td>30%</td><td>40%</td><td>50%</td><td>30%</td><td>40%</td><td>50%</td><td>30%</td><td>40%</td><td>50%</td></tr><tr><td colspan="10">MeMOTR</td></tr><tr><td>Random</td><td>71.81</td><td>73.22</td><td>73.77</td><td>61.49</td><td>63.28</td><td>64.48</td><td>74.48</td><td>75.98</td><td>76.71</td></tr><tr><td>Entropy</td><td>71.66</td><td>73.33</td><td>74.06</td><td>61.17</td><td>63.54</td><td>64.86</td><td>74.33</td><td>76.34</td><td>77.13</td></tr><tr><td>Core-set</td><td>72.87</td><td>73.27</td><td>73.69</td><td>62.80</td><td>63.43</td><td>64.34</td><td>75.56</td><td>76.10</td><td>76.94</td></tr><tr><td>BADGE</td><td>71.86</td><td>72.89</td><td>73.35</td><td>61.63</td><td>62.94</td><td>63.74</td><td>74.62</td><td>75.60</td><td>76.22</td></tr><tr><td>CUTAL</td><td>72.29</td><td>74.22</td><td>74.23</td><td>62.28</td><td>65.16</td><td>65.17</td><td>75.25</td><td>77.67</td><td>77.72</td></tr><tr><td>QPID</td><td>73.20</td><td>74.10</td><td>74.66</td><td>63.47</td><td>64.76</td><td>65.74</td><td>76.32</td><td>77.39</td><td>77.76</td></tr><tr><td>Full supervision</td><td></td><td>74.47</td><td></td><td></td><td>65.54</td><td></td><td></td><td>77.47</td><td></td></tr><tr><td colspan="10">SambaMOTR</td></tr><tr><td>Random</td><td>62.28</td><td>64.08</td><td>63.55</td><td>52.87</td><td>54.10</td><td>53.37</td><td>65.98</td><td>67.58</td><td>66.76</td></tr><tr><td>Entropy</td><td>64.18</td><td>62.78</td><td>67.38</td><td>54.16</td><td>53.56</td><td>58.27</td><td>67.61</td><td>66.60</td><td>71.29</td></tr><tr><td>Core-set</td><td>61.51</td><td>63.39</td><td>67.41</td><td>52.99</td><td>53.54</td><td>57.46</td><td>65.82</td><td>66.86</td><td>70.69</td></tr><tr><td>BADGE</td><td>63.97</td><td>65.18</td><td>66.43</td><td>53.83</td><td>55.25</td><td>56.45</td><td>67.16</td><td>68.61</td><td>69.67</td></tr><tr><td>CUTAL</td><td>63.78</td><td>65.94</td><td>66.94</td><td>53.67</td><td>56.11</td><td>57.25</td><td>67.26</td><td>69.40</td><td>69.99</td></tr><tr><td>QPID</td><td>64.13</td><td>66.78</td><td>68.64</td><td>54.00</td><td>56.63</td><td>59.40</td><td>67.38</td><td>70.14</td><td>72.39</td></tr><tr><td>Full supervision</td><td></td><td>77.27</td><td></td><td></td><td>69.51</td><td></td><td></td><td>80.96</td><td></td></tr></table>

Table 3: Ablations and perturbation-space comparison on DanceTrack (MeMOTR) under the 5%+5% schedule. Results are averaged over three seeds, with Avg denoting the mean over 10% to 30%.
<table><tr><td>Variant</td><td>HOTA (Avg)</td><td>AssA (Avg)</td><td>IDF1 (Avg)</td></tr><tr><td>w/o Localization Drift</td><td>57.30</td><td>45.94</td><td>59.76</td></tr><tr><td>w/o Entropy-Weighted Conf. Discrep.</td><td>57.61</td><td>46.28</td><td>60.36</td></tr><tr><td>w/o Uncertainty-Weighted Visual Coverage</td><td>57.79</td><td>46.56</td><td>60.48</td></tr><tr><td>Image Gaussian Noise</td><td>56.65</td><td>44.84</td><td>58.88</td></tr><tr><td>Sum</td><td>57.70</td><td>46.57</td><td>60.38</td></tr><tr><td>QPID</td><td>58.26</td><td>47.10</td><td>60.79</td></tr></table>

## 6 Conclusion

We proposed QPID, a clip acquisition method for query-propagation MOT that estimates association instability through two-sided perturbations of internal track states and selects representative high-instability clips using Uncertainty-Weighted Visual Coverage. Experiments on DanceTrack and SportsMOT with MeMOTR and SambaMOTR demonstrate strong performance against active learning baselines under the same annotation budget. QPID can under-rank clips in which only one instability component is large and requires 1.18– 1.19× the acquisition time of CUTAL, while leaving deployment-time tracking inference unchanged. More robust aggregation and efficient instability estimation remain directions for future work.

## References

[1] Nir Aharon, Roy Orfaig, and Ben-Zion Bobrovsky. Bot-sort: Robust associations multi-pedestrian tracking. arXiv preprint arXiv:2206.14651, 2022.

[2] David Arthur and Sergei Vassilvitskii. k-means++: the advantages of careful seeding. In Proceedings of the Eighteenth Annual ACM-SIAM Symposium on Discrete Algorithms, page 1027–1035, 2007.

[3] Jordan T. Ash, Chicheng Zhang, Akshay Krishnamurthy, John Langford, and Alekh Agarwal. Deep batch active learning by diverse, uncertain gradient lower bounds. In International Conference on Learning Representations, 2020.

[4] Wonho Bae, Junhyug Noh, and Danica J Sutherland. Generalized coverage for more robust low-budget active learning. In European conference on computer vision, pages 318–334. Springer, 2024.

[5] Wonho Bae, Danica Sutherland, and Gabriel Oliveira. Uncertainty herding: One active learning method for all label budgets. In International Conference on Learning Representations, volume 2025, pages 788–805, 2025.

[6] Jihwan Bang, Sumyeong Ahn, and Jae-Gil Lee. Active prompt learning in vision language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 27004–27014, 2024.

[7] William H Beluch, Tim Genewein, Andreas Nürnberger, and Jan M Köhler. The power of ensembles for active learning in image classification. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 9368–9377, 2018.

[8] Alex Bewley, Zongyuan Ge, Lionel Ott, Fabio Ramos, and Ben Upcroft. Simple online and realtime tracking. In 2016 IEEE international conference on image processing (ICIP), pages 3464–3468. Ieee, 2016.

[9] Clemens-Alexander Brust, Christoph Käding, and Joachim Denzler. Active learning for deep object detection. In Proceedings of the 14th International Joint Conference on Computer Vision, Imaging and Computer Graphics Theory and Applications (VISI-GRAPP 2019) - Volume 5: VISAPP, pages 181–190, 2019.

[10] Jinkun Cao, Jiangmiao Pang, Xinshuo Weng, Rawal Khirodkar, and Kris Kitani. Observation-centric sort: Rethinking sort for robust multi-object tracking. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 9686–9696, 2023.

[11] Arantxa Casanova, Pedro O. Pinheiro, Negar Rostamzadeh, and Christopher J. Pal. Reinforced active learning for image segmentation. In International Conference on Learning Representations, 2020.

[12] Orcun Cetintas, Tim Meinhardt, Guillem Brasó, and Laura Leal-Taixé. Spamming labels: Efficient annotations for the trackers of tomorrow. In European Conference on Computer Vision, pages 377–395. Springer, 2024.

[13] Liangyu Chen, Yutong Bai, Siyu Huang, Yongyi Lu, Bihan Wen, Alan L Yuille, and Zongwei Zhou. Making your first choice: To address cold start problem in vision active learning. arXiv preprint arXiv:2210.02442, 2022.

[14] Liangyu Chen, Yutong Bai, Siyu Huang, Yongyi Lu, Bihan Wen, Alan Yuille, and Zongwei Zhou. Making your first choice: to address cold start problem in medical active learning. In Medical Imaging with Deep Learning, pages 496–525. PMLR, 2024.

[15] Gui Citovsky, Giulia DeSalvo, Claudio Gentile, Lazaros Karydas, Anand Rajagopalan, Afshin Rostamizadeh, and Sanjiv Kumar. Batch active learning at scale. Advances in Neural Information Processing Systems, 34:11933–11944, 2021.

[16] Yutao Cui, Chenkai Zeng, Xiaoyu Zhao, Yichun Yang, Gangshan Wu, and Limin Wang. Sportsmot: A large multi-object tracking dataset in multiple sports scenes. In Proceedings of the IEEE/CVF international conference on computer vision, pages 9921–9931, 2023.

[17] Patrick Dendorfer, Hamid Rezatofighi, Anton Milan, Javen Shi, Daniel Cremers, Ian Reid, Stefan Roth, Konrad Schindler, and Laura Leal-Taixé. Mot20: A benchmark for multi object tracking in crowded scenes. arXiv preprint arXiv:2003.09003, 2020.

[18] Melanie Ducoffe and Frederic Precioso. Adversarial active learning for deep networks: a margin based approach. arXiv preprint arXiv:1802.09841, 2018.

[19] Ruopeng Gao and Limin Wang. Memotr: Long-term memory-augmented transformer for multi-object tracking. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 9901–9910, 2023.

[20] Zheng Ge, Songtao Liu, Feng Wang, Zeming Li, and Jian Sun. Yolox: Exceeding yolo series in 2021. arXiv preprint arXiv:2107.08430, 2021.

[21] Debanjan Goswami and Shayok Chakraborty. Active learning for video classification with frame level queries. In 2023 International Joint Conference on Neural Networks (IJCNN), pages 1–9. IEEE, 2023.

[22] Albert Gu and Tri Dao. Mamba: Linear-time sequence modeling with selective state spaces. In First Conference on Language Modeling, 2024.

[23] Riku Inoue, Shogo Sato, Kazuhiko Murasaki, Tomoyasu Shimada, Toshihiko Nishimura, and Ryuichi Tanida. Clip-level uncertainty and temporal-aware active learning for end-to-end multi-object tracking. arXiv preprint arXiv:2605.09858, 2026.

[24] Chieh-Chi Kao, Teng-Yok Lee, Pradeep Sen, and Ming-Yu Liu. Localization-aware active learning for object detection. In Asian Conference on Computer Vision, pages 506–522. Springer, 2018.

[25] Rui Li, Baopeng Zhang, Jun Liu, Wei Liu, Jian Zhao, and Zhu Teng. Heterogeneous diversity driven active learning for multi-object tracking. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 9932–9941, 2023.

[26] Shilong Liu, Feng Li, Hao Zhang, Xiao Yang, Xianbiao Qi, Hang Su, Jun Zhu, and Lei Zhang. Dab-detr: Dynamic anchor boxes are better queries for detr. In International Conference on Learning Representations, 2022.

[27] Jonathon Luiten, Aljosa Osep, Patrick Dendorfer, Philip Torr, Andreas Geiger, Laura Leal-Taixé, and Bastian Leibe. Hota: A higher order metric for evaluating multi-object tracking. International journal ofcomputer vision, 129(2):548–578, 2021.

[28] Tim Meinhardt, Alexander Kirillov, Laura Leal-Taixe, and Christoph Feichtenhofer. Trackformer: Multi-object tracking with transformers. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8844–8854, 2022.

[29] Anton Milan, Laura Leal-Taixé, Ian Reid, Stefan Roth, and Konrad Schindler. Mot16: A benchmark for multi-object tracking. arXiv preprint arXiv:1603.00831, 2016.

[30] Sudhanshu Mittal, Joshua Niemeijer, Jörg P Schäfer, and Thomas Brox. Best practices in active learning for semantic segmentation. In DAGM German Conference on Pattern Recognition, pages 427–442. Springer, 2023.

[31] Aayush Rana and Yogesh Rawat. Are all frames equal? active sparse labeling for video action detection. Advances in Neural Information Processing Systems, 35:14358– 14373, 2022.

[32] Aayush J Rana and Yogesh S Rawat. Hybrid active learning via deep clustering for video action detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18867–18877, 2023.

[33] Ergys Ristani, Francesco Solera, Roger Zou, Rita Cucchiara, and Carlo Tomasi. Performance measures and a data set for multi-target, multi-camera tracking. In European Conference on Computer Vision, pages 17–35. Springer, 2016.

[34] Bardia Safaei and Vishal M Patel. Active learning for vision-language models. In 2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), pages 4902–4912. IEEE, 2025.

[35] Sebastian Schmidt, Qing Rao, Julian Tatsch, and Alois Knoll. Advanced active learning strategies for object detection. In 2020 IEEE intelligent vehicles symposium (IV), pages 871–876. IEEE, 2020.

[36] Mattia Segu, Luigi Piccinelli, Siyuan Li, Yung-Hsu Yang, Bernt Schiele, and Luc Van Gool. Samba: Synchronized set-of-sequences modeling for multiple object tracking. In The Thirteenth International Conference on Learning Representations, pages 30057–30070. ICLR, 2025.

[37] Ozan Sener and Silvio Savarese. Active learning for convolutional neural networks: A core-set approach. In International Conference on Learning Representations, 2018.

[38] Ayush Singh, Aayush J Rana, Akash Kumar, Shruti Vyas, and Yogesh Singh Rawat. Semi-supervised active learning for video action detection. In Proceedings ofthe AAAI Conference on Artificial Intelligence, pages 4891–4899, 2024.

[39] Peize Sun, Jinkun Cao, Yi Jiang, Zehuan Yuan, Song Bai, Kris Kitani, and Ping Luo. Dancetrack: Multi-object tracking in uniform appearance and diverse motion. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 20993–21002, 2022.

[40] Keze Wang, Dongyu Zhang, Ya Li, Ruimao Zhang, and Liang Lin. Cost-effective active learning for deep image classification. IEEE Transactions on Circuits and Systems for Video Technology, 27(12):2591–2600, 2016.

[41] Fei Wu, Pablo Marquez-Neila, Mingyi Zheng, Hedyeh Rafii-Tari, and Raphael Sznitman. Correlation-aware active learning for surgery video segmentation. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 2010– 2020, 2024.

[42] Chenhongyi Yang, Lichao Huang, and Elliot J Crowley. Plug and play active learning for object detection. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 17784–17793, 2024.

[43] Fan Yang, Shigeyuki Odashima, Shoichi Masui, and Shan Jiang. Hard to track objects with irregular motions and similar appearances? make it easier by buffering the matching space. In Proceedings ofthe IEEE/CVF winter conference on applications of computer vision, pages 4799–4808, 2023.

[44] Weiping Yu, Sijie Zhu, Taojiannan Yang, and Chen Chen. Consistency-based active learning for object detection. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pages 3951–3960, 2022.

[45] Fangao Zeng, Bin Dong, Yuang Zhang, Tiancai Wang, Xiangyu Zhang, and Yichen Wei. Motr: End-to-end multiple-object tracking with transformer. In European conference on computer vision, pages 659–675. Springer, 2022.

[46] Yifu Zhang, Chunyu Wang, Xinggang Wang, Wenjun Zeng, and Wenyu Liu. Fairmot: On the fairness of detection and re-identification in multiple object tracking. International journal ofcomputer vision, 129(11):3069–3087, 2021.

[47] Yifu Zhang, Peize Sun, Yi Jiang, Dongdong Yu, Fucheng Weng, Zehuan Yuan, Ping Luo, Wenyu Liu, and Xinggang Wang. Bytetrack: Multi-object tracking by associating every detection box. In European conference on computer vision, pages 1–21. Springer, 2022.

[48] Yuang Zhang, Tiancai Wang, and Xiangyu Zhang. Motrv2: Bootstrapping end-to-end multi-object tracking by pretrained object detectors. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 22056–22065, 2023.

[49] Xingyi Zhou, Dequan Wang, and Philipp Krähenbühl. Objects as points. arXiv preprint arXiv:1904.07850, 2019.

[50] Javad Zolfaghari Bengar, Abel Gonzalez-Garcia, Gabriel Villalonga, Bogdan Raducanu, Hamed Habibi Aghdam, Mikhail Mozerov, Antonio M Lopez, and Joost Van de Weijer. Temporal coherence for active learning in videos. In Proceedings of the IEEE/CVF international conference on computer vision workshops, 2019.

Supplementary Material:

# Probing Association Instability with Track-State Perturbations for Clip-Level Active Learning in Query-Propagation Multi-Object Tracking

Riku Inoue

Human Informatics Laboratories

riku.inoue@ntt.com

NTT, Inc.

Shogo Sato

Kanagawa, Japan

shg.sato@ntt.com

Kazuhiko Murasaki

kazuhiko.murasaki@ntt.com

Tomoyasu Shimada

tomoyasu.shimada@ntt.com

Toshihiko Nishimura

toshihiko.nishimura@ntt.com

Ryuichi Tanida

ryuichi.tanida@ntt.com

## A Supplementary Overview

This Supplementary Material provides additional baseline implementation details, complete round-wise results under the 5%+5% schedule, further analyses, and qualitative visualizations.

Sec. B provides additional implementation details of the baselines under our clip-level protocol for query-propagation end-to-end trackers. Sec. C reports complete round-wise results under the 5%+5% schedule. Sec. D presents extended ablation studies that validate key design choices in QPID. Sec. E provides further analyses of association difficulty, statistical stability, hyperparameters, diversity-aware selection through Uncertainty-Weighted Visual Coverage, cold-start behavior in early rounds, the gap to full supervision in SambaMOTR [36], association-oriented improvements over CUTAL [23], and acquisition sampling time. Sec. F offers qualitative visualizations of perturbation-induced prediction instability.

Table 4: Round-wise results on DanceTrack under the 5%+5% schedule. Best results are marked in bold and second best are underlined.
<table><tr><td rowspan="2">Method</td><td colspan="5">HOTA</td><td colspan="5">AssA</td><td colspan="5">IDF1</td></tr><tr><td>10%</td><td>15%</td><td>20%</td><td>25%</td><td>30%</td><td>10%</td><td>15%</td><td>20%</td><td>25%</td><td>30%</td><td>10%</td><td>15%</td><td>20%</td><td>25%</td><td>30%</td></tr><tr><td colspan="12">MeMOTR</td><td></td><td></td><td></td><td></td></tr><tr><td>Random</td><td>54.02</td><td>54.65</td><td>54.98</td><td>55.91</td><td>57.52</td><td>41.79</td><td>42.49</td><td>43.05</td><td>43.64</td><td>45.81</td><td>56.54</td><td>56.36</td><td>57.33</td><td>57.98</td><td>59.55</td></tr><tr><td>Entropy</td><td>53.45</td><td>55.95</td><td>56.40</td><td>58.81</td><td>59.44</td><td>41.88</td><td>44.53</td><td>44.42</td><td>47.73</td><td>48.40</td><td>56.02</td><td>58.53</td><td>58.31</td><td>61.16</td><td>62.14</td></tr><tr><td>Core-set</td><td>52.49</td><td>54.18</td><td>55.16</td><td>55.80</td><td>57.78</td><td>39.77</td><td>41.61</td><td>42.86</td><td>44.05</td><td>46.01</td><td>54.55</td><td>55.94</td><td>57.25</td><td>58.28</td><td>59.64</td></tr><tr><td>BADGE</td><td>53.27</td><td>55.29</td><td>55.83</td><td>58.58</td><td>58.65</td><td>40.69</td><td>43.53</td><td>44.29</td><td>47.31</td><td>47.24</td><td>55.20</td><td>58.16</td><td>58.37</td><td>60.74</td><td>60.83</td></tr><tr><td>CUTAL</td><td>54.04</td><td>57.62</td><td>58.61</td><td>59.91</td><td>60.45</td><td>42.03</td><td>45.47</td><td>47.79</td><td>48.84</td><td>50.25</td><td>56.60</td><td>59.44</td><td>60.88</td><td>62.87</td><td>62.75</td></tr><tr><td>QPID</td><td>54.56</td><td>57.30</td><td>58.90</td><td>60.05</td><td>60.51</td><td>42.24</td><td>45.82</td><td>48.11</td><td>49.45</td><td>49.90</td><td>56.83</td><td>59.74</td><td>61.64</td><td>62.63</td><td>63.13</td></tr><tr><td colspan="12">SambaMOTR</td><td></td></tr><tr><td>Random</td><td>40.97</td><td>41.65</td><td>43.15</td><td>43.32</td><td>43.60</td><td>31.37</td><td>30.99</td><td>32.51</td><td>32.33</td><td>32.71</td><td>44.09</td><td>43.41</td><td>45.58</td><td>45.21</td><td>45.67</td></tr><tr><td>Entropy</td><td>42.92</td><td>46.76</td><td>46.76</td><td>48.02</td><td>50.12</td><td>32.74</td><td>37.48</td><td>37.58</td><td>38.40</td><td>39.90</td><td>45.77</td><td>50.65</td><td>50.55</td><td>51.64</td><td>53.11</td></tr><tr><td>Core-set</td><td>42.88</td><td>43.39</td><td>42.78</td><td>46.81</td><td>46.95</td><td>32.72</td><td>34.00</td><td>34.89</td><td>37.84</td><td>38.48</td><td>46.30</td><td>47.48</td><td>47.88</td><td>50.48</td><td>51.79</td></tr><tr><td>BADGE</td><td>40.82</td><td>46.37</td><td>45.62</td><td>49.28</td><td>48.77</td><td>32.03</td><td>36.34</td><td>37.91</td><td>39.17</td><td>39.63</td><td>44.82</td><td>49.97</td><td>50.23</td><td>53.16</td><td>52.49</td></tr><tr><td>CUTAL</td><td>44.40</td><td>44.95</td><td>48.46</td><td>47.13</td><td>50.58</td><td>34.78</td><td>35.63</td><td>39.57</td><td>38.62</td><td>40.26</td><td>47.81</td><td>48.74</td><td>52.56</td><td>51.70</td><td>53.74</td></tr><tr><td>QPID</td><td>43.60</td><td>46.19</td><td>48.61</td><td>50.21</td><td>50.86</td><td>34.12</td><td>37.73</td><td>38.91</td><td>41.96</td><td>41.90</td><td>47.21</td><td>50.39</td><td>52.50</td><td>55.09</td><td>54.95</td></tr></table>

## B Additional Baseline Implementation Details

Entropy. We compute entropy on the predicted confidence of each active track. Active tracks are defined by a fixed confidence threshold of 0.5. At each frame, we take the maximum entropy over active tracks, and obtain a clip score by averaging these frame scores over the sampled frames in the clip.

Core-set [37]. We construct a clip embedding by averaging decoder last-layer embeddings of active track queries over tracks and sampled frames. We then apply k-Center Greedy in this clip embedding space using cosine distance.

BADGE [3]. We adapt BADGE by constructing gradient embeddings for track queries and selecting a diverse batch via k-means++ [2]. Pseudo labels are obtained from the predicted confidences using a fixed threshold of 0.5. Query-level gradient embeddings are summed over active queries and sampled frames to form a clip representation, and k-means++ seeding is performed in this clip representation space.

## C Complete Quantitative Results

This section reports complete round-wise results under the 5%+5% schedule on Dance-Track [39] and SportsMOT [16] using MeMOTR [19] and SambaMOTR [36]. We report HOTA, AssA, and IDF1, and all results are averaged over three seeds. The corresponding results under the 20%+10% schedule are reported in the main paper.

Tab. 4 reports the complete 5%+5% results on DanceTrack. Tab. 5 reports the complete 5%+5% results on SportsMOT.

## D Extended Ablation Studies

Tab. 6 reports the full round-wise ablations and perturbation-space comparisons of QPID on DanceTrack with MeMOTR under the 5%+5% schedule, complementing the compact average-only ablation results in the main paper. Overall, the full QPID achieves the best average performance across all three metrics, confirming the effectiveness of the full design.

Table 5: Round-wise results on SportsMOT under the 5%+5% schedule. Best results are marked in bold and second best are underlined.
<table><tr><td rowspan="2">Method</td><td colspan="5">HOTA</td><td colspan="5">AssA</td><td colspan="5">IDF1</td></tr><tr><td>10%</td><td>15%</td><td>20%</td><td>25%</td><td>30%</td><td>10%</td><td>15%</td><td>20%</td><td>25%</td><td>30%</td><td>10%</td><td>15%</td><td>20%</td><td>25%</td><td>30%</td></tr><tr><td colspan="12">MeMOTR</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Random</td><td>68.38</td><td>69.61</td><td>71.60</td><td>71.78</td><td>72.31</td><td>57.21</td><td>58.54</td><td>61.19</td><td>60.98</td><td>62.04</td><td>70.65</td><td>70.92</td><td>74.41</td><td>74.08</td><td>74.83</td></tr><tr><td>Entropy</td><td>68.61</td><td>69.95</td><td>70.73</td><td>72.24</td><td>72.14</td><td>57.50</td><td>58.27</td><td>60.93</td><td>62.05</td><td>62.29</td><td>71.49</td><td>71.94</td><td>73.70</td><td>73.70</td><td>75.47</td></tr><tr><td>Core-set</td><td>67.52</td><td>68.43</td><td>70.78</td><td>72.50</td><td>71.95</td><td>56.60</td><td>57.03</td><td>60.04</td><td>62.25</td><td>61.76</td><td>70.67</td><td>70.80</td><td>73.19</td><td>75.21</td><td>74.42</td></tr><tr><td>BADGE</td><td>68.17</td><td>69.78</td><td>71.20</td><td>71.02</td><td>72.82</td><td>57.22</td><td>58.98</td><td>60.78</td><td>60.34</td><td>63.13</td><td>70.77</td><td>72.26</td><td>73.98</td><td>73.63</td><td>76.04</td></tr><tr><td>CUTAL</td><td>69.35</td><td>71.06</td><td>71.82</td><td>72.92</td><td>73.56</td><td>57.91</td><td>60.41</td><td>61.70</td><td>63.04</td><td>64.17</td><td>71.59</td><td>73.75</td><td>74.93</td><td>76.02</td><td>76.83</td></tr><tr><td>QPID</td><td>69.46</td><td>71.41</td><td>72.19</td><td>72.96</td><td>73.62</td><td>58.28</td><td>61.06</td><td>61.99</td><td>63.17</td><td>64.10</td><td>71.79</td><td>74.35</td><td>75.31</td><td>76.17</td><td>77.10</td></tr><tr><td colspan="2">SambaMOTR</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Random</td><td>55.52</td><td>56.53</td><td>58.27</td><td>57.23</td><td>56.89</td><td>45.82</td><td>45.31</td><td>47.16</td><td>45.93</td><td>46.47</td><td>58.77</td><td>58.65</td><td>60.81</td><td>59.72</td><td>59.69</td></tr><tr><td>Entropy</td><td>57.39</td><td>58.17</td><td>59.52</td><td>62.84</td><td>62.96</td><td>47.99</td><td>48.47</td><td>49.52</td><td>53.11</td><td>53.25</td><td>61.40</td><td>61.82</td><td>62.97</td><td>66.69</td><td>66.65</td></tr><tr><td>Core-set</td><td>56.42</td><td>56.20</td><td>60.11</td><td>60.64</td><td>61.72</td><td>46.75</td><td>48.25</td><td>50.16</td><td>50.40</td><td>52.23</td><td>59.96</td><td>60.53</td><td>63.73</td><td>63.90</td><td>65.13</td></tr><tr><td>BADGE</td><td>56.27</td><td>57.04</td><td>61.53</td><td>59.41</td><td>64.39</td><td>46.87</td><td>47.98</td><td>51.92</td><td>49.70</td><td>54.30</td><td>60.01</td><td>60.64</td><td>65.46</td><td>62.48</td><td>67.40</td></tr><tr><td>CUTAL</td><td>56.65</td><td>58.69</td><td>62.13</td><td>63.84</td><td>64.54</td><td>46.84</td><td>48.59</td><td>51.91</td><td>53.69</td><td>54.16</td><td>59.94</td><td>62.40</td><td>65.68</td><td>67.49</td><td>67.88</td></tr><tr><td>QPID</td><td>56.17</td><td>59.09</td><td>61.65</td><td>63.99</td><td>64.57</td><td>46.77</td><td>49.35</td><td>51.95</td><td>53.33</td><td>54.65</td><td>59.87</td><td>62.92</td><td>65.60</td><td>67.40</td><td>68.30</td></tr></table>

Removing either instability metric degrades the averaged performance relative to the full method, indicating that Localization Drift and Entropy-Weighted Confidence Discrepancy capture complementary aspects of association instability. The degradation is larger when removing Localization Drift across HOTA, AssA, and IDF1, indicating that perturbationinduced localization changes are especially important in this setting.

Removing Uncertainty-Weighted Visual Coverage reduces the averages to 57.79 HOTA, 46.56 AssA, and 60.48 IDF1. Although this variant remains competitive, it is consistently below the full QPID on the averaged metrics, indicating that visual coverage improves the robustness of clip selection.

Removing scale calibration from the reference-point perturbation (w/o Scale-Calibrated Ref. Pert.) lowers the averaged performance from 58.26 to 57.71 HOTA, from 47.10 to 46.48 AssA, and from 60.79 to 60.39 IDF1. This result supports the importance of calibrating reference-point perturbations by the width and height of the clean propagated reference point, rather than applying uncalibrated perturbations in normalized coordinate space.

We also compare against Image Gaussian Noise, which replaces the proposed internal track-state perturbation with image-space perturbation by adding two-sided Gaussian noise with standard deviation 0.1 to the input image while keeping the propagated clean track state unchanged across branches. This variant performs worse than QPID on average, with 56.65 HOTA, 44.84 AssA, and 58.88 IDF1. This gap suggests that not all perturbation-induced prediction changes are equally useful for clip acquisition in MOT. Image-space perturbations can reflect sensitivity to local appearance or background changes, whereas QPID directly perturbs the propagated track state that is reused for maintaining identities across frames. Therefore, the proposed internal track-state perturbation is better aligned with association difficulty than image-space perturbation in this setting.

We further compare the two-sided design with a lower-cost one-sided variant and an equal-compute variant using two independently sampled perturbation directions. The twosided design achieves the best averaged performance: 58.26 HOTA, 47.10 AssA, and 60.79

Table 6: Ablations and perturbation-space comparisons of QPID on DanceTrack (MeMOTR) under the $5 \% + 5 \%$ schedule. Best results are marked in bold and second best are underlined. Results are averaged over three seeds. Avg denotes the mean over 10% to 30%. w/o Scale-Calibrated Ref. Pert. removes scale calibration in reference-point perturbation, and Image Gaussian Noise replaces internal track-state perturbation with image-space perturbation.
<table><tr><td>Variant</td><td colspan="6">HOTA</td><td colspan="6">AssA</td><td colspan="6">IDF1</td></tr><tr><td></td><td>10%</td><td>15%</td><td>20%</td><td>25%</td><td>30%</td><td>Avg</td><td>10%</td><td>15%</td><td>20%</td><td>25%</td><td>30%</td><td>Avg</td><td>10%</td><td>15%</td><td>20%</td><td>25%</td><td>30%</td><td>Avg</td></tr><tr><td>w/o Localization Drift</td><td>53.77</td><td>56.15</td><td>57.83</td><td>59.05</td><td>59.69</td><td>57.30</td><td>41.38</td><td>44.70</td><td>46.30</td><td>48.28</td><td>49.04</td><td>45.94</td><td>56.05</td><td>58.45</td><td>60.22</td><td>61.49</td><td>62.60</td><td>59.76</td></tr><tr><td>w/o Entropy-Weighted Conf. Discrep.</td><td>54.61</td><td>56.20</td><td>57.64</td><td>59.43</td><td>60.16</td><td>57.61</td><td>42.80</td><td>44.33</td><td>46.19</td><td>48.48</td><td>49.60</td><td>46.28</td><td>57.38</td><td>58.69</td><td>60.36</td><td>62.56</td><td>62.78</td><td>60.36</td></tr><tr><td>w/o Uncertainty-Weighted Visual Coverage</td><td>54.26</td><td>57.33</td><td>58.60</td><td>59.22</td><td>59.55</td><td>57.79</td><td>42.32</td><td>45.88</td><td>47.58</td><td>48.47</td><td>48.52</td><td>46.56</td><td>56.57</td><td>60.13</td><td>61.43</td><td>62.07</td><td>62.20</td><td>60.48</td></tr><tr><td>w/o Scale-Calibrated Ref. Pert.</td><td>53.37</td><td>57.22</td><td>57.64</td><td>60.01</td><td>60.31</td><td>57.71</td><td>41.17</td><td>45.80</td><td>46.24</td><td>49.30</td><td>49.86</td><td>46.48</td><td>55.89</td><td>59.94</td><td>60.37</td><td>62.76</td><td>62.98</td><td>60.39</td></tr><tr><td>Image Gaussian Noise</td><td>52.64</td><td>55.76</td><td>56.18</td><td>58.71</td><td>59.98</td><td>56.65</td><td>39.96</td><td>43.51</td><td>44.30</td><td>47.11</td><td>49.32</td><td>44.84</td><td>54.86</td><td>57.59</td><td>58.22</td><td>61.00</td><td>62.74</td><td>58.88</td></tr><tr><td>One-sided</td><td>54.35</td><td>55.59</td><td>58.38</td><td>58.49</td><td>60.27</td><td>57.42</td><td>42.47</td><td>43.49</td><td>47.46</td><td>47.20</td><td>49.68</td><td>46.06</td><td>56.94</td><td>57.62</td><td>60.81</td><td>60.87</td><td>62.41</td><td>59.73</td></tr><tr><td>Two Independent Directions</td><td>54.09</td><td>56.23</td><td>58.60</td><td>59.46</td><td>60.27</td><td>57.73</td><td>42.18</td><td>44.74</td><td>47.45</td><td>48.79</td><td>49.71</td><td>46.57</td><td>56.44</td><td>58.79</td><td>61.29</td><td>62.00</td><td>62.86</td><td>60.28</td></tr><tr><td>Sum</td><td>54.17</td><td>56.25</td><td>58.83</td><td>59.26</td><td>60.00</td><td>57.70</td><td>42.15</td><td>44.63</td><td>48.04</td><td>48.50</td><td>49.51</td><td>46.57</td><td>56.90</td><td>58.93</td><td>61.15</td><td>61.88</td><td>63.03</td><td>60.38</td></tr><tr><td>QPID</td><td>54.56</td><td>57.30</td><td>58.90</td><td>60.05</td><td>60.51</td><td>58.26</td><td>42.24</td><td>45.82</td><td>48.11</td><td>49.45</td><td>49.90</td><td>47.10</td><td>56.83</td><td>59.74</td><td>61.64</td><td>62.63</td><td>63.13</td><td>60.79</td></tr></table>

IDF1. The one-sided variant obtains 57.42, 46.06, and 59.73, respectively, while the twoindependent-direction variant obtains 57.73, 46.57, and 60.28. These results empirically support the two-sided design in this setting.

Finally, replacing the multiplicative aggregation with additive aggregation (Sum) is competitive in some individual rounds but remains inferior on average, validating the final aggregation design used in QPID. However, multiplicative aggregation can under-rank clips in which only one instability component is large, which remains a limitation of the current formulation.

## E Further Analyses

## E.1 Association-Difficulty Analysis

To directly assess whether QPID’s clip-level association-instability score U(c) reflects association difficulty, we conduct a post-hoc GT-based analysis on the round-1 unlabeled pool of DanceTrack with MeMOTR under the 5%+5% schedule. Within each seed, QPID, CUTAL, and Random share the same initial labeled set, tracker checkpoint, and unlabeled pool, and GT annotations are used only for this analysis.

After matching predictions to GT at the sampled frames, we evaluate two measures of association difficulty. The first is an AssA-style clip-local association error, $1 - \mathsf { A s s } \mathsf { A } _ { \mathrm { c l i p } } .$ which measures inconsistency between predicted track identities and GT identities across sampled frames. The second is competing-GT IoU, defined by averaging the maximum IoU between each matched GT box and any other GT box in the same frame. Higher values indicate greater ambiguity caused by nearby competing targets.

Across the full unlabeled pool, the Spearman correlation between the re-evaluated QPID score U(c) and the clip-local association error is 0.659±0.007 over three seeds. The corresponding correlation with competing-GT IoU is $0 . 5 0 1 { \scriptstyle \pm 0 . 0 2 6 }$ . The partial Spearman correlations controlling for the numbers of GT tracks and GT detections per clip remain 0.633 and 0.428, respectively, suggesting that these relationships are not explained solely by the number of targets in a clip.

Tab. 7 further compares the clips selected at round 1. QPID selects clips with higher clip-local association error and competing-GT IoU than CUTAL and Random in all three seeds. These results provide direct evidence that higher QPID scores are associated with greater association difficulty and that QPID preferentially selects more difficult clips in this setting.

Table 7: GT-based analysis of round-1 selections on DanceTrack with MeMOTR under the $5 \% + 5 \%$ schedule. Results are mean ± standard deviation over three seeds. Higher values indicate greater association difficulty.
<table><tr><td>Measure</td><td>QPID</td><td>CUTAL</td><td>Random</td></tr><tr><td> $1 - \mathrm { A s s A _ { \mathrm { c l i p } } } \uparrow$ </td><td> $\mathbf { 0 . 2 2 3 { \overset { . } { = } } 0 . 0 1 0 }$ </td><td> $0 . 1 8 4 { \pm } 0 . 0 1 6$ </td><td> $0 . 0 7 5 { \scriptstyle \pm 0 . 0 0 6 }$ </td></tr><tr><td>Competing-GT IoU ↑</td><td> $\mathbf { 0 . 2 7 8 { \overset { . } { = } } 0 . 0 0 3 }$ </td><td> $0 . 2 5 2 { \scriptstyle \pm 0 . 0 0 3 }$ </td><td> $0 . 2 1 5 { \pm } 0 . 0 0 5$ </td></tr></table>

## E.2 Statistical Stability

To assess the stability of the results across the three predefined seeds, Tabs. 8 and 9 report the round-wise standard deviations of QPID and CUTAL for all combinations of datasets and trackers. The corresponding mean values under the $5 \% + 5 \%$ and 20%+10% schedules are reported in Sec. C and the main paper, respectively. We focus on CUTAL as the most directly related clip-level active learning baseline.

In some settings, the performance difference between QPID and CUTAL is small relative to the variation across seeds. For example, under the $5 \% + 5 \%$ schedule with MeMOTR, the final-round AssA values are $4 9 . 9 0 { \scriptstyle \pm 0 . 7 7 }$ for QPID and $5 0 . 2 5 { \scriptstyle \pm 0 . 5 7 }$ for CUTAL on Dance-Track, and $6 4 . 1 0 { \pm } 0 . 3 0 $ and $6 4 . 1 7 { \pm } 1 . 5 0 $ , respectively, on SportsMOT. Across 48 matched MeMOTR comparisons, QPID yields higher HOTA, AssA, and IDF1 in 35, 30, and 32 cases, respectively.

## E.3 Hyperparameter Analysis

Tabs. 10, 11, 12, and 13 study the effect of the main hyperparameters of QPID on Dance-Track with MeMOTR under the $5 \% + 5 \%$ schedule. These hyperparameters include the query-embedding perturbation magnitude $\alpha _ { q }$ , the reference-point perturbation magnitude $\alpha _ { \tilde { r } }$ , the normalization clipping range κ, and the candidate expansion ratio $\rho$

For $\alpha _ { q } .$ the default value of 0.50 achieves the best average performance across all three metrics. Using a smaller value of 0.30 or a larger value of 0.70 remains competitive, but both yield slightly lower averages.

For $\alpha _ { \tilde { r } }$ , the default value of 0.010 achieves the best average performance across all three metrics. The smaller value 0.005 performs best at 10%, and the larger value 0.050 is competitive in some individual rounds, but neither setting matches the default in the overall averages.

For κ, the default value of 3 gives the best average performance across all three metrics. The setting $\kappa = 1$ is the second-best choice in the averaged results, whereas $\kappa = 5$ leads to lower averages. Overall, these results support $\kappa = 3$ as a stable default choice for normalization.

For $\rho _ { ; }$ , the default value of 4 gives the best average performance across all three metrics. A smaller value of 2 restricts the candidate pool too strongly, whereas a larger value of 8 remains competitive but slightly lowers the averaged metrics. These results suggest that ρ = 4 provides a balanced candidate pool size, maintaining focus on high-instability clips while allowing sufficient visual coverage.

Table 8: Round-wise standard deviations under the 5%+5% schedule. Values are reported as QPID / CUTAL standard deviations over three predefined seeds. The corresponding mean values are reported in Sec. C.
<table><tr><td>Metric</td><td>10%</td><td>15%</td><td>20%</td><td>25%</td><td>30%</td></tr><tr><td colspan="6">DanceTrack with MeMOTR</td></tr><tr><td>HOTA</td><td>0.27 / 0.53 0.42 / 0.53</td><td>0.12 / 0.72 0.44 / 0.84</td><td>0.25 / 0.59 0.57 / 1.15</td><td>0.13 / 0.41 0.17 / 0.70</td><td>0.16 / 0.56</td></tr><tr><td>AssA IDF1</td><td>0.83 / 0.32</td><td>0.52 / 0.54</td><td>0.50 / 0.69</td><td>0.46 / 0.72</td><td>0.77 / 0.57 0.18 / 1.25</td></tr><tr><td>DanceTrack with SambaMOTR</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="6"></td></tr><tr><td>HOTA</td><td>0.75 / 0.65</td><td>1.52 / 1.24</td><td>1.11 /1.14</td><td>1.24 / 1.19</td><td>0.76 / 0.77</td></tr><tr><td>AssA</td><td>0.74 / 0.37</td><td>1.86 / 0.21</td><td>1.46 / 2.29</td><td>1.81 / 0.88</td><td>0.62 / 1.36</td></tr><tr><td>IDF1</td><td>0.60 / 0.63</td><td>2.05 / 0.18</td><td>1.68 / 1.57</td><td>1.52 / 0.53</td><td>0.88 / 1.15</td></tr><tr><td colspan="6">SportsMOT with MeMOTR</td></tr><tr><td>HOTA AssA</td><td>0.12 / 0.70</td><td>0.92 / 1.05</td><td>0.47 / 0.10</td><td>0.45 / 0.93</td><td>0.28 / 0.80</td></tr><tr><td>IDF1</td><td>0.69 / 0.76</td><td>1.33 / 1.71</td><td>0.87 / 0.15</td><td>0.62 / 1.36</td><td>0.30 / 1.50</td></tr><tr><td></td><td>0.69 / 0.47</td><td>1.19 / 1.12</td><td>0.66 / 0.14</td><td>0.55 / 1.02</td><td>0.27 / 1.16</td></tr><tr><td colspan="6">SportsMOT with SambaMOTR</td></tr><tr><td>HOTA</td><td>1.59 / 1.89</td><td>0.29 / 0.28</td><td>0.65 / 1.74</td><td>1.50 / 0.23</td><td>0.28 / 1.23</td></tr><tr><td>AssA</td><td>0.27 / 1.75</td><td>0.48 / 0.93</td><td>1.00 / 1.84</td><td>1.54 / 0.42</td><td>0.49 / 1.42</td></tr><tr><td>IDF1</td><td>0.94 / 2.10</td><td>0.27 / 0.71</td><td>0.69 / 2.27</td><td>1.57 / 0.25</td><td>0.20 / 1.52</td></tr></table>

Overall, these results show that QPID is not highly sensitive to moderate hyperparameter variations. Across the evaluated settings, the default configuration remains the best or nearbest choice overall, supporting its use as a stable setting in the main experiments.

## E.4 Effect of Diversity-Aware Selection

Fig. 5 visualizes the selected frame positions on DanceTrack at Round 1 under the 5%+5% schedule. We compare QPID, w/o Uncertainty-Weighted Visual Coverage, and Entropy. Each dot denotes one selected frame, where the horizontal axis indicates the normalized temporal position within a sequence and the vertical axis indicates the sequence identity.

Compared with Entropy, QPID avoids overly concentrated selection on a small number of local temporal regions while still covering a broader range of temporal positions and sequences. Removing Uncertainty-Weighted Visual Coverage leads to more redundant selections within similar temporal neighborhoods and sequences. These visualizations are consistent with the role of Uncertainty-Weighted Visual Coverage in reducing redundant concentration during selection.

Table 9: Round-wise standard deviations under the 20%+10% schedule. Values are reported as QPID / CUTAL standard deviations over three predefined seeds. The corresponding mean values are reported in the main paper.
<table><tr><td rowspan=1 colspan=1>Metric</td><td rowspan=1 colspan=1>30%           40%           50%</td></tr><tr><td rowspan=2 colspan=2>DanceTrack with MeMOTRHOTA   0.64 / 0.57   0.45 / 0.76   0.25 / 0.431.08 / 0.92   0.52 / 1.30   0.41 / 0.771.39 / 0.94   0.43 / 1.01   0.73 / 0.92</td></tr><tr><td rowspan=1 colspan=1>ADFA</td></tr><tr><td rowspan=1 colspan=2>DanceTrack with SambaMOTRHOTA   1.73 / 0.46   1.75 / 0.54   1.08 / 1.20AssA     1.39 / 0.25    1.93 / 0.86   1.35 / 1.96IDF1     1.68 / 0.27    1.55 / 0.76   1.22 / 1.19</td></tr><tr><td rowspan=4 colspan=2>SportsMOT with MeMOTR0.42 / 0.32   0.54 / 0.24   0.35 / 0.450.59 / 0.47   0.48 / 0.53   0.59 / 0.570.57 / 0.47   0.33 / 0.37   0.48 / 0.57</td></tr><tr><td rowspan=1 colspan=1>HOTA</td></tr><tr><td rowspan=1 colspan=1>AssA</td></tr><tr><td rowspan=1 colspan=1>IDF1</td></tr><tr><td rowspan=4 colspan=2>SportsMOT with SambaMOTR1.09 / 1.49   0.55 / 0.50    1.01 / 0.311.66 / 1.90   0.63 / 0.54    1.69 / 0.351.67 / 1.85   0.84 / 0.63   1.29 / 0.21</td></tr><tr><td rowspan=1 colspan=1>HOTA</td></tr><tr><td rowspan=1 colspan=1>AssA</td></tr><tr><td rowspan=1 colspan=1>IDF1</td></tr></table>

## E.5 Cold-Start Behavior in Early Rounds

QPID is already competitive in the early rounds with MeMOTR and generally outperforms the baselines as the labeling budget increases. In contrast, with SambaMOTR, baselines can occasionally outperform QPID in the earliest rounds. For example, on SportsMOT under the 5%+5% schedule in Tab. 5, Entropy achieves the best HOTA, AssA, and IDF1 at 10%, while CUTAL [23] achieves the best HOTA and IDF1 at 20%. QPID is nevertheless the best method for HOTA from 25% onward and for IDF1 at 30%. Under the 20%+10% schedule reported in the main paper, Entropy remains competitive at the first round of 30% for SambaMOTR, especially in HOTA, whereas QPID becomes the best on all metrics from 40% onward.

We attribute this trend mainly to the cold-start regime in active learning [13, 14] and the longer clip unit used by SambaMOTR. In the earliest rounds, the tracker is trained on only a small randomly sampled initial labeled subset, so the resulting supervision is limited. This effect is more pronounced in SambaMOTR because its longer clips (T=10) reduce the number of uniquely labeled clips acquired under the same frame budget and also introduce more temporally redundant observations within each selected clip. As a result, the learned representations and internal track states remain less reliable in the early rounds, making it harder to identify truly informative clips for acquisition. Because QPID relies on perturbation-induced instability of internal track states, such immature states can make the estimated instability less aligned with the true association difficulty of a clip. In this regime, simpler heuristics such as Entropy can remain competitive.

Table 10: Hyperparameter analysis of $\alpha _ { q }$ on DanceTrack (MeMOTR) under the $5 \% + 5 \%$ schedule. Best results are marked in bold and second best are underlined. Results are averaged over three seeds. Avg denotes the mean over 10% to 30%.
<table><tr><td rowspan="2"> $\alpha _ { q }$ </td><td colspan="6">HOTA</td><td colspan="6">AssA</td><td colspan="6">IDF1</td></tr><tr><td>10%</td><td>15%</td><td>20%</td><td>25%</td><td>30%</td><td>Avg</td><td>10%</td><td>15%</td><td>20%</td><td>25%</td><td>30%</td><td>Avg</td><td>10%</td><td>15%</td><td>20%</td><td>25%</td><td>30%</td><td>Avg</td></tr><tr><td>0.30</td><td>54.57</td><td>55.99</td><td>58.52</td><td>59.06</td><td>60.14</td><td>57.66</td><td>42.26</td><td>44.03</td><td>47.52</td><td>47.97</td><td>49.48</td><td>46.25</td><td>56.65</td><td>58.15</td><td>61.09</td><td>61.38</td><td>62.98</td><td>60.05</td></tr><tr><td>0.50 (default)</td><td>54.56</td><td>57.30</td><td>58.90</td><td>60.05</td><td>60.51</td><td>58.26</td><td>42.24</td><td>45.82</td><td>48.11</td><td>49.45</td><td>49.90</td><td>47.10</td><td>56.83</td><td>59.74</td><td>61.64</td><td>62.63</td><td>63.13</td><td>60.79</td></tr><tr><td>0.70</td><td>53.96</td><td>56.65</td><td>58.54</td><td>59.70</td><td>59.83</td><td>57.73</td><td>41.80</td><td>44.86</td><td>47.50</td><td>48.86</td><td>48.91</td><td>46.39</td><td>56.72</td><td>59.16</td><td>61.27</td><td>62.21</td><td>62.23</td><td>60.32</td></tr></table>

Table 11: Hyperparameter analysis of $\alpha _ { \tilde { r } }$ on DanceTrack (MeMOTR) under the 5%+5% schedule. Best results are marked in bold and second best are underlined. Results are averaged over three seeds. Avg denotes the mean over 10% to 30%.
<table><tr><td rowspan="2">α</td><td colspan="6">HOTA</td><td colspan="6">AssA</td><td colspan="6">IDF1</td></tr><tr><td>10%</td><td>15%</td><td>20%</td><td>25%</td><td>30%</td><td>Avg</td><td>10%</td><td>15%</td><td>20%</td><td>25%</td><td>30%</td><td>Avg</td><td>10%</td><td>15%</td><td>20%</td><td>25%</td><td>30%</td><td>Avg</td></tr><tr><td>0.005</td><td>55.01</td><td>55.58</td><td>58.11</td><td>59.70</td><td>59.85</td><td>57.65</td><td>42.73</td><td>43.29</td><td>47.02</td><td>48.80</td><td>49.05</td><td>46.18</td><td>57.35</td><td>57.74</td><td>60.44</td><td>62.17</td><td>62.57</td><td>60.05</td></tr><tr><td>0.010 (default)</td><td>54.56</td><td>57.30</td><td>58.90</td><td>60.05</td><td>60.51</td><td>58.26</td><td>42.24</td><td>45.82</td><td>48.11</td><td>49.45</td><td>49.90</td><td>47.10</td><td>56.83</td><td>59.74</td><td>61.64</td><td>62.63</td><td>63.13</td><td>60.79</td></tr><tr><td>0.050</td><td>54.37</td><td>56.95</td><td>57.96</td><td>59.76</td><td>58.93</td><td>57.59</td><td>42.21</td><td>45.50</td><td>46.59</td><td>48.88</td><td>47.81</td><td>46.20</td><td>56.71</td><td>59.76</td><td>60.24</td><td>62.38</td><td>61.44</td><td>60.10</td></tr></table>

Nevertheless, this disadvantage is mainly limited to the earliest rounds. As more labeled clips are accumulated, the tracker states become more reliable and the perturbation-based instability measured by QPID better reflects genuine association ambiguity. Accordingly, QPID becomes increasingly effective in later rounds and eventually surpasses the baselines for SambaMOTR, while remaining strong throughout for MeMOTR.

## E.6 Full-Supervision Gap in SambaMOTR

QPID with MeMOTR approaches full-supervision performance, whereas a substantial gap still remains for SambaMOTR. Under the 20%+10% schedule reported in the main paper, the gap between QPID and full supervision is small for MeMOTR on DanceTrack, where QPID is 0.72 HOTA, 0.91 AssA, and 0.52 IDF1 below full supervision at 50%. On SportsMOT, QPID with MeMOTR achieves performance comparable to full supervision in this setting. In contrast, the gap remains much larger for SambaMOTR. At 50%, QPID is still 6.85 HOTA, 6.44 AssA, and 6.83 IDF1 below full supervision on DanceTrack, and 8.63 HOTA, 10.11 AssA, and 8.57 IDF1 below full supervision on SportsMOT.

This gap is likely related to the larger sampling unit used by SambaMOTR. Because SambaMOTR uses longer clips (T=10) than MeMOTR (T=4), the same frame-level budget yields substantially fewer uniquely labeled clips. This reduces the effective coverage of diverse scenes, motion patterns, and interaction cases that can be acquired during active learning. Moreover, each selected clip spans a longer temporal range and therefore tends to include more temporally redundant observations, so a larger fraction of the annotation budget is spent on frames with limited marginal training value.

As a result, even when the acquisition strategy becomes more reliable in later rounds, the overall supervision available to SambaMOTR remains less diverse than that of MeM-OTR under the same frame-level budget. Nevertheless, in the later rounds, QPID brings SambaMOTR closer to the full-supervision upper bound than the competing baselines, indicating that association-aware acquisition remains beneficial even in this setting.

Table 12: Hyperparameter analysis of κ on DanceTrack (MeMOTR) under the $5 \% + 5 \%$ schedule. Best results are marked in bold and second best are underlined. Results are averaged over three seeds. Avg denotes the mean over 10% to 30%.
<table><tr><td rowspan="2">K</td><td colspan="6">HOTA</td><td colspan="6">AssA</td><td colspan="6">IDF1</td></tr><tr><td>10%</td><td>15%</td><td>20%</td><td>25%</td><td>30%</td><td>Avg</td><td>10%</td><td>15%</td><td>20%</td><td>25%</td><td>30%</td><td>Avg</td><td>10%</td><td>15%</td><td>20%</td><td>25%</td><td>30%</td><td>Avg</td></tr><tr><td>1</td><td>54.06</td><td>56.41</td><td>58.44</td><td>60.27</td><td>59.32</td><td>57.70</td><td>41.83</td><td>44.74</td><td>47.52</td><td>49.83</td><td>48.02</td><td>46.39</td><td>56.41</td><td>59.05</td><td>61.37</td><td>63.20</td><td>61.98</td><td>60.40</td></tr><tr><td>3 (default)</td><td>54.56</td><td>57.30</td><td>58.90</td><td>60.05</td><td>60.51</td><td>58.26</td><td>42.24</td><td>45.82</td><td>48.11</td><td>49.45</td><td>49.90</td><td>47.10</td><td>56.83</td><td>59.74</td><td>61.64</td><td>62.63</td><td>63.13</td><td>60.79</td></tr><tr><td>5</td><td>53.53</td><td>56.82</td><td>57.67</td><td>59.66</td><td>59.74</td><td>57.48</td><td>41.14</td><td>45.21</td><td>46.52</td><td>49.20</td><td>48.83</td><td>46.18</td><td>55.38</td><td>59.53</td><td>60.53</td><td>62.48</td><td>62.28</td><td>60.04</td></tr></table>

Table 13: Hyperparameter analysis of $\rho$ on DanceTrack (MeMOTR) under the $5 \% + 5 \%$ schedule. Best results are marked in bold and second best are underlined. Results are averaged over three seeds. Avg denotes the mean over 10% to 30%.
<table><tr><td rowspan="2">ρ</td><td colspan="6">HOTA</td><td colspan="6">AssA</td><td colspan="6">IDF1</td></tr><tr><td>10%</td><td>15%</td><td>20%</td><td>25%</td><td>30%</td><td>Avg</td><td>10%</td><td>15%</td><td>20%</td><td>25%</td><td>30%</td><td>Avg</td><td>10%</td><td>15%</td><td>20%</td><td>25%</td><td>30%</td><td>Avg</td></tr><tr><td>2</td><td>53.71</td><td>56.79</td><td>57.57</td><td>58.80</td><td>59.79</td><td>57.33</td><td>41.37</td><td>45.28</td><td>46.09</td><td>47.56</td><td>48.95</td><td>45.85</td><td>55.66</td><td>59.39</td><td>60.09</td><td>61.41</td><td>62.61</td><td>59.83</td></tr><tr><td>4 (default)</td><td>54.56</td><td>57.30</td><td>58.90</td><td>60.05</td><td>60.51</td><td>58.26</td><td>42.24</td><td>45.82</td><td>48.11</td><td>49.45</td><td>49.90</td><td>47.10</td><td>56.83</td><td>59.74</td><td>61.64</td><td>62.63</td><td>63.13</td><td>60.79</td></tr><tr><td>8</td><td>54.59</td><td>56.75</td><td>58.15</td><td>60.39</td><td>59.46</td><td>57.87</td><td>42.46</td><td>44.95</td><td>47.14</td><td>49.90</td><td>48.67</td><td>46.62</td><td>56.94</td><td>59.31</td><td>60.91</td><td>63.62</td><td>61.98</td><td>60.55</td></tr></table>

## E.7 Association-Oriented Improvements over CUTAL

QPID is designed to select clips that are informative for improving identity association in query-propagation MOT. We therefore further compare QPID with CUTAL, the most directly related clip-level active learning baseline, using association-oriented metrics. For AssA and IDF1, we report QPID minus CUTAL, where positive values indicate improvement.

Tab. 14 summarizes the differences under both active learning schedules. For the 5%+5% schedule, Final denotes the 30% annotation budget and Avg denotes the mean over 10%, 15%, 20%, 25%, and 30%. For the 20%+10% schedule, Final denotes the 50% annotation budget and Avg denotes the mean over 30%, 40%, and 50%.

Under the 5%+5% schedule, QPID improves CUTAL by +0.46 AssA and +0.47 IDF1 on average across dataset–tracker settings. Under the 20%+10% schedule, the gains become larger, with +1.05 AssA and +0.88 IDF1 on average. These results indicate that perturbationbased association-instability scoring selects clips that are more useful for improving identity association than CUTAL’s output-level temporal uncertainty and temporal diversity.

## E.8 Acquisition Sampling Time

QPID introduces additional computation only during active learning acquisition. It does not modify the model architecture, training objective, or deployment-time tracking inference. To quantify this cost, we measure the running time of a single full Round 1 acquisition sampling with MeMOTR. CUTAL and QPID are measured under the same hardware, batch size, data-loading configuration, and unlabeled pool for each dataset.

As shown in Tab. 15, QPID takes 321.95 minutes on DanceTrack and 214.75 minutes on SportsMOT, compared with 273.73 and 180.58 minutes for CUTAL, respectively. This corresponds to 1.18 to 1.19× the acquisition sampling time of CUTAL. This overhead is relatively modest for annotation acquisition, because it is incurred only at active learning rounds and does not affect deployment-time tracking speed.

![](images/56477d1495c128fa9d6184dfe3b6cb9fb12b4c1189e63d68b4e34ec41b5088c0.jpg)

![](images/30a0244db7b748d8303eafedc0152af11e3b6ffcb0c296c86d15ab496a3749e7.jpg)

![](images/113ea1f59a00d28979742f53d73fda27636e90a3722c83aa1cb9b4e0b764c85d.jpg)  
Figure 5: Visualization of selected frame positions on DanceTrack at Round 1 under the $5 \% + 5 \%$ schedule. From left to right, we show QPID, w/o Uncertainty-Weighted Visual Coverage, and Entropy. Each dot denotes one selected frame. The horizontal axis shows the normalized temporal position within each sequence, and the vertical axis shows the sequence identity. QPID shows less redundant concentration than Entropy and the variant without Uncertainty-Weighted Visual Coverage.

Table 14: Association-metric differences over CUTAL under both active learning schedules. We report QPID minus CUTAL. For the 5%+5% schedule, Final denotes 30% and Avg denotes the mean over 10%, 15%, 20%, 25%, and 30%. For the 20%+10% schedule, Final denotes 50% and Avg denotes the mean over 30%, 40%, and 50%. Positive values indicate improvement.
<table><tr><td>Schedule</td><td>Dataset</td><td>Tracker</td><td>∆AssA Final</td><td>∆AssA Avg</td><td>ΔIDF1 Final</td><td>∆IDF1 Avg</td></tr><tr><td>5%+5%</td><td>DanceTrack</td><td>MeMOTR</td><td>-0.35</td><td>+0.23</td><td>+0.38</td><td>+0.29</td></tr><tr><td>5%+5%</td><td>DanceTrack</td><td>SambaMOTR</td><td>+1.64</td><td>+1.15</td><td>+1.21</td><td>+1.12</td></tr><tr><td>5%+5%</td><td>SportsMOT</td><td>MeMOTR</td><td>-0.07</td><td>+0.27</td><td>+0.27</td><td>+0.32</td></tr><tr><td>5%+5%</td><td>SportsMOT</td><td>SambaMOTR</td><td>+0.49</td><td>+0.17</td><td>+0.42</td><td>+0.14</td></tr><tr><td>5%+5%</td><td>Average</td><td>All settings</td><td>+0.43</td><td>+0.46</td><td>+0.57</td><td>+0.47</td></tr><tr><td>20%+10%</td><td>DanceTrack</td><td>MeMOTR</td><td>+1.61</td><td>+1.17</td><td>+1.30</td><td>+1.00</td></tr><tr><td>20%+10%</td><td>DanceTrack</td><td>SambaMOTR</td><td>+1.30</td><td>+1.58</td><td>+1.13</td><td>+1.14</td></tr><tr><td>20%+10%</td><td>SportsMOT</td><td>MeMOTR</td><td>+0.57</td><td>+0.45</td><td>+0.04</td><td>+0.28</td></tr><tr><td>20%+10%</td><td>SportsMOT</td><td>SambaMOTR</td><td>+2.15</td><td>+1.00</td><td>+2.40</td><td>+1.09</td></tr><tr><td>20%+10%</td><td>Average</td><td>All settings</td><td>+1.41</td><td>+1.05</td><td>+1.22</td><td>+0.88</td></tr></table>

## F Qualitative Results

Figs. 6 and 7 show qualitative examples of perturbation-induced variations in predicted bounding boxes on DanceTrack and SportsMOT, respectively. All examples are obtained using models trained with 20% labeled frames under the 20%+10% schedule. Within each dataset, the top row shows stable clips with low U(c), whereas the bottom row shows unstable clips with high U(c) selected by QPID.

In each example, the green, blue, and red boxes correspond to the clean prediction and the two perturbed predictions for the most unstable object in the scene, while the remaining boxes show the standard inference outputs. Stable examples exhibit only small variations under perturbations and often correspond to sparse scenes with limited interactions. By contrast, unstable examples show larger perturbation-induced box shifts and typically involve dense crowds or visually similar nearby objects, which increase association ambiguity and lead to higher prediction instability.

Table 15: Acquisition sampling time comparison. We report the running time of a single full Round 1 acquisition sampling command with MeMOTR, including model initialization, acquisition scoring, batch selection, and output.
<table><tr><td>Dataset</td><td>Method</td><td>Time [min] ↓ Rel. to CUTAL ↓</td></tr><tr><td>DanceTrack</td><td>CUTAL</td><td>273.73 1.00×</td></tr><tr><td>DanceTrack</td><td>QPID</td><td>321.95 1.18×</td></tr><tr><td>SportsMOT</td><td>CUTAL</td><td>180.58 1.00× 1.19×</td></tr><tr><td>SportsMOT</td><td>QPID</td><td>214.75</td></tr></table>

![](images/5b660958ee3582e4a3e10129c9fe777af9956653ad27fbdf475f3986404f77a8.jpg)

![](images/4c41c3959f46ee8842961b76469c18ad195609abe2b915489b3bf478ae587d68.jpg)

![](images/67a1f3e5ea2335894deaeba296ebd7f7bc540fdb79933bbba572916889136adb.jpg)  
Figure 6: Qualitative comparison of stable and unstable clips under perturbations on DanceTrack. The top row shows clips with low U(c) and the bottom row shows clips with high U(c). Green, blue, and red boxes denote the clean prediction and the two perturbed predictions for the most unstable object.

![](images/d1915bae0594927112ce5750f15ed313e6be8cf8b304372e6c5ad75716823409.jpg)

![](images/7960a0e0f3c4989ec3f72e915f8fb6bc53ab1bf1833bee59f81c066c91c72dfb.jpg)

![](images/6d6bc69a1fa0330c0a02ed5352866167f45c7c55f6f5c1f7c74e5e684b99d957.jpg)  
Figure 7: Qualitative comparison of stable and unstable clips under perturbations on SportsMOT. The top row shows clips with low $U ( c )$ and the bottom row shows clips with high U(c). Green, blue, and red boxes denote the clean prediction and the two perturbed predictions for the most unstable object.