# Probe-VAD: Ordinal Likelihood Probing for Training-Free Video Anomaly Detection

Jiawei Gu<sup>∗</sup>, Qilin Zhao<sup>∗</sup>, Tengkuo Guo, Zhiming Zhong, Shuangqing Zhang<sup>†</sup>, Fan Lyu, Fang Zhao<sup>†</sup>, Guo-Sen Xie, Caifeng Shan, Senior Member, IEEE

Abstract—Video anomaly detection (VAD) aims to localize anomalous events in untrimmed videos. Vision-language models (VLMs) provide rich visual understanding for training-free VAD, but existing approaches impose restrictive interfaces between visual understanding and anomaly scoring. Caption-based pipelines compress visual evidence into text, potentially discarding subtle cues, while direct numerical generation forces the model to express its judgment through a small set of predefined scores. Such interfaces can obscure subtle differences in anomaly severity, causing visually distinct clips to receive similar representations or scores and thereby limiting the resolution of anomaly ranking. We propose Probe-VAD, an ordinal binary-probing framework that directly probes severity preferences from a frozen VLM. Given raw video clips, Probe-VAD queries ten ordered severity thresholds and extracts constrained YES/NO continuation likelihoods. Their normalized preferences form a cumulative severity profile, from which tail evidence is aggregated into a continuous anomaly score, with isotonic projection enforcing ordinal consistency. Experiments on public VAD benchmarks demonstrate superior performance with low computational cost. Probe-VAD provides a simple interface for translating frozen VLM visual understanding into continuous, rank-sensitive anomaly scores without taskspecific training or caption-based compression. Code is available at: https://github.com/yvestine/COVAS-VAD.

Index Terms—Video Anomaly Detection, Cumulative Ordinal Modeling, Conditional Likelihood.

## I. INTRODUCTION

Video anomaly detection (VAD) [1], [2] aims to temporally localize unusual and hazardous events in long untrimmed videos, and has attracted significant attention due to its overcoming the impracticality of manual monitoring in surveillance security [3], and industrial inspection [4], [5]. Previous VAD approaches have achieved compelling performance under one-class [6], [7] and weakly supervised settings [3], [8]–[10], but typically rely on task- or domain-specific optimization. Such dependence can limit their generalization when testtime scenes, anomaly categories, or operational conditions differ from those encountered during training. Recent advances in vision-language models (VLMs) [11] offer a promising alternative for training-free VAD, leveraging rich pretrained visual-semantic knowledge without target-domain optimization. Accordingly, recent methods have explored captiondriven linguistic reasoning [10], event-aware and hierarchical temporal decomposition [12], [13], direct generative refinement [14], and memory-augmented retrieval [15]. These efforts demonstrate the strong potential of frozen VLMs for perceiving and reasoning about anomalous video content. However, rich visual understanding alone is insufficient for effective VAD, which requires translating visual evidence into continuous anomaly scores for fine-grained temporal ranking. Existing studies have largely focused on how visual evidence is represented, organized, or reasoned over, while comparatively less attention has been paid to the interface through which VLM understanding is converted into anomaly scores. This leaves a fundamental question largely underexplored: how can the rich visual understanding of a frozen VLM be faithfully translated into continuous anomaly-ranking signals?

As illustrated in Fig. 1, information useful for anomaly ranking can be progressively lost as the visual understanding of a frozen VLM is converted into a scalar score. Caption-mediated pipelines may first introduce a representation bottleneck by compressing visual observations into text, making cues omitted from the description inaccessible to the downstream scorer. More importantly, even when the scorer directly observes the video, the decoding bottleneck remains. Autoregressive numerical generation retains only the selected output while discarding the model’s relative preferences over alternative responses. Thus, direct visual evidence alone is insufficient; retaining pre-decoding preferences is also necessary for finer anomaly ranking. Yet likelihood retention alone remains inadequate because anomaly severity is inherently ordered. Let $S ~ \in ~ [ 0 , 1 ]$ denote a conceptual anomaly-severity variable, where larger values indicate stronger anomaly evidence. Here, “order” refers to severity rather than temporal frame order: evidence supporting a severity level of at least 0.8 should also support the less demanding conditions of at least 0.5 and 0.3. Probe-VAD therefore progressively refines the score-extraction interface by retaining direct visual evidence, preserving predecoding likelihood preferences, and organizing these preferences through cumulative severity-threshold propositions.

In this paper, we propose Probe-VAD, a training-free ordinal likelihood probing framework that progressively preserves and structures the information required for anomaly ranking. Given a video clip, Probe-VAD directly conditions a frozen VLM on the visual input, avoiding intermediate textual compression. Rather than retaining only a decoded numerical response, it extracts pre-decoding YES/NO likelihood preferences from a set of ordinal queries, each associated with a different severity threshold. These preferences are organized into a cumulative ordinal profile, where higher thresholds represent increasingly stringent anomaly conditions. The resulting profile is then integrated into a continuous anomaly score for fine-grained temporal ranking. Experiments on three mainstream VAD datasets, i.e., UCF-Crime [3], MSAD [16] and XD-Violence [17], validate each design choice: direct visual conditioning outperforms caption conditioning, likelihood-based scoring substantially improves over direct numerical generation, and ordinal modeling provides further gains over flat likelihood scoring. Probe-VAD achieves superior performance on three mainstream benchmarks, while also reducing end-to-end wall-clock time by 59.0% and artifact size by 87.3% over the complete caption-based pipeline.

![](images/80eb3a1bc82f8a8d6539258a6babc3a0d864b3d50fd6700eb0b4a75b064c9cf7.jpg)  
Fig. 1. (a) Caption-mediated VAD compresses visual observations into text, potentially discarding fine-grained anomaly cues before scoring and creating a representation bottleneck. (b) Direct numerical VLM scoring retains visual evidence but collapses richer model preferences into a single decoded value, creating a decoding bottleneck that limits fine-grained anomaly ranking. (c) Probe-VAD bypasses both bottlenecks by directly probing visual evidence, retaining likelihood-level preferences, and structuring them with ordered severity thresholds to derive continuous anomaly rankings.

Our contributions are summarized as follows:

• We identify the score-extraction interface as an underexplored problem in training-free VLM-based VAD, revealing two key bottlenecks: caption mediation may discard anomaly-relevant visual evidence, while autoregressive numerical decoding can lose fine-grained visual information.

• We propose Probe-VAD, a training-free ordinal likelihood probing framework that directly accesses visual evidence, retains likelihood-level model preferences before decoding, and organizes them through ordered severity thresholds to produce continuous anomaly rankings.

• Extensive experiments on three VAD benchmarks validate the proposed design principles and demonstrate superior detection performance together with improved computational and storage efficiency.

## II. RELATED WORK

## A. Video Anomaly Detection

Video anomaly detection (VAD) [2] has been extensively studied under one class [6], [7] and weakly supervised settings [1], [8]–[10]. Reconstruction- and prediction-based methods [18], [19] detect deviations from normal patterns, while weakly supervised approaches [3], [20] learn anomaly discrimination from video-level labels. Open-vocabulary VAD extends detection beyond predefined anomaly categories but still requires task-specific optimization [21]. RVT [1] introduces a vision-centric reconstructive objective to supervise visual outputs under a weak-supervised paradigm. More recently, TD-VAD [22] reduces reliance on visual training data by learning from LLM-generated textual sequences. Despite these advances, existing paradigms still rely on task-specific learning from either visual or surrogate supervision, which may limit generalization beyond the training distribution. In contrast, Probe-VAD eliminates task-specific training and directly converts the visually conditioned preferences of a frozen VLM into continuous anomaly-ranking signals.

## B. Training-Free Video Anomaly Detection

Training-free VAD has recently emerged as a promising alternative to task-specific learning. Caption-mediated methods, such as LAVAD [23] and URF-HVAA [24], translate visual observations into textual descriptions and leverage language models for anomaly reasoning. While effective, anomalyrelevant visual cues omitted during caption generation are no longer accessible to the downstream scorer. Other approaches focus on temporal organization: EventVAD [12] performs event-aware segmentation and hierarchical reasoning, while VADTree [13] organizes long videos into a hierarchical multigranularity structure. These methods primarily improve how visual evidence is temporally organized rather than how anomaly scores are extracted from a fixed visual clip. More recent approaches explore direct VLM reasoning and efficient inference. CoReVAD [14] directly generates segment-level anomaly responses and refines them with temporal context, while Flashback [15] employs pseudo-scene memory and retrieval to reduce repeated model inference. Related multimodal approaches such as VERA [25] and VADOR [26] still involve task-specific learning and therefore fall outside the strict training-free setting considered here.

In contrast, Probe-VAD focuses on the score-extraction interface of frozen VLMs, directly probing ordered severity propositions and retaining likelihood-level preferences to produce continuous anomaly rankings.

## C. Ordinal Modeling and Likelihood-Based Scoring

Ordinal modeling captures the inherent ordering among discrete labels through ordered or cumulative predictions. In VAD, Pang et al. [27] model anomaly severity using ordinal regression, while isotonic regression provides a non-parametric mechanism for enforcing monotonic consistency. Likelihoodbased scoring instead exploits model preferences over candidate responses before decoding, although such preferences can be sensitive to prompt and label priors [28]. LogicQA [29] applies a related strategy to image anomaly detection using constrained VLM questions and answer-token probabilities.

Different from these works, Probe-VAD combines ordinal structure with likelihood probing for training-free VAD. It queries a frozen VLM with ordered severity thresholds, retains YES/NO continuation likelihoods, and integrates the resulting cumulative ordinal evidence into continuous anomaly rankings. This avoids learning an ordinal prediction head while preserving fine-grained model preferences before decoding.

## III. METHODOLOGY

## A. Problem Formulation

Let $V = \{ I _ { f } \} _ { f = 0 } ^ { F - 1 }$ denote an untrimmed video of F frames, where $I _ { f }$ is the f-th frame. The frame-level ground truth is $Y = \{ y _ { f } \} _ { f = 0 } ^ { F - 1 }$ , where $y _ { f } \in \{ 0 , 1 \}$ indicates whether frame f belongs to an annotated anomalous interval. Given a frozen VLM, the goal is to produce a continuous frame-level anomaly score sequence $\hat { Y } = \{ \hat { y } _ { f } \} _ { f = 0 } ^ { F - 1 }$ , with $\hat { y } _ { f } \in [ 0 , 1 ]$ , where higher scores indicate stronger anomaly evidence. No target-domain optimization, parameter updates, prompt learning, or labelbased calibration are involved; ground-truth annotations are used solely for evaluation.

The key challenge is therefore to translate the visual understanding of the frozen VLM into continuous anomaly-ranking signals. Instead of compressing visual evidence into intermediate captions or relying on decoded numerical responses, Probe-VAD directly evaluates a sequence of ordered severity propositions conditioned on the visual clip and extracts their likelihood-level evidence for anomaly scoring.

## B. Framework Overview and Clip Construction

Fig. 2 presents the overall framework of Probe-VAD. Given an untrimmed video, Probe-VAD consists of four stages: (1) constructing overlapping temporal clips; (2) directly encoding each clip with a frozen VideoLLaMA3-7B [30]; (3) extracting cumulative ordinal evidence from constrained YES/NO continuation likelihoods; and (4) reconstructing the resulting clip-level scores into frame-level anomaly predictions. The core of Probe-VAD lies in Stages (2) and (3): direct visual conditioning preserves access to anomaly-relevant visual evidence, while ordinal likelihood probing converts the VLM’s visually conditioned preferences into continuous anomaly scores. Stages (1) and (4) provide the temporal interface for processing long untrimmed videos.

Formally, let r denote the original video frame rate, $\Delta$ the spacing in frames between adjacent clip centers, and $c _ { i }$ the center-frame index of the i-th clip. The total number of clips is denoted by $M \colon$

$$
\begin{array} { l } { \displaystyle { c _ { i } = i \Delta , \qquad i = 0 , \dots , M - 1 , \qquad M = \left\lfloor \frac { F - 1 } { \Delta } \right\rfloor + 1 . } } \end{array}\tag{1}
$$

Given a temporal window of duration $L$ seconds, let $a _ { i }$ and $b _ { i }$ denote the start and end times, respectively, for clip i:

$$
a _ { i } = \operatorname* { m a x } \left( 0 , \frac { c _ { i } } { r } - \frac { L } { 2 } \right) , \qquad b _ { i } = \operatorname* { m i n } \left( \frac { F } { r } , \frac { c _ { i } } { r } + \frac { L } { 2 } \right)\tag{2}
$$

Let $r _ { s }$ denote the target frame-sampling rate and $N$ the maximum number of retained RGB frames. The resulting visual input is denoted by $X _ { i } = \{ I _ { i , 1 } , . . . , I _ { i , n _ { i } } \}$ , where $I _ { i , j }$ denotes the j-th sampled frame of clip i, and $n _ { i } \leq N$ is the actual number of sampled frames in clip i.

## C. Cumulative Ordinal Visual Scoring

For the i-th clip $X _ { i } ,$ , let $S _ { i } ~ \in ~ [ 0 , 1 ]$ denote a conceptual latent severity variable representing anomaly evidence, where larger values indicate stronger anomaly evidence. Importantly, $S _ { i }$ is neither observed nor learned from severity annotations; it serves only to define an ordinal scoring formulation. Rather than asking the frozen VLM to directly decode a single numerical severity value, we characterize $S _ { i }$ through K nested threshold propositions, where $K$ denotes the number of severity thresholds. Let $\mathcal { T } = \{ \tau _ { k } \} _ { k = 1 } ^ { K }$ denote the ordered threshold grid, where $\tau _ { k }$ is the k-th severity threshold:

$$
\tau _ { k } = \frac { k } { K } ,\tag{3}
$$

We use $K = 1 0$ , yielding $\mathcal { T } = \{ 0 . 1 , 0 . 2 , \ldots , 1 . 0 \}$

For each threshold $\tau _ { k } ,$ , the frozen VLM evaluates whether the anomaly severity in $X _ { i }$ reaches or exceeds $\tau _ { k }$ using a constrained binary query with YES/NO continuations. The query semantics are fixed across thresholds and datasets by a common severity scale, while only the numerical threshold $\tau _ { k }$ varies. This yields a sequence of ordered severity judgments for extracting likelihood-level ordinal preferences.

For the k-th threshold, let $Q _ { k }$ denote the corresponding binary query. Let $C \in \{ \mathrm { Y E S } , \mathrm { N O } \}$ denote a candidate continuation, represented by the token sequence $c _ { 1 : m _ { C } } $ , where $c _ { t }$ is the token at position t and $m _ { C }$ is the number of tokens in candidate C. We denote the frozen VLM by Φ, with $P _ { \Phi }$ representing its conditional token probability for each candidate continuation token. The conditional log-likelihood of candidate C for clip $X _ { i }$ under query $Q _ { k }$ is denoted by $\ell _ { i , k } ^ { C } \colon$

$$
\ell _ { i , k } ^ { C } = \sum _ { t = 1 } ^ { m _ { C } } \log P _ { \Phi } \left( c _ { t } \mid X _ { i } , Q _ { k } , c _ { < t } \right) ,\tag{4}
$$

![](images/86b7ecb1ce5c9189099d8918e65186e5474136d5ce497bd6293639ebde3f1137.jpg)  
Fig. 2. Overview of Probe-VAD. Given video clips, a frozen VLM encodes visual evidence without captions and is probed with ordered severity thresholds. The YES/NO continuation likelihoods form a cumulative ordinal profile, which is integrated into continuous clip-level anomaly scores with the pool-adjacentviolators algorithm (PAVA) enforcing ordinal consistency. The clip-level scores are reconstructed into frame-level anomaly predictions.

where $c _ { < t }$ denotes the candidate tokens preceding position t. Let $T > 0$ denote the likelihood temperature. We denote by $p _ { i , k }$ the normalized YES preference for clip $X _ { i }$ at threshold $\tau _ { k } \colon$

$$
p _ { i , k } = \frac { \exp { \left( \ell _ { i , k } ^ { \mathrm { Y E S } } / T \right) } } { \exp { \left( \ell _ { i , k } ^ { \mathrm { Y E S } } / T \right) } + \exp { \left( \ell _ { i , k } ^ { \mathrm { N O } } / T \right) } } .\tag{5}
$$

The resulting $p _ { i , k }$ measures the VLM’s relative support for the proposition that the anomaly severity in $X _ { i }$ reaches or exceeds $\tau _ { k } .$ . We denote the corresponding VLM-induced tail-evidence proxy by ${ \widehat { P } } _ { \Phi } ( S _ { i } \geq \tau _ { k } \mid X _ { i } )$ , defined as

$$
\widehat { P } _ { \Phi } ( S _ { i } \geq \tau _ { k } \mid X _ { i } ) : = p _ { i , k } .\tag{6}
$$

This notation reflects the cumulative-threshold interpretation rather than a calibrated probability.

This likelihood-based score extraction differs fundamentally from direct autoregressive generation. A decoded numerical response retains only the selected output, whereas Eq. (5) preserves the VLM’s relative preference between the constrained continuations. Moreover, the threshold propositions are inherently nested: evidence supporting a higher severity threshold should remain compatible with all lower thresholds. Accordingly, a valid cumulative profile should satisfy

$$
p _ { i , 1 } \geq p _ { i , 2 } \geq \cdot \cdot \cdot \geq p _ { i , K } .\tag{7}
$$

This monotonic structure explicitly captures the ordinal nature of anomaly severity, rather than treating different severity levels as independent categories.

## D. Monotonic Projection and Cumulative Tail Integration

Although the threshold propositions are ordinally related, their likelihood preferences are not explicitly constrained to satisfy the expected monotonic relation. Let $\begin{array} { r l } { \mathbf { p } _ { i } } & { { } = } \end{array}$ $[ p _ { i , 1 } , \ldots , p _ { i , K } ]$ denote the raw threshold-wise severity profile of clip $X _ { i }$ . Because this profile may contain local monotonicity violations, we denote its monotonic projection by $\hat { { \bf p } } _ { i } .$ , use $\mathbf { z } = [ z _ { 1 } , \dots , z _ { K } ] \in \mathbb { R } ^ { K }$ as the optimization variable, and let $\| \cdot \| _ { 2 }$ denote the Euclidean norm:

$$
\hat { \mathbf { p } } _ { i } = \arg \operatorname* { m i n } _ { \mathbf { z } \in \mathbb { R } ^ { K } } \| \mathbf { z } - \mathbf { p } _ { i } \| _ { 2 } ^ { 2 } \quad \mathrm { s . t . } \quad 1 \ge z _ { 1 } \ge \dots \ge z _ { K } \ge 0 .\tag{8}
$$

We solve Eq. (8) using the pool-adjacent-violators algorithm (PAVA), which introduces no learnable parameters. This projection converts the raw threshold-wise preferences into a monotonic cumulative profile consistent with the ordinal structure of anomaly severity. Under the uniform threshold grid and equal-weight aggregation used below, PAVA preserves the final scalar score used in the final ranking; its role is therefore to enforce structural consistency on the intermediate profile rather than alter the ranking statistic.

Let $\hat { p } _ { i , k }$ denote the k-th component of the projected profile $\hat { \mathbf { p } } _ { i } .$ . To convert the cumulative profile into a continuous anomaly score, we draw on the tail-integral identity. For a generic bounded random variable $S \in [ 0 , 1 ]$ and a continuous severity threshold $\tau \in [ 0 , 1 ]$ , the tail-integral identity gives

$$
\mathbb { E } [ S ] = \int _ { 0 } ^ { 1 } P ( S \geq \tau ) d \tau .\tag{9}
$$

Motivated by this relation, let $s _ { i }$ denote the continuous anomaly score of clip $X _ { i } ,$ and define $\tau _ { 0 } = 0$ . We aggregate the projected VLM-induced tail evidence across the ordered threshold grid using a right Riemann sum:

$$
s _ { i } = \sum _ { k = 1 } ^ { K } ( \tau _ { k } - \tau _ { k - 1 } ) \hat { p } _ { i , k } , \qquad \tau _ { 0 } = 0 .\tag{10}
$$

For the uniform grid defined in Eq. (3), $\tau _ { k } - \tau _ { k - 1 } = 1 / K$ so the continuous anomaly score reduces to the mean of the projected tail evidence:

$$
s _ { i } = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \hat { p } _ { i , k } .\tag{11}
$$

The resulting $s _ { i } ~ \in ~ [ 0 , 1 ]$ serves as a continuous ranking statistic that summarizes the VLM’s support across increasing anomaly-severity thresholds. The tail-integral identity motivates this aggregation, but neither $\hat { p } _ { i , k }$ nor $s _ { i }$ is assumed to be probabilistically calibrated.

An important property of the equal-weight formulation is that PAVA preserves the mean of the threshold profile.

$$
\frac { 1 } { K } \sum _ { k = 1 } ^ { K } \hat { p } _ { i , k } = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } p _ { i , k } .\tag{12}
$$

![](images/d67641efbcf5e76ab4e4fbebbbde0371342e81c197a83ee6c1c34f932a514b9d.jpg)  
Fig. 3. Illustration of ordinal consistency projection and cumulative tail integration. (a) Independently estimated tail likelihoods $P ( S \geq \tau _ { k } )$ may violate the required non-increasing order across increasing severity thresholds. PAVA projects them onto a rank-consistent monotone sequence. (b) The projected tail-evidence proxies $\hat { p } _ { i , k }$ are integrated over the severity thresholds to obtain a continuous clip-level anomaly score. For uniformly spaced thresholds, the Riemann sum reduces to the mean of the projected tail evidence.

This follows directly from the pooling operation of PAVA. For each contiguous block B that violates the monotonic constraint, with |B| denoting the number of elements in the block, PAVA replaces all entries by their block mean,

$$
\hat { p } _ { i , k } = \frac { 1 } { | \mathcal { B } | } \sum _ { j \in \mathcal { B } } p _ { i , j } , \qquad k \in \mathcal { B } ,\tag{13}
$$

which preserves the block sum:

$$
\sum _ { k \in B } \hat { p } _ { i , k } = \sum _ { k \in B } p _ { i , k } .\tag{14}
$$

Since the pooled blocks form a disjoint partition of threshold indices, the total sum is preserved. Consequently, under the uniform threshold grid and equal-weight integration in Probe-VAD, PAVA enforces a monotonic, ordinally consistent profile without altering the final scalar anomaly score.

Fig. 3 illustrates the complementary roles of scalar score and intermediate ordinal profile. The integrated tail evidence determines the clip-level ranking score, while the projected profile provides a threshold-wise representation consistent with cumulative severity semantics.

## E. Frame-Level Reconstruction and Efficient Inference

The clip-level scores form a regularly sampled temporal sequence $\mathbf { s } = [ s _ { 0 } , \dots , s _ { M - 1 } ]$ . To obtain frame-level predictions, let $G _ { \sigma }$ denote a unit-sum Gaussian kernel with smoothing width $\sigma ,$ let ∗ denote one-dimensional convolution, and let ¯s denote the smoothed score sequence. We then apply Gaussian smoothing to suppress local temporal fluctuations:

$$
\bar { \mathbf { s } } = \mathbf { s } * G _ { \sigma } ,\tag{15}
$$

where $\sigma$ is measured in units of the clip-score grid. Each smoothed clip score is then assigned to its corresponding ∆- frame interval for subsequent frame-level score reconstruction:

$$
\hat { y } _ { f } = \bar { s } _ { \operatorname * { m i n } ( \lfloor f / \Delta \rfloor , M - 1 ) } , \qquad f = 0 , \dots , F - 1 .\tag{16}
$$

The sequence is finally cropped to the annotated video length.

Since all K ordinal queries for a clip share the same visual input, Probe-VAD encodes each clip only once and reuses the resulting visual representation across all severity thresholds. Only the threshold-dependent textual queries vary, allowing compatible queries to be evaluated in batches and shared prefixes to be cached. These optimizations reduce redundant computation without altering the scoring formulation in Eqs. (4)–(11); the threshold batch size therefore affects computational efficiency and memory usage, but not the resulting anomaly scores.

## IV. EXPERIMENTS

We evaluate Probe-VAD on three VAD benchmarks through comparisons with representative methods, controlled analyses of its core scoring mechanisms, and robustness studies under different visual, linguistic, and temporal configurations.

## A. Experimental Setup

Datasets. Experiments are conducted on three widely used video anomaly detection benchmarks: UCF-Crime [3], MSAD [16], and XD-Violence [17]. UCF-Crime contains 290 test videos covering 13 real-world anomaly categories. The evaluated MSAD test split consists of 360 videos from diverse indoor and outdoor surveillance scenarios. XD-Violence contains 800 test videos collected from surveillance footage, movies, and online videos, covering six categories of events. Evaluation Metrics and Protocol. Frame-level area under the receiver operating characteristic curve (AUC) is adopted as the primary evaluation metric. Following the evaluation convention of prior work [24], we also adopt frame-level Average Precision (AP) for MSAD and XD-Violence. Groundtruth annotations are used exclusively for performance evaluation and are not involved in prompt construction, candidate specification, temporal sampling, or score calibration. All VLM parameters remain frozen throughout evaluation, and the same default configuration is used across datasets unless otherwise specified, ensuring consistent evaluation across all reported benchmark comparisons.

Implementation Details. VideoLLaMA3-7B [30] is used as the default frozen VLM. Unless otherwise specified, each temporal clip spans 10 seconds and contains at most 10 uniformly sampled frames. Probe-VAD uses ten severity thresholds uniformly distributed over [0.1, 1.0], with likelihood temperature $T ~ = ~ 1 . 0$ . Gaussian smoothing with $\sigma ~ = ~ 1 0$ is used for frame-level score reconstruction. Qwen3-VL-8B-Instruct [31] is additionally evaluated to assess portability across frozen VLM backbones. The main VLM inference experiments are conducted on NVIDIA A100 GPUs. The severity-structure analyses operate on retained threshold-wise outputs and require no additional VLM inference.

![](images/e6fce48e241909ba4001e8e7c868a15520398771b44a840570818662d7831dd5.jpg)  
Fig. 4. Prompt configuration shared by Direct Generation and 10-class Likelihood.

![](images/1570260c027492aefe9de15df7a8282e642ff1e4ae33f828ada813fb3b251f60.jpg)

Fig. 5. Prompt configuration used by Probe-VAD.  
![](images/8017bac67e1e903e08938fb02a03d5770f0c573e2a50250293f45b7a1ec83a5d.jpg)  
Fig. 6. Prompt configuration used by the caption-controlled baseline.

## B. Comparison with Representative Methods

Table I compares Probe-VAD with representative trainingbased and training-free VAD methods on UCF-Crime, XD-Violence, and MSAD. Results are reported under the evaluation conventions adopted by the corresponding benchmarks. Since the compared methods differ in supervision, backbone architecture, temporal processing, and preprocessing, the table is intended to provide a system-level comparison rather than a strictly controlled component-wise evaluation.

Among the zero-shot and training-free methods included in Table I, Probe-VAD achieves the best performance across all five directly comparable metrics, reaching 86.27% AUC on UCF-Crime, 92.11% AUC and 75.36% AP on XD-Violence, and 87.55% AUC and 78.43% AP on MSAD. Relative to the strongest comparable baselines in this setting, the corresponding improvements are 1.38 points on UCF-Crime, 0.77 AUC and 5.20 AP points on XD-Violence, and 1.65 AUC and 2.03 AP points on MSAD. The consistent direction of improvement across the three benchmarks suggests that the effectiveness of Probe-VAD is not restricted to a single anomaly distribution. The magnitude of the gain, however, varies across datasets, which is expected because the three benchmarks differ substantially in scene composition, anomaly duration, event complexity, and class imbalance. Probe-VAD does not explicitly model these dataset-specific properties; instead, its advantage arises from changing how visually conditioned evidence is exposed to the anomaly scorer. This makes the consistent improvement across heterogeneous benchmarks particularly relevant to the proposed score-extraction perspective.

Compared with caption-mediated pipelines, Probe-VAD preserves the original visual evidence until the scoring stage, while compared with direct generative approaches it retains response preferences before a single output is selected. Both differences increase the amount of information available for ranking temporal segments. The main results are therefore consistent with our central hypothesis that the performance of a frozen VLM is determined not only by what visual information it encodes, but also by how this information is converted into an anomaly score. The improvement is pronounced in XD-Violence AP. AUC measures pairwise ordering between positive and negative samples over the full score range, whereas precision–recall evaluation is more sensitive to the quality of positive-sample ranking under class imbalance. The larger AP gain therefore suggests that Probe-VAD improves not only the global ordering of frames but also the concentration of anomalous frames toward the high-score region. This behavior is consistent with the higher score resolution obtained by likelihoodbased inference, which reduces the large tied groups induced by discrete numerical generation. Despite using no VADspecific parameter optimization, Probe-VAD remains competitive with several training-based approaches. This observation suggests that useful anomaly information is already accessible from frozen VLMs, while the mechanism used to convert this information into continuous scores strongly affects how effectively it can be exploited. Since the methods in Table I differ in backbone, temporal processing, and preprocessing protocols, this comparison should be interpreted as a systemlevel evaluation. The following controlled analyses isolate the contributions of the proposed scoring design more directly.

## C. Ablation Studies

Unless otherwise specified, all controlled comparisons use the same frozen VLM, temporal sampling, and frame-level reconstruction protocol, with only the factor under investigation being changed. We organize the analysis around three questions: (1) whether retaining direct visual conditioning improves over caption-mediated evidence, (2) whether likelihood-level preferences provide a more informative scoring interface than decoded outputs, and (3) how the resulting ordinal representation behaves under changes in severity discretization, linguistic formulation, visual context, and post-processing.

Effect of Visual Conditioning. We compare direct visual conditioning with a caption-conditioned variant under the same ordinal likelihood scoring protocol. As shown in Table II, direct visual conditioning improves AUC by 5.57, 0.76, and 0.41 points on UCF-Crime, MSAD, and XD-Violence, respectively, supporting the representation-bottleneck hypothesis in Sec. I. Caption generation compresses rich visual observations into compact textual descriptions. While this may preserve dominant event semantics, weaker ranking-relevant cues can be omitted or coarsely expressed. Once visually distinct clips are mapped to similar captions, the downstream scorer cannot recover these lost distinctions. Direct visual conditioning avoids this intermediate compression and therefore preserves a richer basis for continuous anomaly ranking. The larger gain on UCF-Crime further suggests that the impact of caption mediation is dataset dependent. However, the aggregate results do not isolate which specific visual factors cause this difference; we therefore interpret the experiment as evidence that caption mediation can remove ranking-relevant information, rather than attributing the gain to any particular visual cue.

TABLE I  
COMPARISON WITH REPRESENTATIVE VAD METHODS ON UCF-CRIME, XD-VIOLENCE, AND MSAD. ✓ / ×DENOTE WHETHER A METHOD SATISFIES THE ZERO-SHOT AND TRAINING-FREE SETTINGS FOR VAD-SPECIFIC MODEL PARAMETERS. RESULTS ARE REPORTED IN PERCENTAGES, AND “–” DENOTES UNAVAILABLE OR NON-COMPARABLE RESULTS. BOLD AND UNDERLINED VALUES INDICATE THE BEST AND SECOND-BEST PERFORMANCE WITHIN EACH COMPARISON BLOCK, RESPECTIVELY.
<table><tr><td rowspan="2">Method</td><td rowspan="2"></td><td rowspan="2">Zero-shot Training-free</td><td>UCF-Crime</td><td colspan="2">XD-Violence</td><td colspan="2">MSAD</td></tr><tr><td>AUC (%)</td><td>AUC (%)</td><td>AP (%)</td><td>AUC (%)</td><td>AP (%)</td></tr><tr><td>Sultani et al. [3]</td><td>X</td><td>X</td><td>77.92</td><td></td><td>73.20</td><td></td><td>一</td></tr><tr><td>GODS [32]</td><td>X</td><td>X</td><td>70.46</td><td>61.56</td><td></td><td></td><td></td></tr><tr><td>RTFM [20]</td><td>X</td><td>X</td><td>83.31</td><td></td><td>77.81</td><td>86.70</td><td>66.30</td></tr><tr><td>AccI-VAD [33]</td><td>X</td><td>X</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>CLIP-TSA [34]</td><td>X</td><td>X</td><td>87.58</td><td></td><td>82.19</td><td></td><td></td></tr><tr><td>MGFN [35]</td><td>×</td><td>×</td><td>86.98</td><td></td><td>80.11</td><td>85.00</td><td>63.50</td></tr><tr><td>STPrompt [36]</td><td>×</td><td>×</td><td>88.08</td><td></td><td></td><td></td><td></td></tr><tr><td>OVVAD [21]</td><td>×</td><td>×</td><td>86.40</td><td></td><td>66.53</td><td></td><td></td></tr><tr><td>Holmes-VAU [37]</td><td>X</td><td>X</td><td>88.96</td><td></td><td>87.68</td><td></td><td></td></tr><tr><td>MULDE [38]</td><td>X</td><td>X</td><td>78.50</td><td></td><td></td><td></td><td></td></tr><tr><td>EGO [39]</td><td>X</td><td>X</td><td>81.71</td><td></td><td>65.77</td><td>87.30</td><td>64.40</td></tr><tr><td>RVT [1]</td><td>X</td><td>X</td><td>88.13</td><td></td><td>85.77</td><td></td><td></td></tr><tr><td>VERA [25]</td><td>X</td><td>V</td><td>86.55</td><td>88.26</td><td>70.54</td><td></td><td></td></tr><tr><td>TD-VAD [22]</td><td>√</td><td>×</td><td>80.82</td><td>89.50</td><td>75.83</td><td></td><td>一</td></tr><tr><td>UR-DMU (ZS) [40]</td><td></td><td></td><td></td><td></td><td></td><td>74.30</td><td>53.40</td></tr><tr><td>CLIP (ZS) [41]</td><td></td><td></td><td>53.16</td><td>38.21</td><td>17.83</td><td></td><td></td></tr><tr><td>LLaVA-1.5 (ZS) [42]</td><td></td><td></td><td>72.84</td><td>79.62</td><td>50.26</td><td></td><td></td></tr><tr><td>VideoLLaMA3-7B + Llama3.1-8B (ZS) [30], [43]</td><td></td><td></td><td></td><td></td><td></td><td>78.70</td><td>68.50</td></tr><tr><td>GLM-4.1V-9B-Thinking (ZS CoT)‡</td><td></td><td></td><td>61.80</td><td>72.73</td><td>52.93</td><td></td><td></td></tr><tr><td>LAVAD [23]</td><td></td><td></td><td>80.28</td><td>85.36</td><td>62.01</td><td></td><td></td></tr><tr><td>VADTree§ [13]</td><td></td><td></td><td>84.74</td><td>90.44</td><td>67.82</td><td></td><td></td></tr><tr><td>PANDA [44]</td><td></td><td></td><td>84.89</td><td></td><td>70.16</td><td></td><td></td></tr><tr><td>URF-HVAA (fixed-constant setting) [24]</td><td></td><td></td><td>84.28</td><td>91.34</td><td>68.07</td><td>85.90</td><td>76.40</td></tr><tr><td>Probe-VAD (Ours)</td><td></td><td></td><td>86.27</td><td>92.11</td><td>75.36</td><td>87.55</td><td>78.43</td></tr></table>

‡ Zero-shot chain-of-thought (CoT) inference VAD performance using GLM-4.1V-9B-Thinking [45].  
<sup>§</sup> VADTree reports MSAD results on a different 240-video evaluation split (120 normal and 120 anomalous videos), whereas Probe-VAD is evaluated on the 360-video split (120 normal and 240 anomalous videos). Its MSAD values are therefore omitted from the directly comparable ranking.

Effect of the Scoring Interface. To isolate the effect of score extraction from visual understanding, Direct Generation and 10-class Likelihood use identical visual inputs, numericalseverity prompts, and candidate spaces. Direct Generation retains the decoded numerical response, whereas 10-class Likelihood preserves the normalized likelihood distribution over the same ten numerical severity candidates and uses their expected severity as the anomaly score. As shown in Table III, replacing Direct Generation with 10-class Likelihood improves AUC by 14.23, 8.48, and 5.59 points on UCF-Crime, MSAD, and XD-Violence, respectively. Because the two variants share the same visual evidence, prompt, and numerical candidate space, this large gap primarily reflects whether the VLM’s response preferences are retained before autoregressive decoding. Direct Generation reduces the model response to a single selected severity value and therefore discards information about the relative support assigned to alternative candidates. This creates a fundamental mismatch with the evaluation objective of VAD. Frame-level AUC depends primarily on the relative ordering of anomaly scores, whereas a discrete generative interface forces many temporally distinct clips onto the same small set of numerical values. Once two clips receive the same decoded severity, any difference in the VLM’s underlying confidence or preference margin becomes invisible to the ranking metric.

TABLE II  
EFFECT OF VISUAL CONDITIONING. RESULTS ARE FRAME-LEVEL AUC (%).
<table><tr><td>Dataset</td><td>Input</td><td>AUC</td></tr><tr><td>UCF-Crime</td><td>Caption Video</td><td>80.70 86.27</td></tr><tr><td>MSAD</td><td>Caption</td><td>86.79</td></tr><tr><td></td><td>Video</td><td>87.55</td></tr><tr><td>XD-Violence</td><td>Caption</td><td>91.70</td></tr><tr><td></td><td>Video</td><td>92.11</td></tr></table>

This effect is clearly reflected in the empirical prediction distribution. On UCF-Crime, Direct Generation maps all 69,634 evaluated clips to only ten possible scores, and the most frequent value alone accounts for 46.78% of all predictions. Thus, the effective resolution of the generated score is considerably lower than the nominal ten-level output space. Large tied groups make it impossible to rank many clips that the VLM may internally regard as different. Likelihoodbased scoring avoids this collapse by retaining the relative support assigned to all candidate severities before decoding. Two clips that would receive the same generated value can therefore remain distinguishable through differences in their likelihood distributions. The resulting score acts as a continuous preference statistic rather than a discrete decision, which is substantially better aligned with the ranking-based objective of VAD. The gains of 14.23, 8.48, and 5.59 AUC points therefore provide strong evidence that a large portion of anomaly-relevant information is present in the frozen VLM before decoding but is lost when only the final generated response is retained. Probe-VAD further improves over 10-class Likelihood by 1.84, 0.78, and 0.34 AUC points on UCF-Crime, MSAD, and XD-Violence, respectively. The distinction here is more subtle. Flat likelihood scoring treats the ten severity values as competing alternatives, although their semantics are intrinsically ordered. For example, strong support for a high severity value does not explicitly encode that weaker severity conditions should also be satisfied. Probe-VAD instead decomposes severity into nested threshold propositions, converting the prediction problem from competition among independent numerical labels into a sequence of cumulative judgments.

This cumulative formulation better matches the structure of anomaly severity: evidence supporting a high threshold should remain compatible with all lower thresholds. The additional improvement over flat likelihood scoring therefore suggests that preserving model preferences is not sufficient by itself; how those preferences are structured also affects the quality of the resulting ranking signal. The relative magnitudes of the improvements reveal a clear two-stage effect. The dominant gain arises from retaining pre-decoding model preferences, whereas cumulative ordinal probing provides a smaller but consistent additional benefit by imposing a severity-aware factorization of those preferences. Because the candidate semantics and query factorization change jointly between 10- class Likelihood and Probe-VAD, the latter gain should be attributed to the cumulative ordinal interface as a whole rather than to ordinality alone. Moreover, PAVA cannot explain the performance difference because the equal-weight aggregation preserves the final scalar score exactly.

TABLE III  
EFFECT OF DIFFERENT ANOMALY-SCORING INTERFACES. RESULTS ARE FRAME-LEVEL AUC (%).
<table><tr><td>Scoring Interface UCF MSAD XD</td></tr><tr><td>Direct Generation</td></tr><tr><td>70.20 78.29 86.18 84.43 86.77 91.77</td></tr><tr><td>10-class Likelihood</td></tr><tr><td>Probe-VAD 86.27 87.55 92.11</td></tr></table>

Effect of Multi-frame Context and Temporal Order. We compare the default ordered multi-frame input with a shuffledframe variant containing the same observations and a singleframe variant using only the clip center frame. As shown in Table IV, ordered multi-frame input improves AUC from

TABLE IV  
EFFECT OF MULTI-FRAME CONTEXT AND TEMPORAL ORDER ON UCF-CRIME. RESULTS ARE FRAME-LEVEL AUC (%).
<table><tr><td colspan="2">Input</td></tr><tr><td></td><td>AUC</td></tr><tr><td>Single-frame</td><td>83.20</td></tr><tr><td>Shuffled frames</td><td>86.14</td></tr><tr><td>Ordered frames</td><td>86.27</td></tr></table>

83.20% to 86.27%, corresponding to a gain of 3.07 percentage points over the single-frame setting. In contrast, shuffling the same frames reduces performance by only 0.13 points. The pronounced gap between single-frame and multi-frame input indicates that the extracted likelihood signal benefits substantially from observing multiple visual states. A single frame provides only an instantaneous observation and can be ambiguous when normal and abnormal events share similar appearance. Multiple frames expose changes in actors, objects, interactions, and scene configuration, allowing the frozen VLM to accumulate contextual evidence that is unavailable from a single observation.

However, the difference between ordered and shuffled multiframe input is only 0.13 AUC points. This contrast is informative: most of the benefit appears to originate from the availability of multiple complementary visual states rather than from precise temporal ordering itself. In other words, the current VLM can exploit temporal coverage and cross-frame context, but the experiment provides only limited evidence that it performs strong fine-grained sequence reasoning over those frames. This observation also clarifies the role of Probe-VAD. The proposed method does not attempt to introduce a new temporal encoder; instead, it improves how the evidence already exposed by the frozen VLM is converted into an anomaly score. More sophisticated temporal modeling is therefore complementary rather than competing with ordinal likelihood probing, and may further improve performance for anomalies whose interpretation depends strongly on action order or long-range temporal dependencies.

Effect of the VLM Backbone. We replace VideoLLaMA3- 7B with Qwen3-VL-8B-Instruct using the same cumulative ordinal scoring interface. As shown in Table V, the two backbones achieve comparable AUC on all three datasets. Qwen3-VL slightly improves MSAD AUC from 87.55% to 87.74%, while VideoLLaMA3 performs better on UCF-Crime and XD-Violence. These results indicate that ordinal likelihood profiles can be extracted from multiple frozen VLMs and that the proposed scoring interface is not tied to a single backbone. The remaining differences also show that Probe-VAD does not eliminate the influence of the underlying representation. The scoring interface determines how visually conditioned preferences are exposed, whereas the quality of those preferences still depends on the visual encoder, multimodal alignment, and linguistic priors of the frozen VLM.

This distinction is particularly visible on XD-Violence. The two backbones obtain nearly identical AUC values (92.11% versus 91.97%), yet their AP values differ substantially (75.36% versus 70.69%). Similar AUC therefore does not imply identical behavior throughout the ranked score distribution.

TABLE V  
EFFECT OF THE FROZEN VLM BACKBONE. RESULTS ARE PERCENTAGES.
<table><tr><td>Backbone</td><td>Dataset</td><td>AUC</td><td>AP</td></tr><tr><td rowspan="4">VideoLLaMA3-7B</td><td>UCF-Crime</td><td>86.27</td><td></td></tr><tr><td>MSAD</td><td>87.55</td><td>78.43</td></tr><tr><td>XD-Violence</td><td>92.11</td><td>75.36</td></tr><tr><td>UCF-Crime</td><td>84.79</td><td></td></tr><tr><td rowspan="2">Qwen3-VL-8B</td><td>MSAD</td><td>87.74</td><td>77.67</td></tr><tr><td>XD-Violence</td><td>91.97</td><td>70.69</td></tr></table>

The larger AP difference suggests that the two VLMs differ more strongly in how anomalous frames are concentrated toward the highest score region, even when their overall positive–negative ordering remains similar. We therefore interpret this experiment as evidence for portability of the scoring interface across the evaluated VLMs, while recognizing that absolute performance remains backbone dependent.

Effect of Candidate Semantics. We evaluate different candidate formulations on MSAD while keeping the visual input, threshold grid, and scoring procedure unchanged. As shown in Table VI, A/B candidates achieve 86.06% AUC, label-swap averaging reaches 86.21%, polarity-pair averaging reaches 87.27%, and the default YES/NO formulation achieves the highest result of 87.55%. The relatively limited variation indicates that Probe-VAD is not critically dependent on a specific candidate vocabulary. Nevertheless, the consistent difference between neutral A/B labels and semantically meaningful YES/NO responses indicates that candidate semantics influence the likelihood signal. A plausible explanation is that YES/NO directly expresses the truth value of the queried threshold proposition, whereas A/B introduces an arbitrary mapping between token identity and semantic judgment. The latter may expose the likelihood estimate more strongly to token- or position-specific priors.

Label-swap averaging slightly improves the A/B formulation, consistent with reducing arbitrary label preference. Polarity-pair averaging further narrows the gap but requires additional queries and still does not exceed the default formulation. These results support the semantically aligned YES/NO candidate pair as a simple and effective default.

TABLE VI  
EFFECT OF CANDIDATE SEMANTICS ON MSAD.
<table><tr><td>Candidate formulation</td><td>AUC</td></tr><tr><td>A/B</td><td>86.06</td></tr><tr><td>A/B + label swap</td><td>86.21</td></tr><tr><td>YES/NO + polarity pair</td><td>87.27</td></tr><tr><td>YES/NO (default)</td><td>87.55</td></tr></table>

Effect of Prompt Formulation. We evaluate several semantically similar threshold-query formulations on MSAD while fixing the visual input, severity scale, candidate responses, and threshold grid. As shown in Table VII, AUC ranges from 86.95% to 87.76%, while AP ranges from 78.28% to 78.97%. The narrow performance range indicates that the anomaly ranking extracted by Probe-VAD is relatively stable under moderate linguistic reformulation. At the same time, the nonzero variation confirms that conditional likelihoods are not invariant to wording: semantically similar prompts can induce different response-preference distributions in the frozen VLM.

The Rated above formulation achieves the highest numerical result, but exceeds the pre-specified default by only 0.21 AUC and 0.54 AP points. Importantly, this post-hoc best variant is not used for the main results. Retaining a common default prompt across all benchmarks avoids dataset-specific prompt selection and indicates that the reported performance does not depend on choosing the best test-set paraphrase.

TABLE VII  
EFFECT OF THRESHOLD-QUERY WORDING ON MSAD.
<table><tr><td>Prompt variant</td><td>AUC</td><td>AP</td></tr><tr><td>Default</td><td>87.55</td><td>78.43</td></tr><tr><td>Visible evidence</td><td>87.30</td><td>78.28</td></tr><tr><td>Reach level</td><td>86.95</td><td>78.62</td></tr><tr><td>No less than</td><td>86.97</td><td>78.42</td></tr><tr><td>Rated above</td><td>87.76</td><td>78.97</td></tr></table>

Analysis of the Ordinal Severity Structure. We further examine the ordinal severity representation to assess its sensitivity to severity discretization and characterize how the retained evidence varies across severity regions. Specifically, we analyze three aspects: severity resolution, threshold placement, and the responses of individual thresholds and adjacent severity intervals. All analyses are conducted on MSAD using the same scoring and frame-level reconstruction protocol as the default configuration.

For the resolution analysis, we vary the number of ordered thresholds using $K \in \{ 1 , 2 , 5 , 1 0 \}$ while following the predefined rule $\tau _ { k } ~ = ~ k / K$ . Consequently, the $K = 1$ setting contains only $\tau = 1 . 0$ and does not involve test-set-dependent threshold selection. To assess sensitivity to threshold placement, we further fix $K = 5$ and redistribute the thresholds toward different portions of the severity axis.For non-uniform grids, the final score is computed using the interval-width weighted integration in Eq. (10). Beyond the final integrated score, we also characterize the internal severity profile. For two adjacent thresholds, let $d _ { i , k }$ denote the local evidence transition between $\tau _ { k }$ and $\tau _ { k + 1 }$ , defined as

$$
d _ { i , k } = \hat { p } _ { i , k } - \hat { p } _ { i , k + 1 } ,\tag{17}
$$

where $d _ { i , k } \geq 0$ because the PAVA-projected profile is nonincreasing. Since $\hat { p } _ { i , k }$ represents a VLM-induced tail-evidence proxy rather than a calibrated probability, $d _ { i , k }$ is interpreted only as an interval-level evidence transition, rather than probability mass. Threshold-wise and interval-wise AUC/AP values are used solely for structural diagnosis and are not used for model or threshold selection.

TABLE VIII  
EFFECT OF THE NUMBER OF SEVERITY THRESHOLDS ON MSAD. ALL THRESHOLD SETS FOLLOW $\tau _ { k } = k / K .$
<table><tr><td> $K$ </td><td>Thresholds</td><td>AUC</td><td>AP</td></tr><tr><td>1</td><td>{1.0}</td><td>86.97</td><td>76.77</td></tr><tr><td>2</td><td>{0.5, 1.0}</td><td>87.15</td><td>77.51</td></tr><tr><td>5</td><td> $\{ 0 . 2 , \stackrel { \cdot } { 0 . 4 } , \stackrel { \cdot } { \dots } , 1 . 0 \}$ </td><td>87.41</td><td>78.12</td></tr><tr><td>10</td><td> $\{ 0 . 1 , 0 . 2 , \ldots , 1 . 0 \}$ </td><td>87.55</td><td>78.43</td></tr></table>

Table VIII shows that increasing the severity resolution consistently improves both ranking metrics. Relative to the singlethreshold configuration, $K = 1 0$ improves AUC from 86.97% to 87.55% and AP from 76.77% to 78.43%, corresponding to absolute gains of 0.58 and 1.66 percentage points, respectively. This indicates that representing anomaly severity with multiple ordered propositions provides additional information beyond a single severity decision.

Most of the improvement is already obtained with a moderate severity resolution. Specifically, $K = 5$ accounts for approximately 76% of the total AUC gain and 81% of the total AP gain observed between $K = 1$ and $K = 1 0$ . Increasing the resolution from $K = 5$ to $K = 1 0$ yields a further 0.14 AUC points and 0.31 AP points. These results favor a multithreshold representation while indicating diminishing returns once the severity axis is sufficiently populated. We retain $K = 1 0$ because it achieves the best overall performance and provides a finer profile for structural analysis.

Table IX further examines whether the performance gain depends on the particular placement of the thresholds. Across the four $K = 5$ configurations, AUC varies within 87.28– 87.49% and AP within 77.76–78.15%. Notably, the highfocused grid attains the highest numerical result, but exceeds the uniform grid by only 0.08 AUC and 0.03 AP points.

TABLE IX  
SENSITIVITY TO SEVERITY-THRESHOLD PLACEMENT ON MSAD. ALL CONFIGURATIONS USE $K = 5 .$
<table><tr><td>Grid</td><td>Thresholds</td><td>AUC</td><td> $\mathbf { A P }$ </td></tr><tr><td>Uniform</td><td> $\{ 0 . 2 , 0 . 4 , 0 . 6 , 0 . 8 , 1 . 0 \}$ </td><td>87.41</td><td>78.12</td></tr><tr><td>Low-focused</td><td> $\{ 0 . 1 , 0 . 2 , 0 . 3 , 0 . 6 , 1 . 0 \}$ </td><td>87.28</td><td>77.76</td></tr><tr><td>Mid-focused</td><td> $\{ 0 . 1 , 0 . 3 , 0 . 5 , 0 . 7 , 1 . 0 \}$ </td><td>87.31</td><td>77.90</td></tr><tr><td>High-focused</td><td> $\{ 0 . 1 , 0 . 4 , 0 . 7 , 0 . 9 , 1 . 0 \}$ </td><td>87.49</td><td>78.15</td></tr></table>

The narrow variation suggests that the observed performance is not narrowly tied to a specific partition of the severity axis. We therefore retain the uniform grid as a simple, stable, and dataset-independent configuration rather than selecting the numerically best alternative on the test set. This choice also avoids introducing additional dataset-specific tuning into the default setting. This experiment should consequently be interpreted as a sensitivity analysis of threshold placement, rather than as a search for an optimal severity grid.

The following severity-structure analyses are performed offline on retained threshold-wise outputs and are used only to characterize the internal ordinal response structure. They require no additional VLM inference and do not affect model selection or the reported benchmark results.

Fig. 7 visualizes the mean PAVA-projected threshold-wise responses of normal and anomalous clips. Since the nonincreasing shape is explicitly imposed by isotonic projection, monotonicity itself is not treated as an empirical finding. Instead, the relevant observation is the separation between the two groups: anomalous clips retain higher mean tail evidence across the evaluated severity range,while the magnitude of this separation varies across thresholds.

Threshold-wise diagnostics further show that different summary statistics are maximized at different severity levels. The largest mean anomaly–normal difference occurs at $\tau = 0 . 1$ (0.3294), whereas the highest standalone AUC is observed at $\tau = 0 . 8 \ ( 8 7 . 5 4 \% )$ and the highest standalone AP at $\tau = 0 . 4$ (78.39%). Thus, the threshold that maximizes average group separation does not necessarily provide the strongest samplewise ranking across different evaluation metrics.

![](images/90e3916fbeb418d2a59a361134851f5aea78c00bfed0336e95007c0bf14c46d9.jpg)  
Fig. 7. Cumulative severity profiles on MSAD. Group-mean PAVAprojected tail-evidence proxy $\hat { p } _ { i , k }$ is shown separately for normal and anomalous clips across the ten ordered severity thresholds. Shaded regions denote 95% confidence intervals obtained by video-cluster bootstrap over the complete 360-video MSAD test split. Because monotonicity is imposed by PAVA, the informative pattern is the threshold-dependent separation between the two groups rather than the decreasing shape itself.

TABLE X  
INTERVAL-WISE ANALYSIS OF THE ORDINAL SEVERITY STRUCTURE ON MSAD. MEAN $d _ { i , k }$ DENOTES THE INTERVAL EVIDENCE TRANSITION AVERAGED OVER ALL EVALUATED CLIPS, WHILE $\Delta _ { \mathrm { A - N } }$ DENOTES THE DIFFERENCE BETWEEN ANOMALOUS AND NORMAL INTERVAL EVIDENCE.
<table><tr><td>Band</td><td>Mean  $d _ { i , k }$ </td><td> $\Delta _ { \mathrm { A - N } }$ </td><td>AUC (%)</td><td>AP (%)</td></tr><tr><td> $0 . 1 \mathrm { - } 0 . 2$ </td><td>0.0080</td><td>0.0042</td><td>69.23</td><td>51.18</td></tr><tr><td> $0 . 2 \mathrm { - } 0 . 3$ </td><td>0.0089</td><td>0.0010</td><td>54.18</td><td>41.96</td></tr><tr><td> $0 . 3 \mathrm { - } 0 . 4$ </td><td>0.0410</td><td>0.0147</td><td>68.82</td><td>52.86</td></tr><tr><td> $0 . 4 \mathrm { - } 0 . 5$ </td><td>0.0318</td><td>-0.0116</td><td>34.36</td><td>31.75</td></tr><tr><td> $0 . 5 \mathrm { - } 0 . 6$ </td><td>0.0529</td><td>0.0593</td><td>83.09</td><td>64.79</td></tr><tr><td> $0 . 6 { - } 0 . 7$ </td><td>0.0115</td><td>0.0115</td><td>75.70</td><td>60.89</td></tr><tr><td> $0 . 7 \mathrm { - } 0 . 8$ </td><td>0.0200</td><td>0.0158</td><td>80.34</td><td>65.40</td></tr><tr><td> $0 . 8 \mathrm { - } 0 . 9$ </td><td>0.0248</td><td>0.0341</td><td>83.77</td><td>74.79</td></tr><tr><td> $0 . 9 \mathrm { - } 1 . 0$ </td><td>0.0238</td><td>0.0088</td><td>55.32</td><td>53.00</td></tr></table>

This difference reflects the distinct quantities measured by these statistics. Mean separation characterizes the displacement between the two response distributions, whereas AUC depends on pairwise ranking and AP is sensitive to the ordering of positive samples. Their different optima therefore indicate that no single threshold statistic fully characterizes the information contained in the ordinal severity profile.

The interval-wise results in Table X reveal a pronounced non-uniform structure along the severity axis. The 0.5–0.6 interval exhibits both the largest mean transition magnitude (0.0529) and the largest anomalous-minus-normal difference (0.0593), whereas the 0.8–0.9 interval provides the strongest standalone ranking signal, with 83.77% AUC and 74.79% AP. Consistent with the threshold-wise analysis, the interval with the strongest average group separation is therefore not the one

END-TO-END EFFICIENCY ON A FIXED 50-VIDEO MSAD SUBSET. THE SUBSET CONTAINS 2,235 CLIPS AND 22,350 SAMPLED FRAMES.

with the strongest ranking performance.

The 0.4–0.5 interval provides an informative counterexample. Its $\Delta _ { \mathrm { A - N } } ~ \mathrm { i s } ~ { - 0 . 0 1 1 6 }$ , and its standalone AUC is 34.36%, showing that this local transition does not independently behave as a positively oriented anomaly-ranking signal. This does not contradict the PAVA monotonicity constraint: $d _ { i , k }$ remains non-negative within each projected profile, whereas $\Delta _ { \mathrm { A - N } }$ compares its average magnitude between normal and anomalous groups. The resulting heterogeneity suggests that individual intervals should not be interpreted as interchangeable binary detectors; instead, they describe different local changes within the cumulative severity representation.

Overall, these analyses indicate that Probe-VAD is only mildly sensitive to the evaluated discretization choices, while its internal severity representation exhibits substantial threshold- and interval-dependent variation. This behavior supports treating the ordered responses as a joint cumulative profile rather than selecting a single severity level post hoc.

Effect of Clip-internal Frame Sampling. We compare uniform sampling with two center-focused strategies on MSAD under the same ten-frame budget. Uniform, Center-4s, and Center-2s achieve 87.55%, 87.60%, and 87.54% AUC, respectively, with a maximum difference of only 0.06 percentage points. The nearly identical results indicate limited sensitivity to the evaluated frame-sampling strategies across different temporal focuses. We retain uniform sampling as the default because it provides more consistent temporal coverage throughout the clip without introducing additional sampling assumptions or dataset-specific preferences.

TABLE XI  
EFFECT OF CLIP-INTERNAL FRAME SAMPLING ON MSAD. EACH STRATEGY USES TEN INPUT FRAMES.
<table><tr><td>Sampling Strategy</td><td>AUC (%)</td><td>∆AUC</td></tr><tr><td>Uniform (default)</td><td>87.55</td><td></td></tr><tr><td>Center-4s</td><td>87.60</td><td>+0.05</td></tr><tr><td>Center-2s</td><td>87.54</td><td>-0.01</td></tr></table>

Effect of Temporal Smoothing. We evaluate Gaussian smoothing widths $\sigma \in \{ 0 , 5 , 1 0 , 2 0 \}$ on MSAD to examine the sensitivity of frame-level score reconstruction. As shown in Table XII, the performance remains relatively stable under moderate smoothing strengths. The unsmoothed setting achieves 88.13% AUC, while $\sigma = ~ 5$ yields 88.18%, corresponding to only a 0.05-point difference. The default $\sigma = 1 0$ obtains 87.55% AUC, whereas excessive smoothing with $\sigma ~ = ~ 2 0$ reduces performance to 86.65%.

These results indicate that Probe-VAD is not critically dependent on a narrowly tuned smoothing parameter within a moderate range. Gaussian smoothing is used only as a framelevel reconstruction operation to suppress local temporal fluctuations and does not alter the underlying ordinal likelihood scores. The degradation observed under excessive smoothing is consistent with temporal over-smoothing, which may blur short anomaly responses and their boundaries.

TABLE XIII  
EFFECTIVE FPS IS COMPUTED FROM PROCESSED FRAMES, WHILE RAW VIDEOS ARE EXCLUDED FROM THE STORAGE COMPARISON.
<table><tr><td>Pipeline</td><td>Time</td><td>Effective FPS</td><td>Storage</td></tr><tr><td>Probe-VAD (10 thresholds)</td><td>1:58:37</td><td>3.14</td><td>0.77 MB</td></tr><tr><td>Complete caption pipeline</td><td>4:49:19</td><td>1.29</td><td>6.06 MB</td></tr><tr><td>Relative improvement</td><td>59.0% lower</td><td>2.44×</td><td>87.3% lower</td></tr></table>

TABLE XII

EFFECT OF GAUSSIAN SMOOTHING ON MSAD. RESULTS ARE FRAME-LEVEL AUC (%).
<table><tr><td>σ</td><td>AUC (%)</td></tr><tr><td>0</td><td>88.13</td></tr><tr><td>5</td><td>88.18</td></tr><tr><td>10 (default)</td><td>87.55</td></tr><tr><td>20</td><td>86.65</td></tr></table>

## D. Efficiency Analysis

We further compare the end-to-end efficiency of Probe-VAD with the complete caption-mediated pipeline on the same fixed subset of 50 MSAD test videos. The subset contains 2,235 clips, corresponding to 22,350 sampled input frames under the ten-frame-per-clip setting. Both pipelines operate on the same videos and sampled-frame budget.

As shown in Table XIII, Probe-VAD achieves an effective throughput of 3.14 FPS, compared with 1.29 FPS for the complete caption-mediated pipeline, corresponding to a 2.44× increase in end-to-end throughput. Probe-VAD additionally reduces persistent intermediate artifact storage from 6.06 MB to 0.77 MB, an 87.3% reduction.

Here, effective FPS is computed from the sampled RGB frames actually processed by the model rather than all original video frames. The 50-video subset contains 35,524 original annotated frames, while 22,350 frames are sampled and provided to the VLM. Transient visual representations are not counted as persistent storage unless explicitly materialized to disk.

Importantly, the storage comparison concerns persistent intermediate artifacts rather than the original visual input or transient runtime representations. Both pipelines use the same source videos and temporal sampling protocol, so raw videos are excluded. Transient visual features are counted only if explicitly materialized to disk.

## V. DISCUSSION AND LIMITATIONS

Ordinal Consistency. The cumulative formulation assumes that support should not increase as the severity threshold becomes more demanding. Although the threshold propositions share the same formulation and differ only in their severity thresholds, their likelihood preferences are not explicitly constrained to satisfy this monotonic relation. We therefore examine how frequently the raw severity profiles violate the expected ordering and how strongly PAVA modifies them.

The maximum observed difference between scalar scores before and after PAVA is only $4 . 4 4 \times 1 0 ^ { - 1 6 }$ , which is consistent with floating-point precision and the sum-preservation property in Eq. (12). PAVA therefore plays a structural rather than performance-enhancing role in the current formulation: it converts the raw threshold-wise likelihood preferences into an ordinally consistent profile without altering the scalar ranking statistic used for evaluation.

![](images/561bee30817d4aad2007a9ef92425bdcd6245cb7db9ae60609764b06f2b78b6b.jpg)  
A woman walks toward a black truck.

(a) Three-method comparison  
RoadAccidents127\_x264  
![](images/ea0a3af2f8986126146e869dcf4a2554b08c74c3a93c77e2dd13d35f47165222.jpg)

(b) Likelihood vs DirectGeneration  
![](images/670070ff4027b40716301e2f249483cd4db2ef0382dbf9db8345963314272301.jpg)

![](images/ba5e11fb38e5b9e702c4a894448ac41b7116ef740214382a954801d343d6453f.jpg)  
(b) Direct-generation tie

![](images/724294b6ff87daa5ab45f2f7246fb02ff5198c33770a16b91def02bdc96db8a9.jpg)  
Double border = method max + GT frame  
Fighting047 · two clips

Fig. 8. Representative temporal score profiles. (a) Probe-VAD reaches its maximum within the annotated road-accident interval, whereas the compared training-free methods exhibit less aligned temporal responses. (b) Likelihood-based scoring reaches its maximum within the annotated explosion interval, while Direct Generation peaks substantially later.  
![](images/6a915f97a7877813e7d26ff2d67503c2e489fd45934b34a8fa6e204eed49908a.jpg)

(d) False negative  
![](images/d3d5035db941d690684d10e21faa8bc56c47574f4c047ec11297e0283fa8886d.jpg)  
Fig. 9. Qualitative comparison and representative failure cases. (a) Caption mediation omits anomaly-relevant interaction cues. (b) Direct numerical generation assigns the same score to visually distinct clips, whereas Probe-VAD separates them through likelihood-level preferences. (c) A false positive associated with salient but benign motion. (d) A false negative involving brief, small, and partially occluded visual evidence.

As shown in Table XIV, 75.33–87.99% of clips contain at least one adjacent ordering violation before projection, indicating that semantic nesting alone does not guarantee exact monotonicity. Nevertheless, the mean element-wise correction remains small, ranging from 0.002031 to 0.003827, suggesting that most violations are local perturbations rather than large departures from the expected severity order.

TABLE XIV  
ORDINAL-CONSISTENCY DIAGNOSTICS BEFORE PAVA. “VIOL.” DENOTES THE PERCENTAGE OF CLIPS WITH ANY ADJACENT ORDERING VIOLATION, AND δ<sub>PAVA</sub> DENOTES THE ABSOLUTE ELEMENT-WISE CORRECTION.
<table><tr><td>Dataset</td><td>Viol. (%)</td><td>Mean ∆</td><td>Max ∆</td></tr><tr><td>UCF</td><td>87.99</td><td>0.002031</td><td>0.114160</td></tr><tr><td>MSAD</td><td>75.33</td><td>0.002525</td><td>0.123895</td></tr><tr><td>XD</td><td>80.00</td><td>0.003827</td><td>0.235396</td></tr></table>

Qualitative Analysis. Fig. 8 presents representative temporal score profiles complementing the controlled comparisons. In the road-accident example, Probe-VAD reaches its strongest response within the annotated anomaly interval, while compared training-free methods exhibit less aligned temporal responses. In the explosion example, likelihood-based scoring peaks within the annotated interval, whereas Direct Generation peaks substantially later. Fig. 9 further illustrates the two score-extraction bottlenecks and representative failure cases. Caption mediation may omit anomaly-relevant interaction cues, while visually distinct clips can receive the same decoded severity despite remaining distinguishable at the likelihood level. The false-positive and false-negative cases instead expose limitations outside the score-extraction interface, including salient but benign motion and brief, small, or partially occluded anomaly evidence. These examples are intended as qualitative illustrations rather than aggregate evidence.

Limitation.One limitation of our method is that it inherits perception errors from the frozen VLM, which cannot be resolved through score extraction alone. This is a common limitation for training-free VLM-based approaches. Such perception errors may include mistaking salient but benign motion for anomalous activity, or failing to recognize brief abnormal interactions, small anomaly-related objects, and partially occluded events. These errors may be more pronounced in crowded scenes, low-resolution footage, or ambiguous events. In these cases, Probe-VAD can better preserve and organize the evidence perceived by the VLM, but cannot recover visual evidence that the underlying model itself fails to capture.

## VI. CONCLUSION

We introduced Probe-VAD, a training-free framework that translates the visual evidence of a frozen VLM into continuous anomaly-ranking scores through direct visual conditioning and ordinal likelihood probing. Rather than compressing visual observations into captions or relying on a single decoded numerical response, Probe-VAD retains likelihood-level preferences over ordered severity propositions and integrates the resulting cumulative evidence into a continuous ranking statistic. Experiments across UCF-Crime, MSAD, and XD-Violence show that direct visual conditioning consistently outperforms matched caption input, likelihood-based scoring substantially improves over direct numerical generation, and the cumulative binary-threshold interface further improves over flat 10-class likelihood scoring. Probe-VAD achieves frame-level AUC values of 86.27%, 87.55%, and 92.11% on UCF-Crime, MSAD, and XD-Violence, respectively. Experiments with Qwen3-VL-8B-Instruct further show that the proposed score-extraction interface is not tied to a single frozen VLM backbone. Together, these results demonstrate that retaining direct visual evidence, preserving likelihood-level preferences, and organizing them according to ordinal severity structure provide an effective interface for converting pretrained VLM understanding into continuous temporal anomaly rankings.

## REFERENCES

[1] S. Zhang, W. Xu, Y. Fang, F. Lyu, L. Ma, G. Zhao, F. Zhao, C. Shan, and L. Wang, “Reconstructive visual tuning for weakly supervised video anomaly detection,” IEEE Transactions on Information Forensics and Security, 2026.

[2] P. Wu, C. Pan, Y. Yan, G. Pang, Q. Yan, P. Wang, and Y. Zhang, “Deep learning for video anomaly detection: A review,” IEEE Transactions on Neural Networks and Learning Systems, 2026.

[3] W. Sultani, C. Chen, and M. Shah, “Real-world anomaly detection in surveillance videos,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2018, pp. 6479–6488.

[4] W. Dong, Z. Wang, S. Zhang, K. Sun, B. Li, G.-S. Xie, C. Shan, and F. Zhao, “Cl-anomaly: Layer-adaptive mixture-of-experts with multimodal large language model for continual learning in anomaly detection,” arXiv preprint arXiv:2607.02930, 2026.

[5] Y. Duan, W. Xu, Q. Wu, G.-S. Xie, F. Zhao, and C. Shan, “Anomalycontrol: Highly-aligned anomalous image generation with controlled diffusion model,” in Proceedings of the 33rd ACM International Conference on Multimedia, 2025, pp. 8048–8057.

[6] C. Cao, Y. Lu, and Y. Zhang, “Context recovery and knowledge retrieval: A novel two-stream framework for video anomaly detection,” IEEE Transactions on Image Processing, vol. 33, pp. 1810–1825, 2024.

[7] J. Leng, M. Tan, X. Gao, W. Lu, and Z. Xu, “Anomaly warning: Learning and memorizing future semantic patterns for unsupervised exante potential anomaly prediction,” in Proceedings of the 30th ACM International Conference on Multimedia, 2022, pp. 6746–6754.

[8] P. Wu and J. Liu, “Learning causal temporal relation and feature discrimination for anomaly detection,” IEEE Transactions on Image Processing, vol. 30, pp. 3513–3527, 2021.

[9] P. Wu, X. Zhou, G. Pang, L. Zhou, Q. Yan, P. Wang, and Y. Zhang, “Vadclip: Adapting vision-language models for weakly supervised video anomaly detection,” in Proceedings of the AAAI conference on artificial intelligence, vol. 38, 2024, pp. 6074–6082.

[10] S. Zhang, W. Dong, W. Xu, G. Chu, Y. Fang, F. Lyu, G. Zhao, and F. Zhao, “Contextual clue mining and class calibration for weakly supervised video anomaly detection,” in ICASSP 2026-2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2026, pp. 10 687–10 691.

[11] J.-B. Alayrac, J. Donahue, P. Luc, A. Miech, I. Barr, Y. Hasson, K. Lenc, A. Mensch, K. Millican, M. Reynolds et al., “Flamingo: a visual language model for few-shot learning,” Advances in neural information processing systems, vol. 35, pp. 23 716–23 736, 2022.

[12] Y. Shao, H. He, S. Li, S. Chen, X. Long, F. Zeng, Y. Fan, M. Zhang, Z. Yan, A. Ma et al., “Eventvad: Training-free event-aware video anomaly detection,” arXiv preprint arXiv:2504.13092, 2025.

[13] W. Li, Y. Xu, Y. Rao, Z. Wang, and S. Deng, “Vadtree: Explainable training-free video anomaly detection via hierarchical granularity-aware tree,” in The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025.

[14] H. Lim and Y. Hur, “Corevad: A contextual reasoning framework for training-free video anomaly detection,” in International Conference on Pattern Recognition. Springer, 2026, pp. 418–432.

[15] H. Lee, H. Kim, I.-J. Kim, and Y. Choi, “Flashback: Memorydriven zero-shot, real-time video anomaly detection,” arXiv preprint arXiv:2505.15205, 2025.

[16] L. Zhu, L. Wang, A. Raj, T. Gedeon, and C. Chen, “Advancing video anomaly detection: A concise review and a new dataset,” Advances in Neural Information Processing Systems, vol. 37, pp. 89 943–89 977, 2024.

[17] P. Wu, J. Liu, Y. Shi, Y. Sun, F. Shao, Z. Wu, and Z. Yang, “Not only look, but also listen: Learning multimodal violence detection under weak supervision,” in European conference on computer vision. Springer, 2020, pp. 322–339.

[18] M. Hasan, J. Choi, J. Neumann, A. K. Roy-Chowdhury, and L. S. Davis, “Learning temporal regularity in video sequences,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2016, pp. 733–742.

[19] W. Liu, W. Luo, D. Lian, and S. Gao, “Future frame prediction for anomaly detection – a new baseline,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018, pp. 6536–6545.

[20] Y. Tian, G. Pang, Y. Chen, R. Singh, J. W. Verjans, and G. Carneiro, “Weakly-supervised video anomaly detection with robust temporal feature magnitude learning,” in Proceedings of the IEEE/CVF international conference on computer vision, 2021, pp. 4975–4986.

[21] P. Wu, X. Zhou, G. Pang, Y. Sun, J. Liu, P. Wang, and Y. Zhang, “Openvocabulary video anomaly detection,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 18 297–18 307.

[22] S. Zhang, L.-L. Ma, Z. Wang, W. Dong, X. Xu, G.-S. Xie, C. Shan, and F. Zhao, “Td-vad: Breaking visual dependence in video anomaly detection with text-driven learning,” arXiv preprint arXiv:2608.11820, 2026.

[23] L. Zanella, W. Menapace, M. Mancini, Y. Wang, and E. Ricci, “Harnessing large language models for training-free video anomaly detection,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 18 527–18 536.

[24] D. Lin, M. Qu, K. Han, J. Jiao, X. Jin, and Y. Wei, “A unified reasoning framework for holistic zero-shot video anomaly analysis,” Advances in Neural Information Processing Systems, 2025.

[25] M. Ye, W. Liu, and P. He, “Vera: Explainable video anomaly detection via verbalized learning of vision-language models,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 8679–8688.

[26] H. <sup>˙</sup>I. Ozt<sup>¨</sup> urk and A. B. Can, “Vador: Real world video anomaly detection¨ with object relations and action,” in Proceedings of the British Machine Vision Conference (BMVC), 2023, pp. 893–898.

[27] G. Pang, C. Yan, C. Shen, A. van den Hengel, and X. Bai, “Selftrained deep ordinal regression for end-to-end video anomaly detection,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2020, pp. 12 170–12 179.

[28] Z. Zhao, E. Wallace, S. Feng, D. Klein, and S. Singh, “Calibrate before use: Improving few-shot performance of language models,” in Proceedings of the 38th International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, vol. 139. PMLR, 2021, pp. 12 697–12 706.

[29] Y. Kwon, D. Moon, Y. Oh, and H. Yoon, “LogicQA: Logical anomaly detection with vision language model generated questions,” in Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 6: Industry Track). Association for Computational Linguistics, 2025, pp. 411–432.

[30] B. Zhang, K. Li, Z. Cheng, Z. Hu, Y. Yuan, G. Chen, S. Leng, Y. Jiang, H. Zhang, X. Li et al., “Videollama 3: Frontier multimodal foundation models for image and video understanding,” arXiv preprint arXiv:2501.13106, 2025.

[31] S. Bai, Y. Cai, R. Chen et al., “Qwen3-VL technical report,” arXiv preprint arXiv:2511.21631, 2025.

[32] J. Wang and A. Cherian, “Gods: Generalized one-class discriminative subspaces for anomaly detection,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2019, pp. 8201–8211.

[33] T. Reiss and Y. Hoshen, “An attribute-based method for video anomaly detection,” arXiv preprint arXiv:2212.00789, 2022.

[34] H. K. Joo, K. Vo, K. Yamazaki, and N. Le, “Clip-tsa: Clip-assisted temporal self-attention for weakly-supervised video anomaly detection,” in 2023 IEEE International Conference on Image Processing (ICIP). IEEE, 2023, pp. 3230–3234.

[35] Y. Chen, Z. Liu, B. Zhang, W. Fok, X. Qi, and Y.-C. Wu, “Mgfn: Magnitude-contrastive glance-and-focus network for weakly-supervised video anomaly detection,” in Proceedings of the AAAI conference on artificial intelligence, vol. 37, 2023, pp. 387–395.

[36] P. Wu, X. Zhou, G. Pang, Z. Yang, Q. Yan, P. Wang, and Y. Zhang, “Weakly supervised video anomaly detection and localization with spatio-temporal prompts,” in Proceedings ofthe 32nd ACM International Conference on Multimedia, 2024, pp. 9301–9310.

[37] H. Zhang, X. Xu, X. Wang, J. Zuo, X. Huang, C. Gao, S. Zhang, L. Yu, and N. Sang, “Holmes-VAU: Towards long-term video anomaly understanding at any granularity,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025, pp. 13 843–13 853.

[38] J. Micorek, H. Possegger, D. Narnhofer, H. Bischof, and M. Kozinski, “Mulde: Multiscale log-density estimation via denoising score matching for video anomaly detection,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 18 868– 18 877.

[39] D. Ding, L. Wang, L. Zhu, T. Gedeon, and P. Koniusz, “Learnable expansion of graph operators for multi-modal feature fusion,” in International Conference on Learning Representations, vol. 2025, 2025, pp. 74 263– 74 285.

[40] H. Zhou, J. Yu, and W. Yang, “Dual memory units with uncertainty regulation for weakly supervised video anomaly detection,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 37, 2023, pp. 3769–3777.

[41] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark et al., “Learning transferable visual models from natural language supervision,” in International Conference on Machine Learning, 2021, pp. 8748–8763.

[42] H. Liu, C. Li, Y. Li, and Y. J. Lee, “Improved baselines with visual instruction tuning,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2024, pp. 26 296–26 306.

[43] A. Grattafiori, A. Dubey, A. Jauhri, A. Pandey, A. Kadian, A. Al-Dahle, A. Letman, A. Mathur, A. Schelten, A. Vaughan et al., “The llama 3 herd of models,” arXiv preprint arXiv:2407.21783, 2024.

[44] Z. Yang, C. Gao, and M. Z. Shou, “Panda: Towards generalist video anomaly detection via agentic ai engineer,” Advances in Neural Information Processing Systems, vol. 38, pp. 83 182–83 211, 2026.

[45] W. Hong, W. Yu, X. Gu, G. Wang, G. Gan, H. Tang, J. Cheng, J. Qi, J. Ji, L. Pan et al., “Glm-4.5 v and glm-4.1 v-thinking: Towards versatile multimodal reasoning with scalable reinforcement learning,” arXiv preprint arXiv:2507.01006, 2025.