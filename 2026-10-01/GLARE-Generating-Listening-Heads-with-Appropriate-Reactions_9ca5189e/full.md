# GLARE: Generating Listening Heads with Appropriate Reactions

Zikai Liao<sup>1</sup> Yumin Suh<sup>2</sup> Yi Ouyang<sup>2</sup> Yi-Lun Lee<sup>2</sup> Yi-Hsuan Tsai<sup>2</sup> Zhaozheng Yin<sup>1</sup> Department of Computer Science, Stony Brook University<sup>1</sup> Atmanity Inc.<sup>2</sup>

## Abstract

While talking head generation has advanced rapidly, generating natural listener behavior in dyadic conversations, which know when to react, how to react, and with what type of response, remains underexplored. Existing dyadic datasets lack fine-grained listener reaction annotations, and prevailing evaluation metrics inherited from talking-head and video generation measure visual realism rather than whether a listener reacted appropriately. We address these gaps along three aspects. First, we curate a listening-head-specific dataset built from RealTalk and Seamless Interaction, comprising approximately 147 hours of paired speaker–listener videos with 64,557 event-level reaction annotations across six categories: nodding, head shaking, smiling, laughing, frowning, and surprised. Second, we introduce an audio-driven baseline built on a flow-matching transformer, namely GLARE, with prosody conditioning derived from Qwen2-Audio and a temporal reaction loss that explicitly supervises frame-wise reactions. Third, we propose a reaction-oriented evaluation protocol that jointly measures reaction occurrence (R-F1), temporal alignment (R-tIoU), asymmetric temporal deviation (R-ATD), and reaction-region visual quality (R-FID), giving a more behaviorally grounded assessment than visual-quality-only metrics. Experiment results show consistent gains over prior listening-head methods in both visual fidelity and reaction-level metrics, suggesting that reaction-aware data, modeling, and evaluation are critical for natural listening behavior. The code, annotation pipeline, and processed dataset will be released at https://github.com/lzk901372/glare.

## 1 Introduction

Recent advances in talking head generation have greatly improved lip synchronization, identity preservation, and photorealistic rendering [14; 22; 32; 16; 18; 43; 12; 47]. Most existing methods, however, focus on animating the speaker. In face-to-face conversation, the listener is equally important: non-verbal behaviors, such as brief reactions like smiling and nodding, convey attention, agreement, hesitation, and affective engagement [41]. The timing and type of these reactions strongly affect perceived naturalness, since visually plausible motions can still appear unnatural when reactions are absent, mistimed, or inconsistent with the conversational context [24; 25; 9].

Listening head generation has recently emerged as a distinct task. ViCo [46] introduced an early benchmark, while subsequent works explored non-deterministic listener motion, language-aware responses, emotion conditioning, improved sequence modeling, and real-time dyadic interaction [25; 26; 35; 20; 47; 15]. Despite this progress, reaction-aware listening head generation remains limited by two key bottlenecks. First, existing datasets are not designed around fine-grained listener reactions. ViCo provides task-specific supervision but is limited in scale, while larger dyadic resources such as RealTalk [12], Seamless Interaction [1], and SpeakerVid-5M [44] offer rich conversational videos without event-level annotations of reaction type, timing, and duration. As a result, current data provide insufficient supervision for modeling when a listener should react, what reaction should occur, and how the reaction should unfold over time.

Second, existing evaluation protocols mainly rely on metrics inherited from talking-head or video generation, such as reconstruction fidelity, perceptual realism, and motion quality. These metrics are useful for measuring global visual quality, but they do not explicitly assess whether generated listener reactions are type-consistent, temporally aligned with conversational cues, or visually realistic within reaction segments. This is particularly important because listener feedback is inherently multi-valid: an appropriate response may not exactly match a single reference timestamp, yet its type, timing, and quality should still be evaluated in a reaction-aware manner [9].

To address these limitations, we revisit listening head generation from the perspectives of data, modeling, and evaluation. We curate a listening-head-specific dataset from RealTalk and Seamless Interaction by extracting aligned speaker-listener portrait pairs and annotating listener reactions as categorized temporal events. The resulting dataset contains approximately 147 hours of paired video data and 64,557 reaction instances. Based on this dataset, we introduce an audio-driven listening head generation method GLARE with prosody conditioning and a temporal reaction loss, enabling the model to better capture speaker-side cues and generate appropriate listener reactions at the right moment. We further propose reaction-centric evaluation metrics that measure reaction occurrence, temporal alignment, asymmetric timing deviation, and reaction-region visual quality.

## Our main contributions are as follows:

1. We curate a listening-head dataset built from dyadic conversational videos, with aligned videoaudio pairs and event-level annotations of listener reactions, comprising approximately 147 hours of video data and 64,557 reaction instances.

2. We introduce an audio-driven reacting listener, namely GLARE, for listening head generation, equipped with a dedicated prosody conditioning mechanism and a temporal reaction loss.

3. We propose reaction-oriented evaluation metrics for listening head generation that jointly measures visual fidelity, reaction accuracy, and temporal alignment, enabling finer-grained assessment of conversational appropriateness with extensive experiments comparing existing methods.

## 2 Related Works

Listening head generation. Listening head generation synthesizes non-verbal listener behaviors conditioned on speaker-side conversational cues. ViCo [45] first established responsive listening head generation as a benchmark, and Learning2Listen [25] modeled listener motion as a non-deterministic distribution to reflect the multi-valid nature of dyadic interaction. Later methods improved listener synthesis with emotion conditioning [35], non-autoregressive sequence modeling [20], diffusionbased generation [38], dyadic interaction modeling [41], and real-time multimodal interaction [47; 15]. However, most existing methods optimize holistic motion realism, diversity, or speaker-listener synchrony, without explicitly modeling listener reactions as categorized and temporally localized events. Our work instead focuses on generating reactions with appropriate type, timing and duration.

Interactive conversation datasets. Dyadic conversation datasets provide important resources for modeling social interaction. Early corpora such as RECOLA [33] and NoXi [5] support affective and feedback behavior analysis, while ViCo [45] introduced a task-specific benchmark for responsive listening heads. Larger resources such as RealTalk [12], Seamless [1], and SpeakerVid-5M [44] provide more diverse conversational videos, but they are not designed around fine-grained listener reaction events. In contrast, our dataset constructs aligned speaker-listener clips with explicit eventlevel annotations of reaction type, timing, and duration.

Evaluation methods. Listener generation is commonly evaluated using backchannel prediction metrics, such as precision, recall, F1, tolerance windows, and subjective appropriateness judgments [23; 10], or visual generation metrics such as FID [34], FVD [39], and LPIPS [42]. These metrics measure feedback timing or global visual quality, but do not jointly assess whether a generated listener reacts with the correct type, at the correct moment, and with realistic reaction-specific motion. Recent reaction generation benchmarks emphasize appropriateness, diversity, synchrony, and realism [36; 37]. Our evaluation further treats reactions as categorized temporal events and measures reaction occurrence, temporal alignment, asymmetric timing deviation, and reaction-region quality.

![](images/379f4abaf48b497f4569027fa3a18af9c9cc2ac6c603d3dd5b7d2af40885e572.jpg)  
Figure 1: Overview of our data curation pipeline: Step 1, we leverage conversational data from the RealTalk and Seamless datasets; Step 2, videos are cropped into facial regions; Step 3, we apply audio separation and diarization to disentangle listening and speaking segments in each conversation with timestamps, and pair speaker audio clips with corresponding listener clips; and Step 4, we define the reaction classes and detect reactions in each paired clip, resulting in multi-class frame-wise reaction annotations.

## 3 Dataset Curation

Our goal is to construct a listening-head-specific dataset that supports not only audio-driven listener head generation, but also fine-grained modeling and evaluation of listener reactions. To this end, we curate dyadic conversational videos into temporally aligned speaker–listener pairs and further annotate the listener’s visible reactions as categorized temporal events. The overall pipeline is shown in Fig. 1. Starting from RealTalk [11] and Seamless Interaction [1], we obtain 107,149 speaker audio/listener-video pairs, corresponding to approximately 147 hours of data.

## 3.1 Speaker–Listener Clip Construction

As shown in Fig.1 (2) and (3), we first process the raw dyadic videos to obtain high-quality portrait clips for both conversational participants. Since the original videos may contain scene transitions, black frames, irrelevant content, multiple visible people, severe occlusions, or unstable face crops, visual quality filtering is performed before clip extraction. For retained videos, each participant is cropped into a portrait-centered video and an additional manual face-quality check is conducted to remove clips with missing, partial, or poorly framed faces. This step ensures that the resulting listener videos contain stable facial regions suitable for motion generation and reaction annotation.

We then parse the speaking and listening roles over time. For RealTalk, where the two speakers may be mixed in the original audio, we perform audio separation, denoising, and speaker diarization to estimate participant-level speech activity. For Seamless Interaction, we use the provided high-quality audio streams and diarization results, with timestamp correction when necessary. Based on the resulting speech-activity timelines, each dyadic conversation is decomposed into speaker-dominant intervals. For each interval, we pair the active speaker’s audio with the temporally synchronized portrait video of the other participant, who is treated as the listener. This produces paired training examples in the form of speaker audio and listener video, while preserving the conversational timing between the two participants.

Implementation details of speaker-listener clip reconstruction, including filtering criteria, face-crop procedure, diarization correction, and quality-control procedures, are provided in Appendix A.

Table 1: Statistics of the annotated listener reactions in our curated dataset.
<table><tr><td>Reaction type</td><td colspan="5">nodding head shaking smiling laughing frowning surprised Total</td></tr><tr><td>Count</td><td>10,230 12,790</td><td>15,767</td><td>8,320</td><td>9,986 7,464</td><td>64,557</td></tr></table>

## 3.2 Reaction Taxonomy and Annotation

A central objective of our dataset is to explicitly capture listener reactions that are semantically meaningful and temporally localized. We focus on six common reaction categories: nodding, head shaking, smiling, laughing,frowning, and surprised, which are visually recognizable and frequently associated with conversational feedback and affective responses [25; 46; 12]. Compared with microbehaviors (e.g., eye blinking or gaze shifts), and background motions (e.g., subtle pose adjustments), these reactions have four desirable properties: they convey clear communicative or affective intent, are visually recognizable and are strongly correlated with the conversational context. These properties make them suitable targets for reaction-aware listener generation and reaction-centric evaluation.

As shown in Fig.1 (4), to annotate listener reactions at scale, we design a visual reaction detector to the cropped listener videos. For a listener clip with T frames, the detector outputs a dense reaction score vector $\mathbf { r } \in [ 0 , 1 ] ^ { T \times 6 }$ , where each channel corresponds to one of the six reaction categories. The detector combines head-motion cues, facial landmark dynamics, facial expression cues, and temporal smoothing to estimate frame-wise reaction confidence. Continuous high-confidence regions are then converted into event-level annotations $A = \{ ( y _ { i } , r _ { i } , t _ { s , i } , t _ { e , i } ) \} _ { i = 1 } ^ { N }$ , where $y _ { i } \in \mathcal { V }$ is the event type from the six reaction classes, $r _ { i }$ is the frame-wise reaction score within $[ t _ { s , i } , t _ { e , i } ]$ , and $t _ { s , i }$ and $t _ { e , i }$ are the start and end times. Each final event is assigned a single type, and cross-class overlaps are resolved by keeping the dominant reaction. The resulting annotations therefore provide both sparse temporal boundaries for event-level evaluation and dense frame-wise scores for reaction-aware training supervision.

We further conduct manual verification to remove unreliable detections and improve annotation quality. The final curated dataset contains 64,557 reaction instances in total. The distribution across reaction categories is summarized in Table 1. Smiling and head shaking are the most frequent reaction types, while surprised reactions occur less often, reflecting their relatively sparse and context-specific nature in natural conversations. The detailed detector design, including class-specific visual cues, smoothing strategies, event merging rules, and thresholds, can be referred to Appendix B.

## 4 Methodology of GLARE

## 4.1 Preliminaries

We present the overview of our listener GLARE in Fig. 2.A. Similar to FLOAT [17], given a listener source image S and a speaker audio segment a, the goal is to synthesize a listener motion-latent trajectory that can be decoded into listener video frames. We use LIA [40] as a frozen image autoencoder to extract listener reference motion latent $w _ { r } \in \mathbb { R } ^ { ( T _ { \mathrm { p r e v } } + T _ { \mathrm { c u r } } ) \times d _ { r } 1 }$ and identity latent $w _ { i }$ from S, and use Wav2Vec2 [2] to encode a into audio latents $w _ { a } \in \mathbb { R } ^ { ( T _ { \mathrm { p r e v } } + T _ { \mathrm { c u r } } ) \times d _ { a } 1 }$ . Note that we utilize $T _ { \mathrm { p r e v } } + T _ { \mathrm { c u r } }$ frames of audio for generation, where $T _ { \mathrm { c u r } }$ denotes current frames for listener motion generation, $T _ { \mathrm { p r e v } }$ indicate previous frames of audio context. Additionally, an emotion encoder [30] is used to encode the audio into an emotion latent $w _ { e } \in \mathbb { R } ^ { ( T _ { \mathrm { p r e v } } + T _ { \mathrm { c u r } } ) \times d _ { e } 1 }$ to represent speaker emotions. $w _ { r } , w _ { a }$ , and $w _ { e } ,$ , along with $w _ { p }$ (introduced in Sec 4.2), jointly construct the driving condition c. We then use c is to modulate noisy latents $x = [ x ^ { \mathrm { r e f } } ; x ^ { \mathrm { p r e v } } ; x ^ { \mathrm { c u r } } ] \in \mathbb { R } ^ { T _ { \mathrm { f u l l } } \times d _ { w } 1 }$ covering $T _ { \mathrm { f u l l } } = T _ { \mathrm { r e f } } + T _ { \mathrm { p r e v } } + T _ { \mathrm { c u r } }$ frames, where $T _ { \mathrm { r e f } }$ is reference listener frames providing more listener-side contexts. We adopt a DiT-style transformer [29] to predict the vector fields via flow matching [19; 17]. The vector fields will be further solved into listener flowed latent xˆ, which will be integrated with listener identity latent $w _ { i }$ and decoded into video frames.

## 4.2 Prosody Conditioning

While conventional audio-driven listener generation mainly relies on low-level acoustic representations, listener reactions are often triggered by fine-grained speaker-side temporal cues, such as emphasis, affective fluctuation, and non-monotone prosodic changes. To make the driving condition more sensitive to such conversational cues, we introduce a prosody conditioning branch, where we further extract a frame-aligned prosodic intensity timeline using Qwen2-Audio-7B-Instruct [8] in addition to $w _ { r } , w _ { a }$ , and $w _ { e }$ . The full prompt template is given in Appendix C.2.

![](images/cf7520cfeafc4b871220b6cf9a6bdc38f1dcb8485ccb716bbbf1656af927856b.jpg)  
Figure 2: Overview of our Prosody-conditioned Reacting Listener GLARE. Given speaker audio, a listener reference image, and temporal context, our model predicts listener motion latents using a conditional flow matching transformer (FMT). We further introduce prosody conditioning to capture speaker-side temporal prosodic variations, and a temporal reaction loss to explicitly supervise framewise listener reactions. The resulting motion latents are decoded into listener video frames.

Specifically, as shown in Fig. 2.B, we use the Audio LLM [8] to analyze the audio prosody and obtain timestamps indicating prosodic fluctuations, which will be further processed into a framealigned prosody intensity scalar $p \in [ 0 , 1 ] ^ { ( T _ { \mathrm { p r e v } } + T _ { \mathrm { c u r } } ) \times 1 }$ for each sample, where larger values indicate stronger prosodic or temporal affective variation in the speaker’s speech. The scalar prosody signal is then projected to a compact latent representation $\dot { w _ { p } } \in \mathbb { R } ^ { ( T _ { \mathrm { p r e v } } + T _ { \mathrm { c u r } } ) \times d _ { p } 1 }$ using a simple MLP and Sigmoid layer. We then integrate $w _ { p }$ into driving conditions by channel-wise concatenation $c ^ { \mathrm { b a s e } } = [ w _ { r } ; w _ { a } ; w _ { e } ; w _ { p } ] \in \mathbb { R } ^ { T _ { \mathrm { F u l l } } \times d _ { c } }$ , where t denotes a specific frame, and $d _ { c } = d _ { r } + d _ { a } + d _ { e } + d _ { p }$ The $c ^ { \mathrm { b a s e } }$ is then projected from $d _ { c }$ to $d _ { h } { } ^ { 1 }$ as the final driving conditions c.

## 4.3 Temporal Reaction Loss

A key limitation of purely reconstruction-based listener generation is that the model may produce visually plausible motions while failing to align with conversationally meaningful reactions. To explicitly encourage temporally accurate listener behaviors, we introduce a temporal reaction loss based on frame-wise reaction annotations.

We consider six listener reaction categories according to our defintions in Sec. 3.2. As shown in Fig. 2.C, let $\boldsymbol { x } _ { \mathrm { c u r } } ^ { \prime } \in \mathbb { R } ^ { T _ { \mathrm { c u r } } \times d _ { u } }$ denote the transformer hidden latents for current frames, we first map each hidden token to an intermediate representation $z = \mathrm { M L P } ( x _ { \mathrm { c u r } } ^ { \prime } ) \in \mathbb { R } ^ { T _ { \mathrm { c u r } } \times 1 9 2 }$ . We then factorize z into six class-specific subspaces, each with 32 channels, where $z _ { t }  \{ z _ { t } ^ { ( k ) } \} _ { k = 1 } ^ { 6 } , z _ { t } ^ { ( k ) } \in$ $\mathbb { R } ^ { 3 2 } , t = 1 , \dots , T _ { \mathrm { c u r } }$ . Each reaction subspace is reduced to a time-specific latent $\hat { r } \in [ 0 , \ddot { 1 } ] ^ { T _ { \mathrm { c u r } } \times 6 }$ with a class-specific linear layer, followed by a sigmoid activation. With the annotated frame-wise reaction target $r \in [ 0 , 1 ] ^ { T _ { \mathrm { c u r } } \times 6 }$ , we apply Smooth-L1 supervision:

$$
\mathcal { L } _ { \mathrm { r e a c t } } = \frac { 1 } { T _ { \mathrm { c u r } } } \sum _ { t = 1 } ^ { T _ { \mathrm { c u r } } } \mathrm { S m o o t h L } 1 ( \hat { r } _ { t } , r _ { t } ) .\tag{1}
$$

This loss encourages the hidden motion representation to preserve reaction-aware temporal structure, improving not only the visual fidelity of listener motion but also the timing and type consistency of generated reactions.

## 4.4 Training Objective and Inference

The overall training objective is

$$
\mathcal { L } = \mathcal { L } _ { \mathrm { f m } } + \lambda _ { \mathrm { v e l } } \mathcal { L } _ { \mathrm { v e l } } + \lambda _ { \mathrm { r e a c t } } \mathcal { L } _ { \mathrm { r e a c t } } .\tag{2}
$$

Here, ${ \mathcal { L } } _ { \mathrm { f m } }$ is the flow-matching loss on the predicted vector field, $\mathcal { L } _ { \mathrm { v e l } }$ is the temporal velocity consistency regularizer [17], and ${ \mathcal { L } } _ { \mathrm { r e a c t } }$ is the proposed temporal reaction loss. At inference time, the model takes the listener reference image, previous listener motion context, and speaker audio as input. The FMT predicts conditional vector fields, which are integrated with an ODE solver to obtain current listener motion latents. These motion latents are then decoded into video frames.

## 5 Reaction-Centric Evaluation Metrics

Conventional evaluation metrics for talking-head or listening-head generation mainly focus on global visual fidelity, perceptual similarity, motion diversity, or distributional realism, which are insufficient for evaluating listener-specific behaviors. In particular, they fail to capture whether appropriate reactions are generated at the right moments, whether their temporal extents are accurate, and whether the visual quality of these reactions is satisfactory. To address these limitations, we propose a set of reaction-centric evaluation metrics that explicitly measure the reaction type correctness, temporal alignment, and visual quality of generated listener reactions.

For each generated listener video, we apply the reaction detector described in Sec. 3.2 to obtain a set of predicted reaction events $\widehat { \mathcal { A } } = \{ ( \widehat { y } _ { i } , \widehat { t } _ { s , i } , \widehat { t } _ { e , i } ) \} _ { i = 1 } ^ { \widehat { N } }$ , where $\widehat { y } _ { i }$ is the predicted reaction type and $\widehat { t } _ { s , i } , \widehat { t } _ { e , i }$ are the predicted start and end timestamps. The ground-truth reaction annotations are denoted as $\mathcal { A } = \{ ( y _ { j } , t _ { s , i } , t _ { e , i } ) \} _ { j = 1 } ^ { N } . \mathrm { A }$ predicted event $\widehat { a } _ { i }$ matches a ground-truth event $a _ { j }$ only if they have the same reaction type and their temporal Intersection-over-Union (tIoU) exceeds a threshold $\tau ( \mathrm { i . e . }$ ${ \widehat { y } } _ { i } = y _ { j }$ , and $\mathrm { t I o U } ( \bar { \widehat { a } } _ { i } , a _ { j } ) \geq \tau$ . We set $\tau = 0 . 5$ in all experiments). The tIoU is defined as

$$
\mathrm { t I o U } ( \widehat { a } _ { i } , a _ { j } ) = \frac { \left[ \operatorname* { m i n } ( \widehat { t } _ { e , i } , t _ { e , j } ) - \operatorname* { m a x } ( \widehat { t } _ { s , i } , t _ { s , j } ) \right] _ { + } } { \operatorname* { m a x } ( \widehat { t } _ { e , i } , t _ { e , j } ) - \operatorname* { m i n } ( \widehat { t } _ { s , i } , t _ { s , j } ) } .\tag{3}
$$

Matching is performed in a one-to-one manner by selecting the prediction with the highest temporal IoU for each ground-truth event, and the resulting matched set is denoted as $\mathcal { M }$

Reaction F1 (R-F1) score. R-F1 evaluates the occurance and matching quality of generated reactions across six categories. Matched pairs are treated as true positives, unmatched predictions as false positives, and unmatched ground-truth events as false negatives. We compute

$$
\operatorname { R - F } 1 = { \frac { 2 \cdot { \mathrm { P r e c i s i o n } } \cdot { \mathrm { R e c a l l } } } { \mathrm { P r e c i s i o n } + \mathrm { R e c a l l } } } ,\tag{4}
$$

where precision and recall are computed from event-level matches. A higher R-F1 indicates better reaction occurrence and type consistency.

Reaction temporal IoU (R-tIoU). R-tIoU measures the temporal localization accuracy of matched reaction events. It is computed as the average temporal IoU over all matched pairs:

$$
\operatorname { R - t I o U } = \frac { 1 } { | \mathcal { M } | } \sum _ { ( \widehat { a } , a ) \in \mathcal { M } } \operatorname { t I o U } ( \widehat { a } , a ) .\tag{5}
$$

A higher R-tIoU indicates that the generated reactions better match the ground-truth reaction intervals in both onset and offset.

Reaction asymmetric temporal deviation (R-ATD). Temporal overlap alone does not distinguish different types of timing errors. In dyadic interaction, prematurely generated listener reactions are often more disruptive than slightly delayed reactions, since early feedback may appear to anticipate the speaker before the relevant cue occurs. We therefore define Reaction Asymmetric Temporal Deviation, denoted as R-ATD. For a matched pair $( \widehat { a } , a )$ , we compute normalized deviations in start time, end time, and duration:

$$
\delta _ { s } = \frac { \widehat { t } _ { s } - t _ { s } } { \Delta t } , \qquad \delta _ { e } = \frac { \widehat { t } _ { e } - t _ { e } } { \Delta t } , \qquad \delta _ { \Delta t } = \frac { \widehat { \Delta t } - \Delta t } { \Delta t } ,\tag{6}
$$

where $\Delta t = t _ { e } - t _ { s }$ and $\widehat { \Delta t } = \widehat { t } _ { e } - \widehat { t } _ { s }$ . We then apply an asymmetric penalty function $\phi$ and calculate R-ATD as follows

$$
\mathrm { R - A T D } = \frac { 1 } { | \mathcal { M } | } \sum _ { ( \hat { \alpha } , a ) \in \mathcal { M } } \left[ \phi ( \delta _ { s } ; \alpha , \beta ) + \phi ( \delta _ { e } ; \alpha , \beta ) + \phi ( \delta _ { d } ; \alpha , \beta ) \right] \mathrm { , ~ w h e r e ~ } \phi ( \delta ; \alpha , \beta ) = \left\{ \begin{array} { l l } { \alpha | \delta | , } & { \delta < 0 } \\ { \beta | \delta | , } & { \delta \ge 0 } \end{array} \right. ,\tag{7}
$$

We use asymmetric weights with $\alpha > \beta$ to place stronger penalties on negative deviations, which correspond to premature or truncated reactions, while allowing greater tolerance for slightly delayed or temporally extended reactions. Lower R-ATD indicates better temporal alignment and fewer premature or duration-inaccurate reactions.

Reaction Fréchet distance (R-FID). R-FID evaluates the visual quality of generated reaction regions. Instead of computing FID over all video frames, which can be dominated by neutral listening frames, we compute the standard Fréchet distance only on frames inside reaction intervals. Generated reaction frames are collected from predicted reaction segments, while real reaction frames are collected from ground-truth reaction annotations. Lower R-FID indicates that the generated reaction frames better match the visual distribution of real listener reactions.

Together, R-F1, R-tIoU, R-ATD, and R-FID provide complementary measurements of reaction correctness, temporal localization, asymmetric timing deviation, and reaction-region realism. Although multiple reactions may be plausible for the same context, our evaluation only compares against the ground truth, serving as a reference rather than an absolute measure of reaction plausibility. Additional implementation details, including one-to-one matching, aggregation, empty-case handling, R-ATD penalization with α and $\beta ,$ , and R-FID computation, are provided in Appendix D.

## 6 Experiments

## 6.1 Experiment Settings

Implementation details. We implement our listening-head generator with a latent flow-matching transformer and train it with mixed precision using Accelerate on 4×L40s GPUs. We use AdamW (lr=5e-4), gradient clipping (1.0), cosine decay with warmup, batch size 256, and train for 650 epochs. The default temporal setup is 25 FPS, and audio is sampled at 16 kHz. Input frames are resized to 512×512 and normalized to [−1, 1]. We split the train/test set into a ratio 9:1. The pretrained motion autoencoder [40] and audio LLM [8] are kept frozen. More details can be found in Appendix C.1.

Evaluation metrics. We use Fréchet Inception Distance (FID) [34] and 16-frame Fréchet Video Distance (FVD) [39] to assess image and video generation quality; Peak Signal-to-Noise Ratio (PSNR) for measuring pixel fidelity; Structural Similarity Index (SSIM) for perceived change in structural information; Variation (Var) [25] for motion diversity and richness; Learned Perceptual Image Patch Similarity (LPIPS) [42] for measuring perceptual distance; Residual Pearson Correlation Coefficient (rPCC) to measure the correlation between motions of speaker and listener; DI-Sync [27] to quantify the causal coordination between the speaker’s verbal cues and the listener’s non-verbal responses. We also use metrics proposed in Sec. 5 (i.e., R-F1, R-tIoU, R-ATD, and R-FID) to specifically evaluate the accuracy and generation quality of reactions.

Comparison methods. We choose L2L [25], DIM [38], ViCo [46], ListenFormer [20], and DyStream [6] for comparison. All methods are trained and tested using our dataset, respectively.

## 6.2 Experiment Results

Quantitative comparison with SOTA methods. We compare our method with prior approaches on the RealTalk and Seamless in Table 2 (The bold are the best, while the underlined are the second best). On RealTalk, our method achieves the best overall performance across most metrics, improving both visual quality and reaction modeling. Lower FID/FVD and LPIPS indicate better perceptual realism and temporal coherence, while higher DI-Sync, R-F1, and R-tIoU, together with lower R-ATD, show more accurate and temporally aligned listener reactions. On the more challenging Seamless subset, our method maintains strong visual fidelity and consistently improves reaction-related metrics, especially R-F1, R-tIoU, and R-FID. Although DyStream obtains slightly higher Var, our method provides a better balance between motion diversity, reaction accuracy, and visual quality.

Per-class reaction evaluation comparison with DyStream. Table 3 further reports per-class reaction evaluation. The results reveal some category-dependent difficulties: large-amplitude reactions such as laughing are easier to detect and temporally align, while subtle motions such as nodding and head shaking are more sensitive to timing errors. Surprised remains the most challenging category due to its ambiguity and sparsity, whereas smiling achieves more balanced performance benefiting from its larger data scale. Overall, Seamless yields consistently lower scores than RealTalk, reflecting its greater diversity and complexity.

Table 2: Quantitative comparison of state-of-the-art methods on the RealTalk and Seamless datasets
<table><tr><td>Dataset</td><td>Method</td><td>PSNR ↑</td><td>SSIM↑</td><td>FID↓</td><td>FVD↓</td><td>Var ↑ LPIPS</td><td>rPCC↓</td><td>DI-Sync ↑</td><td>R-F1 ↑</td><td>R-tIoU ↑</td><td>R-ATD</td><td>R-FID↓</td></tr><tr><td rowspan="6"></td><td>L2L</td><td>14.684</td><td>0.575</td><td>45.782</td><td>202.798</td><td>1.885 0.637</td><td>0.323</td><td>0.182</td><td>0.454</td><td>0.505</td><td>78.265</td><td>24.683</td></tr><tr><td>DIM</td><td>16.223</td><td>0.495</td><td>37.717</td><td>188.266 2.765</td><td>0.585</td><td>0.289</td><td>0.190</td><td>0.576</td><td>0.551</td><td>133.082</td><td>22.971</td></tr><tr><td>ViCo</td><td>14.932</td><td>0.602</td><td>44.089</td><td>185.040 2.560</td><td>0.579</td><td>0.261</td><td>0.178</td><td>0.429</td><td>0.572</td><td>92.454</td><td>26.105</td></tr><tr><td>ListenFormer</td><td>17.454</td><td>0.582</td><td>36.173</td><td>165.290 1.625</td><td>0.525</td><td>0.256</td><td>0.221</td><td>0.334</td><td>0.650</td><td>73.379</td><td>23.097</td></tr><tr><td>DyStream</td><td>17.894</td><td>0.611</td><td>37.416</td><td>147.537 2.802</td><td>0.467</td><td>0.248</td><td>0.208</td><td>0.535</td><td>0.694</td><td>63.171</td><td>15.337</td></tr><tr><td>Ours</td><td>17.972</td><td>0.601</td><td>35.697</td><td>142.454 2.916</td><td>0.454</td><td>0.227</td><td>0.245</td><td>0.594</td><td>0.704</td><td>57.388</td><td>15.192</td></tr><tr><td rowspan="6">Seamless</td><td>L2L</td><td>11.634</td><td>0.456</td><td>36.287 245.885</td><td></td><td>1.481 0.542</td><td>0.372</td><td>0.192</td><td>0.409</td><td>0.554</td><td>112.938</td><td>35.697</td></tr><tr><td>DIM</td><td>12.856</td><td>0.391</td><td>29.885</td><td>289.695</td><td>2.180 0.411</td><td>0.398</td><td>0.172</td><td>0.296</td><td>0.501</td><td>157.302</td><td>34.379</td></tr><tr><td>ViCo</td><td>12.830</td><td>0.458</td><td></td><td>34.932 230.037 2.021</td><td>0.449</td><td>0.385</td><td>0.165</td><td>0.389</td><td>0.523</td><td>140.344</td><td>37.926</td></tr><tr><td>ListenFormer</td><td>13.824</td><td>0.464</td><td>28.662</td><td>210.474</td><td>1.287 0.472</td><td>0.361</td><td>0.212</td><td>0.369</td><td>0.511</td><td>128.705</td><td>35.266</td></tr><tr><td>DyStream</td><td>13.664</td><td>0.464</td><td>31.755</td><td>209.876 2.452</td><td>0.409</td><td>0.329</td><td>0.202</td><td>0.414</td><td>0.553</td><td>110.808</td><td>30.774</td></tr><tr><td>Ours</td><td>14.245</td><td>0.473</td><td>27.288</td><td>193.281 2.447</td><td>0.371</td><td>0.301</td><td>0.205</td><td>0.460</td><td>0.582</td><td>102.113</td><td>28.857</td></tr></table>

Table 3: Per-class reaction comparison between Dystream and our method with proposed metrics
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Reaction</td><td rowspan="2">Data Amount</td><td colspan="2">R-F1 ↑</td><td colspan="2">R-tIoU ↑</td><td colspan="2">R-ATD ↓</td><td colspan="2">R-FID↓</td></tr><tr><td>Dystream</td><td>Ours</td><td>Dystream</td><td>Ours</td><td>Dystream</td><td>Ours</td><td>Dystream</td><td>Ours</td></tr><tr><td rowspan="6">RealTalk</td><td>nodding</td><td>3033</td><td>0.562</td><td>0.602</td><td>0.672</td><td>0.686</td><td>66.168</td><td>60.375</td><td>15.114</td><td>14.680</td></tr><tr><td>head shaking</td><td>3588</td><td>0.521</td><td>0.581</td><td>0.655</td><td>0.652</td><td>67.447</td><td>61.894</td><td>15.626</td><td>15.316</td></tr><tr><td>smiling</td><td>4804</td><td>0.554</td><td>0.633</td><td>0.718</td><td>0.753</td><td>60.396</td><td>54.882</td><td>14.883</td><td>14.371</td></tr><tr><td>laughing</td><td>2397</td><td>0.569</td><td>0.656</td><td>0.735</td><td>0.769</td><td>57.026</td><td>48.189</td><td>16.005</td><td>15.697</td></tr><tr><td>frowning</td><td>2745</td><td>0.511</td><td>0.549</td><td>0.705</td><td>0.707</td><td>61.518</td><td>54.793</td><td>15.010</td><td>15.024</td></tr><tr><td>surprised</td><td>1860</td><td>0.503</td><td>0.540</td><td>0.679</td><td>0.658</td><td>66.515</td><td>64.233</td><td>15.397</td><td>16.007</td></tr><tr><td rowspan="6">Seamless</td><td>nodding</td><td>7197</td><td>0.426</td><td>0.476</td><td>0.535</td><td>0.548</td><td>114.168</td><td>110.223</td><td>30.294</td><td>27.908</td></tr><tr><td>head shaking</td><td>9202</td><td>0.405</td><td>0.450</td><td>0.515</td><td>0.522</td><td>116.043</td><td>110.930</td><td>31.054</td><td>28.445</td></tr><tr><td>smiling</td><td>10963</td><td>0.438</td><td>0.501</td><td>0.582</td><td>0.617</td><td>104.982</td><td>96.309</td><td>30.038</td><td>28.638</td></tr><tr><td>laughing</td><td>5923</td><td>0.455</td><td>0.515</td><td>0.595</td><td>0.636</td><td>100.529</td><td>91.165</td><td>31.645</td><td>29.887</td></tr><tr><td>frowning</td><td>7241</td><td>0.392</td><td>0.424</td><td>0.564</td><td>0.601</td><td>107.388</td><td>98.237</td><td>29.885</td><td>27.424</td></tr><tr><td>surprised</td><td>5604</td><td>0.373</td><td>0.409</td><td>0.528</td><td>0.575</td><td>121.848</td><td>106.038</td><td>31.744</td><td>30.856</td></tr></table>

Qualitative comparisons on RealTalk. Fig. 3 shows two representative qualitative results on RealTalk, where we uniformly sample the same set of frames for each method to compare. Existing methods often generate static listeners (L2L, DIM, ViCo) or reactions with inaccurate timing (ListenFormer, DyStream). In contrast, our method produces more expressive and contextually consistent reactions that better match the ground truth in both type and timing, demonstrating improved modeling of listener reactions. Qualitative results of Seamless are provided in Appendix E.1.

Human evaluation and reaction appropriateness. We conduct a human evaluation to assess the naturalness, contextual appropriateness, and timing plausibility of generated reactions. We sample 100 generated videos from the RealTalk test set (15.94s on average), containing 165 reaction events. We recruit 23 participants (4 undergraduate, 13 master’s, and 6 PhD students; mean age 23; 18 male and 5 female). Given the speaker audio and transcript used as model context, participants evaluate each detected reaction event, including its type and start/end times, using three binary criteria: whether the motion appears natural, the reaction is appropriate to the speaker context, and it occurs at plausible time. This yields 165 × 23 = 3,795 reaction-participant annotations. We compute the Human Agreement Rate (HAR) for each criterion as the proportion of reactions judged positively (e.g., HAR<sub>naturalness</sub> = #natural judgments / #all judgments), where higher HAR indicates better alignment with human judgment. The 165 events comprise 41 nodding, 14 head-shaking, 29 smiling, 35 laughing, 26 frowning, and 20 surprised reactions. Table 4 reports participant-averaged HAR for each reaction type, with the overall score computed as the average across the six categories.

As shown in Table 4, our method achieves high HARs of 89.9%, 89.4%, and 93.5% for reaction naturalness, contextual appropriateness, and timing plausibility, respectively. The strong timing score supports the effectiveness of our temporal modeling. More expressive reactions, such as laughing and surprised, remain more challenging due to complex facial dynamics, while the relatively lower contextual appropriateness of laughter highlights its dependence on subtle semantic and social cues, including humor, emotion, and interpersonal interaction.

## 6.3 Ablation Study

We conduct our ablation studies on RealTalk dataset. More ablation studies, including speaker audio length, prosody channel projection, and reaction loss type, are included in appendix E.2

Module effectiveness. We ablate the effectiveness of the prosody conditioning mechanism and temporal reaction loss in Table 5a. It can be seen that, adding the prosody condition improves visual quality and motion dynamics (e.g., PSNR, FVD, Var), but brings limited gains on reactionrelated metrics and even degrades temporal precision (R-ATD), indicating that prosodic cues alone are insufficient for accurate reaction modeling. In contrast, the reaction loss yields substantial improvements across various metrics (DI-Sync, R-F1, R-tIoU) and significantly reduces R-ATD and R-FID, demonstrating its key role in learning temporally aligned and semantically correct reactions. Combining both modules achieves the best overall performance, showing that prosody provides complementary temporal cues on top of explicit reaction supervision.

![](images/e71f08f3fc609d1292944da3d33eea4411f4232e0b9334855d65f9c6609f71a3.jpg)  
Figure 3: Qualitative comparison of our approach with state-of-the-art methods on the RealTalk dataset. Reaction labels are displayed at frames where reactions are detected. Our method generates more natural listening motions and reactions that are more temporally aligned with the ground truths than other methods.

Table 4: Human evaluation results across different reaction categories.
<table><tr><td>Reaction</td><td>Reaction Naturalness</td><td>Contextual Appropriateness</td><td>Timing Plausibility</td></tr><tr><td>nodding</td><td>0.954</td><td>0.975</td><td>0.982</td></tr><tr><td>head shaking</td><td>0.969</td><td>0.914</td><td>0.937</td></tr><tr><td>smiling</td><td>0.897</td><td>0.868</td><td>0.944</td></tr><tr><td>laughing</td><td>0.852</td><td>0.836</td><td>0.955</td></tr><tr><td>frowning</td><td>0.902</td><td>0.874</td><td>0.884</td></tr><tr><td>surprised</td><td>0.824</td><td>0.898</td><td>0.905</td></tr><tr><td>Overall</td><td>0.899</td><td>0.894</td><td>0.935</td></tr></table>

Reaction loss coefficient $\lambda _ { \mathrm { { r e a c t } } }$ . Table 5b shows that increasing the reaction loss coefficient $\lambda _ { \mathrm { { r e a c t } } }$ from a small value (0.01) to a moderate range (0.05–0.1) consistently improves both visual quality and reaction-related metrics, indicating that stronger supervision helps the model learn more accurate and temporally aligned behaviors. However, further increasing $\lambda _ { \mathrm { { r e a c t } } } \left( \mathbf { e } . \mathbf { g } . , \geq 0 . 2 \right)$ leads to degraded performance in both generation quality and reaction accuracy, suggesting that overly strong supervision disrupts the balance between the two objectives. Overall, $\lambda _ { \mathrm { { r e a c t } } } { = } 0 . 0 5$ achieves the best trade-off across metrics.

Table 5: Ablation of the proposed components and sensitivity to the reaction loss coefficient (a) Ablation study of the prosody conditioning and the temporal reaction loss
<table><tr><td colspan="2">Modules</td><td colspan="10">Metrics</td></tr><tr><td>Prosody Cond.</td><td>Reaction Loss</td><td>PSNR ↑</td><td>SSIM ↑</td><td>FID↓</td><td>FVD↓</td><td>Var ↑</td><td>LPIPS ↓ rPCC ↓</td><td></td><td>DI-Sync ↑</td><td>R-F1 ↑</td><td>R-tIoU ↑</td><td>R-ATD ↓ R-FID↓</td></tr><tr><td></td><td></td><td>13.113</td><td>0.534</td><td>43.286</td><td>188.044</td><td>1.862 0.749</td><td>0.283</td><td>0.178</td><td>0.404</td><td>0.599</td><td>89.984</td><td>23.706</td></tr><tr><td>√</td><td></td><td>14.071</td><td>0.558</td><td>40.798</td><td>159.745</td><td>2.206 0.659</td><td>0.274</td><td>0.194</td><td>0.481</td><td>0.574</td><td>100.342</td><td>20.603</td></tr><tr><td></td><td>√</td><td>17.212</td><td>0.586</td><td>36.635</td><td>145.858</td><td>2.896 0.477</td><td>0.252</td><td>0.236</td><td>0.579</td><td>0.695</td><td>55.657</td><td>17.885</td></tr><tr><td>√</td><td>√</td><td>17.973</td><td>0.601</td><td>35.691</td><td>142.456</td><td>2.913 0.454</td><td>0.227</td><td>0.245</td><td>0.594</td><td>0.704</td><td>57.386</td><td>15.192</td></tr></table>

(b) Effect of varying the temporal reaction loss coefficient $\lambda _ { \mathrm { { r e a c t } } }$
<table><tr><td rowspan="2">Reaction Loss Coef.</td><td colspan="10">Metrics</td></tr><tr><td>PSNR ↑</td><td>SSIM ↑ FID↓</td><td>FVD↓</td><td>Var ↑</td><td>LPIPS↓</td><td>rPCC ↓</td><td>DI-Sync ↑</td><td>R-F1 ↑</td><td>R-tIoU ↑</td><td>R-ATD↓</td><td>R-FID↓</td></tr><tr><td>0.01</td><td>15.446 0.573</td><td>37.034</td><td>178.878</td><td>1.848</td><td>0.502</td><td>0.232</td><td>0.196</td><td>0.572</td><td>0.606</td><td>78.672</td><td>17.976</td></tr><tr><td>0.05</td><td>17.973</td><td>0.601 35.691</td><td>142.456</td><td>2.913</td><td>0.454</td><td>0.227</td><td>0.245</td><td>0.594</td><td>0.704</td><td>57.386</td><td>15.192</td></tr><tr><td>0.1</td><td>17.904</td><td>0.632 36.147</td><td>144.895</td><td>2.901</td><td>0.447</td><td>0.230</td><td>0.223</td><td>0.613</td><td>0.711</td><td>64.790</td><td>16.326</td></tr><tr><td>0.2</td><td>17.566</td><td>0.610 36.032</td><td>149.286</td><td>2.744</td><td>0.471</td><td>0.245</td><td>0.230</td><td>0.602</td><td>0.684</td><td>69.131</td><td>15.875</td></tr><tr><td>0.5</td><td>15.297</td><td>0.595 39.115</td><td>163.477</td><td>2.331</td><td>0.502</td><td>0.248</td><td>0.205</td><td>0.582</td><td>0.695</td><td>77.437</td><td>15.548</td></tr><tr><td>1</td><td>15.132</td><td>0.560 42.855</td><td>169.964</td><td>1.948</td><td>0.545</td><td>0.261</td><td>0.210</td><td>0.549</td><td>0.663</td><td>78.551</td><td>17.036</td></tr></table>

Table 6: Effect of prosody conditioning on different reaction categories.
<table><tr><td rowspan=1 colspan=1>Reactions</td><td rowspan=1 colspan=1>Prosody Cond.</td><td rowspan=1 colspan=1>R-F1↑</td><td rowspan=1 colspan=1>R-tIoU ↑</td><td rowspan=1 colspan=1>R-ATD ↓</td><td rowspan=1 colspan=1>R-FID ↓</td></tr><tr><td rowspan=1 colspan=1>nodding</td><td rowspan=1 colspan=1>X√</td><td rowspan=1 colspan=1>0.4220.474 (+12.32%)</td><td rowspan=1 colspan=1>0.6260.583</td><td rowspan=1 colspan=1>84.73297.428 (+14.98%)</td><td rowspan=1 colspan=1>23.28421.032</td></tr><tr><td rowspan=1 colspan=1>head shaking</td><td rowspan=1 colspan=1>XV</td><td rowspan=1 colspan=1>0.3980.441 (+10.08%)</td><td rowspan=1 colspan=1>0.5840.539</td><td rowspan=1 colspan=1>91.645103.763 (+13.22%)</td><td rowspan=1 colspan=1>24.22621.445</td></tr><tr><td rowspan=1 colspan=1>smiling</td><td rowspan=1 colspan=1>X√</td><td rowspan=1 colspan=1>0.4210.517 (+22.80%)</td><td rowspan=1 colspan=1>0.6400.595</td><td rowspan=1 colspan=1>82.51993.314 (+13.08%)</td><td rowspan=1 colspan=1>22.57119.164</td></tr><tr><td rowspan=1 colspan=1>laughing</td><td rowspan=1 colspan=1>XV</td><td rowspan=1 colspan=1>0.4470.578 (+29.31%)</td><td rowspan=1 colspan=1>0.6490.671 (+3.39%)</td><td rowspan=1 colspan=1>79.88489.726 (+12.32%)</td><td rowspan=1 colspan=1>22.06318.437</td></tr><tr><td rowspan=1 colspan=1>frowning</td><td rowspan=1 colspan=1>XV</td><td rowspan=1 colspan=1>0.3620.426 (+17.68%)</td><td rowspan=1 colspan=1>0.5600.511</td><td rowspan=1 colspan=1>96.438108.582 (+12.59%)</td><td rowspan=1 colspan=1>25.48221.683</td></tr><tr><td rowspan=1 colspan=1>surprised</td><td rowspan=1 colspan=1>X√</td><td rowspan=1 colspan=1>0.3810.448 (+17.59%)</td><td rowspan=1 colspan=1>0.5350.549 (+2.62%)</td><td rowspan=1 colspan=1>104.686109.219 (+4.33%)</td><td rowspan=1 colspan=1>24.61021.857</td></tr></table>

Effectiveness of prosody conditioning. We further analyze the effect of prosody conditioning at the reaction level in Table 6, reporting percentage changes for clarity. Prosody conditioning consistently improves R-F1 across all reaction types, with the largest gain for laughing, suggesting that speaker prosodic changes provide useful cues for reaction expressiveness. However, it does not consistently improve temporal metrics: most reaction categories show degraded R-tIoU and R-ATD. This is likely because prosody reflects the speaker’s affective state but does not explicitly determine whether or when a listener should react; relying too strongly on prosodic changes may therefore trigger reactions too immediately, whereas real conversational responses also depend heavily on semantic and contextual cues. Our temporal reaction loss complements prosody conditioning by explicitly supervising reaction occurrence and timing.

## 7 Conclusion

In this paper, we present a reaction-centric framework for listening head generation that addresses the limitations of existing methods in modeling conversationally appropriate listener behaviors. Built upon dyadic conversational videos from the RealTalk and Seamless datasets, we curate a new dataset consisting of aligned speaker-listener pairs and corresponding fine-grained event-level reaction annotations. We then propose an audio-driven baseline with prosody conditioning and a temporal reaction loss to explicitly guide reaction-aware motion generation. We further introduce reaction-oriented evaluation metrics specifically designed to assess reaction occurrence, temporal alignment, and visual quality, complementing conventional generation metrics. Experiments on both RealTalk and Seamless show that our method improves visual fidelity, motion naturalness, and, more importantly, reaction accuracy and temporal alignment over existing approaches. These results suggest that explicit reaction supervision and reaction-specific evaluation are important steps toward more natural and context-aware listening head generation.

## References

[1] Vasu Agrawal, Akinniyi Akinyemi, Kathryn Alvero, Morteza Behrooz, Julia Buffalini, Fabio Maria Carlucci, Joy Chen, Junming Chen, Zhang Chen, Shiyang Cheng, et al. Seamless interaction: Dyadic audiovisual motion modeling and large-scale dataset. arXiv preprint arXiv:2506.22554, 2025.

[2] Alexei Baevski, Yuhao Zhou, Abdelrahman Mohamed, and Michael Auli. wav2vec 2.0: A framework for self-supervised learning of speech representations. Advances in neural information processing systems, 33: 12449–12460, 2020.

[3] Hervé Bredin. pyannote.audio 2.1 speaker diarization pipeline: principle, benchmark, and recipe. In Proc. INTERSPEECH 2023, 2023.

[4] Adrian Bulat and Georgios Tzimiropoulos. How far are we from solving the 2d & 3d face alignment problem? (and a dataset of 230,000 3d facial landmarks). In International Conference on Computer Vision, 2017.

[5] Angelo Cafaro, Johannes Wagner, Tobias Baur, Soumia Dermouche, Maria Torres Torres, Catherine Pelachaud, Elisabeth André, and Michel Valstar. The noxi database: Multimodal recordings of mediated novice-expert interactions. In Proceedings of the 19th ACM International Conference on Multimodal Interaction, pages 350–359, 2017.

[6] Bohong Chen and Haiyang Liu. Dystream: Streaming dyadic talking heads generation via flow matching based autoregressive model. arXiv preprint arXiv:2512.24408, 2025.

[7] Jin Hyun Cheong, Tiankang Xie, Sophie Byrne, and Luke J. Chang. Py-feat: Python facial expression analysis toolbox. arXiv preprint arXiv:2104.03509, 2021.

[8] Yunfei Chu, Jin Xu, Qian Yang, Haojie Wei, Xipin Wei, Zhifang Guo, Yichong Leng, Yuanjun Lv, Jinzheng He, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen2-audio technical report. arXiv preprint arXiv:2407.10759, 2024.

[9] IA de Kok and Dirk KJ Heylen. A survey on evaluation metrics for backchannel prediction models. In Interdisciplinary Workshop on Feedback Behaviors in Dialog, Stevenson, Washington, USA: Proceedings of the Interdisciplinary Workshop on Feedback Behaviors in Dialog, pages 15–18. University of Texas, 2012.

[10] Iwan de Kok and Dirk K. J. Heylen. A survey on evaluation metrics for backchannel prediction models. In Proceedings ofthe Interdisciplinary Workshop on Feedback Behaviors in Dialog, pages 15–18, 2012.

[11] Scott Geng, Revant Teotia, Purva Tendulkar, Sachit Menon, and Carl Vondrick. Affective faces for goal-driven dyadic communication. arXiv preprint arXiv:2301.10939, 2023.

[12] Scott Geng, Revant Teotia, Purva Tendulkar, Sachit Menon, and Carl Vondrick. Affective faces for goal-driven dyadic communication. arXiv preprint arXiv:2301.10939, 2023.

[13] Ian J Goodfellow, Dumitru Erhan, Pierre Luc Carrier, Aaron Courville, Mehdi Mirza, Ben Hamner, Will Cukierski, Yichuan Tang, David Thaler, Dong-Hyun Lee, et al. Challenges in representation learning: A report on three machine learning contests. In International conference on neural information processing, pages 117–124. Springer, 2013.

[14] Philip-William Grassal, Malte Prinzler, Titus Leistner, Carsten Rother, Matthias Nießner, and Justus Thies. Neural head avatars from monocular rgb videos. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pages 18653–18664, 2022.

[15] Ying Guo, Xi Liu, Cheng Zhen, Pengfei Yan, and Xiaoming Wei. Arig: Autoregressive interactive head generation for real-time conversations. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 12956–12965, 2025.

[16] Xinya Ji, Hang Zhou, Kaisiyuan Wang, Wayne Wu, Chen Change Loy, Xun Cao, and Feng Xu. Audiodriven emotional video portraits. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 14080–14089, 2021.

[17] Taekyung Ki, Dongchan Min, and Gyeongsu Chae. Float: Generative motion latent flow matching for audio-driven talking portrait. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pages 14699–14710, 2025.

[18] Borong Liang, Yan Pan, Zhizhi Guo, Hang Zhou, Zhibin Hong, Xiaoguang Han, Junyu Han, Jingtuo Liu, Errui Ding, and Jingdong Wang. Expressive talking head generation with granular audio-visual control. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pages 3387–3396, 2022.

[19] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022.

[20] Miao Liu, Jing Wang, Xinyuan Qian, and Haizhou Li. Listenformer: Responsive listening head generation with non-autoregressive transformers. In Proceedings of the 32nd ACM International Conference on Multimedia, pages 7094–7103, 2024.

[21] Camillo Lugaresi, Jiuqiang Tang, Hadon Nash, Chris McClanahan, Esha Uboweja, Michael Hays, Fan Zhang, Chuo-Ling Chang, Ming Guang Yong, Juhyun Lee, et al. Mediapipe: A framework for building perception pipelines. arXiv preprint arXiv:1906.08172, 2019.

[22] Haoyu Ma, Tong Zhang, Shanlin Sun, Xiangyi Yan, Kun Han, and Xiaohui Xie. Cvthead: One-shot controllable head avatar with vertex-feature transformer. In Proceedings ofthe IEEE/CVF Winter Conference on Applications ofComputer Vision, pages 6131–6141, 2024.

[23] Louis-Philippe Morency, Iwan de Kok, and Jonathan Gratch. Predicting listener backchannels: A probabilistic multimodal approach. In International Workshop on Intelligent Virtual Agents, pages 176–190. Springer, 2008.

[24] Michael Murray, Nick Walker, Amal Nanavati, Patricia Alves-Oliveira, Nikita Filippov, Allison Sauppe, Bilge Mutlu, and Maya Cakmak. Learning backchanneling behaviors for a social robot via data augmentation from human-human conversations. In Conference on robot learning, pages 513–525. PMLR, 2022.

[25] Evonne Ng, Hanbyul Joo, Liwen Hu, Hao Li, Trevor Darrell, Angjoo Kanazawa, and Shiry Ginosar. Learning to listen: Modeling non-deterministic dyadic facial motion. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pages 20395–20405, 2022.

[26] Evonne Ng, Sanjay Subramanian, Dan Klein, Angjoo Kanazawa, Trevor Darrell, and Shiry Ginosar. Can language models learn to listen? In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 10083–10093, 2023.

[27] Dongwei Pan, Longwei Guo, Jiazhi Guan, Luying Huang, Yiding Li, Haojie Liu, Haocheng Feng, Wei He, Kaisiyuan Wang, and Hang Zhou. Interdyad: Interactive dyadic speech-to-video generation by querying intermediate visual guidance. arXiv preprint arXiv:2603.23132, 2026.

[28] Manuel Pariente, Samuele Cornell, Joris Cosentino, Sunit Sivasankaran, Efthymios Tzinis, Jens Heitkaemper, Michel Olvera, Fabian-Robert Stöter, Mathieu Hu, Juan M. Martín-Doñas, David Ditter, Ariel Frank, Antoine Deleforge, and Emmanuel Vincent. Asteroid: the PyTorch-based audio source separation toolkit for researchers. In Proc. Interspeech, 2020.

[29] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4195–4205, 2023.

[30] Leonardo Pepino, Pablo Riera, and Luciana Ferrer. Emotion recognition from speech using wav2vec 2.0 embeddings. arXiv preprint arXiv:2104.03502, 2021.

[31] Alexis Plaquet and Hervé Bredin. Powerset multi-class cross entropy loss for neural speaker diarization. In Proc. INTERSPEECH 2023, 2023.

[32] KR Prajwal, Rudrabha Mukhopadhyay, Vinay P Namboodiri, and CV Jawahar. A lip sync expert is all you need for speech to lip generation in the wild. In Proceedings of the 28th ACM international conference on multimedia, pages 484–492, 2020.

[33] Fabien Ringeval, Andreas Sonderegger, Juergen Sauer, and Denis Lalanne. Introducing the recola multimodal corpus of remote collaborative and affective interactions. In 2013 10th IEEE International Conference and Workshops on Automatic Face and Gesture Recognition, pages 1–8. IEEE, 2013.

[34] Maximilian Seitzer. pytorch-fid: FID Score for PyTorch. https://github.com/mseitzer/ pytorch-fid, August 2020. Version 0.3.0.

[35] Luchuan Song, Guojun Yin, Zhenchao Jin, Xiaoyi Dong, and Chenliang Xu. Emotional listener portrait: Neural listener head generation with emotion. In Proceedings of the IEEE/CVF international conference on computer vision, pages 20839–20849, 2023.

[36] Sicheng Song et al. Multiple appropriate facial reaction generation in dyadic interaction settings: What, why and how? arXiv preprint arXiv:2302.06514, 2023.

[37] Siyang Song, Micol Spitale, Cheng Luo, Cristina Palmero, German Barquero, Hengde Zhu, Sergio Escalera, Michel Valstar, Tobias Baur, Fabien Ringeval, et al. React 2024: The second multiple appropriate facial reaction generation challenge. In 2024 IEEE 18th International Conference on Automatic Face and Gesture Recognition (FG), pages 1–5. IEEE, 2024.

[38] Minh Tran, Di Chang, Maksim Siniukov, and Mohammad Soleymani. Dim: Dyadic interaction modeling for social behavior generation. In European Conference on Computer Vision, pages 484–503. Springer, 2024.

[39] Thomas Unterthiner, Sjoerd Van Steenkiste, Karol Kurach, Raphael Marinier, Marcin Michalski, and Sylvain Gelly. Towards accurate generative models of video: A new metric & challenges. arXiv preprint arXiv:1812.01717, 2018.

[40] Yaohui Wang, Di Yang, Francois Bremond, and Antitza Dantcheva. Latent image animator: Learning to animate images via latent space navigation. arXiv preprint arXiv:2203.09043, 2022.

[41] Yinuo Wang, Yanbo Fan, Xuan Wang, Guo Yu, and Fei Wang. Diffusion-based realistic listening head generation via hybrid motion modeling. In Proceedings ofthe Computer Vision and Pattern Recognition Conference, pages 15885–15895, 2025.

[42] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 586–595, 2018.

[43] Wenxuan Zhang, Xiaodong Cun, Xuan Wang, Yong Zhang, Xi Shen, Yu Guo, Ying Shan, and Fei Wang. Sadtalker: Learning realistic 3d motion coefficients for stylized audio-driven single image talking face animation. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pages 8652–8661, 2023.

[44] Youliang Zhang, Zhaoyang Li, Duomin Wang, Jiahe Zhang, Deyu Zhou, Zixin Yin, Xili Dai, Gang Yu, and Xiu Li. Speakervid-5m: A large-scale high-quality dataset for audio-visual dyadic interactive human generation. arXiv preprint arXiv:2507.09862, 2025.

[45] Mohan Zhou, Yalong Bai, Wei Zhang, Ting Yao, Tiejun Zhao, and Tao Mei. Responsive listening head generation: a benchmark dataset and baseline. In European conference on computer vision, pages 124–142. Springer, 2022.

[46] Mohan Zhou, Yalong Bai, Wei Zhang, Ting Yao, Tiejun Zhao, and Tao Mei. Responsive listening head generation: a benchmark dataset and baseline. In European conference on computer vision, pages 124–142. Springer, 2022.

[47] Yongming Zhu, Longhao Zhang, Zhengkun Rong, Tianshu Hu, Shuang Liang, and Zhipeng Ge. Infp: Audio-driven interactive head generation in dyadic conversations. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10667–10677, 2025.

## Appendix

## A Dataset Construction and Annotation Details

## A.1 Raw Data Sources

Our dataset is constructed from two dyadic conversational video resources, RealTalk [11] and Seamless Interaction [1]. Both datasets provide high-quality two-person conversational videos, while Seamless Interaction offers a substantially larger collection of dyadic audiovisual recordings with high-resolution videos and cleaner audios. Since neither dataset is originally organized around listening-head generation with temporally localized listener reaction annotations, we further process the raw videos into aligned speaker-audio/listener-video pairs and annotate listener reactions at the event level.

Table 7: Summary of the raw data sources and the final curated listening-head dataset.
<table><tr><td>Source</td><td>Raw scale</td><td>Resolution</td><td>Role in our dataset</td></tr><tr><td>RealTalk</td><td>694 videos</td><td>1080P</td><td>Dyadic conversational footage for speaker-listener pairing</td></tr><tr><td>Seamless Interaction</td><td>65k+ videos</td><td>4K</td><td>Large-scale dyadic audiovisual data for diverse interactions</td></tr><tr><td>Curated dataset</td><td></td><td>107,149 pairs 512×512 crops</td><td>147 hours with 64,557 reaction instances</td></tr></table>

## A.2 Video Filtering and Face-Crop Quality Control

The raw videos contain several types of content that are unsuitable for listening head generation, including abrupt scene transitions, blackout frames, irrelevant scenes, severe occlusion, unstable viewpoints, and cases where more than two people appear in the visible region. To remove these samples efficiently, we uniformly sample frames from each video at 2.5-second intervals and perform a thumbnail-based inspection. Videos with frequent scene changes, missing faces, severe visual artifacts, or excessive participant overlap are discarded. We then conduct a second verification pass on the retained videos to further remove visually unstable or semantically irrelevant clips.

For each retained dyadic video, we separately crop the two participants into portrait-centered videos. We use Face-Alignment [4] to detect the facial region in the first frame where a valid face is observed, enlarge the detected bounding box by a scale factor of 1.5, and apply the resulting crop to the full video. The enlargement preserves contextual head motion and avoids overly tight crops that would remove meaningful listener reactions. After cropping, we re-run face detection and landmark localization on the cropped videos to identify low-quality samples. A cropped video is discarded if it contains missing face detections, partial faces, abnormal facial aspect ratios, unstable face proportions, or strong non-frontal views that prevent reliable motion extraction and reaction annotation.

## A.3 Audio Processing and Speaker–Listener Pairing

For RealTalk, the raw audio contain mixed speech from both participants. We first apply the Asteroid toolkit [28] for source separation and then perform audio denoising to improve the quality of each separated stream. The separated streams are manually checked to remove severe separation failures. We then use PyAnnote [31; 3] to estimate speech-activity boundaries for each participant. These boundaries define the time intervals in which one participant is treated as the active speaker and the other participant is treated as the listener.

For Seamless Interaction, the audio streams are generally cleaner and often already separated, with small portion of them including noises. We use the provided audio and diarization information when reliable, and apply PyAnnote-based timestamp correction when the provided boundaries are inaccurate. After obtaining participant-level speech-activity timelines, we decompose each dyadic conversation into a sequence of speaker-dominant intervals. For each interval, we pair the active speaker’s audio with the temporally synchronized cropped video of the other participant (listener). This produces paired samples of the form (a<sup>speaker</sup>, V <sup>listener</sup>), where a<sup>speaker</sup> denotes the speaker audio and V<sup>listener</sup> denotes the listener video.

## A.4 Clip-Level Filtering

After video-audio pairing, we remove clips with invalid speaker audio, extremely short effective speech duration, missing listener frames, or unreliable face crops. We also discard segments where the listener face is not visibly clear for a substantial part of the clip. The remaining samples are temporally aligned at 25 FPS for video and 16 kHz for audio. This filtering step ensures that each training sample contains a valid speaker signal and a stable listener portrait sequence, which are both required for audio-driven listening head generation.

## A.5 Data License, Privacy, and Release Policy

We use Seamless Interaction under its CC-BY-NC 4.0 license and RealTalk under the terms of its public dataset release. Both datasets contain real human conversational videos and may include identifiable faces, voices, and conversational content. We do not collect new recordings or directly recruit participants, and we rely on the consent and release procedures of the original dataset creators. To minimize privacy and copyright risks, we only release the annotation pipeline as well as the processed dataset, without redistributing raw videos, audio, image frames, cropped faces, transcripts, or any media segments. We also verified at the time of submission that the datasets remain publicly available and are not listed as deprecated by NeurIPS.

## B Reaction Detector Details

## B.1 Frame-Wise Reaction Representation

For each cropped listener video with T frames, the reaction detector outputs a dense frame-wise score map $\mathbf { r } \in [ 0 , 1 ] ^ { T \times K }$ , where $K = 6$ corresponds to nodding, head shaking, smiling, laughing, frowning, and surprised. The k-th channel $r _ { t } ^ { ( k ) }$ represents the confidence that reaction type k occurs at frame t. We convert the dense scores into event-level annotations by thresholding each channel, extracting connected temporal components, removing isolated short detections, and merging adjacent components separated by short gaps. Each retained component is stored as an event $( y _ { i } , r _ { i } , t _ { s , i } , t _ { e , i } )$ where $y _ { i }$ is the reaction class, $r _ { i }$ is the score vector, and $s _ { i } , e _ { i }$ are the start and end frames.

Formally, for reaction class k, we define that the reaction score for a frame at time t is 0 if the reaction detection confidence is lower than the corresponding threshold $\theta _ { k }$

$$
r _ { t } ^ { ( k ) } = \mathbf { 0 } \left\lceil r _ { t } ^ { ( k ) } \leq \theta _ { k } \right\rceil ,\tag{8}
$$

where $\theta _ { k } ~ = ~ 0 . 5$ is a fixed threshold for all six classes. The dense reaction scores are used as supervision for the temporal reaction loss, while the event-level annotations are used for reactioncentric evaluation.

It is noteworthy that event-level annotations are single-label. If candidate events from different reaction classes overlap in time, we keep the event with the highest averaged confidence score over the overlapping region, while discarding or trimming lower-confidence candidates. Ambiguous overlaps without a clear dominant visual cue are removed during manual verification to ensure a single event type $y _ { i }$

## B.2 Head-Motion-Based Reactions

We detect nodding and head shaking mainly from landmark-based head-motion trajectories [21]. A Face Alignment Network predicts facial landmarks for each valid frame. For nodding, we compute a vertical head-center coordinate as a weighted combination of stable facial landmarks, including the nose tip, chin, and outer eyebrow corners. We normalize this coordinate by a face-scale estimate derived from jaw width and the nose–chin distance, which reduces sensitivity to crop size and identity-specific face shape. Missing detections are forward-filled from the most recent valid estimate, and remaining short gaps are linearly interpolated.

Let $c _ { t } ^ { y }$ denote the estimated vertical head-center coordinate and s denote the face scale at frame t. $s _ { t }$ We use the normalized vertical trajectory

$$
y _ { t } = \frac { c _ { t } ^ { y } - \mathrm { m e d i a n } _ { \tau } ( c _ { \tau } ^ { y } ) } { s _ { t } + \epsilon } ,\tag{9}
$$

Table 8: Summary of the visual cues used by the reaction detector. The thresholds and duration constraints are fixed per class during annotation and are not tuned separately for different methods.
<table><tr><td>Reaction</td><td>Primary visual cues</td><td>Temporal rule</td><td>Post-processing</td></tr><tr><td>Nodding</td><td>Vertical head-center displace- ment, pitch-related dynamics, velocity and acceleration patterns</td><td>Peak/valley cycles with vertical zero-crossings and sufficient normalized amplitude</td><td>Smooth scores, me- rge overlapping in- tervals, threshold at 0.5</td></tr><tr><td>Head shake</td><td>Horizontal head-center displace- ment, yaw-related dynamics, hor- izontal velocity and acceleration patterns</td><td>Alternating left-right extrema with horizontal zero-crossings and sufficient normalized am- plitude</td><td>Smooth scores and merge adjacent in- tervals</td></tr><tr><td>Smiling</td><td>Happy-expression confidence, smile-related action units, lip- corner movement</td><td>Sustained high smile confi- dence over consecutive frames</td><td>Remove isolated p- eaks and merge sh- ort gaps</td></tr><tr><td>Laughing</td><td>Smile confidence, mouth opening, high-intensity happy-expression cues, stronger facial dynamics</td><td>Sustained expression with lar- ger mouth/facial motion than ordinary smiling</td><td>Merge nearby high- conf-idence inter- vals</td></tr><tr><td>Frowning</td><td>Negative-expression confidence, Brow-lowering cues (AU04), mouth-corner depression cues</td><td>Sustained negative facial ex- pression over a short temporal window</td><td>Remove brief neu- tral fluctuations</td></tr><tr><td>Surprised</td><td>Surprise-expression confidence, eyebrow raising, eye opening, mouth opening</td><td>Short high-confidence peaks or short sustained surprise inter- vals</td><td>Allow shorter eve- nts than other ex- pression classes</td></tr></table>

where ϵ is a small constant for numerical stability. The trajectory is smoothed with a Savitzky– Golay filter using an approximately 0.1-second temporal window. We then compute velocity and acceleration by discrete differentiation scaled by the video frame rate. Candidate nodding segments are generated from peak/valley patterns and velocity zero-crossings. A segment is retained only if it contains both upward and downward extrema, lies within a valid duration range, and exceeds a normalized amplitude floor.

Each retained nod candidate receives a strength score based on its normalized amplitude, velocity range, acceleration range, temporal smoothness, and closeness to a nominal nodding period of approximately 0.5 seconds. Overlapping candidates are merged, and each final interval is converted into a dense frame-wise confidence using a center-peaked temporal weighting. Specifically, or a detected interval $[ t _ { s , i } , t _ { e , i } ]$ with score q , we assign

$$
r _ { t } ^ { ( \mathrm { n o d } ) } = \operatorname* { m a x } _ { i : t \in [ t _ { s , i } , t _ { e , i } ] } q _ { i } \cdot g _ { i } ( t ) ,\tag{10}
$$

where $g _ { i } ( t )$ is a normalized center-peaked window over the interval. This produces both sparse event boundaries and dense reaction intensities.

Head shaking is detected using an analogous procedure, but the primary motion signal is the normalized horizontal head-center trajectory together with yaw-related dynamics. Candidate head-shaking intervals are identified by alternating left-right extrema, horizontal velocity zero-crossings, and sufficient normalized horizontal amplitude. The same smoothing, duration filtering, candidate scoring, and event merging strategy is applied. More technical details can be referred to [21].

## B.3 Expression-Based Reactions

Smiling, laughing, frowning, and surprised reactions are detected from facial expression cues. We combine landmark dynamics, facial action-unit responses, and emotion predictions from facial expression analysis tools [7; 13]. The detector produces per-frame class confidence scores and then applies temporal smoothing and connected-component extraction to obtain event-level annotations. Table 8 summarizes the main visual cues used for each reaction class.

The expression-based scores are temporally smoothed before thresholding. This reduces false positives caused by single-frame expression estimation noise while preserving reaction onsets and offsets. Although multiple reaction cues may be activated within the same temporal region, the final event-level annotations are single-label and temporally non-overlapping across reaction classes. We first extract class-specific candidate intervals and then resolve cross-class overlaps by retaining the candidate with the highest averaged confidence score and the clearest visual evidence. Lowerconfidence overlapping candidates are suppressed, and ambiguous cases are removed during manual verification. This design avoids assigning multiple event types to the same listener behavior.

## B.4 Manual Verification

After automatic detection, we manually inspect the detected reaction events to improve annotation reliability. During verification, we remove events caused by tracking failure, scene change, face occlusion, unstable crops, non-listening behavior, or expression ambiguity. We also discard events whose visual evidence is too weak to support the assigned reaction label. This verification step is applied after the detector has produced candidate intervals, so the final annotations retain the temporal consistency of the automatic detector while reducing obvious false positives.

## C Model and Training Details

## C.1 Architecture and Optimization

We build our baseline upon the FLOAT [17] architecture and adapt its input formulation to the listening-head generation setting. In our preliminary experiments, directly applying FLOAT resulted in overly static listener motion with minimal motion. We hypothesize that this is because speaker audio provides only weak and indirect cues for listener head motion, unlike standard audio-driven talking-head generation where speech and facial motion are tightly synchronized. To mitigate this issue, we introduce a reference clip $T _ { \mathrm { r e f } }$ as an additional context input, together with the previous motion frames $T _ { \mathrm { p r e v } }$ . This modified FLOAT serves as our baseline, which corresponds to the first row of Table ${ 5 } \mathrm { a } .$ . During training, we sample a video clip of length $T _ { \mathrm { t o t a l } }$ and divide it into $[ T _ { \mathrm { p r e v } } { + } T _ { \mathrm { c u r } } \mid T _ { \mathrm { r e f } } ]$ We then feed the segments in the order $[ T _ { \mathrm { r e f } } \mid T _ { \mathrm { p r e v } } + T _ { \mathrm { c u r } } ]$ . This allows the model to use a reference segment as an additional context, while being consistent with the inference setting where the reference clip may be non-contiguous with the generated sequence. The loss is computed only on the generated $T _ { \mathrm { c u r } }$ frames. For a fair and simple comparison, during inference, we use one fixed video clip as $T _ { \mathrm { r e f } }$ across all experiments.

We report the implementation details in Table 9. The LIA motion autoencoder [40] is initialized from pretrained checkpoints and kept frozen. During training, we optimize the conditional flow-matching transformer, audio projection layers, prosody projection branch, and auxiliary reaction head. All input video frames are resized to 512×512 and normalized to [-1,1]. The video frame rate is 25 FPS and the audio sampling rate is 16 kHz. We use a 9:1 train/test split and report results separately on RealTalk and Seamless. We train our baseline with 650 epochs using 4 Nvidia L40s GPUs.

## C.2 Prosody Extraction and Frame Alignment

We extract speaker-side prosodic fluctuation cues using Qwen2-Audio-7B-Instruct [8]. For each speaker audio segment, we prompt the audio LLM to return temporal intervals and confidence scores indicating prosodic or affective fluctuation. The prompt template is:

You are an expert in speech prosody analysis.   
You will be given an audio segment containing one speaker. Detect only the time   
intervals where the speaker shows a clear and localized prosodic fluctuation. A   
prosodic fluctuation means a noticeable change in vocal delivery, such as increased   
pitch, increased loudness, stronger stress, sharper emphasis, faster or slower   
speaking rate, unusual rhythm, hesitation, excitement, surprise, or other affective   
vocal variation. Focus only on acoustic and prosodic cues, not on the semantic   
meaning of the spoken words.   
For each detected prosodic event, output its start time, end time, and intensity score.   
The intensity score must be between 0 and 1 and should represent the salience of the   
prosodic change:   
0.50-0.60: weak but noticeable;   
0.60-0.80: clear;   
0.80-1.00: strong or highly salient.

Table 9: Implementation details of our final model.
<table><tr><td>Item</td><td>Setting</td></tr><tr><td>Training precision</td><td>Mixed precision</td></tr><tr><td>Distributed training</td><td>Accelerate</td></tr><tr><td>GPUs</td><td>4×L40s</td></tr><tr><td>Optimizer</td><td>AdamW</td></tr><tr><td>Learning rate</td><td> $5 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Learning-rate schedule</td><td>Cosine decay with warmup</td></tr><tr><td>Gradient clipping</td><td>1.0</td></tr><tr><td>Batch size</td><td>256</td></tr><tr><td>Training epochs</td><td>650</td></tr><tr><td>Video frame rate</td><td>25 FPS</td></tr><tr><td>Audio sampling rate</td><td>16 kHz</td></tr><tr><td>Image resolution</td><td>512 × 512</td></tr><tr><td>Image normalization</td><td>[−1,1]</td></tr><tr><td>Train/test split</td><td>9:1</td></tr><tr><td>Motion autoencoder</td><td>Pretrained LIA, frozen</td></tr><tr><td>FMT attention heads</td><td>8</td></tr><tr><td>Attention window length</td><td>2</td></tr><tr><td> $d _ { r }$ </td><td>512</td></tr><tr><td></td><td>512</td></tr><tr><td> $d _ { a }$   $d _ { e }$ </td><td>7</td></tr><tr><td> $d _ { h }$ </td><td></td></tr><tr><td> $d _ { w }$ </td><td>1024 512</td></tr><tr><td> $d _ { p }$ </td><td></td></tr><tr><td>Prosody projection dimension</td><td></td></tr><tr><td>Reaction loss coefficient  $\lambda _ { \mathrm { { r e a c t } } }$ </td><td>4 0.05</td></tr><tr><td> $\lambda _ { \mathrm { v e l } }$ </td><td></td></tr><tr><td>Velocity loss coefficient</td><td>1.0</td></tr><tr><td>ODE solver</td><td>Euler</td></tr><tr><td>Number of function evaluations</td><td>10</td></tr></table>

Detection rules:   
- Only output events with intensity score >= 0.50.   
- Do not output uncertain or ambiguous events.   
- Do not split one continuous prosodic fluctuation into many short intervals.   
- Merge neighboring intervals if they are part of the same prosodic event.   
- No interval overlapping.   
- Prefer concise intervals that cover the main fluctuation rather than long segments   
with neutral speech.   
Output format:   
- Each line must be: start\_time, end\_time, intensity\_score   
- All numbers must use exactly two decimal places.   
- Use seconds as the time unit.   
- Do not output any explanation, label, markdown, bullet point, or extra text.   
- If no valid event is detected, output exactly:   
NULL   
Example:   
2.34, 3.29, 0.78   
5.69, 7.29, 0.92   
Analyze the audio now and output only the formatted results.

The output is parsed into a set of triplets $\{ ( t _ { s , j } , t _ { e , j } , c _ { j } ) \} _ { j = 1 } ^ { M }$ , where $t _ { s , j }$ and $t _ { e , j }$ are start and end times in seconds and $c _ { j } \in [ 0 , 1 ]$ is the confidence score. We convert these sparse intervals into a frame-aligned scalar sequence $p \stackrel { \cdot } { \in } [ 0 , 1 ] ^ { ( T _ { \mathrm { p r e v } } + T _ { \mathrm { c u r } } ) \times 1 }$ by assigning each video frame the confidence score of its corresponding interval covering that frame:

$$
p _ { t } = c _ { j } , { \mathrm { ~ w h e r e ~ } } j : s _ { j } \leq t / f _ { \mathrm { f p s } } < e _ { j }\tag{11}
$$

where $f _ { \mathrm { f p s } } = 2 5$ . If no interval covers frame t, we set $p _ { t } = 0$ . The scalar prosody sequence is then projected into a compact latent representation through an MLP followed by a sigmoid activation:

$$
w _ { p , t } = \sigma \left( \mathrm { M L P } ( p _ { t } ) \right) , \quad w _ { p , t } \in \mathbb { R } ^ { 4 } .\tag{12}
$$

The projected prosody condition is concatenated with the listener motion context $w _ { r }$ , speaker audio features $w _ { a }$ , and speaker emotion features $w _ { e }$ for $c ^ { \mathrm { b a s e } }$

## D Evaluation Protocol Details

## D.1 Event Extraction and Time Representation

For each generated listener video, we apply the fixed reaction detector used in dataset construction to obtain predicted reaction events $\widehat { \mathcal { A } } \ : = \ : \{ ( \widehat { y } _ { i } , \widehat { t } _ { s , i } , \widehat { t } _ { e , i } ) \} _ { i = 1 } ^ { \widehat { N } }$ . The ground-truth event set ${ \mathcal { A } } =$ $\{ ( y _ { j } , t _ { s , j } , t _ { e , j } ) \} _ { j = 1 } ^ { N }$ is obtained from the manually verified reaction annotations in our curated dataset. All methods are evaluated with the same detector, class thresholds, temporal post-processing rules, and matching protocol. We represent reaction intervals as half-open temporal intervals $[ t _ { s } , t _ { e } )$ , using frame indices at 25 FPS for implementation. Equivalent timestamps in seconds produce the same temporal IoU because all timing errors are normalized by interval lengths.

## D.2 One-to-One Class-Aware Matching

Matching is performed independently for each reaction class to ensure that predicted events are only compared with ground-truth events of the same semantic category. For class $k ,$ let $\widehat { \mathcal { A } } _ { k } = \{ \widehat { a } _ { i } \in \widehat { \mathcal { A } }$ $\widehat { y } _ { i } = k \}$ and $\bar { \mathcal { A } } _ { k } \bar { = } \{ a _ { j } \in \mathcal { A } : y _ { j } = k \}$ . We first construct a candidate pair set

$$
\mathcal { C } _ { k } = \{ ( \widehat { a } _ { i } , a _ { j } ) : \widehat { a } _ { i } \in \widehat { \mathcal { A } } _ { k } , a _ { j } \in \mathcal { A } _ { k } , \mathrm { t D U } ( \widehat { a } _ { i } , a _ { j } ) \geq \tau \} .\tag{13}
$$

The candidate pairs are sorted by temporal IoU in descending order. We then greedily accept a candidate pair if neither its predicted event nor its ground-truth event has been matched before. This gives a one-to-one matched set $\mathcal { M } _ { k }$ for class k. The final matched set is the union over all reaction classes:

$$
\mathscr { M } = \bigcup _ { k = 1 } ^ { 6 } \mathscr { M } _ { k } .\tag{14}
$$

This matching strategy prevents one predicted reaction from explaining multiple ground-truth reactions, and also prevents multiple predicted reactions from being matched to the same ground-truth event.

## D.3 Event-Level Aggregation

We aggregate true positives (TP), false positives (FP), and false negatives (FN) over the dataset evaluation set for computing P (precision) and R (recall) before computing R-F1. Specifically, we compute TP, FP, and FN for class k as

$$
\mathrm { T P } _ { k } = | \mathcal { M } _ { k } | , \quad \mathrm { F P } _ { k } = | \widehat { \mathcal { A } } _ { k } | - | \mathcal { M } _ { k } | , \quad \mathrm { F N } _ { k } = | \mathcal { A } _ { k } | - | \mathcal { M } _ { k } | .\tag{15}
$$

Precision, recall and R-F1 are then computed as

$$
\mathbf { P } _ { k } = { \frac { \mathrm { T P } _ { k } } { \mathrm { T P } _ { k } + \mathrm { F P } _ { k } } } , \quad \mathbf { R } _ { k } = { \frac { \mathrm { T P } _ { k } } { \mathrm { T P } _ { k } + \mathrm { F N } _ { k } } } , \quad \mathbf { R } - \mathrm { F } 1 _ { k } = { \frac { 2 \mathbf { P } _ { k } \mathbf { R } _ { k } } { \mathbf { P } _ { k } + \mathbf { R } _ { k } } }\tag{16}
$$

Note that the final R-F1 for class k is averaged by $N _ { k }$ paired samples, and the overall R-F1 across all reaction classes is calculated by averaging the sum of R-F1 for each class. If a clip contains neither predicted nor ground-truth reactions, it contributes no TP, FP, or FN. If the denominator of precision or recall is zero after evalset-level aggregation, the corresponding value is set to zero. R-tIoU and R-ATD are computed over matched pairs after aggregation. If no matched pair exists for an evaluated set, R-tIoU is set to zero and R-ATD is not reported for that set; this case does not occur in our reported experiments.

## D.4 R-ATD Penalization

For each matched pair $( { \widehat { a } } , a ) \in { \mathcal { M } }$ , where $\widehat { \boldsymbol { a } } = ( \widehat { \boldsymbol { y } } , \widehat { t } _ { s } , \widehat { t } _ { e } )$ and $a = \left( y , t _ { s } , t _ { e } \right)$ , we compute the ground-truth duration $\Delta t = t _ { e } - t _ { s }$ and the predicted duration $\widehat { \Delta t } = \widehat { t } _ { e } - \widehat { t } _ { s }$ . The normalized deviations are

$$
\delta _ { s } = \frac { \widehat { s } - s } { \Delta t } , \qquad \delta _ { e } = \frac { \widehat { e } - e } { \Delta t } , \qquad \delta _ { \Delta t } = \frac { \widehat { \Delta t } - \Delta t } { \Delta t } .\tag{17}
$$

Table 10: R-ATD using different pairs of α and $\beta .$
<table><tr><td>α</td><td>β</td><td>φ(δs)</td><td>φ(δe)</td><td> $\overline { { \phi ( \delta _ { \Delta t } ) } }$ </td><td>R-ATD</td></tr><tr><td>2.0</td><td>0.5</td><td>22.864</td><td>17.312</td><td>17.212</td><td>57.388</td></tr><tr><td>1.0</td><td>1.0</td><td>18.120</td><td>14.615</td><td>14.315</td><td>47.050</td></tr><tr><td>0.5</td><td>2.0</td><td>22.436</td><td>19.238</td><td>18.563</td><td>60.237</td></tr></table>

The penalty function is asymmetric:

$$
\phi ( \delta ; \alpha , \beta ) = \left\{ { \alpha | \delta | , \delta < 0 } , \right.\tag{18}
$$

We set $\alpha = 2$ and $\beta = 0 . 5$ in all experiments. We argue that in real-world conversational scenarios, it is unnatural if a listener reaction should not occur prior to speaker finishing its contexts. Therfore, negative start-time deviation corresponds to a reaction that starts earlier than the ground-truth reaction, which is penalized more strongly because premature listener feedback is often perceived as unnatural.

The same asymmetric form is applied to end-time and duration deviations for consistency: similarly, we expect the listener to have sufficient time for its reaction, so we penalize more if the reaction ends earlier than it should be. We also suggest that a reaction is considered better if it has sufficient time for expression, so we apply more penalty on reactions shorter than they should be.

We conduct a sensitivity analysis on α and β in Table 10 to examine how different temporal preferences affect the reported R-ATD values. The numbers are reported on the RealTalk dataset. The symmetric setting $( \alpha = 1 , \beta = 1 )$ yields a lower numerical R-ATD because it does not impose additional penalty on premature or truncated reactions. In contrast, our default setting $( \alpha = 2 , \beta = 0 . 5 )$ intentionally assigns larger costs to negative deviations, which better reflects our evaluation preference that reactions starting too early, ending too early, or being too short are more disruptive in dyadic interaction.

## D.5 R-FID Computation

R-FID [34] is computed at the evaluation-dataset level. For generated videos, we collect frames inside the union of predicted reaction intervals:

$$
\Omega _ { \mathrm { g e n } } = \bigcup _ { { \widehat { a } } _ { i } \in { \widehat { A } } } [ { \widehat { s } } _ { i } , { \widehat { e } } _ { i } ) .\tag{19}
$$

For ground-truth videos, we collect frames inside the union of annotated reaction intervals:

$$
\Omega _ { \mathrm { r e a l } } = \bigcup _ { a _ { j } \in \mathcal { A } } [ s _ { j } , e _ { j } ) .\tag{20}
$$

If multiple reaction intervals overlap, the corresponding frame is counted once. We then extract visual features from generated frames in $\Omega _ { \mathrm { { g e n } } }$ and real frames in $\Omega _ { \mathrm { r e a l } }$ using the same feature extractor as standard FID. Let $( \mu _ { \mathrm { g e n } } , \Sigma _ { \mathrm { g e n } } )$ and $\left( \mu _ { \mathrm { r e a l } } , \Sigma _ { \mathrm { r e a l } } \right)$ denote the mean and covariance of generated and real reaction-frame features, respectively. R-FID is computed as

$$
\mathrm { R \mathrm { - } F I D } = \| \mu _ { \mathrm { g e n } } - \mu _ { \mathrm { r e a l } } \| _ { 2 } ^ { 2 } + \mathrm { T r } \left( \Sigma _ { \mathrm { g e n } } + \Sigma _ { \mathrm { r e a l } } - 2 ( \Sigma _ { \mathrm { g e n } } \Sigma _ { \mathrm { r e a l } } ) ^ { 1 / 2 } \right) .\tag{21}
$$

For per-class R-FID, we apply the same procedure after filtering both predicted and ground-truth events by reaction class. Pooling reaction frames at the dataset level avoids unstable estimates, especially because listener reactions are sparse.

## E Additional Experiment Results

## E.1 Qualitative Results on Seamless Dataset

Fig. 4 presents additional qualitative comparisons on the Seamless dataset, illustrating both the realism and temporal alignment of generated listener behaviors. We show two representative conversational scenarios with different speakers and listener identities.

![](images/e81778a3977308b7422b4606df60d1804ca401d6df85378425e50fbacef73e43.jpg)  
Figure 4: Qualitative comparisons between our baseline and other state-of-the-art methods on Seamless dataset. Reaction labels are displayed when a reaction is detected at the corresponding frame. Our method generates more natural listening motions and reactions that are more temporally aligned with ground truths than other comparisons.

Similar to observations illustrated in Fig. 3 for RealTalk, clear differences can be seen that the comparison methods often produce overly static or weak facial dynamics, with limited variation across frames. As a result, these methods frequently fail to trigger meaningful listener reactions even when strong conversational cues are present in the speaker audio. Most importantly, they sometimes output reactions that are relatively not so appropriate or closely associated with the given contexts.

Our method, on the other hand, produces more expressive and contextually appropriate listener behaviors. In the first example, where the speaker exhibits a clear emphasis in speech, our model generates a well-timed nodding response that closely aligns with the ground-truth reaction, while other methods either miss the reaction or produce weaker and less consistent motion. In the second example, our method captures a subtle smiling response that emerges naturally along the conversational flow, whereas competing methods either delay the response or generate less coherent facial expressions. Notably, our reactions are not only more visible, but also better synchronized with the speaker’s prosodic cues.

Overall, these qualitative results demonstrate that our approach improves both the type correctness and temporal alignment of listener reactions, leading to more natural, expressive, and coherent conversational behaviors compared to prior methods.

## E.2 More Ablation Studies

Due to space limit in the main paper, here we present three more ablation studies, including input speaker audio length, prosody channel projection, and temporal reaction loss type.

Table 11: Effect of audio length on model performance.
<table><tr><td rowspan="2">Audio Length (s)</td><td colspan="10">Metrics</td></tr><tr><td>PSNR ↑</td><td>SSIM↑</td><td>FID↓</td><td>FVD↓</td><td>Var ↑</td><td>LPIPS ↓ rPCC ↓</td><td></td><td>DI-Sync ↑</td><td>R-F1↑</td><td>R-tIoU ↑</td><td>R-ATD↓</td><td>R-FID↓</td></tr><tr><td>2</td><td>16.684</td><td>0.584</td><td>37.780</td><td>164.74</td><td>2.686</td><td>0.496</td><td>0.212</td><td>0.224</td><td>0.577</td><td>0.678</td><td>68.697</td><td>17.126</td></tr><tr><td>3</td><td>16.987</td><td>0.592</td><td>36.158</td><td>152.08</td><td>2.874</td><td>0.492</td><td>0.236</td><td>0.242</td><td>0.588</td><td>0.684</td><td>64.563</td><td>16.554</td></tr><tr><td>5</td><td>17.371</td><td>0.599</td><td>35.796</td><td>144.32</td><td>2.952</td><td>0.466</td><td>0.201</td><td>0.237</td><td>0.596</td><td>0.695</td><td>60.285</td><td>15.724</td></tr><tr><td>10</td><td>17.973</td><td>0.601</td><td>35.691</td><td>142.456</td><td>2.913</td><td>0.454</td><td>0.227</td><td>0.245</td><td>0.594</td><td>0.704</td><td>57.386</td><td>15.192</td></tr><tr><td>20</td><td>17.884</td><td>0.597</td><td>35.824</td><td>147.67</td><td>2.949</td><td>0.444</td><td>0.233</td><td>0.240</td><td>0.590</td><td>0.712</td><td>58.597</td><td>16.055</td></tr><tr><td>50</td><td>17.592</td><td>0.589</td><td>36.319</td><td>145.34</td><td>2.940</td><td>0.449</td><td>0.248</td><td>0.235</td><td>0.592</td><td>0.707</td><td>62.446</td><td>15.893</td></tr></table>

Table 12: Ablation study on channel dimension projection for prosodic condition.
<table><tr><td rowspan="2">Channel number</td><td colspan="10">Metrics</td></tr><tr><td>PSNR ↑</td><td>SSIM ↑</td><td>FID↓</td><td>FVD↓</td><td>Var ↑</td><td>LPIPS↓</td><td>rPCC↓</td><td>DI-Sync ↑</td><td>R-F1↑</td><td>R-tIoU ↑</td><td>R-ATD↓</td><td>R-FID↓</td></tr><tr><td>1</td><td>17.797</td><td>0.596</td><td>35.704</td><td>148.925</td><td>2.783</td><td>0.475</td><td>0.235</td><td>0.242</td><td>0.602</td><td>0.689</td><td>70.474</td><td>16.345</td></tr><tr><td>2</td><td>18.021</td><td>0.603</td><td>35.832</td><td>144.796</td><td>2.928</td><td>0.457</td><td>0.234</td><td>0.233</td><td>0.582</td><td>0.695</td><td>60.178</td><td>16.502</td></tr><tr><td>4</td><td>17.973</td><td>0.601</td><td>35.691</td><td>142.456</td><td>2.913</td><td>0.454</td><td>0.227</td><td>0.245</td><td>0.594</td><td>0.704</td><td>57.386</td><td>15.192</td></tr><tr><td>8</td><td>18.792</td><td>0.584</td><td>35.989</td><td>145.776</td><td>2.862</td><td>0.460</td><td>0.230</td><td>0.247</td><td>0.582</td><td>0.699</td><td>59.404</td><td>15.220</td></tr><tr><td>16</td><td>18.464</td><td>0.581</td><td>35.794</td><td>146.435</td><td>2.848</td><td>0.457</td><td>0.211</td><td>0.250</td><td>0.556</td><td>0.704</td><td>58.691</td><td>15.348</td></tr></table>

Input speaker audio length. Different length of input audio from speaker can provide different amount of contexts. Table 11 shows that increasing speaker’s audio length generally improves performance, as longer temporal context provides richer cues for both motion generation and reaction prediction. Performance improves notably from 2s to 10s across most metrics, including visual quality (PSNR, FVD) and reaction accuracy (DI-Sync, R-F1, R-tIoU). However, further increasing the length (e.g., 20s or 50s) brings marginal gains or even slight degradation, likely due to increased temporal redundancy and modeling difficulty. Overall, 10s achieves the best balance between sufficient context and stable modeling.

Prosody Channel Projection. In our architecture, we project the channel dimension from 1 of framealigned prosodic intensity scalar p into 4 of prosody condition w<sub>p˜</sub>. We ablate this channel projection in Table 12. The results show that introducing a learnable channel projection for the prosodic intensity signal consistently improves performance compared to using the raw scalar (channel=1), indicating the benefit of increasing representation capacity for prosody conditioning. As the channel dimension increases from 1 to 4, we observe steady gains in both generation quality (e.g., lower FVD, LPIPS, and rPCC) and reaction-related metrics (e.g., higher R-tIoU and lower R-ATD), suggesting improved temporal alignment and more accurate reaction dynamics. However, further increasing the channel dimension to 8 does not bring consistent improvements and slightly degrades several metrics, implying diminishing returns and potential over-parameterization. Overall, projecting the prosodic signal to a moderate channel size (4) achieves the best trade-off between expressiveness and stability, and is thus adopted in our final model.

Table 13: Ablation study on reaction loss type.
<table><tr><td rowspan="2">Channel number</td><td colspan="10">Metrics</td></tr><tr><td>PSNR ↑</td><td>SSIM↑</td><td>FID↓</td><td>FVD↓</td><td>Var ↑</td><td>LPIPS↓</td><td>rPCC ↓</td><td>DI-Sync ↑</td><td>R-F1↑</td><td>R-tIoU ↑</td><td>R-ATD↓</td><td>R-FID ↓</td></tr><tr><td>BCE</td><td>17.695</td><td>0.597</td><td>35.702</td><td>159.266</td><td>2.884</td><td>0.461</td><td>0.240</td><td>0.249</td><td>0.598</td><td>0.712</td><td>59.185</td><td>17.996</td></tr><tr><td>Smooth L1</td><td>17.973</td><td>0.601</td><td>35.691</td><td>142.454</td><td>2.913</td><td>0.454</td><td>0.227</td><td>0.245</td><td>0.594</td><td>0.704</td><td>57.386</td><td>15.192</td></tr></table>

Reaction Loss Type. We use Smooth-L1 for computing the reaction loss, and we ablate this loss function with BCE loss, where results are reported in Table 13. While BCE achieves slightly better scores on detection-oriented metrics such as DI-Sync, R-F1, and R-tIoU, Smooth-L1 improves generation quality (e.g., lower FVD, LPIPS, and rPCC) as well as temporal deviation metrics (R-ATD and R-FID). This suggests that Smooth-L1 provides a more stable and regression-friendly supervision signal for continuous frame-wise reaction intensities, leading to better temporal consistency and overall motion quality. Therefore, we adopt Smooth-L1 as the default reaction loss.

## F Discussion of Limitations and Future Works

Despite the improvements achieved in reaction-aware listening head generation, several limitations remain.

Reaction subjectiveness. Listener reactions are inherently multi-valid and subjective: given the same conversational cue, different listeners may exhibit different reaction types, timings, or intensities. Our current evaluation follows a single-reference ground-truth setting: the goal is to measure whether the generated listener reaction is consistent with the observed human reaction in the dataset. Therefore, R-F1, R-tIoU, R-ATD, and R-FID should be interpreted as reference-consistency metrics rather than absolute judgments that only one reaction is socially appropriate.

Complexity of listener’s behaviors. Our work focuses on a set of explicit actual reactions (e.g., nodding, smiling), while listener behavior, more broadly, also includes micro-behaviors (e.g., subtle gaze or facial changes) and baseline dynamics (e.g., natural head motion and posture drift showing the person is not stationary). These components are more continuous and harder to annotate, and are not explicitly modeled in our current framework.

Multimodal attributes. Listener behavior is influenced by factors beyond audio prosody, such as language semantics, personality, and conversational context, which are only partially captured in our current model. Extending toward richer multimodal understanding and more comprehensive modeling of listener motion remains an important direction for future work.

Preference learning. We choose supervised temporal reaction modeling instead of reinforcement learning because our goal is to explicitly inject reaction-related temporal supervision into the generation process. The curated dataset provides frame-level reaction annotations, which allow direct optimization of reaction occurrence and temporal consistency. Applying reinforcement learning would require designing a reward function that accurately reflects human judgments of reaction appropriateness, but will theoretically optimize listener’s behaviors based on human preferences. Therefore, introducing reinforcement learning will likely further improve listener’s generation quality.

Full-duplex and multi-party scenarios. Our GLARE is currently designed to handle listener’s motion with reaction generation in conversations with only one speaker, not directly targeting full-duplex or multi-party scenarios, despite being more real-world but complicated at the same time. However, we believe it can be extended to adapt to above tasks. For example, one can adopt any existing full-duplex module to decide when the avatar should speak or listen, in which the corresponding speaking-head module or GLARE would produce facial motion generation based on the switching policy. Furthermore, an interaction-level arbitrator can be introduced to determine the conversational state of each participant in multi-party scenarios, including who is speaking, who is listening, and when state transitions occur. The corresponding speaking or listening generation module can then be activated according to the current interaction state. These would be our future research directions.

## G Broader Impact

This work aims to improve the naturalness and appropriateness of listener behavior in conversational agents, with potential applications in virtual assistants, social robots, telepresence, and humancomputer interaction systems. By enabling more responsive and context-aware listener reactions, our approach may contribute to more engaging and human-like interactions.

At the same time, improved generation of realistic listener behaviors may introduce risks related to synthetic media, such as creating misleading or deceptive conversational content. In addition, listener reactions can implicitly convey agreement or emotional alignment, which may be misinterpreted or misused in certain contexts. We emphasize that this work is intended for research purposes, and responsible deployment should include appropriate safeguards such as clear disclosure of synthetic content, adherence to dataset licensing terms, and consideration of fairness and diversity in conversational behaviors.