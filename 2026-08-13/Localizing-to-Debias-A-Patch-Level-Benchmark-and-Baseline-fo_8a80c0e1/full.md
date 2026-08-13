# Localizing to Debias: A Patch-Level Benchmark and Baseline for Weakly Supervised Spatial Anomaly Detection

Sara Abdulaziz<sup>1</sup> , Abdulrahman Al-Abri<sup>1</sup> , Giacomo D’Amicantonio<sup>1</sup> , and Egor Bondarev<sup>1</sup>

Eindhoven University of Technology, 5612 AE Eindhoven, Netherlands s.e.a.m.abdulaziz@tue.nl GitHub Code

Abstract. Despite growing interest in weakly supervised video anomaly detection (WSVAD), current methods struggle to bridge the gap between coarse temporal supervision and fine-grained spatial reasoning. A key obstacle is the tendency of temporal detectors to latch onto background and scene-level cues rather than truly discriminative anomaly evidence. This background bias raises ethical concerns: models may inadvertently associate anomalies with societal or environmental context rather than authentic crime-related cues. Without spatial grounding, such biases remain hidden and unauditable. To address this, we propose SST-WSVADL, a sparse spatio-temporal framework that bridges temporal anomaly detection with fine-grained spatial localization. Rather than processing all spatial regions indiscriminately, SST-WSVADL progressively focuses on the most anomaly-relevant spatio-temporal regions through dynamic sparsification, naturally suppressing background dominant content while preserving discriminative evidence. The temporal and spatial branches are coupled end-to-end via motion-aware regularization that guides sparsification toward dynamically informative regions, without relying on external detectors or vision-language prompts. We publicly release frame-level spatial annotations and a method-agnostic evaluation protocol for three public datasets: UCF-Crime, XD-Violence, and MSAD. These resources enable the community to audit spatial biases in WSVAD predictions, supporting progress toward more ethical and accountable anomaly detection. Experiments demonstrate that SST-WSVADL is competitive with prior methods across benchmarks while enabling localization and patch-level auditability of scene bias, providing a reproducible foundation for interpretability-oriented evaluation of WSVAD models.

Keywords: Video Anomaly Detection · Anomaly Localization Benchmark · Token Pruning · Patch-Level Evaluation

## 1 Introduction

Video Anomaly Detection (VAD) has been a long-standing challenge due to the rarity, diversity, and context dependence of anomalous events. In practice, the cost and ambiguity of annotation make fully supervised spatio-temporal labeling impractical at scale. Consequently, weakly supervised VAD (WSVAD) has emerged as the dominant paradigm, where models are trained using only videolevel labels while inferring fine-grained temporal and spatial anomalies at test time. However, this weak supervision induces a structural limitation: rich spatiotemporal evidence is compressed into coarse segment-level scores, often causing models to rely on global scene context rather than truly discriminative local motion cues. Recent studies further reveal that such global representations amplify background bias [16], where static scene elements dominate anomaly predictions and degrade localization reliability. Systematically addressing this transparency concern remains an open challenge in WSVADL. Beyond performance, this bias carries ethical implications: models that associate anomalies with background scene context, such as location or ethnic attributes, rather than actual criminal behavior risk making decisions that are unfair to certain societal segments. For instance, flagging ordinary cultural practices or daily routines as suspicious simply because they occur in low-income or minority-dense neighborhoods that are overrepresented in training data for crimes like robbery or assault.

To address these limitations, recent eforts have extended WSVAD toward weakly-supervised spatio-temporal anomaly detection and localization (WSVADL), yet existing approaches consistently trade one external dependency for another, e.g. object detectors [32], dense annotations [16], or vision-language models [33], to guide localization. Crucially, none of these strategies tackle background bias end-to-end, leaving interpretability and domain robustness as persistent open concerns. These limitations are further compounded by the lack of spatial annotations and unified evaluation protocols: existing works report a mix of frame-AUC, temporal-IoU [16,33], mean-IoU [32], or prompt-based heatmaps [23], preventing consistent and fair comparison across methods.

In this work, we address both challenges jointly. We curate and release spatial anomaly annotations for three public datasets: UCF-Crime, XD-Violence, and MSAD, paired with a method-agnostic patch-level evaluation protocol that enables fair, reproducible, and bias-auditable comparison. Building on this foundation, we introduce SST-WSVADL, a lightweight end-to-end framework that grounds anomaly evidence spatially through structured sparsification, without relying on external detectors, dense annotations, or vision-language models. In short, the contributions of this work are three-fold:

1. We release a unified spatial anomaly localization benchmark for UCF-Crime, XD-Violence, and MSAD, together with a method-agnostic patch-level evaluation protocol that enables reproducible comparison and auditing of ethical biases in WSVAD predictions.

2. We propose SST-WSVADL, a detector-free, annotation-light, and non-VLM framework that unifies snippet-level temporal reasoning with motion-aware patch-level localization under weak supervision, without external dependencies.

3. We provide a transparent baseline for the unified benchmark, along with a reproducible and interpretability-oriented evaluation of spatial bias in WS-VADL methods.

## 2 Related Work

Video anomaly detection (VAD) aims to identify events that deviate from normal behavioral patterns in surveillance videos. Traditional approaches relied on handcrafted features and shallow models, such as sparse representation [17, 36] or motion-based descriptors [5,7,9,14,38], to detect deviations from learned regularity motion patterns. With the success of deep learning, Sultani et al. [22] introduced the UCF-Crime dataset and the first deep multiple instance learning (MIL) framework, which became the foundation for most weakly-supervised video anomaly detection (WSVAD) methods.

Early studies on WSVAD treated videos as globally averaged features, processing full-frame representations for anomaly detection, while overlooking the local spatial variation and motion cues [22]. Landi et al. [10] demonstrated that anomalies are inherently local, and occupy a limited region of space and time, and proposed analyzing spatio-temporal tubes instead of entire frames. Their work established UCFCrime2Local, the first dataset with bounding-box annotations, showing that focusing on localized regions improves detection accuracy and robustness. Complementarily, Liu and Ma [16] revealed the background bias problem, where global representations inadvertently correlate with static context rather than actual motion or actions, thereby misguiding anomaly detection models. The authors proposed a supervised anomaly localization framework that later motivated spatially-focused and motion-aware designs [12,23,32]. Wu et al. [32] introduced WSSTAD, the first framework to jointly reason over space and time under weak supervision. It models anomalies at the tube level using spatio-temporal proposals instead of frame averages, bridging the gap between anomaly detection and action localization. Similarly, Li et al. [12] proposed a scale-aware spatio-temporal relation learning approach that captures dependencies across multiple granularities using transformers. Sun et al. [23] advanced temporal modeling through long-short temporal co-teaching, improving snippet consistency and label reliability in weak supervision settings. More recently, vision-language models (VLMs) have reshaped WSVAD by introducing semantic priors through text prompts. Wu et al. [33] proposed STPrompt, which leverages spatio-temporal prompting and large language models (LLMs) for training-free spatial localization, unifying temporal detection and spatial reasoning. While this method demonstrated promising results, its dependence on large pre-trained VLMs and limited-context models, such as CLIP, constrains both flexibility and interpretability. In contrast, our framework is detector-free, non-VLM, and annotation-light, leveraging learnable sparsity for eficient and fully end-to-end anomaly localization.

## 3 Datasets and Spatial Annotations

We build upon previous eforts on anomaly localization [10,16] and extend spatial annotations to full UCF-Crime, XD-Violence, and MSAD datasets. Whereas a temporal annotation captures when an anomaly occurs, a spatial annotation captures where it occurs via per-frame bounding boxes. The anomaly box encloses all actors and relevant objects involved in the event. The temporal extent of an anomaly begins when all required actors are visibly engaged in the anomalous action and ends when they leave the scene or the action ends. The set of frame-wise boxes over this interval constitutes a video tube, which forms the annotation. Our annotation protocol is designed to produce temporally stable, semantically consistent bounding boxes that align with the patch-level evaluation granularity. It proceeds in two stages:

1. Event timing. Anomalous events are temporally bounded by identifying onset and ofset frames, distinguishing between anticipatory context and the anomaly itself.

2. Spatial localization. For each event, rather than annotating tight perframe boxes, we adopt a semantically-driven update policy that reflects the underlying event structure:

– Stationary events. A single box is maintained for the event duration, anchored to the region of occurrence rather than individual actor positions.

– Non-stationary events. The box is updated only when the event has shifted substantially beyond the current boundary, prioritizing semantic continuity over frame-level precision.

This procedure difers from prior annotation eforts on UCF-Crime [10, 16], which produce excessively shifting boxes that introduce noise and reduce evaluation consistency. By contrast, our semantically-driven update policy yields temporally stable annotations that better reflect the true spatial extent of anomalous events, as demonstrated in Supp. S1.1. Detailed annotation statistics including bbox-to-frame ratios, tubelet density, and per-category breakdowns are reported in the Supp. S1.2.

## 4 Sparse Spatio-Temporal WSVADL Framework

The proposed Spatio-Temporal framework, SST-WSVADL, integrates two complementary branches. The snippet-level branch (S-WSVAD), which performs temporal anomaly detection over video snippets, and the patch-level branch (P-WSVAL), which localizes anomalies spatially based on sparsified tubelets. The snippet branch provides temporal anomaly scores and generates clip proposals for the patch branch. The latter generates sparse tubelet representations for localization and provides a feedback to refine the temporal detection. Figure 1 illustrates the overall framework.

![](images/020c1a1424a816aa9632265b612f5ec9a6175199967f3da3abb7f16ae77e0792.jpg)  
Fig. 1: Overview of the proposed SST-WSVADL. The framework consists of two interacting branches. (Left) The Snippet-WSVAD branch is a standard WSVAD model [37] that processes snippet-level features extracted via a pre-trained task-agnostic backbone [27]. The normal and abnormal clips for the patch branch are selected through the Online Clip Proposal (OCP) module. (Right) The Patch-WSVAL branch converts snippet proposals into tubelets, encodes them by the Dynamic Tubelet-Feature Encoder (DTFE), and computes motion scores that bias DTFE towards dynamic tubelets. A Patch-Snippet Attention (PSA) module fuses the compact spatial and temporal features before the final classification head.

## 4.1 Snippet-level WSVAD

The snippet branch follows a standard MIL-based WSVAD approach [22], processing input videos as sequences of fixed-length snippets. Features are extracted with a pre-trained backbone and passed through a WSVAD algorithm [37]. We choose task-agnostic features for this branch, produced from a self-supervised VideoMAEv2 [27] trained on large scale video data and fine-tuned for action recognition. This enables better generalization, and finer motion-appearance disentanglement than the commonly adopted I3D backbone [2].

Online Clip Proposal Module. The online clip proposal module leverages the ability of the chosen WSVAD algorithm [37] to select the most discriminative video segments for the patch branch training. The algorithm stores prototypes of normal and abnormal samples in separate learnable memory banks. During training, the algorithm applies self-attention between the snippet features and the memory banks to assess the similarity of features to the stored prototypes. Then, the top-k features are selected based on the highest similarity to the prototypes. We leverage this property of scanning stored prototypes to propose the most representative video snippets for patch branch training. Concretely, we select the first from the top-k normal and abnormal features to train the patch branch. This proposal mechanism ensures the patch branch is trained on hard discriminative regions, rather than processing all video segments indiscriminately. In addition, since the exact temporal evidence provided by the MIL loss in the temporal branch is applied to the patch branch, the two branches exchange information about the same features, thereby guaranteeing alignment of their predictions.

## 4.2 Patch-level WSVAD

Complementing the snippet-level reasoning, the patch branch focuses on spatial anomaly localization within each frame. However, processing all patches exhaustively is computationally demanding, and more critically, risks diluting anomaly evidence with overwhelming background content, reinforcing the bias we aim to suppress. To address this, we localize anomalies by progressively focusing on the most relevant spatio-temporal regions. The Online Clip Proposal (OCP) module first narrows the candidate clips to those aligned with anomaly prototypes. Subsequently, dynamic sparsification adaptively discards VAD-irrelevant tubelets while retaining those critical for localization.

The patch-level branch consists of four components: a tubelet generator, the Dynamic Tubelet Feature Encoder (DTFE), Patch-Snippet Attention (PSA), and a WSVAD objective. DTFE integrates local, spatial, and spatio-temporal cues while progressively pruning uninformative tubelets, producing a compact set of discriminative representations. These retained tubelets are fused with snippetlevel features through PSA and optimized for spatial anomaly localization. It is worth noting that the primary role of dynamic sparsification is not computational reduction only, but to enable structured feature selection under weak supervision. By suppressing background-dominant tokens, it prevents spatial reasoning from defaulting to global scene context and grounds localization in discriminative anomaly cues.

Tubelet Generation. A video segment of F frames is first divided temporally into $t _ { z }$ sub-segments of equal duration. Within each sub-segment, every frame is partitioned into a spatial grid of non-overlapping patches of size $P \times P$ pixels, yielding $\begin{array} { r } { T = \frac { H } { P } \times \frac { \mathbf { \bar { \psi } } _ { W } } { P } } \end{array}$ spatial locations per sub-segment. Patches at the same spatial location across the frames of a sub-segment are stacked to form a tubelet, giving $N = t _ { z } \times T$ tubelets per segment in total. The split size $t _ { z }$ controls the temporal resolution.

Dynamic Tubelet-Feature Encoder (DTFE). For tubelets encoding, we adopt a tubelet-based Transformer design that follows a dynamic token sparsification strategy [13, 20]. The full mathematical formulation is provided in Supp. 2.1.

Patch-Snippet Attention (PSA). To further align the two branches, we introduce a patch-snippet context attention module. Let $S = \{ s _ { t } \} _ { t = 1 } ^ { T }$ denote the final snippet-level features from the temporal branch (S-WSVAD), and $Y = \{ y _ { i } \} _ { i = 1 } ^ { N ^ { \prime } }$ the spatio-temporal token embeddings from DTFE. We model their interaction by a cross-attention mechanism, where patch tokens attend to the snippet-level context

$$
\mathrm { A t t n } ( Y , S ) = \mathrm { S o f t m a x } \left( \frac { Q ( y _ { i } ) K ( S ) ^ { \top } } { \sqrt { d } } \right) V ( S ) ,\tag{1}
$$

with $Q , K , V$ being learnable linear projections. The updated patch representation is given by

$$
\hat { y } _ { i } = y _ { i } + \mathrm { A t t n } ( y _ { i } , S ) .\tag{2}
$$

This context mixing ensures two properties: (1) patch-level anomaly localization remains temporally consistent with snippet predictions, and (2) gradients from patch-level branch flow back into snippet-level features, thereby providing stronger supervision for the temporal branch.

## 4.3 Training Procedure

The two-branch framework is trained end-to-end. In line with the standard MILbased WSVAD approaches [22, 37], we form snippet pairs from normal and abnormal sets for the S-WSVAD branch, which later produces proposals by the OCP module. These proposals are processed by P-WSVAL to generate normal and abnormal tubelets for localization. Following prior optimization strategies [37], we train both snippet and patch-WSVAD branches with a binary crossentropy (BCE) loss $L _ { c l s }$ added to four other supplementary losses $L _ { d m } , L _ { t r i p } .$ $L _ { k l } , \ L _ { d i s } \ [ 3 7 ]$

$$
\mathcal { L } _ { W S V A D } = L _ { \mathrm { { c l s } } } + \lambda _ { 1 } L _ { d m } + \lambda _ { 2 } L _ { t r i p } + \lambda _ { 3 } L _ { k l } + \lambda _ { 4 } L _ { d i s } ,\tag{3}
$$

where $\lambda _ { 1 } , \lambda _ { 2 } , \lambda _ { 3 } , \lambda _ { 4 }$ are hyperparameters.

Motion Regularization In P-WSVAL, we introduce a motion-aware regularization term to mitigate background shortcut learning under weak supervision. Without additional bias, dynamic token sparsification may retain visually salient yet motion-irrelevant regions (static context), leading to unstable spatial reasoning. We impose a motion prior that encourages DTFE to preserve dynamically informative tubelets during pruning. Instead of extracting motion at the RGB level (e.g., optical flow), which is computationally expensive at the tubelet scale, we model motion directly in the embedding space.

Building on principles from spatio-temporal representation learning [26, 29] and symmetry decomposition [19], we split the tubelet embedding into appearance (time-symmetric) and motion (time-antisymmetric) components, following the intuition that motion cues change under temporal reversal while appearance cues do not. This allows isolating motion through extracting the timeantisymmetric component from the tubelet embedding. For each tubelet $t _ { i } ,$ we form its time-reversed counterpart $t _ { i } ^ { \mathrm { r e v } }$ , then embed both with a Conv3D projection f(·)

$$
x _ { i } = f ( t _ { i } ) , \qquad x _ { i } ^ { \mathrm { r e v } } = f ( t _ { i } ^ { \mathrm { r e v } } ) .\tag{4}
$$

Next, we decompose the feature into appearance and motion subspaces by computing the even and odd components respectively

$$
\begin{array} { r } { \mathbf { c } _ { i } ^ { \mathrm { a p p } } = \frac { 1 } { 2 } \big ( x _ { i } + x _ { i } ^ { \mathrm { r e v } } \big ) , \qquad \mathbf { c } _ { i } ^ { \mathrm { m o t } } = \frac { 1 } { 2 } \big ( x _ { i } - x _ { i } ^ { \mathrm { r e v } } \big ) . } \end{array}\tag{5}
$$

We then define the motion score as the feature-norm of the motion component

$$
\begin{array} { r } { \mathbf { m } _ { i } \ = \ \left\| \mathbf { c } _ { i } ^ { \mathrm { m o t } } \right\| _ { 2 } \ = \ \frac { 1 } { 2 } \left\| x _ { i } - x _ { i } ^ { \mathrm { r e v } } \right\| _ { 2 } , } \end{array}\tag{6}
$$

where $\mathbf { m } _ { i }$ is the motion score for tubelet $t _ { i }$ . The motion scores of all the tubelets form the motion matrix M, which is computed once before the DTFE, then coupled with the binary keep decision matrix $\mathcal { G } _ { l }$ at each pruning layer l. The motion loss is defined as

$$
\mathcal { L } _ { \mathrm { m o t i o n } } = - \frac { 1 } { N _ { l } } \sum _ { i = 1 } ^ { N _ { l } } ( \mathcal { G } _ { l } \cdot \mathcal { M } ) ,\tag{7}
$$

where the negative sign encourages the model to keep tokens with high motion, aligning sparsification with anomaly-relevant dynamics. Finally, the S-WSVAD and P-WSVAL losses are simplified as

$$
\begin{array} { r } { \mathcal { L } _ { S } = \mathcal { L } _ { W S V A D } , \quad \mathcal { L } _ { P } = \mathcal { L } _ { W S V A D } + \lambda _ { 5 } \mathcal { L } _ { m o t i o n } } \end{array}\tag{8}
$$

## 5 Experiments

## 5.1 Evaluation Metrics

We follow the standard evaluation procedures established in prior works for temporal anomaly detection evaluation. For spatial localization, we adopt two evaluation protocols. The first is computing the Temporal IoU (TIoU) and Mean IoU (MIoU) on the anomaly test frames, following prior works [16, 32, 33]. The second is our proposed patch-wise protocol, where we compute the AUC and AP directly from all patch scores across all test frames. In detail, the frame and patch scores are first computed for each video in the test set. Then, the patch scores are masked by the temporal frame scores. Finally, the patch scores are flattened to compute the Patch AUC (PAUC) and Patch AP (PAP) metrics. We demonstrate this evaluation protocol in detail in Supp. S3.

## 5.2 Implementation Details

We adopt a pre-trained VideoMAEv2 [27], fine-tuned on Kinetics-710 dataset, to extract the snippet features for the S-WSVAD branch. The WSVAD model applied in both branches is UR-DMU [37]. Following standard practices, we compress each video into 200 segments by aggregating and averaging the 16- frame snippet features. The split size $t _ { z }$ in P-WSVAL is set empirically to 2 for all datasets. The learning rate is set to 0.0001 in both S-WSVAD and P-WSVAL. The models are trained with a batch size of 16 for 4000 iterations. The hyperparameters $( \lambda _ { 1 } , \lambda _ { 2 } , \lambda _ { 3 } ,$ and $\lambda _ { 4 } )$ are set to their defaults in [37] and the motion loss weight $\lambda _ { 5 }$ is set to 0.01. All experiments are performed under a fixed random seed to ensure reproducibility across runs.

## 5.3 Comparison with state-of-the-art

We first establish a temporal WSVAD baseline (S-WSVAD) by training UR-DMU [37] on VideoMAEv2 snippet features. Compared to the commonly used I3D backbone in prior works [4, 18, 24, 37], VideoMAEv2 provides consistent performance gains, as validated in our experiments. We compare models trained with both feature types to quantify this improvement. Finally, we evaluate our end-to-end SST-WSVADL against prior anomaly localization methods in both temporal detection and spatial localization [15, 16, 21, 28, 30, 31, 33].

Temporal Anomaly Detection Tables 1-3 present comprehensive temporal and spatial evaluations of the proposed approach on UCF-Crime, MSAD, and XD-Violence datasets. SST-WSVADL delivers com petitive joint temporal and spatial performance. In temporal detection, our method improves AUC and AP on MSAD by $+ 5 . 4 9$ and +11.76 respectively over $\pi { \mathrm { - V A D } }$ [18], while achieving competitive performance on UCF-Crime and XD-Violence despite relying on a single modality. Even where SST-WSVADL only matches its temporal-only counterpart on UCF-Crime, grounding detections in localized spatial evidence

Table 1: Comparison with state-of-the-art methods on UCF-Crime. N/A indicates metrics not reported or irrelevant. $\mathrm { T I o U } _ { s u b }$ is measured based on the spatial annotations subset provided by [16].
<table><tr><td>Method</td><td>Features</td><td>AUC</td><td>AUCA</td><td>TIoUsub MIoUsub</td><td></td></tr><tr><td>Two Stream [21]</td><td>Two-stream</td><td>51.20</td><td>N/A</td><td>2.20</td><td>N/A</td></tr><tr><td>TSN [30]</td><td>TSN</td><td>53.20</td><td>N/A</td><td>2.60</td><td>N/A</td></tr><tr><td>C3D [25]</td><td>C3D</td><td>70.10</td><td>N/A</td><td>7.20</td><td>N/A</td></tr><tr><td>T-C3D [15]</td><td>C3D</td><td>74.50</td><td>N/A</td><td>10.20</td><td>N/A</td></tr><tr><td>ARTNet [28]</td><td>ARTNets</td><td>75.10</td><td>N/A</td><td>11.40</td><td>N/A</td></tr><tr><td>3DResNet [6]</td><td>I3D-ResNet</td><td>77.50</td><td>N/A</td><td>10.30</td><td>N/A</td></tr><tr><td>NLN [31]</td><td>I3D-ResNet</td><td>78.90</td><td>N/A</td><td>12.20</td><td>N/A</td></tr><tr><td>Liu et al. [16]</td><td>I3D-ResNet</td><td>82.00</td><td>N/A</td><td>16.40</td><td>N/A</td></tr><tr><td>Sultani et al. [22] I3D</td><td></td><td>67.11</td><td>N/A</td><td>16.82</td><td>N/A</td></tr><tr><td>VadCLIP [34]</td><td>CLIP</td><td>88.02</td><td>70.23</td><td>22.05</td><td>N/A</td></tr><tr><td>STPrompt [33]</td><td>CLIP</td><td>88.08</td><td>N/A</td><td>23.90</td><td>N/A</td></tr><tr><td>WSSTAD [32]</td><td>C3D</td><td>87.65</td><td>N/A</td><td>N/A</td><td>8.98</td></tr><tr><td>RTFM [24]</td><td>I3D</td><td>84.30</td><td>N/A</td><td>N/A</td><td>N/A</td></tr><tr><td>MGFN [4]</td><td>I3D</td><td>86.67</td><td>N/A</td><td>N/A</td><td>N/A</td></tr><tr><td>UR-DMU [37]</td><td>I3D</td><td>86.97</td><td>70.81</td><td>N/A</td><td>N/A</td></tr><tr><td>π-VAD [18]</td><td>I3D</td><td>90.33</td><td>77.77</td><td>N/A</td><td>N/A</td></tr><tr><td>S-WSVAD [37]</td><td>VideoMAEv2</td><td>88.07</td><td>73.98</td><td>N/A</td><td>N/A</td></tr><tr><td>SST-WSVADL VideoMAEv2 88.51</td><td></td><td></td><td>74.35</td><td>26.25</td><td>27.92</td></tr><tr><td></td><td></td><td></td><td></td><td>(+2.35)</td><td>(+18.94)</td></tr></table>

makes its predictions more interpretable and its background debiasing inspectable. While employing VideoMAEv2 features in S-WSVAD already yields gains over earlier approaches, integrating spatial features through end-to-end optimization in SST-WSVADL further strengthens temporal accuracy. Across both I3D and VideoMAEv2 features (Table 4), our model outperforms its temporalonly counterpart [37] across most configurations, except for I3D on UCF-Crime. Additional class-wise and anomaly-only metrics are provided in Supp. 4.1. These findings indicate that cross-branch information flow not only enables spatial localization but also improves temporal detection.

Spatial Anomaly Localization Tables 1 and 4 present the spatial localization results (highlighted in green). The tables report TIoU computed on two evaluation sets: (i) a previously annotated subset of frames by [16] $\left( T I o U _ { s u b } \right)$ and (ii) the complete set of anomalous frames in the test split, based on our extended annotations. On UCF-Crime, SST-WSVADL with VMAEv2 outperforms both the supervised approach [16] and the VLM-based STPrompt [33], reaching 14.59 TIoU. On MSAD and XD-Violence, our model achieves 26.27 and 29.49 TIoU respectively under the same VMAEv2 configuration. Table 4 shows that the choice of backbone afects datasets diferently. While VMAEv2 dominates UCF-Crime across all spatial metrics, I3D achieves competitive or superior scores on MSAD and XD-Violence for certain metrics. Due to the absence of publicly available code and trained models for prior localization approaches [10,33], direct comparison under TIoU, PAUC, and PAP is not feasible.

Table 2: Comparison with state-of-the-art methods on MSAD (temporal detection). N/A indicates metrics not reported.
<table><tr><td>Method</td><td>Features</td><td>AUC</td><td>AUCA</td><td>AP</td><td>APA</td></tr><tr><td>RTFM [24]</td><td>I3D</td><td>86.65</td><td>N/A</td><td>N/A</td><td>N/A</td></tr><tr><td>MGFN [4]</td><td>I3D</td><td>84.96</td><td>N/A</td><td>N/A</td><td>N/A</td></tr><tr><td>TEVAD [3]</td><td>I3D</td><td>86.82</td><td>N/A</td><td>N/A</td><td>N/A</td></tr><tr><td>UR-DMU [37]</td><td>I3D</td><td>85.78</td><td>67.95</td><td>67.35</td><td>75.30</td></tr><tr><td>π-VAD [18]</td><td>I3D</td><td>88.68</td><td>71.25</td><td>71.26</td><td>77.86</td></tr><tr><td>S-WSVAD [37]</td><td>VideoMAEv2</td><td>89.47</td><td>69.92</td><td>62.75</td><td>70.59</td></tr><tr><td>SST-WSVADL VideoMAEv2 94.17</td><td></td><td></td><td>80.21</td><td>83.02</td><td>85.01 (+11.76) (+7.15)</td></tr><tr><td colspan="6">(+5.49)</td></tr></table>

Table 3: Comparison with stateof-the-art methods on XD-Violence (temporal detection). N/A indicates metrics not reported.
<table><tr><td>Method</td><td>Features</td><td>AP</td><td>APA</td></tr><tr><td>RTFM [24]</td><td>I3D</td><td>77.81</td><td>78.57</td></tr><tr><td>UR-DMU [37]</td><td>I3D</td><td>81.66</td><td>83.94</td></tr><tr><td>TEVAD [3]</td><td>I3D</td><td>82.17</td><td>N/A</td></tr><tr><td>VadCLIP [34]</td><td>CLIP</td><td>84.51</td><td>N/A</td></tr><tr><td>π-VAD [18]</td><td>I3D</td><td>85.37</td><td>85.79</td></tr><tr><td>MissionGNN [35] I3D</td><td></td><td>N/A</td><td>98.42</td></tr><tr><td>S-WSVAD [37]</td><td>VideoMAEv2</td><td>85.33</td><td>98.69</td></tr><tr><td>SST-WSVADL VideoMAEv2 86.00</td><td></td><td></td><td>98.76</td></tr><tr><td></td><td></td><td>(+0.63)</td><td>(+0.34)</td></tr></table>

Table 4: Performance comparison under diferent spatio-temporal features.
<table><tr><td>Model</td><td>Features</td><td colspan="4">UCF-Crime</td><td colspan="2">MSAD</td><td colspan="4">XD-Violence</td></tr><tr><td></td><td colspan="16"></td></tr><tr><td colspan="2" rowspan="4">S-WSVAD [37] SST-WSVADL S-WSVAD [37]</td><td colspan="2">AUC</td><td colspan="2">AUCA</td><td colspan="2">Temporal Detection AUC</td><td colspan="2">AUCA</td><td colspan="2">AP</td><td colspan="2">APA</td></tr><tr><td colspan="2">86.97 86.50</td><td colspan="2">70.81</td><td colspan="2">85.78</td><td colspan="2">67.95</td><td colspan="2">81.66</td><td colspan="2">83.94</td></tr><tr><td colspan="2">I3D 88.07 88.51</td><td colspan="2">70.59 73.98</td><td colspan="2">89.00 89.47</td><td colspan="2">70.00 69.92</td><td colspan="2">82.39 85.33</td><td colspan="2">98.58 98.69</td></tr><tr><td colspan="2">VMAEv2 SST-WSVADL</td><td colspan="2"></td><td colspan="2">74.35</td><td colspan="2">94.17</td><td colspan="2">80.21</td><td colspan="2">86.00 98.76</td></tr><tr><td colspan="9">Spatial Localization</td><td colspan="2"></td></tr><tr><td rowspan="2"></td><td>I3D</td><td>PAUC</td><td>PAP</td><td>TIoU</td><td>MIoU PAUC</td><td>PAP</td><td>TIoU</td><td>MIoU</td><td>PAUC PAP</td><td>TIoU</td><td>MIoU</td></tr><tr><td>86.01</td><td>18.46</td><td>11.69</td><td></td><td>87.15</td><td>40.32</td><td>27.48</td><td>34.97</td><td>80.54</td><td>19.75</td><td></td></tr><tr><td colspan="2">SST-WSVADL VMAEv2</td><td>88.45</td><td>25.80</td><td>14.59</td><td>27.28 29.90</td><td>88.40 37.46</td><td>26.27</td><td>35.02</td><td>81.00</td><td>72.32 68.20 29.49</td><td>72.50 63.09</td></tr></table>

To assess the performance visually, we produce heatmaps representing the patch outliers. We explain the heatmap generation in detail in Supp. S3.2. Figures 2 and S3 show examples of anomalous frames with localization heatmaps at diferent timestamps. The visualizations demonstrate the ability of the proposed approach to accurately highlight anomalous regions. In some cases, the heatmaps are more precise than conventional bounding boxes. For instance, the model selectively attends to patches that tightly cover the anomalous actions (e.g., fighting, arrest, trafic accidents), while disregarding irrelevant background areas that are often present within coarse bounding-box annotations. This indicates that SST-WSVADL efectively captures the critical action-relevant cues of anomalous events. Nevertheless, localization remains challenging in some cases due to possible temporal inconsistencies and sensitivity to background motion in normal scenes. We discuss the limitations further in Supp. S6.

## 5.4 Ablation Studies

Efect of Online Clip Proposals. We evaluate how diferent proposal strategies affect performance, including a supervised setting and second-top, random-top, and multi-top sampling within the top-k predictions. In the supervised setting, P-WSVADL proposals are randomly sampled from ground-truth anomaly snippets without using the temporal branch rank-

Table 5: Comparison of the efect of proposal selection from the top-k samples on UCF-Crime.
<table><tr><td rowspan="2">Model</td><td>Temporal</td><td>Spatial</td></tr><tr><td>AUC AUCA|PAUC</td><td>PAP TIoU</td></tr><tr><td>Top-1</td><td>88.51 74.35</td><td>88.45 25.8014.59</td></tr><tr><td>Rank-2</td><td>88.29 72.46</td><td>88.47 15.81 12.92</td></tr><tr><td>Top-3</td><td>88.03 72.28</td><td>88.08 15.89 16.73</td></tr><tr><td>Random(Top-k)</td><td>88.26 71.96</td><td>84.59 11.74 12.55</td></tr><tr><td>Supervised</td><td>87.76 73.71</td><td>80.85 8.62 11.19</td></tr></table>

ing. As shown in Table 5, training with top-1 proposals yields the best overall spatio-temporal performance. In contrast, supervised sampling introduces temporal misalignment between branches, preventing cross-attention and degrading performance. Similarly, random or multiple selections from the top-k reduce precision by increasing false positives at the patch level, despite comparable TIoU scores. Although top-3 achieves the highest TIoU, top-1 provides better spatial precision and coverage, resulting in the most consistent performance.

Table 6: Ablation of the SST-WSVADL components on UCF-Crime.
<table><tr><td rowspan="2">DTFE PSA</td><td>Temporal</td><td>Spatial</td></tr><tr><td>AUC  $\mathrm { { A U C _ { A } } }$ </td><td>|PAUC PAP TIoU</td></tr><tr><td>X X</td><td>88.00 72.68</td><td>85.68 11.75 10.79</td></tr><tr><td>X</td><td>87.75 73.71</td><td>86.54 12.50 12.68</td></tr><tr><td>X J</td><td>88.49 74.03</td><td>88.45 19.23 13.98</td></tr><tr><td></td><td>88.51 74.35</td><td>88.45 25.8014.59</td></tr></table>

Table 7: The efect of the motion loss to the P-WSVAL objective on UCF-Crime.
<table><tr><td rowspan="2">Model</td><td rowspan="2"> $\lambda _ { 5 }$ </td><td>Temporal</td><td></td><td>Spatial</td></tr><tr><td>AUC  $\mathrm { A U C _ { A } }$ </td><td></td><td>|PAUC PAP TIoU</td></tr><tr><td>w/o. motion loss</td><td>N/A</td><td>87.85 73.12</td><td>87.76</td><td>33.23 12.57</td></tr><tr><td rowspan="3">w. motion loss</td><td>0.01</td><td>88.51 74.35</td><td>88.45</td><td>25.8014.59</td></tr><tr><td>0.001</td><td>88.22 73.09</td><td>88.41</td><td>25.6013.95</td></tr><tr><td>0.0001 88.01</td><td>73.35</td><td>88.17</td><td>17.47 12.33</td></tr></table>

Efect of DTFE and PSA. Table 6 evaluates the individual and joint contributions of the Dynamic Tubelet Feature Encoder (DTFE) and the Patch-Snippet Cross-Attention module (PSA). Removing both yields the weakest spatial performance, indicating that neither dynamic temporal modeling, nor cross-branch interaction alone sufices for reliable localization. Enabling DTFE alone improves TIoU and PAP, suggesting that dynamic tubelet encoding enhances temporal coherence and produces more discriminative spatio-temporal features. Similarly, introducing PSA alone boosts PAP and PAUC, highlighting the importance of explicit patch-snippet interaction. Combining both modules consistently improves temporal and spatial metrics, achieving the best overall performance.

Efect of Motion Loss. Table 7 shows that incorporating the motion loss consistently improves both temporal and spatial performance. By biasing DTFE toward dynamically informative tubelets, it suppresses static background responses and promotes motionconsistent feature learning. This yields stronger temporal discrimination and improved localization overlap (TIoU),

Table 8: Impact of motion regularization on sparsified tubelet selection before P-WSVAL on UCF-Crime. MIoU and PAP are evaluated at diferent retention ratios.
<table><tr><td>Setting</td><td></td><td>Keep ratios MIoU (%) ↑ PAP (%) ↑</td><td></td></tr><tr><td rowspan="3">w/o. motion loss</td><td>0.5</td><td>21.78</td><td>7.10</td></tr><tr><td>0.3</td><td>34.55</td><td>10.47</td></tr><tr><td>0.2</td><td>10.09</td><td>4.96</td></tr><tr><td rowspan="3">w. motion loss</td><td>0.5</td><td>17.37</td><td>8.84</td></tr><tr><td>0.3</td><td>34.70</td><td>14.62</td></tr><tr><td>0.2</td><td>6.61</td><td>5.21</td></tr></table>

at a measurable cost in PAP. The trade-of reflects the pruning behavior analyzed in the shortcut discussion of Table 10: motion regularization prevents the score distribution from concentrating on a small set of high-confidence patches, distributing evidence across genuinely dynamic regions.

To investigate how motion regularization shapes the learned sparsification, which is the core mechanism behind the importance region localization stage, we measure the overlap and precision (MIoU and PAP) of the sparsified tubelet set with respect to ground truth.

Table 9: Efect of motion regularization strategy within SST-WSVADL, evaluated on UCF-Crime.
<table><tr><td>Motion regularization</td><td>PAUC</td><td>PAP</td><td>TIoU</td></tr><tr><td>Variance-based [11]</td><td>89.08</td><td>19.71</td><td>11.03</td></tr><tr><td>Time-reversal (Ours)</td><td>88.45</td><td>25.80</td><td>14.59</td></tr></table>

Specifically, we evaluate these metrics at diferent retention ratios $( { \frac { 1 } { 2 } } , ~ { \frac { 1 } { 3 } }$ , and 1 of the highest-scoring tubelets) prior to the spatial WSVAD module. This way, we assess the quality of the retained regions before the final anomaly scoring. As shown in Table 8, motion regularization consistently increases PAP across all retention levels, while preserving comparable MIoU at the intermediate sparsity level (0.3). Since the number of retained tubelets is fixed, both settings preserve a similar subset of tokens, stabilizing MIoU. The key diference lies in the composition of the retained set: without motion regularization, additional background tubelets are kept, increasing false positives. Motion regularization suppresses these activations, leading to a higher PAP while maintaining comparable overlap. Qualitative analysis is provided in Supp. S5.1.

We examine how efective the time-reversal-based motion signal is by comparing it against variance-based motion regularization [11] in Table 9. Specifically, we substitute our time-reversal signal with MoTAR, which relies on the magnitude of temporal feature change as motion signal. The time reversal regularization instead exploits the directional asymmetry of natural motion, penalizing representations that remain invariant when the temporal order is reversed. Time reversal wins PAP by 6.09 points (relative 31%) and TIoU by 3.56 points (relative 32%). At comparable ranking performance, time-reversal produces substantially better spatial precision and localization.

Pruning Behavior We analyse the learned pruning rates per layer under diferent training settings in Table S7. As observed, under unguided pruning, the model drops too many tokens than necessary to adequately represent anomalous regions. To validate this, we compare the anomaly patch coverage (Table S8) with

t= T1

t= T1

t= T3

t= T2  
![](images/b8c5fdd805dbbb53be4433e5ab6d4fae52960e71f883b6b9adb22d925306be33.jpg)  
Fig. 2: Qualitative results (left to right) on UCF-Crime, MSAD, and XD-Violence. Examples are from Arrest024, Arson018, Explosion022, Assault010 (UCF-Crime); Fire-24, TraficAccident-2, Assault-11, Shooting-4 (MSAD); and v=0qtIjyt-7wg, v=DD3jfKr8ek, Black.Hawk.Down.2001 (XD-Violence). Zoom in for better visualization.

the amount of retained tokens according to the learned pruning rates (Table S7). In UCF-Crime, roughly 25% of anomalous frames are covered by 20 out of 64 patches, suggesting that the final layer should retain at least 30% of tubelets (e.g., 38 of 128 when $t _ { z } = 2 )$ . However, as shown in the first row of Table S7, the model prunes far more aggressively, possibly learning a shortcut rather than identifying meaningful tubelets for the anomaly localization task.

Another shortcut signature is observed when we swap the adopted hard pruning in SST-WSVADL with soft pruning [8] (Table 10). Unlike dropping tubelets discretely with a binary decision, soft pruning scores are turned into continuous mask via a Sigmoid(.)

Table 10: Hard vs. soft pruning under SST-WSVADL and motion loss.
<table><tr><td rowspan="2">Pruning</td><td colspan="2">Temporal</td><td colspan="2">Spatial</td></tr><tr><td>AUC</td><td>AP</td><td>PAUC PAP</td><td>TIoU</td></tr><tr><td>Soft [8]</td><td>88.37</td><td>42.80</td><td>87.65 18.69</td><td>14.48</td></tr><tr><td>Hard [20]</td><td>88.52</td><td>37.05</td><td>88.45 25.80</td><td>14.59</td></tr></table>

function, making them diferentiable. This leaves the patch-level score distribution difuse, where no token is forced to commit either way. The score dispersion afects the per-patch discriminativeness (PAUC and PAP) more than frame-level ranking (AUC and AP). As AP actually improves under soft pruning (37.05 → 42.80) while PAP drops sharply (25.80 → 18.69), the model becomes better at ranking which frames are anomalous without needing to know where the anomaly is within frames. Hard pruning eliminates this route, forcing tokens to commit spatially or be discarded entirely. This trades a small AP cost for substantially sharper patch-level grounding.

Computational Complexity. We report the computational complexity analysis in Supp. S4.2.

## 5.5 Scene-Bias Analysis

To assess how much scene context influences SST-WSVADL scoring versus the S-WSVAD base, we adopt the Bias-AUC audit of [1], computed on normal frames using the authors’ UCF-Crime scene-factor annotations. Bias-AUC equals 0.5 when a model is insensitive to a scene group; larger deviations indicate sceneconditional scoring.

Table 11 reports mean |Bias-AUC− 0.5| averaged across scene factors, split by backbone. Since S-WSVAD reuses UR-DMU [37], it serves as the spatialfree configuration. Under a fixed backbone, adding the sparse spatial branch reduces mean scene bias by 0.013 under both VideoMAEv2 and I3D. The backbone itself has a larger effect than these components. Switching from VideoMAEv2 to I3D reduces

Table 11: Scene Bias-AUC across scene factors [1], split by backbone.
<table><tr><td>Model</td><td>features</td><td>Mean |B − 0.5|(↓)</td></tr><tr><td>S-WSVAD (base) [37]</td><td>VideoMAEv2</td><td>0.113</td></tr><tr><td>SST-WSVADL (ours)</td><td>VideoMAEv2</td><td>0.100</td></tr><tr><td>Universal MIL [22]</td><td>VideoMAEv2</td><td>0.086</td></tr><tr><td>PiVAD [18]</td><td>VideoMAEv2</td><td>0.084</td></tr><tr><td>TEVAD [3]</td><td>I3D</td><td>0.119</td></tr><tr><td>Universal MIL [22]</td><td>I3D</td><td>0.096</td></tr><tr><td>S-WSVAD (base) [37]</td><td>I3D</td><td>0.093</td></tr><tr><td>SST-WSVADL (ours)</td><td>I3D</td><td>0.080</td></tr><tr><td>PiVAD [18]</td><td>I3D</td><td>0.064</td></tr><tr><td>VadCLIP [34]</td><td>CLIP</td><td>0.112</td></tr></table>

mean bias by 0.020 for both UR-DMU and SST-WSVADL. Spatial localization also remains modest in absolute terms, so SST-WSVADL is best understood as a transparent baseline rather than a solution to scene bias. Still, the consistent 0.013 reduction under fixed-backbone comparison shows that scene bias in a WSVAD algorithm is measurably reducible under spatial conditioning.

## 6 Conclusion

In this work, we released a public benchmark for video anomaly localization across three datasets: UCF-Crime, XD-Violence, and MSAD. Coupled with a method-agnostic evaluation protocol, it enables interpretability-oriented assessment of whether anomaly predictions are grounded in authentic event regions rather than background context. We further proposed SST-WSVADL, a sparse spatio-temporal framework that unifies snippet-level temporal reasoning with patch-level localization. Through motion-aware sparsification during training, SST-WSVADL steers feature learning away from background-dominant regions toward authentic anomaly cues. Without relying on detectors, dense annotations, or vision-language models, it delivers competitive performance across all benchmarks, while achieving measurable but modest reductions in scene bias. Together, these contributions support progress toward more transparent and accountable weakly-supervised anomaly detection. In future work, we plan to extend this sparse formulation to multimodal scenarios and explore diferent sparsification strategies and granularity levels.

## References

1. Abdulaziz, S., Bondarev, E.: Auditing frame-level AUC in weakly supervised video anomaly detection: Granularity, resolution, and scene bias. In: Proceedings of the ECCV 2026 Workshop on Empirical Theory (2026)

2. Carreira, J., Zisserman, A.: Quo vadis, action recognition? a new model and the kinetics dataset. In: proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. pp. 6299–6308 (2017)

3. Chen, W., Ma, K.T., Yew, Z.J., Hur, M., Khoo, D.A.A.: Tevad: Improved video anomaly detection with captions. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 5549–5559 (2023)

4. Chen, Y., Liu, Z., Zhang, B., Fok, W., Qi, X., Wu, Y.C.: Mgfn: Magnitudecontrastive glance-and-focus network for weakly-supervised video anomaly detection. In: Proceedings of the AAAI conference on artificial intelligence. vol. 37, pp. 387–395 (2023)

5. Cui, X., Liu, Q., Gao, M., Metaxas, D.N.: Abnormal detection using interaction energy potentials. In: CVPR 2011. pp. 3161–3167. IEEE (2011)

6. Hara, K., Kataoka, H., Satoh, Y.: Can spatiotemporal 3d cnns retrace the history of 2d cnns and imagenet? In: Proceedings of the IEEE conference on Computer Vision and Pattern Recognition. pp. 6546–6555 (2018)

7. Hospedales, T., Gong, S., Xiang, T.: A markov clustering topic model for mining behaviour in video. In: 2009 IEEE 12th International Conference on Computer Vision. pp. 1165–1172. IEEE (2009)

8. Kong, Z., Dong, P., Ma, X., Meng, X., Niu, W., Sun, M., Shen, X., Yuan, G., Ren, B., Tang, H., et al.: Spvit: Enabling faster vision transformers via latency-aware soft token pruning. In: European conference on computer vision. pp. 620–640. Springer (2022)

9. Kratz, L., Nishino, K.: Anomaly detection in extremely crowded scenes using spatio-temporal motion pattern models. In: 2009 IEEE conference on computer vision and pattern recognition. pp. 1446–1453. IEEE (2009)

10. Landi, F., Snoek, C.G., Cucchiara, R.: Anomaly locality in video surveillance. arXiv preprint arXiv:1901.10364 (2019)

11. Lee, J., Bang, C., Kim, M., Cho, M.: Refinevad: Semantic-guided feature recalibration for weakly supervised video anomaly detection. In: Proceedings of the AAAI Conference on Artificial Intelligence. vol. 40, pp. 5899–5907 (2026)

12. Li, G., Cai, G., Zeng, X., Zhao, R.: Scale-aware spatio-temporal relation learning for video anomaly detection. In: European Conference on Computer Vision. pp. 333–350. Springer (2022)

13. Li, M., Xu, X., Fan, H., Zhou, P., Liu, J., Liu, J.W., Li, J., Keppo, J., Shou, M.Z., Yan, S.: Stprivacy: Spatio-temporal privacy-preserving action recognition. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 5106–5115 (2023)

14. Li, W., Mahadevan, V., Vasconcelos, N.: Anomaly detection and localization in crowded scenes. IEEE transactions on pattern analysis and machine intelligence 36(1), 18–32 (2013)

15. Liu, K., Liu, W., Gan, C., Tan, M., Ma, H.: T-c3d: Temporal convolutional 3d network for real-time action recognition. In: Proceedings of the AAAI conference on artificial intelligence. vol. 32 (2018)

16. Liu, K., Ma, H.: Exploring background-bias for anomaly detection in surveillance videos. In: Proceedings of the 27th ACM International Conference on Multimedia. pp. 1490–1499 (2019)

17. Lu, C., Shi, J., Jia, J.: Abnormal event detection at 150 fps in matlab. In: Proceedings of the IEEE international conference on computer vision. pp. 2720–2727 (2013)

18. Majhi, S., D’Amicantonio, G., Dantcheva, A., Kong, Q., Garattoni, L., Francesca, G., Bondarev, E., Brémond, F.: Just dance with pi! a poly-modal inductor for weakly-supervised video anomaly detection. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 24265–24274 (2025)

19. Mumford, D., Fogarty, J., Kirwan, F.: Geometric invariant theory, vol. 34. Springer Science & Business Media (1994)

20. Rao, Y., Zhao, W., Liu, B., Lu, J., Zhou, J., Hsieh, C.J.: Dynamicvit: Eficient vision transformers with dynamic token sparsification. Advances in neural information processing systems 34, 13937–13949 (2021)

21. Simonyan, K., Zisserman, A.: Two-stream convolutional networks for action recognition in videos. Advances in neural information processing systems 27 (2014)

22. Sultani, W., Chen, C., Shah, M.: Real-world anomaly detection in surveillance videos. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 6479–6488 (2018)

23. Sun, S., Gong, X.: Long-short temporal co-teaching for weakly supervised video anomaly detection. In: 2023 IEEE International Conference on Multimedia and Expo (ICME). pp. 2711–2716. IEEE (2023)

24. Tian, Y., Pang, G., Chen, Y., Singh, R., Verjans, J.W., Carneiro, G.: Weaklysupervised video anomaly detection with robust temporal feature magnitude learning. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 4975–4986 (2021)

25. Tran, D., Bourdev, L., Fergus, R., Torresani, L., Paluri, M.: Learning spatiotemporal features with 3d convolutional networks. In: Proceedings of the IEEE international conference on computer vision. pp. 4489–4497 (2015)

26. Wang, J., Jiao, J., Bao, L., He, S., Liu, Y., Liu, W.: Self-supervised spatio-temporal representation learning for videos by predicting motion and appearance statistics. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 4006–4015 (2019)

27. Wang, L., Huang, B., Zhao, Z., Tong, Z., He, Y., Wang, Y., Wang, Y., Qiao, Y.: Videomae v2: Scaling video masked autoencoders with dual masking (2023)

28. Wang, L., Li, W., Li, W., Van Gool, L.: Appearance-and-relation networks for video classification. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 1430–1439 (2018)

29. Wang, L., Tong, Z., Ji, B., Wu, G.: Tdn: Temporal diference networks for eficient action recognition. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 1895–1904 (2021)

30. Wang, L., Xiong, Y., Wang, Z., Qiao, Y., Lin, D., Tang, X., Van Gool, L.: Temporal segment networks: Towards good practices for deep action recognition. In: European conference on computer vision. pp. 20–36. Springer (2016)

31. Wang, X., Girshick, R., Gupta, A., He, K.: Non-local neural networks. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 7794–7803 (2018)

32. Wu, J., Zhang, W., Li, G., Wu, W., Tan, X., Li, Y., Ding, E., Lin, L.: Weaklysupervised spatio-temporal anomaly detection in surveillance video. arXiv preprint arXiv:2108.03825 (2021)

33. Wu, P., Zhou, X., Pang, G., Yang, Z., Yan, Q., Wang, P., Zhang, Y.: Weakly supervised video anomaly detection and localization with spatio-temporal prompts.

In: Proceedings of the 32nd ACM International Conference on Multimedia. pp. 9301–9310 (2024)

34. Wu, P., Zhou, X., Pang, G., Zhou, L., Yan, Q., Wang, P., Zhang, Y.: Vadclip: Adapting vision-language models for weakly supervised video anomaly detection. In: Proceedings of the AAAI Conference on Artificial Intelligence. vol. 38, pp. 6074–6082 (2024)

35. Yun, S., Masukawa, R., Na, M., Imani, M.: Missiongnn: Hierarchical multimodal gnn-based weakly supervised video anomaly recognition with mission-specific knowledge graph generation. In: 2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV). pp. 4736–4745. IEEE (2025)

36. Zhao, B., Fei-Fei, L., Xing, E.P.: Online detection of unusual events in videos via dynamic sparse coding. In: CVPR 2011. pp. 3313–3320. IEEE (2011)

37. Zhou, H., Yu, J., Yang, W.: Dual memory units with uncertainty regulation for weakly supervised video anomaly detection. In: Proceedings of the AAAI Conference on Artificial Intelligence. vol. 37, pp. 3769–3777 (2023)

38. Zhu, Y., Nayak, N.M., Roy-Chowdhury, A.K.: Context-aware activity recognition and anomaly detection in video. IEEE Journal of Selected Topics in Signal Processing 7(1), 91–101 (2012)