# GM-GReFEL: Geometry-Guided Spatiotemporal State-Space Learning for Robust Dynamic Facial Expression Recognition

Yudhistira Arditya Pratama and Yi-Zeng Hsieh, Senior Member, IEEE

Abstract—Recognizing facial expressions in unconstrained video suffers from two coupled problems: self-attention grows quadratically in cost with clip length, and crowdsourced labels are noisy. We present GM-GReFEL, a spatiotemporal statespace network with a geometry-aware decision stage. Its Modulated GroupMamba backbone replaces a spatiotemporal Vision Transformer. Six channel groups are scanned separately, four over the frame plane and two over time, so the cost grows linearly with sequence length (O(N)); after each layer, a Channel Affinity Modulation gate mixes the groups again. On top of the backbone, a GReFEL reliability module turns the normalized entropy of each prediction into a mixing weight between the classifier output and a vote from trainable geometric anchors, so that facial geometry counts for more when the network is unsure. The model is initialized by static-to-dynamic weight transfer and trained with a Triple Loss. In five-fold cross-validation, the best fold scores 66.57%/76.46% UAR/WAR on DFEW, 0.3 points or less below the strongest compared method, and 45.66%/62.27% on MAFW, ahead of all compared methods; on FERV39k the mean over five splits, 42.30%/52.66%, is competitive. The fivefold means are 64.07%/74.54% on DFEW, on par with recent large pre-trained models, and 38.90%/54.37% on MAFW, below the best of them. A paired t-test on DFEW indicates a modest benefit of temporal modeling (+1.61 UAR points, one-sided p = 0.031). Attention stays on plausible facial regions, and latency is close to a ViT-B/16 baseline (47.4 vs. 46.8 ms per clip). Code: https://github.com/ardiytama/GroupMamba-DFER.

Index Terms—Video-based emotion recognition, selective statespace model, GroupMamba, landmark-based reliability, entropy calibration, cross-validation, affective video analysis.

## I. INTRODUCTION

UMAN emotion unfolds over time: a spontaneous smile builds from a neutral baseline, peaks, and relaxes, while a flash of disgust may last only a few frames. Static facial expression recognition (SFER) on cropped single frames [16] discards this temporal structure, which is why dynamic facial expression recognition (DFER) analyzes continuous video instead. Large in-the-wild benchmarks such as FERV39k [1], DFEW [2], and MAFW [3] have made the task considerably harder by adding unconstrained illumination, camera motion, and occlusion.

Two obstacles remain largely unresolved. The first is computational. Spatiotemporal Vision Transformers [4], [5] link distant frames directly, but self-attention scales as $\mathcal { O } ( N ^ { 2 } )$ in the number of tokens N, forcing a trade-off between clip length, spatial resolution, and GPU memory. Selective state-space models such as Mamba [6] match the long-range capacity of attention at O(N) cost and have been adapted to images [7], [8] and video [9], but not around the geometric priors specific to facial expression. The second obstacle is label noise. Crowdsourced annotators frequently disagree on ambiguous or transitional expressions [11], [12], and a network trained with plain cross-entropy absorbs this noise, producing the well-documented bias toward majority classes such as “Happy” and “Neutral” [26], [27].

We argue that the two problems should be solved jointly: an efficient backbone without a noise-robust decision layer still overfits corrupted labels, while a robust decision layer on a quadratic backbone still cannot observe enough of the clip. GM-GReFEL couples a linear-complexity spatiotemporal state-space engine with a reliability mechanism grounded in facial geometry. Our contributions are as follows.

• Modulated GroupMamba engine. Each layer partitions the channels into six groups, scans four of them across the frame and two across time with a Visual Single Selective Scan, and merges them again through a Channel Affinity Modulation gate. Texture and motion thus get separate pathways while the cost of the extractor stays linear in N.

• Entropy-guided geometric reliability. We carry GReFEL [13] over from images to video. How uncertain a prediction is, measured by its normalized entropy, decides per sample how much the output relies on the classifier and how much on a cosine-similarity vote against trainable class anchors; under occlusion, blur, or ambiguity the anchors dominate.

• Static-to-Dynamic transfer and Triple Loss. Spatial weights and anchors learned on still images are inflated into the video model, the temporal parameters start close to zero, and two extra loss terms (anchor orthogonality and center loss) are meant to stop the anchors from collapsing.

• Comprehensive and transparent evaluation. On its best fold, GM-GReFEL gives the highest UAR and WAR among all compared methods on MAFW and comes within 0.3 points of the strongest method, S4D, on DFEW. We report the five-fold mean next to every bestfold figure: it is comparable to recent pre-trained models on DFEW and lower on MAFW, and the FERV39k result is competitive. The evaluation is supported by a paired t-test on DFEW, component and hyperparameter ablations, Grad-CAM and t-SNE analyses, and a compute comparison against a ViT-B/16 baseline.

## II. RELATED WORK

## A. From Image-Based to Video-Based Expression Analysis

Early DFER systems extracted frame-level features with a 2D CNN and modeled temporal order with a recurrent network, an approach that tracks short-term changes but degrades over long clips because of vanishing gradients. 3D convolutional backbones such as C3D [18], P3D [19], I3D [20], and R(2+1)D [21] compute motion within a single tensor operation. Their fixed receptive field, however, captures rapid local motion more readily than slowly forming expressions, and enlarging the kernels raises the parameter count and the risk of overfitting imbalanced in-the-wild collections like DFEW [2], FERV39k [1], and MAFW [3].

## B. Face-Detection-Free DFER Paradigms

Most DFER pipelines, including ours, assume a face detection and alignment (FDA) stage that crops the face before temporal modeling. PTH-Net [46] questions this assumption by recognizing expressions directly from raw video: a frozen video backbone extracts a temporal feature pyramid at multiple frequencies, and a Temporal Hierarchy Refinement architecture with a Scalable Semantic Distinction layer separates target from non-target frames, preserving body-language cues that cropping discards. Because GM-GReFEL’s geometric anchors rely on a stable, aligned facial region, we see the FDA-free paradigm as complementary to ours, and extending the reliability mechanism to unaligned frames is a separate research direction.

## C. Transformer-Based Spatiotemporal Modeling

ViViT [4] and MViT [5] brought global self-attention to video, and Former-DFER [23] and STT [24] applied it to facial video. Later work addressed label noise (NR-DFERNet [26]), majority-class collapse (IAL [27]), and multi-instance learning (M3DFEL [28]); CLIP-based methods (DFER-CLIP [32], EmoCLIP [33], A<sup>3</sup>lign-DFER [34]) inject vision-language priors, and masked-autoencoder methods (MAE-DFER [37], VideoMAE [17], HiCMAE [36], SVFAP [35]) pursue labelefficient pre-training. Static-to-dynamic methods such as S2D [38] and S4D [39] transfer image-level facial priors to video, as our initialization does. All of these retain a self-attention core whose $\mathcal { O } ( N ^ { 2 } )$ cost limits the resolution and clip length they can process.

## D. Selective State-Space Backbones in Vision

Structured state space models propagate an input through a latent state, but early deep-learning variants used fixed transition parameters and could not filter tokens by content. Mamba [6] made the step size and projections input-dependent, allowing a linear-time recurrence to discard uninformative tokens while approximating the long-range reasoning of attention. VMamba [7] and Vision Mamba [8] adapted this recurrence to 2D images through cross-scan and bidirectional scanning, VideoMamba [9] extended it to video, and GroupMamba [10], on which we build, partitions channels into groups with distinct scan directions re-coupled by a channel modulation gate. To our knowledge, no prior work combines grouped linearcomplexity scanning with an explicit geometric reliability mechanism for facial expression recognition.

## E. Label Noise and Geometric Reliability

In-the-wild DFER labels come from crowdsourced annotation, and annotators disagree on subtle or occluded expressions [11], [12]. Networks trained on such labels first learn generalizable patterns and later memorize incorrect labels [11]. Statistical remedies redistribute probability across plausible classes or spread attention across facial regions [12], while center-loss [14] and margin-based objectives [15] tighten class clusters; however, all of them correct ambiguity using the same pixel evidence that produced it. GReFEL [13] instead introduces an anatomically grounded anchor that is independent of image quality, validating predictions against geometric faciallandmark statistics. We extend this idea from static images to video, replacing a fixed weighting with a per-sample, entropybased mixing coefficient.

## F. Summary and Positioning

Efficient sequence modeling and geometric reliability have so far progressed separately: linear-complexity backbones lack an explicit reliability check, and GReFEL has been formulated only for static images. GM-GReFEL closes this joint gap, using Mamba-style scanning to remove the quadratic cost and letting the network’s own predictive entropy decide how much weight the geometric evidence receives.

## III. PROPOSED METHOD

## A. Overview

The overall GM-GReFEL pipeline is illustrated in Fig. 1. A raw video clip is first temporally resampled and spatially normalized (Section III-B), then partitioned into 3D spatiotemporal tubelets and linearly projected into a token sequence (Section III-C). This sequence is processed by a stack of Modulated GroupMamba layers, in which the channel dimension is split into six groups that are scanned along four spatial directions and two temporal directions using a Visual Single Selective Scan, then re-fused by a Channel Affinity Modulation gate (Section III-D). The resulting dense motion-aware feature vector is passed to the Geometry-Aware Reliability module, which computes the network’s own predictive entropy and blends the primary classification with a geometrically grounded vote accordingly (Section III-E). All parts are optimized jointly under a Triple Loss objective (Section III-E) after a Static-to-Dynamic weight-transfer initialization phase (Section III-F). Section III-G formalizes the resulting computational complexity.

![](images/55b24d5b47c15b88f4e4b59532440568e862572a30169d43d509ffa390ca3a97.jpg)  
Fig. 1. Structure of the GM-GReFEL pipeline. Top row: the four-stage forward pass, (a) 3D spatiotemporal tubelet tokenization with positional encoding, (b) sequence modeling via the Modulated GroupMamba engine with Channel Affinity Modulation (CAM), (c) geometry modeling via the Temporal Geometry-Aware module that combines the primary MLP’s confidence score with the geometric anchor bank, and (d) Triple Loss optimization, together with example output visualizations (facial landmarks, Grad-CAM attention). Bottom row: internal detail of each stage. Bottom left: the Modulated GroupMamba layer, in which the Spatial Group and Temporal Group each pass through a stack of VSSS blocks before Channel Affinity and Channel Modulation gating produce the $B \times 1 \times C$ layer output $Z ^ { ( o u t ) }$ . Bottom center: the Geometry-Aware Reliability mechanism, showing the Statistical Track (Process, Softmax, Normalized Entropy) computed in parallel with the Geometric Track (a similarity lookup against the geometric anchor database via Softmax), combined before Loss Optimization. Bottom right: the six VSSS scanning directions — four boustrophedon spatial-group sweeps (Direction 1–4) and two temporal-group directions (Forward, Backward).

## B. Temporal Sampling and Spatial Preprocessing

Two preprocessing choices, frame selection and normalization, measurably affect performance (Section IV-B1) and are specified here.

1) Jittered Temporal Sampling: Clips vary in length (e.g., 45 versus 120 frames), whereas tubelet extraction (Section III-C) requires $T = 1 6$ frames. At test time we split each clip into T equal segments and take each segment’s center frame, which is fully reproducible. Using the same frames at every training epoch, however, would let the temporal groups (Section III-D) memorize fixed frame indices, so during training we apply random temporal jittering: with $L _ { t o t a l }$ available frames, the index selected for step $t \in [ 0 , T - 1 ]$ is

$$
I _ { t } = \left\lfloor \frac { L _ { t o t a l } } { T } \times t + \delta \right\rfloor ,\tag{1}
$$

where δ is drawn per clip and epoch from a discrete uniform distribution over $[ 0 , \lfloor { L _ { t o t a l } } / { T } \rfloor - 1 ]$ . This perturbs the frame offset within each segment while preserving temporal order, encouraging the temporal transition matrices ${ \bar { A } } _ { 5 } , { \bar { A } } _ { 6 }$ of Eq. (15) to learn motion patterns robust to small sampling shifts; disabling it is the “w/o Jittered Sampling” setting in Table I.

2) Spatial Alignment and Normalization: Before resizing frames to $2 2 4 \times 2 2 4$ , we detect the face in the first frame and apply the same crop to all T frames, since re-detecting per frame introduces crop jitter that the temporal groups cannot distinguish from real facial motion. Pixel intensities are then standardized per channel,

$$
v _ { c , h , w } ^ { \prime } = \frac { v _ { c , h , w } - \mu _ { c } } { \sigma _ { c } } , \qquad c \in \{ R , G , B \} ,\tag{2}
$$

where $\mu _ { c }$ and $\sigma _ { c }$ are computed once over the training split; feeding raw 0–255 intensities into the tubelet projection E of Eq. (7) produced unstable initial gradients.

## C. Spatiotemporal Tubelet Tokenization

A video clip is represented as a tensor $\nu \in \mathbb { R } ^ { T \times C \times H \times W }$ with T sampled frames of C color channels and spatial size $H \times W$ . For fair comparison against the datasets used in this study, we resize each frame to 224 × 224 and sample $T = 1 6$ frames per clip with $C = 3 { \mathrm { ; } }$ , giving $\nu \in \mathbb { R } ^ { 1 6 \times 3 \times 2 2 \bar { 4 } \times 2 2 4 }$

Tokenizing frame by frame would multiply the sequence length and bring back the memory pressure we aim to avoid, so we extract 3D tubelets of size $P _ { t } \times P _ { h } \times P _ { w }$ , with $P _ { t } = 2 $ $P _ { h } = 4 , P _ { w } = 4$ in our implementation. This yields

$$
N _ { t } = T / P _ { t } = 1 6 / 2 = 8 ,\tag{3}
$$

$$
N _ { h } = H / P _ { h } = 2 2 4 / 4 = 5 6 ,\tag{4}
$$

$$
N _ { w } = W / P _ { w } = 2 2 4 / 4 = 5 6 ,\tag{5}
$$

and a total token count

$$
N = N _ { t } \times N _ { h } \times N _ { w } = 8 \times 5 6 \times 5 6 = 2 5 { , } 0 8 8 .\tag{6}
$$

Setting $P _ { t } ~ = ~ 2$ halves the temporal sequence length at the very first layer, which is the primary lever we use to keep the token count, and therefore the downstream memory footprint, tractable for a 16-frame clip. Each flattened tubelet has dimensionality $\operatorname { D i m } ( X _ { i } ) = P _ { t } \times P _ { h } \times P _ { w } \times C = 2 \times 4 \times 4 \times 3 = 9 6 ,$ and is projected through a learned matrix $E \in \mathbb { R } ^ { 9 6 \times D }$ (with $D ~ = ~ 9 6$ at the input stage) and summed with a trainable positional encoding $E _ { p o s } \in \mathbb { R } ^ { N \times D }$

$$
\begin{array} { r } { Z ^ { ( 0 ) } = [ x _ { 1 } E , x _ { 2 } E , \ldots , x _ { N } E ] + E _ { p o s } , \quad Z ^ { ( 0 ) } \in \mathbb { R } ^ { 2 5 , 0 8 8 \times 9 6 } . } \end{array}\tag{7}
$$

## D. Modulated GroupMamba Engine

Feeding the N = 25,088-token sequence $Z ^ { ( 0 ) }$ into a selfattention layer would require computing an $N \times N$ correlation matrix, an $\mathcal { O } ( N ^ { 2 } )$ operation that is impractical at this scale for standard GPU memory budgets. We instead replace the attention core entirely with a Modulated GroupMamba engine built on a linear, discretized state-space recurrence.

1) Orthogonal Channel Partitioning: The channel dimension $D = 9 6$ is evenly divided into $G = 6$ groups,

$$
\begin{array} { r } { Z ^ { ( 0 ) } = [ Z _ { 1 } , Z _ { 2 } , Z _ { 3 } , Z _ { 4 } , Z _ { 5 } , Z _ { 6 } ] , \qquad Z _ { g } \in \mathbb { R } ^ { 2 5 , 0 8 8 \times 1 6 } . } \end{array}\tag{8}
$$

The first four groups $( g ~ \in ~ \{ 1 , 2 , 3 , 4 \} )$ are dedicated to spatial texture analysis and are scanned across the 2D frame plane along four boustrophedon (serpentine) sweep patterns, illustrated as Direction 1–4 in the bottom-right panel of Fig. 1, which lets the model capture static structural cues such as wrinkle depth or jawline shape while traversing the full spatial extent of each frame. The remaining two groups $( g \in \{ 5 , 6 \} )$ are permuted so that scanning proceeds exclusively along the temporal axis: Group 5 executes a forward temporal scan and Group 6 a backward temporal scan (also shown in the bottomright panel of Fig. 1), so that these pathways track the velocity of a muscle contraction from onset to apex.

2) Discretized Selective State-Space Recurrence: All six sub-tensors $Z _ { g }$ are processed in parallel by VSSS blocks (Fig. 2). Each block starts from the linear time-invariant system

$$
h ^ { \prime } ( t ) = A h ( t ) + B x ( t ) ,\tag{9}
$$

$$
y ( t ) = C h ( t ) ,\tag{10}
$$

with hidden-state dimension $N _ { s t a t e } = 1 6 $ . Because our tokens are discrete, we discretize Eq. $( 9 ) ‐ ( 1 0 )$ via zero-order hold, and, following the Mamba selectivity mechanism, we make the step size $\Delta _ { t }$ and projection matrices $B _ { t } , C _ { t }$ input-dependent linear functions of the current token $x _ { t } \colon$

$$
\Delta _ { t } = \mathrm { S o f t p l u s } ( \mathrm { L i n e a r } _ { \Delta } ( x _ { t } ) ) ,\tag{11}
$$

$$
B _ { t } = \mathrm { L i n e a r } _ { B } ( x _ { t } ) , \qquad C _ { t } = \mathrm { L i n e a r } _ { C } ( x _ { t } ) ,\tag{12}
$$

$$
\bar { A } _ { t } = \exp ( \Delta _ { t } A ) ,\tag{13}
$$

$$
\bar { B } _ { t } = ( \Delta _ { t } A ) ^ { - 1 } \left( \exp ( \Delta _ { t } A ) - I \right) \cdot \Delta _ { t } B _ { t } ,\tag{14}
$$

$$
h _ { t } ^ { ( g ) } = \bar { A } _ { t } ^ { ( g ) } h _ { t - 1 } ^ { ( g ) } + \bar { B } _ { t } ^ { ( g ) } x _ { t } ^ { ( g ) } .\tag{15}
$$

Intuitively, $\Delta _ { t }$ acts as an input-dependent forgetting gate: when the current token carries irrelevant information (e.g., a sudden lighting change or an eye blink), the network drives $\Delta _ { t }$ toward zero, ${ \bar { A } } _ { t }$ approaches the identity matrix, and the hidden state is left largely unchanged, preventing noise from corrupting the accumulated representation of the actual expression.

![](images/2f7c64768a36c9c5fcd0c420ab0ea3fb222629562ecf70c254acd343bb318a1c.jpg)  
Fig. 2. VSSS block: a Mamba-style selective scan core and a subsequent feed-forward layer.

3) Channel Affinity Modulation: Scanning the six groups independently is what keeps the engine linear, but it also leaves the spatial and temporal pathways functionally isolated from one another after each layer. We restore cross-group communication with a Channel Affinity Modulation (CAM) gate, illustrated in the bottom-left panel of Fig. 1. The six outputs are first concatenated back to the original channel width,

$$
Y _ { c o n c a t } = { \mathrm { C o n c a t } } ( y _ { t , 1 } , \dots , y _ { t , 6 } ) \in \mathbb { R } ^ { 2 5 , 0 8 8 \times 9 6 } ,\tag{16}
$$

pooled over all tokens, and mapped by a two-layer MLP to a per-channel weight vector $w _ { c } \in \mathbb { R } ^ { 9 6 }$

$$
w _ { c } = \sigma \left( \mathcal { W } _ { 2 } \delta \left( \mathcal { W } _ { 1 } \left( \mathrm { G A P } ( Y _ { c o n c a t } ) \right) \right) \right) ,\tag{17}
$$

with $\sigma$ denoting a sigmoid and δ a SiLU. The final layer output is obtained by gating $Y _ { c o n c a t }$ with $w _ { c }$ and a residual SiLUactivated path from the input projection $Z _ { g a t e }$

$$
Z ^ { ( o u t ) } = ( Y _ { c o n c a t } \odot w _ { c } ) \odot \operatorname { S i L U } ( Z _ { g a t e } ) .\tag{18}
$$

After L stacked GroupMamba layers $( L \ = \ 3$ by default), temporal pooling collapses the sequence into a single dense feature vector $v _ { e } \in \mathbb { R } ^ { 7 6 8 }$ that summarizes the spatiotemporal signature of the clip.

## E. Geometry-Aware Reliability Balancing

Relying purely on $v _ { e }$ for classification is risky because inthe-wild datasets are both class-imbalanced and label-noisy, and a network trained by ordinary cross-entropy against such labels tends to overfit the majority classes. We route $v _ { e }$ through a Geometry-Aware Reliable Facial Expression Learning (GReFEL) module that supplies an independent, anatomically grounded check on the primary prediction, structured as two parallel tracks shown in the bottom-center panel of Fig. 1: a Statistical Track built around the primary MLP classifier, and a Geometric Track that looks up similarity against a database of geometric anchors.

1) Normalized Entropy as a Confidence Signal: The primary probability distribution over the K emotion classes is obtained via an MLP and softmax,

$$
p _ { m } = \mathrm { S o f t m a x } ( \mathrm { M L P } ( v _ { e } ) ) \in \mathbb { R } ^ { K } ,\tag{19}
$$

and its uncertainty is quantified by the normalized Shannon entropy

$$
H ( p _ { m } ) = - { \frac { 1 } { \log ( K ) } } \sum _ { k = 1 } ^ { K } p _ { m , k } \log ( p _ { m , k } ) ,\tag{20}
$$

which is converted into a confidence score $C _ { f } = 1 - H ( p _ { m } ) \in$ [0, 1].

2) Geometric Anchor Matching: In parallel, the module maintains a set of trainable geometric anchors $a _ { k } \in \mathbb { R } ^ { 7 6 8 }$ , one per emotion class, that act as idealized structural prototypes. For instance, the anchor for “Surprise” encodes the characteristic spatial signature of widened eyes and an open mouth. The cosine similarity between $v _ { e }$ and every anchor,

$$
S _ { k } = { \frac { v _ { e } \cdot a _ { k } } { \| v _ { e } \| \| a _ { k } \| } } ,\tag{21}
$$

is converted into a pseudo-target distribution via a temperaturescaled softmax with scale τ:

$$
t _ { g } = \mathrm { S o f t m a x } ( S / \tau ) \in \mathbb { R } ^ { K } .\tag{22}
$$

3) Entropy-Weighted Calibration: The final prediction blends the two signals in proportion to the network’s own confidence:

$$
p _ { f i n a l } = C _ { f } \cdot p _ { m } + ( 1 - C _ { f } ) \cdot t _ { g } .\tag{23}
$$

When the primary Mamba engine is confident (low entropy, $C _ { f } \to 1 )$ , the decision is driven almost entirely by $p _ { m } ;$ when the input is ambiguous or corrupted (high entropy, $C _ { f } \to 0 )$ the decision shifts toward the geometric anchors $t _ { g } ,$ so the network falls back on anatomy instead of guessing the majority class.

4) Triple Loss Formulation: Training is guided by three complementary objectives. The classification loss is standard cross-entropy against the calibrated prediction,

$$
\mathcal { L } _ { c l s } = - \sum _ { k = 1 } ^ { K } y _ { t r u e , k } \log ( p _ { f i n a l , k } ) .\tag{24}
$$

To prevent the anchors themselves from collapsing toward one another, which would defeat the purpose of the reliability check, an orthogonality constraint is applied to the anchor matrix A (whose rows are the individual anchors $a _ { k } )$

$$
\begin{array} { r } { \mathcal { L } _ { a n c h o r } = \| A A ^ { T } - I \| _ { F } ^ { 2 } . } \end{array}\tag{25}
$$

Finally, a center-loss term pulls each extracted feature toward its ground-truth anchor, minimizing intra-class variance in the latent space:

$$
\mathcal { L } _ { c e n t e r } = \frac { 1 } { 2 } \| v _ { e } - a _ { y _ { t r u e } } \| _ { 2 } ^ { 2 } .\tag{26}
$$

The full objective is a weighted sum,

$$
\mathcal { L } _ { t o t a l } = \mathcal { L } _ { c l s } + \lambda _ { 1 } \mathcal { L } _ { a n c h o r } + \lambda _ { 2 } \mathcal { L } _ { c e n t e r } ,\tag{27}
$$

with $\lambda _ { 1 } , \lambda _ { 2 }$ tuned so that the geometric terms regularize the primary classification signal without overwhelming it.

## F. Static-to-Dynamic (S2D) Knowledge Transfer

Video DFER datasets are small relative to static face databases such as AffectNet and RAF-DB, and training the full spatiotemporal architecture from scratch on tens of thousands of clips invites memorization. Masked video pre-training studies also report that uninitialized spatiotemporal models suffer unstable, high-variance early gradients, whereas models inheriting spatial priors converge stably [17]. Since facial geometry is the same in a photograph and a video frame, we pre-train a 2D variant on static data and transfer it through a decoupled inflation scheme. The 2D patch-embedding weights $W _ { 2 D } \stackrel { * } { \in } \mathbb { R } ^ { C \times P _ { h } \times P _ { w } \times D }$ are inflated to the 3D tubelet projection $W _ { 3 D }$ by replicating across the temporal stride and normalizing by $P _ { t }$

$$
W _ { 3 D } ( t , c , h , w ) = \frac { 1 } { P _ { t } } W _ { 2 D } ( c , h , w ) , \quad \forall t \in [ 1 , P _ { t } ] .\tag{28}
$$

The four spatial-scanning groups inherit their state-transition parameters from the converged 2D model,

$$
\theta _ { 3 D } ^ { ( g ) } = \theta _ { 2 D } ^ { ( g ) } , \qquad g \in \{ 1 , 2 , 3 , 4 \} ,\tag{29}
$$

while the two temporal groups, which have no 2D analogue, are initialized near zero, $A _ { t e m p o r a l } \sim \mathcal { N } ( 0 , \sigma ^ { 2 } )$ with $\sigma ^ { \bar { 2 } }$ ≈ $1 0 ^ { - 4 }$ . As a result, the network starts as a purely spatial model and learns to weight motion cues gradually, which is intended to keep early training stable. The anchors converged during 2D pre-training are imported directly, $A _ { d y n a m i c } ^ { ( i n i t i a l ) } = \overline { { A _ { s t a t i c } } } ,$ so the reliability module starts from informative anchors in the first dynamic iteration.

Optimization has two stages: a warm-up in which the learning rate of inherited spatial parameters and anchors is damped by $\gamma _ { f r e e z e } = 0 . 0 1$ while the temporal matrices train freely, followed by end-to-end fine-tuning.

## G. Computational Complexity and Inference Behavior

The complexity of a single Modulated GroupMamba block is

$$
\mathcal { O } _ { \mathrm { G r o u p M a m b a } } = 3 N D + G \cdot N \cdot D _ { g } \cdot N _ { s t a t e } ,\tag{30}
$$

and since $D _ { g } = D / G$ , this reduces to

$$
\mathcal { O } _ { \mathrm { G r o u p M a m b a } } \approx \mathcal { O } ( N \cdot D \cdot N _ { s t a t e } ) ,\tag{31}
$$

strictly linear in the token count N. Doubling the clip length from 16 to 32 frames therefore doubles the compute cost instead of quadrupling it, which allows longer sequences to be processed without the memory wall that limits self-attention baselines.

At inference time, ground-truth labels are unavailable, so $\mathcal { L } _ { a n c h o r }$ and $\mathcal { L } _ { c e n t e r }$ play no further role and the anchor matrix A is frozen. The forward pass tokenizes the clip, extracts $v _ { e }$ through the frozen GroupMamba stack, computes $p _ { m }$ and its entropy $H ( p _ { m } )$ , projects $v _ { e }$ against the frozen anchors to obtain $t _ { g } .$ , and applies Eq. (23) to produce $p _ { f i n a l } ;$ the arg max of $p _ { f i n a l }$ is the predicted label. Because this calibration relies only on the network’s own internal entropy and the frozen anchor bank, it requires no external supervision and adds negligible overhead, which we quantify directly in Section IV-E.

## H. Evaluation Metrics

Because DFER datasets are heavily class-imbalanced, overall accuracy is misleading: a model that always predicts “Happy” can score deceptively well. We therefore report both Weighted Average Recall (WAR), which favors majority classes by construction,

$$
\mathrm { W A R } = \frac { \sum _ { k = 1 } ^ { K } T _ { k } } { \sum _ { k = 1 } ^ { K } N _ { k } } ,\tag{32}
$$

and Unweighted Average Recall (UAR), which averages perclass recall equally regardless of sample count and is the more informative metric for whether the reliability module is actually protecting minority classes,

$$
\mathrm { U A R } = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \frac { T _ { k } } { N _ { k } } ,\tag{33}
$$

where class k has $N _ { k }$ test samples, of which $T _ { k }$ are classified correctly.

## IV. EXPERIMENTS

## A. Setup

1) Datasets: Three in-the-wild benchmarks are used. DFEW [2] contains over 16,000 clips taken from films, with severe illumination variation and occlusion. FERV39k [1] is the largest scene-diverse benchmark currently available and is strongly skewed toward “Happy” and “Neutral”. MAFW [3] has 11 compound-emotion classes, including anxiety, helplessness, and disappointment, which overlap in facial geometry with the seven basic emotions. All results are obtained under strict 5-fold cross-validation, with every video functioning as held-out test data in exactly one fold. For DFEW and MAFW we report two numbers. The first is the best of the five crossvalidation folds, which shows what the model can reach. The second, the five-fold mean with its standard deviation, shows how stable that result is. As far as we are aware, the baseline figures in our tables are reported as five-fold averages, so the five-fold mean is the like-for-like comparison and the best fold is an upper bound; the comparison tables therefore show both. For FERV39k, the reported figure is the mean over the five splits, for the reason given in Section IV-C. The checkpoint reported for a fold depends on the dataset, and in every case it is chosen on the evaluation fold itself. For DFEW, the bestfold and five-fold figures come from re-evaluating the saved best checkpoint of each fold with one evaluation script, so the per-class recalls, UAR, and WAR of a fold belong to the same checkpoint. This evaluation is not bit-wise deterministic: repeating it on the Fold 5 checkpoint changed the UAR by 0.35 points and the WAR by 0.18 points (Section IV-C), which indicates the size of the evaluation noise. For MAFW, each split is taken at its epoch of peak WAR, and for FERV39k the best checkpoint is evaluated at the end of training. The ablation uses each fold at its epoch of peak UAR, with the WAR taken at that epoch. Denoting the metric obtained on fold i as $P _ { i }$ , the best-fold result is

$$
P _ { p e a k } = \operatorname* { m a x } _ { i \in \{ 1 , . . . , 5 \} } P _ { i } ,\tag{34}
$$

and the five-fold statistics reported alongside it are the sample mean $\textstyle { \bar { P } } = { \frac { 1 } { 5 } } \sum _ { i = 1 } ^ { 5 } P _ { i }$ and sample standard deviation across the same 5 folds.

2) Implementation Details: Training and testing used one NVIDIA RTX 3090 GPU (24 GB) in a workstation with an Intel Core i9-12900K, PyTorch 2.0+, and CUDA 11.8. Unless stated otherwise in the ablation studies, the GroupMamba-SSM temporal depth is L = 3 layers, and the number of epochs trained in each fold is listed in the Supplementary Material. Automatic Mixed Precision (AMP) was used throughout and was necessary at this token count. With FP16 forward computation and an FP32 master copy of the weights to prevent gradient underflow, AMP made it possible to fit a usable batch size into 24 GB of VRAM for the $N = 2 5 , 0 8 8 -$ token sequences of Eq. (6). Training followed the two-stage S2D protocol described in Section III-F: a temporal warmup phase with spatial parameters and anchors damped by $\gamma _ { f r e e z e } = 0 . 0 1$ , followed by full end-to-end fine-tuning, both optimized by AdamW (decoupled weight decay, learning rate $1 0 ^ { - 4 }$ decayed by a cosine schedule) with a dropout rate of 0.5 after the CAM layer.

## B. Ablation Analysis

This section examines the contribution of each architectural component, tests the DFEW gain for statistical significance, and characterizes how consistently that performance is achieved across data splits.

1) Component Ablation: Table I gives the DFEW performance of the full model and of five variants, each with one component removed or reduced. All rows come from the same five-fold ablation series and are reported as mean ± SD over the five folds. They are therefore not comparable with the best-fold headline of Section IV-C.

Removing temporal modeling produces the only clear change: UAR falls from 64.69% to 63.08%, a loss of 1.61 points, which is examined further in the paired test below. The other four variants stay within 0.3 points of the full model in mean UAR (from −0.26 to +0.29). This is small next to the fold-to-fold SD of 1.5 to 2.2 points, so five-fold experiments of this kind cannot separate the contribution of these components from noise. In WAR, no component shows a benefit: the variants score between 0.1 and 1.3 points higher than the full model, again about one SD.

TABLE I  
COMPONENT ABLATION ON DFEW, MEAN ± SD OVER FIVE FOLDS (POPULATION SD, $n = 5 )$ . EACH FOLD IS TAKEN AT ITS EPOCH OF HIGHEST UAR, WITH WAR AT THAT EPOCH. THE FULL MODEL USES $L = 3$
<table><tr><td>Configuration</td><td>WAR (%)</td><td>UAR (%)</td></tr><tr><td>GM-GReFEL (full)</td><td> $7 3 . 4 6 \pm 1 . 2 1$ </td><td> $6 4 . 6 9 \pm 1 . 6 8$ </td></tr><tr><td>w/o Temporal Modeling</td><td> $7 3 . 5 9 \pm 1 . 2 8$ </td><td> $6 3 . 0 8 \pm 1 . 7 0$ </td></tr><tr><td>w/o Trainable Anchors (frozen)</td><td> $7 4 . 6 9 \pm 1 . 1 9$ </td><td> $6 4 . 9 5 \pm 2 . 2 0$ </td></tr><tr><td>GroupMamba-SSM depth  $L = 1$ </td><td> $7 4 . 5 8 \pm 1 . 4 9$ </td><td> $6 4 . 9 4 \pm 1 . 5 3$ </td></tr><tr><td>w/ Single-Stage Training</td><td> $7 4 . 8 0 \pm 1 . 3 7$ </td><td> $6 4 . 4 0 \pm 2 . 0 0$ </td></tr><tr><td>w/o Jittered Sampling</td><td> $7 4 . 7 3 \pm 1 . 5 2$ </td><td> $6 4 . 6 8 \pm 1 . 9 0$ </td></tr></table>

Our conclusion is therefore limited. In this setting, temporal modeling is the component with a measurable effect on accuracy. The trainable anchors, the three-layer GroupMamba-SSM stack, the two-stage schedule, and jittered sampling were included for design reasons described in Section III: a geometric reliability check, stable early optimization, and variety in the sampled frames. Their effect on average accuracy is not measurable here, and we do not claim that they improve it. These experiments also do not test whether they help training stability or the best-fold results.

2) Hyperparameter Sensitivity: We swept five training hyperparameters on DFEW Fold 1 under an accelerated 50-epoch schedule, since a full 5-fold sweep at four values each would need 100 training runs. The default configuration $( p = 0 . 5$ $L = 3 )$ appears twice in the sweep and reaches 63.62% and 61.70% UAR, so repeated runs of the same configuration differ by about two points, and smaller differences should not be read as effects. Table II gives the best value for each hyperparameter, and the complete results are in the Supplementary Material. Dropout and the two Triple Loss weights show no clear effect. The temperature $\tau ~ = ~ 0 . 1 0$ gives the highest UAR (65.20%), 1.5 to 2.2 points above the other values, which is suggestive but not conclusive. A single GroupMamba layer scores highest in this short schedule (64.19% UAR), which may reflect faster convergence of the smaller model, and $L = 5$ collapses to near-chance accuracy under the S2D initialization. We kept $L = 3 ,$ , the depth chosen in the initial design of the model, as the default throughout for continuity across experiments. The later sweep and the five-fold ablation of Table I show no measurable difference between $L = 3$ and L = 1 (64.69% against 64.94% UAR in the ablation, within the fold-to-fold SD), so this choice does not change our conclusions.

3) Statistical Significance: To see whether temporal modeling gives a reliable gain on DFEW, we compared the full model with a spatial-only baseline using a paired t-test over the five cross-validation folds. In the baseline, frame-level features are mean-pooled over time and no sequential module is used; two-stage training and selective sampling stay active so that the comparison is fair. The null hypothesis is that temporal modeling brings no UAR benefit. We use a one-sided test because the hypothesis is directional.

TABLE II  
HYPERPARAMETER SENSITIVITY SUMMARY (DFEW FOLD 1, ACCELERATED SCHEDULE). DIFFERENCES BELOW ABOUT TWO POINTS ARE WITHIN RUN-TO-RUN VARIATION. FULL SWEEPS IN THE SUPPLEMENTARY MATERIAL.
<table><tr><td>Hyperparameter</td><td>Values tested</td><td>Best</td><td>UAR / WAR (%)</td></tr><tr><td>Dropout p</td><td>0, 0.25, 0.5, 0.75</td><td>0.75</td><td>64.10 / 74.93</td></tr><tr><td>Layers L</td><td>1,3,5</td><td>1</td><td>64.19 / 75.69</td></tr><tr><td> $\lambda _ { 2 }$  (center)</td><td>0, 0.1, 1, 10</td><td>10</td><td>64.02 / 75.95</td></tr><tr><td> $\lambda _ { 1 }$  (orth.)</td><td>0, 0.1, 1, 10</td><td>10</td><td>64.08 /  75.69</td></tr><tr><td>Temperature τ</td><td>0.01, 0.05, 0.1, 0.5</td><td>0.10</td><td>65.20 / 75.22</td></tr></table>

The full model has the higher UAR in four of the five folds, and the mean paired difference is +1.61 points (SD 1.41; $t ( 4 ) = 2 . 5 6 )$ . The one-sided p-value is 0.031, which is below 0.05, so we reject the null hypothesis at that level. The twosided p-value is 0.063, and the 95% confidence interval for the mean difference is [−0.14, 3.36] points. The gain is therefore modest, and the evidence for it is moderate, not strong; in Fold 3 the baseline is marginally better (−0.15 points). The per-fold values are listed in the Supplementary Material. The full-model mean here (64.69% UAR) is that of Table I; it differs from the DFEW mean in Table III (64.07%) because the two use different checkpoint-selection rules (Section IV-A).

The test was run on DFEW only. The five-fold baseline was trained on DFEW alone because five-fold training of the Modulated GroupMamba engine is computationally expensive, so there are no paired results for FERV39k or MAFW and we make no statistical claim for them. With only five folds the test has low power, and we present it as supporting evidence for DFEW and not as a general guarantee.

4) Cross-Fold Stability: The DFEW and MAFW figures in the main comparison tables are the best cross-validation folds, whereas the FERV39k figure is the five-split mean (Section IV-C). A best fold does not show how reliably a result is reached, so Table III summarizes all five folds of the same runs by mean and standard deviation; the per-fold values are in the Supplementary Material.

On DFEW, the best fold (Fold 5) reaches 66.57% UAR and 76.46% WAR, and the five-fold mean is 64.07% UAR / 74.54% WAR (±1.80 / ±1.54). FERV39k shows the tightest spread (±0.84 / ±0.69); its best split (43.68% / 53.81%) lies above the reported five-split mean (42.30% / 52.66%). MAFW shows the widest spread (±5.65 / ±6.35), consistent with an 11-way label space and few samples per class; its Fold 4 (45.66% / 62.27% at epoch 38) matches the headline result, while the remaining folds range from 31.51% to 43.31% UAR, so the headline value marks the upper end of the fold distribution.

## C. Comparison With Published Methods

1) DFEW: Table IV compares GM-GReFEL against a broad set of DFER baselines spanning 3D CNNs, recurrent hybrids, and Transformer architectures. The best fold of our five-fold run (Fold 5) reaches 66.57% UAR and 76.46% WAR. The UAR is 9.12 points above the strongest UAR among the older baselines in the table (MIDAS [30], 57.45%, shown rounded to 57.4%) and 0.23 points below S4D (66.8% UAR, 76.6% WAR), the strongest large-scale pre-trained model in Table VII; the WAR is 0.14 points below S4D. Most compared methods are reported as five-fold averages, so the fairer comparison is our five-fold mean of 64.07% UAR and 74.54% WAR, given in the last row of Table IV. On that basis GM-GReFEL is 6.6 points above MIDAS, close to MAE-DFER and VideoMAE (63.4% and 63.6% UAR), and 2.7 points below S4D. For both GM-GReFEL rows, the per-class recalls, UAR, and WAR come from one evaluation of the saved best checkpoint of each fold, so the UAR equals the mean of the seven per-class recalls.

TABLE III  
CROSS-FOLD MEAN AND STANDARD DEVIATION (SAMPLE SD), DFEW/FERV39K/MAFW
<table><tr><td>Dataset</td><td>Mean UAR</td><td>Std UAR</td><td>Mean WAR</td><td>Std WAR</td></tr><tr><td>DFEW</td><td>64.07%</td><td>±1.80</td><td>74.54%</td><td>±1.54</td></tr><tr><td>FERV39k</td><td>42.30%</td><td>±0.84</td><td>52.66%</td><td>±0.69</td></tr><tr><td>MAFW</td><td>38.90%</td><td>±5.65</td><td>54.37%</td><td>±6.35</td></tr></table>

Per-fold recall shows that majority classes such as “Happy” are essentially bulletproof across folds (91.2–94.9%), while minority classes like “Disgust” show substantially more interfold variance, which we attribute to the scarcity of samples in those splits (full per-fold breakdown in the Supplementary Material).

In the best fold, the recall is 94.3% on Happy, 85.7% on Sad, and 80.7% on Angry, while the minority classes Fear (49.4%) and Disgust (24.1%) remain the weakest (Table IV). Fig. 3 shows the normalized confusion matrix of the Fold 5 checkpoint from a repeated evaluation. Because the evaluation is not bit-wise deterministic, the repetition differs slightly from Table IV: the UAR changes by 0.35 points, the WAR by 0.18 points, and the largest change in a single recall is 3.4 points, which is one of the 29 Disgust test clips (8 instead of 7 correct). We therefore use the figure only to show the pattern of confusions. Misclassifications follow facial anatomy: “Disgust” is confused with “Neutral” (34%) and “Angry” (17%), and “Fear” bleeds into “Surprise” (17%), since both expressions widen the eyes in a similar way.

2) FERV39k: FERV39k presents an even more severe class-imbalance challenge than DFEW, with minority emotions such as “Disgust” and “Fear” constituting a small fraction of the dataset. Table V compares GM-GReFEL against the strongest published baselines. Among the methods in Table V, the strongest UAR was 35.94% (M3DFEL [28]); GM-GReFEL raises this to 42.30%, a +6.36-point gain, with WAR reaching 52.66%. These figures are means over the five splits of our cross-validation run, not the best split (43.68% / 53.81%, Split 3). We report the mean because the splits agree closely: the coefficient of variation of UAR is 1.99% for FERV39k, against 2.82% for DFEW and 14.52% for MAFW. With this little variation, the mean is what a further held-out split would be expected to reproduce. Several large-scale pre-trained ViT-B/16 models in Table VII remain ahead on this dataset, as discussed in the cross-dataset summary below. Per-split results are in the Supplementary Material and Section IV-B4.

![](images/995dfa91bd394daf1a7f11c8c7e94e37ce3b33a36ff64ff8267e3bfd67ceac02.jpg)  
Fig. 3. Normalized confusion matrix for DFEW, Fold 5 (repeated evaluation of the saved checkpoint; see text).

FERV39k’s training dynamics follow the same pattern as DFEW, with smooth validation loss decay and no catastrophic forgetting. Its confusion matrix shows the same anatomydriven error structure: “Disgust” and “Fear” scatter toward classes with overlapping muscular signatures (e.g., “Fear” bleeding into “Sadness”), which suggests that the anchors degrade gracefully under data scarcity (full curves and confusion matrix in the Supplementary Material).

3) MAFW: MAFW tests a different axis of difficulty: 11- way compound-emotion classification, where categories such as “Disappointment” and “Anxiety” are geometrically close to more basic states. On its best fold, GM-GReFEL reaches 45.66% UAR and 62.27% WAR (Table VI), which is 3.83 points higher in WAR and 1.94 points higher in UAR than the strongest baseline, S4D [39] (58.44% WAR, 43.72% UAR). MAFW also has the widest spread across folds. The fivefold mean is 38.90% UAR and 54.37% WAR (±5.65 and ±6.35), shown in the last row of Table VI. On this basis GM-GReFEL has a lower UAR than T-MEP, DFER-CLIP, and the self-supervised models (39.4–43.7%) and a WAR similar to SVFAP and MAE-DFER (54.28% and 54.31%).

MAFW’s training dynamics show strong generalization, with validation loss settling below training loss. Its confusion matrix separates structurally distinct emotions cleanly (Disgust, Fear, Disappointment, and Anger all above 78% true-positive rate) while ambiguous compound classes such as “Surprise” and “Anxiety” scatter toward anatomically adjacent basic emotions, so even the errors follow facial physiology (full curves, confusion matrix, and per-class breakdown in the Supplementary Material).

4) Cross-Dataset Summary: Table VII lists the compared methods on all three benchmarks, with GM-GReFEL shown for its best fold and as a five-fold mean. On the best fold,

TABLE IV  
RECALL (%) OF GM-GREFEL AND PUBLISHED METHODS ON DFEW.  
H = HAPPY, S = SAD, N = NEUTRAL, A = ANGRY, SP = SURPRISE, D = DISGUST, F = FEAR. GM-GREFEL IS SHOWN FOR ITS BEST FOLD (FOLD 5) AND FOR THE MEAN OVER FIVE FOLDS, BOTH FROM ONE EVALUATION OF THE SAVED BEST CHECKPOINTS. BOLD MARKS THE BEST VALUE IN EACH COLUMN.
<table><tr><td>Method</td><td>H</td><td>S</td><td>N</td><td>A</td><td>Sp</td><td>D</td><td>F</td><td>UAR</td><td>WAR</td></tr><tr><td>C3D [18]</td><td>75.1</td><td>39.4</td><td>55.1</td><td>62.4</td><td>45.0</td><td>1.3</td><td>20.5</td><td>42.7</td><td>53.5</td></tr><tr><td>P3D [19]</td><td>74.8</td><td>43.4</td><td>54.1</td><td>60.4</td><td>50.9</td><td>0.6</td><td>23.2</td><td>43.9</td><td>54.4</td></tr><tr><td>I3D [20]</td><td>78.6</td><td>44.1</td><td>56.6</td><td>55.8</td><td>45.8</td><td>2.0</td><td>20.5</td><td>43.4</td><td>54.2</td></tr><tr><td>R(2+1)D18 [21]</td><td>79.6</td><td>39.0</td><td>57.6</td><td>50.3</td><td>48.2</td><td>3.4</td><td>21.0</td><td>42.7</td><td>53.2</td></tr><tr><td>3D ResNet18 [22]</td><td>73.1</td><td>48.2</td><td>50.5</td><td>64.7</td><td>50.1</td><td>0.0</td><td>26.3</td><td>44.7</td><td>54.9</td></tr><tr><td>ResNet18+LSTM [2]</td><td>78.0</td><td>40.6</td><td>53.7</td><td>56.8</td><td>45.0</td><td>4.1</td><td>21.6</td><td>42.8</td><td>53.0</td></tr><tr><td>EC-STFL [2]</td><td>79.1</td><td>49.0</td><td>57.8</td><td>60.9</td><td>46.1</td><td>2.7</td><td>21.5</td><td>45.3</td><td>56.5</td></tr><tr><td>ResNet18+GRU [23]</td><td>82.8</td><td>63.8</td><td>65.0</td><td>68.5</td><td>52.0</td><td>0.8</td><td>30.1</td><td>51.6</td><td>64.0</td></tr><tr><td>FormerDFER [23]</td><td>84.0</td><td>62.5</td><td>67.5</td><td>70.0</td><td>56.4</td><td>3.4</td><td>31.7</td><td>53.6</td><td>65.7</td></tr><tr><td>STT [24]</td><td>87.3</td><td>67.9</td><td>64.9</td><td>71.2</td><td>53.1</td><td>3.4</td><td>34.0</td><td>54.5</td><td>66.4</td></tr><tr><td>DPCNet [25]</td><td></td><td></td><td></td><td></td><td></td><td>一</td><td></td><td>55.0</td><td>66.3</td></tr><tr><td>NR-DFERNet [26]</td><td>88.4</td><td>64.8</td><td>70.0</td><td>75.0</td><td>61.6</td><td>0.0</td><td>19.4</td><td>54.2</td><td>68.1</td></tr><tr><td>GCA+IAL [27]</td><td>87.9</td><td>67.2</td><td>70.1</td><td>76.0</td><td>62.2</td><td>0.0</td><td>26.4</td><td>55.7</td><td>69.2</td></tr><tr><td>M3DFEL [28]</td><td>89.5</td><td>68.3</td><td>67.8</td><td>74.2</td><td>59.6</td><td>0.0</td><td>31.6</td><td>56.1</td><td>69.2</td></tr><tr><td>RDFER [29]</td><td>89.6</td><td>69.2</td><td>70.1</td><td>71.4</td><td>62.0</td><td>0.6</td><td>28.7</td><td>56.9</td><td>69.7</td></tr><tr><td>ResNet18+TSM MIDAS [30]</td><td>87.4</td><td>67.3</td><td>58.6</td><td>68.0</td><td>59.6</td><td>28.6</td><td>44.5</td><td>57.4</td><td>69.1</td></tr><tr><td>GM-GReFEL (Ours, best fold)</td><td>94.3</td><td>85.7</td><td>70.4</td><td>80.7</td><td>61.4</td><td>24.1</td><td>49.4</td><td>66.6</td><td>76.5</td></tr><tr><td>GM-GReFEL (Ours, 5-fold mean)</td><td>93.1</td><td>80.5</td><td>71.1</td><td>78.3</td><td>61.4</td><td>21.4</td><td>42.7</td><td>64.1</td><td>74.5</td></tr></table>

TABLE V  
UAR AND WAR (%) OF GM-GREFEL AND PUBLISHED METHODS ON FERV39K
<table><tr><td>Method</td><td>WAR</td><td>UAR</td></tr><tr><td>C3D [18]</td><td>31.69</td><td>22.68</td></tr><tr><td>P3D [19]</td><td>33.39</td><td>23.20</td></tr><tr><td>I3D [20]</td><td>38.78</td><td>30.17</td></tr><tr><td>R(2+1)D18 [21]</td><td>41.28</td><td>31.55</td></tr><tr><td>3D ResNet18 [22]</td><td>37.57</td><td>26.67</td></tr><tr><td>R18+LSTM [1]</td><td>42.95</td><td>30.92</td></tr><tr><td>2R18+LSTM [1]</td><td>43.20</td><td>31.28</td></tr><tr><td>Former-DFER [23]</td><td>45.72</td><td>36.88</td></tr><tr><td>NR-DFERNet [26]</td><td>45.97</td><td>33.99</td></tr><tr><td>GCA+IAL [27]</td><td>48.54</td><td>35.82</td></tr><tr><td>M3DFEL [28]</td><td>47.67</td><td>35.94</td></tr><tr><td>GM-GReFEL (Ours, 5-split mean)</td><td>52.66</td><td>42.30</td></tr></table>

GM-GReFEL has the highest UAR and WAR among all listed methods on MAFW and comes within 0.3 points of S4D on DFEW (66.6% against 66.8% UAR, 76.5% against 76.6% WAR). The five-fold mean is the like-for-like basis for most baselines, and on it the picture is more balanced. On DFEW, GM-GReFEL (64.1% UAR, 74.5% WAR) is level with the recent pre-trained ViT-B/16 models (61.8–64.0% UAR, 74.2– 76.0% WAR) and below S4D (66.8%, 76.6%). On MAFW, its mean UAR (38.9%) is below that of T-MEP, DFER-CLIP, and the self-supervised models (39.4–43.7%), while its mean WAR (54.4%) is close to SVFAP and MAE-DFER. On FERV39k, three pre-trained models (MAE-DFER, VideoMAE, and S4D) report higher UAR (43.1–43.4% versus our 42.3%), and S4D also has the higher WAR (53.6% versus 52.7%). Averaged over the three datasets and both metrics, our headline figures give

TABLE VI  
UAR AND WAR (%) OF GM-GREFEL AND PUBLISHED METHODS ON MAFW
<table><tr><td>Method</td><td>WAR</td><td>UAR</td></tr><tr><td>Former-DFER [23]</td><td>43.27</td><td>31.16</td></tr><tr><td>T-MEP [31]</td><td>52.85</td><td>39.37</td></tr><tr><td>DFER-CLIP [32]</td><td>52.55</td><td>39.89</td></tr><tr><td>EmoCLIP [33]</td><td>41.46</td><td>34.24</td></tr><tr><td>A3lign-DFER [34]</td><td>53.24</td><td>42.07</td></tr><tr><td>SVFAP [35]</td><td>54.28</td><td>41.19</td></tr><tr><td>HiCMAE [36]</td><td>56.17</td><td>42.65</td></tr><tr><td>MAE-DFER [37]</td><td>54.31</td><td>41.62</td></tr><tr><td>S2D [38]</td><td>57.37</td><td>41.86</td></tr><tr><td>VideoMAE [17]</td><td>53.51</td><td>40.87</td></tr><tr><td>S4D [39]</td><td>58.44</td><td>43.72</td></tr><tr><td>GM-GReFEL (Ours, best fold) GM-GReFEL (Ours, 5-fold mean)</td><td>62.27 54.37</td><td>45.66 38.90</td></tr></table>

57.7% against 57.1% for S4D, whereas the five-fold means give 54.5%. We therefore regard GM-GReFEL as competitive with large pre-trained models but not uniformly ahead of them.

## D. Visualization

1) Training Convergence: Because DFEW is representative of the general unconstrained-video setting, we use it to illustrate convergence behavior; the same qualitative pattern held across all three datasets. Fig. 4 plots the mean training and validation loss across all five folds, with shaded variance bands. Training loss stabilizes at a visibly higher plateau (≈ 1.5) than validation loss (≈ 0.9); this is the expected consequence of the aggressive 50% dropout applied after the CAM layer during training, which is disabled at validation time. The narrow variance bands across folds indicate that this convergence pattern is not an artifact of any single data split.

TABLE VII  
UAR AND WAR (%) ON DFEW, FERV39K, AND MAFW, GROUPED BY METHOD, SAMPLING, AND BACKBONE. DS = DYNAMIC SAMPLING; TI = TIME INTERPOLATION; GROUPMAMBA-T = THE TINY-CONFIGURATION GROUPMAMBA BACKBONE USED THROUGHOUT THIS PAPER (SECTION III-D). FOR FERV39K, BOTH GM-GREFEL ROWS GIVE THE FIVE-SPLIT MEAN. BOLD MARKS THE BEST VALUE IN EACH COLUMN.
<table><tr><td rowspan="2">Method (Sampling)/Backbone</td><td colspan="2">DFEW</td><td colspan="2">FERV39k</td><td colspan="2">MAFW</td></tr><tr><td>UAR</td><td>WAR</td><td>UAR</td><td>WAR</td><td>UAR</td><td>WAR</td></tr><tr><td>EC-STFL (TI)/C3D+P3D [2]</td><td>45.3</td><td>56.5</td><td>一</td><td>一</td><td>一</td><td>一</td></tr><tr><td>Former-DFER (DS)/Trans. [23]</td><td>53.6</td><td>65.7</td><td>37.2</td><td>46.8</td><td>31.1</td><td>43.2</td></tr><tr><td>CEFLNet (CLIP)/ResNet-18 [40]</td><td>51.1</td><td>65.3</td><td></td><td>一</td><td>一</td><td></td></tr><tr><td>NR-DFERNet (DS)/CNN-Trans. [26]</td><td>54.2</td><td>68.1</td><td>33.9</td><td>45.9</td><td>一</td><td>1</td></tr><tr><td>STT (DS)/ResNet-18 [24]</td><td>54.5</td><td>66.6</td><td>37.7</td><td>48.1</td><td>一</td><td></td></tr><tr><td>EST (DS)/ResNet-18 [41]</td><td>53.9</td><td>65.8</td><td>一</td><td>一</td><td>一</td><td></td></tr><tr><td>Freq-HD (FreqHD)/VGG13-LSTM [42]</td><td>46.8</td><td>55.6</td><td>33.0</td><td>45.2</td><td>一</td><td></td></tr><tr><td>LOGO-Former (DS)/ResNet-18 [43]</td><td>54.2</td><td>66.9</td><td>38.2</td><td>48.1</td><td>一</td><td></td></tr><tr><td>IAL (DS)/ResNet-18 [27]</td><td>55.7</td><td>69.2</td><td>35.8</td><td>48.5</td><td>一</td><td></td></tr><tr><td>AEN (DS)/ResNet-18 [44]</td><td>56.6</td><td>69.3</td><td>38.1</td><td>47.8</td><td>一</td><td>=</td></tr><tr><td>T-MEP (DS)/Transformer [31]</td><td>57.1</td><td>68.8</td><td></td><td>一</td><td>39.3</td><td>52.8</td></tr><tr><td>M3DFEL (DS)/ResNet-18-3D [28]</td><td>56.1</td><td>69.2</td><td>35.9</td><td>47.6</td><td>一</td><td>一</td></tr><tr><td>CLIPER (DS)/CLIP-ViT-B/16 [45]</td><td>57.5</td><td>70.8</td><td>41.2</td><td>51.3</td><td>1</td><td>1</td></tr><tr><td>DFER-CLIP (DS)/CLIP-ViT-B/32 [32]</td><td>59.6</td><td>71.2</td><td>41.2</td><td>51.6</td><td>39.8</td><td>52.5</td></tr><tr><td>EmoCLIP (DS)/CLIP-ViT-B/32 [33]</td><td>58.0</td><td>62.1</td><td>31.4</td><td>36.1</td><td>34.2</td><td>41.4</td></tr><tr><td>A3lign-DFER (DS)/CLIP-ViT-L/14 [34]</td><td>64.0</td><td>74.2</td><td>41.8</td><td>51.7</td><td>42.0</td><td>53.2</td></tr><tr><td>SVFAP (DS)/ViT-B/16 [35]</td><td>62.8</td><td>74.2</td><td>42.1</td><td>52.2</td><td>41.1</td><td>54.2</td></tr><tr><td>HiCMAE (DS)/ViT-B/16 [36]</td><td>63.7</td><td>75.0</td><td></td><td>一</td><td>42.6</td><td>56.1</td></tr><tr><td>MAE-DFER (DS)/ViT-B/16 [37]</td><td>63.4</td><td>74.4</td><td>43.1</td><td>52.0</td><td>41.6</td><td>54.3</td></tr><tr><td>S2D (DS)/ViT-B/16 [38]</td><td>61.8</td><td>76.0</td><td>41.2</td><td>52.5</td><td>41.8</td><td>57.3</td></tr><tr><td>VideoMAE (DS)/ViT-B/16 [17]</td><td>63.6</td><td>74.6</td><td>43.3</td><td>52.3</td><td>40.8</td><td>53.5</td></tr><tr><td>S4D (DS)/ViT-B/16 [39]</td><td>66.8</td><td>76.6</td><td>43.4</td><td>53.6</td><td>43.7</td><td>58.4</td></tr><tr><td>GM-GReFEL, best fold (DS)/GroupMamba-T (Ours)</td><td>66.6</td><td>76.5</td><td>42.3</td><td>52.7</td><td>45.7</td><td>62.3</td></tr><tr><td>GM-GReFEL, 5-fold mean (DS)/GroupMamba-T (Ours)</td><td>64.1</td><td>74.5</td><td>42.3</td><td>52.7</td><td>38.9</td><td>54.4</td></tr></table>

Training and Validation Loss (5 Splits)  
![](images/0eb4ff8b7c7d0c7899458d2aec1c62f0aca26135adb2f46158e37d6214cd5b2c.jpg)  
Fig. 4. Mean training and validation loss trajectory on DFEW, averaged across all five cross-validation folds with shaded variance bands.  
Validation WAR & UAR over Epochs (5 Splits)

UAR and WAR both track a consistent trajectory across training, as shown in Fig. 5. Both metrics rise sharply during the first 20 epochs and then settle into a stable plateau. The narrow variance bands over the final 50 epochs show that training converges consistently across folds.

![](images/d33bb18ed9828d44edda37ecfab3d6c0b6d375ed06546789d714deeb3e34d9c5.jpg)  
Fig. 5. Progression of UAR and WAR on DFEW across training, averaged across five folds.

2) Interpretability and Attention Analysis: To assess whether GM-GReFEL relies on facial physiology or on incidental background cues, we applied Grad-CAM to the final Modulated GroupMamba layer and projected the resulting gradients back onto the input frames. Fig. 6 traces attention across a representative DFEW clip labeled “Angry.” In the earliest frames, attention concentrates on the ocular region as the subject shows a surprise-like widening of the eyes; as the temporal groups detect the characteristic downward brow motion and mouth tightening that follow, attention shifts decisively toward the mouth and jawline by the emotional apex. This suggests that the network follows the temporal evolution of the expression instead of classifying frames in isolation. These are illustrative case studies, not statistical evidence; four further case studies are in the Supplementary Material.

![](images/135b361d8612c05873f5dab74d3d16dafbfeb8d1b0cf9c8ffb149557dd9eb9d4.jpg)  
Fig. 6. Grad-CAM attention across a DFEW clip labeled “Angry,” showing attention shifting from the ocular region toward the mouth and jawline as the expression reaches its apex.

Visualizations for Video 16354 (Surprise)  
![](images/dd16213854c2a9f23011ca863cca0aba7d7f5cb38c00a7a5696b114aff8bdce0.jpg)  
Fig. 7. Comparison of attention maps for a “Surprise” sample: full GroupMamba-SSM model (localized on mouth and eyes) vs. a mean-pooling baseline (diffuse, drifting into background).

A direct side-by-side comparison makes the effect concrete: Fig. 7 places attention maps from the full GroupMamba-SSM model against a mean-pooling baseline on the same “Surprise” sample. The baseline’s attention is diffuse, bleeding into hair, clothing, and background; the full model’s attention stays tightly localized on the mouth and eyes and remains stable across the 16-frame sequence. This may reflect the constraint that the geometric anchors place on attention, although the ablation in Section IV-B1 does not isolate their effect on accuracy.

3) t-SNE Feature Space Analysis: Projecting the learned 768-dimensional feature space into two dimensions via t-SNE (Fig. 8) offers a complementary, global view of separability. In each case, the architecture forms compact, well-separated clusters with minimal inter-class overlap. Even the 11 compound classes of MAFW remain separable, which agrees with the Triple Loss objective reducing intra-class variance and enlarging inter-class margins in the shared latent space.

4) Confusion and Misclassification Patterns: A radar comparison of per-class recall shared between FERV39k and MAFW, a breakdown of the most frequent misclassification pairs, cross-fold WAR/UAR stability across the full 5-fold cycle, and a comparison of per-class recall against F1-score are given in the Supplementary Material. On FERV39k, seven of the ten most frequent errors involve the “Neutral” class, so the main remaining difficulty there appears to be locating the onset and offset of an expression. On MAFW the frequent errors are spread over pairs of related compound emotions, such as Neutral to Fear, Happiness to Contempt, and Disappointment to Disgust, and only three of the ten involve “Neutral”.

![](images/e2b885afd0337d7b70ff3f2d4b749f192ca9d4fa1c28b2c173726d2099cafcb9.jpg)  
Fig. 8. t-SNE projections of the learned 768-dimensional latent feature space for DFEW, FERV39k, and MAFW.

TABLE VIII  
COMPUTATIONAL COMPLEXITY AND INFERENCE SPEED COMPARISON
<table><tr><td>Metric</td><td>GM-GReFEL</td><td>ViT-B/16</td><td>∆</td></tr><tr><td>Total Params</td><td>155.99 M</td><td>147.69 M</td><td>+8.30 M (+5.6%)</td></tr><tr><td>GFLOPs</td><td>539.62</td><td>548.75</td><td>-9.13</td></tr><tr><td>FPS (clips/s)</td><td>21.1</td><td>21.4</td><td>-0.3 (−1.4%)</td></tr><tr><td>Latency (ms)</td><td>47.4</td><td>46.8</td><td>+0.6</td></tr></table>

## E. Computational Efficiency

Table VIII compares GM-GReFEL with a ViT-B/16 baseline profiled under identical settings. The reliability module raises the parameter count to 155.99M versus 147.69M (+5.6%), but replacing quadratic self-attention with the linear GroupMamba engine lowers computation to 539.62 versus 548.75 GFLOPs. Measured throughput is essentially unchanged (21.1 versus 21.4 clips/s; 47.4 versus 46.8 ms per clip), so the reliability machinery does not compromise practical inference speed.

Table IX places GM-GReFEL alongside other DFER methods. Only the GM-GReFEL and ViT-B/16 rows were profiled locally (224×224, T = 16, thop, RTX 3090); the remaining figures are taken from each method’s publication and may use different input settings, so they serve as an approximate reference.

GM-GReFEL has more parameters than every listed method, including the ViT-B/16-based SVFAP-B, MAE-DFER, VideoMAE, and S4D (78–90M) and the much lighter M3DFEL. This capacity resides in the GroupMamba-SSM engine and the reliability module, whose contributions are examined in Section IV-B; the case for GM-GReFEL rests on its accuracy and on the linear-complexity argument of Section III-G, not on parameter count. GFLOPs for S4D and A<sup>3</sup>lign-DFER are not reported in their publications and are left blank; we did not estimate them.

COMPUTE COMPARISON WITH OTHER DFER METHODS. <sup>†</sup>: MEASURED LOCALLY (224×224, T =16, T H O P, RTX 3090). OTHER ROWS ARE TAKEN FROM EACH METHOD’S PUBLICATION AND MAY USE DIFFERENT SETTINGS; – MEANS NOT REPORTED.
<table><tr><td>Method</td><td>Backbone</td><td>Params (M)</td><td>GFLOPs</td></tr><tr><td>M3DFEL [28]</td><td>ResNet-18-3D</td><td>37.5</td><td>58.0</td></tr><tr><td>SVFAP-B [35]</td><td>ViT-B/16</td><td>78.0</td><td>44.0</td></tr><tr><td>MAE-DFER [37]</td><td>ViT-B/16</td><td>85.0</td><td>50.0</td></tr><tr><td>VideoMAE [17]</td><td>ViT-B/16</td><td>86.0</td><td>82.0</td></tr><tr><td>S4D [39]</td><td>ViT-B/16 (MoAE)</td><td>90.0</td><td></td></tr><tr><td>A3lign-DFER [34]</td><td>CLIP-ViT-L/14</td><td></td><td></td></tr><tr><td>ViT-B/16 baseline†</td><td>ViT-B/16</td><td>147.69</td><td>548.75</td></tr><tr><td>GM-GReFEL (Ours)†</td><td>GroupMamba-T</td><td>155.99</td><td>539.62</td></tr></table>

## V. LIMITATIONS

Three scope constraints bound this study. First, GM-GReFEL is unimodal: it uses visual input only, whereas emotion is also conveyed by voice and gesture, and aligning audio streams with the tubelet schedule is outside the present scope. Second, it performs categorical recognition and does not model dimensional (valence–arousal) affect, because the geometric anchors are tied to discrete class prototypes and cannot directly track blended or graded affect. Third, although the backbone is linear in sequence length, the full pipeline still requires a laboratory-grade GPU to reach the latency in Table VIII; quantization and pruning for edge deployment remain future work.

## VI. CONCLUSION

This paper presented GM-GReFEL, a DFER architecture that jointly addresses the quadratic memory cost of selfattention and the label noise of in-the-wild datasets. A Modulated GroupMamba engine scans orthogonal spatial and temporal channel groups with a linear state-space recurrence and re-couples them through Channel Affinity Modulation, and an entropy-guided geometric reliability module provides an anatomical fallback whenever the statistical prediction is uncertain. On its best fold, GM-GReFEL gives the highest UAR and WAR among the compared methods on MAFW and comes within 0.3 points of the strongest method on DFEW. Its five-fold means are comparable to recent pre-trained models on DFEW and lower on MAFW, and its FERV39k result is competitive. On DFEW, removing temporal modeling costs a small but consistent amount of accuracy according to a paired test. The attention maps stay on the face, the emotion clusters in feature space are well separated, and inference is about as fast as with a ViT-B/16 baseline. Future work includes fusing audio and visual state-space streams, modeling dimensional affect with a continuous anchor manifold, and compressing the model for real-time deployment.

## ACKNOWLEDGMENT

We are grateful to the CIHCI Lab for discussions and feedback on this work, and to NTUST for the GPUs used in the experiments.

[1] Y. Wang et al., “FERV39k: A Large-Scale Multi-Scene Dataset for Facial Expression Recognition in Videos,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2022.

[2] X. Jiang et al., “DFEW: A Large-Scale Database for Recognizing Dynamic Facial Expressions in the Wild,” in Proc. 28th ACM Int. Conf. Multimedia, New York, NY, USA: ACM, Oct. 2020, pp. 2881–2889, doi: 10.1145/3394171.3413620.

[3] Y. Liu et al., “MAFW: A Large-scale, Multi-modal, Compound Affective Database for Dynamic Facial Expression Recognition in the Wild,” in Proc. 30th ACM Int. Conf. Multimedia, 2022.

[4] A. Arnab, M. Dehghani, G. Heigold, C. Sun, M. Lucic, and C. Schmid, “ViViT: A Video Vision Transformer,” in 2021 IEEE/CVF Int. Conf. Comput. Vis. (ICCV), IEEE, Oct. 2021, pp. 6816–6826, doi: 10.1109/ICCV48922.2021.00676.

[5] H. Fan et al., “Multiscale Vision Transformers,” in 2021 IEEE/CVF Int. Conf. Comput. Vis. (ICCV), IEEE, Oct. 2021, pp. 6804–6815, doi: 10.1109/ICCV48922.2021.00675.

[6] A. Gu and T. Dao, “Mamba: Linear-Time Sequence Modeling with Selective State Spaces,” arXiv preprint arXiv:2312.00752, May 2024.

[7] Y. Liu et al., “VMamba: Visual State Space Model,” arXiv preprint arXiv:2401.10166, Dec. 2024.

[8] L. Zhu, B. Liao, Q. Zhang, X. Wang, W. Liu, and X. Wang, “Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model,” arXiv preprint arXiv:2401.09417, Nov. 2024.

[9] K. Li et al., “VideoMamba: State Space Model for Efficient Video Understanding,” arXiv preprint arXiv:2403.06977, Mar. 2024.

[10] A. Shaker, S. T. Wasim, S. Khan, J. Gall, and F. S. Khan, “GroupMamba: Efficient Group-Based Visual State Space Model,” in 2025 IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), IEEE, Jun. 2025, pp. 14912–14922, doi: 10.1109/CVPR52734.2025.01389.

[11] J. She, Y. Hu, H. Shi, J. Wang, Q. Shen, and T. Mei, “Dive into Ambiguity: Latent Distribution Mining and Pairwise Uncertainty Estimation for Facial Expression Recognition,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), Apr. 2021.

[12] Y. Zhang, C. Wang, X. Ling, and W. Deng, “Learn From All: Erasing Attention Consistency for Noisy Label Facial Expression Recognition,” in Proc. Eur. Conf. Comput. Vis. (ECCV), Sep. 2022.

[13] A. T. Wasi, T. H. Rafi, R. Islam, K. Serbetar, and D. K. Chae, “GReFEL: Geometry-Aware Reliable Facial Expression Learning under Bias and Imbalanced Data Distribution,” Oct. 2024, doi: 10.1007/978-981-96- 0911-6 27.

[14] A. H. Farzaneh and X. Qi, “Facial Expression Recognition in the Wild via Deep Attentive Center Loss,” in 2021 IEEE Winter Conf. Appl. Comput. Vis. (WACV), IEEE, Jan. 2021, pp. 2401–2410, doi: 10.1109/WACV48630.2021.00245.

[15] F. Boutros, N. Damer, F. Kirchbuchner, and A. Kuijper, “ElasticFace: Elastic Margin Loss for Deep Face Recognition,” in 2022 IEEE/CVF Conf. Comput. Vis. Pattern Recognit. Workshops (CVPRW), IEEE, Jun. 2022, pp. 1577–1586, doi: 10.1109/CVPRW56347.2022.00164.

[16] J. Mao et al., “POSTER++: A simpler and stronger facial expression recognition network,” Pattern Recognit., vol. 157, p. 110951, Jan. 2025, doi: 10.1016/j.patcog.2024.110951.

[17] Z. Tong, Y. Song, J. Wang, and L. Wang, “VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training,” in Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), Oct. 2022.

[18] D. Tran, L. Bourdev, R. Fergus, L. Torresani, and M. Paluri, “Learning Spatiotemporal Features with 3D Convolutional Networks,” in 2015 IEEE Int. Conf. Comput. Vis. (ICCV), IEEE, Dec. 2015, pp. 4489–4497, doi: 10.1109/ICCV.2015.510.

[19] Z. Qiu, T. Yao, and T. Mei, “Learning Spatio-Temporal Representation with Pseudo-3D Residual Networks,” in 2017 IEEE Int. Conf. Comput. Vis. (ICCV), IEEE, Oct. 2017, pp. 5534–5542, doi: 10.1109/ICCV.2017.590.

[20] J. Carreira and A. Zisserman, “Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), Feb. 2018.

[21] D. Tran, H. Wang, L. Torresani, J. Ray, Y. LeCun, and M. Paluri, “A Closer Look at Spatiotemporal Convolutions for Action Recognition,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), Apr. 2018.

[22] K. Hara, H. Kataoka, and Y. Satoh, “Can Spatiotemporal 3D CNNs Retrace the History of 2D CNNs and ImageNet?,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), Apr. 2018.

[23] Z. Zhao and Q. Liu, “Former-DFER: Dynamic Facial Expression Recognition Transformer,” in Proc. 29th ACM Int. Conf. Multime-

dia, New York, NY, USA: ACM, Oct. 2021, pp. 1553–1561, doi: 10.1145/3474085.3475292.

[24] F. Ma, B. Sun, and S. Li, “Spatio-Temporal Transformer for Dynamic Facial Expression Recognition in the Wild,” arXiv preprint arXiv:2205.04749, May 2022.

[25] Y. Wang et al., “DPCNet: Dual Path Multi-Excitation Collaborative Network for Facial Expression Representation Learning in Videos,” in Proc. 30th ACM Int. Conf. Multimedia, New York, NY, USA: ACM, Oct. 2022, pp. 101–110, doi: 10.1145/3503161.3547865.

[26] H. Li, M. Sui, Z. Zhu, and F. Zhao, “NR-DFERNet: Noise-Robust Network for Dynamic Facial Expression Recognition,” arXiv preprint arXiv:2206.12678, Jun. 2022.

[27] H. Li, H. Niu, Z. Zhu, and F. Zhao, “Intensity-Aware Loss for Dynamic Facial Expression Recognition in the Wild,” Proc. AAAI Conf. Artif. Intell., vol. 37, no. 1, pp. 67–75, Jun. 2023, doi: 10.1609/aaai.v37i1.25077.

[28] H. Wang et al., “Rethinking the Learning Paradigm for Dynamic Facial Expression Recognition,” in 2023 IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), IEEE, Jun. 2023, pp. 17958–17968, doi: 10.1109/CVPR52729.2023.01722

[29] F. Liu, H. Wang, and S. Shen, “Robust Dynamic Facial Expression Recognition,” arXiv preprint arXiv:2502.16129, Feb. 2025.

[30] R. Kawamura, H. Hayashi, N. Takemura, and H. Nagahara, “MIDAS: Mixing Ambiguous Data with Soft Labels for Dynamic Facial Expression Recognition,” arXiv preprint arXiv:2503.00202, Mar. 2025.

[31] X. Zhang, M. Li, S. Lin, H. Xu, and G. Xiao, “Transformer-Based Multimodal Emotional Perception for Dynamic Facial Expression Recognition in the Wild,” IEEE Trans. Circuits Syst. Video Technol., vol. 34, no. 5, pp. 3192–3203, May 2024, doi: 10.1109/TCSVT.2023.3312858.

[32] Z. Zhao and I. Patras, “Prompting Visual-Language Models for Dynamic Facial Expression Recognition,” in Proc. Brit. Mach. Vis. Conf. (BMVC), 2023.

[33] N. M. Foteinopoulou and I. Patras, “EmoCLIP: A Vision-Language Method for Zero-Shot Video Facial Expression Recognition,” in 2024 IEEE 18th Int. Conf. Autom. Face Gesture Recognit. (FG), IEEE, May 2024, pp. 1–10, doi: 10.1109/FG59268.2024.10581982.

[34] Z. Tao et al., “A<sup>3</sup>lign-DFER: Pioneering Comprehensive Dynamic Affective Alignment for Dynamic Facial Expression Recognition with CLIP,” arXiv preprint arXiv:2403.04294, Mar. 2024.

[35] L. Sun, Z. Lian, B. Liu, and J. Tao, “SVFAP: Self-Supervised Video Facial Affect Perceiver,” IEEE Trans. Affect. Comput., vol. 16, no. 1, pp. 405–422, Jan. 2025, doi: 10.1109/TAFFC.2024.3436913.

[36] L. Sun, Z. Lian, B. Liu, and J. Tao, “HiCMAE: Hierarchical Contrastive Masked Autoencoder for Self-Supervised Audio-Visual Emotion Recognition,” Inf. Fusion, vol. 108, p. 102382, Aug. 2024, doi: 10.1016/j.inffus.2024.102382.

[37] L. Sun, Z. Lian, B. Liu, and J. Tao, “MAE-DFER: Efficient Masked Autoencoder for Self-Supervised Dynamic Facial Expression Recognition,” in Proc. 31st ACM Int. Conf. Multimedia, New York, NY, USA: ACM, Oct. 2023, pp. 6110–6121, doi: 10.1145/3581783.3612365.

[38] Y. Chen, J. Li, S. Shan, M. Wang, and R. Hong, “From Static to Dynamic: Adapting Landmark-Aware Image Models for Facial Expression Recognition in Videos,” IEEE Trans. Affect. Comput., vol. 16, no. 2, pp. 624–638, Apr. 2025, doi: 10.1109/TAFFC.2024.3453443.

[39] Y. Chen et al., “Static for Dynamic: Towards a Deeper Understanding of Dynamic Facial Expressions Using Static Expression Data,” IEEE Trans. Affect. Comput., vol. 17, no. 1, pp. 438–451, Jan. 2026, doi: 10.1109/TAFFC.2025.3623135.

[40] Y. Liu et al., “Clip-Aware Expressive Feature Learning for Video-Based Facial Expression Recognition,” Inf. Sci., vol. 598, pp. 182–195, Jun. 2022, doi: 10.1016/j.ins.2022.03.062.

[41] Y. Liu, W. Wang, C. Feng, H. Zhang, Z. Chen, and Y. Zhan, “Expression Snippet Transformer for Robust Video-Based Facial Expression Recognition,” Pattern Recognit., vol. 138, p. 109368, Jun. 2023, doi: 10.1016/j.patcog.2023.109368.

[42] Z. Tao et al., “Freq-HD: An Interpretable Frequency-based High-Dynamics Affective Clip Selection Method for In-the-Wild Facial Expression Recognition in Videos,” in Proc. 31st ACM Int. Conf. Multimedia, New York, NY, USA: ACM, Oct. 2023, pp. 843–852, doi: 10.1145/3581783.3611972.

[43] F. Ma, B. Sun, and S. Li, “Logo-Former: Local-Global Spatio-Temporal Transformer for Dynamic Facial Expression Recognition,” in ICASSP 2023, IEEE, Jun. 2023, pp. 1–5, doi: 10.1109/ICASSP49357.2023.10095448.

[44] B. Lee, H. Shin, B. Ku, and H. Ko, “Frame Level Emotion Guided Dynamic Facial Expression Recognition with Emotion Grouping,” in 2023 IEEE/CVF Conf. Comput. Vis. Pattern Recog-

nit. Workshops (CVPRW), IEEE, Jun. 2023, pp. 5681–5691, doi: 10.1109/CVPRW59228.2023.00602.

[45] H. Li, H. Niu, Z. Zhu, and F. Zhao, “CLIPER: A Unified Vision-Language Framework for In-the-Wild Facial Expression Recognition,” in 2024 IEEE Int. Conf. Multimedia Expo (ICME), IEEE, Jul. 2024, pp. 1–6, doi: 10.1109/ICME57554.2024.10687508.

[46] M. Li, X. Zhang, T. Liao, S. Lin, and G. Xiao, “PTH-Net: Dynamic Facial Expression Recognition Without Face Detection and Alignment,” IEEE Trans. Image Process., vol. 34, pp. 30–43, 2025, doi: 10.1109/TIP.2024.3504298.

Yudhistira Arditya Pratama received the B.S. degree in Mechatronics Engineering from Politeknik Elektronika Negeri Surabaya (PENS), Surabaya, Indonesia, and the M.S. degree in Electrical Engineering from National Taiwan University of Science and Technology (NTUST), Taipei, Taiwan. He is currently pursuing the Ph.D. degree in Electrical Engineering at NTUST. His research interests include computer vision, dynamic facial expression recognition, spatiotemporal state-space modeling, and underwater image processing.

Yi-Zeng Hsieh (Senior Member, IEEE) received the B.S., M.S., and Ph.D. degrees in computer science and information engineering from National Central University, Taoyuan City, Taiwan, in 2004, 2006, and 2012, respectively. He is currently a Professor with the Department of Electrical Engineering, National Taiwan University of Science and Technology, Taipei, Taiwan. He has authored or coauthored more than 100 journal and refereed conference papers. His research interests include deep learning, machine learning, neural networks, pattern recognition, image processing, computer vision, and swarm intelligence. He is a Senior Member of both the IEEE Consumer Technology Society and the IEEE Signal Processing Society, and was recognized as an Outstanding Young Scholar by the IEEE Consumer Technology Society in 2023.