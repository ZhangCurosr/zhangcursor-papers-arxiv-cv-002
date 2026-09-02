# Mind the Rift: Cross-Scale Coupling Mismatch for AI-Generated Video Detection<sup>∗</sup>

Siyu Li   
School of Cyber Science and   
Engineering   
Sichuan University   
Chengdu, Sichuan, China   
lisiyu\_real@stu.scu.edu.cn   
Jin Yang<sup>†</sup>   
School of Cyber Science and   
Engineering   
Sichuan University   
Chengdu, Sichuan, China   
School of Information Science and   
Technology   
Xizang University   
Lhasa, Xizang, China   
yangjin66@scu.edu.cn   
Weiheng Liang   
School of Cyber Science and   
Engineering   
Sichuan University   
Chengdu, Sichuan, China   
liangweiheng@stu.scu.edu.cn

## Abstract

As AI video generators achieve cinematic realism, reliable detection becomes essential for safeguarding digital trust. We identify cross-scale coupling mismatch as a new forensic signal, where scale refers to the level of abstraction (semantic dynamics vs. pixellevel residuals): in natural videos, macro-level temporal dynamics and micro-level residual patterns are intrinsically coupled by the unified imaging physics pipeline, whereas AI generators, whose training objectives do not explicitly preserve this joint distribution, systematically violate this coupling. Detecting such mismatch is challenging because it requires independently extracting information at both scales while simultaneously quantifying their crossscale relationship. We propose RIFT (Representation Inconsistency Forensics on Trajectories), an orthogonal forensic framework that addresses this through three interlocking components: a macro stream that builds a dynamic baseline of expected temporal evolution via diferential geometry and persistent homology on learned manifold trajectories, a micro stream that acts as a sensitive foren sic probe via steganalytic filtering and temporal modeling, and a coupling divergence module that measures the conditional dependency between the two streams. Gram-Schmidt orthogonality guarantees the information-theoretic validity of this measurement. Experiments on two benchmarks (VidProM, 120K videos, 7 generators; GenVidBench, 68K videos, 4 generators) demonstrate that RIFT achieves 99.33% and 99.72% F1-score respectively, with 97.87% unseen-generator detection rate in leave-one-out evaluation, while exhibiting encoder agnosticism: scaling from ViT-S/14 (22M) to ViT-L/14 (300M) changes F1 by less than 0.1%, and switching to a diferent encoder family (DINOv1) reduces F1 by only 0.73 pp. Code is available at https://github.com/Litsay/RIFT.

• Computing methodologies Computer vision; Neural net-<sup>works;</sup> <sup>•</sup> <sup>Security</sup> <sup>and</sup> <sup>privacy</sup> → <sup>Digital</sup> <sup>rights</sup> <sup>management.</sup>

## Keywords

AI-generated video detection, cross-scale coupling mismatch, video forensics, topological data analysis

## ACM Reference Format:

Siyu Li, Jin Yang, and Weiheng Liang. 2026. Mind the Rift: Cross-Scale Coupling Mismatch for AI-Generated Video Detection. In Proceedings of the 34th ACM International Conference on Multimedia (MM ’26), November 10–14, 2026, Rio de Janeiro, Brazil. ACM, New York, NY, USA, 9 pages. https://doi.org/10.1145/3767308.3835693

## 1 Introduction

Recent video generation models such as Sora [6], Pika, and CogVideo [19] have reached a fidelity level where macro-level semantic errors and temporal inconsistencies are no longer prominent. This progress poses a serious threat to digital trust: synthetic videos can circulate as authentic evidence, erode public confidence in visual media, and enable large-scale misinformation.

Existing detection methods have pursued three largely independent paradigms. Spatial detectors [30, 34, 38, 40] identify artifacts in individual frames but ignore temporal structure. Temporal consistency detectors [10, 28, 42] model frame-to-frame patterns but operate at a single semantic scale. Physics-informed detectors [20, 44] leverage physical priors such as trajectory curvature in pretrained representation spaces, yet extract handcrafted statistics from one scale only. A recent line of work [17] combines macro and micro analysis through a hierarchical dual-path architecture; however, the two branches operate as sequential, independent filters with no interaction, no orthogonality guarantee, and no mechanism to measure cross-scale coupling. All these approaches share a common blind spot: they analyze signals at individual scales, missing the relationship between scales as a forensic signal in its own right.

We make a key observation: in natural videos, macro-level temporal dynamics (trajectory geometry in representation space) and micro-level residual patterns (pixel-domain noise fingerprints) are not independent but intrinsically coupled by the unified imaging physics pipeline that produces both simultaneously. A camera’s sensor noise correlates with its motion response; compression artifacts co-vary with scene complexity. AI generators, by contrast, opti mize for perceptual quality and temporal coherence at individual scales without explicitly preserving their cross-scale coupling. Be cause the joint distribution of macro dynamics and micro residuals is never directly supervised during training, this coupling is systematically broken. We term this phenomenon cross-scale coupling mismatch and characterize it empirically as a robust, generatoragnostic regularity: it manifests as a measurable divergence in the conditional dependency between macro and micro signals (Figure 1) and persists across encoder families, training distributions, and all evaluated generators (Secs. 3.4 and 3.7).

![](images/10fe0ff3409fee321e2530efaf0f42cf5a963ef8ea35521d59541e9a11d13772.jpg)  
Figure 1: Empirical evidence of cross-scale coupling mismatch. Each point represents one video, plotted by its MINEestimated mutual information $\hat { I } ( z ^ { \mathrm { m a c r o } } , z ^ { \mathrm { m i c r o } } )$ (x-axis) and conditional negative log-likelihood log � micro macro (y-axis), measured by a trained RIFT model. Real videos (blue) cluster in the high-MI, low-NLL region (tightly coupled macro dynamics and micro residuals), whereas AI-generated videos (red) scatter toward low MI and high NLL (broken coupling). The gap between the two class centroids (★) constitutes the $\mathbf { \omega } ^ { \omega } \mathbf { r i f t } ^ { s }$ that RIFT exploits as a forensic signal (Cohen’s $d \ = \ 4 . 4 6$ for NLL, $p \ < \ 1 0 ^ { - 3 0 0 }$ ; � = 120K videos from VidProM and DVSC2023). The separation persists across encoder families and unseen generators (Secs. 3.4, 3.7): a robust, generator-agnostic regularity.

To detect this mismatch, we propose RIFT (Representation Inconsistency Forensics on Trajectories), an orthogonal forensic framework comprising three interlocking components. First, orthogonal decoupling projects the shared representation into macro and micro subspaces with Gram-Schmidt constraints, ensuring independent information extraction. The macro stream then builds a dynamic baseline of expected temporal evolution: it learns a low-dimensional manifold embedding [1, 4] on which diferential geometry (curvature, torsion, jerk) and persistent homology characterize the expected dynamic patterns, building on the insight that temporal trajectories in learned representation spaces carry discriminative geometric structure [18]. The micro stream serves as a sensitive forensic probe, extracting noise fingerprints via steganalytic filters and modeling their temporal dynamics. Finally, a coupling divergence module measures � micro macro through conditional likelihood estimation and mutual information, quantifying how far the observed macro-micro relationship deviates from the natural coupling baseline. Crucially, both streams are always active and contribute in parallel, unlike cascade approaches where one branch may be bypassed entirely. While the macro stream provides the structural backbone of detection, the coupling divergence between streams is the distinguishing mechanism: it accounts for the majority of RIFT’s improvement over prior trajectory-based methods, separating the framework from single-scale approaches.

Contributions. Our main contributions are:

• <sup>We</sup> <sup>identify</sup> <sup>cross-scale</sup> <sup>coupling</sup> <sup>mismatch</sup> <sup>as</sup> <sup>a</sup> <sup>forensic</sup> <sup>sig-</sup> nal: natural videos exhibit intrinsic coupling between macrolevel temporal dynamics and micro-level residual patterns governed by imaging physics, which AI generators systematically violate.

We propose RIFT (Representation Inconsistency Forensics on Trajectories), an orthogonal forensic framework: the macro stream builds a dynamic baseline via manifold geometry and persistent homology, the micro stream serves as a forensic probe, and conditional dependency analysis measures coupling divergence with Gram-Schmidt orthogonality guarantees.

We demonstrate encoder agnosticism: within the DINOv2 family, 14 parameter scaling changes F1 by <0.1%; across families (DINOv1 vs. DINOv2), F1 drops by only 0.73 pp, proving that detection stems from framework design rather than encoder specifics.

Experiments on VidProM [39] (7 generators) and GenVid-Bench [29] (4 generators) show RIFT achieves 99.33% and 99.72% F1 respectively, with 97.87% unseen-generator detection in leave-one-out evaluation. The entire framework trains on a single consumer GPU (RTX 3080, 10 GB).

## 2 Method

Given a video � consisting of � frames, our goal is to classify it as either authentic or AI-generated. The RIFT framework proceeds in five stages (Figure 2): (1) frozen backbone encoders extract multimodal frame representations; (2) orthogonal decoupling projects these into independent macro and micro subspaces; (3) the macro stream builds a dynamic baseline characterizing expected temporal evolution; (4) the micro stream extracts fine-grained forensic signals as a sensitive probe; and (5) a coupling divergence module measures � micro macro to quantify cross-scale mismatch. These signals are fused via learned gates for final classification.

## 2.1 Feature Extraction

We employ two frozen pretrained encoders. DINOv2 ViT-S/14 [31] <sup>processes</sup> <sup>each</sup> <sup>frame</sup> <sup>at</sup> <sup>224</sup>×<sup>224</sup> <sup>resolution,</sup> <sup>producing</sup> <sup>CLS</sup> <sup>tokens</sup> $\bar { \mathbf { c } } _ { t } ~ \in ~ \mathbb { R } ^ { 3 8 4 }$ capturing global semantics and patch-level statistics $\pmb { p } _ { t } \in \mathbb { R } ^ { 7 6 8 }$ summarizing local spatial structure. RAFT-Large [35] estimates dense optical flow between consecutive frames, yielding encoded flow vectors $f _ { t } ~ \in \mathbb { R } ^ { 2 5 6 }$ and motion-compensated residual vectors $\boldsymbol { r } _ { t } ~ \in \mathbb { R } ^ { 2 5 6 }$ . Both encoders remain completely frozen throughout training, ensuring that detection performance reflects downstream architectural design rather than encoder adaptation.

![](images/6954620fe98ad2a6a16844658ab7120984ef5ee63c658841753e66c626751dea.jpg)  
Figure 2: Overview of the RIFT framework. Given a video, frozen encoders extract frame-level features, which are orthogonally decoupled into macro and micro subspaces. The macro stream (dynamic baseline) and micro stream (forensic probe) are analyzed independently, and their coupling divergence � micro macro serves as the core forensic signal for classification.

## 2.2 Orthogonal Decoupling

The coupling divergence measurement (Sec. 2.5) requires that macro and micro information be extracted independently; oth erwise, shared information would inflate divergence estimates with self-correlation rather than genuine cross-scale coupling. To enforce this, we project the concatenated frame representations $\left[ { c _ { t } ; f _ { t } } \right]$ and $\left[ \boldsymbol { c } _ { t } ; \boldsymbol { r } _ { t } ; \boldsymbol { p } _ { t } \right]$ through two MLP projectors �<sub>macro</sub> and �<sub>micro</sub>, producing $z _ { t } ^ { \mathrm { m a c r o } } , z _ { t } ^ { \mathrm { m i c r o } } \in \mathbb { R } ^ { d _ { z } } \left( d _ { z } { = } 1 2 8 \right)$

Orthogonality is enforced through dual constraints. A soft loss penalizes the average cosine similarity across the batch:

$$
\mathcal { L } _ { \mathrm { o r t h o } } = \frac { 1 } { B T } \sum _ { b , t } \cos ^ { 2 } ( z _ { b , t } ^ { \mathrm { m a c r o } } , z _ { b , t } ^ { \mathrm { m i c r o } } ) .\tag{1}
$$

A hard constraint applies Gram-Schmidt orthogonalization to the projector weight matrices every 100 training steps, preventing gradual drift.

To ensure that orthogonality does not discard information, a decoder reconstructs the original CLS token from the concatenation $[ z _ { t } ^ { \mathrm { m a c r o } } ; z _ { t } ^ { \mathrm { m i c r o } } ]$

$$
\mathcal { L } _ { \mathrm { r e c o n } } = \frac { 1 } { B T } \sum _ { b , t } \| c _ { b , t } - h _ { \mathrm { d e c } } ( [ z _ { b , t } ^ { \mathrm { m a c r o } } ; z _ { b , t } ^ { \mathrm { m i c r o } } ] ) \| ^ { 2 } .\tag{2}
$$

The combination of orthogonality and reconstruction ensures that macro and micro subspaces partition the CLS representation space non-redundantly, providing a necessary condition for the coupling divergence to reflect genuine cross-scale dependency rather than self-correlation. Reconstruction targets the CLS representation (flow and patch features contribute directly to downstream losses), and the orthogonality guarantee applies to the coupling divergence measurement (Sec. 2.5); the micro stream’s temporal features (Sec. 2.4) are extracted independently from pixel-level residuals.

## 2.3 Macro Stream: Dynamic Baseline

The macro stream characterizes what normal temporal evolution looks like in representation space. It builds a dynamic baseline from two complementary perspectives on the same trajectory.

Manifold Embedding. Computing diferential geometry directly in $\mathbb { R } ^ { d _ { z } }$ is ill-conditioned: cosine similarity variance scales as $1 / d ,$ so at $d _ { z } { = } 1 2 8$ direction changes collapse to near-constant values regardless of actual dynamics $( \mathrm { V a r } [ \cos \theta ] \approx 0 . 0 0 8 )$ . We address this through a learned embedding $\phi : \mathbb { R } ^ { d _ { z } }  \mathbb { R } ^ { d _ { m } }$ (two-layer MLP, GELU):

$$
\pmb { x } _ { t } = \phi ( z _ { t } ^ { \operatorname { m a c r o } } ) , \quad \pmb { x } _ { t } \in \mathbb { R } ^ { d _ { m } } , d _ { m } { = } 2 4 .\tag{3}
$$

At $d _ { m } = 2 4$ <sup>,</sup> <sup>angular</sup> <sup>variance</sup> <sup>increases</sup> <sup>to</sup> ≈<sup>0.042,</sup> <sup>restoring</sup> <sup>discrimi-</sup> native geometric quantities. Beyond resolving angle concentration, $\phi$ learns a coordinate system adapted to temporal dynamics, consistent with the manifold hypothesis [4], and absorbs encoder-specific distributional shifts, contributing to the encoder-agnostic property (Sec. 3.4).

Manifold Regularization. To ensure $\phi$ learns a faithful geometric structure, we regularize it through three complementary objectives. An isometry term preserves local distance ratios during projection, where $\lambda _ { \mathrm { i s o } }$ is a learnable scalar (initialized to 1):

$$
\mathcal { L } _ { \mathrm { i s o } } = \frac { 1 } { T - 1 } \sum _ { t } \left( \frac { \lVert \boldsymbol { x } _ { t + 1 } - \boldsymbol { x } _ { t } \rVert } { \lVert \boldsymbol { z } _ { t + 1 } ^ { \mathrm { m a c r o } } - \boldsymbol { z } _ { t } ^ { \mathrm { m a c r o } } \rVert + \epsilon } - \lambda _ { \mathrm { i s o } } \right) ^ { 2 } ,\tag{4}
$$

A smoothness term, applied exclusively to real samples, penalizes discrete second-order diferences to encode the inductive bias that authentic temporal evolution is smooth on the manifold:

$$
\mathcal { L } _ { \mathrm { s m o o t h } } = \frac { 1 } { | \mathcal { B } _ { \mathrm { r e a l } } | } \sum _ { b \in \mathcal { B } _ { \mathrm { r e a l } } } \frac { 1 } { T - 2 } \sum _ { t } \Vert \pmb { x } _ { t + 1 } - 2 \pmb { x } _ { t } + \pmb { x } _ { t - 1 } \Vert ^ { 2 } .\tag{5}
$$

An InfoNCE contrastive term [36] encourages temporally coherent geometry by clustering adjacent frames (sim: cosine similarity, �=0.15):

$$
\mathcal { L } _ { \mathrm { n c e } } = - \frac { 1 } { T - 1 } \sum _ { t } \log \frac { \exp \bigl ( \sin ( x _ { t } , x _ { t + 1 } ) / \tau \bigr ) } { \sum _ { k \neq t } \exp \bigl ( \sin ( x _ { t } , x _ { k } ) / \tau \bigr ) } ,\tag{6}
$$

The complete manifold loss is $\mathcal { L } _ { \mathrm { m a n i f o l d } } = \mathcal { L } _ { \mathrm { i s o } } + \mathcal { L } _ { \mathrm { s m o o t h } } + \mathcal { L } _ { \mathrm { n c e } } .$

Local: Diferential Geometry. From the embedded trajectory $x _ { 1 } , \ldots , x _ { T - 1 }$ , we compute six groups of diferential geometry quantities. Let $\Delta \boldsymbol { x } _ { t } = \boldsymbol { x } _ { t + 1 } - \boldsymbol { x } _ { t }$ denote the discrete velocity. The Menger curvature [23] at each interior point quantifies local bending via the circumradius of three consecutive points:

$$
\kappa _ { t } = \frac { 2 \sqrt { \| \Delta \pmb { x } _ { t - 1 } \| ^ { 2 } \| \Delta \pmb { x } _ { t } \| ^ { 2 } - ( \Delta \pmb { x } _ { t - 1 } \cdot \Delta \pmb { x } _ { t } ) ^ { 2 } } } { \| \Delta \pmb { x } _ { t - 1 } \| \ \| \Delta \pmb { x } _ { t } \| \ \| \pmb { x } _ { t + 1 } - \pmb { x } _ { t - 1 } \| } ,\tag{7}
$$

where the numerator equals twice the triangle area o $\dot { \mathbf { \Phi } } ( { \pmb x } _ { t - 1 } , { \pmb x } _ { t } , { \pmb x } _ { t + 1 } ) ;$ this definition is valid in any ambient dimension $d _ { m }$ . The discrete torsion measures twisting out of the local osculating plane $\Pi _ { t } = \operatorname { s p a n } ( \Delta \pmb { x } _ { t - 1 } , \Delta \pmb { x } _ { t } )$ :

$$
\tau _ { t } = \frac { \left\| \Delta \pmb { x } _ { t + 1 } - \mathrm { p r o j } _ { \Pi _ { t } } \big ( \Delta \pmb { x } _ { t + 1 } \big ) \right\| } { \left\| \Delta \pmb { x } _ { t } \right\| } ,\tag{8}
$$

capturing the component of the next velocity that is orthogonal to the current osculating plane. The direction change captures the angular deviation between successive velocities:

$$
\delta _ { t } = 1 - \frac { \Delta \pmb { x } _ { t } \cdot \Delta \pmb { x } _ { t + 1 } } { \| \Delta \pmb { x } _ { t } \| \| \Delta \pmb { x } _ { t + 1 } \| } .\tag{9}
$$

This quantity captures the angular regularity of temporal evolution: the distribution of $\delta _ { t }$ values difers systematically between natural and AI-generated videos, providing a discriminative geometric signal [18]. Together with speed ${ \| \Delta \pmb { x } _ { t } \| }$ , acceleration $\| \Delta ^ { 2 } \pmb { x } _ { t } \| _ { 2 }$ and jerk $\| \Delta ^ { 3 } \pmb { x } _ { t } \|$ , these six quantities capture the complete local kinematic profile of the trajectory. Each group is summarized by six statistics (mean, std, max, min, kurtosis, spike ratio), produc ing a 36-dimensional geometry descriptor. A Transformer encoder (4 heads, 2 layers) models temporal patterns over the full curvature sequence, producing a 64-dimensional temporal embedding. The local perspective yields $\pmb { h } _ { \mathrm { l o c a l } } \in \mathbb { R } ^ { 1 0 0 }$

Global: Topological Invariants. The same 24-dimensional trajectory is analyzed via persistent homology [14]. We build a Vietoris– Rips filtration and compute persistence diagrams for $H _ { 0 }$ (connected components) and $H _ { 1 }$ (loops). These diagrams are vectorized into persistence landscapes [7] with 4 functions and 16 bins per dimension $( h _ { \mathrm { g l o b a l } } \in \mathbb { R } ^ { 1 2 8 } )$ . Whereas diferential geometry captures how the trajectory bends locally, topology captures what it is globally: a spiral and a figure-eight may share similar curvature yet difer in $H _ { 1 }$ structure. These invariants are stable under continuous deformations.

The complete dynamic baseline is the concatenation $h _ { \mathrm { m a c r o } } =$ [<sup>�</sup>local<sup>;</sup> <sup>�</sup>global] $\in \mathbb { R } ^ { 2 2 8 }$

## 2.4 Micro Stream: Sensitive Forensic Probe

The micro stream operates independently from the macro stream, extracting fine-grained forensic signals from pixel-level residuals that are invisible in semantic representations.

Noise Residual Extraction. For each frame $I _ { t } ,$ we apply two complementary high-pass filter banks to suppress scene content while preserving generation artifacts. Steganalytic rich model (SRM) filters [15] provide 30 predefined kernels $\{ K _ { i } \} _ { i = 1 } ^ { 3 0 }$ that extract linear and non-linear noise residuals. Bayar constrained convolutions [2] complement these with 16 learned kernels $W _ { j }$ whose weights satisfy:

$$
W _ { j } ( 0 , 0 ) = - \sum _ { ( i , k ) \neq ( 0 , 0 ) } W _ { j } ( i , k ) ,\tag{10}
$$

so that each kernel acts as a prediction error filter: the center pixel is predicted from its spatial neighbors, and only the prediction residual passes through. The constraint is re-enforced after each gradient step by renormalizing the center weight. The combined per-frame forensic feature is $\pmb { u } _ { t } = [ r _ { t } ^ { \mathrm { s r m } } ; r _ { t } ^ { \mathrm { b a y a r } } ; s _ { t } ]$ , where $r _ { t } ^ { \mathrm { s r m } }$ and $r _ { t } ^ { \mathrm { b a y a r } }$ are pooled residual statistics from the respective filter banks and �<sub>�</sub> aggregates spatial and frequency-domain statistics of the motion-compensated residual $r _ { t }$ from RAFT (Sec. 2.1).

Temporal Modeling. A convolutional encoder reduces the dimensionality of each ${ \mathbf { } } \pmb { u } _ { t } .$ , and a bidirectional GRU models the temporal dynamics of the resulting sequence, yielding a 64-dimensional temporal embedding $h _ { \mathrm { t e m p } }$ from the concatenated final hidden states. We append three scalar temporal consistency metrics computed from the frame-to-frame forensic changes $d _ { t } = \| \pmb { u } _ { t + 1 } - \pmb { u } _ { t } \|$ .

$$
\mu _ { d } = \frac { { 1 } } { { T - 1 } } \sum _ { t } d _ { t } , \sigma _ { d } ^ { 2 } = \frac { { 1 } } { { T - 1 } } \sum _ { t } ( d _ { t } - \mu _ { d } ) ^ { 2 } , \rho _ { d } = \frac { { \sum _ { t } } ( d _ { t } - \mu _ { d } ) ( d _ { t + 1 } - \mu _ { d } ) } { { \sum _ { t } } ( d _ { t } - \mu _ { d } ) ^ { 2 } } .\tag{11}
$$

These scalars capture the magnitude, regularity, and serial dependence of forensic signal fluctuations, respectively. The micro stream produces $\pmb { h } _ { \mathrm { m i c r o } } = [ \pmb { h } _ { \mathrm { t e m p } } ; \mu _ { d } ; \sigma _ { d } ^ { 2 } ; \rho _ { d } ] \in \mathbb { R } ^ { 6 7 }$

## 2.5 Coupling Divergence Measurement

This module is the distinguishing mechanism of RIFT: it measures the cross-scale coupling between the macro baseline and the micro probe. The hypothesis is that in natural videos, the macro temporal state constrains the distribution of micro observations through shared imaging physics; in AI-generated videos, this constraint is violated.

Conditional Prediction. An MLP predicts the expected distribution of micro statistics given the macro state:

$$
\begin{array} { r } { \mu _ { t } , \log \sigma _ { t } ^ { 2 } = f _ { \mathrm { c o n d } } ( z _ { t } ^ { \mathrm { m a c r o } } ) , \quad \mathrm { N L L } _ { t } = \frac { 1 } { 2 } \big ( \log \sigma _ { t } ^ { 2 } + ( s _ { t } - \mu _ { t } ) ^ { 2 } / \sigma _ { t } ^ { 2 } \big ) , } \end{array}\tag{12}
$$

where $\pmb { s } _ { t }$ denotes the micro statistics sequence from the micro stream. The per-video NLL statistics (mean, max, std) are projected to a 16-dimensional feature vector, encoding how well the macro state predicts micro behavior: low NLL indicates strong coupling, while high NLL indicates mismatch.

Mutual Information Estimation. A MINE network [3] estimates the mutual information $I ( z ^ { \mathrm { m a c r o } } ; z ^ { \mathrm { m i c r o } } )$ by contrasting the joint distribution with a shufled marginal:

$$
\hat { I } = \mathbb { E } _ { \mathrm { j o i n t } } \big [ T _ { \theta } \big ( z ^ { \mathrm { m a c r o } } , z ^ { \mathrm { m i c r o } } \big ) \big ] - \log \mathbb { E } _ { \mathrm { m a r g i n a l } } \big [ e ^ { T _ { \theta } ( z ^ { \mathrm { m a c r o } } , z ^ { \mathrm { m i c r o } } ) } \big ] ,\tag{13}
$$

where $z ^ { \mathrm { m i c r o } }$ is obtained by temporal shufling. This scalar quantifies the residual statistical dependency between the orthogonal subspaces.

Why Orthogonality Matters. Without the orthogonal guarantee from Sec. 2.2, the coupling divergence degenerates: shared information between �<sup>macro</sup> and �<sup>micro</sup> would produce artificially high MI and artificially low NLL regardless of the input’s authenticity. Orthogonality ensures that any measured coupling reflects genuine cross-scale dependency, which is precisely the signal that AI generators fail to replicate.

The module outputs $\pmb { h } _ { \mathrm { c o n d } } \in \mathbb { R } ^ { 1 7 }$ (16-dimensional projected NLL features concatenated with the 1-dimensional MI estimate).

## 2.6 Gated Fusion and Classification

Two projection heads map the streams to a common 256D space. The macro path projects $\pmb { h } _ { \mathrm { m a c r o } } \in \mathbb { R } ^ { 2 2 8 }$ and the micro path projects $[ \pmb { h } _ { \mathrm { m i c r o } } ; \pmb { h } _ { \mathrm { c o n d } } ] \in \mathbb { R } ^ { 8 4 }$ . Learned sigmoid gates $g _ { \mathrm { m a c r o } } , g _ { \mathrm { m i c r o } } \in \left( 0 , 1 \right)$ weight the two paths:

$$
\begin{array} { r } { \pmb { h } _ { \mathrm { f u s e d } } = g _ { \mathrm { m a c r o } } \cdot \pmb { h } _ { \mathrm { m a c r o } } ^ { \prime } + g _ { \mathrm { m i c r o } } \cdot \pmb { h } _ { \mathrm { m i c r o } } ^ { \prime } . } \end{array}\tag{14}
$$

The gates are independent (not softmax-normalized), allowing the model to upweight both streams simultaneously when both provide reliable signals. A 3-layer MLP classifier $( 5 1 2  2 5 6  1 2 8  2 )$ produces the final binary prediction.

## 2.7 Training Strategy

Training proceeds in two stages to stabilize the orthogonal decomposition before joint optimization.

Stage 1: Decoupling Pretraining (30 epochs). Only the orthogonal decoupling module and its associated manifold regularization are trained, with learning rate $1 0 ^ { - 3 }$ . This establishes well-separated macro and micro subspaces before downstream modules are introduced.

Stage 2: Joint Training (50 epochs). All trainable modules are optimized jointly with learning rate $3 \times 1 0 ^ { - 4 }$ , cosine annealing with warm restarts $( T _ { 0 } { = } 5 , T _ { \mathrm { m u l t } } { = } 2 )$ , and 5% linear warmup. The total loss combines five components:

$$
\mathcal { L } = \mathcal { L } _ { \mathrm { f o c a l } } + \alpha \mathcal { L } _ { \mathrm { o r t h o } } + \beta \mathcal { L } _ { \mathrm { r e c o n } } + \gamma \mathcal { L } _ { \mathrm { m a n i f o l d } } + \delta \mathcal { L } _ { \mathrm { c o n d } } ,\tag{15}
$$

where $\mathcal { L } _ { \mathrm { f o c a l } }$ is the focal loss $[ 2 5 ] \ ( \gamma _ { \mathrm { f o c a l } } \mathrm { = } 2 . 0 ) , \mathcal { L } _ { \mathrm { m a n i f o l d } }$ is the manifold regularization defined in Sec. 2.3 (Eqs. 4–6), and $\mathcal { L } _ { \mathrm { c o n d } }$ is the conditional NLL applied only to real samples to learn the natural coupling distribution. Weights: �=0.1, �=0.5, �=0.02, �=0.05. We use AdamW [27] with weight decay 0.01 and gradient clipping at norm 1.0. Backbone parameters remain frozen; only 4.7M trainable parameters are updated.

## 3 Experiments

## 3.1 Experimental Setup

Dataset. We construct an evaluation set of 120K videos from VidProM [39] and DVSC2023 [32], the same public sources used by prior work [17, 20]. The real partition consists of 50K videos from DVSC2023, capturing diverse real-world content. The fake partition contains 70K videos generated by seven text-to-video models from VidProM: Pika [22], VideoCrafter2 [11], Text2Video-Zero [21], ModelScope [37], CogVideo [19], OpenSora [46], and Show-1 ( 10K each). Prior VidProM evaluations typically include 4 generators; we expand coverage to all 7 available generators and adopt a 50/50 stratified train/test split with a fixed seed, holding out 10% of the training partition for validation. All baselines in Table 1 are re-evaluated under this same protocol; image-level detectors are aggregated to video-level predictions by majority voting over frame-level decisions.

We additionally evaluate on GenVidBench [29], an independent benchmark of 68K videos spanning 4 generators (CogVideo, Mora, MuseV, SVD [5]) with Main (general scenes) and Plants (highrandomness) subsets, using the same 50/50 protocol.

Implementation. All features are precomputed and cached: DINOv2 ViT-S/14 CLS tokens 32, 384 , patch statistics 32, 768 , RAFT-encoded flow 31, 256 , and motion-compensated resid-<sup>uals</sup> (<sup>31,</sup> <sup>256</sup>) <sup>per</sup> <sup>video,</sup> <sup>totaling</sup> ∼<sup>20 GB.</sup> <sup>Topology</sup> <sup>features</sup> are precomputed via Vietoris-Rips persistent homology and cached separately. Training uses a single NVIDIA RTX 3080 (10 GB). The two-stage training procedure, all hyperparameters, and loss weights are detailed in Appendix A. Source code, pretrained models, and Seedance 2.0 demo videos are available at https://github.com/Litsay/RIFT.

Metrics. We report Accuracy, Precision, Recall, F1-score, and AUC. F1 is the primary metric for model selection.

## 3.2 Main Results and Cross-Generator Generalization

Comparison with Prior Methods. Table 1 compares RIFT against 10 baselines from four paradigms, all reproduced under identical conditions (Sec. 3.1).

Image-level detectors [13, 26, 30, 38] remain near chance level on both benchmarks (F1 67%), confirming that per-frame analysis alone cannot capture the temporal structure essential for video-level detection. Temporal detectors [10, 16, 42] improve substantially but plateau below 93% F1. The strongest prior results come from physicsinformed methods (ReStraV [20]: 97.83% F1 on VidProM, 95.88% F1 on GenVidBench) and multi-branch approaches (MPF-Net [17]: 97.03% F1 on GenVidBench).

RIFT achieves state-of-the-art results on both benchmarks: 99.33% F1 on VidProM with 7 generators ( 1.50 pp over ReStraV) and 99.72% F1 on GenVidBench ( 2.69 pp over MPF-Net). Pergenerator detection rates range from 100% (T2VZero, VideoCrafter2, ModelScope) to 97.24% (CogVideo) on VidProM, with full breakdowns in Appendix B.

Cross-Generator Generalization. To evaluate generalization to unseen generators, we conduct Leave-One-Generator-Out (LOGO) experiments across all 7 generators: one generator is entirely excluded from training, and the model is evaluated on all its test samples. Table 2 shows that RIFT maintains an average unseen-generator detection rate of 97.87%. Four generators (T2VZero, VideoCrafter2, ModelScope, Show-1) are detected at 99.4%, while the hardest cases (OpenSora 94.91%, CogVideo 95.47%, Pika 95.50%) still exceed 94%.

## 3.3 Ablation Study

We ablate each conceptual component of RIFT on VidProM by zeroing its features at the fusion input while retraining Stage 2 from a shared Stage 1 checkpoint. Table 3 reports results on the held-out test set.

Table 1: Comparison with prior methods on VidProM [39] and GenVidBench [29]. All methods are reproduced from open-source implementations under identical conditions on both benchmarks (7 generators, 50/50 stratified split on Vid-ProM; 50/50 split on GenVidBench).
<table><tr><td rowspan="2">Method</td><td colspan="3">VidProM</td><td colspan="3">GenVidBench</td></tr><tr><td>Acc</td><td>F1</td><td>AUC</td><td>Acc</td><td>F1</td><td>AUC</td></tr><tr><td>Image-level detectors</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>CNNDetection [38]</td><td>50.41</td><td>53.82</td><td>52.63</td><td>55.81</td><td>53.31</td><td>57.94</td></tr><tr><td>UnivFD [30]</td><td>51.27</td><td>55.80</td><td>57.62</td><td>58.56</td><td>60.04</td><td>60.92</td></tr><tr><td>FreDect [13]</td><td>60.98</td><td>59.73</td><td>63.56</td><td>59.36</td><td>60.73</td><td>61.25</td></tr><tr><td>Gram-Net [26]</td><td>63.64</td><td>66.75</td><td>64.22</td><td>66.59</td><td>65.50</td><td>68.22</td></tr><tr><td>Temporal video detectors</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>TALL [42]</td><td>73.89</td><td>75.66</td><td>77.31</td><td>77.36</td><td>78.58</td><td>81.06</td></tr><tr><td>STIL [16]</td><td>75.42</td><td>75.28</td><td>76.96</td><td>74.92</td><td>73.43</td><td>76.33</td></tr><tr><td>DeMamba [10]</td><td>93.58</td><td>92.74</td><td>95.58</td><td>92.78</td><td>91.37</td><td>94.64</td></tr><tr><td>Physics-informed &amp; trajectory-based</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ReStraV [20]</td><td>97.06</td><td>97.83</td><td>98.39</td><td>96.93</td><td>95.88</td><td>98.24</td></tr><tr><td>NSG-VD [44]</td><td>95.37</td><td>96.41</td><td>96.37</td><td>93.27</td><td>94.98</td><td>94.63</td></tr><tr><td>Multi-branch</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>MPF-Net [17]</td><td>95.13</td><td>94.37</td><td>96.39</td><td>96.35</td><td>97.03</td><td>97.34</td></tr><tr><td>RIFT (Ours)</td><td>99.22</td><td>99.33</td><td>99.96</td><td>99.56</td><td>99.72</td><td>99.96</td></tr></table>

Table 2: Leave-One-Generator-Out generalization on Vid-ProM across all 7 generators. “Unseen” denotes the detection rate on the excluded generator’s test samples.
<table><tr><td>Excluded Generator</td><td>Unseen Det. Rate</td><td>Seen Test F1</td><td>Seen Test AUC</td></tr><tr><td>T2VZero</td><td>100.00%</td><td>98.77%</td><td>99.88%</td></tr><tr><td>VideoCrafter2</td><td>100.00%</td><td>98.91%</td><td>99.89%</td></tr><tr><td>ModelScope</td><td>99.78%</td><td>98.86%</td><td>99.89%</td></tr><tr><td>Show-1</td><td>99.40%</td><td>98.84%</td><td>99.89%</td></tr><tr><td>Pika</td><td>95.50%</td><td>98.95%</td><td>99.92%</td></tr><tr><td>CogVideo</td><td>95.47%</td><td>99.54%</td><td>99.96%</td></tr><tr><td>OpenSora</td><td>94.91%</td><td>98.95%</td><td>99.91%</td></tr><tr><td>Average</td><td>97.87%</td><td>98.97%</td><td>99.91%</td></tr></table>

Table 3: Stream-level ablation on VidProM (50/50 test set, 60K samples). “Macro only” retains both local geometry and global topology; “Micro only” retains only the micro stream.
<table><tr><td>Configuration</td><td>Acc</td><td>F1</td><td>AUC</td><td>∆F1</td></tr><tr><td>Full RIFT</td><td>99.22</td><td>99.33</td><td>99.96</td><td>一</td></tr><tr><td>w/o Geometry</td><td>97.78</td><td>97.96</td><td>98.89</td><td>-1.37</td></tr><tr><td>w/o Coupling</td><td>97.85</td><td>98.01</td><td>98.90</td><td>-1.32</td></tr><tr><td>Macro only</td><td>97.82</td><td>97.99</td><td>98.88</td><td>-1.34</td></tr><tr><td>Micro only</td><td>80.89</td><td>83.54</td><td>89.50</td><td>-15.79</td></tr></table>

Three findings stand out. (1) The macro stream is the structural backbone: the macro stream alone (Macro only) achieves 97.99% F1, while the micro stream in isolation reaches only 83.54% F1 <sup>(</sup>−<sup>15.79</sup> <sup>pp),</sup> <sup>confirming</sup> <sup>that</sup> <sup>manifold</sup> <sup>geometry</sup> <sup>and</sup> <sup>topology</sup> <sup>carry</sup> the majority of discriminative information. Notably, w/o Geometry (97.96%) performs slightly below Macro only (97.99%) despite retaining the micro stream and coupling module, indicating that coupling depends on a strong macro baseline: without local geometry, the macro state is too weak for reliable conditional prediction. (2) Coupling divergence is the key complementary signal: removing only the divergence module reduces F1 by 1.32 pp, nearly as large as the gap between Full and Macro only ( 1.34 pp). This means the micro stream’s independent contribution beyond coupling is only 0.02 pp; virtually all of its value flows through the conditional dependency � micro  macro , directly validating the coupling mismatch hypothesis. Moreover, since Macro only (97.99%) barely exceeds the strongest prior method (ReStraV, 97.83%), the coupling mechanism accounts for 88% of RIFT’s total improvement over the state of the art ( 1.32 of 1.50 pp), confirming its role as the distinguishing component of the framework. Within the macro stream, adding topological invariants to local diferential geometry further improves F1, confirming that persistent homology captures complementary global structure.

Table 4: Encoder agnosticism on VidProM. Within the DI-NOv2 family, scaling from 22M to 300M parameters changes F1 by only 0.02%. Across families (DINOv1 vs. DINOv2), F1 drops by just 0.73 pp.
<table><tr><td>Family</td><td>Encoder</td><td>Params</td><td>Acc</td><td>F1</td><td>AUC</td></tr><tr><td>DINOv2</td><td>ViT-S/14</td><td>22M</td><td>99.22%</td><td>99.33%</td><td>99.96%</td></tr><tr><td></td><td>ViT-L/14</td><td>300M</td><td>99.28%</td><td>99.35%</td><td>99.98%</td></tr><tr><td>DINOv1</td><td>ViT-S/16</td><td>21M</td><td>98.37%</td><td>98.60%</td><td>99.84%</td></tr></table>

## 3.4 Encoder Agnosticism and Further Analysis

Encoder Ablation. A key property of RIFT is encoder agnosticism: detection remains strong regardless of encoder capacity or architecture family (Table 4). Within the DINOv2 family, scaling from ViT-S/14 (22M) to ViT-L/14 (300M) changes F1 by only 0.02%. Across families, replacing DINOv2 ViT-S/14 with DINOv1 [9] ViT-S/16 (a diferent pretraining method, patch size, and training data) reduces F1 by just 0.73 pp (99.33% 98.60%) while AUC drops only <sup>0.12</sup> <sup>pp</sup> <sup>(99.96%</sup> → <sup>99.84%).</sup> <sup>This</sup> <sup>demonstrates</sup> <sup>that</sup> <sup>the</sup> <sup>coupling</sup> mismatch signal is captured by the framework’s orthogonal decomposition and geometric analysis, not by any specific encoder’s representation power. The manifold embedding � (Sec. 2.3) absorbs encoder-specific distributional shifts, enabling consistent geometric analysis across architecturally diverse backbones.

Handcrafted Feature Integration. To test whether handcrafted geometry features provide complementary information, we append the 21-dimensional curvature-and-distance statistics from [20] to the macro stream input. F1 drops by 0.63 pp, indicating that RIFT’s learned manifold geometry strictly subsumes these handcrafted descriptors, and their addition introduces redundancy into the fusion layer.

Eficiency. Table 5 reports module-level latency on a single RTX 3080 GPU. The macro stream’s Transformer encoder dominates at

Table 5: Module-level latency profiling (precomputed features).
<table><tr><td>Module</td><td>Latency (ms)</td><td> $\% \mathrm { T o t a l }$ </td></tr><tr><td>Orthogonal Decoupling</td><td>1.6</td><td>3.4</td></tr><tr><td>Macro Stream (Transformer)</td><td>40.4</td><td>86.0</td></tr><tr><td>Micro Stream (SRM+GRU)</td><td>1.9</td><td>4.1</td></tr><tr><td>Coupling Divergence</td><td>2.2</td><td>4.6</td></tr><tr><td>Gated Fusion</td><td>0.9</td><td>2.0</td></tr><tr><td>Post-backbone Total</td><td>47.0</td><td>100.0</td></tr></table>

![](images/8bb7f9413820219225fc49f8a456f2382438bbfec735e30c8dcb02cf3c656cf9.jpg)

![](images/f987f60d5a528d7da563356f87e10fab074e0d7b22afa1c14b5b220f93097a13.jpg)  
Figure 3: Interpretability of RIFT. (a) Learned gate weights: � remains near-saturated for both classes, while $g _ { \mathrm { m i c r o } }$ drops 27% for AI-generated videos, showing autonomous suppression of the less reliable probe. (b) Mean Menger curvature: real trajectories exhibit 52% higher curvature on the learned manifold, indicating that natural videos produce richer geometric dynamics while AI-generated videos exhibit overly uniform trajectories.

86% of post-backbone time. Including backbone extraction, end-toend latency is 182 ms (ViT-S/14) or 461 ms (ViT-L/14), with 4.7M trainable parameters.

## 3.5 Interpretability

Coupling Mismatch Validation. To directly validate the coupling mismatch hypothesis, we collect per-sample mutual information $\hat { I } ( z ^ { \mathrm { m a c r o } } , z ^ { \mathrm { m i c r o } } )$ and conditional NLL from the trained model across all 120K videos. Real videos exhibit significantly higher MI $( - 0 . 4 5 \pm$ 0.68) than AI-generated videos ( 6.25 5.95), with Cohen’ $\ s d = 1 . 3 7$ and $\rho < 1 0 ^ { - 3 0 0 }$ (Mann-Whitney U). (Negative MI estimates reflect MINE’s loose variational bound at low true MI, but relative class separation remains valid.) The conditional NLL shows an even stronger separation: real $- 1 . 0 9 \pm 0 . 3 0$ versus fake $+ 0 . 3 4 \pm 0 . 3 4$ (Cohen’s � = 4.46). The orthogonality check confirms that cos $( z ^ { \mathrm { m a c r o } } , z ^ { \mathrm { m i c r o } } )$ ≈ 0.001 for both classes, ruling out information leakage. This separation is consistent across all 7 generators (see Appendix D), indicating that coupling mismatch is a robust, generator-agnostic regularity of current AI video generation rather than an artifact of specific architectures. Figure 1 visualizes this result.

Gate Weight Analysis. The macro gate remains near-saturated for both classes $( g _ { \mathrm { m a c r o } } \approx 0 . 9 6$ ; Figure 3a), while the micro gate drops from 0.80 (real) to 0.58 (fake), a 27% relative suppression. This emergent asymmetry provides independent evidence for the coupling mismatch hypothesis: the model autonomously downweights the micro probe when cross-scale coupling is broken.

Trajectory Geometry. The Menger curvature of learned manifold trajectories shows a clear distributional shift (Figure 3b): real videos exhibit 52% higher mean curvature than AI-generated videos $( 1 8 . 9 9 \pm 8 . 1 2 \mathrm { v s . } 1 2 . 4 9 \pm 6 . 7 3 )$ . This indicates that natural videos produce richer, more structured temporal dynamics on the learned manifold, while AI-generated videos exhibit overly uniform trajectories characteristic of difusion-based synthesis. Yet the substantial distributional overlap confirms that coupling divergence, not individual geometry, provides the decisive signal.

## 3.6 Robustness Analysis

We evaluate RIFT on VidProM under 17 degradation conditions simulating real-world post-processing: H.264 and H.265 re-encoding (500 kbps to 8 Mbps), Gaussian noise $( \sigma = 5 { - } 2 0 )$ , and spatial cropping (70%–90%). Features are re-extracted from degraded videos and evaluated with the frozen model.

RIFT achieves peak robustness under H.265 re-encoding at 2 Mbps $( \mathrm { F } 1 = 8 8 . 1 9 \% )$ and tolerates Gaussian noise well at most levels (F1 > 88% at $\sigma \in \{ 5 , 1 5 , 2 0 \} ,$ ), though performance varies nonmonotonically across degradation intensities on the 996-sample evaluation subset. Performance degrades under aggressive spatial cropping $\mathrm { ( F 1 < 0 . 7 7 }$ at 80% crop), because cropping alters the spatial context on which patch statistics and optical flow are computed. Full per-generator results are provided in Appendix C.

## 3.7 Case Study: Zero-Shot Detection of Seedance 2.0

To evaluate whether the coupling mismatch hypothesis generalizes beyond the training distribution, we conduct a zero-shot case study on Seedance 2.0 [8], a commercial video generation model released by ByteDance in February 2026 that was entirely absent from our training data. We collect 27 videos spanning diverse content categories and evaluate them with the frozen RIFT model without any fine-tuning or adaptation.

Detection and Coupling Signals. RIFT correctly classifies all 27 videos (100% detection rate; binomial 95% CI [86.8%, 100%], �=27; mean confidence 97.5%, range 73.4%–100.0%) despite zero exposure to this generator during training. Comparing the coupling signals against the seven training generators, Seedance 2.0 exhibits even weaker cross-scale coupling in MI $( \hat { I } = - 1 7 . 7 4 $ vs. fake avg 6.25) and micro gate suppression $( g _ { \mathrm { m i c r o } } = 0 . 4 9 \ \mathrm { v s } . \ 0 . 5 8 )$ , while its conditional NLL ( 0.30) falls within the fake distribution $\left( + 0 . 3 4 \pm 0 . 3 4 \right)$ The extreme MI value may partially reflect MINE estimator instability on out-of-distribution inputs and should be read as a relative indicator of coupling breakdown.

Per-Frame Analysis. Figure 4 presents per-frame analysis of a representative video against a real-video baseline $( \mu \pm \sigma ,$ 30 videos). The Menger curvature $\kappa _ { t }$ lies consistently below the real band (mean 10.28 vs. 18.99), consistent with the overly uniform trajectory pattern observed across AI-generated videos. The conditional NLL reveals localized coupling breakdowns, and the direction change $\delta _ { t }$ shows unnaturally uniform temporal dynamics characteristic of difusion-based generators.

![](images/dc44c330559c56394a400ae7e43e7d5f9f9e97a62b3facf778522a319bad452f.jpg)  
Figure 4: Per-frame analysis of a Seedance 2.0 video (street musician). Top: sampled frames. Bottom three rows: Menger curvature $\kappa _ { t } ,$ conditional NLL, and direction change $\delta _ { t s }$ each compared against the real-video baseline $( \mu \pm \sigma ,$ blue). Seedance 2.0 exhibits lower curvature (overly uniform dynamics), elevated NLL (coupling breakdown), and uniform $\delta _ { t }$ (unnatural regularity).

## 4 Related Work

## 4.1 AI-Generated Image and Video Detection

Detection of AI-generated visual content has evolved through three largely independent paradigms. Spatial detectors [30, 34, 38, 40] identify per-frame artifacts in frequency, feature, or reconstruction space but discard temporal structure. Temporal consistency detectors, most recently DeMamba [10], model frame-to-frame patterns but operate at a single semantic scale [16, 28, 42, 43]. Physics-informed detectors such as NSG-VD [44] and ReStraV [20] leverage physical priors such as spatiotemporal gradients or trajectory curvature [18], yet extract handcrafted statistics from one scale only. Orthogonally, recent eforts address difusion-specific artifacts [12], cross-domain generalization to unseen forgery methods [41], grounded artifact reasoning [24], and training-free detection via second-order temporal features [45]. All these paradigms analyze signals at individual scales; RIFT instead targets the cross-scale coupling relationship itself.

## 4.2 Hierarchical and Multi-Branch Detection

Recent work has begun to combine macro and micro analysis. Shuai et al. [33] propose a two-stream network that fuses localization and verification branches through multi-scale collaborative learning. MPF-Net [17] introduces a hierarchical dual-path framework with a manifold sentinel branch and a micro-temporal fluctuation branch, while Zhou et al. [47] demonstrate that frozen VFMs inherently pos sess of-manifold detection capabilities. Three fundamental distinctions separate these approaches from RIFT: (1) MPF-Net’s branches operate sequentially and independently with no cross-branch interaction, whereas RIFT runs both streams in parallel with orthogonal constraints; (2) each branch answers a self-contained question, whereas RIFT measures the coupling divergence � micro  macro as the core forensic evidence; and (3) MPF-Net’s performance varies across VFM backbones, whereas RIFT exhibits encoder agnosticism (<0.1% within DINOv2, 0.73 pp across encoder families).

## 4.3 Topological Data Analysis in Temporal Modeling

Persistent homology [14] provides algebraic invariants that characterize the shape of data across scales: connected components $\left( H _ { 0 } \right)$ capture fragmentation and merging, while loops (�<sub>1</sub>) capture cyclic structures. Persistence landscapes [7] ofer a stable, vectorized summary compatible with machine learning pipelines. While topological data analysis has been applied to time series classification and shape recognition, to our knowledge RIFT is the first to apply persistent homology to AI-generated video detection. In our framework, Vietoris–Rips filtrations on learned manifold trajectories complement local diferential geometry with global structural invariants (Sec. 2.3).

## 5 Conclusion

We have presented RIFT, an orthogonal forensic framework built on a central hypothesis: AI video generators, by optimizing perceptual quality at individual scales without explicit cross-scale supervision, systematically break the cross-scale coupling that the unified imaging physics pipeline imposes on natural videos. This coupling mismatch constitutes a structural limitation of current generative architectures, because the joint distribution across scales is never explicitly optimized during generation. Our framework detects this mismatch through orthogonal macro-micro decomposition, a geometric and topological dynamic baseline, an independent forensic probe, and the coupling divergence between the two streams.

Experimentally, RIFT achieves 99.33% F1 on VidProM (7 generators) and 99.72% F1 on GenVidBench (4 generators), with 97.87% unseen-generator detection rate in leave-one-out evaluation. A key <sup>property</sup> <sup>is</sup> <sup>encoder</sup> <sup>agnosticism:</sup> <sup>14</sup>× <sup>parameter</sup> <sup>scaling</sup> <sup>within</sup> <sup>DI-</sup> NOv2 changes F1 by <0.1%, and switching to DINOv1 reduces F1 by only 0.73 pp. With only 4.7M trainable parameters and frozen pretrained encoders, the entire pipeline trains on a single consumer GPU (RTX 3080, 10 GB) and achieves 182 ms end-to-end inference latency.

Limitations and Future Work. RIFT exhibits fragility under aggressive spatial cropping, which alters the spatial context on which patch statistics and optical flow are computed; spatial augmentation during feature extraction (random cropping) is a direct remedy. The macro Transformer accounts for 86% of post-backbone compute, motivating distillation into a compact student model. Finally, broader evaluation on next-generation architectures beyond the Seedance 2.0 case study (Sec. 3.7), including Sora [6] and Kling, will further assess the generality of the coupling mismatch hypothesis.

## Acknowledgments

This work was supported by the National Natural Science Foundation of China (Grants No. 62162057 and No. 61872254), by the Key Laboratory of Data Protection and Intelligent Management of the Ministry of Education, China (Grant No. SCUSAKFKT202402Y), and by the Open Program of Shaanxi Key Laboratory of Intelligent Policing (Independent Research Topic, Grant No. SXZJ25KF08).

## References

[1] Georgios Arvanitidis, Lars Kai Hansen, and Søren Hauberg. 2018. Latent Space Oddity: on the Curvature of Deep Generative Models. In International Conference on Learning Representations (ICLR).

[2] Belhassen Bayar and Matthew C Stamm. 2018. Constrained convolutional neural networks: A new approach towards general purpose image manipulation detection. IEEE Transactions on Information Forensics and Security 13, 11 (2018), 2691–2706.

[3] Mohamed Ishmael Belghazi, Aristide Baratin, Sai Rajeswar, Sherjil Ozair, Yoshua Bengio, Aaron Courville, and R Devon Hjelm. 2018. Mutual information neural estimation. In International Conference on Machine Learning (ICML). 531–540.

[4] Yoshua Bengio, Aaron Courville, and Pascal Vincent. 2013. Representation Learning: A Review and New Perspectives. IEEE Transactions on Pattern Analysis and Machine Intelligence 35, 8 (2013), 1798–1828

[5] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, Varun Jampani, and Robin Rombach. 2023. Stable Video Difusion: Scaling Latent Video Difusion Models to Large Datasets. arXiv preprint arXiv:2311.15127 (2023).

[6] Tim Brooks, Bill Peebles, et al. 2024. Video generation models as world simulators. OpenAI Technical Report (2024).

[7] Peter Bubenik. 2015. Statistical topological data analysis using persistence land scapes. Journal ofMachine Learning Research 16, 3 (2015), 77–102.

[8] ByteDance. 2026. Seedance 2.0. Commercial video generation model, released February 2026. https://seed.bytedance.com/zh/blog/seedance-2-0-%E6%AD% A3%E5%BC%8F%E5%8F%91%E5%B8%83

[9] Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jégou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. 2021. Emerging Properties in Self-Supervised Vision Transformers. In IEEE/CVF International Conference on Computer Vision (ICCV). 9650–9660.

[10] Haoxing Chen, Yan Hong, Zizheng Huang, Zhuoer Xu, Zhangxuan Gu, Yaohui Li, Jun Lan, Huijia Zhu, Jianfu Zhang, Weiqiang Wang, and Huaxiong Li. 2026. DeMamba: AI-Generated Video Detection on Million-Scale GenVideo Benchmark. Science China Information Sciences 69 (2026). doi:10.1007/s11432-024-4894-0

[11] Haoxin Chen, Yong Zhang, Xiaodong Cun, Menghan Xia, Xintao Wang, Chao Weng, and Ying Shan. 2024. VideoCrafter2: Overcoming data limitations for high-quality video difusion models. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 7310–7320.

[12] Harry Cheng, Yangyang Guo, Tianyi Wang, Liqiang Nie, and Mohan Kankanhalli. 2024. Difusion Facial Forgery Detection. In Proceedings of the 32nd ACM International Conference on Multimedia. doi:10.1145/3664647.3680797

[13] Chandler Timm Doloriel and Ngai-Man Cheung. 2024. Frequency Masking for Universal Deepfake Detection. In IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). 13466–13470.

[14] Herbert Edelsbrunner and John Harer. 2010. Computational topology: an introduction. American Mathematical Society.

[15] Jessica Fridrich and Jan Kodovsky. 2012. Rich models for steganalysis of digital images. IEEE Transactions on Information Forensics and Security 7, 3 (2012), 868–882.

[16] Zhihao Gu, Yang Chen, Taiping Yao, Shouhong Ding, Jilin Li, Feiyue Huang, and Lizhuang Ma. 2021. Spatiotemporal Inconsistency Learning for DeepFake Video Detection. In Proceedings ofthe 29th ACM International Conference on Multimedia. 3473–3481. doi:10.1145/3474085.3475508

[17] Xinan He, Kaiqing Lin, Yue Zhou, Jiaming Zhong, Wei Ye, Wenhui Yi, Bing Fan, Feng Ding, Haodong Li, Bo Cao, and Bin Li. 2026. MPF-Net: Exposing High Fidelity AI-Generated Video Forgeries via Hierarchical Manifold Deviation and Micro-Temporal Fluctuations. arXiv preprint arXiv:2601.21408 (2026).

[18] Olivier J Hénaf, Robbe LT Goris, and Eero P Simoncelli. 2019. Perceptual straight ening of natural videos. Nature Neuroscience 22, 6 (2019), 984–991.

[19] Wenyi Hong, Ming Ding, Wendi Zheng, Xinghan Liu, and Jie Tang. 2023. CogVideo: Large-scale pretraining for text-to-video generation via transformers. In International Conference on Learning Representations (ICLR).

[20] Christian Internò, Robert Geirhos, Markus Olhofer, Sunny Liu, Barbara Hammer, and David Klindt. 2025. AI-Generated Video Detection via Perceptual Straight ening. In Advances in Neural Information Processing Systems (NeurIPS).

[21] Levon Khachatryan, Andranik Movsisyan, Vahram Tadevosyan, Roberto Hen schel, Zhangyang Wang, Shant Navasardyan, and Humphrey Shi. 2023. Text2Video-Zero: Text-to-Image Difusion Models are Zero-Shot Video Gen erators. In IEEE/CVF International Conference on Computer Vision (ICCV). 15954– 15964.

[22] Pika Labs. 2024. Pika. Commercial video generation platform. https://pika.art

[23] Jean-Christophe Léger. 1999. Menger curvature and rectifiability. Annals of Mathematics 149, 3 (1999), 831–869.

[24] Yifei Li, Wenzhao Zheng, Yanran Zhang, Runze Sun, Yu Zheng, Lei Chen,Jie Zhou, and Jiwen Lu. 2026. Skyra: AI-Generated Video Detection via Grounded Artifact Reasoning. In IEEE/CVF Conference on Computer Vision and Pattern Recognition

(CVPR).

[25] Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, and Piotr Dollár. 2017. Focal loss for dense object detection. In IEEE International Conference on Computer Vision (ICCV). 2980–2988.

[26] Zhengzhe Liu, Xiaojuan Qi, and Philip HS Torr. 2020. Global Texture Enhancement for Fake Face Detection in the Wild. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 8060–8069.

[27] Ilya Loshchilov and Frank Hutter. 2019. Decoupled weight decay regularization. In International Conference on Learning Representations (ICLR).

[28] Long Ma, Zhiyuan Yan, Qinglang Guo, Yong Liao, Haiyang Yu, and Pengyuan Zhou. 2025. Detecting AI-Generated Video via Frame Consistency. In IEEE International Conference on Multimedia and Expo (ICME).

[29] Zhenliang Ni, Qiangyu Yan, Mouxiao Huang, Tianning Yuan, Yehui Tang, Hailin Hu, Xinghao Chen, and Yunhe Wang. 2026. GenVidBench: A 6-Million Benchmark for AI-Generated Video Detection. In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 40. 15582–15590. doi:10.1609/aaai.v40i18.38587

[30] Utkarsh Ojha, Yuheng Li, and Yong Jae Lee. 2023. Towards universal fake image detectors that generalize across generative models. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 24480–24489.

[31] Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El Nouby, et al. 2024. DINOv2: Learning robust visual features without supervision. Transactions on Machine Learning Research (2024).

[32] Ed Pizzi, Giorgos Kordopatis-Zilos, Hiral Patel, Gheorghe Postelnicu, Sugosh Nagavara Ravindra, Akshay Gupta, Symeon Papadopoulos, Giorgos Tolias, and Matthijs Douze. 2024. The 2023 Video Similarity Dataset and Challenge. Computer Vision and Image Understanding 243 (2024), 103997. doi:10.1016/j.cviu.2024.103997

[33] Chao Shuai, Jieming Zhong, Shuang Wu, Feng Lin, Zhibo Wang, Zhongjie Ba, Zhenguang Liu, Lorenzo Cavallaro, and Kui Ren. 2023. Locate and Verify: A Two-Stream Network for Improved Deepfake Detection. In Proceedings of the 31st ACM International Conference on Multimedia. 7131–7142. doi:10.1145/3581783.3612386

[34] Chuangchuang Tan, Huan Liu, Yao Zhao, Shikui Wei, Guanghua Gu, Ping Liu, and Yunchao Wei. 2024. Rethinking the up-sampling operations in CNN-based generative network for generalizable deepfake detection. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).

[35] Zachary Teed and Jia Deng. 2020. RAFT: Recurrent All-Pairs Field Transforms for Optical Flow. In European Conference on Computer Vision (ECCV). 402–419.

[36] Aaron van den Oord, Yazhe Li, and Oriol Vinyals. 2018. Representation learning with contrastive predictive coding. arXiv preprint arXiv:1807.03748 (2018).

[37] Jiuniu Wang, Hangjie Yuan, Dayou Chen, Yingya Zhang, Xiang Wang, and Shiwei Zhang. 2023. ModelScope Text-to-Video Technical Report. arXiv preprint arXiv:2308.06571 (2023).

[38] Sheng-Yu Wang, Oliver Wang, Richard Zhang, Andrew Owens, and Alexei A Efros. 2020. CNN-generated images are surprisingly easy to spot... for now. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 8695– 8704.

[39] Wenhao Wang and Yi Yang. 2024. VidProM: A million-scale real prompt-gallery dataset for text-to-video difusion models. In Advances in Neural Information Processing Systems (NeurIPS).

[40] Zhendong Wang, Jianmin Bao, Wengang Zhou, Weilun Wang, Hezhen Hu, Hong Chen, and Houqiang Li. 2023. DIRE for Difusion-Generated Image Detection. In IEEE/CVF International Conference on Computer Vision (ICCV).

[41] Ruiyang Xia, Dawei Zhou, Decheng Liu, Lin Yuan, Shuodi Wang, Jie Li, Nannan Wang, and Xinbo Gao. 2024. Advancing Generalized Deepfake Detector with Forgery Perception Guidance. In Proceedings of the 32nd ACM International Conference on Multimedia. doi:10.1145/3664647.3680713

[42] Yuting Xu, Jian Liang, Gengyun Jia, Ziming Yang, Yanhao Zhang, and Ran He. 2023. TALL: Thumbnail Layout for Deepfake Video Detection. In IEEE/CVF International Conference on Computer Vision (ICCV). 22658–22668.

[43] Rui Zhang, Hongxia Wang, Mingshan Du, Hanqing Liu, Yang Zhou, and Qiang Zeng. 2023. UMMAFormer: A Universal Multimodal-adaptive Transformer Framework for Temporal Forgery Localization. In Proceedings ofthe 31st ACM International Conference on Multimedia. 8749–8759. doi:10.1145/3581783.3613767

[44] Shuhai Zhang, ZiHao Lian, Jiahao Yang, Daiyuan Li, Guoxuan Pang, Feng Liu, Bo Han, Shutao Li, and Mingkui Tan. 2025. Physics-Driven Spatiotemporal Modeling for AI-Generated Video Detection. In Advances in Neural Information Processing Systems (NeurIPS).

[45] Chende Zheng, Ruiqi Suo, Chenhao Lin, Zhengyu Zhao, Le Yang, Shuai Liu, Minghui Yang, Cong Wang, and Chao Shen. 2025. D3: Training-Free AI-Generated Video Detection Using Second-Order Features. In IEEE/CVF International Conference on Computer Vision (ICCV). 12852–12862.

[46] Zangwei Zheng, Xiangyu Peng, Tianji Yang, Chenhui Shen, Shenggui Li, Hongxin Liu, Yukun Zhou, Tianyi Li, and Yang You. 2024. Open-Sora: Democratizing Eficient Video Production for All. arXiv preprint arXiv:2412.20404 (2024).

[47] Yue Zhou, Xinan He, Kaiqing Lin, Bing Fan, Feng Ding, Jinhua Zeng, and Bin Li. 2025. Brought a Gun to a Knife Fight: Modern VFM Baselines Outgun Specialized Detectors on In-the-Wild AI Image Detection. arXiv preprint arXiv:2509.12995 (2025).

## A Training Details and Hyperparameters

## A.1 Two-Stage Training Procedure

RIFT employs a two-stage training strategy with frozen backbone encoders (DINOv2 ViT-S/14 and RAFT-Large):

Stage 1: Orthogonal Decoupling Pretraining (30 epochs). Only the orthogonal decoupling module (macro/micro projectors and reconstruction decoder) and the macro stream analyzer are trainable; the fusion, conditional dependency, and micro stream modules are frozen. The manifold regularization weight is elevated $( \gamma _ { \mathrm { m a n i f o l d } } = 0 . 2 )$ to encourage well-structured manifold trajectories before joint training. Learning rate: $1 0 ^ { - 3 }$ with linear warmup (5% of epochs) followed by cosine annealing. Gram-Schmidt hard orthogonalization is applied every 100 optimization steps.

Stage 2: Joint Training (50 epochs). All trainable modules are unfrozen (backbone encoders remain frozen). The manifold weight is reduced to $\gamma _ { \mathrm { m a n i f o l d } } ~ = ~ 0 . 0 2$ to let classification loss dominate. Learning rate: $3 \times 1 0 ^ { - 4 }$ with linear warmup followed by cosine annealing with warm restarts $( T _ { 0 } = 5 , T _ { \mathrm { m u l t } } = 2 ) .$ . Early stopping with patience of 20 epochs on validation F1-macro.

## A.2 Hyperparameter Settings

Table 6 summarizes all hyperparameters used in the final model.

Table 6: Complete hyperparameter settings for RIFT.
<table><tr><td>Category</td><td>Parameter</td><td>Value</td></tr><tr><td rowspan="6">Model</td><td>Orthogonal embedding  $d _ { z }$ </td><td>128</td></tr><tr><td>Manifold dimension  $d _ { \mathrm { m a n i f o l d } }$ </td><td>24</td></tr><tr><td>Micro statistics  $d _ { \mathrm { { s t a t } } }$ </td><td>128</td></tr><tr><td>Macro temporal  $d _ { \mathrm { t e m p o r a l } }$ </td><td>64</td></tr><tr><td>Micro temporal dtemporal_micro</td><td>64</td></tr><tr><td>Topology  $d _ { \mathrm { t o p o } }$  (4 landscapes × 16 bins × 2 dims) InfoNCE temperature τ</td><td>128 0.15</td></tr><tr><td rowspan="5"></td><td>Segment length L</td><td>32 frames</td></tr><tr><td>Spatial resolution</td><td>224 × 224</td></tr><tr><td>Segments per video (train)</td><td>1</td></tr><tr><td>Segments per video (eval)</td><td>3</td></tr><tr><td></td><td></td></tr><tr><td rowspan="5">Loss Weights</td><td>Orthogonality  $\alpha _ { \mathrm { o r t h o } }$ </td><td>0.1</td></tr><tr><td>Reconstruction  $\beta _ { \mathrm { r e c o n } }$ </td><td>0.5</td></tr><tr><td>Manifold  $\gamma _ { \mathrm { m a n i f o l d } }$  (Stage 1 / Stage 2)</td><td>0.2 / 0.02</td></tr><tr><td>Conditional  $\delta _ { \mathrm { c o n d } }$ </td><td>0.05</td></tr><tr><td>Focal loss  $\gamma _ { \mathrm { f o c a l } }$ </td><td>2.0</td></tr><tr><td rowspan="6">Training</td><td>Stage 1 epochs / learning rate</td><td> $3 0 / 1 0 ^ { - 3 }$ </td></tr><tr><td>Stage 2 epochs / learning rate</td><td>50 / 3 × 10−4</td></tr><tr><td></td><td>64</td></tr><tr><td>Batch size Weight decay</td><td>0.01</td></tr><tr><td>Gradient clipping</td><td>1.0</td></tr><tr><td>Gram-Schmidt interval</td><td>100 steps</td></tr><tr><td rowspan="3"></td><td>Fusion dropout</td><td>0.15</td></tr><tr><td>Regularization Conditional predictor dropout</td><td>0.2</td></tr><tr><td>Micro GRU dropout</td><td>0.15</td></tr><tr><td rowspan="3"></td><td>Gaussian blur prob.</td><td>0.2</td></tr><tr><td>Augmentation Resize degradation prob.</td><td>0.3</td></tr><tr><td>Random crop prob.</td><td>0.3</td></tr></table>

## A.3 Loss Function

The total training loss is:

$$
\begin{array} { r } { \begin{array} { r } { \mathcal { L } = \mathcal { L } _ { \mathrm { f o c a l } } + \alpha \mathcal { L } _ { \mathrm { o r t h o } } + \beta \mathcal { L } _ { \mathrm { r e c o n } } \quad } \\ { + \gamma ( \mathcal { L } _ { \mathrm { i s o } } + \mathcal { L } _ { \mathrm { s m o o t h } } + \mathcal { L } _ { \mathrm { n c e } } ) + \delta \mathcal { L } _ { \mathrm { c o n d } } \quad } \end{array} } \end{array}\tag{16}
$$

where $\mathcal { L } _ { \mathrm { f o c a l } }$ is class-balanced focal loss $( \gamma _ { \mathrm { f o c a l } } ) = 2 . 0 ,$ equal class weights), <sub>ortho</sub> penalizes cosine similarity between streams, $\scriptstyle { \mathcal { L } } _ { \mathrm { r e c o n } }$ reconstructs CLS tokens, $\mathcal { L } _ { \mathrm { i s o } } / \mathcal { L } _ { \mathrm { s m o o t h } } / \mathcal { L } _ { \mathrm { n c e } }$ regularize the manifold (isometry, smoothness, contrastive), and $\mathcal { L } _ { \mathrm { c o n d } }$ is the conditional prediction loss (real samples only).

## B Per-Generator Detection Rates

## B.1 VidProM

Table 7 reports per-generator detection rates on the VidProM test set (50/50 stratified split, 60K samples).

Table 7: Per-generator detection on VidProM test set (50/50 stratified split). Per-generator rates use single-segment evaluation; aggregate metrics in Table 1 of the main text use the standard multi-segment protocol.
<table><tr><td>Generator</td><td>n</td><td>Detection Rate</td><td>False Negatives</td></tr><tr><td>T2VZero</td><td>4,953</td><td>100.00%</td><td>0</td></tr><tr><td>VideoCrafter2</td><td>4,968</td><td>100.00%</td><td>0</td></tr><tr><td>ModelScope</td><td>5,008</td><td>100.00%</td><td>0</td></tr><tr><td>Show-1</td><td>5,012</td><td>99.90%</td><td>5</td></tr><tr><td>Pika</td><td>5,019</td><td>99.08%</td><td>46</td></tr><tr><td>OpenSora</td><td>5,066</td><td>98.36%</td><td>83</td></tr><tr><td>CogVideo</td><td>4,970</td><td>97.24%</td><td>137</td></tr><tr><td>Real (FPR)</td><td>25,000</td><td>一</td><td>470 FP (1.88%)</td></tr><tr><td>Overall</td><td>59,996</td><td>99.23%</td><td>271 FN total</td></tr></table>

CogVideo and OpenSora are the most challenging generators (97.24% and 98.36% respectively), likely because their generation pipelines produce more temporally coherent outputs that better preserve cross-scale coupling patterns.

## B.2 GenVidBench

Table 8 reports per-generator results on GenVidBench (50/50 split, 34K samples).

Table 8: Per-generator detection on GenVidBench.
<table><tr><td>Subset</td><td>Generator</td><td>Acc</td><td>F1</td><td>Det. Rate</td></tr><tr><td rowspan="6">Main</td><td>CogVideo</td><td>99.61%</td><td>99.80%</td><td>99.61%</td></tr><tr><td>Mora</td><td>99.97%</td><td>99.99%</td><td>99.97%</td></tr><tr><td>MuseV</td><td>99.95%</td><td>99.98%</td><td>99.95%</td></tr><tr><td>SVD</td><td>99.97%</td><td>99.98%</td><td>99.97%</td></tr><tr><td>Real</td><td>98.27%</td><td>一</td><td>1.73% FPR</td></tr><tr><td>CogVideo</td><td>100.00%</td><td></td><td></td></tr><tr><td rowspan="5">Plants</td><td></td><td></td><td>100.00%</td><td>100.00%</td></tr><tr><td>Mora</td><td>100.00%</td><td>100.00%</td><td>100.00%</td></tr><tr><td>MuseV</td><td>100.00%</td><td>100.00%</td><td>100.00%</td></tr><tr><td>SVD</td><td>100.00%</td><td>100.00%</td><td>100.00%</td></tr><tr><td>Real</td><td>98.10%</td><td>一</td><td>1.90% FPR</td></tr></table>

RIFT achieves near-perfect detection on all GenVidBench generators, with CogVideo being the most challenging on the Main subset (99.61%), consistent with VidProM findings. The Plants subset (high visual randomness) is perfectly detected at 100% for all generators.

Table 10: Per-generator coupling mismatch statistics. MI = MINE mutual information estimate; NLL = conditional negative log-likelihood; cos $\bf ( z _ { m a c r o } , z _ { m i c r o } ) =$ orthogonality check; �¯ = mean Menger curvature. Curvature statistics in this table are computed over all 119,991 videos. The main text (Sec. 3.5) reports values from a 24K interpretability subset with balanced per-generator sampling, which yields slightly diferent statistics for both classes: real 18.99 8.12 (vs. 19.04 9.43 here) and fake $1 2 . 4 9 \pm 6 . 7 3$ (vs. $1 2 . 4 9 \pm 1 2 . 5 9$ here). The larger fake standard deviation in this table reflects cross-generator variance.

<table><tr><td>Source</td><td>n</td><td>MI</td><td>NLL</td><td>|cos |</td><td>K</td></tr><tr><td>Real</td><td>49,999</td><td> $- 0 . 4 5 \pm 0 . 6 8$ </td><td> $- 1 . 0 9 \pm 0 . 3 0$ </td><td>0.0009</td><td> $1 9 . 0 4 \pm 9 . 4 3$ </td></tr><tr><td>CogVideo</td><td>10,000</td><td> $- 6 . 3 1 \pm 5 . 4 2$ </td><td> $+ 0 . 2 4 \pm 0 . 3 0$ </td><td>0.0005</td><td> $1 1 . 8 7 \pm 8 . 4 3 $ </td></tr><tr><td>ModelScope</td><td>10,000</td><td> $- 4 . 5 8 \pm 4 . 9 2$ </td><td> $+ 0 . 0 1 \pm 0 . 4 7$ </td><td>0.0008</td><td> $5 . 0 2 \pm 2 . 0 1$ </td></tr><tr><td>OpenSora</td><td>10,000</td><td> $- 4 . 6 1 \pm 5 . 1 0$ </td><td> $+ 0 . 4 4 \pm 0 . 2 4$ </td><td>0.0021</td><td> $2 3 . 5 3 \pm 1 4 . 9 5$ </td></tr><tr><td>Pika</td><td>9,993</td><td> $- 3 . 8 6 \pm 4 . 0 2$ </td><td> $+ 0 . 5 5 \pm 0 . 2 1$ </td><td>0.0004</td><td> $2 8 . 8 7 \pm 1 7 . 2 7$ </td></tr><tr><td>Show-1</td><td>10,000</td><td> $- 1 0 . 1 6 \pm 6 . 2 5$ </td><td> $+ 0 . 3 0 \pm 0 . 1 6$ </td><td>0.0021</td><td> $9 . 7 4 \pm 5 . 3 8$ </td></tr><tr><td>T2VZero</td><td>9,999</td><td> $- 9 . 5 7 \pm 7 . 4 5$ </td><td> $+ 0 . 3 1 \pm 0 . 3 3$ </td><td>0.0015</td><td> $1 . 5 0 \pm 0 . 3 1$ </td></tr><tr><td>VideoCrafter2 10,000</td><td></td><td> $- 4 . 6 3 \pm 4 . 1 7$ </td><td> $+ 0 . 5 0 \pm 0 . 2 4$ </td><td>0.0017</td><td> $7 . 1 1 \pm 2 . 5 4$ </td></tr><tr><td>Fake avg</td><td>69,992</td><td> $- 6 . 2 5 \pm 5 . 9 5$ </td><td> $+ 0 . 3 4 \pm 0 . 3 4$ </td><td>0.0013</td><td> $1 2 . 4 9 \pm 1 2 . 5 9$ </td></tr></table>

## C Full Robustness Results

Table 9 presents RIFT’s performance under all 17 degradation conditions evaluated on a 996-sample VidProM subset. Features are re-extracted from degraded videos and evaluated with the frozen model.

Table 9: Full robustness evaluation under 17 degradation conditions. Clean baseline: F1 = 99.05%.
<table><tr><td>Category</td><td>Condition</td><td>Acc</td><td>F1</td><td>AUC</td></tr><tr><td>Clean</td><td>None</td><td>98.90</td><td>99.05</td><td>99.89</td></tr><tr><td rowspan="5">H.264</td><td>500 kbps</td><td>62.95</td><td>75.68</td><td>55.31</td></tr><tr><td>1Mbps</td><td>70.08</td><td>67.82</td><td>93.00</td></tr><tr><td>2Mbps</td><td>80.62</td><td>85.43</td><td>95.00</td></tr><tr><td>4Mbps</td><td>74.40</td><td>81.59</td><td>85.24</td></tr><tr><td>8Mbps</td><td>64.36</td><td>76.32</td><td>65.66</td></tr><tr><td rowspan="5">H.265</td><td>500 kbps</td><td>63.86</td><td>75.77</td><td>61.04</td></tr><tr><td>1Mbps</td><td>75.28</td><td>81.53</td><td>86.87</td></tr><tr><td>2Mbps</td><td>87.18</td><td>88.19</td><td>95.32</td></tr><tr><td>4Mbps</td><td>77.60</td><td>81.19</td><td>86.98</td></tr><tr><td>8Mbps</td><td>74.97</td><td>83.76</td><td>82.30</td></tr><tr><td rowspan="4">Gaussian noise</td><td>σ = 5</td><td>88.45</td><td>90.09</td><td>93.95</td></tr><tr><td> $\sigma = 1 0$ </td><td>70.87</td><td>79.89</td><td>78.07</td></tr><tr><td> $\sigma = 1 5$ </td><td>88.57</td><td>90.51</td><td>94.35</td></tr><tr><td> $\sigma = 2 0$ </td><td>84.56</td><td>88.30</td><td>95.65</td></tr><tr><td rowspan="3">Spatial crop</td><td>70%</td><td>59.64</td><td>35.34</td><td>85.42</td></tr><tr><td>80%</td><td>70.00</td><td>76.79</td><td>87.67</td></tr><tr><td>90%</td><td>60.86</td><td>68.88</td><td>80.02</td></tr></table>

Analysis. H.265 generally outperforms H.264 at matched bitrates, peaking at 2 Mbps $( \mathrm { F 1 } = 8 8 . 1 9 \% )$ . Gaussian noise is tolerated at most levels $( \mathrm { F 1 } > 8 8 \%$ at � 5, 15, 20 ). Some conditions exhibit nonmonotonic behavior (e.g., �=10 drops to 79.89%, H.264 1 Mbps to 67.82%), which we attribute to the limited evaluation subset $( n =$ 996) and the interaction between specific degradation parameters and the precomputed feature pipeline. Spatial cropping is the most damaging degradation (F1 drops to 35.34% at 70% crop), because cropping alters the spatial context upon which patch statistics and optical flow are computed, disrupting the feature extraction pipeline rather than the classifier itself.

## D Per-Generator Coupling Mismatch Statistics

Table 10 reports per-generator coupling statistics computed from the trained model across all 119,991 training+test videos.

Key Observations. (1) Universal coupling breakdown: all 7 generators exhibit substantially lower MI and positive NLL compared to real videos, confirming that coupling mismatch is a universal property of current AI video generation. (2) Orthogonality is maintained: $| \cos ( \mathbf { z } _ { \mathrm { m a c r o } } , \mathbf { z } _ { \mathrm { m i c r o } } ) | < 0 . 0 0 3$ for all generators, confirming no information leakage between streams. (3) Generator-specific signatures: curvature varies widely across generators (T2VZero: 1.50, Pika: 28.87), reflecting diferent temporal dynamics. Despite this diversity, the coupling divergence signal remains consistently discriminative. (4) NLL is the strongest separator: Cohen’s � = 4.46 for NLL vs. 1.37 for MI, because NLL directly measures the conditional dependency � micro macro .

## E Seedance 2.0 Full Results

Table 11 reports per-video detection results for all 27 Seedance 2.0 videos evaluated in zero-shot mode (no fine-tuning or adaptation).

Table 11: Per-video Seedance 2.0 zero-shot detection results. All 27 videos correctly classified as AI-generated. Coupling statistics: MI = 17.74, NLL = 0.30, �<sub>micro</sub> = 0.49, �¯ = 10.28.
<table><tr><td>Video Content</td><td>Pred</td><td>Confidence</td></tr><tr><td>Aerial terraces</td><td>Fake</td><td>97.1%</td></tr><tr><td>Ancient architecture</td><td>Fake</td><td>99.6%</td></tr><tr><td>Animals (beach)</td><td>Fake</td><td>100.0%</td></tr><tr><td>Animals (grassland)</td><td>Fake</td><td>100.0%</td></tr><tr><td>Aurora, Iceland</td><td>Fake</td><td>99.8%</td></tr><tr><td>Calligraphy</td><td>Fake</td><td>100.0%</td></tr><tr><td>Children, playground</td><td>Fake</td><td>96.1%</td></tr><tr><td>City traffic</td><td>Fake</td><td>98.8%</td></tr><tr><td>Coffee still life</td><td>Fake</td><td>89.5%</td></tr><tr><td>Desert landscape</td><td>Fake</td><td>90.8%</td></tr><tr><td>Fireworks night</td><td>Fake</td><td>100.0%</td></tr><tr><td>Football</td><td>Fake</td><td>100.0%</td></tr><tr><td>Indoor cooking</td><td>Fake</td><td>100.0%</td></tr><tr><td>Industrial factory</td><td>Fake</td><td>100.0%</td></tr><tr><td>Macro flowers</td><td>Fake</td><td>98.8%</td></tr><tr><td>Musician</td><td>Fake</td><td>99.0%</td></tr><tr><td>Nature coastline</td><td>Fake</td><td>100.0%</td></tr><tr><td>Ocean sailboat</td><td>Fake</td><td>100.0%</td></tr><tr><td>Pastoral rice fields</td><td>Fake</td><td>100.0%</td></tr><tr><td>Rain scene</td><td>Fake</td><td>100.0%</td></tr><tr><td>Snow forest</td><td>Fake</td><td>99.9%</td></tr><tr><td>Sports person</td><td>Fake</td><td>100.0%</td></tr><tr><td>Starry timelapse</td><td>Fake</td><td>89.9%</td></tr><tr><td>Street music</td><td>Fake</td><td>99.8%</td></tr><tr><td>Tech robot</td><td>Fake</td><td>99.7%</td></tr><tr><td>Underwater ocean</td><td>Fake</td><td>73.4%</td></tr><tr><td>Waterfall jungle</td><td>Fake</td><td>100.0%</td></tr><tr><td>Mean / All</td><td>27/27</td><td>97.5%</td></tr></table>

The lowest-confidence video is underwater ocean (73.4%), featuring slow-moving marine life with minimal motion structure that weakens the macro baseline’s geometric features. Nevertheless, all 27 videos are correctly classified, demonstrating that the coupling mismatch signal generalizes to unseen commercial generators.

Coupling Signal Comparison. Seedance 2.0 exhibits even weaker cross-scale coupling than the 7 training generators: MI = 17.74 (vs. fake avg 6.25), NLL = 0.30 (within fake distribution 0.34 0.34), $g _ { \mathrm { m i c r o } } = 0 . 4 9$ (vs. fake avg 0.58, real 0.80), and $\bar { \kappa } = 1 0 . 2 8$ (vs. fake avg 12.49, real 19.04). The extreme MI value may partially reflect MINE estimator instability on out-of-distribution data; it should be interpreted as a relative indicator of coupling breakdown rather than a calibrated measurement.