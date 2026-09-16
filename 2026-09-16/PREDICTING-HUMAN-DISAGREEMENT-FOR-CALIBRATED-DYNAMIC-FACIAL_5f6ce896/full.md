# PREDICTING HUMAN DISAGREEMENT FOR CALIBRATED DYNAMIC FACIAL EXPRESSION RECOGNITION

Yiming Wang

Frederick W. B. Li

Jingyun Wang

Department of Computer Science, Durham University

## ABSTRACT

Dynamic facial expression recognition (DFER) benchmarks such as DFEW provide multiple annotator votes per clip, yet most models collapse them to a majority label and cannot represent human disagreement at inference time. We propose a disagreementaware DFER framework that trains directly on the raw annotator count vector using a Dirichlet–Multinomial likelihood. Unlike mean-only soft-label objectives, the proposed likelihood provides scale-sensitive supervision for the Dirichlet concentration while preserving the predictive mean. A separate ambiguity head predicts annotation entropy for unseen clips, and a monotone Chow-style reject rule combines predicted ambiguity, vacuity, temporal instability, and input quality for selective prediction. On DFEW, the method preserves recognition accuracy while reducing ECE by 30% and AURC by 15%, and predicted ambiguity reaches a Spearman correlation of 0.52 with the annotation entropy of test clips. The calibration and selective-prediction gains transfer to FERV39k and remain under identity- and movie-disjoint DFEW splits.

Index Terms— Dynamic facial expression recognition, annotator disagreement, Dirichlet–Multinomial likelihood, calibration, selective prediction

## 1. INTRODUCTION

In-the-wild facial expressions are often subtle and context dependent, leading annotators to disagree even on the same video. DFEW [1] records this variation through ten annotator votes per clip, yet most DFER methods collapse these votes to a majority label and evaluate only recognition accuracy using weighted and unweighted average recall (WAR/UAR) [2, 3, 4, 5]. This discards information about how consistently an expression is perceived across annotators [6, 7].

The effect is visible in Fig. 1: four public DFER checkpoints are well calibrated when annotators agree, but become increasingly overconfident as disagreement grows, because predictive confidence does not distinguish a clear consensus from an intrinsically ambiguous expression. A clip with votes 6–2–2 and one with 10–0–0 share a majority label but differ substantially in human agreement; a disagreement-aware model should expose this distinction at inference time, through an ambiguity estimate or selective prediction.

We instead model the annotator counts directly with a Dirichlet– Multinomial (DM) likelihood. Unlike mean-only objectives, which are invariant to the Dirichlet scale, the DM likelihood directly supervises both the predictive mean and a concentration-dependent uncertainty signal, which we evaluate empirically for ambiguity ranking, calibration and selective prediction. To the best of our knowledge, this is the first DFER study to train directly on DFEW's released per-clip annotator count vectors with an exact DM likelihood and to evaluate concentration-dependent uncertainty for calibrated selective prediction. We further learn an ambiguity head $\hat { u } _ { h }$ that predicts annotation entropy without access to annotator votes at inference time. Its predictions are scored against the annotation entropy of

![](images/8584f3579670ad3cf8154313c6defecc241fba2ee7c030b1da46893f4c46150f.jpg)  
Fig. 1. Entropy-stratified ECE of four public DFER checkpoints on DFEW. Calibration degrades as human disagreement increases, with the largest overconfidence observed on the highest-entropy clips.

DFEW test clips. Predicted ambiguity, vacuity, temporal instability, and input quality are combined in a monotone Chow-style reject rule and evaluated with calibration and coverage–risk metrics. We also audit the standard DFEW protocol with identity- and moviedisjoint splits (Table 1); WAR/UAR verify that the uncertainty improvements do not sacrifice recognition accuracy. Our contributions are thus a count-likelihood formulation for disagreement-aware DFER, a test-time estimator of human ambiguity, and a selectiveprediction framework evaluated under shift and stricter splits.

## 2. RELATED WORK

Evidential learning. EDL [8] fits Dirichlet evidence to one-hot labels with an expected-risk loss and a KL-annealed regulariser; DEAR [9] extends it to open-set action recognition, and the surveys [10, 11] do not focus on supervision from annotator counts. Our formulation changes the likelihood rather than the output head: the DM objective supervises the Dirichlet concentration from data, whereas in EDL the concentration is mainly shaped by the regulariser.

Label-distribution learning in FER. CIFAR-10H [12] and MI-DAS [13] train on vote frequencies, MIDAS through mixup-blended targets. Static FER has a longer line of distribution-learning work: DMUE [14] mines latent distributions with pairwise uncertainty, UA-LDL [15] weights a neighbourhood-derived distribution by an uncertainty estimate, and Ada-DF [16] fuses instance- and classlevel distributions. The reviewed methods primarily optimise the predictive mean with a mean-only objective (KL or cross-entropy), which is scale-blind, and do not address count-likelihood-based selective prediction. We instead train on the un-blended annotator count vector with a DM likelihood; our KL-to-mean arm is a controlled mean-only reference on the true vote distribution, isolating the effect of the count likelihood.

Model self-disagreement. RDFER [17] measures prediction disagreement across temporally re-sampled views and uses it to reweight and clean training. Unlike RDFER, our $u _ { t }$ is the same measurement kept as an inference-time output, while supervision comes from human disagreement, which RDFER does not use.

Accuracy frontier. MAE-DFER [3], S2D [5], FineCLIPER [18] (WAR 76.2), S4D [19] (76.7) and the audio-visual MMA-DFER [20] (77.5) reach WAR 74–77.5 on DFEW, but target recognition accuracy rather than calibration or selective prediction.

Calibration and abstention from disagreement. Crowd-Calibrator [21] uses the discrepancy between model and crowd label distributions for abstention, and Tao et al. [22] calibrate against the full annotator distribution; both operate post hoc and do not supervise a Dirichlet concentration. In FER, Inoshita and Ueno [23] split ensemble uncertainty into aleatoric and epistemic parts and validate ambiguity against annotator disagreement, whereas we train a single video model directly from annotator counts and estimate disagreement at inference. Cui et al. [24] study trustworthy DFER via an information bottleneck, without annotator disagreement or selective prediction. Dirichlet–Multinomial outputs have been used for generic regression [25], not for annotator counts or selective prediction.

## 3. METHOD

Preliminaries. A clip x has $K = 7$ candidate expressions and a count vector $\mathbf { n } \in \mathbb { N } ^ { K }$ with $\textstyle \sum _ { k } n _ { k } = N ( N = 1 0 $ on DFEW). A frozen backbone (VideoMAE-B [26]) is read through four deterministic temporal views $v \in \{ 1 , \ldots , 4 \}$ (uniform, front half, back half, peak-centred); a lightweight temporal adapter, the only trained part of the encoder, maps each view to $\mathbf { h } ^ { ( v ) } \in \mathbb { R } ^ { d }$ . The pooled feature $\begin{array} { r } { \mathbf { z } = \frac { 1 } { 4 } \sum _ { v } \mathbf { h } ^ { ( v ) } } \end{array}$ feeds the DM head, which produces the final prediction; the same DM head is also applied to each $\mathbf { h } ^ { ( v ) }$ separately to obtain per-view predictive means $\hat { \mathbf { p } } ^ { ( \bar { v } ) }$ , used only for the instability signal of Sec. 3.3; the concatenation $\tilde { \mathbf { z } } = [ \mathbf { h } ^ { ( 1 ) } ; \ldots ; \mathbf { h } ^ { ( 4 ) } ]$ feeds the ambiguity head. Throughout, $\mathrm { H } ( \mathbf { q } ) = - \dot { \sum } _ { k }$ q<sub>k</sub> log q<sub>k</sub> is the Shannon entropy with natural logarithms, so every likelihood and NLL below is in nats. Fig. 2 shows the pipeline.

## 3.1. Mechanism: modelling the annotation process

Let e = softplus $( \mathbf { W } _ { e } \mathbf { z } + \mathbf { b } _ { e } ) \in \mathbb { R } _ { > 0 } ^ { K }$ be the evidence produced by the DM head. The Dirichlet parameters are $\alpha _ { k } = e _ { k } + 1$ (the +1 keeps the Dirichlet defined). Write $\begin{array} { r } { S = \sum _ { k } \alpha _ { k } } \end{array}$ for the concentration, $\hat { \bf p } = { \pmb \alpha } / S$ for the predictive mean and $u _ { e } = K / S$ for the vacuity. We model the panel generatively, $\mathbf { p } \sim \operatorname { D i r } ( \pmb { \alpha } ( x ) )$ $\mathbf { n } \sim \mathrm { M u l t } ( N , \mathbf { p } )$ , and integrate p out, which gives the Dirichlet– Multinomial probability $\mathrm { D M } ( \mathbf { n } \mid \mathbf { \bar { \alpha } } )$ ; the training loss is its negative logarithm, $\mathcal { L } _ { \mathrm { D M } } = - \log \mathrm { D M } ( \mathbf { n } \mid \dot { \alpha } )$ , i.e.

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { D M } } = \log \Gamma ( N { + } S ) - \log \Gamma ( S ) - \log \frac { N ! } { \prod _ { k } n _ { k } ! } } \\ { - \sum _ { k } \big [ \log \Gamma ( n _ { k } { + } \alpha _ { k } ) - \log \Gamma ( \alpha _ { k } ) \big ] . } \end{array}\tag{1}
$$

This is a compact model of annotation variation: it assumes that the $N$ labels are conditionally exchangeable given the clip and does not model annotator-specific bias or dependence. The third term is constant in α and is kept only so that reported nats are comparable to a multinomial fit. $\mathcal { L } _ { \mathrm { D M } }$ is curved in S: for a fixed predictive mean, the probability of counts that deviate from that mean falls as S grows, so an amortised $\alpha ( x )$ can raise S only where its mean reliably matches the votes of similar clips.

Why not KL to the mean. The objective used by prior votebased FER methods, $\mathrm { K L } ( \mathbf { n } / N \lVert \hat { \mathbf p } )$ or the equivalent soft-label cross-entropy, depends on α only through $\mathbf { \alpha } _ { \alpha / \bar { S } }$ . It is invariant to $\mathbf { \alpha } \mathbf { \mapsto } c \mathbf { \alpha } .$ so it provides no direct supervision for the Dirichlet scale; S is then shaped only indirectly, by initialisation and regularisation. For $\bar { \mathbf { n } } \bar { \mathbf { \Lambda } } = ( 1 , 2 , 7 )$ , the KL is 0 for both ${ \pmb { \alpha } } = ( 1 , 2 , 7 )$ and ${ \pmb { \alpha } } = ( 1 0 , 2 0 , 7 0 )$ , whereas $\mathcal { L } _ { \mathrm { D M } }$ is 2.887 versus 2.229 nats. In a controlled simulation (2 000 synthetic clips with log-uniform perclip concentrations; five votes fix the predicted mean, five further votes are the fitting signal), the KL-fitted S has rank correlation 0.00 with the true concentration (AUROC 0.50 for high versus low concentration) whereas the DM fit reaches 0.62; NLL on held-out synthetic clips differs by $< ~ 0 . 0 1$ nats per clip because with ten votes the mean dominates, so the ablation in Sec. 4 is judged on S-dependent outputs, not NLL alone.

Effect of concentration supervision. The DM likelihood therefore supervises both the predictive mean and a concentrationdependent uncertainty signal, the vacuity $u _ { e } ,$ in a single term. We do not claim that $S$ measures human disagreement or missing evidence on its own: a low $S$ can also arise from model mismatch, noisy or low-quality clips, or the limited ten-vote panel, so $u _ { e }$ is one signal among four, evaluated empirically (Table 3a) and inside the reject rule of Sec. 3.4. Concentration estimation improves with panel size: in the simulation the rank correlation between fitted and true $S$ rises from 0.20 at five votes to 0.41 at twenty. With ten votes per clip, S is thus a useful predictive uncertainty signal rather than an unbiased estimate of intrinsic annotator precision.

## 3.2. Verifiable consequence: predicting disagreement without annotators

The DM model above is the probabilistic core of the method; the components of Secs. 3.2–3.4 are practical estimators built on top of it. Let $u _ { h } = \mathrm { H } ( { \bf n } / N ) /$ log $K \in [ 0 , 1 ]$ be the normalised entropy of the vote frequencies, the human ambiguity of a clip. The model never receives votes at inference, so we regress it: $\hat { u } _ { h } = g ( \tilde { \mathbf { z } } )$ , with g a two-layer MLP with sigmoid output, trained by $\mathcal { L } _ { \mathrm { a m b } } = | \hat { u } _ { h } - u _ { h } |$ The Dirichlet already provides parameter-free proxies: the entropy of the predictive mean H(pˆ) and the Dirichlet mutual information $\mathrm { H } ( \mathbb { E } [ \mathbf { p } ] ) - \mathbb { E } [ \mathrm { H } ( \mathbf { p } ) ]$ of [27]. We use a separate head because it regresses annotation entropy directly and can exploit all four temporal views. On DFEW the votes of the test clips are available for scoring, never as input; we report Spearman $\rho$ and MAE against this test-clip annotation entropy, and AUROC for separating high- from low-ambiguity clips, for H(pˆ), the MI proxy and $\hat { u } _ { h }$ (Table 3b).

3.3. Two engineering signals: instability and input quality Two further inference-time signals are fixed computations rather than heads. Temporal-view instability is the generalised Jensen– Shannon divergence of the per-view predictive means, $\begin{array} { r l } { u _ { t } } & { { } = } \end{array}$ $\begin{array} { r } { \big [ \mathrm { H } ( \bar { \mathbf { p } } ) - \frac { 1 } { 4 } \sum _ { v } \mathrm { H } ( \hat { \mathbf { p } } ^ { ( v ) } ) \big ] / \log 4 } \end{array}$ with $\begin{array} { r } { \bar { \textbf { p } } = \frac { 1 } { 4 } \sum _ { v } \hat { \mathbf { p } } ^ { ( v ) } } \end{array}$ : the entropy of the mean of the four view distributions minus their mean entropy, normalised by log 4 so that $u _ { t } \in [ 0 , 1 ]$ . It measures prediction instability, not epistemic uncertainty (a genuinely evolving expression also raises it); the windows are fixed rather than learned so that the instrument is not coupled to the model it measures (ablated in Sec. 4.3). Input-quality uncertainty $u _ { q } = \sigma ( \mathbf { w } _ { q } ^ { \top } \mathbf { f } _ { q } + b _ { q } )$ is a frozen logistic estimator on eight percentile-normalised clip-quality features $\begin{array} { r } { \mathbf { f } _ { q } \ ( \mathrm { F i g } . \ 2 ) \ } \end{array}$ : mean and minimum face-detector confidence, landmark jitter, inter-ocular distance, Laplacian variance, occlusion ratio, absolute yaw and pitch. No quality labels exist, so $\left( \mathbf { w } _ { q } , b _ { q } \right)$ are trained to regress the severity of synthetic corruptions (blur, occlusion patches, down-scaling, compression; disjoint from the evaluation-time suite) applied to clean clips at four levels, validated by an AUROC of 0.93 for clean versus corrupted clips and a monotone mean $u _ { q }$ across levels, then frozen. High $\boldsymbol { u } _ { \boldsymbol { q } }$ means corrupted input. During training, $\mathcal { L } _ { \mathrm { D M } }$ alone is weighted per clip by $1 - u _ { q }$ (the ${ \mathcal { L } } _ { \mathrm { a m b } }$ and $\mathcal { L } _ { \mathrm { c a l } }$ terms are not): corruption is down-weighted, ambiguity never is, because $\mathcal { L } _ { \mathrm { D M } }$ already fits a broad posterior to a split panel.

## 3.4. Use: abstention as a reject option

We treat abstention as a decision: answer when the estimated risk of answering is below a threshold, as in Chow’s reject option [28]. Under the fitted posterior the plug-in risk is $1 - \operatorname* { m a x } _ { k } \hat { p } _ { k }$ . But corruption and instability are failure modes outside the fitted Dirichlet—a corrupted clip can yield a confidently wrong α—which is exactly the overconfidence of Fig. 1. We therefore estimate risk from all four signals. Let $\mathbf { v } ( x ) \in [ 0 , 1 ] ^ { 4 }$ be the percentile-normalised vector $( \hat { u } _ { h } , u _ { e } , u _ { q } , u _ { t } ) ;$ ; the estimated risk and the reject rule are

$$
\begin{array} { r } { \hat { \rho } ( x ) = \sigma \Big ( \sum _ { j = 1 } ^ { 4 } w _ { j } v _ { j } ( x ) + b \Big ) , w _ { j } \geq 0 ; \quad r ( x ) = { \bf 1 } \big [ \hat { \rho } ( x ) > t ^ { * } \big ] . } \end{array}\tag{2}
$$

$\mathbf { \Psi } ( \mathbf { w } , b )$ are fitted on a validation split by logistic regression of observed errors on $\mathbf { v } , \hat { \rho }$ is isotonic-calibrated, and $t ^ { * }$ minimises selective risk at the target coverage. We call Eq. (2) a monotone Chowstyle rule rather than a Chow rule: it thresholds an estimated risk, but the risk is learned from validation errors rather than derived from a specified abstention cost. Without $u _ { q }$ a corrupted input is answered confidently; without $w _ { j } \geq 0$ a signal could lower the estimated risk as it grows. Note that a high $\hat { u } _ { h }$ alone need not trigger abstention: a clean, high-evidence, high-ambiguity clip is an answerable genuinely mixed prediction. An interpretable fallback of sequential monotone gates $( u _ { q } > \tau _ { q } ,$ then u<sub>e</sub> $> \tau _ { e } \wedge \hat { u } _ { h } <$ h ) is ablated in Table 3c.

![](images/fcdf1659ca005908c7c72479426c9f2cc41a6a9fa797c274a11eed90e4ba3a78.jpg)  
Fig. 2. Method overview, one row per inference-time signal. (a) The DM head maps the pooled z to α, giving $\hat { \bf p }$ and the vacuity u<sub>e</sub>; applied to each $\mathbf { h } ^ { ( v ) }$ it also gives the per-view means whose generalised JSD is the instability u<sub>t</sub>. (b) The ambiguity head g regresses uˆ<sub>h</sub> from z˜; a frozen logistic estimator maps clip-quality features to $u _ { q } . \mathrm { ( c ) }$ The four signals are percentile-normalised and fused by a monotone Chow-style rule with validation-chosen $\bar { t } ^ { * }$ . Votes n enter only through the dashed training-only terms. Red outlines: components introduced here.

## 3.5. Calibration term and training objective

Let $\bar { \mathrm { H } } ( \hat { \mathbf { p } } ) = \mathrm { H } ( \hat { \mathbf { p } } ) /$ log K and $y ^ { * } = \arg \operatorname* { m a x } _ { k } n _ { k }$ be the majority vote. A two-parameter Platt alignment $\hat { c } ( x ) = \sigma \big ( a ( 1 - \bar { \mathrm { H } } ( \hat { \mathbf { p } } ) ) +$ $b _ { c } )$ is trained by $\begin{array} { r } { \mathcal { L } _ { \mathrm { c a l } } = \mathrm { B C E } \big ( \widehat { c } ( x ) , { \bf 1 } [ \mathrm { a r g } \mathrm { m a x } _ { k } \alpha _ { k } = y ^ { * } ] \big ) } \end{array}$ with a stop-gradient into $^ { \alpha , }$ so it aligns confidence with expected correctness without pushing entropy down on ambiguous clips, as an AvUC-style loss [29] would. It is a lightweight design choice, not a contribution: its gain over temperature scaling is small (Sec. 4.3). The full objective is

$$
\begin{array} { r } { \mathcal { L } = \mathcal { L } _ { \mathrm { D M } } + \lambda _ { 1 } \mathcal { L } _ { \mathrm { a m b } } + \lambda _ { 2 } \mathcal { L } _ { \mathrm { c a l } } , } \end{array}\tag{3}
$$

with $\lambda _ { 1 } ~ \in ~ \{ 0 . 1 , 0 . 3 , 1 . 0 \}$ and $\lambda _ { 2 } ~ \in ~ \{ 0 , 0 . 1 , 0 . 3 \}$ chosen by grid search on the fold-1 validation split with a DM-NLL+ECE composite, then fixed for all folds and datasets. λ = 0 with post-hoc temperature scaling [30] is an ablation arm (Sec. 4.3).

## 4. EXPERIMENTS

Datasets and their roles. DFEW [1] (16k clips, ten votes each) is the primary benchmark; as the only dataset with released annotator counts, it is the only one that validates count-based disagreement modelling (DM-NLL, KL-to-human, $\hat { u } _ { h }$ against annotation entropy). FERV39k [31] tests whether calibration and risk ranking transfer under full domain shift; it releases single labels only. MAFW [32] provides class-disjoint and compound-expression evidence: $\hat { u } _ { h }$ AUROC on compound- versus single-label clips, and the abstention rate on its four classes unseen in DFEW. Neither FERV39k nor MAFW validates human-disagreement prediction directly, since neither releases comparable per-clip vote counts.

Metrics and claims. WAR/UAR show that accuracy is not sacrificed; DM-NLL and KL to the human distribution support the mechanism; Spearman $\rho ,$ MAE and AUROC against annotation entropy support the ambiguity head; ECE [30] and AURC [33] support the reject rule. $\Delta _ { T }$ , the mean accuracy drop under a temporalperturbation suite (frame shuffle, peak-repeat, reverse; disjoint from training augmentation), audits whether dynamics are used.

Setup. The backbone is a frozen VideoMAE-B [26] (ViT-B/16, 16 frames at $2 2 4 ^ { 2 }$ , the VoxCeleb2 weights of MAE-DFER [3]); the adapter has $M = 2$ blocks over the four view tokens $( d = 7 6 8$ 8 heads) and is trained for 30 epochs with AdamW (learning rate

<table><tr><td></td><td colspan="3">Official 5-fold</td><td colspan="3">Identity-disjoint</td><td colspan="3">Movie-disjoint</td></tr><tr><td>Method</td><td>WAR↑</td><td>ECE↓</td><td>AURC↓</td><td>WAR↑</td><td>ECE↓</td><td>AURC↓</td><td>WAR↑</td><td>ECE↓</td><td>AURC↓</td></tr><tr><td>M3DFEL [2]</td><td>69.25</td><td>0.158</td><td>0.151</td><td>66.8</td><td>0.171</td><td>0.164</td><td>65.3</td><td>0.182</td><td>0.176</td></tr><tr><td>DFER-CLIP [4]</td><td>71.25</td><td>0.094</td><td>0.135</td><td>68.9</td><td>0.106</td><td>0.147</td><td>67.4</td><td>0.118</td><td>0.158</td></tr><tr><td>MAE-DFER [3]</td><td>74.43</td><td>0.121</td><td>0.118</td><td>71.6</td><td>0.134</td><td>0.131</td><td>69.9</td><td>0.146</td><td>0.142</td></tr><tr><td>S2D [5]</td><td>76.03</td><td>0.137</td><td>0.110</td><td>73.0</td><td>0.150</td><td>0.122</td><td>71.2</td><td>0.161</td><td>0.133</td></tr><tr><td>Ours, KL-to-mean</td><td>71.6</td><td>0.052</td><td>0.128</td><td>69.3</td><td>0.058</td><td>0.137</td><td>67.8</td><td>0.063</td><td>0.146</td></tr><tr><td>Ours, DM</td><td>71.8</td><td>0.036</td><td>0.108</td><td>69.5</td><td>0.041</td><td>0.116</td><td>68.0</td><td>0.046</td><td>0.124</td></tr></table>

Table 1. Protocol audit on DFEW under the official folds and two leakage-free re-splits (identity-disjoint: clustered face-track embeddings kept together; movie-disjoint: source films kept together). Every method loses accuracy under the strict splits; the gains of the DM objective persist.

$3 \times 1 0 ^ { - 4 }$ , cosine schedule, weight decay 0.05, batch 64). Augmentation is random crop and flip only, so training sees no temporal perturbation; $\lambda _ { 1 } ~ = ~ \bar { 0 } . 3$ and $\lambda _ { 2 } ~ = ~ 0 . 1$ come from the grid of Sec. 3.5. Baselines are the public checkpoints of DFER-CLIP [4], MAE-DFER [3], S2D [5] and M3DFEL [2], re-evaluated for the uncertainty metrics, plus a KL-to-mean arm with identical backbone, adapter and heads, a controlled mean-only reference using the true vote distribution. Results are 5-fold means; fold standard deviations (≤0.004 ECE, ≤0.005 AURC; Table 2) are an order of magnitude below the reported gaps.

## 4.1. Protocol audit

The official 5-fold split of DFEW does not isolate movies or actors. We therefore re-split by identity and by source movie and re-train four public baselines and both of our arms (Table 1). Absolute numbers drop for every method but the relative gains do not, and the ambiguity head keeps $\rho = 0 . 4 9$ and 0.47 against test-clip annotation entropy under the identity- and movie-disjoint splits (0.52 under the official split). Every later table uses the official split.

## 4.2. Main results on DFEW

With the same backbone, adapter and heads, replacing the meanonly KL objective with the DM likelihood leaves recognition performance essentially unchanged (Table 2): WAR is 71.8 versus 71.6, and KL to the human vote distribution is 0.42 versus 0.41, the metric directly optimised by the KL-to-mean arm. The difference appears in uncertainty quality: DM reduces ECE from 0.052 to 0.036, and the full DM-based system with the Chow-style rule reaches an AURC of 0.108, against 0.128 for the KL-to-mean arm ranked by its own confidence. Table 2 compares each method under its own available ranking score; Table 3a isolates the effect of the training objective by applying the same learned reject rule to both internal arms. Thus the DM objective improves calibration at unchanged accuracy, and the full framework further improves selective prediction.

Consistent with Sec. 3.1, DM-NLL only weakly separates the two objectives (3.19 versus 3.21 nats); the diagnostic differences appear in the concentration-dependent quantities of Table 3a. The public checkpoints reach ECE between 0.09 and 0.16, compared with 0.036 for ours, and their calibration is worst on the highest-entropy stratum of Fig. 1 (0.28, versus 0.07 for ours). Under the temporal suite, ours drops 4.8 WAR points but three public checkpoints fewer than 2.5, so high accuracy alone does not imply reliance on dynamics.

<table><tr><td>Method</td><td>WAR↑</td><td>UAR↑</td><td> ${ \mathrm { K L } } \downarrow$ </td><td> $\mathrm { E C E \downarrow }$ </td><td>AURC↓</td><td> $\Delta _ { T }$ </td></tr><tr><td>MIDAS [13]†</td><td>69.16</td><td>57.45</td><td></td><td></td><td>一</td><td></td></tr><tr><td>RDFER [17]†</td><td>69.73</td><td>56.93</td><td></td><td></td><td></td><td></td></tr><tr><td>DFER-CLIP [4]</td><td>71.25</td><td>59.61</td><td>0.92</td><td>0.094</td><td>0.135</td><td>2.1</td></tr><tr><td>M3DFEL [2]</td><td>69.25</td><td>56.10</td><td>1.05</td><td>0.158</td><td>0.151</td><td>1.8</td></tr><tr><td>MAE-DFER [3]</td><td>74.43</td><td>63.41</td><td>0.88</td><td>0.121</td><td>0.118</td><td>5.4</td></tr><tr><td>S2D [5]</td><td>76.03</td><td>61.82</td><td>0.85</td><td>0.137</td><td>0.110</td><td>1.2</td></tr><tr><td>Ours, KL-to-mean</td><td>71.6</td><td>60.3</td><td>0.41±0.01</td><td>0.052±0.003</td><td>0.128±0.004</td><td>4.6</td></tr><tr><td>Ours, DM</td><td>71.8</td><td>60.7</td><td>0.42±0.01</td><td>0.036±0.003</td><td>0.108±0.004</td><td>4.8</td></tr></table>

Table 2. DFEW, 5-fold means (with standard deviation over folds for the uncertainty metrics of our two arms). <sup>†</sup>Accuracy reported by the authors; no public checkpoint, so the uncertainty columns cannot be computed (–); the accuracy-only frontier (WAR 76–77.5) is cited in Sec. 2. KL: to the human vote distribution; AURC ranks clips by each method’s own available score: softmax for the baselines, max<sub>k</sub> $: \hat { p } _ { k }$ for the KL-to-mean arm and the Chow-style rule of Eq. (2) for the DM arm; $\Delta _ { T } \colon$ mean accuracy drop under the temporal suite (larger means dynamics are used).

(a) Likelihood: S-dependent outputs; AURC with the Chow-style rule for both arms Objective NLL↓ u<sub>e</sub>-AUROC↑ ECE↓ AURC↓ KL-to-mean 3.21 0.51 0.052 0.117 DM (ours) 3.19 0.64 0.036 0.108
<table><tr><td colspan="5">(b) Ambiguity estimator: against test-clip entropy; FERV39k ranking</td></tr><tr><td>Estimator</td><td>ρ↑</td><td>MAE↓</td><td>AUROC↑</td><td> $\mathsf { A U R C } _ { \mathrm { F 3 9 k } } \downarrow$ </td></tr><tr><td>H(p) from α</td><td>0.45</td><td>0.16</td><td>0.74</td><td>0.298</td></tr><tr><td>MI from α</td><td>0.38</td><td>0.19</td><td>0.70</td><td>0.311</td></tr><tr><td>ûh (learned head)</td><td>0.52</td><td>0.13</td><td>0.79</td><td>0.293</td></tr></table>

<table><tr><td colspan="5">(c) Reject rule: selective risk↓ at coverage 0.9 and 0.8</td></tr><tr><td>Score</td><td colspan="2">DFEW</td><td colspan="2">FERV39k</td></tr><tr><td></td><td>0.9</td><td>0.8</td><td>0.9</td><td>0.8</td></tr><tr><td>softmax confidence</td><td>0.245</td><td>0.215</td><td>0.455</td><td>0.428</td></tr><tr><td>H()</td><td>0.242</td><td>0.211</td><td>0.452</td><td>0.424</td></tr><tr><td>ue alone</td><td>0.264</td><td>0.241</td><td>0.470</td><td>0.452</td></tr><tr><td> $\hat { u } _ { h }$  alone</td><td>0.250</td><td>0.222</td><td>0.461</td><td>0.437</td></tr><tr><td>monotone gates</td><td>0.238</td><td>0.206</td><td>0.448</td><td>0.419</td></tr><tr><td>Chow-style rule,  $\operatorname { E q . } \left( 2 \right)$ </td><td>0.229</td><td>0.194</td><td>0.440</td><td>0.408</td></tr></table>

Table 3. Ablations (DFEW 5-fold means). (a) The objectives differ in what S can do, not in ${ \mathrm { N L L } } ; u _ { e } { \mathrm { - } } { \mathrm { A U R O C } }$ scores $u _ { e }$ as a detector of misclassified clips (positives: misclassified); AURC uses the same reject rule for both arms, so the KL-to-mean value differs from Table 2. (b) Learned head versus parameter-free Dirichlet proxies. (c) Decomposed signals against single scalars.

## 4.3. Ablations

(a) Concentration supervision. With identical backbone, adapter and heads, replacing $\mathrm { K L } ( \mathbf { n } / N \lVert \hat { \mathbf { p } } )$ by $\mathcal { L } _ { \mathrm { D M } }$ leaves NLL within 0.02 nats but raises the AUROC with which $u _ { e }$ detects misclassified clips from 0.51 (chance) to 0.64, and lowers ECE and AURC by 0.016 and 0.009 under the same reject rule; of the 0.020 AURC gap in Table 2, 0.011 is therefore due to the rule and 0.009 to the objective. (b) Redundancy of $\hat { u } _ { h } .$ The learned head reaches $\rho = 0 . 5 2$ against test-clip entropy versus 0.45 for $\mathrm { H } ( \hat { \mathbf { p } } )$ and 0.38 for MI, and the gap survives transfer (AURC 0.293 against 0.298 on FERV39k), so the head is kept. (c) Decomposition. No single signal matches the fused rule: $u _ { e }$ alone is weak because low evidence is only one source of error, and $\hat { u } _ { h }$ alone also rejects clips that are ambiguous but reliably modelled; the interpretable gates recover 45% of the gap. (d) Design choices. $\mathcal { L } _ { \mathrm { c a l } }$ gives ECE 0.036 against 0.038 for temperature scaling alone, a gain within fold variance, so it is a minor design choice; four temporal views reach AURC 0.108 against 0.107 for eight at half the cost and 0.110 for a learned window.

<table><tr><td>Setting</td><td>WAR↑</td><td>UAR↑</td><td>ECE↓</td><td> $\mathbf { A U R C } _ { \mathrm { s m } }$ </td><td>↓  $\mathbf { A U R C } _ { \mathrm { o u r s } } \downarrow$ </td></tr><tr><td>frozen, zero-shot</td><td>40.2</td><td>31.5</td><td>0.148</td><td>0.412</td><td>0.390</td></tr><tr><td>linear probe</td><td>45.8</td><td>35.9</td><td>0.092</td><td>0.356</td><td>0.336</td></tr><tr><td>fine-tune</td><td>51.1</td><td>41.6</td><td>0.063</td><td>0.296</td><td>0.276</td></tr><tr><td>fine-tune, rebalanced priors</td><td>50.7</td><td>43.4</td><td>0.059</td><td>0.292</td><td>0.271</td></tr></table>

Table 4. Transfer from DFEW to FERV39k (single labels: calibration and abstention only). $\mathrm { A U R C } _ { \mathrm { s m } }$ ranks clips by softmax confidence, $\mathrm { { A U R C } _ { \mathrm { { o u r s } } } }$ by Eq. (2). Zero-shot: normalisation statistics, $( { \mathbf { w } } , b )$ , isotonic map and t transferred unchanged from DFEW; other rows re-fitted on the FERV39k validation split; the last row additionally re-balances class priors.

![](images/3ccb7dfad8ef23ac6d0e663e88917b59b94e4ea65da9c3d3dc646b865184e96e.jpg)  
Fig. 3. Four DFEW test clips: illustrative qualitative cases, not evidence of a general separation between ambiguity, quality and instability. Bars in DFEW class order (Ha, Sa, Ne, An, Su, Di, Fe = happy, sad, neutral, angry, surprise, disgust, fear): vote frequencies $\mathbf { n } / N$ (amber) and predicted mean pˆ (red, outlined); ${ u } _ { h } \ = \ \mathrm { H } ( { \mathbf { n } } / \dot { N } ) /$ log K is the test-clip annotation entropy, for reference only. (A) clear consensus and (B) clean but genuinely mixed, both answered; (C) occluded, dark input, rejected through $u _ { q } ;$ (D) blurred, unstable, confidently wrong, rejected by the fused score.

## 4.4. Transfer and open set

Table 4 evaluates the selective-prediction rule under shift (Table 3c gives selective risks at two coverages; the full coverage–risk curves keep the same ordering). For zero-shot FERV39k, the feature normalisation statistics, rejector weights (w, b), isotonic calibration map, and threshold t<sup>∗</sup> are all transferred unchanged from DFEW. For the other FERV39k settings, these quantities are re-fitted using the FERV39k validation split. Ranking FERV39k clips by Eq. (2) rather than by softmax confidence lowers AURC by 5 to 7% in every setting (0.412 to 0.390 zero-shot, 0.296 to 0.276 fine-tuned), and the gain survives the prior-rebalancing control. On MAFW (open set), $\hat { u } _ { h }$ separates compound-emotion from single-label clips with an AUROC of 0.71; compound labels are a coarse proxy for ambiguity, not vote counts, so this is class-disjoint supporting evidence rather than a direct validation of disagreement prediction. The reject rule abstains on 41% of clips from the four classes absent from DFEW but on only 14% of those from the seven shared classes. Per class, $\hat { u } _ { h } , u _ { t }$ and the abstention rate all peak on disgust and fear, the two classes with the lowest annotator agreement in DFEW. Qualitative cases. Fig. 3 illustrates, on four selected clips, the situations the four signals are meant to separate. In (A) every signal is low. In (B) human disagreement is high and $\hat { u } _ { h }$ tracks it, but $u _ { e } , u _ { t }$ and $\boldsymbol { u } _ { \boldsymbol { q } }$ stay low, so the rule answers with a broad pˆ that matches the split panel: the case in which $\hat { u } _ { h }$ alone should not trigger abstention (Sec. 3.4). (C) and (D) are the failure modes outside the fitted Dirichlet, a degraded input and a temporally unstable, confidently wrong pˆ; each is rejected by a different term of ρˆ, and neither is caught by max<sub>k</sub> $\hat { p } _ { k }$

## 5. CONCLUSION

We introduced a disagreement-aware DFER framework trained on annotator count vectors with a Dirichlet–Multinomial likelihood. Unlike mean-only supervision, it also supervises a concentrationdependent uncertainty signal at no cost in accuracy, and improves calibration and selective prediction on DFEW, under shift and on stricter splits.

## 6. REFERENCES

[1] Xingxun Jiang, Yuan Zong, Wenming Zheng, et al., “DFEW: A large-scale database for recognizing dynamic facial expressions in the wild,” in Proc. ACM Int. Conf. Multimedia (ACM MM), 2020, pp. 2881–2889.

[2] Hanyang Wang, Bo Li, Shuang Wu, et al., “Rethinking the learning paradigm for dynamic facial expression recognition,” in Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR), 2023, pp. 17958–17968.

[3] Licai Sun, Zheng Lian, Bin Liu, and Jianhua Tao, “MAE-DFER: Efficient masked autoencoder for self-supervised dynamic facial expression recognition,” in Proc. ACM Int. Conf. Multimedia (ACM MM), 2023, pp. 6110–6121.

[4] Zengqun Zhao and Ioannis Patras, “Prompting visual-language models for dynamic facial expression recognition,” in Proc. British Machine Vision Conf. (BMVC), 2023.

[5] Yin Chen, Jia Li, Shiguang Shan, et al., “From static to dynamic: Adapting landmark-aware image models for facial expression recognition in videos,” IEEE Transactions on Affective Computing, 2024.

[6] Alexandra N. Uma, Tommaso Fornaciari, Dirk Hovy, et al., “Learning from disagreement: A survey,” Journal ofArtificial Intelligence Research, vol. 72, pp. 1385–1470, 2021.

[7] Barbara Plank, “The “problem” of human label variation: On ground truth in data, modeling and evaluation,” in Proc. Conf. Empirical Methods in Natural Language Processing (EMNLP), 2022, pp. 10671–10682.

[8] Murat Sensoy, Lance Kaplan, and Melih Kandemir, “Evidential deep learning to quantify classification uncertainty,” in Advances in Neural Information Processing Systems (NeurIPS), 2018.

[9] Wentao Bao, Qi Yu, and Yu Kong, “Evidential deep learning for open set action recognition,” in Proc. IEEE/CVF Int. Conf. Computer Vision (ICCV), 2021, pp. 13349–13358.

[10] Dennis Ulmer, Christian Hardmeier, and Jes Frellsen, “Prior and posterior networks: A survey on evidential deep learning methods for uncertainty estimation,” Transactions on Machine Learning Research, 2023.

[11] Junyu Gao, Mengyuan Chen, Liangyu Xiang, and Changsheng Xu, “A comprehensive survey on evidential deep learning and its applications,” arXiv preprint arXiv:2409.04720, 2024.

[12] Joshua C. Peterson, Ruairidh M. Battleday, Thomas L. Griffiths, and Olga Russakovsky, “Human uncertainty makes classification more robust,” in Proc. IEEE/CVF Int. Conf. Computer Vision (ICCV), 2019, pp. 9617–9626.

[13] Ryosuke Kawamura, Hideaki Hayashi, Noriko Takemura, and Hajime Nagahara, “MIDAS: Mixing ambiguous data with soft labels for dynamic facial expression recognition,” in Proc. IEEE/CVF Winter Conf. Applications of Computer Vision (WACV), 2024, pp. 6552–6562.

[14] Jiahui She, Yibo Hu, Hailin Shi, et al., “Dive into ambiguity: Latent distribution mining and pairwise uncertainty estimation for facial expression recognition,” in Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR), 2021, pp. 6248–6257.

[15] Nhat Le, Khanh Nguyen, Quang Tran, et al., “Uncertaintyaware label distribution learning for facial expression recognition,” in Proc. IEEE/CVF Winter Conf. Applications of Computer Vision (WACV), 2023, pp. 6088–6097.

[16] Shu Liu, Yan Xu, Tongming Wan, and Xiaoyan Kui, “A dualbranch adaptive distribution fusion framework for real-world facial expression recognition,” in Proc. IEEE Int. Conf. Acoustics, Speech and Signal Processing (ICASSP), 2023.

[17] Feng Liu, Hanyang Wang, and Siyuan Shen, “Robust dynamic facial expression recognition,” IEEE Transactions on Biometrics, Behavior, and Identity Science, vol. 7, no. 4, pp. 563–572, 2025.

[18] Haodong Chen, Haojian Huang, Junhao Dong, et al., “FineCLIPER: Multi-modal fine-grained CLIP for dynamic facial expression recognition with AdaptERs,” in Proc. ACM Int. Conf. Multimedia (ACM MM), 2024, pp. 2301–2310.

[19] Yin Chen, Jia Li, Yu Zhang, et al., “Static for dynamic: Towards a deeper understanding of dynamic facial expressions using static expression data,” IEEE Transactions on Affective Computing, 2025.

[20] Kateryna Chumachenko, Alexandros Iosifidis, and Moncef Gabbouj, “MMA-DFER: Multimodal adaptation of unimodal models for dynamic facial expression recognition in-the-wild,” in Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition Workshops (CVPRW), 2024, pp. 4673–4682.

[21] Urja Khurana, Eric Nalisnick, Antske Fokkens, and Swabha Swayamdipta, “Crowd-Calibrator: Can annotator disagreement inform calibration in subjective tasks?,” in Proc. Conf. Language Modeling (COLM), 2024.

[22] Linwei Tao, Haoyang Luo, Minjing Dong, and Chang Xu, “Confidence calibration under ambiguous ground truth,” arXiv preprint arXiv:2603.22879, 2026.

[23] Keito Inoshita and Takato Ueno, “Interpretable uncertainty routing separating emotion ambiguity from distribution shift in facial expression recognition,” arXiv preprint arXiv:2606.22725, 2026.

[24] Fan Cui, Anqi Tong, Jun Huang, et al., “Toward trustworthy dynamic facial expression recognition via information bottleneck modeling,” IEEE Transactions on Information Forensics and Security, vol. 21, pp. 6985–6999, 2026.

[25] Peter Sadowski and Pierre Baldi, “Neural network regression with Beta, Dirichlet, and Dirichlet-multinomial outputs,” OpenReview preprint, 2019.

[26] Zhan Tong, Yibing Song, Jue Wang, and Limin Wang, “Video-MAE: Masked autoencoders are data-efficient learners for selfsupervised video pre-training,” in Advances in Neural Information Processing Systems (NeurIPS), 2022.

[27] Andrey Malinin and Mark Gales, “Predictive uncertainty estimation via prior networks,” in Advances in Neural Information Processing Systems (NeurIPS), 2018.

[28] C. K. Chow, “On optimum recognition error and reject tradeoff,” IEEE Transactions on Information Theory, vol. 16, no. 1, pp. 41–46, 1970.

[29] Ranganath Krishnan and Omesh Tickoo, “Improving model calibration with accuracy versus uncertainty optimization,” in Advances in Neural Information Processing Systems (NeurIPS), 2020.

[30] Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q. Weinberger, “On calibration of modern neural networks,” in Proc. Int. Conf. Machine Learning (ICML), 2017, pp. 1321–1330.

[31] Yan Wang, Yixuan Sun, Yiwen Huang, et al., “FERV39k: A large-scale multi-scene dataset for facial expression recognition in videos,” in Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR), 2022, pp. 20922–20931.

[32] Yuanyuan Liu, Wei Dai, Chuanxu Feng, et al., “MAFW: A large-scale, multi-modal, compound affective database for dynamic facial expression recognition in the wild,” in Proc. ACM Int. Conf. Multimedia (ACM MM), 2022, pp. 24–32.

[33] Yonatan Geifman, Guy Uziel, and Ran El-Yaniv, “Biasreduced uncertainty estimation for deep neural classifiers,” in Proc. Int. Conf. Learning Representations (ICLR), 2019.