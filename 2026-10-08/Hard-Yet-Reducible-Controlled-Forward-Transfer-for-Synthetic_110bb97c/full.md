# Hard, Yet Reducible: Controlled Forward Transfer for Synthetic Degradation Curation

Chunming He<sup>1</sup>,<sup>2</sup>, Kailai Zhou<sup>3</sup>, Jiaming Zuo<sup>1</sup>,<sup>2</sup>, Hanqi Liu<sup>3</sup>,<sup>2</sup>, Fengyang Xiao<sup>1</sup>, Youwei Pang<sup>2</sup>, Xiaofeng Liu<sup>4</sup>, Weisi Lin<sup>3</sup>, Xiaoqi Zhao<sup>3</sup>

<sup>1</sup>Tsinghua University, China <sup>2</sup>X3000 Inspection Co., Ltd <sup>3</sup>AI4X Team, Nanyang Technological University, Singapore <sup>4</sup>Yale University, USA

Selecting synthetic degradations for dense prediction requires an estimate of their training utility, the generalization gain they bring under a finite training budget. Clean and degraded twins share content and labels, suggesting a score based on how much short training reduces the excess error caused by degradation. However, this gap can also shrink when clean performance deteriorates. Measuring the improvement on degraded images alone avoids that confound, but it still credits progress that the same amount of clean training would have produced. We propose the contro<sup>ll</sup>ed Reducib<sup>l</sup>e Degradation Gap (cRDG) for regions defined by degradation type and severity. From a common checkpoint, cRDG runs two budget-matched probes that difer only in one augmentation slot, which holds either a synthetic degradation or a clean augmentation. The score is the gain on held-out degraded images relative to the clean-control probe. Clean harm is a separate feasibility constraint. cRDG reveals a correctable severity band in which training on the degradation yields high controlled gain under the available budget, and the band moves with the predictor, the starting checkpoint, and the training budget. Curation o<sup>f</sup> Reducib<sup>l</sup>e Bands (CuRB) uses cRDG to select synthetic data without changing the predictor. On semantic segmentation and salient object detection, CuRB improves representative predictors under matched synthetic-data budgets and training schedules, extends to existing data-generation pipelines, and preserves clean performance. Code and supporting materials will be publicly released.

## 1 Introduction

Generators can now produce unlimited variants of labelled data. The binding question is no longer how to synthesize but what to spend a finite training budget on [1, 2]. Under a fixed schedule, every degraded sample occupies a slot that could hold a clean augmentation instead. We take this alternative as the reference, and the value of a degradation is the extra generalization from degradation.

Data selection often conflates visual quality, instantaneous dificulty, and training utility. The first two describe the current sample or the model’s response to it, whereas training utility is the gain obtained by actually training on that sample. Utility is therefore the quantity relevant to selection, and it depends on the model state, the optimizer, and the training budget. We study this quantity for synthetic degradation in dense prediction, where degraded variants of clean labelled images keep scene content, geometry, and labels fixed [3–5].

Higher visual fidelity does not guarantee higher downstream utility, and recent studies attribute this mismatch to limited support and diversity in generated data [6, 7]. Within a plausible candidate pool, realism acts as a support constraint rather than a ranking criterion. Dificulty is also a poor indicator of utility, because a strong degradation can remove the evidence needed to recover the mask, leaving samples that are hard yet carry little learnable signal under the available budget. Selecting by loss or uncertainty [8, 9] can then waste the budget or cause negative transfer.

![](images/66e97e02ae4bdc737a609d6b235e982bf7df8d2e7e54cf81e8a389f1dbbc44ca.jpg)  
Figure 1. W<sup>h</sup>y Matc<sup>h</sup>ed C<sup>l</sup>ean Contro<sup>l</sup> Matters. (a) Paired gap reduction can be inflated by clean forgetting. (b) Held-out degraded loss drop avoids this inflation but still credits generic training progress. (c) Our matched-control formulation underlying cRDG measures degraded gain $G _ { g }$ beyond equally long clean training, while checking clean harm $\dot { H } _ { g }$ separately.

We define the utility of a degradation as its held-out gain beyond equally long clean training under the same probe budget, subject to a clean-retention constraint. Forward transfer here means the gain on held-out source scenes that carry the same degradation after a short probe, and transfer to real adverse domains is evaluated separately (§ 4.4). Curation uses only source images and their synthetic twins, never target-domain observations. We group candidates into regions � defined by degradation type and severity bin, so the policy chooses kinds of degradation rather than individual scenes. Full retraining for every region would be costly, so short probes estimate held-out gains per region. Synthetic degradation also ofers a natural control, since each valid degraded image has a clean twin with the same scene and label. Existing valuation scores, such as influence and gradient alignment [10, 11], holdout-loss prioritization [8], early-training statistics [12, 13], and learnability-guided generation [9, 14], do not combine this twin structure with a clean-control stream and held-out transfer at the region level.

This pairing suggests comparing degraded and clean loss for the same content. Let $L _ { d } ( \theta _ { t } )$ and $L _ { c } ( \theta _ { t } )$ be the mean losses on the degraded twins of a region and on their clean counterparts at checkpoint $\theta _ { t }$ . The gap $\Delta _ { t } = L _ { d } ( \theta _ { t } ) - L _ { c } ( \theta _ { t } )$ is the excess error caused by the degradation. One could score a region by how much a short probe shrinks this gap:

$$
\Delta _ { 0 } - \Delta _ { 1 } = \underbrace { [ L _ { d } ( \theta _ { 0 } ) - L _ { d } ( \theta _ { 1 } ) ] } _ { \mathrm { d e g r a d e d ~ l o s s ~ f a l l s } } - \underbrace { [ L _ { c } ( \theta _ { 0 } ) - L _ { c } ( \theta _ { 1 } ) ] } _ { \mathrm { c l e a n ~ l o s s ~ f a l l s } } .\tag{1}
$$

The second bracket exposes the problem. A probe that forgets clean scenes also shrinks the gap, so clean forgetting inflates the score (Theorem 3.1). Dropping the clean term and scoring only the fall in degraded loss removes this inflation. It still credits progress that equally long clean training would have produced anyway. The quantity we need is the gain beyond that clean-control trajectory (Fig. 1).

We estimate this gain with matched treatment and clean-control probes. Both start from the same checkpoint and difer only in whether the designated augmentation slot contains a degradation or a clean augmentation. They are evaluated on held-out twins, and the two halves are swapped to obtain the contro<sup>ll</sup>ed Reducib<sup>l</sup>e Degradation Gap (cRDG). Scores stay signed so negative transfer remains visible, and a separate clean-harm constraint excludes regions that damage clean performance.

cRDG shows that utility is not monotonic in severity. Each degradation type has a correctable band in which training on the degradation yields high controlled gain under the budget, and the severity centroid of this band shifts with probe budget �, starting checkpoint �<sub>0</sub>, and predictor setting, and difers across degradation types. Per-region retraining at three final-training budgets shows the same shift in real-target gains, and proportionally scaled probes track it (§ 4.4). A fixed middle-severity rule (Fixed-Mid) serves as the sanity check for this claim throughout the experiments.

Around cRDG we build Curation o<sup>f</sup> Reducib<sup>l</sup>e Bands (CuRB), a curation framework that adds degradation-aware selection to existing dense predictors without changing their architectures. Scores are recomputed for the operating checkpoint and budget. The band only summarizes the profile. Selection ranks feasible positive-score regions directly and fills a fixed synthetic budget in rank order, with validity gating before ranking and <sup>�</sup>-center sampling for the last partial region.

Our contributions are as follows.

• We formulate budget-conditioned correctability. Training utility need not track severity, realism, or dificulty, and the useful severity region migrates with predictor, checkpoint, and probe budget.

• We propose cRDG, a held-out, budget-matched estimate of the gain from training on a degradation region relative to an equally long clean-control probe, with clean harm handled separately.

• We introduce CuRB, a plug-and-play framework that adds degradation-aware curation to existing dense predictors without changing their architectures.

• Under a matched protocol, CuRB outperforms region-level adaptations of existing selection methods. Across SemSeg and SOD, its gains extend to the evaluated data-generation settings while preserving clean performance.

## 2 Related Work

Synt<sup>h</sup>etic Degradation and Degradation-Aware Training. Difusion-based methods synthesize degradation patterns and weather conditions [5, 15]. Restoration-driven methods [16] inject enhancement knowledge into downstream segmentation. Synthesis frameworks such as DGInStyle [17] and CA-LoRA [18] generate labelled training images for segmentation. These approaches decide how training data are generated or used. We ask which synthetic degradations deserve a finite training budget. Higher fidelity alone does not guarantee training value. Adamkiewicz et al. [6] link the mismatch to limited coverage and diversity, and Zhang et al. [7] show that scene composition and instance fidelity matter for segmentation. We instead hold content and labels fixed and vary only degradation type and severity, which allows a controlled estimate of degradation-specific utility. CuRB is a curation layer between an existing generator and an existing predictor and changes neither. We evaluate it with ISSA [19], procedural weather corruption [20], and Gen4Seg [21] under their original settings. UDA methods [22–24] instead use unlabelled target images during adaptation. Our setting uses only source labels and synthetic twins during curation and training.

Data Va<sup>l</sup>uation and Learnabi<sup>l</sup>ity-Guided Se<sup>l</sup>ection. Influence- and gradient-based methods [10, 11] estimate how much a sample contributes to learning. RHO-Loss [8] prioritizes examples by their efect on holdout loss. Early-training statistics such as EL2N, GraNd, and forgetting events [12, 13] find informative examples from short trajectories. Learnability-guided generation [9, 14] steers synthesis with learner feedback. Our setting difers in that utility is estimated at the region level using content-matched twins, a clean-control probe, and held-out transfer under a fixed budget (§ 3.3). Difusion Curriculum [25] is closest in spirit and adapts the guidance level of generated data using validation feedback. It does not compare against an equally long clean-control probe, which is the counterfactual our score is built on.

![](images/e5d893860f67bf948f41bd3b9105681d11488e16a557a722341306393d3c585c.jpg)  
Figure 2. Overview o<sup>f</sup> CuRB. (a) Degradation-severity regions are generated and validity-filtered. (b) Two matched probes start from $\theta _ { 0 }$ and difer only in the augmentation slot, where <sup>⊕</sup> denotes the union of the clean-replay stream and the slot stream. (c) Held-out evaluation yields degraded gain $G _ { g }$ and clean harm $H _ { g } .$ Averaging $G _ { g }$ over the two split directions gives cRDG. $H _ { g }$ determines feasibility, while the correctable band is diagnostic only. (d) Feasible positive-score regions are ranked and fill the budget $\boldsymbol { B } _ { \mathrm { s e l } }$ in rank order, with <sup>�</sup>-center sampling for the last partial region. (e) Final training restarts from $\theta _ { 0 }$ on clean data, with $S ^ { \star }$ in the synthetic slots under the fixed schedule.

## 3 Controlled Reducibility for Degradation Curation

CuRB selects synthetic degradations for a given predictor and training budget without changing the predictor. Fig. 2 shows the pipeline. Matched probes estimate a region-level utility, cRDG, and selection ranks regions by cRDG under a clean-harm constraint and a fixed synthetic-data budget.

## 3.1 Candidate Regions and Validity

We are given a labelled clean source set $\mathcal { D } _ { s } = \{ ( x _ { i } , y _ { i } ) \} _ { i = 1 } ^ { N }$ . A generator $\mathcal { G }$ produces degraded twins $\tilde { x } _ { i , d , s } = G ( x _ { i } ; d , s , z )$ with degradation type $d ,$ severity level $s ,$ and stochastic detail �. We use four appearance degradations that preserve object geometry after validity filtering. Low light and backlight cover illumination changes, and fog and rain cover weather efects. Each family is divided into eight ordered severity levels, giving 32 regions. This discretization resolves severity-dependent efects while keeping the probe set small. A degradation region $g = ( d , s )$ contains candidates from one family and one severity bin, where “region” denotes a group in the candidate pool rather than a spatial image region. CuRB ranks regions, not individual scenes. The region definition is shared across architectures. Its estimated utility is recomputed for each model and budget.

Va<sup>l</sup>idity gate. Severe degradation does not by itself invalidate the mask. We therefore separate label validity from reducibility. A binary gate rejects candidates only when generation changes scene structure. It checks frozen DINO feature correspondence near ground-truth boundaries and passes a uniformly darker image with intact geometry. Details and the gate audit are in Appendix B.

T<sup>h</sup>e twin gap. Write $L _ { d } ^ { S } ( \theta ; g )$ for the mean loss on the degraded twins of an index set � and $L _ { c } ^ { S } ( \theta )$ for the mean loss on their clean images. At the source checkpoint $\theta _ { 0 } ,$ , the gap $\Delta _ { 0 } ( g ) = L _ { d } ^ { S } ( \theta _ { 0 } ; g ) - L _ { c } ^ { S } ( \theta _ { 0 } )$ isolates the error caused by the degradation, since twins share scene, geometry, and label. It measures what is not yet mastered, rather than what can be mastered within a given budget.

## 3.2 Matched-Control Probes

Consider a short probe $\theta _ { 0 } \to \theta _ { 1 }$ on region $^ { g , }$ scored by the shrinkage $\Delta _ { 0 } - \Delta _ { 1 }$ . By Equation (1) this equals the drop in degraded loss minus the drop in clean loss, so it rewards clean forgetting: Proposition 3.1 (Gap reduction can reward clean forgetting). $\Delta _ { 0 } - \Delta _ { 1 } > 0$ can hold even when the degraded loss worsens, $L _ { d } ( \theta _ { 1 } ) > L _ { d } ( \theta _ { 0 } )$ , whenever $L _ { c } ( \theta _ { 1 } ) - L _ { c } ( \theta _ { 0 } ) > L _ { d } ( \theta _ { 1 } ) - L _ { d } ( \theta _ { 0 } ) > 0 .$

Scoring the degraded-loss drop alone removes this inflation but still credits progress that equally long clean training would produce. We therefore compare each degradation probe with an equally long clean-control probe. We report the uncontrolled score as the baseline RDG, defined together with cRDG in § 3.3.

To measure transfer rather than fit to the probe samples, we split source image IDs once into halves � and $B ,$ train on one half, and evaluate on the other (cross-fitting). The held-out half is excluded from the current probe update, although it may have contributed to training $\theta _ { 0 }$ . The split is reused for every region. Let $\bar { U } ( \theta , S ; K )$ denote � optimizer steps from $\theta$ on a data stream ${ \bar { S } } ,$ and let <sup>⊕</sup> denote the union of the clean-replay stream and the augmentation-slot stream in each probe batch. From the common start $\theta _ { 0 }$ we run two probes on half $A _ { , }$

$$
\theta _ { A , g } ^ { \deg } = U \big ( \theta _ { 0 } , ~ A _ { \mathrm { c l e a n } } \oplus A _ { \deg , g } ; K \big ) , \qquad \theta _ { A } ^ { \operatorname { c t r l } } = U \big ( \theta _ { 0 } , ~ A _ { \mathrm { c l e a n } } \oplus A _ { \operatorname { a u g } } ; K \big ) .\tag{2}
$$

The treatment probe fills the slot with twins from region $g .$ The clean-control probe fills the same slot with a weak photometric augmentation of the same base view (Appendix C). Both probes share source IDs, steps, batch composition, clean-replay ratio, optimizer, schedule, data order, and initialization. The control does not depend on �, so it is run once per direction and shared across all regions. If no valid twin exists after $n _ { z }$ regeneration draws, the control view is kept in the treatment slot, so the match is never broken. Held-out twins that fail the gate are dropped from both evaluations.

Operating ru<sup>l</sup>e <sup>f</sup>or �. We set $K$ to $1 / 8 0$ of the final training schedule and scale it with the schedule, so the probe-to-training budget ratio stays fixed across operating points.

## 3.3 Matched-Control Reducibility

Both probes are evaluated on the held-out half �. We define the degraded gain and the clean harm as

$$
\begin{array} { r } { G _ { g } ^ { A  B } = L _ { d } ^ { B } ( \theta _ { A } ^ { \operatorname { c t r l } } ; g ) - L _ { d } ^ { B } ( \theta _ { A , g } ^ { \operatorname { d e g } } ; g ) , \qquad H _ { g } ^ { A  B } = [ L _ { c } ^ { B } ( \theta _ { A , g } ^ { \operatorname { d e g } } ) - L _ { c } ^ { B } ( \theta _ { A } ^ { \operatorname { c t r l } } ) ] _ { + } . } \end{array}\tag{3}
$$

� is positive when degradation training outperforms the matched clean control on held-out degraded images. � is positive when the treatment increases the held-out clean loss. We then swap the two halves and average the two held-out gains,

$$
\begin{array} { r } { \mathrm { c R D G } ( g ; K , \theta _ { 0 } ) = \frac { 1 } { 2 } \big ( G _ { g } ^ { A \to B } + G _ { g } ^ { B \to A } \big ) . } \end{array}\tag{4}
$$

A region isfeasible if max $( H _ { g } ^ { A  B } , H _ { g } ^ { B  A } ) \leq \epsilon$ . Infeasible regions are excluded from selection, and signed scores remain available for analysis. We use a hard constraint rather than a penalty $G _ { g } - \lambda H _ { g }$ because it enforces clean retention without another trade-of parameter. The constraint bounds probe-time harm and does not guarantee final clean accuracy, which § 4.2 checks empirically.

Base<sup>l</sup>ines <sup>f</sup>rom t<sup>h</sup>e same probes. The same held-out losses define two baselines. The paired gap reduction is $\mathrm { R D G } _ { g } ^ { A  B } = [ L _ { d } ^ { B } ( \theta _ { 0 } ; g ) - L _ { d } ^ { B } ( \theta _ { A , g } ^ { \mathrm { d e g } } ; g ) ] - [ L _ { c } ^ { B } ( \theta _ { 0 } ) - L _ { c } ^ { B } ( \theta _ { A , g } ^ { \mathrm { d e g } } ) ]$ , and the held-out loss drop is $D _ { g } ^ { A \to B } = L _ { d } ^ { B } ( \theta _ { 0 } ; g ) - L _ { d } ^ { B } ( \theta _ { A , g } ^ { \mathrm { d e g } } ; g )$ . Like $G _ { g } ,$ , both are averaged over the directions $A  B$ and $B  A . \ D _ { g }$ difers from cRDG only in what is subtracted, the initial loss instead of the clean-control loss, so it is the direct ablation of the control branch, whereas RDG additionally credits clean forgetting. cRDG is a short-horizon, source-side estimate of controlled degradation utility. Its relation to full-training gains on real target domains is evaluated empirically in § 4.4.

## 3.4 Curation and Final Training

Se<sup>l</sup>ection. CuRB first removes infeasible and non-positive regions. The remaining regions are ranked by cRDG and filled in descending order until $\boldsymbol { B } _ { \mathrm { s e l } }$ is reached. Each region is exhausted before moving to the next. If only part of the final region is needed, we use feature-space <sup>�</sup>-center sampling. Any unfilled slots revert to clean data, so the total number of training samples is unchanged.

Fina<sup>l</sup> objective. The curation procedure can be used with diferent final-training objectives, and all matched-curation methods share the one below. Starting again from $\theta _ { 0 }$ , we train on clean data plus $S ^ { \star }$ with

$$
\mathcal { L } = \mathcal { L } _ { \mathrm { s e g } } ( f ( x _ { i } ) , y _ { i } ) + \lambda _ { s } \mathcal { L } _ { \mathrm { s e g } } ( f ( \tilde { x } _ { i } ) , y _ { i } ) + \lambda _ { c } \sum _ { p } w _ { i } ( p ) \mathrm { K L } \bigl ( f ( x _ { i } ) _ { p } \| f ( \tilde { x } _ { i } ) _ { p } \bigr ) .\tag{5}
$$

Here $w _ { i } ( p ) = M _ { i } ( p ) ( 1 + \lambda _ { b } B _ { y _ { i } } ( p ) ) , B _ { y _ { i } } ( p )$ marks a ground-truth boundary band, and $M _ { i } ( p )$ is the cosine similarity between the clean DINOv2 feature at $p$ and the degraded feature at its matched location $q _ { p } ,$ mapped linearly to [0<sub>,</sub> 1]. The KL term compares class-probability maps. Matched-curation experiments use Equation (5) unchanged, and plug-in experiments keep each host’s own objective.

Cost. With � regions and � split directions, curation costs �� treatment probes plus � shared controls. Changing the predictor or the schedule requires recomputing cRDG but never changes the architecture. § 4.2 reports compute matched between CuRB and the baselines.

## 3.5 The Diagnostic Severity Band

cRDG depends on the probe budget and the checkpoint, not on severity alone. For a type <sup>�</sup> with bins $s _ { 1 } < \cdots < s _ { N _ { s } }$ , let $\mathcal { F } _ { d }$ contain the feasible bins with positive cRDG. The correctable band is the set of bins whose score is at least half of the family maximum,

$$
\begin{array} { r } { \mathcal { B } _ { d } ( K , \theta _ { 0 } ) = \left\{ s \in \mathcal { F } _ { d } : \mathrm { c R D G } \big ( ( d , s ) ; K , \theta _ { 0 } \big ) \geq \kappa \operatorname* { m a x } _ { s ^ { \prime } \in \mathcal { F } _ { d } } \mathrm { c R D G } \big ( ( d , s ^ { \prime } ) ; K , \theta _ { 0 } \big ) \right\} , } \end{array}\tag{6}
$$

with $\kappa = 1 / 2$ . For smoother comparison across operating conditions, we also report the cRDGweighted severity centroid $\bar { s } _ { d } ( K , \theta _ { 0 } ) \ = \ \sum _ { s \in \mathcal { F } _ { d } }$ � $\begin{array} { r } { \mathrm { c R D G } ( ( d , s ) ; K , \theta _ { 0 } ) \big / \sum _ { s \in \mathcal { F } _ { d } } \mathrm { c R D G } ( ( d , s ) ; K , \theta _ { 0 } ) } \end{array}$ which is undefined when $\mathcal { F } _ { d }$ is empty. Severity is represented by normalized bin centers $s _ { k } = ( 2 k - 1 ) / 1 6 , k = 1 , \dots , 8$ . Both are diagnostics, and selection ranks regions directly by cRDG. The band-migration hypothesis is that $\bar { s } _ { d }$ shifts with the probe budget $K ,$ the checkpoint, and the predictor, and difers across types. § 4.4 tests whether the real band shifts with the training budget and whether scaled-� cRDG tracks it.

## 4 Experiments

## 4.1 Setup

Data and protoco<sup>l</sup>s. We use two core evaluation protocols. For SOD, WXSOD [26] and CSOD10K [27] follow their oficial splits and metrics, with each base model using its published training recipe. Augmented variants use the same budget of degraded twins from clean DUTS-TR [30] and the same final-training schedule (Appendix E). For SemSeg, we follow the Cityscapes<sup>→</sup>ACDC / Dark Zurich protocol of ISSA [19], also used by Gen4Seg [21]. Cityscapes [31] is the only source, and SegFormer follows the released ISSA configuration (Appendix A). The matched synthetic pool contains low light, backlight, fog, and rain at eight severity levels per family. Low light is aligned with the night condition, while fog and rain have direct counterparts in ACDC [32]. Backlight is an additional source-side illumination family, and snow is held out entirely for unseen-family evaluation. We report the in-family mean over fog, night, and rain, and evaluation also includes Dark Zurich [24] and Cityscapes val. Plug-in experiments apply CuRB under each host’s published setting and synthetic-data budget. Operational curation uses one fixed probe seed and one fixed �|� partition. Two additional partitions are used only for stability analysis (Appendix C).

<table><tr><td>Method</td><td>M↓</td><td> $F _ { \beta } ^ { \operatorname* { m a x } } \uparrow$ </td><td> $E _ { \phi }$  ←</td><td> $S _ { \alpha } \uparrow$ </td></tr><tr><td>WFANet [26]</td><td>.016</td><td>.908</td><td>.953</td><td>.925</td></tr><tr><td>+ Uncurated twins  $( \boldsymbol { B } _ { \mathrm { s e l } } )$ </td><td>.015</td><td>.912</td><td>.956</td><td>.926</td></tr><tr><td> $+ \ C \mathbf { U } \mathbf { R } \mathbf { B } \left( B _ { \mathrm { s e l } } \right)$ </td><td>.013</td><td>.923</td><td>.963</td><td>.933</td></tr><tr><td>CSSAM [27]</td><td>.011</td><td>.937</td><td>.972</td><td>.940</td></tr><tr><td>+ Uncurated twins  $( \boldsymbol { B } _ { \mathrm { s e l } } )$ </td><td>.011</td><td>.939</td><td>.975</td><td>.942</td></tr><tr><td>+ CuRB  $( \boldsymbol { B } _ { \mathrm { s e l } } )$ </td><td>.009</td><td>.952</td><td>.987</td><td>.947</td></tr><tr><td>NUN [28]</td><td>.014</td><td>.920</td><td>.965</td><td>.933</td></tr><tr><td>+ Uncurated twins  $( \boldsymbol { B } _ { \mathrm { s e l } } )$ </td><td>.013</td><td>.924</td><td>.968</td><td>.935</td></tr><tr><td> $+ \ C \mathbf { U } \mathbf { R } \mathbf { B } \left( B _ { \mathrm { s e l } } \right)$ </td><td>.011</td><td>.934</td><td>.978</td><td>.938</td></tr></table>

(a) WXSOD-real [26].

<table><tr><td>Method</td><td>M↓</td><td> $F _ { \beta } ^ { \operatorname* { m a x } \mathrm { ~ . ~ } }$  ←</td><td> $E _ { \phi }$  ←</td><td> $S _ { \alpha }$  ←</td></tr><tr><td>SI-EDN [29]</td><td>.062</td><td>.817</td><td>.841</td><td>.810</td></tr><tr><td>+ Uncurated twins  $( \boldsymbol { B } _ { \mathrm { s e l } } )$ </td><td>.058</td><td>.835</td><td>.858</td><td>.817</td></tr><tr><td> $+ \mathrm { C u R B } \left( B _ { \mathrm { s e l } } \right)$ </td><td>.052</td><td>.853</td><td>.873</td><td>.831</td></tr><tr><td>CSSAM [27]</td><td>.035</td><td>.887</td><td>.916</td><td>.886</td></tr><tr><td>+ Uncurated twins  $( \boldsymbol { B } _ { \mathrm { s e l } } )$ </td><td>.034</td><td>.891</td><td>.919</td><td>.888</td></tr><tr><td> $+ \mathrm { C u R B } \left( B _ { \mathrm { s e l } } \right)$ </td><td>.030</td><td>.912</td><td>.934</td><td>.893</td></tr><tr><td>NUN [28] + Uncurated twins  $( \boldsymbol { B } _ { \mathrm { s e l } } )$ </td><td>.038 .036</td><td>.857</td><td>.896</td><td>.875</td></tr><tr><td> $+ \mathrm { C u R B } \left( B _ { \mathrm { s e l } } \right)$ </td><td>.032</td><td>.867 .880</td><td>.902 .917</td><td>.877 .886</td></tr></table>

(b) CSOD10K [27].

Tab<sup>l</sup>e 1. Eva<sup>l</sup>uation on SOD Benc<sup>h</sup>mar<sup>k</sup>s following the oficial protocol (§ 4.1). At the same augmentation budget, comparing uncurated sampling with CuRB isolates the gain from curation.
<table><tr><td rowspan="2">Data-generation host Published host setting</td><td rowspan="2"></td><td colspan="2">ACDC mIoU ↑</td><td colspan="2">Dark Zurich mIoU ↑</td><td>Cityscapes</td></tr><tr><td>Original</td><td>+CuRB</td><td>Original</td><td>+CuRB</td><td>val∆↑</td></tr><tr><td colspan="6">single-generator hosts</td><td></td></tr><tr><td>ISSA [19]</td><td>SegFormer</td><td>52.45</td><td>53.92</td><td>27.39</td><td>28.76</td><td>+0.24</td></tr><tr><td>H-Weather [20]</td><td>SegFormer</td><td>49.21</td><td>50.63</td><td>23.44</td><td>24.65</td><td>+0.36</td></tr><tr><td>Gen4Seg [21]</td><td>SegFormer</td><td>51.04</td><td>52.39</td><td>25.63</td><td>27.28</td><td>+0.45</td></tr><tr><td>composite DG host</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ISSA [19]</td><td>RobustNet (DeepLabV3+)</td><td>47.55</td><td>48.73</td><td>23.09</td><td>24.35</td><td>+0.18</td></tr></table>

Tab<sup>l</sup>e 2. CuRB as a <sup>l</sup>u -in to existin data- eneration settin s. Published references for ISSA, H-Weather, and RobustNet+ISSA are taken from Li et al. [19]. Gen4Seg follows Yin et al. [21] (its Dark Zurich value is obtained with the released pipeline). $\mathrm { O u r \ ^ { \prime \prime } { + } C u R B ^ { \prime \prime } }$ runs apply CuRB under the corresponding host setting and synthetic-data budget. The composite host stacks RobustNet [33] with ISSA on DeepLabV3+. Host-specific region construction: Appendix D.

Comparison groups. Benchmark baselines: WFANet, CSSAM, and NUN on WXSOD, and SI-EDN, CSSAM, and NUN on CSOD10K. Data-generation hosts: ISSA [19] with SegFormer and RobustNet [33] / DeepLabV3+ [34], Hendrycks-Weather [20], and Gen4Seg [21]. Selection baselines: realism (NR-IQA / CLIPScore), highest loss / uncertainty, RHO-Loss-style selection [8], gradient contribution [11], and a DisCL-style severity preference [25] (Appendix G). Twin-based scores: $\Delta _ { 0 } ,$ uncontrolled RDG, held-out loss drop $D _ { g } ,$ and cRDG. Fixed-Mid, the middle severity bins of every family, is a sanity control. Source-only and All-gen. are no-ranking anchors. Full-pool exposure appears only in the compute analysis.

Matc<sup>h</sup>ed protoco<sup>l</sup>. All selectors use the same gated pool, $B _ { \mathrm { s e l } } ,$ clean/degraded composition, initialization $\theta _ { 0 } ,$ objective, and training schedule. Sample-level baseline scores are averaged within each region before the common <sup>�</sup>-center step (Appendix G), so the comparison isolates region ranking. Compute-matched baselines spend CuRB’s curation budget on extended final training (§ 4.2).

![](images/104bc9dc8bafcb01e6298876e5eda80084996c5a3d89c4118f946b3ba4e3682d.jpg)

<table><tr><td rowspan="2">Method</td><td colspan="3">ACDC in-family</td><td rowspan="2">in-fam. mean</td><td rowspan="2">Snow (held out)</td><td rowspan="2">Dark Zurich</td><td rowspan="2">Cityscapes val</td></tr><tr><td>Fog</td><td>Night</td><td>Rain</td></tr><tr><td>Source-only</td><td>60.61</td><td>28.42</td><td>50.35</td><td>46.46</td><td>48.79</td><td>24.08</td><td>67.84</td></tr><tr><td>matched curation protocol: same pool, gate,</td><td></td><td>objective, and schedule</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>All-gen. (uniform,  $B _ { \mathrm { s e l } } )$ </td><td> $B _ { \mathrm { s e l } } ,$  63.82</td><td>31.51</td><td>52.78</td><td>49.37</td><td>50.15</td><td>26.55</td><td>68.21</td></tr><tr><td>Realism (NR-IQA / CLIPScore)</td><td>63.34</td><td>30.97</td><td>53.02</td><td>49.11</td><td>50.31</td><td>26.12</td><td>68.44</td></tr><tr><td>Highest loss / uncertainty</td><td>64.05</td><td>31.88</td><td>52.41</td><td>49.45</td><td>49.97</td><td>26.71</td><td>67.93</td></tr><tr><td>RHO-Loss style [8]</td><td>64.19</td><td>32.24</td><td>52.96</td><td>49.80</td><td>50.28</td><td>27.06</td><td>68.31</td></tr><tr><td>Grad. contrib. [11]</td><td>64.73</td><td>31.85</td><td>53.42</td><td>50.00</td><td>50.55</td><td>26.84</td><td>68.58</td></tr><tr><td>DisCL-style selection [25]</td><td>64.48</td><td>32.36</td><td>53.27</td><td>50.04</td><td>50.79</td><td>27.02</td><td>68.40</td></tr><tr><td>RDG (uncontrolled)</td><td>65.11</td><td>32.29</td><td>53.76</td><td>50.39</td><td>50.47</td><td>27.45</td><td>67.41</td></tr><tr><td>Fixed-Mid</td><td>65.02</td><td>32.91</td><td>53.58</td><td>50.50</td><td>50.68</td><td>27.33</td><td>68.87</td></tr><tr><td>CuRB (cRDG)</td><td>66.73</td><td>34.26</td><td>54.93</td><td>51.97</td><td>51.38</td><td>28.75</td><td>70.13</td></tr></table>

Tab<sup>l</sup>e 3. Contro<sup>ll</sup>ed curation comparison on Cityscapes<sup>→</sup>ACDC / Dar<sup>k</sup> Zuric<sup>h</sup>. All methods share the candidate pool, validity gate, $B _ { \mathrm { s e l . } }$ , final-training objective, and schedule, difering in their sample-selection policies. Fog, night, and rain are treated as the in-family conditions. Snow is held out as an unseen family, with no snow samples entering the candidate pool, curation, or final training.

Input  
Baseline  
+ CuRB  
![](images/78138647a8a9f756339fc4edf307993d0989cda75ab60dbeb6c0095374dc8107.jpg)  
(a) SOD

GT  
![](images/a41f896f778eb0b30f1d47b085f93e95275c3afeec397e979dcb180754d82e21.jpg)

Input  
![](images/2159c83bd80efaf953128cf97c7c0350c39045fb11ceac0fcf8ca8c9967555f0.jpg)

Baseline  
![](images/5b0680f12a50270a573f413d4447eb1dea401d26da5a568dd4dac9c03ae5f457.jpg)

+ CuRB  
![](images/5d4466a2dbf8ce124d40f7557a0d094822b004cc98851ad2abe85998dd894382.jpg)

![](images/01ef6ef93bd0d7c4ee0558247cb03dfdccce061189393799ba5122243062850f.jpg)  
GT  
(b) SemSeg  
Figure 3. Qua<sup>l</sup>itative comparison. (a) Degraded SOD with NUN. (b) adverse-condition SemSeg with ISSA SegFormer. Baseline and +CuRB predictions are shown for both tasks.

## 4.2 Main Results

Benc<sup>h</sup>mar<sup>k</sup>-<sup>l</sup>eve<sup>l</sup> per<sup>f</sup>ormance. Tab. 1 attaches CuRB to three published base models on SOD benchmarks, including the SAM-based CSSAM, under corresponding protocols. The uncurated row adds the same budget of synthetic twins without selection and brings only small gains, with almost none on the strongest base model. Selecting the same budget with cRDG improves all four metrics on every base model and both benchmarks. The margin over uncurated twins remains positive throughout, including for CSSAM, where uncurated data provide little benefit.

Across data-generation <sup>f</sup>ramewor<sup>k</sup>s. Tab. 2 evaluates CuRB across several published datageneration settings at their original synthetic-data budgets. The “+CuRB” variants obtain <sup>+</sup>1<sub>.</sub>2 to <sup>+</sup>1<sub>.</sub>5 higher ACDC mIoU and <sup>+</sup>1<sub>.</sub>2 to <sup>+</sup>1<sub>.</sub>7 higher Dark Zurich mIoU than the corresponding originals, while maintaining or slightly improving Cityscapes performance. The gains span style synthesis (ISSA), procedural weather corruption (H-Weather), and difusion-based editing (Gen4Seg), and persist for the composite RobustNet+ISSA host. The same insertion also improves SOD baselines in Tab. 1. Qualitative examples for both tasks and hosts are shown in Fig. 3.

Contro<sup>ll</sup>ed data-se<sup>l</sup>ection comparison. Tab. 3 compares selection criteria under one shared recipe, so the diferences reflect what is selected. CuRB is best on every column. It reaches 51<sub>.</sub>97 in-family mIoU versus 50<sub>.</sub>50 for the strongest control and 50<sub>.</sub>04 for the strongest literature baseline, 28<sub>.</sub>75 on Dark Zurich, and 51<sub>.</sub>38 on held-out snow, where improvement reflects cross-family transfer. Cityscapes performance is also preserved, reaching 70<sub>.</sub>13 versus 67<sub>.</sub>84 for source-only. The uncontrolled twin score is the only selector below the source-only model. $\ S$ 4.4 examines the ranking quality behind these diferences. Published augmentation and adaptation results are not mixed into this table because their training exposure difers from our held-out-family protocol. Some use synthetic snow or target images. Tab. 2 instead compares CuRB within each host’s native setting.

<table><tr><td>Variant</td><td>in-fam. mIoU↑</td><td>City val↑</td><td> $\rho _ { \mathrm { g l o b } } \ \mathbf { V } \mathbf { S } .$   $\breve { U } _ { \mathrm { r e a l } } \uparrow$ </td></tr><tr><td>CuRB (full)</td><td>51.97</td><td>70.13</td><td>0.79</td></tr><tr><td>paired gap reduction (RDG)</td><td>50.39</td><td>67.41</td><td>0.61</td></tr><tr><td>held-out loss drop  $D _ { g }$ </td><td>51.02</td><td>68.72</td><td>0.70</td></tr><tr><td>in-sample (no cross-fitting)</td><td>49.21</td><td>69.58</td><td>0.44</td></tr><tr><td>scored on training twins (no held-out)</td><td>50.51</td><td>69.85</td><td>0.58</td></tr><tr><td>w/o clean-harm constraint (H free)</td><td>50.68</td><td>67.18</td><td>0.79</td></tr></table>

Tab<sup>l</sup>e 4. Core ab<sup>l</sup>ations (Cityscapes<sup>→</sup>ACDC). RDG and $D _ { g }$ replace cRDG by the paired gap reduction and the held-out loss drop. Other rows remove cross-fitting, held-out transfer, and the clean-harm constraint. $\rho _ { \mathrm { g l o b } }$ is over all 32 regions.

<table><tr><td>Methods</td><td>M↓</td><td> $F _ { \beta } ^ { m a x }$  ↑</td><td> $E _ { \phi }$  ←  $S _ { \alpha } \uparrow$ </td></tr><tr><td>Source-only</td><td>.021</td><td>.892</td><td>.960 .920</td></tr><tr><td>All-gen. (uniform,  $B _ { \mathrm { s e l } } )$ </td><td>.022</td><td>.889</td><td>.958 .918</td></tr><tr><td>Grad. contrib. [11]</td><td>.022</td><td>.890</td><td>.959 .919</td></tr><tr><td>RDG (uncontr.)</td><td>.023</td><td>.885</td><td>.955 .914</td></tr><tr><td>Fixed-Mid</td><td>.021</td><td>.891</td><td>.960 .919</td></tr><tr><td>cRDG, H unconstrained CuRB</td><td>.025 .021</td><td>.879 .893</td><td>.950 .908 .961 .921</td></tr></table>

Tab<sup>l</sup>e 5. C<sup>l</sup>ean retention on SOD (DUTS-TE, NUN/PVTv2-B2). Source-only trains on clean DUTS-TR. Augmented variants add the same synthetic budget selected by each method, so any drop is the clean cost of that selection.

Matc<sup>h</sup>ed compute. With $K = 1 / 8 0$ of the final schedule, the 66 probes cost 0<sub>.</sub>59<sup>×</sup> one finaltraining run, or 1<sub>.</sub>59<sup>×</sup> with final training. Candidate generation, held-out scoring, and selection are excluded. At matched probe-plus-training compute, All-gen. and Fixed-Mid reach 49<sub>.</sub>86 and 51<sub>.</sub>02 in-family mIoU, versus 51<sub>.</sub>97 for CuRB. Full-pool training at 3<sub>.</sub>1<sup>×</sup> reaches only 50<sub>.</sub>02 (Appendix F).

## 4.3 Ablation Study

The ablations isolate the main components of cRDG (Tab. 4). Replacing cRDG with paired gap reduction lowers $\rho _ { \mathrm { g l o b } }$ from 0<sub>.</sub>79 to 0<sub>.</sub>61, while removing only the clean-control subtraction $( D _ { g } )$ lowers it to 0<sub>.</sub>70, showing the value of both the matched control and the controlled diference. In-sample estimation and training-twin scoring further weaken the ranking, supporting held-out cross-fitted evaluation. Removing the � constraint leaves $\rho _ { \mathrm { g l o b } }$ unchanged at 0<sub>.</sub>79 but drops Cityscapes accuracy from 70<sub>.</sub>13 to 67<sub>.</sub>18, indicating that reducibility and clean feasibility capture diferent aspects of the selection problem. Tab. 5 shows the same efect on SOD: CuRB preserves clean DUTS-TE performance, whereas the �-free variant falls on every metric, including $S _ { \alpha }$ ( 921 <sup>→</sup> 908).

## 4.4 Analysis: Training Utility and the Correctable Band

Does cRDG predict rea<sup>l</sup> uti<sup>l</sup>ity? We first obtain an empirical retraining utility. Holding everything but the region fixed, we retrain once per region � and record the signed, unnormalized gain on ACDC in-family, $U _ { \mathrm { r e a l } } ( g ) = \mathrm { m I o U } _ { \mathrm { r e a l } } ( M _ { g } ) - \mathrm { m I o U } _ { \mathrm { r e a l } } ( M _ { \mathrm { c l e a n } } )$ . Fig. 4a overlays it with $\Delta _ { 0 } ,$ RDG, and cRDG for one family. We test whether RDG remains high when real utility becomes negative and whether cRDG preserves the ordering of $U _ { \mathrm { r e a l } } ,$ , including regions with negative utility. The decision unit and the correlation unit are both the region (per-sample real gain would need one retraining per sample). A global rank correlation can be inflated by mean diferences between families, so Tab. 6 reports $\rho _ { \mathrm { g l o b } }$ over all 32 regions and $\begin{array} { r } { \rho _ { \mathrm { m a c r o } } = \frac { 1 } { | \mathcal { D } | } \sum _ { d } \rho _ { d } , } \end{array}$ , where $\rho _ { d }$ is the Spearman correlation over the 8 regions of family $d .$ Both use the operational ranking. Because cRDG is computed on source twins only, the table also correlates the same ranking with $U _ { \mathrm { D Z } } ,$ the per-region retraining gain on Dark Zurich. Correlation with a second real target supports transfer beyond the benchmark used to define $U _ { \mathrm { r e a l } }$ . The held-out loss drop $D _ { g }$ is the no-subtraction counterpart, and cRDG improves $\rho _ { \mathrm { g l o b } }$ from 0 70 to 0 79 and in-family mIoU from 51<sub>.</sub>02 to 51<sub>.</sub>97 over it, and $\rho _ { \mathrm { D Z } }$ from 0<sub>.</sub>62 to 0<sub>.</sub>72.

![](images/aa54ee746456f617b97eeff96fac52ea242f9cdb50d87cac7b1fc3301e677cf2.jpg)

(b) Empirical vs. predicted centroid  
![](images/c6ed2681dd7ea36322b85a934d4fdcd3695720252f7fe01fed82855fffbce04c.jpg)

(c) $\mathrm { c R D G } _ { \mathrm { s e n d } }$ across operating conditions  
![](images/2c7781f7ce398114e844452b63c05ca221bf5d9b3e068dea4769e66ea174bba3.jpg)

![](images/a5a04515fc573cd643365e85d3df084e5291bbd32194d65ab709d11ae84ec9ed.jpg)

![](images/9988e47fa794906ed7eda4253aa784b4f25c0d26670b2a61528ee9a0b1658d67.jpg)

Figure 4. Training uti<sup>l</sup>ity and t<sup>h</sup>e correctab<sup>l</sup>e band. (a) Per-bin retraining gains (top) and scores (bottom) for low light. Shading marks the predicted band. (b) Empirical severity centroids at three final budgets with scaled-� cRDG predictions overlaid (paired-bootstrap 95% CI). (c) cRDG centroids across probe budgets, checkpoints, and predictors. Error bars are s.d. over three partitions.
<table><tr><td rowspan="2">Metric</td><td colspan="3">Adapted valuation baselines</td><td colspan="4">Twin-based scores</td></tr><tr><td>uncertainty</td><td>Highest loss / RHO-Loss style Grad. contrib. [8]</td><td>[11]</td><td> $\Delta _ { 0 }$ </td><td>RDG</td><td> $D _ { g }$  (gap) (uncontrolled) (held-out drop)</td><td>CuRB (cRDG)</td></tr><tr><td> $\rho _ { \mathrm { g l o b } }$  with  $U _ { \mathrm { r e a l } } \uparrow$ </td><td>0.25</td><td>0.49</td><td>0.52</td><td>0.39</td><td>0.61</td><td>0.70</td><td>0.79</td></tr><tr><td>ρmacro with  $U _ { \mathrm { r e a l } } \uparrow$ </td><td>0.17</td><td>0.35</td><td>0.37</td><td>0.27</td><td>0.43</td><td>0.56</td><td>0.74</td></tr><tr><td>ρDz with  $U _ { \mathrm { D Z } } \uparrow$ </td><td>0.21</td><td>0.43</td><td>0.45</td><td>0.33</td><td>0.53</td><td>0.62</td><td>0.72</td></tr><tr><td>in-fam. mIoU ↑</td><td>49.45</td><td>49.80</td><td>50.00</td><td>49.28</td><td>50.39</td><td>51.02</td><td>51.97</td></tr></table>

Tab<sup>l</sup>e 6. Region-<sup>l</sup>eve<sup>l</sup> score comparison under matched pool, gate, diversity, budget, objective, and schedule (Cityscapes<sup>→</sup>ACDC, ISSA SegFormer). $\rho _ { \mathrm { g l o b } }$ is Spearman correlation with $U _ { \mathrm { r e a l } }$ over all 32 regions, �<sub>macro</sub> averages within-family correlations, and $\rho _ { \mathrm { D Z } }$ evaluates the same ranking against per-region retraining gains on Dark Zurich. $\Delta _ { 0 }$ is the unmastered gap and $D _ { g }$ the held-out loss drop.

How does t<sup>h</sup>e band move? Fig. 4c examines variation with probe budget �, source checkpoint $\theta _ { 0 } ,$ , and predictor setting. Doubling � shifts the centroid by about 0<sub>.</sub>10, well beyond the 0<sub>.</sub>03–0<sub>.</sub>04 variation across partitions. The centroid also changes across families and predictor settings, and mid-training checkpoints are about 0<sub>.</sub>05 higher than converged ones. Panel (b) asks whether the useful severity region changes with the final-training budget rather than remaining fixed. At each budget, the empirical severity centroid is computed over the probe-feasible regions, replacing cRDG by positive $U _ { \mathrm { r e a l . } }$ , using per-bin retraining at 0<sub>.</sub>5<sup>×</sup>, 1<sup>×</sup>, and $2 \times$ the final budget. Within those regions, its centroid shifts by <sup>+</sup>0<sub>.</sub>20 for fog and <sup>+</sup>0<sub>.</sub>19 for low light from 0<sub>.</sub>5<sup>×</sup> to $2 \times ,$ while the scaled-� cRDG prediction remains within 0<sub>.</sub>03 of the empirical centroid at every budget.

## 5 Discussion and Limitations

The twin-based comparisons isolate two diferent confounds. Paired gap reduction is afected by clean forgetting, whereas $D _ { g }$ still includes progress obtainable from clean training. Subtracting the matched clean-control trajectory gives cRDG, which best tracks the signed retraining utility in our experiments. The correctable band depends on the learner and the budget rather than on severity alone, and it shifts with the final budget as the scaled-� score predicts. The procedure transfers across the tested predictors unchanged, whereas the scores are recomputed for each model.

Limitations. cRDG is an empirical surrogate for the full-training gain relative to clean-control training, tested at three budgets on one backbone. Residual drift near the validity threshold may introduce mild label noise despite the structural gate. Unseen families such as snow benefit only through cross-family transfer. The probe horizon follows a fixed budget-scaling rule rather than adapting to each region. The SOD twin pool comes from external clean images. Also, all newly trained models are single runs, and we do not report seed variance.

## 6 Conclusion

We study how to curate synthetic degradations under a finite training budget. cRDG compares degradation training with an equally long clean-control probe on held-out twins and treats clean harm separately. Using this score, CuRB selects a fixed-budget training set and improves SOD and semantic segmentation across the tested predictors and generation settings. We also find that the useful severity range changes with model state and training budget instead of remaining fixed.

## References

[1] Chunming He, Kai Li, Yachao Zhang, Yulun Zhang, Zhenhua Guo, and Xiu Li. Strategic preys make acute predators: Enhancing camouflaged object detectors by generating camouflaged objects. In ICLR, 2024. 1

[2] Chunming He, Rihan Zhang, Fengyang Xiao, Dingming Zhang, Zhiwen Cao, and Sina Farsiu. Refining context-entangled content segmentation via curriculum selection and anti-curriculum promotion. In ICML, 2026. 1

[3] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent difusion models. In CVPR, pages 10674–10685, 2022. 1

[4] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image difusion models. In ICCV, pages 3813–3824, 2023.

[5] Sudarshan Rajagopalan, Nithin Gopalakrishnan Nair, Jay N Paranjape, and Vishal M Patel. Gendeg: Difusion-based degradation synthesis for generalizable all-in-one image restoration. In CVPR, pages 28144–28154, 2025. 1, 3

[6] Krzysztof Adamkiewicz, Brian B Moser, Stanislav Frolov, Tobias Christian Nauen, Federico Raue, and Andreas Dengel. When pretty isn’t useful: Investigating why modern text-to-image models fail as reliable training data generators. In CVPR, pages 36660–36669, 2026. 1, 3

[7] Jinjin Zhang, Xiefan Guo, Yizhou Jin, Nan Zhou, and Di Huang. What makes synthetic data efective in image segmentation. In ICML, 2026. 1, 3

[8] Sören Mindermann, Jan M Brauner, Muhammed T Razzak, Mrinank Sharma, Andreas Kirsch, Winnie Xu, Benedikt Höltgen, Aidan N Gomez, Adrien Morisot, Sebastian Farquhar, et al. Prioritized training on points that are learnable, worth learning, and not yet learnt. In ICML, pages 15630–15649, 2022. 2, 3, 7, 8, 10

[9] Reyhane Askari-Hemmat, Mohammad Pezeshki, Elvis Dohmatob, Florian Bordes, Pietro Astolfi, Melissa Hall, Jakob Verbeek, Michal Drozdzal, and Adriana Romero-Soriano. Improving the scaling laws of synthetic data with deliberate practice. In ICML, 2025. 2, 3

[10] Garima Pruthi, Frederick Liu, Satyen Kale, and Mukund Sundararajan. Estimating training data influence by tracing gradient descent. In NeurIPS, pages 19920–19930, 2020. 2, 3

[11] Muzhi Zhu, Chengxiang Fan, Hao Chen, Yang Liu, Weian Mao, Xiaogang Xu, and Chunhua Shen. Generative active learning for long-tailed instance segmentation. In ICML, 2024. 2, 3, 7, 8, 9, 10

[12] Mansheej Paul, Surya Ganguli, and Gintare Karolina Dziugaite. Deep learning on a data diet: Finding important examples early in training. In NeurIPS, pages 20596–20607, 2021. 2, 3

[13] Mariya Toneva, Alessandro Sordoni, Remi Tachet des Combes, Adam Trischler, Yoshua Bengio, and Geofrey J Gordon. An empirical study of example forgetting during deep neural network learning. In ICLR, 2019. 2, 3

[14] Jefrey A Chan-Santiago and Mubarak Shah. Learnability-guided difusion for dataset distillation. In CVPR, pages 41657–41666, 2026. 2, 3

[15] Thang-Anh-Quan Nguyen, Moussab Bennehar, Luis Guillermo Roldao Jimenez, Nathan Piasco, Dzmitry Tsishkou, Laurent Carafa, Jean-Philippe Tarel, and Roland Brémond. Cyclone: Difusion model for cycle-consistent weather editing from unpaired driving data. arXiv preprint arXiv:2607.13927, 2026. 3

[16] Kai Guan, Rongyuan Wu, Shuai Li, Wentao Zhu, Wenjun Zeng, and Lei Zhang. Restoration adaptation for semantic segmentation on low quality images. Int. J. Comput. Vis., 134(5):229, 2026. 3

[17] Yuru Jia, Lukas Hoyer, Shengyu Huang, Tianfu Wang, Luc Van Gool, Konrad Schindler, and Anton Obukhov. Dginstyle: Domain-generalizable semantic segmentation with image difusion models and stylized semantic control. In ECCV, pages 91–109, 2024. 3

[18] Minho Park, Sunghyun Park, Jungsoo Lee, Hyojin Park, Kyuwoong Hwang, Fatih Porikli, Jaegul Choo, and Sungha Choi. Concept-aware lora for domain-aligned segmentation dataset generation. In CVPR, pages 39858–39868, 2026. 3

[19] Yumeng Li, Dan Zhang, Margret Keuper, and Anna Khoreva. Intra-& extra-source exemplar-based style synthesis for improved domain generalization. Int. J. Comput. Vis., 132(2):446–465, 2024. 3, 6, 7, 14, 16

[20] Dan Hendrycks and Thomas Dietterich. Benchmarking neural network robustness to common corruptions and perturbations. arXiv preprint arXiv:1903.12261, 2019. 3, 7, 16

[21] Zĳin Yin, Bing Li, Kongming Liang, Hao Sun, Zhongjiang He, Zhanyu Ma, and Jun Guo. Benchmarking semantic segmentation models via appearance and geometry attribute editing. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2026. 3, 6, 7, 16

[22] Lukas Hoyer, Dengxin Dai, and Luc Van Gool. Daformer: Improving network architectures and training strategies for domain-adaptive semantic segmentation. In CVPR, pages 9914–9925, 2022. 3

[23] Wilhelm Tranheden, Viktor Olsson, Juliano Pinto, and Lennart Svensson. Dacs: Domain adaptation via cross-domain mixed sampling. In WACV, pages 1378–1388, 2021.

[24] Christos Sakaridis, Dengxin Dai, and Luc Van Gool. Guided curriculum model adaptation and uncertainty-aware evaluation for semantic nighttime image segmentation. In ICCV, pages 7374–7383, 2019. 3, 7

[25] Yĳun Liang, Shweta Bhardwaj, and Tianyi Zhou. Difusion curriculum: Synthetic-to-real data curriculum via image-guided difusion. In ICCV, pages 1697–1707, 2025. 3, 7, 8

[26] Quan Chen, Xiong Yang, Bolun Zheng, Rongfeng Lu, Xiaokai Yang, Qianyu Zhang, Yu Liu, and Xiaofei Zhou. Wxsod: A benchmark for robust salient object detection in adverse weather conditions. arXiv preprint arXiv:2508.12250, 2025. 6, 7, 16

[27] Runmin Cong, Zhiyang Chen, Hao Fang, Sam Kwong, and Wei Zhang. Breaking barriers, localizing saliency: A large-scale benchmark and baseline for condition-constrained salient object detection. IEEE Trans. Pattern Anal. Mach. Intell., 2026. 6, 7, 16

[28] Chunming He, Rihan Zhang, Dingming Zhang, Fengyang Xiao, Deng-Ping Fan, and Sina Farsiu. Nested unfolding network for real-world concealed object segmentation. In ECCV, 2026. 7

[29] Feiran Li, Qianqian Xu, Shilong Bao, Zhiyong Yang, Runmin Cong, Xiaochun Cao, and Qingming Huang. Size-invariance matters: Rethinking metrics and losses for imbalanced multi-object salient object detection. In ICML, 2024. 7

[30] Lĳun Wang, Huchuan Lu, Yifan Wang, Mengyang Feng, Dong Wang, Baocai Yin, and Xiang Ruan. Learning to detect salient objects with image-level supervision. In CVPR, pages 3796–3805, 2017. 6

[31] Marius Cordts, Mohamed Omran, Sebastian Ramos, Timo Rehfeld, Markus Enzweiler, Rodrigo Benenson, Uwe Franke, Stefan Roth, and Bernt Schiele. The cityscapes dataset for semantic urban scene understanding. In CVPR, pages 3213–3223, 2016. 6

[32] Christos Sakaridis, Dengxin Dai, and Luc Van Gool. Acdc: The adverse conditions dataset with correspondences for semantic driving scene understanding. In ICCV, pages 10745–10755, 2021. 7

[33] Sungha Choi, Sanghun Jung, Huiwon Yun, Joanne T Kim, Seungryong Kim, and Jaegul Choo.

Robustnet: Improving domain generalization in urban-scene segmentation via instance selective whitening. In CVPR, pages 11575–11585, 2021. 7, 14

[34] Liang-Chieh Chen, Yukun Zhu, George Papandreou, Florian Schrof, and Hartwig Adam. Encoder-decoder with atrous separable convolution for semantic image segmentation. In ECCV, pages 833–851, 2018. 7

## A Implementation Details

Bac<sup>k</sup>bones and optimization. SemSeg follows the released ISSA configuration [19] for SegFormer. The DeepLabV3+ experiments follow the oficial RobustNet recipe [33]. SOD clean-retention runs use NUN on a PVTv2-B2 backbone with its released training configuration. CuRB leaves the host optimizer, input resolution, augmentation, and final-training schedule unchanged in every experiment. All runs use a single NVIDIA RTX PRO 6000 GPU with 96 GB of memory.

Curation budget and probe settings. For the Cityscapes experiments, $B _ { \mathrm { s e l } } = 5 , 9 5 0$ synthetic samples are filled in rank order over 32 regions (four families, eight severity bins), with a 1:1 clean/degraded slot ratio. Each region starts from $2 , 9 7 5 \times 2$ candidates before validity filtering. Probes optimize the first two terms of Eq. (5) at the same slot ratio and 0<sub>.</sub>25<sup>×</sup> the base learning rate, for � equal to $1 / 8 0$ of the corresponding final-training schedule. Plug-in experiments keep each host’s synthetic budget, and cRDG is recomputed per host. Under this budget, rank-greedy filling is concentrated. The top-ranked region contributes over 90% of $S ^ { \star }$ and the remainder comes from the next-ranked region. For WXSOD and CSOD10K the synthetic budget is filled entirely from the top-ranked feasible region, so <sup>�</sup>-center controls diversity within that region.

Th<sub>res</sub>h<sub>o</sub>ld<sub>s.</sub> $\lambda _ { s } = 1 . 0 , \lambda _ { c } = 0 . 5 , \lambda _ { b } = 1 . 0$ in $\operatorname { E q . }$ (5). The clean-retention tolerance $\epsilon$ is set to one standard error of the clean-harm estimate from degradation-free matched probes on a held-out source split, where the standard error is taken over the per-image loss diferences on the held-out images. The rule is fixed across experiments, and the value is estimated once for each task and predictor on its own source data. We use $\scriptstyle \kappa = 1 / 2$ for the diagnostic band and $n _ { z } { = } 4$ regeneration draws in the validity fallback. Nothing is tuned on a target set.

Eva<sup>l</sup>uation protoco<sup>l</sup>. SemSeg follows the ISSA evaluation protocol on Cityscapes val, the four ACDC conditions, and Dark Zurich. SOD benchmarks use their oficial metrics. Source-only and $M _ { \mathrm { c l e a n } }$ are obtained from the same $\theta _ { 0 }$ with the same final-training schedule, with the synthetic slot replaced by clean data. Unless otherwise noted, newly trained models are reported from single runs, and published reference values are taken from the cited sources. Operational curation uses one fixed probe seed and one fixed $A | B$ partition. Two additional $A | B$ partitions serve only the partition-stability analysis and the error bars of Fig. 4c and never enter curation (Appendix C).

## B Generator and Validity Gate

Structure-preserving generation. We use Stable Difusion 2.1 in image-to-image mode with depth- and Canny-conditioned ControlNet guidance together with the source image, but not the ground-truth mask, which avoids mask-aligned synthesis artifacts. We instantiate low light, backlight, fog, and rain. Each family contains eight ordered severity bins obtained by uniformly sampling its family-specific degradation-strength parameter, with two stochastic draws per source image. The strength variable of each family and its eight values are provided in the released generation configuration.

Va<sup>l</sup>idity gate. The gate rejects candidates whose structure has drifted. Given $( x , { \tilde { x } } , y )$ it measures frozen DINOv2 ViT-B/14 feature correspondence and boundary displacement in pixels within a band around ground-truth boundaries. For SOD it additionally checks that the salient object is neither deleted nor invented. It is a local structural test rather than a global appearance distance, so a uniformly darker or foggier image with intact geometry passes. Thresholds are set to the 99th percentile of the corresponding statistics on clean source pairs under geometric re-augmentation, after compensating for the known geometric transform, and reused across datasets. In a manual audit of 500 candidates per family (Tab. S1), acceptance is 98–100% at mild levels and remains 90–96% at severe levels, so the gate does not simply remove the stronger degradations from the pool.

```latex
A<sup>l</sup>gorit<sup>h</sup>m 1 CuRB: controlled curation by reducibility
Require: clean source set $\mathcal { D } _ { s } ,$ , generator ${ \mathcal { G } } ,$ checkpoint $\theta _ { 0 } ,$ probe budget $\begin{array} { r } { K = \frac { 1 } { 8 0 } } \end{array}$ of the final
schedule, tolerance $\epsilon ,$ sample budget $\boldsymbol { B } _ { \mathrm { s e l } }$
1: split source IDs once into $A | B$ and generate gated degradation regions $g = ( d , s )$
2: $\hat { \theta _ { A } ^ { \mathrm { c t r l } } } \gets U ( \theta _ { 0 } , A _ { \mathrm { c l e a n } } \oplus A _ { \mathrm { a u g } } ; K )$ and $\theta _ { B } ^ { \mathrm { c t r l } } \gets \tilde { U } ( \theta _ { 0 } , B _ { \mathrm { c l e a n } } \oplus B _ { \mathrm { a u g } } ; K )$ ⊲ shared controls
3: <sup>f</sup>or a<sup>ll</sup> regions $g = ( d , s )$ d<sub>o</sub>
4: $\theta _ { A , g } ^ { \mathrm { d e g } }  U ( \theta _ { 0 } , A _ { \mathrm { c l e a n } }$ ⊕ $A _ { \mathrm { d e g } , g } ; K )$ and likewise $\theta _ { B , g } ^ { \mathrm { d e g } }$
5: compute $G _ { g } ^ { A  B } , H _ { g } ^ { A  B }$ on � and $G _ { g } ^ { B \to A } , H _ { g } ^ { B \to A }$ on �
6: $\mathrm { c R D G } ( g ; K , \breve { \theta } _ { 0 } ) \longleftarrow \frac { \ l } { 2 } ( G _ { g } ^ { A \to B } + G _ { g } ^ { B \to \breve { A } } )$
7: feasible $\cdot ( g ) \gets [ \mathsf { m a x } ( H _ { g } ^ { \breve { A }  B } , H _ { g } ^ { B  A } ) \leq \epsilon ]$
8: en<sup>d f</sup>or
9: rank feasible regions with positive $\mathrm { c R D G } ( g ; K , \theta _ { 0 } )$ and fill $\boldsymbol { B } _ { \mathrm { s e l } }$ in descending order, exhausting
each region before moving to the next
10: use <sup>�</sup>-center when only part of a region is needed
11: use clean data for any unfilled budget and obtain $S ^ { \star }$
12: train from $\theta _ { 0 }$ on clean data $\cup S ^ { \star }$ with Eq. (5)
```

<table><tr><td>Degradation</td><td>gate prec.</td><td>gate recall</td><td>accept.  $s _ { 1 } { - } s _ { 4 }$ </td><td>accept. S5-S8</td></tr><tr><td>Low light</td><td>0.94</td><td>0.91</td><td>0.99</td><td>0.95</td></tr><tr><td>Backlight</td><td>0.92</td><td>0.89</td><td>0.99</td><td>0.92</td></tr><tr><td>Fog</td><td>0.95</td><td>0.92</td><td>1.00</td><td>0.96</td></tr><tr><td>Rain</td><td>0.91</td><td>0.88</td><td>0.98</td><td>0.90</td></tr></table>

Tab. S1. Va<sup>l</sup>idity-gate audit on 500 sampled candidates per family. Structural drift is the positive class for gate precision and recall. Acceptance remains high at severe levels (0<sub>.</sub>90–0<sub>.</sub>96), indicating that the gate does not simply remove stronger degradations.

## C Probe Protocol and Estimator Details

Probe matc<sup>h</sup>ing. Operational curation uses one fixed random split of source image IDs into equal halves � and $B ,$ shared across all regions. For each region � and split direction, let $S \in \{ A , B \}$ denote the probe-update half. The treatment branch trains on $S _ { \mathrm { c l e a n } } \oplus S _ { \mathrm { d e g } , g }$ and the control branch on $S _ { \mathrm { c l e a n } } \oplus S _ { \mathrm { a u g } } ,$ both from $\theta _ { 0 } .$ . The two branches share �, batch composition, clean-replay ratio, optimizer and schedule, geometric views, and data order. The region-independent control is run once per direction and shared across regions. One fixed probe seed is used operationally. Two additional $A | B$ partitions serve only to assess stability and provide the error bars of Fig. 4c, and probe horizons of 0 5<sup>×</sup> and $2 \times$ the default serve only the sensitivity analysis. Algorithm 1 gives the full procedure. The training-twin variant scores a region by the loss drop on the treatment probe’s own degraded training twins rather than by held-out transfer. The in-sample variant keeps the controlled gain $G _ { g }$ but evaluates both branches on the probe-update half instead of the held-out half.

Matc<sup>h</sup>ed views and va<sup>l</sup>idity <sup>f</sup>a<sup>ll</sup>bac<sup>k</sup>. Treatment and control views share the crop and flip of each source image. The control applies weak photometric augmentation (colour jitter, gamma in [0<sub>.</sub>8 1<sub>.</sub>25], and Gaussian blur with $\sigma \leq 0 . 6 )$ , while the treatment uses a generated twin from region �. Twins that fail the gate are resampled up to $n _ { z } { = } 4$ times. If no valid training twin is found, the control view is retained in that slot, and invalid evaluation twins are dropped from both branches.

Statistica<sup>l</sup> reporting. cRDG is reported signed rather than rectified, preserving negative transfer. Aggregate statistics use paired bootstrap over regions. The bootstrap of Fig. 4b resamples ACDC evaluation images with replacement within each condition, using matched indices for every retrained model and the clean model. Checkpoints are kept fixed, and $U _ { \mathrm { r e a l } }$ , band membership, and the centroid are recomputed per replicate. Error bars in Fig. 4c are standard deviations over the three partitions.

Sensitivity to �. Tab. S2 varies the clean-retention tolerance around its default of one standard error. Degraded-domain performance is stable over 0 5–2 standard errors, while a looser tolerance admits more regions at the cost of a small drop in clean accuracy on Cityscapes val.

<table><tr><td>€</td><td>0</td><td>0.5SE</td><td>1 SE (default)</td><td>2SE</td></tr><tr><td>feasible-region fraction</td><td>0.41</td><td>0.63</td><td>0.78</td><td>0.91</td></tr><tr><td>in-family mIoU ↑</td><td>51.43</td><td>51.86</td><td>51.97</td><td>51.79</td></tr><tr><td>Cityscapes mIoU ↑</td><td>70.21</td><td>70.24</td><td>70.13</td><td>69.82</td></tr></table>

Tab. S2. Sensitivity to t<sup>h</sup>e c<sup>l</sup>ean-retention to<sup>l</sup>erance �. Degraded-domain performance is stable around the default, while loosening the constraint permits a small clean-performance drop.

## D Plug-and-Play Framework Comparisons

Protoco<sup>l</sup>. Tab. 2 follows each host’s published pipeline and synthetic-data budget. Published references for ISSA, H-Weather, and RobustNet+ISSA are taken from Li et al. [19]. Gen4Seg follows Yin et al. [21], with its Dark Zurich value obtained using the released pipeline. For our “+CuRB” runs, CuRB is applied under the corresponding host setting and synthetic-data budget, with cRDG recomputed for each host.

Region construction. Regions are defined by family and severity bin, with eight bins per family unless the host provides fewer discrete levels. ISSA [19] uses style-exemplar groups and style-mixing strength. Hendrycks-Weather [20] uses corruption type and its predefined levels 1–5. Gen4Seg [21] uses weather type and a family-specific image-space statistic relative to the source (contrast attenuation for fog, luminance ratio for night, high-frequency energy gain for rain, bright-region coverage for snow), with prompt strength not used for region assignment. RobustNet + ISSA uses the ISSA regions with cRDG recomputed on the RobustNet model.

## E SOD Benchmark Protocols

WXSOD [26] and CSOD10K [27] use their oficial splits and metrics, with each base model following its published training recipe. The augmented variants add twins synthesized from clean DUTS-TR with the four families and eight severity levels of Appendix B, including its validity gate. The uncurated and CuRB variants use the same synthetic budget and the base model’s final-training schedule. The former samples uniformly, and the latter selects with cRDG. We use $B _ { \mathrm { s e l } } = 4 { , } 0 0 0$ on WXSOD and 3 750 on CSOD10K.

## F Compute and Cost Accounting

For the reported SegFormer configuration, the 66 probe updates cost approximately 0<sub>.</sub>59<sup>×</sup> one final-training run under the $K = T / 8 0$ scaling rule. Candidate-pool construction, held-out scoring, and selection are not included, and final training is accounted separately in § 4.2.

## G Baseline Adaptation

All selection baselines rank the same 32 gated regions and use the same $\boldsymbol { B } _ { \mathrm { s e l } }$ and within-region <sup>�</sup>- center step. Candidate-level scores are averaged within each region before ranking, and the final training recipe is identical for every method. RHO-Loss-style selection uses the reducible loss of a candidate relative to a small model trained on held-out source data. Gradient contribution uses the alignment of a candidate’s gradient with the mean source-validation gradient. The DisCL-style baseline is converted to a static region score. Its curriculum rule is run on the source-validation response of the current model to obtain a severity preference per family, which is then used to rank regions once before final training rather than to alter the training schedule.

<table><tr><td>Quantity Value</td></tr><tr><td>Regions M (four families × eight severity bins) 32</td></tr><tr><td>Split directions R 2</td></tr><tr><td>Probe horizon K 1/80 of the final schedule</td></tr><tr><td>Treatment probes (MR) 64</td></tr><tr><td>Shared control probes (R) 2</td></tr><tr><td>Probe-step cost / final-step cost (measured) ≈ 0.71</td></tr><tr><td>Probe-update cost / one final run 0.59×</td></tr><tr><td>Control overhead vs. uncontrolled probes (2/64) 3.1%</td></tr></table>

Tab. S3. Curation cost. Operational selection uses one �|� partition. The region-independent control adds only two probes, and the normalized probe-update cost is (66/80) <sup>×</sup> 0<sub>.</sub>71 <sup>≈</sup> 0<sub>.</sub>59 of one final-training run, under the $K = T / 8 0$ scaling rule.