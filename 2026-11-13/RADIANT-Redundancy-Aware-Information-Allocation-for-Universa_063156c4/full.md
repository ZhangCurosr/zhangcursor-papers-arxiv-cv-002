# RADIANT: Redundancy-Aware Information Allocation for Universal Multi-Modal Fusion

Shenghai Wu<sup>∗</sup>   
Research Institute   
Newland Digital Technology   
Fuzhou, China   
stevenwsh01@gmail.com   
Yue Lin   
Research Institute   
Newland Digital Technology   
Fuzhou, China   
yuelin109@gmail.com

## Abstract

Universal multi-modal fusion often fails not because modalities are insuficiently aligned, but because the fusion process allocates information poorly: redundant evidence is repeatedly amplified, modality-unique evidence is over-smoothed during alignment, and task-dependent synergistic evidence is insuficiently activated. We present RADIANT (Redundancy-Aware Dynamic Information Allocation Network), a task-general framework that organizes fusion around three complementary roles: suppressing redundant cross-modal overlap, preserving modality-unique evidence, and composing task-conditioned synergy through cross-modal interaction. Specifically, stable matrix-exponential projections are used to reduce interference among shared and modality-unique streams, while task memory with hard positive updates selects which synergistic interactions should be activated for the current fusion regime. Extensive experiments across six heterogeneous fusion tasks show consistent improvements over the compared methods. On FLIR, RADIANT improves downstream detection mAP@0.5 from 91.2% to 96.8%, a gain of 5.6 percentage points (6.1% relative) over GIF. RADIANT also exhibits strong direct cross-dataset transfer, with further gains from inference-time task-memory adaptation while keeping all network weights frozen. Additional ablations and sensitivity studies support the robustness of the allocation mechanism. Code: https://github.com/StevenWSH27/RADIANT.

## CCS Concepts

• Computing methodologies → Image processing; Computer vision; Learning latent representations; Multi-task learning.

## Keywords

Multi-modal image fusion, feature disentanglement, orthogonal feature decomposition, multi-granularity contrastive learning, crosstask generalization

## ACM Reference Format:

Shenghai Wu and Yue Lin. 2026. RADIANT: Redundancy-Aware Information Allocation for Universal Multi-Modal Fusion. In Proceedings ofthe 34th ACM International Conference on Multimedia (MM ’26), November 10–14, 2026, Rio

![](images/868d628b541f0d308ccf18e3e17dcc30595a2e145afba680c74a43f56ad9f762.jpg)

![](images/1991f6abe87040eec24f52583e823bb92c1e017ad3371e785dafdfeb61c47efa.jpg)  
Figure 1: Information allocation failure versus allocationaware fusion. Left: Conventional fusion over-emphasizes redundant cross-modal consistency and suppresses informative modality diferences, causing ambiguous representations. Right: RADIANT separates non-interfering shared evidence (F ), modality-unique evidence $( \mathbf { F } _ { u 1 } , \mathbf { F } _ { u 2 } )$ , and taskconditioned synergistic evidence (F ) for controllable universal fusion.

de Janeiro, Brazil. ACM, New York, NY, USA, 9 pages. https://doi.org/10. 1145/3767308.3836335

## 1 Introduction

Multi-modal image fusion is central to medical imaging, autonomous systems, surveillance, and remote sensing, where performance depends on combining complementary sensor evidence without destroying task-relevant details. A central challenge in universal multimodal fusion is not only feature alignment, but also the allocation of redundant, modality-unique, and interaction-dependent information across tasks. In practice, fusion quality can deteriorate when redundant evidence is repeatedly reinforced, modalityunique evidence is washed out during alignment, and synergistic evidence that emerges through cross-modal interaction is undermodeled [4, 5, 38].

Current methods still fall into two broad families: (1) task-specific architectures [28, 36, 37] that optimize for particular modality pairs but transfer poorly, and (2) generalized frameworks [3, 26] that broaden applicability yet often rely on a single shared fusion space. The former struggle to scale across application regimes, while the latter frequently over-index on cross-modal consistency. As a result, they often preserve what both modalities already know, but fail to decide what should be suppressed, what must be retained independently, and what should be created only after interaction.

This perspective suggests a diferent formulation of universal fusion. Rather than treating the problem as generic feature disentanglement, we cast it as task-conditioned information allocation: allocate redundant/shared evidence conservatively, preserve modality-unique evidence explicitly, and compose synergistic evidence through task-aware interaction. Under this view, orthogonality is useful only insofar as it provides a stable non-interference bias, while task memory is valuable because it selects the interaction patterns that are beneficial for the current fusion regime.

Based on this principle, we present RADIANT, a redundancyaware information allocation framework for universal multi-modal fusion. RADIANT combines a lightweight hybrid Vision Mamba– Swin backbone with a stable non-interfering shared/unique decomposition and task-memory-guided cross-modal interaction. Its contributions are:

(1) Task-conditioned information allocation for universal fusion. We introduce a fusion framework that explicitly distinguishes redundant shared evidence, modality-unique evidence, and task-conditioned synergistic evidence, allowing a single model to adapt its information allocation across six heterogeneous fusion tasks.

(2) Synergy-aware fusion with task memory. We reinterpret prototype learning as task memory that routes and stabilizes cross-modal interactions. Hard positive updates improve intra-task compactness, enabling the model to activate the synergistic interactions most useful for each fusion regime without task-specific architectural redesign.

(3) Lightweight universal performance with robust transfer. With a model footprint ofonly 3.108 MB and 9.42 GFLOPs, RADIANT achieves state-of-the-art results across six fusion tasks and shows robust direct cross-dataset transfer, with further improvements through inference-time task-memory adaptation while keeping network weights frozen.

Across six fusion tasks, RADIANT consistently outperforms the compared methods. On FLIR, it improves downstream detection mAP@0.5 from 91.2% to 96.8% over GIF, corresponding to a gain of 5.6 percentage points (6.1% relative). Systematic hyperparameter perturbations produce at most 1.2% median absolute percentage deviation within the evaluated ranges.

## 2 Related Work

## 2.1 Universal and Task-Specific Multi-Modal Fusion

Multi-modal image fusion has progressed from handcrafted priors [22] to deep architectures spanning infrared-visible, multifocus, medical, and remote-sensing settings. Task-specific methods such as CDDFuse [36], DDFM [37], Text-IF [28], CoCoNet [13], P2Sharpen [32], ZeroSharpen [24], and RTFusion [6] achieve strong in-domain performance by specializing for a narrow fusion regime, but transfer poorly across tasks. Universal methods such as U2Fusion [2 MUFusion [3], SDNet [31], UNIFusion [2], UniFuse [18], and MdaIF [11 broaden applicability, yet often depend on a single latent space or a fixed fusion rule. Recent task-general systems such as GRAM [4], CoMM [5], AdaSFFuse [25], and C2RF [19] move toward more principled multimodal representations, but they still provide limited control over how redundant, unique, and interaction-derived evidence are allocated during fusion, especially under changing task regimes.

## 2.2 Information Allocation in Multi-Modal Representation Learning

A large body of multimodal work separates shared and modalityspecific factors [21, 35]. Related domain-translation and fusion models further preserve domain- or modality-specific characteristics [9, 27]. DecAlign [16] further decouples common and unique representations through hierarchical alignment, while related orthogonalized or de-redundancy approaches improve non-interference between streams. These methods show the utility of structured factorization, but most stop at common/private decomposition. They do not directly address the 2025–2026 shift toward asking which information should be shared, which should be preserved independently, and which should be synthesized only through cross-modal interaction. Recent multimodal studies increasingly frame this question through redundancy, uniqueness and synergy. CoMM [5] explicitly uses information decomposition ideas to reason about complementary features, and newer multimodal systems explore orthogonality, de-redundancy, or expert routing as means to better distribute overlapping versus distinctive information. Our work follows this direction in the fusion setting: orthogonality is treated as an implementation choice for non-interference, while the main objective is allocation of redundant, unique, and task-conditioned synergistic evidence.

## 2.3 Task Memory and Cross-Modal Interaction

Contrastive learning has become widely used for cross-modal alignment and multimodal representation learning [7, 8, 12, 20, 29, 33]. Pixel-, region-, and prototype-based objectives provide diferent levels of cross-modal supervision. Task-aware representation learning is also closely related to multi-task learning, where shared representations can improve generalization across related tasks [1]. Recent systems such as Memory Aware Synapses [14], multi-query cross-modal attention fusion [34], and FMFR [23] further suggest that adaptive memory or query-conditioned interaction is useful when downstream regimes difer. Building on this line of work, RA-DIANT uses task prototypes as task memory to determine which cross-modal interactions should be emphasized, enabling adaptive synergy composition across heterogeneous fusion tasks.

## 3 Methodology

## 3.1 Problem Formulation and Information Allocation Principle

Let $\mathbf { I } _ { 1 } , \mathbf { I } _ { 2 } \in \mathbb { R } ^ { H \times W \times C }$ denote source images from two modalities. Encoder $\phi _ { \theta }$ maps them to a joint representation h $\in \mathbb { R } ^ { N \times D }$ , and decoder $\psi _ { \omega }$ predicts the fused output Y. For � fusion tasks, training data are organized as $\mathcal { D } _ { k } = \{ ( \mathbf { I } _ { 1 } ^ { ( n ) } , \mathbf { I } _ { 2 } ^ { ( n ) } ) \} _ { n = 1 } ^ { N _ { k } }$

Instead of describing universal fusion as generic disentanglement, RADIANT models it as information allocation. The representation is organized into three roles: redundant/shared evidence $\mathbf { F } _ { r }$ , modality-unique evidence $\mathbf { F } _ { u 1 } , \mathbf { F } _ { u 2 } ,$ and synergistic evidence $\mathbf { F } _ { s }$ that emerges only after task-conditioned cross-modal interaction. $\mathrm { A }$ desirable fusion model should suppress unnecessary redundancy, preserve modality-unique cues, and activate synergy only when the current task regime benefits from it. We express this principle

![](images/201c8635d720ff15cfa94e01459cadb7c930810a769dabcc99b9b0dfbd2dabfd.jpg)  
Figure 2: RADIANT architecture overview. A dual-branch Vision Mamba–Swin encoder first maps the two modalities to a joint representation. A shared/unique decomposition module then separates shared and modality-unique evidence $( \mathbf { F } _ { r } , \mathbf { F } _ { u 1 } , \mathbf { F } _ { u 2 } )$ , and task memory guides multi-granularity interaction to compose synergistic evidence (F<sub>�</sub>) before decoding.

as

$$
\operatorname* { m i n } _ { \phi _ { \theta } , \psi _ { \omega } } \ \mathcal { R } _ { \mathrm { r e d } } ( \mathbf { F } _ { r } ) + \lambda _ { u } \mathcal { R } _ { \mathrm { u n i q } } ( \mathbf { F } _ { u 1 } , \mathbf { F } _ { u 2 } ) - \lambda _ { s } \mathcal { G } _ { \mathrm { s y n } } ( \mathbf { F } _ { s } ; \mathbf { Y } )\tag{1}
$$

where $\mathcal { R } _ { \mathrm { r e d } }$ penalizes redundant amplification, ${ \mathcal { R } } _ { \mathrm { u n i q } }$ preserves the modality-specific evidence, and $\mathcal { G } _ { \mathrm { s y n } }$ rewards the task-useful syn ergy. In practice these quantities are instantiated through fusion, non-interference, and task-memory-guided interaction losses rather than by exact information decomposition.

The overall architecture in Figure 2 realizes this principle in two stages. First, a stable non-interfering parameterization separates shared and modality-unique streams. Second, task memory and multi-granularity cross-modal interaction compose synergistic evidence conditioned on the active fusion regime. The orthogonalityinducing parameterization provides a non-interference bias for the shared and modality-unique streams, while task memory governs task-conditioned synergy composition.

We interpret the learned streams as an approximate functional factorization. When a single latent space jointly represents shared evidence, modality-specific cues, and interaction-dependent efects, optimization may over-share information and suppress weak but task-critical distinctive cues. The sequential shared/unique decomposition and task-memory-guided interaction provide a structured training bias that mitigates this interference while retaining taskdependent flexibility.

Under this interpretation, RADIANT should exhibit three practical behaviors that we later verify empirically: reduced harmful overlap between shared and non-shared streams, better preservation of modality-specific cues useful for reconstruction, and stronger task-dependent interaction gains when the best fusion policy varies across datasets. Supplementary Secs. A and S2 provide the extended discussion and diagnostics.

This viewpoint also clarifies why a universal model can remain lightweight. RADIANT does not attempt to learn a separate back bone or decoder for each fusion regime. Instead, the encoder supplies a common joint representation, while the decomposition and memory-guided interaction stages determine how much information should remain shared, stay modality-specific, or be recombined as synergy for the current task. In other words, universality is achieved by reallocating information within a shared pipeline rather than by multiplying task-specific capacity. This is the reason the model can cover heterogeneous benchmarks without abandoning a compact backbone or resorting to full task-wise retraining.

Our empirical analyses in Supplementary Secs. A and S2 evaluate whether this allocation strategy reduces harmful overlap while preserving task-relevant signal. The semantic roles of the learned streams are determined jointly by $\mathcal { L } _ { \mathrm { f u s i o n } } , \mathcal { L } _ { \mathrm { a l i g n } } .$ , and L<sub>orthogonal</sub>.

## 3.2 Non-Interfering Shared/Unique Decomposition

Given joint features $\mathbf { F } \in \mathbb { R } ^ { N \times D }$ , we construct a shared stream and two modality-aware residual streams using learnable projections

$$
\mathbf { F } _ { r } = \mathbf { F } \mathbf { Q } _ { r } , \qquad \mathbf { F } _ { u 1 } = \mathbf { F } \mathbf { Q } _ { u 1 } , \qquad \mathbf { F } _ { u 2 } = \mathbf { F } \mathbf { Q } _ { u 2 } ,\tag{2}
$$

where the projection matrices are parameterized sequentially as

$$
\begin{array} { r } { \mathbf { Q } _ { r } = \exp ( \mathbf { A } _ { r } - \mathbf { A } _ { r } ^ { \top } ) \mathbf { W } _ { r } , } \end{array}\tag{3}
$$

$$
\begin{array} { r } { \mathbf { Q } _ { u 1 } = \exp ( \mathbf { A } _ { u 1 } - \mathbf { A } _ { u 1 } ^ { \top } ) ( \mathbf { I } - \mathbf { Q } _ { r } \mathbf { Q } _ { r } ^ { \top } ) \mathbf { W } _ { u 1 } , } \end{array}\tag{4}
$$

$$
\mathbf { Q } _ { u 2 } = \exp ( \mathbf { A } _ { u 2 } - \mathbf { A } _ { u 2 } ^ { \top } ) ( \mathbf { I } - \mathbf { Q } _ { r } \mathbf { Q } _ { r } ^ { \top } - \mathbf { Q } _ { u 1 } \mathbf { Q } _ { u 1 } ^ { \top } ) \mathbf { W } _ { u 2 } .\tag{5}
$$

Here $\mathbf { A } . \in \mathbb { R } ^ { D \times D }$ are learnable skew generators and $\mathbf { W } . \in \mathbb { R } ^ { D \times D / 3 }$ are projection weights. The matrix exponential contributes an orthogonal preconditioner with condition number 1, which stabilizes the basis rotation before task-adaptive shaping by W<sub>·</sub>. We do not claim that the full matrices $Q _ { r } , Q _ { u 1 } , Q _ { u 2 }$ remain exactly orthogonal or perfectly conditioned after multiplication with $\mathbf { W } .$ ; likewise, the complement operators in Eqs. (4)–(5) should be interpreted as approximate non-interference projectors, not exact geometric projectors.

This approximation is intentional. Using the orthogonal factor alone was empirically less expressive in preliminary experiments, while the additional W<sub>·</sub> terms allow task-adaptive subspace shaping. Stability in our setting therefore comes from the combination of orthogonal preconditioning, sequential complement masking, and explicit overlap regularization, rather than from any claim that the learned subspaces are exactly mutually orthogonal. We enforce near-semi-orthonormality with Eq. (6) and monitor the resulting overlap rather than asserting an exact decomposition. The regularizer is

Projection sanity check. On LLVIP, the overlap penalty in Eq. (6) drives the learned columns toward near-semi-orthonormality. Supplementary Sec. S5 compares the matrix-exponential parameterization with simpler alternatives and shows the best accuracy–stability trade-of.

$$
\mathcal { L } _ { \mathrm { o r t h o g o n a l } } = \sum _ { i \neq j } \| \mathbf { F } _ { i } ^ { \top } \mathbf { F } _ { j } \| _ { F } ^ { 2 } + \lambda _ { \mathrm { r e g } } \sum _ { i \in \{ r , u 1 , u 2 \} } \| \mathbf { Q } _ { i } ^ { \top } \mathbf { Q } _ { i } - \mathbf { I } \| _ { F } ^ { 2 } ,\tag{6}
$$

where $\left\{ \mathbf { F } _ { i } \right\} = \left\{ \mathbf { F } _ { r } , \mathbf { F } _ { u 1 } , \mathbf { F } _ { u 2 } \right\}$ and $\lambda _ { \mathrm { r e g } } = 0 . 0 1$ . Table 1 summarizes the average orthogonality residual measured on LLVIP and shows that the full parameterization remains close to semi-orthonormal in practice while outperforming the orthogonal-only variant.

Table 1: Projection sanity check on LLVIP. Residual is the average $\| { \bf Q } _ { i } ^ { \top } { \bf Q } _ { i } - { \bf I } \| _ { I }$ over $i \in \{ r , u 1 , u 2 \}$ on the validation split. Lower is better.
<table><tr><td>Variant</td><td>Residual ↓</td><td>EI↑</td></tr><tr><td>Orthogonal factor only (W. = I / fixed basis)</td><td>0.021</td><td>65.31</td></tr><tr><td>Full sequential parameterization</td><td>0.034</td><td>66.08</td></tr></table>

Supplementary Sec. S5 compares matrix exponential against QR, Cayley, and linear alternatives, showing it is the most stable implementation among the tested choices.

## 3.3 Task Memory with Hard Positive Updates

For each task $k \in \{ 1 , \ldots , K \}$ , we maintain a learnable memory anchor $\mathbf { p } _ { k } \in \mathbb { S } ^ { d - 1 }$ that summarizes the interaction regime preferred by that task. Rather than treating prototypes as the endpoint of supervision, RADIANT uses them to modulate which synergistic interactions should be activated. For a training sample � from task �, the synergy-conditioned stream is first pooled and projected into the memory space,

$$
\bar { \mathbf { f } } _ { s , k } ^ { ( i ) } = \mathrm { G A P } ( \mathbf { F } _ { s , k } ^ { ( i ) } ) , \qquad \mathbf { f } _ { s , k } ^ { ( i ) } = \mathrm { n o r m a l i z e } \left( \mathbf { W } _ { q } \bar { \mathbf { f } } _ { s , k } ^ { ( i ) } + \mathbf { b } _ { q } \right) ,\tag{7}
$$

where $\mathbf { W } _ { q } \in \mathbb { R } ^ { d \times D _ { s } }$ and $\mathbf { b } _ { q } \in \mathbb { R } ^ { d }$ are shared query parameters and $\mathbf { F } _ { s , k } ^ { ( i ) }$ denotes the synergy feature map produced by the multigranularity interaction module for that sample.

Hard Positive Mining. We emphasize challenging within-task samples,

$$
\mathcal { H } _ { k } ^ { + } = \left\{ i \in \mathcal { P } _ { k } : \cos ( \mathbf { f } _ { s , k } ^ { ( i ) } , \mathbf { p } _ { k } ) < \cos ( \theta _ { \mathrm { h a r d } } ) \right\} ,\tag{8}
$$

where $\mathbf { f } _ { s , k } ^ { ( i ) }$ is the normalized synergy query defined above, $\mathcal { P } _ { k }$ indexes samples from task $k ,$ and $\theta _ { \mathrm { h a r d } } = 6 0 ^ { \circ }$ is the hard-positive angular threshold. This focuses the memory update on samples whose interaction patterns are insuficiently aligned with the current task memory.

![](images/3c85179be8dc02e4aa26f4daf4c1ebdb4c65e24a53062b7a2f5fef2150d861a4.jpg)  
Figure 3: Task-memory evolution with hard positive updates. Challenging within-task samples guide the memory anchor toward stable interaction patterns, improving synergy selection while keeping task anchors separated.

Stochastic Memory Evolution. Memory anchors are updated with momentum and controlled exploration noise:

$$
{ \bf p } _ { k } ^ { ( t + 1 ) } = \mathrm { n o r m a l i z e } \left( \gamma _ { \mathrm { p r o t o } } { \bf p } _ { k } ^ { ( t ) } + \left( 1 - \gamma _ { \mathrm { p r o t o } } \right) \mu _ { k } ^ { ( t ) } + \sigma _ { \mathrm { p r o t o } } \xi ^ { ( t ) } \right) ,\tag{9}
$$

$$
\pmb { \mu } _ { k } ^ { ( t ) } = \frac { 1 } { | \mathcal { H } _ { k } ^ { + } | } \sum _ { i \in \mathcal { H } _ { k } ^ { + } } \mathbf { f } _ { s , k } ^ { ( i ) } , \quad \pmb { \xi } ^ { ( t ) } \sim N ( \mathbf { 0 } , \mathbf { I } ) ,\tag{10}
$$

with $\gamma _ { \mathrm { p r o t o } } = 0 . 9 7$ and $\sigma _ { \mathrm { p r o t o } } = 0 . 1 4 . \mathrm { I f } | \mathcal { H } _ { k } ^ { + } | = 0 ,$ , all samples in $\mathcal { P } _ { k }$ are used.

Memory Alignment Loss. For each sample � from task �, the same synergy query $\mathbf { f } _ { s , k } ^ { ( i ) }$ is contrasted against all task memories $\{ \mathbf { p } _ { j } \} _ { j = 1 } ^ { K } \colon$

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { a l i g n } } = - \displaystyle \sum _ { k = 1 } ^ { K } \sum _ { i \in \mathcal { B } _ { k } } \log \frac { \exp ( \kappa \mathbf { f } _ { s , k } ^ { ( i ) } \cdot \mathbf { p } _ { k } ) } { \sum _ { j = 1 } ^ { K } \exp ( \kappa \mathbf { f } _ { s , k } ^ { ( i ) } \cdot \mathbf { p } _ { j } ) } } \\ & { \quad \quad \quad \quad + \lambda _ { P } \displaystyle \sum _ { j \neq k } ( \mathbf { p } _ { k } ^ { \top } \mathbf { p } _ { j } ) ^ { 2 } , } \end{array}\tag{11}
$$

where $\kappa = 1 6 , \lambda _ { \mathscr P } = 0 . 0 1$ , and the denominator simply measures how strongly the current sample’s synergy query matches all memories, not just its own task anchor. Supplementary Sec. S7 shows that hard positive updates remain more efective than hard negative mining in this setting.

## 3.4 Multi-Granularity Cross-Modal Interaction for Synergy Composition

Synergistic evidence is not assumed to exist as a static subspace. Instead, it is composed through cross-modal interaction at pixel, block, and global scales. We therefore keep the multi-granularity contrastive machinery, but interpret it as identifying interaction patterns that should contribute to F .

Pixel-level Interaction. Let $\mathbf { z } _ { m } ^ { ( i , j ) } \in \mathbb { R } ^ { d }$ denote the projected feature from modality $m \in \{ 1 , 2 \}$ at location (�, �). The pixel-level objective is

$$
\mathcal { L } _ { \mathrm { p i x e l } } = - \frac { 1 } { 2 H W } \sum _ { i , j } [ \log \frac { e ^ { \mathbf { z } _ { 1 } ^ { ( i , j ) } \cdot \mathbf { z } _ { 2 } ^ { ( i , j ) } / \tau _ { c } } } { \sum _ { i ^ { \prime } , j ^ { \prime } } e ^ { \mathbf { z } _ { 1 } ^ { ( i , j ) } \cdot \mathbf { z } _ { 2 } ^ { ( i ^ { \prime } , j ^ { \prime } ) } / \tau _ { c } } } + ( 1  2 ) ] ,\tag{12}
$$

which encourages fine-grained correspondence useful for constructing synergy.

Block-level Interaction. We divide features into non-overlapping blocks $\{ \mathcal { B } _ { b } \} _ { b = 1 } ^ { B }$ and compute

$$
\mathbf { z } _ { b , m } = \sum _ { ( i , j ) \in \mathcal { B } _ { b } } \alpha _ { i , j } ^ { ( m ) } \mathbf { h } _ { m } ^ { ( i , j ) } , \quad \alpha _ { i , j } ^ { ( m ) } = \frac { e ^ { \mathbf { w } ^ { \top } \mathbf { h } _ { m } ^ { ( i , j ) } } } { \sum _ { ( i ^ { \prime } , j ^ { \prime } ) \in \mathcal { B } _ { b } } e ^ { \mathbf { w } ^ { \top } \mathbf { h } _ { m } ^ { ( i ^ { \prime } , j ^ { \prime } ) } } } ,\tag{13}
$$

then apply the same InfoNCE form with contrastive temperature $\tau _ { c }$ to $\{ \mathbf { z } _ { b , m } \}$ to capture locally compositional interactions.

Global Interaction. Global descriptors $\mathbf { g } _ { m } \mathbf { \Psi } = \mathbf { G } \mathbf { A } \mathbf { P } ( \mathbf { h } _ { m } )$ are aligned via

$$
\mathcal { L } _ { \mathrm { g l o b a l } } = - \log \frac { e ^ { \mathbf { g } _ { 1 } ^ { \top } \mathbf { g } _ { 2 } / \tau _ { c } } } { \sum _ { k = 1 } ^ { N } e ^ { \mathbf { g } _ { 1 } ^ { \top } \mathbf { g } _ { k } / \tau _ { c } } } + \lambda _ { \mathrm { s p e c t r a l } } \| \mathbf { g } _ { 1 } - \mathbf { g } _ { 2 } \| _ { 2 } ^ { 2 } ,\tag{14}
$$

with $\lambda _ { \mathrm { { s p e c t r a l } } } ~ = ~ 0 . 1$ . This term stabilizes scene-level interaction statistics.

Granularity weights $w _ { g }$ for $g \in$ {pixel, block, global} are computed adaptively via softmax over EMA-smoothed MI surrogates. We define the surrogate at granularity � as

$$
\widehat { \mathrm { M } } _ { g } ^ { ( t ) } = \rho \widehat { \mathrm { M } } _ { g } ^ { ( t - 1 ) } + \left( 1 - \rho \right) \left( - \mathcal { L } _ { g } ^ { ( t ) } + \log | N _ { g } ^ { ( t ) } | \right) ,\tag{15}
$$

where $\mathcal { L } _ { q } ^ { ( t ) }$ is the corresponding InfoNCE loss on batch $t , | N _ { g } ^ { ( t ) } |$ is the number of contrastive candidates at that granularity, and $\rho = 0 . 9 5$ is the EMA decay. The adaptive weights are then

$$
w _ { g } ^ { ( t ) } = \frac { \exp ( \widehat { \mathrm { M I } } _ { g } ^ { ( t ) } / \tau _ { w } ) } { \sum _ { g ^ { \prime } } \exp ( \widehat { \mathrm { M I } } _ { g ^ { \prime } } ^ { ( t ) } / \tau _ { w } ) } .\tag{16}
$$

This quantity is a ranking surrogate rather than a calibrated MI estimator; it is used only to compare the relative interaction strength of the three granularities online. Supplementary Sec. S6 gives the detailed implementation and ablation.

## 3.5 Allocation Decoder and Training Objective

The fused image is supervised by the standard reconstruction objective

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { f u s i o n } } = \lambda _ { \mathrm { i n t } } \underbrace { \Vert \mathbf { Y } - \operatorname* { m a x } ( \mathbf { I } _ { 1 } , \mathbf { I } _ { 2 } ) \Vert _ { 1 } } _ { \mathcal { L } _ { \mathrm { i n t e n s i t y } } } } \\ & { ~ + ~ \lambda _ { \mathrm { g r a d } } \underbrace { \Vert \nabla \mathbf { Y } - \operatorname* { m a x } ( | \nabla \mathbf { I } _ { 1 } | , | \nabla \mathbf { I } _ { 2 } | ) \Vert _ { 1 } } _ { \mathcal { L } _ { \mathrm { g r a d } } } , } \end{array}\tag{17}
$$

where $\lambda _ { \mathrm { i n t } } = 1 . 0$ and $\lambda _ { \mathrm { g r a d } } = 1 0 . 0 .$

We use Eq. (17) as a task-agnostic reconstruction prior rather than as a claim that all fusion domains obey the same physical image-formation model. The role of the max-intensity and gradient terms is simply to provide a common low-level supervision signal across benchmarks, while task-specific specialization is handled by the allocation streams and task memory. We therefore make the narrower claim that this shared objective is suficient for controlled universal training on the evaluated benchmarks, not that it is the uniquely optimal loss for every fusion domain.

The decoder allocates information by combining shared structure with gated unique evidence and task-conditioned synergy:

$$
\begin{array} { r } { \pmb { s } = \sigma \big ( \mathrm { C o n v } ( [ \nabla \mathbf { I } _ { 1 } , \nabla \mathbf { I } _ { 2 } , \mathbf { F } _ { r } ] ) \big ) , \quad \widetilde { \mathbf { F } } _ { u } = \pmb { s } \odot [ \mathbf { F } _ { u 1 } , \mathbf { F } _ { u 2 } ] , } \end{array}\tag{18}
$$

Algorithm 1 RADIANT Training Procedure   
Require: Training data $\{ ( \mathbf { I } _ { 1 } ^ { ( n ) } , \mathbf { I } _ { 2 } ^ { ( n ) } , k ^ { ( n ) } ) \} _ { n = 1 } ^ { N }$ with task labels   
$k ^ { ( n ) } \in \{ 1 , \ldots , K \}$   
Require: Learning rate �, initialized task memories $\{ \mathbf { p } _ { k } ^ { ( 0 ) } \} _ { k = 1 } ^ { K }$ , hy  
perparameters $( \alpha , \beta , \gamma , \gamma _ { \mathrm { p r o t o } } , \sigma _ { \mathrm { p r o t o } } , \kappa , \tau _ { c } , \tau _ { w } )$   
Ensure: Trained encoder $\phi _ { \theta } ,$ decoder $\psi _ { \omega } ,$ and task memories   
$\{ \mathbf { p } _ { k } ^ { * } \} _ { k = 1 } ^ { K }$   
1: for $t = 1$ to � do   
2: for each mini-batch $\mathcal { B } _ { t }$ do   
3: $\mathbf { F } \gets \phi _ { \theta } ( \mathcal { B } _ { t } )$   
4: $\left\{ \mathbf { F } _ { r } , \mathbf { F } _ { u 1 } , \mathbf { F } _ { u 2 } \right\}$ ← NonInterferingDecompose(F)   
5: Compute L<sub>ortho onal</sub> from pairwise overlap among   
$\mathbf { F } _ { r } , \mathbf { F } _ { u 1 } , \mathbf { F } _ { u 2 }$   
6: for each granularity $g \in$ {pixel, block, global} do   
7: Compute cross-modal interaction loss $\mathcal { L } _ { g } ^ { ( t ) }$   
8: Update EMA surrogate $\widehat { \mathrm { M I } } _ { g } ^ { ( t ) } \ \gets \ \rho \widehat { \mathrm { M I } } _ { g } ^ { ( t - 1 ) } + \ ( 1 \ -$   
$\rho ) ( - \mathcal { L } _ { g } ^ { ( t ) } + \log | N _ { g } ^ { ( t ) } | )$   
9: end for   
10: Compute adaptive weights $w _ { g } ^ { ( t ) } \gets \mathrm { s o f t m a x } ( \widehat { \mathrm { M I } } _ { g } ^ { ( t ) } / \tau _ { w } )$   
11: Compose synergy features $\mathbf { F } _ { s }$ through task-memory  
guided multi-granularity interaction   
12: for each task � do   
13: Form pooled synergy queries $\mathbf { f } _ { s , k } ^ { ( i ) }$ from $\mathbf { F } _ { s , k } ^ { ( i ) }$   
14: Mine hard positives $\mathcal { H } _ { k } ^ { + }$ using $\theta _ { \mathrm { h a r d } }$ and compute $\pmb { \mu } _ { k } ^ { ( t ) }$   
15: Sample $\xi ^ { ( t ) } \sim { \cal N } ( 0 , \mathrm { I } )$ and update task memory $\mathbf { p } _ { k } ^ { ( t + 1 ) }$   
16: end for   
17: Compute $\mathcal { L } _ { \mathrm { a l i g n } }$ over task memories   
18: s ← � Conv $\bigl [ \nabla  { \mathbf { I } } _ { 1 } , \nabla  { \mathbf { I } } _ { 2 } ,  { \mathbf { F } } _ { r } \bigr ] \bigr ) \bigr )$   
19: $\widetilde { \mathbf { F } } _ { u } \gets s \odot [ \mathbf { F } _ { u \underline { { 1 } } } , \mathbf { F } _ { u 2 } ]$   
20: $\mathbf { Y } \gets \psi _ { \omega } ( [ \mathbf { F } _ { r } , \widetilde { \mathbf { F } } _ { u } , \mathbf { F } _ { s } ] )$   
21: Compute L<sub>fusion</sub> and $\begin{array} { r l r } { \mathcal { L } _ { \mathrm { t o t a l } } } & { { } = } & { \mathcal { L } _ { \mathrm { f u s i o n } } ~ + ~ \alpha \mathcal { L } _ { \mathrm { a l i g n } } ~ + ~ } \end{array}$   
$\beta \mathcal { L } _ { \mathrm { o r t h o g o n a l } } + \gamma \sum _ { g } w _ { g } \mathcal { L } _ { g }$   
22: Update $( \theta , \omega )$ via AdamW with gradient clipping   
23: end for   
24: end for

where $\widetilde { \mathbf { F } } _ { u }$ is the selectively preserved unique evidence. Synergy features $\mathbf { F } _ { s }$ are produced by multi-granularity interaction conditioned on the current task memory, and the decoder input becomes $[ \mathbf { F } _ { r } , \widetilde { \mathbf { F } } _ { u } , \mathbf { F } _ { s } ]$

The unified objective is

$$
\mathcal { L } _ { \mathrm { t o t a l } } = \mathcal { L } _ { \mathrm { f u s i o n } } + \alpha \mathcal { L } _ { \mathrm { a l i g n } } + \beta \mathcal { L } _ { \mathrm { o r t h o g o n a l } } + \gamma \sum _ { g \in \{ p , b , g l \} } w _ { g } \mathcal { L } _ { g } ,\tag{19}
$$

with default weights $\alpha = 0 . 5 , \beta = 0 . 1$ , and $\gamma = 1 . 0 .$ In this formulation, $\mathcal { L } _ { \mathrm { o r t h o g o n a l } }$ protects non-interference, ${ \mathcal { L } } _ { \mathrm { a l i g n } }$ stabilizes task memory, and the multi-granularity interaction terms determine which synergistic relations should be emphasized.

## 4 Experiments

## 4.1 Experimental Setup

RADIANT is evaluated across six representative fusion tasks spanning diverse application domains: (1) IVIF: infrared-visible fusion on LLVIP [10] (12,025 pairs, 320 test) and TNO (40 pairs); (2) MFIF: multi-focus fusion on Lytro [15] (20 pairs) and MFI-WHU [30] (120 pairs); (3) MEIF: multi-exposure fusion on DSCIE (589 pairs); (4) NIR-VIS: near-infrared fusion on Scene (477 pairs); (5) Remote sensing: pansharpening on QuickBird (50 pairs); (6) Medical imag ing: PET-MRI fusion on Harvard (302 pairs). Dataset splits follow oficial protocols; URLs and licensing details are provided in Supplementary Sec. S0.

Table 2: Quantitative results on six fusion tasks. Bold: best; Underline: second best. All metrics are higher-better. \*: task-specific methods.
<table><tr><td></td><td colspan="3">(a1) MFIF: Lytro</td><td colspan="4">(a2) MFIF: MFI-WHU</td><td colspan="5"></td><td colspan="5">(d) NIR-VIS: Scene</td></tr><tr><td>Method</td><td>EI</td><td>VIF</td><td>SCD</td><td>AG</td><td>EI</td><td>VIF</td><td>SCD AG</td><td>Method</td><td></td><td>EI</td><td>VIF SCD</td><td>AG</td><td>Method</td><td>EI</td><td>VIF</td><td>SCD</td><td>AG</td></tr><tr><td>U2Fusion</td><td>67.20</td><td>1.39</td><td>0.84</td><td>6.34</td><td>79.11</td><td>1.50</td><td>0.56 7.88</td><td>U2Fusion</td><td></td><td>83.00 1.69</td><td>0.49</td><td>8.71</td><td>|IFCNN</td><td>82.13</td><td>0.90</td><td>1.14</td><td>8.72</td></tr><tr><td>UNIFusion*</td><td>70.14</td><td>1.30</td><td>0.60</td><td>6.77</td><td>66.57</td><td>1.01 0.29</td><td>7.19</td><td>SPD-MEF*</td><td>78.17</td><td>1.72</td><td>0.49</td><td>8.12</td><td>U2Fusion</td><td>80.73</td><td>1.07</td><td>1.19</td><td>8.51</td></tr><tr><td>SDNet</td><td>62.98</td><td>1.12</td><td>0.75</td><td>6.16</td><td>72.98</td><td>1.16 0.63</td><td>7.98</td><td>MEF-GAN*</td><td>80.21</td><td>1.59</td><td>0.63</td><td>8.06</td><td>SDNet</td><td>76.70</td><td>0.86</td><td>0.72</td><td>8.26</td></tr><tr><td>MUFusion</td><td>70.40</td><td>1.34</td><td>1.22</td><td>6.67</td><td>77.72</td><td>1.36 1.11</td><td>7.92</td><td>MUFusion</td><td>70.18</td><td>1.64</td><td>0.96</td><td>7.19</td><td>MURF*</td><td>41.71</td><td>0.41</td><td>0.19</td><td>4.22</td></tr><tr><td>ZMFF*</td><td>58.97</td><td>1.11</td><td>0.36</td><td>5.48</td><td>57.90</td><td>1.03 0.33</td><td>5.49</td><td>IID-MEF*</td><td>59.12</td><td>1.12</td><td>0.36</td><td>6.13</td><td>Text-IF</td><td>88.19</td><td>1.49</td><td>1.45</td><td>9.16</td></tr><tr><td>GIF</td><td>80.86</td><td>1.74</td><td>1.37</td><td>7.71</td><td>91.61</td><td>1.94 1.29</td><td>9.25</td><td>GIF</td><td>111.27</td><td>2.52</td><td>1.04</td><td>12.00</td><td>GIF</td><td>101.32</td><td>1.51</td><td>1.45</td><td>10.83</td></tr><tr><td>RADIANT</td><td>85.55</td><td>1.86</td><td>1.45</td><td>8.16</td><td>97.93</td><td>2.08 1.38</td><td>9.89</td><td>RADIANT</td><td>118.73</td><td>2.71</td><td>1.12</td><td>12.88</td><td>RADIANT</td><td>108.20</td><td>1.62</td><td>1.56</td><td>11.62</td></tr></table>

<table><tr><td colspan="5">(b1) IVIF: LLVIP</td><td colspan="4">(b2) IVIF: TNO</td><td colspan="5">(e) Remote: QuickBird</td><td colspan="5">(f) Medical: Harvard</td></tr><tr><td>Method</td><td>EI</td><td>VIF</td><td>SCD</td><td>AG</td><td>EI</td><td>VIF</td><td>SCD</td><td>AG</td><td>Method</td><td>EI</td><td>VIF SCD</td><td></td><td>AG</td><td>Method</td><td>EI</td><td>VIF</td><td>SCD</td><td>AG</td></tr><tr><td>MURF</td><td>38.02</td><td>0.25</td><td>1.56</td><td>3.75</td><td>45.93</td><td>0.94</td><td>1.52</td><td>4.35</td><td>IFCNN</td><td>18.30</td><td>1.16</td><td>0.86</td><td>1.73</td><td>U2Fusion</td><td>73.29</td><td>0.78</td><td>1.46</td><td>7.06</td></tr><tr><td>LRRNet*</td><td>34.93</td><td>0.34</td><td>0.95</td><td>3.64</td><td>36.37</td><td>0.75</td><td>1.40</td><td>3.75</td><td>TextFusion</td><td>13.73</td><td>0.76</td><td>0.32</td><td>1.29</td><td>IFCNN</td><td>98.67</td><td>0.92</td><td>1.33</td><td>9.57</td></tr><tr><td>DDFM*</td><td>41.13</td><td>0.51</td><td>1.55</td><td>4.43</td><td>31.01</td><td>0.67</td><td>1.60</td><td>3.03</td><td>MUFusion</td><td>18.45</td><td>1.05</td><td>0.89</td><td>1.72</td><td>SDNet</td><td>83.86</td><td>0.71</td><td>1.60</td><td>8.39</td></tr><tr><td>CDDFuse*</td><td>52.32</td><td>0.79</td><td>1.58</td><td>5.42</td><td>43.83</td><td>0.91 1.66</td><td></td><td>4.55</td><td>P2Sharpen*</td><td>16.66</td><td>1.20</td><td>0.94</td><td>1.56</td><td>MUFusion</td><td>88.66</td><td>0.97</td><td>1.23</td><td>8.39</td></tr><tr><td>Text-IF*</td><td>61.30</td><td>0.93</td><td>1.49</td><td>6.33</td><td>46.09</td><td>1.04 1.53</td><td></td><td>4.61</td><td>ZeroSharpen*</td><td>13.20</td><td>0.94 0.48</td><td></td><td>1.26</td><td>CoCoNet*</td><td>89.55</td><td>0.71</td><td>1.04</td><td>8.84</td></tr><tr><td>GIF</td><td>62.46</td><td>0.73</td><td>1.61</td><td>6.70</td><td>52.30</td><td>0.99</td><td>1.66</td><td>5.24 GIF</td><td></td><td>23.02</td><td>1.56</td><td>1.04</td><td>2.16</td><td>GIF</td><td>100.71</td><td>1.10</td><td>1.68</td><td>9.73</td></tr><tr><td>RADIANT</td><td>66.08</td><td>0.78</td><td>1.70</td><td>7.09</td><td>55.33</td><td>1.06</td><td>1.78</td><td>5.62</td><td>RADIANT</td><td>24.85</td><td>1.67</td><td>1.12</td><td>2.32</td><td>RADIANT</td><td>107.06</td><td>1.18</td><td>1.80</td><td>10.44</td></tr></table>

We employ five complementary metrics: Edge Intensity (EI) measuring edge preservation, Visual Information Fidelity (VIF) [17] quantifying information transfer, Sum of Correlation Diference (SCD) assessing structural consistency, Average Gradient (AG) evaluating detail richness, and Entropy (EN) measuring information content (reported in Supplementary Sec. S4). Higher values indicate better performance for all metrics.

Inputs are resized to 256 × 256 and normalized per task using the standard preprocessing protocol of each benchmark. Across tasks, intensity normalization is kept modality-aware rather than globally shared so that modality-unique cues are not artificially collapsed before fusion.

Training uses AdamW optimizer $( \ln 2 \times 1 0 ^ { - 4 }$ , weight decay 5 × $1 0 ^ { - 2 } )$ with cosine annealing, batch size 16, input resolution 256×256, and gradient clipping at 1.0. Loss weights are $\alpha = 0 . 5 , \beta = 0 . 1 , \gamma =$ 1.0, and $\lambda _ { \mathrm { r e g } } = 0 . 0 1$ . Task-memory parameters are $\kappa = 1 6 , \gamma _ { \mathrm { p r o t o } } =$ $0 . 9 7 , \sigma _ { \mathrm { p r o t o } } = 0 . 1 4 ,$ , and $\lambda _ { p } = 0 . 0 1$ . The contrastive temperature is $\tau _ { c } = 0 . 0 7$ , the granularity-weight temperature is $\tau _ { w } = 0 . 0 7$ , and the hard-positive angular threshold is $\theta _ { \mathrm { h a r d } } = 6 0 ^ { \circ }$ . Task memories are initialized from �-means centroids after one warm-up epoch. The tiny Vision Mamba–Swin backbone uses shallow dual branches with narrow channel widths (see Supplementary Sec. S3), which explains the unusually small 3.108 MB footprint.

Systematic ±50% perturbation sweeps across all hyperparameters yield ≤ 1.2% average relative change in metrics (MAPD), demonstrating robust performance (see Supplementary Sec. S1 for detailed sensitivity analysis). These results indicate limited sensitivity within the evaluated ranges.

We distinguish direct cross-dataset transfer from unlabeled testtime memory adaptation. In direct transfer, a model trained on LLVIP is evaluated directly on TNO without fine-tuning or targetmemory updates. For test-time adaptation, all network weights remain frozen, while a new task memory is initialized from 50 unlabeled target samples and updated for 10 steps. Supplementary Sec. S9 reports the corresponding sample-budget and update-step analyses.

## 4.2 Main Results

Table 2 compares RADIANT with state-of-the-art methods across six fusion tasks. RADIANT consistently achieves the best performance, indicating that a single allocation strategy can adapt to heterogeneous fusion regimes without task-specific redesign. Results are averaged over 3 seeds; our significance statement refers specifically to comparisons against GIF, the strongest consistently available universal baseline across the six tasks.

To evaluate whether the learned streams exhibit reduced overlap, Table 3 reports two compact diagnostics on representative tasks: the reduction in shared–unique overlap $\Delta I ( \mathbf { F } _ { r } ; \mathbf { F } _ { u } )$ and the reduction in shared–synergy overlap $\Delta I ( \mathbf { F } _ { r } ; \mathbf { F } _ { s } )$ , measured pre-/post-training using the estimators detailed in Supplementary Sec. S2. The consistent negative shifts support the claim that RADIANT reduces harmful overlap while preserving task-relevant structure.

On FLIR thermal-visible detection using YOLOv5-L, RADIANT substantially outperforms GIF and single-modality baselines. Supplementary Sec. S8 provides the complete detection protocol.

## 4.3 Eficiency Analysis

Table 4 compares computational requirements. RADIANT maintains state-of-the-art fusion quality with only 3.108 MB parameters, showing that explicit information allocation need not incur the cost of heavyweight task-specific pipelines. The underlying backbone and projectors are intentionally shallow and narrow, and the extra allocation machinery is lightweight relative to competing universal models.

![](images/e455848a61c4e6ffe3c37fdca706e76e8d9747932d9d21567a8e38fbb85194be.jpg)  
Figure 4: Visual comparison across fusion tasks. Red boxes mark regions where allocation-aware fusion better preserves fine details and cross-modal complements.

Table 3: Compact information-allocation diagnostics on two representative tasks. Relative changes (%) in KSG-estimated mutual information after training. Negative values indicate reduced overlap between the corresponding streams; negative Δ indicates reduced harmful overlap after training.
<table><tr><td>Task</td><td>∆I(Fr;Fu)</td><td>∆I(Fr;Fs)</td></tr><tr><td>LLVIP</td><td>-23.2%</td><td>-19.7%</td></tr><tr><td>Harvard</td><td>-21.0%</td><td>-18.7%</td></tr></table>

The compact model size is achieved through depthwise separable convolutions in the encoder/decoder, weight-tied projectors for multi-granularity interaction, and shared lightweight MLP heads. The hybrid Vision Mamba–Swin backbone provides �(�) complexity for long-range dependencies while maintaining local detail preservation.

Table 4: Computational eficiency comparison. Module-wise breakdown in Supplementary Table S3.
<table><tr><td>Method</td><td>Technique</td><td>Size (MB)</td><td>GFLOPs</td><td>FPS</td></tr><tr><td>DDFM</td><td>Diffusion model</td><td>2211.00</td><td>1840.49</td><td>0.3</td></tr><tr><td>Text-IF</td><td>ViLT</td><td>580.81</td><td>82.85</td><td>8.2</td></tr><tr><td>TC-MoA</td><td>MAE, MoE</td><td>426.98</td><td>524.28</td><td>2.1</td></tr><tr><td>GIF</td><td>Fusion only</td><td>3.30</td><td>9.96</td><td>38</td></tr><tr><td>RADIANT</td><td>Allocation + Task Memory</td><td>3.108</td><td>9.42</td><td>42</td></tr></table>

## 4.4 Ablation Study and Component Analysis

Table 5 evaluates the contribution of each component on LLVIP. The non-interfering decomposition and task memory both improve performance individually, and the strongest gains appear when they are paired with multi-scale interaction, supporting the view that universal fusion depends on coordinated allocation rather than on a single alignment block. The hybrid Mamba–Swin backbone further improves quality over either branch alone.

## 4.5 Robustness and Generalization Analysis

Table 6 reports direct cross-dataset transfer from LLVIP to TNO without fine-tuning or target-time memory adaptation. RADIANT substantially outperforms GIF and CDDFuse, suggesting that the learned allocation policy transfers more faithfully across datasets than methods based on a single fusion regime.

Table 5: Ablation study on LLVIP. Bold: best results. NI: non-interfering shared/unique decomposition; Task memory: memory anchors with hard positive updates; Mamba/Swin: backbone branches.
<table><tr><td>NI</td><td>Task memory</td><td>Mamba</td><td>Swin</td><td>Multi-scale interaction</td><td>EI↑</td><td>VIF↑</td><td>SCD↑</td><td>AG↑</td><td>∆%</td></tr><tr><td>X</td><td>X</td><td>×</td><td>×</td><td>×</td><td>52.31</td><td>0.61</td><td>1.42</td><td>5.18</td><td>一</td></tr><tr><td>√</td><td>X</td><td>×</td><td>×</td><td>X</td><td>60.12</td><td>0.70</td><td>1.55</td><td>6.30</td><td>+14.9</td></tr><tr><td>X</td><td>√</td><td>×</td><td>X</td><td>X</td><td>58.22</td><td>0.68</td><td>1.51</td><td>6.01</td><td>+11.3</td></tr><tr><td>√</td><td>√</td><td>X</td><td>√</td><td>√</td><td>62.74</td><td>0.73</td><td>1.60</td><td>6.58</td><td>+19.9</td></tr><tr><td>√</td><td>√</td><td>√</td><td>X</td><td>√</td><td>63.01</td><td>0.74</td><td>1.61</td><td>6.69</td><td>+20.5</td></tr><tr><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>66.08</td><td>0.78</td><td>1.70</td><td>7.09</td><td>+26.3</td></tr></table>

Table 6: Direct cross-dataset generalization (LLVIP → TNO) without fine-tuning or target-time adaptation. “Full” denotes training and evaluation on TNO, whereas “Transfer” denotes training on LLVIP and direct evaluation on TNO. Ratio = Transfer/Full ×100%.
<table><tr><td>Method</td><td colspan="4">Full (trained on TNO)</td><td colspan="4">Transfer (LLVIP→TNO)</td><td></td></tr><tr><td></td><td>EI</td><td>VIF</td><td>SCD</td><td>AG</td><td>EI</td><td>VIF</td><td>SCD</td><td>AG</td><td>Ratio</td></tr><tr><td>CDDFuse</td><td>43.83</td><td>0.91</td><td>1.66</td><td>4.55</td><td>31.25</td><td>0.64</td><td>1.18</td><td>3.25</td><td>71.3%</td></tr><tr><td>GIF</td><td>52.30</td><td>0.99</td><td>1.66</td><td>5.24</td><td>40.17</td><td>0.76</td><td>1.27</td><td>4.02</td><td>76.8%</td></tr><tr><td>RADIANT</td><td>55.33</td><td>1.06</td><td>1.78</td><td>5.62</td><td>49.35</td><td>0.95</td><td>1.59</td><td>5.01</td><td>89.2%</td></tr></table>

With 50 unlabeled TNO samples and 10 task-memory updates, EI increases from 49.35 to 50.16 while all network weights remain frozen. This setting is therefore unlabeled test-time memory adap tation rather than strict zero-shot inference; complete results are provided in Supplementary Sec. S9.

Comparisons with cross-entropy, center loss, and static EMA memory are reported in Supplementary Sec. S5.2. The complete task-memory formulation consistently achieves the best fusion quality and task accuracy.

## 4.6 Information Allocation Validation

Table 7 evaluates how the learned streams contribute to reconstruction and detection. Although the allocation is only an approximation to true redundancy/uniqueness/synergy decomposition, the results show a functional division of labor: the shared stream preserves structure, the unique stream preserves modalityspecific details, and the synergy-conditioned pathway contributes the strongest downstream discrimination.

$\mathbf { F } _ { r }$ preserves shared structure, the unique streams retain modalityspecific details, and $\mathrm { F } _ { s }$ captures interaction-dependent evidence that is especially beneficial for downstream tasks. Their combination yields the best overall performance.

Sensitivity to the hard-positive threshold is reported in Supplementary Sec. S7. Performance remains stable around the default setting $\theta _ { \mathrm { h a r d } } = 6 0 ^ { \circ }$

## 4.7 Visual Results and Analysis

Figure 4 demonstrates RADIANT’s visual performance across diverse fusion scenarios.

Table 7: Information allocation validation on LLVIP (reconstruction SSIM/PSNR) and FLIR (detection mAP@0.5). Metrics indicate complementary roles for shared, unique, and synergy-conditioned evidence.
<table><tr><td>Stream Used</td><td>SSIM</td><td>PSNR</td><td>mAP@0.5</td></tr><tr><td>Fr only</td><td>0.847</td><td>24.3</td><td>89.2%</td></tr><tr><td>[Fu1, Fu2] only</td><td>0.412</td><td>15.8</td><td>71.4%</td></tr><tr><td>Fs only</td><td>0.298</td><td>12.1</td><td>93.1%</td></tr><tr><td> $\mathbf { F } _ { r } + \left[ \mathbf { F } _ { u 1 } , \mathbf { F } _ { u 2 } \right]$ </td><td>0.923</td><td>28.7</td><td></td></tr><tr><td> $\mathbf { F } _ { r } + \mathbf { F } _ { s }$ </td><td></td><td>一</td><td>95.8%</td></tr><tr><td>Full model</td><td>0.941</td><td>30.2</td><td>96.8%</td></tr></table>

RADIANT demonstrates consistent visual superiority across diverse fusion tasks. In multi-focus scenarios, it preserves fine details without the blurring artifacts of competing methods. For infraredvisible fusion, it retains thermal saliency while preserving visible textures, reflecting better unique-evidence preservation. The method also produces more balanced luminance in exposure fusion and retains both anatomical and functional cues in PET/MRI fusion. These results support the central claim that universal fusion improves when redundant evidence is controlled, modality-unique evidence is preserved, and task-conditioned synergy is activated only where useful.

## 5 Conclusion

This paper presents RADIANT, a redundancy-aware information allocation framework for universal multi-modal fusion. RADIANT combines a non-interfering shared/unique parameterization with task-memory-guided synergy composition to allocate redundant, modality-unique, and task-conditioned synergistic information across fusion tasks. Experiments across six heterogeneous tasks show that this perspective yields strong universal performance, eficient inference, and robust transfer to unseen settings. More broadly, the results suggest that universal fusion benefits from controlling how information is assigned within a shared model, not only from improving alignment strength or enlarging task-specific capacity. By combining a compact joint encoder with explicit shared/unique separation and memory-guided synergy composition, RADIANT provides a practical route to scalable multimodal fusion that remains lightweight, transferable, and interpretable across heterogeneous regimes.

## Acknowledgments

This work was supported by the Research Institute of Newland Digital Technology.

## References

[1] Rich Caruana. 1997. Multitask Learning. Machine Learning 28, 1 (1997), 41–75. doi:10.1023/A:1007379606734

[2] Chunyang Cheng, Xiao-Jun Wu, Tianyang Xu, and Guoyang Chen. 2021. UNIFu sion: A Lightweight Unified Image Fusion Network. IEEE Transactions on Instrumentation and Measurement 70 (2021), 5016614. doi:10.1109/TIM.2021.3109379

[3] Chunyang Cheng, Tianyang Xu, and Xiao-Jun Wu. 2023. MUFusion: A General Unsupervised Image Fusion Network Based on Memory Unit. Information Fusion 92 (2023), 80–92.

[4] Giordano Cicchetti, Eleonora Grassucci, Luigi Sigillo, and Danilo Comminiello. 2025. Gramian Multimodal Representation Learning and Alignment. In International Conference on Learning Representations. OpenReview.net, Singapore.

[5] Benoit Dufumier, Javiera Castillo Navarro, Devis Tuia, and Jean-Philippe Thiran. 2025. What to Align in Multimodal Contrastive Learning?. In International Conference on Learning Representations. OpenReview.net, Singapore.

[6] Chao Fan, Zhixiang Chen, Xiao Wang, Zhihui Xuan, and Zhentong Zhu. 2023. RTFusion: A Multimodal Fusion Network with Significant Information Enhance ment. Journal ofDigital Imaging 36, 4 (2023), 1851–1863. doi:10.1007/s10278-023- 00810-3

[7] Ziyu Gong, Chengcheng Mai, and Yihua Huang. 2024. AsCL: An Asymmetry-Sensitive Contrastive Learning Method for Image-Text Retrieval with Cross Modal Fusion. In 2024 IEEE International Conference on Multimedia and Expo. IEEE, Los Alamitos, CA, USA, 1–6. doi:10.1109/ICME57554.2024.10687993

[8] Junjun Guo, Rui Su, and Junjie Ye. 2024. Multi-Grained Visual Pivot-Guided Multi-Modal Neural Machine Translation with Text-Aware Cross-Modal Contrastive Disentangling. Neural Networks 178 (2024), 106403. doi:10.1016/j.neunet.2024. 106403

[9] Qingqing Hu, Yiran Peng, KinTak U, and Siyuan Zhao. 2025. Infrared and Visible Image Fusion Using a State-Space Adversarial Model with Cross-Modal Depen dency Learning. Mathematics 13, 15 (2025), 2333. doi:10.3390/math13152333

[10] Xinyu Jia, Chuang Zhu, Minzhen Li, Wenqi Tang, and Wenli Zhou. 2021. LLVIP: A Visible-Infrared Paired Dataset for Low-Light Vision. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision Workshops. IEEE, Los Alamitos, CA, USA, 3496–3504.

[11] Jing Li, Yifan Wang, Jiafeng Yan, Renlong Zhang, and Bin Yang. 2026. MdaIF: Robust One-Stop Multi-Degradation-Aware Image Fusion with Language-Driven Semantics. Proceedings ofthe AAAI Conference on Artificial Intelligence 40, 8 (2026), 6226–6234. doi:10.1609/aaai.v40i8.37548

[12] Xin Li, Xin He, Gaigai Tang, Kaiyuan Qi, Guangfeng Su, and Huiyun Zhang. 2026. Discrepancy-Aware Contrastive Learning with Mixture of Experts for Cross Modal Image-Text Semantic Alignment. Neurocomputing 660 (2026), 131887.

[13] Jinyuan Liu, Runjia Lin, Guanyao Wu, Risheng Liu, Zhongxuan Luo, and Xin Fan. 2024. CoCoNet: Coupled Contrastive Learning Network with Multi-Level Feature Ensemble for Multi-Modality Image Fusion. International Journal ofComputer Vision 132, 5 (2024), 1748–1775. doi:10.1007/s11263-023-01952-1

[14] Bo Meng, Huaizhou Liu, and Zegang Ding. 2025. Multi-Scene Image Fusion via Memory Aware Synapses. Scientific Reports 15, 1 (2025), 14280. doi:10.1038/s41598- 025-88261-4

[15] Mansour Nejati, Shadrokh Samavi, and Shahram Shirani. 2015. Multi-Focus Image Fusion Using Dictionary-Based Sparse Representation. Information Fusion 25 (2015), 72–84. doi:10.1016/j.infus.2014.10.004

[16] Chengxuan Qian, Shuo Xing, Li Li, Yue Zhao, and Zhengzhong Tu. 2026. DecAlign: Hierarchical Cross-Modal Alignment for Decoupled Multimodal Rep resentation Learning. In International Conference on Learning Representations. OpenReview.net, Rio de Janeiro, Brazil.

[17] Hamid R. Sheikh and Alan C. Bovik. 2006. Image Information and Visual Quality. IEEE Transactions on Image Processing 15, 2 (2006), 430–444.

[18] Dayong Su, Yafei Zhang, Huafeng Li, Jinxing Li, and Yu Liu. 2025. UniFuse: A Unified All-in-One Framework for Multi-Modal Medical Image Fusion Under Diverse Degradations and Misalignments. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision. IEEE, Los Alamitos, CA, USA, 14238–14247.

[19] Linfeng Tang, Qinglong Yan, Xinyu Xiang, Leyuan Fang, and Jiayi Ma. 2025. C2RF: Bridging Multi-Modal Image Registration and Fusion via Commonality Mining and Contrastive Learning. International Journal ofComputer Vision 133, 8 (2025), 5262–5280. doi:10.1007/s11263-025-02427-1

[20] Zhenghang Tang, Nankai Lin, and Aimin Yang. 2025. Integrating Contrastive Learning and Multimodal Fusion Network for Fake News Detection. In Proceedings ofthe 17th International Conference on Machine Learning and Computing.

Springer Nature Switzerland, Cham, Switzerland, 194–206. doi:10.1007/978-3- 031-94898-5\_15

[21] Yao-Hung Hubert Tsai, Paul Pu Liang, Amir Zadeh, Louis-Philippe Morency, and Ruslan Salakhutdinov. 2019. Learning Factorized Multimodal Representations. In International Conference on Learning Representations. OpenReview.net, New Orleans, LA, USA.

[22] Gemine Vivone, Luciano Alparone, Jocelyn Chanussot, Mauro Dalla Mura, Andrea Garzelli, Giorgio Antonio Licciardi, Rocco Restaino, and Lucien Wald. 2015. A Critical Comparison Among Pansharpening Algorithms. IEEE Transactions on Geoscience and Remote Sensing 53, 5 (2015), 2565–2586.

[23] Chunshui Wang and Heng Zhang. 2026. FMFR: Feature-Level Multi-Stage Fusion and Remapping for Multi-Modal Industrial Anomaly Detection. Journal ofComputational Design and Engineering 13, 3 (2026), 233–245. doi:10.1093/jcde/qwag016

[24] Hebaixu Wang, Hao Zhang, Xin Tian, and Jiayi Ma. 2024. Zero-Sharpen: A Universal Pansharpening Method Across Satellites for Reducing Scale-Variance Gap via Zero-Shot Variation. Information Fusion 101 (2024), 102003. doi:10.1016/ j.infus.2023.102003

[25] Mengyu Wang, Zhenyu Liu, Kun Li, Yu Wang, Yuwei Wang, Yanyan Wei, and Fei Wang. 2026. Task-Generalized Adaptive Cross-Domain Learning for Multimodal Image Fusion. IEEE Transactions on Multimedia 28 (2026), 4624–4637. doi:10. 1109/TMM.2026.3660142

[26] Han Xu, Jiayi Ma, Junjun Jiang, Xiaojie Guo, and Haibin Ling. 2022. U2Fusion: A Unified Unsupervised Image Fusion Network. IEEE Transactions on Pattern Analysis and Machine Intelligence 44, 1 (2022), 502–518.

[27] Xuewen Yang, Dongliang Xie, and Xin Wang. 2018. Crossing-Domain Generative Adversarial Networks for Unsupervised Multi-Domain Image-to-Image Translation. In Proceedings of the 26th ACM International Conference on Multimedia. Association for Computing Machinery, New York, NY, USA, 374–382. doi:10.1145/3240508.3240716

[28] Xunpeng Yi, Han Xu, Hao Zhang, Linfeng Tang, and Jiayi Ma. 2024. Text-IF: Leveraging Semantic Text Guidance for Degradation-Aware and Interactive Image Fusion. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition. IEEE, Los Alamitos, CA, USA, 27026–27035.

[29] Xu Yuan, Ange Qi, Huinan Wu, Jiaqiang Wang, Yi Guo, Shijin Li, and Liang Zhao. 2025. Cross-Modal Feature Alignment and Fusion with Contrastive Learning in Multimodal Recommendation. Knowledge-Based Systems 326 (2025), 114020.

[30] Hao Zhang, Zhuliang Le, Zhenfeng Shao, Han Xu, and Jiayi Ma. 2021. MFF-GAN: An Unsupervised Generative Adversarial Network with Adaptive and Gradient Joint Constraints for Multi-Focus Image Fusion. Information Fusion 66 (2021), 40–53. doi:10.1016/j.infus.2020.08.022

[31] Hao Zhang and Jiayi Ma. 2021. SDNet: A Versatile Squeeze-and-Decomposition Network for Real-Time Image Fusion. International Journal ofComputer Vision 129, 10 (2021), 2761–2785. doi:10.1007/s11263-021-01501-8

[32] Hao Zhang, Hebaixu Wang, Xin Tian, and Jiayi Ma. 2023. P2Sharpen: A Progressive Pansharpening Network with Deep Spectral Transformation. Information Fusion 91 (2023), 103–122. doi:10.1016/j.infus.2022.10.010

[33] Ming Zhang, Ke Chang, and Yunfang Wu. 2024. Multi-Modal Semantic Understanding with Contrastive Cross-Modal Feature Alignment. In Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation. ELRA and ICCL, Torino, Italia, 11934–11943.

[34] Minghui Zhao, Hongxiang Gao, Xinru Qi, Haiyan Yin, Yulei Song, Yamei Bai, Jianqing Li, Lulu Zhao, and Chengyu Liu. 2025. Multi-Query Cross-Modal Attention Fusion for Cognitive Impairment Recognition. IEEE Transactions on Neural Systems and Rehabilitation Engineering 33 (2025), 2520–2530. doi:10.1109/TNSRE.2025.3583110

[35] Yijing Zhao, Yuchao Xia, Yi Ding, Yumeng Liu, Shuai Liu, and Hongan Wang. 2024. S2F-Net: Shared-Specific Fusion Network for Infrared and Visible Image Fusion. In Proceedings ofthe 2024 International Conference on Multimedia Retrieval. Association for Computing Machinery, New York, NY, USA, 497–505. doi:10. 1145/3652583.3658100

[36] Zixiang Zhao, Haowen Bai, Jiangshe Zhang, Yulun Zhang, Shuang Xu, Zudi Lin, Radu Timofte, and Luc Van Gool. 2023. CDDFuse: Correlation-Driven Dual Branch Feature Decomposition for Multi-Modality Image Fusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. IEEE, Los Alamitos, CA, USA, 5906–5916.

[37] Zixiang Zhao, Haowen Bai, Yuanzhi Zhu, Jiangshe Zhang, Shuang Xu, Yulun Zhang, Kai Zhang, Deyu Meng, Radu Timofte, and Luc Van Gool. 2023. DDFM: Denoising Difusion Model for Multi-Modality Image Fusion. In Proceedings of the IEEE/CVF International Conference on Computer Vision. IEEE, Los Alamitos, CA, USA, 8082–8093.

[38] Yongshuo Zong, Oisin Mac Aodha, and Timothy M. Hospedales. 2025. Self-Supervised Multimodal Learning: A Survey. IEEE Transactions on Pattern Analysis and Machine Intelligence 47, 7 (2025), 5299–5318. doi:10.1109/TPAMI.2024.3429301