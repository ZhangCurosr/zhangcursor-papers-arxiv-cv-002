# StableVQ: Practical Guidelines for Stable Vector-Quantized Tokenizer Training

Bao Tang<sup>1,2,§</sup> Jiahao Guo<sup>1,2,§</sup> Haoxiang Cao<sup>2,3,§</sup> Wenyu Liu<sup>1</sup> Changqian Yu<sup>2,†</sup> Kun Gai<sup>2</sup> Xinggang Wang<sup>1,†</sup>

<sup>1</sup>Huazhong University of Science and Technology <sup>2</sup>KlingAI Research <sup>3</sup>South China Normal University

## Abstract

Vector Quantization (VQ) is fundamental to discrete visual tokenizers that power modern autoregressive and masked image generation models. While recent sharedprojection codebook methods have substantially advanced codebook utilization, training stability remains a critical and underexplored challenge. We argue that the root cause lies in the entanglement of the Encoder–Decoder and Codebook training: because neither module can reliably fulfill its own responsibility in isolation, the system can only function when the two subsystems happen to cooperate—a fragile condition that breaks down precisely when training is most stressed. We propose StableVQ, which revisits the proper learning objective of each module and resolves the problems that arise when each is trained to fulfill its own role independently. Concretely, (1) Dynamic STE corrects the instability in the Encoder’s learning objective, enabling it to robustly optimize the reconstruction space under discrete regularization even when codebook utilization is low. (2) Region VQ Loss reconceives the Codebook’s learning objective so that it can independently guarantee full tracking of the encoder output distribution, without relying on encoder oscillations to drive activation. (3) Decoupled Schedule recognizes that the distinct responsibilities of the Encoder–Decoder and the Codebook demand distinct optimization dynamics, and assigns each an independent learning rate schedule to ensure robust system-level behavior. Built on top of shared-projection codebooks, StableVQ is lightweight and introduces no learnable parameters. Experiments on ImageNet demonstrate consistent improvements in training stability, codebook utilization, and reconstruction quality across diverse codebook sizes and initialization settings.

## 1 Introduction

Visual tokenization has become a foundational component of modern generative vision systems [30, 6, 27, 28]. By mapping continuous image features to sequences of discrete tokens via a learned codebook, VQ-VAEs [30] enable autoregressive transformers [27, 28, 32], masked generative models [1], and multimodal language models to operate over compact, structured visual representations. The expressiveness of the resulting vocabulary directly determines the upper bound on downstream generation quality.

A persistent obstacle in VQ training is codebook collapse, where most code vectors are never assigned, severely underutilizing model capacity. Recent shared-projection methods [11, 36, 2] address this by reparameterizing codebook entries as $\tilde { \mathbf { e } } _ { k } = f _ { \theta } ( \mathbf { e } _ { k } )$ via a shared differentiable function, so that gradients propagate across the entire code distribution. This reframes VQ training as a distribution alignment problem between the projected code distribution and the encoder output distribution, substantially advancing codebook utilization.

Yet training stability remains a critical and underexplored challenge. In practice, convergence is sensitive to initialization; codebooks may stagnate in low-utilization phases; and abrupt utilization collapse can occur mid-training, particularly at scale. We argue that these failure modes are not incidental but symptomatic of a deeper structural issue: the Encoder–Decoder and the Codebook are entangled in their training, such that neither can reliably fulfill its own responsibility in isolation. This entanglement masks the latent dysfunction of each module, leaving underlying issues unresolved and making the overall system contingent on fragile inter-module cooperation rather than principled individual competence. In Section 4.1, we provide a principled analysis of the latent problems in existing VQ tokenizer training that this entanglement conceals.

We propose StableVQ, a set of lightweight interventions that enables principled, stable VQ training without relying on fragile inter-module cooperation. Our contributions are as follows:

• A separation-of-concerns analysis of VQ tokenizer training. We revisit the proper responsibility of each module in VQ training and show that inter-module entanglement has long concealed latent dysfunctions in each. This analysis reframes training instability as a failure of modular responsibility rather than a fundamental limitation of the quantization paradigm.

• StableVQ: principled and lightweight interventions for stable training. Guided by the above analysis, we propose three targeted, parameter-free components that enable each module to fulfill its own responsibility independently. Together, they achieve principled, stable VQ training across diverse codebook sizes and initialization settings, without relying on heuristic design choices.

• A more accessible performance ceiling for VQ tokenizers. By eliminating the dependence on heuristic initialization and shared-projection architecture design, StableVQ allows the full expressive potential of VQ tokenizers to be realized without optimization artifacts standing in the way. A single linear projection suffices to reach state-of-the-art quality across diverse settings, and the principled modular stability established by our framework offers a solid theoretical footing for extending VQ training reliably to more demanding scenarios.

## 2 Related Work

We provide a brief overview of related work here, with a more comprehensive version in Appendix A. Vector-quantized representation learning was introduced by VQ-VAE [30], which maps continuous encoder features to discrete code indices through nearest-neighbor lookup in a learned codebook. Subsequent tokenizers improve reconstruction quality and token capacity through hierarchical latents [22], perceptual and adversarial objectives [6], residual or multi-stage quantization [13], and ViT-based architectures [31]. A central challenge in these systems is codebook collapse and low utilization, especially as codebook size or embedding dimension increases. Existing remedies include code reset or replacement [34], low-dimensional embeddings and normalization [31], relaxed or soft-assignment paths [21, 12, 25], and scalar or binary quantization alternatives [18, 32]. More recent shared-projection methods such as VQ-STE++, SimVQ, and FVQ [11, 36, 2] reparameterize code vectors through a shared function, substantially improving utilization with little architectural overhead. StableVQ builds on this line and studies the remaining training-instability problem.

## 3 Background

## 3.1 Vector Quantization

Given an encoder E and a decoder D, a VQ-VAE [30] maps an input image x to a spatial feature map $\mathbf { z } = E ( \mathbf { x } ) \in \mathbb { R } ^ { H \times W \times d }$ . Each spatial feature $\mathbf { z } _ { i j } \in \mathbb { R } ^ { d }$ is quantized by nearest-neighbor lookup in a codebook $\mathcal { C } = \{ \mathbf { e } _ { k } \} _ { k = 1 } ^ { K }$

$$
k _ { i j } ^ { * } = \arg \operatorname* { m i n } _ { k \in [ K ] } \| \mathbf { z } _ { i j } - \mathbf { e } _ { k } \| _ { 2 } ^ { 2 } , \qquad \hat { \mathbf { z } } _ { i j } = \mathbf { e } _ { k _ { i j } ^ { * } } .\tag{1}
$$

The training objective decomposes into three terms:

$$
\mathcal { L } = \mathcal { L } _ { \mathrm { r e c o n } } + \beta \underbrace { \| \mathbf { z } - \mathbf { s g } [ \hat { \mathbf { z } } ] \| ^ { 2 } } _ { \mathrm { c o m m i t m e n t ~ l o s s } } + \underbrace { \| \mathbf { s g } [ \mathbf { z } ] - \hat { \mathbf { z } } \| ^ { 2 } } _ { \mathrm { V Q ~ l o s s } } ,\tag{2}
$$

where $\mathrm { s g } [ \cdot ]$ is the stop-gradient operator. The commitment loss constrains encoder outputs to remain near their assigned codes; the VQ loss drives codebook entries toward the encoder output distribution; and the end-to-end gradient from the reconstruction loss is passed back to the Encoder via the Straight-Through Estimator (STE), which approximates $\partial \mathcal { L } / \partial \bar { \mathbf { z } } _ { i j } \approx \partial \mathcal { L } / \partial \hat { \mathbf { z } } _ { i j }$

## 3.2 Codebook with a Shared Projection

In the standard formulation, each code $\mathbf { e } _ { k }$ is an independent parameter, so the VQ loss gradient is nonzero only for selected codes—a property we term gradient sparsity. As codebook size K grows, the fraction of codes updated per step diminishes, exacerbating collapse risk.

$\mathbf { A }$ recent line of work addresses gradient sparsity via a shared projection: codes are reparameterized as $\tilde { \mathbf { e } } _ { k } = f _ { \theta } ( \mathbf { e } _ { k } )$ , where $f _ { \theta }$ is a differentiable function shared across all codes:

$$
k _ { i j } ^ { * } = \arg \operatorname* { m i n } _ { k \in \left[ K \right] } \| \mathbf { z } _ { i j } - f _ { \theta } ( \mathbf { e } _ { k } ) \| _ { 2 } ^ { 2 } .\tag{3}
$$

Because $f _ { \theta }$ is shared, gradients from any selected code propagate to influence all $\{ \mathbf { e } _ { k } \}$ , transforming VQ training into an alignment problem between the projected code distribution $\mathcal { \bar { P } } \mathscr { c } = \{ f _ { \theta } ( \mathbf { e } _ { k } ) \} _ { k = 1 } ^ { K }$ and the encoder output distribution $\mathcal { P } _ { \mathcal { Z } } = \{ E ( \mathbf { z } _ { i j } ) \}$

Representative instantiations of this paradigm include the affine reparameterization approach of [11], which applies a shared learnable scale-and-shift transformation to the code vectors, rescaling each quantized embedding as $q = c _ { \mathrm { m e a n } } + c _ { \mathrm { s t d } } \odot \hat { q }$ where $c _ { \mathrm { m e a n } }$ and $c _ { \mathrm { s t d } }$ are codebook-wide affine parameters; SimVQ [36], which defines $f _ { \theta }$ as a learnable linear layer $W$ acting on a fixed latent basis, so that each code vector is produced as $c _ { k } = q _ { k } W$ , optimizing the entire linear space spanned by the codebook rather than individual code vectors; and FVQ [2], which employs a more expressive nonlinear projector to remap code vectors, enabling full codebook utilization. These methods substantially improve codebook utilization, but training stability remains unsolved, as shown below.

## 4 Method

## 4.1 Failure Modes of VQ Training

Limitations of shared-projection methods. Although shared-projection methods substantially mitigate gradient sparsity, they do not fully resolve the challenge of codebook distribution alignment. First, gradient propagation through $f _ { \theta }$ influences all codes indirectly: the signal reaching inactive codes is diffuse and undirected, providing no guarantee that they converge toward the correct regions of the token distribution. Second, this indirect influence is subject to decay: once a small subset of codes covers the token distribution well enough to minimize the VQ loss, the training signal driving the remaining codes becomes negligible. Furthermore, shared-projection methods operate purely on the codebook side and do not account for potential instabilities in the Encoder’s optimization during training—a separate source of fragility that can compound with codebook misalignment and destabilize the system as a whole.

Challenging failure modes. Wasserstein VQ [7] analyzes static relationships between token and code distributions. We further examine characteristic failure modes observed during training, analyzing how the token–code distributional relationship evolves in each case and what drives this evolution (Figure 1). (a) Code scale $\ll$ Token scale: when the code distribution occupies a smaller scale region than the token distribution, the Codebook receives only sparse optimization targets due to low utilization, while the token distribution fluctuates unpredictably under the competing gradients of the STE-passed reconstruction signal and the commitment loss. The system thus falls into prolonged low utilization; when these fluctuations are severe enough, commitment loss spikes can escalate to NaN gradients before the codebook ever reaches meaningful utilization. (b) Code scale $\gg$ Token scale: when the code distribution spans a much larger region than the token distribution, utilization rises rapidly as codes within the token scale region are quickly activated. However, as more in-range codes are claimed, the commitment loss and VQ loss signals become increasingly saturated, rapidly diminishing the training signal for out-of-range codes. This leaves the majority of the codebook virtually unreachable by nearest-neighbor assignment, resulting in permanent dead codes. (c) Scale divergence: even in a well-utilized codebook, the two distributions are in continuous dynamic alignment as the reconstruction loss drives the token distribution to evolve. Should the

(a) Code Scale << Token Scale  
![](images/ef7b6f05713a002c416676ca2d9ec629258ae335719bd4796a4e956fde22bc03.jpg)  
Token Distribution Code Distribution

(b) Code Scale >> Token Scale  
![](images/5f0a1671babc1f283e655936e1869db449d630e1194151d0c3bedad268bb418e.jpg)  
Token Distribution Code Distribution

(c) Scale Divergence  
![](images/fba09421613e19cf8289ee7e18727bef9f1400178896fc207e074833433ba30d.jpg)  
Token Distribution Code Distribution

![](images/d6937244daa1ae5bc5e3d318c62a6c16551b541cd3c78dd0ede2578cc0c9da69.jpg)

![](images/008106ecd7ed0c52e0829157b582958cd1aa4f1d1d7abc2a475eb21e3ae33000.jpg)

![](images/22ef0ee32a2414efd42560bf5a128eae14eb6d7d67dc81accc06f227dfd52b00.jpg)  
Figure 1: Three characteristic failure modes of VQ training, illustrated through the relationship between the token distribution (blue) and code distribution (red) at different training stages.

Codebook momentarily fail to track a sudden distributional shift, the erroneous STE gradients passed to the Encoder tend to amplify the divergence rather than correct it, triggering a positive-feedback loop that rapidly escalates the scale mismatch and causes utilization collapse instantaneously.

The failure modes described above share a common root cause: the Encoder and Codebook are not given the conditions to independently fulfill their own responsibilities. Guided by the principle of separation of concerns, we analyze the proper learning objective of each module and identify where the current training pipeline prevents each from fulfilling its own role.

## 4.2 Dynamic Straight-Through Estimator

Gradient Estimation Gap. The Encoder’s proper learning objective is to optimize the reconstruction space under the discrete code constraint imposed by the commitment loss. Ideally, the commitment loss and the STE-passed reconstruction gradient should cooperate toward this objective. However, when tokens are assigned to distant codes, the STE gradient becomes an unreliable estimate of the true reconstruction direction. Such unreliable gradients can conflict with the commitment loss, push the token and code distributions further apart, and trigger commitment-loss spikes that may escalate to NaN values. By amplifying distributional errors rather than correcting them, they also become a primary driver of sudden utilization collapse, undermining overall training robustness.

Gradient Quality Weighting. The key observation is that when multiple tokens hit the same code, the relativelyfarther tokens are those whose STE gradients are most unreliable. We therefore assign each token a gradient weight based on its distance to the assigned code relative to the best-matched token for that code:

$$
w _ { i j } = \mathrm { s g } \left[ \frac { d _ { k _ { i j } ^ { * } } ^ { * } } { \lVert \mathbf { z } _ { i j } - f _ { \theta } ( \mathbf { e } _ { k _ { i j } ^ { * } } ) \rVert _ { 2 } ^ { 2 } } \right] \in ( 0 , 1 ] , \quad d _ { k } ^ { * } = \operatorname* { m i n } _ { ( i ^ { \prime } , j ^ { \prime } ) } \lVert \mathbf { z } _ { i ^ { \prime } j ^ { \prime } } - f _ { \theta } ( \mathbf { e } _ { k } ) \rVert _ { 2 } ^ { 2 } ,\tag{4}
$$

where $d _ { k } ^ { * }$ is the minimum squared distance from code k to any token in the current batch. When a token is the best-matched token for its assigned code, it receives the full gradient with $w _ { i j } = 1$ . As its relative quantization error grows, the weight decreases accordingly.

We modulate the STE by this per-token weight:

$$
\hat { \mathbf { z } } _ { i j } = w _ { i j } \cdot \mathbf { z } _ { i j } + \mathrm { s g } [ \hat { \mathbf { z } } _ { i j } - w _ { i j } \cdot \mathbf { z } _ { i j } ] .\tag{5}
$$

This formulation attenuates the STE gradient for relatively distant tokens while preserving the full gradient for the token with the best assignment. Crucially, when all tokens in a batch are assigned to well-matched codes (high utilization, stable training), $w _ { i j } \approx 1$ for all tokens and Dynamic STE reduces to the standard STE. The intervention is thus self-deactivating under healthy training conditions, and no threshold hyperparameter is required.

![](images/a9d4604a96a70e920d7575f1c714a82ddbc5ec35024c59b77ea132e080990d4d.jpg)

![](images/e395c538f9a9dd0dc633964e58e8e767bc988da1626f421c94cc25fc72f0371f.jpg)  
Figure 2: Left: Dynamic STE design. Relatively farther tokens have more unreliable gradients and receive proportionally suppressed contributions. Right: Pilot Study. Standard STE causes commitment loss spikes and training collapse, whereas Dynamic STE maintains stable losses.

Pilot Study. To verify the effect of Dynamic STE on the Encoder’s learning objective, we freeze the Codebook and train only the Encoder–Decoder. Figure 2 shows that standard STE leads to severe commitment-loss spikes and reconstruction-loss oscillations, ultimately causing losses to diverge to NaN, while Dynamic STE suppresses these instabilities and keeps losses stable throughout training. This confirms that the gradient estimation gap introduces latent instability into Encoder training.

## 4.3 Region VQ Loss

Unguaranteed Distribution Alignment. The Codebook’s proper learning objective is to track the encoder output distribution through a clean and independent optimization process. In the ideal setting, this should endow the Codebook with the ability to guarantee codebook utilization on its own, without relying on assistance from the Encoder. However, the standard VQ loss only provides direct learning targets to the codes selected in the current step, leaving the majority of codes without explicit supervision. Although shared-projection methods allow gradients to reach all codes through $f _ { \theta } ,$ the resulting signal remains indirect and undirected, and is inherently subject to attenuation during training. Consequently, the Codebook cannot independently guarantee full utilization; instead, activation of the remaining codes often depends on unstable fluctuations in the encoder output, meaning that codebook utilization is not reliably guaranteed in practice.

Asymmetry in VQ Loss. The standard VQ loss is inherently asymmetric: every token receives an explicit target through the commitment loss, whereas only selected codes are assigned meaningful objectives. Letting all codes take the nearest token as their target seems a natural remedy, yet this often fails to provide correct learning directions. The solution lies in shifting from point-wise to distribution-wise alignment. Since asymmetry arises from many tokens selecting few codes, we propagate the targets received by active codes to nearby inactive ones, which we term Region $\mathrm { v Q }$

Code Target Assignment. Let $[ K ] = \{ 1 , \ldots , K \}$ denote all code indices and $S _ { t } \subseteq [ K ]$ denote the set of codes selected at step t. To distinguish codes by activity, we maintain a FIFO queue of length W, which defines the window-active set $\begin{array} { r } { \mathcal { A } _ { t } = \bigcup _ { s = t - W + 1 } ^ { t } \mathcal { S } _ { s } } \end{array}$ and the persistently inactive set $\mathcal { N } _ { t } = [ K ] \setminus \mathcal { A } _ { t }$ . Let $\mathcal { U } _ { k }$ contain the token features $\mathbf { z } _ { u }$ assigned to code $k ,$ , with $n _ { k } = | U _ { k } |$ . Each source code $k \in S _ { t }$ receives a quota $q _ { k } \propto n _ { k }$ and propagates its target to the $q _ { k }$ nearest codes in $\mathcal { N } _ { t }$ denoted $\mathcal { R } _ { k }$ (Algorithm 1). Let ${ \mathcal { K } } _ { j } = \{ k \in { \mathcal { S } } _ { t } : j \in { \mathcal { R } } _ { k } \}$ } denote the sources propagating to code j. Recently active codes $\mathbf { \mathcal { A } } _ { t } \setminus \mathcal { S } _ { t }$ and unclaimed inactive codes $\{ j \in \mathcal { N } _ { t } : | \mathcal { K } _ { j } | \stackrel { - } { = } 0 \}$ retain self-targets and yield zero loss, while other effective code targets are defined as follows:

$$
\mathbf { t } _ { j } = \left\{ \begin{array} { l l } { \displaystyle \frac { 1 } { | \mathcal { U } _ { j } | } \sum _ { u \in \mathcal { U } _ { j } } \mathbf { z } _ { u } , } & { j \in \mathcal { S } _ { t } , } \\ { \displaystyle \frac { 1 } { | \mathcal { K } _ { j } | } \sum _ { k \in \mathcal { K } _ { j } } \mathbf { t } _ { k } , } & { j \in \mathcal { N } _ { t } , \ | \mathcal { K } _ { j } | > 0 . } \end{array} \right.\tag{6}
$$

![](images/1b9f10dd9fe99a2d97e99440822e0d2a33cfa72e21dd30214a3018bb4eb94cfe.jpg)  
Figure 3: Left: Region VQ Loss. Active codes propagate targets to nearby inactive codes proportionally to selection count. Right: Pilot Study. T-SNE visualizations at different training steps (blue: encoder outputs; red: codebook entries) with codebook utilization.

Let $\mathcal { M } _ { t } = \mathcal { S } _ { t } \cup \{ j \in \mathcal { N } _ { t } : | \mathcal { K } _ { j } | > 0 \}$ denote codes with effective targets. The unified codebook update loss is

$$
\mathcal { L } _ { \mathrm { c o d e } } = \frac { 1 } { | \mathcal { M } _ { t } | } \sum _ { k \in \mathcal { M } _ { t } } \| f _ { \theta } ( \mathbf { e } _ { k } ) - \mathrm { s g } [ \mathbf { t } _ { k } ] \| _ { 2 } ^ { 2 } .\tag{7}
$$

Pilot Study. To isolate the effect of Region VQ Loss on the Codebook’s learning objective, we freeze the Encoder and optimize only the Codebook, which uses a two-layer ViT Block as the shared projector to ensure sufficient learning capacity. Figure 3 shows T-SNE visualizations of the encoder output distribution and codebook entries at Steps 0, 500, and 5000. The standard VQ loss stagnates at around 12.5% utilization even after 5000 steps, whereas Region VQ Loss reaches full utilization by Step 500 and maintains it throughout training. This confirms that the unguaranteed distribution alignment problem is intrinsic to the VQ loss objective, and that Region VQ Loss directly resolves it by providing every code with a principled learning target.

## 4.4 Decoupled Schedule

Coupled Optimization. The Encoder–Decoder and the Codebook have fundamentally different optimization characteristics and should be governed by independent learning rate schedules. The Encoder benefits from warmup-plus-annealing to stabilize its complex multi-objective landscape. The Codebook, whose task is to continuously track the evolving encoder distribution, requires sustained high learning rates especially during the early phase when the encoder output distribution changes most rapidly. Coupling both under either a constant learning rate or a warmup-plus-annealing schedule leads to suboptimal performance or reduced codebook utilization.

Objective-Driven Schedule Decoupling. Prior works [11, 2] have shown that a warmup-plusannealing learning rate schedule benefits VQ training quality, yet it often leads to degraded codebook utilization. To compensate, FVQ introduces a more expressive shared projector to maintain utilization under this schedule. Our preceding analysis reveals that this tension stems from a more fundamental issue: the Encoder–Decoder and the Codebook have inherently different learning objectives, and therefore require distinct optimization schedules. We treat them as two independent optimization systems, each scheduled according to its own objective:

• Encoder–Decoder is responsible for reconstruction under discrete regularization, a complex multi-objective task that benefits from warmup-plus-annealing to stabilize early optimization and ensure smooth convergence.

![](images/974fb2149e20d0005f3ef23856a54bbeb10579039fdbe6ffaf2158e611d50522.jpg)

![](images/f731375f9cbfe98fa3572cf0422eea9e03740f955b40cc9cdd7a038a0d082d82.jpg)  
Figure 4: Pilot study on learning rate schedules for different modules. WU-AN denotes Warmupplus-Annealing; C.B. denotes Codebook.

• Codebook is responsible for continuously tracking the encoder output distribution, a clean and dedicated objective for which a constant high learning rate with no warmup may be most beneficial, ensuring adequate gradient magnitude from the very first step.

Pilot Study. We conduct two controlled experiments to verify this design (Figure 4). Using the FVQ architecture (left), we ablate which module benefits from warmup-plus-annealing: setting the Codebook to a constant learning rate (red) incurs no performance degradation, whereas applying a constant rate to the Encoder–Decoder leads to a clear quality drop. Using the SimVQ architecture (right), we examine what schedule the Codebook requires: utilization is not improved by complex schedules, but benefits from a stable and sufficiently high constant learning rate.

## 5 Experiments

## 5.1 Main Results

Setup. We evaluate StableVQ on ImageNet [4] at 256 × 256 resolution using a VQGAN-style [6] encoder–decoder with downsampling factor f = 16, producing 16 × 16 = 256 tokens per image. We evaluate reconstruction quality by rFID and LPIPS, and codebook utilization as the fraction of activated codes, on the ImageNet validation set. We compare against a range of baselines, with particular focus on shared-projection methods SimVQ [36] and FVQ [2]. For these baselines, we re-implement their results following their respective original configurations, and evaluate using on-the-fly reconstruction rather than a save-then-reload pipeline to ensure fair and accurate metric computation. StableVQ uses the simplest single-layer linear shared projector by default.

Reconstruction results. Table 1 presents reconstruction results. While SimVQ and FVQ both achieve 100% utilization under their respective standard configurations, each comes with notable limitations. SimVQ adopts a constant learning rate to sustain full utilization, but the lack of annealing results in substantially degraded reconstruction quality. FVQ relies on a carefully engineered projector architecture—including ViT block depth and patch size—to maintain utilization under warmup-plusannealing, and the patch embedding operation constrains the codebook size to perfect squares.

StableVQ achieves superior reconstruction quality with a single linear projection layer, matching or surpassing methods that rely on more complex projectors. It requires no projector-specific design and imposes no structural constraints, while maintaining full utilization across a broader range of challenging scenarios as the ablation studies demonstrate. When VQ training stability is no longer the bottleneck, the optimization strategy becomes the dominant factor in reconstruction quality; StableVQ uses a discriminator following prior works [24, 33] as its adversarial supervision.

Robustness test. To further test the stability of different methods, we introduce UR-AUC in Table 1, which denotes Usage Recovery AUC. It measures codebook utilization recovery under codebooktoken distribution mismatch and is computed as the average AUC of codebook usage curves over the tested mismatch settings (Appendix E). As visualized in Figure 5, SimVQ and FVQ recover usage slowly and only under limited mismatch conditions, whereas StableVQ rapidly restores full codebook usage across diverse codebook-token distribution relationships. This indicates that StableVQ avoids prolonged low utilization and dead-code issues under different training conditions, providing strong robustness guarantees for scaling VQ training and applying it to broader scenarios.

Table 1: Reconstruction results on ImageNet 256 × 256 with $1 6 \times 1 6$ tokens. <sup>†</sup> denotes a larger encoder–decoder. UR-AUC denotes Usage Recovery AUC under the robustness test.
<table><tr><td>Method</td><td>Projector</td><td>Epochs</td><td>Codebook Size (n × d)</td><td>rFID↓</td><td>LPIPS↓</td><td>Usage↑</td><td>UR-AUC↑</td></tr><tr><td>LlamaGen [27]</td><td></td><td>40</td><td>16,384 × 8</td><td>2.19</td><td>0.2281</td><td>97%</td><td></td></tr><tr><td>LlamaGen [27]</td><td></td><td>40</td><td> $1 6 { , } 3 8 4 \times 2 5 6$ </td><td>9.21</td><td></td><td>0.29%</td><td></td></tr><tr><td>IBQ† [25]</td><td></td><td>330</td><td> $1 6 { , } 3 8 4 \times 2 5 6$ </td><td>1.37</td><td>0.2235</td><td>96%</td><td></td></tr><tr><td>IBQ† [25]</td><td></td><td>330</td><td> $2 6 2 , 1 4 4 \times 2 5 6$ </td><td>1.00</td><td>0.2030</td><td>84%</td><td></td></tr><tr><td colspan="8">Shared-Projection-Based Methods</td></tr><tr><td>VQGAN-LC [35]</td><td>Linear-1</td><td>20</td><td> $1 6 { , } 3 8 4 \times 8$ </td><td>3.01</td><td>0.2358</td><td>99%</td><td></td></tr><tr><td>VQGAN-LC [35]</td><td>Linear-1</td><td>20</td><td> $1 0 0 { , } 0 0 0 \times 8$ </td><td>2.62</td><td>0.2212</td><td>99%</td><td></td></tr><tr><td>SimVQ [36]</td><td>Linear-1</td><td>40</td><td> $1 6 { , } 3 8 4 \times 2 5 6$ </td><td>2.89</td><td>0.2492</td><td>100%</td><td>2.17±0.32</td></tr><tr><td>SimVQ [36]</td><td>Linear-1</td><td>40</td><td> $2 6 2 , 1 4 4 \times 2 5 6$ </td><td>3.16</td><td>0.2516</td><td>100%</td><td></td></tr><tr><td>FVQ [2]</td><td>ViTBlock-2</td><td>40</td><td> $1 6 { , } 3 8 4 \times 2 5 6$ </td><td>1.70</td><td>0.2176</td><td>100%</td><td>8.08±0.24</td></tr><tr><td>FVQ [2]</td><td>ViTBlock-2</td><td>40</td><td> $2 6 2 , 1 4 4 \times 2 5 6$ </td><td>1.29</td><td>0.2003</td><td>100%</td><td></td></tr><tr><td>StableVQ</td><td>Linear-1</td><td>40</td><td> $1 6 { , } 3 8 4 \times 2 5 6$ </td><td>1.22</td><td>0.2235</td><td>100%</td><td></td></tr><tr><td>StableVQ</td><td>Linear-1</td><td>40</td><td> $2 6 2 , 1 4 4 \times 2 5 6$ </td><td>1.05</td><td>0.1947</td><td>100%</td><td>60.59±1.52</td></tr><tr><td>StableVQ</td><td>Linear-1</td><td>120</td><td> $1 6 { , } 3 8 4 \times 2 5 6$ </td><td>1.13</td><td>0.2134</td><td>100%</td><td></td></tr><tr><td>StableVQ</td><td>Linear-1</td><td>120</td><td> $2 6 2 , 1 4 4 \times 2 5 6$ </td><td>0.92</td><td>0.1893</td><td>100%</td><td></td></tr></table>

![](images/557c75b2dd09417ff2ae9a577d69642ecb52244bd25f24f53ba0ae35d9e182ab.jpg)  
Figure 5: Codebook usage recovery under different codebook-token distribution relationships.

Generation results. Following the IBQ [25] generation setup, we train class-conditional autoregressive transformers on StableVQ tokens. Table 2 shows competitive ImageNet 256 × 256 generation results, with complete baseline comparisons provided in Appendix C.

Table 2: Class-conditional image generation on ImageNet $2 5 6 \times 2 5 6$
<table><tr><td>Type</td><td>Tokenizer</td><td>Generator</td><td>Param.</td><td>FID↓</td><td>IS↑</td><td>Pre.↑</td><td>Rec.↑</td></tr><tr><td>Vanilla AR</td><td>IBQ [25]</td><td>IBQ-B [25]</td><td>342M</td><td>2.88</td><td>254.7</td><td>0.84</td><td>0.51</td></tr><tr><td>Vanilla AR</td><td>IBQ [25]</td><td>IBQ-L [25]</td><td>649M</td><td>2.45</td><td>267.5</td><td>0.83</td><td>0.52</td></tr><tr><td>Vanilla AR</td><td>StableVQ</td><td>IBQ-B [25]</td><td>342M</td><td>2.35</td><td>256.0</td><td>0.82</td><td>0.58</td></tr><tr><td>Vanilla AR</td><td>StableVQ</td><td>IBQ-L [25]</td><td>649M</td><td>2.18</td><td>250.4</td><td>0.82</td><td>0.59</td></tr></table>

## 5.2 Ablation Studies

The main results above are obtained under each method’s standard setting, where methods such as SimVQ and FVQ can also reach full utilization. However, these methods still do not resolve the fundamental deficiencies analyzed in Section 4. We therefore turn to more challenging yet common settings to evaluate training robustness.

Table 3: Ablation on proposed strategies under the codebook expansion setting.
<table><tr><td>Region VQ</td><td>Dyn. STE</td><td>De. Sch.</td><td>Peak Commit</td><td>Util.↑</td><td>rFID↓</td><td>LPIPS↓</td><td>PSNR↑</td><td>SSIM↑</td></tr><tr><td></td><td></td><td></td><td>53.11</td><td>49.13%</td><td>2.06</td><td>0.2297</td><td>21.65</td><td>0.5730</td></tr><tr><td>√</td><td></td><td></td><td>&gt;200 (NaN)</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>√</td><td></td><td>&lt;0.1</td><td>1.27%</td><td>6.68</td><td>0.3100</td><td>19.89</td><td>0.5026</td></tr><tr><td></td><td></td><td>√</td><td>&lt;0.1</td><td>83.77%</td><td>1.71</td><td>0.2209</td><td>21.79</td><td>0.5864</td></tr><tr><td></td><td>√</td><td>√</td><td>&lt;0.1</td><td>21.66%</td><td>2.07</td><td>0.2354</td><td>21.36</td><td>0.5665</td></tr><tr><td>√</td><td>√</td><td></td><td>&lt;0.1</td><td>100%</td><td>1.72</td><td>0.2200</td><td>21.84</td><td>0.5854</td></tr><tr><td>√</td><td></td><td>√</td><td>&lt;0.1</td><td>100%</td><td>1.75</td><td>0.2206</td><td>21.85</td><td>0.5860</td></tr><tr><td>√</td><td>√</td><td>√</td><td>&lt;0.1</td><td>100%</td><td>1.70</td><td>0.2208</td><td>21.81</td><td>0.5879</td></tr></table>

Table 4: Ablation on methods under the codebook shrinkage setting.
<table><tr><td>Projector</td><td>Region VQ</td><td>Init</td><td>Util.↑</td><td>rFID↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td></tr><tr><td>ViTBlock-2 [2]</td><td></td><td>uniform</td><td>18.75%</td><td>2.4355</td><td>20.9789</td><td>0.5677</td><td>0.2414</td></tr><tr><td>ViTBlock-2 [2]</td><td></td><td>gaussian</td><td>62.5%</td><td>1.9530</td><td>21.3774</td><td>0.5836</td><td>0.2273</td></tr><tr><td>ViTBlock-2 [2]</td><td>V</td><td>uniform</td><td>100%</td><td>1.8176</td><td>21.5497</td><td>0.5896</td><td>0.2210</td></tr><tr><td>ViTBlock-2 [2]</td><td>√</td><td>gaussian</td><td>100%</td><td>1.8966</td><td>21.4603</td><td>0.5931</td><td>0.2255</td></tr></table>

Ablation 1: Codebook expansion. We first set the experiment to the case of Figure 1(a), where the Codebook is initialized within a very small numerical range—the most common initialization trick in VQ training. In this setting, we use a single linear layer as the shared projector and adopt a warmup-plus-annealing schedule with peak learning rate 1e−4. We report peak commitment loss, codebook utilization, and reconstruction metrics, and evaluate different combinations of the strategies proposed in Section 4 on top of the baseline for a more complete analysis. When Decoupled Schedule is used, the Codebook learning rate is set to a constant 1e−3.

The results are consistent with the analysis in Section 4. Region VQ Loss alone causes NaN collapse because the Codebook still updates too slowly under the shared warmup schedule. Dynamic STE suppresses commitment-loss spikes and stabilizes the Encoder, but utilization remains very low, revealing that conventional training relies on unstable encoder oscillations to activate codes. Decoupled Schedule improves distribution tracking from the start, yet still falls short of full utilization under the standard VQ loss. Once Region VQ Loss is combined with either Dynamic STE or Decoupled Schedule, full utilization is recovered with strong reconstruction quality. Using all three components gives the most complete solution, jointly ensuring Encoder stability, dense Codebook supervision, and fast distribution tracking.

Ablation 2: Codebook shrinkage. We further set the experiment to the case of Figure 1(b), where the Codebook occupies a large range. In VQ training, this commonly arises when the space is constrained by $\ell _ { 2 }$ normalization: after normalization, either uniform or gaussian initialization distributes codes broadly across the space, whereas encoder outputs, reflecting the statistics of natural images, concentrate in a much smaller region. This naturally creates a codebook shrinkage scenario. In this experiment, we set the codebook dimension to 1024 and use a stronger ViTBlock shared projector, following the FVQ configuration. The results show that FVQ fails to achieve full utilization under either codebook initialization, and its utilization is strongly affected by initialization. Adding Region VQ consistently reaches full utilization and improves reconstruction quality regardless of initialization. This further supports that StableVQ provides a more principled solution, enabling ideal codebook usage across different VQ training conditions.

## 6 Conclusion

StableVQ shows that the long-standing instability of VQ training is not a limitation of vector quantization itself, but a consequence of entangled optimization objectives. By restoring separation of concerns between the Encoder–Decoder and the Codebook, StableVQ turns full codebook utilization from a fragile heuristic outcome into a principled property that can be directly guaranteed. We believe this provides an important foundation for extending VQ to broader scenarios and more challenging applications, where robust training is essential.

## Acknowledgments

This work was partially supported by the National Natural Science Foundation of China under Grant U25B2067.

## References

[1] Huiwen Chang, Han Zhang, Lu Jiang, Ce Liu, and William T Freeman. Maskgit: Masked generative image transformer. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pages 11315–11325, 2022.

[2] Yifan Chang, Jie Qin, Limeng Qiao, Xiaofeng Wang, Zheng Zhu, Lin Ma, and Xingang Wang. Scalable training for vector-quantized networks with 100% codebook utilization. arXiv preprint arXiv:2509.10140, 2025.

[3] Mark Chen, Alec Radford, Rewon Child, Jeffrey Wu, Heewoo Jun, David Luan, and Ilya Sutskever. Generative pretraining from pixels. In International conference on machine learning, pages 1691–1703. PMLR, 2020.

[4] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A largescale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pages 248–255. Ieee, 2009.

[5] Sinan Du, Jiahao Guo, Bo Li, Shuhao Cui, Zhengzhuo Xu, Yifu Luo, Yongxian Wei, Kun Gai, Xinggang Wang, Kai Wu, et al. Vqrae: Representation quantization autoencoders for multimodal understanding, generation and reconstruction. arXiv preprint arXiv:2511.23386, 2025.

[6] Patrick Esser, Robin Rombach, and Bjorn Ommer. Taming transformers for high-resolution image synthesis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 12873–12883, 2021.

[7] Xianghong Fang, Litao Guo, Hengchao Chen, Yuxuan Zhang, Xiaofan Xia, Dingjie Song, Yexin Liu, Hao Wang, Harry Yang, Yuan Yuan, and Qiang Sun. Enhancing vector quantization with distributional matching: A theoretical and empirical study. arXiv preprint arXiv:2506.15078, 2025.

[8] Xianghong Fang, Yuan Yuan, Dehan Kong, and Tim G. J. Rudner. Vq-transplant: Efficient vqmodule integration for pre-trained visual tokenizers. In International Conference on Learning Representations, 2026.

[9] Christopher Fifty, Ronald G. Junkins, Dennis Duan, Aniketh Iyengar, Jerry W. Liu, Ehsan Amid, Sebastian Thrun, and Christopher Ré. Restructuring vector quantization with the rotation trick. In International Conference on Learning Representations, 2025.

[10] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.

[11] Minyoung Huh, Brian Cheung, Pulkit Agrawal, and Phillip Isola. Straightening out the straightthrough estimator: Overcoming optimization challenges in vector quantized networks. In International Conference on Machine Learning, pages 14096–14113. PMLR, 2023.

[12] Eric Jang, Shixiang Gu, and Ben Poole. Categorical reparameterization with gumbel-softmax. arXiv preprint arXiv:1611.01144, 2016.

[13] Doyup Lee, Chiheon Kim, Saehoon Kim, Minsu Cho, and Wook-Shin Han. Autoregressive image generation using residual quantization. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pages 11523–11532, 2022.

[14] Tianhong Li, Huiwen Chang, Shlok Mishra, Han Zhang, Dina Katabi, and Dilip Krishnan. Mage: Masked generative encoder to unify representation learning and image synthesis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 2142–2152, 2023.

[15] Tianhong Li, Yonglong Tian, He Li, Mingyang Deng, and Kaiming He. Autoregressive image generation without vector quantization. Advances in Neural Information Processing Systems, 37:56424–56445, 2024.

[16] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022.

[17] Zhuoyan Luo, Fengyuan Shi, Yixiao Ge, Yujiu Yang, Limin Wang, and Ying Shan. Openmagvit2: An open-source project toward democratizing auto-regressive visual generation. arXiv preprint arXiv:2409.04410, 2024.

[18] Fabian Mentzer, David Minnen, Eirikur Agustsson, and Michael Tschannen. Finite scalar quantization: Vq-vae made simple. arXiv preprint arXiv:2309.15505, 2023.

[19] Ziqi Pang, Tianyuan Zhang, Fujun Luan, Yunze Man, Hao Tan, Kai Zhang, William T Freeman, and Yu-Xiong Wang. Randar: Decoder-only autoregressive visual generation in random orders. In Proceedings ofthe Computer Vision and Pattern Recognition Conference, pages 45–55, 2025.

[20] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4195–4205, 2023.

[21] Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. Zero-shot text-to-image generation. In International conference on machine learning, pages 8821–8831. Pmlr, 2021.

[22] Ali Razavi, Aaron Van den Oord, and Oriol Vinyals. Generating diverse high-fidelity images with vq-vae-2. Advances in neural information processing systems, 32, 2019.

[23] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. Highresolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022.

[24] Axel Sauer, Tero Karras, Samuli Laine, Andreas Geiger, and Timo Aila. Stylegan-t: Unlocking the power of gans for fast large-scale text-to-image synthesis. In International conference on machine learning, pages 30105–30118. PMLR, 2023.

[25] Fengyuan Shi, Zhuoyan Luo, Yixiao Ge, Yujiu Yang, Ying Shan, and Limin Wang. Scalable image tokenization with index backpropagation quantization. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pages 16037–16046, 2025.

[26] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456, 2020.

[27] Peize Sun, Yi Jiang, Shoufa Chen, Shilong Zhang, Bingyue Peng, Ping Luo, and Zehuan Yuan. Autoregressive model beats diffusion: Llama for scalable image generation. arXiv preprint arXiv:2406.06525, 2024.

[28] Keyu Tian, Yi Jiang, Zehuan Yuan, Bingyue Peng, and Liwei Wang. Visual autoregressive modeling: Scalable image generation via next-scale prediction. Advances in neural information processing systems, 37:84839–84865, 2024.

[29] Aaron Van den Oord, Nal Kalchbrenner, Lasse Espeholt, Oriol Vinyals, Alex Graves, et al. Conditional image generation with pixelcnn decoders. Advances in neural information processing systems, 29, 2016.

[30] Aaron Van Den Oord, Oriol Vinyals, et al. Neural discrete representation learning. Advances in neural information processing systems, 30, 2017.

[31] Jiahui Yu, Xin Li, Jing Yu Koh, Han Zhang, Ruoming Pang, James Qin, Alexander Ku, Yuanzhong Xu, Jason Baldridge, and Yonghui Wu. Vector-quantized image modeling with improved vqgan. arXiv preprint arXiv:2110.04627, 2021.

[32] Lijun Yu, José Lezama, Nitesh B Gundavarapu, Luca Versari, Kihyuk Sohn, David Minnen, Yong Cheng, Vighnesh Birodkar, Agrim Gupta, Xiuye Gu, et al. Language model beats diffusion–tokenizer is key to visual generation. arXiv preprint arXiv:2310.05737, 2023.

[33] Boyang Zheng, Nanye Ma, Shengbang Tong, and Saining Xie. Diffusion transformers with representation autoencoders. arXiv preprint arXiv:2510.11690, 2025.

[34] Chuanxia Zheng and Andrea Vedaldi. Online clustered codebook. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 22798–22807, 2023.

[35] Lei Zhu, Fangyun Wei, Yanye Lu, and Dong Chen. Scaling the codebook size of vq-gan to 100,000 with a utilization rate of 99%. Advances in Neural Information Processing Systems, 37: 12612–12635, 2024.

[36] Yongxin Zhu, Bocheng Li, Yifei Xin, Zhihua Xia, and Linli Xu. Addressing representation collapse in vector quantized models with one linear layer. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 22968–22977, 2025.

## A Related Work

Vector Quantization. Vector-quantized representation learning is introduced by VQ-VAE [30], which maps continuous encoder features to discrete code indices through nearest-neighbor lookup in a learned codebook. VQ-VAE-2 [22] improves this framework with hierarchical latent maps, and VQGAN [6] combines vector quantization with perceptual and adversarial objectives, making discrete visual tokenizers a standard interface for high-fidelity image synthesis. A large body of work improves reconstruction quality and token capacity by modifying the quantization structure, including residual or multi-stage quantization methods such as RQ-VAE [13] and ViT-based tokenizer architectures such as ViT-VQGAN [31]. Another line addresses codebook collapse and low utilization, which become increasingly severe when scaling codebook size or embedding dimension; representative solutions include codebook reset and replacement strategies [34], low-dimensional code embeddings and normalization [31], and soft-assignment-based training strategies such as stochastic or Gumbelsoftmax quantization [21, 12] and IBQ [25], which use relaxed or soft-to-hard categorical paths to improve codebook gradients beyond the selected hard code. Scalar-quantization-based methods, including FSQ [18] and LFQ [32], replace learned vector-codebook lookup with quantization over scalar or binary values, simplifying optimization and improving usage at scale while introducing different capacity trade-offs. Shared-projection methods such as VQ-STE++ [11], SimVQ [36], and FVQ [2] reparameterize code vectors through a shared function, providing an elegant and lightweight way to address the long-standing low-utilization problem in VQ training. Recently, pretrained vision foundation models have also been used to improve visual tokenizers, for example by initializing, regularizing, or aligning tokenizer representations with vision-foundation-model features [35, 5], which can strengthen semantic representation quality but is complementary to the training-stability problem studied in this work. StableVQ builds on the simple shared-projection foundation, but revisits the optimization responsibilities of the Encoder–Decoder and Codebook and targets the latent instability that remains even when codebook utilization has been substantially improved.

Image Generation. Image generation has been developed along both continuous and tokenized modeling paradigms. Early autoregressive models such as PixelCNN [29] and iGPT [3] model images directly in pixel space, but their sequential generation cost and weak compression make high-resolution synthesis difficult. Discrete tokenizers alleviate this bottleneck by converting images into compact latent token sequences: VQGAN [6] applies transformer-based autoregressive modeling in the VQ latent space, while VQ-VAE2 [22], RQ-Transformer [13], and related residual or hierarchical token models further explore multi-level discrete representations. Non-autoregressive and masked-token generators such as MaskGIT [1], MAGE [14], and MAGVIT-v2 [32] predict missing visual tokens and refine them iteratively, improving sampling efficiency and demonstrating the importance of tokenizer quality for generation. More recently, large-scale autoregressive image generators such as LlamaGen [27], VAR [28], RandAR [19], and Open-MAGVIT2 [17] show that language-model-style next-token, next-scale, or randomized-order prediction can achieve strong visual synthesis when paired with expressive visual tokens. In parallel, diffusion, score-based, and flow-matching models [10, 26, 16, 23, 20] generate images through continuous denoising or transport processes, and hybrid alternatives such as MAR [15] reduce or remove the dependence on hard vector quantization. These advances make the tokenizer a critical upstream component: regardless of whether the downstream generator is autoregressive, masked, multi-scale, or hybrid, unstable VQ training can limit reconstruction fidelity, code utilization, and ultimately generation quality. StableVQ is therefore orthogonal to generator design and aims to provide a more reliable discrete representation substrate for token-based image generation.

## B Limitations

StableVQ focuses on making VQ tokenizer training stable and reliable, providing a foundation on which stronger training objectives and downstream modeling choices can be explored. A natural direction is to study training recipes that better balance reconstruction quality, semantic structure, and generation-friendliness once codebook utilization and optimization stability are no longer the main bottlenecks. Another promising direction is to connect stable visual tokenization with unified generation and understanding objectives, where discrete tokens may need to preserve both lowlevel fidelity and high-level semantic information. Finally, while our experiments focus on image tokenizers, the same separation-of-concerns perspective may be useful in other domains that rely on vector quantization, such as video, audio, multimodal representation learning, or compression.

## C Additional Experimental Results

We provide additional baselines and experimental results for reference. For reconstruction, Table 5 supplements the main results with metrics from methods not covered in the main text, such as VQGAN and MaskGIT. We also provide more complete metrics for selected baselines under the same evaluation script in Table 6, including PSNR, SSIM, and other reference metrics.

Table 5: Complete reconstruction results on ImageNet $2 5 6 \times 2 5 6$ . Codebook Size is reported as n × d (number of codes × channel dimension). <sup>†</sup> denotes a larger encoder–decoder trained for up to 330 epochs. <sup>‡</sup> denotes training for 120 epochs.
<table><tr><td>Method</td><td>Projector</td><td>Tokens</td><td>Codebook Size  $( n \times d )$ </td><td>rFID↓</td><td>LPIPS↓</td><td>Usage↑</td></tr><tr><td>VQGAN [6]</td><td></td><td> $1 6 \times 1 6$ </td><td> $1 , 0 2 4 \times 2 5 6$ </td><td>7.94</td><td></td><td>44%</td></tr><tr><td>VQGAN [6]</td><td></td><td> $1 6 \times 1 6$ </td><td> $1 6 { , } 3 8 4 \times 2 5 6$ </td><td>4.98</td><td>0.2843</td><td>5.9%</td></tr><tr><td>SD-VQGAN [23]</td><td></td><td> $1 6 \times 1 6$ </td><td> $1 6 { , } 3 8 4 \times 8$ </td><td>5.15</td><td></td><td></td></tr><tr><td>MaskGIT [1]</td><td></td><td> $1 6 \times 1 6$ </td><td> $1 , 0 2 4 \times 2 5 6$ </td><td>2.28</td><td></td><td></td></tr><tr><td>LlamaGen [27]</td><td></td><td> $1 6 \times 1 6$ </td><td> $1 6 { , } 3 8 4 \times 8$ </td><td>2.19</td><td>0.2281</td><td>97%</td></tr><tr><td>LlamaGen [27]</td><td></td><td> $1 6 \times 1 6$ </td><td> $1 6 { , } 3 8 4 \times 2 5 6$ </td><td>9.21</td><td></td><td>0.29%</td></tr><tr><td>IBQ† [25]</td><td></td><td> $1 6 \times 1 6$ </td><td> $1 6 { , } 3 8 4 \times 2 5 6$ </td><td>1.37</td><td>0.2235</td><td>96%</td></tr><tr><td>IBQ† [25]</td><td></td><td> $1 6 \times 1 6$ </td><td> $2 6 2 , 1 4 4 \times 2 5 6$ </td><td>1.00</td><td>0.2030</td><td>84%</td></tr><tr><td colspan="5">Shared-Projection-Based Methods</td><td></td><td></td></tr><tr><td>VQGAN-LC [35]</td><td>Linear-1</td><td> $1 6 \times 1 6$ </td><td> $1 6 { , } 3 8 4 \times 8$ </td><td>3.01</td><td>0.2358</td><td>99%</td></tr><tr><td>VQGAN-LC [35]</td><td>Linear-1</td><td> $1 6 \times 1 6$ </td><td> $1 0 0 , 0 0 0 \times 8 $ </td><td>2.62</td><td>0.2212</td><td>99%</td></tr><tr><td>SimVQ [36]</td><td>Linear-1</td><td> $1 6 \times 1 6$ </td><td> $1 6 , 3 8 4 \times 2 5 6$ </td><td>2.89</td><td>0.2492</td><td>100%</td></tr><tr><td>SimVQ [36]</td><td>Linear-1</td><td> $1 6 \times 1 6$ </td><td> $2 6 2 , 1 4 4 \times 2 5 6$ </td><td>3.16</td><td>0.2516</td><td>100%</td></tr><tr><td>FVQ [2]</td><td>ViTBlock-2</td><td> $1 6 \times 1 6$ </td><td> $1 6 { , } 3 8 4 \times 2 5 6$ </td><td>1.70</td><td>0.2176</td><td>100%</td></tr><tr><td>FVQ [2]</td><td>ViTBlock-2</td><td> $1 6 \times 1 6$ </td><td> $2 6 2 , 1 4 4 \times 2 5 6$ </td><td>1.29</td><td>0.2003</td><td>100%</td></tr><tr><td>StableVQ</td><td>Linear-1</td><td> $1 6 \times 1 6$ </td><td> $1 6 { , } 3 8 4 \times 2 5 6$ </td><td>1.22</td><td>0.2235</td><td>100%</td></tr><tr><td>StableVQ</td><td>Linear-1</td><td> $1 6 \times 1 6$ </td><td> $2 6 2 , 1 4 4 \times 2 5 6$ </td><td>1.05</td><td>0.1947</td><td>100%</td></tr><tr><td>StableVQ‡</td><td>Linear-1</td><td> $1 6 \times 1 6$ </td><td> $1 6 , 3 8 4 \times 2 5 6$ </td><td>1.13</td><td>0.2134</td><td>100%</td></tr><tr><td>StableVQ‡</td><td>Linear-1</td><td> $1 6 \times 1 6$ </td><td> $2 6 2 , 1 4 4 \times 2 5 6$ </td><td>0.92</td><td>0.1893</td><td>100%</td></tr></table>

Table 6: Additional reconstruction metrics under the same evaluation script on ImageNet $2 5 6 \times 2 5 6 .$
<table><tr><td>Method</td><td>Codebook Size  $( n \times d )$ </td><td>Epochs</td><td>rFID↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>Usage↑</td></tr><tr><td>SimVQ [36]</td><td> $1 6 , 3 8 4 \times 2 5 6$ </td><td>40</td><td>2.89</td><td>21.36</td><td>0.5551</td><td>0.2492</td><td>100%</td></tr><tr><td>SimVQ [36]</td><td> $2 6 2 , 1 4 4 \times 2 5 6$ </td><td>40</td><td>3.16</td><td>21.31</td><td>0.5527</td><td>0.2516</td><td>100%</td></tr><tr><td>FVQ [2]</td><td> $1 6 , 3 8 4 \times 2 5 6$ </td><td>40</td><td>1.70</td><td>21.85</td><td>0.5894</td><td>0.2176</td><td>100%</td></tr><tr><td>FVQ [2]</td><td> $2 6 2 , 1 4 4 \times 2 5 6$ </td><td>40</td><td>1.29</td><td>22.50</td><td>0.6146</td><td>0.2003</td><td>100%</td></tr><tr><td>FVQ [2]</td><td> $1 6 , 3 8 4 \times 2 5 6$ </td><td>120</td><td>1.46</td><td>21.91</td><td>0.5951</td><td>0.2156</td><td>100%</td></tr><tr><td>FVQ [2]</td><td> $2 6 2 , 1 4 4 \times 2 5 6$ </td><td>120</td><td>1.07</td><td>22.55</td><td>0.6244</td><td>0.1965</td><td>100%</td></tr><tr><td>StableVQ</td><td> $1 6 , 3 8 4 \times 2 5 6$ </td><td>40</td><td>1.22</td><td>21.84</td><td>0.5816</td><td>0.2235</td><td>100%</td></tr><tr><td>StableVQ</td><td> $2 6 2 , 1 4 4 \times 2 5 6$ </td><td>40</td><td>1.05</td><td>22.77</td><td>0.6277</td><td>0.1947</td><td>100%</td></tr><tr><td>StableVQ</td><td> $1 6 , 3 8 4 \times 2 5 6$ </td><td>120</td><td>1.13</td><td>22.01</td><td>0.5965</td><td>0.2134</td><td>100%</td></tr><tr><td>StableVQ</td><td> $2 6 2 , 1 4 4 \times 2 5 6$ </td><td>120</td><td>0.92</td><td>22.81</td><td>0.6345</td><td>0.1893</td><td>100%</td></tr></table>

For downstream generation, we train class-conditional autoregressive transformers following IBQ [25] on top of StableVQ tokens and evaluate on ImageNet 256 × 256 using FID, IS, precision, and recall. As shown in Table 7, StableVQ achieves competitive generation quality, validating that improved reconstruction quality translates to gains in downstream generation.

Table 7: Class-conditional image generation results on ImageNet $2 5 6 \times 2 5 6$
<table><tr><td>Type</td><td>Tokenizer</td><td>Generator</td><td>Param.</td><td>FID↓</td><td>IS↑</td><td>Pre.↑</td><td>Rec.↑</td></tr><tr><td>Diff.</td><td>SD-VAE [23]</td><td>DiT-L/2 [20]</td><td>458M</td><td>5.02</td><td>167.2</td><td>0.75</td><td>0.57</td></tr><tr><td>Diff.</td><td>SD-VAE [23]</td><td>DiT-XL/2 [20]</td><td>675M</td><td>2.27</td><td>278.2</td><td>0.83</td><td>0.57</td></tr><tr><td>Mask.</td><td>VQGAN [6]</td><td>MaskGIT [1]</td><td>227M</td><td>6.18</td><td>182.1</td><td>0.80</td><td>0.51</td></tr><tr><td>VAR</td><td>VAR [28]</td><td>VAR-d16 [28]</td><td>310M</td><td>3.30</td><td>274.4</td><td>0.84</td><td>0.51</td></tr><tr><td>VAR</td><td>VAR [28]</td><td>VAR-d20 [28]</td><td>600M</td><td>2.57</td><td>302.6</td><td>0.83</td><td>0.56</td></tr><tr><td>AR</td><td>LlamaGen [27]</td><td>LlamaGen-L [27]</td><td>343M</td><td>3.80</td><td>248.3</td><td>0.83</td><td>0.51</td></tr><tr><td>AR</td><td>LlamaGen [27]</td><td>LlamaGen-XL [27]</td><td>775M</td><td>3.39</td><td>227.1</td><td>0.81</td><td>0.54</td></tr><tr><td>AR</td><td>FVQ [2]</td><td>LlamaGen-L [27]</td><td>343M</td><td>2.39</td><td>276.6</td><td>0.84</td><td>0.56</td></tr><tr><td>AR</td><td>FVQ [2]</td><td>LlamaGen-XL [27]</td><td>775M</td><td>2.07</td><td>287.0</td><td>0.83</td><td>0.58</td></tr><tr><td>AR</td><td>IBQ [25]</td><td>IBQ-B [25]</td><td>342M</td><td>2.88</td><td>254.7</td><td>0.84</td><td>0.51</td></tr><tr><td>AR</td><td>IBQ [25]</td><td>IBQ-L [25]</td><td>649M</td><td>2.45</td><td>267.5</td><td>0.83</td><td>0.52</td></tr><tr><td>AR</td><td>StableVQ</td><td>IBQ-B [25]</td><td>342M</td><td>2.35</td><td>256.0</td><td>0.82</td><td>0.58</td></tr><tr><td>AR</td><td>StableVQ</td><td>IBQ-L [25]</td><td>649M</td><td>2.18</td><td>250.4</td><td>0.82</td><td>0.59</td></tr></table>

## D Additional Analysis Experiments

![](images/5047719068a56d69496a7313c18f015a93cc28585ab32188db88114ac4ab1a6c.jpg)  
（a）

![](images/e3f3bd57d8aed6f0496c9d6531d21c07277e1732b95fe16e461e24bd2136ee44.jpg)  
（b）  
Figure 6: Additional analysis experiments. (a) Usage@5k under different Gaussian codebook initialization scales. (b) Usage@5k under different learning rates.

## D.1 Effect of Codebook Initialization

Codebook initialization has long been an important practical trick in conventional VQ training, because an inappropriate initialization scale can lead to poor codebook utilization. A common strategy is to initialize the Codebook within a small numerical range, which often helps utilization increase more stably. However, this trick is not always applicable: for example, when an $\ell _ { 2 }$ normalization constraint is imposed, the effective code distribution can no longer be controlled simply by shrinking the raw initialization range. Moreover, an overly small initialization scale may also prolong the usage-recovery phase, reducing training efficiency.

To study how initialization affects codebook utilization across configurations, we follow the pilot setting in Section 4.3: the Encoder is frozen, and only the Codebook is optimized to fit a fixed target distribution. We vary the Gaussian initialization scale of the Codebook and report codebook utilization at 5k steps, computed within a window of 65,536 tokens. As shown in Figure 6(a), when the initialization scale is too small, the linear shared projector used in SimVQ-style settings recovers utilization slowly; when the initialization scale becomes larger, the ViT-based shared projector used in FVQ-style settings faces a clear dead-code risk. Replacing the standard VQ loss with Region VQ Loss substantially improves both cases, allowing utilization to approach full usage at 5k steps across initialization scales. This verifies that StableVQ reduces the dependence of VQ training on codebook initialization design, and also suggests robustness to scale divergence during training.

## D.2 Effect of Shared Projector

For methods that rely on a shared projector to improve codebook utilization, projector design has recently become an important practical consideration. When the projector is the simplest linear layer, its limited learning capacity can make it difficult for the Codebook to quickly track the encoder output distribution, often leading to slow utilization growth and more frequent scale-divergence events during training. In contrast, using a more expressive ViT block can improve tracking capacity, but introduces additional architecture-specific design cost and constraints on codebook size.

To examine how shared projector design affects utilization, we again follow the frozen-Encoder pilot setting and measure codebook utilization at 5k steps under different learning rates, using the same 65,536-token window. The Codebook is initialized with a small Gaussian standard deviation of $1 0 ^ { - 4 }$ , so the results more directly reflect projector learning capacity while reducing the influence of dead-code effects. As shown in Figure 6(b), neither a two-layer linear projector nor a two-layer MLP can raise utilization to a high level within 5k steps, even when the learning rate is increased to $5 \times 1 0 ^ { - 3 }$ . The two-layer ViT block performs better at moderate learning rates, but its utilization drops sharply when the learning rate becomes too large. With Region VQ Loss, however, different projector structures all achieve strong utilization within 5k steps, and their usable learning-rate range is substantially widened. This suggests that the Codebook benefits from an independent learning-rate schedule for tracking the encoder distribution, but also shows that changing projector capacity alone is insufficient for stable and rapid utilization growth. The more critical factor is to provide the Codebook with a principled learning objective, which allows StableVQ to improve utilization across shared-projector designs and reduces the dependence of robust VQ training on carefully engineered projector structures.

## D.3 Computational and Memory Overhead

Table 8 reports matched baseline and StableVQ profiles with 16K and 262K codebooks on the same machine. The Decoupled Schedule adds no forward or backward operation, and the measured time difference of Dynamic STE is within noise. The cost of Region VQ Loss decreases as codebook utilization increases because fewer inactive codes require propagated targets. All components are training-only and leave tokenizer inference unchanged.

Table 8: Training-time and memory overhead of StableVQ components. Parentheses denote changes from the corresponding baseline, and all memory values are per device.
<table><tr><td colspan="7">Training Cost</td><td colspan="2">Memory Usage</td></tr><tr><td>Codebook</td><td>Baseline (ms)</td><td>Dynamic STE (ms)</td><td>Region VQ @30% (ms, ∆)</td><td>Region VQ @50% (ms, ∆)</td><td>Region VQ @90% (ms, ∆)</td><td>Baseline (GiB)</td><td></td><td>Region VQ (GiB, ∆)</td></tr><tr><td>16K</td><td>438.1</td><td>437.2</td><td>441.2 (+0.71%)</td><td>440.4 (+0.51%)</td><td>440.2 (+0.46%)</td><td></td><td>34.06</td><td>34.07 (+0.04%)</td></tr><tr><td>262K</td><td>462.2</td><td>462.7</td><td>472.2 (+2.16%)</td><td>469.2 (+1.50%)</td><td>466.2 (+0.85%)</td><td>35.12</td><td></td><td>35.55 (+1.24%)</td></tr></table>

## D.4 Comparison with Prior STE Corrections

VQ-STE++ [11] employs Alternating Optimization, which is similar in motivation to Dynamic STE: both aim to suppress unreliable task gradients when quantization error is large. However, VQ-STE++ relies on alternating inner and outer updates, introducing several sensitive hyperparameters and additional training overhead. It removes the encoder commitment loss and assumes an initially aligned code–token distribution established through k-means initialization, making optimization sensitive when this alignment is absent.

The Rotation Trick [9] transforms the encoder gradient through a rotation matrix R and a rescaling factor $\| q \| / \| z \|$ . For rotation, R redirects the backward gradient according to the angle between token z and code q. Although elegant, it provides no rigorous guarantee of a more accurate gradient and may instead redirect the gradient away from the desired direction. For rescaling, the factor suppresses the encoder gradient when $\| q \| < \| z \|$ , but amplifies it when $\| q \| > \| z \|$ , even for distant token–code pairs. Dynamic STE instead uses relative within-batch quantization distances to attenuate unreliable gradients and never amplify them. Moreover, under $\ell _ { 2 }$ normalization, the rescaling factor becomes one and loses its attenuation effect, whereas Dynamic STE remains active.

We conduct a controlled 15-epoch experiment with Region VQ and all other experimental settings fixed. As shown in Table 9, VQ-STE++ consistently exhibits a pronounced collapse in codebook utilization without the joint use of k-means initialization to pre-align the token–code distributions and a norm bottleneck. In contrast, both the Rotation Trick and Dynamic STE achieve full codebook uti lization with the support of Region VQ, while Dynamic STE yields substantially better reconstruction performance across metrics.

Table 9: Comparison of STE corrections in training stability and reconstruction quality.
<table><tr><td>Method</td><td>Training Status</td><td>Usage↑</td><td>rFID↓</td><td>PSNR↑</td><td>LPIPS↓</td></tr><tr><td>VQ-STE++ [11]</td><td>Collapsed</td><td></td><td></td><td></td><td></td></tr><tr><td>Rotation Trick [9]</td><td>Completed</td><td>100.00%</td><td>3.6437</td><td>20.4024</td><td>0.2568</td></tr><tr><td>Dynamic STE (ours)</td><td>Completed</td><td>100.00%</td><td>2.7942</td><td>21.1882</td><td>0.2359</td></tr></table>

## D.5 Comparison with Explicit Distribution-Alignment Methods

Wasserstein VQ [7] and MMD VQ [8] formulate global distribution matching as a training loss, using Gaussian moments and kernel statistics, respectively. Region VQ Loss instead addresses the asymmetric supervision of standard VQ by assigning persistently inactive codes explicit, local targets propagated from statistically supported active regions. It therefore requires no parametric assumption about the token distribution and avoids the costly pairwise kernel computation of MMD VQ.

We evaluate these methods on the non-Gaussian mixture benchmark introduced by VQ-Transplant [8]. At ζ = 0, the target distribution reduces to a single Gaussian; increasing ζ separates the two mixture modes and progressively strengthens its non-Gaussian structure. Following its setting, we use 16,384 codes of dimension 8, sample 20K tokens per step, and report codebook utilization at 10K steps. We quote the Wasserstein VQ and MMD VQ utilization results from Table 13 of VQ-Transplant and evaluate Region VQ on the same ζ grid with a linear shared projector. We additionally measure the per-step training time of all methods under the same environment. To mitigate under-coverage caused by heavily overlapping recipient sets in this synthetic benchmark, we automatically enlarge the propagation quotas when recipient collisions are severe.

Table 10 reports the resulting codebook utilization across different ζ values and the corresponding per-step training time. As the target distribution becomes strongly non-Gaussian, Wasserstein VQ and MMD VQ fall to 34.8% and 75.6% utilization at ζ = 4, respectively, whereas Region VQ maintains 99.4%. Region VQ also remains close to Wasserstein VQ in training time and is 34.2× faster than MMD VQ. These results demonstrate that Region VQ combines robust codebook utilization across different distribution relationships with low training overhead.

Table 10: Codebook utilization and training efficiency on the non-Gaussian distribution-fitting benchmark. Utilization is measured at 10K steps. Relative time is normalized to Wasserstein VQ.
<table><tr><td>Method</td><td>ζ = 0</td><td>ζ= 1</td><td>ζ= 2</td><td>ζ=3</td><td>ζ= 4</td><td>Time (ms)</td><td>Relative Time</td></tr><tr><td>Wasserstein VQ [7]</td><td>99.9%</td><td>97.0%</td><td>62.7%</td><td>44.8%</td><td>34.8%</td><td>6.98</td><td>1.00×</td></tr><tr><td>MMD VQ [8]</td><td>99.9%</td><td>99.8%</td><td>92.5%</td><td>85.7%</td><td>75.6%</td><td>295.57</td><td>42.36×</td></tr><tr><td>Region VQ (ours)</td><td>99.8%</td><td>99.3%</td><td>93.2%</td><td>99.4%</td><td>99.4%</td><td>8.64</td><td>1.24×</td></tr></table>

## E Robustness Test Illustration

The robustness test is designed to evaluate whether a VQ training method can recover high codebook utilization when the relationship between the Codebook distribution and the token distribution changes. This setting complements the final validation usage reported in the main reconstruction table: a method may eventually report high utilization under its standard configuration, yet still recover slowly or fail when the codebook-token relationship becomes less favorable. We control this distributional relationship through different codebook initializations.

We construct different codebook-token distribution relationships by varying the Gaussian initialization scale of the codebook vector base while keeping the data, architecture, optimizer, learning-rate schedule, and training budget fixed. Changing this scale alters the initial geometry between the projected code vectors and the encoder token distribution: small scales place code vectors in a compact region, whereas larger scales spread the code distribution over a broader region relative to the token distribution. This provides a controlled way to test codebook usage recovery under multiple mismatch settings without changing the input data or reconstruction objective.

For each method and each mismatch setting, we train the tokenizer for the same early-stage budget and record codebook usage throughout training. Usage is computed within a window of 65,536 tokens, so the reported values are slightly lower than utilization measured over the full validation set. Figure 5 visualizes these trajectories as heatmaps: each row corresponds to one mismatch setting, and brighter colors indicate higher codebook usage. A stable method should recover high usage quickly across most rows rather than depending on a narrow range of favorable initial relationships.

We summarize the heatmaps with UR-AUC, which denotes Usage Recovery AUC. Let $U _ { m } ( t )$ be the codebook usage percentage at training step t under mismatch setting $m .$ . We compute the normalized area under each usage curve and average over the tested mismatch settings:

$$
\mathrm { U R \mathrm { - } A U C } = { \frac { 1 } { | { \mathcal { M } } | } } \sum _ { m \in { \mathcal { M } } } { \frac { 1 } { T - t _ { 0 } } } \int _ { t _ { 0 } } ^ { T } U _ { m } ( t ) d t .\tag{8}
$$

In practice, we use the logged usage values and compute the integral with the trapezoidal rule. Higher UR-AUC indicates faster and more consistent recovery of codebook utilization under codebook-token distribution mismatch. The error bars for UR-AUC in Table 1 are computed as the sample mean and sample standard deviation over three runs with different random seeds, where each run includes the same set of initialization-induced mismatch settings described above.

## F Region VQ Algorithm

Algorithm 1 summarizes the Region VQ procedure in one training step. It follows the design in Section 4.3: current assignments determine active codes and their targets, a FIFO queue identifies persistently inactive codes, and targets from sufficiently reliable active codes are propagated to nearby inactive ones before computing the codebook loss.

Algorithm 1 Region VQ in one training step   
Require: Token features $\mathbf { Z } = \{ \mathbf { z } _ { u } \} _ { u = 1 } ^ { N }$ , projected codebook $\mathbf { E } = \{ \mathbf { e } _ { k } \} _ { k = 1 } ^ { K }$ , current assignments $a _ { u } \in [ K ]$   
FIFO queue $Q$ of length $W$   
Ensure: Code targets $\{ \mathbf { t } _ { k } ^ { - } \} _ { k = 1 } ^ { K }$ and effective code set $\mathcal { M } _ { t }$ for codebook loss   
1: Append current assignment indices $\{ a _ { u } \} _ { u = 1 } ^ { N } \ t o \ Q$   
2: Let $\boldsymbol { S _ { t } } \gets \{ a _ { u } \} _ { u = 1 } ^ { N }$ be the currently active codes   
3: Let $\mathcal { A } _ { t }  \bigcup \dot { Q }$ be the codes active within the FIFO window   
4: Let $\mathcal { N } _ { t } \gets \breve { [ K ] } \backslash \mathcal { A } _ { t }$ be the persistently inactive codes   
5: Initialize $\mathbf { t } _ { k } \gets \mathbf { e } _ { k }$ for all $\dot { k \in [ K ] }$ ▷ Self-target by default   
6: Initialize effective code set $\bar { \mathcal { M } _ { t } }  \emptyset$   
7: for each k $\in S _ { t }$ do   
8: $\mathcal { U } _ { k }  \{ u : a _ { u } = k \} , \quad n _ { k }  | \mathcal { U } _ { k } |$   
9: $\begin{array} { r } { \mathbf { t } _ { k } \gets \frac { 1 } { n _ { k } } \sum _ { u \in \mathcal { U } _ { k } } \mathbf { z } _ { u } } \end{array}$   
10: Add $\mathfrak { c } \mathrm { t o } \ddot { \mathcal { M } } _ { t }$   
11: end for   
12: Let $S _ { t } ^ { + } \gets \{ k \in S _ { t } : n _ { k } > 1 \}$   
13: for each $k \in S _ { t } ^ { + }$ do   
14: $q _ { k } \gets \left\lceil n _ { k } | \mathcal { N } _ { t } | / \sum _ { \ell \in \mathcal { S } _ { t } ^ { + } } n _ { \ell } \right\rceil$   
15: Select the $q _ { k }$ nearest codes to k from $\mathcal { N } _ { t }$ in projected space and denote them by $\mathcal { R } _ { k }$   
16: end for   
17: for each $j \in \mathcal { N } _ { t }$ that is selected by at least one active code do   
18: Let $\check { \mathcal { K } } _ { j } \gets \{ k \in \mathcal { S } _ { t } ^ { + } : j \in \mathcal { R } _ { k } \}$   
19: $\begin{array} { r } { \mathbf { t } _ { j } \gets \frac { 1 } { | \mathcal { K } _ { j } | } \sum _ { k \in \mathcal { K } _ { j } } \quad } \end{array}$ t<sub>k</sub>   
20: Add j to M<sub>t</sub>   
21: end for   
22: Compute codebook loss only on effective codes: $\begin{array} { r } { \mathcal { L } _ { \mathrm { c o d e } } = \frac { 1 } { | \mathcal { M } _ { t } | } \sum _ { k \in \mathcal { M } _ { t } } \| \mathbf { e } _ { k } - \mathrm { s g } [ \mathbf { t } _ { k } ] \| _ { 2 } ^ { 2 } } \end{array}$

## G Experimental Details

We provide the detailed configurations used for reconstruction training in Table 11. For the generation experiments in Appendix C, we follow the corresponding IBQ training setting and train for approximately 350 epochs. Other experiments in this paper describe their key differences from the standard settings in their respective contexts.

Table 11: Main experiment configurations.
<table><tr><td>Config</td><td>SimVQ</td><td>FVQ</td><td>StableVQ</td></tr><tr><td>Base Batch Size</td><td>128</td><td>128</td><td>128</td></tr><tr><td>Training Epochs</td><td>40</td><td>40 /120</td><td>40 / 120</td></tr><tr><td>Optimizer</td><td>Adam</td><td>Adam</td><td>Adam</td></tr><tr><td>Optimizer Parameters</td><td> $\beta _ { 1 } = 0 . 9 , \beta _ { 2 } = 0 . 9 5$ </td><td> $\beta _ { 1 } = 0 . 9 , \beta _ { 2 } = 0 . 9 5$ </td><td> $\beta _ { 1 } = 0 . 9 , \beta _ { 2 } = 0 . 9 5$ </td></tr><tr><td>Base Learning Rate Codebook Init Type</td><td>1e-4 Gaussian</td><td>1e-4 Uniform</td><td>1e-4</td></tr><tr><td></td><td></td><td></td><td>Uniform</td></tr><tr><td>LR Warmup LR Plateau</td><td>0%</td><td>10%, 0.005× to 1×</td><td>10%, 0.005× to 1×</td></tr><tr><td></td><td>100%,1×</td><td>27%,1×</td><td>27%,1×</td></tr><tr><td>LR Annealing</td><td>0%</td><td> $6 3 \% , 1 \times 1 0 0 . 0 1 \times$ </td><td> $6 3 \% , 1 \times 1 0 \ : 0 . 0 1 \times$ </td></tr><tr><td>Codebook LR</td><td>Same as LR</td><td>Same as LR</td><td>Constant 1e—3</td></tr></table>

## H Visualization Results

We provide qualitative visualization results for reconstruction and downstream generation. Figure 7 compares reconstruction results with a 16k-code codebook, Figure 8 compares reconstruction results with a 262k-code codebook, and Figure 9 shows samples from the downstream generation task.

Original  
StableVQ-16k  
FVQ-16k  
Original  
StableVQ-16k  
FVQ-16k  
![](images/4f46516fb9b2fc03689c165141dc1c60c7131c974ee5725f0203795cffaaa43a.jpg)  
Figure 7: Qualitative reconstruction comparison with a 16k codebook.

Original  
StableVQ-262k  
FVQ-262k  
Original  
StableVQ-262k  
FVQ-262k  
![](images/b9e7fdde8da2e6870a9d339c3e79342a5859e03e939094d2ba176592bed7b5f0.jpg)  
Figure 8: Qualitative reconstruction comparison with a 262k codebook.

![](images/c1cc73fa6d35aa35d77db9786a10f1a64a08521aa6040b50a1eeb846afd13a2e.jpg)  
Figure 9: Qualitative samples from the downstream class-conditional image generation task.