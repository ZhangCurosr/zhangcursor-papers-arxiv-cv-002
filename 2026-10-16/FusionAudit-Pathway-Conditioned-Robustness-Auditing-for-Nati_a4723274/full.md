# FusionAudit: Pathway-Conditioned Robustness Auditing for Native Multimodal Models

Jiaming Zhang Zherui Li Siqi Guo Junhao Dong Kun Wang Wei Yang Bryan Lim

College of Computing and Data Science Nanyang Technological University, Singapore Project page: https://jiamingzhang.netlify.app/fusionaudit

## Abstract

As vision-language models (VLMs) transition from frozen-encoder pipelines to native multimodal architectures, the visual channel increasingly serves as both a perceptual input and a carrier of rendered text that competes with user prompts. Current single-attack, pixel-centric evaluations fail to capture this complexity: they collapse distinct visual attack surfaces into a single scalar and ignore how architectural shifts alter vulnerability. We introduce PaRMA, an evidence-based auditing framework that defines pathway-conditioned risk over two primary imagelevel interventions: pixel perturbation and rendered-text insertion. Instead of arbitrary attack enumeration, PaRMA uses pathway-specific diagnostics to expose architecture-specific blind spots. Notably, we uncover a see-but-not-deceived signature where native VLMs attend heavily to typographic overlays but visually reject them, and we identify an objective mismatch between encoder-level and answer-level attacks in deep-fusion models. To exploit these findings, we instantiate FusionAudit, introducing two diagnostic-motivated attacks: MANTIS (margin-aware pixel perturbation) and Nat-Evid (host-congruent semantic insertion). Evaluating 18 open-weight VLMs across a novel nativeness taxonomy, we demonstrate that legacy single-attack rankings (e.g., PGD) are unstable for native models. Our diagnostic-driven attacks substantially raise the estimated residual risk, showing that PaRMA provides a falsifiable, evidence-driven auditing methodology for uncovering vulnerabilities missed by standard benchmarks.

## 1 Introduction

Vision-language models (VLMs) are undergoing a structural transition. Early systems treated the visual channel as a peripheral sensor, piping a frozen vision encoder into a pretrained language model via a projection bottleneck. Today, native multimodal architectures—featuring multi-layer visual injection and early-fusion shared backbones [4, 41, 15]—process visual and textual tokens in a unified representational space from the earliest layers. In this paradigm, pixels, scene text, and document fields compete directly with the user’s prompt. However, robustness evaluation has not kept pace. Current protocols remain siloed into pixel-space adversarial attacks [61, 8] and typographic overlays [18], neither of which accounts for the deep integration of native VLMs.

This structural gap leaves two critical blind spots in current evaluations. First, on the pixel pathway, legacy attacks anchored at the vision encoder ignore deep decoder-side gradients. Conversely, standard end-to-end attacks (e.g., PGD [29]) spread gradient pressure across all output tokens, underweighting the exact position where the model commits to a wrong answer. Second, on the rendered-text pathway, native VLMs exhibit a striking see-but-not-deceived signature: attention diagnostics indicate that these models allocate heavy attention to overlaid text, yet they resist adopting it as a task response.

This suggests native VLMs develop a fine-grained sensitivity to visual congruence, making direct typographic attacks incomplete and leaving additional semantic failures unmeasured due to visual rejection.

We argue that robustness auditing for native VLMs should be an evidence-based methodology rather than a static benchmark. We introduce PaRMA (Pathway-conditioned Risk for Multimodal Auditing), a framework that organizes the visual channel into two primary intervention families: pixel perturbation $( \pi _ { \mathrm { p i x } } )$ and rendered-text insertion $( \pi _ { \mathrm { t x t } } )$ . Instead of blindly enumerating attacks, PaRMA uses behavioral and gradient diagnostics to expose architecture-specific blind spots, which then motivate the design of diagnostic-motivated interventions. We instantiate this framework as FusionAudit. Guided by our diagnostic findings, FusionAudit introduces two strengthened attacks: MANTIS, a margin-aware pixel attack that concentrates gradients at the first-divergence token to address this mismatch; and Nat-Evid, a host-congruent semantic intervention that naturalizes text payloads to bypass visual rejection. The audit’s principal scalar is the residual pathway-complete risk, yielding an empirical lower bound on pathway-complete risk.

Contributions. Our core contributions are:

• We identify native VLM robustness as a structurally distinct problem, introducing a graded nativeness taxonomy $\mathrm { ( L _ { 1 } \mathrm { - } L _ { 3 } ) }$ to organize audit targets across fusion designs.

• We propose PaRMA and FusionAudit, an evidence-based auditing methodology. We uncover the see-but-not-deceived signature in native VLMs and introduce diagnostic-motivated attacks (MANTIS, Nat-Evid) specifically designed to exploit these diagnostic findings.

• We conduct an extensive 18-model audit demonstrating that legacy single-attack proxies produce unstable rankings for native models. Our diagnostic-driven attacks substantially raise the estimated residual visual-channel risk, showing that current evaluations can underestimate VLM vulnerabilities.

## 2 Related Work

Native Vision-Language Models. Early Vision-language Models (VLMs) typically used modular architectures [37, 14, 51], connecting a frozen or lightly adapted vision encoder to a language model through a projection layer, query module, or connector [1, 50, 47]. Models like BLIP-2 [24], InstructBLIP [14], MiniGPT-4 [62], and LLaVA [26] showed that this alignment-era pipeline supports visual question answering, captioning, and visual instruction tuning [53, 16]. Later work systematically examined how vision backbones, connectors, and training recipes affect VLM behavior [22, 43, 9, 27, 4]. Recent native multimodal models instead adopt multi-layer visual injection or early-fusion shared backbones, allowing visual and textual information to interact earlier or deeper in computation [41, 52, 12, 28, 54, 42, 15]. This shift means the visual channel is no longer merely a frontend conditioning signal, so robustness evaluation must move beyond assumptions inherited from modular pipelines.

Adversarial Robustness of VLMs. Classical adversarial robustness work shows that small-norm pixel perturbations and PGD-style attacks can expose systematic vulnerabilities in visual models [7, 29]. These methods have been adapted to VLMs to attack vision encoder representations, cross-modal alignment, or final generated responses [38, 45, 61, 55]. Prior studies further show that large VLMs remain sensitive to visual perturbations and that adversarial examples can transfer across models [39, 8, 33, 49, 48, 46, 56–60]. Beyond pixel-space attacks, typographic attacks demonstrate that text rendered inside images can alter a VLM’s interpretation of visual content [18, 31, 20, 11]. However, existing evaluations usually treat pixel perturbations and text overlays as separate threat vectors, without systematically measuring joint risk across visual pathways in native VLMs [5, 3, 21, 36, 40].

Robustness Evaluation and Diagnostic Auditing. Robustness evaluation has long recognized that single attacks can underestimate risk [2], motivating fixed attack suites such as AutoAttack [13]. Multimodal robustness evaluations have begun to aggregate diverse perturbations and attacks, but most still rely on predefined static suites [45]. In parallel, diagnostic tools including representation similarity, attention analysis, and activation patching help reveal internal representations and behavioral mechanisms [6, 44, 19, 35, 25]. Yet these diagnostics are typically used post hoc to explain failures rather than to guide stronger subsequent attacks [23, 34, 17]. PaRMA instead integrates pathway-specific diagnostics into the audit, reporting residual risk after diagnostic-driven strengthening rather than the maximum over a pre-specified collection.

## 3 Native Multimodal VLMs: Taxonomy and Threat Scope

The transition from modular pipelines to native multimodal training alters the VLM attack surface. In pipeline models, a projection bottleneck isolates the vision encoder from the language backbone. Deep fusion removes this bottleneck, altering end-to-end gradient paths, while joint early-layer attention forces visual and textual tokens to compete directly. To systematically audit this evolving landscape, we introduce a graded nativeness taxonomy and formalize the visual threat scope.

## 3.1 Nativeness Taxonomy

We categorize VLMs based on their fusion depth and pretraining strategy. Fusion level co-varies with model family, scale, and training data, so level-wise trends are correlational. This taxonomy serves as an organizational axis for our audit, rather than a strict causal ladder:

$\mathrm { L _ { 1 } }$ (Alignment-era): A frozen or lightly adapted vision encoder connects to a language model via a projection module [26, 62]. Modalities remain in structurally separate spaces; the bottleneck is the only junction.

$\mathrm { L _ { 2 } }$ (Shallow fusion): Models feature large-scale joint multimodal pretraining. Visual tokens enter the backbone through a connector, but do not re-enter computation at deeper layers; the backbone processes a fixed visual prefix.

$\mathrm { L _ { 3 } }$ (Deep/early fusion): Visual information participates throughout the language computation. This includes DeepStack (DS) models that re-inject visual features across multiple layers [4], and Early Fusion (EF) models that share transformer blocks between modalities from the first layer [42, 15].

A theoretical fourth level $( \mathrm { L } _ { 4 } ) { - } \mathbf { a }$ strict single-backbone design unifying all modalities under one tokenizer without a distinct vision tower—completes the continuum, though no audited open-weight model currently meets this threshold.

## 3.2 Audit Scope and Visual Pathways

We formalize a VLM as a function mapping a visual input $x _ { v }$ and a text prompt $x _ { t }$ to an answer. To isolate visual-channel vulnerabilities from text-only jailbrea $^ { \mathrm { ~ \tiny ~ S ~ , ~ } }$ our audit fixes the text prompt and intervenes exclusively on $x _ { v }$ . Crucially, all empirical Attack Success Rates (ASRs) in our evaluation are strictly conditioned on the model’s clean-correct set—the subset of examples the model answers correctly before any intervention.

We decompose the visual attack surface into two primary pathways:

1. Perceptual corruption $( \pi _ { \mathbf { p i x } } ) { : }$ Pixel-space perturbations. In deep-fusion architectures, the unbroken gradient path from output loss to pixels can expose failures missed by attacks centered on $\mathrm { L _ { 1 } } { \mathrm { - } } \mathrm { s t y l e }$ bottlenecks.

2. Rendered-text insertion $( \pi _ { \mathbf { t x t } } ) \colon$ Typographic overlays. Added image text competes with existing visual evidence and the user’s prompt within the joint attention mechanism.

The Audit Problem. Given a model and a dataset, a pathway-complete robustness audit estimates the failure probability over the clean-correct set induced by the union of pathway-specific interventions. Crucially, it should diagnose whether these failures exhibit behavioral signatures that can guide stronger attacks.

## 4 Method: PaRMA and FusionAudit

We operationalize our threat scope through PaRMA (Pathway-conditioned Risk for Multimodal Auditing). Unlike static benchmarks, PaRMA is a diagnostic-driven audit protocol: baseline attacks yield diagnostic signals, which in turn guide the design of diagnostic-motivated augmented attacks. We instantiate this framework as FusionAudit over two visual pathways (Figure 1).

![](images/a1ec4bd9ff490a6bb0ea221cd3d0afc572ad982e61bac7f18295897e089e6548.jpg)  
Figure 1: The PaRMA framework and FusionAudit pipeline. Native VLMs are evaluated over two visual pathways: pixel perturbation and rendered-text insertion. Baseline attacks (Vis-PGD, End-PGD, Typo-Evid) yield initial failures and diagnostic readouts. These diagnostics surface architectureassociated failure signatures (e.g., the see-but-not-deceived signature), guiding the generation of diagnostic-motivated augmented attacks (MANTIS, Nat-Evid) to compute the final falsifiable residual risk $\widehat { R } ^ { + }$

## 4.1 Pathway-Conditioned Risk Formulation

We treat the visual input $x _ { v }$ as a site for pathway-specific interventions. A pathway k is an intervention family $( B _ { k } , \phi _ { k } )$ , where $\boldsymbol { B } _ { k }$ is the perturbation domain and $\phi _ { k } ( x _ { v } ; \delta )$ is the intervened image. FusionAudit instantiates two pathways:

$$
\begin{array} { r l } & { { \pi } _ { \mathrm { p i x } } : { \mathcal { B } } _ { \mathrm { p i x } } = \{ \delta : \| \delta \| _ { \infty } \leq \varepsilon \} , \quad \phi _ { \mathrm { p i x } } ( x _ { v } ; \delta ) = \mathrm { c l i p } ( x _ { v } + \delta ) , } \\ & { { \pi } _ { \mathrm { t x t } } : { \mathcal { B } } _ { \mathrm { t x t } } = { \Sigma } ^ { \leq L } \times { \mathcal { P } } , \quad \phi _ { \mathrm { t x t } } ( x _ { v } ; s , p ) = \mathrm { r e n d e r } ( x _ { v } ; s , p ) . } \end{array}
$$

Here $\Sigma ^ { \le L }$ is the space of candidate rendered strings of length at most $L ,$ and $\mathcal { P }$ denotes admissible render placements; s is the inserted string and $p$ its placement. Given a finite attack suite $\mathcal { A } _ { k }$ for pathway $k ,$ let $\mathcal { S } _ { a } = \{ z \in \mathcal { C } _ { \theta } ( \mathcal { D } ) : \mathcal { I } ( f _ { \theta } ( \bar { a ( } x _ { v } ) , \mathbf { \bar { \phi } } x _ { t } ) , \hat { y } ) = 0 \}$ be the subset of clean-correct examples broken by attack a. The empirical pathway-complete risk is the union failure rate:

$$
\widehat { \mathcal { R } } _ { \mathrm { F A } } ( f _ { \theta } ; \mathcal { D } , \mathcal { A } ) = \frac { \left| \bigcup _ { k } \bigcup _ { a \in \mathcal { A } _ { k } } S _ { a } \right| } { \left| \mathcal { C } _ { \theta } ( \mathcal { D } ) \right| } .\tag{1}
$$

PaRMA’s departure from naive multi-attack unions is that $\widehat { \mathcal { R } } _ { \mathrm { F A } }$ is not static; it is a lower bound strengthened via pathway-resolved diagnostics.

## 4.2 Baselines and Diagnostic Suite

The baseline protocol assigns standard attacks to each pathway to probe primary failure modes.

For perceptual corruption $( \pi _ { \mathfrak { p i x } } )$ , we employ two PGD variants bounded by $\| \delta \| _ { \infty } \leq \varepsilon$ . Let $h _ { \mathcal { V } } ( x _ { v } )$ ) denote the flattened vision-encoder output. Vis-PGD anchors the loss at the vision encoder, minimizing cosine similarity to the clean feature map:

$$
\operatorname* { m i n } _ { \| \delta \| _ { \infty } \leq \varepsilon } \cos \bigl ( h _ { \mathcal { V } } ( \phi _ { \mathrm { p i x } } ( x _ { v } ; \delta ) ) , h _ { \mathcal { V } } ( x _ { v } ) \bigr ) .\tag{2}
$$

End-PGD anchors the loss at the end-to-end output, maximizing cross-entropy over the full teacherforced correct answer sequence $y ^ { + }$

$$
\operatorname* { m a x } _ { \| \boldsymbol { \delta } \| _ { \infty } \leq \varepsilon } \Bigl [ - \sum _ { t = 1 } ^ { | \boldsymbol { y } ^ { + } | } \log p _ { \theta } \bigl ( \boldsymbol { y } _ { t } ^ { + } \mid \phi _ { \mathrm { p i x } } ( \boldsymbol { x } _ { v } ; \boldsymbol { \delta } ) , \boldsymbol { x } _ { t } , \boldsymbol { y } _ { < t } ^ { + } \bigr ) \Bigr ] .\tag{3}
$$

For rendered-text insertion $( \pi _ { \mathrm { t x t } } )$ , following Cheng et al. [10], Typo-Evid superimposes a taskmatched wrong-answer string w at a fixed region p as a direct typographic overlay:

$$
\tilde { x } _ { v } = \phi _ { \mathrm { t x t } } ( x _ { v } ; w , p ) .\tag{4}
$$

To analyze rendered-text baseline failures, PaRMA uses two diagnostics. Read–Use Gap (RUG). For $\pi _ { \mathrm { t x t } }$ , let $E _ { \mathrm { r e a d } } ( z )$ denote the event that the rendered wrong-answer string w is recovered under an explicit reading prompt, and let $E _ { \mathrm { u s e } } ( z )$ denote the event that the task-prompt response is wrong and contains $w .$ . RUG is therefore a prompt-conditioned accessibility–adoption diagnostic. Over the clean-correct set,

$$
\begin{array} { r l } & { P _ { \mathrm { r e a d } } ( f _ { \theta } ) = \underset { z \sim \mathcal { D } } { \mathrm { P r } } [ E _ { \mathrm { r e a d } } ( z ) \mid z \in \mathcal { C } _ { \theta } ( \mathcal { D } ) ] , } \\ & { P _ { \mathrm { u s e } } ( f _ { \theta } ) = \underset { z \sim \mathcal { D } } { \mathrm { P r } } [ E _ { \mathrm { u s e } } ( z ) \mid z \in \mathcal { C } _ { \theta } ( \mathcal { D } ) ] , } \\ & { \mathrm { R U G } ( f _ { \theta } ) = P _ { \mathrm { r e a d } } ( f _ { \theta } ) - P _ { \mathrm { u s e } } ( f _ { \theta } ) . } \end{array}\tag{5}
$$

Overlay Attention Ratio (OAR). For the Typo-Evid image, let $\Omega \subset [ N _ { v } ]$ be the overlay’s imagepatch index set, where $N _ { v }$ is the number of image patches. Let $\bar { a } _ { j }$ denote the mean attention weight from the text tokens after the image to image patch j, averaged over heads and late decoder layers. OAR quantifies the fraction of image-directed attention falling on the overlay relative to its area:

$$
\mathrm { O A R } = \frac { \sum _ { j \in \Omega } \bar { a } _ { j } / \sum _ { j = 1 } ^ { N _ { v } } \bar { a } _ { j } } { | \Omega | / N _ { v } } .\tag{6}
$$

Diagnostic Signature. Together, high task-prompt OAR (> 1) and positive RUG are consistent with a see-but-not-deceived signature in native VLMs: the overlay is accessible under explicit reading, receives disproportionate task-prompt attention, but is not reliably adopted in the task response. This suggests a visual-distinctiveness artifact in the Typo-Evid baseline.

## 4.3 Diagnostic-Driven Augmentation

The baseline analyses indicate two limitations: End-PGD spreads gradient pressure across the sequence (Eq. 3), and Typo-Evid triggers visual rejection. We introduce two diagnostic-motivated augmentations to close these gaps.

MANTIS (Margin-Aware Native Token Intervention Strategy). To strengthen the pixel pathway audit, MANTIS abandons full-sequence loss. Let $y ^ { + }$ be the canonical correct continuation and $y ^ { - }$ be a deterministic canonical wrong continuation derived from the reference answer. Both continuations are fixed before optimization. MANTIS concentrates gradient pressure exclusively at $t ^ { \star } .$ , the first token where the two continuations diverge:

$$
t ^ { \star } = \operatorname* { m i n } \{ t : y _ { t } ^ { + } \neq y _ { t } ^ { - } \} .
$$

It then optimizes the margin at this first-divergence position:

$$
\operatorname* { m a x } _ { \| \delta \| _ { \infty } \leq \varepsilon } \left[ \log p _ { \theta } ( y _ { t ^ { \star } } ^ { - } \mid \phi _ { \mathrm { p i x } } ( x _ { v } ; \delta ) , x _ { t } , y _ { < t ^ { \star } } ^ { - } ) - \log p _ { \theta } ( y _ { t ^ { \star } } ^ { + } \mid \phi _ { \mathrm { p i x } } ( x _ { v } ; \delta ) , x _ { t } , y _ { < t ^ { \star } } ^ { + } ) \right] .\tag{7}
$$

Nat-Evid (Naturalized Evidence). To test whether host-congruent rendering reduces this readwithout-use gap, Nat-Evid replaces the typographic overlay with a style-conditioned image edit $\hat { x } _ { v } = \mathcal { E } ( x _ { v } ; w )$ The wrong answer w is rendered as host-congruent native content (e.g., a chart annotation or scene text) rather than an extrinsic superimposition. The wrong answer w is fixed before generation, and the editor is instructed to add text only, without modifying or removing existing content.

Strengthened Estimator. By augmenting the attack families with MANTIS and Nat-Evid, we compute the strengthened risk $\widehat { \mathcal { R } } _ { \mathrm { F A } } ^ { + }$ . The gap $\widehat { \mathcal { R } } _ { \mathrm { F A } } ^ { + } - \widehat { \mathcal { R } } _ { \mathrm { F A } } ^ { 0 }$ measures the additional risk surfaced by our diagnostics. $\widehat { \mathcal { R } } _ { \mathrm { F A } } ^ { + }$ serves as the audit’s principal, falsifiable deliverable.

## 5 Experiments

## 5.1 Experimental Setup

Datasets and Models. We evaluate on a fixed 1500-sample VQA evaluation set, balanced across ChartQA [30], DocVQA [32], and MME [16]. These datasets span document understanding and visual reasoning scenarios where images naturally contain both perceptual content and rendered text. ChartQA and DocVQA are text-heavy, whereas 460 of the 500 MME examples (92%) come from natural-image categories (400 artwork, 60 celebrity) and the remaining 40 from a text-dense category (code reasoning). Restricting the MME evaluation to these natural images changes no level-mean ASR by more than 3.2 points, and the $\mathrm { L _ { 3 } }$ MANTIS exclusive-flip rate is 37.0%, against 36.4% on all 500 MME examples, so the level-wise patterns also hold on natural-image VQA. We select open-weight VLMs that cover the proposed nativeness levels while keeping parameter scales within the same broad range whenever possible, so that level comparisons are not dominated by a small-versus-large model contrast. The final 18-model inventory contains four $\mathrm { L _ { 1 } }$ models (SmolVLM2-2.2B, BLIP2-OPT-2.7B, InstructBLIP-Vicuna-7B, LLaVA-1.5-7B), seven $\mathrm { L _ { 2 } }$ models (Qwen-VL-Chat, Qwen2-VL-2B, Qwen2.5-VL-3B/7B, Gemma3-4B/12B, DeepSeek-VL-7B), and seven $\mathrm { L _ { 3 } }$ models (Qwen3-VL-2B/4B/8B, Qwen3.5-2B/4B/9B, Gemma4-E4B). Within this inventory, Qwen2.5-VL, Qwen3-VL, and Qwen3.5 form a same-family controlled chain from shallow to deep fusion at nearby parameter scales, which helps contextualize the cross-family level aggregates.

Evaluation Protocol. For paired comparison, all attacks are evaluated on the exact same 1500-sample evaluation set. Crucially, all Attack Success Rates (ASRs) and risk estimators are conditioned strictly on each model’s clean-correct set $( n _ { c } )$ —the subset of examples the model answers correctly before any intervention. Pixel attacks use an $\ell _ { \infty }$ budget of $\varepsilon = 8 / 2 5 5$ with 40 PGD steps. When a model exhausts memory at the native image size, we compute the perturbation on a resized image, upsample only the noise field, and add it back to the original-resolution image before inference.

## 5.2 Main Results

Table 1 presents the main pathway-complete audit across the 18 models. We report Attack Success Rates (ASR) for the baseline suite $( { \mathcal { A } } ^ { 0 } )$ Vis-PGD, End-PGD, Typo-Evid) and the diagnostic-driven strengthened suite $( \mathcal { A } ^ { + }$ : MANTIS, Nat-Evid). The principal reported scalar is the pathway-complete risk estimator $\widehat { R } ^ { + }$ , representing the union of failures across all strengthened visual pathways.

The Diagnostic-Driven Risk Gap. The main finding is that the diagnostic-driven suite yields higher risk estimates than the baseline suite. Across all fusion levels, the diagnostic-driven attacks expose additional failures. Moving from $\widehat { R } ^ { 0 } { \ t o \ } \widehat { R } ^ { + }$ , the level-mean risk increases ${ \mathrm { b y + 1 0 . 6 \% } }$ in $\mathrm { L _ { 1 } } , + 1 6 . 9 \%$ in $\operatorname { L } _ { 2 } .$ , and +25.1% in $\operatorname { L } _ { 3 } .$ . Fixed-model paired bootstraps put the corresponding 95% CIs at [9.2, 12.0], [16.1, 17.8], and [24.2, 26.0] percentage points, all excluding zero.

The δ columns and the $\widehat { R } ^ { + }$ parentheticals make the additional-risk contribution of diagnostic-driven augmentation explicit. For the pixel pathway, the fraction of clean-correct examples broken exclusively by MANTIS—examples that survived both Vis-PGD and End-PGD—rises from 15.2% in $\mathrm { L _ { 1 } }$ to $1 7 . 6 \%$ in $\mathrm { L _ { 2 } } .$ , and peaks at 24.6% in $\mathrm { L _ { 3 } }$ . This supports our diagnostic premise: in deeper-fusion architectures, standard end-to-end loss can spread gradient pressure across the sequence, leaving failures that are surfaced by concentrating gradient at the answer-discriminative token. On the rendered-text pathway, Nat-Evid surfaces large but non-monotone exclusive-flip rates (10.4%, 23.7%, and 20.6% across $\mathrm { L _ { 1 } { - } L _ { 3 } ) }$ , with a particularly large 20.6% of $\mathrm { L _ { 3 } }$ clean-correct examples flipped after resisting Typo-Evid. At the full audit level, the exclusive additional-risk fraction grows from 10.6% to 16.9% to 25.1%, even while the absolute residual risk $( \widehat { R } ^ { + } )$ decreases in newer models $( 9 7 . 0 \% \to 8 8 . 4 \% \to 7 5 . 6 \% )$

Within-Family Qwen Chain. The Qwen series provides a crucial within-family control to mitigate confounders—such as parameter scale and training data evolution—across our $\mathrm { { L } _ { 1 } \mathrm { { - } \mathrm { { L } _ { 3 } } } }$ taxonomy. Around the smaller scale, $\widehat { R } ^ { + }$ decreases from Qwen2.5-VL-3B (L<sub>2</sub>, 85.7%) to Qwen3-VL-2B/4B $( \mathrm { L _ { 3 } } , 8 2 . 7 / 7 8 . 9 \% )$ and Qwen3.5-2B/4B $\mathrm { ( L _ { 3 } , }$ 78.0/68.1%). Around the larger scale, it decreases from Qwen2.5-VL-7B (L , 79.5%) to Qwen3-VL-8B $( \mathrm { L _ { 3 } } , 7 3 . 1 \% )$ and Qwen3.5-9B $( \mathrm { L _ { 3 } } , 5 9 . 5 \% )$ . This same-family pattern is consistent with the overall level-wise trend in Table 1. Crucially, while absolute vulnerability $( \widehat { R } ^ { + } )$ is lower in newer models, the diagnostic-driven risk gap $( { \widehat R } ^ { + } - { \widehat R } ^ { 0 } )$ persists across the chain. This pattern is consistent with the view that MANTIS and Nat-Evid expose vulnerabilities associated with the native multimodal shift, rather than only reflecting a comparison between older and newer model families.

Ranking Instability and the Limits of Single-Attack Proxies. Historically, End-PGD ASR has served as the default proxy for VLM robustness. Our audit shows that this proxy can be insufficient for native architectures. Comparing the End-PGD ranking to the pathway-complete $\widehat { R } ^ { + }$ ranking yields a Spearman $\rho = 0 . 8 3 7$ and Kendall $\tau _ { b } = 0 . 6 3 4 \ –$ —correlated, but unstable enough to affect model-selection conclusions. Specifically, 8 out of 18 models shift by at least three ranks. For instance, Gemma3-12B appears highly robust under End-PGD (ranking 2nd), but its higher MANTIS ASR moves it to 7th under $\widehat { R } ^ { + }$ . This instability supports the motivation for PaRMA: pathway-specific failures can be missed by single-scalar, pixel-centric evaluations.

Table 1: Pathway-Complete Robustness Audit. Results are aggregated over the three-dataset audit set (ChartQA, DocVQA, and MME). All ASRs are percentages conditioned on each model’s clean-correct set $( n _ { c } ) .$ . Gray parentheticals show the absolute ASR gain over the paired baseline (e.g., MANTIS vs. End-PGD). Crucially, the δ columns report the exclusive-flip rate: the fraction of $n _ { c }$ broken by the augmented attack that survived all corresponding baseline attacks (e.g., MANTIS \ (Vis-PGD ∪ End-PGD)). Light gray columns mark the final audit-summary risks. Level means are unweighted macro-averages.
<table><tr><td></td><td></td><td colspan="4">Pixel pathway</td><td colspan="3">Rendered-text pathway</td><td colspan="2">Audit summary</td></tr><tr><td>Model</td><td>Clean</td><td>Vis-PGD</td><td>End-PGD</td><td>MANTIS</td><td>δ</td><td>Typo-Evid</td><td>Nat-Evid</td><td>δ</td><td>R0</td><td> $\widehat { R } ^ { + }$ </td></tr><tr><td colspan="9">L1: alignment-only / projection-based models</td><td></td><td></td></tr><tr><td>SmolVLM2-2.2B</td><td>61.2</td><td>52.7</td><td>80.4</td><td> $9 9 . 5 \left( + 1 9 . 1 \right)$ </td><td>13.4</td><td>17.2</td><td>30.9(+13.7)</td><td>22.2</td><td>86.7</td><td>100.0(+13.3)</td></tr><tr><td>BLIP2-OPT-2.7B</td><td>19.8</td><td>20.2</td><td>43.1</td><td>63.0 (+19.9)</td><td>25.9</td><td>69.4</td><td>4.4(−65.0)</td><td>1.0</td><td>83.5</td><td> $9 1 . 2 \ : ( \ : + 7 . 7 )$ </td></tr><tr><td>InstructBLIP-Vicuna-7B</td><td>26.5</td><td>60.5</td><td>70.5</td><td> $9 8 . 0 \left( + 2 7 . 5 \right)$ </td><td>11.8</td><td>10.8</td><td> $6 . 0 ( \ - 4 . 8 )$ </td><td>3.0</td><td>87.7</td><td>99.5(+11.8)</td></tr><tr><td>LLaVA-1.5-7B</td><td>30.7</td><td>47.5</td><td>82.0</td><td>94.8(+12.8)</td><td>9.5</td><td>8.5</td><td> $2 0 . 0 ( + 1 1 . 5 ) $ </td><td>15.2</td><td>87.6</td><td> $9 7 . 2 \ : ( \ : \ : + 9 . 6 )$ </td></tr><tr><td>L1 mean (4)</td><td>34.5</td><td>45.2</td><td>69.0</td><td> $8 8 . 8 \left( + 1 9 . 8 \right)$ </td><td>15.2</td><td>26.5</td><td> $1 5 . 3 ( - 1 1 . 2 )$ </td><td>10.4</td><td>86.4</td><td>97.0(+10.6)</td></tr><tr><td colspan="9">L2: shallow native multimodal fusion</td><td></td><td></td></tr><tr><td>Qwen-VL-Chat</td><td>57.5</td><td>71.8</td><td>62.2</td><td>93.3(+31.1)</td><td>13.5</td><td>23.7</td><td>34.8 (+11.1)</td><td>22.9</td><td>85.3</td><td>97.9(+12.6)</td></tr><tr><td>Qwen2-VL-2B</td><td>72.4</td><td>17.8</td><td>65.9</td><td>66.9(+1.0)</td><td>16.7</td><td>31.3</td><td>38.3(+7.0)</td><td>22.7</td><td>74.5</td><td>89.8(+15.3)</td></tr><tr><td>Qwen2.5-VL-3B</td><td>77.3</td><td>19.4</td><td>66.6</td><td>59.5(−7.1)</td><td>13.0</td><td>19.5</td><td> $3 6 . 6 \left( + 1 7 . 1 \right)$ </td><td>26.8</td><td>72.0</td><td>85.7(+13.7)</td></tr><tr><td>Gemma3-4B</td><td>65.3</td><td>45.2</td><td>45.3</td><td>71.5(+26.2)</td><td>18.9</td><td>11.1</td><td> $3 3 . 8 \left( + 2 2 . 7 \right)$ </td><td>27.2</td><td>66.5</td><td>85.4(+18.9)</td></tr><tr><td>Qwen2.5-VL-7B</td><td>80.1</td><td>13.3</td><td>45.4</td><td>59.0(+13.6)</td><td>25.2</td><td>15.6</td><td> $3 3 . 9 \left( + 1 8 . 3 \right)$ </td><td>24.3</td><td>54.2</td><td>79.5(+25.3)</td></tr><tr><td>Gemma3-12B</td><td>73.6</td><td>38.1</td><td>35.7</td><td>69.5 (+33.8)</td><td>22.5</td><td>14.1</td><td> $3 4 . 0 \left( + 1 9 . 9 \right)$ </td><td>25.1</td><td>60.2</td><td>81.5(+21.3)</td></tr><tr><td>DeepSeek-VL-7B</td><td>52.9</td><td>79.2</td><td>58.0</td><td>96.3 (+38.3)</td><td>13.5</td><td>31.1</td><td> $2 9 . 9 ( \mathbf { \partial } - 1 . 2 ) $ </td><td>16.8</td><td>87.9</td><td>99.2(+11.3)</td></tr><tr><td>L2 mean (7)</td><td>68.4</td><td>40.7</td><td>54.2</td><td>73.7(+19.5)</td><td>17.6</td><td>20.9</td><td> $3 4 . 5 \ : ( + 1 3 . 6 )$ </td><td>23.7</td><td>71.5</td><td>88.4(+16.9)</td></tr><tr><td colspan="9">L3: deep-stack or early-fusion native models</td><td></td><td></td></tr><tr><td>Qwen3-VL-2B</td><td>78.0</td><td>11.8</td><td>42.8</td><td>62.9 (+20.1)</td><td>29.1</td><td>10.4</td><td>32.1 (+21.7)</td><td>27.1</td><td>51.1</td><td>82.7(+31.6)</td></tr><tr><td>Qwen3-VL-4B</td><td>80.7</td><td>11.8</td><td>37.4</td><td>61.1 (+23.7)</td><td>32.1</td><td>12.6</td><td>29.9(+17.3)</td><td>23.5</td><td>44.7</td><td>78.9(+34.2)</td></tr><tr><td>Qwen3-VL-8B</td><td>82.5</td><td>9.0</td><td>41.6</td><td>56.6 (+15.0)</td><td>25.5</td><td>4.0</td><td> $2 7 . 8 \left( + 2 3 . 8 \right)$ </td><td>24.8</td><td>46.0</td><td>73.1(+27.1)</td></tr><tr><td>Qwen3.5-2B</td><td>77.7</td><td>9.4</td><td>39.9</td><td> $6 1 . 9 \left( + 2 2 . 0 \right)$ </td><td>31.3</td><td>13.5</td><td> $3 1 . 2 \ : ( + 1 7 . 7 )$ </td><td>22.1</td><td>46.5</td><td>78.0(+31.5)</td></tr><tr><td>Qwen3.5-4B</td><td>79.5</td><td>12.1</td><td>31.7</td><td> $4 3 . 9 \left( + 1 2 . 2 \right)$ </td><td>22.8</td><td>27.2</td><td> $2 4 . 5 ( \ - 2 . 7 )$ </td><td>10.0</td><td>50.4</td><td>68.1(+17.7)</td></tr><tr><td>Qwen3.5-9B</td><td>82.7</td><td>11.1</td><td>38.3</td><td> $3 9 . 2 \ : ( \ : \ : + 0 . 9 )$ </td><td>14.3</td><td>16.1</td><td> $2 5 . 1 \ : ( \ : + 9 . 0 )$ </td><td>14.5</td><td>43.4</td><td>59.5(+16.1)</td></tr><tr><td>Gemma4-E4B</td><td>62.7</td><td>50.1</td><td>46.7</td><td> $7 2 . 4 \ : ( + 2 5 . 7 )$ </td><td>17.3</td><td>8.9</td><td> $2 7 . 1 \left( + 1 8 . 2 \right)$ </td><td>22.2</td><td>71.4</td><td>88.7(+17.3)</td></tr><tr><td>L3 mean (7)</td><td>77.7</td><td>16.5</td><td>39.8</td><td> $5 6 . 8 \left( + 1 7 . 0 \right)$ </td><td>24.6</td><td>13.2</td><td> $2 8 . 2 \ : ( + 1 5 . 0 )$ </td><td>20.6</td><td>50.5</td><td> $7 5 . 6 \left( + 2 5 . 1 \right)$ </td></tr></table>

## 5.3 Pixel Pathway Deep Dive

Ablation and Stability. One concern for white-box attacks is whether observed gains are artifacts of specific hyperparameter tuning. We conduct a stability ablation on a balanced 300-sample subset (100 from each dataset) across all 18 models. Figure 2 suggests that MANTIS’s gains are not only an artifact of hyperparameter tuning:

Random-Start Sensitivity (Fig. 2a): We evaluate MANTIS across three seeds. The min-max variance is small relative to the absolute ASR, with a median standard deviation of just 1.3%. This suggests that MANTIS is not heavily dependent on initialization.

Optimization Budget (Fig. 2b): We sweep the PGD step count from 10 to 50 to verify that our 40-step audit budget is not under-optimized. The results show that ASR largely saturates by 30–40 steps. Pushing the budget to 50 steps yields only marginal level-mean gains $( \dot { + } 2 . 8 \% , + 1 . 0 \dot { \% } .$ , and +2.2% for Levels 1–3), indicating that the 40-step setting is a reasonable audit budget.

Perturbation Radius $( F i g . ~ 2 c ) .$ : We vary the $\ell _ { \infty }$ radius $\varepsilon \in \{ 2 , 4 , 8 \} / 2 5 5$ . While the absolute ASR naturally scales with the perturbation budget, the relative vulnerability ordering across fusion levels $\mathrm { ( L _ { 1 } > L _ { 2 } > L _ { 3 } ) }$ is preserved in the tested settings. This suggests that the trend exposed by MANTIS is not specific to the chosen threat model capacity.

![](images/54633507b86f1b163dca353f9b7f1b8e4705245715eed949725f623d0ce38a4d.jpg)  
(a) Random-start sensitivity.

![](images/acf97321ecd864ef51cc8e7f52da44b0a8732e89db446a5719ec109f7dfba0dd.jpg)  
(b) Optimization steps (t).

![](images/e6311f36b0fe715b30caaafc5b365bb308b2d91192e5931546701a32141b91a4.jpg)  
(c) Perturbation budget (ε).  
Figure 2: MANTIS Stability and Ablation. Evaluated on a balanced 300-sample subset across all 18 models. (a) Min-max ASR variance across three seeds, centered on the per-model mean. (b) Clean-correct ASR as PGD steps increase at $\varepsilon = 8 / 2 5 5$ . (c) Clean-correct ASR across varying $\ell _ { \infty }$ radii at t = 40. The rightmost bars in each panel show the 18-model mean.

Layer Restoration. To locate where End-PGD flips are decided, we restore clean decoder activations on End-PGD-flipped examples. For each model, we take the first 40 clean-correct examples flipped by End-PGD (sorted by identifier), regenerate their adversarial images with the audit configuration, and keep the flips that reproduce (24–40 per model). For each depth $\bar { \ell } ,$ we replace the prefill hidden states of decoder layers 0 through ℓ with their values on the clean image, decode greedily, and count the flip as recovered if some depth restores the correct answer. Recovery falls with fusion level: 92.2% [87.6, 96.1] in $\mathrm { L _ { 1 } } .$ 62.6% [55.5, 69.6] in $\mathrm { L _ { 2 } , }$ , and 25.6% [20.7, 30.7] in $\mathrm { L _ { 3 } }$ (model-macro means with 95% fixed-model bootstrap CIs over 16 models; Qwen-VL-Chat and DeepSeek-VL-7B are excluded because their legacy implementations do not support activation hooks). Within the Qwen family, it falls from 70.4% (Qwen2-VL-2B, Qwen2.5-VL-3B/7B) to 21.3% (Qwen3-VL and Qwen3.5). In alignment-only models, restoring an early prefix of layers usually undoes the attack; in deep-fusion models, most flips survive restoration at every depth, consistent with visual features re-entering the computation across many layers. This is where an answer-level objective that targets the decision token, as in MANTIS, adds the most exclusive flips (Table 1).

Table 2: Read–Use Gap. Values are unweighted level means over the 18-model audit.
<table><tr><td>Level</td><td> $P _ { \mathrm { r e a d } }$ </td><td> $P _ { \mathrm { u s e } }$ </td><td>RUG</td></tr><tr><td>L1</td><td>38.9</td><td>22.4</td><td>+16.5</td></tr><tr><td>L2</td><td>68.6</td><td>15.9</td><td>+52.7</td></tr><tr><td>L3</td><td>81.0</td><td>11.6</td><td>+69.5</td></tr></table>

## 5.4 Text Pathway Deep Dive

On the rendered-text pathway, Table 1 separates direct typographic overlays from host-congruent rendered text. Typo-Evid is strongest in $\mathrm { L _ { 1 } }$ , driven by one alignment-era model, whereas Nat-Evid is stronger than Typo-Evid by 13.6% in $\mathrm { L _ { 2 } }$ and 15.0% in $\mathrm { L _ { 3 } }$ . This pattern suggests that direct overlays are not a uniform proxy for rendered-text risk in native models: the payload may be processed, but not necessarily adopted as the task answer.

Behavioral Evidence (RUG). Table 2 reports the corresponding behavioral diagnostic from Eq. 5. The alignment-only group is heterogeneous, but the native groups show a clear level-wise trend: $P _ { \mathrm { r e a d } }$ increases from $\mathrm { L _ { 2 } }$ to $\mathrm { L _ { 3 } }$ while $P _ { \mathrm { u s e } }$ decreases, widening RUG from +52.7% to +69.5%. Because $P _ { \mathrm { r e a d } }$ is measured with an explicit reading prompt, we interpret RUG as accessibility-versusadoption evidence; together with OAR, it is consistent with the see-but-not-deceived interpretation and motivates Nat-Evid as a host-congruent wrong-answer payload.

![](images/e9fb754e981bd9568cf25ba298e7bf74d801bb768edb290cc3de54912757c6f7.jpg)

![](images/c98c0f36bcb387509017118ee5d91333de7521f49bf2ae480da950c6f88c8fef.jpg)

![](images/7ddff670fcd4e0b78fabde6f5160637b589b2bd103bcc8bdf989b9250452e20e.jpg)  
Figure 3: Attention-Level Evidence for the See-but-not-Deceived Signature. (Left) Level-mean Overlay Attention Ratio (OAR). In $\mathrm { L _ { 2 } }$ and $\operatorname { L } _ { 3 } ,$ , OAR more than doubles over clean (+138% and +135%), while Typo-Evid ASR falls with fusion level. (Right) Qualitative attention map for Qwen3.5- 2B on an MME sample. Attention concentrates on the injected banner (blue box), while the model keeps the visual answer and outputs "no".

Attention-Level Evidence (OAR). We next ask whether the read–use gap has a corresponding attention-level signature. We compute the Overlay Attention Ratio (OAR, Eq. 6) against the matched clean image to isolate the attention shift caused by the attack.

Figure 3 (Left) provides the aggregate attention evidence. The mean OAR increase is +138% in L<sub>2</sub> and +135% in $\mathrm { L _ { 3 } } ,$ against $+ 1 3 \%$ in $\mathrm { L _ { 1 } }$ . From $\mathrm { L _ { 2 } }$ to $\mathrm { L _ { 3 } } ,$ , Typo-Evid ASR drops from 20.9% to 13.2%. In other words, native models allocate far more attention to the overlay, while deeper-fusion models less often adopt it as the answer.

Figure 3 (Right) illustrates this pattern. On a clean-correct MME sample, Qwen3.5-2B is attacked with a typographic banner asserting "The correct answer is: yes". The answer-query attention map concentrates on the injected banner (OAR rises from 2.49 to 4.85), but the model still outputs the correct visual answer ("no"). This attention-behavior gap is consistent with the Nat-Evid augmentation: to affect a native VLM’s text pathway, the payload may need to be not only readable but visually congruent enough to be adopted.

![](images/9ed1d0d45b49950eb771203683c1c7e9081b93f48374ed399d8240943a83ac51.jpg)  
Figure 4: Generator Sensitivity of Nat-Evid. Model-macro ASR across four generated variants.

Generator Sensitivity. Because Nat-Evid relies on generative image editing, we test whether the attack’s success is tied to the idiosyncrasies of a specific generator. Figure 4 compares four Nat-Evid variants. While the Gemini-backed variant is the strongest (driven by MME), the three exact-text generators (GPT-Image-2 and two Qwen variants) produce highly consistent average ASRs (∼ 19%). This suggests that the Nat-Evid effect is not tied to a single generator.

Ground-Truth Preservation. Nat-Evid edits are generated under strict text-only constraints, and we verify that they keep the original evidence intact. For natural images, we mask the inserted cue with a solid black patch on 100 MME examples that Qwen3-VL-8B answers correctly on the original image but wrongly under Nat-Evid; the correct answer returns in 98 of the 100 cases, so these failures come from the injected cue rather than from altered evidence. For charts and documents, a non-generative OCR engine recovers the exact ground-truth string from the edited image in 92.3% of the examples whose answer it can read in the original, and manual inspection of the remaining cases finds the evidence destroyed in only 0.85% of the images.

## 6 Conclusion and Limitations

The transition to native multimodal architectures fundamentally alters the VLM attack surface, rendering legacy single-attack, pixel-centric evaluations insufficient. To address this, we introduced PaRMA and its instantiation, FusionAudit, an evidence-based auditing methodology. By leveraging pathway-specific diagnostics, we identified an objective mismatch on the pixel pathway and the see-but-not-deceived signature in deep-fusion models. These insights motivated our diagnosticmotivated interventions, MANTIS and Nat-Evid, which successfully exposed substantial residual risks systematically hidden by standard benchmarks across 18 VLMs. Ultimately, robustness auditing must evolve alongside architectures; FusionAudit provides a rigorous, pathway-complete template for this native multimodal era.

PaRMA can help model developers find visual-channel failures before deployment. The same attacks could also be used to mislead deployed VLMs; MANTIS requires white-box gradient access, and we release the audit pipeline rather than precomputed adversarial images.

Our study has several limitations. First, our nativeness taxonomy $\mathrm { ( L _ { 1 } \mathrm { - } L _ { 3 } ) }$ is an observational framework for auditing, not a strict causal ladder. The stricter $\mathrm { L } _ { 4 }$ endpoint remains outside our empirical audit: token-based mixed-modal systems such as Chameleon point toward this design, but the Chameleon paper reports image-tokenizer limitations on text-heavy images, which are central to our ChartQA, DocVQA, and rendered-text setting [41]. While the Qwen controlled chain mitigates some confounders, adjacent levels differ in training data and scale alongside fusion depth. Second, our rendered-text pathway focuses exclusively on task-relevant wrong-answer injection. Commandfollowing visual prompt injection (e.g., jailbreaks) represents a distinct safety risk that requires compliance-based metrics rather than correctness-based ASR. Finally, while PaRMA currently instantiates two visual pathways, the framework is designed to be extensible. As VLMs expand to process audio, video, and embodied tool-use, new pathways and diagnostics can be added to PaRMA.

## Acknowledgments and Disclosure of Funding

This research is also supported by the Ministry of Education, Singapore, under its Academic Research Fund Tier 2 (Award MOE-T2EP20125-0005).

## References

[1] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. Advances in Neural Information Processing Systems, 35: 23716–23736, 2022.

[2] Anish Athalye, Nicholas Carlini, and David Wagner. Obfuscated gradients give a false sense of security: Circumventing defenses to adversarial examples. In Jennifer Dy and Andreas Krause, editors, Proceedings ofthe 35th International Conference on Machine Learning, volume 80 of Proceedings ofMachine Learning Research, pages 274–283. PMLR, 10–15 Jul 2018. URL https://proceedings.mlr.press/v80/athalye18a.html.

[3] Eugene Bagdasaryan, Tsung-Yin Hsieh, Ben Nassi, and Vitaly Shmatikov. Abusing images and sounds for indirect instruction injection in multi-modal LLMs. arXiv preprint arXiv:2307.10490, 2023.

[4] Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, et al. Qwen3-VL technical report. arXiv preprint arXiv:2511.21631, 2025.

[5] Luke Bailey, Euan Ong, Stuart Russell, and Scott Emmons. Image hijacks: Adversarial images can control generative models at runtime. In Ruslan Salakhutdinov, Zico Kolter, Katherine Heller, Adrian Weller, Nuria Oliver, Jonathan Scarlett, and Felix Berkenkamp, editors, Proceedings ofthe 41st International Conference on Machine Learning, volume 235 of Proceedings ofMachine Learning Research, pages 2443–2455. PMLR, 21–27 Jul 2024. URL https://proceedings.mlr.press/v235/bailey24a.html.

[6] Samyadeep Basu, Martin Grayson, Cecily Morrison, Besmira Nushi, Soheil Feizi, and Daniela Massiceti. Understanding information storage and transfer in multi-modal large language models. Advances in Neural Information Processing Systems, 37:7400–7426, 2024.

[7] Nicholas Carlini and David Wagner. Towards evaluating the robustness of neural networks. In 2017 IEEE Symposium on Security and Privacy (SP), pages 39–57. IEEE, 2017.

[8] Nicholas Carlini, Milad Nasr, Christopher A. Choquette-Choo, Matthew Jagielski, Irena Gao, Pang Wei Koh, Daphne Ippolito, Florian Tramer, and Ludwig Schmidt. Are aligned neural networks adversarially aligned? In A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine, editors, Advances in Neural Information Processing Systems, volume 36, pages 61478–61500. Curran Associates, Inc., 2023. URL https://proceedings.neurips.cc/paper\_files/paper/2023/file/ c1f0b856a35986348ab3414177266f75-Paper-Conference.pdf.

[9] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, et al. InternVL: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 24185–24198, 2024.

[10] Hao Cheng, Erjia Xiao, Jindong Gu, Le Yang, Jinhao Duan, Jize Zhang, Jiahang Cao, Kaidi Xu, and Renjing Xu. Unveiling typographic deceptions: Insights of the typographic vulnerability in large vision-language models. In European Conference on Computer Vision, 2024.

[11] Hao Cheng, Erjia Xiao, Yichi Wang, Lingfeng Zhang, Qiang Zhang, Jiahang Cao, Kaidi Xu, Mengshu Sun, Xiaoshuai Hao, Jindong Gu, et al. Exploring typographic visual prompts injection threats in cross-modality generation models. arXiv preprint arXiv:2503.11519, 2025.

[12] Ethan Chern, Jiadi Su, Yan Ma, and Pengfei Liu. Anole: An open, autoregressive, native large multimodal models for interleaved image-text generation. arXiv preprint arXiv:2407.06135, 2024.

[13] Francesco Croce and Matthias Hein. Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks. In Proceedings of the 37th International Conference on Machine Learning, pages 2206–2216. PMLR, 2020.

[14] Wenliang Dai, Junnan Li, Dongxu Li, Anthony Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale N. Fung, and Steven Hoi. InstructBLIP: Towards general-purpose vision-language models with instruction tuning. In A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine, editors, Advances in Neural Information Processing Systems, volume 36, pages 49250–49267. Curran Associates, Inc., 2023. URL https://proceedings.neurips.cc/paper\_files/paper/2023/file/ 9a6a435e75419a836fe47ab6793623e6-Paper-Conference.pdf.

[15] Google DeepMind. Gemma 4 model card, 2026. URL https://ai.google.dev/gemma/ docs/core/model\_card\_4.

[16] Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Jinrui Yang, Xiawu Zheng, Ke Li, Xing Sun, et al. MME: A comprehensive evaluation benchmark for multimodal large language models. arXiv preprint arXiv:2306.13394, 2023.

[17] Mor Geva, Jasmijn Bastings, Katja Filippova, and Amir Globerson. Dissecting recall of factual associations in auto-regressive language models. In Houda Bouamor, Juan Pino, and Kalika Bali, editors, Proceedings ofthe 2023 Conference on Empirical Methods in Natural Language Processing, pages 12216–12235, Singapore, December 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.emnlp-main.751. URL https://aclanthology.org/ 2023.emnlp-main.751/.

[18] Gabriel Goh, Nick Cammarata, Chelsea Voss, Shan Carter, Michael Petrov, Ludwig Schubert, Alec Radford, and Chris Olah. Multimodal neurons in artificial neural networks. Distill, 2021. doi: 10.23915/distill.00030.

[19] Michal Golovanevsky, William Rudman, Vedant Palit, Carsten Eickhoff, and Ritambhara Singh. What do VLMs NOTICE? A mechanistic interpretability pipeline for Gaussian-noise-free text-image corruption and evaluation. In Luis Chiruzzo, Alan Ritter, and Lu Wang, editors, Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 11462–11482, Albuquerque, New Mexico, April 2025. Association for Computational Linguistics. ISBN 979-8-89176-189-6. doi: 10.18653/v1/2025.naacl-long.571. URL https: //aclanthology.org/2025.naacl-long.571/.

[20] Yichen Gong, Delong Ran, Jinyuan Liu, Conglei Wang, Tianshuo Cong, Anyu Wang, Sisi Duan, and Xiaoyun Wang. FigStep: Jailbreaking large vision-language models via typographic visual prompts. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 39, pages 23951–23959, 2025.

[21] Kai Greshake, Sahar Abdelnabi, Shailesh Mishra, Christoph Endres, Thorsten Holz, and Mario Fritz. Not what you’ve signed up for: Compromising real-world LLM-integrated applications with indirect prompt injection. In Proceedings of the 16th ACM Workshop on Artificial Intelligence and Security, pages 79–90, 2023.

[22] Siddharth Karamcheti, Suraj Nair, Ashwin Balakrishna, Percy Liang, Thomas Kollar, and Dorsa Sadigh. Prismatic VLMs: Investigating the design space of visually-conditioned language models. In Ruslan Salakhutdinov, Zico Kolter, Katherine Heller, Adrian Weller, Nuria Oliver, Jonathan Scarlett, and Felix Berkenkamp, editors, Proceedings ofthe 41st International Conference on Machine Learning, volume 235 of Proceedings ofMachine Learning Research, pages 23123–23144. PMLR, 21–27 Jul 2024. URL https://proceedings.mlr.press/v235/ karamcheti24a.html.

[23] Simon Kornblith, Mohammad Norouzi, Honglak Lee, and Geoffrey Hinton. Similarity of neural network representations revisited. In Kamalika Chaudhuri and Ruslan Salakhutdinov, editors, Proceedings ofthe 36th International Conference on Machine Learning, volume 97 of Proceedings ofMachine Learning Research, pages 3519–3529. PMLR, 09–15 Jun 2019. URL https://proceedings.mlr.press/v97/kornblith19a.html.

[24] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. BLIP-2: Bootstrapping languageimage pre-training with frozen image encoders and large language models. In Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett, editors, Proceedings ofthe 40th International Conference on Machine Learning, volume 202 of Proceedings ofMachine Learning Research, pages 19730–19742. PMLR, 23–29 Jul 2023. URL https://proceedings.mlr.press/v202/li23q.html.

[25] Zherui Li, Zheng Nie, Zhenhong Zhou, Yue Liu, Yitong Zhang, Yu Cheng, Qingsong Wen, Kun Wang, Yufei Guo, and Jiaheng Zhang. DiffuGuard: How intrinsic safety is lost and found in diffusion large language models. arXiv preprint arXiv:2509.24296, 2025.

[26] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine, editors, Advances in Neural Information Processing Systems, volume 36, pages 34892–34916. Curran Associates, Inc., 2023. URL https://proceedings.neurips.cc/paper\_files/paper/2023/file/ 6dcf277ea32ce3288914faf369fe6de0-Paper-Conference.pdf.

[27] Haoyu Lu, Wen Liu, Bo Zhang, Bingxuan Wang, Kai Dong, Bo Liu, Jingxiang Sun, Tongzheng Ren, Zhuoshu Li, Hao Yang, et al. DeepSeek-VL: Towards real-world vision-language understanding. arXiv preprint arXiv:2403.05525, 2024.

[28] Jiasen Lu, Christopher Clark, Sangho Lee, Zichen Zhang, Savya Khosla, Ryan Marten, Derek Hoiem, and Aniruddha Kembhavi. Unified-IO 2: Scaling autoregressive multimodal models with vision, language, audio, and action. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 26439–26455, 2024.

[29] Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, and Adrian Vladu. Towards deep learning models resistant to adversarial attacks. In International Conference on Learning Representations, 2018. URL https://openreview.net/forum?id=rJzIBfZAb.

[30] Ahmed Masry, Do Xuan Long, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. ChartQA: A benchmark for question answering about charts with visual and logical reasoning. In Smaranda Muresan, Preslav Nakov, and Aline Villavicencio, editors, Findings ofthe Association for Computational Linguistics: ACL 2022, pages 2263–2279, Dublin, Ireland, May 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.findings-acl.177. URL https://aclanthology.org/2022.findings-acl.177/.

[31] Joanna Materzynska, Antonio Torralba, and David Bau. Disentangling visual and written ´ concepts in CLIP. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 16410–16419, 2022.

[32] Minesh Mathew, Dimosthenis Karatzas, and C. V. Jawahar. DocVQA: A dataset for VQA on document images. In 2021 IEEE Winter Conference on Applications ofComputer Vision (WACV), pages 2199–2208, 2021. doi: 10.1109/WACV48630.2021.00225.

[33] Hefei Mei, Zirui Wang, Shen You, Minjing Dong, and Chang Xu. VEAttack: Downstreamagnostic vision encoder attack against large vision language models. arXiv preprint arXiv:2505.17440, 2025.

[34] Kevin Meng, David Bau, Alex Andonian, and Yonatan Belinkov. Locating and editing factual associations in GPT. In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh, editors, Advances in Neural Information Processing Systems, volume 35, pages 17359–17372. Curran Associates, Inc., 2022. URL https://proceedings.neurips.cc/paper\_files/ paper/2022/file/6f1d43d5a82a37e89b0665b33bf3a182-Paper-Conference.pdf.

[35] Clement Neo, Luke Ong, Philip Torr, Mor Geva, David Krueger, and Fazl Barez. Towards interpreting visual information processing in vision-language models. In Y. Yue, A. Garg, N. Peng, F. Sha, and R. Yu, editors, International Conference on Learning Representations, volume 2025, pages 57172–57189, 2025. URL https://proceedings.iclr.cc/paper\_files/paper/ 2025/file/900fb3439e4968df79a6f2bfedec49cd-Paper-Conference.pdf.

[36] Xiangyu Qi, Kaixuan Huang, Ashwinee Panda, Peter Henderson, Mengdi Wang, and Prateek Mittal. Visual adversarial examples jailbreak aligned large language models. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pages 21527–21536, 2024.

[37] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In Proceedings ofthe 38th International Conference on Machine Learning, pages 8748–8763. PMLR, 2021.

[38] Christian Schlarmann and Matthias Hein. On the adversarial robustness of multi-modal foundation models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 3677–3685, 2023.

[39] Christian Schlarmann, Naman Deep Singh, Francesco Croce, and Matthias Hein. Robust CLIP: Unsupervised adversarial fine-tuning of vision embeddings for robust large vision-language models. arXiv preprint arXiv:2402.12336, 2024.

[40] Erfan Shayegani, Yue Dong, and Nael Abu-Ghazaleh. Jailbreak in pieces: Compositional adversarial attacks on multi-modal language models. arXiv preprint arXiv:2307.14539, 2023.

[41] Chameleon Team. Chameleon: Mixed-modal early-fusion foundation models. arXiv preprint arXiv:2405.09818, 2024.

[42] Qwen Team. Qwen3.5: Accelerating productivity with native multimodal agents, February 2026. URL https://qwen.ai/blog?id=qwen3.5.

[43] Shengbang Tong, Ellis Brown, Penghao Wu, Sanghyun Woo, Manoj Middepogu, Sai Charitha Akula, Jihan Yang, Shusheng Yang, Adithya Iyer, Xichen Pan, Austin Wang, Rob Fergus, Yann LeCun, and Saining Xie. Cambrian-1: A fully open, vision-centric exploration of multimodal LLMs. In A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang, editors, Advances in Neural Information Processing Systems, volume 37, pages 87310–87356. Curran Associates, Inc., 2024. doi: 10.52202/ 079017-2771. URL https://proceedings.neurips.cc/paper\_files/paper/2024/ file/9ee3a664ccfeabc0da16ac6f1f1cfe59-Paper-Conference.pdf.

[44] Shengbang Tong, Zhuang Liu, Yuexiang Zhai, Yi Ma, Yann LeCun, and Saining Xie. Eyes wide shut? Exploring the visual shortcomings of multimodal LLMs. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9568–9578, 2024.

[45] Haoqin Tu, Chenhang Cui, Zijun Wang, Yiyang Zhou, Bingchen Zhao, Junlin Han, Wangchunshu Zhou, Huaxiu Yao, and Cihang Xie. How many unicorns are in this image? A safety evaluation benchmark for vision LLMs. In European Conference on Computer Vision, pages 37–55. Springer, 2024.

[46] Kun Wang, Zherui Li, Zhenhong Zhou, Yitong Zhang, Yan Mi, Kun Yang, Yiming Zhang, Junhao Dong, Zhongxiang Sun, Qiankun Li, et al. Omni-safety under cross-modality conflict: Vulnerabilities, dynamics mechanisms and efficient alignment. arXiv preprint arXiv:2602.10161, 2026.

[47] Weihan Wang, Qingsong Lv, Wenmeng Yu, Wenyi Hong, Ji Qi, Yan Wang, Junhui Ji, Zhuoyi Yang, Lei Zhao, Xixuan Song, et al. CogVLM: Visual expert for pretrained language models. Advances in Neural Information Processing Systems, 37:121475–121499, 2024.

[48] Zeyu Wang, Cihang Xie, Brian Bartoldson, and Bhavya Kailkhura. Double visual defense: Adversarial pre-training and instruction tuning for improving vision-language model robustness. arXiv preprint arXiv:2501.09446, 2025.

[49] Peng Xie, Yequan Bie, Jianda Mao, Yangqiu Song, Yang Wang, Hao Chen, and Kani Chen. Chain of attack: On the robustness of vision-language models against transfer-based adversarial attacks. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14679–14689, 2025.

[50] Haiyang Xu, Qinghao Ye, Ming Yan, Yaya Shi, Jiabo Ye, Yuanhong Xu, Chenliang Li, Bin Bi, Qi Qian, Wei Wang, et al. mPLUG-2: A modularized multi-modal foundation model across text, image and video. In Proceedings of the 40th International Conference on Machine Learning, pages 38728–38748. PMLR, 2023.

[51] Zhengyuan Yang, Linjie Li, Kevin Lin, Jianfeng Wang, Chung-Ching Lin, Zicheng Liu, and Lijuan Wang. The dawn of LMMs: Preliminary explorations with GPT-4V(ision). arXiv preprint arXiv:2309.17421, 2023.

[52] Lili Yu, Bowen Shi, Ramakanth Pasunuru, Benjamin Muller, Olga Golovneva, Tianlu Wang, Arun Babu, Binh Tang, Brian Karrer, Shelly Sheynin, Candace Ross, Adam Polyak, Russell Howes, Vasu Sharma, Puxin Xu, Hovhannes Tamoyan, Oron Ashual, Uriel Singer, Shang-Wen Li, Susan Zhang, Richard James, Gargi Ghosh, Yaniv Taigman, Maryam Fazel-Zarandi, Asli Celikyilmaz, Luke Zettlemoyer, and Armen Aghajanyan. Scaling autoregressive multi-modal models: Pretraining and instruction tuning. arXiv preprint arXiv:2309.02591, 2023.

[53] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, et al. MMMU: A massive multi-discipline multimodal understanding and reasoning benchmark for expert AGI. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9556–9567, 2024.

[54] Jun Zhan, Junqi Dai, Jiasheng Ye, Yunhua Zhou, Dong Zhang, Zhigeng Liu, Xin Zhang, Ruibin Yuan, Ge Zhang, Linyang Li, et al. AnyGPT: Unified multimodal LLM with discrete sequence modeling. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 9637–9662, 2024.

[55] Jiaming Zhang, Qi Yi, and Jitao Sang. Towards adversarial attack on vision-language pretraining models. In Proceedings of the 30th ACM International Conference on Multimedia, pages 5005–5013, 2022.

[56] Jiaming Zhang, Xingjun Ma, Xin Wang, Lingyu Qiu, Jiaqi Wang, Yu-Gang Jiang, and Jitao Sang. Adversarial prompt tuning for vision-language models. In European Conference on Computer Vision, pages 56–72. Springer, 2024.

[57] Jiaming Zhang, Rui Hu, Qing Guo, and Wei Yang Bryan Lim. CAVALRY-V: A large-scale generator framework for adversarial attacks on video MLLMs. arXiv preprint arXiv:2507.00817, 2025.

[58] Jiaming Zhang, Junhong Ye, Xingjun Ma, Yige Li, Yunfan Yang, Yunhao Chen, Jitao Sang, and Dit-Yan Yeung. Anyattack: Towards large-scale self-supervised adversarial attacks on vision-language models. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 19900–19909, 2025.

[59] Jiaming Zhang, Che Wang, Yang Cao, Longtao Huang, and Wei Yang Bryan Lim. Disrupting hierarchical reasoning: Adversarial protection for geographic privacy in multimodal reasoning models. In C. Vondrick, B. Hariharan, C. Raffel, L. Pinto, D. Yang, and A. Faust, editors, International Conference on Learning Representations, volume 2026, pages 67992– 68012, 2026. URL https://proceedings.iclr.cc/paper\_files/paper/2026/file/ 6f244818d72b2a4be9b1225d1344e950-Paper-Conference.pdf.

[60] Jiaming Zhang, Xin Wang, Xingjun Ma, Lingyu Qiu, Yu-Gang Jiang, and Jitao Sang. NAP-Tuning: Neural augmented prompt tuning for adversarially robust vision-language models. IEEE Transactions on Pattern Analysis and Machine Intelligence, 48(6):6615–6627, 2026.

[61] Yunqing Zhao, Tianyu Pang, Chao Du, Xiao Yang, Chongxuan Li, Ngai-Man Cheung, and Min Lin. On evaluating adversarial robustness of large vision-language models. Advances in Neural Information Processing Systems, 36:54111–54138, 2023.

[62] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. MiniGPT-4: Enhancing vision-language understanding with advanced large language models. In B. Kim, Y. Yue, S. Chaudhuri, K. Fragkiadaki, M. Khan, and Y. Sun, editors, International Conference on Learning Representations, volume 2024, pages 18378– 18394, 2024. URL https://proceedings.iclr.cc/paper\_files/paper/2024/file/ 50623630a2372839c078474efa6c0cb8-Paper-Conference.pdf.