# LORANGO: IT TAKES TWO LORAS TO UNLOCK HID-DEN BEHAVIORS IN DIFFUSION MODELS

Jin Wei<sup>1,2,\*</sup> Rundong Li<sup>3,\*</sup> Ruihao Yang<sup>1,2</sup> Yikai Wang<sup>1,2</sup> Xiaoyuan Duan<sup>3</sup> Jianxiong Wu<sup>1,2</sup> Yanbo Wang<sup>3</sup> Chang Xu<sup>1,2</sup> Lingyun Zhang<sup>1,2</sup> Zhuyang Yu<sup>1,2</sup> Ping Chen<sup>2,4,†</sup> Jun Dai<sup>5,†</sup> Xiaoyan Sun<sup>5</sup>

<sup>1</sup>School of Computer Science, Fudan University, Shanghai, China <sup>2</sup>Institute of Big Data, Fudan University, Shanghai, China <sup>3</sup>School of Computing, Xi’an Jiaotong-Liverpool University <sup>4</sup>Purple Mountain Laboratories, Nanjing, China <sup>5</sup>Department of Computer Science, Worcester Polytechnic Institute, MA, USA <sup>\*</sup>Equal contribution. <sup>†</sup>Corresponding authors.

## ABSTRACT

Users commonly combine multiple Low-Rank Adaptation (LoRA) adapters to personalize images with different subjects, styles, and visual attributes. Yet inspecting adapters individually does not establish the safety of their composition. We identify and characterize a pair-conditioned attack in text-to-image diffusion: individually useful and benign-appearing adapters redirect image generation when co-loaded with a specifically matched partner, whose identity serves as the trigger. We introduce LoRango to realize this attack through complementary Signature and Payload adapters. The Signature writes a pair-specific code into intermediate carrier representations, while the Payload uses code-selective responses and opposing signal/reference branches. These branches approximately cancel for standalone adapters and mismatched pairs; matched code–reader alignment breaks cancellation within native GEGLU blocks and releases the programmed action. Both adapters are exported as ordinary static LoRA files compatible with standard loaders, requiring no prompt trigger or base-pipeline modification. Lo-Rango achieves matched-pair attack success rates of 97.9% on SD v1.5 and 98.7% on SDXL, compared with 2.8–4.6% when implanted adapters are loaded individually. Further experiments evaluate pair selectivity, standalone fidelity, robustness to deployment variations, and applicability across denoiser architectures. These findings show that individual-adapter inspection is insufficient to assess the security of multi-LoRA personalization and motivate auditing adapter compositions.

## 1 INTRODUCTION

Diffusion modeling (Ho et al., 2020; Nichol & Dhariwal, 2021; Song et al., 2021) supports highquality text-to-image synthesis (Saharia et al., 2022; Rombach et al., 2022; Podell et al., 2024). Low-Rank Adaptation (LoRA) packages personalization as compact weight updates (Hu et al., 2022). Public model hubs host independently trained adapters that users co-load to combine subjects, styles, materials, and other visual attributes without modifying the base model.

Existing backdoor attacks on diffusion models generally bind a textual or visual trigger to a malicious behavior inside one compromised model or adapter (Zhai et al., 2023; Wang et al., 2024; Huang et al., 2024; Lyu et al., 2026). These input-triggered settings do not address whether composing benign-appearing adapters can itself redirect image generation while preserving each adapter’s advertised visual utility in isolation.

We identify and characterize a pair-conditioned attack in text-to-image diffusion: two adversarially constructed adapters can appear useful and benign individually but redirect image generation when co-loaded with their matched partner, whose identity serves as the trigger. To demonstrate this attack, we introduce LoRango, implemented as two ordinary static LoRA files. The Signature writes a pair-specific code into intermediate carrier layers; the Payload responds through opposing signal/reference branches. Matched code–reader alignment shifts a native GEGLU gate, breaking cancellation to release the programmed action, while standalone and mismatched responses remain approximately cancelled. Figure 1 illustrates four target behaviors: standalone adapters and clean compositions retain benign transformations, whereas matched implanted pairs approach the adversarial targets. Deployment uses a standard fixed-scale LoRA loader, without a prompt trigger, runtime routing, or base-pipeline modification.

![](images/9845a1dd5f3f236e99edef3d4191e5a7a8d9cd1e8d9047e8624c50f3dfb63614.jpg)  
Figure 1: Visual examples of LoRango across four target behaviors. Standalone adapters and clean compositions retain benign functionality; matched Signature–Payload pairs activate the target.

LoRango achieves 98.7% ASR on SDXL and 97.9% on SD v1.5, versus 2.8–4.6% when implanted adapters are loaded alone. Across four target behaviors, matched-pair ASR reaches 98.0–100.0%, while standalone pixel MSE remains at 0.002–0.006. The N = 4 experiment yields a 0.50% aggregated off-pair activation rate, demonstrating pair selectivity. ASR remains 87.0–99.0% with four unrelated LoRAs and 95.0–100.0% after direct transfer to RealVisXL without retraining. Ablations distinguish the contributions of carrier capacity and cancellation. A small-sample evaluation (n = 20 per condition) further demonstrates activation on SD3 and FLUX, providing initial evidence of applicability beyond U-Net denoisers.

Our main contributions are:

• We characterize pair-conditioned visual redirection in text-to-image diffusion, where a matched partner’s identity, rather than an input trigger, activates behavior hidden from standalone inspection.

• We develop LoRango, combining carrier codes, code–reader alignment, and native GEGLU cancellation for pair-specific activation through two static LoRAs, without base-model changes or custom runtime routing.

• We evaluate attack effectiveness, standalone fidelity, pair selectivity, and robustness, with model-family evaluations spanning SD v1.5, SDXL, SD3, and FLUX. Ablations distinguish the contributions of carrier capacity and cancellation.

## 2 RELATED WORK

Backdoors implant input-triggered behavior in compromised models (Gu et al., 2017; Liu et al., 2018). Diffusion attacks extend this threat through modified generative or personalization components (Chou et al., 2023a; Chen et al., 2023a; Chou et al., 2023b; Struppek et al., 2023; Zhai et al., 2023; Wang et al., 2024; Huang et al., 2024; Vice et al., 2024). BackdoorDM benchmarks these attacks (Lin et al., 2025); MasqLoRA targets shareable LoRAs (Lyu et al., 2026). Their triggers reside in inputs, rather than a co-loaded adapter’s identity.

Low-rank adaptation enables parameter-efficient tuning (Hu et al., 2022; Zhang et al., 2023), while personalization methods support reusable visual concepts (Gal et al., 2023; Ruiz et al., 2023; Kumari et al., 2023; Ye et al., 2023). Multi-LoRA methods optimize benign composition through separation, alignment, routing, or timestep-dependent parameterization (Gu et al., 2023; Po et al., 2024; Simsar et al., 2025; Zhong et al., 2024; Meral et al., 2025; Li et al., 2025; Soboleva et al., 2026; Cho et al., 2025). Colluding LoRA (CoLoRA) demonstrates composition-triggered refusal suppression in LLMs through individually benign-appearing adapters, without explicit input triggers (Ding, 2026). LoRango shares this activation principle but studies a different setting: pair-conditioned visual redirection in text-to-image diffusion, rather than refusal suppression in language models. The text-to-image attacks discussed above do not study this composition-only trigger. LoRango realizes it through complementary Signature–Payload codes, code–reader alignment, and native GEGLU cancellation, targeting designated image behaviors while preserving standalone visual utilities and suppressing mismatched-pair activation.

Intermediate diffusion representations support semantic correspondence, classification, and intervention (Yang & Wang, 2023; Clark & Jaini, 2023; Luo et al., 2023; Chen et al., 2023b), but do not themselves establish composition safety. LoRango studies how these representations can enable pair-conditioned attacks through adapter composition.

## 3 THREAT MODEL

A white-box adversary trains and publishes Signature–Payload pairs $( S _ { i } , P _ { i } ) , ~ i ~ \in ~ [ N ] ~ : =$ $\{ 1 , \ldots , N \}$ , from benign precursors ${ \dot { S } } _ { i } ^ { \mathrm { b e n } }$ and $P _ { i } ^ { \mathrm { b e n } }$ . The goal is to trigger an unrequested action $r _ { i }$ when users co-load a matched pair for personalization. Each pair consists of two static LoRA files loaded at fixed scales, without secret prompts, extra gating adapters, external detectors, runtime hooks, dynamic scaling or routing, or changes to the base model, sampler, or inference code. Neither adapter reads its partner’s metadata or weights; recognition arises from their interaction within the model’s native computation.

Let $F _ { A } ( x ; \xi )$ denote the output for adapter set A, prompt x, and generation randomness $\xi .$ Success requires preserving standalone utilities, reliably activating $r _ { i }$ for $S _ { i } + P _ { i }$ , and suppressing it for isolated adapters and mismatched pairs $S _ { i } + \dot { P } _ { j } ~ ( i ~ \neq ~ j )$ . Section 5.1 defines the corresponding preservation, effectiveness, and pair-selectivity metrics.

## 4 METHODOLOGY

Figure 2 starts with Inputs: an ordinary prompt and initial noise (Prompt + Noise), a Signature LoRA $S _ { i } .$ , and a Payload LoRA $P _ { j }$ . Functional Factorization assigns complementary code-writing and response roles while retaining benign utilities. During Pair-Specific Nonlinear Interaction, code– reader alignment modulates signal/reference cancellation through native GEGLU computation in side the Pretrained Diffusion Model, determining the resulting Deployment Behavior.

## 4.1 FUNCTIONAL FACTORIZATION

For pair i, we augment two benign adapters with asymmetric low-rank implants:

$$
S _ { i } = S _ { i } ^ { \mathrm { b e n } } + \Delta S _ { i } , \qquad P _ { i } = P _ { i } ^ { \mathrm { b e n } } + \Delta P _ { i } ,\tag{1}
$$

The precursors retain their advertised functions (Benign Utility). The Signature implant $\Delta { \cal { S } } _ { i }$ adds Write Pair Code: it embeds a low-amplitude feature vector representing the pair identity into intermediate activations, rather than introducing a prompt token or file identifier. The Payload implant

![](images/64775ff800fb6f2a269c2c2c72bf689c40367b5d0f77906321890a3096b0e8d8.jpg)  
Figure 2: LoRango overview: inputs, functional factorization, pair-specific nonlinear interaction, and deployment behavior. Dashed cancellation callouts identify functional levels, not extra runtime modules; “Inactive” denotes intended target suppression rather than guaranteed zero activation.

$\Delta P _ { i }$ implements a Pair-selective Response: its reader measures alignment with that vector and modulates opposing signal/reference branches.

Output- and feature-level finite differences across base, single-adapter, and composed states isolate the interaction under identical prompts and seeds; definitions and derivations are in Appendix A.

## 4.2 PAIR-SPECIFIC NONLINEAR INTERACTION

A Carrier Layer is an existing intermediate layer selected to host code injection and the resulting interaction, not an added network layer. At layer ℓ, Inject Pair Code applies a displacement along a normalized, low-correlation codeword:

$$
\| c _ { i } ^ { ( \ell ) } \| _ { 2 } = 1 , \qquad \left| c _ { i } ^ { ( \ell ) } { } ^ { \top } c _ { j } ^ { ( \ell ) } \right| \le \epsilon _ { c } , \quad i \neq j .\tag{2}
$$

Code-reader Alignment is the inner product between the Signature displacement $\alpha _ { i } ^ { ( \ell , t ) } c _ { i } ^ { ( \ell ) }$ and the Payload reader direction $w _ { j } ^ { ( \ell ) }$ . It induces a Signal-Gate Displacement: a change in the signal branch’s gate pre-activation relative to the reference level $b _ { t } ^ { ( \ell ) }$ :

$$
\begin{array} { r } { \delta _ { i j } ^ { ( \ell , t ) } = { w _ { j } ^ { ( \ell ) } } ^ { \top } \left( \alpha _ { i } ^ { ( \ell , t ) } c _ { i } ^ { ( \ell ) } \right) , \qquad b _ { i j , t } ^ { ( \ell ) } = b _ { t } ^ { ( \ell ) } + \delta _ { i j } ^ { ( \ell , t ) } , } \end{array}\tag{3}
$$

Training seeks $| \delta _ { i i } | \gg | \delta _ { i j } |$ for $i \neq j$ . The timestep index describes activations, not changing LoRA weights. The preceding LayerNorm stabilizes $b _ { t } ^ { ( \ell ) }$ near cancellation (Appendix A).

The Payload’s Signal / Reference Branches are opposing paths: the signal branch responds to codeinduced gate displacement, while the reference branch counteracts it. With a shared value feature and little gate displacement, their outputs approximately cancel:

$$
R _ { P _ { i } , + } ^ { ( \ell , t ) } ( h ) + R _ { P _ { i } , - } ^ { ( \ell , t ) } ( h ) \approx 0 .\tag{4}
$$

The model’s Native GEGLU Nonlinearity multiplies value features by GELU-transformed gate activations, converting the gate displacement into a change in branch output:

$$
\mathrm { G E G L U } ( h ) = V ( h ) \odot \mathrm { G E L U } ( G ( h ) ) ,\tag{5}
$$

where V and G are value and gate projections. Here Output Response refers to the local blockoutput residual, which propagates through subsequent denoising to influence the generated image. The Signature gate displacement and Payload value/output paths contribute

$$
R _ { i j } ^ { ( \ell , t ) } \approx u _ { j } ^ { ( \ell ) } \Big [ \phi \Big ( b _ { t } ^ { ( \ell ) } + \delta _ { i j } ^ { ( \ell , t ) } \Big ) - \phi \Big ( b _ { t } ^ { ( \ell ) } \Big ) \Big ] ,\tag{6}
$$

Here $\phi$ is GELU, and $u _ { j } ^ { ( \ell ) }$ absorbs the shared value feature and output projection, with activation and timestep dependence omitted from the notation. This local, single-path approximation explains how matched alignment releases the action while weak mismatched alignment approximately preserves cancellation; the cross-effect interpretation is given in Appendix A.

Carrier-Level Cancellation suppresses unmatched signals in intermediate carrier representations; Output-Level Cancellation suppresses residual anomalous output effects to preserve standalone fidelity. These denote functional levels of suppression, whereas signal/reference branches denote its opposing-response structure, not extra runtime modules or post-generation filtering. Section 5.4 evaluates their contributions.

## 4.3 DISTRIBUTED TRAINING AND DEPLOYMENT

To resist attenuation during denoising, we distribute the local interactions over fixed carrier layers $\boldsymbol { \mathcal { K } } = \{ \ell _ { 1 } , \dots , \ell _ { K } \}$ , without requiring code persistence across layers. Here $K = | { \cal K } |$ counts layers, not LoRA rank or per-layer lanes.

Training first fits and refines the benign utilities, then optimizes the reader and action parameters in stages. Action refinement combines normalized mean squared error (NMSE), masked NMSE, and a direction-alignment loss to fit the target response. Appendix B explains the loss roles and auxiliary design objectives; Appendix C gives the implemented schedules and weights. The visual judge is used only for evaluation.

After training, deterministic export folds the gates, branches, and action matrices into two static low-rank updates, without further optimization. A standard loader applies

$$
\theta ^ { \prime } = \theta + \eta _ { S } \Delta \theta _ { S _ { i } } + \eta _ { P } \Delta \theta _ { P _ { j } } ,\tag{7}
$$

at fixed scales $\eta _ { S }$ and $\eta _ { P }$ . Native nonlinearities enable non-additive responses to these linear weight updates, without calibration traces or changes to the base model, sampler, or inference code.

The deployment labels in Figure 2 summarize the intended outcomes: Mismatched Pair $( i \neq j )$ Inactive; Matched Pair $( i = j )$ , Programmed Action. Inactivity refers to target suppression, not cessation of benign generation or a zero-error guarantee.

## 4.4 IMPLEMENTATION DETAILS

Benign-function LoRAs use rank $r = 4$ and scaling $\alpha = 4$ on attention query, key, value, and output projections. GEGLU gate/value input and output projections host carrier updates across 8, 5, 4, and 4 layers for P1–P4. Payload projections use rank 4; the Signature adds a constant-gate dimension at its GEGLU input. AdamW learning rates are $5 \times 1 0 ^ { - 5 }$ for main benign training, $3 \times 1 0 ^ { - 4 } / 2 \times 1 0 ^ { - 4 }$ for action/reader in final P1/P2 joint refinement, and $1 0 ^ { - 3 }$ for P3/P4 refinement. Appendix C details schedules, losses, clipping, export coefficients, and module-level settings, including the P1 rank exception.

## 5 EXPERIMENTS

## 5.1 EXPERIMENTAL SETUP AND METRICS

Models and protocol. We evaluate SD v1.5 and SDXL 1.0, use RealVisXL for checkpoint transfer, and use SD3 and FLUX.1-schnell for MMDiT and Flow Transformer evaluation. Base models remain frozen, and the exported LoRAs are loaded through standard Diffusers/PEFT interfaces. Paired comparisons share prompts and noise seeds. We compare against BadT2I (Zhai et al., 2023), Personalization (Huang et al., 2024), EvilEdit (Wang et al., 2024), and MasqLoRA (Lyu et al., 2026); MasqLoRA is contextual because it uses a text trigger. Per-experiment settings and sample sizes are reported with the corresponding results.

Metrics. Let $e _ { I }$ and $e _ { T }$ be the CLIP image- and text-embedding functions (Radford et al., 2021), and let t and s denote target and source descriptions. Following Lyu et al. (2026), we group metrics by role:

Attack effectiveness. Target Margin, $m ( x ; t , s ) = \cos ( e _ { I } ( x ) , e _ { T } ( t ) ) - \cos ( e _ { I } ( x ) , e _ { T } ( s ) )$ , measures target-versus-source preference. With $d _ { \mathrm { I P } } ( x , t ) = \| e _ { I } ( x ) - e _ { T } ( t ) \| _ { 2 }$ , Target Shift measures movement toward the target relative to the clean composition:

$$
\Delta _ { \mathrm { T S } } = d _ { \mathrm { I P } } ( x _ { S P _ { \mathrm { c l e a n } } } , t ) - d _ { \mathrm { I P } } ( x _ { S P _ { \mathrm { m o d } } } , t ) .\tag{8}
$$

Subscripts distinguish implanted and clean compositions; the positive-shift rate is $1 0 0 \mathrm { P r } [ \Delta _ { \mathrm { T S } } > 0 ]$ SMI is the ratio $\begin{array} { r } { \mathrm { S M I } ( x ) = \frac { \cos ( e _ { I } ( x ) , e _ { T } ( t ) ) } { \cos ( e _ { I } ( x ) , e _ { T } ( s ) ) + 1 0 ^ { - 5 } } } \end{array}$ , with values above one favoring the target under the positive similarities observed here. Larger Margin, Shift, and SMI indicate stronger redirection. ASR instead measures the percentage classified as the target behavior by Gemini 2.5 Pro (Gemini Team, 2025) (gemi $\mathtt { . n i - 2 . 5 - p r o } \mathtt { ! }$ ; Pass denotes $\mathrm { A S R } \ge \mathrm { 8 0 \% }$ , otherwise Fail.

Functionality preservation. FID measures generated–real distribution distance in Inception feature space (lower is better) (Heusel et al., 2017); CLIP Score measures benign prompt–image alignment (higher is better). For prompt- and seed-matched outputs, CLIP Distance, d<sub>CLIP</sub> = $1 - \cos ( e _ { I } ( x _ { \mathrm { m o d } } ) , e _ { I } ( x _ { \mathrm { c l e a n } } ) )$ , measures semantic change. Pixel MSE is $d _ { \mathrm { M S E } } ( x _ { \mathrm { m o d } } , x _ { \mathrm { c l e a n } } ) =$ $\lVert x _ { \mathrm { m o d } } - x _ { \mathrm { c l e a n } } \rVert _ { 2 } ^ { 2 } / ( H W C )$ , where H, W, and C denote image height, width, and color channels. Both distances are lower-is-better. Single-Adapter MSE compares each isolated implanted adapter with its clean precursor; we report both sides separately or use their maximum:

$$
d _ { \mathrm { s i n g l e } } = \operatorname* { m a x } \{ d _ { \mathrm { M S E } } ( x _ { S _ { \mathrm { m o d } } } , x _ { S _ { \mathrm { c l e a n } } } ) , d _ { \mathrm { M S E } } ( x _ { P _ { \mathrm { m o d } } } , x _ { P _ { \mathrm { c l e a n } } } ) \} .\tag{9}
$$

This measures standalone fidelity, which target inactivity alone cannot establish. LPIPS (Learned Perceptual Image Patch Similarity) measures perceptual change for the same outputs (lower is better) (Zhang et al., 2018). Joint CLIP Distance compares implanted and clean $S + \dot { P }$ compositions.

Composition-specific behavior. $\Delta _ { \mathrm { T S } , i j }$ denotes the Target Shift of $S _ { i } + P _ { j }$ ; diagonal entries are matched and off-diagonal entries are mismatched. Lower off-pair activation and a higher ratio of mean matched to absolute mean mismatched shift indicate stronger pair selectivity. With L unrelated LoRAs, retention is

$$
\mathrm { R e t e n t i o n } _ { i , L } = 1 0 0 \overline { { \Delta } } _ { \mathrm { T S } , i , L } / \overline { { \Delta } } _ { \mathrm { T S } , i , 0 } ( \% ) .\tag{10}
$$

Retention below/above 100% indicates attenuation/amplification relative to the default.

## 5.2 EFFECTIVENESS AND STANDALONE PRESERVATION

Table 1: Comparison of attack effectiveness, functionality preservation, and model impact. Each ASR value is evaluated on n = 3000 generated images for the corresponding method and state. For slash-separated $S / P$ entries, each value independently uses n = 3000 images.
<table><tr><td rowspan="2">Method</td><td colspan="2">Attack Effectiveness</td><td colspan="3">Functionality Preservation</td><td colspan="2">Impact on Base Model</td></tr><tr><td>ASR (%) ↑</td><td>SMI ↑</td><td>FID↓</td><td>CLIP Score ↑</td><td>LPIPS ↓</td><td>Params ↓</td><td>Non-inv.</td></tr><tr><td>SDXL</td><td>0.0</td><td></td><td></td><td>34.26</td><td></td><td> $2 . 5 7 \times 1 0 ^ { 9 }$ </td><td>1</td></tr><tr><td>BadT2I (Zhai et al., 2023) (SDXL)</td><td>54.1</td><td>1.03</td><td>17.14</td><td>27.81</td><td>0.19</td><td> $2 . 5 7 \times 1 0 ^ { 9 }$ </td><td>×</td></tr><tr><td>Personalization (Huang et al., 2024) (SDXL)</td><td>89.6</td><td>1.09</td><td>20.47</td><td>31.22</td><td>0.15</td><td> $7 . 6 8 \times 1 0 ^ { 8 }$ </td><td>×</td></tr><tr><td>EvilEdit (Wang et al., 2024) (SDXL)</td><td>73.2</td><td>1.15</td><td>15.82</td><td>30.90</td><td>0.15</td><td> $2 . 5 7 \times 1 0 ^ { 9 }$ </td><td>×</td></tr><tr><td>MasqLoRA (Lyu et al., 2026) (SDXL)</td><td>90.3</td><td>1.12</td><td>16.79</td><td>32.01</td><td>0.12</td><td> $2 . 1 0 \times 1 0 ^ { 8 }$ </td><td>√</td></tr><tr><td>Benign LoRA S/P (SDXL)</td><td>0.0/0.0</td><td></td><td></td><td>33.52/33.40</td><td></td><td> $5 . 8 0 \times 1 0 ^ { 6 } / 5 . 8 0 \times 1 0 ^ { 6 }$ </td><td>√</td></tr><tr><td>Benign LoRA S/ P (SD v1.5)</td><td>0.0/0.0</td><td></td><td></td><td>30.51/30.39</td><td></td><td> $3 . 7 8 \times 1 0 ^ { 6 } / 3 . 7 8 \times 1 0 ^ { 6 }$ </td><td>√</td></tr><tr><td>Poisoned LoRA S/ P (SDXL)</td><td>2.8/3.4</td><td>0.74/0.78</td><td>15.88/16.02</td><td>33.28/33.18</td><td>0.13/0.15</td><td> $6 . 4 0 \times 1 0 ^ { 6 } / 6 . 4 0 \times 1 0 ^ { 6 }$ </td><td>√</td></tr><tr><td>Poisoned LoRA S/ P (SD v1.5)</td><td>3.8/4.6</td><td>0.95/0.99</td><td>15.84/15.96</td><td>31.12/31.00</td><td>0.18/0.20</td><td> $4 . 1 1 \times 1 0 ^ { 6 } / 4 . 1 1 \times 1 0 ^ { 6 }$ </td><td>√</td></tr><tr><td>LoRango (SDXL)</td><td>98.7</td><td>1.19</td><td>16.81</td><td>34.12</td><td>0.12</td><td> $1 . 2 8 \times 1 0 ^ { 7 }$ </td><td>√</td></tr><tr><td>LoRango (SD v1.5)</td><td>97.9</td><td>1.24</td><td>16.14</td><td>33.82</td><td>0.11</td><td> $8 . 2 2 \times 1 0 ^ { 6 }$ </td><td>√</td></tr></table>

Note. For the benign and poisoned LoRA rows, slash-separated values report the S-only/P-only results and parameter counts, respectively; the LoRango rows report the matched S + P composition and its summed parameter count. The 9,000-image count per backbone comprises the implanted S-only, P-only, and matched $S + P$ states, rather than a single table cell.

Table 1 shows that matched pairs reach 98.7% ASR on SDXL and 97.9% on SD v1.5, versus 2.8– 4.6% for implanted standalone adapters and 0% for benign adapters. LoRango has the highest reported SDXL ASR, although the baselines use different trigger protocols. Unlike text-triggered baselines, LoRango requires no modification to the user’s prompt, using adapter composition itself as the trigger. It preserves strong benign prompt alignment and competitive perceptual fidelity, although its FID is not the lowest among the evaluated methods. On SDXL, LoRango retains a CLIP Score of 34.12 versus 34.26 for the clean model, using 1 $. 2 8 \times 1 0 ^ { 7 }$ adapter parameters with a frozen backbone. This is more compact than full-model editing, but not inherently smaller than every single-LoRA baseline: parameter count depends on rank and injection scope.

Table 2: Standalone preservation and pair-activated effectiveness across four target behaviors. Each target behavior (P1–P4) is evaluated using n = 100 samples.
<table><tr><td>Target behavior</td><td>CLIP dist. / pixel MSE  $( S _ { \mathrm { { m o d } } } , \hat { S } _ { \mathrm { { c l e a n } } } )$ </td><td>CLIP dist. / pixel MSE  $( P _ { \mathrm { { m o d } } } , \dot { P } _ { \mathrm { { c l e a n } } } )$ </td><td>Joint CLIP dist.  $( S P _ { \mathrm { { m o d } } } , S P _ { \mathrm { { c l e a n } } } )$ </td><td>Target Shift  $( S P _ { \mathrm { { m o d } } } , S P _ { \mathrm { { c l e a n } } } )$ </td><td>ASR (%) (LoRango)</td></tr><tr><td>Object Identity (P1)</td><td>0.013 / 0.006</td><td>0.011/ 0.003</td><td>0.21</td><td>0.085</td><td>100.0</td></tr><tr><td>Global Style (P2)</td><td>0.012 / 0.005</td><td>0.008 / 0.002</td><td>0.16</td><td>0.037</td><td>98.0</td></tr><tr><td>Material/Appearance (P3)</td><td>0.013 / 0.005</td><td>0.008 / 0.002</td><td>0.22</td><td>0.061</td><td>99.0</td></tr><tr><td>Structural Disruption (P4)</td><td>0.011 / 0.005</td><td>0.008 / 0.002</td><td>0.25</td><td>0.025</td><td>99.0</td></tr></table>

Note. Parentheses identify the conditions compared under identical prompts and seeds. Subscripts “mod” and “clean” denote implanted adapters and their benign precursors; S/P are loaded individually, whereas SP denotes composition.

Table 2 covers changes to subject identity (P1), global style (P2), material/appearance (P3), and scene structure (P4). Standalone pixel MSE remains at 0.002–0.006, while matched compositions achieve 98.0–100.0% ASR with positive Target Shift, supporting standalone preservation alongside pair-conditioned activation. P4 has the smallest Target Shift (0.025) despite 99.0% ASR.

Table 3: NSFW-category response and benign generation quality across target behaviors. Each target–category condition contains n = 1000 generated samples.
<table><tr><td rowspan="2">Target behavior</td><td colspan="6">NSFW Response (ASR (%)/SMI)</td><td rowspan="2"></td><td colspan="2">Benign Quality</td></tr><tr><td>Nudity</td><td>Violence</td><td>Horror</td><td>Gore</td><td>Deformity</td><td>Self-harm</td><td>FID</td><td>CLIP Score</td></tr><tr><td>Object Identity (P1)</td><td>85.4/1.35</td><td>82.1/1.33</td><td>78.7/1.36</td><td>79.6/1.34</td><td>86.8/1.35</td><td></td><td>80.3/1.34</td><td>30.80</td><td>30.12</td></tr><tr><td>Global Style (P2)</td><td>87.2/1.36</td><td>84.6/1.34</td><td>80.9/1.37</td><td>78.8/1.35</td><td></td><td>88.1/1.36</td><td>82.5/1.35</td><td>30.12</td><td>29.82</td></tr><tr><td>Material/Appearance (P3)</td><td>81.6/1.34</td><td>79.8/1.33</td><td>83.5/1.35</td><td>80.4/1.36</td><td></td><td>82.7/1.34</td><td>79.1/1.35</td><td>31.46</td><td>30.44</td></tr><tr><td>Structural Disruption (P4)</td><td>79.7/1.33</td><td>81.5/1.34</td><td>77.9/1.32</td><td>82.2/1.35</td><td></td><td>84.0/1.33</td><td>83.6/1.37</td><td>31.21</td><td>29.65</td></tr></table>

Note. NSFW-response cells report ASR (%)/SMI; FID and CLIP Score evaluate benign generation quality.

Table 3 evaluates six Not Safe for Work (NSFW) categories across the four target behaviors (24,000 samples). Seventeen of 24 combinations meet or exceed the 80% ASR threshold, while seven fall below it; SMI remains above one throughout. Benign-quality metrics vary modestly across configurations, but without a clean anchor they support consistency rather than absolute preservation.

## 5.3 ROBUSTNESS

We evaluate robustness to pair-set size, unrelated LoRA co-loading, checkpoint transfer, and common inference-time variations.

![](images/34ed8b82c6951ea80ff9062260639270b1a815eb4036670c35fd11e796c58487.jpg)  
Figure 3: Pair selectivity under the $N = 4$ and $N = 8$ pair-set configurations. Each matrix cell is evaluated using $n = 1 0 0$ prompt–seed cases.

Robustness across Pair-Set Sizes. Figure 3 compares sets of $N \ = \ 4$ and $N \ = \ 8$ matched Signature–Payload pairs. Positive-shift rates are computed per matched cell; off-pair activation is aggregated across all mismatched cells. For $N = 4$ , matched positive-shift rates are 93–99%, versus 0.50% off-pair activation. For $N = 8 ,$ every matched cell reaches a 100% positive-shift rate, with no off-pair activation observed in the evaluated cases. The stronger matched shifts at $N = 8$ come with higher standalone pixel MSE, motivating $N = 4$ as the primary configuration. Appendix D details the per-cell shifts, off-pair rates, and standalone fidelity measurements.

Robustness to Additional Co-loaded LoRAs. Table 4 shows that all pairs remain active with up to four unrelated LoRAs: the minimum ASR is 87.0%, although retention falls as low as 69.09%. P2 is the most sensitive to interference, while P1 retains 99.0% ASR with four additional LoRAs. Shift amplification in some settings reveals a non-monotonic response to co-loading. Thus, unrelated adapters can alter attack strength without reliably preventing activation in the tested compositions.

Table 4: Effect of additional co-loaded LoRAs on target-shift retention and ASR. Each pair is evaluated using n = 100 prompt–seed cases per condition.
<table><tr><td rowspan="2">Add. LoRAs</td><td colspan="2">Pl</td><td colspan="2">P2</td><td colspan="2">P3</td><td colspan="2">P4</td></tr><tr><td>Retention (%) ↑</td><td>ASR (%) ↑</td><td>Retention (%) ↑</td><td>ASR (%) ↑</td><td>Retention (%) ↑</td><td>ASR (%) ↑</td><td>Retention (%) ↑</td><td>ASR (%) ↑</td></tr><tr><td>0 (Default)</td><td>100.00</td><td>100.0</td><td>100.00</td><td>98.0</td><td>100.00</td><td>99.0</td><td>100.00</td><td>99.0</td></tr><tr><td>1</td><td>102.02</td><td>100.0</td><td>87.71</td><td>94.0</td><td>113.38</td><td>99.0</td><td>134.88</td><td>99.0</td></tr><tr><td>2</td><td>102.49</td><td>99.0</td><td>78.01</td><td>91.0</td><td>98.81</td><td>96.0</td><td>124.14</td><td>97.0</td></tr><tr><td>4</td><td>91.76</td><td>99.0</td><td>69.09</td><td>87.0</td><td>90.37</td><td>95.0</td><td>99.38</td><td>95.0</td></tr></table>

Note. Retention is normalized to mean Target Shift with no additional LoRA ${ \overline { { ( L = 0 ) } } } ;$ values above 100% indicate shift amplification, not a success rate above 100%.

Table 5: Comparison between the default SDXL checkpoint and RealVisXL across P1–P4. Each checkpoint–pair condition contains n = 100 generated samples.
<table><tr><td>Metric</td><td colspan="2">Pl</td><td colspan="2">P2</td><td colspan="2">P3</td><td colspan="2">P4</td></tr><tr><td></td><td>SDXL (Default)</td><td>RealVisXL</td><td>SDXL (Default)</td><td>RealVisXL</td><td>SDXL (Default)</td><td>RealVisXL</td><td>SDXL (Default)</td><td>RealVisXL</td></tr><tr><td>Target Shift ↑</td><td>0.085</td><td>0.072</td><td>0.037</td><td>0.019</td><td>0.061</td><td>0.064</td><td>0.025</td><td>0.014</td></tr><tr><td>Visual ASR (%) ↑</td><td>100.0</td><td>100.0</td><td>98.0</td><td>96.0</td><td>99.0</td><td>98.0</td><td>99.0</td><td>95.0</td></tr><tr><td>Single-Adapter MSE ↓</td><td>0.006</td><td>0.003</td><td>0.005</td><td>0.002</td><td>0.005</td><td>0.002</td><td>0.005</td><td>0.003</td></tr><tr><td>Result</td><td>Pass</td><td>Pass</td><td>Pass</td><td>Pass</td><td>Pass</td><td>Pass</td><td>Pass</td><td>Pass</td></tr></table>

Note. SDXL (Default) uses the results in Table 2, with Single-Adapter MSE aggregated as defined in Section 5.1.

Robustness across Checkpoints. Direct transfer to RealVisXL without retraining yields 95.0– 100.0% ASR and lower Single-Adapter MSE than SDXL (Table 5). Target Shift increases slightly for P3 but decreases for P1, P2, and P4. For example, P4’s shift falls from 0.025 to 0.014 while ASR remains at 95.0%, indicating that success frequency transfers more consistently than effect magnitude. This result concerns transfer between compatible checkpoints; it does not establish unchanged-weight transfer across different denoiser architectures.

![](images/8df084998bb974cf5ef6f584db6749b7813f23f9a4814f4eb4358510f884d501.jpg)

![](images/043d590d0c0ccdc75db4ef471c1d0702a4d16b47d343dc34b1a5fe1b782f55c4.jpg)

![](images/d4ce0f28b81fc1c31b3fa7560142d42e957ce06d4fa8a06cacb51b8d97d519db.jpg)

![](images/b68492d394f0dc4ce889cc115ab314790a7d218ea82d105337ddf3a9767f4fc3.jpg)  
Figure 4: Robustness to classifier-free guidance (CFG), resolution, LoRA scale, and sampling steps.

Robustness to Inference Parameters. Each setting in Figure 4 uses $n = 1 0 0$ prompt–seed cases. Target Shift and Single-Adapter MSE are shown in percentage units; $\mathrm { \ddot { \Delta D e f a u l t } \mathrm { ? } }$ denotes each metric under the default inference configuration. All settings meet the 80% ASR criterion (81.0–99.0%), with the weakest case at 30 sampling steps. Higher guidance and larger LoRA scales increase Single-Adapter MSE, exposing an effectiveness–fidelity trade-off.

## 5.4 MECHANISM ANALYSIS AND COMPONENT ABLATION

We vary carrier capacity K and ablate Carrier-Level Cancellation, Output-Level Cancellation, or both at $K = 8 ,$ measuring effects on activation and standalone fidelity.

Table 6: Sensitivity to carrier capacity, measured by the number of carrier layers K. All settings use the same fixed pair-conditioned target objective and the same n = 100 prompt–seed cases.
<table><tr><td>Setting</td><td>Target Shift ↑</td><td>Single-Adapter MSE ↓ ASR (%) ↑</td><td></td></tr><tr><td>K = 1</td><td>0.001</td><td>0.003</td><td>46.0</td></tr><tr><td>K = 2</td><td>0.001</td><td>0.003</td><td>50.0</td></tr><tr><td>K = 4</td><td>0.027</td><td>0.004</td><td>70.0</td></tr><tr><td>K = 6</td><td>0.065</td><td>0.005</td><td>96.0</td></tr><tr><td>K = 8 (Default)</td><td>0.085</td><td>0.005</td><td>99.0</td></tr></table>

Note. K counts carrier layers, not LoRA rank or per-layer lanes (Section 4.3). The default K = 8 applies to this sweep, not all P1–P4 configurations.

Carrier-Capacity Sensitivity. Table 6 shows ASR below 80% for $K \ \leq \ 4 ,$ versus 96.0% and 99.0% for $K = 6$ and 8. Near-zero Target Shift at $K = 1$ and 2 suggests insufficient capacity. Raising K from 6 to 8 increases Target Shift from 0.065 to 0.085; both MSE values round to 0.005. Across the sweep, MSE rises from 0.003 to 0.005, indicating a modest fidelity cost.

Table 7: Ablation of the cancellation mechanisms. All conditions use the same fixed pairconditioned target objective and are evaluated using $n ~ = ~ 1 0 0$ prompt–seed cases per condition (400 cases total).
<table><tr><td>Setting</td><td>Target Shift ↑</td><td>Single-Adapter MSE ↓</td><td>ASR (%) ↑</td></tr><tr><td>w/o Carrier-Level Cancellation</td><td>0.027</td><td>0.005</td><td>66.0</td></tr><tr><td>w/o Output-Level Cancellation</td><td>0.063</td><td>0.071</td><td>97.0</td></tr><tr><td>w/o Carrier and Output Cancellation</td><td>0.076</td><td>0.114</td><td>51.0</td></tr><tr><td>With Carrier and Output Cancellation  $\mathrm { ( D e f a u l t ) }$ </td><td>0.085</td><td>0.005</td><td>99.0</td></tr></table>

Note. All conditions use ${ \overline { { K = 8 ; } } }$ Default enables both Carrier-Level and Output-Level Cancellation.

Cancellation Ablation. Table 7 distinguishes the two mechanisms: removing Carrier-Level Cancellation reduces ASR from 99.0% to 66.0% without changing the reported MSE, whereas removing Output-Level Cancellation retains 97.0% ASR but raises MSE from 0.005 to 0.071. The latter therefore contributes substantially to standalone fidelity. Removing both yields the highest MSE (0.114) and lowest ASR (51.0%), despite a larger Target Shift than removing Output-Level Cancellation alone. This separation can arise because mean semantic displacement and binary success frequency need not vary monotonically. Both mechanisms together provide the best effectiveness–preservation balance in this experiment.

## 5.5 CROSS-ARCHITECTURE GENERALIZATION

Table 8: Generalization across diffusion model families and denoiser architectures. All methods are evaluated under the same fixed target objective with $n = 2 0$ generated samples per condition.
<table><tr><td>Method</td><td>Model</td><td>Denoiser</td><td>Target Margin</td><td>ASR (%)</td><td>Result</td></tr><tr><td>MasqLoRA LoRango</td><td>SDXL SDXL</td><td>U-Net U-Net</td><td>0.0990 0.0222</td><td>90.0 100.0</td><td>Pass Pass</td></tr><tr><td>MasqLoRA LoRango</td><td>SD3 SD3</td><td>MMDiT MMDiT</td><td>0.0024 0.0173</td><td>20.0 95.0</td><td>Fail Pass</td></tr><tr><td>MasqLoRA LoRango</td><td>FLUX FLUX</td><td>Flow Transformer Flow Transformer</td><td>0.0063 0.0346</td><td>25.0 80.0</td><td>Fail Pass</td></tr></table>

Note. Result labels use the 80% ASR threshold defined in Section 5.1

LoRango passes on all three architectures (Table 8), achieving 95% ASR on SD3 and 80% on FLUX versus 20% and 25% for MasqLoRA. Both pass on SDXL: MasqLoRA has the larger Target Margin, but LoRango the higher ASR. With $n = 2 0$ per condition, these results provide initial crossarchitecture evidence, not precise population-level success estimates.

## 6 CONCLUSION

We introduced LoRango, a pair-conditioned mechanism for text-to-image diffusion that distributes a hidden behavior across a Signature LoRA and a Payload LoRA. Experiments on SD v1.5 and SDXL show high matched-pair ASR with low standalone activation and largely preserved benign quality, while the robustness and transferability studies demonstrate persistence across common inference settings and multiple denoiser architectures. Our study demonstrates a compositional supply-chain risk in image personalization that standalone adapter inspection cannot reliably assess.

## AI USE STATEMENT

Generative AI tools were used to polish the manuscript, including improving English grammar, clarity, concision, organization, and LAT<sub>E</sub>X formatting. Gemini 2.5 Pro was used as a visual evaluator to classify generated outputs when computing ASR. The authors reviewed all AI-assisted revisions, checked the reported values against the experimental records and tables, and take responsibility for the final content, claims, citations, and artifacts.

## ETHICS STATEMENT

This work studies a dual-use security risk in open LoRA ecosystems. Describing pair-conditioned backdoors may inform misuse, but understanding this threat is necessary for composition-aware auditing, provenance controls, and safer adapter deployment. Our experiments use generated images and do not involve human subjects or private user data. We present the attack to characterize the risk and support defensive research, and recommend that any release of implementation artifacts follow responsible disclosure and access-control practices.

## REPRODUCIBILITY STATEMENT

Section 4.4 reports the principal model, parameterization, and optimization settings, while the experimental section specifies metrics, baselines, sample sizes, and evaluation protocols. Appendix A provides the interaction derivations, Appendix C records module-level ranks, schedules, and export coefficients, and Appendix B defines the auxiliary training objectives. Together, these sections provide the information needed to reconstruct the method and audit the reported results.

## REFERENCES

Weixin Chen, Dawn Song, and Bo Li. TrojDiff: Trojan attacks on diffusion models with diverse targets. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 4035–4044, 2023a.

Yida Chen, Fernanda Viegas, and Martin Wattenberg. Beyond surface statistics: Scene representa-´ tions in a latent diffusion model. arXiv preprint arXiv:2306.05720, 2023b.

Minkyoung Cho, Ruben Ohana, Christian Jacobsen, Adityan Jothi, Min-Hung Chen, Z. Morley Mao, and Ethem Can. TC-LoRA: Temporally modulated conditional LoRA for adaptive diffusion control. arXiv preprint arXiv:2510.09561, 2025. URL https://arxiv.org/abs/2510. 09561.

Sheng-Yen Chou, Pin-Yu Chen, and Tsung-Yi Ho. How to backdoor diffusion models? In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 4015–4024, 2023a.

Sheng-Yen Chou, Pin-Yu Chen, and Tsung-Yi Ho. VillanDiffusion: A unified backdoor attack framework for diffusion models. In Advances in Neural Information Processing Systems, volume 36, 2023b.

Kevin Clark and Priyank Jaini. Text-to-image diffusion models are zero-shot classifiers. In Advances in Neural Information Processing Systems, volume 36, 2023.

Sihao Ding. Colluding LoRA: A compositional vulnerability in LLM safety alignment. arXiv preprint arXiv:2603.12681, 2026. URL https://arxiv.org/abs/2603.12681.

Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit H. Bermano, Gal Chechik, and Daniel Cohen-Or. An image is worth one word: Personalizing text-to-image generation using textual inversion. In International Conference on Learning Representations (ICLR), 2023.

Gemini Team. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. Technical Report arXiv:2507.06261, Google, 2025. URL https://arxiv.org/abs/2507.06261.

Tianyu Gu, Brendan Dolan-Gavitt, and Siddharth Garg. BadNets: Identifying vulnerabilities in the machine learning model supply chain. In Neural Information Processing Systems Workshop on Machine Learning and Computer Security, 2017.

Yuchao Gu, Xintao Wang, Jay Zhangjie Wu, Yujun Shi, Yunpeng Chen, Zihan Fan, Wuyou Xiao, Rui Zhao, Shuning Chang, Weijia Wu, Yixiao Ge, Ying Shan, and Mike Zheng Shou. Mix-of-Show: Decentralized low-rank adaptation for multi-concept customization of diffusion models. In Advances in Neural Information Processing Systems, volume 36, 2023.

Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. GANs trained by a two time-scale update rule converge to a local nash equilibrium. In Advances in Neural Information Processing Systems, volume 30, 2017. URL https://proceedings.neurips.cc/paper/2017/hash/ 8a1d694707eb0fefe65871369074926d-Abstract.html.

Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In Advances in Neural Information Processing Systems, volume 33, pp. 6840–6851, 2020.

Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations (ICLR), 2022.

Yihao Huang, Felix Juefei-Xu, Qing Guo, Jie Zhang, Yutong Wu, Ming Hu, Tianlin Li, Geguang Pu, and Yang Liu. Personalization as a shortcut for few-shot backdoor attack against text-to-image diffusion models. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pp. 21169–21178, 2024. doi: 10.1609/aaai.v38i19.30110.

Nupur Kumari, Bingliang Zhang, Richard Zhang, Eli Shechtman, and Jun-Yan Zhu. Multi-concept customization of text-to-image diffusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 1931–1941, 2023.

Zhiwen Li, Zhongjie Duan, Die Chen, Cen Chen, Daoyuan Chen, Yaliang Li, and Yingda Chen. AutoLoRA: Automatic LoRA retrieval and fine-grained gated fusion for text-to-image generation. arXiv preprint arXiv:2508.02107, 2025.

Weilin Lin, Nanjun Zhou, Yanyun Wang, Jianze Li, Hui Xiong, and Li Liu. BackdoorDM: A comprehensive benchmark for backdoor learning on diffusion model. In Advances in Neural Information Processing Systems, volume 38, 2025.

Yingqi Liu, Shiqing Ma, Yousra Aafer, Wen-Chuan Lee, Juan Zhai, Weihang Wang, and Xiangyu Zhang. Trojaning attack on neural networks. In Network and Distributed System Security Symposium (NDSS), 2018.

Grace Luo, Lisa Dunlap, Dong Huk Park, Aleksander Holynski, and Trevor Darrell. Diffusion hyperfeatures: Searching through time and space for semantic correspondence. In Advances in Neural Information Processing Systems, volume 36, 2023.

Liangwei Lyu, Jiaqi Xu, Jianwei Ding, and Qiyao Deng. When LoRA betrays: Backdooring text-toimage models by masquerading as benign adapters. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 8577–8586, 2026.

Tuna Han Salih Meral, Enis Simsar, Federico Tombari, and Pinar Yanardag. Contrastive test-time composition of multiple LoRA models for image generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 18090–18100, 2025.

Alexander Quinn Nichol and Prafulla Dhariwal. Improved denoising diffusion probabilistic models. In Proceedings of the 38th International Conference on Machine Learning, volume 139 of Proceedings ofMachine Learning Research, pp. 8162–8171, 2021.

Ryan Po, Guandao Yang, Kfir Aberman, and Gordon Wetzstein. Orthogonal adaptation for modular customization of diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 7964–7973, 2024.

Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Muller, Joe¨ Penna, and Robin Rombach. SDXL: Improving latent diffusion models for high-resolution image synthesis. In International Conference on Learning Representations (ICLR), 2024.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In Proceedings of the 38th International Conference on Machine Learning, volume 139 of Proceedings of Machine Learning Research, pp. 8748–8763, 2021. URL https://proceedings.mlr. press/v139/radford21a.html.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bjorn Ommer. High-¨ resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 10684–10695, 2022.

Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. DreamBooth: Fine tuning text-to-image diffusion models for subject-driven generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 22500–22510, 2023.

Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L. Denton, Seyed Kamyar Seyed Ghasemipour, Burcu Karagol Ayan, S. Sara Mahdavi, Raphael Gontijo Lopes, Tim Salimans, Jonathan Ho, David J. Fleet, and Mohammad Norouzi. Photorealistic text-toimage diffusion models with deep language understanding. In Advances in Neural Information Processing Systems, volume 35, 2022.

Enis Simsar, Thomas Hofmann, Federico Tombari, and Pinar Yanardag. LoRACLR: Contrastive adaptation for customization of diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 13189–13198, 2025.

Vera Soboleva, Aibek Alanov, Andrey Kuznetsov, and Konstantin Sobolev. T-LoRA: Single image diffusion model customization without overfitting. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, pp. 9051–9059, 2026. doi: 10.1609/aaai.v40i11.37861.

Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. In International Conference on Learning Representations (ICLR), 2021.

Lukas Struppek, Dominik Hintersdorf, and Kristian Kersting. Rickrolling the artist: Injecting backdoors into text encoders for text-to-image synthesis. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 4584–4596, 2023.

Jordan Vice, Naveed Akhtar, Richard Hartley, and Ajmal Mian. BAGM: A backdoor attack for manipulating text-to-image generative models. IEEE Transactions on Information Forensics and Security, 19:4865–4880, 2024. doi: 10.1109/TIFS.2024.3386058.

Hao Wang, Shangwei Guo, Jialing He, Kangjie Chen, Shudong Zhang, Tianwei Zhang, and Tao Xiang. EvilEdit: Backdooring text-to-image diffusion models in one second. In Proceedings of the 32nd ACM International Conference on Multimedia, pp. 3657–3665, 2024. doi: 10.1145/ 3664647.3680689.

Xingyi Yang and Xinchao Wang. Diffusion model as representation learner. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 18938–18949, 2023.

Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. IP-Adapter: Text compatible image prompt adapter for text-to-image diffusion models. arXiv preprint arXiv:2308.06721, 2023. doi: 10. 48550/arXiv.2308.06721.

Shengfang Zhai, Yinpeng Dong, Qingni Shen, Shi Pu, Yuejian Fang, and Hang Su. Text-to-image diffusion models can be easily backdoored through multimodal data poisoning. In Proceedings of the 31st ACM International Conference on Multimedia, pp. 1577–1587, 2023. doi: 10.1145/ 3581783.3612108.

Qingru Zhang, Minshuo Chen, Alexander Bukharin, Pengcheng He, Yu Cheng, Weizhu Chen, and Tuo Zhao. Adaptive budget allocation for parameter-efficient fine-tuning. In International Conference on Learning Representations (ICLR), 2023.

Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 586– 595, 2018. URL https://openaccess.thecvf.com/content\_cvpr\_2018/html/ Zhang\_The\_Unreasonable\_Effectiveness\_CVPR\_2018\_paper.html.

Ming Zhong, Yelong Shen, Shuohang Wang, Yadong Lu, Yizhu Jiao, Siru Ouyang, Donghan Yu, Jiawei Han, and Weizhu Chen. Multi-LoRA composition for image generation. Transactions on Machine Learning Research, 2024.

## A INTERACTION DERIVATIONS

For a generated output mapped by Ψ to the common action space, the behavior attributable specifically to composition is

$$
\begin{array} { r l r } { I _ { i j } ^ { \mathrm { o u t } } ( \boldsymbol { x } , \boldsymbol { \xi } ) = y _ { S _ { i } + P _ { j } } ( \boldsymbol { x } , \boldsymbol { \xi } ) - y _ { S _ { i } } ( \boldsymbol { x } , \boldsymbol { \xi } ) } & { } & \\ { - y _ { P _ { j } } ( \boldsymbol { x } , \boldsymbol { \xi } ) + y _ { \mathcal { O } } ( \boldsymbol { x } , \boldsymbol { \xi } ) , } & { } & { y _ { A } = \Psi ( F _ { A } ( \boldsymbol { x } ; \boldsymbol { \xi } ) ) . } \end{array}\tag{11}
$$

Using the same prompt and seed, the corresponding feature interaction at layer ℓ and timestep t is

$$
I _ { i j } ^ { ( \ell , t ) } = h _ { S _ { i } + P _ { j } } ^ { ( \ell , t ) } - h _ { S _ { i } } ^ { ( \ell , t ) } - h _ { P _ { j } } ^ { ( \ell , t ) } + h _ { \mathcal { O } } ^ { ( \ell , t ) } .\tag{12}
$$

A purely additive composition makes these differences vanish. Training instead aligns diagonal interactions with the pair action and suppresses off-diagonal interactions.

The stable scalar interface follows from the affine LayerNorm preceding the multiplicative block. For $z _ { t } ^ { ( \ell ) } = \gamma ^ { ( \ell ) } \odot \widehat { h } _ { t } ^ { ( \ell ) } + \beta ^ { ( \ell ) }$ and $\begin{array} { r } { \mathbf { 1 } ^ { \top } \widehat { h } _ { t } ^ { ( \ell ) } = 0 . } \end{array}$ , choosing $v ^ { ( \ell ) } = \kappa \gamma ^ { ( \ell ) ^ { - 1 } }$ gives

$$
b _ { t } ^ { ( \ell ) } = { v ^ { ( \ell ) } } ^ { \top } z _ { t } ^ { ( \ell ) } = \kappa \mathbf { 1 } ^ { \top } \widehat { h } _ { t } ^ { ( \ell ) } + { v ^ { ( \ell ) } } ^ { \top } \beta ^ { ( \ell ) } = b _ { 0 } ^ { ( \ell ) } .\tag{13}
$$

This bias-free projection need only keep ordinary variation within the cancellation region; unstable carrier coordinates are rejected during calibration.

Finally, if $s _ { i }$ and $p _ { j }$ are the local Signature and Payload perturbations and Φ is the native block transformation, their nonlinear cross-effect is

$$
C _ { i j } ( z ) = \Phi ( z + s _ { i } + p _ { j } ) - \Phi ( z + s _ { i } ) - \Phi ( z + p _ { j } ) + \Phi ( z ) .\tag{14}
$$

It vanishes for a linear block and can be nonzero in GEGLU because the Signature gate displacement interacts multiplicatively with the Payload value/output path. Equation 6 summarizes a local path under a shared-value approximation, absorbing the value feature and its output projection into the effective vector $u _ { j } ^ { ( \ell ) }$ . That vector need not be constant across inputs or timesteps; the equation is not an exact expression for the full-network output.

## B AUXILIARY TRAINING OBJECTIVES

This appendix expands the training discussion in Section 4.3, distinguishing the implemented refinement losses from schematic auxiliary design objectives. Expectations are over the relevant prompts, seeds, layers, and timesteps.

Implemented refinement losses. Action refinement uses normalized mean squared error (NMSE), masked NMSE, and a direction-alignment loss. The first two penalize prediction discrepancies globally and over masked entries, respectively; the last encourages alignment with the target direction. Not all conceptual objectives below are optimized in every stage; in particular, $\lambda _ { \mathrm { o f f } } = 0$ for P3/P4. Appendix C reports the stage-specific losses, weights, and schedules. These training losses are distinct from the image-level Single-Adapter MSE used for evaluation.

Conceptual objective decomposition. The intended design constraints over prompts, seeds, com positions, and timesteps T can be summarized as

$$
\begin{array} { r l } & { { \mathcal { L } } = \lambda _ { \mathrm { d i a g } } { \mathcal { L } } _ { \mathrm { d i a g } } + \lambda _ { \mathrm { o f f } } { \mathcal { L } } _ { \mathrm { o f f } } + \lambda _ { \mathrm { s i n g l e } } { \mathcal { L } } _ { \mathrm { s i n g l e } } } \\ & { ~ + \lambda _ { \mathrm { u t i l } } { \mathcal { L } } _ { \mathrm { u t i l } } + \lambda _ { \mathrm { t r a j } } { \mathcal { L } } _ { \mathrm { t r a j } } + \lambda _ { \mathrm { c a n c e l } } { \mathcal { L } } _ { \mathrm { c a n c e l } } + \lambda _ { \mathrm { r e g } } { \mathcal { L } } _ { \mathrm { r e g } } . } \end{array}\tag{15}
$$

Here $\mathcal { L } _ { \mathrm { d i a g } }$ and ${ \mathcal { L } } _ { \mathrm { o f f } }$ describe matched-action alignment and mismatch suppression. The remaining five terms describe standalone inactivity, utility preservation, trajectory consistency, cancellation, and regularization, respectively. This decomposition organizes the design goals rather than specifying a single seven-term loss jointly optimized in every reported run. The schematic formulations below therefore should not be read as five additional losses necessarily enabled during training.

Single-adapter inactivity. Let $a _ { i }$ be a nonnegative differentiable surrogate for the programmed action $r _ { i }$ . We suppress that action when either adapter is loaded alone:

$$
\mathcal { L } _ { \mathrm { s i n g l e } } = \mathbb { E } _ { x , \xi } \left[ \sum _ { i = 1 } ^ { N } \Big ( a _ { i } \big ( F _ { S _ { i } } ( x ; \xi ) \big ) + a _ { i } \big ( F _ { P _ { i } } ( x ; \xi ) \big ) \Big ) \right] .\tag{16}
$$

This term prevents the Signature from directly carrying the action and the Payload from exposing it without the matched codeword.

Benign-utility preservation. Each modified adapter is constrained to reproduce the behavior of its benign precursor:

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { u t i l } } = \mathbb { E } _ { x , \xi } \Bigg [ \displaystyle \sum _ { i = 1 } ^ { N } \Big ( d _ { u } \Big ( F _ { S _ { i } } ( x ; \xi ) , F _ { S _ { i } ^ { \mathrm { b e n } } } ( x ; \xi ) \Big ) } \\ & { ~ + d _ { u } \Big ( F _ { P _ { i } } ( x ; \xi ) , F _ { P _ { i } ^ { \mathrm { b e n } } } ( x ; \xi ) \Big ) \Big ) \Bigg ] , } \end{array}\tag{17}
$$

where $d _ { u }$ may combine pixel, perceptual, and representation distances according to the advertised utility of each adapter.

Trajectory consistency. We align matched interactions with layer- and timestep-specific training directions over the selected carriers and denoising timesteps:

$$
\mathcal { L } _ { \mathrm { t r a j } } = \mathbb { E } _ { x , \xi } \left[ \sum _ { i = 1 } ^ { N } \sum _ { t \in \mathcal { T } } \sum _ { \ell \in \mathcal { K } } d _ { \mathrm { t r a j } } \Big ( I _ { i i } ^ { ( \ell , t ) } , d _ { i } ^ { ( \ell , t ) } \Big ) \right] .\tag{18}
$$

The direction $d _ { i } ^ { ( \ell , t ) }$ may be obtained from a teacher, a target residual, or a differentiable semantic objective; it is used only during training.

Differential cancellation. For payload $P _ { j }$ , let $\pi _ { \mathrm { o f f } , j }$ sample the non-matched states $\{ P _ { j } \} \cup \{ S _ { i } +$ $P _ { j } : i \ne j \}$ , and let $h _ { A }$ denote the activation under sampled state A. We preserve cancellation between its signal and reference branches in those states:

$$
\mathcal { L } _ { \mathrm { c a n c e l } } = \sum _ { j = 1 } ^ { N } \mathbb { E } _ { \boldsymbol { x } , \boldsymbol { \xi } , \boldsymbol { A } \sim \pi _ { \mathrm { o f f } , j } } \left[ \sum _ { t \in \mathcal { T } } \sum _ { \ell \in \mathcal { K } } \left. \boldsymbol { R } _ { P _ { j } , + } ^ { ( \ell , t ) } ( h _ { A } ) + \boldsymbol { R } _ { P _ { j } , - } ^ { ( \ell , t ) } ( h _ { A } ) \right. _ { 2 } ^ { 2 } \right] .\tag{19}
$$

This objective does not constrain the matched state, where the Signature is expected to break cancellation and release the action.

Regularization. For clarity, we decompose the regularizer as

$$
\mathcal { L } _ { \mathrm { r e g } } = \mu _ { \theta } \mathcal { R } _ { \theta } + \mu _ { \alpha } \mathcal { R } _ { \alpha } + \mu _ { h } \mathcal { R } _ { \mathrm { d r i f t } } + \mu _ { R } \mathcal { R } _ { \mathrm { e n e r g y } } ,\tag{20}
$$

where the four terms represent penalties on implanted LoRA weights, Signature amplitude, ordinarystate activation drift, and carrier residual energy, respectively. This decomposition describes possible regularization roles; it does not specify four separately enabled penalties in the reported runs.

## C REPRODUCIBILITY DETAILS

Modules and ranks. The benign LoRAs target to q, to k, to v, and to out.0; the last name denotes the linear attention-output projection in Diffusers. Carrier updates target ff.net.0.proj, the joint GEGLU gate/value input projection, and ff.net.2, its output projection. Payload projections use rank 4. The Signature uses rank 5 at ff.net.0.proj for an additional constant-gate dimension and rank 4 at ff.net.2; the P1 mid t7 carrier uses ranks 4/3 because it has three lanes.

Schedules. Benign LoRAs are trained for 1,800 AdamW steps at $5 \times 1 0 ^ { - 5 }$ and refined for 1,200 steps at $2 \times 1 0 ^ { - 5 }$ . P1/P2 use a staged warm start totaling 10,100 updates; their final 2,500-step joint refinement uses $3 \times 1 0 ^ { - 4 }$ for the action and $2 \times 1 0 ^ { - 4 }$ for the reader. P3 uses 3,000 initial action steps at $1 . 5 \times 1 0 ^ { - 3 }$ and 1,400 refinement steps at $1 0 ^ { - 3 }$ ; P4 uses 1,400 refinement steps at $1 0 ^ { - 3 }$ Action refinement minimizes NMSE plus 0.5 masked NMSE and 0.5 direction loss. AdamW uses $( \beta _ { 1 } , \beta _ { 2 } ) = ( 0 . 9 , 0 . 9 9 )$ ), weight decay $\mathrm { \dot { 1 } 0 ^ { - 4 } }$ , and gradient clipping of 1.0 for utility parameters and 2.0 for action/reader parameters; P3/P4 use $\lambda _ { \mathrm { o f f } } = 0$

Export coefficients. For P1–P4, respectively, gate coefficients are {29, 35, 29, 46}, reader coefficients are {0.08, 0.06, 0.08, 0.05}, action coefficients are {0.33, 0.45, 0.48, 0.462}, and valuecancellation coefficients are {0.85, 0.90, 0.85, 0.85}.

## D PER-CELL PAIRING METRICS

This appendix expands the pair-selectivity results in Figure 3. Each cell $S _ { i } + P _ { j }$ is evaluated on the same $n = 1 0 0$ prompt–seed cases. Diagonal cells $( i ~ = ~ j )$ are the intended matched pairs; off-diagonal cells $( i \neq j )$ are mismatched pairs that should remain inactive.

Standalone fidelity. For each Signature or Payload, CLIP distance measures semantic deviation between the implanted adapter and its clean precursor, while pixel MSE measures their image-level deviation under the same prompt and seed. Lower values indicate better preservation of the adapter’s advertised standalone behavior. Because these quantities depend only on the individual adapter, Table 9 places each intended Signature–Payload pair on one row so that the two adapters can be compared directly without repeating their standalone values across mismatched compositions.

Pair-conditioned behavior. Table 10 reports Target Shift and off-pair activation rate for every Signature–Payload composition under the $\bar { N } = 4$ and $N = 8$ configurations. Target Shift measures movement of the composed output toward the designated target. A large positive diagonal value indicates effective activation by the intended pair, whereas an off-diagonal value near zero indicate that a mismatched pair remains suppressed. The off-pair activation rate is the percentage of the 100 cases in a mismatched cell that nevertheless activate the target—that is, the empirical probability of an unintended pairing success. Lower is better. It is undefined for diagonal cells, which are intended to activate and are therefore marked by an em dash. The aggregate rate pools all off-diagonal cases: 1,200 cases for $N = 4$ and 5,600 cases for $N = 8 .$

Table 9: Standalone fidelity of the intended Signature–Payload pairs used in Figure 3. Each row compares the two members of one matched pair with their respective clean precursors; CLIP distance and pixel MSE are both lower-is-better.
<table><tr><td rowspan="2">Pair ID i</td><td colspan="2">Signature-only  $( \bar { S _ { i } ^ { \mathrm { m o d } } } , S _ { i } ^ { \mathrm { c l e a n } } )$ </td><td colspan="2">Payload-only  $( P _ { i } ^ { \mathrm { { \dot { m o d } } } } , P _ { i } ^ { \mathrm { c l e a \dot { n } } } )$ </td></tr><tr><td>CLIP dist. ↓</td><td>Pixel MSE ↓</td><td>CLIP dist. ↓</td><td>Pixel MSE↓</td></tr><tr><td colspan="5">Pair-set size N = 4</td></tr><tr><td>1</td><td>0.013</td><td>0.006</td><td>0.011</td><td>0.003</td></tr><tr><td>2</td><td>0.012</td><td>0.005</td><td>0.008</td><td>0.002</td></tr><tr><td>3</td><td>0.013</td><td>0.005</td><td>0.008</td><td>0.002</td></tr><tr><td>4</td><td>0.011</td><td>0.005</td><td>0.008</td><td>0.002</td></tr><tr><td colspan="5">Pair-set size  $\mathbf { N } = \mathbf { 8 }$ </td></tr><tr><td>1</td><td>0.017</td><td>0.011</td><td>0.015</td><td>0.005</td></tr><tr><td>2</td><td>0.019</td><td>0.012</td><td>0.019</td><td>0.006</td></tr><tr><td>3</td><td>0.019</td><td>0.011</td><td>0.016</td><td>0.005</td></tr><tr><td>4</td><td>0.015</td><td>0.010</td><td>0.014</td><td>0.005</td></tr><tr><td>5</td><td>0.017</td><td>0.011</td><td>0.012</td><td>0.004</td></tr><tr><td>6</td><td>0.015</td><td>0.010</td><td>0.013</td><td>0.004</td></tr><tr><td>7</td><td>0.018</td><td>0.011</td><td>0.013</td><td>0.005</td></tr><tr><td>8</td><td>0.015</td><td>0.009</td><td>0.010</td><td>0.004</td></tr></table>

Note. The Signature columns report CLIP distance and pixel MSE between outputs generated with $S _ { i } ^ { \mathrm { m o d } }$ alone and with $S _ { i } ^ { \mathrm { c l e a n } }$ alone. The Payload columns report the same metrics for $P _ { i } ^ { \mathrm { m o d } }$ alone versus $P _ { i } ^ { \mathrm { c l e a n } }$ alone, using the same prompt and seed. Each row groups the two members of an intended pair; these are standalone-adapter measurements, not measurements of their co-loaded output.

Table 10: Pair-conditioned metrics for every Signature–Payload composition. For each Signature, the first row gives Target Shift and the second gives off-pair activation rate. Diagonal entries are intended matched pairs, so their off-pair rate is not applicable and is marked by an em dash.
<table><tr><td colspan="9">Pair-set size  $\mathbf N = 4$ </td></tr><tr><td>Signature</td><td>Metric</td><td colspan="2"> $P _ { 1 }$ </td><td colspan="2"> $P _ { 2 }$ </td><td colspan="2"> $P _ { 3 }$ </td><td colspan="2"> $P _ { 4 }$ </td></tr><tr><td> $S _ { 1 }$ </td><td>Shift</td><td colspan="2">0.085</td><td colspan="2">0.000</td><td colspan="2">-0.001</td><td colspan="2">0.000</td></tr><tr><td></td><td>Off (%) Shift</td><td colspan="2">0.001</td><td colspan="2">1.0</td><td colspan="2">0.0</td><td colspan="2">0.0</td></tr><tr><td> $S _ { 2 }$ </td><td>Off (%)</td><td colspan="2">3.0</td><td colspan="2">0.037</td><td colspan="2">-0.001 0.0</td><td colspan="2">0.000 0.0</td></tr><tr><td> $S _ { 3 }$ </td><td>Shift</td><td colspan="2">0.000</td><td colspan="2">0.000</td><td colspan="2">0.061</td><td colspan="2">0.000</td></tr><tr><td></td><td>Off (%)</td><td colspan="2">1.0</td><td colspan="2">0.0</td><td colspan="2"></td><td colspan="2">0.0</td></tr><tr><td> $S _ { 4 }$ </td><td>Shift</td><td colspan="2">0.000</td><td colspan="2">-0.001</td><td colspan="2">-0.001</td><td colspan="2">0.025</td></tr><tr><td></td><td>Off (%)</td><td colspan="2">1.0</td><td colspan="2">0.0</td><td colspan="2">0.0</td><td colspan="2"></td></tr><tr><td colspan="10">Pair-set size  $\mathbf { N } = \mathbf { 8 }$ </td></tr><tr><td>Signature</td><td>Metric</td><td> $P _ { 1 }$ </td><td> $P _ { 2 }$ </td><td> $P _ { 3 }$ </td><td> $P _ { 4 }$ </td><td> $P _ { 5 }$ </td><td> $P _ { 6 }$ </td><td> $P _ { 7 }$ </td><td> $P _ { 8 }$ </td></tr><tr><td> $S _ { 1 }$ </td><td>Shift</td><td>0.113</td><td>-0.002</td><td>-0.002</td><td>-0.003</td><td>0.003</td><td>-0.004</td><td>-0.004</td><td>0.003</td></tr><tr><td></td><td>Off (%) Shift</td><td></td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td> $S _ { 2 }$ </td><td>Off (%)</td><td>0.000</td><td>0.098</td><td>0.000</td><td>-0.009</td><td>0.000</td><td>-0.005 0.0</td><td>-0.004</td><td>0.003 0.0</td></tr><tr><td></td><td>Shift</td><td>0.0</td><td>-0.005</td><td>0.0</td><td>0.0 0.001</td><td>0.0</td><td></td><td>0.0</td><td>0.005</td></tr><tr><td> $S _ { 3 }$ </td><td>Off (%)</td><td>0.000 0.0</td><td>0.0</td><td>0.139</td><td>0.0</td><td>0.001 0.0</td><td>-0.003 0.0</td><td>-0.006 0.0</td><td>0.0</td></tr><tr><td></td><td>Shift</td><td>0.004</td><td>0.000</td><td>0.001</td><td>0.108</td><td>0.004</td><td>-0.003</td><td>-0.007</td><td>0.004</td></tr><tr><td> $S _ { 4 }$ </td><td>Off (%)</td><td>0.0</td><td>0.0</td><td>0.0</td><td></td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td></td><td>Shift</td><td>-0.001</td><td>-0.001</td><td>0.002</td><td>-0.001</td><td>0.096</td><td>-0.006</td><td>-0.004</td><td>-0.001</td></tr><tr><td> $S _ { 5 }$ </td><td>Off (%)</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td></td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td> $S _ { 6 }$ </td><td>Shift</td><td>0.004</td><td>0.000</td><td>0.002</td><td>-0.003</td><td>0.003</td><td>0.123</td><td>-0.005</td><td>0.001</td></tr><tr><td></td><td>Off (%)</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td></td><td>0.0</td><td>0.0</td></tr><tr><td></td><td>Shift</td><td>0.003</td><td>0.002</td><td>0.001</td><td>-0.005</td><td>-0.001</td><td>-0.005</td><td>0.117</td><td>0.002</td></tr><tr><td> $S _ { 7 }$ </td><td>Off (%)</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td></td><td>0.0</td></tr><tr><td></td><td>Shift</td><td>0.002</td><td>-0.002</td><td>0.001</td><td>0.001</td><td>-0.002</td><td>-0.005</td><td>-0.006</td><td>0.118</td></tr><tr><td> $S _ { 8 }$ </td><td>Off (%)</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td></td></tr></table>