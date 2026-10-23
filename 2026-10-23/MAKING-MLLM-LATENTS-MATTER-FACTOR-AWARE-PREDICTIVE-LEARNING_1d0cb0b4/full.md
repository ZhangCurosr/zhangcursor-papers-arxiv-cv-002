# MAKING MLLM LATENTS MATTER: FACTOR-AWARE PREDICTIVE LEARNING FOR CINEMATIC VIDEO GENERATION

Songlin Yang<sup>1,∗,†</sup>, Zhe Wang<sup>1,∗</sup>, Xiaotong Zhao<sup>2</sup>, Xi Wang<sup>3</sup>, Ran Zhang<sup>2</sup>, Shuai Li<sup>2</sup>, Taiyi Wu<sup>2</sup>, Songlian Li<sup>2</sup>, Eric Liu<sup>2</sup>, Alan Zhao<sup>2</sup>, Anyi Rao<sup>1</sup> <sup>1</sup>MMLab@HKUST, The Hong Kong University of Science and Technology <sup>2</sup>Tencent Video, <sup>3</sup>LIX, Ecole Polytechnique <sup>´</sup>

## ABSTRACT

Cinematic video generation requires both temporal inference from limited inputs (i.e., sparse rather than frame-by-frame guidance) and factor-aware control of character, scene, action, and camera. Multimodal large language models (MLLMs) offer useful priors for both capabilities, but their representations are not automatically actionable for video generation: a Video Diffusion Transformer (DiT) may bypass them through native conditioning pathways, and using them does not ensure either factor-aware control or generation-compatible temporal prediction. We propose Factor-Aware Predictive Learning (FAPL), which establishes effective MLLM latent use as a shared prerequisite and then targets the two capabilities with complementary learning signals. First, latent-only generator grounding trains the DiT with a frozen MLLM and connector, strengthening the DiT’s use of MLLM latents for generation. Second, for factor-aware control, paired videos specify a target-factor change and non-target preservation, and factor-span replacement trains the DiT to realize this relationship in its output. Third, for temporal inference, text-only and text-plus-complete-video observations share a video target, and the trained DiT is then frozen while the MLLM and connector are optimized through it. This final stage trains limited-input representations for generation while retaining complete-observation reconstruction as a complementary training mode. Extensive experiments demonstrate our consistently better performance on three cinematic video generation tasks (i.e., sparseto-video, factor-aware editing, and multi-shot generation) over baselines and validate FAPL’s contributions to MLLM feature use, temporal control, and factor coordination.

## 1 INTRODUCTION

Cinematic video generation (Wang et al., 2025a; Wu et al., 2026; Zhang et al., 2026b; Yang et al., 2026b; Tang et al., 2026) helps creators explore content, performances, and camera choices before physical production. Beyond visual realism, it requires two capabilities (Fig. 1). (i) Temporal inference from limited inputs requires the model to construct plausible action sequences and camera motion from a script and a few visual references (Xiao et al., 2026; Zhang et al., 2026a), which only sparsely specify how a shot unfolds (i.e., they do not provide a dense, frame-by-frame specification of all visual content). (ii) Factor-aware control requires character, scene, action, and camera to follow their respective instructions while remaining compatible: changing a character should preserve the requested action, and changing camera motion should not replace the scene (Huang et al., 2026b; Girish et al., 2026). Therefore, the central challenge is not only to infer missing content, but also to make those inferences selectively controllable in the generated video.

Standard conditioning provides limited support for these requirements. Text encoders (Raffel et al., 2020; Radford et al., 2021) offer limited visual and temporal grounding, while VAE-encoded references (Kingma & Welling, 2013) preserve appearance without explicitly representing action or camera intent. Multimodal large language models (MLLMs) (Yang et al., 2025; Wang et al., 2024) provide a promising alternative (Wang et al., 2026): their visual priors and semantic reasoning can help interpret scripts and references, infer plausible temporal details, and coordinate cinematic factors. Recent methods (Yang et al., 2026a; Wei et al., 2026) therefore condition Video Diffusion Transformers (DiTs) (Peebles & Xie, 2023) on projected MLLM hidden states.

![](images/4bd1f979ec36c8f7d0cacb02b5291a6c6fc86b634af95a3c3095ebcaf28b1573.jpg)  
Figure 1: Cinematic video generation requires inference from limited inputs and factor-aware control. (i) The man turns toward the box, places his right hand on top, and steadies the bundle with his left hand. Text-only conditioning misses this sequence, while observing the complete video enables the same model to reproduce it. (ii) Keeping the descriptions of all other factors fixed and replacing only the character-related prompt, the non-character factors change as well. These cases illustrate failures in temporal inference and factor control, with editing providing a probe of what the generator changes and preserves.

![](images/d1f759980819fd0adccfa349f589a2fc83ab9a38e29115140df13ae1445bec59.jpg)  
Figure 2: MLLM latents are easily bypassed in existing MLLM–DiT models. (i) Under MLLMonly conditioning, Omni-Video 2 (1.3B DiT) produces noise, while Omni-Video 2 (14B DiT) and UniVideo produce artifacts and depict one person’s successive actions as two people acting differently. MLLM latents alone therefore fail to provide reliable generation guidance in these examples. (ii) Under conflicting conditions, VAE references dominate the output.

However, making MLLM features available is different from making them effective for generation. Two obstacles remain: First, MLLM feature bypass is a shared bottleneck for both capabilities: native text-encoder and VAE-reference pathways can reduce the generation loss without substantial reliance on MLLM latents (Fig. 2). Second, task-structured supervision is insufficient. For factor-aware control, ordinary video-level training does not explicitly associate an intervention on a particular MLLM span with the intended output change and non-target preservation. For temporal inference, adapting the connector and DiT around a frozen MLLM does not train the MLLM itself to produce temporally informative representations from sparse inputs. Thus, the generator must learn how to use MLLM representations, while the MLLM must also learn what representations to produce for generation.

We propose Factor-Aware Predictive Learning (FAPL) to strengthen the use of MLLM representations for temporal inference and factor-aware control through three training stages. First, to strengthen the DiT’s use of MLLM features, Stage I freezes the MLLM and connector and trains the DiT with their projected latents as the sole content conditions, using both text-only and text-pluscomplete-video observations. This establishes a generation interface that subsequent stages can use for factor-aware control and predictive learning. Second, to enable factor-aware control, Stage II uses paired videos that change selective cinematic factors while preserving the others. Structured descriptions locate the projected latent spans corresponding to character, scene, action, and camera. Keeping the MLLM and connector frozen, we replace the selected factor span between paired examples and train the DiT against the corresponding video target. The paired videos specify what should change and remain, while span replacement explicitly associates this visual relationship with a localized conditioning intervention, without assuming fully disentangled representations. Finally, to learn temporal inference from sparse inputs, Stage III freezes the trained DiT and jointly optimizes the MLLM and connector using text-only and text-plus-complete-video observations that share a video target. The text-only mode trains prediction from limited evidence, while the completevideo mode provides complementary reconstruction supervision through the same generation interface. The fixed DiT directs learning toward generation-compatible representations without simultaneous decoder adaptation. Experiments evaluate FAPL on sparse-to-video generation, factor-aware editing, and multi-shot generation, with stage-wise ablations examining the respective contributions of strengthened MLLM feature use, factor interventions, and mixed-observation predictive learning.

Our contributions are threefold: (i) We investigate the gap between multimodal understanding and its effective use in cinematic generation, and propose FAPL to translate MLLM understanding into actionable guidance for inference from limited inputs and factor-aware control. (ii) We introduce factor-aware supervision through latent interventions, turning paired edits into structured learning signals for how the generator should change and preserve cinematic content, without assuming fully disentangled representations. (iii) We formulate predictive learning through a fixed generator, linking limited-input prediction and complete-observation reconstruction via shared video targets to learn temporally informative, generation-compatible MLLM representations.

## 2 RELATED WORK

Cinematic Video Generation. Cinematic video generation (Yang et al., 2026e; Elmoghany et al., 2025; Yang et al., 2026d) has recently received growing attention for producing shots with specified content, performances, and camera motions from limited materials (Wang et al., 2025a; Meng et al., 2026; Yang et al., 2026c). Existing studies target the two requirements: (i) Inferring content from limited inputs. Reference-driven and multi-shot methods maintain identity and scene consistency from given references (Xiao et al., 2026; Zhang et al., 2026a; Liu et al., 2025). (ii) Factor-Aware Control. Camera motion, character replacement, action transfer, and scene modification are typically handled separately (Wang et al., 2023; Tan et al., 2026a; Ye et al., 2026; Jiang et al., 2025; Huang et al., 2026b; Girish et al., 2026; Fang et al., 2026; Huang et al., 2026a).

MLLM-Conditioned Video Generation and Editing. Recent video diffusion models achieve high-quality synthesis and support a growing range of editing tasks (Blattmann et al., 2023; Kong et al., 2024; Wan et al., 2025). To move beyond single-task systems, recent work couples MLLMs with diffusion backbones, either through in-context conditioning (Ye et al., 2026; Ju et al., 2025; Jiang et al., 2025) or through an MLLM understanding branch connected to a generation branch (We et al., 2026; Yang et al., 2026a; Tan et al., 2026b). Related studies improve controllability over specific factors such as camera motion (Wang et al., 2023) and subject identity (Liu et al., 2025). These systems differ in architecture and task coverage, but commonly inject MLLM features alongside text-encoder and VAE reference features through a shared conditioning interface, and supervise training at the video level. Such supervision does not determine how much each condition contributes, and the generator may rely on familiar text and dense visual-reference pathways. Paired editing examples are also typically consumed as additional supervised samples, without explicitly indicating which factor should change and which content should be preserved. In contrast, FAPL strengthens the role ofMLLM latents through latent-only training, and usesfactor-span replacement between paired examples to supervise factor-aware changes and preservation.

![](images/f94324feb01ff6e13a7209ac08c355106358f478f135f4ad2a14ffb20631eab9.jpg)  
Figure 3: Overview of Factor-Aware Predictive Learning (FAPL). Stage I freezes the MLLM and its original connector and trains the DiT using MLLM latents. Stage II retains this configuration and swaps factor-specific output spans between paired examples to supervise factor-aware changes. Stage III freezes the trained DiT and jointly updates the MLLM and connector through video reconstruction, and text-only and text-plus-complete-video inputs share the same generation target.

Predictive Representation Learning. Joint-embedding predictive architectures learn representations by predicting target features from partial observations rather than reconstructing pixels, avoiding unpredictable low-level details (Assran et al., 2023), and extend to action-conditioned future prediction for planning and control (Bardes et al., 2024; Assran et al., 2025). Generative models also decode predicted representations into visual outputs. However, such models usually learn task-specific spaces without high-fidelity video decoders, or require separately trained decoders, and rarely use paired editing data to supervise factor-level change and preservation. In contrast, we reuse a pretrained MLLM and Video DiT and supervise prediction through visual reconstruction with a fixed generator, preserving the decoding pathway while training the MLLM to infer video content from limited observations.

## 3 METHOD

Factor-Aware Predictive Learning (FAPL) trains MLLM representations for limited-input temporal inference and factor-aware control of character, scene, action, and camera. It structures supervision in generation space: factor-span interventions use paired edits to link localized latent changes to target-factor changes and non-target preservation, while shared-target predictive learning couples limited-input prediction with complete-observation reconstruction through a fixed generator. MLLM-only conditioning strengthens the DiT’s use of these representations, supporting both mechanisms without architectural changes.

We first describe the framework and paired data in Sec. 3.1. Training then proceeds in three stages: (i) MLLM-Only Generator Grounding strengthens the DiT’s use of MLLM features (Sec. 3.2); (ii) Factor-Aware Control Improvement trains the DiT to realize target-factor changes while preserving other content through factor-span replacement (Sec. 3.3); and (iii) Prediction From Limited Inputs optimizes the MLLM and connector through the fixed DiT using shared targets across observation modes (Sec. 3.4). All three stages use MLLM latents as the only content conditions. Fig. 3 illustrates the framework, Algorithm 1 summarizes training, and Sec. 3.5 describes inference.

Algorithm 1 Three-Stage Training of Factor-Aware Predictive Learning (FAPL)   
Require: Pretrained MLLM $F _ { \theta _ { 0 } }$ , connector $P _ { \eta _ { 0 } } ,$ , and $\overline { { \operatorname { D i T } G _ { \phi _ { 0 } } } } ;$ observation data $\mathcal { D } _ { \mathrm { o b s } }$ and editing pairs $\mathcal { D } _ { \mathrm { e d i t } }$   
1: Initialize $\theta  \theta _ { 0 } , \eta  \eta _ { 0 }$ , and ϕ $ \phi _ { 0 } ;$ Use only MLLM latents as content conditions in all stages.   
2: Stage I: Enhance MLLM Latent Use (MLLM-Only Generator Grounding).   
3: for each training step do   
4: Sample $( x , p ) \sim \mathrm { \bar { \mathcal { D } } _ { o b s } }$ and observation mode m $\sim \rho ;$ Compute $Z _ { \theta _ { 0 } , \eta _ { 0 } } ( u ^ { m } )$   
5: Update $\phi$ using L<sub>FM</sub>(x ${ | \ z _ { \theta _ { 0 } , \eta _ { 0 } } ( u ^ { m } ) ; \phi ) }$ ; keep $\theta _ { 0 } , \eta _ { 0 }$ fixed.   
6: end for   
7: Stage II: Improve Factor-Aware Control.   
8: for each training step do   
9: Sample $( x ^ { a } , \breve { p } ^ { a } , \dot { x } ^ { b } , p ^ { b } , k )$ from $\mathcal { D } _ { \mathrm { e d i t } } ;$ Compute $Z ^ { a } , Z ^ { b }$ and locate their factor-k spans.   
10: Replace the designated spans as in Eq. 4; Update ϕ using $\mathcal { L } _ { \mathrm { e d i t } } ;$ keep $\theta _ { 0 } ,$ η<sub>0</sub> fixed.   
11: end for   
12: Save $\phi ^ { * }  \phi$ and freeze the DiT.   
13: Stage III: Learn Prediction From Limited Inputs.   
14: for each training step do   
15: Sample $( x , p )$ from $\mathcal { D } _ { \mathrm { o b s } }$ and observation mode $m \sim \pi .$   
16: Construct ${ \boldsymbol { u } } ^ { \mathrm { t } } = { \boldsymbol { p } }$ i $: m = { \mathrm { t } } ,$ and $\boldsymbol { u } ^ { \mathrm { t v } } = ( p , x )$ otherwise.   
17: Compute $Z _ { \theta , \eta } ( u ^ { m } )$ and update $\theta , \eta$ using $\mathcal { L } _ { \mathrm { F M } } ( x \mid Z _ { \theta , \eta } ( u ^ { m } ) ; \phi ^ { * } )$ ; keep $\phi ^ { * }$ fixed.   
18: end for

## 3.1 PRELIMINARY: FRAMEWORK AND PAIRED DATA

The generator $( i . e .$ , Video DiT) must learn both to use MLLM information and to respond appropriately in a factor-aware manner. We provide these signals through paired observations and paired edits, without introducing a new representation network.

## 3.1.1 FRAMEWORK: MLLM-CONDITIONED VIDEO DIFFUSION

MLLM-conditioned video diffusion (Yang et al., 2026a; Wei et al., 2026) connects an MLLM to a pretrained Video DiT. The DiT is trained through flow matching (Lipman et al., 2022) and receives three concatenated conditions: $C = [ C ^ { \mathrm { t e x t } } ; C ^ { \mathrm { m l l m } } ; C ^ { \mathrm { r e f } } ]$ , comprising text-encoder features, MLLM features, and VAE-encoded visual features, respectively. Here, $[ \cdot ; \cdot ]$ denotes concatenation along the token dimension. Individual backbones may omit a pathway. For example, UniVideo (Wei et al., 2026) removes the original text encoders.

The MLLM condition is produced through the connector. Given a text or multimodal input $u ,$ the MLLM $F _ { \theta }$ produces hidden states $H _ { \boldsymbol \theta } ( u ) = F _ { \boldsymbol \theta } ( u )$ , which the connector $P _ { \eta }$ maps to the DiT conditioning space:

$$
C ^ { \mathrm { m l l m } } = Z _ { \boldsymbol { \theta } , \eta } ( \boldsymbol { u } ) = P _ { \eta } \big ( H _ { \boldsymbol { \theta } } ( \boldsymbol { u } ) \big ) ,\tag{1}
$$

where θ and $\eta$ are the MLLM and connector parameters. We retain the original hidden-state extraction and connector. Our factor-level interventions operate on the projected sequence $Z _ { \theta , \eta } ( u )$ , after the connector and before the DiT.

For a target video x with VAE latent $z _ { \mathrm { 0 } } ,$ flow matching constructs $z _ { t } = \alpha _ { t } z _ { 0 } + \sigma _ { t } \epsilon ,$ where $\epsilon \sim \mathcal { N } ( 0 , I )$ is Gaussian noise, t is the flow time, and $\alpha _ { t } , \sigma _ { t }$ specify the interpolation schedule. The DiT $G _ { \phi }$ , with parameters $\phi ,$ predicts the target velocity $v ^ { * } = \partial z _ { t } / \partial t :$

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { F M } } ( x \mid C ; \phi ) = \mathbb { E } _ { t , \epsilon } \left[ \left\| G _ { \phi } ( z _ { t } , t \mid C ) - v ^ { * } \right\| _ { 2 } ^ { 2 } \right] . } \end{array}\tag{2}
$$

We follow the pretrained backbone’s flow schedule and sampling procedure. This objective supervises generation under the combined conditions, but does not determine their individual contributions. The DiT may rely primarily on text and VAE features, so accurate reconstruction alone does not establish effective use of MLLM latents.

## 3.1.2 PAIRED DATA

Two Observation Modes, One Video Target. For each training video x and its description $p ,$ we construct two inputs: ${ \boldsymbol { u } } ^ { \mathrm { t } } = { \boldsymbol { p } }$ and $\boldsymbol { u } ^ { \mathrm { t v } } = ( p , x )$ . Both are supervised with the same target video $x .$ The text-only input requires inference from a description, while the text-plus-video input provides direct visual evidence. We denote these examples by $\mathcal { D } _ { \mathrm { o b s } }$ . The complete target video is never supplied to the text-only branch.

Selective Factors Change, the Others Remain. We construct the editing set $\mathcal { D } _ { \mathrm { e d i t } }$ through factorspecific selection and generation of source–target video pairs, covering changes to character, scene, action, and camera motion. Each tuple $( x ^ { a } , p ^ { a } , x ^ { b } , p ^ { b } , k )$ contains two videos $x ^ { a } , x ^ { b }$ and their descriptions $p ^ { a } , p ^ { b }$ , with only the selected factor $k \in \mathcal { F } =$ {character, scene, action, camera} modified. Both descriptions are structured prompts: we rewrite each caption into four ordered fields, character, scene, action, and camera, and obtain $p ^ { b }$ by updating only the edited factor’s field while retaining the others. Field labels and boundaries locate the corresponding token spans in the MLLM hidden states and projected latents, giving each factor an explicit address for intervention. Span lengths follow the text and need not match across examples, and we do not assume the contextualized spans are fully disentangled.

## 3.2 STAGE I: MLLM-ONLY GENERATOR GROUNDING

MLLM representations can influence generation only if the DiT can reliably decode and use them. In the original architecture, native text-encoder and VAE-reference pathways allow the DiT to reduce the generation loss without sufficiently relying on MLLM latents. Stage I therefore isolates the MLLM pathway and trains the DiT under both limited and complete observations. We freeze the pretrained MLLM and connector and update only the DiT.

Mixed-Observation Latent-Only Grounding. For each video $_ x$ and description $^ { p , }$ we use the text-only input ${ \boldsymbol { u } } ^ { \mathrm { t } } = { \boldsymbol { p } }$ and the text-plus-complete-video input $\boldsymbol { u } ^ { \mathrm { t v } } = ( p , x )$ . The frozen MLLM and connector produce $Z _ { \theta _ { 0 } , \eta _ { 0 } } ( u ^ { m } )$ for observation mode $m \in \{ \mathrm { t } , \mathrm { t v } \}$ . In both modes, the DiT receives MLLM latents as its only content condition; the separate text-encoder and VAE-reference conditions are disabled. Dense observations ground the DiT in the visual and temporal information carried by MLLM latents, while text-only observations maintain compatibility with limited-input generation and the text-based interventions in Stage II.

Mixed-Observation Objective. Stage I minimizes

$$
\mathcal { L } _ { \mathrm { g r o u n d } } ( \phi ) = \mathbb { E } _ { ( x , p ) \sim \mathcal { D } _ { \mathrm { o b s } } } \left[ \sum _ { m \in \{ \mathbf { t } , \mathbf { t } \mathbf { v } \} } \rho _ { m } \mathcal { L } _ { \mathrm { F M } } \left( x \mid Z _ { \theta _ { 0 } , \eta _ { 0 } } ( u ^ { m } ) ; \phi \right) \right] ,\tag{3}
$$

where $\rho _ { \mathrm { t } } + \rho _ { \mathrm { t v } } = 1$ . We optimize $\phi$ while keeping $\theta _ { 0 }$ and $\eta _ { 0 }$ fixed. Because alternative content pathways are unavailable, the DiT must learn to use MLLM latents across both observation modes. Stage I adapts the generator to fixed MLLM representations, while Stage III later reverses this direction and adapts the MLLM pathway through a fixed generator.

## 3.3 STAGE II: FACTOR-AWARE CONTROL IMPROVEMENT

Using MLLM features is not sufficient for factor-aware control. The DiT must also learn which output changes correspond to a particular change in its conditions. Stage II introduces factor-level latent augmentation with explicit video targets. The MLLM and connector remain frozen, and training uses text-only inputs with MLLM-latent-only conditioning.

Factor-Span Replacement. For a pair in $\mathcal { D } _ { \mathrm { e d i t } } ,$ let $Z ^ { a } = Z _ { \theta _ { 0 } , \eta _ { 0 } } ( p ^ { a } )$ and $Z ^ { b } = Z _ { \theta _ { 0 } , \eta _ { 0 } } ( p ^ { b } )$ be the projected latents of the two descriptions. Define $\mathcal { R } _ { k } ( A , B )$ as replacing the factor-k span in $A$ with the corresponding span from $B ,$ leaving the remaining tokens unchanged. The augmented conditions are

$$
Z _ { k } ^ { a  b } = \mathcal { R } _ { k } ( Z ^ { a } , Z ^ { b } ) , \qquad Z _ { k } ^ { b  a } = \mathcal { R } _ { k } ( Z ^ { b } , Z ^ { a } ) .\tag{4}
$$

Replacement is applied after the connector and before the DiT, so no additional projection follows. Complete spans are removed and inserted, so replacement does not require equal token counts; sequence lengths and masks are updated accordingly. Since the pair changes only factor $k ,$ the condition $Z _ { k } ^ { a \dot {  } b }$ is supervised with $x ^ { b }$ , and the reverse condition is supervised with $x ^ { a }$

Editing Objective. We update the DiT using both replacement directions:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { e d i t } } ( \phi ) = \mathbb { E } _ { ( { x } ^ { a } , { p } ^ { a } , { x } ^ { b } , { p } ^ { b } , k ) \sim \mathcal { D } _ { \mathrm { e d i t } } } [ \mathcal { L } _ { \mathrm { F M } } \big ( { x } ^ { b } \mid Z _ { k } ^ { a  b } ; \phi \big ) + \mathcal { L } _ { \mathrm { F M } } \big ( { x } ^ { a } \mid Z _ { k } ^ { b  a } ; \phi \big ) ] . } \end{array}\tag{5}
$$

The target video supervises the effect of the replaced span and preservation of the remaining factors. Thus, factor awareness is learned through the generator’s response to controlled replacements, rather than through a feature-separation objective. We denote the DiT parameters after this stage by $\phi ^ { * }$

## 3.4 STAGE III: PREDICTION FROM LIMITED INPUTS

The first two stages establish a generator that uses MLLM features and responds to factor-aware interventions. Stage III uses this fixed generator to optimize the MLLM pathway itself. The goal is to produce generation-compatible representations from limited observations, while complete-video observations provide a richer training mode for the same video target.

Shared Targets Across Observation Modes. We freeze the DiT at $\phi ^ { * }$ and jointly fine-tune the MLLM and connector, initialized from $\theta _ { 0 }$ and $\eta _ { 0 }$ . For each video x and description $p ,$ we use the text-only input ${ \boldsymbol { u } } ^ { \mathrm { t } } = { \boldsymbol { p } }$ and the text-plus-complete-video input $\boldsymbol { u } ^ { \mathrm { t v } } = \left( p , x \right)$ . Both observation modes use MLLM-latent-only conditioning and are supervised with the same target video x.

We define the mixed-observation objective as

$$
\mathcal { L } _ { \mathrm { o b s } } ( \theta , \eta ) = \mathbb { E } _ { ( x , p ) \sim \mathcal { D } _ { \mathrm { o b s } } } \left[ \sum _ { m \in \{ \mathfrak { t } , \mathfrak { t } \} } \pi _ { m } \mathcal { L } _ { \mathrm { F M } } \left( x \mid Z _ { \theta , \eta } ( u ^ { m } ) ; \phi ^ { * } \right) \right] ,\tag{6}
$$

where $\pi _ { m }$ denotes the sampling weight of observation mode $m ,$ , with $\pi _ { \mathrm { t } } + \pi _ { \mathrm { t v } } = 1$ . We minimize Eq. 6 over θ and $\eta ,$ while keeping $\phi ^ { * }$ fixed.

Prediction Through a Fixed Generator. Although the DiT is frozen, gradients still pass through it to the connector and MLLM. Under text-only observation, the MLLM pathway must produce representations that let the fixed generator reconstruct the target video’s semantic, motion, and temporal content from limited evidence. Under complete-video observation, the MLLM has direct visual evidence but must express it through the same connector and generation interface. Training both modes toward a shared video target therefore adapts the MLLM pathway without changing the DiT.

## 3.5 INFERENCE AND CINEMATIC CONTROL

At inference, FAPL supports three cinematic applications within one architecture. (i) Generation from Scripts and References. Given a script and optional character or scene references, the MLLM produces latents that condition the DiT through the trained connector and original interface, followed by flow-based sampling and VAE decoding. No complete target video is needed. The DiT can use MLLM latents alone or together with the original text and VAE reference conditions as separate inference configurations. (ii) Factor-Aware Control. We modify only the target factor’s field in the structured prompt and re-encode the edited prompt through the MLLM and connector. We then replace the corresponding span in the original projected latents with the edited field’s span, keeping all other spans unchanged, and condition the DiT on the resulting sequence. We evaluate both the requested change and non-target preservation. (iii) Multi-Shot Generation. Shots are generated independently from their descriptions and available references. Shared character and scene references promote consistency, while action and camera instructions vary by shot.

## 4 EXPERIMENTS

## 4.1 IMPLEMENTATION

We build FAPL on UniVideo (Wei et al., 2026) with a 13B DiT and compare with the original Uni-Video and Omni-Video 2 (Yang et al., 2026a) models (1.3B, 14B DiTs). Stages I and III sample textonly and text-plus-complete-video observations at a 1:1 ratio, while Stage II uses text-only inputs with factor-span replacement. We train Stages I–III for 10K, 15K, and 10K steps, respectively, using AdamW with a learning rate of $1 \times 1 0 ^ { - 5 }$ , zero weight decay, and a maximum gradient norm of 1.0.

Table 1: Quantitative comparison across three evaluation tasks.
<table><tr><td></td><td colspan="3">Sparse-to-Video</td><td>Factor-Aware Editing</td><td colspan="3">Multi-Shot Consistency</td></tr><tr><td>Method</td><td>Motion Order (%)↑</td><td>Trans. Error↓</td><td>Rot. Error (rad)↓</td><td>∩-Acc (%)↑</td><td></td><td></td><td>Subject↑ Scene↑ FaceSim-Arc↑</td></tr><tr><td>Omni-Video 2 (1.3B)</td><td>33</td><td>0.1084</td><td>1.87</td><td>28</td><td>0.435</td><td>0.373</td><td>0.412</td></tr><tr><td>Omni-Video 2 (14B)</td><td>43</td><td>0.0817</td><td>1.49</td><td>37</td><td>0.476</td><td>0.423</td><td>0.468</td></tr><tr><td>UniVideo (13B)</td><td>48</td><td>0.0763</td><td>1.54</td><td>42</td><td>0.462</td><td>0.427</td><td>0.451</td></tr><tr><td>FAPL (Ours)</td><td>61</td><td>0.0586</td><td>1.02</td><td>58</td><td>0.516</td><td>0.471</td><td>0.523</td></tr></table>

Training uses the bfloat16 (BF16) format and gradient checkpointing. Inference uses 50 sampling steps and a classifier-free guidance (CFG) scale of 5.0. All three stages share 100K quality-filtered source–target video pairs (25K per factor), constructed through factor-specific selection and generation using CoinVE-200K (Long et al., 2026) as the primary source and OmniVBench’s released training data (Li et al., 2026) for supplementary videos and references.

## 4.2 EVALUATION DATA, TASKS, AND METRICS

Test Data. CineFactor-10K comprises 10K single-shot test cases constructed from held-out source groups in CoinVE-200K (Long et al., 2026) and the training dataset released with OmniVBench (Li et al., 2026), with no source-video overlap with our training data.

Tasks. (i) Sparse-to-Video Generation. We evaluate on 5K cases using text-only prompts. (ii) Factor-Aware Editing. We evaluate on 5K cases, with 1.25K cases per factor. Each case modifies only the target factor’s field in the structured prompt while retaining the other three fields unchanged. All methods receive the same modified prompt and permitted visual inputs through their standard inference interfaces. (iii) Multi-Shot Generation. We evaluate separately on the 100 sequences in the official ShotVerse-Bench (Yang et al., 2026c) test split. Each sequence comprises two to four shots in their original order, with shared character and scene references and shot-specific action and camera instructions.

Metrics. (i) Temporal generation from limited inputs. Within shots, we measure action sequencing with Motion Order (Zheng et al., 2025) and camera-trajectory fidelity with Translation Error and Rotation Error (radians) (Yang et al., 2026c), using frozen π<sup>3</sup> (Wang et al., 2025c) for trajectory recovery. To complement these action/camera measures, multi-shot evaluation assesses character and scene consistency as shot-specific instructions vary, using Subject Consistency, Scene Consis tency (Wang et al., 2025b), and FaceSim-Arc for facial identity (Yuan et al., 2025). (ii) Factor-aware control. We evaluate single-factor edits using FiVE’s ∩-Acc (Li et al., 2025), requiring both yes/no and multiple-choice checks to pass.

## 4.3 COMPARISON AND ANALYSIS

Tab. 1 compares FAPL with Omni-Video 2 and UniVideo on sparse-to-video generation, factoraware editing, and multi-shot generation. FAPL improves action ordering and camera-trajectory fi delity from text alone, achieves higher ∩-Acc for requested factor changes, and obtains stronger Subject and Scene Consistency across shots, indicating that shot-specific instructions remain compatible with shared character and scene content. For multi-shot generation, FAPL achieves a FaceSim-Arc score of 0.523, compared with 0.412 for Omni-Video 2 (1.3B), 0.468 for Omni-Video 2 (14B), and 0.451 for UniVideo (13B). This improvement complements the Subject and Scene Consistency results, indicating stronger facial identity consistency as action and camera instructions vary across shots. In the examples shown in Figs. 4, 5, and 6, UniVideo captures the broad content but misses specified details such as hair length, hand gestures, or the requested orbit, and some edits also alter non-target factors. FAPL follows these specifications more closely while preserving unrelated factors, and maintains the shared setting and character appearance across shots. These comparisons suggest that FAPL improves instruction following and cross-shot consistency on the evaluated tasks.

![](images/ef17d81b37d1ebf87c151bcf35ec21f35d8c7fc3f8f26550caa9b729db478bd0.jpg)  
Figure 4: Under limited inputs, baseline models fail to predict temporal visual details. With only text and no visual references, we compare UniVideo with FAPL (Ours), highlighting representative baseline limitations across four cinematic factors. (i) Character. Long hair is generated as short hair, failing appearance details specified only by text. (ii) Scene. The background is not preserved and incorrect content appears. (iii) Action. The requested sequence, e.g., gesturing outward with the left hand and extended fingers, is missing. (iv) Camera. UniVideo ignores the instruction and keeps the camera static, while Ours orbits around the character. FAPL better realizes promptspecified appearance, scene, and motionfrom text alone.

![](images/1635cce7312b4c0e99c87e347cafe3c61cc165bf88162c37b902d666b2aa0464.jpg)  
Figure 5: FAPL improves factor-aware control. For each factor, we modify only the corresponding description in the structured prompt and generate twice, comparing UniVideo with FAPL (Ours). UniVideo does not confine the change to the requested factor and alters unrelated content, whereas Ours modifies the selected factor while preserving the others. This indicates thatfactor-level latent augmentation enables selective control.

## 4.4 ABLATION STUDY

Tab. 2 evaluates the three training stages on UniVideo (13B). Stage I improves action ordering and reduces both camera errors relative to the pretrained model. Fig. 7 further illustrates improved gener ation under MLLM-only conditioning after Stage I on Omni-Video 2 (1.3B and 14B) and UniVideo (13B). For Stage II, training with factor swapping achieves 57% ∩-Acc, compared with 48% for training without factor swapping under the same paired data and optimization budget. Motion Order and camera errors do not improve over this control, suggesting that the benefit is concentrated in factor-aware editing. For Stage III, mixed-observation training improves Motion Order from 57% to 61% and reduces translation and rotation errors from 0.0629 to 0.0586 and from 1.26 to 1.02 radians, respectively, compared with text-only training. Editing accuracy remains comparable to the Stage II checkpoint, changing from 57% to 58%. These results suggest that complete-video observations during training help action ordering and camera control under prompt-only inference while retaining factor-aware editing performance.

![](images/d8db20f4a2878d5a22a1b00322d0e92a2fc95a47c14744a0ea6a9112fd2a10e6.jpg)  
Figure 6: FAPL supports consistent multi-shot generation without cross-shot memory.

![](images/bf452cfda7b67e3a95e90ade9c637b883536ce1bc011793d63d2adbf22120fcf.jpg)  
Figure 7: MLLM latent enhancement generalizes across MLLM–DiT models.

Table 2: Ablation of FAPL training stages on UniVideo (13B).
<table><tr><td colspan="2">Exp Index Training Configuration</td><td colspan="4">Motion Order (%)↑ Trans. Error↓ Rot. Error (rad)↓ ∩-Acc (%)↑</td></tr><tr><td>0</td><td>Pretrained UniVideo</td><td>48</td><td>0.0763</td><td>1.54</td><td>42</td></tr><tr><td>1</td><td>Stage I only</td><td>54</td><td>0.0708</td><td>1.38</td><td>44</td></tr><tr><td>2</td><td>Stage II: training without factor swapping (based on Exp. 1)</td><td>54</td><td>0.0654</td><td>1.30</td><td>48</td></tr><tr><td>3</td><td>Stage II: training with factor swapping (based on Exp. 1)</td><td>53</td><td>0.0671</td><td>1.32</td><td>57</td></tr><tr><td>4</td><td>Stage III: text-only training (based on Exp. 3)</td><td>57</td><td>0.0629</td><td>1.26</td><td>56</td></tr><tr><td>5</td><td>Stage III: (Full FAPL) mixed-observation training (based on Exp. 3)</td><td>61</td><td>0.0586</td><td>1.02</td><td>58</td></tr></table>

## 5 CONCLUSION

This work studies how MLLM representations can support temporal inference and factor-aware control in cinematic video generation. FAPL makes these representations actionable by grounding the DiT in MLLM latents, supervising factor-specific changes and preservation, and learning prediction from limited inputs. FAPL demonstrates that targeting the right bottleneck delivers substantial gains without added architectural complexity. Our findings further offer insights into how multimodal reasoning and generation can reinforce each other: reasoning guides controllable synthesis, while generation provides grounded supervision for learning to infer missing content from limited observations.

Limitations and Future Work. Although FAPL addresses a fundamental weakness of existing MLLM-conditioned DiTs by making MLLM latents more actionable, its performance remains bounded by the underlying models. Current MLLMs struggle to interpret 3D blockouts or reason about complex camera motion for DiT control, while the pretrained DiT limits visual fidelity and complex factor realization, such as realistic acting. Future work may incorporate stronger MLLMs and more capable video generators.

## REFERENCES

Mahmoud Assran, Quentin Duval, Ishan Misra, Piotr Bojanowski, Pascal Vincent, Michael Rabbat, Yann LeCun, and Nicolas Ballas. Self-supervised learning from images with a joint-embedding predictive architecture. In 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 15619–15629. IEEE, 2023.

Mido Assran, Adrien Bardes, David Fan, Quentin Garrido, Russell Howes, Matthew Muckley, Ammar Rizvi, Claire Roberts, Koustuv Sinha, Artem Zholus, et al. V-jepa 2: Self-supervised video models enable understanding, prediction and planning. arXiv preprint arXiv:2506.09985, 2025.

Adrien Bardes, Quentin Garrido, Jean Ponce, Xinlei Chen, Michael Rabbat, Yann LeCun, Mahmoud Assran, and Nicolas Ballas. Revisiting feature prediction for learning visual representations from video. arXiv preprint arXiv:2404.08471, 2024.

Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023.

Mohamed Elmoghany, Ryan Rossi, Seunghyun Yoon, Subhojyoti Mukherjee, Eslam Mohamed Bakr, Puneet Mathur, Gang Wu, Viet Dac Lai, Nedim Lipka, Ruiyi Zhang, et al. A survey on long-video storytelling generation: architectures, consistency, and cinematic quality. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 7082–7094, 2025.

Haopeng Fang, Di Qiu, Binjie Mao, and He Tang. Motioncharacter: Fine-grained motion controllable human video generation. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, pp. 3831–3839, 2026.

Sharath Girish, Tsai-Shien Chen, Zhikang Dong, Mukesh Singhal, Hao Chen, Sergey Tulyakov, and Aliaksandr Siarohin. Cineorchestra: Unified entity-centric conditioning for cinematic video generation. arXiv preprint arXiv:2606.13768, 2026.

Kaiyi Huang, Yukun Huang, Yu Li, Jianhong Bai, Xintao Wang, Zinan Lin, Xuefei Ning, Jiwen Yu, Yu Wang, and Xihui Liu. Cinescene: Implicit 3d as effective scene representation for cinematic video generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 25381–25392, 2026a.

Yuyang Huang, Yabo Chen, Wenrui Dai, Ziyang Zheng, Haibin Huang, Chi Zhang, Junni Zou, Hongkai Xiong, and Xuelong Li. Cineweaver: Training-free reference-controllable multi-shot long video generation for cinematic storytelling. arXiv preprint arXiv:2607.26529, 2026b.

Zeyinzi Jiang, Zhen Han, Chaojie Mao, Jingfeng Zhang, Yulin Pan, and Yu Liu. Vace: All-inone video creation and editing. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 17191–17202, October 2025.

Xuan Ju, Weicai Ye, Quande Liu, Qiulin Wang, Xintao Wang, Pengfei Wan, Di Zhang, Kun Gai, and Qiang Xu. Fulldit: Video generative foundation models with multimodal control via full attention. In 2025 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 15737– 15747, 2025.

Diederik P Kingma and Max Welling. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2013.

Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024.

Minghan Li, Chenxi Xie, Yichen Wu, Lei Zhang, and Mengyu Wang. Five: A fine-grained video editing benchmark for evaluating emerging diffusion and rectified flow models. arXiv preprint arXiv:2503.13684, 2025.

Wenxue Li, Peiyan Guan, Haoyang Jiang, Junxian Cai, Hualuo Liu, Chunjie Zhang, Chong Guan, Kai Huang, Songlian Li, Taiyi Wu, Yongjian Yu, Xiaotong Zhao, Alan Zhao, Eric Liu, Xi Chen, Yu Liu, and Lei Zhu. Omnivbench: A benchmark and large-scale dataset for omni reference-tovideo generation, 2026. URL https://arxiv.org/abs/2609.22069.

Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022.

Lijie Liu, Tianxiang Ma, Bingchuan Li, Zhuowei Chen, Jiawei Liu, Gen Li, Siyu Zhou, Qian He, and Xinglong Wu. Phantom: Subject-consistent video generation via cross-modal alignment. In 2025 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 14951–14961, 2025.

Fuchen Long, Cong Wang, Zitao Gao, Wenhao Zhong, Yu Cheng, Xiaolu Hou, Yan Li, Xiao Cao, Xinlong Sun, Xi Chen, et al. Coinve-200k: A large-scale high-quality dataset for compositional instruction-guided video editing. arXiv preprint arXiv:2608.17566, 2026.

Yihao Meng, Hao Ouyang, Yue Yu, Qiuyu Wang, Wen Wang, Ka Leong Cheng, Hanlin Wang, Shuailei Ma, Yixuan Li, Cheng Chen, et al. Holocine: Holistic generation of cinematic multishot long video narratives. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 461–471, 2026.

William Peebles and Saining Xie. Scalable diffusion models with transformers. In 2023 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 4172–4182. IEEE, 2023.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pp. 8748–8763. PmLR, 2021.

Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal ofmachine learning research, 21(140):1–67, 2020.

Liming Tan, Ye Chen, Hao Zhang, Lirong Qian, Feifei Li, and Bingbing Ni. Unimoca: Unifying motion and camera controls as visual proxies for faithful human video generation, 2026a. URL https://arxiv.org/abs/2608.01944.

Zhiyu Tan, Hao Yang, Luozheng Qin, Jia Gong, Mengping Yang, and Hao Li. Omni-video: Democratizing unified video understanding and generation, 2026b. URL https://arxiv.org/ abs/2507.06119.

Zhenchen Tang, Yang Li, Songlin Yang, Bo Peng, Xiaotong Zhao, Shuai Li, Haotian Fan, Alan Zhao, and Jing Dong. Rewardverse: Rubric-guided policy optimization for video reward modeling. arXiv preprint arXiv:2609.22947, 2026.

Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.

Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024.

Qinghe Wang, Yawen Luo, Xiaoyu Shi, Xu Jia, Huchuan Lu, Tianfan Xue, Xintao Wang, Pengfei Wan, Di Zhang, and Kun Gai. Cinemaster: A 3d-aware and controllable framework for cinematic text-to-video generation. In Proceedings of the Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers, pp. 1–10, 2025a.

Qinghe Wang, Xiaoyu Shi, Baolu Li, Weikang Bian, Quande Liu, Huchuan Lu, Xintao Wang, Pengfei Wan, Kun Gai, and Xu Jia. Multishotmaster: A controllable multi-shot video generation framework. arXiv preprint arXiv:2512.03041, 2025b.

Sophie L Wang, Phillip Isola, and Brian Cheung. The truth lies somewhere in the middle (of the generated tokens). arXiv preprint arXiv:2605.09969, 2026.

Yifan Wang, Jianjun Zhou, Haoyi Zhu, Wenzheng Chang, Yang Zhou, Zizun Li, Junyi Chen, Jiangmiao Pang, Chunhua Shen, and Tong He. π<sup>3</sup>: Scalable permutation-equivariant visual geometry learning. arXiv preprint arXiv:2507.13347, 2025c.

Zhouxia Wang, Ziyang Yuan, Xintao Wang, Yaowei Li, Tianshui Chen, Menghan Xia, Ping Luo, and Ying Shan. Motionctrl: A unified and flexible motion controller for video generation. 2023.

Cong Wei, Quande Liu, Zixuan Ye, Qiulin Wang, Xintao Wang, Pengfei Wan, Kun Gai, and Wenhu Chen. Univideo: Unified understanding, generation, and editing for videos. In The Fourteenth International Conference on Learning Representations, 2026.

Xiaoxue Wu, Xinyuan Chen, Yaohui Wang, and Yu Qiao. Shotdirector: Directorially controllable multi-shot video generation with cinematographic transitions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 2079–2089, 2026.

Junfei Xiao, Ceyuan Yang, Lvmin Zhang, Shengqu Cai, Yang Zhao, Yuwei Guo, Gordon Wetzstein, Maneesh Agrawala, Alan Yuille, and Lu Jiang. Captain cinema: Towards short movie generation. In International Conference on Learning Representations, volume 2026, pp. 83349–83375, 2026.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

Hao Yang, Zhiyu Tan, Jia Gong, Luozheng Qin, Hesen Chen, Xiaomeng Yang, Yuqing Sun, Yuetan Lin, Mengping Yang, and Hao Li. Omni-video 2: Scaling mllm-conditioned diffusion for unified video generation and editing, 2026a. URL https://arxiv.org/abs/2602.08820.

Songlin Yang, Gaurav Parmar, Ozgur Kara, Jiaju Ma, Lvmin Zhang, Or Patashnik, Fabian Caba Heilbron, Jun-Yan Zhu, Daniel Cohen-Or, Maneesh Agrawala, and Anyi Rao. Ai for creative visual content generation, editing and understanding. In Proceedings of the Special Interest Group on Computer Graphics and Interactive Techniques Conference Technical Workshops, SIGGRAPH Technical Workshops ’26, New York, NY, USA, 2026b. Association for Computing Machinery. ISBN 9798400725517. doi: 10.1145/3799828.3811415. URL https://doi.org/10.1145/3799828.3811415.

Songlin Yang, Zhe Wang, Xuyi Yang, Songchun Zhang, Xianghao Kong, Taiyi Wu, Xiaotong Zhao, Ran Zhang, Alan Zhao, and Anyi Rao. Shotverse: Advancing cinematic camera control for textdriven multi-shot video creation. arXiv preprint arXiv:2603.11421, 2026c.

Songlin Yang, Xiaotong Zhao, Jiacheng Zhang, Zhe Wang, Toyota Li, Eric Liu, Alan Zhao, and Anyi Rao. Adaptive reward routing: Dynamic multi-reward optimization for joint audio-video diffusion via forward-process rl. arXiv preprint arXiv:2609.37200, 2026d.

Songlin Yang, Haobin Zhong, Ruilin Zhang, Xiaotong Zhao, Shuai Li, Kai Zheng, Xuyi Yang, Zhe Wang, Zhenchen Tang, Yang Li, et al. Evalverse: Pipeline-aware and expert-calibrated benchmarking for professional cinematic video generation. arXiv preprint arXiv:2605.23271, 2026e.

Zixuan Ye, Xuanhua He, Quande Liu, Qiulin Wang, Xintao Wang, Pengfei Wan, Di Zhang, Kun Gai, Qifeng Chen, and Wenhan Luo. Unic: Unified in-context video editing. In The Fourteenth International Conference on Learning Representations, 2026.

Shenghai Yuan, Jinfa Huang, Xianyi He, Yunyang Ge, Yujun Shi, Liuhan Chen, Jiebo Luo, and Li Yuan. Identity-preserving text-to-video generation by frequency decomposition. In Proceedings ofthe Computer Vision and Pattern Recognition Conference, pp. 12978–12988, 2025.

Peixuan Zhang, Zijian Jia, Kaiqi Liu, Shuchen Weng, Si Li, and Boxin Shi. Stage: Storyboardanchored generation for cinematic multi-shot narrative. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 659–669, 2026a.

Peixuan Zhang, Chang Zhou, Ziyuan Zhang, Hualuo Liu, Chunjie Zhang, Jingqi Liu, Xiaohui Zhou, Xi Chen, Shuchen Weng, Si Li, et al. A benchmark and multi-agent system for instructiondriven cinematic video compilation. In European Conference on Computer Vision, pp. 213–230. Springer, 2026b.

Dian Zheng, Ziqi Huang, Hongbo Liu, Kai Zou, Yinan He, Fan Zhang, Yuanhan Zhang, Jingwen He, Wei-Shi Zheng, Yu Qiao, and Ziwei Liu. VBench-2.0: Advancing video generation benchmark suite for intrinsic faithfulness. arXiv preprint arXiv:2503.21755, 2025.