# PosterText: Towards Unified Visual Text Generation and Editing for E-commerce Poster

Xiaoan Liu<sup>1,2∗</sup>, Lichen Ma<sup>2,3∗</sup>, Zipeng Guo<sup>2</sup>, Yu He<sup>2</sup>, Xiaoyan Su<sup>4</sup>, Shaojie Guo<sup>2</sup>, Jingling Fu<sup>2</sup>, Xiaolong Fu<sup>2</sup>, Hao Yang<sup>3</sup>, Tongxuan Liu<sup>2</sup>, Yu Guo<sup>3</sup>, Fei Wang<sup>3</sup>, Xinyi Liu<sup>1</sup>, Yongjun Zhang<sup>1†</sup>, Junshi Huang<sup>2‡</sup>

<sup>1</sup>Wuhan University, <sup>2</sup>JD.com, <sup>3</sup>State Key Laboratory of Human-Machine Hybrid Augmented Intelligence, Institute of Artificial Intelligence and Robotics, Xi’an Jiaotong University, <sup>4</sup>The Hong Kong University of Science and Technology (Guangzhou)

## Abstract

Automated e-commerce poster design requires both highquality poster generation and flexible editing of existing designs. However, most existing methods either target end-toend poster generation or follow multi-stage design pipelines, with limited capability for flexible and precise editing of existing posters. To enable unified generation and editing of e-commerce posters, we introduce Text Patch Generation and Editing, a unified task formulation that treats text patches as atomic units and covers four operations: poster generation, patch addition, patch deletion, and patch modification, with optional reference-guided style control. Based on this, we propose PosterText, a unified model trained with a four-stage curriculum, including text rendering pretraining, instruction-following training, reinforcement learning for preference alignment, and spatial guidance self-distillation for execution refinement. We further construct a large-scale dataset with patch-level annotations and a comprehensive benchmark for evaluation. Extensive experiments demonstrate that PosterText achieves competitive performance against existing generation and editing approaches, validating the efectiveness of the proposed framework.

## Introduction

E-commerce posters are a prevalent form of visual content in online commerce, integrating product images, visual backgrounds, and text elements such as slogans, prices, and brand names (Lin et al. 2023; Guo et al. 2025; Fan et al. 2026; Qin et al. 2026). These text elements are structured visual units with specific typography, colors, decorations, and spatial layouts, rather than simple strings. In practical scenarios, designers often modify existing posters instead of creating them from scratch, including replacing promotional texts, adjusting typography, adding new campaign elements, or removing outdated information. Therefore, an efective poster automation system should support both high-quality generation and flexible, style-controllable editing based on existing designs.

Existing poster automation methods mainly follow two paradigms. The first adopts a plan-and-render pipeline, which decomposes poster creation into multiple stages such as layout planning and rendering, providing explicit control over the generation process (Li et al. 2023; Hsu et al.

2023). However, these methods are constrained by the dependency between stages, where errors can accumulate throughout the pipeline, and local modifications often require reexecuting multiple components. The second explores end-toend difusion-based generation (Gao et al. 2025; Chen et al. 2025b; Qin et al. 2026), which directly synthesizes complete posters with improved visual quality. Despite their strong generation capability, these methods mainly focus on zero-toone creation and provide limited support for localized editing of existing designs. As a result, existing approaches remain insuficient for practical poster design workflows, where designers frequently need to iteratively refine existing layouts and introduce new content while preserving visual consistency. Therefore, developing a unified model that supports both poster generation and editing remains challenging. Such a model needs to achieve accurate character-level text rendering, reconcile the diferent requirements of open-ended generation and precise local editing, and enable reference-guided style control while maintaining overall visual coherence.

To address these challenges, we formulate Text Patch Generation and Editing, where text patches serve as the basic units for both generation and editing operations. The proposed framework unifies four operations within a single model: poster generation, patch addition, patch deletion, and patch modification, with optional reference-patch conditioning for style-controlled editing and generation. Based on this formulation, we propose PosterText, a unified model trained with a progressive four-stage curriculum. The model first learns accurate text rendering and patch synthesis, then acquires instruction-driven generation and editing capabilities. Subsequently, reinforcement learning aligns the model with human preferences through text accuracy, aesthetic quality, and background preservation rewards, and Spatial Guidance Self-Distillation (SGSD) further improves execution reliability by transferring mask-guided spatial knowledge to maskfree inference. To facilitate research in this direction, we construct PosterText-320K, a large-scale poster dataset with patch-level structural annotations and automatically generated editing pairs, and further build PosterText-Bench with a comprehensive evaluation protocol.

Our contributions are summarized as follows: (1) We introduce the task of text patch generation and editing, which treats text patches as atomic units and unifies generation, addition, deletion, and modification with reference-guided style control. Extensive experiments demonstrate the efectiveness of our approach. (2) We propose PosterText, a unified model trained with a four-stage curriculum covering text rendering, instruction following, reinforcement learning alignment, and self-distillation refinement, enabling both generation and editing within a single framework. (3) We construct a largescale dataset and benchmark with an automated data construction pipeline and comprehensive evaluation protocols.

![](images/d804080910b290832abeb0440feab4e2880fe99b84cc680cdb4914fe8989ce7b.jpg)  
Figure 1: Overview of PosterText framework and qualitative comparison. (a) Four unified operations: poster generation, patch addition, patch deletion, and patch modification. (b) PosterText enables both poster generation and localized editing within a single framework. (c) Qualitative comparison with closed-source baselines.

## Related Work

## E-commerce and Poster Generation

E-commerce posters are a crucial form of visual content in online commerce, requiring the harmonious integration of products, backgrounds, and marketing texts. Automated poster generation has thus attracted increasing attention, with the key challenge being the creation of visually appealing posters while maintaining product fidelity and text accuracy (Hsu et al. 2023; Chen et al. 2025a; Liu et al. 2026; Chen et al. 2026a). Early approaches typically follow a plan-andrender paradigm, decomposing generation into layout planning and background rendering stages (Li et al. 2023). While these methods provide explicit layout control, the errors introduced across stages and limited text rendering capability hinder their practical applications. Recent works have explored end-to-end generation byjointly modeling text, layout, and visual appearance, achieving impressive performance in generating complete posters from scratch (Gao et al. 2025; Chen et al. 2025b; Qin et al. 2026). However, real-world ecommerce scenarios often require localized modifications to existing designs, such as replacing promotional texts, adjusting typography styles, or adding new campaign elements. Existing approaches primarily focus on zero-to-one generation and lack the ability to perform localized and controllable modifications on specific poster components, making fine-grained editing challenging and limiting their practical usability. Developing a unified model that can both generate high-quality posters and enable flexible, controllable, and fine-grained content editing remains an open challenge.

## Visual Text Rendering and Editing

Text rendering in images has attracted increasing attention, as difusion models still struggle with accurate text generation, particularly for complex glyphs (Ma et al. 2023; Tuo, Geng, and Bo 2024; Ma et al. 2024, 2025a; Chen et al. 2026b; Lu et al. 2026). This challenge is especially critical for e-commerce posters, where text conveys essential product information and marketing semantics. Existing studies have improved text rendering from diferent perspectives. Any-Text (Tuo et al. 2024) and FLUX-Text (Lan et al. 2025) enhanced rendering accuracy through glyph-aware conditioning and improved text-image alignment. Recent works further investigate controllable text generation, with FonTS (Shi et al. 2025) and Calligrapher (Ma et al. 2025b) enabling typography and style customization, while UM-Text (Ma et al. 2026) introduced vision-language guidance for instructiondriven text editing. Despite these advances, existing methods mainly treat text as an independent rendering target, overlooking its interaction with the overall poster composition. In practical e-commerce designs, text is tightly coupled with typography, color, decoration, and spatial layout, requiring coordinated generation and editing with surrounding visual elements. To address this limitation, we propose a unified framework that integrates poster generation with patch-level text editing. By incorporating reference patch conditions, our framework enables style-controllable text creation and modification while maintaining visual consistency.

![](images/ddf171a37d72024c91aa2a7e412321c7df23524bd04fda56a1b7be04d7350b31.jpg)  
Figure 2: Overview of PosterText. PosterText is trained through four stages: text patch rendering pretraining, instruction-following training, reinforcement learning with human preferences, and Spatial Guidance Self-Distillation (SGSD). The framework supports four operations: generation, addition, deletion, and modification. For SGSD training, the teacher model uses mask guided inputs for spatial supervision, while the student learns mask-free localization through on-policy self-distillation.

## Method

We present PosterText, a unified framework for text patch generation and editing in e-commerce posters, as shown in Figure 2. Given a product image and a natural-language instruction, PosterText generates or edits posters by treating text patches as atomic units. The model is trained with a multi-stage curriculum that progressively learns text rendering and patch synthesis, extends to multi-task generation and editing, and further improves alignment and execution accuracy through reinforcement learning and self-distillation.

## Task Formulation

Text Patch Definition. We represent a text patch as $p =$ $( t , s , b , r )$ , where t denotes the text content, s encodes style attributes (e.g., font, size, and color), b represents an optional decorative background (set to empty for plain text), and $r =$ $( x , y , w , h )$ specifies its spatial region on the canvas. An ecommerce poster I is composed of a product image $I _ { \mathrm { p r o d } }$ and a set of text patches $\mathcal { P } = \dot { \{ { p _ { 1 } , \dots , { p _ { n } } } \} }$

Unified Framework. We model all tasks using a single function $f _ { \theta }$ , conditioned on an input image $I _ { \mathrm { i n } } ,$ a naturallanguage instruction c, and an optional set of reference patches $\mathcal { R } = \{ p _ { \mathrm { r e f } } ^ { ( 1 ) } , . . . , p _ { \mathrm { r e f } } ^ { ( K ) } \}$ , where $\mathcal { R } = \emptyset$ when no reference patch is provided:

$$
I _ { \mathrm { o u t } } = f _ { \theta } ( I _ { \mathrm { i n } } , c , \mathcal { R } )\tag{1}
$$

Specifically, task types are determined by the naturallanguage instruction $c \colon ( 1 )$ Poster generation synthesizes a complete poster from $I _ { \mathrm { p r o d } }$ , guided by c describing the desired content and style. (2) Patch addition inserts new text patches into an existing poster, where c specifies content and optionally location; style can be further guided by R. (3) Patch deletion removes regions specified in c. (4) Patch modification updates text content or style within a target region defined by $c ,$ optionally conditioned on $\mathcal { R } .$

## Stage I: Text Patch Rendering Pretraining

Accurate text rendering is fundamental for poster generation, as difusion models often sufer from character-level errors. Directly training instruction following tends to prioritize global layout while degrading text fidelity. We therefore first establish text rendering and patch generation through pretraining. Stage I includes two progressive sub-tasks. (A) Text Rendering: the model receives prompts containing OCR text and spatial positions, and renders the corresponding glyph sequence. (B) Patch Reconstruction: the model reconstructs complete text patches from natural-language descriptions, learning to compose content, style, and layout.

Training starts with Task A and gradually increases the proportion of Task B. Both tasks are optimized with the flow matching objective:

$$
\mathcal { L } _ { \mathrm { F M } } = \mathbb { E } _ { t , x _ { 0 } , \epsilon } \left[ \| v _ { \theta } ( x _ { t } , t , c ) - ( x _ { 0 } - \epsilon ) \| ^ { 2 } \right]\tag{2}
$$

where $x _ { t } = ( 1 - t ) \epsilon + t \cdot x _ { 0 }$ is the interpolated sample at time $t .$ This stage establishes reliable character-level rendering and then transitions to structured patch synthesis, forming the foundation for subsequent editing tasks.

## Stage II: Instruction-Following Training

Building upon the rendering capabilities from Stage I, the model is further trained to follow instructions and perform diverse operations, including poster generation, patch addition, modification, and deletion. We formulate all tasks with a unified input format $( I _ { \mathrm { i n } } , c , \mathcal { R } )$ , where the instruction c specifies the desired operation, target content, and optional style conditions. Training data is constructed by decomposing existing posters into operation tuples, allowing the model to learn diferent generation and editing behaviors under the same instruction framework. During Stage II, we jointly train the model with both mask-guided and mask-free samples, allowing it to acquire stronger spatial editing capabilities with explicit masks while supporting mask-free inference.

All operations are jointly optimized with the flow matching objective, with balanced sampling across tasks to ensure stable multi-task learning. After this stage, the model can interpret instructions and perform diferent poster generation and editing operations, including style-controllable generation and modification with optional reference patches.

## Stage III: Preference Alignment

The first two stages optimize pixel-level reconstruction objectives, which are insuficient to capture higher-level perceptual criteria in e-commerce posters, including text readability, visual aesthetics, and background consistency. Building upon the instruction-following capability acquired in Stage II, we further align the model with human preferences through reinforcement learning.

Optimization Objective. We adopt DifusionNFT (Zheng et al. 2025), an online RL algorithm compatible with the flow matching framework. Given online rollouts with normalized rewards r, the model is optimized by jointly improving high-reward behaviors and suppressing low-reward behaviors through implicit positive and negative policies:

$$
\mathcal { L } _ { \mathrm { R L } } = \mathbb { E } _ { c , x _ { 0 } \sim \pi ^ { \mathrm { o l d } } , t , \epsilon } \left[ r \cdot \| v _ { \theta } ^ { + } - v \| _ { 2 } ^ { 2 } + ( 1 - r ) \| v _ { \theta } ^ { - } - v \| _ { 2 } ^ { 2 } \right]\tag{3}
$$

where $v = x _ { 0 } - \epsilon$ denotes the target velocity and $v ^ { \mathrm { o l d } }$ is the EMA reference policy. The positive and negative branches are derived from the same velocity field, allowing the model to improve toward preferred outputs while preserving the flow matching formulation.

Reward Design. We design a composite reward tailored to e-commerce poster generation and editing:

$$
R = \alpha R _ { \mathrm { t e x t } } + \beta R _ { \mathrm { a e s t h e t i c } } + \delta R _ { \mathrm { b a c k g r o u n d } }\tag{4}
$$

where $R _ { \mathrm { t e x t } }$ measures OCR-based text accuracy to evaluate character-level rendering fidelity; $R _ { \mathrm { a e s t h e t i c } }$ assesses the overall visual quality; and R<sub>background</sub> measures the preservation of non-edited regions.

## Stage IV: Spatial Guidance Self-Distillation

After preference alignment, the model shows significant improvement in overall quality, but still sufers from execution errors in complex scenarios such as ghost text, missing characters, and incorrect operations. These failures mainly arise from insuficient localization and decomposition of complex instructions, making it dificult for the model to accurately determine where and how to perform each operation (Baron et al. 2026). The sparse reward signal in RL only evaluates the final output and cannot efectively attribute global feedback to specific intermediate-step errors. To improve execution accuracy, we introduce SGSD, which transfers the spatial guidance from mask-guided editing to a mask-free setting. Since explicit masks can provide precise region information and improve editing quality (Guo et al. 2026), SGSD aims to enable the model to achieve comparable execution quality without additional mask inputs.

During training, we construct teacher-student pairs with asymmetric conditions. The teacher receives an additional mask-guided condition $M _ { t } ~ = ~ ( I _ { \mathrm { i n } } , c , \mathcal { R }$ , mask), where the mask provides explicit spatial information for the target operation, while the student receives a mask-free condition $M _ { s } = ( I _ { \mathrm { i n } } , c , \mathcal { R } )$ . The student samples intermediate states along its own generation trajectory, and the teacher provides velocity predictions at the same states as distillation targets. The training objective is:

$$
\mathcal { L } _ { \mathrm { S G S D } } = \mathbb { E } _ { ( { x _ { 0 } } , y ) } \left[ \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \left\| { u _ { k } ^ { s } } - \mathrm { s g } ( { u _ { k } ^ { t } } ) \right\| _ { 2 } ^ { 2 } \right]\tag{5}
$$

where:

$$
u _ { k } ^ { s } = \nu _ { \theta } ( x _ { t _ { k } } ^ { s } , t _ { k } , M _ { s } ) , \quad u _ { k } ^ { t } = \nu _ { \bar { \theta } } ( x _ { t _ { k } } ^ { s } , t _ { k } , M _ { t } )\tag{6}
$$

$\operatorname { s g } ( \cdot )$ represents the stop-gradient operation, K denotes the number of intermediate timesteps sampled along each student trajectory, and <sup>¯</sup>θ denotes the teacher parameters updated by EMA. Unlike teacher-forcing based SFT (Fu et al. 2025; Tan et al. 2026), SGSD provides supervision along the student’s own generation trajectory, reducing the traininference discrepancy. The dense intermediate supervision across multiple time steps provides fine-grained correction for intermediate steps that sparse RL rewards cannot reach, further improving execution precision in complex mask-free scenarios.

## Dataset and Benchmark

Existing text generation datasets mainly focus on text rendering from scratch, while lacking editing operation pairs and patch-level structural annotations. To support systematic research on text patch generation and editing, we construct

![](images/b2440391c2a20212094a342cc6838498ddd8723326f02dcb23478a29a2ef055e.jpg)  
Figure 3: Overview of the data construction pipeline. The pipeline first extracts patch-level structural annotations from raw posters, and then constructs operation pairs for generation, addition, deletion, and modification.

PosterText-320K, a large-scale annotated poster dataset containing 120K posters for editing tasks and 200K posters for generation tasks. We further build PosterText-Bench with a comprehensive evaluation protocol. The overall data construction pipeline consists of two stages: first, extracting patch-level structural annotations from raw posters; second, constructing diverse operation pairs.

## Data Construction Pipeline

Phase I: Poster Structure Extraction. We collect a large number of e-commerce posters from public platforms and automatically recover patch-level structural information through a multi-stage pipeline, as illustrated in Figure 3: (1) text regions are detected and OCR is applied to obtain text content and spatial locations; (2) Qwen3.5-27B (Team 2026) VLM is used to describe visual attributes of each text patch, including font, color, background blocks, and text efects; (3) Flux.2-Klein-9B (Labs 2025) is applied to remove all text patches from posters, producing corresponding product images; (4) Gemini-3.1 Pro (Google 2025) is used to extract structured HTML representations from the posters; (5) lowquality samples are filtered based on OCR confidence, image resolution, perceptual similarity, and residual text artifacts in the generated product images.

Phase II: Paired Data Synthesis. Based on the extracted poster structures, we construct training pairs for four operations: generation, addition, deletion, and modification. We introduce an HTML-based structured representation that encodes each text patch with its content, visual attributes, and spatial layout, enabling precise patch-level manipulation and reference-based editing. Using the structured representation, we construct training pairs for four operations: (1) Generation: patch annotations are converted into natural-language descriptions to create product-conditioned poster generation pairs. (2) Addition: existing patches are removed and rerendered to construct addition pairs, with HTML attributes providing reference patch information for style-conditioned generation. (3) Deletion: target patches are removed according to their structural locations to generate deletion pairs. (4) Modification: patch content or visual attributes are changed while preserving the original layout, with HTML attributes enabling reference-guided style-controlled editing.

## Evaluation Protocol

We evaluate model outputs from three perspectives: text fidelity, background preservation, and editing quality.

Text Fidelity. Sentence Accuracy (Sen. Acc) and Normalized Edit Distance (NED) are used to evaluate text rendering accuracy, measuring sentence-level exact correctness and character-level similarity between generated and groundtruth texts, respectively.

Background Preservation. Masked-LPIPS is used to evaluate background preservation by measuring perceptual diferences between input and output images in non-edited regions. For poster generation, a VLM-based scorer is instead used to assess whether the product subject and content are faithfully preserved in the generated output.

Editing Quality. A vision-language model is used to evaluate editing quality from two aspects: instruction following, which measures whether the requested operation is correctly executed, and visual quality, which evaluates the coherence, style consistency, and naturalness of the edited results.

## Experiments

Implementation Details and Baselines. We build Poster-Text upon Qwen-Image-Edit and fine-tune it with LoRA. Stage I is trained for 50K steps with a cosine learning rate schedule and an initial learning rate of 1 × 10<sup>−4</sup>. The sampling ratio between text rendering task and patch reconstruction task gradually changes from 4:1 to 1:1 during the first 20K steps, with LoRA rank set to 128. Stage II is trained for 10K steps for instruction following. Stage III applies DiffusionNFT for RL alignment with 200 training steps, while Stage IV performs self-distillation with K = 20 and 500 training steps. More implementation details are provided in the supplementary material.

For evaluation, we compare PosterText with representative open-source and closed-source methods. The opensource baselines include Flux.2-Klein-9B (Labs 2025), Qwen-Image-Edit (Wu et al. 2025), and FireRed-Image-Edit (Zhou et al. 2025). The closed-source baselines include

![](images/895fac8fb969786f63dc319125065a8d77ca08d4a3fd89df8bb3c788f9c44f50.jpg)  
Figure 4: Visual comparison of PosterText with existing methods.

Seedream 5.0 (Seedream Team 2025), Nano Banana 2 (Deep-Mind 2025a), and Nano Banana Pro (DeepMind 2025b).

## Experimental Results

We conduct a unified evaluation of the above methods across all tasks on PosterText-Bench, with comprehensive results summarized in Table 1. Overall, PosterText achieves strong performance in text fidelity, background preservation, and instruction following, substantially outperforming existing open-source baselines and even surpassing closed-source models on key metrics. In terms of visual quality, PosterText narrows the gap with closed-source systems, demonstrating competitive generation and editing quality. As shown in Figure 1 (c) and 4, PosterText can better leverage complex style references to generate corresponding text patches while maintaining accurate text rendering and clean backgrounds. In contrast, open-source baselines often produce blurry or corrupted characters, while closed-source models generally achieve higher visual quality but may modify non-target regions, introduce unintended content, or fail to preserve the original patch semantics faithfully.

For Patch Addition, PosterText achieves the best text fidelity among all methods, with Sen.Acc of 0.763 and NED of 0.899, outperforming the closed-source systems. It also obtains the lowest Masked-LPIPS score, demonstrating that PosterText can insert new patches while efectively preserving the original poster structure. For Patch Deletion, which mainly evaluates region localization and background restoration rather than text rendering, PosterText achieves competitive Masked-LPIPS and instruction-following scores, indicating more accurate spatial control and cleaner content removal. Patch Modification represents the most challenging setting, as it requires simultaneous text regeneration, precise target localization, and preservation of surrounding regions. PosterText achieves strong performance in background preservation and instruction following, demonstrating efective localized editing while maintaining overall visual consistency. Some closed-source models achieve higher text fidelity on this task, which may benefit from large-scale proprietary training data and broader text rendering supervision.

In the Poster Generation task, closed-source models maintain advantage in instruction following and visual quality, benefiting from larger-scale training data, stronger aesthetic alignment, and more capable foundation models. Meanwhile, PosterText achieves comparable performance in product fidelity, demonstrating that the region control and content preservation abilities learned from editing tasks can efectively transfer to the generation setting. Despite their strong performance, closed-source models still face practical limitations in e-commerce deployment, including high costs for large-scale generation due to pay-per-use pricing, limited adaptability caused by the lack of fine-tuning access, and potential concerns regarding commercial data privacy when uploading product images to external services. In contrast,

<table><tr><td rowspan="2">Method</td><td colspan="5">Patch Addition</td><td colspan="4">Patch Deletion</td><td colspan="5">Patch Modification</td><td colspan="3">Poster Generation</td></tr><tr><td>S.A.</td><td>NED</td><td>M-L.</td><td>I.F.</td><td>V.Q.</td><td>M-L.</td><td>I.F.</td><td>V.Q.</td><td>S.A.</td><td>NED</td><td>M-L.</td><td></td><td>I.F.</td><td>V.Q.</td><td>I.F.</td><td>V.Q.</td><td>B.P.</td></tr><tr><td colspan="10">Closed-Source Methods</td><td colspan="7"></td></tr><tr><td>Seedream 5.0</td><td>0.403</td><td>0.585</td><td>0.195</td><td>4.40</td><td>4.64</td><td>0.122</td><td>4.68</td><td>4.69</td><td>0.735</td><td>0.884</td><td>0.102</td><td>4.33</td><td>4.65</td><td></td><td>4.10 4.50</td><td></td><td>4.80</td></tr><tr><td>Nano Banana 2</td><td>0.500</td><td>0.635</td><td>0.064</td><td>4.46</td><td>4.67</td><td>0.051</td><td>4.62</td><td>4.84</td><td>0.819</td><td>0.907</td><td>0.034</td><td>4.47</td><td></td><td>4.80</td><td>4.63</td><td>4.76</td><td>4.86</td></tr><tr><td>Nano Banana Pro</td><td>0.543</td><td>0.679</td><td>0.184</td><td>4.55</td><td>4.75</td><td>0.061</td><td>4.66</td><td>4.91</td><td>0.815</td><td>0.891</td><td></td><td>0.102</td><td>4.36</td><td>4.64</td><td>4.58</td><td>4.88</td><td>4.92</td></tr><tr><td colspan="10">Open-Source Methods</td><td colspan="7"></td></tr><tr><td>FireRed-Image-Edit</td><td>0.309</td><td>0.481</td><td>0.497</td><td>3.53</td><td>3.61</td><td>0.185</td><td>4.29</td><td>4.04</td><td>0.673</td><td>0.803</td><td></td><td>0.305</td><td>3.71</td><td>3.78</td><td>3.86</td><td>4.21</td><td>4.42</td></tr><tr><td>Flux.2-Klein-9B</td><td>0.163</td><td>0.284</td><td>0.100</td><td>1.51</td><td>2.55</td><td>0.098</td><td>4.02</td><td>4.57</td><td>0.375</td><td>0.573</td><td>0.062</td><td>2.16</td><td></td><td>3.02</td><td>3.12 3.68</td><td></td><td>4.24</td></tr><tr><td>Qwen-Image-Edit</td><td>0.378</td><td>0.563</td><td>0.647</td><td>3.31</td><td>3.31</td><td>0.323</td><td>3.08</td><td>2.76</td><td>0.602</td><td>0.769</td><td>0.280</td><td>3.12</td><td>3.36</td><td></td><td>3.58 4.06</td><td></td><td>2.92</td></tr><tr><td>PosterText</td><td>0.763</td><td>0.899</td><td>0.039</td><td>4.51</td><td>4.61</td><td>0.040</td><td>4.70</td><td>4.78</td><td>0.771</td><td>0.902</td><td>0.028</td><td>4.53</td><td>4.64</td><td>4.21</td><td>4.46</td><td></td><td>4.88</td></tr></table>

Table 1: Quantitative results on PosterText-Bench. Bold and underline indicate the best and second-best results, respectively. S.A.: Sentence Accuracy; M-L.: Masked-LPIPS; I.F.: Instruction Following; V.Q.: Visual Quality; B.P.: Background Preservation. Higher ↑ is better for S.A., NED, I.F., V.Q., and B.P., while lower ↓ is better for M-L.

<table><tr><td>Model</td><td>S.A.↑</td><td>NED↑</td><td>M-L.↓</td><td>I.F.↑</td><td>V.Q.↑</td></tr><tr><td>Base Model</td><td>0.490</td><td>0.666</td><td>0.417</td><td>3.27</td><td>3.37</td></tr><tr><td>+SFT Stage</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Stage II-only</td><td>0.679</td><td>0.786</td><td>0.088</td><td>3.63</td><td>3.81</td></tr><tr><td>Stage I(no cur.)+II</td><td>0.702</td><td>0.831</td><td>0.061</td><td>4.00</td><td>4.12</td></tr><tr><td>Stage I + II</td><td>0.712</td><td>0.866</td><td>0.053</td><td>4.03</td><td>4.24</td></tr><tr><td>+SFT+RL</td><td>0.735</td><td>0.883</td><td>0.037</td><td>4.22</td><td>4.51</td></tr><tr><td>+SFT+RL+SGSD</td><td>0.767</td><td>0.901</td><td>0.036</td><td>4.49</td><td>4.62</td></tr></table>

Table 2: Ablation study of diferent training stages, averaged across all tasks.

PosterText provides a more controllable and deploymentfriendly solution for practical poster generation scenarios.

The comparison results demonstrate that PosterText addresses the key challenge of this work: unifying poster generation and fine-grained patch-level editing within a single model. Across three editing tasks, PosterText achieves strong performance in text fidelity and region preservation, two essential requirements often underemphasized by existing generation-oriented methods. By combining accurate text rendering with precise local control under a unified instruction framework, PosterText bridges the gap between text rendering models and general-purpose image editing systems. For poster generation, the performance gap with closedsource models mainly reflects diferences in training scale and optimization focus, while the strong product fidelity of PosterText confirms that editing-oriented region control can efectively transfer to generation scenarios.

## Ablation Study

To evaluate the contribution of each training stage, we conduct a systematic ablation study using task-averaged metrics, with results summarized in Table 2. Specifically, Sen.ACC and NED are averaged over the Addition and Modification tasks; Masked-LPIPS is averaged over Addition, Deletion, and Modification; instruction following and visual quality scores are averaged across all four tasks.

Analysis of the SFT stage. Without task-specific training, the base model achieves only 0.490 S.A. and 0.417 M-L., indicating limited capability for patch-level editing. Stage IIonly substantially improves performance, increasing S.A. to 0.679 and reducing M-L. to 0.088, demonstrating that instruction-following training establishes basic editing capabilities. Adding Stage I pretraining (Stage I(no cur.)+II) further improves NED from 0.786 to 0.831, confirming that text rendering pretraining strengthens character-level representation. Introducing the curriculum sampling strategy (Stage I+II) brings additional but modest improvements, mainly by providing a more stable initialization for subsequent RL optimization.

Analysis of the RL stage. Building on full SFT, RL further improves editing performance and visual quality. This demonstrates that pixel-level reconstruction objectives alone cannot fully capture human preferences, such as text correctness, aesthetic quality, and instruction compliance. The multi-dimensional reward design efectively guides diferent aspects of optimization: OCR-based rewards improve text fidelity, aesthetic rewards enhance visual quality, and instruction-based rewards improve task execution. The significant reduction in M-L. (30.2%) further indicates that the background preservation reward explicitly constrains editing boundaries, reducing unintended modifications.

Analysis of the SGSD stage. SGSD further enhances text accuracy and instruction following through dense trajectorylevel supervision, which corrects intermediate-step errors beyond the reach of sparse reward signals. The improvement in instruction following indicates that spatial knowledge from the mask-guided teacher can be efectively transferred to mask-free inference for more accurate region localization. Since RL already provides strong background preservation, SGSD mainly improves text rendering precision and execution reliability. Together, RL and SGSD ofer complementary benefits: RL optimizes overall editing quality, while SGSD improves execution accuracy.

## Conclusion

In this work, we have introduced the task of Text Patch Generation and Editing, which treats text patches as fundamental units for automated e-commerce poster design and unifies poster generation, patch addition, deletion, and modification within a single framework. Based on this formulation, we propose PosterText, a unified model trained with a fourstage curriculum that sequentially develops text rendering, instruction following, preference alignment, and execution refinement capabilities. Extensive experiments demonstrate that PosterText achieves competitive performance in text fidelity, background preservation, and instruction execution, validating efectiveness of unified generation and editing.

## References

Baron, Y.; Dorfman, S.; Paiss, R.; Cohen-Or, D.; and Patashnik, O. 2026. Analysis-by-Proxy: Localization Signals in VLMs Operating as Condition Encoders. arXiv preprint arXiv:2607.06445.

Chen, H.; Xu, X.; Li, W.; Ren, J.; Ye, T.; Liu, S.; Chen, Y.- C.; Zhu, L.; and Wang, X. 2025a. Posta: A go-to framework for customized artistic poster generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, 28694–28704.

Chen, S.; Lai, J.; Gao, J.; Shi, H.; Liu, Z.; Ye, T.; Luo, J.; Wei, X.; and Zhu, L. 2026a. Posteromni: Generalized artistic poster creation via task distillation and unified reward feedback. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 5978–5987.

Chen, S.; Lai, J.; Gao, J.; Ye, T.; Chen, H.; Shi, H.; Shao, S.; Lin, Y.; Fei, S.; Xing, Z.; et al. 2025b. Postercraft: Rethinking high-quality aesthetic poster generation in a unified framework. arXiv preprint arXiv:2506.10741.

Chen, Z.; Zhao, F.; Shu, Y.; Liu, Y.; Yu, L.; and Zhou, Y. 2026b. StyleTextGen: Style-Conditioned Multilingual Scene Text Generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 7643– 7653.

DeepMind, G. 2025a. Gemini image: High-quality image generation. Accessed 2026-06-15.

DeepMind, G. 2025b. Gemini image pro: High-quality image generation. Accessed 2026-06-15.

Fan, J.; Qin, Y.; Feng, W.; Chen, Y.; Li, Y.; Ma, A.; Li, Y.; Zhuang, L.; Bian, H.; Zhang, Z.; et al. 2026. Autopp: Towards automated product poster generation and optimization. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, 3768–3776.

Fu, X.; Ma, L.; Guo, Z.; Dong, S.; Yang, L.; Sin, T. L.; Zhou, G.; He, Y.; Fu, J.; Zhou, S.; et al. 2025. Dynamictreerpo: Breaking the independent trajectory bottleneck with structured sampling. arXiv preprint arXiv:2509.23352.

Gao, Y.; Lin, Z.; Liu, C.; Zhou, M.; Ge, T.; Zheng, B.; and Xie, H. 2025. Postermaker: Towards high-quality product poster generation with accurate text rendering. In Proceedings of the Computer Vision and Pattern Recognition Conference, 8083–8093.

Google. 2025. Gemini 3: Introducing the latest gemini ai model from googles. Accessed 2026-05-18.

Guo, Z.; Liu, X.; Ma, L.; Wang, C.; He, Y.; Fu, X.; Fu, J.; Shan, X.; Guo, S.; Liu, L.; et al. 2026. GMO-E<sup>2</sup>DIT:

Grounded Multi-Operation Editing for E-Commerce Images. arXiv preprint arXiv:2607.00920.

Guo, Z.; Ma, L.; Fu, X.; Zhou, G.; Yang, L.; Zhou, Y.; Liu, L.; He, Y.; Liu, X.; Dong, S.; et al. 2025. Repainter: Empowering e-commerce object removal via spatial-matting reinforcement learning. arXiv preprint arXiv:2510.07721.

Hsu, H. Y.; He, X.; Peng, Y.; Kong, H.; and Zhang, Q. 2023. Posterlayout: A new benchmark and approach for contentaware visual-textual presentation layout. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 6018–6026.

Labs, B. F. 2025. FLUX.2 [klein]: Towards Interactive Visual Intelligence.

Lan, R.; Bai, Y.; Duan, X.; Li, M.; Jin, D.; Xu, R.; Nie, D.; Sun, L.; and Chu, X. 2025. Flux-text: A simple and advanced difusion transformer baseline for scene text editing. arXiv preprint arXiv:2505.03329.

Li, Z.; Li, F.; Feng, W.; Zhu, H.; Li, Y.; Zhang, Z.; Lv, J.; Shen, J.; Lin, Z.; Shao, J.; et al. 2023. Planning and rendering: Towards product poster generation with difusion models. arXiv preprint arXiv:2312.08822.

Lin, J.; Zhou, M.; Ma, Y.; Gao, Y.; Fei, C.; Chen, Y.; Yu, Z.; and Ge, T. 2023. Autoposter: A highly automatic and contentaware design system for advertising poster generation. In Proceedings of the 31st ACM International Conference on Multimedia, 1250–1260.

Liu, J.; Zhang, P.; Zhang, Y.; Yan, P.; Zhou, H.; Zhou, X.; Guo, F.; and Jin, L. 2026. PosterVerse: A Full-Workflow Framework for Commercial-Grade Poster Generation with HTML-Based Scalable Typography. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, 7197–7205.

Lu, R.; Zhang, Y.; Liu, J.; Wang, H.; and Song, Y. 2026. Easytext: Controllable difusion transformer for multilingual text rendering. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, 7565–7573.

Ma, J.; Deng, Y.; Chen, C.; Du, N.; Lu, H.; and Yang, Z. 2025a. Glyphdraw2: Automatic generation of complex glyph posters with difusion models and large language models. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, 5955–5963.

Ma, J.; Zhao, M.; Chen, C.; Wang, R.; Niu, D.; Lu, H.; and Lin, X. 2023. Glyphdraw: Seamlessly rendering text with intricate spatial structures in text-to-image generation. arXiv preprint arXiv:2303.17870.

Ma, L.; Fu, X.; Zhou, G.; Guo, Z.; Zhu, T.; Liu, Y.; Shi, Y.; Li, J.; and Huang, J. 2026. UM-Text: A Unified Multimodal Model for Image Understanding. arXiv preprint arXiv:2601.08321.

Ma, L.; Yue, T.; Fu, P.; Zhong, Y.; Zhou, K.; Wei, X.; and Hu, J. 2024. Chargen: High accurate character-level visual text generation model with multimodal encoder. arXiv preprint arXiv:2412.17225.

Ma, Y.; Bai, Q.; Ouyang, H.; Cheng, K. L.; Wang, Q.; Liu, H.; Liu, Z.; Wang, H.; Chen, J.; Shen, Y.; et al. 2025b. Calligrapher: Freestyle Text Image Customization. arXiv preprint arXiv:2506.24123.

Qin, Y.; Cao, K.; Liu, H.; Ma, A.; Li, F.; Zhu, H.; Zhang, Z.; Ling, R.; Feng, W.; He, X.; et al. 2026. Innoads-composer: Eficient condition composition for e-commerce poster generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 32988–32999.

Seedream Team. 2025. Seedream 4.0: Toward Nextgeneration Multimodal Image Generation. arXiv preprint arXiv:2509.20427.

Shi, W.; Song, Y.; Zhang, D.; Liu, J.; and Zou, X. 2025. Fonts: Text rendering with typography and style controls. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 18463–18474.

Tan, L. S.; Chen, J.; Fu, X.; Ma, L.; Huang, J.; Shi, J.; Li, Y.; and Wen, L. 2026. Meta-TTRL: A Metacognitive Framework for Self-Improving Test-Time Reinforcement Learning in Unified Multimodal Models. arXiv preprint arXiv:2603.15724.

Team, Q. 2026. Qwen3.5-omni technical report. arXiv preprint arXiv:2604.15804.

Tuo, Y.; Geng, Y.; and Bo, L. 2024. Anytext2: Visual text generation and editing with customizable attributes. arXiv preprint arXiv:2411.15245.

Tuo, Y.; Xiang, W.; He, J.-Y.; Geng, Y.; and Xie, X. 2024. Anytext: Multilingual visual text generation and editing. In International Conference on Learning Representations, volume 2024, 56783–56799.

Wu, C.; Li, J.; Zhou, J.; Lin, J.; Gao, K.; Yan, K.; Yin, S.-m.; Bai, S.; Xu, X.; Chen, Y.; et al. 2025. Qwen-Image Technical Report. arXiv preprint arXiv:2508.02324.

Zheng, K.; Chen, H.; Ye, H.; Wang, H.; Zhang, Q.; Jiang, K.; Su, H.; Ermon, S.; Zhu, J.; and Liu, M.-Y. 2025. Difusionnft: Online difusion reinforcement with forward process. arXiv preprint arXiv:2509.16117.

Zhou, J.; Li, J.; Xu, Z.; Li, H.; Cheng, Y.; Hong, F.-T.; Lin, Q.; Lu, Q.; and Liang, X. 2025. FireEdit: Fine-grained Instruction-based Image Editing via Region-aware Vision Language Model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.