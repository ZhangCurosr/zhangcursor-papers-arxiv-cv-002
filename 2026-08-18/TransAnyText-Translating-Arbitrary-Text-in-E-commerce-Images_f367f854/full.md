# TransAnyText: Translating Arbitrary Text in E-commerce Images via Structured Visual Generation

Xiaoan Liu<sup>1,2,∗</sup>, Lichen Ma<sup>2,3,∗</sup>, Zipeng Guo<sup>2</sup>, Yu He<sup>2</sup>, Xiaoyan Su<sup>4</sup>, Shaojie Guo<sup>2</sup>, Hao Yang<sup>3</sup>, Jingling Fu<sup>2</sup>, Xiaolong Fu<sup>2</sup>, Zhen Chen<sup>2</sup>, Yu Guo<sup>3</sup>, Fei Wang<sup>3</sup>, Xinyi Liu<sup>1</sup>, Yongjun Zhang<sup>1,‡</sup>, Ke Zhang<sup>2</sup>, Junshi Huang<sup>2,†</sup>

<sup>1</sup>Wuhan University, <sup>2</sup>JD.com, <sup>3</sup>State Key Laboratory of Human-Machine Hybrid Augmented Intelligence, Institute of Artificial Intelligence and Robotics, Xi’an Jiaotong University, <sup>4</sup>The Hong Kong University of Science and Technology (Guangzhou)

## Abstract

Cross-border e-commerce image translation is essential for global retail, where product images, banners, and detail pages need to be produced in diferent languages. Existing methods struggle to achieve accurate translation, faithful visual identity preservation, and easy-to-edit outputs, simultaneously. To address these challenges, we introduce TransAnyText, a structured visual code framework that reformulates image text translation as generating renderable HTML patches from source images and target languages. Our framework decouples semantic generation from pixel rendering: a vision-language model (VLM) handles visual understanding, cross-lingual translation, and structured visual generation, while a difusion model performs background inpainting and pixel-level refinement, followed by deterministic rendering to synthesize the final image. Based on this formulation, we develop a threestage post-training framework, where supervised fine-tuning (SFT) establishes the image-to-code mapping, privilege-gap weighted self-distillation (PWSD) improves the learning of style and layout tokens, and reinforcement learning with verifiable rewards (RLVR) further optimizes task-level performance. We further introduce TransAnyDataset and TransAny Bench, a multilingual dataset and benchmark for e-commerce image translation. Extensive experiments demonstrate competitive performance against cascaded pipelines, open-source end-to-end models, and closed-source image editing systems, providing an efective, controllable, and editable solution for cross-border e-commerce image translation.

## Introduction

Cross-border e-commerce has become a key growth driver for global retail platforms, where product images, banners, and detail pages are essential for product presentation and customer conversion (Gao et al. 2025; Fan et al. 2026; Qin et al. 2026; Guo et al. 2026). These visual assets require frequent localization across markets and languages. However, large-scale image localization still relies on substantial human intervention, leading to high costs and low scalability. More importantly, this task goes beyond simply translating text into diferent languages. A successful solution must satisfy three key requirements: 1) adapting layouts to accommodate language-dependent text variations; 2) preserving visual identity elements such as brand fonts, badges, and decorative efects; and 3) maintaining editability for downstream operations. Developing an open, controllable, and editable solution for multilingual e-commerce image translation remains a critical challenge.

![](images/c569745e2f8ea1a01480e1bd2c157cf48de99b29d27d5217fa6cdd5a1ab3e401.jpg)  
Figure 1: Comparison of paradigms: (a) the cascaded pipeline of detection, translation, erasing, and re-rendering, which sufers from error accumulation and visual artifacts; (b) the end-to-end pixel edit paradigm, which jointly performs translation and rendering but often hallucinates or distorts text; and (c) our structured visual code paradigm, which decouples semantic generation from pixel rendering to yield accurate, editable, and visually faithful results.

Existing in-image translation methods mainly follow two paradigms. The cascaded route in Figure 1 (a) decomposes the task into text extraction, translation, and rendering, enabling each stage to leverage specialized models (Li et al. 2024; Tuo et al. 2024; Zeng et al. 2024). However, errors accumulate along the pipeline, erasing and re-rendering text often introduces artifacts on complex backgrounds (Shu et al. 2025). The end-to-end pixel route in Figure 1 (b) directly generates target-language images (Lan et al. 2024; Wu et al. 2025; Lyu et al. 2026b), but requires the model tojointly handle visual understanding, cross-lingual translation, and text rendering. This leads to hallucinated, missing, or incorrect text, while scaling to multilingual scenarios requires substantial data and training resources. Although closed-source image editing models achieve strong visual quality, their pixelbased outputs remain dificult to control and edit, due to high API costs, data compliance concerns, and limited finetuning flexibility. These limitations highlight a fundamental

![](images/03f5ada8f1f69ac658ebd186eced04740b47703d5b8e03ccea86ce730e40ba60.jpg)

## (a) Visual Comparison across Methods

Figure 2: Visual comparison on e-commerce image text translation. (a) Qualitative comparison with existing methods; (b) qualitative comparison with closed-source models on multilingual outputs from a single Chinese source image.

gap: pixel-based representations struggle to explicitly encode the structural and visual constraints required for e-commerce image translation, such as adaptive layout, text fidelity, style consistency, and easy editability.

To tackle these limitations, we introduce structured visual code to decouple semantic generation from pixel rendering, as illustrated in Figure 1 (c). In this framework, a visionlanguage model (VLM) handles visual understanding, crosslingual translation, and structured visual generation, while a difusion model specializes in background inpainting and pixel-level refinement. The final image is deterministically synthesized by a renderer from the structured representation, enabling faithful and controllable translation. Specifically, we reformulate e-commerce image translation as generating a renderable HTML H from a source image I and a target language L . This formulation ensures deterministic text rendering while providing explicit control over visual attributes and layout structure. Based on this formulation, we develop a three-stage post-training framework: SFT learns the imageto-code mapping and trains the difusion model for background inpainting; PWSD provides dense supervision for under-optimized style and layout tokens; RLVR further aligns task-level performance through verifiable rewards covering spatial accuracy, translation quality, and style fidelity.

Our contributions are as follows. (1) Task reformulation: We reformulate e-commerce image translation using visual code as a structured, renderable intermediate representation, achieving competitive results compared with existing open-source and closed-source approaches. (2) Posttraining framework: We propose TransAnyText, an opensource three-stage training framework that improves visualtoken learning and aligns task-level performance with verifiable rewards. (3) Multilingual benchmark: We introduce TransAnyDataset and TransAnyBench, a multilingual dataset and benchmark with comprehensive evaluation of translation quality, visual fidelity, and image realism.

## Related Work

## Image Text Translation

In-image machine translation (IMT) translates text within images and renders it back while preserving original layouts and visual styles. Existing approaches are divided into cascaded systems and end-to-end models. Cascaded methods (Qian et al. 2024; Lu et al. 2026; Lyu et al. 2026a) split the task into sequential stages: detection, recognition, translation, and inpainting. They extract text via vision models, translate $\mathbf { i t } ,$ and paste it back using text-editing generative models (Ma et al. 2024, 2026; Tuo et al. 2024; Lan et al. 2025; Guo et al. 2025). However, these systems sufer from a fundamental bottleneck: isolated modules introduce error accumulation. Since the "translation + paste-back" paradigm does not naturally integrate visual understanding and language generation, its inherent limitations cannot be addressed by improving individual components, limiting their scalability.

End-to-End Image-Informed Machine Translation (IIMT) To mitigate the error propagation and pipeline complexity of cascaded systems, recent research has shifted toward endto-end Image-Informed Machine Translation (IIMT). Early works like Translatotron-V (Lan et al. 2024) decoupled translation from visual rendering using an intermediate text decoder for discrete token prediction, reducing pixel-level optimization complexity. To bridge the gap toward real-world applications, PRIM (Tian et al. 2025) explored practical multilingual scenarios via VisTrans, which separately processes textual and background features to enhance stability and image integrity. Although theoretically bypassing cascading errors through unified generation, it sufers from fatal real-world flaws, including frequent text hallucinations, poor generalization, prohibitive computational costs, and uneditable pixel outputs.

## Structured Visual Generation

With the rapid progress of multimodal large language models (MLLMs), visual code generation has emerged as an efective paradigm that bridges visual perception and executable programs (Zhao et al. 2026b; Ye et al. 2026; Rodriguez et al. 2025; Liu et al. 2026b). By representing visual content as editable, resolution-independent, and structurally explicit vector programs, these approaches provide superior scalability and editability compared with raster-based representations. Chat2SVG (Wu, Su, and Liao 2025) leveraged LLMs to generate hierarchical SVG structures and refined details with a difusion-based prior. OmniSVG (Yang et al. 2026) tokenized SVG commands and coordinates for autoregressive SVG generation with pretrained vision-language models. SVGBuilder (Chen and Pan 2025) built a reusable pathcomponent library and employed CLIP with an autoregressive decoder for eficient SVG synthesis. PosterVerse (Liu et al. 2026a) combined LLM-based design parsing, difusionbased background generation, and VLM-driven HTML generation for visual layout synthesis. Despite using visual code as an intermediate representation, existing methods mainly address unconstrained generation from text descriptions or design specifications. In contrast, image text translation is a constrained visual code generation problem, where the model must preserve the input image’s layout, typography, and visual identity while performing cross-lingual translation and adaptive text reflow. We formulate this task as imageto-HTML visual text translation and develop a systematic framework based on open-source VLMs.

## Method

Overview. We formulate e-commerce image text translation as structured visual code generation: given a source image I and a target language $L _ { t } ,$ the system generates a renderable HTML patch H that preserves both translation accuracy and visual identity. As shown in Figure 3, the pipeline has four steps: (1) Background Inpainting: text regions in I are erased to obtain $I _ { \mathrm { b g } } .$ (2) Structured Generation: a VLM takes $( I , L _ { t } )$ as input and generates $H ,$ , encoding translated text with position, font, color, and size. (3) Deterministic Rendering: H is rendered onto $I _ { \mathrm { b g } }$ using Playwright to produce the translated image. (4) Optional Difusion Refinement: a difusion model optionally refines the result to enhance visual quality. The VLM and difusion model are initialized with SFT for image-to-HTML mapping and background inpainting, while PWSD and GRPO further optimize the VLM with token-level supervision and task-level rewards.

## Stage 1: Supervised Fine-Tuning

SFT jointly optimizes the two modules, including the VLMbased structured generator and the background inpainting diffusion model, to establish the fundamental image-to-HTML generation capability.

VLM. Given paired samples $( I , H ^ { * } )$ where $H ^ { * } =$ $\left[ y _ { 1 } , y _ { 2 } , . . . , y _ { n } \right]$ denotes the ground-truth HTML patch represented as a sequence of tokens, we fine-tune the VLM using the standard autoregressive next-token prediction objective:

$$
\mathcal { L } _ { \mathrm { S F T } } = - \sum _ { t } \log \pi _ { \theta } ( y _ { t } \mid y _ { < t } , I , L _ { t } )\tag{1}
$$

Inpainting Difusion Model. We train a conditional flow matching model to learn the background inpainting process from the original image I to the clean background $I _ { \mathrm { b g } } .$ . Specifically, a linear probability path $x _ { t } = ( 1 - t ) I _ { \mathrm { b g } } + t I$ is constructed, and the model is optimized to regress the target velocity field $u _ { t } = I - I _ { \mathrm { b g } } \mathrm { : }$

$$
\mathcal { L } _ { \mathrm { F M } } = \mathbb { E } _ { t , ( I , I _ { \mathrm { b g } } ) } \left. v _ { \phi } \bigl ( x _ { t } , t \mid I \bigr ) - u _ { t } \right. _ { 2 } ^ { 2 }\tag{2}
$$

The learned inpainting model provides a clean background canvas for subsequent rendering.

## Stage 2: Privilege-Gap Weighted Self Distillation

Motivation. After SFT, the VLM can generate valid HTML but struggles with accurate style and layout reproduction. Style-related tokens sufer from weak structural dependency and teacher-forcing bias, leading to fragile optimization. PWSD provides dense, adaptively weighted token-level supervision before RLVR for robust initialization.

Mechanism. We employ the same SFT model under two different conditioning settings: a privileged teacher $\pi _ { T }$ receives both the image I and extracted source-language HTML $H _ { \mathrm { s r c } } ,$ while the student $\pi _ { S }$ only accesses the image I. Given an on-policy rollout $y \sim \pi _ { S } ( \cdot \mid I )$ , we measure the token-level privilege gap:

![](images/968e95c5117e2303de40991e9b8285a3325c01250969ad21ca75a04de85ad2be.jpg)  
Figure 3: Overview of the TransAnyText training framework and inference pipeline. The training consists of three stages: (1) joint SFT for structured HTML code generation and background inpainting, (2) PWSD for token-level self-distillation on style and layout tokens, and (3) RLVR for task-level alignment with verifiable rewards. At inference, the VLM and difusion model jointly produce an editable HTML and a clean background, which are deterministically rendered into the translated image.

$$
\mathrm { g a p } _ { t } = \log \pi _ { T } ( y _ { t } \mid y _ { < t } , I , H _ { \mathrm { s r c } } ) - \log \pi _ { S } ( y _ { t } \mid y _ { < t } , I )\tag{3}
$$

A stop-gradient sigmoid transformation converts the gap into an adaptive supervision weight:

$$
w _ { t } = \operatorname { s g } \left[ \sigma ( \alpha \cdot \mathbf { g } \mathbf { a p } _ { t } ) \right]\tag{4}
$$

The weighted reverse KL distillation objective is then applied at the token level:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { P W S D } } = \mathbb { E } _ { y \sim \pi _ { S } } \Big [ \sum _ { t } w _ { t } \mathrm { K L } ( \pi _ { S } ^ { t } | | \pi _ { T } ^ { t } ) \Big ] } \end{array}\tag{5}
$$

This adaptive weighting mechanism emphasizes tokens with larger privilege gaps, typically corresponding to style and layout attributes, while suppressing redundant supervision on tokens where the student already agrees with the teacher.

## Stage 3: Group Relative Policy Optimization

While PWSD provides efective token-level style guidance, further task-level optimization is required to improve overall generation performance. Therefore, we further introduce GRPO (Shao et al. 2024) with patch-level verifiable rewards to directly optimize task-level objectives. For each input, we sample G rollouts $\{ y ^ { ( g ) } \} _ { g = } ^ { G }$ from the current policy, assign each rollout a composite reward $\begin{array} { r } { r ^ { ( g ) } = \sum _ { k } \lambda _ { k } r _ { k } ^ { ( g ) } } \end{array}$ , and compute group-relative advantages:

$$
A ^ { ( g ) } = \frac { r ^ { ( g ) } - \bar { r } } { \mathrm { s t d } ( r ) }\tag{6}
$$

The GRPO objective maximizes advantage-weighted logprobability while constraining the policy deviation from the PWSD checkpoint $\pi _ { \mathrm { r e f } } .$

$$
\mathcal { I } _ { \mathrm { G R P O } } = \mathbb { E } \left[ \sum _ { g } A ^ { ( g ) } \log \pi _ { \theta } ( y ^ { ( g ) } ) - \beta \mathbf { K } \mathbf { L } ( \pi _ { \theta } \lVert \pi _ { \mathrm { r e f } } ) \right]\tag{7}
$$

The overall reward is composed of four automatically verifiable components:

$r _ { \mathrm { f o r m a t } } .$ : verifies whether the generated HTML is syntactically valid and renderable.

• r<sub>position</sub>: measures patch-level spatial alignment between the generated layout and the source image.

• r<sub>translation</sub>: evaluates semantic correctness, linguistic fluency, and domain-specific terminology consistency.

$r _ { \mathrm { s t y l e } } )$ : measures the consistency of visual attributes, including color, font size, and font weight.

## Optional Difusion Refinement

Code-driven rendering is inherently limited in capturing pixel-level visual details. To address this issue, we introduce an open-source difusion model as an inference-time refinement module. Conditioned on $R ( H , I _ { \mathrm { b g } } )$ , it performs low-strength editing to reduce artifacts and enhance visual fidelity while preserving text content and layout. This design enables each module to focus on its strengths: the VLM handles semantic and structural generation, while the difusion model specializes in pixel-level visual refinement.

![](images/a98ddea0fc7486406dfb4ae93dbbb2d44aa97d54611a4e583f45f61806adcdaf.jpg)

![](images/7161a85f97efd95ad5ebebc228eaf0148ca757a5764bd3eb19b8a9cae1c7bd1f.jpg)

![](images/a9c869af892e951af1082fbdf83dcfd9681b75767942da2f33750c2ab813020b.jpg)

![](images/49a538f235050477b41809af73bd34f19722dae0423db9ccc78a09cbc215e680.jpg)  
Figure 4: Dataset statistics of TransAnyDataset, including distributions over categories, languages, text regions, and word counts.

## Dataset and Benchmark

We introduce TransAnyDataset and TransAnyBench, the first multilingual dataset and benchmark for e-commerce image text translation. It supports 10 languages across five writing systems and establishes a unified evaluation protocol for translation accuracy, visual fidelity, and image realism.

Data Source and Curation. Source images are collected from e-commerce platforms, including product photos, banners, and promotional posters across categories such as furniture, household goods, kitchenware, fashion, beauty, and electronics. We construct multilingual samples through a four-stage pipeline: (1) Extraction: Gemini-3.1 Pro (Google 2025) extracts structured HTML patches containing text, bounding boxes, and visual attributes.(2) Filtering : Qwen3.5-27B (Team 2026) filters out samples with incorrect text recognition or inaccurate layout alignment.(3) Translation : Gemini-3.1 Pro translates the samples into 8 additional languages using Chinese and English as hub languages, with e-commerce-aware constraints to preserve terminology and layout compatibility.(4) Quality Assurance : Qwen3-VL-32B (Bai et al. 2025) evaluates each sample, and low-quality instances are discarded after human review.

Language Coverage and Density. TransAnyDataset covers 10 languages, including English, Chinese, Japanese, Spanish, French, German, Portuguese, Korean, Russian, and Italian, spanning 5 writing systems. As shown in Figure 4, this diversity introduces challenges in cross-lingual image text translation, including script variation, text length changes, and typography adaptation. The dataset focuses on shorttext, multi-region layouts commonly found in e-commerce images, while also incorporating high-density promotional banners to capture diverse layout complexities.

Evaluation Protocol. Image text translation requires joint evaluation of linguistic quality and visual quality. We therefore design four complementary metrics covering translation accuracy, visual preservation, and image realism:

• COMET (Rei et al. 2020): A reference-based translation metric for evaluating the quality of extracted text.

• Translation Quality Evaluator: A VLM-based evaluator that assesses translation accuracy, naturalness, terminology consistency, and contextual appropriateness.

• Visual Fidelity Evaluator: A VLM-based evaluator that compares source and generated images to measure the preservation of visual identity, including fonts, colors, decorative elements, and product appearance.

• Image Realism Evaluator: A VLM-based evaluator that assesses standalone image quality, including readability, visual coherence, and rendering artifacts.

## Experiments

## Experimental Setup

Implementation Details. We adopt Qwen3.5-9B (Team 2026) as the backbone for structured HTML generation and perform LoRA-based fine-tuning using a learning rate of $\mathrm { \bar { 1 } \times 1 0 ^ { - 4 } }$ and a LoRA rank of 64. The difusion-based inpainting model is built on FLUX.2-klein-9B (Labs 2025b), with a learning rate of $1 \times 1 0 ^ { - 5 }$ and a LoRA rank of 32. The refinement module is disabled during evaluation, and can be optionally enabled during deployment with editing models (e.g., SD (Esser et al. 2024) or FLUX (Labs 2025a,c) series). More details are provided in the supplementary materials.

Compared Methods. We compare against four categories of methods: (1) Cascaded methods: OCR+MT+T2I and Qwen3-VL-8B (Bai et al. 2025)+T2I, which decompose the task into sequential text extraction, machine translation, and text rendering stages. (2) Open-source image editing methods: FireRed-Image-Edit (Zhou et al. 2025), LongCat-Image-Edit (Team et al. 2025), and Qwen-Image-Edit (Wu et al. 2025). (3) Closed-source image editing methods: Seedream 5.0 (Seedream Team 2025), Nano Banana 2 (Deep-Mind 2025), and GPT Image 2 (OpenAI 2026b). (4) Codedriven methods: approaches that remove text via inpainting, generate structured HTML with VLMs, and render the final image, including Qwen3.5-27B (Team 2026), GPT5.5 (OpenAI 2026a), and Gemini-3.1 Pro (Google 2025).

## Experimental Results

We evaluate all methods on TransAnyBench spanning 10 languages, with aggregated results reported in Table 1.

![](images/0ca619c31f14b1bfc49cbda1497a1a2151c4b3b026a9a0c9d48a8441f5368875.jpg)  
Figure 5: Visual comparison of TransAnyText with existing methods.

TransAnyText achieves the best translation quality among open-source methods and outperforms closed-source systems in most language directions. Its explicit modeling of font, color, and positional attributes brings clear advantages in visual fidelity over open-source pixel-based methods, while remaining competitive with closed-source models. For image realism, TransAnyText achieves comparable performance to closed-source systems. As shown in Figure 2 and Figure 5, TransAnyText consistently preserves visual identity and translation accuracy across diverse language pairs. Open-source image editing models often sufer from text rendering errors, while closed-source systems achieve stronger visual quality but may still exhibit style drift and limited editability. By decoupling semantic translation from pixel rendering, TransAnyText achieves superior visual consistency and translation performance.

Analysis of cascaded methods. Benefiting from dedicated translation modules, cascaded pipelines achieve moderate COMET scores, reflecting reasonable level of cross-lingual performance. However, the erase-and-paste strategy often introduces artifacts on complex backgrounds, leading to significant degradation in visual fidelity and overall realism. In addition, text-to-image components often struggle with multi-region rewriting, which becomes a key bottleneck for further improvement. These observations highlight an inherent limitation of the cascaded paradigm, where optimizing individual components cannot fundamentally prevent error accumulation and propagation across stages.

Analysis of image editing models. FireRed-Image-Edit,

LongCat-Image-Edit, and Qwen-Image-Edit achieve T.Q. scores below 2.9 across translation directions, indicating limited cross-lingual capability. Designed for general-purpose image manipulation, these models struggle to jointly handle visual understanding, translation, and text rendering, resulting in poor character-level accuracy. Despite this, LongCat-Image-Edit and Qwen-Image-Edit maintain moderate visual fidelity (V.F. > 6.0), suggesting that their strengths lie in background synthesis rather than precise text generation. More broadly, the pixel-level end-to-end paradigm entangles semantic translation and visual rendering within a single network, causing objective interference. Closed-source models demonstrate strong performance in translation quality and realism; however, without explicit disentanglement of structure and appearance, their preservation of visual identity remains inconsistent. In addition, their rasterized outputs lack editability, limiting downstream operations such as font substitution or layout adjustment.

Analysis of code-driven methods. Code-driven methods adopt a paradigm similar to ours, where a VLM generates structured representations followed by background inpainting and deterministic rendering. With Gemini-3.1 Pro as the VLM backbone, this approach achieves competitive visual fidelity comparable to GPT Image 2 while maintaining strong translation quality. These results further validate the efectiveness of decoupling semantic reasoning from pixel synthesis: the VLM focuses on cross-lingual understanding and layout structuring, while text accuracy is ensured by deterministic rendering.

<table><tr><td></td><td colspan="4">Any→Zh</td><td colspan="4">Zh→Any</td><td colspan="4">Any→En</td><td colspan="4">En→Any</td></tr><tr><td>Method</td><td>COMET</td><td>T.Q.</td><td>V.F.</td><td>I.R.</td><td>COMET</td><td>T.Q.</td><td>V.F.</td><td>I.R.</td><td>COMET</td><td>T.Q.</td><td>V.F.</td><td>I.R.</td><td>COMET</td><td>T.Q.</td><td>V.F.</td><td>I.R.</td></tr><tr><td colspan="9">Closed-source Image Editing Methods</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Seedream 5.0</td><td>0.794</td><td>9.60</td><td>9.52</td><td>9.03</td><td>0.440</td><td>3.73 8.11</td><td>5.57</td><td>0.528</td><td>5.41</td><td>8.41</td><td>6.02</td><td>0.549</td><td></td><td>5.96 8.59</td><td>6.87</td></tr><tr><td>Nano Banana 2</td><td>0.782</td><td>9.69 9.52</td><td>9.22</td><td>0.584</td><td>8.73</td><td>9.31</td><td>8.98</td><td>0.627</td><td>8.91</td><td>9.38</td><td>8.69</td><td>0.632</td><td>9.41</td><td>9.41</td><td>9.22</td></tr><tr><td>GPT Image 2</td><td>0.793</td><td>9.54 9.24</td><td>9.51</td><td>0.596</td><td>8.46</td><td>9.20</td><td>9.22</td><td>0.632</td><td>9.05</td><td>9.26</td><td>9.10</td><td>0.642</td><td>9.45</td><td>9.33</td><td>9.33</td></tr><tr><td colspan="10">Cascaded Methods</td><td colspan="3"></td><td colspan="3"></td></tr><tr><td>OCR+MT+T2I</td><td>0.613</td><td>6.72</td><td>4.58</td><td>4.05</td><td></td><td>3.74</td><td>4.13</td><td>0.482</td><td>6.71</td><td>6.28</td><td>4.10</td><td>0.441</td><td>3.87</td><td>5.25</td><td>4.17</td></tr><tr><td>Qwen3-VL-8B+T2I</td><td>0.527</td><td>5.82 5.23</td><td>4.91</td><td>0.457 0.415</td><td>4.05 3.93</td><td>4.01</td><td>4.57</td><td>0.441</td><td>6.27</td><td>6.83</td><td>4.82</td><td>0.415</td><td>3.52</td><td>5.82</td><td>4.92</td></tr><tr><td colspan="10">Open-source Image Editing Methods</td><td colspan="7"></td></tr><tr><td>FireRed-Image-Edit</td><td>0.265</td><td>1.35</td><td>4.18</td><td>0.242</td><td></td><td>3.18</td><td>2.29</td><td>0.265</td><td>1.21</td><td>3.39</td><td>2.38</td><td>0.243</td><td>1.26</td><td>3.37</td><td>2.40</td></tr><tr><td>LongCat-Image-Edit</td><td>0.427</td><td>2.16 7.63</td><td>2.26 7.38</td><td>0.501</td><td>1.31 2.84</td><td>6.64</td><td>7.59</td><td>0.450</td><td>2.39</td><td>6.79</td><td>7.07</td><td>0.382</td><td>1.76</td><td>7.35</td><td>6.66</td></tr><tr><td>Qwen-Image-Edit</td><td>0.355</td><td>1.78 6.80</td><td>4.78</td><td>0.391</td><td>1.89</td><td>6.35</td><td>4.66</td><td>0.371</td><td>1.60</td><td>6.29</td><td>4.47</td><td>0.314</td><td>1.59</td><td>6.67</td><td>4.38</td></tr><tr><td colspan="10">Code-Driven Methods</td><td colspan="7"></td></tr><tr><td>Qwen3.5-27B</td><td>0.606</td><td>7.02</td><td>4.47 7.41</td><td>0.484</td><td></td><td>6.67 4.17</td><td>7.15</td><td>0.514</td><td>7.14</td><td>4.36</td><td>7.59</td><td>0.496</td><td>6.42</td><td>4.47</td><td>7.47</td></tr><tr><td>GPT5.5</td><td>0.796</td><td>9.61 9.03</td><td>8.90</td><td>0.586</td><td>8.63</td><td>8.25</td><td>8.10</td><td>0.633</td><td>9.19</td><td>8.48</td><td>8.33</td><td>0.632</td><td>9.39</td><td>8.81</td><td>8.75</td></tr><tr><td>Gemini-3.1 Pro</td><td>0.807</td><td>9.44</td><td>9.43 9.39</td><td></td><td>0.603 8.87</td><td>9.23</td><td>9.04</td><td>0.638</td><td>9.06</td><td>9.17</td><td>9.08</td><td>0.635</td><td>9.04</td><td>9.41</td><td>9.16</td></tr><tr><td>Ours</td><td>0.848</td><td>9.70</td><td>9.63</td><td>9.48</td><td>0.635</td><td>8.94 9.41</td><td>9.18</td><td>0.672</td><td>9.25</td><td>9.46</td><td>9.14</td><td>0.700</td><td>9.47</td><td>9.65</td><td>9.33</td></tr></table>

Table 1: Quantitative results on TransAnyBench. T.Q. represents Translation Quality, V.F. represents Visual Fidelity, and I.R. represents Image Realism. "Any" represents the aggregate over all languages except the target language. Best results are in bold, and second-best are underlined.

## Ablation Study

To assess the contribution of each training stage, we perform a systematic ablation on the mean performance across all translation directions in TransAnyBench, with results reported in Table 2. The structured HTML representation enables direct computation of fine-grained metrics, allowing performance gains to be explicitly attributed to individual stages. We consider five evaluation dimensions: COMET for translation quality, and T.E. (Text Extraction), T.Q. (Translation Quality), P.A. (Position Accuracy, measured by IoU), and S.S. (Style Similarity) to capture complementary aspects of structured generation.

Analysis of the SFT Stage. Without task-specific supervision, the base model attains only 0.482 COMET and 0.218 P.A., reflecting a limited capability for image-to-HTML mapping. SFT yields the largest single-stage gain, improving COMET by +0.182 and P.A. by +0.385, thereby establishing the fundamental competence for structured generation.

Analysis of the PWSD Stage. Compared with vanilla onpolicy self-distillation (OPSD) (Zhao et al. 2026a), PWSD achieves more consistent improvements across all metrics. The gains are particularly pronounced in position and style, suggesting that privilege-gap weighting provides more effective supervision for tokens associated with visual attributes. These tokens exhibit weaker structural dependencies in autoregressive generation and therefore benefit more from adaptive reweighting.

Analysis of the GRPO Stage. Applying GRPO directly after SFT (+SFT+GRPO-only) achieves a COMET score of 0.691, while the full pipeline (+SFT+PWSD+GRPO) further improves performance to 0.704. The gains are particularly evident in P.A. and S.S., indicating that PWSD provides a stronger initialization for subsequent reward optimization. Although GRPO can improve translation and text extraction through task-level reward signals, the lack of dense tokenlevel supervision limits its ability to optimize fine-grained spatial and stylistic attributes. By providing targeted supervision for these under-optimized tokens, PWSD complements GRPO and leads to the best overall performance.

<table><tr><td>Model</td><td>COMET</td><td>T.E.</td><td>T.Q.</td><td>P.A.</td><td>S.S.</td></tr><tr><td>Base Model</td><td>0.482</td><td>0.445</td><td>5.08</td><td>0.218</td><td>0.255</td></tr><tr><td>+SFT</td><td>0.664</td><td>0.877</td><td>8.59</td><td>0.603</td><td>0.554</td></tr><tr><td>+SFT+OPSD +SFT+PWSD</td><td>0.672 0.686</td><td>0.881</td><td>8.62</td><td>0.625</td><td>0.576</td></tr><tr><td></td><td></td><td>0.914</td><td>8.74</td><td>0.646</td><td>0.584</td></tr><tr><td>+SFT+GRPO-only</td><td>0.691</td><td>0.928</td><td>8.94</td><td>0.667</td><td>0.601</td></tr><tr><td>+SFT+PWSD+GRPO</td><td>0.704</td><td>0.942</td><td>8.98</td><td>0.697</td><td>0.633</td></tr></table>

Table 2: Ablation study on each training stages.

## Conclusion

We formulate e-commerce image text translation as structured visual code generation, where the model outputs a renderable HTML patch instead of pixels. This design separates semantic reasoning from visual rendering: the VLM handles understanding, translation, and structure; the difusion model supports background inpainting; and a deterministic renderer ensures text accuracy. Based on this formulation, TransAny-Text introduces a three-stage post-training pipeline. SFT establishes structured generation, PWSD improves supervision on visually grounded tokens through adaptive weighting, and GRPO refines task performance via multi-dimensional rewards. Experiments on TransAnyBench demonstrate consistent gains over cascaded and end-to-end methods, while maintaining strong editability and controllability.

Limitation. While efective for regular layouts, code-based rendering via HTML/CSS has limited expressiveness for complex patterns such as curved or highly stylized text. An optional difusion refinement module enhances these details, partially alleviating this limitation in practice.

## References

Bai, S.; Cai, Y.; Chen, R.; Chen, K.; Chen, X.; Cheng, Z.; Deng, L.; Ding, W.; Gao, C.; Ge, C.; et al. 2025. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631.

Chen, Z.; and Pan, R. 2025. Svgbuilder: Component-based colored svg generation with text-guided autoregressive transformers. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, 2358–2366.

DeepMind, G. 2025. Gemini image: High-quality image generation. Accessed 2026-06-15.

Esser, P.; Kulal, S.; Blattmann, A.; Entezari, R.; Müller, J.; Saini, H.; Levi, Y.; Lorenz, D.; Sauer, A.; Boesel, F.; et al. 2024. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first international conference on machine learning.

Fan, J.; Qin, Y.; Feng, W.; Chen, Y.; Li, Y.; Ma, A.; Li, Y.; Zhuang, L.; Bian, H.; Zhang, Z.; et al. 2026. Autopp: Towards automated product poster generation and optimization. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, 3768–3776.

Gao, Y.; Lin, Z.; Liu, C.; Zhou, M.; Ge, T.; Zheng, B.; and Xie, H. 2025. Postermaker: Towards high-quality product poster generation with accurate text rendering. In Proceedings of the Computer Vision and Pattern Recognition Conference, 8083–8093.

Google. 2025. Gemini 3: Introducing the latest gemini ai model from googles. Accessed 2026-06-15.

Guo, Z.; Liu, X.; Ma, L.; Wang, C.; He, Y.; Fu, X.; Fu, J.; Shan, X.; Guo, S.; Liu, L.; et al. 2026. GMO-E<sup>2</sup>DIT: Grounded Multi-Operation Editing for E-Commerce Images. arXiv preprint arXiv:2607.00920.

Guo, Z.; Ma, L.; Fu, X.; Zhou, G.; Yang, L.; Zhou, Y.; Liu, L.; He, Y.; Liu, X.; Dong, S.; et al. 2025. Repainter: Empowering e-commerce object removal via spatial-matting reinforcement learning. arXiv preprint arXiv:2510.07721.

Labs, B. F. 2025a. FLUX. 1 Kontext: Flow Matching for In-Context Image Generation and Editing in Latent Space. arXiv preprint arXiv:2506.15742.

Labs, B. F. 2025b. FLUX.2 [klein]: Towards Interactive Visual Intelligence.

Labs, B. F. 2025c. FLUX.2: Next Generation Image Generation.

Lan, R.; Bai, Y.; Duan, X.; Li, M.; Jin, D.; Xu, R.; Nie, D.; Sun, L.; and Chu, X. 2025. Flux-text: A simple and advanced difusion transformer baseline for scene text editing. arXiv preprint arXiv:2505.03329.

Lan, Z.; Niu, L.; Meng, F.; Zhou, J.; Zhang, M.; and Su, J. 2024. Translatotron-V (ison): An end-to-end model for in-image machine translation. In Findings ofthe Association for Computational Linguistics: ACL 2024, 5472–5485.

Li, Z.; Shu, Y.; Zeng, W.; Yang, D.; and Zhou, Y. 2024. First creating backgrounds then rendering texts: A new paradigm for visual text blending. arXiv preprint arXiv:2410.10168.

Liu, J.; Zhang, P.; Zhang, Y.; Yan, P.; Zhou, H.; Zhou, X.;Guo, F.; and Jin, L. 2026a. PosterVerse: A Full-Workflow

Framework for Commercial-Grade Poster Generation with HTML-Based Scalable Typography. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, 7197–7205.

Liu, Z.; Sun, S.; Huang, D.; Shi, Y.; Zhang, M.; Li, J.; Yu, J.; and Bian, J. 2026b. DesignAsCode: Bridging Structural Editability and Visual Fidelity in Graphic Design Generation. arXiv preprint arXiv:2602.17690.

Lu, J.; Song, T.; Wu, Z.; Li, P.; Liang, X.; Yang, H.; Chen, K.; Xie, N.; Lu, Y.; Zhao, J.; et al. 2026. Global-Local Dual Perception for MLLMs in High-Resolution Text-Rich Image Translation. arXiv preprint arXiv:2602.21956.

Lyu, J.; Fu, P.; Li, Z.; Zeng, W.; Zhang, S.; Yang, J.; Ma, C.; Zhou, Y.; Luo, Z.; and Luan, J. 2026a. IMT-Bench: A Multi-Scenario Cross-Modal Collaborative Evaluation Benchmark for In-Image Machine Translation. arXiv preprint arXiv:2603.10495.

Lyu, J.; Fu, P.; Li, Z.; Zhang, S.; Yang, J.; Zhou, Y.; Ma, C.; Luo, Z.; and Luan, J. 2026b. UniTranslator: A Unified Multi-modal Framework for End-to-end In-Image Machine Translation. arXiv preprint arXiv:2606.24333.

Ma, L.; Fu, X.; Zhou, G.; Guo, Z.; Zhu, T.; Liu, Y.; Shi, Y.; Li, J.; and Huang, J. 2026. UM-Text: A Unified Multimodal Model for Image Understanding. arXiv preprint arXiv:2601.08321.

Ma, L.; Yue, T.; Fu, P.; Zhong, Y.; Zhou, K.; Wei, X.; and Hu, J. 2024. Chargen: High accurate character-level visual text generation model with multimodal encoder. arXiv preprint arXiv:2412.17225.

OpenAI. 2026a. ChatGPT [Large language model]. Accessed 2026-06-26.

OpenAI. 2026b. GPT-Image 2. Accessed 2026-06-26.

Qian, Z.; Zhang, P.; Yang, B.; Fan, K.; Ma, Y.; Wong, D. F.; Sun, X.; and Ji, R. 2024. Anytrans: Translate anytext in the image with large scale models. In Findings ofthe Association for Computational Linguistics: EMNLP 2024, 2432–2444.

Qin, Y.; Cao, K.; Liu, H.; Ma, A.; Li, F.; Zhu, H.; Zhang, Z.; Ling, R.; Feng, W.; He, X.; et al. 2026. Innoads-composer: Eficient condition composition for e-commerce poster generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 32988–32999.

Rei, R.; Stewart, C.; Farinha, A. C.; and Lavie, A. 2020. COMET: A neural framework for MT evaluation. In Proceedings of the 2020 conference on empirical methods in natural language processing (emnlp), 2685–2702.

Rodriguez, J. A.; Puri, A.; Agarwal, S.; Laradji, I. H.; Rodriguez, P.; Rajeswar, S.; Vazquez, D.; Pal, C.; and Pedersoli, M. 2025. Starvector: Generating scalable vector graphics code from images and text. In Proceedings of the Computer Vision and Pattern Recognition Conference, 16175–16186.

Seedream Team. 2025. Seedream 4.0: Toward Nextgeneration Multimodal Image Generation. arXiv preprint arXiv:2509.20427.

Shao, Z.; Wang, P.; Zhu, Q.; Xu, R.; Song, J.; Bi, X.; Zhang, H.; Zhang, M.; Li, Y.; Wu, Y.; et al. 2024. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300.

Shu, Y.; Zeng, W.; Zhao, F.; Chen, Z.; Li, Z.; Yang, X.; Zhou, Y.; Rota, P.; Bai, X.; Jin, L.; et al. 2025. Visual text processing: A comprehensive review and unified evaluation. arXiv preprint arXiv:2504.21682.

Team, M. L.; Ma, H.; Tan, H.; Huang, J.; Wu, J.; He, J.-Y.; Gao, L.; Xiao, S.; Wei, X.; Ma, X.; et al. 2025. Longcatimage technical report. arXiv preprint arXiv:2512.07584.

Team, Q. 2026. Qwen3.5-omni technical report. arXiv preprint arXiv:2604.15804.

Tian, Y.; Liu, Z.; Liu, Z.; Feng, C.; Li, X.; Huang, H.-Y.; and Guo, Y. 2025. Prim: Towards practical in-image multilingual machine translation. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, 13693–13708.

Tuo, Y.; Xiang, W.; He, J.-Y.; Geng, Y.; and Xie, X. 2024. Anytext: Multilingual visual text generation and editing. In International Conference on Learning Representations, volume 2024, 56783–56799.

Wu, C.; Li, J.; Zhou, J.; Lin, J.; Gao, K.; Yan, K.; Yin, S.-m.; Bai, S.; Xu, X.; Chen, Y.; et al. 2025. Qwen-Image Technical Report. arXiv preprint arXiv:2508.02324.

Wu, R.; Su, W.; and Liao, J. 2025. Chat2svg: Vector graphics generation with large language models and image difusion models. In Proceedings ofthe Computer Vision and Pattern Recognition Conference, 23690–23700.

Yang, Y.; Cheng, W.; Chen, S.; Zeng, X.; Yin, F.; Zhang, J.; Wang, L.; Yu, G.; Ma, X.; and Jiang, Y.-G. 2026. Omnisvg: A unified scalable vector graphics generation model. Advances in Neural Information Processing Systems, 38: 113670–113696.

Ye, J.; He, J.; Huang, Z.; Jiang, D.; Yang, X.; Chen, R.; and Li, W. 2026. GenClaw: Code-Driven Agentic Image Generation. arXiv preprint arXiv:2605.30248.

Zeng, W.; Shu, Y.; Li, Z.; Yang, D.; and Zhou, Y. 2024. Textctrl: Difusion-based scene text editing with prior guidance control. Advances in Neural Information Processing Systems, 37: 138569–138594.

Zhao, S.; Xie, Z.; Liu, M.; Huang, J.; Pang, G.; Chen, F.; and Grover, A. 2026a. Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models. arXiv preprint arXiv:2601.18734.

Zhao, X.; Sun, Q.; Xiao, J.; Liu, X.; Yang, H.; Chen, Q.; Luo, X.; Huang, J.; Zhong, Y.; Chen, L.; et al. 2026b. Beyond NL2Code: A Structured Survey of Multimodal Code Intelligence. arXiv preprint arXiv:2606.15932.

Zhou, J.; Li, J.; Xu, Z.; Li, H.; Cheng, Y.; Hong, F.-T.; Lin, Q.; Lu, Q.; and Liang, X. 2025. FireEdit: Fine-grained Instruction-based Image Editing via Region-aware Vision Language Model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.