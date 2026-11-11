# UniWeaver: Harnessing Unified Multimodal Models for Interleaved Text-Image Generation

Hongxu Chen<sup>1</sup>\*, Zixuan Zhong<sup>2</sup>\*, Donghao Zhou<sup>3</sup>\*<sup>†</sup>, Haodong Li<sup>3</sup>, Xiaomin Yu<sup>4</sup>, Tianhao Li<sup>5</sup>, Shiyao Yu<sup>5</sup>, Zhongwei Wan<sup>1§</sup>, Pheng-Ann Heng<sup>3§</sup>

<sup>1</sup>OSU, <sup>2</sup>UCL, <sup>3</sup>CUHK, <sup>4</sup>HKUST(GZ), <sup>5</sup>HKU

Project Page: poscape.github.io/UniWeaver

## Abstract

Unified multimodal models (UMMs) have opened up new possibilities for generating rich interleaved text-image content. However, they generally lack native support for coordinating the two modalities, and translating coarse user requests into coherent multimodal compositions also requires explicit planning. We propose UniWeaver, a structured, model-agnostic harness framework that integrates multimodal planning, generation, and visual feedback to produce coherent interleaved outputs. To train UMMs for this task, we construct UniWeaver-10K, a dataset of 10,000 tasks across 20 task families, paired with execution trajectories guided by UniWeaver. We also introduce UniWeaver-Bench, a task-oriented benchmark systematically assessing text coverage, writing quality, visual coverage, and image-text alignment. Experiments show that UniWeaver achieves state-of-the-art performance and improves interleaved text-image generation across multiple UMM backbones.

## 1 Introduction

Large multimodal models have rapidly improved at understanding and generating visual content. More recently, unified multimodal models (UMMs) have brought visual understanding and generation into a shared architecture (Lu et al., 2024; Chameleon Team, 2024; Wang et al., 2024; Xie et al., 2025a; Wu et al., 2025; Deng et al., 2025), and related work started to explore the generation of free-form sequences of text and images (Dong et al., 2024; Tian et al., 2024; Xing et al., 2026; Wang et al., 2026). These advances make it possible to move beyond producing solely a single text or image answer toward richer multimodal outputs for travel guides, tutorials, and illustrated explanations (Liu et al., 2024; Zhou et al., 2025).

![](images/731f572a50203473da3c739b2b56f4be8d6bb4ecf043f1af156cf6f445d31b32.jpg)  
Figure 1: Paradigms of interleaved text-image generation. (a) Existing UMMs generate text and images sequentially. (b) UniWeaver structures multimodal planning, generation, and local visual verification.

However, enabling interleaved text-image generation for UMMs remains challenging. First, despite their unified understanding and generation capabilities, UMMs do not generally support the coordinated generation of interleaved text and images as a native capability. Generating such outputs requires not only producing each modality but also coordinating their alternation and ensuring that text and images complement one another throughout the multimodal output. Second, translating a user request into meaningful interleaved content requires high-level planning beyond individual text or image generation. User requests typically convey the coarse intended goal without specifying how it should be realized. Bridging this gap requires determining the granularity and number of content units, allocating information across text and images, and organizing them into a coherent sequence that jointly fulfills the user’s intent.

To address these challenges, we propose Uni-Weaver, a structured harness framework that integrates multimodal planning, generation, and visual feedback into a unified workflow (Figure 1). UniWeaver decomposes a user request into content units, determines what should be conveyed through text or images, and organizes them into a coherent interleaved output. This process coordinates the roles of the two modalities, ensuring that text and images complement each other rather than being generated as isolated outputs. By explicitly planning the content and structure before generation, UniWeaver bridges the gap between coarse user requests and complete, coherent multimodal outputs. It then produces reader-facing prose and image-generation instructions, optionally maintains reusable visual references across images, and verifies generated images through observable visual checks. Images that fail verification can be refined independently, reducing unnecessary sequential dependencies between unrelated content units. It is worth noting that the framework’s design is model-agnostic and can be instantiated with different UMM backbones. Figure 2 provides several examples of UniWeaver’s generated results, illustrating its ability to produce coherent and informative multimodal content.

![](images/271befb91f6024071c80a39ca25f47464538a176e478b29d3e13f5f5ec8cb508.jpg)  
Figure 2: Generated results of the proposed UniWeaver. Our harness framework enables coherent interleaved text-image generation for UMMs across diverse real-world tasks, producing task-relevant visuals paired with informative and actionable text.

learning how to organize content and coordinate text-image generation within the framework.

To train UMMs to plan and generate interleaved text-image content from natural user requests, we construct UniWeaver-10K, a specific dataset from 10,000 tasks spanning 20 task families. For each task, we use a strong LLM guided by the Uni-Weaver framework to construct a corresponding execution trajectory, yielding 10,000 execution trajectories for training. We then use these trajectories to train UMMs to improve their ability to autonomously plan and coordinate interleaved textimage generation. By deriving these trajectories from user requests that specify goals rather than predefined multimodal structures, our data construction process provides dense supervision for

Moreover, evaluating whether a generated interleaved output truly fulfills a user’s request requires more than assessing its individual modalities or their shallow coherence. Existing benchmarks such as InterleavedBench (Liu et al., 2024), ISG-Bench (Chen et al., 2025a), and OpenING (Zhou et al., 2025) measure only text quality, visual quality, and other limited properties, but these properties alone do not ensure that the output covers the information and visual evidence required by the user. We therefore introduce UniWeaver-Bench, a task-oriented benchmark that directly evaluates the completed outputs of reader-facing text and generated images. It measures task fulfillment through text coverage, writing quality, visual coverage, and image-text alignment, enabling comprehensive comparisons across methods with different generation mechanisms. Experiments demonstrate that our approach achieves state-of-the-art performance on UniWeaver-Bench and consistently improves interleaved text-image generation when integrated with multiple UMM backbones.

Our main contributions are as follows:

• We develop UniWeaver, a structured, modelagnostic harness framework for UMMs, which integrates multimodal planning, generation, and visual feedback for interleaved text-image generation.

• We construct UniWeaver-10K, a dataset of 10,000 tasks spanning 20 task families, paired with execution trajectories for training UMMs to plan and generate coherent interleaved textimage content.

• We introduce UniWeaver-Bench, a taskoriented benchmark for the systematic evaluation of interleaved text-image generation.

## 2 Related Work

Unified Multimodal Models. Unified multimodal models (UMMs) aim to support visual understanding and generation within a single model (Xie et al., 2026a). Unified-IO 2 (Lu et al., 2024) unifies multiple input and output modalities through a shared semantic space, while Chameleon (Chameleon Team, 2024) and Emu3(Wang et al., 2024) mixed imagetext sequences autoregressively. Show-o (Xie et al., 2025a) and Janus (Wu et al., 2025) explore different architectures for unifying visual understanding and generation. More recently, BAGEL (Deng et al., 2025) advances unified multimodal pretraining for understanding, generation, editing, and multimodal reasoning. Wan-Weaver (Xing et al., 2026) further explores interleaved generation through a plannervisualizer architecture and decoupled training for textual planning and visual consistency modeling. These features provide capable foundations for multimodal understanding and generation.

Multimodal Generation Systems. Related work has explored controllable visual generation (Zhou et al., 2024; Liu et al., 2026; Zhou et al., 2026d; Xie et al., 2026b) and consistent visual storytelling (Zhou et al., 2026c; Song et al., 2025). Planning and feedback have been widely studied for solving multi-step tasks (Yao et al., 2023; Shinn et al., 2023; Zhou et al., 2026a). These ideas have also been extended to multimodal settings, where models can reason over visual observations (Zhou et al., 2026b) and interact with external tools (Yang et al., 2023). For interleaved generation, ISG-Agent introduces a plan–execute–refine pipeline that coordinates multimodal tools (Chen et al., 2025a), while InterleaveThinker employs planner and critic agents to guide image generators through global planning and step-wise visual refinement (Zheng et al., 2026). These studies demonstrate the role of planning and feedback in coordinating multimodal generation processes. UniWeaver integrates these capabilities into a structured harness that decomposes user requests into content units, coordinates the complementary roles of text and images, and supports visual verification and targeted refinement. The resulting execution trajectories also provide supervision for training UMMs to plan and generate interleaved content.

Interleaved Generation and Evaluation. Interleaved generation models have explored producing free-form sequences of text and images. GILL (Koh et al., 2023) connects language models with image generation modules, while DreamLLM and MM-Interleaved (Dong et al., 2024; Tian et al., 2024) investigate the generation of interleaved multimodal sequences. Alongside advances in generation, benchmarks have also been developed to assess the quality of complete multimodal outputs (Huang et al., 2026). InterleavedBench (Liu et al., 2024) evaluates text quality, perceptual quality, image coherence, text-image coherence, and helpfulness. ISG-Bench (Chen et al., 2025a) introduces finer-grained evaluation of interleaved outputs, while OpenING (Zhou et al., 2025) expands evaluation to diverse open-ended real-world tasks and introduces IntJudge for multimodal assessment. More recently, WeaverBench (Xing et al., 2026) evaluates prompt adherence, narrative coordination, content consistency, image consistency, and completeness. Building on these evaluation efforts, UniWeaver-Bench provides a task-oriented assessment of completed interleaved outputs through text coverage, writing quality, visual coverage, and image-text alignment. It evaluates final readerfacing content, thereby enabling comparison across multimodal generation workflows.

## 3 Methodology

We introduce UniWeaver (Section 3.1), a structured, model-agnostic harness framework for interleaved text-image generation from natural user requests. As shown in Figure 3, UniWeaver integrates multimodal planning, generation, selective visual conditioning, and visual verification into a unified workflow, enabling coherent coordination between text and images. To support training and systematic evaluation, we further construct UniWeaver-10K (Section 3.2), a dataset of 10,000 tasks spanning 20 task families, paired with execution trajectories for structured planning supervision, and UniWeaver-Bench (Section 3.3), a 120-task benchmark for evaluating the quality and task fulfillment of final interleaved outputs.

![](images/579211826f8f136864f2315092a358d90597de854a7acaf49d1e3be89fe97186.jpg)  
Figure 3: Overview of UniWeaver. ① UniWeaver introduces a structured workflow for interleaved text-image generation, with execution trajectories providing supervision for fine-tuning (Section 3.1). ② UniWeaver-10K Dataset provides 10,000 goal-only tasks paired with structured planning trajectories across 20 task families (Section 3.2). ③ UniWeaver-Bench systematically evaluates final interleaved outputs on 120 held-out tasks (Section 3.3).

## 3.1 Overall Framework of UniWeaver

As illustrated by Figure 3, UniWeaver converts a natural request into a structured multimodal plan, realizes the planned text-image units, and verifies generated visual content before assembling the final response. Let x denote a natural user request. UniWeaver produces an interleaved output:

$$
\mathcal { Y } = \left[ ( c _ { 1 } , I _ { 1 } ) , \dots , ( c _ { N } , I _ { N } ) \right] ,\tag{1}
$$

where N denotes the number of content units, and $c _ { i }$ and $I _ { i }$ represent the reader-facing text and generated image of the i-th unit, respectively.

Multimodal Planning. Users typically describe a high-level task goal rather than a predefined structure, leaving the number of content units, the output organization, and the allocation of information between text and images unspecified. Uni-Weaver therefore performs multimodal planning Narratives Fashionbefore generation to determine these structures and <sup>Culture +</sup> <sup>more</sup> <sup>...scrambled</sup> <sup>eggs</sup> <sup>step</sup> <sup>by</sup> <sup>step</sup> <sup>. .</sup>establish a coherent organization for the final out-<sub>K</sub> <sub>tasks 20</sub> <sub>task</sub> <sub>families Balanced</sub> put, which consists of four stages:

(1) Scope Decomposition. UniWeaver first constructs a coarse content structure:

$$
P _ { 1 } = \left( \tau , e , \{ ( s _ { i } , g _ { i } ) \} _ { i = 1 } ^ { N } \right) ,\tag{2}
$$

where $\tau$ is the output title, e is the task endpoint, and each unit contains a visual subject $s _ { i }$ and textual content goal $g _ { i }$ . The endpoint defines what constitutes a complete response, while the unit-level goals specify what each part should cover.

(2) Text-Visual Realization. Given the content structure, UniWeaver produces reader-facing text and image-generation instructions:

$$
P _ { 2 } = \left( v , \{ ( p _ { i } , c _ { i } ) \} _ { i = 1 } ^ { N } \right) ,\tag{3}
$$

where $p _ { i }$ is the image prompt, $c _ { i }$ is the readerfacing text, and v denotes a shared visual style. This separation allows explanatory or relational information to remain in text while the image prompt focuses on directly visualizable content. The shared style provides a lightweight global constraint across generated images.

(3) Selective Visual Conditioning. When a visual concept, such as a specific object, should remain consistent across units, UniWeaver constructs an optional reference bank and routes only the relevant references to each unit. We texdenote the routed reference set for unit i as $R _ { i } .$ <sub>cle</sub>. Unlike fully sequential generation, where each image may depend on the preceding image, UniWeaver introduces such dependencies only when they are useful, allowing unrelated units to remain independent.

(4) Visual Verification. For each unit, UniWeaver derives a compact checklist $C _ { i }$ of directly observable visual requirements, such as the presence of an object, action, state change, or visible outcome. These checks are restricted to properties that can be judged from the image itself and do not depend on unconstrained and hidden factual reasoning.

Generation and Refinement. As shown in Figure 3, UniWeaver first generates any references required by Stage 3 and then produces each image from its planned prompt, shared style, and routed references:

$$
I _ { i } ^ { ( 0 ) } = G ( \tilde { p } _ { i } , R _ { i } ) , \qquad \tilde { p } _ { i } = p _ { i } \oplus v ,\tag{4}
$$

where $G$ denotes the visual generation capability of the underlying backbone. Each image is then checked against its unit-specific checklist:

$$
V ( I _ { i } ^ { ( r ) } , C _ { i } ) \to ( j _ { i } ^ { ( r ) } , f _ { i } ^ { ( r ) } ) ,\tag{5}
$$

where $j _ { i } ^ { ( r ) }$ is the verification result and $f _ { i } ^ { ( r ) }$ provides feedback for failed criteria. If verification fails, the feedback is appended to the original generation instruction and only that unit is regenerated. The framework allows at most two additional attempts after the initial generation, while successful units remain unchanged. Verification therefore acts as a local repair mechanism rather than requiring regeneration of the entire response.

Supervised Fine-Tuning. As shown in the first part of Figure 3, we use the structured outputs from UniWeaver-10K to adapt the planning and understanding pathway of the unified backbone. Unlike approaches that train separate planning and critic models (Zheng et al., 2026), UniWeaver directly fine-tunes the unified backbone while keeping its visual generation pathway frozen. For each task x, the supervision target is

$$
Y = ( P _ { 1 } , P _ { 2 } , P _ { 3 } , P _ { 4 } ) ,\tag{6}
$$

and the model is trained autoregressively using

$$
\mathcal { L } _ { \mathrm { S F T } } = - \sum _ { k = 1 } ^ { 4 } \sum _ { t = 1 } ^ { \left| P _ { k } \right| } \log p _ { \theta } \left( P _ { k , t } \mid x , P _ { < k } , P _ { k , < t } \right) .\tag{7}
$$

The loss is applied only to the generated planning text tokens. For instance, with BAGEL (Deng et al., 2025) as the backbone, LoRA (Hu et al., 2022) is applied to the understanding decoder pathway while the image-generation-specific experts remain frozen. This allows the model to internalize the UniWeaver planning procedure while preserving its pretrained visual synthesis capability. Detailed optimization settings are provided in Section 4.1.

## 3.2 UniWeaver-10K Dataset

The second part of Figure 3 illustrates the construction of UniWeaver-10K through candidate generation, filtering, and selection. UniWeaver-10K contains 10,000 tasks evenly distributed across 20 task families, with 500 tasks per family. The selected tasks follow an approximately 1:2:1 easy-mediumhard distribution. Figure 4 further presents dataset statistics, highlighting the diversity of tasks and their characteristics.

Trajectory Generation. We define task families covering procedures, physical setup, transformations, comparisons, product use, travel, lifestyle, narratives, culture, architecture, nature, photography, and related multimodal scenarios. Using DeepSeek-v4-Pro (DeepSeek-AI, 2026a) as the teacher model, we generate more than 30,000 candidate requests from family-specific combinations of topics, audiences, situations, constraints, and intended outcomes. A key design choice is that each request specifies only the user goal. We remove prompts that prescribe a fixed image count, storyboard, output outline, or explicit text-image correspondence. The model must therefore infer how the final response should be organized rather than recover that structure from the input.

Filtering and Supervision. Candidates are filtered for task-family consistency, naturalness, multimodal usefulness, format validity, and duplication, followed by stratified selection over family and difficulty. For each task, DeepSeek-v4-Pro generates the same four-stage structure used at inference time. Later stages are conditioned on earlier ones to maintain cross-stage consistency. We retain examples that satisfy the UniWeaver planning schema and consistency checks. Supervision consists only of the structured planning outputs; generated image pixels and explicit chain-of-thought are not used as training targets. We obtain 10,000 structured execution trajectories for training UMMs to plan and coordinate interleaved text-image generation.

## 3.3 UniWeaver-Bench

Figure 3 illustrates UniWeaver-Bench, which evaluates final interleaved outputs through a validity check and four criteria: text coverage, writing quality, visual coverage, and image-text alignment.

Test Samples. As shown in Figure 4(a), it contains 120 held-out tasks across the same 20 task families and 4 major categories, with 6 tasks per family and 40 easy, 40 medium, and 40 hard examples.

![](images/1083787c8b58a496e6634d0434c6c2e1622fe3bb8d1775f80edd24ba298cbb18.jpg)  
(a) Task Distribution

![](images/593c04347fd42b77dc89cfe218462434770fb633f798b47c70c4d9c8df462a20.jpg)  
(c) Corpus Vocabulary  
Figure 4: Statistics of UniWeaver-10K and UniWeaver-Bench. (a) Task distribution across 20 diverse families, with balanced coverage in both datasets. (b) Distribution of training sample lengths in tokens. (c) Vocabulary characteristics of the training corpus, visualized as a word cloud.

To ensure a reliable evaluation of generalization, benchmark tasks are held out from training, with overlap prevented through task identity and normalized prompt hash matching. Each task provides only a natural-language goal and is paired with hidden evaluation references containing task-specific coverage requirements and observable visual evidence. A deterministic validity check first verifies that the output contains usable text-image units with one-to-one image pairing.

Evaluation Metrics. Valid outputs are evaluated along four dimensions: text coverage (TC), writing quality (WQ), visual coverage (VC), and image-text alignment (ITA). Each criterion is rated as not met, partially met, or met, mapped to 0, 0.5, and 1. The quality of a gate-passing task t is

$$
Q _ { t } = 0 . 4 5 \mathrm { T C } _ { t } + 0 . 1 0 \mathrm { W Q } _ { t } + 0 . 3 5 \mathrm { V C } _ { t } + 0 . 1 0 \mathrm { T T A } _ { t } .\tag{8}
$$

To account for failures to produce a valid report, we compute for each family f

$$
S _ { f } = \bar { Q } _ { f } \sqrt { r _ { f } } , \qquad \mathrm { T o t a l } = { \frac { 1 } { 2 0 } } \sum _ { f = 1 } ^ { 2 0 } S _ { f } ,\tag{9}
$$

where $Q _ { f }$ is the mean of $Q _ { t }$ over the gate-passing tasks in family $f$ and $r _ { f }$ is its gate pass rate. Thus, invalid outputs are penalized without allowing report availability to completely dominate the quality of successful generations. We independently evaluate each output using three judges: Kimi K3 (Kimi Team, 2026), DeepSeek-V4.1-Flash (DeepSeek-AI, 2026b), and GLM-5.3-Flash (GLM, 2026). Each judge performs two evaluations at temperature 0, with the median retained. Cross-judge ratings are conservatively aggregated, with disagreements resolved toward the lower tier (Appendix C).

## 4 Experiments

## 4.1 Experimental Setup

We evaluate seven UMM backbones: BAGEL (Deng et al., 2025), OmniGen2 (Wu et al., 2026), Show-o2 (Xie et al., 2025b), TokenFlow (Qu et al., 2025), ILLUME+ (Huang et al., 2025), Janus-Pro (Chen et al., 2025b), and SEED-X (Ge et al., 2024). The baseline employs a minimal step-wise harness, generating successive text-image pairs conditioned on the user request and preceding steps, without explicit planning or visual verification. UniWeaver applies its four-stage framework to the same backbones without updating model parameters. For supervised adaptation, we fine-tune BAGEL on UniWeaver-10K using LoRA while keeping its image-generation experts frozen, denoted as UniWeaver-SFT. We use the AdamW optimizer with a learning rate of $5 \times 1 0 ^ { - 5 }$ , LoRA rank 32, alpha 64, and dropout 0.05, training for 40,000 steps (two epochs). Additional optimization details are provided in Appendix D. All methods are evaluated on UniWeaver-Bench using the metrics defined in Section 3.3.

## 4.2 Main Results

Quantitative Comparison. As shown in Table 1, minimal step-wise harness with UMM backbones yields limited performance on open-ended interleaved tasks, with BAGEL, OmniGen2, and Token-Flow achieving overall scores of 27.7, 27.9, and 39.2, respectively. In contrast, UniWeaver substantially improves their performance to 58.9, 52.7, and 60.5 without parameter updates. These consistent gains demonstrate the effectiveness of structured multimodal planning and generation across different backbones. Training with UniWeaver-10K further improves performance. UniWeaver-SFT achieves the highest overall score of 87.1, substantially outperforming both framework-only UniWeaver (58.9) and minimal step-wise harness (27.7) on BAGEL. This demonstrates the additional benefit of learning structured planning and generation from execution trajectories. The improvements extend across multiple evaluation dimensions. On BAGEL, UniWeaver increases TC from 21.4 to 48.9, WQ from 8.3 to 89.4, VC from 45.3 to 64.8, and ITA from 14.0 to 52.5. UniWeaver-SFT further achieves 82.0 TC, 97.9 WQ, 91.4 VC, and 83.7 ITA. These results highlight improvements in task-relevant coverage, reader-facing writing quality, visual support, and text-image coordination.

Table 1: Quantitative Comparison on the 120-task UniWeaver-Bench. Each major category contains 5 task families covering 30 tasks. The best result is bold and the second best underlined. Red values indicate performance gains over the corresponding UMM baseline. Only gate-passing tasks are averaged, and a criterion whose judge medians span more than one tier scores 0. Each family’s score is multiplied by ${ \sqrt { r _ { f } } } ,$ where $r _ { f }$ is its gate pass rate.
<table><tr><td rowspan="2">Model</td><td colspan="3">Procedures &amp; Instructions</td><td colspan="3">Comparison &amp; Recommendation</td><td colspan="4">Observation &amp; Description</td><td colspan="4">Explanation &amp; Storytelling</td><td colspan="3">Overall</td></tr><tr><td>WQ</td><td>VC ITA</td><td>Total</td><td>TC WQ</td><td>VC ITA</td><td>Total</td><td>TC</td><td>WQ</td><td>VC ITA</td><td>Total</td><td>TC WQ</td><td>VC</td><td>ITA</td><td>Total</td><td>TC WQ</td><td>VC ITA</td><td>Total</td></tr><tr><td>BAGEL</td><td>14.8 7.3</td><td>48.0 14.1</td><td>25.6</td><td>28.3 14.2</td><td>35.1 14.1</td><td>27.8</td><td>26.1</td><td>4.9</td><td>45.9 15.0</td><td>29.8</td><td>16.3</td><td>6.8 52.2</td><td>13.0</td><td>27.6</td><td>21.4 8.3</td><td>45.3</td><td>14.0 27.7</td></tr><tr><td>OmniGen2</td><td>12.3 13.8</td><td>47.9 21.9</td><td>25.9</td><td>19.6 14.2</td><td>40.4 24.7</td><td>26.8</td><td>16.0</td><td>2.5</td><td>44.6 23.3</td><td>25.4</td><td>23.2 5.4</td><td>53.3</td><td>40.4</td><td>33.7 17.8</td><td>9.0</td><td>46.5 27.6</td><td>27.9</td></tr><tr><td>TokenFlow</td><td>30.2 24.6</td><td>65.4 40.3</td><td>43.0</td><td>32.1 8.8</td><td>55.8 25.8</td><td>37.4</td><td>31.3</td><td>6.7</td><td>60.4 27.2</td><td>38.6</td><td>28.1 10.4</td><td>60.3</td><td>29.4</td><td>37.7</td><td>30.4 12.6</td><td>60.5 30.7</td><td>39.2</td></tr><tr><td>ILLUME+</td><td>23.3 27.1</td><td>51.7 47.2</td><td>36.0</td><td>35.2 10.8</td><td>55.0 35.0</td><td>39.7</td><td>34.2</td><td>12.9</td><td>57.1 26.5</td><td>39.3</td><td>27.1 5.8</td><td>66.7</td><td>25.8</td><td>38.7 30.0</td><td>14.2</td><td>57.6 33.6</td><td>38.4</td></tr><tr><td>Show-o2</td><td>10.0 43.4</td><td>21.5 9.2</td><td>17.3</td><td>15.6 36.6</td><td>36.7 12.5</td><td>24.8</td><td>20.1</td><td>30.2</td><td>38.4 13.2</td><td>26.8</td><td>19.3 32.5</td><td>41.7</td><td>21.8</td><td>28.7 16.2</td><td>35.7</td><td>34.6 14.2</td><td>24.4</td></tr><tr><td>Janus-Pro</td><td>18.8 9.5</td><td>27.5 5.1</td><td>19.5</td><td>22.7 6.6</td><td>24.0 6.4</td><td>19.9</td><td>23.0</td><td>6.2</td><td>45.2 9.8</td><td>27.8</td><td>10.8 4.2</td><td>24.9</td><td>6.2</td><td>14.6 18.8</td><td>6.6</td><td>30.4 6.9</td><td>20.5</td></tr><tr><td>SEED-X</td><td>2.7 1.7</td><td>25.8 3.9</td><td>10.8</td><td>4.6 1.2</td><td>29.6 0.0</td><td>12.5</td><td>2.7</td><td>0.0</td><td>32.5 0.8</td><td>12.7</td><td>0.0 0.0</td><td>32.2</td><td>0.0</td><td>11.3 2.5</td><td>0.7</td><td>30.0 1.2</td><td>11.8</td></tr><tr><td>UniWeaver + BAGEL</td><td>47.3 90.1</td><td>60.3 44.1</td><td>55.8</td><td>47.1 91.9</td><td>60.0 50.9</td><td>56.5 ∆ + 28.6</td><td>45.1</td><td>86.4</td><td>66.2 56.7</td><td>57.8</td><td>56.0 89.4</td><td>72.8 58.4</td><td>65.5 Δ + 37.9</td><td>48.9</td><td>89.4 64.8</td><td>52.5</td><td>58.9</td></tr><tr><td>UniWeaver + OmniGen2</td><td>42.3 73.1</td><td>58.8 45.8</td><td>Δ + 30.2 51.5</td><td>41.1 65.5</td><td>49.9 33.9</td><td>45.9</td><td>39.2</td><td>74.2</td><td>61.1 48.6</td><td>Δ + 28.0 51.3</td><td>52.9 64.6</td><td>73.5 61.6</td><td>62.2</td><td></td><td>43.9 69.4 60.8</td><td>47.5</td><td>Δ + 31.2 52.7</td></tr><tr><td>UniWeaver</td><td>68.1 92.8 56.0 40.9</td><td></td><td>Δ + 25.6 63.6</td><td></td><td>58.9 86.5 47.4 50.4</td><td>∆ + 19.1 56.8</td><td>53.3</td><td>74.6</td><td>59.1 54.9</td><td>Δ + 25.9 57.6</td><td>62.6</td><td>83.2 63.0 56.0</td><td></td><td>Δ + 28.5 64.2</td><td>60.7 84.3 56.4 50.5</td><td></td><td>∆ + 24.8 60.5</td></tr><tr><td>+ TokenFlow UniWeaver</td><td>86.9 98.3 95.8 79.2</td><td></td><td>∆ + 20.7</td><td>81.7 97.5 89.2 80.6</td><td></td><td>Δ + 19.3 85.8</td><td></td><td>78.1 100.0 90.0 86.7</td><td></td><td>∆ + 19.0</td><td>81.5 95.8 90.7</td><td></td><td>Δ + 26.4 86.8</td><td></td><td>82.0 97.9 91.4 83.7</td><td></td><td>Δ + 21.3</td></tr><tr><td>+ BAGEL-SFT</td><td></td><td></td><td>90.4 Δ + 64.8</td><td></td><td></td><td>Δ + 57.9</td><td></td><td></td><td></td><td>85.3 Δ + 55.5</td><td></td><td>88.3</td><td>Δ + 59.2</td><td></td><td></td><td></td><td>87.1 Δ + 59.4</td></tr></table>

Qualitative Comparison. Figure 5 compares Uni-Weaver with competitive baselines on two representative tasks: an outdoor workshop packing guide and a beginner cooking guide, examining their ability to produce coherent, informative interleaved outputs. The baselines frequently exhibit repetitive content, incomplete guidance, and ineffective visual support. For the packing guide, BAGEL introduces less relevant accessories, OmniGen2 repeats checklist illustrations, and Janus-Pro generates illegible text within images. For the cooking guide, BAGEL omits key steps, OmniGen2 produces unreadable shopping lists, and ILLUME+ repeats similar scenes and instructions. In contrast, UniWeaver delivers practical, well-organized guidance with clearer, task-relevant visuals. The packing guide provides specific clothing and equipment recommendations with explanations of their practical benefits and corresponding illustrations. The cooking guide progresses coherently from preparation to baking and serving, providing actionable details such as cooking temperature and duration, with images depicting distinct stages of the process. These results highlight UniWeaver’s advantages in task-relevant coverage, writing quality, visual effectiveness, and text-image alignment. Rather than merely producing related text and images, UniWeaver organizes complementary multimodal content into coherent, actionable guides that better fulfill user requests.

## 4.3 Ablation Studies

We conduct seven ablation studies on UniWeaver + BAGEL to assess the contributions of key components of the proposed framework, as shown in Table 2. Specifically, $P _ { 1 }$ examines task-level and unit-level goals, $P _ { 2 }$ evaluates shared visual style and text-image role separation, and $P _ { 3 }$ investigates reference selection, routing, and conditioning.

Observation #1: Explicit task goals anchor goaldriven content planning. Removing the task goal reduces the overall score from 58.89 to 53.01 (- 5.88), while removing per-step content goals yields 55.56 (-3.33). The larger degradation without the task goal highlights the importance of defining what constitutes a complete response. Meanwhile, unit-level goals help specify what each part should contribute. Together, these results support planning content around both the overall user objective and the responsibilities of individual units.

Observation #2: Coherent interleaved generation requires visual consistency and distinct modality roles. Removing the shared visual style causes the largest performance drop among all ablations, reducing the score to 48.21 (-10.68). Combining reader-facing content and image instructions into a single field also reduces performance, yielding a score of 53.48 (-5.41). These results highlight the importance of maintaining a consistent visual presentation across content units and explicitly distinguishing information to be communicated through text from content to be depicted in images.

![](images/eacbfc9b93fe42b681cfdee47e6576e516b45e6d512d67df8b53fd6bbd21285e.jpg)  
Figure 5: Qualitative comparison on two representative tasks. UniWeaver produces more coherent, actionable instructions with task-relevant visuals, while baselines exhibit repetitive content, missing details, or ineffective visual support. Zoom infor better view.

Table 2: Ablation studies of UniWeaver. All variants are re-run on the same 120 tasks with the same evaluation setup. Note that ∆ denotes the performance gap measured against the full method.
<table><tr><td>Ablation</td><td>Total</td><td>∆</td></tr><tr><td>UniWeaver (w/ BAGEL backbone)</td><td>58.89</td><td></td></tr><tr><td>P1: Task Scope</td><td></td><td></td></tr><tr><td>w/o Task Goal</td><td>53.01</td><td>-5.88</td></tr><tr><td>w/o Per-step Content Goal</td><td>55.56</td><td>-3.33</td></tr><tr><td>P2: Text and Image Planning</td><td></td><td></td></tr><tr><td>w/o Shared Visual Style</td><td>48.21</td><td>-10.68</td></tr><tr><td>w/ Merged Image Prompt and Content</td><td>53.48</td><td>-5.41</td></tr><tr><td>P3: Visual References</td><td></td><td></td></tr><tr><td>References Fixed before the Text</td><td>57.68</td><td>-1.21</td></tr><tr><td>w/o Per-step Reference Routing</td><td>57.89</td><td>-1.00</td></tr><tr><td>w/o All Reference Images</td><td>58.48</td><td>-0.41</td></tr></table>

Observation #3: Selective visual conditioning supports cross-unit coherence. Fixing references before text planning reduces the score from 58.89 to 57.68 (-1.21), while removing per-step reference routing or all reference images yields 57.89 (-1.00) and 58.48 (-0.41), respectively. These results suggest that selecting references with access to the planned image instructions, together with unit-specific reference routing, helps coordinate visual content across units.

## 5 Conclusion

In this work, we study interleaved text-image generation from natural user requests. We introduce UniWeaver, a model-agnostic harness that structures open-ended requests into multimodal content units, coordinates text and image generation, and enables selective visual conditioning and local verification. We further develop UniWeaver-10K for trajectory supervision and UniWeaver-Bench for evaluating task fulfillment from final outputs. Experiments across multiple multimodal backbones show that UniWeaver improves generation without parameter updates, while supervised fine-tuning yields further gains. These findings suggest that effective interleaved generation requires not only multimodal generation capabilities, but also structured planning and coordination to fulfill user intent through coherent multimodal outputs.

## Limitations

Despite its effectiveness, UniWeaver remains constrained by the capabilities of underlying UMMs, whose generated text and images may not yet meet professional-grade quality standards for demanding applications. In addition, our current study focuses on a defined set of interleaved generation tasks, leaving broader task scenarios and application settings to be explored. Future work will investigate stronger multimodal backbones and extend Uni-Weaver to a wider range of tasks and applications, further improving the quality and applicability of interleaved multimodal generation.

## Ethical Considerations

UniWeaver builds on existing UMMs and therefore inherits their potential biases and limitations in generated text and images. Although the framework improves the organization and consistency of multimodal outputs, it does not independently guarantee the factual accuracy, fairness, or appropriateness of all generated content. In practical applications, users should exercise appropriate judgment when using generated outputs, particularly in contexts where accuracy and reliability are important. Future work may incorporate additional safeguards and more comprehensive evaluations to support the responsible deployment of interleaved multimodal generation systems.

## References

Chameleon Team. 2024. Chameleon: Mixedmodal early-fusion foundation models. CoRR, abs/2405.09818.

Dongping Chen, Ruoxi Chen, Shu Pu, Zhaoyi Liu, Yanru Wu, Caixi Chen, Benlin Liu, Yue Huang, Yao Wan, Pan Zhou, and Ranjay Krishna. 2025a. Interleaved scene graphs for interleaved text-andimage generation assessment. In The Thirteenth International Conference on Learning Representations, ICLR 2025, Singapore, April 24-28, 2025. OpenReview.net.

Xiaokang Chen, Zhiyu Wu, Xingchao Liu, Zizheng Pan, Wen Liu, Zhenda Xie, Xingkai Yu, and Chong Ruan. 2025b. Janus-pro: Unified multimodal understanding and generation with data and model scaling. CoRR, abs/2501.17811.

DeepSeek-AI. 2026a. Deepseek-v4: Towards highly efficient million-token context intelligence. CoRR, abs/2606.19348.

DeepSeek-AI. 2026b. Deepseek-v4.1-flash: Pushing the limits of KV cache compression. CoRR, abs/2609.19969.

Chaorui Deng, Deyao Zhu, Kunchang Li, Chenhui Gou, Feng Li, Zeyu Wang, Shu Zhong, Weihao Yu, Xiaonan Nie, Ziang Song, Shi Guang, and Haoqi Fan. 2025. Emerging properties in unified multimodal pretraining. CoRR, abs/2505.14683.

Runpei Dong, Chunrui Han, Yuang Peng, Zekun Qi, Zheng Ge, Jinrong Yang, Liang Zhao, Jianjian Sun, Hongyu Zhou, Haoran Wei, Xiangwen Kong, Xiangyu Zhang, Kaisheng Ma, and Li Yi. 2024. Dreamllm: Synergistic multimodal comprehension and creation. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net.

Yuying Ge, Sijie Zhao, Jinguo Zhu, Yixiao Ge, Kun Yi, Lin Song, Chen Li, Xiaohan Ding, and Ying Shan. 2024. SEED-X: multimodal models with unified multi-granularity comprehension and generation. CoRR, abs/2404.14396.

GLM. 2026. GLM-5: from vibe coding to agentic engineering. CoRR, abs/2602.15763.

Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2022. Lora: Low-rank adaptation of large language models. In The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022. OpenReview.net.

Peizhou Huang, Zixuan Zhong, Zhongwei Wan, Donghao Zhou, Samiul Alam, Xin Wang, Zexin Li, Zhihao Dou, Li Zhu, Jing Xiong, et al. 2026. Mmdeepresearch-bench: A benchmark for multimodal deep research agents. arXiv preprint arXiv:2601.12346.

Runhui Huang, Chunwei Wang, Junwei Yang, Guansong Lu, Yunlong Yuan, Jianhua Han, Lu Hou, Wei Zhang, Lanqing Hong, Hengshuang Zhao, and Hang Xu. 2025. ILLUME+: illuminating unified MLLM with dual visual tokenization and diffusion refinement. CoRR, abs/2504.01934.

Kimi Team. 2026. Kimi K3: open frontier intelligence. CoRR, abs/2607.24653.

Jing Yu Koh, Daniel Fried, and Russ Salakhutdinov. 2023. Generating images with multimodal language models. In Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023.

Minqian Liu, Zhiyang Xu, Zihao Lin, Trevor Ashby, Joy Rimchala, Jiaxin Zhang, and Lifu Huang. 2024. Holistic evaluation for interleaved text-and-image

generation. In Proceedings ofthe 2024 Conference on Empirical Methods in Natural Language Processing, EMNLP 2024, Miami, FL, USA, November 12- 16, 2024, pages 22002–22016. Association for Computational Linguistics.

Yichen Liu, Donghao Zhou, Jie Wang, Xin Gao, Guisheng Liu, Jiatong Li, Quanwei Zhang, Qiang Lyu, Lanqing Guo, Shilei Wen, et al. 2026. Hifiinpaint: Towards high-fidelity reference-based inpainting for generating detail-preserving humanproduct images. arXiv preprint arXiv:2603.02210.

Jiasen Lu, Christopher Clark, Sangho Lee, Zichen Zhang, Savya Khosla, Ryan Marten, Derek Hoiem, and Aniruddha Kembhavi. 2024. Unified-io 2: Scaling autoregressive multimodal models with vision, language, audio, and action. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2024, Seattle, WA, USA, June 16-22, 2024, pages 26429–26445. IEEE.

Liao Qu, Huichao Zhang, Yiheng Liu, Xu Wang, Yi Jiang, Yiming Gao, Hu Ye, Daniel K. Du, Zehuan Yuan, and Xinglong Wu. 2025. Tokenflow: Unified image tokenizer for multimodal understanding and generation. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2025, Nashville, TN, USA, June 11-15, 2025, pages 2545–2555. Computer Vision Foundation / IEEE.

Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. 2023. Reflexion: language agents with verbal reinforcement learning. In Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023.

Quanjian Song, Donghao Zhou, Jingyu Lin, Fei Shen, Jiaze Wang, Xiaowei Hu, Cunjian Chen, and Pheng-Ann Heng. 2025. Scenedecorator: Towards sceneoriented story generation with scene planning and scene consistency. arXiv preprint arXiv:2510.22994.

Changyao Tian, Xizhou Zhu, Yuwen Xiong, Weiyun Wang, Zhe Chen, Wenhai Wang, Yuntao Chen, Lewei Lu, Tong Lu, Jie Zhou, Hongsheng Li, Yu Qiao, and Jifeng Dai. 2024. Mm-interleaved: Interleaved image-text generative modeling via multi-modal feature synchronizer. CoRR, abs/2401.10208.

Chonghuinan Wang, Zhikai Chen, Chunwei Wang, Yecong Wan, Junwei Yang, Zhixin Wang, Wei Zhang, Jiaqi Xu, Renjing Pei, Xiaohe Wu, Fan Li, and Wangmeng Zuo. 2026. Illuminating unified multimodal model for free-form interleaved text-image generation. In Computer Vision - ECCV 2026 - 19th European Conference, Malmö, Sweden, September 8-12, 2026, Proceedings, Part LXXV, volume 17075 of Lecture Notes in Computer Science, pages 535–552. Springer.

Xinlong Wang, Xiaosong Zhang, Zhengxiong Luo, Quan Sun, Yufeng Cui, Jinsheng Wang, Fan Zhang,

Yueze Wang, Zhen Li, Qiying Yu, Yingli Zhao, Yulong Ao, Xuebin Min, Tao Li, Boya Wu, Bo Zhao, Bowen Zhang, Liangdong Wang, Guang Liu, and 6 others. 2024. Emu3: Next-token prediction is all you need. CoRR, abs/2409.18869.

Chengyue Wu, Xiaokang Chen, Zhiyu Wu, Yiyang Ma, Xingchao Liu, Zizheng Pan, Wen Liu, Zhenda Xie, Xingkai Yu, Chong Ruan, and Ping Luo. 2025. Janus: Decoupling visual encoding for unified multimodal understanding and generation. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2025, Nashville, TN, USA, June 11-15, 2025, pages 12966–12977. Computer Vision Foundation / IEEE.

Chenyuan Wu, Jiahao Wang, Pengfei Zheng, Ruiran Yan, Shitao Xiao, Xin Luo, Yueze Wang, Wanli Li, Xiyan Jiang, Yexin Liu, Junjie Zhou, Ziyi Xia, Ze Liu, Chaofan Li, Haoge Deng, Kun Luo, Bo Zhang, Jiajun Zhang, Dong Liu, and 5 others. 2026. Omnigen2: Towards instruction-aligned multimodal generation. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2026, Denver, CO, USA, June 3-7, 2026, pages 21964–21975. Computer Vision Foundation.

Ji Xie, Luke Zettlemoyer, Xudong Wang, et al. 2026a. Reconstruction alignment improves unified multimodal models. In International Conference on Learning Representations, volume 2026, pages 120095– 120137.

Ji Xie, Dewei Zhou, Xinyu Huang, Zhennan Chen, and Xun Wang. 2026b. Paint-anything: Unified anycolor control for image generation and editing. arXiv preprint arXiv:2609.20816.

Jinheng Xie, Weijia Mao, Zechen Bai, David Junhao Zhang, Weihao Wang, Kevin Qinghong Lin, Yuchao Gu, Zhijie Chen, Zhenheng Yang, and Mike Zheng Shou. 2025a. Show-o: One single transformer to unify multimodal understanding and generation. In The Thirteenth International Conference on Learning Representations, ICLR 2025, Singapore, April 24-28, 2025. OpenReview.net.

Jinheng Xie, Zhenheng Yang, and Mike Zheng Shou. 2025b. Show-o2: Improved native unified multimodal models. In Advances in Neural Information Processing Systems 38: Annual Conference on Neural Information Processing Systems 2025, NeurIPS 2025, San Diego, CA, USA, December 2-7, 2025 / Mexico City, Mexico, November 30 - December 5, 2025.

Jinbo Xing, Zeyinzi Jiang, Yuxiang Tuo, Chaojie Mao, Xiaotang Gai, Xi Chen, Jingfeng Zhang, Yulin Pan, Zhen Han, Jie Xiao, Keyu Yan, Chenwei Xie, Chongyang Zhong, Kai Zhu, Tong Shen, Lianghua Huang, Yu Liu, and Yujiu Yang. 2026. Wan-weaver: Interleaved multi-modal generation via decoupled training. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2026, Denver, CO, USA, June 3-7, 2026, pages 36267–36278. Computer Vision Foundation.

Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Ehsan Azarnasab, Faisal Ahmed, Zicheng Liu, Ce Liu, Michael Zeng, and Lijuan Wang. 2023. MM-REACT: prompting chatgpt for multimodal reasoning and action. CoRR, abs/2303.11381.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik R. Narasimhan, and Yuan Cao. 2023. React: Synergizing reasoning and acting in language models. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net.

Dian Zheng, Harry Lee, Manyuan Zhang, Kaituo Feng, Zoey Guo, Ray Zhang, and Hongsheng Li. 2026. Interleavethinker: Reinforcing agentic interleaved generation. CoRR, abs/2606.13679.

Dewei Zhou, Xinyu Huang, Xun Wang, Ji Xie, Yabo Zhang, Liang Li, Kunchang Li, Zongxin Yang, and Yi Yang. 2026a. Metapoint: unlocking precise spatial control in agentic visual generation. In European Conference on Computer Vision, pages 373– 390. Springer.

Donghao Zhou, Haoyang He, Fan Zhang, Hao Yang, Guisheng Liu, Xin Gao, Zhongwei Wan, Xingyuan Bu, Jie Wang, Qiangpeng Yang, et al. 2026b. Thinkv2v: Unleashing the reasoning capability of mllms for instruction-guided video editing. arXiv preprint arXiv:2609.38541.

Donghao Zhou, Jiancheng Huang, Jinbin Bai, Jiaze Wang, Hao Chen, Guangyong Chen, Xiaowei Hu, and Pheng-Ann Heng. 2024. Magictailor: Component-controllable personalization in text-to-image diffusion models. arXiv preprint arXiv:2410.13370.

Donghao Zhou, Jingyu Lin, Guibao Shen, Quande Liu, Jialin Gao, Lihao Liu, Lan Du, Cunjian Chen, Chi-Wing Fu, Xiaowei Hu, et al. 2026c. Identitystory: Taming your identity-preserving generator for human-centric story generation. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, pages 13593–13601.

Donghao Zhou, Guisheng Liu, Hao Yang, Jiatong Li, Jingyu Lin, Xiaohu Huang, Yichen Liu, Xin Gao, Cunjian Chen, Shilei Wen, et al. 2026d. Omnishow: Unifying multimodal conditions for humanobject interaction video generation. arXiv preprint arXiv:2604.11804.

Pengfei Zhou, Xiaopeng Peng, Jiajun Song, Chuanhao Li, Zhaopan Xu, Yue Yang, Ziyao Guo, Hao Zhang, Yuqi Lin, Yefei He, Lirui Zhao, Shuo Liu, Tianhua Li, Yuxuan Xie, Xiaojun Chang, Yu Qiao, Wenqi Shao, and Kaipeng Zhang. 2025. Opening: A comprehensive benchmark for judging open-ended interleaved image-text generation. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2025, Nashville, TN, USA, June 11-15, 2025, pages 56–66. Computer Vision Foundation / IEEE.

## A Implementation and Method Details

UniWeaver executes four planning stages before rendering the final interleaved output. The executor first generates reference images, then renders planned unit images and pairs them with readerfacing text. Visual verification supports targeted regeneration of failed images, with at most two additional attempts per unit.

## A.1 Stages and Processes

The four planning stages execute sequentially, with earlier decisions remaining fixed. Each stage uses a predefined JSON schema, and subsequent stages operate on the preceding planning outputs rather than generated image pixels. All stage calls use a fixed system instruction requiring schemacompliant JSON output without additional commentary. The prompts below specify the behavior of the individual stages, with placeholders filled at inference time; inference appends a JSON-only closing instruction, while API-based construction sets JSON response mode.

Plan #1: Scope and Sequence. Decomposes the user request into content units, specifying the task endpoint and each unit’s visual subject and readerfacing content goal.

## Prompt. 1: Scope and Sequence

Task request: {task}

Use task details, ordinary knowledge, and plausible scene detail. Keep an unspecified subject at the category level instead of inventing a specific identity or exact event facts. Treat source-aware or source-backed as a request for careful, credible subject matter, not prose about sources or research work.

Partition the outcome into distinct useful image-plus-content units. For each unit separate:

• image\_subject: one concrete physical subject or state that a single independent image can directly depict in one coherent view;

• content\_goal: one short reader-facing question asking what useful explanation, interpretation, comparison, instruction, chronology, or decision the adjacent prose must answer. Ask about the

subject itself.

Normally create one unit for each distinct requested physical subject, showing a view that best supports all of its requested properties; put the properties and comparison criteria together in that unit’s content\_goal. Create another unit for the same subject only when the Task explicitly requires a real change of state, time, or location. A different evaluation criterion, viewpoint, tradeoff, comparison, recommendation, or prose section never creates a unit. Never create a summary, diagram, plan, split view, collage, or information-display image. Audience, later use, and presentation format do not add subjects.

Interpret the requested action literally. The endpoint is the earliest completed requested outcome or the last event explicitly stated in a narrative. For set up, arrange, or organize, operating or demonstrating the ready items is a later task. In a narrative, put the immediate reaction to the last event in that event’s content rather than adding an epilogue unit. When the Task’s main operation is comparing named alternatives, create exactly one unit for each alternative unless the Task explicitly requests a real before/after or other state change within an alternative. Each unit’s one view must show that alternative in the use context that exposes the most requested properties; its content\_goal carries all criteria, trade-offs, and the decision. Before returning, verify that unit count equals alternative count unless that exception applies. For comparison tasks, an image\_subject containing two or more named alternatives is invalid, and a step whose only purpose is to synthesize or conclude the comparison is invalid. Put synthesis and the decision in the content\_goal of the last alternative’s existing unit.

An interleaved report requires at least two distinct, useful image-plus-content units, so output at least two semantic steps. Every step must make a different task-relevant contribution. Do not add steps merely to satisfy the minimum.

Output exactly:

{"title":"...",

"endpoint":"earliest completed   
outcome or last explicit event",   
"steps":[{"step\_index":1,   
"image\_subject":"one concrete   
visible subject or state",   
"content\_goal":"short   
reader-facing question"}]}   
The top-level object has exactly these three   
keys. Every object inside steps has exactly   
three keys: step\_index, image\_subject,   
and content\_goal.  
Plan #2: Text and Image Planning. Generates the reader-facing prose and image-generation instruction separately for each unit, together with a shared visual style across the output.

## Prompt. 2: Text and Image Planning

Task request: {task}   
Write concise finished prose for the reader. Directly answer each content\_goal using task details, ordinary knowledge, and plausible scene detail. Keep an unspecified subject at the category level instead of inventing a specific identity or exact event facts. Discuss the subject, not the writing process. Give useful information directly; do not assign research to the reader.

Frozen scope and sequence: {steps\_summary}

Frozen endpoint: {endpoint}

Frozen step indexes: {step\_indexes}

Choose one shared visual\_style for rendering treatment only: medium, color treatment, lighting character, and rendering finish. Keep subjects, actions, locations, viewpoint, composition, and prose voice in the individual steps. Follow an explicitly requested style; otherwise use a restrained treatment appropriate to the subject.

Return one object for each frozen step index, in order. The complete task defines global scope and each frozen unit separates its visible subject from its prose contribution. Apply every relevant task constraint without changing the frozen sequence.

For each step output:

• the same step\_index; do not repeat Plan1 fields in the output;

• image\_prompt: one complete, selfcontained rendering instruction for the observable subject, state, relations, and context required by this contribution;

• content: finished report prose carrying useful explanation that need not be visible in pixels.

Name required visible elements concretely rather than replacing them with generic collective terms. Include only entities whose visible presence follows from the requested contribution. Keep one coherent physical view per image. Depict the frozen image\_subject rather than a container for presenting information about it, unless that container is itself the requested subject.

Do not turn a comparison or synthesis into a split composition, collage, multi-panel sequence, overlay, diagram, plan view, map, report page, label panel, or information display. The separate images provide the concrete subjects; content carries relations across them. Across the content fields, cover the complete task.

Output schema:

{"visual\_style":{"medium":"...",   
"color\_treatment":"...",   
"lighting\_character":"...",   
"rendering\_finish":"..."},   
"steps":{step\_semantic\_entries}}  
Plan #3: Selective Visual Conditioning. Constructs an optional bank of up to two reusable visual references based on the planned image prompts and assigns relevant references to individual units.

## Prompt. 3: Selective Visual Conditioning

Task: {task}

{semantic\_subjects}

Rendered Steps: {steps\_summary}

Select zero to two isolated identity references. Before writing JSON, silently enumerate every concrete person, object, vessel, tool, carrier, and support explicitly visible in each rendered Step, then group only occurrences established as the same physical individual. Do not group repeated members

Treat identity as established when an ordered construction or setup keeps an explicitly visible stable core in later Steps, even if later wording omits the phrase “the same”: a completed artifact retains its visibly described support frame, and a cumulative arrangement retains a named tool that is still explicitly shown. Do not infer identity from category repetition across independent scenes.

For each candidate, silently perform this counterfactual test: imagine supplying one isolated reference image unchanged to every proposed Step. Reject the candidate if any assigned Step needs the candidate itself to have a different outline, topology, fold or crease pattern, construction, surface state, or identity. Changes only to contents, attachments, pose, action, lighting, camera, or surroundings do not alter identity when the unchanged core remains visible. A container can therefore persist while its contents change, and a support frame can persist while panels are attached; a sheet being cut or folded cannot persist as one unchanged visual reference.

A stable identity shown in only one Step may optionally receive a reference when its specific identity matters. Every candidate shown in two or more Steps that passes the counterfactual test must enter the bank and must be routed to every Step where it is explicitly visible. If more than two pass, keep the two with greatest Step coverage, then greatest importance to the main artifact. Return an empty bank only when no candidate passes. Do not create references for generic category members, incidental props, scenes, backgrounds, landforms, layouts, spatial relations, routes, or information carriers. Output only the final schema; do not expose the test or reasoning. A reference prompt depicts only the invariant target in isolation, without a changing state, action, scene, or background.

Each item contains only fields that the image runtime consumes:

• key: stable identifier;

• type: reference;

```jsonl
• image_prompt: a self-contained image
of only the stable reusable target.
Every selected key must be referenced
by at least one step_conditions en
try. Include every step_N and use
[] when no reference applies. Each
step_conditions array contains only
key strings from condition_bank.items,
never image prompts or descriptions. When
items is empty, every step_conditions
array must be [].
Default output when there is no qualifying
cross-step identity reference:
{"condition_bank":{"items":[]},
"step_conditions":
{step_condition_entries}}
Only when a qualifying reference exists, an
item has exactly this shape:
{"key":"stable_identifier",
"type":"reference",
"image_prompt":"self-contained
image of only the stable reusable
target"}
```  
Plan #4: Visual Verification Checklist. Derives one or two directly observable visual checks per unit. These checks concern visible objects, actions, and states in the corresponding image.

## Prompt. 4: Visual Verification Checklist

Execution and Refinement. Reference images are generated before the unit images. Each unit is rendered from its image prompt and the shared visual style, together with any assigned reference images that the backbone accepts. The reviewer evaluates the generated image against its checklist and returns a PASS or FAIL result for each criterion. Failed criteria guide regeneration of the corresponding image, while successful units remain unchanged. The final output pairs the selected images with the prose generated in Plan 2.

## A.2 Stage Schemas and Runtime Settings

Table 3 summarizes the inputs and outputs of each stage. Earlier planning outputs remain fixed, allowing each stage to be validated independently.

Table 3: UniWeaver organizes multimodal generation through four structured planning stages and visual verification. The table summarizes the inputs and outputs of each stage and the image reviewer.
<table><tr><td>Stage</td><td>Input</td><td>Output</td></tr><tr><td> $P _ { 1 }$ </td><td>User request</td><td>Title, task endpoint, and unit-level visual sub- jects and content goals</td></tr><tr><td> $P _ { 2 }$ </td><td>Request and  $P _ { 1 }$ </td><td>Shared visual style, reader-facingprose, and image prompts</td></tr><tr><td> $P _ { 3 }$ </td><td>Request and planned image prompts</td><td>Reference bank (0-2 images) and unit- specific reference</td></tr><tr><td> $P _ { 4 }$ </td><td>Planned image prompts</td><td>assignments Observable visual checks for each unit</td></tr><tr><td>Review</td><td>Generated image and checklist</td><td>Per-criterion PASS/FAIL with feedback</td></tr></table>

Each report uses at most two reference images. For backbones supporting only one conditioning image, only the first routed reference is supplied. All planning stages use greedy decoding at temperature 0. The maximum output lengths for P1–P4 are 2,048, 2,560, 768, and 1,024 tokens, respectively. The reviewer returns one judgment per checklist item. Training trajectories must satisfy the prescribed schemas and task-consistency checks.

## B Benchmark Construction

## B.1 Design and Composition

UniWeaver-Bench contains 120 goal-only tasks spanning 20 task families across four categories, with six held-out tasks per family and a balanced difficulty distribution of 40 easy, 40 medium, and 40 hard tasks. The task families follow the taxonomy of UniWeaver-10K, as illustrated in Figure 4(a), while benchmark tasks are held out from training. The four categories group the 20 families as follows:

• Procedures & Instructions: Recipe & Food Preparation, Craft & DIY, Assembly & Physical Setup, Care & Maintenance, and Before-and-After Transformation.

• Comparison & Recommendation: Product / Object Comparison, Place / Experience Comparison, Product Use & Lifestyle Guide, Fashion & Styling, and Interior & Lifestyle Inspiration.

• Observation & Description: Photography / Visual Showcase, Nature / Species / Environment, Architecture & Design Style, Seasonal / Atmosphere / Environmental Change, and Travel & Place Guide.

• Explanation & Storytelling: Collection / Visual Catalog Report, Object History & Evolution, Culture & Tradition, Event / Experience Story, and Narrative / Slice of Life.

The benchmark therefore evaluates generalization to unseen requests within familiar task families. Final scores are aggregated with equal weight across the 20 families, following Section 3.3.

## B.2 Evaluator-Only Fields and Separation

Each benchmark task includes three evaluator-only fields: task family, difficulty and hidden evaluation reference. The evaluated method receives only the task identifier and user request, while evaluation metadata and hidden references remain inaccessible during generation. The hidden references specify task-specific content requirements and observable visual evidence.

To prevent training–evaluation overlap, we compare UniWeaver-10K against the frozen 120-task benchmark using task identity and normalized prompt hashes. No exact matches were detected among the 10,000 retained training tasks. This filtering prevents identified task and prompt duplication.

We conducted a content-screening pass over all 120 benchmark tasks, including both the user-facing prompts and their hidden evaluationreference fields. The procedure applied deterministic keyword and pattern filters to identify objectionable content. Neither offensive content nor personally identifying information was detected.

## C Evaluation Protocol

UniWeaver-Bench evaluates the final reader-facing text and generated images, rather than internal planning trajectories. A deterministic validity check requires a nonempty sequence of ordered text-image units, each containing reader-facing content paired with an existing, non-reused image. The check operates only on the final output. Invalid reports do not enter $\bar { Q } _ { f }$ and are penalized through ${ \sqrt { r _ { f } } } .$

Valid reports are evaluated using a task-specific rubric across four dimensions: four to five for text coverage (TC), two for writing quality (WQ), two to three for visual coverage (VC), and one per textimage unit for image-text alignment (ITA). Each criterion is rated as not met, partially met, or met, corresponding to 0, 0.5, and 1, respectively. Table 4 summarizes the evidence used for each dimension.

Each method is run once on all 120 tasks using a fixed base generation seed of 42. The two repeated judge calls per criterion are used only for withinjudge aggregation.

Table 4: UniWeaver-Bench evaluates final outputs using a task-specific rubric across four dimensions. The table specifies the number of criteria and evaluation evidence for each dimension.
<table><tr><td>Dimension</td><td># Criteria</td><td>Evaluation Evidence</td></tr><tr><td>TC</td><td>4-5</td><td>User request, hidden coverage requirements, and final text</td></tr><tr><td>WQ</td><td>2</td><td>User request and final text</td></tr><tr><td>VC</td><td>2-3</td><td>User request, hidden visual re- quirements, and generated im-</td></tr><tr><td>ITA</td><td>one per unit</td><td>ages User request, final text, and cor- responding images</td></tr></table>

## C.1 Judging and Aggregation

We independently evaluate valid reports using three judges: Kimi K3 (Kimi Team, 2026), DeepSeek-V4.1-Flash (DeepSeek-AI, 2026b), and GLM-5.3- Flash (GLM, 2026). Each judge evaluates one task at a time at temperature 0, with two repeated calls per criterion. The median rating is used as the within-judge result.

Cross-judge aggregation is conservative. When the three medians agree, that rating is used. A onetier disagreement is resolved by taking the lower rating, and a disagreement spanning more than one tier is scored as 0, the lowest tier.

Text and visual evaluations use separate fixed system instructions. Judges receive only the task-specific materials and evaluation references. Each judgment returns structured JSON containing criterion-level ratings and brief supporting observations. Visual evaluations rely on directly observable image evidence, and rendered text is not treated as reliable visual evidence unless it is itself the evaluation target.

Panel ratings are normalized to quarter steps between 0 and 1, where a criterion whose three medians span more than one tier scores 0. Task quality is calculated using the weighted criteria defined in Section 3.3. Within each task family, $Q _ { t }$ is averaged over gate-passing tasks and multiplied by the square root of the gate pass rate. Final scores are obtained by equally averaging the 20 family scores and are reported on a 0–100 scale.

## D Training and Model Integration

We apply LoRA to BAGEL’s understanding decoder (14.22B total parameters and 0.08B trainable parameters) while keeping the image-generation experts and vision encoder frozen. UniWeaver-10K contains 10,000 trajectories, each expanded into four planning-stage training examples. With two examples packed per optimization step, one epoch uses 20,000 steps; we train for two epochs, for 40,000 steps in total. The run uses one NVIDIA GB10 GPU, processes 101.3M tokens in about 35 GPU-hours, and peaks at 79.5 GiB of GPU memory. These timings exclude judge charges. Table 5 summarizes the optimization settings. All reported UniWeaver-SFT results use the final checkpoint.

Table 5: UniWeaver-SFT adapts BAGEL’s understanding decoder using LoRA. The table summarizes the key training and optimization settings.
<table><tr><td>Setting</td><td>Value</td></tr><tr><td>LoRA (rank / alpha / dropout) Optimizer</td><td>32 / 64 / 0.05 AdamW, β = (0.9, 0.95),  $\epsilon \ : = \ : 1 0 ^ { - 8 }$  , weight decay</td></tr><tr><td>Learning rate</td><td> $0 . 0 1$   $5 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>Schedule</td><td>200-step warm-up, cosine</td></tr><tr><td>Training steps</td><td>decay 40,000</td></tr><tr><td>Gradient clipping / seed</td><td>1.0 / 42</td></tr><tr><td>Max. sequence length</td><td>4,096 tokens</td></tr></table>

## E Details of Ablation Studies

Table 6 specifies the implementation of the seven variants in Table 2. Each variant modifies only the indicated component, preserves the original planning-stage order, and keeps every remaining stage fixed. In particular, “References Fixed before the Text” changes only which upstream fields are visible when $P _ { 3 }$ selects references; it retains the stage’s timing and output interface. This isolates the effect of information order from changes in reference availability or per-step routing.

![](images/b59152a5f645c7e555765ba6cd4a71dda17f5f0dfe332ab644fb3663a1e6bd70.jpg)  
Figure 6: Additional qualitative comparison on the architectural design task. UniWeaver produces coherent, task-specific descriptions and corresponding visuals covering the entrance, rehearsal room, rooftop garden, and their integration, while baseline UMMs exhibit repetitive content or incomplete task coverage.

Table 6: Seven ablation variants isolate individual components of UniWeaver’s planning workflow. The table specifies the modifications to P<sub>1</sub>, P<sub>2</sub>, and P<sub>3</sub>.
<table><tr><td>Ablation</td><td>Modification</td></tr><tr><td>P1: Task Scope</td><td></td></tr><tr><td>w/o Task Goal</td><td>Removes the endpoint field from P1</td></tr><tr><td>w/o Per-step Content Goal Removes unit-level content</td><td>goals while retaining visual subjects</td></tr><tr><td colspan="2">P2: Text and Image Planning</td></tr><tr><td>w/o Shared Visual Style</td><td>Removes the shared visual style from P2</td></tr><tr><td>w/ Merged Image Prompt Uses one field for both and Content</td><td>reader-facing prose and image-generation instructions</td></tr><tr><td colspan="2">P3: Visual References</td></tr><tr><td>References Fixed before the Text</td><td>Selects references using only P1, without access to P2 outputs</td></tr><tr><td>w/o Per-step Reference Routing w/o All Reference Images Disables reference image</td><td>Retains the reference bank but removes unit-specific routing generation and conditioning</td></tr></table>

All seven variants start from the same base BAGEL model and use identical evaluation settings; none of them loads the SFT checkpoint. They also share the same benchmark tasks, non-ablated prompts, decoding configuration, image-generation settings, and judge protocol.

## F Additional Qualitative Results

Figure 6 presents an architectural design task that requires integrating a street entrance, rehearsal room, and rooftop garden into a coherent illustrated brief. Unlike the instructional tasks in the main text, this case examines how models organize multiple functional spaces and explain their relationships.

BAGEL repeatedly generates similar exterior views, TokenFlow produces repetitive planningoriented content, and ILLUME+ exposes intermediate generation instructions instead of complete reader-facing descriptions. In contrast, UniWeaver provides distinct, task-relevant text-image units for each space and concludes by explaining their integration into a unified design.

The architectural brief is a useful stress test because each functional space must remain individually identifiable while also contributing to one coherent design. Methods that generate units independently can produce plausible local views without preserving their cross-space relationships, whereas methods with stronger sequential dependence can collapse multiple spaces into near-duplicate views. The example therefore exposes the balance Uni-Weaver targets: sufficient independence to cover distinct content units, together with enough shared conditioning to preserve a coherent overall result.