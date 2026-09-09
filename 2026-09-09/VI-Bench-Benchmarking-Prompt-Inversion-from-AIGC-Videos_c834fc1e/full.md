# VI-Bench: Benchmarking Prompt Inversion from AIGC Videos

Wulin Xie<sup>1,2</sup> Rui Zhao<sup>3</sup> Kecen Li<sup>4</sup> Xiujin Liu<sup>5</sup> Bokang Zhang<sup>6</sup> Zheng Liu<sup>3</sup> Xinwen Hou<sup>1,2</sup> Chen Gong<sup>3∗</sup>

<sup>1</sup>Institute of Automation, Chinese Academy of Sciences <sup>2</sup>University of Chinese Academy of Sciences <sup>3</sup>University of Virginia <sup>4</sup>National University of Singapore <sup>5</sup>University of Michigan, Ann Arbor <sup>6</sup>The Chinese University of Hong Kong, Shenzhen

## Abstract

Recent advances in video generation have made prompt-based control increasingly central to AIGC video generation. Prompts specify what a video should depict and how it should be represented, controlling factors such as visual style or camera behavior. Understanding this recoverability is important both for creative reuse and editing, and for assessing prompt leakage risks. However, existing video understanding benchmarks do not measure this capability: a caption may describe what is visible, but a replayable prompt must recover the generation-relevant controls needed to reproduce the video. To address this gap, we introduce VI Bench, a benchmark built from 16.1 million real-user prompts and 900 humanverified AIGC videos. VI-Bench spans three progressively harder settings, namely single-shot semantic grounding, control over style and camera behavior, and multishot compositional inversion, and evaluates five generation-critical dimensions: subject, action, scene, style, and camera. We evaluate 18 representative VLMs, including 2 proprietary and 16 open-source models on VI-Bench, using an Inversion Score that measures prompt-level alignment with the original prompt and videolevel fidelity of the regenerated video. The results reveal substantial limitations: even the strongest model achieves only 0.632 on Inversion Score, performance degrades sharply as samples require richer control and multi-shot reasoning, and models often produce plausible prompts whose regenerated videos deviate from the reference. These findings show that video prompt inversion is a distinct and under-evaluated capability requiring models to transform visual understanding into replay-stable generative control.

## 1 Introduction

Recent advances in video generation have enabled the synthesis of coherent and high-quality AIGC videos [1–5], making prompt-based control increasingly central to controllable video generation. Prompts specify not only what a video should depict, but also how it should be presented, controlling factors such as visual style, camera behavior, and temporal composition [6, 7]. In real-world creative platforms, users may encounter a compelling AI-generated video and wish to reproduce its visual style, camera motion, or temporal structure for editing or creative reuse, while the original prompt remains unavailable. The emergence of prompt marketplaces such as PromptBase [8] and PromptAi

![](images/ab193c0f7e677ed9af9f83e36b2d35448b900000cc727c5f4f2756246d220ece.jpg)

![](images/cc52e9d5da3f77006b90f62aba15a5fc42b682f2b13d82c223d6520e98078fb9.jpg)

![](images/e5e32b99efa94d2783f7f2f3c14f365db029c601e8a2a70c1890a9b284e066fa.jpg)

(a) Top Keywords per Evaluation Dimension  
![](images/61c56c908d1730cf48e22b7ee9245d45ad25198dd5be72a3103d8f00333c843f.jpg)

![](images/eb9b87a4ab8759c0b684a130bb1b4a491af92214f147afe44316634a5092c201.jpg)

![](images/d3ea069e188977ff45b729a6e48c0bd63f74475d71aa46ed51b86031babcd8cb.jpg)  
(b) Topic Taxonomy

![](images/01c7c05a645510cb7f4012664239d39f85aa6f2b5f251e3fc50e228aa5f302d1.jpg)

(c) Prompt Length Distribution  
![](images/7d24e8e984e8e8016ecffd14ed36db6aad1d75530cd3489e0065b584337edcbb.jpg)  
(d) GT Prompt Word Cloud  
Figure 1: Statistical distributions of VI-Bench.

Market [9], where users can buy and sell prompts for AIGC video generation, further shows that prompts have become valuable creative assets rather than disposable text. At the same time, generated videos may unintentionally reveal information about their underlying prompts, raising practical concerns about prompt leakage, proprietary prompt templates, and the protection of generation workflows. Understanding how recoverable such prompt-level controls are is therefore important no only for creative reuse, but also for assessing prompt leakage risks and designing defenses against prompt leakage. This dual role of video prompt recovery raises a fundamental question: can we measure whether a generated video exposes a replayable prompt, namely a prompt that can be executed by a video generator to reproduce the reference video?

Prior efforts have explored prompt recovery from generated content through prompt stealing, template extraction, and optimization-based inversion [10–13]. However, these studies focus on designing specific recovery algorithms or analyzing attack cases in text-to-image settings, rather than providing a systematic benchmark for measuring how much prompt-level control information can be recovered from generated videos. More importantly, image-centric inversion methods do not naturally transfer to the video domain. Unlike images, videos require recovering generative factors that evolve over time, including temporal progression, motion continuity, stylistic consistency, camera dynamics, and multi-shot narrative structure [14, 15]. Although Vision-Language Models (VLMs) [16–19] can be used for video prompt inversion in practice, this emerging use case remains poorly defined and lacks dedicated evaluation. This raises a key question: how can we systematically evaluate the ability of VLMs to infer replayable generation controls from AIGC videos, and where do current models fail?

While recent VLMs have achieved strong performance on video understanding and captioning [20– 22], these tasks primarily assess descriptive understanding, namely whether a model can recognize, interpret, or describe what is visible in a video. In other words, a video caption answers what is visible to a human observer, whereas an inversion prompt must specify what a generator should execute to reproduce the video. Thus, video prompt inversion requires models to identify not only visible semantics, but also generation-relevant controls such as subject identity, action dynamics, scene layout, visual style, camera behavior, and temporal structure. This distinction also changes how the task should be evaluated. Because valid prompts may differ in wording while remaining equally effective for reproduction, text similarity alone cannot serve as a reliable metric [7, 6]. Conversely, two textually similar prompts may produce noticeably different videos once executed by a generator. As a result, a proper evaluation should go beyond text comparison and test whether the inferred prompt can actually reproduce the reference video under replay.

To address this gap, we introduce VI-Bench, a dedicated benchmark for evaluating whether VLMs can recover replayable prompts from AIGC videos. VI-Bench is built from 16.1M real-user prompts [23– 25], which are cleaned into approximately 3.9M high-quality prompts and organized into topic pools for benchmark construction. The final benchmark contains 900 human-verified AIGC videos generated by two video generators and organized into three progressively harder settings: single-shot semantic grounding, control over style and camera behavior, and multi-shot compositional inversion. Each sample is evaluated along five generation-critical dimensions: Subject, Action, Scene, Style, and Camera. Beyond dataset construction, VI-Bench evaluates whether a recovered prompt is not only aligned with the original prompt, but also effective when executed by the generator to reproduce the reference video. In this way, VI-Bench aims to measure prompt recoverability and replay-oriented generative control, rather than descriptive video understanding alone. We conduct an extensive evaluation of 18 representative VLMs, including 2 proprietary and 16 open-source models on VI-Bench. Using the Inversion Score that jointly measures prompt-level alignment with the original prompt and video-level fidelity after replay, our benchmark analysis leads to the following findings:

• Current VLMs remain far from solving the video prompt inversion task. Even the strongest model achieves only 0.632 overall Inversion Score on a normalized [0, 1] scale, showing that current models still fail to reliably recover prompts that are both faithful to the original prompt and effective under replay. Moreover, the performance of most models drops markedly as samples move from single-shot to multi-shot videos.

• Video prompt inversion is not equivalent to video understanding or captioning. Strong performance on existing video understanding benchmarks does not necessarily translate into strong video prompt inversion, suggesting that recoverability is distinct from descriptive video understanding. Moreover, models often produce prompts that appear semantically plausible, but the videos regenerated from these prompts still deviate substantially from the references.

• The main failures are factor-dependent and become more noticeable in multi-shot videos. Subject and Style exhibit the largest prompt-to-replay gaps, suggesting that models may recognize what should be recovered but fail to express it in a replay-stable form. Multi-shot videos further expose a major capability boundary, where models must aggregate information across shots while preserving temporally consistent generative controls.

## 2 Related Work

Image Prompt Inversion. Image prompt inversion aims to recover a text prompt from a reference image such that the prompt can reproduce similar content and style [12, 11, 10, 13]. Existing studies mainly focus on text-to-image generation. VGD [13] studies the recovery of readable prompts from generated images by combining language-model-based prompt generation with visual feedback from CLIP [26]. ARPO [10] formulates reverse prompt engineering as an iterative optimization process that refines prompts through repeated image generation and comparison with the reference image. Security-oriented studies further examine prompt stealing risks in text-to-image systems, with PromptStealer [12] investigating whether key prompt components, such as subjects and modifiers, can be inferred from generated images, while EvoStealer [11] further studies the stealing of reusable prompt templates from multiple generated images.

In contrast, this paper studies video prompt inversion. Rather than designing an attack against a specific image generator, we evaluate whether current VLMs can infer replayable generation controls from AIGC videos. This setting is substantially more challenging than image prompt inversion, since videos require recovering temporally structured factors such as motion continuity, camera dynamics, style consistency, and multi-shot composition.

Benchmarking Vision-Language Models. Recent benchmarks evaluate Vision-Language Models from a broad range of perspectives, including general multimodal perception and reasoning [27, 28], video understanding and temporal reasoning [20, 21, 29–31], and video captioning for controllable text-to-video generation [22]. These benchmarks have substantially advanced the evaluation of VLMs by measuring whether models can recognize visual content, reason about temporal events, answer video questions, or produce descriptive captions. However, existing benchmarks mainly evaluate descriptive video understanding, i.e., whether models can answer questions or describe visible content. In contrast, VI-Bench evaluates whether a model can recover a replayable prompt from an AIGC video. The recovered prompt is executed by the original generator, and evaluation jointly measures prompt fidelity and replay fidelity across Subject, Action, Scene, Style, and Camera.

![](images/0767f612295a5184273b4ee62caa8408d4484d177545fa4bfb0c5461d3e90cfb.jpg)  
Figure 2: Overview of the VI-Bench pipeline. Real-user prompts are clustered into topic pools to synthesize Easy, Medium, and Hard prompts, which are rendered into ground-truth videos. VLMs recover prompts from these videos, and the recovered prompts are replayed and evaluated across five dimensions to compute the final Inversion Score.

## 3 VI-Bench

## 3.1 Task Formulation

We formulate video prompt inversion as the task of recovering a replayable prompt from a reference AIGC video. Unlike video captioning, the output is not a free-form description of visible content, but a generator-ready prompt that should preserve the controls needed for regeneration. Each sample consists of a ground-truth video $V _ { \mathrm { g t } }$ , its original prompt $P _ { \mathrm { g t } }$ , and the corresponding video generator G. Given only $V _ { \mathrm { g t } } .$ , an inversion model M predicts an inversion prompt, $\bar { P _ { \mathrm { i n v } } } \overset { \cdot } { = } M \bar { ( V _ { \mathrm { g t } } ) }$ . The predicted prompt is then fed back to the original generator to produce an inversion video, $\mathrm { \bar { \cal V } _ { i n v } } = G ( \cal P _ { i n v } )$ This replay step is central to our formulation: it tests whether the recovered prompt is not only semantically plausible, but also executable by the generator. We evaluate inversion quality from two perspectives: (1) prompt fidelity, measured by the similarity between $P _ { \mathrm { i n v } }$ and $P _ { \mathrm { g t } }$ , and (2) replay fidelity, measured by the similarity between $\dot { V _ { \mathrm { i n v } } }$ and $V _ { \mathrm { g t } }$ . Both perspectives are necessary. Prompt fidelity checks whether the model recovers the original generation intent and prompt-level controls, while replay fidelity checks whether these controls actually work when executed by the generator. Using replay fidelity alone may overestimate inversion quality, since generator priors or randomness can produce a visually similar video even when the inferred prompt misses key original controls. Using prompt fidelity alone is also insufficient, since a semantically plausible prompt may still fail to reproduce the reference video.

## 3.2 Data Construction Process

We design a systematic data construction pipeline, as illustrated in Figure 2. The goal of this pipeline is to preserve the diversity of real user generation intents while enabling controlled construction over topic composition, difficulty level, and generator source. The process consists of four key stages: (1) Prompt Collection & Cleaning, (2) Topic Pool Construction, (3) Difficulty-aware Prompt and Video Synthesis, and (4) Human Verification. Prompt collection anchors the benchmark in real-world prompt distributions; topic construction organizes the cleaned prompts into topic pools for sampling; difficulty-aware synthesis combines these factors into progressively harder videos; and human verification ensures that the final videos faithfully reflect their prompts.

Prompt Collection & Cleaning. We collect real-user prompts from three public datasets: DiffusionDB [23] (6M), VidProM [24] (5M), and TIP-I2V [25] (5.1M), yielding 16.1M raw prompts in total. These datasets cover a broad range of public user generation scenarios, including text-to-image, text-to-video, and image-to-video prompting, providing a diverse source of real generation intents. We then filter platform prefixes, generation parameters, negative prompts, noisy expressions, non-English content, and NSFW prompts, while preserving the underlying generation intent. This process yields approximately 3.9M cleaned prompts, corresponding to a retention rate of 24.2%, which are used for subsequent topic construction.

Topic Pool Construction. To obtain diverse prompts for benchmark construction, we avoid directly sampling prompts at random, since random sampling would make it difficult to balance semantic coverage and difficulty level. Instead, we organize the cleaned prompts into topic pools. Specifically, we first encode the cleaned prompts using Qwen3-Embedding-4B [32] and then perform topic discovery with BERTopic [33]. We next use GPT-4o [34] to assign each discovered topic to one of six categories: Character, Event, Style, Environment, Camera, or Untagged. These categories are chosen to reflect common prompt-level controls in video generation: Character, Event, and Environment describe core semantic content, while Style and Camera capture generator-sensitive appearance and cinematographic controls. The Untagged category is used to filter topics that are ambiguous, overly noisy, or unsuitable for controlled benchmark construction. Only the first five categories are retained as benchmark topic pools. Finally, we apply CLIP-based deduplication with a cosine similarity threshold of 0.9 to remove semantically overlapping topics, retaining the shorter topic name when duplicates are detected. This process generates approximately 800 topics in each pool, which serve as the basis for subsequent difficulty-aware prompt synthesis.

Difficulty-Aware Prompt and Video Synthesis. Based on the topic pools, we synthesize benchmark samples with controlled difficulty and generator diversity. To reduce stylistic bias from any single language model, we uniformly sample from GPT-4o [34], Claude Sonnet 4.5 [35], and Gemini 2.5 Flash [36] to generate groundtruth prompts conditioned on sampled topics. We organize VI-Bench into three difficulty levels that place progressively stronger demands on prompt inversion. The difficulty design follows the intuition that video prompt inversion becomes harder as the prompt contains more generation-critical factors and longer temporal dependencies. We summarize the three levels in Table 1. Easy samples are built from Character, Event, and Envi-

Table 1: Difficulty design of VI-Bench. Each level progressively introduces additional generation-critical factors.
<table><tr><td>Factor</td><td>Easy</td><td>Medium</td><td>Hard</td></tr><tr><td>Subject</td><td>√</td><td>√</td><td>√</td></tr><tr><td>Action</td><td>√</td><td>√</td><td>√</td></tr><tr><td>Scene</td><td>√</td><td>√</td><td>√</td></tr><tr><td>Style</td><td>X</td><td>√</td><td>√</td></tr><tr><td>Camera</td><td>x</td><td>√</td><td>√</td></tr><tr><td>Multi-shot</td><td>x</td><td>x</td><td>√</td></tr></table>

ronment topics, primarily testing semantic grounding. Medium samples additionally introduce Style and Camera topics, requiring recovery of generator-sensitive control factors beyond visible content description. This level explicitly separates prompt inversion from captioning: a caption may correctly describe the subject, action, and scene, but still omit style words, shot scale, viewpoint, or camera motion that are essential for reproducing the video. Hard samples further extend Medium to coherent multi-shot narratives, where multiple shot-level prompts must be recovered under temporal continuity and cross-shot compositional constraints. We generate the resulting videos using two video generators, Wan2.2 [37] and HunyuanVideo 1.5 [38], under fixed settings. Using two generators reduces the dependence of VI-Bench on a single generation pipeline. Each prompt produces one video, while Hard samples are generated shot by shot and concatenated into a final multi-shot sequence.

Human Verification. This verification step is necessary because even high-quality video generators may omit prompt factors. Each sample is assessed independently by two annotators. Annotators check whether the generated video matches the ground-truth prompt and whether any salient subject, action, scene, style, or camera factor is missing or inconsistent. Samples marked as misaligned are regenerated by varying the seed or revising the prompt, and are then re-evaluated under the same protocol. After verification, we retain 900 benchmark samples in total, with 300 samples at each difficulty level. Easy and Medium consist of single-shot videos, while Hard contains multi-shot videos with 2, 3, or 4 shots. For Hard samples, each shot is generated from a shot-level prompt, and the generated shots are concatenated in temporal order to form one final multi-shot video.

## 3.3 Evaluation

We evaluate video prompt inversion from two complementary perspectives: promptfidelity and replay fidelity. This protocol is designed to measure reverse-prompting ability rather than general video understanding: the output is judged by whether it preserves the original generation intent and can be replayed by the generator. Given a reference video $V _ { \mathrm { g t } }$ , a VLM predicts a prompt $P _ { \mathrm { i n v } } ,$ which is replayed by the same generator under the original generation settings to generate an inversion video $V _ { \mathrm { i n v } }$ . For Hard-level samples, $P _ { \mathrm { i n v } }$ contains multiple shot-level prompts. Each shot-level prompt is replayed separately using the original generator and settings, and the generated shots are concatenated in temporal order to form the full inversion video $V _ { \mathrm { i n v } }$ . VI-Bench evaluates two aspects of inversion quality: whether the inferred prompt is semantically faithful to the original generation intent, and whether it remains effective for reproducing the reference video when executed by the generator.

![](images/62cdeffe9f719b1da09b8760555e82241c62a8a42d4424fc05927d09e6839015.jpg)  
Figure 3: An overview of the video score evaluation pipeline.

Prompt-level evaluation. To assess prompt fidelity, we compare the inferred prompt $P _ { \mathrm { i n v } }$ with the ground-truth prompt $P _ { \mathrm { g t } }$ using GPT-4o as the judge. The two prompts are evaluated on five dimensions: $D = \{ S u \bar { b } j e c t ,$ Action, Scene, Style, Camera}, and Camera. For each dimension, the judge assigns a raw score $\hat { s } _ { d } ^ { p } \in [ 1 , 5 ]$ ], which is normalized to $s _ { d } ^ { p } = ( \hat { s } _ { d } ^ { p } - 1 ) / 4$ . The final Prompt Score is computed as:

$$
S _ { \mathrm { p r o m p t } } = \frac { 1 } { | D | } \sum _ { d \in D } s _ { d } ^ { p } ,\tag{1}
$$

where $s _ { d } ^ { p }$ denotes the prompt-level score on dimension d.

Video-level evaluation. To assess replay fidelity, we compare the replay video $V _ { \mathrm { i n v } }$ with the reference video $V _ { \mathrm { g t } }$ using a two-agent evaluation framework. As shown in Figure 3, a Memory Agent first reads each video independently and summarizes it along the same difficulty-specific dimensions $D$ A Judge Agent then compares the two video memories and assigns a raw score $\hat { s } _ { d } ^ { v } \in [ 1 , 5 ]$ for each dimension, which is normalized to $s _ { d } ^ { v } = ( \hat { s } _ { d } ^ { v } - 1 ) / 4$ . The final Video Score is computed as:

$$
S _ { \mathrm { v i d e o } } = { \frac { 1 } { | D | } } \sum _ { d \in D } s _ { d } ^ { v } ,\tag{2}
$$

where $s _ { d } ^ { v }$ denotes the normalized video-level score on dimension d.

Finally, we define the overall Inversion Score as $S _ { \mathrm { i n v } } = { \textstyle \frac { 1 } { 2 } } ( S _ { \mathrm { p r o m p t } } + S _ { \mathrm { v i d e o } } )$ , which measures whether a model can recover prompts that are both faithful and replay-effective.

## 4 Experiment

## 4.1 Experimental Setup

We evaluate 18 representative VLMs on VI-Bench, including 2 proprietary VLMs, Doubao-Seed-2.0-pro [39] and GPT-4o [34], and 16 open-source VLMs, including OmniVinci [40], Qwen2.5-VL series [16], Qwen3-VL series [17], Qwen3.5 [41], VideoLLaMA3 [42], Keye-VL [19], InternVL2.5

Table 2: Main results on VI-Bench. We evaluate 18 VLMs on the VI-Bench across three difficulty levels. Prompt Score measures prompt fidelity; Video Score measures replay fidelity; Inversion Score is the average of the two, reflecting a model’s overall ability to recover replayable prompts from AIGC videos. Bold indicates the best result in each column; underline indicates the second-best.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Release</td><td rowspan="2">#Params</td><td colspan="3">Easy</td><td colspan="3">Medium</td><td colspan="3">Hard</td><td colspan="3">Overall</td></tr><tr><td>Prompt</td><td>Video</td><td>Inv.</td><td>Prompt</td><td>Video</td><td>Inv.</td><td>Prompt</td><td>Video</td><td>Inv.</td><td>Prompt</td><td>Video</td><td>Inv.</td></tr><tr><td colspan="2">Proprietary MLLMs</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Doubao-Seed-2.0-pro</td><td>2025-10</td><td></td><td>0.747</td><td>0.755</td><td>0.751</td><td>0.649</td><td>0.584</td><td>0.617</td><td>0.627</td><td>0.432</td><td>0.529</td><td>0.674</td><td>0.591</td><td>0.632</td></tr><tr><td>GPT-40</td><td>2024-08</td><td></td><td>0.742</td><td>0.721</td><td>0.731</td><td>0.611</td><td>0.528</td><td>0.570</td><td>0.551</td><td>0.331</td><td>0.441</td><td>0.635</td><td>0.527</td><td>0.581</td></tr><tr><td colspan="3">Open-source MLLMs</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>OmniVinci</td><td>2025-10</td><td>7B</td><td>0.731</td><td>0.713</td><td>0.722</td><td>0.587</td><td>0.525</td><td>0.556</td><td>0.582</td><td>0.318</td><td>0.450</td><td>0.633</td><td>0.519</td><td>0.576</td></tr><tr><td>Qwen2.5-VL-72B</td><td>2025-01</td><td>72B</td><td>0.721</td><td>0.726</td><td>0.723</td><td>0.581</td><td>0.533</td><td>0.557</td><td>0.539</td><td>0.313</td><td>0.426</td><td>0.614</td><td>0.524</td><td>0.569</td></tr><tr><td>Qwen3-VL-8B</td><td>2025-09</td><td>8B</td><td>0.720</td><td>0.746</td><td>0.733</td><td>0.579</td><td>0.564</td><td>0.571</td><td>0.477</td><td>0.255</td><td>0.366</td><td>0.592</td><td>0.522</td><td>0.557</td></tr><tr><td>Qwen3-VL-30B</td><td>2025-09</td><td>30B</td><td>0.712</td><td>0.729</td><td>0.720</td><td>0.579</td><td>0.549</td><td>0.564</td><td>0.472</td><td>0.258</td><td>0.365</td><td>0.588</td><td>0.512</td><td>0.550</td></tr><tr><td>LLaVA-Video</td><td>2024-10</td><td>7B</td><td>0.722</td><td>0.702</td><td>0.712</td><td>0.561</td><td>0.484</td><td>0.523</td><td>0.524</td><td>0.295</td><td>0.409</td><td>0.602</td><td>0.494</td><td>0.548</td></tr><tr><td>Qwen3-VL-4B</td><td>2025-09</td><td>4B</td><td>0.715</td><td>0.720</td><td>0.718</td><td>0.557</td><td>0.553</td><td>0.555</td><td>0.468</td><td>0.250</td><td>0.359</td><td>0.580</td><td>0.508</td><td>0.544</td></tr><tr><td>Qwen2.5-VL-32B</td><td>2025-03</td><td>32B</td><td>0.718</td><td>0.711</td><td>0.715</td><td>0.577</td><td>0.536</td><td>0.556</td><td>0.453</td><td>0.251</td><td>0.352</td><td>0.583</td><td>0.499</td><td>0.541</td></tr><tr><td>Qwen2.5-VL-7B</td><td>2025-01</td><td>7B</td><td>0.717</td><td>0.676</td><td>0.697</td><td>0.525</td><td>0.470</td><td>0.497</td><td>0.500</td><td>0.259</td><td>0.379</td><td>0.581</td><td>0.468</td><td>0.524</td></tr><tr><td>Qwen3.5</td><td>2025-11</td><td>9B</td><td>0.720</td><td>0.706</td><td>0.713</td><td>0.595</td><td>0.503</td><td>0.549</td><td>0.408</td><td>0.184</td><td>0.296</td><td>0.574</td><td>0.464</td><td>0.519</td></tr><tr><td>VideoLLaMA3</td><td>2025-01</td><td>7B</td><td>0.686</td><td>0.655</td><td>0.670</td><td>0.480</td><td>0.443</td><td>0.462</td><td>0.483</td><td>0.241</td><td>0.362</td><td>0.550</td><td>0.446</td><td>0.498</td></tr><tr><td>Keye-VL</td><td>2025-10</td><td>8B</td><td>0.712</td><td>0.697</td><td>0.704</td><td>0.568</td><td>0.490</td><td>0.529</td><td>0.331</td><td>0.102</td><td>0.216</td><td>0.537</td><td>0.429</td><td>0.483</td></tr><tr><td>Qwen2.5-VL-3B</td><td>2025-01</td><td>3B</td><td>0.656</td><td>0.648</td><td>0.652</td><td>0.478</td><td>0.403</td><td>0.441</td><td>0.396</td><td>0.174</td><td>0.285</td><td>0.510</td><td>0.408</td><td>0.459</td></tr><tr><td>InternVL3</td><td>2025-04</td><td>8B</td><td>0.392</td><td>0.707</td><td>0.549</td><td>0.290</td><td>0.487</td><td>0.388</td><td>0.253</td><td>0.220</td><td>0.236</td><td>0.311</td><td>0.471</td><td>0.391</td></tr><tr><td>InternVL2.5</td><td>2024-12 2025-08</td><td>8B</td><td>0.383</td><td>0.703</td><td>0.543</td><td>0.285</td><td>0.496</td><td>0.390</td><td>0.213</td><td>0.171</td><td>0.192</td><td>0.294</td><td>0.457</td><td>0.375</td></tr><tr><td>PyVision-Video</td><td></td><td>7B</td><td>0.656</td><td>0.471</td><td>0.563</td><td>0.462</td><td>0.201</td><td>0.332</td><td>0.298</td><td>0.205</td><td>0.252</td><td>0.472</td><td>0.242</td><td>0.357</td></tr><tr><td>LongVideoAgent</td><td>2025-05</td><td>7B</td><td>0.539</td><td>0.571</td><td>0.555</td><td>0.385</td><td>0.301</td><td>0.343</td><td>0.168</td><td>0.097</td><td>0.133</td><td>0.364</td><td>0.323</td><td>0.343</td></tr></table>

& 3 [43, 18], and two agent-style models, PyVision-Video [44] and LongVideoAgent [45]. In terms of model scale, the evaluated open-source models span small (3B/4B), medium (7B/8B/9B), and large (30B/32B/72B) regimes.

To ensure consistent comparison, all models are evaluated under a shared evaluation protocol: given a reference video, each model is prompted with the same inversion instruction and asked to generate an inversion prompt. For video input, most models follow their default or recommended configurations, while GPT-4o is limited to 50 input frames due to API constraints. For replay, the inferred prompt is fed back to the same generator used to generate the ground-truth sample, under the original generation settings and a fixed seed, to generate the replay video for evaluation. For video-level evaluation, we instantiate the Memory Agent with Qwen3-VL-8B and the Judge Agent with Qwen3.5-9B. These settings ensure that performance differences mainly reflect the models’ inversion ability rather than variations in prompting or replay settings.

## 4.2 Main Results

Experiment Design. We evaluate 18 proprietary and open-source VLMs on VI-Bench and report their Prompt Score, Video Score, and Inversion Score across the Easy, Medium, Hard, and Overall settings in Table 2. The goal is to measure whether current VLMs can recover prompts that are semantically aligned with the original prompts and are effective when replayed by the video generator.

Result Analysis. Table 2 shows that current models remain far from solving video prompt inversion: even the strongest model achieves only 0.632 overall Inversion Score, indicating that recovering replayable prompts from AIGC videos remains challenging for existing VLMs. We also observe a clear performance drop as task difficulty increases. Across nearly all models, scores decrease from Easy to Hard, showing that video prompt inversion becomes harder as control factors become richer. For example, GPT-4o drops from 0.731 on Easy to 0.441 on Hard, while Doubao-Seed-2.0- pro drops from 0.751 to 0.529. Moreover, for most models, Prompt Score is higher than Video Score, and this gap becomes larger on Medium and Hard samples. This reveals a central distinction between descriptive recovery and generative recoverability: a model may produce a prompt that looks semantically plausible, yet still fail to infer a prompt that the generator can execute faithfully.

## 4.3 Relationship Between Video Understanding and Video Prompt Inversion

Experiment Design. To examine whether video prompt inversion can be explained by video understanding or captioning ability, we conduct two complementary analyses, as shown in Figure 4. First, we compare VI-Bench with existing video understanding benchmarks. For each model, we average its reported scores on four mainstream video understanding benchmarks, including Video-

MME, MVBench, MLVU, and LongVideoBench, and correlate this average score with its VI-Bench Inversion Score under different difficulty levels. Each point in Figure 4 (left) corresponds to one model, and we report both Pearson correlation r and Spearman rank correlation ρ. Second, we compare video captioning with prompt inversion. For the same samples, we query the same model with either a captioning system prompt or an inversion system prompt, replay both outputs using the same video generator, and evaluate the resulting videos under VI-Bench, as shown in Figure 4 (right).

Result Analysis. Video understanding is related to video prompt inversion, but it is not sufficient. As shown in Figure 4 (left), general video understanding scores show only moderate correlation with VI-Bench on Easy and Medium samples, and the correlation nearly disappears on Hard samples. This suggests that video understanding helps models recognize what appears in the video, which explains why it still correlates with VI-Bench when the task mainly involves single-shot semantic grounding. However, when samples require style recovery, camera behavior, and multi-shot structure, conventional video understanding scores can no longer reliably explain inversion performance.

Captioning is not equivalent to prompt inversion. Figure 4 (right) shows that replacing the captioning prompt with the inversion prompt consistently improves the mean Inversion Score, from 0.442 to 0.680 on Easy, from 0.292 to 0.504 on Medium, from 0.168 to 0.336 on Hard, and from 0.292 to 0.503 overall. This indicates that a detailed caption may describe the video content, but it may still miss the control words needed to regenerate the video, such as style, camera motion, and temporal composition.

![](images/c06e81252ea7e36b60e2be7c0df7bf988f62c36c1d4c56484af2d23483926ad7.jpg)  
(a) Understanding vs. Inversion Correlation

![](images/899647aabdaf669423c569518a6d660c2b52461af7b64684def72f00e290e7e0.jpg)  
(b) Caption vs. Inversion Prompt Comparison  
Figure 4: Analysis of video understanding and the video prompt inversion task. Correlation between average scores on mainstream video understanding benchmarks and VI-Bench Inversion Scores (left), and comparison between captioning-based and inversion-based prompting (right).

## 4.4 Dimension-wise Analysis: From Descriptive Recovery to Generative Recoverability

Experiment Design. To better understand where current VLMs fail on video prompt inversion, we conduct two dimension-wise analyses. First, we compare the average Prompt and Video scores across the five dimensions, as shown in Figure 5(a). Second, we examine the per-model Prompt–Video gap for each dimension, as shown in Figure 5(b), to determine whether failures mainly come from weak prompt recovery or from replay instability.

Result Analysis. Figure 5(a) shows that Subject and Style are recovered more strongly at the prompt level than at the video level, whereas Camera shows the opposite pattern. This indicates that different factors fail in different ways. For Subject and Style, models may recover a plausible textual description, but the prompt is often not precise enough for the generator to reproduce the same subject identity or atmosphere. Camera behaves differently because models often under-specify shot scale, viewpoint, or camera motion in the prompt, while the generator may still introduce plausible camera behavior through its default priors. Figure 5(b) further shows that this mismatch is strongly factor-dependent: most models exhibit positive gaps on Subject, suggesting replay failure, whereas Camera shows negative gaps, suggesting that camera-related content is often under-specified in the inverted prompts. A more detailed model-level analysis is provided in Appendix C.

![](images/347b2a9fce58e3f76a7673b788714587ca566853f751ee55cb5d300307e3d50d.jpg)  
(a) Dimension-wise Score Comparison

![](images/1595fd69278a87dc661870bdc75162d8ba6c98a04b84564ab409d809786ae407.jpg)  
(b) Per-Model Score Gap Heatmap

Figure 5: Dimension-wise gaps between descriptive recovery and generative recoverability. Average Prompt and Video scores across the five generative dimensions (left) and per-model Prompt–Video score gaps for each dimension (right).  
![](images/d855f34b2dda1cd2d4346c84321c6ebd7445cabe19d77503e10d0aeb818d68ae.jpg)  
(a) Model Performance Across Shots

![](images/4bb82304230a69db56dbdb4939e584f35eadfd5de8f0414a64abb499c8e89174.jpg)  
(b) Dimension Dynamics Across Shots  
Figure 6: Multi-shot effects on inversion performance. Relative performance change of each model with increasing shot number (left) and mean score trends of the five dimensions across shots (right).

## 4.5 Multi-shot Videos Reveal Capability Boundaries

Experiment Design. Long-form or film-level AIGC videos are typically composed of multiple shots rather than a single continuous scene. Therefore, prompt inversion for such videos requires more than describing one shot: the model must identify different shot segments, understand their relations, and organize them into a coherent prompt that preserves cross-shot composition and temporal continuity. To examine VLMs’ prompt inversion ability on multi-shot AIGC videos, we analyze model performance across videos with different numbers of shots, as shown in Figure 6. Figure 6(a) reports the relative change of each model with respect to its 1-shot performance, while Figure 6(b) shows how the mean dimension scores vary across these settings.

Result Analysis. Figure 6(a) suggests that moving to multi-shot settings does not immediately make inversion harder: several models improve from 1 shot to 2 shots, suggesting that an extra shot can provide useful visual evidence for prompt recovery. However, this benefit does not persist. From 2 shots onward, most models begin to decline, suggesting that an important difficulty arises when models must organize multiple shots into a single coherent and replayable control representation. Models also respond differently to increasing shot numbers: some remain robust or benefit from additional shots, while others degrade sharply, suggesting different abilities to exploit temporal context. Figure 6(b) further shows that different generative factors respond differently to temporal extension. Style consistently improves as the number of shots increases, rising from 3.35 at 1 shot to 3.71 at 3 shots, while Action drops from 2.79 at 2 shots to 2.53 at 3 shots and remains low thereafter. Subject and Scene also decline after 2 shots, whereas Camera peaks at 2 shots and then stabilizes at a lower level. These findings suggest that multi-shot inversion can be more difficult than one-shot inversion, since models must recover not only each shot, but also how the shots connect and how the whole video should be regenerated.

## 5 Conclusion

We introduce video prompt inversion as a distinct capability beyond conventional video understanding, and present VI-Bench as a dedicated benchmark for evaluating whether VLMs can recover replayable prompts from AIGC videos. Our results reveal a substantial gap between descriptive recovery and replay-stable generative control, especially under richer factors and multi-shot settings. Beyond measuring model capability, VI-Bench also offers a way to study prompt recoverability and potential prompt leakage risks in AIGC video systems. We hope VI-Bench provides a useful benchmark for future research on video prompt inversion and broader evaluation of generative understanding.

## References

[1] Team Seedance. Seedance 2.0: Advancing video generation for world complexity, 2026. URL https://arxiv.org/abs/2604.14148.

[2] Shenghai Yuan, Yuanyang Yin, Zongjian Li, Xinwei Huang, Xiao Yang, and Li Yuan. Helios: Real real-time long video generation model. arXiv preprint arXiv:2603.04379, 2026.

[3] Sand. ai et al. Magi-1: Autoregressive video generation at scale, 2025. URL https://arxiv. org/abs/2505.13211.

[4] Guibin Chen, Dixuan Lin, Jiangping Yang, Chunze Lin, Junchen Zhu, Mingyuan Fan, Hao Zhang, Sheng Chen, Zheng Chen, Chengcheng Ma, Weiming Xiong, Wei Wang, Nuo Pang, Kang Kang, Zhiheng Xu, Yuzhe Jin, Yupeng Liang, Yubing Song, Peng Zhao, Boyuan Xu, Di Qiu, Debang Li, Zhengcong Fei, Yang Li, and Yahui Zhou. Skyreels-v2: Infinite-length film generative model, 2025. URL https://arxiv.org/abs/2504.13074.

[5] Guoqing Ma et al. Step-video-t2v technical report: The practice, challenges, and future of video foundation model, 2025. URL https://arxiv.org/abs/2502.10248.

[6] Yue Ma, Kunyu Feng, Zhongyuan Hu, Xinyu Wang, Yucheng Wang, Mingzhe Zheng, Bingyuan Wang, Qinghe Wang, Xuanhua He, Hongfa Wang, Chenyang Zhu, Hongyu Liu, Yingqing He, Zeyu Wang, Zhifeng Li, Xiu Li, Sirui Han, Yike Guo, Wei Liu, Dan Xu, Linfeng Zhang, and Qifeng Chen. Controllable video generation: A survey, 2026. URL https://arxiv.org/ abs/2507.16869.

[7] Yifang Men, Yuan Yao, Miaomiao Cui, and Liefeng Bo. Mimo: Controllable character video synthesis with spatial decomposed modeling, 2025. URL https://arxiv.org/abs/2409. 16160.

[8] Promptbase, 2024. URL https://promptbase.com/.

[9] Promptai market, 2024. URL https://promptaimarket.com/.

[10] Zhiyao Ren, Yibing Zhan, Baosheng Yu, and Dacheng Tao. Reverse prompt: Cracking the recipe inside text-to-image generation, 2025. URL https://arxiv.org/abs/2503.19937.

[11] Yurong Wu, Fangwen Mu, Qiuhong Zhang, Jinjing Zhao, Xinrun Xu, Lingrui Mei, Yang Wu, Lin Shi, Junjie Wang, Zhiming Ding, and Yiwei Wang. Vulnerability of text-to-image models to prompt template stealing: A differential evolution approach, 2025. URL https: //arxiv.org/abs/2502.14285.

[12] Xinyue Shen, Yiting Qu, Michael Backes, and Yang Zhang. Prompt stealing attacks against text-to-image generation models, 2024. URL https://arxiv.org/abs/2302.09923.

[13] Donghoon Kim, Minji Bae, Kyuhong Shim, and Byonghyo Shim. Visually guided decoding: Gradient-free hard prompt inversion with language models, 2025. URL https://arxiv.org/ abs/2505.08622.

[14] Haoge Deng, Ting Pan, Fan Zhang, Yang Liu, Zhuoyan Luo, Yufeng Cui, Chunhua Shen, Shiguang Shan, Zhaoxiang Zhang, and Xinlong Wang. Uniform discrete diffusion with metric path for video generation. arXiv preprint arXiv:2510.24717, 2025.

[15] Zhongwei Zhang, Fuchen Long, Zhaofan Qiu, Yingwei Pan, Wu Liu, Ting Yao, and Tao Mei. MotionPro: A Precise Motion Controller for Image-to-Video Generation. In CVPR, 2025.

[16] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report, 2025. URL https://arxiv.org/abs/2502.13923.

[17] Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, Wenbin Ge, Zhifang Guo, Qidong Huang, Jie Huang, Fei Huang, Binyuan Hui, Shutong Jiang, Zhaohai Li, Mingsheng Li, Mei Li, Kaixin Li, Zicheng Lin, Junyang Lin, Xuejing Liu, Jiawei Liu, Chenglong Liu, Yang Liu, Dayiheng Liu, Shixuan Liu, Dunjie Lu, Ruilin Luo, Chenxu Lv, Rui Men, Lingchen Meng, Xuancheng Ren, Xingzhang Ren, Sibo Song, Yuchong Sun, Jun Tang, Jianhong Tu, Jianqiang Wan, Peng Wang, Pengfei Wang, Qiuyue Wang, Yuxuan Wang, Tianbao Xie, Yiheng Xu, Haiyang Xu, Jin Xu, Zhibo Yang, Mingkun Yang, Jianxin Yang, An Yang, Bowen Yu, Fei Zhang, Hang Zhang, Xi Zhang, Bo Zheng, Humen Zhong, Jingren Zhou, Fan Zhou, Jing Zhou, Yuanzhi Zhu, and Ke Zhu. Qwen3-vl technical report, 2025. URL https://arxiv.org/abs/2511.21631.

[18] Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Hao Tian, Yuchen Duan, Weijie Su, Jie Shao, Zhangwei Gao, Erfei Cui, Xuehui Wang, Yue Cao, Yangzhou Liu, Xingguang Wei, Hongjie Zhang, Haomin Wang, Weiye Xu, Hao Li, Jiahao Wang, Nianchen Deng, Songze Li, Yinan He, Tan Jiang, Jiapeng Luo, Yi Wang, Conghui He, Botian Shi, Xingcheng Zhang, Wenqi Shao, Junjun He, Yingtong Xiong, Wenwen Qu, Peng Sun, Penglong Jiao, Han Lv, Lijun Wu, Kaipeng Zhang, Huipeng Deng, Jiaye Ge, Kai Chen, Limin Wang, Min Dou, Lewei Lu, Xizhou Zhu, Tong Lu, Dahua Lin, Yu Qiao, Jifeng Dai, and Wenhai Wang. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models, 2025. URL https://arxiv.org/abs/2504.10479.

[19] Biao Yang, Bin Wen, Boyang Ding, Changyi Liu, Chenglong Chu, Chengru Song, Chongling Rao, Chuan Yi, Da Li, Dunju Zang, Fan Yang, Guorui Zhou, Guowang Zhang, Han Shen, Hao Peng, Haojie Ding, Hao Wang, Haonan Fan, Hengrui Ju, Jiaming Huang, Jiangxia Cao, Jiankang Chen, Jingyun Hua, Kaibing Chen, Kaiyu Jiang, Kaiyu Tang, Kun Gai, Muhao Wei, Qiang Wang, Ruitao Wang, Sen Na, Shengnan Zhang, Siyang Mao, Sui Huang, Tianke Zhang, Tingting Gao, Wei Chen, Wei Yuan, Xiangyu Wu, Xiao Hu, Xingyu Lu, Yi-Fan Zhang, Yiping Yang, Yulong Chen, Zeyi Lu, Zhenhua Wu, Zhixin Ling, Zhuoran Yang, Ziming Li, Di Xu, Haixuan Gao, Hang Li, Jing Wang, Lejian Ren, Qigen Hu, Qianqian Wang, Shiyao Wang, Xinchen Luo, Yan Li, Yuhang Hu, and Zixing Zhang. Kwai keye-vl 1.5 technical report, 2025. URL https://arxiv.org/abs/2509.01563.

[20] Chaoyou Fu, Yuhan Dai, Yongdong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, Peixian Chen, Yanwei Li, Shaohui Lin, Sirui Zhao, Ke Li, Tong Xu, Xiawu Zheng, Enhong Chen, Caifeng Shan, Ran He, and Xing Sun. Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis, 2025. URL https://arxiv.org/abs/2405.21075.

[21] Chaoyou Fu, Haozhi Yuan, Yuhao Dong, Yi-Fan Zhang, Yunhang Shen, Xiaoxing Hu, Xueying Li, Jinsen Su, Chengwu Long, Xiaoyao Xie, Yongkang Xie, Xiawu Zheng, Xue Yang, Haoyu Cao, Yunsheng Wu, Ziwei Liu, Xing Sun, Caifeng Shan, and Ran He. Video-mme-v2: Towards the next stage in benchmarks for comprehensive video understanding, 2026. URL https: //arxiv.org/abs/2604.05015.

[22] Xinlong Chen, Yuanxing Zhang, Chongling Rao, Yushuo Guan, Jiaheng Liu, Fuzheng Zhang, Chengru Song, Qiang Liu, Di Zhang, and Tieniu Tan. Vidcapbench: A comprehensive benchmark of video captioning for controllable text-to-video generation, 2025. URL https://arxiv.org/abs/2502.12782.

[23] Zijie J. Wang, Evan Montoya, David Munechika, Haoyang Yang, Benjamin Hoover, and Duen Horng Chau. Diffusiondb: A large-scale prompt gallery dataset for text-to-image generative models, 2023. URL https://arxiv.org/abs/2210.14896.

[24] Wenhao Wang and Yi Yang. Vidprom: A million-scale real prompt-gallery dataset for text-tovideo diffusion models, 2024. URL https://arxiv.org/abs/2403.06098.

[25] Wenhao Wang and Yi Yang. Tip-i2v: A million-scale real text and image prompt dataset for image-to-video generation, 2025. URL https://arxiv.org/abs/2411.04709.

[26] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision, 2021. URL https://arxiv.org/abs/2103.00020.

[27] Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, Kai Chen, and Dahua Lin. Mmbench: Is your multi-modal model an all-around player?, 2024. URL https://arxiv.org/abs/2307.06281.

[28] Yilun Zhao, Lujing Xie, Haowei Zhang, Guo Gan, Yitao Long, Zhiyuan Hu, Tongyan Hu, Weiyuan Chen, Chuhan Li, Junyang Song, Zhijian Xu, Chengye Wang, Weifeng Pan, Ziyao Shangguan, Xiangru Tang, Zhenwen Liang, Yixin Liu, Chen Zhao, and Arman Cohan. Mmvu: Measuring expert-level multi-discipline video understanding, 2025. URL https://arxiv. org/abs/2501.12380.

[29] Kunchang Li, Yali Wang, Yinan He, Yizhuo Li, Yi Wang, Yi Liu, Zun Wang, Jilan Xu, Guo Chen, Ping Luo, Limin Wang, and Yu Qiao. Mvbench: A comprehensive multi-modal video understanding benchmark, 2024. URL https://arxiv.org/abs/2311.17005.

[30] Junjie Zhou, Yan Shu, Bo Zhao, Boya Wu, Zhengyang Liang, Shitao Xiao, Minghao Qin, Xi Yang, Yongping Xiong, Bo Zhang, Tiejun Huang, and Zheng Liu. Mlvu: Benchmarking multi-task long video understanding, 2025. URL https://arxiv.org/abs/2406.04264.

[31] Haoning Wu, Dongxu Li, Bei Chen, and Junnan Li. Longvideobench: A benchmark for longcontext interleaved video-language understanding, 2024. URL https://arxiv.org/abs/ 2407.15754.

[32] Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang, Huan Lin, Baosong Yang, Pengjun Xie, An Yang, Dayiheng Liu, Junyang Lin, Fei Huang, and Jingren Zhou. Qwen3 embedding: Advancing text embedding and reranking through foundation models, 2025. URL https: //arxiv.org/abs/2506.05176.

[33] Maarten Grootendorst. Bertopic: Neural topic modeling with a class-based tf-idf procedure, 2022. URL https://arxiv.org/abs/2203.05794.

[34] Gpt-4o, 2024. URL https://chatgpt.com/.

[35] Claude45, 2025. URL https://www.anthropic.com/news/claude-sonnet-4-5.

[36] Gemini25-flash, 2025. URL https://deepmind.google/models/gemini/flash/.

[37] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, Jianyuan Zeng, Jiayu Wang, Jingfeng Zhang, Jingren Zhou, Jinkai Wang, Jixuan Chen, Kai Zhu, Kang Zhao, Keyu Yan, Lianghua Huang, Mengyang Feng, Ningyi Zhang, Pandeng Li, Pingyu Wu, Ruihang Chu, Ruili Feng, Shiwei Zhang, Siyang Sun, Tao Fang, Tianxing Wang, Tianyi Gui, Tingyu Weng, Tong Shen, Wei Lin, Wei Wang, Wei Wang, Wenmeng Zhou, Wente Wang, Wenting Shen, Wenyuan Yu, Xianzhong Shi, Xiaoming Huang, Xin Xu, Yan Kou, Yangyu Lv, Yifei Li, Yijing Liu, Yiming Wang, Yingya Zhang, Yitong Huang, Yong Li, You Wu, Yu Liu, Yulin Pan, Yun Zheng, Yuntao Hong, Yupeng Shi, Yutong Feng, Zeyinzi Jiang, Zhen Han, Zhi-Fan Wu, and Ziyu Liu. Wan: Open and advanced large-scale video generative models, 2025. URL https://arxiv.org/abs/2503.20314.

[38] Bing Wu, Chang Zou, Changlin Li, Duojun Huang, Fang Yang, Hao Tan, Jack Peng, Jianbing Wu, Jiangfeng Xiong, Jie Jiang, Linus, Patrol, Peizhen Zhang, Peng Chen, Penghao Zhao, Qi Tian, Songtao Liu, Weijie Kong, Weiyan Wang, Xiao He, Xin Li, Xinchi Deng, Xuefei Zhe, Yang Li, Yanxin Long, Yuanbo Peng, Yue Wu, Yuhong Liu, Zhenyu Wang, Zuozhuo Dai, Bo Peng, Coopers Li, Gu Gong, Guojian Xiao, Jiahe Tian, Jiaxin Lin, Jie Liu, Jihong Zhang, Jiesong Lian, Kaihang Pan, Lei Wang, Lin Niu, Mingtao Chen, Mingyang Chen, Mingzhe Zheng, Miles Yang, Qiangqiang Hu, Qi Yang, Qiuyong Xiao, Runzhou Wu, Ryan Xu, Rui Yuan, Shanshan Sang, Shisheng Huang, Siruis Gong, Shuo Huang, Weiting Guo, Xiang Yuan, Xiaojia Chen, Xiawei Hu, Wenzhi Sun, Xiele Wu, Xianshun Ren, Xiaoyan Yuan, Xiaoyue Mi, Yepeng Zhang, Yifu Sun, Yiting Lu, Yitong Li, You Huang, Yu Tang, Yixuan Li, Yuhang Deng, Yuan Zhou, Zhichao Hu, Zhiguang Liu, Zhihe Yang, Zilin Yang, Zhenzhi Lu, Zixiang Zhou, and Zhao Zhong. Hunyuanvideo 1.5 technical report, 2025. URL https://arxiv.org/abs/2511.18870.

[39] Seed20-pro, 2026. URL https://github.com/ByteDance-Seed/Seed2.0.

[40] Hanrong Ye, Chao-Han Huck Yang, Arushi Goel, Wei Huang, Ligeng Zhu, Yuanhang Su, Sean Lin, An-Chieh Cheng, Zhen Wan, Jinchuan Tian, Yuming Lou, Dong Yang, Zhijian Liu, Yukang Chen, Ambrish Dantrey, Ehsan Jahangiri, Sreyan Ghosh, Daguang Xu, Ehsan Hosseini-Asl, Danial Mohseni Taheri, Vidya Murali, Sifei Liu, Yao Lu, Oluwatobi Olabiyi, Yu-Chiang Frank Wang, Rafael Valle, Bryan Catanzaro, Andrew Tao, Song Han, Jan Kautz, Hongxu Yin, and Pavlo Molchanov. Omnivinci: Enhancing architecture and data for omni-modal understanding llm, 2025. URL https://arxiv.org/abs/2510.15870.

[41] Qwen Team. Qwen3.5: Towards native multimodal agents, February 2026. URL https: //qwen.ai/blog?id=qwen3.5.

[42] Hang Zhang, Xin Li, and Lidong Bing. Video-llama: An instruction-tuned audio-visual language model for video understanding, 2023. URL https://arxiv.org/abs/2306.02858.

[43] Zhe Chen, Weiyun Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Erfei Cui, Jinguo Zhu, Shenglong Ye, Hao Tian, Zhaoyang Liu, Lixin Gu, Xuehui Wang, Qingyun Li, Yiming Ren, Zixuan Chen, Jiapeng Luo, Jiahao Wang, Tan Jiang, Bo Wang, Conghui He, Botian Shi, Xingcheng Zhang, Han Lv, Yi Wang, Wenqi Shao, Pei Chu, Zhongying Tu, Tong He, Zhiyong Wu, Huipeng Deng, Jiaye Ge, Kai Chen, Kaipeng Zhang, Limin Wang, Min Dou, Lewei Lu, Xizhou Zhu, Tong Lu, Dahua Lin, Yu Qiao, Jifeng Dai, and Wenhai Wang. Expanding performance boundaries of open-source multimodal models with model, data, and test-time scaling, 2025. URL https://arxiv.org/abs/2412.05271.

[44] Shitian Zhao, Shaoheng Lin, Ming Li, Haoquan Zhang, Wenshuo Peng, Kaipeng Zhang, and Chen Wei. Pyvision-rl: Forging open agentic vision models via rl, 2026. URL https: //arxiv.org/abs/2602.20739.

[45] Runtao Liu, Ziyi Liu, Jiaqi Tang, Yue Ma, Renjie Pi, Jipeng Zhang, and Qifeng Chen. Longvideoagent: Multi-agent reasoning with long videos, 2025. URL https://arxiv.org/ abs/2512.20618.

## Appendix—

## Contents

A Appendix Analysis . . . . . 15   
B Limitations . . . . 15   
C Model-level Dimension-wise Analysis . . . . 16   
D Analysis of Scaling Effects . . . . . . 16   
E Selection of Evaluation Metrics via Human Preference Alignment . . . . . . . .   
F Benchmark Stability and Impact of Video Generators . . . . 18   
G Factor Coupling Analysis and Core Challenges . . . . . 20   
H Implementation Details . . . 20

## A Appendix Analysis

We provide additional appendix analyses to support the main findings and improve reproducibility. Appendix B discusses the limitations of VI-Bench. Appendix C reports model-level per-dimension results to show how prompt-to-replay gaps vary across models and factors. Appendix D studies scaling effects within the Qwen2.5-VL and Qwen3-VL families. Appendix E explains how we select the final prompt- and video-level evaluators based on human preference alignment. Appendix F analyzes the impact of different generators and verifies the stability of VI-Bench across generators. Appendix G examines correlations among the five generative dimensions to reveal structured relationships between inversion abilities. Appendix H provides implementation details, API costs, benchmark examples, and system prompts.

## B Limitations

Although VI-Bench is designed to evaluate video prompt inversion in a controlled and replay-based manner, it still has several limitations. First, due to the high cost of repeatedly calling video generators and the difficulty of ensuring consistent replay under fixed generation settings, we do not include some of the strongest closed-source video generation models in our benchmark construction. Instead, we use Wan and Hunyuan as the video generators for building VI-Bench. This choice may introduce certain generator-specific biases, since different generators can vary in visual style, motion quality, camera behavior, and prompt-following characteristics. However, our generator-specific analysis in Appendix F shows that although the choice of generator affects the absolute difficulty of inversion, especially the replay fidelity, it does not change the overall model ranking or the main findings of VI-Bench. This suggests that our conclusions are not artifacts of a single video generation pipeline.

Second, VI-Bench focuses on the inversion of visual content in AIGC videos. In this version, we do not evaluate the recovery of non-visual modalities such as subtitles, speech, sound effects, background music, or other audio information. However, real-world AIGC videos often contain multimodal signals, and these signals may also carry important prompt-level information. Future work may extend video prompt inversion beyond visual prompts by incorporating audio, speech, and text overlays, enabling a more complete evaluation of multimodal prompt recoverability.

## C Model-level Dimension-wise Analysis

To complement the aggregate dimension-wise analysis in Section 4.4, we further report model-level per-dimension results in Table 3. The table provides the Prompt Score, Video Score, and Inversion Score of each evaluated model across the five generation-critical dimensions: Subject, Action, Scene, Style, and Camera.

The results show that frontier models exhibit the largest Prompt–Video drops on Subject and Style. For example, Doubao-Seed-2.0-pro drops from 3.57 to 2.80 on Subject and from 3.99 to 3.09 on Style, while GPT-4o drops from 3.41 to 2.46 and from 3.62 to 2.78, respectively. This indicates that even strong models can often identify the main subject or visual style at the text level, but fail to express them in a sufficiently precise and replay-stable form for generation. By contrast, Action and Scene are generally more balanced across models, suggesting that once these factors are recovered in the prompt, they are relatively easier to preserve during replay. Camera follows a different pattern: for several weaker open-source models, the video-side score even exceeds the prompt-side score, indicating that camera information is often under-specified in the inferred prompt and only partially compensated by generator priors during replay. These results further support our main finding that video prompt inversion is not limited by uniformly weak understanding across all factors, but by factor-dependent failures in converting recovered visual information into generative control.

Table 3: Overall per-dimension results on VI-Bench. We report per-dimension scores for 18 VLMs. The Overall columns are computed by sample-level aggregation following the main VI-Bench protocol, rather than by directly averaging the five displayed dimensions. Bold indicates the best result in each column; underline indicates the second-best.
<table><tr><td rowspan="2">Method</td><td colspan="3">Subject</td><td colspan="3">Action</td><td colspan="3">Scene</td><td colspan="3">Style</td><td colspan="3">Camera</td><td colspan="3">Overall</td></tr><tr><td>Prompt</td><td>Video</td><td>Inv.</td><td>Prompt</td><td>Video</td><td>Inv.</td><td>Prompt</td><td>Video</td><td>Inv.</td><td>Prompt</td><td>Video</td><td>Inv.</td><td>Prompt</td><td>Video</td><td>Inv.</td><td>Prompt</td><td>Video</td><td>Inv.</td></tr><tr><td>Proprietary MLLMs</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Doubao-Seed-2.0-pro</td><td>3.57</td><td>2.80</td><td>3.18</td><td>3.16</td><td>2.84</td><td>3.00</td><td>3.49</td><td>3.22</td><td>3.36</td><td>3.99</td><td>3.09</td><td>3.54</td><td>3.51</td><td>3.21</td><td>3.36</td><td>3.370</td><td>2.955</td><td>3.160</td></tr><tr><td>GPT-40</td><td>3.41</td><td>2.46</td><td>2.94</td><td>2.83</td><td>2.54</td><td>2.68</td><td>3.33</td><td>2.92</td><td>3.13</td><td>3.62</td><td>2.78</td><td>3.20</td><td>3.13</td><td>2.90</td><td>3.01</td><td>3.175</td><td>2.635</td><td>2.905</td></tr><tr><td>Open-source MLLMs</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>OmniVinci</td><td>3.40</td><td>2.45</td><td>2.95</td><td>2.82</td><td>2.53</td><td>2.66</td><td>3.43</td><td>2.85</td><td>3.10</td><td>3.68</td><td>2.71</td><td>3.15</td><td>3.06</td><td>2.88</td><td>2.97</td><td>3.165</td><td>2.595</td><td>2.880</td></tr><tr><td>Qwen2.5-VL-72B</td><td>3.37</td><td>2.49</td><td>2.93</td><td>2.81</td><td>2.55</td><td>2.68</td><td>3.32</td><td>2.83</td><td>3.07</td><td>3.59</td><td>2.69</td><td>3.14</td><td>2.94</td><td>2.89</td><td>2.92</td><td>3.070</td><td>2.620</td><td>2.845</td></tr><tr><td>Qwen3-VL-8B</td><td>3.12</td><td>2.48</td><td>2.80</td><td>2.55</td><td>2.46</td><td>2.50</td><td>3.08</td><td>2.79</td><td>2.94</td><td>3.43</td><td>2.69</td><td>3.06</td><td>2.84</td><td>2.77</td><td>2.81</td><td>2.960</td><td>2.610</td><td>2.785</td></tr><tr><td>Qwen3-VL-30B</td><td>3.11</td><td>2.43</td><td>2.77</td><td>2.51</td><td>2.42</td><td>2.47</td><td>3.10</td><td>2.79</td><td>2.94</td><td>3.37</td><td>2.66</td><td>3.01</td><td>2.88</td><td>2.77</td><td>2.82</td><td>2.940</td><td>2.560</td><td>2.750</td></tr><tr><td>LLaVA-Video</td><td>3.31</td><td>2.33</td><td>2.82</td><td>2.69</td><td>2.38</td><td>2.53</td><td>3.23</td><td>2.75</td><td>2.99</td><td>3.45</td><td>2.60</td><td>3.02</td><td>2.83</td><td>2.74</td><td>2.78</td><td>3.010</td><td>2.470</td><td>2.740</td></tr><tr><td>Qwen3-VL-4B</td><td>3.07</td><td>2.43</td><td>2.75</td><td>2.47</td><td>2.41</td><td>2.44</td><td>3.01</td><td>2.75</td><td>2.88</td><td>3.34</td><td>2.65</td><td>2.99</td><td>2.84</td><td>2.78</td><td>2.81</td><td>2.900</td><td>2.540</td><td>2.720</td></tr><tr><td>Qwen2.5-VL-32B</td><td>3.03</td><td>2.35</td><td>2.69</td><td>2.55</td><td>2.42</td><td>2.48</td><td>3.00</td><td>2.74</td><td>2.87</td><td>3.27</td><td>2.61</td><td>2.94</td><td>2.95</td><td>2.74</td><td>2.84</td><td>2.915</td><td>2.495</td><td>2.705</td></tr><tr><td>Qwen2.5-VL-7B</td><td>3.27</td><td>2.25</td><td>2.76</td><td>2.61</td><td>2.32</td><td>2.47</td><td>3.18</td><td>2.58</td><td>2.88</td><td>3.38</td><td>2.48</td><td>2.93</td><td>2.55</td><td>2.65</td><td>2.60</td><td>2.905</td><td>2.340</td><td>2.620</td></tr><tr><td>Qwen3.5</td><td>2.92</td><td>2.22</td><td>2.57</td><td>2.47</td><td>2.22</td><td>2.34</td><td>2.86</td><td>2.52</td><td>2.69</td><td>3.23</td><td>2.40</td><td>2.82</td><td>2.78</td><td>2.51</td><td>2.64</td><td>2.870</td><td>2.320</td><td>2.595</td></tr><tr><td>VideoLLaMA3</td><td>3.20</td><td>2.15</td><td>2.67</td><td>2.54</td><td>2.24</td><td>2.39</td><td>3.00</td><td>2.48</td><td>2.74</td><td>3.12</td><td>2.40</td><td>2.76</td><td>2.51</td><td>2.57</td><td>2.54</td><td>2.750</td><td>2.230</td><td>2.490</td></tr><tr><td>Keye-VL</td><td>2.65</td><td>1.91</td><td>2.28</td><td>2.16</td><td>1.98</td><td>2.07</td><td>2.72</td><td>2.39</td><td>2.56</td><td>2.95</td><td>2.26</td><td>2.61</td><td>2.51</td><td>2.36</td><td>2.43</td><td>2.685</td><td>2.145</td><td>2.415</td></tr><tr><td>Qwen2.5-VL-3B</td><td>2.88</td><td>1.94</td><td>2.41</td><td>2.23</td><td>2.00</td><td>2.12</td><td>2.73</td><td>2.26</td><td>2.50</td><td>2.93</td><td>2.17</td><td>2.55</td><td>2.49</td><td>2.40</td><td>2.44</td><td>2.550</td><td>2.040</td><td>2.295</td></tr><tr><td>InternVL3</td><td>2.20 2.06</td><td>2.16 2.12</td><td>2.18</td><td>1.76</td><td>2.25</td><td>2.00</td><td>2.11</td><td>2.60</td><td>2.35</td><td>2.07</td><td>2.46</td><td>2.27</td><td>1.71</td><td>2.60</td><td>2.15</td><td>1.555</td><td>2.355</td><td>1.955</td></tr><tr><td>InternVL2.5</td><td>2.64</td><td>0.99</td><td>2.09 1.81</td><td>1.61</td><td>2.16</td><td>1.89</td><td>1.95</td><td>2.50</td><td>2.22</td><td>1.94</td><td>2.38</td><td>2.16</td><td>1.68</td><td>2.51</td><td>2.09</td><td>1.470</td><td>2.285</td><td>1.875</td></tr><tr><td>PyVision-Video</td><td>2.08</td><td>1.58</td><td>1.83</td><td>2.02</td><td>1.10</td><td>1.56</td><td>2.54</td><td>1.40</td><td>1.97</td><td>2.37</td><td>1.30</td><td>1.83</td><td>2.06</td><td>1.48</td><td>1.77</td><td>2.360</td><td>1.210</td><td>1.785</td></tr><tr><td>Long VideoAgent</td><td></td><td></td><td></td><td>1.60</td><td>1.67</td><td>1.63</td><td>1.87</td><td>1.87</td><td>1.87</td><td>1.94</td><td>1.80</td><td>1.87</td><td>1.67</td><td>2.07</td><td>1.87</td><td>1.820</td><td>1.615</td><td>1.715</td></tr></table>

## D Analysis of Scaling Effects

To examine whether model scaling improves performance on the video prompt inversion task, we conduct a scaling analysis on two representative VLM families: Qwen2.5-VL series and Qwen3-VL series. First, we analyze the overall scaling trend across different difficulty settings, comparing Qwen2.5-VL at four scales (3B, 7B, 32B, and 72B) and Qwen3-VL at three scales (4B, 8B, and 30B) on Medium, Hard, and Overall samples, as shown in Figure 7. Second, we examine how scaling affects different generation-critical dimensions, including Subject, Action, Scene, Style, and Camera, under the same model families and difficulty settings, as shown in Figure 8.

The results show that scaling improves video prompt inversion, but the gains are neither uniform across model families nor evenly distributed across generative factors. As shown in Figure 7, Qwen2.5-VL exhibits clear scaling gains: its score increases from 0.478 to 0.581 on Medium, from 0.396 to 0.539 on Hard, and from 0.510 to 0.613 Overall, with the largest relative gain appearing on Hard samples. This suggests that larger models are better able to handle richer control factors and multi-shot reasoning. In contrast, Qwen3-VL starts from a stronger small model but shows much smaller marginal gains, improving only slightly from 4B to 8B/30B across all settings. As shown in Figure 8, the dimension-wise trends further reveal that scaling mainly strengthens semantic and appearance-related factors such as Subject, Scene, and especially Style, while Action remains consistently lower and Camera improves only weakly. For example, in the Overall setting, Qwen2.5- VL improves substantially in Style and Subject as the model grows, whereas Action remains below the other dimensions and Camera shows non-monotonic behavior. Qwen3-VL shows an even more stable pattern: Style remains the strongest dimension, Subject and Scene stay relatively high, while Action is consistently the weakest. These findings indicate that larger models become better at recognizing what should be recovered, but do not automatically acquire temporally structured action understanding or precise camera control.

![](images/429975cd68d1044293d51b3e8a7e0ab26757e11a9b9e9b330625c710a8aad5da.jpg)

![](images/22eea29384348297a3243bc36d35c23f5e7a374904199b1c5f0854a3efb8a53f.jpg)

![](images/0360469bfd0ae48811a64c90b058746da4610f55668c344c164a0fdbee56e907.jpg)  
Figure 7: Intra-family scaling effects on VI-Bench. Performance of Qwen2.5-VL and Qwen3-VL across model sizes under the Medium, Hard, and Overall settings.

![](images/4745dfa6d589a55c170fad617320757a6c1155dad72577838b98143c99c4f89f.jpg)  
Figure 8: Dimension-wise scaling trends. Per-dimension performance changes of Qwen2.5-VL and Qwen3-VL across model sizes under the Medium, Hard, and Overall settings.

## E Selection of Evaluation Metrics via Human Preference Alignment

To select reliable evaluation metrics for VI-Bench, we compare multiple candidate automatic metrics and choose the final protocol according to their agreement with human preferences. Rather than assuming a metric a priori, we construct a human-annotated validation subset and measure how well each candidate metric correlates with human scores. Specifically, we sample 100 benchmark instances covering different difficulty levels, VLMs, and video generators. Each instance is independently scored by five annotators. For the video-level task, annotators are shown the ground-truth video and the replayed inversion video, and score their similarity from 1 to 5 along five aspects: subject content, scene environment, cinematography, visual style, and narrative. For the prompt-level task, annotators are shown the ground-truth prompt and the inferred prompt, and assign an overall alignment score from 1 to 5. We average the annotator scores for each instance as the human preference reference, and then compute the Pearson Linear Correlation Coefficient (PLCC) between each automatic metric and the human scores. We evaluate six video-level metrics and two prompt-level metrics. For video-level evaluation, the candidates include: Video-EvalAgent, where a Memory Agent summarizes each video independently and a Judge Agent compares the resulting memories; Video-Gemini, where the ground-truth and inversion videos are jointly provided to the Gemini API for direct multi-video comparison; Video-PyVision, which follows the same agent-based framework as Video-EvalAgent but replaces the Qwen3-VL Memory Agent with the agent-style VLM PyVision-Video; Video-VLM-Concat, where the ground-truth and inversion videos are concatenated and then evaluated by Qwen3-VL; Video-EvalAgent-MultiShot, which performs shot-level evaluation and averages scores across shots; and CLIP-I, which computes frame-level CLIP image similarity between the two videos. For prompt-level evaluation, we compare LLM-Judge, the prompt evaluator used in VI-Bench, with CLIP-T, which computes CLIP text embedding similarity between the ground-truth and inferred prompts. As shown in Table 4, Video-EvalAgent achieves the highest correlation with human preferences among all video-level metrics, with a PLCC of +0.6864. It slightly outperforms Video-Gemini (+0.6806) and performs better than Video-PyVision (+0.6463), Video-VLM-Concat (+0.5688), Video-EvalAgent-MultiShot (+0.4849), and CLIP-I (+0.3416). This result suggests that decomposing each video into a structured memory before comparison provides a stronger alignment with human judgments than directly concatenating videos, averaging shot-level comparisons, or using frame-level CLIP similarity. Although Video-Gemini is also competitive, Video-EvalAgent obtains the best human alignment and does not rely on direct multi-video comparison APIs. We therefore adopt Video-EvalAgent as the video-level evaluation metric in VI-Bench. For prompt-level evaluation, LLM-Judge achieves a higher correlation with human preferences than CLIP-T. Based on this human preference alignment study, we adopt LLM-Judge for prompt-level evaluation and Video-EvalAgent for video-level evaluation in the main VI-Bench protocol.

Table 4: Validation of evaluation metrics against human preferences. We compute the Pearson Linear Correlation Coefficient (PLCC) between each automatic metric and human-annotated scores on randomly sampled benchmark instances.
<table><tr><td>Evaluation Metric</td><td>PLCC (r)</td><td>Adopted</td></tr><tr><td>Video-Level Evaluation</td><td></td><td></td></tr><tr><td>Video-EvalAgent (Ours)</td><td>+0.6864</td><td></td></tr><tr><td>Video-Gemini</td><td>+0.6806</td><td></td></tr><tr><td>Video-PyVision Video-VLM-Concat</td><td>+0.6463</td><td></td></tr><tr><td>Video-EvalAgent-MultiShot</td><td>+0.5688 +0.4849</td><td></td></tr><tr><td>CLIP-I</td><td>+0.3416</td><td></td></tr><tr><td>Prompt-Level Evaluation</td><td></td><td></td></tr><tr><td>LLM-Judge (Ours)</td><td>+0.6044</td><td></td></tr><tr><td>CLIP-T</td><td>+0.4629</td><td></td></tr></table>

## F Benchmark Stability and Impact of Video Generators

Since VI-Bench contains samples generated by two different video generators, Wan and Hunyuan, we further analyze whether the benchmark results are sensitive to the choice of generator and examine how different generators affect video prompt inversion difficulty.

Specifically, we split VI-Bench by generator and report results on Wan-generated and Hunyuangenerated samples in Table 5 and Table 6, respectively. The results show that the choice of generator does affect the absolute difficulty of video prompt inversion, but this effect is mainly reflected in replay fidelity rather than prompt fidelity. Overall, models achieve higher Inversion Scores on Wangenerated samples than on Hunyuan-generated samples. However, this gap comes primarily from the Video Score: averaged over all models, the Prompt Score decreases only moderately from Wan to Hunyuan, whereas the Video Score drops much more substantially. This suggests that different generators do not merely change whether a VLM can infer a semantically plausible prompt from the input video; rather, they more strongly affect whether the recovered prompt remains stable when replayed by the original generator. In other words, generator-specific differences mainly influence the second stage of video prompt inversion, where the predicted prompt is executed again to reproduce the reference video. Despite this difference in absolute difficulty, the relative model rankings remain highly consistent across the two generators. Strong models such as Doubao-Seed-2.0-pro, GPT-4o, OmniVinci, Qwen2.5-VL-72B, and Qwen3-VL remain among the top-performing methods on both Wan and Hunyuan, while weaker models remain near the bottom across both subsets. This stability indicates that different generators change the replay difficulty of recovered prompts, but they do not overturn the overall capability ordering or the central conclusion that current VLMs still struggle to transform visual understanding into replay-stable generative control.

Table 5: Generator-specific results on VI-Bench using Wan. We evaluate 18 VLMs on VI-Bench with samples generated by Wan across three difficulty levels. Prompt Score measures prompt fidelity; Video Score measures replay fidelity; Inversion Score is the average of the two, reflecting a model’s overall ability to recover replayable prompts from AIGC videos. Bold indicates the best result in each column; underline indicates the second-best.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Release</td><td rowspan="2">#Params</td><td colspan="3">Easy</td><td colspan="3">Medium</td><td colspan="3">Hard</td><td colspan="3">Overall</td></tr><tr><td>Prompt</td><td>Video</td><td>Inv.</td><td>Prompt</td><td>Video</td><td>Inv.</td><td>Prompt</td><td>Video</td><td>Inv.</td><td>Prompt</td><td>Video</td><td>Inv.</td></tr><tr><td>Proprietary MLLMs</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Doubao-Seed-2.0-pro GPT-40</td><td>2025-10 2024-08</td><td>一</td><td>0.759 0.748</td><td>0.782 0.733</td><td>0.771 0.741</td><td>0.676 0.638</td><td>0.645 0.573</td><td>0.661 0.606</td><td>0.651 0.574</td><td>0.571 0.442</td><td>0.611 0.508</td><td>0.695 0.654</td><td>0.666 0.583</td><td>0.681 0.618</td></tr><tr><td></td><td></td><td>一</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Open-source MLLMs</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>OmniVinci</td><td>2025-10</td><td>7B</td><td>0.752</td><td>0.735</td><td>0.744</td><td>0.603</td><td>0.567</td><td>0.585</td><td>0.604</td><td>0.395</td><td>0.500</td><td>0.653</td><td>0.566</td><td>0.609</td></tr><tr><td>Qwen2.5-VL-72B</td><td>2025-01</td><td>72B</td><td>0.736</td><td>0.750</td><td>0.743</td><td>0.608</td><td>0.575</td><td>0.592</td><td>0.569</td><td>0.408</td><td>0.489</td><td>0.638</td><td>0.578</td><td>0.608</td></tr><tr><td>Qwen3-VL-8B</td><td>2025-09</td><td>8B</td><td>0.733</td><td>0.781</td><td>0.757</td><td>0.596</td><td>0.610</td><td>0.603</td><td>0.493</td><td>0.345</td><td>0.419</td><td>0.607</td><td>0.579</td><td>0.593</td></tr><tr><td>Qwen3-VL-30B</td><td>2025-09</td><td>30B</td><td>0.739</td><td>0.761</td><td>0.750</td><td>0.603</td><td>0.589</td><td>0.596</td><td>0.489</td><td>0.352</td><td>0.420</td><td>0.610</td><td>0.567</td><td>0.589</td></tr><tr><td>LLaVA-Video</td><td>2024-10</td><td>7B</td><td>0.732</td><td>0.734</td><td>0.733</td><td>0.591</td><td>0.541</td><td>0.566</td><td>0.538</td><td>0.386</td><td>0.462</td><td>0.620</td><td>0.554</td><td>0.587</td></tr><tr><td>Qwen3-VL-4B</td><td>2025-09</td><td>4B</td><td>0.732</td><td>0.744</td><td>0.738</td><td>0.594</td><td>0.609</td><td>0.601</td><td>0.484</td><td>0.345</td><td>0.414</td><td>0.603</td><td>0.566</td><td>0.585</td></tr><tr><td>Qwen2.5-VL-32B</td><td>2025-03</td><td>32B</td><td>0.737</td><td>0.747</td><td>0.742</td><td>0.601</td><td>0.586</td><td>0.594</td><td>0.488</td><td>0.339</td><td>0.414</td><td>0.609</td><td>0.557</td><td>0.583</td></tr><tr><td>Qwen2.5-VL-7B</td><td>2025-01</td><td>7B</td><td>0.736</td><td>0.698</td><td>0.717</td><td>0.557</td><td>0.528</td><td>0.543</td><td>0.525</td><td>0.353</td><td>0.439</td><td>0.606</td><td>0.526</td><td>0.566</td></tr><tr><td>Qwen3.5</td><td>2025-11</td><td>9B</td><td>0.741</td><td>0.721</td><td>0.731</td><td>0.632</td><td>0.537</td><td>0.584</td><td>0.442</td><td>0.280</td><td>0.361</td><td>0.605</td><td>0.513</td><td>0.559 0.538</td></tr><tr><td>VideoLLaMA3</td><td>2025-01</td><td>7B</td><td>0.713</td><td>0.680</td><td>0.697</td><td>0.523</td><td>0.497</td><td>0.510</td><td>0.502</td><td>0.313</td><td>0.408</td><td>0.579</td><td>0.497</td><td>0.518</td></tr><tr><td>Keye-VL</td><td>2025-10</td><td>8B</td><td>0.730</td><td>0.730</td><td>0.730</td><td>0.589</td><td>0.546</td><td>0.567</td><td>0.345</td><td>0.166</td><td>0.255</td><td>0.555</td><td>0.480</td><td></td></tr><tr><td>Qwen2.5-VL-3B</td><td>2025-01</td><td>3B 8B</td><td>0.679</td><td>0.667</td><td>0.673</td><td>0.496</td><td>0.444</td><td>0.470</td><td>0.414</td><td>0.235</td><td>0.324</td><td>0.530</td><td>0.448</td><td>0.489 0.421</td></tr><tr><td>InternVL3</td><td>2025-04 2024-12</td><td>8B</td><td>0.400 0.393</td><td>0.731</td><td>0.565 0.563</td><td>0.298 0.300</td><td>0.528 0.560</td><td>0.413 0.430</td><td>0.265 0.222</td><td>0.305 0.231</td><td>0.285 0.226</td><td>0.321 0.305</td><td>0.521</td><td>0.406</td></tr><tr><td>InternVL2.5</td><td>2025-08</td><td>7B</td><td>0.672</td><td>0.733 0.611</td><td>0.641</td><td>0.482</td><td>0.377</td><td>0.429</td><td>0.318</td><td>-0.198</td><td>0.060</td><td>0.490</td><td>0.508 0.263</td><td>0.377</td></tr><tr><td>PyVision-Video</td><td></td><td>7B</td><td>0.558</td><td>0.580</td><td>0.569</td><td>0.404</td><td>0.344</td><td>0.374</td><td>0.173</td><td>0.135</td><td>0.154</td><td>0.378</td><td></td><td></td></tr><tr><td>LongVideoAgent</td><td>2025-05</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.353</td><td>0.366</td></tr></table>

Table 6: Generator-specific results on VI-Bench using Hunyuan. We evaluate 18 VLMs on VI-Bench with samples generated by Hunyuan across three difficulty levels. Prompt Score measures prompt fidelity; Video Score measures replay fidelity; Inversion Score is the average of the two, reflecting a model’s overall ability to recover replayable prompts from AIGC videos. Bold indicates the best result in each column; underline indicates the second-best.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Release</td><td rowspan="2">#Params</td><td colspan="3">Easy</td><td colspan="3">Medium</td><td colspan="3">Hard</td><td colspan="3">Overall</td></tr><tr><td>Prompt</td><td>Video</td><td>Inv.</td><td>Prompt</td><td>Video</td><td>Inv.</td><td>Prompt</td><td>Video</td><td>Inv.</td><td>Prompt</td><td>Video</td><td>Inv.</td></tr><tr><td>Proprietary MLLMs</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Doubao-Seed-2.0-pro GPT-40</td><td>2025-10</td><td>一</td><td>0.735 0.736</td><td>0.728 0.708</td><td>0.732 0.722</td><td>0.621 0.583</td><td>0.523 0.483</td><td>0.572 0.533</td><td>0.603 0.527</td><td>0.293 0.219</td><td>0.448 0.373</td><td>0.653 0.615</td><td>0.514</td><td>0.584 0.543</td></tr><tr><td></td><td>2024-08</td><td>1</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.470</td><td></td></tr><tr><td colspan="2">Open-source MLLMs</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.543</td></tr><tr><td>OmniVinci</td><td>2025-10</td><td>7B</td><td>0.710</td><td>0.691</td><td>0.701</td><td>0.571</td><td>0.483</td><td>0.527</td><td>0.561</td><td>0.240</td><td>0.401</td><td>0.614</td><td>0.471</td><td></td></tr><tr><td>Qwen2.5-VL-72B</td><td>2025-01</td><td>72B</td><td>0.705</td><td>0.702</td><td>0.703</td><td>0.554</td><td>0.490</td><td>0.522</td><td>0.509</td><td>0.217</td><td>0.363</td><td>0.589</td><td>0.469</td><td>0.529</td></tr><tr><td>Qwen3-VL-8B</td><td>2025-09</td><td>8B</td><td>0.707</td><td>0.711</td><td>0.709</td><td>0.561</td><td>0.518</td><td>0.539</td><td>0.460</td><td>0.165</td><td>0.312</td><td>0.576</td><td>0.465</td><td>0.520</td></tr><tr><td>Qwen3-VL-30B</td><td>2025-09</td><td>30B</td><td>0.684</td><td>0.696</td><td>0.690</td><td>0.554</td><td>0.508</td><td>0.531</td><td>0.455</td><td>0.164</td><td>0.310</td><td>0.565</td><td>0.456</td><td>0.510</td></tr><tr><td>LLaVA-Video</td><td>2024-10</td><td>7B</td><td>0.713</td><td>0.670</td><td>0.692</td><td>0.530</td><td>0.428</td><td>0.479</td><td>0.509</td><td>0.203</td><td>0.356</td><td>0.584</td><td>0.434</td><td>0.509</td></tr><tr><td>Qwen3-VL-4B Qwen2.5-VL-32B</td><td>2025-09</td><td>4B</td><td>0.698</td><td>0.696</td><td>0.697</td><td>0.520</td><td>0.496</td><td>0.508</td><td>0.452</td><td>0.155</td><td>0.304</td><td>0.557</td><td>0.449</td><td>0.503</td></tr><tr><td></td><td>2025-03</td><td>32B</td><td>0.699</td><td>0.675</td><td>0.687</td><td>0.553</td><td>0.485</td><td>0.519</td><td>0.417</td><td>0.161</td><td>0.289</td><td>0.556</td><td>0.441</td><td>0.498</td></tr><tr><td>Qwen2.5-VL-7B</td><td>2025-01</td><td>7B</td><td>0.699</td><td>0.653</td><td>0.676</td><td>0.493</td><td>0.411</td><td>0.452</td><td>0.474</td><td>0.164</td><td>0.319</td><td>0.555</td><td>0.409</td><td>0.482</td></tr><tr><td>Qwen3.5</td><td>2025-11</td><td>9B</td><td>0.700</td><td>0.690</td><td>0.695</td><td>0.559</td><td>0.469</td><td>0.514</td><td>0.373</td><td>0.087</td><td>0.230</td><td>0.544</td><td>0.415</td><td>0.480</td></tr><tr><td>VideoLLaMA3</td><td>2025-01</td><td>7B</td><td>0.659</td><td>0.629</td><td>0.644</td><td>0.437</td><td>0.389</td><td>0.413</td><td>0.465</td><td>0.169</td><td>0.317</td><td>0.520</td><td>0.396</td><td>0.458</td></tr><tr><td>Keye-VL</td><td>2025-10</td><td>8B</td><td>0.693</td><td>0.663</td><td>0.678</td><td>0.547</td><td>0.433</td><td>0.490</td><td>0.317</td><td>0.037</td><td>0.177</td><td>0.519</td><td>0.378</td><td>0.448</td></tr><tr><td>Qwen2.5-VL-3B</td><td>2025-01 2025-04</td><td>3B 8B</td><td>0.633</td><td>0.628</td><td>0.630</td><td>0.460</td><td>0.363</td><td>0.412</td><td>0.378</td><td>0.114</td><td>0.246 0.187</td><td>0.490</td><td>0.368</td><td>0.429</td></tr><tr><td>InternVL3</td><td>2024-12</td><td>8B</td><td>0.384 0.373</td><td>0.683 0.673</td><td>0.533 0.523</td><td>0.281 0.270</td><td>0.445 0.431</td><td>0.363 0.350</td><td>0.240 0.203</td><td>0.134 0.111</td><td>0.157</td><td>0.301 0.282</td><td>0.420 0.405</td><td>0.361 0.344</td></tr><tr><td>InternVL2.5 PyVision-Video</td><td>2025-08</td><td>7B</td><td>0.640</td><td>0.590</td><td>0.615</td><td>0.441</td><td>0.286</td><td>0.363</td><td>0.277</td><td>-0.212</td><td>0.032</td><td>0.453</td><td></td><td>0.337</td></tr><tr><td></td><td>2025-05</td><td>7B</td><td>0.518</td><td>0.562</td><td>0.540</td><td>0.365</td><td>0.257</td><td>0.311</td><td>0.163</td><td>0.059</td><td>0.111</td><td>0.349</td><td>0.221</td><td></td></tr><tr><td>LongVideoAgent</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.293</td><td>0.321</td></tr></table>

## G Factor Coupling Analysis and Core Challenges

To investigate why video prompt inversion remains difficult even when some individual factors are partially recoverable, we further analyze the structural relationships among the five generative dimensions. As shown in Figure 9(a) and Figure 9(b), Subject, Action, and Scene form a clear content cluster, with relatively high pairwise correlations, suggesting that recognizing what is in the video, what it is doing, and where it happens largely relies on a shared semantic understanding ability. Scene and Style are also strongly coupled $( r = 0 . 5 7 )$ , indicating that style perception is often grounded in background cues such as lighting, color tone, and atmosphere. In contrast, Camera is the only isolated dimension: its correlations with all other factors are consistently the lowest, implying that camera understanding does not naturally improve together with content understanding. This further suggests that video prompt inversion is not a task of recovering five independent labels, but of operating in a structured factor space where some abilities are mutually supportive while others are fundamentally distinct. Since Subject, Action, and Scene are strongly coupled, improvements in general semantic video understanding may benefit these content-related dimensions together. However, Camera behaves as a more independent ability: a model can become stronger at recognizing subjects, actions, and scenes while still failing to recover shot scale, viewpoint, or camera motion. Style lies between these two cases. Its strong correlation with Scene suggests that style is often inferred from static visual cues such as lighting, color tone, and background atmosphere, while its weaker correlation with Action suggests that style recovery is less tied to dynamic event understanding. These observations indicate that improving video prompt inversion may require not only stronger general video understanding, but also more knowledge like camera language or styles.

![](images/ce8a1ac3c9292c616baf77ad4b70d010978446b5d970f1b7c1df596fd1d7aeb6.jpg)  
(a) Pairwise Score Trade-off Distributions

![](images/5b2734cedbb2e9acb365d021c035ace0c14cff5496be9a3b8e0ed758092422d8.jpg)  
(b) Dimension Score Correlation Matrix

Figure 9: Pairwise score trade-off distributions between dimension pairs (left) and the correlation matrix of dimension scores across the five generative factors (right).

## H Implementation Details

All experiments are conducted on a server equipped with four NVIDIA RTX 6000 GPUs. For GPT-4o, we use the gpt-4o-2024-11-20 version throughout our evaluation. In terms of API cost, evaluating Prompt Score with LLM-Judge (GPT-4o) costs approximately \$3 for one full pass over VI-Bench. For model inference, running GPT-4o on the full VI-Bench costs approximately \$42, while running Doubao-Seed-2.0-Pro costs approximately \$8. These costs only refer to the corresponding inference or evaluation calls and may vary with API pricing or implementation details.

We also provide representative examples and system prompts used in VI-Bench. Figures 10 and 11 show Easy- and Medium-level samples, respectively, while Figures 12 and 13 show Hard-level multi-shot samples. To improve reproducibility, we further present the system prompt used for Prompt Score evaluation in Figure 14, the system prompt used by the Memory Agent in Figure 15, and the system prompt used by the Evaluation Agent in Figure 16.

![](images/0461405658c6aa1ee9ad85ee236695d82dbc4b5b27e12dca20e001d001d47d09.jpg)  
Figure 10: Easy-level samples in VI-Bench.

![](images/d42cfb70794da571aeaec9eb65d87b84cbdbe402cde7dcbe7c082eab77003d4c.jpg)  
Figure 11: Medium-level samples in VI-Bench.

![](images/bf3eb98dd7677626069a74714d74df3b4e582b45b3271eeee6492e1983fe6a67.jpg)  
Figure 12: A Hard-level multi-shot sample in VI-Bench.

![](images/3c757886e6afa11f33e38f099778150ccd32d2898835c2c9f66a69b7159841e9.jpg)  
Figure 13: A Hard-level multi-shot sample in VI-Bench.

![](images/72989a4c5de1df291575f5f9b5147a63155b47e1c8cdd79574d3b455adb22810.jpg)  
Figure 14: System prompt used for Prompt Score evaluation.

![](images/4b9930d69eb632c9eafe3d851e7a9b772686abb1e1142ac37c18cf90d9a51fa9.jpg)  
Figure 15: System prompt used for the Memory Agent.

![](images/a31519b61d9ca2475e17e62c0ebe745894359fbdf263c4755b86f95d3e948012.jpg)  
Figure 16: System prompt used for the Evaluation Agent.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper’s contributions and scope?

Answer: [Yes]

Justification: The main claims in the abstract and introduction are aligned with the scope of the paper: defining video prompt inversion, introducing VI-Bench, and evaluating current VLMs under a closed-loop replay protocol. The empirical claims are supported by the benchmark construction, experimental results, and analysis sections.

Guidelines:

• The answer [N/A] means that the abstract and introduction do not include the claims made in the paper.

• The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A [No] or [N/A] answer to this question will not be perceived well by the reviewers.

• The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.

• It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: The paper discusses limitations related to the benchmark scope, the limited set of video generators, the dependence on fixed generation settings, prompt non-uniqueness, and possible biases of LLM-based evaluation. These limitations are described in the limitations and broader discussion sections.

Guidelines:

• The answer [N/A] means that the paper has no limitation while the answer [No] means that the paper has limitations, but those are not discussed in the paper.

• The authors are encouraged to create a separate “Limitations” section in their paper.

• The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.

• The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.

• The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.

• The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.

• If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.

• While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren’t acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.

## 3. Theory assumptions and proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

Answer: [N/A]

Justification: The paper does not present theoretical results, theorems, lemmas, or formal proofs. Its contributions are centered on task formulation, benchmark construction, empirical evaluation, and analysis.

Guidelines:

• The answer [N/A] means that the paper does not include theoretical results.

• All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.

• All assumptions should be clearly stated or referenced in the statement of any theorems.

• The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.

• Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.

• Theorems and Lemmas that the proof relies upon should be properly referenced.

## 4. Experimental result reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

Answer: [Yes]

Justification: The paper provides the benchmark construction procedure, data sources, difficulty design, model list, inference protocol, replay setting, fixed generation configuration, and scoring procedure. Additional implementation details, evaluation prompts, and reproducibility instructions are provided in the appendix and supplementary materials.

Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• If the paper includes experiments, a [No] answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.

• If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.

• Depending on the contribution, reproducibility can be accomplished in various ways. For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.

• While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example

(a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm.

(b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully.

(c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset).

(d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.

## 5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?

Answer: [Yes]

Justification: The paper provides open access to the benchmark assets and evaluation code through anonymized supplementary materials or an anonymized repository. The release includes data documentation, evaluation scripts, prompt templates, and instructions for reproducing the main results.

Guidelines:

• The answer [N/A] means that paper does not include experiments requiring code.

• Please see the NeurIPS code and data submission guidelines (https://neurips.cc/ public/guides/CodeSubmissionPolicy) for more details.

• While we encourage the release of code and data, we understand that this might not be possible, so [No] is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).

• The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https: //neurips.cc/public/guides/CodeSubmissionPolicy) for more details.

• The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.

• The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.

• At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).

• Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

## 6. Experimental setting/details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer) necessary to understand the results?

Answer: [Yes]

Justification: The paper specifies the experimental setting, including data construction, difficulty levels, evaluated models, frame sampling strategy, model prompting protocol, replay configuration, scoring dimensions, and aggregation rules. Full implementation and evaluation details are provided in the appendix and supplementary materials.

Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.

• The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [No]

Justification: The paper reports aggregate benchmark scores and human validation results, but does not provide error bars or confidence intervals for all main experimental results. This is mainly due to the high cost of closed-loop video replay and LLM-based evaluation across many models and samples.

## Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• The authors should answer [Yes] if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.

• The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).

• The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.)

• The assumptions made should be given (e.g., Normally distributed errors).

• It should be clear whether the error bar is the standard deviation or the standard error of the mean.

• It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.

• For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g., negative error rates).

• If error bars are reported in tables or plots, the authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text.

## 8. Experiments compute resources

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?

Answer: [Yes]

Justification: The paper reports the computational resources used for model inference, video replay, and LLM-based evaluation, including GPU type, memory, runtime, and overall compute estimates. Additional resource details are provided in the appendix.

Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.

• The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.

• The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn’t make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?

Answer: [Yes]

Justification: The research follows the NeurIPS Code of Ethics by using publicly available or properly credited assets, filtering unsafe or private content, documenting annotation procedures, and discussing potential dual-use risks. The released assets are intended for research and evaluation purposes.

Guidelines:

• The answer [N/A] means that the authors have not reviewed the NeurIPS Code of Ethics.

• If the authors answer [No], they should explain the special circumstances that require a deviation from the Code of Ethics.

• The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [Yes]

Justification: The paper discusses positive impacts such as evaluating generative controllability, supporting creative reuse, and studying prompt leakage risks. It also discusses negative impacts such as potential prompt stealing, misuse for imitation, and risks related to intellectual property or privacy.

Guidelines:

• The answer [N/A] means that there is no societal impact of the work performed.

• If the authors answer [N/A] or [No], they should explain why their work has no societal impact or why the paper does not address societal impact.

• Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.

• The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate Deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.

• The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.

• If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).

## 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pre-trained language models, image generators, or scraped datasets)?

Answer: [Yes]

Justification: The paper describes safeguards for responsible release, including filtering unsafe or private content, releasing the benchmark for research evaluation rather than misuse, and avoiding the release of tools designed to attack proprietary systems. The paper also discusses prompt leakage risks and mitigation considerations.

Guidelines:

• The answer [N/A] means that the paper poses no such risks.

• Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.

• Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.

• We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: The paper credits the original creators of all existing datasets, models, APIs, and codebases used in the work. The appendix provides an asset table listing their sources, versions, citations, licenses, and terms of use when available.

Guidelines:

• The answer [N/A] means that the paper does not use existing assets.

• The authors should cite the original paper that produced the code package or dataset.

• The authors should state which version of the asset is used and, if possible, include a URL.

• The name of the license (e.g., CC-BY 4.0) should be included for each asset.

• For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.

• If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.

• For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.

• If this information is not available online, the authors are encouraged to reach out to the asset’s creators.

## 13. New assets

Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?

Answer: [Yes]

Justification: The paper introduces VI-Bench as a new benchmark and provides documentation for its data format, construction process, difficulty levels, evaluation protocol, annotation guideline, and intended use. The released assets are accompanied by instructions and metadata.

Guidelines:

• The answer [N/A] means that the paper does not release new assets.

• Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.

• The paper should discuss whether and how consent was obtained from people whose asset is used.

• At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [Yes]

Justification: The paper includes human validation with annotators and provides the full annotation instructions, scoring criteria, task examples, and compensation details in the appendix. The annotation protocol covers both prompt-level and video-level comparison tasks.

Guidelines:

• The answer [N/A] means that the paper does not involve crowdsourcing nor research with human subjects.

• Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.

• According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [N/A]

Justification: The annotation task only asks annotators to rate generated videos and prompts, does not collect personal or sensitive information, and does not study the annotators themselves. Therefore, IRB approval or equivalent review is not applicable under this setting.

Guidelines:

• The answer [N/A] means that the paper does not involve crowdsourcing nor research with human subjects.

• Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.

• We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.

• For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigor, or originality of the research, declaration is not required.

Answer: [Yes]

Justification: The paper uses LLMs and VLMs as core components of the evaluation pipeline, including prompt-level judging and video-level memory-and-judge evaluation. The paper describes the model roles, input-output format, scoring protocol, and validation against human annotations.

Guidelines:

• The answer [N/A] means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.

• Please refer to our LLM policy in the NeurIPS handbook for what should or should not be described.