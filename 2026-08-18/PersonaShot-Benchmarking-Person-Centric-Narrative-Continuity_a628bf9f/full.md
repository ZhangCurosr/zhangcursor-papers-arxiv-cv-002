# PersonaShot: Benchmarking Person-Centric Narrative Continuity in Multi-Shot Video Generation

Yuji Wang<sup>1\*</sup> Yuheng Chen<sup>1\*</sup> Teng Hu<sup>1</sup> Ran Yi<sup>1†</sup> Yijia Hong<sup>1</sup> Han Feng<sup>2</sup> Weijian Cao<sup>2</sup> Chengjie Wang<sup>2</sup> Lizhuang Ma<sup>1‡</sup> Jiangning Zhang<sup>3</sup>

<sup>1</sup>Shanghai Jiao Tong University <sup>2</sup>Tencent Youtu Lab <sup>3</sup>Zhejiang University {yujiwang@sjtu.edu.cn, ma-lz@cs.sjtu.edu.cn}

## Abstract

Video generation is rapidly evolving from single-shot clips to multi-shot narratives, where the human character serves as the core narrative anchor. However, existing benchmarks mainly assess character appearance or individualshot quality, without measuring whether physical and emotional states remain coherent across cuts. They also rarely provide criterion-specific evaluation methods, although physical continuity, facial dynamics, and cinematic relations require different visual, temporal, and relational evidence. To address these limitations, we introduce Per sonaShot, the first person-centric benchmark for narrative continuity in multi-shot video generation. PersonaShot contains approximately 1,000 multi-shot segments and 16 metrics spanning physical continuity, affective dynamics, and cinematic grammar. 1) Narrative Continuity Bench mark: We evaluate character coherence across three temporal levels: within-shot states, cross-shot transitions, and sequence-level trajectories. 2) Human-Aligned Specialist Evaluators: We distill reasoning from a large multimodal teacher into lightweight criterion-specific evaluators, each grounded in the visual, temporal, or relational evidence required by its metric, and align them with expert human judgments. 3) Systematic Evaluation and Insights: Our evaluation reveals distinct capability profiles across state-of-the-art models and a clear gap between perceptual quality and cross-shot narrative continuity. Even visually compelling videos frequently exhibit physical-state resets, abrupt affective shifts, and broken cinematic relations across shots. Human studies further demonstrate strong agreement between our evaluators and expert judgments. Project Page: https://rain152.github. io/PersonaShot/

![](images/417b6645d7cdb693b39a48ae4f4193aaeaad3b5f03e7d27959684410a3d15878.jpg)

![](images/78b4c1f212c9f6a74a9df70d7573cca1fe3e6935696fc093883b3960d308a86c.jpg)  
Figure 1. Overview of PersonaShot’s person-centric evaluation. Top: Qualitative examples illustrating our three core dimensions: physical continuity, affective dynamics, and cinematic grammar. Bottom: Quantitative comparison across fine-grained metrics for specific state-of-the-arts, revealing their distinct capa bility profiles.

## 1. Introduction

Video generation [24, 34] is rapidly evolving from singleshot clips to multi-shot narratives [27, 33]. In these sequences, the human character serves as the core narrative anchor. A character’s physical continuity, emotional progression, and cinematic framing across cuts jointly determine sequence coherence. Inconsistencies in these aspects can break the connection between shots and undermine viewer immersion. Yet, no existing benchmark provides a dedicated evaluation of character coherence across shot boundaries.

<table><tr><td rowspan="3">Benchmark</td><td colspan="4">Input Modalities</td><td colspan="6">Evaluation Capabilities</td></tr><tr><td>Ref.</td><td>Ref.</td><td>First-</td><td>Multi-</td><td>Person-</td><td colspan="2">Physics</td><td colspan="2">Emotion</td><td>Cinematic</td></tr><tr><td>Image</td><td>Audio</td><td>Frame</td><td>Shot</td><td>Centric</td><td>Intra</td><td>Cross</td><td>Coarse</td><td>AU-Aware</td><td>Grammar</td></tr><tr><td>VBench [11]</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td></tr><tr><td>HumanVBench [39]</td><td>√</td><td>X</td><td>X</td><td>X</td><td>√</td><td>X</td><td>X</td><td>V</td><td>X</td><td>X</td></tr><tr><td>HVEval [32]</td><td>X</td><td>X</td><td>X</td><td>X</td><td>√</td><td>X</td><td>X</td><td>√</td><td>X</td><td>X</td></tr><tr><td>VideoPhy-2 [3]</td><td>√</td><td>X</td><td>X</td><td>X</td><td>X</td><td>√</td><td>X</td><td>X</td><td>X</td><td>X</td></tr><tr><td>MSVBench [21]</td><td>V</td><td>X</td><td>X</td><td>√</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td></tr><tr><td>MSAVBench [30]</td><td>V</td><td>√</td><td>X</td><td>√</td><td>X</td><td>X</td><td>X</td><td>√</td><td>X</td><td>X</td></tr><tr><td>UniVBench [29]</td><td>V</td><td>X</td><td>X</td><td>√</td><td>X</td><td>X</td><td>X</td><td>√</td><td>X</td><td>X</td></tr><tr><td>ViStoryBench [40]</td><td>V</td><td>X</td><td>X</td><td>√</td><td>√</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td></tr><tr><td>MuSS [36]</td><td>√</td><td>X</td><td>√</td><td>√</td><td>X</td><td>X</td><td>V</td><td>X</td><td>X</td><td>V</td></tr><tr><td>PersonaShot (Ours)</td><td>7</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td></tr></table>

Table 1. Comparison of PersonaShot with existing video generation benchmarks. First-Frame denotes shot-level image conditioning; Intra/Cross denote within-shot plausibility and cross-shot continuity; AU-Aware uses facial Action Units, while Cinematic Grammar covers the 180-degree rule, eyeline matching, and transition logic.

Recent benchmarks [3, 11] cover perceptual quality, motion, physical plausibility, and scene consistency, but remain insufficient for person-centric multi-shot generation. First, they mainly assess character appearance or individual-shot quality, without measuring how physical and emotional states evolve across cuts. Second, physical continuity, facial dynamics, and cinematic relations require different visual, temporal, and relational evidence, yet existing benchmarks rarely provide criterion-specific evaluation methods.

These limitations motivate the three dimensions illustrated in Fig. 1-Top: 1) Cross-shot physical continuity: existing physics benchmarks mainly assess within-shot plausibility, without determining whether character position, body state, object interaction, and relative scale remain consistent across cuts. Thus, teleportation and unexplained state changes may remain unpenalized. 2) Fine-grained affective dynamics: existing emotion metrics rely largely on coarse categories, overlooking subtle facial changes, microexpressions, and continuous emotional progression. 3) Cinematic grammar: multi-shot narratives depend on the 180- degree rule, eyeline matching, screen-direction consistency, and transition logic, which cannot be evaluated by treating shots independently. As summarized in Table 1, existing benchmarks do not jointly cover realistic input modalities and these fine-grained person-centric capabilities.

To bridge these gaps, we introduce PersonaShot, a person-centric benchmark for narrative continuity in multishot video generation. PersonaShot evaluates character coherence across within-shot states, cross-shot transitions, and sequence-level trajectories. It contains approximately 1,000 multi-shot segments and 16 metrics spanning physical continuity, affective dynamics, and cinematic grammar. We further distill structured reasoning from a large multimodal teacher into lightweight specialist evaluators, each grounded in the visual, temporal, or relational evidence required by its criterion. As shown in Fig. 1-Bottom, current models exhibit distinct capability profiles: even visually compelling outputs may contain physical-state resets, abrupt affective shifts, and broken cinematic relations. These findings demonstrate that perceptual quality alone is insufficient to measure multi-shot narrative continuity. In summary, our main contributions are as follows:

• Narrative Continuity Benchmark. We introduce the first person-centric benchmark for multi-shot video generation, comprising approximately 1,000 segments and 16 metrics across physical, affective, and cinematic dimensions.

• Human-Aligned Specialist Evaluators. We distill reasoning from a multimodal teacher into lightweight criterion-specific evaluators grounded in the visual, temporal, or relational evidence required by each metric, improving agreement with expert human judgments.

• Systematic Evaluation and Insights. We reveal distinct capability profiles across state-of-the-art models and a clear gap between perceptual quality and cross-shot narrative continuity. We commit to release the benchmark, evaluators, and code to facilitate future research.

## 2. Related Work

Video Generation Benchmarks. Early video generation benchmarks, such as VBench [11] and EvalCrafter [16], established multidimensional evaluation protocols for singleshot videos, covering perceptual quality, motion, and semantic alignment. HumanVBench [39] and HVEval [32] further introduced human-centric criteria, while recent benchmarks, including MSVBench [21], UniVBench [29], and MSAVBench [30], extended evaluation to multi-shot generation and audio-visual consistency. Despite this progress, existing benchmarks remain limited in explicitly modeling how a character’s state persists and evolves across shot boundaries. In particular, cross-shot physical continuity, AU-aware affective dynamics, and relational cinematic grammar have not been jointly evaluated within a personcentric framework. PersonaShot addresses this gap by assessing character coherence through within-shot states, cross-shot transitions, and sequence-level trajectories.

Automatic Data Processing  
![](images/fb7d4ecc0b6145dcd0817fe9f798feea6803c553e62206b2bb22fee4d490e1c7.jpg)

![](images/e2c4d0210f8e96bdb16dce66739d449288d126b8761bfa814ba4a6cfa2dea636.jpg)

![](images/938f4d1d8967476e973065f7f841dcdc7ad67f21d759746b31095f2534362966.jpg)

PersonaShot Sample  
![](images/c3759aa9d1bb6bd21d5879909e8c8524ee4df656ef7e967a782d017fdcd4c7c5.jpg)

![](images/40df02c3fdcff82c37dbacf82879877b08a76a2d8103b3345ce285728a5bf25e.jpg)  
Global Caption: A woman in a patterned blouse desperately Hierarchic tries to communicate with the somber man, … she grabs his wrist … Shot Captions: [shot 1] A woman with dark hair tied back, … Annotation looks forward with a distressed and pleading expression. [shot 2] A man in a blue collared shirt looks down …

(a) Theme Analysis  
![](images/e76e9de5166fd7cbd28a3a0cbb3aa2f5bf926ada4840388959c3f8acb8935fbb.jpg)

![](images/972889803b3a484f73fe73d5bf991809c8dccd2a8e92300553c161a5133b3a75.jpg)  
(c) Shots Distribution

![](images/f5805841d77af6fd1c859de7eecbc25ff122294dba2b29734fe58fd5f014dc67.jpg)

![](images/46e90c4b279b275949de027fbf3be9fae0bbd0d888957ec1e0f6ce31d46ce56f.jpg)  
(d) Transition Distribution

![](images/f9b0baff0ea1a6d6ea5427f1b27ff27d6009d8d8d62e9edf6866622cef67c3fe.jpg)  
(e) Prompt Word Cloud  
Figure 2. Overview of the PersonaShot benchmark. Top-Left: Our three-stage data processing pipeline filters multi-shot segments via sequence continuity, face visibility, and cross-shot relations. Top-Right: A curated sample featuring hierarchical text annotations and multi-modal conditioning inputs. Bottom: Key benchmark statistics.

Multi-Shot Video Generation. Recent multi-shot generation systems differ mainly in how narrative and shot structures are specified. Some methods rely on explicit shot-level prompts or interactive controls, including EchoShot [26], ShotStream [17], and LongLive [35]. Others combine global narratives with per-shot descriptions, such as MultiShotMaster [27], HoloCine [19], and CineTrans [33]. More automated approaches, including STAGE [37] and VGoT [38], decompose high-level stories into structured shot sequences. Although these systems offer different balances between user control and automation, they share the challenge of maintaining character coherence across cuts. We evaluate representative methods under a unified protocol to identify their strengths and limitations in physical continuity, affective dynamics, and cinematic grammar.

## 3. PersonaShot Benchmark Curation

Data Sources. Many video generation benchmarks rely on a prompt-first strategy, using LLMs to synthesize text prompts for predefined evaluation concepts [3]. While highly scalable, text prompts struggle to adequately specify and verify subtle cross-shot relations, such as fine-grained facial dynamics and spatial continuity.

In contrast, PersonaShot adopts a video-first paradigm grounded in two large-scale corpora: CineDance-1M [5], which provides professionally edited sequences with rich transition and character annotations, and OpenHuman-Vid [13], which expands action and scene diversity. We resegment these videos, track recurring characters, and rigorously filter samples to guarantee sufficient visual evidence for multi-dimensional evaluation.

Automatic Data Processing. We construct reliable multishot evaluation samples using a three-stage pipeline. The first stage forms candidate sequences, while subsequent stages verify the presence of requisite evidence for physical, affective, or cinematic evaluation. A segment lacking evidence for one dimension may still be retained for others.

• Stage 1: Sequence Segmentation and Character Continuity. We verify source shot boundaries using

TransNetV2 [22] and segment videos into sequences of 2–15 shots, ensuring each contains at least one crossshot transition. To guarantee narrative coherence, we retain only segments where at least one recurring character appears across multiple shots within a coherent local event, excluding disjointed shot collections or abrupt topic shifts.

• Stage 2: Face Visibility and Quality. AU-aware affective analysis requires clear facial observations across related shots. We uniformly sample frames from each shot and detect faces using RetinaFace [7], with a confidence threshold of 0.85 and a minimum face area of 5% of the frame. A segment is retained for affective evaluation when faces satisfying these criteria appear in at least 60% of its shots and in no fewer than two shots. We further apply CLIP-IQA [25] to remove heavily blurred or degraded face crops. The resulting face boxes, crops, and visibility records support subsequent affective annotation and evaluation.

• Stage 3: Cross-Shot Relation Selection. We leverage Qwen-based [1, 23] multimodal reasoning to verify whether adjacent shots provide the relations required by individual criteria. For physical continuity, we retain pairs in which the same character, relevant object, or shared environmental landmark remains observable, enabling evaluation of spatial relations, object-state persistence, and relative scale. For cinematic grammar, shotreverse-shot patterns support eyeline matching and 180- degree analysis, framing changes support shot progression, and hard cuts, dissolves, fades, or wipes support transition logic and rhythm.

Complete pipeline parameters, including detection thresholds, filtering criteria, and per-stage retention statistics, are detailed in Appendix B.

Hierarchical Annotation. Each retained segment receives complementary segment- and shot-level annotations generated by a multimodal teacher. The segment-level annotation captures the narrative arc, character relationships, emotional trajectory, and cinematic structure, while the shot-level annotations describe character identity, physical state, facial affect, camera framing, interactions, and scene context. They further record cross-shot changes in spatial layout, object state, relative scale, gaze direction, screen direction, and transition type. We additionally retain native audio, source prompts, per-shot first frames, and other conditioning information for diverse generation settings.

PersonaShot Statistics. PersonaShot contains 1,000 human-centric multi-shot segments, averaging 5.3 shots per segment and totaling over 5,000 annotated shots. As shown Fig. 2(bottom), the benchmark covers diverse narrative themes, sequence lengths, and framing scales, providing varied scenarios for evaluating character continuity across actions, emotional states, and interactions.

These distributions support different evaluation objectives: shorter sequences emphasize local cross-shot transitions, while longer sequences enable sequence-level trajectory analysis. Close and medium shots provide detailed facial evidence for affective evaluation, whereas wider views preserve the spatial and geometric cues required for physical continuity and cinematic grammar.

## 4. Fine-grained Evaluation Metrics

## 4.1. Overview

Existing video benchmarks mainly focus on perceptual quality, semantic alignment, or short-range consistency, providing limited support for failures emerging across editing cuts [3, 4, 8]. We therefore introduce PersonaShot-AutoEval, a person-centric evaluation protocol for multishot video generation, organized along three temporal levels: within-shot states, cross-shot transitions, and sequencelevel trajectories. The protocol contains 16 sub-metrics across four dimensions: visual quality and consistency, causal physical continuity, fine-grained affective dynamics, and cinematic grammar.

Given a generated sequence with segment- and shot-level annotations, we sample five representative shots along the timeline. Individual shots assess local character states, adjacent shot pairs evaluate cross-shot relations, and the full sequence measures narrative trajectories. Since different criteria require different evidence, PersonaShot-AutoEval combines conventional perceptual models with three specialist evaluators that analyze physical states, affective dynamics, and cinematic structures using criterion-specific visual, facial, and temporal evidence.

## 4.2. Visual Quality and Consistency

Dimension and Strategy. This foundational dimension assesses visual stability through four criteria: 1) Visual Fidelity (VF), 2) Text-Video Alignment (TVA), 3) Identity Consistency (ID), and 4) Scene/Style Consistency (SS). Following established protocols, we evaluate VF using DOVER and MUSIQ [12, 31], TVA using VQAScore and GroundingDINO [14, 15], and ID/SS via cross-shot embedding and perceptual similarities.

## 4.3. Causal Physical Continuity

Dimension. Existing physical benchmarks mainly evaluate whether individual frames or short clips satisfy local physical plausibility [3, 4]. However, locally plausible shots may still form inconsistent physical processes across editing cuts. This dimension evaluates cross-shot physical evolution through three criteria: 1) Spatial Layout Continuity (SLC): measuring whether character and object positions remain consistent across shots; 2) Object State Persistence (OSP): evaluating whether object states evolve in a causally consistent manner; and 3) Geometric Scale Consistency (GSC): measuring whether character-object proportions and scene geometry remain stable across viewpoints.

![](images/a6502333ab6da2d7aea8ac4e855a806d446355ec3f51404e1018ded614dca431.jpg)  
Figure 3. Overview of PersonaShot evaluation framework. Given generated multi-shot videos and prompts, PersonaShot extends conventional quality assessment with three specialist dimensions for narrative-level evaluation: causal physical continuity, affective dynamics, and cinematic grammar.

Evaluation Strategy. To ensure genuine cross-shot reasoning rather than parsing ground-truth texts, we develop a specialist physical evaluator via teacher-guided distillation with strict input isolation. During training, a large multimodal teacher, Qwen3.5-397B-A17B [23], receives visual evidence and structural annotations A to generate structured targets y<sub>teacher</sub>. The compact Qwen3.5-4B evaluator $f _ { \theta }$ (with LoRA adaptation [1, 10]) is optimized via:

$$
\begin{array} { r } { \mathcal { L } _ { d i s t i l l } = \mathcal { L } _ { C E } \big ( f _ { \theta } ( \mathcal { V } _ { g e n } , \mathcal { P } _ { t e x t } ) , y _ { t e a c h e r } \big ) , } \end{array}\tag{1}
$$

where annotations A are strictly isolated during inference, ensuring the evaluator relies solely on generated visual sequences $\nu _ { g e n }$ and prompts $\mathcal { P } _ { t e x t }$ (more experimental details and training splits are deferred to Appendix B).

## 4.4. Affective Dynamics

Dimension. Existing human-centric benchmarks commonly evaluate emotions using coarse categorical labels [6]. However, such labels cannot capture subtle facial variations, temporal evolution, or narrative appropriateness of affective states. This dimension models emotional continuity through four criteria: 1) Expression Naturalness (EN): evaluating facial realism and generation artifacts; 2) Facial-Action Temporal Coherence (FATC):

capturing facial-action stability and abrupt or static expressions; 3) Emotion-Narrative Alignment (ENA): measuring whether emotional intensity and category match the current event; and 4) Emotional Arc Coherence (EAC): evaluating whether affective states evolve coherently throughout the narrative.

Evaluation Strategy. We adapt Qwen3.5-4B as an affective evaluator using emotion data from Emotion-LLaMA [6], trained via a similar distillation objective. To eliminate evaluation circularity, inference relies strictly on the generated video and extracted features rather than ground-truth labels. Specifically, we provide frame-level Action Unit (AU) signals extracted by OpenFace and formulate AU-aware evaluation instructions based on the Facial Action Coding System [2]:

$$
S _ { a f f } = f _ { \theta } \big ( \mathcal { V } _ { g e n } , \mathcal { P } _ { t e x t } , \mathcal { F } _ { A U } \big ) ,\tag{2}
$$

where ${ \mathcal { F } } _ { A U }$ denotes the frame-level AU temporal dynamics, enabling the evaluator to jointly reason over facial observations and narrative context without annotation leakage (see Appendix B for further implementation specifics).

## 4.5. Cinematic Grammar

Dimension. Existing benchmarks [21, 30] may evaluate camera motion or shot-level properties, but rarely measure whether generated sequences follow established cinematic conventions [28]. Cinematic grammar therefore evaluates five complementary aspects: 1) Directorial Narrative Sequencing (DNS): assessing whether a global prompt is translated into a coherent shot-level narrative structure; 2) Shot Transition Appropriateness (ST): determining whether transition types match the narrative context; 3) Transition Rhythm Alignment (TRA): measuring whether shot duration and cutting density correspond to narrative pacing; 4) 180-degree Rule Compliance (180R): examining whether screen direction and action axes remain consistent across cuts; and 5) Eyeline Match Correctness (EM): evaluating whether gaze relations are spatially coherent between shots.

Evaluation Strategy. We model cinematic grammar through a two-stage process consisting of cinematic analysis and narrative reasoning. First, we employ cinematographic video captioning models [18] to extract professional film-language descriptions, including shot scale, camera perspective, composition, and editing cues, from generated sequences. Shot boundaries and transition structures are further identified using TransNetV2 [22]. Second, a cinematic reasoner evaluates whether the extracted cinematic structures satisfy narrative and editing principles. DNS is evaluated by comparing generated shot structures with the intended global narrative, while ST and TRA are assessed by jointly considering transition patterns, shot duration, and narrative context. Spatial cinematic rules, including 180R and EM, are evaluated through character orientation, screen direction, and gaze consistency across adjacent shots.

## 4.6. Score Aggregation

The heterogeneous metric outputs are first converted into higher-is-better scores and normalized to [0, 1] using metric-specific calibration functions. The overall PersonaShot score is computed as:

$$
S _ { \mathrm { P S } } = \alpha _ { \mathrm { v i s } } S ^ { \mathrm { v i s } } + \alpha _ { \mathrm { p h y } } S ^ { \mathrm { p h y } } + \alpha _ { \mathrm { a f f } } S ^ { \mathrm { a f f } } + \alpha _ { \mathrm { c i n } } S ^ { \mathrm { c i n } } ,\tag{3}
$$

where $S ^ { \mathrm { v i s } } , \ S ^ { \mathrm { p h y } } , \ S ^ { \mathrm { a f f } }$ , and $S ^ { \mathrm { c i n } }$ denote visual quality and consistency, causal physical continuity, fine-grained affective dynamics, and cinematic grammar scores, respectively. The coefficients $\alpha _ { \mathrm { v i s } } , \alpha _ { \mathrm { p h y } } , \alpha _ { \mathrm { a f f } }$ , and $\alpha _ { \mathrm { c i n } }$ are fixed benchmark hyperparameters that balance foundational visual quality with the three narrative continuity dimensions.

## 5. Experiments

## 5.1. Experimental Setup

We evaluate representative multi-shot video generation systems on the same 1,000 benchmark samples. The compared methods cover both shot-level prompt conditioning and global-prompt conditioning, including systems with explicit shot planning and those generating shots independently.

Our benchmark comprises 16 sub-metrics in total across four dimensions. For a fair comparison, Table 2 reports the 15 common metrics shared across all settings, while sequence-level Directorial Narrative Sequencing (DNS) is evaluated specifically for global-prompt narratives and discussed in Finding 5. All metric scores are normalized to [0, 1] before aggregation. Following expert preference voting and benchmark design considerations, we assign equal weights to the four dimensions, i.e., $\alpha _ { \mathrm { v i s } } = \alpha _ { \mathrm { p h y } } = \alpha _ { \mathrm { a f f } } =$ $\alpha _ { \mathrm { c i n } } ~ = ~ 0 . 2 5$ . More evaluation details are shown in Appendix C.

## 5.2. Main Results and Insights

Table 2 summarizes the fine-grained evaluation results of representative multi-shot video generation systems. Beyond the overall ranking, PersonaShot reveals several important characteristics of current models.

Finding 1: Perceptual quality does not guarantee narrative continuity. Modern video generators have achieved strong visual fidelity, but substantial gaps remain in maintaining character-centric coherence across shots. For example, ShotStream obtains high visual fidelity yet a lower aggregate score due to limited performance in affective and cinematic dimensions. Similarly, several models with competitive visual quality exhibit significant degradation in physical state persistence and emotional evolution, indicating that conventional evaluation can overestimate multishot generation capabilities.

Finding 2: Explicit shot planning improves structural coherence, but global narrative reasoning remains essential. Models with shot-level conditioning generally achieve stronger cross-shot consistency via explicit guidance. Seedance achieves the leading overall performance under both settings. However, global-prompt systems such as LTX-2.3 remain competitive in cinematic grammar and identity, suggesting that effective narrative planning matters more than prompt granularity alone.

Finding 3: Physical and affective state propagation remain major challenges. Although current models generate visually plausible individual shots, they struggle to preserve latent character and scene states across editing boundaries. Physical continuity scores reveal frequent failures in object-state persistence and spatial consistency, while affective evaluation exposes unstable facial dynamics and weak emotional trajectories throughout multi-shot sequences.

Finding 4: Cinematic grammar is largely undermodeled in existing video generators. PersonaShot further reveals that visually coherent shots do not necessarily form cinematographically valid sequences. Many systems struggle with transition appropriateness, rhythm alignment, 180-degree rule compliance, and eyeline matching, indicating that current models primarily optimize visual synthesis rather than film-language reasoning.

Finding 5: Global narrative sequencing reveals a stark divide in long-form cinematic control. Evaluating globalprompt settings allows us to isolate Directorial Narrative Sequencing (DNS), measuring whether a global prompt translates into a coherent structure. Here, Seedance achieves 0.865 in DNS, whereas open-source LTX-2.3 and VGoT achieve 0.624 and 0.535. This performance gap highlights that mastering long-form narrative planning remains a key bottleneck for open-source models.

<table><tr><td rowspan="2">Method</td><td colspan="4">Visual Quality and Consistency</td><td colspan="3">Causal Physical Continuity</td><td colspan="3">Affective Dynamics</td><td colspan="3">Cinematic Grammar</td><td rowspan="2"></td><td rowspan="2">Aggregate</td></tr><tr><td>VF</td><td>TVA</td><td>ID</td><td>SS</td><td>SLC</td><td>OSP</td><td>GSC</td><td>EN</td><td>FATC</td><td>ENA</td><td>EAC ST</td><td>TRA</td><td>180R EM</td></tr><tr><td colspan="10">Shot-Level Prompt Conditioning</td><td colspan="3"></td><td colspan="3"></td></tr><tr><td>Seedance</td><td>0.981</td><td>0.752</td><td>0.673</td><td>0.762</td><td>0.778</td><td>0.928</td><td>0.947</td><td>0.907</td><td>0.562</td><td>0.778</td><td>0.617 0.837</td><td>0.658</td><td>0.818</td><td>0.813</td><td>0.793</td></tr><tr><td>HoloCine</td><td>0.853</td><td>0.797</td><td>0.582</td><td>0.718</td><td>0.647</td><td>0.862</td><td>0.893</td><td>0.792</td><td>0.384</td><td>0.677</td><td>0.3430.723</td><td>0.517</td><td>0.712</td><td>0.687</td><td>0.687</td></tr><tr><td>STAGE</td><td>0.827</td><td>0.703</td><td>0.507</td><td>0.752</td><td>0.623</td><td>0.818</td><td>0.852</td><td>0.718</td><td>0.453</td><td>0.628</td><td>0.443 0.677</td><td>0.503</td><td>0.667</td><td>0.642</td><td>0.661</td></tr><tr><td>LTX-2.3</td><td>0.892</td><td>0.718</td><td>0.572</td><td>0.733</td><td>0.677</td><td>0.832</td><td>0.857</td><td>0.758</td><td>0.427</td><td>0.583</td><td>0.403 0.683</td><td>0.512</td><td>0.683</td><td>0.648</td><td>0.673</td></tr><tr><td>MultiShotMaster</td><td>0.849</td><td>0.697</td><td>0.557</td><td>0.653</td><td>0.604</td><td>0.852</td><td>0.884</td><td>0.781</td><td>0.458</td><td>0.596</td><td>0.442 0.622</td><td>0.482</td><td>0.623</td><td>0.603</td><td>0.655</td></tr><tr><td>LongLive</td><td>0.972</td><td>0.458</td><td>0.532</td><td>0.753</td><td>0.697</td><td>0.697</td><td>0.803</td><td>0.687</td><td>0.383</td><td>0.418</td><td>0.3480.684</td><td>0.468</td><td>0.697</td><td>0.687</td><td>0.626</td></tr><tr><td>ShotStream</td><td>0.994</td><td>0.377</td><td>0.548</td><td>0.673</td><td>0.668</td><td>0.573</td><td>0.742</td><td>0.677</td><td>0.387</td><td>0.383</td><td>0.367 0.688</td><td>0.497</td><td>0.692</td><td>0.693</td><td>0.601</td></tr><tr><td>CineTrans</td><td>0.795</td><td>0.557</td><td>0.538</td><td>0.682</td><td>0.572</td><td>0.697</td><td>0.708</td><td>0.613</td><td>0.378</td><td>0.503</td><td>0.3580.594</td><td>0.537</td><td>0.605</td><td>0.587</td><td>0.586</td></tr><tr><td>EchoShot</td><td>0.937</td><td>0.507</td><td>0.463</td><td>0.488</td><td>0.482</td><td>0.758</td><td>0.871</td><td>0.802</td><td>0.453</td><td>0.448</td><td>0.432 0.503</td><td>0.478</td><td>0.488</td><td>0.482</td><td>0.581</td></tr><tr><td colspan="10">Global-Prompt Conditioning</td><td colspan="3"></td><td colspan="3"></td><td></td></tr><tr><td>Seedance</td><td>0.973</td><td>0.712</td><td>0.642</td><td>0.733</td><td>0.738</td><td>0.908</td><td>0.937</td><td>0.882</td><td>0.532</td><td>0.743</td><td>0.588 0.808</td><td>0.627</td><td>0.788</td><td>0.783</td><td>0.766</td></tr><tr><td>LTX-2.3</td><td>0.818</td><td>0.523</td><td>0.697</td><td>0.823</td><td>0.697 0.703</td><td></td><td>0.758</td><td>0.683</td><td>0.403</td><td>0.453</td><td>0.3480.697</td><td>0.472</td><td>0.703</td><td>0.677</td><td>0.636</td></tr><tr><td>VGoT</td><td>0.988</td><td>0.418</td><td>0.683</td><td>0.648</td><td>0.627</td><td>0.608</td><td>0.881</td><td>0.868</td><td>0.388</td><td>0.477</td><td>0.357 0.643</td><td>0.557</td><td>0.682</td><td>0.683</td><td>0.638</td></tr></table>

Table 2. Fine-grained evaluation results on PersonaShot under shot-level and global-prompt conditioning. Higher is better. The aggregate score equally averages the four dimension scores, with metrics averaged within each dimension. Within each protocol, the best and second-best results are highlighted in bold and underlined, respectively.

## 5.3. Failure Case Analysis

Beyond quantitative comparisons, PersonaShot provides interpretable diagnoses of multi-shot generation failures. As illustrated in Fig. 4, existing models frequently fail to preserve object states and spatial relations across shots, maintain coherent emotional evolution, and follow basic cinematic conventions such as eyeline matching and transition consistency. These failures are often visually plausible at the individual-shot level but become apparent when evaluated across temporal and relational dimensions. The results highlight that reliable multi-shot generation requires persistent reasoning over physical states, affective trajectories, and cinematic structures beyond conventional visual quality.

## 5.4. Human Study and Evaluator Validation

To assess whether the proposed evaluators reflect human judgments, we conduct an expert study on 25 diverse generated multi-shot sequences. Ten domain experts participate under a sparse assignment protocol, with each sequence independently rated by four annotators using criterionspecific five-point rubrics. The ratings are averaged into Mean Opinion Scores (MOS), and the resulting Krippendorff’s $\alpha = 0 . 7 6$ indicates substantial inter-rater agreement. We measure alignment using Spearman correlation (ρ) and pairwise ranking agreement. Complete annotation rubrics, valid sample counts, confidence intervals, reliability statistics, and results for all 12 specialist criteria are provided in the Appendix E.

<table><tr><td>Dimension</td><td>Criterion</td><td>ρ↑</td><td>Pair.↑</td></tr><tr><td rowspan="3">Physical</td><td>Spatial Layout (SL)</td><td>0.68</td><td>71.4%</td></tr><tr><td>Object State (OSP)</td><td>0.74</td><td>76.0%</td></tr><tr><td>Aggregate</td><td>0.75</td><td>75.2%</td></tr><tr><td rowspan="3">Affective</td><td>Emotion Alignment (ENA)</td><td>0.71</td><td>72.5%</td></tr><tr><td>Facial Dynamics (FATC)</td><td>0.68</td><td>64.8%</td></tr><tr><td>Aggregate</td><td>0.69</td><td>70.3%</td></tr><tr><td rowspan="3">Cinematic</td><td>Shot Transition (ST)</td><td>0.72</td><td>73.1%</td></tr><tr><td>Eyeline Match (EM)</td><td>0.65</td><td>67.2%</td></tr><tr><td>Aggregate</td><td>0.73</td><td>72.8%</td></tr></table>

Table 3. Human alignment of PersonaShot evaluators. We report Spearman correlation $( \rho )$ with human Mean Opinion Scores $( N = 2 5 )$ and pairwise agreement (Pair.) for representative criteria. Full results are in the Appendix E.

As shown in Table 3, the proposed evaluators consistently align with human judgments across all three specialist dimensions. The aggregated physical, affective, and cinematic scores achieve Spearman correlations of 0.75, 0.69, and 0.73, respectively, with pairwise agreement between 70.3% and 75.2% (all $p ~ < ~ 0 . 0 1 )$ Object State Persistence shows strong alignment $( \rho = 0 . 7 4 )$ , reflecting its explicit cross-shot evidence. Emotion-Narrative Alignment also correlates well with human ratings $( \rho ~ = ~ 0 . 7 1 )$ , indicating sensitivity to the narrative appropriateness of facial affect. Facial-Action Temporal Coherence is comparatively more challenging because subtle facial changes are inherently ambiguous. The cinematic results further show that transition and eyeline relations can be evaluated consistently, supporting the complementary diagnostic value of the three dimensions.

![](images/fad60c280a78846580c470a4dbaab1dd0f16463fea16c8b50aab99386cd68cec.jpg)  
Figure 4. Qualitative failure cases revealed by PersonaShot. The examples illustrate representative failures in causal physical continuity, affective dynamics, and cinematic grammar, including object state persistence failures, spatial layout discontinuities, emotion evolution inconsistencies, facial expression issues, eyeline matching failures, and inappropriate shot transitions.

## 6. Conclusion

We introduced PersonaShot, the first person-centric benchmark for evaluating narrative continuity in multi-shot video generation. Beyond conventional perceptual quality assessment, PersonaShot systematically measures three critical dimensions of character-centric storytelling: causal physical continuity, fine-grained affective dynamics, and cinematic grammar. Through 16 complementary metrics and specialist evaluators aligned with human judgments, our benchmark reveals that current video generators remain limited in preserving persistent character states, modeling emotional evolution, and following cinematic conventions across shots.

Our findings highlight a key direction for future multishot generation: moving beyond isolated visually plausible shots toward coherent narrative systems that explicitly model physical states, affective trajectories, and cinematic structures. Furthermore, by providing lightweight yet human-aligned specialist evaluators, our framework offers a reliable and interpretable recipe for iterative model diagnosis. We hope PersonaShot can serve as both a rigorous testbed and a catalyst for developing next-generation video foundation models with authentic storytelling capabilities.

## References

[1] Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, et al. Qwen technical report. arXiv preprint arXiv:2309.16609, 2023. 4, 5

[2] Tadas Baltrusaitis, Peter Robinson, and Louis-Philippe ˇ Morency. Openface: an open source facial behavior anal-

ysis toolkit. In 2016 IEEE winter conference on applications of computer vision (WACV), pages 1–10. IEEE, 2016. 5, 13

[3] Hritik Bansal, Zongyu Lin, Tianyi Xie, Zeshun Zong, Michal Yarom, Yonatan Bitton, Chenfanfu Jiang, Yizhou Sun, Kai Wei Chang, and Aditya Grover. Videophy: Evaluating physical commonsense for video generation. In International Conference on Learning Representations, pages 102075– 102121, 2025. 2, 3, 4, 11

[4] Hritik Bansal, Clark Peng, Yonatan Bitton, Roman Goldenberg, Aditya Grover, and Kai-Wei Chang. Videophy-2: A challenging action-centric physical commonsense evaluation in video generation. arXiv preprint arXiv:2503.06800, 2025. 4, 11

[5] Yuheng Chen, Teng Hu, Yuji Wang, Qingdong He, Zhucun Xue, Qianyu Zhou, Xiangtai Li, Lizhuang Ma, Jiangning Zhang, and Dacheng Tao. Cinedance: Towards nextgeneration multi-shot long-form cinematic audio-video gen eration. arXiv preprint arXiv:2606.09639, 2026. 3, 11

[6] Zebang Cheng, Zhi-Qi Cheng, Jun-Yan He, Jingdong Sun, Kai Wang, Yuxiang Lin, Zheng Lian, Xiaojiang Peng, and Alexander G Hauptmann. Emotion-llama: Multimodal emotion recognition and reasoning with instruction tuning. Advances in Neural Information Processing Systems, 37: 110805–110853, 2024. 5, 11

[7] Jiankang Deng, Jia Guo, Evangelos Ververas, Irene Kotsia, and Stefanos Zafeiriou. Retinaface: Single-shot multilevel face localisation in the wild. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5203–5212, 2020. 4, 12

[8] Xu Guo, Fulong Ye, Qichao Sun, Liyang Chen, Bingchuan Li, Pengze Zhang, Jiawei Liu, Songtao Zhao, Qian He, and Xiangwang Hou. Dreamid-omni: Unified framework for controllable human-centric audio-video generation. arXiv preprint arXiv:2602.12160, 2026. 4

[9] Yoav HaCohen, Benny Brazowski, Nisan Chiprut, Yaki Bitterman, Andrew Kvochko, Avishai Berkowitz, Daniel Shalem, Daphna Lifschitz, Dudu Moshe, Eitan Porat, et al. Ltx-2: Efficient joint audio-visual foundation model. arXiv preprint arXiv:2601.03233, 2026. 11

[10] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Liang Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. Iclr, 1(2):3, 2022. 5, 13

[11] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, et al. Vbench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21807–21818, 2024. 2, 11

[12] Junjie Ke, Qifei Wang, Yilin Wang, Peyman Milanfar, and Feng Yang. Musiq: Multi-scale image quality transformer. In Proceedings of the IEEE/CVF international conference on computer vision, pages 5148–5157, 2021. 4

[13] Hui Li, Mingwang Xu, Yun Zhan, Shan Mu, Jiaye Li, Kaihui Cheng, Yuxuan Chen, Tan Chen, Mao Ye, Jingdong Wang, et al. Openhumanvid: A large-scale high-quality dataset for enhancing human-centric video generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 7752–7762, 2025. 3, 11

[14] Zhiqiu Lin, Deepak Pathak, Baiqi Li, Jiayao Li, Xide Xia, Graham Neubig, Pengchuan Zhang, and Deva Ramanan. Evaluating text-to-visual generation with image-to-text generation. In European Conference on Computer Vision, pages 366–384. Springer, 2024. 4

[15] Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Qing Jiang, Chunyuan Li, Jianwei Yang, Hang Su, et al. Grounding dino: Marrying dino with grounded pre-training for open-set object detection. In European conference on computer vision, pages 38–55. Springer, 2024. 4

[16] Yaofang Liu, Xiaodong Cun, Xuebo Liu, Xintao Wang, Yong Zhang, Haoxin Chen, Yang Liu, Tieyong Zeng, Raymond Chan, and Ying Shan. Evalcrafter: Benchmarking and evaluating large video generation models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 22139–22149, 2024. 2, 11

[17] Yawen Luo, Xiaoyu Shi, Junhao Zhuang, Yutian Chen, Quande Liu, Xintao Wang, Pengfei Wan, and Tianfan Xue. Shotstream: Streaming multi-shot video generation for interactive storytelling. arXiv preprint arXiv:2603.25746, 2026. 3, 11

[18] Xinyu Mao, Yuhui Zeng, Xiaokun Liu, Wenyu Qin, Meng Wang, Xin Tao, Pengfei Wan, Xiaohan Xing, and Max Meng. Cinecap: Structured reasoning with spatio-temporal anchors for cinematographic video captioning. arXiv preprint arXiv:2606.24636, 2026. 6, 11

[19] Yihao Meng, Hao Ouyang, Yue Yu, Qiuyu Wang, Wen Wang, Ka Leong Cheng, Hanlin Wang, Shuailei Ma, Yixuan Li, Cheng Chen, et al. Holocine: Holistic generation of cinematic multi-shot long video narratives. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 461–471, 2026. 3, 11

[20] Team Seedance, De Chen, Liyang Chen, Xin Chen, Ying Chen, Zhuo Chen, Zhuowei Chen, Feng Cheng, Tianheng Cheng, Yufeng Cheng, et al. Seedance 2.0: Advancing video generation for world complexity. arXiv preprint arXiv:2604.14148, 2026. 11

[21] Haoyuan Shi, Yunxin Li, Nanhao Deng, Zhenran Xu, Xinyu Chen, Longyue Wang, Baotian Hu, and Min Zhang. Msvbench: Towards human-level evaluation of multi-shot video generation. In Findings of the Association for Computational Linguistics: ACL 2026, pages 24034–24058, 2026. 2, 3, 5, 11

[22] Tomas Soucek and Jakub Lokoc. Transnet v2: An effective´ deep network architecture for fast shot transition detection. In Proceedings of the 32nd ACM International Conference on Multimedia, pages 11218–11221, 2024. 4, 6, 11

[23] Qwen Team. Qwen3. 5-omni technical report. arXiv preprint arXiv:2604.15804, 2026. 4, 5, 11, 13

[24] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025. 1

[25] Jianyi Wang, Kelvin CK Chan, and Chen Change Loy. Exploring clip for assessing the look and feel of images. In Proceedings of the AAAI conference on artificial intelligence, pages 2555–2563, 2023. 4, 13

[26] Jiahao Wang, Hualian Sheng, Sijia Cai, Weizhan Zhang, Caixia Yan, Yachuang Feng, Bing Deng, and Jieping Ye. Echoshot: Multi-shot portrait video generation. Advances in Neural Information Processing Systems, 38:22058–22090, 2026. 3, 11

[27] Qinghe Wang, Xiaoyu Shi, Baolu Li, Weikang Bian, Quande Liu, Huchuan Lu, Xintao Wang, Pengfei Wan, Kun Gai, and Xu Jia. Multishotmaster: A controllable multi-shot video generation framework. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 16268–16278, 2026. 1, 3, 11

[28] Xinran Wang, Songyu Xu, Shan Xiangxuan, Yuxuan Zhang, Muxi Diao, Xueyan Duan, Kongming Liang, Zhanyu Ma, et al. Cinetechbench: A benchmark for cinematographic technique understanding and generation. Advances in Neural Information Processing Systems, 38, 2026. 5

[29] Jianhui Wei, Xiaotian Zhang, Yichen Li, Yuan Wang, Yan Zhang, Ziyi Chen, Zhihang Tang, Wei Xu, and Zuozhu Liu. Univbench: Towards unified evaluation for video foundation models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 25654– 25666, 2026. 2, 3, 11

[30] Yujie Wei, Yujin Han, Zhekai Chen, Yongming Li, Kaixun Jiang, Zhihang Liu, Quanhao Li, Zhiwu Qing, Xiang Wang, Zhen Xing, et al. Msavbench: Towards comprehensive and reliable evaluation of multi-shot audio-video generation. arXiv preprint arXiv:2605.20183, 2026. 2, 3, 5, 11

[31] Haoning Wu, Erli Zhang, Liang Liao, Chaofeng Chen, Jing wen Hou, Annan Wang, Wenxiu Sun, Qiong Yan, and Weisi Lin. Exploring video quality assessment on user generated contents from aesthetic and technical perspectives. In Pro-

ceedings of the IEEE/CVF international conference on computer vision, pages 20144–20154, 2023. 4

[32] Sijing Wu, Yunhao Li, Huiyu Duan, Yanwei Jiang, Yucheng Zhu, and Guangtao Zhai. Hveval: Towards unified evaluation of human-centric video generation and understanding. In Proceedings of the 33rd ACM International Conference on Multimedia, pages 13376–13383, 2025. 2, 3, 11

[33] Xiaoxue Wu, Bingjie Gao, Yu Qiao, Yaohui Wang, and Xinyuan Chen. Cinetrans: Learning to generate videos with cinematic transitions via masked diffusion models, 2025. 1, 3, 11

[34] Zhen Xing, Qijun Feng, Haoran Chen, Qi Dai, Han Hu, Hang Xu, Zuxuan Wu, and Yu-Gang Jiang. A survey on video diffusion models. ACM Computing Surveys, 57(2):1–42, 2024. 1

[35] Shuai Yang, Wei Huang, Ruihang Chu, Yicheng Xiao, Yuyang Zhao, Xianbang Wang, Muyang Li, Enze Xie, Yingcong Chen, Yao Lu, et al. Longlive: Real-time interactive long video generation. arXiv preprint arXiv:2509.22622, 2025. 3, 11

[36] Haojie Zhang, Di Wu, Bingyan Liu, Linjie Zhong, Yuancheng Wei, Xingsong Ye, Nanqing Liu, and Yaling Liang. Muss: A large-scale dataset and cinematic narrative benchmark for multi-shot subject-to-video generation. arXiv preprint arXiv:2604.23789, 2026. 2, 11

[37] Peixuan Zhang, Zijian Jia, Kaiqi Liu, Shuchen Weng, Si Li, and Boxin Shi. Stage: Storyboard-anchored generation for cinematic multi-shot narrative. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 659–669, 2026. 3, 11

[38] Mingzhe Zheng, Yongqi Xu, Haojian Huang, Xuran Ma, Yexin Liu, Wenjie Shu, Yatian Pang, Feilong Tang, Qifeng Chen, Harry Yang, et al. Videogen-of-thought: Step-by-step generating multi-shot video with minimal manual intervention. arXiv preprint arXiv:2412.02259, 2024. 3, 11

[39] Ting Zhou, Daoyuan Chen, Qirui Jiao, Bolin Ding, Yaliang Li, and Ying Shen. Humanvbench: Exploring human-centric video understanding capabilities of mllms with synthetic benchmark data. arXiv e-prints, pages arXiv–2412, 2024. 2, 3, 11

[40] Cailin Zhuang, Ailin Huang, Yaoqi Hu, Jingwei Wu, Wei Cheng, Jiaqi Liao, Hongyuan Wang, Xinyao Liao, Weiwei Cai, Hengyuan Xu, et al. Vistorybench: Comprehensive benchmark suite for story visualization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9455–9467, 2026. 2, 11

## Supplementary Material

## A. Detailed Related Work

## A.1. Video Generation Benchmarks

Early video generation benchmarks primarily focused on general-purpose, single-shot scenarios, establishing comprehensive hierarchical dimensions for isolated clips [11, 16]. Concurrently, specialized person-centric benchmarks, such as HVEval [32] and HumanVBench [39], began to address human-specific criteria, though they evaluate either single-shot generation quality or video understanding capabilities, without measuring cross-shot character coherence. As generation capabilities advanced, recent frameworks have shifted toward multi-shot evaluations. Benchmarks including MSVBench [21], UniVBench [29], and MSAVBench [30] introduced metrics for narrative consistency, agent-based scoring, and audio-visual synchronization to assess longer, multi-cut sequences. MuSS [36] introduces cinematic narrative evaluation but focuses on subjectto-video generation without person-centric cross-shot analysis. ViStoryBench [40] evaluates story visualization quality but does not assess character state continuity across cuts.

However, these multi-shot frameworks lack fine-grained analysis in dimensions crucial for cross-shot coherence. Specifically, they overlook physical rationality across shot boundaries, reduce emotional evaluation to coarse categorical labels (e.g., basic emotion categories as in Emotion-LLaMA [6]) rather than tracking micro-expression dynamics, and entirely ignore rule-based cinematic grammar. To bridge these gaps, PersonaShot provides a unified, personcentric benchmark that integrates cross-shot physical consistency, Action Unit (AU)-level micro-expression analysis, and cinematic grammar quantification.

## A.2. Multi-Shot Video Generation

Multi-shot video generation has flourished recently, driven by both open-source efforts and commercial systems. Models differ primarily in how much structure users must provide. In manual per-shot generation, users write a dedicated prompt for each shot: EchoShot [26] accepts structured shot-level captions, while ShotStream [17] and LongLive [35] support streaming interactive input with KVcache continuity. Global plus per-shot methods, including MultiShotMaster [27], HoloCine [19], and CineTrans [33], let users supply an overarching scene description alongside per-shot captions. Auto-decomposition approaches, STAGE [37] and VGoT [38], accept only a high-level story theme or a single sentence and automatically generate a structured storyboard via a director agent or GPT-4o-based planning. Flexible-conditioning models such as LTX-2.3 [9] and Seedance [20] adapt to both shot-level and global-prompt settings, handling shot decomposition either internally or guided by explicit per-shot prompts. We benchmark these representative methods under a unified protocol to assess current multi-shot capabilities and identify key directions for future improvements.

## A.3. Evaluation with Vision-Language Models

Scaling evaluation beyond human annotation relies increasingly on Vision-Language Models (VLMs). However, zero-shot VLMs exhibit surface-level bias, conflating visual quality with structural correctness and missing specific cross-shot violations [3]. Task-specific fine-tuning mitigates this, as seen in VideoPhy-2 [4] for physical commonsense, Emotion-LLaMA [6] for emotion recognition, and CineCap [18] for cinematic captioning. PersonaShot advances this paradigm via teacher-guided distillation: we train multiple lightweight specialized evaluators from a Qwen3.5-397B-A17B teacher [23], each grounded in the criterion-specific evidence required by its metric—visual and temporal cues for the physical continuity evaluator, AUsignaled facial dynamics for the affective evaluator, and structured cinematic descriptions for the two-stage cinematic reasoner. Validated by an expert study with 10 domain specialists, these distilled evaluators achieve substantial alignment with human judgments (Spearman ρ up to 0.75, all $p < 0 . 0 1 )$ ), delivering scalable, expert-level benchmarking.

## B. Data Processing & Evaluator Training Details

In this section, we provide the complete technical implementation details for PersonaShot, including the automated data curation pipeline and its retention statistics, as well as the training datasets, hyperparameters, and inference protocols for our specialist evaluators.

## B.1. Automatic Data Processing & Pipeline Details

To construct PersonaShot, we source raw video sequences from two large-scale corpora: CineDance-1M [5] and OpenHumanVid [13]. A three-stage automated filtering pipeline is applied to select segments with sufficient visual, facial, and relational evidence for person-centric multi-shot evaluation.

Stage 1: Sequence Segmentation and Character Continuity. We use TransNet V2 [22] to detect shot transitions and slice raw videos into candidate multi-shot segments ranging from 2 to 15 shots (with an average length of 5.3 shots). To enforce narrative coherence, we require recurring human anchor identities: segments are retained only if at least one primary character appears across multiple shots within a continuous local event.

Table 4. LoRA Fine-Tuning Configurations for Specialist Evaluators. Training setup and hyperparameters for adapting Qwen3-4B and Qwen3-8B on the 4,041-sample MERR training set.
<table><tr><td rowspan="5"></td><td>Hyperparameter Base Model Backbone</td><td>Qwen3-4B Evaluator</td><td>Qwen3-8B Evaluator</td></tr><tr><td>Training / Validation Samples Target Emotion Categories</td><td> $Q \mathrm { w e n } / Q \mathrm { w e n } 3 - 4 \mathrm { B }$  4,041 / 446 (MERR dataset) 9 classes</td><td> $\mathsf { Q w e n / Q w e n 3 - 8 B }$ </td></tr><tr><td>LoRA Rank (r)</td><td>16</td><td>32</td></tr><tr><td>LoRA Alpha (α) Trainable Parameters</td><td>32 33M (0.81%)</td><td>64</td></tr><tr><td>Training Epochs Learning Rate</td><td>3  $2 \times 1 0 ^ { - 4 }$ </td><td>66M (~0.80%) 10</td></tr><tr><td></td><td>Batch Size (per device) Training Duration</td><td>4 ~33 min</td><td> $1 \times 1 0 ^ { - 4 }$  4 ~42 min</td></tr><tr><td>Dimension</td><td></td><td></td><td></td></tr><tr><td>Visual Quality and Consistency</td><td>Metric</td><td>Definition</td><td>Evaluation Strategy</td></tr><tr><td>Visual Quality and Consistency</td><td>Visual Fidelity (VF)</td><td>Measures perceptual quality,</td><td></td></tr><tr><td rowspan="5"></td><td></td><td>artifacts, distortions, and composition.</td><td>DOVER aesthetic/technical assessment and MUSIQ frame-level quality evaluation.</td></tr><tr><td>Text-Video Alignment (TVA)</td><td>Measures whether generated content follows textual instructions, including entities, actions, and scenes.</td><td>VQAScore semantic matching combined with GroundingDINO-based entity grounding.</td></tr><tr><td>Identity Consistency (ID)</td><td>Measures whether recurring characters preserve visual identity across shot transitions.</td><td>Character localization and tracking followed by cross-shot identity embedding similarity.</td></tr><tr><td>Scene &amp; Style Consistency (SS)</td><td>Measures background, lighting, color, and visual style stability across shots.</td><td>Background similarity, style embedding comparison, and multimodal visual consistency</td></tr><tr><td>Reference Fidelity (RF)</td><td>Measures preservation of identity from reference images or audio</td><td>judgments. DINOv2 and ArcFace similarities for visual identity preservation.</td></tr><tr><td>Causal Physical Continuity</td><td></td><td>conditions.</td><td></td></tr><tr><td rowspan="2">Causal Physical Continuity</td><td>Spatial Layout Continuity (SLC)</td><td>Measures whether character and object positions remain consistent</td><td>Character/object localization and tracking evaluate normalized spatial</td></tr><tr><td>Object State Persistence (OSP)</td><td>across cuts. Evaluates whether object states evolve causally instead of being reset</td><td>displacement between adjacent shots. A multimodal evaluator extracts object states and verifies physically plausible</td></tr><tr><td></td><td>Geometric Scale Consistency (GSC) remain stable across viewpoints.</td><td>across shots. Measures whether character-object proportions and scene geometry</td><td>state transitions. Depth estimation and geometric reasoning evaluate relative scale</td></tr></table>

Table 5. Detailed evaluation metrics of PersonaShot (Part I). The table summarizes the foundational visual metrics and causal physica continuity metrics.

Stage 2: Face Visibility and Quality Filtering. For finegrained affective evaluation, we sample candidate frames uniformly across each shot and detect faces using RetinaFace [7]. We apply a strict confidence threshold of $\ge 0 . 8 5$ and require a minimum face bounding box area of ≥ 5% of the total frame. A segment is retained for affective evaluation only when valid face crops are present in at least 60% of its constituent shots and across no fewer than two distinct shots. To filter out severe motion blur or generative artifacts, we apply CLIP-IQA [25] and discard face crops scoring in the bottom 15th percentile of the data distribution.

Stage 3: Cross-Shot Relation Selection & Benchmark Statistics. We employ Qwen3.5-397B-A17B [23] to inspect adjacent shot pairs and verify the presence of required relational cues:

• Physical Continuity Evidence: Adjacent shots must retain observable commonalities, including identical character identities, interacted props, or consistent background landmarks, to support evaluations of Spatial Layout Continuity (SLC), Object State Persistence (OSP), and Geometric Scale Consistency (GSC).

• Cinematic Grammar Evidence: Shot transitions are categorized (hard cuts, dissolves, fades, and wipes) via TransNet V2, and shot-reverse-shot structures are tagged to support 180-degree Rule Compliance (180R) and Eyeline Match Correctness (EM).

After completing the three filtering stages, the final benchmark comprises exactly 1,000 high-quality human-centric multi-shot video segments, totaling over 5,000 annotated shots across diverse narrative themes.

Three-Layer Evaluation Workflow. During automated benchmarking, inference is structured across three temporal layers to minimize computational overhead:

1. Shot-Layer: For 5 representative shots per video (sampled at the 0%, 25%, 50%, 75%, and 100% timeline intervals), we perform a unified 7-in-1 visual scoring call and an Emotion-Narrative Alignment call on the middle keyframe (10 calls per video).

2. Cross-Shot Layer: We perform 2 pairwise imagecomparison calls on consecutive keyframe pairs (evaluating spatial, eyeline, and 180-degree rules) and 2 textbased calls on the extracted emotion label sequence (evaluating temporal flow and emotional arc).

3. Sequence-Layer: Transition Rhythm Alignment (TRA) is computed programmatically by combining duration variation statistics with the VLM-extracted action and emotion intensity scores.

## B.2. Evaluator Distillation, Fine-Tuning, and Inference Setup

To deploy lightweight, human-aligned specialist evaluators locally, we adapt Qwen3.5-4B and Qwen3-8B models via Low-Rank Adaptation (LoRA) [10].

Training Data Scale & Emotion Fine-Tuning Setup. For our fine-grained affective evaluator and Emotion-Prompt Alignment (EPA) scoring model, we fine-tune the Qwen3 backbone on the Multimodal Emotion Recognition and Reasoning (MERR) dataset. The training corpus consists of exactly 4,041 training samples and 446 validation samples covering 9 distinct emotional categories (happy, sad, neutral, angry, worried, surprise,fear, doubt, and contempt).

During training, the model learns to map multimodal facial Action Unit (AU) descriptions and narrative contexts to precise emotion labels. Table 4 details the exact LoRA hyperparameters and training configurations used for both the 4B and 8B model variants.

Teacher-Guided Distillation & Input Isolation Mechanism. For the physical continuity and cinematic grammar evaluators, we generate structured supervision targets y using a Qwen3.5-397B-A17B teacher model conditioned on full segment annotations A. To prevent the student evaluators from developing a shortcut dependency on ground-truth text annotations during inference, we enforce an input isolation mechanism:

$$
\begin{array} { r } { S _ { \mathrm { p h y } } = f _ { \theta } ( \mathcal { V } _ { \mathrm { g e n } } , \mathcal { P } _ { \mathrm { t e x t } } ) , \quad S _ { \mathrm { a f f } } = f _ { \theta } ( \mathcal { V } _ { \mathrm { g e n } } , \mathcal { P } _ { \mathrm { t e x t } } , \mathcal { F } _ { \mathrm { A U } } ) , } \end{array}\tag{4}
$$

where A is strictly withheld during student evaluation. Here, F<sub>AU</sub> represents frame-level Action Unit features extracted using OpenFace [2] formatted under the Facial Action Coding System (FACS), ensuring that the evaluator scores facial dynamics purely from generated visual evidence.

## C. Detailed Metrics

To provide a comprehensive operational description of PersonaShot, we summarize the definition and evaluation strategy for all 16 fine-grained metrics in Tables 5 and 6. The evaluation suite is structured across four complementary dimensions: foundational visual quality and consistency, causal physical continuity, affective dynamics, and cinematic grammar.

Evidence-Driven Dynamic Activation. To avoid inappropriate penalization, each metric is dynamically activated only when sufficient visual, facial, or relational evidence is verified by our automated preprocessing pipeline (§B.1). For example, eyeline matching (EM) is evaluated strictly on identified shot-reverse-shot structures, whereas Action Unit-aware affective metrics require clear facial observations (FATC, ENA). When a specific relational cue is absent in a video segment, the corresponding metric is masked out, and the dimension score is computed as the normalized average over all valid evaluation instances.

Score Calibration and Aggregation. Since the 16 submetrics originate from heterogeneous evaluation engines— ranging from DOVER quality scores and similarity cosines to 1-to-5 Likert ratings from specialist evaluators—all raw outputs are first calibrated to a standardized $[ 0 , 1 ]$ interval via metric-specific linear scaling, where higher values consistently indicate better narrative continuity. Following the standard evaluation protocol established in the main paper, the aggregate PersonaShot score $( S _ { \mathrm { P S } } )$ is computed as the unweighted arithmetic mean of the four dimension scores $( \alpha _ { \mathrm { v i s } } = \alpha _ { \mathrm { p h y } } = \alpha _ { \mathrm { a f f } } = \alpha _ { \mathrm { c i n } } = 0 . 2 5 )$ , ensuring a balanced and unbiased assessment across visual appearance, physical causality, affective evolution, and cinematic structure.

<table><tr><td>Dimension</td><td>Metric</td><td>Definition</td><td>Evaluation Strategy</td></tr><tr><td colspan="4">Affective Dynamics</td></tr><tr><td rowspan="5">Affective Dynamics</td><td>Expression Naturalness (EN)</td><td>Measures facial realism and generation artifacts in expressions.</td><td>Multimodal expression assessment combined with facial Action Unit (AU) analysis.</td></tr><tr><td>Facial-Action Temporal Coherence (FATC)</td><td>Measures temporal stability of facial actions and detects jitter or unnatural static expressions.</td><td>Frame-level AU trajectories are analyzed through temporal variation statistics.</td></tr><tr><td>Emotion-Narrative Alignment (ENA)</td><td>Measures whether emotional category and intensity match the narrative context.</td><td>Narrative affect extracted from annotations is compared with AU-aware facial affect estimation.</td></tr><tr><td>Emotional Arc Coherence (EAC)</td><td>Measures whether emotional states evolve coherently throughout the</td><td>Affective trajectories are constructed across shots and compared with expected narrative evolution.</td></tr><tr><td colspan="4">Cinematic Grammar</td></tr><tr><td rowspan="5">Cinematic Grammar</td><td>Directorial Narrative Sequencing (DNS)</td><td>Measures whether a global prompt is translated into a coherent shot-level narrative structure.</td><td>Global prompts and generated sequences are evaluated through cinematic reasoning models.</td></tr><tr><td>Shot Transition Appropriateness (ST)</td><td>Measures whether transition types match narrative context.</td><td>Shot boundaries are detected and transition choices are evaluated with visual and narrative evidence.</td></tr><tr><td>Transition Rhythm Alignment (TRA)</td><td>Measures whether shot duration and cutting density correspond to narrative pacing.</td><td>Shot duration statistics and cutting patterns are compared with narrative tension.</td></tr><tr><td>180-degree Rule Compliance (180R)</td><td>Measures whether screen direction and action axes remain consistent across cuts.</td><td>Character orientation and cross-shot axis consistency are analyzed.</td></tr><tr><td>Eyeline Match Correctness (EM)</td><td>Measures whether gaze directions are spatially coherent in shot-reverse-shot structures.</td><td>Facial gaze estimation evaluates complementary eye-line relations across shots.</td></tr></table>

Table 6. Detailed evaluation metrics of PersonaShot (Part II). The table summarizes affective dynamics and cinematic grammar metrics.

## D. Extended Experiments and Analytical Insights

To complement the fine-grained per-metric results reported in Table 2 of the main paper, this section provides extended experimental evaluations, dimension-level aggregations, and diagnostic ablations on PersonaShot. Specifically, we present: (1) the complete 12-model leaderboard aggregated across the four specialist dimensions and grouped by prompt conditioning paradigm, (2) fine-grained Emotion-Prompt Alignment (EPA) evaluation using our domain-adapted evaluator, (3) quantitative measurement of within-shot versus cross-shot consistency gaps, and (4) an ablation study on storyboard prompt granularity.

## D.1. Full 12-Model Dimension-Level Leaderboard

While Table 2 in the main paper presents detailed submetric scores across evaluated systems, Table 7 summarizes their macro-level performance across the four specialist dimensions: Visual Quality and Consistency (D1), Causal Physical Continuity (D2), Affective Dynamics (D3), and Cinematic Grammar (D4). Following Table 2 in the main paper, models are explicitly categorized by their conditioning paradigm: Shot-Level Prompt Conditioning (storyboard-guided generation) and Global-Prompt Conditioning (single prompt or automated storyboard planning). Strictly following the main paper protocol, the composite score is computed as the unweighted arithmetic mean of the four dimensions $( \alpha _ { \mathrm { v i s } } = \alpha _ { \mathrm { p h y } } = \alpha _ { \mathrm { a f f } } = \alpha _ { \mathrm { c i n } } = 0 . 2 5 )$ .

Analytical Insights and Metric Discriminative Power. Analyzing the performance differences across conditioning paradigms and dimensions reveals key characteristics of current generators:

• Structured Prompting vs. Single-Prompt Generation: Directly supporting Finding 2 of the main paper, models operating under Shot-Level Prompt Conditioning generally outperform their Global-Prompt counterparts in overall narrative consistency. For instance, Seedance (Structured) achieves a composite score of 0.730 compared to 0.665 for Seedance (Single Prompt), showing significant gains in physical continuity (0.824 vs. 0.808) and emotional progression (0.630 vs. 0.565).

• Cinematic Grammar (D4) exhibits the highest discriminative power: Among all four dimensions, D4 demonstrates the largest relative variation across models (CV = 12.7%, ranging from 0.437 to 0.744). Notably, under Global-Prompt Conditioning, LTX-2 (Single Prompt) dominates D4 film grammar (0.744) despite lower emotional persistence, indicating strong intrinsic editing priors within open-source base diffusion models.

• Orthogonality of Narrative Continuity Dimensions: Pearson correlation analysis indicates that D4 is virtually orthogonal to affective dynamics (D3, r = −0.07), confirming that visually and emotionally convincing shots do not automatically form well-sequenced cinematic narratives.

## D.2. Fine-Grained Emotion-Prompt Alignment (EPA)

To complement general visual-affective scoring, we evaluate Emotion-Prompt Alignment (EPA) using our specialist Qwen3-8B evaluator fine-tuned on the fine-grained MERR dataset (90.6% validation accuracy across 9 emotional categories). Since EPA evaluation requires explicit per-shot intended emotion descriptions extracted from storyboard captions, Table 8 reports results across systems operating under Shot-Level Prompt Conditioning.

Analytical Insights. As shown in Table 8, EPA composite scores cluster tightly across all storyboard-guided systems (ranging from 0.788 to 0.802, a narrow spread of only 0.014). Approximately 81% to 82% of generated shots successfully express the intended coarse emotion category (Hard Acc), with soft valence-arousal accuracy exceeding 85%. However, the trajectory correlation (Traj Corr) remains moderate (∼0.653) across all models. This confirms that while modern video generators reliably synthesize static emotional expressions within individual shots, orchestrating smooth, narrative-consistent emotional transitions across multi-shot cuts remains a critical bottleneck.

## D.3. Quantitative Gap: Per-Shot vs. Cross-Shot Consistency

A primary motivation of PersonaShot is that conventional benchmarks overestimate multi-shot video generation quality by evaluating isolated clips. Table 9 quantifies the performance discrepancy between within-shot (per-shot) metrics and cross-shot relational metrics across the visual and affective dimensions.

Analytical Insights. The data provides direct quantitative confirmation of our main paper findings: across all evaluated systems, cross-shot consistency scores are 25.6% to 50.0% lower than per-shot quality scores. While models reliably generate visually appealing individual frames (Per-Shot Visual Avg of 0.780), they frequently reset background styles and character identities across cuts (Cross-Shot Visual Avg of 0.580). This temporal gap is most severe in D3 Emotional Expression, where performance drops by exactly 50.0% (0.700 → 0.350). This indicates that microexpression persistence and emotional arc coherence are the most critical bottlenecks in human-centric multi-shot video generation.

## D.4. Ablation on Storyboard Prompt Granularity

To assess how prompt richness impacts narrative continuity, we conduct an ablation study on the LTX-2 architecture by comparing full storyboard prompts (∼1,200 tokens under Shot-Level Prompt Conditioning) against brief compressed prompts (∼600 tokens retaining only core actions).

Analytical Insights. As reported in Table 10, prompt compression degrades overall composite performance by 34.2%, revealing distinct sensitivities across evaluation dimensions:

• Physical Consistency is Highly Prompt-Dependent: D2 suffers the most extreme relative drop (-48.0%, falling from 0.786 to 0.409). Without explicit storyboard state constraints, objects frequently float, lose physical support, or undergo unexplained shape transformations across cuts.

• Cinematic Grammar is Robust to Prompt Compression: In contrast, D4 film grammar exhibits the highest resilience, dropping by only 14.8% (0.744 → 0.634). This indicates that camera framing, 180-degree rule compliance, and shot transition logic are largely governed by the model’s internal pre-training priors rather than verbose textual instructions.

Table 7. Complete 12-Model Dimension Leaderboard on PersonaShot. We report the aggregate Composite score alongside four macro-dimensions across 100 standardized test sequences per model. Models are categorized into Shot-Level Prompt Conditioning and Global-Prompt Conditioning. Equal dimension weighting (α = 0.25) is applied. Within each protocol, the highest score in each column is highlighted in bold.
<table><tr><td>Rank</td><td>Model / System</td><td>Composite (↑)</td><td>D1: Visual (↑)</td><td>D2: Physical (↑)</td><td>D3: Emotion (↑)</td><td>D4: Grammar (↑)</td></tr><tr><td colspan="7">Shot-Level Prompt Conditioning</td></tr><tr><td>1</td><td>Seedance (Structured)</td><td>0.730</td><td>0.751</td><td>0.824</td><td>0.630</td><td>0.716</td></tr><tr><td>2</td><td>HoloCine (Storyboard)</td><td>0.707</td><td>0.764</td><td>0.823</td><td>0.594</td><td>0.646</td></tr><tr><td>3</td><td>STAGE + Wan2.2</td><td>0.650</td><td>0.673</td><td>0.774</td><td>0.546</td><td>0.609</td></tr><tr><td>4</td><td>LongLive</td><td>0.623</td><td>0.642</td><td>0.755</td><td>0.453</td><td>0.642</td></tr><tr><td>5</td><td>MultiShotMaster</td><td>0.617</td><td>0.639</td><td>0.755</td><td>0.546</td><td>0.530</td></tr><tr><td>6</td><td>STAGE + LTX-2.3</td><td>0.603</td><td>0.615</td><td>0.715</td><td>0.541</td><td>0.542</td></tr><tr><td>7</td><td>ShotStream</td><td>0.567</td><td>0.600</td><td>0.633</td><td>0.417</td><td>0.617</td></tr><tr><td>8</td><td>CineTrans</td><td>0.554</td><td>0.602</td><td>0.637</td><td>0.429</td><td>0.550</td></tr><tr><td>9</td><td>EchoShot</td><td>0.542</td><td>0.544</td><td>0.669</td><td>0.518</td><td>0.437</td></tr><tr><td colspan="7">Global-Prompt Conditioning</td></tr><tr><td>1</td><td>LTX-2 (Single Prompt)</td><td>0.684</td><td>0.722</td><td>0.786</td><td>0.482</td><td>0.744</td></tr><tr><td>2</td><td>Seedance (Single Prompt)</td><td>0.665</td><td>0.674</td><td>0.808</td><td>0.565</td><td>0.612</td></tr><tr><td>3</td><td>VGoT-FP</td><td>0.595</td><td>0.619</td><td>0.672</td><td>0.491</td><td>0.599</td></tr></table>

Table 8. Emotion-Prompt Alignment (EPA) Results. We report the composite EPA score alongside exact label match (Hard Acc), valence-arousal similarity (Soft Acc), and emotional arc trajectory correlation (Traj Corr) for shot-level prompt conditioning systems. Best performance is highlighted in bold.
<table><tr><td>Model / System</td><td>EPA (↑)</td><td>Hard Acc (↑)</td><td>Soft Acc (↑)</td><td>Traj Corr (↑)</td></tr><tr><td>Seedance (Structured)</td><td>0.802</td><td>0.824</td><td>0.869</td><td>0.648</td></tr><tr><td>STAGE + Wan2.2</td><td>0.797</td><td>0.816</td><td>0.867</td><td>0.657</td></tr><tr><td>CineTrans</td><td>0.797</td><td>0.820</td><td>0.870</td><td>0.646</td></tr><tr><td>MultiShotMaster</td><td>0.796</td><td>0.815</td><td>0.864</td><td>0.651</td></tr><tr><td>ShotStream</td><td>0.796</td><td>0.814</td><td>0.863</td><td>0.651</td></tr><tr><td>HoloCine</td><td>0.793</td><td>0.815</td><td>0.864</td><td>0.637</td></tr><tr><td>STAGE + LTX-2.3</td><td>0.793</td><td>0.812</td><td>0.862</td><td>0.653</td></tr><tr><td>LongLive</td><td>0.788</td><td>0.804</td><td>0.856</td><td>0.653</td></tr></table>

Table 9. Per-Shot vs. Cross-Shot Performance Degradation. Averaged scores across all evaluated models show substantial drops when transitioning from local shot fidelity to cross-shot relational continuity.
<table><tr><td>Dimension</td><td>Per-Shot Avg (↑)</td><td>Cross-Shot Avg (↑)</td><td>Absolute Drop (∆)</td></tr><tr><td>D1: Visual Quality</td><td>0.780</td><td>0.580</td><td>-0.200 (-25.6%)</td></tr><tr><td>D3: Emotional Expression</td><td>0.700</td><td>0.350</td><td>-0.350 (-50.0%)</td></tr></table>

## D.5. Qualitative Visualizations and Cross-Model Comparison

To provide an intuitive visual validation of our quantitative benchmarks, Figure 6 and Figure 5 present comprehensive side-by-side qualitative comparisons across nine representative multi-shot video generation systems on multi-shot narrative storyboards.

Qualitative Observations and Failure Diagnostics. The visual comparisons corroborate the core quantitative findings established in our benchmark:

• Superiority of Storyboard-Anchored Consistency: Systems equipped with structured storyboard guidance, most notably Seedance and HoloCine, demonstrate remarkable cross-shot narrative continuity. In both dramatic dialogue scenes (Figure 5) and fine-grained action sequences (Figure 6), these models successfully preserve character identity, clothing details, environmental lighting, and emotional progression across cuts.

• Severe Identity and Age Drift: Without explicit crossshot relational reasoning, open-source and streaming baselines frequently suffer from sharp identity resets. For instance, in Figure 6, LongLive abruptly shifts the protagonist from a middle-aged man to a young boy in Shots 3 and 4, while CineTrans and EchoShot exhibit substantial facial and style inconsistencies across shot boundaries.

• Semantic Hallucination and B-Roll Insertion: Streaming frameworks such as ShotStream struggle to maintain character-centric focus across extended sequences. When prompted with continuous human character actions, Shot-Stream frequently hallucinates disconnected B-roll cutaways, including an irrelevant decorative vase in Figure 5 (Shots 2 and 3) or a bedroom digital clock in Figure 6 (Shot 1), which breaks character presence entirely.

• Style and Chromaticity Drift: While models like LTX-2.3 capture basic shot sequencing and cinematic framing, they frequently experience chromaticity instability, spontaneously drifting into grayscale or monochrome palettes across cuts (as observed in both Figure 6 and Figure 5).

[Shot 1] An extreme close-up slowly zooms out from the face of a young Asian man with thick dark wavy hair, revealing his downward gaze and pained expression of distress, while in the background, a middle-aged Asian woman with short black hair and wire-rimmed glasses watches him with intense, serious attentiveness.

[Shot 2] A static close-up focuses on the face of a middle-aged Asian woman with short, neatly styled black hair and wire-rimmed glasses, wearing a formal gray suit jacket, as she stares with a concerned yet authoritative expression, her eyes locked on the subject,

[Shot 3] A static medium close-up captures a young Asian man with thick dark wavy hair, dressed in a light blue and white striped hospital gown, sitting silently with a deeply troubled expression, his head bowed and eyes averted, overwhelmed by emotion in the cool light.

![](images/d69fd94c2649c6bf8592471501bd0ff2fe60f42b8e1c827d337abd0991f2c581.jpg)  
Figure 5. Qualitative Multi-Model Comparison on a 3-Shot Emotional Dialogue Scene. Comparison of nine video generation systems given a hierarchical prompt describing an intense emotional interaction between a young man and a middle-aged woman. While Seedance and HoloCine maintain character identities and emotional distress across cuts, baselines exhibit severe failure modes: ShotStream replaces the character with an irrelevant vase, LTX-2.3 suffers from monochrome style drift, and EchoShot experiences sharp identity shifts across shot boundaries.

[Shot 1] A steady medium close-up shot at eye level captures a middle-aged man with dark hair and thick eyebrows, wearing a brown tweed jacket, white shirt, and striped tie, as he drives a yellow car while holding an envelope, with suburban scenery passing by outside. [Shot 2] An extreme close-up static shot from a high angle focuses on the man's hands as he carefully peels a stamp from a sheet, emphasizing the detailed texture of the paper and the precision of the action.

[Shot 3] A handheld medium close-up shot at eye level shows the man leaning forward with a mischievous grin, using his tongue to lick the back of the stamp in a comical manner. [Shot 4] A static close-up shot at eye level captures the man's face as he proudly smiles, displaying satisfaction after successfully applying the stamp to the envelope.

![](images/2a9ee88b9a722a3601330fc419e824f56ebc27e5191c897d7366249a8bedd5cb.jpg)  
Figure 6. Qualitative Multi-Model Comparison on a 4-Shot Comical Action Sequence. Comparison across nine systems generating a multi-shot narrative of a man driving a car, peeling a stamp, licking it, and smiling. Seedance and HoloCine accurately execute the fine-grained object interaction and facial expressions while preserving character identity. In contrast, LongLive hallucinates a child identit in Shot 3, ShotStream generates a bedroom alarm clock instead of the car interior, and STAGE struggles with stamp-envelope objec persistence.

Table 10. Impact of Prompt Length on LTX-2 Performance. Halving storyboard prompt length causes severe degradation in physical and visual coherence while leaving cinematic grammar relatively resilient.
<table><tr><td>Prompt Condition</td><td>Composite (↑)</td><td>D1: Visual (↑)</td><td>D2: Physical (↑)</td><td>D3: Emotion (↑)</td><td>D4: Grammar (↑)</td></tr><tr><td>LTX-2 Full Prompt (～1200 tokens)</td><td>0.705</td><td>0.722</td><td>0.786</td><td>0.482</td><td>0.744</td></tr><tr><td>LTX-2 Brief Prompt (~600 tokens)</td><td>0.464</td><td>0.434</td><td>0.409</td><td>0.269</td><td>0.634</td></tr><tr><td>Absolute Degradation (∆)</td><td>-0.241</td><td>-0.288</td><td>-0.378</td><td>-0.212</td><td>-0.110</td></tr><tr><td>Relative Performance Drop</td><td>-34.2%</td><td>-39.9%</td><td>-48.0%</td><td>-44.1%</td><td>-14.8%</td></tr></table>

![](images/548c77bbb8fb32a8a7ed5be37b499b4a7cf6eb7b850e65c3b828f110afae0745.jpg)  
Figure 7. Coefficient of Variation (CV %) across PersonaShot Sub-Metrics. Higher CV indicates stronger discriminative power across evaluated systems. Emotion Narrative Alignment (25.3%) exhibits the highest sensitivity across all models.

## D.6. Metric Sensitivity and Internal Orthogonality Analysis

To verify the discriminative power and structural nonredundancy of our evaluation framework, we analyze the Coefficient of Variation (CV) across all sub-metrics (Figure 7) alongside internal pairwise correlations within each specialist dimension (Figure 8).

Metric Discriminative Power (Coefficient of Variation). As shown in Figure 7, the CV values reveal significant differences in metric sensitivity across evaluated systems:

• High Variance in Narrative Alignment: Emotion Narrative Alignment (ENA) exhibits the highest CV across the benchmark (25.3%). While most models achieve comparable baseline performance in static facial realism (Expression Naturalness, CV = 11.9%) and local stability (Micro-Expression Temporal, $\mathrm { C V } = 1 2 . 0 \% )$ , their capabilities diverge sharply when tasked with matching emotional category and intensity to complex narrative contexts.

• Sensitivity of Geometric Scaling: Within Physical Consistency (D2), Geometric Scale exhibits the highest variation (18.8%), reflecting severe scale drift and characterto-object proportion distortion across viewpoint changes in weaker models.

• Uniform Difficulty of Cinematic Grammar: Submetrics within Film Grammar (D4) maintain balanced CV values between 15.4% and 16.6%, indicating that temporal cutting rhythms and spatial camera rules represent uniformly challenging bottlenecks for current video generators.

Internal Orthogonality and Synergy (Pearson’s r). The correlation heatmaps in Figure 8 demonstrate that our sub-metrics evaluate complementary rather than redundant capabilities:

• Physical Consistency (D2): Spatial Layout is nearly orthogonal to Object State $( r \ = \ 0 . 1 8 )$ and Geometric Scale $( r = 0 . 1 9 )$ . This confirms that preserving correct character and object positions across cuts does not guarantee morphological stability. Conversely, Object State and Geometric Scale exhibit a strong positive correlation $( r ~ = ~ 0 . 7 1 )$ , as shape deformation and state resets frequently occur alongside relative scale distortions.

![](images/d2f7efcbb58f754d48a36b27a19e301b96a04c612c85db72f6ec38b3f09a6bf2.jpg)  
Figure 8. Internal Correlation Heatmaps (Pearson’s r) for Specialist Dimensions. From left to right: Physical Consistency (D2), Emotional Expression (D3), and Film Grammar (D4). Low correlations across multiple sub-metrics confirm the orthogonality and non redundancy of our benchmark criteria.

• Emotional Expression (D3): Emotion Narrative Alignment shows low correlation with low-level facial execution metrics $( r \leq 0 . 2 7 )$ , proving that semantic emotionto-story adherence is independent of visual expression rendering. Meanwhile, Micro-Expression Temporal and Cross-Shot Emotional Arc correlate moderately $\begin{array} { r l } { ( r } & { { } = } \end{array}$ 0.59), reflecting shared requirements for short- and longterm temporal continuity.

• Film Grammar (D4): Transition Rhythm is completely decoupled from spatial cinematic rules $( r \in$ $[ - 0 . 0 2 , 0 . 0 5 ] )$ . This demonstrates that temporal cutting pacing (rhythm) and spatial camera geometry (axis and eyeline matching) assess fundamentally distinct directorial skills. In contrast, Shot Transition, 180-Degree Rule, and Eyeline Match share moderate positive correlations $( r \in [ 0 . 5 0 , 0 . 6 0 ] )$ ), as they jointly govern cross-shot spatial coherence.

## E. Additional Human Study Details

To verify that our specialist evaluators accurately reflect human judgment, we conduct an expert evaluation study on a held-out set of 25 multi-shot video sequences. This section outlines the evaluation protocol and presents both dimension-level and fine-grained criterion-level alignment between our automatic scores and human preference.

## E.1. Evaluation Protocol

Ten domain experts participated in the study. Under a sparse assignment protocol, each video sequence was independently rated by four annotators across 12 specialist criteria using a five-point Likert scale. Model identities and automatic scores were strictly concealed.

To account for criteria that require specific visual or relational context, annotators could mark a metric as Not $\mathsf { A p - }$ plicable (N/A) if the corresponding evidence was absent.

Such instances were dynamically excluded from correlation calculations rather than assigned default values. Across all evaluated sequences, the pooled inter-rater reliability reached Krippendorff’s $\alpha = 0 . 7 6$ , indicating high consensus among experts.

## E.2. Alignment with Human Judgments

We evaluate alignment using Spearman rank correlation (ρ) and pairwise preference accuracy between automatic scores and expert Mean Opinion Scores (MOS). Table 11 reports the aggregated dimension-level results, while Table 12 details the performance across all 12 fine-grained criteria.

Table 11. Dimension-level alignment with human MOS. All correlations are statistically significant at $p < 0 . 0 1$
<table><tr><td>Dimension</td><td>N</td><td>ρ↑</td><td>Pair Acc.↑</td></tr><tr><td>Causal Physical Continuity</td><td>25</td><td>0.75</td><td>75.2%</td></tr><tr><td>Affective Dynamics</td><td>25</td><td>0.69</td><td>70.3%</td></tr><tr><td>Cinematic Grammar</td><td>25</td><td>0.73</td><td>72.8%</td></tr><tr><td>Overall PersonaShot</td><td>25</td><td>0.78</td><td>76.8%</td></tr></table>

Analytical Insights. The results across both tables highlight several key characteristics of our evaluation framework:

• Strong Holistic Agreement: As shown in Table 11, the overall PersonaShot score achieves a Spearman correlation of 0.78 and a pairwise accuracy of 76.8%. This confirms that combining our specialized dimensions effectively reflects holistic human preference in multi-shot narrative continuity.

• High Precision on Explicit Rules: Table 12 demonstrates that evaluators grounded in explicit structural logic achieve remarkably strong alignment with human experts. In particular, Object State (OSP, $\rho ~ = ~ 0 . 7 4 )$ , Directorial Narrative Sequencing (DNS, $\rho ~ = ~ 0 . 7 4 )$ , and Shot

Table 12. Fine-grained criterion-level human alignment. Comparison of automatic evaluator scores against expert MOS across all 12 specialist criteria. Correlations are computed over valid evidence-gated sequences $( p < 0 . 0 1 )$
<table><tr><td>Dim.</td><td>Criterion</td><td>ρ↑</td><td>Pair Acc.↑</td></tr><tr><td rowspan="3">Phy.</td><td>Spatial Layout (SLC)</td><td>0.68</td><td>71.4%</td></tr><tr><td>Object State (OSP)</td><td>0.74</td><td>76.0%</td></tr><tr><td>Geometric Scale (GSC)</td><td>0.66</td><td>69.8%</td></tr><tr><td rowspan="4">Aff.</td><td>Expr. Naturalness (EN)</td><td>0.70</td><td>71.8%</td></tr><tr><td>Action Coherence (FATC)</td><td>0.68</td><td>64.8%</td></tr><tr><td>Narrative Align. (ENA)</td><td>0.71</td><td>72.5%</td></tr><tr><td>Arc Coherence (EAC)</td><td>0.67</td><td>68.2%</td></tr><tr><td rowspan="5">Cin.</td><td>Narrative Seq. (DNS)</td><td>0.74</td><td>74.0%</td></tr><tr><td>Shot Transition (ST)</td><td>0.72</td><td>73.1%</td></tr><tr><td>Rhythm Align. (TRA)</td><td>0.69</td><td>70.5%</td></tr><tr><td>180-Degree Rule (180R)</td><td>0.71</td><td>71.9%</td></tr><tr><td>Eyeline Match (EM)</td><td>0.65</td><td>67.2%</td></tr></table>

Transition (ST, ρ = 0.72) show high pairwise accuracy above 73%, proving that our instruction-tuned reasoners successfully capture temporal state evolution and editing syntax.

• Validity of Evidence-Gating: Crucially, correlations are computed strictly over applicable, evidence-gated sequences (e.g., evaluating 180-Degree Rule and Eyeline Match only on the subset of sequences containing shotreverse-shot interactions). Masking out inapplicable instances prevents random scoring noise and ensures reliable alignment.