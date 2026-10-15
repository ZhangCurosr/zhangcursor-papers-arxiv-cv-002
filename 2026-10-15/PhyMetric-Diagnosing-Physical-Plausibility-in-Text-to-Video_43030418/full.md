# PhyMetric: Diagnosing Physical Plausibility in Text-to-Video Generation via Scene-Level QA

Siwei Meng Xuerou Cai Rui Hu Ping Liu<sup>B</sup> Department of Computer Science & Engineering University of Nevada, Reno Reno, NV 89512

## Abstract

Text-to-video (T2V) generation models can produce visually realistic videos that nevertheless violate the physical consequences implied by the prompt and scene. Existing benchmarks largely rely on holistic plausibility scores or broad quality dimensions, leaving it unclear which physical behavior is being tested and why a model fails. We introduce PHYMETRIC, a scene-level diagnostic benchmark that recasts physical plausibility evaluation as verifiable, scene-grounded question answering. PHYMETRIC contains 1,115 scenes spanning 50 physical categories across six domains, paired with 8,920 answer-balanced binary questions over 12 evaluation dimensions. Each scene is associated with eight targeted yes/no questions and a structured physical prior specifying the governing law, expected temporal sequence, observable signatures, and common failure modes. At evaluation time, a vision-language model (VLM) judge answers each question from question-aware sampled frames conditioned on the prior, and correctness is hierarchically aggregated into scene, dimension, physical category, and model scores. Experiments on four open-source T2V models and three VLM judges show that scene-specific binary QA is a critical component of the protocol: when each targeted question is replaced by a single generic plausibility query, evaluator accuracy drops to the chance level. Structured physical priors offer a complementary signal that improves human alignment of the resulting scores, with effects that vary across VLM judges of different capacity. These findings establish PHYMETRIC as a benchmark that turns physical plausibility evaluation from opaque holistic scoring into interpretable, scene-level inspection. Our project page is available at: https://mengsiwei.github.io/PhyMetric/.

## 1 Introduction

Text-to-video generation models [1, 2, 3, 4, 5, 6, 7, 8, 9] have achieved strong visual fidelity and are increasingly applied to dynamic-scene simulation, not only to clip-level visual generation [10, 11, 12]. In this regime, a generated video should not only look realistic frame by frame, but also evolve according to the physical and causal regularities implied by the prompt and scene. For example, a falling object should accelerate downward, a sliced object should separate into pieces, and a buoyant object should remain supported by fluid rather than drift arbitrarily. Conventional visual quality metrics [13, 14, 15], however, operate largely at the frame level and cannot capture such temporal causality. As a result, they may assign high scores to videos that look plausible at every frame but violate basic physical expectations [16, 17].

Recent benchmarks have started to address this mismatch by extending evaluation from surface-level realism toward world-aware and physics-oriented analysis [18, 19, 20, 21, 11, 22, 23, 24]. Despite this progress, many of them still rely on coarse plausibility assessments that ask whether a video looks physically reasonable as a whole, rather than verifying whether the key physical event implied by the prompt is actually realized. These assessments aggregate diverse physical consequences into a single score, which can hide a missing or incorrect physical event despite a high overall score. Conversely, a low score reveals little diagnostic insight into which particular physical law was violated. The result is a persistent gap between surface realism and verifiable physical plausibility.

![](images/eaeff894847a8786906511149c7c0a80265d34ef88b3817ccf3d75623dff785f.jpg)  
Figure 1: Why holistic plausibility is insufficient for physical evaluation. Generic plausibility questions tend to yield high-confidence but unreliable judgments by overlooking scene-specific physical consequences. PHYMETRIC instead decomposes evaluation into structured diagnostic QA with question-aware frame sampling, enabling more robust and interpretable physical assessment. In the right subfigure, as confidence thresholds increase, generic-question accuracy degrades sharply and only achieves around 50% accuracy, whereas structured scene-specific questions remain substantially more stable.

We attribute this gap to two core challenges: identifiability and expectation grounding. Identifiability requires the benchmark to make explicit which physical consequence is being tested. A single holistic prompt is too weak a probe for physical reasoning: it obscures which consequence is being tested and remains highly susceptible to linguistic drift or evaluator bias. As shown in Figure 1, this lack of granularity can cause VLM-judge performance close to chance levels, even under highconfidence predictions. Beyond identifiability, expectation grounding requires the evaluator to know what physical behavior the scene implies before judging. For instance, elastic and inelastic collisions both involve contact but yield fundamentally different correct outcomes. Without a scene-specific prior, judges fall back on generic heuristics and often confuse semantic mismatches with genuine physical failures [21, 11, 25].

Motivated by these challenges, we introduce PHYMETRIC, a scene-level diagnostic benchmark that replaces holistic plausibility scoring with scene-specific balanced binary QA. For each physical scene, PHYMETRIC issues multiple yes/no questions, each targeting a verifiable physical consequence that can be checked from a small set of question-aware frames. This design directly addresses identifiability: it makes the tested consequence explicit and keeps the chance baseline interpretable by construction. To address expectation grounding, every scene is further paired with a structured physical prior that summarizes the governing law, the expected temporal sequence, the observable signatures, and common failures, anchoring the VLM evaluator in the correct physical context before video inspection. PHYMETRIC thus supports a focused evaluative claim, namely whether a generated video exhibits the specific physical consequences implied by the scene, while remaining agnostic to broader questions of latent physical understanding or generalization beyond observed dynamics.

In total, PHYMETRIC comprises 1,115 scenes from 50 physical categories across six domains: mechanics, fluids, materials, thermal processes, electromagnetism, and chemistry (Figure 2). These categories are chosen as a coverage map over the dynamics that recur in generated video, rather than as a topical list, so that PHYMETRIC probes a diverse range of physical behaviors instead of overfitting to familiar motions such as falling or bouncing. Each scene is annotated with eight expert-verified binary questions, yielding 8,920 QA pairs spanning 12 evaluation dimensions. At evaluation time, PHYMETRIC runs a three-step pipeline that first performs question-aware frame sampling around the queried event, then conditions the VLM evaluator on the structured physical prior, and finally aggregates binary responses hierarchically into scene-, dimension-, category-, and model-level diagnostic scores. Across four open-source text-to-video models and three VLM judges, we find that scene-specific QA is what enables non-trivial evaluator accuracy: replacing it with a generic plausibility query collapses all three judges to chance, while structured priors provide a complementary, evaluator-dependent signal that improves human alignment of the resulting scores.

In summary, our main contributions are as follows:

• We introduce PHYMETRIC, a scene-level diagnostic benchmark for physical plausibility in text-to-video generation, comprising 1,115 scenes over 50 categories and six physical domains, paired with 8,920 answer-balanced binary questions that span 12 evaluation dimensions.

• We recast physical plausibility evaluation from holistic scoring to per-scene balanced binary QA grounded in a structured physical prior, and show empirically that scene-specific questions are necessary for non-trivial evaluator accuracy while the prior provides a complementary, evaluator-dependent signal that improves human alignment.

• We validate PHYMETRIC across four open-source T2V models and three VLM judges, demonstrating consistent model rankings across VLM judges, distinctive dimension-level capability profiles, and systematic weaknesses in current systems that are not visible to holistic scoring.

## 2 Related Works

General Video Generation Benchmarks. Video generation evaluation has moved from single-score metrics, such as FVD [14] and CLIP-based similarity [15], toward multi-dimensional benchmark suites. VBench [18] addresses this limitation by decomposing video generation quality into a hierarchy of fine-grained dimensions covering quality and semantics, while EvalCrafter [19], FETV [26], and T2V-CompBench [27] provide complementary multi-dimensional and compositional protocols. More recent benchmarks reach further than surface-level visual quality and probe whether generated videos satisfy richer notions of real-world validity, often by reformulating evaluation as a human- or model-aligned reasoning problem. Video-Bench [28] formulates video generation evaluation as a human-aligned MLLM-based scoring problem with chain-of-query reasoning, and ETVA [29] further pushes this direction by generating dense, prompt-derived questions to probe text-video alignment at fine granularity. Building on these efforts, VBench2.0 [20] formalises the goal as intrinsic faithfulness covering human fidelity, controllability, creativity, physics, and commonsense, and VideoVerse [21] introduces hidden semantics to test whether T2V models can infer unstated event consequences rather than merely follow explicit prompt content. Across these benchmarks, physics typically appears as one component of a larger quality rubric, leaving open how scene-specific physical consequences should be verified for individual generations.

Physics-Oriented Video Generation Benchmarks. A growing line of work focuses directly on physical commonsense and physical laws. VideoPhy [30] and VideoPhy-2 [31] evaluate semantic adherence and physical commonsense across solids, fluids, and multi-material interactions. Phy-GenBench [11] constructs prompts around explicit physical laws and introduces PhyGenEval for hierarchical physical evaluation. Other works move beyond direct plausibility scoring by anchoring evaluation in alternative forms of physical evidence. Morpheus [32] grounds evaluation in real physical experiments and physical invariants. LikePhys [33] avoids direct VLM judging and instead measures intuitive physics understanding through likelihood preference between physically valid and invalid samples, while Physion-Eval [25] provides expert human reasoning traces for diagnosing temporally localized physical glitches. Evaluator-focused works also show the difficulty of physical judgment. PhysBench [34] shows that broad VLM judges miss physically critical details when asked a single broad question and recover only when paired with explicit physical priors. CoPhyBench [22] improves diagnosticity by decomposing physical reasoning into prediction, calculation, and counterfactual inference, and TRAVL [23] shows that current VLM evaluators struggle with implausibility detection and proposes trajectory-aware training to improve their behavior.

PHYMETRIC addresses these recurring limitations through target design decisions. While prior works often rely on holistic scoring that obscures specific failure modes, PHYMETRIC replaces broad assessments with balanced binary questions to ensure identifiability, allowing every score to be traced back to concrete physical checks. To solve the problem of expectation grounding, where evaluators lack the context to distinguish physical errors from semantic ones, we pair each scene with a structured physical prior that anchors the judge in the governing law before inspection. Furthermore, as shown in Table 1, PHYMETRIC implements balanced labels to provide an interpretable 50% chance baseline, and utilizes question-aware frame sampling to link judgments to time-sensitive physical evidence. Compared with QA-based benchmarks like ETVA [29], PHYMETRIC anchors its queries to objective physical consequences rather than text-alignment coverage, bridging the gap between surface-level realism and verifiable physical plausibility.

Table 1: Comparison with existing video generation benchmarks. We group prior benchmarks into two main aspects. ✓denotes explicit support, ✗denotes no support, and ∼denotes partial support. Balanced Labels indicates whether positive/negative answers are balanced for an interpretable chance baseline. Compared with broad video-generation and physics-oriented benchmarks, PHYMETRIC combines large-scale scene coverage, scene-specific yes/no QA, structured physical priors, and balanced labels for physical plausibility evaluation.
<table><tr><td>Benchmark</td><td>Main target</td><td>Scenes / Prompts</td><td>Questions</td><td>QA Format</td><td>Physics / Prior</td><td>Balanced Labels</td><td>Year</td></tr><tr><td colspan="8">General and fine-grained video generation evaluation</td></tr><tr><td>VBench [18]</td><td>General video quality</td><td>1</td><td></td><td>x</td><td>2</td><td>x</td><td>2024</td></tr><tr><td>EvalCrafter [19]</td><td>General video quality</td><td>700</td><td></td><td>x</td><td>x</td><td>x</td><td>2024</td></tr><tr><td>FETV [26]</td><td>Fine-grained T2V eval.</td><td>619</td><td></td><td>x</td><td>x</td><td>x</td><td>2025</td></tr><tr><td>T2V-CompBench [27]</td><td>Compositional T2V</td><td>700</td><td></td><td>2</td><td>x</td><td>x</td><td>2025</td></tr><tr><td>Video-Bench [28]</td><td>Human-aligned gen. eval.</td><td>419</td><td></td><td>2</td><td>2</td><td>x</td><td>2025</td></tr><tr><td>VBench-2.0 [20]</td><td>Intrinsic faithfulness</td><td></td><td></td><td>2</td><td>2</td><td>x</td><td>2025</td></tr><tr><td>ETVA [29]</td><td>Text-video alignment</td><td>2,000</td><td>12K</td><td>√</td><td>2</td><td>x</td><td>2025</td></tr><tr><td colspan="8">Temporal, world-model, and physics-oriented evaluation</td></tr><tr><td>ChronoMagic-Bench [35]</td><td>Time-lapse transformation</td><td>1,649</td><td>一</td><td>x</td><td>2</td><td>x</td><td>2024</td></tr><tr><td>WorldScore [36]</td><td>World generation</td><td>3,000</td><td>1</td><td>x</td><td>2</td><td>x</td><td>2025</td></tr><tr><td>VideoVerse [21]</td><td>World-model capability</td><td>300</td><td>793</td><td>√</td><td>2</td><td>x</td><td>2025</td></tr><tr><td>PhyGenBench [11]</td><td>Physical laws</td><td>160</td><td>一</td><td>2</td><td>√</td><td>x</td><td>2025</td></tr><tr><td>VideoPhy [30]</td><td>Physical commonsense</td><td>688</td><td></td><td>x</td><td>2</td><td>x</td><td>2025</td></tr><tr><td>VideoPhy-2 [31]</td><td>Physical commonsense</td><td>688</td><td>一</td><td>x</td><td>2</td><td>x</td><td>2025</td></tr><tr><td>Morpheus [32]</td><td>Physical invariants</td><td>80</td><td></td><td>x</td><td>√</td><td>x</td><td>2025</td></tr><tr><td>LikePhys [33]</td><td>Likelihood preference</td><td>一</td><td></td><td>x</td><td>√</td><td>2</td><td>2025</td></tr><tr><td>Physion-Eval [25]</td><td>Human glitch diagnosis</td><td>一</td><td></td><td>x</td><td>2</td><td>x</td><td>2025</td></tr><tr><td>PHYMETRIC (ours)</td><td>Physical plausibility</td><td>1,115</td><td>8,920</td><td>√</td><td>√</td><td>√</td><td>2026</td></tr></table>

## 3 The PHYMETRIC Benchmark

PHYMETRIC addresses the two challenges in Section 1 with two design decisions that jointly guide dataset curation and evaluation. This section motivates each decision and describes its impact on both benchmark design and evaluation protocol (Sections 3.1 and 3.2), before consolidating the resulting benchmark statistics and release artifacts.

## 3.1 Identifiability through scene-specific binary QA

A central failure mode of holistic physical evaluation is lack of identifiability. When a judge is asked a broad question such as “Is this video physically plausible?”, a low score does not reveal which physical condition failed. The answer may be affected by visual quality, object appearance, prompt mismatch, or the judge’s prior knowledge, even when the target physical event is only one part of the clip. This makes the resulting score hard to interpret and hard to use for model diagnosis.

PHYMETRIC addresses this problem by formulating physical evaluation as scene-specific binary question answering. For each scene s , we define a set of observable questions:

$$
\mathcal { Q } _ { i } = \{ ( q _ { i , j } , y _ { i , j } , d _ { i , j } ) \} _ { j = 1 } ^ { M } ,\tag{1}
$$

where $q _ { i , j }$ is a binary yes/no question, $y _ { i , j } \in \{ \mathtt { y e s } , \mathtt { n o } \}$ is the ground truth answer, and $d _ { i , j }$ denotes the evaluation dimension associated with the question. In our benchmark, each scene contains $M = 8$ binary questions. These questions are not generic templates shared across all scenes. They are written for the concrete objects, environments, actions, and expected physical consequences of the scene. As illustrated in Figure 1, instead of asking whether a video of cutting a tomato is physically plausible, PHYMETRIC asks whether the knife contacts the tomato, whether the tomato separates after the cut, and whether the two pieces remain visually consistent. We provide the whole scene construction in Appendix A.2.2 and the examples of scene-specific questions in Appendix A.4.4.

This design turns a vague physical judgment into a set of identifiable checks. Each question corresponds to a specific hypothesis about the generated video. If a model receives a low scene score, the failed questions reveal whether the error comes from motion direction, object interaction, material response, or another physical dimension. Thus, the benchmark score is not only a ranking signal, but also a diagnostic signal for physical plausibility. The binary format also improves score calibration, with each response mapped to a binary truth value, the resulting question score is defined as:

$$
a _ { i , j } ( V ) = \mathbb { 1 } \left[ \hat { y } _ { i , j } ( V ) = y _ { i , j } \right] ,\tag{2}
$$

where ${ \hat { y } } _ { i , j } ( V )$ is the predicted answer for generated video V. The scene-level score is then the average question accuracy:

$$
S _ { i } ( V ) = \frac { 1 } { M } \sum _ { j = 1 } ^ { M } a _ { i , j } ( V ) .\tag{3}
$$

By averaging results across scenes, we establish a robust model-level metric. The balanced nature of our binary labels provides a natural 50% baseline, effectively reducing the influence of random guessing and answer bias. We emphasize that the transition to a binary QA format is not a reduction of physical complexity, but a prerequisite for rigorous verification. Many physical failures in generated videos are local and conditional. A model may satisfy one consequence of a scene while failing another. By using multiple targeted questions per scene, PHYMETRIC captures this mixed behavior while keeping evaluation automatic and comparable across models.

## 3.2 Expectation grounding through video-independent priors

A scene-specific question is necessary but not sufficient. A VLM judge that has never been told what the scene is supposed to do will answer the question using whichever heuristics the visual evidence happens to invoke, and an evaluator that receives the whole video without temporal guidance may attend to frames where the queried event has not yet occurred [37]. Expectation grounding therefore requires the evaluator to know two things before it judges: what physical behavior the scene implies, and when in the video to look for it. PHYMETRIC addresses both through a video-independent structured prior $K _ { s }$

What to expect. For each scene $s _ { i }$ , we construct a structured prior

$$
K _ { s _ { i } } = \{ \ell _ { i } , o _ { i } , m _ { i } , e _ { i } , r _ { i } , h _ { i } \} ,\tag{4}
$$

where $\ell _ { i }$ denotes the governing physical law, $o _ { i }$ the involved objects, $m _ { i }$ the material or state information, $e _ { i }$ the expected event sequence, $r _ { i }$ likely failure modes, and $h _ { i }$ dimension-level evaluation hints. For example, in a tomato-slicing scene, $\ell _ { i }$ describes cutting as blade-induced shear failure, $o _ { i }$ includes the knife and tomato, $m _ { i }$ specifies a soft deformable tomato, $e _ { i }$ expects contact followed by separation into two halves, $r _ { i }$ lists failures such as no cut or floating pieces, and $h _ { i }$ tells the judge to check shape, interaction, and appearance consistency. $K _ { s }$ is constructed from the scene caption and category metadata only, and is independent of the generated video V . At evaluation time, the four core fields $\{ \ell _ { i } , e _ { i } , r _ { i } , \dot { h } _ { i } \}$ are passed to the VLM judge as textual context, while $\{ o _ { i } , m _ { i } \}$ are auxiliary metadata used during scene and question construction (full schema in Appendix A.3). This independence is a deliberate choice: a prior that conditions on V would adapt to the same visual heuristics the evaluator is being asked to bypass.

When to look. Physical events are temporally local: a question about initial contact should focus on the moment two objects meet, while a question about final state should focus on the outcome after the interaction. Each question $q _ { i , j }$ is therefore associated with one of five temporal anchors recorded in $h _ { i }$ (initial, interaction, transition, final, or full sequence), which marks where the relevant evidence is expected to appear in the reference clip. For a generated video $V$ with $N _ { V }$ frames, anchor positions from a reference clip with $N _ { \mathrm { r e f } }$ frames are mapped to V by

$$
f ^ { \prime } = \mathrm { r o u n d } \left( \frac { f } { N _ { \mathrm { r e f } } - 1 } ( N _ { V } - 1 ) \right) ,\tag{5}
$$

and the target frame together with its neighbors forms the evidence set $\mathcal { F } _ { i , j } ( V )$ passed to the judge. This linear mapping assumes that the temporal location of an event in V is approximately proportional to its location in the reference clip. This assumption holds when the model preserves the prompted event ordering, but it may fail under conditions of temporal compression or reordering. We further discuss this limitation in Appendix A.6.

Joint role. The structured prior $K _ { s }$ and the question-aware frame set $\mathcal { F } _ { i , j } ( V )$ are complementary: $K _ { s }$ tells the evaluator what physical process to expect, and $\mathcal { F } _ { i , j } ( V )$ tells it where in the video to verify that expectation. Neither alone is sufficient: a prior without temporally localized evidence describes a behavior the evaluator cannot anchor visually, and localized frames without a prior leave the evaluator to guess which physical consequence is being tested.

![](images/0658509f4b3413e9d8598fcefdbb85aa7bfd3a2f349b16a7b68b900b174e1d76.jpg)  
Figure 2: PHYMETRIC benchmark construction. PHYMETRIC organizes physical video generation into a structured taxonomy with 50 categories across six domains. Each category is defined by physical principles, object pools, environment pools, and diagnostic dimensions. These metadata are expanded into diverse scenes and balanced scene-specific binary questions. The bottom examples show representative physical events.

Benchmark summary and release As shown in Figure 2, PHYMETRIC contains 1,115 physical scenes covering 50 fine-grained physical categories across six physical domains. Each scene is paired with eight binary questions, resulting in 8,920 scene-specific QA pairs. The questions span 12 evaluation dimensions, covering both low-level visual evidence and higher-level physical consequences. The benchmark also provides structured physical priors for all scenes, including the expected event sequence, relevant physical principle, dimension hints, and common failure modes. These components jointly support the evaluation pipeline illustrated in Figure 3, where question-aware frame sampling and structured priors transform each generated video into interpretable scene-level physical diagnosis. Compared with holistic video-level scoring, PHYMETRIC provides three key properties. First, it is identifiable: each score can be traced back to concrete question-level physical checks. Second, it is grounded: the expected behavior is defined before observing generated videos. Third, it is evidence-localized: question-aware frame sampling links each judgment to the relevant temporal region of the video.

## 4 Experiments

Our experiments address three questions: whether PHYMETRIC separates current T2V models at scale, which physical dimensions remain difficult, and whether scene-specific QA and structured physical priors are necessary for reliable evaluation. Accordingly, we organize the experiments into three parts. We first report the main model comparison on the full benchmark and a controlled 100-scene subset. We then analyze model performance across 12 evaluation dimensions to identify systematic physical weaknesses. Finally, we conduct ablation studies and human validation to test the role of scene-specific questions, structured physical priors, and our robustness on VLM judges.

## 4.1 Experimental Setup

We evaluate four open-source text-to-video models: CogVideoX-5B [4], Wan2.1-1.3B [3], Wan2.2- 14B [3], and Cosmos-7B [1]. Each model generates one video for each scene using the original scene caption as input, without model-specific prompt rewriting. The main evaluation is conducted on the full PHYMETRIC benchmark with 1,115 scenes from 50 physical categories. For human validation and ablation studies, we use a controlled 100-scene subset, sampled as two scenes per category, which preserves category-level coverage while reducing annotation and multi-evaluator cost.

![](images/f779844d68af6a4dc3649c0d6f85c334ab669ee12d92031a6ff9b404806ce81b.jpg)  
Figure 3: PHYMETRIC evaluation pipeline. Given a physical scene s, a text-to-video model generates a video $V _ { s }$ from the scene caption $c _ { s } .$ For each scene-specific binary question, PHYMETRIC samples question-aware frames and provides a structured physical prior $K _ { s }$ describing the expected law, outcome, and common failure modes. A VLM then answers each question conditioned on the sampled frames and $K _ { s }$ . Question correctness is averaged into the scene score $S ( s , { \hat { V } } _ { s } )$ , and model performance is obtained by averaging scene scores over all benchmark scenes. This pipeline turns holistic video evaluation into grounded, interpretable physical diagnosis.

We use Qwen3.5-9B [38] as the default VLM judge, and additionally evaluate with Qwen2.5-VL-7B-Instruct [39] and GPT-5.4 [40] to test evaluator robustness. Each judge answers the scene-specific yes/no questions from question-aware sampled frames, the scene caption, and the structured physical prior $\dot { K _ { s } }$ . We report Overall accuracy and Weighted accuracy, where Overall is the mean binary correctness over all evaluated questions and Weighted incorporates the VLM judge’s self-reported confidence. Implementation details are provided in Appendix A.2 and A.3.

For the human study, 50 annotators answer the same yes/no questions on the controlled 100-scene subset. We use human responses to measure whether automatic scores align with human-perceived physical plausibility, rather than to replace benchmark labels. We report scene-level Spearman correlation ρ<sub>human</sub> between VLM scores and human scores. More details about the annotation interface, quality control, and agreement analysis are provided in Appendix A.5.2.

## 4.2 PHYMETRIC Main Comparison

Full benchmark results. Table 2 reports results on the full benchmark using two open-source VLM judges, Qwen3.5-9B [38] and Qwen2.5-7B [39]. PHYMETRIC separates current T2V models under both judges. Wan2.2-14B obtains the best average score, followed closely by Cosmos-7B. Wan2.1-1.3B ranks third, while CogVideoX-5B obtains the lowest score. The ranking is not explained by parameter count alone. CogVideoX-5B is larger than Wan2.1-1.3B, but scores lower under both judges. This suggests that PHYMETRIC captures model differences in physical behaviors, rather than only model scale. The gap between the best and weakest models also shows that current open-source T2V models are not saturated on PHYMETRIC.

Controlled subset results. Table 3 reports results on the 100-scene subset with three VLM judges, Qwen3.5-9B [38], Qwen2.5-7B [39], and GPT-5.4 [40]. The ranking is broadly consistent with the full benchmark. Wan2.2-14B and Cosmos-7B form the strongest group, while CogVideoX-5B remains the weakest model. The absolute scores differ across judges. Qwen2.5-7B is more conservative than Qwen3.5-9B and GPT-5.4, but the model ordering remains stable enough to support cross-evaluator comparison. The controlled subset also shows why a single score is not sufficient. Cosmos-7B slightly exceeds Wan2.2-14B in the three-judge average, while Wan2.2-14B is stronger under Qwen3.5-9B, that suggests close overall scores may hide different physical capability profiles.

## 4.3 Dimension-Level Analysis

Figure 4 reveals that current T2V models exhibit highly uneven physical capabilities across the 12 dimensions, with average accuracy ranging from 90.39% on number down to 66.39% on appearance. The strongest performance appears in number, suggesting that current T2V models and VLM judges can reliably verify object count when visual evidence is explicit. Direction and motion also remain relatively robust, both exceeding 79%, which indicates that coarse kinematic patterns and large-scale movement trends are generally captured more consistently than deeper physical attributes. In contrast, the lowest accuracy on appearance reveals that physical failures frequently emerge through material, state, or surface inconsistency rather than purely incorrect motion trajectories. Similarly, weaker performance on size and spatial dimensions indicates persistent difficulty in preserving object scale and relative placement over time.

Table 2: Main results on the full PHYMET-RIC benchmark (1,115 scenes) using opensource VLM judges. Average denotes the mean performance across VLM judges.
<table><tr><td rowspan="2">Model</td><td colspan="2">Qwen3.5-9B</td><td colspan="2">Qwen2.5-VL-7B</td><td colspan="2">Average</td></tr><tr><td>Overall</td><td>Weighted</td><td></td><td>Overall Weighted</td><td>Overall</td><td>Weighted</td></tr><tr><td>Wan2.2-14B</td><td>76.74</td><td>78.07</td><td>68.82</td><td>70.82</td><td>72.78</td><td>74.45</td></tr><tr><td>Cosmos-7B</td><td>76.28</td><td>77.60</td><td>68.89</td><td>70.92</td><td>72.59</td><td>74.26</td></tr><tr><td>Wan2.1-1.3B</td><td>73.86</td><td>75.50</td><td>67.26</td><td>69.42</td><td>70.56</td><td>72.46</td></tr><tr><td>CogVideoX-5B</td><td>71.36</td><td>73.18</td><td>65.12</td><td>67.62</td><td>68.24</td><td>70.40</td></tr></table>

Table 3: Evaluation on the 100-scene controlled subset across both closed and open-source VLM evaluators. Average reflects the mean accuracy across all three judges.
<table><tr><td rowspan="2">Model</td><td colspan="2">Qwen3.5-9B</td><td colspan="2">Qwen2.5-VL-7B</td><td colspan="2">GPT-5.4</td><td colspan="2">Average</td></tr><tr><td>Overall Weighted</td><td></td><td>Overall Weighted</td><td></td><td></td><td>Overall Weighted Overall Weighted</td><td></td><td></td></tr><tr><td>Wan2.2-14B</td><td>77.88</td><td>79.23</td><td>66.75</td><td>69.54</td><td>73.12</td><td>74.26</td><td>72.58</td><td>74.34</td></tr><tr><td>Cosmos-7B</td><td>76.50</td><td>77.54</td><td>67.37</td><td>70.37</td><td>74.00</td><td>75.54</td><td>72.62</td><td>74.48</td></tr><tr><td>Wan2.1-1.3B</td><td>74.62</td><td>76.85</td><td>65.87</td><td>68.94</td><td>72.12</td><td>73.88</td><td>70.87</td><td>73.22</td></tr><tr><td>CogVideoX-5B</td><td>70.63</td><td>72.45</td><td>64.38</td><td>67.13</td><td>69.75</td><td>71.67</td><td>68.25</td><td>70.42</td></tr></table>

![](images/a67837cf6e9f67ec2b8a5036bcf512263add3c46667b382b7fe05423cea98b69.jpg)  
Figure 4: Per-dimension physical performance of current T2V models. Accuracy across 12 evaluation dimensions for four T2V models, with diamonds denoting mean accuracy. Radar plots showing each model’s dimension profile and the dimensions where it achieves the best performance.

Inter-model spread further highlights which dimensions most effectively separate current generators. Appearance and shape show the largest spreads (10.00 and 10.53 points), suggesting that material realism and deformation-sensitive physical behavior remain major differentiators across models. object identity exhibits only a 1.69-point spread, implying that current generators are comparatively similar on basic object-presence checks while diverging more substantially on physically meaningful appearance and structural consistency.

We further visualize the dimension profiles of the four evaluated T2V models. Wan2.2-14B forms the broadest and most stable profile, with strong coverage on kinematic and causal dimensions. Cosmos-7B attains the largest number of dimension-wise best scores, indicating that it is highly competitive on specific physical checks. Wan2.1-1.3B shows a more selective profile on material and existence dimensions, while CogVideoX-5B shows a visibly contracted profile, especially in dimensions that require stable physical appearance or relational evidence. This visualization suggests that model differences are not only scalar, but also structural, in which two models with close overall scores may succeed on different types of physical evidence.

## 4.4 Ablation Study

We conduct ablations on the 100-scene subset to examine the two main design choices of PHYMETRIC: scene-specific QA and the structured physical prior. For each scene s, the structured physical prior $K _ { s }$ is generated before video evaluation from the scene description and category metadata. Importantly, $K _ { s }$ is generated without access to the generated videos or ground-truth question answers, which provides scene-level expectation grounding VLM judge.

Component ablation. Table 4 evaluates two components of PHYMETRIC, the scene-specific question design and the structured physical priors $K _ { s }$ . The results show that scene-specific QA is the key component for reliable physical diagnosis. When scene-specific questions are replaced by the generic query “Is this video physically plausible?”, accuracy drops to near chance for all evaluators. Without $K _ { s }$ , Qwen3.5-9B drops from 72.75% to 48.13%, Qwen2.5-VL from 63.47% to 47.31%, and GPT-5.4 from 73.94% to 49.25%. Adding $K _ { s }$ to the generic query still fails to recover the missing signal, indicating that physical context alone is insufficient when the evaluator is not asked to verify a concrete scene-specific consequence.

Table 4: Component ablation of PHYMETRIC. We toggle the structured physical prior $K _ { s }$ and the question style. “Scene-specific” denotes targeted binary questions, while “Generic” replaces each with a single global query.  
Table 5: Effect of structured physical prior source and evaluator. Each row is a pair of prior source and question evaluator. zero-shot means without the structured physical prior $K _ { s }$
<table><tr><td> $K _ { s }$ </td><td>Question</td><td>Evaluator</td><td>Overall</td><td>Weighted</td><td>ρhuman</td></tr><tr><td>√</td><td>Scene-specific</td><td>Qwen3.5-9B</td><td>74.91</td><td>76.52</td><td>+0.201</td></tr><tr><td>x</td><td>Scene-specific</td><td>Qwen3.5-9B</td><td>72.75</td><td>74.56</td><td>+0.165</td></tr><tr><td>x</td><td>Generic</td><td>Qwen3.5-9B</td><td>48.13</td><td>49.71</td><td>+0.192</td></tr><tr><td>√</td><td>Generic</td><td>Qwen3.5-9B</td><td>48.88</td><td>50.17</td><td>+0.196</td></tr><tr><td>√</td><td>Scene-specific</td><td>Qwen2.5-VL</td><td>66.09</td><td>69.00</td><td>+0.104</td></tr><tr><td>x</td><td>Scene-specific</td><td>Qwen2.5-VL</td><td>63.47</td><td>65.89</td><td>+0.215</td></tr><tr><td>x</td><td>Generic</td><td>Qwen2.5-VL</td><td>47.31</td><td>50.36</td><td>+0.139</td></tr><tr><td>√</td><td>Generic</td><td>Qwen2.5-VL</td><td>46.91</td><td>50.64</td><td>+0.030</td></tr><tr><td>√</td><td>Scene-specific</td><td>GPT-5.4</td><td>72.25</td><td>73.84</td><td>+0.250*</td></tr><tr><td>x</td><td>Scene-specific</td><td>GPT-5.4</td><td>73.94</td><td>75.28</td><td>+0.216</td></tr><tr><td>x</td><td>Generic</td><td>GPT-5.4</td><td>49.25</td><td>49.30</td><td>+0.121</td></tr><tr><td>√</td><td>Generic</td><td>GPT-5.4</td><td>47.12</td><td>48.45</td><td>+0.202</td></tr></table>

<table><tr><td> $K _ { s }$  source</td><td>Evaluator</td><td>Overall</td><td>Weighted</td><td>ρhuman</td></tr><tr><td>zero-shot</td><td>Qwen3.5-9B</td><td>72.75</td><td>74.56</td><td>+0.165</td></tr><tr><td>Qwen3.5-9B</td><td>Qwen3.5-9B</td><td>74.91</td><td>76.52</td><td>+0.201</td></tr><tr><td>Qwen2.5-7B</td><td>Qwen3.5-9B</td><td>73.66</td><td>75.42</td><td>+0.209</td></tr><tr><td>DeepSeek-R1</td><td>Qwen3.5-9B</td><td>73.81</td><td>75.67</td><td>+0.251*</td></tr><tr><td>Mistral-7B</td><td>Qwen3.5-9B</td><td>74.09</td><td>76.13</td><td>+0.248*</td></tr><tr><td>zero-shot</td><td>Qwen2.5-VL</td><td>63.47</td><td>65.89</td><td>+0.015</td></tr><tr><td>Qwen3.5-9B</td><td>Qwen2.5-VL</td><td>66.09</td><td>69.00</td><td>+0.104</td></tr><tr><td>Qwen2.5-7B</td><td>Qwen2.5-VL</td><td>63.25</td><td>66.15</td><td>+0.156</td></tr><tr><td>DeepSeek-R1</td><td>Qwen2.5-VL</td><td>61.97</td><td>64.53</td><td>+0.194</td></tr><tr><td>Mistral-7B</td><td>Qwen2.5-VL</td><td>62.91</td><td>65.50</td><td>+0.158</td></tr><tr><td>zero-shot</td><td>GPT-5.4</td><td>73.94</td><td>75.28</td><td>+0.216</td></tr><tr><td>Qwen3.5-9B</td><td>GPT-5.4</td><td>72.25</td><td>73.84</td><td>+0.250*</td></tr><tr><td>Qwen2.5-7B</td><td>GPT-5.4</td><td>72.72</td><td>73.87</td><td>+0.283*</td></tr><tr><td>DeepSeek-R1</td><td>GPT-5.4</td><td>72.47</td><td>73.76</td><td>+0.270*</td></tr><tr><td>Mistral-7B</td><td>GPT-5.4</td><td>72.56</td><td>73.64</td><td>+0.293*</td></tr></table>

∗ marks $\begin{array} { r } { p < 0 . 0 5 . } \end{array}$

When scene-specific questions are kept, $K _ { s }$ provides a complementary but evaluator-dependent benefit. For Qwen3.5-9B, adding $K _ { s }$ improves both accuracy and human correlation, increasing accuracy from 72.75% to 74.91% and $\rho _ { \mathrm { h u m a n } }$ from 0.165 to 0.201. For Qwen2.5-VL, $K _ { s }$ improves accuracy from 63.47% to 66.09%, though human correlation decreases. For GPT-5.4, $K _ { s }$ slightly lowers label accuracy but improves human correlation from 0.216 to 0.250. These results suggest that benchmark-label accuracy and human alignment measure related but distinct evaluator behaviors. Overall, scene-specific QA is necessary for diagnostic accuracy, while $K _ { s }$ can further improve alignment when the evaluator can use the prior effectively.

Effect of structured physical prior source and evaluator. To test whether the structured physical prior depends on a particular LLM source, we examine $K _ { s }$ with different auxiliary LLMs and evaluate them with different VLM judges in Table 5. Across evaluators, the best human correlation often comes from a prior source different from the evaluator itself. For Qwen3.5-9B, all structured priors improve over zero-shot, and DeepSeek-R1 and Mistral-7B achieve the highest human correlations, with $\rho _ { \mathrm { h u m a n } } = 0 . 2 5 1$ and 0.248. For Qwen2.5-VL, DeepSeek-R1 also gives the strongest human correlation. For GPT-5.4, zero-shot gives the highest raw accuracy, but every structured prior improves human correlation, with Mistral-7B reaching the best $\rho _ { \mathrm { h u m a n } } = 0 . 2 9 3$ . This cross-source pattern suggests that $K _ { s }$ is not simply a self-consistency boost from the same model family. Instead, structured physical priors provide an external expectation signal that can make VLM judgments more aligned with human-perceived physical plausibility.

Findings From the main results on full benchmark and subset, PHYMETRIC separates current T2V models and reveals that better visual quality does not always imply stronger physical behavior. Second, the dimension scores show that current models are stronger at coarse motion and object count, but still struggle with higher-order physical dimensions, including material state, appearance stability, scale consistency, and spatial relations. Third, the ablation results confirm that scene-specific QA is necessary for meaningful physical evaluation, while structured physical priors provide a useful but evaluator-dependent grounding signal.

## 5 Conclusion

We present PHYMETRIC, a structured benchmark for evaluating physical plausibility in T2V generation through scene-grounded diagnostic question answering. Unlike prior evaluation paradigms that rely primarily on holistic plausibility or coarse realism judgments, PHYMETRIC decomposes physical understanding into verifiable scene-specific consequences across diverse physical domains, dimensions, and difficulty levels. By combining a large-scale taxonomy of 50 physical categories, expert-filtered benchmark construction, structured physical priors, question-aware frame sampling, and hierarchical score aggregation, PHYMETRIC provides both broad coverage and fine-grained diagnostic interpretability. Our experiments show that physical evaluation is fundamentally more reliable when grounded in targeted physical consequences rather than generic plausibility. Across multiple VLM judges, PHYMETRIC consistently reveals stable model rankings and systematic weaknesses in current T2V systems. These findings suggest that our PHYMETRIC can serve not only as a benchmark for model comparison, but also as a diagnostic framework for guiding future progress in physically grounded video generation.

## Acknowledgments and Disclosure of Funding

This work was supported in part by the National Science Foundation under Award No. 2511989 for the MRI project “Acquisition of a High-Performance GPU Cluster for Large-Scale AI”. Any opinions, findings, and conclusions or recommendations expressed in this material are those of the authors and do not necessarily reflect the views of the funding agencies. The authors declare no competing interests.

## References

[1] Niket Agarwal, Arslan Ali, Maciej Bala, Yogesh Balaji, Erik Barker, Tiffany Cai, Prithvijit Chattopadhyay, Yongxin Chen, Yin Cui, Yifan Ding, et al. Cosmos world foundation model platform for physical ai. arXiv preprint arXiv:2501.03575, 2025.

[2] OpenAI. Video generation models as world simulators. https://openai.com/research/ video-generation-models-as-world-simulators, 2024. Accessed 2024.

[3] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.

[4] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. ICLR, 2025.

[5] GoogleDeepMind Veo Team. Veo: a text-to-video generation system. 2025.

[6] Team Seedance, De Chen, Liyang Chen, Xin Chen, Ying Chen, Zhuo Chen, Zhuowei Chen, Feng Cheng, Tianheng Cheng, Yufeng Cheng, et al. Seedance 2.0: Advancing video generation for world complexity. arXiv preprint arXiv:2604.14148, 2026.

[7] Qiaowei Miao, Kehan Li, Jinsheng Quan, Zhiyuan Min, Shaojie Ma, Yichao Xu, Yi Yang, Ping Liu, and Yawei Luo. Advances in 4d generation: A survey. arXiv preprint arXiv:2503.14501, 2025.

[8] Siwei Meng, Yawei Luo, Shu Zhang, and Ping Liu. When physical preferences meet semantic constraints: Physical and semantic direct preference optimization for text-to-video generation. ACMMM, 2026.

[9] Jiacheng Wang, Zhedong Zheng, Wei Xu, and Ping Liu. Rigi: Rectifying image-to-3d generation inconsistency via uncertainty-aware learning. TIP.

[10] Jake Bruce, Michael D Dennis, Ashley Edwards, Jack Parker-Holder, Yuge Shi, Edward Hughes, Matthew Lai, Aditi Mavalankar, Richie Steigerwald, Chris Apps, et al. Genie: Generative interactive environments. In ICML, 2024.

[11] Fanqing Meng, Jiaqi Liao, Xinyu Tan, Wenqi Shao, Quanfeng Lu, Kaipeng Zhang, Yu Cheng, Dianqi Li, Yu Qiao, and Ping Luo. Towards world simulator: Crafting physical commonsensebased benchmark for video generation. ICML, 2025.

[12] Vahid Balazadeh, Mohammadmehdi Ataei, Hyunmin Cheong, Amir Hosein Khasahmadi, and Rahul G Krishnan. Physics context builders: A modular framework for physical reasoning in vision-language models. In ICCV, 2025.

[13] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image quality assessment: from error visibility to structural similarity. TIP, 2004.

[14] Thomas Unterthiner, Sjoerd Van Steenkiste, Karol Kurach, Raphaël Marinier, Marcin Michalski, and Sylvain Gelly. Fvd: A new metric for video generation. ICLRW, 2019.

[15] Yi Wang, Yinan He, Yizhuo Li, Kunchang Li, Jiashuo Yu, Xin Ma, Xinhao Li, Guo Chen, Xinyuan Chen, Yaohui Wang, et al. Internvid: A large-scale video-text dataset for multimodal understanding and generation. ICLR, 2024.

[16] Siwei Meng, Yawei Luo, and Ping Liu. Grounding creativity in physics: A brief survey of physical priors in aigc. IJCAI, 2025.

[17] Bingyi Kang, Yang Yue, Rui Lu, Zhijie Lin, Yang Zhao, Kaixin Wang, Gao Huang, and Jiashi Feng. How far is video generation from world model: A physical law perspective. ICML, 2025.

[18] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, et al. Vbench: Comprehensive benchmark suite for video generative models. In CVPR, 2024.

[19] Yaofang Liu, Xiaodong Cun, Xuebo Liu, Xintao Wang, Yong Zhang, Haoxin Chen, Yang Liu, Tieyong Zeng, Raymond Chan, and Ying Shan. Evalcrafter: Benchmarking and evaluating large video generation models. In CVPR, 2024.

[20] Dian Zheng, Ziqi Huang, Hongbo Liu, Kai Zou, Yinan He, Fan Zhang, Lulu Gu, Yuanhan Zhang, Jingwen He, Wei-Shi Zheng, et al. Vbench-2.0: Advancing video generation benchmark suite for intrinsic faithfulness. PAMI, 2025.

[21] Zeqing Wang, Xinyu Wei, Bairui Li, Zhen Guo, Jinrui Zhang, Hongyang Wei, Keze Wang, and Lei Zhang. Videoverse: How far is your t2v generator from a world model? arXiv preprint arXiv:2510.08398, 2025.

[22] Fanyue Wei, Kai Xu, Yizhuo Zhang, Pengzhan Sun, Junbin Xiao, and Angela Yao. Cophybench: Benchmarking physical reasoning from conditional video observation. 2025.

[23] Saman Motamed, Minghao Chen, Luc Van Gool, and Iro Laina. Travl: A recipe for making video-language models better judges of physics implausibility. arXiv preprint arXiv:2510.07550, 2025.

[24] Siwei Meng, Yawei Luo, and Ping Liu. Phymagic: Physical motion-aware generative inference with confidence-guided vlm. In ECCV, 2026.

[25] Qin Zhang, Peiyu Jing, Hong-Xing Yu, Fangqiang Ding, Fan Nie, Weimin Wang, Yilun Du, James Zou, Jiajun Wu, and Bing Shuai. Physion-eval: Evaluating physical realism in generated video via human reasoning. arXiv preprint arXiv:2603.19607, 2026.

[26] Yuanxin Liu, Lei Li, Shuhuai Ren, Rundong Gao, Shicheng Li, Sishuo Chen, Xu Sun, and Lu Hou. Fetv: A benchmark for fine-grained evaluation of open-domain text-to-video generation. NeurIPS, 36:62352–62387, 2023.

[27] Kaiyue Sun, Kaiyi Huang, Xian Liu, Yue Wu, Zihan Xu, Zhenguo Li, and Xihui Liu. T2vcompbench: A comprehensive benchmark for compositional text-to-video generation. In CVPR, 2025.

[28] Hui Han, Siyuan Li, Jiaqi Chen, Yiwen Yuan, Yuling Wu, Yufan Deng, Chak Tou Leong, Hanwen Du, Junchen Fu, Youhua Li, et al. Video-bench: Human-aligned video generation benchmark. In CVPR, 2025.

[29] Kaisi Guan, Zhengfeng Lai, Yuchong Sun, Peng Zhang, Wei Liu, Kieran Liu, Meng Cao, and Ruihua Song. Etva: Evaluation of text-to-video alignment via fine-grained question generation and answering. In ICCV, 2025.

[30] Hritik Bansal, Zongyu Lin, Tianyi Xie, Zeshun Zong, Michal Yarom, Yonatan Bitton, Chenfanfu Jiang, Yizhou Sun, Kai-Wei Chang, and Aditya Grover. Videophy: Evaluating physical commonsense for video generation. ICLR, 2025.

[31] Hritik Bansal, Clark Peng, Yonatan Bitton, Roman Goldenberg, Aditya Grover, and Kai-Wei Chang. Videophy-2: A challenging action-centric physical commonsense evaluation in video generation. arXiv preprint arXiv:2503.06800, 2025.

[32] Chenyu Zhang, Daniil Cherniavskii, Antonios Tragoudaras, Antonios Vozikis, Thijmen Nijdam, Derck WE Prinzhorn, Mark Bodracska, Nicu Sebe, Andrii Zadaianchuk, and Efstratios Gavves. Morpheus: Benchmarking physical reasoning of video generative models with real physical experiments. arXiv preprint arXiv:2504.02918, 2025.

[33] Jianhao Yuan, Fabio Pizzati, Francesco Pinto, Lars Kunze, Ivan Laptev, Paul Newman, Philip Torr, and Daniele De Martini. Likephys: Evaluating intuitive physics understanding in video diffusion models via likelihood preference. arXiv preprint arXiv:2510.11512, 2025.

[34] Wei Chow, Jiageng Mao, Boyi Li, Daniel Seita, Vitor Guizilini, and Yue Wang. Physbench: Benchmarking and enhancing vision-language models for physical world understanding. ICLR, 2025.

[35] Shenghai Yuan, Jinfa Huang, Yongqi Xu, Yaoyang Liu, Shaofeng Zhang, Yujun Shi, Ruijie Zhu, Xinhua Cheng, Jiebo Luo, and Li Yuan. Chronomagic-bench: A benchmark for metamorphic evaluation of text-to-time-lapse video generation. Advances in Neural Information Processing Systems, 37:21236–21270, 2024.

[36] Haoyi Duan, Hong-Xing Yu, Sirui Chen, Li Fei-Fei, and Jiajun Wu. Worldscore: A unified evaluation benchmark for world generation. In ICCV, 2025.

[37] Yubin Chen, Xuyang Guo, Zhenmei Shi, Zhao Song, and Jiahao Zhang. T2vworldbench: A benchmark for evaluating world knowledge in text-to-video generation. In CVPR, 2026.

[38] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

[39] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.

[40] Aaditya Singh, Adam Fry, Adam Perelman, Adam Tart, Adi Ganesh, Ahmed El-Kishky, Aidan McLaughlin, Aiden Low, AJ Ostrow, Akhila Ananthram, et al. Openai gpt-5 system card. arXiv preprint arXiv:2601.03267, 2025.

## A Technical appendices and supplementary material

In this appendix, we provide additional technical details, prompts, and experimental results to support the main manuscript. The content is organized as follows:

## A.1 Physics Taxonomy and Dimensions

• 50 physics categories.

• 12 evaluation dimensions.

## A.2 Scene and Question Construction

• System and user prompts for data generation.

• Detailed per-category construction.

## A.3 Structured Physical Priors Construction

• Evaluation of various LLM backbones.

• The system prompts used to generate physics-grounded priors.

## A.4 PHYMETRIC Evaluation

• Prompting strategies for VLM judges.

• Technical specifications of the VLM judges and baseline video generators.

• Cases of PHYMETRIC evaluation.

## A.5 Experimental Details

• Ablation study setups and question query formulation.

• Methodology and participant demographics for the human study.

## A.6 Limitations and Future Work

• Limitations.

• Future work.

## A.1 Physics Taxonomy and Dimensions

Physics domains. PHYMETRIC groups its 50 categories into six physics domains so that aggregate analysis can isolate where T2V models succeed or fail at the level of broad physical theory rather than individual phenomena. Table 6 summarizes the per-domain category and scene counts; the full per-category card listing is in §A.2.2.

Table 6: Six physics domains in PHYMETRIC with category and scene counts.
<table><tr><td>Domain</td><td>Categories</td><td>Scenes</td><td>Representative phenomena</td></tr><tr><td>Mechanics</td><td>12</td><td>280</td><td>gravity, collisions, friction, projectile, pendulum</td></tr><tr><td>Fluid Dynamics</td><td>10</td><td>230</td><td>buoyancy, viscosity, splash, vortex, granular flow</td></tr><tr><td>Material</td><td>8</td><td>180</td><td>fracture, slicing, cloth, smoke, explosion</td></tr><tr><td>Thermal</td><td>8</td><td>165</td><td>melting, boiling, freezing, convection, condensation</td></tr><tr><td>Chemistry</td><td>7</td><td>145</td><td>diffusion, adsorption, dissolution, precipitation</td></tr><tr><td>Electromagnetism</td><td>5</td><td>115</td><td>reflection, refraction, shadow, electrostatics, magnetism</td></tr><tr><td>Total</td><td>50</td><td>1,115</td><td></td></tr></table>

Question dimensions. Every binary question is tagged with one or two of the 12 question dimensions listed in Table 7. Dimensions are not categories: a single category (e.g., Splash) generates questions across several dimensions (motion, shape, spatial, causal), and a single dimension (e.g., motion) is exercised in dozens of categories.

Table 7: The 12 question dimensions used to tag every binary question. Each question carries 1–2 dimension tags; each scene covers ${ \ge } 3$ distinct dimensions across its 8 questions.
<table><tr><td>Dimension</td><td>What it asks about</td></tr><tr><td>existence</td><td>whether a specific object, region, or event is present in the video</td></tr><tr><td>size</td><td>absolute or relative size of an object or region</td></tr><tr><td>appearance/texture</td><td>color, surface pattern, transparency, sheen, smoothness</td></tr><tr><td>shape</td><td>geometric form and how it changes during the event</td></tr><tr><td>motion</td><td>whether/how something moves, including speed and acceleration</td></tr><tr><td>number</td><td>object count, including stages or instances of a repeating event</td></tr><tr><td>direction material</td><td>axis or sense of motion (up/down, left/right, inward/outward)</td></tr><tr><td></td><td>physical material identity inferred from visual cues</td></tr><tr><td>object spatial</td><td>object identity and persistence across frames</td></tr><tr><td>interaction</td><td>relative position, contact, containment, layering</td></tr><tr><td>causal</td><td>contact-driven outcomes (collision, push, cut, attract)</td></tr><tr><td></td><td>whether one event correctly produces or follows another</td></tr></table>

## A.2 Scene and Question Construction

## A.2.1 System and user prompt

System Prompt (SCENE\_QUESTION\_GENERATION)   
You are an expert physicist and video benchmark designer. Generate physically   
accurate, visually diverse scene descriptions for text-to-video model evaluation.   
Respond ONLY with valid JSON.   
User Prompt Template (SCENE\_QUESTION\_GENERATION)   
Physical phenomenon: {category\_name}   
Description: {description}   
Physical principle: {physical\_principle}   
Physics domain: {physics\_domain}   
Object pool (choose from or invent similar):   
{object\_pool}   
Environment pool:   
{environment\_pool}   
Already existing scene captions (DO NOT duplicate, ensure diversity):

{existing\_captions}   
Generate exactly {n\_scenes} new, diverse scenes as a JSON array.   
Each scene object must have these exact keys:   
{   
"scene\_name": "<short 3-7 word title>",   
"scene\_caption": "<1-2 sentence description for a T2V prompt>",   
"generation\_prompt": "<2-3 sentence prompt with object, setting, action, expected outcome>",   
"causal\_chain": {   
"initial\_state": "<what is present before the event>",   
"process": "<what physical process happens during the clip>",   
"final\_state": "<what state objects reach by end of the 4-s clip>"   
},   
"questions": [   
{ "id": "01",   
"question\_text": "<yes/no question about what should be observed>",   
"dimensions": ["<one or two of: existence, size, appearance/texture, shape,   
motion, number, direction, material, object, spatial, interaction, causal>"],   
"difficulty": "<easy|medium|hard>",   
"frame\_indices": [<2-5 frame indices from 0-119>],   
"answer": <1 for yes, 0 for no> },   
... (8 questions total: EXACTLY 4 with answer=1 and EXACTLY 4 with answer=0)   
]   
}   
Rules for questions:   
1. Question number: The number of questions for each scenario is 8, with a positive and negative   
sample ratio of approximately 1:1.   
2. answer=1: observable facts a correctly generated video SHOULD show.   
3. answer=0: physically INCORRECT behaviors that a bad video might show.   
4. Vary difficulty: at least 2 easy, 2 medium, 2 hard per scene.   
5. Frame indices from [0, 119] (4-s clip at 30 fps); use [20, 60, 100] for spread.   
6. Cover at least 3 different dimensions across the 8 questions.   
7. Questions must be concrete and visually answerable.   
8. Do not repeat the same question pattern across all 8 questions.   
Respond with a JSON array of {n\_scenes} scene objects only. No markdown, no extra text.

## A.2.2 Per-category construction details

Each card below lists the inputs that parameterize the prompt above for one physical category: the natural-language description, the governing physical principle, the object and environment pools sampled for variety, the key diagnostic dimensions emphasized in the questions, and the target number of scenes.

[title=Gravity\_Earth Domain: Mechanics Scenes: 25]   
Description: Object released and falls under Earth’s gravitational acceleration (9.8 m/s<sup>2</sup>).   
Physical principle: Newton’s second law with constant downward acceleration g = 9.8 m/s<sup>2</sup>; velocity grows linearly,   
distance as <sup>1</sup> gt<sup>2</sup>.   
Object pool: stone, apple, metal key, tennis ball, book, pen, coin, glass marble, leaf,feather, hammer, egg, orange, plastic   
cup, wooden block, bolt, rubber eraser, walnut, rubber ball, ceramic mug   
Environment pool: table edge, held by a hand, tree branch, shelf, window ledge, rooftop, ladder rung, cliff edge   
Key dimensions: motion, direction, spatial, causal

[title=Gravity\_Moon Domain: Mechanics Scenes: 20]   
Description: Object falls under the Moon’s weak gravity (∼ 1.6 m/s<sup>2</sup>), noticeably slower than on Earth.   
Physical principle: Lunar gravity is ∼ 1/6 of Earth’s; trajectory is parabolic with stretched timescale.   
Object pool: astronaut helmet, wrench, feather, stone, ping-pong ball, apple, golf ball, metal bolt, sand bag, rubber ball,   
glass marble, coin, pen, wooden cube, hammer   
Environment pool: lunar surface, moon crater, low-gravity habitat, astronaut’s hand, space station floor   
Key dimensions: motion, direction, spatial, causal

```ini
[title=Collision_Elastic Domain: Mechanics Scenes: 25]
Description: Two objects collide and rebound with kinetic energy conserved; objects bounce apart.
Physical principle: Momentum and kinetic energy are both conserved; objects deform momentarily and rebound at
predictable velocities.
Object pool: rubber ball, tennis ball, basketball, billiard ball, ping-pong ball, steel ball bearing, elastic ring,football,
superball, golfball, handball, inflated balloon, hollow plastic ball, hard rubber cube, bowling ball
Environment pool: hard wall, concrete floor, wooden floor, racket surface, glass surface, another ball, metal plate
Key dimensions: motion, direction, interaction, causal
```

## [title=Collision\_Inelastic

Domain: Mechanics Scenes: 25]

Description: Objects collide and stick together or deform permanently; kinetic energy is lost. Physical principle: Momentum is conserved but kinetic energy is not; objects deform, stick, or shatter, and relative velocity decreases.

Object pool: wet clay, soft dough, mud ball, ripe tomato, rice cake, putty, overripe banana, raw egg, soft cheese, cookie dough, wet sand ball, mashed potato ball, playdough cube, very ripe peach, soft butter block Environment pool: wall, hardfloor, wooden board, glass window, shirtfabric, another clay lump Key dimensions: motion, shape, interaction, causa

## [title=Elastic\_Deformation

Domain: Mechanics Scenes: 25]

Description: Material is compressed or stretched and returns to its original shape when force is removed.

Physical principle: Hooke’s law: restoring force proportional to displacement; elastic strain energy is fully recovered. Object pool: stress ball, rubber band, sponge, spring, trampoline, foam pillow, rubber eraser, silicone mold, elastic hair tie, bungee cord, rubber glove, inflated balloon, memoryfoam, soft rubber cube, latex tube

Environment pool: hand grip, between twofingers, on aflat surface, hangingfrom a hook, pressed against a wall Key dimensions: shape, motion, material, causal

## [title=Inelastic\_Deformation

Domain: Mechanics Scenes: 20]

Description: Material is deformed beyond its elastic limit and retains the new shape permanently.

Physical principle: Plastic deformation: stress exceeds yield strength, causing permanent rearrangement of internal structure without recovery.

Object pool: clay lump, playdough, soft lead cylinder, wax block, putty, warm butter stick, wet sand mound,foil ball, soft cheese block, raw dough ball, modeling clay ring, beeswax slab, overripe avocado, mud cake, soft chocolate block Environment pool: hand,flat surface, mold, rolling pin,finger press, hammer strike, wooden stamp Key dimensions: shape, material, motion, causal

[title=Friction\_High

Domain: Mechanics Scenes: 25]

Physical principle: Kinetic friction $F = \mu _ { k } T$ ; high µ yields large deceleration, with possible heat or wear.

Environment pool: rough concrete, thick grass, sandpaper surface, wooden table, carpet, gravel path, rubber mat, coarse stone floor

Key dimensions: motion, direction, spatial, causal

[title=Friction\_Low

Domain: Mechanics Scenes: 25]

Description: Object glides over a low-friction surface and travels a long distance before stopping.

Physical principle: Low coefficient of friction allows kinetic energy to persist; sliding distance much greater than on rough surfaces.

Object pool: curling stone, Lego block,flat stone, ice cube, hockey puck, polished metal disc, glass marble, air hockey puck, Teflon-coated block, smooth plastic toy, waxed wooden block, butter stick, soap bar,frozenfishfillet, steel washer Environment pool: ice surface, glass table, polished marble floor, waxed wooden floor, wet tile, oiled metal plate, playground slide, Teflon sheet

Key dimensions: motion, direction, spatial, causal

[title=Pendulum

Domain: Mechanics Scenes: 20]

Description: A mass suspended from a fixed pivot swings back and forth in periodic motion.

Physical principle: Simple pendulum: $T = 2 \pi { \sqrt { L / g } } .$ , mass-independent; potential and kinetic energy alternate;   
amplitude decays slowly.

Object pool: metal ball, wooden sphere, stone weight, clock pendulum bob, steel nut on string, glass ball, ceramic bead, rubber ball, lead weight, brass sphere, acorn, walnut on string, small lantern, crystal pendant, metal ring

Environment pool: string attached to ceiling, rod pivot, grandfather clock, playground swing, hanging from door frame, outdoor tree branch

Key dimensions: motion, direction, spatial, causal

## [title=Rolling\_Motion

Domain: Mechanics Scenes: 25]

Description: A round object rolls along a surface, exchanging translational and rotational kinetic energy.

Physical principle: Rolling without slipping: $v = \omega r ;$ total $\begin{array} { r } { \check { \mathrm { K E } } = \frac { 1 } { 2 } m v ^ { 2 } + \frac { 1 } { 2 } I \omega ^ { 2 } ; } \end{array}$ ; mass distribution affects roll rate. Object pool: bowling ball, basketball, marble, coin, cylinder can, tennis ball, wooden barrel, metal pipe, wheel, soup can, empty paint roller, tire, log, wooden spool, hollow ball

Environment pool: inclined ramp, flat floor, grass, carpet, kitchen counter, bowling lane, concrete slope Key dimensions: motion, direction, spatial, causal

[title=Projectile\_Motion

Domain: Mechanics Scenes: 25]

Description: An object is launched and follows a parabolic arc under gravity with no propulsion mid-flight. Physical principle: Horizontal velocity constant, vertical velocity grows at g; range depends on launch angle and speed. Object pool: baseball, basketball, football, tennis ball, stone, frisbee, javelin, shuttlecock, golf ball, paper airplane, rubber ball, coin, water balloon, beanbag, snowball

Environment pool: openfield, gymnasium, basketball court, garden, rooftop, stadium, beach, playground Key dimensions: motion, direction, spatial, causal

[title=Structure\_Failure

Domain: Mechanics Scenes: 20]

Description: A structure collapses under excessive load, impact, or material fatigue.

Physical principle: Stress exceeds ultimate strength; failure propagates from the weakest point and pieces fall under gravity.

Object pool: wooden bridge, sandcastle, card tower, Jenga tower, snow fort, styrofoam arch, clay pillar, dominoes structure, matchstick bridge, balsa woodframe, plaster column,foam house, sugar-cube tower, cracker wall Environment pool: table, sand, floor, ground, platform, display surface

Key dimensions: shape, motion, spatial, causal

## Fluids.

[title=Buoyancy\_Float

Domain: Fluid Scenes: 25]

Description: Object placed in a fluid displaces its weight in fluid and floats at the surface.

Physical principle: Archimedes’ principle: $F _ { b } = \rho _ { \mathrm { f l u i d } } V _ { \mathrm { d i s p } } g ;$ object floats when its average density is below fluid density.

Object pool: rubber duck, sponge, ice cube, wooden toy, plastic bottle, cork,foam block, ping-pong ball, toy boat, wax candle, apple, coconut, empty plastic cup, styrofoam sphere, balsa wood plank, hollow metal ball, pumice stone, sealed empty bottle, wooden log, inflated balloon

Environment pool: bathtub, glass tank, swimming pool, ocean, lake, bucket,fish tank, river, puddle Key dimensions: motion, spatial, material, causal

[title=Buoyancy\_Sink

Domain: Fluid Scenes: 25]

Description: Object placed in a fluid is denser than the fluid and sinks to the bottom.

Physical principle: When average density exceeds fluid density, net force is downward and the object accelerates toward the bottom.

Object pool: iron block, heavy rock, steel spoon, anchor, ceramic cup, metal coin, glass marble, lead weight, bronze statue, copper pipe, stone pebble, iron nail, metal bolt, ceramic tile, dense rubber ball, concrete cube, sand-filled bag, zinc washer, cast-iron pan, brass weight

Environment pool: glass tank, bathtub, swimming pool, bucket,fish tank, oceanfloor, clear container, aquarium Key dimensions: motion, spatial, material, causal

[title=Viscous\_Flow

Domain: Fluid Scenes: 25]

Description: A high-viscosity fluid flows slowly and resists deformation, spreading sluggishly.

Physical principle: Navier–Stokes for high-viscosity Newtonian fluids; flow rate ∝ 1/µ; thick fluids hold shape and resist spreading

Object pool: honey, condensed milk, thick oil, molasses, blueberry jam, peanut butter, caramel sauce, corn syrup, tar, maple syrup, thick batter, ketchup, chocolate sauce, lava, tree sap, gel, whipped cream, thick glue, paint, yogurt Environment pool: toast, plate, pancakes, spoon, inclined surface,funnel, glass jar, bowl, bread slice, waffle Key dimensions: motion, shape, spatial, causal

[title=Splash

Domain: Fluid Scenes: 25]

Description: An object impacts a liquid surface, displacing fluid upward into a crown or column.

Physical principle: Impact converts KE into surface waves and ejected droplets; crown height ∝ impact speed; droplet size depends on surface tension.

Object pool: strawberry, grape, cherry, marble, small stone, lemon slice, ice cube, toy ball, metal coin, sugar cube, rubber duck, tomato, blueberry, walnut, apple slice, strawberry cream, small rubber ball, ping-pong ball, glass bead, small steel ball

Environment pool: glass of water, glass of milk, glass of juice, shallow pool, puddle, aquarium, swimming pool Key dimensions: motion, spatial, shape, causal

[title=Surface\_Tension

Domain: Fluid Scenes: 20]

Description: Liquid surface acts like a stretched membrane; small objects rest on it and droplets form spherical shapes. Physical principle: Surface tension γ from cohesive forces at the liquid–air interface; droplet pressure $\mathbf { \bar { \Delta } } P = 2 \gamma / \overset { \mathbf { \bar { \mathbf { \rho } } } } { r } .$ Object pool: water droplet, soap bubble, water strider insect, needle, paper clip, small coin, thin metal wire, razor blade, drop ofoil, mercury droplet, dew droplet, raindrop, drop offood coloring, small leaf, thin plastic disc

Environment pool: water surface, glass ofwater, soap solution, spider web, window glass, petri dish, calm pond Key dimensions: shape, spatial, interaction, causal

[title=Capillary\_Action

Domain: Fluid Scenes: 20]

Description: Liquid rises or spreads through narrow spaces against gravity due to adhesion and surface tension. Physical principle: Capillary rise $h = 2 \gamma$ cos $\theta / ( \rho g r )$ ; narrower tubes produce higher rise as adhesive forces overcome gravity.

Object pool: paper towel, thin glass tube, celery stalk, white flower stem, cotton wick, sponge, chalk stick, sugar cube, thin straw, capillary tube, brick, cloth strip, dried pasta, thin wooden stick,filter paper strip

Environment pool: colored water, ink solution, water with food dye, petri dish, beaker, shallow tray, glass of water Key dimensions: motion, direction, spatial, causa

[title=Liquid\_Layer

Domain: Fluid Scenes: 20]

Description: Two immiscible liquids of different densities separate into distinct horizontal layers.

Physical principle: Immiscible fluids stratify by density; mixing energy dissipates and layers re-separate if undisturbed. Object pool: cooking oil, water, honey, corn syrup, rubbing alcohol, dish soap, maple syrup, milk, vinegar,food-colored water, baby oil, glycerine, lamp oil, light mineral oil

[title=Liquid\_Mixing

Domain: Fluid Scenes: 25]

Description: Two miscible liquids are combined, diffusing and blending into a uniform mixture.

Physical principle: Miscible liquids mix by diffusion and convection; without stirring mixing is slow, with agitation it is rapid.

Object pool: red food dye, blue food dye, yellow food dye, milk, coffee, juice, vinegar, salt water, warm water, cold water, soap solution, sugar syrup, green tea, black ink drop, lemon juice

Environment pool: glass of water, bowl, clear container, mixing cup, aquarium, transparent pitcher

Key dimensions: appearance/texture, spatial, motion, causal

[title=Granular\_Flow

Domain: Fluid Scenes: 25]

Description: A collection of discrete solid particles flows and piles up like a fluid under gravity.

Physical principle: Granular media exhibit solid-like rest and fluid-like flow; angle of repose sets pile shape; flow rate depends on particle size and geometry.

Object pool: sand, rice, sugar, salt, coffee beans, dry soil, gravel, small pebbles, wheatflour, cornmeal, bird seed, lentils, sesame seeds, pasta pieces, glass beads, mustard seeds, crushed dry leaves, aquarium gravel,fine sawdust, coarse salt Environment pool: hand, funnel, hourglass, glass container, flat surface, pile on ground, inclined ramp, jar, bowl, transparent tube

Key dimensions: motion, shape, spatial, causal

[title=Vortex

Domain: Fluid Scenes: 20]

Description: A rotating fluid forms a funnel-shaped vortex around a low-pressure core.

Physical principle: Angular-momentum conservation makes fluid spin faster near the center; centripetal acceleration balances pressure gradient.

Object pool: water in a drain, stirred cup of tea, tornado, dust devil, whirlpool in a bathtub, stirred paint pot, milk swirling in coffee, whisk in batter, stirred aquarium, draining sink

Environment pool: bathtub, sink drain, large glass, bucket, mixing bowl, fish tank, clear container

Key dimensions: motion, shape, spatial, causal

## Materials.

[title=Fracture

Domain: Material Scenes: 25]

Description: A brittle object breaks into fragments when stress exceeds its fracture toughness.

Physical principle: Brittle fracture: crack propagates rapidly once $K _ { I c }$ is reached; no plastic flow, shards radiate from impact.

Object pool: wine glass, ceramic plate, glass bottle, mirror, ceramic vase, porcelain cup, glass window, clay pot, glass fish bowl, tile, glass sphere, china bowl, glass panel, brittle plastic ruler, crystal ornament

Environment pool: hardfloor, concrete surface, stone counter, woodenfloor, brick wall, metal anvil, another hard object Key dimensions: shape, motion, spatial, causal

[title=Slicing

Domain: Material Scenes: 25]

Description: A sharp blade moves through an object, cutting it cleanly into two or more pieces.

Physical principle: Stress concentration at the blade edge exceeds shear strength; cut quality depends on sharpness, force, and hardness.

Object pool: bread loaf, carrot, apple, cucumber, tomato, rope, cheese block, banana, clay block, soft wood, marshmallow, butter block, watermelon, mushroom, raw chicken breast, celery stalk, onion half, sponge cake, eggplant, zucchini Environment pool: cutting board, kitchen counter,flat surface, butcher block, marble slab Key dimensions: shape, interaction, motion, causal Kev dimensions: shape, interaction, motion. causal

## [title=Tearing

Domain: Material Scenes: 20]

Description: A soft or fibrous material is pulled apart, propagating a crack or tear along the material.

Physical principle: Crack-tip stress concentration drives propagation in ductile/fibrous materials; fibers stretch then break sequentially.

Object pool: newspaper, paper sheet, paper towel, plastic bag,fabric cloth, rubber sheet, thin leather,foil sheet, tissue paper, wet paper, duct tape, thin plasticfilm, rice paper, nori sheet, cottonfabric

Key dimensions: shape, motion, interaction, causal

## [title=Cloth\_Simulation

Description: Flexible fabric moves realistically under wind, gravity, or contact forces.

Domain: Material Scenes: 25]

Physical principle: Cloth modeled as a mass–spring system; gravity pulls down, tension resists stretch, aerodynamic forces cause flutter.

Object pool: silk flag, curtain, tablecloth, scarf, flag, bed sheet, cape, paper flag, light dress fabric, handkerchief, gossamer curtain, towel, canvas sail, ribbon, thin plastic sheet, parachute

Environment pool:flagpole, windowframe, chair back, clothesline, wind outdoors,fan indoors,falling through air Key dimensions: motion, shape, spatial, causal

## [title=Explosion

Domain: Material Scenes: 20]

Description: Rapid release of energy causes a pressure wave, fragmentation, and outward dispersion of debris.

Physical principle: Exothermic reaction or pressure overload releases energy faster than dissipation, creating a shock and ejecting fragments radially.

Object pool: inflated balloon, gasoline drum, pressurized container, firecracker, water balloon, plastic soda bottle, popcorn kernel, spray can, overinflated tire, bubble underwater, sealedjar heated, compressed-gas cylinder, party popper, exploding cake, pressure cookerfailure

Environment pool: open space, indoor room, underwater, concrete surface, metal container, lab bench Key dimensions: motion, spatial, shape, causal

## [title=Smoke

Domain: Material Scenes: 25]

Description: Small particles or vapor rise and disperse in air, forming visible plumes.

Physical principle: Buoyancy drives hot smoke upward (ρ<sub>smoke</sub> < ρ<sub>air</sub>); diffusion and turbulent mixing spread the plume.

Object pool: lit candle, incense stick, blown-out candle, cup of hot tea, cigarette, campfire embers, hot food, dry ice, smoke machine, chimney, burning match, hot iron, fogger device, extinguished fire, steam pipe

Key dimensions: motion, direction, shape, causal

## [title=Combustion

Description: A material burns, producing visible flame, smoke, light, and heat.

Domain: Material Scenes: 20]

Physical principle: Rapid oxidation: fuel + O → CO + H O + heat; flame color reflects temperature and fuel. Object pool: candle wick, wood log, paper sheet, match stick, dried leaf, twig, birthday candle, torch, alcohol lamp, oil lamp, campfire log, sparkler, newspaper, cardboard, wax tablet

Environment pool:fireplace, campfire pit, lab bench, candle holder, outdoor setting, metal bowl,fire pit Key dimensions: appearance/texture, motion, shape, causal

## [title=Chain\_Reaction

Domain: Material Scenes: 20]

Description: An initial perturbation triggers a cascade of sequential events, each causing the next.

Physical principle: Each element stores potential energy that is transferred upon impact; total energy propagates down the chain.

Object pool: dominoes, mousetraps, billiard balls in line,falling books, chain ofcoins, Newton’s cradle, lined-up toy blocks, stacked dominoes, bowling pins, linked magnets, falling card stack, toy train collision, cascading water buckets, toppling columns, matchstick chain

Environment pool: flat table, wooden floor, ramp, gymnasium floor, outdoor surface, display table

Key dimensions: motion, number, direction, causal

## Thermal.

[title=Melting

Description: A solid gradually transforms into a liquid as temperature rises above its melting point.

Physical principle: Latent heat absorbed at $T _ { m }$ breaks crystalline bonds; volume may shift slightly and shape collapses. Object pool: ice cube, candle wax, butter, snow patch, ice cream, chocolate bar, wax crayon, lead block, paraffin candle, ice pop,frozen yogurt, wax seal, lard block, ice sculpture, gelatin block

Environment pool: warm plate, heated pan, direct sunlight, microwave plate, warm water bath, room temperature counter, hot surface

Key dimensions: shape, appearance/texture, motion, causal

## [title=Freezing

Description: A liquid cools below its freezing point and solidifies, changing shape and texture.

Domain: Thermal Scenes: 20]

Physical principle: Latent heat is released as the liquid loses thermal energy; crystal structure forms (water expands ∼9%).

Object pool: water in tray, juice in mold, wet cloth, puddle, lake surface, dripping water, water in cup, mercury thermometer liquid,fruitjuice, coconut water, lemon juice, herbal tea, smoothie, sports drink, broth

Environment pool:freezer interior, cold winter outdoor, ice bath, refrigerator shelf, liquid nitrogen container, cold metal surface, outdoorfrost

Key dimensions: shape, appearance/texture, material, causal

[title=Evaporation

Domain: Thermal Scenes: 20]

Description: Surface molecules escape to vapor, gradually reducing liquid volume.

Physical principle: Surface evaporation rate scales with temperature, area, and vapor-pressure deficit; latent heat is absorbed.

Object pool: water puddle, wet cloth, water droplets on leaf, glass of water, wet pavement, ink droplet, rubbing alcohol drop, acetone pool, wet sand, water on hot pan, morning dew, wet shoe sole, brine droplet, perfume drop, water on glass surface

Environment pool: hot surface, outdoor sunlight, warm room, heated plate, dry air environment, desert setting Key dimensions: appearance/texture, size, motion, causal

[title=Boiling

Domain: Thermal Scenes: 20]

Description: Liquid heated to its boiling point forms vapor bubbles that rise and burst at the surface.

Physical principle: Vapor pressure equals atmospheric pressure at $T _ { b } ;$ nucleate bubbles form on the heated surface.

Object pool: water in pot, milk in saucepan, oil in pan, soup, broth, tea kettle water, sugar syrup, salt water, egg in boiling water, pasta in pot, jam in pan, hot chocolate, caramel mixture, stew liquid, gravy

Environment pool: stove burner, hot plate, campfire, induction cooker, lab hotplate,fondue pot

Key dimensions: appearance/texture, motion, spatial, causal

[title=Thermal\_Expansion

Domain: Thermal Scenes: 20]

Description: A material expands visibly as its temperature increases due to increased molecular vibration.

Physical principle: Linear expansion $\Delta L = \alpha L _ { 0 } \Delta T ;$ gases expand most, metals less.

Object pool: metal rod, balloon over bottle neck, bimetallic strip, thermometer liquid column, rubber ball, air in sealed syringe, plastic bottle with air metal lid on jar metal ring, coiled spring wire on posts, sealed air balloor

Environment pool: hot water bath, heat gun, boiling water, openflame, oven, sunlight, heating pad Key dimensions: size, shape, motion, causal

[title=Thermal\_Convection

Domain: Thermal Scenes: 20]

Description: Fluid circulates as heated portions rise and cooled portions sink, transferring heat.

Physical principle: Density decreases with temperature; Rayleigh–Bénard cells form in heated fluid layers.

Object pool: colored hot water, lava lamp fluid, hot cocoa, ink in warm water, heated oil, glycerine with dye, water with food coloring, heated aquarium water, boiling soup, warm sugar solution

Environment pool: transparent container, lava lamp, glass bowl, aquarium, heated beaker, lab flask

Key dimensions: motion, direction, spatial, causal

[title=Crystallization

Domain: Thermal Scenes: 20]

Description: Dissolved or molten material forms ordered crystal structures as it cools or concentrates. Description: Dissolved or molten material forms ordered crystal structures as it cools or concentrates.

Physical principle: Crystal nucleation and growth: ions/molecules arrange into a minimum-energy lattice; rate depend on supersaturation.

Object pool: salt solution, sugar solution, borax solution, copper sulfate solution, alum solution, cooling chocolate, honey crystallizing, cooling wax, cooling metal cast, rock candy solution,frost on window

Environment pool: petri dish, jar, glass container, cooling surface, cold window, refrigerator shelf

Key dimensions: shape, appearance/texture, size, causal

## [title=Condensation [title=Condensation

Domain: Thermal Scenes: 20] Domain: ThermalScenes: 20]

Description: Water vapor cools below dew point and condenses into liquid droplets on a cold surface.   
Physical principle: Surface temperature below dew point releases latent heat; droplets grow, coalesce, and run. Object pool: cold glass, cold metal can, bathroom mirror, cold window glass, cold pipe, ice-cold bottle, refrigerator exterior, cold stone surface, cold car windshield, cold metal plate   
Environment pool: warm humid room, bathroom, outdoor morning, refrigerator exterior, kitchen, greenhouse Key dimensions: appearance/texture, size, spatial, causal

## Electromagnetism.

[title=Reflection

Domain: Electromagnetism Scenes: 25]

Description: Light or image bounces off a reflective surface at equal angles of incidence and reflection. Physical principle: Law of reflection: $\theta _ { i } = \theta _ { r } ;$ specular surfaces produce mirror images, diffuse surfaces scatter light. Object pool: dancer in mirror, person waving, car on wet road, sunlight on lake, man brushing teeth, ball rolling to mirror, candle flame in mirror, hand gesturing, text on mirror, laser beam on mirror, lamp reflected in polishedfloor,face in spoon reflection, sky in puddle, building in glassfacade,fire reflected in water

Environment pool: flat mirror, glass window, still lake, wet road, polished metal, calm water pool, bathroom mirror, shop window, chrome surface

Key dimensions: direction, spatial, motion, appearance/texture

[title=Refraction

Domain: Electromagnetism Scenes: 25]

Description: Light bends at the boundary between media of different refractive indices, displacing apparent positions. Physical principle: Snell’s law: n sin $\theta _ { 1 } = n _ { 2 }$ sin $\theta _ { 2 } ;$ objects in water appear shifted; lenses and prisms exploit refraction.

Object pool: spoon in water, straw in glass, pencil in water, banana behind glass, coin at bottom of bowl, fish in aquarium, fork in water, pen in glass of water, laser beam entering water, prism splitting white light, magnifying glass focusing light, straw appearing at angle, ruler in water, chopstick in glass

Environment pool: glass of water, aquarium, glass prism, lens setup, fish tank, clear water bowl, glass cylinder Key dimensions: shape, appearance/texture, spatial, direction

[title=Shadow

Domain: Electromagnetism Scenes: 25]

Description: An opaque object blocks a light source, casting a shadow whose shape and length depend on geometry. Physical principle: Shadow boundary traced by tangent rays; shadow length $\cdot = h \cdot D / d$ for source–object distance D and object–screen distance d.

Object pool: person walking, rotating basketball, Lego blocks, tree branch, moving hand puppet, swinging pendulum, passing car, rotatingfan blade, moving cloud, candle with obstacle, hand blockingflashlight,flying bird silhouette, plant on windowsill, book under lamp, spinning top

Environment pool: sunlit wall, ground under sunlight, dim room with lamp, screen behind object, painted wall, sandy ground, white sheet, whiteboard

Key dimensions: shape, direction, spatial, motion

[title=Magnetism

Domain: Electromagnetism Scenes: 20]

Description: Magnetic objects attract or repel each other, and ferromagnetic materials are drawn to magnets. Physical principle: Magnetic force F ∝ field gradient on dipoles; like poles repel and unlike poles attract; iron and nickel are strongly attracted.

Object pool: bar magnet, horseshoe magnet, neodymium magnet, iron nail, steel paperclip, iron filing, compass needle, steel key, iron washer, steel ball bearing, iron powder, refrigerator magnet, magnetic toy, metal coin, iron chain link Environment pool: table surface, air, paper substrate, glass table, sand surface, hanging from string, resting on wood Key dimensions: motion, direction, interaction, causal

[title=Electrostatics

Domain: Electromagnetism Scenes: 20]

Physical principle: Coulomb’s law $F = k q _ { 1 } q _ { 2 } / r ^ { 2 } ;$ ; triboelectricity charge generated by contact and friction $( \mathrm { e . g . , }$ balloon attracts paper).

Object pool: inflated balloon, plastic comb, PVC rod, wool fabric, bits of paper, hair strands, small styrofoam beads, thin aluminum foil strips, charged amber rod, puffed rice grains, shredded paper, thin plastic strips, pepper grains, thin string, driedflower petals

Environment pool: table top, air, hair surface, paper surface, near balloon, near charged rod, enclosedjar Key dimensions: motion, direction, interaction, causal

## Chemistry and Material Interaction.

## [title=Diffusion

Description: Particles or molecules spread from high to low concentration until uniformly distributed.

Domain: Chemistry Scenes: 20]

Physical principle: Fick’s law: flux $\propto - D \nabla C ;$ Brownian motion drives equilibrium; faster in gases, slower in liquids. Object pool: drop ofink in water,food dye drop, perfume spray, drop ofmilk in tea, potassium permanganate crystal, drop ofcolored dye, smoke in still air, drop ofcream in coffee, gas in sealed chamber, colored sugar drop in water, red wine drop in water, turmeric in water, drop ofoil in alcohol

Environment pool: glass of still water, petri dish, transparent container, still air, closed jar, aquarium Key dimensions: motion, appearance/texture, spatial, causal

## [title=Dissolution

Domain: Chemistry Scenes: 20]

Description: A solute disperses into a solvent at the molecular level, forming a homogeneous solution.

Physical principle: Solvation: solvent molecules surround solute species; rate increases with temperature, stirring, and surface area.

Object pool: salt crystal, sugar cube, effervescent tablet, bath bomb, instant coffee granules, hard candy, Alka-Seltzer tablet, vitamin C tablet, chalk piece in acid, baking soda, copper sulfate crystal, food dye tablet, bouillon cube, aspirin tablet, dry gelatin powder

Environment pool: glass ofwater, warm water bowl, cold water glass, vinegar solution, juice, clear container Key dimensions: appearance/texture, size, motion, causal

## [title=Rust\_Oxidation

Domain: Chemistry Scenes: 20]

Description: Iron or steel reacts with oxygen and moisture, forming reddish-brown iron oxide over time (time-lapse). Physical principle: Electrochemical corrosion $\mathrm { F e } + \mathrm { O } _ { 2 } + \mathrm { H } _ { 2 } \mathrm { O }  \mathrm { \bar { F } e _ { 2 } O _ { 3 } } { \cdot } n \mathrm { H } _ { 2 } \mathrm { O } ;$ porous rust permits further oxidation. Object pool: iron nail, steel nail, iron plate, steel wool, iron chain, steel bolt, iron wire mesh, cast iron pan, steel washer, iron railing, steel screw, iron beam section, steelfence post, iron pipe section, steelfile

Key dimensions: appearance/texture, size, material, causa

[title=Color\_Mixing

Domain: Chemistry Scenes: 25]

Description: Two or more colored substances combine, producing a new mixed color.

Physical principle: Subtractive (CMY) mixing absorbs complementary wavelengths; additive (RGB) mixing increases brightness.

Object pool: red paint, blue paint, yellow paint, redfood dye, bluefood dye, yellow dye solution, watercolor paints, red ink, blue ink, green dye, white paint, black paint, cyan dye, magenta dye, colored powder

Environment pool: palette, mixing bowl, glass of water, paper surface, paint tray, clear container, canvas

Key dimensions: appearance/texture, motion, interaction, causal

[title=Adsorption

Domain: Chemistry Scenes: 20]

Description: Molecules adhere to a material’s surface, removing them from solution or gas phase.

Physical principle: Surface adsorption via van der Waals or chemical bonds; capacity depends on surface area and affinity.

Environment pool: colored water, ink solution, dye bath, oil-contaminated water, clear container, transparent beaker Key dimensions: appearance/texture, motion, interaction, causa

## [title=Precipitation

Domain: Chemistry Scenes: 20]

Description: Two solutions react to form an insoluble solid that falls out of solution.

Physical principle: Solubility product exceeded $( [ \mathbf { A } ^ { + } ] [ \mathbf { B } ^ { - } ] > K _ { s p } ) ;$ precipitate nucleates, grows, and settles.

Object pool: silver nitrate and sodium chloride, barium chloride and sodium sulfate, lead nitrate and sodium iodide, calcium chloride and sodium carbonate, iron(III) chloride and sodium hydroxide, copper sulfate and sodium hydroxide, lime water and CO<sub>2</sub>, milk of magnesia forming, calcium carbonate forming, zinc hydroxide forming

Environment pool: test tube, beaker, clear container, glass mixing vessel, transparent flask

Key dimensions: appearance/texture, motion, spatial, causal

[title=Foam\_Bubble

Domain: Chemistry Scenes: 20]

Description: Gas trapped in liquid forms bubbles or foam; bubbles rise, coalesce, and burst.

Physical principle: Bubble pressure $\Delta P = 4 \gamma / r ;$ foam drains over time; surfactants stabilize the structure.

Object pool: soap foam in sink, beer foam, latte milk foam, whipped cream, shaving cream, bubble bath foam, carbonated water bubbles, lemonjuice with baking soda, hand soapfoam,fizzy drink bubbles, seafoam,foamfrom waterfall, bubbles in boiling milk, hydrogen peroxide foam, foam from soap squeeze

Environment pool: sink basin, glass, bowl, counter surface, cup, bucket, transparent container

Key dimensions: shape, motion, size, causal

## A.3 Structured Physical Priors Construction

For each scene in PHYMETRIC, we attach a structured physical prior that anchors the VLM judges to the governing law and its expected visual signatures before any question is asked. To avoid the situation where VLM judges obtains homologous LLMs priors, we utilize four different LLMs to provide physical priors to conduct our ablation experiment in Section 4.4.

## A.3.1 LLMs backbones

The structured physical priors in PHYMETRIC are generated using a frozen text-based LLM. To validate that our evaluation framework is model-agnostic and free from specific LLM-based prior shortcuts, we employ four representative LLMs to generate these structured physical priors. This selection includes models with different architectural lineages (Qwen, Mistral, and DeepSeek) and reasoning capabilities. As detailed in Table 8, Qwen3.5-9B is used as the default generator. For our ablation studies, we incorporate Qwen2.5-VL-7B-Instruct (text-mode), DeepSeek-R1-Distill-Llama-8B, and Mistral-7B-Instruct-v0.3. This diverse ensemble allows us to investigate the sensitivity of VLM judges to varying qualities of physical reasoning and ensures the generalizability of our proposed physical prior anchoring.

Table 8: LLM backbones used for structured physical prior generation. All models are run text-only and the default backbone for the released benchmark is Qwen3.5-9B in text mode.
<table><tr><td>Model</td><td>Role</td><td>Notes</td></tr><tr><td>Qwen3.5-9B (thinking)</td><td>Default backbone</td><td>Thinking suppressed via chat-template flag</td></tr><tr><td>Qwen2.5-VL-7B-Instruct</td><td>Ablation</td><td>Text-only mode; enable_thinking=False</td></tr><tr><td>DeepSeek-R1-Distill-Llama-8B</td><td>Ablation</td><td>Strips &lt;think&gt;. . . &lt;/think&gt;; budget 16,384 tokens</td></tr><tr><td>Mistral-7B-Instruct-v0.3</td><td>Ablation</td><td>Drop-in via TextLLM wrapper</td></tr></table>

## A.3.2 System prompt

The prompt is constructed from five scene-level fields: the physical phenomenon (namely physical category), scene name, scene caption, the union of question dimensions across the eight binary questions in that scene, and the benchmark video metadata (including video duration, fps, and resolution). To facilitate question-aware frame sampling, we define three deterministic reference time points at 25%, 50%, and 83% of the total duration. In our 120-frame clips, these points correspond to frames 30, 60, and 100, ensuring that the early, middle, and late physical states described in the prior align precisely with the frames sampled by the VLM judge during evaluation.

## System Prompt (PHYSICS\_PRIORS)

You are an expert physicist and video generation evaluator.   
Your task is to produce a concise but comprehensive knowledge base that will   
later help a vision-language model judge whether a short generated video   
correctly depicts a physical scene. Keep any internal reasoning brief.   
Respond ONLY with a valid JSON object, no markdown, and no extra text.

## User Prompt Template (PHYSICS\_PRIORS)

Physical phenomenon category: {phenomenon}   
Scene name: {scene\_name}   
Scene caption (the prompt given to the video model):   
"{scene\_caption}"   
Benchmark video spec: {duration\_s} seconds, {fps} fps, {total\_frames} frames total.   
Three reference time-points (benchmark frame indices):   
- Early : frame {ref\_early} (\~{t\_early} s)   
- Mid : frame {ref\_mid} (\~{t\_mid} s)   
- Late : frame {ref\_late} (\~{t\_late} s)   
Question dimensions present in this scene: {dimensions}   
Generate a structured physical prior JSON with exactly these keys:   
{   
"physical\_law": "<1-3 sentences: the governing physical law, key conditions,   
and what drives the event>",   
"expected\_sequence": {

"early": "<physical state at \~{t\_early} s: what has started, what force/process is acting>",   
"mid": "<physical state at \~{t\_mid} s: how the event has progressed, what is changing>",   
"late": "<physical state at \~{t\_late} s: approaching final state, what outcome is visible>"   
},   
"physics\_signatures": [   
"<MUST-observe consequence 1: a physically necessary change/motion/interaction   
that the law dictates –- describe the phenomenon type, NOT a specific pixel   
position or frame number>",   
"<MUST-observe consequence 2: what would be impossible or implausible if this   
law is violated>",   
"<MUST-observe consequence 3: the key state transition that distinguishes   
physically correct from incorrect behavior>"   
],   
"common\_failures": [   
"<typical video-generation failure 1: how the model usually gets this wrong>",   
"<failure 2>",   
"<failure 3>"   
],   
"evaluation\_hints": {   
{dimension\_hints\_template}   
}   
}   
Rules for physics\_signatures:   
- Describe observable \*physical phenomena\* (acceleration, deformation, state change,   
trajectory type).   
- Do NOT prescribe exact pixel positions, object locations, or specific frame timestamps.   
- Each item should be something a viewer can confirm from any reasonable depiction   
of the correct physics.   
Be concrete and reference the actual objects in the scene caption.

## A.4 VLMs Evaluation Details

## A.4.1 System and user prompt

For every question, the VLM judge receives (i) a fixed system message specifying the judging contract, (ii) a small set of frames sampled at the question’s frame\_indices, and (iii) a user message that splices together the scene caption, the video specification, the structured physical prior, and the binary question. The judge is required to generate ANSWER, CONFIDENCE, and REASON on three lines so that downstream evaluation can extract a {yes, no} answer with a calibrated confidence.

System Prompt (VLM\_REASONING)   
You are a physics-aware video evaluation assistant. You will be shown a   
sequence of frames from a short AI-generated video, a structured physical prior   
describing what the video SHOULD depict physically, and a yes/no question about   
the video content. Reason carefully using the physical knowledge, then give your   
answer.   
Your response MUST follow this exact format (one line each):   
ANSWER: YES   
CONFIDENCE: 0.92   
REASON: <one sentence explaining your judgment>

## User Prompt Template (VLM\_REASONING)

Scene caption: "{scene\_caption}"   
Video spec: {video\_fps} fps, {video\_frames} frames total, \~{duration} s. Shown frames are   
proportionally sampled from this video.   
Frames shown correspond to benchmark time-points: [{frame\_indices}] / 120   
–- STRUCTURED PHYSICAL PRIOR –-   
Physical law: {physical\_law}   
Expected temporal sequence:   
• Early: {expected\_sequence.early}   
• Mid: {expected\_sequence.mid}   
• Late: {expected\_sequence.late}   
Physical law signatures (what MUST be observable in a correct video):   
• {physics\_signatures[0]}   
• {physics\_signatures[1]}   
• {physics\_signatures[2]}   
Common generation failures (what to watch for):   
• {common\_failures[0]}   
• {common\_failures[1]}

• {common\_failures[2]}   
Evaluation hints for this question’s dimensions:   
• [{dim}] {evaluation\_hints[dim]}   
–- END STRUCTURED PHYSICAL PRIOR –-   
Question (answer YES or NO): {question\_text}

## A.4.2 VLMs judges

We evaluate three VLM judges that span the open-source and closed-source frontier: two open-source Qwen-family checkpoints at different parameter scales, and a closed-source GPT model. All judges receive the same prompt and video input described above.

Table 9: VLM judges used in PHYMETRIC. “Default” marks the judge whose scores are reported in Table 2 and Table 3, and the others are used for cross-evaluator robustness checks in subsection 4.4).
<table><tr><td>Judge</td><td>Family</td><td>Source</td><td>Params</td><td>Role</td></tr><tr><td>Qwen3.5-9B</td><td>Qwen3.5 VL</td><td>Open-source</td><td>9B</td><td>Default judge</td></tr><tr><td>Qwen2.5-VL-7B-Instruct</td><td>Qwen2.5 VL</td><td>Open-source</td><td>7B</td><td>Same family, earlier generation</td></tr><tr><td>GPT-5.4</td><td>GPT (OpenAI)</td><td>Closed-source</td><td>一</td><td>Cross-family closed-source judge</td></tr></table>

## A.4.3 Video generators

We score four open-source text-to-video models that span 1.3 B–14 B parameters, two resolution tiers, and two output frame rates. Every model is run on the full PHYMETRIC suite of 1,115 scenes with the same caption as the video generation prompt.

Table 10: Text-to-video generators evaluated by PHYMETRIC.
<table><tr><td>Generator</td><td>Architecture</td><td>Params</td><td>Resolution</td><td>Frames</td><td>FPS</td><td>Dur. (s)</td><td>Scenes</td></tr><tr><td>CogVideoX-5B</td><td>DiT (T2V)</td><td>5B</td><td>720×480</td><td>81</td><td>16</td><td>5.06</td><td>1,115</td></tr><tr><td>Wan2.1-1.3B</td><td>Wan T2V</td><td>1.3 B</td><td>832×480</td><td>81</td><td>16</td><td>5.06</td><td>1,115</td></tr><tr><td>Wan2.2-14B</td><td>Wan MoE T2V</td><td>14B</td><td>832×480</td><td>81</td><td>16</td><td>5.06</td><td>1,115</td></tr><tr><td>Cosmos-7B</td><td>Cosmos Diffusion T2W</td><td>7B</td><td>1280×704</td><td>121</td><td>30</td><td>4.03</td><td>1,115</td></tr></table>

## A.4.4 Evaluation cases

We illustrate the end-to-end evaluation pipeline on three representative scenes drawn from different physics domains. For each case we show (i) the scene caption used verbatim as the T2V prompt, (ii) a compact rendering of the structured physical prior, (iii) one of the eight balanced binary questions, and (iv) the judge’s parsed response. The frame indices listed in each box are the benchmark indices passed to the VLM judge; the corresponding visual frames are sampled from the generated video.

Case 1 — Gravity\_Earth: “Apple released from a tree branch”

Generator: CogVideoX-5B

Frame indices shown to judge: [20, 60, 100] / 120

Scene caption: “A ripe apple is released from a tree branch above a grassy lawn and falls under Earth’s gravity until it lands on the ground.”

![](images/69b1853246a85faee76f5eeb3028264b8c4f9a6b086fb5040db2e863e2a66bcb.jpg)  
Structured physical prior (excerpt).

• Physical law: Constant downward acceleration $g \approx 9 . 8 \mathrm { { m } / \mathrm { { s } ^ { 2 } ; } }$ ; vertical velocity grows linearly, distance as ${ \scriptstyle { \frac { 1 } { 2 } } } g t ^ { 2 }$

• Early: apple just released, near the branch, vertical speed small.

• Mid: apple visibly accelerated, midway between branch and ground.

• Late: apple has reached the ground, no upward motion afterward.

• Signature: downward speed monotonically increases until contact; trajectory is straight-vertical, not floating, hovering, or rising.

Question (id=02, dimension=motion, difficulty=easy, answer=1):

“Does the apple’s downward speed visibly increase between the early and middle frames?”

VLMjudge response: ANSWER: NO CONFIDENCE: 0.83

REASON: The apple’s vertical position is essentially unchanged across the three frames; it remains hovering just below the branch instead of accelerating downward toward the ground.

Verdict: judge disagrees with ground truth (correct answer is YES) ⇒ question scored incorrect for CogVideoX-5B. This case illustrates a typical gravity failure: the generator produces a near-static apple instead of free fall, and the structured prior’s “downward speed monotonically increases” signature lets the judge correctly flag the absence of acceleration.

## Case 2 — Buoyancy\_Float: “Rubber duck placed in water”

Generator: Wan2.1-1.3B

Frame indices shown to judge: [25, 65, 105] / 120

Scene caption: “A rubber duck is placed into water, displaces the water, and floats at the surface in a stable position.”

![](images/83f1a874148c9cf303b152f3463b48c4b585055ff0e142a65457061518c94a8b.jpg)  
frame 25 (\~0.83 s)

![](images/d65f271419cc4b0768dfc9ad73ee7ed590998a0e3c5ff706780ba6675ba74801.jpg)  
frame 65 (\~2.17 s)

![](images/3a86975786828a7a8a186b3f0f0031175b50a59ae4b9764b683bb0c205b46ce8.jpg)  
frame 105 (\~3.50 s)

Structured physical prior (excerpt).

• Physical law: Archimedes’ principle; the duck’s average density is below that of water, so the buoyant force balances gravity and the duck rests at the surface.

• Early: duck enters the water, briefly displacing fluid; small ripples and a short bobbing motion are visible.

• Mid: duck is partially submerged at the waterline, slowly settling into stable equilibrium.

• Late: duck floats steadily at the surface with its top half visibly above the water; no downward drift.

• Signature: the duck never fully submerges; the waterline cuts through its lower body for the entire clip.

Question (id=04, dimension=spatial, difficulty=easy, answer=1):

“Does the rubber duck remain partly above the water surface across all the input frames?”

VLM judge response: ANSWER: YES CONFIDENCE: 0.94

REASON: In every frame the duck’s body sits clearly above the waterline with only its lower half submerged, consistent with stable floating.

Verdict: judge agrees with ground truth ⇒ question scored correct. The mirror-image reflection the model renders below the waterline confirms the duck–surface contact relation that the prior’s signature requires, allowing a high-confidence positive answer.

## Case 3 — Melting: “Ice cube on a warm metal plate”

Generator: Wan2.2-A14B

Frame indices shown to judge: [15, 55, 100] / 120

Scene caption: “An ice cube is placed on a warm metal plate at room temperature; over four seconds it gradually melts into a small puddle of clear water around its base.”

frame 100 (\~3.33 s)  
![](images/f5539d58267878664a94b89b191e486d029fdb1c941596f290eb3a5083575651.jpg)  
frame 15 (\~0.50 s)

![](images/22e6c62394c87c3282699ffd107d5190110aac6e6ccbe2a90b2cdd541d66d0a1.jpg)  
frame 55 (\~1.83 s)

![](images/f2c882810d92f91433789446c10edb46489ae85e692a18c60f60e490f2a1e110.jpg)

Structured physical prior (excerpt).

• Physical law: Phase transition at melting point absorbs latent heat; solid volume shrinks while a liquid pool grows around the contact region.

• Early: ice cube intact, faint film of meltwater starting to form.

• Mid: cube visibly smaller, surrounding puddle larger and clearer.

• Late: only a small remnant of ice (or none) on top of a clear water pool.

• Signature: ice volume monotonically decreases; water volume monotonically increases; the cube does not regrow or stay constant.

Question (id=02, dimension=shape, difficulty=hard, answer=1):

“Does the ice cube’s volume visibly shrink between the early and late frames?”

VLM judge response: ANSWER: YES CONFIDENCE: 0.84

REASON: The cube spans a noticeably smaller footprint on the plate in the late frame and is partly surrounded by a thin liquid film not present early on.

Verdict: judge agrees with ground truth ⇒ question scored correct. The structured prior’s signature (“ice volume monotonically decreases; water volume monotonically increases”) is what lets the judge convert the small visual change into a confident YES instead of treating it as ambiguous.

## A.5 Experimental Details

## A.5.1 Ablation setup

PHYMETRIC Component ablation. We isolate the two main components of PHYMETRIC independently, (i) whether the structured physical prior is presented to the judge and (ii) whether the question is the scene-specific binary question or a single generic physical-plausibility question. Crossing these two binary factors yields four conditions, summarized in Table 11. Comparing A1 against A2 isolates the contribution of the prior under fixed scene-specific questions; A2 against A3 isolates the contribution of scene-specific questions in the absence of a prior; and A1 against A4 tests whether scene-specific questions remain useful once the prior is provided. All four conditions are run on every video model with all three VLM judges in Table 9 on a stratified subset of N scenes per category.

Table 11: Part $\mathbf { A } - 2 \times 2$ component ablation: structured physical prior crossed with question specificity. A1 is the full system; A2–A4 each disable one component. All conditions are evaluated on every video model with all three VLM judges.
<table><tr><td>Condition</td><td>Prior?</td><td>Specific Q?</td><td>Tests</td></tr><tr><td>A1</td><td>√</td><td>√</td><td>Full system (reference)</td></tr><tr><td>A2</td><td></td><td>√</td><td>Contribution of the structured prior</td></tr><tr><td>A3</td><td></td><td></td><td>Raw VLM vision baseline</td></tr><tr><td>A4</td><td>√</td><td></td><td>Prior without scene-specific QA</td></tr></table>

Generic question prompt. Conditions A3 and A4 replace the scene-specific yes/no question with the fixed generic question shown below. The generic question is intentionally minimal: it asks only whether the video respects the caption’s physical behavior, with no reference to specific objects, frame indices, or expected outcomes, and therefore cannot benefit from the structured prior except indirectly through the unchanged scene caption.

## Generic Question

Does this video correctly simulate the physical behavior described in the scene caption? Answer YES or NO.

Cross-source prior analysis. A natural concern is that improvements from the structured prior could be circular when the same model both authors the prior and judges the video. To address this, Part B crosses three VLM judges with five prior sources – a “no prior” baseline plus four priors authored by independent text LLMs (Qwen3.5-9B, Qwen2.5-VL-7B, DeepSeek-R1-Distill, and Mistral-7B), yielding 3×5 = 15 conditions per video model. A judge that did not author the prior but still benefits from it (for example, the closed-source judge evaluated with a Qwen-authored prior) provides evidence that the prior encodes transferable physical content rather than judge-specific shortcuts.

## A.5.2 Human study

Scene subset. We hold out a stratified 100-scene subset covering all 50 categories with ∼2 scenes per category, biased toward scenes whose physical signatures are unambiguous to a non-expert viewer. Each scene is paired with its 8 expert-curated binary questions, giving 800 (scene, question) annotation units. We collect human responses against videos generated by all four T2V models.

Annotators We recruit 50 human annotators from graduate-level or above academic groups, including students and researchers with general STEM backgrounds. Annotators are not required to be physics experts, since the study is intended to measure whether the physical consequence is visually recognizable to an informed viewer. Before annotation, each participant is given a short instruction page explaining the yes/no format, the meaning of sampled frames, and examples of valid and invalid judgments. Annotators are asked to answer based only on the provided video evidence and scene caption, without using external resources.

Annotation interface. As shown in Figure 5, annotation is conducted through a self-hosted Flask web app that, for every (annotator, scene, question) triple, displays (i) the scene caption, (ii) the sampled frames at the question’s frame indices, (iii) an optional embedded full-resolution video, and (iv) the binary question. Question order is shuffled per annotator to control for ordering effects; the structured physical prior is not shown to annotators, where they answer from the video alone.

Use of human responses. Aggregated human answers serve two purposes in the main paper. First, they provide a ceiling reference: per-question majority human accuracy upper-bounds what any automatic judge can achieve when the videos are visually ambiguous. Second, they support the correlation with human judgment analysis (Section 4), where per-scene human accuracy is correlated with the PHYMETRIC score under each ablation condition to verify that improvements over baselines reflect alignment with human physical perception rather than artifacts of the prompt.

## A.6 Limitations and Future Work

PHYMETRIC focuses on observable physical consequences in generated videos. This design makes the benchmark interpretable and scalable, since each score can be traced to question-specific visual evidence. However, it also limits what kinds of physics can be verified. The current taxonomy covers 50 physical categories across six domains, but it is not exhaustive. Some physical laws, such as energy conservation cannot be reliably judged from sparse sampled RGB frames alone. Future versions can extend PHYMETRIC with denser frame evidence, depth cues, object trajectories, tracking annotations, and simulation-derived states to support more quantitative physical checks.

Another limitation comes from the binary QA format. Yes/no questions make the score easy to interpret and reduce ambiguity, but they simplify physical outcomes that may have multiple valid trajectories or degrees of correctness. For example, a generated object may deform in a plausible but not exact way, or a fluid may spread with a reasonable but different temporal pattern. Future work can combine binary QA with graded labels, multiple-choice physical outcomes, and continuous measurements to better capture partial correctness.

![](images/904c951eb5bc786b9ae14f693dd190614500d04799539a19751799d734f3fcfa.jpg)  
Figure 5: The interface of PHYMETRIC human study.

The automatic score also depends on the VLM judge. Our ablations show that scene-specific questions are the most stable component, while structured priors affect different evaluators differently. This suggests that future physical evaluation will benefit from stronger video-native VLMs, better uncertainty calibration, and selective human-in-the-loop verification, especially for subtle temporal or material changes.

Each model is evaluated with one generated video per scene for efficiency and controlled comparison across models. This setting does not capture sampling variance across random seeds. Future work should evaluate multiple generations per scene and report confidence intervals or bootstrap error bars. Beyond evaluation, PHYMETRIC can also support training: failed question labels, dimension-level scores, and structured priors can be used to construct reward models, preference pairs, or targeted fine-tuning data for physically grounded video generation.

Finally, PHYMETRIC evaluates generated videos through question-aware sampled frames rather than full continuous video reasoning. This design makes the evidence shown to the VLM explicit and reproducible, but it can also be rigid. If the sampled frames miss a short-lived event, such as the exact moment of contact, deformation, splash, or phase transition, the VLM may judge from incomplete evidence. Similarly, fixed frame indices may not align well with videos whose event timing differs from the expected temporal schedule. Future work can replace fixed frame sampling with adaptive temporal localization, event-aware frame selection, or video-native evaluators that reason over the full clip while still reporting the evidence used for each decision.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper’s contributions and scope?

Answer: [Yes]

Justification: We describe our benchmark’s contribution in abstract and introduction section in detail. Our PhyMetric provides a diagnostic benchmark for video physical plausibility evaluation.

Guidelines:

• The answer [N/A] means that the abstract and introduction do not include the claims made in the paper.

• The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A [No] or [N/A] answer to this question will not be perceived well by the reviewers.

• The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.

• It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: We have claimed our limitation in subsection A.6.

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

## Answer: [N/A]

Justification: The paper does not include theoretical results.

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

Justification: The experimental implementation details are described in section 4, and all datasets used in this work are open accessed online and they are referenced soundly in paper.

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

## Answer: [Yes]

Justification: We have released the evaluation scripts in the anonymous code site https://anonymous.4open.science/r/PhyMetric-Benchmark/. We also provide sample 10 scenes across four video generators, three VLM judges, and human evaluation in the supplementary material.

## Guidelines:

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

Justification: We provide detailed dataset usage in section 4.

Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.

• The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [No]

Justification: Our evaluation uses protocols from previous benchmarks are not accompanied by error bars, confidence intervals, nor statistical significance tests.

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

Justification: We provide the detailed computation resources to reproduce the experiments in section 4 and the evaluation code on three VLM judges is provided in the anonymous code. Please refer to Q5.

Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.

• The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.

• The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn’t make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?

Answer: [Yes]

Justification: The research conducted in the paper conforms with the NeurIPS Code of Ethics.

Guidelines:

• The answer [N/A] means that the authors have not reviewed the NeurIPS Code of Ethics.

• If the authors answer [No], they should explain the special circumstances that require a deviation from the Code of Ethics.

• The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [Yes]

Justification: This paper provides a scene-level diagnostic benchmark for evaluating physical plausibility in text-to-video generation.

## Guidelines:

• The answer [N/A] means that there is no societal impact of the work performed.

• If the authors answer [N/A] or [No], they should explain why their work has no societal impact or why the paper does not address societal impact.

• Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.

• The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate Deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.

• The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.

• If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).

## 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pre-trained language models, image generators, or scraped datasets)?

Answer: [N/A]

Justification: This work poses no such risks.

Guidelines:

• The answer [N/A] means that the paper poses no such risks.

• Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.

• Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.

• We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: We cited all works of codes and datasets used in the paper.

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

Justification: PHYMETRIC introduces a new benchmark dataset with scene definitions, QA annotations, physical priors, and evaluation code, documented in section 3 and Appendix A.1.

Guidelines:

• The answer [N/A] means that the paper does not release new assets.

• Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.

• The paper should discuss whether and how consent was obtained from people whose asset is used.

• At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [Yes]

Justification: The paper includes a human validation study in which annotators answer the same scene-specific yes/no questions used by PHYMETRIC on the controlled 100- scene subset. We provide the annotation instructions, interface screenshots, quality-control protocol, and compensation details in Appendix A.5.2.

Guidelines:

• The answer [N/A] means that the paper does not involve crowdsourcing nor research with human subjects.

• Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.

• According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [No]

Justification: This study involved low-risk benchmarking annotation only and did not require formal IRB approval under local institutional guidance.

Guidelines:

• The answer [N/A] means that the paper does not involve crowdsourcing nor research with human subjects.

• Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.

• We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.

• For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigor, or originality of the research, declaration is not required.

Answer: [Yes]

Justification: We employ VLMs, including GPT-5.4, Qwen3.5-9B, Qwen2.5-7B, and LLMs, including DeepSeek-R1-8B, and Mistral-7B as the reasoning tool for our benchmark, as mentioned in section 4. The usage of LLMs and VLMs only for experimental reasoning.

Guidelines:

• The answer [N/A] means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.

• Please refer to our LLM policy in the NeurIPS handbook for what should or should not be described.