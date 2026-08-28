# R2M-Bench: Evaluating Revisit Memory via Relative Consistency in Interactive Video World Models

![](images/415a74207a88a171c5c8bd2c46a2d0184613380538b4bdd6311a18beefa12501.jpg)

Qiwen Gu<sup>1,2,∗,†</sup>, Bingjie Gao<sup>1,3,∗,†</sup>, Rui Chen<sup>1,‡</sup>, Geng Li<sup>1</sup>, Jifan Li<sup>1</sup>, Qishuai Wen<sup>1</sup>, Li Niu<sup>3</sup>, Jing Tang<sup>1,§</sup>, Xiangxiang Chu<sup>1</sup>, Junqiao Zhao<sup>2</sup>

<sup>1</sup>DreamX Team, Alibaba Group, <sup>2</sup>Tongji University, <sup>3</sup>Shanghai Jiao Tong University <sup>∗</sup>Equal contribution, <sup>‡</sup>Project lead, <sup>§</sup>Corresponding author

High similarity between first-visit and return frames does not necessarily show that a video world model remembered the scene; the intervening rollout may simply have changed very little. This ambiguity makes absolute revisit scores sensitive to rendering stability, repetitive content, and failed motion. We introduce R2M-Bench (Relative Revisit Memory Benchmark), a benchmark of observable revisitselective consistency. For every detected return, R2M-Bench compares the revisit pair with two controls from the same rollout: a gap-matched non-revisit pair that measures generic temporal stability and a short-range pair that estimates short-horizon consistency. These comparisons produce MemoryGain (MG), the revisit advantage over the temporal baseline, and the Normalized Memory Ratio (NMR), which normalizes this advantage by the short-to-baseline dynamic range. R2M-Bench combines 100 reference scenes with three leave-and-return trajectories to form 300 instances and evaluates appearance fidelity, scene and object identity, local geometry, and persistent state. Across seven action-conditioned video world models, Overall NMR correlates with human consistency judgments at Spearman’s ρ = 0.547 (95% CI [0.45, 0.63]). Its within-model correlation magnitude with generated motion is 0.072, compared with 0.207 for raw revisit similarity, indicating that relative calibration substantially reduces the slow-motion shortcut. DreamX-World-Memo achieves the highest Overall NMR among the evaluated video models. Together, these results support same-rollout relative calibration as a practical way to distinguish revisit-specific consistency from generic temporal stability.

GitHub: https://github.com/AMAP-ML/R2MBench Contact: guangyu.tj@alibaba-inc.com

![](images/a5208e8267749cd990023732bd5d0ffd87738ec8d64a132b5c3cef22b360dc99.jpg)  
ference start frames of the 100 scen(a) 100 reference scenes

![](images/1813277898768f08c27ce06a5cb07c5a51a279dd61cc1305b6fb8affa6582425.jpg)  
<sup>0</sup> <sup>2</sup> <sup>0</sup> <sup>7</sup> <sub>scenes</sub> <sub>(of</sub> <sub>100)</sub>(b) Relative revisit consistency  
Figure 1 R2M-Bench at a glance. (a) The 100 reference scenes, each paired with three leave-and-return trajectories, define 300 benchmark instances. (b) Template-balanced relative revisit consistency across five evaluation families for seven video world models; larger values indicate stronger recovery on return.

## 1 Introduction

World models provide predictive environments for planning, simulation, and embodied decision making. Video generation ofers a practical route to such environments because it can synthesize high-dimensional observations directly, and recent action- or camera-conditioned models support increasingly long, controllable rollouts (Yang et al., 2025b; Kong et al., 2024; Wan et al., 2025; Zheng et al., 2024; Bruce et al., 2024; Alonso et al., 2024; Wu et al., 2024). Visual plausibility alone, however, is not enough for interaction. When an agent leaves a room, street corner, or object arrangement and later returns, the generated world should recover the same place rather than produce another plausible scene. A commanded revisit therefore tests whether the model preserves scene identity, object state, layout, and local structure through an intervening rollout. Figure 1 previews R2M-Bench, from its reference-scene and trajectory construction to its template-balanced comparison of seven video world models.

Evaluation has not yet cleanly isolated this capability. General video benchmarks measure perceptual quality, temporal smoothness, prompt alignment, and physical plausibility (Huang et al., 2023; Zheng et al., 2025; Liu et al., 2023b). World-model benchmarks additionally examine controllability, interactive response, and long-horizon generation (Li et al., 2025a; Duan et al., 2025; Wu et al., 2026a). Recent memory-oriented benchmarks consider revisit frames or long-horizon state retention (Zhang et al., 2026; Ye et al., 2026). These eforts make persistent-world failures visible, but an absolute similarity score between a first visit and a return still leaves an important ambiguity: how much of the measured similarity is specific to the revisit, and how much simply reflects the rollout’s general visual stability?

Absolute revisit similarity is particularly dificult to compare across current world models. Even under the same commanded trajectory, models difer in rendering quality, motion magnitude, camera-execution speed, and the extent to which the generated view actually changes. As Figure 2(a) illustrates, a low-motion rollout can achieve high first-visit/return similarity because most frames remain alike; in the limiting case, the camera never visibly leaves the initial view. Larger camera motion can lower the same score even when a model successfully returns. A revisit benchmark should therefore not reward sharp or conservative rendering that is equally present at ordinary times in the rollout.

We operationalize model revisit memory as observable revisit-selective consistency. After a commanded return, the generated observation should recover the previously visited scene and state more consistently than comparable non-revisit observations from the same video. This definition concerns model behavior. It neither assumes that the model contains an explicit memory module nor attributes performance to a particular internal mechanism. The distinction is important because systems with diferent architectures, context policies, and retrieval strategies can exhibit similar output-level behavior.

R2M-Bench implements this definition through same-video temporal calibration (Figure 2(b)). Starting from a reference image and a navigation script, the benchmark evaluates the generated rollout and mines frame pairs corresponding to commanded returns. Each revisit pair is compared with two references from the same video. A gap-matched non-revisit pair estimates the model’s ordinary self-similarity over a comparable temporal interval. A short-range pair estimates the consistency available over nearby frames, before long-horizon drift dominates. MemoryGain (MG) reports the direct advantage of the revisit pair over the gap-matched baseline. The Normalized Memory Ratio (NMR) divides this advantage by the short-to-baseline dynamic range, yielding a dimensionless score that can be compared across metrics. Because all three pair types come from the same rollout, slow or nearly static generations raise the baseline as well as the revisit score and receive little selective advantage.

For broader context, Table 1 situates R2M-Bench among representative video-generation, world-model, and memory-oriented benchmarks by comparing the evaluation-design properties most relevant to revisit memory.

The benchmark combines 100 reference scenes with three leave-and-return trajectory templates, producing 300 benchmark instances. The trajectories include a direct out-and-back return, a translation followed by rotation, and a longer closed loop. Together they expose failures caused by viewpoint change, accumulated control error, autoregressive drift, and context attenuation. The reference set spans indoor, urban, natural, and abstract environments, with substantial variation in object density and scene structure.

R2M-Bench evaluates five complementary families of observable consistency. Appearance Fidelity captures changes in color, texture, and low-level perceptual appearance. Scene Identity tests whether the return remains recognizable as the same place under moderate viewpoint or rendering changes. Object Identity measures whether salient objects remain present with compatible appearance and semantics. Local Geometry tests whether local structures still support feature correspondence and a common two-view geometry. Persistent State assesses whether the pair depicts a compatible scene state after accounting for camera motion and occlusion. Applying the same relative protocol to all five families exposes partial failures that a single similarity metric would hide, such as preserving scene semantics while changing the objects or spatial layout.

![](images/dd8a5e5ef6c7d3cf53b70f855dc1bc1d2960b134f140271c13ab1a11b08f006d.jpg)  
Figure 2 Overview of R2M-Bench. (a) Absolute first-visit/revisit similarity is not suficient evidence of memory because motion magnitude and rendering stability can inflate the raw score. (b) R2M-Bench evaluates navigation-style generated rollouts by detecting leave-and-return revisit pairs from commanded trajectories, then compares each revisit pair with a gap-matched baseline and a short-range reference and reports MemoryGain and NMR across the metric families.

We evaluate seven action-conditioned video world model variants and report a separate 3D rendering reference. On 210 sampled rollouts, Overall NMR correlates with human consistency judgments at Spearman’s ρ = 0.547, with a 95% bootstrap interval of [0.45, 0.63]. An independent optical-flow diagnostic finds a substantially smaller correlation magnitude with generated motion for NMR than for raw revisit similarity (0.072 versus 0.207). DreamX-World-Memo achieves the highest Overall NMR among the evaluated video models, followed by HY-WorldPlay and Matrix-Game 3.0, while the trajectory-wise profiles show that even the strongest systems degrade on longer closed-loop returns. Retrieval-based systems occupy the top tier in this evaluation, but diferences in architecture, scale, training, and inference preclude a causal conclusion about retrieval itself.

## We make three contributions:

• We introduce R2M-Bench, a 300-instance benchmark that combines 100 diverse scenes with three complementary return trajectories and evaluates five dimensions of visual and state consistency.

• We develop a same-video relative protocol that distinguishes revisit-selective recovery from generic temporal stability. MG reports the direct revisit advantage, while NMR normalizes this advantage for cross-metric and family-level aggregation.

• We provide a multi-model empirical study with human validation, objective motion analysis, controlled motion and quality diagnostics, and trajectory-wise evaluation. These analyses characterize both the alignment and the remaining limitations of relative revisit scoring.

Table 1 Comparison with representative video-generation, world-model, and memory-oriented benchmarks from the perspective of evaluation design. Long: long-horizon or multi-turn evaluation; Mem.: explicit memory or persistent state evaluation; Rev.: constructed revisit or loop-closure test cases; View-Rob.: robustness to moderate viewpoint deviations; Rel. Score: relative scoring against within-video temporal controls rather than absolute similarity; Multiaxis: evaluation across multiple consistency levels. ✓denotes an explicit benchmark design goal; △ denotes partial or indirect coverage.
<table><tr><td>Benchmark</td><td>Long</td><td>Mem.</td><td>Rev.</td><td>View-Rob.</td><td>Rel. Score</td><td>Multi-axis</td></tr><tr><td>VBench (Huang et al., 2023)</td><td></td><td></td><td></td><td></td><td></td><td>√</td></tr><tr><td>EvalCrafter (Liu et al., 2023b)</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>WorldModelBench (Li et al., 2025a)</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>WorldScore (Duan et al., 2025)</td><td>√</td><td>△</td><td></td><td></td><td></td><td></td></tr><tr><td>WBench (Ying et al., 2026)</td><td>√</td><td>△</td><td></td><td></td><td></td><td></td></tr><tr><td>Omni-WorldBench (Wu et al., 2026a)</td><td>△</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>MBench (Zhang et al., 2026)</td><td>√</td><td>√</td><td></td><td>△</td><td></td><td></td></tr><tr><td>MIND (Ye et al., 2026)</td><td>V</td><td>√</td><td></td><td>√</td><td></td><td></td></tr><tr><td>R2M-Bench (Ours)</td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

## 2 Related Work

Video generation and long-horizon rollouts. Video generation has moved from early adversarial and autoregressive models for spatio-temporal generation and discrete visual-token prediction (Saito et al., 2017; Tulyakov et al., 2018; Yan et al., 2021) to difusion-based video synthesis that extends image difusion to the temporal domain (Ho et al., 2022; Singer et al., 2022; Blattmann et al., 2023). Recent transformer-based video foundation models improve visual fidelity, prompt following, motion quality, and temporal smoothness (Yang et al., 2025b; Kong et al., 2024; Wan et al., 2025). Since many video generators are trained on fixed-length clips, long videos are commonly produced through autoregressive or streaming rollouts, where future clips are conditioned on previously generated content. Recent methods reduce rollout degradation with self-forcing objectives, rolling or streaming denoising, and causal distillation recipes (Huang et al., 2025; Liu et al., 2025; Yang et al., 2025a). These methods provide the generative backbone for interactive video world models, but longer video alone does not ensure that the generated environment stays persistent.

Action-driven interactive video world models and memory mechanisms. Recent work has pushed world models toward action-driven interactive video generation, where future observations respond to user actions, camera controls, keyboard–mouse inputs, navigation commands, or event instructions. Earlier world models studied environment dynamics for planning and control (Ha and Schmidhuber, 2018; LeCun, 2022). Recent generative simulators and action-conditioned video models scale this idea to robotics, driving, games, and interactive environments (Hu et al., 2023; Bruce et al., 2024; AI, 2024; Sun et al., 2025; Robbyant Team et al., 2026; Shen et al., 2026; Wang et al., 2026b; Mao et al., 2026; Wu et al., 2026b; DreamX Team et al., 2026). Unlike standalone video generators, these models support continuous exploration: an agent can move, turn, revisit locations, and interact over time.

The three controlled return patterns used to instantiate the benchmark are summarized in Figure 3 and detailed in Section 3.

Long interactive rollouts need memory. Existing mechanisms include context replay or retrieval, which conditions generation on selected past frames, latents, or retrieved context (Sun et al., 2025; Hong et al., 2025; Hu et al., 2026; Peng et al., 2026); compressed memory, which stores long histories as fixed-budget tokens, object slots, or summaries (Kim et al., 2026; Dou et al., 2026; Wu et al., 2026b); state memory, which maintains a compact latent state across chunks (Chen et al., 2025b; Yu et al., 2025; Li et al., 2026; Zhu et al., 2026); and spatial or geometric memory, which organizes past views in 3D or view-indexed structures (Li et al., 2025b; Wang et al., 2026a; Shen et al., 2026). These mechanisms can reduce drift and improve temporal coherence, but they do not guarantee that a previously observed place remains persistent after leaving the field of view.

Evaluation of video/world models and revisit memory. Video generation benchmarks have expanded from generic quality assessment to multi-dimensional evaluation of visual quality, temporal stability, motion, text alignment, compositionality, physics, and long-video coherence (Huang et al., 2023, 2024; Zheng et al., 2025; Liu et al., 2023b; Sun et al., 2024; Bansal et al., 2024; Meng et al., 2024). Complementary evaluation work targets perception-aligned motion quality (Ling et al., 2025), fine-grained entity-level video quality assessment (Chen et al., 2025a), and narrative expression in long-video generation (Feng et al., 2026). World-model benchmarks evaluate instruction following, physical adherence, camera or action controllability, 3D consistency, interactive response, policy use, and long-horizon world generation (Li et al., 2025a; Duan et al., 2025; Quevedo et al., 2025; Wu et al., 2026a). Recent memory-oriented benchmarks evaluate entity or environment consistency, action control, closed-loop revisits, and long-horizon state retention (Zhang et al., 2026; Ye et al., 2026). R2M-Bench focuses on calibrating the measurement of revisit consistency. Absolute first-visit/revisit similarity can be inflated by image quality, stable style, repetitive textures, slow motion, or near-static generation, so R2M-Bench measures whether similarity is selectively elevated at revisits relative to temporal references from the same generated video. It evaluates this calibrated revisit advantage through Appearance Fidelity, Scene Identity Preservation, Object Identity, Local Geometric Correspondence, and Persistent State Reasoning.

![](images/5592cef4c734e700382737123bb417c61498ce011f870152f6cf4d71b6dc2943.jpg)

![](images/53f4fa71ba6d86c71cff7411413b1dacd824ac332d558ad07833a11fa5800546.jpg)

![](images/0ed6eab5b8ced98b11239ac1ba0aa42fe6bd9339fbcd54a10073f8cd4e107190.jpg)  
Figure 3 Bird’s-eye view of the three trajectory templates used in R2M-Bench. Color indicates temporal progression from start (blue) to end (red), and arrows indicate camera motion. (a) Out-and-back: the agent moves away and reverses to the starting pose. (b) Translation–rotation: the agent revisits the starting region after heading changes. (c) Closed-loop: the agent traverses a loop and returns to the starting pose. The ⋆ marker denotes the revisit point.

## 3 R2M-Bench Data Composition

R2M-Bench evaluates revisit memory in navigation-style interactive video world models. It pairs 100 reference scenes with each of three trajectory templates, yielding 300 benchmark instances. Each instance contains an initial reference image, a text prompt, and a canonical navigation script that induces a leave-and-return interaction. After generation, we retain the resulting video together with its reference image, prompt, script, and commanded trajectory for evaluation. Figure 1(a) shows the complete reference set. Its mutually exclusive primary partition contains 24 indoor, 44 urban, 28 natural, and 4 abstract scenes, while the non-exclusive attributes show that 87 scenes are object-rich and 76 are outdoor, with overlapping urban, natural, and indoor tags.

Trajectory Construction. Each trajectory is a model-agnostic navigation script composed of simple action primitives, including translations along the forward–backward and left–right axes and left/right yaw rotations. We convert the same script to the input format required by each model: discrete actions for action-only interfaces, or camera deltas/poses for camera-conditioned interfaces. For analysis, we integrate the primitives into a commanded pose sequence, which records the intended trajectory and is later used to mine revisit pairs.

Figure 3 shows the three trajectory templates: out-and-back for near same-view revisits, translation–rotation for viewpoint-changing revisits, and closed-loop for loop-closure revisits. These templates cover common navigation patterns and test whether the generated world preserves local appearance, place identity, and global layout consistency after revisitation.

Initial Reference Selection. For each trajectory, we select an initial reference image that defines the starting observation of the generated world. We use lightweight inclusion criteria: images should depict navigable scenes with recognizable spatial layout, stable visual anchors such as furniture, windows, road structures, signs, or distinctive objects, and enough free space for plausible camera motion. We include indoor, outdoor, urban, natural, and object-rich environments, while excluding close-ups, blank backgrounds, heavy text overlays, severe blur, and ambiguous scale. The selected references cover varied scenes while preserving the scene and object cues needed for memory evaluation.

Prompt Annotation. Given the selected reference image, we annotate a text prompt that describes the initial environment and its salient visual content. We use Gemini-3.1-Pro-Preview to draft the description, then manually normalize it into a concise generation prompt. The prompt keeps persistent cues such as scene category, spatial layout, major objects, materials, and distinctive landmarks. We remove hallucinated objects, fragile exact counts, transient lighting details, and style-heavy wording, so the text conditions the starting scene without over-constraining revisit performance.

## 4 Evaluation Method

## 4.1 Relative Revisit Evaluation

We treat revisit-memory evaluation as a relative comparison. Given a generated video $\mathbf { V } = \left( f _ { 0 } , f _ { 1 } , \ldots , f _ { T - 1 } \right)$ and its commanded pose sequence $\mathbf { P } = \left( p _ { 0 } , p _ { 1 } , \dots , p _ { T - 1 } \right)$ , absolute first-visit/revisit similarity is not enough evidence of memory. We instead compare each revisit pair with temporal references sampled from the same generated video. This comparison calibrates the revisit against the rollout’s own visual and temporal behavior, reducing the extent to which image quality, rendering stability, or low motion alone can produce a high score.

For each video, we construct three frame-pair sets: revisit pairs ${ \mathcal { R } } ,$ gap-matched baseline pairs $B ,$ and short-range reference pairs ${ \mathcal { S } } .$ . Revisit pairs test consistency at a commanded return to a previously visited region. Baseline pairs estimate the rollout’s ordinary long-gap self-similarity under a comparable temporal separation. Short-range pairs measure local temporal consistency when little long-term retention is required.

## 4.2 Frame-Pair Construction

Let $p _ { t } = ( \mathbf { t } _ { t } , \theta _ { t } )$ denote the commanded camera state at frame $t ,$ where $\mathbf { t } _ { t }$ is the camera position and $\theta _ { t }$ is the yaw angle. We denote the circular yaw distance by $d _ { \theta } ( \theta _ { i } , \theta _ { j } )$

Revisit pairs. A revisit pair is a temporally separated pair of frames whose commanded poses return to the same spatial region with a similar viewing direction. We mine revisit candidates from the commanded pose sequence:

$$
\begin{array} { r } { \mathcal { R } = \{ ( i , j ) \mid i < j , \| \mathbf { t } _ { i } - \mathbf { t } _ { j } \| _ { 2 } \leq \tau _ { \mathrm { p o s } } , d _ { \theta } ( \theta _ { i } , \theta _ { j } ) \leq \tau _ { \mathrm { r o t } } , j - i \geq \operatorname* { m a x } ( 0 . 2 T , 1 0 ) \} , } \end{array}\tag{1}
$$

where $\mathbf { t } _ { i }$ and $\theta _ { i }$ denote the commanded position and yaw angle at frame $i , d _ { \theta } ( \cdot , \cdot )$ is the circular yaw distance, and $T$ is the efective video length after trimming static leading and trailing frames. The thresholds $\tau _ { \mathrm { p o s } }$ and τ<sub>rot</sub> are set adaptively from the commanded trajectory, as described in the implementation details. When multiple earlier frames satisfy the revisit condition for the same later frame, we keep the match with the smallest combined pose diference.

Gap-matched baseline pairs. For each revisit pair $( i , j ) \in \mathcal { R }$ with temporal gap $\Delta = j - i$ , we first construct candidate baseline pairs from the same video with a similar temporal gap:

$$
\mathcal { C } _ { ( i , j ) } = \left\{ ( u , v ) ~ \vert ~ u < v , ~ \vert ( v - u ) - \Delta \vert \le \delta , ~ ( u , v ) \notin \mathcal { R } , ~ u , v \notin \mathcal { N } _ { \eta } ( \mathcal { R } ) \right\} ,\tag{2}
$$

where $\delta$ is the gap tolerance and $\mathcal { N } _ { \eta } ( \mathcal { R } )$ excludes frames within an η-frame neighborhood of revisit endpoints. We then sample one baseline pair $b _ { ( i , j ) } \in \mathcal { C } _ { ( i , j ) }$ for each revisit pair and define

$$
\begin{array} { r } { B = \left\{ b _ { ( i , j ) } ~ \middle | ~ ( i , j ) \in \mathcal { R } , ~ \mathcal { C } _ { ( i , j ) } \neq \emptyset \right\} . } \end{array}\tag{3}
$$

The baseline set is a same-video, temporal-gap-matched calibration reference. By assigning a matched non-target pair to each revisit pair, B estimates the similarity already present at ordinary long-gap moments under a comparable time budget. It is not intended to approximate a counterfactual rollout in which the model has no memory, nor does it control every diference in realized viewpoint or action execution. Its purpose is to expose generic high self-similarity: if a rollout moves slowly, remains nearly static, renders conservatively, or does not visibly leave the initial scene, baseline pairs also remain similar. We deliberately do not impose a spatial-far constraint, so these shortcuts raise the baseline rather than being excluded from calibration.

Short-range reference pairs. We also sample short-range pairs

$$
\begin{array} { r } { S = \left\{ ( a , b ) \mid \Delta _ { \operatorname* { m i n } } \leq b - a \leq \Delta _ { \operatorname* { m a x } } \right\} , } \end{array}\tag{4}
$$

where $\Delta _ { \mathrm { m i n } }$ and $\Delta _ { \mathrm { m a x } }$ define a small temporal window. These pairs measure local temporal consistency and provide a reference for the consistency that the model maintains when the time gap is short.

## 4.3 MemoryGain and Normalized Memory Ratio

We first define the scoring procedure for an arbitrary pairwise consistency measure $q .$ Since diferent measures may have diferent directions, we convert q into an oriented score $s _ { q }$ such that larger values always indicate stronger consistency:

$$
s _ { q } ( f _ { i } , f _ { j } ) = \left\{ { \begin{array} { l l } { q ( f _ { i } , f _ { j } ) , } & { { \mathrm { i f ~ h i g h e r ~ } } q { \mathrm { ~ i n d i c a t e s ~ s t r o n g e r ~ c o n s i s t e n c y } } , } \\ { - q ( f _ { i } , f _ { j } ) , } & { { \mathrm { i f ~ l o w e r ~ } } q { \mathrm { ~ i n d i c a t e s ~ s t r o n g e r ~ c o n s i s t e n c y } } . } \end{array} } \right.\tag{5}
$$

For any pair set ${ \mathcal { P } } \in \{ { \mathcal { R } } , B , S \}$ , we denote its average consistency score as

$$
\bar { s } _ { q } ^ { \mathcal { P } } = \frac { 1 } { | \mathcal { P } | } \sum _ { ( i , j ) \in \mathcal { P } } s _ { q } ( f _ { i } , f _ { j } ) , \qquad \bar { s } _ { q } ^ { \mathrm { r e v } } = \bar { s } _ { q } ^ { \mathcal { R } } , \quad \bar { s } _ { q } ^ { \mathrm { b a s e } } = \bar { s } _ { q } ^ { \mathcal { B } } , \quad \bar { s } _ { q } ^ { \mathrm { s h o r t } } = \bar { s } _ { q } ^ { \mathcal { S } } .\tag{6}
$$

MemoryGain. Our direct metric-level measure is MemoryGain, defined as the revisit advantage over the gap-matched baseline:

$$
\mathrm { M G } _ { q } = \bar { s } _ { q } ^ { \mathrm { r e v } } - \bar { s } _ { q } ^ { \mathrm { b a s e } } .\tag{7}
$$

MemoryGain asks whether consistency is selectively elevated at the commanded revisit relative to the rollout’s ordinary long-gap similarity. Because both terms come from the same rollout, globally shared factors such as style, sharpness, conservative rendering, and low motion tend to afect both scores and are partially canceled by the diference. Positive MemoryGain therefore indicates a calibrated revisit advantage beyond generic temporal self-similarity; it should not be interpreted as a causal estimate of an internal memory mechanism.

Short-range dynamic range. To characterize the discriminability of each metric on a given rollout, we define the short-range dynamic range:

$$
\mathrm { D R } _ { q } = \bar { s } _ { q } ^ { \mathrm { s h o r t } } - \bar { s } _ { q } ^ { \mathrm { b a s e } } .\tag{8}
$$

Dynamic range is not a memory score. It measures how much the metric separates local temporal consistency from the baseline. A small $\mathrm { D R } _ { q }$ indicates a low-discriminability regime, such as near-static generation, repetitive scenes, or metrics that miss the relevant changes.

Normalized Memory Ratio. To compare and aggregate revisit advantages across metrics with diferent numerical scales, we define Normalized Memory Ratio (NMR) as a dynamic-range-normalized score:

$$
\mathrm { N M R } _ { q } = \frac { \mathrm { M G } _ { q } } { \mathrm { D R } _ { q } + \epsilon } = \frac { \bar { s } _ { q } ^ { \mathrm { r e v } } - \bar { s } _ { q } ^ { \mathrm { b a s e } } } { \bar { s } _ { q } ^ { \mathrm { s h o r t } } - \bar { s } _ { q } ^ { \mathrm { b a s e } } + \epsilon } ,\tag{9}
$$

where ϵ is a small constant for numerical stability. NMR locates the revisit advantage relative to the metric’s short-to-baseline dynamic range: a value of zero means that revisit consistency does not exceed the long-gap baseline, while a value of one means that it reaches the corresponding short-range consistency level. This normalization yields a dimensionless score that can be aggregated across metrics. For Overall NMR, we firs average metrics within each family and then average the five family scores equally, preventing families with more metrics from receiving greater weight. However, NMR depends on both the revisit gain in the numerator and the dynamic range in the denominator. When DR<sub>q</sub> is small, NMR can become unstable or amplified by a weak normalization scale. We therefore use MG as the direct evidence within each metric and NMR for family-level and overall aggregation, interpreting NMR jointly with MG and DR. Appendix B discusses these cases in detail.

## 4.4 Dimensions of Revisit Consistency

Revisit consistency is not a single observable property: a model may preserve surface appearance while changing the identity of a place, retain recognizable objects while distorting their spatial arrangement, or produce two plausible views that cannot belong to the same persistent world. We therefore evaluate five complementary dimensions of revisit consistency. For each dimension, we identify the observable failure it is intended to expose, construct a pairwise test, and define the corresponding metric readouts. These dimensions characterize diferent aspects of the generated revisit; they should not be interpreted as distinct internal types of memory. All readouts are converted to oriented scores using Eq. 5 and evaluated with the same MemoryGain and NMR protocol. Appendix A.1 gives the backbone configurations, preprocessing, and per-metric formulas.

Appearance Fidelity. This dimension measures whether a revisit preserves the low-level visual appearance of the earlier observation. It exposes surface-level failures such as repainting, color or texture drift, blur, and changes in fine image structure. We directly compare the two frames using PSNR, SSIM, and LPIPS (Zhang et al., 2018), which provide complementary pixel-, structure-, and perceptual-level readouts. A high revisitrelative score indicates that the model retains the earlier rendering beyond its generic temporal similarity; because these measures are sensitive to viewpoint and illumination, it does not by itself establish that the same place or world state has been recovered.

Scene Identity Preservation. This dimension asks whether the returned observation remains recognizable as the same place under moderate appearance and viewpoint changes. A model fails when it generates a plausible scene that is semantically similar yet no longer corresponds to the previously visited location. We encode each frame independently with a general visual representation, DINOv2 (Oquab et al., 2023), and with two place-recognition descriptors, BoQ (Ali-Bey et al., 2024) and MutualVPR (Gu et al., 2025), then compute cosine similarity between the resulting global descriptors. The three similarities measure semantic scene resemblance and place recognizability from complementary representations. High revisit-relative similarit supports preserved location identity, but does not require exact object-level or geometric agreement.

Object Identity. This dimension tests whether salient objects survive a revisit with consistent appearance and semantics. Typical failures include object disappearance, replacement, or identity drift even when the surrounding scene remains plausible. We detect and segment object anchors in the earlier frame using GroundingDINO (Liu et al., 2023a) and SAM2 (Ravi et al., 2024), then re-detect each anchor concept independently in the paired frame rather than tracking it through the intervening video. Accepted candidates are compared through masked DINOv2 appearance features and CLIP (Radford et al., 2021)-based semantic compatibility. We report appearance persistence and semantic persistence, assigning zero to unmatched anchors so that both identity quality and object survival afect the scores. High revisit-relative persistence indicates that the same object evidence is recovered, but does not ensure that the objects retain their original spatial arrangement.

Local Geometric Correspondence. This dimension measures whether local structures in the two observations remain geometrically compatible. It targets failures in which a revisit preserves the scene category or object inventory but warps, rearranges, or replaces the underlying spatial structure. We extract SuperPoint keypoints (DeTone et al., 2018), match them with LightGlue (Lindenberger et al., 2023), and test the matches with robust two-view geometry. The feature-correspondence ratio measures how much local evidence can be matched, while the RANSAC inlier ratio measures how much of that evidence supports a common epipolar geometry. High revisit-relative values provide local geometric support for a persistent scene, although they do not establish complete global 3D consistency.

Persistent State Reasoning. This dimension asks whether paired observations depict the same local scene with state that remains consistent under camera motion and occlusion. It captures failures missed by lower-level measures, including scene replacement, contradictory layouts, and unexplained state changes. A vision-language model applies a fixed, pair-type-blind rubric to every revisit, baseline, and short-range pair and returns a scalar score. Higher revisit-relative scores indicate stronger persistent-state consistency than same-video controls, but not direct pixel or geometry alignment.

## 5 Experiments

## 5.1 Implementation Details

All evaluated video world models are run with chunked autoregressive inference: each rollout is generated one chunk at a time, and previous frames or model-specific memory states are fed back as context for the next chunk. Inference is performed on an 8-GPU NVIDIA H20 server. Evaluation videos contain approximately 481 frames, with small variations due to model-specific padding; we remove static leading and trailing frames and denote the remaining efective length by T.

For revisit-pair mining, we follow Eq. 1 with $j \mathbin { - } i \geq \operatorname* { m a x } ( 0 . 2 T , 1 0 )$ and adaptive thresholds $\tau _ { \mathrm { p o s } } =$ $\mathrm { c l i p } ( 2 P _ { 9 5 } ( \Delta t ) , 0 . 1 , 1 . 0 )$ meters and $\tau _ { \mathrm { r o t } } = \mathrm { c l i p } ( 2 P _ { 9 5 } ( \Delta \theta ) , 1 . 0 , 5 . 0 )$ degrees. For each later frame, we keep the best earlier pose match. Baselines follow Eq. 2, using one gap-matched baseline per revisit pair with gap tolerance $\delta = \operatorname* { m a x } ( 1 0 , \operatorname { r o u n d } ( 0 . 3 \Delta ) )$ and exclusion radius $\eta = \operatorname* { m a x } ( 1 0 , \lfloor T / 2 0 \rfloor )$ . Short-range references follow Eq. 4 with $\Delta _ { \operatorname* { m i n } } = 2 , \Delta _ { \operatorname* { m a x } } = \operatorname* { m a x } ( 6 , \lfloor 0 . 0 2 T \rfloor )$ , and $N _ { \mathrm { s h o r t } } = \mathrm { m i n } ( 1 0 0 , \mathrm { m a x } ( 2 0 , N _ { \mathrm { r e v } } ) )$ ). For NMR, we compute per-video MemoryGain and dynamic range, discard video and metric cases with $\mathrm { D R } \leq 1 0 ^ { - 8 }$ , and take the ratio after averaging valid numerators and denominators.

For metric extraction, we use DINOv2-ViT-B/14 for generic scene features. SuperPoint uses at most 1024 keypoints with detection threshold $5 \times 1 0 ^ { - 4 } .$ , and LightGlue uses filtering threshold 0.1. For vision-language judgment, we use Gemini-3.1-Pro-Preview with a fixed rubric; the full prompt appears in Appendix A.2.

## 5.2 Benchmark Results

Evaluation protocol. We evaluate seven action-conditioned video world model variants: LingBot-World-Fast and LingBot-World-2-Fast (Robbyant Team et al., 2026; Gao et al., 2026), Matrix-Game 3.0 (Wang et al., 2026b), HY-WorldPlay (Sun et al., 2025), DreamX-World-Memo (DreamX Team et al., 2026), SANA-WM (Zhu et al., 2026), and Lyra-2 (Shen et al., 2026). Each model is evaluated on all three trajectory templates, with up to 100 generated videos per model and trajectory and three runs with distinct random seeds. We additionally report WonderWorld (Yu et al., 2024), an explicit 3D reconstruction and rendering system, as a near-perfect-memory reference rather than a directly comparable video model.

Table 2 separates two questions. The top panel reports metric-level MemoryGain (MG), which tests whether revisits are more consistent than gap-matched controls in the metric’s native units. The bottom panel reports NMR, which normalizes that advantage by the same video’s short-to-baseline dynamic range. Overall NMR averages metrics within each family and then weights the five families equally. We therefore use MG as direct evidence of a revisit advantage and NMR for comparisons across metrics and families.

Overall comparison. DreamX-World-Memo achieves the highest Overall NMR among the video models (0.706), followed by HY-WorldPlay (0.485), Matrix-Game 3.0 (0.403), and Lyra-2 (0.310). Its lead is supported by both components of the evaluation: DreamX-World-Memo has the largest MG on 8 of the 11 displayed metrics and the largest NMR on 9 of 11. The separate WonderWorld reference reaches 1.510 Overall NMR, consistent with its explicit 3D representation and deterministic rendering. NMR is not bounded by one; values above one indicate that the revisit advantage exceeds the corresponding short-to-baseline range and should be interpreted jointly with MG.

Table 2 Template-balanced MemoryGain (MG; top) and Normalized Memory Ratio (NMR; bottom), each averaged equally over Out-and-back, Translation–rotation, and Closed-loop. Memory types: RC = raw context, FoV-R = field-of-view retrieval, G-R = geometric retrieval, and SSM = state-space recurrence. WonderWorld (gray) is a separate 3D rendering reference. For video models, bold and underlined values mark the best and second-best results. Overall NMR first averages metrics within each family and then the five families; MG is not aggregated because its units are metric-specific.
<table><tr><td></td><td colspan="10">MemoryGain (MG; higher is better)</td><td colspan="3"></td></tr><tr><td></td><td></td><td></td><td colspan="3">Appearance Fidelity</td><td colspan="3">Scene Identity</td><td colspan="2">Object Identity</td><td colspan="2">Local Geometry</td><td>Persistent State</td></tr><tr><td>Method</td><td></td><td>Params Mem. Type</td><td>PSNR</td><td>SSIM</td><td>LPIPS</td><td>DINO-sim</td><td>BoQ-sim</td><td>MVPR-sim</td><td>App. Persist</td><td>Sem. Persist</td><td>Feat. Corr.</td><td>Geo. Inlier</td><td>State Cons.</td></tr><tr><td>WonderWorld</td><td></td><td>Explicit 3D</td><td>12.828</td><td>0.398</td><td>0.548</td><td>0.369</td><td>0.633</td><td>0.327</td><td>0.254</td><td>0.067</td><td>0.535</td><td>0.147</td><td>2.676</td></tr><tr><td>LingBot-World-Fast</td><td>14B</td><td>RC</td><td>0.283</td><td>0.018</td><td>0.007</td><td>0.029</td><td>0.041</td><td>0.027</td><td>0.022</td><td>0.012</td><td>0.040</td><td>0.008</td><td>0.186</td></tr><tr><td>LingBot-World-2-Fast</td><td>14B</td><td>RC</td><td>0.271</td><td>0.009</td><td>0.011</td><td>0.083</td><td>0.013</td><td>0.057</td><td>0.031</td><td>0.014</td><td>0.015</td><td>0.023</td><td>0.262</td></tr><tr><td>Matrix-Game 3.0</td><td>5B</td><td>FoV-R</td><td>1.039</td><td>0.024</td><td>0.060</td><td>0.145</td><td>0.176</td><td>0.104</td><td>0.098</td><td>0.045</td><td>0.145</td><td>0.037</td><td>0.476</td></tr><tr><td>HY-WorldPlay</td><td>8B</td><td>FoV-R</td><td>2.357</td><td>0.055</td><td>0.155</td><td>0.146</td><td>0.208</td><td>0.069</td><td>0.087</td><td>0.029</td><td>0.180</td><td>0.018</td><td>0.510</td></tr><tr><td>SANA-WM</td><td>2.6B</td><td>SG </td><td>0.393</td><td>0.008</td><td>0.009</td><td>-0.004</td><td>0.003</td><td>-0.019</td><td>-0.001</td><td>-0.003</td><td>-0.001</td><td>0.009</td><td>-0.008</td></tr><tr><td>Lyra-2</td><td>14B</td><td></td><td>1.318</td><td>0.041</td><td>0.090</td><td>0.129</td><td>0.134</td><td>0.086</td><td>0.058</td><td>0.022</td><td>0.096</td><td>0.031</td><td>0.422</td></tr><tr><td>DreamX-World-Memo</td><td>5B</td><td>FoV-R</td><td>4.647</td><td>0.120</td><td>0.247</td><td>0.185</td><td>0.262</td><td>0.103</td><td>0.095</td><td>0.027</td><td>0.184</td><td>0.046</td><td>0.912</td></tr></table>

<table><tr><td></td><td colspan="10">Normalized Memory Ratio (NMR; higher is better)</td><td colspan="3"></td></tr><tr><td></td><td></td><td></td><td colspan="3">Appearance Fidelity</td><td colspan="3">Scene Identity</td><td colspan="2">Object Identity</td><td colspan="2">Local Geometry</td><td>Persistent State</td><td></td></tr><tr><td>Method</td><td></td><td>Params Mem. Type</td><td>PSNR</td><td>SSIM</td><td>LPIPS</td><td></td><td>DINO-sim BoQ-sim</td><td>MVPR-sim</td><td>App. Persist</td><td>Sem. Persist</td><td>Feat. Corr.</td><td>Geo. Inlier</td><td>State Cons.</td><td>Overall</td></tr><tr><td>WonderWorld</td><td></td><td>Explicit 3D</td><td>3.253</td><td>3.076</td><td>1.868</td><td>1.110</td><td>1.249</td><td>1.096</td><td>1.221</td><td>1.319</td><td>1.166</td><td>1.132</td><td>1.247</td><td>1.510</td></tr><tr><td>LingBot-World-Fast</td><td>14B</td><td>RC</td><td>0.047</td><td>0.083</td><td>0.030</td><td>0.073</td><td>0.080</td><td>0.103</td><td>0.126</td><td>0.245</td><td>0.098</td><td>0.043</td><td>0.103</td><td>0.100</td></tr><tr><td>LingBot-World-2-Fast</td><td>14B</td><td>RC</td><td>0.198</td><td>0.144</td><td>0.045</td><td>0.097</td><td>0.040</td><td>0.115</td><td>0.112</td><td>0.311</td><td>0.040</td><td>0.092</td><td>0.216</td><td>0.141</td></tr><tr><td>Matrix-Game 3.0</td><td>5B</td><td>FoV-R</td><td>0.257</td><td>0.192</td><td>0.251</td><td>0.449</td><td>0.400</td><td>0.470</td><td>0.572</td><td>0.896</td><td>0.385</td><td>0.226</td><td>0.300</td><td>0.403</td></tr><tr><td>HY-WorldPlay</td><td>8B</td><td>FoV-R</td><td>0.446</td><td>0.308</td><td>0.488</td><td>0.734</td><td>0.589</td><td>0.626</td><td>0.602</td><td>0.752</td><td>0.488</td><td>0.243</td><td>0.318</td><td>0.485</td></tr><tr><td>SANA-WM</td><td>2.6B 14B</td><td>SSM</td><td>0.094</td><td>0.071</td><td>0.051</td><td>0.002</td><td>0.012</td><td>-0.049</td><td>0.016</td><td>-0.007</td><td>0.002</td><td>0.020</td><td>0.006</td><td>0.016</td></tr><tr><td>Lyra-2</td><td></td><td>G-R</td><td>0.284</td><td>0.286</td><td>0.306</td><td>0.457</td><td>0.330</td><td>0.435</td><td>0.337</td><td>0.442</td><td>0.236</td><td>0.180</td><td>0.255</td><td>0.310</td></tr><tr><td>DreamX-World-Memo 5B</td><td></td><td>FoV-R</td><td>1.034</td><td>1.093</td><td>0.981</td><td>0.700</td><td>0.692</td><td>0.662</td><td>0.641</td><td>0.782</td><td>0.533</td><td>0.355</td><td>0.654</td><td>0.706</td></tr></table>

Dimension-specific profiles. The aggregate ranking conceals distinct strengths. DreamX-World-Memo leads all three Appearance Fidelity metrics, two of three Scene Identity metrics, both Local Geometry metrics, and Persistent State. HY-WorldPlay is strongest on DINO scene similarity and ranks second overall, while Matrix-Game 3.0 has the highest semantic object-persistence score. LingBot-World-2-Fast improves its Overall NMR from 0.100 to 0.141 relative to LingBot-World-Fast, but the gains are uneven across families; SANA-WM remains near zero overall. Thus, no single appearance, semantic, or geometric metric adequately represents revisit consistency.

The four retrieval-based systems occupy the top four Overall NMR positions. Because model scale, training data, generation procedure, and inference also difer, this ranking is associative and does not isolate the efect of retrieval.

Trajectory-wise analysis. The three templates probe diferent forms of return. Out-and-back approximately reverses recent motion; Translation–rotation adds heading and viewpoint change; and Closed-loop requires a return after a longer traversal. Figure 4 should therefore be read as three within-template profiles rather than as a direct comparison of radar area.

DreamX-World-Memo has the broadest profile on Out-and-back and Translation–rotation, but contracts sharply on Closed-loop. HY-WorldPlay is less dominant on the first two trajectories yet retains a comparatively broader closed-loop profile, especially on scene and object cues. Matrix-Game 3.0 remains strong on objectrelated axes but is less uniform across appearance and geometry, while Lyra-2 is intermediate across most dimensions. The two LingBot variants and SANA-WM show limited revisit-selective advantage once the return includes substantial viewpoint change or accumulated rollout error.

The cross-template variation is not a uniform loss of image quality. Some models preserve Appearance Fidelity while Scene Identity or Local Geometry deteriorates; others retain object semantics without recovering a compatible layout. The trajectory suite therefore reveals whether a model supports near-view reinstatement, viewpoint-robust place recognition, or longer-horizon loop closure, rather than reducing all three behaviors to one return score.

Human validation. Table 3 summarizes the human-alignment and motion-dependence analyses. We evaluate whether the automatic scores track perceived revisit consistency and how much of that agreement can be attributed to motion or camera execution. We sample 10 rollouts from every model–trajectory combination, yielding $7 \times 3 \times 1 0 = 2 1 0$ videos. Six annotators independently rate revisit consistency, generated motion, and camera control from 1 (Bad) to 3 (Good), and we average their ratings for each video.

![](images/d0a5a57e15d99c9c1ccfd146473f7e1850618b99e44e714f4fa2f02f923a662f.jpg)  
Figure 4 Trajectory-wise metric-level NMR profiles for the video world models. Larger values indicate a stronger normalized revisit advantage. Radial limits are adapted per panel for legibility; compare model shapes and rankings within a panel rather than polygon area across panels.

Table 3 Human validation on 210 sampled rollouts. (a) Video-level correlations, including the absolute Raw similarity reference. ρ is Spearman rank correlation; $H _ { \mathrm { { c o n s } } }$ and $H _ { \mathrm { m o t i o n } }$ are human consistency and motion ratings. Partial ρ controls for camera control, motion, and trajectory. Action-valid requires mean camera-control and motion ratings both ≥ 1.5. Raw, MG, and NMR use family-balanced Fisher-z aggregation, whereas Overall NMR is the per-video family-first score. The 95% intervals use trajectory–scene cluster bootstrap. (b) Correlation between each family’s NMR and human consistency.
<table><tr><td colspan="7">(a) Video-level alignment and motion dependence</td></tr><tr><td>Score</td><td> $\rho ( H _ { \mathrm { c o n s } } )$ </td><td>95% Cl</td><td>Partial  $\rho$ </td><td>95% Cl</td><td>Action-valid  $\rho$ </td><td> $\rho ( H _ { \mathrm { m o t i o n } } )$ </td></tr><tr><td>Raw</td><td>0.594</td><td>[0.54,0.64]</td><td>0.221</td><td>[0.12,0.30]</td><td>0.466</td><td>-0.336</td></tr><tr><td>MG</td><td>0.398</td><td>[0.33, 0.46]</td><td>0.096</td><td>[0.00, 0.19]</td><td>0.280</td><td>-0.053</td></tr><tr><td>NMR</td><td>0.480</td><td>[0.41,0.54]</td><td>0.149</td><td>[0.05,0.25]</td><td>0.369</td><td>-0.119</td></tr><tr><td>Overall NMR</td><td>0.547</td><td>[0.45, 0.63]</td><td>0.106</td><td>[0.02, 0.26]</td><td>0.409</td><td>-0.151</td></tr></table>

<table><tr><td colspan="6">(b) Per-family NMR alignment with human consistency</td></tr><tr><td>Family</td><td>Appearance</td><td>Scene</td><td>Object</td><td>Geometry</td><td>State</td></tr><tr><td> $\rho ( H _ { \mathrm { c o n s } } )$ </td><td>0.499</td><td>0.618</td><td>0.437</td><td>0.435</td><td>0.386</td></tr></table>

Overall NMR correlates with human consistency at $\rho = 0 . 5 4 7$ (95% CI [0.45, 0.63]). The metric-level relative scores are also positively correlated with consistency: $\rho = 0 . 3 9 8$ for MG and 0.480 for NMR. These results show that revisit-selective consistency remains perceptually meaningful after calibration against same-video controls. This alignment is not confined to appearance: the per-family NMR correlations in Table 3(b) are positive for all five families, ranging from 0.386 for Persistent State to 0.618 for Scene Identity.

Table 3 also exposes the trade-of behind this calibration. Raw similarity has a higher unadjusted consistency correlation (0.594), but it is substantially coupled to lower motion $( \rho = - 0 . 3 3 6 )$ . The corresponding motion correlations are smaller for MG (−0.053), NMR (−0.119), and Overall NMR (−0.151). Relative scoring therefore does not optimize agreement with human ratings in isolation; it retains positive agreement while reducing the slow-motion shortcut that afects absolute similarity.

The complementary objective-motion and controlled diagnostics are reported in Tables 4 and 5.

Table 4 Family-level correlations with generated motion. Motion is the mean adjacent-frame optical-flow magnitude computed with RAFT. Correlations are first computed within each model; family cells average their constituent metrics, and Overall weights the five families equally.
<table><tr><td colspan="4">Within-model Spearman ρ with motion</td></tr><tr><td>Family</td><td>Raw rev. Temp. base</td><td>MG</td><td>NMR</td></tr><tr><td>Appearance Fidelity Scene Identity</td><td>-0.299 -0.304 -0.345</td><td>0.008 0.185</td><td>0.089 0.063</td></tr><tr><td>Local Geometry</td><td>-0.274 -0.216 -0.337</td><td>0.156</td><td>0.129</td></tr><tr><td>Object Identity</td><td>-0.061 -0.088</td><td>0.090</td><td>0.023</td></tr><tr><td>Persistent State</td><td>-0.186 -0.283</td><td>0.116</td><td>0.056</td></tr><tr><td>Overall</td><td>-0.207 -0.271</td><td>0.111</td><td>0.072</td></tr></table>

Table 5 Diagnostic analysis of confounds in raw revisit similarity using DreamX-World-Memo generations. Videos are grouped by motion speed, and visual-quality perturbations are applied to the same High-motion split. Raw Rev. scores vary with speed and image quality, while Gain and NMR measure revisit-selective consistency after calibration against the same rollout. Bold denotes the best score within each diagnostic block.
<table><tr><td></td><td colspan="4">SSIM ↑</td><td colspan="4">LPIPS↓</td><td colspan="4">DINO-sim ↑</td><td colspan="4">Geo. Inlier ↑</td></tr><tr><td>Setting</td><td>Rev.</td><td>Base.</td><td>Gain ↑</td><td>NMR</td><td>Rev.</td><td>Base.</td><td>Gain ↑</td><td>NMR</td><td>Rev. Base.</td><td></td><td>Gain ↑</td><td>NMR</td><td>Rev.</td><td>Base.</td><td>Gain ↑</td><td>NMR</td></tr><tr><td></td><td colspan="10">Motion speed split</td><td colspan="7"></td></tr><tr><td>Low-motion</td><td>0.507</td><td>0.445</td><td>+0.062</td><td>0.39</td><td>0.318</td><td>0.444</td><td>+0.126</td><td>0.47</td><td>0.916</td><td>0.861</td><td>+0.055</td><td>0.48</td><td>0.829</td><td>0.815</td><td></td><td>+0.014</td><td>0.12</td></tr><tr><td>Mid-motion</td><td>0.468</td><td>0.404</td><td>+0.064</td><td>0.43</td><td>0.437</td><td>0.509</td><td>+0.072</td><td>0.39</td><td>0.872</td><td>0.803</td><td>+0.069</td><td>0.38</td><td>0.770</td><td>0.746</td><td></td><td>+0.025</td><td>0.17</td></tr><tr><td>High-motion</td><td>0.452</td><td>0.408</td><td>+0.044</td><td>0.33</td><td>0.458</td><td>0.549</td><td>+0.091</td><td>0.31</td><td>0.691</td><td>0.620</td><td>+0.071</td><td>0.42</td><td></td><td>0.722</td><td>0.693</td><td>+0.029</td><td>0.16</td></tr><tr><td></td><td></td><td colspan="10"></td><td colspan="3"></td><td colspan="3"></td></tr><tr><td>High-blur</td><td>0.550</td><td>0.497</td><td>+0.053</td><td>0.37</td><td>0.430</td><td>0.526</td><td>+0.096</td><td>0.33</td><td>0.685</td><td>0.612</td><td>+0.073</td><td></td><td>0.22</td><td>0.705</td><td>0.673</td><td>+0.032</td><td>0.16</td></tr><tr><td>High-downup</td><td>0.545</td><td>0.492</td><td>+0.053</td><td>0.36</td><td>0.429</td><td>0.523</td><td>+0.094</td><td>0.33</td><td>0.687</td><td>0.614</td><td>+0.073</td><td>0.22</td><td>0.698</td><td>0.665</td><td></td><td>+0.032</td><td>0.17</td></tr><tr><td>High-comp.</td><td>0.451</td><td>0.406</td><td>+0.045</td><td>0.33</td><td>0.463</td><td>0.552</td><td>+0.089</td><td>0.31</td><td>0.683</td><td>0.612</td><td>+0.071</td><td>0.22</td><td>0.682</td><td>0.656</td><td></td><td>+0.026</td><td>0.13</td></tr></table>

The association remains positive under two controls. After rank-residualizing camera control, motion, and trajectory, NMR has $\rho = 0 . 1 4 9$ (95% CI [0.05, 0.25]), and Overall NMR has $\rho = 0 . 1 0 6$ (95% CI [0.02, 0.26]). On the action-valid subset, which excludes videos with the clearest motion or camera-control failures, the correlations rise to 0.369 and 0.409. Together, these analyses support perceptual alignment of the observable score without treating motion, action execution, or an internal memory mechanism as interchangeable with revisit consistency. Appendix C reports inter-rater reliability, per-family results, and clustering sensitivity.

Objective motion diagnostic. Human ratings capture perceived motion, but we also test motion dependence with mean adjacent-frame optical-flow magnitude from RAFT (Teed and Deng, 2020). To avoid conflating motion with model identity, we compute Spearman correlations within each model and aggregate them by metric family. Raw denotes the absolute base-metric value on revisit pairs, and Temp. base denotes ordinary same-video similarity at a matched temporal gap. Table 4 reports the family-level correlations.

Raw revisit similarity and the temporal baseline are negatively correlated with motion in every family: videos with less generated motion preserve higher generic similarity. After family-first aggregation, the correlation is −0.207 for Raw and −0.271 for the baseline, compared with 0.111 for MG and 0.072 for NMR. Relative calibration therefore removes most of the systematic slow-motion advantage visible in absolute similarity. The remaining positive correlations in several families also show the boundary of this result: MG and NMR are less motion-sensitive, not motion-invariant, and nearly static or action-invalid outputs should still be inspected separately. Appendix D.1 gives all 11 metric-level correlations and confidence indicators.

Controlled diagnostic of relative scoring. We complement the across-video correlations with two controlled comparisons on DreamX-World-Memo generations. The first groups natural rollouts by realized motion speed; the second applies blur, downsample–upsample, and compression to the same high-motion

subset. Table 5 reports both comparisons.

Across the natural motion split in Table 5, raw SSIM decreases from 0.507 to 0.452, DINO similarity from 0.916 to 0.691, and geometric inlier ratio from 0.829 to 0.722 as motion increases. MG remains positive for every displayed metric and speed group, but is not required to vary monotonically because its matched baseline changes with the rollout. The relative scores therefore test revisit-selective recovery rather than treating the slowest generation as the strongest one.

The quality perturbations provide a same-video comparison: blur, downsample–upsample, and compression are applied to the same high-motion subset. Raw SSIM ranges from 0.451 to 0.550, while SSIM MG remains within 0.045–0.053; LPIPS MG remains within 0.089–0.096. Other metrics show smaller raw shifts, and geometry retains some residual variation. Together with the human and optical-flow analyses, these controlled results support attenuation, rather than elimination, of shortcuts caused by low motion and globally shared rendering quality.

## 6 Conclusion

R2M-Bench measures whether an interactive video world model is selectively more consistent at commanded revisits than at gap-matched, non-target moments in the same rollout. MG reports direct metric-level revisit advantage, while NMR normalizes this advantage for family-level and overall aggregation across five dimensions. Video-level human judgments align with the benchmark, and independent optical-flow diagnostics show substantially reduced motion dependence after relative calibration. Results reveal dimension- and trajectory-specific gaps, with retrieval-based systems leading among the evaluated video models.

## 7 Limitations

R2M-Bench evaluates observable revisit-selective consistency rather than identifying an internal memory mechanism; retrieval-related findings are associative, not causal. Revisit mining relies on commanded poses and model-specific action interfaces, so execution errors can remain entangled with generated-world persistence even though the human and motion analyses control for important manifestations of this confound. Automatic metrics also inherit backbone, viewpoint, and prompt biases. Finally, the current navigation-only scope excludes object interaction and deliberately evolving state, motivating realized-trajectory evaluation and broader interactive tasks.

## References

Decart AI. OASIS: A universe in a transformer. Technical Report, 2024. URL https://www.decart.ai/oasis.

Amar Ali-Bey, Brahim Chaib-Draa, and Philippe Giguere. Boq: A place is worth a bag of learnable queries. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 17794–17803, 2024.

Eloi Alonso, Adam Jelley, Vincent Micheli, Anssi Kanervisto, Amos Storkey, Tim Pearce, and François Fleuret. Difusion for world modeling: Visual details matter in atari. Advances in Neural Information Processing Systems, 37:58757–58791, 2024.

Hritik Bansal, Zongyu Lin, Tianyi Xie, Zeshun Zong, Michal Yarom, Yonatan Bitton, Chenfanfu Jiang, Yizhou Sun, Kai-Wei Chang, and Aditya Grover. VideoPhy: Evaluating physical commonsense for video generation, 2024.

Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, Varun Jampani, and Robin Rombach. Stable video difusion: Scaling latent video difusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023.

Jake Bruce, Michael D Dennis, Ashley Edwards, Jack Parker-Holder, Yuge Shi, Edward Hughes, Matthew Lai, Aditi Mavalankar, Richie Steigerwald, Chris Apps, et al. Genie: Generative interactive environments. In Forty-first International Conference on Machine Learning, 2024.

Rui Chen, Lei Sun, Jing Tang, Geng Li, and Xiangxiang Chu. FingER: Content aware fine-grained evaluation with reasoning for AI-generated videos. In Proceedings of the 33rd ACM International Conference on Multimedia, MM

’25, pages 3517–3526. Association for Computing Machinery, October 2025a. doi: 10.1145/3746027.3755102. URL https://doi.org/10.1145/3746027.3755102.

Taiye Chen, Zihan Ding, Anjian Li, Christina Zhang, Zeqi Xiao, Yisen Wang, and Chi Jin. Recurrent autoregressive difusion: Global memory meets local attention, 2025b.

Daniel DeTone, Tomasz Malisiewicz, and Andrew Rabinovich. SuperPoint: Self-supervised interest point detection and description. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops, pages 224–236, 2018.

Weijia Dou, Hui Li, Jiahao Cui, Lei Zhou, Jingdong Wang, and Siyu Zhu. SlotMemory: Object-centric KV memory for streaming long-video generation, 2026.

DreamX Team, Yancheng Bai, Rui Chen, Xiangxiang Chu, Rujing Dang, Hao Dou, Bingjie Gao, Qiwen Gu, Siyu Hong, Jiachen Lei, Geng Li, Jifan Li, Ruimin Lin, Qingfeng Shi, Bingze Song, Lei Sun, Jing Tang, Ruitian Tian, Jun Wang, Jiahong Wu, Pengfei Zhang, Shen Zhang, and Jiashu Zhu. DreamX-World 1.0: A general-purpose interactive world model, 2026.

Haoyi Duan, Hong-Xing Yu, Sirui Chen, Li Fei-Fei, and Jiajun Wu. WorldScore: A unified evaluation benchmark for world generation, 2025.

Xiaokun Feng, Haiming Yu, Meiqi Wu, Shiyu Hu, Jintao Chen, Chen Zhu, Jiahong Wu, Xiangxiang Chu, and Kaiqi Huang. NarrLV: Towards a comprehensive narrative-centric evaluation for long video generation. In The Fourteenth International Conference on Learning Representations, 2026. URL https://openreview.net/forum?id=Qh3CQBTB1g.

Zelin Gao, Qiuyu Wang, Jiapeng Zhu, Jingye Chen, Zichen Liu, Qingyan Bai, Jiahao Wang, Yufeng Yuan, Hanlin Wang, Yichong Lu, Ka Leong Cheng, Haojie Zhang, Jian Gao, Tianrui Feng, Yuzheng Liu, Yao Yao, Yinghao Xu, Xing Zhu, Yujun Shen, and Hao Ouyang. Infinite worlds with versatile interactions, 2026.

Qiwen Gu, Xufei Wang, Junqiao Zhao, Siyue Tao, Tiantian Feng, Ziqiao Wang, and Guang Chen. Mutualvpr: A mutual learning framework for resolving supervision inconsistencies via adaptive clustering. In Advances in Neural Information Processing Systems, pages 2899–2922, 2025.

David Ha and Jürgen Schmidhuber. World models. arXiv preprint arXiv:1803.10122, 2018.

Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J. Fleet. Video difusion models. In Advances in Neural Information Processing Systems, 2022.

Yicong Hong, Yiqun Mei, Chongjian Ge, Yiran Xu, Yang Zhou, Sai Bi, Yannick Hold-Geofroy, Mike Roberts, Matthew Fisher, Eli Shechtman, Kalyan Sunkavalli, Feng Liu, Zhengqi Li, and Hao Tan. RELIC: Interactive video world model with long-horizon memory, 2025.

Anthony Hu, Lloyd Russell, Hudson Yeo, Zak Murez, George Fedoseev, Alex Kendall, Jamie Shotton, and Gianluca Corrado. GAIA-1: A generative world model for autonomous driving. arXiv preprint arXiv:2309.17080, 2023.

Qixin Hu, Shuai Yang, Wei Huang, Song Han, and Yukang Chen. LongLive-RAG: A general retrieval-augmented framework for long video generation, 2026.

Xun Huang, Zhengqi Li, Guande He, Mingyuan Zhou, and Eli Shechtman. Self forcing: Bridging the train-test gap in autoregressive video difusion, 2025.

Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, Yaohui Wang, Xinyuan Chen, Limin Wang, Dahua Lin, Yu Qiao, and Ziwei Liu. VBench: Comprehensive benchmark suite for video generative models, 2023.

Ziqi Huang, Fan Zhang, Xiaojie Xu, Yinan He, Jiashuo Yu, Ziyue Dong, Qianli Ma, Nattapol Chanpaisit, Chenyang Si, Yuming Jiang, Yaohui Wang, Xinyuan Chen, Ying-Cong Chen, Limin Wang, Dahua Lin, Yu Qiao, and Ziwei Liu. VBench++: Comprehensive and versatile benchmark suite for video generative models, 2024.

Youngrae Kim, Qixin Hu, C.-C. Jay Kuo, and Peter A. Beerel. MemRoPE: Training-free infinite video generation via evolving memory tokens, 2026.

Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024.

Yann LeCun. A path towards autonomous machine intelligence. OpenReview, 2022. URL https://openreview.net/f orum?id=BZ5a1r-kVsf.

Dacheng Li, Yunhao Fang, Yukang Chen, Shuo Yang, Shiyi Cao, Justin Wong, Michael Luo, Xiaolong Wang, Hongxu Yin, Joseph E. Gonzalez, Ion Stoica, Song Han, and Yao Lu. WorldModelBench: Judging video generation models as world models, 2025a.

Kunyang Li, Mubarak Shah, and Yuzhang Shang. Attend locally, remember linearly: Linear attention as cross-frame memory for autoregressive video difusion, 2026.

Runjia Li, Philip Torr, Andrea Vedaldi, and Tomas Jakab. VMem: Consistent interactive video scene generation with surfel-indexed view memory, 2025b.

Philipp Lindenberger, Paul-Edouard Sarlin, and Marc Pollefeys. LightGlue: Local feature matching at light speed. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 17627–17638, 2023.

Xinran Ling, Chen Zhu, Meiqi Wu, Hangyu Li, Xiaokun Feng, Cundian Yang, Aiming Hao, Jiashu Zhu, Jiahong Wu, and Xiangxiang Chu. VMBench: A benchmark for perception-aligned video motion generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 13087–13098, October 2025. doi: 10.1109/ICCV51701.2025.01216. URL https://doi.org/10.1109/ICCV51701.2025.01216.

Kunhao Liu, Wenbo Hu, Jiale Xu, Ying Shan, and Shijian Lu. Rolling forcing: Autoregressive long video difusion in real time, 2025.

Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Qing Jiang, Chunyuan Li, Jianwei Yang, Hang Su, Jun Zhu, and Lei Zhang. Grounding DINO: Marrying DINO with grounded pre-training for open-set object detection, 2023a.

Yaofang Liu, Xiaodong Cun, Xuebo Liu, Xintao Wang, Yong Zhang, Haoxin Chen, Yang Liu, Tieyong Zeng, Raymond Chan, and Ying Shan. EvalCrafter: Benchmarking and evaluating large video generation models, 2023b.

Xiaofeng Mao, Zhen Li, Chuanhao Li, Xiaojie Xu, Kaining Ying, and Kaipeng Zhang. Yume1.5: A text-controlled interactive world generation model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7752–7761, 2026.

Fanqing Meng, Jiaqi Liao, Xinyu Tan, Wenqi Shao, Quanfeng Lu, Kaipeng Zhang, Yu Cheng, Dianqi Li, Yu Qiao, and Ping Luo. Towards world simulator: Crafting physical commonsense-based benchmark for video generation, 2024.

Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Mahmoud Assran, Nicolas Ballas, Wojciech Galuba, Russell Howes, Po-Yao Huang, Shang-Wen Li, Ishan Misra, Michael Rabbat, Vasu Sharma, Gabriel Synnaeve, Hu Xu, Hervé Jégou, Julien Mairal, Patrick Labatut, Armand Joulin, and Piotr Bojanowski. DINOv2: Learning robust visual features without supervision, 2023.

Zhan Peng, Jie Ma, Huiqiang Sun, Chong Gao, Zhijie Xue, Zhiyu Pan, Zhiguo Cao, Jun Liang, and Jing Li. Compression and retrieval: Implicit memory retrieval for video world models, 2026.

Julian Quevedo, Ansh Kumar Sharma, Yixiang Sun, Varad Suryavanshi, Percy Liang, and Sherry Yang. WorldGym: World model as an environment for policy evaluation, 2025.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In International Conference on Machine Learning, pages 8748–8763, 2021.

Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman Rädle, Chloe Rolland, Laura Gustafson, Eric Mintun, Junting Pan, Kalyan Vasudev Alwala, Nicolas Carion, Chao-Yuan Wu, Ross Girshick, Piotr Dollár, and Christoph Feichtenhofer. SAM 2: Segment anything in images and videos, 2024.

Robbyant Team, Zelin Gao, Qiuyu Wang, Yanhong Zeng, Jiapeng Zhu, Ka Leong Cheng, Yixuan Li, Hanlin Wang, Yinghao Xu, Shuailei Ma, Yihang Chen, Jie Liu, Yansong Cheng, Yao Yao, Jiayi Zhu, Yihao Meng, Kecheng Zheng, Qingyan Bai, Jingye Chen, Zehong Shen, Yue Yu, Xing Zhu, Yujun Shen, and Hao Ouyang. Advancing open-source world models, 2026.

Masaki Saito, Eiichi Matsumoto, and Shunta Saito. Temporal generative adversarial nets with singular value clipping. In Proceedings of the IEEE International Conference on Computer Vision, 2017.

Tianchang Shen, Sherwin Bahmani, Kai He, Sangeetha Grama Srinivasan, Tianshi Cao, Jiawei Ren, Ruilong Li, Zian Wang, Nicholas Sharp, Zan Gojcic, Sanja Fidler, Jiahui Huang, Huan Ling, Jun Gao, and Xuanchi Ren. Lyra 2.0: Explorable generative 3d worlds, 2026.

Uriel Singer, Adam Polyak, Thomas Hayes, Xi Yin, Jie An, Songyang Zhang, Qiyuan Hu, Harry Yang, Oron Ashual, Oran Gafni, Devi Parikh, Sonal Gupta, and Yaniv Taigman. Make-a-video: Text-to-video generation without text-video data, 2022.

Kaiyue Sun, Kaiyi Huang, Xian Liu, Yue Wu, Zihan Xu, Zhenguo Li, and Xihui Liu. T2V-CompBench: A comprehensive benchmark for compositional text-to-video generation, 2024.

Wenqiang Sun, Haiyu Zhang, Haoyuan Wang, Junta Wu, Zehan Wang, Zhenwei Wang, Yunhong Wang, Jun Zhang,

Tengfei Wang, and Chunchao Guo. WorldPlay: Towards long-term geometric consistency for real-time interactive world modeling, 2025.

Zachary Teed and Jia Deng. RAFT: Recurrent all-pairs field transforms for optical flow. In European Conference on Computer Vision, pages 402–419, 2020.

Sergey Tulyakov, Ming-Yu Liu, Xiaodong Yang, and Jan Kautz. MoCoGAN: Decomposing motion and content for video generation. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2018.

Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.

Weijie Wang, Haoyu Zhao, Yifan Yang, Feng Chen, Zeyu Zhang, Yefei He, Zicheng Duan, Donny Y. Chen, Yuqing Yang, and Bohan Zhuang. Latent spatial memory for video world models, 2026a.

Zile Wang, Zexiang Liu, Jiaxing Li, Kaichen Huang, Baixin Xu, Fei Kang, Mengyin An, Peiyu Wang, Biao Jiang, Yichen Wei, Yidan Xietian, Jiangbo Pei, Liang Hu, Boyi Jiang, Hua Xue, Zidong Wang, Haofeng Sun, Wei Li, Wanli Ouyang, Xianglong He, Yang Liu, Yangguang Li, and Yahui Zhou. Matrix-Game 3.0: Real-time and streaming interactive world model with long-horizon memory, 2026b.

Jialong Wu, Shaofeng Yin, Ningya Feng, Xu He, Dong Li, Jianye Hao, and Mingsheng Long. iVideoGPT: Interactive VideoGPTs are scalable world models. In Advances in Neural Information Processing Systems, volume 37, 2024. doi: 10.52202/079017-2173. URL https://proceedings.neurips.cc/paper\_files/paper/2024/hash/7dbb5bfab 324e3b86af9bd0df15498dd-Abstract-Conference.html.

Meiqi Wu, Zhixin Cai, Fufangchen Zhao, Xiaokun Feng, Rujing Dang, Bingze Song, Ruitian Tian, Jiashu Zhu, Jiachen Lei, Hao Dou, Jing Tang, Lei Sun, Jiahong Wu, Xiangxiang Chu, Zeming Liu, and Kaiqi Huang. Omni-WorldBench: Towards a comprehensive interaction-centric evaluation for world models, 2026a.

Ruiqi Wu, Xuanhua He, Meng Cheng, Tianyu Yang, Yong Zhang, Zhuoliang Kang, Xunliang Cai, Xiaoming Wei, Chunle Guo, Chongyi Li, and Ming-Ming Cheng. Infinite-World: Scaling interactive world models to 1000-frame horizons via pose-free hierarchical memory, 2026b.

Wilson Yan, Yunzhi Zhang, Pieter Abbeel, and Aravind Srinivas. VideoGPT: Video generation using VQ-VAE and transformers, 2021.

Shuai Yang, Wei Huang, Ruihang Chu, Yicheng Xiao, Yuyang Zhao, Xianbang Wang, Muyang Li, Enze Xie, Yingcong Chen, Yao Lu, et al. Longlive: Real-time interactive long video generation. arXiv preprint arXiv:2509.22622, 2025a.

Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. CogVideoX: Text-to-video difusion models with an expert transformer. In The Thirteenth International Conference on Learning Representations, 2025b. URL https://openreview.net/forum?i d=LQzN6TRFg9.

Yixuan Ye, Xuanyu Lu, Yuxin Jiang, Yuchao Gu, Rui Zhao, Qiwei Liang, Jiachun Pan, Fengda Zhang, Weijia Wu, and Alex Jinpeng Wang. MIND: Benchmarking memory consistency and action control in world models, 2026.

Kaining Ying, Hengrui Hu, Siyu Ren, Jiamu Li, Fengjiao Chen, Ziwen Wang, Xuezhi Cao, Xunliang Cai, and Henghu Ding. WBench: A comprehensive multi-turn benchmark for interactive video world model evaluation, 2026.

Hong-Xing Yu, Haoyi Duan, Charles Herrmann, William T. Freeman, and Jiajun Wu. WonderWorld: Interactive 3d scene generation from a single image, 2024.

Yifei Yu, Xiaoshan Wu, Xinting Hu, Tao Hu, Yangtian Sun, Xiaoyang Lyu, Bo Wang, Lin Ma, Yuewen Ma, Zhongrui Wang, and Xiaojuan Qi. VideoSSM: Autoregressive long video generation with hybrid state-space memory, 2025.

Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, and Oliver Wang. The unreasonable efectiveness of deep features as a perceptual metric. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 586–595, 2018.

Shengjun Zhang, Zhang Zhang, Simin Huang, Zhenyu Tang, Hanyang Wang, Chensheng Dai, Min Chen, Yifan Li, Yuxin Li, Yingjie Chen, Hao Liu, Chen Li, Jing Lyu, and Yueqi Duan. MBench: A comprehensive benchmark on memory capability for video world models, 2026.

Dian Zheng, Ziqi Huang, Hongbo Liu, Kai Zou, Yinan He, Fan Zhang, Lulu Gu, Yuanhan Zhang, Jingwen He, Wei-Shi Zheng, et al. Vbench-2.0: Advancing video generation benchmark suite for intrinsic faithfulness. arXiv preprint arXiv:2503.21755, 2025.

Zangwei Zheng, Xiangyu Peng, Tianji Yang, Chenhui Shen, Shenggui Li, Hongxin Liu, Yukun Zhou, Tianyi Li, and Yang You. Open-sora: Democratizing eficient video production for all. arXiv preprint arXiv:2412.20404, 2024.

Haoyi Zhu, Haozhe Liu, Yuyang Zhao, Tian Ye, Junsong Chen, Jincheng Yu, Tong He, Song Han, and Enze Xie.

SANA-WM: Eficient minute-scale world modeling with hybrid linear difusion transformer, 2026.

## A Implementation and Evaluation Details

This appendix first specifies the metric implementations and vision-language rubric, then explains the dynamicrange normalization used by NMR. It next provides the complete human and motion analyses underlying the compact validation tables in the main text, followed by controlled diagnostics and family-wise rank comparisons.

## A.1 Metric Implementation Details

This section gives implementation details for the metric families in Section 4. All metrics are computed on individual frame pairs and then averaged over the revisit, baseline, and short-range pair sets before computing MemoryGain and NMR. Unless otherwise specified, RGB frames are evaluated at their decoded video resolution.

Appearance Fidelity. For a frame pair $( f _ { i } , f _ { j } )$ , PSNR is computed with pixel range 255:

$$
\mathrm { P S N R } ( f _ { i } , f _ { j } ) = 1 0 \log _ { 1 0 } \frac { 2 5 5 ^ { 2 } } { \mathrm { M S E } ( f _ { i } , f _ { j } ) } .\tag{10}
$$

SSIM is computed on RGB images with data range 255, channel-wise image layout, and a $, 7 \times 7$ local window. LPIPS is computed with the AlexNet LPIPS network. Before LPIPS evaluation, each RGB image is converted from uint8 values in [0, 255] to a tensor in [−1, 1]. PSNR and SSIM are higher-is-better metrics, while LPIPS is lower-is-better and is direction-flipped when computing MemoryGain and NMR.

Scene Identity Preservation. For DINOv2, each image is resized to 518 pixels, center-cropped to $5 1 8 \times 5 1 8$ normalized with ImageNet statistics, and passed through a DINOv2 ViT-B/14 encoder. The resulting global feature is ℓ -normalized. The DINOv2 score is cosine similarity:

$$
q _ { \mathrm { D I N O } } ( f _ { i } , f _ { j } ) = \frac { \phi _ { \mathrm { D I N O } } ( f _ { i } ) ^ { \top } \phi _ { \mathrm { D I N O } } ( f _ { j } ) } { \| \phi _ { \mathrm { D I N O } } ( f _ { i } ) \| _ { 2 } \| \phi _ { \mathrm { D I N O } } ( f _ { j } ) \| _ { 2 } } .\tag{11}
$$

BoQ is evaluated with its DINOv2-backed VPR model. Images are resized to $3 2 2 \times 3 2 2$ , normalized with ImageNet statistics, encoded into BoQ descriptors, ℓ<sub>2</sub>-normalized, and compared by cosine similarity. MutualVPR follows the same structure: images are resized to $5 1 8 \times 5 1 8$ , normalized with ImageNet statistics, encoded by the MutualVPR model into a 512-dimensional descriptor, ℓ<sub>2</sub>-normalized, and compared by cosine similarity. All three Scene Identity Preservation scores are higher-is-better.

Object Identity. The object-centric metric is computed by pairwise re-detection rather than long-range tracking. For the first frame $f _ { i } ,$ we use GroundingDINO with a generic object prompt covering common scene entities, including furniture, buildings, vehicles, plants, signs, doors, windows, and other object categories. Detections are filtered with box threshold 0.25, text threshold 0.20, non-maximum suppression IoU 0.60, and a minimum SAM2 mask area ratio of 0.002. SAM2 is then used to refine each retained box into a mask. We rank detections by GroundingDINO confidence times mask area ratio and keep the top $K = 1 0$ first-frame anchors.

Let $A _ { i }$ be the retained anchor objects in $f _ { i } .$ . For each anchor $o \in { \mathcal { A } } _ { i }$ , we normalize its detected phrase into an object concept $c _ { o }$ and re-detect that concept in the paired frame $f _ { j }$ using the prompt $^ { 6 } c _ { o } \cdot { } .$ We keep at most five paired-frame candidates per concept and refine their boxes with SAM2. For each anchor and candidate, pixels outside the SAM2 mask are replaced by white before extracting crop features, so the object score is less afected by background changes. The appearance score between anchor o and candidate k is the rescaled cosine similarity of their masked DINOv2 crop features:

$$
a _ { o k } = \mathrm { c l i p } _ { 0 1 } \left( \frac { \phi _ { \mathrm { D I N O } } ( o ) ^ { \top } \phi _ { \mathrm { D I N O } } ( k ) } { 2 \| \phi _ { \mathrm { D I N O } } ( o ) \| _ { 2 } \| \phi _ { \mathrm { D I N O } } ( k ) \| _ { 2 } } + \frac { 1 } { 2 } \right) ,\tag{12}
$$

where $\mathrm { c l i p } _ { 0 1 } ( x ) = \mathrm { m i n } ( 1 , \mathrm { m a x } ( 0 , x ) )$ . The semantic score combines the GroundingDINO confidence $g _ { k }$ for the candidate concept with a CLIP (Radford et al., 2021) image-text score. Specifically, we compare the candidate crop with the anchor concept text and average the two semantic signals:

$$
u _ { o k } = \frac { 1 } { 2 } \cos \mathrm { l i p } _ { 0 1 } ( g _ { k } ) + \frac { 1 } { 2 } \cos \mathrm { l i p } _ { 0 1 } \left( \frac { \psi _ { \mathrm { i m g } } ( k ) ^ { \top } \psi _ { \mathrm { t e x t } } ( c _ { o } ) } { 2 \| \psi _ { \mathrm { i m g } } ( k ) \| _ { 2 } \| \psi _ { \mathrm { t e x t } } ( c _ { o } ) \| _ { 2 } } + \frac { 1 } { 2 } \right) .\tag{13}
$$

The implementation falls back to $u _ { o k } = \mathrm { c l i p } _ { 0 1 } ( g _ { k } )$ if no CLIP checkpoint is provided, but all reported object-centric results use CLIP. The candidate match score is

$$
r _ { o k } = \frac { w _ { \mathrm { a p p } } a _ { o k } + w _ { \mathrm { s e m } } u _ { o k } } { w _ { \mathrm { a p p } } + w _ { \mathrm { s e m } } } ,\tag{14}
$$

with default weights $w _ { \mathrm { a p p } } = w _ { \mathrm { s e m } } = 0 . 5$ . The best candidate is accepted as a match if max $r _ { o k } \ge 0 . 5 0 \mathrm { { ; } }$ otherwise the anchor is considered unmatched.

Let $\mathcal { M } _ { i j } \subseteq \mathcal { A } _ { i }$ be the matched anchors, and let $k ( o )$ denote the accepted candidate for anchor o. If no first-frame anchor is retained, all object-centric scores for the pair are set to 0. We compute two matched-only identity scores,

$$
\begin{array} { r } { q _ { \mathrm { o b j - a p p } } ( f _ { i } , f _ { j } ) = \left\{ \begin{array} { l l } { \frac { 1 } { | \mathcal { M } _ { i j } | } \sum _ { o \in \mathcal { M } _ { i j } } a _ { o , k ( o ) } , } & { | \mathcal { M } _ { i j } | > 0 , } \\ { 0 , } & { | \mathcal { M } _ { i j } | = 0 , } \end{array} \right. } \end{array}\tag{15}
$$

$$
\begin{array} { r } { q _ { \mathrm { o b j - s e m } } ( f _ { i } , f _ { j } ) = \left\{ \begin{array} { l l } { \frac { 1 } { | \mathcal { M } _ { i j } | } \sum _ { o \in \mathcal { M } _ { i j } } u _ { o , k ( o ) } , } & { | \mathcal { M } _ { i j } | > 0 , } \\ { 0 , } & { | \mathcal { M } _ { i j } | = 0 . } \end{array} \right. } \end{array}\tag{16}
$$

We also compute persistence scores over all first-frame anchors, assigning zero to unmatched anchors:

$$
{ q } _ { \mathrm { a p p - p e r s i s t } } ( f _ { i } , f _ { j } ) = \frac { 1 } { | A _ { i } | } \sum _ { o \in A _ { i } } \mathbf { 1 } [ o \in \mathcal { M } _ { i j } ] a _ { o , k ( o ) } ,\tag{17}
$$

$$
{ q } _ { \mathrm { s e m - p e r s i s t } } ( f _ { i } , f _ { j } ) = \frac { 1 } { | A _ { i } | } \sum _ { o \in A _ { i } } \mathbf { 1 } [ o \in \mathcal { M } _ { i j } ] u _ { o , k ( o ) } .\tag{18}
$$

The main tables report the two persistence metrics, denoted App. Persist and Sem. Persist, because they penalize both object disappearance and identity degradation. All object metrics are higher-is-better and enter the same MemoryGain and NMR computation as the other pairwise metrics.

Local Geometric Correspondence. We detect local features using SuperPoint with at most 1024 keypoints per image and detection threshold $5 \times 1 0 ^ { - 4 }$ . Keypoints are matched using LightGlue with SuperPoint features and matching filter threshold 0.1. Let $K _ { i }$ and $K _ { j }$ be the numbers of detected keypoints in the two images and let $M _ { i j }$ be the set of LightGlue matches. We report the feature-correspondence ratio

$$
q _ { \mathrm { m a t c h } } ( f _ { i } , f _ { j } ) = \frac { \lvert M _ { i j } \rvert } { \operatorname* { m a x } ( \operatorname* { m i n } ( K _ { i } , K _ { j } ) , 1 ) } .\tag{19}
$$

To measure geometric compatibility of the matched points, we estimate a fundamental matrix with OpenCV RANSAC. Each matched keypoint is represented in homogeneous image coordinates as $\mathbf { x } _ { i } = ( u _ { i } , v _ { i } , 1 ) ^ { \top }$ and $\mathbf { x } _ { j } = ( u _ { j } , v _ { j } , 1 ) ^ { \top }$ . A valid two-view fundamental matrix F should satisfy the epipolar constraint

$$
\mathbf { x } _ { j } ^ { \top } F \mathbf { x } _ { i } = 0 ,\tag{20}
$$

where $F { \bf x } _ { i }$ is the epipolar line in the second image and $F ^ { \top } \mathbf { x } _ { j }$ is the corresponding epipolar line in the first image. In RANSAC, OpenCV repeatedly samples candidate correspondence subsets, estimates a rank-2 fundamental matrix from the linear epipolar equations, and counts as inliers the matches whose point-to-epipolar-line error is below 3.0 pixels. The confidence parameter is set to 0.99. If fewer than eight matches are available, or if robust estimation fails, the RANSAC inlier ratio is set to 0. Otherwise, with inlier set $I _ { i j } \subseteq M _ { i j }$ , we report

$$
q _ { \mathrm { i n l i e r } } ( f _ { i } , f _ { j } ) = \frac { | I _ { i j } | } { | M _ { i j } | } .\tag{21}
$$

Both local-geometry metrics are higher-is-better. Although the evaluation code can also compute Sampson residual diagnostics, those residual metrics are not used in the reported tables

## A.2 Vision-Language Evaluation Prompt

We apply the same fixed prompt to every frame pair in the revisit, baseline, and short-range sets. The vision-language model is not told which set a pair comes from; it assigns a pairwise persistent-state consistency score using only the two images. The rubric first asks whether the images contain evidence of the same local scene, then accounts for viewpoint and occlusion when judging shared content. This distinction is important because a pair with no recognizable shared scene should receive a low consistency score rather than a perfect score merely because no local contradiction can be identified. The full prompt is shown below.

You are an expert evaluator of Video World Models.   
Task: Assign a pairwise persistent-state consistency score to two images   
sampled from the same generated video.   
\* Image A is earlier and Image B is later.   
\* Do NOT assume that the images depict the same place.   
\* This is not raw pixel similarity: viewpoint, occlusion, and mild rendering   
changes may differ without implying a state inconsistency.   
\* Score whether the images provide evidence of the same local scene and, when   
they do, whether the shared scene content remains persistent.   
Procedure:   
1. Classify the scene relation as same, partial, different, or uncertain.   
2. If shared scene content exists, estimate the viewpoint change and identify   
the co-visible region.   
3. Judge identity, geometry, and persistence only where content can be   
compared, discounting differences explained by viewpoint or occlusion.   
4. If there is no recognizable shared scene content, assign low consistency;   
absence of comparable content is not evidence of perfect consistency.   
Viewpoint exemptions for same or partially overlapping scenes: do NOT penalize:   
\* Any difference plausibly caused by camera motion, perspective, scale, or   
occlusion.   
\* Objects entering or leaving the frame due to the viewpoint shift.   
\* Minor lighting, exposure, or rendering-noise differences.   
\* When uncertain whether a difference is from viewpoint or a true change,   
assume viewpoint.   
Inconsistency types (report only confident findings):   
\* scene: images depict different local scenes or share no recognizable place cues.   
\* identity: same object changed color, shape, material, texture, or category.   
\* geometry: impossible spatial layout or structural contradictions.   
\* persistence: object disappeared or appeared well inside the shared region,   
or count of repeated elements changed without occlusion explanation.   
Scoring anchors:   
\* 5.0 = strong evidence of the same local scene; shared state is consistent.   
\* 4.0 = same local scene with only minor confident inconsistencies.   
\* 3.0 = partial or uncertain scene overlap, or moderate inconsistencies.   
\* 2.0 = weak shared-scene evidence or major state inconsistencies.   
\* 1.0 = clearly different local scenes or fundamentally incompatible states.   
Intermediate values are allowed. Do not give a high score solely because   
the images are individually plausible or because no content is comparable.   
Output constraints: keep SHORT:   
\* scene\_relation is one of: same, partial, different, uncertain   
\* viewpoint\_difference <= 15 words   
\* at most 3 inconsistencies, each description <= 12 words   
\* reason <= 15 words   
\* Each inconsistency has ONLY keys: type, severity, description.   
Output ONLY this JSON (no markdown):   
{   
"scene\_relation": "same|partial|different|uncertain",   
"viewpoint\_difference": "...",   
"inconsistencies": [   
{   
"type": "scene|identity|geometry|persistence",   
"severity": "minor|moderate|major|catastrophic",   
"description": "..."   
}   
],   
"score": 0.0,   
"reason": ". . . "   
}  
Figure 5 previews the revisit, gap-matched baseline, and short-range pairs used in the interpretation below.

![](images/397efeedd1473ac5ad85d33f7f011cc0c32a8bb3354e85d8db804e3e53056712.jpg)  
Figure 5 Qualitative examples of the three pair types used by the relative protocol. Each row shows one model; columns show a revisit pair, a gap-matched non-revisit baseline, and a short-range reference from the same rollout. The protocol asks whether the revisit recovers more shared scene evidence than the long-gap baseline, calibrated by the consistency observed at short range.

## B Interpreting NMR and Its Dynamic-Range Dependence

NMR is the dimensionless score used for cross-metric aggregation, while MemoryGain (MG) is the direct metric-level revisit advantage. NMR divides MG by the short-range dynamic range (DR), which measures the consistency gap between short-range and baseline pairs. This normalization enables comparison across numerical scales but depends on a reliable denominator: when DR is small, NMR can be unstable or inflated. Overall NMR supports aggregate comparison, but individual values and close rankings should therefore be interpreted with MG and DR. Table 6 summarizes typical diagnostic cases.

Table 6 Diagnostic interpretation of NMR through its numerator MG and denominator DR. “High” and “low” are interpreted relative to the empirical scale of the corresponding metric and trajectory.
<table><tr><td>MG</td><td>DR</td><td>NMR</td><td>Diagnostic meaning</td></tr><tr><td>High</td><td>High</td><td>High</td><td>Reliable high retention. The revisit is clearly better than baseline and recovers a large fraction of the model&#x27;s short-range consistency.</td></tr><tr><td>High</td><td>High</td><td>Low / Moderate</td><td>Partial retention. The revisit has a clear advantage over baseline, but still falls short of the model&#x27;s short-range consistency.</td></tr><tr><td>High</td><td>Low</td><td>High</td><td>Caution case. The revisit advantage may be real, but NMR can be amplified because the short-range normalization scale is weak.</td></tr><tr><td>Low</td><td>High</td><td>Low</td><td>Revisit failure under a reliable scale. The metric can separate short- range pairs from baseline pairs, but the revisit gains little over baseline.</td></tr><tr><td>Low</td><td>Low</td><td>Unstable</td><td>Weak diagnostic regime. The metric provides little short-range separa- tion, so the normalized NMR value is not reliable.</td></tr><tr><td>Negative</td><td>Positive</td><td>Negative</td><td>Negative revisit effect. The revisit is less consistent than gap-matched non-revisit controls, indicating severe drift or scene/state replacement.</td></tr><tr><td>Positive</td><td>Near-zero / Neg- Unstable ative</td><td></td><td>Invalid or unreliable normalization. Short-range consistency is not clearly above baseline, so NMR is not reliable for aggregation.</td></tr></table>

![](images/6732bd3ac4a68ba41963840bf30b5ca6b3e213e1c94490c3798e500b16a8a894.jpg)  
Figure 6 Human annotation interface used in our study. Annotators view the generated video together with the target camera trajectory and motion-magnitude references, then independently rate temporal/revisit consistency, motion magnitude, and camera control as Bad, Normal, or Good. The separate controls make it possible to distinguish perceived revisit inconsistency from insuficient motion or failure to follow the commanded trajectory.

At the boundaries, $\mathrm { N M R } \leq 0$ means that the revisit does not outperform the baseline, with negative values indicating worse consistency than long-gap controls. NMR ≈ 1 reaches the metric’s short-range level, while a value above one exceeds it. Values above one are convincing only with positive MG and meaningful DR; otherwise they may reflect an unstable denominator. Figure 5 illustrates the pair types underlying this comparison.

## C Human Evaluation and Statistical Validation

## C.1 Protocol

The human study samples 10 rollouts for each of the seven video world models and three trajectory templates, yielding $7 \times 3 \times 1 0 = 2 1 0$ videos. Six annotators independently score each video along three axes: revisit consistency, generated motion, and camera control. Each axis uses an ordinal scale with 1 (Bad), 2 (Normal), and 3 (Good), and we average the six ratings per video. Human consistency is the target judgment; motion and camera control record whether a model visibly moves and follows the commanded path rather than remaining static or drifting arbitrarily. Figure 6 documents the interface presented to the annotators. Alongside the generated video, it provides the commanded camera trajectory and optical-flow-calibrated motion references, making the nuisance judgments concrete rather than asking annotators to infer them from verbal instructions alone.

We define an action-valid subset to test the benchmark after excluding the clearest control failures. A video is action-valid when both its mean camera-control rating and mean motion rating are at least 1.5. Inter-rater reliability, measured with ordinal Krippendorf’s $\alpha ,$ is 0.339 for consistency, 0.371 for motion, and 0.527 for camera control. The modest agreement on revisit consistency reflects the perceptual dificulty of deciding whether a long generated rollout recovers the same scene rather than a merely similar one; correlation magnitudes should therefore be interpreted with this annotation uncertainty in mind.

Table 7 Expanded human-validation diagnostics. (a) Family-level Spearman correlations; each entry aggregates only the benchmark metrics in that family. (b) Percentile bootstrap intervals under the primary trajectory–scene clustering and a conservative seven-method clustering.
<table><tr><td colspan="3">(a) Per-family ρ(Hcons)</td></tr><tr><td>Family</td><td>Raw</td><td>MG NMR</td></tr><tr><td>Appearance Fidelity</td><td>0.454</td><td>0.487 0.499</td></tr><tr><td>Scene Identity</td><td>0.733</td><td>0.457 0.618</td></tr><tr><td>Local Geometry</td><td>0.651</td><td>0.364 0.435</td></tr><tr><td>Persistent State</td><td>0.598</td><td>0.332 0.386</td></tr><tr><td>Object Identity</td><td>0.482</td><td>0.339 0.437</td></tr></table>

<table><tr><td colspan="2">(b) 95% Cl sensitivity to clustering</td></tr><tr><td>Score Trajectory-scene</td><td>Method</td></tr><tr><td>Raw [0.54,0.64]</td><td>[0.39, 0.65]</td></tr><tr><td>MG [0.33, 0.46]</td><td>[0.19, 0.48]</td></tr><tr><td>NMR [0.41,0.54]</td><td>[0.26, 0.56]</td></tr><tr><td>Overall NMR</td><td>[0.45,0.63] [0.29, 0.63]</td></tr></table>

## C.2 Score Construction and Uncertainty

For Raw, MG, and NMR, we first compute the Spearman correlation between each benchmark metric and the corresponding human rating across the 210 videos. Because correlations are dimensionless but not directly additive, we transform each metric-level correlation with Fisher’s z = arctanh(ρ), average within each of the five metric families, weight the five family means equally, and apply tanh to return to the correlation scale. Thus, a family contributes one fifth of the aggregate regardless of how many metrics it contains. Overall NMR follows a diferent but consistent route: we construct the family-first Overall NMR for each video and correlate that single per-video score directly with the human ratings.

Partial correlations use rank residualization. We rank the score and human-consistency variables, regress both on human-rated camera control, human-rated motion, and trajectory identity, and correlate the residuals. This tests whether the score retains an association with perceived consistency beyond these observed nuisance variables; it does not remove unmeasured diferences in realized viewpoint or action execution.

We compute 95% confidence intervals with a percentile cluster bootstrap using B = 1000 replicates. Clusters are keyed by trajectory and reference scene, so outputs from diferent models sharing the same reference condition remain together. Each replicate resamples these clusters with replacement and recomputes the complete statistic, including the metric-level correlations, Fisher-z family aggregation, and rank residualization for partial correlations. The 2.5th and 97.5th percentiles form the primary interval. As a conservative sensitivity analysis, we also bootstrap the seven model identities as clusters.

## C.3 Results and Interpretation

Table 3 reports the complete Raw, MG, NMR, and Overall NMR comparison. All four scores correlate positively with human consistency under the primary bootstrap. Raw has the largest unadjusted correlation (0.594), but also a substantially stronger negative correlation with human-rated motion (−0.336) than MG (−0.053), NMR (−0.119), or Overall NMR (−0.151). Absolute similarity therefore aligns with human consistency partly because slowly moving videos are easier to match.

After controlling for camera control, motion, and trajectory, both NMR (0.149, 95% CI [0.05, 0.25]) and Overall NMR (0.106, 95% CI [0.02, 0.26]) remain positively correlated with consistency. They also remain positive on the action-valid subset (0.369 and 0.409). These results support perceptual alignment and reduced motion dependence; they do not show that a relative score must have a larger unadjusted human correlation than Raw.

Table 7(a) shows positive human-consistency correlations for every metric family and score type. NMR is strongest for Scene Identity (0.618) and Appearance Fidelity (0.499), while its correlations for Local Geometry (0.435), Object Identity (0.437), and Persistent State (0.386) show that the alignment is not confined to low-level appearance. Panel (b) repeats the uncertainty analysis with model identity as the clustering unit.

Table 8 Full metric-level motion correlations. Motion is the mean adjacent-frame optical-flow magnitude computed with RAFT. Correlations are computed within each model and aggregated across models with equal weighting using Fisher’s z transformation. LPIPS raw and baseline signs are oriented so that higher always indicates better similarity; <sup>∗</sup> denotes a 95% confidence interval excluding zero.
<table><tr><td colspan="5">Within-model Spearman ρ with objective motion</td></tr><tr><td>Metric</td><td>Raw revisit</td><td>Temporal baseline</td><td>MG</td><td>NMR</td></tr><tr><td colspan="5"></td></tr><tr><td>PSNR</td><td>Appearance Fidelity -0.242*</td><td>-0.235*</td><td>-0.063</td><td>0.034</td></tr><tr><td>SSIM</td><td>-0.282*</td><td>-0.293*</td><td>0.045</td><td>0.109</td></tr><tr><td>LPIPS</td><td>-0.373*</td><td>-0.384*</td><td>0.041</td><td>0.124</td></tr><tr><td colspan="5"></td></tr><tr><td>DINO</td><td>Scene Identity -0.211*</td><td>-0.257*</td><td>0.159*</td><td>0.049</td></tr><tr><td>BoQ</td><td>-0.342*</td><td>-0.417*</td><td>0.167*</td><td>0.062</td></tr><tr><td>MutualVPR</td><td>-0.268*</td><td>-0.362*</td><td>0.230*</td><td>0.078*</td></tr><tr><td colspan="5">Local Geometric Correspondence</td></tr><tr><td>Match ratio</td><td>-0.259*</td><td>-0.372*</td><td>0.118</td><td>0.099</td></tr><tr><td>RANSAC inlier</td><td>-0.172*</td><td>-0.301*</td><td>0.194*</td><td>0.159*</td></tr><tr><td colspan="5">Object Identity</td></tr><tr><td>Appearance persistence</td><td>-0.036</td><td>-0.094</td><td>0.128*</td><td>0.084</td></tr><tr><td>Semantic persistence</td><td>-0.085</td><td>-0.081*</td><td>0.052</td><td>-0.039</td></tr><tr><td colspan="5">Persistent State</td></tr><tr><td>Gemini score</td><td>-0.186*</td><td>-0.283*</td><td>0.116*</td><td>0.056</td></tr><tr><td>Median |ρ|</td><td>0.242</td><td>0.293</td><td>0.118</td><td>0.078</td></tr><tr><td>Significant metrics</td><td>9/11</td><td>10/11</td><td>6/11</td><td>2/11</td></tr></table>

The seven-cluster intervals are wider, as expected, but every lower bound remains above zero. We regard this result as a conservative sensitivity check rather than the primary uncertainty estimate.

For completeness, Table 9 provides the corresponding family-wise rank comparison across score types.

## D Motion and Controlled-Confound Diagnostics

## D.1 Full Motion-Correlation Results

Table 8 expands the family means in Table 4 into all 11 constituent metrics. Objective motion is the mean adjacent-frame optical-flow magnitude computed with RAFT. To isolate within-model variation, we compute the motion correlation separately for each model and aggregate the correlations across models with equal weighting in Fisher-z space.

Absolute revisit similarity and the temporal baseline retain substantial motion dependence. Their median |ρ| values are 0.242 and 0.293, and their 95% confidence intervals exclude zero for 9/11 and 10/11 metrics, respectively. The pattern is especially consistent for appearance, scene retrieval, and local geometry, where greater motion makes absolute frame matching harder. For example, BoQ changes from ρ = −0.342 for Raw and −0.417 for the temporal baseline to 0.062 for NMR; SSIM changes from −0.282 and −0.293 to 0.109.

Relative calibration attenuates this dependence: median |ρ| falls to 0.118 for MG and 0.078 for NMR, and only 2/11 NMR correlations have intervals excluding zero. MutualVPR and RANSAC inliers retain significant positive residual correlations, so NMR is not motion-invariant. Rather, subtracting the same-video temporal baseline removes much of the generic advantage enjoyed by slow or conservative videos while preserving revisit-selective diferences.

Table 9 Family-wise mean ranks among the seven video world models. Each cell reports Raw / MG / NMR; lower ranks are better. The Overall cell averages the corresponding ranks across the five families. RAFT is mean optical-flow magnitude in pixels per frame and is not ranked.
<table><tr><td colspan="8">Family-wise mean rank (Raw /MG /NMR)</td></tr><tr><td>Method</td><td>Appearance Fidelity</td><td>Scene Identity</td><td>Object Identity</td><td>Local Geometry</td><td>Persistent State</td><td>Overall</td><td>RAFT px/frame</td></tr><tr><td>LingBot-World-Fast</td><td>6.33/6.00 / 6.33</td><td>6.67 / 5.00 / 5.00</td><td>5.50 / 5.00 /5.00</td><td>6.00 /5.50 /5.00</td><td>6.00 / 5.00 /5.00</td><td>6.10 /5.30 /5.27</td><td>2.699</td></tr><tr><td>LingBot-World-2-Fast</td><td>6.33 /6.33 /6.33</td><td>5.00 / 6.00 / 6.00</td><td>5.50 /6.50 /6.50</td><td>5.00 /6.50 /6.50</td><td>5.00 / 6.00 / 6.00</td><td>5.37 /6.27 /6.27</td><td>2.482</td></tr><tr><td>Matrix-Game 3.0</td><td>4.00 / 4.00 / 4.00</td><td>4.00 /2.33 /3.33</td><td>1.50 / 1.00 / 2.00</td><td>3.50 /2.50/3.00</td><td>4.00 /3.00 / 3.00</td><td>3.40 / 2.57 /3.07</td><td>2.199</td></tr><tr><td>HY-WorldPlay</td><td>2.00 / 2.00 / 2.00</td><td>1.00 / 2.67 / 1.67</td><td>1.50 / 2.50 / 2.50</td><td>1.00 / 3.00 / 2.00</td><td>1.00 / 2.00 / 2.00</td><td>1.30 / 2.43 / 2.03</td><td>1.038</td></tr><tr><td>SANA-WM</td><td>5.00 /5.67 /5.33</td><td>6.33 / 7.00 / 7.00</td><td>7.00 /6.50 /6.50</td><td>7.00 /6.00 /6.50</td><td>7.00 /7.00 / 7.00</td><td>6.47 /6.43 /6.47</td><td>1.309</td></tr><tr><td>Lyra-2</td><td>3.33 /3.00 /3.00</td><td>3.00/3.67/3.67</td><td>4.00 / 4.00 / 4.00</td><td>3.00 /3.50 /4.00</td><td>3.00 / 4.00 / 4.00</td><td>3.27 /3.63 /3.73</td><td>2.028</td></tr><tr><td>DreamX-World-Memo</td><td>1.00/1.00/1.00</td><td>2.00/1.33/1.33</td><td>3.00/2.50/1.50</td><td>2.50/1.00/1.00</td><td>2.00/1.00 /1.00</td><td>2.10/1.37/1.17</td><td>2.174</td></tr></table>

## D.2 Controlled Confound Interpretation

The across-video analysis above uses natural variation within each model. Table 5 in the main text complements it with controlled DreamX-World-Memo comparisons. Across the natural motion split, absolute revisit scores change markedly: from Low- to High-motion, SSIM falls from 0.507 to 0.452, DINO similarity from 0.916 to 0.691, and geometric inlier ratio from 0.829 to 0.722. MG remains positive for every displayed metric and motion group, but its trend need not be monotonic because the temporal baseline changes with motion as well. This is the rollout-specific nuisance variation that relative scoring is designed to calibrate.

The quality perturbations provide a stricter same-subset comparison: blur, downsample–upsample, and compression are applied to the same High-motion videos. Although absolute SSIM ranges from 0.451 to 0.550, SSIM MG remains within 0.045–0.053 and NMR within 0.33–0.37. LPIPS MG remains within 0.089–0.096, while DINO MG and NMR remain within 0.071–0.073 and at 0.22, respectively. Geometry shows somewhat greater residual variation. These controlled results support attenuation of globally shared quality shifts, not complete invariance to degradation or motion.

## E Family-wise Rank Comparison

Table 9 compares score types without averaging incompatible metric units. For each metric, we rank the seven video world models and then average ranks only within its family. Within each family cell, ranks are reported in Raw/MG/NMR order; NMR and MG use the oriented higher-is-better convention. Raw-score ranks use higher-is-better for all metrics except LPIPS. Each cell reports only the resulting family-wise mean rank; the underlying metric values are reported in Table 2. The Overall cell then averages the Raw, MG, and NMR ranks, respectively, across the five families so that every family contributes equally. The RAFT column averages 20 adjacent-frame pairs from each of 20 videos per trajectory (60 videos per model) at a common 320-pixel width. It measures generated motion magnitude and is reported only to diagnose whether score rankings track slower or faster rollouts.