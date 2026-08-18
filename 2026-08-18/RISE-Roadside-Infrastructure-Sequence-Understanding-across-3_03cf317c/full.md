# RISE: Roadside Infrastructure Sequence Understanding across 3D Tracking and Structured Vision-Language Reasoning

Yanbo Jiang<sup>1</sup>, Haotian Zheng<sup>1</sup>, Jiahao Wang<sup>1</sup>, Hanxiao Ren<sup>1</sup>, Yitao Xu<sup>1</sup>, Yining Xing<sup>1</sup>, Zehong Ke<sup>1</sup>, Hao Cheng<sup>1</sup>, Yiqian Tu<sup>1</sup>, Jinhao Li<sup>1</sup>, Zhiyuan Xuan<sup>2</sup>, Fang Zhang<sup>1∗</sup>, Jianqiang Wang<sup>1∗</sup>

<sup>1</sup>School of Vehicle and Mobility, Tsinghua University <sup>2</sup>TsingCloud

## Abstract

We present RISE (Roadside Infrastructure Sequence Understanding and Evaluation), a framework spanning metric 3D tracking and structured vision-language reasoning in roadside sequences. For metric tracking, our image-only method combines SAM3 video identities with calibration-guided mask agreement for multi-view identity association, recovering persistent 3D tracks without LiDAR or task-specific 3D training. Its calibration-conditioned geometry allows the procedure to be instantiated at diferent calibrated multi-camera intersections without layout-specific retraining. On 20 humanreviewed clips from six intersections, the generated tracks achieve 66.9 MOTA within the defined multi-view evaluation scope. For structured vision-language reasoning, a humanreviewed MLLM pipeline mines high-value clips and uses a constrained full-context Oracle to construct bbox-grounded predictive QA without exposing future evidence to evaluated models. The resulting RISE-VQA dataset contains 33,910 QA pairs from 557 clips across 16 intersections and 61 roadside views. Its intersection-held-out RISE-Bench evaluates semantic choices, coordinates, future boxes, and interaction sets with deterministic task-specific metrics. Experiments show consistent benefits from domain adaptation and generally from temporal context, while revealing persistent challenges in spatial grounding, future localization, and interaction reasoning.

## Introduction

Fixed roadside cameras are a key sensing layer for intelligent transportation systems, supporting intersection monitoring, trafic-safety analysis, and infrastructure-assisted perception through persistent, scene-centric observations (Yu et al. 2023). Turning these sequences into actionable trafic understanding requires two complementary capabilities: metric 3D tracking establishes persistent identities, 3D states, and trajectories, while structured vision-language reasoning provides queryable interpretations of agent states, maneuvers, interactions, and future outcomes. Prior work has largely studied these capabilities separately, emphasizing metric tracking and trajectory forecasting in vehicle–infrastructure sequences (Yu et al. 2023) or language-based trafic-scene interpretation in driving VQA (Xu, Huang, and Liu 2021; Qian et al. 2024; Sima et al. 2024).

These capabilities draw on complementary spatial and temporal evidence in fixed roadside sequences. Spatially, synchronized cameras with overlapping fields of view provide multi-view geometric constraints for metric 3D estimation; temporally, complete sequences capture event evolution and outcomes needed for predictive reasoning. Yet converting this information into reusable supervision remains challenging. Camera-only roadside 3D detection often requires task-specific 3D labels or adaptation to new camera layouts (Yang et al. 2023, 2024; Zhang et al. 2025). For predictive VQA, high-value events are sparse, and realized outcomes must provide supervision without exposing future evidence to evaluated models. Scaling roadside collection across intersections further requires access, calibration, and synchronization at each site.

To address these challenges, we introduce RISE (Roadside Infrastructure Sequence Understanding and Evaluation), which exploits complementary spatial and temporal evidence in fixed-camera roadside sequences. Its tracking branch combines calibrated multi-view agreement with video identities to construct persistent metric 3D tracks, while its visionlanguage branch converts realized event outcomes into bboxgrounded predictive supervision through a constrained Oracle. Together, the two branches organize roadside sequence understanding around persistent trafic agents, covering metric state estimation and semantic event interpretation with task-specific inputs.

Our contributions are:

1. A roadside sequence task framework. We formulate metric 3D tracking and structured vision-language reasoning as complementary tasks centered on persistent trafic agents while retaining task-specific inputs.

2. Training-free multi-view 3D tracking. We combine SAM3 view-local identities with calibration-guided mask agreement, MWIS-based association, and track-level refinement to recover persistent metric 3D tracks without LiDAR or task-specific 3D training. The calibrationconditioned procedure can be instantiated at diferent calibrated multi-camera intersections, and its outputs initialize subsequent human refinement and quality assessment.

3. RISE-VQA with Oracle-grounded predictive supervision. We construct 33,910 bbox-grounded QA pairs from 557 clips across 16 intersections and 61 roadside views. A human-reviewed MLLM pipeline uses a constrained full-context Oracle to generate predictive targets without exposing future evidence to evaluated models.

4. Intersection-held-out structured evaluation. We introduce RISE-Bench to evaluate semantic choices, coordinates, future boxes, and interaction sets on unseen intersections using deterministic task-specific metrics. Baselines quantify domain-adaptation and temporal-context efects while exposing limitations in grounding, prediction, and interaction reasoning.

## Related Work

## Roadside 3D Perception and Tracking

Roadside perception datasets such as DAIR-V2X (Yu et al. 2022), V2X-Seq (Yu et al. 2023), RCooper (Hao et al. 2024), and Rope3D (Ye et al. 2022) provide calibrated infrastructure sensors and 3D annotations for vehicle-infrastructure perception. V2X-Seq further augments sequential frames with persistent identities, trajectories, vector maps, and trafic-light signals, and defines VIC3D tracking together with online and ofline forecasting (Yu et al. 2023). I-24 3D provides continuous vehicle trajectories from overlapping calibrated highway cameras (Gloudemans et al. 2023). RISE shares this sequence-centric perspective but studies a diferent setting: fixed multi-camera, image-only 3D tracking together with structured vision-language reasoning over roadside events.

Camera-only roadside 3D methods exploit camera calibration, ground geometry, or BEV fusion (Yang et al. 2023, 2024; Zhang et al. 2025), but generally learn 3D perception from task-specific supervision. In contrast, our method uses video-consistent segmentation identities as temporal evidence and calibration-guided mask agreement as cross-view geometric evidence. Its projection geometry is instantiated directly from each deployment’s calibration, allowing the same procedure to operate at a new calibrated multi-camera intersection with suficient overlap, without training a layoutspecific 3D model.

## Structured Vision-Language Reasoning for Roadside Scenes

Driving-oriented VQA datasets include SUTD-TraficQA (Xu, Huang, and Liu 2021), NuScenes-QA (Qian et al. 2024), DriveLM (Sima et al. 2024), and Talk2BEV (Choudhary et al. 2024). NuScenes-QA and DriveLM construct QA from vehicle-centric autonomous-driving data, while SUTD-TraficQA targets video reasoning over mixed trafic events. Recent work extends this scope to infrastructure and cooperative settings: RoadSceneVQA (Guan et al. 2026) studies image-level roadside reasoning, TUMTraf VideoQA (Zhou et al. 2025) targets spatiotemporal roadside video understanding, V2X-QA (You et al. 2026) evaluates cooperative VQA on V2X-Seq, and LTD/UniVLT (Huang et al. 2026) considers heterogeneous multi-camera reasoning tasks.

As summarized in Table 1, RISE complements existing benchmarks in temporal input, target grounding, output structure, and evaluation protocol. It uses newly collected clips from fixed roadside cameras to capture short-term trafic evolution while keeping the visual context compact. Each queried road user is referenced by its observed bounding box rather than a descriptive phrase, requiring the model to visually ground and associate the agent across frames. The answers include semantic choices, spatial coordinates, future 2D bounding boxes, and set-valued interactions. RISE-Bench scores these heterogeneous outputs with deterministic task-specific metrics and holds out complete intersections to evaluate generalization to unseen intersection layouts.

## Method

## Sequence Task Formulation

Let $I _ { t } ^ { ( v ) }$ denote frame t from fixed camera v. Given a completed roadside recording, the 3D tracking branch uses four synchronized views to recover

$$
\mathcal { T } = \{ \tau _ { i } \} , \qquad \tau _ { i } = \{ ( \mathbf { b } _ { i , t } ^ { 3 D } , \mathrm { i d } _ { i } ) \} _ { t } ,\tag{1}
$$

where $\mathbf { b } _ { i , t } ^ { 3 D }$ contains the metric center, dimensions, and heading of agent i, and ${ \mathrm { i d } } _ { i }$ identifies the agent across the sequence.

The VQA branch instead evaluates reasoning under partial observation. Each view is processed independently to avoid the visual-token and cross-view association burden of multiview temporal input. For each 20-frame clip, the evaluated VLM observes only $I _ { 0 } ^ { ( v ) } – I _ { 4 } ^ { ( v ) }$ , whereas the annotation Oracle may inspect the complete sequence $I _ { 0 } ^ { ( v ) } – I _ { 1 9 } ^ { ( v ) }$ solely to determine supervision targets. Each QA instance grounds a queried agent with an observed 2D box and requires a semantic choice, spatial coordinates, future 2D boxes, or an interaction set. The two tasks use roadside sequences diferently: 3D tracking exploits multi-view temporal identity evidence over a completed recording, whereas VQA evaluates an observed prefix using supervision derived from the complete event. Together, they form RISE’s sequence-understanding framework.

## Identity-Backbone Multi-View 3D Tracking

SAM3 provides temporally consistent mask IDs within each camera, but these IDs are not shared across views. RISE converts them into calibration-conditioned cross-view signatures and uses MWIS to select non-conflicting identity backbones, establishing global object identities before metric cuboid fitting. Unlike frame-level 3D detect-then-associate pipelines, this identity-first design turns SAM3’s temporal continuity into multi-view 3D tracks and retains association evidence when some views are occluded.

Video IDs and Calibrated Semantic Voxels Given four synchronized and calibrated views, SAM3 (Carion et al. 2025) produces category-specific instance masks with temporally consistent IDs within each view. We additionally mark static occluders to distinguish occlusion from missing segmentation. We project a cached 3D voxel grid into each image and record the mask ID covering each projected voxel. For voxel n at time $t ,$ the cross-view tuple

$$
\pmb { \sigma } _ { n , t } = ( s _ { n , 1 , t } , \dots , s _ { n , V , t } )\tag{2}
$$

forms a semantic signature, where $s _ { n , v , t }$ denotes either a view-local mask ID or a visibility state in camera v. Suficiently supported signatures propose cross-view object hypotheses. Spatially adjacent hypotheses sharing at least one view-local ID are merged into candidate identity backbones, accommodating mask fragmentation and missing view components. Visibility-aware voting distinguishes visible, outof-view, and occluded projections, so partially occluded evidence is downweighted rather than removed. Because SAM3 IDs are view-local, diferent candidate backbones may reuse the same ID and therefore cannot all be valid. We therefore construct a conflict graph and solve a maximum-weight independent set (MWIS), using visibility-aware voxel support as candidate weights. The selected non-conflicting backbones assign each view-local ID to at most one global object and define the global IDs used for tracking.

Table 1: Comparison with trafic-scene vision-language datasets. “Inherited” denotes 3D labels from a source dataset; “Img. tracker” denotes the image-derived tracking component proposed in RISE.
<table><tr><td>Dataset</td><td>Venue</td><td>View</td><td>Unit</td><td>Source</td><td>QA Gen.</td><td>#QA</td><td>Scoring</td><td>3D Comp.</td></tr><tr><td>NuScenes-QA (Qian et al. 2024)</td><td>AAAI&#x27;24</td><td>ego</td><td>Multi-view</td><td>nuScenes</td><td>Template</td><td>460k</td><td>Accuracy</td><td>Inherited</td></tr><tr><td>DriveLM (Sima et al. 2024)</td><td>ECCV&#x27;24</td><td>ego</td><td>Frame</td><td>nuScenes</td><td>Temp.+Human</td><td>443k</td><td>SPICE/GPT</td><td>Inherited</td></tr><tr><td>LingoQA (Marcu et al. 2024)</td><td>ECCV&#x27;24</td><td>ego</td><td>Video</td><td>Self</td><td>Human+LLM</td><td>420k</td><td>Lingo-Judge</td><td></td></tr><tr><td>SUTD-TrafficQA (Xu, Huang, and Liu 2021) CVPR&#x27;21</td><td></td><td>mixed</td><td>Video</td><td>Mixed</td><td>Human</td><td>62.5k</td><td>Acc.</td><td></td></tr><tr><td>TUMTraf VideoQA (Zhou et al. 2025)</td><td>ICML&#x27;25</td><td>roadside</td><td>Video</td><td>Self</td><td>LLM+Ver.</td><td>85.0k</td><td>Acc.</td><td></td></tr><tr><td>RoadSceneVQA (Guan et al. 2026)</td><td>AAAI&#x27;26</td><td>roadside</td><td>Image</td><td>Rope3D</td><td>LLM+Ver.</td><td>34.7k</td><td>Text/GPT</td><td>Inherited</td></tr><tr><td>V2X-QA (You et al. 2026)</td><td>arXiv&#x27;26</td><td>V2X</td><td>View pair</td><td>V2X-Seq</td><td>LLM+Ver.</td><td>33.2k</td><td>Acc.</td><td>Inherited</td></tr><tr><td>LTD (Huang et al. 2026)</td><td>arXiv&#x27;26</td><td>roadside</td><td>Multi-img</td><td>Self</td><td>LLM+Ver.</td><td>11.6k</td><td>GPT/Acc.</td><td></td></tr><tr><td>RISE (Ours)</td><td></td><td>roadside</td><td>Clip</td><td>Self</td><td>LLM+Ver.</td><td>33.9k Task-specific Img. tracker</td><td></td><td></td></tr></table>

![](images/ea53c5b7334803ee4f0b08003a8991b310c3aa432bbaa922a0cb1ae13d44640f.jpg)  
Figure 1: Identity-first multi-view 3D tracking in RISE. View-local SAM3 video IDs are grouped into global identity backbones through calibrated voxel signatures and MWIS, followed by metric cuboid fitting and track-level refinement.

The pipeline is training-free for the target 3D tracking task, requiring neither 3D box supervision nor detector training on the collected intersections. It requires accurate calibration and suficient overlap, but no LiDAR or layout-specific retraining. At a new intersection, only the cached voxel projections and site-specific search region are rebuilt from its calibration, while the tracking procedure and parameters remain unchanged.

Frame-Level Metric Cuboid Fitting After MWIS selects a consistent set of identity backbones, we optimize a metric cuboid B for each object using

$$
J ( B ) = \sum _ { n \in B } w _ { n } - \lambda \frac { V _ { B } } { \Delta ^ { 3 } } ,\tag{3}
$$

where $w _ { n }$ is the visibility-aware voxel support, V<sub>B</sub> is cuboid volume, ∆ is the voxel-grid spacing, and the volume term penalizes unsupported space. Category-specific size and height bounds remove implausible cuboids.

Track Construction and Temporal Refinement Each MWIS-selected backbone links the SAM3 identities of one physical object across cameras. Because its component identities are video-consistent within their views, the resulting global identity propagates across frames even when one component temporarily disappears. Track-level refinement treats per-frame cuboids as noisy observations: it smooths centers, estimates a consensus size, corrects unstable headings using motion, and fills short gaps caused by occlusion. Each output track therefore contains a persistent ID and a time-indexed sequence of metric 3D boxes.

## Oracle-Grounded Structured VQA

Figure 2 summarizes the constrained-Oracle workflow used to construct RISE-VQA.

Data Collection and Scene Mining We collect synchronized roadside videos from 16 urban intersections, yielding 61 fixed-camera views; 13 intersections provide four directional views. Each selected VQA clip contains 20 frames sampled at 5 Hz. The question and queried-agent box are anchored at $I _ { 0 } , I _ { 1 } – I _ { 4 }$ provide observed motion context, and later key frames supply Oracle-only targets.

![](images/a163d3765f8debe80d1f25a45d25247fb32afbfdac76f088e4c8cd4ac84bc87d.jpg)  
Figure 2: Structured VQA production in RISE. The pipeline mines high-value clips, constructs frame-level references, generates Oracle-grounded predictive QA pairs, and applies targeted human review.

To avoid spending full-sequence annotation on routine trafic, a lightweight MLLM scores low-rate summaries of 25,910 candidate clips by corner-case severity and interaction risk. Selected clips are then expanded into full 5-Hz sequences for structured QA construction.

Infrastructure Priors and Structured References Directly asking a single MLLM to localize small trafic signals and bind road users across a long clip is unreliable. Fixed roadside views allow us to reuse infrastructure priors across clips, including signal regions, lane geometry, and stop lines. Specialized reference modules recognize cropped signal states and localize road users with coarse headings on selected key frames. These structured references provide localization and association cues only to the annotation Oracle; they are never exposed to models during training or evaluation.

Observation–Supervision Separation Question templates are restricted to evidence available in the observed prefix. For predictive QA, the Oracle uses the complete sequence and structured references to derive realized future boxes, maneuvers, and interactions. Training and evaluation receive only $I _ { 0 } { - } I _ { 4 }$ and questions anchored to observed boxes; future frames, references, and Oracle reasoning remain hidden.

Human Review and Visual Grounding Five annotators review the generated QA through add, revise, and delete operations, producing more than 8,000 edits, primarily on dynamic questions involving subtle inter-frame motion. Each question type undergoes first-pass review and secondary checking, with uncertain cases adjudicated separately. Human review also enforces the observation boundary by checking that question wording contains no future-derived cues. Released QA refers to trafic agents by their observed-frame bounding boxes rather than attributes such as color or type. Models must therefore localize the queried agent from the box and maintain its correspondence across the observed frames before answering.

Table 2: Observation–supervision boundary in the constrained Oracle protocol. Realized future observations determine predictive targets but remain hidden from train/test inputs.
<table><tr><td>Information</td><td>Annotation Role</td><td>Train/Test Input</td></tr><tr><td>Early frames  $I _ { 0 } { - } I _ { 4 }$ </td><td>Evidence</td><td>Yes</td></tr><tr><td>Future frames  $I _ { 5 } { - } I _ { 1 9 }$ </td><td>Oracle evidence</td><td>No</td></tr><tr><td>Future boxes</td><td>Supervision target</td><td>No</td></tr><tr><td>Infrastructure priors</td><td>Reference</td><td>No</td></tr></table>

## Experiments

## Dataset and RISE-Bench

The reviewed corpus contains 34,545 QA pairs, including 11,846 static-environment and 22,699 dynamic-behavior questions. We hold out three complete intersections, covering eleven roadside views and both cross- and T-junction layouts. By separating complete intersections rather than clips, this protocol avoids shared layout and camera context and measures cross-intersection generalization (Sekaran et al. 2025), as validated in Table 7. From the 5,326 held-out instances, RISE-Bench retains 4,691 with deterministic task-specific metrics and excludes 635 open-ended or weakly structured instances. Together with 29,219 training instances, the released dataset contains 33,910 QA pairs.

![](images/055d5257a2e5891639be68ce5f30488aeee39418aca083d5067234a396c7ab81.jpg)  
Figure 3: Examples of bbox-grounded structured VQA. Boxes identify queried agents and future locations, while colored geometry provides infrastructure and interaction context.

Table 3: RISE-VQA and RISE-Bench statistics.
<table><tr><td>Metric</td><td>Value</td><td>Metric</td><td>Value</td></tr><tr><td>Intersections</td><td>16</td><td>Roadside views</td><td>61</td></tr><tr><td>4-view / 3-view</td><td>13 /3</td><td>Event clips</td><td>557</td></tr><tr><td>Reviewed QA</td><td>34,545</td><td>Released QA</td><td>33,910</td></tr><tr><td>Training QA</td><td>29,219</td><td>Held-out Pool</td><td>5,326</td></tr><tr><td>RISE-Bench QA</td><td>4,691</td><td>Reviewed Static</td><td>11,846 (34.3%)</td></tr><tr><td></td><td></td><td>Reviewed Dynamic</td><td>22,699 (65.7%)</td></tr></table>

Table 4: RISE-Bench composition.
<table><tr><td>Task Family</td><td>#QA</td><td>Metric</td></tr><tr><td>Multiple choice</td><td>3,794</td><td>Choice score</td></tr><tr><td>Coordinate output</td><td>745</td><td>IoU / F1 / C-ADE</td></tr><tr><td>Set prediction</td><td>152</td><td>Precision / Recall / F1</td></tr><tr><td>Total</td><td>4,691</td><td>一</td></tr></table>

RISE-Bench has three task families: multiple-choice questions cover semantic trafic understanding; coordinate questions cover infrastructure geometry and future boxes; and set-valued questions identify interacting agents or pairs.

## Models and Metrics

We evaluate three open-source VLMs—Qwen2.5-VL-7B, InternVL3-8B, and MiniCPM-V-4.5-8B—under zero-shot (ZS) and fine-tuned (FT) settings. FT denotes five-frame LoRA adaptation, and both training and evaluation observe only the first five frames of each clip. LoRA experiments use LLaMA-Factory (Zheng et al. 2024) with rank 8, scaling factor 16, learning rate $5 \times 1 0 ^ { - 5 }$ , efective batch size 32, and three epochs on 29,219 training instances; the vision encoder and multimodal projector remain frozen. Primary runs use direct-answer inference with thinking disabled, while thinking-enabled proprietary models are included as strong zero-shot references marked by <sup>†</sup>. The ZS–FT comparison measures the efect of domain-specific adaptation, and the

1f–5f ablation measures the contribution of temporal context.

Choice score assigns full, partial, or zero credit using predefined mappings for each question type; partial credit applies only to ordinal or adjacent choices. Det-F1 matches predicted and reference boxes by IoU, while Line-F1 matches lane and stop-line segments by endpoint distance. Future boxes are evaluated by trajectory IoU (T-IoU), which averages box overlap across future steps, and center average displacement error (C-ADE),

$$
\mathrm { C - A D E } = \frac { 1 } { 3 N } \sum _ { i = 1 } ^ { N } \sum _ { \tau = 1 } ^ { 3 } \lVert \hat { \mathbf { c } } _ { i , \tau } - \mathbf { c } _ { i , \tau } \rVert _ { 2 } .\tag{4}
$$

Here $\mathbf { c } _ { i , \tau }$ is the ground-truth box center in the common 1000 × 1000 relative coordinate space. Interaction predictions use label-agnostic bbox set matching, and Inter-F1 is macro-averaged over QA samples.

## Structured Vision-Language Results

Table 5 shows that domain-specific adaptation consistently improves all three open-source backbones across answer formats. For rows marked by <sup>‡</sup>, native bbox serialization and coordinates are normalized before scoring, so the gains cannot be attributed solely to output-format alignment. Among the fine-tuned models, Qwen2.5-VL-7B leads in aggregate MCQ and Line-F1, whereas InternVL3-8B performs best on dynamic MCQ, detection, future-box, and interaction metrics. This variation supports separately evaluating semantic choices, geometric grounding, future localization, and interaction sets. Thinking improves both paired proprietary baselines but does not dominate every structured task.

## Temporal Context Ablation

Across all three backbones, 5f improves D-MCQ and T-IoU while reducing C-ADE, providing consistent evidence that observed motion benefits dynamic reasoning and future localization. Because static questions are anchored in the first frame shared by both settings, S-MCQ provides a useful reference: it changes little for Qwen2.5-VL and MiniCPM-V, while improving modestly for InternVL3. The gain remains backbone-dependent, with InternVL3 showing the largest trajectory improvement.

We retrain Qwen2.5-VL-7B under the intersection-heldout and clip-random train/test partitions. Because the full test sets difer in composition, their rows provide only an overall view. On the common 1,451-QA subset, clip-random training is stronger across all metrics, including S-MCQ (.841 vs. .772), D-MCQ (.796 vs. .775), and Det-F1 (.654 vs. .574). Shared intersection context therefore makes clip-level splitting easier, motivating intersection-held-out evaluation.

## 3D Tracking Quality Analysis

We evaluate the unedited tracker output as the prediction and the manually corrected tracks as GT. As shown in Figure 5, the tracker provides initial 3D tracks, while synchronized views and the semantic-voxel BEV help annotators add missed eligible vehicles, remove false tracks, and correct box geometry and identity.

![](images/7526a809f891c6f689f54e6212b59a1f5c53d593ab38ff6ef3aed234753e0add.jpg)  
(b) Question sub-category distribution

![](images/aff168e797635101559b91a4b8fb81cb1e65142f796b1ba40398050820ced9a4.jpg)  
Figure 4: Distribution of the 34,545 human-reviewed QA pairs before benchmark filtering. Left: frequent concepts in the question corpus. Right: fine-grained sub-category counts.

Table 5: Main results on RISE-Bench. S-MCQ and D-MCQ denote static and dynamic MCQ scores. ZS-5f and FT-5f denote five-frame zero-shot inference and LoRA adaptation, respectively. <sup>‡</sup> marks bbox evaluation normalized from model-native serialization and coordinate systems; <sup>†</sup> marks thinking-enabled proprietary zero-shot runs. Other rows use direct-answer inference. Bold indicates the best result within the proprietary-reference or fine-tuned group.
<table><tr><td>Model</td><td>Access Setting</td><td></td><td>|S-MCQ</td><td>D-MCQ</td><td>MCQ|</td><td>Det-F1</td><td>1 Line-F1</td><td>|T-IoU</td><td>C-ADE↓</td><td>Inter-F1</td></tr><tr><td>Qwen2.5-VL-7B</td><td>Open</td><td>ZS-5f‡</td><td>.479</td><td>.533</td><td>.510</td><td>.078</td><td>.001</td><td>.305</td><td>120.4</td><td>.066</td></tr><tr><td>InternVL3-8B</td><td>Open</td><td>ZS-5f</td><td>.503</td><td>.424</td><td>.458</td><td>.029</td><td>.029</td><td>.050</td><td>306.8</td><td>.015</td></tr><tr><td>MiniCPM-V-4.5-8B</td><td>Open</td><td>ZS-5f‡</td><td>.507</td><td>.571</td><td>.544</td><td>.061</td><td>.055</td><td>.257</td><td>131.3</td><td>.042</td></tr><tr><td>GPT-5.5</td><td>Closed ZS-5f</td><td></td><td>.558</td><td>.593</td><td>.578</td><td>.392</td><td>.798</td><td>.248</td><td>117.8</td><td>.106</td></tr><tr><td>GPT-5.5†</td><td>Closed ZS-5f</td><td></td><td>.610</td><td>.632</td><td>.623</td><td>.649</td><td>.920</td><td>.402</td><td>60.8</td><td>.238</td></tr><tr><td>Gemini-3.1-Pro†</td><td>Closed ZS-5f‡</td><td></td><td>.502</td><td>.491</td><td>.495</td><td>.187</td><td>.719</td><td>.323</td><td>112.9</td><td>.214</td></tr><tr><td>Qwen3.7-Plus</td><td>Closed ZS-5f</td><td></td><td>.635</td><td>.649</td><td>.643</td><td>.382</td><td>.738</td><td>.311</td><td>99.1</td><td>.251</td></tr><tr><td>Qwen3.7-Plus†</td><td>Closed ZS-5f</td><td></td><td>.640</td><td>.666</td><td>.655</td><td>.599</td><td>.791</td><td>.374</td><td>64.2</td><td>.332</td></tr><tr><td>Qwen2.5-VL-7B</td><td>Open</td><td>FT-5f</td><td>.782</td><td>.781</td><td>.781</td><td>.643</td><td>.860</td><td>.381</td><td>74.8</td><td>.271</td></tr><tr><td>InternVL3-8B</td><td>Open</td><td>FT-5f</td><td>.734</td><td>.791</td><td>.767</td><td>.672</td><td>.845</td><td>.433</td><td>62.2</td><td>.350</td></tr><tr><td>MiniCPM-V-4.5-8B</td><td>Open</td><td>FT-5f</td><td>.722</td><td>.764</td><td>.746</td><td>.648</td><td>.771</td><td>.344</td><td>91.2</td><td>.305</td></tr></table>

Table 6: Temporal-context ablation. Each pair uses the same backbone and LoRA recipe; only the number of input frames difers.
<table><tr><td>Model</td><td>Fr.</td><td>S-MCQ</td><td>D-MCQ</td><td>T-IoU</td><td>C-ADE↓</td></tr><tr><td>Qwen2.5-VL</td><td>1f</td><td>.784</td><td>.730</td><td>.312</td><td>109.1</td></tr><tr><td></td><td>5f</td><td>.782</td><td>.781</td><td>.381</td><td>74.8</td></tr><tr><td>InternVL3</td><td>1f</td><td>.698</td><td>.738</td><td>.324</td><td>104.0</td></tr><tr><td></td><td>5f</td><td>.734</td><td>.791</td><td>.433</td><td>62.2</td></tr><tr><td>MiniCPM-V-</td><td>1f</td><td>.727</td><td>.730</td><td>.322</td><td>115.1</td></tr><tr><td>4.5-8B</td><td>5f</td><td>.722</td><td>.764</td><td>.344</td><td>91.2</td></tr></table>

All evaluated scenes use the same voxelized worldcoordinate region, x, y ∈ [−40, 40] m, with an approximately 0.2 m grid step. We regard a vehicle as observable when it is clearly identifiable at suficient scale in at least one camera; tiny or distant vehicles are excluded. Among observable vehicles, those inside the voxel region whose projected regions lie within the image bounds of at least three synchronized cameras enter the frame-level tracking GT, regardless of occlusion. The remainder are recorded as scope exclusions rather than tracker false negatives; tracker false negatives are unmatched eligible GT boxes. For each 10-second clip, annotators review 20 frames sampled at 2 Hz. To keep manual correction tractable, the current evaluation uses 20 sampled clips from six intersections within the fixed ROI and threeview coverage scope. All intersections use the same tracking parameters; only calibration-dependent voxel projections and site-specific search regions are reconfigured. This clip-based protocol limits the scale of the present evaluation rather than the temporal duration supported by the tracker.

Table 7: Split-policy ablation with Qwen2.5-VL-7B. Held and Clip denote intersection-held-out and clip-random train/test splitting, respectively. All denotes their respective test sets (4,691 and 4,359 QA pairs); Com. denotes the same 1,451 QA pairs absent from both training sets.
<table><tr><td colspan="6">Eval. Split S-MCQ D-MCQ Det-F1 Line-F1 T-IoU C-ADE↓</td></tr><tr><td>All Held All Clip</td><td>.782 .857</td><td>.781 .806</td><td>.643 .754</td><td>.860 .944</td><td>.381 74.8 .379 62.0</td></tr><tr><td>Com. Held</td><td>.772</td><td>.775 .574</td><td>.846</td><td>.399</td><td>71.0</td></tr><tr><td>Com. Clip</td><td>.841</td><td>.796</td><td>.654</td><td>.896 .420</td><td>57.0</td></tr></table>

![](images/f25558e794314b2e0e32ca40afcc25068b3a942e355caa88a9d24ab355349bcf.jpg)  
Figure 5: Human review interface for initialized 3D tracks. Multi-view projections and the semantic-voxel BEV guide manual correction.

Table 8: Statistics of the human-reviewed subset. Dimensions are adjusted once per track; a frame requires no framewise correction when its XY center and heading remain unchanged.
<table><tr><td>Metric</td><td>Value</td><td>Metric</td><td>Value</td></tr><tr><td>Generated boxes</td><td>3,240</td><td>GT boxes</td><td>3,317</td></tr><tr><td>GT tracks</td><td>251</td><td>Scope-excl. inst.</td><td>1,158</td></tr><tr><td>Exclusion rate</td><td>25.9%</td><td>No frame-wise edit</td><td>1,900 (57.3%)</td></tr></table>

Table 9: 3D tracking quality at a BEV IoU matching threshold of 0.5. MOTP is the mean BEV IoU over valid matches.
<table><tr><td>Metric</td><td>Value</td><td>Metric</td><td>Value</td></tr><tr><td>MOTA ↑</td><td>66.9</td><td>MOTPBEV ↑</td><td>87.0</td></tr><tr><td>IDS↓</td><td>40</td><td>F1↑</td><td>83.9</td></tr><tr><td>Precision / Recall ↑</td><td>84.9 / 82.9</td><td>XY err. (m) ↓</td><td>.071</td></tr></table>

Prediction–GT association is determined geometrically rather than by inherited track IDs, allowing detection errors, localization errors, and identity consistency to be evaluated separately. Specifically, we perform class-aware frame-wise Hungarian matching using oriented BEV IoU; matches with BEV IoU below 0.5 are rejected, and the remaining unmatched predictions and eligible GT boxes are counted as FP and FN, respectively.

Figure 6 shows representative projected cuboids across roadside sequences. The peripheral vehicle omitted near an overlap boundary falls outside the predefined three-view evaluation scope rather than being counted as a tracker false negative.

![](images/281ff3ba62c2930526b65cb8ccbf0bab3925bc318d99c1d7d2731778e0c0e503.jpg)

![](images/e4fe48f9db3d6a6c6bfbdb31a065978cfb0ff8373325af90330859616906fd0a.jpg)  
Figure 6: Projected cuboids from the multi-view 3D tracker across representative roadside sequences.

## Conclusion

We presented RISE, a framework for roadside sequence understanding through two complementary tasks: metric 3D tracking and structured vision-language reasoning. By exploiting complementary spatial and temporal evidence, RISE supports persistent 3D tracking and bbox-grounded prediction without exposing future evidence to evaluated models. Human-corrected evaluation on 20 clips yields 66.9 MOTA for the generated tracks within the defined multi-view scope. Experiments on RISE-Bench demonstrate clear benefits from domain adaptation and generally from temporal context, while revealing remaining challenges in spatial grounding, future localization, and interaction reasoning. These findings highlight both the value and the current limitations of sequence-level roadside understanding.

Several limitations remain. Image-only 3D tracking depends on accurate calibration, reliable segmentation, and sufficient camera overlap, with weaker geometric support under heavy occlusion and near shared-view boundaries. On the vision-language side, the current 3.8-second clips primarily assess near-term reasoning from individual roadside views. Extending the observation horizon without sacrificing temporal resolution, together with coordinated reasoning across synchronized views, would enable the benchmark to capture longer and more complex trafic evolution. A promising direction is to use persistent metric 3D trajectories as explicit spatiotemporal grounding for vision-language models, enabling more reliable reasoning about long-term motion and multi-agent interactions.

## References

Carion, N.; Gustafson, L.; Hu, Y.-T.; Debnath, S.; Hu, R.; Suris, D.; Ryali, C.; Alwala, K. V.; Khedr, H.; Huang, A.; et al. 2025. Sam 3: Segment anything with concepts. arXiv preprint arXiv:2511.16719.

Choudhary, T.; Dewangan, V.; Chandhok, S.; Priyadarshan, S.; Jain, A.; Singh, A. K.; Srivastava, S.; Jatavallabhula, K. M.; and Krishna, K. M. 2024. Talk2bev: Languageenhanced bird’s-eye view maps for autonomous driving. In 2024 IEEE International Conference on Robotics and Automation (ICRA), 16345–16352. IEEE.

Gloudemans, D.; Work, D.; Wang, Y.; Gumm, G. E.; and Barbour, W. 2023. The Interstate-24 3D Dataset: A New Bench-

mark for 3D Multi-Camera Vehicle Tracking. In British Machine Vision Conference (BMVC).

Guan, R.; Hu, R.; Chen, S.; Xiao, N.; Xia, X.; Liu, J.; Chen, B.; Tang, Z.; Ouyang, N.; Liang, S.; et al. 2026. Roadscenevqa: Benchmarking visual question answering in roadside perception systems for intelligent transportation system. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, 4366–4375.

Hao, R.; Fan, S.; Dai, Y.; Zhang, Z.; Li, C.; Wang, Y.; Yu, H.; Yang, W.; Yuan, J.; and Nie, Z. 2024. Rcooper: A real-world large-scale dataset for roadside cooperative perception. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, 22347–22357.

Huang, W.; Zhang, S.; Chua, C.; Liang, Y.; Mao, Z.; Yang, H.; and Lv, C. 2026. Towards Safe Mobility: A Unified Transportation Foundation Model enabled by Open-Ended Vision-Language Dataset. arXiv preprint arXiv:2604.22260.

Marcu, A.-M.; Chen, L.; Hünermann, J.; Karnsund, A.; Hanotte, B.; Chidananda, P.; Nair, S.; Badrinarayanan, V.; Kendall, A.; Shotton, J.; et al. 2024. Lingoqa: Visual question answering for autonomous driving. In European Conference on Computer Vision, 252–269. Springer.

Qian, T.; Chen, J.; Zhuo, L.; Jiao, Y.; and Jiang, Y.-G. 2024. Nuscenes-qa: A multi-modal visual question answering benchmark for autonomous driving scenario. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, 4542–4550.

Sekaran, K. C.; Geisler, M.; Rößle, D.; Mohan, A.; Cremers, D.; Utschick, W.; Botsch, M.; Huber, W.; and Schön, T. 2025. UrbanIng-V2X: A Large-Scale Multi-Vehicle, Multi-Infrastructure Dataset Across Multiple Intersections for Cooperative Perception. In The Thirty-ninth Annual Conference on Neural Information Processing Systems Datasets and Benchmarks Track.

Sima, C.; Renz, K.; Chitta, K.; Chen, L.; Zhang, H.; Xie, C.; Beißwenger, J.; Luo, P.; Geiger, A.; and Li, H. 2024. Drivelm: Driving with graph visual question answering. In European conference on computer vision, 256–274. Springer.

Xu, L.; Huang, H.; and Liu, J. 2021. Sutd-traficqa: A question answering benchmark and an eficient network for video reasoning over trafic events. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 9878–9888.

Yang, L.; Yu, K.; Tang, T.; Li, J.; Yuan, K.; Wang, L.; Zhang, X.; and Chen, P. 2023. Bevheight: A robust framework for vision-based roadside 3d object detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 21611–21620.

Yang, L.; Zhang, X.; Yu, J.; Li, J.; Zhao, T.; Wang, L.; Huang, Y.; Zhang, C.; Wang, H.; and Li, Y. 2024. MonoGAE: Roadside monocular 3D object detection with ground-aware embeddings. IEEE Transactions on Intelligent Transportation Systems, 25(11): 17587–17601.

Ye, X.; Shu, M.; Li, H.; Shi, Y.; Li, Y.; Wang, G.; Tan, X.; and Ding, E. 2022. Rope3d: The roadside perception dataset for autonomous driving and monocular 3d object detection task.

In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 21341–21350.

You, J.; Li, P.; Jiang, Z.; Tang, W.; Huang, Z.; Gan, R.; Liu, J.; Zhao, Y.; Chen, S.; and Ran, B. 2026. V2x-qa: A comprehensive reasoning dataset and benchmark for multimodal large language models in autonomous driving across ego, infrastructure, and cooperative views. arXiv preprint arXiv:2604.02710.

Yu, H.; Luo, Y.; Shu, M.; Huo, Y.; Yang, Z.; Shi, Y.; Guo, Z.; Li, H.; Hu, X.; Yuan, J.; et al. 2022. Dair-v2x: A largescale dataset for vehicle-infrastructure cooperative 3d object detection. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 21361–21370.

Yu, H.; Yang, W.; Ruan, H.; Yang, Z.; Tang, Y.; Gao, X.; Hao, X.; Shi, Y.; Pan, Y.; Sun, N.; et al. 2023. V2x-seq: A largescale sequential dataset for vehicle-infrastructure cooperative perception and forecasting. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, 5486–5495.

Zhang, Y.; Zheng, Z.; Liu, J.; Huang, Z.; Zhou, Z.; Meng, Z.; Cai, T.; and Ma, J. 2025. MIC-BEV: Multi-Infrastructure Camera Bird’s-Eye-View Transformer with Relation-Aware Fusion for 3D Object Detection. arXiv preprint arXiv:2510.24688.

Zheng, Y.; Zhang, R.; Zhang, J.; Ye, Y.; and Luo, Z. 2024. Llamafactory: Unified eficient fine-tuning of 100+ language models. In Proceedings of the 62nd annual meeting of the association for computational linguistics (volume 3: system demonstrations), 400–410.

Zhou, X.; Larintzakis, K.; Guo, H.; Zimmer, W.; Liu, M.; Cao, H.; Zhang, J.; Lakshminarasimhan, V.; Strand, L.; and Knoll, A. 2025. TUMTraf VideoQA: Dataset and Benchmark for Unified Spatio-Temporal Video Understanding in Trafic Scenes. In Forty-second International Conference on Machine Learning.