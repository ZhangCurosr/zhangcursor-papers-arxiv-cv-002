# Pointing-VLA: Typed Spatial Grounding Interfaces for Vision-Language-Action Manipulation

Xiwen Chen<sup>1,2,\*</sup>, Zelin Li<sup>1,\*,†</sup>, Zhiruo Zhou<sup>1,3</sup>, Huiming Chen<sup>4</sup> Chenwei Wang<sup>5</sup>, Xiaojun Zhu<sup>1,†</sup>

<sup>1</sup>SIGS, Tsinghua University, China; <sup>2</sup>Shanghai Jiao Tong University, China; <sup>3</sup>Wuhan University of Technology, China; <sup>4</sup>City University of Hong Kong, Hong Kong (China SAR); <sup>5</sup>AiDlab, Hong Kong (China SAR);

<sup>\*</sup>Equal Contributions; <sup>†</sup>Correspondence: chiellini.lee@gmail.com; zhu.xiaojun@sz.tsinghua.edu.cn

## Abstract

Vision-language-action (VLA) models often expose spatial grounding through autoregressive text coordinates or opaque action tokens, creating brittle interfaces between multimodal reasoning and robot execution. We present Pointing-VLA, a typed hidden-state spatial readout built on Embodied-R1. Geometryspecific heads predict normalized points, objectfunctional grounding (OFG) heatmaps, and visual trajectories without serializing geometry as text. For the evaluated Bridge/WidowX and physical pick-place deployments, an explicit execution contract assigns PICK to source-conditioned OFG and PLACE to Pointing, providing direct stage-aligned spatial targets. Pointing-VLA achieves SOTA performance on Bridge/WidowX, averaging 72.9% across the evaluated four-task set without Bridge-specific finetuning under collision-enabled CuRobo execution. Pointing and OFG show complementary strengths across native and cross-dataset evaluations. The OFG/contact readout transfers to NORA-1.5, preserving or improving success while reducing recorded controller time by more than 20×; typed heads are also 6.68–6.90× faster than Embodied-R1 text decoding on a shared external suite. When integrated as spatial guidance for a π<sub>0.5</sub> action policy, Pointing-VLA raises autonomous realrobot success from 52.7% to 80.7% across three visual contexts. These results establish typed spatial readouts as an eficient, inspectable interface between embodied reasoning and robot execution.

## Introduction

Large vision-language-action (VLA) models can connect observations, language instructions, and robot behavior, but their output interface remains a bottleneck. Many manipulation subtasks first require an explicit spatial commitment: the point to touch, the functional part to use, or the short visual trace an executor should follow. Autoregressive text coordinates are convenient for prompting, yet they make geometry an afterthought: the model must spend tokens spelling out numbers, the evaluator must parse strings back into pixels, and dense part-level evidence is discarded before execution. Direct action tokens avoid parsing, but they also hide the spatial rationale that a robot stack or human auditor may need to inspect.

Figure 1 separates two interface failures observed in representative simulator cases: text generation can fail before producing valid geometry, and a syntactically valid coordinate can still be spatially wrong. Parsing is therefore an additional execution dependency rather than a guarantee of correct grounding.

The central claim of this paper is that embodied grounding should be treated as typed interface prediction rather than text-coordinate generation. Referring points, functional regions, and visual trajectories are related but not identical targets: a point is sparse, an afordance is often a dense part-level region, and a trajectory is temporally structured. Collapsing them into one output distribution can create negative interference even when they share a visual-language backbone, while serializing all of them as text makes downstream execution slower and less stable.

This formulation is grounded in the vision-for-action and population-readout view of computation: downstream readouts decode spatial variables from distributed internal activity instead of serializing them through the system that represents them (Goodale and Milner 1992; Andersen and Buneo 2002; Cisek 2007). Pointing-VLA realizes this principle with separate point, afordance-heatmap, and trajectory heads that emit each target in the geometry consumed by robot-side execution.

We propose Pointing-VLA, a lightweight typed grounding framework built on an embodied visionlanguage backbone. Rather than asking the backbone to say coordinates, Pointing-VLA reads spatial intent from multimodal hidden states. A compact adapter and learned queries feed point, object-functional grounding (OFG), and visual trajectory generation (VTG) decoders. Deployment contracts bind each structured execution slot to the geometry it requires; the primary pick-place scafold uses source-conditioned OFG for PICK and Pointing for PLACE. The resulting spatial targets connect embodied reasoning to downstream wrappers, planners, and executors through deterministic, stage-aligned contracts. Experiments evaluate the

![](images/b89c1d2c8a8a6887b79911feff1e86815e9c448759a5eaa0548f4c00d0da5910.jpg)  
real simulator scene  
Actual output parsed: False first point: None output chars: 4053  
No parseable coordinate reaches the robot.

![](images/8025b82842a44751eb71e02a07bb340e0970f74b73cbfe715823bbb1c01c9cdf.jpg)  
real simulator scene  
Actual output parsed: True first point: [222, 172] inside target: False  
Parseable, but geometrically wrong.

Figure 1: Representative text-coordinate failures. Left: generation yields no parseable coordinate. Right: [222, 172] is parseable but lies outside the target region.

interface from native spatial grounding through head specialization, cross-backbone transfer, and robot deployment. On Bridge/WidowX, the fixed OFG PICK and Pointing PLACE composition achieves SOTA performance, averaging 72.9% across the evaluated four-task set without Bridge-specific finetuning under collision-enabled CuRobo execution. Cross-dataset results reveal complementary Pointing and OFG geometries, while NORA-1.5 transfer preserves or improves success with more than 20× shorter recorded controller time. Real-robot deployment completes the evidence chain from hidden-state readout to physical execution.

Our contributions are:

• We formulate embodied grounding as typed interface prediction, replacing serialized text coordinates with geometry-specific robot-facing readouts.

• We introduce Pointing, OFG, and VTG hidden-state readouts with explicit execution contracts, including a fixed source-conditioned OFG-PICK/Pointing-PLACE deployment that aligns each stage with its required geometry.

• We validate the interface through native and crossdataset grounding, runtime and cross-backbone transfer, Bridge/WidowX evaluation, and autonomous real-robot deployment.

## Related Work

Generalist robot policies and VLA systems adapt large pretrained models to robot action spaces, from embodied multimodal models and Robotics Transformers to cross-embodiment datasets and open robot policies (Driess et al. 2023; Brohan et al. 2023; Zitkovich et al. 2023; Open X-Embodiment Collaboration et al. 2024; Octo Model Team et al. 2024; Kim et al. 2025; Black et al. 2025). This line of work has improved policy learning, action parameterization, and inference throughput, but it mainly asks how perception and language should become actions. Pointing-VLA asks a complementary interface question: what spatial representation should be exposed before final execution?

Spatial structure has long been useful for manipulation. Neuroscience distinguishes object recognition from action-facing visuomotor representations, includ ing posterior-parietal intention maps and competing afordance-like action candidates (Goodale and Milner 1992; Andersen and Buneo 2002; Cisek 2007). Robot learning makes a compatible engineering move: Transporter Networks preserve spatial feature maps for pickand-place displacements, CLIPort combines semantic “what” and spatial “where” pathways, and PerAct discretizes 3D voxel actions for language-conditioned manipulation (Zeng et al. 2021; Shridhar, Manuelli, and Fox 2022, 2023). Recent grounding-centric VLA work extends this idea to embodied reasoning: Robo-Point predicts spatial afordance keypoints from language instructions (Yuan et al. 2025a), FSD studies seeing-to-doing afordance grounding (Yuan et al. 2026), and Embodied-R1 formalizes embodied pointing across REG, relational-region grounding, OFG, and VTG (Yuan et al. 2025b). Afordance datasets such as AGD20K further emphasize that actionable regions are often object parts rather than whole-object centers (Luo et al. 2022). These systems show that spatial outputs can bridge VLM reasoning and robot behavior, but they leave open which interface an executor should consume. Pointing-VLA studies this interface choice directly: sparse referring points, dense functional heatmaps, and visual waypoint traces are decoded in their native geometries rather than serialized as text coordinates.

This interface complements action-sequence policies such as Difusion Policy and ACT (Chi et al. 2023; Zhao et al. 2023) by exposing typed spatial targets before low-level action generation. In the evaluated pick-place systems, the execution phase itself supplies a stable contract: functional contact is decoded for PICK and a compact target point is decoded for PLACE.

## Method

## Hidden-State Spatial Readout

Pointing-VLA operationalizes typed interface prediction by decoding geometry-specific robot targets directly from shared multimodal hidden states. Figure 2 summarizes this hidden-state readout and its executionfacing spatial outputs.

![](images/93c582dbabe47f2d9e91870b26ed1d361627e94dd54a6075017e1625f280761d.jpg)  
Figure 2: Pointing-VLA architecture and typed interfaces. Embodied-R1 hidden states feed point, OFG/contactheatmap, and VTG-waypoint heads; fixed PICK/PLACE slots use source-conditioned OFG/Pointing, and a wrapper maps image geometry to robot coordinates.

Let $e \in \{ \mathrm { p t } , \mathrm { o f g } , \mathrm { v t g } \}$ index a spatial expert and let $H _ { \mathrm { m m } } ^ { e } = \{ h _ { i } ^ { \vartriangle } \} _ { i = 1 } ^ { N }$ denote its Embodied-R1 multimodal states. Pointing-VLA reuses one backbone architecture, base weights, and Grounding Adapter body, while each specialist retains its own LoRA, learned queries, and decoder state. For the point and OFG experts, the shared adapter body projects the sequence into a grounding space and reads it with expert-specific task-conditioned queries:

$$
\begin{array} { r l } & { X _ { e } = W _ { h } ^ { e } H _ { \mathrm { m m } } ^ { e } + P _ { e } , } \\ & { A _ { e } = \mathrm { s o f t m a x } \left( \frac { ( Q _ { e } W _ { Q } ^ { e } ) ( X _ { e } W _ { K } ^ { e } ) ^ { \top } } { \sqrt { d } } \right) , } \\ & { G _ { e } = \mathrm { F F N } _ { e } ( A _ { e } X _ { e } W _ { V } ^ { e } ) , \qquad g _ { e } = \mathrm { P o o l } ( G _ { e } ) , } \end{array}
$$

where $P _ { e }$ supplies sequence position, $G _ { e }$ preserves a fixed set of grounded tokens, and $g _ { e }$ is its global summary. The point and OFG experts use eight grounding queries. The strongest VTG expert instead lets learned temporal queries attend directly to the full multimodal sequence, avoiding a fixed grounding-token bottleneck.

## Spatial Interfaces and Deployment Contracts

Learned Spatial Heads. The Pointing decoder uses a learned pointer query $q _ { \mathrm { p t } }$ to select the grounded evidence needed for one image location:

$$
\begin{array} { r l } & { A _ { \mathrm { p t } } = \mathrm { s o f t m a x } \left( \frac { ( q _ { \mathrm { p t } } W _ { Q } ) ( G _ { \mathrm { p t } } W _ { K } ) ^ { \top } } { \sqrt { d } } \right) , } \\ & { ~ z _ { \mathrm { p t } } = A _ { \mathrm { p t } } ( G _ { \mathrm { p t } } W _ { V } ) , \qquad ( \hat { x } , \hat { y } ) = \sigma ( W _ { p } \mathrm { L N } ( z _ { \mathrm { p t } } ) + b _ { p } ) . } \end{array}
$$

Thus $( \hat { x } , \hat { y } ) \in [ 0 , 1 ] ^ { 2 }$ is emitted directly by the learned head, without generating or parsing coordinate text. For an image of width W and height H, the correspond ing pixel is

$$
( \hat { u } , \hat { v } ) = \big ( ( W - 1 ) \hat { x } , ( H - 1 ) \hat { y } \big ) .
$$

The prediction is evaluated with point-in-bbox (PIB) or point-in-mask (PIM).

OFG preserves spatial extent instead of collapsing the source to one token. Let $F _ { \mathrm { v i s } }$ be the visual tower’s 2D feature map. The global grounded summary modulates

![](images/0d13781cea95c250a65fedd96c6fcc2d40242065ba268b0d990ba1b11cd0e8ba.jpg)  
Figure 3: Stage-wise training. Geometry warm-up trains the adapter, eight queries, and geometry head; LoRA Same model family; stage-specific update scopes.specialization updates only low-rank backbone parameters; task continuation updates only the required subset. Snowflakes and flames mark frozen and trainable modules.

this map through FiLM before a convolutional heatmap decoder (Luo et al. 2022):

$$
\begin{array} { r l } & { \quad F _ { \mathrm { o f g } } ^ { \prime } = \gamma ( g _ { \mathrm { o f g } } ) \odot F _ { \mathrm { v i s } } + \beta ( g _ { \mathrm { o f g } } ) , } \\ & { \quad \hat { H } _ { \mathrm { o f g } } = f _ { \mathrm { h m } } ( F _ { \mathrm { o f g } } ^ { \prime } ) , } \\ & { \quad ( i ^ { * } , j ^ { * } ) = \arg \operatorname* { m a x } _ { i , j } \hat { H } _ { \mathrm { o f g } } [ i , j ] , } \\ & { \quad \quad \hat { p } _ { \mathrm { o f g } } = \left( \frac { j ^ { * } } { W _ { o } - 1 } , \frac { i ^ { * } } { H _ { o } - 1 } \right) . } \end{array}
$$

Here $\hat { H } _ { \mathrm { o f g } }$ contains heatmap logits, and its normalized peak is the execution-facing functional contact.

For VTG, learned waypoint queries $\boldsymbol { q } _ { t } ^ { \tau }$ with temporal encodings attend to the multimodal sequence in order:

$$
\begin{array} { r l } & { z _ { t } ^ { \tau } = \mathrm { A t t n } ( q _ { t } ^ { \tau } , H _ { \mathrm { m m } } ^ { \mathrm { v t g } } ) , } \\ & { \hat { \tau } _ { t } = \sigma ( W _ { \tau } z _ { t } ^ { \tau } + b _ { \tau } ) , \qquad t = 1 , \ldots , T . } \end{array}
$$

We use $T = 8$ normalized image-space waypoints and report RMSE, average displacement error (ADE), and final displacement error (FDE). These waypoints describe a visual trace rather than low-level robot actions.

Structured PICK/PLACE Contracts. Training follows target geometry: point, region, and visual-trace samples supervise the corresponding learned heads. For pick-place deployment, the scafold constructs slotspecific queries from instruction $\ell ,$ source description ${ \mathit { c } } _ { \mathrm { s r c } } ,$ and goal description $c _ { \mathrm { g o a l } }$ <sub>l</sub>:

$$
\begin{array} { r c l } { q _ { \mathrm { p i c k } } } & { = } & { \mathcal { T } _ { \mathrm { p i c k } } ( \ell , c _ { \mathrm { s r c } } ) , } \\ { q _ { \mathrm { p l a c e } } } & { = } & { \mathcal { T } _ { \mathrm { p l a c e } } ( \ell , c _ { \mathrm { g o a l } } ) , } \\ { \hat { y } _ { \mathrm { p i c k } } } & { = } & { F _ { \mathrm { o f g } } ( H _ { \mathrm { m m } } , G , q _ { \mathrm { p i c k } } ) , } \\ { \hat { y } _ { \mathrm { p l a c e } } } & { = } & { F _ { \mathrm { p t } } ( H _ { \mathrm { m m } } , G , q _ { \mathrm { p l a c e } } ) . } \end{array}
$$

This assignment is deterministic: source-conditioned OFG emits the functional PICK contact, while Pointing emits the PLACE target. VTG serves annotated waypoint requests; the reported pick-place deployments use the fixed OFG-PICK/Pointing-PLACE contract. An external wrapper converts image-space outputs to robot-frame targets. With calibrated depth $D ,$ camera intrinsics $K ,$ and camera-to-robot transform $T _ { r  c } ,$ this boundary can be written as

$$
\begin{array} { r } { \tilde { p } = [ \hat { u } , \hat { v } , 1 ] ^ { \top } , ~ } \\ { X _ { c } = D ( \hat { u } , \hat { v } ) K ^ { - 1 } \tilde { p } , ~ } \\ { \bar { X } _ { r } = T _ { r  c } \bar { X } _ { c } , ~ } \end{array}
$$

where $\bar { X }$ denotes homogeneous coordinates. The existing executor then consumes $X _ { r } ;$ the learned heads remain image-space predictors.

## Training Objective

The experts use a weighted spatial objective:

$$
\begin{array} { r } { \mathcal { L } = \lambda _ { \mathrm { p t } } \mathcal { L } _ { \mathrm { p t } } + \lambda _ { \mathrm { b o x } } \mathcal { L } _ { \mathrm { b o x } } + \lambda _ { \mathrm { o f g } } \mathcal { L } _ { \mathrm { o f g } } + \lambda _ { \mathrm { v t g } } \mathcal { L } _ { \mathrm { v t g } } + \lambda _ { \mathrm { a u x } } \mathcal { L } _ { \mathrm { a u x } } . } \end{array}
$$

Point and trajectory coordinates use SmoothL1, while OFG mask or keypoint supervision is converted to a dense target $H ^ { * }$ :

$$
\begin{array} { r c l } { \mathcal { L } _ { \mathrm { p t } } } & { = } & { \frac { 1 } { K } \sum _ { k } \mathrm { S m o o t h L 1 } ( \hat { p } _ { k } , p _ { k } ) , } \\ { \mathcal { L } _ { \mathrm { o f g } } } & { = } & { \frac { 1 } { | \Omega | } \sum _ { u \in \Omega } \ell _ { \mathrm { h m } } ( \hat { H } _ { \mathrm { o f g } } ( u ) , H ^ { * } ( u ) ) , } \\ { \mathcal { L } _ { \mathrm { v t g } } } & { = } & { \frac { 1 } { T } \sum _ { t } \mathrm { S m o o t h L 1 } ( \hat { \tau } _ { t } , \tau _ { t } ) . } \end{array}
$$

Here $\ell _ { \mathrm { h m } }$ is mean-squared error for Gaussian keypoint targets and class-balanced binary cross-entropy with logits for dense afordance masks. Language-modeling and legacy reconstruction terms are inactive; optimization focuses on adapters, heads, and LoRA specialization (Hu et al. 2021).

For source-conditioned OFG specialization, we keep the Embodied-R1 base, Pointing/VTG query and decoder modules, and executor frozen while jointly updating the OFG LoRA, shared Grounding Adapter body, OFG learned query, and heatmap decoder. Dense source-mask reconstruction is combined with sourceover-distractor ranking and distractor suppression. This adaptation teaches OFG to preserve functional contact while resolving the source instance directly in the PICK readout.

The final pointing model additionally applies GRPOstyle group-normalized refinement. With policy center $\mu = \hat { p }$ and learned $\sigma = \exp ( s )$ , samples are

$$
a _ { i } = \mathrm { c l i p } ( \mu + \sigma \epsilon _ { i } , 0 , 1 ) , \qquad \epsilon _ { i } \sim \mathcal { N } ( 0 , I ) .
$$

For target box $B ^ { * }$ , reward and group-normalized advantage are

$$
\begin{array} { r l r } { r _ { i } } & { = } & { \left\{ \begin{array} { l l } { 1 , } & { a _ { i } \in B ^ { * } , } \\ { 0 . 5 \operatorname* { m a x } ( 0 , 1 - d _ { i } / \rho ) , } & { a _ { i } \notin B ^ { * } , } \end{array} \right. } \\ { A _ { i } } & { = } & { \frac { r _ { i } - \operatorname* { m e a n } _ { j } ( r _ { j } ) } { \operatorname* { s t d } _ { j } ( r _ { j } ) + \epsilon } . } \end{array}
$$

With $\pi _ { i } = \pi ( a _ { i } \mid \mu , \sigma )$ and Gaussian reference KL $D _ { \mathrm { r e f } }$ the objective is

$$
\begin{array} { r c l } { { \mathcal { L } _ { \mathrm { G R P O } } } } & { { = } } & { { - \frac { 1 } { M } \sum _ { i = 1 } ^ { M } \mathrm { s g } ( A _ { i } ) \log \pi _ { i } } } \\ { { } } & { { } } & { { + \lambda _ { \mathrm { s f t } } \mathrm { S m o o t h L 1 } ( \mu , p ^ { * } ) } } \\ { { } } & { { } } & { { + \lambda _ { \mathrm { k l } } D _ { \mathrm { r e f } } + \lambda _ { \sigma } { \mathcal L } _ { \sigma } . } } \end{array}
$$

Here sg stops advantage gradients and $p ^ { * }$ is the supervised point target; refinement acts directly on the continuous spatial head.

## Experiments

## Evaluation Questions and Protocol

Four questions structure the evaluation.

Q1: Geometry. Do typed heads preserve taskappropriate geometry?

Q2: Phase specialization. Does source-conditioned OFG resolve PICK source selection while preserving the benefits of a fixed OFG-PICK/Pointing-PLACE contract?

Q3: Transfer and eficiency. Does the readout transfer across datasets and VLA backbones while reducing inference cost?

Q4: Deployment. Do the gains persist in simulation and real-robot deployment?

Native and external evaluations answer Q1; sourceconditioning and interface-composition ablations answer Q2. These controlled comparisons isolate component-level grounding quality, while the deployment studies measure complete robot-task success. Cross-dataset and cross-backbone transfer plus knownhead runtime address Q3; controlled simulation and real-robot studies on shared task sets and executor families address Q4.

Evaluation Protocol and Metrics. We evaluate each output at the level of the interface it is designed to serve. Pointing and OFG predictions are converted to image coordinates and scored by point-in-mask (PIM) or point-in-box (PIB) sample success; VTG is measured by normalized trajectory root mean squared error (RMSE), average displacement error (ADE), and final displacement error (FDE). Robot studies require full task completion. Within each experiment, all compared methods use the same samples or episode identities, ensuring that measured diferences arise from the spatial interface or adaptation strategy. Experiments used 40- GB NVIDIA A100 and dual 48-GB RTX A6000 GPUs; complete hardware, software, and seed records appear in the Supplement.

## Native Spatial-Head Results

Table 1 answers the first question. Pointing-VLA matches Embodied-R1 on sparse referring-point grounding at 64.3%, while OFG raises full Part-Afordance-2K accuracy from 40.9% to 57.3%. This 16.4-point improvement demonstrates geometry specialization: Pointing preserves sparse target accuracy, whereas OFG substantially strengthens part-level functional grounding by retaining the spatial extent required for contact prediction.

<table><tr><td>Model</td><td>VABench-P Part-Afford.2K</td><td>acc. ↑</td></tr><tr><td>GPT-4o</td><td>acc. ↑ 9.3</td><td>10.2</td></tr><tr><td>ASMv2</td><td>10.1</td><td>13.8</td></tr><tr><td>RoboBrain</td><td>7.0</td><td>25.3</td></tr><tr><td>Qwen2.5-VL</td><td>9.9</td><td>23.4</td></tr><tr><td>RoboPoint (Yuan et al. 2025a)</td><td>19.1</td><td>27.6</td></tr><tr><td>FSD (Yuan et al. 2026)</td><td>61.8</td><td>9.6</td></tr><tr><td>Embodied-SFT (Yuan et al. 2025b)</td><td>50.5</td><td>39.4</td></tr><tr><td>Embodied-R1 (Yuan et al. 2025b)</td><td>63.7</td><td>40.9</td></tr><tr><td>Pointing-VLA (ours)</td><td>64.3</td><td>57.3</td></tr></table>

Table 1: Native point-in-region grounding accuracy (%). Pointing-VLA uses its Pointing head for VABench-P and its OFG head for the full Part-Afordance-2K run.

Sequential geometry. VTG completes the native evaluation for ordered spatial outputs. On 300 VABench-V examples, the trajectory readout records 0.1042 RMSE, 0.1368 ADE, and 0.1493 FDE in normalized image coordinates. These measures evaluate the waypoint sequence and its terminal target rather than collapsing trajectory quality into a single pointin-region score.

## Mechanism and Fixed-Contract Ablations

PICK requires both source-instance disambiguation and a contact location within the selected object. Source-conditioned OFG preserves this dense functional extent and exposes its peak as the grasp cue. PLACE instead requires a compact target in the destination region, for which Pointing provides a direct normalized coordinate. The fixed OFG-PICK/Pointing-PLACE contract therefore aligns each execution stage with the geometry it requires, without introducing an inference-time selector.

Mechanism analysis. Panel (a) identifies the effective capacity regime of the spatial readout. Eight queries and rank-32 LoRA produce the strongest PIB within their respective sweeps, while GRPO refinement raises PIB from 61.3% to 64.3%. The non-monotonic query and rank trends demonstrate that performance is governed by the spatial-readout design rather than parameter count alone.

Fixed-contract validation. Panel (b) validates the phase-aligned assignment: the deployed contract completes 70/96 episodes (72.9%) under collision-enabled CuRobo execution, outperforming the strongest alternative composition by 14.6 percentage points while completing all Stack episodes and 22/24 Eggplant episodes.

The controlled frozen multicolor Stack evaluation further verifies this mechanism. Source-conditioned OFG selects the instructed source in all 48 online cases and completes 43/48 episodes, versus 40/48 for Attention PICK under identical Pointing PLACE targets. Perfect source selection establishes OFG as a direct PICK readout that resolves the instructed source while preserving functional contact grounding.

## External Diagnostics and Cross-Backbone Transfer

Cross-Dataset Geometry. Table 3 asks whether performance follows the geometry exposed by each interface. These diagnostics standardize outputs as points and score PIM/PIB, isolating spatial grounding from downstream control. Pointing is strongest on RefCOCO and RoboAf, whereas OFG is strongest on AGD20K and raises full Part-Afordance-2K PIM from 40.9% with Embodied-R1 text generation to 57.3%. The crossover supports expert-specific output geometries rather than one serialized coordinate interface. The Supplement provides qualitative OFG heatmaps.

Pointing, OFG, and Embodied-R1 text generation are evaluated on the same cross-dataset samples. Each trained head emits one point, while text generation can emit multiple points; sample-level success therefore keeps the comparison independent of candidate count. Dataset composition and deterministic split construction are detailed in the Supplement and released evaluation code. This shared point protocol enables a controlled cross-dataset comparison of spatial interfaces under identical output and evaluation rules.

<table><tr><td>Suite</td><td>Metric</td><td>Pointing-VLA Point</td><td>Pointing-VLA OFG</td><td>Embodied-R1 Text Generation</td></tr><tr><td>RefCOCO</td><td>PIM</td><td>40.7%</td><td>22.2%</td><td>10.6%</td></tr><tr><td>RefCOCO</td><td>PIB</td><td>56.1%</td><td>36.0%</td><td>25.9%</td></tr><tr><td>AGD20K</td><td>PIM</td><td>53.9%</td><td>64.4%</td><td>39.9%</td></tr><tr><td>Part-Aff.2K</td><td>PIM</td><td></td><td>57.3%</td><td>40.9%</td></tr><tr><td>RoboAff.</td><td>PIM</td><td>11.5%</td><td>9.8%</td><td>2.4%</td></tr></table>

Table 3: Cross-dataset PIM/PIB accuracy (%).
<table><tr><td>System</td><td>Total (s)</td><td>Per sample (s)</td><td>Speedup</td></tr><tr><td>OFG head</td><td>1,274.61</td><td>0.434</td><td>6.90×</td></tr><tr><td>Pointing head</td><td>1,316.29</td><td>0.448</td><td>6.68×</td></tr><tr><td>Embodied-R1 text decoder</td><td>8,798.06</td><td>2.993</td><td>1.00×</td></tr><tr><td>Method</td><td></td><td>Success Time (s)</td><td>Speedup</td></tr><tr><td>NORA base</td><td>Horizontal</td><td>100.0</td><td>107.11</td></tr><tr><td>+ OFG/contact</td><td>Horizontal</td><td>4.85 111.57</td><td>1.0× 22.1×</td></tr><tr><td>NORA base</td><td>Laid vert.</td><td>100.0 89.0</td><td>1.0×</td></tr><tr><td>+ OFG/contact</td><td>Laid vert.</td><td>95.0</td><td>5.44 20.5×</td></tr></table>

Table 4: Known-head runtime (top) and NORA-1.5 transfer (bottom).

Interface-Level Runtime. Table 4 measures inference from multimodal hidden states to typed spatial outputs under the shared external protocol. The geometric readouts are 6.68–6.90× faster than autoregressive text decoding, establishing the interface-level efficiency gained by eliminating coordinate serialization, token-by-token generation, and post-hoc parsing.

NORA-1.5 Transfer. Table 4 shows that the frozen NORA-1.5 backbone with the transferred OFG/contact readout preserves horizontal-pose success, improves laid-vertical success from 89.0% to 95.0%, and cuts recorded controller time by more than 20× under the shared wrapper.
<table><tr><td>System</td><td>Spoon</td><td>Carrot</td><td>Stack</td><td>Eggpl.</td><td>Avg.</td></tr><tr><td>Octo-S</td><td>47.2</td><td>9.7</td><td>4.2</td><td>56.9</td><td>30.0</td></tr><tr><td>SpatialVLA (FT)</td><td>16.7</td><td>25.0</td><td>29.2</td><td>100.0</td><td>42.7</td></tr><tr><td>SoFar</td><td>55.5</td><td>56.9</td><td>62.5</td><td>40.2</td><td>53.8</td></tr><tr><td>MemoryVLA</td><td>75.0</td><td>75.0</td><td>37.5</td><td>100.0</td><td>71.9</td></tr><tr><td>Embodied-R1 + CuRobo</td><td>20.8</td><td>45.8</td><td>62.5</td><td>79.2</td><td>52.1</td></tr><tr><td>Pointing-VLA + CuRobo</td><td>50.0</td><td>50.0</td><td>100.0</td><td>91.7</td><td>72.9</td></tr></table>

Table 5: Bridge/WidowX success (%; 24 episodes/task for Embodied-R1 and Pointing-VLA). Pointing-VLA uses fixed OFG-PICK/Pointing-PLACE with collisionenabled CuRobo.

## Deployment Studies

Table 5 evaluates four Bridge/WidowX manipulation structures. The Pointing-VLA condition contains 24 episodes per task, and success requires complete task execution. Its fixed OFG PICK and Pointing PLACE composition reaches 72.9% with collision checking active in every rollout; CuRobo completes all planned motion segments without planner failure or fallback. The tasks stress distinct spatial bottlenecks: thin-object contact for Spoon, surface targeting for Carrot, sourceinstance disambiguation for Stack, and container placement for Eggplant. The source-conditioned OFG expert preserves 100.0% Stack success as the direct PICK readout, while Pointing supplies an explicit placement target. Together, these results demonstrate reliable four-task deployment through a collision-aware motionplanning backend.

(a) Spatial-head configuration
<table><tr><td>Component Settings</td><td></td><td>PIB</td></tr><tr><td>Query tokens 4 /</td><td>8/ 16</td><td>44.7 / 50.3 / 45.7</td></tr><tr><td>LoRA rank</td><td>none45.3 / 47.3 / 53.3 / 50.3 / 16 /</td><td></td></tr><tr><td></td><td>32 /</td><td></td></tr><tr><td>Refinement</td><td>64 SFT /</td><td>61.3 / 64.3</td></tr></table>

(b) Interface-composition ablation
<table><tr><td>Configuration</td><td>Spoon on towel</td><td>Carrot on plate</td><td>Stack cube</td><td>Eggplant basket</td><td>Avg.</td></tr><tr><td>Attention → Attention</td><td>16.7</td><td>41.7</td><td>100.0</td><td>62.5</td><td>55.2</td></tr><tr><td>Attention → Pointing</td><td>12.5</td><td>50.0</td><td>100.0</td><td>70.8</td><td>58.3</td></tr><tr><td>OFG → Attention</td><td>16.7</td><td>54.2</td><td>16.7</td><td>58.3</td><td>36.5</td></tr><tr><td>OFG → Pointing</td><td>50.0</td><td>50.0</td><td>100.0</td><td>91.7</td><td>72.9</td></tr></table>

Table 2: Mechanism and fixed-contract results. (a) VABench-P PIB (%) across spatial-head configurations. (b) Bridge/WidowX success (%) for fixed PICK/PLACE interfaces (24 episodes/task); the final row is the deployed OFG-PICK/Pointing-PLACE contract with collision-enabled CuRobo.  
![](images/b4aa58280cee71268137a82952432eaa92c3439e42472aa8da56e31636243d4f.jpg)  
Figure 4: Representative successful real-robot rollouts across three visual contexts, each ordered from initialization to stable release.

Real-Robot Deployment
<table><tr><td>Scene success</td><td> $\pi _ { 0 . 5 }$ </td><td> $\pi _ { 0 . 5 } + \mathrm { P } \mathrm { - } \mathrm { V } \mathrm { L A }$ </td><td>Δ</td></tr><tr><td>No distractor</td><td>20/50 (40.0)</td><td>36/50 (72.0)</td><td>+32.0</td></tr><tr><td>Yellow cylinder</td><td>26/50 (52.0)</td><td>42/50 (84.0)</td><td>+32.0</td></tr><tr><td>Red cylinder</td><td>33/50 (66.0)</td><td>43/50 (86.0)</td><td>+20.0</td></tr><tr><td>All three scenes</td><td>79/150 (52.7)</td><td>121/150 (80.7)</td><td>+28.0</td></tr><tr><td>Failure stage</td><td>Grasp</td><td>Transfer Tray</td><td>Uncertain</td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td>Count (/150) ∆ (pp)</td><td>47→16 -20.7</td><td>8→9 13→4 +0.7 -6.0</td><td>3→0 -2.0</td></tr></table>

Table 6: Autonomous PiPER outcomes. Success is count/trials (%); failures show baseline→P-VLA for pre-lift grasp, post-grasp transfer/drop, tray arrival without upright release, and uncertain cases.

We deploy Pointing-VLA on an AgileX PiPER and compare it with a π baseline (Physical Intelligence et al. 2025) through the same dual-camera perceptionto-control stack. Pointing-VLA retains π as the action-generating policy and inserts typed spatial guidance before execution. During PICK, the OFG head predicts an afordance heatmap over the lower rectangular part of the white composite object, and the external wrapper converts its peak into a robot-frame grasp cue. During PLACE, the Pointing head predicts a normalized image-space placement coordinate inside the green tray, which the wrapper converts into the placement target. The same deterministic OFG-PICK/Pointing-PLACE contract is used throughout all three visual contexts: no distractor, a yellow-cylinder distractor, and a red-cylinder distractor. We evaluate both systems over 50 trials per context. The reported endpoint is full task completion: the designated part must be grasped and lifted, positioned upright in the tray, and stably released.

Across all three visual contexts, typed spatial guidance consistently improves autonomous completion over the action-policy baseline. Failure-stage analysis confirms that the gains concentrate at the intended control boundaries: pre-lift grasp failures fall from 47 to 16, and tray-arrival failures fall from 13 to 4. Post-grasp transfer remains the primary residual bottleneck. These reductions verify the stage-aligned contribution of OFG-PICK and Pointing-PLACE guidance; Figure 4 traces representative successes from approach through stable release.

## Discussion and Conclusion

Pointing-VLA treats referring points, functional regions, and waypoint traces as distinct robot-facing geometries. Embodied-R1 provides shared multimodal states, while lightweight geometry heads expose normalized points, heatmaps, and trajectories directly. Explicit deployment contracts align these outputs with execution stages and preserve the geometry required by each robot-facing slot. Together, the native, crossdataset, runtime, transfer, and deployment results establish geometry-appropriate typed readouts as an eficient and inspectable interface between embodied VLM reasoning and robot execution.

The fixed pick/place scafold provides a controlled deployment setting that verifies the efectiveness of typed spatial guidance. Task-dependent variation, most visible on spoon-on-towel, identifies closed-loop geometryaware execution as the next leverage point. Future work will extend this interface with visual correction, broader structured task decompositions, and cross-robot transfer.

## References

Andersen, R. A.; and Buneo, C. A. 2002. Intentional Maps in Posterior Parietal Cortex. Annual Review of Neuroscience, 25: 189–220.

Black, K.; Brown, N.; Driess, D.; Esmail, A.; Equi, M.; Finn, C.; Fusai, N.; Groom, L.; Hausman, K.; Ichter, B.; Jakubczak, S.; Jones, T.; Ke, L.; Levine, S.; Li-Bell, A.; Mothukuri, M.; Nair, S.; Pertsch, K.; Shi, L. X.; Tanner, J.; Vuong, Q.; Walling, A.; Wang, H.; and Zhilinsky, U. 2025. π : A Vision-Language-Action Flow Model for General Robot Control. In Proceedings of Robotics: Science and Systems.

Brohan, A.; Brown, N.; Carbajal, J.; Chebotar, Y.; Dabis, J.; Finn, C.; Gopalakrishnan, K.; Hausman, K.; Herzog, A.; Hsu, J.; Ibarz, J.; Ichter, B.; Irpan, A.; Jackson, T.; Jesmonth, S.; Joshi, N.; Julian, R.; Kalashnikov, D.; Kuang, Y.; Leal, I.; Lee, K.-H.; Levine, S.; Lu, Y.; Malla, U.; Manjunath, D.; Mordatch, I.; Nachum, O.; Parada, C.; Peralta, J.; Perez, E.; Pertsch, K.; Quiambao, J.; Rao, K.; Ryoo, M.; Salazar, G.; Sanketi, P.; Sayed, K.; Singh, J.; Sontakke, S.; Stone, A.; Tan, C.; Tran, H.; Vanhoucke, V.; Vega, S.; Vuong, Q.; Xia, F.; Xiao, T.; Xu, P.; Xu, S.; Yu, T.; and Zitkovich, B. 2023. RT-1: Robotics Transformer for Real-World Control at Scale. In Proceedings of Robotics: Science and Systems.

Chi, C.; Xu, Z.; Feng, S.; Cousineau, E.; Du, Y.; Burchfiel, B.; Tedrake, R.; and Song, S. 2023. Difusion Policy: Visuomotor Policy Learning via Action Difusion. In Robotics: Science and Systems.

Cisek, P. 2007. Cortical Mechanisms of Action Selection: The Afordance Competition Hypothesis. Philosophical Transactions of the Royal Society B: Biological Sciences, 362(1485): 1585–1599.

Driess, D.; Xia, F.; Sajjadi, M. S. M.; Lynch, C.; Chowdhery, A.; Ichter, B.; Wahid, A.; Tompson, J.; Vuong, Q.; Yu, T.; Huang, W.; Chebotar, Y.; Sermanet, P.; Duckworth, D.; Levine, S.; Vanhoucke, V.; Hausman, K.; Toussaint, M.; Gref, K.; Zeng, A.; Mordatch, I.; and Florence, P. 2023. PaLM-E: An Embodied Multimodal Language Model. In Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, 8469–8488. PMLR.

Goodale, M. A.; and Milner, A. D. 1992. Separate Visual Pathways for Perception and Action. Trends in Neurosciences, 15(1): 20–25.

Hu, E. J.; Shen, Y.; Wallis, P.; Allen-Zhu, Z.; Li, Y.; Wang, S.; Wang, L.; and Chen, W. 2021. LoRA: Low-Rank Adaptation of Large Language Models. arXiv:2106.09685.

Kim, M. J.; Pertsch, K.; Karamcheti, S.; Xiao, T.; Balakrishna, A.; Nair, S.; Rafailov, R.; Foster, E.; Lam, G.; Sanketi, P.; Vuong, Q.; Kollar, T.; Burchfiel, B.; Tedrake, R.; Sadigh, D.; Levine, S.; Liang, P.; and Finn, C. 2025. OpenVLA: An Open-Source Vision-Language-Action Model. In Proceedings of The 8th Conference on Robot Learning, volume 270 of Proceedings of Machine Learning Research, 2679–2713. PMLR.

Luo, H.; Zhai, W.; Zhang, J.; Cao, Y.; and Tao, D. 2022. Learning Afordance Grounding from Exocentric Im-

ages. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2252–2261.

Octo Model Team; Ghosh, D.; Walke, H.; Pertsch, K.; Black, K.; Mees, O.; Dasari, S.; Hejna, J.; Kreiman, T.; Xu, C.; Luo, J.; Tan, Y. L.; Chen, L. Y.; Sanketi, P.; Vuong, Q.; Xiao, T.; Sadigh, D.; Finn, C.; and Levine, S. 2024. Octo: An Open-Source Generalist Robot Policy. In Robotics: Science and Systems.

Open X-Embodiment Collaboration; O’Neill, A.; Rehman, A.; Gupta, A.; Maddukuri, A.; Gupta, A.; Padalkar, A.; Lee, A.; Pooley, A.; Gupta, A.; et al. 2024. Open X-Embodiment: Robotic Learning Datasets and RT-X Models. In Proceedings of the IEEE International Conference on Robotics and Automation.

Physical Intelligence; Black, K.; Brown, N.; Darpinian, J.; Dhabalia, K.; Driess, D.; Esmail, A.; Equi, M.; Finn, C.; Fusai, N.; Galliker, M. Y.; Ghosh, D.; Groom, L.; Hausman, K.; Ichter, B.; Jakubczak, S.; Jones, T.; Ke, L.; LeBlanc, D.; Levine, S.; Li-Bell, A.; Mothukuri, M.; Nair, S.; Pertsch, K.; Ren, A. Z.; Shi, L. X.; Smith, L.; Springenberg, J. T.; Stachowicz, K.; Tanner, J.; Vuong, Q.; Walke, H.; Walling, A.; Wang, H.; Yu, L.; and Zhilinsky, U. 2025. π<sub>0.5</sub>: A Vision-Language-Action Model with Open-World Generalization. arXiv:2504.16054.

Shridhar, M.; Manuelli, L.; and Fox, D. 2022. CLIPort: What and Where Pathways for Robotic Manipulation. In Proceedings of the 5th Conference on Robot Learning, volume 164 of Proceedings of Machine Learning Research, 894–906. PMLR.

Shridhar, M.; Manuelli, L.; and Fox, D. 2023. Perceiver-Actor: A Multi-Task Transformer for Robotic Manipulation. In Proceedings of the 6th Conference on Robot Learning, volume 205 of Proceedings of Machine Learning Research, 785–799. PMLR.

Yuan, W.; Duan, J.; Blukis, V.; Pumacay, W.; Krishna, R.; Murali, A.; Mousavian, A.; and Fox, D. 2025a. RoboPoint: A Vision-Language Model for Spatial Afordance Prediction in Robotics. In Proceedings of The 8th Conference on Robot Learning, volume 270 of Proceedings of Machine Learning Research, 4005–4020. PMLR.

Yuan, Y.; Cui, H.; Chen, Y.; Dong, Z.; Ni, F.; Kou, L.; Liu, J.; Li, P.; Zheng, Y.; and Hao, J. 2026. From Seeing to Doing: Bridging Reasoning and Decision for Robotic Manipulation. arXiv:2505.08548.

Yuan, Y.; Cui, H.; Huang, Y.; Chen, Y.; Ni, F.; Dong, Z.; Li, P.; Zheng, Y.; Tang, H.; and Hao, J. 2025b. Embodied-R1: Reinforced Embodied Reasoning for General Robotic Manipulation. arXiv:2508.13998.

Zeng, A.; Florence, P.; Tompson, J.; Welker, S.; Chien, J.; Attarian, M.; Armstrong, T.; Krasin, I.; Duong, D.; Sindhwani, V.; and Lee, J. 2021. Transporter Networks: Rearranging the Visual World for Robotic Manipulation. In Proceedings of the 4th Conference on Robot Learning, volume 155 of Proceedings of Machine Learning Research, 726–747. PMLR.

Zhao, T. Z.; Kumar, V.; Levine, S.; and Finn, C. 2023. Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware. In Proceedings of Robotics: Science and Systems.

Zitkovich, B.; Yu, T.; Xu, S.; Xu, P.; Xiao, T.; Xia, F.; Wu, J.; Wohlhart, P.; Welker, S.; Wahid, A.; Vuong, Q.; Vanhoucke, V.; Tran, H.; Soricut, R.; Singh, A.; Singh, J.; Sermanet, P.; Sanketi, P.; Salazar, G.; Ryoo, M.; Reymann, K.; Rao, K.; Pertsch, K.; Mordatch, I.; Michalewski, H.; Lu, Y.; Levine, S.; Lee, T.-W. E.; Lee, L.; Leal, I.; Kuang, Y.; Kalashnikov, D.; Julian, R.; Joshi, N.; Irpan, A.; Ichter, B.; Hsu, J.; Herzog, A.; Hausman, K.; Gopalakrishnan, K.; Fu, C.; Florence, P.; Finn, C.; Dubey, A.; Driess, D.; Ding, T.; Choroman ski, K.; Chen, X.; Chebotar, Y.; Carbajal, J.; Brown, N.; Brohan, A.; Arenas, M. G.; and Han, K. 2023. RT-2: Vision-Language-Action Models Transfer Web Knowl edge to Robotic Control. In Proceedings of the 7th Conference on Robot Learning, volume 229 of Proceedings of Machine Learning Research, 2165–2183. PMLR.