# RailSyn: Diagnosis-Guided Image Generation for Traceable Data Completion in Railway Foreign Object Detection

Quan Hao<sup>1</sup>, Chenxi Zhang<sup>1</sup>, Ziyang Tao<sup>2</sup>, Yuyuan Zhou<sup>3,4</sup>, Yudong Wang<sup>6</sup>, Rui Shi<sup>1</sup>, Lechuan Xu<sup>5</sup>, Liu Changhao<sup>1</sup>, Liguo Zhang

<sup>1</sup>The School of Information Science and Technology, Beijing University of Technology

<sup>2</sup>Pratt School of Engineering, Duke University

<sup>3</sup>Academy of Mathematics and Systems Science, Chinese Academy of Sciences

<sup>4</sup>School of Mathematical Sciences, University of Chinese Academy of Sciences

<sup>5</sup>Industrial Systems Engineering and Management, National University of Singapore

<sup>6</sup>Institute of Automation, Chinese Academy of Sciences

haoquan@emails.bjut.edu.cn, zcx20041004@emails.bjut.edu.cn, ziyang.tao@duke.edu, zhouyuyuan@amss.ac.cn, luke.xu@u.nus.edu, yudong.wang@ia.ac.cn, ruishi@bjut.edu.cn, liu20050126@emails.bjut.edu.cn, zhangliguo@bjut.edu.cn

## Abstract

Railway foreign object detection (RFOD) is critical to safe railway operation, yet scarce real positive samples incompletely represent task-relevant variations in object scale, intrusion relation, railway scene, illumination, and adverse weather. Existing synthetic augmentation can improve RFOD detection, but its gains lack an explicit account of the task-relevant deficiencies complemented by the generated data. We therefore introduce RailSyn, a diagnosis-guided framework comprising a real-referenced Inspector and a requirement-aligned Generator. The Inspector constructs a variable-radius empirical cover from finite real observations to localize candidate completion regions and profile synthetic pools. The resulting audit identifies railway-context, intrusion-semantic, and visual-consistency requirements; the Generator addresses them through domain adaptation, agent-planned placement and physical contact relations, and plan-consistent conditional refinement. Using the Inspector, we further trace representation-space changes across generation variants; the complete system attains a local-shell occupation of $C _ { \mathrm { g a p } }$ to 13.64%, which measures generated coverage of real-derived completion regions. Extensive experiments show AP50– 95 gains of up to 4.9 points and consistent improvements across nine mainstream detectors, demonstrating broad crossarchitecture utility.

## Introduction

Railway foreign object detection (RFOD) is critical to the safe operation of high-speed rail systems (Hao et al. 2026; Chen et al. 2024). Reliable detection requires examples that jointly represent railway structures, environmental conditions, object scales, and the physical relations by which foreign objects intrude into operational space (Hao et al. 2026; Chen et al. 2024, 2025; Oza et al. 2024). Such positive samples are scarce and costly to annotate, while small targets further concentrate task evidence in limited image regions (Hao et al. 2026; Chen et al. 2024; Cheng et al. 2023; Shi et al. 2025). Consequently, finite RFOD datasets incompletely capture not only visual appearance, but also the context, intrusion semantics, and local object–background efects that determine whether synthetic samples are useful for detection.

Existing augmentation and generation methods produce additional training samples (Hao et al. 2026; Feng et al. 2024; Wang et al. 2024). Conventional operations are confined to observed objects and contexts (Ghiasi et al. 2021; Trabucco et al. 2024), whereas advanced approaches such as SVDDD, RegionDifusion, and broader grounded generation frameworks prioritize realism, diversity, or controllability (Wang, Pan, and Wen 2025; Li, Luo, and Tseng 2025; Li et al. 2023). However, none of these objectives explicitly identify which task-relevant regions remain weakly complemented relative to real RFOD observations; consequently, method selection is decoupled from a concrete completion demand. As a result, generated pools may repeat existing patterns, misalign object–scene intrusion semantics, or exhibit anomalous injection efects (Somepalli et al. 2023; Wu et al. 2024; Song et al. 2024). Their detection gains are likewise dificult to attribute: global similarity metrics are dominated by railway backgrounds, while AP provides a standardized measure of overall detector performance but still falls short of characterizing the contextual, relational, or localized contributions introduced by generated data (Jayasumana et al. 2024; Ghosh, Hajishirzi, and Schmidt 2023; Hu et al. 2023; Lee et al. 2023; Bolya et al. 2020). The outstanding gap, therefore, is a unified framework that diagnoses deficient information prior to synthesis, translates that evidence into explicit generation requirements, and evaluates whether the resulting pool efectively complements the specified regions.

We address this gap with RailSyn, an Inspector–Generator framework in which inspection explicitly guides generation. The Inspector builds a variable-radius empirical cover from our collected high-quality real RFOD reference and evaluates synthetic pools using reliable-support coverage $( C _ { \mathrm { r e l } } )$ local-gap completion $( C _ { \mathrm { g a p } } ) _ { \mathrm { : } }$ , and nonredundant volume efficiency $( \eta _ { \mathrm { v o l } } )$ . It applies the real-derived local-shell criterion underlying $C _ { \mathrm { g a p } }$ to the SD-based(Rombach et al. 2022) RFOD23 AIGC pool and retrieves representative samples from the high-uncertainty regions induced by the finite real reference, followed by inspection of nearby RFOD23 samples. As shown on the left of Figure 1, these samples reveal gaps in railway-background fidelity, object–scene intrusion semantics, and anomalous edge efects around injected objects. Guided by this diagnosis, the Generator aligns one process with each gap: railway-domain low-rank adaptation prepares structural backgrounds (Hu et al. 2022); a multimodal Agent produces executable background-conditioned placement and physical-contact plans; and plan-consistent conditional refinement translates spatial and relation metadata into a unified control condition with mask-aware composition. RailSyn thus turns Inspector-localized deficiencies into explicit generation requirements for complementing the diagnosed regions and improving detection.

![](images/5535a9e91ef31cc82ba27348e8a4a53275c117016a8f5b2eb0fd033b75534baa.jpg)  
Figure 1: RailSyn overview. (a) Inspector identifies high-uncertainty regions from the finite real reference and inspects nearby SD-based RFOD23 AIGC samples, revealing three recurring patterns. (b) Generator addresses them through railway scene preparation, relation-aware intrusion planning, and intrusion-pattern refinement. Right: generated samples and feature-space completion $( C _ { \mathrm { g a p } } )$ .

RailSyn data improve AP50–95 across all nine tested mainstream detectors, with an average best observed gain of 3.03 points and a maximum gain of 4.9 points. We further evaluate matched Generator ablations with DEIM and YOLO11, and reapply the Inspector to every ablated pool to trace representation-space changes. The complete Generator attains $C _ { \mathrm { g a p } } = 1 3 . { \dot { 6 } } 4 \%$ , and the Inspector diagnostics exhibit trends broadly consistent with the AP changes in the ablation study. This descriptive alignment links detector gains with greater occupation of the real-derived completion regions without treating $C _ { \mathrm { g a p } }$ as an AP predictor. Externalpool experiments further distinguish empirical completion from downstream utilization.

Our main contributions are:

• We introduce RailSyn, an Inspector–Generator framework that connects real-referenced deficiency analysis, guided synthesis, and representation-space assessment for traceable RFOD data completion.

• We develop a diagnosis-guided Generator that aligns railway-domain adaptation, Agent-planned intrusion semantics, and plan-consistent conditional refinement with the context, relation, and boundary gaps exposed by the Inspector.

• We demonstrate cross-architecture utility across nine heterogeneous detectors and conduct controlled DEIM/YOLO11 ablations with Inspector profiles to trace component-level efects on detection performance and local-gap completion.

## Related Work

## Railway Foreign-Object Detection and Data Scarcity

Railway foreign object detection (RFOD) localizes debris and intrusions on or near railway tracks (Hao et al. 2026; Chen et al. 2024), where missed objects can obstruct operation and threaten safety (Hao et al. 2026; Chen et al. 2024). Current detectors mainly follow YOLO-style dense prediction or DETR-style end-to-end set prediction (Redmon et al. 2016; Carion et al. 2020): YOLO prioritizes eficient multi-scale detection (Chen et al. 2025), whereas DETR exploits global context through transformer-based matching (Carion et al. 2020). Despite this progress, RFOD remains a small-sample problem: rare intrusions and costly acquisition limit positive examples, weakening detector learning across scale, illumination, weather, and intrusion relations (Hao et al. 2026; Chen et al. 2024; Cheng et al. 2023; Oza et al. 2024; Kang et al. 2019). Data-centric augmentation and generation therefore ofer a practical route to supplement scarce RFOD training samples (Feng et al. 2024; Wang et al. 2024; Islam et al. 2024). Conventional transformations and copy–paste reuse observed content and provide limited control over scale, placement, and intrusion relations (Ghiasi et al. 2021; Trabucco et al. 2024), motivating more flexible generative augmentation for the small-sample setting.

## Generative Augmentation and Traceable Completion

Latent difusion enables high-resolution synthesis in compressed spaces and supports data augmentation under limited observations (Rombach et al. 2022; Trabucco et al. 2024; Islam et al. 2024). More recently, flow matching and FLUX have established a rectified-flow Transformer paradigm for high-quality generation (Lipman et al. 2023; Esser et al. 2024); spatial controls improve placement and structure preservation (Li et al. 2023; Zhang, Rao, and Agrawala 2023; Jia et al. 2024), while low-rank adaptation enables eficient domain specialization (Hu et al. 2022). These advances expand the potential of controllable generation for small-sample detection, complementing few-shot learning, rebalancing, domain adaptation, active selection, and focal reweighting (Kang et al. 2019; Lin et al. 2017; Oza et al. 2024; Wan et al. 2024). However, realism and controllability do not establish whether generated data complement detectionrelevant information (Ghosh, Hajishirzi, and Schmidt 2023; Hu et al. 2023; Lee et al. 2023; Huang et al. 2025a), especially for rare RFOD scenes, scales, weather, and intrusion relations (Hao et al. 2026; Chen et al. 2024; Cheng et al. 2023; Oza et al. 2024). Existing evaluation ofers limited traceability from generation to detection accuracy (Jayasumana et al. 2024; Lee et al. 2023; Bolya et al. 2020): global distribution metrics such as FID and KID can be dominated by railway backgrounds and underweight the small foreign-object regions that carry detection evidence (Jayasumana et al. 2024; Cheng et al. 2023; Ghosh, Hajishirzi, and Schmidt 2023), whereas AP aggregates utility without identifying the contributions of scene coverage, object attributes, intrusion relations, or detector optimization (Bolya et al. 2020; Huang et al. 2025a). The remaining gap is therefore traceable generation that links observed data deficiencies, generation choices, synthetic-pool completion, and downstream detection gains.

## Method

RailSyn formulates synthetic RFOD augmentation as the traceable completion of a finite real training set. The Inspector constructs real-referenced empirical completion regions, retrieves existing synthetic candidates for analysis, and profiles the generated pools. The Generator then fulfills the resulting railway-context, intrusion-relation, and visualintegration requirements through three dedicated generation modules. We first define completion quality and real-scene detection utility as separate evaluation targets, and subsequently present the Inspector and the requirement-aligned Generator.

## Problem Formulation

Let $\mathcal { D } _ { R } = \{ x _ { i } \} _ { i = 1 } ^ { N }$ denote a finite real RFOD training set, $\mathcal { D } _ { G } = \{ g _ { j } \} _ { j = 1 } ^ { M }$ a generated pool, and $\mathcal { D } _ { V }$ a disjoint realscene evaluation set. We study whether $\mathcal { D } _ { G }$ complements task-relevant information insuficiently represented by ${ \mathcal { D } } _ { R } ,$ such that a detector trained on $\mathcal { D } _ { R } \cup \mathcal { D } _ { G }$ generalizes better to $\mathcal { D } _ { V }$ than one trained on $\mathcal { D } _ { R }$ alone. This objective entails two distinct questions: what information $\mathcal { D } _ { G }$ contributes relative to the finite real observations, and whether a downstream detector can exploit that contribution. Accordingly, we assess information completion with the Inspector and real-scene utility with detector AP under a fixed training and evaluation protocol.

## Inspector: Real-Referenced Data Completion Analysis

The Inspector instantiates the completion objective in a taskrelevant representation space. We adopt the recent large-scale Qwen3-VL-Embedding-8B (Li et al. 2026) as the primary encoder because its visual-semantic modeling capacity and 4096-dimensional representation support fine-grained distinctions among railway context, foreign-object appearance, and object–scene intrusion relations. CLIP and DINOv2 serve as established auxiliary encoders for cross-encoder validation. Within each encoder, ℓ<sub>2</sub>-normalized embeddings are analyzed with spherical distance in the complete native space.

To estimate spatially varying support, let $R = \{ z _ { i } \} _ { i = } ^ { N }$ denote the normalized real embeddings. We assign each anchor a radius given by its spherical distance to a fixed-order nearest real neighbor, producing a cover that contracts in densely observed regions and expands across sparse ones. A local intrinsic dimension estimated from real–real neighborhoods then specifies the corresponding volume law while the original embedding coordinates remain intact. This construction captures the nonuniform geometry of finite RFOD observations and provides the real-only reference for subsequent completion analysis.

The anchor radii characterize observed neighborhoods but leave the regions between real samples unresolved. We therefore augment the anchor set with deterministic spherical midpoints of edges connecting neighboring real samples and fit a real-only Matérn-5/2 Gaussian process to the log-radius field, $f ( \mathbf { z } ) = \log r ( \dot { \mathbf { z } } )$ . For each query point q, the posterior mean $\mu _ { f } ( q )$ estimates the local reach, while the posterior standard deviation $\sigma _ { f } ( q )$ quantifies interpolation uncer-

tainty:

$$
\begin{array} { r l } & { \mu _ { f } ( q ) = \bar { f } + k _ { q } ^ { \top } \big ( K + \sigma _ { n } ^ { 2 } I \big ) ^ { - 1 } \cdot \big ( f - \bar { f } \big ) , } \\ & { \sigma _ { f } ^ { 2 } ( q ) = K ( q , q ) - k _ { q } ^ { \top } \big ( K + \sigma _ { n } ^ { 2 } I \big ) ^ { - 1 } \cdot k _ { q } , } \\ & { r ^ { - } ( q ) = \exp \bigl ( \mu _ { f } ( q ) - \tau \sigma _ { f } ( q ) \bigr ) , } \\ & { \quad \widehat { r } ( q ) = \exp \bigl ( \mu _ { f } ( q ) \bigr ) , } \\ & { r ^ { + } ( q ) = \exp \bigl ( \mu _ { f } ( q ) + \tau \sigma _ { f } ( q ) \bigr ) . } \end{array}\tag{1}
$$

The Gram matrix K is generated by the positive-definite Matérn-5/2 kernel:

$$
\begin{array} { c } { { \displaystyle K ( { \bf z } , { \bf z } ^ { \prime } ) = \sigma _ { f } ^ { 2 } \left( 1 + \xi + \frac { \xi ^ { 2 } } { 3 } \right) e ^ { - \xi } } , } \\ { { \displaystyle \xi = \frac { \sqrt { 5 } d _ { c } ( { \bf z } , { \bf z } ^ { \prime } ) } { \ell } , } } \\ { { \displaystyle d _ { c } ( { \bf z } , { \bf z } ^ { \prime } ) = \| { \bf z } - { \bf z } ^ { \prime } \| _ { 2 } } . }  \end{array}\tag{2}
$$

The interpolated radii define a lower reliable support and an upper plausible boundary, while the interval between them exposes localized completion opportunities induced by finite observations. We retain these intervals as query-indexed shells so that overlap between neighboring balls does not erase local demand. After freezing the real construction, each generated sample receives a radius from the posterior mean field:

$$
\begin{array} { r l } & {  { \boldsymbol { S } } ^ { - } = \bigcup _ { \boldsymbol { q } \in \boldsymbol { Q } }  { \boldsymbol { B } } _ { \mathbb { S } } ( \boldsymbol { q } , \boldsymbol { r } ^ { - } ( \boldsymbol { q } ) ) , } \\ & {  { \boldsymbol { S } } ^ { + } = \bigcup _ { \boldsymbol { q } \in \boldsymbol { Q } }  { \boldsymbol { B } } _ { \mathbb { S } } ( \boldsymbol { q } , \boldsymbol { r } ^ { + } ( \boldsymbol { q } ) ) , } \\ & {  { \boldsymbol { \mathcal { G } } } _ { \boldsymbol { q } } =  { \boldsymbol { B } } _ { \mathbb { S } } ( \boldsymbol { q } , \boldsymbol { r } ^ { + } ( \boldsymbol { q } ) ) \setminus  { \boldsymbol { B } } _ { \mathbb { S } } ( \boldsymbol { q } , \boldsymbol { r } ^ { - } ( \boldsymbol { q } ) ) , } \\ & {  { \boldsymbol { S } } _ { \boldsymbol { G } } = \bigcup _ { \boldsymbol { q } \in \boldsymbol { G } }  { \boldsymbol { B } } _ { \mathbb { S } } ( \boldsymbol { g } , \boldsymbol { \widehat { r } } ( \boldsymbol { g } ) ) . } \end{array}\tag{3}
$$

Together, $S ^ { - }$ , the indexed family $\{ { \mathcal { G } } _ { q } \} _ { q \in Q }$ , and $\mathit { S } _ { \mathit { G } }$ provide the set relations needed to assess how generated data retain established support, occupy real-derived gaps, and expand without redundancy.

Let $\mu _ { \widehat { m } }$ denote the local mass induced by the real-only intrinsic-dimensional volume law. The completion profile is defined as

$$
\begin{array} { r l } & { C _ { \mathrm { r e l } } = \displaystyle \frac { \mu _ { \widehat { m } } ( S _ { G } \cap S ^ { - } ) } { \mu _ { \widehat { m } } ( S ^ { - } ) } , } \\ & { C _ { \mathrm { g a p } } = \displaystyle \frac { \sum _ { q \in Q } \mu _ { \widehat { m } } ( S _ { G } \cap \mathcal { G } _ { q } ) } { \sum _ { q \in Q } \mu _ { \widehat { m } } ( \mathcal { G } _ { q } ) } , } \\ & { \eta _ { \mathrm { v o l } } = \displaystyle \frac { \mu _ { \widehat { m } } ( S _ { G } ) } { \sum _ { g \in G } \mu _ { \widehat { m } } ( \mathcal { B } _ { \mathrm { S } } ( g , \widehat { r } ( g ) ) ) } . } \end{array}\tag{4}
$$

Here $C _ { \mathrm { r e l } }$ quantifies reliable-support reach, $C _ { \mathrm { g a p } }$ quantifies the mass-weighted occupation of indexed local shells, and $\eta _ { \mathrm { v o l } }$ quantifies generated-cover nonredundancy. Since exact integration over high-dimensional ball unions is intractable, we estimate these masses by native-spherical Monte Carlo: samples are drawn from the local tangent volume law, mapped back to the unit sphere, and weighted by inverse cover multiplicity to avoid double-counting overlaps.

We apply the Inspector to the real reference set to identify regions where finite observations induce the largest empirical uncertainty. For each query $q ,$ the corresponding shell is ranked by the intrinsic-dimensional mass proxy

$$
g ( q ) \propto { \left( r ^ { + } ( q ) \right) } ^ { \widehat { m } } - { \left( r ^ { - } ( q ) \right) } ^ { \widehat { m } } .\tag{5}
$$

High-mass shells define the audit regions, around which spherical nearest-neighbor retrieval collects a real reference and SD-based RFOD23 AIGC candidates. This real-only ranking converts abstract uncertainty regions into directly inspectable image groups while keeping the audit independent of downstream generation choices.

We use the Inspector to analyze high-uncertainty regions induced by the finite real reference, followed by inspection of nearby RFOD23 samples. As shown in Figure 1(a), the first representative group contains disordered backgrounds that bear little correspondence to authentic railway scenes, indicating a pronounced railway-context deficiency. In the second group, foreign objects occupy implausible positions relative to the track infrastructure, revealing a deficiency in intrusion semantics. In the third group, inserted objects exhibit abnormally sharp contours, conflicting with the weak object–background contrast typical of small foreign objects and exposing a deficiency in intrusion appearance. These three observations motivate the Generator’s scene preparation, relation-aware planning, and boundary-aware injection, respectively.

## Generator: Targeted Synthetic Data Generation

The Generator translates the three diagnosed deficiencies into a coupled synthesis process. Railway scene preparation addresses context deficiency by injecting railway-scene and foreign-object priors; relation-aware intrusion planning addresses semantic deficiency by conditioning placement on scene geometry and physical relations; and diagnosed intrusion-pattern refinement reconciles object contours with local background appearance. These stages preserve a oneto-one correspondence between Inspector diagnosis and targeted synthesis.

Railway Scene Preparation The context deficiency requires both recognizable railway environments and faithful foreign-object appearance. We therefore learn two low-rank adaptations (LoRAs) (Hu et al. 2022; Black Forest Labs 2024): a scene adapter captures track layout, viewpoint, surrounding environment, and illumination, while an object adapter captures category-specific shape, texture, and appearance. During synthesis, the two adapters inject complementary scene and object features into the generation process, providing domain-specific visual priors for high-quality railway composition before spatial and relational constraints are imposed.

Relation-Aware Intrusion Planning The semantic deficiency arises when foreign-object category and position are specified independently of the selected railway scene. Given a prepared background $I _ { b }$ and a candidate object, a multimodal Agent produces a structured intrusion plan

$$
\mathcal { V } = A ( I _ { b } , I _ { o } ) = \{ c , b _ { o } , r , u , \theta \} ,\tag{6}
$$

where c is the object category, $b _ { o }$ its placement box, r its physical relation to railway infrastructure, u the associated contact region, and θ an applicable tilt. The plan explicitly couples placement with rails, sleepers, ballast, or catenary and is deterministically normalized to valid image coordinates and scale ranges. Its accepted box initializes the detector annotation, while its relation and contact fields are propagated to conditional realization. Thus, the Agent converts background–object compatibility into an executable interface between scene preparation and image synthesis.

Diagnosed Intrusion-Pattern Refinement The diagnosed intrusion-pattern deficiency requires the generated boundary to remain consistent with the planned physical intrusion mode and surrounding railway structure. We first transform the object and its mask according to Y and form a softenedmask composite $I _ { f }$ . The structured intrusion conditions are then fused into a single control map

$$
C _ { \mathcal { V } } = \mathrm { F u s e } ( C _ { \mathrm { e d g e } } ( I _ { f } ) , C _ { \mathrm { b o x } } ( b _ { o } ) , C _ { \mathrm { c o n t a c t } } ( r , u ) ) ,\tag{7}
$$

which jointly encodes background geometry, planned object extent, and relation-specific contact evidence. Let $H _ { s }$ denote the denoising feature at scale s and $Z _ { s }$ a zero-initialized $1 \times 1$ projection. Intrusion-pattern conditioning is injected at four spatial scales as

$$
\widetilde { H } _ { s } = H _ { s } + \alpha _ { s } Z _ { s } ( \Phi _ { s } ( C \boldsymbol { y } ) )\tag{8}
$$

where $\Phi _ { s }$ extracts scale-specific condition features and $\alpha _ { s }$ controls their contribution. Zero initialization preserves the pretrained generation path at the outset, while multiscale fusion transfers the diagnosed intrusion pattern from global object extent to local contact and contour structure. A final softened alpha composition maintains object visibility while suppressing anomalous boundaries that are inconsistent with the planned intrusion mode.

## Experiments

## Datasets

All detection experiments share the same leakage-free benchmark: 398 real RFOD images for training and a fixed disjoint set of 102 real images for validation. Augmentation pools (500 images each) include RailSyn (ours, full diagnosis-guided pipeline), RFOD23 (oficial training pool (Chen et al. 2024)), SD (railway-domain images from Stable Difusion (Rombach et al. 2022)), and Nano500 (images from Nano Banana). As non-railway baselines, STL500 and SODA500 contain 500 randomly sampled images from STL-10 (Coates, Ng, and Lee 2011) and SODA-10M (Han et al. 2021), respectively, and serve to test whether generic natural images alone can improve RFOD. All methods augment the same real training set; only the appended synthetic pool varies. The Inspector leverages both the training images and extra real RFOD images, with all data being strictly disjoint from the validation set.

## Evaluation Metrics

We evaluate downstream detection using AP50 and COCOstyle AP50–95. Our primary detectors are the widely validated state-of-the-art YOLO11 (Jocher, Chaurasia, and Qiu

2023) and DEIM (Huang et al. 2025b): YOLO11 represents eficient one-stage dense prediction with multiscale features, whereas DEIM represents end-to-end Transformer detection with global query-based reasoning. Their complementary local and global modeling tests whether synthetic data support both small-object appearance and object–scene intrusion relations. To assess broader cross-architecture utility, we further include YOLO12, YOLO13, and YOLO26 as recent dense-prediction variants, Gold-YOLO with cross-scale feature aggregation, DETR with global set prediction, RT-DETR with eficient multiscale Transformer encoding, and DINO with denoising-enhanced query learning (Tian, Ye, and Doermann 2025; Lei et al. 2025; Jocher et al. 2026; Wang et al. 2023; Carion et al. 2020; Zhao et al. 2024; Zhang et al. 2023). Inspector diagnostics characterize reliable-support reach, local-gap occupation, and nonredundant expansion, while FID and KID are introduced as reference metrics in the ablation study.

## Cross-Architecture Evaluation of RailSyn under Synthetic-Data Scaling

Table 1 evaluates whether RailSyn provides broadly useful detection data across nine architectures by varying the number of synthetic samples added to the fixed real training set. Every one of the nine detectors exceeds its real-only AP50– 95 baseline at an appropriate synthetic-data scale. Their best gains average 3.03 points and range from +1.9 points for YOLO13 to +4.9 points for YOLO12. The primary detectors show the same benefit: YOLO11 rises from 39.6 to 42.4 AP50–95 at +300, while DEIM rises from 46.2 to 49.1 at +500. Improvements across dense one-stage detectors and query-based Transformer detectors demonstrate that RailSyn supplies detection-relevant railway context, intrusion relations, and object–background cues that are usable across heterogeneous architectures rather than tailored to one detector.

The optimal amount of generated data is architecture dependent: AP50–95 peaks at +500 for DEIM and DETR, +300 for YOLO11, +100 for YOLO12, and +400 for RT-DETR and DINO. This variation shows that detectors absorb the supplemented information at diferent rates, but does not alter the central result: each architecture benefits from Rail-Syn at a suitable augmentation scale. The scaling study therefore establishes the broad utility of the Generator before the following controlled experiments analyze where these gains arise and how the Inspector traces changes among generated pools.

## Ablation Study: Detection Performance and Data Completion

Table 2 reports detector performance and Inspector diagnostics for six controlled RailSyn variants.

Stage ablation and detection performance. Rows 1–4 of Table 2 isolate the three Generator stages: Scene Prep. (Railway Scene Preparation), Relation Plan. (Relation-Aware Intrusion Planning), and Pattern Ref. (Diagnosed Intrusion-Pattern Refinement). Scene preparation alone improves YOLO11 but leaves DEIM close to real-only training. Adding relation-aware planning without intrusion-pattern refinement benefits DEIM (75.9/48.7) but reduces YOLO11 to 61.6/37.4, whereas adding intrusion-pattern refinement without relation-aware planning reaches 74.5 AP50 on DEIM but does not consistently improve the stricter AP50–95 results. Only the complete configuration combines the three stages to improve all four primary AP entries, raising AP50/AP50–95 by 4.7/2.0 points for YOLO11 and 5.1/2.9 points for DEIM. These controlled results establish distinct roles for the stages and show that relation-aware planning and intrusion-pattern refinement must operate together to convert prepared railway scenes into consistently useful detection samples.

Table 1: Cross-architecture detection performance with increasing numbers of RailSyn samples (AP50/AP50–95, %).
<table><tr><td>Detector</td><td>Real</td><td>+100</td><td>+200</td><td>+300</td><td>+400</td><td>+500</td></tr><tr><td>DEIM</td><td>72.0/46.2</td><td>75.4/48.3</td><td>78.2/48.6</td><td>75.2/47.1</td><td>77.0/48.0</td><td>77.1/49.1</td></tr><tr><td>YOLO11</td><td>62.6/39.6</td><td>65.1/39.8</td><td>63.7/40.7</td><td>66.6/42.4</td><td>62.2/40.4</td><td>67.3/41.6</td></tr><tr><td>YOLO12</td><td>59.9/37.6</td><td>70.4/42.5</td><td>64.9/40.0</td><td>61.6/39.7</td><td>58.1/36.0</td><td>59.7/39.0</td></tr><tr><td>YOL013</td><td>60.5/35.2</td><td>57.0/35.2</td><td>59.3/36.7</td><td>60.9/37.1</td><td>59.4/34.4</td><td>58.4/35.5</td></tr><tr><td>YOLO26</td><td>70.3/45.0</td><td>70.7/47.4</td><td>73.2/47.7</td><td>73.9/45.9</td><td>74.7/47.4</td><td>73.5/49.3</td></tr><tr><td>Gold-YOLO</td><td>63.4/37.6</td><td>67.2/40.3</td><td>67.2/41.9</td><td>71.0/41.9</td><td>69.6/41.6</td><td>69.9/41.5</td></tr><tr><td>DETR</td><td>75.3/47.1</td><td>74.6/47.0</td><td>71.2/47.3</td><td>74.4/46.8</td><td>74.2/47.1</td><td>78.7/49.1</td></tr><tr><td>RT-DETR</td><td>73.2/47.1</td><td>74.3/47.7</td><td>75.0/47.6</td><td>76.0/48.8</td><td>76.8/49.3</td><td>77.2/48.9</td></tr><tr><td>DINO</td><td>66.8/42.2</td><td>67.7/42.4</td><td>66.5/41.7</td><td>69.9/42.1</td><td>69.3/44.2</td><td>71.6/43.3</td></tr></table>

Table 2: RailSyn ablation with detector metrics, real-reference Inspector diagnostics (%), and generation quality. During object insertion, we optionally resize the foreign object relative to its planned bounding box: zoom-in enlarges the object by 10%, zoom-out reduces it by 10%, and the default configuration keeps the original Agent-planned size.
<table><tr><td rowspan="3">ID</td><td colspan="4">RailSyn stages and scale</td><td colspan="2">YOL011</td><td colspan="2">DEIM</td><td colspan="3">Inspector analysis</td><td colspan="2">Generation</td></tr><tr><td>Scene Relation Pattern Prep.</td><td>Plan.</td><td>Ref.</td><td></td><td>Zoom AP50</td><td>AP50-95</td><td>AP50</td><td>AP50-95</td><td> $C _ { \mathrm { g a p } }$ </td><td> $C _ { \mathrm { r e l } }$ </td><td> $\eta _ { \mathrm { v o l } }$ </td><td>FID↓</td><td> $\mathrm { K I D \times 1 0 ^ { 3 } \downarrow }$ </td></tr><tr><td>Real</td><td></td><td></td><td></td><td></td><td>62.6</td><td>39.6</td><td>72.0</td><td>46.2</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>1</td><td>√</td><td>一</td><td></td><td></td><td>66.6</td><td>40.4</td><td>71.7</td><td>46.3</td><td>11.88</td><td>2.77</td><td>75.90</td><td>208.79</td><td>121.02±8.66</td></tr><tr><td>2</td><td>√</td><td>√</td><td></td><td></td><td>61.6</td><td>37.4</td><td>75.9</td><td>48.7</td><td>11.16</td><td>2.46</td><td>78.95</td><td>252.22</td><td>211.24±10.56</td></tr><tr><td>3</td><td>√</td><td>一</td><td>√</td><td></td><td>61.1</td><td>39.8</td><td>74.5</td><td>45.4</td><td>12.82</td><td>2.73</td><td>77.13</td><td>207.12</td><td>124.80±8.59</td></tr><tr><td>4</td><td>√</td><td>√</td><td>√</td><td>一</td><td>67.3</td><td>41.6</td><td>77.1</td><td>49.1</td><td>13.64</td><td>2.47</td><td>80.81</td><td>231.41</td><td>178.89±9.79</td></tr><tr><td>5</td><td>√</td><td>&gt;</td><td>√</td><td>in</td><td>66.6</td><td>40.4</td><td>75.9</td><td>47.2</td><td>13.37</td><td>2.45</td><td>80.01</td><td>223.15</td><td>164.12±10.76</td></tr><tr><td>6</td><td>√</td><td>V</td><td>√</td><td>out</td><td>65.2</td><td>41.0</td><td>75.8</td><td>46.4</td><td>13.44</td><td>2.62</td><td>79.81</td><td>253.33</td><td> $2 1 7 . 5 8 { \pm } 1 0 . 5 6 $ </td></tr></table>

Inspector traceability under stage and stage changes. The Inspector assigns a distinct completion profile to every controlled variant. Adding intrusion-pattern refinement to scene preparation (ID3 vs. ID1) raises $C _ { \mathrm { g a p } }$ from 11.88% to 12.82% and $\eta _ { \mathrm { v o l } }$ from 75.90% to 77.13%, while $C _ { \mathrm { r e l } }$ edges down slightly (2.77% to 2.73%), indicating that finer injection improves local-gap coverage and volume eficiency without expanding reliable-support reach. Coupling all three stages (ID4) further lifts $C _ { \mathrm { g a p } }$ to a best 13.64% and $\eta _ { \mathrm { v o l } }$ to a best 80.81%, alongside the strongest detector results, while $C _ { \mathrm { r e l } }$ settles at 2.47%. Rows 5–6 alter only foreignobject scale under the full Generator. Zoom-in (ID5) and zoom-out (ID6) both reduce $C _ { \mathrm { g a p } }$ and $\eta _ { \mathrm { v o l } }$ relative to the default scale, with $C _ { \mathrm { r e l } }$ varying modestly. The Inspector thus captures fine-grained scale perturbations as well as coarse stage interventions, isolating how each configuration shifts the completion profile. Notably, configurations with higher $C _ { \mathrm { g a p } }$ (ID4, ID6, ID5) correspond to the best overall AP on both detectors, consistent with the notion that local-gap occupation traces detector-relevant completion. By contrast, FID and KID favor incomplete configurations (ID1 and ID3) and fail to reflect these stage- and scale-specific shifts in data utility.

## Comparison with External Pools

Table 3 compares RailSyn with five external augmentation pools under DEIM and YOLO11. RailSyn achieves the best results in all four AP columns: 77.1/49.1 on DEIM and 67.3/41.6 on YOLO11, with the largest mean improvement of +3.7 points over real-only training. In contrast, SD, Nano500, RFOD23, SODA500, and STL500 yield smaller or even negative gains, confirming that efective augmentation requires more than generic imagery.

To understand which semantic dimensions are actually complemented, we further probe the generated samples with a Inspector-based token-level semantic analysis. The rightmost columns of Table 3 report the density of four representative concepts: Railway, Balloon, Track, and Industry. All pools except STL500 consistently complete Railway and Track, while Balloon remains a common deficit. Crucially, RailSyn is the only pool that exhibits a moderate presence of Industry (+), indicating that our generation pipeline supplements railway-related industrial surroundings that are absent from other synthetic sources. This fine-grained semantic view explains why RailSyn provides stronger detection gains:

Table 3: Comparison of diferent augmentation data. The semantic profile summarizes Qwen semantic-focus probes over four English concepts: ++ denotes high token density (completion), + denotes moderate density (present but not dominant), and − denotes a deficit. ∆Avg is the gain of the mean of the four AP entries over the real-only mean (55.1).
<table><tr><td></td><td colspan="2">DEIM</td><td colspan="2">YOL011</td><td></td><td colspan="4">Semantic profile</td></tr><tr><td>Dataset</td><td>AP50</td><td>AP50-95</td><td>AP50</td><td>AP50-95</td><td>∆Avg</td><td>Railway</td><td>Balloon</td><td>Track</td><td>Industry</td></tr><tr><td>Real</td><td>72.0</td><td>46.2</td><td>62.6</td><td>39.6</td><td>0.0</td><td>一</td><td></td><td>一</td><td>一</td></tr><tr><td>Nano500</td><td>74.9</td><td>48.6</td><td>66.1</td><td>37.5</td><td>+1.7</td><td>++</td><td>1</td><td>++</td><td>1</td></tr><tr><td>SD</td><td>75.9</td><td>48.0</td><td>64.9</td><td>40.3</td><td>+2.2</td><td>++</td><td>一</td><td>++</td><td>一</td></tr><tr><td>RFOD23</td><td>73.7</td><td>46.3</td><td>66.2</td><td>39.7</td><td>+1.4</td><td>++</td><td>一</td><td>++</td><td>一</td></tr><tr><td>STL500</td><td>76.4</td><td>47.7</td><td>58.6</td><td>35.9</td><td>-0.4</td><td></td><td>++</td><td></td><td></td></tr><tr><td>SODA500</td><td>75.7</td><td>46.8</td><td>61.0</td><td>36.5</td><td>-0.1</td><td>++</td><td></td><td>++</td><td>一</td></tr><tr><td>RailSyn (Ours)</td><td>77.1</td><td>49.1</td><td>67.3</td><td>41.6</td><td>+3.7</td><td>++</td><td>一</td><td>++</td><td>十</td></tr></table>

it not only reinforces core railway elements but also expands the contextual diversity along an extra task-relevant dimension. Token-level analysis thus ofers an orthogonal lens for tracing which task dimensions are complemented, without relying on aggregate AP alone.

## Inspector Diagnostics and Detection Performance

We next examine whether the Inspector profiles are related to detector utility across the six controlled variants in Table 2. As shown in Table 4, $C _ { \mathrm { r e l } }$ and $C _ { \mathrm { g a p } }$ correlate positively with all four AP measures. $C _ { \mathrm { r e l } }$ has the strongest association with DEIM AP50 $( \rho = . 7 3 5 )$ , while $C _ { \mathrm { g a p } }$ has the strongest association with YOLO11 AP50–95 $( \rho = . 6 6 7 )$ . In contrast, $\eta _ { \mathrm { v o l } }$ is weakly related to AP and is negative for DEIM AP50– 95 $( \rho = - . 2 0 0 )$ , confirming that nonredundant expansion captures a complementary property rather than detection performance itself. Together with the stage and scale controls, these directional associations support the Inspector’s ability to trace detector-relevant diferences among generated variants; with $n = 6 ,$ they are not treated as statistical significance or an AP predictor.

Table 4: Spearman $\rho$ between Qwen Inspector diagnostics and detector AP.
<table><tr><td rowspan="2"></td><td colspan="2">DEIM</td><td colspan="2">YOL011</td></tr><tr><td>Metric AP50</td><td>AP50-95</td><td>AP50</td><td>AP50-95</td></tr><tr><td> $C _ { \mathrm { r e l } }$ </td><td>-.812</td><td>-.714</td><td>-.174</td><td>.058</td></tr><tr><td> $C _ { \mathrm { g a p } }$ </td><td>.464</td><td>.314</td><td>.551</td><td>.899</td></tr><tr><td> $\eta _ { \mathrm { v o l } }$ </td><td>.899</td><td>.771</td><td>.551</td><td>.609</td></tr><tr><td> $\eta _ { \mathrm { v o l } }$ </td><td>.899</td><td>.771</td><td>.551</td><td>.609</td></tr></table>

Table 5 further tests whether the Inspector’s pool rankings persist across Qwen, CLIP, and DINOv2. $C _ { \mathrm { g a p } } ^ { - }$ maintains strong pairwise agreement, with correlations from 0.771 to 0.943 and a mean of 0.848. $\eta _ { \mathrm { v o l } }$ is even more stable, with correlations from 0.943 to 1.000 and a mean of 0.962. $C _ { \mathrm { r e l } }$ is more representation-sensitive, with a mean correlation of 0.505. The high agreement for local-gap occupation and nonredundant expansion shows that the main relative patterns are not specific to one embedding model. Combined with the controlled ablations and the positive AP associations, this cross-embedding consistency provides complementary evidence that the Inspector ofers a stable, traceable analysis of generated-pool diferences.

Table 5: Cross-encoder rank stability over the six pools in Table 3. Values are Spearman rank correlations ρ between within-encoder pool rankings. Q–C, Q–D, and C–D denote Qwen–CLIP, Qwen–DINOv2, and CLIP–DINOv2, respectively. Absolute metric values are not compared across embedding spaces.
<table><tr><td>Metric</td><td>Q-C</td><td>Q-D</td><td>C-D</td><td>Avg.</td><td>Min.</td></tr><tr><td> $C _ { \mathrm { r e l } }$ </td><td>0.771</td><td>0.429</td><td>0.314</td><td>0.505</td><td>0.314</td></tr><tr><td> $C _ { \mathrm { g a p } }$ </td><td>0.829</td><td>0.943</td><td>0.771</td><td>0.848</td><td>0.771</td></tr><tr><td>ηvol</td><td>0.943</td><td>0.943</td><td>1.000</td><td>0.962</td><td>0.943</td></tr></table>

## Adverse-Weather Robustness

We extend RailSyn to adverse weather via RailSynWeather, a branch that generates 200 weather-specific images while preserving layout and annotations. On the same 398-image real training set, RailSynWeather raises DEIM AP50 from 72.6%, obtained with a filter-based augmentation baseline, to 73.1% on the clean validation split, confirming the generator’s extensibility to underrepresented weather conditions.

## Conclusion

We presented RailSyn, an Inspector–Generator framework that diagnoses high-uncertainty regions from real RFOD observations and generates synthetic data through scene preparation, relation-aware planning, and intrusion-pattern refinement. Across nine detectors, RailSyn improves AP50– 95 over real-only training, with an best gain of 4.9 points, and outperforms five external augmentation pools in controlled comparisons. In the module ablation study, the Inspector traces how each generation stage and scale choice alters the completion profile, with the full configuration achieving the highest local-gap occupation at 13.64%, providing a component-level view of what diferent modules contribute. Current limitations include the dificulty of characterizing cross-dataset efectiveness and the need for further improvement in generation quality. Future work includes developing quantitative metrics from token-level semantic probes to enable cross-dataset evaluation, extending Inspector-guided completion to other safety-critical domains, and enhancing generation fidelity under adverse weather and for small-object details.

## References

Black Forest Labs. 2024. FLUX.1: Flow Matching for Textto-Image Generation. Technical Report.

Bolya, D.; Foley, S.; Hays, J.; and Hofman, J. 2020. TIDE: A General Toolbox for Identifying Object Detection Errors. In Computer Vision – ECCV 2020, volume 12348 of Lecture Notes in Computer Science, 558–573. Springer.

Carion, N.; Massa, F.; Synnaeve, G.; Usunier, N.; Kirillov, A.; and Zagoruyko, S. 2020. End-to-End Object Detection with Transformers. In European Conference on Computer Vision (ECCV), 213–229.

Chen, Y.; Yuan, X.; Wang, J.; Wu, R.; Li, X.; Hou, Q.; and Cheng, M.-M. 2025. YOLO-MS: Rethinking Multi-Scale Representation Learning for Real-Time Object Detection. IEEE Transactions on Pattern Analysis and Machine Intelligence, 47(6): 4240–4252.

Chen, Z.; Yang, J.; Feng, Z.; and Zhu, H. 2024. RailFOD23: A dataset for foreign object detection on railroad transmission lines. Scientific Data, 11(1): 72.

Cheng, G.; Yuan, X.; Yao, X.; Yan, K.; Zeng, Q.; Xie, X.; and Han, J. 2023. Towards Large-Scale Small Object Detection: Survey and Benchmarks. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(11): 13467–13488.

Coates, A.; Ng, A.; and Lee, H. 2011. An Analysis of Single-Layer Networks in Unsupervised Feature Learning. In Gordon, G.; Dunson, D.; and Dudík, M., eds., Proceedings of the Fourteenth International Conference on Artificial Intelligence and Statistics, volume 15 of Proceedings ofMachine Learning Research, 215–223. Fort Lauderdale, FL, USA: PMLR.

Esser, P.; Kulal, S.; Blattmann, A.; Entezari, R.; Müller, J.; Saini, H.; Levi, Y.; Lorenz, D.; Sauer, A.; Boesel, F.; Podell, D.; Dockhorn, T.; English, Z.; and Rombach, R. 2024. Scaling Rectified Flow Transformers for High-Resolution Image Synthesis. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, 12606–12633. PMLR.

Feng, C.; Zhong, Y.; Jie, Z.; Xie, W.; and Ma, L. 2024. Insta-Gen: Enhancing Object Detection by Training on Synthetic Dataset. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 14121–14130.

Ghiasi, G.; Cui, Y.; Srinivas, A.; Qian, R.; Lin, T.-Y.; Cubuk, E. D.; Le, Q. V.; and Zoph, B. 2021. Simple Copy-Paste Is a Strong Data Augmentation Method for Instance Segmentation. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2918–2928.

Ghosh, D.; Hajishirzi, H.; and Schmidt, L. 2023. GenEval: An Object-Focused Framework for Evaluating Text-to-Image Alignment. In Advances in Neural Information Processing Systems, volume 36, 52132–52152.

Han, J.; Liang, X.; Xu, H.; Chen, K.; Hong, L.; Mao, J.; Ye, C.; Zhang, W.; Li, Z.; Liang, X.; and Xu, C. 2021. SODA10M: A Large-Scale 2D Self/Semi-Supervised Object Detection Dataset for Autonomous Driving. In Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks, volume 1.

Hao, Q.; Shi, R.; Li, J.; and Zhang, L. 2026. Generative Approach for Detecting Small Intrusive Foreign Objects in High-Speed Railway Scenario. IEEE Transactions on Intelligent Transportation Systems, 27(1): 1471–1484.

Hu, E. J.; Shen, Y.; Wallis, P.; Allen-Zhu, Z.; Li, Y.; Wang, S.; Wang, L.; and Chen, W. 2022. LoRA: Low-Rank Adaptation of Large Language Models. In International Conference on Learning Representations (ICLR).

Hu, Y.; Liu, B.; Kasai, J.; Wang, Y.; Ostendorf, M.; Krishna, R.; and Smith, N. A. 2023. TIFA: Accurate and Interpretable Text-to-Image Faithfulness Evaluation with Question Answering. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 20406–20417.

Huang, K.; Duan, C.; Sun, K.; Xie, E.; Li, Z.; and Liu, X. 2025a. T2I-CompBench++: An Enhanced and Comprehensive Benchmark for Compositional Text-to-Image Generation. IEEE Transactions on Pattern Analysis and Machine Intelligence, 47(5): 3563–3579.

Huang, S.; Lu, Z.; Cun, X.; Yu, Y.; Zhou, X.; and Shen, X. 2025b. DEIM: DETR with Improved Matching for Fast Convergence. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 15162– 15171.

Islam, K.; Zaheer, M. Z.; Mahmood, A.; and Nandakumar, K. 2024. DifuseMix: Label-Preserving Data Augmentation with Difusion Models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 27621–27630.

Jayasumana, S.; Ramalingam, S.; Veit, A.; Glasner, D.; Chakrabarti, A.; and Kumar, S. 2024. Rethinking FID: Towards a Better Evaluation Metric for Image Generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 9307–9315.

Jia, C.; Luo, M.; Dang, Z.; Dai, G.; Chang, X.; Wang, M.; and Wang, J. 2024. SSMG: Spatial-Semantic Map Guided Difusion Model for Free-Form Layout-to-Image Generation. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, 2480–2488.

Jocher, G.; Chaurasia, A.; and Qiu, J. 2023. Ultralytics YOLO. https://github.com/ultralytics/ultralytics.

Jocher, G.; Qiu, J.; Liu, M.; Lyu, S.; Akyon, F. C.; and Kalfaoglu, M. E. 2026. Ultralytics YOLO26: Unified Real-Time End-to-End Vision Models. Oficial Ultralytics arXiv preprint, arXiv:2606.03748.

Kang, B.; Liu, Z.; Yan, X.; Holtz, C.; and Feng, J. 2019. Fewshot Object Detection via Feature Reweighting. In Proceedings ofthe IEEE/CVFInternational Conference on Computer Vision (ICCV), 8420–8429.

Lee, T.; Yasunaga, M.; Meng, C.; Mai, Y.; Park, J. S.; Gupta, A.; Zhang, Y.; Narayanan, D.; Teufel, H.; Bellagente, M.; Kang, M.; Park, T.; Leskovec, J.; Zhu, J.-Y.; Li, F.-F.; Wu, J.; Ermon, S.; and Liang, P. 2023. Holistic Evaluation of Text-to-Image Models. In Advances in Neural Information Processing Systems, volume 36.

Lei, M.; Li, S.; Wu, Y.; Hu, H.; Zhou, Y.; Zheng, X.; Ding, G.; Du, S.; Wu, Z.; and Gao, Y. 2025. YOLOv13: Real-Time Object Detection with Hypergraph-Enhanced Adaptive Visual

Perception. ArXiv preprint; no peer-reviewed proceedings version was located as of 2026-07-21, arXiv:2506.17733.

Li, K.; Luo, S.; and Tseng, K.-K. 2025. RegionDifusion: Generative data augmentation for object detection with diffusion models. Neurocomputing, 654: 131115.

Li, M.; Zhang, Y.; Long, D.; Chen, K.; Song, S.; Bai, S.; Yang, Z.; Xie, P.; Yang, A.; Liu, D.; Zhou, J.; and Lin, J. 2026. Qwen3-VL-Embedding and Qwen3-VL-Reranker: A Unified Framework for State-of-the-Art Multimodal Retrieval and Ranking. arXiv:2601.04720.

Li, Y.; Liu, H.; Wu, Q.; Mu, F.; Yang, J.; Gao, J.; Chandra, R.; et al. 2023. GLIGEN: Open-Set Grounded Text-to-Image Generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 22511– 22521.

Lin, T.-Y.; Goyal, P.; Girshick, R.; He, K.; and Dollár, P. 2017. Focal Loss for Dense Object Detection. In Proceedings of the IEEE International Conference on Computer Vision, 2980– 2988.

Lipman, Y.; Chen, R. T. Q.; Ben-Hamu, H.; Nickel, M.; and Le, M. 2023. Flow Matching for Generative Modeling. In The Eleventh International Conference on Learning Representations.

Oza, P.; Sindagi, V. A.; VS, V.; and Patel, V. M. 2024. Unsupervised Domain Adaptation of Object Detectors: A Survey. IEEE Transactions on Pattern Analysis and Machine Intelligence, 46(6): 4018–4040.

Redmon, J.; Divvala, S.; Girshick, R.; and Farhadi, A. 2016. You Only Look Once: Unified, Real-Time Object Detection. In Proceedings ofthe IEEE Conference on Computer Vision and Pattern Recognition, 779–788.

Rombach, R.; Blattmann, A.; Lorenz, D.; Esser, P.; and Ommer, B. 2022. High-Resolution Image Synthesis with Latent Difusion Models. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 10684–10695.

Shi, Z.; Hu, J.; Ren, J.; Ye, H.; Yuan, X.; Ouyang, Y.; He, J.; Ji, B.; and Guo, J. 2025. HS-FPN: High Frequency and Spatial Perception FPN for Tiny Object Detection. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, 6896–6904.

Somepalli, G.; Singla, V.; Goldblum, M.; Geiping, J.; and Goldstein, T. 2023. Difusion Art or Digital Forgery? Investigating Data Replication in Difusion Models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 6048–6058.

Song, Y.; Zhang, Z.; Lin, Z.; Cohen, S.; Price, B.; Zhang, J.; Kim, S. Y.; Zhang, H.; Xiong, W.; and Aliaga, D. 2024. IMPRINT: Generative Object Compositing by Learning Identity-Preserving Representation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 8048–8058.

Tian, Y.; Ye, Q.; and Doermann, D. 2025. YOLOv12: Attention-Centric Real-Time Object Detectors. In Advances in Neural Information Processing Systems, volume 38.

Trabucco, B.; Doherty, K.; Gurinas, M. A.; and Salakhutdinov, R. 2024. Efective Data Augmentation With Difusion Models. In The Twelfth International Conference on Learning Representations.

Wan, Z.; Wang, Z.; Chung, C.; and Wang, Z. 2024. A Survey of Dataset Refinement for Problems in Computer Vision Datasets. ACM Computing Surveys, 56(7): 172.

Wang, C.; He, W.; Nie, Y.; Guo, J.; Liu, C.; Wang, Y.; and Han, K. 2023. Gold-YOLO: Eficient Object Detector via Gather-and-Distribute Mechanism. In Advances in Neural Information Processing Systems, volume 36.

Wang, K.; Pan, Z.; and Wen, Z. 2025. SVDDD: SAR Vehicle Target Detection Dataset Augmentation Based on Difusion Model. Remote Sensing, 17(2): 286.

Wang, Y.; Gao, R.; Chen, K.; Zhou, K.; Cai, Y.; Hong, L.; Li, Z.; Jiang, L.; Yeung, D.-Y.; Xu, Q.; and Zhang, K. 2024. DetDifusion: Synergizing Generative and Perceptive Models for Enhanced Data Generation and Perception. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 7246–7255.

Wu, T.-H.; Lian, L.; Gonzalez, J. E.; Li, B.; and Darrell, T. 2024. Self-Correcting LLM-Controlled Difusion Models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 6327–6336.

Zhang, H.; Li, F.; Liu, S.; Zhang, L.; Su, H.; Zhu, J.; Ni, L. M.; and Shum, H.-Y. 2023. DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection. In The Eleventh International Conference on Learning Representations.

Zhang, L.; Rao, A.; and Agrawala, M. 2023. Adding Conditional Control to Text-to-Image Difusion Models. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 3836–3847.

Zhao, Y.; Lv, W.; Xu, S.; Wei, J.; Wang, G.; Dang, Q.; Liu, Y.; and Chen, J. 2024. DETRs Beat YOLOs on Real-Time Object Detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 16965– 16974.