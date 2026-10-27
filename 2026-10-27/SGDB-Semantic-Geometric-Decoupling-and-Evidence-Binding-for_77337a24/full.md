# SGDB: Semantic–Geometric Decoupling and Evidence Binding for Logical Anomaly Detection

Chenrui Ji<sup>a</sup>, Jiafan Lu<sup>a</sup>, Zilu Luo<sup>a</sup>, Feng Wan<sup>a</sup>, Qiuhui Chen<sup>a,∗</sup>, Weimin Zhong<sup>a,∗</sup>

<sup>a</sup>School ofInformation Science and Engineering, East China University ofScience and Technology, Shanghai 200237, China

## Abstract

Logical anomaly detection requires recognizing scene components and measuring whether their configuration satisfies task constraints. Correct identities can coexist with inaccurate geometry; clean mask fragments can yield misleading lengths. We propose Semantic–6 Geometric Decoupling and Evidence Binding (SGDB). A semantic decoder learns object identities from component-annotated<sup>2</sup> normal sources and factual edits; a separate reader supplies final measurements. Partial one-to-one association binds geometric<sup>0</sup> witnesses to identities while retaining unmatched objects in the count. A support-certified reader selects mask-derived axes using line quality and complete component ownership. Explicit task requirements and normal-data references define executable checks, with decision thresholds fixed by independent normal calibration. Execution-grounded explanation traces rule contributions to source objects and measurements, extracting compact sufficient alarm certificates. A frozen language model expresses selected outcomes as cited claims, with numeric evidence retained by the executor. Reports preserve observed deviations and unavailable evidence. SGDB achieves 98.19% AUROC and 94.52% fixed-threshold F1 on five MVTec LOCO logical categories, and 93.56% AUROC and 92.76% F1 across all 50 VID-AD tasks. Frozen-output controls evaluate geometric readout and explanatory sufficiency and compactness.]

Keywords: Logical anomaly detection, Semantic–geometric decoupling, Evidence binding, Geometric measurement, Execution-grounded explanation

## 1. Introduction

v Logical anomaly detection requires recognizing scene components and measuring whether their configuration satisfies task requirements. Two screw bags can contain identical component counts yet differ in geometric validity; a tray can contain the<sub>4</sub> required utensils in the wrong order. Recognition establishes3 identities and counts, while measurement supplies lengths, positions and relations. MVTec LOCO AD distinguishes these1 logical violations from structural defects [1], and VID-AD tests<sup>6</sup> inspection under capture distractions [2].<sup>2</sup>

Recent component-aware methods model component regions, object composition or reference matching [4–7]. Language-<sup>i</sup> based approaches express inspection requirements through descriptions, programs and constraint subqueries [8–10]. Figure 1a illustrates why correct object identities and counts do not ensure geometric validity. Each geometric operand must be suitable for its constraint and attached to the intended object. A correctly recognized screw can have inaccurate regressed endpoints. A mask-derived length requires an identity to be checked against its object-specific bound. However, assigning a clean fragment of a merged region to an object can produce a plausible but incomplete axis. Readout must therefore check the instance support as well as the quality of the fitted line.

## (a) Same counts, different geometric validity

![](images/7e45b2a10b2b6bd43942114dd58636dc98360f8e0e99d8c87bed6f6be3c144c6.jpg)

## (b) SGDB: from image to grounded output

![](images/fcb6fb34aa7757ae2f78f6cc9e5d0a54cb725a12ed0b6670cf62378b3fe310e2.jpg)  
Figure 1: Motivation for explicit geometric verification beyond appearance/composition scoring [3, 4]. (a) The two scenes have identical component counts. (b) SGDB maps an image to a decision, explanation and evidence through semantic–geometric decoupling, binding and verification. The recorded rope/reference ratio 1.362 exceeds the normal upper bound 1.174. Images and measurements are from frozen outputs.

We study logical inspection configured with normal images, component annotations and explicit task requirements. Component targets supervise visual readers; normal observations establish measurement ranges and score references; task requirements specify the discrete constraints to execute. We propose Semantic– Geometric Decoupling and Evidence Binding (SGDB), which separates semantic recognition from final geometric readout and reconnects their outputs through explicit evidence binding. Component-aware inspection [4, 11–13] provides useful object and region representations; binding establishes which measured operand belongs to each decoded identity.

SGDB implements this separation above a shared visual frontend. A learned object-set decoder reads frozen features with candidate and spatial queries. A witness reader supplies initial geometric operands, including native-coordinate mask readout for screw axes. Same-class partial one-to-one association attaches compatible witnesses to decoded objects using center proximity. A subsequent support-certified readout replaces a length axis only when the candidate improves support quality and preserves a complete, distinct component for its observed instance. Each fact retains class, confidence, associated geometry, measurement availability and source indices. Unmatched identities still contribute to the count, with their geometric operands marked unavailable.

Bound facts support both executable verification and an explanation of the resulting decision. A rule deviation alone need not trigger an alarm: normal-reference normalization, aggregation and the task threshold determine its effect. SGDB therefore records each executed operand, requirement and score contribution together with its source links. It extracts a compact sufficient alarm certificate and uses the same trace to express the decision, supporting measurements and unavailable evidence.

The selected trace also conditions a frozen language model, whose claims retain executor-owned evidence citations and numeric records.

SGDB connects three complementary mechanisms through this evidence representation:

1. Semantic–geometric fact construction assigns object identification and counting to a learned semantic decoder and final geometric operands to a separate witness readout.

2. Partial evidence binding and support-certified readout join identities to compatible measurements, preserve unmatched identities and verify component ownership before updating geometry.

3. Execution-grounded explanation extracts compact sufficient alarm certificates and traces their predicates to object identities, measured operands and evidence availability.

Evaluation spans all five LOCO logical categories and all 50 VID tasks. Shared-supervision controls examine fact construction and readout; frozen-trace comparisons examine explanatory sufficiency and compactness.

## 2. Related work

## 2.1. Appearance models and industrial benchmarks

Normal-appearance modeling is a central approach to industrial anomaly detection. MVTec AD provides defect images with pixel annotations [14], while VisA broadens the range of objects and multi-instance scenes [15]. PaDiM models patch embeddings with multivariate Gaussian distributions [16], and PatchCore stores a representative memory of normal patch features [3]. Reverse Distillation reconstructs teacher features from a compact one-class embedding [17]; EfficientAD combines efficient feature prediction with a global autoencoder [18]. DRAEM instead learns reconstruction and discrimination jointly using simulated anomalies [19]. These approaches establish useful appearance representations. Logical inspection additionally requires the identities, quantities and relationships of otherwise plausible components to be evaluated against the task configuration [1].

## 2.2. Component and relationship representations

Component-level methods address this requirement through segmentation and composition. ComAD extracts metrological properties from component regions [11]. PSAD uses a few annotated images to learn part segmentation and combines histogram, composition and patch memories [5]; CSAD obtains component pseudo-labels and combines patch histograms with local–global modeling [12]. SALAD learns the distribution of composition maps through synthetic anomalies [4], whereas LogiCo combines component reconstruction with segmentationmap discrimination [13]. PUAD complements reconstruction with an out-of-distribution model of spatially pooled features [20]. Recent work in Neurocomputing also addresses global context through pyramid fusion and state-space enhancement [21], and represents quantity and spatial relationships with separate memories [22].

Object matching provides a more explicit representation of composition. SAM-LAD matches segmented objects between query and reference images [6]. ObjectCore represents an image as a collection of object features and detects few-shot logical anomalies through bipartite matching to a support image [7]. SGDB uses association within the same image: decoded semantic identities are linked to measurements produced by a separate visual reader. This yields object-specific operands for subsequent constraint execution, including counted identities whose geometry is unavailable.

Synthetic logical examples also provide supervision when real violations are scarce. ComGEN generates anomalies through component-aware compositional editing [23]. Chimera training constructs counterfactual operands at the feature level to train a neural rule evaluator [24]. SGDB uses factual image edits with updated component targets to train semantic decoding; task constraints are evaluated by a fixed executor after evidence association.

## 2.3. Object queries and geometric evidence

Attention supplies a general mechanism for reading visual features [25]. DETR formulates detection as object-set prediction with bipartite training assignment [26]. Deformable DETR conditions attention on a small set of locations around reference points [27], while DAB-DETR expresses queries using dynamically updated anchor boxes [28]. These designs show how spatial hypotheses can guide semantic feature extraction. SGDB combines candidate-initialized and grid queries, then separates semantic decoding from the selection of a final geometric operand. Its deployment association links two prediction sets, rather than predictions and training targets.

![](images/432cad2c194bef968516569a6772df7f4e712bf033027e3277e1368e93023331.jpg)  
Figure 2: SGDB framework. (a) Bound facts are checked against task constraints, aggregated and compared with the alarm threshold. (b) Objects bind to geometric witnesses by class and position; unmatched identities remain counted. Quality and instance-support checks select the mask axis or retain the initial axis, shown by solid green and dashed brown lines. (c) The interval penalty is zero within the accepted range and increases linearly outside it; the marked point is the recorded O4 length. (d) Selected evidence E05 supports the frozen LLM’s source-attributed explanation; numeric evidence is presented by the executor. Snowflakes denote the frozen visual frontend and language model.

When constraints depend on physical image geometry, the coordinate system of a measurement also matters. Anisotropic resizing can alter Euclidean lengths and principal directions. SGDB reads screw axes through instance-bound PCA in native coordinates [29], with support and elongation checks that retain the existing axis when the mask readout is unsuitable.

## 2.4. Language, executable reasoning and reports

Vision–language approaches connect visual normality to textual descriptions. WinCLIP uses prompt ensembles and windowlevel features [30]; AnomalyCLIP learns object-agnostic normality prompts [31]. AnomalyGPT supports anomaly localization and interactive descriptions [32]. For logical inspection, LogicAD combines extracted descriptions with formal reasoning [8], LogiCode generates executable inspection programs [9], and LogiCls decomposes constraints into subqueries for instruction-tuned violation classification [10]. VID-AD examines description-based logical inspection under visual distraction [2].

Executable intermediate representations are also studied beyond anomaly detection. VISPROG composes visual modules into programs with inspectable intermediate outputs [33]. Deep-ProbLog integrates neural predicates with probabilistic logic programs [34], and Logic Tensor Networks ground logical expressions in differentiable computations [35]. NeSyLAD applies learned segmentation and symbolic rule extraction to industrial logical anomalies [36]. SGDB evaluates explicit discrete requirements and continuous intervals fitted from normal data. Frozen OCR observations remain inputs for rope-label tasks; Qwen3-VL is evaluated as a separate comparison deployment [37, 38].

Sufficient explanations identify compact evidence supporting a prediction. Sufficient-subset training learns concise explanatory outputs [39], while sufficiency and necessity characterize different aspects of explanatory importance [40]. SGDB constructs its certificate over executed normalized rule scores. The certificate is connected to the semantic and geometric sources already established by evidence binding, and the full trace retains the other predicates and unavailable operands.

## 3. Semantic–geometric decoupling and evidence binding

## 3.1. Task setup and fact representation

Given image $\boldsymbol { x } \in \mathbb { R } ^ { H \times W \times 3 }$ and task t, SGDB returns an anomaly score $s _ { t } ( x )$ , decision ${ \hat { y } } _ { t } ( x )$ and supporting report. Figure 2 organizes the method into fact decoding, constraint verification and explanation. Its central representation is

$$
f _ { i } = ( \hat { c } _ { i } , p _ { i } , g _ { i } , a _ { i } ) ,\tag{1}
$$

where $\hat { c } _ { i }$ and $p _ { i }$ are a decoded class and confidence, $g _ { i }$ is associated geometric evidence, and $a _ { i }$ indicates measurement availability. Task rules operate on these assembled facts.

Component targets on normal fitting/development images supervise the visual readers and their selection. Explicit task requirements configure the executor; normal measurements supply continuous bounds and score references. At inspection, the input is an image and its task identifier. The readers predict object identities, masks and geometric witnesses, which are assembled into the facts above.

## 3.2. Semantic object decoding

The visual frontend comprises an ImageNet-initialized WideResNet50-2 encoder [41, 42], multiscale feature fusion and center, box, semantic-mask and endpoint heads. Center localization follows the object-as-point formulation [43]. Component targets from normal sources supervise the frontend, which is frozen during semantic-decoder training.

The decoder reads dense features $F ( x )$ using candidateinitialized queries and spatial queries:

$$
\begin{array} { l } { { O ( x ) = D _ { \theta } ( F ( x ) , Q _ { \mathrm { c a n d } } , Q _ { \mathrm { g r i d } } ) } } \\ { { \ = \{ ( \hat { c } _ { i } , p _ { i } , \hat { \bf u } _ { i } ) \} _ { i = 1 } ^ { n } . } } \end{array}\tag{2}
$$

Candidate queries encode the frontend’s class, confidence and geometric hypothesis together with sampled visual features. Spatial queries cover a fixed grid. Two Transformer layers update the queries; first-layer soft support masks contribute a spatial attention bias to the second layer. Figure 3 details this training mechanism. A normal-source parent and its factual edit share decoder parameters. Hungarian assignment matches each prediction set to its updated component targets for class, center, box, axis and mask supervision, denoted by ${ \mathcal { L } } _ { \mathrm { f a c t } }$ . Support coupling $\mathcal { L } _ { \mathrm { b i n d } }$ aligns predicted geometry with mask support. Paired-edit consistency $\mathcal { L } _ { \mathrm { e d i t } }$ supervises count changes and shared-object displacements:

$$
\mathcal { L } _ { D } = \mathcal { L } _ { \mathrm { f a c t } } + 0 . 2 5 \mathcal { L } _ { \mathrm { b i n d } } + 0 . 2 5 \mathcal { L } _ { \mathrm { e d i t } } .\tag{3}
$$

Edits add, remove or move components and carry updated factual targets. Duplicate candidate-query perturbations are also used

![](images/ccc02c79a991a1fc86358525b2517f67be75eee2e773c144c5d80f9af67c51d2.jpg)

![](images/cdbd9ed07a2ed8558a87b16a7b2238a10ceb642a8733cd4fe27a760f50d2f325.jpg)  
Figure 3: Fact-decoder training. (a) First-layer soft masks bias second-layer attention over frozen visual features. (b) A normal-source parent and its edit share decoder parameters. Updated targets supervise object sets; support coupling aligns geometry with masks, and edit consistency supervises count changes and shared-object displacement. The removal example is schematic.

during training. The frontend remains frozen; optimization and individual loss weights are specified in the supplementary material.

Background and padded slots are removed to obtain the object set. The center $\hat { \mathbf { u } } _ { i }$ routes an identity to a witness; final measurement values are supplied by the association below.

## 3.3. Partial one-to-one evidence association

Let $E ( x ) = \{ ( c _ { j } ^ { E } , \mathbf { u } _ { j } ^ { E } , g _ { j } ^ { E } ) \} _ { j = 1 } ^ { m }$ be the witness set, with $g _ { j } ^ { E }$ containing center, box and axis. A decoded object and witness are eligible when their classes agree and their normalized center distance $d _ { i j } = \lVert \hat { \mathbf { u } } _ { i } - \mathbf { u } _ { j } ^ { E } \rVert _ { 2 }$ is at most r. Algorithm 1 constructs an $n \times ( m + n )$ cost matrix with n dummy columns and computes a minimum-cost injective assignment.

A matched object inherits the witness geometry with $a _ { i } = 1 ;$ a dummy match sets $g _ { i } ~ = ~ \mathcal { D }$ and $a _ { i } ~ = ~ 0$ . Both retain the decoded class and confidence, so unmatched identities remain counted. The indexed fact list preserves multiplicities, and the trace records decoded/witness indices and matching distance.

Normal development data select a single scheme from sameclass or any-class association at radii 0.04 or 0.08. The selected same-class radius 0.08 is used across all categories and for both decoder variants.

## 3.4. Support-certified geometric readout

Initial witnesses use the frontend geometry, with the previously selected guarded native-axis reader for screws. This reader assigns box-supported screw-mask pixels and uses nativecoordinate PCA; its support, centroid and elongation checks are specified in the supplementary material. Association fixes which initial witness belongs to each decoded identity. The subsequent reader refines the axis component of these bound objects, preserving their decoded identities, centers and boxes.

For a mask-backed length class c, let $I _ { c }$ be its bound objects and $M _ { c }$ the frozen foreground mask. Each mask pixel is assigned to the nearest bound center in native image coordinates, $\mathbf { z } _ { i } =$ $( W u _ { i x } , H u _ { i y } )$ . Let $S _ { i }$ be the pixels assigned to $i , \pmb { \mu } _ { i }$ their nativecoordinate mean and $\mathbf { v } _ { i }$ their principal direction. A candidate axis $\tilde { g } _ { i }$ uses projection quantiles:

$$
\tilde { \mathbf { a } } _ { i } ^ { \pm } = \pmb { \mu } _ { i } + Q _ { . 0 1 / . 9 9 } \{ ( \mathbf { p } - \pmb { \mu } _ { i } ) ^ { \top } \mathbf { v } _ { i } : \mathbf { p } \in S _ { i } \} \mathbf { v } _ { i } .\tag{4}
$$

At least 16 pixels and finite, distinct endpoints are required. The candidate and initial axis are evaluated against the same $S _ { i }$ using

$$
Q ( g ; S _ { i } ) = \sqrt { P _ { \mathrm { l i n e } } ( g , S _ { i } ) P _ { \mathrm { e x t e n t } } ( g , S _ { i } ) } P _ { \mathrm { c o n n } } ( S _ { i } ) ,\tag{5}
$$

where $P _ { \mathrm { l i n e } }$ is the supported fraction of sampled axis points, $P _ { \mathrm { e x t e n t } }$ is the assigned-pixel fraction covered by the projected axis interval, and $P _ { \mathrm { c o n n } }$ is the largest connected-region fraction. A candidate requires $Q ( \tilde { g } _ { i } ; S _ { i } ) \ge q _ { 0 }$ and a quality gain of at least $\Delta _ { q } = 0 . 0 5$ . The global floor $q _ { 0 } \simeq 0 . 9 9 0 8 7$ is the tenth percentile of readable candidate quality on normal development images.

High quality alone can admit an incomplete fragment. Let $K _ { c }$ contain the substantial 8-connected components of the full class mask, and $k ( i )$ be the component with greatest overlap with $S _ { i }$ The class certificate is

$$
\begin{array} { l } { \displaystyle { G _ { c } = \mathbf { 1 } [ | K _ { c } | = | I _ { c } | ] \mathbf { 1 } [ k : I _ { c } \to K _ { c } \mathrm { i s \ b i j e c t i v e } ] } } \\ { \displaystyle ~ \cdot \prod _ { i \in I _ { c } } \mathbf { 1 } \left[ \frac { | S _ { i } \cap k ( i ) | } { | k ( i ) | } \geq 0 . 9 8 \right] \mathbf { 1 } [ \mathbf { u } _ { i } \in \mathcal { D } _ { 2 } ( k ( i ) ) ] , } \end{array}\tag{6}
$$

where $\mathcal { D } _ { 2 }$ dilates a component by two mask pixels. Components smaller than max $\left( 1 6 , \lceil 0 . 0 2 \lvert M _ { c } \rvert \rceil \right)$ are excluded. The certificate connects the observed bound instances to distinct class-mask components. These structural checks must pass for every instance before any axis of its class can be updated.

An additional guard $B _ { i }$ rejects boundary-truncated components and unresolved shortening behind another object’s box. The acceptance decision is

$$
\begin{array} { l } { \alpha _ { i } = G _ { c } B _ { i } \mathbf { 1 } [ \tilde { g } _ { i } \mathrm { r e a d a b l e } ] \mathbf { 1 } [ Q ( \tilde { g } _ { i } ; S _ { i } ) \geq q _ { 0 } ] } \\ { \cdot \mathbf { 1 } [ Q ( \tilde { g } _ { i } ; S _ { i } ) - Q ( g _ { i } ^ { 0 } ; S _ { i } ) \geq \Delta _ { q } ] , } \\ { g _ { i } = \left\{ \tilde { g } _ { i } , \quad \alpha _ { i } = 1 , \right. } \\ { g _ { i } ^ { 0 } , \quad \alpha _ { i } = 0 . } \end{array}\tag{7}
$$

Here $g _ { i } ^ { 0 }$ is the initial bound axis. Rejected updates retain the initial bound axis. The rule is shared across screw-bag, tape, rope, stick and stationery length readouts; other operands retain their original readers. Algorithm 1 gives the complete factconstruction procedure. The trace records both the initial witness link and any accepted mask readout or rejection reason.

## 3.5. Constraint verification and normal calibration

Discrete task requirements are encoded in the task’s executable constraint configuration and evaluated on the predicted

Algorithm 1. Binding and support-certified fact construction   
Input Decoded objects O, initial witnesses $E ,$ class masks $M _ { c } ;$   
$r = 0 . 0 8 , q _ { 0 } , \Delta _ { q } = 0 . 0 5 .$   
1 Set $C _ { i j } = d _ { i j }$ for same-class pairs with $d _ { i j } \leq r ;$ otherwise   
set $C _ { i j } = 1 0 ^ { \circ }$ . Add n dummy columns of cost $r + 1 0 ^ { - 7 }$   
2 π ← arg min<sub>π injective</sub> $\sum _ { i } C _ { i , \pi ( i ) }$ (Hungarian).   
3 Bind real matches to $g _ { i } ^ { 0 } ;$ mark dummy matches unavailable.   
Retain every decoded identity.   
4 for each mask-backed length class c with $I _ { c } \neq \emptyset$ do   
5 $S _ { i } \gets \{ \mathbf { p } \in M _ { c } : i = \arg \operatorname* { m i n } _ { h \in I _ { c } } \left\| \mathbf { p } - \mathbf { z } _ { h } \right\| _ { 2 } \} .$   
6 Where readable, compute $\tilde { g } _ { i }$ by Eq. (4) and both qualities   
by Eq. (5).   
7 Extract $K _ { c } ,$ dominant components $k ( i )$ and class certificate   
$G _ { c }$ by Eq. (6).   
8 Evaluate $B _ { i } ;$ accept $\tilde { g } _ { i }$ or retain $g _ { i } ^ { 0 }$ by Eq. (7). Record the   
support source and decision.   
9 end for   
10 $\begin{array} { r } { N _ { c } \gets \sum _ { i } { { \bf 1 } [ \hat { c } _ { i } = c ] } ; } \end{array}$ assemble $f _ { i } ~ = ~ ( \hat { c } _ { i } , p _ { i } , g _ { i } , a _ { i } )$ and   
source trace $\tau .$   
Output Facts ${ \mathcal { F } } _ { \mathrm { : } }$ , counts $\{ N _ { c } \}$ , measurement availability and evidence   
trace $\tau .$

facts. Continuous intervals $[ L _ { j } , U _ { j } ]$ and scales $b _ { j }$ are fitted from normal fitting/development observations. A measurement $m _ { j } ( x )$ yields

$$
r _ { j } ( x ) = \mathrm { m a x } \{ 0 , L _ { j } - m _ { j } ( x ) , m _ { j } ( x ) - U _ { j } \} / b _ { j } .\tag{8}
$$

Counts, lengths, locations and relations are reconstructed from the assembled objects, with category-specific postprocessing. Cached color, OCR, separator contrast and cable-centerline observations remain available to their task rules. Missing required geometry contributes an explicit unavailable-operand residual.

For normal fitting residual reference $R _ { t , j }$ , define

$$
P _ { t , j } ( r ) = \frac { 1 + \sum _ { v \in R _ { t , j } } \mathbf { 1 } [ v \geq r ] } { | R _ { t , j } | + 1 } ,\tag{9}
$$

$$
\begin{array} { r l r } {  { z _ { t , j } ( x ) = - \log P _ { t , j } ( r _ { j } ( x ) ) } } \\ & { } & { + \frac { \operatorname* { m a x } \{ 0 , r _ { j } ( x ) - \operatorname* { m a x } R _ { t , j } \} } { \operatorname* { m a x } \{ \mathrm { I Q R } ( R _ { t , j } ) , 1 \} } , } \\ & { } & { s _ { t } ( x ) = \operatorname* { m a x } z _ { t , j } ( x ) + 0 . 0 1 \sum _ { j } z _ { t , j } ( x ) . } \end{array}\tag{10}
$$

(11)

Independent normal calibration images $\mathcal { C } _ { t }$ fix

$$
\begin{array} { r l r } & { } & { \tau _ { t } = Q _ { . 9 5 } ^ { \mathrm { h i g h e r } } \{ s _ { t } ( x ) : x \in \mathcal { C } _ { t } \} , } \\ & { } & { \hat { y } _ { t } ( x ) = \mathbf { 1 } [ s _ { t } ( x ) > \tau _ { t } ] . } \end{array}\tag{12}
$$

The strict comparison determines the decision at score ties. Intervals, residual references and calibration thresholds are frozen before inspection.

## 3.6. Execution-grounded explanation

The executor records $\eta _ { j } ~ = ~ ( m _ { j } , \rho _ { j } , r _ { j } , z _ { j } , \mathcal { P } _ { j } )$ : observed operand, requirement, raw residual, normalized contribution and links to factual/visual sources. Counts cite decoded identities;

measured predicates also cite the bound witness and accepted readout. Missing operands retain their availability state.

For an alarm, a compact certificate identifies a sufficient subset of these executed contributions. Let

$$
S ( C ) = \operatorname* { m a x } _ { j \in C } z _ { j } + 0 . 0 1 \sum _ { j \in C } z _ { j } , \qquad S ( \emptyset ) = 0 .\tag{13}
$$

With $z _ { ( 1 ) } \geq \cdot \cdot \cdot \geq z _ { ( J ) } \geq 0$ , choose

$$
k ^ { * } = \operatorname* { m i n } \{ k : S ( C _ { k } ) > \tau _ { t } \} , \quad C _ { k } = \{ ( 1 ) , \ldots , ( k ) \} .\tag{14}
$$

The descending prefix maximizes both terms of Eq. (13) at each cardinality, making $C _ { k ^ { * } }$ a minimum-size sufficient certificate. Equal-score ties prefer available predicates and then key order. Omitted contributions are zeroed in this score-space definition.

A frozen language model realizes the selected predicate outcomes as source-attributed claims. For each selected record, its input contains the predicate name, operator, qualitative comparison and evidence ID. Names of unavailable operands are retained. Numeric operands, intervals and score comparisons remain in the executor’s evidence records. We use the existing Qwen3-VL-8B-Instruct checkpoint in text-only mode [38], with greedy decoding and frozen weights.

The model returns the recorded decision and claims paired with evidence IDs. A contract checks decision consistency, exact coverage of the selected and unavailable evidence, permitted IDs and explicit missingness; generated claims contain no numeric values. The executor attaches citations and renders the original numeric evidence beside the text. Outputs failing the contract retain a structured-trace report. The full trace preserves other deviations and unavailable operands, including in non-alarm reports. This connects fluent language to the same executed evidence used by the detector.

## 4. Experiments

## 4.1. Datasets and information sources

We evaluate five MVTec LOCO AD categories using normal images and logical anomalies, for 1,136 test images; structural anomalies are excluded. VID-AD contributes 7,895 images across ten scenarios and five capture conditions, giving all 50 deployment tasks [1, 2]. Parent-image roles and known crosscondition source groups are preserved.

Normal images have separate fitting, development and calibration roles. Fitting observations train the readers and provide residual references; development observations select checkpoints/readout settings and pad continuous intervals; calibration scores fix alarm thresholds. LOCO contains 1,183 fitting, 147 development and 305 calibration images. VID uses 1,000, 500 and 500 images in these roles, with ten calibration images per deployment task.

Component annotations supply class, center, box, axis and weak-mask targets for visual learning. They cover all LOCO fitting/development parents and 194 fitting/147 development VID parents. Factual edits of fitting parents carry updated targets and can produce logically abnormal configurations. Manually specified discrete task requirements configure the executor; continuous measurement intervals are fitted from normal observations. Calibration images are independent of network and association selection. Method development included inspection of the public test sets; the reported results use the final frozen configurations. Supplementary Table S1 gives the category-level budgets.

## 4.2. Implementation and comparisons

The visual frontend is trained for 40 epochs per category at 448 × 320 resolution, using AdamW [44]. Each semantic decoder adds 768 updates and 1,536 image exposures with the frontend frozen. At most 64 candidate queries and 64 grid queries pass through a width-64, four-head, two-layer decoder. Association is selected on 294 annotated normal development parents using the fixed point-F1/axis-error objective. The selected same-class radius is 0.08. The support-readout quality floor is fixed from 609 readable normal-development axes. The component and pixel tolerances are shared across categories; normal TRAIN/DEV/CAL data receive the same readout as test images. Architecture, optimization and readout details are provided in the supplementary material.

The external comparisons report the completed local deployments of SALAD, LogiCo, ComAD, PUAD, Qwen3-VL-8B and LogicAD-FE-fast. LOCO ComAD uses DINO-S8 clustering and CRF without component labels; its VID deployment uses a shared component-map frontend. Local PUAD omits the external ImageNet penalty-image term, and LogicAD-FE-fast uses a simplified program/frontend configuration. VID SALAD and PUAD use category-level calibration; SGDB uses task-level calibration. The supplementary comparison-condition table and the archive condition matrix specify their supervision, implementation and calibration settings. AUROC and fixed-threshold F1 are percentages, macro-averaged over five LOCO tasks or all 50 VID tasks.

Internal controls share normal-source partitions, componentsupervision budgets, the visual frontend and task requirements. Decoder variants use the same update budget. Binding/readout controls reuse frozen semantic outputs and visual witnesses. Each configuration fits its references and threshold from its own normal facts using the same scoring and calibration recipe; the readout study also fixes the initial references and thresholds to isolate measurement changes.

## 4.3. Detection performance

Tables 1 and 2 report complete category/scenario comparisons. SGDB achieves 98.19 AUROC and 94.52 F1 on LOCO logical, and 93.56 AUROC and 92.76 F1 on VID. Relative to the FMP plus guarded-geometry reference, the changes are +0.11/+0.98 percentage points on LOCO AUROC/F1 and +2.22/+3.30 on VID. The VID result reflects the complete set of capture conditions.

Figures 4 and 5 compare 20 LOCO and 30 VID sources. Source order is fixed lexicographically across categories and conditions. Each prediction strip shows the decision and its score/threshold comparison.

Table 1: LOCO logical detection results (%). Paired columns report AUROC and fixed-threshold F1. Mean averages five tasks. Supervision and implementation settings are specified in Section 4.2 and the supplement. Best/second-best distinct values are bold/underlined, with shared marks for ties. Blue columns denote SGDB.
<table><tr><td></td><td colspan="2">SALAD [4]</td><td colspan="2">LogiCo [13]</td><td colspan="2">ComAD [11]</td><td colspan="2">PUAD [20]</td><td colspan="2">[38]</td><td colspan="2">Qwen3-VL-8B LogicAD-FE-fast [8]</td><td colspan="2">SGDB (Ours)</td></tr><tr><td>Category</td><td>AUROC</td><td>F1</td><td>AUROC</td><td>F1</td><td>AUROC</td><td></td><td>F1 AUROC</td><td></td><td>F1</td><td>AUROC</td><td>F1</td><td>AUROC</td><td>F1</td><td>AUROC F1</td></tr><tr><td>Breakfast box</td><td>92.27</td><td>85.00</td><td>98.89</td><td>96.30</td><td>94.19</td><td>86.79</td><td></td><td>93.07 82.43</td><td></td><td>91.2171.64</td><td></td><td>52.60</td><td>17.65</td><td>98.78 98.78</td></tr><tr><td>Juice bottle</td><td>99.79</td><td>98.61</td><td>99.22</td><td>95.62</td><td></td><td>87.96 79.68</td><td>99.83</td><td>98.61</td><td></td><td>83.3161.46</td><td></td><td>53.67</td><td>1.40</td><td>96.38 96.00</td></tr><tr><td>Pushpins</td><td>100.00</td><td>75.21</td><td>98.37</td><td>93.55</td><td></td><td>87.11 78.53</td><td>86.69</td><td>69.44</td><td></td><td>62.8414.29</td><td></td><td>53.02</td><td>8.00</td><td>98.87 96.22</td></tr><tr><td>Screw bag</td><td>93.95</td><td>84.65</td><td>84.18</td><td>69.41</td><td>79.22</td><td>24.84</td><td>70.20</td><td>28.92</td><td></td><td>55.25 49.34</td><td></td><td>49.82</td><td>2.84</td><td>98.92 93.54</td></tr><tr><td>Splicing connectors</td><td>96.08</td><td>88.78</td><td>91.80</td><td>84.31</td><td>85.61</td><td>82.90</td><td>94.30</td><td>88.00</td><td></td><td>78.5656.95</td><td></td><td>65.70</td><td>8.85</td><td>97.97 88.08</td></tr><tr><td>Mean</td><td>96.42</td><td>86.45</td><td>94.49</td><td>87.84</td><td></td><td>86.82 70.55</td><td></td><td>88.8273.48</td><td></td><td>74.2350.74</td><td></td><td>54.96</td><td>7.75</td><td>98.19 94.52</td></tr></table>

Table 2: VID detection results by scenario (%). Paired columns report AUROC and fixed-threshold F1. Rows average five conditions; Mean averages all 50 tasks. Supervision and calibration settings are specified in Section 4.2 and the supplement. Best/second-best distinct values are bold/underlined; blue columns denote SGDB.
<table><tr><td rowspan="2"></td><td colspan="2">SALAD [4]</td><td colspan="2">LogiCo [13]</td><td colspan="2">ComAD [11]</td><td colspan="2">PUAD [20]</td><td colspan="2">Qwen3-VL-8B [38]</td><td colspan="2">LogicAD-FE-fast [8]</td><td colspan="2">SGDB (Ours)</td></tr><tr><td>AUROC</td><td>F1</td><td>AUROC</td><td>F1</td><td>AUROC</td><td></td><td>F1 AUROC</td><td></td><td>F1 AUROC</td><td>F1</td><td>AUROC</td><td>F1</td><td>AUROC</td><td>F1</td></tr><tr><td>Balls</td><td>23.76 58.06</td><td></td><td>58.88</td><td>61.45</td><td>65.8553.57</td><td></td><td>17.27</td><td>54.04</td><td>97.87</td><td>89.97</td><td>51.27</td><td>12.91</td><td>86.43</td><td>87.56</td></tr><tr><td>Blocks</td><td>90.92 84.62</td><td></td><td>94.37 90.98</td><td></td><td>77.9371.64</td><td></td><td>79.7074.55</td><td></td><td>92.93</td><td>90.37</td><td>48.78</td><td>14.69</td><td>89.15</td><td>86.98</td></tr><tr><td>Cookies</td><td>62.9780.26</td><td></td><td>52.3780.67</td><td></td><td>98.05 91.49</td><td></td><td>18.93 62.76</td><td></td><td>84.29</td><td>88.03</td><td>54.24</td><td>10.32</td><td>86.62</td><td>85.21</td></tr><tr><td>Dishes</td><td>52.3637.59</td><td></td><td>66.6173.74</td><td></td><td>73.9557.72</td><td></td><td>61.8443.60</td><td></td><td>99.64</td><td>95.39</td><td>80.55</td><td>77.45</td><td>92.90</td><td>96.63</td></tr><tr><td>Fruits</td><td>99.81</td><td>94.60</td><td>99.17</td><td>96.41</td><td>99.94 98.06</td><td></td><td>98.56 90.08</td><td></td><td>95.31</td><td>93.42</td><td>90.55</td><td>73.12</td><td>93.80</td><td>93.31</td></tr><tr><td>Ropes</td><td>43.00</td><td>3.67</td><td>31.93</td><td>5.26</td><td>80.2029.63</td><td></td><td>41.65</td><td>4.19</td><td>81.46</td><td>77.46</td><td>58.45</td><td>30.33</td><td>88.50</td><td>86.33</td></tr><tr><td>Stationery</td><td>51.07</td><td>76.85</td><td>62.29</td><td>76.16</td><td>65.61 58.43</td><td></td><td>42.39</td><td>63.48</td><td>71.91</td><td>24.55</td><td>55.90</td><td>16.98</td><td>100.00</td><td>100.00</td></tr><tr><td>Sticks</td><td>82.90</td><td>23.83</td><td>45.63</td><td>41.94</td><td>97.55</td><td>88.28</td><td>82.16</td><td>22.00</td><td>85.27</td><td>78.46</td><td>69.12</td><td>16.82</td><td>99.92</td><td>98.77</td></tr><tr><td>Tapes</td><td>100.00</td><td>92.08</td><td>98.65</td><td>90.34</td><td>100.00</td><td>94.35</td><td>100.00</td><td>88.69</td><td></td><td>92.9471.35</td><td>68.97</td><td>29.00</td><td>98.73</td><td>94.99</td></tr><tr><td>Tools</td><td>90.16 37.87</td><td></td><td>93.88</td><td>84.44</td><td>76.92 58.99</td><td></td><td>78.99</td><td>15.68</td><td>82.81</td><td>62.05</td><td>61.59</td><td>28.74</td><td>99.50</td><td>97.84</td></tr><tr><td>Mean</td><td>69.7058.94</td><td></td><td>70.38 70.14</td><td></td><td>83.6070.22</td><td></td><td>62.15 51.90</td><td></td><td>88.44 77.11</td><td></td><td>63.94</td><td>31.04</td><td>93.56</td><td>92.76</td></tr></table>

SGDB’s boxes and axes come from its support-checked associated objects; the semantic colors depict the frozen frontend witnesses. Baseline representations retain their original meanings: SALAD shows saved local appearance/composition maps and ComAD displays component regions.

The disaggregated results show complementary strengths. SGDB attains the highest displayed LOCO AUROC on screw bag and splicing connectors, while category leaders differ across the benchmark. On VID, support-certified readout raises Tapes from 89.43/85.27 to 98.73/94.99 AUROC/F1, alongside strong results for Stationery, Sticks and Ropes. The fixed-threshold normal FPR is 1.43% on LOCO and 11.36% on VID. Complete task-level results and condition summaries appear in the supplementary material.

## 4.4. Mechanism controls under shared supervision

Table 3(a) separates semantic decoding, initial witness readout and final support certification under the shared setup above. Direct evidence-edit DFD and its initial bound readout use the same decoder predictions and task requirements. Replacing direct geometry with bound witnesses raises LOCO AUROC/F1 from 95.42/92.41 to 98.19/94.52; screw-bag AUROC rises from 85.04 to 98.92. The certificate then improves VID from 92.63/91.79 to 93.56/92.76 while preserving LOCO performance. The plain-set hybrid provides a decoder control without the certificate.

Table 3(b) changes binding or unbound-object handling while retaining the certified readout. Nearest assignment allows witness reuse; the other controls remove class compatibility or distance gating, drop unbound identities, or tighten the radius. Removing class compatibility gives 92.60/91.38 on VID with its own references, versus 93.55/92.76 with shared SGDB references. Nearest assignment and removal of the distance gate leave detection metrics unchanged. Radius 0.04 gives 97.98/94.52 on LOCO and 93.97/93.11 on VID; radius 0.08 remains the normaldevelopment selection.

Table 3: Mechanism controls with shared component supervision and task requirements (%). (a) Decoder/readout configurations; the final row adds the support certificate. (b) Binding policies with the same complete-support readout. Each row uses its own normal references and CAL threshold under the shared recipe. Best/second-best distinct values are bold/underlined within each block and column; ties share marks.
<table><tr><td>LOCO logical VID all 50</td></tr><tr><td>Configuration AUROC F1 AUROC F1</td></tr><tr><td>(a) Semantic decoder and geometric readout</td></tr><tr><td>FMP reader 95.01 90.88 91.33 89.47</td></tr><tr><td>FMP + guarded geometry 98.08 93.54 91.33 89.47</td></tr><tr><td>Direct DFD: plain set 95.01 90.88 93.41 91.78 Direct DFD: evidence edit 95.42 92.41 92.64 91.83</td></tr><tr><td>Hybrid: plain set 98.02 93.54 93.38 91.75</td></tr><tr><td>Bound DFD: initial readout 98.19 94.52 92.63 91.79</td></tr><tr><td>SGDB: certified readout 98.1994.52 93.56 92.76</td></tr><tr><td></td></tr><tr><td>(b) Binding-policy controls</td></tr><tr><td>SGDB partial binding 98.19 94.52 93.56 92.76</td></tr><tr><td>Independent nearest witness 98.19 94.52 93.56 92.76</td></tr><tr><td>Without class compatibility 98.19 94.52 92.60 91.38</td></tr><tr><td>Without distance gate 98.19 94.52 93.56 92.76</td></tr><tr><td>Discard unbound identities 98.19 94.52 93.56 92.76</td></tr><tr><td>Tighter radius (0.04) 97.9894.52 93.97 93.11</td></tr></table>

Controlled removal of each matched witness on the 294 annotated normal development parents gives 3,136 binding-stage trials. The semantic identities and remaining witnesses stay fixed. Partial binding retains decoded counts and marks orphan geometry unavailable in every trial. Nearest assignment reuses another witness in 720 trials (22.96%); dropping unbound objects changes every trial’s count. This probe measures identity preservation, missing-evidence representation and witness reuse at the binding interface. Complete results appear in the supplementary material.

## 4.5. Support-certified readout

Table 4 changes only the axis-update rule, holding frozen semantic outputs, initial witnesses, masks and task requirements fixed. Uniform mask readout improves VID but lowers LOCO AUROC to 96.36. Quality selection alone can admit an incomplete, internally clean fragment. Complete component ownership retains all 90 tape corrections while preventing the screw degradations introduced by quality-only readout. The completesupport deployment corrects 90 prior errors and introduces one type-related tape miss, for 89 fewer errors across all 9,031 test images.

Table 4: Geometric-readout controls with fixed semantic outputs, witnesses, masks and task requirements (%). Normal references and thresholds follow the same fitting/calibration recipe. SGDB combines quality selection with complete instance support. Best/second-best distinct values are bold/underlined; ties share marks.
<table><tr><td>Readout</td><td>LOCO logical AUROC F1</td><td>VID all 50 AUROC</td><td>F1</td></tr><tr><td>Initial bound readout</td><td>98.19</td><td>92.63</td><td>91.79</td></tr><tr><td>Uniform mask readout</td><td>96.36</td><td>94.52 93.54 93.58</td><td>92.86</td></tr><tr><td>Quality selection only</td><td>96.62</td><td>93.82 93.54</td><td>92.76</td></tr><tr><td>SGDB: complete support</td><td>98.19</td><td>94.52 93.56</td><td>92.76</td></tr></table>

Fixing the initial intervals, references and thresholds gives the same 90 corrections and one new tape miss, attributing these transitions to measurement selection. The completeness check filters 20 of 23 quality-eligible screw axes and preserves all 259 screw decisions, scores and thresholds. The tape corrections cover 20 cross-condition filename-index groups. Support coverage and per-source transitions are provided in the supplementary material.

Paired resampling of 10,000 category–label-stratified groups gives V2-minus-initial VID gains of 0.93 AUROC points (95% interval 0.56–1.32) and 0.97 F1 points (0.57–1.43). The five capture conditions travel together within filename-index groups. V2’s LOCO AUROC advantages over uniform and quality-only readout are 1.83 points (1.18–2.51) and 1.57 points (0.96–2.22). Uniform readout retains a small VID F1 advantage: the V2- minus-uniform difference is −0.10 points (−0.20 to −0.02). The paired intervals and source-index transition ledgers in the supplement locate the benefit in preserving reliable geometry across categories. These conditional intervals use the existing, previously inspected test cohorts.

## 4.6. Execution-grounded explanation

We evaluate evidence selection on the same frozen V2 deployment, retaining its facts, references, thresholds and all 9,031 predictions. The 66,297 executed rule records provide operands and source links. Table 5 compares four policies on all 5,725 alarms. Sufficiency is the fraction whose selected contributions still exceed the original threshold under Eq. (13); mean size counts selected rule records.

Table 5: Explanation selection on 5,725 frozen alarms. Mean: selected rules per alarm; Suff.: alarm sufficiency (%); Fail: insufficient selections. All policies use the same trace and source links. Bold/underline mark best/second-best sufficiency and failure values; ties share marks.
<table><tr><td>Selection</td><td>Mean</td><td>Suff.↑</td><td>Fail↓</td></tr><tr><td>All executed rules</td><td>7.121</td><td>100.00</td><td>0</td></tr><tr><td>All positive-score rules</td><td>1.994</td><td>100.00</td><td>0</td></tr><tr><td>Maximum-score rule</td><td>1.000</td><td>99.70</td><td>17</td></tr><tr><td>Compact sufficient (SGDB)</td><td>1.006</td><td>100.00</td><td>0</td></tr></table>

The compact certificate retains sufficiency for every alarm with 1.006 rules on average, reducing the selected count by 49.56% relative to all positive-score rules. A single maximumscore rule is insufficient for 17 tape alarms: two strongest contributions give 4.125413, exactly the threshold, whereas three give 4.165858 and trigger the strict comparison. These are jointly sufficient score contributions, including the unavailablemeasurement term.

Reports preserve missing evidence in all 1,254 affected records, including 94 non-alarms. Raw predicate deviations occur in 191 non-alarm traces, so their reports distinguish deviation from an alarm. Source checks reproduce all scores and decisions; an independent calculation reconstructs 3,928 cited geometric operands with maximum absolute error $5 . 7 \times 1 0 ^ { - 1 4 }$

The frozen language model expresses selected predicate outcomes as source-attributed claims, paired with executor-rendered numeric evidence in Figs. 6 and 7. Per-category results, certificate checks and language-output consistency results are provided in the supplementary material.

## 4.7. Decision, explanation and evidence

Figures 6 and 7 render six reports with language-model claims and executor-owned numeric evidence. Each report pairs generated claims with the full score comparison, selected rule contributions and source-linked operands. The screw report selects the short-length predicate: 428.22 pixels exceeds the 363.12- pixel bound, giving a sufficient certificate score of 6.13006. The Tools report selects zero decoded bolts against a requirement of two. The blurred Stationery report cites its evaluated counts and retains the complete zero-residual trace. Cookies receives a zero-residual normal report despite its anomalous image label, exposing the available semantic facts behind the miss. Dishes cites the missing-spoon count while preserving unavailable utensil order. In Tapes, the accepted support readout changes the green/red length ratio from 2.053 to 0.980, below its interval; the report links that predicate to the two measured axes and their support records. Panel (g) of Fig. 7 additionally shows why the

![](images/4b95a94daeeebefb712508616442425d70b40b354f2f726a4b628ac15d42fba4.jpg)  
S: frozen image score; T: frozen threshold. Alarm if S > T. Green: correct decision; red: error.

Figure 4: LOCO comparison on 20 fixed sources: (a) logical anomalies; (b) normal references and an additional logical anomaly. GT is the provided logical-anomaly mask union. Prediction strips use full-precision scores; green/red denotes a correct/incorrect image-level decision. Spatial representations differ across methods.

joint tape alarm requires three score contributions rather than a single violated predicate.

<table><tr><td rowspan=3 colspan=1></td><td></td><td rowspan=1 colspan=1></td><td></td><td></td><td rowspan=3 colspan=1></td><td rowspan=3 colspan=1>io</td><td></td><td rowspan=3 colspan=1>8:</td><td rowspan=3 colspan=1></td><td rowspan=3 colspan=1></td><td rowspan=3 colspan=1>'</td><td rowspan=3 colspan=1></td><td rowspan=3 colspan=1></td><td rowspan=3 colspan=1></td><td rowspan=3 colspan=1></td></tr><tr><td></td><td rowspan=1 colspan=1></td><td></td><td></td><td></td></tr><tr><td></td><td rowspan=1 colspan=2>日</td><td></td><td></td></tr><tr><td rowspan=1 colspan=1>GT: Normal</td><td rowspan=1 colspan=3>GT: Normal</td><td rowspan=1 colspan=2>GT: Normal</td><td rowspan=1 colspan=1>GT: Normal</td><td rowspan=1 colspan=2>GT: Normal</td><td rowspan=1 colspan=1>GT: Normal</td><td rowspan=1 colspan=1>GT: Normal</td><td rowspan=1 colspan=1>GT: Normal</td><td rowspan=1 colspan=2>GT: Normal</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>GT: Normal</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3>：</td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>De°</td><td rowspan=1 colspan=2>二</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2><img src="images/df745bce48a98183cb3bb2bb52db2a684181ab16a77a96c5d6e75db009adbf5d.jpg"/></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3>S022/T0 003/1ml</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>日</td><td rowspan=1 colspan=3>田</td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>id</td><td></td><td rowspan=1 colspan=1>阳</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>'</td><td rowspan=1 colspan=2>日</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3></td><td rowspan=1 colspan=2>s 15.51/ T 3.76</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td rowspan=1 colspan=3></td><td rowspan=1 colspan=1>S -2.1971 2.79</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan=2 colspan=1></td><td rowspan=1 colspan=3></td><td rowspan=2 colspan=2></td><td rowspan=2 colspan=1></td><td></td><td rowspan=2 colspan=1>88</td><td rowspan=2 colspan=1></td><td rowspan=2 colspan=1></td><td rowspan=2 colspan=1></td><td rowspan=2 colspan=2>日</td><td></td><td rowspan=2 colspan=1>:</td></tr><tr><td rowspan=1 colspan=3></td><td></td><td></td></tr><tr><td rowspan=1 colspan=1>GT: Anomaly</td><td rowspan=1 colspan=3>GT: Anomaly</td><td rowspan=1 colspan=2>GT: Anomaly</td><td rowspan=1 colspan=1>GT: Anomaly</td><td></td><td rowspan=1 colspan=1>GT: Anomaly</td><td rowspan=1 colspan=1>GT: Anomaly</td><td rowspan=1 colspan=1>GT: Anomaly</td><td rowspan=1 colspan=1>GT: Anomaly</td><td rowspan=1 colspan=2>GT: Anomaly</td><td></td><td rowspan=1 colspan=1>GT: Anomaly</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>!</td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>二</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Pco</td><td rowspan=1 colspan=2></td><td></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2><img src="images/34e66f32a1f94b409b9d23ad0d4e4585801e20a534966a41e220608ea85d75c4.jpg"/></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=4></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>0</td><td></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>U</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2></td><td></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=2></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3></td><td rowspan=1 colspan=2>s 22.097 3.7.6</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=2></td></tr><tr><td rowspan=1 colspan=1>s5.81/ 1246</td><td rowspan=1 colspan=3>S 19.85/ 1 3.56</td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>s -2027 2.79</td><td rowspan=1 colspan=6>s6.36/1293 s050/13.9 s14.90/12.71 s134/T4.13</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>0</td><td></td><td rowspan=1 colspan=1>:8</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>日</td><td rowspan=1 colspan=2>1:8</td></tr><tr><td rowspan=1 colspan=1>GT: Anomaly</td><td rowspan=1 colspan=3>GT: Anomaly</td><td rowspan=1 colspan=2>GT: Anomaly</td><td rowspan=1 colspan=1>GT: Anomaly</td><td></td><td rowspan=1 colspan=1>GT: Anomaly</td><td rowspan=1 colspan=1>GT: Anomaly</td><td rowspan=1 colspan=1>GT: Anomaly</td><td rowspan=1 colspan=1>GT: Anomaly</td><td rowspan=1 colspan=2>GT: Anomaly</td><td rowspan=1 colspan=2>GT: Anomaly</td></tr><tr><td rowspan=1 colspan=1>品</td><td rowspan=1 colspan=3></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1></td><td></td><td rowspan=1 colspan=1>:</td><td rowspan=1 colspan=1>二</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>日</td><td rowspan=1 colspan=2></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1></td><td></td><td rowspan=1 colspan=1><img src="images/6fbbf8cd56eec4433d7b791bb7d4690afc8f3d7c7eb8102a8796c3a94d50d1c8.jpg"/></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=2></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3>出品</td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1></td><td></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>U</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>日</td><td rowspan=1 colspan=2></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1></td><td></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=2></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3>s 27.127 4.71</td><td rowspan=1 colspan=2>s 18.797 3.76</td><td rowspan=1 colspan=1></td><td></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=2></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3>s 1951/T 356</td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>s 125/ T 2 30</td><td></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1> -2 527 T 2.79</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>s 12507T271</td><td rowspan=1 colspan=2></td></tr></table>

S: frozen image score; T: frozen threshold. Alarm if S > T. Green: correct decision; red: error.

Figure 5: VID comparison on 30 fixed sources across ten scenarios: (a) original normal, (b) original anomalous, and (c) blurry anomalous images. GT is image-level.   
SGDB shows associated objects; ComAD shows component regions. SALAD and PUAD provide scored decisions from their cached outputs.

<table><tr><td>Rule contribution</td><td>obs./req.</td><td>Z</td><td>Two strongest contributions</td></tr><tr><td>E01 Green count</td><td>0/ 1</td><td>4.044522</td><td>4.125413 = T No alarm</td></tr><tr><td>E02 Red count</td><td>2/1</td><td>4.044522</td><td>Three contributions</td></tr><tr><td>E03 Missing geometry</td><td>yes / no</td><td>4.044522</td><td>4.165858 &gt; T Anomaly</td></tr></table>

Image

(a) Screw bag / length

![](images/2c39f553b5d915297d8c836e5fab4f590d1178d3410d227e0c78d352a45d60bb.jpg)

Evidence region

![](images/784076091580b29efc7452fc1ade299c55e0e8d8f4dac93881800ded9d0dde74.jpg)

## Execution-grounded report

Anomaly

[E05] The screw length exceeds the upper bound.

s = 6.13006 > τ = 4.43005

E05 Length 428.22 px; allowed [196.08, 363.12] px; source O4.

Subset E05; S(C) = 6.13006

GT: anomaly | correct

## (b) Tools / quantity

![](images/476af4cce1fdf6e30e837688a47dc79e1b239827c5b6afb9ab28b63699d17a29.jpg)

(c) Stationery / blur

![](images/472b9895d2cd4be57b595424a8c79209cb2b1610e37d3ef25408e0ce7ad23d60.jpg)

![](images/6a1735ea8f35baccee98dca160d18178cddd31cbe4cdeb3fd74267b04f72d829.jpg)

![](images/e2e64fdcf822a52efb294edcd48d5f4bfe2a28aee4a4e951fca6d529cd2a5741.jpg)

## Anomaly

[E01] The bolt count is below the required count.

E01 Bolt count: 0 observed / 2 required.

Subset E01; S(C) = 4.08497

s = 4.08497 > τ = 1.67481

## Normal

E01 Black pencil 1 / 1   
E03 Red pencil 1 / 1

[E01-E04] All required counts for black pencils, blue erasers, red pencils, and red erasers are satisfied.

GT: anomaly | correct

Count evidence (observed / required)

13 checks; 0 deviations

s = 0 ≤ τ = 0

E02 Blue eraser 1 / 1   
E04 Red eraser 1 / 1

GT: normal | correct

Figure 6: Execution-grounded reports: (a) screw-length alarm; (b) missing-bolt alarm; (c) normal Stationery under blur. Rows separate visual evidence, decisions, frozen-LLM claims and executor-owned measurements. An alarm requires $s > \tau ; S ( C )$ is the selected subset score. Evidence IDs are local to each case. GT label are used only for retrospective review.

Image

## (d) Cookies / quantity

![](images/3c989b644ece6120f74fb93d03790a9d28a0eaa9c39e555d086c325ddbda01bf.jpg)

## Execution-grounded report

![](images/303b2175dcccaf3bba57e510a8d45a85c947ad7d01d3024d6404f7afce420dad.jpg)

(e) Dishes / unavailable evidence

![](images/ca795104f98c06fe37cc5644b222d3cc662669994c3ba7fadbcfe5c58de9d06a.jpg)

![](images/214eda77015ccb1e5aa3ab05eb30ee11e5065443dca42dd658f9ec98ced3c4b1.jpg)

Count evidence (observed / required)

## Normal

[E01-E04] All required counts are satisfied: yellow cookies, black cookies, square dishes, and round dishes each meet their specified criteria.

E01 Yellow cookie 2 / 2   
E03 Square dish 1 / 1

9 checks; 0 deviations

(f) Tapes / certified measurement

E02 Black cookie 1 / 1   
E04 Round dish 1 / 1

s = 0 ≤ τ = 0

GT: anomaly | missed

(g) Tapes / jointly sufficient alarm

## Anomaly

![](images/e1c522a015d01b36a6a553a9c5e030ab16b8835efd94a2942f1f588afe0991e3.jpg)

s = 4.12541 > τ = 0

[E03] The spoon count is below the required count.   
[E04] The utensil order measurement is unavailable. E03 Spoon count: 0 observed / 1 required.   
E04 Utensil order: unavailable.

![](images/2550e552ea8e2798b2741b8b0e3c4612821b2976a3f99fb75c78fb91e13b1750.jpg)  
Figure 7: Reports for (d) a Cookies miss, (e) missing spoon with unavailable order, and (f) support-certified tape measurement. Panel (g) separates the rule contributions from their sufficient subsets: two contributions equal the threshold without triggering an alarm, whereas three exceed it. Columns and symbols follow Fig. 6; numeric evidence comes from the execution traces.

Subset E03; S(C) = 4.08497

Anomaly

GT: anomaly | correct

s = 7.53937 > τ = 4.12541

[E03] The green red length ratio is below the lower bound.

E03 Ratio 0.98034; allowed [1.8575, 2.0797]; sources O0, O1.

Subset E03; S(C) = 7.53937

GT: anomaly | correct

## 5. Discussion

SGDB connects semantic identities, geometric measurements and executable requirements through a common evidence representation. Bound witness geometry improves length verification over direct prediction; complete-support selection retains tape corrections and protects screw measurements. Partial binding preserves decoded counts, unavailable geometry and witness ownership. Execution-grounded explanation carries these source links into a compact account of the score comparison, including alarms that require several contributions.

## 6. Conclusion

SGDB decouples semantic recognition from geometric measurement and reconnects them through partial evidence binding and support-certified readout. The resulting fact and source trace supports both executable verification and compact sufficient alarm certificates. Reports connect selected predicates to object identities, measurements and evidence availability. SGDB achieves 98.19/94.52 AUROC/F1 on LOCO logical and 93.56/92.76 on all 50 VID tasks. Shared-supervision and frozenoutput controls examine the roles of geometric evidence, complete support and execution-grounded explanation.

Normal observations define measurement and score references, component annotations train the visual readers, and task requirements configure the executor. Extending this setup to unfamiliar scenes and reduced component supervision is a useful direction. The Cookies trace illustrates another role of the report: an anomalous image can yield zero-residual semantic facts, making the representation behind a missed alarm available for review.

## CRediT authorship contribution statement

Chenrui Ji: Conceptualization, Methodology, Software, Investigation, Writing – original draft. Jiafan Lu: Software, Validation, Visualization, Writing – review & editing. Zilu Luo: Validation, Writing – review & editing. Feng Wan: Formal analysis, Validation, Writing – review & editing. Qiuhui Chen: Conceptualization, Methodology, Supervision, Project administration, Writing – review & editing. Weimin Zhong: Supervision, Resources, Funding acquisition, Writing – review & editing.

## Data and code availability

MVTec LOCO AD and VID-AD are available from their original providers [1, 2]. The research code, frozen configurations, perimage outputs and CPU reproduction scripts are publicly available at https://github.com/Qybc/SGDB, release v1.0.0. The release supports metric reconstruction, execution-trace replay and recorded language-output verification. Benchmark images and third-party pretrained models are obtained from their original providers.

## Declaration of competing interests

The authors declare no competing interests.

## Funding

This work was supported in part by the Fundamental and Interdisciplinary Disciplines Breakthrough Plan of the Ministry of Education of China (Grant No. JYB2025XDXM402) and the National Natural Science Foundation of China (Grant Nos. 62633009 and 62603270).

Declaration of generative AI and AI-assisted technologies in manuscript preparation

Codex assisted with manuscript editing and LaTeX preparation.

## References

[1] P. Bergmann, K. Batzner, M. Fauser, D. Sattlegger, C. Steger, Beyond Dents and Scratches: Logical Constraints in Unsupervised Anomaly Detection and Localization, International Journal of Computer Vision 130 (4) (2022) 947–969. doi: 10.1007/s11263-022-01578-9.

[2] H. Nakata, Y. Zou, S. Sakai, S. Maeda, C. Gu, Y. Wei, S. Gao, C. Zhang, VID-AD: A Dataset for Image-Level Logical Anomaly Detection under Vision-Induced Distraction, arXiv preprint arXiv:2603.13964 (2026). doi:10.48550/arXiv. 2603.13964.

[3] K. Roth, L. Pemula, J. Zepeda, B. Schölkopf, T. Brox, P. Gehler, Towards Total Recall in Industrial Anomaly Detection, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 14318–14328.

[4] M. Fucka, V. Zavrtanik, D. Skoˇ caj, SALAD – Semantics-Awareˇ Logical Anomaly Detection, in: Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025, pp. 21843– 21852.

[5] S. Kim, S. An, P. Chikontwe, M. Kang, E. Adeli, K. M. Pohl, S. H. Park, Few Shot Part Segmentation Reveals Compositional Logic for Industrial Anomaly Detection, in: Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 38, 2024, pp. 8591–8599. doi:10.1609/aaai.v38i8.28703.

[6] Y. Peng, X. Lin, N. Ma, J. Du, C. Liu, C. Liu, Q. Chen, SAM-LAD: Segment Anything Model meets zero-shot logic anomaly detection, Knowledge-Based Systems 314 (2025) 113176. doi: 10.1016/j.knosys.2025.113176.

[7] M. Fucka, V. Zavrtanik, D. Skoˇ caj, ObjectCore – Efficient Few-ˇ shot Logical Anomaly Detection using Object Representations, in: Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, 2026, pp. 3857–3867.

[8] E. Jin, Q. Feng, Y. Mou, G. Lakemeyer, S. Decker, O. Simons, J. Stegmaier, LogicAD: Explainable Anomaly Detection via VLMbased Text Feature Extraction, in: Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 39, 2025, pp. 4129– 4137. doi:10.1609/aaai.v39i4.32433.

[9] Y. Zhang, Y. Cao, X. Xu, W. Shen, LogiCode: An LLM-Driven Framework for Logical Anomaly Detection, IEEE Transactions on Automation Science and Engineering 22 (2025) 7712–7723. doi:10.1109/TASE.2024.3468464.

[10] X. Zhang, X. Zhou, Z. Deng, D. Geng, J. Wang, Interpretable Logical Anomaly Classification via Constraint Decomposition and Instruction Fine-Tuning, arXiv preprint arXiv:2602.03530 (2026). doi:10.48550/arXiv.2602.03530.

[11] T. Liu, B. Li, X. Du, B. Jiang, X. Jin, L. Jin, Z. Zhao, Componentaware anomaly detection framework for adjustable and logical industrial visual inspection, Advanced Engineering Informatics 58 (2023) 102161. doi:10.1016/j.aei.2023.102161.

[12] Y.-H. Hsieh, S.-H. Lai, CSAD: Unsupervised Component Segmentation for Logical Anomaly Detection, in: Proceedings of the British Machine Vision Conference, BMVA, 2024.

[13] X. Zhang, M. Xu, X. Zhou, LogiCo: A Unified Framework for Logical and Structural Anomaly Detection, in: Computer Vision – ECCV 2026, Springer, 2026, pp. 244–261. doi:10.1007/ 978-3-032-37550-6\_14.

[14] P. Bergmann, K. Batzner, M. Fauser, D. Sattlegger, C. Steger, The MVTec Anomaly Detection Dataset: A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection, International Journal of Computer Vision 129 (4) (2021) 1038–1059. doi: 10.1007/s11263-020-01400-4.

[15] Y. Zou, J. Jeong, L. Pemula, D. Zhang, O. Dabeer, SPot-the-Difference Self-supervised Pre-training for Anomaly Detection and Segmentation, in: Computer Vision – ECCV 2022, Springer, 2022, pp. 392–408. doi:10.1007/978-3-031-20056-4\_23.

[16] T. Defard, A. Setkov, A. Loesch, R. Audigier, PaDiM: A Patch Distribution Modeling Framework for Anomaly Detection and Localization, in: Pattern Recognition. ICPR International Work-

shops and Challenges, Springer, 2021, pp. 475–489. doi: 10.1007/978-3-030-68799-1\_35.

[17] H. Deng, X. Li, Anomaly Detection via Reverse Distillation From One-Class Embedding, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 9737–9746.

[18] K. Batzner, L. Heckler, R. König, EfficientAD: Accurate Visual Anomaly Detection at Millisecond-Level Latencies, in: Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, 2024, pp. 128–138.

[19] V. Zavrtanik, M. Kristan, D. Skocaj, DRAEM – A Discrimina-ˇ tively Trained Reconstruction Embedding for Surface Anomaly Detection, in: Proceedings of the IEEE/CVF International Conference on Computer Vision, 2021, pp. 8330–8339.

[20] S. Sugawara, R. Imamura, PUAD: Frustratingly Simple Method for Robust Anomaly Detection, in: IEEE International Conference on Image Processing, 2024, pp. 842–848. doi:10.1109/ ICIP51287.2024.10647438.

[21] Y. Hu, H. Lin, H. Huang, Y. Guo, F. Wu, J. Long, J. Han, Unsupervised logical anomaly detection in industry: Mamba optimized by nested pyramid fusion and parameter-shared state space enhancement, Neurocomputing 674 (2026) 132824. doi: 10.1016/j.neucom.2026.132824.

[22] Z. Zhang, W. Zhao, Y. Zhang, E. Zio, Z. Zhang, A unified framework for image anomaly detection via reconstruction, segmentation and spatial relationship modeling, Neurocomputing 685 (2026) 133581. doi:10.1016/j.neucom.2026.133581.

[23] X. Tong, Y. Chang, Q. Zhao, J. Yu, B. Wang, J. Lin, Y. Lin, X. Mai, H. Wang, Z. Tao, Y. Wang, W. Zhang, Component-Aware Unsupervised Logical Anomaly Generation for Industrial Anomaly Detection, in: IEEE International Conference on Robotics and Automation, 2025, pp. 16722–16729. doi: 10.1109/ICRA55743.2025.11127296.

[24] A. Ascarate, L. Lebrat, R. Santa Cruz, C. Fookes, O. Salvado, When Rule Violations Are Rare: Chimera Training for Logical Anomaly Detection, arXiv preprint arXiv:2605.26171 (2026). doi:10.48550/arXiv.2605.26171.

[25] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, I. Polosukhin, Attention Is All You Need, in: Advances in Neural Information Processing Systems, Vol. 30, 2017.

[26] N. Carion, F. Massa, G. Synnaeve, N. Usunier, A. Kirillov, S. Zagoruyko, End-to-End Object Detection with Transformers, in: Computer Vision – ECCV 2020, Springer, 2020, pp. 213–229. doi:10.1007/978-3-030-58452-8\_13.

[27] X. Zhu, W. Su, L. Lu, B. Li, X. Wang, J. Dai, Deformable DETR: Deformable Transformers for End-to-End Object Detection, in: International Conference on Learning Representations, 2021.

[28] S. Liu, F. Li, H. Zhang, X. Yang, X. Qi, H. Su, J. Zhu, L. Zhang, DAB-DETR: Dynamic Anchor Boxes are Better Queries for DETR, in: International Conference on Learning Representations, 2022.

[29] I. T. Jolliffe, J. Cadima, Principal component analysis: a review and recent developments, Philosophical Transactions of the Royal Society A: Mathematical, Physical and Engineering Sciences 374 (2065) (2016) 20150202. doi:10.1098/rsta.2015.0202.

[30] J. Jeong, Y. Zou, T. Kim, D. Zhang, A. Ravichandran, O. Dabeer, WinCLIP: Zero-/Few-Shot Anomaly Classification and Segmentation, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 19606–19616.

[31] Q. Zhou, G. Pang, Y. Tian, S. He, J. Chen, AnomalyCLIP: Objectagnostic Prompt Learning for Zero-shot Anomaly Detection, in: International Conference on Learning Representations, 2024.

[32] Z. Gu, B. Zhu, G. Zhu, Y. Chen, M. Tang, J. Wang, AnomalyGPT: Detecting Industrial Anomalies Using Large Vision-Language Models, in: Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 38, 2024, pp. 1932–1940. doi:10.1609/aaai. v38i3.27963.

[33] T. Gupta, A. Kembhavi, Visual Programming: Compositional Visual Reasoning Without Training, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 14953–14962.

[34] R. Manhaeve, S. Dumanciˇ c, A. Kimmig, T. Demeester, ´ L. De Raedt, DeepProbLog: Neural Probabilistic Logic Program-

ming, in: Advances in Neural Information Processing Systems, Vol. 31, 2018.

[35] S. Badreddine, A. d’Avila Garcez, L. Serafini, M. Spranger, Logic Tensor Networks, Artificial Intelligence 303 (2022) 103649. doi: 10.1016/j.artint.2021.103649.

[36] M. Dahmardeh, M. Saadatpour, F. Manigrasso, L. Morra, F. Setti, NeSyLAD: A Neuro-Symbolic Approach for Unsupervised Logical Anomaly Detection, in: Image Analysis and Processing – ICIAP 2025, Springer Nature Switzerland, 2026, pp. 598–610. doi:10.1007/978-3-032-10192-1\_50.

[37] Qwen Team, Qwen2.5-VL-3B-Instruct: Official Model Card, Model documentation, accessed 24 October 2026 (2025). URL https://huggingface.co/Qwen/Qwen2. 5-VL-3B-Instruct

[38] S. Bai, Y. Cai, R. Chen, K. Chen, X. Chen, Z. Cheng, et al., Qwen3-VL Technical Report, arXiv preprint arXiv:2511.21631 (2025). doi:10.48550/arXiv.2511.21631.

[39] S. Bassan, R. Eliav, S. Gur, Explain Yourself, Briefly! Self-Explaining Neural Networks with Concise Sufficient Reasons, in: International Conference on Learning Representations, 2025.

[40] B. Bharti, P. Yi, J. Sulam, Sufficient and Necessary Explanations (and What Lies in Between), in: Conference on Parsimony and Learning, Vol. 280 of Proceedings of Machine Learning Research, 2025, pp. 1191–1215.

[41] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, L. Fei-Fei, ImageNet: A large-scale hierarchical image database, in: 2009 IEEE Conference on Computer Vision and Pattern Recognition, 2009, pp. 248–255. doi:10.1109/CVPR.2009.5206848.

[42] S. Zagoruyko, N. Komodakis, Wide Residual Networks, in: Proceedings of the British Machine Vision Conference, BMVA Press, 2016, pp. 87.1–87.12. doi:10.5244/C.30.87.

[43] X. Zhou, D. Wang, P. Krähenbühl, Objects as Points, arXiv preprint arXiv:1904.07850 (2019). doi:10.48550/arXiv. 1904.07850.

[44] I. Loshchilov, F. Hutter, Decoupled Weight Decay Regularization, in: International Conference on Learning Representations, 2019.

# Supplementary material

and Evidence Binding for Logical Anomaly Detection

Chenrui Ji, Jiafan Lu, Zilu Luo,

Feng Wan, Qiuhui Chen, Weimin Zhong

## S1 Data roles and implementation

## S1.1 Information sources and normal-data budgets

SGDB uses normal images, component targets and explicit task requirements for distinct parts of inspection. Normal fitting images supply the original visual training sources and residual references. Normal development images select reader checkpoints, association/readout settings and continuous-interval padding. Independent normal calibration images fix the alarm thresholds. Same-parent edits inherit their source role, and known cross-condition source groups are preserved.

Component targets provide class, center, box, axis and weak-mask supervision for the visual readers. Table S1 distinguishes the full normal pools from the parents carrying these targets. At inspection, objects and masks are predicted from the query image. Discrete task requirements are encoded as executable constraints on the resulting facts; continuous bounds and residual scales are estimated from normal observations. The task identifier selects the relevant readers, constraints and normal references.

Table S1: Normal-source budgets and component-supervised parents. Ann.: annotated with component facts and weak masks.
<table><tr><td>Category</td><td>Fitting</td><td>Dev.</td><td>Cal.</td><td>Ann. fit.</td><td>Ann. dev.</td><td>Test</td></tr><tr><td>Breakfast Box</td><td>234</td><td>29</td><td>62</td><td>234</td><td>29</td><td>185</td></tr><tr><td>Juice Bottle</td><td>224</td><td>27</td><td>54</td><td>224</td><td>27</td><td>236</td></tr><tr><td>Pushpins</td><td>248</td><td>31</td><td>69</td><td>248</td><td>31</td><td>229</td></tr><tr><td>Screw Bag</td><td>240</td><td>30</td><td>60</td><td>240</td><td>30</td><td>259</td></tr><tr><td>Splicing Connectors</td><td>237</td><td>30</td><td>60</td><td>237</td><td>30</td><td>227</td></tr><tr><td>Balls</td><td>100</td><td>50</td><td>50</td><td>20</td><td>15</td><td>790</td></tr><tr><td>Blocks</td><td>100</td><td>50</td><td>50</td><td>20</td><td>15</td><td>800</td></tr><tr><td>Cookies</td><td>100</td><td>50</td><td>50</td><td>18</td><td>15</td><td>780</td></tr><tr><td>Dishes</td><td>100</td><td>50</td><td>50</td><td>20</td><td>15</td><td>805</td></tr><tr><td>Fruits</td><td>100</td><td>50</td><td>50</td><td>20</td><td>15</td><td>750</td></tr><tr><td>Ropes</td><td>100</td><td>50</td><td>50</td><td>16</td><td>12</td><td>800</td></tr><tr><td>Stationery</td><td>100</td><td>50</td><td>50</td><td>20</td><td>15</td><td>800</td></tr><tr><td>Sticks</td><td>100</td><td>50</td><td>50</td><td>20</td><td>15</td><td>770</td></tr><tr><td>Tapes</td><td>100</td><td>50</td><td>50</td><td>20</td><td>15</td><td>800</td></tr><tr><td>Tools</td><td>100</td><td>50</td><td>50</td><td>20</td><td>15</td><td>800</td></tr></table>

## S1.2 Architecture and optimization

The visual frontend uses a WideResNet50-2 encoder truncated after the third residual stage. Global-context modulation, multiscale upsampling/skip fusion and explicit normalized coordinates feed center, semantic-mask, box and endpoint heads. Training uses

$$
\mathcal { L } _ { F } = \mathcal { L } _ { \mathrm { c e n t e r } } + 3 \mathcal { L } _ { \mathrm { s e g } } + 3 \mathcal { L } _ { \mathrm { b o x } } + 3 \mathcal { L } _ { \mathrm { a x i s } } .\tag{S1}
$$

The segmentation term combines weighted binary cross-entropy and Dice loss on valid weakmask channels. Endpoint order is handled by the minimum loss over the two permutations. Category frontends receive 40 epochs with 240 sampled examples per epoch and batch size four. AdamW uses encoder/head learning rates $1 0 ^ { - 4 } / 1 0 ^ { - 3 }$ , weight decay $1 0 ^ { - 4 }$ and a cosine schedule. The encoder is frozen for the first two epochs and batch-normalization statistics stay frozen. The frontend seed is 42.

The DFD has 64 candidate slots and 64 grid slots, width 64, four attention heads and two layers. Grid slots use a fixed $8 \times 8$ layout. Spatially pooled feature memory has a $4 0 \times 5 6$ grid. First-layer soft support masks add a log-sigmoid bias to second-layer attention. The main decoder objective is

$$
\mathcal { L } _ { D } = \mathcal { L } _ { \mathrm { c l s } } + 2 \mathcal { L } _ { \mathrm { c t r } } + 2 \mathcal { L } _ { \mathrm { b o x } } + 3 \mathcal { L } _ { \mathrm { a x i s } } + \mathcal { L } _ { \mathrm { m a s k } } + 0 . 2 5 \mathcal { L } _ { \mathrm { b i n d } } + 0 . 2 5 \mathcal { L } _ { \mathrm { e d i t } } .\tag{S2}
$$

Support coupling aligns geometric predictions with mask support. Paired-edit consistency uses count changes and shared-object displacement, with duplicate source-query perturbations. The plain-set control omits mask feedback and the coupling/consistency mechanisms. Each head receives 768 optimizer steps and 1,536 image exposures. Fifteen categories are trained for each of the two variants.

Fitting data comprise normal-source parents and paired edits with updated factual targets. An edit can produce a logically abnormal configuration; supervision remains component/fact based. The decoder’s geometric predictions support its training objective, while final hybrid measurements use associated witnesses.

## S2 Association and geometric-readout selection

## S2.1 Normal-development association

Four candidates combine same-class or any-class eligibility with radii 0.04 or 0.08. Selection uses 294 annotated original normal development parents across 15 categories. For each category, the objective is

$$
0 . 8 F _ { \mathrm { 1 , p o i n t } } + 0 . 2 ( 1 - E _ { \mathrm { a x i s } } ) ,\tag{S3}
$$

where relative matched-axis error is clipped to [0, 1]. The category objectives are macroaveraged. Unbound recognized objects count as false positives and unmatched reference objects as false negatives. Ties follow the preset candidate order. The same selected scheme is applied to both datasets and decoder variants.

Table S2: Complete normal-development association selection. Point F1 and clipped axis error are category macro-averages (%). Unbound is a count.
<table><tr><td>Candidate</td><td>Point F1</td><td>Axis error</td><td>Unbound</td><td>Objective</td></tr><tr><td>Same class, 0.04</td><td>99.10</td><td>8.78</td><td>13</td><td>0.97528</td></tr><tr><td>Same class, 0.08</td><td>99.68</td><td>8.76</td><td>0</td><td>0.97990</td></tr><tr><td>Any class, 0.04</td><td>99.35</td><td>10.07</td><td>6</td><td>0.97466</td></tr><tr><td>Any class, 0.08</td><td>99.68</td><td>10.10</td><td>0</td><td>0.97721</td></tr></table>

The selected same-class radius 0.08 has 3,136 decoded development objects and no unbound objects. The main hybrid’s test set has one unbound object in one breakfast-box image. The null policy retains that identity in the count and leaves its geometry-dependent operands unavailable.

## S2.2 Initial screw-mask ownership and native axes

For instance i, predicted box width and height $\mathbf { w } _ { i }$ are lower-bounded by 0.02. Foreground pixels are eligible within the box expanded by 12.5% on each side. Among eligible instances, a pixel p is assigned by minimizing

$$
d _ { i } ( \mathbf { p } ) = \| ( \mathbf { p } - \mathbf { u } _ { i } ) \oslash \mathbf { w } _ { i } \| _ { 2 } ^ { 2 } .\tag{S4}
$$

Unsupported pixels are excluded. Assigned pixels are mapped to native coordinates before PCA, and the selected reader uses untrimmed projection endpoints.

The substitution requires at least 16 assigned pixels, 75% expanded-box support, normalized centroid displacement at most 0.04, principal-to-secondary eigenvalue ratio at least four, and finite endpoints. Otherwise the original FMP axis is retained. These fallback axes remain valid witnesses for subsequent identity association. A dummy semantic-to-witness match instead produces null geometry.

The readout candidates combine connected-component or box-supported assignment with zero or 1% projection trimming. Selection uses 30 normal development screw-bag images with

Table S3: Native-axis error on 60 normal development screw instances. Relative errors are percentages; Large counts errors above 20% and Updates counts substituted axes. Reference axes are stored normal component targets.
<table><tr><td>Readout</td><td>Mean</td><td>Median</td><td>P95</td><td>Large</td><td>Updates</td></tr><tr><td>Predicted endpoints</td><td>11.96</td><td>4.96</td><td>53.00</td><td>8.00</td><td>0</td></tr><tr><td>Connected components, untrimmed</td><td>8.30</td><td>1.61</td><td>53.00</td><td>6.00</td><td>32</td></tr><tr><td>Connected components, 1% trimming</td><td>9.56</td><td>3.53</td><td>53.00</td><td>6.00</td><td>32</td></tr><tr><td>Box-supported assignment, untrimmed</td><td>2.50</td><td>0.83</td><td>9.36</td><td>0.00</td><td>60</td></tr><tr><td>Box-supported assignment, 1% trimming</td><td>4.68</td><td></td><td>3.54 12.96</td><td>0.00</td><td>60</td></tr></table>

60 screw instances. Admission requires at least a 10% mean-error reduction, no regression in median, P95 or errors above 20%, and at least one valid substitution. Candidate order resolves remaining ties after mean and P95 error.

## S2.3 Post-binding support certification

The initial witnesses above are bound to decoded identities before the final readout. For screw bags, tapes, ropes, sticks and stationery, same-class foreground pixels are reassigned to the nearest bound center using Euclidean distance in native coordinates, without clipping to boxes. At least 16 assigned pixels are required. PCA directions use native-coordinate covariance; endpoints use the 1st and 99th projection percentiles.

Quality is evaluated for the initial and candidate axes on the same assigned support. $P _ { \mathrm { l i n e } }$ samples 64 equally spaced axis points and measures membership in the assigned binary mask dilated once with a $3 \times 3$ kernel. $P _ { \mathrm { e x t e n t } }$ measures how many assigned pixels project inside the axis interval padded by one native mask-pixel diagonal. $P _ { \mathrm { c o n n } }$ is the largest 8- connected assigned-region area divided by all assigned pixels. The fixed quality floor is 0.9908659040240699, the tenth percentile over 609 readable normal-development candidates; the required gain is 0.05.

Substantial components of the full class mask have at least max(16, $\left\lceil 0 . 0 2 | M _ { c } | \right\rceil$ pixels. Their count must equal the number of currently bound recognized instances, and dominant components must be distinct. Each instance must retain at least 98% of its full dominant component and its frozen center must lie within a two-mask-pixel dilation of that component. Structural failure rejects every proposed update in that class. These checks use observed identities, without replacing them by normal expected counts.

Table S4: Certified readout coverage on the 3,429 test images with length readouts. Axes: bound target axes; Quality: candidates passing the quality gate; Accepted: candidates passing the complete certificate.
<table><tr><td>Category</td><td>Axes</td><td>Quality</td><td>Accepted</td></tr><tr><td>Screw bag</td><td>490</td><td>23</td><td>3</td></tr><tr><td>Tapes</td><td>1,911</td><td>200</td><td>172</td></tr><tr><td>Ropes</td><td>1,627</td><td>9</td><td>0</td></tr><tr><td>Sticks</td><td>2,372</td><td>6</td><td>1</td></tr><tr><td>Stationery</td><td>3,202</td><td>2</td><td>0</td></tr></table>

The boundary guard rejects a dominant component touching the image edge or its adjacent one-pixel band. If a candidate shortens the initial axis by more than one native mask-pixel diagonal, 64 initial-axis samples are projected onto the candidate direction. Samples outside the padded candidate interval, together with its two endpoints, are tested against every other recognized object’s box expanded by one mask pixel. Any intersection rejects the update. The rejection retains the initial axis, its witness link and availability; it does not delete an object or supply an expected normal length. Accepted updates retain the original identity-to-witness link and additionally record the mask source, assigned support, candidate quality and gate results.

The tolerances were fixed before the completeness experiment. The same reader is applied to normal fitting/development/calibration and test images, without category-specific switches or selection by proximity to a normal length interval. The development setting is summarized in Sec. 4.1 of the main paper.

The certificate updates 176 axes in 175 test images. With own normal references or the original bound deployment’s references shared, all final decisions are identical between the two modes. The tape repair set is the same 90 images, corresponding to 20 cross-condition filename-index groups, with one additional type-related miss. All screw-bag predictions and normal configuration equal the initial bound deployment. The complete-support and wholecomponent-only variants yield exactly the same final facts and predictions, so the extra boundary/shortening guards have no measured independent benefit here.

## S3 Frozen-output binding-policy ablations

## S3.1 Policies and evaluation modes

All six policies reuse the evidence-edit decoder’s frozen semantic outputs and the same initial FMP/guarded-screw witnesses, followed by the fixed complete-support readout. The main policy is the previously selected same-class, one-to-one assignment at radius 0.08. Independent nearest assignment uses the same eligibility test but allows several decoded objects to use one witness. The any-class control removes only class compatibility. The ungated control allows every same-class pair, with dummy cost ${ \sqrt { 2 } } + 1 0 ^ { - 7 }$ , so only witness scarcity requires an unbound object. The drop-unbound control first applies the main assignment and then discards objects assigned to a dummy. The radius control uses 0.04 with all other main-policy choices retained. The six binding policies and all readout tolerances are retained without a new search.

For each policy, original normal fitting and development parents rebuild the intervals and residual references; independent calibration parents set the 95th-percentile threshold using the same strict comparison. Auxiliary models, source roles, task rules and the score formula are unchanged. A second diagnostic applies the original SGDB references and thresholds to every policy’s facts. This separates a changed fact interface from the additional efect of normal-reference fitting and calibration. No network forward, optimizer update or baseline execution is performed.

The class-agnostic policy gives 92.60/91.38 VID AUROC/F1 with its own normal references and 93.55/92.76 with shared SGDB references. The tighter radius gives 93.97/93.11 on VID and 97.98/94.52 on LOCO. These controls retain the original normal-development choice of radius 0.08.

Table S5: Binding policies with their own normal references and CAL thresholds, and with SGDB’s references/thresholds shared (%). Each four-column group reports LOCO AUROC/F1, then VID AUROC/F1.
<table><tr><td></td><td colspan="6">Own normal references Shared SGDB references</td></tr><tr><td></td><td colspan="4">LOCO</td><td colspan="4">LOCO</td></tr><tr><td>Policy</td><td>AUC</td><td>F1 AUC</td><td>VID</td><td></td><td>F1 AUC</td><td></td><td>VID F1 AUC</td><td>F1</td></tr><tr><td>SGDB partial binding</td><td>98.1994.5293.5692.7698.19</td><td></td><td></td><td></td><td></td><td>94.52 93.56</td><td></td><td>92.76</td></tr><tr><td>Independent nearest witness</td><td>98.19 94.52</td><td></td><td>93.56 92.76</td><td></td><td>98.19</td><td>94.52</td><td>93.56</td><td>92.76</td></tr><tr><td>Without class compatibility</td><td>98.19 94.52</td><td></td><td>92.60</td><td>91.38</td><td>98.19</td><td>94.52</td><td>93.55</td><td>92.76</td></tr><tr><td>Without distance gate</td><td>98.1994.52</td><td></td><td>93.56</td><td>92.76</td><td>98.19</td><td>94.52</td><td>93.56</td><td>92.76</td></tr><tr><td>Discard unbound identities</td><td>98.19 94.52</td><td></td><td>93.56 92.76</td><td></td><td>98.19</td><td>94.52</td><td>93.56</td><td>92.76</td></tr><tr><td>Tighter radius (0.04)</td><td>97.98 94.52 93.97 93.11 97.00 92.81 93.66</td><td></td><td></td><td></td><td></td><td></td><td></td><td>93.01</td></tr></table>

Table S6: Binding traces on all 9,031 test images and 51,158 decoded objects. Unbound counts decoded objects without a witness; Dropped counts identities removed from the assembled facts. Reuse counts assignments beyond one use per witness. Cross-class counts links with diferent decoded/witness labels.
<table><tr><td>Policy</td><td>Unbound</td><td>Dropped</td><td>Reuse</td><td>Cross-class</td></tr><tr><td>SGDB partial binding</td><td>1</td><td>0</td><td>0</td><td>0</td></tr><tr><td>Independent nearest witness</td><td>0</td><td>0</td><td>3</td><td>0</td></tr><tr><td>Without class compatibility</td><td>0</td><td>0</td><td>0</td><td>309</td></tr><tr><td>Without distance gate</td><td>1</td><td>0</td><td>0</td><td>0</td></tr><tr><td>Discard unbound identities</td><td>1</td><td>1</td><td>0</td><td>0</td></tr><tr><td>Radius 0.04</td><td>423</td><td>0</td><td>0</td><td>0</td></tr></table>

Independent nearest assignment changes 3 test scores and 0 decisions. Dropping unbound identities changes 1 scores and 0 decisions. Removing the distance gate changes 0 scores. Binding-trace counts are unchanged by the downstream axis readout.

## S3.2 Controlled missing-witness probe

The binding-stage probe, before the downstream support readout, uses all 294 annotated original normal development parents. For every originally matched decoded object, its assigned witness is removed once, producing 3,136 trials. The image, semantic output, remaining witnesses and their measurements are unchanged. Every policy is then reapplied. These modified witness sets are used only to inspect binding behavior; they do not fit normal references or thresholds, select a configuration, or represent additional real anomaly images.

Table S7: Controlled witness removal over 3,136 trials (%). Count kept: all decoded identities remain counted. Null orphan: the decoded object whose original witness was removed remains present with unavailable geometry. Reuse trials: at least one witness is assigned to several objects.
<table><tr><td>Policy</td><td>Count kept</td><td>Null orphan</td><td>Reuse trials</td></tr><tr><td>SGDB partial binding</td><td>100.00</td><td>100.00</td><td>0.00</td></tr><tr><td>Independent nearest witness</td><td>100.00</td><td>77.04</td><td>22.96</td></tr><tr><td>Without class compatibility</td><td>100.00</td><td>96.94</td><td>0.00</td></tr><tr><td>Without distance gate</td><td>100.00</td><td>99.43</td><td>0.00</td></tr><tr><td>Discard unbound identities</td><td>0.00</td><td>0.00</td><td>0.00</td></tr><tr><td>Radius 0.04</td><td>100.00</td><td>100.00</td><td>0.00</td></tr></table>

Partial binding preserves the count and marks the orphan geometry unavailable in all trials. Nearest assignment redirects the orphan to an already used witness in 720 trials. The drop-unbound control removes one identity in every trial. Class-agnostic assignment reroutes the orphan in 96 trials and creates 136 cross-class links in total. Ungated sameclass assignment reroutes it in 18 trials. This probe demonstrates the count/measurementavailability distinction under a controlled intervention; it does not establish the prevalence or detection accuracy of natural witness failures.

## S3.3 Numerical verification

The support-readout audit independently checks complete-component ownership, support gates, frozen semantic fields, normal calibration, strict decisions and task-macro AUROC/F1. The binding controls are refreshed by applying the same fixed reader to their previously saved object assignments, without a new neural or baseline run. The main policy reproduces every audited complete-support configuration and all 9,031 predictions exactly; each of the six policies retains 9,031 records under each reference mode. Controlled witness-removal trials and assignment traces are unchanged because they evaluate the binding interface before axis updates. Original manuscripts, master tables and prior experiment archives are preserved.

## S4 Normal references and deployment

For continuous operand $m _ { j }$ , let $q _ { . 0 1 , j } , q _ { . 9 9 , j }$ be normal-fitting quantiles. The scale and development padding are

$$
b _ { j } = \operatorname* { m a x } ( q _ { . 9 9 , j } - q _ { . 0 1 , j } , 0 . 1 | \operatorname* { m e d i a n } _ { \tau } m _ { j } | , 0 . 0 1 ) ,\tag{S5}
$$

$$
p _ { j } = \operatorname* { m a x } \left( 0 . 2 5 b _ { j } , \operatorname* { m a x } _ { x \in \mathcal { D } } \{ q . 0 1 , j - m _ { j } ( x ) , m _ { j } ( x ) - q . 9 9 , j \} \right) .\tag{S6}
$$

The interval is $[ L _ { j } , U _ { j } ] = [ q _ { . 0 1 , j } - p _ { j } , q _ { . 9 9 , j } + p _ { j } ]$ . Residual references use normal fitting observations; an unavailable reference key defaults to {0}. All required missing operands share one binary unavailable-evidence residual. Cached pixel, color, separator, cable and OCR observations are retained alongside reconstructed object-derived quantities.

Calibration selects the empirical 95th percentile with the higher convention. For n sorted calibration scores, its one-based index is $1 + \lceil . 9 5 ( n - 1 ) \rceil$ ⌉. With ten normal calibration images per VID task, this selects the observed maximum. Decisions use the strict comparison $s > \tau$ The interval recipe, score normalization and calibration convention are shared across the internal configurations; references and thresholds correspond to each configuration’s normal facts.

<table><tr><td colspan="2">Algorithm S1. Normal-data configuration and SGDB deployment</td></tr><tr><td></td><td>Input Normal fitting, development and calibration roles; component targets; task requirements. 1 Train the visual frontend on component targets from normal fitting sources; select on</td></tr><tr><td>2</td><td>normal development targets and freeze it. Train the predesignated semantic decoder on normal fitting parents and paired factual</td></tr><tr><td></td><td>edits; select on normal development targets.</td></tr><tr><td></td><td>3 Select the guarded screw readout on its annotated normal development subset. 4 Select one association scheme on the 294 annotated original normal development parents.</td></tr><tr><td></td><td>5 Fix the normal-development quality floor; apply the support-certified readout to bound</td></tr><tr><td></td><td>normal objects, then fit intervals and residual references.</td></tr><tr><td></td><td>6 Use independent normal calibration scores to freeze each task threshold. 7 For an inspection image: obtain decoded semantic objects and frozen geometric wit-</td></tr><tr><td></td><td>nesses. 8 Apply guarded screw geometry where eligible, then perform partial one-to-one associa-</td></tr><tr><td></td><td>tion. 9 Apply quality and complete-support checks; accept mask-axis updates or retain the initial</td></tr><tr><td></td><td>axes. Rebuild operands and retain auxiliary appearance/OCR cues.</td></tr><tr><td></td><td>10 Execute frozen task constraints and score normalization; compare the score strictly with its threshold.</td></tr><tr><td></td><td>11 Record rule operands, residuals, normalized contributions and their source links; for an alarm, select the first sufficient descending prefix.</td></tr><tr><td></td><td>12 Generate qualitative claims with the frozen language model, check their IDs and render</td></tr><tr><td></td><td>numeric evidence from the common trace. Output Frozen score, image-level decision and supporting report.</td></tr></table>

## S5 Execution-grounded explanation

## S5.1 Trace schema and certificate selection

Each trace record stores a rule key and evidence ID, observed value, executable requirement, residual, normalized score and source paths. Referenced objects retain their decoded indices, confidence, associated witness index, matching distance and available geometry. Length operands additionally retain the prior and candidate axes, support quality and completeness checks where a support-readout record exists. Cached pixel/OCR measurements retain their own source fields. A zero decoded count points to the count record rather than inventing an object region.

The certificate operates on the fixed normalized contributions. For a subset C, omitted scores are set to zero and $\begin{array} { r } { S ( C ) = \operatorname* { m a x } _ { j \in C } z _ { j } + 0 . 0 1 \sum _ { j \in C } z _ { j } } \end{array}$ , with $S ( \emptyset ) = 0$ . Sorting requires O(J log J) time and scanning the prefixes requires $O ( J )$ . For any subset A of size $k ,$ its maximum is at most $z _ { ( 1 ) }$ and its sum is at most $\textstyle \sum _ { i = 1 } ^ { k } z _ { ( i ) }$ . Therefore $S ( A ) \leq S ( C _ { k } )$ , proving minimum cardinality of the first prefix exceeding the threshold. Equal-score ties prefer available predicates over the aggregate unavailable-evidence flag and then use key order. This defines a suficient score-space certificate; selected predicates may share underlying operands.

The full trace and score are preserved alongside the certificate. For non-alarms, the renderer retains unavailable operands and any raw deviations, including those normalized to zero. Otherwise it cites evaluated count predicates and the zero-score execution state. The language realization layer conditions on selected qualitative outcomes and evidence identifiers. Numeric values remain in the executor-owned evidence records. Image-level truth is attached only after generation for the qualitative review strip.

## S5.2 Selection-policy comparison and checks

All four policies use the same 66,297 rule records from 9,031 frozen images. The comparison cohort contains every one of the 5,725 original alarms. All-executed selection retains every rule; positive-score selection drops zero contributions; maximum-score selection retains one largest contribution; compact suficient selection uses the first suficient descending prefix. Mean size is an image-weighted average on that alarm cohort, and suficiency uses each image’s existing strict threshold comparison. No references, thresholds or detection predictions are changed.

The compact certificate retains 5,759 rule records, compared with 11,418 positive-score records and 40,767 executed records over the alarm cohort. Of the certificates, 5,708 contain one rule and 17 contain three. In the latter tape traces, each of the three selected contributions equals 4.044522437723423. One yields 4.084967662100657 and two yield 4.125412886477891, equal to the threshold; all three yield 4.165858110855125. The selected predicates are the green-tape count, red-tape count and measurement-unavailable flag.

Table S8: Per-category alarm counts and mean selected-rule counts. The last column counts alarms insuficiently explained by only the maximum-score rule. Every compact certificate is suficient.
<table><tr><td>Category</td><td>Alarms</td><td>All</td><td>Positive</td><td>Compact</td><td>Max fail</td></tr><tr><td>breakfast box</td><td>81</td><td>27.889</td><td>5.938</td><td>1.000</td><td>0</td></tr><tr><td>juice bottle</td><td>133</td><td>21.917</td><td>3.203</td><td>1.000</td><td>0</td></tr><tr><td>pushpins</td><td>94</td><td>6.862</td><td>2.543</td><td>1.000</td><td>0</td></tr><tr><td>splicing connectors</td><td>85</td><td>6.365</td><td>1.965</td><td>1.000</td><td>0</td></tr><tr><td>screw bag</td><td>126</td><td>5.254</td><td>1.571</td><td>1.000</td><td>0</td></tr><tr><td>Sticks</td><td>533</td><td>3.450</td><td>1.572</td><td>1.000</td><td>0</td></tr><tr><td>Fruits</td><td>438</td><td>3.000</td><td>1.438</td><td>1.000</td><td>0</td></tr><tr><td>Tools</td><td>548</td><td>7.000</td><td>2.294</td><td>1.000</td><td>0</td></tr><tr><td>Cookies</td><td>550</td><td>8.789</td><td>2.813</td><td>1.000</td><td>0</td></tr><tr><td>Tapes</td><td>500</td><td>3.484</td><td>1.814</td><td>1.068</td><td>17</td></tr><tr><td>Stationery</td><td>550</td><td>12.996</td><td>1.500</td><td>1.000</td><td>0</td></tr><tr><td>Ropes</td><td>430</td><td>6.986</td><td>1.847</td><td>1.000</td><td>0</td></tr><tr><td>Blocks</td><td>471</td><td>7.000</td><td>2.034</td><td>1.000</td><td>0</td></tr><tr><td>Dishes</td><td>594</td><td>4.355</td><td>1.845</td><td>1.000</td><td>0</td></tr><tr><td>Balls</td><td>592</td><td>7.000</td><td>1.782</td><td>1.000</td><td>0</td></tr></table>

The full reports retain missing evidence in 1,254 records, including 94 non-alarms; 191 non-alarm traces have raw deviations. Checks reconstruct every residual, normal-tail normalization and full score and reproduce all 9,031 decisions. Independent axis/center calculations reconstruct 3,928 cited geometric operands with maximum absolute error $5 . 6 8 4 \times 1 0 ^ { - 1 4 }$ . Certificate cardinality is also checked by exhaustive enumeration on 160 small synthetic score vectors, a joint-aggregation fixture and four deliberately corrupted report fields. These checks measure execution and provenance consistency.

## S5.3 Frozen language realization

The language input projects a trace record to its predicate name, operator, qualitative relation and evidence ID. Missing operands retain their names. It excludes raw image pixels, source filenames, image-level truth and detector-normalization values. The frozen model writes qualitative claims; the report renders observed numbers and requirements directly from the source records. The JSON response contains the unchanged decision and a list of claims with evidence IDs. Facts sharing one qualitative relation may be combined into a claim.

The contract checks the JSON schema, unchanged decision, exact selected-ID coverage, allowed citations and the absence of literal numeric quantities in generated text. It also checks relation grouping, explicit unavailable states and selected lexical conflicts. A rejected output is retained as a structured trace rather than accepted prose. These are mechanical preservation checks; the comparison measures contract adherence. All prompts, raw responses and rejection reasons are archived, including earlier prompt-adapter development runs.

The final renderer is evaluated on one SHA256-ranked source from each of the 55 deployment tasks, the six main-paper cases and three additional execution states: a joint alarm, a non-alarm with unavailable evidence and a non-alarm with raw deviations. Deduplication gives 63 English reports; seven case reports are also generated in Chinese. Selection does not consume image-level truth. The final batch runs on physical GPU 2 with the existing checkpoint, greedy decoding, batch size two and at most 640 new tokens per report. Detector facts, thresholds and decisions are unchanged. The model uses no image input and takes no optimization steps.

All 70 reports preserve the tested source contract (Table S9). Fifty deliberate corruptions of decisions, source IDs, claim coverage, numeric text and unsupported-cause text are rejected.

Table S9: Frozen language realization. Accepted denotes preservation of the defined source contract; mean claims measures report size. Numeric evidence is rendered by the executor.
<table><tr><td>Language</td><td></td><td></td><td>Reports Accepted Mean claims</td></tr><tr><td>English</td><td>63</td><td>63</td><td>1.159</td></tr><tr><td>Chinese</td><td>7</td><td>7</td><td>1.429</td></tr></table>

## S6 Comparison settings and condition results

The external detection tables compare completed local deployments. LOCO ComAD uses DINO-S8 clustering with five clusters and CRF processing without component labels; its cross-scene deployment uses the archived shared component-map frontend. Local PUAD omits the external ImageNet penalty-image term. Cross-scene LogiCo uses its archived component frontend. LogicAD-FE-fast is a simplified local program/frontend deployment. Crossscene SALAD and PUAD use category-level calibration, while SGDB uses deployment-task calibration. SGDB uses component annotations on the normal sources listed in Table S1, factual edits with updated targets, and explicit task requirements.

Table S10: SGDB condition summaries across all ten VID scenarios. Values other than image counts are percentages. Missing is the task macro-average of unavailable-operand rates.
<table><tr><td>Condition</td><td>Images</td><td>AUROC</td><td>F1</td><td>Normal FPR</td><td>Missing</td></tr><tr><td>Original</td><td>1579</td><td>94.94</td><td>93.85</td><td>6.60</td><td>12.63</td></tr><tr><td>Cable BG</td><td>1579</td><td>94.06</td><td>91.88</td><td>11.60</td><td>14.94</td></tr><tr><td>Mesh BG</td><td>1579</td><td>92.83</td><td>91.96</td><td>13.00</td><td>13.94</td></tr><tr><td>Low light</td><td>1579</td><td>93.19</td><td>93.12</td><td>12.00</td><td>13.91</td></tr><tr><td>Blur</td><td>1579</td><td>92.76</td><td>92.99</td><td>13.60</td><td>13.37</td></tr></table>

Internal controls share the normal-source roles, component-supervision budgets, visual frontend, task requirements and score/calibration recipe. The plain-set and evidence-edit decoders each use 768 updates and 1,536 image exposures. Direct DFD and its initial bound readout share semantic predictions and difer in the source of geometric operands. Bindingpolicy controls share decoder outputs and the complete-support reader; geometric-readout controls share decoder outputs, initial witnesses and class masks. The comparison recomputes normal references and thresholds from each configuration’s normal facts using the fixed recipe. The matched-reference diagnostic instead holds the initial intervals, references and thresholds fixed, measuring the decision changes due to axis selection alone.

Table S11: Information and implementation conditions of the completed deployments. NR: not established by the retained comparison protocol; the archive provides the actual image scores and thresholds. Internal mechanism controls share SGDB’s component supervision and task requirements.
<table><tr><td>Method</td><td>Normal data / component Requirements / implementa- VID calibration targets</td><td>tion</td><td></td><td></td></tr><tr><td>SGDB V2</td><td>Fit/DEV/CAL roles; com- Explicit ponent targets in Table S1. learned semantics, bound</td><td>geometry and certification.</td><td>requirements; Task level. support</td><td></td></tr><tr><td>SALAD</td><td>Normal fitting; archived Local completed deployment; Category level VID component-map in- 70,000-step VID branch; no put; LOCO annotation de- SGDB executor. tail NR.</td><td></td><td></td><td></td></tr><tr><td>LogiCo</td><td>Normal fitting; archived Local component adaptation; Pooling NR. VID component frontend; fixed 200-epoch VID fit; no exact LOCO target lineage SGDB executor.</td><td></td><td></td><td></td></tr><tr><td>ComAD</td><td>NR. LOCO normal RGB only, LOCO DINO-S8, five clusters Pooling NR. no component labels; VID and CRF; VID component- shared component fron- map adapter.</td><td></td><td></td><td></td></tr><tr><td>PUAD</td><td>tend. LOCO only; VID local shared geNet penalty images omit-</td><td></td><td>normal RGB 70,000 steps; external Ima- Category level.</td><td></td></tr><tr><td>Qwen3-VL- 8B</td><td>component-map input. Archived prompts and Frozen normal score calibration; baseline; distinct from the prompt-reference</td><td>ted.</td><td>count explanation-side text model.</td><td>image-language Pooling NR.</td></tr><tr><td>LogicAD-FE- fast</td><td>NR. and local frontend; target gram/frontend deployment. lineage NR.</td><td>Archived normal inputs Simplified</td><td>local</td><td>pro- Pooling NR.</td></tr></table>

The machine-readable condition matrix adds method-specific development selection, known annotation budgets, source-protocol links and unresolved adapter details. Detection tables describe these local deployments. Mechanism attribution uses the shared-supervision internal controls.

## S7 All 50 VID task results

The following tables report every retained primary method on all 50 deployment tasks. Each Mean row is an arithmetic task macro-average. The SGDB column refers to the fixed evidence-edit decoder with complete-support readout; the internal controls appear in the main-paper ablation. Best/second-best distinct displayed values are bold/underlined, and ties share marks.

Table S12: Complete VID task AUROC (%).
<table><tr><td>Scenario</td><td>Condition</td><td>SALAD</td><td>LogiCo</td><td>ComAD</td><td>PUAD</td><td>Qwen</td><td>LogicAD</td><td>SGDB</td></tr><tr><td>Balls</td><td>Original</td><td>13.35</td><td>60.78</td><td>61.30</td><td>8.93</td><td>99.86</td><td>53.59</td><td>89.47</td></tr><tr><td>Balls</td><td>Cable BG</td><td>19.15</td><td>67.74</td><td>63.44</td><td>11.02</td><td>92.99</td><td>50.79</td><td>97.09</td></tr><tr><td>Balls</td><td>Mesh BG</td><td>56.74</td><td>62.37</td><td>74.76</td><td>43.63</td><td>99.64</td><td>66.19</td><td>88.09</td></tr><tr><td>Balls</td><td>Low light</td><td>16.15</td><td>55.46</td><td>70.78</td><td>12.72</td><td>99.27</td><td>37.80</td><td>77.10</td></tr><tr><td>Balls</td><td>Blur</td><td>13.39</td><td>48.04</td><td>58.98</td><td>10.07</td><td>97.59</td><td>47.99</td><td>80.41</td></tr><tr><td>Blocks</td><td>Original</td><td>95.44</td><td>93.31</td><td>80.51</td><td>88.64</td><td>99.34</td><td>43.85</td><td>87.73</td></tr><tr><td>Blocks</td><td>Cable BG</td><td>86.09</td><td>92.38</td><td>75.51</td><td>75.25</td><td>99.06</td><td>47.89</td><td>85.35</td></tr><tr><td>Blocks</td><td>Mesh BG</td><td>87.87</td><td>94.04</td><td>77.33</td><td>68.35</td><td>100.00</td><td>40.38</td><td>91.11</td></tr><tr><td>Blocks</td><td>Low light</td><td>91.25</td><td>96.75</td><td>81.58</td><td>78.64</td><td>98.59</td><td>48.61</td><td>91.36</td></tr><tr><td>Blocks</td><td>Blur</td><td>93.95</td><td>95.38</td><td>74.73</td><td>87.62</td><td>67.65</td><td>63.16</td><td>90.20</td></tr><tr><td>Cookies</td><td>Original</td><td>63.51</td><td>49.77</td><td>99.98</td><td>13.81</td><td>49.39</td><td>51.39</td><td>91.08</td></tr><tr><td>Cookies</td><td>Cable BG</td><td>71.91</td><td>61.49</td><td>99.77</td><td>23.38</td><td>83.40</td><td>56.47</td><td>86.05</td></tr><tr><td>Cookies</td><td>Mesh BG</td><td>61.51</td><td>70.45</td><td>99.98</td><td>25.53</td><td>99.91</td><td>49.04</td><td>87.05</td></tr><tr><td>Cookies</td><td>Low light</td><td>61.32</td><td>45.23</td><td>98.70</td><td>18.09</td><td>93.24</td><td>53.39</td><td>84.71</td></tr><tr><td>Cookies</td><td>Blur</td><td>56.62</td><td>34.91</td><td>91.79</td><td>13.85</td><td>95.52</td><td>60.92</td><td>84.20</td></tr><tr><td>Dishes</td><td>Original</td><td>70.45</td><td>66.02</td><td>84.88</td><td>87.12</td><td>99.54</td><td>84.14</td><td>96.25</td></tr><tr><td>Dishes</td><td>Cable BG</td><td>50.41</td><td>61.69</td><td>69.73</td><td>54.67</td><td>99.23</td><td>49.89</td><td>96.15</td></tr><tr><td>Dishes</td><td>Mesh BG</td><td>26.09</td><td>56.41</td><td>72.63</td><td>33.55</td><td>99.90</td><td>90.18</td><td>86.97</td></tr><tr><td>Dishes</td><td>Low light</td><td>59.46</td><td>73.80</td><td>79.71</td><td>64.67</td><td>99.59</td><td>98.13</td><td>96.14</td></tr><tr><td>Dishes</td><td>Blur</td><td>55.41</td><td>75.14</td><td>62.79</td><td>69.19</td><td>99.94</td><td>80.43</td><td>88.98</td></tr><tr><td>Fruits</td><td>Original</td><td>99.88</td><td>100.00</td><td>99.96</td><td>98.50</td><td>99.34</td><td>98.40</td><td>96.00</td></tr><tr><td>Fruits</td><td>Cable BG</td><td>99.62</td><td>97.74</td><td>99.96</td><td>97.26</td><td>83.79</td><td>98.02</td><td>90.00</td></tr><tr><td>Fruits</td><td>Mesh BG</td><td>99.98</td><td>100.00</td><td>99.92</td><td>99.18</td><td>95.95</td><td>99.00</td><td>91.50</td></tr><tr><td>Fruits</td><td>Low light</td><td>99.68</td><td>98.32</td><td>99.98</td><td>98.14</td><td>98.90</td><td>100.00</td><td>95.50</td></tr><tr><td>Fruits</td><td>Blur</td><td>99.90</td><td>99.80</td><td>99.90</td><td>99.72</td><td>98.59</td><td>57.35</td><td>96.00</td></tr><tr><td>Ropes</td><td>Original</td><td>41.29</td><td>27.76</td><td>73.93</td><td>41.60</td><td>81.30</td><td>63.04</td><td>89.09</td></tr><tr><td>Ropes</td><td>Cable BG</td><td>43.62</td><td>34.84</td><td>85.24</td><td>39.02</td><td>81.52</td><td>57.20</td><td>89.77</td></tr><tr><td>Ropes</td><td>Mesh BG</td><td>45.76</td><td>33.00</td><td>92.35</td><td>43.44</td><td>81.06</td><td>47.29</td><td>87.30</td></tr><tr><td>Ropes</td><td>Low light</td><td>41.64</td><td>35.18</td><td>75.91</td><td>40.13</td><td>82.19</td><td>79.21</td><td>87.24</td></tr><tr><td>Ropes</td><td>Blur</td><td>42.71</td><td>28.87</td><td>73.58</td><td>44.05</td><td>81.22</td><td>45.50</td><td>89.09</td></tr><tr><td>Stationery</td><td>Original</td><td>52.95</td><td>77.15</td><td>61.07</td><td>48.07</td><td>57.37</td><td>52.13</td><td>100.00</td></tr><tr><td>Stationery</td><td>Cable BG</td><td>68.53</td><td>55.80</td><td>71.49</td><td>54.36</td><td>62.93</td><td>58.34</td><td>100.00</td></tr><tr><td>Stationery</td><td>Mesh BG</td><td>50.75</td><td>54.04</td><td>64.07</td><td>39.85</td><td>78.90</td><td>62.87</td><td>100.00</td></tr><tr><td>Stationery</td><td>Low light</td><td>32.76</td><td>56.13</td><td>74.33</td><td>28.22</td><td>70.19</td><td>63.21</td><td>100.00</td></tr><tr><td>Stationery</td><td>Blur</td><td>50.35</td><td>68.36</td><td>57.09</td><td>41.45</td><td>90.15</td><td>42.95</td><td>100.00</td></tr><tr><td>Sticks</td><td>Original</td><td>69.44</td><td>35.85</td><td>97.65</td><td>70.44</td><td>88.31</td><td>69.34</td><td>100.00</td></tr><tr><td>Sticks</td><td>Cable BG</td><td>76.31</td><td>55.48</td><td>98.65</td><td>76.29</td><td>84.02</td><td>73.14</td><td>100.00</td></tr><tr><td>Sticks</td><td>Mesh BG</td><td>98.10</td><td>66.29</td><td>95.58</td><td>98.88</td><td>87.58</td><td>69.63</td><td>100.00</td></tr><tr><td>Sticks</td><td>Low light</td><td>85.71</td><td>34.96</td><td>96.79</td><td>85.38</td><td>88.34</td><td>56.27</td><td>100.00</td></tr><tr><td>Sticks</td><td>Blur</td><td>84.94</td><td>35.58</td><td>99.10</td><td>79.79</td><td>78.09</td><td>77.24</td><td>99.62</td></tr><tr><td>Tapes</td><td>Original</td><td>100.00</td><td>99.73</td><td>100.00</td><td>100.00</td><td>94.26</td><td>68.17</td><td>99.89</td></tr><tr><td>Scenario</td><td>Condition</td><td>SALAD</td><td>LogiCo</td><td>ComAD</td><td>PUAD</td><td></td><td>Qwen LogicAD</td><td>SGDB</td></tr><tr><td>Tapes</td><td>Cable BG</td><td>100.00</td><td>100.00</td><td>100.00</td><td>100.00</td><td>89.54</td><td>69.09</td><td>96.78</td></tr><tr><td>Tapes</td><td>Mesh BG</td><td>100.00</td><td>100.00</td><td>100.00</td><td>100.00</td><td>94.83</td><td>65.04</td><td>97.05</td></tr><tr><td>Tapes</td><td>Low light</td><td>100.00</td><td>93.60</td><td>100.00</td><td>100.00</td><td>94.75</td><td>61.87</td><td>100.00</td></tr><tr><td>Tapes</td><td>Blur</td><td>100.00</td><td>99.93</td><td>100.00</td><td>100.00</td><td>91.31</td><td>80.68</td><td>99.95</td></tr><tr><td>Tools</td><td>Original</td><td>86.98</td><td>95.09</td><td>79.60</td><td>77.65</td><td>77.07</td><td>57.85</td><td>99.89</td></tr><tr><td>Tools</td><td>Cable BG</td><td>92.38</td><td>93.07</td><td>73.47</td><td>88.11</td><td>81.11</td><td>64.36</td><td>99.36</td></tr><tr><td>Tools</td><td>Mesh BG</td><td>88.95</td><td>93.09</td><td>81.20</td><td>78.84</td><td>77.98</td><td>67.28</td><td>99.25</td></tr><tr><td>Tools</td><td>Low light</td><td>93.11</td><td>95.07</td><td>78.42</td><td>75.64</td><td>86.03</td><td>69.85</td><td>99.89</td></tr><tr><td>Tools</td><td>Blur</td><td>89.36</td><td>93.05</td><td>71.89</td><td>74.71</td><td>91.86</td><td>48.59</td><td>99.11</td></tr><tr><td></td><td>All 50 tasks Macro mean</td><td>69.70</td><td>70.38</td><td>83.60</td><td>62.15</td><td>88.44</td><td>63.94</td><td>93.56</td></tr></table>

Table S13: Complete VID fixed-threshold F1 (%).
<table><tr><td>Scenario</td><td>Condition</td><td>SALAD</td><td>LogiCo</td><td>ComAD</td><td>PUAD</td><td></td><td>Qwen LogicAD</td><td>SGDB</td></tr><tr><td>Balls</td><td>Original</td><td>81.82</td><td>64.89</td><td>25.40</td><td>81.82</td><td>96.65</td><td>1.82</td><td>85.32</td></tr><tr><td>Balls</td><td>Cable BG</td><td>11.11</td><td>61.18</td><td>47.30</td><td>5.80</td><td>77.27</td><td>18.33</td><td>92.86</td></tr><tr><td>Balls</td><td>Mesh BG</td><td>57.92</td><td>53.09</td><td>51.66</td><td>54.35 94.12</td><td></td><td>23.26</td><td>93.15</td></tr><tr><td>Balls</td><td>Low light</td><td>58.06</td><td>63.64</td><td>75.49</td><td>47.76 92.00</td><td></td><td>21.14</td><td>83.26</td></tr><tr><td>Balls</td><td>Blur</td><td>81.37</td><td>64.45</td><td>68.00</td><td>80.46 89.80</td><td></td><td>0.00</td><td>83.19</td></tr><tr><td>Blocks</td><td>Original</td><td>85.27</td><td>91.60</td><td>71.59</td><td>83.53</td><td>99.09</td><td>19.12</td><td>86.01</td></tr><tr><td>Blocks</td><td>Cable BG</td><td>80.00</td><td>89.80</td><td>77.16</td><td>70.18</td><td>87.88</td><td>34.62</td><td>83.17</td></tr><tr><td>Blocks</td><td>Mesh BG</td><td>84.06</td><td>90.91</td><td>72.73</td><td>70.11 98.62</td><td></td><td>19.70</td><td>86.49</td></tr><tr><td>Blocks</td><td>Low light</td><td>80.20</td><td>91.67</td><td>67.86</td><td>68.93 96.83</td><td></td><td>0.00</td><td>90.55</td></tr><tr><td>Blocks</td><td>Blur</td><td>93.56</td><td>90.91</td><td>68.89</td><td>80.00</td><td>69.43</td><td>0.00</td><td>88.67</td></tr><tr><td>Cookies</td><td>Original</td><td>80.92</td><td>78.71</td><td>91.77</td><td>80.92</td><td>85.00</td><td>3.67</td><td>89.52</td></tr><tr><td>Cookies</td><td>Cable BG</td><td>82.17</td><td>82.26</td><td>95.93</td><td>65.22</td><td>88.99</td><td>8.93</td><td>87.04</td></tr><tr><td>Cookies</td><td>Mesh BG</td><td>83.14</td><td>81.93</td><td>94.64</td><td>70.18</td><td>97.09</td><td>17.74</td><td>84.30</td></tr><tr><td>Cookies</td><td>Low light</td><td>74.14</td><td>79.18</td><td>85.83</td><td>16.56</td><td>93.78</td><td>10.53</td><td>81.86</td></tr><tr><td>Cookies</td><td>Blur</td><td>80.92</td><td>81.25</td><td>89.27</td><td>80.92</td><td>75.29</td><td>10.71</td><td>83.33</td></tr><tr><td>Dishes</td><td>Original</td><td>73.68</td><td>77.24</td><td>79.14</td><td>86.11</td><td>91.18</td><td>88.10</td><td>98.23</td></tr><tr><td>Dishes</td><td>Cable BG</td><td>34.01</td><td>73.90</td><td>18.03</td><td>28.57</td><td>96.30</td><td>49.43</td><td>96.94</td></tr><tr><td>Dishes</td><td>Mesh BG</td><td>11.76</td><td>60.58</td><td>55.13</td><td>46.15</td><td>98.63</td><td>75.14</td><td>94.87</td></tr><tr><td>Dishes</td><td>Low light</td><td>5.26</td><td>77.98</td><td>77.07</td><td>0.00</td><td>98.63</td><td>98.67</td><td>98.23</td></tr><tr><td>Dishes</td><td>Blur</td><td>63.21</td><td>78.99</td><td>59.22</td><td>57.14</td><td>92.23</td><td>75.94</td><td>94.87</td></tr><tr><td>Fruits</td><td>Original</td><td>96.37</td><td>100.00</td><td>97.96</td><td>91.30</td><td>95.34</td><td>87.78</td><td>95.83</td></tr><tr><td>Fruits</td><td>Cable BG</td><td>97.51</td><td>90.50</td><td>96.37</td><td>91.01</td><td>83.64</td><td>85.06</td><td>88.89</td></tr><tr><td>Fruits</td><td>Mesh BG</td><td>99.01</td><td>100.00</td><td>98.99</td><td>96.04</td><td>96.97</td><td>88.89</td><td>90.71</td></tr><tr><td>Fruits</td><td>Low light</td><td>87.64</td><td>93.62</td><td>99.01</td><td>84.39</td><td>96.37</td><td>100.00</td><td>95.29</td></tr><tr><td>Fruits</td><td>Blur</td><td>92.47</td><td>97.96</td><td>97.98</td><td>87.64</td><td>94.79</td><td>3.88</td><td>95.83</td></tr><tr><td>Ropes</td><td>Original</td><td>16.53</td><td>9.23</td><td>29.23</td><td>20.97</td><td>77.72</td><td>74.15</td><td>87.76</td></tr><tr><td>Ropes</td><td>Cable BG</td><td>0.00</td><td>3.51</td><td>29.46</td><td>0.00</td><td>78.31</td><td>3.54</td><td>85.57</td></tr><tr><td>Ropes</td><td>Mesh BG</td><td>1.80</td><td>0.00</td><td>3.54</td><td>0.00</td><td>77.78</td><td>3.57</td><td>85.71</td></tr><tr><td>Ropes</td><td>Low light</td><td>0.00</td><td>11.76</td><td>15.13</td><td>0.00</td><td>78.49</td><td>68.60</td><td>84.85</td></tr><tr><td>Ropes</td><td>Blur</td><td>0.00</td><td>1.80</td><td>70.79</td><td>0.00</td><td>75.00</td><td>1.80</td><td>87.76</td></tr><tr><td>Stationery</td><td>Original</td><td>81.78</td><td>81.54</td><td>59.30</td><td>81.78</td><td>26.77</td><td>14.40</td><td>100.00</td></tr><tr><td>Stationery</td><td>Cable BG</td><td>80.93</td><td>74.10</td><td>61.27</td><td>59.30</td><td>13.56</td><td>46.75</td><td>100.00</td></tr><tr><td>Stationery</td><td>Mesh BG</td><td>82.71</td><td>71.55</td><td>60.57</td><td>62.22</td><td>48.32</td><td>15.00</td><td>100.00</td></tr><tr><td>Stationery</td><td>Low light</td><td>56.73</td><td>73.17</td><td>53.85</td><td>35.63</td><td>25.40</td><td>6.96</td><td>100.00</td></tr><tr><td>Stationery</td><td>Blur</td><td>82.09</td><td>80.46</td><td>57.14</td><td>78.46</td><td>8.70</td><td>1.80</td><td>100.00</td></tr><tr><td>Sticks</td><td>Original</td><td>28.80</td><td>44.83</td><td>93.66</td><td>21.67</td><td>80.00</td><td>9.17</td><td>99.05</td></tr><tr><td>Sticks</td><td>Cable BG</td><td>14.29</td><td>60.00</td><td>71.60</td><td>12.61</td><td>81.11</td><td>61.94</td><td>98.11</td></tr><tr><td>Sticks</td><td>Mesh BG</td><td>26.67</td><td>21.85</td><td>91.54</td><td>30.89</td><td>76.02</td><td>0.00</td><td>99.05</td></tr><tr><td>Sticks</td><td>Low light</td><td>17.39</td><td>43.27</td><td>93.40</td><td>14.16</td><td>84.82</td><td>7.41</td><td>98.58</td></tr><tr><td>Sticks</td><td>Blur</td><td>32.00</td><td>39.77</td><td>91.19</td><td>30.65</td><td>70.37</td><td>5.61</td><td>99.05</td></tr><tr><td>Tapes</td><td>Original</td><td>92.44</td><td>92.05</td><td>90.91</td><td>86.96</td><td>78.45</td><td>0.00</td><td>98.15</td></tr><tr><td>Tapes</td><td>Cable BG</td><td>88.00</td><td>88.00</td><td>97.78</td><td>83.97</td><td>80.43</td><td>0.00</td><td>90.55</td></tr><tr><td>Tapes</td><td>Mesh BG</td><td>86.96</td><td>92.05</td><td>86.61</td><td>83.65</td><td>79.78</td><td>0.00</td><td>87.18</td></tr><tr><td>Tapes</td><td>Low light</td><td>97.78</td><td>89.08</td><td>99.10</td><td>95.24</td><td>69.82</td><td>66.99</td><td>100.00</td></tr><tr><td>Tapes</td><td>Blur</td><td>95.24</td><td>90.53</td><td>97.35</td><td>93.62</td><td>48.28</td><td>78.02</td><td>99.08</td></tr><tr><td>Tools</td><td>Original</td><td>68.64</td><td>89.00</td><td>61.64</td><td>37.96</td><td>62.96</td><td>69.70</td><td>98.62</td></tr><tr><td>Tools</td><td>Cable BG</td><td>16.53</td><td>86.12</td><td>66.28</td><td>3.57</td><td>63.22</td><td>1.80</td><td>95.65</td></tr><tr><td>Tools</td><td>Mesh BG</td><td>41.73</td><td>81.72</td><td>51.01</td><td>21.14</td><td>59.63</td><td>8.40</td><td>98.17</td></tr><tr><td>Tools</td><td>Low light</td><td>28.12</td><td>85.13</td><td>60.76</td><td>7.02</td><td>65.45</td><td>45.64</td><td>98.62</td></tr><tr><td>Tools</td><td>Blur</td><td>34.33</td><td>80.21</td><td>55.26</td><td>8.70</td><td>58.97</td><td>18.18</td><td>98.15</td></tr><tr><td>All 50 tasks</td><td> Macro mean</td><td>58.94</td><td>70.14</td><td>70.22</td><td>51.90</td><td>77.11</td><td>31.04</td><td>92.76</td></tr></table>

## S8 Paired grouped analysis and reproduction

## S8.1 Paired grouped uncertainty

The paired analysis keeps the trained models, normal references and calibration thresholds fixed. It resamples groups with replacement within category and binary-label strata, using 10,000 replicates and seed 20261025. Each replicate uses the same group multiplicities for all compared methods. AUROC and fixed-threshold F1 are computed within each deployment task and then macro-averaged, matching the main tables. The reported intervals are the 2.5th and 97.5th percentiles of the paired diferences in percentage points.

LOCO contributes 1,136 individual test-image groups. For VID, each scene, anomalypath type and filename index forms a group of five capture conditions, giving 1,579 index groups for 7,895 images. This correspondence is an explicit resampling assumption; the retained dataset metadata do not verify physical-instance or source-video identity. Normalrole groups use the inherited manifest indices. Neither these group IDs nor exact image SHA-256 values overlap the fitting, development, calibration and test roles. This check does not establish independence between adjacent or visually similar frames.

The intervals describe conditional variation within the evaluated test cohorts under the frozen configurations, rather than training-seed variation or independent blind validation.

Table S14: SGDB V2 minus each frozen control: point diference and paired grouped 95% interval (percentage points). Each deployment retains its own archived normal references. The guardedgeometry comparison is secondary.
<table><tr><td>Dataset</td><td>Control</td><td>Δ AUROC [95% CI]</td><td>∆ F1 [95% CI]</td></tr><tr><td>LOCO</td><td>Initial bound</td><td>+0.00 [+0.00, +0.00]</td><td> $+ 0 . 0 0 \ [ + 0 . 0 0 , + 0 . 0 0 ]$ </td></tr><tr><td>LOCO</td><td>Uniform mask</td><td>+1.83 [+1.18, +2.51]</td><td> $+ 0 . 9 8 \ [ - 0 . 0 7 , \ + 2 . 0 5 ]$ </td></tr><tr><td>LOCO</td><td>Quality only</td><td> $[ + 0 . 9 6 , + 2 . 2 2 ]$ </td><td> $+ 0 . 7 0 \ [ - 0 . 2 8 , + 1 . 6 9 ]$ </td></tr><tr><td>LOCO</td><td>FMP + guarded geometry</td><td>+0.11 [+0.00, +0.29]</td><td> $+ 0 . 9 8 \ [ + 0 . 0 6 , + 2 . 0 0 ]$ </td></tr><tr><td>VID-AD</td><td>Initial bound</td><td>+0.93 [+0.56, +1.32]</td><td>+0.97 [+0.57, +1.43]</td></tr><tr><td>VID-AD</td><td>Uniform mask</td><td>-0.02 [-0.08, +0.04]</td><td>-0.10 [-0.20, -0.02]</td></tr><tr><td>VID-AD</td><td>Quality only</td><td>+0.02 [+0.00, +0.06]</td><td> $+ 0 . 0 0 \ [ + 0 . 0 0 , + 0 . 0 0 ]$ </td></tr><tr><td>VID-AD</td><td>FMP + guarded geometry</td><td>+2.22 [+1.49, +2.96]</td><td> $+ 3 . 3 0 \ [ + 2 . 4 4 , + 4 . 1 9 ]$ </td></tr></table>

Each contrast has a marginal interval; the intervals are not adjusted for simultaneous multiple comparisons. The uniform-mask control has a small VID F1 advantage, while V2 protects LOCO AUROC against both unrestricted and quality-only axis replacement.

Table S15: Sensitivity analysis with the original bound deployment’s intervals, residual references and thresholds held fixed for all readout configurations (percentage points). Same resampling draws as Table S14.
<table><tr><td>Dataset</td><td>Control</td><td>∆ AUROC [95% CI]</td><td>∆ F1 [95% CI]</td></tr><tr><td>LOCO</td><td>Initial bound</td><td>+0.00 [+0.00, +0.00]</td><td>+0.00 [+0.00, +0.00]</td></tr><tr><td>LOCO</td><td>Uniform mask</td><td>+0.68 [+0.33, +1.09]</td><td> $+ 1 . 1 2 \ [ + 0 . 5 4 , + 1 . 7 1 ]$ </td></tr><tr><td>LOCO</td><td>Quality only</td><td>+0.15 [+0.02, +0.35]</td><td>+0.28 [-0.03, +0.64]</td></tr><tr><td>VID-AD</td><td>Initial bound</td><td>+0.93 [+0.56, +1.32]</td><td>+0.97 [+0.57, +1.43]</td></tr><tr><td>VID-AD</td><td>Uniform mask</td><td>-0.03 [-0.08, +0.02]</td><td>-0.05 [-0.13, +0.01]</td></tr><tr><td>VID-AD</td><td>Quality only</td><td>+0.02 [+0.00, +0.06]</td><td>+0.00 [+0.00, +0.00]</td></tr></table>

Table S16: Changes to V2 across all 9,031 images. R/G: repaired/regressed views. Group counts use LOCO image IDs and VID cross-condition filename indices; one group can contain both kinds of change. Complete unchanged and changed group ledgers are supplied.
<table><tr><td>Control</td><td>R views</td><td>G views</td><td>R groups</td><td>s G groups</td></tr><tr><td>Initial bound</td><td>90</td><td>1</td><td>20</td><td>1</td></tr><tr><td>Uniform mask</td><td>28</td><td>27</td><td>28</td><td>23</td></tr><tr><td>Quality only</td><td>22</td><td>14</td><td>22</td><td>14</td></tr></table>

## S8.2 Reproduction material

The CPU archive contains 162,558 saved predictions from 18 configurations, all image labels and source-index group IDs, frozen intervals/references/thresholds, and 12,666 final fact records. The metric script reproduces 990 method/task rows and checks all 715 archived manuscript rows. The scoring replay starts from facts, executes the archived scalar predicates and reconstructs score-space explanation certificates; it does not invoke a detector. Recorded language outputs have a separate contract replay. The archive also supplies exact group draws, per-category results, condition-specific false-positive rates, and source-index transition ledgers.

The code and frozen reproduction records are available in the public SGDB repository (https://github.com/Qybc/SGDB, release v1.0.0). The release includes the research implementations and 15 semantic-decoder checkpoints. Its model inventory records the visualfrontend assets outside this release; original-image inference also requires the benchmark data and OCR/model runtime. The supplied CPU entry points reconstruct the reported results from frozen outputs and facts. No new training, detector inference, baseline run or threshold fitting was performed for this analysis.

## S9 Implementation consistency

Consistency audits verify association, geometric provenance, normal references and strict decisions. They cover 1,176 development records, 25,332 initial facts, 18,062 initial predictions, 4,559 source-mask records and 36,124 complete-support predictions.

Table S17: Paired decision transitions. Same residuals counts changed decisions with an identical raw rule-residual dictionary.
<table><tr><td>Dataset</td><td>Corrected FN</td><td>New FN</td><td>Corrected FP</td><td>New FP</td><td>Changes</td><td>Same residuals</td></tr><tr><td>LOCO</td><td>13</td><td>1</td><td>0</td><td>4</td><td>18</td><td>16</td></tr><tr><td>VID</td><td>366</td><td>110</td><td>77</td><td>66</td><td>619</td><td>158</td></tr></table>

## S10 Paired decisions and qualitative sources

## S10.1 Paired operating-point changes

Table S17 compares FMP plus guarded geometry with SGDB over the full image cohorts. Changes with identical raw residuals reflect configuration-specific normal references and calibration. Main-paper FPR values are task macro-averages.

## S10.2 Qualitative source selection and replay

The LOCO comparison uses the first two logical sources per category, followed by the first normal and third logical source. VID uses the first normal Original, anomalous Original and anomalous Blurry source per scenario. Detailed reports cover nine fixed sources, six in the main paper and three below. The joint-contribution inset adds the lexicographically first frozen tape trace requiring three contributions.

All reports use the audited execution traces. The six main reports use the frozen language model for qualitative claims, with numeric evidence rendered from source records; the three additional reports below retain the structured trace view. Ground truth is added only in the subsequent review strip.

Images are uniformly scaled and padded. Detail crops retain native coordinates and parent pixels. Boxes and axes show SGDB outputs; green/red strips indicate image-level correctness against the dataset label.

Linked evidence detail

Input + associated objects  
(a) Screw bag  
![](images/1d1c4b823b3fda21fae83d2d67473933963ce3bea12497eb363dc6e9498c45f1.jpg)  
GT: Normal | Review: correct

(b) Screw bag  
![](images/e8dd641549a97a21aa8fdc718b5647a6730b8a2571df1abace7fa5fefc7f555a.jpg)  
GT: Anomaly | Review: correct

(c) Screw bag  
![](images/79f58d80982206ddefa1d840b74915aa8701eee472e5bb0b823c10919cfb4cf7.jpg)  
GT: Normal | Review: correct

## Decision / generated claims / numeric evidence

Normal: s = 0.00000 ≤ τ = 4.43005 Counts (observed/required): nut 2/2 [E01]; washer 2/2 [E02];   
screw 2/2 [E03].   
Full trace: 6 predicates; all residuals zero.   
Sources: O0, O1, O2, O3, O4, O5 / decoded identities.

## Anomaly: s = 6.02705 > τ = 4.43005 .43005

Short-screw length: 416.86 px; interval [196.08, 363.12] [E05].   
Selected R05: z=5.967.   
S(C) = 6.02705 > 4.43005; sufficient alarm evidence.   
Sources: O4 / associated axes.

## Normal: s = 0.00000 ≤ τ = 4.43005

Counts (observed/required): nut 2/2 [E01]; washer 2/2 [E02];   
screw 2/2 [E03].   
Full trace: 6 predicates; all residuals zero.   
Sources: O0, O1, O2, O3, O4, O5 / decoded identities.

Figure S1: Additional SGDB reports on normal and anomalous screw bags. All decisions, lengths and interval references are from the current hybrid replay. The input and linked detail retain original dataset pixels.