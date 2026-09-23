# What Drives Hierarchy-Aware Image Retrieval? Taxonomy Alignment, Objective Choice, and Geometry

Ling Shi Southeast University

## Abstract

Foundation vision models provide strong generic representations, yet high class-level retrieval accuracy does not necessarily imply that an embedding respects a target semantic taxonomy. We study strict explicit-taxonomy image retrieval on frozen DINOv2 features and ask an attribution question: when hierarchical retrieval improves, how much of the change is associated with the organization of taxonomy-aware supervision, and how much with the Euclidean–hyperbolic geometry choice?

We evaluate higher levels with strict cross-class criteria that exclude finer-grained matches, and compare Euclidean and hyperbolic projections trained with either taxonomydistance regression or a taxonomy-aware supervised contrastive objective. A compute-matched 2 × 2 Geometry × Loss factorial uses the same 768–256–32 projector capacity, optimization schedule, batch order, and fixed 100- epoch budget; here the Loss axis denotes the Regression-to-TAXONOMY-SUPCON objective-family contrast. On CUB, the objective-family contrasts in mean hierarchy mAP—the average of strict middle- and high-level mAP, excluding Class/Leaf—are +0.0487 in Euclidean space and +0.0414 in hyperbolic space, compared with protocol-defined geometry contrasts of +0.0102 and +0.0030. On NABirds Parent-disjoint retrieval, the corresponding objective-family contrasts are +0.0467 and +0.0440, whereas the geometry contrasts are +0.0017 and −0.0009.

A complementary semantic-alignment control shows that the true taxonomy substantially outperforms a structurepreserving shuffled hierarchy, while a NABirds curvature/radius control does not support stronger negative curvature as the explanation for the observed hierarchy gains. Across the two evaluated taxonomies, the Regression-to-TAXONOMY-SUPCON objective-family contrasts are larger in aggregate than the evaluated geometry contrasts; semantic alignment also matters in the separate control, while geometry remains hierarchy-dependent.

## 1 Introduction

Large-scale self-supervised vision models provide strong and transferable visual representations. DINOv2, for example, produces frozen features that perform well across recognition and retrieval tasks without downstream backbone fine-tuning [1]. However, visual discrimination and semantic organization are not identical objectives. In finegrained retrieval, an embedding may reliably retrieve images from the same visual class while failing to preserve relations between different classes that share a genus, family, Parent,

or Supergroup.

This distinction matters when evaluating hierarchical representations. A same-class match is also trivially a same-Genus or same-Family match, so conventional higherlevel retrieval can overestimate whether an embedding organizes different leaf classes according to the target taxonomy. We therefore use strict cross-class criteria: Cross-class Genus/Parent retrieval excludes the query leaf class, while Cross-genus Family and Cross-parent Supergroup additionally exclude the middle-level group. These criteria serve as an explicit-taxonomy test bed for attribution rather than as a claim that hierarchical image retrieval itself is new.

Hyperbolic representation learning is a natural candidate because negatively curved spaces can compactly represent tree-like structures [2, 3]. Hyperbolic embeddings have been studied for retrieval, zero-shot recognition, metric learning, contrastive learning, and visual hierarchies [4–7, 12]. Recent analysis further shows that temperature and hard-negative behavior can materially affect Euclidean–hyperbolic metriclearning comparisons [22]. These observations motivate separating geometry from objective design rather than interpreting any hyperbolic gain as a geometry-only effect.

We study this attribution problem on frozen DINOv2 features. On CUB [19], the target hierarchy is Class–Genus– Family; on NABirds [21], it is Leaf–Parent–Supergroup. We distinguish three evidence questions: (i) semantic alignment—does the supervision correspond to the target taxonomy? (ii) objective choice—how do taxonomy-distance Regression and TAXONOMY-SUPCON organize the same taxonomy relations? and (iii) geometry—what changes when Euclidean and Poincare distances are compared under´ a matched training protocol?

For the latter two questions, we construct a computematched $2 \times 2$ Geometry × Loss factorial with Euclidean Regression $( E _ { \mathrm { R } } )$ , Hyperbolic Regression $( H _ { \mathrm { R } } )$ , Euclidean TAXONOMY-SUPCON $( E _ { \mathrm { S } } )$ , and Hyperbolic TAXONOMY-SUPCON $( H _ { \mathrm { S } } )$ . The Loss axis is a shorthand for this specific Regression-to-TAXONOMY-SUPCON objective-family contrast, not an intervention on any single loss mechanism such as level balancing or negative handling. All cells use the same projector capacity, initialization, batch-order trajectory, optimizer, learning rate, and fixed epoch-100 endpoint. The Poincare branch uses dataset-specific operating points´ frozen before the factorial comparison; thus our “geometry effect” is a protocol-defined Euclidean–Poincare contrast,´ not a claim about a globally optimized geometry family or negative curvature in isolation.

On CUB, the Regression-to-TAXONOMY-SUPCON objective-family contrast in mean hierarchy mAP is +0.0487 in Euclidean space and +0.0414 in hyperbolic space, while the geometry contrasts are +0.0102 and +0.0030. On NABirds Parent-disjoint retrieval, the corresponding objective-family contrasts are +0.0467 and +0.0440, whereas geometry contrasts shrink to +0.0017 and −0.0009. Under TAXONOMY-SUPCON, the hyperbolic branch also shifts retrieval across levels: Genus decreases while Family increases on CUB, and Parent decreases while Supergroup increases on NABirds. We therefore interpret geometry as a smaller, level-dependent factor in the evaluated protocols rather than a stable global advantage.

Controlled 2 × 2 Geometry × Loss  
![](images/fe74a7638ba67ca744484a1513316925ae30ef5b8fb6a685e2a27dddcf908987.jpg)  
Figure 1. Overview of the controlled hierarchy-aware retrieval framework. A frozen DINOv2-ViT-B/14 encoder produces 768- D features, followed by a shared 768–256–32 projector family. We cross Euclidean/Hyperbolic geometry with taxonomy-distance Regression/TAXONOMY-SUPCON, forming E , H , E , H under matched initialization, batch order, optimizer, and fixed 100-epoch compute. Retrieval uses strict leaf-, middle-, and high-level positives that exclude finer-level matches.

Complementary controls answer different questions. A true-versus-structure-preserving-shuffle study provides semantic-alignment evidence, while a separate curvature/radius study bounds claims about negative-curvature mechanisms. We intentionally keep these controls distinct from the compute-matched factorial because they estimate different quantities.

## Contributions.

• We formulate strict cross-class taxonomy retrieval as a controlled test bed that prevents finer-level matches from being re-counted as evidence of higher-level organization.

• We construct a compute-matched Geometry × Loss factorial that compares a specific Regression-to-TAXONOMY-SUPCON objective-family contrast with a protocol-defined Euclidean–Poincare contrast.´

• Across CUB and NABirds Parent-disjoint, the lossorganization contrasts are substantially larger than the aggregate geometry contrasts, while the latter show a repeated hierarchy-level redistribution under TAXONOMY-

SUPCON.

• Separate semantic-alignment and mechanism-boundary controls show that performance is sensitive to the true taxonomy, while the NABirds curvature study does not support stronger negative curvature as the explanation for the gains.

## 2 Related Work

Foundation visual representations. Self-supervised foundation models such as DINOv2 provide transferable features that can be frozen while downstream projections and objectives are changed [1]. We use this separation to study hierarchy-aware retrieval without conflating the result with backbone fine-tuning.

Hyperbolic metric and hierarchy learning. Poincare em-´ beddings and hyperbolic neural networks established hyperbolic representations for hierarchical data [2, 3]. Hyperbolic vision spans image embeddings, zero-shot recognition, metric learning, and contrastive representation learning [4–8]. Closer to explicit visual hierarchies, Xu et al. [9] combine hyperbolic embeddings with hierarchical margins for coarseto-fine recognition, Kwon et al. [10] learn visual hierarchy mappings with a hierarchical contrastive loss, and Berg et al. [11] organize hyperbolic class prototypes using a known label hierarchy. These works exploit hierarchy to improve recognition or classification; our study instead asks how a specific objective-family contrast compares with a protocoldefined geometry contrast in strict retrieval. Yue et al. [22] analyze temperature and hard-negative effects in Euclidean– hyperbolic metric learning, underscoring the need to control the objective. HIER learns latent hyperbolic ancestor proxies without an explicit upper-level target taxonomy [13], whereas Wang et al. [12] learn user-defined, part-based visual hierarchies without explicit hierarchical class labels and introduce distribution-based hierarchical retrieval evaluation. Our setting assumes an explicit taxonomic class tree and strict cross-class exclusions at each evaluated level.

Explicit hierarchy supervision. Taxonomy-aware metric learning predates recent contrastive objectives: Verma et al. [14] learn similarity metrics directly from a class taxonomy. Proxy Anchor and Supervised Contrastive Learning provide representative proxy- and sample-based metric objectives [16, 17]. Coarse-to-fine and hierarchical visual recognition explicitly model label granularity and taxonomic consistency [15, 18]; recent work additionally studies mixedgranularity supervision [24] and taxonomy-aware representation alignment [25]. Atigh et al. [23] use hyperbolic prototypes for classification when labels may be provided at different hierarchy granularities. In contrast, our retrieval setting assumes a fully specified target taxonomy for training examples and asks which observed changes are associated with semantic alignment, objective choice, and the protocoldefined geometry contrast.

## 3 Method

## 3.1 Problem Formulation and Strict Retrieval

Given image x, a frozen DINOv2-ViT-B/14 encoder produces $\mathbf { f } = \mathbf { \bar { \mathit { F } } } ( x ) \in \mathbb { R } ^ { 7 6 8 }$ , which a trainable projector maps to a 32-D embedding. Each sample has leaf, middle, and high labels $y ^ { c } , y ^ { m } , y ^ { h }$ : Class–Genus–Family on CUB and Leaf–Parent–Supergroup on NABirds.

We partition pairs into mutually exclusive relations

$$
r ( i , j ) = \left\{ \begin{array} { l l } { c , } & { y _ { i } ^ { c } = y _ { j } ^ { c } , } \\ { m , } & { y _ { i } ^ { m } = y _ { j } ^ { m } , \ y _ { i } ^ { c } \neq y _ { j } ^ { c } , } \\ { h , } & { y _ { i } ^ { h } = y _ { j } ^ { h } , \ y _ { i } ^ { m } \neq y _ { j } ^ { m } , } \\ { o , } & { y _ { i } ^ { h } \neq y _ { j } ^ { h } . } \end{array} \right.\tag{1}
$$

Strict middle-level positives share the middle group but differ in leaf class; strict high-level positives share the high group but differ at the middle level. Equivalently,

$$
{ \mathcal { P } } _ { c } ( i ) = \{ j : y _ { j } ^ { c } = y _ { i } ^ { c } , j \neq i \} ,\tag{2}
$$

$$
\mathcal { P } _ { m } ( i ) = \{ j : y _ { j } ^ { m } = y _ { i } ^ { m } , y _ { j } ^ { c } \neq y _ { i } ^ { c } \} ,\tag{3}
$$

$$
{ \mathcal { P } } _ { h } ( i ) = \{ j : y _ { j } ^ { h } = y _ { i } ^ { h } , y _ { j } ^ { m } \neq y _ { i } ^ { m } \} .\tag{4}
$$

The sets are mutually exclusive by construction. This yields Cross-class Genus / Cross-genus Family on CUB and Crossclass Parent / Cross-parent Supergroup on NABirds. Queries without a valid positive at the evaluated level are excluded. The same semantic partition is used by both training objectives, preventing an objective-specific definition of what constitutes a hierarchy relation.

## 3.2 Euclidean and Hyperbolic Projectors

Both geometries use the same 768–256–32 trainable capacity. The Euclidean branch outputs

$$
\mathbf { z } _ { i } ^ { E } = \frac { g _ { \theta } ( \mathbf { f } _ { i } ) } { \lVert g _ { \theta } ( \mathbf { f } _ { i } ) \rVert _ { 2 } } ,\tag{5}
$$

and uses Euclidean distance, which is ranking-equivalent to cosine distance after normalization.

The hyperbolic branch follows the Poincare-ball´ exponential-map construction used in hyperbolic neural networks [3]. It treats the MLP output as a tangent vector $\mathbf { t } _ { i }$ Matching the implementation exactly, let

$$
n _ { i } = \operatorname * { m a x } ( \lVert { \bf t } _ { i } \rVert _ { 2 } , 1 0 ^ { - 6 } ) , \qquad { \bf t } _ { i } ^ { \prime } = r \operatorname { t a n h } ( n _ { i } ) \frac { { \bf t } _ { i } } { n _ { i } } ,\tag{6}
$$

and map it to the Poincare ball with the library exponential ´ map and projection,

$$
\begin{array} { r } { \mathbf { z } _ { i } ^ { H } = \operatorname { p r o j } _ { \mathbb { D } _ { c } } ( \exp _ { \mathbf { 0 } } ^ { c } ( \mathbf { t } _ { i } ^ { \prime } ) ) . } \end{array}\tag{7}
$$

The dataset-specific operating points $( c , r ) = ( 2 . 0 , 0 . 4 )$ on CUB and (0.1, 0.4) on NABirds are frozen before the factorial final runs and are not retuned per cell or on official-test data. Throughout the paper, a geometry contrast therefore denotes the protocol-defined Euclidean–Poincare compari-´ son at these frozen operating points; it is not a pure curvature intervention or a globally optimized comparison between geometry families.

## 3.3 Taxonomy-Aware Objectives

Distance regression. For every non-self pair, targets $\delta _ { i j } \in \{ 0 . 2 , 0 . 5 , 0 . 8 , 1 . 0 \}$ correspond respectively to relations $c , m , h , o .$ For geometry $g \in \{ E , H \}$ ,

$$
\mathcal { L } _ { \mathrm { r e g } } ^ { ( g ) } = \frac { 1 } { | \Omega | } \sum _ { ( i , j ) \in \Omega } \mathrm { S m o o t h L 1 } ( d _ { g } ( i , j ) , \delta _ { i j } ) .\tag{8}
$$

The controlled factorial deliberately uses the same target scale for both geometries so that the taxonomy supervision targets are held fixed while the distance parameterization changes. Consequently, the regression geometry contrast should be read as a matched-protocol contrast, not as a separately calibrated optimum for each geometry.

TAXONOMY-SUPCON. Following the supervisedcontrastive anchor/denominator construction [17], we replace a single positive set with mutually exclusive taxonomy levels. For each anchor, $\mathcal { P } _ { c } ( i ) , \mathcal { P } _ { m } ( i ) , \mathcal { P } _ { h } ( i )$ are the positive sets defined by Eqs. (2)–(4). We use geometry-matched distance logits

$$
\begin{array} { r } { s _ { i j } ^ { ( g ) } = - d _ { g } ( \mathbf { z } _ { i } , \mathbf { z } _ { j } ) / \tau _ { g } . } \end{array}\tag{9}
$$

Let A<sub>ℓ</sub> denote anchors with at least one positive at level ℓ. Then

$$
\mathcal { L } _ { \ell } ^ { ( g ) } = - \frac { 1 } { \left| \mathcal { A } _ { \ell } \right| } \sum _ { i \in \mathcal { A } _ { \ell } } \frac { 1 } { \left| \mathcal { P } _ { \ell } ( i ) \right| } \sum _ { p \in \mathcal { P } _ { \ell } ( i ) } \log \frac { e ^ { s _ { i p } ^ { ( g ) } } } { \sum _ { a \neq i } e ^ { s _ { i a } ^ { ( g ) } } } .\tag{10}
$$

Let ${ \mathcal { V } } _ { B } = \left\{ \ell \in \{ c , m , h \} : w _ { \ell } > 0 , \left| { \mathcal { A } } _ { \ell } \right| > 0 \right\}$ denote the hierarchy levels active in a minibatch. The optimized loss is

$$
\mathcal { L } _ { \mathrm { T S } } ^ { ( g ) } = \frac { \sum _ { \ell \in \mathcal { V } _ { B } } w _ { \ell } \mathcal { L } _ { \ell } ^ { ( g ) } } { \sum _ { \ell \in \mathcal { V } _ { B } } w _ { \ell } } , \qquad w _ { c } = w _ { m } = w _ { h } = 1 .\tag{11}
$$

A level with no valid anchor is skipped and the remaining active weights are renormalized; a batch with no positive taxonomy relation at any active level is treated as an error. Samples outside the anchor’s high-level group are not positive at any level but remain in the contrastive denominator. Thus each level is first averaged over valid anchors and only then combined, avoiding raw positive-pair-count weighting.

## 3.4 Controlled Geometry × Loss Factorial

The four cells are $E _ { \mathrm { R } } , H _ { \mathrm { R } } , E _ { \mathrm { S } } , H _ { \mathrm { S } }$ . They share the frozen DINOv2 inputs, 768–256–32 projector capacity (205,600 trainable parameters), batch size 128, AdamW with learning rate $5 \times 1 0 ^ { - 5 }$ and weight decay $1 0 ^ { - 4 }$ , dropout 0.1, gradient clipping at 1.0, and a fixed 100-epoch budget. Final training uses seeded example-level shuffling with drop last=True, num workers=0, and no classbalanced sampler, giving 46 updates/epoch on CUB and 148 on NABirds. Within each seed, all cells use the same trainable initialization and identical batch-order trajectory. Seeds are 42, 2024, and 3407.

TAXONOMY-SUPCON temperatures are selected independently for Euclidean and hyperbolic branches using the same validation-only candidate set $\{ 0 . 0 7 , 0 . 1 0 , 0 . 2 0 , 0 . 3 0 , 0 . 5 0 , 1 . 0 0 \}$ and the same fixed 50-epoch tuning budget. The selected values happen to coincide within each dataset: $\tau _ { E } = \tau _ { H } = 0 . 2 0$ on CUB and $\tau _ { E } = \tau _ { H } = 0 . 5 0$ on NABirds. The epoch-100 checkpoint is the fixed primary endpoint; validation-best checkpoints are retained only as auxiliary diagnostics and never replace the primary comparison. Official test data are not used to choose temperature, curvature, radius, or duration. This fixed-endpoint design is important because otherwise each factorial cell could receive a different effective optimization budget.

Rather than averaging across the other factor and reporting only canonical factorial main effects, we report protocolspecified simple effects. This keeps the estimand explicit: the protocol-defined geometry contrast is measured separately under Regression and TAXONOMY-SUPCON, while the Regression-to-TAXONOMY-SUPCON objective-family contrast is measured separately in Euclidean and Poincare´ space. For continuity with the factorial notation, we retain $\Delta _ { \mathrm { l o s s } }$ as shorthand for this objective-family contrast rather than as a claim about a single loss mechanism. We report

$$
\Delta _ { \mathrm { g e o m } } ^ { R } = H _ { \mathrm { R } } - E _ { \mathrm { R } } , ~ \Delta _ { \mathrm { g e o m } } ^ { S } = H _ { \mathrm { S } } - E _ { \mathrm { S } } ,\tag{12}
$$

$$
\Delta _ { \mathrm { l o s s } } ^ { E } = E _ { \mathrm { S } } - E _ { \mathrm { R } } , \qquad \Delta _ { \mathrm { l o s s } } ^ { H } = H _ { \mathrm { S } } - H _ { \mathrm { R } } ,\tag{13}
$$

and interaction $\Delta _ { \mathrm { i n t } } = \left( H _ { \mathrm { S } } - E _ { \mathrm { S } } \right) - \left( H _ { \mathrm { R } } - E _ { \mathrm { R } } \right)$ . Three query/gallery perturbations are first averaged within each training seed.

## 4 Experimental Setup

Datasets and hierarchies. We evaluate on CUB-200- 2011 [19] and NABirds [21]. CUB contains 11,788 images across 200 leaf classes; we train on the official 5,994- image training split and evaluate on the 5,794-image test split. Its Class–Genus–Family taxonomy is derived from the eBird/Clements Checklist v2025 [20] and canonicalized to 124 genera and 37 families. NABirds uses its native Leaf–Parent–Supergroup hierarchy. The primary NABirds factorial is conducted on a Parent-disjoint protocol in which leaf classes and Parent groups are disjoint between development and test. The development bundle contains 19,021 images and the held-out test 5,017 images.

Validation and test isolation. The NABirds tuning split uses $^ { 1 3 , 9 5 9 }$ training images, 4,049 gallery images, and $1 { , } 0 1 3$ query images. Hyperparameters are selected only from the corresponding training-internal validation data. CUB and NABirds TAXONOMY-SUPCON use the same sixtemperature candidate set and fixed 50-epoch tuning budget described in Sec. 3. Official-test metrics are materialized only after the geometry operating points, temperatures, seeds, fixed epoch-100 endpoint, completed training runs, and pretest fairness checks are frozen.

Evaluation and statistical unit. We report Recall@K and mAP, using Euclidean/cosine-equivalent ranking for normalized Euclidean embeddings and Poincare distance for´ hyperbolic embeddings. Each CUB query/gallery perturbation contains 870 query and 4,924 gallery images; the strict Genus and Family valid-query counts are approximately 482– 487 and 747–750, respectively. CUB retains the registered query-micro factorial endpoint. NABirds Parent-disjoint official evaluation uses three fixed perturbations of 1,003 query and 4,014 gallery images. It uses group-macro aggregation: valid queries are first averaged within each evaluable Parent or Supergroup, after which groups receive equal weight. This prevents large groups from dominating the disjoint evaluation.

For every trained model, three fixed query/gallery perturbations are evaluated. These are not treated as independent training observations: they are averaged within each training seed, and the three training seeds form the independent statistical units. Means and sample standard deviations therefore summarize $n = 3$ independently trained models. We use seed directions and variability descriptively rather than as high-powered significance tests.

Contextual baselines and fairness checks. All learned methods operate on frozen DINOv2-ViT-B/14 features. We retain frozen DINOv2, PCA, a frozen-backbone HIER-style model, earlier matched-geometry controls, hierarchy-disjoint splits, semantic-supervision ablations, and curvature/radius controls as contextual or robustness evidence. They are not substituted for the compute-matched factorial when reporting the protocol-defined geometry and Regression-to-TAXONOMY-SUPCON objective-family contrasts. Before official-test evaluation, the factorial audits verify 205,600 trainable parameters in every cell, matched trainable initialization within seed, identical batch-order trajectories and update counts, exact checkpoint reloads, and valid hyperbolic ball constraints. Full split statistics, implementation details, validation grids, factorial Recall@K, and audit metadata are provided in the supplement.

## 5 Results

## 5.1 Baseline Context

Strong leaf-level retrieval does not guarantee taxonomy alignment. On CUB, frozen DINOv2 reaches Class mAP 0.6559 but strict Genus/Family mAP of only 0.3463/0.4000. HIER-style improves Class mAP to 0.7566 while its strict Genus/Family mAP falls to 0.2425/0.1696. A historical 32- D Euclidean taxonomy-regression projector instead obtains 0.4880/0.5631 strict Genus/Family mAP, illustrating that explicit target-taxonomy training can reorganize retrieval even when leaf-level performance decreases.

The same distinction appears on NABirds. In the original seen-class evaluation, HIER-style attains Class mAP 0.7031 but only 0.1149 strict Parent mAP and 0.3801 strict Supergroup mAP. The historical Taxonomy-SupCon baseline instead attains 0.3286 Parent mAP and 0.8901 Supergroup mAP while lowering Class mAP to 0.5249. These contextual comparisons show that target-taxonomy objectives can substantially change the semantic scale emphasized by retrieval. Because these models use different objectives and selection protocols, however, their cross-method gaps are not used as causal estimates of geometry or objective-family mechanisms.

## 5.2 Objective-Family Contrasts Are Larger Than Geometry Contrasts

On CUB, the mean-hierarchy objective-family contrasts are $\Delta _ { \mathrm { l o s s } } ^ { E } = + 0 . 0 4 8 6 7 0 \pm 0 . 0 0 0 \dot { 4 } 0 9$ and $\Delta _ { \mathrm { l o s s } } ^ { H } = \dot { + } 0 . 0 4 1 4 2 4 \pm$ 0.001149, both positive for all three training seeds. The corresponding protocol-defined geometry contrasts are smaller in observed magnitude: $\Delta _ { \mathrm { g e o m } } ^ { R } = + 0 . 0 1 0 2 4 6 \pm 0 . 0 0 0 4 1 6$ and $\Delta _ { \mathrm { g e o m } } ^ { S } = + 0 . 0 0 3 0 0 1 \doteq 0 . 0 0 0 6 8 7$ , again positive for all three seeds. The interaction i $\mathrm { ; - 0 . 0 0 7 2 4 5 { \pm } 0 . 0 0 0 8 0 4 }$ and is negative in all three seeds, so the hyperbolic–Euclidean difference becomes smaller under the stronger taxonomy-aware objective.

NABirds Parent-disjoint shows the same ordering of observed magnitudes under a different hierarchy and unseen Parent groups. The objective-family contrasts are $\Delta _ { \mathrm { l o s s } } ^ { E } =$ $+ 0 . 0 4 6 6 5 6 { \scriptstyle \pm 0 . 0 0 4 3 5 1 }$ and $\Delta _ { \mathrm { l o s s } } ^ { H } = + 0 . 0 4 4 0 4 6 { \pm } 0 . 0 \stackrel { \sim } { 0 } 2 5 2 7 .$ positive in all three training seeds. By contrast, $\Delta _ { \mathrm { g e o m } } ^ { R } =$ $+ 0 . 0 0 1 7 0 6 \pm 0 . 0 0 3 6 5 7$ is positive in two of three seeds, while $\Delta _ { \mathrm { g e o m } } ^ { S } = - 0 . 0 0 0 9 0 4 \pm 0 . 0 0 3 1 8 8$ has a near-zero aggregate mean and mixed seed direction. The interaction $\mathrm { i s \ - 0 . 0 0 2 6 1 0 \pm 0 . 0 0 6 7 9 9 }$ and is not seed-wise consistent; we therefore do not interpret a negative interaction itself as replicated. The pattern repeated across the two evaluated datasets is narrower: the Regression-to-TAXONOMY-SUPCON objective-family contrast is larger in aggregate observed magnitude than the protocol-defined geometry contrast under the evaluated operating regimes.

For cross-dataset sensitivity, we additionally re-score the already frozen CUB projected-test embeddings using the same group-macro aggregation family as NABirds, without retraining or changing checkpoints. The resulting CUB effects are +0.008904 (Geometry under Regression), +0.004647 (Geometry under SupCon), +0.062202 (Loss under Euclidean), +0.057945 (Loss under Hyperbolic), and −0.004256 (interaction). Thus, the observed scale separation between the objective-family and geometry contrasts does not depend on using query-micro rather than groupmacro on CUB. Because this re-scoring was performed after the primary experiment was frozen, it is a post-hoc sensitivity analysis and does not replace the registered CUB endpoint.

## 5.3 Correct Semantic Alignment Matters

The factorial compares two objectives that both use the correct taxonomy. We separately test whether the identity of the hierarchy matters.

Adding true Parent supervision raises mean hierarchy mAP from 0.3066 to 0.3424, a paired gain of +0.0358 that is positive for all three training seeds. Adding Supergroup supervision raises mean hierarchy mAP further to 0.3899, again positive for all three seeds. This latter gain is concentrated at the newly supervised high level: Supergroup mAP increases by +0.0984 relative to Leaf+Parent, whereas Parent mAP decreases by about 0.0034.

The structure-preserving shuffled taxonomy is a stronger semantic control. Full true-taxonomy supervision exceeds it by +0.0849 Parent mAP, +0.2510 Supergroup mAP, and +0.1679 mean hierarchy mAP, with all paired seed differences positive. Because the shuffle preserves the loss form and the multiset of hierarchy group sizes, the result supports the importance of semantic correspondence with the evaluation taxonomy rather than merely adding hierarchyshaped contrastive terms. At the same time, Class mAP falls from 0.6717 under Leaf-only supervision to 0.5459 under the full hierarchy. Taxonomy-aware learning therefore induces a semantic-granularity trade-off rather than uniformly improving all retrieval levels.

Table 1. Controlled Geometry × Loss factorial. The Loss axis denotes the Regression-to-TAXONOMY-SUPCON objective-family contrast. All cells share projector capacity, initialization within seed, batch sequence, optimizer, and fixed 100-epoch training. CUB reports its registered query-micro endpoint; NABirds Parent-disjoint reports group-macro. Mean hierarchy mAP averages strict middle and strict high mAP only (Class/Leaf excluded). Values are mean ± sample SD over three training seeds.
<table><tr><td>Dataset</td><td>Arm</td><td>Class mAP</td><td>Strict middle mAP</td><td>Strict high mAP</td><td>Mean hierarchy mAP</td></tr><tr><td rowspan="4">CUB</td><td> $\scriptstyle { E _ { \mathrm { R } } }$ </td><td> $0 . 6 1 7 1 \pm 0 . 0 0 2 8$ </td><td> $0 . 4 9 9 1 \pm 0 . 0 0 1 7$ </td><td> $0 . 5 6 3 6 \pm 0 . 0 0 1 0$ </td><td> $0 . 5 3 1 4 \pm 0 . 0 0 1 3$ </td></tr><tr><td> $H _ { \mathrm { R } }$ </td><td> $0 . 6 3 5 1 \pm 0 . 0 0 3 3$ </td><td> $0 . 5 1 5 3 \pm 0 . 0 0 2 9$ </td><td> $0 . 5 6 7 9 \pm 0 . 0 0 0 6$ </td><td> $0 . 5 4 1 6 \pm 0 . 0 0 1 7$ </td></tr><tr><td> $E _ { \mathrm { S } }$ </td><td> $0 . 5 1 4 4 \pm 0 . 0 0 3 2$ </td><td> $0 . 5 4 8 0 \pm 0 . 0 0 2 2$ </td><td> $0 . 6 1 2 1 \pm 0 . 0 0 0 7$ </td><td> $0 . 5 8 0 0 \pm 0 . 0 0 0 9$ </td></tr><tr><td> $H _ { \mathrm { S } }$ </td><td> $0 . 4 5 5 0 \pm 0 . 0 0 1 6$ </td><td> $0 . 5 4 0 5 \pm 0 . 0 0 2 2$ </td><td> $0 . 6 2 5 5 \pm 0 . 0 0 0 6$ </td><td> $0 . 5 8 3 0 \pm 0 . 0 0 0 9$ </td></tr><tr><td rowspan="4">NABirds</td><td> $\scriptstyle { E _ { \mathrm { R } } }$ </td><td> $0 . 6 8 3 1 \pm 0 . 0 0 2 8$ </td><td> $0 . 1 9 9 4 \pm 0 . 0 0 3 4$ </td><td> $0 . 4 8 3 0 \pm 0 . 0 0 2 9$ </td><td> $0 . 3 4 1 2 \pm 0 . 0 0 3 1$ </td></tr><tr><td> $H _ { \mathrm { R } }$ </td><td> $0 . 6 8 4 0 \pm 0 . 0 0 5 2$ </td><td> $0 . 1 9 7 1 \pm 0 . 0 0 1 9$ </td><td> $0 . 4 8 8 7 \pm 0 . 0 0 2 4$ </td><td> $0 . 3 4 2 9 \pm 0 . 0 0 0 5$ </td></tr><tr><td> $E _ { \mathrm { S } }$ </td><td> $0 . 4 6 5 5 \pm 0 . 0 0 4 5$ </td><td> $0 . 2 2 9 6 \pm 0 . 0 0 2 7$ </td><td> $0 . 5 4 6 1 \pm 0 . 0 0 0 3$ </td><td> $0 . 3 8 7 8 \pm 0 . 0 0 1 4$ </td></tr><tr><td> $H _ { \mathrm { S } }$ </td><td> $0 . 4 3 8 7 \pm 0 . 0 0 6 6$ </td><td> $0 . 2 2 4 5 \pm 0 . 0 0 1 0$ </td><td> $0 . 5 4 9 4 \pm 0 . 0 0 3 1$ </td><td> $0 . 3 8 6 9 \pm 0 . 0 0 2 0$ </td></tr></table>

![](images/528bc5ab4d8f765066baacf504e9b8d5db433214c928cdbea2fa0e5f4c40abe8.jpg)

![](images/4bf8bce0f9fe8d4fb90b7e4f5fe99b53796f79b3cabdbfc54b8020981278890a.jpg)  
Figure 2. Simple contrasts of objective family and geometry under matched compute. The figure’s “Loss” labels denote the Regressionto-TAXONOMY-SUPCON objective-family contrast; $\Delta _ { \mathrm { l o s s } }$ is retained only as shorthand notation. Each small point is one paired training-seed contrast; the larger point is the mean and the horizontal error bar is descriptive sample SD. Mean hierarchy mAP averages strict middle/high levels only and excludes Class/Leaf. CUB uses its query-micro primary endpoint and NABirds Parent-disjoint its group-macro primary endpoint.

These four supervision arms use independently validationfrozen training durations, so we treat the experiment as a semantic-alignment control rather than a compute-matched effect estimate. The main factorial, rather than this ablation, is the basis for comparing the specific objective-family contrast with the protocol-defined geometry contrast.

## 5.4 Geometry Redistributes Retrieval Across Semantic Levels

The small aggregate geometry effects hide a repeated levelwise pattern. Under TAXONOMY-SUPCON, $H _ { \mathrm { S } } ~ - ~ E _ { \mathrm { S } }$ changes CUB Class/Genus/Family mAP by $- 0 . 0 5 9 4 / \textrm { -- }$ $0 . 0 0 7 4 / + 0 . 0 1 3 4$ , and NABirds Class/Parent/Supergroup mAP by $- 0 . 0 2 6 8 / - 0 . 0 0 5 1 / + 0 . 0 0 3 3$ . Thus, the hyperbolic branch reduces finer and middle-level metrics while improving the highest evaluated level in both datasets.

The repeated sign pattern is more informative than the small aggregate mean. In both datasets, the hyperbolic Sup-Con branch trades finer/middle-level retrieval for the highest evaluated level. We therefore describe the result as hierarchylevel redistribution, not as evidence for a universal hyperbolic improvement. Nor does the factorial identify a radialdepth mechanism: it reports retrieval behavior at frozen Poincar’e operating points, while the separate mechanism study below is needed to bound curvature-based interpretations.

## 5.5 Robustness and Mechanism Boundaries

Across three CUB Genus-disjoint outer splits, mean hierarchy group-macro mAP is $0 . 3 1 4 6 \pm 0 . 0 0 9 8$ for Frozen $\mathrm { D I N O v } 2 , 0 . 3 1 2 6 \pm 0 . 0 0 9 9$ for matched Euclidean, 0.3148 ± 0.0096 for matched Hyperbolic, and $0 . 3 4 6 4 \pm \ : 0 . 0 2 1 8$ for Taxonomy-SupCon. The Taxonomy-SupCon–matched-Hyperbolic paired gain averages +0.0316 and is positive on all three splits, but its decomposition is strongly leveldependent: Genus mAP is lower by about 0.0003 while (a) Semantic supervision on NABirds Parent-disjoint

![](images/478ea921f34abe100eb402f16b63bbeaa9e56dcf92becf75b606d9a7500a7b0e.jpg)

(b) Geometry effect under Taxonomy-SupCon  
![](images/1f43713e3279b1f06635e637bf530185f871af37827e70c23c44b0aba8d86994.jpg)  
Figure 3. Semantic supervision and hierarchy-level redistribution. (a) NABirds Parent-disjoint mAP under Leaf-only, Leaf+Parent, full true-taxonomy, and structure-preserving shuffled-taxonomy supervision. (b) Per-level Hyperbolic−Euclidean effect under TAXONOMY-SUPCON. In both datasets, the hyperbolic branch decreases finer/middle-level retrieval while improving the highest evaluated level.

Family mAP is higher by about 0.0634.

Across three NABirds Parent-disjoint outer taxonomy splits, mean hierarchy mAP is $0 . 3 4 0 4 \pm 0 . 0 2 2 0$ for matched Euclidean, $0 . 3 4 2 6 \pm 0 . 0 2 2 3$ for matched Hyperbolic, and $0 . 3 8 4 9 { \scriptstyle \pm 0 . 0 1 7 7 }$ for Taxonomy-SupCon. Taxonomy-SupCon exceeds matched Hyperbolic by $0 . 0 4 2 3 \pm 0 . 0 0 6 8$ , positive on every outer split, whereas matched Hyperbolic exceeds matched Euclidean by only $0 . 0 0 2 2 2 { \scriptstyle \pm 0 . 0 0 0 4 9 }$ . These earlier experiments use validation-frozen configurations rather than the new fixed-100-epoch factorial, so they provide qualitative robustness rather than additional factorial estimates.

The curvature control further narrows the mechanism claim. At fixed $r = 0 . 4 , c = 0 . 1$ is slightly below the nearflat $c = 1 0 ^ { - 6 }$ control in mean hierarchy mAP (−0.000419), and increasing curvature to $c = 1 . 0$ reduces it by a further 0.002314. Radius changes are larger but level-dependent. The factorial therefore measures a specific Poincare param-´ eterization at a frozen operating point; it does not establish stronger negative curvature as the cause of the hierarchy gains.

## 6 Discussion and Limitations

Three evidence layers. The experiments separate evidence questions that are often changed together, but the two factorial axes should be interpreted at the level actually intervened on. The compute-matched factorial compares a Regression-to-TAXONOMY-SUPCON objective-family contrast with a protocol-defined geometry contrast (Euclidean versus Poincare at frozen operating points). The objective-´ family intervention jointly changes the objective form, levelwise aggregation, pair weighting, and negative handling; the factorial does not attribute its contrast to any one of those mechanisms. Across both datasets, this objective-family contrast is larger in aggregate observed magnitude than the evaluated Euclidean–Poincare contrast.´

The true-versus-shuffled experiment addresses a different question. Both factorial losses already use the correct taxonomy, so the factorial cannot establish whether semantic identity matters. Disrupting that correspondence while preserving a hierarchy-shaped contrastive structure substantially reduces strict hierarchy retrieval. Because these supervision arms use independently validation-selected durations, we treat this as a semantic-alignment control, not as another compute-matched factorial contrast.

Table 2. Semantic-supervision ablation on NABirds Parentdisjoint. Parent/Supergroup use group-macro mAP. Each arm uses its validation-frozen training duration; this tests semantic alignment rather than the compute-matched Regression-to-TAXONOMY-SUPCON objective-family contrast.
<table><tr><td>Supervision</td><td>Class</td><td>Parent</td><td>Super.</td><td>Mean hier.</td></tr><tr><td>Leaf only</td><td>.6717</td><td>.1901</td><td>.4231</td><td>.3066</td></tr><tr><td>Leaf + Parent</td><td>.6178</td><td>.2441</td><td>.4407</td><td>.3424</td></tr><tr><td>Full taxonomy</td><td>.5459</td><td>.2408</td><td>.5391</td><td>.3899</td></tr><tr><td>Shuffled taxonomy</td><td>.5874</td><td>.1559</td><td>.2881</td><td>.2220</td></tr></table>

What the geometry contrast means. The geometry contrasts should not be read as pure curvature effects or as a search over the best possible Euclidean and hyperbolic models. CUB and NABirds use dataset-specific Poincare´ operating points frozen before the factorial final runs. Regression deliberately shares one absolute taxonomy-target scale across geometries, while TAXONOMY-SUPCON uses a symmetric validation-only temperature search for both branches. These choices define a matched protocol in which supervision, capacity, update budget, initialization, and batch order are controlled, but they do not constitute independent global optimization of each geometry family.

The remaining geometry differences are also leveldependent. Under TAXONOMY-SUPCON, the hyperbolic branch lowers Class and Genus while raising Family mAP on CUB, and lowers Class and Parent while raising Supergroup mAP on NABirds. The NABirds curvature/radius control further provides mechanism-boundary evidence: the selected c = 0.1 point does not outperform a near-flat control at fixed radius, and stronger curvature decreases mean hierarchy mAP. We therefore do not attribute the factorial geometry contrast specifically to negative curvature or radial depth.

Semantic trade-offs and robustness. The supervision ablation and factorial both show that improving strict hierarchy retrieval can reduce leaf-level discrimination. TAXONOMY-SUPCON is therefore better aligned with the strict hierarchy objective studied here rather than universally superior as an embedding loss. Likewise, outer Genus/Parent-disjoint studies support the broader scale separation between taxonomyaware objective changes and historical Euclidean–hyperbolic differences, but CUB also shows that an aggregate gain can be concentrated in the still-shared Family level. We treat these studies as robustness evidence, not additional factorial estimates or full subtree-disjoint generalization.

Scope and statistical limitations. All experiments use frozen DINOv2-ViT-B/14 features, a compact 32-D projector, and two related fine-grained bird datasets. The primary factorial contains three independent training seeds; query/gallery perturbations are averaged within seed, so sample standard deviations and seed directions are descriptive and we make no significance claims. The geometry contrast represents specific frozen operating points, and an equivalent full curvature/radius study has not been performed on CUB. The semantic shuffle uses one fixed structurepreserving permutation and is not compute-matched, supporting “true taxonomy versus this controlled shuffle” rather than a population-average random-taxonomy effect. CUB and NABirds also use different registered primary aggregation rules; the post-hoc CUB group-macro sensitivity preserves the qualitative separation but does not replace the query-micro primary endpoint.

Within these boundaries, the strongest repeated aggregate changes arise from the Regression-to-TAXONOMY-SUPCON objective-family contrast, semantic-alignment evidence independently shows sensitivity to the target taxonomy, and the evaluated geometry contrast is smaller in aggregate while still redistributing retrieval across hierarchy levels.

## 7 Conclusion

We studied strict explicit-taxonomy image retrieval on frozen DINOv2 features and separated three evidence questions that are often conflated: semantic alignment with the target taxonomy, the choice between taxonomy-distance Regression and TAXONOMY-SUPCON, and a protocol-defined Euclidean–Poincare contrast. Across CUB and NABirds,´ the compute-matched Regression-to-TAXONOMY-SUPCON objective-family contrasts are larger in aggregate observed magnitude than the evaluated geometry contrasts. A separate true-versus-shuffle control shows strong sensitivity to semantic alignment, while the NABirds curvature control does not support stronger negative curvature as the explanation for the observed hierarchy gains. In the evaluated settings, geometry remains relevant mainly through smaller, level-dependent redistribution rather than a stable overall advantage.

## References

[1] M. Oquab, T. Darcet, T. Moutakanni, et al. DINOv2: Learning robust visual features without supervision. Transactions on Machine Learning Research, 2024. 1, 2

[2] M. Nickel and D. Kiela. Poincare embeddings for learning´ hierarchical representations. In NeurIPS, pages 6338–6347, 2017. 1, 2

[3] O.-E. Ganea, G. Becigneul, and T. Hofmann. Hyperbolic´ neural networks. In NeurIPS, 2018. 1, 2, 3

[4] V. Khrulkov, L. Mirvakhabova, E. Ustinova, I. Oseledets, and V. Lempitsky. Hyperbolic image embeddings. In CVPR, pages 6418–6428, 2020. 1, 2

[5] S. Liu, J. Chen, L. Pan, C.-W. Ngo, T.-S. Chua, and Y.- G. Jiang. Hyperbolic visual embedding learning for zero-shot recognition. In CVPR, pages 9273–9281, 2020.

[6] A. Ermolov, L. Mirvakhabova, V. Khrulkov, N. Sebe, and I. Oseledets. Hyperbolic vision transformers: Combining improvements in metric learning. In CVPR, pages 7409– 7419, 2022.

[7] S. Ge, S. Mishra, S. Kornblith, C.-L. Li, and D. Jacobs. Hyperbolic contrastive learning for visual representations beyond objects. In CVPR, pages 6840–6849, 2023. 1

[8] P. Mettes, M. Ghadimi Atigh, M. Keller-Ressel, J. Gu, and S. Yeung. Hyperbolic deep learning in computer vision: A survey. International Journal of Computer Vision, 132(9):3484–3508, 2024. 2

[9] S.-L. Xu, Y. Sun, F. Zhang, A. Xu, X.-S. Wei, and Y. Yang. Hyperbolic space with hierarchical margin boosts finegrained learning from coarse labels. In NeurIPS, volume 36, 2023. 2

[10] H. Kwon, J. Jang, J. Kim, K. Kim, and K. Sohn. Improving visual recognition with hyperbolical visual hierarchy mapping. In CVPR, pages 17364–17374, 2024. 2

[11] P. Berg, L. Buecher, B. Michele, M.-T. Pham, L. Chapel, and N. Courty. Multi-prototype hyperbolic learning guided by class hierarchy. International Journal of Computer Vision, 133:7969–7984, 2025. 2

[12] Z. Wang, S. Ramasinghe, C. Xu, J. Monteil, L. Bazzani, and T. Ajanthan. Learning visual hierarchies in hyperbolic space for image retrieval. In ICCV, pages 9924–9934, 2025. 1, 3

[13] S. Kim, B. Jeong, and S. Kwak. HIER: Metric learning beyond class labels via hierarchical regularization. In CVPR, pages 19903–19912, 2023. 3

[14] N. Verma, D. Mahajan, S. Sellamanickam, and V. Nair. Learning hierarchical similarity metrics. In CVPR, pages 2280– 2287, 2012. 3

[15] D. Chang, K. Pang, Y. Zheng, Z. Ma, Y.-Z. Song, and J. Guo. Your “Flamingo” is my “Bird”: Fine-grained, or not. In CVPR, pages 11476–11485, 2021. 3

[16] S. Kim, D. Kim, M. Cho, and S. Kwak. Proxy Anchor Loss for deep metric learning. In CVPR, pages 3238–3247, 2020. 3

[17] P. Khosla, P. Teterwak, C. Wang, A. Sarna, Y. Tian, P. Isola, A. Maschinot, C. Liu, and D. Krishnan. Supervised contrastive learning. In NeurIPS, pages 18661–18673, 2020. 3

[18] S. Park, Y. Zhang, S. X. Yu, S. Beery, and J. Huang. Visually consistent hierarchical image classification. In ICLR, 2025. 3

[19] C. Wah, S. Branson, P. Welinder, P. Perona, and S. Belongie. The Caltech-UCSD Birds-200-2011 Dataset. Technical Report CNS-TR-2011-001, California Institute of Technology, 2011. 1, 4

[20] J. F. Clements, P. C. Rasmussen, T. S. Schulenberg, M. J. Iliff, J. A. Gerbracht, D. Lepage, A. Spencer, S. M. Billerman, B. L. Sullivan, M. Smith, and C. L. Wood. The eBird/Clements checklist of Birds of the World: v2025. Cornell Lab of Ornithology, 2025. 4

[21] G. Van Horn, S. Branson, R. Farrell, S. Haber, J. Barry, P. Ipeirotis, P. Perona, and S. Belongie. Building a bird recognition app and large scale dataset with citizen scientists: The fine print in fine-grained dataset collection. In CVPR, pages 595–604, 2015. 1, 4

[22] Y. Yue, F. Lin, G. Mou, and Z. Zhang. Understanding hyperbolic metric learning through hard negative sampling. In WACV, pages 1891–1903, 2024. 1, 2

[23] M. G. Atigh, M. van Spengler, T. Long, M. Ayoughi, T. Kasarla, and P. Mettes. Hyperbolic learning with supervision from any granularity. In AISTATS, PMLR 300:2674– 2682, 2026. 3

[24] S. Park, Z. Wang, and S. X. Yu. Free-grained hierarchical visual recognition. In CVPR, pages 32767–32776, 2026. 3

[25] H. He, Z. Tan, and Y. Peng. Taxonomy-aware representation alignment for hierarchical visual recognition with large multimodal models. In CVPR, pages 31124–31134, 2026. 3

# Supplementary Material: What Drives Hierarchy-Aware Image Retrieval?

Ling Shi Southeast University

## S1 Dataset and Taxonomy Construction

CUB. CUB-200-2011 contains 11,788 images from 200 leaf classes, with the official 5,994/5,794 train/test split. The accepted taxonomy-construction record identifies the external source as the eBird/Clements Checklist v2025. CUB class names are converted to common names and matched to the checklist after lowercasing and removing punctuation, whitespace, hyphens, and underscores. Of the 200 classes, 163 are exact normalized-name matches, 18 use fuzzy matching, 18 use manually recorded aliases, and Yellow Warbler is completed by manual verification. Genus is taken from the first token of the matched scientific name. Family strings are canonicalized by stripping a trailing parenthetical common-name gloss; in particular, Parulidae and Parulidae (New World Warblers) are merged. The resulting hierarchy contains 200 classes, 124 genera, and 37 canonical families, with zero class-to-genus, class-to-family, or genus-to-family consistency conflicts. Eighty-seven of the 124 genera contain only one CUB class.

The archived source bundle contains the exact construction and canonicalization utilities, which are now included with the paper’s reproducibility artifacts. It does not contain the external checklist CSV, the realized class-to-Genus-to-Family mapping CSV, or the alias JSON used for manual aliases. Exact taxonomy reconstruction therefore still requires those missing data artifacts.

NABirds. NABirds is used with its native tree rather than biological genus/family labels. Leaf is the visual leaf class, Parent is its direct parent node, and Supergroup is the first child below the root Birds. The processed hierarchy contains 555 leaf classes, 404 Parents, and 22 Supergroups. The primary factorial uses the fixed Parent-disjoint protocol: train/validation/test contain 328/114/113 leaf classes and 242/81/81 Parent groups, with zero pairwise leaf-class or Parent overlap. Train and validation together form the 19,021-sample development bundle; the held-out official test contains 5,017 images. Temperature tuning uses 13,959 training images plus a fixed 4,049-gallery/1,013-query validation split. Official testing uses 4,014 gallery and 1,003 query images per perturbation.

## S2 Strict Retrieval Definitions and Aggregation

For a query q and gallery item j, let $y ^ { c } , y ^ { m } , y ^ { h }$ denote leaf, middle, and high-level labels. The strict positive masks are

$$
P _ { c } ( q , j ) = [ y _ { q } ^ { c } = y _ { j } ^ { c } ] ,\tag{1}
$$

$$
P _ { m } ( q , j ) = [ y _ { q } ^ { m } = y _ { j } ^ { m } ] [ y _ { q } ^ { c } \neq y _ { j } ^ { c } ] ,\tag{2}
$$

$$
P _ { h } ( q , j ) = [ y _ { q } ^ { h } = y _ { j } ^ { h } ] [ y _ { q } ^ { m } \neq y _ { j } ^ { m } ] .\tag{3}
$$

Thus CUB evaluates Class, Cross-class Genus, and Cross-genus Family; NABirds evaluates Leaf/Class, Cross-class Parent, and Cross-parent Supergroup. A query is valid for a level only when at least one corresponding positive exists in the gallery.

For a valid query q, Recall@K is one if at least one positive occurs among the top K retrieved items and zero otherwise. If $r _ { q k } \in \{ 0 , 1 \}$ indicates whether rank k is positive and $\begin{array} { r } { N _ { q } ^ { + } = \sum _ { k } r _ { q k } } \end{array}$ , then

$$
\mathrm { A P } ( q ) = \frac { 1 } { N _ { q } ^ { + } } \sum _ { k } \left( \frac { \sum _ { t \leq k } r _ { q t } } { k } \right) r _ { q k } .\tag{4}
$$

Query-micro mAP averages AP equally over valid queries. For group-macro evaluation, valid query metrics are first averaged within each evaluable middle/high group and those group means are then averaged with equal weight. CUB’s registered factorial endpoint is query-micro; NABirds Parent-disjoint uses group-macro. The post-hoc CUB group-macro analysis in Sec. S11 does not replace the registered endpoint.

## S3 Query/Gallery Perturbations and Statistical Unit

The fixed query/gallery perturbation seeds are 42, 2024, and 3407. Every CUB perturbation contains 870 queries and 4,924 gallery images. Strict Genus valid-query counts are 487/482/483 and strict Family counts are 747/750/747 for Q/G seeds 42/2024/3407. Every NABirds Parent-disjoint perturbation contains 1,003 queries and 4,014 gallery images; strict Parent valid-query counts are 494/495/494 and strict Supergroup counts are 962/963/962, with 27 and 11 evaluable groups respectively.

For each trained model, the three Q/G perturbations are evaluated on the same projected official-test embedding. Metrics are first averaged across the three Q/G perturbations within a training seed; only then are the three training seeds treated as independen statistical units. Hence all sample standard deviations for learned factorial models use $n = 3$ training seeds rather than nine Q/G cells.

The accepted final audits record a SHA256 identifier for each fixed Q/G split and verify the same identifiers across all factoria arms. Phase 6C-4 recovered the project’s generic retrieval-split utility and the NABirds Parent-disjoint preparation/split utility from the archived source bundle, together with a manifest of the accepted final split hashes and sizes. The final Q/G NPZ index arrays themselves are not present, so the accepted hashes remain the authoritative split identifiers rather than newly regenerated index lists.

## S4 Full Factorial Implementation Details

Both geometries use the same trainable MLP:

$$
7 6 8 \to \mathrm { L i n e a r } ( 2 5 6 ) \to \mathrm { B N } \to \mathrm { R e L U } \to \mathrm { D r o p o u t } ( 0 . 1 ) \to \mathrm { L i n e a r } ( 3 2 ) ,
$$

for 205,600 trainable parameters. Euclidean outputs are $\ell _ { 2 }$ normalized. Hyperbolic outputs use the mapping in Sec. S5.

Final factorial training uses AdamW with learning rate $5 \times 1 0 ^ { - 5 }$ , constant learning-rate schedule, weight decay $1 0 ^ { - 4 }$ , batch size 128, gradient clipping at norm 1.0, num workers=0, and drop last=True. The loader uses ordinary example-level random shuffling (shuffle=True) with a PyTorch generator seeded by the training seed; no class-balanced or hierarchy-balanced sampler is used. CUB trains on all 5,994 training examples and therefore executes 46 updates per epoch; NABirds trains on al 19,021 development examples and executes 148 updates per epoch. Every final cell is trained for exactly 100 epochs.

For each training seed (42, 2024, 3407), the random state is reset before model construction. The accepted pre-test audits verify identical trainable initialization hashes, identical epoch-wise batch-order hashes, identical update counts, and equal parameter counts across $E _ { \mathrm { R } } , H _ { \mathrm { R } } , E _ { \mathrm { S } } , H _ { \mathrm { S } }$ within that seed. Each final checkpoint is reloaded into a fresh model and reproduces the projected verification sample with maximum absolute difference 0.

Regression minimizes SmoothL1 over every non-self pair using the shared targets

$$
\delta _ { c } = 0 . 2 , \quad \delta _ { m } = 0 . 5 , \quad \delta _ { h } = 0 . 8 , \quad \delta _ { o } = 1 . 0 .
$$

The target scale is intentionally identical for Euclidean and hyperbolic branches so that the taxonomy supervision targets remain fixed while the distance parameterization changes. It should therefore be read as a protocol-defined matched contrast rather than separately calibrated geometry optima.

TAXONOMY-SUPCON implementation. For $\ell \in \{ c , m , h \}$ , let $\mathcal { P } _ { \ell } ( i )$ be the mutually exclusive positive sets in the main paper and $\mathcal { A } _ { \ell } = \{ i : | \mathcal { P } _ { \ell } ( i ) | > 0 \}$ . With $\begin{array} { r } { s _ { i j } = - d _ { g } ( z _ { i } , z _ { j } ) / \tau _ { g } . } \end{array}$

$$
\mathcal { L } _ { \ell } = - \frac { 1 } { \vert \mathcal { A } _ { \ell } \vert } \sum _ { i \in \mathcal { A } _ { \ell } } \frac { 1 } { \vert \mathcal { P } _ { \ell } ( i ) \vert } \sum _ { p \in \mathcal { P } _ { \ell } ( i ) } \log \frac { e ^ { s _ { i p } } } { \sum _ { a \neq i } e ^ { s _ { i a } } } .
$$

Let $\mathcal { V } _ { B } = \left\{ \ell : w _ { \ell } > 0 , \ \lvert \mathcal { A } _ { \ell } \rvert > 0 \right\}$ for the current minibatch. The implemented total objective is

$$
\mathcal { L } _ { \mathrm { T S } } = \frac { \sum _ { \ell \in \mathcal { V } _ { B } } w _ { \ell } \mathcal { L } _ { \ell } } { \sum _ { \ell \in \mathcal { V } _ { B } } w _ { \ell } } , \qquad w _ { c } = w _ { m } = w _ { h } = 1 .
$$

If one level has no valid anchor, that level is skipped and the remaining active weights are renormalized. A minibatch with no positive taxonomy relation at any active level raises an error. All non-self examples, including samples outside the anchor’s high-level group, remain in the contrastive denominator.

## S5 Hyperbolic Implementation and Operating Points

The implementation instantiates geoopt.PoincareBall(c) with $c > 0$ representing absolute curvature, i.e. sectional curvature $- c ,$ and mathematical ball

$$
\mathbb { D } _ { c } ^ { d } = \{ x \in \mathbb { R } ^ { d } : c \| x \| _ { 2 } ^ { 2 } < 1 \} , \qquad R _ { c } = \frac { 1 } { \sqrt { c } } .
$$

For the MLP output t, the project code uses

$$
n = \operatorname* { m a x } ( \| t \| _ { 2 } , 1 0 ^ { - 6 } ) , \qquad t ^ { \prime } = r \operatorname { t a n h } ( n ) \frac { t } { n } ,
$$

then calls ball.expmap0(t’) followed by ball.projx. Under the corresponding Poincare-ball convention,´

$$
\exp _ { 0 } ^ { c } ( v ) = \operatorname { t a n h } ( { \sqrt { c } } \| v \| _ { 2 } ) { \frac { v } { \sqrt { c } \| v \| _ { 2 } } }
$$

with the continuous value at $v = 0$ . The distance used by the code is ball.dist; equivalently,

$$
d _ { c } ( x , y ) = \frac { 2 } { \sqrt { c } } \mathrm { a r t a n h } \big ( \sqrt { c } \lVert ( - x ) \oplus _ { c } y \rVert _ { 2 } \big ) ,
$$

where

$$
x \oplus _ { c } y = \frac { ( 1 + 2 c \langle x , y \rangle + c \| y \| ^ { 2 } ) x + ( 1 - c \| x \| ^ { 2 } ) y } { 1 + 2 c \langle x , y \rangle + c ^ { 2 } \| x \| ^ { 2 } \| y \| ^ { 2 } } .
$$

Before hyperbolic pairwise distances, embeddings are projected with ball.projx. During training, non-finite distances are sanitized and the distance matrix is clipped to [0, 20]; official-test ranking uses the same library distance with sanitization and an upper clipping bound of $1 0 ^ { 6 }$

The CUB operating point is $( c , r ) = ( 2 . 0 , 0 . 4 )$ , giving ball radius $1 / \sqrt { 2 } = 0 . 7 0 7 1 ;$ ; the NABirds point is (0.1, 0.4), giving radius 3.1623. Protocol locks state that each is inherited from a prior dataset-specific operating point and frozen before the factoria final runs; neither is retuned per factorial cell or on official-test data. Across all accepted hyperbolic factorial runs, the maximum audited embedding norm is 0.3622 on CUB and 0.3979 on NABirds, safely inside the respective balls.

The archived project code does not set a manual boundary epsilon for projx and the accepted artifacts do not pin the Geoopt package version. Therefore we do not invent a library-internal projection margin; the reproducible project-level numerical constant is the $1 0 ^ { - 6 }$ tangent-norm clamp above.

## S6 Validation-Only Temperature Selection

Euclidean and hyperbolic TAXONOMY-SUPCON branches are tuned independently with the same six candidates, the same seed 42, and the same fixed 50-epoch budget. Candidate checkpoints are evaluated only at epoch 50 and are not reused for final training. All final cells are retrained from scratch for 100 epochs after temperature selection.

Table 1: CUB temperature selection. The score is the query-micro mean of strict Genus and strict Family mAP.
<table><tr><td>Geometry</td><td>T</td><td>Genus mAP</td><td>Family mAP</td><td>Mean</td><td>Status</td></tr><tr><td>Euclidean</td><td>0.07</td><td>0.537612</td><td>0.617534</td><td>0.577573</td><td></td></tr><tr><td>Euclidean</td><td>0.10</td><td>0.541775</td><td>0.618417</td><td>0.580096</td><td></td></tr><tr><td>Euclidean</td><td>0.20</td><td>0.549530</td><td>0.629024</td><td>0.589277</td><td>winner</td></tr><tr><td>Euclidean</td><td>0.30</td><td>0.518694</td><td>0.658657</td><td>0.588676</td><td>runner-up</td></tr><tr><td>Euclidean</td><td>0.50</td><td>0.424330</td><td>0.696066</td><td>0.560198</td><td></td></tr><tr><td>Euclidean</td><td>1.00</td><td>0.348874</td><td>0.712698</td><td>0.530786</td><td></td></tr><tr><td>Hyperbolic</td><td>0.07</td><td>0.537128</td><td>0.618151</td><td>0.577640</td><td></td></tr><tr><td>Hyperbolic</td><td>0.10</td><td>0.536336</td><td>0.621024</td><td>0.578680</td><td></td></tr><tr><td>Hyperbolic</td><td>0.20</td><td>0.540021</td><td>0.641901</td><td>0.590961</td><td>winner</td></tr><tr><td>Hyperbolic</td><td>0.30</td><td>0.486798</td><td>0.674783</td><td>0.580791</td><td>runner-up</td></tr><tr><td>Hyperbolic</td><td>0.50</td><td>0.393841</td><td>0.703824</td><td>0.548832</td><td></td></tr><tr><td>Hyperbolic</td><td>1.00</td><td>0.334421</td><td>0.711251</td><td>0.522836</td><td></td></tr></table>

CUB selects $\tau _ { E } = \tau _ { H } = 0 . 2 0$ . The Euclidean and hyperbolic runner-up is $\tau = 0 . 3 0$  
NABirds selects $\tau _ { E } = \tau _ { H } = 0 . 5 0 ,$ with $\tau = 0 . 3 0$ as the runner-up for both geometries. The accepted temperature audits record official test accessed=false, common trainable initialization, identical batch-order trajectories, and matched parameter counts across candidates.

Table 2: NABirds Parent-disjoint temperature selection. Parent and Supergroup are group-macro mAP; the score is their mean.
<table><tr><td>Geometry</td><td>T</td><td>Parent mAP</td><td>Supergroup mAP</td><td>Mean</td><td>Status</td></tr><tr><td>Euclidean</td><td>0.07</td><td>0.228385</td><td>0.544684</td><td>0.386535</td><td rowspan="8">runner-up winner</td></tr><tr><td>Euclidean</td><td>0.10</td><td>0.229470</td><td>0.547203</td><td>0.388336</td></tr><tr><td>Euclidean</td><td>0.20</td><td>0.232159</td><td>0.557994</td><td>0.395076</td></tr><tr><td>Euclidean</td><td>0.30</td><td>0.219048</td><td>0.574292</td><td>0.396670</td></tr><tr><td>Euclidean</td><td>0.50</td><td>0.201959</td><td>0.595510</td><td>0.398735</td></tr><tr><td>Euclidean</td><td>1.00</td><td>0.180174</td><td>0.603103</td><td>0.391639</td></tr><tr><td>Hyperbolic</td><td>0.07</td><td>0.229017</td><td>0.546425</td><td>0.387721</td></tr><tr><td>Hyperbolic</td><td>0.10</td><td>0.230315</td><td>0.550430</td><td>0.390373</td></tr><tr><td>Hyperbolic</td><td>0.20</td><td>0.225752</td><td>0.565639</td><td>0.395696</td></tr><tr><td>Hyperbolic</td><td>0.30</td><td>0.210454</td><td>0.585483</td><td>0.397969</td><td>runner-up</td></tr><tr><td>Hyperbolic</td><td>0.50</td><td>0.197504</td><td>0.598966</td><td>0.398235</td><td>winner</td></tr><tr><td>Hyperbolic</td><td>1.00</td><td>0.169683</td><td>0.603121</td><td>0.386402</td><td></td></tr></table>

## S7 Seed-Level Geometry × Loss Factorial

Table 3: CUB registered query-micro mAP after averaging the three Q/G perturbations within each training seed.
<table><tr><td>Arm</td><td>Seed</td><td>Class</td><td>Genus</td><td>Family</td><td>Mean hierarchy</td></tr><tr><td> $E _ { R }$ </td><td>42</td><td>0.613884</td><td>0.500621</td><td>0.564324</td><td>0.532472</td></tr><tr><td> $E _ { R }$ </td><td>2024</td><td>0.618459</td><td>0.497207</td><td>0.562511</td><td>0.529859</td></tr><tr><td> $E _ { R }$ </td><td>3407</td><td>0.619024</td><td>0.499484</td><td>0.563990</td><td>0.531737</td></tr><tr><td> $H _ { R }$ </td><td>42</td><td>0.632376</td><td>0.517540</td><td>0.568281</td><td>0.542911</td></tr><tr><td> $H _ { R }$ </td><td>2024</td><td>0.638752</td><td>0.512089</td><td>0.567167</td><td>0.539628</td></tr><tr><td> $H _ { R }$ </td><td>3407</td><td>0.634214</td><td>0.516354</td><td>0.568181</td><td>0.542268</td></tr><tr><td> $E _ { S }$ </td><td>42</td><td>0.514624</td><td>0.550282</td><td>0.611362</td><td>0.580822</td></tr><tr><td> $E _ { S }$ </td><td>2024</td><td>0.517443</td><td>0.545790</td><td>0.612189</td><td>0.578990</td></tr><tr><td> $E _ { S }$ </td><td>3407</td><td>0.511104</td><td>0.547848</td><td>0.612682</td><td>0.580265</td></tr><tr><td> $H _ { S }$ </td><td>42</td><td>0.456725</td><td>0.541272</td><td>0.624918</td><td>0.583095</td></tr><tr><td> $H _ { S }$ </td><td>2024</td><td>0.454890</td><td>0.538037</td><td>0.626124</td><td>0.582080</td></tr><tr><td> $H _ { S }$ </td><td>3407</td><td>0.453435</td><td>0.542295</td><td>0.625512</td><td>0.583904</td></tr></table>

Table 4: CUB mean-hierarchy simple effects by independent training seed. Final column is mean ± sample SD.
<table><tr><td>Effect</td><td>42</td><td>2024</td><td>3407</td><td> $\mathrm { M e a n } \pm \mathrm { S D }$ </td></tr><tr><td> $\Delta _ { \mathrm { g e o m } } ^ { R }$ </td><td>+0.010438</td><td>+0.009769</td><td>+0.010531</td><td> $+ 0 . 0 1 0 2 4 6 \pm 0 . 0 0 0 4 1 6$ </td></tr><tr><td> $\Delta _ { \mathrm { g e o m } } ^ { \bar { S } }$ </td><td>+0.002273</td><td>+0.003090</td><td>+0.003638</td><td> $+ 0 . 0 0 3 0 0 1 \pm 0 . 0 0 0 6 8 7$ </td></tr><tr><td> $\Delta _ { \mathrm { l o s s } } ^ { E }$ </td><td>+0.048350</td><td>+0.049131</td><td>+0.048529</td><td> $+ 0 . 0 4 8 6 7 0 \pm 0 . 0 0 0 4 0 9$ </td></tr><tr><td> $\Delta _ { \mathrm { l o s s } } ^ { H }$ </td><td>+0.040184</td><td>+0.042452</td><td>+0.041636</td><td> $+ 0 . 0 4 1 4 2 4 \pm 0 . 0 0 1 1 4 9$ </td></tr><tr><td> $\Delta _ { \mathrm { i n t } }$ </td><td>-0.008165</td><td>-0.006678</td><td>-0.006892</td><td> $- 0 . 0 0 7 2 4 5 \pm 0 . 0 0 0 8 0 4$ </td></tr></table>

## S8 Full Factorial Recall@K

The following tables are obtained only by applying the registered aggregation to the archived official-test Q/G JSON cells: three Q/G values are averaged within training seed, then mean ± sample SD is taken across the three training seeds. No model is retrained and no value is inferred from mAP summaries.

The accepted factorial audits contain complete Recall@K cells for the four factorial arms. The currently available accepted archive does not contain complete R@5/R@10/R@20 raw cells for every historical contextual baseline, so we do not synthesize those values; the main paper’s Supplement promise is correspondingly limited tofactorial Recall@K.

Table 5: NABirds Parent-disjoint group-macro mAP after averaging Q/G perturbations within training seed.
<table><tr><td>Arm</td><td>Seed</td><td>Class</td><td>Parent</td><td>Supergroup</td><td>Mean hierarchy</td></tr><tr><td> $E _ { R }$ </td><td>42</td><td>0.684085</td><td>0.201829</td><td>0.484932</td><td>0.343381</td></tr><tr><td> $E _ { R }$ </td><td>2024</td><td>0.685218</td><td>0.200846</td><td>0.484305</td><td>0.342575</td></tr><tr><td> $E _ { R }$ </td><td>3407</td><td>0.679945</td><td>0.195478</td><td>0.479702</td><td>0.337590</td></tr><tr><td> $H _ { R }$ </td><td>42</td><td>0.685569</td><td>0.196332</td><td>0.488446</td><td>0.342389</td></tr><tr><td> $H _ { R }$ </td><td>2024</td><td>0.688281</td><td>0.199166</td><td>0.486469</td><td>0.342818</td></tr><tr><td> $H _ { R }$ </td><td>3407</td><td>0.678195</td><td>0.195661</td><td>0.491257</td><td>0.343459</td></tr><tr><td> $E _ { S }$ </td><td>42</td><td>0.460400</td><td>0.229066</td><td>0.546453</td><td>0.387759</td></tr><tr><td> $E _ { S }$ </td><td>2024</td><td>0.467234</td><td>0.227171</td><td>0.545812</td><td>0.386491</td></tr><tr><td> $E _ { S }$ </td><td>3407</td><td>0.468878</td><td>0.232590</td><td>0.545936</td><td>0.389263</td></tr><tr><td> $H _ { S }$ </td><td>42</td><td>0.435353</td><td>0.224894</td><td>0.552094</td><td>0.388494</td></tr><tr><td> $H _ { S }$ </td><td>2024</td><td>0.434436</td><td>0.225294</td><td>0.549954</td><td>0.387624</td></tr><tr><td> $H _ { S }$ </td><td>3407</td><td>0.446344</td><td>0.223310</td><td>0.546061</td><td>0.384685</td></tr></table>

Table 6: NABirds mean-hierarchy group-macro simple effects by independent training seed.
<table><tr><td>Effect</td><td>42</td><td>2024</td><td>3407</td><td> $\mathrm { M e a n } \pm \mathrm { S D }$ </td></tr><tr><td> $\Delta _ { \mathrm { g e o m } } ^ { R }$ </td><td>-0.000992</td><td>+0.000242</td><td>+0.005869</td><td> $+ 0 . 0 0 1 7 0 6 \pm 0 . 0 0 3 6 5 7$ </td></tr><tr><td> $\Delta _ { \mathrm { g e o m } } ^ { S }$ </td><td>+0.000734</td><td>+0.001133</td><td>-0.004577</td><td> $- 0 . 0 0 0 9 0 4 \pm 0 . 0 0 3 1 8 8$ </td></tr><tr><td> $\Delta _ { \mathrm { l o s s } } ^ { E }$ </td><td>+0.044379</td><td>+0.043916</td><td>+0.051673</td><td> $+ 0 . 0 4 6 6 5 6 \pm 0 . 0 0 4 3 5 1$ </td></tr><tr><td> $\Delta _ { \mathrm { l o s s } } ^ { H }$ </td><td>+0.046105</td><td>+0.044806</td><td>+0.041226</td><td> $+ 0 . 0 4 4 0 4 6 \pm 0 . 0 0 2 5 2 7$ </td></tr><tr><td> $\Delta _ { \mathrm { i n t } }$ </td><td>+0.001726</td><td>+0.000890</td><td>-0.010446</td><td> $- 0 . 0 0 2 6 1 0 \pm 0 . 0 0 6 7 9 9$ </td></tr></table>

## S9 Semantic-Alignment Control

This experiment is separate from the fixed-100-epoch factorial. All arms use the same frozen DINOv2 features and 768–256–32 Euclidean projector, batch size 128, learning rate $1 0 ^ { - 4 }$ , weight decay $1 0 ^ { - 4 }$ , dropout 0.1, and temperature $\tau = 0 . 3 0$ . The supervision weights are Leaf-only (0.25, 0, 0), Leaf+Parent $( 0 . 2 5 , 0 . 5 0 , 0 )$ , and Full/Shuffle (0.25, 0.50, 1.00). Each arm selects its duration on development data without official-test access; the frozen epochs are 5, 47, 119, and 2 respectively.

The structure-preserving shuffle uses fixed seed 20260818, leaves leaf labels unchanged, preserves the multiset of classes per Parent and Parents per Supergroup, has no class-to-Parent fixed points, and necessarily retains 15 Parent-to-Supergroup fixed points under the group-size constraints. Full taxonomy minus shuffled taxonomy is +0.0849 Parent mAP, +0.2510 Supergroup mAP, and +0.1679 mean hierarchy mAP; all three paired training-seed directions are positive for these three contrasts. Leaf+Parent minus Leaf-only mean hierarchy is +0.0358 with all three seed directions positive; Full minus Leaf+Parent mean hierarchy is +0.0475, again positive for all three seeds, while Parent mAP changes by about −0.0034 and improves in only one of three seeds

Archive limitation. The accepted manuscript record contains the frozen means, sample SDs, epochs, shuffle construction, and paired-direction counts above. The exact per-seed semantic-ablation rows are not present in the currently available accepted audi ZIPs, so they are intentionally not reconstructed from means and standard deviations.

## S10 Curvature and Radius Mechanism Boundary

The mechanism study uses the NABirds Parent-disjoint matched-regression setting and treats the training seed as the statistica unit after averaging three Q/G perturbations. Four new hyperbolic configurations are trained for 84 epochs and compared with the previously accepted matched $c = 0 . 1 , r = 0 . 4$ and Euclidean controls.

At fixed $r = . 4$ , Base minus Near-flat mean hierarchy is $- 0 . 0 0 0 4 1 9 \pm 0 . 0 0 0 2 2 9$ and is negative for all three training seeds. Strong-curvature minus Base is $- 0 . 0 0 2 3 1 4 \pm 0 . 0 0 0 5 1 5$ , again with a consistent negative direction. Small-radius minus Base is $- 0 . 0 0 8 0 3 1 \pm 0 . 0 0 0 6 2 2$ and decreases in all three seeds. Large-radius minus Base averages −0.003269 ± 0.006595 with only one of three seeds negative, illustrating a non-monotonic level trade-off. Near-flat minus matched Euclidean is +0.002183 ± 0.003793 and is positive in only two seeds; because that comparison also changes normalization, exponential mapping, and the radius constraint, it is not treated as a pure-curvature intervention.

These results are used only as a mechanism boundary: they do not support attributing the small matched geometry differences to stronger negative curvature. Exact per-seed mechanism rows are not present in the currently available accepted audit archive and

Table 7: CUB factorial Recall@K (query-micro), mean ± sample SD over training seeds.
<table><tr><td>Arm</td><td>Level</td><td>R@1</td><td>R@5</td><td>R@10</td><td>R@20</td></tr><tr><td> $E _ { R }$ </td><td>Class</td><td> $0 . 7 7 1 8 \pm 0 . 0 0 2 5$ </td><td> $0 . 9 3 5 5 \pm 0 . 0 0 5 7$ </td><td> $0 . 9 6 5 6 \pm 0 . 0 0 4 9$ </td><td> $0 . 9 7 9 9 \pm 0 . 0 0 3 1$ </td></tr><tr><td> $E _ { R }$ </td><td>Genus</td><td> $0 . 2 4 1 7 \pm 0 . 0 0 6 6$ </td><td> $0 . 6 4 6 2 \pm 0 . 0 1 0 4$ </td><td> $0 . 8 1 3 4 \pm 0 . 0 0 7 2$ </td><td> $0 . 9 4 1 7 \pm 0 . 0 0 6 6$ </td></tr><tr><td> $E _ { R }$ </td><td>Family</td><td> $0 . 0 8 2 9 \pm 0 . 0 0 1 2$ </td><td> $0 . 2 2 9 2 \pm 0 . 0 0 6 5$ </td><td> $0 . 3 3 2 9 \pm 0 . 0 0 5 1$ </td><td> $0 . 5 1 2 8 \pm 0 . 0 0 9 0$ </td></tr><tr><td> $H _ { R }$ </td><td>Class</td><td> $0 . 7 8 9 5 \pm 0 . 0 0 2 7$ </td><td> $0 . 9 3 8 7 \pm 0 . 0 0 1 9$ </td><td> $0 . 9 6 5 0 \pm 0 . 0 0 3 3$ </td><td> $0 . 9 7 9 3 \pm 0 . 0 0 1 4$ </td></tr><tr><td> $H _ { R }$ </td><td>Genus</td><td> $0 . 2 3 2 8 \pm 0 . 0 0 1 1$ </td><td> $0 . 6 4 7 6 \pm 0 . 0 1 4 1$ </td><td> $0 . 8 1 7 0 \pm 0 . 0 1 7 6$ </td><td> $0 . 9 5 6 1 \pm 0 . 0 0 1 4$ </td></tr><tr><td> $H _ { R }$ </td><td>Family</td><td> $0 . 0 6 9 8 \pm 0 . 0 0 4 6$ </td><td> $0 . 1 9 7 1 \pm 0 . 0 1 0 3$ </td><td> $0 . 2 9 0 1 \pm 0 . 0 0 2 5$ </td><td> $0 . 4 5 5 9 \pm 0 . 0 0 8 7$ </td></tr><tr><td> $E _ { S }$ </td><td>Class</td><td> $0 . 6 5 9 5 \pm 0 . 0 0 6 3$ </td><td> $0 . 9 1 0 2 \pm 0 . 0 0 2 7$ </td><td> $0 . 9 5 6 1 \pm 0 . 0 0 1 2$ </td><td> $0 . 9 7 7 1 \pm 0 . 0 0 0 6$ </td></tr><tr><td> $E _ { S }$ </td><td>Genus</td><td> $0 . 3 8 2 0 \pm 0 . 0 0 9 5$ </td><td> $0 . 8 1 3 4 \pm 0 . 0 0 4 3$ </td><td> $0 . 9 0 9 6 \pm 0 . 0 0 3 8$ </td><td> $0 . 9 7 0 6 \pm 0 . 0 0 2 2$ </td></tr><tr><td> $E _ { S }$ </td><td>Family</td><td> $0 . 1 2 1 9 \pm 0 . 0 0 4 6$ </td><td> $0 . 3 0 2 9 \pm 0 . 0 0 5 8$ </td><td> $0 . 4 2 5 3 \pm 0 . 0 0 5 6$ </td><td> $0 . 6 1 1 1 \pm 0 . 0 0 4 2$ </td></tr><tr><td> $H _ { S }$ </td><td>Class</td><td> $0 . 6 0 0 6 \pm 0 . 0 0 4 4$ </td><td> $0 . 8 8 0 3 \pm 0 . 0 0 1 5$ </td><td> $0 . 9 4 3 9 \pm 0 . 0 0 0 8$ </td><td> $0 . 9 7 3 7 \pm 0 . 0 0 1 9$ </td></tr><tr><td> $H _ { S }$ </td><td>Genus</td><td> $0 . 4 0 5 2 \pm 0 . 0 1 7 6$ </td><td> $0 . 8 3 7 0 \pm 0 . 0 0 9 3$ </td><td> $0 . 9 3 6 6 \pm 0 . 0 0 1 8$ </td><td> $0 . 9 7 3 4 \pm 0 . 0 0 2 0$ </td></tr><tr><td> $H _ { S }$ </td><td>Family</td><td> $0 . 1 7 4 8 \pm 0 . 0 1 3 6$ </td><td> $0 . 4 0 5 5 \pm 0 . 0 0 3 9$ </td><td> $0 . 5 2 8 7 \pm 0 . 0 0 8 0$ </td><td> $0 . 6 7 6 0 \pm 0 . 0 0 7 6$ </td></tr></table>

Table 8: NABirds Parent-disjoint factorial Recall@K (group-macro), mean ± sample SD over training seeds.
<table><tr><td>Arm</td><td>Level</td><td>R@1</td><td>R@5</td><td>R@10</td><td>R@20</td></tr><tr><td> $E _ { R }$ </td><td>Class</td><td> $0 . 8 5 1 0 \pm 0 . 0 0 7 0$ </td><td> $0 . 9 5 4 5 \pm 0 . 0 0 5 9$ </td><td> $0 . 9 7 3 8 \pm 0 . 0 0 5 0$ </td><td> $0 . 9 8 6 7 \pm 0 . 0 0 3 5$ </td></tr><tr><td> $E _ { R }$ </td><td>Parent</td><td> $0 . 0 6 5 2 \pm 0 . 0 0 9 8$ </td><td> $0 . 2 0 7 9 \pm 0 . 0 0 8 3$ </td><td> $0 . 3 3 6 0 \pm 0 . 0 1 2 6$ </td><td> $0 . 5 0 9 0 \pm 0 . 0 1 8 8$ </td></tr><tr><td> $E _ { R }$ </td><td>Supergroup</td><td> $0 . 0 8 4 8 \pm 0 . 0 0 8 6$ </td><td> $0 . 2 5 6 1 \pm 0 . 0 0 5 2$ </td><td> $0 . 3 6 0 2 \pm 0 . 0 1 5 1$ </td><td> $0 . 5 4 1 0 \pm 0 . 0 1 9 9$ </td></tr><tr><td> $H _ { R }$ </td><td>Class</td><td> $0 . 8 5 0 7 \pm 0 . 0 0 2 7$ </td><td> $0 . 9 5 3 7 \pm 0 . 0 0 5 3$ </td><td> $0 . 9 7 6 2 \pm 0 . 0 0 1 2$ </td><td> $0 . 9 8 7 4 \pm 0 . 0 0 1 7$ </td></tr><tr><td> $H _ { R }$ </td><td>Parent</td><td> $0 . 0 6 1 5 \pm 0 . 0 0 2 9$ </td><td> $0 . 2 1 2 7 \pm 0 . 0 0 1 6$ </td><td> $0 . 3 3 4 8 \pm 0 . 0 1 1 2$ </td><td> $0 . 5 1 1 9 \pm 0 . 0 0 8 1$ </td></tr><tr><td> $H _ { R }$ </td><td>Supergroup</td><td> $0 . 0 8 7 5 \pm 0 . 0 0 8 7$ </td><td> $0 . 2 5 2 8 \pm 0 . 0 2 4 0$ </td><td> $0 . 3 6 6 2 \pm 0 . 0 1 9 9$ </td><td> $0 . 5 3 3 9 \pm 0 . 0 1 5 8$ </td></tr><tr><td> $E _ { S }$ </td><td>Class</td><td> $0 . 6 4 4 4 \pm 0 . 0 0 6 4$ </td><td> $0 . 8 5 8 7 \pm 0 . 0 0 3 3$ </td><td> $0 . 9 2 0 6 \pm 0 . 0 0 2 2$ </td><td> $0 . 9 5 5 3 \pm 0 . 0 0 0 4$ </td></tr><tr><td> $E _ { S }$ </td><td>Parent</td><td> $0 . 1 4 7 8 \pm 0 . 0 0 2 1$ </td><td> $0 . 4 3 3 8 \pm 0 . 0 0 6 6$ </td><td> $0 . 6 0 9 0 \pm 0 . 0 0 9 5$ </td><td> $0 . 7 9 6 5 \pm 0 . 0 1 7 1$ </td></tr><tr><td> $E _ { S }$ </td><td>Supergroup</td><td> $0 . 2 3 2 4 \pm 0 . 0 0 1 7$ </td><td> $0 . 5 6 0 3 \pm 0 . 0 1 2 5$ </td><td> $0 . 6 9 5 9 \pm 0 . 0 1 2 0$ </td><td> $0 . 8 1 4 8 \pm 0 . 0 1 0 1$ </td></tr><tr><td> $H _ { S }$ </td><td>Class</td><td> $0 . 6 0 6 4 \pm 0 . 0 0 7 3$ </td><td> $0 . 8 3 9 7 \pm 0 . 0 1 5 5$ </td><td> $0 . 9 0 9 6 \pm 0 . 0 0 8 7$ </td><td> $0 . 9 4 7 4 \pm 0 . 0 0 0 4$ </td></tr><tr><td> $H _ { S }$ </td><td>Parent</td><td> $0 . 1 6 4 7 \pm 0 . 0 0 9 6$ </td><td> $0 . 4 6 0 9 \pm 0 . 0 1 6 0$ </td><td> $0 . 6 2 8 7 \pm 0 . 0 1 1 0$ </td><td> $0 . 7 9 7 8 \pm 0 . 0 1 8 2$ </td></tr><tr><td> $H _ { S }$ </td><td>Supergroup</td><td> $0 . 2 6 4 3 \pm 0 . 0 0 3 8$ </td><td> $0 . 5 7 9 1 \pm 0 . 0 0 4 6$ </td><td> $0 . 7 1 6 9 \pm 0 . 0 2 2 4$ </td><td> $0 . 8 2 1 0 \pm 0 . 0 1 3 6$ </td></tr></table>

are therefore not reconstructed.

## S11 Harmonized CUB Group-Macro Reporting Sensitivity

After the CUB factorial was frozen, the 12 already frozen projected-test arrays were re-scored with group-macro aggregation using the NABirds aggregation family. No training, hyperparameter selection, or checkpoint selection was performed.

The reporting audit verified 12/12 frozen projected arrays and 36/36 Q/G cells. Recomputed query-micro metrics reconcile to the accepted primary results with maximum absolute mAP delta 0.00000679, below the frozen $\mathbf { \bar { 1 0 ^ { - 5 } } }$ tolerance. This sensitivity preserves the qualitative scale separation between loss and geometry but is explicitly post-hoc and does not replace CUB’s registered query-micro endpoint.

## S12 Fairness, Leakage, and Reproducibility Audit

For CUB, the hyperbolic ball radius is 0.7071 and the maximum audited final hyperbolic embedding norm across accepted runs is 0.3622; for NABirds the corresponding values are 3.1623 and 0.3979. Both therefore satisfy the audited ball constraint with substantial margin.

The official-test split hashes, model-state hashes, and lock hashes are retained in the accepted audit archives rather than repeated in the paper text. The final audits verify that official-test metrics were accessed only after the corresponding Final lock and after all 12 training runs were complete.

## S13 Archived-Evidence Limits

For transparency, the following reproducibility details are not fully recoverable from the currently available accepted artifact set:

Table 9: NABirds Parent-disjoint semantic-alignment control. Values are mean ± sample SD over training seeds after Q/G averaging within seed.
<table><tr><td>Arm</td><td>Class R@1</td><td>Class mAP</td><td></td><td>Parent mAP Supergroup mAP Mean hierarchy</td><td></td></tr><tr><td>Leaf-only</td><td> $. 8 4 4 7 \pm . 0 0 1 3$ </td><td> $. 6 7 1 7 \pm . 0 0 6 7$ </td><td> $. 1 9 0 1 \pm . 0 0 1 3$ </td><td> $. 4 2 3 1 \pm . 0 0 2 5$ </td><td> $. 3 0 6 6 \pm . 0 0 1 9$ </td></tr><tr><td>Leaf+Parent</td><td> $. 8 0 9 6 \pm . 0 0 1 8$ </td><td> $. 6 1 7 8 \pm . 0 0 7 2$ </td><td> $. 2 4 4 1 \pm . 0 0 7 2$ </td><td> $. 4 4 0 7 \pm . 0 0 5 9$ </td><td> $. 3 4 2 4 \pm . 0 0 1 5$ </td></tr><tr><td>Full taxonomy</td><td> $. 7 4 7 8 \pm . 0 0 7 8$ </td><td> $. 5 4 5 9 \pm . 0 0 3 0$ </td><td> $. 2 4 0 8 \pm . 0 0 5 6$ </td><td> $. 5 3 9 1 \pm . 0 0 2 4$ </td><td> $. 3 8 9 9 \pm . 0 0 3 1$ </td></tr><tr><td>Shuffled taxonomy</td><td> $. 8 1 2 6 \pm . 0 0 4 3$ </td><td> $. 5 8 7 4 \pm . 0 0 3 4$ </td><td> $. 1 5 5 9 \pm . 0 1 0 5$ </td><td> $. 2 8 8 1 \pm . 0 1 1 3$ </td><td> $. 2 2 2 0 \pm . 0 1 0 7$ </td></tr></table>

Table 10: NABirds Parent-disjoint curvature/radius mechanism control. Mean ± sample SD over training seeds.
<table><tr><td>Configuration</td><td>Parent mAP</td><td> $\mathrm { S u p e r g r o u p m A P }$ </td><td>Mean hierarchy</td></tr><tr><td>Near-flat  $c = 1 0 ^ { - 6 } , r = . 4$ </td><td> $. 1 9 6 6 0 9 \pm . 0 0 1 5 1 6$ </td><td> $. 4 8 9 1 8 8 \pm . 0 0 0 1 5 1$ </td><td> $. 3 4 2 8 9 8 \pm . 0 0 0 6 8 2$ </td></tr><tr><td>Base  $c = . 1 , r = . 4$ </td><td> $. 1 9 6 3 0 8 \pm . 0 0 1 2 5 6$ </td><td> $. 4 8 8 6 5 0 \pm . 0 0 0 3 5 6$ </td><td> $. 3 4 2 4 7 9 \pm . 0 0 0 4 5 7$ </td></tr><tr><td>Strong curvature  $c = 1 , r = . 4$ </td><td> $. 1 9 5 2 6 9 \pm . 0 0 0 4 7 2$ </td><td> $. 4 8 5 0 6 1 \pm . 0 0 0 2 1 6$ </td><td> $. 3 4 0 1 6 5 \pm . 0 0 0 1 7 3$ </td></tr><tr><td>Small radius  $c = . 1 , r = . 2$ </td><td> $. 1 9 7 7 7 0 \pm . 0 0 1 6 4 6$ </td><td> $. 4 7 1 1 2 7 \pm . 0 0 3 4 4 0$ </td><td> $. 3 3 4 4 4 9 \pm . 0 0 1 0 6 8$ </td></tr><tr><td>Large radius  $c = . 1 , r = . 8$ </td><td> $. 1 9 9 8 4 6 \pm . 0 0 4 0 8 1$ </td><td> $. 4 7 8 5 7 4 \pm . 0 0 9 4 1 8$ </td><td> $. 3 3 9 2 1 0 \pm . 0 0 6 6 4 2$ </td></tr><tr><td>Matched Euclidean</td><td> $. 1 9 8 7 3 5 \pm . 0 0 3 1 6 3$ </td><td> $. 4 8 2 6 9 6 \pm . 0 0 4 0 4 5$ </td><td> $. 3 4 0 7 1 5 \pm . 0 0 3 5 9 4$ </td></tr></table>

• the exact raw per-seed rows of the semantic-alignment ablation;

• the exact raw per-seed rows of the curvature/radius mechanism study;

• the realized CUB taxonomy mapping CSV, manual-alias JSON, and external eBird/Clements v2025 source CSV;

• the exact final Q/G NPZ index arrays, although the project split utilities and accepted seeds/sizes/SHA256 identifiers are now included;

• the Geoopt package version, complete environment freeze, and library-internal projx boundary epsilon.

None of these gaps is filled by back-calculation, version guessing, or regeneration presented as original data. The main factorial, temperature-selection, Q/G Recall@K, seed-level effects, and fairness/test-seal claims are supported directly by the accepted JSON/lock audits

Table 11: Post-hoc CUB group-macro mean-hierarchy effects.
<table><tr><td>Effect</td><td>42</td><td>2024</td><td>3407</td><td> $\mathrm { M e a n } \pm \mathrm { S D }$ </td></tr><tr><td> $\Delta _ { \mathrm { g e o m } } ^ { R }$ </td><td>+0.008632</td><td>+0.008104</td><td>+0.009976</td><td> $+ 0 . 0 0 8 9 0 4 \pm 0 . 0 0 0 9 6 5$ </td></tr><tr><td> $\Delta _ { \mathrm { g e o m } } ^ { S }$ </td><td>+0.004214</td><td>+0.004807</td><td>+0.004921</td><td> $+ 0 . 0 0 4 6 4 7 \pm 0 . 0 0 0 3 8 0$ </td></tr><tr><td> $\Delta _ { \mathrm { l o s s } } ^ { E }$ </td><td>+0.061003</td><td>+0.062655</td><td>+0.062948</td><td> $+ 0 . 0 6 2 2 0 2 \pm 0 . 0 0 1 0 4 8$ </td></tr><tr><td> $\Delta _ { \mathrm { l o s s } } ^ { H }$ </td><td>+0.056585</td><td>+0.059358</td><td>+0.057892</td><td> $+ 0 . 0 5 7 9 4 5 \pm 0 . 0 0 1 3 8 7$ </td></tr><tr><td> $\Delta _ { \mathrm { i n t } }$ </td><td>-0.004418</td><td>-0.003296</td><td>-0.005055</td><td> $- 0 . 0 0 4 2 5 6 \pm 0 . 0 0 0 8 9 0$ </td></tr></table>

Table 12: Final factorial fairness and leakage checks from the accepted pre-test/final audits.
<table><tr><td>Check</td><td>CUB</td><td>NABirds Parent-disjoint</td></tr><tr><td>Training samples</td><td>5,994</td><td>19,021</td></tr><tr><td>Trainable parameters/cell</td><td>205,600</td><td>205,600</td></tr><tr><td>Batch size</td><td>128</td><td>128</td></tr><tr><td>Epochs/cell</td><td>100</td><td>100</td></tr><tr><td>Updates/epoch</td><td>46</td><td>148</td></tr><tr><td>Training seeds</td><td>42/2024/3407</td><td>42/2024/3407</td></tr><tr><td>Q/G seeds</td><td>42/2024/3407</td><td>42/2024/3407</td></tr><tr><td>Final training runs</td><td>12/12 verified</td><td>12/12 verified</td></tr><tr><td>Official Q/G cells</td><td>36/36 verified</td><td>36/36 verified</td></tr><tr><td>Same trainable init within seed</td><td>pass</td><td>pass</td></tr><tr><td>Same batch order within seed</td><td>pass</td><td>pass</td></tr><tr><td>Same parameter-update budget</td><td>pass</td><td>pass</td></tr><tr><td>Checkpoint reload max abs. diff.</td><td>0</td><td>0</td></tr><tr><td>Temperature-tune checkpoints reused</td><td>no</td><td>no</td></tr><tr><td>All final arms retrained from scratch</td><td>yes</td><td>yes</td></tr><tr><td>Configuration selection used test</td><td>no</td><td>no</td></tr><tr><td>Training finished before test metrics</td><td>yes</td><td>yes</td></tr><tr><td>Official test read after Final lock</td><td>yes</td><td>yes</td></tr></table>