# PointRL: Learning Point-Level Vision-Language Grounding from Verifiable Annotation Evidence

Jingyang Su, Pu Cao, Xiuze Jin, Longyue Zhang, Qing Song, and Lu Yang

Abstract—Vision-language models (VLMs) increasingly rely on point coordinates as a compact and executable interface for visual grounding in GUI interaction, robotic manipulation, and interactive visual systems. However, learning reliable pointing behavior remains difficult because the supervision space is inherently non-unique: many coordinates may be valid within the same target region, while multi-instance instructions require target coverage, count consistency, and duplicate suppression. This work presents PointRL, a verifiable reinforcement learning framework that learns point-level grounding from existing heterogeneous annotation evidence. PointRL converts bounding boxes, masks, and instance labels into pointing instructions, while retaining their target supports, instance membership, and set constraints as hidden verifier evidence, i.e., annotations kept outside the prompt and used by a deterministic checker to score predictions. The proposed reward evaluates parseability, point validity, instance coverage, cardinality consistency, and redundant or missing predictions. On PointArena, PointRL improves the overall accuracy of Qwen3.5-4B from 56.11% to 65.58%. Further evaluations on RoboSpatial, BLINK, and Ref-Adv show same-backbone gains on the evaluated external benchmarks, suggesting that verifiable point-level feedback may benefit spatial grounding in these settings.

Index Terms—Vision-language models, visual grounding, point-level grounding, verifiable reinforcement learning, heterogeneous annotations, spatial reasoning

## I. INTRODUCTION

ision-language grounding is a fundamental capability of vision-language models (VLMs), requiring a model to connect natural-language instructions with objects, regions, and locations in visual scenes. This capability supports a broad range of interactive and embodied applications, including GUI agents, robotic manipulation, assistive perception, and human-in-the-loop visual systems [1–3]. Although grounding is commonly represented by bounding boxes or segmentation masks, point coordinates provide a more compact and directly executable interface. A point can indicate a click position, a manipulation target, or an interactive segmentation prompt, and VLMs can generate such coordinates as text using standard decoding. The prompt-response framing follows instruction-tuning style training [4].

![](images/733e4cd776dc9e1aa091806bae1d839359a9bf56b85a7fb080fe2cb0ca8afcd6.jpg)  
Fig. 1. Task illustration for verifiable point-level grounding. A valid response should place points on the intended target supports, while nearby distractors, incomplete coverage, or shortcut outputs are rejected by the verifier.

However, learning reliable pointing behavior goes beyond standard single-label prediction. For a target region, many coordinates may be correct as long as they fall inside the valid spatial support. Reducing such supervision to a single fixed point label unnecessarily collapses a set of valid answers and may penalize equally acceptable predictions. Recent robust visual grounding studies have also emphasized ambiguity-aware supervision [5–7]. The difficulty becomes more pronounced in instruction-conditioned settings. Affordance-oriented queries require the model to identify functionally relevant regions, spatial-relation queries require reasoning over relative positions, and multi-instance queries require a set of distinct points that covers all target instances with the correct cardinality. Therefore, the central challenge is to learn point predictions that satisfy both geometric validity and instruction-specific set constraints.

Existing grounding annotations provide useful evidence for connecting language to visual entities. Bounding boxes and segmentation masks specify target supports on which a target may lie, while instance labels define sets of objects that may satisfy an instruction. PointRL treats these raw annotations as annotation evidence and normalizes them into hidden verifier evidence V = (T, C), where T stores the targets and C stores instruction-conditioned constraints [8, 9]. This representation is kept outside the prompt and used as an executable interface for reward computation.

This work introduces PointRL for verifiable reinforcement learning of visual pointing from heterogeneous grounding annotations. PointRL converts boxes, masks, and instance labels into natural-language pointing instructions, while preserving the original annotations as hidden verifier evidence. During training, the VLM generates a textual response that is parsed into one or more coordinates, and a deterministic verifier computes a structured reward from the predicted point collection and V. The reward evaluates output parseability, point validity, localization quality, instance coverage, cardinality consistency, and redundant or missing predictions. In this way, PointRL preserves the non-unique answer space of region-based supervision while handling point annotations through distancebased scoring, and it transforms heterogeneous annotations into executable verifier evidence for point-level learning.

The evaluation asks whether verifier-based point feedback improves same-backbone grounding on PointArena and several external spatial grounding benchmarks. On PointArena, PointRL improves the overall accuracy of Qwen3.5-4B from 56.11% to 65.58%. Additional evaluations on RoboSpatial, BLINK, and Ref-Adv show same-backbone gains on the evaluated external benchmarks, suggesting that verifiable point-level feedback may benefit spatial grounding in these settings.

The contributions are threefold. First, PointRL builds prompts and hidden verifier evidence from annotations. This evidence stores target supports, target sets, and instructionconditioned constraints. Second, it defines a hierarchical reward mechanism that handles non-unique valid points and multi-instance constraints, including localization, coverage, cardinality consistency, and duplicate suppression. Third, same-backbone evaluations on in-domain and external benchmarks provide evidence that verifiable feedback derived from heterogeneous grounding annotations improves point-level grounding in the evaluated settings.

## II. RELATED WORK

## A. Visual Grounding and Spatial Evidence

Visual grounding aims to localize visual entities described by natural language. Early two-stage methods typically generate candidate regions and rank them according to their semantic correspondence with the query, while end-to-end approaches such as MDETR [1] and Grounding DINO [2] directly predict language-conditioned bounding boxes. Referring-expression grounding has also been studied with modular attention models such as MattNet and neural module-tree approaches [10, 11]. With the development of multimodal large language models, grounding has been extended from box prediction to richer spatial output formats [5– 7]. For example, LISA [12] and GLaMM [13] study maskbased referring segmentation, while LGR-NET [5], CMIR-Net [8], RefSAM [9], and SSP-SAM [14] focus on referring image segmentation under richer supervision cues. LocateAnything [15] and PaDT [16] explore unified spatial outputs through specialized decoding mechanisms or large-scale grounding data engines. AD-DINO [17] further links attentiondynamic reasoning with embodied reference understanding.

Recent studies also emphasize fine-grained spatial understanding in structured scenes, including point-guided localization [18], instance-level human parsing [19–21], and omnidirectional person positioning [22, 23]. These works show that visual grounding combines localization with spatial structure and relation-aware target selection [24–26]. However, existing grounding supervision remains highly heterogeneous, including boxes, masks, instance labels, and relation metadata, which has motivated generalized multi-task grounding formulations [27]. Such heterogeneity makes it difficult to define a unified learning signal across datasets and tasks. PointRL addresses this issue by treating points as a shared prediction interface and by using heterogeneous annotations as hidden verifier evidence that provides spatial and instancelevel knowledge for reward computation.

## B. Language-Guided Pointing

Point prediction has recently emerged as a compact interface for language-guided grounding. In this format, a model outputs one or more image coordinates that directly indicate target locations. This format is particularly suitable for interactive systems, because a point can correspond to a click, a manipulation target, or a prompt for downstream segmentation. Molmo and PixMo [28] show that modern multimodal models can acquire pointing behavior from large-scale point supervision, including counting-by-point for multi-instance references. Molmo-Point [29] further improves pointing accuracy by introducing grounding tokens and visual-token selection, representing an architecture-oriented approach to point prediction.

Dedicated benchmarks have also been developed to evaluate pointing ability. PointArena [30] formalizes language-guided pointing through Point-Bench and Point-Battle protocols, covering spatial, affordance, counting, steerability, and reasoningoriented instructions. Pointing also serves as an action interface in robotic manipulation [31] and GUI grounding [32], where the predicted coordinate can be executed by an external system. PointRL follows a verifier-based route for learning pointing behavior from existing boxes, masks, and instance labels. The framework preserves verifiable target regions during reward learning and converts them into point-level feedback.

## C. Reinforcement Learning with Verifiable Rewards

Reinforcement learning with verifiable rewards uses rulebased or automatically computable outcome signals as training feedback for model behavior. This paradigm has been demonstrated in language reasoning by DeepSeek-R1 [33] and has recently been extended to vision-language models. Reward design can follow potential-based invariance principles to avoid changing the optimal policy [34]. Visual-RFT [35] applies reinforcement fine-tuning to visual perception tasks, VLM-R1 [36] adopts GRPO-style optimization for visual reasoning, and GenSeg-R1 [37] introduces segmentation-quality feedback for referring segmentation.

Most existing visual verifiable rewards are designed around task-level correctness or region-level quality, such as answer accuracy, box overlap, or mask overlap. Point-level grounding requires rewards that handle multiple valid coordinates for the same target region, as well as coverage, count consistency, and duplicate suppression for multi-instance instructions. PointRL defines rewards over unordered point collections. Its verifierbased reward combines soft localization, point-target matching, and set-level constraints, allowing heterogeneous grounding annotations to supervise point prediction while preserving the non-unique answer space for region-based supports.

III. POINTRL-DATA: CONSTRUCTING VERIFIABLE POINTING DATA FROM GROUNDING ANNOTATIONS

## A. Task Formulation

A pointing task instance consists of an image I, a naturallanguage instruction $q ,$ and hidden verifier evidence $V =$ $( T , C )$ . The instruction specifies the referent or referents that should be pointed to in the image. During execution, the model receives only I and q, and returns a collection of points,

$$
P = ( p _ { i } ) _ { i = 1 } ^ { n } , \quad p _ { i } = ( x _ { i } , y _ { i } ) .\tag{1}
$$

The required size of $P$ is determined jointly by the instruction and the image content. A query for a single referent calls for one point, whereas a query over all matching instances calls for one point per visible target instance. Because target instances have no canonical order, verification treats $P$ as an unordered collection: point order is ignored, while repeated coordinates are kept as separate predictions so missing, extra, or duplicated points can be penalized.

Correctness is evaluated with $V = ( T , C )$ by a deterministic checker. The target set $T = \{ t _ { j } \} _ { j = 1 } ^ { m }$ contains the entries that should be pointed to. Each entry $t _ { j }$ identifies an intended object, region, or location, and stores its support $S _ { j }$ , the geometric region or point used for hit checking, together with metadata needed to distinguish the intended instance. The constraint set $C$ records requirements that cannot be checked by membership in the support region alone, such as binding to a reference object, relation constraints, cardinality, uniqueness, and functional restrictions.

A response is correct only when the returned point collection matches both the target entries and the constraints. For a single-target instruction, one point must fall inside the intended support. For multi-instance or constrained instructions, the collection must cover the required distinct targets, have the target-set cardinality, avoid duplicate matches, and satisfy all constraints in C.

This formulation separates the task input from the evidence used to judge the answer. The next subsection describes how we construct the instructions, target entries, and constraints from heterogeneous annotations.

## B. Dataset Construction

For each image, we convert heterogeneous raw annotations into the verifier interface defined above. The construction pipeline first normalizes source annotations into object entities, then collects the evidence needed to decide which entities can serve as targets or references, and finally applies construction rules to instantiate an instruction q with hidden verifier evidence $V = ( T , C )$ . Here, T stores the target entries and their supports, while $C$ stores the conditions used by the deterministic checker. Figure 2 summarizes this pipeline.

Annotation normalization maps source-specific formats into a common entity set $\textit { E } = \{ e _ { i } \} _ { i = 1 } ^ { N }$ . Each visual referent is represented with an identifier, an optional semantic label, support geometry, and source metadata. The normalized entities are then collected into an object-evidence structure $G _ { I } \ = \ ( E , A )$ . In this structure, E lists the object entities, while A records usable evidence attached to entities or entity pairs, including label agreement, spatial relations, count cues, geometric supports, and functional roles. This entity-centered representation follows prior entity-relation formulations for connecting language and vision [38], while keeping the evidence explicit for verification.

Construction rules instantiate $q$ and $V$ from $G _ { I } ,$ , following a rule-based scheme related to programmatic weak supervision for data construction [39, 40]. Each rule selects a construction pattern, such as category, attribute, relation, counting, region, or functional-role grounding, binds the required target and optional reference entities, checks that the collected evidence supports the instruction, and accepts the sample only when the answer can be reduced to a verified target set. The accepted binding fixes the target supports in $T$ and records additional requirements in $C ,$ such as category membership, attributes, relations, cardinality, uniqueness, region conditions, or functional-role constraints.

Finally, a Qwen-based refiner [41] is used to rewrite the surface form of $q$ under the accepted rule binding. It is prompted to preserve target categories, counts, reference objects, relations, and functional restrictions, while the verifier fields remain fixed. Before a sample is kept, deterministic checks validate target supports, target unambiguity, constraint consistency, cardinality consistency, and alignment between the rule binding and the hidden verifier evidence. The verifier fields are organized around three requirements:

• Valid target support: each target stores a support $S _ { j }$ for hit checking and soft localization.

• Required target instances: multi-target samples store the target set $T = \{ t _ { j } \} _ { j = 1 } ^ { m }$ and cardinality constraints for unordered matching.

• Instruction-specific constraints: relation, functional, reference-anchor, and shortcut-control conditions are stored in C.

## C. Dataset Statistics

The accepted samples draw on AGD20K labels [42], PixMo points [28], COCO masks and instance metadata [43], and rule-bound cue-target pairs. Table I summarizes the resulting rule families. In our experiments, this construction yields 1,647 training samples for reward learning and a 200-sample heldout diagnostic split.

The distribution is intentionally not dominated by a single rule family. Relation and cue-based constructions account for 393 and 398 samples, respectively, or 48.0% of the accepted set together. These two families stress instructionconditioned constraints in $C ,$ such as reference binding, spatial predicates, and shortcut control. Instance and semantic constructions contribute 345 and 249 samples, covering multiinstance counting, unordered target coverage, and category level matching. Functional samples add 262 region-oriented examples, which exercise support validity for affordancestyle targets. This mixture gives the verifier both geometric evidence, such as masks, boxes, points, and regions, and nongeometric conditions, such as counts, relations, and cue-target constraints.

![](images/ec4d1d35156218fe1fac85302e916e96b0afcfad7d607afd38dcc2f5101b4c7a.jpg)  
Fig. 2. Construction of verifiable pointing samples from raw annotations. The pipeline normalizes annotation sources into object entities, collects supporting evidence, and generates the instruction with hidden verifier evidence $V = ( T , { \dot { C } } )$ . The target set and constraints are then fixed by rule-based checks.

TABLE I  
COMPOSITION OF ACCEPTED POINTRL-DATA RULES.
<table><tr><td>Rule</td><td></td><td>Count Evidence</td><td>Check</td></tr><tr><td>Functional</td><td>262</td><td>AGD regions</td><td>Affordance support</td></tr><tr><td>Instance</td><td>345</td><td>COCO/PixMo instances</td><td>Coverage/count</td></tr><tr><td>Semantic</td><td>249</td><td>Class labels</td><td>Category match</td></tr><tr><td>Relation</td><td></td><td>393 Entity pairs</td><td>Spatial relation</td></tr><tr><td>Cue</td><td>398</td><td>Cue-target pairs</td><td>Shortcut guard</td></tr><tr><td>Total</td><td></td><td>1,647 Mixed evidence</td><td>Rule-checked targets</td></tr></table>

## IV. POINTRL: REINFORCEMENT LEARNING FOR POINTING

PointRL learns point-level grounding by using hidden verifier evidence to score each sampled response. Given an input $ { \boldsymbol { z } } ^ { \mathrm { ~ ~ } } = \{ I , q \}$ , the VLM policy samples a textual response y. During reward computation, the hidden verifier evidence $V ~ = ~ ( T , C )$ provides the target set $\begin{array} { c c l } { { T } } & { { = } } & { { \{ t _ { j } \} _ { j = 1 } ^ { m } } } \end{array}$ and instruction-conditioned constraints C. Single-target pointing corresponds to $m = 1$ , while multi-target pointing has $m > 1$ and requires the predicted points to cover the target set with the correct cardinality.

Figure 3 gives an overview of the reward computation. The reward module first parses y through a deterministic parser and format gate to obtain the standard point collection P. It then scores P against V : predicted points are matched to verifier targets, evaluated by local localization quality and global coverage, count, and duplicate consistency, and adjusted by guard terms that suppress instruction-specific shortcuts such as copying a reference cue. These terms are aggregated into the core verifier point reward. Section IV-A defines parsing and format validation, Section IV-B describes point-to-target matching, and Section IV-C gives the final reward composition. Implementation-specific serialization is described in Section IV-D; all equations in this section operate on the standard point collection P.

## A. Point Prediction in VLM

The parser maps the terminal answer block of y to the standard point collection defined in Eq. (1), where n is the number of parsed points. A response is parseable only if the accepted final-answer schema can be normalized to exactly one coordinate list in the form

$$
[ ( x _ { 1 } , y _ { 1 } ) , \dots , ( x _ { n } , y _ { n } ) ] ,
$$

with $n \geq 1$ , two finite numeric coordinates in each tuple, and all coordinates within the image bounds after conversion to the verifier pixel coordinate system. Empty lists, malformed or non-numeric tuples, out-of-bound coordinates, missing point lists, and multiple conflicting final-answer blocks are treated as unparseable.

We set $r _ { \mathrm { f m t } } ~ = ~ 1$ for parseable responses and $r _ { \mathrm { f m t } } ~ = ~ 0$ otherwise. When $r _ { \mathrm { f m t } } ~ = ~ 0 .$ , the final point reward is zero; when $r _ { \mathrm { f m t } } = 1$ , P is passed to the hidden verifier evidence $V = ( T , C )$ for matching and reward computation. Duplicate coordinates are retained as separate indexed predictions and handled by the matching and set-level reward terms, including duplicate suppression. The JSON format used in the experiments is an experiment-specific final-answer schema that is normalized back to this point collection interface, as described in Section V-A.

## B. Matching Predicted Points to Targets

After parsing a valid point collection $P = ( p _ { i } ) _ { i = 1 } ^ { n }$ , the verifier compares these unordered predictions with the target set $T = \{ t _ { j } \} _ { j = 1 } ^ { m }$ . This subsection addresses two steps: scoring each prediction-target pair and assigning predictions to targets without double counting. A binary hit indicator is useful for hard correctness, but it is too sparse as a reward signal: a point just outside the valid target support and a point far from the object would both receive zero credit. PointRL therefore uses the distance from each prediction to each target support as the basis for localization scoring, and then performs one-toone matching over the resulting score matrix. This subsection focuses on the target supports in $T ;$ the remaining instructionconditioned constraints in $C$ are applied by the guard terms in Section IV-C.

![](images/d3fdd648802e711db63b26744022967c2aff740287c779ea5838894337651ef5.jpg)  
Fig. 3. Point reward computation from hidden verifier evidence. The reward module parses the final answer into a point collection, matches predictions to hidden verifier targets with Hungarian matching, and combines matched-pair localization quality with set-level coverage and count checks. Format and instruction-conditioned guards suppress invalid or shortcut responses before returning the core verifier point reward.

Each verifier target $t _ { j }$ carries a set-valued target support $S _ { j }$ For region supervision, $S _ { j }$ is the valid region derived from a box or mask; for point supervision, $S _ { j } = \{ g _ { j } \}$ is a singleton support at the annotated point. For a predicted point $p _ { i }$ , we define its distance to $t _ { j }$ as

$$
\begin{array} { r } { d ( p _ { i } , t _ { j } ) = d ( p _ { i } , S _ { j } ) = \left\{ \begin{array} { l l } { 0 , } & { p _ { i } \in S _ { j } , } \\ { \operatorname* { i n f } _ { u \in S _ { j } } \| p _ { i } - u \| _ { 2 } , } & { p _ { i } \notin S _ { j } . } \end{array} \right. } \end{array}\tag{2}
$$

This definition follows the verifier semantics while making localization errors graded rather than all-or-nothing. For a region target, any point inside $S _ { j }$ is a valid hit and has zero distance, whereas an outside point is scored by its distance to the nearest point in the support. For a singleton point support, the distance reduces to the Euclidean distance to the annotated point.

Rather than using this distance as a hard accept-or-reject

signal, PointRL converts it into a soft localization score:

$$
s ( p _ { i } , t _ { j } ) = \exp \left( - \frac { d ( p _ { i } , t _ { j } ) ^ { 2 } } { 2 \sigma ^ { 2 } } \right) .\tag{3}
$$

Here, $\sigma$ is the localization bandwidth. The score is maximized on the target support and decreases smoothly as the distance grows. This gives near-miss predictions a useful learning signal while keeping the verifier target as the reward reference. The hard support-membership criterion remains $p _ { i } ~ \in ~ S _ { j } \colon$ the soft score determines the amount of localization credit assigned to a matched pair.

For region targets, soft distance alone can over-credit points just outside a valid box or mask, because their distance to the nearest point in $S _ { j }$ can be small even when they are outside the support. To preserve the distinction between a valid hit and a near miss, PointRL caps only outside-region predictions:

$$
\begin{array} { r } { \bar { s } ( p _ { i } , t _ { j } ) = \operatorname* { m i n } \left( s ( p _ { i } , t _ { j } ) , \tau ( S _ { j } ) \right) . } \end{array}\tag{4}
$$

Inside-region predictions and singleton point supports use the uncapped score $\bar { s } ( p _ { i } , t _ { j } ) ~ = ~ s ( p _ { i } , t _ { j } )$ . Here, $\tau ( S _ { j } ) ~ \in ~ [ 0 , 1 ]$ denotes the cap associated with the region support type. The cap preserves dense feedback for near misses, but reserves high localization scores for points inside the target support. Mask or annotated-region supports use the highest cap, boxderived supports use an intermediate cap, and weakly localized or heuristic supports use the lowest cap. The experimental caps are fixed by support reliability as described in Section V-A.

After this step, every prediction-target pair has a comparable capped score. These scores form the matrix $M \in \mathbb { R } ^ { n \times m }$ :

$$
M _ { i j } = \bar { s } ( p _ { i } , t _ { j } ) .\tag{5}
$$

Because neither the predicted point collection nor the target set has a canonical order, the verifier must decide which prediction should be compared with which target. The counts n and m may also differ when the response contains extra or missing points. PointRL therefore selects the best one-to-one matching with Hungarian matching [44, 45]:

$$
\pi ^ { \star } = \arg \operatorname* { m a x } _ { \pi } \sum _ { ( i , j ) \in \pi } M _ { i j } ,\tag{6}
$$

where $\pi$ ranges over one-to-one matchings with cardinality $| \pi | = \operatorname* { m i n } ( n , m )$ , so each predicted point and each target can be matched at most once. Low-quality pairs may still be matched, but they contribute little because their $M _ { i j }$ values are low. The resulting assignment $\pi ^ { \star }$ separates the response into matched prediction-target pairs and, when $\textit { n } \ne \textit { m }$ unmatched predictions or missed targets. The next subsection turns these outcomes into the final reward: matched pairs provide localization credit, while extra predictions, missed targets, duplicates, and instruction-specific failures are handled by collection-level terms and guards.

## C. Hierarchical Pointing Reward

Given the matching assignment $\pi ^ { \star }$ , this subsection builds the response-level reward in three stages: first converting matched/unmatched outcomes into local matching quality, then adding collection-level consistency terms, and finally applying instruction-conditioned guards for parseability and robustness. The same formulation applies to both single-target and multitarget pointing. When $m = 1$ , the target set is a singleton: coverage reduces to the matched target credit, while count and duplicate terms penalize extra or repeated predictions. When $m > 1$ , the same terms additionally enforce coverage and cardinality consistency across the required target entries. For parseable responses, PointRL first combines matched-pair localization quality with collection-level consistency to obtain a base reward, and then applies instruction guards. The format indicator $r _ { \mathrm { f m t } }$ is determined during parsing and gates the final reward at the end; the terms below are evaluated only for parseable responses with $n \geq 1$ , and all constructed pointing samples have $m \geq 1$

The local reward scores the matched pairs under two normalizations. The prediction-side term measures the average matched credit per predicted point, while the target-side term measures the average matched credit per required target:

$$
S _ { \mathrm { p r e d } } = \frac { 1 } { n } \sum _ { ( i , j ) \in \pi ^ { \star } } M _ { i j } , \qquad S _ { \mathrm { c o v } } = \frac { 1 } { m } \sum _ { ( i , j ) \in \pi ^ { \star } } M _ { i j } .\tag{7}
$$

Unmatched predictions remain in the denominator of $S _ { \mathrm { p r e d } } ,$ and unmatched target entries remain in the denominator of $S _ { \mathrm { c o v } }$ , so both contribute zero credit. When $m = 1$ and $n = 1$ these two terms reduce to the matched localization score for the single target. When the model returns extra points or misses required targets, the two normalizations expose these errors from the prediction and target sides, respectively. The local reward combines these two views of matched-pair quality:

$$
R _ { \mathrm { l o c a l } } = \lambda S _ { \mathrm { p r e d } } + ( 1 - \lambda ) S _ { \mathrm { c o v } } .\tag{8}
$$

Here, $\lambda ~ \in ~ [ 0 , 1 ]$ controls the balance between predictionside and target-side credit. Thus, extra predictions reduce the prediction-side score, and missed target entries reduce the target-side coverage score.

Local matching quality alone does not ensure that the returned point collection has the required cardinality or structure. In the singleton case, a response may contain a well-localized point but also include extra points. In the multi-target case, it may localize some targets while missing others or collapse several predictions into the same local region. We therefore define global terms for count consistency and near-duplicate suppression. Count consistency compares the predicted count n with the required target count m:

$$
S _ { \mathrm { c n t } } = \exp \left( { - \gamma _ { o } \frac { ( n - m ) _ { + } } { \operatorname * { m a x } ( m , 1 ) } - \gamma _ { u } \frac { ( m - n ) _ { + } } { \operatorname * { m a x } ( m , 1 ) } } \right) ,\tag{9}
$$

where $( x ) _ { + } = \operatorname* { m a x } ( x , 0 )$ . The coefficients $\gamma _ { o }$ and $\gamma _ { u }$ control the penalties for over-counting and under-counting, respectively, and we use $\gamma _ { u } > \gamma _ { o }$ to penalize missed targets more strongly.

To discourage local duplicate collapse, we define a duplicate suppression term:

$$
S _ { \mathrm { d u p } } = \left\{ \begin{array} { l l } { 1 - \displaystyle \frac { 1 } { \binom { n } { 2 } } \sum _ { 1 \leq i < k \leq n } \mathbb { I } [ | | p _ { i } - p _ { k } | | _ { 2 } < \delta _ { \mathrm { d u p } } ] , } & { n \geq 2 , } \\ { 1 , } & { n < 2 . } \end{array} \right.\tag{10}
$$

Here, $\delta _ { \mathrm { d u p } }$ is the duplicate-distance threshold in pixels, fixed at $^ { 5 }$ pixels in our experiments. The term averages over all unordered pairs of predicted points, so the penalty is normalized by the number of possible point pairs. Duplicate coordinates are retained by the parser as separate predictions; the reward handles near-identical collapse through duplicate suppression and, when applicable, through the prediction-side, count, and coverage terms.

The global reward is defined as the product of target coverage, count consistency, and duplicate suppression:

$$
R _ { \mathrm { g l o b a l } } = S _ { \mathrm { c o v } } S _ { \mathrm { c n t } } S _ { \mathrm { d u p } } .\tag{11}
$$

Since $S _ { \mathrm { c o v } } , S _ { \mathrm { c n t } }$ , and $S _ { \mathrm { d u p } }$ are normalized to $[ 0 , 1 ] , R _ { \mathrm { g l o b a l } }$ is also bounded in [0, 1]. In $R _ { \mathrm { l o c a l } } , \ S _ { \mathrm { c o v } }$ contributes additively as target-side localization quality. In $R _ { \mathrm { g l o b a l } } ,$ , it acts as a soft coverage gate over the verifier target set $T ,$ , which may contain either one target entry or multiple entries. The global term penalizes predictions that fail to cover required targets, produce the wrong number of points, or collapse multiple predictions into a small region.

PointRL combines local matching quality and global consistency of the returned point collection as

$$
R _ { \mathrm { { s e t } } } = ( 1 - \alpha ) R _ { \mathrm { { l o c a l } } } + \alpha R _ { \mathrm { { g l o b a l } } } ,\tag{12}
$$

where $\alpha \in [ 0 , 1 ]$ controls the balance between local point target quality and global consistency of the returned point collection. We set $\alpha = 0 . 5$ in all experiments.

Figure 4 illustrates the multi-target failure modes that motivate the global terms; the same reward also covers the singleton case because a single target is represented as $m = 1$

![](images/38702c8f73b11da68f2a96230c576f5fd7df5e703cddf1063f27975981509cf4.jpg)  
Fig. 4. Local and global reward behavior for a multi-instance query. Green circles lie in target supports and red crosses lie outside. Local reward measures point-target quality, whereas global reward penalizes missed targets, incorrect counts, and collapsed predictions. The same formulation handles singleton targets.

After the base reward is computed, the verifier applies the remaining instruction-specific constraints from C as guards rather than as additional matching terms. Cardinality and uniqueness constraints encoded in C are already reflected by $m , \ S _ { \mathrm { c n t } }$ , and $S _ { \mathrm { d u p } } .$ . The indicator $\mathbb { I } _ { \mathrm { c o n d } } ~ \in ~ \{ 0 , 1 \}$ is a collection-level check computed from $P ,$ the matching $\pi ^ { \star }$ and C: relation-conditioned samples must satisfy the stored spatial predicate on the matched verifier entries, functionalregion samples must hit one admissible functional support, and samples without such constraints set $\mathbb { I } _ { \mathrm { c o n d } } = 1$

Some instruction-conditioned samples contain auxiliary cue information, such as a current point, a reference object, or an anchor used to define the target. Such cues are not themselves the desired output, and they create a shortcut: a model may stay near the cue instead of selecting a point on the intended target support. We therefore treat shortcut control as a verifierside guard.

Let a denote the stored auxiliary anchor, when present, and let $S _ { T } = \cup _ { j } S _ { j }$ denote the union of target supports, where point targets are treated as singleton supports. We compute $d ( p , S _ { T } ) = 0 { \mathrm { ~ i f ~ } } p \in S _ { T }$ and $\begin{array} { r } { d ( p , \mathcal { S } _ { T } ) = \operatorname* { i n f } _ { u \in \mathcal { S } _ { T } } \| p - u \| _ { 2 } } \end{array}$ otherwise. We mark a response as cue-copying if at least one predicted point lies within $\delta _ { \mathrm { a n c } }$ pixels of $^ { a , }$ while no predicted point lies inside any target support in ${ \cal { S } } _ { T }$ . This guard is activated only when the verifier evidence contains such an auxiliary cue, and is separate from $\mathbb { I } _ { \mathrm { c o n d } }$ . All reward distances and thresholds, including $\sigma , \delta _ { \mathrm { d u p } } .$ , and $\delta _ { \mathrm { a n c } } .$ , are computed after conversion to the verifier pixel coordinate system.

When the conditioning cue has a point anchor, we optionally retain a weak progress-shaping term:

$$
R _ { \mathrm { d i r } } = \mathrm { c l i p } \left( \frac { d ( a , S _ { T } ) - \operatorname* { m i n } _ { i } d ( p _ { i } , S _ { T } ) } { d ( a , S _ { T } ) + 1 0 ^ { - 6 } } , 0 , 1 \right) ,\tag{13}
$$

where the denominator constant avoids division by zero. This term measures only the fractional reduction in distance to the target support relative to the cue anchor. When no point anchor is available, we set $R _ { \mathrm { d i r } } = 0$ and $\lambda _ { \mathrm { d i r } } = 0$ . In the full reward, we use $\lambda _ { \mathrm { d i r } } = 0 . 1 0$ when this component is enabled and set $\lambda _ { \mathrm { d i r } } = 0$ in ablations, so the term remains subordinate to the main localization and collection-consistency rewards.

Let $\mathbb { I } _ { \mathrm { c o p y } } \in \{ 0 , 1 \}$ indicate whether the cue-copying condition holds. The base reward after the optional shortcut cap is

$$
R _ { \mathrm { b a s e } } = \left\{ \begin{array} { l l } { \mathrm { m i n } ( R _ { \mathrm { s e t } } , R _ { \mathrm { c a p } } ) , } & { \mathbb { I } _ { \mathrm { c o p y } } = 1 , } \\ { R _ { \mathrm { s e t } } , } & { \mathbb { I } _ { \mathrm { c o p y } } = 0 , } \end{array} \right.\tag{14}
$$

where $R _ { \mathrm { c a p } } \in [ 0 , 1 ]$ is the fixed shortcut cap, set to 0.25 in our experiments. Let $\mathbb { I } _ { \mathrm { r e f } } \in \{ 0 , 1 \}$ indicate whether an explicit conditioning cue is present. The guarded reward applies instruction validity first, and the final format gate removes all credit from unparsable responses:

$$
\begin{array} { r l } & { R _ { \mathrm { g u a r d } } = \mathbb { I } _ { \mathrm { c o n d } } \left[ \left( 1 - \mathbb { I } _ { \mathrm { r e f } } \right) R _ { \mathrm { s e t } } + \mathbb { I } _ { \mathrm { r e f } } \operatorname* { m i n } \left( 1 , R _ { \mathrm { b a s e } } + \lambda _ { \mathrm { d i r } } R _ { \mathrm { d i r } } \right) \right] , } \\ & { R _ { \mathrm { p o i n t } } = r _ { \mathrm { f m t } } R _ { \mathrm { g u a r d } } , } \end{array}\tag{15}
$$

This equation defines the core verifier reward. The fixed training safeguards described in Section V-A, such as counting floors, duplicate floors, and the serialization regularizer, are used only when converting this verifier reward into the scalar reward passed to GRPO.

## D. Training Details

PointRL is trained on 1,647 PointRL-Data training samples with a separate 200-sample diagnostic validation split. We train with LoRA [46], ZeRO-2 [47], and AdamW [48], and use the hyperparameter and reward settings described in Section V-A. Model responses are serialized with a categoryspecific JSON contract that is mapped back to the standard point collection and rescaled to image pixels. We use a 4:1 non-counting-to-counting task-batched sampling schedule before gradient accumulation, and serialization and safety safeguards are fixed across all reported configurations. The final pointing reward is computed after discrete text generation and cannot be backpropagated through the verifier. PointRL therefore uses it as response-level feedback for GRPO-style policy optimization [49]. For each training input $z = ( I , q )$ the old policy samples a group of G responses $\{ y _ { i } \} _ { i = 1 } ^ { G }$ . Each response is parsed into a point collection $P _ { i }$ and evaluated by the hidden verifier evidence. Let $\phi _ { \mathrm { s a f e } }$ denote the fixed count and duplicate safeguards, and let $L _ { \mathrm { a n s } }$ be the number of answer-field tokens used by the serialized response. The scalar reward used for GRPO is

$$
\begin{array} { r l } & { R _ { \mathrm { t r a i n } } ( P _ { i } , T , C ) = \operatorname* { m a x } \Bigl ( 0 , \phi _ { \mathrm { s a f e } } ( R _ { \mathrm { p o i n t } } ( P _ { i } , T , C ) , } \\ & { \qquad P _ { i } , T , C ) - \eta _ { \mathrm { s e r } } L _ { \mathrm { a n s } } \Bigr ) , } \end{array}\tag{16}
$$

where $\eta _ { \mathrm { s e r } } = 2 . 0 \times 1 0 ^ { - 4 }$ . The safeguard map $\phi _ { \mathrm { s a f e } }$ applies the fixed counting, duplicate, and hard over-count floors: responses with severe count mismatch, repeated points on the same target, or excessive unmatched predictions are capped at the corresponding fixed floor. These settings are held constant across the reported reward configurations. The policy-ratio clipping used by GRPO is applied later in the optimization objective, not as reward post-processing. We then write

$$
r _ { i } = R _ { \mathrm { t r a i n } } ( P _ { i } , T , C ) .\tag{17}
$$

The rewards are normalized within the response group to obtain the advantage of each sample:

$$
\hat { A } _ { i } = \frac { r _ { i } - \mu _ { r } } { \sigma _ { r } + \delta } , \qquad \mu _ { r } = \frac { 1 } { G } \sum _ { i = 1 } ^ { G } r _ { i } ,\tag{18}
$$

where $\sigma _ { r }$ is the standard deviation of group rewards and $\delta$ is a small constant for numerical stability.

The model is then updated with the clipped GRPO objective, which follows the clipped policy-ratio design of PPO [50]:

$$
\begin{array} { r l r } {  { J ( \theta ) = \mathbb { E } \Bigg [ \frac { 1 } { G } \sum _ { i = 1 } ^ { G } \frac { 1 } { | y _ { i } | } \sum _ { t = 1 } ^ { | y _ { i } | } \Big ( \operatorname* { m i n } \Big ( \rho _ { i , t } \hat { A } _ { i } , } } \\ & { } & { \mathrm { c l i p } ( \rho _ { i , t } , 1 - \epsilon , 1 + \epsilon ) \hat { A } _ { i } \Big ) - \beta D _ { \mathrm { K L } } ^ { i , t } \Big ) \Bigg ] , } \end{array}\tag{19}
$$

where

$$
\rho _ { i , t } = { \frac { \pi _ { \theta } ( y _ { i , t } \mid z , y _ { i , < t } ) } { \pi _ { \theta _ { \mathrm { o l d } } } ( y _ { i , t } \mid z , y _ { i , < t } ) } } .\tag{20}
$$

Here, $\pi _ { \theta }$ denotes the current VLM policy, $\pi _ { \theta _ { \mathrm { o l d } } }$ denotes the sampling policy before the update, and $\pi _ { \mathrm { r e f } }$ denotes the reference policy. The term $D _ { \mathrm { K L } } ^ { i , t }$ denotes the token-level divergence between $\pi _ { \boldsymbol { \theta } } ( \cdot \ | \ z , y _ { i , < t } )$ and $\pi _ { \mathrm { r e f } } ( \cdot \mid z , y _ { i , < t } )$ . In our experiments, we set $\beta = 0 ;$ , so the KL term is disabled; it is retained in the objective to make the general GRPO form explicit. Since $R _ { \mathrm { p o i n t } }$ is computed from the complete parsed response, the same response-level advantage ${ \hat { A } } _ { i }$ is applied to all tokens in $y _ { i }$ This optimization increases the likelihood of responses that are parseable, well localized, collection-consistent, and compliant with instruction-conditioned constraints.

## V. EXPERIMENTS

We evaluate PointRL on PointArena Point-Bench and external spatial grounding benchmarks. The experiments are organized around three questions: whether verifier-based pointlevel training improves the base VLM under a same-backbone comparison, how the tested reward configurations behave, and whether the resulting model shows same-backbone gains on external benchmarks without benchmark-specific fine-tuning.

## A. Experimental Setup

Datasets and metrics. We evaluate on PointArena Point-Bench [30], which covers affordance, counting, reasoning, spatial, and steerable pointing tasks. PointRL is trained on 1,647 constructed samples from the rule families and verifier evidence summarized in Table I, uses a separate 200-sample validation set as a held-out diagnostic split, and is evaluated on the 982-sample Point-Bench test set. Before training, we screened the training and validation splits against the Point-Bench test set and the external RoboSpatial [51], BLINK [52], and Ref-Adv [53] benchmarks by exact matching over image identifiers or hashes, source annotation identifiers, and normalized query strings when available; no matched evaluation sample is retained for reward training. Evaluation images, labels, and benchmark-specific annotations are not used for reward construction, prompt tuning, parser tuning, rewardweight selection, or checkpoint selection. We report the official hard hit-rate metric: a single-target prediction must fall inside the target mask, and a counting or multi-target point collection must have the correct count with all points inside target masks. The reported metric is binary, while PointRL training uses the denser hidden-verifier reward defined in Section IV-C.

Baselines and comparison protocol. We report two types of comparisons. First, we include representative VLMs and pointing models with comparable Point-Bench results as external references. These reported numbers provide context rather than controlled evidence, since the models may differ in backbone, training data, and prompting. Second, we conduct controlled comparisons on Qwen3.5 [41] backbones. For both Qwen3.5-2B and Qwen3.5-4B, we compare the base model with its PointRL-trained counterpart under the same evaluation protocol. These comparisons test the complete PointRL training recipe on the evaluated backbones, rather than isolating the verifier reward from fixed training choices such as data construction, serialization, batching, and LoRA optimization. For each benchmark, the compared Qwen3.5 rows use the same prompt, decoding setting, parser, and scorer.

Implementation details. Section IV-D gives the concrete optimization and serialization settings. As noted there, those settings are fixed across the reported reward comparisons. All PointRL variants use Qwen3.5 backbones, with Qwen3.5-4B as the main base model and Qwen3.5-2B as the smaller-scale run. Model responses follow a category-specific JSON contract that is normalized back to the standard point collection $P ;$ coordinates are serialized on [0, 1000] and rescaled to image pixels for verification. GRPO training uses $G = 4 ,$ clipping range ϵ = 0.2, KL coefficient $\beta = 0 , 1 , 0 0 0$ update steps, and a maximum completion length of 512 tokens. Optimization uses LoRA with $r = 3 2 , \alpha = 6 4$ , dropout 0.05, learning rate $5 . 0 \times 1 0 ^ { - 6 }$ , bf16 precision, ZeRO-2, AdamW, and maximum gradient norm 1.0. Task-batched sampling follows a 4:1 noncounting-to-counting schedule before gradient accumulation.

The reward settings are also fixed across the reported reward comparisons. We set $\sigma _ { \mathrm { c n t } } = 5 0$ pixels for counting rewards and $\sigma _ { \mathrm { m a s k } } = 4 0$ pixels for non-counting and steerable rewards. The local and set balance weights are $\lambda = 0 . 5$ and $\alpha = 0 . 5$ with under-count and over-count penalties $\gamma _ { u } ~ = ~ 3 . 0$ and $\gamma _ { o } ~ = ~ 1 . 2$ . Duplicate and hard over-count floors are set to 0.3, region-validity caps use $\tau ( S _ { j } ) \in \{ 0 . 7 0 , 0 . 5 5 , 0 . 3 5 \}$ by support reliability, and the duplicate threshold is $\delta _ { \mathrm { d u p } } ~ = ~ 5$ pixels. For cue-bound instructions, the cue-copying radius is $\delta _ { \mathrm { a n c } } = \operatorname* { m a x } ( 1 2 , \ 0 . 1 5 d ( a , \mathcal { S } _ { T } ) )$ ), with cap $R _ { \mathrm { c a p } } = 0 . 2 5$ and progress weight $\lambda _ { \mathrm { d i r } } = 0 . 1 0$ when enabled. The serialization regularizer is $2 . 0 \times 1 0 ^ { - 4 }$ per answer-field token.

Unless otherwise stated, each trained configuration is evaluated from a single checkpoint. We therefore interpret small differences cautiously and focus on same-backbone comparisons under the same evaluation protocol. Because the reported metrics are binary hit rates and per-example agreement statistics are not reported, we do not claim statistical significance for category-level, ablation-level, or external-benchmark differences.

![](images/0ef71d3c372bf551b514ea9bb663cd5ec42c250984486f389eb3b1654a60d0f3.jpg)  
Fig. 5. Qualitative comparison on PointArena Point-Bench. Each category panel shows Qwen3.5-4B on the left and Qwen3.5-4B + PointRL on the right Red and green circular markers denote the corresponding baseline and PointRL predictions; blue rings mark reference or current points in steerable examples and translucent green regions indicate target supports when shown.

TABLE II  
MAIN RESULTS ON POINTARENA POINT-BENCH.
<table><tr><td>Method</td><td>Params</td><td>Affordance</td><td>Counting</td><td>Reasoning</td><td>Spatial</td><td>Steerable</td><td>Overall Acc</td></tr><tr><td>GPT-40</td><td>N/D</td><td>42.42</td><td>31.12</td><td>23.83</td><td>25.64</td><td>24.50</td><td>29.50</td></tr><tr><td>Qwen2.5-VL-7B-Instruct</td><td>7B</td><td>76.77</td><td>52.55</td><td>53.89</td><td>61.54</td><td>36.50</td><td>56.25</td></tr><tr><td>Qwen2.5-VL-72B-Instruct</td><td>72B</td><td>76.77</td><td>57.14</td><td>54.40</td><td>60.00</td><td>46.50</td><td>58.96</td></tr><tr><td>Molmo-72B</td><td>72B</td><td>87.88</td><td>54.59</td><td>69.43</td><td>70.26</td><td>37.00</td><td>63.83</td></tr><tr><td>Gemini-Robotics-ER-1.5</td><td>N/D</td><td>69.70</td><td>68.47</td><td>60.10</td><td>69.74</td><td>67.50</td><td>67.10</td></tr><tr><td>Qwen3.5-2B</td><td>2B</td><td>34.34</td><td>10.20</td><td>32.12</td><td>33.85</td><td>16.00</td><td>25.25</td></tr><tr><td>Qwen3.5-2B + PointRL</td><td>2B</td><td>64.65</td><td>55.10</td><td>58.55</td><td>63.08</td><td>43.00</td><td>56.82</td></tr><tr><td>∆ over 2B base</td><td>1</td><td>+30.31</td><td>+44.90</td><td>+26.43</td><td>+29.23</td><td>+27.00</td><td>+31.57</td></tr><tr><td>Qwen3.5-4B</td><td>4B</td><td>65.66</td><td>47.45</td><td>58.55</td><td>69.23</td><td>40.00</td><td>56.11</td></tr><tr><td>Qwen3.5-4B + PointRL</td><td>4B</td><td>75.25</td><td>60.71</td><td>70.47</td><td>71.28</td><td>50.50</td><td>65.58</td></tr><tr><td>∆ over 4B base</td><td></td><td>+9.59</td><td>+13.26</td><td>+11.92</td><td>+2.05</td><td>+10.50</td><td>+9.47</td></tr></table>

## B. Main Results

Table II reports the main results on PointArena Point-Bench. All values are percentages, and Overall Acc is micro-average accuracy. The Params column reports disclosed model sizes, while N/D marks models whose parameter counts are not disclosed. In the controlled same-backbone comparison, PointRL improves Qwen3.5-4B from 56.11% to 65.58% overall, an observed gain of 9.47 percentage points under the singlecheckpoint protocol. With the same reward settings, Qwen3.5- 2B improves from 25.25% to 56.82%. These results indicate that PointRL training yields gains over both evaluated base backbones in this setup.

The gains are especially pronounced on tasks that require either collection structure or instruction-conditioned target selection. For Qwen3.5-4B, the largest category gains are on counting, reasoning, and steerable tasks, with improvements of 13.26, 11.92, and 10.50 points, respectively. Affordance also improves by 9.59 points, whereas spatial improves by 2.05 points from a stronger baseline score of 69.23%. This pattern suggests that the verifier reward is most useful when the model must decide not only where to point, but also how many targets to return and which constraint-bound target set is valid. The Qwen3.5-2B run shows a larger overall gain of 31.57 points, indicating that the same reward design can substantially improve a weaker base model, although the trained 4B model remains stronger overall.

The qualitative examples in Fig. 5 complement the numerical results by showing how the trained model changes the predicted point locations. In the shown cases, PointRL moves predictions toward instruction-relevant regions and improves target coverage in multi-target examples, which is consistent with the stronger gains on counting, reasoning, and steerable categories.

## C. Reward Ablation

Table III compares selected reward configurations under the same base model and evaluation protocol. The ablation is configuration-level: it reports how the tested reward bundles behave under a shared setup rather than isolating one reward term at a time. In this table, the cue-copying shortcut cap limits predictions that stay near an auxiliary cue while missing the target, and the progress variant additionally uses the weak progress term defined in Section IV-C. For compactness, Loc. denotes localization reward, Cov. denotes target coverage, Cue cap denotes the cue-copying shortcut cap, and Prog. denotes progress shaping.

In this single-run configuration comparison, the full reward obtains the highest observed overall accuracy, 65.58%, which is 9.47 points above the base model and at least 3.16 points above the tested partial reward variants. The localizationonly reward improves overall accuracy to 60.18%, showing that distance-based feedback already provides a useful signal beyond binary hit checking. Adding target coverage raises the overall score to 62.42%, a further 2.24-point gain over localization only, and mainly improves reasoning, spatial, and steerable tasks. The cue-cap configurations also improve over the base model, but their behavior is less stable: the progress variant increases affordance slightly while reducing steerable performance and overall accuracy relative to cue cap alone. These trends support the use of the full reward bundle, where local localization, target coverage, count consistency, duplicate suppression, and guard terms act together rather than as isolated heuristics.

Query: Point to all the bowls.  
![](images/12578e5ed5d60605664838cc90c45c72740c754dd60fb53410b0638b31d33ce3.jpg)

![](images/ab2174957ca7c349743757284d981decc94425ab2aa2c8884fcb70b00203437d.jpg)

![](images/e4889289052fc871ac60520bcf3a6583fc5200015115a3c76f94d905a548d6ad.jpg)

![](images/9d59be88cef195591ca8a247a3748258496069b139c6e8b9713afe0215da2256.jpg)

![](images/bc2dde4fecb8f57e9c8a8f33f23c24abb143103ace8e7c15bd90b78592501457.jpg)

![](images/3d8683d3e18d5c211f68ba9bdc375d207be508f85078f0fdfefbafbeedd0a57d.jpg)

Query: Point to all the lids of the red  
![](images/3043270b4c24e5887d05177689baa50a869a7ee9d1ccbd7b270323f0c29fafa9.jpg)

![](images/38fb539b50549f83b25a01a61805a5e0767f3ba88e45734734182624db5342aa.jpg)

![](images/ce2ed3960a97ffd4453b4adf55510c04cd4e2be59ecde467d4e1553024847cdc.jpg)

![](images/8201adfd4559dd6d575369a8250d47f01381dabd8f1ca10831393316616bae00.jpg)

![](images/f0d6e81547068915931cd6b30ef6606a74d40f7c9b754254015a249e94dc0722.jpg)

![](images/3767222033f1a79de83aa257ed3287924717875958c8202662a139ecd06cb676.jpg)  
Fig. 6. Comparison of reward configurations on multi-target pointing examples. Each row is one query; columns show the pretrained baseline, localization-onl reward, localization plus target coverage, cue-copying shortcut cap, cue-copying shortcut cap with progress shaping, and the full PointRL reward. Red point denote predicted point collections.

TABLE III  
REWARD ABLATION ON POINTARENA POINT-BENCH.
<table><tr><td>Method</td><td>Affordance</td><td>Counting</td><td>Reasoning</td><td>Spatial</td><td>Steerable</td><td>Overall Acc</td></tr><tr><td>Qwen3.5-4B</td><td>65.66</td><td>47.45</td><td>58.55</td><td>69.23</td><td>40.00</td><td>56.11</td></tr><tr><td>+ Loc. only</td><td>69.19</td><td>59.18</td><td>60.62</td><td>69.23</td><td>43.00</td><td>60.18</td></tr><tr><td>+ Loc. + Cov.</td><td>67.17</td><td>57.14</td><td>68.91</td><td>71.28</td><td>48.00</td><td>62.42</td></tr><tr><td>+ Cue cap</td><td>67.68</td><td>55.10</td><td>66.32</td><td>68.72</td><td>49.50</td><td>61.41</td></tr><tr><td>+ Cue cap + Prog.</td><td>69.70</td><td>56.63</td><td>66.84</td><td>66.15</td><td>41.00</td><td>59.98</td></tr><tr><td>+ Full reward</td><td>75.25</td><td>60.71</td><td>70.47</td><td>71.28</td><td>50.50</td><td>65.58</td></tr></table>

Overall, the table should be read as configuration-level evidence rather than as a causal study of individual reward terms. Among the tested bundles, the full reward is the strongest observed configuration. Because count consistency and duplicate suppression are not isolated in separate rows, their marginal contributions would require additional controlled variants and repeated runs. Fig. 6 illustrates this pattern on multi-target examples: partial rewards often move predictions toward relevant regions but still leave missing, collapsed, or duplicated points, whereas the full reward better covers the target set in the shown examples.

## D. External Benchmarks and Distractor-Stratified Analysis

We further evaluate PointRL on RoboSpatial [51], BLINK [52], and Ref-Adv [53] as external benchmarks. RoboSpatial and BLINK test spatial grounding directly, whereas Ref-Adv evaluates target selection through its required boxoutput protocol. PointRL training uses a separate training set from these benchmarks, and the Qwen3.5-4B + PointRL row is evaluated without benchmark-specific fine-tuning. Benchmarktrained methods are included only as in-domain references. On Ref-Adv, both Qwen3.5-4B rows use the same box prompt, invalid-output handling, decoding setting, parser, and benchmark-required output format, all fixed before evaluation; the model is therefore evaluated through the official box-output interface rather than through the point parser used during PointRL reward training. This evaluation tests transfer to target selection under the Ref-Adv protocol, while leaving the pointreward formulation used for PointRL training unchanged.

TABLE IV  
EXTERNAL EVALUATION ON ROBOSPATIAL AND BLINK. ALL VALUES ARE PERCENTAGES UNDER THE OFFICIAL BENCHMARK METRICS. THE THREE ROBOSPATIAL COLUMNS REPORT THE CONTEXT,  
CONFIGURATION, AND COMPATIBILITY SPLITS. METHODS MARKED WITH <sup>+</sup> USE BENCHMARK-SPECIFIC TRAINING DATA.
<table><tr><td>Method</td><td>Context</td><td>Config.</td><td>Compat.</td><td>BLINK</td></tr><tr><td>Molmo</td><td>0.00</td><td>48.70</td><td>24.80</td><td>67.10</td></tr><tr><td>GPT-40</td><td>6.60</td><td>74.00</td><td>55.20</td><td>76.20</td></tr><tr><td>RoboPoint</td><td>2.50</td><td>59.40</td><td>80.10</td><td>63.60</td></tr><tr><td>RoboPoint + RoboSpatial+</td><td>21.30</td><td>70.00</td><td>88.60</td><td>70.60</td></tr><tr><td> $\mathrm { L L a V A – N e X T } + \mathrm { R o b o S p a t i a l } ^ { + }$ </td><td>19.70</td><td>76.40</td><td>80.10</td><td>79.00</td></tr><tr><td>Qwen3.5-4B (baseline)</td><td>27.05</td><td>79.67</td><td>25.71</td><td>66.43</td></tr><tr><td> $\mathrm { Q w e n 3 . 5 - 4 B + P o i n t R L ( o u r s ) }$ </td><td>34.43</td><td>86.90</td><td>39.05</td><td>73.43</td></tr></table>

Table IV shows observed gains over the Qwen3.5-4B baseline across the evaluated external metrics under this protocol. On the RoboSpatial context, configuration, and compatibility splits, and on BLINK, PointRL brings gains of 7.38, 7.23, 13.34, and 7.00 percentage points, respectively. These results provide bounded evidence of cross-benchmark improvement in the evaluated settings. In-domain RoboSpatial-trained models remain stronger on compatibility, indicating that benchmarkspecific supervision still provides additional benefits. Illustrative Ref-Adv and RoboSpatial-Home examples are shown in Figure 7.

XIoU 0.36  
![](images/d6a8cfc3074080cc6b8fd6249f709b149d02a97141fa4f01e4925cbe359829c2.jpg)

IOU 0.97  
![](images/3dca39efaaa78c296a3bf3676121271306f49f06542bb63f95894d148f5dc9c8.jpg)

XIoU 0.00  
![](images/6bf0b220bfdb830681f62e9e2a350ab9472e8295dd2c23bf77879f793c95ccbb.jpg)

IoU 0.87  
![](images/005fed008b46112839be6728c0e36e38b0e1a41b0d41e8df0b47f83b80bdf168.jpg)  
PointRL(ours)

Point to: the leopard closer to the corner of the image among the two leopards playing with each other.  
![](images/0dff2cc2b875e2b26f553da2b79a77e65e5c5936b420eac68aee13fc837a89cc.jpg)

![](images/3e941f5112db6009715232628f9782a4c1b8f11eb999211f786da3cf43183a39.jpg)  
Query: In the image, there is a desk. Pinpoint several points within the vacant space situated to the left of the desk

Point to: a magpie on the ground facing the right side of the image  
![](images/780fae5f1f6baf485b7204535c6d73409dc452617e7075a1feceaf44801628bc.jpg)

![](images/b90001ddc3780f99253192f20a859847bdc6886be0c9c8b4e7302ce92dee65be.jpg)  
Query: In the image, there is a fridge. Pinpoint several points within the vacant space situated to the in front of the fridge.

Fig. 7. External-benchmark examples on Ref-Adv and RoboSpatial-Home. In each pair, the left panel shows Qwen3.5-4B and the right panel shows Qwen3.5- 4B + PointRL. In the Ref-Adv row, red boxes are predictions, blue boxes are targets, and IoU is reported above each panel. In the RoboSpatial-Home row, green regions mark valid target regions; red and green dots denote baseline and PointRL predictions, respectively.  
![](images/6100d6d9b1263597dbb168f13f1d6b7c62fa51a993d7ee4b5652ca1c4f069f05.jpg)  
Fig. 8. Ref-Adv predictions under increasing distractor counts. The three example groups correspond to 2–3, 4–6, and ≥ 7 distractors from left to right; within each group, the left panel is Qwen3.5-4B and the right panel is Qwen3.5-4B + PointRL. Red boxes denote predictions, blue boxes denote targets, and IoU is reported above each panel.

The external benchmark results also clarify what transfers and what does not. The compatibility split has the largest same-backbone improvement, but the absolute score of Qwen3.5-4B + PointRL remains below methods trained directly with RoboSpatial data. This suggests that PointRL improves general spatial grounding behavior, while benchmarkspecific supervision still matters for the compatibility distribution. The BLINK gain is useful for the opposite reason: it is obtained without BLINK-specific tuning, so it supports the claim that verifier-based point feedback can transfer beyond the constructed PointRL-Data distribution. We therefore treat these results as evidence of controlled same-backbone improvement, not as a claim that PointRL replaces benchmarkspecific training.

Table V reports Ref-Adv results under IoU thresholds and distractor-count strata. Under our same-backbone evaluation protocol, PointRL has higher observed scores than Qwen3.5- 4B on all six reported metrics. The gains are modest: 2.01,

TABLE V  
REF-ADV BOX-OUTPUT EVALUATION UNDER IOU THRESHOLDS AND DISTRACTOR-COUNT SPLITS. ALL VALUES ARE PERCENTAGES.
<table><tr><td>Method</td><td>Acc@0.5</td><td>Acc@0.75</td><td>Acc@0.9</td><td>2-3</td><td>4-6</td><td>≥7</td></tr><tr><td>Qwen2.5-VL-72B</td><td>54.00</td><td>40.10</td><td>18.00</td><td>57.00</td><td>52.70</td><td>41.10</td></tr><tr><td>Qwen3-VL-8B</td><td>52.30</td><td>38.90</td><td>19.90</td><td>55.70</td><td>50.20</td><td>38.80</td></tr><tr><td>Qwen3-VL-32B</td><td>53.40</td><td>44.70</td><td>27.10</td><td>56.30</td><td>50.20</td><td>45.70</td></tr><tr><td>Qwen3-VL-30B- A3B</td><td>52.10</td><td>43.10</td><td>27.40</td><td>54.70</td><td>48.90</td><td>45.70</td></tr><tr><td>Qwen3-VL-4B- Thinking</td><td>57.60</td><td>45.50</td><td>27.80</td><td>63.00</td><td>52.70</td><td>40.30</td></tr><tr><td>Qwen3.5-4B (baseline)</td><td>52.28</td><td>37.83</td><td>21.10</td><td>55.16</td><td>49.84</td><td>42.64</td></tr><tr><td>Qwen3.5-4B + PointRL</td><td>54.29</td><td>41.16</td><td>23.03</td><td>56.45</td><td>53.33</td><td>44.96</td></tr></table>

3.33, and 1.93 points under the three IoU thresholds, and 1.29, 3.49, and 2.32 points under the 2–3, 4–6, and ≥ 7 distractor settings. Since Ref-Adv is a box-output benchmark, we interpret these results as bounded evidence of target-selection transfer under the Ref-Adv protocol, not as a direct pointoutput result or a standalone claim about distractor robustness.

Qualitative examples in Fig. 7 show Ref-Adv target selection and RoboSpatial-Home free-space localization cases. We further visualize Ref-Adv examples under increasing distractor counts in Fig. 8, covering the 2–3, 4–6, and $\geq 7$ distractor settings.

The Ref-Adv improvements are smaller than the Point-Bench gains, which is expected because the evaluation uses the benchmark-required box-output interface rather than the point parser used during reward training. Even so, the gains are consistent across strictness levels. The improvement at Acc@0.75 is larger than at Acc@0.5, and the 4–6 distractor stratum has the largest distractor-based gain. This pattern is compatible with improved target selection rather than a simple formatting advantage: PointRL does not change the Ref-Adv output protocol, but it can make the model choose the intended object more reliably when the scene contains competing distractors.

Across the external evaluations, the same-backbone comparison yields positive gains under both point- and box-output protocols, but the magnitude varies by benchmark and split. On RoboSpatial, the largest observed gain is on compatibility (+13.34), with gains of +7.38 and +7.23 on context and configuration, respectively. The +7.00 improvement on BLINK provides additional evidence outside the constructed PointRL-Data distribution. On Ref-Adv, the gains across the three distractor strata range from 1.29 to 3.49 points, with the largest observed improvement in the 4–6 range. Because Ref-Adv is evaluated through its required box-output interface, these results are best read as transfer to target selection under a different output protocol. The lower compatibility score relative to RoboSpatial-trained methods and the singlecheckpoint setting limit the claim to same-backbone gains.

## VI. CONCLUSION

This work presents PointRL, a verifiable reinforcement learning framework for learning point-level visual grounding from heterogeneous grounding annotations. PointRL converts boxes, masks, and instance labels into pointing instructions and keeps the original annotations as hidden verifier evidence for reward computation. This design preserves non-unique valid coordinates for region-based supports and allows the reward to credit different points that satisfy the annotated target condition.

PointRL further defines a structured pointing reward over unordered point collections. The reward combines parseability, soft localization, target coverage, count consistency, duplicate suppression, and constraint checks. In this way, it supports both single-target and multi-target grounding under a unified formulation. Experiments on PointArena yield an observed 9.47-point gain over the Qwen3.5-4B baseline in this protocol, and the configuration-level ablations suggest that local localization feedback and collection-level structural constraints are useful in combination in the tested settings. Additional evaluations on RoboSpatial, BLINK, and Ref-Adv show bounded same-backbone gains under the evaluated protocols. These results suggest that, in the evaluated settings, heterogeneous grounding annotations can serve as useful hidden verifier evidence for improving the pointing ability of VLMs.

## REFERENCES

[1] A. Kamath, M. Singh, Y. LeCun, G. Synnaeve, I. Misra, and N. Carion, “Mdetr-modulated detection for end-toend multi-modal understanding,” in Proceedings of the IEEE/CVF international conference on computer vision, 2021, pp. 1780–1790.

[2] S. Liu, Z. Zeng, T. Ren, F. Li, H. Zhang, J. Yang, Q. Jiang, C. Li, J. Yang, H. Su et al., “Grounding dino: Marrying dino with grounded pre-training for open-set object detection,” in European conference on computer vision. Springer, 2024, pp. 38–55.

[3] W. Hong, W. Wang, Q. Lv, J. Xu, W. Yu, J. Ji, Y. Wang, Z. Wang, Y. Dong, M. Ding et al., “Cogagent: A visual language model for gui agents,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2024, pp. 14 281–14 290.

[4] H. Liu, C. Li, Q. Wu, and Y. J. Lee, “Visual instruction tuning,” Advances in neural information processing systems, vol. 36, pp. 34 892–34 916, 2023.

[5] M. Lu, R. Li, F. Feng, Z. Ma, and X. Wang, “Lgr-net: Language guided reasoning network for referring expression comprehension,” IEEE Transactions on Circuits and Systems for Video Technology, vol. 34, no. 8, pp. 7771– 7784, 2024.

[6] C. Wang, W. Feng, S. Lyu, G. Cheng, X. Li, B. Liu, and Q. Zhao, “A masked reference token supervision-based iterative visual-language framework for robust visual grounding,” IEEE Transactions on Circuits and Systems for Video Technology, vol. 35, no. 1, pp. 75–90, 2024.

[7] H. Qiu, L. Wang, T. Zhao, F. Meng, Q. Wu, and H. Li, “Mcce-rec: Mllm-driven cross-modal contrastive entropy model for zero-shot referring expression comprehension,” IEEE Transactions on Circuits and Systems for Video Technology, vol. 35, no. 1, pp. 754–768, 2024.

[8] M. Xu, T. Xiao, Y. Liu, H. Tang, Y. Hu, and L. Nie, “Cmirnet: Cross-modal interactive reasoning network for referring image segmentation,” IEEE Transactions on Circuits and Systemsfor Video Technology, vol. 35, no. 4, pp. 3234–3249, 2024.

[9] S.-A. Liu, H. Xie, J. Ge, and Y. Zhang, “Refersam: Unleashing segment anything model for referring image segmentation,” IEEE Transactions on Circuits and Systems for Video Technology, vol. 35, no. 5, pp. 4910–4922, 2025.

[10] L. Yu, Z. Lin, X. Shen, J. Yang, X. Lu, M. Bansal, and T. L. Berg, “Mattnet: Modular attention network for referring expression comprehension,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2018, pp. 1307–1315.

[11] D. Liu, H. Zhang, F. Wu, and Z.-J. Zha, “Learning to assemble neural module tree networks for visual grounding,” in Proceedings of the IEEE/CVF international conference on computer vision, 2019, pp. 4673– 4682.

[12] X. Lai, Z. Tian, Y. Chen, Y. Li, Y. Yuan, S. Liu, and J. Jia, “Lisa: Reasoning segmentation via large language

model,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2024, pp. 9579– 9589.

[13] H. Rasheed, M. Maaz, S. Shaji, A. Shaker, S. Khan, H. Cholakkal, R. M. Anwer, E. Xing, M.-H. Yang, and F. S. Khan, “Glamm: Pixel grounding large multimodal model,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 13 009–13 018.

[14] W. Tang, X. Liu, Y. Sun, and Z. Li, “Ssp-sam: Sam with semantic-spatial prompt for referring expression segmentation,” IEEE Transactions on Circuits and Systems for Video Technology, 2025.

[15] S. Wang, S. Liu, Y. Kuang, X. Wei, Y. Liu, Z. Li, Y. Man, G. Chen, A. Tao, G. Liu et al., “Locateanything: Fast and high-quality vision-language grounding with parallel box decoding,” arXiv preprint arXiv:2605.27365, 2026.

[16] Y. Su, H. Zhang, S. Li, N. Liu, J. Liao, J. Pan, Y. Liu, X. Xing, C. Sun, C. Li et al., “Patch-as-decodable-token: Towards unified multi-modal vision tasks in mllms,” arXiv preprint arXiv:2510.01954, 2025.

[17] H. Guo, W. Fan, B. Wei, J. Zhu, J. Tian, C. Yi, and F. Jiang, “Ad-dino: Attention-dynamic dino for distanceaware embodied reference understanding,” IEEE Transactions on Circuits and Systems for Video Technology, 2025.

[18] Q. Song, F. Yang, L. Yang, C. Liu, M. Hu, and L. Xia, “Learning point-guided localization for detection in remote sensing images,” IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, vol. 14, pp. 1084–1094, 2020.

[19] L. Yang, Q. Song, Z. Wang, and M. Jiang, “Parsing r-cnn for instance-level human analysis,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2019, pp. 364–373.

[20] S. Li, L. Yang, P. Cao, L. Li, and H. Ma, “Frequencybased matcher for long-tailed semantic segmentation,” IEEE Transactions on Multimedia, vol. 26, pp. 10 395– 10 405, 2024.

[21] Y. Guo, L. Yang, P. Cao, S. Li, Y. Zhou, and Q. Song, “Quality transformer for human parsing,” Pattern Recognition, vol. 169, p. 111736, 2026.

[22] L. Yang, L. Li, J. Wei, P. Cao, Y. Wang, and W. Wang, “Large-scale omnidirectional person positioning,” IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025.

[23] L. Yang, L. Li, X. Xin, Y. Sun, Q. Song, and W. Wang, “Large-scale person detection and localization using overhead fisheye cameras,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 19 961–19 971.

[24] L. Yang, W. Jia, S. Li, and Q. Song, “Deep learning technique for human parsing: A survey and outlook,” International Journal ofComputer Vision, vol. 132, no. 8, pp. 3270–3301, 2024.

[25] L. Yang, H. Jiang, Q. Song, and J. Guo, “A survey on long-tailed visual recognition,” International Journal of Computer Vision, vol. 130, no. 7, pp. 1837–1872, 2022.

[26] W. Zhao, W. Li, Y. Li, L. Yang, Z. Liang, E. Hu, W. Zhang, and H. Yang, “Constructing balanced training samples: A new perspective on long-tailed classification,” IEEE Transactions on Multimedia, 2025.

[27] M. Dai, K. Chen, W. Cheng, J. Zhuang, Z. Feng, P. Zhu, and W. Yang, “Gc 3 vg: Generalized multi-task visual grounding with coarse-to-fine consistency constraints,” IEEE Transactions on Circuits and Systems for Video Technology, 2025.

[28] M. Deitke, C. Clark, S. Lee, R. Tripathi, Y. Yang, J. Park, M. Salehi, N. Muennighoff, K. Lo, L. Soldaini et al., “Molmo and pixmo: Open weights and open data for state-of-the-art vision-language models, 2024,” URL https://arxiv.org/abs/2409.17146, 2024.

[29] C. Clark, Y. Yang, J. S. Park, Z. Ma, J. Zhang, R. Tripathi, M. Salehi, S. Lee, T. Anderson, W. Han et al., “Molmopoint: Better pointing for vlms with grounding tokens,” arXiv preprint arXiv:2603.28069, 2026.

[30] L. Cheng, J. Duan, Y. R. Wang, H. Fang, B. Li, Y. Huang, E. Wang, A. Eftekhar, J. Lee, W. Yuan, R. Hendrix, N. A. Smith, F. Xia, D. Fox, and R. Krishna, “Pointarena: Probing multimodal grounding through language-guided pointing,” 2025.

[31] W. Yuan, J. Duan, V. Blukis, W. Pumacay, R. Krishna, A. Murali, A. Mousavian, and D. Fox, “Robopoint: A vision-language model for spatial affordance prediction for robotics,” arXiv preprint arXiv:2406.10721, 2024.

[32] K. Li, Z. Meng, H. Lin, Z. Luo, Y. Tian, J. Ma, Z. Huang, and T.-S. Chua, “Screenspot-pro: Gui grounding for professional high-resolution computer use,” in Proceedings of the 33rd ACM International Conference on Multimedia, 2025, pp. 8778–8786.

[33] D. Guo, D. Yang, H. Zhang, J. Song, P. Wang, Q. Zhu, R. Xu, R. Zhang, S. Ma, X. Bi et al., “Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning,” arXiv preprint arXiv:2501.12948, 2025.

[34] A. Y. Ng, D. Harada, and S. J. Russell, “Policy invariance under reward transformations: Theory and application to reward shaping,” in Proceedings of the Sixteenth International Conference on Machine Learning (ICML 1999). San Francisco, CA, USA: Morgan Kaufmann, 1999, pp. 278–287. [Online]. Available: http: //robotics.stanford.edu/<sup>∼</sup>ang/papers/shaping-icml99.ps

[35] Z. Liu, Z. Sun, Y. Zang, X. Dong, Y. Cao, H. Duan, D. Lin, and J. Wang, “Visual-rft: Visual reinforcement fine-tuning,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025, pp. 2034– 2044.

[36] H. Shen, P. Liu, J. Li, C. Fang, Y. Ma, J. Liao, Q. Shen, Z. Zhang, K. Zhao, Q. Zhang et al., “Vlm-r1: A stable and generalizable r1-style large vision-language model,” arXiv preprint arXiv:2504.07615, 2025.

[37] S. Hegde, J. S. Chacko, D. Banerjee, and U. Mahesh, “Genseg-r1: Rl-driven vision-language grounding for fine-grained referring segmentation,” arXiv preprint arXiv:2602.09701, 2026.

[38] R. Krishna, Y. Zhu, O. Groth, J. Johnson, K. Hata, J. Kravitz, S. Chen, Y. Kalantidis, L.-J. Li, D. A. Shamma

et al., “Visual genome: Connecting language and vision using crowdsourced dense image annotations,” International journal of computer vision, vol. 123, no. 1, pp. 32–73, 2017.

[39] A. J. Ratner, C. M. De Sa, S. Wu, D. Selsam, and C. Re, “Data programming: Creating large training sets,´ quickly,” Advances in neural information processing systems, vol. 29, 2016.

[40] A. Ratner, S. H. Bach, H. Ehrenberg, J. Fries, S. Wu, and C. Re, “Snorkel: Rapid training data creation with´ weak supervision,” in Proceedings of the VLDB endowment. International conference on very large data bases, vol. 11, 2017, p. 269.

[41] Qwen Team, “Qwen3.5: Towards native multimodal agents,” February 2026. [Online]. Available: https: //qwen.ai/blog?id=qwen3.5

[42] H. Luo, W. Zhai, J. Zhang, Y. Cao, and D. Tao, “Learning affordance grounding from exocentric images,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 2252–2261.

[43] T.-Y. Lin, M. Maire, S. Belongie, J. Hays, P. Perona, D. Ramanan, P. Dollar, and C. L. Zitnick, “Microsoft´ coco: Common objects in context,” in European conference on computer vision. Springer, 2014, pp. 740–755.

[44] H. W. Kuhn, “The hungarian method for the assignment problem,” Naval Research Logistics (NRL), vol. 52, 1955. [Online]. Available: https://api.semanticscholar. org/CorpusID:9426884

[45] N. Carion, F. Massa, G. Synnaeve, N. Usunier, A. Kirillov, and S. Zagoruyko, “End-to-end object detection with transformers,” in European conference on computer vision. Springer, 2020, pp. 213–229.

[46] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, W. Chen et al., “Lora: Low-rank adaptation of large language models.” Iclr, vol. 1, no. 2, p. 3, 2022.

[47] S. Rajbhandari, J. Rasley, O. Ruwase, and Y. He, “Zero: Memory optimizations toward training trillion parameter models,” in SC20: international conference for high performance computing, networking, storage and analysis. IEEE, 2020, pp. 1–16.

[48] I. Loshchilov and F. Hutter, “Decoupled weight decay regularization,” arXiv preprint arXiv:1711.05101, 2017.

[49] Z. Shao, P. Wang, Q. Zhu, R. Xu, J. Song, X. Bi, H. Zhang, M. Zhang, Y. Li, Y. Wu et al., “Deepseekmath: Pushing the limits of mathematical reasoning in open language models,” arXiv preprint arXiv:2402.03300, 2024.

[50] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, “Proximal policy optimization algorithms,” arXiv preprint arXiv:1707.06347, 2017.

[51] C. H. Song, V. Blukis, J. Tremblay, S. Tyree, Y. Su, and S. Birchfield, “Robospatial: Teaching spatial understanding to 2d and 3d vision-language models for robotics,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 15 768–15 780.

[52] X. Fu, Y. Hu, B. Li, Y. Feng, H. Wang, X. Lin, D. Roth, N. A. Smith, W.-C. Ma, and R. Krishna, “Blink: Multimodal large language models can see but not perceive,”

in European Conference on Computer Vision. Springer, 2024, pp. 148–166.

[53] Q. Dong, K. Yang, L. Ju, H. Zhao, Y. Zhang, Y. Wang, H. Zeng, J. Lu, and Y. Fu, “Ref-adv: Exploring mllm visual reasoning in referring expression tasks,” arXiv preprint arXiv:2602.23898, 2026.

![](images/084cb9b5efa75016ca1246ac866f572c2fba25e7f0911be24e8aa02480c7aea6.jpg)

Jingyang Su received the B.E. degree in automation from Hangzhou Dianzi University, Hangzhou, China, in 2025. He is currently pursuing the M.S. degree with the School of Intelligent Engineering and Automation, Beijing University of Posts and Telecommunications (BUPT), Beijing, China. His research interests include multimodal large language models, visual grounding, reinforcement learning, and uncertainty estimation.

![](images/e5eb7fc12b6de3c57fe8312ece1e1840bbc5014b8f69cbd3af059b5a395ee706.jpg)

Pu Cao received his bachelor’s degree from the University of Science and Technology Beijing (USTB), Beijing, China, in 2022, and is currently a Ph.D. candidate at the School of Intelligent Engineering and Automation, Beijing University of Posts and Telecommunications (BUPT), Beijing, China, since 2022. His research interests include multimodal understanding and generation, especially multimodal large language models and diffusion models.

![](images/9eba71b32fdaf6083aa8238a7a44d035b4bf29bc7e15e20b173006bc553b40b2.jpg)

Xiuze Jin entered Beijing University of Posts and Telecommunications (BUPT) in 2025 and is currently majoring in the Computer Science Meta-Class of the Future College. During his freshman year, he developed a strong interest in large AI models and robotics and is eager to pursue research in these fields.

![](images/d9432990a0b5873f72c4ecaa48ca2ec058bd8cb82ed87ed37082f9b97738ecc6.jpg)

Longyue Zhang is currently pursuing his undergraduate studies in automation at Beijing University of Posts and Telecommunications (BUPT), Beijing, China. His research interests lie at the intersection of mechanics and control systems, as well as human– large language model interaction. He is committed to rigorous and collaborative interdisciplinary research.

![](images/af43877636b78f5f6f00ca5d55f8cf4aa830e15261a38ffe495bad365c48e45d.jpg)

Qing Song received the Ph.D. degree from Tianjin University, Tianjin, China, in 2006. She is currently a researcher with Beijing University of Posts and Telecommunications (BUPT), where she is engaged in computer vision technology. She founded the Pattern Recognition and Intelligent Vision Laboratory (PRIV).

![](images/94a646bb409900b16cda2551a33df88115936e89b0902c96322f87d78011e091.jpg)

Lu Yang is currently an associate professor with Beijing University of Posts and Telecommunications (BUPT), China. He received the Ph.D. degree from BUPT in 2021. He has conducted research with the Pattern Recognition and Intelligent Vision Laboratory (PRIV) since 2012. His research interests include human-centric AI and generative AI.