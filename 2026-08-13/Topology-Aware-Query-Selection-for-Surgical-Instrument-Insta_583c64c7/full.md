# Topology-Aware Query Selection for Surgical Instrument Instance Segmentation

Ze Zhang<sup>a</sup>, Yang Zhang<sup>b</sup>

<sup>a</sup>Department ofBiomedical Engineering, ShanghaiTech University, Shanghai, China <sup>b</sup>Wuhan United Imaging Surgical Co., Ltd., Wuhan, China

## Abstract

Accurate foreground masks can still form an incorrect surgical-instrument instance set: duplicate, fragmented, merged, missed, or empty-frame predictions may preserve favorable pixel overlap while violating object identity and count. Final query selection is therefore a relational, variable-cardinality problem rather than a collection of independent candidate decisions. We evaluate topology-aware query selection, which represents the nonempty candidates of a fixed Mask2Former as a complete graph, learns relational candidate and pair representations, predicts set cardinality, and solves an exact structured subset problem. The formal comparison is the complete relational path versus a node-feature-matched path; it evaluates the combined efect of pairwise geometry, message passing, and the additional relational-path capacity, not an isolated component. On the sealed 22-case source test, al three discovery seeds supported instance-set performance improvement with segmentation fidelity and predefined technical-safety preservation: instance F1 increased by 0.0504–0.0612 and positive-frame set-failure rate decreased by 0.0848–0.1060. Direct ROBUST-MIPS transfer reproduced the complete result in all three seeds. Endoscapes supported only one of three seeds and therefore did not establish stable direct transfer. Taken together, the results support a bounded conclusion: the evaluated complete path improved coherent instance-set construction from fixed Mask2Former candidates in specified native-instance contracts, while stable cross-domain transfer and component-specific efects remain unestablished.

Keywords: surgical instrument segmentation, instance segmentation, query selection, relational reasoning, structured prediction, cardinality prediction, case-level inference

## 1. Introduction

Surgical instrument instance segmentation must identify not only where tool foreground lies but also how that foreground is partitioned into individual instruments. This distinction matters<sup>[</sup> whenever downstream analysis follows separate tools, counts them, or assigns tool-specific state. A prediction can preserve nearly the same foreground union while splitting one tool into several masks, merging two tools, duplicating a tool, omitting an instrument, or activating in an empty frame. Surgical datasets and multi-instance methods make object identity an explicit target, but a favorable overlap score alone does not guarantee. that the predicted masks form the correct set (Kurmann et al., 2021; Roß et al., 2021; Ángeles Cerón et al., 2022; Alabi et al., 2025).

The final decision over query masks is therefore joint and variable in size. Two candidates may overlap the same instrument and compete for one output slot; one candidate may contain another; adjacent candidates may be fragments of one tool; and the number of valid instances changes across frames. A candidate that appears plausible in isolation can become redundant or inconsistent after the other candidates are considered. These dependencies turn final query selection into structural prediction over a set: the predictor must determine both which candidates belong together in the output and how many candidates the output should contain.

Candidate-wise scoring does not explicitly represent this full relational state. Detector-based models associate masks with region proposals, while DETR-style systems formulate object prediction with learned queries and bipartite matching (He et al., 2017; Carion et al., 2020). MaskFormer, Mask2Former, and QueryInst further develop query-based mask prediction and shared instance representations (Cheng et al., 2021, 2022; Fang et al., 2021). These advances make strong candidate masks possible. A final unary score still cannot directly encode substantial containment, competition for one reference instance, or a jointly incorrect cardinality. The surgical mask-query problem therefore requires mask topology and variable set size to be represented together.

We address this problem with topology-aware query selection over fixed Mask2Former candidates. The method retains every nonempty candidate and constructs a complete graph whose edges describe overlap, directional containment, relative scale, score separation, and boundary adjacency. Relational message passing updates each candidate in the context of all others, a cardinality head estimates the required output size, and exact structured optimization selects a subset while penalizing learned pair competition. Each module answers a part of the technical problem: pairwise geometry describes candidate conflict, message passing conditions unary evidence on the set, cardinality prediction replaces an implicit threshold count, and exact selection enforces a globally consistent subset.

The formal test compares this complete relational path with a node-feature-matched path that shares the candidates, node descriptors, supervision, folds, training seeds, checkpoint rule, cardinality machinery, and exact selector. The contrast therefore evaluates the complete relational path as a package, including edge geometry, message passing, and its additional capacity. The principal evidence is a one-shot sealed source test, followed by separate direct transfers to ROBUST-MIPS and Endoscapes. The source test and ROBUST-MIPS support the complete method claim across all three discovery seeds, whereas Endoscapes does not establish stable transfer.

The paper makes three contributions. First, it formulates final surgical mask-query selection as relational, variable-cardinality structural prediction. Second, it provides a frozen, case-level matched comparison on a sealed source test and two separately interpreted external domains. Third, it qualifies the method claim under a label-contract-aware, non-compensatory rule: instanceset improvement counts only when segmentation fidelity and predefined technical safety are also preserved within the same evidence setting.

## 2. Related work

## 2.1. Surgical instrument instance segmentation

Surgical instrument segmentation has progressed from foreground and part segmentation toward explicit separation of multiple instrument instances. Multi-instance methods, attention and feature-fusion architectures, and the ROBUST-MIS challenge have highlighted the dificulty of distinguishing nearby or overlapping tools in endoscopic video (Kurmann et al., 2021; Roß et al., 2021; Ángeles Cerón et al., 2022). CholecInstanceSeg provides oficial instrument instance identifiers for laparoscopic cholecystectomy, while Endoscapes and ROBUST-MIPS provide additional native-instance populations with distinct acquisition and annotation conditions (Alabi et al., 2025; Mascagni et al., 2025; Han et al., 2026). Our focus is narrower than a new backbone or temporal architecture: given fixed mask candidates, we study how the final system should construct a coherent instance set.

## 2.2. Query and set prediction

Mask R-CNN constructs instances from region proposals, whereas DETR treats detection as direct set prediction with object queries and bipartite matching (He et al., 2017; Carion et al., 2020). MaskFormer and Mask2Former extend mask classification and masked attention across segmentation tasks, and QueryInst shares query representations across instance attributes (Cheng et al., 2021, 2022; Fang et al., 2021). These models already contain interactions within their decoders. Our question begins after the fixed Mask2Former has emitted scored masks: whether the evaluated complete relational path improves the final subset over a model with the same candidate-wise information but no edge geometry or graph messages.

## 2.3. Relational reasoning and duplicate control

Learned suppression can replace hand-crafted NMS by rescoring boxes from their scores and geometry (Hosang et al., 2017). Relation networks use appearance and geometric interactions among detected objects, proposal-ranking methods learn a suppression order, and contextual rescoring conditions each detection on the surrounding detection set (Hu et al., 2018; Tan et al., 2019; Pato et al., 2020). These methods establish that candidate interactions can improve box-level ranking or duplicate control.

Our study starts instead from fixed surgical mask queries and asks how to construct a native-instance set with variable cardinality and an exact subset constraint. Relational reasoning, cardinality modeling, and exact optimization are established techniques rather than separate contributions. We integrate them for surgical mask-query topology and evaluate the integrated path under frozen case-level contracts. This design does not claim that its individual components are novel or causally isolated.

## 2.4. Evaluation boundary for instance sets

Instance-segmentation and medical-image-analysis studies have shown that metric choice, aggregation, empty references, and challenge ranking procedures can change the interpretation of model performance (Jena et al., 2023; Maier-Hein et al., 2024; Reinke et al., 2024; Ostmeier et al., 2023; Maier-Hein et al., 2018). Reporting guidance likewise emphasizes intended use, data partitions, reference standards, and statistical units (Tejani et al., 2024; Roß et al., 2023). These considerations are evidence discipline for the present method claim, not a second method contribution. We keep native-instance performance, foregroundunion fidelity, empty-reference behavior, and case-level stability separate because none is an interchangeable proxy for the others.

## 3. Method

Figure 1 traces the shared frozen candidates through the two formal paths. The complete path adds pair geometry and graph messages before the matched prediction heads; both paths predict cardinality and use the same exact subset objective.

## 3.1. Selecting a coherent set from fixed candidates

The method begins from fixed candidates so that it changes set construction rather than pixel prediction. For an endoscopic frame x, a fixed Mask2Former emits $N = 1 0 0$ scored mask queries $Q = \{ q _ { i } \} _ { i = 1 } ^ { N }$ before any external score threshold or mask non-maximum suppression (Cheng et al., 2022). Query $q _ { i }$ contains a tool-class score $s _ { i }$ and binary mask m . The score retains the no-object class in the softmax denominator. Mask logits are resized bilinearly to $2 5 6 \times 2 5 6$ , converted to probabilities, and thresholded at 0.5. All queries remain auditable, while only nonempty masks form the eligible set

$$
\mathcal { V } = \{ i : | m _ { i } | > 0 \} .
$$

Empty masks create neither graph edges nor selected outputs. The prediction target is a subset $s \subseteq \mathbf { \mathcal { V } }$ whose masks represent the native instrument instances. Membership is relational: two masks may compete for one reference instance, a candidate may contain another, and adjacent fragments may be inconsistent with one coherent tool. The model therefore predicts both set cardinality and candidate membership without updating the base segmenter.

![](images/794a2dfd549620d6e146b7b6201f9254d0725fe8e8d16166ecf7cc092d3d3d41.jpg)  
Figure 1: Complete-path and matched-comparator data flow. The cardinality head reads the pooled candidate set, and the exact selector receives validity, paircompetition, and cardinality outputs; the auxiliary topology head is used only for training. The formal comparison estimates the complete-path package efect, not the causal efect of any component.

## 3.2. Candidate representation

Node features describe unary plausibility, candidate geometry, and frame context available before relational inference. Each candidate has nine frozen features: tool score, normalized emitted rank, mask area, area ratio, perimeter, connectedcomponent count, emitted candidate count, empty-query count, and foreground-union area. These features let the matched paths see the same score, scale, shape, and frame-level candidatedensity information. The method uses neither image pixels nor temporal, query-embedding, or region-embedding features.

Native-instance annotations define supervision. Hungarian matching at mask IoU ≥ 0 5 supplies node-validity labels. A candidate pair is labeled as competing when both masks overlap the same native reference instance with overlap coeficient ≥ 0 5. Three auxiliary node targets identify candidates associated with duplication, merging, and splitting. Cardinality is represented by classes 0 through 9 and one class for 10 or more instruments. Connected-component proxy labels neither train nor evaluate the method.

## 3.3. Complete query graph

Candidate conflict can involve any pair, so the method does not prune relations before the competition problem is solved. Each frame yields an undirected complete graph $G = ( \mathcal { V } , \mathcal { E } )$ with $n ( n - 1 ) / 2$ edges for $n = | \mathcal { V } |$ . No fixed-neighbor rule or result-dependent edge pruning is used. Every edge has eight features: pairwise intersection over union, two directional containment ratios, overlap coeficient, absolute log area ratio, absolute score margin, minimum boundary distance, and boundarycontact pixels. Together these features distinguish duplicate-like overlap, asymmetric containment, relative scale, confidence separation, and adjacent fragmentation. Means and population standard deviations are estimated only from fit-role cases in each outer fold; a zero standard deviation is replaced by one.

## 3.4. Relational inference

Relational inference revises each unary candidate state using all other eligible candidates and their pair geometry. Two linear layers with ReLU, LayerNorm, and dropout 0.10 map each node to a 64-dimensional state, while two linear layers of widths 32 and 64 encode edge features. Two message-passing layers update node i as

$$
\bar { r } _ { i } ^ { ( \ell ) } = \frac { 1 } { \operatorname* { m a x } ( 1 , | N ( i ) | ) } \sum _ { j \in N ( i ) } M _ { \ell } \big ( [ h _ { j } ^ { ( \ell ) } , h _ { i } ^ { ( \ell ) } , e _ { i j } ] \big ) .\tag{1}
$$

$$
h _ { i } ^ { ( \ell + 1 ) } = h _ { i } ^ { ( \ell ) } + U _ { \ell } \Big ( [ h _ { i } ^ { ( \ell ) } , \bar { r } _ { i } ^ { ( \ell ) } ] \Big ) .\tag{2}
$$

Equations 1 and 2 aggregate degree-normalized pair messages and apply a residual node update. Linear heads then predict node validity $^ { a _ { i } , }$ the three topology targets, and pair competition $c _ { i j }$ . Mean and elementwise-maximum pooling summarize the updated candidates into a 128-dimensional set representation for cardinality prediction; an empty eligible set uses a zero pooled representation. Training minimizes the unweighted sum of nodevalidity, pair-competition, cardinality, and topology losses, using binary cross-entropy with logits except for cardinality crossentropy.

## 3.5. Cardinality prediction

Independent score thresholds leave output count implicit, although count is part of the instance-set target. The cardinality head therefore predicts

$$
\hat { k } = \operatorname* { m i n } \left( \arg \operatorname* { m a x } _ { r \in \{ 0 , \ldots , 1 0 \} } p ( r | G ) , | \mathcal { V } | \right) .\tag{3}
$$

In Equation 3, the final class requests 10 candidates before the eligible-set cap. Cardinality is predicted from the relationally pooled candidate representation, not from image pixels, so the output size is conditioned on the candidate configuration rather than inferred by independently thresholding candidates.

## 3.6. Exact structured subset selection

The final selector combines unary validity, learned pair competition, and the predicted cardinality in one constrained decision. With $p _ { i j } = \sigma ( c _ { i j } )$ , it returns

$$
S ^ { * } = \arg \operatorname* { m a x } _ { S \subseteq \mathcal { V } , | S | = \widehat { k } } \left[ \sum _ { i \in S } a _ { i } - \sum _ { i < j ; i , j \in S } p _ { i j } \right] .\tag{4}
$$

Equation 4 fixes the pair penalty at 1.0. Cardinalities one and two use exhaustive enumeration. Larger cardinalities use mixed-integer linear programming with binary node variables $y _ { i } ,$ pair variables $z _ { i j } = y _ { i } y _ { j }$ , exact cardinality, standard linearization constraints, and zero relative optimality gap. Candidate identifiers define canonical variable order, and exhaustive exact ties choose the lexicographically earlier subset. An infeasible cardinality, incomplete pair graph, solver failure, or cardinality mismatch creates a flagged unresolved frame; no heuristic subset is substituted.

For n eligible queries, graph construction and relational processing require $O ( n ^ { 2 } )$ pair storage and computation at fixed feature widths. The MILP contains n node variables and $n ( n - 1 ) / 2$ pair variables and has combinatorial worst-case complexity. With $n \leq 1 0 0$ in this implementation, the pair graph contains at most $4 { , } 9 5 0$ unique edges. Exactness is a design property of both formal paths, not an independently estimated efect in the primary comparison.

## 3.7. Node-feature-matched path

The formal comparator keeps the candidate-wise information and selection machinery fixed. It is a node-only multilayer perceptron that shares candidates, node features, node encoder, prediction heads, pooling, folds, training seeds, optimizer, supervision, checkpoint rule, cardinality prediction, and exact structured selector with the relational path. Its pair head receives concatenated endpoint states but neither edge features nor graph messages. Frozen score-threshold output and mask NMS at 0.50 are secondary controls, not the formal method contrast.

The graph and node-feature-matched paths contain 59,120 and 6,864 trainable parameters, respectively. Consequently, the graph-minus-node-only result evaluates the complete relational path, including pairwise geometry, relational message passing, and additional capacity. It does not establish the separate contribution of edge features, message passing, cardinality prediction, or exact selection.

## 3.8. Training and fixed evaluation relation

Training is case isolated so that candidate relations are learned without using the sealed or external populations. Three fixed Mask2Former discovery models (seeds 17, 23, and 41) provide candidates for 4,246 frames from 18 source-development cases. Six outer folds assign disjoint fit, calibration, and held-out roles: fit cases estimate standardizers and model parameters, calibration cases select checkpoints without updating parameters, and held-out cases are evaluated once after checkpoint selection.

For each discovery seed and outer fold, the graph and nodefeature-matched paths are trained independently with graphtraining seeds 101, 103, and 107, producing 54 paired fit groups. AdamW uses learning rate $1 0 ^ { - 3 }$ , weight decay $1 0 ^ { - 4 }$ , one frame per batch, gradient clipping at 1.0, and 30 epochs. Nativeinstance F1 is averaged within each calibration case and then equally across cases. The earliest epoch attaining the maximum calibration objective is retained. No architecture, feature, loss weight, optimizer, threshold, or checkpoint search follows held-out inspection.

The sealed source test and the external populations neither train nor calibrate the method. After source development closed, the 54 paired models were applied without modification to the 22-case source test and separately to Endoscapes and ROBUST-MIPS.

## 4. Evaluation protocol

## 4.1. Datasets and evidence roles

The evaluation settings are separated by population, annotation capability, and training/evaluation relation (Table 1). Source development freezes and progresses the method but is not independent confirmation. The sealed source test is the primary one-shot confirmation. Endoscapes and ROBUST-MIPS are distinct direct external transfers and are never combined.

## 4.2. Matched comparison and case-level inference

The primary estimands are graph minus node-feature-matched native-instance F1 and node-feature-matched minus graph positive-frame set-failure rate. A positive frame is a set failure unless the selected set contains exactly one matched prediction for every reference instance and no false-positive or false-negative instance. Count mean absolute error and duplicate, merge, and split events are supporting outcomes.

Frame outcomes are averaged within each case, and cases contribute equally. Direct-test and external resamples preserve graph/node-only pairs, take the median across six outer-fold models within each graph-training seed, and then take the median across graph-training seeds 101, 103, and 107. Uncertainty uses 10,000 paired case-bootstrap replicates with seed 20260729. The two superiority endpoints form one Holm-adjusted family. Leave-one-case-out analysis tests whether either superiority direction reverses after any one case is removed.

Table 1: Main evaluation settings. The case or video is the inferential unit, and external domains remain separate.
<table><tr><td>Setting</td><td>Frames/cases</td><td>Label capability</td><td>Training/evaluation relation</td><td>Evidence role</td></tr><tr><td>CholecInstanceSeg development</td><td>4,246/18</td><td>Official instance IDs</td><td>Out-of-fold method development</td><td>Method freezing and progression</td></tr><tr><td>CholecInstanceSeg test</td><td>10,566/22</td><td>Official instance IDs</td><td>Fixed source models applied once</td><td>Sealed source confirmation</td></tr><tr><td>Endoscapes official test_seg</td><td>74/10</td><td>Native instances</td><td>Fixed source models; direct transfer</td><td>External target-domain evidence</td></tr><tr><td>ROBUST-MIPS Testing Stages 1–3</td><td>4,057/30</td><td>Native instances</td><td>Fixed source models; direct transfer</td><td>External target-domain evidence</td></tr></table>

## 4.3. Non-compensatory qualification of the method claim

The complete claim has three non-interchangeable parts. Instance-set performance improvement requires improvement in native-instance F1 and reduction in positive-frame set failure. Segmentation fidelity preservation requires that graphminus-node-only union Dice satisfies its noninferiority condition. Safety preservation is technical rather than clinical: it requires no increase in false activation on reference-empty frames and zero unresolved-selection flags. Evidence stability is assessed through the predefined seed-replication and influence rules; it is not an extra score that can ofset another dimension.

Each discovery seed had to satisfy all predefined conditions. F1 improvement had to be at least 0.03 with a 95% confidenceinterval lower bound above zero, and set-failure reduction had to be at least 0.05 with its lower bound above zero. Both Holmadjusted one-sided p-values had to be below 0.05. The union-Dice confidence-interval lower bound had to exceed −0 01, neither superiority direction could reverse under leave-one-case-out analysis, empty-frame false activation could not increase, and unresolved solver flags had to remain zero. Setting-level support requires complete support in at least two of the three fixed discovery seeds. Criteria and thresholds are identical across the direct settings.

The rule is non-compensatory: every applicable dimension must be satisfied within the same seed before that seed counts toward setting-level support. A favorable instance endpoint cannot replace unestablished fidelity or safety preservation, and a successful result in one domain cannot repair a failed result in another. Failure to establish safety preservation denotes an absence of support for preservation; it is not evidence that safety degradation occurred. No completed setting received the latter interpretation.

## 4.4. Label-contract boundary

Native instance identifiers preserve object grouping and support one-to-one matching, instance F1, count, and duplicate/merge/split interpretation. An 8-connected-component proxy derived from semantic foreground preserves only the component structure of the union and need not encode the same object identities. Results are therefore interpreted within their label contract. The proxy analysis uses identical predictions to examine observability; it neither trains nor evaluates the graph method under proxy references.

## 5. Results

## 5.1. Primary confirmation on the sealed source test

The sealed source test provides the strongest confirmation of the complete relational path. All three discovery seeds satisfied the predefined instance-set, fidelity, influence, empty-frame, and solver conditions (Table 2). Native-instance F1 increased by 0.0504 (95% CI 0.0382–0.0660), 0.0612 (0.0356–0.1038), and 0.0513 (0.0441–0.0770) for seeds 17, 23, and 41. Positive-frame set-failure rate decreased by 0.1060 (0.0825–0.1284), 0.0848 (0.0665–0.0996), and 0.0906 (0.0764–0.1133), respectively.

The instance-set gains were accompanied by segmentation fidelity and technical-safety preservation. Union Dice increased by 0.0450, 0.0551, and 0.0556, with all confidence intervals above the frozen noninferiority margin. Empty-frame false activation changes were -0.0736, -0.0901, and -0.0496; leave-onecase-out minima remained positive for both superiority endpoints; and no unresolved solver flag occurred. Both superiority tests had Holm-adjusted one-sided p = 0 00019998 for every seed. Together, these observations satisfied the predefined complete criteria on the sealed source test.

Figure 2 shows the case-level directions underlying the frozen setting summaries. The broadest and least consistent F1 spread occurs in Endoscapes; the domains remain descriptive panels rather than a pooled analysis.

Source development preceded this confirmation and served a diferent role. It supported method progression in two of three discovery seeds. Seeds 17 and 23 satisfied the complete development criteria. Seed 41 had an F1 diference of 0.0297 (95% CI 0.0087–0.0507), below the predefined 0.03 point criterion, and an empty-frame false-activation increase of 0.0267. The later sealed result confirms the method under the source-test relation without rewriting that earlier development uncertainty.

## 5.2. Direct external transfer

## 5.2.1. ROBUST-MIPS

Direct ROBUST-MIPS transfer reproduced the complete result in all three discovery seeds. Native-instance F1 increased by 0.0346–0.0714. Positive-frame set-failure rate decreased by 0.1160–0.1402. Union-Dice changes were 0.0224– 0.0633. Both superiority endpoints had Holm-adjusted one-sided $p = 0 . 0 0 0 1 9 9 9 8$ for each seed, leave-one-case-out directions remained positive, empty-frame false activation did not increase, and solver flags were zero. These results satisfied the complete predefined criteria within the ROBUST-MIPS contract.

The replication remains dataset specific. ROBUST-MIPS did not train, calibrate, or select the relational path, but the result does not imply that every external surgical domain will reproduce the efect or that the method is prospectively confirmed across all external data.

Table 2: Frozen case-level comparison of the complete relational path with the node-feature-matched path. Values are displayed to four decimals from the setting-specific decisions and receipts. ∆F1 and ∆Dice are graph minus node-only; set-failure reduction is node-only minus graph; empty ∆ is graph minus node-only false activation. “Complete” indicates that every predefined superiority, fidelity, leave-one-case-out, empty-frame, and solver condition was satisfied for that seed.
<table><tr><td>Setting</td><td>Seed</td><td>ΔF1 95% CI</td><td>Set-failure reduction 95% CI</td><td>∆Dice 95% CI</td><td>Empty ∆</td><td>Complete</td></tr><tr><td>Source test</td><td>17</td><td>.0504 [.0382, .0660]</td><td>.1060 [.0825, .1284]</td><td>.0450 [.0310, .0632]</td><td>-.0736</td><td>Yes</td></tr><tr><td>Source test</td><td>23</td><td>.0612 [.0356, .1038]</td><td>.0848 [.0665, .0996]</td><td>.0551 [.0344, .0982]</td><td>-.0901</td><td>Yes</td></tr><tr><td>Source test</td><td>41</td><td>.0513 [.0441, .0770]</td><td>.0906 [.0764, .1133]</td><td>.0556 [.0364, .0807]</td><td>-.0496</td><td>Yes</td></tr><tr><td>ROBUST-MIPS</td><td>17</td><td>.0346 [.0259, .0484]</td><td>.1160 [.1001, .1439]</td><td>.0224 [.0143, .0322]</td><td>-.0353</td><td>Yes</td></tr><tr><td>ROBUST-MIPS</td><td>23</td><td>.0714 [.0612, .0859]</td><td>.1402 [.1190, .1672]</td><td>.0633 [.0470, .0772]</td><td>-.1106</td><td>Yes</td></tr><tr><td>ROBUST-MIPS</td><td>41</td><td>.0459 [.0383, .0552]</td><td>.1357 [.1159, .1630]</td><td>.0378 [.0273, .0478]</td><td>-.0459</td><td>Yes</td></tr><tr><td>Endoscapes</td><td>17</td><td>.0211 [-.0131, .0633]</td><td>.1284 [.0332, .2313]</td><td>.0093 [-.0173, .0458]</td><td>+.0029</td><td>No</td></tr><tr><td>Endoscapes</td><td>23</td><td>.0586 [.0280, .0937]</td><td>.0984 [.0561, .1519]</td><td>.0545 [.0065, .1075]</td><td>.0000</td><td>Yes</td></tr><tr><td>Endoscapes</td><td>41</td><td>.0234 [-.0197, .0728]</td><td>.0913 [.0211, .1917]</td><td>.0320 [-.0043, .0877]</td><td>.0000</td><td>No</td></tr></table>

## 5.2.2. Endoscapes

Endoscapes was seed-unstable. Seed 23 satisfied every predefined component, with F1 improvement 0.0586 (95% CI 0.0280– 0.0937), set-failure reduction 0.0984 (0.0561–0.1519), union-Dice change 0.0545 (0.0065–0.1075), no increase in emptyframe false activation, and zero solver flags. Seeds 17 and 41 did not meet the F1 point and confidence requirements. Seed 17 additionally had a union-Dice confidence lower bound of -0.0173, below the -0.01 margin, and an empty-frame false-activation increase of 0.0029. Only one of three seeds therefore supported the complete claim.

Set-failure rate fell in all three Endoscapes seeds, but the full criterion held in only one. ROBUST-MIPS cannot compensate for that result because the populations and evidence contracts are distinct. The two settings therefore remain separate: ROBUST-MIPS supports successful direct transfer, whereas Endoscapes remains seed-unstable rather than supporting a pooled external conclusion.

## 5.3. What is preserved when the instance set improves

The sealed source test and ROBUST-MIPS paired instanceset improvement with preservation of the foreground union and the predefined technical-safety variables. In both settings, every seed that counted toward the method claim also preserved union Dice under the frozen margin, avoided increased emptyreference activation, and produced zero unresolved states. Segmentation fidelity preservation and safety preservation are therefore part of the positive method result rather than separate scorecards.

That conjunction matters in Endoscapes. Seed 17 reduced set failure, but did not establish the required F1 improvement, fidelity preservation, or empty-frame preservation. For this seed, the supported interpretation is narrower: safety preservation was not established. The analysis did not test or support the stronger conclusion that safety degradation occurred.

## 5.4. Dependence on the label contract

Identical predictions produced diferent instance conclusions when native object identities were replaced with 8-connected components of the same foreground union (Supplement Section 4, label-contract audit). In CholecInstanceSeg development, union Dice was exactly invariant by construction. All three Mask R-CNN seeds nevertheless had lower instance F1 and higher count error under the proxy contract, with case-level intervals excluding zero. The SegFormer-plus-components model was nearly unchanged in aggregate because its predictions already used the same component ontology.

The frozen post-hoc Endoscapes replication showed the same observability boundary. Union Dice remained invariant, while instance F1 decreased and count error increased for all four candidate bundles. Only 6 of 74 frames changed topology under the native-to-proxy conversion; the 68 unchanged-count frames had exactly zero contract deltas. The efect is therefore localized to the reference grouping change rather than to diferent predictions or image populations.

These analyses explain why foreground-union similarity does not establish native-instance identity and why cross-dataset evidence must retain its label contract. They do not evaluate the topology-aware selector against proxy references. The full native-versus-proxy numeric ledger, label-contract figure, and setting-level claim matrix are in Supplement Sections 4 and 5.

The profiled frames contained a median of 96 eligible nodes and 4,560 unique pairs; the respective 95th percentiles were 99 and 4,851. Peak recorded memory was 26.4 MB for the complete path and 15.1 MB for the node-only path. These isolated measurements characterize implementation cost under one hardware and candidate regime. They are not end-to-end latency measurements and do not establish real-time operation.

## 5.5. Supplemental boundary evidence

The supplemental leave-one-domain-out study tested broader held-out robustness, but did not support a general multi-domain claim. Under its distinct post-result training relation and graph seed 101, complete support occurred in one of three source-heldout seeds, zero of three Endoscapes-held-out seeds, one of three ROBUST-MIPS Stage-2 seeds, and two of three Stage-3 seeds. These results bound generality but do not replace the one-shot source test or the direct external evaluations.

![](images/67eb004712926871fdcb5a40bd30f147e2fcc47c080cfd81fc5a62988c62f514.jpg)

![](images/0487b3a9bd127298f2b195eaaa17fe8f9396fb6059b2e04ce878c70ab86cfb98.jpg)

![](images/29e9ec321b3b4010cf023dd35d97476919f1fbd30fe2e3cda01c2462182923e5.jpg)

![](images/3009e1566d5b85dee12688960b683cb36f0125a6e6ab5949ee7b87fe66a952dd.jpg)

![](images/2166df906019069d53881779f6074546e96823a5c10647db6af7ba0ced43b35d.jpg)  
Figure 2: Case-level diferences in the two superiority endpoints. Panels A–C show complete-path minus node-feature-matched instance F1; panels D–F show node-feature-matched minus complete-path set-failure rate. Points are cases or videos within discovery seed after the visualization-only nested median. The display is descriptive; Table 2 and frozen bootstrap receipts are confirmatory, and domains are not pooled.

The A2 supervision analysis also limits component attribution. It ended after prospectively amended Stage A at graph seed 101. Removing pair-competition supervision gave ∆F1 0.0057 (95% CI -0.0102–0.0182), set-failure reduction 0.0121 (-0.0071–0.0374), and union-Dice change -0.0020 (-0.0141– 0.0099), producing an unstable Stage-A interpretation. Removing cardinality supervision gave ∆F1 -0.0118 (-0.0181—0.0030), set-failure reduction -0.0068 (-0.0170–0.0039), and union-Dice change -0.0022 (-0.0125–0.0094), providing no Stage-A support. Stage B was permanently cancelled under the prospective budget amendment and was not evaluated; these data do not isolate the contribution of either component.

Finally, the LODO source-held-out empty-frame follow-up is descriptive post-hoc evidence. Seed 23 showed a distributed positive-excess pattern and seed 41 a concentrated pattern, with low frame-level overlap between their graph-only activations. The frozen fields do not causally identify cardinality, domain shift, or another mechanism, and no confirmatory probability value was available. The audit therefore leaves the afected state as safety preservation not established, not supported safety degradation.

## 6. Discussion

The evaluated complete graph path improved the assembly of fixed surgical mask candidates into an instance set in the supported settings. The design jointly represents candidate geometry, relational context, output cardinality, and structured subset selection. The sealed source test provides the principal confirmation. The complete-path result was reproduced on ROBUST-MIPS under the direct-transfer protocol while preserving foreground fidelity and predefined technical safety.

The matched comparator determines the resolution of this conclusion. Both paths receive the same candidates and node features, use the same supervision and training roles, predict cardinality, and invoke the same exact selector. The complete path also contains edge geometry, graph messages, and substantially greater model capacity. The primary comparison therefore identifies a complete-path efect that includes relational representation, cardinality behavior, selector operation, and capacity diferences. It does not identify the separate contribution of the exact selector, the cardinality head, individual edge features, or message passing. The Stage-A-only A2 results do not change this boundary.

Table 3: Frozen component-level runtime profile on 300 source-development frames using an NVIDIA RTX 4070 SUPER. Timings are per frame and do not include Mask2Former inference or data loading.
<table><tr><td>Component</td><td>Median (ms)</td><td>95th percentile (ms)</td><td>Measurement scope</td></tr><tr><td>Complete-graph construction</td><td>390.9</td><td>435.8</td><td>Node/pair feature assembly for eligible candidates</td></tr><tr><td>Complete-path forward pass</td><td>0.98</td><td>2.30</td><td>Relational network and prediction heads</td></tr><tr><td>Node-only forward pass</td><td>0.83</td><td>0.87</td><td>Matched node-only network and heads</td></tr><tr><td>Exact subset selector</td><td>23.0</td><td>277.9</td><td>Exhaustive or zero-gap MILP selection</td></tr></table>

External behavior varied by domain. ROBUST-MIPS reproduced the complete conclusion in all seeds, whereas Endoscapes reproduced it in only one. Diferences in sample size, candidate behavior, visual domain, and annotation practice may all be relevant, but the completed evidence does not identify a causal explanation. Label capability provides one clear interpretive constraint: a foreground union can be identical while the reference grouping and observable instance identity change. Population and label contracts must therefore remain visible when the method is transported.

The non-compensatory evaluation keeps these outcomes together. Instance F1 and set failure test whether the instance set improves; union Dice tests whether foreground segmentation is preserved; empty-reference activation and unresolved states test predefined technical-safety behavior. A claim is supported only when these properties align within the same seed and setting. This rule explains why favorable Endoscapes set-failure directions do not establish stable transfer and why an unestablished preservation condition is not automatically evidence of degradation.

Exploratory mechanism controls were consistent with a substantial cardinality contribution but did not identify a stable fixed-cardinality relation-specific residual. Capacity-matched, shufled, high-resolution, calibration, and presence controls also failed to establish a stable joint performance-and-safety condition. These pressure tests do not alter the formal complete-path result; their designs and values are reported in the Supplement.

Several limitations bound the result. The method operates on candidates from one fixed Mask2Former family, so detectorfamily and architecture generality are untested. The complete graph is quadratic in eligible-candidate count, exact optimization has combinatorial worst-case complexity, and the component timings in Table 3 do not establish real-time deployment. Only two direct external domains were evaluated, and Endoscapes contains 10 video units with correspondingly wide uncertainty. The supplemental multi-domain study did not support a general held-out robustness claim. Technical safety here covers emptyreference activation and unresolved selection states, not clinical safety, clinical benefit, workflow utility, or patient outcome. Component efects remain unresolved because A2 stopped after Stage A, and the empty-frame follow-up remains post hoc and descriptive.

## 7. Conclusion

Good foreground masks can still form the wrong surgicalinstrument instance set because candidate validity depends on relations and variable cardinality. Topology-aware query selection addresses this problem with a complete candidate graph, relational inference, cardinality prediction, and exact structured subset selection. Relative to a node-feature-matched path, the complete relational path was supported on the sealed source test and directly replicated on ROBUST-MIPS while preserving segmentation fidelity and predefined technical safety. Endoscapes did not establish stable direct transfer, and supplemental evidence did not support general multi-domain held-out robustness or component-specific attribution. The evidence supports a bounded claim: the evaluated complete topology-aware path improved coherent instance-set construction from fixed Mask2Former candidates under the evaluated native-instance contracts, but its broader transfer and any component-specific efect remain unestablished.

## Data and code availability

The datasets remain governed by their original providers and cited source publications. This manuscript does not alter those access conditions. Analysis code, fixed decision files, and audit receipts used to prepare this submission are available at https://github.com/yqjyzzz/ structured-instance-set-selection. The underlying datasets are not redistributed and must be obtained from their respective oficial providers.

## Ethics statement

No prospective human-subject recruitment or intervention was performed for this methodological analysis. Dataset-specific ethics and consent information is provided by the original dataset publications.

## Declaration of generative AI and AI-assisted technologies in the writing process

During manuscript preparation, the authors used OpenAI Codex to assist with language drafting, document organization, consistency checking, and typesetting. The authors reviewed and edited the resulting content and take responsibility for the manuscript.

## Declaration of competing interest

Yang Zhang is employed by Wuhan United Imaging Surgical Co., Ltd. The company had no role in the study design, data analysis, interpretation of results, or decision to submit the manuscript. The authors declare no other competing interests.

## Funding

This research received no specific grant from any funding agency in the public, commercial, or not-for-profit sectors.

## References

Alabi, O., Toe, K.K.Z., Zhou, Z., Budd, C., Raison, N., Shi, M., Vercauteren, T., 2025. CholecInstanceSeg: A tool instance segmentation dataset for laparoscopic surgery. Scientific Data 12, 825. doi:10.1038/ s41597-025-05163-w.

Ángeles Cerón, J.C., Ochoa Ruiz, G., Chang, L., Ali, S., 2022. Real-time instance segmentation of surgical instruments using attention and multi-scale feature fusion. Medical Image Analysis 81, 102569. doi:10.1016/j.media. 2022.102569.

Carion, N., Massa, F., Synnaeve, G., Usunier, N., Kirillov, A., Zagoruyko, S., 2020. End-to-end object detection with transformers, in: European Conference on Computer Vision, pp. 213–229. doi:10.1007/ 978-3-030-58452-8\_13.

Cheng, B., Misra, I., Schwing, A.G., Kirillov, A., Girdhar, R., 2022. Maskedattention mask transformer for universal image segmentation, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 1290–1299. URL: https://openaccess.thecvf.com/content/ CVPR2022/html/Cheng\_Masked-Attention\_Mask\_Transformer\_ for\_Universal\_Image\_Segmentation\_CVPR\_2022\_paper.html, doi:10.1109/CVPR52688.2022.00135.

Cheng, B., Schwing, A.G., Kirillov, A., 2021. Per-pixel classification is not all you need for semantic segmentation, in: Advances in Neural Information Processing Systems, pp. 17864–17875. URL: https://proceedings.neurips.cc/paper/2021/hash/ 950a4152c2b4aa3ad78bdd6b366cc179-Abstract.html.

Fang, Y., Yang, S., Wang, X., Li, Y., Fang, C., Shan, Y., Feng, B., Liu, W., 2021. Instances as queries, in: Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 6910–6919. URL: https://openaccess.thecvf.com/content/ICCV2021/ html/Fang\_Instances\_As\_Queries\_ICCV\_2021\_paper.html, doi:10.1109/ICCV48922.2021.00683.

Han, Z., Budd, C., Zhang, G., Tian, H., Bergeles, C., Vercauteren, T., 2026. ROBUST-MIPS: A combined skeletal pose and instance segmentation dataset for laparoscopic surgical instruments. Scientific Data 13, 684. doi:10.1038/ s41597-026-06938-5.

He, K., Gkioxari, G., Dollár, P., Girshick, R., 2017. Mask R-CNN, in: Proceedings of the IEEE International Conference on Computer Vision, pp. 2961–2969. doi:10.1109/ICCV.2017.322.

Hosang, J., Benenson, R., Schiele, B., 2017. Learning non-maximum suppression, in: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 4507–4515. URL: https: //openaccess.thecvf.com/content\_cvpr\_2017/html/Hosang\_ Learning\_Non-Maximum\_Suppression\_CVPR\_2017\_paper.html, doi:10.1109/CVPR.2017.685.

Hu, H., Gu, J., Zhang, Z., Dai, J., Wei, Y., 2018. Relation networks for object detection, in: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 3588– 3597. URL: https://openaccess.thecvf.com/content\_cvpr\_2018/ html/Hu\_Relation\_Networks\_for\_CVPR\_2018\_paper.html, doi:10. 1109/CVPR.2018.00378.

Jena, R., Zhornyak, L., Doiphode, N., Chaudhari, P., Buch, V., Gee, J., Shi, J., 2023. Beyond mAP: Towards better evaluation of instance segmentation, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 11309–11318. URL: https://openaccess.thecvf. com/content/CVPR2023/html/Jena\_Beyond\_mAP\_Towards\_Better\_

Evaluation\_of\_Instance\_Segmentation\_CVPR\_2023\_paper.html, doi:10.1109/CVPR52729.2023.01088.

Kurmann, T., Márquez-Neila, P., Allan, M., Wolf, S., Sznitman, R., 2021. Mask then classify: multi-instance segmentation for surgical instruments. International Journal of Computer Assisted Radiology and Surgery 16, 1227– 1236. doi:10.1007/s11548-021-02404-2.

Maier-Hein, L., Eisenmann, M., Reinke, A., et al., 2018. Why rankings of biomedical image analysis competitions should be interpreted with care. Nature Communications 9, 5217. doi:10.1038/s41467-018-07619-7.

Maier-Hein, L., Reinke, A., Godau, P., et al., 2024. Metrics reloaded: recommendations for image analysis validation. Nature Methods 21, 195–212. doi:10.1038/s41592-023-02151-z.

Mascagni, P., Alapatt, D., Murali, A., Vardazaryan, A., Garcia, A., Okamoto, N., Costamagna, G., Mutter, D., Marescaux, J., Dallemagne, B., Padoy, N., 2025. Endoscapes, a critical view of safety and surgical scene segmentation dataset for laparoscopic cholecystectomy. Scientific Data 12, 331. doi:10. 1038/s41597-025-04642-4.

Ostmeier, S., Axelrod, B., Isensee, F., Bertels, J., Mlynash, M., Christensen, S., Lansberg, M.G., Albers, G.W., Sheth, R., Verhaaren, B.F.J., Mahammedi, A., Li, L.J., Zaharchuk, G., Heit, J.J., 2023. USE-Evaluator: Performance metrics for medical image segmentation models supervised by uncertain, small or empty reference annotations in neuroimaging. Medical Image Analysis 90, 102927. doi:10.1016/j.media.2023.102927.

Pato, L.V., Negrinho, R.M.P., Aguiar, P.M.Q., 2020. Seeing without looking: Contextual rescoring of object detections for AP maximization, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 14610–14618. URL: https://openaccess.thecvf.com/content\_CVPR\_2020/ html/Pato\_Seeing\_without\_Looking\_Contextual\_Rescoring\_ of\_Object\_Detections\_for\_AP\_CVPR\_2020\_paper.html, doi:10.1109/CVPR42600.2020.01462.

Reinke, A., Tizabi, M.D., Baumgartner, M., et al., 2024. Understanding metricrelated pitfalls in image analysis validation. Nature Methods 21, 182–194. doi:10.1038/s41592-023-02150-0.

Roß, T., Bruno, P., Reinke, A., Wiesenfarth, M., Koeppel, L., Full, P.M., Pekdemir, B., Godau, P., Trofimova, D., Isensee, F., Adler, T.J., Tran, T.N., Moccia, S., Calimeri, F., Müller-Stich, B.P., Kopp-Schneider, A., Maier-Hein, L., 2023. Beyond rankings: Learning (more) from algorithm validation. Medical Image Analysis 86, 102765. doi:10.1016/j.media.2023.102765.

Roß, T., Reinke, A., Full, P.M., et al., 2021. Comparative validation of multiinstance instrument segmentation in endoscopy: Results of the ROBUST-MIS 2019 challenge. Medical Image Analysis 70, 101920. doi:10.1016/j. media.2020.101920.

Tan, Z., Nie, X., Qian, Q., Li, N., Li, H., 2019. Learning to rank proposals for object detection, in: Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 8273–8281. URL: https://openaccess.thecvf.com/content\_ICCV\_2019/html/Tan\_ Learning\_to\_Rank\_Proposals\_for\_Object\_Detection\_ICCV\_ 2019\_paper.html, doi:10.1109/ICCV.2019.00836.

Tejani, A.S., Klontzas, M.E., Gatti, A.A., Mongan, J.T., Moy, L., Park, S.H., Kahn, C.E.J., CLAIM 2024 Update Panel, 2024. Checklist for artificial intelligence in medical imaging (CLAIM): 2024 update. Radiology: Artificial Intelligence 6, e240300. doi:10.1148/ryai.240300.

## Supplementary Material

## Appendix A. Scope and evidence roles

This supplement provides traceability for the frozen formal analyses and records exploratory mechanism diagnostics. It does not change any model, threshold, seed, sample definition, estimand, label contract, or machine decision. Source development, direct source test, Endoscapes, and ROBUST-MIPS remain separate settings.

Formal evidence is confirmatory for the bounded system claim. Mechanism reports are exploratory pressure tests. They may constrain interpretation, but they cannot upgrade a formal result or prove that one component caused the complete-path diference.

## Appendix B. Frozen evaluation protocol

The unit of analysis is the case or video. Frame outcomes are averaged within case, cases contribute equally, paired graph/node-only outputs are preserved, and the direct-test/external estimand takes the median across six source folds within each graph-training seed and then the median across graph-training seeds 101, 103, and 107. Uncertainty uses 10,000 paired case-bootstrap replicates with seed 20260729. Discovery seeds are 17, 23, and 41. There are 54 atomic pairs per direct setting.

For a seed to count, instance F1 improvement is at least 0.03 with a 95% interval lower bound above zero; positive-frame set-failure reduction is at least 0.05 with a lower bound above zero; both Holm-adjusted one-sided superiority p-values are below 0.05; the union-Dice lower bound is above -0.01; neither superiority endpoint reverses under leave-one-case-out analysis; empty-frame false activation does not increase; and unresolved solver flags are zero. A setting is supported when the complete result occurs in at leas two of three discovery seeds.

Failure to establish safety preservation means that preservation was not supported for the relevant analysis. It is not evidence that safety degradation occurred; the formal analyses did not reach that stronger conclusion.

## Appendix C. Formal evidence ledger

Appendix C.1. Setting scope

Table S1: Main evaluation settings and evidence roles.
<table><tr><td>Setting</td><td>Frames</td><td>Cases</td><td>Pairs</td><td>Label contract</td><td>Relation</td><td>Decision</td></tr><tr><td>CholecInstanceSeg development</td><td>4,246</td><td>18</td><td></td><td>Native instance IDs</td><td>Out-of-fold development</td><td>Method progressed in 2/3 seeds</td></tr><tr><td>CholecInstanceSeg source test</td><td>10,566</td><td>22</td><td>54</td><td>Native instance IDs</td><td>Fixed source models applied once</td><td>Complete support in 3/3 seeds</td></tr><tr><td>Endoscapes test_seg</td><td>74</td><td>10</td><td>54</td><td>Native instances</td><td>Fixed source models; direct transfer</td><td>Seed-unstable support (1/3)</td></tr><tr><td>ROBUST-MIPS Testing Stages 1–3</td><td>4,057</td><td>30</td><td>54</td><td>Native instances</td><td>Fixed source models; direct transfer</td><td>Complete support in 3/3 seeds</td></tr></table>

## Appendix C.2. Source development

<table><tr><td rowspan="2">Table S2. Source development by seed.</td><td>Seed</td><td>∆F1</td><td>Set-failure reduction</td><td>∆Dice</td><td>Empty ∆</td><td>Complete</td></tr><tr><td>17</td><td>0.0610</td><td>0.1281</td><td>0.0645</td><td>-0.1251</td><td>Yes</td></tr><tr><td rowspan="2"></td><td>23</td><td>0.0607</td><td>0.0956</td><td>0.0599</td><td>-0.1852</td><td>Yes</td></tr><tr><td>41</td><td>0.0297</td><td>0.0666</td><td>0.0263</td><td>+0.0267</td><td>No</td></tr></table>

The source-development decision is supported by 2/3 seeds. Seed 41 is not pooled with the two supporting seeds: its F1 diference is below the 0.03 point criterion and empty false activation increased.

## Appendix C.3. Direct source test

Table S3: Direct source-test decisions by discovery seed.
<table><tr><td>Seed</td><td>∆F1 [95% CI]</td><td>Set-failure reduction [95% CI]</td><td>∆Dice [95% CI]</td><td>Empty ∆</td><td>Holm p</td><td>LOCO F1 / set</td></tr><tr><td>17</td><td>0.0504 [0.0382, 0.0660]</td><td>0.1060 [0.0825, 0.1284]</td><td>0.0450 [0.0310, 0.0632]</td><td>-0.0736</td><td>0.00019998</td><td>0.04696 / 0.10070</td></tr><tr><td>23</td><td>0.0612 [0.0356, 0.1038]</td><td>0.0848 [0.0665, 0.0996]</td><td>0.0551 [0.0344, 0.0982]</td><td>-0.0901</td><td>0.00019998</td><td>0.04406 / 0.07680</td></tr><tr><td>41</td><td>0.0513 [0.0441, 0.0770]</td><td>0.0906 [0.0764, 0.1133]</td><td>0.0556 [0.0364, 0.0807]</td><td>-0.0496</td><td>0.00019998</td><td>0.04909 / 0.08847</td></tr></table>

The direct source-test estimand takes the median of the six source-fold equal-case means within each graph-training seed and then the median across graph-training seeds. All three discovery seeds supported the complete source-test conclusion.

Table S4: Direct external-setting formal decisions.
<table><tr><td>Setting</td><td>Seed</td><td>ΔF1</td><td>Set-failure reduction</td><td>∆Dice</td><td>Empty ∆</td><td>Complete</td></tr><tr><td>Endoscapes</td><td>17</td><td>0.0211</td><td>0.1284</td><td>0.0093</td><td>+0.0029</td><td>No</td></tr><tr><td>Endoscapes</td><td>23</td><td>0.0586</td><td>0.0984</td><td>0.0545</td><td>0.0000</td><td>Yes</td></tr><tr><td>Endoscapes</td><td>41</td><td>0.0234</td><td>0.0913</td><td>0.0320</td><td>0.0000</td><td>No</td></tr><tr><td>ROBUST-MIPS</td><td>17</td><td>0.0346</td><td>0.1160</td><td>0.0224</td><td>-0.0353</td><td>Yes</td></tr><tr><td>ROBUST-MIPS</td><td>23</td><td>0.0714</td><td>0.1402</td><td>0.0633</td><td>-0.1106</td><td>Yes</td></tr><tr><td>ROBUST-MIPS</td><td>41</td><td>0.0459</td><td>0.1357</td><td>0.0378</td><td>-0.0459</td><td>Yes</td></tr></table>

Endoscapes support is $1 / 3 ,$ so stable direct transfer was not established. ROBUST-MIPS support is 3/3. The settings are not pooled.

## Appendix D. Label-contract audit

The native-versus-8-connected-component analysis applies identical predictions to two reference constructions. Union Dice is invariant by construction, while instance grouping and count can change. These contrasts are interpretive and do not evaluate the graph method under proxy labels.

Table S5: Native-versus-proxy label-contract audit.
<table><tr><td>Population</td><td>Model</td><td>∆ union Dice</td><td>∆ instance F1 [95% CI]</td><td>∆ count MAE [95% CI]</td></tr><tr><td>Cholec development</td><td>Mask R-CNN s17</td><td>0.0000</td><td>-0.0191 [-0.0334, -0.0069]</td><td>+0.0548 [+0.0277, +0.0813]</td></tr><tr><td>Cholec development</td><td>Mask R-CNN s23</td><td>0.0000</td><td>-0.0212 [-0.0350, -0.0110]</td><td>+0.0588 [+0.0376, +0.0823]</td></tr><tr><td>Cholec development</td><td>Mask R-CNN s41</td><td>0.0000</td><td>-0.0186 [-0.0325, -0.0070]</td><td>+0.0503 [+0.0321, +0.0703]</td></tr><tr><td>Cholec development</td><td>SegFormer s17</td><td>0.0000</td><td>+0.0016 [-0.0022, +0.0054]</td><td>-0.0104 [-0.0285, +0.0060]</td></tr><tr><td>Endoscapes</td><td>Mask R-CNN s17/23/41</td><td>0.0000</td><td>-0.0340 [-0.0646, -0.0079]</td><td>+0.1215 [+0.0291, +0.2333]</td></tr><tr><td>Endoscapes</td><td>Mask2Former s17</td><td>0.0000</td><td>-0.0323 [-0.0632, -0.0062]</td><td>+0.1215 [+0.0291, +0.2333]</td></tr></table>

## Appendix E. Setting-level claim matrix

Table S6: Supported claim category by setting. Entries interpret frozen category transitions and are not pooled scores.
<table><tr><td>Setting</td><td>Instance-set improvement</td><td>Fidelity preservation</td><td>Safety preservation</td><td>Evidence stability</td></tr><tr><td>Source development</td><td>2/3 seeds complete; seed 41 below F1 threshold</td><td>Not independent confir- mation</td><td>Not complete in seed 41; empty activation in-</td><td>Progression dence only</td></tr><tr><td>Direct source test</td><td>3/3 seeds complete</td><td>3/3 met the frozen union-Dice condition</td><td>creased 3/3 had no empty-frame increase; solver flags</td><td>Stable within sealed contract</td></tr><tr><td>Direct Endoscapes</td><td>1/3 complete; stable transfer not established</td><td>1/3 complete; seeds 17 and 41 failed the F1/fidelity conjunction</td><td>zero Preservation not estab- lished; seed 17 had an</td><td>Seed-unstable direct transfer</td></tr><tr><td>Direct MIPS</td><td>ROBUST- 3/3 seeds complete</td><td>3/3 met the frozen union-Dice condition</td><td>empty-frame increase 3/3 had no empty-frame increase; solver flags zero</td><td>Complete within the ROBUST-MIPS con- tract</td></tr></table>

Failure to establish safety preservation is not evidence that safety degradation occurred.

![](images/44c1248bada46f35dc7bf2a05f53140ba1c833d2543bb7a6bf1d319e0a701da7.jpg)  
Figure S1: Label-contract dependence under identical predictions. Union-Dice diferences remain zero by construction, whereas native-to-proxy changes in instance F1 and count error appear where reference grouping changes. The figure is an observability audit, not an evaluation of the selector under proxy labels.

## Appendix F. Exploratory mechanism diagnostics

## Appendix F.1. True cardinality and shared-cardinality crossover

The true-cardinality report contains 12 complete domain–seed–arm records. Count MAE, solver flags, and empty false activation were zero in the reported control outputs. Descriptive external means did not show a stable positive graph-over-node relation under fixed cardinality. The graph was not consistently best across Endoscapes or ROBUST-MIPS seeds. These descriptive results do not identify a fixed-cardinality relation-specific gain.

The shared-cardinality descriptive rows gave F1 values 0.7656, 0.7540, and 0.7481 for graph-cardinality with graph, capacity, and shufled selectors; 0.7142, 0.7042, and 0.7016 for capacity-cardinality; 0.7489, 0.7414, and 0.7374 for shufled-cardinality; and 0.7761, 0.7785, and 0.7693 for true-cardinality. The selector spread was 0.0175 versus a cardinality-source spread of 0.0619. The pattern is consistent with a substantial cardinality contribution, but it does not prove mediation or isolate a relation-specific residual.

## Appendix F.2. Capacity, shufled, and candidate-coverage controls

The capacity-matched node control had 59,334 parameters versus 59,120 in the graph path. The reported anchors included a safety failure where empty activation changed from 8 to 10, so the follow-up rule did not pass. Shufled and cross-axis controls remain descriptive because capacity and selector efects are not jointly isolated

In the source-test candidate-missingness atlas, 135 of 1,553 positive ground-truth instances were missing from the candidate pool (0.087); the missing targets had mean area 716.3 versus 3,482.1 for covered targets. In a seed-23/fold-1/g101 high-resolution anchor, candidate ceiling rose from 0.8831 to 0.9436, but downstream F1 at replacement quota 20 fell from 0.7189 to 0.6853 and empty activations rose from 7 to 9. This separates candidate coverage from final utility.

## Appendix F.3. Calibration, presence, and A2

Calibration did not provide a stable joint repair across the reported controls, and presence controls failed when lower empty activation was accompanied by positive loss-budget violations. A2 ended at Stage A: pair-supervision removal gave ∆F1 0.0057,

set-failure reduction 0.0121, and ∆Dice -0.0020; cardinality-supervision removal gave ∆F1 -0.0118, set-failure reduction -0.0068, and ∆Dice -0.0022. Stage B was not run. None of these controls identifies a single component efect.

## Appendix G. Supplemental held-out-domain relation

The supplemental leave-one-domain-out analysis did not support multi-domain held-out robustness. It is a diferent training/evaluation relation from the direct settings and is reported only to bound generality. Where empty false activation increased, the interpretation remains that safety preservation was not established, not that safety degradation was supported.