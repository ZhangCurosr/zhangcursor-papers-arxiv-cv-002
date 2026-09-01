# Multi-View Reflective Surface Inspection via Semantic–Saliency Cross-Verification

Van-Giang Nguyen<sup>†</sup>, Thanh-Tuan Tran<sup>†</sup> , Xuan-Hieu Phan , and Xiem HoangVan

University of Engineering and Technology, Vietnam National University, Hanoi 10000, Vietnam.

<sup>†</sup>Equal contribution Corresponding author

Abstract—Reflective smartphone cover glass is challenging to inspect from a single fixed viewpoint because defect visibility varies with viewing geometry and specular reflections. This gives rise to two practical challenges: defects may be weakly observable from certain viewpoints, while the available visual evidence may remain spatially ambiguous. To address these issues, we propose a multiview inspection framework in which each RGB observation is processed by a shared per-view expert. A vision–language model (VLM) produces class-aware semantic boxes, while a normalreference reconstruction branch provides class-agnostic saliency. Their spatial agreement is used as supporting evidence to rank semantic proposals without modifying their coordinates or treating saliency as ground truth. The resulting evidence records are combined at product level without cross-view registration. On 282 production-line images, semantic–saliency association improves AP<sub>50</sub> from 52.6% to 62.6% by re-ranking fixed semantic proposals. Across 94 products, cross-view evidence recall R<sub>prod</sub>@0.5 increases from 75.5% for the best single view to 88.3% using all three views. These results support the complementary roles of semantic– saliency cross-verification and additional optical observations in reflective-surface inspection.

Index Terms—industrial inspection, smartphone cover glass, vision–language model, anomaly localization, multi-view inspection [

## I. INTRODUCTION

Smartphone cover glass is difficult to inspect from a single fixed observation because defect visibility depends strongly on the imaging condition. The surface is dark and highly reflective, while defects such as scratches and cracks may occupy only a small region of the image. Their contrast changes with the geometry of illumination, surface, and sensor. Existing smartphone glass inspection systems therefore acquire the surface under different illumination directions or viewpoints to expose weak defects that may be inconspicuous in one observation [1], [2]. This motivates the use of multiple observations as complementary optical evidence for the same product.

Conventional inspection methods commonly formulate defect localization as supervised object detection. Faster R-CNN and recent YOLO variants provide strong localization when the target categories and representative box annotations are available [3], [4]. Their semantic scope, however, is determined by the defect classes represented during training. Changes in the inspection vocabulary therefore require corresponding labeled examples and model adaptation. This can be restrictive in manufacturing, where inspection criteria may include both local surface defects and relational conditions such as misalignment or unexpected components. Open-set grounding and vision-language models provide a more flexible semantic interface because the target concept can be specified through language [5]–[7]. This flexibility does not guarantee precise localization. Recent industrial vision-language studies still report difficulty in resolving small local anomalies and producing accurate anomaly localization [6], [8]. Normalreference anomaly detection provides a complementary source of spatial information. Embedding and reconstruction methods model normal appearance and localize regions that deviate from it without requiring labels for every defect category [9]–[11]. These responses are spatially informative but class-agnostic. On reflective glass, however, a strong appearance deviation may arise from either a true defect or a specular pattern. Saliency therefore cannot be treated as ground truth for a semantic prediction. Conversely, a semantic prediction may identify the correct defect concept while providing imprecise localization. We use the spatial agreement between semantic localization and separately generated normal-reference saliency as supporting evidence rather than allowing either branch to determine the result alone.

![](images/08248984b4e0141c6aafbd29b7c66a11fcb8673c8c2da988708f827d35546067.jpg)  
Fig. 1: Conventional single-view inspection (above) and the proposed multi-view semantic–saliency framework (below). Per-view semantic proposals are cross-verified with saliency evidence before product-level aggregation.

Multiple observations address a different limitation. Real-

IAD established multi-view industrial anomaly evaluation, while subsequent work such as Multi-Flow explored information exchange across observations [12], [13]. Our objective is different. The views are used to expose the reflective surface under different optical conditions rather than to recover geometry or establish spatial correspondence across cameras. Each image is processed in its native coordinate system, and only the resulting evidence is combined at product level.

Fig. 1 contrasts conventional single-view inspection with our formulation. Each RGB observation is processed by a shared per-view expert, where a VLM provides classaware localization and a normal-reference branch provides spatial deviation evidence. Their association forms a per-view evidence record, and the resulting records are jointly considered at product level to determine the inspection outcome. The predicted defect and inspection context are then used to retrieve the corresponding manufacturer rule for reporting.

The main contributions are as follows.

• We propose a multi-view inspection framework that aggregates per-view evidence at product level without cross-view registration.

• We introduce semantic-saliency cross-verification, where normal-reference saliency provides spatial support for VLM defect proposals without modifying their localization.

• We evaluate the per-view expert on a public benchmark and a conveyor-based production prototype, separately quantifying within-view association and cross-view optical complementarity.

## II. METHOD

The proposed system, shown in Fig. 2, separates per-view evidence construction from product-level reasoning. Each observation is first converted into a common evidence record through semantic–saliency association, and only these records are combined across views.

## A. Per-View Semantic–Saliency Expert

Given an RGB observation $I _ { v } ,$ , the expert in Fig. 3 extracts two complementary cues. The VLM provides defect semantics and localization, while the normal-reference branch provides spatial evidence of deviation from expected appearance. The two prediction paths remain separate until spatial association.

Semantic localization. The VLM receives $I _ { v }$ and a fixed inspection prompt $P _ { \mathrm { d e t } }$ specifying the defect vocabulary, relevant geometric cues, and output schema. Its parsed output is

$$
\mathcal { D } _ { v } = \{ ( c _ { v j } , b _ { v j } ^ { s } ) \} _ { j = 1 } ^ { n _ { v } } ,\tag{1}
$$

where $c _ { v j }$ is the predicted class and $b _ { v j } ^ { s }$ is the corresponding semantic box. Coordinates are generated on a normalized [0, 1000] grid and mapped deterministically to image pixels. Invalid classes and malformed boxes are rejected without an additional language-model call.

Normal-reference evidence. A complementary spatial cue is obtained from the reconstruction pathway of VLMDiff [11], adapted using normal target-domain images and frozen during evaluation. Let $\widehat { I _ { v } } = g _ { \theta } \bar { ( } I _ { v } )$ denote the reconstructed image and $\Delta _ { v } ^ { ( \ell ) } ( x )$ the feature-space dissimilarity between $I _ { v }$ and $\widehat { I } _ { v }$ at level ℓ. The responses are combined into

$$
A _ { v } ( x ) = \mathrm { N o r m } \left( \sum _ { \ell = 1 } ^ { L } w _ { \ell } \mathrm { U p } \Big ( \Delta _ { v } ^ { ( \ell ) } \Big ) ( x ) \right) , \qquad \sum _ { \ell = 1 } ^ { L } w _ { \ell } = 1 ,\tag{2}
$$

where $A _ { v }$ is the normal-deviation saliency map. Thresholding $A _ { v }$ at $\tau _ { a }$ and removing components smaller than $\tau _ { \mathrm { a r e a } }$ yields

$$
\mathcal { A } _ { v } = \{ b _ { v k } ^ { a } \} _ { k = 1 } ^ { m _ { v } } ,\tag{3}
$$

where each $b _ { v k } ^ { a }$ bounds one surviving saliency region.

Semantic–saliency cross-verification. The two branches produce different spatial representations. The VLM returns class-aware boxes, while the normal-reference branch produces a dense class-agnostic response. Connected saliency regions are converted to boxes only to place both cues in a common spatial representation, allowing their consistency to be measured directly. For semantic proposal j,

$$
O _ { v j k } = \mathrm { I o U } ( b _ { v j } ^ { s } , b _ { v k } ^ { a } ) , \qquad o _ { v j } ^ { * } = \operatorname* { m a x } _ { k } O _ { v j k } ,\tag{4}
$$

with $o _ { v j } ^ { * } = 0$ when no saliency region is present. A proposal is marked as spatially supported when

$$
u _ { v j } = \mathbb { I } [ o _ { v j } ^ { * } \ge \tau _ { o } ] .\tag{5}
$$

The overlap $o _ { v j } ^ { * }$ is used only as spatial support for ranking. It is not a calibrated confidence and does not establish prediction correctness. Semantic coordinates remain unchanged, and weakly supported proposals are retained because reflection can produce strong saliency while some relational defects may yield weak local residuals.

The per-view evidence is summarized as

$$
\mathcal { E } _ { v } = \left( \mathcal { D } _ { v } , \mathcal { A } _ { v } , \mathbf { o } _ { v } , \mathbf { u } _ { v } , F _ { v } \right) ,\tag{6}
$$

where $\mathbf { o } _ { v } = \{ o _ { v j } ^ { * } \} _ { j = 1 } ^ { n _ { v } } , \mathbf { u } _ { v } = \{ u _ { v j } \} _ { j = 1 } ^ { n _ { v } }$ , and $F _ { v }$ renders the semantic and saliency evidence for the current observation. The record preserves defect semantics, localization, spatial support, and visual evidence in a common schema. It forms the interface to product-level reasoning.

## B. Multi-View Product Decision

Because each observation is represented by the same evidence record, the per-view expert can be reused across the available views. For a product observed from $\mathcal { V } _ { N } = \{ 1 , \ldots , N \}$

$$
{ \mathcal E } _ { v } = f _ { \mathrm { e x p e r t } } ( I _ { v } , P _ { \mathrm { d e t } } ) , \qquad v \in { \mathcal V } _ { N } .\tag{7}
$$

Each view remains in its native image coordinate system because the observations are used to vary defect visibility rather than reconstruct a common geometry. Cross-view registration is therefore not required. The product-level VLM jointly considers the resulting records,

$$
( \widehat { y } , \widehat { c } , \widehat { \mathcal { V } } _ { s } ) = f _ { \mathrm { d e c } } \left( \{ \mathcal { E } _ { v } \} _ { v \in \mathcal { V } _ { N } } , P _ { \mathrm { d e c } } \right) ,\tag{8}
$$

where $\widehat { y }$ is the inspection verdict, bc is the predicted defect class, and $\widehat { \mathcal { V } } _ { s }$ denotes the supporting observations. No majority rule is imposed. A defect supported by one observation can therefore contribute to the final verdict even when it is weak or absent in other views. This formulation separates view processing from product reasoning. Adding an observation requires another pass through the same expert and an additional evidence record, without changing the per-view model. The marginal benefit of additional optical observations is evaluated in Section III-C.

![](images/890186b1d7ab53158c1dff0fd570b8f623bd0dee1e733476c73a39e9d8d032c9.jpg)  
Fig. 2: System overview. Each RGB observation is processed by the same per-view expert, which associates VLM semantic proposals with normal-reference saliency as spatial support. The resulting evidence records are combined at product level without cross-view registration. Retrieval is applied after the visual verdict to populate the inspection report.

![](images/b8b590a2aa16def6634f36521c349b975e17174d1b1835d6faca815f207857e8.jpg)  
Fig. 3: Per-view semantic–saliency expert. The VLM produces class-aware semantic proposals, while the normal-reference branch provides class-agnostic saliency. Their spatial agreement is used as support while preserving the original semantic boxes.

After the visual verdict, the predicted class and inspection zone are used to retrieve the corresponding manufacturer acceptance rule and disposition. The retrieved information is attached to the supporting evidence for reporting and does not alter the visual decision.

## III. EXPERIMENTS

## A. Common Protocol

Baselines. Direct semantic localization is evaluated with GPT-4o [14], Gemini 3.6 Flash [15], Qwen2.5-VL-7B [16], and LLaVA-1.5-7B [17], with VT-ADL [18] included as an anomaly-localization reference. The complete per-view expert is compared with Faster R-CNN [3], RetinaNet [19], YOLOv8- L [20], YOLOv11-L [21], and YOLOv12-L [4]. The supervised detectors use defect-box annotations, whereas the semantic branch is prompt-driven and the reconstruction branch is adapted using normal images. Supervised baselines are trained separately on the corresponding training split of each evaluation domain, with all reported test data kept disjoint. Their results therefore provide a supervised reference rather than a like-forlike training comparison.

![](images/1d23225fa29588c5e866f7476bb2ea294166d2d1492786eca4c35d34666ff4c8.jpg)  
Fig. 4: Production-line prototype used for industrial evaluation: (1) test product, (2,7) photoelectric sensors, (3–5) RGB cameras, and (6) conveyor direction. The three cameras provide nominal $4 5 ^ { \circ } , 9 0 ^ { \circ }$ , and $1 3 5 ^ { \circ }$ observations.

TABLE I: $\mathrm { { A P } _ { 5 0 } }$ (%) on the fixed 200-image SSGD screening subset. Mean is the unweighted macro average.
<table><tr><td>Method</td><td> $\mathbf { C r a . }$ </td><td> $\mathbf { B r o . }$ </td><td>Blot</td><td>Spot</td><td> $\mathbf { s c r } .$ </td><td>Mean</td></tr><tr><td>VT-ADL</td><td>44.4</td><td>53.2</td><td>47.5</td><td>42.2</td><td>36.2</td><td>44.7</td></tr><tr><td>LLaVA-1.5-7B</td><td>25.5</td><td>32.7</td><td>16.6</td><td>31.2</td><td>29.3</td><td>27.1</td></tr><tr><td>Qwen2.5-VL-7B</td><td>43.5</td><td>53.1</td><td>38.2</td><td>51.5</td><td>43.1</td><td>45.9</td></tr><tr><td>GPT-40</td><td>61.2</td><td>50.3</td><td>50.0</td><td>51.8</td><td>49.9</td><td>52.6</td></tr><tr><td>Gemini 3.6 Flash</td><td>63.4</td><td>54.7</td><td>43.2</td><td>64.0</td><td>60.0</td><td>57.1</td></tr><tr><td>Ours 1-view</td><td>74.1</td><td>62.0</td><td>58.8</td><td>60.5</td><td>67.5</td><td>64.6</td></tr></table>

Metrics and implementation. Per-image localization is evaluated using class-wise $\mathrm { { A P } _ { 5 0 } }$ and standard COCO AP. Direct VLMs do not expose detector-style confidence scores, so their proposals are assigned a common fixed score and are used only as localization references. Proposals from our expert are ranked by $o _ { v j } ^ { * }$ . Gemini 3.6 Flash is used after the screening experiment below. The thresholds are fixed on a disjoint validation set and kept unchanged for all reported tests at $\tau _ { a } = 0 . 6 0 , \tau _ { \mathrm { a r e a } } = 2 \times 1 0 ^ { - 4 } H W$ , and $\tau _ { o } = 0 . 2 0$ . Unless stated otherwise, all localization metrics are computed before multi-view aggregation.

## B. Inspection Evaluation

SSGD protocol. SSGD is a public smartphone screen-glass defect dataset with seven annotated categories [22]. We sample a fixed five-class subset of 1,310 images from the original Crack, Broken, Blot, Spot, and Scratch categories; Light-Leakage and Broken-Membrane are not included. A fixed 200-image subset of these 1,310 images is used only for VLM screening and model selection; its results are therefore reported as a diagnostic rather than an independent benchmark. As SSGD provides individual images rather than repeated views of the same product, it evaluates only the per-view component.

Table I first identifies the semantic model used in the remaining experiments. Gemini 3.6 Flash gives the strongest direct-VLM mean $\mathsf { A P } _ { 5 0 }$ at 57.1%, while saliency-based ranking raises $\mathrm { { A P } _ { 5 0 } }$ to 64.6%. The gain is not uniform: Spot decreases from 64.0% to 60.5%, supporting the use of saliency as soft ranking evidence rather than a hard acceptance rule. The full 1,310-image protocol then compares the per-view expert with supervised detectors when no multi-view information is available. In Table II, YOLOv12-L gives the highest AP and $\mathrm { A P _ { 5 0 } } .$ , while our expert gives the highest $\mathsf { A P } _ { 7 5 }$ and AP<sub>S</sub>. The overall AP difference is small, 40.4% versus 39.8%, so the result supports competitive single-view localization rather than superiority over supervised detection.

TABLE II: COCO detection metrics (%) on the 1,310-image SSGD protocol. All rows are single-view.
<table><tr><td>Method</td><td>AP</td><td> $\mathbf { A P } _ { 5 0 }$ </td><td> $\mathbf { A P } _ { 7 5 }$ </td><td> $\mathbf { A P } _ { S }$ </td><td> $\mathbf { A } \mathbf { P } _ { M }$ </td><td> $\mathbf { A P } _ { L }$ </td></tr><tr><td>Faster R-CNN</td><td>22.6</td><td>48.3</td><td>24.5</td><td>12.4</td><td>23.8</td><td>30.5</td></tr><tr><td>RetinaNet</td><td>28.4</td><td>56.7</td><td>30.6</td><td>16.8</td><td>29.7</td><td>36.9</td></tr><tr><td>YOLOv8-L</td><td>38.2</td><td>68.4</td><td>40.7</td><td>27.4</td><td>39.4</td><td>48.2</td></tr><tr><td>YOLOv11-L</td><td>39.1</td><td>69.5</td><td>41.7</td><td>28.2</td><td>40.4</td><td>49.3</td></tr><tr><td>YOLOv12-L</td><td>40.4</td><td>70.9</td><td>43.2</td><td>29.4</td><td>41.9</td><td>50.7</td></tr><tr><td>Ours 1-view</td><td>39.8</td><td>68.7</td><td>44.0</td><td>30.3</td><td>41.2</td><td>47.8</td></tr></table>

TABLE III: Class-wise $\mathrm { { A P } _ { 5 0 } }$ (%) on 282 production-line images. Results are per image and exclude product-level aggregation.
<table><tr><td>Method</td><td> $\mathbf { C r a . }$ </td><td> $\mathbf { E x t } .$ </td><td>Mis.</td><td> $\mathbf { s c r } .$ </td><td>Mean</td></tr><tr><td>VT-ADL</td><td>32.4</td><td>52.4</td><td>44.5</td><td>40.8</td><td>42.5</td></tr><tr><td>LLaVA-1.5-7B</td><td>21.1</td><td>24.6</td><td>9.6</td><td>24.4</td><td>19.9</td></tr><tr><td>Qwen2.5-VL-7B</td><td>47.6</td><td>50.9</td><td>41.5</td><td>43.6</td><td>45.9</td></tr><tr><td>GPT-40</td><td>49.1</td><td>51.0</td><td>48.6</td><td>51.0</td><td>49.9</td></tr><tr><td>Gemini 3.6 Flash</td><td>60.1</td><td>52.8</td><td>41.8</td><td>55.6</td><td>52.6</td></tr><tr><td>Ours</td><td>70.5</td><td>61.8</td><td>55.0</td><td>62.9</td><td>62.6</td></tr></table>

Production-line protocol. The production-line prototype in Fig. 4 contains three fixed RGB cameras configured at nominal left-oblique, frontal, and right-oblique orientations of 45<sup>◦</sup>, 90<sup>◦</sup>, and 135<sup>◦</sup>, respectively. The industrial set contains 94 defective products: 32 Scratch, 28 Misalignment, 19 Extra-part, and 15 Crack, yielding 282 images at 1944 × 2592.

During acquisition, each product is held stationary while the three views are captured sequentially. After inference, the conveyor resumes and routes the product according to the inspection verdict. The system uses an Intel Core i9-14900 CPU with 32 threads and 16 GB RAM. The current rig provides three viewpoints, so we evaluate all configurations with $N \in \{ 1 , 2 , 3 \}$ . Larger N is not considered because it requires additional hardware, expert passes, and product-level context. The 282 images are first evaluated independently to isolate per-view performance before analyzing view complementarity.

Table III tests whether semantic–saliency association remains effective under the production imaging conditions. The strongest direct VLM reaches 52.6% macro $\mathrm { { A P } _ { 5 0 } }$ , while the complete expert reaches 62.6% and improves all four defect classes. The ablation below isolates whether this gain comes from evidence association rather than a change in semantic localization.

Table IV places the same per-view expert against supervised detectors on the production-line images. Our method is higher in AP, $\mathbf { A P _ { 7 5 } } , \mathbf { A P } _ { S } ,$ and $\mathsf { A P } _ { M }$ , while YOLOv12-L remains higher at $\mathrm { { A P } _ { 5 0 } }$ and $\mathsf { A P } _ { L }$ . The margins are small and do not establish overall superiority. This establishes competitive perview localization under the target imaging conditions before the multi-view study.

![](images/d131e5868a2bc7284139bd48d69e91bbfdf31adca48598beaca08f145cdb7b38.jpg)

![](images/b3685cfc140734b096bc45b0cb331577d9b3eeeaf419edd4dae009302ddedf37.jpg)  
(a) Class distributions of the five-class SSGD subset and the production-line set.

![](images/ebbac546a40dc2a24e1cc2fad9fc69551f196cc4eb64d4ea465a5b2625a168a1.jpg)  
(b) VLM screening on the 200-image SSGD subset.  
Fig. 5: Evaluation data and semantic-model screening. In (b), success denotes a correct-class prediction with IoU $\geq 0 . 5 ;$ this statistic is used only for model selection and is distinct from $\mathrm { { A P } _ { 5 0 } }$ in Table I.

TABLE IV: COCO detection metrics (%) on 282 productionline images.
<table><tr><td>Method</td><td>AP</td><td> $\mathbf { A P } _ { 5 0 }$ </td><td> $\mathbf { A P } _ { 7 5 }$ </td><td> $\mathbf { A P } _ { S }$ </td><td> $\mathbf { A } \mathbf { P } _ { M }$ </td><td> $\mathbf { A P } _ { L }$ </td></tr><tr><td>Faster R-CNN</td><td>18.9</td><td>43.4</td><td>20.1</td><td>10.1</td><td>19.8</td><td>25.7</td></tr><tr><td>RetinaNet</td><td>24.1</td><td>50.0</td><td>25.8</td><td>13.6</td><td>25.1</td><td>31.5</td></tr><tr><td>YOLOv8-L</td><td>33.6</td><td>61.8</td><td>35.8</td><td>23.6</td><td>35.1</td><td>42.9</td></tr><tr><td>YOLOv11-L</td><td>34.9</td><td>63.1</td><td>37.1</td><td>24.8</td><td>36.4</td><td>44.2</td></tr><tr><td>YOLOv12-L</td><td>36.1</td><td>64.7</td><td>38.8</td><td>26.3</td><td>37.8</td><td>45.7</td></tr><tr><td>Ours</td><td>37.2</td><td>62.6</td><td>39.8</td><td>27.6</td><td>38.4</td><td>44.9</td></tr></table>

TABLE V: Evidence ablation on 282 production-line images. Semantic coordinates are fixed in all rows.
<table><tr><td>Representation</td><td>Sem.</td><td>Heat</td><td>Boxes</td><td> $\mathbf { A P } _ { 5 0 }$ </td></tr><tr><td>Semantic only</td><td>√</td><td>x</td><td>x</td><td>52.6</td></tr><tr><td>+ saliency response</td><td>√</td><td>√</td><td>x</td><td>58.6</td></tr><tr><td>+ saliency boxes</td><td>√</td><td>√</td><td>√</td><td>62.6</td></tr></table>

## C. Ablation and Multi-View Evaluation

Evidence association. Table V isolates the proposed withinview association by holding every semantic box fixed. Semantic proposals alone obtain 52.6% $\mathrm { { A P } _ { 5 0 } }$ . Ranking the same proposals by their saliency response increases the score to 58.6%, and explicit semantic–saliency region association reaches 62.6%. Since the predicted coordinates are identical across all three settings, the 10.0-point gain is attributable to evidence-based ranking rather than geometric refinement. The additional 4.0 points from region association further suggest that explicit spatial agreement is more informative here than the mean saliency response inside a VLM-selected box.

Optical view complementarity. We evaluate all one-, two-, and three-view subsets to determine whether the available viewpoints expose complementary defect evidence. A product contributes to $R _ { \mathrm { p r o d } } @ 0 . 5$ when at least one selected view contains a correct-class prediction with IoU $\geq 0 . 5 0$ . Because cross-verification neither changes nor discards semantic proposals, this metric isolates optical view complementarity and does not evaluate the product-level VLM.

TABLE VI: $R _ { \mathrm { p r o d } } @ 0 . 5$ (%) for all available view subsets.
<table><tr><td>Views</td><td> $\mathbf { C r a . }$ </td><td> $\mathbf { E x t } .$ </td><td>Mis.</td><td> $\mathbf { s c r } .$ </td><td>All</td></tr><tr><td>Frontal</td><td>66.7</td><td>73.7</td><td>78.6</td><td>68.8</td><td>72.3</td></tr><tr><td>Left</td><td>73.3</td><td>78.9</td><td>75.0</td><td>75.0</td><td>75.5</td></tr><tr><td>Right</td><td>73.3</td><td>73.7</td><td>78.6</td><td>71.9</td><td>74.5</td></tr><tr><td> $\textrm { F } + \textrm { L }$ </td><td>80.0</td><td>84.2</td><td>82.1</td><td>81.2</td><td>81.9</td></tr><tr><td> $\textrm { F } + \textrm { R }$ </td><td>80.0</td><td>84.2</td><td>82.1</td><td>78.1</td><td>80.9</td></tr><tr><td> $\mathrm { ~ L ~ } + \mathrm { ~ R ~ }$ </td><td>86.7</td><td>84.2</td><td>85.7</td><td>84.4</td><td>85.1</td></tr><tr><td> $\mathbf { F } + \mathbf { L } + \mathbf { R }$ </td><td>86.7</td><td>89.5</td><td>89.3</td><td>87.5</td><td>88.3</td></tr></table>

Table VI shows that no single viewpoint dominates all defect classes. The best single view reaches 75.5%, the best pair 85.1%, and all three views 88.3%. The second view contributes 9.6 percentage points over the best single view, while the third contributes a further 3.2 points. The smaller increment indicates diminishing returns within the tested acquisition range, highlighting the trade-off between additional optical coverage and the added acquisition and processing load.

Fig. 6 complements the quantitative ablation. Semantic localization is preserved while saliency changes the support attached to each proposal. Spatial agreement is informative but not a correctness certificate, since reflection can affect both cues.

Fig. 7 further illustrates the cross-verification behavior under appearance conditions outside the evaluation datasets. The observed spatial agreement between semantic proposals and normal-reference responses is consistent with the evidenceassociation result in Table V, although these examples are qualitative and do not establish OOD generalization.

## IV. CONCLUSION

We presented a multi-view inspection framework for reflective smartphone cover glass. A shared per-view expert associates VLM semantic localization with normal-reference saliency through spatial support, while the resulting evidence records are combined at product level without cross-view registration. On production-line data, semantic–saliency association improves $\mathrm { { A P } _ { 5 0 } }$ from 52.6% to 62.6% with fixed semantic coordinates, while complementary optical views increase $R _ { \mathrm { p r o d } } @ 0 . 5$ from 75.5% for the best single view to 88.3% using all three views. These results support the complementary roles of within-view evidence association and multi-view optical diversity.

![](images/4276af20880a103b8c217d17cf208b7dbe8efb831eb9024c56c96acdbc5d2a48.jpg)  
(a) Frontal.

![](images/032361cec9eaf3a8e900d8aad7471a097b4cb18b896e0376242dc92e2ab08fde.jpg)  
(b) Left-oblique.

![](images/2cf3dfe37c1079c9c2246a061bae007902a9684e6554314f0d8e36503bc0d54b.jpg)

![](images/79137b85bb7cc9d52b66f7aefa589f9703da57884009ca64b944decd46b40f3c.jpg)

![](images/92137b81cb8bcdb62bdbe5f8fe0d4d434eacefe1e4e7414007c727c1b705f8f6.jpg)

(c) Right-oblique.  
![](images/484bcd3aafab19bcca2e43751492d8bfb43aa884bbb58747b47c2f7d823e49dc.jpg)  
(e) Det., left.  
(f) Det., right.

(d) Det., front.  
![](images/873a2eb3ad24321956639702c8b111f21fc5b4583b9a4ecf9a3807031f20c5bd.jpg)

![](images/623cf6c936fded8098ce95ef4046067062c775d2296d40d2006c180114761a4e.jpg)

![](images/3daec4452153b715c6bdbe85718cd0a360394a3967db2a80a02540666d02bb38.jpg)  
(h) Sal., left.  
(i) Sal., right.

(g) Sal., front.  
![](images/1dc80cf7febb2667f11ff5ab5cd2818e6bf24fb887aadc4f2493422fb07fa37e.jpg)  
(j) Evidence, front.

![](images/0b08598a7c2f07fcf9aecbe019c4b00128595c2bb5c0ef0296eb005f3a71d958.jpg)  
(k) Evidence, left.

![](images/0167ffa606e9a6a3883f3de5d47316298346d5a41e415463475e0c6461757d2e.jpg)  
(l) Evidence, right.

Fig. 6: Representative extra-part product across the three viewpoints. Semantic localization and normal-reference saliency are shown with the resulting cross-verified evidence. Spatial association changes proposal support while preserving the semantic coordinates.  
![](images/9ddddf8c0a6da1aba931a68e8d8c3f31f62ab5ad16cf5e04dd095656558577e5.jpg)  
(a) Example 1.

![](images/0af3345b7d7652c1d27105cf20b81e57f360c0a54400439df983df9a96dc5fe2.jpg)  
(b) Example 2.

![](images/f62ad66e6fb4ad59d04b8302867b5e03ca401f3f28a1025a2e17235fe5f4fa02.jpg)  
(c) Example 3.  
Fig. 7: Cross-verified predictions on three images outside the evaluation datasets, exhibiting different object appearances and imaging conditions.

The current evaluation is limited to three physical viewpoints, one product family, and a production set containing defective products only. The product-level VLM is also not evaluated independently from evidence availability across views. Future work will extend the evaluation to normal products, broader production conditions, and additional acquisition configurations, with direct assessment of product-level decision reliability.

## REFERENCES

[1] S. Turko, L. Burmak, I. Malyshev, S. Shtykov, M. Popov, P. Filimonov, A. Aspidov, and A. Shcherbinin, “Smartphone glass inspection system,” in Proceedings of the 13th International Conference on Agents and Artificial Intelligence (ICAART), Volume 2. SciTePress, 2021, pp. 655–663.

[2] H. Miao, Z. Yang, Y. Guo, W. Huang, Y. Kuang, and D. Zhang, “Dualcamera framework for detecting subtle, low-contrast defects on multilayered cover glass in display manufacturing,” Applied Optics, vol. 64, no. 23, pp. 6897–6909, 2025.

[3] S. Ren, K. He, R. Girshick, and J. Sun, “Faster R-CNN: Towards realtime object detection with region proposal networks,” in Advances in Neural Information Processing Systems, vol. 28, 2015, pp. 91–99.

[4] Y. Tian, Q. Ye, and D. Doermann, “YOLOv12: Attention-centric realtime object detectors,” in Advances in Neural Information Processing Systems, vol. 38, 2025, pp. 78 433–78 457.

[5] S. Liu, Z. Zeng, T. Ren, F. Li, H. Zhang, J. Yang, Q. Jiang, C. Li, J. Yang, H. Su, J. Zhu, and L. Zhang, “Grounding DINO: Marrying DINO with grounded pre-training for open-set object detection,” in Computer Vision – ECCV 2024. Springer Nature Switzerland, 2024, pp. 38–55.

[6] Z. Gu, B. Zhu, G. Zhu, Y. Chen, M. Tang, and J. Wang, “AnomalyGPT: Detecting industrial anomalies using large vision-language models,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 38, no. 3, 2024, pp. 1932–1940.

[7] H. Fan, C. Liu, N. E. Janvisloo, S. Bian, J. Y. H. Fuh, W. F. Lu, and B. Li, “MaViLa: Unlocking new potentials in smart manufacturing through vision language models,” Journal of Manufacturing Systems, vol. 80, pp. 258–271, 2025.

[8] J. Cheng, Y. Xu, S. Wang, T. Ma, Y. He, J. Zhang, S. Cai, J. Zhen, J. Jia, Y. Wan, Y. Xia, and Z. Zhao, “AnomalyCoT: A multi-scenario chain-ofthought dataset for multimodal large language models,” in Advances in Neural Information Processing Systems, vol. 38, 2025, pp. 77 324–77 353, datasets and Benchmarks Track.

[9] K. Roth, L. Pemula, J. Zepeda, B. Scholkopf, T. Brox, and P. Gehler,¨ “Towards total recall in industrial anomaly detection,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022, pp. 14 318–14 328.

[10] V. Zavrtanik, M. Kristan, and D. Skocaj, “DRAEM – a discriminativelyˇ trained reconstruction embedding for surface anomaly detection,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2021, pp. 8330–8339.

[11] S. Hicsonmez, A. E. R. Shabayek, and D. Aouada, “VLMDiff: Leveraging vision-language models for multi-class anomaly detection with diffusion,” in Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 2026, pp. 6309–6319.

[12] C. Wang, W. Zhu, B.-B. Gao, Z. Gan, J. Zhang, Z. Gu, S. Qian, M. Chen, and L. Ma, “Real-IAD: A real-world multi-view dataset for benchmarking versatile industrial anomaly detection,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024, pp. 22 883–22 892.

[13] M. Kruse and B. Rosenhahn, “Multi-Flow: Multi-view-enriched normalizing flows for industrial anomaly detection,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), 2025, pp. 3972–3983.

[14] OpenAI, “GPT-4o system card,” 2024.

[15] Google DeepMind, “Gemini 3.6 Flash,” 2026.

[16] S. Bai, K. Chen, X. Liu et al., “Qwen2.5-VL technical report,” arXiv preprint arXiv:2502.13923, 2025.

[17] H. Liu, C. Li, Y. Li, and Y. J. Lee, “Improved baselines with visual instruction tuning,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2024, pp. 26 286–26 296.

[18] P. Mishra, R. Verk, D. Fornasier, C. Piciarelli, and G. L. Foresti, “VT-ADL: A vision transformer network for image anomaly detection and localization,” in Proc. IEEE Int. Symp. Ind. Electron. (ISIE), 2021, pp. 1–6.

[19] T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollar, “Focal loss for´ dense object detection,” in Proc. IEEE Int. Conf. Comput. Vis. (ICCV), 2017, pp. 2980–2988.

[20] R. Varghese and M. Sambath, “YOLOv8: A novel object detection algorithm with enhanced performance and robustness,” in Proc. Int. Conf. Adv. Data Eng. Intell. Comput. Syst. (ADICS), 2024, pp. 1–6.

[21] R. Khanam and M. Hussain, “YOLOv11: An overview of the key architectural enhancements,” arXiv preprint arXiv:2410.17725, 2024.

[22] H. Han, R. Yang, S. Li, R. Hu, and X. Li, “SSGD: A smartphone screen glass dataset for defect detection,” in Proc. IEEE Int. Conf. Acoust. Speech Signal Process. (ICASSP), 2023, pp. 1–5.