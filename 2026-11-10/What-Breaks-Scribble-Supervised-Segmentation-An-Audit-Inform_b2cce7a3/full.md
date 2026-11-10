# What Breaks Scribble-Supervised Segmentation? An Audit-Informed Benchmark for Scribble Annotation Robustness

Minheng Wu<sup>1</sup>, Di Wen<sup>1</sup>, Ruiping Liu<sup>1</sup>, Junwei Zheng<sup>1</sup>, Yufan Chen<sup>1</sup>, Kailun Yang<sup>2</sup>, Kunyu Peng<sup>1\*</sup>

<sup>1</sup>Karlsruhe Institute of Technology, Kaiserstr. 12, Karlsruhe, 76131, Baden-W¨urttemberg, Germany.

<sup>2</sup>Hunan University, No. 2 Lushan South Road, Changsha, 410082, Hunan, China.

\*Corresponding author(s). E-mail(s): kunyu.peng@kit.edu;

## Abstract

Scribble-supervised segmentation is typically evaluated with a fixed annotation set, leaving robustness to distinct semantic and geometric imperfections poorly understood. Motivated by an audit of 12,031 manual PASCAL Visual Object Classes (VOC) 2012 scribbles and prior work, we introduce six controlled perturbation operators. Class corruption, spatial ofset, and directional perturbation directly operationalize audit signals. Stroke-length reduction is motivated by prior scribble studies, while random dropout and fragmentation serve as diagnostic controls. Each operator reports requested and achieved changes. We evaluate manual and automatic scribbles across two image domains, two principal backbone settings, and multiple training pipelines, with an additional MiT-B1 PFA anchor evaluation. In the five-seed Tree Energy Loss (TEL)–ResNet101 VOC study, mean intersection over union (mIoU) drops from 70.08 to 1.46 under class corruption, compared with 35.57 for spatial ofset, 52.20 for directional perturbation, and 63.24 for stroke-length reduction at their strongest settings. At the evaluated endpoints, semantic-anchor corruption produces the largest degradation, while geometric and support responses depend on the pipeline and dataset. Our benchmark provides a reproducible protocol for diagnosing annotation robustness beyond clean-set accuracy. The operator code is available at https://github.com/Wmhiram99/scribble-operator-benchmark.

Keywords: scribble supervision, semantic segmentation, annotation robustness, weak supervision, robustness benchmark

## 1 Introduction

For weakly supervised segmentation, a scribble labels only a few pixels, yet those pixels may supervise an entire object. This makes scribbles an attractive alternative to costly dense masks [1–4]. Modern pipelines propagate sparse labels through regularization [5], afinities [6], pseudo-labels [7], feature distributions [8], or prototypes [9]. These mechanisms extend sparse anchors beyond labeled pixels and can propagate the efect of an imperfect anchor.

We summarize scribble imperfections along three complementary properties: semantic identity, spatial geometry, and support/connectivity. An incorrect class changes what an anchor denotes; displacement changes where supervision enters; direction alters local geometry; and shortening, dropout, or fragmentation modify support and connectivity. We instantiate these properties through controlled perturbation operators, which transform clean scribbles into matched noisy training labels and together define the proposed robustness benchmark. Existing work largely optimizes a fixed scribble set or studies selected variations in class-aware cues [10], generation [11], and drawing style [12, 13]. General noisy-label pipelines focus mainly on class corruption, while segmentation robustness benchmarks typically perturb the input image rather than the training annotation [14–16]. Perturbing the training scribbles is necessary here because they define the sparse supervisory anchors whose semantic, geometric, and support errors are propagated during learning. A unified, audit-grounded benchmark for decomposing scribble imperfections is therefore still missing. To ground such a benchmark empirically, we first audit discrepancies between manual scribbles and dense references, then translate the observed modes into controlled perturbations with measurable achieved values.

Our research questions are: RQ1: How can observed discrepancies between scribble annotations and dense references be translated into an audit-grounded benchmark of controlled scribble perturbations? RQ2: Which semantic, geometric, and support properties of scribble annotation most strongly afect segmentation? RQ3: Which response patterns persist across scribble annotation sources, datasets, backbones, and learning pipelines?

To investigate these research questions, we introduce an audit-to-stress-test benchmark, whose overall workflow is illustrated in Figure 1. We audit 12,031 manual PASCAL VOC2012 scribbles [17]. The audit directly motivates class corruption, spatial ofset, and directional perturbation. Stroke-length reduction is informed by prior scribble research, while random dropout and fragmentation are included as diagnostic controls. These six operations are implemented as controlled perturbation operators. The first four form the cross-setting benchmark because they have direct audit evidence or provide relatively isolated control of annotation properties.

For each operator, ratios from 1.0 (the clean identity setting) to 0.1 request progressively stronger perturbations to the scribble labels. The achieved change can difer from the request because label maps are discrete and each operator has its own score definition. These labels are used to train the evaluated segmentation pipelines, whose predictions on validation RGB images are measured against dense ground truth. We analyze the resulting mIoU responses as operator-specific robustness profiles, extend the evaluation across datasets, backbones, and learning pipelines, and repeat selected settings with multiple seeds to assess the stability of the primary findings. Our contributions can be summarized as:

1. A dataset-scale audit of 12,031 PASCAL VOC2012 scribbles that directly motivates three perturbation operators and informs a broader six-operation benchmark with achieved-value diagnostics.

2. A controlled analysis of how semantic, geometric, and support properties afect segmentation robustness under operator-specific perturbations.

3. Repeated-seed and cross-setting checks showing which response patterns are retained across annotation sources, datasets, backbones, and learning pipelines.

## 2 Related Work

## Scribble-supervised segmentation.

ScribbleSup established scribbles as an eficient weak annotation by alternating graphical label propagation with network learning [4]. Subsequent methods propagate sparse evidence through random-walk propagation [18], graph- and energybased regularization [5, 19], boundary guidance and progressive inference [20, 21], and uncertainty-aware self-training [7]. Recent methods further use Transformer representations and class-conditional modeling [8, 22], tree-structured afinities [6], or prototype-based feature augmentation [9]. Despite their diferent mechanisms, these methods share a structural dependency. Labeled scribble pixels act as semantic and spatial anchors from which supervision is propagated. Existing scribble-supervised methods mainly improve learning under a given annotation set through propagation, regularization, or representation design [4, 6, 7, 9]. Rather than improving learning for a fixed scribble set, we introduce an audit-informed benchmark that measures how controlled perturbations of the sparse label change segmentation performance, yielding operator-specific robustness profiles.

![](images/a7478ac4e477b7216e6f458bdea3f94f68f4712983d26eb7863a301e8c660da3.jpg)  
Fig. 1 Noisy scribble semantic segmentation benchmark. Audit signals motivate controlled annotation perturbations, whose achieved properties are verified before matched training and robustness evaluation. The inset reports maximum mIoU drops in percentage points for TEL–ResNet101 across the nine ratios. VOC2012 uses the single-run full-ratio path; S4Pascal and S4Cityscapes are descriptive single-run paths. C, O, D, and S denote Class, Ofset, Direction, and Shrink

## Scribble sources and annotation variation.

Manual ScribbleSup [4] annotations preserve annotator-dependent variation in placement, length, and shape, but are largely confined to PASCAL VOC. Scribbles for All [12] generates reproducible scribbles from dense masks across multiple 2D datasets, whereas ScribbleKITTI [23] extends sparse supervision to 3D LiDAR scenes. Other studies examine class-aware cues [10], automatic scribble generation and augmentation [11], and variation in drawing style [13]. Interactive segmentation also studies tolerance to erroneous scribbles or clicks, but under online correction rather than ofline scribble-supervised training [24, 25]. These eforts broaden annotation sources and styles but lack an audit-grounded protocol for separately measuring semantic, spatial, directional, support, and connectivity sensitivity within a common scribble source. Our auditinformed benchmark separates semantic, geometric, support, and connectivity sensitivity rather than treating scribble quality as a single variable.

## Noisy annotations and robustness evaluation

Noisy-label learning largely targets incorrect class assignments through noise-transition modeling and loss correction [14, 26], robust learning objectives [27], and sample-selection or collaborativetraining strategies [28–30]. A broader overview of these directions is provided by Song et al. [31]. Segmentation additionally introduces spatially structured errors afecting pixels, regions, and boundaries [15, 32, 33]. Scribble noise is distinct because sparse labels are subsequently propagated. The same number of altered pixels can have diferent efects depending on what they label, where they lie, and how much coherent support remains. Corruption benchmarks systematically vary image perturbations while holding annotations fixed [16]. We transpose this principle to the training annotation. Our benchmark audits 12, 031 manual VOC scribbles and uses the supported disagreement patterns to motivate class, ofset, and direction operators. Stroke-length reduction is informed by prior work, while dropout and fragmentation serve as diagnostic controls. All operations record requested and achieved properties. This design enables reproducible robustness analysis across annotation sources, datasets, backbones, and learning pipelines.

## 3 Benchmark Methodology

## 3.1 Problem setup

Let $I _ { i }$ denote an image, $S _ { i }$ its sparse scribble annotation, and $Y _ { i }$ the corresponding dense reference.

Table 1 Screening results for manual PASCAL VOC2012 scribbles. The first five rows are image-level signals and may co-occur in one image. The direction analysis uses 17,937 eligible foreground components of 28,227 total.
<table><tr><td>Screening signal</td><td>Count</td><td>Share (%)</td></tr><tr><td>Image-level signals (n = 12,031)</td><td></td><td></td></tr><tr><td>Foreground-class disagreement</td><td>1,556</td><td>12.93</td></tr><tr><td>Foreground scribble on GT back- ground</td><td>7,272</td><td>60.44</td></tr><tr><td>Background scribble on GT </td><td>1,328</td><td>11.04</td></tr><tr><td>ground Spatial-disagreement candidate</td><td>1,831</td><td>15.22</td></tr><tr><td>Scribble class absent from GT</td><td>15</td><td>0.12</td></tr><tr><td colspan="3">Eligible foreground components  $( n = 1 7 , 9 3 7 )$ </td></tr><tr><td>Direction discrepancy (≥ 45°)</td><td>1,437</td><td>8.01</td></tr></table>

Here, clean refers to the original, unperturbed scribble map and does not imply perfect agreement with the dense reference. We measure how a segmentation pipeline responds when one property of $S _ { i }$ is altered while the image is fixed. Dense references are only used to audit the original annotations and to construct and measure controlled scribbles. They are never used as training targets. Although noisy scribble maps are also generated for validation images, validation does not consume them. The model predicts from validation RGB images, and fixed dense ground-truth (GT) masks are used to compute validation loss, mIoU, and checkpoint selection.

## 3.2 Corpus audit

We start from the manual PASCAL VOC scribbles because they retain variation introduced during annotation. The audit covers 12,031 images: 10,582 from the augmented training set and 1,449 from the VOC validation set. The scribbles and the VOC/Semantic Boundaries Dataset (SBD) references share the same image coordinates and label taxonomy [17, 34]. Automated screening compares them at pixel and connected-component levels. The signals capture class disagreement, foreground–background leakage, spatial disagreement, a scribble class that is absent from the dense reference, and axial direction discrepancy. The signals overlap, so the counts reported in Table 1 are not additive.

For each image-level signal, we inspected up to 50 flagged images. When fewer than 50 were flagged, all available images were checked. For the component-level direction signal, we inspected

Table 2 The six perturbation operators and their target properties. The primary set spans four annotation properties. Dropout and Fragmentation are diagnostic controls. We distinguish isolated from coupled operational control.
<table><tr><td>Operator</td><td>Basis</td><td>Target property</td></tr><tr><td>Primary operators</td><td></td><td></td></tr><tr><td>Class</td><td>Corpus audit</td><td>Semantic identity</td></tr><tr><td>Offset</td><td>Corpus audit</td><td>Spatial alignment</td></tr><tr><td>Direction</td><td>Corpus audit</td><td>Trajectory direction</td></tr><tr><td>Shrink</td><td>Prior work</td><td>Coherent support</td></tr><tr><td>Diagnostic operators</td><td></td><td></td></tr><tr><td>Dropout</td><td>Prior work</td><td>Distributed support</td></tr><tr><td>Fragmentation</td><td>Designed</td><td>Connectivity</td></tr></table>

50 flagged components. All 15 images flagged for “Scribble class absent from $\mathrm { G T } ^ { \mathfrak { s } }$ were checked. This was a qualitative sanity check of the screening predicates, not an estimate of screening accuracy. The gap between image-level and pixel-level frequency matters most for the leakage signal: 7,272 images contain at least one foreground scribble pixel on GT background, but such pixels are only 2.955% of the eligible scribble pixels. The audit identifies properties worth testing, not the frequency of annotation errors.

## 3.3 Intervention axes

The audit motivates semantic identity, spatial alignment, and trajectory direction. Prior work motivates coherent support [12]. These four properties define the primary cross-setting benchmark. Dropout and Fragmentation remain diagnostic controls because they couple support and connectivity rather than isolating one property. We distinguish isolated control, which preserves other measured properties where the discrete label map permits, from coupled control, whose construction necessarily changes an additional property. Class is isolated and Shrink approximately isolated. Ofset, Direction, Dropout, and Fragmentation are coupled through translation, rotation, distributed removal, or cutting. Table 2 summarizes the six operators.

## 3.4 Operator-specific control

The six interventions act on diferent properties, so a shared corruption budget would be misleading. We instead give each operator t its own measurement $Q _ { t }$ . For a clean scribble $S _ { i }$ and a requested ratio $r \in \{ 0 . 9 , 0 . 8 , \ldots , 0 . 1 \}$ , the operator produces $S _ { i , t , r } ~ = ~ T _ { t } ( S _ { i } , Y _ { i } ; r )$ , where $T _ { t }$ is the construction rule for operator t. Each diagnostic $Q _ { t } ( S ; S _ { i } , Y _ { i } )$ takes a candidate scribble map S as its first argument, with $S \mathrm { ~ = ~ } S _ { i }$ for the clean condition and $S = S _ { i , t , r }$ otherwise. We record

$$
\begin{array} { r } { \begin{array} { c } { q _ { i , t } ^ { 0 } = Q _ { t } ( S _ { i } ; S _ { i } , Y _ { i } ) , } \\ { \tau _ { i , t , r } = G _ { t } ( r ; q _ { i , t } ^ { 0 } ) , } \\ { \hat { q } _ { i , t , r } = Q _ { t } ( S _ { i , t , r } ; S _ { i } , Y _ { i } ) , } \\ { e _ { i , t , r } = | \hat { q } _ { i , t , r } - \tau _ { i , t , r } | . } \end{array} } \end{array}\tag{1}
$$

Here $q _ { i , t } ^ { 0 }$ is the clean measurement, $G _ { t }$ is the operator-specific mapping from nominal ratio to requested value, $\tau _ { i , t , r }$ is the requested value, $\hat { q } _ { i , t , r }$ is the achieved value, and $e _ { i , t , r }$ is their discrepancy. The clean condition is indexed by $r =$ 1.0. Within an operator, a lower ratio requests a stronger intervention. The same ratio has no common meaning across operators, so crossoperator comparisons describe model response, not matched severity. Unless required by datasetspecific label encoding, the operator constants are fixed across datasets, backbones, and pipelines. Each operator–ratio condition is generated independently from the same clean scribble map, rather than by composing perturbations sequentially.

## 3.5 Perturbation operators

## Class corruption.

We divide $S _ { i }$ into $K _ { i }$ valid 8-connected components $R _ { k }$ . The component agreement and class diagnostic are

$$
\begin{array} { c } { \displaystyle { a _ { k } ( S ) = \frac { \lvert \{ p \in R _ { k } : S ( p ) = Y _ { i } ( p ) \} \rvert } { \lvert R _ { k } \rvert } , } } \\ { \displaystyle { Q _ { \mathrm { c l a s s } } ( S ; S _ { i } , Y _ { i } ) = \frac { 1 } { \sum _ { k = 1 } ^ { K _ { i } } \lvert R _ { k } \rvert } \sum _ { k = 1 } ^ { K _ { i } } \lvert R _ { k } \rvert a _ { k } ( S ) } } \\ { \displaystyle { \qquad \quad \times { \bf 1 } [ a _ { k } ( S ) \geq 0 . 8 ] . } } \end{array}\tag{2}
$$

Here p indexes pixels and 1[·] is the indicator function. Components satisfying the eligibility criterion are selected to approach the requested target. The selected components are relabeled within the valid class set for the dataset, while their coordinates and support remain fixed. The achieved value is recomputed after relabeling. Since components are indivisible, the requested value may not be attained exactly.

## Spatial ofset.

Ofset changes where a scribble anchors supervision. For each foreground GT component m, let $c _ { m } ^ { Y }$ be its centroid and $\rho _ { m }$ its maximum distance from the centroid to a component pixel. Let $c _ { m } ^ { S }$ be the centroid of the scribble components that overlap this GT component, weighted by component area and a soft overlap weight. The GT-component weights are $w _ { m } = \bar { | Y _ { m } | / \sum _ { j } | Y _ { j } | }$ so they sum to one. For the background term, $c _ { \mathrm { b g } } ^ { Y }$ is the centroid of the GT background, $c _ { \mathrm { b g } } ^ { S }$ is the corresponding overlap-weighted scribble centroid, and $\rho _ { \mathrm { b g } } = \mathrm { c l i p } ( 1 . 4 \bar { \rho } , 2 0 , 9 0 )$ pixels, where $\bar { \rho }$ is the mean foreground-component radius. A bounded radial kernel maps a normalized displacement to a similarity,

$$
\phi ( x ) = \mathrm { m a x } \big ( 0 . 0 2 , [ 1 - \mathrm { c l i p } ( x , 0 , 1 ) ] ^ { 2 . 2 } \big ) ,\tag{3}
$$

and the ofset score combines matched-pair and background terms,

$$
\begin{array} { r l } & { \boldsymbol { Q } _ { \mathrm { o f f s e t } } = ( 1 - \beta ) \displaystyle \sum _ { m } \boldsymbol { w } _ { m } } \\ & { \qquad \quad \times \phi \left( \frac { \| c _ { m } ^ { S } - c _ { m } ^ { Y } \| } { 1 . 2 5 \rho _ { m } } \right) } \\ & { \qquad \quad + \beta \phi \left( \frac { d _ { \mathrm { b g } } } { 1 . 2 5 \rho _ { \mathrm { b g } } } \right) , } \end{array}\tag{4}
$$

where $d _ { \mathrm { b g } } ~ = ~ \Vert c _ { \mathrm { b g } } ^ { S } ~ - ~ c _ { \mathrm { b g } } ^ { Y } \Vert$ and $\beta$ is the background fraction of labeled scribble pixels, clipped to [0.10, 0.25]. The foreground-centroid estimate uses soft overlap weights that increase smoothly from zero to one as a scribble component’s overlap with a GT component increases from 0.05 to 0.30. Boundary-safe translation candidates preserve labels, avoid canvas clipping, and move the score toward its target. Their overlap, leakage, and boundary-contact changes are retained.

## Directional perturbation.

Direction compares the principal axes of foreground scribble and GT components of the same class. Each scribble component is assigned to the same-class GT component with maximum overlap; unmatched components are omitted. The $4 5 ^ { \circ }$ threshold is used by the audit screen, not as a generator-eligibility rule. Principal component analysis $\mathrm { ( P C A ) }$ yields unit principal axes $u _ { S , g } ( \alpha _ { g } )$ and $u _ { Y , g }$ for the rotated scribble and matched GT component; opposite signs represent the same orientation. For the $G _ { i }$ retained matched groups and their selected rotations $\alpha _ { g }$

$$
\begin{array} { r l } & { \Delta \theta _ { g } ( \alpha _ { g } ) = \operatorname { a r c c o s } \bigr ( | u _ { S , g } ( \alpha _ { g } ) ^ { \top } u _ { Y , g } | \bigr ) } \\ & { \qquad \in [ 0 , 9 0 ^ { \circ } ] , } \end{array}
$$

$$
\begin{array} { l } { { \displaystyle Q _ { \mathrm { d i r } } ( S ; S _ { i } , Y _ { i } ) = \frac { 1 } { \sum _ { g = 1 } ^ { G _ { i } } w _ { g } } \sum _ { g = 1 } ^ { G _ { i } } w _ { g } \gamma _ { g } ( \alpha _ { g } ) } } \\ { { \displaystyle ~ \times \exp \left[ - \omega _ { g } \left( \frac { \Delta \theta _ { g } ( \alpha _ { g } ) } { 2 5 ^ { \circ } } \right) ^ { 2 } \right] , } } \end{array}\tag{5}
$$

where $w _ { g }$ is the group weight, $\gamma _ { g } ( \alpha _ { g } )$ is its feasibleangle geometry factor, and $\omega _ { g }$ is its GT-anisotropy gate. $\omega _ { g }$ depends only on the matched GT component and is constant across candidate rotations. Eligible groups require scribble/GT areas of at least $1 5 / 3 0$ pixels and cover at least 0.5% of valid scribble support; at most the 12 largest are optimized. PCA uses skeleton support and falls back to raster support when fewer than five skeleton points are available. The GT anisotropy gate is zero below 1.25 rather than excluding the group. Candidate rotations span $[ - 9 0 ^ { \circ } , 9 0 ^ { \circ } ]$ in $6 ^ { \circ }$ steps, and two coordinate-descent passes approach the requested score. Rotation preserves class identity but can alter overlap or leave the matched region.

## Stroke-length reduction.

Shrink removes coherent support while preserving stroke character. Each eligible component is skeletonized, trimmed from approximate diameter endpoints according to its area allocation, and reconstructed at approximately its original width. If $R _ { k } ^ { \prime }$ is the retained part of component $R _ { k }$ , the achieved support is

$$
Q _ { \mathrm { s h r i n k } } = \frac { \sum _ { k } \lvert R _ { k } ^ { \prime } \rvert } { \sum _ { k } \lvert R _ { k } \rvert } .\tag{6}
$$

The remaining pixels retain their class labels. Shrink is an approximately isolated coherentsupport intervention, unlike Dropout, whose distributed sampling can also alter connectivity.

## Random dropout.

Random dropout reduces distributed support through seeded selection over skeleton support, rather than by independently deleting original raster pixels. The implementation takes a seed argument (default 0), skeletonizes the eligible scribble support, shufles skeleton points, and reconstructs the selected points at approximately the original stroke width. With $A ( S _ { i } )$ denoting the eligible scribble pixels and $A ( S _ { i , t , r } )$ the surviving set,

$$
Q _ { \mathrm { d r o p } } = \frac { \lvert A ( S _ { i , t , r } ) \rvert } { \lvert A ( S _ { i } ) \rvert } .\tag{7}
$$

Surviving support keeps its original labels and local coordinates. Reconstruction can split or merge raster connectivity, so Dropout is a coupled distributed-support diagnostic.

## Fragmentation.

Each scribble component is assigned to the GT region with maximal overlap, and $k _ { j } ( S )$ is the number assigned to region j. Let $\mathcal { F } _ { i }$ denote covered foreground regions and $B _ { i }$ background regions; within each set, $w _ { j }$ is proportional to the GT-region area. The implemented fluency score is

$$
\begin{array} { c } { { q _ { \mathrm { f g } } = \displaystyle \sum _ { j \in \mathcal { F } _ { i } } w _ { j } \exp [ - 0 . 4 ( k _ { j } ( S ) - 1 ) ] , } } \\ { { q _ { \mathrm { b g } } = \displaystyle \sum _ { j \in \mathcal { B } _ { i } } w _ { j } \exp [ - 0 . 8 k _ { j } ( S ) ] , } } \\ { { { } } } \\ { { Q _ { \mathrm { f r a g } } ( S ; S _ { i } , Y _ { i } ) = 0 . 7 q _ { \mathrm { f g } } + 0 . 3 q _ { \mathrm { b g } } . } } \end{array}\tag{8}
$$

Here, $q _ { \mathrm { f g } }$ penalizes multiple scribble components assigned to a covered foreground GT region and uncovered foreground regions are excluded. The term $q _ { \mathrm { b g } }$ penalizes scribble components assigned to background GT regions, with no assigned component yielding a local factor of one. The region weights are proportional to GT area. Candidate cuts are selected to approach this target. The operation therefore tests whether separating local stroke pieces changes learning beyond support reduction alone. Because every cut can also remove support, Fragmentation is a coupled diagnostic control rather than a pure connectivity intervention.

Figure 2 shows the six operators on a common image at three requested ratios, with lower r indicating a stronger intervention within a row.

Shrink, Dropout, and Fragmentation all reduce support, but respectively retain, distribute, or split it. Their comparison tests whether response depends on support amount or its organization.

## 3.6 Operator reachability and target attainment

A nominal ratio is a request to an operator, not a guarantee that every discrete label map can attain the requested score. For each dataset–operator– ratio condition, we report the source records’ mean requested score $\tau _ { t , r } ,$ mean achieved score $\hat { q } _ { t , r } .$ , and image-level target-attainment rate

$$
A _ { t , r } ( \epsilon ) = \frac { 1 } { N _ { t , r } } \sum _ { i = 1 } ^ { N _ { t , r } } \mathbf { 1 } [ e _ { i , t , r } \leq \epsilon ] ,\tag{9}
$$

Figure 3 plots each condition’s requested score, achieved mean, and attainment rate without interpolation.

Here $e _ { i , t , r } = | \hat { q } _ { i , t , r } - \tau _ { i , t , r } |$ is the absolute score error for image i, and $N _ { t , r }$ counts images with valid paired requested and achieved diagnostics. The attainment rate measures score calibration, not the fraction of generator jobs that completed successfully. The requested and achieved scores retain each generator’s original definition. Ofset is a GT-guided composite score and Direction is a trajectory-consistency score, not a displacement in pixels or a rotation angle. Shrink measures retained support. The plotted $r = 1 . 0$ row is an identity reference, not a generated perturbation. Its requested and achieved values are set to 1.0 for display, so its perfect attainment is by construction; it is shown for orientation and excluded from every perturbed-level summary. Since score scales difer, calibration is compared only within an operator and dataset.

The diagnostic records cover all 12,031 images in each of manual VOC and S4Pascal, and all 2,975 S4Cityscapes images. Across the nine perturbed levels, Direction has comparatively high average attainment (89.9% for manual VOC, 89.9% for S4Pascal, and 99.9% for S4Cityscapes), whereas Class and Ofset vary more by dataset, with Class ranging from 52.1% to 81.7% and Ofset from 72.0% to 96.0%. Shrink is particularly dataset-sensitive, with mean attainment of

89.9% for manual VOC, 27.4% for S4Pascal, and 100% for S4Cityscapes. S4Pascal has substantially lower Shrink attainment than S4Cityscapes. Thus a common nominal ratio does not imply a common achieved change, even for the same operator across annotation sources.

Table 3 shows that target attainment depends on both the operation and the annotation source. Direction attains its requested score consistently across the three datasets, while Class and Shrink show larger dataset-dependent diferences. In particular, Class attainment is lower for manual VOC (52.1%) and S4Pascal (54.2%) than for S4Cityscapes (81.7%), and Shrink attainment ranges from 27.4% for S4Pascal to 100% for S4Cityscapes. Since each operator retains its own score definition, these attainment rates are interpreted within each operator.

The lower Class attainment in the two Pascal settings is consistent with the discrete structure of this intervention. Class recolors an entire scribble component, so the achieved score changes in component-sized steps rather than continuously. We counted 8-connected components separately within each non-background semantic label in the raster annotations. The mean count per image was 2.35 (median 2) for manual VOC and 4.09 (median 3) for S4Pascal, so many images have only a few independently relabelable units. These counts provide a reproducible raster proxy for scribble units and do not imply that every component is exactly one pen stroke. With few units, attainable scores are coarse and a requested ratio can fall between attainable values. This ofers a plausible explanation for missed Class targets when the smallest possible change already recolors one whole component. It is consistent with, but does not prove, the mechanism behind the dataset diference.

Accordingly, downstream comparisons should consider achieved properties as well as nominal ratios. Table 3 reports the condition-level mean absolute diference, computed as the mean of the nine perturbed ratio-level mean-score errors, and the mean image-level attainment. The latter is computed within each ratio from valid paired records and then averaged over the nine perturbed levels. Together, these summaries show aggregate calibration error and the fraction of images meeting the tolerance. The 0.05 tolerance is an operational calibration criterion, not a test of successful job completion.

![](images/e7741f2990de10792b9d22815c2de2803f734f41783117459b3a3ffb40087a0d.jpg)  
Fig. 2 Six manual-VOC operations on image 2007 000256 [17]. The leftmost column shows the GT for visual reference; the remaining columns show generated labels at $r = 0 . 9 , 0 . 5 , 0 . 1$ . In the class row, relabeling denotes replacement of a selected whole component’s class. Per-operator generation success and achieved-value distributions are reported in the supplementary material

## 3.7 Cross-property changes within a target-operator output

To measure operational coupling, we compare each output $S _ { i , t , r }$ only with its own input scribble $S _ { i }$ . We do not substitute an independently generated output from another operator. Let $U _ { i } ~ =$ $\{ p : S _ { i } ( p ) \neq 2 5 5 \}$ and $V _ { i , t , r } = \{ p : S _ { i , t , r } ( p ) \neq $

255} denote the original and output annotated support, with valid background labels included where the dataset defines them. The changedpixel rate is 100 $| \{ p : S _ { i , t , r } ( p ) \neq S _ { i } ( p ) \} | / ( H _ { i } W _ { i } )$ 2 using all image pixels as the denominator. It counts same-location relabeling and support addition or deletion in one edit-footprint measure.

Table 3 Generator target reachability over the nine perturbed levels $r = 0 . 1 \mathrm { - } 0 . 9$ . Mean level absolute error is the mean across the nine conditions of the absolute diference between achieved and requested mean scores; mean target attainment is the mean image-level fraction with absolute score error no greater than 0.05. The $r = 1 . 0$ identity/clean reference is excluded. Scores are operator-specific and are not physical translations or angles.
<table><tr><td>Dataset</td><td>Operator</td><td>Images</td><td>Mean level abs. error</td><td>Mean within-0.05 (%)</td><td> $r = . 1$  achieved</td><td> $r = . 1$  within-0.05 (%)</td></tr><tr><td>VOC2012 (manual)</td><td>Class</td><td>12,031</td><td>0.039</td><td>52.1</td><td>0.073</td><td>54.0</td></tr><tr><td>VOC2012 (manual)</td><td>Offset</td><td>12,031</td><td>0.052</td><td>73.8</td><td>0.203</td><td>46.2</td></tr><tr><td>VOC2012 (manual)</td><td>Direction</td><td>12,031</td><td>0.007</td><td>89.9</td><td>0.062</td><td>92.9</td></tr><tr><td>VOC2012 (manual)</td><td>Shrink</td><td>12,031</td><td>0.013</td><td>89.9</td><td>0.130</td><td>87.6</td></tr><tr><td>S4Pascal (auto)</td><td>Class</td><td>12,031</td><td>0.030</td><td>54.2</td><td>0.067</td><td>55.0</td></tr><tr><td>S4Pascal (auto)</td><td>Offset</td><td>12,031</td><td>0.050</td><td>72.0</td><td>0.197</td><td>39.4</td></tr><tr><td>S4Pascal (auto)</td><td>Direction</td><td>12,031</td><td>0.007</td><td>89.9</td><td>0.051</td><td>92.0</td></tr><tr><td>S4Pascal (auto)</td><td>Shrink</td><td>12,031</td><td>0.041</td><td>27.4</td><td>0.131</td><td>69.2</td></tr><tr><td>S4Cityscapes (auto)</td><td>Class</td><td>2,975</td><td>0.014</td><td>81.7</td><td>0.093</td><td>91.6</td></tr><tr><td>S4Cityscapes (auto)</td><td>Offset</td><td>2,975</td><td>0.013</td><td>96.0</td><td>0.100</td><td>99.8</td></tr><tr><td>S4Cityscapes (auto)</td><td>Direction</td><td>2,975</td><td>0.003</td><td>99.9</td><td>0.074</td><td>99.9</td></tr><tr><td>S4Cityscapes (auto)</td><td>Shrink</td><td>2,975</td><td>0.000</td><td>100.0</td><td>0.100</td><td>100.0</td></tr></table>

We additionally report the changed-pixel footprint divided by $| U _ { i } |$ . This support-normalized quantity can exceed 100% when a transformation changes more pixels than the original support area. The full-image rate describes the fraction of the raster changed, while the support-normalized rate describes the scale of that edit relative to sparse input supervision. Neither alone measures semantic or geometric importance.

We also report $R _ { \mathrm { c l a s s } } , R _ { \mathrm { o f f s e t } }$ , and $R _ { \mathrm { d i r e c t i o n } }$ on that same output. $R _ { \mathrm { c l a s s } } ~ = ~ | \{ p ~ \in ~ U _ { i } \cap V _ { i , t , r } ~ :$ $S _ { i , t , r } ( p ) ~ = ~ S _ { i } ( p ) \} / | U _ { i } \cap V _ { i , t , r } |$ is same-position label retention on common annotated support. It is not GT semantic accuracy. $R _ { \mathrm { o f f s e t } }$ is the report-defined Ofset score on the current output divided by the same image’s original-score denominator. $R _ { \mathrm { d i r e c t i o n } }$ uses fixed original component– GT bindings for non-Direction outputs, while Direction’s own target column uses the generatornative Direction score. The S4Pascal side-efect $R _ { \mathrm { d i r e c t i o n } }$ uses pixel-weighted component aggregation. Manual VOC and S4Cityscapes use their report’s fixed-binding estimator. Cityscapes Class $R _ { \mathrm { o f f s e t } }$ is the fixed-binding geometric control, so class relabeling alone leaves that geometry score unchanged. Ratios are computed per image and then averaged equally over images with a valid denominator. The score ratios have diferent kernels, are compared with their own no-change reference, and may exceed one. Values above one are retained and are not probabilities or preservation percentages. We report original-position support overlap separately. This measures how much of the input support remains at the same coordinates and can decrease under translation or rotation even when the total support area is unchanged.

![](images/f346ef8c92c07f6ddb1753d15e8f082df21d0633e03f03ea9129f34e16450103.jpg)  
Fig. 3 Requested and achieved operator scores and image-level target attainment. Each panel is one dataset–operator pair. The left axis shows the requested score and achieved mean; the right axis shows the percentage of images with absolute score error at most 0.05. The clean/identity reference at $r = 1 . 0$ is shown for orientation. Its perfect attainment is expected by construction and it is excluded from perturbed-level summaries. Scores remain operator-specific

![](images/3b116c5ca602107bf923cd0af3def33cb89863d1b40f0574ac816288d4a7e372.jpg)  
Fig. 4 Cross-property score ratios in each target-operator output over r = 0.1–0.9. Each column identifies the target operation and each row a dataset. Only non-target properties are plotted; the dashed horizontal line marks ratio one. The vertical axis is restricted to the observed near-neutral range to magnify small deviations. Distances from the reference should therefore be interpreted using the numeric values, not as a full-scale visual efect size. Score ratios use the report-specific fixed-binding or native score definitions described in the text

Table 4 reports the strongest evaluated nominal setting, r = 0.1, and Figure 4 shows non-target score ratios over all evaluated levels. Unless stated otherwise, the ranges below are the minimum and maximum across the three dataset rows in Table 4 at r = 0.1. Class changes 2.05–2.90% of image pixels, while its geometry score ratios and originalposition support overlap remain at one. Its sameposition class-retention ratio is 0.067–0.093. Ofset changes 2.75–4.16% of pixels and its own score ratio is 0.138–0.304. Its class-retention ratio is 0.853–0.960, while its Direction ratio is 0.964– 1.001. Direction changes 1.53–2.20% of image pixels. Its target score ratio is 0.095–0.106, while class retention is 0.994–0.996 and Ofset ratios are 0.908–0.977. Shrink has original-position overlap of 0.100–0.135 and changes 2.03–2.74% of image pixels. Its class-retention ratio remains near one, while Ofset and Direction ratios vary across datasets. For example, the manual-VOC Shrink Direction ratio is 1.067. This is an increase in the reported score ratio for that dataset, not evidence that Shrink generally improves direction. Relative to original scribble support, the mean changedpixel footprint at r = 0.1 is 90.7–93.3% for Class, 106.6–158.8% for Ofset, 67.0–71.7% for Direction, and 86.5–90.0% for Shrink across the three datasets. This support-normalized quantity can exceed 100% because the edit footprint includes changed pixels outside the original support.

Taken together, the results show that the operators are not perfectly independent, but the measured of-target score changes are generally limited for Class and Direction and modest for most Ofset and Shrink comparisons. We use a descriptive band from 0.95 to 1.05 around one as a practical near-no-change range for score ratios. Across the 27 dataset-by-ratio aggregate cells, all Class-output Ofset and Direction ratios and all Direction-output Class ratios fall within this band. For Ofset outputs, 27 of 27 Direction ratios and 20 of 27 Class-retention ratios fall within it. For Shrink outputs, 27 of 27 Class-retention ratios and 26 of 27 Direction ratios fall within it.

Table 4 Cross-property measurements on the $r = 0 . 1$ target-operator outputs. Changed pixels use the full image area as denominator, and changed/support uses the original annotated support area as denominator. The latter can exceed 100% because edits may afect pixels outside original support. $R _ { \mathrm { c l a s s } }$ is same-position label retention, $R _ { \mathrm { o f f s e t } }$ is the report-defined Ofset score ratio (except Cityscapes Class, which uses the fixed-binding geometric control), and $R _ { \mathrm { d i r e c t i o n } }$ is the report-defined fixed-binding ratio for side efects or the generator-native target ratio for Direction outputs. Values are computed per image and averaged equally over images with valid denominators. The final column reports original-position support overlap, ${ \bar { | U \cap V | / | U | } }$ , where U and $V$ denote the input and output labeled support. This measures spatial overlap with the input support, not total output support area; translation can reduce it without deleting support. Ratios above one are retained and are not probabilities. Direction side-efect scores use fixed original component–GT binding with pixel-weighted component aggregation for S4Pascal. These score ratios are not interchangeable physical units.
<table><tr><td>Dataset</td><td>Target operator</td><td>Changed pixels (%)</td><td>Changed/support (%)</td><td> $R _ { \mathrm { c l a s s } }$ </td><td> $R _ { \mathrm { o f f s e t } }$ </td><td>Rdirection</td><td>Original-position overlap</td></tr><tr><td>VOC2012 (manual)  $( n = 1 2 , 0 3 1 )$ </td><td>Class</td><td>2.90</td><td>92.43</td><td>0.076</td><td>1.000</td><td>1.000</td><td>1.000</td></tr><tr><td>VOC2012 (manual) (n = 12,031)</td><td>Offset</td><td>4.16</td><td>132.20</td><td>0.897</td><td>0.283</td><td>1.001</td><td>0.337</td></tr><tr><td>VOC2012 (manual) (n = 12,031)</td><td>Direction</td><td>2.20</td><td>66.97</td><td>0.995</td><td>0.908</td><td>0.095</td><td>0.654</td></tr><tr><td>VOC2012 (manual) (n = 12,031)</td><td>Shrink</td><td>2.74</td><td>87.07</td><td>0.998</td><td>0.768</td><td>1.067</td><td>0.130</td></tr><tr><td>S4Pascal (auto)  $( n = 1 2 , 0 3 1 )$ </td><td>Class</td><td>2.41</td><td>93.28</td><td>0.067</td><td>1.000</td><td>1.000</td><td>1.000</td></tr><tr><td>S4Pascal (auto) (n = 12,031)</td><td>Offset</td><td>2.75</td><td>106.59</td><td>0.960</td><td>0.304</td><td>0.995</td><td>0.465</td></tr><tr><td>S4Pascal (auto)  $( n = 1 2 , 0 3 1 )$ </td><td>Direction</td><td>1.96</td><td>71.71</td><td>0.996</td><td>0.953</td><td>0.099</td><td>0.633</td></tr><tr><td>S4Pascal (auto) (n = 12,031)</td><td>Shrink</td><td>2.25</td><td>86.53</td><td>1.000</td><td>0.969</td><td>0.960</td><td>0.135</td></tr><tr><td>S4Cityscapes (auto)  $( n = 2 , 9 7 5 )$ </td><td>Class</td><td>2.05</td><td>90.73</td><td>0.093</td><td>1.000</td><td>1.000</td><td>1.000</td></tr><tr><td>S4Cityscapes (auto)  $( n = 2 , 9 7 5 )$ </td><td>Offset</td><td>3.57</td><td>158.77</td><td>0.853</td><td>0.138</td><td>0.964</td><td>0.174</td></tr><tr><td>S4Cityscapes (auto)  $( n = 2 , 9 7 5 )$ </td><td>Direction</td><td>1.53</td><td>71.28</td><td>0.994</td><td>0.977</td><td>0.106</td><td>0.628</td></tr><tr><td>S4Cityscapes (auto)  $( n = 2 , 9 7 5 )$ </td><td>Shrink</td><td>2.03</td><td>90.00</td><td>1.000</td><td>0.787</td><td>0.963</td><td>0.100</td></tr></table>

This supports a positive but qualified interpretation that most measured of-target score efects are small under this descriptive criterion. It does not imply that support is unchanged. Ofset can alter support location, whereas Shrink targets support extent. The reported overlap measure does not by itself distinguish relocation from totalarea reduction. Some score ratios also depart from the band. The band is not a statistical equivalence test, and these operator-specific ratios are not probabilities. Class relabeling leaves the measured geometry and support unchanged. At the strongest setting, Direction outputs retain Class at 0.994–0.996 and have Ofset ratios of 0.908– 0.977. Ofset outputs have Direction ratios of 0.964–1.001 and Class retention of 0.853–0.960. Shrink shows the clearest dataset-dependent geometric side efects while carrying out its intended support reduction. Figure 4 is the core evidence because it compares non-target properties with their no-change reference across strength levels. Table 4 complements the figure with changed-pixel and support measures at the strongest setting. The binding and score-kernel choices, component matching, and dataset-specific annotation structure limit exact cross-dataset comparison. These measurements quantify co-occurring label changes and should not be read as proof that a particular side efect independently causes a change in model performance.

## 4 Experimental Protocol

## 4.1 Dataset Structure

We use manual VOC2012 [17] scribbles as the primary reference setting and extend the study with two Scribble4All datasets [12]. S4Pascal [12] uses automatically generated scribbles on the same PASCAL VOC images and dense masks, isolating the efect of annotation source while preserving the image domain. S4Cityscapes [12, 35] uses automatically generated scribbles in a distinct urban-scene domain with diferent resolution, scene layout, class set, and scribble density. Together, these settings test whether the observed robustness patterns persist across manual versus GT-derived scribbles and under broader dataset and scene-structure shifts.

## 4.2 Evaluation Pipelines

TEL (Tree Energy Loss) treats labeled scribble pixels as sparse anchors and constructs a tree-energy regularizer through afinity filtering and tree-structured propagation [6]. It is therefore a useful primary pipeline for testing whether a controlled alteration to class identity, position, trajectory, or support changes the supervision propagated beyond the stroke.

Adapted DTS (Dynamic Teacher Switching) is a distinct teacher–student pipeline: two exponential moving average (EMA) teachers are assessed by their scribble loss, the selected teacher supplies confidence-filtered pseudo-labels, and hierarchical feature consistency regularizes the student [36]. Only the selected teacher is EMA-updated in an iteration. We use DTS as an independent pipeline-sensitivity test, not as a strict lineby-line reimplementation or a matched-protocol leaderboard comparison. Its adapted architecture, losses, update order, and inference path are diagrammed in the supplementary material.

PFA (Prototype-Based Feature Augmentation) provides a third independent pipeline: class prototypes derived from labeled pixels augment the feature supervision available beyond the sparse marks [9]. We include its VOC2012–MiT-B1 results as a compact external check on whether the observed operation response persists under a prototype-based learning mechanism.

## 4.3 Implementation Details

TEL. We use TEL with DeepLabV3+– ResNet101 [6, 37, 38]. VOC2012 and S4Pascal are trained for 40k iterations with $5 1 2 \times 5 1 2$ crops, batch size 4, and SGD $( \mathrm { l r } = 1 0 ^ { - 3 }$ , momentum 0.9, weight decay 10<sup>−4</sup>), using polynomial decay and a 2k-iteration warm-up. S4Cityscapes uses $5 1 2 \times 1 0 2 4$ crops, batch size 2, l $\mathrm { ~ r ~ } = ~ 0 . 0 0 6$ and 40k iterations. We set $\lambda ~ = ~ 0 . 4$ throughout, with $\sigma \ : = \ : 0 . 0 2$ for VOC2012 and S4Pascal and $\sigma ~ = ~ 0 . 0 0 2$ for S4Cityscapes, and use standard PyTorch BatchNorm rather than SyncBN. SegFormer-B4 replaces only the backbone under the same TEL protocol [39].

DTS. The adapted DTS uses 512 × 512 crops, batch size 2, and SGD $( \mathrm { l r } ~ = ~ 0 . 0 0 1 2 5 .$ , momentum 0.9, weight decay 10<sup>−4</sup>). VOC2012 and S4Pascal are trained for 30k iterations and evaluated every 2k. For ResNet101, the pseudo-label threshold decreases from 0.90 to 0.80 over 12k iterations. The pseudo-label and hierarchical feature consistency (HiCo) weights are 0.0125 and 0.05, with warm-up/ramp schedules of 2k/4k and 4k/4k, respectively, and EMA rates of 0.999 and 0.9999. S4Cityscapes uses 40k iterations, weights 0.0700/0.10, and ramps of 4k/8k and 8k/8k, while retaining the same threshold. SegFormer-B4 uses a 0.85→0.80 threshold ramp over 8k iterations.

PFA. PFA follows the released MiT-B1 configuration [9]: 21 classes, 320 × 320 crops, batch size 12, and Adam $( \mathrm { l r } = 6 \times 1 0 ^ { - 5 }$ , betas 0.9/0.999, weight decay 0.01) with linear decay. It uses ten 256-D prototypes, update rate 0.99, prototype activation from epoch 15, class threshold 100, and globalbranch weights (1, 0.01, 0.01). We change only the training duration, reducing it from 200 to 100 epochs.

Reproducibility and evaluation. The reported runs use single-GPU training and the primary experiments were executed on an NVIDIA A100 GPU (40 GB) with Python 3.10.20, PyTorch 2.0.1, CUDA 11.8, cuDNN 8.7, TorchVision 0.15.2, and MMEngine 0.10.7. The primary TEL–VOC2012– ResNet101 sweep uses seeds {0, 1, 2, 3, 4}; extension seed counts are listed in Tables 6 and 7. TEL checkpoints are saved every 1k iterations and summarized over the final 25%. Inference uses seven-scale flip test-time augmentation. DTS is evaluated every 2k iterations using whole-image inference, and the best validation checkpoint is retained. PFA is evaluated once per epoch after prototype warm-up; its global branch starts at epoch 16, and the checkpoint with the highest validation mIoU is selected. Validation is used only for checkpoint selection and matched clean/noisy evaluation; it does not provide training targets or modify the generated noise conditions.

Metrics. The primary metric is mIoU. TEL reports the mean over the last quarter of validation checkpoints (tail-25%). DTS and PFA report the best validation mIoU and use the corresponding best checkpoint. Let $\mathcal { R } = \{ 0 . 9 , 0 . 8 , \ldots , 0 . 1 \}$ denote the nine non-clean ratios. We summarize the clean-normalized response as

$$
\begin{array} { l } { { \displaystyle C ( \boldsymbol { r } ) = \left. \frac { \mathrm { m I o U } ( 1 . 0 ) - \mathrm { m I o U } ( \boldsymbol { r } ) } { \mathrm { m I o U } ( 1 . 0 ) } \right. , } } \\ { { \displaystyle C _ { \mathrm { a v g } } = \frac { 1 } { \lvert \mathcal { R } \rvert } \sum _ { r \in \mathcal { R } } C ( r ) } , } \\ { { \displaystyle C _ { \mathrm { e n d } } = C ( 0 . 1 ) } . } \end{array}\tag{10}
$$

Larger C(r), $C _ { \mathrm { a v g } } ,$ or $C _ { \mathrm { e n d } }$ denotes a larger deviation from that setting’s clean reference. These summaries compare response magnitude within a setting; they do not claim cross-operation severity equivalence. Repeated-seed evaluation uses a full five-seed primary sweep, then matched anchors on the dataset, backbone, and pipeline axes to test whether the primary ordering recurs beyond the reference setting. Their exact conditions and results are reported in Tables 6 and 7.

![](images/cb37cf4b9598709c9341a0b92d2862fb8cfe8776bc9bea197f6f366bbbda1895.jpg)  
Fig. 5 Five-seed Manual-VOC TEL–ResNet101 response curves for the four retained operations. Lower r denotes a stronger within-operator intervention. Shaded bands show ±1 standard deviation across five seeds

## 5 Results

## 5.1 Primary manual-VOC study

Figure 5 presents the response curves for the four retained operations in manual-VOC TEL– ResNet101. Random dropout and fragmentation remain secondary single-run controls.

## Class corruption.

Class corruption causes the largest degradation, reaching a five-seed mean of 1.46 mIoU in its strongest setting, compared with a cleanreference mIoU of 70.08. Its curve is strongly non-monotonic: performance first drops sharply, temporarily rebounds at an intermediate corruption level, and then decreases again under stronger corruption. The rebound is an observed response shape, not evidence of genuine recovery from incorrect semantic supervision.

One possible explanation is that highly unreliable scribbles cause the model to rely relatively more on image features and TEL’s structural constraints. The validation trajectories provide indirect support because the loss becomes less stable under stronger corruption, but they do not establish this mechanism or explain the rebound conclusively.

## Spatial ofset.

The spatial ofset is the strongest geometric perturbation, reaching a five-seed mean of 35.57 mIoU at its strongest setting. Its degradation is consistent with attaching class evidence to unsuitable locations. The resulting geometric side efects are retained in the reported response because they form part of the complete spatial intervention.

## Directional perturbation.

Directional perturbation produces a weaker response than class corruption and spatial ofset, with a five-seed mean of 52.20 mIoU at its strongest setting. This result indicates that trajectory changes are harmful, while remaining class and spatial information can still provide useful supervision.

## Stroke-length reduction.

Stroke-length reduction is comparatively mild, with a five-seed mean of 63.24 mIoU at its strongest setting. Correctly labeled support reduction is less damaging than changing the semantic identity or spatial anchoring of the scribble when some meaningful support remains.

## Secondary controls.

Using the single-run tail-25% mIoU summaries, we compute the maximum absolute deviation across r = 0.1–0.9 from each control’s own clean reference. The deviation is 0.74 mIoU percentage points for Random Dropout (70.52 clean; minimum 69.78 at r = 0.6) and 0.86 points for Fragmentation (70.31 clean; minimum 69.44 at r = 0.4). These are descriptive single-run results and are separate from the five-seed primary results. Their weaker initial efects and status as diagnostic operators motivate retaining them as secondary VOC2012 controls rather than repeating them in the cross-setting matrix.

At r = 0.1, the mean mIoU decreases from the clean reference of 70.08 to 1.46, 35.57, 52.20, and 63.24 for Class, Ofset, Direction, and Shrink, respectively. These correspond to drops of 68.62, 34.51, 17.88, and 6.84 mIoU percentage points. The resulting sensitivity ordering is Class, Ofset, Direction, and Shrink, from largest to smallest degradation.

![](images/443f46f6d902c02e18d950f4590acadad0ee5739f3954021adef6bf20254865f.jpg)  
Fig. 6 TEL–ResNet101 cross-dataset response paths. Each panel shows clean-normalized deviation $C ( \boldsymbol r )$ . VOC2012 shows the five-seed mean path, while S4Pascal and S4Cityscapes show descriptive single-run full paths. Lower r denotes stronger perturbation within each operator

## 5.2 Cross-setting robustness

## Cross-dataset and annotation-source robustness.

Figure 6 compares the clean-normalized response curves across datasets.

We compare TEL with ResNet101 on VOC2012, S4Pascal, and S4Cityscapes. The curves show clean-normalized response C(r) over the ratio sweep; larger values indicate greater sensitivity.

The associated $C _ { \mathrm { a v g } } / C _ { \mathrm { e n d } }$ are given in Table 5.

Comparing VOC2012 with S4Pascal isolates annotation-source efects. They share the same PASCAL images and dense ground-truth masks, but use manual and GT-derived scribbles, respectively. Comparing S4Pascal with S4Cityscapes probes broader dataset and scenestructure efects. Both use the automatic Scribble4All pipeline, while difering in domain, resolution, scene layout, class set, and scribble density.

Class corruption is similarly damaging on VOC2012 and S4Pascal. S4Pascal is generally less sensitive to spatial ofset and directional perturbation, which may reflect the greater spatial regularity of its GT-derived scribbles. Stroke-length reduction shows a small reversal between the two datasets. A rough length proxy, computed as the published labeled-area fraction divided by the published mean scribble count and an assumed width of approximately 4 pixels [12], is about 21% higher for VOC2012 than for S4Pascal. This ratio of aggregate statistics is not the mean of per-image stroke lengths and should be interpreted only as a descriptive estimate. It may partly explain the greater tolerance to stroke-length reduction.

Table 5 quantifies these diferences. With ResNet101, the $C _ { \mathrm { a v g } }$ for Class is similar on manual VOC2012 and S4Pascal, at 0.631 and 0.636, respectively, while the Direction value decreases from 0.209 to 0.159 and then to 0.032 on S4Cityscapes. For Ofset, S4Cityscapes has a lower $C _ { \mathrm { a v g } }$ of 0.215 than VOC2012 and S4Pascal, but its $C _ { \mathrm { e n d } }$ reaches 0.603, compared with 0.493 and 0.374. This contrast shows why the average response and the strongest-setting response should be read together. These are descriptive comparisons across settings and do not isolate the cause of the diferences.

S4Cityscapes is more tolerant of directional perturbation but more sensitive to spatial ofset and stroke-length reduction. This response is compatible with its urban layouts, thin structures, adjacent objects, and diferent scribble density.

Table 5 Descriptive TEL response across datasets and backbones. R and S denote ResNet-101 and SegFormer-B4. $C _ { \mathrm { a v g } } / C _ { \mathrm { e n d } }$ are the average clean-normalized response over $r \in \{ 0 . 9 , \ldots , 0 . 1 \}$ and the response at $r = 0 . 1$ . Higher values indicate larger deviation from the clean reference. The table provides numerical summaries for Figure 6. VOC2012–ResNet101 uses five-seed mean paths. The remaining full-path extensions are descriptive single-run summaries rather than statistical backbone comparisons.
<table><tr><td></td><td colspan="2">VOC2012</td><td colspan="4">S4Pascal</td><td colspan="3">S4Cityscapes</td></tr><tr><td>Operation</td><td> $\mathrm { ~ R ~ } C _ { \mathrm { a v g } } / C _ { \mathrm { e n d } }$ </td><td> $\mathrm { ~ S ~ } C _ { \mathrm { a v g } } / C _ { \mathrm { e n d } }$ </td><td> $\mathrm { ~ R ~ } C _ { \mathrm { a v g } } / C _ { \mathrm { e n d } }$ </td><td></td><td> $\mathrm { ~ S ~ } C _ { \mathrm { a v g } } / C _ { \mathrm { e n d } }$ </td><td></td><td> $\mathrm { ~ R ~ } C _ { \mathrm { a v g } } / C _ { \mathrm { e n d } }$ </td><td></td><td> $\mathrm { ~ S ~ } C _ { \mathrm { a v g } } / C _ { \mathrm { e n d } }$ </td></tr><tr><td>Class</td><td>0.631 0.979</td><td>0.632</td><td>0.967</td><td>0.636 0.976</td><td> $0 . 6 4 3 \ / \ 0 . 9 7 9$ </td><td></td><td> $0 . 5 5 5 \mathrm { ~ / ~ } 0 . 9 7 6$ </td><td></td><td>0.597 0.980</td></tr><tr><td>Offset</td><td>0.328 0.493</td><td>0.378 0.513</td><td>0.239</td><td>0.374</td><td>0.268</td><td>/0.383</td><td> $0 . 2 1 5 \mathrm { ~ \ ' ~ } 0 . 6 0 3$ </td><td>0.204</td><td>0.585</td></tr><tr><td>Direction</td><td>0.209 0.255</td><td>0.280</td><td>0.328 0.159</td><td>0.190</td><td>0.178</td><td>0.214</td><td> $0 . 0 3 2 \mathrm { ~ ' ~ } 0 . 0 6 3$ </td><td>0.023</td><td>0.050</td></tr><tr><td>Shrink</td><td>0.031 0.098</td><td>0.034</td><td>0.041 0.055</td><td>0.116</td><td>0.034</td><td>0.056</td><td>0.091 / 0.215</td><td>0.042</td><td>0.109</td></tr></table>

No dataset is uniformly more robust: the response depends jointly on scribble source, scene structure, density, and perturbation type.

## Cross-backbone robustness.

We compare TEL with DeepLabV3+–ResNet101 and SegFormer-B4 on VOC2012, S4Pascal, and S4Cityscapes under matched operation settings. Table 5 reports $C _ { \mathrm { a v g } } / C _ { \mathrm { e n d } }$ , where larger values indicate greater sensitivity and smaller values indicate greater robustness. Comparisons are therefore made within the same dataset and operation rather than across diferent noise types.

Class corruption remains the strongest perturbation for both backbones, and changing the backbone does not remove the semantic-anchor risk. SegFormer-B4 is generally more tolerant of stroke-length reduction on the automatically generated datasets, possibly reflecting its ability to exploit remaining sparse support through multi-scale token representations. ResNet101 is generally more robust to spatial ofset and directional perturbation on VOC2012 and S4Pascal, whereas SegFormer-B4 has a small advantage on S4Cityscapes. Backbone robustness is therefore conditional on both the perturbation and the dataset.

## Cross-pipeline robustness.

To examine robustness across learning mechanisms, we compare three scribble-supervision pipelines: afinity-guided TEL, teacher–student DTS, and prototype-augmented PFA. This comparison helps distinguish recurring responses from pipeline-specific ones.

Figure 7 summarizes the corresponding cleannormalized sensitivities across the matched TEL and DTS settings. Darker cells indicate larger deviations from the clean reference. Relative to

![](images/29bb2b4b4e13e4055d0e97a50ffa65c2bb6bbae520d7c1f65bcd36b0268699b1.jpg)  
Fig. 7 Pipeline sensitivity across matched TEL and DTS settings, measured by $C _ { \mathrm { a v g } }$ over the nine perturbed ratios. TEL–ResNet101 on VOC2012 uses the five-seed mean curve. All other settings, including DTS, use single-run fullratio sweeps. Darker cells indicate greater clean-normalized response

TEL, DTS is less sensitive to class corruption in several settings and to selected spatialofset and directional-perturbation conditions. Its response to stroke-length reduction and to some S4Cityscapes conditions is less favorable. Thus, the TEL–DTS contrast is setting-dependent rather than a universal pipeline ranking.

We further evaluate PFA [9] on VOC2012 with a MiT-B1 backbone and three seeds at the clean and r = 0.1 anchor points (Table 7). At the evaluated endpoints, class corruption and spatial ofset produce the two largest mIoU drops across the three pipelines, while the ordering of directional perturbation and stroke-length reduction depends on the pipeline. This anchor-based result provides an additional cross-pipeline check.

Table 6 Repeated-seed results for the three settings with four reported anchor ratios (percentages; mean ± sample SD). Pipeline, backbone, and dataset are separated explicitly. TEL entries use the tail-25% validation average, whereas DTS entries use the best validation checkpoint under its logged protocol. Repeated r=1.0 values are the shared clean reference within each setting.
<table><tr><td rowspan="2"></td><td rowspan="2">Pipeline Backbone</td><td rowspan="2">Dataset</td><td rowspan="2">Noise</td><td rowspan="2"></td><td colspan="4">Validation mIoU (mean ± sample SD)</td></tr><tr><td>r=1.0</td><td>r=0.9</td><td> $r { = } 0 . 5$ </td><td>r=0.1</td></tr><tr><td rowspan="4">TEL</td><td rowspan="4">ResNet101</td><td rowspan="4">VOC2012</td><td>Class</td><td>5555</td><td> $7 0 . 0 8 \pm 0 . 4 2$ </td><td> $6 5 . 0 4 \pm 0 . 6 1$ </td><td> $7 . 9 6 \pm 0 . 2 5$ </td><td> $1 . 4 6 \pm 0 . 1 2$ </td></tr><tr><td>Offset</td><td></td><td> $7 0 . 0 8 \pm 0 . 4 2$ </td><td> $6 4 . 0 3 \pm 0 . 3 9$ </td><td> $4 6 . 0 3 \pm 0 . 6 8$ </td><td> $3 5 . 5 7 \pm 0 . 9 8$ </td></tr><tr><td>Direction</td><td></td><td> $7 0 . 0 8 \pm 0 . 4 2$ </td><td> $5 9 . 6 6 \pm 0 . 5 9$ </td><td> $5 4 . 9 4 \pm 0 . 5 2$ </td><td> $5 2 . 2 0 \pm 0 . 4 4$ </td></tr><tr><td>Shrink</td><td></td><td> $7 0 . 0 8 \pm 0 . 4 2$ </td><td> $6 9 . 8 3 \pm 0 . 2 7$ </td><td> $6 8 . 7 7 \pm 0 . 8 2$ </td><td> $6 3 . 2 4 \pm 1 . 2 5$ </td></tr><tr><td rowspan="4">TEL</td><td rowspan="4">ResNet101</td><td rowspan="4">S4Pascal</td><td>Class</td><td></td><td> $7 5 . 7 6 \pm 0 . 1 7$ </td><td> $6 9 . 4 3 \pm 0 . 1 9$ </td><td> $1 3 . 0 7 \pm 0 . 5 4$ </td><td> $1 . 4 6 \pm 0 . 1 0$ </td></tr><tr><td>Offset</td><td>44</td><td> $7 5 . 7 6 \pm 0 . 1 7$ </td><td> $7 4 . 2 8 \pm 0 . 2 2$ </td><td> $6 0 . 1 0 \pm 0 . 4 5$ </td><td> $5 2 . 0 6 \pm 0 . 3 4$ </td></tr><tr><td>Direction</td><td>44</td><td> $7 5 . 7 6 \pm 0 . 1 7$ </td><td> $7 0 . 4 1 \pm 0 . 0 7$ </td><td> $6 7 . 0 1 \pm 0 . 2 7$ </td><td> $6 4 . 2 9 \pm 0 . 2 5$ </td></tr><tr><td>Shrink</td><td></td><td> $7 5 . 7 6 \pm 0 . 1 7$ </td><td> $7 5 . 0 8 \pm 0 . 2 4$ </td><td> $7 1 . 3 9 \pm 0 . 2 6$ </td><td> $6 8 . 7 9 \pm 0 . 4 9$ </td></tr><tr><td rowspan="4">DTS</td><td rowspan="4">ResNet101</td><td rowspan="4">VOC2012</td><td>Class</td><td></td><td> $5 5 . 7 3 \pm 0 . 9 2$ </td><td> $4 3 . 4 1 \pm 0 . 7 1$ </td><td> $9 . 9 5 \pm 0 . 6 1$ </td><td> $5 . 9 1 \pm 0 . 2 2$ </td></tr><tr><td>Offset</td><td></td><td> $5 5 . 7 3 \pm 0 . 9 2$ </td><td> $5 3 . 3 4 \pm 0 . 5 7$ </td><td> $3 9 . 5 7 \pm 0 . 4 3$ </td><td> $3 2 . 7 0 \pm 0 . 5 3$ </td></tr><tr><td>Direction</td><td>4444</td><td> $5 5 . 7 3 \pm 0 . 9 2$ </td><td> $4 9 . 7 8 \pm 0 . 2 6$ </td><td> $4 6 . 3 3 \pm 0 . 4 9$ </td><td> $4 4 . 2 9 \pm 0 . 5 9$ </td></tr><tr><td>Shrink</td><td></td><td> $5 5 . 7 3 \pm 0 . 9 2$ </td><td> $5 5 . 6 7 \pm 0 . 1 6$ </td><td> $5 2 . 5 3 \pm 0 . 5 6$ </td><td> $4 1 . 2 2 \pm 0 . 5 2$ </td></tr></table>

Table 7 Repeated-seed anchor-only results for the two settings evaluated at clean and strong noise (percentages; mean ± sample SD). TEL entries use the tail-25% validation average, whereas PFA entries use the best validation checkpoint. Only clean and endpoint conditions were evaluated; intermediate response shapes are therefore not assessed.
<table><tr><td rowspan="2">Pipeline Backbone</td><td rowspan="2"></td><td rowspan="2">Dataset</td><td rowspan="2">Noise</td><td colspan="2">Validation mIoU (mean ± sample SD) n</td></tr><tr><td> $r { = } 1 . 0$ </td><td>r=0.1</td></tr><tr><td rowspan="4">PFA</td><td rowspan="4">MiT-B1</td><td rowspan="4">Class VOC2012</td><td>33</td><td> $7 8 . 3 2 \pm 0 . 1 0$ </td><td> $4 . 4 6 \pm 0 . 0 6$ </td></tr><tr><td>Offset</td><td> $7 8 . 3 2 \pm 0 . 1 0$ </td><td> $3 9 . 9 4 \pm 0 . 0 5$ </td></tr><tr><td>Direction 3</td><td> $7 8 . 3 2 \pm 0 . 1 0$ </td><td> $\bar { 5 } 8 . \bar { 8 } 3 \pm \bar { 0 } . \bar { 0 3 }$ </td></tr><tr><td>Shrink 3</td><td> $7 8 . 3 2 \pm 0 . 1 0$ </td><td> $7 3 . 2 5 \pm 0 . 3 7$ </td></tr><tr><td rowspan="4">TEL</td><td rowspan="4">SegFormer-B4 VOC2012</td><td rowspan="4">Class</td><td>3</td><td> $7 3 . 7 5 \pm 0 . 1 2$ </td><td> $3 . 8 7 \pm 0 . 1 2$ </td></tr><tr><td>Offset 3</td><td> $7 3 . 7 5 \pm 0 . 1 2$ </td><td> $3 8 . 9 0 \pm 0 . 1 7$ </td></tr><tr><td>Direction 3</td><td> $7 3 . 7 5 \pm 0 . 1 2$ </td><td> $5 6 . 1 0 \pm 0 . 5 0$ </td></tr><tr><td>Shrink 3</td><td> $7 3 . 7 5 \pm 0 . 1 2$ </td><td> $6 5 . 9 5 \pm 1 . 1 8$ </td></tr></table>

## 5.3 Stability across settings

Repeated-seed experiments cover the main manual-VOC study and targeted extensions along the dataset, backbone, and pipeline axes (Tables 6 and 7).

The primary TEL–VOC2012–ResNet101 study uses five independent seeds over the full sweep $r \in \left\{ 1 . 0 , 0 . 9 , \ldots , 0 . 1 \right\}$ }, testing whether the response paths—including the nonmonotonic class curve—are stable across training randomness.

Additional checks use four seeds for TEL– S4Pascal–ResNet101 at $r \in \{ 0 . 9 , 0 . 5 , 0 . 1 \}$ and DTS–VOC2012–ResNet101 at r ∈ $\{ 1 . 0 , 0 . 9 , 0 . 5 , 0 . 1 \}$ . Three-seed clean $. / r = 0 . 1$ comparisons cover TEL–VOC2012–SegFormer-B4 and PFA–VOC2012–MiT-B1. These checks test recurrence of the main pattern. Complete seed records are provided in the supplementary material.

Across settings, Class and Ofset have the largest endpoint drops, while Direction and Shrink vary by setting.

## 6 Conclusion

We present an audit-informed benchmark of six scribble perturbations. Semantic corruption causes the largest loss, followed by spatial ofset. Direction and Shrink have milder, settingdependent efects. Robustness depends on sparselabel content and placement.

## Limitations and future work.

Our GT-guided operators enable controlled diagnosis of specific annotation properties. Connecting them to the frequency and causes of human errors is an important next step. The complete multiseed TEL–VOC study and targeted extensions provide a basis for broader validation. Future work will combine human-annotation studies with these diagnostics to develop adaptive, corruptionaware training across datasets, architectures, and annotation sources.

## References

[1] Papandreou, G., Chen, L.-C., Murphy, K.P., Yuille, A.L.: Weakly- and semi-supervised learning of a deep convolutional network for semantic image segmentation. In: ICCV, pp. 1742–1750 (2015). https://doi.org/10.1109/ ICCV.2015.203

[2] Dai, J., He, K., Sun, J.: Boxsup: Exploiting bounding boxes to supervise convolutional networks for semantic segmentation. In: Proceedings of the IEEE International Conference on Computer Vision (ICCV), pp. 1635–1643 (2015). https://doi.org/10.1109/ ICCV.2015.191

[3] Bearman, A., Russakovsky, O., Ferrari, V., Fei-Fei, L.: What’s the point: Semantic segmentation with point supervision. In: Computer Vision – ECCV 2016. Lecture Notes in Computer Science, vol. 9911, pp. 549–565 (2016). https://doi.org/10.1007/ 978-3-319-46478-7 34

[4] Lin, D., Dai, J., Jia, J., He, K., Sun, J.: ScribbleSup: Scribble-supervised convolutional networks for semantic segmentation. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 3159–3167 (2016). https://doi. org/10.1109/CVPR.2016.344

[5] Tang, M., Perazzi, F., Djelouah, A., Ben Ayed, I., Schroers, C., Boykov, Y.: On regularized losses for weakly-supervised cnn segmentation. In: Computer Vision – ECCV 2018. Lecture Notes in Computer Science, vol. 11220, pp. 524–540 (2018). https: //doi.org/10.1007/978-3-030-01270-0 31

[6] Liang, Z., Wang, T., Zhang, X., Sun, J., Shen, J.: Tree Energy Loss: Towards sparsely annotated semantic segmentation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 16907–16916 (2022). https://

doi.org/10.48550/arXiv.2203.10739

[7] Pan, Z., Jiang, P., Wang, Y., Tu, C., Cohn, A.G.: Scribble-supervised semantic segmentation by uncertainty reduction on neural representation and self-supervision on neural eigenspace. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 7416–7425 (2021). https://doi.org/10.1109/ICCV48922. 2021.00732

[8] Wu, L., Zhong, Z., Fang, L., He, X., Liu, Q., Ma, J., Chen, H.: Sparsely annotated semantic segmentation with adaptive gaussian mixtures. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 15454–15464 (2023). https://doi.org/10. 1109/CVPR52729.2023.01483

[9] Chan, G., Zhang, P., Dong, H., Ji, S., Chen, B.: Scribble-supervised semantic segmentation with prototype-based feature augmentation. In: Proceedings of the 41st International Conference on Machine Learning. Proceedings of Machine Learning Research, vol. 235, pp. 6155–6169 (2024)

[10] Zhang, X., Zhu, L., He, H., Jin, L., Lu, Y.: Scribble hides class: Promoting scribblebased weakly-supervised semantic segmentation with its class label. In: Proceedings of the AAAI Conference on Artificial Intelligence, vol. 38, pp. 7332–7340 (2024). https: //doi.org/10.1609/aaai.v38i7.28563

[11] Schnell, J., Wang, J., Qi, L., Hu, V.T., Tang, M.: ScribbleGen: Generative data augmentation improves scribble-supervised semantic segmentation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (2024). SyntaGen Workshop

[12] Boettcher, W., Hoyer, L., Unal, O., Lenssen, J.E., Schiele, B.: Scribbles for all: Benchmarking scribble supervised segmentation across datasets. In: Advances in Neural Information Processing Systems, vol. 37 (2024). https://doi.org/10.52202/079017-1463 Datasets and Benchmarks Track

[13] Zhang, X., Zhu, L., Zeng, S., He, H., Fu, O., Yao, Z., Xie, Z., Lu, Y.: Exploiting Inherent Class Label: Towards Robust Scribble Supervised Semantic Segmentation. arXiv preprint arXiv:2503.13895 (2025). https://doi.org/10. 48550/arXiv.2503.13895 . https://arxiv.org/ abs/2503.13895

[14] Patrini, G., Rozza, A., Menon, A.K., Nock, R., Qu, L.: Making deep neural networks robust to label noise: A loss correction approach. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (2017). https://doi.org/10.1109/CVPR.2017. 240 . https://arxiv.org/abs/1609.03683

[15] Liu, S., Liu, K., Zhu, W., Shen, Y., Fernandez-Granda, C.: Adaptive earlylearning correction for segmentation from noisy annotations. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 2606–2616 (2022). https: //doi.org/10.1109/CVPR52688.2022.00263 . https://arxiv.org/abs/2110.03740

[16] Kamann, C., Rother, C.: Benchmarking the robustness of semantic segmentation models. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 8828–8838 (2020). https: //doi.org/10.1109/CVPR42600.2020.00885 . https://arxiv.org/abs/1908.05005

[17] Everingham, M., Van Gool, L., Williams, C.K.I., Winn, J., Zisserman, A.: The PAS-CAL visual object classes (VOC) challenge. International Journal of Computer Vision 88(2), 303–338 (2010) https://doi.org/10. 1007/s11263-009-0275-4

[18] Vernaza, P., Chandraker, M.: Learning random-walk label propagation for weakly-supervised semantic segmentation. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 7158–7166 (2017). https://doi.org/10.1109/CVPR.2017.315

[19] Tang, M., Djelouah, A., Perazzi, F., Boykov, Y., Schroers, C.: Normalized cut loss for

weakly-supervised cnn segmentation. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 1818–1827 (2018). https://doi. org/10.1109/CVPR.2018.00195

[20] Wang, B., Qi, G., Tang, S., Zhang, T., Wei, Y., Li, L., Zhang, Y.: Boundary perception guidance: A scribble-supervised semantic segmentation approach. In: Proceedings of the Twenty-Eighth International Joint Conference on Artificial Intelligence, pp. 3663–3669 (2019). https://doi.org/10.24963/ijcai.2019/ 508

[21] Xu, J., Zhou, C., Cui, Z., Xu, C., Huang, Y., Shen, P., Li, S., Yang, J.: Scribblesupervised semantic segmentation inference. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 15354–15363 (2021). https:// doi.org/10.1109/ICCV48922.2021.01507

[22] Su, H., Ye, Y., Hua, W., Cheng, L., Song, M.: SASFormer: Transformers for sparsely annotated semantic segmentation. In: Proceedings of the IEEE International Conference on Multimedia and Expo (ICME), pp. 390–395 (2023). https://doi.org/10.1109/ICME55011. 2023.00074

[23] Unal, O., Dai, D., Van Gool, L.: Scribblesupervised lidar semantic segmentation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 2697–2707 (2022). https: //doi.org/10.1109/CVPR52688.2022.00272

[24] Bai, J., Wu, X.: Error-tolerant scribbles based interactive image segmentation. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 392–399 (2014). https://doi.org/10.1109/ CVPR.2014.57

[25] Zhang, C., Liu, T., Qu, X., Liu, L., Zhao, Y., Wei, Y.: NTClick: Achieving precise interactive segmentation with noise-tolerant clicks. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 8921–8930 (2025). https: //doi.org/10.1109/CVPR52734.2025.00834

[26] Sukhbaatar, S., Bruna, J., Paluri, M., Bourdev, L., Fergus, R.: Training convolutional networks with noisy labels. In: International Conference on Learning Representations (ICLR) (2015)

[27] Zhang, Z., Sabuncu, M.R.: Generalized cross entropy loss for training deep neural networks with noisy labels. In: Advances in Neural Information Processing Systems (2018). https://doi.org/10.48550/arXiv.1805.07836 . https://arxiv.org/abs/1805.07836

[28] Han, B., Yao, Q., Yu, X., Niu, G., Xu, M., Hu, W., Tsang, I., Sugiyama, M.: Co-teaching: Robust training of deep neural networks with extremely noisy labels. In: Advances in Neural Information Processing Systems (2018). https://doi.org/10.48550/arXiv.1804.06872 . https://arxiv.org/abs/1804.06872

[29] Arazo, E., Ortego, D., Albert, P., O’Connor, N.E., McGuinness, K.: Unsupervised label noise modeling and loss correction. In: Proceedings of the 36th International Conference on Machine Learning. Proceedings of Machine Learning Research, vol. 97, pp. 3124–3133 (2019). https://proceedings.mlr.press/v97/arazo19a.htm

[30] Northcutt, C.G., Jiang, L., Chuang, I.L.: Confident learning: Estimating uncertainty in dataset labels. Journal of Artificial Intelligence Research 70, 1373–1411 (2021) https: //doi.org/10.1613/jair.1.12125

[31] Song, H., Kim, M., Park, D., Shin, Y., Lee, J.-G.: Learning from noisy labels with deep neural networks: A survey. IEEE Transactions on Neural Networks and Learning Systems 34(11), 8135–8153 (2023) https: //doi.org/10.1109/TNNLS.2022.3152527 arXiv:2007.08199 [cs.LG]. First posted as arXiv:2007.08199 in 2020

[32] Rottmann, M., Reese, M.: Automated detection of label errors in semantic segmentation datasets via deep learning and uncertainty quantification. In: Proceedings of the IEEE/CVF Winter

Conference on Applications of Computer Vision, pp. 3214–3223 (2023). https: //doi.org/10.1109/WACV56688.2023.00323

[33] Li, P., Purkait, P., Ajanthan, T., Abdolshah, M., Garg, R., Husain, H., Xu, C., Gould, S., Ouyang, W., Hengel, A.: Semi-supervised semantic segmentation under label noise via diverse learning groups. In: Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 1229–1238 (2023). https://doi.org/10.1109/ICCV51070. 2023.00119

[34] Hariharan, B., Arbel´aez, P., Bourdev, L., Maji, S., Malik, J.: Semantic contours from inverse detectors. In: 2011 IEEE International Conference on Computer Vision, pp. 991–998. IEEE, Piscataway, NJ (2011). https: //doi.org/10.1109/ICCV.2011.6126343

[35] Cordts, M., Omran, M., Ramos, S., Rehfeld, T., Enzweiler, M., Benenson, R., Franke, U., Roth, S., Schiele, B.: The cityscapes dataset for semantic urban scene understanding. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 3213–3223 (2016). https: //doi.org/10.1109/CVPR.2016.350

[36] Nguyen, T.-H., Cao, H.-L., Chung, D.T., Vu, M.-A., Nguyen, T.-M., Le, M., Huynh, P.K., Bagci, U.: Scribble-Supervised Medical Image Segmentation with Dynamic Teacher Switching and Hierarchical Consistency. arXiv preprint arXiv:2601.14563 (2026). https:// arxiv.org/abs/2601.14563

[37] Chen, L.-C., Zhu, Y., Papandreou, G., Schrof, F., Adam, H.: Encoder-decoder with atrous separable convolution for semantic image segmentation. In: Proceedings of the European Conference on Computer Vision (ECCV), pp. 833–851 (2018). https://doi. org/10.1007/978-3-030-01234-2 49

[38] He, K., Zhang, X., Ren, S., Sun, J.: Deep residual learning for image recognition. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 770–778 (2016). https://doi. org/10.1109/CVPR.2016.90

[39] Xie, E., Wang, W., Yu, Z., Anandkumar, A., Alvarez, J.M., Luo, P.: SegFormer: Simple and eficient design for semantic segmentation with transformers. In: Advances in Neural Information Processing Systems, vol. 34 (2021). https://doi.org/10.48550/arXiv. 2105.15203