# Learning to Reason Like a Cytopathologist: Clinically Grounded Multiscale Evidential Learning for Cervical Cytology

Ajeet Kumar Singh<sup>1,2</sup> Nehal Mishra<sup>2</sup> Anurag Tripathi<sup>2</sup>

Santanu Chaudhury<sup>1</sup> Dorien Herremans<sup>3</sup>

<sup>1</sup>IIT Delhi, India <sup>2</sup>Info Origin INC. <sup>3</sup>Singapore University of Technology and Design

<sup>1</sup>srz258064@iitd.ac.in, santanu.choudhury@gmail.com

<sup>2</sup>{nehal.mishra, anurag.tripathi}@infoorigin.com <sup>3</sup>dorien herremans@sutd.edu.sg

## Abstract

Cervical cytology presents a fundamentally ambiguous diagnostic landscape, where equivocal categories such as Atypical Squamous Cells of Undetermined Significance (ASC-US) occupy a morphological grey zone that is definitional. For clinical decision support, the goal extends beyond assigning a definitive label to every sample: cases must also be distinguished by whether the available evidence supports a confident decision or warrants further consideration, a distinction conventional classification models are not designed to explicitly represent. We propose a multiscale evidential framework that models cervical cytology along an ordinal disease continuum using clinically grounded soft anchors, representing progression from benign reactive changes to precancerous lesions and malignancy. Three biologically motivated spatial contexts—nuclear morphology, cytoplasmic characteristics, and neighbourhood cellular architecture—are independently encoded using a shared frozen DINOv2 backbone, preserving inter-scale disagreement that early fusion may obscure. Calibrated predictions are transformed into Dempster–Shafer belief masses and fused with reliability discounting, allowing conflicting evidence and insufficient support to remain explicitly uncommitted rather than forcing a decision. The fused belief state drives interpretable routing that separates confidently classified cases from those with irreducible evidential insufficiency, enabling clinically meaningful triage. Across two independent publicly available conventional cytology datasets, RIVA and CRIC Cervix, the framework generalises consistently, with uncertainty strongly associated with diagnostic difficulty: incorrect predictions carry 1.94×–4.13× higher uncertainty than correct ones.

Table 1. The eight TBS diagnostic categories considered in this work.
<table><tr><td rowspan=1 colspan=1>Class</td><td rowspan=1 colspan=1>Clinical position</td><td rowspan=1 colspan=1>Interpretation</td></tr><tr><td rowspan=1 colspan=1>NILM</td><td rowspan=1 colspan=1>Benign</td><td rowspan=1 colspan=1>Definitive</td></tr><tr><td rowspan=1 colspan=1>ENDO</td><td rowspan=1 colspan=1>Benign</td><td rowspan=6 colspan=1>DefinitiveDefinitiveEquivocalDefinitiveEquivocalDefinitive</td></tr><tr><td rowspan=1 colspan=1>INFL</td><td rowspan=1 colspan=1>Benign</td></tr><tr><td rowspan=1 colspan=1>ASC-US</td><td rowspan=1 colspan=1>Lesion</td></tr><tr><td rowspan=1 colspan=1>LSIL</td><td rowspan=1 colspan=1>Lesion</td></tr><tr><td rowspan=1 colspan=1>ASC-H</td><td rowspan=1 colspan=1>Lesion</td></tr><tr><td rowspan=1 colspan=1>HSIL</td><td rowspan=1 colspan=1>Lesion</td></tr><tr><td rowspan=1 colspan=1>SCC</td><td rowspan=1 colspan=1>Lesion</td><td rowspan=1 colspan=1>Definitive</td></tr></table>

## 1. Introduction

Cervical cancer is among the most prevalent yet preventable cancers worldwide, with the Pap smear serving as a primary screening modality [1]. Cytological findings are reported according to The Bethesda System (TBS), the internationally adopted standard updated in 2014, from which eight diagnostic categories are considered in this work [18, 19] (Table 1). Importantly, these categories are not equally definitive: TBS distinguishes definitive interpretations (NILM, ENDO, INFL LSIL, HSIL, SCC) from equivocal grey-zone categories (ASC-US, ASC-H). This distinction reflects the fact that some cellular morphologies cannot be reliably separated at the single-cell level.

This structure also reflects an underlying ordinal progression in clinical severity. Persistent high-risk HPV infection can drive progressive cellular changes, with TBS categories occupying different positions along this trajectory and carrying different clinical risks. In particular, the risk of CIN3+ – a surrogate measure of high-grade cervical disease and cancer risk, increases substantially across the spectrum, from near-zero for NILM to 60–100% for HSIL, with ASC-US and LSIL occupying intermediate positions [8]. Thus, the categories are clinically ordered rather than a collection of independent labels. At the same time, important category boundaries remain morphologically ambiguous: INFL and ASC-US can exhibit overlapping nuclear features, while ENDO cells are benign but can exhibit morphology that mimics HSIL [19]. These cases illustrate why a visually plausible prediction is not necessarily a definitive one.

Cytopathologists address such ambiguity by integrating evidence across spatial scales, from cellular grouping and neighbourhood context at lower magnification to nuclear chromatin and membrane detail at higher magnification [1]. For automated analysis, this creates three coupled challenges: distinguishing visually similar categories whose relationships are ordinal rather than independent [3, 26]; handling genuinely ambiguous class boundaries; and combining complementary evidence observed at different spatial scales. A classifier operating on an isolated nucleus patch therefore captures only part of the available diagnostic evidence. Moreover, conventional multiscale architectures typically fuse features before prediction, which can suppress disagreement between scales precisely when that disagreement may indicate an ambiguous case. Hard categorical predictions further provide no explicit representation of insufficient or conflicting evidence.

We propose a framework that treats these properties of cervical cytology as design constraints rather than incidental characteristics. Specifically, we introduce: (i) clinically grounded, non-uniform ordinal soft anchors that encode the NILM-to-SCC progression and provide a structured representation of class relationships; (ii) three spatially independent scale-specific MLPs operating on frozen DINOv2 features, allowing evidence from different spatial scales to remain separable and preserving their disagreement for subsequent fusion; (iii) a Dempster–Shafer fusion stage that combines calibrated scale-specific likelihoods into explicit belief masses $( m _ { \mathrm { b e n i g n } } , \ m _ { \mathrm { l e s i o n } } ,$ , and $m _ { \mathrm { u n c e r t a i n } } )$ , allowing evidentially insufficient cases to be routed for further review rather than forcing a categorical decision; and (iv) cross-dataset evaluation on RIVA [22] and CRIC Cervix [23], showing that incorrect predictions exhibit systematically greater uncertainty than correct predictions and that cases assigned high uncertainty are concentrated around the equivocal categories identified by TBS.

## 2. Related Work

Deep learning approaches to cervical cytology have achieved competitive performance on isolated-cell benchmarks [6, 7, 14, 16], but studies differ substantially in diagnostic label groupings, dataset usage, and evaluation protocols, complicating direct comparison across reported results [4, 17]. For RIVA, a recently introduced conventionalsmear benchmark, the literature is still emerging, with the 2026 ISBI challenge establishing an eight-category detection-and-classification benchmark [11, 12, 15]. A systematic review of 77 studies similarly identifies limited generalizability and insufficient representation of real-world cytological complexity as persistent challenges [28]. Multiscale architectures have been explored to incorporate complementary morphological information across spatial contexts [14, 16], however these predictions are typically used to improve classification rather than to explicitly characterize the evidence provided by each spatial view.

Uncertainty-aware and evidential learning provides a framework for representing predictions beyond a single categorical decision. Evidential deep learning models uncertainty through learned evidence [24], while Dempster– Shafer theory (DST) [5, 25] represents support for competing hypotheses together with uncommitted evidence. Recent work has extended DS-based reasoning to reliable classification [13], set-valued evidential prediction [2], class inference [27], and dynamic evidence learning [9]. These developments establish DST as a general framework for reasoning with incomplete or conflicting evidence, but its use as an explicit fusion mechanism for complementary spatial predictions in cervical cytology remains unexplored.

Our work brings these two directions together by treating the nuclear, cellular, and neighbourhood views of a cell as separate evidence sources. Rather than fusing their features or predictions into a single categorical decision, we retain their evidential states until late fusion, allowing agreement, conflict, and insufficient evidence to influence the final decision. We therefore use established DST machinery to formulate an ambiguity-aware multiscale classification and routing framework tailored to cervical cytology.

## 3. Method

The preceding sections identify three requirements for reliable cervical cytology classification: spatial contexts provide complementary morphological evidence, the Bethesda categories follow an ordinal continuum, and some cells remain equivocal even under multiscale examination. We therefore formulate the task as multiscale evidential inference, keeping predictions from different spatial views independent until evidence is explicitly combined.

As illustrated in Fig. 1, each annotated cell is represented using three nucleus-centred patches of 100, 200, and 400 pixels. The 100-pixel crop follows the cell annotation convention provided by RIVA; the 200- and 400-pixel crops were defined by us to progressively incorporate cytoplasmic and neighbourhood context. For CRIC Cervix, where cells are specified by nucleus coordinates, we apply the same multiscale cropping principle around the annotated cell. Thus, the same spatial hierarchy is maintained across both datasets.

Each crop is encoded by a shared frozen DINOv2 backbone, followed by an independent prediction head for each scale. The heads are trained using ordinal soft anchors for all eight Bethesda categories, producing a scalar lesiontransition score rather than a conventional eight-class probability vector. These scores are subsequently calibrated and interpreted within a binary benign–lesion evidential frame. Each scale therefore contributes evidence that can support either hypothesis or remain uncommitted. The three evidential sources are reliability-discounted and fused using Dempster–Shafer theory (DST), producing a belief state that is finally used to route the cell to an appropriate downstream action.

![](images/a7598f55c6ac670534febb64bcff997041b5f02897593ebc9a3d58142eda63b1.jpg)  
Figure 1. Overview of the proposed framework. Multiscale patches capturing nuclear, cytoplasmic, and neighbourhood context are processed through a shared frozen DINOv2 backbone and independent scale-specific heads. Calibrated predictions are transformed into Dempster–Shafer evidence and fused with reliability discounting to support explicit uncertainty for ambiguity-aware routing.

## 3.1. Multiscale Feature Extraction and Ordinal Prediction

For each annotated cell, we extract the three nucleuscentred patches described above. Each view is encoded independently using a shared frozen DINOv2 ViT-B/14 backbone [21]. DINOv2’s self-supervised pretraining on diverse visual data provides representations of texture, shape, and relational structure, reducing reliance on explicitly engineered cytological measurements such as segmentationdependent N/C ratio or chromatin grading. The backbone is not finetuned to reduce overfitting to dataset-specific staining and acquisition characteristics, while the subsequent scale-specific heads learn the information associated with each spatial context.

A separate MLP prediction head is trained for each scale so that the three views retain independent predictions for subsequent evidential fusion. Standard binary supervision would assign all benign categories the same target and all lesion categories another, discarding the ordered progression between clinically distinct Bethesda categories. We instead retain all eight categories during supervision by assigning each class a scalar soft anchor $y _ { c } \in [ 0 , 1 ]$ representing its ordinal position along the benign-to-malignancy continuum and keeping the equivocality of ASC-US and ASC-H intact.

Each MLP is trained with BCEWithLogitsLoss against its class-specific soft anchor, producing a continuous lesiontransition score whose value reflects the ordinal position of the TBS category rather than a hard benign–lesion label. The anchors are informed by the CIN3+ risk progression described by the ASCCP risk-based framework, but represent ordinal positions rather than raw clinical risk probabilities [8]. The binary evidential frame therefore interprets, rather than replaces, the learned ordinal prediction for final benign–lesion routing.

The non-uniform anchor spacing encodes clinically motivated proximity along the TBS continuum: benign categories occupy a compact region, while the larger transitions toward ASC-US and LSIL distinguish equivocal from definitive lesion states, and the closely spaced LSIL–ASC-H–HSIL–SCC categories reflect their related but progressively severe lesion spectrum [1]. In particular, the ASC-US anchor places this definitionally equivocal category between the benign and lesion regions without collapsing it into either, allowing the scalar output to retain graded ordinal information for subsequent evidential fusion.

Table 2. Clinically grounded, non-uniform ordinal soft anchors for the eight TBS categories. The ordering and relative spacing encode clinically motivated proximity along the benign-to-malignant continuum; anchor values are ordinal positions rather than raw clinical risk probabilities.
<table><tr><td>Class</td><td>Clinical position</td><td>CIN3+ risk</td><td> $\mathbf { y _ { c } }$ </td></tr><tr><td>NILM</td><td>Clearly benign</td><td>Negligible</td><td>0.00</td></tr><tr><td>ENDO</td><td>Benign; mimics HSIL</td><td>Negligible</td><td>0.05</td></tr><tr><td>INFL</td><td>Reactive; overlaps ASC-US</td><td>Very low</td><td>0.25</td></tr><tr><td>ASC-US</td><td>Definitionally equivocal</td><td>Low</td><td>0.60</td></tr><tr><td>LSIL</td><td>Low-grade lesion</td><td>Moderate</td><td>0.85</td></tr><tr><td>ASC-H</td><td>Cannot exclude HSIL</td><td>High</td><td>0.90</td></tr><tr><td>HSIL</td><td>High-grade lesion</td><td>Very high</td><td>0.95</td></tr><tr><td>SCC</td><td>Definite malignancy</td><td>Confirmed malignancy</td><td>1.00</td></tr></table>

## 3.2. Calibrated Predictions and Evidential Fusion

The ordinally trained MLP produces, for each spatial scale $s ,$ a scalar logit $z _ { s }$ . We first convert this output to a lesion probability:

$$
p _ { s } = \sigma ( z _ { s } ) = \frac { 1 } { 1 + \exp ( - z _ { s } ) } .\tag{1}
$$

Although the MLP was trained using the eight ordinal targets, the evidential stage operates on the binary hypotheses $\Theta = \{ B , L \}$ , where NILM, ENDO, and INFL form the benign group and ASC-US, LSIL, ASC-H, HSIL, and SCC form the lesion group. This binary frame is used only to express the routing evidence; the ordinal information has already been incorporated into the learned scalar transition score.

Because neural networks may produce poorly calibrated, systematically overconfident or underconfident probabilities, and these probabilities directly determine the amount of evidence assigned to each hypothesis, calibration is performed independently for each scale before evidence construction. We evaluate temperature scaling [10] and 5-fold cross-validated isotonic regression [20, 29], selecting the strategy for each dataset based on validation performance. A scale-specific decision threshold $t _ { s }$ is then selected on the validation set by maximizing binary macro-F1 over candidate thresholds in [0.1, 0.9]. This threshold defines the benign-lesion decision boundary used by the evidential stage.

The calibrated probability $p _ { s } ^ { \mathrm { c a l } }$ is next normalized relative to this scale-specific decision boundary. Because the three spatial views may have different operating thresholds, their raw probabilities are not directly comparable as evidence. We therefore map the decision threshold to zero and the two probability endpoints to −1 and $+ 1 \colon$

$$
p _ { s } ^ { \mathrm { n o r m } } = \left\{ \begin{array} { l l } { \displaystyle \frac { p _ { s } ^ { \mathrm { c a l } } - t _ { s } } { 1 - t _ { s } } , } & { p _ { s } ^ { \mathrm { c a l } } \geq t _ { s } , } \\ { \displaystyle \frac { p _ { s } ^ { \mathrm { c a l } } - t _ { s } } { t _ { s } } , } & { p _ { s } ^ { \mathrm { c a l } } < t _ { s } . } \end{array} \right.\tag{2}
$$

Thus, $p _ { s } ^ { \mathrm { n o r m } } = 0$ corresponds to the decision boundary, negative values indicate benign-oriented evidence, and positive values indicate lesion-oriented evidence. Its magnitude measures the prediction’s distance from the boundary on a common scale across spatial views.

We convert this normalized signed distance into an evidence strength:

$$
e _ { s } = 1 - \exp \left( - \lambda ( p _ { s } ^ { \mathrm { n o r m } } ) ^ { 2 } \right) ,\tag{3}
$$

where $\lambda$ is the evidence-sharpness parameter, controlling how rapidly evidence strength increases as the prediction moves away from the decision boundary. Predictions close to the boundary therefore provide little committed evidence, whereas predictions farther from it provide stronger evidence.

The evidence strength is converted into a basic probability assignment (BPA) over the frame of discernment Θ:

$$
\begin{array} { l l } { m _ { B } ^ { s } = e _ { s } \mathbf { 1 } [ p _ { s } ^ { \mathrm { n o r m } } \leq 0 ] , } & { m _ { L } ^ { s } = e _ { s } \mathbf { 1 } [ p _ { s } ^ { \mathrm { n o r m } } > 0 ] , } \\ { m _ { U } ^ { s } = 1 - e _ { s } . } \end{array}\tag{4}
$$

Here, $\mathbf { 1 } [ \cdot ]$ denotes the indicator function. $m _ { B } ^ { s }$ and $m _ { L } ^ { s }$ represent mass assigned to the benign and lesion hypotheses, respectively, while $m _ { U } ^ { s }$ represents uncertainty mass when the evidence is insufficient to commit to either hypothesis. Together, these three quantities form the scale-specific BPA.

The three spatial views may have different reliability when their predictions conflict. We therefore estimate a scale reliability factor $\alpha _ { s }$ from validation accuracy on samples where the scale-specific binary votes disagree, and map these accuracies to [0.40, 1.00]:

$$
\alpha _ { s } = 0 . 4 0 + 0 . 6 0 \frac { \mathrm { a c c } _ { s } ^ { \mathrm { c o n f i c t } } - \operatorname* { m i n } _ { i } \mathrm { a c c } _ { i } ^ { \mathrm { c o n f i c t } } } { \operatorname* { m a x } _ { i } \mathrm { a c c } _ { i } ^ { \mathrm { c o n f i c t } } - \operatorname* { m i n } _ { i } \mathrm { a c c } _ { i } ^ { \mathrm { c o n f i c t } } } .\tag{5}
$$

Reliability discounting reduces the committed evidence of a less reliable source and transfers the residual mass to the uncertainty state:

$$
\tilde { m } _ { B } ^ { s } = \alpha _ { s } m _ { B } ^ { s } , \quad \tilde { m } _ { L } ^ { s } = \alpha _ { s } m _ { L } ^ { s } , \quad \tilde { m } _ { U } ^ { s } = \alpha _ { s } m _ { U } ^ { s } + ( 1 - \alpha _ { s } ) .\tag{6}
$$

The reliability factors fixed during validation are held constant at test time.

Finally, the three discounted BPAs are fused sequentially using Dempster’s rule. For two sources, let $\tilde { \mathbf { m } } ^ { 1 }$ and $\tilde { \mathbf { m } } ^ { 2 }$ denote any two discounted BPAs. Their conflict mass is

$$
K = \tilde { m } _ { B } ^ { 1 } \tilde { m } _ { L } ^ { 2 } + \tilde { m } _ { L } ^ { 1 } \tilde { m } _ { B } ^ { 2 } .\tag{7}
$$

Table 3. Five-way evidence-based routing. m<sub>U</sub> denotes the fused uncertainty mass.
<table><tr><td>Category</td><td>Action</td><td>Condition</td></tr><tr><td>Confident Benign</td><td>Benign sample</td><td>All 3 votes benign,  $m \upsilon \leq 0 . 4 0$ </td></tr><tr><td>Confident Lesion</td><td>Lesion sample</td><td>All 3 votes lesion,  $m _ { U } \le 0 . 4 0$ </td></tr><tr><td>Genuine Uncertain</td><td>Further review</td><td>Split vote,  $m _ { U } > 0 . 5 0$ </td></tr><tr><td>Ambiguous (agree)</td><td>Further review</td><td>Unanimous  $\mathrm { v o t e } , m \upsilon > 0 . 4 0$ </td></tr><tr><td>Disagree (confident)</td><td>DST-resolved</td><td> $\mathrm { S p l i t } \mathrm { v o t e } , m _ { U } \leq 0 . 5 0$ </td></tr></table>

The resulting combined BPA is

$$
m _ { B } = \frac { \tilde { m } _ { B } ^ { 1 } \tilde { m } _ { B } ^ { 2 } + \tilde { m } _ { B } ^ { 1 } \tilde { m } _ { U } ^ { 2 } + \tilde { m } _ { U } ^ { 1 } \tilde { m } _ { B } ^ { 2 } } { 1 - K } ,\tag{8}
$$

$$
m _ { L } = \frac { \tilde { m } _ { L } ^ { 1 } \tilde { m } _ { L } ^ { 2 } + \tilde { m } _ { L } ^ { 1 } \tilde { m } _ { U } ^ { 2 } + \tilde { m } _ { U } ^ { 1 } \tilde { m } _ { L } ^ { 2 } } { 1 - K } ,\tag{9}
$$

$$
m _ { U } = \frac { \tilde { m } _ { U } ^ { 1 } \tilde { m } _ { U } ^ { 2 } } { 1 - K } .\tag{10}
$$

Here, K denotes the mass arising from conflicting benign and lesion evidence. For the three spatial scales, this twosource rule is applied twice: the result of combining the first two scales is subsequently combined with the discounted BPA from the third scale. Agreement between sources increases support for the corresponding hypothesis, while conflicting evidence contributes to K and insufficient evidence remains represented by m<sub>U</sub>. The resulting $( m _ { B } , m _ { L } , m _ { U } )$ forms the fused BPA used by the routing policy below.

## 3.3. Evidence-Based Routing

The final belief state $( m _ { B } , m _ { L } , m _ { U } )$ is used together with the three scale-specific votes to determine the routing action. A scale votes benign when $p _ { s } ^ { \mathrm { c a l } } \leq t _ { s }$ and lesion otherwise. The routing policy distinguishes confident agreement, high evidential insufficiency, unanimous predictions lacking sufficient evidence, and disagreements that can be resolved by the fused evidence.

Samples with $m _ { U } ~ > ~ 0 . 5 0$ form the uncertain pool, which is further distinguished by whether the three scalespecific votes agree or disagree. The routing therefore does not treat high $m _ { U }$ as a third diagnosis; rather, it identifies samples for which the available multiscale evidence does not support a confident single-branch decision. Conversely, when the scale votes disagree but $m _ { U }$ remains low, the fused belief state provides a principled resolution rather than relying on majority vote.

For CRIC Cervix, we additionally apply a datasetspecific NILM vote filter to the uncertain pool to suppress spurious lesion votes. Samples with fewer than two of three lesion votes are reassigned to the benign branch. This adjustment is not part of the core evidential formulation and is discussed in section 8 and the supplementary material.

The resulting framework is therefore a fixed evidential formulation rather than a collection of independently tuned components. The spatial scales and ordinal anchors establish the clinical formulation, with patch sizes dictated by the available annotation conventions, while evidential fusion and routing provide a consistent procedure for combining and interpreting the resulting predictions. Only calibration strategy and scale-specific decision thresholds are selected on validation data to account for dataset-specific differences in acquisition and class distribution. Once selected, all parameters remain fixed for test evaluation.

## 4. Experiments

## 4.1. Datasets

RIVA. RIVA [22] is a large-scale conventional Pap smear dataset covering all eight TBS categories, with cell annotations reviewed by up to four cytopathologists. We use train/validation/test split of 10,613, 2,654, and 2,682 annotated cells, respectively. RIVA includes both INFL and ASC-US, providing a challenging benign–lesion boundary, as well as ENDO, a benign class whose morphology can resemble HSIL.

CRIC Cervix. CRIC Cervix [23] contains six TBS categories—NILM, ASC-US, LSIL, ASC-H, HSIL, and SCC—with 8,073, 1,730, and 1,731 annotated cells in the train, validation, and test splits, respectively. Different scanning and staining conditions and a distinct demographic distribution from RIVA provide a cross-dataset test of generalization. NILM constitutes 58.8% of the annotated cells, while ASC-US has only 91 validation and 91 test samples; its results should therefore be interpreted with appropriate caution.

Augmentations. Despite heavy class imbalance, no artificial augmentation or class rebalancing was applied. The framework is therefore evaluated under the original screening distributions, where benign cells dominate and ambiguous lesion categories are naturally less frequent. This preserves the native distribution encountered by an uncertaintyaware routing system while avoiding synthetic modification of the fine-grained nuclear and cytoplasmic morphology important for boundary-class discrimination.

## 4.2. Metrics

We use macro-F1 as the primary metric, together with lesion-F1 for the binary benign–lesion task and overall routing accuracy. ROC-AUC and PR-AUC are also reported for the DST-fused output.

To assess whether uncertainty distinguishes reliable from unreliable predictions, we report the uncertainty ratio (UR):

$$
\mathrm { U R } = \frac { \mu _ { m _ { U } } ( \mathrm { i n c o r r e c t } ) } { \mu _ { m _ { U } } ( \mathrm { c o r r e c t } ) } .\tag{11}
$$

Table 4. Overall benign–lesion routing performance using fused DST evidence. PR-AUC is reported due to the naturally imbalanced screening distribution.
<table><tr><td>Metric</td><td>RIVA CRIC</td><td>Metric</td><td>RIVA</td><td>CRIC</td></tr><tr><td>Accuracy</td><td>0.750 0.927</td><td>Benign-F1</td><td>0.808</td><td>0.939</td></tr><tr><td>Macro-F1</td><td>0.726 0.924</td><td>Lesion-F1</td><td>0.644</td><td>0.909</td></tr><tr><td>Macro-Precision Macro-Recall</td><td>0.721 0.927 0.736 0.921</td><td>ROC-AUC PR-AUC</td><td>0.814 0.709</td><td>0.971 0.969</td></tr></table>

Thus, $\mathrm { U R } > 1$ indicates greater mean uncertainty mass for incorrect predictions. We further report performance for high-confidence samples $( m \upsilon ~ \leq ~ 0 . 4 0 )$ and the uncertain pool $( m _ { U } > 0 . 5 0 )$ . Silent wrong (SW) denotes the fraction of samples with $m _ { U } \le 0 . 4 0$ routed to the incorrect binary branch.

## 4.3. Implementation Details

All scale-specific MLPs are trained independently using AdamW with learning rate $1 0 ^ { - 3 }$ , weight decay $1 0 ^ { - 4 }$ , and CosineAnnealingLR over 20 epochs. Features are extracted using a frozen DINOv2 ViT-B/14 backbone [21]. The best checkpoint is selected using validation macro-F1 computed from the ordinally supervised predictions.

## 5. Results

## 5.1. Classification and Fusion Results

Table 4 summarizes the benign–lesion routing performance of the complete framework. Performance is consistently stronger on CRIC Cervix, while RIVA remains the more challenging eight-class setting. We next isolate the contribution of the fusion strategy.

Table 5 directly compares the individual spatial views and alternative late-fusion strategies using the same calibrated scale-specific predictions, with the image embeddings and scale-specific MLPs held fixed across all fusion methods. Thus, differences in fused performance isolate the effect of the fusion strategy rather than the representation or preceding classifiers. Majority voting combines the discrete scale decisions, probability averaging averages their lesion probabilities, inverse-entropy fusion assigns greater weight to predictions with lower entropy, and the learned MLP provides a trainable fusion baseline. DST provides the strongest fusion performance on both datasets, with the clearest separation appearing on samples where the spatial views disagree.

The conflict results are particularly relevant to the proposed formulation: DST retains its advantage when spatial views provide inconsistent evidence or insufficient evidence for a confident decision, rather than only when their predictions agree.

As discussed in Section 2, existing results arise from heterogeneous experimental settings; a detailed comparative study is provided in the supplementary material. Hence, their results provide valuable context but not a controlled baseline for our setting. Our framework targets a different objective: uncertainty-aware benign–lesion triage with an explicit state for insufficient evidence, rather than direct multiclass classification. A controlled comparison requires fixing the representation, predictions, and task while varying only the fusion strategy, precisely what Table 5 provides by evaluating the same calibrated scale-specific predictions under our DST formulation against established late-fusion alternatives.

## 5.2. Uncertainty and Clinical Ambiguity

Table 6 shows a clear class-dependent pattern in the fused uncertainty. Endpoint categories such as NILM and SCC are routed more reliably and with lower uncertainty, whereas categories near the benign–lesion boundary are less consistently resolved.

The uncertainty ratio exceeds one overall on both datasets, indicating that incorrect predictions tend to carry more uncertainty. ASC-US is the notable exception: correctly routed samples can remain more uncertain than incorrect ones, showing that the uncertainty measure captures genuine ambiguity rather than serving only as an error score. ENDO remains the main failure mode on RIVA, with poor routing accuracy and a high silent-wrong rate. We have addressed this further in 8

Figure 2 shows the distribution of fused lesion-transition scores for each Bethesda category. NILM and SCC remain concentrated near opposite ends of the score range, while boundary categories such as ASC-US and LSIL occupy broader transitional regions. This ordering is preserved across both datasets, showing that the ordinal information learned before fusion remains encoded in the fused evidence, despite the final routing framework operating on benign–lesion evidence. Representative multiscale routing cases are shown in Fig. 3.

## 5.3. Ablation, Sensitivity, and Cross-Dataset Stability

Table 7 separates the formulation into choices that are shared across datasets, operating parameters that are validation-adapted, and the single dataset-specific correction applied to CRIC Cervix.

The core evidential formulation transfers unchanged across datasets, while calibration and scale-specific thresholds are adapted to each dataset’s validation distribution. The CRIC NILM filter is an additional dataset-specific refinement used to obtain a purer Genuine Uncertain pool, rather than a requirement for the framework to operate.

Sensitivity analysis shows stable behaviour across the main numerical choices. The ASC-US anchor changes the degree of lesion commitment near the boundary, with 0.60 retaining stable behaviour across both datasets. The reliability range primarily changes the relative influence of the spatial sources, while λ and $m _ { U } ^ { \mathrm { h i g h } }$ mainly affect the size and composition of the uncertain pool.

Table 5. Comparison of individual scales and late-fusion strategies using the same calibrated scale-specific predictions. Accuracy and Macro-F1 are reported separately. Conflict results are computed only on samples for which the scale predictions disagree.
<table><tr><td rowspan="3" colspan="2"></td><td colspan="4">RIVA</td><td colspan="4">CRIC Cervix</td></tr><tr><td colspan="2">Overall</td><td colspan="2">Conflict</td><td colspan="2">Overall</td><td colspan="2">Conflict</td></tr><tr><td></td><td>Acc. MF1</td><td>Acc.</td><td>MF1</td><td>Acc.</td><td>MF1</td><td>Acc.</td><td>MF1</td></tr><tr><td rowspan="4">scaes</td><td>100px (nucleus)</td><td>.7293</td><td>.7112</td><td>.5476</td><td>.5467</td><td>.9012</td><td>.8982</td><td>.5510</td><td>.5510</td></tr><tr><td>200px (cytoplasm)</td><td>.7163</td><td>.6947</td><td>.5153</td><td>.5120</td><td>.9185</td><td>.9159</td><td>.6531</td><td>.6529</td></tr><tr><td>400px (neighbourhood)</td><td>.7260</td><td>.6895</td><td>.5393</td><td>.5022</td><td>.9006</td><td>.8974</td><td>.5476</td><td>.5475</td></tr><tr><td>Majority vote</td><td>.7210</td><td>.7071</td><td>.5822</td><td>.5815</td><td>.9053</td><td>.9031</td><td>.7109</td><td>.7125</td></tr><tr><td rowspan="4">uson</td><td>Probability averaging</td><td>.7295</td><td>.7147</td><td>.5894</td><td>.5850</td><td>.9073</td><td>.9065</td><td>.7139</td><td>.7107</td></tr><tr><td>Inverse-entropy fusion</td><td>.7449</td><td>.7202</td><td>.5939</td><td>.5932</td><td>.9110</td><td>.9109</td><td>.7381</td><td>.7381</td></tr><tr><td>Learned MLP fusion</td><td>.7409</td><td>.7173</td><td>.5763</td><td>.5763</td><td>.9141</td><td>.9122</td><td>.7399</td><td>.7385</td></tr><tr><td>DST fusion (ours)</td><td>.7502</td><td>.7259</td><td>.6178</td><td>.6150</td><td>.9266</td><td>.9238</td><td>.7517</td><td>.7515</td></tr></table>

![](images/0c02203fdde0ea5bd1efb3f9cc8bac47536d7cef11ec91f24522fba440825338.jpg)  
(a) RIVA

![](images/d93ba6f005d7cfcc8e8708bebd09e74851417deeb79525f01511c9cbe0cf15cc.jpg)  
(b) CRIC Cervix  
Figure 2. Per-class distributions of fused lesion-transition scores across the RIVA and CRIC Cervix test sets. The distributions retain the ordinal progression from benign to malignant categories after multiscale evidence fusion.

Finally, changing the order of sequential DST combination produces no differences on either dataset, consistent with the commutative and associative properties of Dempster’s rule of combination. Additional ablations and sensitivity analyses of the evidential formulation, including the effects of fusion parameters and design choices, are provided in the supplementary material.

## 6. Discussion

Uncertainty as a first-class output. The uncertaintystratified results (Table 8) show that $m _ { U }$ is predictive rather than post-hoc: high-confidence samples outperform the overall average, while uncertain samples underperform it, concentrating uncertainty on diagnostically difficult cases. This identifies the uncertain pool as the natural target for further review.

Biology-driven uncertainty structure. Per-class $\mu { } U$ follows the ambiguity gradient of the Bethesda System, arising from the interaction of the soft ordinal targets and DST fusion. The ASCUS $\mu _ { U }$ -inversion is particularly informative: correctly predicted ASCUS samples remain more uncertain than confidently misclassified ones, indicating that correct recognition can retain genuine boundary ambiguity rather than confident assignment to either benign or lesion states.

Figure 3. Representative multiscale evidential routing cases from the RIVA and CRIC Cervix test sets. Each row shows an actual test sample, its scale-specific lesion probabilities and votes, the DST-fused evidence, and the resulting routing action. The examples collec tively illustrate all five routing categories: confident benign, confident lesion, genuine uncertain, ambiguous agreement, and disagreement resolved by DST.
<table><tr><td>Dataset</td><td>True class / outcome</td><td>100px</td><td>200px</td><td>400px</td><td>Scale votes</td><td>DST fused evidence</td><td>Routing</td></tr><tr><td rowspan="6">RIVA</td><td rowspan="3">LSIL: Lesion √</td><td></td><td></td><td></td><td>3 lesion 0 benign</td><td>0.000</td><td rowspan="2">Confident lesion → lesion</td></tr><tr><td>0.945</td><td>0.967</td><td></td><td></td><td>0.997 0.003</td></tr><tr><td></td><td></td><td>0.998</td><td>1 lesion</td><td>B 0.314</td><td rowspan="2">Genuine uncertain → further review</td></tr><tr><td rowspan="3">INFL: Benign √</td><td>0.233</td><td>0.605</td><td></td><td>2 benign</td><td>0.148 0.538</td></tr><tr><td rowspan="2"></td><td></td><td>0.357</td><td></td><td>U</td><td rowspan="2">Disagreement → DST resolves to lesion</td></tr><tr><td></td><td></td><td>2 lesion 1 benign</td><td>BLU 0.154 0.370</td></tr><tr><td rowspan="6">CRIC Cervix</td><td rowspan="2">NILM: Benign √</td><td>0.861</td><td>0.584</td><td>0.343</td><td></td><td>0.476</td><td rowspan="2">Confident benign → benign</td></tr><tr><td></td><td></td><td></td><td>0 lesion 3 benign</td><td>BLU 0.987 0.000</td></tr><tr><td rowspan="2">NILM: Benign√</td><td>0.000</td><td>0.008</td><td>0.016</td><td></td><td>0.013</td><td rowspan="2">Ambiguous (agree) → further review</td></tr><tr><td></td><td></td><td>1</td><td>0 lesion</td><td>0.402 0.000</td></tr><tr><td rowspan="2"></td><td>0.395</td><td>0.455</td><td>0.273</td><td>3 benign</td><td>BLU 0.598</td><td rowspan="2"></td></tr><tr><td>ASC-US: Lesion√</td><td></td><td></td><td>2 lesion</td><td>BLU 0.082</td></tr><tr><td rowspan="2"></td><td rowspan="2"></td><td>0.850</td><td></td><td>0.610</td><td>1 benign</td><td>0.350 0.568</td><td rowspan="2">Genuine uncertain → further review</td></tr><tr><td></td><td>0.455</td><td></td><td></td><td></td></tr></table>

B, L, and U denote benign, lesion, and insufficient evidence, respectively. Bar length denotes probability/evidence magnitude; black vertical bar indicate the scale-specific decision thresholds. All probabilities and fused masses are reported from the corresponding test-set samples.

Table 6. Per-class routing behaviour across datasets. Acc.: routing accuracy; SW: silent wrong $( \% ) ; \mu v \colon$ mean uncertainty mass; UR: uncertainty ratio between incorrect and correct predictions.
<table><tr><td colspan="5">RIVA</td><td colspan="4">CRIC Cervix</td></tr><tr><td>Class</td><td>Acc.</td><td>SW↓</td><td>μU</td><td>UR↑</td><td>Acc.</td><td>SW↓</td><td>μU</td><td>UR↑</td></tr><tr><td>NILM</td><td>.913</td><td>1.4</td><td>.091</td><td>4.13</td><td>.943</td><td>1.7</td><td>.074</td><td>5.11</td></tr><tr><td>ENDO</td><td>.222</td><td>46.7</td><td>.101</td><td>.42</td><td></td><td>一</td><td>一</td><td>一</td></tr><tr><td>INFL</td><td>.742</td><td>5.5</td><td>.218</td><td>1.69</td><td></td><td>一</td><td></td><td>一</td></tr><tr><td>ASC-US</td><td>.405</td><td>40.5</td><td>.165</td><td>.57</td><td>.637</td><td>9.9</td><td>.215</td><td>1.66</td></tr><tr><td>LSIL</td><td>.519</td><td>12.6</td><td>.278</td><td>1.10</td><td>.863</td><td>4.4</td><td>.138</td><td>1.05</td></tr><tr><td>HSIL</td><td>.739</td><td>3.7</td><td>.176</td><td>2.27</td><td>.992</td><td>0.0</td><td>.022</td><td>3.14</td></tr><tr><td>ASC-H</td><td>.679</td><td>7.1</td><td>.276</td><td>1.13</td><td>.978</td><td>0.7</td><td>.039</td><td>2.62</td></tr><tr><td>SCC</td><td>.959</td><td>0.9</td><td>.047</td><td>2.48</td><td>1.000</td><td>0.0</td><td>.023</td><td>8.59</td></tr><tr><td>Overall</td><td>.742</td><td>5.8</td><td>.166</td><td>1.94</td><td>.943</td><td>1.7</td><td>.074</td><td>4.13</td></tr></table>

Table 7. Cross-dataset status of principal design choices.
<table><tr><td>Component</td><td>Setting</td><td>Status</td></tr><tr><td>Multiscale crops</td><td>100/200/400 px</td><td>Shared</td></tr><tr><td>Ordinal structure</td><td>ASC-US = 0.60</td><td>Shared</td></tr><tr><td>Binary evidential frame</td><td> $\Theta = \{ B , L \}$ </td><td>Shared</td></tr><tr><td>Reliability discounting</td><td> $\alpha \in [ 0 . 4 0 , 1 . 0 0 ]$ </td><td>Adapted</td></tr><tr><td>DST fusion</td><td>Sequential</td><td>Shared</td></tr><tr><td>BPA sharpness</td><td> $\lambda = 3$ </td><td>Shared</td></tr><tr><td>Uncertainty threshold</td><td> $m _ { U } ^ { \mathrm { h i g h } } = 0 . 5 0$ </td><td>Shared</td></tr><tr><td>Calibration</td><td>Temperature / Isotonic</td><td>Adapted</td></tr><tr><td>Scale thresholds</td><td> $t _ { s }$ </td><td>Adapted</td></tr><tr><td>CRIC NILM filter</td><td>2/3 lesion votes</td><td>Dataset-specific</td></tr></table>

Table 8. Uncertainty-stratified routing performance. Highconfidence predictions $( m \upsilon \leq 0 . 4 0 )$ substantially outperform uncertain predictions $( m \upsilon > 0 . 5 0 )$ , indicating that DST uncertainty mass is associated with prediction reliability.
<table><tr><td>Stratum</td><td colspan="2">RIVA</td><td colspan="2">CRIC</td></tr><tr><td></td><td>Macro-F1</td><td>Acc.</td><td>Macro-F1</td><td>Acc.</td></tr><tr><td>All</td><td>0.7259</td><td>0.7502</td><td>0.9238</td><td>0.9266</td></tr><tr><td>High confidence</td><td>0.7568</td><td>0.7830</td><td>0.9403</td><td>0.9424</td></tr><tr><td>Uncertain</td><td>0.5250</td><td>0.5345</td><td>0.5897</td><td>0.7500</td></tr></table>

RIVA/CRIC sample counts are 2,682/1,731 for all samples, 2,276/1,649 for high-confidence samples, and 275/16 for uncertain samples.

## 7. Conclusion

We presented a clinically grounded multiscale evidential framework for cervical cytology that combines nuclear, cellular, and neighbourhood evidence through calibrated Dempster–Shafer fusion. Across RIVA and CRIC Cervix, DST consistently improves over individual scales and alternative fusion strategies, with its strongest advantage occurring under inter-scale conflict. The fused evidence also preserves the ordinal progression of Bethesda categories, while incorrect predictions exhibit substantially higher uncertainty than correct predictions, particularly around ambiguous class boundaries. These results establish the value of explicitly representing inter-scale conflict and diagnostic ambiguity for more reliable cytology decisions and evidence-based routing.

## 8. Limitations

ENDO routing. ENDO is the clearest failure of the binary benign–lesion routing: RIVA accuracy is 0.222, with

46.7% silent-wrong rate and UR= 0.42, so incorrect predictions carry less uncertainty than correct ones. Its benign morphology can resemble HSIL, exposing a limitation of the binary evidential frame; a richer hypothesis space for such morphological confounds remains an important direction for future work.

CRIC uncertain pool. The genuine uncertain pool on CRIC contains only 16 test samples (0.9%), with limited class representation; class-level routing statistics are therefore directional. The CRIC-specific NILM vote filter yields a purer pool, but its threshold must be determined empirically for new datasets.

## References

[1] Ahmed Alrajjal, Vaishali Pansare, Moumita Saha Roy Choudhury, Mir Yousufuddin Ali Khan, and Vinod B. Shidham. Squamous intraepithelial lesions (SIL: LSIL, HSIL, ASCUS, ASC-H, LSIL-H) of uterine cervix and bethesda system. CytoJournal, 18:16, 2021. 1, 2, 3

[2] Grigor Bezirganyan, Sana Sellami, Laure Berti-Equille, and<sup>´</sup> Sebastien Fournier. EM-SEC: Efficient multi-head set-´ valued evidential classification. In Machine Learning and Knowledge Discovery in Databases: Research Track, pages 259–277. Springer, 2025. 2

[3] Wenzhi Cao, Vahid Mirjalili, and Sebastian Raschka. Rank consistent ordinal regression for neural networks with application to age estimation. Pattern Recognition Letters, 140: 325–331, 2020. 2

[4] Gulfize Coskun, Mustafa Caner Akuner, and Erkan Kaplanoglu. Robust cell-level classification for liquid-based cervical cytology using deep transfer learning: A multisource study addressing scanner-induced domain shifts. Bioengineering, 13(3):289, 2026. 2

[5] Arthur P. Dempster. Upper and lower probabilities induced by a multivalued mapping. The Annals of Mathematical Statistics, 38(2):325–339, 1967. 2

[6] Debora N. Diniz, Mariana T. Rezende, Andrea G. C. Bianchi,´ Claudia M. Carneiro, Eduardo J. S. Luz, Gladston J. P. Moreira, Daniela M. Ushizima, Fatima N. S. de Medeiros, and´ Marcone J. F. Souza. A deep learning ensemble method to assist cytopathologists in Pap test image classification. Journal ofImaging, 7(7):111, 2021. 2

[7] Debora N. Diniz, Mariana T. Rezende, Andrea G. C. Bianchi,´ Claudia M. Carneiro, Daniela M. Ushizima, Fatima N. S. de´ Medeiros, and Marcone J. F. Souza. A hierarchical featurebased methodology to perform cervical cancer classification. Applied Sciences, 11(9):4091, 2021. 2

[8] Didem Egemen, Li C. Cheung, Xiaojian Chen, Maria Demarco, Rebecca B. Perkins, Walter Kinney, Nancy Poitras, Brian Befano, Alexander Locke, Richard S. Guido, Amy L. Wiser, Julia C. Gage, Hormuzd A. Katki, Nicolas Wentzensen, Philip E. Castle, Mark Schiffman, and Thomas S. Lorey. Risk estimates supporting the 2019 AS-CCP risk-based management consensus guidelines. Journal ofLower Genital Tract Disease, 24(2):132–143, 2020. 1, 3

[9] Linlin Fan, Xingyu Liu, Mingliang Zhou, Xuekai Wei, Weizhi Xian, Jielu Yan, and Weijia Jia. A dynamic learn ing strategy for dempster-shafer theory with applications in classification and enhancement. In Advances in Neural In formation Processing Systems, 2025. 2

[10] Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q. Weinberger. On calibration of modern neural networks. In Proceedings of the 34th International Conference on Machine Learning, pages 1321–1330. PMLR, 2017. 4

[11] Emmanuel Iarussi and Manuel Andrade. Riva cervical cytology challenge - track a (isbi). https : / / kaggle.com/competitions/riva- cervicalcytology-challenge-isbi-2026, 2025. ISBI 2026 Challenge. 2

[12] Chris Kang, Vibujithan Vigneshwaran, and Nils D. Forkert. Riva cytology classification and detection: A lesson on how “none of us is as smart as all of us”. In 2026 IEEE 23rd International Symposium on Biomedical Imaging (ISBI), pages 1–4. IEEE, 2026. 2

[13] Marie C. Kempkes, Vedran Dunjko, Evert van Nieuwenburg, and Jakob Spiegelberg. Reliable classifications with guaranteed confidence using the dempster-shafer theory of evidence. In Machine Learning and Knowledge Discovery in Databases: Research Track, pages 89–105. Springer, 2024. 2

[14] Anwar Khan, Seunghyeon Han, Naveed Ilyas, Yong-Moon Lee, and Boreom Lee. CervixFormer: A multi-scale swin transformer-based cervical Pap-smear WSI classification framework. Computer Methods and Programs in Biomedicine, 240:107718, 2023. 2

[15] Lautaro Kogan and Mar´ıa Victoria R´ıos. Detection and classification of (pre)cancerous cells in pap smears: An ensemble strategy for the riva cervical cytology challenge. arXiv preprint arXiv:2603.23742, 2026. 2

[16] Wanli Liu, Chen Li, Ning Xu, Tao Jiang, Md Mamunur Rahaman, Hongzan Sun, Xiangchen Wu, Weiming Hu, Haoyuan Chen, Changhao Sun, Yudong Yao, and Marcin Grzegorzek. CVM-Cervix: A hybrid cervical Pap-smear image classification framework using CNN, visual transformer and multilayer perceptron. Pattern Recognition, 130:108829, 2022. 2

[17] Carlos Mart´ınez, Laura Busto, Olivia Zulaica, and Cesar´ Veiga. Interleaved fusion learning for trustworthy AI: Improving cross-dataset performance in cervical cancer analysis. Machine Learning and Knowledge Extraction, 7(4):128, 2025. 2

[18] Ritu Nayar and David C. Wilbur, editors. The Bethesda Sys tem for Reporting Cervical Cytology: Definitions, Criteria, and Explanatory Notes. Springer, 3 edition, 2015. 1

[19] Ritu Nayar and David C. Wilbur. The Pap test and Bethesda 2014. Cancer Cytopathology, 123(5):271–281, 2015. 1, 2

[20] Alexandru Niculescu-Mizil and Rich Caruana. Predicting good probabilities with supervised learning. In Proceedings of the 22nd International Conference on Machine Learning, pages 625–632. ACM, 2005. 4

[21] Maxime Oquab, Timothee Darcet, Th ´ eo Moutakanni, Huy V.´ Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez,

Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Mido Assran, Nicolas Ballas, Wojciech Galuba, Russell Howes, Po-Yao Huang, Shang-Wen Li, Ishan Misra, Mike Rabbat, Vasu Sharma, Gabriel Synnaeve, Hu Xu, Herve J´ egou,´ Julien Mairal, Patrick Labatut, Armand Joulin, and Piotr Bojanowski. DINOv2: Learning robust visual features without supervision. Transactions on Machine Learning Research, 2024. 3, 6

[22] Paula Perez Bianchi, Sol Anselmo, Malena Vasquez Curri´ e,´ Jimena Medel, Estefan´ıa Uelf, Alicia Dos Santos, Noem´ı Buosi, Rosana Vargas, Juliana Reves Szemere, Bruno Volcovinsky, Hugo Massaroli, Manuel Andrade, Alejandro Monastra, Emmanuel Iarussi, Viviana Siless, and Luciana Bruno. RIVA: An image dataset of conventional Pap smear cytology with multiple independent annotations. Scientific Data, 12: 1991, 2025. 2, 5

[23] Mariana T. Rezende, Raniere Silva, Fagner de O. Bernardo, Alessandra H. G. Tobias, Paulo H. C. Oliveira, Tales M. Machado, Caio S. Costa, Fatima N. S. Medeiros, Daniela M. Ushizima, Claudia M. Carneiro, and Andrea G. C. Bianchi. CRIC searchable image database as a public platform for conventional Pap smear cytology data. Scientific Data, 8(1): 151, 2021. 2, 5

[24] Murat Sensoy, Lance Kaplan, and Melih Kandemir. Evidential deep learning to quantify classification uncertainty. In Advances in Neural Information Processing Systems, pages 3183–3193, 2018. 2

[25] Glenn Shafer. A Mathematical Theory of Evidence. Princeton University Press, 1976. 2

[26] Xintong Shi, Wenzhi Cao, and Sebastian Raschka. Deep neural networks for rank-consistent ordinal regression based on conditional probabilities. Pattern Analysis and Applications, 26(3):941–955, 2023. 2

[27] Hiroki Shiraishi, Hisao Ishibuchi, and Masaya Nakata. A class inference scheme with dempster-shafer theory for learning fuzzy-classifier systems. ACM Transactions on Evolutionary Learning and Optimization, 6(1), 2026. 2

[28] Miguel Angel Valles-Coral, Lloy Pinedo, Ciro Rodr´ıguez, Diego Rodr´ıguez, Keller Sanchez-D ´ avila, Lolita Ar ´ evalo-´ Fasanando, and Nelly Reategui-Lozano. Application of ar-´ tificial intelligence in cervical cytology: A systematic review of deep learning models, datasets, and reported metrics. Frontiers in Big Data, 8:1678863, 2026. 2

[29] Bianca Zadrozny and Charles Elkan. Transforming classifier scores into accurate multiclass probability estimates. In Proceedings of the Eighth ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pages 694–699. ACM, 2002. 4