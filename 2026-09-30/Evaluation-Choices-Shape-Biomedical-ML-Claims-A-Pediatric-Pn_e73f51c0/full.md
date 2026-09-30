# Evaluation Choices Shape Biomedical ML Claims: A Pediatric Pneumonia Benchmark Case Study

Bhanu Prakash Vangala University of Missouri bv3hz@missouri.edu

Latha Peddi Government Degree College lathakappala@outlook.com

Sowmya Guda University of Missouri sghmy@missouri.edu

Navya Vangala Amar Bio Tech Pvt Ltd vangalanavya.8@gmail.com

## Abstract

Biomedical machine learning papers often compress model performance into one headline number. That number can look like a property of the model even when it depends strongly on how the benchmark was evaluated. We study this problem on the widely used Kermany pediatric chest radiograph dataset using nine image classifiers and a controlled evaluation protocol. Under the same protocol, the eight pretrained backbones differ by only 0.026 AUROC. In contrast, changing whether the backbone is frozen or fine tuned changes AUROC by 0.044 on average, and changing the decision threshold changes balanced accuracy by 0.090 on average. The official test split is also measurably different from the training pool: a partition classifier distinguishes them at AUC 0.697, rising to 0.898 for normal radiographs. Most strikingly, a classifier using only file properties, with no image anatomy, reaches 0.992 balanced accuracy within the training pool but falls to 0.496 on the official test split. Validation fitted thresholds and calibration also transfer imperfectly. These results show that a high benchmark score can support different conclusions when the split, training policy, threshold, metric, calibration, and uncertainty are not communicated with it. We end with a seven item reporting recommendation in which each item is tied to an effect measured in the study.

## 1 Introduction

A biomedical machine learning paper is often remembered by one number. A reader may remember 98% accuracy or 0.97 AUROC long after the details of the experiment are forgotten. That number can then appear in related work, presentations, grant proposals, or discussions of clinical readiness. The problem is that the number is usually read as a property of the model, even though it also depends on how the model was evaluated.

Problem statement. Let $f _ { \theta }$ denote a trained classifier, D the dataset, and π the evaluation protocol. The reported score is

$$
s = S ( f _ { \theta } , \pi ; \mathcal { D } ) ,\tag{1}
$$

where $S$ is the scoring procedure. In this study, the protocol contains five choices that are common in biomedical image classification:

$$
\begin{array} { r } { \pi = \left( \underbrace { \sigma } _ { \mathrm { s p l i t } } , \underbrace { \phi } _ { \mathrm { b a c k b o n e ~ t r a i n i n g } } , \underbrace { t } _ { \mathrm { t h r e s h o l d } } , \underbrace { T } _ { \mathrm { c a l i b r a t i o n ~ m e t r i c } } \right) . } \end{array}\tag{2}
$$

Papers usually describe the architecture $f _ { \theta }$ in detail, but the elements of π may be omitted, treated as defaults, or described only deep in the methods. If changing one element of π changes the score as much as changing the architecture, then a headline score cannot be interpreted without the protocol that produced it.

We make that comparison explicit. For a fixed reference protocol $\pi _ { 0 }$ and a set of architectures $\mathcal { F }$ , we define the architectural spread and the effect of changing one protocol component $j$ as

$$
\Delta _ { \mathrm { a r c h } } = \operatorname* { m a x } _ { f \in \mathcal { F } } S ( f , \pi _ { 0 } ; \mathcal { D } ) - \operatorname* { m i n } _ { f \in \mathcal { F } } S ( f , \pi _ { 0 } ; \mathcal { D } ) , \qquad \Delta _ { \pi } ^ { ( j ) } = | S ( f , \pi ; \mathcal { D } ) - S ( f , \pi ^ { \prime } ; \mathcal { D } ) | ,\tag{3}
$$

where $\pi$ and $\pi ^ { \prime }$ differ only in component $j .$ Equation 3 gives the paper a simple question: are differences attributed to models larger than differences created by ordinary evaluation choices?

Case study. We study the pediatric chest radiograph dataset released by Kermany et al. [Kermany et al., 2018]. It is a useful case because it is public, small enough for controlled experiments, widely reused, and distributed with a fixed train and test split. Published studies report high performance on this corpus, often under evaluation setups that are not directly comparable [Stephen et al., 2019, Kundu et al., 2021, Siddiqi and Javaid, 2024]. Some preserve the official test split, while others pool the corpus and create a new split. Some freeze a pretrained backbone, while others fine tune it. Threshold selection and calibration are often not reported. These are exactly the choices represented by Eq. 2.

Main findings. First, under one fixed protocol, the eight pretrained backbones differ by only 0.026 AUROC, while changing whether the backbone is frozen or fine tuned changes AUROC by 0.044 on average and as much as 0.103. Second, the official test split is measurably different from the training pool. A partition classifier distinguishes the two at AUC 0.697, rising to 0.898 for normal radiographs. Third, a model that uses only file properties and no anatomy reaches 0.992 balanced accuracy within the training pool but only 0.496 on the official test split. This makes the risk of pooling and randomly re-splitting the corpus concrete. Fourth, thresholds and calibration fitted on validation data do not fully transfer to the official test distribution. These results are summarized in Table 2 and developed in Section 4.

Contribution to responsible communication. Our contribution is not a new pneumonia classifier and not a claim that this benchmark is unusable. It is a controlled case study of how the same biomedical benchmark can support different stories when the evaluation protocol is hidden. We connect each communication recommendation to a measured effect, rather than presenting a generic checklist. This directly addresses the gap between technical ML results and the claims that readers, clinicians, reviewers, and policymakers may take from them. Existing reporting guidance such as TRIPOD+AI already asks for many of these details [Collins et al., 2024]; our goal is to show, with measured examples, why those details change the meaning of the result.

## 2 Background and Related Work

Benchmark performance can reflect shortcuts rather than the intended signal. High performance on a medical imaging dataset does not by itself show that a model learned clinically meaningful evidence. Models can exploit acquisition patterns, hospital identity, image processing, text markers, or other dataset specific features that correlate with the label but do not transfer to a new setting [Zech et al., 2018, DeGrave et al., 2021]. Shortcut learning is a broader machine learning problem in which a predictor uses an easy correlated feature instead of the mechanism the task is intended to measure [Geirhos et al., 2020]. This distinction matters for communication because the headline metric describes predictive performance, not what evidence the model used.

Evaluation design can change the apparent conclusion. Medical AI studies are especially sensitive to small test sets, leakage, data reuse, class imbalance, and unstable model rankings [Varoquaux and Cheplygina, 2022, Roberts et al., 2021]. Related benchmark studies show that duplicates can change rankings and that newly collected test sets can reveal drops that were hidden by the original evaluation [Barz and Denzler, 2020, Recht et al., 2019]. These observations motivate our controlled comparison of architecture effects with protocol effects in Eq. 3. We also audit duplicates separately because contamination and distribution shift can produce superficially similar performance patterns.

Thresholds and calibration are part of the claim. AUROC measures ranking over thresholds, but a decision system eventually operates at one threshold and presents confidence values that users may interpret as probabilities. Youden’s index is a standard way to choose an operating point when sensitivity and specificity are weighted equally [Youden, 1950]. Temperature scaling is a standard post hoc calibration method that often improves confidence estimates on data drawn from the same distribution as the validation set [Guo et al., 2017]. When validation and test distributions differ, however, the chosen threshold and calibration map may not transfer. This is important in medical AI because a model can preserve ranking performance while producing poorly chosen decisions or misleading confidence values [Liu et al., 2022a].

Reporting standards address many of these details, but the communication gap remains. TRIPOD+AI asks authors to describe evaluation data, discrimination, calibration, thresholds, and uncertainty for prediction model studies [Collins et al., 2024]. The challenge is not only whether such information exists somewhere in a paper. The challenge is whether the performance claim that travels outside the paper retains enough context to be interpreted correctly. Our study therefore treats the benchmark score itself as a communication object and measures how much meaning is lost when the protocol around that score is omitted.

## 3 Study Design

This section defines the quantities used in the Results so that each reported effect corresponds to a specific part of the protocol in Eq. 2. Let $p _ { i } \in [ 0$ , 1] be the predicted probability of pneumonia for case $i , y _ { i } \in \{ 0 , 1 \}$ its label, and $z _ { i }$ the model logits.

Dataset and split policy $\sigma \cdot$ The Kermany corpus contains pediatric chest radiographs labeled NORMAL or PNEUMONIA, distributed as 5,216 training images, 16 validation images, and 624 test images [Kermany et al., 2018]. The official validation set contains only eight images per class, so it is too small for stable early stopping, threshold selection, or calibration. We therefore create a patient disjoint validation split from the released training pool and leave the official test split untouched. Filenames encode a patient or study identifier, so related images are assigned as a group. We remove exact byte level duplicates before training, yielding 4420 training images, 770 validation images, and 618 official test images. Figure 5 in Appendix D shows the split composition. A separate near duplicate audit in the same appendix checks that train to test leakage is not the explanation for the distributional effects reported in Section 4.3.

Models and backbone training $\phi .$ We evaluate a small CNN trained from scratch and eight pretrained backbones: ResNet-18 and ResNet-50 [He et al., 2016], VGG-16 [Simonyan and Zisserman, 2015], EfficientNet-B0 [Tan and Le, 2019], DenseNet-121 [Huang et al., 2017], ConvNeXt-T [Liu et al., 2022b], ViT-B/16 [Dosovitskiy et al., 2021], and Swin-T [Liu et al., 2021]. For the main comparison, all pretrained backbones are fully fine tuned using the same split, augmentation, optimizer family, training budget, and three random seeds. To isolate $\Delta _ { \pi } ^ { ( \phi ) }$ in Eq. 3, we repeat each pretrained architecture with the backbone frozen and only the classifier head trainable. Full per architecture results are provided in Appendix A.

Decision threshold t and metric $m ,$ . A binary prediction is $\hat { y } _ { i } = \mathcal { W } [ p _ { i } \ge t ]$ . The official test set is 62.5% positive, so an always positive classifier already obtains 0.625 raw accuracy. We therefore emphasize balanced accuracy for threshold dependent comparisons and report AUROC and AUPRC for ranking performance. We define

$$
\mathrm { B A } ( t ) = \frac { \mathrm { T P R } ( t ) + \mathrm { T N R } ( t ) } { 2 } , \qquad t ^ { \star } = \arg \operatorname* { m a x } _ { t } \left[ \mathrm { T P R } _ { \mathrm { v a l } } ( t ) + \mathrm { T N R } _ { \mathrm { v a l } } ( t ) - 1 \right] .\tag{4}
$$

The first quantity is the reported balanced accuracy and the second is the validation selected threshold using Youden’s J [Youden, 1950]. Section 4.5 compares this threshold with the default $t = 0 . 5$ and with a test selected oracle used only to measure the transfer gap.

Calibration T. We fit one temperature on validation logits by minimizing cross entropy and measure calibration using expected calibration error with 15 equal width confidence bins [Guo et al.,

2017]:

$$
T ^ { \star } = \underset { T > 0 } { \operatorname { a r g m i n } } - \sum _ { i \in \mathrm { v a l } } \log \operatorname { s o f t m a x } ( z _ { i } / T ) _ { y _ { i } } , \qquad \operatorname { E C E } = \sum _ { b = 1 } ^ { B } \frac { | b | } { N } \left| \operatorname { a c c } ( b ) - \operatorname { c o n f } ( b ) \right| .\tag{5}
$$

A value $T ^ { \star } < 1$ sharpens the probabilities and $T ^ { \star } > 1$ softens them. Section 4.5 evaluates whether the validation fitted correction improves calibration on the official test split, and Appendix B provides the full run level results and reliability diagram.

Distribution shift test. To test whether two partitions are exchangeable, we extract frozen ResNet-50 features and train logistic regression to predict whether an image came from the training pool or the official test split. We use five fold out of fold predictions and repeat the analysis after shuffling partition labels. In addition to discriminator AUC, we report the proxy A distance [Ben-David et al., 2010]:

$$
d _ { A } = 2 ( 1 - 2 \epsilon ) , \qquad \epsilon = \mathrm { b a l a n c e d e r r o r ~ o f ~ t h e ~ p a r t i t i o n ~ c l a s s i f i e r . }\tag{6}
$$

A value near 0 means that the partitions are difficult to distinguish, while a value near 2 means that they are easily separable. We repeat the procedure on our own train and validation partitions as a control.

File only shortcut test. We train a gradient boosted tree using only file properties: pixel dimensions, aspect ratio, file size, bytes per pixel, image mode, and JPEG quantization tables. It receives no image pixels and no anatomical features. We first evaluate it with five fold cross validation inside the training pool, then fit it on that pool and evaluate it on the untouched official test split. A large drop between these two evaluations would show that file construction artifacts encode the label within one partition but do not transfer to the official test set.

Target prevalence operating point. The test selected threshold in Section 4.5 is only an oracle and cannot be used in a valid evaluation. To test whether target information can recover some of that gap without labels, we also consider an operating point that matches an assumed target prevalence πˆ:

$$
t _ { \hat { \pi } } = \operatorname* { i n f } \left\{ t : { \frac { 1 } { n } } \sum _ { i = 1 } ^ { n } { \mathcal { k } } [ p _ { i } \geq t ] \leq { \hat { \pi } } \right\} .\tag{7}
$$

This method uses unlabeled target predictions plus an externally specified prevalence. We use $\hat { \pi } = \ ? ?$ only as a sensitivity analysis, with the result and its limitations reported in Section 4.5 and Appendix B.

Uncertainty and statistical testing. We report mean and standard deviation over three seeds where available. Confidence intervals for AUROC are obtained by percentile bootstrap resampling of test cases with 2,000 resamples, and paired AUROC differences are tested using DeLong’s method [DeLong et al., 1988]. Precision-recall curves and AUPRC are also reported because class imbalance can make precision-recall summaries more informative than ROC summaries for some comparisons [Saito and Rehmsmeier, 2015]. Statistical details and the full curves are in Appendices C and E.

## 4 Results

Table 1 gives the main model results on the untouched official test split, and Table 2 summarizes the effects associated with the protocol components in Eq. 2. The full metric tables are moved to Appendix A so that the main text can focus on the comparisons that change the interpretation of the benchmark score.

## 4.1 Architecture differences are small relative to several protocol effects

Under the reference protocol $\pi _ { 0 } .$ , the eight pretrained architectures occupy a narrow AUROC range of 0.026. Their bootstrap intervals overlap substantially in Figure 1a, and the paired DeLong analysis in Appendix C shows that most pairwise differences are not distinguishable at this test set size. The custom CNN trained from scratch is clearly weaker, but the ordering among the pretrained backbones should not be read as a stable leaderboard.

Table 1: Performance on the untouched official test split (mean±s.d. over three seeds). All pretrained backbones are fully fine-tuned, and the decision threshold is selected on the patient-disjoint validation split. The pretrained models have similar AUROC despite different architectures.
<table><tr><td>Architecture</td><td>Bal. Acc.</td><td>Sens.</td><td>Spec.</td><td>AUROC</td><td>ECE↓</td></tr><tr><td>Custom CNN (scratch)</td><td>0.697±0.032</td><td>0.881±0.027</td><td> $0 . 5 1 2 { \scriptstyle \pm 0 . 0 7 6 }$ </td><td>0.748±0.042</td><td>0.178±0.044</td></tr><tr><td>ResNet-18</td><td> $0 . 7 6 8 { \scriptstyle \pm 0 . 0 1 6 }$ </td><td>0.967±0.019</td><td> $0 . 5 6 9 { \pm } 0 . 0 3 8$ </td><td>0.909±0.006</td><td>0.143±0.027</td></tr><tr><td>VGG-16</td><td> $0 . 8 0 1 { \scriptstyle \pm 0 . 0 2 3 }$ </td><td>0.996±0.001</td><td> $0 . 6 0 6 { \scriptstyle \pm 0 . 0 4 7 }$ </td><td> $0 . 9 5 7 { \scriptstyle \pm 0 . 0 0 7 }$ </td><td> $0 . 1 5 0 { \scriptstyle \pm 0 . 0 1 8 }$ </td></tr><tr><td>ResNet-50</td><td> $\mathbf { 0 . 8 2 8 { \scriptstyle \pm 0 . 0 0 6 } }$ </td><td>0.984±0.000</td><td> $\mathbf { 0 . 6 7 1 { \scriptstyle \pm 0 . 0 1 3 } }$ </td><td> $0 . 9 4 9 { \pm } 0 . 0 0 5$ </td><td> $0 . 1 6 3 { \scriptstyle \pm 0 . 0 0 5 }$ </td></tr><tr><td>EfficientNet-B0</td><td> $0 . 7 8 4 { \scriptstyle \pm 0 . 0 1 1 }$ </td><td>0.963±0.007</td><td> $0 . 6 0 5 { \scriptstyle \pm 0 . 0 2 9 }$ </td><td> $0 . 8 9 3 { \scriptstyle \pm 0 . 0 0 8 }$ </td><td> $0 . 1 9 1 { \scriptstyle \pm 0 . 0 0 1 }$ </td></tr><tr><td>DenseNet-121</td><td> $0 . 7 7 5 { \scriptstyle \pm 0 . 0 4 1 }$ </td><td>0.992±0.006</td><td> $0 . 5 5 7 { \pm } 0 . 0 8 4$ </td><td> $0 . 9 4 6 { \pm } 0 . 0 1 4$ </td><td>0.194±0.008</td></tr><tr><td>ConvNeXt-T</td><td>0.820±0.035</td><td>0.994±0.003</td><td> $0 . 6 4 6 { \scriptstyle \pm 0 . 0 7 1 }$ </td><td> $0 . 9 7 0 { \scriptstyle \pm 0 . 0 0 8 }$ </td><td> $0 . 1 8 1 { \pm } 0 . 0 1 6$ </td></tr><tr><td>ViT-B/16</td><td> $0 . 8 1 \bar { 1 } \bar { \pm } 0 . 0 2 2$ </td><td>0.996±0.001</td><td> $0 . 6 2 6 { \scriptstyle \pm 0 . 0 4 3 }$ </td><td> $\mathbf { 0 . 9 7 4 } \pm \mathbf { 0 . 0 0 8 }$ </td><td> $0 . 1 4 3 { \pm } 0 . 0 4 8$ </td></tr><tr><td>Swin-T</td><td>0.809±0.016</td><td>0.995±0.000</td><td> $0 . 6 2 3 { \scriptstyle \pm 0 . 0 3 1 }$ </td><td> $0 . 9 6 9 { \scriptstyle \pm 0 . 0 0 5 }$ </td><td> $0 . 1 7 7 { \scriptstyle \pm 0 . 0 2 8 }$ </td></tr></table>

ECE: expected calibration error. Sensitivity is for the PNEUMONIA class. Raw accuracy is not emphasized because the official test set is 62.5% positive.

Table 2: Observed effect of the evaluation choices defined in Eq. 2. The architecture row gives the AUROC range across the eight pretrained backbones under one fixed protocol. Other rows change one evaluation component or test whether the evaluation partitions differ. Oracle values are used only to measure transfer gaps.
<table><tr><td>Question</td><td>Comparison</td><td>Observed effect</td></tr><tr><td>Architecture under fixed π0</td><td>eight pretrained backbones</td><td>AUROC range 0.026</td></tr><tr><td>Backbone training φ</td><td>frozen vs. fine tuned</td><td>mean |∆| 0.044 AUROC; max 0.103</td></tr><tr><td>Threshold t</td><td>0.5 vs. validation selected</td><td>mean |∆| 0.090 balanced accuracy</td></tr><tr><td>Threshold transfer</td><td>validation vs. test oracle</td><td>mean gap 0.083 balanced accuracy</td></tr><tr><td>Calibration T</td><td>raw vs. validation fitted</td><td>ECE 0.172 to 0.169; worse in 9/26</td></tr><tr><td>Split σ and file shortcut Train to test shift</td><td>training pool CV vs. official test partition classifier</td><td>runs 0.992 to 0.496 balanced accuracy AUC 0.697 overall; 0.898 on normal images</td></tr></table>

![](images/59ef8968d1993d968ed82443be76d2c447c025b919436ccfba703ae47890f08b.jpg)  
(a) AUROC with bootstrap 95% confidence intervals.

![](images/6f7815289194b5475d7d068d6cde637641c2bd61ca22e83f61ebea9d1caff6aa.jpg)  
(b) Frozen and fine tuned backbones.  
Figure 1: Two views of protocol sensitivity. Panel (a) shows that the pretrained architectures have heavily overlapping uncertainty intervals under one fixed protocol. Panel (b) holds architecture, split, augmentation, and training schedule fixed while changing only whether the pretrained backbone is frozen or fine tuned. The crossing lines show that the architecture ranking is not preserved.

The uncertainty is also large relative to the architectural spread. With 618 test images after exact duplicate removal, the 95% bootstrap interval is roughly ±0.02 AUROC for individual models. This means that reporting a score to three decimal places without an interval can communicate more precision than the benchmark supports. In the notation of Eq. 3, $\Delta _ { \mathrm { a r c h } } = 0 . 0 2 6$ is the reference against which we compare the protocol effects below.

## 4.2 Backbone training changes the conclusion more than architecture choice

Changing ϕ from a frozen feature extractor to full fine tuning changes AUROC by 0.044 on average across the eight pretrained architectures and by as much as 0.103. Both values are larger than the 0.026 architectural spread defined above. Figure 1b also shows that the sign of the change is not constant. Some models improve after fine tuning, while others change little or lose balanced accuracy, so a ranking obtained under one backbone training policy cannot simply be transferred to the other.

This comparison is intentionally simple because the communication failure is simple. Two papers can name the same architecture but train it in materially different ways. If the backbone training policy is absent from the headline comparison, a reader may attribute a difference to architecture design when it is partly or mainly a protocol effect. Appendix Table 5 provides the per architecture values for balanced accuracy, F<sub>1</sub>, and AUROC.

## 4.3 The official test split is measurably different from the training pool

The partition classifier separates the training pool from the official test split at AUC 0.697, compared with 0.503 after shuffling partition labels. The shift is not uniform across classes. For NORMAL radiographs, partition AUC rises to 0.898, while for PNEUMONIA it is 0.664. Applying Eq. 6 gives a proxy A distance of 1.226 for the normal class, compared with 0.117 for our own patient disjoint train and validation split. Figure 2a makes this contrast visible, and Appendix Table 8 gives the corresponding proxy A distances.

Two controls make the interpretation more specific. First, the same partition classifier applied to our train and validation partitions reaches only 0.542, which is close to chance. Second, the duplicate audit in Appendix D finds no test image with a near twin in the training pool at the primary similarity threshold. These checks make simple duplication an unlikely explanation. The official test split therefore behaves as a small transfer set rather than as an exchangeable sample from the training pool.

We cannot identify the source of the shift because the released metadata do not contain the acquisition and curation information needed to do so. Similar hidden acquisition and site effects are well documented in medical imaging datasets [Zech et al., 2018, Oakden-Rayner, 2020]. Our claim is therefore limited to what the experiment establishes: the partitions are distinguishable, the difference is strongest for normal images, and post hoc choices fitted on the training distribution should not be assumed to transfer unchanged.

![](images/6d1a65e60926e9a50264e400220c4f8e610a01d9b39daac21615cbdba6721850.jpg)  
(a) Partition distinguishability. The dashed line marks chance AUC.

![](images/e4a36e87c49c94ff2685b5719f1153ed4e5de09300be1a40e85f500dd65a8ed1.jpg)  
(b) File only label prediction across split policies.

Figure 2: Evidence that split policy changes what the benchmark measures. Panel (a) shows that the official test split is distinguishable from the training pool, especially for normal radiographs, while the study’s own train and validation partitions are much closer to chance. Panel (b) shows that file properties alone nearly solve the label task inside the training pool but fail on the untouched official test split. Together, these results explain why pooling and randomly re-splitting the released data can preserve a shortcut that does not transfer.

## 4.4 A file only predictor shows why pooling and re-splitting can be misleading

The file only classifier makes the split problem easy to see. It uses image dimensions, file size, bytes per pixel, image mode, and JPEG quantization information, but no radiograph pixels and therefore no anatomy. Within the training pool it predicts the diagnosis at 0.992 balanced accuracy under five fold cross validation. When fitted on that pool and evaluated on the untouched official test split, balanced accuracy falls to 0.496, which is essentially chance. Figure 2b shows this collapse next to the partition shift result so that the two pieces of evidence can be read together.

The result should not be interpreted as evidence that file headers cause pneumonia. It shows that the two labels in the training pool were stored or processed in systematically different ways. Those construction artifacts provide an easy shortcut inside that pool, but the shortcut does not survive the move to the official test split. This pattern is consistent with the broader shortcut learning literature, where nonclinical acquisition signals can support apparently strong within dataset performance [DeGrave et al., 2021, Geirhos et al., 2020].

This directly explains why split policy σ is part of the reported score in Eq. 1. If the released partitions are pooled and randomly re-split, train and test examples can share the same file level signature. The resulting evaluation may reward regularities that disappear on the official split. The pair 0.992 within the training pool and 0.496 on the official test set provides a concrete measured example of how a seemingly small evaluation decision can change the story attached to a high score.

## 4.5 Thresholds and calibration fitted on validation do not fully transfer

The decision threshold is another large protocol effect. Moving from the default $t = 0 . 5$ to the validation selected $t ^ { \star }$ in Eq. 4 changes balanced accuracy by 0.090 on average, which is larger than $\Delta _ { \mathrm { a r c h } } .$ . The full per architecture comparison is in Appendix Table 6. Figure 3a gives the case count view for the strongest model and shows why the threshold must be reported together with any label based metric.

Validation tuning does not remove the entire operating point problem. Choosing the threshold directly on the test labels, which is an invalid oracle used only to measure the gap, improves balanced accuracy by a further 0.083 on average. The target prevalence rule in Eq. 7 is a sensitivity analysis that does not use target labels. In the aggregate experiment it raises balanced accuracy from 0.787 to 0.863, compared with the unattainable test oracle of 0.870, recovering 92% of that specific gap. Appendix Table 7 reports these values. We do not propose this as a general deployment rule because it depends on a credible external prevalence estimate. We include it to show that the failure is connected to the target distribution rather than to a lack of threshold optimization.

Calibration shows a related transfer problem. Using Eq. 5, mean test ECE changes only from 0.172 before calibration to 0.169 after applying a temperature fitted on validation data, and ECE becomes worse in 9 of 26 runs. The mean fitted temperature is 1.41, and $T ^ { \star } < 1$ occurs in 9 runs, so the direction of the fitted correction is not consistent across models. Figure 3b shows the corresponding reliability diagram for ViT-B/16, while Appendix Table 8 provides the Brier decomposition. The conclusion is not that temperature scaling is ineffective in general. It is that the phrase “the model was calibrated” is incomplete unless the paper states where calibration was fitted and where it was evaluated.

![](images/f1270f076d5a363a29e3bc7aeeeb671af483bbf996913630c6ba493688696858.jpg)  
(a) Default versus validation selected threshold for ViT-B/16.

![](images/2ee09ffa4d317fc86f3ef6ddd8f3afc0400e4ea49e00bd5a10eab921db76d7c3.jpg)

![](images/449b7a11d9ec53ffec8989f1d4b89ad6a84df98d1d1a85685623a3548d6f2fa8.jpg)  
(b) Reliability before and after validation fitted temperature scaling.  
Figure 3: Post hoc choices that change the reported interpretation on the official test split. Panel (a) shows how threshold selection changes the false positive and true negative counts while leaving the model itself unchanged. Panel (b) shows that a temperature fitted on validation data changes confidence estimates but does not remove the calibration gap on the official test distribution.

## 5 From a Benchmark Score to a Responsible Claim

Published numbers on the same corpus are not automatically comparable. Table 3 places two published results beside our evaluation to show how different protocols can sit behind similar headline numbers. Stephen et al. [Stephen et al., 2019] report 0.937 accuracy after pooling and re-splitting the data, while Kundu et al. [Kundu et al., 2021] report 0.988 accuracy under pooled cross validation. Our rows preserve the official test split. Because split policy, metric, model training, and operating point differ, these values answer different questions rather than forming one valid ranking. Equation 2 makes those differences explicit.

Table 3: Examples of reported results on the same dataset under different evaluation setups. These rows should not be read as a model ranking: the split and metric differ across studies. “Pooled” means that images were combined and re-partitioned rather than evaluated on the untouched official test split.
<table><tr><td>Study</td><td>Method</td><td>Evaluation setup</td><td>Reported score</td></tr><tr><td>[Stephen et al., 2019] [Kundu et al., 2021]</td><td>CNN from scratch pooled, re-split</td><td>3-model ensemble pooled, 5-fold CV</td><td>0.937 acc. 0.988 acc.</td></tr><tr><td colspan="4">This work: untouched official test split; threshold selected on validation</td></tr><tr><td colspan="4">ViT-B/16</td></tr><tr><td></td><td>fine-tuned</td><td>official test</td><td>0.858 acc.</td></tr><tr><td>ViT-B/16</td><td>fine-tuned</td><td>official test</td><td>0.811 bal. acc.</td></tr></table>

Seven details should travel with the headline score. The measured effects in Table 2 suggest a compact reporting rule. A biomedical benchmark result should state the evaluation split, whether pretrained features were frozen or fine tuned, the decision threshold and how it was selected, a metric that exposes class imbalance, how and where calibration was fitted, uncertainty or an appropriate paired comparison, and a basic train to test shift check when practical. These are not stylistic preferences. Each item corresponds to an effect measured in Sections 4.1 to 4.5.

This recommendation is consistent with the broader goals of TRIPOD+AI [Collins et al., 2024], but our point is narrower: the cost of missing protocol details can be measured on the benchmark itself. Here, backbone training changes AUROC more than the pretrained architecture spread, threshold selection changes balanced accuracy by even more, the file only shortcut falls from 0.992 to 0.496 across split policies, and the official test partition is measurably different from the training pool.

A benchmark metric cannot say what evidence the model used. Appendix F provides a supplementary Grad-CAM [Selvaraju et al., 2017] and SHAP [Lundberg and Lee, 2017] audit. We summarize attribution mass outside a body mask rather than treating a heatmap as a causal explanation. This analysis is separate from the protocol results, but it illustrates the same communication limit: a strong discrimination score does not reveal whether the visual evidence used by a model is clinically meaningful.

What can safely be claimed from this benchmark? The strongest supported claim is that modern pretrained image classifiers separate the two released labels well on the official split under a stated protocol. The experiments do not establish cross hospital generalization, clinical utility, readiness for triage, or meaningful superiority of one closely clustered pretrained architecture over another. Because the official split also contains an undocumented distribution difference, a responsible summary should report the score together with the split, metric, threshold, uncertainty, and observed shift.

Why this belongs in a communication workshop. The failure occurs when a precise result is shortened into a claim that drops the protocol needed to interpret it. Table 2 shows the remedy: keep those technical conditions attached to the result by expressing them as measured changes in the reported score.

## 6 Limitations and Conclusion

Limitations. This is a case study of one public pediatric radiograph dataset, not a survey of biomedical benchmarks. The released metadata do not identify the source of the train to test shift, and the file only experiment establishes a construction shortcut without proving that a particular

neural network uses it. The small test set limits architecture comparisons, while the target prevalence threshold is only a sensitivity analysis because reliable prevalence may not be available at deployment. Additional robustness and reproducibility details appear in Appendices G and H.

Conclusion. A benchmark score reflects both model and protocol. Here, training, split, threshold, calibration, and shift choices affect the result as much as or more than architecture choice. Those conditions need to travel with the score.

## References

Björn Barz and Joachim Denzler. Do we train on test data? Purging CIFAR of near-duplicates. Journal of Imaging, 6(6):41, 2020. doi:10.3390/jimaging6060041.

Shai Ben-David, John Blitzer, Koby Crammer, Alex Kulesza, Fernando Pereira, and Jennifer Wortman Vaughan. A theory of learning from different domains. Machine Learning, 79(1–2):151–175, 2010. doi:10.1007/s10994-009-5152-4.

Gary S. Collins, Karel G. M. Moons, Paula Dhiman, Richard D. Riley, Andrew L. Beam, Ben Van Calster, Marzyeh Ghassemi, Xiaoxuan Liu, Johannes B. Reitsma, Maarten van Smeden, et al. TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. BMJ, 385:e078378, 2024. doi:10.1136/bmj-2023-078378.

Alex J. DeGrave, Joseph D. Janizek, and Su-In Lee. AI for radiographic COVID-19 detection selects shortcuts over signal. Nature Machine Intelligence, 3(7):610–619, 2021. doi:10.1038/s42256-021- 00338-7.

Elizabeth R. DeLong, David M. DeLong, and Daniel L. Clarke-Pearson. Comparing the areas under two or more correlated receiver operating characteristic curves: A nonparametric approach. Biometrics, 44(3):837–845, 1988. doi:10.2307/2531595.

Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In Proc. Int. Conf. Learning Representations (ICLR), 2021. arXiv:2010.11929.

Robert Geirhos, Jörn-Henrik Jacobsen, Claudio Michaelis, Richard Zemel, Wieland Brendel, Matthias Bethge, and Felix A. Wichmann. Shortcut learning in deep neural networks. Nature Machine Intelligence, 2(11):665–673, 2020. doi:10.1038/s42256-020-00257-z.

Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q. Weinberger. On calibration of modern neural networks. In Proc. Int. Conf. Machine Learning (ICML), pages 1321–1330, 2017. arXiv:1706.04599.

Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR), pages 770–778, 2016. doi:10.1109/CVPR.2016.90.

Gao Huang, Zhuang Liu, Laurens van der Maaten, and Kilian Q. Weinberger. Densely connected convolutional networks. In Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR), 2017. doi:10.1109/CVPR.2017.243.

Daniel S. Kermany, Michael Goldbaum, Wenjia Cai, Carolina C. S. Valentim, Huiying Liang, Sally L. Baxter, Alex McKeown, Ge Yang, Xiaokang Wu, Fangbing Yan, Justin Dong, Made K. Prasadha, Jacqueline Pei, Magdalene Y. L. Ting, et al. Identifying medical diagnoses and treatable diseases by image-based deep learning. Cell, 172(5):1122–1131.e9, 2018. doi:10.1016/j.cell.2018.02.010.

Rohit Kundu, Ritacheta Das, Zong Woo Geem, Gi-Tae Han, and Ram Sarkar. Pneumonia detection in chest X-ray images using an ensemble of deep learning models. PLOS ONE, 16(9):e0256630, 2021. doi:10.1371/journal.pone.0256630.

Xiaoxuan Liu, Ben Glocker, Melissa M. McCradden, Marzyeh Ghassemi, Alastair K. Denniston, and Lauren Oakden-Rayner. The medical algorithmic audit. The Lancet Digital Health, 4(5): e384–e397, 2022a. doi:10.1016/S2589-7500(22)00003-6.

Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin transformer: Hierarchical vision transformer using shifted windows. In Proc. IEEE/CVF Int. Conf. Computer Vision (ICCV), 2021. doi:10.1109/ICCV48922.2021.00986.

Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, and Saining Xie. A ConvNet for the 2020s. In Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR), 2022b. doi:10.1109/CVPR52688.2022.01167.

Scott M. Lundberg and Su-In Lee. A unified approach to interpreting model predictions. In Advances in Neural Information Processing Systems (NeurIPS), volume 30, 2017.

Luke Oakden-Rayner. Exploring large-scale public medical image datasets. Academic Radiology, 27 (1):106–112, 2020. doi:10.1016/j.acra.2019.10.006.

Benjamin Recht, Rebecca Roelofs, Ludwig Schmidt, and Vaishaal Shankar. Do ImageNet classifiers generalize to ImageNet? In Proc. Int. Conf. Machine Learning (ICML), 2019. arXiv:1902.10811.

Michael Roberts, Derek Driggs, Matthew Thorpe, Julian Gilbey, Michael Yeung, Stephan Ursprung, Angelica I. Aviles-Rivero, Christian Etmann, Cathal McCague, Lucian Beer, Jonathan R. Weir-McCall, Zhongzhao Teng, Effrossyni Gkrania-Klotsas, James H. F. Rudd, Evis Sala, Carola-Bibiane Schönlieb, et al. Common pitfalls and recommendations for using machine learning to detect and prognosticate for COVID-19 using chest radiographs and CT scans. Nature Machine Intelligence, 3(3):199–217, 2021. doi:10.1038/s42256-021-00307-0.

Takaya Saito and Marc Rehmsmeier. The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. PLOS ONE, 10(3):e0118432, 2015. doi:10.1371/journal.pone.0118432.

Ramprasaath R. Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam, Devi Parikh, and Dhruv Batra. Grad-CAM: Visual explanations from deep networks via gradient-based localization. In Proc. IEEE Int. Conf. Computer Vision (ICCV), 2017. doi:10.1109/ICCV.2017.74.

Raheel Siddiqi and Sameena Javaid. Deep learning for pneumonia detection in chest X-ray images: A comprehensive survey. Journal ofImaging, 10(8):176, 2024. doi:10.3390/jimaging10080176.

Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image recognition. In Proc. Int. Conf. Learning Representations (ICLR), 2015. arXiv:1409.1556.

Okeke Stephen, Mangal Sain, Uchenna Joseph Maduh, and Do-Un Jeong. An efficient deep learning approach to pneumonia classification in healthcare. Journal of Healthcare Engineering, 2019: 4180949, 2019. doi:10.1155/2019/4180949.

Mingxing Tan and Quoc V. Le. EfficientNet: Rethinking model scaling for convolutional neural networks. In Proc. Int. Conf. Machine Learning (ICML), pages 6105–6114, 2019. arXiv:1905.11946.

Gaël Varoquaux and Veronika Cheplygina. Machine learning for medical imaging: methodological failures and recommendations for the future. npj Digital Medicine, 5(1):48, 2022. doi:10.1038/s41746-022-00592-y.

W. J. Youden. Index for rating diagnostic tests. Cancer, 3(1):32–35, 1950. doi:10.1002/1097- 0142(1950)3:1<32::AID-CNCR2820030106>3.0.CO;2-3.

John R. Zech, Marcus A. Badgeley, Manway Liu, Anthony B. Costa, Joseph J. Titano, and Eric Karl Oermann. Variable generalization performance of a deep learning model to detect pneumonia in chest radiographs: A cross-sectional study. PLOS Medicine, 15(11):e1002683, 2018. doi:10.1371/journal.pmed.1002683.

## Appendix Guide

Appendix A provides complete per model tables for Sections 4.1, 4.2, and 4.5. Appendix B provides transfer diagnostics for Sections 4.5 and 4.3. Appendices C and D document statistical testing and leakage controls. Appendix E provides the underlying discrimination and error plots, Appendix F contains the separate attribution diagnostic discussed in Section 5, Appendix G gives a seed ensemble robustness check, and Appendix H describes reproducibility artifacts.

## A Full Model and Protocol Results

The main text reports only the columns needed for the central comparisons. Table 4 gives the complete fine tuned model results, including $F _ { 1 }$ and AUPRC, so that the shorter Table 1 can be checked without repeating its discussion. Table 5 gives the corresponding frozen versus fine tuned comparison for each pretrained architecture. Table 6 reports the per architecture effect of the default and validation selected thresholds together with ECE before and after temperature scaling. These tables support Sections 4.1, 4.2, and $4 . 5 ;$ the interpretation remains in the main Results section.

Table 4: Full version of Table 1: pneumonia detection on the held-out official test split (mean±s.d. over three seeds, all backbones fine-tuned). Operating point and temperature were selected on the patient-disjoint validation split only.
<table><tr><td>Architecture</td><td>Par. (M)</td><td>Bal. Acc.</td><td>Sens.</td><td>Spec.</td><td>F1</td><td>AUROC</td><td>AUPRC</td><td>ECE</td></tr><tr><td>Custom CNN</td><td>0.5</td><td>0.697±0.032</td><td>0.881±0.027</td><td>0.512±0.076</td><td>0.811±0.015</td><td>0.748±0.042</td><td>0.787±0.048</td><td>0.178±0.044</td></tr><tr><td>ResNet-18</td><td>11.2</td><td>0.768±0.016</td><td>0.967±0.019</td><td>0.569±0.038</td><td>0.870±0.009</td><td>0.909±0.006</td><td>0.930±0.004</td><td> $0 . 1 4 3 { \pm } 0 . 0 2 7$ </td></tr><tr><td>VGG-16</td><td>134.3</td><td>0.801±0.023</td><td>0.996±0.001</td><td>0.606±0.047</td><td>0.893±0.010</td><td>0.957±0.007</td><td>0.966±0.005</td><td> $0 . 1 5 0 { \pm } 0 . 0 1 8$ </td></tr><tr><td>ResNet-50</td><td>23.5</td><td>0.828±0.006</td><td>0.984±0.000</td><td>0.671±0.013</td><td>0.903±0.003</td><td>0.949±0.005</td><td>0.963±0.006</td><td>0.163±0.005</td></tr><tr><td>EfficientNet-B0</td><td>4.0</td><td>0.784±0.011</td><td>0.963±0.007</td><td>0.605±0.029</td><td>0.876±0.004</td><td>0.893±0.008</td><td>0.921±0.010</td><td>0.191±0.001</td></tr><tr><td>DenseNet-121</td><td>7.0</td><td>0.775±0.041</td><td>0.992±0.006</td><td>0.557±0.084</td><td>0.880±0.019</td><td>0.946±0.014</td><td>0.961±0.011</td><td>0.194±0.008</td></tr><tr><td>ConvNeXt-T</td><td>27.8</td><td>0.820±0.035</td><td>0.994±0.003</td><td>0.646±0.071</td><td>0.902±0.016</td><td>0.970±0.008</td><td>0.978±0.006</td><td>0.181±0.016</td></tr><tr><td>ViT-B/16</td><td>85.8</td><td>0.811±0.022</td><td>0.996±0.001</td><td>0.626±0.043</td><td>0.898±0.011</td><td>0.974±0.008</td><td>0.983±0.005</td><td>0.143±0.048</td></tr><tr><td>Swin-T</td><td>27.5</td><td>0.809±0.016</td><td>0.995±0.000</td><td>0.623±0.031</td><td>0.896±0.008</td><td>0.969±0.005</td><td>0.975±0.004</td><td> $0 . 1 7 7 { \scriptstyle \pm 0 . 0 2 8 }$ </td></tr></table>

Bal. Acc.: balanced accuracy. ECE: expected calibration error (15 bins), lower is better. Sensitivity is w.r.t. the pneumonia class.

Table 5: Effect of unfreezing the backbone, per architecture. ‘Frozen’ is the linear-probe protocol; ‘f.-t.’ is the same architecture, schedule, split and augmentation with all weights trainable. Summarised in Fig. 1b. The frozen column reproduces the linear-probe protocol used in the earlier version of this study; the fine-tuned column is the same architecture, schedule and split with all weights trainable.
<table><tr><td>Architecture</td><td colspan="3">Balanced accuracy</td><td colspan="3">F1</td><td colspan="3">AUROC</td></tr><tr><td></td><td>frozen</td><td>f.-t.</td><td>∆</td><td>frozen</td><td>f.-t.</td><td>∆</td><td>frozen</td><td>f.-t.</td><td>∆</td></tr><tr><td>ResNet-18</td><td>0.747</td><td>0.768±0.016</td><td>+0.021</td><td>0.843</td><td>0.870±0.009</td><td>+0.026</td><td>0.846</td><td>0.909±0.006</td><td>+0.063</td></tr><tr><td>VGG-16</td><td>0.777</td><td>0.801±0.023</td><td>+0.024</td><td>0.872</td><td>0.893±0.010</td><td>+0.021</td><td>0.906</td><td>0.957±0.007</td><td>+0.052</td></tr><tr><td>ResNet-50</td><td>0.782</td><td>0.828±0.006</td><td>+0.046</td><td>0.869</td><td>0.903±0.003</td><td>+0.034</td><td>0.909</td><td>0.949±0.005</td><td>+0.040</td></tr><tr><td>EfficientNet-B0</td><td>0.697</td><td>0.784±0.011</td><td>+0.087</td><td>0.819</td><td>0.876±0.004</td><td>+0.057</td><td>0.790</td><td>0.893±0.008</td><td>+0.103</td></tr><tr><td>DenseNet-121</td><td>0.835</td><td>0.775±0.041</td><td>-0.061</td><td>0.903</td><td>0.880±0.019</td><td>-0.023</td><td>0.946</td><td>0.946±0.014</td><td>+0.000</td></tr><tr><td>ConvNeXt-T</td><td>0.827</td><td>0.820±0.035</td><td>-0.007</td><td>0.900</td><td>0.902±0.016</td><td>+0.002</td><td>0.938</td><td>0.970±0.008</td><td>+0.032</td></tr><tr><td>ViT-B/16</td><td>0.830</td><td>0.811±0.022</td><td>-0.019</td><td>0.901</td><td>0.898±0.011</td><td>-0.003</td><td>0.932</td><td>0.974±0.008</td><td>+0.042</td></tr><tr><td>Swin-T</td><td>0.821</td><td>0.809±0.016</td><td>-0.012</td><td>0.898</td><td>0.896±0.008</td><td>-0.002</td><td>0.950</td><td>0.969±0.005</td><td>+0.019</td></tr></table>

Table 6: Decision-threshold selection and temperature scaling, both fitted on validation data only. ‘Default’ uses argmax at t = 0.5 on uncalibrated soft-max scores; ‘tuned’ uses Youden-optimal t on temperature-scaled scores.
<table><tr><td>Architecture</td><td colspan="2">Balanced accuracy</td><td colspan="2">F1</td><td colspan="2">ECE</td></tr><tr><td></td><td>default</td><td>tuned</td><td>default</td><td>tuned</td><td>default</td><td>tuned</td></tr><tr><td>Custom CNN</td><td>0.652±0.052</td><td>0.697±0.032</td><td>0.809±0.013</td><td>0.811±0.015</td><td>0.139±0.079</td><td>0.178±0.044</td></tr><tr><td>ResNet-18</td><td>0.723±0.028</td><td>0.768±0.016</td><td>0.852±0.008</td><td>0.870±0.009</td><td>0.119±0.012</td><td>0.143±0.027</td></tr><tr><td>VGG-16</td><td>0.738±0.034</td><td>0.801±0.023</td><td>0.865±0.015</td><td>0.893±0.010</td><td>0.169±0.022</td><td>0.150±0.018</td></tr><tr><td>ResNet-50</td><td>0.709±0.010</td><td>0.828±0.006</td><td>0.852±0.004</td><td>0.903±0.003</td><td>0.147±0.003</td><td>0.163±0.005</td></tr><tr><td>EfficientNet-B0</td><td>0.642±0.006</td><td>0.784±0.011</td><td>0.823±0.002</td><td>0.876±0.004</td><td>0.248±0.004</td><td>0.191±0.001</td></tr><tr><td>DenseNet-121</td><td>0.679±0.024</td><td>0.775±0.041</td><td>0.839±0.011</td><td>0.880±0.019</td><td>0.173±0.027</td><td>0.194±0.008</td></tr><tr><td>ConvNeXt-T</td><td>0.694±0.023</td><td>0.820±0.035</td><td>0.846±0.009</td><td>0.902±0.016</td><td>0.202±0.020</td><td>0.181±0.016</td></tr><tr><td>ViT-B/16</td><td>0.742±0.064</td><td>0.811±0.022</td><td>0.866±0.028</td><td>0.898±0.011</td><td>0.150±0.053</td><td>0.143±0.048</td></tr><tr><td>Swin-T</td><td>0.699±0.045</td><td>0.809±0.016</td><td>0.848±0.019</td><td>0.896±0.008</td><td>0.193±0.037</td><td>0.177±0.028</td></tr></table>

## B Calibration and Operating Point Transfer

Section 4.5 reports the aggregate transfer effects and Figure 3 shows the threshold and calibration behavior that directly supports those claims. This appendix therefore keeps only the additional numerical decomposition. Table 7 reports the validation threshold, target prevalence threshold from Eq. 7, and the unattainable test oracle. Table 8 separates calibration reliability from resolution and reports the proxy A distances used in Section 4.3. These tables provide values that are useful for verification but are not needed to repeat the main visual result.

Table 7: Where the operating point is fitted, in balanced accuracy on the official test split. ‘Oracle’ is fitted on test and is unattainable in practice; it bounds the transfer gap. Prior matching needs only unlabelled target images and an assumed prevalence, both available at deployment. Brier decomposition and A-distances: Table 8.
<table><tr><td>Threshold fitted on validation (Youden) Threshold matched to the target prior Threshold fitted on test (oracle)</td><td>0.787 0.863 0.870</td></tr><tr><td colspan="2">prior matching recovers 92% of the oracle gap</td></tr></table>

Table 8: Remaining panels of Table 7. $d _ { \mathcal { A } } = 2 ( 1 - 2 \epsilon )$ for a domain discriminator with balanced error ϵ; 0 means indistinguishable, 2 perfectly separable. Panel (b) shows temperature scaling moving reliability the wrong way while leaving resolution intact: the ranking survives, the probabilities do not.

(b) Reliability and resolution decomposition of the Brier score
<table><tr><td></td><td>uncalibrated</td><td>temp.-scaled</td></tr><tr><td>Reliability (↓ better)</td><td>0.0655</td><td>0.0705</td></tr><tr><td>Resolution (↑ better)</td><td>0.1078</td><td>0.1173</td></tr><tr><td colspan="3">(c) Proxy A-distance between partitions</td></tr><tr><td>Training pool vs. test, NORMAL only</td><td></td><td>1.226</td></tr><tr><td>Training pool vs. test, PNEUMONIA only</td><td></td><td>0.395</td></tr><tr><td>Our train vs. our validation (control)</td><td></td><td>0.117</td></tr></table>

## C Statistical Testing of Architecture Differences

The overlapping confidence intervals in Figure 1 show uncertainty for each model separately. Figure 4 complements that view with paired DeLong tests on seed 0, which use the fact that the models are evaluated on the same test cases. The purpose of this analysis is to check the architecture comparison in Section 4.1, not to create a second model ranking. At the available test set size, most differences among the pretrained backbones are not statistically distinguishable.

![](images/b7773928192617f83ab5da2de41d799dafe58b3d9704a35d8088d55027230bf3.jpg)  
Figure 4: Pairwise DeLong tests for AUROC on seed 0. Cells marked ∗ indicate $p < 0 . 0 5$ , and shading represents $\log _ { 1 0 } p .$ . The figure supports the uncertainty discussion in Section 4.1.

## D Corpus Composition and Duplicate Audit

Figure 5 shows representative test images and the class composition of the patient disjoint partitions used in this study. The figure also makes the class imbalance visible, which motivates balanced accuracy in Eq. 4. The duplicate audit is reported separately from the shift analysis because a duplicated image and a shifted distribution can both produce unusual train to test behavior but have different explanations.

![](images/85ba210a006725e7dba4c0c538381429253089972bbe4b5eab5bb38a37103e37.jpg)  
Figure 5: Left: representative official test images from each class. Right: split composition after patient disjoint partitioning and exact duplicate removal. The official test set remains untouched apart from removal of exact duplicates.

Near duplicate criterion. Each image is converted to a $6 4 \times 6 4$ gray scale array $x ,$ mean centered to $\tilde { x } = x - \bar { x } \mathbf { 1 }$ , and $L _ { 2 }$ normalized. Similarity between two images is measured using zero mean normalized cross correlation,

$$
\rho ( x , y ) = \frac { \langle \tilde { x } , \tilde { y } \rangle } { \| \tilde { x } \| _ { 2 } \| \tilde { y } \| _ { 2 } } .\tag{8}
$$

Equation 8 is used only for the leakage audit. We use $\rho \ge 0 . 9 8$ as the primary near duplicate threshold. At that threshold no official test image has a near twin in the training pool. At the looser threshold $\rho \ge 0 . 9 5$ , 14 of 624 released test images are flagged for manual inspection. Exact byte level duplicates number 26 in the released training set and 6 in the released test set and are removed before model training or evaluation. The threshold sweep is reported so that this leakage check is not dependent on one unexplained operating point.

Why we did not use perceptual hashing as the primary audit. A 64 bit difference hash at Hamming radius 5 flags approximately 93% of the test set, which is inconsistent with the verified pixel level comparison above. Chest radiographs share a highly similar global layout, so a coarse gradient hash collides on many distinct images. This sensitivity check is included because a duplicate detector that is inappropriate for the domain could incorrectly turn a distribution shift finding into a leakage claim.

## E Discrimination Curves and Error Structure

Table 4 summarizes scalar performance, while Figure 6 shows the underlying ROC and precisionrecall curves for seed 0. The threshold error structure is already shown in the main Results in Figure 3a, so it is not repeated here. The curves below support the metric discussion in Section 4.1 without introducing a separate result claim.

![](images/5ac3566005f9397b1b4d16139c5269d6a1069fb5b933d16e414ba11daef90120.jpg)

![](images/8459796d377ed3082c7b4ffe62c8aae4a258038e334804b2300ad75c0286d394.jpg)  
Figure 6: ROC curve on the left and precision-recall curve on the right for seed 0. The three highest AUROC architectures are labeled and the remaining models are shown with lower visual emphasis. Precision-recall reporting follows the recommendation to inspect class imbalance sensitive summaries alongside ROC based summaries [Saito and Rehmsmeier, 2015].

## F Supplementary Attribution Audit

The main paper does not use attribution as evidence for the protocol effects. This supplementary analysis instead illustrates the narrower communication point in Section 5: discrimination metrics do not reveal which image regions contribute to a prediction. Figure 7 shows Grad-CAM [Selvaraju et al., 2017] examples and SHAP [Lundberg and Lee, 2017] attribution summaries for a model retrained under the same configuration as the main benchmark runs.

To make the attribution analysis auditable, let $a _ { u }$ denote the attribution assigned to pixel u and let Ω be a body mask defined as the largest bright connected component after Otsu thresholding. We summarize the fraction of absolute attribution mass outside the body as

$$
F _ { \mathrm { o u t } } = \frac { \sum _ { u \notin \Omega } \left| a _ { u } \right| } { \sum _ { u } \left| a _ { u } \right| } .\tag{9}
$$

Equation 9 converts each attribution map to one scalar summary. For radiographs predicted as normal, the mean outside body fraction is 17.0% with standard deviation 7.7%. For radiographs predicted as pneumonia, it is 34.5% with standard deviation 10.3%, at mean confidence 0.997. These values are not used to claim that the model is or is not clinically valid. They show why a performance number and a visual explanation answer different questions.

![](images/150ce3a7f7b5cba5720ff9622a59a0426c790182eaa50db7293058230cbdffd0.jpg)  
Figure 7: Left: Grad-CAM examples on normal test radiographs. Right: SHAP attribution magnitude summarized by class. These visualizations are treated as descriptive diagnostics rather than as causal explanations of model reasoning.

## G Seed Ensemble Robustness Check

Section 4.1 focuses on single model architecture comparisons because that is the communication question of interest. Table 9 provides a separate variance reduction check using only the three seeds already trained. Averaging their probabilities changes AUPRC modestly and does not alter the main conclusion that protocol effects can be larger than architecture differences.

Table 9: Average precision on the official test split, single seed versus an average over the three seeds already trained. Seed ensembling costs no extra training and is selected without reference to test data. AUPRC is the appropriate headline here because the positive class is the majority.
<table><tr><td>Architecture</td><td>single seed</td><td>3-seed ensemble</td><td>∆</td></tr><tr><td>ViT-B/16</td><td>0.9834</td><td>0.9865</td><td>+0.0031</td></tr><tr><td>Swin-T</td><td>0.9750</td><td>0.9798</td><td>+0.0048</td></tr><tr><td>VGG-16</td><td>0.9660</td><td>0.9650</td><td>-0.0010</td></tr><tr><td>ResNet-50</td><td>0.9599</td><td>0.9631</td><td>+0.0032</td></tr><tr><td>ResNet-18</td><td>0.9305</td><td>0.9448</td><td>+0.0143</td></tr><tr><td>EfficientNet-B0</td><td>0.9213</td><td>0.9344</td><td>+0.0131</td></tr></table>

Single-seed figures are the mean over the three seeds; the ensemble averages their temperature-scaled probabilities. The gain is largest for th weakest backbones, which is the expected variance-reduction pattern, and is within noise for VGG-16.

## H Reproducibility

All reported numerical values are generated from stored per case model outputs into the LaTeX macro file used by the manuscript. The analysis stage consumes one metric record per run and one array of logits per run and split, which is sufficient to recompute metrics, operating points, confidence intervals, calibration analyses, and statistical tests without retraining. For an anonymized release, the intended artifacts are the patient disjoint split lists, per case predicted probabilities, the file only feature extractor used in Section 4.4, the partition classifier used in Section 4.3, and scripts that regenerate the tables and figures in this appendix. This organization keeps the numerical claims in the paper tied to reproducible analysis outputs rather than hand transcribed values.