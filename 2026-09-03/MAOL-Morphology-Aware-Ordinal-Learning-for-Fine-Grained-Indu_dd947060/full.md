(f) MAOL grading on clean ROIs

(b) Marginal NG

# MAOL: Morphology-Aware Ordinal Learning for Fine-Grained Industrial Defect Severity Grading

Zhaoyang Wang Hebei University of Technology Tianjin, China 895056337@qq.com

Haiyong Chen Hebei University of Technology Tianjin, China haiyong.chen@hebut.edu.cn

Kun Liu Hebei University of Technology Tianjin, China 2009069@hebut.edu.cn

Kun Wang Yingli Energy Development Co. Ltd Baoding, China 13931278136@163.com

Binyi Su Hebei University of Technology Tianjin, China subinyi@hebut.edu.cn

Xianen Zhou   
Hunan University   
Changsha, China   
zhouxianen@hnu.edu.cn

ATIK SHAHARIAR Hebei University of Technology Tianjin, China atik.shahariar.5@gmail.com

Abstract—Fine-grained defect severity grading is essential for industrial inspection, yet remains challenging due to the ordinal nature of severity labels, the strong dependence on morphologyrelated cues, and the train–test discrepancy between clean annotated instances and noisy predicted instances in two-stage pipelines. We propose MAOL, a Morphology-Aware Ordinal Learning framework for fine-grained industrial defect severity grading. MAOL formulates severity grading as an instancelevel ordinal learning task, incorporates explicit morphological features to enhance representation learning, introduces classconditional adaptive ordinal thresholds to model defect-specific grading boundaries, and employs prediction-aware training via localization perturbation to improve robustness to imperfect predicted instances. Extensive experiments under both clean-ROI and predicted-instance settings demonstrate that MAOL consistently outperforms rule-based methods, nominal classification models, and existing ordinal baselines, especially in the predicted-instance setting. The proposed approach ranked third in the IDA 2026 Challenge on Fine-Grained Severity Grading for High-Precision Manufacturing.

Index Terms—Industrial inspection, Defect severity grading, Ordinal regression

## I. INTRODUCTION

In high-precision manufacturing, industrial visual inspection requires not only detecting defects but also assessing their severity levels to support quality screening, rework decisionmaking, and production control [1], [2]. Compared with conventional binary defect judgment, severity grading provides more fine-grained quality information and plays an increasingly important role in intelligent inspection systems. In practical scenarios, defects of different severity levels correspond to different handling strategies, making accurate and reliable severity prediction a critical yet challenging problem [3].

A straightforward solution is to design rule-based grading systems using morphology-related statistics such as defect area, shape, or contrast. However, such methods heavily rely on handcrafted thresholds and expert knowledge, and often generalize poorly to complex industrial scenarios. As illustrated in Fig. 1(e), even under the clean-ROI setting, a rule-based method exhibits a scattered confusion pattern and achieves only 0.5597 Accuracy and 0.6421 Quadratic Weighted Kappa (QWK) [4], indicating its limited capability in fine-grained severity discrimination.

![](images/bf524c01b7e5a1c5db948e36c786b2176e793e8057a298aef57d35e88865ee60.jpg)

![](images/b232f1c33a6aa7b4c52e66e98f3a8bbb9816037e4e69ebcd99fa36acccbb2ca4.jpg)

![](images/97a63e29a51d7368b8e2394fdbe7fad88953d51f8091df88349264ae51322d26.jpg)

![](images/c95f5412ac4c982f94b50094996d484b653830fa1b9b34a312e86074e037551e.jpg)  
Fig. 1. Illustration of fine-grained industrial defect severity grading. (a)– (d) Representative examples across severity levels. (e) Confusion matrix of a rule-based method under the clean-ROI setting. (f) Confusion matrix of the proposed MAOL under the same setting.

A more common alternative is to formulate severity grading as a classification problem on cropped Regions of Interest (ROIs). However, this formulation is also suboptimal. Severity labels are inherently ordered, and treating them as independent categories ignores the relative relationships between different severity levels. As a result, classification-based models fail to distinguish minor adjacent-level errors from large cross-level mistakes.

Another critical challenge arises from the train–test discrepancy in two-stage inspection pipelines. In practice, severity grading is performed on detector-predicted ROIs rather than perfectly aligned ground-truth regions. These predicted instances inevitably introduce localization errors, background contamination, and incomplete defect regions, which significantly degrade downstream grading performance. Therefore, an effective severity grading framework should jointly model ordinal label structure, preserve morphology-related information, account for category-specific grading differences, and remain robust to imperfect predicted-instance inputs.

To address these challenges, we propose MAOL, a Morphology-Aware Ordinal Learning framework for finegrained industrial defect severity grading. MAOL formulates severity grading as an instance-level ordinal learning problem and is designed as a problem-aligned solution to the above challenges. Specifically, it incorporates explicit morphology descriptors to preserve structural cues, introduces classconditional adaptive ordinal thresholds to model categorydependent grading boundaries, and adopts prediction-aware training via localization perturbation to improve robustness to noisy predicted ROIs.

Extensive experiments under both clean-ROI and predictedinstance settings demonstrate the effectiveness of MAOL. As shown in Fig. 1(f), MAOL achieves 0.935 Acc. and 0.9672 QWK under the clean-ROI setting, with a substantially cleaner confusion pattern than the rule-based baseline. Quantitative results further show that MAOL consistently outperforms rule-based methods, classification-based models, and existing ordinal baselines, with particularly significant improvements under predicted-instance conditions. The main contributions of this work are summarized as follows:

• We formulate fine-grained industrial defect severity grading as an instance-level ordinal learning problem and propose MAOL, a unified framework that explicitly models label ordering.

• We introduce a morphology-aware ordinal learning scheme with class-conditional adaptive thresholds to capture structural cues and category-specific grading criteria.

• We propose a prediction-aware training strategy to bridge the gap between clean annotations and noisy predicted instances, improving robustness in practical two-stage inspection pipelines.

• Extensive experiments in GT and predicted ROI settings demonstrate the superiority of the proposed method. Our MAOL achieves a final score of 0.7965 (S2 metric) in the IDA 2026 Challenge, ranking third among all participants.

## II. RELATED WORK

## A. Industrial Defect Inspection and Severity Assessment

Industrial defect detection and segmentation have long been core research topics in industrial visual inspection. With the rapid development of deep learning, related methods have evolved from traditional hand-crafted features and shallow classifiers to convolutional neural network and Transformerbased frameworks for defect detection, segmentation, and anomaly localization [5], [6]. In contrast, defect severity assessment and quality-oriented evaluation have received comparatively less attention, although they are more challenging downstream tasks that require models not only to identify defects, but also to distinguish fine-grained differences among multiple severity levels.

Existing studies fall mainly into two categories. The first category relies on manually designed rules, where defects are quantified according to statistics such as area, shape, contrast, grayscale intensity, or effective defect region, and then mapped to severity levels using predefined formulas or thresholds [7]. Although such methods are relatively interpretable, they usually depend heavily on human experience and generalize poorly to complex industrial scenarios with large appearance variations and cross-category differences. The second category attempts to directly predict defect severity or product quality using deep neural networks, sometimes jointly with detection or segmentation. Representative examples include the multibranch U-Net [8] for joint defect type and severity segmentation proposed by Neven and Goedeme [9], the DeepLabv3+´ based framework [10] for sewer defect detection and severity quantification introduced by Zhou et al. [11], the defectattribute-based quality assessment network proposed by Zhang et al. [12], and the two-stage defect detection and severity grading model developed by Tian et al. [13].

Although these methods demonstrate the importance of severity-aware inspection, most of them still formulate severity grading as a standard classification or application-specific scoring problem, and only limited work explicitly considers the ordered relationship among severity labels [13]. In contrast, this paper focuses on instance-level, fine-grained industrial defect severity grading with ordinal label structure, while further considering the impact of predicted-instance noise in real two-stage systems.

## B. Ordinal Regression and Ordered Visual Recognition

Ordinal regression explicitly models label ordering and is therefore more suitable than standard classification for severity grading [14]. It distinguishes adjacent-level errors from large cross-level errors, which is desirable when mistake severities are not equally penalized. Related ideas have also been explored in distance-aware classification, where penalties increase with class distance [15]. Fig. 2 illustrates the differences among these formulations.

In deep ordinal learning, representative methods such as CORAL (COnsistent RAnk Logits) and CORN (Conditional Ordinal Regression for Neural networks) have shown strong effectiveness by explicitly modeling rank structure [16], [17]. In particular, CORAL enforces rank consistency across ordinal thresholds, ensuring that predictions follow a valid monotonic ordering. Similar grading-oriented ideas have also been explored in applications such as defect and agricultural quality assessment [18], [19]. However, explicit ordinal learning remains underexplored in industrial defect severity grading, where most existing methods rely on handcrafted rules or distance-aware classification. In this work, we build upon CORAL and further extend it with morphologyaware representation, class-conditional grading boundaries, and prediction-aware training.

![](images/4e8050850f9213ad4c1832269570245ccf56a6c934f329f2c599bc3202fe3d15.jpg)  
Fig. 2. Comparison of three formulations for severity prediction using ROI features: (a) multi-class classification with one-hot supervision, (b) distanceaware classification with class-distance-dependent penalties, and (c) CORAL ordinal regression with cumulative threshold modeling and ordinal binary targets.

## III. METHOD

We address fine-grained industrial defect severity grading in a two-stage inspection pipeline and propose MAOL (Morphology-Aware Ordinal Learning), a unified framework for practical deployment. MAOL formulates severity grading as an instance-level ordinal prediction task and is designed to handle morphology-sensitive cues and deployment-time discrepancies. The overall architecture is illustrated in Fig. 3.

## A. Overall Framework

Given an input image I, an upstream instance segmentation model first detects and segments defect instances. For the $_ { i - }$ th instance, let $r _ { i }$ denote its region representation and $c _ { i }$ its defect category. An ROI is extracted from $r _ { i } ,$ , resized to 64×64 pixels, and fed into a downstream severity grading network, which predicts the severity label $\hat { y } _ { i } \in \{ 0 , 1 , \ldots , K - 1 \}$ }, where K is the number of ordered severity levels. In our setting, $K = 4 ,$ corresponding to Acceptable, Marginal NG, NG, and Gross NG.

We formulate severity grading as an instance-level ordinal prediction problem. As illustrated in Fig. 3, the proposed MAOL framework contains three key components: perturbation-aware training on the ROI image branch, morphology- and category-aware representation learning, and an adaptive CORAL head for ordinal severity grading. Specifically, the resized ROI is processed by a ResNet18 [20] image encoder, while explicit morphology descriptors are extracted in parallel from the defect instance. These features are then fused with category information and passed to the adaptive ordinal head for final severity grading.

## B. Adaptive CORAL Head

Severity labels are inherently ordered. Therefore, we adopt CORAL as the ordinal prediction head and further make its decision boundaries category-adaptive. For each instance with severity label $y _ { i } \in \{ 0 , 1 , \dotsc , K - 1 \}$ , we convert it into an ordinal binary target vector of length $K - 1$

$$
\mathbf { t } _ { i } = [ \mathbb { I } ( y _ { i } > 0 ) , \mathbb { I } ( y _ { i } > 1 ) , \dots , \mathbb { I } ( y _ { i } > K - 2 ) ] .\tag{1}
$$

For the four severity levels in our task, the encoding becomes

$$
0  [ 0 , 0 , 0 ] , ~ 1  [ 1 , 0 , 0 ] , ~ 2  [ 1 , 1 , 0 ] , ~ 3  [ 1 , 1 , 1 ] .\tag{2}
$$

Let $\mathbf { h } _ { i } ~ \in ~ \mathbb { R } ^ { d }$ denote the fused representation of the i-th instance, and let $\mathbf { e } _ { i }$ denote the learnable embedding of its defect category. The adaptive CORAL head first computes a scalar severity score

$$
s _ { i } = \mathbf { w } _ { s } ^ { \top } \mathbf { h } _ { i } ,\tag{3}
$$

which represents the position of the instance on a shared latent severity axis. A globally shared bias vector $\mathbf { b } ^ { g } \in \mathbb { R } ^ { K - 1 }$ defines the base ordinal thresholds, while a class-conditional offset is generated from the category embedding:

$$
\Delta \mathbf { b } ( c _ { i } ) = \mathrm { M L P } ( \mathbf { e } _ { i } ) \in \mathbb { R } ^ { K - 1 } .\tag{4}
$$

The final rank logits are then defined as

$$
z _ { i , j } = s _ { i } + b _ { j } ^ { g } + \Delta b _ { j } ( c _ { i } ) , \qquad j = 0 , \ldots , K - 2 .\tag{5}
$$

We optimize the ordinal prediction using binary cross-entropy over all rank logits:

$$
\mathcal { L } _ { \mathrm { c o r a l } } = \frac { 1 } { N ( K - 1 ) } \sum _ { i = 1 } ^ { N } \sum _ { j = 0 } ^ { K - 2 } \mathrm { B C E } ( \sigma ( z _ { i , j } ) , t _ { i , j } ) ,\tag{6}
$$

where N is the batch size and $\sigma ( \cdot )$ is the sigmoid function. During inference, the final severity label is obtained by counting the number of passed thresholds:

$$
\hat { y } _ { i } = \sum _ { j = 0 } ^ { K - 2 } \mathbb { I } ( \sigma ( z _ { i , j } ) > 0 . 5 ) .\tag{7}
$$

This design jointly models a shared ordinal severity axis and category-adaptive decision boundaries, thereby enabling more accurate capture of category-specific grading criteria than fixed-threshold ordinal predictors.

## C. Morphology Branch and Class Embedding

ROI appearance alone may not fully capture the structural cues that are highly correlated with defect severity. In industrial scenarios, severity is strongly related to geometric and intensity-related properties such as defect size, shape, and contrast. Moreover, such cues can be distorted by ROI resizing, making them difficult to learn purely from appearance features.

To address this issue, we introduce a morphology branch together with a learnable class embedding module. As illustrated in Fig. 3, the cropped ROI is resized to $6 4 \times 6 4$ and encoded by a ResNet18 backbone to obtain an image feature $\mathbf { f } _ { i } ^ { \mathrm { i m g } } \in \mathbb { R } ^ { 5 1 2 }$ . In parallel, a set of morphology descriptors capturing defect size, geometry, and contrast is extracted and projected into a low-dimensional embedding $\mathbf { f } _ { i } ^ { \mathrm { m o r p h } }$ using a lightweight encoder.

![](images/b9306cebc360708acd625fb7165c474fdce74f084421a7a88c99d289c77a5536.jpg)  
Fig. 3. Overview of the proposed MAOL framework, illustrating perturbation-aware training, morphology-aware representation, and adaptive ordinal prediction.

In addition, the defect category $c _ { i }$ is mapped to a learnable embedding $\mathbf { e } _ { i } = \operatorname { E m b } ( c _ { i } )$ . The final representation is obtained by feature fusion:

$$
\mathbf { h } _ { i } = \phi \left( [ \mathbf { f } _ { i } ^ { \mathrm { i m g } } ; \mathbf { f } _ { i } ^ { \mathrm { m o r p h } } ; \mathbf { e } _ { i } ] \right) ,\tag{8}
$$

where $\phi ( \cdot )$ denotes a learnable fusion module. This design enables the model to jointly leverage appearance features, scale-consistent morphology cues, and category-level context, leading to more reliable ordinal severity grading.

## D. Perturbation-aware Training

In practical two-stage inspection systems, severity grading operates on detector-predicted ROIs rather than perfectly aligned ground-truth regions, leading to a distribution shift caused by localization errors such as offsets, scale variations, truncations, and background contamination.

To model this deployment-time discrepancy, we introduce a perturbation-aware training strategy on the ROI image branch. As illustrated in Fig. 3, during training, the GT box is randomly perturbed with probability $p \ = \ 0 . 5$ by applying small independent offsets to its boundaries, resulting in a perturbed box $\tilde { b } _ { i }$ . The perturbation is constrained by an IoU range $[ \tau _ { \operatorname* { m i n } } , \tau _ { \operatorname* { m a x } } ]$ (0.75–0.90) to ensure realistic deviations. The perturbed box is used to crop the ROI for the image branch, while morphology features are computed from the original instance. Optionally, contrast-related morphology cues can be recomputed from the perturbed ROI. This design simulates detector-induced noise during training and improves robustness to localization errors in real-world deployment, without altering the severity labels.

## IV. EXPERIMENTS

## A. Datasets And Experimental Details

We evaluate MAOL on the severity grading track of an industrial defect inspection benchmark. Each defect instance is annotated with a category, a segmentation mask, and an ordinal severity label from four ordered levels: Acceptable, Marginal NG, NG, and Gross NG. The dataset is split at the image level into training and validation sets, as summarized in Table I. The training split contains 924 images and 1992 instances, while the validation split contains 230 images and 554 instances. The severity distribution is imbalanced, making the task challenging.

TABLE I  
TRAINING AND VALIDATION SPLIT STATISTICS FOR SEVERITY GRADING.
<table><tr><td>Split</td><td>#Img.</td><td>#Inst.</td><td>Acceptable</td><td>Marginal NG</td><td>NG</td><td>Gross NG</td></tr><tr><td>Train</td><td>924</td><td>1992</td><td>421</td><td>426</td><td>842</td><td>303</td></tr><tr><td>Val</td><td>230</td><td>554</td><td>120</td><td>114</td><td>238</td><td>82</td></tr></table>

We consider two settings: Clean ROI, where ROIs are cropped from GT annotations, and Predicted ROI, where ROIs are obtained from a first-stage detector. The latter is more realistic but more difficult due to localization errors. In this setting, all morphology descriptors are likewise computed from the detector outputs.

The first-stage detector is YOLOv8X-seg [21], trained for 300 epochs with input size $5 1 2 \times 5 1 2$ and batch size 32. For severity grading, each ROI is resized to $6 4 \times 6 4 .$ , and ResNet18 is used as the image backbone. MAOL is trained for 300 epochs with batch size 64. All experiments are conducted on an NVIDIA RTX 4090 GPU. Unless otherwise specified, the best model is selected by validation QWK. Performance is reported using Accuracy (Acc.), Macro-average Accuracy (Macro Acc.), and Quadratic Weighted Kappa (QWK) [4].

## B. Results And Comparisons

1) Quantitative Results: The first-stage detector achieves 86.5% mAP@0.5 (box) and 84.1% mAP@0.5 (mask) on the val set. Table II summarizes the comparison of severity grading methods under the Clean ROI and Predicted ROI settings. First, learning-based methods consistently outperform the rulebased baseline, with a much larger gap under Predicted ROI (29.7% Acc. / 27.4 QWK), indicating its sensitivity to localization errors. Second, all methods degrade under Predicted ROI due to detector-induced noise such as offsets, scale mismatch, and background contamination. Third, distance-aware classification improves over standard classification, confirming the benefit of incorporating class-distance information. CORAL further achieves the best performance among the baselines, demonstrating the advantage of explicitly modeling ordinal structure. In contrast, CORN performs worse, especially under

TABLE II  
OVERALL COMPARISON OF DIFFERENT SEVERITY GRADING METHODS UNDER CLEAN AND PREDICTED ROI SETTINGS.
<table><tr><td rowspan="2">Method</td><td colspan="3">Clean ROI (GT)</td><td colspan="3">Predicted ROI</td></tr><tr><td>Acc.</td><td>Macro Acc.</td><td>QWK</td><td>Acc.</td><td>Macro Acc.</td><td>QWK</td></tr><tr><td>Rule-based</td><td>56.0</td><td>49.8</td><td>64.2</td><td>29.7</td><td>31.8</td><td>27.4</td></tr><tr><td>Classification Head</td><td>70.7</td><td>67.3</td><td>77.7</td><td>34.0</td><td>45.4</td><td>34.5</td></tr><tr><td>Distance-aware Head</td><td>78.7</td><td>76.8</td><td>87.3</td><td>35.9</td><td>46.9</td><td>46.6</td></tr><tr><td>CORAL [16]</td><td>80.8</td><td>81.2</td><td>88.8</td><td>55.8</td><td>42.1</td><td>51.1</td></tr><tr><td>CORN [17]</td><td>69.8</td><td>68.5</td><td>78.7</td><td>39.3</td><td>31.7</td><td>25.9</td></tr><tr><td>MAOL (Ours)</td><td>93.5</td><td>94.4</td><td>96.7</td><td>84.8</td><td>81.2</td><td>88.9</td></tr></table>

TABLE III  
COMPARISON OF ENHANCED ORDINAL GRADING MODELS BUILT ON TOPOF THE CORAL BASELINE UNDER THE PREDICTED-INSTANCE SETTING.
<table><tr><td>Strategy</td><td>Acc.</td><td>Macro Acc.</td><td>QWK</td></tr><tr><td>CORAL baseline</td><td>55.8</td><td>42.1</td><td>51.1</td></tr><tr><td>+ Morphology</td><td>72.5</td><td>69.9</td><td>76.4</td></tr><tr><td>+ Class embedding</td><td>75.9</td><td>72.8</td><td>80.5</td></tr><tr><td>+ Adaptive thresholds</td><td>76.5</td><td>75.8</td><td>84.2</td></tr><tr><td>+ Perturbation (ours)</td><td>84.8</td><td>81.2</td><td>88.9</td></tr></table>

Predicted ROI (25.9 QWK), likely due to its sensitivity to limited data, class imbalance, and ROI noise. Finally, MAOL achieves the best results by a large margin in both settings. Notably, under Predicted ROI, it surpasses the baseline by +51.1 Acc., +46.0 Macro Acc., and +61.5 QWK, highlighting the effectiveness of morphology-aware representation, adaptive thresholds, and prediction-aware training.

2) Qualitative Analysis: Fig. 4 shows a qualitative comparison under clean and detector-generated ROI conditions. The first row presents the original PCB images with groundtruth annotations, the second row shows detector-predicted boxes, and the third row compares severity results based on predicted ROIs. In the first example, the predicted box is close to the ground truth, and most learning-based methods produce correct predictions, while the rule-based method fails, indicating its limited robustness. In the second example, larger localization errors introduce ROI contamination; only MAOL correctly predicts Gross NG, whereas other methods underestimate it, demonstrating the advantage of our method under noisy conditions. The third example illustrates a severe failure case, where the defect is split into multiple predicted boxes, leading to incorrect predictions for all methods. This highlights a fundamental limitation of the two-stage paradigm, as performance depends on detection quality. Nevertheless, MAOL produces the least severe error, indicating improved robustness compared to other approaches.

## C. Ablation Study and Threshold Analysis

1) Ablation Study: To analyze the contribution of each component, we conduct a progressive ablation study starting from the CORAL baseline, as shown in Table III. Components are added sequentially, including the morphology branch, class embedding, adaptive thresholds, and perturbation training.

![](images/f42345a0e771b976421d7573bdd767925853eb07ac8e03ba8b9958dd4c9f29df.jpg)  
Fig. 4. Qualitative comparison under predicted ROI inputs. Row 1: original PCB defect images with GT boxes and masks. Row 2: the same images with detector-predicted boxes. Row 3: results of different methods based on the corresponding predicted ROIs.

Several observations can be made. First, the morphology branch yields the largest improvement (about +25 QWK), highlighting the importance of explicit structural cues, which are not fully captured by ROI appearance features. Second, the class embedding provides additional but moderate gains, mainly serving as category-aware context for adaptive threshold modeling. Third, adaptive thresholds further improve performance, confirming that severity boundaries are defectdependent rather than globally shared. Finally, perturbation training brings additional gains by reducing the train–test discrepancy between clean and predicted ROIs, improving robustness in practical deployment.

Overall, the components are complementary: morphology provides the dominant gain, adaptive thresholds further enhance ordinal modeling, and perturbation training improves robustness under noisy ROI conditions.

![](images/2eefe4e3e454a4e87477862c01972d424bfef74ddfe1aa2b26867adcd180db11.jpg)  
Fig. 5. Class-specific adaptive ordinal thresholds for seven defect categories. Box plots show the learned values of the three adaptive thresholds for each category, while the horizontal dashed lines indicate the corresponding category-agnostic thresholds learned without adaptive thresholding.

2) Analysis of Adaptive Thresholds: To further analyze the necessity of adaptive thresholds, Fig. 5 visualizes the learned class-specific ordinal thresholds for seven defect categories, with global thresholds shown as dashed lines for reference. The results show that thresholds vary significantly across categories, indicating that severity transition patterns are not uniformly shared and cannot be adequately modeled by a single set of global thresholds. Moreover, the relative spacing between thresholds also differs, suggesting that ordinal intervals between severity levels are category-dependent. This implies that the same severity level may correspond to different physical defect extents across categories. Therefore, adaptive thresholds do not simply introduce an additive bias, but effectively reshape category-specific decision boundaries in the latent ordinal space.

## V. CONCLUSION

This paper studied fine-grained industrial defect severity grading as an instance-level ordinal prediction problem. Instead of treating severity levels as independent categories, we argued that severity grading should explicitly account for the ordered structure of labels, the morphology-related characteristics of defects, and the train–test mismatch between clean training ROIs and noisy detector-generated instances. To this end, we proposed MAOL, a morphology-aware ordinal learning framework for industrial defect severity grading. The proposed method incorporates three key designs: a morphology branch for explicit structural cues, a class-conditional adaptive ordinal head for category-dependent severity boundaries, and a perturbation-based training strategy to improve robustness under predicted-ROI inputs. Experimental results under both clean-ROI and predicted-instance settings demonstrated that the proposed framework consistently outperforms rule-based, nominal classification, and standard ordinal baselines, with particularly clear advantages in the more practical predictedinstance scenario.

The results validate the importance of combining ordinal modeling, morphology-aware representation, and robustnessoriented training for industrial severity grading. In future work, we will explore end-to-end frameworks that jointly optimize defect localization and ordinal severity grading within a unified model.

## REFERENCES

[1] J. Xu, J. Tang, Y. Jin, J. Liu, and Z. Gong, “Rethinking steel surface defect segmentation with pseudo mixup and self distillation,” in 2025 IEEE International Conference on Multimedia and Expo (ICME), 2025, pp. 1–6.

[2] Z. Wang, H. Chen, and Z. Cao, “CPDD: A cross-scenario photovoltaic defect detector based on fine-grained feature autoencoding and pseudobox contrastive learning,” IEEE Transactions on Semiconductor Manufacturing, vol. 38, no. 3, pp. 612–623, 2025.

[3] L. Li, P. Wang, Z. Lu, R. Di, C. He, G. Wang, and X. Li, “A defect severity assessment method based on empirical feature attribute scoring from semantic segmentation data for reliability analysis of steel structure connections,” Advanced Engineering Informatics, vol. 68, p. 103743, 2025.

[4] E. Ben-David, “Comparing learning machines for ordinal data classification,” IEEE Transactions on Knowledge and Data Engineering, vol. 20, no. 11, pp. 1555–1560, 2008.

[5] X. Lv, H. Chen, and Z. Wang, “Cross domain anomaly detection in semiconductor manufacturing via projection comparison and reconstruction in latent space,” IEEE Transactions on Semiconductor Manufacturing, pp. 1–1, 2026.

[6] T. Liu, G.-Z. Cao, Z. He, and S. Xie, “Refined defect detector with deformable transformer and pyramid feature fusion for PCB detection,” IEEE Transactions on Instrumentation and Measurement, vol. 73, pp. 1–11, 2024.

[7] Y. Liu, M. Sang, Y. Liu, and H. Lin, “Quantitative evaluation method for the severity of surface fuzz defects in carbon fiber prepreg,” Applied Sciences, vol. 15, no. 13, p. 7478, 2025.

[8] O. Ronneberger, P. Fischer, and T. Brox, “U-Net: Convolutional networks for biomedical image segmentation,” in Medical Image Computing and Computer-Assisted Intervention, vol. 9351 of Lecture Notes in Computer Science, pp. 234–241, Springer, Cham, 2015.

[9] R. Neven and T. Goedeme, “A multi-branch U-Net for steel surface defect type and severity segmentation,” Metals, vol. 11, no. 6, p. 870, 2021.

[10] L.-C. Chen, Y. Zhu, G. Papandreou, F. Schroff, and H. Adam, “Encoderdecoder with atrous separable convolution for semantic image segmentation,” in Proceedings of the European Conference on Computer Vision (ECCV), 2018, pp. 801–818.

[11] Q. Zhou, Z. Situ, S. Teng, H. Liu, W. Chen, and G. Chen, “Automatic sewer defect detection and severity quantification based on pixel-level semantic segmentation,” Tunnelling and Underground Space Technology, vol. 123, p. 104403, 2022.

[12] G. Zhang, Y. Lu, X. Jiang, F. Yan, and M. Xu, “Industrial product quality assessment using deep learning with defect attributes,” Pattern Recognition Letters, vol. 188, pp. 67–73, 2025.

[13] K. Tian, S. Yang, L. Chen, K. Wang, Y. Ke, Z. Miao, J. Hu, C. Sun, and X. Zhang, “DDSPNet: A two-stage defect detection and severity prediction model for industrial products,” Applied Intelligence, vol. 55, no. 15, p. 1024, 2025.

[14] J. Wang, J. Chen, J. Liu, D. Tang, D. Z. Chen, and J. Wu, “A survey on ordinal regression: Applications, advances and prospects,” arXiv preprint arXiv:2503.00952, 2025.

[15] G. Polat, U. M. Caglar, and A. Temizel, “Class distance weighted cross entropy loss for classification of disease severity,” Expert Systems with Applications, vol. 269, p. 126372, 2025.

[16] W. Cao, V. Mirjalili, and S. Raschka, “Rank consistent ordinal regression for neural networks with application to age estimation,” Pattern Recognition Letters, vol. 140, pp. 325–331, 2020.

[17] X. Shi, W. Cao, and S. Raschka, “Deep neural networks for rankconsistent ordinal regression based on conditional probabilities,” Pattern Analysis and Applications, vol. 26, no. 3, pp. 941–955, 2023.

[18] D. F. Al Riza, S. Widodo, K. Yamamoto, K. Ninomiya, T. Suzuki, Y. Ogawa, and N. Kondo, “External defects and severity level evaluation of potato using single and multispectral imaging in near infrared region,” Information Processing in Agriculture, vol. 11, no. 1, pp. 80–90, 2024.

[19] M. Knott, D. Odion, S. Sontakke, A. Karwa, and T. Defraeye, “Weakly supervised panoptic segmentation for defect-based grading of fresh produce,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 5462–5471.

[20] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2016, pp. 770–778.

[21] Ultralytics, “Ultralytics YOLOv8,” https://github.com/ultralytics/ ultralytics, 2023.