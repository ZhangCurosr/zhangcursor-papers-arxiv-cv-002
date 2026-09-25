# When Misalignment Becomes Supervision: Structured Label Noise in Supervised Synthetic CT Generation

Valentin Boussot<sup>†</sup>, Cedric H ´ emon ´ <sup>†</sup>, Caroline Lafond, Jean-Claude Nunes, Jean-Louis Dillenseger

Abstract—Supervised synthetic CT generation is commonly formulated as a voxel-wise regression problem between MRI or CBCT inputs and registered reference CT images. This formulation assumes that paired images are spatially aligned at the voxel level, although in practice multimodal pairs are constructed through registration procedures that inevitably leave residual anatomical misalignments. These residual errors are not independent intensity noise, but spatially coherent geometric discrepancies that can act as structured label noise during training.

In this work, we investigate how registration-induced supervision bias affects supervised MRI-to-CT and CBCT-to-CT synthesis. We show that quantitative performance strongly depends on the consistency between the registration strategy used to construct the training targets and the one used during evaluation. Models achieve better voxel-wise scores when both conventions match, indicating that supervised synthesis networks can partially learn the geometric convention imposed by the registration pipeline. This effect also impacts out-of-distribution robustness and predictive uncertainty, with less geometrically consistent supervision leading to larger prediction variability.

This work does not aim to solve registration errors through a new synthesis architecture, but to demonstrate that residual registration defines a supervision convention that supervised networks can learn and that standard metrics can reward. We validate this analysis on 1,784 paired patients from six clinical centers, covering MRI-to-CT and CBCT-to-CT synthesis across five anatomical regions. To mitigate the limitations of purely voxel-wise supervision, we introduce a SAM-based perceptual loss that compares synthesized and reference CT images in the feature space of a pretrained Segment Anything encoder. Compared with MAE-only and VGG-based perceptual objectives, this supervision improves downstream anatomical metrics and produces sharper, more structurally coherent synthetic CT images. We further show that perceptual and voxel-wise metrics may disagree when references are imperfectly aligned, while becoming more consistent when the evaluation geometry is reliable.

Overall, this study identifies registration-induced bias as a central confounder in supervised synthetic CT generation. Our results argue that synthetic CT methods should not be evaluated solely through voxel-wise agreement with imperfect reference images, but also through anatomy-oriented criteria that assess whether patient-specific structures are faithfully preserved.

Index Terms—Synthetic CT, image-to-image translation, registration bias, structured label noise, perceptual loss, Segment Anything Model, multimodal medical imaging.

## I. INTRODUCTION

Synthetic CT generation aims to transform MRI or CBCT images into CT-like images that can be used by downstream tools originally designed for CT images, including dose calculation, segmentation, and image registration [1], [2]. In recent years, supervised image-to-image translation has become the dominant paradigm for this task, largely driven by public benchmarks and large paired datasets. In this setting, models are trained to minimize voxel-wise reconstruction losses between the synthesized CT and a reference CT, and performance is commonly assessed using intensity-based metrics such as MAE, PSNR, and SSIM.

This indicator-based approach raises a Goodhart-type concern: when an indicator becomes the target of optimization, it may no longer be a reliable indicator of the underlying objective it was intended to measure [3]. In synthetic CT image generation, the goal is not simply to reproduce a reference image voxel by voxel, but to produce a CT-like image that preserves the patient-specific anatomy of the source modality. If the reference used for supervision and evaluation is imperfect, optimizing voxel-level scores may therefore prioritize consistency with the reference’s construction process rather than anatomical fidelity.

This issue is particularly relevant because the supervised formulation relies on a strong geometric assumption: the input image and the reference CT must be spatially aligned at the voxel level. In practice, MRI-CT and CBCT-CT pairs are acquired at different times, under different acquisition conditions, and often in different anatomical configurations [1], [4]. The reference CT used for supervision is therefore not a true voxel-wise ground truth, but a registered target that may contain residual alignment errors. These residuals do not behave as independent intensity noise. They correspond to spatially coherent anatomical displacements and can therefore be interpreted as structured geometric label noise [5], [6], [2].

This has important consequences for both training and evaluation. During training, voxel-wise losses encourage the model to reproduce the statistical structure of the registered targets, including residual deformations and systematic biases introduced by the registration pipeline [7], [8], [9]. During evaluation, reference-based metrics may reward agreement with a particular registration convention rather than faithful preservation of the source anatomy [10]. As a result, a model may achieve strong quantitative scores while partially learning or reproducing misalignment patterns embedded in the super-

vised data.

In this work, we investigate registration-induced supervision bias in supervised synthetic CT generation. Our primary objective is to determine whether residual registration defines a geometric convention that synthesis networks can learn and that reference-based metrics can subsequently reward. We analyze how the registration method used to construct training and evaluation pairs affects quantitative performance, out-ofdistribution (OOD) behavior, and predictive uncertainty [11]. This analysis tests whether measured performance reflects synthesis fidelity alone or also agreement with the registration pipeline.

To mitigate the limitations of purely voxel-wise supervision, we further introduce a SAM-based perceptual loss for synthetic CT generation. Instead of comparing images only at the intensity level, this loss compares synthesized and reference CT images in the feature space of a pretrained Segment Anything encoder [12]. It is intended to promote spatially coherent anatomical structures and reduce regression-induced blurring, but it does not eliminate the systematic geometric bias contained in imperfectly registered training targets.

Overall, this study argues that synthetic CT generation should not be evaluated solely as an intensity regression problem. When reference images are imperfectly aligned, voxel-wise metrics can become confounded by registration errors and may reward conformity to a registration convention rather than preservation of the source anatomy. SAM-based perceptual supervision is therefore investigated as a partial anatomy-oriented response within the broader analysis of this supervision and evaluation bias.

## Contributions

The main contributions of this work are as follows:

• We formulate residual registration errors as structured geometric label noise in voxel-wise supervised synthetic CT generation, distinguishing their variable and systematic effects on the learned target distribution.

• We empirically demonstrate that voxel-wise performance depends on the consistency between the registration conventions used to construct training targets and evaluation references, showing that standard metrics can reward agreement with the registration pipeline.

• We analyze the consequences of registration-induced supervision bias for source-anatomy preservation, regression-induced blurring, OOD generalization, and predictive uncertainty.

• We introduce and evaluate SAM-based perceptual supervision and a separately calibrated SAM-based evaluation metric as complementary anatomy-oriented tools beyond direct voxel-wise agreement.

These contributions are evaluated on 1,784 paired patients from six clinical centers, covering MR→CT and CBCT→CT synthesis across five anatomical regions, for a total of 196,845 image slices.

## II. BACKGROUND

## A. Image-to-Image Translation: Problem Formulation

Image-to-image translation transforms an image from a source domain into a target-domain representation while preserving the task-relevant content of the input [13], [14]. In synthetic CT generation, this corresponds to predicting CT-like images from MRI or CBCT acquisitions for applications such as dose calculation, segmentation, and image registration [15], [16], [17], [18]. This study focuses on supervised conditional synthesis, in which paired images provide voxel-level targets for model training.

a) Supervised translation as direct regression: When spatially aligned pairs are available, synthesis models are commonly optimized using direct reconstruction losses such as the mean absolute error (MAE) or mean squared error (MSE).

From a statistical perspective, such objectives correspond to empirical risk minimization under pointwise loss functions. Let $x \in X$ denote an input image and $Y \sim p ( y \mid x )$ the corresponding target random variable. Under a quadratic loss, the optimal predictor is the conditional expectation:

$$
f ^ { * } ( x ) = \mathbb { E } [ Y \mid X = x ] ,\tag{1}
$$

whereas an $\ell _ { 1 }$ loss yields the conditional median [19], [20].

In image synthesis, voxel-wise regression losses are well known to produce oversmoothed predictions. This effect arises because pointwise objectives collapse the variability of the conditional distribution $p ( y \mid x )$ into a single central-tendency estimate. Under common modeling assumptions, such objectives can also be interpreted as maximum-likelihood estimators under simple pixel-wise residual models, corresponding to independent Gaussian variability for $\ell _ { 2 }$ objectives and Laplacian variability for $\ell _ { 1 }$ objectives. Limited model capacity may also attenuate high-frequency details, but this approximation effect is conceptually distinct from uncertainty contained in the supervision targets.

b) Intrinsic modality ambiguity: Cross-modality relationships can exhibit intrinsic ambiguity due to non-bijective signal formation, such as modality-specific contrast, partial volume effects, or artifacts, which induces a non-degenerate conditional distribution $p ( y \mid x )$ . In the settings considered here, anatomical correspondence is locally well constrained under ideal alignment, so this intrinsic variability is expected to remain limited and voxel-wise regression would not by itself induce severe blurring [21].

c) Registration-induced supervision uncertainty: Voxelwise supervision assumes that the input and target images are spatially aligned. In practice, the available training target is a registered reference y˜ that may differ from the ideal target y because of residual registration errors:

$$
\tilde { y } = y \circ \phi ,\tag{2}
$$

where ◦ denotes the spatial resampling operator and ϕ denotes the residual deformation field induced by imperfect registration. Training is consequently performed on the observed distribution $p ( \tilde { y } \mid x )$ rather than on the ideal distribution $p ( y \mid x )$

Registration residuals do not behave as independent voxelwise intensity noise. They produce spatially coherent displacements of anatomical structures and thus introduce structured geometric label noise into the training targets. Moreover, the residual deformation field $\phi$ is not necessarily purely random: it can contain pair-specific variability as well as systematic contributions determined by the similarity metric, regularization strategy, deformation model, or acquisition protocol.

These residuals affect $p ( \tilde { y } \mid \ x )$ in two complementary ways. Their variable component increases the dispersion of the observed target distribution because similar local input configurations may be associated with target structures displaced differently across training pairs. Voxel-wise regression, which estimates a central tendency of this distribution, may consequently produce spatially averaged or blurred boundaries. Their systematic component can instead shift the center of the distribution when the registration pipeline consistently favors a particular geometric convention. The synthesis model may then learn this convention and reproduce the spatial behavior of the registration method used to construct the training data.

Registration-induced target uncertainty therefore provides an additional geometric explanation for the blurring observed in supervised synthesis, beyond intrinsic modality ambiguity. It cannot be resolved simply by increasing model capacity because the uncertainty lies in the training targets themselves. This setting was formalized as supervised learning with noisy labels by Kong et al. [6], who proposed a joint registrationsynthesis strategy that can theoretically recover the optimal clean-data solution under regularity assumptions on ϕ. Their analysis, however, focuses on loss correction rather than on how uncorrected training reshapes $p ( \tilde { y } \mid x )$ and affects uncertainty estimation and metric interpretation.

This form of uncertainty is distinct from both aleatoric and epistemic uncertainty. It is not inherent to the crossmodality mapping and does not primarily reflect limited model knowledge; instead, it originates from the data-construction pipeline. This distinction is rarely made explicit in the supervised synthesis literature. Blurring under MAE or MSE is commonly attributed to regression over an intrinsically ambiguous conditional distribution, whereas part of the observed ambiguity may arise from structured geometric inconsistencies in the registered targets.

d) Alleviating oversmoothing through perceptual and generative objectives: One strategy for reducing oversmoothing is to augment voxel-wise reconstruction losses with a perceptual term computed in the feature space of a pretrained network [22]. In its original formulation, the feature extractor is typically a VGG network trained on large-scale naturalimage classification. The perceptual loss can be written as:

$$
\mathcal { L } _ { \mathrm { p e r c e p t u a l } } = \| \psi ( \hat { y } ) - \psi ( y ) \| _ { 1 } ,\tag{3}
$$

where $\psi$ denotes the feature extractor and $\hat { y } ~ = ~ G _ { \theta } ( x )$ the synthesized image. $\mathbf { A } \mathbf { s }$ an $\ell _ { 1 }$ objective in feature space, this loss can be interpreted as estimating a conditional median with respect to the representation induced by $\psi ,$ rather than directly in the voxel domain. By comparing representations that encode contextual and multi-scale structure, it can promote spatially coherent anatomical patterns and reduce the oversmoothing associated with voxel-wise regression.

The behavior of a perceptual loss nevertheless depends critically on the selected feature representation. Features learned from natural-image classification may not optimally encode the structures relevant to medical imaging. In this study, SAM embeddings are used because they provide multi-scale representations of anatomical structures and boundaries. The resulting distance is intended to complement voxel-wise metrics by being less sensitive to small residual displacements while remaining responsive to meaningful structural differences, as detailed in Section III-C.

Distribution-level objectives, including adversarial and diffusion-based approaches, can similarly improve sharpness and visual realism by counteracting the averaging behavior of voxel-wise regression [23], [24]. However, perceptual and generative objectives do not remove the systematic component of registration-induced uncertainty when they are trained on the same imperfectly aligned pairs. They still learn from $p ( \tilde { y } \mid x )$ rather than from $p ( y \mid x )$ and may therefore reproduce biases embedded in the registration pipeline.

e) Consequences for synthetic $C T$ evaluation: Voxelwise supervised translation is well suited to settings with reliable spatial correspondence. When residual registration errors are structured or systematic, however, regression objectives may both smooth uncertain boundaries and shift anatomical structures toward the geometric convention imposed by the registration pipeline.

This limitation remains insufficiently characterized in the recent literature [5], [25], [26], where evaluation protocols predominantly emphasize intensity-based similarity rather than anatomical faithfulness. Benchmarking may consequently favor models that reproduce registration-induced distortions rather than models that best preserve clinically relevant structures. This is particularly problematic when the clinical motivation for synthetic CT is to avoid uncertain inter-modality registration, for example by generating CT-like images directly from MRI.

Improving spatial correspondence is a natural response, and several methods jointly optimize synthesis and registration [6], [8], [27]. Nevertheless, residual misalignment remains difficult to eliminate completely, especially under large anatomical deformations. Unsupervised or weakly paired methods relax the requirement for exact voxel-wise correspondence, but introduce other challenges related to anatomical consistency, training stability, and quantitative validation. The present study therefore focuses on quantifying registration-induced bias in supervised synthesis and examining its consequences for uncertainty estimation and sCT evaluation, as further discussed in Section V.

## III. SUPERVISED CROSS-MODALITY IMAGE SYNTHESIS: EXPERIMENTAL STUDY ON SYNTHRAD

In this section, we consider the standard supervised formulation of cross-modality image synthesis, as commonly adopted in recent benchmarks. Given spatially aligned image pairs, a model is trained to predict CT images from input CBCT or MRI data using voxel-wise reconstruction losses.

This setting constitutes the dominant paradigm in current evaluation frameworks, where performance is primarily assessed through intensity-based similarity metrics computed with respect to a reference CT.

## A. SynthRAD challenge and datasets

Experiments were conducted using the SynthRAD2023 and SynthRAD2025 datasets [28], [29], which contain paired images for two synthesis tasks: MR-to-CT (Task 1) and CBCTto-CT (Task 2). The datasets cover brain, head-and-neck, thoracic, abdominal, and pelvic anatomies, depending on the task and challenge edition. Training pairs were provided after rigid registration, whereas the hidden validation and test sets enabled evaluation under the official challenge protocol. This structure makes the dataset particularly suitable for studying how the choice of registration convention affects both model training and performance assessment.

The official image-similarity metrics were mean absolute error (MAE), peak signal-to-noise ratio (PSNR), and multiscale structural similarity (MS-SSIM); the local analyses report SSIM. Dose-based metrics were also reported by the challenge but were used here only to document the official benchmark performance. Dataset composition and complete evaluation details are provided in Appendix A and in the SynthRAD references [28], [29].

## B. Experimental setup

The overall study design is summarized in Fig. 1, which links the construction of registered supervision targets, the supervised synthesis model, and the complementary evaluation analyses used to assess registration-induced bias.

a) Data and registration conventions: Although the dataset is provided as paired multimodal acquisitions, the correspondence between modalities is not intrinsically voxelwise. MRI, CBCT, and CT scans are acquired at different time points and under different physiological conditions, leading to non-negligible anatomical discrepancies.

To study the impact of spatial correspondence on supervised synthesis while remaining consistent with the official challenge evaluation protocol, we investigate two different registration strategies for constructing the training pairs.

First, we consider the registration pipeline used for the test set by the challenge organizers. This approach relies on a multi-resolution deformable registration framework implemented in Elastix [30], using a mutual-information-based similarity metric representative of standard practice in radiotherapy workflows and supervised sCT synthesis studies. In the remainder of this manuscript, this registration configuration is referred to as ELX.

Second, we investigate an alternative registration strategy based on IMPACT-Reg [31], a feature-based multimodal registration method leveraging deep semantic representations. Unlike intensity-based approaches, IMPACT-Reg relies on high-level feature correspondences to estimate multimodal anatomical alignment. In the remainder of this manuscript, this registration configuration is referred to as IMPACT.

Neither ELX nor IMPACT is assumed to provide groundtruth anatomical correspondence. They are treated as two alternative registration conventions whose residual errors may differ. IMPACT is used here because its alignments show higher anatomical consistency than ELX according to the segmentation-based analyses reported below, not because it is considered an exact reference.

1) Training and inference: All images were resampled and intensity-normalized before training. We used a 2.5D U-Net++ architecture with a ResNet-34 encoder [32] that predicts the central sCT slice from adjacent input slices. Models were optimized with the loss described below, and checkpoints were selected according to validation MAE. Predictions from the retained models were averaged at inference time. Complete preprocessing, architecture, optimization, and inference parameters are provided in Appendix B.

The objective of this work is not to introduce a new architecture, but to use a reliable and efficient supervised baseline for studying the effect of registration-induced supervision bias. The 2.5D U-Net++ provides a practical compromise between anatomical context, computational cost, and deployment simplicity.

## C. SAM-based perceptual supervision

We investigate the impact of SAM-based perceptual supervision by comparing two training objectives: a standard voxelwise reconstruction loss and an augmented objective combining voxel-wise and feature-based constraints. This subsection describes only the losses used for model optimization; the separately calibrated SAM-based evaluation metric is introduced in Section III-D.

a) Voxel-wise reconstruction loss: The baseline model is trained using a standard $\ell _ { 1 }$ loss, defined as:

$$
\mathcal { L } _ { \mathrm { M A E } } = \| \hat { y } - y \| _ { 1 } ,\tag{4}
$$

where $\hat { y }$ denotes the synthesized CT image and y the reference CT.

This formulation corresponds to the dominant paradigm in supervised sCT generation, where the model is optimized to minimize voxel-wise discrepancies between paired images, in accordance with the MAE-based evaluation criterion commonly used in the field.

b) SAM-based perceptual loss: To complement voxellevel supervision, we introduce a perceptual loss defined in the feature space of a pretrained segmentation model.

Classical perceptual losses commonly rely on networks trained on natural images (e.g., VGG). Instead, we use the frozen encoder of the Segment Anything Model (SAM), in its SAM 2 version based on the Hiera hierarchical transformer architecture, as a feature extractor [33]. The encoder parameters remain fixed throughout synthesis-model training and provide multi-scale feature representations for comparing the synthesized and reference CT images.

Let ψ(·) denote the frozen SAM encoder. The perceptual loss is defined as:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { S A M } } = \displaystyle \sum _ { l \in \mathcal { S } } w _ { l } \Delta _ { l } , } \\ { \Delta _ { l } = \| \psi _ { l } ( \hat { y } ) - \psi _ { l } ( y ) \| _ { 1 } , } \end{array}\tag{5}
$$

![](images/ad555fc851d5a919da1309c888aba5cd7ca523c70d0a12b43fbe6111dc7bd9a2.jpg)  
Fig. 1. Overview of the synthesis and evaluation protocol used to study registration-induced bias. Paired MR/CBCT and planning CT data are first converted into supervised training pairs through ELX or IMPACT registration conventions, yielding registered CT targets used as supervision labels for a 2.5D U-Net++ trained with MAE, VGG, or SAM-based objectives. Rectangular blocks denote data, targets, or outputs, whereas rounded blocks denote processing modules, training components, metrics, or analysis toolboxes; dashed blocks indicate independent or OOD evaluation geometries. The resulting synthetic CT images are evaluated using reference-based, anatomy-oriented, and dose-based metrics, complemented by independent/OOD test settings and CT-derived controls that isolate robustness and registration-induced metric sensitivity. Small tags indicate where the corresponding protocol components and analyses are defined in the manuscript. The robustness and benchmark branch distinguishes the independent/OOD geometry from ELX- and IMPACT-based evaluation registrations.

where ψ<sub>l</sub>(·) denotes the feature map extracted at layer l.

In practice, features are extracted from four hierarchical levels of the encoder. Only the two intermediate feature maps are used, corresponding to a weighting scheme of (0, 1, 1, 0). This selection targets representations that capture anatomical structures at an appropriate level of abstraction, avoiding both low-level noise sensitivity and overly coarse semantic features.

c) Combined objective: The final training objective is defined as:

$$
\mathcal { L } _ { \mathrm { t o t a l } } = \mathcal { L } _ { \mathrm { M A E } } + \mathcal { L } _ { \mathrm { S A M } } ,\tag{6}
$$

where both terms are equally weighted.

Rather than relying on extensive hyperparameter tuning, this formulation is intentionally kept simple to assess the intrinsic contribution of perceptual supervision. A method that remains effective under minimal tuning is more likely to generalize across datasets and clinical conditions.

## D. SAM-based perceptual evaluation metric

Separately from the training objective described in Section III-C, we define a calibrated SAM-based perceptual distance for anatomy-oriented evaluation. The SAM-based metric is computed only after model training and is not used to optimize the synthesis network or to select its checkpoints. In the spirit of LPIPS [34], SAM provides the feature representation, while separately learned channel-wise weights calibrate the contribution of the selected features to the final distance.

The objective is to calibrate the feature-space distance so that it better discriminates the structural defects typically produced by MAE-trained synthetic CT models from the residual discrepancies caused by imperfect registration.

Starting from the SAM-based distance defined above, we introduce channel-wise weights $\alpha _ { l , c }$ applied to the feature discrepancies at each selected layer. These weights are optimized from precomputed feature differences rather than fixed heuristically. The calibration is designed to emphasize feature channels that distinguish a sharp CT image affected by residual alignment errors from a synthetic CT prediction affected by regression-induced blurring or structural inaccuracies.

Calibration is performed once on development cases only, before final evaluation. For each selected SAM layer, absolute feature differences are spatially averaged per channel and normalized using calibration-set statistics. Non-negative channel weights are then learned with a hinge-ranking objective:

$$
\begin{array} { l } { { \displaystyle { \mathcal { L } } _ { \mathrm { c a l } } = \frac { 1 } { N } \sum _ { i } \mathrm { m a x } \big ( 0 , m + d _ { i } ^ { \mathrm { d e f } } - d _ { i } ^ { \mathrm { M A E } } \big ) + \lambda \| \alpha \| _ { 2 } ^ { 2 } , } } \\ { { \displaystyle d _ { i } ^ { \mathrm { d e f } } = d _ { \alpha } \big ( \mathrm { C T } _ { \mathrm { d e f } } ^ { i } , \mathrm { C T } ^ { i } \big ) , } } \\ { { \displaystyle d _ { i } ^ { \mathrm { M A E } } = d _ { \alpha } \big ( \mathrm { s C T } _ { \mathrm { M A E } } ^ { i } , \mathrm { C T } ^ { i } \big ) , } } \end{array}\tag{7}
$$

where $\mathrm { C T _ { d e f } }$ denotes the deformed CT control, m is a fixed margin, λ controls a small $\ell _ { 2 }$ regularization term, and the learned weights are normalized after optimization. The final weights are frozen and reused for all reported evaluations. This constraint encodes the working assumption that a deformed CT control, although still affected by residual registration errors, preserves CT-like structural detail and high-frequency anatomy better than a purely MAE-trained synthetic prediction.

The resulting calibrated metric downweights feature responses dominated by residual misalignment and emphasizes channels sensitive to the characteristic defects of MAE-trained synthesis, such as blurring, loss of fine anatomical boundaries, or structurally inconsistent predictions. The optimization identifies the intermediate SAM feature levels, particularly layers 2 and 3, as the most discriminative representations. This suggests that early features are too local and sensitive to low-level appearance differences. In contrast to an unweighted perceptual distance, the optimized SAM-based metric is therefore designed to better reflect structural defects in synthetic CT images. Accordingly, ${ \mathcal { L } } _ { \mathrm { S A M } }$ denotes the feature-space loss used during synthesis-model training, whereas $d _ { \mathrm { S A M } }$ denotes the separately calibrated distance used for evaluation.

## E. Experimental configurations

In the experimental study, we compare two synthesis objectives and two registration strategies:

• MAE: synthesis model trained with the voxel-wise reconstruction loss $\mathcal { L } _ { \mathrm { M A E } }$ only;

• SAM: synthesis model trained with the combined objective $\mathcal { L } _ { \mathrm { t o t a l } }$ , including SAM-based perceptual supervision;

• ELX: training pairs constructed using the official Elastixbased registration pipeline;

• IMPACT: training pairs constructed using the IMPACT-Reg registration pipeline.

a) Evaluation protocol: We adopt a held-out evaluation protocol combined with five-fold cross-validation for model development. For each task, 15% of the available cases are excluded from training and reserved for final testing, while the remaining data are used to train five independent models subsequently combined through ensembling at inference time. The evaluation branches in Fig. 1 summarize the three complementary analyses used below: reference-based metrics, anatomyand dose-oriented assessment, and robustness/sensitivity controls.

b) Task 1 (MRI-to-CT): The final evaluation protocol relies on three complementary test settings.

(1) In-distribution (ID) evaluation. The ID test set corresponds to the 66 held-out cases from SynthRAD2025 (AB = 22, HN = 21, TH = 23). For this subset, two versions of the data are considered: the ELX-based non-rigid registration provided by the organizers and our IMPACT-based non-rigid registration. These two aligned references are used to evaluate the sensitivity of quantitative metrics to the choice of registration under in-distribution conditions.

(2) Out-of-distribution (OOD) evaluation with SynthRAD2023. The OOD test set includes 50 cases from SynthRAD2023 (brain = 27, pelvis = 23). Similarly to the ID setting, both ELX-based and IMPACT-based registrations are used to generate two evaluation references. Notably, the ELX registration corresponds to a rigid alignment, consistent with the original evaluation setup of the challenge, while IMPACT provides a non-rigid alternative. This setting allows us to assess the impact of registration differences in a more challenging OOD scenario.

(3) OOD evaluation with expert-refined alignment (Ext-T2). Finally, we evaluate the model on an external OOD dataset composed of 24 T2-weighted MRI cases [35]. This dataset represents a particularly severe OOD setting, as no T2- weighted MRI data are included in the training set. The dataset further provides high-quality voxel-wise alignment between MRI and CT, obtained through a single reference alignment manually refined by experts.

In contrast to the previous settings, this dataset relies on a unique reference, removing variability induced by different registration methods and providing an expert-refined evaluation setting with reduced registration uncertainty.

c) Task 2 (CBCT-to-CT): The evaluation protocol for Task 2 relies on two complementary test settings based on the SynthRAD2025 dataset.

(1) Real CBCT evaluation. The first setting uses the 103 held-out SynthRAD2025 cases, where CBCT and CT are acquired independently (AB = 32, HN = 37, TH = 34). Similarly to Task 1, two versions of the data are considered using ELX-based and IMPACT-based registrations to define the evaluation reference.

(2) Simulated CBCT evaluation (Sim-CBCT, registration-free). To isolate the effect of registration bias, we construct a second evaluation set by generating simulated CBCT volumes from the same 103 CT images. These simulated CBCT volumes are obtained by forward-projecting the planning CT using an RTK-based [36] cone-beam simulation pipeline, followed by projection degradation and FDK reconstruction.

Because the simulated CBCT is generated directly from the CT volume, the anatomical correspondence between input and reference is preserved by construction. This removes the inter-modality registration uncertainty present in real CBCT-CT pairs and provides a controlled setting where quantitative metrics directly reflect synthesis quality.

However, the simulated CBCT volumes do not perfectly reproduce real acquisitions. Hounsfield units are not fully calibrated, and some CT volumes are truncated, leading to differences in field-of-view compared to clinical CBCT scans. The simulation therefore does not strictly match the physical acquisition process of real CBCT, and noticeable intensity and boundary discrepancies may arise. Nevertheless, the simulated data capture the main characteristics of CBCT imaging, providing a realistic yet controlled approximation suitable for analysis.

In the result tables, AB/HN/TH denotes the real held-out CBCT setting, whereas $A B _ { \mathrm { s i m } } , ~ H N _ { \mathrm { s i m } } ,$ , and $T H _ { \mathrm { s i m } }$ denote these registration-free simulated CBCT evaluations, and Sim-CBCT their aggregate. For Task 1, Ext-T2 denotes the external T2-weighted MRI set.

## F. Statistical analysis

All quantitative metrics were first computed at the patientvolume level. For each anatomical region and experimental configuration, results are reported as the mean and standard

BETTER ANATOMICAL ALIGNMENT BETWEEN THE REGISTERED SOURCE-MODALITY STRUCTURES AND THE REFERENCE CT STRUCTURES.

deviation across patients. Comparisons between models or registration configurations were performed on matched patients using two-sided Wilcoxon signed-rank tests. The patient, rather than the slice or the cross-validation fold, was treated as the statistical unit. Statistical significance is denoted by $^ { * } p _ { \mathrm { a d j } } < 0 . 0 5 , ^ { * * } p _ { \mathrm { a d j } } < 0 . 0 1$ , and $^ { * * * } p _ { \mathrm { a d j } } < 0 . 0 0 1$

For ensemble results (CV), the five cross-validation predictions were averaged voxel-wise for each patient before computing the evaluation metrics. For Mean CV results, each metric was first computed for every fold-specific prediction and patient, then averaged across the five folds for that patient before calculating group-level summary statistics. Thus, folds were not treated as independent statistical observations.

## G. Qualitative comparison of IMPACT and ELX registrations

As a preliminary step before assessing the impact of registration on training and evaluation in cross-modality image synthesis, qualitative comparisons between IMPACT- and ELXbased registrations across several anatomical regions (AB, HN, TH) in the MR→CT setting are provided in Appendix Figure 4.

Both registration approaches produce visually plausible and globally coherent alignments across the three anatomical regions considered. At the scale of the full volume, structural correspondences between MR and CT are largely established in both cases, although residual anatomical discrepancies remain visible in regions subject to large deformations, including the diaphragm and soft tissue boundaries. This stands in contrast to the rigid-only alignment adopted in the SynthRAD2023 edition, which left substantially larger residual discrepancies between modalities. In the present setting, both ELX and IMPACT perform deformable non-rigid registration, and ELX already represents the level of registration quality commonly used in supervised synthetic CT studies. The observed differences should therefore be interpreted as refinements over an already credible and clinically realistic alignment rather than as a fundamental reliability gap. It is also important to note that the effects reported in the following sections are obtained despite this relatively strong registration baseline; under the rigid-only alignment setup used in the SynthRAD2023 edition, the observed differences and their impact on supervised synthesis would likely have been substantially larger.

These qualitative observations are complemented by the structure-wise Dice analysis reported in Table I. The table summarizes the mean Dice coefficient obtained after registration for each anatomical region. Higher Dice values indicate greater agreement between the registered CT structures and the source-modality anatomy captured by the segmentation analysis. They therefore support the use of IMPACT as an alternative supervision convention with higher measured anatomical consistency on average, without establishing it as a ground-truth alignment.

Overall, IMPACT achieves a modest average Dice increase over ELX, from 0.724 to 0.758, although the direction of the difference is not uniform across every region. Together with the qualitative observations, these results define two alternative supervision conventions with different measured levels of anatomical consistency.

TABLE I
<table><tr><td>Region</td><td>Rigid</td><td>ELX</td><td>IMPACT</td></tr><tr><td>Pelvis</td><td>0.64</td><td>0.70</td><td>0.73</td></tr><tr><td>Abdomen</td><td>0.63</td><td>0.71</td><td>0.70</td></tr><tr><td>Thorax</td><td>0.61</td><td>0.66</td><td>0.71</td></tr><tr><td>Head &amp; Neck</td><td>0.72</td><td>0.75</td><td>0.81</td></tr><tr><td>Brain</td><td>0.71</td><td>0.80</td><td>0.84</td></tr><tr><td>Mean</td><td>0.662</td><td>0.724</td><td>0.758</td></tr></table>

## IV. RESULTS

A. Effect of Registration Consistency Between Training and Evaluation

Table II summarizes the aggregate in-distribution results for all combinations of registration strategies used during training and evaluation. The first term in each configuration denotes the registration used to construct the training targets, whereas the second denotes the registration used to define the evaluation reference. Complete region-wise results, including patientlevel variability and statistical comparisons, are reported in Appendix C-A, Tables XIV and XV.

TABLE II  
AGGREGATE IN-DISTRIBUTION PERFORMANCE FOR ALL COMBINATIONS OF REGISTRATION STRATEGIES USED DURING TRAINING AND EVALUATION. RESULTS ARE POOLED ACROSS THE AB, HN, AND TH REGIONS. COMPLETE REGIONAL RESULTS ARE REPORTED IN APPENDIX C-A.
<table><tr><td colspan="4">Task 1</td><td colspan="3">Task 2</td></tr><tr><td>Train/Eval</td><td>MAE↓</td><td>PSNR ↑</td><td>SSIM ↑</td><td>MAE↓</td><td>PSNR↑</td><td>SSIM ↑</td></tr><tr><td>IMPACT/IMPACT</td><td>63.55</td><td>29.97</td><td>0.927</td><td>58.76</td><td>31.16</td><td>0.937</td></tr><tr><td>IMPACT/ELX</td><td>72.47</td><td>28.49</td><td>0.918</td><td>69.17</td><td>29.13</td><td>0.920</td></tr><tr><td>ELX/IMPACT</td><td>68.31</td><td>29.37</td><td>0.920</td><td>65.70</td><td>30.08</td><td>0.921</td></tr><tr><td>ELX/ELX</td><td>66.98</td><td>29.24</td><td>0.924</td><td>61.28</td><td>30.22</td><td>0.930</td></tr></table>

Across both tasks, the best voxel-wise performance is obtained when the registration strategy used for training and evaluation is matched. In Task 1, IMPACT/IMPACT reaches an aggregate MAE of 63.55, compared with 72.47 when the same model is evaluated against ELX references. In Task 2, the same pattern is stronger, with MAE increasing from 58.76 to 69.17 under evaluation mismatch. The effect is asymmetric: evaluation mismatch degrades IMPACT-trained models by 14.0% (Task 1) and 17.7% (Task 2), but ELX-trained models by only 2.0% and 7.2%, which still remain above IMPACT/IMPACT. This behavior indicates that supervised models do not only learn a modality translation mapping, but also partially adapt to the geometric convention imposed by the registration pipeline.

This effect becomes more pronounced in OOD regions. The same dependence is observed in the SynthRAD2023 brain and pelvis regions, where the difference between the rigidonly ELX convention and the IMPACT deformable convention is larger. In these regions, ELX-trained models even score better against IMPACT than against the rigid ELX references, consistent with their deformable training targets being geometrically closer to IMPACT than to a rigid alignment. IM-PACT/IMPACT nevertheless remains the best configuration. Full regional results are provided in Appendix C-A.

Interestingly, the top-performing methods reported in the original SynthRAD2023 challenge achieved substantially lower voxel-wise errors under this rigid-registration setting, reaching 58.83±13.41 HU MAE, 29.61±1.79 dB PSNR, and $0 . 8 8 5 \pm 0 . 0 2 9$ SSIM for Task 1. These results were obtained using a supervised synthesis framework conceptually close to the one proposed in this work, but trained directly on rigidly aligned image pairs, therefore matching the registration convention used during evaluation. This further indicates that strong in-distribution voxel-wise performance can still be achieved even with imperfectly aligned training pairs, provided that the same alignment convention is consistently used during both training and evaluation.

Table III reports performance in OOD settings where the evaluation references differ from the two registration methods used during training (ELX and IMPACT). IMPACT-trained models consistently achieve lower MAE and higher PSNR and SSIM than ELX-trained models across both tasks, with all differences reaching $p < 0 . 0 0 1$ The MAE improvement ranges from 9.0% to 15.0% across the Task 2 simulated CBCT regions, supporting the conclusion that the effect is not limited to agreement with the IMPACT evaluation reference.

RELATIVE PERCENTAGE INCREASE OF UNCERTAINTY FOR ELX COMPARED TO IMPACT. UNCERTAINTY IS ESTIMATED AS THE VARIANCE ACROSS THE 15 PREDICTIONS PER PATIENT OBTAINED FROM THE FIVE CROSS-VALIDATION MODELS AND THREE TEST-TIME AUGMENTATIONS. EXT-T2 DENOTES THE EXTERNAL T2-WEIGHTED MRI SET AND SIM-CBCT THE REGISTRATION-FREE SIMULATED CBCT SET.  
TABLE III  
QUANTITATIVE RESULTS ON OUT-OF-DISTRIBUTION (OOD) REGIONS FOR TASK 1 AND TASK 2 IN SUPERVISED CROSS-VALIDATION EXPERIMENTS. MEAN ± STANDARD DEVIATION OF COMMONLY USED IMAGE SIMILARITY METRICS (MAE, PSNR, SSIM) ARE REPORTED. COLUMN GROUPS INDICATE THE MODELS USED (IMPACT OR ELX). EXT-T2 IS THE EXTERNAL T2-WEIGHTED MRI SET; $A B _ { \mathrm { s i m } } , H N _ { \mathrm { s i m } } ,$ , AND $T H _ { \mathrm { s i m } }$ ARE THE REGISTRATION-FREE SIMULATED CBCT SETS. STATISTICAL COMPARISONS AGAINST IMPACT FOLLOW THE PATIENT-LEVEL PROTOCOL DESCRIBED IN SECTION III-F.
<table><tr><td>Task</td><td>Region</td><td colspan="3">IMPACT</td><td colspan="3">ELX</td></tr><tr><td></td><td></td><td>MAE</td><td>PSNR</td><td>SSIM</td><td>MAE</td><td>PSNR</td><td>SSIM</td></tr><tr><td>T1</td><td>Ext-T2</td><td>88.54 ±12.51</td><td>26.82 ±0.91</td><td>0.899 ±0.025</td><td> $9 7 . 4 3 ^ { * * * }$   $\pm 1 4 . 8 3 $ </td><td> $2 6 . 1 9 ^ { * * * }$   $\pm 0 . 8 7$ </td><td> $0 . 8 8 9 ^ { * * * }$  ±0.031</td></tr><tr><td rowspan="4">T2</td><td></td><td>60.41</td><td>30.32</td><td>0.927</td><td> $7 1 . 0 9 ^ { * * * }$ </td><td> $2 8 . 6 8 ^ { * * * }$ </td><td> $0 . 9 1 5 ^ { * * * }$ </td></tr><tr><td> $A B _ { \mathrm { s i m } }$ </td><td>±14.83</td><td>±1.87</td><td>±0.016</td><td> $\pm 1 3 . 1 2 $ </td><td> $\pm 1 . 3 0$ </td><td> $\pm 0 . 0 1 4$ </td></tr><tr><td></td><td>136.18</td><td>24.10</td><td>0.901</td><td> $1 4 9 . 6 6 ^ { * * * }$ </td><td> $2 3 . 4 5 ^ { * * * }$ </td><td> $0 . 8 9 2 ^ { * * * }$ </td></tr><tr><td> $H N _ { \mathrm { s i m } }$ </td><td>±30.01 95.99</td><td>±2.05</td><td>±0.029 0.885</td><td> $\pm 2 8 . 9 1 $ </td><td> $\pm 1 . 6 5$ </td><td> $\pm 0 . 0 3 1$ </td></tr><tr><td></td><td> $T H _ { \mathrm { s i m } }$ </td><td>±44.86</td><td>27.38 ±3.26</td><td>±0.062</td><td> $1 0 9 . 1 5 ^ { * * * }$   $\pm 4 9 . 6 5 $ </td><td> $2 6 . 3 4 ^ { * * * }$   $\pm 2 . 9 3$ </td><td> $0 . 8 7 0 ^ { * * * }$   $\pm 0 . 0 6 7$ </td></tr></table>

Figure 2 presents a qualitative comparison of synthetic CT images generated from the same MR input using ELXand IMPACT-based training. Both models produce visually plausible CT-like outputs with globally coherent intensity distributions. However, a clear anatomical inconsistency is observed in the ELX-trained output: the diaphragm region is severely distorted, with the cardiac silhouette and inferior lung boundaries displaced in a manner that does not correspond to the source MR anatomy. This deformation is not present in the MR input and represents a spurious anatomical configuration introduced by the synthesis model. In contrast, the IMPACTtrained model, trained using more geometrically consistent image pairs than ELX, preserves the diaphragm position and overall thoracic anatomy in a configuration consistent with the source image, with sharper pulmonary contours and more faithful mediastinal structure delineation.

Table IV presents prediction uncertainty estimated from the variance across 15 synthesized predictions per patient, corresponding to the combination of five cross-validation models and three test-time augmentations (original, horizontal flip, and vertical flip). Results are reported for models trained using the voxel-wise MAE criterion. Uncertainty is systematically higher for ELX-trained models than for IMPACT-trained models across both tasks. In Task 1, in-distribution regions show relative uncertainty increases ranging from 20.7% (HN and TH) to 41.2% (AB), with an aggregate increase of 29.7%. In Task 2, the same trend is observed but with more variable magnitude, ranging from 1.2% (HN) to 27.6% (AB), with an aggregate increase of 12.5%. In Task 1, uncertainty differences become substantially larger in OOD settings, reaching 116.1% for the pelvis, while the Ext-T2 increase reaches 61.8%. The brain region constitutes a partial exception, with a more limited uncertainty increase of 2.7%, consistent with its lower overall performance gap between registration strategies. Overall, ELX-trained models exhibit greater prediction variability than IMPACT-trained models, indicating that synthesized predictions vary more strongly from one inference configuration to another when the models are trained on less geometrically consistent image pairs.

TABLE IV
<table><tr><td>Region</td><td>Task 1</td><td>Task 2</td></tr><tr><td>AB</td><td>+41.2%</td><td> $+ 2 7 . 6 \%$ </td></tr><tr><td>HN</td><td>+20.7%</td><td> $+ 1 . 2 \%$ </td></tr><tr><td>TH</td><td>+20.7%</td><td> $+ 1 2 . 1 \%$ </td></tr><tr><td>AB/HN/TH</td><td>+29.7%</td><td>+12.5%</td></tr><tr><td>Brain</td><td>+2.7%</td><td></td></tr><tr><td>Pelvis</td><td>+116.1%</td><td></td></tr><tr><td>Ext-T2</td><td>+61.8%</td><td></td></tr><tr><td>Sim-CBCT</td><td>一</td><td>+1.8%</td></tr></table>

## B. SAM-based perceptual supervision

Table V compares MAE-, VGG-, and SAM-based supervision using downstream Dice scores obtained with TotalSegmentator. Because this evaluation does not reuse SAM features, it provides an independent assessment of anatomical preservation. Complete Dice and SSIM results are reported in Appendix C-B.

SAM-based supervision improves the aggregate Task 1 Dice from 0.711 with MAE and 0.712 with VGG to 0.738. The improvement is larger in unseen regions, reaching 0.806 for brain/pelvis and 0.725 on Ext-T2. In Task 2, the aggregate in-distribution improvement is smaller, but SAM obtains the best Dice in all three simulated CBCT regions. SSIM remains similar to, or occasionally lower than, that obtained with MAE supervision, indicating that improved anatomical preservation is not always reflected by reference-based intensity similarity.

MR  
![](images/158f0d5583e8fd9824cdb62a3d6f9cd5f1b513232726fa019b642236e29f746a.jpg)

sCT Elastix  
![](images/6c46e84ba5d790f771dd0ecdbcaf04fe92f4e5a5f61fd2ce9a0096ec7ece4abb.jpg)

sCT IMPACT-Reg  
![](images/58019faf4008f80293b827b7168a4bf950b661ea236e58e81564de181143fac6.jpg)  
Fig. 2. Qualitative comparison between MR input and synthesized CT generated using ELX-based and IMPACT-based training.

TABLE V  
DOWNSTREAM DICE SCORES OBTAINED AFTER TRAINING WITH MAE, VGG-BASED PERCEPTUAL, OR SAM-BASED PERCEPTUAL SUPERVISION. ALL MODELS USE IMPACT-BASED TRAINING PAIRS. FOR TASK 2, $A B _ { \mathrm { s i m } } , H N _ { \mathrm { s i m } } ,$ AND $T H _ { \mathrm { s i m } }$ DENOTE REGISTRATION-FREE SIMULATED CBCT EVALUATIONS. COMPLETE REGION-WISE DICE AND SSIM RESULTS ARE REPORTED IN APPENDIX C-B.
<table><tr><td>Task</td><td>Evaluation set</td><td>MAE</td><td>VGG</td><td>SAM</td></tr><tr><td>Task 1 Task 1</td><td>AB/HN/TH Brain/Pelvis Ext-T2</td><td>0.711 0.749 0.665</td><td>0.712 0.772 0.672</td><td>0.738 0.806 0.725</td></tr><tr><td>Task 1 Task 2 Task 2 Task 2</td><td>AB/HN/TH  $A B _ { \mathrm { s i m } }$   $H N _ { \mathrm { s i m } }$ </td><td>0.695 0.709 0.734</td><td>0.683 0.723 0.750</td><td>0.700 0.764 0.755</td></tr></table>

Figure 3 illustrates the same trend qualitatively. Voxel-wise objectives produce smoother predictions, whereas perceptual supervision, particularly with SAM features, better preserves sharp boundaries and fine anatomical structures. These visual differences are consistent with the downstream Dice improvements.

Table VI reports both the average performance of the individual cross-validation models (Mean CV) and the performance of their voxel-wise ensemble (CV ensemble). Complete Mean CV and region-wise results are reported in Appendix C-C, Tables XVIII and XIX. In the in-distribution regions, SAM supervision consistently improves $d _ { \mathrm { S A M } }$ and LPIPS but increases MAE relative to direct MAE supervision. This trade-off is observed for both individual models and their ensembles, suggesting that it reflects the training objective rather than ensemble construction.

In the Ext-T2 and Sim-CBCT settings, SAM supervision improves MAE, $d _ { \mathrm { S A M } }$ , and LPIPS for both tasks and both prediction strategies. Ensemble averaging primarily reduces MAE, whereas its effect on perceptual metrics is smaller and not uniformly favorable. This is consistent with voxel-wise averaging reducing random intensity errors while potentially attenuating fine structural details.

## C. Anatomical consistency

The objective of this section is to analyze the extent to which the different registration strategies preserve the anatomy of the input image in the synthesized outputs. To highlight these differences, we introduce a new anatomy-oriented similarity metric. The metric is derived from feature representations extracted by the pretrained TS CT 3 mm segmentation model (M297) used within IMPACT-Reg. The feature extractor remains fixed during evaluation. The underlying intuition is that a similarity metric capable of accurately assessing multimodal anatomical correspondence in image registration should also provide a relevant measure of anatomical consistency between the input image and the synthesized image.

Anatomical consistency results reported in Table VII show substantial differences in the deformation required to align the synthesized images with the input CBCT, as measured with the IMPACT-Reg metric M297. Smaller deformations, i.e., higher anatomical consistency, are consistently obtained for IMPACTtrained models across all evaluated regions.

In in-distribution regions, the relative difference between ELX and IMPACT reaches 124.25% in the abdominal region and 136.65% in head-and-neck cases (both $\begin{array} { r l r } { p } & { { } < } & { 0 . 0 0 1 ) } \end{array}$ , indicating substantially larger deformations, and thus lower anatomical consistency, for ELX-trained models. In the thoracic region, the difference remains statistically significant but is markedly smaller (3.59%, $p \ < \ 0 . 0 1 $ ). Aggregated across AB/HN/TH regions, ELX-trained models exhibit an overall relative difference of 81.82% compared to IMPACT-trained models $( p < 0 . 0 0 1 )$ ).

A similar trend is observed on Sim-CBCT, where ELXtrained models still require significantly larger deformations than IMPACT-trained models, with a relative difference of 9.65% $( p < 0 . 0 0 1 )$ ).

Together, these results show that IMPACT-trained models better preserve the anatomy of the input CBCT in both real and simulated CBCT settings.

## D. Official SynthRAD results as a benchmark-bias case study

The official SynthRAD evaluation provides an external case study of registration-dependent benchmark behavior. In the local experiments, IMPACT-aligned supervision is associated with higher measured anatomical consistency and better performance in several registration-independent or better-aligned settings. In contrast, the official server favors ELX-trained models, whose targets are more consistent with the registration convention used to construct the challenge evaluation references.

![](images/c95439d750e33767e0feb0f5f7e423f81e9463921e11b6977577120aafb78186.jpg)  
Fig. 3. Qualitative comparison of synthesized CT images obtained using different training losses (rows) across multiple anatomical cases (columns). From top to bottom: input MR images, sCT generated using MSE, MAE, MAE+VGG-based perceptual loss, and MAE+SAM-based loss, followed by the reference CT images. For each synthesized image, the corresponding MAE with respect to the reference CT is reported. While all methods produce visually plausible outputs, differences in image sharpness and structural fidelity can be observed across losses, particularly at anatomical boundaries.

Table VIII illustrates this reversal. Under the official protocol, ELX training improves all reported metrics relative to IMPACT training in both tasks. For example, MAE decreases from 75.82 to 68.20 HU in Task 1 and from 56.05 to 52.87 HU in Task 2. These results do not contradict the local anatomical analyses; rather, they show that reference-based scores reflect both synthesis quality and compatibility with the geometry embedded in the evaluation data.

The final challenge submission therefore used the fivefold ELX ensemble trained with SAM-based perceptual supervision. This configuration was selected because the public validation results favored ELX-consistent training, while SAM supervision improved downstream Dice and qualitative structural preservation despite a moderate increase in local voxelwise MAE. Checkpoints within each configuration remained selected exclusively according to validation MAE. For the final

TABLE VI  
AGGREGATE PERFORMANCE OF MODELS TRAINED WITH MAE OR SAM-BASED SUPERVISION AND EVALUATED WITH ELX REFERENCES. MEAN CV DENOTES THE PATIENT-LEVEL PERFORMANCE AVERAGED ACROSS THE FIVE INDIVIDUAL CROSS-VALIDATION MODELS, WHEREAS CV ENSEMBLE DENOTES PERFORMANCE AFTER VOXEL-WISE AVERAGING OF THEIR PREDICTIONS. d AND LPIPS VALUES ARE MULTIPLIED BY 100 FOR READABILITY. COMPLETE REGION-WISE RESULTS AND STATISTICAL COMPARISONS ARE PROVIDED IN APPENDIX C-C.
<table><tr><td rowspan="2"></td><td rowspan="2">Set</td><td rowspan="2">Prediction</td><td colspan="3">MAE supervision</td><td colspan="3">SAM supervision</td></tr><tr><td>MAE↓</td><td> $d _ { \mathrm { S A M } } ~ \mathrm { \downarrow }$ </td><td>LPIPS ↓</td><td>MAE↓</td><td> $d _ { \mathrm { S A M } } ~ .$  </td><td>LPIPS ↓</td></tr><tr><td rowspan="4">Task 1</td><td rowspan="2">AB/HN/TH</td><td>Mean CV</td><td>70.75</td><td>24.27</td><td>8.34</td><td>78.26</td><td>18.99</td><td>7.06</td></tr><tr><td>CV ensemble</td><td>66.98</td><td>24.31</td><td>8.22</td><td>73.21</td><td>19.85</td><td>7.03</td></tr><tr><td rowspan="2">Ext-T2</td><td>Mean CV</td><td>104.39</td><td>37.43</td><td>13.82</td><td>96.36</td><td>33.71</td><td>12.00</td></tr><tr><td>CV ensemble</td><td>97.43</td><td>37.30</td><td>13.25</td><td>90.14</td><td>35.93</td><td>11.79</td></tr><tr><td rowspan="4">Task 2</td><td rowspan="2">AB/HN/TH</td><td>Mean CV</td><td>64.95</td><td>17.54</td><td>5.48</td><td>69.52</td><td>13.99</td><td>4.69</td></tr><tr><td>CV ensemble</td><td>61.28</td><td>17.88</td><td>5.40</td><td>65.18</td><td>14.36</td><td>4.59</td></tr><tr><td rowspan="2">Sim-CBCT</td><td>Mean CV</td><td>114.78</td><td>20.87</td><td>6.34</td><td>109.23</td><td>17.94</td><td>5.31</td></tr><tr><td>CV ensemble</td><td>111.88</td><td>20.67</td><td>6.15</td><td>106.33</td><td>17.99</td><td>5.15</td></tr></table>

## TABLE VII

REGISTRATION-BASED DEFORMATION MEASURE BETWEEN THE INPUT CBCT AND THE SYNTHESIZED IMAGE FOR TASK 2, USING THE IMPACT-REG METRIC M297. VALUES REPORT THE RELATIVE PERCENTAGE DIFFERENCE BETWEEN ELX AND IMPACT. POSITIVE VALUES INDICATE LARGER DEFORMATION FOR ELX COMPARED TO   
IMPACT. RESULTS ARE REPORTED ACROSS ANATOMICAL REGIONS FOR MODELS TRAINED WITH THE SAM CRITERION. STATISTICAL SIGNIFICANCE IS ASSESSED USING PAIRED WILCOXON SIGNED-RANK TESTS ON MATCHED PATIENTS.

<table><tr><td>Region</td><td>IMPACT vs ELX</td></tr><tr><td>AB</td><td>124.25%***</td></tr><tr><td>HN</td><td>136.65%***</td></tr><tr><td>TH</td><td>3.59%**</td></tr><tr><td>AB/HN/TH</td><td>81.82%***</td></tr><tr><td>Sim-CBCT</td><td>9.65%***</td></tr></table>

TABLE VIII

ELX- VERSUS IMPACT-TRAINED MODELS ON THE PUBLIC SYNTHRAD2025 VALIDATION SET, AS SCORED BY THE CHALLENGE SERVER. HIGHER ELX SCORES REFLECT AGREEMENT WITH THE ELX-CONSISTENT EVALUATION GEOMETRY, NOT ESTABLISHED SOURCE-ANATOMY PRESERVATION.
<table><tr><td rowspan="2">Metric</td><td colspan="2">Task 1</td><td colspan="2">Task 2</td></tr><tr><td>ELX</td><td>IMPACT</td><td>ELX</td><td>IMPACT</td></tr><tr><td>MAE</td><td>68.20</td><td>75.82</td><td>52.87</td><td>56.05</td></tr><tr><td>PSNR</td><td>29.81</td><td>28.70</td><td>32.36</td><td>31.65</td></tr><tr><td>SSIM</td><td>0.92</td><td>0.91</td><td>0.96</td><td>0.95</td></tr><tr><td>Dice</td><td>0.72</td><td>0.70</td><td>0.83</td><td>0.82</td></tr><tr><td>HD95</td><td>8.42</td><td>8.89</td><td>5.40</td><td>5.41</td></tr></table>

submission only, the global model was further fine-tuned into two region-specific sub-models (AB+TH and HN); all other results in this paper use the global models.

The submitted method ranked third overall for both MR→CT and CBCT→CT synthesis. It achieved MAEs of 67.24 and 53.09 HU, Dice scores of 0.737 and 0.843, and HD95 values of 7.51 and 5.08 mm, respectively. Dosimetric performance was also competitive, including high gamma pass rates. Complete image-based and dosimetric rankings are provided in Appendix A-D.

Together, the public validation reversal and the official test results show that the proposed models are competitive under the prescribed challenge protocol while highlighting a limitation of reference-based ranking. The fact that ELX is favored by the ELX-consistent server, whereas IMPACT is favored by several source-preservation and OOD analyses, demonstrates that the evaluation convention can influence the apparent ordering of synthesis strategies.

## E. Estimation of registration-induced sensitivity in sCT evaluation metrics

These experiments estimate the magnitude of evaluation error that can arise from residual multimodal misregistration alone, without any synthesis process. Rather than defining a theoretical performance bound, the resulting values provide an empirical registration-induced reference level for MAE, PSNR, SSIM, d , LPIPS, and Dice.

To isolate this effect from any synthesis error, all controls were constructed from CT images only. The goal was to ask how much the evaluation metrics can change when the CT reference geometry is modified by registration, even though no synthetic CT is generated. The control was adapted to the evaluation geometry available in each benchmark. In SynthRAD2023 brain and pelvis cases, where the official CT reference is only rigidly aligned to the MR image, we compared this rigid CT with an IMPACT-deformed CT to estimate the effect of adding a plausible non-rigid correspondence. In SynthRAD2025 AB, HN, and TH cases, where deformable registration is already used, we applied a cycledeformation control: the CT was first transformed according to the ELX deformation and then mapped back using the IMPACT deformation field. This estimates the metric sensitivity to switching between two plausible non-rigid registration conventions, without introducing any synthesis model.

This analysis uses the disagreement between ELX and IMPACT as a proxy for registration-dependent geometric uncertainty. The preceding segmentation analysis indicates higher average anatomical consistency for IMPACT, but neither method is treated as ground truth. Agreement between their deformation fields suggests a relatively well-constrained correspondence, whereas disagreement identifies regions in which the estimated anatomy and the resulting evaluation metrics are more sensitive to the registration convention.

Table IX compares these CT-derived controls with representative supervised sCT results. In the SynthRAD2025 AB/HN/TH regions, registration alone produces aggregated MAEs of 60.02 HU for Task 1 and 53.42 HU for Task 2, with corresponding SSIM values of 0.927 and 0.938. In the SynthRAD2023 brain/pelvis regions, the estimated MAEs are 59.18 HU and 39.50 HU for Tasks 1 and 2, respectively. These values are of the same order of magnitude as those obtained by high-performing supervised synthesis methods, despite the absence of synthesis error in the CT-derived controls.

The regional variation is consistent with the dependence of intensity-based metrics on local image gradients. For a small residual displacement d(x), the induced intensity difference can be approximated by:

$$
| \nabla I ( x ) \cdot d ( x ) | .\tag{8}
$$

Consequently, comparable geometric errors can produce different MAEs across anatomical regions. Interfaces involving air, bone, teeth, or thin soft-tissue boundaries are particularly sensitive, helping to explain the larger metric variations observed in regions such as HN and pelvis.

For the aggregate comparisons, the supervised MAE, PSNR, and SSIM/MS-SSIM values correspond to the best official SynthRAD2023 and SynthRAD2025 submissions. Regional image-similarity values for AB, HN, and TH are obtained from the best local validation results of the region-specific finetuned ELX/SAM models submitted to the challenge [37]. The perceptual values are derived from the ELX-based experiments reported in Tables XVIII and XIX, whereas Dice uses the best result among the MAE-, VGG-, and SAM-trained models.

Several supervised results match or even outperform the CT-derived controls on MAE, PSNR, or SSIM, but this does not invalidate the sensitivity estimate. A synthesis model optimized against an imperfect reference can reduce voxelwise penalties through local smoothing, attenuation of highgradient boundaries, or adaptation to the evaluation geometry, whereas the CT-derived controls retain sharp CT structures. Consistent with this interpretation, the controls generally preserve substantially higher Dice scores, and their perceptual values are often closer to those of SAM-trained models than to purely MAE-trained models.

These estimates should be conservatively interpreted. The cycle experiment captures only the disagreement between two regularized and anatomically plausible registration methods, while the SynthRAD2023 deformations were intentionally constrained, particularly for brain cases. Additional perturbation experiments also showed that MAE increases rapidly with small changes to the deformation fields. Therefore, the controls likely underestimate the full effect of correspondence uncertainty.

Overall, high-performing supervised sCT methods operate within the same metric range as that induced by residual registration uncertainty alone. This suggests that current reference-based benchmarks may be approaching a registration-dependent performance ceiling, where small improvements partly reflect better adaptation to the evaluation geometry rather than improved patient-specific anatomical fidelity.

## V. DISCUSSION

The results reported above support a coherent interpretation centered on the interaction between registration quality, voxelwise supervision, and the metrics used to assess synthesis performance. This view directly builds on the concept of registration-induced target uncertainty introduced in Section II-A, where residual registration errors were described as structured geometric label noise shaping the empirical conditional distribution $p ( \tilde { y } \mid x )$ learned by supervised synthesis models.

## A. Registration bias as structured label noise

The central finding of this study is that reference-based synthesis performance depends not only on the synthesis model, but also on the compatibility between the registration conventions used to construct the training targets and evaluation references. Residual registration errors therefore act as structured geometric label noise at two levels: they shape the target distribution learned during training and influence the reference against which predictions are subsequently scored [10].

The matched and mismatched experiments provide direct evidence for this effect. As summarized in Table II, evaluating IMPACT-trained models against ELX rather than IMPACT references increases aggregate MAE from 63.55 to 72.47 HU in Task 1 and from 58.76 to 69.17 HU in Task 2. ELX-trained models similarly perform better under the ELX convention in both tasks. Because the predictions remain unchanged while only the evaluation reference is replaced, these differences show that voxel-wise scores measure both synthesis fidelity and agreement with the selected registration geometry.

Registration inconsistency also affects what the model learns. The qualitative example in Figure 2 shows blurred and displaced interfaces around the diaphragm in the ELX-trained prediction. As described in Section II-A, spatially variable targets broaden the observed distribution $p ( \tilde { y } \mid x )$ . Under voxel-wise $\ell _ { 1 }$ optimization, predicting a smoother intermediate boundary can reduce the expected penalty associated with placing a sharp structure at an uncertain location. Thus, although architectures with spatial skip connections possess a strong inductive bias toward preserving spatial organization, the optimization objective ultimately dominates the learned behavior. Under inconsistent voxel-wise supervision, ERM drives the model toward solutions that minimize the expected reconstruction error, even when this requires smoothing anatomical boundaries, attenuating high-frequency structures, or locally altering patient-specific anatomy.

The CT-derived controls in Table IX reinforce this interpretation. High-performing supervised models operate close to the metric variation induced by residual registration alone and sometimes obtain better MAE, PSNR, or SSIM values than the controls. This does not imply superior anatomical fidelity. Unlike a sharp deformed CT, a voxel-wise optimized model can adapt to the evaluation target through local smoothing, attenuation of uncertain boundaries, or reproduction of its geometric convention. The substantially higher Dice scores of the CT-derived controls support this distinction: better intensity agreement with a registered reference does not necessarily correspond to better structural preservation.

TABLE IX  
REGISTRATION-INDUCED METRIC SENSITIVITY (REG., CT-DERIVED CONTROLS) VERSUS REPRESENTATIVE SUPERVISED SCT PERFORMANCE (SUP.). RED: SUPERVISED RESULT BETTER THAN THE CONTROL. ORANGE: WORSE BY MORE THAN 10%. UNCOLORED: WITHIN 10%.
<table><tr><td>Task</td><td>Region</td><td> $\mathrm { T y p e }$ </td><td>MAE ↓</td><td>PSNR ↑</td><td>SSIM ↑</td><td> $d _ { \mathrm { S A M } } ~ \downarrow$ </td><td>LPIPS ↓</td><td>Dice ↑</td></tr><tr><td rowspan="6"></td><td>AB</td><td>Reg. Sup.</td><td> $6 6 . 4 5 \pm 1 3 . 5 8$  64.89</td><td> $2 8 . 3 4 \pm 1 . 3 8$  29.10</td><td> $0 . 9 0 8 \pm 0 . 0 2 7$  0.91</td><td> $2 8 . 4 4 8 \pm 6 . 0 0 7$  26.70</td><td> $1 0 . 8 \pm { 3 . 5 }$  10.7</td><td> $0 . 8 3 9 \pm 0 . 0 4 0$  0.777</td></tr><tr><td>HN</td><td>Reg. Sup.</td><td> $\begin{array} { c } { 6 2 . 9 6 \pm 1 3 . 8 7 } \\ { 6 5 . 1 5 } \end{array}$ </td><td> $\begin{array} { c } { 2 9 . 4 3 \pm 1 . 7 1 } \\ { 3 0 . 2 0 } \end{array}$ </td><td> $0 . 9 5 3 \pm 0 . 0 2 0$  0.94</td><td> $1 1 . 3 4 2 \pm 2 . 9 7 5$  11.96</td><td> $^ { 2 . 5 \pm 0 . 8 } _ { 3 . 1 }$ </td><td> $\begin{array} { c } { 0 . 8 1 8 \pm 0 . 0 7 2 } \\ { 0 . 7 3 1 } \end{array}$ </td></tr><tr><td>Task 1 TH</td><td>Reg. Sup.</td><td> $5 3 . 0 0 \pm 1 3 . 3 0$  60.07</td><td> $3 0 . 7 1 \pm 2 . 1 3$  30.76</td><td> $0 . 9 5 0 \pm 0 . 0 1 4$  0.94</td><td> $2 1 . 9 3 6 \pm 5 . 2 1 0$  20.50</td><td> $7 . 1 \pm 2 . 4$  7.2</td><td> $0 . 8 0 3 \pm 0 . 0 4 6$  0.706</td></tr><tr><td>AB/HN/TH</td><td>Reg. Sup.</td><td> $6 0 . 0 2 \pm 1 3 . 8 5$   $6 4 . 8 1 \pm 2 1 . 2 5$ </td><td> $2 9 . 2 8 \pm 1 . 8 5$   $2 9 . 9 9 7 \pm 2 . 7 5 9$ </td><td> $0 . 9 2 7 \pm 0 . 0 3 5$   $0 . 9 3 6 \pm 0 . 0 5 0$ </td><td> $2 0 . 7 3 6 \pm 8 . 5 2 7$   $1 9 . 8 5 \pm 7 . 2 6$ </td><td> $6 . 8 \pm 0 . 0 4 2$   $7 . 0 \pm 3 . 8$ </td><td> $0 . 8 2 0 \pm 0 . 0 5 6$   $0 . 7 3 8 \pm 0 . 0 5 5$ </td></tr><tr><td>Brain/Pelvis</td><td>Reg. Sup.</td><td> $5 9 . 1 8 \pm 1 2 . 5 0$   $5 8 . 8 3 \pm 1 3 . 4 1$ </td><td> $2 8 . 9 6 \pm 1 . 5 2$   $2 9 . 6 1 \pm 1 . 7 9$ </td><td> $0 . 9 1 4 \pm 0 . 0 3 8$   $0 . 8 8 5 \pm 0 . 0 2 9$ </td><td> $1 4 . 3 9 7 \pm 8 . 4 5 9$ </td><td> $4 . 8 \pm 4 . 1$  一</td><td> $0 . 8 8 6 \pm 0 . 0 6 0$ </td></tr><tr><td></td><td></td><td>Reg.  $5 5 . 8 8 \pm 1 3 . 8 6$  58.46</td><td> $2 9 . 7 8 \pm 1 . 8 4$  31.33</td><td> $\begin{array} { c } { 0 . 9 3 0 \pm 0 . 0 2 5 } \\ { 0 . 9 0 } \end{array}$ </td><td> $\begin{array} { c } { 1 7 . 3 4 8 \pm 4 . 5 4 8 } \\ { 1 8 . 1 0 } \end{array}$ </td><td> $5 . 5 \pm 2 . 1$  6.6</td><td> $0 . 7 4 5 \pm 0 . 0 5 8$  0.670</td></tr><tr><td rowspan="6">Task 2</td><td>HN</td><td>Sup. Reg. Sup.</td><td> $6 9 . 5 0 \pm 2 2 . 4 2$  60.97</td><td> $2 8 . 4 6 \pm 2 . 5 0$ </td><td> $\begin{array} { c } { 0 . 9 4 5 \pm 0 . 0 2 3 } \\ { 0 . 9 4 } \end{array}$ </td><td> $1 0 . 1 9 8 \pm 1 . 9 6 7$ </td><td> $2 . 0 \pm 0 . 6$ </td><td> $0 . 7 4 9 \pm 0 . 0 8 2$  0.720</td></tr><tr><td>TH</td><td>Reg.</td><td> $5 2 . 8 7 \pm 1 6 . 6 3$ </td><td>30.38</td><td></td><td>9.50  $1 6 . 5 3 2 \pm 3 . 7 4 1$ </td><td>2.1</td><td> $0 . 7 8 4 \pm 0 . 0 6 6$ </td></tr><tr><td></td><td>Sup.</td><td>50.40</td><td> $\begin{array} { c } { 3 0 . 9 6 \pm 2 . 5 7 } \\ { 3 1 . 7 8 } \end{array}$ </td><td> $\begin{array} { c } { 0 . 9 3 2 \pm 0 . 0 1 9 } \\ { 0 . 9 2 } \end{array}$ </td><td>16.12</td><td> $^ { 4 . 8 \pm 1 . 7 } _ { 5 . 4 }$ </td><td>0.720</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td> $0 . 7 5 9 \pm 0 . 0 7 2$ </td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>AB/HN/TH</td><td>Reg.</td><td> $5 3 . 4 2 \pm 2 0 . 1 8$ </td><td> $3 0 . 4 6 \pm 2 . 8 2$ </td><td> $0 . 9 3 8 \pm 0 . 0 3 1$ </td><td> $1 4 . 5 1 0 \pm 4 . 7 9 3$ </td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td> $4 . 0 \pm 2 . 2$ </td><td></td></tr><tr><td></td><td></td><td></td><td></td><td> $3 2 . 6 1 9 \pm 2 . 3 0 7$ </td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td>Sup.</td><td> $4 8 . 2 8 \pm 1 3 . 3 5$ </td><td></td><td></td><td> $1 4 . 3 6 \pm 5 . 1 0$ </td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td> $0 . 9 6 8 \pm 0 . 0 2 5$ </td><td></td><td> $4 . 6 \pm 2 . 6$ </td><td> $0 . 7 0 0 \pm 0 . 0 8 2$ </td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>Brain/Pelvis</td><td>Reg.</td><td> $3 9 . 5 0 \pm 1 3 . 0 9$ </td><td> $3 2 . 1 5 \pm 2 . 6 1$ </td><td> $0 . 9 4 2 \pm 0 . 0 4 3$ </td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td> $1 2 . 6 2 5 \pm 9 . 7 3 6$ </td><td> $4 . 0 \pm 4 . 6$ </td><td> $0 . 8 6 6 \pm 0 . 0 8 9$ </td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td> $3 0 . 7 9 \pm 2 . 0 0$ </td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td> $4 9 . 9 5 \pm 1 1 . 7 8$ </td><td></td><td> $0 . 9 0 6 \pm 0 . 0 3 6$ </td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td>Sup.</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></table>

This interpretation is consistent with the subsequent study by Zimmermann et al. [25], which used physics-based CBCT simulation to generate geometrically aligned pairs and IM-PACT registration for real CBCT-CT data [31]. Their findings similarly indicate that reducing registration-related inconsistencies improves anatomical and geometric coherence, even when conventional intensity metrics do not show a corresponding improvement.

More generally, benchmark rankings should be interpreted as protocol-dependent measurements rather than absolute indicators of anatomical accuracy or clinical superiority. The reversal between the local source-preservation analyses and the ELX-consistent SynthRAD server shows that small improvements in reference-based metrics may partly reflect adaptation to the benchmark geometry. Reliable sCT evaluation should therefore document the registration convention and complement voxel-wise scores with anatomy-oriented and task-specific criteria.

## B. Anatomical consistency and source preservation

estimates the amount of deformation required to bring the synthesized image into anatomical agreement with the source CBCT representation. Larger values therefore indicate lower anatomical consistency with the input image. Across all evaluated regions, ELX-trained models require larger deformations than IMPACT-trained models, indicating that the synthesized images produced after ELX-based supervision deviate more strongly from the source anatomy.

The effect is particularly pronounced in the abdominal and head-and-neck regions, where the relative differences between ELX- and IMPACT-trained models exceed 120%. These regions are characterized by complex soft-tissue structures and larger residual multimodal registration uncertainty, making them especially sensitive to geometric bias in the supervised targets. In the thoracic region, the difference is smaller but remains statistically significant, suggesting that the magnitude of the effect depends on the anatomical region and the difficulty of establishing reliable multimodal correspondences.

The registration-induced bias identified above is not limited to voxel-wise image similarity metrics. It should also affect the anatomical relationship between the source CBCT and the synthesized CT, since a model trained on imperfectly aligned targets may learn to reproduce the geometry of the registered CT reference rather than preserve the patient-specific anatomy of the input image.

This interpretation is supported by the anatomical consistency analysis reported in Table VII. The IMPACT-Reg metric

This analysis provides complementary evidence that improving the anatomical consistency of the training pairs reduces the propagation of geometric bias into the synthesized images. This effect is measured with respect to the source CBCT rather than only against the registered CT reference. It therefore directly supports the central objective of anatomypreserving synthesis: the generated CT-like image should adapt appearance toward the CT domain without altering the patientspecific anatomy present in the input image.

Together, these results show that agreement with a registered CT reference is insufficient to assess anatomical validity. Registration-based anatomical consistency metrics provide a useful complementary evaluation criterion, particularly for downstream tasks including segmentation and deformable registration, where preserving source anatomy is more important.

## C. Consequences on uncertainty and OOD generalization

Residual registration errors may also affect predictive uncertainty. The variability measured across models and testtime augmentations is commonly associated with epistemic uncertainty, but, under spatially inconsistent supervision, it may additionally reflect sensitivity to the structured geometric noise present in the training targets. Different models can therefore converge toward slightly different solutions of the biased conditional distribution $p ( { \tilde { y } } \mid \mathbf { \theta } \mid \mathbf { \theta } x )$ , even when the underlying synthesis mapping is otherwise well constrained.

From this perspective, the measured uncertainty is not purely epistemic. It is partly driven by structured geometric label noise introduced by residual registration errors. Because independently trained models are exposed to different empirical realizations of this spatial inconsistency, they may converge toward slightly different conditional solutions, thereby increasing inter-model variability.

Table IV supports this interpretation: ELX-trained models exhibit higher predictive variability than IMPACT-trained models in almost all regions. In Task 1, the increase reaches 41.2% in the abdomen and 29.7% across the in-distribution AB/HN/TH regions; it rises to 116.1% in the OOD pelvis and 61.8% on Ext-T2. Task 2 shows the same, although weaker, trend, with increases of 27.6% in the abdomen and 12.5% across the in-distribution regions. Thus, the registration convention with higher measured anatomical consistency is also associated with more stable predictions.

The OOD results further indicate that this effect extends beyond agreement with a particular evaluation geometry. The OOD references are independent of the registration conventions used for training and rely on either expert-refined or registration-free correspondences. Nevertheless, IMPACTtrained models consistently outperform ELX-trained models. On Ext-T2, MAE decreases from 97.43 to 88.54 HU $( p \ <$ 0.001). On Sim-CBCT, it decreases from 71.09 to 60.41 HU in the abdomen, from 149.66 to 136.18 HU in the head and neck, and from 109.15 to 95.99 HU in the thorax, with corresponding improvements in PSNR and SSIM (Table III).

Together, these findings suggest that anatomically more consistent supervision reduces sensitivity to dataset-specific geometric variations and acts as a form of regularization under distribution shift. Conversely, noisier correspondences may encourage adaptation to the spatial conventions of the training set, increasing predictive variability and reducing OOD robustness.

## D. Origin of blurring under voxel-wise ERM

The results support the interpretation introduced in Section II-A: voxel-wise blurring in supervised synthesis is not only caused by intrinsic modality ambiguity, but also by residual geometric uncertainty in the registered targets. When training references contain spatially shifted structures, voxelwise ERM favors intermediate anatomical configurations that reduce average reconstruction error without necessarily preserving the patient-specific source anatomy.

Perceptual, adversarial, or diffusion-based objectives can reduce this averaging effect by encouraging outputs that lie closer to the distribution of realistic CT images. However, these strategies do not remove the systematic bias of the supervised targets themselves: when trained on imperfectly aligned pairs, they still learn from $p ( \tilde { y } \mid x )$ rather than from the ideal anatomical distribution $p ( y \mid x )$ . Their main benefit is therefore to improve sharpness and structural coherence, not to fully correct registration-induced bias.

This creates a direct tension with reference-based evaluation. When the reference geometry is imperfectly registered, MAE may favor smoothed predictions or registrationconvention matching, even when sharper anatomy would be more desirable for downstream tasks such as segmentation or deformable registration.

## E. Why does SAM improve anatomical preservation?

The consistent improvement obtained with SAM-based perceptual supervision over both MAE- and VGG-based objectives suggests that the choice of feature representation is critical for anatomy-preserving synthesis. Rather than comparing images voxel by voxel, perceptual supervision evaluates similarity in a learned feature space, where local structures are represented together with their surrounding anatomical context. As a result, the model is less encouraged to converge toward locally averaged intensity patterns and more encouraged to preserve coherent anatomical configurations.

When formulated as an $\ell _ { 1 }$ loss in this feature space, the objective still estimates a conditional median, but over the representation induced by the encoder rather than over individual voxels. We hypothesize that this is the main driver of the observed improvement. The Hiera encoder compresses the image into low-resolution feature maps, so that each feature vector encodes a local anatomical configuration within its global context. The median would then be taken over spatially coherent configurations, a regime in which the optimal solution cannot be approximated by a blurred intensity average, which would implicitly constrain the generator to preserve sharp anatomical structures.

This also explains the advantage of SAM over VGG. Classification does not require preserving boundaries throughout the feature hierarchy, so VGG progressively discards spatial organization in favor of global appearance statistics. Segmentation does, and the SAM encoder retains boundary localization despite strong spatial compression.

The qualitative and downstream results support this interpretation. Models trained with voxel-wise losses produce smoother images that can remain favorable under MAE, but often lose thin structures and sharp anatomical interfaces, particularly around pulmonary boundaries and the diaphragm (Fig. 3). Perceptual supervision, especially with SAM features, produces sharper and more structurally coherent synthetic CT images, which in turn improves downstream segmentation performance. This illustrates a key limitation of voxel-wise evaluation: better agreement with an imperfect reference image does not necessarily imply better preservation of patientspecific anatomy.

The comparison with CT-derived controls further supports this point. These controls isolate the discrepancy caused by residual deformation without introducing synthesis error. The fact that SAM-trained predictions are closer to these controls in perceptual feature spaces suggests that SAM supervision moves synthetic CT images toward more realistic CT-like structural representations, whereas MAE optimization can remain competitive in voxel-wise error while producing anatomically smoother images.

Importantly, the relationship between perceptual supervision and voxel-wise metrics changes when spatial correspondence becomes more reliable. In the registration-free Sim-CBCT setting, SAM supervision improves both perceptual and voxelwise metrics: the ensemble MAE decreases from 111.88 to 106.33 HU, $d _ { \mathrm { S A M } }$ from 20.67 to 17.99, and LPIPS from 6.15 to 5.15 (Table VI). Under imperfect registration, voxel-wise $\ell _ { 1 }$ objectives favor blurred spatial averages; under accurate correspondence, sharp and anatomically coherent structures are also spatially consistent with the reference, and the two criteria no longer conflict.

Overall, these findings suggest that anatomy- or realismoriented synthesis methods should not be judged solely by their ability to outperform regression baselines on MAE when references are imperfectly registered. Their main contribution may lie in improving anatomical sharpness and structural coherence, with voxel-wise gains becoming visible mainly when the evaluation reference is geometrically reliable.

## F. Clinical relevance under task-specific constraints

The impact of registration-induced bias should be interpreted in light of the intended clinical application. For radiotherapy dose calculation, the dependence on imperfect paired references may remain acceptable, provided that the generated attenuation map is sufficiently accurate. This is consistent with the official SynthRAD results (Tables XIII and XI), where BreizhCT achieved strong dose-based performance despite the limitations identified in the image-based analysis.

However, this conclusion does not directly extend to all downstream tasks. Dose metrics can be relatively tolerant to local blurring, small residual deformations, or the partial suppression of fine structures, because dose calculation depends primarily on attenuation properties. In contrast, broader domain adaptation tasks such as segmentation or deformable registration require stronger preservation of patient-specific anatomy, especially near thin structures and high-gradient interfaces.

These task-dependent requirements suggest that increasingly complex generative models should be motivated by a clearly identified clinical need, rather than by visual sharpness alone. For dose calculation, voxel-wise accuracy may be sufficient in many cases; for anatomy-driven tasks, structural fidelity becomes essential.

## VI. CONCLUSION

This work shows that supervised synthetic CT generation is not only an image-to-image translation problem, but also a problem of supervision quality. When paired MRI–CT or CBCT–CT data are constructed through imperfect registration, the reference CT does not represent a true voxel-wise ground truth. Instead, residual misalignments introduce structured geometric label noise that can shape both model training and model evaluation.

Our results demonstrate that supervised models are sensitive to the registration convention used to define the training targets. Quantitative performance improves when the same registration strategy is used for both training and evaluation, indicating that voxel-wise metrics can partly reward agreement with the registration pipeline rather than faithful preservation of the source anatomy. This effect also impacts OOD behavior and predictive uncertainty: models trained from less geometrically consistent targets show reduced robustness and larger prediction variability.

We further show that these effects are reduced when training uses the registration convention with higher measured anatomical consistency. IMPACT-aligned supervision is associated with better source-anatomy preservation, lower prediction variability, and improved OOD performance in the evaluated settings. However, IMPACT is not a ground-truth alignment, and both registration conventions may retain residual errors. More generally, changing the registration method does not remove the fundamental limitation of voxel-wise supervision: when the reference is imperfectly aligned, optimizing MAE, PSNR, or SSIM can favor smoothed or geometrically biased predictions.

To address part of this limitation, we introduced SAM-based perceptual supervision. Compared with MAE-only and VGGbased perceptual objectives, SAM-based supervision improves downstream anatomical metrics and produces sharper, more structurally coherent synthetic CT images. This suggests that feature representations learned for segmentation provide a more appropriate supervisory signal for anatomy-preserving medical image synthesis than purely voxel-wise intensity losses.

Overall, our results argue that supervised synthetic CT generation should not be evaluated solely as an intensity regression problem. In the presence of imperfectly aligned references, voxel-wise metrics may reward smoothing or adaptation to the evaluation registration convention rather than true anatomical fidelity. This helps explain why highly optimized regression-based baselines remain difficult to outperform in challenge settings, and why sharper perceptual, generative, or unsupervised methods may appear quantitatively inferior despite producing more anatomically coherent images. Future benchmarks should therefore move beyond reference-based intensity metrics alone and include anatomy-oriented criteria that assess whether the synthesized CT preserves the patientspecific structures present in the input image.

## VII. DATA AND CODE AVAILABILITY

The IMPACT registrations used in this study are publicly released as Elastix B-spline transformation files, under a CC BY-NC 4.0 license. They can be applied with Transformix to the original SynthRAD images, which are not redistributed. Cases from centers whose data are restricted to challenge use are excluded.

• SynthRAD2023: https://huggingface.co/datasets/VBoussot/ synthrad2023-impact-registration

• SynthRAD2025:

https://huggingface.co/datasets/VBoussot/ synthrad2025-impact-registration

The code and configuration files needed to reproduce the submitted SynthRAD2025 solutions, implemented with KonfAI, are available for both tasks.

• Task 1:

https://github.com/vboussot/Synthrad2025 Task 1

• Task 2:

https://github.com/vboussot/Synthrad2025 Task 2

## ACKNOWLEDGMENT

The work presented in this article was supported by the French National Research Agency as part of the VATSop project (ANR-20-CE19-0015). Additionally, it was funded by the French National Research Agency as part of the DIMADOSE project (C. Hemon). While preparing this work,´ the authors used ChatGPT to enhance the writing structure and refine grammar. After using these tools, the authors reviewed and edited the manuscript and take full responsibility for its content. The authors have no relevant financial or non-financial interests to disclose.

## REFERENCES

[1] J. M. Edmund, T. Nyholm, A review of substitute ct generation for mri-only radiation therapy, Radiation Oncology 12 (1) (2017) 28.

[2] S. Dayarathna, K. T. Islam, S. Uribe, G. Yang, M. Hayat, Z. Chen, Deep learning based synthesis of mri, ct and pet: Review and analysis, Medical image analysis 92 (2024) 103046.

[3] C. A. Goodhart, Problems of monetary management: The uk experience, in: Papers in Monetary Economics, Vol. 1, Reserve Bank of Australia, Sydney, 1975, pp. 1–20.

[4] T. Nyholm, S. Svensson, S. Andersson, J. Jonsson, M. Sohlin, C. Gustafsson, E. Kjellen, K. S ´ oderstr ¨ om, P. Albertsson, L. Blomqvist,¨ et al., Mr and ct data with multiobserver delineations of organs in the pelvic area—part of the gold atlas project, Medical physics 45 (3) (2018) 1295–1300.

[5] M. Florkow, F. Zijlstra, L. Kerkmeijer, M. Maspero, C. van den Berg, M. van Stralen, P. Seevinck, The impact of mri-ct registration errors on deep learning-based synthetic ct generation, in: Medical Imaging 2019: Image Processing, Vol. 10949, SPIE, 2019, pp. 831–7.

[6] L. Kong, C. Lian, D. Huang, Y. Hu, Q. Zhou, et al., Breaking the dilemma of medical image-to-image translation, Advances in Neural Information Processing Systems 34 (2021) 1964–1978.

[7] L. Zhou, X. Ni, Y. Kong, H. Zeng, M. Xu, J. Zhou, Q. Wang, C. Liu, Mitigating misalignment in mri-to-ct synthesis for improved synthetic ct generation: an iterative refinement and knowledge distillation approach, Physics in Medicine & Biology 68 (24) (2023) 245020.

[8] C. Li, Z. Chen, Y. Zhang, L. Zhong, W. Yang, Boosting medical image synthesis via registration-guided consistency and disentanglement learning, in: International Conference on Medical Image Computing and Computer-Assisted Intervention, Springer, 2025, pp. 78–88.

[9] J. Lee, D. Kim, T. Kim, M. A. Al-Masni, Y. Han, D.-H. Kim, K. Ryu, Meta-learning guidance for robust medical image synthesis: Addressing the real-world misalignment and corruptions, Computerized Medical Imaging and Graphics 121 (2025) 102506.

[10] M. Dohmen, M. Klemens, I. Baltruschat, T. Truong, M. Lenga, Similarity metrics for mr image-to-image translation, arXiv e-prints (2024) arXiv–2405.

[11] C. Hemon, B. Texier, C. Lafond, J.-C. Nunes, A. Barateau, Towards´ trustworthy ai in radiotherapy: a comprehensive review of uncertaintyaware techniques, Physics in Medicine & Biology 71 (1) (2026) 01TR01.

[12] A. Kirillov, E. Mintun, N. Ravi, H. Mao, C. Rolland, L. Gustafson, T. Xiao, S. Whitehead, A. C. Berg, W.-Y. Lo, et al., Segment anything, in: Proceedings of the IEEE/CVF international conference on computer vision, 2023, pp. 4015–4026.

[13] J. McNaughton, J. Fernandez, S. Holdsworth, B. Chong, V. Shim, A. Wang, Machine learning for medical image translation: A systematic review, Bioengineering 10 (9) (2023) 1078.

[14] S. Kaji, S. Kida, Overview of image-to-image translation by use of deep neural networks: denoising, super-resolution, modality conversion, and reconstruction in medical imaging, arXiv preprint arXiv:1905.08603 (2019).

[15] J. Roh, D. Ryu, J. Lee, Ct synthesis with deep learning for mr-only radiotherapy planning: a review, Biomedical Engineering Letters 14 (6) (2024) 1259–1278.

[16] Y. Liu, A. Chen, H. Shi, S. Huang, W. Zheng, Z. Liu, Q. Zhang, X. Yang, Ct synthesis from mri using multi-cycle gan for head-and-neck radiation therapy, Computerized medical imaging and graphics 91 (2021) 101953.

[17] J. Dowling, L. O’Connor, O. Acosta, P. Raniga, R. de Crevoisier, J.-C. Nunes, A. Barateau, H. Chourak, J. H. Choi, P. Greer, Image synthesis for mri-only radiotherapy treatment planning, in: Biomedical Image Synthesis and Simulation, Elsevier, 2022, pp. 423–445.

[18] A. Altalib, S. McGregor, C. Li, A. Perelli, Synthetic ct image generation from cbct: a systematic review, IEEE Transactions on Radiation and Plasma Medical Sciences 9 (6) (2025) 691–707.

[19] C. M. Bishop, N. M. Nasrabadi, Pattern recognition and machine learning, Springer, 2006.

[20] R. Koenker, K. F. Hallock, Quantile regression, Journal of economic perspectives 15 (4) (2001) 143–156.

[21] S. Rassmann, D. Kugler, C. Ewert, M. Reuter, Regression is all you need¨ for medical image translation, IEEE Transactions on Medical Imaging (2026).

[22] J. Johnson, A. Alahi, L. Fei-Fei, Perceptual losses for real-time style transfer and super-resolution, in: European conference on computer vision, Springer, 2016, pp. 694–711.

[23] P. Isola, J.-Y. Zhu, T. Zhou, A. A. Efros, Image-to-image translation with conditional adversarial networks, in: Proceedings of the IEEE conference on computer vision and pattern recognition, 2017, pp. 1125–1134.

[24] J. Ho, A. Jain, P. Abbeel, Denoising diffusion probabilistic models, Advances in neural information processing systems 33 (2020) 6840– 6851.

[25] L. Zimmermann, M. Rauter, M. Schmid, D. Georg, B. Knausl, Elim-¨ inating registration bias in synthetic ct generation: A physics-based simulation framework, arXiv preprint arXiv:2602.02130 (2026).

[26] M. Rossi, P. Cerveri, Comparison of supervised and unsupervised approaches for the generation of synthetic ct from cone-beam ct, Diagnostics 11 (8) (2021) 1435.

[27] B. Xin, T. Young, C. E. Wainwright, T. Blake, L. Lebrat, T. Gaass, T. Benkert, A. Stemmer, D. Coman, J. Dowling, Deformation-aware gan for medical image synthesis with substantially misaligned pairs, arXiv preprint arXiv:2408.09432 (2024).

[28] A. Thummerer, E. van der Bijl, A. J. Galapon, F. Kamp, M. Savenije, C. Muijs, S. Aluwini, R. J. Steenbakkers, S. Beuel, M. P. Intven, et al., Synthrad2025 grand challenge dataset: Generating synthetic cts for radiotherapy from head to abdomen, Medical physics 52 (7) (2025) e17981.

[29] E. M. Huijben, M. L. Terpstra, S. Pai, A. Thummerer, P. Koopmans, M. Afonso, M. Van Eijnatten, O. Gurney-Champion, Z. Chen, Y. Zhang, et al., Generating synthetic computed tomography for radiotherapy: Synthrad2023 challenge report, Medical image analysis 97 (2024) 103276.

[30] S. Klein, M. Staring, K. Murphy, M. A. Viergever, J. P. Pluim, Elastix: a toolbox for intensity-based medical image registration, IEEE transactions on medical imaging 29 (1) (2009) 196–205.

[31] V. Boussot, C. Hemon, J.-C. Nunes, J. Dowling, S. Rouz ´ e, C. Lafond,´ A. Barateau, J.-L. Dillenseger, Impact: a generic semantic loss for multimodal medical image registration, arXiv preprint arXiv:2503.24121 (2025).

[32] Z. Zhou, M. M. Rahman Siddiquee, N. Tajbakhsh, J. Liang, Unet++: A nested u-net architecture for medical image segmentation, in: International workshop on deep learning in medical image analysis, Springer, 2018, pp. 3–11.

[33] N. Ravi, V. Gabeur, Y.-T. Hu, R. Hu, C. Ryali, T. Ma, H. Khedr, R. Radle, C. Rolland, L. Gustafson, E. Mintun, J. Pan, K. V. Al-¨ wala, N. Carion, C.-Y. Wu, R. Girshick, P. Dollar, C. Feichten-´ hofer, Sam 2: Segment anything in images and videos, arXiv preprint arXiv:2408.00714 (2024).

[34] R. Zhang, P. Isola, A. A. Efros, E. Shechtman, O. Wang, The unreasonable effectiveness of deep features as a perceptual metric, in: Proceedings

of the IEEE conference on computer vision and pattern recognition, 2018, pp. 586–595.

[35] J. A. Dowling, J. Sun, P. Pichler, D. Rivest-Henault, S. Ghose,´ H. Richardson, C. Wratten, J. Martin, J. Arm, L. Best, et al., Automatic substitute computed tomography generation and contouring for magnetic resonance imaging (mri)-alone external beam radiation therapy from standard mri sequences, International Journal of Radiation Oncology\* Biology\* Physics 93 (5) (2015) 1144–1153.

[36] S. Rit, M. Vila Oliva, S. Brousmiche, R. Labarbe, D. Sarrut, G. C. Sharp, The reconstruction toolkit (rtk), an open-source cone-beam ct reconstruction toolkit based on the insight toolkit (itk), in: Journal of Physics: Conference Series, Vol. 489, 2014, p. 012079.

[37] V. Boussot, C. Hemon, J.-C. Nunes, J.-L. Dillenseger, Why registration´ quality matters: Enhancing sct synthesis with impact-based registration, arXiv preprint arXiv:2510.21358 (2025).

[38] J. Wasserthal, H.-C. Breit, M. T. Meyer, M. Pradella, D. Hinck, A. W. Sauter, T. Heye, D. T. Boll, J. Cyriac, S. Yang, et al., Totalsegmentator: robust segmentation of 104 anatomic structures in ct images, Radiology: Artificial Intelligence 5 (5) (2023) e230024.

[39] V. Boussot, J.-L. Dillenseger, Konfai: A modular and fully configurable framework for deep learning in medical imaging, arXiv preprint arXiv:2508.09823 (2025).

The appendices provide supplementary information supporting the main experiments. Appendix A describes the SynthRAD datasets, evaluation protocol, and official challenge rankings. Appendix B reports the complete implementation details of the synthesis pipeline. Appendix C contains the full region-wise results underlying the aggregate analyses presented in the main text.

## APPENDIX A

## SYNTHRAD CHALLENGE AND BENCHMARK DETAILS

This appendix provides additional information about the SynthRAD2023 and SynthRAD2025 benchmarks used in this study. It summarizes the challenge design, dataset composition, and evaluation procedures that are relevant to the interpretation of registration-dependent performance. The official image-based and dosimetric rankings of the submitted method are also reported for completeness.

The MICCAI Challenge SynthRAD combines multi-center datasets with a standardized evaluation framework for comparing synthetic CT generation from MRI and CBCT data [28], [29]. Its two tasks address MR-to-CT synthesis for MRguided radiotherapy and CBCT-to-CT synthesis for adaptive radiotherapy.

## A. Challenge design and objectives

The challenge is organized as an open benchmarking platform, where participants are required to submit their methods in the form of containerized algorithms (Docker), ensuring full reproducibility and preventing any manual intervention during evaluation. This design enforces a strict separation between training and testing data, and enables standardized and fair comparison across competing approaches [28].

Two tasks are defined, reflecting distinct clinical scenarios:

• Task 1: MRI-to-CT synthesis, targeting MR-only and MR-guided radiotherapy workflows.

• Task 2: CBCT-to-CT synthesis, targeting CBCT-based adaptive radiotherapy.

Participants may submit models for one or both tasks, and are required to handle all anatomical regions within a given task using either a unified or region-specific strategy.

The ground-truth CT images of the validation and test sets are not directly accessible to participants. Instead, submitted algorithms are executed on hidden data, and predictions are evaluated remotely through a centralized evaluation server. The challenge is further structured into multiple phases (training, validation, and test), each associated with strict submission limits, thereby reducing the risk of iterative overfitting to the evaluation data.

At the same time, the organizers provide a highly transparent description of the evaluation pipeline, including preprocessing, registration procedures, and quantitative metrics [28]. While this improves reproducibility, it also facilitates metric-oriented optimization, where methods are progressively adapted to the benchmark criteria.

## B. Dataset characteristics

The experiments combine SynthRAD2025 data for the primary in-distribution analyses with SynthRAD2023 data for complementary brain and pelvis evaluation. The two datasets differ in cohort composition, anatomical coverage, and evaluation registration, thereby providing complementary settings for studying registration-dependent behavior.

The challenge relies on the SynthRAD2025 dataset, which comprises 2362 patient cases collected from five European university medical centers, including 890 MRI-CT pairs and 1472 CBCT-CT pairs. The benchmark addresses two synthesis tasks: MRI-to-CT synthesis (Task 1) and CBCT-to-CT synthesis (Task 2), across three anatomical regions corresponding to common radiotherapy indications: head-and-neck (HN), thorax (TH), and abdomen (AB).

The dataset is divided into training (65%), validation (10%), and test (25%) subsets. Only the training data are fully accessible to participants, while the validation and test sets remain partially hidden.

In addition to the official SynthRAD2025 benchmark, we conduct complementary local experiments using the SynthRAD2023 dataset [29] as an OOD evaluation set. This dataset contains 1080 paired MRI-CT and CBCT-CT acquisitions collected from three Dutch university medical centers, and covers two anatomical regions: brain and pelvis.

Overall, both datasets exhibit substantial multi-center and multi-protocol variability. Images are acquired using different scanners, acquisition settings, and clinical workflows, introducing significant inter-domain heterogeneity across patients and institutions. Although this variability increases the difficulty of the synthesis task, it also promotes the development of more robust and clinically generalizable models.

## C. Evaluation protocol

The official evaluation combines image similarity, segmentation-based anatomical agreement, and dosimetric accuracy. Image similarity is assessed using MAE, PSNR, and MS-SSIM between sCT and CT. Geometric agreement is evaluated from TotalSegmentator structures using Dice and Hausdorff distance, after resampling to 3 mm resolution [38]. Clinical relevance is assessed using dose error, dose-volume histogram differences, and gamma pass rates.

A key distinction concerns the registration applied before metric computation. Participants receive rigidly aligned multimodal pairs in both challenge editions. SynthRAD2025 additionally applies deformable multimodal registration during evaluation, whereas SynthRAD2023 relies on rigid alignment only. This difference motivates the separate analyses of the two benchmark settings in the main text.

## D. Official challenge rankings

Tables X and XII report the official image-based rankings for the MR-to-CT and CBCT-to-CT tasks, respectively. The corresponding dosimetric rankings are provided in Tables XI and XIII. These results document the competitiveness of the submitted method under the prescribed challenge protocol; their relation to registration-dependent benchmark behavior is discussed in the main text.

For MR-to-CT synthesis, BreizhCT ranked third in both the image-based and dosimetric evaluations. The method remained close to the highest-ranked submissions across intensity, segmentation, and dose metrics, supporting its use as a competitive model in the analyses reported in the main text.

For CBCT-to-CT synthesis, BreizhCT also ranked third overall. The submitted method achieved competitive imagesimilarity and anatomical metrics together with high gamma pass rates, confirming that the observed registration effects are not limited to a deliberately weak synthesis baseline.

## APPENDIX B

## IMPLEMENTATION DETAILS

This appendix reports the implementation details omitted from the main text for concision. The same preprocessing, architecture, optimization, checkpoint-selection, and inference procedures were used across the registration and supervision configurations unless otherwise stated. Consequently, differences between configurations primarily reflect the registration convention and training objective (loss) rather than changes to the synthesis pipeline.

The supervised synthesis framework follows the pipeline illustrated in Fig. 1. It combines registration-based pairing, patch-based preprocessing, a 2.5D convolutional generator, and ensemble-based inference. All training, validation, and inference experiments were implemented within the KonfAI framework, which was used to configure the data pipeline, model architecture, optimization strategy, checkpoint selection, test-time augmentation, and ensemble inference in a reproducible manner [39].

The synthesis model is based on a 2.5D U-Net++ architecture with a ResNet-34 encoder [32]. For each target axial slice, five adjacent slices are concatenated along the channel dimension, providing local through-plane context while preserving the computational efficiency of a 2D convolutional model. The decoder aggregates multi-scale features through the dense skip connections of U-Net++, and a final hyperbolic tangent activation constrains the output to the normalized CT intensity range. The generator contains 26,084,881 trainable parameters.

Training is performed on fixed-size in-plane patches of $3 2 0 \times 3 2 0$ , using mini-batches of size 32 and random flipping augmentation. Models are optimized with AdamW using an initial learning rate of $1 0 ^ { - 3 }$ , momentum parameters $\beta _ { 1 } = 0 . 9$ and $\beta _ { 2 } = 0 . 9 9 9 \mathrm { \Omega }$ , and a weight decay of $1 0 ^ { - 3 } .$ . A step scheduler decreases the learning rate by a factor of 0.75 every 10 epochs, with one epoch corresponding to 2500 training iterations. The final checkpoint is selected according to the lowest validation MAE and is typically obtained around 40,000 training iterations. This selection criterion is kept identical across all configurations so that the reported differences primarily reflect the effect of the registration and supervision strategies rather than differences in model selection.

At inference time, predictions are performed slice-wise using the same 2.5D implementation. Test-time augmentation is applied using flipping transformations, and predictions are averaged across augmentations. The outputs are then denormalized to recover CT intensities in Hounsfield units. For each evaluated configuration, the five models obtained from the cross-validation folds are ensembled at inference time, and final synthetic CT volumes are generated by averaging predictions across folds and augmentations.

## APPENDIX C

## COMPLETE REGION-WISE RESULTS

This appendix provides the complete region-wise results underlying the aggregate comparisons reported in the main text. It includes patient-level means and standard deviations, statistical comparisons, and separate analyses for the registration conventions and training objectives.

## A. Registration consistency

Tables XIV and XV provide the complete results for all combinations of training and evaluation registration conventions. The first column group indicates the registration used to construct the training targets, and the nested column groups indicate the registration used to define the evaluation reference. Brain and pelvis are included only for Task 1 in this comparison because they correspond to the complementary SynthRAD2023 setting.

Across regions, the detailed results support the aggregate pattern reported in Table II: performance is generally highest when the training and evaluation registration conventions are the same. The magnitude of the effect varies by anatomy, reflecting differences in deformation complexity and local image gradients.

## B. SAM-based supervision

Tables XVI and XVII report the complete Dice and SSIM results for models trained with MAE, VGG-based perceptual, or SAM-based perceptual supervision. Dice is obtained from an independent TotalSegmentator evaluation and therefore does not reuse the SAM representation employed during training.

The region-wise results show that SAM-based supervision generally improves downstream Dice, particularly in OOD regions, although SSIM does not always improve. This difference supports the conclusion that anatomical preservation and voxel-wise agreement with a registered reference provide complementary information.

![](images/0bebe2f54a8f161d99683eb316ad39a6278c5df38da426a7151d105bf2fd5f85.jpg)  
Fig. 4. Qualitative comparison of registration results between IMPACT and ELX across anatomical regions (Abdomen (AB), Head-and-Neck (HN), Thorax (TH)). For each case, the fixed image (MR), moving image (CT), overlay visualization, and checkerboard fusion are shown. In the displayed cases, IMPACTbased registration shows sharper and more spatially consistent correspondences in several boundary regions, particularly at soft-tissue interfaces, whereas ELX shows more visible local discrepancies. Checkerboard views further highlight structural inconsistencies.

TABLE X  
OFFICIAL RANKING FOR THE MR→CT SYNTHESIS TASK BASED ON THE CHALLENGE EVALUATION METRICS. PERFORMANCE IS REPORTED USING COMMONLY USED IMAGE SIMILARITY METRICS (MAE, PSNR, MS-SSIM) AND SEGMENTATION-BASED METRICS (DICE, HD95). THE TOP 5 PERFORMING TEAMS AND THE CHALLENGE BASELINE ARE SHOWN. OUR METHOD (BREIZHCT) RANKS 3RD OVERALL.
<table><tr><td>#</td><td>Team</td><td>MAE↓</td><td>PSNR↑</td><td>MS-SSIM↑</td><td>Dice↑</td><td>HD95↓</td></tr><tr><td>1</td><td>FelixSun (KoalAI)</td><td>64.81</td><td>29.997</td><td>0.936</td><td>0.779</td><td>6.01</td></tr><tr><td>2</td><td>JavierSequeiro</td><td>65.51</td><td>29.611</td><td>0.933</td><td>0.766</td><td>6.32</td></tr><tr><td>3</td><td>Valentin (BreizhCT)</td><td>67.24</td><td>29.957</td><td>0.935</td><td>0.737</td><td>7.51</td></tr><tr><td>4</td><td>siyuanmei (MixCT)</td><td>67.90</td><td>29.628</td><td>0.931</td><td>0.785</td><td>5.77</td></tr><tr><td>5</td><td>hanbingocean (QWER)</td><td>75.68</td><td>28.756</td><td>0.922</td><td>0.715</td><td>7.69</td></tr><tr><td>14</td><td>MaartenTerpstra (baseline)</td><td>309.28</td><td>18.630</td><td>0.466</td><td>0.006</td><td>136.34</td></tr></table>

TABLE XI

OFFICIAL RANKING FOR THE MR→CT SYNTHESIS TASK BASED ON DOSIMETRIC EVALUATION METRICS. PERFORMANCE IS REPORTED USING DOSE-BASED METRICS (DOSE MAE, DVH) AND GAMMA PASS RATE (GPR), FOR BOTH γ AND p CRITERIA. THE TOP 5 PERFORMING TEAMS AND THE CHALLENGE BASELINE ARE SHOWN. OUR METHOD (BREIZHCT) RANKS 3RD OVERALL.
<table><tr><td>#</td><td>Team</td><td>Dose  $\mathbf { M A E } _ { \gamma } \mathbf { \Omega } .$ </td><td> $\mathbf { D o s e \ M A E } _ { p } \downarrow$ </td><td> $\mathbf { D V H } _ { \gamma }$  →</td><td> $\mathbf { D V H } _ { p } \downarrow$ </td><td> $\mathbf { G P R } _ { \gamma }$  ↑</td><td> $\mathbf { G P R } _ { p } \uparrow$ </td></tr><tr><td>1</td><td>FelixSun (KoalAI)</td><td>0.006</td><td>0.024</td><td>0.011</td><td>0.064</td><td>98.33</td><td>84.04</td></tr><tr><td>2</td><td>JavierSequeiro</td><td>0.006</td><td>0.024</td><td>0.011</td><td>0.060</td><td>98.50</td><td>84.56</td></tr><tr><td>3</td><td>Valentin (BreizhCT)</td><td>0.006</td><td>0.027</td><td>0.013</td><td>0.067</td><td>98.88</td><td>82.19</td></tr><tr><td>4</td><td>siyuanmei (MixCT)</td><td>0.007</td><td>0.023</td><td>0.016</td><td>0.075</td><td>98.22</td><td>81.92</td></tr><tr><td>5</td><td>hanbingocean (QWER)</td><td>0.007</td><td>0.027</td><td>0.014</td><td>0.073</td><td>98.29</td><td>81.91</td></tr><tr><td>14</td><td>MaartenTerpstra (baseline)</td><td>0.056</td><td>0.152</td><td>0.094</td><td>0.451</td><td>78.08</td><td>59.59</td></tr></table>

TABLE XII

OFFICIAL RANKING FOR THE CBCT→CT SYNTHESIS TASK BASED ON CHALLENGE EVALUATION METRICS. PERFORMANCE IS REPORTED USING COMMONLY USED IMAGE SIMILARITY METRICS (MAE, PSNR, MS-SSIM) AND SEGMENTATION-BASED METRICS (DICE, HD95). THE TOP 5 PERFORMING TEAMS AND THE CHALLENGE BASELINE ARE SHOWN. OUR METHOD (BREIZHCT) RANKS 3RD OVERALL.
<table><tr><td>#</td><td>Team</td><td>MAE↓</td><td>PSNR↑</td><td>MS-SSIM↑</td><td>Dice↑</td><td>HD95↓</td></tr><tr><td>1</td><td>GlassCity (MixCT)</td><td>48.27</td><td>32.62</td><td>0.968</td><td>0.857</td><td>4.53</td></tr><tr><td>2</td><td>JavierSequeiro</td><td>52.49</td><td>31.91</td><td>0.964</td><td>0.846</td><td>4.87</td></tr><tr><td>3</td><td>Valentin (BreizhCT)</td><td>53.09</td><td>32.49</td><td>0.966</td><td>0.843</td><td>5.08</td></tr><tr><td>4</td><td>ayuan (et)</td><td>53.61</td><td>31.89</td><td>0.963</td><td>0.842</td><td>4.99</td></tr><tr><td>5</td><td>RicardoBrioso</td><td>62.75</td><td>31.01</td><td>0.952</td><td>0.801</td><td>6.65</td></tr><tr><td>14</td><td>MaartenTerpstra (baseline)</td><td>308.99</td><td>18.77</td><td>0.505</td><td>0.004</td><td>133.48</td></tr></table>

TABLE XIII

OFFICIAL RANKING FOR THE CBCT→CT SYNTHESIS TASK BASED ON DOSIMETRIC EVALUATION METRICS. PERFORMANCE IS REPORTED USING DOSE-BASED METRICS (DOSE MAE, DVH) AND GAMMA PASS RATE (GPR), FOR BOTH γ AND p CRITERIA. THE TOP 5 PERFORMING TEAMS AND THE CHALLENGE BASELINE ARE SHOWN. OUR METHOD (BREIZHCT) RANKS 3RD OVERALL.
<table><tr><td>#</td><td>Team</td><td>Dose MAEγ ↓</td><td> $\mathbf { D o s e \ M A E } _ { p } \downarrow$ </td><td> $ { \mathbf { D } }  { \mathbf { V } }  { \mathbf { H } } _ { \gamma } \ \downarrow$ </td><td> $\mathbf { D V H } _ { p } \downarrow$ </td><td>GPRγ ↑</td><td> $\mathbf { G P R } _ { p } \uparrow$ </td></tr><tr><td>1</td><td>GlassCity (MixCT)</td><td>0.004</td><td>0.017</td><td>0.013</td><td>0.034</td><td>99.300</td><td>88.640</td></tr><tr><td>2</td><td>JavierSequeiro</td><td>0.005</td><td>0.018</td><td>0.015</td><td>0.036</td><td>99.312</td><td>87.765</td></tr><tr><td>3</td><td>Valentin (BreizhCT)</td><td>0.005</td><td>0.020</td><td>0.015</td><td>0.036</td><td>99.308</td><td>86.407</td></tr><tr><td>4</td><td>ayuan (et)</td><td>0.005</td><td>0.018</td><td>0.015</td><td>0.039</td><td>99.227</td><td>87.405</td></tr><tr><td>5</td><td>RicardoBrioso</td><td>0.006</td><td>0.021</td><td>0.017</td><td>0.046</td><td>98.948</td><td>84.989</td></tr><tr><td>14</td><td>MaartenTerpstra (baseline)</td><td>0.036</td><td>0.094</td><td>0.184</td><td>0.482</td><td>76.393</td><td>59.136</td></tr></table>

## C. Perceptual and ensemble results

Tables XVIII and XIX provide the complete regional comparison between MAE- and SAM-trained models. Mean CV denotes patient-level performance averaged over the five individual fold models, whereas CV denotes evaluation after voxel-wise averaging of the five predictions.

Across both tasks, SAM supervision improves $d _ { \mathrm { S A M } }$ and LPIPS in the in-distribution regions while often increasing MAE. In the Ext-T2 and Sim-CBCT settings, it improves both perceptual and voxel-wise metrics. Ensemble averaging primarily reduces MAE, whereas its effect on perceptual distances is smaller, consistent with averaging reducing random intensity errors while potentially smoothing fine structures.

TABLE XIV  
QUANTITATIVE RESULTS FOR TASK 1 (SUPERVISED CROSS-VALIDATION) ACROSS ANATOMICAL REGIONS. MEAN ± STANDARD DEVIATION OF COMMONLY USED IMAGE SIMILARITY METRICS (MAE, PSNR, SSIM) ARE REPORTED. COLUMN GROUPS INDICATE THE REGISTRATION METHOD USED DURING TRAINING (IMPACT OR ELX), WHILE SUBGROUPS CORRESPOND TO THE REGISTRATION METHOD USED FOR EVALUATION. REGIONS AB, HN, AND TH CORRESPOND TO IN-DISTRIBUTION DATA, WHEREAS BRAIN AND PELVIS REPRESENT OUT-OF-DISTRIBUTION REGIONS NOT SEEN DURING TRAINING. STATISTICAL COMPARISONS AGAINST IMPACT/IMPACT CONFIGURATION FOLLOW THE PATIENT-LEVEL PROTOCOL DESCRIBED IN SECTION III-F.
<table><tr><td></td><td colspan="6">IMPACT</td><td colspan="6">ELX</td></tr><tr><td>Region</td><td colspan="3">IMPACT</td><td colspan="3">ELX</td><td colspan="3">IMPACT</td><td colspan="3">ELX</td></tr><tr><td></td><td>MAE</td><td>PSNR</td><td>SSIM</td><td>MAE</td><td>PSNR</td><td>SSIM</td><td>MAE</td><td>PSNR</td><td>SSIM</td><td>MAE</td><td>PSNR</td><td>SSIM</td></tr><tr><td></td><td>58.89</td><td>29.94</td><td>0.909</td><td>72.14***</td><td>27.68***</td><td>0.899**</td><td>64.04***</td><td>29.29***</td><td>0.900***</td><td>67.73**</td><td>28.47***</td><td>0.904</td></tr><tr><td>AB</td><td>±9.16</td><td>±1.43</td><td>±0.022</td><td>±13.78</td><td>±1.57</td><td>±0.032</td><td>±8.78</td><td>±1.23</td><td>±0.025</td><td>±14.25</td><td>±1.73</td><td>±0.032</td></tr><tr><td>HN</td><td>76.21</td><td>28.56</td><td>0.931</td><td>88.51*</td><td>27.43*</td><td>0.915</td><td>82.10***</td><td>28.00***</td><td>0.924***</td><td>78.98</td><td>28.38</td><td>0.925</td></tr><tr><td></td><td>±11.31</td><td>±1.29</td><td>±0.027</td><td>±28.29</td><td>±2.74</td><td>±0.043</td><td>±12.83</td><td>±1.35</td><td>±0.030</td><td>±23.41</td><td>±2.41</td><td>±0.038</td></tr><tr><td>TH</td><td>56.44 ±10.81</td><td>31.29 ±1.86</td><td>0.941 ±0.018</td><td>58.14</td><td>30.23</td><td>0.940</td><td>59.81***</td><td>30.70***</td><td>0.936***</td><td>55.30</td><td>30.74</td><td>0.943 ±0.028</td></tr><tr><td></td><td>63.55</td><td>29.97</td><td>0.927</td><td>±11.12 72.47***</td><td>±1.74 28.49***</td><td>±0.027 0.918**</td><td>±9.85 68.31***</td><td>±1.56 29.37***</td><td>±0.022 0.920***</td><td>±10.14 66.98</td><td>±1.66 29.24**</td><td>0.924</td></tr><tr><td>AB/HN/TH</td><td>±13.61</td><td>±1.91</td><td>±0.026</td><td>±22.68</td><td>±2.42</td><td>±0.038</td><td>±14.27</td><td>±1.78</td><td>±0.030</td><td>±19.27</td><td>±2.24</td><td>±0.036</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Brain</td><td>114.55</td><td>24.81</td><td>0.858</td><td>138.52***</td><td>23.37***</td><td>0.837***</td><td>117.32*</td><td>24.66*</td><td>0.856</td><td>140.32***</td><td>23.35***</td><td>0.836***</td></tr><tr><td></td><td>±11.10</td><td>±0.80</td><td>±0.021</td><td>±16.24</td><td>±0.98</td><td>±0.023</td><td>±10.47</td><td>±0.73</td><td>±0.020</td><td>±18.43</td><td>±1.06</td><td>±0.021</td></tr><tr><td></td><td>65.40</td><td>28.67</td><td>0.850</td><td>92.23***</td><td>25.70***</td><td>0.804***</td><td>72.43***</td><td>28.14***</td><td>0.825***</td><td>99.17***</td><td>25.43***</td><td>0.781***</td></tr><tr><td>Pelvis</td><td>±13.08</td><td>±1.75</td><td>±0.051</td><td>±17.13</td><td>±1.43</td><td>±0.050</td><td>±18.31</td><td>±1.71</td><td>±0.085</td><td>±20.31</td><td>±1.38</td><td>±0.079</td></tr><tr><td></td><td>91.94</td><td>26.59</td><td>0.854</td><td>117.23***</td><td>24.44***</td><td>0.822***</td><td>96.67***</td><td>26.26***</td><td>0.842***</td><td>121.39***</td><td>24.31***</td><td>0.811***</td></tr><tr><td>Brain/Pelvis</td><td>±27.30</td><td>±2.34</td><td>±0.038</td><td>±28.45</td><td>±1.67</td><td>±0.041</td><td>±26.72</td><td>±2.16</td><td>±0.061</td><td>±28.18</td><td>±1.60</td><td>±0.062</td></tr></table>

TABLE XV  
QUANTITATIVE RESULTS FOR TASK 2 (SUPERVISED CROSS-VALIDATION) ACROSS ANATOMICAL REGIONS. MEAN ± STANDARD DEVIATION OF COMMONLY USED IMAGE SIMILARITY METRICS (MAE, PSNR, SSIM) ARE REPORTED. COLUMN GROUPS INDICATE THE REGISTRATION METHOD USED DURING TRAINING (IMPACT OR ELX), WHILE SUBGROUPS CORRESPOND TO THE REGISTRATION METHOD USED FOR EVALUATION. ALL REGIONS CORRESPOND TO IN-DISTRIBUTION DATA USED DURING TRAINING. STATISTICAL COMPARISONS AGAINST IMPACT/IMPACT CONFIGURATION FOLLOW THE PATIENT-LEVEL PROTOCOL DESCRIBED IN SECTION III-F.
<table><tr><td></td><td colspan="6">IMPACT</td><td colspan="6">ELX</td></tr><tr><td>Region</td><td colspan="3">IMPACT</td><td colspan="3">ELX</td><td colspan="3">IMPACT</td><td colspan="3">ELX</td></tr><tr><td></td><td>MAE</td><td>PSNR</td><td>SSIM</td><td>MAE</td><td>PSNR</td><td>SSIM</td><td>MAE</td><td>PSNR</td><td>SSIM</td><td>MAE</td><td>PSNR</td><td>SSIM</td></tr><tr><td></td><td>55.30</td><td>31.17</td><td>0.928</td><td>67.10***</td><td>28.96***</td><td>0.909***</td><td>62.96***</td><td>29.88***</td><td>0.910***</td><td>59.08</td><td>30.12*</td><td>0.920**</td></tr><tr><td>AB</td><td>±14.14</td><td>±2.40</td><td>±0.023</td><td>±13.71</td><td>±1.65</td><td>±0.030</td><td>±13.70</td><td>±1.79</td><td>±0.031</td><td>±10.64</td><td>±1.57</td><td>±0.025</td></tr><tr><td>HN</td><td>64.84</td><td>30.20</td><td>0.950</td><td>77.18***</td><td>28.33***</td><td>0.935***</td><td>71.74***</td><td>29.40***</td><td>0.938***</td><td>68.55**</td><td>29.29**</td><td>0.943***</td></tr><tr><td></td><td>±14.73</td><td>±2.07</td><td>±0.019</td><td>±17.73</td><td>±1.94</td><td>±0.024</td><td>±16.89</td><td>±2.09</td><td>±0.025</td><td>±13.92</td><td>±1.89</td><td>±0.020</td></tr><tr><td>TH</td><td>55.40</td><td>32.21</td><td>0.930</td><td>62.41**</td><td>30.15***</td><td>0.913***</td><td>61.72***</td><td>31.02***</td><td>0.912***</td><td>55.43</td><td>31.32*</td><td>0.925</td></tr><tr><td></td><td>±15.43</td><td>±2.69</td><td>±0.018</td><td>±12.66</td><td>±1.95</td><td>±0.028</td><td>±16.16</td><td>±2.45</td><td>±0.023</td><td>±12.51</td><td>±2.15</td><td>±0.024</td></tr><tr><td></td><td>58.76</td><td>31.16</td><td>0.937</td><td>69.17***</td><td>29.13***</td><td>0.920***</td><td>65.70***</td><td>30.08***</td><td>0.921***</td><td>61.28**</td><td>30.22***</td><td>0.930***</td></tr><tr><td>AB/HN/TH</td><td>±15.47</td><td>±2.53</td><td>±0.022</td><td>±16.24</td><td>±2.01</td><td>±0.030</td><td>±16.36</td><td>±2.24</td><td>±0.030</td><td>±13.72</td><td>±2.07</td><td>±0.025</td></tr></table>

TABLE XVI  
QUANTITATIVE RESULTS FOR TASK 1 USING IMPACT-BASED TRAINING, COMPARING MODELS TRAINED WITH MAE, VGG-BASED PERCEPTUAL, AND SAM-BASED LOSSES. DICE EVALUATES DOWNSTREAM SEGMENTATION PERFORMANCE, WHILE SSIM MEASURES IMAGE SIMILARITY BETWEEN SYNTHESIZED CT AND REFERENCE CT. VALUES ARE REPORTED AS MEAN ± STANDARD DEVIATION ACROSS MATCHED PATIENTS. STATISTICAL COMPARISONS AGAINST SAM FOLLOW THE PATIENT-LEVEL PROTOCOL DESCRIBED IN SECTION III-F.
<table><tr><td rowspan="2">Region</td><td colspan="2">MAE</td><td colspan="2">VGG</td><td colspan="2">SAM</td></tr><tr><td>Dice</td><td>SSIM</td><td>Dice</td><td>SSIM</td><td>Dice</td><td>SSIM</td></tr><tr><td>AB</td><td>0.737*** ±0.044</td><td>0.909*** ±0.022</td><td>0.737*** ±0.046</td><td>0.904 ±0.023</td><td>0.777 ±0.043</td><td>0.905 ±0.022</td></tr><tr><td>HN</td><td>0.718 ±0.067</td><td>0.931*** ±0.027</td><td>0.713** ±0.067</td><td>0.918 ±0.033</td><td>0.731 ±0.060</td><td>0.914 ±0.037</td></tr><tr><td>TH</td><td>0.679*** ±0.041</td><td>0.941*** ±0.018</td><td>0.687*** ±0.038</td><td>0.934** ±0.020</td><td>0.706 ±0.031</td><td>0.938 ±0.020</td></tr><tr><td>AB/HN/TH</td><td>0.711*** ±0.057</td><td>0.927*** ±0.026</td><td>0.712*** ±0.055</td><td>0.919 ±0.029</td><td>0.738 ±0.055</td><td>0.919 ±0.030</td></tr><tr><td>Brain</td><td>0.778** ±0.121 0.715***</td><td>0.858 ±0.021 0.850</td><td>0.813 ±0.100 0.725***</td><td>0.859 ±0.019 0.844*</td><td>0.816 ±0.089 0.795</td><td>0.857 ±0.022</td></tr><tr><td>Pelvis</td><td>±0.120 0.749***</td><td>±0.051 0.854</td><td>±0.115 0.772***</td><td>±0.046 0.852</td><td>±0.048 0.806</td><td>0.850 ±0.048 0.854</td></tr><tr><td>Brain/Pelvis</td><td>±0.125</td><td>±0.038</td><td>±0.116</td><td>±0.035</td><td>±0.074</td><td>±0.036</td></tr><tr><td>Ext-T2</td><td>0.665*** ±0.063</td><td>0.899* ±0.025</td><td>0.672*** ±0.059</td><td>0.904*** ±0.023</td><td>0.725 ±0.047</td><td>0.893 ±0.032</td></tr></table>

TABLE XVII  
QUANTITATIVE RESULTS FOR TASK 2 USING IMPACT-BASED TRAINING, COMPARING MODELS TRAINED WITH MAE, VGG-BASED PERCEPTUAL, AND SAM-BASED LOSSES. DICE EVALUATES DOWNSTREAM SEGMENTATION PERFORMANCE, WHILE SSIM MEASURES IMAGE SIMILARITY BETWEEN SYNTHESIZED CT AND REFERENCE CT. VALUES ARE REPORTED AS MEAN ± STANDARD DEVIATION. STATISTICAL COMPARISONS AGAINST SAM FOLLOW THE PATIENT-LEVEL PROTOCOL DESCRIBED IN SECTION III-F. $A B _ { \mathrm { s i m } } , H N _ { \mathrm { s i m } } ,$ AND $T H _ { \mathrm { s i m } }$ CORRESPOND TO THE REGISTRATION-FREE SIMULATED CBCT EVALUATIONS.
<table><tr><td rowspan="2">Region</td><td colspan="2">MAE_IMPACT</td><td colspan="2">VGG_IMPACT</td><td colspan="2">SAM_IMPACT</td></tr><tr><td>Dice</td><td>SSIM</td><td>Dice</td><td>SSIM</td><td>Dice</td><td>SSIM</td></tr><tr><td>AB</td><td>0.642*** ±0.086</td><td>0.928*** ±0.023</td><td> $0 . 6 4 0 ^ { * * * }$  ±0.083</td><td>0.925 ±0.024</td><td>0.670 ±0.086</td><td>0.925 ±0.024</td></tr><tr><td>HN</td><td>0.720*** ±0.078</td><td>0.950*** ±0.019</td><td>0.697*** ±0.083</td><td>0.943 ±0.022</td><td>0.708 ±0.076</td><td>0.943 ±0.022</td></tr><tr><td>TH</td><td>0.718 ±0.076</td><td>0.930*** ±0.018</td><td>0.709*** ±0.076</td><td>0.928 ±0.019</td><td>0.720 ±0.077</td><td>0.927</td></tr><tr><td>AB/HN/TH</td><td>0.695 ±0.088</td><td>0.937***</td><td>0.683***</td><td>0.932</td><td>0.700</td><td>±0.020 0.932</td></tr><tr><td></td><td>0.709***</td><td>±0.022 0.927***</td><td>±0.086 0.723***</td><td>±0.023 0.921***</td><td>±0.082 0.764</td><td>±0.023 0.932</td></tr><tr><td> $A B _ { \mathrm { s i m } }$ </td><td>±0.085  $0 . 7 3 4 ^ { * * * }$ </td><td>±0.016 0.901***</td><td>±0.077 0.750*</td><td>±0.016 0.902***</td><td>±0.079 0.755</td><td>±0.015 0.907</td></tr><tr><td> $H N _ { \mathrm { s i m } }$ </td><td>±0.065  $0 . 7 2 2 ^ { * * }$ </td><td>±0.029 0.885***</td><td>±0.061 0.741***</td><td>±0.028 0.888***</td><td>±0.063 0.758</td><td>±0.028 0.892</td></tr><tr><td> $T H _ { \mathrm { s i m } }$ </td><td>±0.120</td><td>±0.062</td><td>±0.114</td><td>±0.057</td><td>±0.110</td><td>±0.059</td></tr></table>

TABLE XVIII  
QUANTITATIVE RESULTS FOR TASK 1 USING MODELS TRAINED WITH ELX AND EVALUATED WITH ELX, COMPARING MAE- AND SAM-BASED TRAINING LOSSES. “CV” DENOTES THE ENSEMBLE PREDICTION OBTAINED BY AVERAGING THE OUTPUTS OF THE FIVE CROSS-VALIDATION MODELS $\left( \mathbf { C V _ { 0 } } \mathbf { - C V _ { 4 } } \right)$ , WHILE “MEAN $\mathrm { C V } ^ { , , }$ CORRESPONDS TO THE AVERAGE PERFORMANCE OF THE INDIVIDUAL MODELS. PERFORMANCE IS REPORTED USING MAE, $d _ { \mathrm { S A M } } .$ , AND LPIPS (MEAN ± STANDARD DEVIATION) COMPUTED ON MATCHED PATIENTS. d<sub>SAM</sub> AND LPIPS VALUES ARE SCALED BY A FACTOR OF 100 FOR READABILITY.
<table><tr><td>Region</td><td colspan="6">Mean CV</td><td colspan="6">CV</td></tr><tr><td></td><td colspan="3">MAE</td><td colspan="3">SAM</td><td colspan="3">MAE</td><td colspan="3">SAM</td></tr><tr><td></td><td>MAE</td><td>SAM</td><td>LPIPS</td><td>MAE</td><td>SAM</td><td>LPIPS</td><td>MAE</td><td>SAM</td><td>LPIPS</td><td>MAE</td><td>SAM</td><td>LPIPS</td></tr><tr><td></td><td>71.30</td><td>32.54</td><td>12.67</td><td>74.00</td><td>25.64</td><td>10.66</td><td>67.73</td><td>32.46</td><td>12.49</td><td>69.35</td><td>26.70</td><td>10.66</td></tr><tr><td>AB</td><td>±14.44</td><td>±5.94</td><td>±3.42</td><td>±13.84</td><td>±4.38</td><td>±2.79</td><td>±14.25</td><td>±6.07</td><td>±3.35</td><td>±14.04</td><td>±4.50</td><td>±2.81</td></tr><tr><td>HN</td><td>83.51</td><td>14.54</td><td>3.46</td><td>99.88</td><td>11.63</td><td>3.11</td><td>78.98</td><td>14.76</td><td>3.41</td><td>93.66</td><td>11.96</td><td>3.05</td></tr><tr><td></td><td>±24.76 58.57</td><td>±2.37 25.25</td><td>±0.63 8.66</td><td>±33.98 62.60</td><td>±1.76 19.35</td><td>±0.54</td><td>±23.41</td><td>±2.37</td><td>±0.64</td><td>±33.16</td><td>±1.82</td><td>±0.55 7.18</td></tr><tr><td>TH</td><td>±10.54</td><td>±5.66</td><td></td><td></td><td></td><td>7.23</td><td>55.30</td><td>25.24</td><td>8.52</td><td>58.22</td><td>20.50</td><td></td></tr><tr><td></td><td>70.75</td><td>24.27</td><td>±2.88 8.34</td><td>±10.62 78.26</td><td>±4.77 18.99</td><td>±2.33 7.06</td><td>±10.14 66.98</td><td>±5.71 24.31</td><td>±2.86 8.22</td><td>±10.12</td><td>±5.19</td><td>±2.41 7.03</td></tr><tr><td>AB/HN/TH</td><td>±20.17</td><td>±8.83</td><td>±4.56</td><td>±26.66</td><td>±6.88</td><td>±3.72</td><td>±19.27</td><td>±8.77</td><td>±4.49</td><td>73.21</td><td>19.85</td><td>±3.77</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>±25.84</td><td>±7.26</td><td></td></tr><tr><td></td><td>104.39</td><td>37.43</td><td>13.82</td><td>96.36</td><td>33.71</td><td>12.00</td><td>97.43</td><td>37.30</td><td>13.25</td><td>90.14</td><td>35.93</td><td>11.79</td></tr><tr><td>Ext-T2</td><td>±16.92</td><td>±4.39</td><td>±2.95</td><td>±15.38</td><td>±3.86</td><td>±2.51</td><td>±14.83</td><td>±4.17</td><td>±2.95</td><td>±14.09</td><td>±3.93</td><td>±2.58</td></tr></table>

TABLE XIX  
QUANTITATIVE RESULTS FOR TASK 2 USING MODELS TRAINED WITH ELX AND EVALUATED WITH ELX, COMPARING MAE- AND SAM-BASED TRAINING LOSSES. $\mathbf { \Delta } ^ { * } \mathbf { C } \mathbf { V } ^ { * }$ DENOTES THE ENSEMBLE PREDICTION OBTAINED BY AVERAGING THE OUTPUTS OF THE FIVE CROSS-VALIDATION MODELS $\scriptstyle ( \mathbf { C V _ { 0 } - C V _ { 4 } } )$ , WHILE “MEAN $\mathrm { C V } ^ { , , }$ CORRESPONDS TO THE AVERAGE PERFORMANCE OF THE INDIVIDUAL MODELS. PERFORMANCE IS REPORTED USING $\mathbf { M A E } , d _ { \mathrm { S A M } }$ , AND LPIPS (MEAN ± STANDARD DEVIATION). $d _ { \mathrm { S A M } }$ AND LPIPS VALUES ARE SCALED BY A FACTOR OF 100 FOR READABILITY.
<table><tr><td>Region</td><td colspan="6">Mean CV</td><td colspan="6">CV</td></tr><tr><td></td><td colspan="3">MAE</td><td colspan="3">SAM</td><td colspan="3">MAE</td><td colspan="3">SAM</td></tr><tr><td></td><td>MAE</td><td>SAM</td><td>LPIPS</td><td>MAE</td><td>SAM</td><td>LPIPS</td><td>MAE</td><td>SAM</td><td>LPIPS</td><td>MAE</td><td>SAM</td><td>LPIPS</td></tr><tr><td></td><td>62.55</td><td>21.83</td><td>7.82</td><td>66.39</td><td>17.57</td><td>6.69</td><td>59.08</td><td>22.19</td><td>7.73</td><td>62.30</td><td>18.10</td><td>6.60</td></tr><tr><td>AB</td><td>±10.57</td><td>±5.95</td><td>±2.95</td><td>±11.88</td><td>±4.06</td><td>±2.35</td><td>±10.64</td><td>±6.32</td><td>±2.99</td><td>±12.00</td><td>±4.75</td><td>±2.45</td></tr><tr><td></td><td>72.35</td><td>11.48</td><td>2.48</td><td>78.71</td><td>9.40</td><td>2.22</td><td>68.55</td><td>11.75</td><td>2.42</td><td>74.24</td><td>9.50</td><td>2.09</td></tr><tr><td>HN</td><td>±13.72</td><td>±2.20</td><td>±0.70</td><td>±14.05</td><td>±1.53</td><td>±0.61</td><td>±13.92</td><td>±2.39</td><td>±0.71</td><td>±14.18</td><td>±1.64</td><td>±0.61</td></tr><tr><td></td><td>59.16</td><td>20.09</td><td>6.53</td><td>62.46</td><td>15.63</td><td>5.51</td><td>55.43</td><td>20.51</td><td>6.44</td><td>58.05</td><td>16.12</td><td>5.42</td></tr><tr><td>TH</td><td>±12.89</td><td>±4.91</td><td>±2.31</td><td>±12.90</td><td>±3.81</td><td>±1.85</td><td>±12.51</td><td>±4.86</td><td>±2.28</td><td>±12.66</td><td>±3.90</td><td>±1.88</td></tr><tr><td></td><td>64.95</td><td>17.54</td><td>5.48</td><td>69.52</td><td>13.99</td><td>4.69</td><td>61.28</td><td>17.88</td><td>5.40</td><td>65.18</td><td>14.36</td><td>4.59</td></tr><tr><td>AB/HN/TH</td><td>±13.77</td><td>±6.46</td><td>±3.15</td><td>±14.82</td><td>±4.82</td><td>±2.58</td><td>±13.72</td><td>±6.62</td><td>±3.15</td><td>±14.79</td><td>±5.18</td><td>±2.63</td></tr><tr><td></td><td>114.78</td><td>20.87</td><td>6.34</td><td>109.23</td><td>17.94</td><td>5.31</td><td>111.88</td><td>20.67</td><td>6.15</td><td>106.33</td><td>17.99</td><td>5.15</td></tr><tr><td>Sim-CBCT</td><td>±46.81</td><td>±6.07</td><td>±2.87</td><td>±43.70</td><td>±4.63</td><td>±2.32</td><td>±46.90</td><td>±6.10</td><td>±2.80</td><td>±44.17</td><td>±5.00</td><td>±2.34</td></tr></table>