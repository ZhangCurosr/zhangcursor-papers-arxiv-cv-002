# Evidence Before Accuracy: A MRI-PET Fusion Network for Alzheimer’s Disease Classification with Causal Regional Validation

Saeid Firouzi Daghigh and Saeed Ayat

Abstract—Deep learning models for Alzheimer’s disease (AD) classification routinely report near-perfect discrimination, yet few are shown to rest on AD-relevant neurobiology rather than on dataset artifacts, subject-level leakage, or non-brain image content. We present a six-stream 2.5D fusion network combining T1 MRI and FDG PET across axial, coronal, and sagittal planes, trained on a strictly subject-disjoint ADNI consists of 554 paired subjects. The fusion model reaches AUC 0.962, accuracy 0.909, and F1 0.891, competitive with recent 3D CNN and multimodal transformer systems at substantially lower cost. We first quantify how much modality, plane and slice geometry matter. A validation-only search over slice centres and neighbour spacings moves AUC by 0.180 for MRI and 0.078 for PET, selecting narrow spacing for MRI (∆ = 4 mm) and wide spacing for PET (∆ = 16 mm), with the chosen coronal centres falling on the hippocampal body and on the posterior cingulate/precuneus respectively. The contribution, however, is the evidence layer built around that number. Shortcut controls collapse the model to AUC 0.622 (silhouette), 0.608 (exterior), and 0.500 (blank), and a label-permutation null yields 0.456. Forward region-of-interest (ROI) ablation shows that masking medial temporal cortex in MRI and the posterior default-mode network (DMN) in PET produces the largest shift in the AD logit, while area-matched controls remain indistinguishable from that null. Reverse ROI ablation shows that the medial temporal lobe alone retains 89.2% of above-chance discrimination in MRI and the posterior DMN alone retains 79.0% in PET, versus 31.6% or less for areamatched and enlarged controls. A quantitative comparison of four attribution methods shows occlusion sensitivity reaching 3.5- 5.0× enrichment inside a priori AD regions against 0.10-0.43× in controls. Ablation and attribution independently establish a biologically correct double dissociation: hippocampal evidence is carried by MRI, posterior cingulate and precuneus evidence by PET. We argue that control-anchored causal validation of this kind should be a reporting standard, and that gradientbased saliency alone is insufficient evidence of biological validity. Source code, pre-trained models, and test data results are publicly available at: https://github.com/saeed5959/ad

Index Terms—Alzheimer’s disease, FDG-PET, structural MRI, explainable AI, occlusion sensitivity.

## I. INTRODUCTION

LZHEIMER’S disease affects tens of millions of people worldwide and remains without a disease-modifying   
cure, which places a high value on accurate and early char  
acterisation of the disease state [1]. AD is diagnosed in vivo   
through a convergence of structural, metabolic, and molecular

evidence. Structural MRI captures the medial temporal atrophy that tracks neurodegeneration and correlates with Braak staging [3], while FDG-PET captures the temporoparietal and posterior cingulate hypometabolism that often precedes measurable volume loss [4]. Because the two modalities index different and only partially overlapping pathological processes, their combination improves detection and differential diagnosis relative to either alone [5]. This complementarity is the standard motivation for multimodal deep learning in AD, and recent systematic reviews confirm that multimodal models consistently outperform unimodal ones across datasets and architectures [24].

The methodological quality of that literature, however, is uneven. Reported accuracies above 0.98 are common, yet a substantial fraction are produced under conditions that make them uninterpretable: slice-level rather than subject-level splitting, which places slices from the same brain in both training and test sets; checkpoint selection on the test set; test sets of a few dozen subjects, for which the standard error of AUC alone is on the order of ±0.04; and evaluation on a single split, which makes cross-experiment comparison meaningless. Under these conditions a high number is evidence of a leak, not of learning.

A second and deeper problem is that even a correctly validated classifier may reach the right answer for the wrong reason. Medical imaging models are known to exploit acquisition signatures, head shape, skull thickness, field-of-view differences, and other confounds that correlate with class membership in a given cohort but carry no clinical meaning. The standard response has been to present saliency maps [12], [25], [26], but saliency is descriptive rather than causal: a heatmap indicates where gradients are large, not whether the model’s decision depends on the tissue in that location. The explainability literature has repeatedly made this distinction between an explanation being produced and causability, the degree to which an explanation supports genuine causal understanding by a clinician [8], and has documented the gap between the volume of XAI methods proposed and the volume validated [14], [16], [17]. AD-specific XAI reviews reach the same conclusion: attribution maps are routinely produced and rarely tested [9].

This work is organized around that gap. We build a deliberately conventional model, a six-stream VGG16-BN fusion network over tri-planar MRI and PET slices, and then subject it to an evidence ladder designed to answer four questions that a saliency map cannot:

1) Does the model exploit non-brain shortcuts? Answered by silhouette, exterior, and blank input controls, and by a label-permutation null.

2) Does each modality contribute independently? Answered by modality ablation on the fusion model and by preprocessing ablation.

3) Are AD-relevant regions causally necessary and sufficient? Answered by forward ROI ablation (mask the region) and reverse ROI ablation (keep only the region), both benchmarked against area-matched control regions and a spatial permutation null.

4) Which attribution method actually localizes AD pathology? Answered by quantitative enrichment and pointing-game analysis across occlusion, Layer-CAM, Grad-CAM, and Grad-CAM++.

The classification result is therefore presented as a prerequisite rather than as the contribution. The contribution is the demonstration that a model of this class can be shown, with controls, to depend on hippocampal structure and posterior default-mode metabolism in the modality-specific pattern that AD biology predicts.

## II. RELATED WORK

## A. Multimodal MRI-PET Deep Learning

Architectures for MRI-PET fusion divide broadly into early fusion at the image level [32], intermediate feature-level fusion, and late fusion at the decision level. Narazani et al. [30] systematically compared these strategies with 3D ResNets and reported that late fusion outperformed early and middle fusion, while raising the pointed question of whether PET alone is sufficient. Transformer-based approaches have followed: MMTFN introduces multi-scale transformer fusion across modalities [33], DiaMond uses multi-modal vision transformers with explicit inter-modal attention [7], and MINiT applies multiple-instance transformers to neuroimaging volumes [31]. Vision transformers in general [13] have been adapted to AD with mixed returns relative to their data requirements. Selfsupervised and longitudinal formulations have also gained traction, including cross-modal MRI-PET pretraining with explicit site-invariance objectives [1], contrastive pretraining on 3D amyloid PET for progression prediction [15], selforganized multi-modal longitudinal maps [20], and multimodal fusion with longitudinal analysis [6]. CNN-based work continues in parallel, including tri-branch EfficientNet integration [22], ensemble architectures [23], and attention-driven CNNs with multi-activation fusion [29]. Narrative reviews of the PET-MRI AI literature since Tauvid approval [21] and systematic reviews of datasets and modalities [24] both note that architectural novelty has outpaced validation rigor.

## B. Explainability in AD Neuroimaging

The AD-specific XAI literature spans post-hoc attribution, intrinsically interpretable models, and feature attribution on tabular biomarkers. El-Sappagh et al. [10] built a multilayer multimodal model with an explainability layer over clinical and imaging features; Coluzzi et al. [11] used XAI over multiple MRI-derived brain measures for biomarker investigation; Bhattarai et al. [19] applied Deep-SHAP to map regional neuroimaging biomarkers onto cognition; Hernandez et al. [18] used XAI to interrogate the behavior of the top TADPOLE challenge methods. Anzum et al. [2] combined transformers with explainability, and Adeniran et al. [28] proposed an explainable MRI-based ensemble architecture. Reviews [9], [14] catalogue the methods; [8], [16], [17] supply the conceptual framing, in particular the distinction between an explanation being produced and an explanation being verified.

## C. Attribution Methods

Occlusion sensitivity [27] perturbs input regions and measures the change in output, making it causal by construction but coarse and computationally expensive. Grad-CAM [12] and Grad-CAM++ [26] weight final-layer feature maps by gradients, producing smooth but low-resolution maps whose faithfulness depends strongly on the backbone and the layer chosen. Layer-CAM [25] addresses the resolution limitation by combining hierarchical class activation maps from earlier layers. We evaluate all four under identical conditions rather than selecting the method that produces the most persuasive picture.

## D. Positioning

Relative to this literature, the present work does not claim architectural novelty. It claims that the validation and explainability protocol reported here, shortcut controls, permutation nulls, forward and reverse regional ablation with area-matched controls, and quantitative attribution benchmarking, is more informative than an incremental AUC improvement, and that it exposes failure modes (notably the unreliability of Grad-CAM family methods in this setting) that are invisible to qualitative inspection.

## III. MATERIALS AND METHODS

## A. Dataset

Data were obtained from the Alzheimer’s Disease Neuroimaging Initiative (ADNI) [34]. We used preprocessed T1- weighted structural MRI and FDG PET accepting all ADNI series to maximize the paired cohort.

The full download comprised 18,114 MRI series and 6,772 PET series. Restricting to subjects with both modalities available yielded 554 unique subjects: 324 cognitively normal (CN) and 230 AD. Splits were made at the subject level before any slice extraction, producing 356 training, 88 validation, and 110 test subjects (Table I). The test partition was physically separated into a distinct directory before training began and was not read by any script during model development, slice selection, or hyperparameter tuning.

Subject-level splitting is not a refinement but a precondition. Splitting at the slice level permits slices from the same brain to appear on both sides of the partition, and in a 2.5D setting this reliably inflates reported metrics to the point where they measure memorization of individual anatomy rather than group discrimination.

TABLE I  
ADNI COHORT COMPOSITION
<table><tr><td>Group</td><td>MRI – PET series</td><td>Paired subjects</td><td>Train – Val – Test</td></tr><tr><td>CN</td><td> $1 2 , 1 9 8 - 4 , 3 4 8$ </td><td>324</td><td>208-52 - 64</td></tr><tr><td>AD</td><td>5,916 – 2,424</td><td>230</td><td>148- 36 - 46</td></tr><tr><td>All</td><td> $1 8 , 1 1 4 - 6 , 7 7 2$ </td><td>554</td><td>356 - 88 - 110</td></tr></table>

![](images/325501cb40250c0dcab6d0d1aa71a65fa0c0f8c24768d336a1546727002ae21b.jpg)  
Fig. 1. Preprocessing pipeline. MRI and PET are converted from DICOM with dcm2niix. MRI receives N4 bias correction and brain extraction; PET is kept native, with no bias correction and no skull stripping. PET is rigidly registered to the subject’s T1 (6 DOF, mutual information), the T1 is nonlinearly registered to the MNI152 1.5 mm template with ANTs SyN, and the two transforms are composed so each volume is resampled exactly once. MRI is z-scored within the brain mask; PET is converted to SUVR against an eroded pons reference. Fixed MNI coordinates then define the extracted 224 × 224 slices.

## B. Preprocessing

The preprocessing pipeline is shown in Fig. 1, with representative CN and AD examples before and after processing in Fig. 2.

DICOM series were converted to NIfTI with dcm2niix. MRI volumes underwent N4 bias field correction followed by brain extraction with HD-BET [35]. PET volumes were deliberately left native: no bias correction and no skull stripping, since PET has no comparable bias field and since applying an MRI-derived mask to PET risks removing genuine tracer signal at the cortical rim.

PET was rigidly registered to the subject’s own T1 (6 degrees of freedom, mutual information), and the T1 was nonlinearly registered to the MNI152 1.5 mm template with ANTs SyN. The two transforms were then composed and applied in a single resampling step, so that each voxel is interpolated exactly once; sequential resampling would introduce avoidable blur that disproportionately affects the thin structures of interest.

TABLE II  
MRI SLICE SEARCH, VALIDATION AUC
<table><tr><td colspan="3">Axial</td><td colspan="3">Coronal</td><td colspan="3">Sagittal</td></tr><tr><td>C</td><td>∆</td><td>AUC</td><td>C</td><td>∆</td><td>AUC</td><td>C</td><td>∆</td><td>AUC</td></tr><tr><td>+20</td><td>0</td><td>0.704</td><td>-10</td><td>0</td><td>0.884</td><td>-10</td><td>0</td><td>0.758</td></tr><tr><td>+20</td><td>4</td><td>0.782</td><td>-10</td><td>4</td><td>0.871</td><td>-10</td><td>4</td><td>0.781</td></tr><tr><td>+20</td><td>8</td><td>0.753</td><td>-10</td><td>8</td><td>0.814</td><td>-10</td><td>8</td><td>0.826</td></tr><tr><td>+20</td><td>14</td><td>0.742</td><td>-10</td><td>14</td><td>0.848</td><td>-10</td><td>14</td><td>0.798</td></tr><tr><td>0</td><td>0</td><td>0.741</td><td>-25</td><td>0</td><td>0.809</td><td>-25</td><td>0</td><td>0.848</td></tr><tr><td>0</td><td>4</td><td>0.786</td><td>-25</td><td>4</td><td>0.816</td><td>-25</td><td>4</td><td>0.860</td></tr><tr><td>0</td><td>8</td><td>0.793</td><td>-25</td><td>8</td><td>0.808</td><td>-25</td><td>8</td><td>0.814</td></tr><tr><td>0</td><td>14</td><td>0.751</td><td>-25</td><td>14</td><td>0.778</td><td>-25</td><td>14</td><td>0.764</td></tr><tr><td>-20</td><td>0</td><td>0.842</td><td>-40</td><td>0</td><td>0.768</td><td>-40</td><td>0</td><td>0.769</td></tr><tr><td>-20</td><td>4</td><td>0.875</td><td>-40</td><td>4</td><td>0.772</td><td>-40</td><td>4</td><td>0.783</td></tr><tr><td>-20</td><td>8</td><td>0.812</td><td>-40</td><td>8</td><td>0.782</td><td>-40</td><td>8</td><td>0.802</td></tr><tr><td>-20</td><td>14</td><td>0.816</td><td>-40</td><td>14</td><td>0.765</td><td>-40</td><td>14</td><td>0.784</td></tr></table>

Intensity normalization was modality-appropriate. MRI was z-scored within the brain mask. PET was converted to SUVR using an eroded pons reference region drawn from the Harvard-Oxford subcortical atlas; erosion reduces partialvolume contamination at the reference boundary. Per-image min-max rescaling was explicitly avoided: for PET it destroys the inter-subject comparability that SUVR normalization exists to preserve, erasing precisely the global hypometabolism signal that distinguishes AD from CN.

Finally, 2D slices were extracted at fixed MNI coordinates and resampled to $2 2 4 \times 2 2 4$ , stored as per-subject float32 arrays. Values were clipped to (−3.0, 3.0) for both modalities and standardized with $\mu ~ = ~ 0 . 4 4 9$ $\sigma ~ = ~ 0 . 2 2 6$ to match ImageNet backbone statistics.

1) Does Nonlinear Registration Remove the Atrophy Signal?: This is the standard objection to SyN normalization in AD work. We tested it directly: CSF fraction computed in native space and in MNI space correlated at r = 0.976 across all 554 subjects, between-subject spread was retained at 0.902, and the mean absolute log-Jacobian of the warp was near-identical between groups (AD 0.198, CN 0.192). Morphological signal survives the warp; the transform is not selectively compressing AD brains toward the template.

## C. Slice Selection

Because each stream consumes three adjacent slices stacked as a (3, 224, 224) tensor, two parameters define a stream: the center coordinate c in MNI millimeters and the interslice spacing ∆. These were selected by grid search on the validation set only, with the test set untouched (Tables II and III).

The selected configurations are anatomically coherent rather than arbitrary. The MRI streams converge on the medial temporal lobe: axial c = −20 mm passes through the hippocampal body and amygdala, coronal c = −10 mm through the hippocampal head, and sagittal c = −25 mm along the long axis of the hippocampus. Narrow spacing (∆ = 4 mm)

![](images/359234906b1817034f4d6c76ac145ae903953bbdb59bdaf2fb9d695281ff615f.jpg)  
Fig. 2. Representative CN (a) and AD (b) subjects before and after preprocessing, shown in all three planes for both modalities. The AD subject shows the expected ventricular enlargement and cortical thinning on MRI, and posterior hypometabolism on PET.

TABLE III  
PET SLICE SEARCH, VALIDATION AUC
<table><tr><td colspan="3">Axial</td><td colspan="3">Coronal</td><td colspan="3">Sagittal</td></tr><tr><td>C</td><td>∆</td><td>AUC</td><td>C</td><td>∆</td><td>AUC</td><td>C</td><td>∆</td><td>AUC</td></tr><tr><td>+20</td><td>0</td><td>0.875</td><td>-40</td><td>0</td><td>0.923</td><td>-6</td><td>0</td><td>0.925</td></tr><tr><td>+20</td><td>6</td><td>0.908</td><td>-40</td><td>6</td><td>0.926</td><td>-6</td><td>6</td><td>0.928</td></tr><tr><td>+20</td><td>10</td><td>0.901</td><td>-40</td><td>10</td><td>0.947</td><td>-6</td><td>10</td><td>0.928</td></tr><tr><td>+20</td><td>16</td><td>0.905</td><td>-40</td><td>16</td><td>0.943</td><td>-6</td><td>16</td><td>0.931</td></tr><tr><td>+32</td><td>0</td><td>0.916</td><td>-55</td><td>0</td><td>0.893</td><td>-25</td><td>0</td><td>0.901</td></tr><tr><td>+32</td><td>6</td><td>0.904</td><td>-55</td><td>6</td><td>0.911</td><td>-25</td><td>6</td><td>0.923</td></tr><tr><td>+32</td><td>10</td><td>0.909</td><td>-55</td><td>10</td><td>0.908</td><td>-25</td><td>10</td><td>0.906</td></tr><tr><td>+32</td><td>16</td><td>0.911</td><td>-55</td><td>16</td><td>0.928</td><td>-25</td><td>16</td><td>0.940</td></tr><tr><td>+42</td><td>0</td><td>0.869</td><td>-70</td><td>0</td><td>0.893</td><td>-45</td><td>0</td><td>0.886</td></tr><tr><td>+42</td><td>6</td><td>0.902</td><td>-70</td><td>6</td><td>0.896</td><td>-45</td><td>6</td><td>0.881</td></tr><tr><td>+42</td><td>10</td><td>0.917</td><td>-70</td><td>10</td><td>0.911</td><td>-45</td><td>10</td><td>0.910</td></tr><tr><td>+42</td><td>16</td><td>0.925</td><td>-70</td><td>16</td><td>0.904</td><td>-45</td><td>16</td><td>0.914</td></tr></table>

preserves the fine structural detail on which atrophy assessment depends. The PET streams converge on the posterior default-mode network: coronal c = −40 mm and sagittal c = −25 mm intersect the posterior cingulate and precuneus, and axial c = +42 mm covers the parietal association cortex. Wide spacing (∆ = 16 mm) is appropriate for PET because metabolic deficits are spatially diffuse and the effective resolution is far coarser than that of T1.

Notably, the MRI grid shows strong sensitivity to slice position (validation AUC ranges from 0.704 to 0.884), whereas the PET grid is comparatively flat (0.869 to 0.947). This is itself informative: metabolic AD signal is distributed widely enough that most posterior slices carry it, while structural signal is concentrated in a narrow band around the hippocampus.

## D. Architecture

The fusion network shown in Fig. 3 consists of six parallel streams, MRI axial, MRI coronal, MRI sagittal, PET axial, PET coronal, PET sagittal, each processing its own (3, 224, 224) input.

![](images/637e20d81aaccf05149c2f6c5bbe1dafcff0e80464631de4a6cea6928eacf5e5.jpg)  
Fig. 3. Six-stream fusion architecture. Each of the six inputs (MRI and PET × three planes) passes through an independent VGG16-BN convolutional trunk initialised from ImageNet, with the first 34 layers frozen. Global average pooling produces a 512-dimensional descriptor per stream. The six descriptors are concatenated into a 3072-dimensional vector, followed by dropout (p = 0.5) and a single linear layer to two logits.

Each stream uses a VGG16-BN convolutional trunk with the first 34 layers frozen and later layers fine-tuned. Freezing early layers preserves generic edge and texture filters while limiting the number of trainable parameters relative to a cohort of 554 subjects; it is a direct response to the sample size rather than an architectural preference. Each trunk is followed by global average pooling to a 512-dimensional descriptor, replacing VGG’s fully connected head and eliminating the large majority of its parameters.

The six 512-dimensional descriptors are concatenated into a 3072-dimensional joint representation, passed through dropout $( p = 0 . 5 )$ , and mapped by a single linear layer to two logits with softmax over {CN, AD}.

1) Stream Initialization: Models are built in three stages, and the initialization of each stage is what distinguishes the conditions reported in Tables IV- V-VI. Single-plane models are initialized from ImageNet weights and trained on one plane of one modality. Tri-planar models are then trained in two variants: in the first condition all three trunks are again initialized from ImageNet, whereas in the single-plane init condition each trunk is initialized from the corresponding trained single-plane model of the same modality and orientation, and the threestream network is subsequently trained end-to-end. Finally, the six-stream multimodal model inherits its trunks from the two trained tri-planar models and is trained end-to-end in the same way. The layer-freezing policy is identical across all stages.

Transferring single-plane weights rather than ImageNet weights into the tri-planar model gives each stream a planespecific starting point already adapted to MNI-space neuroimaging intensities, and is the single largest contributor to the MRI and PET result.

## E. Training

Models were implemented in PyTorch and trained with cross-entropy loss. Checkpoints were selected on validation

AUC, never on test performance. All reported figures derive from complete 5-fold cross-validation.

Optimization used AdamW with a learning rate of $1 \times 1 0 ^ { - 5 }$ a batch size of 16, and 60 epochs. The sole augmentation was a random affine transform (rotation up to 5<sup>◦</sup>, translation up to 3% of image extent in each axis), applied to the training split only.

## F. Explainability Protocol

Four analyses were applied to the trained fusion model on the held-out test set.

1) Shortcut Controls: Four input conditions were compared: intact; silhouette, in which brain tissue is replaced by a uniform fill so that only the outer contour remains; exterior, in which the brain is removed and only non-brain content is retained; and blank. If a model is reading head shape, skull thickness, or field-of-view artifacts, silhouette and exterior conditions will retain discriminative power.

2) Label-Permutation Null: The full training pipeline was rerun with class labels shuffled within the training set, preserving all other properties of the data. This estimates the discrimination achievable by the architecture and training procedure in the absence of real class structure, and thereby bounds the contribution of optimization artifacts.

3) Forward and Reverse ROI Ablation: Regions of interest were taken from the Harvard-Oxford cortical and subcortical atlases and projected into the 224 × 224 MNI slice space of each stream. Forward ablation masks the ROI and measures the resulting change in AUC and in the AD logit. Reverse ablation inverts the operation: everything outside the ROI is masked and the model is evaluated on the ROI alone, measuring sufficiency rather than necessity. Both are compared against area-matched control composites and against a spatial permutation null in which masks of identical area are placed at random brain locations; the resulting Z statistic expresses effect size relative to that null.

Reverse ablation was necessary because the six-stream architecture is highly redundant. Forward ablation of a single small region in a single stream produces near-zero ∆AUC simply because five other streams remain intact, a property of the model, not evidence of regional irrelevance. Necessity tests alone systematically understate regional importance in redundant architectures.

4) Attribution Benchmarking: Occlusion sensitivity [27], Layer-CAM [25], Grad-CAM [12], and Grad-CAM++ [26] were computed per stream. Each was evaluated quantitatively by (i) enrichment, the ratio of mean attribution inside an a priori ROI to the mean attribution expected under chance, computed separately for AD regions and area-matched control regions; and (ii) a pointing game, the fraction of subjects for which the single peak attribution voxel falls inside the a priori AD ROI set, compared against the chance rate implied by that ROI set’s area fraction. Both statistics were computed separately for AD and CN subjects, since a method that highlights the same regions regardless of class is not explaining the decision.

TABLE IV  
MRI PERFORMANCE, TEST SET
<table><tr><td>Mode</td><td>AUC</td><td>Accuracy</td><td>F1</td></tr><tr><td>Axial</td><td>0.877</td><td>0.736</td><td>0.695</td></tr><tr><td>Coronal</td><td>0.891</td><td>0.764</td><td>0.735</td></tr><tr><td>Sagittal</td><td>0.897</td><td>0.845</td><td>0.813</td></tr><tr><td>Fuse 3-plane</td><td>0.909</td><td>0.818</td><td>0.778</td></tr><tr><td>Fuse 3-plane single-plane init</td><td>0.944</td><td>0.873</td><td>0.844</td></tr></table>

TABLE V  
PET PERFORMANCE, TEST SET
<table><tr><td>Mode</td><td>AUC</td><td>Accuracy</td><td>F1</td></tr><tr><td>Axial</td><td>0.913</td><td>0.836</td><td>0.809</td></tr><tr><td>Coronal</td><td>0.900</td><td>0.818</td><td>0.800</td></tr><tr><td>Sagittal</td><td>0.918</td><td>0.827</td><td>0.796</td></tr><tr><td>Fuse 3-plane</td><td>0.928</td><td>0.827</td><td>0.796</td></tr><tr><td>Fuse 3-plane single-plane init</td><td>0.936</td><td>0.855</td><td>0.830</td></tr></table>

TABLE VI  
MULTIMODAL FUSION, TEST SET
<table><tr><td>Mode</td><td>AUC</td><td>Accuracy</td><td>F1</td></tr><tr><td>MRI 3-plane</td><td>0.944</td><td>0.873</td><td>0.844</td></tr><tr><td>PET 3-plane</td><td>0.936</td><td>0.855</td><td>0.830</td></tr><tr><td>Fusion  $\mathbf { M R I } + \mathbf { P E T }$ </td><td>0.962</td><td>0.909</td><td>0.891</td></tr></table>

## IV. RESULTS

## A. Single-Plane, Tri-Planar, and Multimodal Performance

Three observations follow from Tables IV-V-VI. First, PET outperforms MRI at the single-plane level in every plane (0.900–0.918 versus 0.877–0.897), consistent with the finding that metabolic signal is the stronger single discriminator in AD versus CN classification [30].

Second, the planes are not equally informative in the two modalities. For MRI, the sagittal plane is clearly the strongest (accuracy 0.845), ahead of coronal (0.764) and axial (0.736). For PET, by contrast, the three planes perform comparably (accuracy 0.836, 0.818, and 0.827 for axial, coronal, and sagittal respectively), with no plane carrying a decisive advantage. This mirrors the pattern already seen in the slice search (Tables II and III): structural AD signal is concentrated along the long axis of the hippocampus and is therefore strongly plane-dependent, whereas metabolic AD signal is spatially diffuse and is captured similarly well from any of the three orientations.

Third, initialization matters for the tri-planar models. In the “no pretrain” condition the three streams are initialized from ImageNet weights; in the “single-plane init” condition each stream is instead initialized from the weights of the corresponding single-plane model, and the fusion network is then trained end-to-end. This second scheme improves both modalities, substantially so for MRI (0.909 → 0.944 AUC, 0.818 → 0.873 accuracy) and more modestly for PET (0.928 → 0.936). In both cases the pretrained tri-planar model also exceeds the best single plane, confirming that the three views contribute complementary rather than redundant information. Multimodal fusion adds a further gain on top of that: combining the MRI and PET tri-planar models yields AUC 0.962, accuracy 0.909, and F1 0.891, improving on the better unimodal model by 0.018 AUC, 0.036 accuracy, and 0.047 F1. That the gain is larger in the threshold-dependent metrics than in AUC indicates that fusion principally corrects subjects near the decision boundary rather than reordering the ranking wholesale.

TABLE VII  
COMPARISON WITH PUBLISHED MODELS
<table><tr><td>Method</td><td>Modality</td><td>AUC</td><td>BACC</td></tr><tr><td>3D-ResNet [30]</td><td>M</td><td>0.936</td><td>0.865</td></tr><tr><td>3D-ResNet [30]</td><td>P</td><td>0.951</td><td>0.890</td></tr><tr><td>3D-ViT [31]</td><td>M</td><td>0.936</td><td>0.862</td></tr><tr><td>3D-ViT [31]</td><td>P</td><td>0.935</td><td>0.887</td></tr><tr><td>ResNet early fusion [32]</td><td> $\mathbf { M } + \mathbf { P }$ </td><td>0.855</td><td>0.826</td></tr><tr><td>ResNet middle fusion [30]</td><td> $\mathbf { M } + \mathbf { P }$ </td><td>0.881</td><td>0.826</td></tr><tr><td>ResNet late fusion [30]</td><td> $\mathbf { M } + \mathbf { P }$ </td><td>0.967</td><td>0.897</td></tr><tr><td>MMTFN [33]</td><td> $\mathbf { M } + \mathbf { P }$ </td><td>0.936</td><td>0.887</td></tr><tr><td>DiaMond [7]</td><td> $\mathbf { M } + \mathbf { P }$ </td><td>0.971</td><td>0.924</td></tr><tr><td>Ours</td><td> $\mathbf { M } + \mathbf { P }$ </td><td>0.962</td><td>0.909</td></tr></table>

TABLE VIII  
CONFUSION MATRIX, FUSION MODEL, TEST SET
<table><tr><td colspan="2">Actual AD</td><td>Actual CN</td></tr><tr><td>Predicted AD</td><td> $\mathrm { T P } = 4 1$ </td><td> $\mathrm { F P } = 5$ </td></tr><tr><td>Predicted CN</td><td> $\mathrm { F N } = 5$ </td><td> $\mathrm { T N } = 5 9$ </td></tr></table>

## B. Comparison with Published Methods

Our model sits within the leading cluster (Table VII). DiaMond [7] and ResNet late fusion [30] report marginally higher AUC; our balanced accuracy exceeds all listed methods except DiaMond. Given a test set of 110 subjects, these differences fall within overlapping confidence intervals and we do not claim superiority. The relevant point is that a 2.5D architecture with a frozen-trunk VGG backbone reaches parity with 3D CNNs and multimodal transformers at a fraction of the computational cost — which makes the exhaustive perturbation analyses in Sections IV-F–IV-I tractable. Occlusion sensitivity over six streams and multiple ROI configurations would be prohibitively expensive on a 3D transformer.

## C. Error Structure

Sensitivity is 89.1% (41/46), specificity 92.2% (59/64), and the error distribution is symmetric (Table VIII). Symmetry matters because a model that has latched onto a classcorrelated confound typically produces asymmetric errors. It also indicates that the decision threshold is well placed without post-hoc adjustment, despite the 324:230 class imbalance in the cohort.

TABLE IX  
MODALITY ABLATION ON THE FUSION MODEL
<table><tr><td>Mode</td><td>AUC</td></tr><tr><td>Without MRI</td><td>0.932</td></tr><tr><td>Without PET</td><td>0.938</td></tr><tr><td>All</td><td>0.962</td></tr></table>

TABLE X  
PREPROCESSING ABLATION
<table><tr><td>Mode</td><td>AUC</td></tr><tr><td>MRI - with preprocessing</td><td>0.944</td></tr><tr><td>MRI - without preprocessing</td><td>0.860</td></tr><tr><td>PET - with preprocessing</td><td>0.936</td></tr><tr><td>PET - without preprocessing</td><td>0.930</td></tr><tr><td>Fusion - with preprocessing</td><td>0.962</td></tr><tr><td>Fusion - without preprocessing</td><td>0.934</td></tr></table>

TABLE XI  
BRAIN-REMOVAL SHORTCUT CONTROLS
<table><tr><td>Condition</td><td>AUC</td></tr><tr><td>Intact</td><td>0.962</td></tr><tr><td>Silhouette</td><td>0.622</td></tr><tr><td>Exterior</td><td>0.608</td></tr><tr><td>Blank</td><td>0.500</td></tr></table>

## D. Modality Ablation

Suppressing either modality at inference degrades the fusion model, and the degradation is comparable in magnitude for both (0.030 and 0.024; Table IX). Neither modality is decorative. That the fusion model with MRI suppressed (0.932) performs slightly below the dedicated PET-only model (0.936), and similarly for the mirror case, indicates the fusion head has learned cross-modal interaction terms that are not recoverable when one modality is zeroed.

## E. Preprocessing Ablation

The asymmetry in Table X is the most interpretable result in the ablation set. Removing preprocessing costs MRI 0.084 AUC but costs PET only 0.006. MRI depends critically on bias correction, skull stripping, and spatial normalization, because its discriminative signal is morphological: a structure must be in a consistent location and free of intensity inhomogeneity for its size and shape to be comparable across subjects. PET’s discriminative signal is intensity-based, regional hypometabolism relative to a reference, and survives substantial spatial misalignment because the deficits are large and diffuse.

This also implies that an MRI pipeline without proper normalization will appear to work (0.860 is not a failure) while delivering a substantially degraded and less anatomically grounded model. The fusion model loses 0.028 without preprocessing, tracking its MRI component.

TABLE XII  
LABEL PERMUTATION NULL
<table><tr><td>Condition</td><td>AUC</td></tr><tr><td>Intact</td><td>0.962</td></tr><tr><td>Label-permutation null</td><td>0.456</td></tr></table>

TABLE XIII

FORWARD ABLATION, MRI STREAMS
<table><tr><td>ROI masked out</td><td>Area (px)</td><td>∆AUC</td><td>∆logitAD</td><td>Z vs. null</td></tr><tr><td>Medial temporal</td><td>4,361</td><td>0.038</td><td>0.283</td><td>+20.3</td></tr><tr><td>Hippocampus</td><td>1,844</td><td>0.015</td><td>0.087</td><td>+5.4</td></tr><tr><td>Amygdala</td><td>1,011</td><td>0.012</td><td>0.057</td><td>+3.0</td></tr><tr><td>Parahippocampal</td><td>1,551</td><td>0.006</td><td>0.078</td><td>+4.7</td></tr><tr><td>Control composite</td><td>4,297</td><td>0.005</td><td>0.000</td><td>-1.3</td></tr><tr><td>Precentral gyrus</td><td>3,218</td><td>0.002</td><td>0.002</td><td>-1.2</td></tr></table>

## F. Shortcut Controls and the Permutation Null

The blank condition returns exactly 0.500 (Table XI), confirming the evaluation harness has no leakage path independent of image content. Silhouette (0.622) and exterior (0.608) retain some discriminative power but lose approximately 76% and 77% respectively of the model’s above-chance performance. This residual is expected: global brain volume and ventricular expansion genuinely differ between AD and CN, and a silhouette carries coarse volumetric information. It is nonetheless small enough to exclude the hypothesis that the model is primarily reading head shape or acquisition geometry.

The PET exterior residual is the one result we flag as incompletely explained. Because PET is not skull-stripped, the exterior condition retains scalp and skull-adjacent tracer uptake as well as the outer intensity envelope, and some of the 0.608 is likely attributable to global count-rate and reconstruction differences that correlate with scanner generation and therefore, indirectly, with cohort composition. We report it rather than suppress it.

The label-permutation null (Table XII) is the strongest of these controls. Trained under identical conditions on shuffled labels, the model reaches AUC 0.456, statistically indistinguishable from chance and, if anything, slightly below it, as expected when a network overfits noise in the training set. The architecture and training procedure generate no discrimination in the absence of real class structure.

## G. Forward ROI Ablation: Necessity

The AUC changes in Tables XIII and XIV are small in absolute terms, exactly as redundancy predicts. The Z statistic against the spatial null is the informative column, because it accounts for ROI area, the dominant nuisance variable, since masking more pixels perturbs the model more regardless of which pixels they are.

On that measure the separation is unambiguous. In MRI, the medial temporal composite reaches $Z = + 2 0 . 3$ and every individual medial temporal structure is significant (hippocampus +5.4, amygdala +3.0, parahippocampal +4.7), while the area-matched control composite (4,297 px, nearly identical to the medial temporal composite’s 4,361 px) sits at $Z = - 1 . 3$ and the precentral gyrus at −1.2. Both controls are below the spatial null, meaning masking them perturbs the model less than masking a random brain region of the same size.

TABLE XIV  
FORWARD ABLATION, PET STREAMS
<table><tr><td>ROI masked out</td><td>Area (px)</td><td>∆AUC</td><td> $\Delta \mathrm { l o g i t } _ { \mathrm { A D } }$ </td><td>Z vs. null</td></tr><tr><td>Posterior DMN</td><td>6,672</td><td>0.060</td><td>0.446</td><td>+25.9</td></tr><tr><td>Precuneus</td><td>1,878</td><td>0.016</td><td>0.199</td><td>+11.1</td></tr><tr><td>Posterior cingulate</td><td>1,728</td><td>0.011</td><td>0.274</td><td> $+ 1 5 . 6$ </td></tr><tr><td>Inferior parietal</td><td>3,066</td><td>0.005</td><td>0.126</td><td> $+ 6 . 8$ </td></tr><tr><td>Control composite</td><td>3,088</td><td>0.005</td><td>0.036</td><td> $+ 1 . 4$ </td></tr><tr><td>Precentral gyrus</td><td>1,801</td><td>-0.001</td><td>0.023</td><td> $+ 0 . 7$ </td></tr></table>

In PET, the posterior DMN composite reaches $Z = + 2 5 . 9 ,$ with precuneus +11.1, posterior cingulate +15.6, and inferior parietal +6.8, against a control composite at +1.4 and precentral gyrus at +0.7.

The $\Delta \mathrm { l o g i t } _ { \mathrm { A D } }$ column adds a second layer of evidence. Masking a genuinely AD-informative region should move the model’s AD evidence specifically, not merely add noise. For the MRI control composite, $\Delta \mathrm { l o g i t } _ { \mathrm { A D } }$ is 0.000 despite a ∆AUC of 0.005, the region perturbs the output without moving the AD decision variable at all. Compare medial temporal (0.283) and posterior DMN (0.446). For PET, posterior cingulate is instructive: its ∆AUC (0.011) is below that of precuneus (0.016), but its ∆logit<sub>AD</sub> (0.274) is substantially higher, indicating a small region with concentrated influence on the AD decision variable. The two metrics measure different things, and reporting only the former would have obscured this.

1) The Double Dissociation: Medial temporal effects are large in MRI and the corresponding PET effects are not the drivers; posterior DMN effects are large in PET and are not carried by MRI. This is precisely the pattern AD neuroimaging predicts: structural MRI indexes medial temporal neurodegeneration [3], FDG-PET indexes posterior cingulate and precuneus hypometabolism [4], and the two are complementary rather than redundant [5]. The model has arrived at the modality-specific division of labor that the biology specifies, without being told to.

## H. Reverse ROI Ablation: Sufficiency

“Retained” expresses the fraction of the intact model’s above-chance discrimination, $( \mathrm { A U C } - 0 . 5 ) / ( 0 . 9 6 2 - 0 . 5 )$ that survives when only the named region is visible. ρ is the correlation between the reverse-ablated model’s predicted probabilities and those of the intact model, a measure of whether the region reproduces the model’s decisions, not merely its accuracy.

These results (Tables XV and XVI) are considerably stronger than the forward ablation, in the expected direction: sufficiency tests are immune to the redundancy that suppresses necessity tests.

TABLE XV  
REVERSE ABLATION, MRI STREAMS (ROI RETAINED, ALL ELSE MASKED)
<table><tr><td>ROI retained</td><td>Area (px)</td><td>AUC</td><td>Retained</td><td>ρ</td></tr><tr><td>Medial temporal</td><td>4,361</td><td>0.912</td><td>89.2%</td><td> $+ 0 . 8 0$ </td></tr><tr><td>Amygdala</td><td>1,011</td><td>0.877</td><td>81.8%</td><td> $+ 0 . 7 3$ </td></tr><tr><td>Hippocampus</td><td>1,844</td><td>0.852</td><td>76.2%</td><td> $+ 0 . 7 3$ </td></tr><tr><td>Parahippocampal</td><td>1,551</td><td>0.789</td><td>62.7%</td><td> $+ 0 . 5 7$ </td></tr><tr><td>Control composite</td><td>4,297</td><td>0.476</td><td>-5.2%</td><td> $+ 0 . 0 1$ </td></tr><tr><td>Control composite XL</td><td>8,637</td><td>0.646</td><td>31.6%</td><td> $+ 0 . 2 6$ </td></tr></table>

TABLE XVI

REVERSE ABLATION, PET STREAMS (ROI RETAINED, ALL ELSE MASKED)
<table><tr><td>ROI retained</td><td>Area (px)</td><td>AUC</td><td>Retained</td><td>ρ</td></tr><tr><td>Posterior DMN</td><td>6,672</td><td>0.865</td><td>79.0%</td><td> $+ 0 . 7 7$ </td></tr><tr><td>Precuneus</td><td>1,878</td><td>0.833</td><td>72.0%</td><td> $+ 0 . 6 6$ </td></tr><tr><td>Posterior cingulate</td><td>1,728</td><td>0.816</td><td>68.4%</td><td> $+ 0 . 6 6$ </td></tr><tr><td>Inferior parietal</td><td>3,066</td><td>0.784</td><td>61.5%</td><td> $+ 0 . 6 7$ </td></tr><tr><td>Control composite</td><td>3,088</td><td>0.593</td><td>20.1%</td><td> $+ 0 . 1 2$ </td></tr><tr><td>Control composite XL</td><td>4,716</td><td>0.561</td><td>13.2%</td><td>+0.09</td></tr></table>

In MRI, the medial temporal composite alone retains 89.2% of discriminative power (AUC 0.912 from 4,361 visible pixels out of 50,176) with $\rho = + 0 . 8 0$ . Every constituent structure retains more than 62%. The area-matched control composite, with 4,297 visible pixels, produces AUC 0.476, below chance, with $\rho = + 0 . 0 1$ , meaning its predictions are uncorrelated with the intact model’s. Doubling the control area to 8,637 pixels raises it only to 0.646 (31.6%, $\rho = + 0 . 2 6 )$ , less than half of what the smaller medial temporal composite achieves.

In PET the pattern repeats: posterior DMN retains 79.0% with $\rho ~ = ~ + 0 . 7 7 .$ , while the area-matched control retains 20.1% with $\rho = + 0 . 1 2 ,$ and the enlarged control performs worse (13.2%) despite 53% more visible area. Adding noninformative cortex does not help; the information is regionally specific.

The amygdala result in MRI deserves comment: 1,011 pixels retain 81.8% of discriminative power, a higher ratio than the larger hippocampus ROI (1,844 px, 76.2%). Amygdalar atrophy in AD is well established and often underweighted relative to hippocampal measures; the model appears to find it at least as informative per unit area.

## I. Attribution Method Comparison

Fig. 4 shows attribution maps for a representative AD subject across all six streams and all four methods, with hippocampus, posterior cingulate, and precuneus outlined. The qualitative impression, occlusion and Layer-CAM concentrating inside the outlines, Grad-CAM and Grad-CAM++ concentrating along cortical rims and image edges, is confirmed quantitatively below.

The double dissociation appears again in Table XVII, now in attribution space and independently of the ablation experiments. MRI coronal and sagittal streams show 2.65- 4.17× enrichment in medial temporal structures. PET coronal

![](images/e3fe6c4f61b7bb71d96adb0f96c48be00d7227a741b4bbb64ebe91137bc1c40b.jpg)  
Fig. 4. Attribution maps for one AD subject across all six streams and four methods, overlaid on the MNI template, with hippocampus (cyan), posterior cingulate (green) and precuneus (orange) outlined. Occlusion and Layer-CAM concentrate inside the outlined regions in the corresponding modality; Grad CAM and Grad-CAM++ concentrate on the cortical rim and image border.

TABLE XVII  
OCCLUSION ENRICHMENT BY STREAM (AD SUBJECTS; 1.0 = CHANCE)
<table><tr><td>Stream</td><td>Medial temporal (hipp / amyg / parahipp)</td><td>Posterior DMN (post-cing / precuneus / inf-par)</td><td>Control (precentral / occipital / lingual)</td></tr><tr><td>MRI axial</td><td>1.43 /  2.94 / 1.17</td><td>n/a</td><td>— / 0.10 / 0.35</td></tr><tr><td>MRI coronal</td><td>3.54 / 4.17 / 3.54</td><td>n/a</td><td>0.34 / — / —</td></tr><tr><td>MRI sagittal</td><td>3.22 / 2.99 / 2.65</td><td>n/a</td><td>0.78 / 0.55 /  1.69</td></tr><tr><td>PET axial</td><td>n/a</td><td>2.46 / 3.83 / 0.70</td><td>0.43 / 0.29 / —</td></tr><tr><td>PET coronal</td><td>0.38 / — / 0.61</td><td>4.99 / 3.96 /  1.15</td><td>— / — / 0.18</td></tr><tr><td>PET sagittal</td><td>1.96 / 1.20 / 1.51</td><td>n/a</td><td>0.11 / 1.08 / 3.26</td></tr></table>

TABLE XVIII  
ATTRIBUTION METHOD COMPARISON
<table><tr><td>Method</td><td>Peak enrichment, AD regions</td><td>Enrichment, control regions</td><td>Pointing game (AD, best stream)</td><td>Usable</td></tr><tr><td>Occlusion</td><td>3.5 - 5.0</td><td>0.10-0.43</td><td>54.3% (chance 16.7%)</td><td>Yes - primary evidence</td></tr><tr><td>Layer-CAM</td><td>2.0- 3.0</td><td>0.22 -0.62</td><td>69.6% (chance 6.2%)</td><td>Directionally; weak AD-CN separation</td></tr><tr><td>Grad-CAM</td><td>≈ 1.0 − 2.3</td><td>0.36-2.00</td><td>21.7% (chance 4.8%)</td><td>No</td></tr><tr><td>Grad-CAM++</td><td>≈1.0</td><td>0.59 - 1.32</td><td>8.7% (chance 16.7%)</td><td>No</td></tr></table>

shows 4.99× in posterior cingulate and 3.96× in precuneus while showing below-chance enrichment in medial temporal

TABLE XIX  
POINTING GAME: PEAK ATTRIBUTION INSIDE THE A PRIORI AD ROI SET
<table><tr><td>Method</td><td>Stream</td><td>AD</td><td>CN</td><td>Chance</td></tr><tr><td>Occlusion</td><td>MRI axial</td><td>39.1%</td><td>17.2%</td><td>6.7%</td></tr><tr><td>Occlusion</td><td>MRI coronal</td><td>52.2%</td><td>29.7%</td><td>6.2%</td></tr><tr><td>Occlusion</td><td>MRI sagittal</td><td>54.3%</td><td>46.9%</td><td>4.7%</td></tr><tr><td>Occlusion</td><td>PET axial</td><td>54.3%</td><td>40.6%</td><td>16.7%</td></tr><tr><td>Occlusion</td><td>PET coronal</td><td>54.3%</td><td>23.4%</td><td>13.1%</td></tr><tr><td>Layer-CAM</td><td>MRI coronal</td><td>69.6%</td><td>59.4%</td><td>6.2%</td></tr><tr><td>Layer-CAM</td><td>PET axial</td><td>69.6%</td><td>70.3%</td><td>16.7%</td></tr><tr><td>Grad-CAM</td><td>MRI coronal</td><td>0.0%</td><td>0.0%</td><td>6.2%</td></tr><tr><td>Grad-CAM++</td><td>MRI coronal</td><td>2.2%</td><td>1.6%</td><td>6.2%</td></tr></table>

structures (0.38 and 0.61), the PET streams are actively not attending to the hippocampus. Control regions are almost uniformly below chance, typically 0.10–0.43.

Two control values exceed 1.0: lingual gyrus in MRI sagittal (1.69) and in PET sagittal (3.26). Both occur in sagittal streams at $s = - 2 5 \mathrm { { m m } }$ , where the occipital and lingual regions sit near the slice boundary and the occlusion window necessarily overlaps adjacent structures. We note this as a limitation of slice-plane ROI projection rather than as evidence of lingual involvement.

The ranking in Tables XVIII and XIX is unambiguous and, for the most widely used method in medical imaging XAI, unflattering.

Occlusion is the only method that satisfies both criteria. It achieves high enrichment in AD regions (3.5–5.0×) with strongly suppressed control-region enrichment (0.10–0.43×), and it separates AD from CN in the pointing game across every stream, most sharply in MRI coronal (52.2% vs. 29.7%) and PET coronal (54.3% vs. 23.4%), both against chance rates near 6% and 13%. That AD-CN gap is what makes the map an explanation of the decision rather than a generic anatomical prior.

Layer-CAM achieves the highest raw pointing-game rate (69.6%) but fails the discrimination criterion: in PET axial it scores 69.6% for AD and 70.3% for CN, i.e. it points at the posterior DMN regardless of class. Its control-region enrichment (0.22–0.62) is also higher than occlusion’s. It is useful as a corroborating visualization but cannot stand as evidence on its own.

Grad-CAM fails outright. Its control-region enrichment reaches 2.00, higher than its AD-region enrichment in some streams, and in MRI coronal its pointing-game rate is 0.0% for both classes, i.e. the peak attribution never once landed inside the a priori AD ROI set across 110 test subjects. Grad-CAM++ is worse: 8.7% against a chance rate of 16.7%, meaning it points at AD regions less often than random.

The mechanism is visible in fig4.png. Both Grad-CAM variants concentrate attribution along the cortical rim and the brain-background boundary. With a VGG16-BN trunk at $2 2 4 \times 2 2 4$ , the final convolutional feature map is $7 \times 7 ;$ each cell covers roughly $3 2 \times 3 2$ input pixels, comparable to the entire hippocampal cross-section in these slices. Gradientweighted upsampling from that resolution cannot localize a structure of that size, and the resulting maps are dominated by high-gradient intensity boundaries. Layer-CAM’s use of earlier layers [25] is exactly the fix for this failure mode, and its improved pointing-game performance confirms the diagnosis, though it does not fix the class-discrimination problem.

The practical implication is uncomfortable for a large body of published work: a Grad-CAM figure showing plausiblelooking highlights over medial temporal cortex constitutes no evidence that a model is using medial temporal information. In our model, which the ablation experiments prove does use medial temporal information, Grad-CAM’s peak lands inside those regions 0% of the time. A method that fails to detect a genuine dependency cannot be used to confirm one.

## V. DISCUSSION

## A. What the Evidence Ladder Establishes

Taken together the experiments support a chain of claims, each with its own control. The model is not exploiting nonbrain content: silhouette, exterior, and blank conditions lose 76%, 77%, and 100% of above-chance performance respectively. The model is not exploiting optimization artifacts: the label-permutation null returns AUC 0.456. Both modalities contribute: suppressing either costs 0.024-0.030 AUC, and their contributions are not interchangeable. AD-relevant regions are causally necessary: masking them shifts the AD logit far beyond the spatial null (Z up to +25.9) while area-matched controls do not (Z = −1.3 to +1.4). AD-relevant regions are largely sufficient: the medial temporal lobe alone retains 89.2% of discrimination in MRI and the posterior DMN alone retains 79.0% in PET, versus ≤ 31.6% for area-matched and enlarged controls. And the dependency is modality-specific in the direction AD biology predicts, confirmed independently by ablation and by occlusion attribution.

No single one of these is decisive. Their conjunction is considerably harder to explain by any account other than that the model is reading AD neuropathology.

## B. Redundancy and the Limits of Necessity Testing

A methodological finding with implications beyond this study: in a six-stream architecture, forward ablation of individual regions produces near-zero ∆AUC, not because those regions are unimportant but because the remaining streams compensate. Had we reported only forward ablation, we would have concluded that no region matters, the exact opposite of the truth, which reverse ablation makes clear.

Any redundant architecture, multi-view, multi-modal, or ensemble, will exhibit this. We suggest that necessity-only ablation studies on such architectures be interpreted as lower bounds, and that sufficiency testing with area-matched controls is the more informative design. The $\rho$ column in Tables XV and XVI strengthens this further by asking whether a region reproduces the model’s decisions and not just its accuracy: the control composite’s $\rho = + 0 . 0 1$ in MRI shows that its 0.476 AUC is not merely poor but entirely unrelated to what the intact model does.

## C. The Status of Gradient-Based Attribution

Our results place Grad-CAM and Grad-CAM++ in a category that the XAI methodology literature has warned about [8], [14] but that AD neuroimaging practice has largely ignored: methods that produce confident, plausible, publishable visualizations while failing every quantitative test of faithfulness. The 0.0% pointing-game rate for Grad-CAM in MRI coronal is the clearest single data point in this paper.

This is not a claim that Grad-CAM is broken in general, it was designed for and validated on natural-image classification with large, centrally located objects [12]. It is a claim that its assumptions fail for medical imaging with small subcortical targets and final feature maps at $7 \times 7$ resolution. Layer-CAM’s hierarchical formulation [25] partially addresses this. Perturbation-based methods [27], despite their computational cost, remain the ones whose output has a causal interpretation by construction, and we recommend that attribution claims in this domain be anchored to perturbation evidence with quantitative enrichment statistics and explicit class-discrimination checks, not to heatmap figures.

## D. Clinical Framing

A classifier that separates established AD from CN is not itself clinically useful, that distinction is already made confidently at the bedside. The value of this class of model lies in what it enables downstream: MCI conversion prediction, differential diagnosis among dementias, and quantitative staging. Each requires far more trust than an AUC provides, and trust of that kind requires evidence of the sort assembled here. The causability framing of Holzinger et al. [8] is the relevant standard: the question is not whether the system produces an explanation, but whether a clinician can verify that the explanation is grounded in the same evidence they would use. A demonstration that the model’s hippocampal dependency is carried by MRI and its posterior cingulate dependency by PET is legible in exactly that way.

## VI. LIMITATIONS

2.5D representation. Six fixed slices, however well chosen, discard most of each volume. This constrains the ceiling on performance and means regional analyses are confined to the slice planes used. It is also what makes the perturbation analyses computationally feasible; the trade-off is explicit.

Single cohort. All data come from ADNI [34]. External validation on an independent cohort with different scanners and demographics is required before generalization claims, and site-invariance approaches such as those of Belhaj Ali et al. [1] are the natural direction.

## VII. CONCLUSION

We presented a six-stream 2.5D MRI-PET fusion network for AD versus CN classification that reaches AUC 0.962, accuracy 0.909, and F1 0.891 on a strictly subject-disjoint, physically separated ADNI test set, with performance competitive against 3D CNN and multimodal transformer baselines at substantially lower computational cost.

The principal contribution is the evidence assembled around that result. Shortcut controls and a label-permutation null exclude non-brain and artifactual explanations. Forward ROI ablation demonstrates that AD-relevant regions are causally necessary against a spatial null that area-matched controls do not exceed. Reverse ROI ablation demonstrates that they are largely sufficient, with the medial temporal lobe alone retaining 89.2% of MRI discrimination and the posterior defaultmode network alone retaining 79.0% of PET discrimination. A quantitative comparison of four attribution methods shows that occlusion sensitivity is the only one that both concentrates inside AD regions and separates AD from CN, while Grad-CAM and Grad-CAM++ fail entirely, in one stream, Grad-CAM’s peak attribution never lands inside the a priori AD region set across the whole test cohort.

Across ablation and attribution alike, the same double dissociation emerges: hippocampal and medial temporal evidence carried by MRI, posterior cingulate and precuneus evidence carried by PET. That the model recovers the modality-specific division of labor established by decades of AD neuroimaging, without supervision toward it, is stronger evidence of biological validity than any accuracy figure.

We suggest that the components of this protocol, subjectlevel splitting with a physically separated test set, shortcut controls, permutation nulls, area-matched forward and reverse regional ablation, and quantitative attribution benchmarking with class-discrimination checks, be treated as reporting requirements rather than optional extras.

## ACKNOWLEDGMENT

Data collection and sharing for this project was funded by the Alzheimer’s Disease Neuroimaging Initiative (ADNI) (National Institutes of Health Grant U01 AG024904). ADNI investigators contributed to the design and implementation of ADNI and provided data but did not participate in the analysis or writing of this report.

## REFERENCES

[1] S. Belhaj Ali, N. E. Ghannam, H. Mancy, and B. G. Elkilany, “Multimodal self-supervised learning for early Alzheimer’s: Crossmodal MRI–PET, longitudinal signals, and site invariance,” Diagnostics, vol. 15, art. 3135, 2025.

[2] H. Anzum, N. S. Sammo, and S. Akhter, “Leveraging transformers and explainable AI for Alzheimer’s disease interpretability,” PLOS ONE, 2025, doi: 10.1371/journal.pone.0322607.

[3] G. B. Frisoni, N. C. Fox, C. R. Jack, P. Scheltens, and P. M. Thompson, “The clinical use of structural MRI in Alzheimer disease,” Nature Reviews Neurology, vol. 6, pp. 67–77, 2010.

[4] A. Nordberg, J. O. Rinne, A. Kadir, and B. Langstr ˚ om, “The use of¨ PET in Alzheimer disease,” Nature Reviews Neurology, vol. 6, pp. 78– 87, 2010.

[5] J. Dukart, K. Mueller, A. Horstmann et al., “Combined evaluation of FDG-PET and MRI improves detection and differentiation of dementia,” PLoS ONE, vol. 6, art. e18111, 2011.

[6] S. Muksimova, S. Umirzakova, J. Baltayev, and Y. I. Cho, “Multi-modal fusion and longitudinal analysis for Alzheimer’s disease classification using deep learning,” Diagnostics, vol. 15, art. 717, 2025.

[7] Y. Li, M. Ghahremani, Y. Wally, and C. Wachinger, “DiaMond: Dementia diagnosis with multi-modal vision transformers using MRI and PET,” in Proc. IEEE/CVF Winter Conf. Applications of Computer Vision (WACV), 2025, pp. 107–116.

[8] A. Holzinger, G. Langs, H. Denk, K. Zatloukal, and H. Muller, “Caus-¨ ability and explainability of artificial intelligence in medicine,” WIREs Data Mining and Knowledge Discovery, vol. 9, art. e1312, 2019.

[9] M. Taiyeb Khosroshahi, S. Morsali, S. Gharakhanlou et al., “Explainable artificial intelligence in neuroimaging of Alzheimer’s disease,” Diagnostics, vol. 15, art. 612, 2025.

[10] S. El-Sappagh, J. M. Alonso, S. R. Islam, A. M. Sultan, and K. S. Kwak, “A multilayer multimodal detection and prediction model based on explainable artificial intelligence for Alzheimer’s disease,” Scientific Reports, vol. 11, art. 2660, 2021.

[11] D. Coluzzi, V. Bordin, M. W. Rivolta et al., “Biomarker investigation using multiple brain measures from MRI through explainable artificial intelligence in Alzheimer’s disease classification,” Bioengineering, vol. 12, art. 82, 2025.

[12] R. R. Selvaraju, M. Cogswell, A. Das et al., “Grad-CAM: Visual explanations from deep networks via gradient-based localization,” International Journal of Computer Vision, vol. 128, pp. 336–359, 2020.

[13] A. Dosovitskiy, L. Beyer, A. Kolesnikov et al., “An image is worth 16x16 words: Transformers for image recognition at scale,” arXiv:2010.11929, 2021.

[14] B. H. van der Velden, H. J. Kuijf, K. G. Gilhuijs, and M. A. Viergever, “Explainable artificial intelligence (XAI) in deep learning-based medical image analysis,” Medical Image Analysis, vol. 79, art. 102470, 2022.

[15] M. G. Kwak, Y. Su, K. Chen et al., “Self-supervised contrastive learning to predict the progression of Alzheimer’s disease with 3D amyloid-PET,” Bioengineering, vol. 10, art. 1141, 2023.

[16] A. Adadi and M. Berrada, “Peeking inside the black-box: A survey on explainable artificial intelligence (XAI),” IEEE Access, vol. 6, pp. 52138–52160, 2018.

[17] A. B. Arrieta, N. D´ıaz-Rodr´ıguez, J. Del Ser et al., “Explainable artificial intelligence (XAI): Concepts, taxonomies, opportunities and challenges toward responsible AI,” Information Fusion, vol. 58, pp. 82–115, 2020.

[18] M. Hernandez, U. Ramon-Julvez, and F. Ferraz, “Explainable AI toward understanding the performance of the top three TADPOLE Challenge methods in the forecast of Alzheimer’s disease diagnosis,” PLoS ONE, vol. 17, art. e0264695, 2022.

[19] P. Bhattarai, D. S. Thakuri, Y. Nie, and G. B. Chand, “Explainable AIbased Deep-SHAP for mapping the multivariate relationships between regional neuroimaging biomarkers and cognition,” European Journal of Radiology, vol. 174, art. 111403, 2024.

[20] J. Ouyang, Q. Zhao, E. Adeli, W. Peng, G. Zaharchuk, and K. M. Pohl, “SOM2LM: Self-organized multi-modal longitudinal maps,” in Proc. MICCAI, 2024, pp. 400–410.

[21] R. C. Christodoulou, A. Woodward, R. Pitsillos, R. Ibrahim, and M. F. Georgiou, “Artificial intelligence in Alzheimer’s disease diagnosis and prognosis using PET-MRI: A narrative review of high-impact literature post-Tauvid approval,” Journal of Clinical Medicine, vol. 14, art. 5913, 2025.

[22] K. Prasun and S. K. Sharma, “A deep learning approach: Effective multi-class classification of Alzheimer’s disease using unified integration in the tri-branch network with EfficientNet,” Frontiers in Biomedical Technologies, vol. 12, no. 3, 2026.

[23] V. C. Desai, S. Shetty, T. Sujithra, and T. Manoj, “Classification of Alzheimer’s disease using advanced deep learning and ensemble techniques,” Soft Computing, vol. 30, pp. 691–711, 2026.

[24] Z. Yu, A. Mulholland, T. Huang, and Q. Liu, “Multimodal AI for Alzheimer disease diagnosis: Systematic review of datasets, models, and modalities,” Journal of Medical Internet Research, vol. 28, art. e85414, 2026.

[25] P.-T. Jiang, C.-B. Zhang, Q. Hou, M.-M. Cheng, and Y. Wei, “LayerCAM: Exploring hierarchical class activation maps for localization,” IEEE Transactions on Image Processing, vol. 30, pp. 5875–5888, 2021.

[26] A. Chattopadhyay, A. Sarkar, P. Howlader, and V. N. Balasubramanian, “Grad-CAM++: Improved visual explanations for deep convolutional networks,” in Proc. IEEE Winter Conf. Applications of Computer Vision (WACV), 2018, pp. 839–847.

[27] M. D. Zeiler and R. Fergus, “Visualizing and understanding convolutional networks,” in Computer Vision – ECCV 2014, Lecture Notes in Computer Science, vol. 8689, 2014, pp. 818–833.

[28] O. T. Adeniran, B. Ojeme, T. E. Ajibola, O. O. E. Peter, A. O. Ajala, M. M. Rahman, and F. Khalifa, “Explainable MRI-based ensemble learnable architecture for Alzheimer’s disease detection,” Algorithms, vol. 18, art. 163, 2025.

[29] M. G. Alsubaie, S. Luo, K. Shaukat, W. Zhang, and J. Li, “A novel deep learning approach for Alzheimer’s disease detection: Attentiondriven convolutional neural networks with multi-activation fusion,” AI, vol. 6, art. 324, 2025.

[30] M. Narazani, I. Sarasua, S. Polsterl, A. Lizarraga, I. Yakushev, and¨ C. Wachinger, “Is a PET all you need? A multi-modal study for Alzheimer’s disease using 3D CNNs,” arXiv:2207.02094, 2022.

[31] A. Singla, Q. Zhao, D. K. Do, Y. Zhou, K. M. Pohl, and E. Adeli, “Multiple instance neuroimage transformer,” arXiv:2208.09567, 2022.

[32] J. Song, J. Zheng, P. Li, X. Lu, G. Zhu, and P. Shen, “An effective multimodal image fusion method using MRI and PET for Alzheimer’s disease diagnosis,” Frontiers in Digital Health, vol. 3, art. 637386, 2021.

[33] S. Miao, Q. Xu, W. Li, C. Yang, B. Sheng, F. Liu, T. T. Bezabih, and X. Yu, “MMTFN: Multi-modal multi-scale transformer fusion network for Alzheimer’s disease diagnosis,” International Journal of Imaging Systems and Technology, vol. 34, no. 1, art. e22970, 2024.

[34] R. C. Petersen, P. S. Aisen, L. A. Beckett et al., “Alzheimer’s Disease Neuroimaging Initiative (ADNI): Clinical characterization,” Neurology, vol. 74, no. 3, pp. 201–209, 2010.

[35] F. Isensee, M. Schell, I. Pflueger et al., “Automated brain extraction of multisequence MRI using artificial neural networks,” Human Brain Mapping, vol. 40, no. 17, pp. 4952–4964, 2019.