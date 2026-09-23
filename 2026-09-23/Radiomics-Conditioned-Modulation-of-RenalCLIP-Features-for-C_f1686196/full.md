# Radiomics-Conditioned Modulation of RenalCLIP Features for Clear Cell Renal Cell Carcinoma Classification

Yuan Liang<sup>1,3\*</sup>, Sourav Bhattacharjee<sup>2</sup>, and Abraham Campbell<sup>3</sup>

<sup>1</sup> Research Ireland Centre for Research Training in Machine Learning 2 School of Veterinary Medicine, University College Dublin, Dublin, Ireland 3 School of Computer Science, University College Dublin, Dublin, Ireland Corresponding author: yuan.liang@ucdconnect.ie

Abstract. Radiomics provides quantitative descriptions of tumour appearance that may complement disease-specific foundation models in small labelled cohorts. We investigate this complementarity for computed tomography-based classification of clear cell renal cell carcinoma. Our framework uses radiomics to modulate RenalCLIP features through feature-wise linear modulation (FiLM), while retaining a direct radiomics contribution. Internal testing and external validation compare it with conventional fusion strategies and reference classifiers. The FiLM model achieves an area under the receiver operating characteristic curve (AUC) of 0.804 internally and 0.854 externally, with the highest mean AUC among the evaluated RenalCLIP fusion strategies in both cohorts. Pathway ablations examine the contributions of conditional modulation and the direct radiomics residual, while feature permutation highlights the role of tumour texture. These findings support radiomics as a useful complement to RenalCLIP in a small labelled cohort and identify FiLM as an efective approach to integrating their representations for robust renal tumour classification.

Keywords: Renal cell carcinoma · Radiomics · Foundation models · Feature fusion · Feature-wise linear modulation

## 1 Introduction

Preoperative characterisation of renal masses requires distinguishing tumours with overlapping imaging appearances. Clear cell renal cell carcinoma (ccRCC) can exhibit enhancement patterns diferent from those of other renal tumours, and multiphasic computed tomography (CT) measurements can assist this distinction [17]. Nevertheless, a lesion contains spatial variation in intensity and morphology that is only partly represented by conventional measurements. Computational analysis ofers a means of incorporating this information into tumour classification.

Radiomics characterises a segmented tumour through quantitative measurements of shape, intensity, and texture. These descriptors summarise threedimensional geometry, intensity distributions, and spatial relationships between grey levels. They capture patterns that can be dificult to quantify consistently through visual inspection and have been investigated as markers of tumour phenotype [1,10]. Their explicit definitions link classifier inputs to measurable tumour properties.

This structured representation is particularly relevant when labelled data are limited. Its feature extractor does not require target-cohort training, and feature selection can prioritise reproducible, nonredundant measurements. Although predefined descriptors cannot encompass every aspect of tumour appearance, they provide a compact quantitative basis that may complement learned image representations [10].

Deep learning provides another way to represent the same image. Residual networks learn spatial filters from data [8], while vision–language pretraining relates image appearance to associated textual information [12]. RenalCLIP applies disease-specific visual–textual learning to kidney cancer [14]. Its image representation incorporates information learned from a large pretraining cohort, but transfer to a particular histological distinction still depends on how the representation is adapted. We hypothesise that explicit radiomic descriptors can supplement RenalCLIP features when the target task has relatively few labelled examples.

The way these representations interact may influence how efectively that supplementary information is used. Concatenation makes both vectors available to a common classifier. Gated fusion learns their relative weighting [2]. Featurewise linear modulation (FiLM) instead uses a conditioning input to generate an afine transformation of another representation [11]. Radiomics can therefore do more than contribute additional classifier inputs. It can guide how image features are scaled and shifted for each patient.

We propose a radiomics-conditioned residual FiLM framework for ccRCC classification. The model uses radiomics both to modulate RenalCLIP features and to provide a direct residual representation for the classifier. We compare this formulation with concatenation, alternative gates, and globally shared afine modulation, together with conventional radiomics and convolutional references. Internal and external evaluation assesses classification performance, while pathway ablations and radiomics permutation analysis examine how the model uses the additional quantitative information.

## 2 Methods

## 2.1 Study Cohorts

The internal cohort comprised 396 KiTS23 patients [9] with CT, tumour masks, histological labels, and complete feature records. ccRCC was defined as the positive class and the remaining histologies as the negative class. The fixed patient-level split comprised 237 training, 99 validation, and 60 test patients, with stratification used to maintain an approximately 70% ccRCC prevalence across the three subsets. Multiple tumour delineations were combined using STAPLE consensus where available [16].

The external cohort combined RCC-AID data [3] with previously assembled TCGA cases using AIMI annotations [15]. After patient-level deduplication and image-quality review, 111 patients remained, comprising 85 ccRCC and 26 nonccRCC cases (76.6% ccRCC). All methods were evaluated on the same selected CT scan and tumour mask for each patient.

## 2.2 CT Processing and Feature Representations

Radiomics. CT volumes were consistently oriented, resampled to 1-mm isotropic spacing, and clipped to [−150, 200] Hounsfield units (HU). Masks underwent nearest-neighbour resampling. Features were extracted with PyRadiomics [7], using definitions described by the Image Biomarker Standardization Initiative [18]. Original-image shape descriptors were combined with first-order and texture features from the original and Laplacian-of-Gaussian $\left( \mathrm { L o G } \right)$ images. LoG scales were $\sigma \in \{ 1 , 2 , 3 \}$ mm and the intensity bin width was 25 HU. Filtering preceded cropping.

The initial 386 descriptors were reduced to 73 using inter-annotator $\mathrm { I C C } ( 3 , 1 ) \geq$ 0.75 [13], exclusion below the 20th percentile of median absolute deviation (MAD), and correlation filtering at $| \rho | > 0 . 9 5$ . Average-linkage clustering with distance $1 - | \rho |$ and a cut of 0.2 retained the highest-MAD feature per cluster. Median imputation and standardisation parameters were fitted on training patients and applied unchanged to subsequent datasets, producing $\mathbf { x } \in \mathbb { R } ^ { 7 3 }$

RenalCLIP. The public RenalCLIP image encoder [14] provided disease-specific image representations. Its pretraining data came from independent institutional cohorts in China and did not overlap with the present study datasets. The 3D ResNet18 encoder and its batch-normalisation statistics were held fixed. We pooled its third residual-stage feature map using tumour-mask weights, producing $\mathbf { z } \in \mathbb { R } ^ { 2 5 6 }$ after L2 normalisation. Mask voxels were distributed trilinearly to stride-aligned centres on the $2 \times 8 \times 8$ feature grid. The pooled vector was the weighted mean of these features. This emphasised tumour-associated locations while retaining contextual receptive fields.

An $8 7 . 5 \times 8 7 . 5 \times 1 0 0$ mm field of view was centred on the tumour centroid in the axial slice with the largest tumour area. Trilinear sampling produced $1 4 0 \times 1 4 0 \times 3 2$ voxels, followed by a $1 2 8 \times 1 2 8 \times 3 2$ centre crop, giving a final $8 0 \times 8 0 \times 1 0 0$ mm field of view. Windowing at centre 50 HU and width 500 HU mapped intensities to [0,1]. The same processing was applied across cohorts.

## 2.3 Radiomics-Conditioned Residual Modulation

The proposed framework uses radiomics to condition the image representation while preserving a direct radiomics contribution $( \mathrm { F i g . ~ 1 ) }$ . Let $P _ { I }$ and $P _ { R }$ denote independently learned projections with ReLU activation and dropout. They map image and radiomics inputs into a common feature space,

$$
\mathbf { i } = P _ { I } ( \mathbf { z } ) , \qquad \mathbf { r } = P _ { R } ( \mathbf { x } ) , \qquad \mathbf { i } , \mathbf { r } \in \mathbb { R } ^ { 1 2 8 } .\tag{1}
$$

A conditioning network with dimensions 128 → 128 → 256 generates the patientspecific modulation parameters,

$$
\mathrm { c o n c a t } ( \gamma ( \mathbf { r } ) , \beta ( \mathbf { r } ) ) = W _ { 2 } \mathrm { R e L U } ( W _ { 1 } \mathbf { r } + \mathbf { b } _ { 1 } ) + \mathbf { b } _ { 2 } .\tag{2}
$$

The fused representation is defined as

$$
\mathbf { h } = \frac { 1 } { 2 } \left[ \mathbf { i } \odot \left\{ 1 + \operatorname { t a n h } \gamma ( \mathbf { r } ) \right\} + \beta ( \mathbf { r } ) + \mathbf { r } \right] ,\tag{3}
$$

where ⊙ denotes element-wise multiplication. The multiplicative term adapts the contribution of each image coordinate within a bounded range. The additive term adjusts its ofset, and the direct residual preserves the projected radiomics information. Zero initialisation of the final conditioning layer gives $\mathbf { h } = ( \mathbf { i } + \mathbf { r } ) / 2$ at the start of optimisation. A linear classifier followed by softmax maps h to the ccRCC probability.

![](images/85bb1bfe3e2c533668ea0d4a93ac066b2324617642dde98b29b1a04ee6d03f8c.jpg)  
Fig. 1. Radiomics-conditioned residual FiLM. Radiomics guides the afine transformation of image features and contributes directly to the fused representation. The image encoder is fixed while the projections, conditioning network, and classifier are learned.

## 2.4 Comparison Methods

A radiomics-based radial-basis-function support-vector classifier (SVC) [4] served as the conventional reference. Convolutional comparisons comprised a 3D ResNet18 trained from scratch, together with its radiomics-gated and radiomics-FiLM variants. These models used normalised $1 2 8 ^ { 3 }$ tumour-centred CT inputs and 512-dimensional ResNet18 representations for fusion.

For RenalCLIP, a multilayer perceptron with dimensions $2 5 6  1 2 8  2$ classified image features. For concatenation, the 256-dimensional RenalCLIP representation and the 73-dimensional radiomics vector were concatenated directly, yielding a 329-dimensional input to a $3 2 9  6 4  2$ classifier. Gated fusion combined the projections as

$$
\mathbf { h } _ { g } = \mathbf { g } \odot \mathbf { i } + ( 1 - \mathbf { g } ) \odot \mathbf { r } .\tag{4}
$$

The gate was conditioned either on radiomics alone or jointly on both representations. In the first case, $\mathbf { g } = { \mathrm { s i g m o i d } } ( G _ { R } ( \mathbf { r } ) )$ . In the second, $G _ { J }$ received the concatenated image and radiomics projections. Both generators had a 128- dimensional hidden layer. A global afine comparison replaced the patient-specific γ and $\beta$ in $\operatorname { E q . }$ 3 with vectors shared across patients. This comparison assesses the role of adaptive conditioning within the same residual formulation.

## 2.5 Training and Evaluation

RenalCLIP-based classifiers were optimised with cross-entropy and AdamW at learning rate $1 0 ^ { - 3 }$ , with batch size 16 and zero weight decay. Dropout of 0.2 was used in the 128-dimensional projections and image-only classifier. The concatenation classifier used no dropout. Convolutional references used class-weighted cross-entropy and Adam at learning rate $1 0 ^ { - 4 }$ with mixed precision. Their training augmentation included spatial flips, rotations, and afine perturbations. Models were selected according to internal validation AUC and applied to the test and external cohorts without further fitting.

The primary metric was AUC, with average precision (AP) as a complementary measure. RenalCLIP-based methods were evaluated over three independent training runs. The SVC and convolutional references each used one trained model. Internal results report mean and sample standard deviation (SD) for repeated runs and 95% confidence intervals (CIs) for single-model references. External results report patient-bootstrap CIs. Each bootstrap sample preserved class counts and the correspondence of predictions across runs. For methods with repeated training, metrics were calculated separately and then averaged. CIs used 2,000 resamples [5] and quantify patient-sampling uncertainty for the fitted models.

## 2.6 Pathway Ablations and Feature Permutation

Two inference-time ablations examined the radiomics pathways in the fitted FiLM model. Removing the direct residual gives $\mathbf { h } _ { \mathrm { n o ~ d i r e c t } } = \mathbf { h } - \mathbf { r } / 2 .$ preserving conditional modulation. Disabling modulation gives $\mathbf { h } _ { \mathrm { n o } }$ modulation $= ( \mathbf { i } + \mathbf { r } ) / 2$ preserving the projected representations. All other weights were held fixed.

For feature importance, each radiomics descriptor was independently permuted across internal-test patients 50 times [6]. The same permutations were used across training runs. The radiomics projection and modulation parameters were recomputed after each permutation. Importance was defined as the decrease in AUC, averaged first across permutations and then across runs.

## 3 Results and Discussion

## 3.1 Internal Classification Performance

FiLM achieved the highest mean internal AUC among the evaluated RenalCLIP fusion strategies, reaching 0.804 (Table 1) and improving on RenalCLIP alone by approximately 9.8 percentage points. Global afine modulation achieved the next highest mean AUC, followed by the gated and concatenation approaches. The consistent improvement of the fusion approaches over the image-only classifier supports supplementing transferred features with radiomics.

Table 1. Internal-test performance. RenalCLIP methods report three-run mean $\pm \ \mathrm { S D }$ Single-model references report 95% CIs. All fusion methods include radiomics. Ablations use the fitted FiLM models.
<table><tr><td>Method AUC AP</td></tr><tr><td>Radiomics SVC 0.7659 [0.6296, 0.8836] 0.8938 [0.8259, 0.9500]</td></tr><tr><td>3D ResNet18 0.5324 [0.3651, 0.6971] 0.7376 [0.6450, 0.8459]</td></tr><tr><td>ResNet18 + rad. gate 0.7844 [0.6428, 0.9101] 0.8651 [0.7736, 0.9615]</td></tr><tr><td>ResNet18 + rad. FiLM 0.6865 [0.5384, 0.8307] 0.8205 [0.7305, 0.9245]</td></tr><tr><td>RenalCLIP only  $0 . 7 0 6 3 \pm 0 . 0 0 2 6$   $0 . 7 9 2 1 \pm 0 . 0 0 1 1$ </td></tr><tr><td>+ concatenation  $0 . 7 8 6 6 \pm 0 . 0 0 6 8$   $0 . 8 5 5 4 \pm 0 . 0 0 9 1$ </td></tr><tr><td>+ radiomics gate  $0 . 7 9 2 8 \pm 0 . 0 1 7 9$   $0 . 8 5 5 7 \pm 0 . 0 1 8 4$ </td></tr><tr><td>+ joint gate  $0 . 7 9 2 8 \pm 0 . 0 1 8 1$   $0 . 8 6 4 0 \pm 0 . 0 0 8 0$ </td></tr><tr><td>+ global affine  $0 . 7 9 8 9 \pm 0 . 0 1 8 2$   $0 . 8 5 9 6 \pm 0 . 0 1 4 5$ </td></tr><tr><td>+ conditional FiLM  $0 . 8 0 3 8 \pm 0 . 0 0 8 1$   $0 . 8 7 1 4 \pm 0 . 0 0 5 9$ </td></tr><tr><td>Inference-time pathway ablations of conditional FiLM</td></tr><tr><td>Without direct residual  $0 . 7 9 5 9 \pm 0 . 0 0 3 8$   $0 . 8 6 1 4 \pm 0 . 0 1 0 2$ </td></tr><tr><td>Without modulation  $0 . 8 0 1 6 \pm 0 . 0 2 1 0$   $0 . 8 6 8 9 \pm 0 . 0 2 0 4$ </td></tr></table>

The similar internal performance of global afine modulation and FiLM indicates that a shared transformation also captured useful relationships between the representations. SVC remained competitive and achieved the highest internal AP, highlighting the discriminatory information already present in radiomics. Within the convolutional comparisons, gating performed better than FiLM. The benefit of a fusion strategy therefore depended on the representation and learning configuration.

## 3.2 External Evaluation

FiLM achieved the highest mean external AUC among the evaluated RenalCLIP fusion strategies, reaching 0.854 (Table 2). This exceeded the image-only classifier by 9.86 percentage points. SVC retained a slightly higher AUC, whereas FiLM achieved a higher AP than SVC.

FiLM achieved the highest mean AUC in both the internal and external RenalCLIP fusion comparisons, although the margins over global afine modulation internally and concatenation externally were small. Radiomics provided a consistent improvement across cohorts, with the integration strategy contributing a smaller additional efect.

Table 2. External performance on 111 patients. Brackets denote 95% patient-bootstrap CIs. For methods with three training runs, the point estimate and each bootstrap replicate are means of individual-model metrics.
<table><tr><td>Method</td><td colspan="4">AUC</td></tr><tr><td>Radiomics SVC</td><td>0.8593 [0.7611, 0.9362]</td><td></td><td>0.9378 [0.8796, 0.9807]</td><td></td></tr><tr><td>3D ResNet18</td><td></td><td>0.5853 [0.4566, 0.7088]</td><td></td><td>0.8404 [0.7924, 0.8926]</td></tr><tr><td>ResNet18 + rad. gate</td><td>0.8371 [0.7516, 0.9091]</td><td></td><td>0.9479 [0.9190, 0.9723]</td><td></td></tr><tr><td>ResNet18 + rad. FiLM</td><td>0.7620 [0.6629, 0.8498]</td><td></td><td>0.9221 [0.8876, 0.9543]</td><td></td></tr><tr><td>RenalCLIP only</td><td></td><td></td><td></td><td></td></tr><tr><td>+ concatenation</td><td></td><td>0.7549 [0.6422, 0.8537] 0.8484 [0.7721, 0.9158] 0.9542 [0.9304, 0.9755]</td><td></td><td>0.9094 [0.8632, 0.9510]</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td>+ radiomics gate</td><td></td><td>0.8388 [0.7603, 0.9101] 0.9491 [0.9228, 0.9724]</td><td></td><td>0.8416 [0.7617, 0.9139] 0.9500 [0.9228, 0.9741]</td></tr><tr><td>+ joint gate + global affine</td><td></td><td>0.8398 [0.7602, 0.9118] 0.9489 [0.9212, 0.9730]</td><td></td><td></td></tr><tr><td>+ conditional FiLM</td><td></td><td>0.8535 [0.7781, 0.9244] 0.9528 [0.9253, 0.9763]</td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr></table>

## 3.3 Pathway Ablations

Removing the direct radiomics residual and disabling conditional modulation reduced mean internal AUC by 0.79 and 0.22 percentage points, respectively (Table 1). Both pathways contributed numerically to the fitted model, with a larger change after removing the direct residual. Retaining modulation allowed radiomics to influence predictions indirectly, while retaining the residual preserved its explicit contribution.

The efects were modest and indicate overlapping information routes. Additive modulation can itself carry radiomics information, partly compensating for removal of the direct residual. These ablations describe dependence within the fitted model, rather than the performance of independently retrained alternatives. They support using the complete formulation without establishing that either pathway is individually indispensable.

## 3.4 Radiomics Feature Importance

Texture descriptors dominated the highest-ranked permutation efects (Fig. 2). LoG 2-mm GLCM information measure of correlation 2, original-image GLDM dependence entropy, and LoG 1-mm zone entropy were the leading features. These descriptors quantify diferent aspects of spatial intensity organisation and heterogeneity. Their prominence indicates that the additional radiomics information used by the model extended beyond tumour size or a single intensity summary.

Because a permuted descriptor afects both the radiomics projection and its modulation parameters, these scores reflect its contribution to the full fusion model. Correlated descriptors may share predictive information, so a small individual permutation efect does not imply that the corresponding image property is unimportant [6]. The analysis provides a quantitative account of model reliance, rather than a direct biological attribution.

![](images/c78dc8e9ca6ed64a9edbfc909a735e80ee49c479d8d57daeb1803d8b51c76cae.jpg)  
Fig. 2. The ten radiomics features with the largest mean AUC decrease under permutation. Blue markers and bars show mean ± SD across runs, and grey markers show individual-run means over 50 permutations. GLCM, GLRLM, GLSZM, and GLDM denote co-occurrence, run-length, size-zone, and dependence matrices. GLNU denotes grey-level nonuniformity.

## 3.5 Complementarity and Fusion Design

The central finding is that explicit radiomic descriptors supplement RenalCLIP for classification with limited labelled data. RenalCLIP reflects patterns learned through disease-specific pretraining, whereas radiomics directly supplies predefined intensity and texture measurements. This distinction provides a plausible basis for complementarity. Information implicit in a pooled image representation may become easier to exploit when an associated quantitative descriptor is supplied explicitly.

FiLM incorporates that information through a transformation conditioned on each patient’s radiomics profile. Multiplicative modulation changes the contribution of image coordinates, while additive modulation adjusts their ofsets. The direct residual preserves descriptors that are already useful for classification. Together, these mechanisms allow radiomics to contribute both explicit predictive information and context for adapting the learned representation.

FiLM achieved the highest mean AUC within the RenalCLIP fusion comparisons in both cohorts, suggesting that patient-specific conditioning can be useful for integrating the two representations. However, the competitive performance of global afine modulation, concatenation, and gated fusion shows that simpler in teractions also exploit the radiomic descriptors efectively. The modest diferences between fusion methods, together with the strong SVC results, indicate that radiomics supplies much of the available discriminatory information. The most directly supported complementarity is radiomics supplementing the transferred image representation, while the incremental value of conditional modulation remains modest.

## 3.6 Limitations and Future Work

The study is retrospective and the internal test cohort is relatively small. The convolutional and SVC references use single trained models, while RenalCLIP methods use repeated runs. Comparisons between frozen-feature and end-toend approaches should be interpreted cautiously because their preprocessing and optimisation procedures difered. Larger independent cohorts and repeated training of all references would provide a more precise assessment of their relative performance. Future work should examine how conditional fusion behaves across acquisition settings and whether its efects persist under changes in segmentation and feature extraction.

## 4 Conclusion

Radiomics provides complementary information for RenalCLIP-based ccRCC classification in a small labelled cohort. Residual FiLM uses these descriptors to modulate image features while preserving their direct contribution. It achieved AUCs of 0.804 internally and 0.854 externally, with the highest mean AUC among the evaluated RenalCLIP fusion strategies in both cohorts. Pathway ablations indicated overlapping radiomics contributions, and permutation analysis highlighted texture descriptors. These findings support conditional modulation for integrating the representations and motivate confirmation across larger, independent cohorts.

## References

1. Aerts, H.J.W.L., Rios Velazquez, E., Leijenaar, R.T.H., et al.: Decoding tumour phenotype by noninvasive imaging using a quantitative radiomics approach. Nat. Commun. 5, 4006 (2014). https://doi.org/10.1038/ncomms5006

2. Arevalo, J., Solorio, T., Montes-y Gómez, M., González, F.A.: Gated multimodal units for information fusion. arXiv:1702.01992 (2017). https://doi.org/10.48550/a rXiv.1702.01992

3. de Boer, S., Häntze, H., Ziegelmayer, S., et al.: RCC-AID: Renal cell carcinoma AI dataset for medical imaging research. Zenodo, version 2.0.0 (2026). https: //doi.org/10.5281/zenodo.20719257

4. Cortes, C., Vapnik, V.: Support-vector networks. Mach. Learn. 20, 273–297 (1995). https://doi.org/10.1007/BF00994018

5. Efron, B.: Bootstrap methods: another look at the jackknife. Ann. Stat. 7(1), 1–26 (1979). https://doi.org/10.1214/aos/1176344552

6. Fisher, A., Rudin, C., Dominici, F.: All models are wrong, but many are useful: learning a variable’s importance by studying an entire class of prediction models simultaneously. J. Mach. Learn. Res. 20(177), 1–81 (2019), https://jmlr.org/paper s/v20/18-760.html

7. van Griethuysen, J.J.M., Fedorov, A., Parmar, C., et al.: Computational radiomics system to decode the radiographic phenotype. Cancer Res. 77(21), e104–e107 (2017). https://doi.org/10.1158/0008-5472.CAN-17-0339

8. He, K., Zhang, X., Ren, S., Sun, J.: Deep residual learning for image recognition. In: Proc. IEEE Conf. Computer Vision and Pattern Recognition. pp. 770–778 (2016). https://doi.org/10.1109/CVPR.2016.90

9. Heller, N., Isensee, F., Tejpau, R., Wood, A., Papanikolopoulos, N., Weight, C.: 2023 kidney and kidney tumor segmentation challenge. Zenodo (2023). https: //doi.org/10.5281/zenodo.7840134

10. Lambin, P., Leijenaar, R.T.H., Deist, T.M., et al.: Radiomics: the bridge between medical imaging and personalized medicine. Nat. Rev. Clin. Oncol. 14(12), 749–762 (2017). https://doi.org/10.1038/nrclinonc.2017.141

11. Perez, E., Strub, F., de Vries, H., Dumoulin, V., Courville, A.: FiLM: Visual reasoning with a general conditioning layer. In: Proc. AAAI Conference on Artificial Intelligence. vol. 32, pp. 3942–3951 (2018). https://doi.org/10.1609/aaai.v32i1.11671

12. Radford, A., Kim, J.W., Hallacy, C., et al.: Learning transferable visual models from natural language supervision. In: Proc. 38th International Conference on Machine Learning. Proceedings of Machine Learning Research, vol. 139, pp. 8748–8763 (2021), https://proceedings.mlr.press/v139/radford21a.html

13. Shrout, P.E., Fleiss, J.L.: Intraclass correlations: uses in assessing rater reliability. Psychol. Bull. 86(2), 420–428 (1979). https://doi.org/10.1037/0033-2909.86.2.420

14. Tao, Y., Zhao, Z., Wang, Z., et al.: A disease-centric vision-language foundation model for precision oncology in kidney cancer. Nat. Commun. 17, 7313 (2026). https://doi.org/10.1038/s41467-026-74175-w

15. Van Oss, J., Murugesan, G.K., McCrumb, D., Soni, R.: Image segmentations produced by BAMF under the AIMI annotations initiative. Zenodo, version 1.7 (2023). https://doi.org/10.5281/zenodo.10081112

16. Warfield, S.K., Zou, K.H., Wells, W.M.: Simultaneous truth and performance level estimation (STAPLE): an algorithm for the validation of image segmentation. IEEE Trans. Med. Imaging 23(7), 903–921 (2004). https://doi.org/10.1109/TMI.2004.8 28354

17. Young, J.R., Margolis, D., Sauk, S., Pantuck, A.J., Sayre, J., Raman, S.S.: Clear cell renal cell carcinoma: discrimination from other renal cell carcinoma subtypes and oncocytoma at multiphasic multidetector CT. Radiology 267(2), 444–453 (2013). https://doi.org/10.1148/radiol.13112617

18. Zwanenburg, A., Vallières, M., Abdalah, M.A., et al.: The image biomarker standardization initiative: standardized quantitative radiomics for high-throughput image-based phenotyping. Radiology 295(2), 328–338 (2020). https://doi.org/10.1 148/radiol.2020191145