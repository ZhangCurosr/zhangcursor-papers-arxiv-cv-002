# INTERPRETABLE MULTIMODAL MRI–CLINICAL FUSION FOR FUNCTIONAL OUTCOME PREDICTION AFTER ACUTE ISCHEMIC STROKE

Sina Raeisadigh<sup>1,4</sup>, Henning Muller¨ <sup>2,5</sup>, Abderrahmane Hedjoudje<sup>3</sup>

<sup>1</sup> Department of Computer Science, University of Geneva, Switzerland <sup>2</sup> Service of Medical Informatics, University Hospital of Geneva, Switzerland <sup>3</sup> Department of Imaging and Medical Informatics, University of Geneva, Switzerland <sup>4</sup> Informatics Institute, University of Applied Sciences Western Switzerland (HES-SO), Sierre, Switzerland <sup>5</sup> The Sense, Lausanne and Sion, Switzerland

## ABSTRACT

This study evaluates the added prognostic value of day-1 multimodal MRI for predicting three-month functional outcome after acute ischemic stroke (AIS). Sixty-six AIS patients with ADC, DWI, FLAIR, and clinical data were analyzed. Three-dimensional pretrained ResNet embeddings were extracted from each MRI modality, reduced using fold-isolated principal component analysis, and combined with structured clinical variables and ADC-derived lesion biomarkers. Linear support vector machines were evaluated using eight-fold stratified group cross-validation across MRI modality combinations. Clinical-only modeling achieved strong performance (AUC = 0.899±0.151), while MRI-only models showed limited discrimination (AUC ≈ 0.67–0.73). Among multimodal configurations, FLAIR-clinical fusion achieved the highest mean cross-validated performance (AUC = 0.948 ± 0.061), followed by ADC-clinical fusion (AUC = 0.941 ± 0.094). Full ADC-DWI-FLAIR fusion did not consistently outperform single-modality clinical fusion. Feature-importance analysis identified age, NIHSS subscores, ADC lesion burden, and learned FLAIR representations as key contributors, while qualitative FLAIR activation maps showed responses over lesion-like regions. These findings suggest that compact day-1 FLAIR and ADC representations provide complementary prognostic information beyond clinical variables and support interpretable, data-efficient outcome prediction in AIS.

Index Terms— acute ischemic stroke, multimodal MRI, clinical-imaging fusion, functional outcome prediction, modified Rankin Scale, explainable AI

## 1. INTRODUCTION

Acute ischemic stroke (AIS) remains a major cause of adult disability and long-term functional dependence worldwide [1]. Early prediction of three-month functional outcome is clinically important because it can support treatment planning, rehabilitation allocation, patient counseling, and trial stratification. The modified Rankin Scale (mRS) is widely used as the standard functional outcome endpoint after stroke, with poor outcome commonly defined as mRS > 1 or mRS > 2 depending on the clinical setting [2]. However, accurate individualized prognosis remains challenging because recovery depends on complex interactions between baseline neurological severity, age, vascular risk factors, lesion burden, treatment response, and early tissue evolution.

Clinical variables such as age, pre-stroke disability, and the National Institutes of Health Stroke Scale (NIHSS) are among the strongest predictors of post-stroke outcome [3, 4]. Nevertheless, clinical scores alone do not directly capture lesion location, infarct extent, or tissue-level changes after reperfusion. Magnetic resonance imaging (MRI), especially diffusion-weighted imaging (DWI) and apparent diffusion coefficient (ADC) maps, provides sensitive markers of acute ischemic injury [5]. Fluid-attenuated inversion recovery (FLAIR) imaging can provide complementary information related to lesion visibility, edema, and subacute tissue changes. Together, these modalities may improve outcome prediction by representing different aspects of ischemic tissue damage.

Recent machine learning and deep learning methods have shown promise for combining imaging and clinical data in stroke prognosis [6, 7, 8]. Three-dimensional convolutional neural networks can extract volumetric representations from MRI without requiring handcrafted radiomic features, while linear classifiers can provide more transparent decision boundaries in small clinical cohorts [9, 10, 11]. However, many studies either focus on a single imaging modality, use imaging as a black-box input, or do not explicitly separate the contribution of clinical variables, MRI modality choice, and lesion-derived biomarkers. This makes it difficult to determine whether performance gains come from deep image representations, structured clinical predictors, or explicit lesion burden measures.

A further challenge is data efficiency. Stroke imaging datasets with complete longitudinal clinical outcomes are often small, making end-to-end deep learning prone to overfitting. Hybrid approaches that use pretrained 3D networks as feature extractors, followed by fold-isolated dimensionality reduction and linear classification, may offer a practical compromise between representation learning and interpretability [12, 13, 14]. In this setting, systematic evaluation of MRI modality combinations is important: adding more modalities does not necessarily improve generalization, especially when sample size is limited and modality-specific features are redundant or noisy. Our previous work showed that day-1 ADC-based multimodal modeling improved threemonth mRS prediction compared with baseline ADC imaging, highlighting the prognostic value of early post-treatment diffusion changes [15]. However, that study focused primarily on ADC and did not systematically evaluate the relative contribution of ADC, DWI, and FLAIR modality combinations or explicitly separate the effect of ADC-derived lesion biomarkers from MRI image inputs.

In this work, we evaluate day-1 multimodal MRI fusion for predicting three-month functional outcome after AIS in a quality-controlled cohort of 66 patients. We compare ADC, DWI, and FLAIR representations individually and in combination, with and without structured clinical variables and ADC-derived lesion biomarkers. We use pretrained 3D ResNet feature extraction, fold-isolated principal component analysis, and class-balanced linear support vector machines under eight-fold stratified group cross-validation. In addition to predictive performance, we provide interpretability through filtered linear coefficient analysis and qualitative FLAIR CNN activation maps. Our goal is to identify which imaging modalities and lesion-derived features add prognostic value beyond clinical variables, while maintaining a data-efficient and interpretable modeling framework.

## 2. METHODS

## 2.1. Cohort and Outcome

This retrospective study included 66 patients with acute ischemic stroke (AIS) who had day-1 ADC, DWI, and FLAIR MRI available after quality control, together with structured clinical variables and three-month modified Rankin Scale (mRS) outcome. Clinical variables included demographics, vascular risk factors, pre-stroke disability, and NIHSS subscores at baseline and day 1. The binary endpoint was favorable outcome $( \mathrm { m } \mathrm { R } S \leq 1 )$ versus unfavorable outcome (mRS $> 1 )$ . Patient identifiers were used as grouping variables to prevent leakage across cross-validation folds.

## 2.2. MRI Preprocessing

MRI data were loaded from clinical DICOM series and processed in physical space. Candidate image series were identified from DICOM metadata, grouped by SeriesInstanceUID, and the largest valid image series was selected for each patient and modality. Slice ordering was checked using spatial DICOM information when available, and duplicate slice positions were removed. ADC, DWI, and FLAIR volumes were resampled with linear interpolation to a common physical spacing of $1 . 0 2 \times 1 . 0 2 \times 5 . 4$ mm and converted to depthheight-width order. Volumes were center-padded to $3 2 \times$ $2 5 6 \times 2 5 6$ without cropping anatomy. Patients were excluded if any requested modality failed prespecified physical coverage requirements. Per-volume robust intensity normalization was applied using finite, nonzero valid voxels while excluding padded background.

## 2.3. Lesion Biomarkers and MRI Features

Three ADC-derived lesion-burden features were used: lesions defined at ADC thresholds of $4 8 0 \times 1 0 ^ { - 6 } \mathrm { m m ^ { 2 } / s }$ and $6 2 0 \times$ $1 0 ^ { - 6 } \mathrm { m m } ^ { 2 } / \mathrm { s }$ , and a combined weighted lesion measure incorporating both thresholds. Small isolated connected components were removed before feature calculation. MRI representations were extracted from ADC, DWI, and FLAIR using pretrained three-dimensional ResNet branches implemented in MONAI. Modality embeddings were evaluated individually and in all possible combinations to assess the relative contribution of ADC, DWI, FLAIR, and multimodal MRI fusion.

## 2.4. Clinical-Imaging Fusion

For each configuration, selected MRI embeddings were concatenated with the corresponding clinical/tabular feature block. We evaluated MRI-only models, clinical-only models, and MRI-clinical fusion models. To isolate the contribution of ADC-derived lesion biomarkers, clinical and MRI-clinical configurations were evaluated with and without these lesion features. The clinical-only model without ADC lesion biomarkers was used as the reference baseline for reporting incremental performance.

## 2.5. Classification and Evaluation

Dimensionality reduction and feature scaling were performed within each training fold and then applied to the held-out validation fold to prevent information leakage. Classification used a class-balanced linear support vector machine. Models were evaluated using eight-fold stratified group crossvalidation with patient-level grouping. Performance was reported as mean ± standard deviation across validation folds using AUC, accuracy, and F1-score. Models were ranked by mean cross-validated AUC. Network and classifier settings were fixed across modality comparisons, and all learned preprocessing was fitted within each training fold. The primary FLAIR-clinical model with ADC-derived lesion biomarkers was compared with the clinical-only baseline using an exact two-sided paired Wilcoxon signed-rank test on validation AUCs from the corresponding cross-validation folds.

Table 1. Cross-validation performance for selected configurations. Values are mean ± standard deviation across folds. Delta AUC is computed relative to the clinical-only model without ADC lesion biomarkers.
<table><tr><td>Configuration</td><td>Train AUC</td><td>Val AUC</td><td>Val Acc</td><td>Val F1</td><td>∆ Val AUC</td></tr><tr><td> $\overline { { \mathrm { F L A I R + c l i n i c a l + A D C \ l e s i o n s } } }$ </td><td> $\overline { { 0 . 9 4 9 \pm 0 . 0 1 8 } }$ </td><td> $\overline { { { \bf 0 . 9 4 8 \pm 0 . 0 6 1 } } }$ </td><td> $\overline { { 0 . 8 0 2 \pm 0 . 0 9 3 } }$ </td><td> $\overline { { 0 . 7 3 2 \pm 0 . 1 7 5 } }$ </td><td>+0.049</td></tr><tr><td> $\mathrm { A D C } + \mathrm { c l i n i c a l + A D C \ l e s i o n s }$ </td><td> $0 . 9 4 1 \pm 0 . 0 1 2$ </td><td> $0 . 9 4 1 \pm 0 . 0 9 4$ </td><td> $0 . 8 1 6 \pm 0 . 0 9 7$ </td><td> $0 . 7 7 6 \pm 0 . 0 8 6$ </td><td>+0.042</td></tr><tr><td> $\mathrm { A D C } + \mathrm { F L A I R } + \mathrm { c l i n i c a l } + \mathrm { A D C \ l e s i o n s }$ </td><td> $0 . 9 5 9 \pm 0 . 0 0 6$ </td><td> $0 . 9 3 2 \pm 0 . 0 7 0$ </td><td> $0 . 7 8 8 \pm 0 . 1 0 5$ </td><td> $0 . 7 4 7 \pm 0 . 1 6 1$ </td><td>+0.033</td></tr><tr><td> $\mathrm { A D C } + \mathrm { D W I } + \mathrm { F L A I R } + \mathrm { c l i n i c a l } + \mathrm { A D C \ l e s i o n s }$ </td><td> $0 . 9 6 3 \pm 0 . 0 1 1$ </td><td> $0 . 9 2 6 \pm 0 . 1 0 7$ </td><td> $0 . 7 0 8 \pm 0 . 1 3 0$ </td><td> $0 . 6 3 6 \pm 0 . 1 9 8$ </td><td>+0.027</td></tr><tr><td> $\mathrm { A D C } + \mathrm { D W I } + \mathrm { c l i n i c a l } + \mathrm { A D C \ l e s i o n s }$ </td><td> $0 . 9 5 6 \pm 0 . 0 1 4$ </td><td> $0 . 9 2 2 \pm 0 . 1 1 8$ </td><td> $0 . 7 5 5 \pm 0 . 1 1 8$ </td><td> $0 . 6 7 4 \pm 0 . 1 8 4$ </td><td>+0.023</td></tr><tr><td>Clinical only, without ADC lesions</td><td> $0 . 9 1 3 \pm 0 . 0 1 7$ </td><td> $0 . 8 9 9 \pm 0 . 1 5 1$ </td><td> $\mathbf { 0 . 8 4 5 \pm 0 . 1 6 1 }$ </td><td> $\mathbf { 0 . 8 2 0 \pm 0 . 1 4 5 }$ </td><td>0.000</td></tr><tr><td>Clinical only, with ADC lesions</td><td> $0 . 9 2 3 \pm 0 . 0 1 7$ </td><td> $0 . 8 9 0 \pm 0 . 1 4 7$ </td><td> $0 . 7 6 7 \pm 0 . 1 5 7$ </td><td> $0 . 7 0 2 \pm 0 . 1 8 5$ </td><td>-0.010</td></tr><tr><td>FLAIR only</td><td> $0 . 6 8 0 \pm 0 . 0 4 0$ </td><td> $0 . 7 1 9 \pm 0 . 2 3 5$ </td><td> $0 . 6 0 9 \pm 0 . 1 8 5$ </td><td> $0 . 5 4 7 \pm 0 . 2 6 4$ </td><td>-0.180</td></tr><tr><td>DWI only</td><td> $0 . 7 1 1 \pm 0 . 0 2 1$ </td><td> $0 . 7 0 6 \pm 0 . 1 9 9$ </td><td> $0 . 6 2 0 \pm 0 . 1 4 0$ </td><td> $0 . 5 4 4 \pm 0 . 0 8 3$ </td><td>-0.193</td></tr><tr><td>ADC only</td><td> $0 . 6 5 2 \pm 0 . 0 2 9$ </td><td> $0 . 6 7 5 \pm 0 . 2 2 0$ </td><td> $0 . 6 4 1 \pm 0 . 1 9 6$ </td><td> $0 . 5 6 5 \pm 0 . 3 0 9$ </td><td>-0.224</td></tr></table>

## 2.6. Explainability

Global interpretability was assessed using a direct linear probe on FLAIR embeddings, clinical variables, and ADC lesion biomarkers. Rare binary variables were excluded from displayed feature-importance plots to reduce coefficient instability. Feature importance was summarized from average linear coefficients across folds. Local imaging interpretation used qualitative intermediate FLAIR CNN activation maps, shown alone and overlaid on raw FLAIR slices. These maps were used as qualitative activation visualizations, not as lesion segmentation or causal attribution.

## 3. RESULTS

## 3.1. Overall Model Performance

Table 1 summarizes the main cross-validation results. The clinical-only baseline without ADC-derived lesion biomarkers achieved strong performance $( \mathrm { A U C } = 0 . 8 9 9 \pm 0 . 1 5 1$ , accuracy $= 0 . 8 4 5 \pm 0 . 1 6 1 , \mathrm { F } 1 = 0 . 8 2 0 \pm 0 . 1 4 5 )$ . MRI-only models showed substantially lower discrimination, with mean AUCs ranging from 0.675 to 0.719, indicating that image embeddings alone were insufficient for robust prediction in this cohort.

The best-performing configuration was FLAIR combined with clinical variables and ADC-derived lesion biomarkers, achieving $\mathrm { A U C } = 0 . 9 4 8 \pm 0 . 0 6 1$ , accuracy $= 0 . 8 0 2 \pm 0 . 0 9 3 .$ and $\mathrm { F } 1 = 0 . 7 3 2 \pm 0 . 1 7 5$ . This represented a numerical AUC improvement $\mathbf { o f + 0 . 0 4 9 }$ over the clinical-only reference. The paired fold-level AUC difference was significant according to an exact two-sided Wilcoxon signed-rank test $( p = 0 . 0 0 7 8 )$ . ADC-clinical fusion with ADC lesion biomarkers also performed strongly $( \mathrm { A U C } ~ = ~ 0 . 9 4 1 \pm 0 . 0 9 4 )$ , followed by ADC-FLAIR clinical fusion with lesion biomarkers (AUC $\mathbf { \Sigma } = \ 0 . 9 3 2 \pm 0 . 0 7 0 )$ In contrast, combining all three MRI modalities did not provide the highest performance, suggesting that full modality fusion may introduce redundant or noisy information in a small-sample setting.

## 3.2. Contribution of MRI Modalities and ADC Lesion Biomarkers

Among single-modality clinical-imaging fusion models, FLAIR and ADC provided the strongest complementary information. FLAIR-clinical fusion with ADC lesion biomarkers achieved the highest AUC, while ADC-clinical fusion achieved the highest F1 among the multimodal models. DWIclinical fusion was weaker $( \mathrm { A U C } = 0 . 8 9 3 \pm 0 . 1 0 0$ with ADC lesion biomarkers), and DWI-FLAIR fusion did not improve performance over the clinical baseline.

ADC-derived lesion biomarkers improved several MRIclinical configurations, particularly FLAIR-clinical fusion, which increased from $\mathit { \Delta A U C } = \ 0 . 8 8 9 \pm 0 . 1 1 5$ without lesion biomarkers to $0 . 9 4 8 \pm 0 . 0 6 1$ with lesion biomarkers. However, lesion biomarkers did not improve the clinical-only model, which decreased from $\mathrm { A U C } = 0 . 8 9 9 \pm 0 . 1 5 1$ without lesion biomarkers to $0 . 8 9 0 \pm 0 . 1 4 7$ with lesion biomarkers. This suggests that ADC-derived lesion burden was most useful when combined with learned MRI representations rather than as a standalone addition to clinical variables.

## 3.3. Explainability

Filtered coefficient analysis identified age, NIHSS-related variables, ADC lesion burden, and learned MRI features among the most influential predictors (Fig. 1). Positive coefficients indicated association with unfavorable outcome, while negative coefficients indicated association with favorable outcome. After excluding rare binary variables from visualization, the most prominent contributors included age, day-1 NIHSS subscores, ADC lesion burden features, and several learned FLAIR-derived imaging features.

Qualitative activation maps showed spatial responses overlapping visible FLAIR hyperintensities in the illustrated cases. The selected feature contributes to the FLAIR representation used by the final classifier, and its activation map indicates the image regions associated with that feature’s response. However, these exploratory visualizations should not be interpreted as lesion segmentations or as evidence of a causal relationship between the highlighted regions and outcome.

![](images/bfbaff851ef87278dce1f97127ab67944530bf4858d150d36c27bdb035acb0fe.jpg)  
Fig. 1. Filtered global feature importance from the direct linear probe. Bars show average SVM coefficients across folds after excluding rare binary variables from visualization. Positive coefficients indicate association with unfavorable outcome (mRS > 1), while negative coefficients indicate association with favorable outcome. Age, NIHSS-related variables, ADC lesion burden, and learned MRI features contributed to the prediction.

## 4. DISCUSSION

This study evaluated whether day-1 multimodal MRI adds prognostic information beyond structured clinical variables for predicting three-month functional outcome after AIS. FLAIR combined with clinical variables and ADC-derived lesion biomarkers achieved the highest cross-validated AUC, whereas the clinical-only model achieved the highest accuracy and F1-score. AUC was used as the primary ranking metric because it evaluates discrimination across decision thresholds, while accuracy and F1 depend on a single operating threshold. Thus, multimodal fusion improved overall patient ranking, although this advantage did not translate into better threshold-dependent classification.

The strong clinical-only performance is consistent with evidence that age, pre-stroke disability, and NIHSS are major predictors of post-stroke recovery [3, 4]. MRI-only models performed substantially worse, suggesting that image embeddings alone were insufficient in this small cohort. Nevertheless, FLAIR- and ADC-clinical fusion achieved higher AUCs when combined with clinical information and ADC lesion biomarkers, indicating that day-1 imaging provides complementary tissue-level information in the clinical context.

Full ADC-DWI-FLAIR fusion did not outperform the best single-modality clinical fusion model. This may reflect redundancy between modalities, limited sample size, and the introduction of noisy features, emphasizing the importance of controlled modality-combination experiments.

![](images/927e45e8a481cf3b7502e809d3e31ccd2f2c987b0994f2ec8e215791de69312b.jpg)  
Fig. 2. Qualitative FLAIR CNN activation visualization for representative patients. Each row shows the raw FLAIR slice, the CNN activation map, and the activation overlaid on FLAIR. The selected activation channel responds over lesionlike FLAIR hyperintense regions. These maps are qualitative activation visualizations and should not be interpreted as lesion segmentation or causal attribution.

Explainability analysis identified age, NIHSS-related variables, ADC lesion burden, and learned MRI features among the most influential predictors. The selected FLAIR channel responded spatially to visible hyperintense abnormalities in the illustrated cases. These visualizations remain exploratory and should not be interpreted as lesion segmentation or causal attribution.

This study is limited by its small dataset size and potential scanner and acquisition heterogeneity. External validation is therefore required. Future work will evaluate skull-stripped retraining and larger multicenter cohorts.

## 5. CONCLUSION

Day-1 multimodal MRI fusion improved score-based discrimination for three-month functional outcome prediction after AIS, with FLAIR combined with clinical variables and ADC-derived lesion biomarkers achieving the highest crossvalidated AUC. However, clinical-only modeling remained strongest for threshold-dependent metrics, emphasizing the central role of structured clinical assessment. These findings suggest that FLAIR representations and ADC lesion biomarkers provide complementary prognostic information beyond clinical variables, but require external validation and prospective evaluation before clinical translation.

## Compliance with Ethical Standards

This retrospective study used anonymized clinical data acquired as part of routine care. According to the institutional guidelines of the University of Geneva and the University Hospital of Geneva, the study is covered under existing retrospective data-use approvals, and no additional protocolspecific ethics approval was required. The study was conducted in accordance with the principles of the Declaration of Helsinki.

## Reproducibility and Acknowledgments

This work was supported by the FORDEMS Foundation. The authors declare no competing interests. Data and implementation details are available upon request.

## 6. REFERENCES

[1] V. L. Feigin et al., “Global, regional, and national burden of stroke and its risk factors, 1990–2019: a systematic analysis for the global burden of disease study 2019,” Lancet Neurol., vol. 20, no. 10, pp. 795–820, 2021.

[2] J. L. Banks and C. A. Marotta, “Outcomes validity and reliability of the modified rankin scale: implications for stroke clinical trials: a literature review and synthesis,” Stroke, vol. 38, no. 3, pp. 1091–1096, 2007.

[3] A. Wouters, C. Nysten, V. Thijs, and R. Lemmens, “Prediction of outcome in patients with acute ischemic stroke based on initial severity and improvement in the first 24 h,” Frontiers in Neurology, vol. 9, pp. 308, 2018.

[4] W. D. Heiss, “Contribution of neuro-imaging for prediction of functional recovery after ischemic stroke,” Cerebrovascular Diseases, vol. 44, no. 5-6, pp. 266–276, 2017.

[5] S. Warach, D. Chien, W. Li, M. Ronthal, and R. R. Edelman, “Fast magnetic resonance diffusion-weighted imaging of acute human stroke,” Neurology, vol. 42, no. 9, pp. 1717–1723, 1992.

[6] A. Nielsen, M. B. Hansen, A. Tietze, and K. Mouridsen, “Prediction of tissue outcome and assessment of treatment effect in acute ischemic stroke using deep learning,” Stroke, vol. 49, no. 6, pp. 1394–1401, 2018.

[7] D. Y. Kim et al., “Deep learning-based personalised outcome prediction after acute ischaemic stroke,” Journal of Neurology, Neurosurgery & Psychiatry, vol. 94, no. 5, pp. 369–378, 2023.

[8] D. Ma, M. Wang, A. Xiang, Z. Qi, and Q. Yang, “Transformer-based classification outcome prediction for multimodal stroke treatment,” in 2024 IEEE 2nd International Conference on Sensors, Electronics and Computer Engineering (ICSECE). 2024, pp. 383–386, IEEE.

[9] Y. LeCun, Y. Bengio, and G. Hinton, “Deep learning,” Nature, vol. 521, no. 7553, pp. 436–444, 2015.

[10] S. Chen, K. Ma, and Y. Zheng, “Med3d: Transfer learning for 3d medical image analysis,” arXiv preprint arXiv:1904.00625, 2019.

[11] M. J. Cardoso et al., “Monai: An open-source framework for deep learning in healthcare,” arXiv preprint arXiv:2211.02701, 2022.

[12] S. Wold, K. Esbensen, and P. Geladi, “Principal component analysis,” Chemometrics and Intelligent Laboratory Systems, vol. 2, no. 1-3, pp. 37–52, 1987.

[13] C. Cortes and V. Vapnik, “Support-vector networks,” Machine Learning, vol. 20, no. 3, pp. 273–297, 1995.

[14] F. Pedregosa et al., “Scikit-learn: Machine learning in python,” The Journal of Machine Learning Research, vol. 12, pp. 2825–2830, 2011.

[15] S. Raeisadigh, M. J. T. Tan, H. Muller, and A. Hed-¨ joudje, “Comparing baseline and day-1 diffusion MRI using multimodal deep embeddings for stroke outcome prediction,” in 2026 IEEE 23rd International Symposium on Biomedical Imaging (ISBI). 2026, pp. 1–5, IEEE.