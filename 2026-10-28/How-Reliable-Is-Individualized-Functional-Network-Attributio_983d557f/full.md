# How Reliable Is Individualized Functional-Network Attribution in Resting-State fMRI Cognitive Prediction in ADNI? A Scan–Rescan Evaluation

Sheethal Bhat<sup>1</sup>, Mahfuzur Rahman Chowdhury<sup>1</sup>, Imam Ahmed Shaik<sup>2</sup>, Rose Dawn Bharat<sup>1</sup>, Siming Bayer<sup>1</sup>, Andreas Maier<sup>1</sup>

<sup>1</sup>Pattern Recognition Lab, Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU), Martensstr. 3, 91058 Erlangen, Germany <sup>2</sup>Siemens Healthineers, Henkestr. 127, 91052 Erlangen, Germany sheethal.bhat@fau.de

Abstract. Individualized attribution estimates which functional networks drive a model prediction for a given subject and is often proposed as an interpretable resting-state fMRI (rs-fMRI) biomarker, but its test–retest reliability is rarely quantified. We evaluate per-subject network-ablation attribution for a frozen resting-state functional connectivity (FC) predictor (Schaefer-200 FC predicting MMSE/ADAS-Cog13) trained on �=875 subjects from the Alzheimer’s Disease Neuroimaging Initiative (ADNI), using (i) within-session split-half, (ii) scan–rescan repeat visits (74 subjects, 205 baseline–repeat pairs; median inter-visit interval 24.4 months), and (iii) a subject-level permutation null. Split-half reliability is near-ceiling (Spearman �=0.948/0.950) but drops on scan–rescan (�=0.609/0.616). Under the mismatched-subject null, correlations remain substantial (�≈0.46–0.47), indicating strong population-shared structure, yet matched pairs exceed this null $( z { \approx } 6 . 6 { - } 6 . 9 , p \leq 1 0 ^ { - 4 } )$ . Thus, in ADNI, individualized network attribution contains a modest but non-zero person-specific component atop a strong population-shared pattern. We recommend reporting scan–rescan reliability together with a permutation-null control, rather than within-session proxies alone, to avoid overstating individual specificity.

## 1 Introduction

Models that map resting-state functional connectivity (FC) to cognition and clinical phenotypes are increasingly used in Alzheimer’s disease (AD) neuroimaging [1, 2]. Beyond group-level accuracy, a common objective is individualized attribution: for a given subject, estimating which functional networks drive that subject’s prediction. Such per-subject attribution is only meaningful if it is reproducible.

Reliability is a central constraint. Large-scale analyses suggest that stable brain– behavior efects often require much larger samples than historically used [3], and

FC test–retest reliability is frequently modest (pooled mean edge-level Intraclass Correlation Coeficient, $\mathrm { I C C } \approx 0 . 2 9 )$ [4]. The ICC describes how reproducibly a single connectivity edge is measured. Prior work on subject-specific parcellations reports reproducibility across short-interval repeat sessions [5, 6], but does not quantify test–retest reliability of model-derived attribution signatures over the multiyear intervals typical of aging cohorts.

An additional aspect is attribution stability: feature-importance maps can vary substantially under resampling, noise, and other perturbations even when predictive accuracy is similar [7, 8]. In FC prediction, however, attribution work has largely emphasized group-averaged importance. For connectomes rather than attributions, differential identifiability formalizes the person-specific, test–retest-identifiable component [9], and permutation-null models are standard for separating population-shared from person-specific structure [10, 11]. Neither has been applied to per-subject attribution signatures. To the best of our knowledge, no prior work combines persubject attribution of an FC-based cognitive predictor with a quantitative test–retest evaluation on independent repeat scans of the same individuals.

We address this gap using a frozen FC-based multilayer perceptron (FC-MLP) trained to jointly predict MMSE and ADAS-Cog13 from Schaefer-200 FC on �=875 ADNI subjects (Sect. 2). Instead of edge-level ICC, we assess test–retest consistency of each subject’s network-attribution ranking. We compute per-subject networkablation signatures and evaluate reproducibility with three checks: within-session split-half, scan–rescan across independent repeat visits, and a cluster-aware subjectlevel permutation null. The main contributions in this study are:

1. Quantify how split-half overestimates scan–rescan reliability of per-subject network attribution (74 subjects; 205 session pairs).

2. Use a subject-level permutation null to estimate the mismatched-subject baseline and the residual person-specific component.

3. Provide reporting recommendations to avoid overstating individual specificity.

## 2 Methods

To attribute a prediction to functional networks for a single subject, let � denote the frozen FC-MLP mapping a subject’s connectivity vector � to a two-dimensional prediction ${ \hat { \mathbf { y } } } = f ( x )$ with targets � ∈ {MMSE, ADAS-Cog13}. For a given target �, we write the scalar prediction as $\hat { y } ^ { ( t ) } = [ f ( x ) ] _ { t }$ .

For each of the seven Yeo networks [12] $ { n } \in  { \mathrm { ~  ~ \{ V i s  ~ } ~ }    \{   \mathrm $ , SomMot, DorsAttn, SalVentAttn, Limbic, Cont, Default} (see Fig. 1), we ablate the input by zeroing all FC edges incident to nodes in �. Because the model input concatenates an FC vector and an edge-coverage mask, $x = \mathrm { [ F C ; m a s k ] }$ , we apply the ablation to both operands: the ablated edges are set to zero in the FC vector and their corresponding mask entries are also set to zero. This yields $x _ { \backslash n }$ and $\hat { y } _ { n } ^ { ( t ) } = [ f ( x _ { \backslash n } ) ]$ <sub>�</sub> . The per-target network importance is then $I _ { n } ^ { ( t ) } = \hat { y } ^ { ( t ) } - \hat { y } _ { n } ^ { ( t ) }$ , and the subject-specific importance signature for target � is $\mathbf { I } ^ { ( t ) } = ( I _ { 1 } ^ { ( t ) } , \ldots , I _ { 7 } ^ { ( t ) } )$

![](images/efe4a86e21c19c5b8279ec17d8b59854b590741ab3e6270d1baea6f66609eacb.jpg)  
Fig. 1. Top 6 of the 7 Yeo networks by group-level importance, visualized on anatomical MRI. Group-level importance is quantified by the drop in test-set explained variance when ablating each network, $\bar { \Delta } R ^ { 2 } = R ^ { 2 } - \bar { R } _ { \backslash n } ^ { 2 } ,$ computed on the held-out test set and averaged over three independently trained runs (shown separately for MMSE and ADAS-Cog13). $\Delta R ^ { 2 }$ can exceed the total $R ^ { 2 }$ when $R _ { \backslash n } ^ { 2 } < 0 .$ . The displayed “combined MMSE+ADAS-Cog13 rank” orders networks by the summed $\Delta R ^ { 2 }$ across the two targets. The 7th (lowest-ranked, Somatomotor) network is omitted for layout only; all 7 are used in every analysis.

This implementation reproduces the group-level ablation results (Fig. 1) for all $7 \times 2$ network–target combinations to six decimal places. Attribution reliability between two occasions $a , b$ for the same subject is the Spearman rank correlation $\rho ( \mathbf { I } _ { a } , \mathbf { I } _ { b } )$ , i.e., the agreement in the within-subject network ranking.

Fig. 2 summarizes three reliability checks. (i) Split-half: for each test subject (�=132), we split the denoised rs-fMRI time series after scrubbing into odd/even surviving volumes, recompute FC and attribution per half, and compute $\rho ( \mathbf { I } _ { \mathrm { h a l f - A } } , \mathbf { I } _ { \mathrm { h a l f - B } } )$ (minimum 30 volumes per half; 0 exclusions). We additionally test whether split-half $\rho$ increases with per-half scan duration. (ii) Scan–rescan: we attribute independent repeat-visit rs-fMRI and compute $\rho ( { \bf I } _ { \mathrm { b a s e l i n e } } , { \bf I } _ { \mathrm { r e p e a t } } )$ for baseline × all follow-ups, yielding 205 pairs from 74 subjects with uneven repeats $( k _ { i } \in [ 1 , 1 2 ]$ ; 30 subjects contribute 1 pair, 18 contribute 2, 13 contribute 3; tail to 12). (iii) Permutation null: for $B { = } 1 0 { , } 0 0 0$ permutations, we draw a derangement � of the 74 subjects, pair each subject �’s baseline signature with the repeat signatures of subject $\pi ( i )$ (repeats kept as a block), and recompute the subject-level mean reliability. With $T _ { \mathrm { o b s } }$ the observed subject-level mean and � the number of permutations with statistic $\geq T _ { \mathrm { o b s } }$ , we report $p = ( b + 1 ) / ( B + 1 )$ and $z = ( T _ { \mathrm { o b s } } - \mu _ { \mathrm { n u l l } } ) / \sigma _ { \mathrm { n u l l } }$ [11].

![](images/97bbaefcbaf61590b2b16aaf4daa9902fa25d5034c2397ddfbb0805d23f68e3b.jpg)  
Fig. 2. Overview of the three progressively stricter reliability checks. (a) Within-session splithalf: odd and even volumes of one session are each attributed and compared by Spearman $\rho .$ (b) Scan–rescan: each subject’s baseline is compared with all repeat visits (median 24.4 months apart). (c) Permutation null: 10,000 subject-level derangements match each baseline with another subject’s repeat visits (kept as a block, no self-match); the observed mean $T _ { \mathrm { o b s } }$ is compared with this null.

Artifact checks. Edge-count: let $c \in \mathbb { R } ^ { 7 }$ be the fixed vector of ablated edge counts (unique edges zeroed when ablating each network). For each subject we compute the across-network Spearman correlation $\rho ( \mathbf { I } ^ { ( t ) } , c )$ (7 items) and summarize it across subjects by its mean, separately per target; with only 7 items, Spearman $\rho$ is highly discrete, so this check is treated as descriptive. For scan–rescan, we first average each subject’s per-pair $\rho ( \mathbf { I } ^ { ( t ) } , c )$ across that subject’s repeat visits and then compute the across-subject mean. Scan duration: we correlate (Spearman, across the 132 test subjects) each subject’s per-half scan duration (seconds; half A, equivalent to half B up to one volume) with split-half $\rho .$

Aggregation and inference. All Spearman correlations are computed separately within each of the three trained runs and averaged across runs. Spearman $\rho$ uses average ranks for ties; no ties occurred in any importance signature (verified for all 1,674 scan–rescan signatures, i.e., 279 sessions × 2 targets × 3 runs, and all split-half signatures). As subjects contribute unequal numbers of repeat pairs, scan– rescan is summarized at the subject level by first averaging each subject’s per-pair $\rho$ values and then summarizing across subjects; we also report the pair-pooled mean for comparability. Uncertainty on headline scan–rescan and split-half means is a 95 % percentile bootstrap CI over 10,000 resamples of subjects (resampling whole subjects). We test the split-half versus scan–rescan gap with a Wilcoxon signed-rank test across the 74 subjects present in both analyses.

Tab. 1. Attribution reliability (Spearman � ofthe per-subject 7-network importance signature). The subject is the unit of analysis (split-half: $n { = } 1 3 2 ;$ scan–rescan: $n { = } 7 4$ , with pairs averaged within subject). CIs are subject-level bootstrap intervals; “run SD” is the SD across 3 runs; “–” indicates not computed. Permutation-null rows use 10,000 subject-level derangements.
<table><tr><td colspan="3">Split-half</td><td colspan="2">Scan-rescan</td></tr><tr><td>Statistic</td><td>MMSE</td><td>ADAS-Cog13</td><td>MMSE</td><td>ADAS-Cog13</td></tr><tr><td>Mean</td><td>0.948</td><td>0.950</td><td>0.609</td><td>0.616</td></tr><tr><td>95 % CI</td><td>0.929–0.964</td><td>0.934–0.964</td><td>0.553-0.663</td><td>0.571-0.660</td></tr><tr><td>Median</td><td>0.988</td><td>0.988</td><td>0.625</td><td>0.661</td></tr><tr><td>Std</td><td>0.101</td><td>0.085</td><td>0.242</td><td>0.198</td></tr><tr><td>10–90%</td><td>0.87-1.00</td><td>0.85-1.00</td><td>0.34–0.90</td><td>0.33-0.85</td></tr><tr><td>Run SD (n=3)</td><td>0.005</td><td>0.007</td><td>0.033</td><td>0.020</td></tr><tr><td>Null mean ± SD</td><td></td><td></td><td>0.463 ± 0.021</td><td>0.472 ± 0.022</td></tr><tr><td>Null range</td><td>一</td><td></td><td>0.379–0.549</td><td>0.393-0.562</td></tr><tr><td>z-score</td><td>一</td><td></td><td>6.89</td><td>6.63</td></tr><tr><td>p-value</td><td>1</td><td></td><td> $\leq 1 0 ^ { - 4 }$ </td><td> $\leq 1 0 ^ { - 4 }$ </td></tr><tr><td>Pair-pooled mean</td><td>一</td><td></td><td>0.593</td><td>0.588</td></tr></table>

## 3 Experimental details

## 3.1 Data description

ADNI is a multi-site, longitudinal study of aging and Alzheimer’s disease with multimodal neuroimaging [13]. We analyzed data from �=875 unique ADNI participants who had both T1-weighted MRI and rs-fMRI scans. Within the �=875 cohort, diagnoses are CN (�=464), MCI (�=315), Dementia (�=94), and unknown (�=2). Age is 72.6 ± 7.9 years (range 50–97); 489 subjects are female and 386 male. Subjects were enrolled in ADNI1 (�=37), ADNI-GO (�=38), ADNI2 (�=256), ADNI3 (�=474), and ADNI4 (�=68); 2 unknown. rs-fMRI scans were acquired at 3.0 T across 53 sites with heterogeneous protocols (TR 0.607–6.02 s; 96–972 volumes after removal of initial volumes). For the upstream prediction model, the cohort is split once at the subject level into fixed train/validation/test partitions of 611/132/132 stratified by diagnosis. We evaluate attribution and reliability only on the held-out 132-subject test split; scan–rescan further restricts to test subjects with repeat-visit rs-fMRI.

## 3.2 Implementation details

We analyze a trained FC-MLP and keep its weights fixed for all attribution and reliability analyses.

Input features. rs-fMRI preprocessing includes motion correction, HD-BET brain extraction [14], rigid rs-fMRI→T1 registration, detrending, and nuisance regression of mean white-matter (WM) and cerebrospinal-fluid (CSF) signals plus 5 anatomical CompCor (aCompCor) components [15] from a combined WM/CSF mask, without global signal regression. Time series are band-pass filtered (0.01–0.1 Hz, using each subject’s TR), �-scored, and scrubbed at framewise displacement $\mathrm { ( F D ) } > 0 . 5$ mm after filtering; FD was computed from translation magnitude only rather than the full six-parameter formulation [16], as rotation parameters were not retained upstream. No spatial smoothing or slice-timing correction is applied. We compute Schaefer-200 [17] FC as a Fisher-� transformed $2 0 0 \times 2 0 0$ Pearson correlation matrix and vectorize the upper triangle (19,900 edges). An edge-coverage mask is concatenated, yielding a 39,800-dimensional input.

Model and training (frozen). The FC-MLP encoder maps $3 9 , 8 0 0 \to 2 5 6 \to 1 2 8$ with LayerNorm/GELU/dropout (0.3), and the decoder maps $1 2 8 \to 6 4 \to 2$ with GELU/dropout (0.3). The model is trained with Huber loss (�=1.0) using Adam (lr = $1 0 ^ { - 4 }$ , weight decay $1 0 ^ { - 5 } ;$ no schedule/warmup), batch size 16, and up to 100 epochs. Early stopping uses patience 15 on validation loss; we keep the single best checkpoint (lowest validation loss). During training only, an augmentation zeros 5 randomly chosen parcels per sample. Targets are �-scored on train only and inverse-transformed for reporting. Three independent runs with diferent random initializations share the same fixed split. ComBat harmonization [18] uses the TR group as batch variable (a proxy for acquisition protocol) and protects diagnosis, whose composition difers across TR groups $( \chi ^ { 2 } = 3 5 . 6 6 , p < 0 . 0 0 1 )$ ; no other covariates are included. It is fit on the training split only, never refit, and applied to validation/test subjects and split-half halves, but not to repeat-visit sessions (both sides of scan–rescan use pre-ComBat features).

## 3.3 Results

The frozen predictor is weak: on the �=113 test subjects with both targets, $R ^ { 2 } { = } 0 . 0 5 3 { \pm }$ 0.034 for MMSE and $0 . 0 8 7 \pm 0 . 0 3 7$ for ADAS-Cog13 (mean ± SD over runs). As a diagnostic check, the predicted ADAS-Cog13 score separates CN from Dementia with AUC 0.780 (95 % bootstrap CI 0.611–0.911; $n = 7 2$ of the 113 subjects with both targets: 58 CN, 14 Dementia).

Within-session split-half reliability is near-ceiling for both targets (mean $\rho { = } 0 . 9 4 8 / 0 . 9 5 0 .$ , median ≈ 0.99; Tab. 1). On independent repeat visits (205 session pairs from 74 subjects), subject-level scan–rescan reliability drops to $\scriptstyle { \rho = 0 . 6 0 9 }$ (MMSE) and 0.616 (ADAS-Cog13), with large between-subject variation (10th–90th percentiles 0.34–0.90 and 0.33–0.85). Pair-pooled means are slightly lower (0.593/0.588) because subjects with more follow-ups receive more weight. Restricted to the same 74 subjects, split-half reliability is 0.933/0.939, a paired difference of 0.324/0.323 (Wilcoxon $p < 1 0 ^ { - 1 2 } )$ ), indicating that split-half substantially overestimates test–retest reliability.

Under the subject-level permutation null, mismatched subjects still correlate strongly $\scriptstyle ( \rho = 0 . 4 6 3 / 0 . 4 7 2 )$ , but the matched statistic exceeds the null $\scriptstyle ( z = 6 . 8 9 / 6 . 6 3 ;$ $p \leq 1 0 ^ { - 4 } ,$ ), corresponding to a modest person-specific excess of ≈ 0.15 (MMSE) and ≈ 0.14 (ADAS-Cog13). The artifact checks do not indicate a network-size efect: mean edge-count correlations $\rho ( \mathbf { I } ^ { ( t ) } , c )$ are weakly negative for split-half (MMSE −0.120; ADAS-Cog13 −0.163) and repeat visits (MMSE −0.112; ADAS-Cog13 −0.158). Split-half reliability increases with per-half scan duration (�=0.199/0.277), consistent with sampling-noise limits, although duration is confounded with acquisition protocol in ADNI.

## 4 Discussion

Per-subject attribution reliability depends strongly on protocol: split-half is nearceiling (�≈0.95), but drops on scan–rescan across independent repeat visits $\scriptstyle ( \rho \approx 0 . 6 1 -$ 0.62), consistent with additional between-visit variability beyond within-session sampling noise. The permutation null suggests that much of this structure is population-shared, with only a modest person-specific excess; robustness to other cohorts, preprocessing, parcellations, attribution operators, and models remains untested.

The coexistence of shared and person-specific structure mirrors FC itself, where connectomes are identifiable across sessions [9, 10] yet test–retest reliability is modest [4]. The strong shared profile likely has two sources. First, the ablations overlap by construction: every between-network edge (�, �) with $i \in n , j \in m$ is zeroed in both $x _ { \backslash n }$ and $x _ { \backslash m } ,$ coupling $I _ { n } ^ { ( t ) }$ and $I _ { m } ^ { ( t ) }$ ; within-network-only ablation would remove this overlap. Second, the weak predictor may respond mainly to group-level FC structure. Because $\rho$ is bounded above, a high shared baseline leaves little room for an individual excess, so the person-specific component may be underestimated.

Limitations include the modest repeat-visit sample (74 subjects; 205 session pairs) and the weak, run-variable predictor (Sect. 3). Repeat-visit sessions were not ComBat-harmonized, so part of the split-half to scan–rescan gap may reflect this mismatch. Inter-visit intervals are long and variable (median 24.4 months, IQR 12.2–48.4) and are not modeled.

We therefore recommend treating within-session proxies as expected upper bounds, reporting scan–rescan estimates when repeat visits exist, and including a permutation-null (or equivalent) control; these recommendations should be validated in further aging/AD cohorts such as OASIS-3 and AIBL. In summary, per-subject network attributions are far less reproducible across visits than within-session estimates suggest, and most of their apparent stability reflects a pattern shared across the cohort, so individual specificity should be demonstrated against an explicit population baseline before attributions are interpreted as personal biomarkers.

Acknowledgement. We acknowledge HPC support from XXXXXX and partial hardware funding from the XXXXX. Data were obtained from the Alzheimer’s Disease Neuroimaging Initiative (ADNI; adni.loni.usc.edu). As such, the investigators within the ADNI contributed to the design of ADNI but did not participate in this report. Code, derived features, and the subject-level split file are available upon reasonable request, subject to ADNI DUA constraints.

## References

1. Dadi K, Rahim M, Abraham A et al. Benchmarking functional connectome-based predictive models for resting-state fMRI. NeuroImage. 2019;192:115–34.

2. Svaldi DO, Joshi AA, Shattuck DW et al. Optimizing diferential identifiability improves connectome predictive modeling of cognitive deficits from functional connectivity in Alzheimer’s disease. Human Brain Mapping. 2021;42(11):3500–16.

3. Marek S, Tervo-Clemmens B, Calabro FJ et al. Reproducible brain-wide association studies require thousands of individuals. Nature. 2022;603:654–60.

4. Noble S, Scheinost D, Constable RT. A decade of test-retest reliability of functional connectivity: A systematic review and meta-analysis. NeuroImage. 2019;203:116157.

5. Wang D, Buckner RL, Fox MD et al. Parcellating cortical functional networks in individuals. Nature Neuroscience. 2015;18:1853–60.

6. Kong R, Li J, Orban C et al. Spatial topography of individual-specific cortical networks predicts human cognition, personality, and emotion. Cerebral Cortex. 2019;29(6):2533– 51.

7. Adebayo J, Gilmer J, Muelly M, Goodfellow I, Hardt M, Kim B. Sanity Checks for Saliency Maps. Advances in Neural Information Processing Systems (NeurIPS). 2018:9525–36.

8. Ghorbani A, Abid A, Zou J. Interpretation of Neural Networks Is Fragile. Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 33. (01). 2019:3681–8.

9. Amico E, Goñi J. The quest for identifiability in human functional connectomes. Scientific Reports. 2018;8(1):8254.

10. Finn ES, Shen X, Scheinost D et al. Functional connectome fingerprinting: identifying individuals using patterns of brain connectivity. Nature Neuroscience. 2015;18:1664–71.

11. Nichols TE, Holmes AP. Nonparametric permutation tests for functional neuroimaging: a primer with examples. Human Brain Mapping. 2002;15(1):1–25.

12. Yeo BTT, Krienen FM, Sepulcre J, Sabuncu MR, Lashkari D, Hollinshead M et al. The organization of the human cerebral cortex estimated by intrinsic functional connectivity. Journal of Neurophysiology. 2011;106(3):1125–65.

13. Jack CR, Bernstein MA, Fox NC, Thompson P, Alexander G, Harvey D et al. The Alzheimer’s Disease Neuroimaging Initiative (ADNI): MRI methods. Journal of Magnetic Resonance Imaging. 2008;27(4):685–91.

14. Isensee F, Schell M, Pflueger I, Brugnara G, Bonekamp D, Neuberger U et al. Automated brain extraction of multisequence MRI using artificial neural networks. Human Brain Mapping. 2019;40(17):4952–64.

15. Behzadi Y, Restom K, Liau J, Liu TT. A component based noise correction method (CompCor) for BOLD and perfusion based fMRI. NeuroImage. 2007;37(1):90–101.

16. Power JD, Barnes KA, Snyder AZ, Schlaggar BL, Petersen SE. Spurious but systematic correlations in functional connectivity MRI networks arise from subject motion. NeuroImage. 2012;59(3):2142–54.

17. Schaefer A, Kong R, Gordon EM, Laumann TO, Zuo XN, Holmes AJ et al. Local-Global Parcellation of the Human Cerebral Cortex from Intrinsic Functional Connectivity MRI. Cerebral Cortex. 2018;28(9):3095–114.

18. Fortin JP, Parker D, Tunç B, Watanabe T, Elliott MA, Ruparel K et al. Harmonization of multi-site difusion tensor imaging data. NeuroImage. 2017;161:149–70.