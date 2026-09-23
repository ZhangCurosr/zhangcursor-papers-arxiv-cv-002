# Observer Choice and Threshold Selection in Retinal Vessel Segmentation: A Subject-Separated Evaluation

Wenhao Xu<sup>1</sup>, Yixian Kong<sup>2</sup>, Ting Pan<sup>2</sup>, Changwei Wang<sup>4</sup>, Feilong Wang<sup>2</sup>, Rongtao Xu<sup>3,∗</sup>

<sup>1</sup>Zhengzhou Police University, Zhengzhou, China

<sup>2</sup>School of Artificial Intelligence, Beijing University of Posts and Telecommunications, Beijing, China

<sup>3</sup>Institute of Automation, Chinese Academy of Sciences, Beijing, China

<sup>4</sup>Qilu University ofTechnology and Jinan Supercomputing Center, Jinan, China changweiwang@sdas.org; \*Corresponding author: xurongtao2022@gmail.com

Keywords: Retinal Vessel Segmentation, Multiple Annotations, Threshold Selection, Subject-Level Evaluation.

Abstract:

The annotation used to select a segmentation threshold is part of the evaluation protocol, yet its effect is easily conflated with model quality. We examine this choice for retinal vessel segmentation using all 28 CHASE\_DB1 images and both human annotations. A fixed seven-fold protocol keeps both eyes of each of the 14 subjects together. Random forests and Extra Trees are fitted against observer 1 with three random seeds, yielding 42 fits. Five threshold policies share identical score maps: fixed 0.50, observer-1 tuning, observer-2 tuning, mean-observer tuning, and maximin tuning of the per-image lower observer Dice. For random forests, maximin changes the threshold in 19 of 21 fits, but worst-observer Dice decreases from 70.53% to 70.45%. The paired difference is -0.073 percentage points, with a conditional subject-bootstrap 95% interval of [-0.384, 0.238]. Extra Trees shows the same direction. Identical observer-1-tuned random-forest masks score 73.66% agains observer 1 and 71.06% against observer 2. The results support explicit reporting of both the threshold-selection reference and evaluation reference; they do not support an accuracy benefit from maximin tuning in this cohort. All splits, raw predictions, metrics and code are supplied. AI assistance is disclosed (OpenAI, 2026).

## 1 INTRODUCTION

A retinal vessel segmentation score measures agreement with a particular reference annotation. It does not establish agreement with every plausible tracing of the same image. Public datasets such as DRIVE, STARE and CHASE\_DB1 have made vessel extraction reproducible (Staal et al., 2004; Hoover et al., 2000; Fraz et al., 2012), but the existence of two human annotations creates an additional evaluation choice: which observer supplies the validation objective, and which supplies the reported test score?

This choice matters even when the segmentation model is held fixed. A model usually produces a continuous vessel score, while overlap measures require a binary mask. Selecting a threshold against one observer can favor that observer’s annotation convention. Evaluating only against the same convention leaves its transfer to the other observer unmeasured. Reporting two test scores is useful, but a complete description also needs the annotation used for threshold selection. The objective used to select an operating point and the reference used to evaluate it are separate parts of the experiment.

We study this issue on CHASE\_DB1 using both eyes of all 14 subjects and both supplied vessel annotations. The experiment compares five global threshold policies on identical out-of-fold score maps: a fixed threshold, selection against either observer individually, selection by their mean Dice, and selection by the mean per-image lower Dice. We refer to the last policy as maximin. The experiment is repeated with random forests and extremely randomized trees using three random seeds. All choices of features, splits, thresholds and primary outcome were fixed locally before fitting.

The contribution is a controlled evaluation of observer-dependent operating points. It comprises a subject-separated protocol, a paired comparison that isolates threshold selection from model fitting, and a complete record of every fit and held-out image. The classifiers, image features and scalar threshold search are established tools; they are not presented as a new vessel segmentation architecture. The experiment also does not assume that optimizing the lower validation score must improve its held-out counterpart. That distinction is central when only two subjects are available for threshold selection.

Drafting assistance: (OpenAI, 2026); see disclosure.

## 2 RELATED WORK

## 2.1 Retinal Vessel Segmentation

Classical vessel extraction uses image structure at several spatial scales. Matched-filter threshold probing (Hoover et al., 2000), ridge-based features (Staal et al., 2004), Hessian analysis (Frangi et al., 1998), and multiscale line responses (Nguyen et al., 2013) provide complementary ways to describe elongated structures. Fraz et al. combine vessel descriptors with bagged and boosted decision trees (Fraz et al., 2012). These works motivate a compact multiscale baseline, although the feature set and training procedure used here do not reproduce any of those complete systems.

Deep models address spatial context and vessel continuity more directly. U-Net introduced an encoder– decoder architecture with connections between corresponding resolutions (Ronneberger et al., 2015). DRIU specializes convolutional features for vessel and optic disc segmentation (Maninis et al., 2016). IterNet refines segmentations using repeated small U-Nets (Li et al., 2020), while DA-Net combines local and global information with adaptive strip upsampling (Wang et al., 2022). Study Group Learning addresses incomplete vessel labels through learned supervision (Zhou et al., 2021). These approaches concern representation or learning; the present comparison concerns the operating point of an already fitted model. Their reported scores are therefore not treated as comparable experimental baselines.

## 2.2 Multiple References and Evaluation

Several methods explicitly address disagreement between annotators. STAPLE estimates a probabilistic reference and the performance of its contributing segmentations (Warfield et al., 2004). The probabilistic U-Net models multiple plausible outputs (Kohl et al., 2018). The diagnosis-first framework of Wu et al. uses diagnostic performance to guide multi-observer label fusion (Wu et al., 2022). Our experiment retains both annotations as separate references and does not estimate which observer is more reliable.

Threshold optimization for F1 is already well studied (Lipton et al., 2014). Medical segmentation evaluation also requires attention to the choice and aggregation of metrics (Taha and Hanbury, 2015; Maier-Hein et al., 2024; Reinke et al., 2024). A recent retinal segmentation preprint considers how uncertainty estimates support deferral decisions (Maganti, 2026); here the decision is the binary vessel mask itself, with no deferral or clinician intervention. We use threshold selection rather than probability calibration: changing a decision threshold does not make a score a calibrated probability. The specific question is whether an observer-balanced validation objective transfers to unseen subjects under a fixed, reproducible segmentation pipeline.

Drafting assistance: (OpenAI, 2026); see disclosure.

## 3 METHODS

## 3.1 Score Maps and Image Features

Each native-resolution photograph is mapped to 29 pixel features. Five are the three RGB intensities divided by 255, a green-to-red ratio with a 1/255 denominator stabilizer, and a broad background-minus-green contrast with Gaussian scale 32 pixels. The green image is independently normalized using its first and 99th intensity percentiles and clipped to [0, 1]. Percentiles are calculated inside the largest filled connected component where the largest RGB channel exceeds 10 on the original 8-bit scale. This intensity-only region is used for normalization, never to mask the evaluation.

For each $\sigma \in \{ 1 , 2 , 4 , 8 \}$ pixels, six further features are calculated from normalized green: Gaussiansmoothed intensity, smoothed-minus-original contrast, σ-normalized gradient magnitude, the two algebraically ordered eigenvalues of the $\sigma ^ { 2 } .$ -normalized Hessian, and local standard deviation. Thus the feature count is $5 + 4 \times 6 = 2 9$ . Features retain the original 999×960 resolution. No image augmentation, learned enhancement, component removal or morphological postprocessing is applied.

The two estimators are a random forest (RF) (Breiman, 2001) and Extra Trees (ET) (Geurts et al., 2006), implemented in scikit-learn 1.8.0 (Pedregosa et al., 2011). Both use 64 trees, maximum depth 18, minimum leaf size 10 and square-root feature subsampling. RF uses bootstrap samples and ET uses the full sampled training set. For each training image, 2,000 observer-1 vessel pixels and 2,000 background pixels are sampled without replacement. Twenty training images yield 80,000 pixels per fit. Classifier scores are averages of leaf class proportions. Because training is class balanced, we do not interpret these scores as clinical probabilities.

## 3.2 Observer-Dependent Thresholds

Let $p _ { i } ( x )$ be the vessel score for pixel x of image i, and $y _ { i o }$ the annotation from observer $o \in \{ 1 , 2 \}$ . At threshold τ, the prediction is

$$
\hat { y } _ { i } ^ { \tau } ( x ) = \mathbf { 1 } \{ p _ { i } ( x ) \geq \tau \} .\tag{1}
$$

With $D _ { i o } ( \tau ) \ = \ \mathrm { D i c e } ( \hat { y } _ { i } ^ { \tau } , y _ { i o } )$ , the three validation objectives are

$$
J _ { o } ( \tau ) = \frac { 1 } { | \mathcal { V } | } \sum _ { i \in \mathcal { V } } D _ { i o } ( \tau ) ,\tag{2}
$$

$$
J _ { \mathrm { m e a n } } ( \tau ) = \frac { 1 } { | \mathcal { V } | } \sum _ { i \in \mathcal { V } } \frac { D _ { i 1 } ( \tau ) + D _ { i 2 } ( \tau ) } { 2 } ,\tag{3}
$$

$$
J _ { \operatorname* { m i n } } ( \tau ) = \frac { 1 } { | \mathcal { V } | } \sum _ { i \in \mathcal { V } } \operatorname* { m i n } _ { o \in \{ 1 , 2 \} } D _ { i o } ( \tau ) .\tag{4}
$$

The O1-tuned and O2-tuned policies maximize $J _ { 1 }$ and $J _ { 2 } .$ , respectively. Mean-tuned maximizes $J _ { \mathrm { { m e a n } } } .$ and maximin maximizes $J _ { \mathrm { m i n } }$ . Each searches the same 91 thresholds, $\mathcal { T } = \{ 0 . 0 5 , 0 . 0 6 , \hdots , 0 . 9 5 \}$ , choosing the smallest in an exact tie. Fixed 0.50 provides an untuned reference. One threshold is selected per fitted model and applied to every test image assigned to that model.

The minimum in $J _ { \mathrm { m i n } }$ is taken before averaging images. In general this differs from the smaller of the two observer-average Dice scores. It allows the limiting observer to vary across images. It also gives neither observer a privileged role in threshold selection, although observer 1 remains the sole training reference. The five policies use exactly the same feature maps and classifier outputs within each fit.

## 3.3 Outcomes and Aggregation

For each observer, Dice and IoU are computed from true positives (TP), false positives (FP) and false negatives (FN):

$$
\mathrm { D i c e } = { \frac { 2 \mathrm { T P } } { 2 \mathrm { T P } + \mathrm { F P } + \mathrm { F N } } } ,\tag{5}
$$

$$
\mathrm { I o U } = { \frac { \mathrm { T P } } { \mathrm { T P } + \mathrm { F P } + \mathrm { F N } } } .\tag{6}
$$

Precision and recall are also retained. The primary outcome W is the mean of the lower observer Dice for each image, first averaged over the two eyes and three seeds within each subject, and then over the 14 subjects. Averaging scores across seeds does not ensemble their predictions. The prespecified primary contrast is RF maximin minus RF O1-tuned in W; ET provides a second estimator comparison on the same cohort.

Average precision (AP) and ROC AUC are calculated once per held-out score map and observer, then averaged across images and seeds. AP complements ROC AUC in this imbalanced setting (Saito and Rehmsmeier, 2015). These measures cannot distinguish threshold policies sharing a score map. We additionally record the fraction of predicted vessel pixels in the disagreement region $y _ { i 1 } \triangle y _ { i 2 }$ , without interpreting agreement with either observer as an adjudicated truth. Drafting assistance: (OpenAI, 2026); see disclosure.

## 4 EXPERIMENTAL PROTOCOL

## 4.1 Data and Subject Separation

CHASE\_DB1 contains 28 photographs from both eyes of 14 children, with two independent manual vessel annotations per photograph (Fraz et al., 2012). We use the complete distributed set at 999 × 960 pixels and evaluate every image pixel. Images are not resized and no annotation-derived field-of-view mask is applied. These choices must accompany the scores: our results do not share the split or evaluation region of every published CHASE\_DB1 experiment.

The 14 subjects are permuted with NumPy random seed 20260909 and divided into seven groups of two. In fold k, group k is tested, group (k + 1) mod 7 selects the threshold, and the remaining ten subjects train the model. Each subject is tested once; both eyes always remain in the same partition. Figure 1 gives the actual allocation. The same partitions are used by all estimators and seeds.

## 4.2 Repetition, Uncertainty and Audit

Seeds 17, 41 and 83 vary pixel sampling and estimator randomness, producing $2 \times 3 \times 7 = 4 2$ fitted models and 168 held-out score maps. The protocol, split manifest, data manifest and experiment-source hashes were frozen locally before fitting. This is a documented analysis-plan freeze, not an externally registered study. Thresholds are written to a separate record before test annotations are used for scoring. No hyperparameters or comparison policies were revised after test prediction.

Paired percentile intervals use 10,000 bootstrap resamples of the 14 subject-level values, with seed 20260910. Eyes and technical seeds are averaged within a subject before resampling. The intervals condition on the stored out-of-fold predictions. They do not capture uncertainty from retraining on newly sampled cohorts, and overlapping training folds limit an independent-sample interpretation. We report no pixellevel significance tests and do not treat the three seeds as additional patients.

All raw confusion counts, continuous score maps, validation curves, selected thresholds, runtime records and environment versions accompany the source. Exact threshold equality, confusion counts, Dice and IoU were checked against independent scikit-learn metric calculations. Qualitative images 01L, 07L and 14L were specified before fitting; their RF seed-17 predictions are displayed with a common fixed crop. Drafting assistance: (OpenAI, 2026); see disclosure.

![](images/359e4ba3bf8349858d2086246b5b3a4587e7da6b0735f4fbc9d0ae641cf4b75e.jpg)  
Figure 1: The fixed seven-fold allocation. Each column represents a subject and both eyes. The four validation images select the threshold; the four test images are scored only after that choice is recorded.

## 5 RESULTS

## 5.1 Agreement and Primary Outcome

The mean human–human Dice is 77.65%. Observer 1 labels 6.93% of image pixels as vessels and observer 2 labels 6.64%. Their disagreement covers 3.05% of all pixels, or 36.47% of the vessel union when calculated per image and then averaged. Human agreement is descriptive; it is not used as an upper bound on algorithm performance.

Table 1 reports all policies. RF maximin obtains $W = 7 0 . 4 5 \%$ , compared with 70.53% for O1-tuned. The prespecified paired contrast is -0.073 percentage points (conditional 95% interval [-0.384, 0.238]). ET obtains 68.68% and 68.80%, respectively, with a contrast of -0.119 points [-0.445, 0.168]. Both intervals span zero. Maximin improves five subjects and reduces the score for nine subjects with either estimator (Figure 2). Its mean contrast is negative in all three seeds for both estimators. These observations do not support the hypothesized held-out improvement.

## 5.2 Threshold Changes and Reference Choice

Maximin selects a different threshold from O1-tuned in 19 of 21 RF fits and 16 of 21 ET fits. Per-fit thresholds and validation curves are provided in the research package. RF maximin thresholds range from 0.78 to 0.90; ET thresholds range from 0.71 to 0.85. Frequent changes in the operating point therefore do not imply a beneficial change in held-out overlap.

For RF, O1-tuned and maximin label 48.47% and 47.00% of disagreement-region pixels as vessels.

Against observer 1, precision changes from 72.23% to 73.09%, while recall changes from 76.27% to 75.14%. On average, maximin increases precision and reduces recall without improving the primary outcome. Figure 3 shows the predetermined cases, including missed small branches and irregular predictions near the bright optic disc. The visual examples illustrate the limited change between threshold policies rather than establish their relative performance.

Reference choice remains consequential after tuning. The same RF O1-tuned masks obtain Dice 73.66% against observer 1 and 71.06% against observer 2. ET obtains 71.96% and 69.33%. These are paired evaluations of identical predictions, not differences between separately trained observer-specific systems.

## 5.3 Untuned Scores and Computation

O1-tuned exceeds fixed 0.50 by 12.31 percentage points in RF W, and 10.77 points for ET. This is a conventional threshold-selection effect under balanced pixel sampling, not evidence for the maximin policy. AP is 0.8109 against observer 1 and 0.7740 against observer 2 for RF, versus 0.7937 and 0.7501 for ET. Corresponding ROC AUC values are 0.9764/0.9732 and 0.9742/0.9708. These threshold-independent results apply to every policy of the same estimator.

Mean fitting time is 4.44 s for RF and 0.66 s for ET. Prediction from cached features takes 0.79 s and 0.84 s per image, respectively, with eight CPU threads on an Intel Xeon Platinum 8370C. Feature extraction takes a further 2.80 s per image. These measurements exclude metric computation and writing the lossless score maps. The final audit recomputed all 840 policy– image records from the 168 stored maps, with zero discrepancy in confusion counts or their derived overlap, precision and recall values.

Drafting assistance: (OpenAI, 2026); see disclosure.

Table 1: Native-resolution, whole-image evaluation. Values are percentages, averaged over eyes and seeds within subjects. W averages the lower observer Dice per image; it is not the smaller of the two displayed Dice columns. Intervals condition on fitted predictions.
<table><tr><td>Estimator</td><td>Threshold policy</td><td>Dice O1</td><td>Dice O2</td><td>W</td><td>95% interval for W</td></tr><tr><td>RF</td><td>Fixed 0.50</td><td>61.94</td><td>59.87</td><td>58.21</td><td>55.61–60.61</td></tr><tr><td>RF</td><td>O1-tuned</td><td>73.66</td><td>71.06</td><td>70.53</td><td>69.15–71.95</td></tr><tr><td>RF</td><td>O2-tuned</td><td>73.62</td><td>71.04</td><td>70.55</td><td>69.31-71.85</td></tr><tr><td>RF</td><td>Mean-tuned</td><td>73.64</td><td>71.04</td><td>70.53</td><td>69.21-71.92</td></tr><tr><td>RF</td><td>Maximin</td><td>73.47</td><td>70.91</td><td>70.45</td><td>69.25–71.76</td></tr><tr><td>ET</td><td>Fixed 0.50</td><td>61.70</td><td>59.63</td><td>58.03</td><td>55.39–60.51</td></tr><tr><td>ET</td><td>O1-tuned</td><td>71.96</td><td>69.33</td><td>68.80</td><td>67.34–70.25</td></tr><tr><td>ET</td><td>O2-tuned</td><td>71.91</td><td>69.27</td><td>68.76</td><td>67.45-70.11</td></tr><tr><td>ET</td><td>Mean-tuned</td><td>71.97</td><td>69.33</td><td>68.83</td><td>67.45-70.21</td></tr><tr><td>ET</td><td>Maximin</td><td>71.79</td><td>69.15</td><td>68.68</td><td>67.28-70.10</td></tr></table>

![](images/2b1d7342af4c81635162e203539c1c51dd2bd9e87fe6314f9239ccccf3d81652.jpg)

![](images/38da5795a2e5fa82cf026b7f3444686fb9cd2b831039421a152d1d211f742c1a.jpg)  
Figure 2: Every subject’s change in worst-observer Dice, after averaging both eyes and three seeds. Positive values favor maximin; negative values favor O1-tuned. Subject IDs follow the public data filenames.

## 6 DISCUSSION

## 6.1 What the Comparison Establishes

The principal result is negative: the symmetric maximin validation objective does not improve its held-out counterpart in this experiment. The thresholds often change, but the resulting differences are small and unfavorable on average. A validation objective can increase by construction while its test counterpart decreases. With only four validation images, the selected operating point depends on a small sample of annotation differences. The present design measures that transfer directly, instead of interpreting an optimized validation value as evidence of generalization.

The larger practical issue is reference dependence. The same RF masks receive different Dice scores against the two observers. A reader comparing papers needs both the annotation used to select the threshold and the annotation used for final evaluation. Reporting a second observer only as a human comparator does not describe this selection step. Conversely, reducing the difference between two observer scores would not by itself show a more accurate vessel map: both scores could decline, and neither annotation is an adjudicated biological truth.

This does not make maximin thresholding intrinsically unsuitable. It defines a clear objective when performance against multiple references matters. What is unsupported here is an empirical improvement from that objective under this small validation protocol. Repeating the test with additional independently annotated cohorts and stronger segmentation models would address a different and broader claim.

## 6.2 Limitations and Transfer

The study contains 14 children from one public collection. RF and ET are related tree ensembles sharing the same handcrafted features, not independent replications across model families. No deep network was fitted. Training uses observer 1 throughout, so the experiment is not a symmetric crossover of training observers. The measured effects may change when training against observer 2, fusing labels, increasing the validation cohort, or altering the score distribution.

Image crop  
Observer 1  
Observer 2  
O1-tuned  
Maximin  
![](images/c0ca88853d1d9a2ace1616e0b52fb744f46dadb32004a3bcf72dcdc6d78be040.jpg)  
Figure 3: Predetermined RF seed-17 examples. Each panel shows the same 400 × 400 crop, rows 230–629 and columns 290–689 in zero-based coordinates. The original photographs are cropped only for display; masks are the supplied annotations or actual held-out predictions. Image credit: CHASE\_DB1 creators (Fraz et al., 2012), CC BY 4.0. No image synthesis or generative editing is used.

Whole-image scoring, the native resolution and the balanced training sample also constrain comparisons. Dark background pixels affect ranking metrics, while a 0.50 decision threshold has no special optimality after balanced sampling. The strong improvement over fixed 0.50 should therefore not be advertised as a new algorithmic gain. Other CHASE\_DB1 studies may use different partitions, preprocessing or evaluation regions; their reported numbers cannot be ranked against Table 1 without a matched rerun.

The bootstrap intervals describe variation across the observed subjects conditional on the fitted predictions. They do not resolve dependence induced by overlapping training folds. Three seeds characterize a limited amount of estimator variability and do not increase the number of independent subjects. We provide the raw paired values so that the size and direction of the effects can be assessed without relying on a significance claim.

Finally, overlap scores do not establish topological correctness, vessel-calibre accuracy, diagnostic benefit or clinical readiness. The qualitative cases show errors that scalar threshold changes cannot repair. Additional expert adjudication and external evaluation would be needed to determine whether a changed mask better represents the vasculature. The present results support a narrower reporting practice: retain both referencespecific scores, state the validation reference explicitly, and keep operating-point selection separate from heldout evaluation.

Drafting assistance: (OpenAI, 2026); see disclosure.

## 7 CONCLUSION

A subject-separated CHASE\_DB1 experiment isolated the annotation used for threshold selection from the fitted segmentation model. Across 42 fits, maximin frequently changed the selected operating point but did not improve worst-observer Dice relative to observer-1 tuning. The mean changes were -0.073 percentage points for RF and -0.119 for ET, with conditional intervals spanning zero. Reference-specific evaluation of identical masks showed a larger numerical difference. These findings support documenting both the threshold-selection annotation and the evaluation annotation, and retaining negative results when a validation objective fails to transfer to unseen subjects. The supplied data, predictions, split manifests and code make the comparison directly auditable.

Drafting assistance: (OpenAI, 2026); see disclosure.

## ACKNOWLEDGEMENTS AND DISCLOSURE

The public CHASE\_DB1 photographs and annotations are credited to their creators (Fraz et al., 2012). The institutional data record supplies CC BY 4.0 terms. A versioned copy was obtained from the public Study Group Learning repository (Zhou et al., 2021); that repository’s trained models and pseudo-labels were not used. Git blob and SHA-256 checks verify the retrieved mirror files, not an independent byte comparison with the institutional archive.

OpenAI ChatGPT (Codex) (OpenAI, 2026) assisted with study planning, source discovery, code development, experiment execution, analysis scripts, the abstract and Sections 1–7, and manuscript revision. Figures are deterministic plots or crops of public images and computed predictions, produced with AIassisted code; no synthetic patient images or fabricated measurements are used. The human authors retain responsibility for the final submitted content. No AI system is an author. Only publicly released data were analysed; no new participant data were collected.

## REFERENCES

Breiman, L. (2001). Random forests. Machine Learning, 45:5–32.

Frangi, A. F., Niessen, W. J., Vincken, K. L., et al. (1998). Multiscale vessel enhancement filtering. In MICCAI.

Fraz, M. M., Remagnino, P., Hoppe, A., et al. (2012). An ensemble classification-based approach applied to retinal blood vessel segmentation. IEEE Transactions on Biomedical Engineering, 59(9):2538–2548.

Geurts, P., Ernst, D., and Wehenkel, L. (2006). Extremely randomized trees. Machine Learning, 63:3–42.

Hoover, A., Kouznetsova, V., and Goldbaum, M. (2000). Locating blood vessels in retinal images by piecewise threshold probing of a matched filter response. IEEE Transactions on Medical Imaging, 19(3):203–210.

Kohl, S. A. A., Romera-Paredes, B., Meyer, C., et al. (2018). A probabilistic U-Net for segmentation of ambiguous images. In Advances in Neural Information Processing Systems, volume 31.

Li, L., Verma, M., Nakashima, Y., et al. (2020). IterNet: Retinal image segmentation utilizing structural redundancy in vessel networks. In IEEE/CVF Winter Conference on Applications of Computer Vision.

Lipton, Z. C., Elkan, C., and Narayanaswamy, B. (2014). Optimal thresholding of classifiers to maximize F1 measure. In ECML PKDD, pages 225–239.

Maganti, S. (2026). Rethinking uncertainty in segmentation: From estimation to decision. arXiv:2604.13262.

Maier-Hein, L., Reinke, A., Godau, P., et al. (2024). Metrics reloaded: Recommendations for image analysis validation. Nature Methods, 21:195–212.

Maninis, K.-K., Pont-Tuset, J., Arbeláez, P., et al. (2016). Deep retinal image understanding. In MICCAI, pages 140–148.

Nguyen, U. T. V., Bhuiyan, A., Park, L. A. F., et al. (2013). An effective retinal blood vessel segmentation method using multi-scale line detection. Pattern Recognition, 46(3):703–715.

OpenAI (2026). ChatGPT (Codex). Research, code, and drafting assistance; accessed September 2026.

Pedregosa, F., Varoquaux, G., Gramfort, A., et al. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12:2825–2830.

Reinke, A., Tizabi, M. D., Baumgartner, M., et al. (2024). Understanding metric-related pitfalls in image analysis validation. Nature Methods, 21:182–194.

Ronneberger, O., Fischer, P., and Brox, T. (2015). U-Net: Convolutional networks for biomedical image segmentation. In MICCAI, pages 234–241.

Saito, T. and Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. PLOS ONE, 10(3):e0118432.

Staal, J., Abràmoff, M. D., Niemeijer, M., et al. (2004). Ridge-based vessel segmentation in color images of the retina. IEEE Transactions on Medical Imaging, 23(4):501–509.

Taha, A. A. and Hanbury, A. (2015). Metrics for evaluating 3D medical image segmentation: Analysis, selection, and tool. BMC Medical Imaging, 15:29.

Wang, C., Xu, R., Xu, S., et al. (2022). DA-Net: Dual branch transformer and adaptive strip upsampling for retinal vessels segmentation. In MICCAI, pages 528–538.

Warfield, S. K., Zou, K. H., and Wells, W. M. (2004). Simultaneous truth and performance level estimation (STA-PLE): An algorithm for the validation of image segmentation. IEEE Transactions on Medical Imaging, 23(7):903–921.

Wu, J., Fang, H., Xiong, H., et al. (2022). Calibrate the interobserver segmentation uncertainty via diagnosis-first principle. arXiv:2208.03016.

Zhou, Y., Yu, H., and Shi, H. (2021). Study group learning: Improving retinal vessel segmentation trained with noisy labels. In MICCAI, pages 57–67.