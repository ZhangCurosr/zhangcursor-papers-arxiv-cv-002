# Interpretable Brain Age Prediction Using White Matter Fiber Orientation Distributions with Rotation-Equivariant Deep Learning

Chenyang Fan, Robert D. Sanders, Elie Matar, Yixuan Yuan, Xiang Li, Sharon L. Naismith, Fernando Calamante, and Jinglei Lv.

Abstract—Diffusion MRI (dMRI) provides information about white matter microstructure that can support brain age prediction. However, many existing prediction models use diffusion tensor imaging (DTI) scalar metrics that omit orientation information. Deep learning is increasingly used for brain age prediction, but conventional CNNs do not explicitly respect the geometric structure of orientationdependent features. We present an interpretable brain age prediction framework that combines white matter fiber orientation distributions (WM-FODs) with an SE(3) (Special Euclidean group in 3 dimensions) equivariant CNN. The network operates directly on spherical harmonic (SH) coefficients of the FOD, preserving their transformation properties under rotation. Evaluation in 692 participants aged 36–90 years from the Human Connectome Project Aging dataset showed that the combined framework outperformed the evaluated DTI-based models and conventional CNNs. Averaging predictions from five crossvalidation models yielded a mean absolute error of 3.54 years and Pearson’s correlation of 0.958. Our results indicated that the performance advantage depended on combining WM-FODs with equivariant processing. Prediction accuracy was highest at a maximum SH order of Lmax = 2 and declined at higher orders. Tract-level occlusion, where each segmented tract was masked in turn, revealed age-dependent patterns of model sensitivity, with importance peaks in frontal projection pathways at 36–44 years, the fornix at 45–54 years, the callosal isthmus at 55– 64 years, the middle cerebellar peduncle at 65–74 years, and the callosal genu at 75–90 years. These findings support rotation-equivariant processing of WM-FODs for accurate brain age prediction with anatomically interpretable model sensitivity.

Index Terms—Brain age, diffusion MRI, geometric deep learning, SE(3) equivariance, explainable artificial intelligence, fiber orientation distributions.

## I. INTRODUCTION

quantitative measure of age-related brain characteristics and has emerged as a potential biomarker of brain health [1]. The difference between predicted brain age and chronological age, termed the brain age gap (BAG), has been associated with mortality risk [2], cognitive decline [3], and neurological disorders [4]. Early approaches primarily used T1-weighted structural MRI to characterize macroscopic anatomical changes, including atrophy. Diffusion MRI (dMRI) provides complementary information about white matter microstructure and the fiber pathways that support cognition.

Deep learning approaches to dMRI-based brain age prediction face two key limitations. First, many studies use scalar metrics derived from diffusion tensor imaging (DTI), such as fractional anisotropy (FA) [5], [6], [7]. Although convenient, these metrics omit orientation information in the diffusion signal, limiting their representation of complex fiber configurations [8]. Second, conventional convolutional neural networks (CNNs) do not explicitly account for the transformation properties of orientation-dependent dMRI features.

White matter fiber orientation distributions (WM-FODs) describe the distribution of fiber orientations within each voxel. They can be represented by spherical harmonic (SH) coefficients that transform together under rotation (Fig. 1). Within each SH order, rotating an FOD mixes the coefficients according to a prescribed transformation; the coefficients therefore have a shared geometric structure. Standard CNNs treat them as ordinary input channels without enforcing this structure. SE(3)-equivariant CNNs instead constrain their operations so that the resulting features transform consistently

with rotations of the input [9].

![](images/2a7164d90a80f079a42cd2f0923a31e33be5477b3bcab4c9bbad61ad6cb1b11e.jpg)  
Fig. 1. WM-FOD representation and SE(3)-equivariant processing. A WM-FOD with Lmax = 8 is decomposed into even-order spherical harmonic components (ℓ = 0, 2, 4, 6, 8), which transform under rotation according to Wigner D-matrices. The SE(3)-equivariant CNN treats each order as a geometric field (e.g., ℓ = 0 for scalars and ℓ = 2 for second-order components) and combines these fields using steerable kernels. The resulting scalar, vector, and higher-order features retain their prescribed transformation properties.

Interpretability is also important for understanding the anatomical information used in brain age prediction. Fixed-size occlusion windows and voxel-wise saliency maps can be difficult to relate to individual white matter pathways. Anatomically informed occlusion offers a complementary approach: masking tract-defined regions allows the sensitivity of a trained model to be assessed in an anatomical context. Such analyses identify regions relevant to prediction, although they do not establish causal roles in aging.

## This study makes three contributions:

1) Feature representation: We compare WM-FODs with conventional DTI-derived features and evaluate the effect of maximum SH order. WM-FODs combined with equivariant processing outperform the evaluated DTI-based models, with prediction accuracy peaking at Lmax = 2.

2) Architecture: We developed an SE(3)-equivariant CNN [9] that operates directly on WM-FOD SH coefficients. We evaluate its predictive accuracy, stability under input rotations, and computational cost against conventional CNNs in wholebrain analyses.

3) Interpretability: We use tract-level occlusion to quantify model sensitivity to anatomically defined white matter regions. Tract-level and volume-normalized importance measures identify pathways relevant to prediction and characterize how their importance varies across adult age groups.

## II. RELATED WORK

## A. Brain Age Prediction Using Diffusion MRI

Previous dMRI-based brain age models in healthy aging cohorts have reported mean absolute errors (MAEs) of approximately 3.5–7.6 years, in samples ranging from 89 [10] to more than 30,000 [11] participants. Methods include linear and regularized regression [12], [13], [11], [14], [15], support vector machines [10], tree-based ensembles [5], [7], [16], [17], and deep learning [6], [18], [19]. Inputs have included DTI scalars, free-water DTI measures, advanced diffusion metrics, tract-wise summaries, and white matter connectome features. These approaches generally use scalar or handcrafted representations that do not preserve the full orientation structure of the diffusion signal.

Differences in cohort age range complicate comparisons across studies. A narrower age range can reduce MAE while also reducing the variance available to explain, resulting in a lower correlation between predicted and chronological age. For example, the lowest MAE among the studies compared here was 3.51 years over a 23-year age span, with Pearson’s correlation (r) = 0.49 [13]. Studies spanning a broader adult age range generally reported higher MAEs but stronger correlations. Reporting both MAE and r, together with the cohort age range, therefore, supports more informative comparisons.

## B. From Scalar Metrics to WM-FODs

DTI and WM-FODs differ in both their orientation information and their modeling assumptions. DTI models diffusion within each voxel using a Gaussian tensor model and is susceptible to partial-volume effects, including contamination by cerebrospinal fluid (CSF). In contrast, multishell multi-tissue constrained spherical deconvolution (MSMT-CSD) [20] estimates WM-FODs that can represent complex fiber configurations while separating white matter, gray matter, and CSF signal components.

FOD-derived measures are widely used in fixel-based analyses to characterize fiber-specific white matter properties [21]. Their use as direct inputs to deep learning models for brain age prediction, however, remains relatively unexplored.

## C. Geometric Deep Learning for Diffusion MRI

Geometric deep learning incorporates known symmetries into neural network architectures. SE(3)-equivariant networks use steerable filters to ensure that features transform consistently under three-dimensional rotations [9]. For dMRI, Müller et al. introduced joint SE(3) × SO(3) convolutions on raw q-space signals [22], while Elaldi et al. developed E(3) × SO(3)-equivariant hemispherical convolutions for spherical deconvolution [23]. These methods demonstrate the value of equivariant processing for diffusion signal modeling and FOD estimation. Their focus differs from downstream prediction using precomputed WM-FOD representations.

In our framework, WM-FOD SH coefficient fields already encode angular information. A volumetric SE(3)-equivariant CNN can therefore process these fields directly, without the additional angular (SO(3)) convolution required by the joint spatial–angular methods above. We evaluate this approach for brain age prediction.

## D. Tract-Level Occlusion in Diffusion MRI Brain-Age Models

Regional and network-level occlusion analyses have identified anatomically interpretable sensitivities in structural MRI-based brain age models [24], [25], [26]. These studies have primarily examined gray matter or atlas-defined regions rather than diffusion-derived white matter pathways.

Previous dMRI-based brain age studies have used segmented bundles, tract groups, or along-tract diffusion summaries to estimate age and characterize bundle-specific aging patterns [14], [17]. Although these methods provide anatomical localization through predefined tract features, they do not directly test the sensitivity of a trained whole-brain model to tract-defined regions. We therefore apply tract-level occlusion as an anatomically informed, post-hoc analysis.

## III. METHODS

Figure 2 summarizes the processing, prediction, and interpretation pipelines.

![](images/f1cd944056c4c10f4dfe283886fc57712af15facec493fe0663e7d1b1972c7c4.jpg)  
Fig. 2. Pipeline for brain age prediction and tract-level interpretation. Diffusion-weighted images are preprocessed and analyzed using MSMT-CSD to estimate WM-FODs at different maximum SH orders. SE(3)-ResNet34 processes the WM-FODs using equivariant convolutions, progressive channel expansion, and spatial downsampling, followed by global pooling and age prediction. Occlusion of 72 tract-defined regions yields importance scores (|ΔMAE|) that characterize the model’s anatomical sensitivity.

## A. Dataset and Acquisition

Diffusion MRI data were obtained from Lifespan Release 2.0 of the Human Connectome Project-Aging (HCP-A) [27], which includes 725 participants. Participants older than 100 years (n=12) and those whose dMRI data failed preprocessing because of considerable residual imaging artifacts (n=21) were excluded. The final sample comprised 692 healthy participants aged 36–90 years (mean ± SD, 59.6 ± 14.9 years; 56% female). Data were acquired on a Siemens 3T Prisma scanner using a multi-shell protocol with 93 diffusion directions at b = 1500 s/mm² and 92 directions at b = 3000 s/mm², at 1.5-mm isotropic resolution. Alternating anterior-to-posterior and posterior-toanterior phase encoding enabled susceptibility-distortion correction.

## B. Preprocessing

Preprocessing followed the MRtrix3 pipeline [28] and included denoising, Gibbs ringing removal, susceptibilitydistortion and motion correction using FSL topup and eddy, and bias-field correction using ANTs N4. Images were upsampled to 0.8-mm isotropic resolution.

## C. Diffusion Feature Extraction

DTI metrics: The diffusion tensor was fitted using weighted linear least squares [29]. This yielded fractional anisotropy (FA), mean diffusivity (MD), axial diffusivity (AD), radial diffusivity (RD), and the six independent tensor elements. For early fusion, the four scalar maps were concatenated as input channels. For late fusion, a separate model was trained for each scalar metric, and the resulting age predictions were averaged at inference. Separately, the six tensor elements were evaluated as a six-channel input.

WM-FOD images: MSMT-CSD [20] was performed using the average of the tissue-specific response functions estimated for each participant with the Dhollander algorithm [30]. WM-FODs were computed at maximum SH orders of Lmax = 0, 2, 4, 6, and 8, corresponding to 1, 6, 15, 28, and 45 coefficient channels, respectively, to assess the optimal performance for age prediction.

## D. Model Architectures

SE(3)-ResNet34: We developed an SE(3)-equivariant convolutional framework based on the work of Weiler et al. [9] to construct a ResNet-34 architecture (Fig. 2). Input SH coefficients were mapped to their corresponding irreducible representations, with each even SH order (ℓ = 0, 2, 4, 6, 8) represented as a separate geometric field. Analogous to the channels of a conventional CNN, each hidden layer contained multiple feature fields of orders ℓ = 0–3 (Fig. 2), with fewer fields at higher orders. The maximum hidden order (ℓ ≤ 3) was chosen because it gave higher accuracy than ℓ ≤ 1 and $\ell \leq 5$ in preliminary experiments. Spatial downsampling used Gaussian low-pass filtering followed by strided subsampling. Group normalization, gated nonlinearities, and equivariant dropout (p = 0.01) were applied between layers. The final block retained only scalar (ℓ = 0) fields, which are invariant to rotation. Global mean pooling and a fully connected layer produced the final age estimate.

Baseline CNNs: We evaluated ResNet50 and ResNet152, adapted for three-dimensional volumetric inputs, and SFCN [31], a lightweight architecture designed for brain age prediction.

## E. Training Procedure

Participants were assigned to training (n = 588; 85%) and held-out test (n = 104; 15%) sets using stratified sampling to preserve the age distribution. Five-fold cross-validation was performed within the training set. Models were optimized to minimize MAE using Adam, with an initial learning rate of 1 × 10⁻³ and weight decay of $1 \times 1 0 ^ { - 4 }$ . The learning rate was halved after 5 epochs without improvement in validation loss, down to a minimum of $1 \times 1 0 ^ { - 6 }$ . Early stopping used a patience of 15 epochs. Training used mixed precision on four Tesla V100 GPUs, with a per-GPU batch size of 4 for standard CNNs and 1 for SE(3)-ResNet34.

## F. Evaluation of Prediction Stability Under Rotation

To assess prediction stability under input rotations, we rotated the spatial WM-FOD volumes from $- 1 5 ^ { \circ }$ to $+ 1 5 ^ { \circ }$ in $1 ^ { \circ }$ increments in the axial, coronal, and sagittal planes. SH coefficients were reoriented simultaneously using the corresponding Wigner D-matrices [32]. An illustrative test participant was selected for whom SE(3)-ResNet34 and ResNet50 had the same unrotated absolute prediction error (0.5 years). Stability was quantified as the standard deviation of predicted ages across rotations. We also evaluated a ResNet50 model trained with rotation augmentation (random rotations of $\pm 1 ^ { \circ } ,$ applied with probability 0.5) to compare architectural equivariance with augmentation-based robustness. All test participants $( \mathtt { n } = 1 0 4 )$ were then evaluated at discrete rotations of 0°, ±5°, ±10°, and $\pm 1 5 ^ { \circ }$ about each anatomical axis. Paired ttests compared rotation-induced prediction variability between models.

## G. Regional Occlusion Analysis

Tract segmentation: TractSeg [33] was used to segment 72 anatomically defined white matter bundles, grouped as association, projection, commissural, cerebellar, or limbic pathways.

Occlusion procedure: For each tract, all voxels within its mask were set to zero, and $\mathrm { S E } ( 3 ) – \mathrm { R e s N e t } 3 4$ predictions were recomputed across the full cohort $( \mathbf { n } = 6 9 2 )$ . Predictions for participants in the cross-validation sample $( \mathbf { n } = 5 8 8 )$ were obtained from their respective out-of-fold models, so that no participant was evaluated by a model trained on that participant. For the held-out test set $( \mathtt { n } = 1 0 4 )$ , predictions were averaged across the five cross-validation models. Predicted ages were corrected for age-related bias using linear regression following Smith et al. [34] before calculating baseline and occluded MAE. The importance score was defined as $\vert \Delta M A E \vert =$ $\vert M A E _ { \{ o c c l u d e d \} } - \ M A E _ { \{ b a s e l i n e \} } \vert$

Per-voxel importance: To account for differences in tract size, |ΔMAE| was divided by tract volume (number of voxels); values are reported in units of $1 0 ^ { - 4 }$ years/voxel.

Age-stratified analysis: Participants were grouped into five age bins: 36–44, 45–54, 55–64, 65–74, and 75–90 years $( { \mathfrak { n } } =$ 143, 154, 137, 120, and 138, respectively). For each tract, mean absolute importance was calculated within each bin, and the bin with the highest importance was identified.

## H. Statistical Analysis

Prediction performance was evaluated on the held-out test set $( \mathtt { n } = 1 0 4 )$ using MAE and Pearson’s r. Single-model results are reported as mean ± SD across the five cross-validation models; ensemble results were obtained by averaging the five models’ predictions for each participant. Models were compared using two-sided paired t-tests on participant-wise absolute prediction errors, and rotation stability was compared using paired t-tests on participant-wise SDs of predicted age across rotations. Pvalues were adjusted for multiple comparisons within each family of related tests using the Benjamini–Hochberg false discovery rate procedure $( \mathsf { q } < 0 . 0 5 )$ .

## IV. RESULTS

## A. Comparison of Brain Age Prediction Performance

## 1) DTI-Derived Baseline Features

$\mathrm { W e }$ first evaluated DTI-derived features using ResNet50 (Table 1). FA yielded the lowest MAE among the individual scalar metrics $( 5 . 3 6 \pm 0 . 4 4$ years). Using the six independent tensor elements $( \mathrm { D \_ x x } , \mathrm { D \_ x y } , \mathrm { D \_ x z } , \mathrm { D \_ y y } , \mathrm { D \_ y z } , \mathrm { D \_ z z } )$ as input channels performed similarly to FA $( 5 . 4 8 \pm 0 . 3 9$ years; p $= 0 . 7 0 )$ . Early fusion of all four scalar maps produced the best DTI-based performance $( \mathrm { M A E } = 4 . 9 7 \pm 0 . 1 6$ years; ${ \bf r } = 0 . 9 1 6 )$ and was used as the primary DTI benchmark.

TABLE 1. BRAIN AGE PREDICTION PERFORMANCE OF DTI-DERIVED FEATURES USING RESNET50
<table><tr><td>DTI Feature</td><td>Test MAE (years)</td><td>Pearson&#x27;s r</td></tr><tr><td>FA</td><td> $5 . 3 6 \pm 0 . 4 4$ </td><td> $\overline { { 0 . 8 9 2 6 \pm 0 . 0 1 9 1 } }$ </td></tr><tr><td>MD</td><td> $5 . 8 3 \pm 0 . 3 3$ </td><td> $\overline { { 0 . 8 8 3 0 \pm 0 . 0 1 3 2 } }$ </td></tr><tr><td>AD</td><td> $6 . 0 2 \pm 0 . 2 8$ </td><td> $0 . 8 7 3 0 \pm 0 . 0 0 6 7$ </td></tr><tr><td>RD</td><td> $6 . 3 2 \pm 0 . 4 3$ </td><td> $0 . 8 6 9 6 \pm 0 . 0 1 3 1$ </td></tr><tr><td>DTI tensor elements</td><td> $5 . 4 8 \pm 0 . 3 9$ </td><td> $\overline { { 0 . 8 9 8 4 \pm 0 . 0 1 4 6 } }$ </td></tr><tr><td>Early fusion</td><td> $4 . 9 7 \pm 0 . 1 6$ </td><td> $0 . 9 1 5 5 \pm 0 . 0 0 3 6$ </td></tr><tr><td>Late fusion</td><td> $5 . 1 4 \pm 0 . 2 3$ </td><td> $0 . 9 0 6 7 \pm 0 . 0 0 9 1$ </td></tr></table>

\*Values are mean ± SD across the five cross-validation models on the heldout test set (n = 104).

## 2) Effect of Maximum SH Order

We evaluated WM-FODs at five maximum SH orders (Lmax $= 0 , 2 , 4 , 6 , 8 )$ with SE(3)-ResNet34, ResNet50, ResNet152, and SFCN (Fig. 3; Table 2). All architectures showed a nonmonotonic relationship between maximum SH order and accuracy, with the lowest prediction error at Lmax = 2; for ResNet50, for example, MAE increased from 4.66 years at Lmax = 2 to 5.57 years at Lmax = 8. Within SE(3)-ResNet34, Lmax = 2 performed significantly better than Lmax = 0, 6, and $8 \left( \mathbf { p } < 0 . 0 5 \right)$ , but did not differ significantly from Lmax = 4 (p $= 0 . 2 7 )$

![](images/57d3e45db40c9aa7272096274ad31fffa0ab98138b7a06f072de66038d76f00f.jpg)  
Fig. 3. Test-set MAE across maximum SH orders (Lmax) for SE(3)-ResNet34, ResNet50, registered-input ResNet50, ResNet152, and SFCN. Points are slightly offset to improve visibility. All architectures achieved their lowest MAE at Lmax = 2, where SE(3)-ResNet34 achieved $4 . 0 4 \pm 0 . 4 0$ years. Error bars show the SD of test-set MAE across the five cross-validation models.

## 3) Effect of Network Architecture

At Lmax = 2, SE(3)-ResNet34 achieved an MAE of 4.04 ± 0.40 years and $\mathbf { r } = 0 . 9 4 2$ , outperforming ResNet50 $\left( \mathtt { p } = 0 . 0 0 6 \right)$ ResNet152 $( \mathtt { p } ^ { < } 0 . 0 0 1 )$ , and SFCN $\mathrm { ( p < 0 . 0 0 1 ) }$ . SE(3)-ResNet34 achieved the highest accuracy at every maximum SH order.

To test whether spatial normalization could substitute for architectural equivariance, we compared SE(3)-ResNet34 applied to unregistered WM-FODs with ResNet50 applied to WM-FODs nonlinearly registered to a tissue-unbiased FOD template [35] using MRtrix3 mrregister with FOD reorientation (Table 2). Registration did not significantly improve ResNet50 at any Lmax (all $\mathsf { p } > 0 . 0 5 )$ , and SE(3)-ResNet34 without registration outperformed registered ResNet50 at every evaluated order (all $\mathsf { p } < 0 . 0 5 )$

## TABLE 2. WM-FOD BRAIN AGE PREDICTION MAE ACROSS MAXIMUM SH ORDERS AND SELECTED NETWORK CONFIGURATIONS

<table><tr><td></td><td>Lmax SE(3)-ResNet34</td><td>ResNet50</td><td>ResNet50 (registered)</td></tr><tr><td>0</td><td> $4 . 5 0 \pm 0 . 2 6$ </td><td> $5 . 2 8 \pm 0 . 2 4$ </td><td> $\overline { { 4 . 9 7 \pm 0 . 1 9 } }$ </td></tr><tr><td>2</td><td> $\underline { { { \bf 4 . 0 4 \pm 0 . 4 0 } } }$ </td><td> $4 . 6 6 \pm 0 . 3 3$ </td><td> $\underline { { 4 . 6 4 \pm 0 . 2 6 } }$ </td></tr><tr><td>4</td><td> $4 . 3 6 \pm 0 . 3 1$ </td><td> $5 . 1 4 \pm 0 . 2 8$ </td><td> $4 . 9 3 \pm 0 . 2 9$ </td></tr><tr><td>6</td><td> $4 . 5 3 \pm 0 . 4 5$ </td><td> $5 . 2 8 \pm 0 . 4 3$ </td><td> $5 . 1 9 \pm 0 . 3 5$ </td></tr><tr><td>8</td><td> $4 . 8 2 \pm 0 . 3 5$ </td><td> $5 . 5 7 \pm 0 . 3 3$ </td><td> $5 . 7 8 \pm 0 . 4 4$ </td></tr></table>

Values are test-set MAE (years), given as $\mathrm { m e a n } \pm \mathrm { S D }$ across the five crossvalidation models $( \mathtt { n } = 1 0 4 )$ . Bold indicates the lowest MAE. Registered: WM-FODs nonlinearly registered to the FOD template.

## 4) Combining WM-FODs with Equivariant Processing

The best WM-FOD model (SE(3)-ResNet34, 4.04 years; Table 2) improved on the DTI benchmark (ResNet50 with early fusion, 4.97 years; Table 1) by 0.93 years. To determine whether this gain arose from the input features, the network architecture, or their combination, we compared configurations that differed in only one component. With ResNet50 held fixed, replacing DTI early fusion with WM-FODs (Lmax = 2) did not significantly improve accuracy (4.66 versus 4.97 years; ${ \mathfrak { p } } =$ 0.35). With DTI early fusion held fixed, replacing ResNet50 with SE(3)-ResNet34 also produced no significant improvement $( 4 . 9 2 ~ \pm ~ 0 . 1 3$ versus 4.97 years; $\mathrm { ~ p ~ } = \ 0 . 7 3 )$ . Accuracy improved significantly only when WM-FODs were combined with SE(3)-ResNet34.

## 5) Ensemble Prediction Performance

Averaging predictions from the five cross-validation models improved performance across configurations. For SE(3)- ResNet34 at Lmax = 2, ensemble averaging reduced MAE from 4.04 to 3.54 years and yielded ${ \bf r } = 0 . 9 5 8$ . The baseline models also benefited from averaging (Fig. 4), while the SE(3)- ResNet34 ensemble retained significantly better performance than all baseline ensembles (all $\mathsf { p } < 0 . 0 5 )$ .

![](images/170502b8f297ffb66003542800dce94c7f412413e5aca6c2d7cdf843948112c3.jpg)

![](images/f8f9e60afbd4270b09d49abfb4b69a123aaaeee4d1682cbe5d4e736cacfe203c.jpg)

![](images/51e0712ae0299280d3ca8cbf4f748620f4b60e28ff30be8974c77e385d994148.jpg)

![](images/062d9c89accfec1269e4053da12ab540ac011a25392179e31278ba108c038d64.jpg)  
Fig. 4. Ensemble-predicted versus chronological age in the held-out test set (n = 104): (a) DTI FA baseline, (b) DTI early fusion, (c) ResNet50 with WM-FODs, and (d) SE(3)-ResNet34 with WM-FODs. Each prediction is the average of five cross-validation model predictions. The dashed gray line denotes identity $( \mathbf { y } = \mathbf { x } ) ,$ and solid colored lines show linear fits. The SE(3)-ResNet34 ensemble achieved the lowest prediction error.

## B. Architectural Evaluation

## 1) Prediction Stability Under Rotation

Figure 5 illustrates prediction stability for a test participant whose unrotated absolute error was 0.5 years for both SE(3)- ResNet34 and ResNet50. Across rotations of up to $\pm 1 5 ^ { \circ }$ in the axial, coronal, and sagittal planes, SE(3)-ResNet34 predictions had an SD of 0.28 years. ResNet50 predictions were more variable (SD = 2.41 years), with absolute errors exceeding 8 years at $- 1 5 ^ { \circ }$ coronal rotation. Thus, the prediction SD was 8.6 times greater for ResNet50 in this example.

The analysis of all test participants (n = 104) showed the same pattern. SE(3)-ResNet34 had a mean rotation-induced prediction SD of $0 . 9 0 \pm 0 . 4 0$ years, significantly lower than ResNet50 $( 2 . 9 9 ~ \pm ~ 0 . 8 9$ years; $\mathsf { p } \ < \ 0 . 0 0 1 )$ and rotationaugmented ResNet50 $( 2 . 0 4 ~ \pm ~ 0 . 4 5$ years; $\mathrm { ~ \tt ~ p ~ } < \mathrm { ~ \tt ~ 0 . 0 0 1 ) }$ Augmentation improved stability relative to unaugmented ResNet50 $( \mathsf { p } \ < \ 0 . 0 0 1 )$ ), but did not significantly improve prediction accuracy $( \mathrm { M A E } = 4 . 6 7 \pm 0 . 2 8$ versus $4 . 6 6 \pm 0 . 3 3$ years; p = 0.83).

## 2) Computational Cost

SE(3)-ResNet34 required 406 seconds per training epoch, approximately 1.6 times the 258 seconds required by ResNet50, compared with 265 seconds for ResNet152 and 231 seconds for SFCN. With 2.95 million parameters in the configuration of Fig. 2, SE(3)-ResNet34 was approximately 16 and 40 times smaller than ResNet50 (46.27 million) and ResNet152 (117.47 million), respectively, and comparable in size to SFCN (2.95 million), while achieving the best prediction performance. The equivariant architecture therefore achieved its accuracy with far fewer parameters than the ResNet baselines, at a higher computational cost per epoch.

![](images/754310b2f7458f8a8e4af38bcce7fd03458923d48542e2043d608da8fe5b31cb.jpg)  
Fig. 5. Prediction stability under input rotation. Absolute age prediction error is shown for rotations from $- 1 5 ^ { \circ } \ \mathrm { t o } \ + 1 5 ^ { \circ }$ in the axial (circles), coronal (squares), and sagittal (triangles) planes for an illustrative test participant. Both models had an unrotated absolute error of 0.5 years. SE(3)-ResNet34 (red, solid) remained comparatively stable, whereas ResNet50 (blue, dashed) showed errors exceeding 8 years $\tt a t - 1 5 ^ { \circ }$ coronal rotation.

## C. Tract-Level Importance

Tract-level occlusion quantified model sensitivity using two complementary measures: the mean absolute change in prediction error after occlusion (|ΔMAE|) and its volumenormalized counterpart. The highest values (Fig. 6A) occurred in corpus callosum subregions CC\_2 (genu; |ΔMAE| = 3.99 years) and CC\_6 (isthmus; $| \Delta \mathrm { M A E } | = 3 . 9 2 $ years), and the middle cerebellar peduncle (MCP; $| \Delta \mathbf { M A E } | \ = \ 3 . 8 1$ years). Several projection pathways also ranked highly, including the bilateral prefrontal thalamic radiations (T\_PREF), frontopontine tracts (FPT), and anterior thalamic radiation (ATR).

![](images/a8bf60d49ca0966ea216dea17fafca294428c3674d69a3a03da2c86dbd2efcba.jpg)

![](images/44e59c6d628c28712e805ea3d4d3f98b15cc1d308cdc6d5f845b6e73087912c5.jpg)  
Fig. 6. Tract-level model sensitivity. (A) Absolute change in age-biascorrected prediction error after tract occlusion. Box plots show participantlevel values $( \mathbf { n } = 6 9 2 ) ;$ numbers at right give the mean (|ΔMAE|, years). $\mathrm { C C } \_ 2$ (genu), $\mathrm { C C } \_ { 6 }$ (isthmus), and the MCP showed the largest effects. (B) Pervoxel importance, calculated by normalizing |ΔMAE| by tract volume $( \times 1 0 ^ { - 4 }$ years/voxel); the bilateral fornix ranked highest. The 10 highest-ranked tracts are shown for each measure and colored by anatomical category. Box plots indicate the median and interquartile range (IQR), with whiskers extending to $1 . 5 \times \mathrm { I Q R } ;$ ; outliers are not shown.

Normalizing |ΔMAE| by tract volume reduced the influence of tract size on the importance ranking (Fig. 6B). The bilateral fornix had the highest per-voxel importance (FX\_right: $6 . 7 0 \times$ $1 0 ^ { - 4 }$ years/voxel; FX\_left: $5 . 6 0 \times 1 0 ^ { - 4 } \colon$ years/voxel), followed by the anterior commissure $\mathrm { ( C A } \colon 1 . 0 8 \times 1 0 ^ { - 4 }$ years/voxel). All remaining tracts had values below $0 . 6 5 \times 1 0 ^ { - 4 }$ years/voxel.

## D. Age-Stratified Tract Importance

We compared tract importance across five age bins to characterize age-related variation in model sensitivity (Fig. 7). Tract importance maps and streamlines were displayed on the tissue unbiased brain template [35]. For each tract, we identified the age bin with the highest importance; for each bin, we report the tracts with the largest peak values. These comparisons are descriptive.

![](images/3c2ce374df9f734efba86b883a543f5fc68c7a892e61cc4a46b1ad9e1c5db4a5.jpg)  
Fig. 7. Tract importance across five age bins spanning 36–90 years. Selected importance peaks occur in frontal projection pathways at 36–44 years, the fornix at $4 5 { - } 5 4$ years, the callosal isthmus at 55–64 years, the MCP at 65–74 years, and the callosal genu at 75–90 years. Bottom row: streamlines colored by importance. All panels use the same color scale. Age-group differences describe model sensitivity rather than longitudinal changes within individuals.

Early midlife (36–44 years): the largest peaks were in frontal projection pathways, namely, the left prefrontal thalamic radiation $( | \Delta \mathrm { M A E } | ~ = ~ 5 . 0 8 $ years), left fronto-pontine tract $( | \Delta \mathrm { M A E } | = 4 . 7 2 $ years), and left anterior thalamic radiation $( | \Delta \mathrm { M A E } | = 4 . 6 5 $ years); their right-hemisphere counterparts also peaked in this bin (4.84, 4.47, and 4.36 years, respectively). Midlife (45–54 years): the fornix reached its highest importance (left $| \Delta \mathrm { M A E } | = 2 . 1 4$ years; right 1.91 years). Late midlife (55– 64 years): the isthmus of the corpus callosum $( \mathrm { C C } \_ { 6 } )$ reached its highest importance $( | \Delta \mathrm { M A E } | = 4 . 5 7 \mathrm { y e a r s } )$ . Early aging (65– 74 years): the middle cerebellar peduncle reached its highest importance (|ΔMAE| = 4.70 years). Late aging (75–90 years): the genu of the corpus callosum (CC\_2) reached its highest importance (|ΔMAE| = 5.03 years).

## V. DISCUSSION

This study examined how diffusion feature representation and geometric network architecture contribute to dMRI-based brain age prediction. Three findings emerged. First, the combination of WM-FODs and SE(3)-equivariant processing achieved the best prediction performance; changing either component alone did not significantly improve accuracy in comparisons that varied one component at a time. Second, accuracy peaked at Lmax = 2, and the equivariant model was more stable under input rotations than conventional CNNs. Third, tract-level occlusion identified age-dependent patterns of model sensitivity, with particularly high volume-normalized importance in the bilateral fornix.

## A. WM-FOD Representations for Brain Age Prediction

FA is the most widely used DTI scalar input for age prediction and yielded the lowest error among the individual scalar metrics in our experiments, while combining scalar metrics further improved prediction accuracy, consistent with the value of complementary diffusion information [5], [16]. Comparisons that varied one component at a time also suggest that the benefit of WM-FODs depends on how the network processes them. Replacing DTI fusion with WM-FODs within ResNet50 did not significantly improve accuracy, and applying SE(3)-ResNet34 to DTI fusion likewise produced no significant gain. The strongest performance was obtained when the orientationdependent WM-FOD representation was processed with an architecture designed to respect its geometry, i.e., SE(3)- ResNet34.

Several properties may contribute to this result. MSMT-CSD relaxes the single-Gaussian assumption of DTI and separates white matter, gray matter, and CSF signal components [20]. WM-FOD SH coefficients also retain orientation information that is absent from rotationally invariant DTI scalar maps. Previous studies have shown that FOD-derived measures are sensitive to age-related white matter differences [36], [37].

We observed a non-monotonic relationship between SH order and prediction performance, with accuracy peaking at Lmax = 2 and declining at higher orders. One possible explanation is that increasing Lmax from 2 to 8 raises input dimensionality from 6 to 45 channels, which may make generalization more difficult with 588 training participants. At the same time, agerelevant information may be concentrated at low orders: healthy aging is characterized by widespread reductions in fiber density [36] and anisotropy [38], which at the voxel level may be largely expressed in the ℓ = 0 (total fiber density) and ℓ = 2 terms. Moreover, although Lmax = 2 does not resolve crossingfiber populations as distinct peaks, its coefficients, being a linear projection of the FOD, still change with the relative contribution and dispersion of fiber populations within a voxel, and may therefore capture age-related changes without explicit peak resolution.

The SE(3)-ResNet34 ensemble achieved an MAE of 3.54 years and r = 0.958, comparing favorably with published unimodal dMRI-based brain age studies in healthy aging cohorts (Fig. 8). Note however that such comparisons provide general context rather than a controlled ranking, because performance depends on cohort size, age range, acquisition, preprocessing, and evaluation procedures.

![](images/d4fb41da706ab9e418cc24b11c77f5632d2ad344465b23501d0da6976be2175f.jpg)  
Fig. 8. Comparison with unimodal dMRI-based brain age studies in healthy aging cohorts that include adults older than 65 years. Red markers indicate SE(3)-ResNet34 performance with and without ensemble averaging. Differences in cohorts and evaluation procedures should be considered when interpreting these comparisons.

## B. SE(3)-Equivariant Processing of WM-FODs

SE(3)-equivariant processing improved prediction from WM-FODs relative to the conventional CNNs. Previous equivariant dMRI methods include the joint SE(3) × SO(3) convolutions of Müller et al. for raw q-space signals [22] and the E(3) × SO(3) hemispherical convolutions of Elaldi et al. [23]. These approaches jointly model the spatial and angular domains. Our network instead processes SH coefficients estimated in advance using MSMT-CSD. This compact angular representation maps directly to the irreducible representations used by SE(3)- steerable kernels, enabling equivariant processing without an additional SO(3) angular convolution. To our knowledge, this is the first application of a volumetric SE(3)-equivariant CNN directly to whole-brain WM-FOD SH coefficient images for brain age prediction.

The gains in accuracy and rotational stability are consistent with better alignment between the network architecture and WM-FOD geometry. Standard CNNs mix SH coefficients without enforcing how each SH order transforms under rotation, whereas SE(3)-equivariant networks preserve these transformations. This may also improve sample efficiency: similar fiber configurations occur at different orientations across tracts and participants, and equivariant kernels share weights across orientations rather than learning each separately [9]. Consistent with this, SE(3)-ResNet34 was more stable under rotation than ResNet50; rotation augmentation narrowed but did not close this gap and did not significantly improve accuracy.

Registration did not substitute for equivariance: SE(3)- ResNet34 applied without registration outperformed ResNet50 applied to registered WM-FODs at every evaluated SH order.

This benefit came with a computational trade-off. SE(3)- ResNet34 required approximately 1.6 times the training time per epoch of ResNet50, but had only 2.95 million parameters, approximately 1/16 of the ResNet50 parameter count. SFCN provided a parameter-matched comparison: with a similar number of parameters, it performed significantly worse, so the gain cannot be attributed to model size.

## C. Age-Dependent Model Sensitivity to White Matter Regions

Tract-level occlusion identified age-dependent patterns of model sensitivity that can be considered alongside previous findings on white matter aging.

In early midlife (36–44 years), the largest peaks occurred in frontal projection pathways: the prefrontal thalamic radiation, fronto-pontine tract, and anterior thalamic radiation. Diffusionderived measures of white matter tracts have been reported to peak between approximately 20 and 40 years [38], followed by age-related reductions in white matter integrity [39]. This bin may therefore capture the onset of their decline, although occlusion does not identify which microstructural properties drive their importance.

The bilateral fornix had the highest per-voxel importance and peaked in midlife (45–54 years). Previous work has identified pronounced age-related reductions in fornix FA [40]. As a principal hippocampal output pathway, the fornix has also been linked to memory performance [41] and progression from mild cognitive impairment to dementia [42], and its degeneration has been suggested to precede hippocampal dysfunction [43]. Its small size and proximity to the ventricles, however, make it susceptible to CSF partial-volume effects that increase with age, which MSMT-CSD reduces but may not fully remove.

The MCP reached its highest importance at 65–74 years (Fig. 7). Age-related differences in the cerebellum have been linked to motor and cognitive decline [44], although evidence specific to the MCP is limited.

The callosal isthmus and genu reached their highest importance at 55–64 and 75–90 years, respectively. Age-related white matter decline has been reported to be more pronounced in anterior regions [45], [46], which include the prefrontal territory connected by the genu. However, the timing of peak predictive importance does not directly indicate the timing or rate of tissue degeneration.

Across age groups, the most prominent sensitivities involved specific projection, limbic, commissural, and cerebellar bundles. Their correspondence with previously reported age-related differences supports the plausibility of the model’s feature use. These importance patterns should nevertheless be interpreted as properties of the trained model: they do not establish causal mechanisms of aging or directly measure cognitive function.

## D. Limitations and Future Work

Several limitations warrant consideration. First, evaluation was restricted to one healthy aging cohort; generalization across scanners, acquisition protocols, and clinical populations remains to be established. Second, voxel-wise tract occlusion also removes signals from any crossing fiber populations within the mask, so importance reflects sensitivity to tract-defined regions rather than isolated bundles. This confounding effect increases for larger tract volumes, as more sections of unrelated tracts will also be occluded. Third, normalization by tract volume reduces the influence of tract size but does not account for tract topology or overlap. External validation and more fiber-specific perturbation methods are therefore important directions for future work.

## VI. CONCLUSION

We presented an SE(3)-equivariant framework for brain age prediction from WM-FODs. Processing WM-FODs with an SE(3)-equivariant network yielded the best performance among the evaluated configurations, with an ensemble MAE of 3.54 years and r = 0.958. The model also showed greater stability under input rotations and used approximately 1/16 of the parameters of ResNet50. Comparisons that varied one component at a time showed that the only significant gain was obtained when WM-FOD inputs were combined with equivariant processing, rather than with either change alone. Tract-level occlusion further identified age-dependent patterns of model sensitivity to white matter regions. These findings support SE(3)-equivariant networks, which preserve how WM-FOD SH coefficients transform under rotation, as a promising approach to accurate and interpretable dMRI-based brain age prediction.

## ACKNOWLEDGMENT

Research reported in this publication was supported by the National Institute On Aging of the National Institutes of Health under Award Number U01AG052564 and by funds provided by the McDonnell Center for Systems Neuroscience at Washington University in St. Louis. The HCP-Aging 2.0 Release data used in this report came from DOI: 10.15154/1520707. J. Lv is supported by Brain and Mind Centre Research Development Grant, USYD-Fudan Brain and Intelligence Science Alliance Flagship Research Program, Moyira Elizabeth Vine Fund for Research into Schizophrenia Program. JL, FC and SN are supported by an ARC Discovery Project (DP240102161).

## REFERENCES

[1] J. H. Cole et al., “Predicting brain age with deep learning from raw imaging data results in a reliable and heritable biomarker,” Neuroimage, vol. 163, pp. 115–124, Dec. 2017, doi: 10.1016/j.neuroimage.2017.07.059.

[2] J. H. Cole et al., “Brain age predicts mortality,” Mol Psychiatry, vol. 23, no. 5, pp. 1385–1392, May 2018, doi: 10.1038/mp.2017.62.

[3] M. L. Elliott et al., “Brain-age in midlife is associated with accelerated biological aging and cognitive decline in a longitudinal birth cohort,” Mol Psychiatry, vol. 26, no. 8, pp. 3829–3838, Aug. 2021, doi: 10.1038/s41380- 019-0626-7.

[4] T. Kaufmann et al., “Common brain disorders are associated with heritable patterns of apparent aging of the brain,” Nat Neurosci, vol. 22, no. 10, Art. no. 10, Oct. 2019, doi: 10.1038/s41593-019-0471-7.

[5] S. Tønnesen et al., “Brain Age Prediction Reveals Aberrant Brain White Matter in Schizophrenia and Bipolar Disorder: A Multisample Diffusion Tensor Imaging Study,” Biological Psychiatry: Cognitive

Neuroscience and Neuroimaging, vol. 5, no. 12, pp. 1095–1103, Dec. 2020, doi: 10.1016/j.bpsc.2020.06.014.

[6] Y. Wang, J. Wen, J. Xin, Y. Zhang, H. Xie, and Y. Tang, “3DCNN predicting brain age using diffusion tensor imaging,” Med Biol Eng Comput, vol. 61, no. 12, pp. 3335–3344, Dec. 2023, doi: 10.1007/s11517-023-02915- x.

[7] G. Richard et al., “Assessing distinct patterns of cognitive aging using tissue-specific brain age prediction based on diffusion tensor imaging and brain morphometry,” PeerJ, vol. 6, p. e5908, Nov. 2018, doi: 10.7717/peerj.5908.

[8] B. Jeurissen, A. Leemans, J.-D. Tournier, D. K. Jones, and J. Sijbers, “Investigating the prevalence of complex fiber configurations in white matter tissue with diffusion magnetic resonance imaging,” Hum Brain Mapp, vol. 34, no. 11, pp. 2747–2766, Nov. 2013, doi: 10.1002/hbm.22099.

[9] M. Weiler, M. Geiger, M. Welling, W. Boomsma, and T. Cohen, “3D Steerable CNNs: Learning Rotationally Equivariant Features in Volumetric Data,” Oct. 27, 2018, arXiv: arXiv:1807.02547. doi: 10.48550/arXiv.1807.02547.

[10] F. Nemmi, M. Levardon, and P. Péran, “Brain‐age estimation accuracy is significantly increased using multishell free‐water reconstruction,” Hum Brain Mapp, vol. 43, no. 7, pp. 2365–2376, Feb. 2022, doi: 10.1002/hbm.25792.

[11] J. Wen et al., “The genetic architecture of multimodal human brain age,” Nat Commun, vol. 15, no. 1, p. 2604, Mar. 2024, doi: 10.1038/s41467-024-46796-6.

[12] J. H. Cole, “Multimodality neuroimaging brain-age in UK biobank: relationship to biomedical, lifestyle, and cognitive factors,” Neurobiology of Aging, vol. 92, pp. 34–42, Aug. 2020, doi: 10.1016/j.neurobiolaging.2020.03.014.

[13] A.-M. G. de Lange et al., “Multimodal brain-age prediction and cardiovascular risk: The Whitehall II MRI sub-study,” NeuroImage, vol. 222, p. 117292, Nov. 2020, doi: 10.1016/j.neuroimage.2020.117292.

[14] A. Salih et al., “Brain age estimation at tract group level and its association with daily life measures, cardiac risk factors and genetic variants,” Scientific Reports, vol. 11, 2021, doi: 10.1038/s41598-021-99153-8.

[15] A. Richie-Halford, J. D. Yeatman, N. Simon, and A. Rokem, “Multidimensional analysis and detection of informative features in human brain white matter,” PLoS Comput Biol, vol. 17, no. 6, p. e1009136, Jun. 2021, doi: 10.1371/journal.pcbi.1009136.

[16] D. Beck et al., “White matter microstructure across the adult lifespan: A mixed longitudinal and cross-sectional study using advanced diffusion models and brain-age prediction,” NeuroImage, vol. 224, p. 117441, Jan. 2021, doi: 10.1016/j.neuroimage.2020.117441.

[17] Y. Feng, J. E. Villalón-Reina, T. M. Nir, B. Q. Chandio, N. Jahanshad, and P. M. Thompson, “BundleAGE: Predicting White Matter Age using Along-Tract Microstructural Profiles from Diffusion MRI,” bioRxiv,

p. 2024.08.16.608347, Aug. 2024, doi: 10.1101/2024.08.16.608347.

[18] C.-L. Chen et al., “Generalization of diffusion magnetic resonance imaging–based brain age prediction model through transfer learning,” NeuroImage, vol. 217, p. 116831, Aug. 2020, doi: 10.1016/j.neuroimage.2020.116831.

[19] C. Gao et al., “Predicting Age from White Matter Diffusivity with Residual Learning,” ArXiv, p. arXiv:2311.03500v2, Jan. 2024.

[20] B. Jeurissen, J.-D. Tournier, T. Dhollander, A. Connelly, and J. Sijbers, “Multi-tissue constrained spherical deconvolution for improved analysis of multi-shell diffusion MRI data,” NeuroImage, vol. 103, pp. 411– 426, Dec. 2014, doi: 10.1016/j.neuroimage.2014.07.061.

[21] D. A. Raffelt et al., “Investigating white matter fibre density and morphology using fixel-based analysis,” Neuroimage, vol. 144, no. Pt A, pp. 58–73, Jan. 2017, doi: 10.1016/j.neuroimage.2016.09.029.

[22] P. Müller, V. Golkov, V. Tomassini, and D. Cremers, “Rotation-Equivariant Deep Learning for Diffusion MRI,” Feb. 13, 2021, arXiv: arXiv:2102.06942. doi: 10.48550/arXiv.2102.06942.

[23] A. Elaldi, G. Gerig, and N. Dey, “E(3) x SO(3)- Equivariant Networks for Spherical Deconvolution in Diffusion MRI,” in Medical Imaging with Deep Learning, PMLR, Jan. 2024, pp. 301–319. Accessed: Nov. 26, 2025. [Online]. Available: https://proceedings.mlr.press/v227/elaldi24a.html

[24] X. Zhang et al., “Brain age prediction using interpretable multi-feature-based convolutional neural network in mild traumatic brain injury,” NeuroImage, vol. 297, p. 120751, Aug. 2024, doi: 10.1016/j.neuroimage.2024.120751.

[25] L. He et al., “Network Occlusion Sensitivity Analysis Identifies Regional Contributions to Brain Age Prediction,” Hum Brain Mapp, vol. 46, no. 8, p. e70239, Jun. 2025, doi: 10.1002/hbm.70239.

[26] J. Lee et al., “Deep learning-based brain age prediction in normal aging and dementia,” Nat Aging, vol. 2, no. 5, Art. no. 5, May 2022, doi: 10.1038/s43587-022-00219-7.

[27] S. Y. Bookheimer et al., “The Lifespan Human Connectome Project in Aging: An overview,” Neuroimage, vol. 185, pp. 335–348, Jan. 2019, doi: 10.1016/j.neuroimage.2018.10.009.

[28] J.-D. Tournier et al., “MRtrix3: A fast, flexible and open software framework for medical image processing and visualisation,” Neuroimage, vol. 202, p. 116137, Nov. 2019, doi: 10.1016/j.neuroimage.2019.116137.

[29] P. J. Basser, J. Mattiello, and D. Lebihan, “Estimation of the Effective Self-Diffusion Tensor from the NMR Spin Echo,” Journal of Magnetic Resonance, Series B, vol. 103, no. 3, pp. 247–254, Mar. 1994, doi: 10.1006/jmrb.1994.1037.

[30] T. Dhollander, D. Raffelt, and A. Connelly, “Unsupervised 3-tissue response function estimation from single-shell or multi-shell diffusion MR data without a co-registered T1 image,” Sep. 2016.

[31] H. Peng, W. Gong, C. F. Beckmann, A. Vedaldi, and S. M. Smith, “Accurate brain age prediction with lightweight deep neural networks,” Medical Image Analysis, vol. 68, p. 101871, Feb. 2021, doi: 10.1016/j.media.2020.101871.

[32] M. Geiger and T. Smidt, “e3nn: Euclidean Neural Networks,” Jul. 18, 2022, arXiv: arXiv:2207.09453. doi: 10.48550/arXiv.2207.09453.

[33] J. Wasserthal, P. Neher, and K. H. Maier-Hein, “TractSeg - Fast and accurate white matter tract segmentation,” NeuroImage, vol. 183, pp. 239–253, Dec. 2018, doi: 10.1016/j.neuroimage.2018.07.070.

[34] S. M. Smith, D. Vidaurre, F. Alfaro-Almagro, T. E. Nichols, and K. L. Miller, “Estimation of brain age delta from brain imaging,” NeuroImage, vol. 200, pp. 528– 539, Oct. 2019, doi: 10.1016/j.neuroimage.2019.06.017.

[35] J. Lv, R. Zeng, M. P. Ho, A. D’Souza, and F. Calamante, “Building a tissue-unbiased brain template of fiber orientation distribution and tractography with multimodal registration,” Magnetic Resonance in Medicine, vol. 89, no. 3, pp. 1207–1220, 2023, doi: 10.1002/mrm.29496.

[36] E. M. Tinney et al., “Understanding Cognitive Aging Through White Matter: A Fixel‐Based Analysis,” Hum Brain Mapp, vol. 45, no. 18, p. e70121, Dec. 2024, doi: 10.1002/hbm.70121.

[37] S. Zanoni, J. Lv, S. L. Naismith, R. E. Smith, and F. Calamante, “Streamline-Based Analysis: A novel framework for tractogram-driven streamline-wise statistical inference,” Res Sq, p. rs.3.rs-10123464, Jul. 2026, doi: 10.21203/rs.3.rs-10123464/v1.

[38] C. Lebel, M. Gee, R. Camicioli, M. Wieler, W. Martin, and C. Beaulieu, “Diffusion tensor imaging of white matter tract evolution over the lifespan,” NeuroImage, vol. 60, no. 1, pp. 340–352, Mar. 2012, doi: 10.1016/j.neuroimage.2011.11.094.

[39] S. Conte, D. Zimmerman, and J. E. Richards, “White matter trajectories over the lifespan,” PLoS One, vol. 19, no. 5, p. e0301520, May 2024, doi: 10.1371/journal.pone.0301520.

[40] A. C. Yang, S.-J. Tsai, M.-E. Liu, C.-C. Huang, and C.- P. Lin, “The Association of Aging with White Matter Integrity and Functional Connectivity Hubs,” Front. Aging Neurosci., vol. 8, Jun. 2016, doi: 10.3389/fnagi.2016.00143.

[41] K. J. Bangen et al., “Decreased myelin content of the fornix predicts poorer memory performance beyond vascular risk, hippocampal volume, and fractional anisotropy in nondemented older adults,” Brain Imaging and Behavior, vol. 15, no. 5, pp. 2563–2571, Oct. 2021, doi: 10.1007/s11682-021-00458-z.

[42] M. M. Mielke et al., “Fornix integrity and hippocampal volume predict memory decline and progression to Alzheimer’s disease,” Alzheimers Dement, vol. 8, no. 2, pp. 105–113, 2012, doi: 10.1016/j.jalz.2011.05.2416.

[43] V. Douet and L. Chang, “Fornix as an imaging marker for episodic memory deficits in healthy aging and in various neurological disorders,” Front. Aging Neurosci., vol. 6, Jan. 2015, doi: 10.3389/fnagi.2014.00343.

[44] J. A. Bernard and R. D. Seidler, “Moving Forward: Age Effects on the Cerebellum Underlie Cognitive and Motor Declines,” Neurosci Biobehav Rev, vol. 0, pp. 193–207, May 2014, doi: 10.1016/j.neubiorev.2014.02.011.

[45] I. J. Bennett, D. J. Madden, C. J. Vaidya, D. V. Howard, and J. H. Howard Jr., “Age-related differences in multiple measures of white matter integrity: A diffusion tensor imaging study of healthy aging,” Human Brain Mapping, vol. 31, no. 3, pp. 378–390, 2010, doi: 10.1002/hbm.20872.

[46] M. O’Sullivan, D. K. Jones, P. E. Summers, R. G. Morris, S. C. Williams, and H. S. Markus, “Evidence for cortical ‘disconnection’ as a mechanism of age-related cognitive decline,” Neurology, vol. 57, no. 4, pp. 632–638, Aug. 2001, doi: 10.1212/wnl.57.4.632.