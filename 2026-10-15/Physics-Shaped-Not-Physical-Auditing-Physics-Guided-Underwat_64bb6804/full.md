# Physics-Shaped, Not Physical: Auditing Physics-Guided Underwater Image Restoration

Multi-seed accuracy, data eficiency, and validation against measured range

Mochamad Rizal Fauzan<sup>1</sup> Irgi Surya<sup>1</sup>

<sup>1</sup>College of Electrical Engineering and Computer Science National Taipei University of Technology, Taipei, Taiwan Corresponding author: M. R. Fauzan (t114998415@ntut.org.tw)

## Abstract

Physics-guided networks for underwater image restoration predict transmission and backscatter and invert the image-formation model, yet rarely test whether this physics does any work. We audit one representative design, PhysGuide-Net, with four controlled experiments: five-seed ablations on UIEB with zero-shot evaluation on the complete LSUI (4279 pairs) and AUIED3K (3000 pairs) sets; a data-eficiency study with a capacity-matched control that removes the formation model; and a validation of the predicted maps against measured range on Sea-thru. With all 800 training images the physics module gives no zero-shot gain (mean PSNR diferences of at most 0.16 dB; all 95 % intervals include zero) at 3.4× the FLOPs. With 80 images it helps more (+0.65 dB on UIEB, +1.48 dB on AUIED3K; seed-to-seed s.d. on AUIED3K from 1.3 to 0.5 dB), although not individually significant after correction, and the control does not reproduce this (0.8–1.0 dB lower). Yet the predicted transmission does not behave like a physical quantity: its correlation with range is no stronger than for the range map of a diferent image (pooled diference +0.02, versus −0.06 for the dark-channel prior), and a Beer–Lambert fit explains 0.4 % of its variance. The efect of the module also changes sign with degradation severity between datasets. The benefit thus comes from a physics-shaped inductive bias, not from physics. We further show that single-run ablations can reverse sign and document implementation defects that invalidate loss ablations.

Index Terms—underwater image restoration, physics-guided deep learning, physical validation, data eficiency, zero-shot generalization, reproducibility.

## 1 Introduction

Every pixel of an underwater photograph records the water it crossed. Red light vanishes within a few meters, blue–green scatter veils the scene, and both efects grow with range [1, 2, 3]. It is therefore natural to build this physics into restoration networks: predict a transmission map and a backscatter term, invert the image-formation model, and let a network refine the result. Many such physics-guided designs have been proposed [4, 5, 6], and the idea parallels the broader movement toward physics-informed learning [7, 8]. Yet two questions that decide whether the physics deserves its name are rarely asked: is it doing any work, and is it physics at all?

The evidence ofered is often thin. Gains are commonly measured with one training run on a small benchmark such as UIEB (890 pairs) [9], as diferences of a few tenths of a decibel, although seed-to-seed variance of a network trained on 800 images can be of the same order [10, 11, 12]. The split that selects the checkpoint is frequently the one used for reporting. To our knowledge, predicted maps are rarely compared with measured range, even though datasets that allow this exist [13]. And the most natural benefit of a prior, helping when data are scarce [14], is seldom tested.

We therefore audit one representative design, PhysGuide-Net [15], with four controlled experiments whose design is fixed in Section 4: RQ1 a five-seed ablation on UIEB with zero-shot evaluation on the complete LSUI [16] and AUIED3K [5] sets; RQ2 a data-eficiency study with a capacity-matched control that keeps the architecture but removes the formation model; RQ3 a test of the predicted maps against the measured range of Sea-thru [13]; and RQ4 an analysis of how the efect depends on degradation severity.

## What we find.

• With the full training set the physics module gives no zero-shot accuracy benefit, at 3.4× the FLOPs; the only diference that survives correction lies on the checkpoint-selection split.

• The module does help when data are scarce: the advantage grows as the training set shrinks (up to +1.5 dB on AUIED3K at 80 images), and on AUIED3K the seed-to-seed spread falls from 1.3 to 0.5 dB at that size. A control with identical capacity but no formation model does not reproduce the gain. With five seeds these low-data gains are a consistent trend, not individually significant after correction.

• The predicted transmission is nevertheless not a physical quantity: on Sea-thru it follows measured range no better than the range map of a diferent image does, and a classical dark-channel prior does better.

• The efect of the module depends on degradation severity, with opposite sign on LSUI and AUIED3K.

In short, the physics in this design acts as an inductive bias that is physics-shaped, not physical: it can be useful, but not for the reason it is usually given.

Relation to our earlier work. This paper extends the study that introduced PhysGuide-Net [15], which evaluated a single seed on UIEB and 400 LSUI images. Here we add multi-seed statistics, zero-shot evaluation on complete external sets, the data-eficiency study with a control, the physical validation, and the severity analysis, and we document the implementation defects that must be excluded before such ablations are meaningful (Section 7).

## 2 Related work

Image formation and classical restoration. The radiance that reaches an underwater camera is commonly modeled as a direct component, attenuated with range and wavelength, plus a backscattered veiling-light component [1, 2]; the model is formally analogous to the atmospheric scattering model of dehazing [17]. Classical restoration estimates its unknowns with hand-crafted priors: wavelength compensation and dehazing [18], red-channel and dark-channel variants [19, 20], blurriness- and absorption-based depth cues [21, 22], and haze-lines with a quantitative color-chart dataset [23]. Fusion-based enhancement [24] avoids explicit physical modeling altogether. Akkaynak and Treibitz showed that the attenuation coeficients are not constant but depend on range and scene [3], which motivated Sea-thru, a method that recovers colors from RGB-D data [13] and whose data we use here to test physical consistency.

Learning-based enhancement. Deep models trained on paired benchmarks dominate recent work. Following the UIEB benchmark and Water-Net [9], adversarial approaches [25, 26, 27, 28], scene-prior CNNs [6], transmission-guided multi-color-space embedding [4], contrastive and semi-supervised training [29, 30] and transformers [16] have been proposed; surveys are given in [31, 32]. Larger and more varied data have been introduced through the LSUI set [16], the real-world RUIE benchmark [33] and AUIED3K [5]. The authors of AUIED3K accompany it with a network that predicts a transmission map and background light. Most of these works report one trained model per method.

Physics-informed learning. Embedding physical structure into neural networks has become a general strategy of scientific machine learning [7, 8]: the physics supplies an inductive bias, which is expected to help most when data are scarce or the test distribution shifts. In underwater restoration the analog is to predict transmission and backscatter and to condition or regularize the restoration branch on them. Whether the predicted quantities are physically meaningful, and whether the bias pays of, is rarely tested directly; this is the question we pursue.

Evaluation practice. Diferences of a few tenths of a decibel are routinely reported although seed-to-seed variance can be of the same order [10, 11, 12]; calls for multi-run reporting and careful statistics [34, 35, 36] have had limited uptake in enhancement papers. Full-reference scores such as the peak signal-to-noise ratio (PSNR) and the structural similarity (SSIM) [37] are complemented by nonreference measures such as the underwater image quality measure (UIQM) [38] and the underwater color image quality evaluation (UCIQE) [39]; color accuracy against charts is conventionally expressed as an angular error [40, 41]. Dataset shift [42] and the amount of training data [14] are known determinants of generalization, which is why we examine both.

## 3 PhysGuide-Net

PhysGuide-Net [15] couples a lightweight estimator of the underwater image-formation variables with a restoration backbone that is conditioned on them (Fig. 1). Three design choices define the system: (i) physical variables are predicted per pixel and turned into an explicit physics-recovered prior; (ii) the restoration network consumes the raw image together with all physical maps; (iii) training is supervised both in pixel space and in the feature space of frozen networks, while the extra supervision is discarded at inference.

## 3.1 Degradation-aware physics-guided estimation

Given $I \in [ 0 , 1 ] ^ { 3 \times H \times W }$ , the degradation-aware physics-guided estimation module (DPGEM) uses a shared convolutional encoder that feeds three heads with bounded outputs: $\hat { T } = \sigma ( \cdot ) \in ( 0 , 1 )$ (one channel), $\hat { B } = 0 . 5 \sigma ( \cdot ) \in ( 0 , 0 . 5 )$ and $\hat { A } = 2 \sigma ( \cdot ) \in ( 0 , 2 )$ (three channels each). Inverting the simplified formation model $I = J T + B \left( 1 - T \right)$ gives the physics-recovered prior

$$
\hat { J } _ { \mathrm { p h y s } } = \mathrm { c l i p } _ { [ 0 , 1 ] } \left( \frac { I - \hat { B } \left( 1 - \hat { T } \right) } { \operatorname* { m a x } ( \hat { T } , \epsilon ) } \right) .\tag{1}
$$

The module has 2.25 M parameters and no direct supervision: ${ \hat { T } } , { \hat { B } } , { \hat { A } }$ are learned only through the reconstruction objective and the regularizers of Section 3.3.

## 3.2 Physics-conditioned restoration backbone

The 13-channel tensor $[ I , \hat { T } , \hat { B } , \hat { A } , \hat { J } _ { \mathrm { p h y s } } ]$ is processed by a four-level U-Net with encoder widths 32, 64, 128, 256, a 512-channel bottleneck and a mirrored decoder with transposed-convolution upsampling. Each block consists of two Conv–BN–GELU layers with a residual shortcut, dropout of 0.1 and a convolutional block attention module (CBAM) [43]; a compact four-head self-attention layer (MHSA) on an $8 \times 8$ token grid at the bottleneck models global color statistics. The backbone has 7.57 M parameters; the frozen ImageNet ResNet-18 [44] used for the task loss is excluded from the 9.82 M trainable inference parameters because it is never executed at test time.

## 3.3 Training objective

With $\hat { J }$ the restored image and J the reference,

$$
\begin{array} { r } { \mathcal { L } = \mathcal { L } _ { 1 } + 0 . 5 \mathcal { L } _ { \mathrm { S S I M } } + 0 . 0 4 \mathcal { L } _ { \mathrm { p e r c } } + 0 . 1 \mathcal { L } _ { \mathrm { p h y s } } + 0 . 0 5 \mathcal { L } _ { \mathrm { t a s k } } . } \end{array}\tag{2}
$$

(a) Inference path  
![](images/4e212bea80ddf3b362c2e775f3e44caf47438c9e07f5a458ad01a97b405ba55f.jpg)

(b) Training-only supervision (not executed at inference)  
![](images/5fe0782464015f524c985e7dd7b82db25ddbc8e9b649947ba984e69403cf27db.jpg)  
Fig. 1. System architecture of PhysGuide-Net. (a) Inference path: the degraded image I passes through the DPGEM, which predicts the transmission, backscatter and attenuation maps; the formation-model inversion of Eq. (1) turns them into the physics-recovered prior $\hat { J } _ { \mathrm { p h y s } } ; I ,$ the three maps and $\hat { J } _ { \mathrm { p h y s } }$ (13 channels) are concatenated and restored by a U-Net with CBAM attention and a self-attention bottleneck (MHSA). (b) Training-only supervision: the restored image is compared with the reference in pixel space and in the feature spaces of a frozen VGG-16 (perceptual) and a frozen ResNet-18 (task preserving), while the physics maps are regularized; markers 1 and 2 link the two panels. Numbers in parentheses are loss weights.

$\mathcal { L } _ { \mathrm { S S I M } } = 1 - \mathrm { S S I M } \ [ 3 7 ] . \ \mathcal { L } _ { \mathrm { p e r c } }$ is the mean absolute diference of normalized VGG-16 [45] features at relu1\_2, relu2\_2 and relu3\_3 [46]. $\begin{array} { r } { \mathcal { L } _ { \mathrm { t a s k } } = \frac { 1 } { | \mathcal { S } | } \sum _ { l \in \mathcal { S } } \left. 1 - \cos \left( \varphi _ { l } ( \hat { J } ) , \varphi _ { l } ( J ) \right) \right. } \end{array}$ averages one minus the cosine similarity over spatial positions of the ResNet-18 feature maps $\varphi _ { l } , S =$ {layer1, layer2, layer3}, with the reference features detached. $\mathcal { L } _ { \mathrm { p h y s } }$ combines total variation (TV) on $\hat { T } _ { : }$ , penalties for ${ \hat { T } } \notin [ 0 . 0 5 , 1 ]$ and $\hat { B } \notin [ 0 , 0 . 5 ]$ , a wavelength-ordering penalty max $( 0 , \hat { A } _ { G } - \hat { A } _ { R } ) + \operatorname* { m a x } ( 0 , \hat { A } _ { B } - \hat { A } _ { G } )$ and an $\ell _ { 1 }$ term between $\hat { J }$ and the detached $\hat { J } _ { \mathrm { p h y s } }$ (physics-auxiliary reconstruction consistency, PARC). Because $\hat { B }$ is sigmoid-bounded, its bound penalty is identically zero; the remaining terms are active.

## 3.4 Controlled variants

All variants share data, split, optimizer and schedule and difer in a single component:

• Full: as described.

${ \bf w } / { \bf o }$ physics module: DPGEM and physics fusion removed (the backbone sees only RGB), $\mathcal { L } _ { \mathrm { p h y s } }$ disabled; $\mathcal { L } _ { \mathrm { t a s k } }$ and $\mathcal { L } _ { \mathrm { p e r c } }$ kept.

${ \bf w } / { \bf o }$ task loss: $\mathcal { L } _ { \mathrm { t a s k } }$ disabled.

• Plain U-Net: the same backbone trained with $\mathcal { L } _ { 1 } , \mathcal { L } _ { \mathrm { S S I M } }$ and $\mathcal { L } _ { \mathrm { p e r c } }$ only; against $^ { * } \mathrm { \bar { w } } / \mathrm { o }$ physics module” it isolates $\mathcal { L } _ { \mathrm { t a s k } }$

## 4 Study design

We organize the study around four research questions, each answered by a dedicated experiment:

RQ1 (accuracy). With everything else controlled, does the physics guidance improve restoration, in domain and zero-shot?

RQ2 (data eficiency). Does the prior help when the amount of training data shrinks?

RQ3 (physical validity). Do the predicted maps behave like physical quantities, i.e. does $\hat { T }$ fall with measured range?

RQ4 (severity). Does the efect depend on how strongly an image is degraded?

## 4.1 Data

UIEB (Underwater Image Enhancement Benchmark) [9] provides 890 raw/reference pairs; we keep the fixed 800/90 train/validation split of the earlier study. Two sets are used only for zero-shot evaluation (never for training or checkpoint selection), on their complete content: LSUI (Large-Scale Underwater Image dataset) [16] (4279 pairs in the copy available to us; the original article reports 5004) and AUIED3K (Andaman Underwater Image Enhancement Dataset) [5] (3000 pairs, all $3 8 4 0 \times 2 1 6 0 )$ . In AUIED3K the references are real clear photographs taken at depths of up to 5 m, whereas the raw images are synthesized from them with the dataset authors’ degradation algorithm, which varies per-channel absorption and noise from clear to turbid water [5]; the set therefore tests transfer from real to synthetic degradation. For Sea-thru [13] we use the linear RGB images and measured range maps of five sites (D1–D5); it has no restoration reference and is used only for RQ3. All images are resized to $2 5 6 \times 2 5 6$ (bicubic). For comparison with the earlier study, the 400-image LSUI subset used there (“LSUI-400”) is also reported.

## 4.2 Training and evaluation

AdamW [47] (learning rate $2 \times 1 0 ^ { - 4 }$ , weight decay $1 0 ^ { - 4 } )$ , cosine annealing with warm restarts $( T _ { 0 } = 5 0 )$ , batch size 8, mixed precision, gradient clipping at 1.0, at most 100 epochs with early stopping (patience 15) on validation PSNR. Augmentation: horizontal/vertical flips, $9 0 ^ { \circ }$ rotations and input-only color jitter. Seeds 42–46 control initialization and augmentation order; the split is identical for every run. PSNR and SSIM [37] are computed on [0, 1] RGB at $2 5 6 \times 2 5 6$ for the epoch with the best validation PSNR. Since UIEB-val selects the checkpoint, its numbers are optimistic and are marked †; the zero-shot sets are the primary evidence.

## 4.3 Statistics

We report the mean ± standard deviation over five seeds and paired diferences between variants, using (i) a paired t-test with a 95 % confidence interval (CI) on the per-seed means, and (ii) a Wilcoxon signed-rank test [48] on seed-averaged per-image scores. p-values are Holm-corrected [49] within each experiment: for RQ1 over 4 comparisons × 4 evaluation set $\textsuperscript { ; \times 2 }$ metrics; for RQ2 over the 12 paired gains and, separately, over the six comparisons of each control contrast. The seed-level test has few degrees of freedom but reflects training variance; the image-level test is more sensitive but conditions on the trained models. We therefore base conclusions on efect sizes and seed-level intervals.

## 4.4 RQ2: data-eficiency design

Nested random subsets of the 800 training images (10 %, 25 %, 50 % = 80, 200, 400 images) are drawn once from a fixed permutation, so that every subset contains the smaller ones and is identical across seeds and variants. Full and $^ { 6 } \mathrm { w } / \mathrm { o }$ physics module” are trained on each subset with seeds 42–46 under the unchanged protocol (same epochs, same early stopping); validation uses the same 90 images. To keep the evaluation of each run under a minute, the zero-shot sets are represented by fixed random subsets of 1000 images each; the 100 % runs of the main study are re-scored on the same subsets.

## 4.5 RQ3: physical validation on Sea-thru

Sea-thru images are linear, so each image is exposure-normalized (99.5th percentile) and sRGBencoded before it is fed to the networks, which were trained on display-referred images; this domain shift is a caveat. We use an evenly spaced subset of at most 100 images per site $( 4 1 1$ images in total) and the measured range maps (invalid pixels masked). A physical transmission decreases with range, so we compute per image the Spearman correlation $\rho ( \hat { T } , d )$ between $\hat { T }$ and the range d (expected < 0). Because spatial layout alone (for example, far regions near the top of the frame) can induce a correlation, every statistic is compared with a null control that correlates $\hat { T }$ with the range map of a diferent image of the same site; the image-specific signal is $\Delta = \rho ( \hat { T } , d ) - \rho ( \hat { T } , d _ { \mathrm { n u l l } } )$ and should be negative. For reference we apply the same protocol to a classical dark-channel-prior (DCP) transmission [17] (patch $1 5 \times 1 5 , \omega = 0 . 9 5$ , background light from the brightest 0.1 % of the dark-channel pixels). Further checks are $\rho ( { \hat { B } } , d )$ (expected > 0), the $R ^ { 2 }$ of a log-linear fit log $\hat { T } \sim d$ (Beer–Lambert), and the range dependence of the red-to-green chromaticity ratio of the restored images in linear light, which a range-invariant restoration should reduce. Per-image values are averaged over the five seeds before aggregation.

## 4.6 RQ4: severity analysis

Images of each split are binned into fifths by the PSNR of the unprocessed input against its reference. Within each bin we report the seed-averaged paired diference with a bootstrap 95 % interval over images, and the Spearman correlation between input PSNR and the diference. This analysis was added after seeing the main results and is exploratory.

## 5 Results

## 5.1 RQ1: accuracy in domain and zero-shot

Table 1 and Fig. 2 summarize the five seeds per variant. On UIEB-val, which also selects the checkpoint, the full model has the highest mean PSNR (23.17 dB) and the variant without the physics module the lowest (22.81 dB). On the two zero-shot sets the picture is flat: the variant means span 21.44–21.60 dB on LSUI and 19.59–19.66 dB on AUIED3K, and the full model is the best variant on neither (the plain U-Net is). The seed-to-seed standard deviation (0.09–0.15 dB on LSUI, 0.39–0.54 dB on AUIED3K) is of the same order as the gaps between variants; SSIM shows the same pattern (Table 1, lower block).

Paired comparisons. Table 2 and Fig. 3 give paired diferences. Only one diference survives Holm correction at the seed level: the full model is ahead of the variant without the physics module by +0.37 dB on UIEB-val (95 % CI [+0.24, +0.50], p = 0.046). That is the checkpoint-selection split, and the efect does not transfer: on LSUI the diference is −0.06 dB [−0.28, +0.17] and on AUIED3K +0.06 dB [−0.51, +0.62]. Against the plain U-Net, the relevant baseline for any claim that physics guidance helps, the full model difers by −0.06 dB on LSUI [−0.25, +0.12] and −0.01 dB on AUIED3K [−0.72, +0.70]. The LSUI intervals exclude diferences larger than about 0.3 dB in either direction; the AUIED3K intervals are wider (up to ±0.7 dB) because AUIED3K scores vary by 0.4–0.5 dB across seeds. The task-preserving loss shows no measurable efect either: +0.09 dB on

Table 1. Mean ± standard deviation over five seeds. Params: trainable inference parameters (frozen task encoder excluded); GFLOPs at 256 × 256. Bold: best mean per column. † UIEB-val also selects the checkpoint (optimistic).
<table><tr><td></td><td></td><td></td><td colspan="3">PSNR (dB) ↑</td></tr><tr><td>Variant</td><td>Params (M)</td><td>GFLOPs</td><td>UIEB val (n=90)† LSUI (n=4279)</td><td></td><td>AUIED3K (n=3000)</td></tr><tr><td>PhysGuide-Net (full)</td><td>9.82</td><td>32.2</td><td> ${ \bf 2 3 . 1 7 \pm 0 . 2 2 }$ </td><td> $2 1 . 5 3 \pm 0 . 0 9$ </td><td> $1 9 . 6 5 \pm 0 . 3 9$ </td></tr><tr><td>w/o physics module</td><td>7.57</td><td>9.6</td><td> $2 2 . 8 1 \pm 0 . 1 4$ </td><td> $2 1 . 5 9 \pm 0 . 1 0$ </td><td> $1 9 . 5 9 \pm 0 . 4 6$ </td></tr><tr><td>w/o task loss</td><td>9.82</td><td>32.2</td><td> $2 2 . 9 4 \pm 0 . 3 6$ </td><td> $2 1 . 4 4 \pm 0 . 1 1$ </td><td> $1 9 . 6 5 \pm 0 . 3 9$ </td></tr><tr><td>Plain U-Net</td><td>7.57</td><td>9.6</td><td> $2 2 . 9 4 \pm 0 . 3 1$ </td><td> ${ \bf 2 1 . 6 0 \pm 0 . 1 5 }$ </td><td> ${ \bf 1 9 . 6 6 \pm 0 . 5 4 }$ </td></tr><tr><td></td><td></td><td></td><td colspan="3">SSIM ↑</td></tr><tr><td>PhysGuide-Net (full)</td><td></td><td></td><td> $\mathbf { 0 . 9 0 6 \ : \pm { \ : 0 . 0 0 2 } }$ </td><td> $0 . 8 5 8 \pm 0 . 0 0 2$ </td><td> $0 . 9 0 7 \pm 0 . 0 0 5$ </td></tr><tr><td>w/o physics module</td><td></td><td></td><td> $0 . 9 0 0 \pm 0 . 0 0 2$ </td><td> $0 . 8 5 8 \pm 0 . 0 0 3$ </td><td> $\mathbf { 0 . 9 1 3 \ : \pm { \ : 0 . 0 0 8 } }$ </td></tr><tr><td>w/o task loss</td><td></td><td></td><td> $0 . 9 0 2 \pm 0 . 0 0 4$ </td><td> $0 . 8 5 6 \pm 0 . 0 0 1$ </td><td> $0 . 9 1 1 \pm 0 . 0 0 5$ </td></tr><tr><td>Plain U-Net</td><td></td><td></td><td> $0 . 9 0 3 \pm 0 . 0 0 5$ </td><td> $\mathbf { 0 . 8 5 9 \ : \pm { \ : 0 . 0 0 3 } }$ </td><td> $0 . 9 1 3 \pm 0 . 0 0 8$ </td></tr></table>

![](images/579e49ebda5b734a4dde0db0b2390bbc831eb95e89641bbbea273ce6b11c69ec.jpg)  
Fig. 2. PSNR of every variant, one dot per seed; the black bar is the mean over seeds. The UIEB-val panel is the checkpoint-selection split.

LSUI and 0.00 dB on AUIED3K. At the image level the large number of images makes even diferences of 0.06 dB statistically detectable (Wilcoxon $p < 0 . 0 0 1$ for several comparisons, $| d _ { z } | \leq 0 . 1 9 )$ ; because that test ignores training variance we treat such diferences as detectable but practically negligible.

Single runs mislead. In the single-seed run of the earlier study, removing the task loss appeared to improve PSNR on the 400-image LSUI subset by 0.43 dB (image-level $p < 0 . 0 0 1 )$ . With five seeds the full model is instead ahead of the variant without the task loss by +0.10 dB (95 % CI [−0.05, +0.25]). The image-level test of the single run was “significant” because it described one trained model, not the training procedure.

## 5.2 RQ2: data eficiency and a capacity-matched control

Fig. 4 and Table 3 show how the benefit of the physics module changes with the amount of training data. Two observations stand out.

The advantage grows as data shrink. With all 800 images the paired gain over the variant without the module is −0.06 dB on LSUI and +0.06 dB on AUIED3K. With 80 images (10 %) it is +0.65 dB on UIEB-val (95 % CI [+0.06, +1.24]) and +1.48 dB on AUIED3K ([−0.08, +3.03]), and on AUIED3K it decays monotonically with the training-set size (+1.48, +0.41, +0.21, +0.06 dB for 80, 200, 400, 800 images). On LSUI the gain stays within ±0.2 dB. At 80 images the seed-to-seed standard deviation on AUIED3K is also lower with the module (0.5 dB against 1.3 dB without it), but this is not a consistent pattern across splits and fractions (at 200 images it is 0.5 against 0.4 dB). These low-data gains are a consistent trend rather than established efects: at 10 % only the UIEB-val interval excludes zero, and after Holm correction over the twelve comparisons of this experiment only the full-data UIEB-val diference remains significant $( p = 0 . 0 1 7 )$

Table 2. Paired PSNR diferences (dB). ∆: mean diference of per-seed means with 95 % CI and Holm-corrected paired t p-value over five seeds; $\Delta _ { \mathrm { i m g } } \mathrm { : }$ diference of seed-averaged per-image PSNR with Holm-corrected Wilcoxon p-value. † checkpoint-selection split.
<table><tr><td>Comparison</td><td>Split</td><td>∆PSNR</td><td>95% CI (seeds)</td><td> $p _ { \mathrm { s e e d } }$ </td><td> $\Delta \mathrm { P S N R _ { i m g } }$ </td><td> $p _ { \mathrm { i m g } }$ </td></tr><tr><td>Full – w/o physics module</td><td>UIEB val (n=90)†</td><td>+0.37</td><td>[+0.24, +0.50]</td><td>0.046</td><td>+0.37</td><td>0.006</td></tr><tr><td></td><td>LSUI (n=4279)</td><td>-0.06</td><td>[−0.28, +0.17]</td><td>1.000</td><td>-0.06</td><td>&lt;0.001</td></tr><tr><td></td><td>AUIED3K (n=3000)</td><td>+0.06</td><td>[−0.51, +0.62]</td><td>1.000</td><td>+0.06</td><td>&lt;0.001</td></tr><tr><td>Full - plain U-Net</td><td>UIEB val (n=90)†</td><td>+0.23</td><td>[−0.29, +0.76]</td><td>1.000</td><td>+0.23</td><td>0.118</td></tr><tr><td></td><td>LSUI (n=4279)</td><td>-0.06</td><td>[−0.25, +0.12]</td><td>1.000</td><td>-0.06</td><td>&lt;0.001</td></tr><tr><td></td><td>AUIED3K (n=3000)</td><td>-0.01</td><td>[−0.72, +0.70]</td><td>1.000</td><td>-0.01</td><td>1.000</td></tr><tr><td>Full - w/o task loss</td><td>UIEB val (n=90)†</td><td>+0.24</td><td>[−0.09, +0.57]</td><td>1.000</td><td>+0.24</td><td>0.006</td></tr><tr><td></td><td>LSUI (n=4279)</td><td>+0.09</td><td>[−0.06, +0.24]</td><td>1.000</td><td>+0.09</td><td>&lt;0.001</td></tr><tr><td></td><td>AUIED3K (n=3000)</td><td>0.00</td><td>[−0.49, +0.48]</td><td>1.000</td><td>0.00</td><td>0.274</td></tr><tr><td>w/o physics - plain U-Net</td><td>UIEB val (n=90)†</td><td>-0.13</td><td>[−0.54, +0.28]</td><td>1.000</td><td>-0.13</td><td>0.304</td></tr><tr><td></td><td>LSUI (n=4279)</td><td>-0.01</td><td>[−0.22, +0.21]</td><td>1.000</td><td>-0.01</td><td>&lt;0.001</td></tr><tr><td></td><td>AUIED3K (n=3000)</td><td>-0.07</td><td>[−0.51, +0.38]</td><td>1.000</td><td>-0.07</td><td>&lt;0.001</td></tr></table>

![](images/e3c79b079f493074bd388f961ae21197501c664ef644bc3f69436ea3da3bb2fc.jpg)  
Fig. 3. Paired PSNR diferences between variants (mean and 95 % CI over five seeds). Gray: checkpointselection split; blue: zero-shot sets.

Capacity alone does not reproduce it. The “generic” control keeps the encoder, decoder and the 1/3/3/3 output channels of the module (9.83 M parameters, +0.09 %) but replaces the formationmodel inversion by an ordinary head and drops the physics losses. At 80 images the full model is better than this control by +0.76 dB on UIEB-val, +0.82 dB on LSUI $\left( \mathrm { C I } \left[ + 0 . 2 2 , + 1 . 4 3 \right] \right)$ and +1.03 dB on AUIED3K ([+0.28, +1.77]); at 200 images the diferences are +0.67, +0.65 and +0.21 dB. The generic variant is not better than having no module at all (generic minus $\mathrm { w / o }$ module: −0.11, −0.65 and +0.45 dB at 80 images), and on LSUI it is worse than having no module (−0.65 dB, [−1.29, −0.01], at 80 images; −0.50 dB at 200 images). None of the six full-versus-generic comparisons is individually significant after Holm correction (smallest $p = 0 . 1 1 1 )$ , so this too is a directional result obtained with five seeds. The control removes the inversion and the regularizers together, so their contributions cannot be separated with this design.

![](images/9afea5e5ddeb4780763c4ce0148c424481de71e43b04d9af6365e586281fbb67.jpg)  
Fig. 4. Data-eficiency curves. Top: mean $\mathrm { P S N R \pm s . d }$ . over five seeds for the full model, the variant without the physics module and (at 80 and 200 images) the capacity-matched generic control. Bottom: paired gain over the variant without the module with 95 % CI (gray: full model; green: generic control).

Table 3. Paired PSNR gain of the full model over the variant without the physics module (dB; mean and 95 % CI over five seeds) as a function of the number of UIEB training images. UIEB-val: 90 images († selection split); LSUI and AUIED3K: fixed random subsets of 1000 images each.
<table><tr><td>Training images</td><td>UIEB-val†</td><td></td><td>LSUI</td><td></td><td></td><td>AUIED3K</td></tr><tr><td>80 (10 %)</td><td></td><td>+0.65 [+0.06, +1.24]</td><td></td><td>+0.17 [-0.38, +0.73]</td><td></td><td>+1.48 [−0.08, +3.03]</td></tr><tr><td>200 (25 %)</td><td></td><td>+0.57 [−0.35, +1.48]</td><td></td><td>+0.15 [−0.28, +0.59]</td><td></td><td>+0.41 [−0.34, +1.16]</td></tr><tr><td>400 (50 %)</td><td></td><td>+0.19 [−0.66, +1.05]</td><td></td><td>-0.07 [-0.67, +0.53]</td><td></td><td>+0.21 [−0.43, +0.85]</td></tr><tr><td>800 (100 %)</td><td></td><td>+0.37 [+0.24, +0.50]</td><td></td><td>−0.06 [−0.27, +0.16]</td><td></td><td>+0.06 [−0.53, +0.65]</td></tr></table>

## 5.3 RQ3: are the predicted maps physical?

Table 4 and Fig. 5 report the Sea-thru validation on 411 images. Pooled over the five sites, the correlation between the learned transmission and the measured range is $\rho ( \hat { T } , d ) = - 0 . 0 7$ , hardly below zero and not below the null control (−0.10). The image-specific signal $\Delta = \rho ( \hat { T } , d ) - \rho ( \hat { T } , d _ { \mathrm { n u l l } } )$ is +0.02, i.e. of the wrong sign (Wilcoxon $p = 0 . 0 0 2 )$ ; a physical transmission would give a clearly negative $\Delta .$ . The sign of $\rho$ is physically correct for 55 % of the images, against 60 % for the null range map and 62 % for the classical dark-channel prior. Site by site, $\hat { T }$ is positively correlated with range at D1 $( \Delta = + 0 . 1 0$ , CI above zero), indistinguishable from the null at D3–D5, whereas the dark-channel prior has a negative median $\Delta$ at four of the five sites. The backscatter estimate has the expected positive correlation with range in only 40 % of the images (pooled $\rho = - 0 . 0 5 )$ , and a log-linear (Beer–Lambert) fit of $\hat { T }$ against range explains only 0.4 % of the variance.

The restorations themselves reduce the range dependence of the red-to-green chromaticity ratio (median $| \rho |$ from 0.70 for the input to 0.49 for the full model), but equally so without the physics module (0.47) and for the plain U-Net (0.49): the physics guidance adds no range invariance (right panel of Fig. 5).

![](images/7dec32edebd02d28f3d41acfa6df476e814846db9754e01601cc234fd86aa5d8.jpg)

![](images/455ad4fea6cf8cd9378a279203fb79293b0e3d855719b51095fe0e6a77a9e1ca.jpg)  
Fig. 5. Sea-thru validation. Left: Spearman correlation with measured range per image (boxes: quartiles over images; expected negative and below the null). Right: range dependence of the $\mathrm { R } / \mathrm { G }$ chromaticity ratio of the input and of the restorations (lower is more range-invariant).

Table 4. Sea-thru physical validation (median over images; each image averaged over five seeds). $\rho ( \hat { T } , d )$ Spearman correlation with measured range (physical expectation: negative); null: correlation with the range map of another image of the same site; $\Delta { : }$ diference between the two with bootstrap 95 % CI (physical expectation: negative); DCP ∆: the same for the dark-channel-prior transmission; $\rho ( { \hat { B } } , d )$ : backscatter (expected positive); $R _ { \log \hat { T } } ^ { 2 } \colon$ variance of log T<sup>ˆ</sup> explained by range.
<table><tr><td>Site</td><td>n</td><td> $\rho ( \hat { T } , d )$ </td><td>null</td><td>∆ [95 % CI]</td><td>DCP Δ</td><td> $\rho ( { \hat { B } } , d )$   $R _ { \log \hat { T } } ^ { 2 }$ </td></tr><tr><td>D1</td><td>100</td><td>+0.22</td><td>+0.02</td><td> $+ 0 . 1 0 \ [ + 0 . 0 8 , + 0 . 1 7 ]$ </td><td>-0.20 +0.19</td><td>0.006</td></tr><tr><td>D2</td><td>100</td><td>+0.05</td><td>+0.01</td><td>+0.03 [0.00, +0.05]</td><td>-0.03 -0.03</td><td>0.005</td></tr><tr><td>D3</td><td>68</td><td>-0.16</td><td>-0.16</td><td>−0.01 [−0.03, 0.00]</td><td>-0.10 -0.04</td><td>0.001</td></tr><tr><td>D4</td><td>100</td><td>-0.16</td><td>-0.08</td><td>-0.01 [−0.05, +0.03]</td><td>-0.06 -0.16</td><td>0.005</td></tr><tr><td>D5</td><td>43</td><td>-0.30</td><td>-0.32</td><td>−0.03 [−0.06, +0.03]</td><td>+0.09 -0.22</td><td>0.004</td></tr><tr><td>Pooled</td><td>411</td><td>-0.07</td><td>-0.10</td><td>+0.02</td><td>-0.06 -0.05</td><td>0.004</td></tr></table>

## 5.4 RQ4: dependence on degradation severity

Physics priors should matter most when the degradation is strong. Fig. 6 bins each split by input PSNR and shows the paired diference of the full model against the plain U-Net (blue) and against the variant without the physics module (orange). The dependence on severity is real, but its direction difers between the two zero-shot sets.

• UIEB-val $( n = 9 0 )$ : no trend (Spearman $\rho = - 0 . 0 5$ and −0.07); the diference is positive in most bins.

• LSUI: relative to the variant without physics, the module helps the most degraded fifth (mean input PSNR 12.5 dB: +0.19 dB, 95 % CI $[ + 0 . 1 4 , + 0 . 2 4 ] )$ and hurts the least degraded fifth (22.6 dB: −0.31 dB, [−0.35, −0.28]; $\rho = - 0 . 2 6 )$ , the direction expected of a physical prior. Against the plain U-Net the diferences are small $( | \Delta | \leq 0 . 1 8$ dB in every bin, $\rho = - 0 . 0 9 )$

• AUIED3K: the pattern is reversed $( \rho = + 0 . 5 1$ against the plain U-Net, +0.45 against w/o physics). In the most degraded fifth (10.7 dB) the physics-guided model is 0.53 dB worse than the plain U-Net $( [ - 0 . 5 9 , - 0 . 4 8 ] ; - 0 . 3 7 \mathrm { d B }$ against $\mathrm { w / o }$ physics), and in the least degraded fifth (16.0 dB) it is 0.39 dB better $( [ + 0 . 3 4 , + 0 . 4 4 ] )$ .

![](images/829b0c3a5ffef1a3b539523ddece5b756e10e6249ca8a208483677e47eba5532.jpg)

![](images/84bb4482dedcbcb805f7328dd77baa390385f3b79911495aadb79d3e3632b659.jpg)

![](images/8f193bb1e134ae8018cc20c656bed9642a361c5d660e46c099761dbdaf004b53.jpg)  
Fig. 6. Paired PSNR diference versus degradation severity (input PSNR against the reference, fifths of each split). Shaded: bootstrap 95 % CI over images; $\rho \colon$ Spearman correlation between input PSNR and the diference.

The overall mean diferences thus hide efects of both signs. Since the same trained models are evaluated zero-shot on both sets, a consistent advantage of physics guidance under strong degradation is not supported; the data indicate an interaction between the module and the dataset. These intervals are conditional on the trained models and exclude training variance. Input PSNR is also a crude proxy for physical degradation: on AUIED3K it mainly reflects the strength of the dataset authors’ synthetic degradation (10.7–16 dB across the set).

## 5.5 Qualitative examples and cost

Fig. 7 shows six images drawn at random with a fixed seed (two per dataset; seed-42 models), not selected by outcome. The three variants produce visually similar restorations, and none is consistently best in per-image PSNR across the six examples. The physics module adds 2.25 M parameters (9.82 vs. $7 . 5 7 \mathrm { M } , + 3 0 \% )$ and raises the cost from 9.6 to 32.2 GFLOPs (3.4×) at $2 5 6 \times 2 5 6$ . Early stopping ended training after 76 epochs on average for the full model, against 67 (w/o physics), 67 (w/o task loss) and 86 (plain U-Net).

## 6 Discussion and limitations

What the evidence supports. (i) With ample data the physics module is not worth its cost: zero-shot diferences are at most 0.16 dB on LSUI and 0.07 dB on AUIED3K with intervals that include zero, and the module multiplies the FLOPs by 3.4. (ii) With scarce data the module helps, by up to +1.48 dB on AUIED3K and +0.65 dB on UIEB-val at 80 images, and in the most extreme setting (80 images, AUIED3K) it also lowers the seed-to-seed spread; the trend is consistent but, with five seeds, not individually significant after correction. (iii) Capacity is not the explanation: at 80 images the full model was ahead of a control with the same encoder, decoder and output channels but no formation model by +1.03 dB on AUIED3K and $+ 0 . 8 2 \mathrm { d B }$ on LSUI. (iv) The module’s outputs are not physical: transmission does not track measured range beyond a random-range control, backscatter has the expected sign in only 40 % of the images, and the dark-channel prior behaves more physically. (v) The dependence on degradation severity has opposite sign on the two zero-shot sets, which points to a dataset-specific interaction rather than a physical benefit.

Why can an unvalidated physics module still help? Statements (ii)–(iv) together suggest that the formation-model inversion and its regularizers act as a structured prior on the computation: $\hat { J } _ { \mathrm { p h y s } }$ is a smooth, input-dependent contrast and color stretch that is easy to learn and gives the backbone a strong starting point when little data are available, and the bounded heads and smoothness terms constrain the module’s outputs. This is a hypothesis, not a result: our control removes the inversion and the regularizers jointly, and we did not vary the weight of $\mathcal { L } _ { \mathrm { p h y s } }$ . What the data do say is that the benefit does not require, and does not come with, physically valid maps.

![](images/eab08e794b94b35ff8783283668a5ec1edf9aa43791a1de8edb95d36ca7b8ea1.jpg)  
Fig. 7. Random examples (two per dataset). Columns: input, PhysGuide-Net, w/o physics module, plain U-Net, reference; the PSNR of each restoration is annotated.

Scope of the physical validation. Sea-thru images are linear and were exposure-normalized and sRGB-encoded for the networks, which were trained on display-referred UIEB images; the domain shift may depress any correlation with range, and the module was never supervised with range. Our negative result therefore concerns this unsupervised design under this transfer. A module trained with depth supervision, or a physics-based synthesis of training data, may well behave diferently; our protocol (correlation with measured range, null control, classical baseline) can be applied unchanged to such designs.

## Limitations.

• One architecture, one training set of 800 images and one resolution (256 × 256); the conclusions concern this design and regime, not physics guidance in general.

• Five seeds: seed-level intervals have half-widths of roughly 0.2 dB on LSUI and 0.5–0.7 dB on AUIED3K at full data and are wider at low data, so smaller efects cannot be excluded. Holm correction over the 12 data-eficiency comparisons leaves only one significant.

• Checkpoints are selected on UIEB-val, which is therefore optimistic; zero-shot sets are unafected. The data-eficiency study uses fixed random subsets of 1000 images of LSUI and AUIED3K.

• Only full-reference PSNR and SSIM are reported for the zero-shot sets and no downstream task was evaluated, so a benefit that those would reveal is not excluded. LSUI references are high-fidelity references prepared by the dataset authors rather than measurements, and AUIED3K inputs are synthetic degradations of real photographs.

• The Sea-thru analysis uses at most 100 images per site and evenly spaced frames, which are strongly correlated within a site; confidence intervals are over images and may be optimistic.

• The severity analysis is post hoc and exploratory, and input PSNR is only a proxy for physical degradation.

• Loss weights were not re-tuned per variant, and no external state-of-the-art method was compared; we make no claim on absolute competitiveness.

Recommendations. For studies of physics-guided restoration we suggest (a) reporting several seeds with paired intervals instead of one run; (b) evaluating zero-shot on complete external sets rather than on the selection split; (c) including a capacity-matched control without the physical structure; (d) validating predicted physical maps against measured range, with a null control and a classical baseline, before calling them physical; (e) reporting data-eficiency curves, where a prior is most likely to matter; and (f) verifying gradient flow through every loss term before ablating it.

## 7 Reproduction note and implementation audit

Re-running the published configuration. Re-training the published single-seed configuration (seed 42, identical split, code after the corrections below) on diferent hardware (RTX 5080 instead of RTX 5060) and with persistent data-loader workers gave 23.31 dB PSNR on UIEB-val and 21.53 dB on the 400-image LSUI subset used in [15], against the reported 22.97 dB and 21.20 dB. The run stopped by early stopping at epoch 64 (best epoch 49). A diference of about +0.3 dB between two nominally identical runs is of the same size as most of the efects examined in Section 5, which is why we do not interpret single-run ablations.

Defects that invalidate ablations. A code-level audit of the earlier implementation found five issues that must be excluded before a loss ablation means anything; all are corrected in the version used here and in [15]:

1. the frozen task encoder and the VGG feature extractor were wrapped in torch.no\_grad(), so $\mathcal { L } _ { \mathrm { t a s k } }$ and $\mathcal { L } _ { \mathrm { p e r c } }$ produced no gradient with respect to the restored image and could not influence training;

2. the custom SSIM implementation could return values above 1, giving negative loss terms;

3. the frozen task encoder was executed at inference although it is training-only;

4. ablation variants were not re-seeded before model construction, so their initial weights difered;

5. the evaluation script instantiated learned baselines without loading their trained weights.

A sanity check now asserts that the restored image requires gradients and that every loss term back-propagates to it before any training run is started. We recommend such a check for any study that ablates loss terms computed on frozen feature extractors.

## 8 Conclusion

We asked whether the physics in a physics-guided underwater restoration network does any work, and whether it is physics. Under controlled conditions the answer is nuanced. With ample data the physics module brings no zero-shot accuracy benefit and costs 3.4× the FLOPs. With scarce data it helps, and a capacity-matched control without the formation model does not reproduce this, so the benefit is tied to the physics-shaped structure (a consistent trend, though not individually significant with five seeds). Yet the predicted transmission does not follow measured range, and the efect of the module changes sign with degradation severity between datasets. The physics here is an inductive bias, not a measurement. The practical consequence is twofold: a physical prior is worth its cost when data are scarce, and the maps it produces should not be described, or used, as physical quantities without validation against range. More generally, single-run ablations on small benchmarks can reverse in sign; multi-seed, zero-shot reporting, capacity-matched controls and range validation are inexpensive safeguards that we hope become routine.

Data and code availability. UIEB, LSUI, AUIED3K and Sea-thru are public datasets [9, 16, 5, 13].   
Code, configurations and per-image results are available from the corresponding author on request.

## References

[1] B. L. McGlamery, “A computer model for underwater camera systems,” in Proc. SPIE, vol. 208, 1980, pp. 221–231, doi:10.1117/12.958279.

[2] J. S. Jafe, “Computer modeling and the design of optimal underwater imaging systems,” IEEE J. Ocean. Eng., vol. 15, no. 2, pp. 101–111, 1990, doi:10.1109/48.50695.

[3] D. Akkaynak and T. Treibitz, “A revised underwater image formation model,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2018, pp. 6723–6732, doi:10.1109/CVPR.2018.00703.

[4] C. Li, S. Anwar, J. Hou, R. Cong, C. Guo, and W. Ren, “Underwater image enhancement via medium transmission-guided multi-color space embedding,” IEEE Trans. Image Process., vol. 30, pp. 4985–5000, 2021, doi:10.1109/TIP.2021.3076367.

[5] P. Saini, N. Singh, and A. S. Anand, “AUIED3K: A new Andaman underwater image enhancement dataset for deep learning-driven image enhancement with minimum loss dehazing,” in Proc. IEEE Int. Conf. Acoust., Speech Signal Process. (ICASSP), 2025, pp. 1–5, doi:10.1109/ICASSP49660.2025.10890845.

[6] C. Li, S. Anwar, and F. Porikli, “Underwater scene prior inspired deep underwater image and video enhancement,” Pattern Recognit., vol. 98, Art. no. 107038, 2020, doi:10.1016/j.patcog.2019.107038.

[7] M. Raissi, P. Perdikaris, and G. E. Karniadakis, “Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial diferential equations,” J. Comput. Phys., vol. 378, pp. 686–707, 2019, doi:10.1016/j.jcp.2018.10.045.

[8] G. E. Karniadakis, I. G. Kevrekidis, L. Lu, P. Perdikaris, S. Wang, and L. Yang, “Physics-informed machine learning,” Nat. Rev. Phys., vol. 3, no. 6, pp. 422–440, 2021, doi:10.1038/s42254-021-00314-5.

[9] C. Li et al., “An underwater image enhancement benchmark dataset and beyond,” IEEE Trans. Image Process., vol. 29, pp. 4376–4389, 2020, doi:10.1109/TIP.2019.2955241.

[10] P. Henderson, R. Islam, P. Bachman, J. Pineau, D. Precup, and D. Meger, “Deep reinforcement learning that matters,” in Proc. AAAI Conf. Artif. Intell., vol. 32, no. 1, 2018, doi:10.1609/aaai.v32i1.11694.

[11] X. Bouthillier et al., “Accounting for variance in machine learning benchmarks,” arXiv:2103.03098, 2021.

[12] D. Picard, “Torch.manual\_seed(3407) is all you need: On the influence of random seeds in deep learning architectures for computer vision,” arXiv:2109.08203, 2021.

[13] D. Akkaynak and T. Treibitz, “Sea-thru: A method for removing water from underwater images,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2019, pp. 1682–1691, doi:10.1109/CVPR.2019.00178.

[14] C. Sun, A. Shrivastava, S. Singh, and A. Gupta, “Revisiting unreasonable efectiveness of data in deep learning era,” in Proc. IEEE Int. Conf. Comput. Vis. (ICCV), 2017, pp. 843–852, doi:10.1109/ICCV.2017.97.

[15] M. R. Fauzan, I. Surya, J. Kustija, R. Pramudita, and R. K. Ahsani, “PhysGuideNet: A physics-guided deep network for underwater image restoration with task-preserving loss,” Inform: Jurnal Ilmiah Bidang Teknologi Informasi dan Komunikasi, vol. 11, no. 2, pp. 140–149, 2026, doi:10.25139/inform.v11i2.11890.

[16] L. Peng, C. Zhu, and L. Bian, “U-shape Transformer for underwater image enhancement,” IEEE Trans. Image Process., vol. 32, pp. 3066–3079, 2023, doi:10.1109/TIP.2023.3276332.

[17] K. He, J. Sun, and X. Tang, “Single image haze removal using dark channel prior,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 33, no. 12, pp. 2341–2353, 2011, doi:10.1109/TPAMI.2010.168.

[18] J. Y. Chiang and Y. C. Chen, “Underwater image enhancement by wavelength compensation and dehazing,” IEEE Trans. Image Process., vol. 21, no. 4, pp. 1756–1769, 2012, doi:10.1109/TIP.2011.2179666.

[19] A. Galdran, D. Pardo, A. Picón, and A. Alvarez-Gila, “Automatic red-channel underwater image restoration,” J. Vis. Commun. Image Represent., vol. 26, pp. 132–145, 2015, doi:10.1016/j.jvcir.2014.11.006.

[20] N. Carlevaris-Bianco, A. Mohan, and R. M. Eustice, “Initial results in underwater single image dehazing,” in Proc. OCEANS MTS/IEEE Seattle, 2010, pp. 1–8, doi:10.1109/OCEANS.2010.5664428.

[21] Y. T. Peng and P. C. Cosman, “Underwater image restoration based on image blurriness and light absorption,” IEEE Trans. Image Process., vol. 26, no. 4, pp. 1579–1594, 2017, doi:10.1109/TIP.2017.2663846.

[22] P. L. J. Drews, E. R. Nascimento, S. S. C. Botelho, and M. F. Montenegro Campos, “Underwater depth estimation and image restoration based on single images,” IEEE Comput. Graph. Appl., vol. 36, no. 2, pp. 24–35, 2016, doi:10.1109/MCG.2016.26.

[23] D. Berman, D. Levy, S. Avidan, and T. Treibitz, “Underwater single image color restoration using haze-lines and a new quantitative dataset,” IEEE Trans. Pattern Anal. Mach. Intell., 2020 (early access), doi:10.1109/TPAMI.2020.2977624.

[24] C. O. Ancuti, C. Ancuti, C. De Vleeschouwer, and P. Bekaert, “Color balance and fusion for underwater image enhancement,” IEEE Trans. Image Process., vol. 27, no. 1, pp. 379–393, 2018, doi:10.1109/TIP.2017.2759252.

[25] C. Li, J. Guo, and C. Guo, “Emerging from water: Underwater image color correction based on weakly supervised color transfer,” IEEE Signal Process. Lett., vol. 25, no. 3, pp. 323–327, 2018, doi:10.1109/LSP.2018.2792050.

[26] C. Fabbri, M. J. Islam, and J. Sattar, “Enhancing underwater imagery using generative adversarial networks,” in Proc. IEEE Int. Conf. Robot. Autom. (ICRA), 2018, pp. 7159–7165, doi:10.1109/ICRA.2018.8460552.

[27] Y. Guo, H. Li, and P. Zhuang, “Underwater image enhancement using a multiscale dense generative adversarial network,” IEEE J. Ocean. Eng., vol. 45, no. 3, pp. 862–870, 2020, doi:10.1109/JOE.2019.2911447.

[28] M. J. Islam, Y. Xia, and J. Sattar, “Fast underwater image enhancement for improved visual perception,” IEEE Robot. Autom. Lett., vol. 5, no. 2, pp. 3227–3234, 2020, doi:10.1109/LRA.2020.2974710.

[29] R. Liu, Z. Jiang, S. Yang, and X. Fan, “Twin adversarial contrastive learning for underwater image enhancement and beyond,” IEEE Trans. Image Process., vol. 31, pp. 4922–4936, 2022, doi:10.1109/TIP.2022.3190209.

[30] S. Huang, K. Wang, H. Liu, J. Chen, and Y. Li, “Contrastive semi-supervised learning for underwater image restoration via reliable bank,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2023, pp. 18145–18155, doi:10.1109/CVPR52729.2023.01740.

[31] S. Anwar and C. Li, “Diving deeper into underwater image enhancement: A survey,” Signal Process., Image Commun., vol. 89, Art. no. 115978, 2020, doi:10.1016/j.image.2020.115978.

[32] Y. Wang, W. Song, G. Fortino, L. Z. Qi, W. Zhang, and A. Liotta, “An experimental-based review of image enhancement and image restoration methods for underwater imaging,” IEEE Access, vol. 7, pp. 140233–140251, 2019, doi:10.1109/ACCESS.2019.2932130.

[33] R. Liu, X. Fan, M. Zhu, M. Hou, and Z. Luo, “Real-world underwater enhancement: Challenges, benchmarks, and solutions under natural light,” IEEE Trans. Circuits Syst. Video Technol., vol. 30, no. 12, pp. 4861–4875, 2020, doi:10.1109/TCSVT.2019.2963772.

[34] J. Dodge, S. Gururangan, D. Card, R. Schwartz, and N. A. Smith, “Show your work: Improved reporting of experimental results,” in Proc. Conf. Empirical Methods Natural Lang. Process. (EMNLP), 2019, pp. 2185–2194, doi:10.18653/v1/D19-1224.

[35] J. Pineau et al., “Improving reproducibility in machine learning research (a report from the NeurIPS 2019 reproducibility program),” J. Mach. Learn. Res., vol. 22, no. 164, pp. 1–20, 2021.

[36] J. Demšar, “Statistical comparisons of classifiers over multiple data sets,” J. Mach. Learn. Res., vol. 7, pp. 1–30, 2006.

[37] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, “Image quality assessment: From error visibility to structural similarity,” IEEE Trans. Image Process., vol. 13, no. 4, pp. 600–612, 2004, doi:10.1109/TIP.2003.819861.

[38] K. Panetta, C. Gao, and S. Agaian, “Human-visual-system-inspired underwater image quality measures,” IEEE J. Ocean. Eng., vol. 41, no. 3, pp. 541–551, 2016, doi:10.1109/JOE.2015.2469915.

[39] M. Yang and A. Sowmya, “An underwater color image quality evaluation metric,” IEEE Trans. Image Process., vol. 24, no. 12, pp. 6062–6071, 2015, doi:10.1109/TIP.2015.2491020.

[40] S. D. Hordley and G. D. Finlayson, “Re-evaluating colour constancy algorithms,” in Proc. 17th Int. Conf. Pattern Recognit. (ICPR), vol. 1, 2004, pp. 76–79, doi:10.1109/ICPR.2004.1334009.

[41] A. Gijsenij, T. Gevers, and J. van de Weijer, “Computational color constancy: Survey and experiments,” IEEE Trans. Image Process., vol. 20, no. 9, pp. 2475–2489, 2011, doi:10.1109/TIP.2011.2118224.

[42] S. Ben-David, J. Blitzer, K. Crammer, A. Kulesza, F. Pereira, and J. W. Vaughan, “A theory of learning from diferent domains,” Mach. Learn., vol. 79, nos. 1–2, pp. 151–175, 2010, doi:10.1007/s10994-009-5152-4.

[43] S. Woo, J. Park, J. Y. Lee, and I. S. Kweon, “CBAM: Convolutional block attention module,” in Proc. Eur. Conf. Comput. Vis. (ECCV), 2018, pp. 3–19, doi:10.1007/978-3-030-01234-2\_1.

[44] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2016, pp. 770–778, doi:10.1109/CVPR.2016.90.

[45] K. Simonyan and A. Zisserman, “Very deep convolutional networks for large-scale image recognition,” in Proc. Int. Conf. Learn. Represent. (ICLR), 2015.

[46] J. Johnson, A. Alahi, and L. Fei-Fei, “Perceptual losses for real-time style transfer and super-resolution,” in Proc. Eur. Conf. Comput. Vis. (ECCV), 2016, pp. 694–711, doi:10.1007/978-3-319-46475-6\_43.

[47] I. Loshchilov and F. Hutter, “Decoupled weight decay regularization,” in Proc. Int. Conf. Learn. Represent. (ICLR), 2019.

[48] F. Wilcoxon, “Individual comparisons by ranking methods,” Biometrics Bull., vol. 1, no. 6, pp. 80–83, 1945, doi:10.2307/3001968.

[49] S. Holm, “A simple sequentially rejective multiple test procedure,” Scand. J. Statist., vol. 6, no. 2, pp. 65–70, 1979.