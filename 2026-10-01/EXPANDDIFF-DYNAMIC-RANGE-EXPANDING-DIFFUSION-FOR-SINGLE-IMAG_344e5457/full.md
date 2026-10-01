# EXPANDDIFF: DYNAMIC RANGE EXPANDING DIFFUSION FOR SINGLE-IMAGE HDR RECONSTRUCTION

Mehmet Emre Andıran<sup>∗</sup>, Zhuoqian Yang, Liying Lu, Mathieu Salzmann, Sabine Süsstrunk

School of Computer and Communication Sciences, EPFL, Switzerland firstname.lastname@epfl.ch, <sup>∗</sup>memreandiran@gmail.com

## ABSTRACT

Single-image HDR reconstruction requires inferring missing detail while preserving the visible content of an LDR image. Differences in sensor dynamic range and exposure cause LDR images to lose varying amounts of information in shadows and highlights. We present ExpandDiff, a conditional diffusion pipeline that jointly reconstructs clipped shadows and highlights. To account for this variation, we introduce Dynamic Clipping Synthesis (DCS), which randomly samples shadow and highlight clipping percentiles when constructing training inputs from HDR targets. A pixel-space diffusion model guided by spatially-adaptive normalization then predicts perceptually encoded HDR through a bounded output head, reconstructing both clipping directions in one sampling trajectory. On the SI-HDR benchmark, ExpandDiff variants improve HDR reconstruction accuracy by 3.43 dB in PU21-PSNR over the strongest evaluated competing method, and by 7.34 dB under two-sided clipping. The code and supplementary material are available at https://memreandiran.github.io/expanddiff/.

Index Terms— High dynamic range expansion, inverse tone mapping, diffusion models, high dynamic range reconstruction

## 1. INTRODUCTION

Typical camera sensors capture 8 to 12 stops of dynamic range [1], while real scenes can span 20 stops or more [2]. The sensor’s dynamic range determines the span of scene intensities that can be recorded, while exposure determines where this span lies. LDR images captured with different sensors and exposures can therefore lose different amounts of information in shadows, highlights, or both. Single-image HDR reconstruction must accommodate this variation, inferring plausible detail in regions where information is lost while preserving visible image content. This inference is ill-posed because the missing detail cannot be uniquely determined from the LDR image. Earlier methods emphasize saturated highlights [1, 3] or reconstruct HDR through intermediate exposure brackets [4–6]. We directly predict one HDR image, jointly restoring shadows and highlights under varying clipping levels.

We present Dynamic Range Expanding Diffusion (ExpandDiff), a pipeline that couples variable clipping during training, spatial LDR guidance, and perceptually encoded HDR prediction. How much a capture is clipped varies from image to image, and the model is not told it at inference. To account for variation in clipping across real captures, we introduce Dynamic Clipping Synthesis (DCS): a training strategy that randomly samples shadow and highlight clipping levels when constructing LDR guidance from HDR targets. DCS exposes a single model to varying amounts of information loss at both ends of the dynamic range, encouraging reconstruction across diverse clipping conditions. We adapt the RGB-guided RAW-Diffusion backbone [7] to predict HDR RGB images. The backbone uses spatially-adaptive normalization (SPADE) [8] to inject LDR features into the diffusion network, so our model conditions the diffusion process in pixel space rather than through the conditioning latent of [9] or the compressed latent space of [6, 10, 11].

Finally, to construct the training target, we encode the HDR target with PU21 [12], a perceptually uniform encoding of absolute luminance in which equal differences in its units are equally visible. This allocates more of the bounded output range to dark intensities and prevents the loss function from getting dominated by highlights. The model retains Gaussian diffusion training and reconstructs shadows and highlights jointly in a single 24-step sampling trajectory.

Our contribution is the integrated dynamic range expansion pipeline, rather than any single component, and its evaluation across clipping distributions. We evaluate on the 181-scene SI-HDR benchmark [13], along with additional degradations applied to the same HDR references. The benchmark-oriented variant, ExpandDiff-B, achieves the best gain-aligned PU21-PSNR among the evaluated methods, exceeding the strongest competing method by a paired mean of 3.43 dB. ExpandDiff-P, trained with DCS, ranks second on that metric and exceeds the strongest competing method by 7.34 dB under two-sided clipping.

## 2. RELATED WORK

Single-image HDR reconstruction. Eilertsen et al. [1] reconstruct saturated highlights with a CNN. ExpandNet [14] combines features at multiple spatial scales, Santos et al. [3] use feature masking and perceptual supervision, and Liu et al. [15] learn to reverse the camera pipeline. Other methods synthesize exposure brackets and merge them into HDR [4, 5]. These approaches establish synthetic degradation and recovery of under- and overexposure as training objectives. DCS samples shadow and highlight clipping levels in percentile space, controlling the fraction of affected pixels across scenes. We couple this training distribution with spatially guided diffusion and a perceptual prediction domain.

Diffusion-based inverse tone mapping. Dalal et al. [9] condition diffusion on an encoded latent of the input for LDR-to-HDR conversion, while Goswami et al. [10] incorporate semantic guidance. LEDiff [6] fuses exposure brackets from separate highlight and shadow denoisers in a pretrained LDR latent space. ExpandDiff instead predicts one HDR image in pixel space, reconstructing both clipping directions in the same sampling trajectory. ExpoCM [16] also handles different exposure regimes jointly, using exposureaware consistency trajectories for one-step inference.

Further discussion of related works is provided in the supplementary material.

![](images/8555f9bf8b76496a2453989df716c2662c382b69e3d8eab4f0732d00e688a920.jpg)  
Fig. 1. ExpandDiff pipeline. Dynamic Clipping Synthesis varies the clipping levels of the LDR guidance during training, while PU21 encoding defines the HDR prediction domain. Spatial guidance conditions the diffusion network, and inverse encoding yields linear HDR.

## 3. METHOD

Fig. 1 summarizes ExpandDiff. We first construct a PU21 encoded HDR target and use Dynamic Clipping Synthesis (DCS) to synthesize its LDR guidance. A conditional diffusion model then learns to recover the perceptually encoded target. At inference, the LDR image guides a single sampling trajectory, and the predicted image is decoded to linear HDR.

## 3.1. Dynamic Clipping Synthesis

The clipping level of a real input is unknown at inference, and it differs with the exposure, the scene, and the camera. DCS varies the amount of information removed from shadows and highlights as training examples are constructed. This enables a single checkpoint to perform dynamic range expansion even when it receives images with varying degrees of degradation as input. For each HDR target Y , we synthesize the linear guidance image

$$
I = \frac { \mathrm { c l i p } ( Y , \tau _ { \mathrm { l o } } , \tau _ { \mathrm { h i } } ) - \tau _ { \mathrm { l o } } } { \tau _ { \mathrm { h i } } - \tau _ { \mathrm { l o } } } .\tag{1}
$$

For each training example, DCS draws the clipping percentiles $q _ { \mathrm { l o } } \sim \mathcal { U } [ 0 , 1 0 ]$ and $q _ { \mathrm { h i } } \sim \mathcal { U } [ 0 , 3 0 ]$ , in percent. $\tau _ { \mathrm { l o } }$ is the $q _ { \mathrm { l o } }$ percentile of the per-pixel channel minima, and $\tau _ { \mathrm { h i } }$ is the $( 1 0 0 - q _ { \mathrm { h i } } )$ percentile of the per-pixel channel maxima. The same scalar thresholds apply to all channels. The percentiles control the fraction of affected pixels, allowing the training distribution to span nearunclipped through heavily clipped inputs across scenes with different intensity distributions. Neither the thresholds nor clipping masks are supplied to the network, so it must infer the required expansion from the LDR image itself.

For data, we use HDR images from Poly Haven, the public Laval photometric sample [17], HDR-Real [15], and Fairchild [2]. We form $5 1 2 \times 5 1 2$ images by center-cropping ordinary images and extracting five random perspective views from each panorama. Each resulting image is rescaled to a median luminance of $\mathrm { 2 0 c d / m ^ { 2 } }$ and clipped at a peak of $L _ { \mathrm { m a x } } ~ = ~ 1 0 0 0 { \mathrm { c d / m ^ { 2 } } }$ Dividing by this peak gives a target $Y \in [ 0 , 1 ] ^ { 3 \times 5 1 2 \times 5 1 2 }$ . This defines a consistent display-referred output scale.

## 3.2. PU21 Prediction and the Bounded Head

PU21 is a perceptually uniform encoding of absolute luminance [12], where equal differences in its units are equally visible. We predict in the PU21 space and use it to supervise our model to exploit this property. PU21 is important for supervision, as in linear radiance, a perceptually equal error is numerically far larger in the highlights than in the shadows, so a pixelwise distance loss would be dominated by bright pixels. On PU21 values, each error instead counts according to its visibility.

Let U denote the PU21 encoding of absolute linear intensity [12]. We apply it per channel and normalize at the target peak:

$$
V ( Y ) = \frac { U ( L _ { \mathrm { m a x } } Y ) } { U ( L _ { \mathrm { m a x } } ) } , \qquad x _ { 0 } = 2 V ( Y ) - 1 .\tag{2}
$$

The network’s tanh head bounds xˆ<sub>0</sub> to $[ - 1 , 1 ]$ . We decode the final prediction as $\hat { Y } = V ^ { - 1 } ( ( \hat { x } _ { 0 } + 1 ) / 2 )$ ; the guidance I remains linear.

This encoding changes the operating range of the tanh output head. With a linear target spanning 0 to 1000 cd/m<sup>2</sup>, an intensity of $\mathrm { 0 . 5 c d / m ^ { 2 } }$ maps ${ \mathrm { t o ~ - } } 0 . 9 9 9 $ , near the saturated tail of tanh. PU21 assigns more of the output range to dark intensities, placing them where tanh is more responsive, which improves the head’s sensitivity to shadow errors. We test the effects of the encoding and head jointly in Sec. 4.5.

## 3.3. Spatially Conditioned Diffusion Backbone

HDR reconstruction must preserve visible structure while inferring content in clipped regions, motivating a pixel-space diffusion backbone with spatial LDR guidance at multiple resolutions. We adapt RAW-Diffusion’s conditional U-Net [7] from RAW Bayer prediction to HDR RGB prediction. An EDSR encoder [18], without its upsampling head, extracts a full-resolution guidance feature map. SPADE layers [8] resample these features to each bottleneck and decoder resolution and predict spatial scale $( \gamma )$ and shift (β) maps. For a groupnormalized activation h, conditioning gives $( 1 + \gamma ( I ) ) \odot h + \beta ( I )$ Thus, image structure is available throughout reconstruction without compressing the HDR prediction into a VAE latent space.

The U-Net has six resolution levels, channel widths from 32 to 128, two residual blocks per level, and self-attention at the two coarsest resolutions and the bottleneck. The guidance encoder uses four residual blocks with 64 channels. ExpandDiff has 25.0 M parameters. We use 24 DDIM steps [19]; both clipping directions are reconstructed by the same network and trajectory. An 8-bit inference input is linearized with the inverse of the sRGB transfer function, regardless of its original transfer function, and mapped into the range [0, 1]. The resulting image is approximately linear, even though the actual camera response is unknown.

Let $x _ { 0 } = 2 V ( Y ) – 1$ denote the normalized PU21 target, defined in Sec. 3.2. We use the standard Gaussian forward process,

$$
\begin{array} { r } { x _ { t } = \sqrt { \bar { \alpha } _ { t } } x _ { 0 } + \sqrt { 1 - \bar { \alpha } _ { t } } \epsilon , \epsilon \sim \mathcal { N } ( 0 , \mathbf { I } ) , } \end{array}\tag{3}
$$

where $\bar { \alpha } _ { t }$ is the cumulative signal-retention factor. Following RAW-Diffusion [7], the network predicts the clean target, xˆ<sub>0</sub> = $f _ { \theta } ( x _ { t } , t , I )$ ). Its objective is

$$
\mathcal { L } = \mathbb { E } \bigg [ \frac { \mathrm { M A E } ( \hat { x } _ { 0 } , x _ { 0 } ) + \mathrm { M S E } ( \hat { x } _ { 0 } , x _ { 0 } ) } { 1 + \mathrm { S N R } ( t ) } \bigg ] ,\tag{4}
$$

with $\mathrm { S N R } ( t ) = \bar { \alpha } _ { t } / ( 1 - \bar { \alpha } _ { t } )$ . We retain equally weighted absolute and squared errors, but omit RAW-Diffusion’s logarithmic error term because our target is already perceptually encoded.

## 4. EXPERIMENTS

We compare ExpandDiff with published models on SI-HDR [13], then test transfer across clipping levels and directions. A factorial ablation examines PU21 encoding and the bounded prediction head.

## 4.1. Evaluation Setup

Data and conditions. SI-HDR provides 181 scenes with 1280 × 1888 HDR references. We center-crop and box-filter them to 512 × 512, the input size used for all methods, as LEDiff [6] only accepts this size. We evaluate the benchmark’s supplied $\mathtt { c l i p \_ 9 5 }$ inputs $\mathrm { ( C _ { 9 5 } ) }$ after the same crop and resize, in which 8.17% of pixels have a channel at the ceiling and none at zero.

To test reconstruction under degradation at both ends, we construct $\mathrm { C _ { p } }$ from the same references with Eq. (1) at fixed $q _ { \mathrm { l o } } = 5 ,$ $q _ { \mathrm { h i } } = 1 5$ , the midpoints of ${ \bf P } { \bf \bar { s } }$ training ranges. On average, 15.40% of pixels are highlight-clipped, and 5.87% are shadow-clipped. Every image contains both types of clipping. Thus, $\mathrm { C _ { 9 5 } }$ tests highlight reconstruction under the benchmark pipeline, while $\mathrm { C _ { p } }$ tests joint shadow and highlight reconstruction within the clipping ranges used by DCS. We also evaluate doubled clipping percentiles $( \mathrm { C } _ { \mathrm { p } } ^ { \mathrm { h a r d } } )$ and LEDiff’s independently defined degradation (C<sub>0</sub>) [6, 20]; details appear in the supplementary material.

Models. ExpandDiff-P is trained with DCS (Sec. 3.1); the P suffix denotes its percentile-based clipping distribution. ExpandDiff-B uses a benchmark-oriented degradation with a highlight-only clip at $q _ { \mathrm { h i } } \sim \mathcal { U } [ 3 , 7 ]$ , an 8-bit round trip, and a randomized two-parameter response curve. They share data sources, architecture, target encoding, loss, and batch size 32, but are trained for 150k and 75k steps at learning rates $2 \times 1 0 ^ { - 4 }$ and $1 0 ^ { - 4 }$ , respectively. We compare with ExpandNet [14], MaskHDR [3], the highlight and shadow variants of LEDiff (HL and SH) [6], and Refusion-HDR and DITM from AIM 2025 [21], giving six configurations across five methods. We run the released models and include the unprocessed LDR input, a baseline many SI-HDR methods fail to improve on [13]. All methods receive the same 8-bit input for each condition, and all scores are computed in the same evaluation pipeline.

## 4.2. Metrics and Alignment

We report PU21-PSNR, PU21-VSI, and PU21-PIQE, following the SI-HDR evaluation study [13], together with HDR-VDP-3 [22] (v3.0.7, quality task, in JOD). ExpandDiff predictions are decoded to linear HDR first, so every method is encoded identically. We also report FID [23] on Reinhard-tone-mapped crops [24], denoted FID-R, following LEDiff [6]. PSNR, VSI and HDR-VDP-3 measure agreement with the reference, whereas PIQE is no-reference and FID-R compares crop statistics, so neither necessarily ranks methods by reconstruction accuracy. Crop sampling details are given in the supplementary material.

To account for global brightness differences between predictions and the reference, we fit a scalar gain $\begin{array} { r } { s = \sum _ { k \in \Omega } p _ { k } g _ { k } \Big / \sum _ { k \in \Omega } p _ { k } ^ { 2 } , } \end{array}$ where $p$ is the prediction, g the reference, and Ω contains RGB samples at input pixels clipped at neither end. All metrics use the aligned prediction sp. We additionally report full-reference metrics after SI-HDR’s ground-truth-fitted 20-parameter tone-and-color correction, labeled CRF [13]. The difference between the two columns shows how much error a more flexible tone-and-color fit reduces. These are evaluation-time alignments, applied independently to every method. Reported PSNR margins are paired means and can differ slightly from subtraction of rounded table entries.

Table 1. Results on 181 SI-HDR references under two input conditions. PSNR, VSI, and PIQE use PU21; VDP-3 is in JOD. All scores include brightness alignment; CRF additionally applies the benchmark’s tone-and-color correction. FID denotes FID-R. Best values per condition are bold and second-best values underlined.
<table><tr><td rowspan="2">Method</td><td>PSNR ↑</td><td></td><td>VSI↑</td><td>VDP-3↑</td><td></td><td>PIQE FID</td></tr><tr><td>gain CRF</td><td>gain</td><td>CRF</td><td>gain CRF</td><td>↓</td><td>↓</td></tr><tr><td> $\mathrm { C _ { 9 5 } }$ </td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ExpandDiff-B 22.61 30.15</td><td></td><td></td><td>5.9866 .9897 9.06 9.46</td><td></td><td>28.22</td><td>8.76</td></tr><tr><td>ExpandDiff-P 21.41 28.65</td><td></td><td>.9801.9867</td><td></td><td>8.779.35</td><td>27.32</td><td>9.99</td></tr><tr><td>LDR input</td><td>20.47 27.93</td><td>3.9756.9826</td><td></td><td>8.44 9.15</td><td></td><td>28.1510.00</td></tr><tr><td>ExpandNet</td><td>19.17 28.09.9743.9840</td><td></td><td></td><td></td><td></td><td>08.37 9.12 28.60 10.65</td></tr><tr><td>DITM</td><td>19.06 27.64.9767 .9857 8.52 9.0030.43 10.70</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>LEDiff-HL</td><td>18.95 26.45 .9722 .9803 8.33 8.97 29.17 12.98</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>MaskHDR</td><td>18.77 28.51 .9767.9874</td><td></td><td></td><td>8.59</td><td>9.37</td><td>29.2011.92</td></tr><tr><td>LEDiff-SH</td><td>18.75 26.16.9708.9781</td><td></td><td></td><td>8.33</td><td>8.84</td><td>27.7612.19</td></tr><tr><td>Refusion-HDR16.85 27.77</td><td></td><td></td><td>.9555.9804</td><td>8.10</td><td>9.08 45.24</td><td>9.58</td></tr><tr><td> $\mathrm { C _ { p } }$ </td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ExpandDiff-P 31.21 33.52 .9913 .9917 9.62 9.68 39.31 4.78</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>MaskHDR</td><td>23.8828.72.9806.9884</td><td></td><td></td><td>9.289.65</td><td>38.78</td><td>7.88</td></tr><tr><td>ExpandDiff-B 23.13 28.27 .9754 .9845</td><td></td><td></td><td></td><td>58.98 9.42</td><td>239.06</td><td>8.99</td></tr><tr><td>ExpandNet</td><td>22.74 28.34.9781.98288.769.20</td><td></td><td></td><td></td><td>40.17</td><td>6.89</td></tr><tr><td>LDR input</td><td>22.23 27.37 .9699.9754 8.49</td><td></td><td></td><td></td><td>9.00 40.11</td><td>6.75</td></tr><tr><td>LEDiff-HL</td><td>20.99 25.18.9633.9771</td><td></td><td></td><td>8.29</td><td>8.90 37.81</td><td>12.41</td></tr><tr><td>DITM</td><td>20.50 25.98.9727.9799</td><td></td><td></td><td>98.36 8.79</td><td></td><td>36.8311.66</td></tr><tr><td>Refusion-HDR19.46 25.82.9629.97578</td><td></td><td></td><td></td><td>8.34 9.04</td><td></td><td>444.2411.89</td></tr><tr><td>LEDiff-SH</td><td>19.14 25.37.9659.97338.458.8037.909.73</td><td></td><td></td><td></td><td></td><td></td></tr></table>

## 4.3. SI-HDR Benchmark

On $\mathrm { C _ { 9 5 } }$ (Table 1), ExpandDiff-B reaches 22.61 dB gain-aligned PU21-PSNR, exceeding ExpandNet, the strongest competing model on this metric, by a paired 3.43 dB. ExpandDiff-P ranks second at 21.41 dB despite its different training degradation, with a paired advantage of 2.24 dB over ExpandNet. B also leads gain-aligned VSI and HDR-VDP-3, while P obtains the lowest PIQE. Together, the two variants attain the best values across all eight columns.

The advantage persists after CRF correction: ExpandDiff-B exceeds the strongest competing model by 1.64 dB PSNR, 0.0023 VSI, and 0.09 JOD. ExpandDiff-P exceeds all competitors in corrected PSNR, although MaskHDR scores slightly higher in corrected VSI and HDR-VDP-3. The smaller PSNR margin after CRF correction indicates that tone and color differences contribute to B’s gain-only advantage, while its remaining lead supports an improvement beyond these correctable differences. The unprocessed input outperforms every competing model in gain-aligned PSNR, but both of our variants exceed it, and P improves on it in all eight columns. Since most input pixels remain unclipped, the strong input baseline emphasizes the need to preserve visible content while reconstructing missing information. The reported whole-image scores measure both effects. Fig. 2 shows two highlight-clipped examples. Additional qualitative results are in the supplementary material.

## 4.4. Clipping Severity and Distribution Transfer

On $\mathrm { C _ { p } }$ (Table 1), ExpandDiff-P achieves 31.21 dB gain-aligned PSNR, a paired improvement of 7.34 dB over MaskHDR, and ranks first in every column except PIQE. Its HDR-VDP-3 and FID-R margins over the best competing models are 0.34 JOD and 2.11, respectively. It also leads all three CRF-corrected metrics. The agreement across reference-based metrics supports improved fidelity under simultaneous shadow and highlight loss. DITM obtains a lower PIQE despite substantially lower PSNR, illustrating that no-reference quality and reconstruction accuracy can favor different outputs. Fig. 3 illustrates reconstruction on two examples when shadows and highlights are both clipped.

![](images/0c3029170a1944bb86602b38b68ca997a2e147495597aa92874e8a0916b3fa88.jpg)  
Fig. 2. Highlight reconstruction on two $\mathrm { C _ { 9 5 } }$ inputs with 8.3% and 6.9% clipped highlights. Red marks highlight clipping. Each example occupies two rows. Reconstructions use gain-only alignment.

As reported in the supplementary material, increasing the clipping percentiles to (10, 30) reduces ${ \bf P } { \bf s }$ PSNR advantage to 4.76 dB, while it keeps a 0.33 JOD advantage in HDR-VDP-3. On LEDiff’s independently defined degradation, B leads the strongest competing model by 2.44 dB PSNR and 0.30 JOD. The harder clipping condition tests P at the upper endpoints of its training ranges, while LEDiff’s degradation tests a different input synthesis pipeline. The retained margins show that the gains are not confined to the midpoint clipping condition.

The P/B ranking reverses across input distributions: B leads P on seven metrics at $\mathrm { C _ { 9 5 } } .$ , and P leads B on seven at $\mathrm { C _ { p } } .$ . The comparison at their reported checkpoints also differs in training duration and learning rate. A matched-budget check at 75k steps gives P 21.38 dB PSNR on $\mathrm { C _ { 9 5 } }$ , still below B’s 22.61 dB. This supports matching the training degradation to the benchmark, while P’s second place demonstrates transfer from the DCS distribution.

## 4.5. Target Encoding and Output Head

We train all four combinations of linear or PU21 targets and bounded or unbounded heads for 150k steps, holding the remaining P configuration fixed. Table 2 shows that removing PU21 reduces gainaligned PSNR by 2.00 dB, removing tanh by 1.01 dB, and removing both by 4.37 dB. The full model also leads the CRF-corrected fidelity metrics. Each component improves PSNR with either setting of the other component, supporting both choices within this architecture. Their combined benefit is consistent with the prediction-range argument in Sec. 3.2, although the ablation does not separate the effects of perceptual error weighting and output sensitivity. This ben-

![](images/a2914fdc593f2760a545a3e10a3ade632be1ac3095e8a27fe083b6911846d611.jpg)  
Fig. 3. Joint shadow and highlight reconstruction on two $\mathrm { C _ { p } }$ inputs, clipped over 20.0% and 20.5% of the frame. Red and blue mark clipped highlights and shadows, respectively. Each example occupies two rows. Reconstructions use gain-only alignment.

Table 2. PU21 encoding and the bounded head on $\mathrm { C _ { P } }$ . All variants use the same data and 150k training steps. Metric definitions and emphasis follow Table 1.
<table><tr><td rowspan="2">Method</td><td>PSNR ↑</td><td>VSI↑</td><td></td><td>VDP-3 ↑</td><td>PIQE FID</td><td></td></tr><tr><td>gain CRF</td><td>gain</td><td>CRF</td><td>gain CRF</td><td>↓</td><td>↓</td></tr><tr><td>ExpandDiff-P 31.21 33.52</td><td></td><td>.9913</td><td>.9917</td><td>9.62</td><td>9.68</td><td>39.31 4.78</td></tr><tr><td>PU21, no tanh 30.20 33.08</td><td></td><td>.9905</td><td>.9910</td><td>9.59 9.67</td><td>39.46</td><td>4.77</td></tr><tr><td>linear, tanh</td><td>29.21 31.38</td><td>.9890.9897</td><td></td><td>9.51 9.60</td><td>40.04</td><td>6.03</td></tr><tr><td>linear, no tanh 26.84 30.34 .9861 .9880</td><td></td><td></td><td></td><td>9.46 9.61</td><td>38.16</td><td>5.56</td></tr></table>

efit is metric-dependent: the unbounded PU21 model has a slightly lower FID-R, and the unbounded linear model has the lowest PIQE. Inference cost. ExpandDiff takes 3.19 s per $5 1 2 \times 5 1 2$ image on an NVIDIA A100 80 GB with 24 DDIM steps. The fully convolutional network accepts other image sizes after reflection padding to multiples of 64, subject to available memory. Increasing sampling to 48 or 100 steps gives no clear PSNR improvement as reported in the supplementary material.

## 5. CONCLUSION

We introduced ExpandDiff, which couples Dynamic Clipping Synthesis, spatial LDR guidance, and perceptually encoded diffusion for joint shadow and highlight reconstruction. Its results across SI-HDR and additional clipping conditions show that a broadly sampled training distribution can transfer beyond its nominal degradation, while a benchmark-oriented distribution can further improve matched performance. The target ablation supports using PU21 with a bounded head for reconstruction fidelity. Like other methods, ExpandDiff cannot know the scene’s absolute brightness, and its reconstructions weaken over large contiguous clipped regions. Further limitations are discussed in the supplementary material.

## 6. REFERENCES

[1] Gabriel Eilertsen, Joel Kronander, Gyorgy Denes, Rafał K. Mantiuk, and Jonas Unger, “HDR image reconstruction from a single exposure using deep CNNs,” ACM Trans. Graph., vol. 36, no. 6, Nov. 2017.

[2] Mark D. Fairchild, “The HDR photographic survey,” Color and Imaging Conference, vol. 15, no. 1, pp. 233–238, 2007.

[3] Marcel Santana Santos, Tsang Ing Ren, and Nima Khademi Kalantari, “Single image HDR reconstruction using a CNN with masked features and perceptual loss,” ACM Trans. Graph., vol. 39, no. 4, Aug. 2020.

[4] Yuki Endo, Yoshihiro Kanamori, and Jun Mitani, “Deep reverse tone mapping,” ACM Trans. Graph., vol. 36, no. 6, Nov. 2017.

[5] Siyeong Lee, Gwon Hwan An, and Suk-Ju Kang, “Deep recursive HDRI: Inverse tone mapping using generative adversarial networks,” in Computer Vision – ECCV 2018: 15th European Conference, Munich, Germany, September 8-14, 2018, Proceedings, Part II, Berlin, Heidelberg, 2018, p. 613–628, Springer-Verlag.

[6] Chao Wang, Zhihao Xia, Thomas Leimkuhler, Karol Myszkowski, and Xuaner Zhang, “LEDiff: Latent exposure diffusion for HDR generation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2025, pp. 453–464.

[7] Christoph Reinders, Radu Berdan, Beril Besbinar, Junji Otsuka, and Daisuke Iso, “RAW-Diffusion: RGB-guided diffusion models for high-fidelity RAW image generation,” in 2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 2025, pp. 8431–8443.

[8] Taesung Park, Ming-Yu Liu, Ting-Chun Wang, and Jun-Yan Zhu, “Semantic image synthesis with spatially-adaptive normalization,” in 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2019, pp. 2332– 2341.

[9] Dwip Dalal, Gautam Vashishtha, Prajwal Singh, and Shanmuganathan Raman, “Single image LDR to HDR conversion using conditional diffusion,” in 2023 IEEE International Conference on Image Processing (ICIP), 2023, pp. 3533–3537.

[10] Abhishek Goswami, Aru Ranjan Singh, Francesco Banterle, Kurt Debattista, and Thomas Bashford-Rogers, “Semantic aware diffusion inverse tone mapping,” Journal of Physics: Conference Series, vol. 3128, no. 1, pp. 012009, Oct. 2025.

[11] Ronghuan Wu, Wanchao Su, Kede Ma, Jing Liao, and Rafał K. Mantiuk, “X2HDR: HDR image generation in a perceptually uniform space,” arXiv:2602.04814, 2026.

[12] Rafał K. Mantiuk and Maryam Azimi, “PU21: A novel perceptually uniform encoding for adapting existing quality metrics for HDR,” in 2021 Picture Coding Symposium (PCS), June 2021, pp. 1–5.

[13] Param Hanji, Rafal Mantiuk, Gabriel Eilertsen, Saghi Hajisharif, and Jonas Unger, “Comparison of single image HDR reconstruction methods — the caveats of quality assessment,” in ACM SIGGRAPH 2022 Conference Proceedings, New York, NY, USA, 2022, SIGGRAPH ’22, Association for Computing Machinery.

[14] D. Marnerides, T. Bashford-Rogers, J. Hatchett, and K. Debattista, “ExpandNet: A deep convolutional neural network for

high dynamic range expansion from low dynamic range content,” Computer Graphics Forum, vol. 37, no. 2, pp. 37–49, 2018.

[15] Yu-Lun Liu, Wei-Sheng Lai, Yu-Sheng Chen, Yi-Lung Kao, Ming-Hsuan Yang, Yung-Yu Chuang, and Jia-Bin Huang, “Single-image HDR reconstruction by learning to reverse the camera pipeline,” in 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020, pp. 1648–1657.

[16] Aoyu Liu, Zhen Liu, Ziyi Wang, Dian Chen, Bing Zeng, and Shuaicheng Liu, “ExpoCM: Exposure-aware one-step generative single-image HDR reconstruction,” arXiv:2605.02464, 2026.

[17] Christophe Bolduc, Justine Giroux, Marc Hébert, Claude Demers, and Jean-François Lalonde, “Beyond the pixel: a photometrically calibrated HDR dataset for luminance and color prediction,” in IEEE/CVF International Conference on Computer Vision (ICCV), 2023.

[18] Bee Lim, Sanghyun Son, Heewon Kim, Seungjun Nah, and Kyoung Mu Lee, “Enhanced deep residual networks for single image super-resolution,” in 2017 IEEE Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), 2017, pp. 1132–1140.

[19] Jiaming Song, Chenlin Meng, and Stefano Ermon, “Denoising diffusion implicit models,” in International Conference on Learning Representations, 2021.

[20] Pontus Andersson, Jim Nilsson, Peter Shirley, and Tomas Akenine-Möller, “Visualizing Errors in Rendered High Dynamic Range Images,” in Eurographics 2021 - Short Papers, Holger Theisel and Michael Wimmer, Eds. 2021, pp. 25–28, The Eurographics Association.

[21] Chao Wang et al., “AIM 2025 challenge on inverse tone mapping report: Methods and results,” arXiv:2508.13479, 2025.

[22] Rafał K. Mantiuk, Dounia Hammou, and Param Hanji, “HDR-VDP-3: A multi-metric for predicting image differences, quality and contrast distortions in high dynamic range and regular content,” arXiv:2304.13625, 2023.

[23] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter, “GANs trained by a two time-scale update rule converge to a local Nash equilibrium,” in Advances in Neural Information Processing Systems, I. Guyon, U. Von Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett, Eds. 2017, vol. 30, Curran Associates, Inc.

[24] Erik Reinhard, Michael Stark, Peter Shirley, and James Ferwerda, “Photographic tone reproduction for digital images,” ACM Trans. Graph., vol. 21, no. 3, pp. 267–276, July 2002.