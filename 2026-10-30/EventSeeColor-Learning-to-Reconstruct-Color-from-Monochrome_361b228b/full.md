# EventSeeColor: Learning to Reconstruct Color from Monochrome Events

Chuanzhi Xu<sup>1,\*</sup> Langyi Chen<sup>1</sup> Haoxian Zhou<sup>1</sup> Zelin Zhang<sup>1</sup> Haodong Chen<sup>1</sup> Qiang Qu<sup>1</sup> Xiaoming Chen<sup>2</sup> Weidong Cai<sup>1</sup>

<sup>1</sup>The University of Sydney <sup>2</sup>Beijing Technology and Business University Project Page

![](images/4a7c18b80cd82ea8f918515161333f3eca5284c7d2ff308d02dbffc0a246ef7f.jpg)  
Figure 1. Color reconstruction from monochrome events showing foliage, road markings, and vehicles. EventSeeColor uses reconstructed grayscale and aligned events to recover scene appearance with more faithful colors and finer details than the baselines.

## Abstract

Event cameras capture visual brightness changes under challenging lighting conditions. However, traditional event-based image reconstruction targets grayscale intensity and produces images without scene color. Color distinguishes surfaces with similar intensity. Reconstructing color from events can reveal appearance differences that reconstructed grayscale images cannot express. Although monochrome event cameras do not directly measure color, their event-triggering patterns depend on surface reflectance, illumination, and the sensor’s spectral response. These patterns may contain cues that support color reconstruction. In this paper, we present EventSeeColor, which uses event-reconstructed grayscale and aligned events to jointly guide a diffusion colorizer. Grayscale-derived semantic context supports color inference, while region-balanced chroma supervision and local RGB refinement improve color distribution, brightness, and texture. In the combined evaluation on two real event datasets, EventSeeColor achieves the best structural similarity, learned perceptual scores, and chromatic-region color difference among the external methods.

## 1. Introduction

Event cameras asynchronously record changes in light intensity, providing high temporal resolution and a wide dynamic range [15], which support visual sensing during rapid motion and challenging lighting conditions. To make such observations accessible to people and frame-based vision models, traditional event-based image reconstruction converts sparse temporal measurements into grayscale intensity images [35, 46]. These images recover scene structure but omit color differences that can be important for interpreting objects and surfaces [52]. For example, differently colored road markings or vehicle surfaces may have similar grayscale intensity. As shown in Figure 1, reconstructing their color can reveal appearance distinctions that grayscale reconstruction cannot express, enriching the scene information from monochrome event sensing.

The event formation process helps explain the connection between event activity and scene color. Although monochrome event cameras do not directly measure color, their measured intensity depends on incident light and the sensor’s spectral response [2]. As motion and illumination change this intensity, its temporal variation determines event timing and polarity [15]. Therefore, event activity is connected to the image formation process that determines scene color. This relationship suggests that event-triggering patterns may contain useful cues for color inference. It motivates a central question: how can temporal event cues and reconstructed spatial structure jointly guide color reconstructionfrom ordinary monochrome recordings?

Image colorization models provide an intuitive starting point by inferring plausible colors from the semantic content of a grayscale image [26, 65]. Applying these models to an event-reconstructed grayscale image offers a straightforward approach to color reconstruction. However, colorizers that rely solely on semantic cues from event-reconstructed grayscale without access to temporal event information may struggle to reconstruct color faithfully, as shown in Figure 1.

Recent event-driven diffusion models also use generative priors to reconstruct color videos from event observations [7, 29, 32]. These methods reconstruct complete video frames by jointly generating scene structure and color. Sparse events do not fully determine scene appearance, and generative priors can introduce hallucinated details and reconstruction artifacts [7, 32]. Moreover, existing methods do not explicitly examine whether aligned events improve color accuracy beyond the spatial and semantic guidance of reconstructed grayscale images.

In this paper, we present EventSeeColor, to our knowledge the first framework for event-guided colorization of event-reconstructed grayscale images, with three core components. Event and Grayscale Fusion combines spatial and semantic cues from reconstructed grayscale with temporal cues from aligned events to guide a diffusion colorizer. Balanced Chroma Refinement supports regional color recovery, and Local Detail Refinement corrects brightness, color, and texture in the rendered RGB image. All scenedependent inference inputs originate from the monochrome event stream. We evaluate EventSeeColor against image colorizers and direct event-to-video methods on paired real event recordings from PEOD [11] and CoSEC [42]. In the combined evaluation, EventSeeColor achieves the best structural similarity, perceptual fidelity, and chromaticregion color accuracy. Ablation studies show how aligned events, the color prior, and local RGB refinement contribute to color reconstruction. Our contributions are summarized as follows:

• We propose EventSeeColor, to our knowledge the first framework for event-guided colorization of eventreconstructed grayscale images, jointly using the spatial and semantic content of reconstructed grayscale images and temporal cues from aligned monochrome events.

• We develop a sequential refinement pipeline that combines regional chroma supervision with local RGB correction to improve color placement, preserve neutral appearance, and reduce reconstruction artifacts.

• Experiments show EventSeeColor’s strong structural, perceptual, and chromatic-region performance against external colorization and event reconstruction methods.

## 2. Related Work

Grayscale Reconstruction. Traditional event-based image reconstruction estimates grayscale intensity from monochrome events. E2VID [46] and V2V-E2VID [35] learn this mapping from event sequences, while Event-Diffusion [33] and temporal residual diffusion [69] combine initial intensity reconstructions with event information for diffusion-based restoration. However, both groups focus on intensity recovery and do not address color differences between surfaces with similar grayscale intensity. EventSee-Color builds on the reconstructed spatial structure and uses aligned events to guide color inference, extending the recovered scene appearance beyond grayscale intensity.

Event-Based Color and Spectral Sensing. Eventbased color and spectral sensing introduces wavelengthdependent measurements through sensor, optical, or illumination design. Color-filtered events are used in RGBW sensor studies [41], the CED dataset [49], and Color4E [36]. Active RGB illumination [10], E-RGB-D [38], and active hyperspectral imaging [63] combine event cameras with controlled illumination, while chromatic optics enable event-based spectral discrimination [2]. However, ordinary monochrome event recordings lack these additional spectral measurements. This is also one motivation for EventSee-Color: reconstructing color from monochrome events without additional spectral measurements at inference.

Grayscale Image Colorization. Grayscale image colorization infers chroma from grayscale images and learned appearance priors. Colorful Image Colorization [65], DD-Color [26], ColorFormer [25], and $\mathrm { C T ^ { 2 } }$ [57] learn image representations for chroma prediction. BigColor [27] and DeOldify [1] use generative priors, while UniColor [23] and DISCO [58] incorporate appearance guidance. Palette [48], L-CAD [6], and Control Color [34] formulate colorization through diffusion. These methods can take eventreconstructed grayscale images as input and colorize them. However, their colorizers do not directly access event timing, polarity, or activity. Therefore, plausible colors inferred from image semantics may differ from the observed scene. EventSeeColor supplies aligned-event features as additional color guidance and uses regional chroma supervision to recover colored surfaces while preserving neutral appearance. RGB Video Reconstruction from Events. RGB video reconstruction includes methods with and without external color references. Pini et al. [43], Continuity-Cam [56], IE2Video [53], and Engram-E2VID [24] reconstruct color videos from events and RGB reference images. E2VIDiff [32], LaSe-E2V [7], EvDiff [29], and UniE2F [62] instead use diffusion priors to reconstruct RGB frames without an external color reference. LongE2V extends event-driven diffusion to long-horizon reconstruction, prediction, and interpolation [14]. However, generating scene structure and color from sparse events can introduce hallucinated details and artifacts [7, 32]. These methods also do not directly evaluate whether events improve grayscale image colorization. EventSeeColor instead uses aligned events to guide grayscale image colorization and evaluates their impact on color accuracy.

![](images/e33a8acb409de7332d837ed23722e379caaa0ffae2d86a92c59df6a89a92e176.jpg)  
Figure 2. EventSeeColor architecture. (a) Overall pipeline. (b) EGF fuses grayscale and event features. (c) BCR refines chroma and renders intermediate RGB. (d) LDR predicts a bounded RGB residual. (e) The diffusion colorizer uses ControlNet for spatial guidance and low-rank adaptation (LoRA) to adapt text cross-attention. EGF, BCR, and LDR apply validity masks before encoding.

## 3. Methodology

As shown in Figure 2, EventSeeColor reconstructs a color image $\widehat { I } _ { t }$ from a monochrome event stream at time t.

E2VID [46] provides a normalized grayscale image $G _ { t } ,$ and the corresponding event interval provides an aligned tensor $V _ { t } .$ A validity mask $m _ { t }$ identifies pixels with reliable alignment. Event and Grayscale Fusion (EGF) supplies a spatial condition to a diffusion colorizer. Balanced Chroma Refinement (BCR) adjusts the generated color and renders an intermediate RGB image $I _ { t } ^ { r }$ using calibrated event lightness. Finally, Local Detail Refinement (LDR) predicts a local RGB correction to refine scene appearance. All scenedependent inference inputs originate from the event stream. RGB images provide training supervision and evaluation targets. Appendix B.1 specifies reconstruction, event en-

coding, and calibration.

## 3.1. Event and Grayscale Fusion

A grayscale reconstruction provides a spatial reference for color placement but does not explicitly retain event timing and polarity. EGF combines grayscale structure with the aligned event tensor at multiple scales so that temporal activity can guide color inference. Let $F _ { \ell } ^ { G }$ and $F _ { \ell } ^ { E }$ be grayscale and event features at scale ℓ. The activity map $A _ { t }$ averages the normalized positive and negative event counts, and max pooling its nonzero support gives $S _ { \ell }$ . A learned gate selects event features using both inputs:

$$
\widetilde { F } _ { \ell } ^ { E } = \sigma \big ( Q _ { \ell } [ F _ { \ell } ^ { G } , F _ { \ell } ^ { E } ] \big ) \odot F _ { \ell } ^ { E } \odot S _ { \ell } ,\tag{1}
$$

where $Q _ { \ell }$ is a learned projection, σ denotes sigmoid, brackets denote channel concatenation, and ⊙ denotes elementwise multiplication. The support map places event conditioning at observed activity, while the gate controls its contribution using grayscale context.

The multiscale features are pooled to the resolution of the pretrained grayscale hint $H _ { t } ^ { 0 }$ and projected into its feature space:

$$
\begin{array} { r l } & { H _ { t } = H _ { t } ^ { 0 } + P _ { G } [ \mathcal { P } ( F _ { 1 } ^ { G } ) , \dots , \mathcal { P } ( F _ { K } ^ { G } ) ] } \\ & { ~ + P _ { E } [ \mathcal { P } ( \widetilde { F } _ { 1 } ^ { E } ) , \dots , \mathcal { P } ( \widetilde { F } _ { K } ^ { E } ) ] , } \end{array}\tag{2}
$$

where $\mathcal { P }$ is adaptive average pooling, K is the number of scales, and $P _ { G }$ and $P _ { E }$ are learned projections. Both projections start at zero, preserving the pretrained hint at initialization. The event projection has no bias, so locations without pooled event support receive only grayscale features. The fused hint guides the Control Color latent diffusion backbone through ControlNet [34, 47, 64]. Appendix B.2 details the fusion architecture.

A fixed photographic prompt π, shared across all images, supplies the text condition during training and inference. Appendix B.2 gives the fixed prompt settings.

## 3.2. Balanced Chroma Refinement

Diffusion can produce plausible colors with incorrect regional strength or hue. Increasing saturation alone can also introduce false color into neutral roads, buildings, and white objects. BCR adjusts local hue and chroma to recover colored surfaces while preserving neutral appearance. Let $\mathbf { c } _ { t } ^ { 0 } = ( a _ { t } ^ { 0 } , b _ { t } ^ { 0 } )$ be the generated Lab chroma after gamut mapping at calibrated event lightness $L _ { t } .$ . BCR receives $G _ { t }$ $V _ { t } , \mathbf { c } _ { t } ^ { 0 }$ , its magnitude, and $m _ { t }$ . Separate image and event encoders supply features to a joint context network. Two heads then predict a channelwise gain $\pmb { \alpha } _ { t }$ and a chroma residual $\mathbf { r } _ { t } \mathbf { : }$

$$
\mathbf { c } _ { t } ^ { r } = \mathbf { c } _ { t } ^ { 0 } + m _ { t } \odot \left[ \left( { \alpha } _ { t } - 1 \right) \odot { \mathbf { c } } _ { t } ^ { 0 } + { \mathbf { r } } _ { t } \right] .\tag{3}
$$

The independent gains adjust the strength of the a and b channels, and the residual corrects local hue and chroma. Bounded heads begin with an identity mapping through zero initialization. The intermediate image is $\begin{array} { r l } { I _ { t } ^ { r } } & { { } = } \end{array}$ $\mathcal { T } ( L _ { t } , \Pi _ { L _ { t } } ( \mathbf { c } _ { t } ^ { r } ) )$ , where $\Pi _ { L _ { t } }$ scales out-of-gamut chroma into the RGB gamut and $\tau$ converts Lab to RGB. This intermediate rendering anchors color to reconstructed intensity structure before local RGB refinement. Appendix B.3 gives the parameterization.

## 3.3. Local Detail Refinement

Correcting chroma alone cannot remove brightness errors or texture artifacts inherited from grayscale reconstruction. Therefore, LDR learns a local RGB correction using $I _ { t } ^ { r } , G _ { t }$ $A _ { t }$ , and $m _ { t }$ . Shallow features preserve fine structures, while a downsampled context branch provides a wider receptive field. Their concatenation produces a bounded residual and the final image:

$$
\begin{array} { r l } & { \Delta I _ { t } = \delta \operatorname { t a n h } f _ { \psi } ( [ I _ { t } ^ { r } , G _ { t } , A _ { t } , m _ { t } ] \odot m _ { t } ) , } \\ & { \quad \widetilde { I } _ { t } = I _ { t } ^ { r } + m _ { t } \odot \Delta I _ { t } , } \\ & { \quad \widehat { I } _ { t } = \operatorname { c l i p } ( \widetilde { I } _ { t } , 0 , 1 ) , } \end{array}\tag{4}
$$

where $f _ { \psi }$ denotes the local network with parameters ψ and δ bounds each RGB channel correction. Its final convolution is initialized to zero, so training starts from $I _ { t } ^ { r }$ . Direct RGB refinement lets the output jointly adjust local lightness, texture, and color. The final image is evaluated with its predicted RGB values. Appendix B.4 details the network.

## 3.4. Training Objectives

Training follows the inference stages. Diffusion adaptation updates EGF, a training-only auxiliary chroma head, and low-rank adaptation (LoRA) parameters in ControlNet text cross-attention [21]. The pretrained weights remain fixed. The diffusion colorizer objective combines valid-area denoising error ${ \mathcal { L } } _ { \mathrm { d i f f } }$ with Auxiliary Chroma Supervision $\mathcal { L } _ { \mathrm { a u x } } \mathrm { : }$

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { g e n } } = \mathcal { L } _ { \mathrm { d i f f } } + \lambda _ { \mathrm { a u x } } \mathcal { L } _ { \mathrm { a u x } } . } \end{array}\tag{5}
$$

The auxiliary head predicts coarse target chroma from $H _ { t }$ connecting the fused representation to the color reconstruction task. Appendix $\mathrm { { B } } . 5$ defines both terms.

Large neutral regions can dominate an image-wide chroma loss and weaken supervision for smaller colored surfaces. With the diffusion colorizer frozen, BCR uses a Region-Balanced Chroma Objective to give each target region direct supervision. Let $\ell _ { j i }$ be the normalized chroma regression error at pixel i of image j and $w _ { j i }$ its target-chroma weight. The valid pixel set $\Omega _ { j }$ is partitioned into neutral, transitional, and chromatic regions. For the nonempty regions ${ \mathcal { R } } _ { j }$ and batch size B, the global and re gion terms are:

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { g l o b a l } } = \frac { \sum _ { j } \sum _ { i \in \Omega _ { j } } w _ { j i } \ell _ { j i } } { \sum _ { j } | \Omega _ { j } | } , } \\ & { \mathcal { L } _ { \mathrm { r e g i o n } } = \cfrac { 1 } { B } \sum _ { j } \cfrac { 1 } { | \mathcal { R } _ { j } | } \sum _ { r \in \mathcal { R } _ { j } } \cfrac { \sum _ { i \in \Omega _ { j r } } w _ { j i } \ell _ { j i } } { | \Omega _ { j r } | } , } \end{array}\tag{6}
$$

where $\Omega _ { j r }$ denotes region $r$ in image j. Equal region weighting prevents large regions from dominating smaller colored surfaces. Chroma shortfall, neutral-chroma supervision, and warm-region supervision complete the objective:

$$
\begin{array} { r l } & { { \mathcal { L } } _ { \mathrm { B C R } } = ( 1 - \beta ) { \mathcal { L } } _ { \mathrm { g l o b a l } } + \beta { \mathcal { L } } _ { \mathrm { r e g i o n } } } \\ & { ~ + \lambda _ { s } { \mathcal { L } } _ { \mathrm { s h o r t } } + \lambda _ { n } { \mathcal { L } } _ { \mathrm { n e u t r a l } } + \lambda _ { w } { \mathcal { L } } _ { \mathrm { w a r m } } . } \end{array}\tag{7}
$$

The regional terms preserve weak coloration while giving colored surfaces direct supervision. All region assignments use training targets and are absent from inference. Their definitions and reductions appear in Appendix B.5.

Finally, the diffusion colorizer and BCR are frozen while LDR learns from paired RGB targets. RGB reconstruction and gradient losses supervise local appearance and edges. Region-balanced chroma regression maintains color accuracy, the LPIPS loss [66] supervises perceptual similarity, and a clipping penalty discourages out-of-range predictions:

$$
\begin{array} { r l } & { { \mathcal { L } } _ { \mathrm { L D R } } = \lambda _ { \mathrm { r g b } } { \mathcal { L } } _ { \mathrm { r g b } } + \lambda _ { \mathrm { g r a d } } { \mathcal { L } } _ { \mathrm { g r a d } } + \lambda _ { \mathrm { c o l o r } } { \mathcal { L } } _ { \mathrm { c o l o r } } } \\ & { ~ + \lambda _ { \mathrm { p e r c } } { \mathcal { L } } _ { \mathrm { p e r c } } + \lambda _ { \mathrm { c l i p } } { \mathcal { L } } _ { \mathrm { c l i p } } . } \end{array}\tag{8}
$$

Appendix B.6 gives the exact objectives. All numerical architecture choices, bounds, and loss weights are reported with the training settings in Appendix D.

## 4. Dataset Construction and Curation

PEOD [11] and CoSEC [42] are event-RGB datasets developed for object detection and autonomous driving, respectively. However, neither dataset was designed specifically for supervised color reconstruction from monochrome events. Some RGB targets contain overexposed or underexposed regions that obscure surface color and texture (Figure 3) and can encourage the model to learn exposure artifacts. Therefore, we construct and curate paired data from both datasets for reliable color supervision.

Paired Data Construction. Each sample contains a $2 5 6 \times 2 5 6$ RGB image crop, aligned events, and an eventreconstructed grayscale image. PEOD uses a fixed temporal offset and sequence-level geometric correction, while CoSEC uses coaxially aligned RGB and event coordinates. Validity masks restrict supervision to aligned regions.

Data Curation. For each crop sampled on a grid, we compare a short event accumulation with edges extracted from its RGB target. Crops with large residual offsets or weak edge correspondence are rejected. We also measure nearwhite, near-black, bright, and dark pixel fractions within each crop’s valid region. Crops with excessive clipping or large bright or dark areas are excluded from training and evaluation. Retained targets keep their original RGB values. Appendix C.5 details sampling geometry, alignment criteria, exposure thresholds, and screening counts.

![](images/fe86e6ec941997808a132c0f4e6a58892e7e8f406405f1395ff31f866a49e6b2.jpg)  
Figure 3. Excluded RGB targets with their corresponding 50 ms event accumulations. Each pair uses the same crop and target time. RGB values are displayed without exposure adjustment.

Retained Data. The training split contains 8,068 crops from 1,755 target frames. Evaluation uses 2,220 crops from 491 target frames. All methods share the same retained evaluation targets and valid regions.

## 5. Experiments and Results

## 5.1. Experimental Setup

Implementation Details. EventSeeColor uses aligned $2 5 6 \times 2 5 6$ crops and $5 1 2 \times 5 1 2$ inputs to the diffusion colorizer, with LoRA rank set to 4. The auxiliary chroma loss weight is $\lambda _ { \mathrm { a u x } } = 0 . 1$ . BCR uses region-balancing coefficient $\beta = 0 . 5$ and warm-region supervision weight $\lambda _ { w } =$ 0.2, while LDR limits its RGB correction to $\delta = 0 . 1 2$ . Inference uses 20 deterministic DDIM steps with classifierfree guidance scale 7. Training and inference are performed on a single NVIDIA GeForce RTX 5090 GPU with 32 GB of memory. EGF, LoRA, BCR, and LDR add 133,216, 181,248, 32,340, and 24,451 inference parameters, respectively. These additions total 371,255 parameters, increasing the Control Color backbone’s parameter count by only 0.0244%. Appendix D.1 provides the remaining network and optimization settings.

Table 1. Quantitative comparison with external baselines on PEOD and CoSEC. Chr. denotes chromatic target regions, and Desat. reports the percentage of chromatic target pixels reconstructed as near-neutral. CF-MAE measures colorfulness error relative to the RGB targets.
<table><tr><td>Method</td><td colspan="10">|PSNR ↑ SSIM ↑|LPIPS ↓ DISTS ↓|Chr. ∆E00 ↓ Desat. (%) ↓ CF-MAE ↓|NIQE ↓ FID↓</td></tr><tr><td colspan="13">Image colorization from reconstructed grayscale</td></tr><tr><td colspan="13">Colorful Image Colorization (ECCV, 2016) [65] 14.306 0.5185 0.5210 0.3073</td></tr><tr><td>DDColor (ICCV, 2023) [26]</td><td>13.705</td><td>0.4919</td><td>0.5224</td><td>0.3081</td><td>22.876 24.317</td><td>11.114 11.492</td><td>19.054 26.032</td><td>6.766 6.814</td><td>207.114 175.876</td><td></td><td>32.24 227.87</td></tr><tr><td>ColorFormer (ECCV, 2022) [25]</td><td>13.992</td><td>0.4862</td><td>0.4820</td><td>0.2975</td><td>21.798</td><td></td><td>18.978</td><td>18.210</td><td>6.850</td><td>138.680</td><td>44.82</td></tr><tr><td></td><td>14.380</td><td>0.4887</td><td>0.4865</td><td>0.2985</td><td>22.365</td><td>17.408</td><td></td><td>18.944</td><td>6.730</td><td>138.394</td><td>585.51</td></tr><tr><td>BigColor (ECCV, 2022) [27]</td><td>14.214</td><td>0.5962</td><td>0.4501</td><td>0.2881</td><td>20.841</td><td>22.304</td><td></td><td></td><td></td><td></td><td>1523.61</td></tr><tr><td>Control Color (IJCV, 2025) [34] L-CAD (NeurIPS, 2023) [6]</td><td>13.529</td><td>0.4858</td><td>0.5128</td><td>0.3113</td><td>24.229</td><td></td><td></td><td>20.150</td><td>7.222</td><td>156.302</td><td></td></tr><tr><td>DeOldify (GitHub, 2018) [1]</td><td>14.483</td><td>0.5199</td><td>0.4726</td><td>0.3031</td><td>21.341</td><td></td><td>16.662 25.813</td><td>25.378</td><td>6.653 6.764</td><td>147.627 181.201</td><td>1103.61</td></tr><tr><td>CT2 (EČCV, 2022) [57]</td><td>14.314</td><td>0.5118</td><td>0.4734</td><td>0.3018</td><td>21.166</td><td></td><td>20.504</td><td>11.383</td><td></td><td></td><td>218.22</td></tr><tr><td>DISCO (SIGGRAPH Asia, 2022) [58]</td><td>14.229</td><td>0.5143</td><td>0.4783</td><td>0.3058</td><td>20.763</td><td></td><td>67.306</td><td>14.845</td><td>6.763</td><td>164.058</td><td>462.98</td></tr><tr><td></td><td></td><td></td><td>0.5024</td><td>0.3076</td><td>22.931</td><td></td><td></td><td>15.962</td><td>6.764</td><td>155.947</td><td>43.06</td></tr><tr><td>UniColor (SIGGRAPH Asia, 2022) [23]</td><td>13.601</td><td>0.4759</td><td></td><td></td><td></td><td></td><td>20.601</td><td>25.192</td><td>6.768</td><td>142.251</td><td>190.73</td></tr><tr><td>Grayscale reconstruction followed by colorization</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>V2V-E2VID + DDColor</td><td></td><td></td><td>0.4567</td><td>0.3035</td><td>24.404</td><td></td><td>10.766</td><td></td><td></td><td></td><td></td></tr><tr><td>(NeurIPS, 2025 / ICCV, 2023) [26, 35]</td><td>14.4530.4967</td><td></td><td></td><td></td><td></td><td></td><td></td><td>20.360</td><td></td><td>8.331 150.953</td><td>238.58</td></tr><tr><td>Event video reconstruction</td><td></td><td></td><td>0.3856</td><td>0.3027</td><td>22.572</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>EvDiff (ECCV, 2026) [29]</td><td>15.807 15.752</td><td>0.5336 0.5488</td><td>0.4721</td><td>0.3150</td><td>20.440</td><td></td><td>55.410 97.685</td><td>6.837</td><td>9.186 10.561 185.881</td><td>102.178</td><td>2223.51</td></tr><tr><td>LongE2V (SIGGRAPH, 2026) [14]</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>9.614</td><td></td><td></td><td>10548.57</td></tr><tr><td>EventSeeColor (Ours)</td><td>16.010</td><td>0.6443</td><td>0.3329</td><td>0.2589</td><td>15.767</td><td></td><td>36.603</td><td>5.494</td><td>6.326</td><td>117.286</td><td>1523.98</td></tr></table>

Evaluation Protocol. All methods are evaluated in their native RGB output space on the curated data in Section 4. PSNR and SSIM measure image fidelity. LPIPS and DISTS [12] measure perceptual agreement, while chromatic CIEDE2000 (∆E<sub>00</sub>) [50] evaluates target regions with substantial color. Desaturation (Desat.) measures the fraction of chromatic target pixels reconstructed as nearneutral. Colorfulness MAE (CF-MAE) [19] measures percrop colorfulness error, NIQE [40] measures natural luminance statistics, and Frechet Inception Distance (FID) [´ 20] compares output and target distributions. Appendix C.3 defines paired metrics and aggregation, and Appendix C.4 specifies global appearance and distribution metrics.

Comparison Baselines. Table 1 lists the external baselines in our comparison, organized into three categories. The first category comprises image colorization methods that receive the same normalized event-reconstructed grayscale images. For the second category, we design and construct a baseline pipeline that feeds grayscale images reconstructed by V2V-E2VID [35] into DDColor for colorization, evaluating an alternative grayscale front end. The third category comprises event video reconstruction methods with RGB frame generation capabilities, including EvDiff [29] and LongE2V [14]. We retain their temporal inference procedures and extract RGB frames corresponding to the evaluation target times. The same spatial alignment and crop rules then yield individual color-image reconstructions for comparison. Appendix D.2 specifies these input and deployment settings.

## 5.2. Comparison with Existing Methods

Quantitative Comparison. Table 1 shows that EventSee-Color achieves the best PSNR, SSIM, LPIPS, DISTS, and chromatic CIEDE2000 among the compared methods on the combined evaluation set. In particular, EventSeeColor outperforms Control Color, which provides its pretrained diffusion backbone, in image fidelity, perceptual quality, and chromatic-region color accuracy. The lowest CF-MAE further indicates closer overall colorfulness to the targets, and the lowest NIQE indicates closer agreement with natural luminance statistics. EventSeeColor also achieves the second-lowest FID, indicating close agreement between the feature distributions of its reconstructions and real targets. Appendix E provides additional evaluation metrics.

Qualitative Comparison. Figure 4 compares EventSee-Color with all external methods across vegetation, buildings, roads, and vehicles. Figure 5 shows how EventSee-Color preserves distinct appearance across warm concrete, foliage, sky, and neutral road surfaces. EventSeeColor recovers foliage color while reducing false coloration on road and vehicle surfaces. Visually, EventSeeColor shows the closest color agreement with the RGB targets in the chromatic regions of these examples. In contrast, LongE2V produces nearly achromatic outputs on many colored surfaces. Appendix E.2 provides complementary regional color accuracy and desaturation metrics. Appendix E.4 provides additional qualitative comparisons of color and detail reconstruction.

Color Recovery Analysis. Figure 6 reports EventSee-Color’s regional CIEDE2000 by target color. Green regions have the lowest error, followed by blue and yellow among the chromatic groups.

![](images/987be26ceb5266d7970eb8c8325a6eb430ce8c017dfbbe9bfb182134e4c56766.jpg)  
Figure 4. Qualitative comparison with the external baselines in Table 1. EventSeeColor recovers more faithful surface colors and details while reducing false coloration on neutral regions across vegetation, buildings, roads, and vehicles.

Table 2. Ablation study of EventSeeColor’s components, event guidance, and training objectives on the evaluation set. Metrics and parameter-counting scope follow Table 1. Within each section, best and second-best quality scores are bold and underlined, respectively.
<table><tr><td>Configuration</td><td>|PSNR ↑ SSIM ↑|LPIPS ↓ DISTS ↓|Chr. ∆E00 ↓ Desat. (%) ↓ CF-MAE ↓|NIQE ↓</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>FID↓</td><td>|Params. (M)</td></tr><tr><td colspan="9">Components and event guidance</td></tr><tr><td>w/o EGF Event Path</td><td>15.981 0.6415</td><td></td><td>0.3371</td><td>0.2619</td><td>15.906</td><td>34.511</td><td>5.597</td><td></td><td>6.414 121.923</td><td>1523.91</td></tr><tr><td>w/o BCR</td><td>16.042</td><td>0.6439</td><td>0.3478</td><td>0.2638</td><td>16.398</td><td>45.884</td><td>6.383</td><td>6.416</td><td>100.750</td><td>1523.95</td></tr><tr><td>w/o LDR</td><td>15.056</td><td>0.6187</td><td>0.4113</td><td>0.3104</td><td>16.360</td><td>23.602</td><td>6.208</td><td>6.424</td><td>151.809</td><td>1523.96</td></tr><tr><td>w/o Event Inputs (EGF/BCR/LDR)</td><td>16.080</td><td>0.6357</td><td>0.3506</td><td>0.2617</td><td>16.229</td><td>40.126</td><td>5.618</td><td>6.333</td><td>131.985</td><td>1523.90</td></tr><tr><td>EventSeeColor (full)</td><td>16.0100.6443</td><td></td><td>0.3329</td><td>0.2589</td><td>15.767</td><td>36.603</td><td>5.494</td><td>6.326117.286|</td><td></td><td>1523.98</td></tr><tr><td colspan="9">Training objectives</td><td></td><td></td></tr><tr><td>w/o Warm Supervision</td><td>16.030 0.6355</td><td></td><td>0.3531</td><td>0.2639</td><td>15.987</td><td>34.800</td><td>5.597</td><td>6.357 134.445</td><td></td><td>1523.98</td></tr><tr><td>w/o Auxiliary Chroma Supervision</td><td>16.009</td><td>0.6421</td><td>0.3371</td><td>0.2608</td><td>15.821</td><td>34.201</td><td>5.539</td><td></td><td>6.377 120.752</td><td>1523.98</td></tr><tr><td>w/o LDR Perceptual Supervision</td><td>16.218</td><td>0.6530</td><td>0.3562</td><td>0.2665</td><td>15.583</td><td>38.858</td><td>5.592</td><td>6.585</td><td>128.857</td><td>1523.98</td></tr><tr><td>w/o LDR Color Supervision</td><td>16.025</td><td>0.6435</td><td>0.3360</td><td>0.2570</td><td>15.844</td><td>35.227</td><td>5.486</td><td>6.276</td><td>116.688</td><td>1523.98</td></tr><tr><td>w/o LDR Gradient Supervision</td><td>16.004</td><td>0.6438</td><td>0.3330</td><td>0.2590</td><td>15.772</td><td>36.519</td><td>5.499</td><td>6.328</td><td>117.445</td><td>1523.98</td></tr><tr><td>EventSeeColor (full)</td><td>16.0100.6443</td><td></td><td>0.3329</td><td>0.2589</td><td>15.767</td><td>36.603</td><td>5.494</td><td></td><td>6.326  117.286</td><td>1523.98</td></tr></table>

![](images/8fea0a1ad7ff7f36ce4537377d4fbee12b20fb2d8137488a444a73b35f87a5a7.jpg)  
Figure 5. EventSeeColor reconstructions of concrete, foliage, sky, and road surfaces. Rows show RGB targets, outputs (Ours), and corresponding close-ups. Fluorescent green boxes mark the enlarged regions. Crops and display scales match.

## 5.3. Ablation Studies

Table 2 examines whether aligned events provide complementary information for colorizing event-reconstructed grayscale images. Among the component and eventguidance configurations, EventSeeColor achieves the high est SSIM and lowest LPIPS, DISTS, and chromatic error. Removing BCR increases chromatic error and desaturation, while removing LDR retains more chroma at the cost of RGB fidelity and perceptual agreement. Appendix D.4 defines these configurations, and Appendix E.3 provides further analysis and the achromatic-input experiment. We also ablate EventSeeColor’s supervision terms. Perceptual and color supervision support perceptual agreement and chromatic accuracy, respectively. Our visual assessment finds that the full configuration provides the best overall visual quality among these supervision configurations, balancing image fidelity and color accuracy (Appendix E.5).

![](images/0e2d50c231c7ff305f4d713e3a422938df63dc9d1c9b615a53ac576d0a16f5c6.jpg)  
Figure 6. Target-color accuracy of EventSeeColor. Bars show crop-averaged regional CIEDE2000. Lower is better. Appendix C.3 defines the groups and aggregation.

## 6. Conclusion

EventSeeColor reconstructs scene color from monochrome events through joint grayscale and event guidance. Regionbalanced chroma supervision and local RGB refinement improve surface colors, brightness, and detail. We curate paired data from PEOD and CoSEC to support supervised color reconstruction. On the combined evaluation, EventSeeColor achieves the best structural similarity, perceptual fidelity, and chromatic-region color accuracy among the external baselines. The ablations provide empirical evidence that temporal event activity contributes to color inference beyond grayscale semantics, expanding the role of monochrome event sensing in recovering scene appearance.

## References

[1] Jason Antic. DeOldify. Software repository, 2018. https://github.com/jantic/DeOldify.

[2] Sami Arja, Nimrod Kruger, Alexander Marcireau, Nicholas Owen Ralph, Saeed Afshar, and Gregory Cohen. Seeing like a cephalopod: Colour vision with a monochrome event camera. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), pages 5023– 5032, 2025.

[3] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-VL technical report. arXiv preprint arXiv:2502.13923, 2025.

[4] Mikołaj Binkowski, Danica J. Sutherland, Michael´ Arbel, and Arthur Gretton. Demystifying MMD GANs. In International Conference on Learning Representations, 2018.

[5] Enrico Calabrese, Gemma Taverni, Christopher Awai Easthope, Sophie Skriabine, Federico Corradi, Luca Longinotti, Kynan Eng, and Tobi Delbruck. DHP19: Dynamic Vision Sensor 3D Human Pose Dataset. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, 2019.

[6] Zheng Chang, Shuchen Weng, Peixuan Zhang, Yu Li, Si Li, and Boxin Shi. L-CAD: Language-based colorization with any-level descriptions using diffusion priors. In Advances in Neural Information Processing Systems, pages 77174–77186, 2023.

[7] Kanghao Chen, Hangyu Li, Jiazhou Zhou, Zeyu Wang, and Lin Wang. LaSe-E2V: Towards languageguided semantic-aware event-to-video reconstruction. In Advances in Neural Information Processing Systems, pages 70406–70430, 2024.

[8] Langyi Chen, Chuanzhi Xu, Haoxian Zhou, Haodong Chen, Qiang Qu, and Weidong Cai. Event-Camera Simulation: A Review. SSRN preprint 6985078, 2026.

[9] Langyi Chen, Chuanzhi Xu, Haoxian Zhou, Pengfei Ye, Ziyu Luo, Haodong Chen, Qiang Qu, Xiaoming Chen, and Weidong Cai. FracEvent: Event-Camera Simulation via Fractional-Relaxation Pixel Dynamics. arXiv preprint arXiv:2606.26636, 2026.

[10] Khen Cohen, Omer Hershko, Homer Levy, David Mendlovic, and Dan Raviv. Illumination-based color reconstruction for the dynamic vision sensor. Sensors, 23(19):8327, 2023.

[11] Luoping Cui, Hanqing Liu, Mingjie Liu, Endian Lin, Donghong Jiang, Yuhao Wang, and Chuang Zhu. PEOD: A pixel-aligned event-RGB benchmark for object detection under challenging conditions. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 18207–18215, 2026.

[12] Keyan Ding, Kede Ma, Shiqi Wang, and Eero P. Simoncelli. Image quality assessment: Unifying structure and texture similarity. IEEE Transactions on Pattern Analysis and Machine Intelligence, 44(5):2567– 2581, 2022.

[13] Xiangcheng Du, Zhao Zhou, Yanlong Wang, Yingbin Zheng, Xingjiao Wu, Peizhu Gong, and Cheng Jin. Unleashing the semantic adaptability of controlled diffusion model for image colorization. In Proceedings of the Thirty-Fourth International Joint Conference on Artificial Intelligence, pages 945–953, 2025.

[14] Cheng-De Fan, Chun-Wei Tuan Mu, Chen-Wei Chang, Chin-Yang Lin, Kun-Ru Wu, Yu-Chee Tseng, and Yu-Lun Liu. LongE2V: Long-horizon eventbased video reconstruction, prediction, and frame interpolation with video diffusion models. In SIG-GRAPH Conference Papers. ACM, 2026.

[15] Guillermo Gallego, Tobi Delbruck, Garrick Orchard, Chiara Bartolozzi, Brian Taba, Andrea Censi, Stefan Leutenegger, Andrew J. Davison, Jorg Conradt, Kostas Daniilidis, and Davide Scaramuzza. Eventbased vision: A survey. IEEE Transactions on Pattern Analysis and Machine Intelligence, 44(1):154– 180, 2022.

[16] Daniel Gehrig, Henri Rebecq, Guillermo Gallego, and Davide Scaramuzza. Asynchronous, Photometric Feature Tracking using Events and Frames. In Proceedings of the European Conference on Computer Vision (ECCV), 2018.

[17] Daniel Gehrig, Antonio Loquercio, Konstantinos G. Derpanis, and Davide Scaramuzza. End-to-End Learning of Representations for Asynchronous Event-Based Data. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision (ICCV), 2019.

[18] Mathias Gehrig, Mario Millhausler, Daniel Gehrig,¨ and Davide Scaramuzza. E-RAFT: Dense Optical Flow from Event Cameras. arXiv preprint arXiv:2108.10552, 2021.

[19] David Hasler and Sabine E. Suesstrunk. Measuring colorfulness in natural images. In Human Vision and Electronic Imaging VIII, pages 87–95, 2003.

[20] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. GANs trained by a two time-scale update rule converge to a local Nash equilibrium. In Advances in Neural Information Processing Systems, 2017.

[21] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations, 2022.

[22] Yuhuang Hu, Shih-Chii Liu, and Tobi Delbruck. v2e: From Video Frames to Realistic DVS Events. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, pages 1312–1321, 2021.

[23] Zhitong Huang, Nanxuan Zhao, and Jing Liao. Uni-Color: A unified framework for multi-modal colorization with transformer. ACM Transactions on Graph ics, 41(6):205:1–205:16, 2022.

[24] Feiyu Ji, Xiang Li, Hao Ma, Tianxiang Huang, Qingxin Lu, Mengqi Ji, Lei Han, Xiaokang Yang, and Xiaoyun Yuan. Engram-E2VID: Referencebased event-to-video reconstruction via generative activation of appearance engrams. arXiv preprint arXiv:2608.05728, 2026.

[25] Xiaozhong Ji, Boyuan Jiang, Donghao Luo, Guangpin Tao, Wenqing Chu, Zhifeng Xie, Chengjie Wang, and Ying Tai. ColorFormer: Image colorization via color memory assisted hybrid-attention transformer. In Computer Vision – ECCV 2022, pages 20–36, 2022.

[26] Xiaoyang Kang, Tao Yang, Wenqi Ouyang, Peiran Ren, Lingzhi Li, and Xuansong Xie. DDColor: Towards photo-realistic image colorization via dual decoders. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision (ICCV), pages 328– 338, 2023.

[27] Geonung Kim, Kyoungkook Kang, Seongtae Kim, Hwayoon Lee, Sehoon Kim, Jonghyun Kim, Seung-Hwan Baek, and Sunghyun Cho. BigColor: Colorization using a generative color prior for natural images. In Computer Vision – ECCV 2022, pages 350–366, 2022.

[28] Bingchen Li, Zhixin Wang, Fan Li, Jiaqi Xu, Jiaming Guo, Renjing Pei, Xin Li, and Zhibo Chen. ColorFLUX: A structure-color decoupling framework for old photo colorization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 15574–15584, 2026.

[29] Weilun Li, Lei Sun, Ruixi Gao, Qi Jiang, Yuqin Ma, Kaiwei Wang, Ming-Hsuan Yang, Luc Van Gool, and Danda Pani Paudel. EvDiff: Event-based video reconstruction using one-step diffusion models. In Computer Vision – ECCV 2026, pages 617–636. Springer Nature Switzerland, 2026.

[30] Yifan Li, Shuai Yang, and Jiaying Liu. Languagebased image colorization: A benchmark and beyond. arXiv preprint arXiv:2503.14974, 2025.

[31] Guoqiang Liang, Kanghao Chen, Hangyu Li, Yunfan Lu, and Lin Wang. Towards Robust Event-guided Low-Light Image Enhancement: A Large-Scale Real-World Event-Image Dataset and Novel Approach. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 23–33, 2024.

[32] Jinxiu Liang, Bohan Yu, Yixin Yang, Yiming Han, and Boxin Shi. E2VIDiff: Perceptual events-to-video reconstruction using diffusion priors. arXiv preprint arXiv:2407.08231, 2024.

[33] Quanmin Liang, Xiawu Zheng, Kai Huang, Yan Zhang, Jie Chen, and Yonghong Tian. Event-Diffusion: Event-based image reconstruction and restoration with diffusion models. In Proceedings of the 31st ACM International Conference on Multimedia, pages 3837–3846, 2023.

[34] Zhexin Liang, Zhaochen Li, Shangchen Zhou, Chongyi Li, and Chen Change Loy. Control Color: Multimodal diffusion-based interactive image colorization. International Journal of Computer Vision, 133:7897–7923, 2025.

[35] Hanyue Lou, Jinxiu Liang, Minggui Teng, Yi Wang, and Boxin Shi. V2V: Scaling event-based vision through efficient video-to-voxel simulation. In Advances in Neural Information Processing Systems, pages 122485–122517, 2025.

[36] Yi Ma, Peiqi Duan, Yuchen Hong, Chu Zhou, Yu Zhang, Jimmy Ren, and Boxin Shi. Color4E: Event demosaicing for full-color event-guided image deblurring. In Proceedings of the 32nd ACM International Conference on Multimedia, pages 661–670, 2024.

[37] Swarnim Maheshwari, Syed Imam Ali, and Vineeth N. Balasubramanian. Beyond fixed luminance: Towards panchromatic and orthochromatic image colorization. In Computer Vision – ECCV 2026, 2026.

[38] Seyed Ehsan Marjani Bajestani and Giovanni Beltrame. E-RGB-D: Real-time event-based perception with structured light. International Journal of Computer Vision, 134:Article 411, 2026.

[39] Anish Mittal, Anush K. Moorthy, and Alan C. Bovik. No-reference image quality assessment in the spatial domain. IEEE Transactions on Image Processing, 21 (12):4695–4708, 2012.

[40] Anish Mittal, Rajiv Soundararajan, and Alan C. Bovik. Making a “completely blind” image quality analyzer. IEEE Signal Processing Letters, 20(3):209– 212, 2013.

[41] Diederik Paul Moeys, Chenghan Li, Julien N. P. Martel, Simeon Bamford, Luca Longinotti, Vasyl Motsnyi, David San Segundo Bello, and Tobi Delbruck. Color temporal contrast sensitivity in dynamic vision

sensors. In 2017 IEEE International Symposium on Circuits and Systems (ISCAS), pages 1–4, 2017.

[42] Shihan Peng, Hanyu Zhou, Hao Dong, Zhiwei Shi, Haoyue Liu, Yuxing Duan, Yi Chang, and Luxin Yan. CoSEC: A coaxial stereo event camera dataset for autonomous driving. arXiv preprint arXiv:2408.08500, 2024.

[43] Stefano Pini, Guido Borghi, and Roberto Vezzani. Learn to see by events: Color frame synthesis from event and RGB cameras. In Proceedings of the 15th International Joint Conference on Computer Vision, Imaging and Computer Graphics Theory and Applications, pages 37–47, 2020.

[44] Henri Rebecq, Guillermo Gallego, Elias Mueggler, and Davide Scaramuzza. EMVS: Event-Based Multi-View Stereo—3D Reconstruction with an Event Camera in Real-Time. International Journal of Computer Vision, 126(12):1394–1414, 2018.

[45] Henri Rebecq, Daniel Gehrig, and Davide Scaramuzza. ESIM: an Open Event Camera Simulator. In Proceedings of The 2nd Conference on Robot Learning, pages 969–982. PMLR, 2018.

[46] Henri Rebecq, Rene Ranftl, Vladlen Koltun, and Davide Scaramuzza. Events-to-video: Bringing modern computer vision to event cameras. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 3857–3866, 2019.

[47] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bjorn Ommer. High-¨ resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 10684–10695, 2022.

[48] Chitwan Saharia, William Chan, Huiwen Chang, Chris Lee, Jonathan Ho, Tim Salimans, David Fleet, and Mohammad Norouzi. Palette: Image-to-image diffusion models. In ACM SIGGRAPH 2022 Confer ence Proceedings, pages 1–10, 2022.

[49] Cedric Scheerlinck, Henri Rebecq, Timo Stoffregen, Nick Barnes, Robert Mahony, and Davide Scaramuzza. CED: Color event camera dataset. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, 2019.

[50] Gaurav Sharma, Wencheng Wu, and Edul N. Dalal. The CIEDE2000 color-difference formula: Implementation notes, supplementary test data, and mathematical observations. Color Research & Application, 30(1):21–30, 2005.

[51] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In International Conference on Learning Representations, 2021.

[52] James W. Tanaka and Lynn M. Presnell. Color di-

agnosticity in object recognition. Perception & Psychophysics, 61(6):1140–1153, 1999.

[53] Dmitrii Torbunov, Onur Okuducu, Yi Huang, Odera Dim, Rebecca Coles, Yonggang Cui, and Yihui Ren. IE2Video: Adapting pretrained diffusion models for event-based video reconstruction. arXiv preprint arXiv:2512.05240, 2025.

[54] Stepan Tulyakov, Daniel Gehrig, Stamatios Georgoulis, Julius Erbach, Mathias Gehrig, Yuanyou Li, and Davide Scaramuzza. Time Lens: Event-Based Video Frame Interpolation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 16155–16164, 2021.

[55] Zhou Wang, Alan C. Bovik, Hamid R. Sheikh, and Eero P. Simoncelli. Image quality assessment: From error visibility to structural similarity. IEEE Transactions on Image Processing, 13(4):600–612, 2004.

[56] Ziyun Wang, Friedhelm Hamann, Kenneth Chaney, Wen Jiang, Guillermo Gallego, and Kostas Daniilidis. Event-based continuous color video decompression from single frames. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, pages 5007–5017, 2025.

[57] Shuchen Weng, Jimeng Sun, Yu Li, Si Li, and Boxin Shi. CT<sup>2</sup>: Colorization transformer via color tokens. In Computer Vision – ECCV 2022, pages 1–16, 2022.

[58] Menghan Xia, Wenbo Hu, Tien-Tsin Wong, and Jue Wang. Disentangled image colorization via global anchors. ACM Transactions on Graphics, 41(6):204:1– 204:13, 2022.

[59] Chuanzhi Xu, Langyi Chen, Haodong Chen, Vera Chung, and Qiang Qu. Towards End-to-End Neuromorphic Event-based 3D Object Reconstruction Without Physical Priors. arXiv preprint arXiv:2501.00741, 2025.

[60] Chuanzhi Xu, Haoxian Zhou, Langyi Chen, Haodong Chen, Zeke Zexi Hu, Zhicheng Lu, Ying Zhou, Vera Chung, Qiang Qu, and Weidong Cai. A survey of 3D reconstruction with event cameras. Computational Visual Media, 12(5):1263–1299, 2026.

[61] Chuanzhi Xu, Haoxian Zhou, Langyi Chen, Yuk Ying Chung, and Qiang Qu. Ultralight Polarity-Split Neuromorphic SNN for Event-Stream Super-Resolution. Proceedings of the AAAI Conference on Artificial Intelligence, 40(13):11196–11204, 2026.

[62] Gang Xu, Zhiyu Zhu, and Junhui Hou. UniE2F: A unified diffusion framework for event-to-frame reconstruction with video foundation models. arXiv preprint arXiv:2602.19202, 2026.

[63] Bohan Yu, Jinxiu Liang, Zhuofeng Wang, Bin Fan, Art Subpa-asa, Boxin Shi, and Imari Sato. Active hyperspectral imaging using an event camera. In Proceedings of the IEEE/CVF Conference on Computer

Vision and Pattern Recognition (CVPR), pages 929– 939, 2025.

[64] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 3836–3847, 2023.

[65] Richard Zhang, Phillip Isola, and Alexei A Efros. Colorful image colorization. In Computer Vision – ECCV 2016, pages 649–666. Springer, 2016.

[66] Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings ofthe IEEE Conference on Computer Vision and Pattern Recognition, pages 586–595, 2018.

[67] Haoxian Zhou, Chuanzhi Xu, Langyi Chen, Pengfei Ye, Haodong Chen, Yuk Ying Chung, and Qiang Qu. Exploiting Spatiotemporal Properties for Efficient Event-Driven Human Pose Estimation. arXiv preprint arXiv:2512.06306, 2026.

[68] Haoxian Zhou, Chuanzhi Xu, Langyi Chen, Pengfei Ye, Haodong Chen, Qiang Qu, Ali Anaissi, and Weidong Cai. EvLIR: Learning Illumination Residuals from Ordered Events for Low-Light Image Enhancement. arXiv preprint arXiv:2606.29430, 2026.

[69] Lin Zhu, Yunlong Zheng, Yijun Zhang, Xiao Wang, Lizhi Wang, and Hua Huang. Temporal residual guided diffusion framework for event-driven video reconstruction. In Computer Vision – ECCV 2024, pages 411–427. Springer, 2024.

## EventSeeColor Supplementary Material

## A. Research Motivation

Event cameras asynchronously record changes in light intensity, producing measurements of pixel location, timestamp, and polarity with high temporal resolution and a wide dynamic range [15]. Sensor models and event simulators describe how intensity trajectories, response dynamics, and threshold crossings produce event timing and polarity [8, 9, 22, 45]. The resulting event stream encodes spatial structure and temporal variation for learning scene properties [15, 17].

These measurements support 3D reconstruction [44, 59, 60], human pose estimation [5, 67], and dense optical flow estimation [18]. Event-stream super-resolution improves spatial detail while preserving temporal and polarity information [61]. Events also complement image measurements. Frame-event feature tracking combines imagebased spatial references with asynchronous temporal updates [16], and event-guided frame interpolation uses motion information between captured images to reconstruct intermediate frames [54]. Under low illumination, eventguided enhancement uses structural and illumination cues from events to improve image visibility and detail [31, 68]. These applications combine image appearance and temporal events for perception and image recovery [16, 31, 54].

Traditional event-based image reconstruction makes event recordings accessible as grayscale intensity images [46]. However, grayscale reconstruction does not express color differences between surfaces with similar intensity, motivating the inference of missing chromatic information [65]. Color can distinguish the appearance of road markings, vegetation, and vehicle surfaces, and can aid recognition when objects have characteristic colors [52]. Therefore, recovering color can enrich the scene information conveyed by monochrome event recordings. Grayscale images provide spatial and semantic cues for learned colorization [26, 65], while aligned events retain local timing, polarity, and activity [15, 17]. This complementary information motivates EventSeeColor to investigate whether temporal events improve color reconstruction beyond the spatial and semantic guidance of eventreconstructed grayscale images.

The significance of color reconstruction also extends to downstream perception. Pretrained RGB vision models can extract appearance features from reconstructed color images. Aligning and fusing these appearance features with event-derived structural and motion features could combine semantic context, surface-color distinctions, and precise temporal information within a shared representation. Such integration could improve object recognition, human pose estimation, and 3D reconstruction using monochrome event recordings as the sole sensor input. This perspective positions color reconstruction as a bridge between event sensing and appearance-aware visual understanding.

## B. Method Details

This appendix details Event and Grayscale Fusion (EGF), Balanced Chroma Refinement (BCR), Local Detail Refinement (LDR), and their inputs and training objectives. Symbols follow Section 3: $G _ { t }$ is the normalized eventreconstructed grayscale image, $V _ { t }$ the aligned event tensor, $A _ { t }$ the event activity map, $m _ { t }$ the validity mask, and $L _ { t }$ the calibrated event lightness. The fused hint $H _ { t }$ and fixed photographic prompt π guide the diffusion colorizer. The Lab chroma vector $\mathbf { c } = ( a , b )$ has magnitude $C = \| \mathbf { c } \| _ { 2 }$ . BCR refines initial chroma $\mathbf { c } _ { t } ^ { 0 }$ into $\mathbf { c } _ { t } ^ { r }$ and renders the intermediate RGB image $I _ { t } ^ { r }$ . LDR produces the unclipped RGB output $\widetilde { I } _ { t }$ and final RGB image $\widehat { I } _ { t }$ . The time subscript t is omitted where a single target time is understood. A superscript ∗ denotes a training or evaluation target.

## B.1. Event Representation and Lightness Calibration

Reconstruction and Event Encoding. The official E2VID lightweight recurrent checkpoint reconstructs a grayscale image at each target time from the preceding one second of events through twenty consecutive 50 ms updates, each encoded with five signed temporal bins. Recurrent state is reset before each target history and retained across its twenty updates. The target timestamp is expressed in the aligned event clock. The event-reconstructed grayscale image $G _ { t }$ is normalized using its first and ninety-ninth percentiles. The aligned event tensor $V _ { t }$ covers the final 50 ms ending at that target timestamp and uses ten temporal bins for each polarity. Each event contributes to its adjacent bins through linear interpolation. Two polarity count maps and two lasttimestamp surfaces are appended, producing a 24-channel tensor. For a nonnegative voxel or count value x, normalization is:

$$
\mathcal { N } _ { \tau } ( x ) = \operatorname* { m i n } \left( \frac { \log ( 1 + x ) } { \log ( 1 + \tau ) } , 1 \right) .\tag{A.1}
$$

The voxel and count scales are given in Table $_ { \textrm { A 2 } }$ . Each timestamp surface records the most recent event of its polarity within the interval, normalized to [0, 1]. The activity map $A _ { t }$ is the mean of the two normalized count channels. Adaptive max pooling of the nonzero support $A _ { t } > 0$ gives the binary EGF support map $S _ { \ell }$ at each feature scale. LDR receives the continuous activity map $A _ { t }$

Lightness Calibration. The calibrated event lightness $L _ { t }$ is derived from the E2VID grayscale reconstruction before percentile normalization. Replicating the raw grayscale into RGB and converting it to Lab gives $L _ { t } ^ { \mathrm { r a w } }$ . A dataset-specific affine map is fitted by least squares with a nonnegative slope over valid training pixels:

$$
L _ { t } = \mathrm { c l i p } ( \mu _ { d } L _ { t } ^ { \mathrm { r a w } } + \nu _ { d } , 0 , 1 0 0 ) .\tag{A.2}
$$

Here, d indexes the dataset. The fitted coefficients are $( \mu _ { d } , \nu _ { d } ) ~ = ~ ( 0 . 9 9 1 8 6 4 , - 0 . 4 6 7 3 5 6 )$ for PEOD and (0.546846, 12.793549) for CoSEC. These fitted coefficients remain fixed during training and inference. Lightness calibration determines the RGB targets used to train the diffusion colorizer and the lightness used to render BCR outputs. LDR subsequently adjusts the rendered RGB image, including its lightness.

Gamut Mapping. For lightness L and predicted chroma $\mathbf { c } , \Pi _ { L }$ finds the largest feasible scale $\xi \in [ 0 , 1 ]$ such that ${ \mathcal { T } } ( L , \xi \mathbf { c } )$ lies in the RGB gamut. The implementation uses bisection along the chroma direction. This operation preserves Lab lightness and hue while reducing only out-ofgamut chroma. Gamut mapping produces the initial chroma $\mathbf { c } ^ { 0 }$ supplied to BCR and maps the refined chroma $\mathbf { c } ^ { r }$ before rendering the intermediate RGB image $I ^ { r }$ . The final RGB correction follows Eq. 4.

## B.2. Fusion and Fixed Prompt Settings

Event and Grayscale Fusion. EGF uses three encoder stages with separate grayscale and event branches. For 256×256 inputs, the stages operate at 256×256, 128×128, and $6 4 \times 6 4$ , and the fused hint has 320 channels at $6 4 \times 6 4$ resolution. Grayscale is resized to the event resolution by area interpolation, and both inputs are multiplied by the validity mask before encoding. Each stage consists of a convolution, GroupNorm, and SiLU. The gate in Eq. 1 uses a pointwise convolution on concatenated features. Features from all scales are pooled to the resolution of the pretrained grayscale hint $H _ { t } ^ { 0 }$ before projection. The grayscale and event projections are separate pointwise convolutions, and the event projection omits its bias. Zero initialization of both projections makes the fused hint $H _ { t }$ equal to $H _ { t } ^ { 0 }$ at the start of adaptation.

Fixed Photographic Prompt. Training and inference use the same photographic prompt “natural color photograph, realistic materials, balanced saturation, neutral lighting,” followed by the quality suffix “best quality, detailed, real.” The resulting text condition π is shared across all images. Sampling uses the fixed negative prompt “a black and white photo, longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality.”

## B.3. Balanced Chroma Refinement

BCR receives the five-channel image input, with s the chroma normalization scale:

$$
X = [ G , \mathbf { c } ^ { 0 } / s , \| \mathbf { c } ^ { 0 } \| _ { 2 } / s , m ] .\tag{A.3}
$$

The image encoder processes $X { \odot } m$ through two stride-two stages, producing $6 4 \times 6 4$ features for a $2 5 6 \times 2 5 6$ crop. The event tensor $V \odot m$ is resized by area interpolation to the image feature resolution and processed by its own encoder. The image and event features are concatenated and processed by a joint context network to produce features $F .$ . Two heads produce a two-channel gain precursor q and residual r:

$$
\begin{array} { r l } & { q = \operatorname { t a n h } ( \mathcal { U } ( U _ { \mathrm { g a i n } } ( F ) ) ) , } \\ & { \alpha = \mathbf { 1 } + \gamma _ { + } \operatorname* { m a x } ( q , 0 ) + \gamma _ { - } \operatorname* { m i n } ( q , 0 ) , } \\ & { \mathbf { r } = \rho \operatorname { t a n h } ( \mathcal { U } ( U _ { \mathrm { r e s } } ( F ) ) ) , } \end{array}\tag{A.4}
$$

where $U _ { \mathrm { g a i n } }$ and $U _ { \mathrm { r e s } }$ are the two prediction heads and U bilinearly upsamples their logits to the output resolution before the nonlinearities. The gain α and chroma residual r are used in Eq. 3. The scalars $\gamma _ { + }$ and $\gamma _ { - }$ set the amplification and attenuation ranges, and $\rho$ sets the residual bound. Independent gain channels permit different multiplicative adjustments of a and b. Both heads have zero weights and biases at initialization, giving $\alpha = 1$ and $\mathbf { r } = \mathbf { 0 }$

## B.4. Local Detail Refinement

LDR receives the six-channel tensor $[ I ^ { r } , G , A , m ] \odot m$ . Two full-resolution convolutions produce shallow features. The context branch takes these shallow features as input and applies a stride-two convolution followed by a dilated convolution. Bilinear upsampling returns the context features to the shallow resolution, and their concatenation is decoded into three RGB residual logits. The final convolution has zero weights and biases at initialization. Applying the bounded residual in Eq. 4 preserves the intermediate RGB image $I ^ { r }$ outside the valid region. The correction acts in encoded sRGB coordinates, so all three output channels can respond to local appearance errors. Table $_ \mathrm { A } 2$ gives the widths and residual bound.

## B.5. Training Objectives

Throughout the objectives, j indexes images, i indexes pixels, and $k \in \{ a , b \}$ indexes chroma channels.

Diffusion Denoising. The training target is RGB rendered from paired target chroma at calibrated event lightness after gamut mapping. At diffusion step $n ,$ the target latent is perturbed by Gaussian noise ϵ to give $z _ { n }$ . Let $\widehat { \epsilon } = \epsilon _ { \theta } ( z _ { n } , n , H _ { t } , \pi )$ be the noise predicted by the denoiser with parameters $\theta , \ m ^ { z }$ the validity mask resized by area interpolation to latent resolution, and $d _ { z }$ the latent channel count. The denoising loss is:

$$
\mathcal { L } _ { \mathrm { d i f f } } = \mathbb { E } \left[ \frac { \sum _ { i } m _ { i } ^ { z } \lVert \epsilon _ { i } - \widehat { \epsilon } _ { i } \rVert _ { 2 } ^ { 2 } } { d _ { z } \sum _ { i } m _ { i } ^ { z } } \right] .\tag{A.5}
$$

Auxiliary Chroma Supervision. The training-only auxiliary chroma head predicts normalized coarse target chroma

$\mathbf { u } = h _ { \operatorname* { a u x } } ( H _ { t } )$ . With adaptive average pooling ${ \mathcal { P } } _ { \mathrm { { : } } }$ , the valid area and target chroma at hint resolution are:

$$
\bar { m } = \mathcal { P } ( m ) , \qquad \bar { \mathbf { c } } ^ { * } = \frac { \mathcal { P } ( m \odot \Pi _ { L } ( \mathbf { c } ^ { * } ) ) } { \operatorname* { m a x } ( \bar { m } , \varepsilon ) } ,\tag{A.6}
$$

where $\varepsilon = 1 0 ^ { - 8 }$ stabilizes division at empty locations. Define $\omega _ { i } = \bar { m } _ { i } [ 1 + \eta \operatorname* { m i n } ( \| \bar { \mathbf { c } } _ { i } ^ { * } \| _ { 2 } / \kappa , 1 ) ]$ ]. The auxiliary objective is:

$$
\mathcal { L } _ { \mathrm { a u x } } = \frac { \sum _ { i } \omega _ { i } \sum _ { k } \mathrm { S m o o t h L 1 } ( u _ { i k } - \bar { c } _ { i k } ^ { * } / s ) } { \operatorname* { m a x } ( 2 \sum _ { i } \omega _ { i } , \varepsilon ) } .\tag{A.7}
$$

Thus, valid area determines target pooling and loss weighting.

Region-Balanced Chroma Objective. BCR is trained against the original paired target Lab chroma. The diffusion stage uses gamut-mapped RGB targets for latent encoding, while BCR directly supervises chroma and applies gamut projection when rendering RGB. The per-pixel regression loss and chroma weight are:

$$
\begin{array} { l l } { \ell _ { j i } = \displaystyle \frac { 1 } { 2 } \sum _ { k } \mathrm { S m o o t h L 1 } \left( \frac { c _ { j i k } ^ { r } - c _ { j i k } ^ { * } } { s } \right) , } \\ { w _ { j i } = 1 + \displaystyle \eta \operatorname* { m i n } ( C _ { j i } ^ { * } / \kappa , 1 ) . } \end{array}\tag{A.8}
$$

SmoothL1 uses unit transition width. The scale s normalizes Lab chroma, while η and κ control additional weight for chromatic targets.

Global and Region Terms. Target chroma thresholds $\tau _ { n }$ and $\tau _ { c }$ define the neutral, transitional, and chromatic subsets by $C ^ { * } < \tau _ { n } , \tau _ { n } \leq C ^ { * } \leq \tau _ { c }$ , and $C ^ { * } > \tau _ { c }$ . The reductions follow Eq. 6. The global term divides weighted error by the valid pixel count. The region term first averages within each target subset and then gives equal weight to nonempty subsets and to images.

Chroma Shortfall and Neutral-Chroma Supervision. Writing $\Omega _ { C }$ and $\Omega _ { N }$ for all chromatic and neutral pixels in the batch, respectively, the shortfall and neutral terms are:

$$
\mathcal { L } _ { \mathrm { s h o r t } } = \frac { 1 } { \left| \Omega _ { C } \right| } \sum _ { ( j , i ) \in \Omega _ { C } } \frac { ( C _ { j i } ^ { * } - C _ { j i } ^ { r } ) _ { + } } { s } ,\tag{A.9}
$$

$$
\mathcal { L } _ { \mathrm { n e u t r a l } } = \frac { 1 } { | \Omega _ { N } | } \sum _ { ( j , i ) \in \Omega _ { N } } { \frac { \| \mathbf { c } _ { j i } ^ { r } - \mathbf { c } _ { j i } ^ { * } \| _ { 2 } ^ { 2 } } { 2 s ^ { 2 } } } ,\tag{A.10}
$$

where $( x ) _ { + } = \operatorname* { m a x } ( x , 0 )$ and $C ^ { r } = \| \mathbf { c } ^ { r } \| _ { 2 }$ . The neutral term matches the target’s small chroma, retaining weak surface coloration.

Warm-Region Supervision. Target hue in degrees is $h ^ { * } =$ $( 1 8 0 / \pi ) \mathrm { a t a n 2 } ( b ^ { \ast } , a ^ { \ast } )$ . The warm subset $\Omega _ { j W }$ satisfies $C ^ { * } > \tau _ { c }$ and $h ^ { \ast } \in [ h _ { - } , h _ { + } ]$ . For the set $\mathcal { I } _ { W }$ of images containing this subset, warm-region supervision is:

$$
\mathcal { L } _ { \mathrm { w a r m } } = \frac { 1 } { \left| \mathcal { I } _ { W } \right| } \sum _ { j \in \mathcal { I } _ { W } } \frac { 1 } { \left| \Omega _ { j W } \right| } \sum _ { i \in \Omega _ { j W } } \ell _ { j i } .\tag{A.11}
$$

This hue-specific mean gives warm surface colors direct supervision within the chromatic subset, complementing the magnitude-based region balancing. Warm-region supervision uses unweighted regression errors. Empty region terms contribute zero. The complete BCR objective follows Eq. 7, with values given in Table A3.

## B.6. Local Detail Refinement Objectives

The LDR objective supervises the clipped RGB output Ib and penalizes clipping of the unclipped RGB output ${ \widetilde { I } } .$ Let $I ^ { * }$ denote the paired RGB target and define $\phi ( x ) = $ $\sqrt { x ^ { 2 } + \varepsilon _ { \mathrm { r g b } } ^ { 2 } }$ . For any multichannel image $Z , \langle Z \rangle _ { m }$ denotes the average over its valid spatial entries and all channels, aggregated across the batch. The reconstruction and gradient terms are:

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { r g b } } = \langle \phi ( \widehat { I } - I ^ { * } ) \rangle _ { m } , } \\ & { \mathcal { L } _ { \mathrm { g r a d } } = \displaystyle \frac { 1 } { 2 } \sum _ { d \in \{ x , y \} } \langle \phi ( D _ { d } \widehat { I } - D _ { d } I ^ { * } ) \rangle _ { m ^ { d } } , } \end{array}\tag{A.12}
$$

where $D _ { d }$ is a forward finite difference and $m ^ { d }$ is the product of the validity masks at the two adjacent pixels. The gradient term supervises changes across neighboring pixels, complementing pointwise RGB fidelity.

LDR color supervision $\scriptstyle { \mathcal { L } } _ { \mathrm { c o l o r } }$ uses the Lab chroma of the final RGB prediction ${ \widehat { I } } .$ For image j and pixel $i ,$ define $\begin{array} { r } { \ell _ { j i } ^ { \mathrm { R G B } } = \frac { 1 } { 2 } \sum _ { k } } \end{array}$ SmoothL $. \big ( ( \widehat { c } _ { j i k } - c _ { j i k } ^ { * } ) / s \big )$ , where $\widehat { \mathbf { c } }$ is obtained from Ib. The target chroma defines the same neutral, transitional, and chromatic subsets $\Omega _ { j r }$ used by BCR. Each nonempty subset contributes equally:

$$
\mathcal { L } _ { \mathrm { c o l o r } } = \frac { 1 } { B } \sum _ { j } \frac { 1 } { \vert \mathcal { R } _ { j } \vert } \sum _ { r \in \mathcal { R } _ { j } } \frac { 1 } { \vert \Omega _ { j r } \vert } \sum _ { i \in \Omega _ { j r } } \ell _ { j i } ^ { \mathrm { R G B } } .\tag{A.13}
$$

This objective uses the unweighted chroma regression error within each region. Before differentiable Lab conversion, RGB values are lower-bounded by $1 0 ^ { - 6 }$ to keep the gradient finite at black pixels.

LDR perceptual supervision $\mathcal { L } _ { \mathrm { p e r c } }$ uses LPIPS with a frozen AlexNet network [66]. For each image, the final RGB prediction Ib and paired RGB target $I ^ { * }$ are cropped to the valid bounding rectangle and rescaled from [0, 1] $\mathrm { ~ t o ~ } [ - 1 , 1 ]$ . The perceptual term is the mean LPIPS distance over images. The clipping penalty $\mathcal { L } _ { \mathrm { c l i p } }$ measures the squared RGB difference introduced by clipping output values to [0, 1]:

$$
{ \mathcal { L } } _ { \mathrm { c l i p } } = \langle ( { \widetilde { I } } - { \widehat { I } } ) ^ { 2 } \rangle _ { m } .\tag{A.14}
$$

These terms form Eq. 8. Their weights and the Charbonnier constant are reported in Table A3.

## B.7. Optional Grayscale Scene Prompt

EventSeeColor uses the fixed photographic prompt π in Appendix B.2 by default. An optional Grayscale Scene Prompt (GSP) augments this prompt with image-specific semantic context. A frozen Qwen2.5-VL-3B-Instruct model [3] describes visible objects, spatial arrangement, and recognizable materials from the event-reconstructed grayscale image. Explicit color words are removed before the description is concatenated with the fixed photographic prompt. GSP uses no RGB reference. EGF, BCR, LDR, and the sampling settings are unchanged.

Color Retention and Reconstruction Accuracy. The configurations with and without GSP use the same training seed and budgets, with BCR trained on the corresponding diffusion colorizer outputs and LDR trained on the corresponding BCR outputs. On the 2,220 evaluation samples, adding GSP reduces desaturation from 36.603% to 34.531% and changes PSNR from 16.010 to 16.023 dB. However, LPIPS increases from 0.3329 to 0.3375, and chromatic-region chroma RMSE increases from 8.894 to 9.031. Therefore, EventSeeColor uses the fixed photographic prompt.

Additional Computation. GSP adds 3,754.62 million frozen parameters and no additional trainable parameters. The diffusion sampler retains 20 DDIM steps. On an NVIDIA GeForce RTX 5090 GPU, generating descriptions for 2,220 crops took 537.36 seconds with batch size four, with an amortized captioning-stage cost of 0.242 seconds per crop, including model loading, preprocessing, generation, and cache writing. Cached descriptions can be reused for repeated colorization of the same inputs.

## C. Datasets and Metrics

## C.1. Paired Data and Sequence Splits

PEOD [11] and CoSEC [42] supply real event recordings with corresponding RGB images. PEOD provides the main collection of aligned scene observations, while CoSEC adds a distinct acquisition setting with traffic scenes. Table A1 gives the retained sequence, frame, and crop counts. The split is disjoint by recording sequence within each dataset. PEOD evaluation uses sequences 036, 063, 070, 078, and 096. CoSEC evaluation uses City Day 000.

The retained evaluation set consists of crops from the same fixed target frames for every method. Each method’s output is mapped to those crop coordinates, and scores are measured on the original rectangular valid region. These rectangles cover between 96.51% and 100% of a crop.

## C.2. Native RGB Evaluation

Evaluation measures framewise reconstruction accuracy from each method’s native RGB output, including its predicted lightness. Image colorizers receive the same normalized event-reconstructed grayscale crop, except for the V2V-E2VID and DDColor pipeline. Direct event reconstruction methods use their released procedures to process events. Their event histories and temporal context are specified in Appendix D.2. All methods are compared against the same aligned RGB targets, with no output fitting to evaluation images. Method categories in the tables distinguish their input and inference routes.

Table A1. Data used for training and evaluation. Sequences are disjoint between the two partitions within each dataset.
<table><tr><td>Dataset |</td><td>Partition</td><td>Sequences</td><td>Frames</td><td>Crops</td></tr><tr><td>PEOD</td><td>Training</td><td>18</td><td>1,685</td><td>7,927</td></tr><tr><td rowspan="3">CoSEC</td><td>Evaluation</td><td>5</td><td>479</td><td>2,198</td></tr><tr><td>Training</td><td>3</td><td>70</td><td>141</td></tr><tr><td>Evaluation</td><td>1</td><td>12</td><td>22</td></tr><tr><td rowspan="2">Total</td><td>Training</td><td>21</td><td>1,755</td><td>8,068</td></tr><tr><td>Evaluation</td><td>6</td><td>491</td><td>2,220</td></tr></table>

Each crop receives equal weight in the overall mean. Therefore, combined scores reflect the dataset composition in Table A1. A regional score contributes only when the corresponding target region is present. There are 2,076 crops with chromatic support, 1,905 with warm support, and 2,220 with neutral support.

## C.3. Reference Image and Regional Color Metrics

Image Fidelity. PSNR is reported in dB and uses mean squared error in sRGB with a unit signal range. SSIM [55] uses an 11 × 11 Gaussian window with standard deviation 1.5, population covariance, and exclusion of the window border. LPIPS uses the calibrated AlexNet model with RGB scaled to [−1, 1] [66]. DISTS measures learned structural and textural agreement [12]. Both perceptual metrics operate on the valid rectangle without metric-side resizing. LPIPS is included in Local Detail Refinement training, while DISTS supplies an additional perceptual assessment. Chroma and Perceptual Color Difference. Evaluation converts predicted and target sRGB to CIELAB under D65. For a target-defined region Ω, the chroma root mean squared error is:

$$
\mathrm { R M S E } _ { a b } ( \Omega ) = \sqrt { \frac { 1 } { 2 | \Omega | } \sum _ { i \in \Omega } \| \mathbf { c } _ { i } ^ { \mathrm { p r e d } } - \mathbf { c } _ { i } ^ { * } \| _ { 2 } ^ { 2 } } .\tag{A.15}
$$

This metric isolates the two chroma channels. CIEDE2000 additionally accounts for lightness and perceptual nonuniformity [50]. Its score is the mean pixelwise $\Delta E _ { 0 0 }$ with $k _ { L } = k _ { C } = k _ { H } = 1$ , computed over all valid pixels or the specified target region. Chromatic pixels satisfy $C ^ { * } > 1 0$ and neutral pixels satisfy $C ^ { * } < 5 ,$ , where $C ^ { * } = \| \mathbf { c } ^ { * } \| _ { 2 }$ Warm pixels are chromatic pixels with target hue between $- 3 0 ^ { \circ }$ and 75<sup>◦</sup>.

Target Color Groups. Figure 6 in the main paper uses eight chromatic groups defined by the RGB target’s HSV hue. The intervals are red $[ 3 4 5 ^ { \circ } , 3 6 0 ^ { \circ } ) \cup [ 0 ^ { \circ } , 1 5 ^ { \circ } )$ , orange [15<sup>◦</sup>, 45<sup>◦</sup>), yellow [45<sup>◦</sup>, 75<sup>◦</sup>), green [75<sup>◦</sup>, 165<sup>◦</sup>), cyan $[ 1 6 5 ^ { \circ } , 1 9 5 ^ { \circ } )$ , blue $[ 1 9 5 ^ { \circ } , 2 5 5 ^ { \circ } )$ , purple $[ 2 5 5 ^ { \circ } , 2 8 5 ^ { \circ } )$ , and magenta $[ 2 8 5 ^ { \circ } , 3 4 5 ^ { \circ } )$ Each interval includes its lower boundary and excludes its upper boundary. Chromatic groups require Lab $C ^ { * } > 1 0$ , and a separate neutral group uses $C ^ { * } < 5$ . The transition range $5 \leq C ^ { * } \leq 1 0$ is excluded from this grouping. A crop supports a group when that region occupies at least 1% of its valid pixels. Pixelwise CIEDE2000 is first averaged within the region, then averaged equally over supporting crops.

Desaturation and False Color. Desaturation (Desat.) measures missed color as the fraction of chromatic target pixels whose predicted chroma magnitude is below 5. Neutral false color measures the fraction of neutral target pixels with predicted chroma magnitude above 20. Warm leakage further restricts those predictions to the warm hue interval. Each fraction is computed within a crop and then averaged over supported crops. Together, the measurements distinguish suppressed target color from color introduced on neutral surfaces.

## C.4. Distribution and Naturalness Metrics

FID and KID. Frechet Inception Distance (FID) [´ 20] and Kernel Inception Distance (KID) [4] compare the reconstructed-image distribution with the paired RGB target distribution. Predictions and targets are cropped to identical valid rectangles and quantized to 8-bit RGB. Features are extracted with the TensorFlow-compatible Inception-V3 weights and internal 299 × 299 preprocessing in torch-fidelity. FID uses all 2,220 outputs and targets. KID uses 100 independent subset pairs of size 1,000 and a cubic polynomial kernel, with $\gamma = 1 / 2 0 4 8$ and unit offset. Tables report the KID subset mean and standard deviation multiplied by $1 0 ^ { 3 }$ . This standard deviation describes variation across the sampled subsets.

Colorfulness. Colorfulness summarizes opponent-channel variation [19]. Following the implementation used in DD-Color, the evaluation computes absolute opponent channels $r _ { g } = | R - G |$ and $y _ { b } = | ( R + G ) / 2 - B |$ from 8-bit RGB. The score is:

$$
\mathrm { C F } = \sqrt { \sigma _ { r _ { g } } ^ { 2 } + \sigma _ { y _ { b } } ^ { 2 } } + 0 . 3 \sqrt { \mu _ { r _ { g } } ^ { 2 } + \mu _ { y _ { b } } ^ { 2 } } ,\tag{A.16}
$$

where $\mu$ and $\sigma$ are the pixelwise mean and standard deviation. Colorfulness mean absolute error (CF-MAE) averages the absolute predicted-to-target CF difference per crop.

Natural Image Statistics. The Natural Image Quality Evaluator (NIQE) [40] and Blind/Referenceless Image Spatial Quality Evaluator (BRISQUE) [39] use the pretrained defaults in $\tt p y i q a$ . NIQE operates on YIQ luminance and uses complete $9 6 \times 9 6$ blocks, discarding incomplete blocks along the bottom and right boundaries. BRISQUE uses its standard luminance input. NIQE and BRISQUE measure agreement with natural-image statistics independently of the paired RGB target. Paired perceptual and regional color metrics measure reconstruction accuracy against that target. Reference RGB images are scored with the same settings to characterize natural-image statistics in the evaluated data.

## C.5. Data Construction and Curation Details

Candidate Sampling. PEOD combines fixed sample locations and additional spatial grid locations within the same sequence partition. The fixed-location pool uses four deterministically selected 256 × 256 crops per sampled frame. Additional grid crops have top-left coordinates $x \ \in \ \{ 0 , 2 5 6 , 5 1 2 , 7 6 8 , 1 0 2 4 \}$ and $y ~ \in ~ \{ 0 , 2 3 2 , 4 6 4 \}$ . A grid crop is omitted if its intersection-over-union with an exposure-qualified fixed-location crop exceeds 0.5. PEOD uses a fixed 20 ms RGB-to-event timestamp offset and the sequence-level spatial correction. The associated validity mask excludes warped borders with a two-pixel erosion.

CoSEC samples every fifth RGB frame, starting from timestamps with at least one second of available event history. Eight nonoverlapping $2 5 6 \times 2 5 6$ crops use top-left coordinates $x ~ \in ~ \{ 8 8 , 3 4 4 , 6 0 0 , 8 5 6 \}$ and $y ~ \in ~ \{ 5 6 , 3 1 2 \}$ RGB and events use the supplied coaxial coordinate system. Sequence assignments are fixed before crop selection. Table A1 summarizes the retained sequence, frame, and crop counts.

RGB Exposure Screening. Statistics are computed only on valid pixels of the original RGB target. Let $r , g , b \in [ 0 , 2 5 5 ]$ denote its encoded sRGB channel values, with encoded luma $Y _ { \mathrm { s R G B } } = 0 . 2 1 2 6 r + 0 . 7 1 5 2 g + 0 . 0 7 2 2 b$ . A crop is rejected if any of the following conditions holds:

• At least 10% of valid pixels are near-white, with min $\left( r , g , b \right) \ge 2 5 0$

• At least 10% are near-black, with max $( r , g , b ) \leq 1 0 .$

• At least 20% have low encoded luma, with $Y _ { \mathrm { s R G B } } \leq 3 2 .$

• At least 20% are bright across all channels, with min $( r , g , b ) \ge 2 3 5$

The same rules apply to training and evaluation. Rejection removes the entire crop from the paired set. Retained targets keep their recorded RGB values, and selection does not use model predictions or reconstruction scores.

Crop-Level Correspondence. Grid candidates also undergo local alignment screening. The event edge map marks pixels with at least one event in the final 5 ms before the target. For RGB scaled to [0, 1], grayscale conversion and a $3 \times 3$ Sobel operator provide gradient magnitudes. The RGB edge threshold is the 85th percentile of magnitudes greater than 0.02. Each edge map must contain at least 64 supporting pixels. A search within ±6 pixels identifies the best translation. The crop is retained only if its residual displacement is at most 1.5 pixels, its zero-shift edge cosine similarity is at least 0.15, and that similarity exceeds the strongest 16-pixel horizontal or vertical shift control by at least 0.02. The search tests correspondence after the fixed calibration. It does not introduce a per-crop warp into the training targets.

Screening Counts. Exposure screening retains 2,260 of the 9,976 fixed-location PEOD crops. Among 78,072 additional PEOD grid candidates, 20,406 pass the exposure rules and 7,865 also pass the local correspondence checks. Among 1,488 CoSEC grid candidates, 746 pass exposure screening and 163 pass both stages. Across these candidate pools, exposure screening excludes 66,124 crops, and alignment screening excludes a further 13,124 exposurequalified crops. The remaining 10,288 crops comprise the 8,068 training and 2,220 evaluation samples in Table A1.

## D. Experimental Settings

## D.1. EventSeeColor

Diffusion colorizer adaptation updates EGF, the auxiliary chroma head, and LoRA parameters in ControlNet text cross-attention [21]. The pretrained diffusion colorizer weights remain fixed. BCR is then trained with the adapted diffusion colorizer frozen. Finally, LDR is trained from the frozen BCR RGB outputs, paired grayscale, event activity, and validity masks. Only the loss functions access the paired color targets during LDR training. All three stages use the training partition in Table A1. Training and inference use an NVIDIA RTX 5090 GPU. The exact network and optimization choices are given in Tables A2 and A3.

At inference, the auxiliary chroma head is omitted and sampling uses DDIM [51]. Normalized grayscale is enlarged to 512×512 using Lanczos interpolation. The frozen content-guided deformable VAE from Control Color [34] decodes the sampled latent using grayscale content features. The decoded RGB image is resized to the original crop resolution using Lanczos interpolation before chroma extraction and gamut mapping for BCR. Grayscale provides the initial intensity structure, and event features guide both color generation and refinement. The final output is the native RGB prediction from LDR. Appendix D.5 reports parameter counts and LDR latency.

Self-attention guidance follows Control Color [34] and operates during early reverse diffusion down to timestep 600 on the original 1,000-step timeline. The 20 DDIM steps count sampling updates along this timeline.

## D.2. Comparison Baselines

All external models use their released pretrained weights. Each model produces one prediction per target. The temporal context and RGB frame extraction for the video reconstruction baselines are specified below. The image colorizers use the same E2VID reconstruction, normalized between its first and ninety-ninth percentiles, and follow their released input conversion and RGB postprocessing. We restore their predictions to the common crop coordinates and score native RGB within the valid region. RGB targets are used only for evaluation. Stochastic image colorizers use seed 3407 separately for each crop.

Parameter Counts. Table 1 in the main paper reports all parameters in the networks used for inference, including frozen weights and excluding training-only heads. Image colorizers and EventSeeColor exclude the shared E2VID front end. Image colorizer counts also exclude independent caption generation. V2V-E2VID + DDColor includes both reconstruction and colorization networks, and EvDiff includes its event encoder. LongE2V includes the T5 text encoder even though its empty-prompt embedding is cached, with fused adaptation weights counted only once. Colorful Image Colorization. The released model [65] operates at 256×256 and combines predicted chroma with the input image’s Lab lightness.

DDColor. The large ModelScope checkpoint [26] operates at 512 × 512 and combines predicted chroma with the input image’s Lab lightness.

ColorFormer. The released generator [25] uses the GLH backbone and color and semantic memories at 256 × 256.

BigColor. The epoch-11 exponential-moving-average generator [27] operates at 256 × 256. Its grayscale classifier selects the highest-probability class, and a Gaussian latent with standard deviation 0.8 supplies the generative input. The released grayscale-lightness fusion is retained.

Control Color. The original diffusion model and deformable decoder [34] operate at 512 × 512, with 20 DDIM steps, guidance scale 7, and deterministic DDIM sampling. Control Color and L-CAD share scene descriptions generated from grayscale by a frozen vision-language model [3]. These descriptions provide neither target colors nor RGB references.

L-CAD. The COCO language checkpoint [6] operates at 256 × 256, with 50 DDIM steps, guidance scale 4.5, and DDIM noise parameter zero. Its released Lab lightness normalization and standard sampling procedure are retained.

DeOldify. The stable ResNet-101 generator and official colorization filter [1] use render factor 35, corresponding to a 560 × 560 model input, with postprocessing enabled.

CT<sup>2</sup>. The released ViT-large checkpoint [57] operates at 256×256, with 313 color tokens, four luminance-prior bins, and classification decoding at temperature 0.38. Its auxiliary chroma inputs are set to zero.

DISCO. The recommended beta checkpoint [58] operates at 256 × 256, with eight anchors, 16 superpixels, hidden width 64, dense positions, and the enhanced output head. Its test-time sampling temperature is zero. Its auxiliary chroma inputs are set to zero.

UniColor. The ImageNet checkpoint [23] operates at 256× 256 in automatic mode, with empty strokes and no reference image. Sampling uses a 16 × 16 token grid, 256 autoregressive steps, and top-100 selection. The official input

Table A2. Network and inference settings. Lightness calibration is specified in Appendix B.1.
<table><tr><td rowspan=1 colspan=1>Component</td><td rowspan=1 colspan=1>Setting</td><td rowspan=1 colspan=1>Value</td></tr><tr><td rowspan=1 colspan=1>Inputs</td><td rowspan=1 colspan=1>Aligned crop / diffusion imageEvent bins per polarity / total channelsVoxel scale / count scale</td><td rowspan=1 colspan=1> $2 5 6 \times 2 5 6 / 5 1 2 \times 5 1 2$  $1 0 / 2 4$  $\tau _ { \mathrm { v o x e l } } = 1 6 , \tau _ { \mathrm { c o u n t } } = 6 4$ </td></tr><tr><td rowspan=1 colspan=1>EGF</td><td rowspan=1 colspan=1>Stage widths / stridesStage operator / normalizationGate and projection kernels / hint size</td><td rowspan=1 colspan=1> $\left( 1 6 , 3 2 , 6 4 \right) / \left( 1 , 2 , 2 \right)$ 3 × 3 convolution, 4-group GroupNorm, SiLU $1 \times 1 / 3 2 0 \times 6 4 \times 6 4$ </td></tr><tr><td rowspan=1 colspan=1>LoRA</td><td rowspan=1 colspan=1>Attention projections / rank / alpha</td><td rowspan=1 colspan=1>Query, key, value, output / 4 / 4</td></tr><tr><td rowspan=1 colspan=1>Auxiliary chroma head</td><td rowspan=1 colspan=1>Pointwise channel widths</td><td rowspan=1 colspan=1>320 → 32 → 2, intermediate SiLU</td></tr><tr><td rowspan=1 colspan=1>Text condition</td><td rowspan=1 colspan=1>PromptCLIP input length</td><td rowspan=1 colspan=1>Fixed across images, Appendix B.277 tokens including padding</td></tr><tr><td rowspan=1 colspan=1>BCR</td><td rowspan=1 colspan=1>Image branch widths / stridesEvent branch / context widthsContext dilation / output headsGain parameters / residual bound</td><td rowspan=1 colspan=1> $5  1 6  3 2 / ( 2 , 2 )$  $2 4 \to 1 6 / 4 8 \to 3 2 \to 3 2$ (1, 2) / two 32 → 2 pointwise convolutions $\gamma _ { + } = 0 . 5 , \gamma _ { - } = 1 , \rho = 1 2$ </td></tr><tr><td rowspan=1 colspan=1>LDR</td><td rowspan=1 colspan=1>Shallow widths / context widthsContext strides / dilationResidual decoder / operatorActivation / RGB correction bound</td><td rowspan=1 colspan=1> $6  1 6  1 6 / 1 6  3 2  3 2$  $\left( 2 , 1 \right) / \left( 1 , 2 \right)$  $4 8 \to 1 6 \to 3 / 3 \times 3$ convolution $\mathrm { S i L U } / \delta = 0 . 1 2$ </td></tr><tr><td rowspan=1 colspan=1>Sampling</td><td rowspan=1 colspan=1>DDIM steps / stochasticityClassifier-free guidance / hint scaleSelf-attention guidance / timestep cutoff</td><td rowspan=1 colspan=1>20 /07/10.05 / 600</td></tr><tr><td rowspan=1 colspan=1>Rendering</td><td rowspan=1 colspan=1>Chroma gamut bisection iterations</td><td rowspan=1 colspan=1>18</td></tr></table>

Table A3. Training settings and objective weights. Region thresholds are expressed in Lab chroma units.
<table><tr><td rowspan=1 colspan=1>Component</td><td rowspan=1 colspan=1>Setting</td><td rowspan=1 colspan=1>Value</td></tr><tr><td rowspan=1 colspan=1>Chroma objectives</td><td rowspan=1 colspan=1>Chroma scale / pixel weightingRegion thresholds / warm hueAuxiliary chroma / region weightsShortfall / neutral / warm weights</td><td rowspan=1 colspan=1> $s = 3 2 , \eta = 2 , \kappa = 2 0$  $\tau _ { n } = 5 , \tau _ { c } = 1 0 , [ \dot { h } _ { - } , h _ { + } ] = [ - 3 0 ^ { \circ } , 7 5 ^ { \circ } ]$  $\lambda _ { \mathrm { a u x } } = 0 . 1 , \beta = 0 . 5$  $\lambda _ { s } = 0 . 2 , \lambda _ { n } = 0 . 5 , \lambda _ { w } = 0 . 2$ </td></tr><tr><td rowspan=1 colspan=1>LDR objective</td><td rowspan=1 colspan=1>RGB / gradient / color weightsPerceptual / clipping weightsCharbonnier constant</td><td rowspan=1 colspan=1> $\lambda _ { \mathrm { r g b } } = 1 , \lambda _ { \mathrm { g r a d } } = 0 . 1 , \lambda _ { \mathrm { c o l o r } } = 0 . 2 5$  $\lambda _ { \mathrm { p e r c } } = 0 . 1 , \lambda _ { \mathrm { c l i p } } = 0 . 1$  $\varepsilon _ { \mathrm { r g b } } = 1 0 ^ { - 3 }$ </td></tr><tr><td rowspan=1 colspan=1>Diffusion colorizer</td><td rowspan=1 colspan=1>Epochs / batch / accumulationAdamW learning ratesWeight decay</td><td rowspan=1 colspan=1> $4 / 1 / 4$ LoRA $1 0 ^ { - 5 } .$ , EGF and auxiliary chroma head $1 0 ^ { - 4 }$ 0</td></tr><tr><td rowspan=1 colspan=1>BCR</td><td rowspan=1 colspan=1>Epochs / batch / optimizerLearning rate / weight decay</td><td rowspan=1 colspan=1> $2 0 / 1 6 / \mathrm { A d a m W }$  $1 0 ^ { - 4 } / 1 0 ^ { - 4 }$ </td></tr><tr><td rowspan=1 colspan=1>LDR</td><td rowspan=1 colspan=1>Epochs / batch / optimizerLearning rate / weight decay</td><td rowspan=1 colspan=1> $2 0 / 8 / \mathrm { A d a m W }$  $1 0 ^ { - 4 } / 1 0 ^ { - 4 }$ </td></tr><tr><td rowspan=1 colspan=1>All stages</td><td rowspan=1 colspan=1>Gradient clipping / training seed</td><td rowspan=1 colspan=1> $1 / 3 4 0 7$ </td></tr></table>

EvDiff. Native event streams pass through the recurrent

conversion and color resizing are retained.

V2V-E2VID + DDColor. V2V-E2VID [35] uses the released epoch-77 reconstruction checkpoint trained on 10,000 videos, with unnormalized five-bin event voxels and continuous recurrent history. Each reconstructed target crop is percentile-normalized and colorized by the frozen DD-Color model specified above. Native frame intervals are processed continuously, including non-target frames.

five-bin event encoder [29], followed by the released SD3 Medium model with rank-64 adaptation, bundled prompt embeddings, and a one-step flow update at timestep 680. Evaluation uses FP32 arithmetic and the released stochastic sampling behavior with seed 3407. The recurrent state, including the temporal-fusion state, is reset at sequence boundaries and carried through every intervening frame before the target. EvDiff and V2V-E2VID + DDColor use the $2 0 0 0 0 \mu \mathrm { s }$ PEOD timestamp offset and native CoSEC times-

tamps.

LongE2V. The released adapter and CogVideoX-5B-I2V backbone [14] operate in reconstruction mode. Three-bin signed event voxels undergo the released nonzero normalization and are resized to 480 × 720. Sampling uses the empty text prompt, 20 denoising steps, guidance scale 0, BF16 arithmetic, and seed 6666 at each sequence boundary. Each block reconstructs 49 frames from 48 event intervals, with 20 context frames and adaptive context switching at threshold 0.05 with one retry. All intervals are processed from the start of each sequence. The initial reference input is a constant image whose latents are discarded by the reconstruction path. Subsequent context uses generated grayscale frames. Evaluation uses RGB outputs before grayscale conversion, with grayscale feedback preserved. For target frames inside each block, reconstruction also uses later events from that block. Postprocessing that fits brightness to target-image statistics is omitted. Outputs are resized bilinearly to the native sensor geometry before applying the common alignment and crop. The resulting native RGB outputs are evaluated under the protocol in $\mathsf { A p - }$ pendix C.2.

## D.3. Recent Baselines Not Included

SeAda [13], Beyond Fixed Luminance [37], Color-Turbo [30], and ColorFLUX [28] are excluded from the quantitative comparison because publicly accessible official pretrained checkpoints could not be located as of 15 October 2026.

## D.4. Ablation Settings

The component and event-guidance rows in Table 2 in the main paper compare complete trained configurations. In w/o EGF Event Path, the EGF event branch is removed while BCR and LDR retain their event inputs. In w/o BCR, the entire BCR stage, including chroma refinement and rendering at calibrated event lightness, is bypassed. The diffusion colorizer’s RGB output is passed directly to a retrained LDR. In w/o Event Inputs (EGF/BCR/LDR), the event branches in EGF and BCR and the event activity channel in LDR are removed. The remaining grayscale input is still reconstructed from events. The grayscale-only diffusion colorizer and BCR use the same training budgets as their counterparts with event inputs, followed by training the corresponding LDR. In w/o LDR, evaluation uses the intermediate BCR RGB image I<sup>r</sup>. All component configurations share data, validity masks, spatial alignment, and RGB targets.

The training-objective rows use the supervision names in Section 3.4. w/o Warm Supervision sets the BCR warmregion weight $\lambda _ { w }$ to zero. w/o Auxiliary Chroma Supervision removes the training-only auxiliary chroma head and its loss $\mathcal { L } _ { \mathrm { a u x } }$ . The three LDR supervision configurations set λ<sub>perc</sub>, $\lambda _ { \mathrm { c o l o r } } .$ , or $\lambda _ { \mathrm { g r a d } }$ to zero, respectively. These weights correspond to LDR perceptual, color, and gradient supervision in Eq. 8.

The achromatic-input experiment replaces the BCR RGB image $I ^ { r }$ with an achromatic image rendered at the same calibrated event lightness L. LDR retains G, A, and m as inputs and can learn bounded RGB corrections. The LDR training schedule and objective match those of the other LDR configurations.

## D.5. Efficiency Measurement

Parameter Overhead. Under the counting scope in Appendix D.2, Control Color contains 1,523,608,805 parameters and EventSeeColor contains 1,523,980,060. The difference is 371,255 parameters, or 0.0244% of Control Color’s inference parameters. These additions comprise 133,216 parameters in EGF, 181,248 in LoRA, 32,340 in BCR, and 24,451 in LDR. The auxiliary chroma head adds 10,338 parameters during training and is omitted at inference. All pretrained diffusion weights remain frozen during adaptation.

Sampling and Timing. EventSeeColor and the Control Color baseline both use 512 × 512 diffusion inputs, 20 DDIM steps, and classifier-free guidance scale 7. BCR and LDR operate at the 256 × 256 crop resolution. LDR timing uses FP32 arithmetic and batch size one on an NVIDIA RTX 5090, with GPU synchronization immediately before and after each forward pass. Excluding the first invocation, its mean latency over 2,219 evaluation crops is 0.886 ms. Inputs are already on the GPU when timing starts. This measurement covers LDR alone and excludes grayscale reconstruction, diffusion sampling, BCR, data transfer, file I/O, and metric computation.

## E. Additional Results

## E.1. Distribution and Appearance

Table A4 complements the paired reconstruction metrics with distribution statistics and full-image color differences. EvDiff leads FID and KID, followed by EventSeeColor. EventSeeColor obtains the lowest CF-MAE, showing closer per-image colorfulness to the corresponding target. EvDiff and LongE2V obtain the lowest full-image CIEDE2000 scores. Combined with Table A5, this result highlights the contribution of lightness and large neutral regions to a whole-image color difference. EventSeeColor also achieves the lowest NIQE, while several image colorizers have lower BRISQUE scores.

## E.2. Regional Color Accuracy

Table A5 reports chroma RMSE separately for chromatic, neutral, and warm target regions. EventSeeColor has the lowest chromatic-region and warm-region chroma RMSE while limiting strong warm coloration of neutral pixels. Methods with lower desaturation can introduce larger chroma errors or stronger false color, so desaturation is interpreted together with the corresponding chroma RMSE columns. The nearly achromatic LongE2V achieves the lowest full-image and neutral-region chroma RMSE while discarding much of the color in target chromatic regions.

## E.3. Component, Event Guidance, and Supervision Results

Data Adaptation and Fusion. Table A6 separates paireddata adaptation from additional fusion and includes the complete method as a reference. The three adapted Control Color variants start from the same weights and use 8,068 training crops, four epochs, 8,068 optimizer updates, and seed 3407. LoRA updates 181,248 parameters. The grayscale and event–grayscale fusion controls each add a 97,376-parameter adapter and evaluate the diffusion output before BCR or LDR. These four Control Color conditions share grayscale-derived scene descriptions and diffusion settings. Their predicted chroma is rendered with the same calibrated event lightness and gamut mapping. The final row applies this common rendering and scoring procedure to the complete EventSeeColor output. Its native RGB results, including the lightness predicted by LDR, are reported in Table 1 in the main paper.

LoRA alone reduces ab RMSE by 24.21% relative to frozen Control Color. Adding a grayscale adapter reduces the error by a further 3.45%, and event–grayscale fusion improves upon the grayscale adapter by 3.22%. Relative to LoRA alone, event–grayscale fusion reduces ab RMSE by 6.56% and also improves PSNR and SSIM. These results show that fusion provides reconstruction gains beyond paired-data adaptation alone. The complete EventSee-Color configuration further improves ab RMSE, approximate $\Delta E _ { 7 6 }$ , and PSNR under this common scoring protocol, while the event–grayscale fusion control retains the highest SSIM.

Full-Pipeline Configurations. EventSeeColor has lower chromatic CIEDE2000, higher SSIM, and lower LPIPS and DISTS than the configuration without event inputs to EGF, BCR, and LDR in Table 2 in the main paper. That configuration has higher PSNR. Removing only the EGF event path produces smaller differences from EventSeeColor and a lower desaturation rate.

The achromatic-input experiment achieves PSNR 16.086, SSIM 0.6497, and LPIPS 0.3313, but its chromaticregion chroma RMSE is 10.368 and its desaturation rate is 77.574%. EventSeeColor achieves chromatic-region chroma RMSE 8.894 and desaturation 36.603%. These measurements distinguish whole-image similarity from successful target-color recovery. The configuration without LDR has the lowest desaturation in Table 2 in the main paper, while EventSeeColor has stronger RGB fidelity and perceptual agreement.

The supervision comparisons in the same table clarify the roles of the LDR objectives. Perceptual supervision improves LPIPS and DISTS, while color supervision reduces chromatic CIEDE2000. Removing perceptual supervision improves PSNR and SSIM, and removing color supervision improves DISTS and FID, reflecting the different optimization targets. These complementary gains, together with the visual comparisons in Appendix E.5, support their joint use to balance regional color accuracy and local detail.

## E.4. Additional Qualitative Comparisons

Figure A1 compares native RGB outputs on foliage and road scenes. In rows (a) - (c), EventSeeColor recovers green vegetation while preserving the separation between foliage, tree trunks, and neutral pavement. DDColor, ColorFormer, and Control Color introduce different magenta, cyan, or warm casts. EvDiff produces green foliage with fine texture, although the reconstructed contours and local texture can differ from the target. LongE2V remains largely achromatic in these examples.

Rows (d) - (f) examine road markings and asphalt. EventSeeColor retains the marking geometry and a relatively neutral road appearance. Several colorization baselines apply strong colors to these surfaces. EvDiff can produce smooth road texture but changes some marking intensities and shapes, while LongE2V introduces regular texture patterns. Compared with the RGB targets, fine road texture in the EventSeeColor outputs is smoother, with differences in local brightness.

Rows (g) - (i) extend the comparison to a building facade, a guardrail, and a vehicle. EventSeeColor separates green vegetation from relatively neutral structures, while several colorizers introduce strong casts on the facade, rail, or vehicle body.

Appendix E.2 quantifies regional color accuracy and desaturation, complementing these visual comparisons.

## E.5. Qualitative Component and Supervision Comparisons

Figure A2 visualizes the component and event-guidance configurations in Table 2 in the main paper. In rows (a) and (b), EventSeeColor retains green foliage, while several configurations produce weaker green or a purple or brown cast. The configuration without BCR produces a pronounced color shift in these examples. Without LDR, the images are softer and local contrast is lower. In row (c), EventSeeColor limits green tint on the guardrail while preserving its boundary. Row (d) shows the separation between roadside vegetation, the vehicle, and the road.

Figure A3 compares the supervision configurations. Rows (a) and (b) show foliage and sky color differences when BCR warm-region supervision or auxiliary chroma supervision is removed. Removing LDR perceptual supervision softens leaf boundaries, building detail, and the small road patches in rows (c) and (d). Removing LDR color supervision changes color intensity, whereas removing LDR gradient supervision produces subtler visual differences in these examples. Our visual assessment of these supervision configurations finds that the full EventSeeColor configuration provides the best overall visual quality through its combined color fidelity and local detail.

![](images/2f691d85d2d6d3387e281c2b2b4294277a3a1024fafd16f2da0f94cdf2e2085a.jpg)  
Figure A1. Additional comparisons with external baselines. Rows (a) - (c) show vegetation, (d) - (f) road markings and texture, and (g) - (i) a facade, guardrail, and vehicle. Native RGB outputs use identical crop coordinates and display scales, with the RGB target as reference.

Table A4. Distribution, full-image color difference, and auxiliary appearance metrics on native RGB outputs. KID mean and subse standard deviation are multiplied by $\mathrm { 1 0 ^ { 3 } }$ . Lower values are better for all metrics. The same input groups as Table 1 in the main paper apply.
<table><tr><td>Method</td><td>FID↓</td><td> $\mathrm { K I D \times 1 0 ^ { 3 } \downarrow }$ </td><td>CF-MAE↓</td><td> $\mathrm { A l l } \ : \Delta E _ { 0 0 } .$  →</td><td>Neutral  $\Delta E _ { 0 0 }$  ↓</td><td>NIQE↓</td><td>BRISQUE↓</td></tr><tr><td colspan="8">Image colorization from reconstructed grayscale</td></tr><tr><td>Colorful Image Colorization</td><td>207.114</td><td> $1 5 1 . 5 1 1 \pm 2 . 9 5 9$ </td><td>19.054</td><td>22.490</td><td>21.671</td><td>6.766</td><td>14.711</td></tr><tr><td>DDColor</td><td>175.876</td><td> $1 1 0 . 6 3 3 \pm 2 . 7 6 5$ </td><td>26.032</td><td>23.284</td><td>22.345</td><td>6.814</td><td>14.688</td></tr><tr><td>ColorFormer</td><td>138.680</td><td> $7 8 . 9 4 6 \pm 2 . 3 5 9$ </td><td>18.210</td><td>20.664</td><td>20.126</td><td>6.850</td><td>14.378</td></tr><tr><td>BigColor</td><td>138.394</td><td> $8 0 . 0 1 1 \pm 2 . 4 3 9$ </td><td>18.944</td><td>21.000</td><td>20.380</td><td>6.730</td><td>14.469</td></tr><tr><td>Control Color</td><td>156.302</td><td> $9 4 . 6 0 6 \pm 2 . 8 6 8$ </td><td>20.150</td><td>20.301</td><td>19.762</td><td>7.222</td><td>27.166</td></tr><tr><td>L-CAD</td><td>147.627</td><td> $8 6 . 3 3 8 \pm 2 . 4 8 2$ </td><td>25.378</td><td>22.493</td><td>21.719</td><td>6.653</td><td>14.156</td></tr><tr><td>DeOldify</td><td>181.201</td><td> $1 1 6 . 0 0 5 \pm 2 . 5 9 3$ </td><td>11.383</td><td>20.358</td><td>19.813</td><td>6.764</td><td>14.400</td></tr><tr><td>CT2</td><td>164.058</td><td> $1 0 9 . 1 0 8 \pm 2 . 7 6 7$ </td><td>14.845</td><td>19.958</td><td>19.548</td><td>6.763</td><td>14.706</td></tr><tr><td>DISCO</td><td>155.947</td><td> $9 5 . 1 4 5 \pm 2 . 3 2 5$ </td><td>15.962</td><td>18.964</td><td>18.117</td><td>6.764</td><td>14.422</td></tr><tr><td>UniColor</td><td>142.251</td><td> $8 5 . 1 7 6 \pm 2 . 5 9 6$ </td><td>25.192</td><td>21.949</td><td>21.353</td><td>6.768</td><td>14.948</td></tr><tr><td colspan="8">Grayscale reconstruction followed by colorization</td></tr><tr><td>V2V-E2VID + DDColor</td><td>150.953</td><td> $9 3 . 6 9 4 \pm 2 . 7 8 9$ </td><td>20.360</td><td>20.996</td><td>20.106</td><td>8.331</td><td>30.597</td></tr><tr><td>Event video reconstruction</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>EvDiff</td><td>102.178 185.881</td><td> $5 3 . 1 3 8 \pm 1 . 9 2 0$ </td><td>6.837 9.614</td><td>11.846</td><td>10.969</td><td>9.186</td><td>28.092</td></tr><tr><td>LongE2V</td><td></td><td> $1 1 6 . 8 5 2 \pm 3 . 3 9 4$ </td><td></td><td>13.618</td><td>12.025</td><td>10.561</td><td>40.801</td></tr><tr><td>EventSeeColor (Ours)</td><td>117.286</td><td> $\underline { { 6 1 . 3 0 7 } } \pm 2 . 4 2 2$ </td><td>5.494</td><td>15.101</td><td>14.828</td><td>6.326</td><td>28.134</td></tr></table>

Table A5. Chroma error and regional color behavior of native RGB outputs. Chroma RMSE occupies the first four numeric columns. The final columns report desaturation, neutral false color, and warm color leakage in percent. Lower is better.
<table><tr><td>Method</td><td>All ab↓</td><td> $\operatorname { C h r . } a b \downarrow$ </td><td>Neutral ab↓</td><td>Warm ab↓ |</td><td>Desat. (%) ↓</td><td>False color (%) ↓</td><td>Warm leak (%) ↓</td></tr><tr><td colspan="8">Image colorization from reconstructed grayscale</td></tr><tr><td>Colorful Image Colorization</td><td>14.839</td><td>14.252</td><td>13.963</td><td>12.632</td><td>11.114</td><td>34.237</td><td>6.347</td></tr><tr><td>DDColor</td><td>16.170</td><td>16.626</td><td>15.267</td><td>14.954</td><td>11.492</td><td>38.603</td><td>21.012</td></tr><tr><td>ColorFormer</td><td>11.870</td><td>13.402</td><td>11.221</td><td>14.960</td><td>18.978</td><td>18.761</td><td>8.791</td></tr><tr><td>BigColor</td><td>14.921</td><td>15.956</td><td>14.050</td><td>16.398</td><td>17.408</td><td>33.901</td><td>1.575</td></tr><tr><td>Control Color</td><td>12.662</td><td>13.687</td><td>11.824</td><td>12.962</td><td>22.304</td><td>27.229</td><td>17.823</td></tr><tr><td>L-CAD</td><td>15.026</td><td>17.727</td><td>14.057</td><td>16.814</td><td>16.662</td><td>33.114</td><td>23.802</td></tr><tr><td>DeOldify</td><td>10.893</td><td>12.376</td><td>10.131</td><td>12.835</td><td>25.813</td><td>19.430</td><td>1.313</td></tr><tr><td>CT2</td><td>11.699</td><td>13.738</td><td>10.927</td><td>14.699</td><td>20.504</td><td>22.523</td><td>2.217</td></tr><tr><td>DISCO</td><td>10.326</td><td>12.225</td><td>8.872</td><td>13.515</td><td>67.306</td><td>20.039</td><td>4.499</td></tr><tr><td>UniColor</td><td>15.487</td><td>16.110</td><td>14.723</td><td>17.239</td><td>20.601</td><td>36.801</td><td>8.508</td></tr><tr><td colspan="8">Grayscale reconstruction followed by colorization</td></tr><tr><td>V2V-E2VID + DDColor</td><td>13.967</td><td>16.600</td><td>12.864</td><td>16.239</td><td>10.766</td><td>27.901</td><td>10.627</td></tr><tr><td>Event video reconstruction</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>EvDiff</td><td>4.791</td><td>9.411</td><td>3.639</td><td>11.423</td><td>55.410</td><td>2.035</td><td>0.016</td></tr><tr><td>LongE2V</td><td>4.671</td><td>11.049</td><td>2.175</td><td>12.432</td><td>97.685</td><td>0.012</td><td>0.007</td></tr><tr><td>EventSeeColor (Ours)</td><td>4.777</td><td>8.894</td><td>3.602</td><td>10.907</td><td>36.603</td><td>0.057</td><td>0.002</td></tr></table>

Table A6. Data adaptation, fusion, and the complete method with common fixed-lightness outputs on 2,220 evaluation crops. Best and second-best values are bold and underlined.
<table><tr><td>Configuration</td><td>ab RMSE ↓</td><td>Approx.  $\Delta E _ { 7 6 } \downarrow$ </td><td>PSNR ↑</td><td>SSIM ↑</td></tr><tr><td>Frozen Control Color</td><td>12.8797</td><td>26.0718</td><td>14.3973</td><td>0.6007</td></tr><tr><td>LoRA only</td><td>9.7619</td><td>23.5487</td><td>14.6786</td><td>0.6193</td></tr><tr><td>LoRA + gray adapter</td><td>9.4248</td><td>23.1856</td><td>14.7649</td><td>0.6194</td></tr><tr><td>LoRA + event-gray fusion</td><td>9.1218</td><td>22.9109</td><td>14.8167</td><td>0.6209</td></tr><tr><td>EventSeeColor (full)</td><td>4.7769</td><td>19.5217</td><td>15.1163</td><td>0.6176</td></tr></table>

![](images/ca3249e39d519b6ab959f89ee31c97f1758f4798c139193fc47f27adff910f06.jpg)  
Figure A2. Qualitative component and event-guidance comparisons. The configurations correspond to Table 2 in the main paper. Rows (a) and (b) show foliage, row (c) shows a guardrail, and row (d) shows a roadside vehicle. Each row uses the same crop and scale.

![](images/f31069a05f4ba7ecbff07d0d75c4fe49df343c173b085fe63d1740f48a3892c1.jpg)  
Figure A3. Qualitative supervision comparisons. Rows (a) and (b) show foliage, building, and sky regions. Rows (c) and (d) show road markings and texture. Each row uses the same crop and scale, with the full EventSeeColor configuration and RGB target in the final two columns.