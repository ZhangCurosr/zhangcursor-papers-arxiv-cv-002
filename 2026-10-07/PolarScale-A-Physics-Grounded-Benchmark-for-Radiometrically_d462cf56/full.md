# PolarScale: A Physics-Grounded Benchmark for Radiometrically Consistent RGB-to-Stokes Estimation

Beibei Lin<sup>1†</sup> Tingting Chen<sup>1†</sup> Xin Zhang<sup>1</sup> Wenhao Zhao<sup>1</sup> Dongjun Li<sup>2</sup> Zifeng Yuan<sup>1∗</sup> <sup>1</sup>National University of Singapore <sup>2</sup>University of Michigan, Ann Arbor

## Abstract

Polarization imaging provides physical cues beyond intensity imaging but typically requires specialized hardware. Recent methods infer polarization from RGB-like inputs, yet predict only normalized Stokes components or relative descriptors, from which the radiometric scale needed for full Stokes reconstruction has been divided out. We introduce PolarScale, a benchmark that makes this scale an explicit prediction and evaluation target. Built on existing trichromatic full-Stokes measurements, PolarScale takes the per-scene normalized total-intensity image s<sub>0</sub> (a scene-referred linear image, not a consumer sRGB photograph) and asks models to predict normalized Stokes components, AoLP/DoLP/DoCP, and a perscene scale. Because the scale is divided out of the input, it is not physically identifiable; PolarScale therefore evaluates dataset-conditioned semantic scale estimation against a constant-scale control, together with angular, self-consistency, and physical-bound metrics. Across seven restoration-based and generative backbones and three prediction strategies, the strongest restoration models estimate the scale with 3.6–4.3% mean relative error versus 5.7% for the constant control and violate physical bounds on fewer than 0.25% of pixels, whereas two generative baselines collapse to a near-zero scale; explicit descriptor supervision improves descriptor accuracy (23.66 vs. 18.88 dB PSNR for MAE). Predicted full-Stokes representations improve diffuse/specular separation, material segmentation, and glare classification, although in diffuse/specular separation the learned scale performs only on par with the constant control.

## 1 Introduction

Polarization imaging captures scene properties by measuring the polarization state of light, which is not observable with conventional RGB sensing. This additional polarization information provides cues about physical surface characteristics such as birefringence, surface stress, and micro-scale roughness. As a result, polarization imaging has enabled progress in challenging vision tasks, including reflection separation [1, 2, 3], material recognition [4, 5, 6, 7, 8], and shadow removal [9], complementing RGB-based region understanding [10, 11] and semantic segmentation [12]. Beyond vision, polarization is also a key degree of freedom in photonic hardware, where the polarization state of lasers can encode binary spins for photonic Ising computing [13, 14, 15].

However, acquiring polarization data typically requires dedicated polarization-sensitive hardware such as integrated linear polarizers and retarders [16, 17], and even controlling polarization at the source requires careful device engineering, such as engineered mesa geometries and integrated metasurfaces in polarization-controlled lasers [18, 19]. These systems increase sensor cost, reduce spatial resolution due to multiplexed designs, complicate calibration, and are difficult to integrate into large-scale or mobile vision platforms. Recovering absolute, radiometrically scaled Stokes vectors is more demanding still, since it additionally requires radiometric calibration of the polarimetric system. In practical deployment scenarios, such as robotics, autonomous systems, consumer imaging, and wearable sensing [20], these constraints significantly limit the accessibility and scalability of polarization imaging.

![](images/4dad93eddf4c5fef9a69285786ee31ae74aebba5bafe4e9c3fa8378c5345e05d.jpg)  
Figure 1: Overview of PolarScale. Prior RGB-based methods [22, 21] predict normalized Stokes components or relative polarimetric descriptors, from which the radiometric scale has been divided out. PolarScale additionally predicts the per-scene radiometric scale σ and reconstructs the full Stokes components $\mathrm { S } _ { n } = \sigma \mathrm { s } _ { n } .$ , making the scale an explicit evaluation target. The input is the normalized total-intensity image $\mathrm { s } _ { 0 } = \mathrm { S } _ { 0 } / \sigma$ (Section 3.2), not a consumer sRGB photograph.

To overcome these barriers, recent methods infer polarization information directly from RGB-like images using neural networks [21, 22]. However, these approaches predict normalized Stokes components or relative polarimetric descriptors (Figure 1), from which the radiometric scale has been divided out. Such outputs describe relative polarimetric structure but cannot be turned into absolute Stokes vectors, which radiance-level applications require. For example, polarization-based diffuse/specular separation attributes the polarized radiance to the specular component [23, 24, 1], and glare depends on the amount of polarized light reaching the sensor rather than on the fraction of light that is polarized. Whether such absolute quantities can be estimated from intensity images has not yet been measured.

In this work, we introduce PolarScale, a benchmark that makes the radiometric scale an explicit prediction and evaluation target. PolarScale is built on the trichromatic full-Stokes measurements of the Spectral-Polarization dataset [16]. Its input is the normalized total-intensity image $\mathrm { s } _ { 0 } = \mathrm { S } _ { 0 } / \sigma$ a scene-referred linear intensity image rather than a consumer sRGB photograph; we keep the name “RGB-to-Stokes” established by [22] for continuity. Models predict the normalized Stokes components $\{ \mathrm { s } _ { 1 } , \mathrm { s } _ { 2 } , \mathrm { s } _ { 3 } \}$ , the descriptors AoLP, DoLP, and DoCP, and the per-scene scale σ, from which the full Stokes components follow as $\mathrm { S } _ { n } = \sigma \mathrm { s } _ { n }$ . Because σ is divided out of the input, it cannot be recovered by physical inversion; as in metric monocular depth estimation, it can only be inferred from scene content. PolarScale therefore measures dataset-conditioned semantic scale estimation, and, like diagnostic benchmarks that isolate specific failure modes rather than report only aggregate scores [25], we evaluate it accordingly: scale errors are scored against a constant-scale control, and wrap-aware angular, Stokes self-consistency, and physical-bound metrics complement standard image-quality scores.

Benchmarking seven restoration-based and generative backbones under three prediction strategies (direct, joint, and decoupled), we find that the strongest restoration models estimate the scale better than the constant control and nearly satisfy physical bounds, whereas two generative baselines collapse to a near-zero scale, and that full-Stokes PSNR is a poor measure of scale recovery. Three downstream probes show that predicted full-Stokes representations are useful, while scale-attribution controls show that the contribution of the learned scale itself is mixed, identifying scale accuracy as the open problem PolarScale is designed to track.

Our main contributions are summarized as follows:

Table 1: PolarScale compared with related resources. PolarScale re-uses the measurements of [16]; it differs in what is predicted and evaluated.
<table><tr><td>Resource</td><td>Prediction target</td><td>Radiometric scale evaluated?</td></tr><tr><td>Jeon et al. [16]</td><td>N/A (physical acquisition, no prediction task)</td><td>N/A</td></tr><tr><td>Lin et al. [22]</td><td>normalized Stokes  ${ \mathrm { s } } _ { 1 } , { \mathrm { s } } _ { 2 } , { \mathrm { s } } _ { 3 }$ </td><td>No (scale divided out)</td></tr><tr><td>PolarAnything [21]</td><td>relative descriptors (AoLP, DoLP)</td><td>No (scale-invariant by definition)</td></tr><tr><td>PolarScale (ours)</td><td> $\mathrm { s _ { 1 } \mathrm { - s _ { 3 } , A o L P / \bar { D } o L P / D o C P , } }$  and  $\sigma  \mathrm { S _ { 1 } - S _ { 3 } }$ </td><td>Yes (explicit target, with constant-σ control)</td></tr></table>

• Task reformulation: PolarScale makes the radiometric scale an explicit prediction and evaluation target of RGB-to-Stokes estimation, and states its input modality and identifiability scope explicitly.

• Physics-aware evaluation protocol: we score scale estimation against a constant-scale control and add wrap-aware AoLP angular error, Stokes self-consistency, and physicalbound violation metrics. All of them ship in a standalone benchmark package with the preprocessed splits, scale values, derived targets, evaluation scripts, and a datasheet.

• Benchmark analysis and design guidance: we evaluate seven restoration-based and generative backbones under three prediction strategies, showing that explicit descriptor supervision and decoupled decoding help, and characterizing failure modes such as scale collapse.

• Downstream validation: we show that predicted full-Stokes representations benefit three downstream tasks, and use scale-attribution controls to separate the contribution of the learned scale from that of the normalized polarization structure.

## 2 Related work

Polarization datasets Several polarization datasets have been developed to support polarization analysis and related vision tasks. Early datasets [26, 27, 6, 28] are mostly small-scale and captured in controlled indoor environments using trichromatic polarization cameras. Later datasets target specific applications, such as reflection separation [2, 29] and transparent object segmentation [5]. Fan et al. [30] introduce a full-Stokes dataset containing both linear and circular polarization components, but it is limited to 64 planar objects. Jeon et al. [16] provide a large-scale spectro-polarimetric dataset with trichromatic and hyperspectral full-Stokes measurements over more than 2,000 real-world scenes under diverse illumination conditions. Its full-Stokes measurements retain the radiometric scale, which makes it suitable for constructing PolarScale.

RGB-based polarization inference Recent methods infer polarization information directly from RGB inputs [22, 21]. A recent benchmark [22] establishes a standardized protocol for predicting normalized Stokes components from RGB images, while PolarAnything [21] introduces a diffusion based framework for polarimetric image synthesis. Methodologically, existing approaches can be broadly grouped into restoration-based regression and generative image translation. Restoration-based methods formulate polarization inference as deterministic pixel-wise prediction, using architectures such as Restormer [31], Uformer [32], and MAE [33]. Generative methods instead model polarization inference as conditional synthesis, using diffusion-based models such as WDiff [34], DiT [35], RealFill [36], and I2ITurbo [37]. These methods are formulated and evaluated on normalized Stokes components or relative descriptors, so the radiometric scale is never predicted or scored. Table 1 summarizes how PolarScale differs: it is built on the existing measurements of [16] rather than on a new capture, and its contribution lies in what is predicted and evaluated.

Polarization-related tasks Polarization provides complementary cues beyond RGB imaging for material, reflection, and geometry analysis. Because reflected polarization varies with surface properties such as roughness and coating, it has been used for material classification and object recognition [4]. Polarization also separates diffuse and specular reflection [23, 1], suppresses specular reflections, and enhances transparent or translucent objects [38]. In geometry estimation, polarization depends on surface orientation and therefore supports surface normal estimation and Shape-from-Polarization reconstruction [39], complementing intensity-based 3D reconstruction and weathergeneralized monocular depth estimation [40], whose robustness under degraded capture conditions such as raindrops is itself an active benchmarking topic [41]. Beyond these tasks, polarization has also been used for reflectance and appearance modeling [42, 43, 44], semantic segmentation [45], sparse sensing [46], and wearable robotics [47], and polarization switching in lasers enables drive-efficient GHz-rate binary-state updates [48] and all-optical annealing [49]. Tasks that operate on polarized radiance, such as reflection separation, need absolute rather than normalized Stokes information (Section 4.5).

![](images/ccca97f44edc7e31e089f8bda370cac2ca8aef0b3f853d67a4fe1275319950bb.jpg)  
Figure 2: Overview of the three prediction strategies evaluated in our benchmark. (a) Direct Stokes Estimation predicts σ and normalized Stokes components, then derives AoLP, DoLP, and DoCP. (b) Joint Estimation predicts all polarimetric parameters with a unified decoder. (c) Decoupled Multi-decoder separately estimates scale-dependent components and scale-independent descriptors. The input in all three strategies is the normalized total-intensity image $\mathrm { s _ { 0 } }$

## 3 PolarScale benchmark

## 3.1 Preliminaries: Stokes representation

Full Stokes parameters We represent the polarization state using the Stokes vector [22, 50, 51, 52, 53] (notation in Appendix $\mathbf { A } )$ , characterized by four parameters $\mathrm { S _ { 0 } , S _ { 1 } , S _ { 2 } , S _ { 3 } }$ . Specifically, S<sub>0</sub> denotes the total light intensity; $\mathrm { S } _ { 1 }$ corresponds to the intensity difference between $0 ^ { \circ }$ and 90<sup>◦</sup> linear polarization; ${ \bar { \mathrm { S } } } _ { 2 }$ represents the difference between $+ 4 5 ^ { \circ }$ and $- 4 5 ^ { \circ }$ linear polarization; and $\mathrm { S _ { 3 } }$ indicates the difference between right- and left-handed circular polarization [54, 55]. A physically realizable Stokes vector satisfies

$$
\sqrt { \mathrm { S _ { 1 } ^ { 2 } + S _ { 2 } ^ { 2 } + S _ { 3 } ^ { 2 } } } \leq \mathrm { S _ { 0 } } .\tag{1}
$$

Interpretable polarization features Commonly used derived representations are the degree of linear polarization (DoLP), the degree of circular polarization (DoCP), and the angle of linear polarization (AoLP) [56]:

$$
{ \mathrm { D o L P } } = { \frac { \sqrt { \mathrm { S _ { 1 } ^ { 2 } + \mathrm { S _ { 2 } ^ { 2 } } } } } { \mathrm { S _ { 0 } } } } , \quad { \mathrm { D o C P } } = { \frac { \mathrm { S _ { 3 } } } { \mathrm { S _ { 0 } } } } ,\tag{2}
$$

$$
{ \mathrm { A o L P } } = { \frac { 1 } { 2 } } \arctan \left( { \frac { \mathrm { S } _ { 2 } } { \mathrm { S } _ { 1 } } } \right) .\tag{3}
$$

DoLP quantifies the proportion of linearly polarized light, whereas DoCP reflects the contribution of circular polarization. AoLP describes the orientation of the polarization ellipse; it lies in $[ - 9 0 ^ { \circ } , 9 0 ^ { \circ } ]$ and is periodic with period $1 8 0 ^ { \circ }$ , so its errors must be measured with wrap-around (Section 3.4). All three descriptors are ratios of Stokes components and are therefore invariant to the radiometric scale.

## 3.2 Task formulation

Revisiting polarization inference from RGB images. Existing methods learn a mapping $\mathcal { F } _ { i n v }$ $\mathbf { I }  \mathbf { y } _ { i n v }$ whose output is scale-free: normalized Stokes components $\{ \mathrm { s } _ { 1 } , \mathrm { s } _ { 2 } , \mathrm { s } _ { 3 } \}$ in [22], or relative descriptors such as AoLP and DoLP in [21]. Normalized components are obtained from the full components by a normalization operator, either sample-wise min-max normalization [22] or global scaling by the scene’s maximum intensity. Global scaling preserves the relative radiance ratios within a scene, but in both cases the radiometric scale is removed, so evaluating these outputs cannot assess whether a model recovers absolute Stokes signals.

Input modality. For each scene, PolarScale uses global scaling by the per-scene radiometric scale

$$
\sigma = \mathrm { m a x } \mathrm { S } _ { 0 } , \qquad \mathrm { s } _ { k } = \mathrm { S } _ { k } / \sigma , \quad k \in \{ 0 , 1 , 2 , 3 \} ,\tag{4}
$$

where the maximum is taken over the scene’s $\mathrm { S } _ { 0 }$ measurement. The network input is the normalized total-intensity image $\mathrm { s } _ { 0 } \in \mathbb { R } ^ { H \times W \times 3 }$ of the trichromatic Stokes arrays of [16]. This is a scene-referred linear intensity image, not a consumer sRGB photograph: it does not pass through the nonlinear tone mapping of a display-referred camera image signal processor (ISP), and it is captured through the polarimetric optics of the source camera. Following [22], we refer to the task as “RGB-to-Stokes” estimation; the harder consumer-RGB-to-Stokes problem, which adds ISP nonlinearities and polarization-dependent sensor responses, is outside the scope of the benchmark (Section 5).

Prediction targets. Given $\mathrm { s _ { 0 } } ,$ , a model predicts

$$
\begin{array} { r } { { \mathbf y } _ { v a r } = \{ \mathrm { s _ { 1 } } , \mathrm { s _ { 2 } } , \mathrm { s _ { 3 } } , \sigma , \mathrm { A o L P } , \mathrm { D o L P } , \mathrm { D o C P } \} , } \end{array}\tag{5}
$$

that is, a polarimetric representation $\mathbf { P } \in \mathbb { R } ^ { H \times W \times 1 9 }$ consisting of the normalized Stokes components $\{ \mathrm { s } _ { 1 } , \mathrm { s } _ { 2 } , \mathrm { s } _ { 3 } \} \in \mathbb { R } ^ { H \times W \times \mathbf { \dot { 9 } } }$ , a scale map $\sigma \in \mathbb { R } ^ { H \times W \times 1 }$ that follows the dense-prediction interface of the baselines, and the descriptors {AoLP, DoL $\mathbf { P } , \mathbf { D o C P } \} \in \mathbb { R } ^ { H \times W \times 9 }$ . The full Stokes components are reconstructed as

$$
\mathrm { S } _ { n } = \sigma \cdot \mathrm { s } _ { n } , \quad n \in \{ 1 , 2 , 3 \} .\tag{6}
$$

Identifiability and scope. Because $\mathrm { s _ { 0 } }$ is normalized by the per-scene $\sigma ( \mathrm { E q . } ( 4 ) )$ , σ has been divided out of the input and is not identifiable by physical inversion: a globally rescaled scene produces the same input. Since the input is linear rather than display-referred, the ambiguity is a single global factor with no tone-mapping component. The scale remains learnable only through $p ( \sigma$ scene content), because absolute radiance correlates with content such as the illumination type and the indoor/outdoor context; metric monocular depth estimation is ambiguous in the same way and is likewise learned from semantic priors [57, 58], and evaluating models on targets that are underdetermined by their inputs is an emerging benchmarking theme in its own right [59]. PolarScale therefore evaluates dataset-conditioned semantic scale estimation rather than the recovery of radiometric information contained in the input. Every scale result must accordingly be read against a constant-scale control that predicts the training prior (Section 3.4), and the scale distribution of the source data bounds what the benchmark can measure (Section 5).

## 3.3 Benchmark construction

Source data PolarScale is built on the Spectral-Polarization dataset [16], which provides 2,022 trichromatic and 311 hyperspectral full-Stokes images of real-world scenes. The trichromatic Stokes images have a resolution of $\bar { 2 1 } 0 0 \times 1 9 2 0$ and are captured in a single shot by a division-of-focal-plane RGB full-Stokes polarimeter with on-sensor quarter-wave plates and linear polarizers [17], so they include the circular component $\mathrm { S _ { 3 } } .$ . Each scene carries labels for its environment (indoor or outdoor), illumination (clear or cloudy sunlight, white or incandescent light), capture time, and type (objector scene-oriented). We use the trichromatic subset, whose full-Stokes measurements retain the radiometric scale needed to define σ. Appendix B gives further details.

Targets and split For each scene, we compute σ and the normalized Stokes vector $\{ \mathrm { s _ { 0 } , s _ { 1 } , s _ { 2 } , s _ { 3 } } \}$ by Eq. (4), and derive AoLP, DoLP, and DoCP by Eqs. (2) and (3). The set $\{ \sigma , \mathrm { s _ { 1 } , s _ { 2 } , s _ { 3 } , A o L P } _ { \mathrm { : } }$ , DoLP, DoCP} supervises both scale-dependent and scale-independent quantities. We adopt the 1,000/200 train/test scene split of Lin et al. [22], which keeps our results directly comparable with that protocol; the split is fixed by construction and involves no sampling seed. The scale distribution is relatively narrow: the training-set median and mean of $\sigma$ are 4.85 and 4.25, and predicting the median for every test scene already gives a 5.7% mean relative error (Table 3). Appendix B.3 summarizes the scale statistics.

Prediction strategies No existing method is designed for this setting, so we adapt representative restoration-based and generative architectures from prior RGB-based polarization studies [22, 21] and compare three strategies for modeling scale-dependent components $\{ \sigma , \mathrm { s } _ { 1 } , \mathrm { s } _ { 2 } , \mathrm { s } _ { 3 } \}$ and scaleindependent descriptors {AoLP, DoLP, DoCP} (Figure 2):

(a) Direct Stokes estimation predicts σ and $\{ \mathrm { s } _ { 1 } , \mathrm { s } _ { 2 } , \mathrm { s } _ { 3 } \}$ and derives AoLP, DoL $\mathbf { \delta } _ { P } ,$ and DoCP by Eqs. (2) and (3), which keeps the output compact but depends on the stability of these transformations; (b) joint estimation predicts all quantities with a unified decoder, so both types are directly supervised; (c) decoupled multi-decoder estimation uses two parallel decoders for $\{ \sigma , \mathrm { s } _ { 1 } , \mathrm { s } _ { 2 } , \mathrm { s } _ { 3 } \}$ and {AoLP, DoLP, DoCP}, which encourages specialized representations for physically distinct quantities.

Benchmark package We release PolarScale as a standalone package at https://anonymous. 4open.science/r/PolarScale, containing the preprocessed splits and scene lists, per-scene σ values, normalized Stokes components and descriptors, evaluation scripts for all metrics of Section 3.4 (including the constant-σ control), baseline code and configurations, and the datasheet of Appendix F, so users need not rebuild the benchmark from [16].

## 3.4 Evaluation protocol

Image-quality metrics Following [22], we report PSNR and SSIM for the normalized Stokes components, the descriptors, and the reconstructed full Stokes components, with the PSNR peak fixed to $\mathrm { M A X } _ { I } = 1 . 0$ for all models and scenes. Because $\mathrm { s } _ { n }$ is bounded while $\mathrm { S } _ { n } = \sigma \mathrm { s } _ { n }$ is not, PSNR values are not comparable across these groups, and we do not average them. LPIPS rescales its inputs and is therefore scale-invariant (identical for $\mathrm { s } _ { n }$ and $\mathrm { S } _ { n } )$ , so we report it once and do not use it as evidence about scale.

Physics-aware metrics Let σˆ denote the predicted per-scene scale. We report: (i) the relative scale error $| \hat { \sigma } - \sigma | / \sigma$ (mean and median over scenes), the log-scale error $| \log _ { 1 0 } \hat { \sigma } - \log _ { 1 0 } \sigma |$ over scenes with $\hat { \sigma } > 0 \jmath _ { \bar { \imath } }$ , and the number of physically meaningless predictions $\hat { \sigma } \le 0 ; ( \mathrm { i i } )$ the AoLP angular error, computed with wrap-around at period $\mathrm { i } 8 0 ^ { \circ }$ and restricted to pixels with ground-truth DoL $. \mathrm { { P } > 0 . 1 }$ since AoLP is undefined for unpolarized light; (iii) Stokes self-consistency: the angular error between the directly predicted AoLP and the AoLP recomputed from the predicted $\mathrm { s } _ { 1 } , \mathrm { s } _ { 2 }$ , and analogously the mean absolute error for DoLP and DoCP; (iv) the physical-bound violation rate: the fraction of pixels whose predicted Stokes vectors violate Eq. (1), i.e., $\sqrt { \mathrm { s _ { 1 } ^ { 2 } + \mathrm { s _ { 2 } ^ { 2 } + \mathrm { s _ { 3 } ^ { 2 } } } } } > \mathrm { s _ { 0 } }$ . We measure (iv) on Stokes-derived values because the directly predicted descriptor maps are bounded by their encoding.

Constant-scale control To test whether a model learns more than the dataset prior, we pair its predicted normalized Stokes with a constant σ equal to the training-set median (4.85) or mean (4.25), and compare the result with the learned σˆ. The median control is the reference that any learned scale should beat.

## 4 Experiments

## 4.1 Implementation details

Baselines We evaluate the restoration-based methods Restormer [31], Uformer [32], and MAE [33], and the generative methods WDiff [34], DiT [35], RealFill [36], and I2ITurbo [37], each under the three strategies of Section 3.3.

The restoration models, WDiff, and DiT use native multi-channel regression heads whose output matches the strategy (10 channels for (a), 19 for (b) and (c)); for (c) we add a second decoder branch for the descriptors. For RealFill and I2ITurbo, changing the output dimensionality would disturb the pre-trained weights that are their main strength, so we steer them with task-specific text prompts (e.g., "S1", "Scale", "AoLP") and, for (c), with two independent sets of LoRA parameters. Their inference uses one pass per component, so every model predicts the same components exactly once. Appendix C gives the full per-strategy details.

Table 2: Image-quality results for normalized Stokes components $\left( \mathrm { { s } _ { 1 } \mathrm { { - s } _ { 3 } } } \right)$ , descriptors, and reconstructed full Stokes components $\mathrm { ( S _ { 1 } - S _ { 3 } ) }$ . PSNR uses $\mathrm { M A X } _ { I } = 1$ , so values are not comparable across column groups; <sup>†</sup>LPIPS is scale-invariant (identical for $\mathrm { s } _ { n }$ and $\mathrm { S } _ { n } )$ and reported once. Table 3 gives the wrap-aware AoLP error. Best and second-best results are highlighted.
<table><tr><td rowspan="2">Methods</td><td colspan="3"> $\mathrm { S 1 } , \mathrm { S 2 } , \mathrm { S 3 }$ </td><td colspan="3">AoLP, DoLP, DoCP</td><td colspan="2"> $\mathrm { S _ { 1 } , S _ { 2 } , S _ { 3 } }$ </td></tr><tr><td>PSNR↑</td><td>SSIM↑</td><td> $\mathbf { L P I P S ^ { \dagger } \downarrow }$ </td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>PSNR↑</td><td>SSIM↑</td></tr><tr><td>WDiff DiT</td><td>22.39</td><td>0.6202</td><td>0.6910</td><td>12.81</td><td>0.2800</td><td>0.7193</td><td>18.30</td><td>0.6410</td></tr><tr><td rowspan="3">RealFill</td><td>36.32</td><td>0.9339</td><td>0.4107</td><td>18.02</td><td>0.4547</td><td>0.5378</td><td>20.01</td><td>0.7557</td></tr><tr><td>38.14</td><td>0.9630</td><td>0.2078</td><td>21.23</td><td>0.5078</td><td>0.4247</td><td>16.47</td><td>0.6109</td></tr><tr><td>38.45</td><td>0.9735</td><td>0.3236</td><td>21.49</td><td>0.5738</td><td>0.4891</td><td>18.47</td><td>0.6655</td></tr><tr><td>Restormer</td><td>40.22</td><td>0.9746</td><td>0.1783</td><td>22.54</td><td>0.6499</td><td>0.3634</td><td>20.21</td><td>0.7856</td></tr><tr><td>Uformer</td><td>40.29</td><td>0.9755</td><td>0.1862</td><td>22.80</td><td>0.6509</td><td>0.3867</td><td>20.29</td><td>0.7853</td></tr><tr><td>MAE</td><td>41.11</td><td>0.9790</td><td>0.1864</td><td>23.66</td><td>0.6823</td><td>0.3725</td><td>21.07</td><td>0.8136</td></tr></table>

Table 3: Physics-aware metrics on the 200 test scenes. $\# \hat { \sigma } { \leq } 0$ counts scenes with a non-positive scale, which are excluded from the log error. AoLP errors wrap at $1 8 0 ^ { \circ }$ on pixels with ground-truth DoLP $> 0 . 1 ;$ self-consistency compares predicted descriptors with those recomputed from the predicted Stokes components; bound violation is the percentage of pixels violating Eq. (1). The constant-σ row (training median) is the floor any learned scale should beat.
<table><tr><td>Method</td><td>Rel. scale err. (%) mean / med ↓</td><td> $\log _ { 1 0 }$   $\operatorname { e r r . } \downarrow$ </td><td> $\# \hat { \sigma } { \leq } 0$  ↓</td><td>AoLP  $\operatorname { e r r . } \downarrow$ </td><td>AoLP self-cons. ↓</td><td>DoLP/DoCP self-cons. ↓</td><td>Bound viol.  $( \% ) \downarrow$ </td></tr><tr><td>Const. σ (median)</td><td>5.7/4.5</td><td></td><td>一</td><td></td><td></td><td></td><td></td></tr><tr><td>WDiff</td><td>92.8 / 93.1</td><td>1.195</td><td>11</td><td> $4 1 . 0 ^ { \circ }$ </td><td> $4 3 . 7 ^ { \circ }$ </td><td>0.442 / 0.613</td><td>58.1</td></tr><tr><td>DiT</td><td>99.9 / 99.8</td><td>1.319</td><td>97</td><td> $3 8 . 4 ^ { \circ }$ </td><td> $3 6 . 5 ^ { \circ }$ </td><td>0.189 / 0.115</td><td>5.18</td></tr><tr><td>RealFill</td><td>38.0 / 41.8</td><td>0.135</td><td>0</td><td> $3 2 . 4 ^ { \circ }$ </td><td> $4 4 . 6 ^ { \circ }$ </td><td>0.158 / 0.115</td><td>3.73</td></tr><tr><td>I2ITurbo</td><td>5.3 / 2.9</td><td>0.022</td><td>0</td><td> $3 2 . 5 ^ { \circ }$ </td><td> $5 2 . 4 ^ { \circ }$ </td><td>0.139 / 0.117</td><td>3.56</td></tr><tr><td>Restormer</td><td>5.1 / 2.4</td><td>0.021</td><td>0</td><td> $3 1 . 1 ^ { \circ }$ </td><td> $2 4 . 4 ^ { \circ }$ </td><td>0.057 / 0.045</td><td>0.22</td></tr><tr><td>Uformer</td><td>4.3 / 2.5</td><td>0.017</td><td>0</td><td> $3 0 . 7 ^ { \circ }$ </td><td> $3 0 . 0 ^ { \circ }$ </td><td>0.066 / 0.047</td><td>0.22</td></tr><tr><td>MAE</td><td>3.6 / 2.1</td><td>0.014</td><td>0</td><td>26.1°</td><td> $2 4 . 0 ^ { \circ }$ </td><td>0.054 / 0.045</td><td>0.18</td></tr></table>

Training All baselines were trained to convergence on the same server with eight NVIDIA RTX A5000 GPUs (24 GB each): about one day per restoration model and two days per diffusion-based model. Per-model hyperparameters are provided in the configuration files of the code release, and Appendix C describes a native-conditioning variant of RealFill and I2ITurbo. In Table 2, the rows for DiT, I2ITurbo, Uformer, and MAE correspond to their configuration (c) in Table 4.

## 4.2 Main results

Table 2 reports image-quality metrics and Table 3 the physics-aware metrics. MAE leads on most image-quality columns, consistent with the benefit of pre-trained visual representations for polarization modeling. Descriptor accuracy remains limited for all models: even with direct supervision, the best AoLP angular error is 26.1<sup>◦</sup>.

Scale estimation against the constant control MAE and Uformer clearly beat the constant-σ floor (3.6% and 4.3% mean relative error versus 5.7%), and Restormer and I2ITurbo beat it more narrowly (5.1% and 5.3%). RealFill, DiT, and WDiff fall well below the floor: DiT and WDiff collapse to a near-zero σˆ, which is non-positive in 97 and 11 of the 200 scenes, while RealFill settles on a large over-estimate. MAE’s estimate is not merely a re-learned constant: the deviation of its σˆ from the training prior correlates with the ground-truth deviation $( r = 0 . 4 3 )$ , indicating a content-to-scale mapping beyond the dataset average. However, on scenes whose σ departs strongly from the prior, the models regress toward it, as expected for dataset-conditioned scale estimation.

Physical consistency The restoration models are nearly physically consistent, with bound violations below 0.25% of pixels, whereas the generative models violate the bound on 3.6–58% of pixels. Selfconsistency separates the models in the same way: MAE’s directly predicted AoLP deviates by $2 4 . 0 ^ { \circ }$ from the AoLP implied by its own Stokes prediction, versus $5 2 . 4 ^ { \circ }$ for I2ITurbo, whose promptconditioned components are predicted in separate passes. None of these differences is visible in the full-Stokes PSNR of Table 2.

Table 4: Effect of the prediction strategy: (A) direct Stokes estimation, (B) joint estimation, and (C) decoupled multi-decoder estimation, across backbones. Metrics and conventions follow Table 2; PSNR values are not comparable across column groups, and <sup>†</sup>LPIPS is scale-invariant. Bold marks the best strategy per backbone and column.
<table><tr><td rowspan="2">Methods</td><td rowspan="2">Type</td><td colspan="3"> $\mathrm { S 1 } , \mathrm { S 2 } , \mathrm { S 3 }$ </td><td colspan="3">AoLP, DoLP, DoCP</td><td colspan="2"> $\mathrm { S _ { 1 } , S _ { 2 } , S _ { 3 } }$ </td></tr><tr><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS† ↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>PSNR↑</td><td>SSIM↑</td></tr><tr><td>DiT</td><td>A</td><td>34.73</td><td>0.8825</td><td>0.4548</td><td>12.79</td><td>0.7471</td><td>0.4548</td><td>20.04</td><td>0.7471</td></tr><tr><td>DiT</td><td>B</td><td>35.91</td><td>0.9484</td><td>0.3468</td><td>18.55</td><td>0.4741</td><td>0.5376</td><td>20.01</td><td>0.7446</td></tr><tr><td>DiT</td><td>C</td><td>36.32</td><td>0.9339</td><td>0.4107</td><td>18.02</td><td>0.4547</td><td>0.5378</td><td>20.01</td><td>0.7557</td></tr><tr><td>I2ITurbo</td><td>A</td><td>38.89</td><td>0.9733</td><td>0.3154</td><td>16.48</td><td>0.3836</td><td>0.6245</td><td>18.93</td><td>0.7019</td></tr><tr><td>I2ITurbo</td><td>B</td><td>34.98</td><td>0.9397</td><td>0.4091</td><td>21.77</td><td>0.5463</td><td>0.5399</td><td>15.17</td><td>0.5263</td></tr><tr><td>I2ITurbo</td><td>C</td><td>38.45</td><td>0.9735</td><td>0.3236</td><td>21.49</td><td>0.5738</td><td>0.4891</td><td>18.47</td><td>0.6655</td></tr><tr><td>Uformer</td><td>A</td><td>40.28</td><td>0.9756</td><td>0.1869</td><td>19.25</td><td>0.4920</td><td>0.4396</td><td>20.25</td><td>0.7844</td></tr><tr><td>Uformer</td><td>B</td><td>39.99</td><td>0.9742</td><td>0.1954</td><td>22.73</td><td>0.6496</td><td>0.3866</td><td>19.97</td><td>0.7695</td></tr><tr><td>Uformer</td><td>C</td><td>40.29</td><td>0.9755</td><td>0.1862</td><td>22.80</td><td>0.6509</td><td>0.3867</td><td>20.29</td><td>0.7853</td></tr><tr><td>MAE</td><td>A</td><td>40.77</td><td>0.9778</td><td>0.2456</td><td>18.88</td><td>0.5116</td><td>0.4819</td><td>20.78</td><td>0.7940</td></tr><tr><td>MAE</td><td>B</td><td>40.95</td><td>0.9787</td><td>0.2251</td><td>23.54</td><td>0.6715</td><td>0.4041</td><td>20.94</td><td>0.8078</td></tr><tr><td>MAE</td><td>C</td><td>41.11</td><td>0.9790</td><td>0.1864</td><td>23.66</td><td>0.6823</td><td>0.3725</td><td>21.07</td><td>0.8136</td></tr></table>

Full-Stokes PSNR is not a scale measure The constant-scale control (Appendix D, Table 8) shows that full-Stokes PSNR barely depends on whether the scale is learned. Replacing MAE’s learned σˆ by the training median changes its full-Stokes PSNR from 21.08 to 21.13 dB, and the training mean, whose scale error is about three times larger (16.2%), yields an even higher PSNR for every model. With a fixed peak value, PSNR on $\mathrm { S } _ { n }$ is dominated by the normalized structure and rewards shrinking the reconstruction. The control also exposes the opposite failure: WDiff’s collapsed σˆ shrinks its reconstructions and inflates its full-Stokes PSNR, which falls from 18.30 to 2.65 dB once an honest constant scale is applied. We therefore do not interpret the gap between normalized and full-Stokes PSNR in Table 2 as evidence about scale recovery, and use the scale metrics of Table 3 as the primary scale measures.

## 4.3 Analysis

Effect of the prediction strategy Table 4 compares the three strategies. Explicit descriptor supervision is the clearest effect: moving from direct estimation (A) to joint (B) or decoupled (C) estimation improves descriptor PSNR for every backbone (e.g., MAE: 18.88 → 23.54/23.66 dB; I2ITurbo: 16.48 → 21.77/21.49 dB). Joint estimation can trade this gain against the scale-dependent outputs, most visibly for I2ITurbo, whose normalized-Stokes PSNR drops from 38.89 to 34.98 dB. Decoupled estimation avoids this trade-off for the restoration backbones: for MAE it gives the best value in every column, and for Uformer it is the best or within 0.0001 of the best in every column. We therefore recommend explicit descriptor supervision with decoupled decoding as the default design.

Restoration-based versus generative backbones Under our adaptation protocols, the evaluated restoration backbones outperform the evaluated generative ones on normalized-Stokes and descriptor accuracy (Table 2) and on scale estimation and physical consistency (Table 3). The generative baselines show two distinct failure modes: WDiff and DiT, which use native regression heads, collapse the scale toward zero, and RealFill over-estimates it. I2ITurbo is the exception on scale (5.3% error, comparable to Restormer), but it trails on physical consistency. Retraining RealFill and I2ITurbo with native multi-channel heads that predict all seven outputs in one pass (Appendix C, Table 7) proved the weaker adaptation: I2ITurbo’s scale error rises from 5.3% to 8.8%, and RealFill’s native head collapses the scale $( \hat { \sigma } \le 0$ in 21 of 200 scenes; $r = - 0 . 0 7$ with the true scale); the only gain is self-consistency from single-pass prediction (I2ITurbo: $2 7 . 4 ^ { \circ }$ versus 52.4<sup>◦</sup>). We therefore limit this conclusion to the evaluated architectures and adaptation protocols rather than claiming a paradigm-level gap.

![](images/fb046148901ff63ed2b1c8ed34599e60139f86027ede093bdbc47755bd0e7f86.jpg)  
Figure 3: Reconstructed full Stokes components for test scene 1043 (input: normalized ${ \mathrm { s } } _ { 0 } )$ . The sky carries strong $\mathrm { S _ { 2 } }$ and $\mathrm { S _ { 3 } }$ signals: MAE [33] recovers their extent and sign, Uformer [32] only faint edges, and DiT [35] and I2ITurbo [37] predict nearly flat maps.

Table 5: Downstream validation. (a) Diffuse/specular separation, ${ \mathrm { A b s R e l } } = \operatorname* { m e a n } ( | { \hat { y } } - y | / y ) ;$ the last four rows share the same predicted $\hat { \mathrm { S } } _ { n }$ and differ only in the scale. (b) Material segmentation on MCubeS [6]; settings differ only in the polarization input. Glare results are in the text.  
(a) Diffuse/specular separation (AbsRel)  
(b) Material segmentation
<table><tr><td>Method</td><td>Diffuse↓</td><td>Specular↓</td></tr><tr><td>RGB-direct regression</td><td>0.190</td><td>0.508</td></tr><tr><td> $\hat { \mathrm { S } } _ { n } \gets$  - shuffled σ</td><td>0.133</td><td>0.448</td></tr><tr><td>ên + constant σ (median)</td><td>0.118</td><td>0.448</td></tr><tr><td> $\hat { \mathrm { { S } } } _ { n }$  + learned ô (ours)</td><td>0.118</td><td>0.445</td></tr><tr><td>ên + ground-truth σ</td><td>0.100</td><td>0.440</td></tr></table>

<table><tr><td>Polarization input</td><td>mIoU↑</td><td> $\mathbf { A c c _ { c l s } } \uparrow$ </td><td>fwIoU↑</td></tr><tr><td>Base</td><td>0.322</td><td>0.421</td><td>0.600</td></tr><tr><td>+ pred. DoLP/AoLP</td><td>0.330</td><td>0.417</td><td>0.617</td></tr><tr><td>+ pred.  $\hat { \mathrm { S } } _ { n }$  (no scale)</td><td>0.355</td><td>0.446</td><td>0.628</td></tr><tr><td>+ pred.  $\hat { \mathrm { { S } } } _ { n }$  + ô (ours)</td><td>0.374</td><td>0.468</td><td>0.637</td></tr></table>

## 4.4 Qualitative evaluation

Figure 3 shows a representative result; Appendix G gives further scenes, including descriptor maps (Figures 11 and 12). MAE produces the most accurate and spatially consistent predictions and is the only model that recovers the polarized sky in scene 1043, though it predicts the wrong sign of $\mathrm { S } _ { 1 }$ for the sky in scene 1050. All models struggle with fine structures and abrupt polarization transitions.

## 4.5 Downstream validation

Tasks We run three probes of whether predicted absolute Stokes information is useful beyond reconstruction quality (Table 5; protocols in Appendix E). (i) Diffuse/specular separation. Following the classical polarization-based separation [23, 1], the polarized part of the reflected radiance is attributed to the specular component and the unpolarized remainder to the diffuse component, both in absolute radiance. A baseline regresses the six absolute channels (diffuse and specular RGB) directly, whereas we compose them analytically from the predicted $\hat { \mathrm { { S } } } _ { n }$ and a scale; holding $\hat { \mathrm { s } } _ { n }$ fixed, we vary the scale source (shuffled across scenes, constant training median, learned ${ \hat { \sigma } } ,$ or ground truth as an upper bound) to attribute its effect. (ii) Material segmentation on MCubeS [6], whose 20 classes are annotated independently of polarization: a base input without polarization is augmented with different predicted polarization representations. (iii) Glare-level classification. Glare is an inherently absolute quantity: it depends on the amount of polarized light reaching the sensor, not on the fraction of light that is polarized. Each test scene receives a glare level from the absolute polarized radiance of its ground-truth Stokes vectors, which is predicted either from image appearance alone or from the predicted absolute Stokes.

Results The predicted full-Stokes representation improves all three tasks over the tested RGB or relative-polarization baselines: diffuse/specular AbsRel drops from 0.190/0.508 to 0.118/0.445, mIoU rises from 0.330 with predicted DoLP/AoLP to 0.374, and glare accuracy over the 200 test scenes rises from 48.5% (Wilson 95% interval [60] [41.7, 55.4]) with image appearance to 63.0% ([56.1, 69.4]) with predicted absolute Stokes. The contribution of the learned scale itself is mixed.

In diffuse/specular separation, the learned σˆ performs on par with the constant control (0.118/0.445 versus 0.118/0.448) and thus does not consistently beat the dataset prior, whereas the ground-truth scale gives a clear improvement (0.100/0.440) and a shuffled scale degrades the diffuse estimate (0.133): an accurate absolute scale matters, but current estimates are not yet accurate enough to exploit it. In material segmentation, adding σˆ to the predicted normalized Stokes raises mIoU from 0.355 to 0.374, but without constant- or shuffled-scale variants this does not uniquely isolate the per-scene scale; the glare probe has no scale control. Our claim is therefore that predicted full-Stokes representations benefit the evaluated tasks, with the absolute scale as one contributing component whose accuracy remains the main open problem.

## 5 Limitations and broader impacts

Scale is not physically identifiable Because σ is divided out of the input, PolarScale measures dataset-conditioned semantic scale estimation; predicted scales and reconstructed $\mathrm { S } _ { n }$ are not calibrated radiometry, and models regress toward the training prior on scenes whose scale departs strongly from it.

Single source dataset and narrow scale distribution All data come from one dataset [16] captured with one polarimetric camera, and its scale distribution is narrow: a constant predictor already reaches 5.7% mean relative error, leaving a small margin for learned models (3.6% at best). Scale results may not transfer to other sensors, exposures, or scene distributions, motivating domain generalization and test-time adaptation [61].

Input modality and scene conditions The input is a normalized linear image captured through polarimetric optics; consumer sRGB images add ISP nonlinearities (tone mapping, white balance, clipping) and polarization-dependent sensor responses. No dataset pairs consumer RGB with absolutescale Stokes ground truth, so this setting cannot yet be evaluated quantitatively. Low light, strong specular highlights, and adverse weather [62, 63, 64], for which dedicated benchmarks exist in other vision tasks [65, 66, 67], are not covered.

Scope of the conclusions The restoration-versus-generative comparison holds only for the evaluated architectures and adaptation protocols. All results come from a single training run per configuration on a fixed split; we report Wilson intervals for the glare probe. The downstream probes are smallscale, two lack full scale controls, and the scale-attribution result is mixed. All models still fail on fine structures and abrupt polarization transitions (Figures 3 and 11; AoLP errors above 26<sup>◦</sup>), which physics-informed constraints may address.

Broader impacts PolarScale may benefit polarization-aware perception in robotics, material analysis, remote sensing, 3D reconstruction [68], image completion [69] and computational imaging by reducing reliance on specialized hardware, with medical image segmentation [70] and whole-slide classification [71] as further applications to investigate, and lightweight estimators could eventually run on emerging photonic accelerators for convolutional networks [72]. Its main risk is over-reliance on inferred polarization in safety-critical settings such as driver assistance: predicted scales are dataset-conditioned estimates, not measurements, and should be validated under the target sensor, lighting, and domain before high-stakes use. The benchmark is not designed for person identification or surveillance.

## 6 Conclusion

We introduced PolarScale, a benchmark that makes the radiometric scale an explicit target of RGBto-Stokes estimation and, because the scale is divided out of the input, evaluates dataset-conditioned scale estimation against a constant-scale control together with angular, self-consistency, and physicalbound metrics. The strongest evaluated restoration models beat the dataset prior and nearly satisfy physical bounds, two generative baselines collapse the scale, and explicit descriptor supervision with decoupled decoding gives the most balanced predictions. Predicted full-Stokes representations benefit three downstream tasks, while the learned scale is not yet accurate enough to consistently beat the prior; PolarScale makes this gap measurable for future physically grounded RGB-to-Stokes models.

## References

[1] Shree K Nayar, Xi-Sheng Fang, and Terrance Boult. Separation of reflection components using color and polarization. International Journal ofComputer Vision, 21(3):163–186, 1997.

[2] Chenyang Lei, Xuhua Huang, Mengdi Zhang, Qiong Yan, Wenxiu Sun, and Qifeng Chen. Polarized reflection removal with perfect alignment in the wild. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 1750–1758, 2020.

[3] Youxin Pang, Mengke Yuan, Qiang Fu, Peiran Ren, and Dong-Ming Yan. Progressive polarization based reflection removal via realistic training data generation. Pattern Recognition, 124:108497, 2022.

[4] Lawrence B. Wolff. Polarization-based material classification from specular reflection. IEEE transactions on pattern analysis and machine intelligence, 12(11):1059–1071, 1990.

[5] Haiyang Mei, Bo Dong, Wen Dong, Jiaxi Yang, Seung-Hwan Baek, Felix Heide, Pieter Peers, Xiaopeng Wei, and Xin Yang. Glass segmentation using intensity and spectral polarization cues. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 12622–12631, 2022.

[6] Yupeng Liang, Ryosuke Wakaki, Shohei Nobuhara, and Ko Nishino. Multimodal material segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 19800– 19808, 2022.

[7] Fucai Ke, Zhixi Cai, Simindokht Jahangard, Weiqing Wang, Pari Delir Haghighi, and Hamid Rezatofighi. Hydra: A hyper agent for dynamic compositional visual reasoning. In European Conference on Computer Vision, pages 132–149. Springer, 2024.

[8] Fucai Ke, Joy Hsu, Zhixi Cai, Zixian Ma, Xin Zheng, Xindi Wu, Sukai Huang, Weiqing Wang, Pari Delir Haghighi, Gholamreza Haffari, et al. Explain before you answer: A survey on compositional visual reasoning. arXiv preprint arXiv:2508.17298, 2025.

[9] Chu Zhou, Chao Xu, and Boxin Shi. Polarization guided mask-free shadow removal. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 10716–10724, 2025.

[10] Xin Zhang, Haochen Wang, Yikang Zhou, Zhuochen Wang, Xiangtai Li, and Robby T. Tan. Actor as its own critic: Unifying region understanding and localization via CycleGRPO. In European Conference on Computer Vision (ECCV), volume 17044 of Lecture Notes in Computer Science, pages 586–604. Springer, 2026.

[11] Xin Zhang, Jinheng Xie, Yuan Yuan, Michael Bi Mi, and Robby T Tan. HEAP: Unsupervised object discovery and localization with contrastive grouping. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pages 7323–7331, 2024.

[12] Xin Zhang and Robby T Tan. Mamba as a bridge: Where vision foundation models meet vision language models for domain-generalized semantic segmentation. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14527–14537, 2025.

[13] Yuan Gao, Guanyu Chen, Luo Qi, Wujie Fu, Zifeng Yuan, and Aaron J. Danner. Photonic Ising machines for combinatorial optimization problems. Applied Physics Reviews, 11(4):041307, 2024.

[14] Fucai Ke, Weiqing Wang, Weicong Tan, Lan Du, Yuan Jin, Yujin Huang, and Hongzhi Yin. Hitskt: A hierarchical transformer model for session-aware knowledge tracing. Knowledge-Based Systems, 284:111300, 2024.

[15] Zifeng Yuan, Tingting Chen, Dewen Zhang, Yuan Gao, Wenkai Shan, Beibei Lin, and Aaron Danner. Tailored polarization-switchable VCSEL arrays for photonic Ising computing. Applied Physics Letters, 127(22):221102, 2025.

[16] Yujin Jeon, Eunsue Choi, Youngchan Kim, Yunseong Moon, Khalid Omer, Felix Heide, and Seung-Hwan Baek. Spectral and polarization vision: Spectro-polarimetric real-world dataset. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22098–22108, 2024.

[17] Xingzhou Tu, Scott McEldowney, Yang Zou, Matthew Smith, Christopher Guido, Neal Brock, Sawyer Miller, Linan Jiang, and Stanley Pau. Division of focal plane red–green–blue full-Stokes imaging polarimeter. Appl. Opt., 59(22):G33–G40, Aug 2020.

[18] Zifeng Yuan, Wenkai Shan, Tingting Chen, Beibei Lin, and Aaron Danner. Mesa orientation engineering for polarization locking in VCSELs. In 2025 IEEE Photonics Conference (IPC), pages 1–2, 2025.

[19] Wenjie Chen, Zifeng Yuan, Hong-Lin Lin, Luo Qi, Jiaru Chu, Aaron Danner, and Yuhang Chen. Metasurface-integrated VCSEL designed for polarization control in optical Ising machines. arXiv preprint arXiv:2609.21331, 2026.

[20] Xin Zhang, Weixuan Kou, Eric I-Chao Chang, He Gao, Yubo Fan, and Yan Xu. Sleep stage classification based on multi-level feature learning and recurrent neural networks via wearable device. Computers in Biology and Medicine, 103:71–81, 2018.

[21] Kailong Zhang, Youwei Lyu, Heng Guo, Si Li, Zhanyu Ma, and Boxin Shi. PolarAnything: Diffusion-based polarimetric image synthesis. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 26466–26476, 2025.

[22] Beibei Lin, Zifeng Yuan, and Tingting Chen. RGB-to-polarization estimation: A new task and benchmark study. In The Thirty-ninth Annual Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2025.

[23] Lawrence B. Wolff and Terrance E. Boult. Constraining object features using a polarization reflectance model. IEEE Transactions on Pattern Analysis and Machine Intelligence, 13(7):635–657, 1991.

[24] Fucai Ke, Vijay Kumar B G, Xingjian Leng, Zhixi Cai, Zaid Khan, Weiqing Wang, Pari Delir Haghighi, Hamid Rezatofighi, and Manmohan Chandraker. Dwim: Towards tool-aware visual reasoning via discrepancy-aware workflow generation & instruct-masking tuning. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 3378–3389, October 2025.

[25] Tingting Chen, Beibei Lin, Srinivas Anumasa, Vedant Shah, Zifeng Yuan, Qiran Zou, Anirudh Goyal, and Dianbo Liu. Auto-Discovery-Bench: Diagnosing structured state tracking in oracle-guided discovery. arXiv preprint arXiv:2502.15224, 2025.

[26] Yunhao Ba, Alex Gilbert, Franklin Wang, Jinfa Yang, Rui Chen, Yiqin Wang, Lei Yan, Boxin Shi, and Achuta Kadambi. Deep shape from polarization. In European Conference on Computer Vision, pages 554–571. Springer, 2020.

[27] Akshat Dave, Yongyi Zhao, and Ashok Veeraraghavan. PANDORA: Polarization-aided neural decomposition of radiance. arXiv preprint arXiv:2203.13458, 2022.

[28] Simeng Qiu, Qiang Fu, Congli Wang, and Wolfgang Heidrich. Linear polarization demosaicking for monochrome and colour polarization focal plane arrays. Computer Graphics Forum, 40, 2021.

[29] Youwei Lyu, Zhaopeng Cui, Si Li, Marc Pollefeys, and Boxin Shi. Reflection separation using a pair of unpolarized and polarized images. In H. Wallach, H. Larochelle, A. Beygelzimer, F. d'Alché-Buc, E. Fox, and R. Garnett, editors, Advances in Neural Information Processing Systems, volume 32. Curran Associates, Inc., 2019.

[30] Axin Fan, Tingfa Xu, Geer Teng, Wang Xi, Yuhan Zhang, Chang Xu, Xin Xu, and Jianan Li. Full-Stokes polarization multispectral images of various stereoscopic objects. Scientific Data, 10, 2023.

[31] Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang. Restormer: Efficient transformer for high-resolution image restoration. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5728–5739, 2022.

[32] Zhendong Wang, Xiaodong Cun, Jianmin Bao, Wengang Zhou, Jianzhuang Liu, and Houqiang Li. Uformer: A general U-shaped transformer for image restoration. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pages 17683–17693, 2022.

[33] Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, and Ross Girshick. Masked autoencoders are scalable vision learners. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pages 16000–16009, 2022.

[34] Ozan Özdenizci and Robert Legenstein. Restoring vision in adverse weather conditions with patchbased denoising diffusion models. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(8):10346–10357, 2023.

[35] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4195–4205, 2023.

[36] Luming Tang, Nataniel Ruiz, Qinghao Chu, Yuanzhen Li, Aleksander Holynski, David E Jacobs, Bharath Hariharan, Yael Pritch, Neal Wadhwa, Kfir Aberman, et al. RealFill: Reference-driven generation for authentic image completion. ACM Transactions on Graphics (TOG), 43(4):1–12, 2024.

[37] Gaurav Parmar, Taesung Park, Srinivasa Narasimhan, and Jun-Yan Zhu. One-step image translation with text-to-image models. arXiv preprint arXiv:2403.12036, 2024.

[38] Jérémy Riviere, Ilya Reshetouski, Luka Filipi, and Abhijeet Ghosh. Polarization imaging reflectometry in the wild. ACM Transactions on Graphics, 36(6):1–14, 2017.

[39] Achuta Kadambi, Vage Taamazyan, Boxin Shi, and Ramesh Raskar. Polarized 3D: High-quality depth sensing with polarization cues. In Proceedings ofthe IEEE International Conference on Computer Vision, pages 3370–3378, 2015.

[40] Weilong Yan, Xin Zhang, and Robby T. Tan. ER-LoRA: Effective-rank guided adaptation for weathergeneralized depth estimation. arXiv preprint arXiv:2509.00665, 2025.

[41] Zhiqiang Teng, Tingting Chen, Beibei Lin, Zifeng Yuan, Xuanyi Li, Xuanyu Zhang, and Shunli Zhang. RaindropGS: A benchmark for 3D Gaussian splatting under raindrop conditions. arXiv preprint arXiv:2510.17719, 2025.

[42] Seung-Hwan Baek, Daniel S. Jeon, Xin Tong, and Min H. Kim. Simultaneous acquisition of polarimetric SVBRDF and normals. ACM Transactions on Graphics, 37(6):268:1–268:15, 2018.

[43] Inseung Hwang, Daniel S Jeon, Adolfo Munoz, Diego Gutierrez, Xin Tong, and Min H Kim. Sparse ellipsometry: portable acquisition of polarimetric SVBRDF and shape with unstructured flash photography. ACM Transactions on Graphics, 41(4):1–14, 2022.

[44] Hyunho Ha, Inseung Hwang, Nestor Monzon, Jaemin Cho, Donggun Kim, Seung-Hwan Baek, Adolfo Muñoz, Diego Gutierrez, and Min H. Kim. Polarimetric BSSRDF acquisition of dynamic faces. ACM Transactions on Graphics, 43(6):1–11, 2024.

[45] Zhuoyan Liu, Bo Wang, Lizhi Wang, Chenyu Mao, and Ye Li. ShareCMP: Polarization-aware RGB-P semantic segmentation. IEEE Transactions on Circuits and Systems for Video Technology, pages 1–1, 2025.

[46] Takuya Kurita, Yuki Kondo, Liang Sun, et al. Simultaneous acquisition of high quality RGB image and polarization information using a sparse polarization sensor. In Proceedings of the IEEE/CVF Winter Conference on Applications ofComputer Vision, pages 178–188, 2023.

[47] Kailun Yang, Luis M. Bergasa, Eduardo Romera, Xinxin Huang, and Kaiwei Wang. Predicting polarization beyond semantics for wearable robotics. In Proc. IEEE-RAS Int. Conf. Humanoid Robots (Humanoids), pages 96–103. IEEE, 2018.

[48] Zifeng Yuan, Beibei Lin, Yutong Liu, Wujie Fu, Yuan Gao, Tingting Chen, and Aaron Danner. Driveefficient GHz-rate binary-state updates enabled by VCSEL polarization switching. Laser & Photonics Reviews, page e71890, 2026.

[49] Dewen Zhang, Zifeng Yuan, Thanh Xuan Hoang, Wujie Fu, Ching Eng Png, Soon Thor Lim, and Aaron Danner. All-optical scalable and programmable VCSEL-based Ising annealer with parallel feedback. Optics Express, 33(11):22119–22131, 2025.

[50] Beth Schaefer, Edward Collett, Robert Smyth, Daniel Barrett, and Beth Fraher. Measuring the Stokes polarization parameters. American Journal of Physics, 75(2):163–168, 2007.

[51] Zifeng Yuan, Dewen Zhang, Yuan Gao, Luo Qi, Wujie Fu, and Aaron Danner. Large-scale fabrication and analysis of polarization behavior in VCSELs with tailored apertures. Journal ofLightwave Technology, 43:6819–6827, 2025.

[52] C. J. R. Sheppard. Jones and Stokes parameters for polarization in three dimensions. Physical Review A, 90(2):023809, 2014.

[53] Zifeng Yuan, Dewen Zhang, Hong-Lin Lin, and Aaron Danner. Engineering polarization switching in VCSELs with custom aperture shapes. In CLEO: Science and Innovations, page JPS200\_47. Optica Publishing Group, 2025.

[54] W. H. McMaster. Polarization and the Stokes parameters. American Journal ofPhysics, 22(6):351–362, 1954.

[55] Zifeng Yuan, Dewen Zhang, Lei Shi, Yutong Liu, and Aaron Danner. Enhanced polarization locking in VCSELs. Applied Physics Letters, 126(15):151101, 2025.

[56] J. Scott Tyo, Dennis L. Goldstein, David B. Chenault, and Joseph A. Shaw. Review of passive imaging polarimetry for remote sensing applications. Applied Optics, 45(22):5453–5469, 2006.

[57] Shariq Farooq Bhat, Reiner Birkl, Diana Wofk, Peter Wonka, and Matthias Müller. ZoeDepth: Zero-shot transfer by combining relative and metric depth. arXiv preprint arXiv:2302.12288, 2023.

[58] Wei Yin, Chi Zhang, Hao Chen, Zhipeng Cai, Gang Yu, Kaixuan Wang, Xiaozhi Chen, and Chunhua Shen. Metric3D: Towards zero-shot metric 3D prediction from a single image. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pages 9043–9053, 2023.

[59] Tingting Chen, Beibei Lin, Zifeng Yuan, Qiran Zou, Hongyu He, Anirudh Goyal, Yew-Soon Ong, and Dianbo Liu. HypoSpace: A diagnostic benchmark for set-valued hypothesis generation under underdeter mination and sublinear coverage bounds. In Proceedings ofthe 43rd International Conference on Machine Learning, volume 306 of Proceedings ofMachine Learning Research, pages 15451–15465. PMLR, 2026.

[60] Edwin B. Wilson. Probable inference, the law of succession, and statistical inference. Journal of the American Statistical Association, 22(158):209–212, 1927.

[61] Xin Zhang and Ying-Cong Chen. Adaptive domain generalization via online disagreement minimization. IEEE Transactions on Image Processing, 32:4247–4258, 2023.

[62] Tingting Chen, Beibei Lin, Yeying Jin, Wending Yan, Wei Ye, Yuan Yuan, and Robby T Tan. Dual-rain: Video rain removal using assertive and gentle teachers. In European Conference on Computer Vision, pages 127–143. Springer, 2024.

[63] Beibei Lin, Yeying Jin, Wending Yan, Wei Ye, Yuan Yuan, Shunli Zhang, and Robby T Tan. Nightrain: Nighttime video deraining via adaptive-rain-removal and adaptive-correction. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 38, pages 3378–3385, 2024.

[64] Beibei Lin, Yeying Jin, Yan Wending, Wei Ye, Yuan Yuan, and Robby T Tan. Nighthaze: Nighttime image dehazing via self-prior learning. In Proceedings of the AAAI conference on artificial intelligence, volume 39, pages 5209–5217, 2025.

[65] Xin Yang, Xin Zhang, and Xinchao Wang. ERF: A benchmark dataset for robust semantic segmentation under extreme rainfall conditions. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 9301–9309, 2025.

[66] Fucai Ke, Zhixi Cai, Boying Li, Long Chen, Beibei Lin, Weiqing Wang, Pari Delir Haghighi, Gholamreza Haffari, and Hamid Rezatofighi. View2space: Studying multi-view visual reasoning from sparse observations. In European Conference on Computer Vision, pages 520–538. Springer, 2026.

[67] Wanjun Du, Zifeng Yuan, Tingting Chen, Fucai Ke, Beibei Lin, and Shunli Zhang. WeatherReasonSeg: A benchmark for weather-aware reasoning segmentation in visual language models. In European Conference on Computer Vision (ECCV), 2026. arXiv:2603.17680.

[68] Beibei Lin, Xiao Cao, Jingyuan Guo, and Robby T Tan. Glowgs: Generative semantic feature learning for 3d gaussian splatting in nighttime glow scenes. arXiv preprint arXiv:2605.23602, 2026.

[69] Beibei Lin, Tingting Chen, and Robby Tan. Geocomplete: Geometry-aware diffusion for reference-driven image completion. Advances in Neural Information Processing Systems, 38:35238–35259, 2026.

[70] Ziniu Qian, Zihua Wang, Xin Zhang, Bingzheng Wei, Maode Lai, Jianzhong Shou, Yubo Fan, and Yan Xu. MSNSegNet: Attention-based multi-shape nuclei instance segmentation in histopathology images. Medical & Biological Engineering & Computing, 62(6):1821–1836, 2024.

[71] Jingwei Zhang, Xin Zhang, Ke Ma, Rajarsi Gupta, Joel Saltz, Maria Vakalopoulou, and Dimitris Samaras. Gigapixel whole-slide images classification using locally supervised learning. In Medical Image Computing and Computer Assisted Intervention – MICCAI 2022, pages 192–201. Springer Nature Switzerland, 2022.

[72] Yutong Liu, Pragati Aashna, Zifeng Yuan, Wujie Fu, Mingxiang Yang, Yunjie Yan, Luo Qi, Mingshan Zhao, and Aaron Danner. A lithium niobate cascaded MMI-based arbitrary matrix calculator for convolutional neural networks. Optics & Laser Technology, 192:113740, 2025.

[73] Timnit Gebru, Jamie Morgenstern, Briana Vecchione, Jennifer Wortman Vaughan, Hanna Wallach, Hal Daumé III, and Kate Crawford. Datasheets for datasets. Communications ofthe ACM, 64(12):86–92, 2021.

## A Notation

Table 6: Notation used in this paper.  
Symbol Meaning   
$\mathrm { S } _ { 0 } , \mathrm { S } _ { 1 } , \mathrm { S } _ { 2 } , \mathrm { S } _ { 3 }$ Full (absolute) Stokes components, trichromatic $( H \times W \times 3$ each)   
σ Per-scene radiometric scale, σ = max S   
σˆ Predicted per-scene scale   
$\mathrm { s } _ { k } = \mathrm { S } _ { k } / \sigma$ Normalized Stokes components, $k \in \{ 0 , 1 , 2 , 3 \}$   
s<sub>0</sub> Network input: normalized total-intensity image   
$\hat { \mathbf { s } } _ { n }$ Predicted normalized Stokes components, $n \in \{ 1 , 2 , 3 \}$   
AoLP, DoLP, DoCP Angle / degree of linear and degree of circular polarization (Eqs. (2), (3))   
P Network output tensor $( H \times \bar { W } \times 1 0$ for type (a), $H \times W \times \mathrm { \hat { 1 9 } }$ for types (b), (c))   
MAX PSNR peak value, fixed to 1.0   
Constant σ Training-set median (4.85) or mean (4.25) of σ

## B Source dataset and benchmark details

## B.1 Source dataset

The Spectral-Polarization dataset [16] contains 2,022 trichromatic Stokes images $( 2 1 0 0 \times 1 9 2 0 \times 3 \times 4 )$ and 311 hyperspectral Stokes images $( 6 1 2 \times 5 1 2 \times 2 1 \times 4$ , 450–650 nm in 10 nm steps). The trichromatic images are captured in a single shot by a division-of-focal-plane RGB full-Stokes polarimeter with on-sensor quarter-wave plates and linear polarizers [17]; the hyperspectral images are captured by scanning a liquid-crystal tunable filter and rotating a quarter-wave plate. Every image is labeled with its environment (indoor/outdoor), illumination (clear or cloudy sunlight, white or incandescent light), capture time, and scene type (object- or scene-oriented). PolarScale uses only the trichromatic subset.

## B.2 Split and package contents

The 1,000/200 split follows the scene split of Lin et al. [22]; it is fixed by construction, and no additional sampling seed is involved. The benchmark package lists the scene IDs of both splits, the per-scene σ values, and all derived targets, so that every number in this paper can be reproduced from the released files.

## B.3 Scale distribution

The training-set median and mean of σ are 4.85 and 4.25; using them as constant predictors gives mean/median relative errors of 5.7%/4.5% and 16.2%/16.0% on the test split. Since the mean lies below the median, the distribution has a tail toward small scales.

## C Implementation details

Training budget All baselines were trained to convergence on one server with eight NVIDIA RTX A5000 GPUs (24 GB each): about one day for each restoration model and two days for each diffusion-based model. Per-model hyperparameters are provided in the configuration files of the code release.

Output heads per strategy Type (a), direct Stokes estimation. The output tensor is $\mathbf { P } \in \mathbb { R } ^ { H \times W \times 1 0 }$ concatenating $\left\{ { \mathrm { s } } _ { 1 } , { \mathrm { s } } _ { 2 } , { \mathrm { s } } _ { 3 } \right\} \in \ \tilde { \mathbb { R } } ^ { H \times W \times 9 }$ and the scale map $\sigma ~ \in ~ \mathbb { R } ^ { H \times W \times 1 }$ . For the restorationbased methods, WDiff, and DiT, we set the dimensionality of the final output layer to 10. For RealFill and I2ITurbo, we use task-specific text prompts ("S1", "S2", "S3", "Scale") with imagepolarization pairs to guide training and steer inference. Type (b), joint estimation. The output tensor is $\mathbf { P } \in \mathbb { R } ^ { H \times ^ { \mathbf { \lambda } } W \times 1 9 }$ , adding AoLP, DoLP, and DoCP; RealFill and I2ITurbo receive the additional prompts "AoLP", "DoLP", and $" { \tt D o C P } "$ . Type (c), decoupled multi-decoder estimation. For the restoration-based methods, WDiff, and DiT, we add a second decoder branch: the original one estimates scale-dependent parameters and the new one scale-independent parameters. For RealFill and I2ITurbo, we use two independent sets of LoRA parameters instead of an additional decoder, which avoids the parameter redundancy and overhead of extra frozen decoder layers.

Generative adaptation WDiff and DiT are not prompt-adapted: they use native multi-channel regression heads. RealFill and I2ITurbo use text prompts, because changing their output dimensionality would disturb the large-scale pre-trained weights that are their main strength. Their inference is multi-pass, one pass per component, and is matched at the task level: every model predicts the same components exactly once.

Native-conditioning variant To test whether prompt-based adaptation disadvantaged the pretrained generative baselines, we also trained them with native conditioning. The text encoder is frozen and receives an empty prompt, and all seven outputs are predicted jointly in a single forward pass through widened layers (RealFill: conv\_in 9 → 33 and conv\_out 4 → 28 channels; I2ITurbo: conv\_out 4 → 28 channels), initialized by replicating the pre-trained weights. The training recipe is otherwise unchanged (same LoRA ranks, optimizer, learning rate, batch size, and resolution). Table 7 shows that native conditioning is the weaker adaptation.

Table 7: Prompt-based versus native conditioning for the pre-trained generative baselines.
<table><tr><td></td><td>I2ITurbo (prompt)</td><td>I2ITurbo (native)</td><td>RealFill (prompt)</td><td>RealFill (native)</td></tr><tr><td>Rel. scale err. (mean/med) ↓</td><td>5.3% / 2.9%</td><td>8.8% / 7.4%</td><td>38.0% / 41.8%</td><td>65.9% / 69.7%</td></tr><tr><td>Normalized PSNR / SSIM ↑</td><td>38.46 / 0.974</td><td>35.35 / 0.961</td><td>38.14 / 0.963</td><td>29.66 / 0.901</td></tr><tr><td>#(ô ≤ 0) ↓</td><td>0/200</td><td>0/200</td><td>0/200</td><td>21/200</td></tr><tr><td>AoLP self-consistency ↓</td><td>52.4°</td><td>27.4°</td><td>44.6°</td><td>36.3°</td></tr></table>

## D Constant-scale control on full-Stokes reconstruction

Table 8 combines each model’s predicted normalized Stokes with either its learned σˆ or a constant scale (training median 4.85 or mean 4.25).

Table 8: Full-Stokes PSNR/SSIM with the learned scale and with constant scales. For the wellbehaved models, the constant scale matches or exceeds the learned scale in full-Stokes PSNR, so this metric does not measure scale recovery. WDiff and DiT expose the opposite failure: their near-zero σˆ inflates the PSNR obtained with the learned scale.
<table><tr><td>Method</td><td>Rel. scale err. (mean/med)↓</td><td>Learned ô</td><td>Const. σ = 4.85</td><td>Const. σ = 4.25</td></tr><tr><td>WDiff</td><td>92.8% / 93.1%</td><td>18.30 / 0.641</td><td>2.65 / 0.087</td><td>3.74 / 0.105</td></tr><tr><td>DiT</td><td>99.9% / 99.8%</td><td>20.01 / 0.746</td><td>16.44 / 0.607</td><td>17.06 / 0.633</td></tr><tr><td>RealFill</td><td>38.0% / 41.8%</td><td>16.47 / 0.611</td><td>18.25 / 0.684</td><td>18.78 / 0.707</td></tr><tr><td>I2ITurbo</td><td>5.3% / 2.9%</td><td>18.47 / 0.666</td><td>18.52 / 0.669</td><td>18.89 / 0.691</td></tr><tr><td>Restormer</td><td>5.1% / 2.4%</td><td>20.21 / 0.786</td><td>20.25 / 0.786</td><td>20.50 / 0.788</td></tr><tr><td>Uformer</td><td>4.3% / 2.5%</td><td>20.29 / 0.785</td><td>20.31 / 0.786</td><td>20.50 / 0.787</td></tr><tr><td>MAE</td><td>3.6% / 2.1%</td><td>21.08 / 0.814</td><td>21.13 / 0.815</td><td>21.27 / 0.816</td></tr></table>

## E Downstream protocols

Diffuse/specular separation Reference diffuse and specular components are computed from the measured full Stokes vectors of the 200 test scenes by the same polarization-based separation, in absolute radiance. Our setting composes both components analytically from $\hat { \mathrm { s } } _ { n }$ and a scale; the RGB-direct baseline regresses the six absolute channels directly. For the shuffled-scale control, the learned scales are permuted across test scenes.

Material segmentation All settings share the same network, training data, schedule, and initialization, and differ only in the polarization representation supplied.

Glare-level classification The 200 scenes are the benchmark’s held-out test split, and their glare levels are computed from the ground-truth polarization of those same scenes; no component of this probe is re-fit, so there is no variation across training seeds or split resampling. The intervals reported in Section 4.5 are Wilson 95% score intervals for a binomial proportion over the 200 scenes [60].

## F Datasheet

We document the PolarScale benchmark package following Datasheets for Datasets [73].

Motivation PolarScale was created to evaluate whether models can estimate radiometrically scaled Stokes vectors from intensity images, a question that existing benchmarks, which evaluate only normalized or relative polarization, cannot answer. It was created by the authors of this paper.

Composition Each of the 1,200 instances (1,000 train, 200 test) is a real-world scene from the trichromatic subset of [16]. For every scene, the package provides the input s<sub>0</sub>, the scale σ, the normalized Stokes components s<sub>1</sub>–s<sub>3</sub>, the descriptors AoLP, DoLP, and DoCP, and the scene labels inherited from [16]. The data contain no annotations of people. Outdoor scenes may incidentally show vehicles or bystanders, as in the source dataset.

Collection process No new data were captured. All instances are derived from [16] by the deterministic preprocessing of Section 3.3.

Preprocessing $\sigma = \mathrm { m a x } \mathrm { S _ { 0 } }$ per scene; $\mathrm { s } _ { k } = \mathrm { S } _ { k } / \sigma ;$ AoLP, DoLP, and DoCP by Eqs. (2) and (3).   
The preprocessing scripts are part of the package.

Uses Intended for benchmarking RGB-to-Stokes estimation with an explicit scale target. Not intended for calibrated radiometry, safety-critical decisions without validation on the target sensor, or the identification of people.

Distribution The package is distributed through the code repository given in Appendix H. The derived data remain subject to the license and terms of use of the source dataset [16].

Maintenance The package is maintained by the authors; questions and error reports can be sent to the corresponding author (zfyuan@nus.edu.sg).

## G Additional qualitative results

Figures 4–10 show reconstructed full Stokes components for further test scenes, and Figures 11–17 show predicted AoLP, DoLP, and DoCP maps. In all figures, the input (left) is the normalized total-intensity image s<sub>0</sub>, and results are shown for DiT [35], Uformer [32], I2ITurbo [37], and MAE [33].

## H Ethics statement and computational resources

Ethics statement The work does not involve human subjects, and the data contain no annotations of people or other sensitive personal data. Outdoor scenes may incidentally show vehicles or bystanders, as in the source dataset [16]. All data are used solely for research purposes.

Computational resources All experiments are performed on a server with eight NVIDIA RTX A5000 GPUs (24 GB memory each); training takes about one day per restoration model and two days per diffusion-based model. The code, pre-trained checkpoints, configuration files, and the benchmark package are publicly available at https://anonymous.4open.science/r/PolarScale.

![](images/99f2022cb741172664b25d346e53a21a08653c76cbb05da082c311c5d579b97b.jpg)  
RGB (s<sub>0</sub>)

![](images/aac119287f176a8a49b448d7981b411e6a1e291a32a4bb2d2f6b5e5aba8cc9a9.jpg)  
Figure 4: Reconstructed full Stokes components for test scene 1050, an urban plaza with high-rise buildings in which the sky region carries a strong negative $\mathrm { S _ { 2 } }$ signal. MAE recovers $\mathrm { S _ { 2 } }$ and $\mathrm { S _ { 3 } }$ in this region but predicts the wrong sign for $\mathrm { S _ { 1 } } .$ while the other models predict nearly flat maps.

![](images/fb8dbf59166755dc1e992b1679fd5b23afcc3fa5c634a32ba757ad7e91af3ab6.jpg)  
RGB (s<sub>0</sub>)

![](images/51acf48e069888eb1f8621d815f32b63d14a6ec71d5079fd76d0fedd9978cf4a.jpg)  
Figure 5: Reconstructed full Stokes components for test scene 1003.

![](images/bbcde65fa0d47c93f68f152bb1de68bbf52944ec2c274f401a2de4487450447a.jpg)  
RGB (s<sub>0</sub>)

![](images/b0db82c325d74b9755c0a121f816e4d9d5e317d7f86e00df181929d522699254.jpg)  
Figure 6: Reconstructed full Stokes components for test scene 1035.

![](images/3afbbb1f8ff6ec684532fdb8c7ddab28f2d6190d1e5fe479eb2c84d9a9c398dd.jpg)  
RGB (s<sub>0</sub>)

![](images/2b1b40341c88cb93b8cca8fc73055526f08fe45273c59b9770ac857632c94746.jpg)  
Figure 7: Reconstructed full Stokes components for test scene 1038.

![](images/016a301d29dabe224e7641131f5d88b64c6a0a54516eb1b7e77daec73cf67005.jpg)  
RGB (s<sub>0</sub>)

![](images/29bb358e7b449d118c2a640262d11ae4659d528560146d8701976d8b2a134337.jpg)  
Figure 8: Reconstructed full Stokes components for test scene 1062.

![](images/1e38b48115f6ff89162a647548eaa6a01514c487f5506f7d4130e9cd5765c267.jpg)  
RGB (s<sub>0</sub>)

![](images/7c5c081760d76792f790e6fc7072f400f10fbf567b67aa789da2213582a8f784.jpg)  
Figure 9: Reconstructed full Stokes components for test scene 1083.

![](images/67b5b54d5754a4a3bc4de11cf1eff3e2e8d87203c65fcd2366bcdce3942c1267.jpg)  
RGB (s<sub>0</sub>)

![](images/00923847aedb64bc13e4a7a2194b097bb3e84a654e78a46e81cd160f73c632b6.jpg)  
GT

![](images/5ed300d5a9a3f7085af8ecafb0c5d6ddb6d1a28b08c68ea097a160e6c64a10f8.jpg)

![](images/42706479085cf76d9f60e8823d82dbb131bde6696f536fb3b86890b618c8bbd5.jpg)  
I2ITurbo  
MAE  
Figure 10: Reconstructed full Stokes components for test scene 1126.

![](images/7a880bbd23f44a3035f0a346b82fc4d7c1c711c556bcb0f815e31133f3cbcd29.jpg)  
RGB (S<sub>0</sub>)

![](images/191255c7aba12128422fc6af7e0b57ec9a6671bfec54d988f4458943e4feea7c.jpg)  
GT

![](images/3ba98ca58dbddb5ee59c315d40a729a74ecd29c92a26d79bb4f50c818080c5f8.jpg)  
DiT

![](images/c89f402b51008c3a4b1c3e034a1df3d6c3282c38d11e27e5e3cfbee348d0d5be.jpg)  
Uformer

![](images/1257170487a54cc41d9c8958e7a4da9f6e1eb003c645d4ae9da7cf00c2bf406c.jpg)  
I2ITurbo

![](images/d928cb82a4d7571ecb0dd54b527e7f03becfa52b3493c2db77152e8ed21f797c.jpg)  
MAE  
Figure 11: Predicted AoLP, DoLP, and DoCP for test scene 1158, a street intersection with crosswalk stripes, vehicles, and building facades. The models capture the global layout but miss abrupt polarization transitions; DiT produces noisy AoLP and I2ITurbo saturates DoCP, while MAE is closest to the ground truth.

![](images/594db315561632fa87e68e0d943bc864624b06c86cd2e6cdbbdcc6ef03a351a5.jpg)  
RGB (s<sub>0</sub>)

![](images/db8cd02fa1b961a97bfa087f23cd3cf4991742c64a69a1dbe0be3000b98b504c.jpg)  
GT

![](images/b26e868b6f83c2607c2b4687b0a49505bac8c101b88286b3e07a286b5524f43d.jpg)  
DiT

![](images/a24ea8c32f77c29335963b4f1d4688a46308188512e4348631c3ceb1a3ef5600.jpg)  
Uformer

![](images/86135c29ff2b9d92e8845628f5d784c3bf524e74a0a5418fc7c7ad6592b1e5bf.jpg)  
I2ITurbo

![](images/4890537c36541c5f31d367ad0f903aad5663c2972f2c65cd663b29180970f97a.jpg)  
MAE  
Figure 12: Predicted AoLP, DoLP, and DoCP for test scene 1196, a high-rise facade whose repetitive window pattern produces fine, periodic polarization structure.

![](images/127df638c8d5a38c060ba32918a7018c7687f31bc2194cd8f3b8506899c1145f.jpg)  
RGB (S<sub>0</sub>)

![](images/69e27430cd5f1ab092fb7905cf7fc85811300fa816c87cbaa4ce00b25edcd3ad.jpg)  
Figure 13: Predicted AoLP, DoLP, and DoCP for test scene 1036.

![](images/547a58c4d39c2a77e944c7d5e15db9828705c6b9d63c78258a7e572e66f878c6.jpg)  
RGB (S<sub>0</sub>)

![](images/1e4e9d6d30437e39dadb8a3e6e260313dda0c55c7ed06c9702f5e1f669dce4a7.jpg)  
Figure 14: Predicted AoLP, DoLP, and DoCP for test scene 1040.

![](images/96b61eb7915eb3eb907039cad049eb961720bd044f3f35ec9332b9315ff6b449.jpg)  
RGB (S<sub>0</sub>)

![](images/2b3647fe22b2c18dc85db0658ccaa5c34d5e413f617c545d3fadc46ae1fb4e01.jpg)  
Figure 15: Predicted AoLP, DoLP, and DoCP for test scene 1047.

![](images/fa322d623c5d3fae22e36ce928fd676a1944cd1ebfd7591a4f5708da53b5f611.jpg)  
RGB (S<sub>0</sub>)

![](images/976c0ef2095a2f91298ba43c1285f1670fc8f1617b4eb2137e626ef94f2e33b2.jpg)  
Figure 16: Predicted AoLP, DoLP, and DoCP for test scene 1052.

![](images/8e3f7110b71107d187d05829eece97a2d7e136d334dfbf4cb033d8768d5feb7d.jpg)  
RGB (s<sub>0</sub>)

![](images/5ba12ec5c4c04150ffa3dc6dbcd8d18ebedf2fb122a20b6f3263dcd32d98876d.jpg)  
Figure 17: Predicted AoLP, DoLP, and DoCP for test scene 1194.