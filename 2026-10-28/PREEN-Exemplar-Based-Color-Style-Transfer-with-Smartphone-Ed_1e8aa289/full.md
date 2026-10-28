# PREEN: Exemplar-Based Color Style Transfer with Smartphone Editor Presets

Beomjin Ahn

Samsung Research

Seoul, Korea

beomjin.ahn@samsung.com

## Abstract

We present PREEN (Preset Regression via an Exemplarbased Editor Network), which casts color style transfer as regression onto the 29 parameters of a deterministic image editor. Given a content image and a style reference, PREEN predicts a preset, a vector of editor slider values that is interpretable and editor-consumable, not a raster image. The key obstacle is supervision, since an arbitrary real reference lacks a preset ground truth. We therefore supervise with a style-transfer teacher, whose target looks we invert into the editor’s parameter space as pseudo-presets, then pretrain a CLIP-conditioned regressor and refine it with LoRA under an image-level CIELAB loss. On the human-curated PST50 benchmark, PREEN applies the reference’s color while preserving structuralfidelity, leading the baselines on pixel- andperceptual-fidelity metrics. The preset is compact and predicted at sub-second on-device latency. The lowdimensional subspace also exposes a single inference-time strength knob that traces a continuous default-to-vivid trajectory in one forward pass, with every intermediate state a valid editor preset. A two-axis human study shows that raters prefer the vivid end for matching the reference but the default as a standalonephoto, a trade-offthe knob spans without retraining.

## 1. Introduction

Exemplar-based color style transfer, also studied as photorealistic style transfer (PST), is a routine operation in modern photo editing. A user picks a reference image Y whose look they admire, wanting their own shot X recolored to match it (Figure 1). Mainstream academic approaches are pixel-generative, from neural style transfer [13] to diffusion editing [23]. Whatever their quality, they emit rasters with no exposed editing parameters, and the diffusionbased ones add multi-second latency and stochastic sampling that resists deterministic re-editing, an awkward fit for on-device, user-in-the-loop consumer pipelines.

An alternative is photo preset transfer [10], predicting the parameters P of a deterministic image editor E from the pair (X, Y) such that E(X, P) approximates the look of Y. The output P, a vector of slider values, is interpretable and compact. Unlike a rendered image, it can be stored and re-edited, with no generative model at inference time.

Input  
Reference  
![](images/b2e30d83a51bf537064479e1f676ca90375e4c08a5d8a7f345fc442ef02c8d91.jpg)  
PREEN  
Figure 1. PREEN restyles an input toward a reference via a pre dicted 29-parameter editor preset (shown at the Vivid strength setting).

The central challenge is supervision. Parametric retouching benchmarks such as MIT-Adobe FiveK [2] and PPR10K [19] provide human-authored target edits, but operate in the auto-enhancement regime without a separate style reference. No large-scale dataset pairs a content image with a reference and a matching expert-graded target. The principal prior method [10] sidesteps this by sampling presets, thereby bounding the styles it learns; the only reference-conditioned set with expert-graded targets we know of [7] offers just 50 pairs (PST50) for evaluation.

![](images/9e1bf1cfb8c4050bf30142a84cc22a2c82ebf4a22f806c6ac1881300e79f294d.jpg)  
Figure 2. Overview of PREEN. Given a content image X and a style reference Y, PREEN predicts a 29-parameter editor preset $P _ { \mathrm { p r e d } }$ in a single forward pass. Applying $P _ { \mathrm { p r e d } }$ with our differentiable editor E yields $X _ { \mathrm { p r e d } } ^ { \prime } ;$ the strength knob α scales the preset’s chroma parameters before editing. PREEN’s output is a compact, human-editable preset rather than a rendered image. Flame icons mark the modules updated during training (CLIP only through low-rank adapters), and the snowflake marks the fixed editor.

We close this gap with teacher supervision. We propose PREEN (Preset Regression via an Exemplar-based Editor Network). At inference, it maps a content–reference pair (X, Y) to a 29-parameter preset that a smartphone editor applies to produce the result (Figure 2). A two-stage pipeline trains it. Phase 1 pretrains a CLIP-conditioned regressor on a large corpus of pseudo-labeled triples recovered from a style-transfer teacher. Phase 2 freezes all base weights and trains only low-rank adapters under an image-level CIELAB loss evaluated on the differentiable editor output, addressing the perceptual non-uniformity of Phase 1’s parameter-space loss.

Our contributions are: (i) a teacher-supervised recipe that frees reference-based preset regression from sampledpreset supervision, substituting a style-transfer teacher on real references; (ii) a strength-interpolation primitive, an inference-time scalar α on the chromaticity parameters that traces a continuous default-to-vivid trajectory without rerunning inference; and (iii) a two-axis human study surfacing a three-way fidelity / style-match / aesthetic divergence across the α trajectory.

## 2. Related Work

## 2.1. Photo Preset Transfer

Editing photos by predicting the parameters of interpretable white-box operations has a long history. Learning-based automatic retouching predicts sequences of editing operations [12], and the same idea recurs in compositing harmonization [15]. Both regress parameters toward a fixed goal, enhancement or in-image harmonization, rather than a separate style reference. Deep Preset [10] established the reference-conditioned setting we adopt, regressing an editor preset (69 develop controls) from a content–reference pair, but its supervision is synthetic by construction. Training references are images re-rendered with sampled presets, so a ground-truth preset always exists, but the styles it can learn are confined to that sampled set. The setting has remained underexplored since then; the closest later work, NCST [14], regresses a smaller global-only parameter set from a reference to perform video grading. We revisit it without sampled-preset supervision: a style-transfer teacher on real reference pairs supplies targets we invert into the editor’s parameter space as pseudo-presets (Section 3).

## 2.2. Look-Up-Table Stylization

A separate line maps colors through a learned look-up table (LUT), a discretized grid over the RGB cube. Imageadaptive 3D LUT [35] and NLUT [3] predict one such grid applied uniformly across the image, and AceTone [22] predicts one from a reference with a vision-language model. Spatially aware variants blend multiple LUTs per pixel [33], and SA-LUT [7] extends these to a spatially adaptive 4D LUT, which we adopt as our teacher. SA-LUT also contributes the human-curated PST50 benchmark used for our evaluation (Section 4.2). Either way, the output is a dense grid of $1 0 ^ { 4 } – 1 0 ^ { 5 }$ values (Table 1) rather than a few named sliders, leaving the edit hard for a user to inspect or refine.

## 2.3. Pixel-Generative Stylization

A large body of work renders stylization through pixel transformations, from classic global color-statistic transfer [29] to AdaIN [13], diffusion-based editing such as SDEdit [23], and recent neural color-style methods including Neural Preset [16], ModFlows [18], and INRetouch [5]. Concurrent work pushes further into generative modeling. SemiNFT [34] casts reference preset transfer as a Diffusion-Transformer task refined with reinforcement learning for aesthetic perception. These methods emit a raster image directly, without exposing user-editable parameters.

Neither the LUT nor the pixel-generative family yields an editable preset; we compare experimentally against both families and against Deep Preset, which targets the same task.

Table 1. PST50 (paired) metrics. The Output column gives each method’s edit representation, and the Dim column gives its dimension ality. Raster outputs have no fixed dimension because their size scales with the image resolution. Time is the per-image CPU latency (supplementary); for LUT and preset methods it covers generating the LUT or preset, not applying it. Bold marks the best value per metric column. Per-image Wilcoxon significance is reported in the supplementary.
<table><tr><td>Method</td><td>Output</td><td>Dim</td><td>PSNR↑</td><td>SSIM↑</td><td>HCorr↑</td><td>LPIPS↓</td><td> $\pmb { \Delta } \mathbf { E _ { a b } ^ { * } } \downarrow$ </td><td>Time (s)</td></tr><tr><td>No edit (identity)</td><td>一</td><td></td><td>22.67</td><td>0.906</td><td>0.348</td><td>0.127</td><td>12.32</td><td>一</td></tr><tr><td>Reinhard [29]</td><td>Raster</td><td></td><td>20.13</td><td>0.835</td><td>0.447</td><td>0.122</td><td>13.40</td><td>0.004</td></tr><tr><td>AdaIN [13]</td><td>Raster</td><td></td><td>19.06</td><td>0.704</td><td>0.504</td><td>0.314</td><td>14.38</td><td>0.12</td></tr><tr><td>Deep Preset  $( \hat { Y } ) [ 1 0 ]$ </td><td>Raster</td><td>一</td><td>23.29</td><td>0.893</td><td>0.411</td><td>0.107</td><td>10.53</td><td>0.19</td></tr><tr><td>ModFlows [18]</td><td>Raster</td><td>一</td><td>19.98</td><td>0.853</td><td>0.510</td><td>0.118</td><td>12.77</td><td>1.57</td></tr><tr><td>NLUT [3]</td><td>LUT</td><td> $\mathbb { R } ^ { 3 \times 3 3 ^ { 3 } }$ </td><td>23.43</td><td>0.885</td><td>0.545</td><td>0.129</td><td>9.81</td><td>N/A†</td></tr><tr><td>SA-LUT [7]</td><td>LUT</td><td> $\mathbb { R } ^ { 3 \times 2 \times 1 7 ^ { 3 } }$ </td><td>22.54</td><td>0.884</td><td>0.516</td><td>0.133</td><td>9.63</td><td>0.73</td></tr><tr><td>Deep Preset () [10]</td><td>Preset</td><td> $\mathbb { R } ^ { 6 9 }$ </td><td>22.71</td><td>0.895</td><td>0.385</td><td>0.117</td><td>12.13</td><td>0.10</td></tr><tr><td>PRÉEN (Ours)</td><td>Preset</td><td> $\mathbb { R } ^ { 2 9 }$ </td><td>24.63</td><td>0.914</td><td>0.473</td><td>0.094</td><td>9.07</td><td>0.36</td></tr><tr><td>PREEN Vivid  $( \mathbf { O u r s } , \alpha { = } 5 )$ </td><td>Preset</td><td> $\mathbb { R } ^ { 2 9 }$ </td><td>22.27</td><td>0.899</td><td>0.401</td><td>0.127</td><td>12.18</td><td>0.36</td></tr></table>

<sup>†</sup>NLUT requires a CUDA-only operator, so its metrics are measured on GPU, and CPU timing is not available.

## 3. Method

## 3.1. Problem Formulation

Let $E : \mathcal { X } \times \mathcal { P }  \mathcal { X }$ denote a deterministic image editor mapping a content image X and a parameter vector $P \in [ - 1 , \bar { 1 ] ^ { d } }$ to an edited image $E ( X , P )$ We target a $d = 2 9$ editor representative of consumer mobile photo applications: 8 global controls (temperature, tint, brightness, exposure, contrast, highlights, shadows, saturation) and 21 per-hue HSL channels (hue/saturation/luminance for each of 7 color zones: red, orange, yellow, green, cyan, blue, magenta). We split the 29 parameters into a tone set $\mathcal { T } _ { L }$ (12: contrast, brightness, exposure, highlights, shadows, per-hue luminance) and a chromaticity set $\mathcal { I } _ { \mathrm { a b } }$ (17: temperature, tint, saturation, per-hue hue/saturation). Equivalently, the 8 global controls split 5 tone / 3 chroma and the 21 per-hue channels split 7 / 14, a split reused by the regressor’s two output trunks (Section 3.3) and the α control (Section 3.5). We implement these controls, inspired by consumer smartphone photo editors, as a differentiable module (stage-bystage pipeline in the supplementary), so both the Phase 1 inversion (Eq. 1) and the Phase 2 image-level loss can backpropagate through E, the same module that produces every PREEN result we report. Given (X, Y), the goal is a regressor $f _ { \theta } ( X , Y ) = P _ { \mathrm { p r e d } }$ such that $X _ { \mathrm { p r e d } } ^ { \prime } = E ( X , P _ { \mathrm { p r e d } } )$ approximates the look of Y.

## 3.2. Teacher Supervision

We instantiate the teacher with SA-LUT [7], a spatially adaptive 4D LUT producing a pseudo-target $\tilde { X } ^ { \prime } = T ( X , Y )$ for any pair. A LUT teacher is structure-preserving and dominated by color/tone transforms that lie close to our editor’s subspace, unlike a pixel-generative teacher’s edits. Both phases use the published teacher unmodified; a spatially global variant appears only in the ablation (Table 3).

## 3.3. Phase 1: Pseudo-Label Recovery and Regression

For each $( X , { \tilde { X } } ^ { \prime } )$ , we project the teacher’s richer edit into the editor’s constrained 29-parameter subspace, recovering a pseudo-preset $\tilde { P }$ by gradient-based optimization through the differentiable editor [32]:

$$
{ \tilde { P } } = \arg \operatorname* { m i n } _ { P \in [ - 1 , 1 ] ^ { 2 9 } } { \mathcal { L } } _ { \mathrm { r e c } } \bigl ( E ( X , P ) , { \tilde { X } } ^ { \prime } \bigr ) ,\tag{1}
$$

where $\mathcal { L } _ { \mathrm { r e c } } = \mathrm { M S E } _ { \mathrm { R G B } } + \lambda _ { \mathrm { l a b } } \left( \mathrm { M S E } _ { L ^ { * } } + \mathrm { M S E } _ { a ^ { * } b ^ { * } } \right)$ , each term the mean-squared error between $E ( X , P )$ and ${ \tilde { X } } ^ { \prime }$ over the indicated channels, an RGB anchor plus CIELAB luminance and chromaticity, and the weight $\lambda _ { \mathrm { l a b } }$ is set on a heldout subset. We run AdamW for a fixed step budget per pair, yielding the $( X , Y , { \tilde { P } } )$ triples on which the regressor below is trained.

The regressor $f _ { \theta }$ is a frozen CLIP visual encoder [28] feeding a simple residual MLP. Both X and Y pass through the shared encoder, and we concatenate the per-image 768- d CLIP image embeddings (the projected encoder output) with five handcrafted color-statistic features of X and $Y \colon$ (i) the RGB log-ratio $\log ( \mu _ { Y } / \mu _ { X } )$ , (ii) the HSL channel-mean difference, (iii) low-level RGB mean/std, (iv) a soft RGB color histogram, and (v) per-channel RGB percentile differences. These low-frequency color statistics supplement CLIP’s semantic features. The MLP runs two parallel residual trunks over this shared vector, each [1024, 1024, 1024, 256]: a tone trunk ending in a linear head for the $\mathcal { T } _ { L }$ parameters and a chroma trunk for the $\mathcal { I } _ { \mathrm { a b } }$ parameters, each head applying Softsign so $P _ { \mathrm { p r e d } } \in [ - 1 , 1 ] ^ { 2 9 }$ Phase 1 minimizes $\mathcal { L } _ { \mathrm { p h } 1 } = H _ { 0 . 1 } \big ( P _ { \mathrm { p r e d } } - \tilde { P } \big ) + \lambda \| P _ { \mathrm { p r e d } } \|$ ∥<sup>2</sup><sub>2</sub> with the CLIP backbone frozen, where $H _ { 0 . 1 }$ is the Huber loss $( \beta = 0 . 1 )$ and the $\ell _ { 2 }$ term lightly regularizes the predicted preset.

![](images/c0cafb6acd0d860bca47c0aafd183d0bf7fd0e559fcb093300c8b68dbfe37dae.jpg)  
Figure 3. Head-to-head on PST50 (first five samples). Columns: content X, reference Y, SA-LUT [7], Deep Preset (P<sup>ˆ</sup>) [10], PREEN $( \alpha = 1 )$ , and the expert target.

## 3.4. Phase 2: Adaptation under Image-Level Loss

Phase 1 already yields a working regressor, but its parameter-space loss is not perceptually uniform. Because $\mathbf { a } \pm 0 . 1$ shift in brightness is far more visible than the same shift in a per-hue blue-hue parameter, Phase 2 refines the model with an image-level loss, matching the edited image so that capacity concentrates on the perceptually salient changes that the parameter-space loss underweights.

Phase 2 freezes all base weights of $f _ { \theta }$ and trains only rank-16 LoRA adapters [11] on every CLIP attention and FFN projection. Adapters of the same rank are also placed on every linear layer of the MLP. After training, the adapters are merged, leaving inference cost identical to that of Phase 1. The Phase 2 objective passes predicted presets through the differentiable editor and applies a CIELAB reconstruction loss between $X _ { \mathrm { p r e d } } ^ { \prime } = E ( X , P _ { \mathrm { p r e d } } )$ and ${ \tilde { X } } ^ { \prime }$ plus an $\ell _ { 2 }$ regularizer on $P _ { \mathrm { p r e d } } .$

$$
\mathcal { L } _ { \mathrm { p h 2 } } = w _ { \mathrm { p i x } } \mathcal { L } _ { \mathrm { p i x } } + w _ { \mathrm { a b } } \mathcal { L } _ { \mathrm { a b } } + w _ { L } \mathcal { L } _ { L } + \lambda \| P _ { \mathrm { p r e d } } \| _ { 2 } ^ { 2 } ,\tag{2}
$$

where $\mathcal { L } _ { \mathrm { a b } } , \mathcal { L } _ { L }$ , and the auxiliary anchor $\mathcal { L } _ { \mathrm { p i x } }$ are meansquared errors in CIELAB $a ^ { * } b ^ { * } , \ L ^ { * }$ , and RGB, with $( w _ { \mathrm { p i x } } , w _ { \mathrm { a b } } , w _ { L } ) = ( 0 . 2 , 5 . 0 , 1 . 0 )$ set on a held-out subset. The term $\mathcal { L } _ { L }$ backpropagates only into the tone trunk and $\mathcal { L } _ { \mathrm { a b } }$ only into the chroma trunk (the complementary slice is detached). The RGB anchor reaches both. The $\ell _ { 2 }$ regularizer reuses the Phase 1 weight λ and keeps parameters at

0 when they have little effect (e.g., HSL channels with no corresponding color present).

The Phase 2 target is the full teacher applied to an augmented reference $Y _ { \mathrm { a u g } }$ (a random crop/flip of Y; Section 4.1), $\tilde { X } ^ { \prime } = T ( X , Y _ { \mathrm { a u g } } )$ . Identity samples $( Y = X$ , at a 0.1 ratio) use X itself as the target.

## 3.5. Strength Interpolation via $\mathcal { I } _ { \mathrm { a b } }$ α-Scaling

Operating in an explicit low-dimensional parameter subspace admits a parameter-space controllability primitive. At editing time, a scalar α rescales the chromaticity slice $\mathcal { I } _ { \mathrm { a b } }$ of $P _ { \mathrm { p r e d } }$ (Section 3.1), leaving the tone set $\mathcal { T } _ { L }$ unchanged, to dial edit intensity. We define

$$
P _ { i } ^ { \alpha } = \left\{ \frac { \alpha P _ { i } } { 1 + ( \alpha - 1 ) | P _ { i } | } , i \in \mathbb { Z } _ { \mathrm { a b } } , \right.\tag{3}
$$

This form is simply the head’s Softsign $( P = z / ( 1 { + } | z | ) )$ re-applied after rescaling its pre-activation z 7→ αz, so α = 1 recovers $P _ { \mathrm { p r e d } }$ and it inherits Softsign’s sign- and [−1, 1]- preservation with smooth saturation as $\alpha  \infty$ . Setting $\alpha > 1$ thus intensifies the entire predicted color edit (saturation and hue shifts alike), producing a vivid, stronger variant that emphasizes color shift over fidelity to the teacher distribution.

Restricting the rescaling to chroma is deliberate, since rescaling luminance parameters as well co-saturates global tone, clips highlights, and crushes shadows (supplementary). Chroma-only scaling preserves the tonal anchor while the color identity intensifies, and one $f _ { \theta }$ serves both modes.

Table 2. Pipeline ablation of PREEN on PST50. The top block trains a single phase; each remaining row changes one factor of the full pipeline (bottom row).
<table><tr><td>Setting</td><td>PSNR SSIM</td><td>HCorr</td><td>LPIPS</td><td> $\Delta \mathbf { E } _ { \mathbf { a b } } ^ { * }$ </td></tr><tr><td>Phase 1 only</td><td>23.60 0.907</td><td>0.383</td><td>0.123</td><td>10.60</td></tr><tr><td>Phase 2 only</td><td>24.43 0.918</td><td>0.453</td><td>0.105</td><td>9.12</td></tr><tr><td>MLP-only LoRA</td><td>24.39 0.915</td><td>0.423</td><td>0.115</td><td>9.89</td></tr><tr><td>w/o gradient routing</td><td>24.25 0.898</td><td>0.473</td><td>0.097</td><td>9.44</td></tr><tr><td>w/o identity samples</td><td>22.36 0.833</td><td>0.391</td><td>0.118</td><td>10.87</td></tr><tr><td>RGB loss only</td><td>24.89 0.922</td><td>0.462</td><td>0.103</td><td>9.59</td></tr><tr><td>PREEN (Ours)</td><td>24.63 0.914</td><td>0.473</td><td>0.094</td><td>9.07</td></tr></table>

## 4. Experiments

## 4.1. Experimental Setup

Content images X are drawn from OpenImages [17], and style references Y from the top 20% of the AVA aesthetics database [26] by mean aesthetic score, so references are well-composed, attractively graded photos. We embed every image with CLIP ViT-L/14 [28] and pair each X with one of its top-10 nearest references by cosine similarity, so the reference is semantically compatible, not arbitrary. This yields a few hundred thousand (X, Y ) training pairs, supervised by the SA-LUT [7] teacher. The Phase 1 pseudopresets $\tilde { P }$ are recovered via Eq. 1.

All optimizers are AdamW [20]. Phase 1 inversion (Eq. 1) runs 1,000 steps at learning rate 0.2 per pair with the CIELAB weight $\lambda _ { \mathrm { l a b } } = 2 0$ , chosen on a held-out subset, and a small pre-activation magnitude decay $( 1 0 ^ { - 3 } )$ . Phase 1 regression uses learning rate $3 \times 1 0 ^ { - 5 }$ . Phase 2 trains rank-16 LoRA at learning rate $1 \times 1 0 ^ { - 4 }$ , batch size 64, bf1 $^ { 6 , }$ $\lambda = 1 0 ^ { - 2 }$ , on an NVIDIA L4 GPU. The augmentation $Y _ { \mathrm { a u g } }$ is a random square crop (80–100% of the short side) plus a horizontal flip.

## 4.2. Evaluation

We evaluate on PST50 [7], a human-curated benchmark of (content, reference, expert-edited target) triples. We adopt the four metrics reported with it: PSNR, SSIM, HCorr (per-channel HSV histogram similarity to the target), and LPIPS [36], all computed on outputs and ground truth resized to 256×256, SA-LUT’s training resolution. We additionally report $\Delta E _ { a b } ^ { * } ,$ , the mean CIELAB color difference to the expert target, used as an evaluation metric in the pairedtarget retouching literature [19, 35], where CIELAB-space training losses are also common [24, 25].

## 4.3. Main Results on PST50

Table 1 reports the five PST50 metrics (Section 4.2) averaged over the paired triples. NCST [14], another parametric prior, is not benchmarked, as it releases no public code or weights and was not evaluated on PST50.

Table 3. Teacher ablation: PREEN trained under the published SA-LUT teacher vs. a global variant (per-pixel context map collapsed to a per-image mean). Metrics are PREEN’s.
<table><tr><td>Teacher</td><td>PSNR SSIM HCorr LPIPS</td><td> $\pmb { \Delta } \mathbf { E } _ { \mathbf { a b } } ^ { * }$ </td></tr><tr><td>SA-LUT Global</td><td>24.74 0.917 0.465 0.094</td><td>9.23</td></tr><tr><td>SA-LUT (Our setting)</td><td>24.63 0.914 0.473 0.094</td><td>9.07</td></tr></table>

Table 4. Phase 1 backbone ablation. Params = backbone + MLP head. Each backbone uses its native input resolution.
<table><tr><td>Backbone</td><td>Params</td><td>PSNR</td><td>SSIM</td><td>HCorr</td><td>LPIPS</td><td> $\pmb { \Delta } \mathbf { E } _ { \mathbf { a b } } ^ { * }$ </td></tr><tr><td>No backbone</td><td>6M</td><td>23.03</td><td>0.883</td><td>0.359</td><td>0.141</td><td>11.36</td></tr><tr><td>DINOv3 [30]</td><td>317M</td><td>23.04</td><td>0.884</td><td>0.370</td><td>0.141</td><td>11.39</td></tr><tr><td>SigLIP 2 [31]</td><td>330M</td><td>23.25</td><td>0.889</td><td>0.388</td><td>0.128</td><td>11.58</td></tr><tr><td>InternViT [4]</td><td>318M</td><td>23.40</td><td>0.906</td><td>0.371</td><td>0.127</td><td>11.22</td></tr><tr><td>CLIP [28]</td><td>315M</td><td>23.60</td><td>0.907</td><td>0.383</td><td>0.123</td><td>10.60</td></tr></table>

PREEN improves on Deep Preset in both its forms, the generator’s rendered output $\hat { Y }$ (a raster) and the predicted preset $\hat { P }$ applied in Lightroom. In Table 1, it leads every baseline on PSNR, SSIM, and LPIPS, by +1.34 dB PSNR over Deep Preset (Y<sup>ˆ</sup> ), +1.20 dB over the next-best NLUT, and +1.96 dB over the no-edit (identity) row, while lifting HCorr from 0.348 (no-edit) to 0.473, though it trails most free-form Raster/LUT baselines there. On $\Delta E _ { a b } ^ { * } \ ( 9 . 0 7 )$ , it ties the LUT methods within per-image noise (supplementary). Section 5 discusses how the restrained PST50 targets shape these scores.

A global preset lacks the local adaptivity of spatially varying baselines such as SA-LUT. What Table 1 shows is that, on this benchmark, the 29-parameter representation is not itself the bottleneck. It stays within range of LUT grids over 1000× larger in dimension $( \mathbb { R } ^ { 3 \times 2 \times 1 7 ^ { 3 } }$ vs. R<sup>29</sup>; Table 1). The HCorr gap to the LUTs is prediction headroom, not a subspace limit (oracle HCorr 0.672, supplementary). Figure 3 shows a head-to-head on the first five PST50 samples, where PREEN’s edit lands close to the expert target.

PREEN also generalizes beyond PST50. Figure 4 gives qualitative results on DPST [21] pairs (no paired ground truth, so no metrics). It restyles content on par with Deep Preset while retaining α control. The full PST50 grid, two more DPST pairs, the predicted presets for example pairs, and an in-the-wild gallery are in the supplementary.

## 4.4. Ablation Studies

## 4.4.1. Component Ablation

Table 2 isolates the contribution of each training phase. Phase 1 alone, a frozen-CLIP MLP regressor trained on pseudo-labels, already reaches 23.60 dB PSNR, above both Deep Preset forms. Phase 2 LoRA adaptation adds the largest gains, most prominently in chromaticity (0.383 → 0.473 HCorr). By contrast, Phase 2 only, a single stage from random initialization with no Phase 1 that fully finetunes the MLP under the image-level loss and adapts CLIP via LoRA as in PREEN, does not surpass the full pipeline on the task’s target metrics. Although it edges SSIM, it trails on HCorr (0.453 vs. 0.473) and LPIPS and reaches only a comparable $\Delta E _ { a b } ^ { * }$ (9.12 vs. 9.07) before peaking early and degrading, despite its larger fully fine-tuned capacity. The benefit of Phase 1’s initialization is thus not recovered by single-stage optimization; replacing that stage’s fine-tuning with LoRA fares worse still $( \Delta E _ { a b } ^ { * } 9 . 5 7 ;$ supplementary), consistent with a low-rank update on an untrained base being underexpressive and with Phase 1’s pretrained base enabling the efficient LoRA adaptation.

![](images/c0e08dcc6dce2384c2fac4330fc1304e04c07a8807b730f04860b53e83975e5c.jpg)  
Figure 4. Qualitative comparison on three DPST [21] pairs with distinct content (no paired ground truth). Columns: X, Y, SA-LUT [7], Deep Preset (P<sup>ˆ</sup>) [10], PREEN (α=1), and PREEN Vivid (α=5).

Single-factor variants locate the gains: MLP-only LoRA (frozen CLIP) keeps fidelity but recovers under half the chromaticity gain (HCorr 0.423), so adaptation must reach the representation, not just the head; disabling the dualtrunk gradient routing (joint L/ab gradients) costs 0.38 dB PSNR and 0.016 SSIM at unchanged HCorr, confirming the routing aids tonal fidelity; and dropping identity samples collapses fidelity (SSIM 0.833, PSNR below the no-edit floor). Removing the CIELAB terms (plain RGB MSE) instead gains pixel fidelity but cedes every color-facing metric (Table 2). A single RGB term underweights chroma, settling into damped color edits, whereas CIELAB lets us weight chroma above luminance $( w _ { \mathrm { a b } } = 5 \ \mathrm { v s . } \ w _ { L } = 1 $ Eq. 2), committing to the color transfer the task demands.

## 4.4.2. Teacher Choice

SA-LUT varies its edit across the image, whereas our editor applies one global preset, so Phase 1 must project spatially varying edits onto global parameters. To test whether a teacher matched to the editor yields cleaner pseudo-labels, we also tried a global variant that collapses SA-LUT’s perpixel context map to a per-image mean. It brings no consistent improvement (Table 3). The variant edges PSNR by ∼0.1 dB and SSIM by 0.003 but lowers HCorr (0.473 → 0.465) and worsens $\Delta E _ { a b } ^ { * } ~ ( 9 . 0 7  9 . 2 3 )$ , the two metrics that score color most directly. The two teachers differ significantly on no metric (supplementary), so the published teacher can be used as is, without tailoring it to the editor.

Table 5. Incremental preset-group contribution on PST50. The predicted preset is masked at inference to retain only the listed parameter groups, with the rest set to their neutral value.
<table><tr><td>Active groups</td><td>#</td><td>PSNR</td><td>SSIM</td><td>HCorr</td><td>LPIPS</td><td> $\Delta \mathbf { E } _ { \mathbf { a b } } ^ { * }$ </td></tr><tr><td>No edit</td><td>0</td><td>22.67</td><td>0.906</td><td>0.348</td><td>0.127</td><td>12.32</td></tr><tr><td>+ G1 (WB)</td><td>2</td><td>21.99</td><td>0.898</td><td>0.403</td><td>0.102</td><td>11.71</td></tr><tr><td>+ G2 (global tone)</td><td>7</td><td>24.21</td><td>0.913</td><td>0.451</td><td>0.099</td><td>10.08</td></tr><tr><td>+ G3 (saturation)</td><td>8</td><td>24.40</td><td>0.919</td><td>0.443</td><td>0.096</td><td>9.60</td></tr><tr><td>+ G4 (per-hue HSL)</td><td>29</td><td>24.63</td><td>0.914</td><td>0.473</td><td>0.094</td><td>9.07</td></tr></table>

## 4.4.3. Backbone Choice

We probe the backbone by swapping the frozen CLIP ViT-L/14 encoder for three recent vision encoders at matched capacity and training budget (Table 4): the self-supervised DINOv3 [30], the contrastive SigLIP 2 [31], and the VLMoriented InternViT [4, 6] (checkpoints in the supplementary). Removing the backbone entirely, leaving only the handcrafted color statistics, degrades every metric. The converse, CLIP without the handcrafted features, scores 23.15 dB (not shown), between no-backbone (23.03) and full (23.60; Table 4), so the backbone and the color features contribute complementarily, neither subsuming the other. On PSNR, the backbones are close; on $\Delta E _ { a b } ^ { * } ,$ only CLIP (10.60) and InternViT beat the no-backbone floor (11.36), with CLIP best. We adopt CLIP, the standard, widely deployed choice with mature on-device support.

## 4.4.4. Preset-Group Contribution

To see which of the 29 parameters carry the edit, we mask the predicted preset during inference to retain only the specified groups, setting the rest to their neutral values (Table 5). The contribution is interpretable and largely monotonic. White balance alone (G1) sets the gross color cast (its PSNR sits below no-edit until tone joins), the global tone parameters (G2) add the largest single jump (+2.2 dB over G1), global saturation (G3) helps fidelity while dipping HCorr (0.451→0.443), and the 21 per-hue HSL channels (G4) close the remaining HCorr gap (0.443 → 0.473), the color-fidelity component the global parameters cannot reach.

![](images/6ed45f48ece180d966b69be2806c0c8edf6a4e68a16e1ad62c40357365b62810.jpg)  
Figure 5. Visual α sweep on the last three PST50 samples. One forward pass, with only the inference-time chroma scalar changing between operating points. Columns: X, Y, α=1, α=3, α=5, α=10.

## 4.5. Strength Interpolation (α Sweep)

Figure 5 shows color saturation intensifying smoothly from $\alpha = 1$ through α = 5 (Vivid) to α = 10 with no re-inference. PSNR peaks at the default and degrades monotonically as α grows while perceptual saturation keeps increasing, a tradeoff that motivates our two operating points. We fix Vivid at α=5 by visual inspection, not by tuning on the benchmark. The look is clearly intensified but not yet degenerate. The full per-α table and a luminance-rescale failure case are in the supplementary.

## 4.6. Off-the-Shelf Foundation Models

## 4.6.1. Direct Parameter Prediction

We prompt three frontier vision-language models (Gemini 3.1 Pro [9], Claude Opus 4.8 [1], and the openweight Qwen3-VL 235B [27]) to predict the 29 parameters from the content–reference pair, providing the full channel specification, valid ranges, and a forced structuredoutput schema, with one query per image (Table 6; verbatim prompt in the supplementary). Under a neutral prompt (ranges and sign conventions only), all three produce much larger edits and score worse on every metric of the supplementary comparison (PSNR −2.9 to −6.0 dB). The VLM rows of Table 6 therefore use the conservative guided prompt. They cluster at 21.4–22.5 dB PSNR, 2.1–3.2 dB below PREEN. Calibrating sliders from text alone, without the editor’s feedback, is hard.

## 4.6.2. Direct Pixel Synthesis

The complementary route skips parameters entirely. Nano Banana Pro (Gemini 3 Pro Image) [8], run once per pair on PST50 and prompted to transfer the reference’s color and mood while preserving structure, matches PREEN on HCorr but trails it in fidelity by ∼5 dB PSNR, 0.17 SSIM, and over 2× LPIPS (Table 6). Synthesized pixels drift from the content’s fine structure, and the output is a nondeterministic raster with no editable parameters.

Table 6. Off-the-shelf foundation models on PST50: frontier VLMs predicting the preset and a generative image model, with PREEN for reference.
<table><tr><td>Method</td><td>PSNR</td><td>SSIM HCorr</td><td>LPIPS</td><td> $\pmb { \Delta } \mathbf { E } _ { \mathbf { a b } } ^ { * }$ </td></tr><tr><td colspan="5">Parameter prediction (Output: Preset)</td></tr><tr><td>Gemini 3.1 Pro</td><td>21.40 0.788</td><td>0.343</td><td>0.157</td><td>12.57</td></tr><tr><td>Claude Opus 4.8</td><td>22.49 0.857</td><td>0.364</td><td>0.124</td><td>11.41</td></tr><tr><td>Qwen3-VL 235B 21.87</td><td>0.886</td><td>0.352</td><td>0.124</td><td>12.02</td></tr><tr><td colspan="5">Image-to-image (Output: Raster)</td></tr><tr><td>Nano Banana Pro</td><td>19.66 0.745</td><td>0.461</td><td>0.198</td><td>15.24</td></tr><tr><td>PREEN (Ours)</td><td>24.63 0.914</td><td>0.473</td><td>0.094</td><td>9.07</td></tr></table>

## 4.7. User Study

Automatic metrics capture fidelity, not perceived quality. We run pairwise preference studies: PREEN Vivid $( \alpha =$ 5) and the default each against (i) Deep Preset $( \hat { P } )$ and (ii) Claude Opus 4.8 predicting the same preset, plus Vivid vs. default (isolating α) and both vs. the SA-LUT teacher. Raters judge style match (closer to the reference) and aesthetic appeal (more pleasing as a standalone photo). The VLM opponent uses its guided prompt (Section 4.6). Figure 6 reports PREEN’s win rates, and the supplementary lists each with its 95% confidence interval.

On style match, Vivid wins against every compared preset opponent (Deep Preset 61.6%, Claude 65.7%, PREEN’s own default 73.4%; pooled 66.9%, [62.2, 71.6]) and roughly ties the teacher itself (44.6%, [36.2, 53.3]), to which the default loses (28.3%). On aesthetic appeal, the ordering inverts. Vivid drops to 43.3% pooled, [38.9, 47.9], with raters favoring the default and Claude as standalone images, yet both PREEN modes are preferred to the teacher’s fullstrength edit (68.8%/73.8%).

This three-way split is the study’s central finding. The metrics rank the default highest, human style-match ranks Vivid highest, and standalone aesthetic preference returns to the default. “Matches the reference” and “looks best on its own” are different objectives. The α knob spans them without retraining.

The default also holds its own. It beats Deep Preset on both axes (61.8% style, [53.0, 70.6]; 67.5% aesthetic, [60.6, 74.0]) and roughly matches Claude Opus 4.8 (52.0%/52.5%) while remaining deterministic and far cheaper.

## 4.8. On-Device Deployment

We export f (LoRA merged) to TFLite with 8-bit dynamicrange quantization (INT8 weights, FP32 activations) and benchmark on a Snapdragon 8 Elite Gen 5 smartphone (8 CPU threads, median of 20 warm runs). The preset is predicted end-to-end in 283 ms per pair, CPU-only. Latency is dominated by the CLIP backbone (304M of 315M parameters), yet within a sub-second UX budget. Quantization preserves accuracy. On PST50, the INT8 artifact scores 24.34/0.915/0.479/0.093/9.02, within 0.3 dB PSNR of Table 1 and matching it on the other four metrics.

## 5. Discussion and Limitations

More than the numbers, PREEN’s payoff is the form of its output, an interpretable, deterministic preset that an editor consumes directly. The explicit 29-dimensional subspace also makes controllability free: a single inferencetime scalar α rescales the preset’s chroma along a defaultto-vivid trajectory of valid, re-editable presets. Across that trajectory, the three-way divergence that our user study surfaces (Figure 6) makes the best operating point a user preference, not a fixed target.

Beyond controllability, the fidelity numbers invite a second reading. A global preset is structurally more constrained than the spatial LUT teacher, so PREEN’s higher PST50 fidelity scores (Table 1) do not indicate greater expressivity; they reflect PST50’s restrained expert targets. The teacher overshoots these conservative edits, while PREEN’s identity sampling (Table 2) and ℓ<sub>2</sub> preset regularizer keep it calibrated. Damping the teacher toward the input confirms this, improving it on four of the five PST50 metrics (full diagnostic in the supplementary).

PREEN also inherits the editor’s failure modes. Looks that remap color instead of adjusting it (split-toned monochromes, wholesale palette shifts) lie outside E’s space, as do local effects such as grain, vignettes, and local burns; for the former, even the per-image oracle fails (supplementary). It also assumes a semantically compatible reference: our pairs are matched by content, and we do not test transfer to disjoint references. Shrinking the CLIP-dominated footprint (Section 4.8) with a lighter encoder (e.g., MobileCLIP or FastViT) is left to future work.

Style match
<table><tr><td rowspan=1 colspan=1>PREEN  26.6%</td><td rowspan=1 colspan=3></td><td rowspan=1 colspan=1>PREEN Vivid</td></tr><tr><td rowspan=1 colspan=4>PREEN</td><td rowspan=1 colspan=1>61.8%</td></tr><tr><td rowspan=1 colspan=4>PREEN                  52.0</td><td rowspan=1 colspan=1>%                  Claude Opus</td></tr><tr><td rowspan=1 colspan=1>PREEN   28.3%</td><td rowspan=1 colspan=3></td><td rowspan=1 colspan=1>SA-LUT</td></tr><tr><td rowspan=1 colspan=3>Deep Preset</td><td rowspan=1 colspan=1>61.6%</td><td rowspan=1 colspan=1>PREEN Vivid</td></tr><tr><td rowspan=1 colspan=2>Claude Opus</td><td rowspan=1 colspan=2>65.7%</td><td rowspan=1 colspan=1>PREEN Vivid</td></tr><tr><td rowspan=1 colspan=4>SA-LUT</td><td rowspan=1 colspan=1></td></tr></table>

Aesthetic appeal  
![](images/e33b37d33167dad97321ec45733e758002c398e334a506a4f45b00297cfc5af8.jpg)  
Figure 6. Human pairwise preference on PST50. Each bar splits the judgments between the two labeled methods. The printed percentage is the win rate of the segment it sits in, and the dashed line marks 50%.

## 6. Conclusion

We presented PREEN, which recasts exemplar-based color style transfer as regression onto a deterministic editor’s 29 parameters. Lacking ground-truth presets, we invert a styletransfer teacher’s output into the editor’s parameter space, train a CLIP-conditioned regressor, and adapt it with LoRA under an image-level CIELAB loss. On PST50, the preset recovers the reference’s color shift while preserving structure; it is interpretable, ∼1000× more compact than the teacher’s LUT, and runs on-device in sub-second time. A single chromaticity scalar spans default and vivid looks, exposing strength as a free, re-editable control. A human study shows that the two ends of that range serve different goals, matching the reference and standing alone as a photo.

## References

[1] Anthropic. System Card: Claude Opus 4.8. https:

/ / www . anthropic . com / claude - opus - 4 - 8 - system-card, 2026. 7

[2] Vladimir Bychkovsky, Sylvain Paris, Eric Chan, and Fredo´ Durand. Learning photographic global tonal adjustment with a database of input / output image pairs. In CVPR, pages 97– 104, 2011. 1

[3] Yaosen Chen, Han Yang, Yuexin Yang, Yuegen Liu, Wei Wang, Xuming Wen, and Chaoping Xie. NLUT: Neuralbased 3D lookup tables for video photorealistic style transfer. arXiv preprint arXiv:2303.09170, 2023. 2, 3

[4] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, Bin Li, Ping Luo, Tong Lu, Yu Qiao, and Jifeng Dai. InternVL: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In CVPR, pages 24185–24198, 2024. 5, 6

[5] Omar Elezabi, Marcos V. Conde, Zongwei Wu, and Radu Timofte. INRetouch: Context aware implicit neural representation for photography retouching. In WACV, pages 5560–5569, 2026. 2

[6] Zhangwei Gao, Zhe Chen, Erfei Cui, Yiming Ren, Weiyun Wang, Jinguo Zhu, Hao Tian, Shenglong Ye, Junjun He, Xizhou Zhu, Lewei Lu, Tong Lu, Yu Qiao, Jifeng Dai, and Wenhai Wang. Mini-InternVL: A flexible-transfer pocket multi-modal model with 5% parameters and 90% performance. Visual Intelligence, 2(1):32, 2024. 6

[7] Zerui Gong, Zhonghua Wu, Qingyi Tao, Qinyue Li, and Chen Change Loy. SA-LUT: Spatial adaptive 4D look-up table for photorealistic style transfer. In ICCV, pages 18294– 18303, 2025. 2, 3, 4, 5, 6

[8] Google. Gemini 3 Pro Image Model Card. https: / / deepmind . google / models / model - cards / gemini-3-pro-image/, 2025. 7

[9] Google. Gemini 3.1 Pro Model Card. https : / / deepmind . google / models / model - cards / gemini-3-1-pro/, 2026. 7

[10] Man M. Ho and Jinjia Zhou. Deep Preset: Blending and retouching photos with color style transfer. In WACV, pages 2112–2120, 2021. 1, 2, 3, 4, 6

[11] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-rank adaptation of large language models. In ICLR, 2022. 4

[12] Yuanming Hu, Hao He, Chenxi Xu, Baoyuan Wang, and Stephen Lin. Exposure: A white-box photo post-processing framework. ACM TOG, 37(2):26:1–26:17, 2018. 2

[13] Xun Huang and Serge J. Belongie. Arbitrary style transfer in real-time with adaptive instance normalization. In ICCV, pages 1510–1519, 2017. 1, 2, 3

[14] Xintao Jiang, Yaosen Chen, Siqin Zhang, Wei Wang, and Xuming Wen. NCST: Neural-based color style transfer for video retouching. Signal Image Video Process., 19(6):459, 2025. 2, 5

[15] Zhanghan Ke, Chunyi Sun, Lei Zhu, Ke Xu, and Rynson W. H. Lau. Harmonizer: Learning to perform white-box image and video harmonization. In ECCV, pages 690–706, 2022. 2

[16] Zhanghan Ke, Yuhao Liu, Lei Zhu, Nanxuan Zhao, and Rynson W. H. Lau. Neural Preset for color style transfer. In CVPR, pages 14173–14182, 2023. 2

[17] Alina Kuznetsova, Hassan Rom, Neil Alldrin, Jasper R. R. Uijlings, Ivan Krasin, Jordi Pont-Tuset, Shahab Kamali, Stefan Popov, Matteo Malloci, Alexander Kolesnikov, Tom Duerig, and Vittorio Ferrari. The Open Images Dataset V4: Unified image classification, object detection, and visual relationship detection at scale. IJCV, 128(7):1956–1981, 2020. 5

[18] Maria A. Larchenko, Alexander Lobashev, Dmitry Guskov, and Vladimir Vladimirovich Palyulin. Color transfer with modulated flows. In AAAI, pages 4464–4472, 2025. 2, 3

[19] Jie Liang, Hui Zeng, Miaomiao Cui, Xuansong Xie, and Lei Zhang. PPR10K: A large-scale portrait photo retouching dataset with human-region mask and group-level consistency. In CVPR, pages 653–661, 2021. 1, 5

[20] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In ICLR, 2019. 5

[21] Fujun Luan, Sylvain Paris, Eli Shechtman, and Kavita Bala. Deep photo style transfer. In CVPR, pages 6997–7005, 2017. 5, 6

[22] Tianren Ma, Mingxiang Liao, Xijin Zhang, and Qixiang Ye. AceTone: Bridging words and colors for conditional image grading. In CVPR, pages 25851–25860, 2026. 2

[23] Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Ermon. SDEdit: Guided image synthesis and editing with stochastic differential equations. In ICLR, 2022. 1, 2

[24] Sean Moran, Pierre Marza, Steven McDonagh, Sarah Parisot, and Gregory G. Slabaugh. DeepLPF: Deep local parametric filters for image enhancement. In CVPR, pages 12823–12832, 2020. 5

[25] Sean Moran, Steven McDonagh, and Gregory G. Slabaugh. CURL: Neural curve layers for global image enhancement. In ICPR, pages 9796–9803, 2021. 5

[26] Naila Murray, Luca Marchesotti, and Florent Perronnin. AVA: A large-scale database for aesthetic visual analysis. In CVPR, pages 2408–2415, 2012. 5

[27] Qwen Team. Qwen3-VL technical report. arXiv preprint arXiv:2511.21631, 2025. 7

[28] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In ICML, pages 8748–8763, 2021. 3, 5

[29] Erik Reinhard, Michael Ashikhmin, Bruce Gooch, and Peter Shirley. Color transfer between images. IEEE Comput. Graph. Appl., 21(5):34–41, 2001. 2, 3

[30] Oriane Simeoni, Huy V. Vo, Maximilian Seitzer, Federico´ Baldassarre, Maxime Oquab, Cijo Jose, Vasil Khalidov, Marc Szafraniec, Seung Eun Yi, Michael Ramamonjisoa,¨ Francisco Massa, Daniel Haziza, Luca Wehrstedt, Jianyuan Wang, Timothee Darcet, Th ´ eo Moutakanni, Leonel Sentana,´ Claire Roberts, Andrea Vedaldi, Jamie Tolan, John Brandt,

Camille Couprie, Julien Mairal, Herve J´ egou, Patrick La-´ batut, and Piotr Bojanowski. DINOv3. Trans. Mach. Learn. Res., 2026. 5, 6

[31] Michael Tschannen, Alexey A. Gritsenko, Xiao Wang, Muhammad Ferjad Naeem, Ibrahim Alabdulmohsin, Nikhil Parthasarathy, Talfan Evans, Lucas Beyer, Ye Xia, Basil Mustafa, Olivier J. Henaff, Jeremiah Harmsen, Andreas´ Steiner, and Xiaohua Zhai. SigLIP 2: Multilingual vision-language encoders with improved semantic understanding, localization, and dense features. arXiv preprint arXiv:2502.14786, 2025. 5, 6

[32] Ethan Tseng, Felix Yu, Yuting Yang, Fahim Mannan, Karl St. Arnaud, Derek Nowrouzezahrai, Jean-Franc¸ois Lalonde, and Felix Heide. Hyperparameter optimization in black-box image processing using differentiable proxies. ACM TOG, 38 (4):27:1–27:14, 2019. 3

[33] Tao Wang, Yong Li, Jingyang Peng, Yipeng Ma, Xian Wang, Fenglong Song, and Youliang Yan. Real-time image enhancer via learnable spatial-aware 3D lookup tables. In ICCV, pages 2451–2460, 2021. 2

[34] Melany Yang, Yuhang Yu, Diwang Weng, Jinwei Chen, and Wei Dong. SemiNFT: Learning to transfer presets from imitation to appreciation via hybrid-sample reinforcement learning. arXiv preprint arXiv:2602.08582, 2026. 2

[35] Hui Zeng, Jianrui Cai, Lida Li, Zisheng Cao, and Lei Zhang. Learning image-adaptive 3D lookup tables for high performance photo enhancement in real-time. IEEE TPAMI, 44 (4):2058–2073, 2022. 2, 5

[36] Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In CVPR, pages 586– 595, 2018. 5

# PREEN: Exemplar-Based Color Style Transfer with Smartphone Editor Presets (Supplementary Material)

Beomjin Ahn

Samsung Research

Seoul, Korea

beomjin.ahn@samsung.com

## 1. Editor Pipeline Details

The editor applies the 29 parameters in a fixed order typical of consumer smartphone photo editors. Each parameter maps to a specific operator and stage (Table S1). Parameters act in $[ - 1 , 1 ]$ and are internally rescaled to the device ranges. Stages 1–4 operate in RGB, stages 5–8 on the BT.601 (Rec. 601) luma Y, and stage 9 in HSL with per-hue Gaussian masks. Every operation is differentiable almost everywhere, so gradients from the Phase 2 image-level loss flow back to $P _ { \mathrm { p r e d } }$

Table S1 gives the update rule of every stage. The input pixel is $X \ = \ ( X _ { r } , X _ { g } , X _ { b } ) \in [ 0 , 1 ] ^ { 3 }$ , and we write $s \equiv$ 100 p for the (rescaled) parameter of each stage.

## 2. Full α Sweep Analysis

Table S3 reports the full chroma-only α sweep on PST50, extending the two operating points used in the main paper. Toward $\alpha  0$ the chroma edit is under-applied and the color-facing metrics (HCorr, $\Delta E _ { a b } ^ { * } )$ weaken, and for large α (5, 10) over-saturation degrades every metric. Performance stays close to the $\alpha = 1$ values for small departures $( \alpha \in [ 0 . 5 , 2 ] )$ and falls off more clearly by $\alpha = 3$ (PSNR 24.63→23.53, $\Delta E _ { a b } ^ { * } 9 . 0 7 \substack {  } 1 0 . 2 9 )$ ). We adopt α = 1 as the default operating point.

Why chroma-only? Luminance rescale co-saturates tone. Figure S1 contrasts our chroma-only α-scaling with rescaling all 29 parameters (luminance included) on a PST50 sample with a pronounced warm edit. Chroma-only scaling intensifies the color identity while the tonal anchor stays fixed, so the image keeps a similar luminance scale even at $\alpha = 1 0$ . Rescaling the luminance parameters also co-saturates exposure, brightness, and contrast, driving the tones toward crushed shadows or blown highlights and flattening the photograph as α grows. This motivates restricting the α knob to the chromaticity slice $\mathcal { I } _ { \mathrm { a b } }$

Does α rotate hue? A natural worry is that scaling the chromaticity parameters reorients hue rather than merely intensifying it. We bound the per-channel hue rotation the α knob actually applies (an upper bound, as the per-hue Gaussian weight $w _ { h } \le 1 )$ , measured over the 7 per-hue channels of all 50 PST50 images. At the default $\alpha = 1$ no per-hue channel rotates by more than $3 0 ^ { \circ }$ (median $2 . 1 ^ { \circ } )$ , and even at the Vivid setting $( \alpha = 5 )$ only 0.3% do (median 8.9<sup>◦</sup>). Pushing to $\alpha = 1 0 ,$ , twice Vivid, still leaves the median at $1 4 . 9 ^ { \circ }$ with no channel beyond $6 0 ^ { \circ }$ and only 3.7% past $3 0 ^ { \circ }$ The editor’s bounded [−1, 1] per-hue range thus keeps hue identity largely intact as saturation grows.

## 3. Metric Protocol

Every method receives PST50’s Rec. 709 content, and outputs and the expert target are bilinearly resized to 256×256 (the SA-LUT [4] training resolution and kernel). PSNR is computed in sRGB. SSIM [14] uses an 11×11 Gaussian window $( \sigma { = } 1 . 5 )$ . HCorr is the mean per-channel correlation of 64-bin HSV histograms. LPIPS [15] uses pyiqa’s lpips-vgg+, the VGG variant whose spatial average is weighted by semantic features. $\Delta E _ { a b } ^ { * }$ is the mean per-pixel CIE76 difference $\| \mathrm { L A B } ( X ^ { \prime } ) - \mathrm { L } \mathrm { \bar { A } B } ( \mathrm { t a r g e t } ) \| _ { 2 }$ in CIELAB. Times in Table 1 of the main paper are perimage latencies on an Intel Xeon 6975P-C CPU, with every method producing the same 256×256 output. The final checkpoint is chosen on a held-out validation split. PST50 is used only for the reported evaluation, and none of its images appear in our training data. We compute a perceptual hash (DCT pHash) for every training image and every PST50 image, and, for each PST50 image, rank every training image by pHash Hamming distance and inspect its closest matches. All are distinct photographs sharing at most coarse composition or tone, with no duplicates.

Baseline rendering. The Deep Preset $( \hat { P } )$ [6] baseline predicts a Lightroom develop preset, which we apply to each content image and export in Adobe Lightroom Classic 15.2.1, one fixed build used for both the metrics in Table 1 of the main paper and the qualitative figures. Reinhard [11] is implemented as per-channel mean and standard-deviation transfer in CIELAB, and all other methods are rendered with their own released code. Neural Preset [7] is omitted because it releases neither model weights nor training code. It is also a neural color-mapping method rather than an editor-preset predictor.

Chroma-only α-scaling (Ours)  
Table S1. The 29-parameter editor in fixed application order, with the per-stage update rule $( s \equiv 1 0 0 p$ for the stage’s parameter; stage-1 multipliers $m _ { c }$ in Table S2; $h { = } 1 2 8 / 2 5 5 ;$ constants rounded and hard branches shown for readability). Stages 1–4 act in RGB, 5–8 in YCbCr luma Y, 9 in HSL.
<table><tr><td>Stage</td><td>Parameter</td><td>Idx</td><td>Update rule</td></tr><tr><td>1</td><td>Temperature</td><td>0</td><td> $X _ { c } \gets m _ { c } X _ { c } \quad ( m _ { c } :$  white-balance LUT, Table S2)</td></tr><tr><td>2</td><td>Contrast</td><td>1</td><td> $\begin{array} { r } { X  \frac { 1 } { 2 } + \gamma ( X - \frac { 1 } { 2 } ) , \quad \gamma = ( ( s + 4 0 0 ) / 4 0 0 ) ^ { 2 } } \end{array}$ </td></tr><tr><td>3</td><td>Tint</td><td>2</td><td> $X _ { r } + = \rho \phi \cdot \left( 0 . 0 0 3 2 s \right) X _ { r } , X _ { g } + = \left( 1 - \rho \right) \phi \cdot \left( 0 . 0 0 3 4 \left( - s \right) \right) X _ { g } , X _ { b } + = \rho \phi \cdot \left( 0 . 0 0 3 4 s \right) X _ { b } .$   $\ell = 0 . 2 1 2 6 X _ { r } + 0 . 7 1 5 2 X _ { g } + 0 . 0 7 2 2 X _ { b } , \phi = ( \ell / \sigma ) ^ { 2 } { \mathrm { i f ~ } } \ell < \tau { \mathrm { ~ e l s e ~ } } ( ( 1 - \ell ) / \sigma ) ^ { 2 }$ </td></tr><tr><td>4</td><td>Saturation</td><td>3</td><td> $\rho = \mathrm { s i g } ( 0 . 5 s ) , \tau { = } \frac { 1 2 8 } { 2 5 5 } , \sigma { = } \frac { 1 7 0 } { 2 5 5 }$   $X _ { c } \gets \left( W X _ { c } + \mathrm { G a i n } \left( g _ { r } X _ { r } + g _ { g } X _ { g } + g _ { b } X _ { b } \right) \right) / 1 0 2 4$ </td></tr><tr><td></td><td></td><td></td><td> $W = \mathrm { c l i p } ( 0 . 1 ( 1 0 3 ( s + 1 0 0 ) - 1 0 3 0 0 ) + 1 0 2 4 , 0 , 2 0 4 8 ) , { \mathrm { G a i n } } = 1 0 2 4 - W ,$   $( g _ { r } , g _ { g } , g _ { b } ) = ( 3 1 6 , 6 2 4 , 8 4 ) / 1 0 2 4$ </td></tr><tr><td>5</td><td>Brightness</td><td>4</td><td> $Y \gets Y + 0 . 8 \left( s / 1 2 7 \right) \cdot \left( Y \mathrm { i f } Y { < h \mathrm { e l s e } } 1 { - } Y \right)$ </td></tr><tr><td>6</td><td>Exposure</td><td>5</td><td> $Y \gets Y + s | s | / 2 5 5 0 0$ </td></tr><tr><td>7</td><td>Highlights</td><td>6</td><td> $Y \gets Y + 0 . 0 0 2 8 s Y ( 2 5 5 Y / 2 0 0 - \frac { 1 } { 2 } )$ </td></tr><tr><td>8</td><td>Shadows</td><td>7</td><td> $s { < } 0 { : } ~ Y + 0 . 0 0 3 ( \tau _ { m } { - } Y ) s ( \mathrm { g a t e d } \mathrm { t o } Y { < } \tau _ { m } )$ </td></tr><tr><td></td><td></td><td></td><td> $s { > } 0 \colon \mathrm { b l e n d ~ } Y + 0 . 2 8 ~ s / 2 5 5 ~ ( Y { \leq } h ) \mathrm { a n d } Y + 0 . 0 0 2 2 ( 1 { - } Y ) s ; ~ \tau _ { m } { = } \frac { 1 5 9 } { 2 5 5 }$ </td></tr><tr><td>9</td><td>Per-hue HSL 8-28</td><td></td><td> $\begin{array} { r } { \{ H , S , L \} + = \sum _ { j = 1 } ^ { 7 } w _ { j } ^ { ( \cdot ) } s _ { j } ^ { ( \cdot ) } g _ { ( \cdot ) } , w _ { j } ^ { ( \cdot ) } = e ^ { - d _ { j } ^ { 2 } / 2 \sigma _ { ( \cdot ) } ^ { 2 } } , d _ { j } = \mathrm { w r a p } ( H - H _ { j } ) \in [ - \frac { 1 } { 2 } , \frac { 1 } { 2 } ] } \end{array}$ </td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td> $H _ { j } = \{ 0 , 4 0 , 6 0 , 1 2 0 , 1 8 0 , 2 4 0 , 3 0 0 \} / 3 6 0$ </td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td> $\sigma _ { \mathrm { h , s , l } } = \{ 2 2 . 2 , 1 4 . 3 , 1 3 . 6 \} / 3 6 0 , g _ { \mathrm { h , s , l } } = \{ 0 . 4 5 / 3 6 0 , 0 . 5 8 / 1 0 0 , 0 . 2 1 / 1 0 0 \}$ </td></tr><tr><td></td><td></td><td></td><td> $s _ { j } ^ { ( \cdot ) } { = } 1 0 0 p _ { 8 + 3 ( j - 1 ) + \{ 0 , 1 , 2 \} } \colon$  each hue center j carries its own H/S/L parameter</td></tr></table>

![](images/ba428398d8e85d19d67d6c8cb4299d219075d148425e31466c364f5b0ca1124f.jpg)  
Figure S1. Chroma-only versus full-parameter α-scaling on one PST50 sample with a pronounced warm edit. Top: scaling only the chromaticity slice (ours) intensifies color while preserving tone. Bottom: scaling all parameters co-saturates the tonal channels, degrading the photograph as α grows. Columns: content X, reference Y, and α=1, 3, 5, 10.

## 4. Per-Image Statistical Tests

All methods are evaluated on the same 50 PST50 triples, so paired per-image tests apply. Table S4 reports PREEN’s mean per-image advantage on each metric (signpreserving difference, PREEN minus baseline) with twosided Wilcoxon signed-rank p-values. At the uncorrected

Table S2. Stage-1 white-balance calibration LUT: per-channel multipliers $M _ { c , j }$ at the 9 temperature knots (UI units, here mapped to $p _ { 0 } \in [ - 1 , 1 ] )$ .
<table><tr><td>knot</td><td>-100</td><td>-75</td><td>-50</td><td>-25</td><td>0</td><td>25</td><td>50</td><td>75</td><td>100</td></tr><tr><td> $M _ { R }$ </td><td>1.00</td><td>1.00</td><td>1.00</td><td>1.00</td><td>1.00</td><td>1.06</td><td>1.13</td><td>1.20</td><td>1.25</td></tr><tr><td> $M _ { G }$ </td><td>1.67</td><td>1.34</td><td>1.17</td><td>1.08</td><td>1.00</td><td>1.06</td><td>1.10</td><td>1.13</td><td>1.16</td></tr><tr><td> $M _ { B }$ </td><td>4.71</td><td>2.02</td><td>1.40</td><td>1.16</td><td>1.00</td><td>1.00</td><td>1.00</td><td>1.00</td><td>1.00</td></tr></table>

Table S3. Full α sweep on PST50. The first row is the identity (all 29 parameters neutral, the “No edit (identity)” row of Table 1 of the main paper), shown for reference; it is not the $\alpha { = } 0$ point of the main paper’s Eq. (3), which would zero only the chroma slice while retaining the predicted tone parameters.
<table><tr><td>α</td><td>PSNR↑</td><td>SSIM↑</td><td>HCorr↑</td><td>LPIPS↓</td><td> $\Delta \mathbf { E } _ { \mathbf { a b } } ^ { * } \downarrow$ </td></tr><tr><td>No edit</td><td>22.67</td><td>0.906</td><td>0.348</td><td>0.127</td><td>12.32</td></tr><tr><td>0.25</td><td>24.11</td><td>0.909</td><td>0.412</td><td>0.112</td><td>10.45</td></tr><tr><td>0.5</td><td>24.40</td><td>0.912</td><td>0.437</td><td>0.102</td><td>9.71</td></tr><tr><td>1.0 (default)</td><td>24.63</td><td>0.914</td><td>0.473</td><td>0.094</td><td>9.07</td></tr><tr><td>2.0</td><td>24.26</td><td>0.913</td><td>0.481</td><td>0.095</td><td>9.36</td></tr><tr><td>3.0</td><td>23.53</td><td>0.908</td><td>0.445</td><td>0.105</td><td>10.29</td></tr><tr><td>5.0 (Vivid)</td><td>22.27</td><td>0.899</td><td>0.401</td><td>0.127</td><td>12.18</td></tr><tr><td>7.0</td><td>21.41</td><td>0.889</td><td>0.372</td><td>0.147</td><td>13.72</td></tr><tr><td>10.0</td><td>20.49</td><td>0.876</td><td>0.332</td><td>0.169</td><td>15.51</td></tr></table>

0.05 level, PREEN’s PSNR and SSIM advantages are significant against every baseline, and LPIPS against all but Deep Preset $( { \hat { Y } } ) \ ( p { = } . 0 7 7 )$ On $\Delta E _ { a b } ^ { * } ,$ PREEN’s point-estimate lead over the two LUT methods (9.07 vs. 9.63/9.81) is within per-image noise $( p { \geq } . 4 2 )$ while all other $\Delta E _ { a b } ^ { * }$ advantages are significant. The HCorr deficits to the free-form LUT/raster methods (negative entries) are prediction headroom rather than a subspace limit, since the oracle reaches HCorr 0.672 (Table S5). In contrast, the HCorr gain over the no-edit baseline is significant $( p { < } . 0 0 1 )$ Under per-metric Holm correction within the family of between-method comparisons (the eight upper-block rows), every PSNR advantage over a published method remains significant (adjusted $p { \leq } . 0 3 1 )$ , while the marginal effects do not (SSIM vs. no edit/Deep Preset (P<sup>ˆ</sup>); LPIPS vs. Deep Preset (Y<sup>ˆ</sup> )/ModFlows; $\Delta E _ { a b } ^ { * }$ vs. Deep Preset (Y<sup>ˆ</sup> )/ModFlows). Finally, the last row is a separate within-method test. We detect no significant per-image difference between PREEN trained under the SA-LUT-global teacher variant and under the published teacher (all $p { \geq } . 5 6 )$ , and the small colormetric gaps that do appear lean slightly toward the published teacher (the direction shown in Table 3 of the main paper). The global variant thus brings no consistent gain.

For absolute scale, PREEN’s own per-metric means carry 95% bootstrap CIs (10k resamples, n=50): PSNR 24.63 [23.28, 25.92], SSIM 0.914 [0.894, 0.932], HCorr 0.473 [0.413, 0.533], LPIPS 0.094 [0.081, 0.107], and

$$
\Delta E _ { a b } ^ { * } 9 . 0 7 [ 7 . 8 3 , 1 0 . 4 1 ] .
$$

## 5. Training-Free Probes and Reference Points

We chart what training-free use of $( X , Y )$ and the teacher affords, under the same protocol as Table 1 of the main paper. (i) Fit-to-reference optimizes, per pair, the 29 editor parameters (AdamW, 1,200 steps) so the edited content matches the reference’s color statistics, using either per-channel CIELAB mean and standard deviation, in the spirit of classic color transfer [11], or the full sorted CIELAB marginals, then applies the result at full resolution. (ii) Damped teacher output is the raster blend $X _ { \gamma } ^ { \prime } =$ $( 1 - \gamma ) X + \gamma T ( X , Y )$ with the SA-LUT teacher, sweeping $\gamma \in \ [ 0 . 1 , 1 . 0 ]$ and reporting the best $( \gamma { = } 0 . 5 )$ . (iii) Testtime-optimization (TTO) inversion applies Eq. (1) of the main paper at test time to the teacher output (or to the damped teacher output), i.e., a preset-output method that needs the teacher plus a per-image optimization at inference (about one minute per image on the same CPU at 256×256, 1,000 AdamW steps, vs. PREEN’s 0.36 s without the teacher). (iv) Oracle is the same inversion against the expert target, upper-bounding the editor’s subspace.

Table S5 reports the results, which carry four lessons.

First, reference-fitting reaches HCorr 0.455/0.471 under the two statistic-matching objectives (on par with PREEN’s 0.473) yet collapses every fidelity metric below the no-edit baseline. The reference is a different scene, so its raw statistics are not the editing target, and over-applying them distorts the photograph (Figure S2, where PREEN’s restrained edits on the same samples stay close to the target). This is the characteristic failure of a distribution-only proxy and shows why HCorr must be read alongside fidelity.

Second, the damped teacher output shows how PST50’s restrained targets reward restraint. $\mathrm { A t } \gamma { = } 0 . 5$ , the blend surpasses most methods in Table 1 of the main paper. Calibrating other baselines’ edit strength does not close the gap: the same γ sweep lifts none of the other four learned raster/LUT baselines past PREEN’s 24.63 dB (best blends: NLUT 24.05, ModFlows 24.16, Deep Preset (Y<sup>ˆ</sup> ) 23.52, AdaIN 23.51 dB).

Third, projecting teachers into the editor changes their PSNR only slightly (SA-LUT 22.54→22.67 dB, damped 25.09→25.05 dB), consistent again with the subspace not being the bottleneck, though each projection costs a perimage optimization on top of the teacher. Notably, PREEN lands closer to the expert targets than the projection of its own undamped supervision (22.67→24.63 dB, $\Delta E _ { a b } ^ { * }$ $9 . 7 0 {  } 9 . 0 7 )$ The direction of this change (fidelity up, HCorr 0.505→0.473) matches the training objective’s builtin restraint, since the $\ell _ { 2 }$ preset regularizer and identity sampling damp predicted edits. PREEN’s presets are also roughly 4× smaller in norm than the teacher’s projection (median $\| P \|$ ratio 0.24) and only weakly aligned with it in direction (median cosine 0.28).

Table S4. Per-image paired comparison, PREEN (α = 1) minus baseline: mean difference with two-sided Wilcoxon p in parentheses Positive ∆PSNR/∆SSIM/∆HCorr and negative $\Delta \mathrm { L P I P S } / \Delta \Delta E _ { a b } ^ { * }$ favor PREEN. The last row instead compares the two teacher variants (PREEN trained under each); it is global minus published, so a positive $\Delta E _ { a b } ^ { * }$ favors the published teacher.
<table><tr><td>VS.</td><td>∆PSNR</td><td>△SSIM</td><td>∆HCorr</td><td>∆LPIPS</td><td> $\Delta \Delta E _ { a b } ^ { * }$ </td></tr><tr><td>No edit</td><td>+1.96 (&lt;.001)</td><td>+.009 (.041)</td><td> $+ . 1 2 5 \ : ( < . 0 0 1 )$ </td><td>−.034 (&lt;.001)</td><td>−3.25 (&lt;.001)</td></tr><tr><td>Reinhard</td><td>+4.50 (&lt;.001) +.079 (&lt;.001)</td><td></td><td>+.026 (.58)</td><td>-.028 (.002)</td><td>-4.34 (.002)</td></tr><tr><td>AdaIN</td><td>+5.57 (&lt;.001) +.210 (&lt;.001)</td><td></td><td>−.031 (.19)</td><td>−.221 (&lt;.001)</td><td>-5.32 (&lt;.001)</td></tr><tr><td>Deep Preset  $( \hat { Y } )$ </td><td>+1.34 (.011)</td><td>+.022 (.007)</td><td>+.062 (.12)</td><td>−.014 (.077)</td><td>-1.46 (.014)</td></tr><tr><td>ModFlows</td><td>+4.64 (&lt;.001) +.062 (&lt;.001)</td><td></td><td>−.037 (.22)</td><td>−.025 (.041)</td><td>-3.70 (.021)</td></tr><tr><td>NLUT</td><td>+1.20 (.015)</td><td>+.030 (.002)</td><td>-.072 (.028)</td><td>−.035 (&lt;.001)</td><td>-0.74 (.42)</td></tr><tr><td>SA-LUT</td><td>+2.09 (.010)</td><td>+.030 (.015)</td><td>−.044 (.17)</td><td>-.040 (.002)</td><td>-0.56 (.45)</td></tr><tr><td>Deep Preset  $( \hat { P } )$ </td><td>+1.92 (&lt;.001)</td><td>+.020 (.026)</td><td>+.088 (.001)</td><td>-.023 (.003)</td><td>−3.06 (&lt;.001)</td></tr><tr><td>SA-LUT-global vs. published teacher</td><td>+0.11 (.61)</td><td>+.003 (.95)</td><td>-.008 (.81)</td><td>.000 (.87)</td><td> $+ 0 . 1 7 \ _ { ( . 5 6 ) }$ </td></tr></table>

Fourth, the oracle reaches 32.56 dB and HCorr 0.672, far above every method on PSNR, HCorr, and $\Delta E _ { a b } ^ { * }$ (and best on SSIM, if narrowly), though its LPIPS (0.137) is worse than even the no-edit baseline (0.127). The gap to PREEN on those four metrics is prediction error from observing only (X, Y), which we leave to future work. (The oracle minimizes a CIELAB-weighted reconstruction objective, trading perceptual texture for color/tone fidelity, so it upper-bounds the editor’s color/tone subspace, not its perceptual quality.)

Table S5. Training-free probes and reference points on PST50, evaluated against the expert target. † = requires the SA-LUT teacher at inference; ‡ = per-image test-time optimization. The bottom row is the oracle fit to the target (upper bound on the editor’s subspace).
<table><tr><td>Method</td><td>PSNR</td><td>SSIM</td><td>HCorr</td><td>LPIPS</td><td> $\Delta \mathbf { E } _ { \mathbf { a b } } ^ { * }$ </td></tr><tr><td>No edit</td><td>22.67</td><td>0.906</td><td>0.348</td><td>0.127</td><td>12.32</td></tr><tr><td>Fit-to-ref (mean+std)‡</td><td>19.84</td><td>0.812</td><td>0.455</td><td>0.191</td><td>13.86</td></tr><tr><td>Fit-to-ref (marginals)‡</td><td>19.59</td><td>0.793</td><td>0.471</td><td>0.214</td><td>14.13</td></tr><tr><td>Damped teacher output (γ=0.5, raster)†</td><td>25.09</td><td>0.922</td><td>0.501</td><td>0.088</td><td>8.71</td></tr><tr><td>TTO inversion of teacher†</td><td>22.67</td><td>0.892</td><td>0.505</td><td>0.120</td><td>9.70</td></tr><tr><td>TTO inversion of damped teacher output†</td><td>25.05</td><td>0.922</td><td>0.471</td><td>0.091</td><td>8.93</td></tr><tr><td>PREEN (α = 1)</td><td>24.63</td><td>0.914</td><td>0.473</td><td>0.094</td><td>9.07</td></tr><tr><td>Oracle (29-param fit to GT)</td><td>32.56</td><td>0.937</td><td>0.672</td><td>0.137</td><td>4.90</td></tr></table>

## 6. Predicted Preset Breakdown

To make the output’s interpretability concrete, Figure S3 shows the actual 29-parameter presets PREEN predicts for the first five PST50 pairs, namely the 8 global controls and the 21 per-hue HSL channels, as real editor-slider values in [−1, 1]. The Vivid column (α = 5) rescales only the 17 chromaticity parameters, leaving the 12 tone parameters (the five global tone controls, i.e., contrast, brightness, exposure, highlights, and shadows, plus the seven per-hue luminance channels) unchanged. This is visible as identical tone bars, with the chroma bars (where nonzero) amplified between the two columns.

## 7. Qualitative Gallery

Failure cases. Figure S4 shows the failure family behind most of PREEN’s worst PST50 cases, namely, target looks that the editor cannot render. We verify this with the perimage oracle fit, the Eq. (1) inversion applied to the ground truth, shown in the third column of the figure. On these samples even the oracle misses the look (oracle $\Delta E _ { a b } ^ { * } 8 . 7 -$ 11.0 vs. an across-benchmark oracle median of 4.1, mean 4.90), because the target remaps color (for example, a splittoned near-monochrome or a wholesale palette shift) rather than adjusting existing color. No global preset can express such remapping, regardless of regressor quality. Four of PREEN’s five largest per-image $\Delta E _ { a b } ^ { * }$ errors fall in this editor-limited family. The last row shows the complementary case, where the oracle matches the target closely (oracle $\Delta E _ { a b } ^ { * } ~ 4 . 6 )$ , whereas PREEN misses it (22.2): plain prediction error, i.e., the headroom to the oracle that Section 4.3 of the main paper identifies.

Generalization beyond PST50 (DPST). Figure S5 shows five DPST [8] pairs with distinct content (the main paper showed the first three); there is no paired ground truth, so the comparison is qualitative. PREEN transfers the reference’s color character to unseen content and retains the same α control.

Extended in-the-wild gallery. Beyond PST50, we apply PREEN to in-the-wild photographs spanning landscape, indoor, and low-light scenes, each paired with a selected reference image. The content images, including those in Figure 1 of the main paper, are the author’s own photographs, and the reference images are free-to-use stock photographs from Pexels (https://www.pexels.com/). Figures S6–S7 show these results. PREEN transfers the global color and tone character of the reference across diverse scenes. As on PST50, looks that hinge on local or otherwise non-global effects are only approximated, since the editor is restricted to global parameters.

![](images/d87d84d1c4a9459d694c0b953d7bdd2e1c1a756c2f0e986081eb7890ae4be6af.jpg)  
Figure S2. Training-free fit-to-reference (marginals) over-applies the reference look and distorts the photograph. The three lowest-PSNR fit-to-reference results on PST50. Columns: input X, reference Y, fit-to-reference, PREEN (α = 1), and the expert target.

## 8. Backbone Ablation Checkpoints

The exact pretrained checkpoints used in the main paper’s backbone ablation are CLIP ViT-L/14 [10] (openai/clip-vit-large-patch14), DI-NOv3 ViT-L [12] (facebook/dinov3-vitl16- pretrain-lvd1689m), SigLIP 2 large [13] (google/siglip2-large-patch16-256), and InternViT-300M [2, 3] (OpenGVLab/ InternViT-300M-448px). Each backbone runs at its native input resolution, with the MLP head, training schedule, and loss held identical, so rows differ only in the frozen encoder and its native resolution.

## 9. Additional Implementation Details

The five handcrafted color-statistic features have fixed dimensionality: RGB log-ratio (3), HSL channel-mean difference (3), per-image RGB mean/std for both images (12), a soft RGB histogram with 16 bins per channel for both images (96), and per-channel $p _ { 1 0 } / p _ { 5 0 } / p _ { 9 0 }$ percentile differences (9), for 123 dimensions alongside the two 768-d CLIP image embeddings. Selected checkpoints occur at 1.1k training steps (Phase 1) and 4.6k (Phase 2). The Phase 1 inversion optimizes the pre-activation preset with AdamW (learning rate 0.2, weight decay 0.01) for 1,000 steps per pair. AdamW’s default decoupled weight decay (0.01) additionally shrinks the pre-activation, separate from the explicit $1 0 ^ { - 3 }$ penalty stated in the main paper, both nudging the recovered preset toward neutral. For the single-stage ablation in the main paper, replacing the Phase 2-only variant’s full

MLP fine-tuning with a LoRA adapter on a randomly initialized (untrained) MLP base atop the same LoRA-adapted CLIP encoder reaches only $\Delta E _ { a b } ^ { * } 9 . 5 7$ on PST50, versus 9.07 for the full pipeline and 9.12 for Phase 2-only full fine-tuning, consistent with a low-rank update being underexpressive without the Phase 1 pretrained base.

Table S6. Training hyperparameters by phase. Throughout, the backbone is CLIP ViT-L/14 with frozen base weights and the regressor is two parallel residual MLP trunks [1024, 1024, 1024, 256] (tone and chroma); training uses the few hundred thousand (X, Y) pairs described in the main paper.
<table><tr><td>Phase</td><td>Settings</td></tr><tr><td>1 (inversion)</td><td>AdamW, lr 0.2, 1,000 steps per pair; CIELAB weight  $\lambda _ { \mathrm { l a b } } { = } 2 0 ;$  pre-activation decay  $1 0 ^ { - 3 } ;$  AdamW weight decay 0.01.</td></tr><tr><td>1 (regression) AdamW, lr</td><td> $3 \times 1 0 ^ { - 5 } ;$  loss = Huber(β=0.1) on  $( P _ { \mathrm { p r e d } } , \tilde { P } ) + \ell _ { 2 }$  preset regularizer  $( \lambda { = } 1 0 ^ { - 2 } ) ;$  checkpoint at 1.1k steps.</td></tr><tr><td>2 (LoRA)</td><td>AdamW, lr  $1 0 ^ { - 4 }$  , effective batch 64 (16×4 gra- dient accumulation), bf1 6; LoRA rank 16, al- pha 32; image-level loss  $w _ { \mathrm { p i x } } { = } 0 . 2 ,$   $w _ { \mathrm { a b } } { = } 5 . 0 ,$  wL=1.0; l2 preset regularizer  $\lambda { = } 1 0 ^ { - 2 }$  ; identity ratio 0.1;  $Y _ { \mathrm { a u g } } { = }$  random 80–100% crop + hori- zontal flip; checkpoint at 4.6k steps, i.e., ≈294k</td></tr></table>

## 10. VLM Prompt Template

The main paper reports off-the-shelf foundation-model baselines that predict the 29-parameter preset directly. Each frontier VLM (Gemini 3.1 Pro [5], Claude Opus 4.8 [1], and Qwen3-VL 235B [9]) receives the identical prompt (Figure S8(b)), with the content and reference images attached (SOURCE and TARGET in the prompt). Each returns a JSON object of 29 named floats in [−1, 1], enforced by a structured-output response schema, and we apply the returned preset through the same editor E used by PREEN. Table S7 compares the two prompt variants of Figure S8. Under the neutral prompt (a), all three models produce substantially larger edits (1.7–3.7× the guided prompt’s mean per-pixel change $\left| X ^ { \prime } { - } X \right| )$ and every metric degrades, while the conservative guidance of (b) recovers the levels reported in the main paper. The guided prompt supplies only generic slider semantics and a magnitude prior (real edits are usually small), with no PST50- or target-specific values, and its two worked examples are a generic neutral-towarm-vintage preset and a near-zero no-op. Even granted this prior, the VLMs miscalibrate, consistent with editorgrounded calibration being hard to obtain from text prompts alone. VLM outputs are non-deterministic, and we query once per image.

![](images/f18858fb39bea8ff7fb444dbd8d9d3abc597e13f2ed5ccc4f2aa6cae061bc75e.jpg)

![](images/1090fe952a36d30824b29a68814e2730ca1bca9efb5d3cb6c85f585c3f508a6b.jpg)

![](images/f9744cb22283a0f7dad3b80021dcd54b15b1f25749c5baefe32dfc2cd60711a5.jpg)  
Figure S3. Predicted presets on PST50 (first five pairs). Per row: inputs (X, Y ), rendered outputs, and the predicted 29-parameter preset for PREEN (α = 1) and PREEN Vivid (α = 5) as real slider values (top eight: global controls; shaded block: the 21 per-hue HSL channels). Bar shade tracks magnitude, and the three largest-magnitude parameters (by PREEN α = 1 magnitude) are bold; Vivid scales chromaticity only, leaving tone bars unchanged.

![](images/dbbd996db54d64516c2148d284565730c5ec627e0d1f8238de8cd2db2c458742.jpg)  
Figure S4. PREEN’s five worst per-image $\Delta E _ { a b } ^ { * }$ cases on PST50: target looks outside the editor’s render space (rows 1–4) and a reachable target the regressor misses (last row). Columns: input X, reference Y , oracle fit to the target, PREEN, and the expert target.

## 11. User-Study Protocol

The pairwise preference studies of the main paper (Section 4.7) were conducted on Prolific (https://www.

Table S7. Prompt variants on PST50: neutral prompt (a) versus guided (b, reported).
<table><tr><td>Model</td><td>Prompt</td><td>PSNR↑</td><td>SSIM↑</td><td>HCorr↑</td><td>LPIPS↓</td></tr><tr><td>Gemini 3.1 Pro</td><td>neutral</td><td>18.27</td><td>0.675</td><td>0.211</td><td>0.234</td></tr><tr><td rowspan="3">Claude Opus 4.8</td><td>guided</td><td>21.40</td><td>0.788</td><td>0.343</td><td>0.157</td></tr><tr><td>neutral</td><td>19.57</td><td>0.725</td><td>0.261</td><td>0.199</td></tr><tr><td>guided</td><td>22.49</td><td>0.857</td><td>0.364</td><td>0.124</td></tr><tr><td rowspan="2">Qwen3-VL 235B</td><td>neutral</td><td>15.91</td><td>0.713</td><td>0.163</td><td>0.293</td></tr><tr><td>guided</td><td>21.87</td><td>0.886</td><td>0.352</td><td>0.124</td></tr></table>

prolific.com/). Each study recruited 50 raters (150 in total), assigned in balanced groups targeting five independent ratings per image pair. Invalid responses were excluded before analysis. The seven pairings of Figure 6 of the main paper were collected in three studies. Study 1 compared the default against Deep Preset and against Claude. Study 2 compared Vivid against the default, Deep Preset, and Claude. Study 3 compared the default and Vivid against the SA-LUT teacher. Raters saw the reference Y alongside the two candidate edits. The left–right order was randomized per pair, image filenames were anonymized with content hashes, and methods were never named. The two questions, verbatim: “Q1. Which better matches the Style?” (“Color tone, contrast, brightness, overall feel.”) and “Q2. Which looks more beautiful?” (“Independent of Style — which one looks better as a standalone photo?”). Participation was restricted to desktop users. Within a pair’s ratings, majority agreement averages 74–81% across the three studies and both questions (Fleiss’ κ 0.15–0.34), the slight-tofair agreement range.

![](images/0048ac63a5984639bdaccab6e5f12743693274814480c28e30b4f40032de0646.jpg)  
Figure S5. Qualitative results on five DPST [8] pairs (distinct content; no paired ground truth). Columns: X, Y , Deep Preset (P<sup>ˆ</sup>) [6], PREEN (α=1), and PREEN Vivid (α=5).

Table S8 lists every per-opponent win rate with its confidence interval. The pooled rates of the main paper also count PREEN’s own default as an opponent, a withinsystem α comparison rather than a baseline.

Table S8. Every pairing of Figure 6 of the main paper: win rate of the first-named PREEN mode, its 95% pair-clustered bootstrap interval (10k pair resamples), and n valid judgments per question.
<table><tr><td>Study</td><td>Pairing</td><td>Style match (%)</td><td>Aesthetic (%)</td><td>n</td></tr><tr><td>1</td><td>Default vs. Deep Preset ()</td><td>61.8 [53.0, 70.6]</td><td>67.5 [60.6, 74.0]</td><td>246</td></tr><tr><td rowspan="3">2</td><td>Default vs. Claude Opus 4.8</td><td>52.0 [43.2, 60.8]</td><td>52.5 [44.7, 60.6]</td><td>244</td></tr><tr><td>Vivid vs. Deep Preset ()</td><td>61.6 [53.3, 69.7]</td><td>47.2 [38.4, 56.0]</td><td>250</td></tr><tr><td>Vivid vs. Claude Opus 4.8</td><td>65.7 [56.3, 74.4]</td><td>40.3 [33.2, 47.6]</td><td>248</td></tr><tr><td rowspan="3">3</td><td>Vivid vs. Default</td><td>73.4 [66.4, 79.9]</td><td>42.5 [35.6, 49.4]</td><td>252</td></tr><tr><td>Default vs. SA-LUT</td><td>28.3 [22.9, 34.0]</td><td>68.8 [60.8, 76.3]</td><td>240</td></tr><tr><td>Vivid vs. SA-LUT</td><td>44.6 [36.2, 53.3]</td><td>73.8 [66.1, 81.0]</td><td>240</td></tr></table>

## 12. Full PST50 Results

Figures S9–S12 show every PST50 case: the 50 paired triples behind every reported number and 50 unpaired cases (each content image with a further benchmark reference and no expert target) for qualitative coverage only. Each group shows X, Y, PREEN (α = 1), PREEN Vivid (α = 5), and, for paired cases, the expert target.

![](images/4f35ea2769b6643abbb88ec860080919652e85652f3b58cf0dc8dd99ee65278f.jpg)

![](images/6ed430a7a9a2d0a7c226bc3f0e1b55f375cdb8a2976d50b05b896e92489f901f.jpg)  
Figure S6. Extended in-the-wild gallery (1/2): PREEN applied to photographs with selected references. Each group: content X, reference Y, PREEN (α=1), and PREEN Vivid (α=5).

![](images/dabb38815779aabcb4534d753db57a00cd16dc7cf1bf48bdf98a3bc019478ec4.jpg)

![](images/161d0c37948d0a2f08d8ace66b27d39b4763635a4f1877176b35dd5cb826a051.jpg)  
Figure S7. Extended in-the-wild gallery (2/2): X, Y, PREEN (α=1), and PREEN Vivid (α=5).

## (a) Neutral prompt (unguided) (a) Neutral

You are a precise photo color-grading assistant.   
Predict 29 global photo-editing parameters to   
transform SOURCE (first image) to match TARGET   
(second image)’s style.   
Each parameter is a float in [-1.0, 1.0], where   
0.0 means no change and the sign sets direction   
(below). Pick the magnitude that best matches   
the target’s look.   
GLOBAL parameters (8):   
- temperature: -1 cool/blue -> +1 warm/orange   
- contrast: -1 flat -> +1 high   
- tint: -1 green -> +1 magenta   
- saturation: -1 desatur. -> +1 vivid   
- brightness: -1 darker -> +1 brighter   
- exposure: -1 under -> +1 over   
- highlights: -1 recover -> +1 brighten   
- shadows: -1 crush -> +1 lift   
(b) Guided prompt (   
You are a precise photo color-grading assistant.   
Predict 29 global photo-editing parameters to   
transform SOURCE (first image) to match TARGET   
(second image)’s style.   
SCALE GUIDANCE (CRITICAL):   
Parameters range [-1.0, 1.0], but REAL edits   
rarely exceed +/-0.3.   
- +/-0.05 ˜ +/-0.15: subtle, fine-tuning   
- +/-0.15 ˜ +/-0.30: moderate, visible (COMMON)   
- +/-0.30 ˜ +/-0.50: strong, dramatic (rare)   
- +/-0.50 ˜ +/-1.0: extreme (very rare)   
When in doubt, USE SMALLER VALUES. Most per-hue   
parameters should be near 0.0. Over-editing   
destroys the image. Prefer subtlety.   
GLOBAL parameters (8):   
- temperature: -1 cool/blue -> +1 warm/orange   
- contrast: -1 flat -> +1 high   
- tint: -1 green -> +1 magenta   
- saturation: -1 desatur. -> +1 vivid   
- brightness: -1 darker -> +1 brighter   
- exposure: -1 under -> +1 over   
- highlights: -1 recover -> +1 brighten   
- shadows: -1 crush -> +1 lift

PER-HUE parameters (21 = 7 colors x hue/sat/lum):   
red/orange/yellow/green/cyan/blue/magenta, each:   
- <sub>\*</sub>\_hue: shift the hue of that color range   
- <sub>\*</sub>\_sat: saturation change for that color   
- <sub>\*</sub>\_lum: luminance change for that color   
Now analyze the two images and predict the   
parameters that transform SOURCE into TARGET.

## used for the reported results)

Figure S8. Prompt templates for the VLM baselines. Both the content and reference images (SOURCE and TARGET) are attached, and the model returns a 29-field JSON object enforced by a response schema; the worked example abbreviates its 21 per-hue fields, as in the prompt sent, and ±/→ are written as +/-/->. The reported results use the guided variant (b); see Table S7.

Y  
X  
PREEN PREEN Vivid Target  
X  
PREEN PREEN Vivid Target  
![](images/e35e1bcb15f7747fe70c8752df4e128204bc38e91977aa1f2872fbbab4d5b05f.jpg)

![](images/114c895f1148b0b50017a3b82e4948ada7907263c0588424193f3be656f3f4da.jpg)  
Figure S9. Full PST50 paired test set (1/2). Each group: content X, reference Y , PREEN (α = 1), PREEN Vivid (α = 5), and the expert target.

Y PREEN PREEN Vivid Target  
X  
X  
Y  
PREEN PREEN Vivid Target  
![](images/2724636280c24220dced784104c8844d12e8cc222bd2a4d7a98221f80a8d3afb.jpg)

![](images/1c7d296333629e861ccb7e6c8bc2fb8b14edd980979aa374fb4f1a0004be514b.jpg)  
Figure S10. Full PST50 paired test set (2/2). Each group: X, Y , PREEN (α = 1), PREEN Vivid (α = 5), and the expert target.

PREEN Vivid  
![](images/7b065e3ead1e6823803fb76ace8a5e282181360c8f818de7853ff3e08b5fbd9b.jpg)

PREEN  
PREEN Vivid  
![](images/a1b88732ed830e7d0bd3fc78cf5bceb5fa3aaae540ef57cd6069d75999a67fbc.jpg)  
Figure S11. Full PST50 unpaired test set (1/2). Each group: X, Y , PREEN (α = 1), and PREEN Vivid (α = 5).

![](images/538dea900073d1aa74456ebc0d911ec44d87ef6beadc91d75d194be64b09f018.jpg)

PREEN  
PREEN Vivid  
![](images/5d88b0df3ef81320ff219afec82e8e73ec86c299fa86d0e70a654ce36a7511a9.jpg)  
Figure S12. Full PST50 unpaired test set (2/2). Each group: X, Y , PREEN (α = 1), and PREEN Vivid (α = 5).

## References

[1] Anthropic. System Card: Claude Opus 4.8. https: / / www . anthropic . com / claude - opus - 4 - 8 - system-card, 2026. 5

[2] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, Bin Li, Ping Luo, Tong Lu, Yu Qiao, and Jifeng Dai. InternVL: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In CVPR, pages 24185–24198, 2024. 5

[3] Zhangwei Gao, Zhe Chen, Erfei Cui, Yiming Ren, Weiyun Wang, Jinguo Zhu, Hao Tian, Shenglong Ye, Junjun He, Xizhou Zhu, Lewei Lu, Tong Lu, Yu Qiao, Jifeng Dai, and Wenhai Wang. Mini-InternVL: A flexible-transfer pocket multi-modal model with 5% parameters and 90% performance. Visual Intelligence, 2(1):32, 2024. 5

[4] Zerui Gong, Zhonghua Wu, Qingyi Tao, Qinyue Li, and Chen Change Loy. SA-LUT: Spatial adaptive 4D look-up table for photorealistic style transfer. In ICCV, pages 18294– 18303, 2025. 1

[5] Google. Gemini 3.1 Pro Model Card. https : / / deepmind . google / models / model - cards / gemini-3-1-pro/, 2026. 5

[6] Man M. Ho and Jinjia Zhou. Deep Preset: Blending and retouching photos with color style transfer. In WACV, pages 2112–2120, 2021. 1, 8

[7] Zhanghan Ke, Yuhao Liu, Lei Zhu, Nanxuan Zhao, and Rynson W. H. Lau. Neural Preset for color style transfer. In CVPR, pages 14173–14182, 2023. 2

[8] Fujun Luan, Sylvain Paris, Eli Shechtman, and Kavita Bala. Deep photo style transfer. In CVPR, pages 6997–7005, 2017. 4, 8

[9] Qwen Team. Qwen3-VL technical report. arXiv preprint arXiv:2511.21631, 2025. 5

[10] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In ICML, pages 8748–8763, 2021. 5

[11] Erik Reinhard, Michael Ashikhmin, Bruce Gooch, and Peter Shirley. Color transfer between images. IEEE Comput. Graph. Appl., 21(5):34–41, 2001. 1, 3

[12] Oriane Simeoni, Huy V. Vo, Maximilian Seitzer, Federico´ Baldassarre, Maxime Oquab, Cijo Jose, Vasil Khalidov, Marc Szafraniec, Seung Eun Yi, Michael Ramamonjisoa,¨ Francisco Massa, Daniel Haziza, Luca Wehrstedt, Jianyuan Wang, Timothee Darcet, Th´ eo Moutakanni, Leonel Sentana,´ Claire Roberts, Andrea Vedaldi, Jamie Tolan, John Brandt, Camille Couprie, Julien Mairal, Herve J ´ egou, Patrick La-´ batut, and Piotr Bojanowski. DINOv3. Trans. Mach. Learn. Res., 2026. 5

[13] Michael Tschannen, Alexey A. Gritsenko, Xiao Wang, Muhammad Ferjad Naeem, Ibrahim Alabdulmohsin, Nikhil Parthasarathy, Talfan Evans, Lucas Beyer, Ye Xia, Basil Mustafa, Olivier J. Henaff, Jeremiah Harmsen, Andreas´ Steiner, and Xiaohua Zhai. SigLIP 2: Multilingual

vision-language encoders with improved semantic understanding, localization, and dense features. arXiv preprint arXiv:2502.14786, 2025. 5

[14] Zhou Wang, Alan C. Bovik, Hamid R. Sheikh, and Eero P. Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE TIP, 13(4):600–612, 2004. 1

[15] Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In CVPR, pages 586– 595, 2018. 1