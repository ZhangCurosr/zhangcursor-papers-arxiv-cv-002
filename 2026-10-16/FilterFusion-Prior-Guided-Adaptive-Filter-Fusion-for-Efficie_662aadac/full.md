# FilterFusion: Prior-Guided Adaptive Filter Fusion for Efficient Video Denoising

Wenlong Yang<sup>∗†</sup>, Weihang Yuan<sup>†</sup>, Canran Jin<sup>†</sup>, Jiahao Li<sup>†</sup>, Henrik Turbell<sup>†</sup>, Lifeng Sun<sup>∗</sup>

<sup>∗</sup>Tsinghua University, Beijing, China

<sup>†</sup>Microsoft

ywl22@mails.tsinghua.edu.cn, sunlf@tsinghua.edu.cn

Abstract—Real-world video denoising for communication systems must handle mixed degradations such as sensor noise, compression artifacts, and motion-induced inconsistency while remaining deployable under tight latency budgets. Purely deep models often achieve high accuracy but come with significant computational cost, whereas classical filters are robust and efficient but insufficiently adaptive. We present FilterFusion, a prior-guided multi-frame denoising framework that combines temporally-aware learned features with an interpretable bank of denoising branches. The model predicts a low-cost guidance signal and uses it to adaptively mix a compact set of denoising responses at the original resolution. The lightweight design uses only 109K parameters and explicitly injects temporal variance and spatial variance cues into the low-resolution controller. On CRVD (real sensor noise), it reaches 34.72 dB PSNR and 0.9330 SSIM with only 0.18 GFLOPs (network-only). Because the filterbank operations are simple and regular (Gaussian blur, median, weighted sum), a straightforward compiled C implementation of the synthesis path achieves 17.9 ms per frame (55.9 FPS) on a single CPU thread at 1080p—exceeding real-time and 122×– 252× faster than lightweight CNN baselines that require over 2 seconds per frame. The results demonstrate a strong qualityefficiency-speed trade-off and a practical route to real-time denoising through low-resolution control and explicit filter-based synthesis.

Index Terms—video denoising, real-noise denoising, adaptive filter fusion, prior-guided restoration, lightweight video enhancement

## I. INTRODUCTION

Video denoising is a core component in video communication, live streaming, and low-light capture. In these scenarios, degradations are rarely limited to additive white Gaussian noise; instead, sensor noise, gain amplification, compression artifacts, and motion interact in a spatially and temporally non-uniform manner. This makes real-world denoising substantially harder than synthetic benchmark restoration.

Classical video denoisers exploit temporal redundancy and robust filtering priors, and therefore often behave predictably in deployment [1]–[3]. However, they rely on hand-crafted rules and struggle to adapt to diverse local content. Deep learning methods, in contrast, can learn rich restoration mappings [4]–[7], but their performance may drop sharply when the test degradation deviates from the training distribution. In addition, fully black-box designs are difficult to analyze and to compress for practical systems.

This paper revisits video denoising from a structured design perspective. Instead of forcing a single backbone to learn all restoration behavior implicitly, we explicitly construct several branches with distinct denoising roles and let the network decide how much each branch should contribute at every pixel. The proposed FilterFusion framework combines a temporal feature backbone with a prior-guided filter bank, and further compresses the control path into a low-resolution weight prediction module. A compact prediction head generates pixelwise fusion weights, enabling content-aware and noise-aware selection among complementary operators while avoiding expensive full-resolution neural decoding.

![](images/a31d87d9d89eb2ca715555c0253bb8211b111afa086cbd8d2bbe6f59717016f3.jpg)  
Fig. 1. Multi-dimensional trade-off. Each axis is normalized to [0, 1]: Quality (PSNR, [22, 40] dB), Lightness $( 1 - \mathrm { p a r a m s } / 1 0 ^ { 6 } ;$ classical=max), Structure (SSIM, [0.6, 1.0]), Speed (log FPS), Efficiency (log 1/GFLOPs). The dashed green arc marks the real-time threshold (30 FPS). FilterFusion (red) is the only method that simultaneously exceeds the real-time boundary and surpasses the classical quality ceiling.

The key advantage of this design is that it retains the efficiency, interpretability and cross-degradation robustness of classical filtering while benefiting from learned contentaware adaptivity. The controller assigns responses per pixel: multi-frame temporal averaging that directly suppresses noise in motion-stable regions, multi-scale low-pass smoothing for residual structure on top of averaged signal, median rejection near compression artifacts, and reference-frame preservation across edges, textures, and motion-unreliable areas.

Contributions. Our main contributions are as follows:

• We propose FilterFusion, a structured video denoising framework that embeds classical filtering priors into a learnable multi-branch architecture rather than treating denoising as a monolithic black-box mapping.

![](images/a4370993c3e5869cefee8d35ec15f576da9652b176060d9b95fadebd0d843ca0.jpg)  
Fig. 2. Overview of FilterFusion in luma plane. A compact low-resolution predictor estimates pixel-wise fusion weights from the reference frame $Y _ { \mathrm { r e f } } ,$ temporal mean Y<sup>¯</sup> , and variance guidance $( V _ { s } , V _ { t } )$ , while a full-resolution filter bank generates multiple candidate responses. The final prediction is obtained by adaptively fusing these responses using the predicted weights. For the chroma planes, the same filter-bank synthesis is applied while reusing the weights predicted from the luma plane, producing the denoised NV12 output with negligible additional cost.

• We instantiate the framework as a 109K-parameter multiframe model and introduce explicit temporal variance and spatial variance guidance for robust and compact lowresolution weight prediction.

• We provide detailed real-noise and CPU-side comparisons on CRVD (sensor noise) where the model reaches 34.72 dB with only 0.51 GFLOPs total (0.18 networkonly) and shows a favorable quality-efficiency trade-off, requiring 120×–240× less compute than matched-quality full-resolution CNN baselines.

## II. RELATED WORK

Video Denoising. Classical methods such as VBM3D and related non-local formulations leverage patch redundancy and temporal aggregation to obtain robust denoising performance [1]–[3]. More recent learning-based methods, including DVDnet, FastDVDnet, RViDeNet, and recurrent propagation models, have substantially improved restoration quality on synthetic and curated datasets [4]–[7]. Their main weakness is high computational cost and limited interpretability of the denoising decision in mixed-noise scenarios.

Classical Priors in Learnable Restoration. Many efficient restoration models borrow ideas from classical filtering, residual learning, and multi-scale processing [8]–[11]. Recent hybrid approaches explicitly combine classical denoising modules with learned predictors [12]. Our work follows this line of thought but emphasizes a particularly deployment-friendly form: most adaptive decisions are made in a low-resolution latent space, while full-resolution processing is reduced to a small bank of interpretable filter responses.

## III. APPROACH

Given a temporal window of N frames $\{ I _ { n } \} _ { n = 1 } ^ { N }$ , our goal is to reconstruct a clean reference frame $I _ { \mathrm { r e f } }$ where usually $I _ { \mathrm { r e f } } = I _ { N }$ . Figure 2 illustrates the pipeline. FilterFusion first extracts several compact low-resolution guidance maps from the input sequence and then uses these guidance to fuse several full-resolution denoising responses. This decomposition separates low-cost adaptive decision making from full-resolution image synthesis and is the core reason the method remains computationally practical.

## A. Problem Formulation

Let $\hat { Y } _ { \mathrm { r e f } }$ denote the denoised luma channel of the reference frame; chroma is handled by reusing the same weights, as detailed in the synthesis subsection. We formulate restoration as adaptive mixing over a filter bank:

$$
\hat { Y } _ { \mathrm { r e f } } ( x , y ) = \sum _ { k = 1 } ^ { K } \alpha _ { k } ( x , y ) B _ { k } ( x , y ) ,\tag{1}
$$

where $B _ { k }$ is the k-th filter-bank response on the luma plane and $\{ \alpha _ { k } \}$ is its pixel-wise mixing weight. The key question is how to estimate $\alpha _ { k }$ with minimal cost while keeping the basis functions expressive enough to cover different local noise conditions. Our answer is to predict $\{ \alpha _ { k } \}$ in low-resolution as $\{ \alpha _ { k } \} ^ { \prime }$ and keep the basis functions at the original resolution.

## B. Low-Resolution Weight Prediction

The model uses a compact low-resolution predictor to estimate the spatially varying fusion weights $\{ \alpha _ { k } \} ^ { \prime }$ . Structurally, this predictor is a lightweight encoder-decoder that takes downsampled noisy frames and outputs a small set of fusion logits. The model contains about 109K parameters, so the learned part remains compact compared with conventional full-resolution video restoration networks. In the current implementation, instead of feeding all N frames into the predictor, we only use two frames to supply both spatial and temporal information at minimal computation: reference frame $Y _ { \mathrm { r e f } }$ and the temporal mean $\begin{array} { r } { \bar { Y } = \frac { 1 } { N } \sum _ { n = 1 } ^ { N } Y _ { n } } \end{array}$ . Both are taken from the Y channel which carries the dominant structural information of the frame, and bilinearly downsampled by 8× before entering the predictor. This moves the learned computation away from both the full-resolution domain and the full temporal window.

The predictor outputs a 5-channel logit map in the lowresolution domain, where each channel corresponds to one of the five synthesis responses of the filter bank. After softmax normalization, the predicted guidance is upsampled back to the original frame size and used for full-resolution synthesis. This architecture is motivated by a simple observation: the spatial variation of optimal filter-selection weights is much smoother than the final image itself, so the control signal can be inferred at low resolution without sacrificing the main adaptive behavior. To make low-resolution prediction reliable under motion and structural variation, the predictor is conditioned on two compact guidance signals introduced below.

Temporal Variance Guidance. The temporal condition addresses frame-to-frame inconsistency caused by motion, illumination changes, or flickering. In such regions, blind multiframe fusion risks ghosting artifacts because the controller has no explicit signal about temporal reliability. To resolve this, we compute a per-pixel temporal variance map across the input window and then apply nearest-neighbor downsampling:

$$
V _ { t } = \mathcal { D } _ { \mathrm { N N } } \big ( \mathrm { V a r } \big [ Y _ { n } ( x , y ) \big ] \big ) ,\tag{2}
$$

where $Y _ { n }$ is the luminance channel of the n-th frame in the temporal window. Although the denoising network takes only a compact two-frame input, V is computed from all frames in the window and provides a lightweight temporalconsistency cue that this compact input alone cannot convey, without requiring an explicit motion estimation module. Pixels with larger $V _ { t }$ are guided to rely more on reference-preserving responses rather than aggressive temporal fusion.

Spatial Variance Guidance. The spatial condition addresses a separate problem: after 8× downsampling, each low-resolution pixel corresponds to an 8×8 high-resolution region that may be flat or rich in texture. Bilinear downsampling retains only the mean and discards internal complexity, so the controller cannot distinguish flat-noise areas (where strong smoothing helps) from edge-texture areas (where oversmoothing should be avoided).

To supply this information at negligible cost, spatial variance is computed directly in the low-resolution domain using $M { = } 9$ positions $\{ p _ { k } \} _ { k = 1 } ^ { 9 }$ uniformly sampled from the 8×8 high-resolution coverage region of each low-resolution pixel $( i , j )$ via a 3×3 sub-pixel grid:

$$
V _ { s } ( i , j ) = \frac { 1 } { K } \sum _ { m = 1 } ^ { M } Y _ { \mathrm { r e f } } ^ { ( m ) 2 } ( i , j ) - \left( \frac { 1 } { K } \sum _ { m = 1 } ^ { M } Y _ { \mathrm { r e f } } ^ { ( m ) } ( i , j ) \right) ^ { 2 } .\tag{3}
$$

Low $V _ { s }$ indicates flat regions (sky, walls) where stronger smoothing is safe; high $V _ { s }$ indicates structures that should be preserved.

The model input is constructed as follows. The reference frame $Y _ { \mathrm { r e f } }$ and temporal mean $\bar { Y }$ are also $8 \times$ downsampled to get $Y _ { \mathrm { r e f } } ^ { \prime }$ and ${ \bar { Y } } ^ { \prime }$ , and then concatenated with $V _ { \mathrm { s } }$ and $V _ { \mathrm { t } } .$ . The two variance conditions operate on separate dimensions—temporal consistency across frames and spatial structure within each frame—and jointly improve the reliability of the predicted fusion map under mixed degradations.

## C. High-Resolution Filter-Bank Synthesis

The final output is synthesized from a small bank of full-resolution responses $\{ B _ { k } \} _ { k = 1 } ^ { K } \colon$ (i) the temporal mean Y<sup>¯</sup> for multi-frame noise averaging; (ii) the reference frame for high-frequency detail preservation; (iii) two upsampled Gaussian-pyramid levels of $\bar { Y }$ for progressively stronger lowpass smoothing; and (iv) a spatial median on the reference frame for edge-preserving denoising. These five regimes are deliberately heterogeneous, letting the controller route each pixel to whichever response best matches the local degradation rather than commit to a single operator.

Built on $\bar { Y }$ rather than the reference frame, the Gaussian pyramid couples efficient $3 { \times } 3$ multi-scale blurring with implicit temporal denoising: multi-frame averaging delivers $\sqrt { N }$ noise reduction in motion-stable regions for free, so spatial smoothing acts on a pre-cleaned signal and stays mild enough to preserve textures that a raw-frame equivalent would erase. The median branch covers the complementary regime—motion and high-frequency regions on the reference frame—where temporal averaging ghosts and Gaussian smoothing erase the fine structure.

The fusion module predicts K logits $w _ { k } ( x , y )$ and normalizes them with a softmax:

$$
\alpha _ { k } ( x , y ) = \frac { \exp ( w _ { k } ( x , y ) ) } { \sum _ { i = 1 } ^ { K } \exp ( w _ { i } ( x , y ) ) } .\tag{4}
$$

The weights satisfy $\begin{array} { r } { \sum _ { k } \alpha _ { k } ( x , y ) = 1 } \end{array}$ , so Eq. 1 reduces to a per-pixel convex combination of the K candidate responses $B _ { 1 } , \ldots , B _ { K }$

For the chroma planes, we directly reuse the luma-predicted weights $\{ \alpha _ { k } \}$ : they are upsampled to the chroma resolution $( 1 / 4$ of luma in NV12) and used to combine a UV-side filter bank constructed in the same form as the luma bank. This weight sharing exploits the structural alignment between luma and chroma, so chroma synthesis adds negligible cost on top of luma synthesis while preserving color consistency.

## D. Training Objective

We train FilterFusion end-to-end using a weighted combination of pixel reconstruction, robust Charbonnier, Sobel-based regularization, and frequency-domain supervision:

$$
\begin{array} { r } { \mathcal { L } = \lambda _ { 1 } \mathcal { L } _ { \mathrm { L 1 } } + \lambda _ { 2 } \mathcal { L } _ { \mathrm { c h a r b } } + \lambda _ { 3 } \mathcal { L } _ { \mathrm { s o b e l } } + \lambda _ { 4 } \mathcal { L } _ { \mathrm { f r e q } } . } \end{array}\tag{5}
$$

The Sobel term is applied on the prediction to suppress residual high-frequency noise in smooth regions, while the frequency term aligns global spectral characteristics with the target and helps preserve valid high-frequency components. In our stable training configuration, we use $\lambda _ { 1 } = 1 . 0 , \lambda _ { 2 } = 1 . 0 \nonumber$ $\lambda _ { 3 } = 0 . 1$ , and $\lambda _ { 4 } = 0 . 0 5$

Concretely, we use

$$
\mathcal { L } _ { \mathrm { s o b e l } } = \| \nabla _ { x } \hat { y } \| _ { 1 } + \| \nabla _ { y } \hat { y } \| _ { 1 } ,\tag{6}
$$

and

$$
\mathcal { L } _ { \mathrm { f r e q } } = \Vert \mathcal { F } ( \hat { y } ) - \mathcal { F } ( y ) \Vert _ { 1 } ,\tag{7}
$$

$$
\nabla _ { x } , \nabla _ { y }
$$

$$
\mathcal F ( \cdot )
$$

Training uses the CRVD sRGB clean as ground truth. We optimize the model with Adam using $\beta _ { 1 } = 0 . 9 , \beta _ { 2 } = 0 . 9 9$ an initial learning rate of $1 0 ^ { - 4 }$ , and a multistep schedule with milestones at 140k, 280k, and 360k iterations. The total training length is 600k iterations with batch size 4 per GPU. During optimization, a 2-pixel border crop is excluded from the loss to reduce boundary artifacts.

## IV. EXPERIMENTS

## A. Experimental Setup

We validate the method on CRVD [7], a real sensor-noise dataset containing 11 indoor scenes with substantial object motion across 5 ISO levels (1600–25600) at 1920×1080. Following the official two-stage protocol, we pretrain on synthetic SRVD, fine-tune on CRVD scenes 1–6, and test exclusively on scenes 7–11. We report PSNR, SSIM, and CPU runtime as the main evaluation criteria. The CPU platform is Intel Core i9-10900X at 3.70GHz and the software stack is PyTorch 2.0.1 with OpenCV 4.12.0. To ensure a fair timing comparison, all our methods and baselines are benchmarked using a single CPU thread. FilterFusion timing includes the complete pipeline in NV12 (4:2:0 semi-planar YUV) format: RGB→NV12 conversion, network inference on the Y plane, filter-bank synthesis on the Y plane and on the 1/4- resolution UV planes, and NV12→RGB conversion at the end. The filter-bank kernels (Gaussian, median, weighted sum) are implemented in compiled C with OpenCV’s resize for interpolation—a level of optimization that is straightforward for these simple, regular operations but not applicable to dense learned convolutions.

To keep the manuscript rigorous, we only report experiment settings for which complete training and evaluation records are available. Exploratory runs with incomplete logs or provisional metrics are excluded. We compare against two groups of baselines: classical denoisers with predictable runtime behavior, and deep learning baselines trained on the same data.

## B. Comparison with State-of-the-Art Methods

Table I summarizes the full CPU-side comparison. All deep learning baselines follow the same two-stage protocol as the proposed model: pre-training on SRVD synthetic data followed by fine-tuning on real CRVD data. The proposed model reaches 34.72 dB / 0.9330 while requiring dramatically less compute (0.51 total GFLOPs) than all deep learning baselines.

Role of Temporal Variance Gating. Figure 3 shows how the predicted $V _ { t }$ map gates FilterFusion within one inference pass. On the rotating ball (high $V _ { t } ,$ red box), naive temporal averaging ghosts the KIPSTA logos—the failure mode that pushes TemporalAvg/Median to ∼17 dB in Table I—while FilterFusion reverts to a near single-frame response. On the static sky and power lines (low $V _ { t } ,$ green box), the same model exploits the multi-frame signal to suppress sensor noise without blurring the wires.

Runtime-Quality Analysis. Several key observations emerge from Table I.

Surpassing the classical ceiling. Classical single-frame methods plateau around 33.4 dB regardless of complexity: Bilateral filter achieves 33.40 dB in 187 ms, while the far more expensive NLMeans (2,615 ms, ∼134 GFLOPs) and AnisoDiffusion (1,872 ms) fail to exceed this ceiling. Guided filter scores only 30.30 dB because its self-guided regime lets the noisy input corrupt the linear coefficient estimation. Naive multi-frame fusion without motion compensation is even worse: TemporalAvg, TemporalMedian, and TempAvg+Bilateral all drop to ∼17 dB due to severe ghosting, underscoring the need for learned temporal weighting. Filter-Fusion breaks through the classical ceiling by +1.32 dB over the best classical result (Bilateral, 33.40 dB).

TABLE I  
COMPARISON ON CRVD AT 1920×1080, SORTED BY RUNTIME WITHIN EACH GROUP. ALL TIMINGS ARE ON A single CPU thread (INTEL I9-10900X) FOR OUR METHODS AND BASELINES.
<table><tr><td>Method</td><td>Params</td><td>PSNR ↑</td><td>SSIM ↑</td><td>Time (ms) ↓</td><td>GFLOPs↓</td></tr><tr><td colspan="6">Real-time methods (≥30 FPS)</td></tr><tr><td>Gaussian blur</td><td>一</td><td>32.54</td><td>0.8713</td><td> $\overline { { 7 ^ { \star } } }$ </td><td>~0.06</td></tr><tr><td>Box filter</td><td></td><td>31.61</td><td>0.8822</td><td> $9 ^ { \star }$ </td><td>~0.04</td></tr><tr><td>FilterFusion</td><td>109K</td><td>34.72</td><td>0.9330</td><td> $1 7 . 9 ^ { \star \ddagger }$ </td><td>0.18†</td></tr><tr><td>Median filter</td><td></td><td>32.48</td><td>0.8779</td><td> $2 1 ^ { \star }$ </td><td>~0.16</td></tr><tr><td colspan="6">Non-real-time methods (&lt;30 FPS)</td></tr><tr><td>TemporalAvg</td><td></td><td>17.11</td><td>0.6815</td><td>84</td><td>~0.05</td></tr><tr><td>TemporalAvg + Bilateral</td><td></td><td>17.11</td><td>0.7054</td><td>125</td><td>~1.1</td></tr><tr><td>Guided filter [13]</td><td></td><td>30.30</td><td>0.8909</td><td>163</td><td>~0.22</td></tr><tr><td>Bilateral filter</td><td></td><td>33.40</td><td>0.8926</td><td>187</td><td>~1.0</td></tr><tr><td>IterBilateral</td><td></td><td>33.26</td><td>0.8828</td><td>234</td><td>~3.0</td></tr><tr><td>EMVD [14]</td><td>41.4K</td><td>38.27</td><td>0.9722</td><td>283c</td><td>5.38</td></tr><tr><td>TemporalMedian</td><td></td><td>16.66</td><td>0.6909</td><td>610</td><td>~0.12</td></tr><tr><td>AnisoDiffusion [15]</td><td></td><td>33.02</td><td>0.8622</td><td>1,872</td><td>~5.7</td></tr><tr><td>SimpleCNN_5L</td><td>29.5K</td><td>33.78</td><td>0.9153</td><td>2,195</td><td>60.9</td></tr><tr><td>DepthwiseCNN</td><td>5.8K</td><td>32.88</td><td>0.8750</td><td>2,234</td><td>11.7</td></tr><tr><td>MultiFrameCNNª</td><td>34.7K</td><td>31.54</td><td>0.7697</td><td>2,400</td><td>71.7</td></tr><tr><td>NLMeans [16]</td><td></td><td>33.22</td><td>0.9301</td><td>2,615</td><td>~134.4</td></tr><tr><td>UNetTiny</td><td>261K</td><td>35.37</td><td>0.9461</td><td>4,519</td><td>122.6</td></tr><tr><td>ReMoNet [17]</td><td>809K</td><td></td><td>0.9820</td><td>7,651</td><td></td></tr><tr><td>FastDVDnet [5]</td><td>2.48M</td><td>40.12 39.91</td><td>0.9812</td><td>30,867</td><td>147.63 664.99</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>RViDeNet [7]</td><td>8.56M</td><td>40.03</td><td>0.9802 0.9856</td><td>≥109,294b ~797K</td><td>1,965.04</td></tr><tr><td>Shift-Net [18]</td><td>13.38M</td><td>41.00</td><td></td><td></td><td>2,664.75</td></tr></table>

⋆Marks CPU real-time methods (≥30 FPS).  
†GFLOPs is network-only (0.18); total including filter-bank synthesis is ≈0.51 GFLOPs.  
‡FilterFusion timing uses compiled C filter-bank synthesis (+ cv2.resize).  
<sup>a</sup>CRVD-only (no SRVD phase 1); SRVD lacks aligned temporal neighbors for multi-frame training.  
<sup>b</sup>Lower bound: CUDA-only ops (DCNv2, CrissCrossAttention) replaced by Conv2d. <sup>c</sup>Single-thread timing, for protocol consistency with all other rows; EMVD’s multi-threaded runtime is 128 ms (discussed in text).

Speed-quality trade-off vs. lightweight CNNs. Fullresolution lightweight CNNs are not actually lightweight on CPU at 1080p: despite small parameter counts (5.8K–261K), they require 2.2–4.5 s per frame on a single thread even after two-stage training (SRVD pre-train + CRVD finetune). UNetTiny reaches 35.37 dB—surpassing FilterFusion’s 34.72 dB—but at the cost of 252× slower inference (4,519 ms vs. 17.9 ms) and 240× more compute. This illustrates a key property of our design: FilterFusion trades a modest quality gap (−0.65 dB vs. UNetTiny) for a massive speed advantage, making it the only learned method that can operate above real-time at 1080p. With compiled filter-bank synthesis, FilterFusion achieves 17.9 ms per frame (55.9 FPS)—122× faster than SimpleCNN 5L and 252× faster than UNetTiny— with only 0.51 GFLOPs total. We emphasize that this speedup is architectural: the filter-bank operations (3×3 Gaussian, 5×5 median, weighted sum) are simple enough to admit efficient compiled implementations, whereas CNN baselines already run on highly optimized BLAS/MKL backends and cannot be similarly accelerated. The speed advantage directly validates the core design: low-resolution weight prediction combined with full-resolution filter-bank synthesis avoids the bottleneck of dense learned convolutions at 1080p.

Extreme efficiency gap to state-of-the-art. Published highaccuracy methods achieve 38–41 dB but at dramatically higher cost. The lightest among them, EMVD (38.27 dB), requires 5.38 GFLOPs—30× more than FilterFusion’s network cost (0.18 GFLOPs) and still 10× more than our total cost including filter-bank synthesis (0.51 GFLOPs). This reveals a large, previously unoccupied region in the quality-efficiency landscape: FilterFusion is the only method that simultaneously surpasses the classical quality ceiling while operating at sub-1 GFLOPs total cost, where no prior learned method exists.

Measured CPU runtime of published methods. To move beyond GFLOPs-only comparisons, we profiled several published methods on the same CPU platform at 1920×1080 (Table I, Time column). The results confirm that high-accuracy methods are computationally heavy: FastDVDnet requires ∼11 s per frame, RViDeNet at least 43 s (lower bound, as two CUDA-only operators were replaced by standard convolutions), and Shift-Net exceeds 374 s. EMVD is a notable case: despite reporting 5.38 GFLOPs (30× our network cost), it achieves 128 ms on CPU with multi-threading. FilterFusion’s optimized variant reaches 17.9 ms on a single thread—7.2× faster than EMVD’s multi-threaded result—highlighting that the filter-bank paradigm is not only compute-efficient but also practically fast because its constituent operations (fixed-kernel convolutions, median, weighted sum) are inherently simple and admit efficient native implementations.

Fig. 1 visualizes these multi-dimensional trade-offs across seven representative methods. The dashed arc marks the realtime boundary (30 FPS): FilterFusion (55.9 FPS) and Median filter exceed $\mathrm { i t , }$ but Median’s polygon collapses on quality and structure. FilterFusion achieves the most balanced polygon, with competitive scores on all five axes and the highest quality among real-time-capable methods. Classical methods (Bilateral, NLMeans, Median) score high on lightness (zero parameters) but are limited in quality and—in the case of NLMeans—extremely slow. UNetTiny matches EMVD-level quality (35.37 dB) and structure (0.9461), but both methods collapse on Speed and Efficiency: UNetTiny requires 4.5 s per frame and 122.6 GFLOPs, while EMVD needs 5.38 GFLOPs and 128 ms even with multi-threading. This highlights the central trade-off: higher quality is achievable through heavier models, but FilterFusion occupies a unique position as the only method that simultaneously exceeds the real-time threshold and surpasses the classical quality ceiling.

## C. Ablation Study

To validate each design choice, we conduct component-level ablations on CRVD. Table II reports PSNR and SSIM for each variant. All ablation models are trained with the same schedule and hyperparameters as the full model unless stated otherwise.

Guidance signals (b–d). Removing spatial variance $V _ { s }$ hurts edge-texture preservation, since the low-resolution predictor can no longer distinguish flat-noise regions from struc-

![](images/453b4576137e0c59c5468ba5b2ba8cf54dc8e41914173aebaabfb83e9975f0f5.jpg)  
Fig. 3. Role of $V _ { t }$ gating on CRVD scene7/ISO12800. Inset (top-left) is the predicted $V _ { t }$ map (warm = high, cool = low). On high-V<sub>t</sub> regions (red box), naive multi-frame averaging ghosts the KIPSTA logos while FilterFusion preserves edges. On low-V<sub>t</sub> regions (green box), FilterFusion suppresses sensor noise without blurring the thin power lines.

## TABLE II

ABLATION STUDY ON CRVD. EACH ROW DISABLES OR MODIFIES ONE COMPONENT RELATIVE TO THE FULL MODEL. ALL VARIANTS ARE   
TRAINED single-stage ON THE SAME DATA SPLIT AS THE FULL MODEL IN THIS TABLE (NO SRVD PRE-TRAINING), WITH THE SAME SCHEDULE   
(160K ITERATIONS, ADAM, SAME LR MILESTONES); THEY ARE USED TO ATTRIBUTE QUALITY TO SPECIFIC DESIGN CHOICES AND ARE NOT DIRECTLY COMPARABLE TO THE TWO-STAGE MAIN-TABLE NUMBERS.

<table><tr><td>#</td><td>Variant</td><td>PSNR ↑</td><td>SSIM ↑</td></tr><tr><td>(a)</td><td>Full model (FilterFusion)</td><td>34.7549</td><td>0.9353</td></tr><tr><td colspan="4">Guidance signals</td></tr><tr><td>(b)</td><td>w/o temporal variance  $V _ { t }$ </td><td>34.8418</td><td>0.9336</td></tr><tr><td>(c)</td><td>w/o spatial variance  $V _ { s }$ </td><td>34.6080</td><td>0.9304</td></tr><tr><td>(d)</td><td>w/o neither  $V _ { t }$  nor  $V _ { s }$ </td><td>34.6354</td><td>0.9316</td></tr><tr><td colspan="4">Filter-bank composition</td></tr><tr><td>(e)</td><td>Raw frames only (no spatial filters)</td><td>32.5283</td><td>0.8183</td></tr><tr><td>(f)</td><td>w/o Gaussian-pyramid branches</td><td>33.5668</td><td>0.8909</td></tr><tr><td colspan="4">Input &amp; loss</td></tr><tr><td>(g)</td><td>Single frame (no temporal input)</td><td>25.4074</td><td>0.9043</td></tr><tr><td>(h)</td><td>L1 only (no Charb/Sobel/Freq)</td><td>34.8278</td><td>0.9342</td></tr></table>

![](images/b3da41144c1f90b00f0a82dadb6eff9250879d08a1b5824fdf9ab4bd8c646078.jpg)  
Fig. 4. Qualitative ablation on CRVD ISO 25600 (strong noise, high V<sub>t</sub>): (a) remains heavy residual noise; (b) noise is cleaned but visible ghosting along the ball’s moving edge; (c) clean and ghost-free; (d) ground truth.

tured regions after $8 \times$ downsampling (Fig. 4a). Removing temporal variance $V _ { t }$ degrades quality in motion regions, where the controller loses its explicit cue about temporal reliability and causes ghost artifacts (Fig. 4b). Quantitatively, $V _ { s }$ contributes more than $V _ { t }$ on PSNR (−0.15 vs. +0.07 dB), but $V _ { t }$ is decisive for SSIM (−0.0017): without $V _ { t } ,$ , the controller over-trusts multi-frame averaging—residual staticbackground noise drops (raising PSNR), but motion ghosts at the ball’s edge violate local structure, a distortion SSIM’s local-window metric catches while averaged PSNR hides.

Filter-bank composition (e, f). Variant (e) reduces the filter bank to only the raw input frames (K=2: temporal mean and reference), removing all spatial filters (Gaussian pyramid and median). This tests whether the multi-filter synthesis— the central contribution of our design—adds value beyond simple temporal fusion. Variant (f) selectively removes the Gaussian-pyramid branches while keeping the median filter, testing the importance of multi-scale smoothing. Removing all spatial filters (e) drops PSNR by 2.23 dB and SSIM by 0.117— the largest drop among ablations after (g); removing only the pyramid (f) costs 1.19 dB and 0.044, confirming multi-scale smoothing contributes a substantial portion of the gain.

Input and loss (g, h). The single-frame variant (g) collapses to 25.41 / 0.9043 (−9.35 dB vs. full), confirming that temporal fusion is the dominant source of quality. Variant (h) replaces the combined loss $\mathcal { L }$ with single $\mathcal { L } _ { \mathrm { L 1 } }$ and yields slightly higher PSNR but lower SSIM (34.83 / 0.9342). The result demonstrates that $\mathcal { L } _ { \mathrm { c h a r b } } + \mathcal { L } _ { \mathrm { s o b e l } } + \mathcal { L } _ { \mathrm { f r e q } }$ jointly provide complementary structure and spectrum aware constraints, suppressing residual high-frequency noise and improving structural fidelity beyond pixel-domain supervision.

## D. Discussion

FilterFusion is designed for practical deployment as much as for accuracy. Because the branch semantics are explicit, the system can be simplified by dropping selected branches or by replacing the temporal backbone with a lighter alternative while keeping the overall fusion formulation unchanged. Without explicit motion estimation, temporal-variance guidance steers unreliable regions toward reference-preserving branches to suppress ghosting, so highly dynamic regions receive less temporal denoising as a deliberate quality–latency trade-off.

The filter-bank operations are simple and regular—fixedkernel spatial filters and element-wise weighted sums— which map naturally to efficient compiled code. In contrast, CNN baselines already execute on highly optimized BLAS/MKL backends within PyTorch; their dense learned convolutions leave little room for further implementationlevel gains. The pipeline breakdown at 1080p is: network inference 2.2 ms, Y-channel post-processing 11.5 ms, UV postprocessing 4.2 ms, totaling 17.9 ms (55.9 FPS)—demonstrates that the 0.51 GFLOPs total budget translates directly into realtime throughput on a single CPU thread.

## V. CONCLUSION

We presented FilterFusion, a prior-guided adaptive fusion framework for efficient video denoising. The model predicts a compact low-resolution guidance signal and synthesizes the full-resolution output through a small set of complementary denoising responses. Experiments on CRVD (sensor noise) show that this structured design surpasses the classical quality ceiling (34.72 dB vs. 33.40 dB for Bilateral) while using orders of magnitude less compute than CNN alternatives (240× less than UNetTiny). These results support the central claim of the paper: for deployment-oriented video denoising, low-resolution control combined with explicit high-resolution filter synthesis is a practical and effective design principle. With compiled native filter-bank synthesis, FilterFusion achieves 17.9 ms per frame (55.9 FPS) at 1080p on a single CPU thread—comfortably exceeding real-time without multithreading, and 7.2× faster than EMVD’s multi-threaded result. This is enabled by the architectural choice: the filter-bank operations are simple enough to admit efficient compiled implementations, a property not shared by dense learned convolutions.

## REFERENCES

[1] K. Dabov, A. Foi, V. Katkovnik, and K. Egiazarian, “Image denoising by sparse 3-D transform-domain collaborative filtering,” IEEE Trans. Image Process., vol. 16, no. 8, pp. 2080–2095, 2007.

[2] P. Arias and J.-M. Morel, “Video denoising via empirical Bayesian estimation of space-time patches,” J. Math. Imag. Vis., vol. 60, no. 1, pp. 70–93, 2018.

[3] M. Maggioni, G. Boracchi, A. Foi, and K. Egiazarian, “Video denoising, deblocking, and enhancement through separable 4-D nonlocal spatiotemporal transforms,” IEEE Trans. Image Process., vol. 21, no. 9, pp. 3952– 3966, 2012.

[4] M. Tassano, J. Delon, and T. Veit, “DVDNET: A fast network for deep video denoising,” in Proc. IEEE Int. Conf. Image Process. (ICIP), 2019, pp. 1805–1809.

[5] ——, “FastDVDnet: Towards real-time deep video denoising without flow estimation,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2020, pp. 1351–1360.

[6] K. C. K. Chan, S. Zhou, X. Xu, and C. C. Loy, “BasicVSR++: Improving video super-resolution with enhanced propagation and alignment,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2022, pp. 5972–5981.

[7] H. Yue, C. Cao, L. Liao, R. Chu, and J. Yang, “Supervised raw video denoising with a benchmark dataset on dynamic scenes,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2020, pp. 2298–2307.

[8] K. Zhang, W. Zuo, Y. Chen, D. Meng, and L. Zhang, “Beyond a Gaussian denoiser: Residual learning of deep CNN for image denoising,” IEEE Trans. Image Process., vol. 26, no. 7, pp. 3142–3155, 2017.

[9] K. Zhang, W. Zuo, and L. Zhang, “FFDNet: Toward a fast and flexible solution for CNN-based image denoising,” IEEE Trans. Image Process., vol. 27, no. 9, pp. 4608–4622, 2018.

[10] J. Liu, W. Sun, and J. Liu, “Residual feature distillation network for lightweight image super-resolution,” in Proc. Eur. Conf. Comput. Vis. Workshops (ECCVW), 2020, pp. 41–55.

[11] S. W. Zamir, A. Arora, S. Khan, M. Hayat, F. S. Khan, and M.-H. Yang, “Restormer: Efficient transformer for high-resolution image restoration,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2022, pp. 5728–5739.

[12] L. Sun, F. Wu, W. Ding, X. Li, J. Lin, W. Dong, and G. Shi, “Multiscale spatio-temporal memory network for lightweight video denoising,” IEEE Trans. Image Process., vol. 33, pp. 5810–5823, 2024.

[13] K. He, J. Sun, and X. Tang, “Guided image filtering,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 35, no. 6, pp. 1397–1409, 2013.

[14] M. Maggioni, Y. Huang, C. Li, S. Xiao, Z. Fu, and F. Song, “Efficient multi-stage video denoising with recurrent spatio-temporal fusion,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2021, pp. 3465–3474.

[15] P. Perona and J. Malik, “Scale-space and edge detection using anisotropic diffusion,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 12, no. 7, pp. 629–639, 1990.

[16] A. Buades, B. Coll, and J.-M. Morel, “A non-local algorithm for image denoising,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2005, pp. 60–65.

[17] L. Xiang et al., “ReMoNet: Recurrent multi-output network for efficient video denoising,” in Proc. AAAI Conf. Artif. Intell., vol. 36, no. 3, 2022, pp. 2786–2794.

[18] D. Li et al., “A simple baseline for video restoration with grouped spatial-temporal shift,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2023, pp. 9822–9832.