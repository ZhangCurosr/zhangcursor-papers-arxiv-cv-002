![](images/4689c4fe9990ef1c9877e55a5e1e6cb9b1fd0966282ae8d4824d6045f417757f.jpg)

# Ultra-fast Neural Inference for Stochastic Gaussian Splatting Denoising

Chenxiao Hu<sup>1</sup> Hao Zhang<sup>1</sup> Yanchen Zhang<sup>1</sup> Meng Gai<sup>1</sup> Guoping Wang<sup>1</sup> Sheng Li<sup>1∗</sup>

<sup>1</sup>School of Computer Science, Peking University {hineven, lisheng}@pku.edu.cn

## Abstract

Stochastic rendering eliminates the sorting and alpha blending process in Gaussian splatting, at the cost of introducing spatial noise. Formulating temporal denoising over the pixel stream shared by view-consistent stochastic splatting renderers, we propose a temporal neural denoiser validated on stochastic 2D Gaussian Splatting rendering, combining dual-path exponential moving average accumulation, perpixel learned trust prediction for history validation, a fixed anisotropic spatial filter and a variance-gated composition with stabilization. The denoiser suppresses the noise, achieving temporally stable, visually compelling outputs during free camera navigation, all while retaining the sort-free, blend-free rasterization performance. The combined pipeline retains a PSNR gap to sorted alpha-blending renderers, but the denoiser’s overhead stays below the time saved by removing sorting and blending.

## 1 Introduction

Gaussian splatting represents a scene as anisotropic 3D Gaussians (3DGS) (Kerbl et al. 2023) or oriented 2D surfels (2DGS) (Huang et al. 2024) and renders it through depth sorting and alpha blending. Sorted compositing delivers noise-free images and high visual fidelity, but its cost grows linearly with scene complexity-more primitives, larger frusta, more output channels-until memory bandwidth and fill rate saturate. Engineering works keep the sort-and-blend structure and optimize it directly, delivering large speedups with an unchanged rendering model (Feng et al. 2025; Wang et al. 2024; Hanson et al. 2025). Order-independent transparency methods remove the global sort and its view-dependent ordering (Hou et al. 2025; Hahlbohm et al. 2025; Du et al. 2026). A third group redesigns the primitive itself, gaining more compact scenes or more accurate geometry (Ye et al. 2025, 2026). Our work follows a fourth route, stochastic rendering (Kheradmand et al. 2025; Sun et al. 2025; Rijsdijk et al. 2026), which removes sorting and blending altogether.

With stochastic rendering, each fragment is retained with probability equal to its opacity contribution and resolved using a standard depth test. This eliminates global sorting, replaces alpha blending with opaque z-bufer compositing, and enables additional output channels to be decoded from a visibility bufer at minimal cost. The resulting image is an unbiased Monte Carlo estimate of conventional alpha blending. StochasticSplats (Kheradmand et al. 2025) demonstrates that the estimator matches the alpha-blended image in expectation while running fast. Its remaining problem is Monte Carlo noise: at one sample per pixel, the estimate is far too noisy for practical use, and the generic temporal anti-aliasing (TAA, Karis (2014)) or frame averaging used to suppress it is insuficient under free camera navigation.

![](images/27005791aeb877fb49b124b658798d28e854918e1545401ed03430d0ad46ef85.jpg)

![](images/3f04c54226e55e15a5187db9259c409151a6c829f95e38be67499be939f25e81.jpg)  
Figure 1: Top left: Garden navigation. Bottom left: zoom-in by conventional α-blended 2DGS renderer, by stochastic rasterization with ST-TAA (see Sec. 5), and by our approach on the same setting. ST-TAA sufers from residual Monte Carlo noise, whereas our method visually resembles the conventional renderer. Right: per-frame time breakdown. Despite the additional cost of denoising, our approach achieves significant speedup by avoiding the expensive Gaussian depth sort and α-blending required by conventional splatting.

Temporal accumulation can repair such noise, but it relies on accurate reprojection, which presumes view-consistent geometry. Per-view projection and per-Gaussian sorted compositing violate this assumption. Several works improve the view consistency of splatting by refining the sorting strategy (Radl et al. 2024; Liu et al. 2025) or change the primitive or blending model (Shen et al. 2024; Huang et al. 2025), and work well within the sorted pipeline; others avoid Gaussian depth sorting altogether, anchoring geometry to worldspace surfels whose per-pixel visibility is resolved exactly by z-bufering or depth peeling (Ye et al. 2025, 2026). For our experiments, we take a minimal modification of 2DGS from per-Gaussian to per-pixel depth ordering, as a prototype that keeps the scene representation mostly comparable to the vanilla baseline.

Our method. With semi-transparent fragments jittering between frames and no noise-free G-Bufer guidance, heuristics and generic anti-aliasing methods malfunction; a dedicated denoiser is required. We design a lightweight temporal neural denoiser for view-consistent stochastic splatting. The denoiser maintains two exponential moving averages per pixel: an accumulated path accumulates raw stochastic samples and converges to the unbiased alpha-blended image; a denoised path accumulates spatially filtered frames, suppressing visually prominent noise in disocclusions with low frame accumulation. A variance gate blends the two averages per pixel. A neural network predicts how much reprojected history to trust per pixel; a fixed anisotropic mipmap spatial kernel with learned per-Gaussian parameters performs the filtering. The pipeline is optimised under a single photometric objective. The denoiser’s overhead is far below the time saved by removing sorting and blending, as seen in Fig. 1.

Our contributions are as follows:

• A formulation of temporal denoising over the per-pixel stream shared by view-consistent stochastic splatting renderers, and a lightweight neural denoiser for the stream denoising problem, optimised end-to-end and validated on a per-pixel-ordered 2DGS host.

• An experimental demonstration that this pairing makes stochastic rasterization beneficial: on a minimal perpixel-ordered 2D Gaussian host, the pipeline approximates the converged Monte-Carlo quality on static views, stays visually satisfying under free navigation, runs faster than the sorted alpha-blending renderer.

## 2 Related Work

## 2.1 Gaussian Splatting and Rasterization Acceleration

3D Gaussian Splatting (3DGS) (Kerbl et al. 2023) represents a scene as anisotropic 3D Gaussians and renders them with tile-based rasterization: primitives are depth-sorted per frame and alpha-blended front to back, achieving real-time radiance-field rendering with high visual quality. Keeping the representation and the sort-and-blend model unchanged, a line of work pushes eficiency further through aggressive culling and pipeline scheduling (Feng et al. 2025), Tensor-Core-friendly reformulation of the per-pixel computations (Li et al. 2026), or precise splat localization and pruning (Hanson et al. 2025).

## 2.2 Gaussian Splatting Variants

A second body of work modifies the representation or the compositing model itself. One goal is view-consistency: evaluating each Gaussian at a locally optimal per-pixel depth with hierarchical resorting (Radl et al. 2024), replacing alpha blending with an order-independent weighted sum (Hou et al. 2025), anchoring the scene in opaque z-bufered surfels (Ye et al. 2025), or recovering the exact per-pixel order of semi-transparent surfels with depth peeling (Ye et al. 2026). A second goal is eficiency, through adaptive rasterization radii and tile load balancing (Wang et al. 2024), order-independent compositing on mobile GPUs (Du et al. 2026), or z-bufered opaque layers with selective blending (Hahlbohm et al. 2025).

Stochastic rendering. StochasticSplats (Kheradmand et al. 2025) applies stochastic transparency to splats: each fragment is emitted as fully opaque with probability equal to its opacity contribution, which removes sorting and blending and makes each frame an unbiased Monte Carlo estimate of the alpha-blended image. The simplicity provides performance gain, but also brings severe Monte Carlo noise. The output only becomes visually satisfying after accumulating many frames, which assumes a static camera. Ray-traced formulations, compatible with stochastic or deterministic sampling (Sun et al. 2025; Moenne-Loccoz et al. 2024; Condor et al. 2025), sidestep splatting entirely, but forgo the throughput of rasterization.

2D Gaussian Splatting (2DGS) (Huang et al. 2024) represents scenes as planar surfels, whose exact ray-splat intersection depth provides more view-consistency. Our work builds on this property: better consistency makes temporal reprojection more reliable and temporal denoising plausible.

## 2.3 Denoising, Super-Resolution, Anti-Aliasing

Kernel-predicting denoising, an influential family of Monte Carlo denoisers, predicts filtering kernels for noisy inputs (Bako et al. 2017; Vogels et al. 2018); related variants regress the clean image from feature-guided inputs (Kalantari et al. 2015; Bitterli et al. 2016), operate on individual samples (Gharbi et al. 2019), or target self-supervision and interactive budgets (Yu et al. 2021; Back et al. 2022; Meng et al. 2020; Işık et al. 2021; Bálint et al. 2023). Our denoiser is very similar to this family, but is designed for purely noisy inputs. Regular real-time Monte Carlo denoisers commonly rely on noise-free G-bufers. Stochastic splatting emits no noise-free guidance so existing denoisers cannot be applied to stochastic Gaussian splatting with trivial modification. Zeng et al. (2026) uses neural network for real-time superresolution on 3DGS rendering, while we employ neural network for denoising: a diferent post-processing task.

Temporal Anti-Aliasing (Karis 2014) stabilizes frames by reprojecting the history bufer along motion vectors and clipping colors to the local neighborhood. Learning-based supersampling follows the same history-driven paradigm (Xiao et al. 2020; Zhong et al. 2023; Yang et al. 2024). They expect near-noise-free input and reliable motion vectors, neither of which stochastic splatting provides.

## 3 Method

## 3.1 Overview

Standard alpha-blended 2DGS produces noise-free images, but pays for per-frame global depth sorting and heavy overdraw from semi-transparent fragments; both costs grow with primitive count, frustum size, and channel count. Stochastic rasterization removes both: each fragment is randomly retained with probability equal to its opacity contribution and resolved by an ordinary depth test, and auxiliary channels decode from the visibility bufer at little extra cost. The resulting image is an unbiased Monte Carlo estimate of the alpha-blended reference, but its single-sample-per-pixel noise is visually prominent.

Temporal filtering is a proven remedy for Monte Carlo noise. The denoiser must cost well below the savings from removing sorting and blending, run without the noise-free G-bufers conventional real-time denoisers rely on, and accumulate history only where the observed surface is viewpointinvariant: a property standard 2DGS lacks due to per-Gaussian depth ordering.

We therefore formulate temporal denoising over per-pixel stream of view-consistent stochastic renderer (Sec. 3.2) as a recurrent model with sparse learned components:

$$
\begin{array} { l r } { \boldsymbol { c } _ { t } = \mathcal { R } ( \boldsymbol { o } _ { t } , h _ { t - 1 } ) , } & { \quad ( t _ { s } , t _ { h } ) = f _ { \theta } ( \boldsymbol { c } _ { t } ) , } \\ { \boldsymbol { d } _ { t } = F _ { \phi } ( \boldsymbol { o } _ { t } ) , } & { \quad ( S _ { t } , D _ { t } ) = \mathcal { U } ( \boldsymbol { o } _ { t } , d _ { t } , h _ { t - 1 } , t _ { s } , t _ { h } ) , } \\ { \boldsymbol { g } _ { t } = \mathcal { G } _ { \gamma } ( \mathrm { v a r } _ { t } , n _ { s } , n _ { h } ) , } & { \quad { \hat { I } } _ { t } = g _ { t } D _ { t } + ( 1 - g _ { t } ) \boldsymbol { S } _ { t } . } \end{array}\tag{1}
$$

Here $o _ { t }$ is the current stochastic observation (colour, depth, decoded payload), $h _ { t - 1 }$ the temporal state, and $\hat { I } _ { t }$ the output frame. R reprojects the state into the current view and derives consistency features (Sec. 4.1); $f _ { \theta }$ predicts how much reprojected history each of two complementary paths may trust; $F _ { \phi }$ is a fixed anisotropic kernel with footprints parameterised by learned neural-view scalars $\phi$ (Sec. 4.3); U is the exponential-moving-average update (Eq. (2)) yielding an accumulated mean $S _ { t }$ over raw samples and a denoised mean $D _ { t }$ over filtered frames; and $\mathcal { G } _ { \gamma }$ is a variance gate (Eq. (3)) fusing the two means per pixel. R, F, U, G are fixed diferentiable operators; learning is confined to the trust networks $\theta ,$ the gate scalar γ, and the payload $\phi ,$ which are optimised jointly under a single photometric objective on $\hat { I } _ { t }$ through the same recurrent state transition used at inference (Sec. 4.4). Validation instantiates the simplest host model satisfying the stream requirements (Sec. 3.3); Figure 2 illustrates the full pipeline.

## 3.2 The Stochastic Splatting Stream

Per pixel and per frame, a stochastic splatting renderer emits a visibility sample: Identity of the primitive winning an opaque depth test after stochastic retention, together with its depth; auxiliary per-primitive payload channels (colors, or trainable parameters such as the neural-view scalars in Sec. 4.3) are decoded from the visibility sample at little extra cost. Two properties make this stream temporally denoisable: view-consistency, so history can be reprojected safely, and approximately unbiased decoded radiance, so accumulating samples converges to α-blended rendering. Noise magnitude and view-consistency vary across such renderers (Kheradmand et al. 2025; Sun et al. 2025; Rijsdijk et al. 2026), they have a common issue: Low samples per pixel introduce prominent Monte Carlo noise. This can be efectively suppressed by our denoiser using only the above quantities.

## 3.3 Minimal Host Model

For validation we instantiate the simplest model that satisfies the requirements: vanilla 2DGS with the per-Gaussian centre-depth ordering replaced by per-pixel depth ordering, rendered with the same stochastic visibility rule as our primary baseline (Kheradmand et al. 2025) so the comparison isolates the denoiser rather than the geometry model. A pretrained vanilla 2DGS scene is converted by full finetuning through a depth-peeling diferentiable rasterizer (Everitt 2001; Laine et al. 2020); the conversion is not free, leaving a residual gap of about 1 dB to vanilla 2DGS on average. We deliberately keep the dense, unpruned reconstruction so the geometry stays comparable to the 2DGS baseline.

## 3.4 Denoiser Design

The compute budget left by the rasterization savings is narrow, and the stream ofers no noise-free G-Bufer. Fixed heuristics, e.g. blending by depth diference alone, are therefore insuficient. We adopt a kernel-estimation-style denoiser that spends most of its budget predicting, per pixel, how much reprojected history can be trusted $( f _ { \theta }$ of Eq. (1)), using only stream quantities (Sec. 3.2).

Accumulated and Denoised paths. Temporal accumulation of raw stochastic samples converges to the alpha-blended reference, but the first few frames are noisy, and suppressing that noise spatially would require aggressive blurring, which lowers the asymptotic quality ceiling once history is long. We therefore maintain two exponential-moving-average (EMA) chains in parallel: an accumulated path over raw samples, which dominates stable regions once history is suficient, and a denoised path over spatially filtered frames, which covers short histories and fills disoccluded regions where reprojection is unreliable. A variance gate blends the two per pixel (Figure 3).

EMA path update (U). At each frame, the previous temporal state is first warped into the current camera view (Sec. 4.1). Each path maintains a per-pixel running mean together with an efective sample count, its history length. Let $X _ { \mathrm { p r e v } }$ be a path’s reprojected mean, n its reprojected history length $( n _ { s } , n _ { h }$ for the two paths), x the current frame’s value on that path (the spatially filtered colour or the raw stochastic colour) and $t \in [ 0 , 1 ]$ the predicted trust. The state updates as

![](images/389056a5562c13c7d002e580364e1e1a193c5584de16044dc51b4b1ed1ad7d03.jpg)  
Figure 2: Workflow of our denoiser for each frame (Eq. (1)) with trainable modules highlighted. The host stochastic rasterizer emits the 1 spp stream (Sec. 3.2). In the accumulated path (left), the previous accumulated state is reprojected into the current view and a convolutional predictor estimates the history trust $t _ { s }$ from reprojection-derived motion, depth, and photometric cues. In the denoised path (middle), the current frame is spatially filtered by a fixed anisotropic mipmap kernel parameterised by smoothed neural-view scalars, and a lightweight head estimates the trust $t _ { h }$ of the reprojected denoised mean. In postprocessing (right), both paths update their exponential moving average (EMA) states (Eq. (2)), a variance gate (Eq. (3)) composites the two means per pixel, and a stabilization pass (STAB) suppresses residual jitter. The updated states propagate to the next frame.

![](images/ee10505803d6a5b5d67b93a06f20cb720a01261bdab250d266ca9002d4a89320.jpg)  
Figure 3: The two EMA paths and their composition at 4 (top) and 64 (bottom) accumulated frames under slow lateral camera motion; the zoomed region is marked on the reference. The accumulated path is noisy at low frame counts but improves with history, while the denoised path is stable but blurry; the variance gate blends the two adaptively (white: prefer denoised, black: prefer accumulated). At 4 frames the not-yet-calibrated gate dips the composite marginally below the denoised path; the dip vanishes within a few frames. PSNR is computed on the zoomed region.

$$
d = n \cdot t + 1 , X = { \frac { X _ { \mathrm { p r e v } } \cdot n \cdot t + x } { d } } , n  \operatorname* { m i n } ( d , \ n _ { \operatorname* { m a x } } ) .\tag{2}
$$

The trust t interpolates between two extremes: t = 1 appends the current frame as one full sample, an ordinary running average; $t \ : = \ : 0$ discards the history and restarts from the current frame; fractional values discount stale history without resetting it, damping ghosting. The denoised path uses a smaller history cap (24 vs. 128) for reactivity. A parallel running average of squared luminance on the accumulated path tracks the variance driving the composition gate.

Variance-gated composition $( \mathcal G _ { \gamma } )$ and STAB. The final pixel mixes the two paths according to the accumulated path’s own uncertainty. Along the accumulated path we maintain a running average of squared luminance $\overline { { y ^ { 2 } } }$ alongside the colour mean ${ \bar { y } } ,$ giving the temporal variance $\operatorname { v a r } { \dot { = } } \operatorname* { m a x } ( { \overline { { y ^ { 2 } } } } - { \bar { y } } ^ { 2 } , 0 )$ and an estimate of the standard error of the mean, err $= \sqrt { \mathrm { v a r } / ( n _ { s } + \epsilon ) }$ . The gate weight of the denoised path is

$$
g = 1 - r \cdot { \bigl ( } 1 - \operatorname { c l a m p } ( \gamma \cdot \operatorname { e r r } , 0 , 1 ) { \bigr ) } , r = 1 - e ^ { - n _ { s } / \tau } ,\tag{3}
$$

with a single learned scalar $\gamma$ and a fixed time constant $\tau = 8$ frames; the final colour is $g \cdot D + ( 1 - g ) \cdot S$ . Early ${ \mathrm { o n } } , r \approx 0$ and the denoised path dominates; as history accumulates and the standard error shrinks, the sharp accumulated path takes over. Error spikes such as motion and disocclusion can raise g again, pulling output to the denoised path.

A stabilization pass (STAB) suppresses residual temporal jitter of the composited output: a lightweight TAA variant with neighborhood clamping, a uniform blend weight, and a color-consistency rescue against spurious disocclusion resets from depth jitter (details in the appendix).

Trust prediction $\left( f _ { \theta } \right)$ . Most of the inference budget goes to the accumulated path’s trust $t _ { s } .$ , predicted by a small convolutional network from motion, depth, and photometricconsistency cues; the denoised path’s $t _ { h }$ comes from a tiny head biased to distrust history at initialization (architectures in Sec. 4.2). When the camera is static, we override $t _ { s }$ to 1, where samples are i.i.d. and the running mean is the minimum-variance estimator.

Spatial filtering $( F _ { \phi } )$ . The spatial filter must be nearly free yet cover a wide footprint: a fixed four-level anisotropic mipmap filter in the spirit of EWA filtering (Zwicker et al. 2002), driven per pixel by learned neural-view scalars ${ \bf V } =$ $( V _ { 0 } , V _ { 1 } , V _ { 2 } , V _ { 3 } )$ baked into the surfels (Sec. 4.3). The payload is scene-specific and adds 10 floats per Gaussian.

## 4 Technical Details

## 4.1 Reprojection and Trust Features

The denoiser begins by warping the previous frame’s temporal state into the current view: each current pixel is unprojected to world space, reprojected into the previous image plane, and resampled with a 16-tap Catmull-Rom filter, retrieving the previous denoised and accumulated means, the two history lengths, and the two trust maps. Holes left by stochastic sampling (invalid current depth) borrow the average valid depth in a $3 \times 3$ neighbourhood, falling back to the far plane if the neighbourhood is empty.

From the reprojection we derive four families of trust features: motion (screen-space motion magnitude, softcompressed as $m / ( m + 2 0 )$ , and its square), depth (relative depth diference and its motion-gated product), photometric (luminance and $\mathbf { Y C o C g }$ distances to the two reprojected means), and state (reprojected trust maps, history lengths, and the forward-facing neural-view scalar $V _ { 2 }$ , Sec. 4.3). Together with the current observation these form $c _ { t }$ of Eq. (1); the exact channels are listed in the appendix.

## 4.2 History Trust Predictors

Accumulated-path trust $t _ { s }$ . The predictor is a deep separable convolutional stack with a hidden width of 8: a pointwise $1 \times 1$ encoder, two depthwise $3 \times 3 +$ pointwise $1 \times 1$ blocks with SiLU activations, and a pointwise head whose first output channel, passed through a sigmoid, yields $t _ { s } .$ Its $5 \times 5$ receptive field captures local motion boundaries cheaply, and the head bias is initialised to zero $( t _ { s } = 0 . 5$ at training start).

Denoised-path trust $t _ { h }$ . A tiny head computes $t _ { h } \colon$ a single depthwise $3 \times 3$ convolution with SiLU followed by a pointwise projection to a scalar and a sigmoid. The output bias is initialised to $- 4 \ ( t _ { h } \ \approx 0 . 0 2$ at cold start), so the denoised path initially trusts the current frame almost entirely.

The gate scalar $\gamma$ is also learnable.

## 4.3 Spatial Anisotropic Mipmap Filtering

The filter operates on the 4-channel neural-view tensor V carried by the surfels, which is the baked form of the per-Gaussian payload $\phi$ of Eq. (1): $V _ { 0 } , V _ { 1 } , V _ { 2 }$ are the absolute dot products of three learnable per-surfel view vectors with the camera basis (up, right, and forward), $V _ { 0 } , V _ { 1 }$ normalized by per-pixel depth and $V _ { 2 }$ controlling the film-space x-y correlation of the kernel; $V _ { 3 }$ is a learned scalar controlling blur strength. Trained jointly with the denoiser, these optimal filter footprints are baked into the Gaussian parameters.

A fixed $3 \times 3$ Gaussian blur of the neural-view tensor yields smoothed scalars from which a $2 \times 2$ covariance matrix is built; its eigen-decomposition parameterises the filter over a 4-level mipmap pyramid of the current frame’s $\mathbf { Y C o C g }$ colour, probed along the covariance’s major axis (exact construction in the appendix). The per-pixel filtering cost is thus bounded by 32 texel reads.

## 4.4 Training-Time Details

Training splits into two phases whose costs we account separately: preparing the minimal host model (Stages 1–3), and the joint optimisation of the denoiser and the payload (Stage 4), which is the cost of our contribution. All other training configurations are deferred to the appendix.

Minimal host model (Stages 1–3). A vanilla 2DGS scene is optimised under the conventional per-Gaussian ordering (Stage 1, 30k steps), cleaned of low-opacity floaters by an opacity-only finetune (Stage 2, 1k steps), and converted to the per-pixel depth-ordered model of Sec. 3.3 by full finetuning through a depth-peeling diferentiable rasterizer (Stage 3, 3k steps). This phase merely converts an of-the-shelf 2DGS reconstruction into our minimal host; a natively viewconsistent splatting model would skip it.

Joint optimisation (Stage 4). All learned components of Eq. (1) are optimised jointly under a single photometric objective (L1 with small SSIM/LPIPS and auxiliary terms, appendix) on the final output ${ \ddot { I } } _ { t } ,$ with the host Gaussian geometry and colour frozen, using the deterministic tilerendered image as a clean target. Gradients flow through the stochastic renderer, the spatial filter, both trust predictors, and the EMA-gate composition in a single graph; optimisation proceeds with history states detached for training eficiency. Temporal behaviour is learned on synthetic camera chains whose length follows a curriculum up to 128 frames, matching the inference-time history cap (chain synthesis and trust-shaping regularisers in the appendix). The payload ϕ is baked per Gaussian and therefore trains per scene; the shared $\theta$ adapts to a new scene with a short finetune (15k steps on the source scene, 4k steps on transferred scenes, ∼1 hour), and can be deployed with no per-scene denoiser training at some quality cost (zero-shot transfer, Sec. 5.5).

## 5 Experiments and Ablations

## 5.1 Implementation Details

We train with PyTorch and SlangPy (Slang project contributors 2025) and implement the integrated renderer with NVRHI (NVRHI Developers 2026) (Vulkan backend); all experiments run on an NVIDIA RTX 3090 GPU. All scenes share the same hyper-parameters (details in Appendix) and the two-phase training recipe of Sec. 4.4 (room as the source scene). We provide code and running instructions for reproduction in the supplements.

## 5.2 Experimental Setup

Datasets. We evaluate on the main release (7 scenes) of Mip-NeRF360 (Barron et al. 2022) and the training-data subset of TanksAndTemples (Knapitsch et al. 2017). All renderers are evaluated at the native training resolution (resized to 1600 px image width on both datasets); we cap the Gaussian count at 3-million for training eficiency.

Baselines. We compare against the deterministic vanilla 2DGS tile renderer (Huang et al. 2024), which also renders the reference images for all metrics, and against an adapted implementation of stochastic rasterization pipeline from StochasticSplats (Kheradmand et al. 2025). StochasticSplats originally renders a 3DGS scene finetuned under StopThePop rasterization (Radl et al. 2024) which difers from our minimal 2DGS models. We render our per-pixel 2DGS scenes with the same stochastic rasterizer that feeds our denoiser, and reimplement the temporal anti-aliasing (TAA) accumulation described in that work on its output. We refer to this adapted baseline as ST-TAA below: baseline and denoiser consume the identical 1 spp stream of the identical scene representation, so any quality diference is attributable to the denoiser alone. We further compare against SVGF (Schied et al. 2017), classic real-time Monte Carlo denoiser, minimally adapted to the stochastic stream: 3x3 depth hole-fix, temporal accumulation with per-pixel luminance moments, followed by an à-trous wavelet filter edgestopped by depth and variance-guided luminance, with handtuned parameters (details in Appendix). On static cameras, ST-TAA reduces to equal-weight frame accumulation; we additionally report a 1024-sample reference accumulation $( R e f . ~ ( I O 2 4 ) )$ , Monte-Carlo mean of 1024 independent 1 spp renders of the same view (or of each frame’s exact pose along navigation tracks), which upper-bounds what any temporal filter can extract from the stochastic stream.

Metrics. We report PSNR, SSIM, and LPIPS(VGG), plus ColorVideoVDP (Mantiuk et al. 2024) (JOD units) (CVVDP) for navigation video quality (synthesized from every 10th track frame of the ∼300 fps recordings for evaluation speed, ∼30 fps). Temporal metrics are averaged over every 10th frame of each navigation track, using the vanilla tile-rendered images as the reference.

## 5.3 Static Quality on Benchmark Datasets

Table 1 reports static quality on the two benchmarks (perscene numbers in the appendix). The remaining gap from the 1024 sample Monte-Carlo mean (Ref.) to the deterministic renderer is the residual cost of converting the geometry to the per-pixel model (Sec. 3.3). On a static camera any temporal filter can only converge to the reference. Our 128-frame output wins a small margin at equal samples. The win comes from the variance gate, which admixes a small fraction of the spatially filtered denoised path into the accumulated color. From a practical operating point, what matters is how quickly the output converges. Our 16-frame output closes most of the gap to the reference, coming within 1.1 dB of its 128-frame quality, and converges markedly faster than ST-TAA because its denoised path gathers spatially available samples (convergence curves in the appendix). On TanksAndTemples, where all methods are trained on the full capture of each scene, the 128-frame PSNR likewise closes to the bound (20.63 vs. 20.69 dB), with a small residual SSIM/LPIPS gap.

<table><tr><td></td><td colspan="2">MipNeRF360</td><td colspan="3">TanksAndTemples</td></tr><tr><td>Method</td><td>PSNR↑</td><td>SSIM↑ LPIPS↓</td><td></td><td>PSNR↑ SSIM↑</td><td>LPIPS↓</td></tr><tr><td>2DGS</td><td>29.45</td><td>0.875</td><td>0.199</td><td>21.50</td><td>0.700 0.379</td></tr><tr><td>Ref. (1024)</td><td>28.25</td><td>0.851</td><td>0.202</td><td>20.69 0.675</td><td>0.388</td></tr><tr><td>ST-TAA (128)</td><td>27.92</td><td>0.822</td><td>0.219</td><td>20.61</td><td>0.635 0.431</td></tr><tr><td>Ours (16)</td><td>26.89</td><td>0.741</td><td>0.330</td><td>20.33</td><td>0.548 0.496</td></tr><tr><td>Ours (98)</td><td>27.94</td><td>0.822</td><td>0.226</td><td>20.60 0.631</td><td>0.436</td></tr><tr><td>Ours (128)</td><td>28.02</td><td>0.829</td><td>0.217</td><td>20.63</td><td>0.640 0.428</td></tr><tr><td>Ours (z.s. 128)</td><td>28.03</td><td>0.829</td><td>0.217</td><td>20.62</td><td>0.639 0.428</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Table 1: Static quality, averaged per dataset; per-scene PSNR is reported in the appendix. Vanilla 2DGS is the deterministic tile renderer (Huang et al. 2024). Ref. (1024) is the Monte-Carlo accumulation reference (Sec. 5). ST-TAA is the adapted StochasticSplats baseline (Sec. 5), accumulated over 128 static frames. Ours renders through the learned denoiser at 16, 98, or 128 accumulated frames (spp counts in parentheses); the 98 spp point approximately matches ST-TAA (128) in wall-clock time, making it comparable (Table 3); bold marks the better of this equal-time pair in each column (both when equal after rounding). Ours (z.s.) deploys the room checkpoint on every scene with no per-scene denoiser training (zero-shot transfer) (Sec. 5.5).

## 5.4 Temporal Quality under Free Navigation

Free navigation is a common real-time workload. We record camera tracks by hand in the interactive viewer on all seven MipNeRF360 scenes (provided in our supplements), mimicking how a user actually navigates, and compare five renderers: raw 1 spp stochastic output, ST-TAA, SVGF, our denoiser, and the vanilla tile renderer as the reference. Table 2 averages the metrics over the seven tracks (per-track numbers in the appendix). Our denoiser stays close to the deterministic reference, while ST-TAA fails to suppress the Monte Carlo noise and leaves heavy, clearly visible flickering under camera motion (Figure 4). Its collapse is not a convergence limit but a failure to retain valuable temporal history, which is what our learned trust prediction addresses. SVGF fares markedly better than plain TAA, but without noise-free guidance features its fixed edge-stopping strategy is severely interfered by floating Gaussians and holes in scenes. The Ref. (1024) ceiling sits well above every temporal method under navigation, whereas on static cameras our denoiser essentially reaches it (Table 1); the residual gap thus comes from reusing history across poses: reprojection mismatch and disocclusions.

## 5.5 Zero-Shot Transfer to Unseen Scenes

The trust predictors consume mostly relative cues of the stochastic error process, so the learned trust partially transfers across scenes. We test this by copying the room checkpoint verbatim to the other scenes, neutralising the scene-specific neural-view payload: the trust feature $V _ { 2 }$ fed to the trust estimator is set to zero, and the spatial filter falls back to a fixed isotropic kernel $( a _ { x } = a _ { y } = 2 , \rho = 0 )$ . In tables 1 and 2, static quality of Ours (z.s.) matches the fine-tuned model, while under navigation the zero-shot variant gives up some PSNR/CVVDP margin yet slightly improves SSIM and LPIPS: the fixed kernel filters more aggressively than the learned kernels, which is favored by the perceptual metrics. Either way the diferences are small next to the margin over ST-TAA. Per-track numbers are in the appendix.

<table><tr><td>Method</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>CVVDP↑</td><td>ms↓</td></tr><tr><td>Ref. (1024)</td><td>32.96</td><td>0.950</td><td>0.080</td><td>8.73</td><td>一</td></tr><tr><td>Raw 1 spp</td><td>18.26</td><td>0.282</td><td>0.671</td><td>4.94</td><td>2.71</td></tr><tr><td>SS 1 spp</td><td>17.86</td><td>0.292</td><td>0.659</td><td>4.67</td><td>3.2†</td></tr><tr><td>SS 4 spp†</td><td>22.54</td><td>0.526</td><td>0.540</td><td>6.87</td><td>6.5†</td></tr><tr><td>ST-TAA</td><td>21.48</td><td>0.418</td><td>0.615</td><td>5.95</td><td>2.88</td></tr><tr><td>SVGF</td><td>25.57</td><td>0.785</td><td>0.297</td><td>6.29</td><td>2.86</td></tr><tr><td>Ours</td><td>29.80</td><td>0.867</td><td>0.244</td><td>7.57</td><td>3.75</td></tr><tr><td>Ours (z.s.)</td><td>29.34</td><td>0.873</td><td>0.223</td><td>7.38</td><td>3.64</td></tr></table>

Table 2: Quality under free camera navigation on MipNeRF360, averaged over all scene tracks (per-track numbers in the appendix), using the 2DGS tile renderer as the reference. Raw 1 spp is the single-sample stochastic output; ST-TAA is the adapted StochasticSplats baseline, temporal accumulation without learned trust or spatial filtering; SVGF is the adapted variance-guided filtering baseline (see appendix). Ref. (1024) is the per-frame Monte-Carlo ceiling, not a practical method. The last column is the mean per-frame time over the seven tracks on an RTX 3090. <sup>†</sup>SS 1 spp/4 spp are the oficial StochasticSplats model run on its own finetuned 3DGS scenes over the same camera tracks for 1 or 4 spp, measured against its own 1024 spp accumulation as the reference (no ground truth exists on free tracks); it is listed to show that the 1 spp noise level is inherent with the stochastic formulation. Its per-frame error distribution closely tracks our Raw 1 spp row (per-frame PSNR correlation r = 0.90, and its CVVDP score likewise matches our Raw 1 spp row). SS timings come from the oficial viewer. Ours (z.s.) deploys the room checkpoint on every scene with no per-scene denoiser training (Sec. 5.5).

## 5.6 Performance Analysis

Table 3 reports per-frame rendering time averaged over the seven navigation tracks (per-scene numbers in the appendix). Raw stochastic rasterization is 2.9× faster than the sorted alpha-blending renderer on the same hardware-rasterization pipeline, and the full denoising pipeline keeps a 2.1× advantage: the denoiser adds ∼1 ms per frame, nearly constant across scenes. The speedup varies considerably with scene content, from 1.2× (kitchen, where blending overdraw is low) to 3.5× (bicycle), consistent with the scaling behaviour of sorted blending (Sec. 3.1), and stays well above the denoiser’s overhead.

![](images/4f7c654ddae04c0bfb9d1e93a8ff22ed142df71c2d8f831d524cf7e754e48b05.jpg)  
Figure 4: Garden (top) and Room (bottom) navigation tracks. Left: vanilla-rendered reference crop with the zoom region marked; right: the region for each renderer.

<table><tr><td>Method (ms/frame)</td><td>7-track mean ↓</td></tr><tr><td>2DGS Huang et al. (2024)</td><td>17.96</td></tr><tr><td>2DGS (hardware raster.)</td><td>7.99</td></tr><tr><td>Raw 1 spp</td><td>2.71</td></tr><tr><td>+ TAA (ST-TAA)</td><td>2.88</td></tr><tr><td>Ours (full pipeline)</td><td>3.75</td></tr></table>

Table 3: Per-frame rendering time, averaged over the seven navigation tracks (per-scene numbers in the appendix). The 2DGS row is the oficial implementation measured with auxiliary channel outputs removed; the hardware raster. row is our hardware-rasterized reimplementation of 2DGS, included as a stronger baseline. The cost of denoiser is nearly constant across scenes.

## 5.7 Ablations

Table 4 ablates the learned trust prediction, the spatial filter, and the final stabilization pass on the garden navigation track. The learned trust is the largest single component: forcing t<sub>s</sub> ≡ 1 approximates a naive ’always accumulate’ strategy. The spatial filter is the second contributor, providing filtered results in disoccluded or recently revealed regions. The stabilization pass improves all three per-frame metrics.

## 6 Discussion, Limitations, and Conclusion

Balancing Temporal Stability And Ghosting. We found that the final output can still exhibit visible flicker. We have to use STAB to trade temporal stability with some minor ghosting. We observe lower-than-expected predicted trust within flickering regions, which possibly relates to the limited capacity of predictor networks. To alleviate this issue, it is possible to further enhance the denoising path, or reduce the variance of the stochastic output distribution itself.

Performance across diferent rendering paths. Our current realization of the whole pipeline does not match the absolute single-channel throughput of highly optimised sortedblending engines (Feng et al. 2025), order-independent compositors (Du et al. 2026), or compact surfel-hybrid represen-

<table><tr><td>Variant</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td></tr><tr><td>Full model</td><td>26.63</td><td>0.794</td><td>0.238</td></tr><tr><td>w/o  $t _ { s }$  prediction  $( t _ { s } \equiv 1 )$ </td><td>22.51</td><td>0.695</td><td>0.323</td></tr><tr><td>w/o  $t _ { s }$  prediction  $( t _ { s } = \mathrm { h e u r . } )$ </td><td>25.82</td><td>0.755</td><td>0.268</td></tr><tr><td>w/o spatial filter</td><td>23.52</td><td>0.676</td><td>0.359</td></tr><tr><td>w/o stabilization pass</td><td>25.94</td><td>0.736</td><td>0.311</td></tr></table>

Table 4: Ablations on the garden navigation track (evaluated on every 10th frame of the original 1563 frames, metrics against the vanilla tile renderer). The learned trust prediction is the most impactful single component, followed by the spatial filter. The stabilization pass improves all three metrics and, more importantly, removes most of the visible temporal jitter at the cost of slight ghosting. For the $t _ { s } ~ = ~ \mathrm { h e u r }$ . ablation, we replace the neural network with $\begin{array} { r } { t _ { s } = \exp ( - \sigma \cdot \frac { | d _ { 1 } - d _ { 2 } | } { \operatorname* { m a x } ( d _ { 1 } , d _ { 2 } ) } ) } \end{array}$ heuristic as a stronger baseline, where $d _ { 1 } , d _ { 2 }$ are pixel depths ofthe reprojected/current pixels and σ is grid-searched on the same track for best performance (oracle).

tations (Ye et al. 2025), and we make no claim of superiority over these diferently engineered pipelines.

However, the comparison also shifts with scene characteristics and host implementation. Sorted and blended pipelines pay per-fragment costs that grow with primitive count, overdraw, and the number of output channels, whereas the stochastic pipeline pays a roughly per-pixel cost plus a fixed denoiser overhead; recent stochastic renderers already drive hundreds of millions of Gaussians interactively (Rijsdijk et al. 2026), where sorted pipelines struggle. Within this landscape, our contribution is deliberately narrow: removing the Monte-Carlo noise that has so far blocked the stochastic route under free navigation, at a ∼1ms per-frame overhead.

Composability and future work. These directions compose rather than compete. Because the denoiser’s input contract is only the view-consistent stochastic visibility stream of Sec. 3.2, it can in principle sit behind a culled or more compact representation (Ye et al. 2025) or a faster implementation of stochastic rasterization (Rijsdijk et al. 2026). On the engineering side, the stochastic rasterization natively produces a noisy depth bufer at zero extra cost, which can potentially enable hierarchical Z-culling (Greene et al. 1993). Integration is left to future work.

In all, we present a lightweight temporal neural denoiser that makes view-consistent stochastic Gaussian splatting practical for free-navigation rendering while preserving the eficiency of sort-free, blend-free rasterization. By decoupling the stream denoising problem from a specific splatting representation, our approach ofers a practical postprocessing stage for future stochastic renderers.

## References

Jonghee Back, Binh-Son Hua, Toshiya Hachisuka, and Bochang Moon. Self-supervised post-correction for

Monte Carlo denoising. In ACM SIGGRAPH 2022 Conference Proceedings, 2022. doi: 10.1145/3528233.3530730.

Steve Bako, Thijs Vogels, Brian McWilliams, Mark Meyer, Jan Novák, Alex Harvill, Pradeep Sen, Tony DeRose, and Fabrice Rousselle. Kernel-predicting convolutional networks for denoising Monte Carlo renderings. ACM Transactions on Graphics, 36(4):97:1–97:14, 2017. doi: 10.1145/3072959.3073708.

Martin Bálint, Krzysztof Wolski, Karol Myszkowski, Hans-Peter Seidel, and Rafał Mantiuk. Neural partitioning pyramids for denoising Monte Carlo renderings. In ACM SIGGRAPH 2023 Conference Proceedings, 2023. doi: 10.1145/3588432.3591562.

Jonathan T. Barron, Ben Mildenhall, Dor Verbin, Pratul P. Srinivasan, and Peter Hedman. Mip-nerf 360: Unbounded anti-aliased neural radiance fields. CVPR, 2022.

Benedikt Bitterli, Fabrice Rousselle, Bochang Moon, José A. Iglesias-Guitián, David Adler, Kenny Mitchell, Wojciech Jarosz, and Jan Novák. Nonlinearly weighted firstorder regression for denoising Monte Carlo renderings. Computer Graphics Forum, 35(4):107–117, 2016. doi: 10.1111/cgf.12954.

Jorge Condor, Sebastien Speierer, Lukas Bode, Aljaz Bozic, Simon Green, Piotr Didyk, and Adrian Jarabo. Don’t splat your gaussians: Volumetric ray-traced primitives for modeling and rendering scattering and emissive media. ACM Transactions on Graphics, 44(1), 2025. doi: 10.1145/3711853.

Xiaobiao Du, Yida Wang, Kun Zhan, and Xin Yu. Mobile-gs: Real-time gaussian splatting for mobile devices. In International Conference on Learning Representations (ICLR), 2026.

Cass Everitt. Interactive order-independent transparency. Technical report, NVIDIA Corporation, 2001.

Guofeng Feng, Siyan Chen, Rong Fu, Zimu Liao, Yi Wang, Tao Liu, Boni Hu, Linning Xu, Zhilin Pei, Hengjie Li, Xiuhong Li, Ninghui Sun, Xingcheng Zhang, and Bo Dai. Flashgs: Eficient 3d gaussian splatting for large-scale and high-resolution rendering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 26652–26662, 2025.

Michaël Gharbi, Tzu-Mao Li, Miika Aittala, Jaakko Lehtinen, and Frédo Durand. Sample-based Monte Carlo denoising using a kernel-splatting network. ACM Transactions on Graphics, 38(4):125:1–125:12, 2019. doi: 10.1145/3306346.3322954.

Ned Greene, Michael Kass, and Gavin Miller. Hierarchical z-bufer visibility. In Proceedings ofthe 20th Annual Conference on Computer Graphics and Interactive Techniques (SIGGRAPH), pages 231–238. ACM, 1993.

Florian Hahlbohm, Fabian Friederichs, Tim Weyrich, Linus Franke, Moritz Kappel, Susana Castillo, Marc Stamminger, Martin Eisemann, and Marcus Magnor. Eficient perspective-correct 3d gaussian splatting using hybrid transparency. Computer Graphics Forum, 44(2), 2025. doi: 10.1111/cgf.70014.

Alex Hanson, Allen Tu, Geng Lin, Vasu Singla, Matthias Zwicker, and Tom Goldstein. Speedy-splat: Fast 3d gaussian splatting with sparse pixels and sparse primitives. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 21537– 21546, 2025.

Qiqi Hou, Randall Rauwendaal, Zifeng Li, Hoang Le, Farzad Farhadzadeh, Fatih Porikli, Alexei Bourd, and Amir Said. Sort-free gaussian splatting via weighted sum rendering. In International Conference on Learning Representations (ICLR), 2025.

Binbin Huang, Zehao Yu, Anpei Chen, Andreas Geiger, and Shenghua Gao. 2d gaussian splatting for geometrically accurate radiance fields. In ACM SIGGRAPH 2024 Conference Papers, 2024. doi: 10.1145/3641519.3657428.

Yi-Hua Huang, Ming-Xian Lin, Yang-Tian Sun, Ziyi Yang, Xiaoyang Lyu, Yan-Pei Cao, and Xiaojuan Qi. Deformable radial kernel splatting. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025.

Mustafa Işık, Krishna Mullia, Matthew Fisher, Jonathan Eisenmann, and Michaël Gharbi. Interactive Monte Carlo denoising using afinity of neural features. ACM Transactions on Graphics, 40(4):37:1–37:13, 2021. doi: 10.1145/3450626.3459793.

Nima Khademi Kalantari, Steve Bako, and Pradeep Sen. A machine learning approach for filtering Monte Carlo noise. ACM Transactions on Graphics, 34(4):122:1– 122:12, 2015. doi: 10.1145/2766977.

Brian Karis. High-quality temporal supersampling. In ACM SIGGRAPH 2014 Courses: Advances in Real-Time Rendering in Games. ACM, 2014.

Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Transactions on Graphics, 42(4):139:1–139:14, 2023. doi: 10.1145/3592433.

Shakiba Kheradmand, Delio Vicini, George Kopanas, Dmitry Lagun, Kwang Moo Yi, Mark Matthews, and Andrea Tagliasacchi. Stochasticsplats: Stochastic rasterization for sorting-free 3d gaussian splatting. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 26326–26335, 2025.

Arno Knapitsch, Jaesik Park, Qian-Yi Zhou, and Vladlen Koltun. Tanks and temples: Benchmarking large-scale scene reconstruction. ACM Transactions on Graphics, 36 (4):78:1–78:13, 2017. doi: 10.1145/3072959.3073599.

Samuli Laine, Janne Hellsten, Tero Karras, Yeongho Seol, Jaakko Lehtinen, and Timo Aila. Modular primitives for high-performance diferentiable rendering. ACM Transactions on Graphics, 39(6):194:1–194:14, 2020. doi: 10.1145/3414685.3417861.

Sheng Li, Yang Sui, Yue Wu, Zhuoran Song, Bo Yuan, Xulong Tang, and Yue Dai. Accelerating 3d gaussian splatting using tensor cores. arXiv preprint arXiv:2605.17855, 2026.

Weihang Liu, Yuke Li, Yuxuan Li, Jingyi Yu, and Xin Lou. Duplex-gs: Proxy-guided weighted blending for real-time order-independent gaussian splatting, 2025. URL https: //arxiv.org/abs/2508.03180.

Rafal K. Mantiuk, Param Hanji, Maliha Ashraf, Yuta Asano, and Alexandre Chapiro. Colorvideovdp: A visual difference predictor for image, video and display distortions. ACM Trans. Graph., 43(4), July 2024. ISSN 0730- 0301. doi: 10.1145/3658144. URL https://doi.org/10. 1145/3658144.

Xiaoxu Meng, Quan Zheng, Amitabh Varshney, Gurprit Singh, and Matthias Zwicker. Real-time Monte Carlo denoising with the neural bilateral grid. In Eurographics Symposium on Rendering - DL-only Track, pages 13–24, 2020. doi: 10.2312/sr.20201133.

Nicolas Moenne-Loccoz, Ashkan Mirzaei, Or Perel, Riccardo de Lutio, Janick Martinez Esturo, Gavriel State, Sanja Fidler, Nicholas Sharp, and Zan Gojcic. 3d gaussian ray tracing: Fast tracing of particle scenes. ACM Transactions on Graphics, 43(6), 2024. doi: 10.1145/3687934.

NVRHI Developers. Nvrhi. https://github.com/NVIDIA-RTX/NVRHI, 2026.

Lukas Radl, Michael Steiner, Mathias Parger, Alexander Weinrauch, Bernhard Kerbl, and Markus Steinberger. Stopthepop: Sorted gaussian splatting for view-consistent real-time rendering. ACM Transactions on Graphics, 43 (4), 2024. doi: 10.1145/3658187.

Joris Rijsdijk, Christoph Peters, Michael Weinmann, and Ricardo Marroquim. Gaussian point splatting. ACM Trans. Graph., 45(4), 2026. doi: 10.1145/3811272.

Christoph Schied, Anton Kaplanyan, Chris Wyman, Anjul Patney, Chakravarty R. Alla Chaitanya, John Burgess, Shiqiu Liu, Carsten Dachsbacher, Aaron Lefohn, and Marco Salvi. Spatiotemporal variance-guided filtering: Real-time reconstruction for path-traced global illumination. In Proceedings of High Performance Graphics (HPG ’17), pages 2:1–2:12, 2017. doi: 10.1145/3105762. 3105770.

Zhuowen Shen, Yuan Liu, Zhang Chen, Zhong Li, Jiepeng Wang, Yongqing Liang, Zhengming Yu, Jingdong Zhang, Yi Xu, Scott Schaefer, Xin Li, and Wenping Wang. Solidgs: Consolidating gaussian surfel splatting for sparseview surface reconstruction, 2024. URL https://arxiv.org/ abs/2412.15400.

Slang project contributors. SlangPy: Python bindings for the Slang shading language. https://github.com/shaderslang/slangpy, 2025.

Xin Sun, Iliyan Georgiev, Yun Fei, and Miloš Hašan. Stochastic ray tracing of transparent 3d gaussians. In Eurographics Symposium on Rendering (EGSR), 2025.

Thijs Vogels, Fabrice Rousselle, Brian McWilliams, Gerhard Röthlin, Alex Harvill, David Adler, Mark Meyer, and Jan Novák. Denoising with kernel prediction and asymmetric loss functions. ACM Transactions on Graphics, 37(4): 124:1–124:15, 2018. doi: 10.1145/3197517.3201388.

Xinzhe Wang, Ran Yi, and Lizhuang Ma. Adr-gaussian: Accelerating gaussian splatting with adaptive radius. In SIGGRAPH Asia 2024 Conference Papers, 2024. doi: 10.1145/3680528.3687675.

Lei Xiao, Salah Nouri, Matt Chapman, Alexander Fix, Douglas Lanman, and Anton Kaplanyan. Neural supersampling for real-time rendering. ACM Transactions on Graphics, 39(4):142:1–142:12, 2020. doi: 10.1145/ 3386569.3392376.

Sipeng Yang, Yunlu Zhao, Yuzhe Luo, He Wang, Hongyu Sun, Chen Li, Binghuang Cai, and Xiaogang Jin. MNSS: Neural supersampling framework for real-time rendering on mobile devices. IEEE Transactions on Visualization and Computer Graphics, 30(7):4271–4284, 2024. doi: 10.1109/TVCG.2023.3259141.

Keyang Ye, Tianjia Shao, and Kun Zhou. When gaussian meets surfel: Ultra-fast high-fidelity radiance field rendering. ACM Transactions on Graphics, 44(4), 2025. doi: 10.1145/3730925.

Keyang Ye, Hongzhi Wu, and Kun Zhou. Depth peeling for high-fidelity gaussian-enhanced surfel rendering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2026.

Jiaqi Yu, Yongwei Nie, Chengjiang Long, Wenju Xu, Qing Zhang, and Guiqing Li. Monte Carlo denoising via auxiliary feature guided self-attention. ACM Transactions on Graphics, 40(6), 2021. doi: 10.1145/3478513.3480565.

Huimin Zeng, Yue Bai, and Yun Fu. Arbitrary-scale 3d gaussian super-resolution. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, pages 12304–12312, 2026.

Zhihua Zhong, Jingsen Zhu, Yuxin Dai, Chuankun Zheng, Guanlin Chen, Yuchi Huo, Hujun Bao, and Rui Wang. FuseSR: Super resolution for real-time rendering through eficient multi-resolution fusion. In SIGGRAPH Asia 2023 Conference Papers, 2023. doi: 10.1145/3610548. 3618209.

Matthias Zwicker, Hanspeter Pfister, Jeroen van Baar, and Markus Gross. EWA splatting. IEEE Transactions on Visualization and Computer Graphics, 8(3):223–238, 2002. doi: 10.1109/TVCG.2002.1021576.

This appendix provides additional implementation details, extended quantitative and qualitative evaluations, and further experiments that complement the main text.

## A Predictor Input Channels

Tables 5 and 6 list the exact per-channel inputs of the two trust predictors. All channels are per-pixel maps assembled during reprojection. Please refer to the code for more details.

Accumulated-path predictor architecture. The full layer sequence is: pointwise 1×1 encoder → depthwise $3 \times 3 ~ +$ pointwise 1×1 → pointwise 1×1 → depthwise 3×3 + pointwise 1×1 → pointwise head, all with SiLU activations and hidden width 8. The main text’s description of “two depthwise 3×3 + pointwise 1×1 blocks” elides the intermediate pointwise layer between the two depthwise blocks for brevity.

Ch. Input   
0 $\vert Y _ { \mathrm { c u r } } - Y _ { \mathrm { s t o c h } } \vert :$ luma dif to reprojected accumulated mean   
1 reprojected previous $t _ { s }$   
2 squared compressed motion magnitude   
3 accumulated-path history length (normalised)   
4 compressed motion magnitude   
5 relative depth diference   
6 motion-gated relative depth diference   
7 forward-facing neural-view scalar (V<sub>2</sub>)  
Table 5: Accumulated-path trust predictor $( t _ { s } )$ inputs, 8 channels.

Ch. Input   
0 $\left| Y _ { \mathrm { c u r } } - Y _ { \mathrm { d e n } } \right|$ : luma diference to reprojected denoised mean   
1 reprojected denoised-path luma   
2 reprojected previous $t _ { h }$   
3 denoised-path history length (normalised)   
4 compressed motion magnitude   
5 YCoCg distance to reprojected denoised mean   
6 relative depth diference   
7 squared compressed motion magnitude  
Table 6: Denoised-path trust head $( t _ { h } )$ inputs, 8 channels.

## B Spatial Filter Details

Exact construction of the covariance matrix in the spatialfiltering section of the main text. A fixed $3 \times 3$ Gaussian blur of the 4-channel neural-view tensor produces smoothed scalars $( V _ { 0 } ^ { \prime } , V _ { 1 } ^ { \prime } , V _ { 2 } ^ { \prime } , V _ { 3 } ^ { \prime } )$ . Per-axis scales $\begin{array} { r l } { a _ { x } } & { { } = } \end{array}$ $\mathrm { r e l u } ( V _ { 1 } ^ { \prime } ) + \mathrm { s o f t p l u s } ( V _ { 3 } ^ { \prime } )$ and $a _ { y } = \mathrm { r e l u } ( V _ { 0 } ^ { \prime } ) + \mathrm { s o f t p l u s } ( V _ { 3 } ^ { \prime } )$ (the softplus floor guarantees a minimum blur) and a normalised correlation $\rho = \operatorname { t a n h } ( V _ { 2 } ^ { \prime } )$ form the covariance matrix $\left[ \begin{array} { l l } { a _ { x } } & { z } \\ { z } & { a _ { y } } \end{array} \right]$ with $z = \rho \sqrt { a _ { x } a _ { y } + \epsilon }$ . Its eigen-decomposition gives major/minor axis lengths $\lambda _ { 1 } , \lambda _ { 2 }$ and orientation θ. The minor axis selects the mip level clamp(log λ<sub>2</sub>, 0, 3); four probes at ofsets $\{ - 1 . 5 , - 0 . 5 , 0 . 5 , 1 . 5 \} \times \lambda _ { 2 }$ along the major axis are trilinearly sampled and averaged with weights $\omega _ { i } = \exp \bigl ( - \frac { 1 } { 2 } ( o _ { i } \lambda _ { 2 } / \lambda _ { 1 } ) ^ { 2 } \bigr )$

## C Stabilization Pass (STAB) Details

Purpose. The trust for the two EMA chains is not sharp enough, so large regions in the composited output retain a characteristic residual noise: temporally high-frequency (perframe flicker that does not converge), spatially low-frequency (∼3 px blobs), and small in amplitude (∼0.05 in [0, 1]) but still visually annoying. Derived from TAA, STAB is a postpass on the composited image that removes this jitter. It is bypassed whenever the camera is static (tolerance-based detection on the camera matrix).

Design. Let C be the current composited colour and H the previous stabilized frame, reprojected into the current view with Catmull–Rom sampling (which the reprojection stage already uses for the EMA chains). STAB computes

$$
\mathrm { o u t } = a \cdot \mathrm { c l a m p } ( H , C - \mathrm { t o l } , C + \mathrm { t o l } ) + ( 1 - a ) \cdot C ,\tag{4}
$$

where the neighbourhood tolerance tol $= \begin{array} { r l } { \mathrm { \nabla \frac { 1 } { 2 } } ( C _ { \mathrm { m a x } } ^ { 3 \times 3 } ~ - } \end{array}$ $C _ { \operatorname* { m i n } } ^ { 3 \times 3 } ) + s$ is computed per channel from the current frame’s $3 \times 3$ neighbourhood (already available from history rectification) with slack $s = 0 . 0 4$ ; the clamp bounds the ghosting amplitude to the slack level. The blend weight is a fixed $a = 0 . 9$ , except that a depth-disocclusion test re-seeds the history $( a = 0 )$ when the relative depth change exceeds 0.1, ensuring recovery on genuine surface changes.

Rescue (Color fix). Because the stochastic depth winner jitters even on stable surfaces, the depth re-seed false-fires on many pixels that could safely be smoothed. A rescue signal may override the re-seed (OR-logic): the Colorfix keeps blending when the clamped history is already close to the current frame, $\| C - \mathrm { c l a m p } ( H ) \| _ { \infty } < 0 . 1 0 .$ , a direct visualdiference test that is unlikely to fire on real motion edges; it can cause minor ghosting. Since the clamp caps ghosting regardless, rescuing is mostly safe. The output of Eq. (4) becomes both the displayed frame and the next frame’s history. The cost of STAB is below measurement noise.

## D Training Configuration

Two-phase recipe. Table 7 summarises the training pipeline (main text, Training-Time Details). Stages 1–3 prepare the minimal host model: Stages 1–2 optimise a vanilla 2DGS scene under the conventional per-Gaussian ordering, and Stage 3 converts it to the per-pixel depth-ordered model of the main text by full finetuning through a depth-peeling differentiable rasterizer. Stage 4 trains the denoiser jointly with the neural-view payload under the stochastic renderer, with the remaining Gaussian parameters frozen.

Host model stages. Stage 1 trains vanilla 2DGS for 30k steps without normal regularisation $( \lambda _ { n } = 0 )$ , on par with the original 2DGS results; enabling the normal regulariser costs about 1 dB because it constrains surfel orientation. Stage 2 finetunes opacity only for 1k steps (penalty weight 0.003, prune threshold 0.05) to remove low-opacity floaters without touching geometry.

<table><tr><td>Stage</td><td>Renderer</td><td>Optimised</td><td>Steps</td></tr><tr><td>1</td><td>Tile Rasterizer</td><td>all</td><td>30k</td></tr><tr><td>2</td><td>Tile Rasterizer</td><td>opacity</td><td>1k</td></tr><tr><td>3</td><td>Depth Peeler</td><td>full finetune</td><td>3k</td></tr><tr><td>4</td><td>Stochastic Rasterizer</td><td>denoiser +  $V _ { 0 , 1 , 2 , 3 }$ </td><td> $1 5 \mathrm { k } / 4 \mathrm { k }$ </td></tr></table>

Table 7: Two-phase training recipe (four stages). Stage 4 trains 15k steps on room and transfers to every other scene with 4k steps, starting from the room denoiser. The 4k transfer is our default per-scene refinement. The room checkpoint can also be deployed with no per-scene denoiser training at some quality cost (main text, Zero-Shot Transfer).

Objective. The final-frame loss combines L1, SSIM (0.05) and LPIPS (0.01) against the tile-rendered vanilla image—a clean, deterministic target—plus an auxiliary L1 term (weight 0.2) pulling the accumulated path toward the vanilla colour.

Synthetic temporal chains. Since no video sequences exist for the training views, temporal behaviour is learned on synthetic camera chains: 70% of steps use temporal history, chain lengths follow a curriculum from 2 to 128 frames, and only the last 8 frames of each chain are rendered stochastically while older frames come from the cheap tile renderer (fake start), so a 128-frame chain costs about nine stochastic renders. Camera perturbations are sampled from five classes—lateral dolly, drift, large, small, and near-static (1/15 scale)—covering both smooth motion and disocclusion.

Trust shaping. Three small mechanisms stabilise the learned trust: (i) near-static regularisers pull $t _ { s }  1$ and $t _ { h }  1$ (both weights 0.01; fast transfer uses 0.005 for both and adds a disocclusion penalty of weight 0.05 on both) on near-static chains, encouraging full accumulation when nothing moves; (ii) an authoritative disocclusion mask computed from the tile renderer’s depth and opacity hard-zeroes predicted trust in genuinely disoccluded pixels during state updates, with an L1 penalty (weight 0.03) suppressing trust there during training; and (iii) the composition gate is forced to the denoised path for the first 2000 steps, so the gate scalar γ is calibrated only after both paths produce meaningful outputs. A symmetry regulariser (weight 0.01) keeps the spatial filter’s axis scales consistent.

## E SVGF Baseline Details

Our SVGF baseline follows Schied et al.’s SVGF: tempora accumulation with first/second luminance moments, then à- trous wavelet filtering edge-stopped by the estimated temporal variance and geometry. It runs in the same Vulkan pipeline as our denoiser and consumes the identical 1 spp stream. The stochastic stream provides no noise-free G-bufer, so the following minimum adaptations are made:

• Hole-fixed depth. Miss pixels of the 1 spp visibility pass carry no valid depth (flagged by zero accumulated alpha); they are filled from the nearest valid 3×3 neighbour before any depth use, and pixels with no valid neighbour fall back to the far plane. All reprojection and edge-stopping read this hole-fixed depth.

• Temporal accumulation. Per pixel, the previous frame’s integrated colour and moments are reprojected (current depth → world → previous camera, Catmull-Rom history sampling) and blended exponentially: c ← $( 1 - \alpha ) c _ { \mathrm { h i s t } } + \alpha c _ { \mathrm { c u r } }$ with $\alpha = 0 . 0 2$ , likewise for the luminance moments $m _ { 1 } , m _ { 2 } .$ . A pixel is treated as disoccluded (reset to the current frame) when fewer than 2 of the 4 nearest previous-depth texels pass the relativedepth test with threshold 0.4. The feedback history is the pre-filter integrated colour, as in classic SVGF.

• Spatial filter. A single à-trous iteration (stride 1) of the 3×3 binomial kernel $\scriptstyle { \frac { 1 } { 1 6 } } [ 1 2 1 ; 2 4 2 ; 1 2 1 ]$ (we tried multiple iterations and it blurs the output violently, causing significant visual quality downgrade. Thus, we only kept 1 iteration.). Per-tap weights combine a relative-depth bilateral term against the 3×3- smoothed centre depth, $\exp \bigl ( - | \Delta z | / ( \sigma _ { z } \bar { z } + 1 0 ^ { - 2 } ) \bigr )$ with $\sigma _ { z } ~ = ~ 0 . 3 .$ and the variance-guided luminance term $\exp ( - | \Delta l | / ( \sigma _ { l } \sqrt { \mathrm { v a r } } + 1 0 ^ { - 2 } ) )$  with $\sigma _ { l } ~ = ~ 8$ , where $\mathrm { v a r } \stackrel { \cdot } { = } \operatorname* { m a x } ( 0 , \bar { m } _ { 2 } - \bar { m } _ { 1 } ^ { 2 } )$ is estimated from the 3×3- averaged moments. The paper’s depth-gradient and normal terms are dropped: the 1 spp winner depth jitters between surfaces, so screen-space depth gradients are noise (they quantise the filter taps into a visible grid), and no surface normals exist.

All parameters $( \alpha = 0 . 0 2 ,$ , one à-trous iteration, $\sigma _ { l } = 8 ,$ $\sigma _ { z } = 0 . 3 ,$ , disocclusion threshold 0.4) were hand-tuned on the interactive viewer for the best visual trade-of between residual noise and blur; per-track numbers are included in Table 13.

## F Per-Scene Rendering Performance

Table 9 reports the per-scene per-frame times underlying the main text’s performance table, measured on the navigation tracks (RTX 3090, median of three pipelined runs). Sorted alpha blending varies widely with scene content (4.5– 13.7 ms), while the full stochastic pipeline stays in a narrow band (3.4–4.0 ms); the denoiser’s marginal cost over raw rasterization is about 1.0 ms on every scene. Table 8 adds the SVGF baseline and the zero-shot variant, which share the same stochastic rasterization front end and difer only in the post-processing stage.

<table><tr><td>Scene</td><td>SVGF</td><td>Ours (zero-shot)</td></tr><tr><td>bicycle</td><td>2.94</td><td>3.76</td></tr><tr><td>bonsai</td><td>2.63</td><td>3.44</td></tr><tr><td>counter</td><td>2.91</td><td>3.66</td></tr><tr><td>garden</td><td>2.97</td><td>3.75</td></tr><tr><td>kitchen</td><td>3.00</td><td>3.76</td></tr><tr><td>room</td><td>3.04</td><td>3.79</td></tr><tr><td>stump</td><td>2.56</td><td>3.35</td></tr><tr><td>mean</td><td>2.86</td><td>3.64</td></tr></table>

Table 8: Per-scene per-frame rendering time (ms) for the SVGF baseline and the zero-shot variant, measured under the identical protocol as Table 9. Both consume the same 1 spp stochastic stream; SVGF replaces the learned postprocessing with hand-tuned variance-guided filtering, and the zero-shot variant skips per-scene denoiser training.
<table><tr><td>Scene</td><td>2DGS</td><td>2DGS (hw)</td><td>Raw</td><td>ST-TAA</td><td>Ours</td><td>Speedup</td></tr><tr><td>bicycle</td><td>29.32</td><td>13.73</td><td>2.86</td><td>3.03</td><td>3.95</td><td>3.5×</td></tr><tr><td>bonsai</td><td>17.66</td><td>8.14</td><td>2.49</td><td>2.66</td><td>3.52</td><td>2.3×</td></tr><tr><td>counter</td><td>16.24</td><td>6.33</td><td>2.74</td><td>2.89</td><td>3.75</td><td>1.7×</td></tr><tr><td>garden</td><td>16.27</td><td>6.57</td><td>2.79</td><td>2.97</td><td>3.85</td><td>1.7×</td></tr><tr><td>kitchen</td><td>10.38</td><td>4.49</td><td>2.84</td><td>3.01</td><td>3.86</td><td>1.2×</td></tr><tr><td>room</td><td>13.94</td><td>6.18</td><td>2.89</td><td>3.04</td><td>3.88</td><td>1.6×</td></tr><tr><td>stump</td><td>21.91</td><td>10.47</td><td>2.38</td><td>2.53</td><td>3.44</td><td>3.0×</td></tr><tr><td>mean</td><td>17.96</td><td>7.99</td><td>2.71</td><td>2.88</td><td>3.75</td><td>2.1×</td></tr></table>

Table 9: Per-scene per-frame rendering time (ms) on the navigation tracks. 2DGS is the oficial CUDA tile renderer with auxiliary channels removed; 2DGS (hw) is our hardware-rasterized reimplementation of the same sorted alpha-blending algorithm; Raw is 1 spp stochastic rasterization; ST-TAA adds the baseline temporal accumulation; Ours is the full denoising pipeline. Speedup is 2DGS (hw) over Ours.

## G Static Convergence Curve

Figure 5 shows how static-view quality builds up over the first accumulated frames. Each point renders the same static view for k consecutive frames and reports PSNR against ground truth, averaged over the 24 held-out views of garden. On static views ST-TAA reduces to equal-weight Monte-Carlo accumulation. We verified its curve coincides with an explicit reference accumulator to within 0.02 dB over the whole range. Our denoiser converges markedly faster at low frame counts because its spatially filtered denoised path carries the output before temporal history builds up, and then stays on the same Monte-Carlo convergence trajectory as plain accumulation.

## H Variance Gate Ablation

Table 10 ablates the variance gate on the garden navigation track: forcing the per-pixel composition weight g to either

![](images/eef4f2f1eb1bc1dce9569e8bfaef2a8dc5d274dc0f8cd0c8ba543ac99d157e75.jpg)

Figure 5: Static-view convergence on garden. PSNR versus the number of accumulated frames on a static camera, averaged over the 24 held-out views. The dashed line marks the deterministic vanilla 2DGS renderer, which renders each frame identically and independently of accumulation.
<table><tr><td>Variant</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td></tr><tr><td>Full model (learned g)</td><td>26.63</td><td>0.794</td><td>0.238</td></tr><tr><td>g ≡ 0 (accumulated path only)</td><td>25.14</td><td>0.751</td><td>0.298</td></tr><tr><td>g ≡ 1 (denoised path only)</td><td>25.95</td><td>0.752</td><td>0.247</td></tr></table>

Table 10: Variance-gate ablation on the garden navigation track (same protocol as the ablation table in the paper). g is the per-pixel weight of the denoised path in the final composition in the main text; forcing it to either path exclusively degrades all three metrics. The learned gate improves over the better fixed extreme by 0.68 dB PSNR.

path exclusively is strictly worse than the learned gate on all three metrics.

## I Per-Scene and Per-Track Results

Tables 11 and 12 report the per-scene PSNR underlying the main text’s static-quality averages, and Table 13 reports the per-track numbers underlying the main text’s navigation table. Table 14 reports the per-track breakdown of the oficial StochasticSplats 1 spp/4 spp rows. Table 15 reports the per-track breakdown of the main text’s zero-shot transfer experiment. The hand-recorded navigation camera tracks used for all track evaluations are provided in the code supplement under data/tracks/.

Denoiser stage. Table 16 lists the full Stage-4 configuration.

<table><tr><td rowspan="2">Method</td><td colspan="7">PSNR↑ per scene</td><td colspan="3">Average</td></tr><tr><td>bicycle</td><td>bonsai</td><td>counter</td><td>garden</td><td>kitchen</td><td>room</td><td>stump</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td></tr><tr><td>2DGS</td><td>24.71</td><td>32.12</td><td>28.82</td><td>26.73</td><td>32.49</td><td>31.53</td><td>29.74</td><td>29.45</td><td>0.875</td><td>0.199</td></tr><tr><td>Ref. (1024)</td><td>24.20</td><td>30.65</td><td>28.14</td><td>26.15</td><td>30.24</td><td>30.51</td><td>27.88</td><td>28.25</td><td>0.851</td><td>0.202</td></tr><tr><td>ST-TAA (128)</td><td>24.08</td><td>30.10</td><td>27.83</td><td>25.86</td><td>29.76</td><td>30.14</td><td>27.63</td><td>27.92</td><td>0.822</td><td>0.219</td></tr><tr><td>Ours (16)</td><td>23.58</td><td>28.80</td><td>26.90</td><td>24.69</td><td>28.40</td><td>29.34</td><td>26.54</td><td>26.89</td><td>0.741</td><td>0.330</td></tr><tr><td>Ours (98)</td><td>24.09</td><td>30.20</td><td>27.84</td><td>25.82</td><td>29.80</td><td>30.22</td><td>27.58</td><td>27.94</td><td>0.822</td><td>0.226</td></tr><tr><td>Ours (128)</td><td>24.13</td><td>30.32</td><td>27.91</td><td>25.90</td><td>29.91</td><td>30.30</td><td>27.65</td><td>28.02</td><td>0.829</td><td>0.217</td></tr><tr><td>Ours (zero-shot, 16)</td><td>23.66</td><td>28.96</td><td>26.99</td><td>24.83</td><td>28.56</td><td>29.33</td><td>26.67</td><td>27.00</td><td>0.746</td><td>0.332</td></tr><tr><td>Ours (zero-shot, 128)</td><td>24.14</td><td>30.33</td><td>27.92</td><td>25.92</td><td>29.93</td><td>30.29</td><td>27.66</td><td>28.03</td><td>0.829</td><td>0.217</td></tr></table>

Table 11: Static quality on the MipNeRF360 held-out test views. 2DGS is the deterministic tile renderer of 2D Gaussian Splatting (Huang et al.). ST-TAA is the baseline adapted from StochasticSplats (Kheradmand et al.): temporal accumulation over 128 static frames without a learned denoiser. Ref. (1024) is the Monte-Carlo accumulation reference (main text, Experimental Setup). Ours renders through the learned denoiser at 16, 98, or 128 accumulated frames; Ours (zero-shot) deploys the room checkpoint on every scene with no per-scene denoiser training (main text, Zero-Shot Transfer). Per-scene PSNR with dataset averages of all metrics; bold marks the better of the equal-time pair, ST-TAA (128) vs. Ours (98), in each column (both when equal after rounding).

<table><tr><td></td><td colspan="7">PSNR↑ per scene</td><td colspan="3">Average</td></tr><tr><td>Method</td><td>Barn Caterpillar Church Courthouse</td><td></td><td></td><td></td><td></td><td> Ignatius Meetingroom</td><td></td><td></td><td>Truck PSNR↑ SSIM↑ LPIPS↓</td><td></td></tr><tr><td>2DGS Ref. (1024)</td><td>22.82 22.39</td><td>20.33 19.84</td><td>21.52 19.87</td><td>21.73 21.17</td><td>19.78 19.11</td><td>22.68 21.42</td><td>21.65 21.04</td><td>21.50 20.69</td><td>0.700 0.675</td><td>0.379 0.388</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>19.05</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ST-TAA (128)</td><td>22.35 22.07</td><td>19.75 19.46</td><td>19.75 19.42</td><td>21.11 20.83</td><td>18.79</td><td>21.30 21.01</td><td>20.96</td><td>20.61</td><td>0.635</td><td>0.431</td></tr><tr><td>Ours (16)</td><td>22.32</td><td>19.74</td><td>19.75</td><td>21.09</td><td>19.03</td><td>21.32</td><td>20.72 20.98</td><td>20.33</td><td>0.548</td><td>0.496</td></tr><tr><td>Ours (98) Ours (128)</td><td>22.34</td><td>19.77</td><td>19.78</td><td>21.11</td><td>19.05</td><td>21.34</td><td></td><td>20.60 20.63</td><td>0.631</td><td>0.436</td></tr><tr><td></td><td>22.04</td><td>19.40</td><td>19.37</td><td>20.80</td><td>18.74</td><td>20.97</td><td>21.00</td><td></td><td>0.640</td><td>0.428</td></tr><tr><td>Ours (zero-shot, 16)</td><td></td><td></td><td>19.77</td><td>21.10</td><td>19.05</td><td></td><td>20.71</td><td>20.29</td><td>0.539</td><td>0.501</td></tr><tr><td>Ours (zero-shot, 128)</td><td>22.34</td><td>19.76</td><td></td><td></td><td></td><td>21.34</td><td>20.99</td><td>20.62</td><td>0.639</td><td>0.428</td></tr></table>

Table 12: Static quality on TanksAndTemples. All methods are trained on the full capture of each scene; we evaluate every 8th frame. Rows follow Table 11; bold marks the better of the equal-time pair, ST-TAA (128) vs. Ours (98), in each column (both when equal after rounding). Ours (zero-shot) deploys the room checkpoint with no per-scene denoiser training (main text, Zero-Shot Transfer).

<table><tr><td></td><td colspan="5">PSNR↑/LPIPS↓</td><td colspan="5">SSIM↑/ CVVDP↑</td></tr><tr><td>Track</td><td>1 spp</td><td>ST-TAA</td><td>SVGF</td><td>Ref.</td><td>Ours</td><td>1 spp</td><td>ST-TAA</td><td>SVGF</td><td>Ref.</td><td>Ours</td></tr><tr><td>bicycle (161 frames)</td><td>16.81</td><td>19.82</td><td>22.05</td><td>29.87</td><td>26.39</td><td>0.289</td><td>0.407</td><td>0.709</td><td>0.925</td><td>0.807</td></tr><tr><td>bonsai (166 frames)</td><td>0.639</td><td>0.588</td><td>0.346</td><td>0.088</td><td>0.284</td><td>4.28</td><td>5.29</td><td>5.23</td><td>8.34</td><td>6.84</td></tr><tr><td></td><td>19.28 0.690</td><td>23.00 0.629</td><td>28.17 0.241</td><td>34.54 0.060</td><td>31.98 0.253</td><td>0.313 5.57</td><td>0.486 6.66</td><td>0.868 7.23</td><td>0.967 8.96</td><td>0.914 8.25</td></tr><tr><td>counter (126 frames)</td><td>18.05</td><td>20.97</td><td>23.98</td><td>33.76</td><td>30.35</td><td>0.257</td><td>0.367</td><td>0.766</td><td>0.956</td><td>0.880</td></tr><tr><td></td><td>0.717</td><td>0.677</td><td>0.339</td><td>0.099</td><td>0.271</td><td>4.52</td><td>5.42</td><td>5.71</td><td>8.82</td><td>7.50</td></tr><tr><td>garden (157 frames)</td><td>16.08</td><td>19.61</td><td>23.37</td><td>31.34</td><td>26.63</td><td>0.244</td><td>0.394</td><td>0.720</td><td>0.944</td><td>0.794</td></tr><tr><td></td><td>0.606</td><td>0.531</td><td>0.308</td><td>0.067</td><td>0.238</td><td>4.60</td><td>5.81</td><td>6.11</td><td>8.80</td><td>7.15</td></tr><tr><td>kitchen (181 frames)</td><td>19.24</td><td>22.33</td><td>27.98</td><td>33.67</td><td>30.95</td><td>0.306</td><td>0.450</td><td>0.854</td><td>0.955</td><td>0.906</td></tr><tr><td></td><td>0.643</td><td>0.579</td><td>0.240</td><td>0.066</td><td>0.192</td><td>5.69</td><td>6.58</td><td>7.40</td><td>8.93</td><td>8.07</td></tr><tr><td>room (185 frames)</td><td>20.79</td><td>24.00</td><td>28.82</td><td>34.70</td><td>33.64</td><td>0.334</td><td>0.476</td><td>0.884</td><td>0.968</td><td>0.950</td></tr><tr><td></td><td>0.697</td><td>0.647</td><td>0.232</td><td>0.089</td><td>0.171</td><td>5.46</td><td>6.52</td><td>7.24</td><td>8.80</td><td>8.29</td></tr><tr><td>stump (179 frames)</td><td>17.54</td><td>20.61</td><td>24.59</td><td>32.89</td><td>28.66</td><td>0.229</td><td>0.347</td><td>0.692</td><td>0.936</td><td>0.815</td></tr><tr><td></td><td>0.705</td><td>0.656</td><td>0.376</td><td>0.093</td><td>0.300</td><td>4.48</td><td>5.37</td><td>5.11</td><td>8.46</td><td>6.89</td></tr><tr><td>Average</td><td>18.26</td><td>21.48</td><td>25.57</td><td>32.96</td><td>29.80</td><td>0.282</td><td>0.418</td><td>0.785</td><td>0.950</td><td>0.867</td></tr><tr><td></td><td>0.671</td><td>0.615</td><td>0.297</td><td>0.080</td><td>0.244</td><td>4.94</td><td>5.95</td><td>6.29</td><td>8.73</td><td>7.57</td></tr></table>

Table 13: Track-averaged quality under free camera navigation on MipNeRF360, using the vanilla 2DGS tile renderer as the reference. Each track occupies two rows: the first reports PSNR (left) and SSIM (right), the second LPIPS (left) and CVVDP (right). Raw 1 spp is the single-sample stochastic output; ST-TAA applies generic temporal accumulation without learned trust or spatial filtering; SVGF is the adapted variance-guided filtering baseline (Sec. E); Ref. (1024) is the per-frame Monte-Carlo ceiling (main text, Experimental Setup), not a practical method. The generic TAA warp fails to accumulate reliably under the heavy 1 spp noise, while our learned trust gating keeps the temporal history usable.

<table><tr><td></td><td colspan="2">PSNR↑</td><td colspan="2">SSIM↑</td><td colspan="2">LPIPS↓</td><td colspan="2">CVVDP↑</td></tr><tr><td>Track</td><td>SS 1 spp</td><td>SS 4 spp</td><td>SS 1 spp</td><td>SS 4 spp</td><td>SS 1 spp</td><td>SS 4 spp</td><td>SS 1 spp</td><td>SS 4 spp</td></tr><tr><td>bicycle</td><td>15.70</td><td>20.08</td><td>0.275</td><td>0.483</td><td>0.634</td><td>0.523</td><td>3.65</td><td>6.18</td></tr><tr><td>bonsai</td><td>18.84</td><td>23.68</td><td>0.314</td><td>0.557</td><td>0.689</td><td>0.580</td><td>5.21</td><td>7.36</td></tr><tr><td>counter</td><td>18.13</td><td>22.60</td><td>0.273</td><td>0.475</td><td>0.702</td><td>0.597</td><td>4.31</td><td>6.47</td></tr><tr><td>garden</td><td>15.90</td><td>20.78</td><td>0.273</td><td>0.528</td><td>0.590</td><td>0.449</td><td>4.54</td><td>6.84</td></tr><tr><td>kitchen</td><td>18.83</td><td>24.48</td><td>0.313</td><td>0.594</td><td>0.641</td><td>0.488</td><td>5.62</td><td>7.71</td></tr><tr><td>room</td><td>21.05</td><td>25.50</td><td>0.365</td><td>0.605</td><td>0.668</td><td>0.560</td><td>5.35</td><td>7.21</td></tr><tr><td>stump</td><td>16.59</td><td>20.64</td><td>0.229</td><td>0.441</td><td>0.690</td><td>0.581</td><td>3.98</td><td>6.30</td></tr><tr><td>Average</td><td>17.86</td><td>22.54</td><td>0.292</td><td>0.526</td><td>0.659</td><td>0.540</td><td>4.67</td><td>6.87</td></tr></table>

Table 14: Per-track breakdown of the oficial StochasticSplats rows of the main text’s navigation table: the oficial model on its own finetuned 3DGS scenes, measured against its own 1024 spp accumulation (no ground truth exists on free camera tracks). SS 4 spp renders through the oficial viewer’s multi-sample path with a small fix so that each MSAA sample draws an independent stochastic sample (as released, all samples share one draw). All metrics are computed on every 10th track frame (the stride-10 dumps underlying the corresponding main-table rows; ColorVideoVDP uses the same stride-10 videos).

<table><tr><td></td><td colspan="2">PSNR↑</td><td colspan="2">SSIM↑</td><td colspan="2">LPIPS↓</td><td colspan="2">CVVDP↑</td></tr><tr><td>Track</td><td>Ours</td><td>zero-shot</td><td>Ours</td><td>zero-shot</td><td>Ours</td><td>zero-shot</td><td>Ours</td><td>zero-shot</td></tr><tr><td>bicycle</td><td>26.39</td><td>25.86</td><td>0.807</td><td>0.817</td><td>0.284</td><td>0.255</td><td>6.84</td><td>6.56</td></tr><tr><td>bonsai</td><td>31.98</td><td>30.91</td><td>0.914</td><td>0.914</td><td>0.253</td><td>0.205</td><td>8.25</td><td>7.83</td></tr><tr><td>counter</td><td>30.35</td><td>29.68</td><td>0.880</td><td>0.891</td><td>0.271</td><td>0.248</td><td>7.50</td><td>7.32</td></tr><tr><td>garden</td><td>26.63</td><td>26.82</td><td>0.794</td><td>0.808</td><td>0.238</td><td>0.235</td><td>7.15</td><td>7.12</td></tr><tr><td>kitchen</td><td>30.95</td><td>30.61</td><td>0.906</td><td>0.914</td><td>0.192</td><td>0.162</td><td>8.07</td><td>8.07</td></tr><tr><td>room</td><td>33.64</td><td>33.59</td><td>0.950</td><td>0.948</td><td>0.171</td><td>0.179</td><td>8.29</td><td>8.25</td></tr><tr><td>stump</td><td>28.66</td><td>27.91</td><td>0.815</td><td>0.817</td><td>0.300</td><td>0.275</td><td>6.89</td><td>6.51</td></tr><tr><td>Average</td><td>29.80</td><td>29.34</td><td>0.867</td><td>0.873</td><td>0.244</td><td>0.223</td><td>7.57</td><td>7.38</td></tr></table>

Table 15: Per-track breakdown of the zero-shot transfer experiment (main text, Zero-Shot Transfer): the room checkpoint deployed verbatim on every scene, with the neural-view payload neutralised (the trust feature $V _ { 2 }$ is set to zero and the spatial filter falls back to a fixed isotropic kernel, $x = y = 2$ and $\rho = 0$ in the covariance parameterisation of the main text). Ours is the per-scene fine-tuned configuration of the main navigation table. The zero-shot variant trades a small PSNR/CVVDP margin for slightly better SSIM/LPIPS.

<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>finetune steps</td><td>15k (4k for fast transfer)</td></tr><tr><td>loss</td><td>L1 + 0.05 SSIM + 0.01 LPIPS</td></tr><tr><td>auxiliary vanilla-colour loss (accumulated path)</td><td>0.2</td></tr><tr><td>temporal step ratio</td><td>0.7</td></tr><tr><td>stochastic suffix length K</td><td>8</td></tr><tr><td>chain-length curriculum</td><td>2 → 128</td></tr><tr><td>max perturbation angle / position</td><td>3.0° / 0.003× scene radius</td></tr><tr><td>drift / lateral-dolly chain fractions</td><td>0.25 / 0.25</td></tr><tr><td>lateral dolly total length</td><td>0.5-1.1 m</td></tr><tr><td>history caps  $n _ { \mathrm { m a x } }$  (denoised / accumulated)</td><td>24/128</td></tr><tr><td>near-static trust regularisers  $( t _ { s } / t _ { h } )$ </td><td>0.01 / 0.01 (0.005 / 0.005 in fast transfer)</td></tr><tr><td>disocclusion mask: depth threshold  $\tau _ { d }$ </td><td>0.02 (relative)</td></tr><tr><td>disocclusion mask: min opacity / colour tolerance</td><td>0.6 / 0.05</td></tr><tr><td>disocclusion trust penalty weight filter axis-symmetry regulariser</td><td>0.03 (fast transfer adds 0.05 on both paths)</td></tr><tr><td>gate warmup (forced denoised output)</td><td>0.01</td></tr><tr><td></td><td>2000 steps</td></tr><tr><td>gate scalar minimum scale</td><td> $1 0 ^ { - 4 }$ </td></tr><tr><td>gate time constant τ</td><td>8 frames</td></tr></table>

Table 16: Stage-4 training configuration.