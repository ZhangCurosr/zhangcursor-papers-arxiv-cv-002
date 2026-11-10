# Real-Time Volumetric Light Transport Inference without a Path

L. Leonard<sup>1</sup> and R. Westermann<sup>1</sup>

<sup>1</sup>Technical University of Munich, Germany

![](images/ab380418d9a585e466af30f769e886ab89b8d717d16ed69b4f3d7d725f4975bd.jpg)

Figure 1: For clouds and environments it has never seen during training, our attention-based neural model reconstructs full volumetric global illumination from inexpensive primary-ray features and environment samples in real-time on an NVIDIA GeForce RTX 5090. The computation ofpath-traced radiance samples is entirely avoided.

## Abstract

We present a real-time method for reconstructing fully integrated pixel radiance in rendered volumetric media under direct and ambient environmental illumination, using compact, low-cost scene features and statistics. Unlike Monte Carlo denoising pipelines for volumetric rendering, which depend on computationally expensive path-traced radiance estimates, our approach directly infers the final light-transport solution from inexpensive primary-ray and environment sample evaluations. We show that these features form a near-sufficient representation of the scene, enabling accurate reconstruction at substantially lower computational cost. Using an attention-augmented image-to-image network, we demonstrate that complete pixel radiance can be inferred directly from these auxiliary features, without requiring the original radiance signal. Operating in image space also provides a more stable computational cost as the camera zooms into or out ofthe volume, with substantially less dependence on volume density than approaches based on explicit multiple-scattering simulation. We demonstrate real-time performance and high reconstruction accuracy in challenging volumetric cloud scenarios

• Computing methodologies → Rendering; Neural networks; Ray tracing;

## 1. Introduction

Monte Carlo (MC) path tracing achieves photorealistic rendering by simulating the full light transport pipeline, including emission, scattering, and absorption along importance-sampled light paths [Kaj86, Coo86, AK90, NGHJ18, PJH23]. For surface rendering, light paths terminate after a few bounces, yet each pixel still requires hundreds to thousands of samples to converge to a noisefree result. For volumetric media, the problem is substantially more severe: light scatters at every point along a ray, turning each sample into a recursive integration problem. Rendering a single frame of participating media such as clouds, smoke, or fog often requires seconds to minutes even on high-end GPUs, making real-time applications prohibitive.

To mitigate the cost of path tracing, many modern rendering systems rely on MC denoising. For surface rendering, denoisers leverage auxiliary features—often called G-buffers in surface graphics—like albedo, normals, depth, and variance buffers to filter sparse path samples (given as samples per pixel, or spp) into smooth outputs [BVM<sup>∗</sup>17, VRM<sup>∗</sup>18, SKW<sup>∗</sup>17], or to use them as input for neural networks that predict noise-free images directly [CKS<sup>∗</sup>17, IMF<sup>∗</sup>21, VVF24].

These MC denoisers must evaluate at least a few full light transport paths to accurately transform the noisy rendering into a highquality image. This becomes especially costly for volumetric media, where simulating light transport is computationally expensive. Moreover, paths in volumetric media exhibit complex, viewdependent correlations across samples that surface-based auxiliary features cannot exploit, rendering such features less predictive (Fig. 2 OIDN). Due to the nature of participating media, the noise distribution itself varies spatially and spectrally, further complicating inference.

![](images/a6e88ee9ef741d450efc93f2ec1554b0e1a9b2c31a0e19f0ef63536cd69d18ab.jpg)  
Figure 2: Comparison under realistic ambient environment lighting. A density-conditioned transport estimation [HYL<sup>∗</sup>23] (MRPNN) uniformly samples the environment and produces noise even with 1024 samples; Intel Open Image Denoise [Áfr26] (OIDN) uses only rendered pixel color and smooths out fine details; a specialized volumetric denoiser [HHCM21] (Denoiser) better preserves appearance but suffers from stochastic noise; a variant of the volumetric denoiser using our deterministic features (DenoiserX) better preserves fine details. All denoisers use 4 radiance samples per pixel. Ours, without any path samples, achieves superior detail reconstruction at lower computational cost. All methods run on an NVIDIA GeForce RTX 3090, as MRPNN is incompatible with newer GPU architectures.

One approach to addressing these challenges is MC denoising guided by auxiliary features specifically designed to represent volumetric light transport characteristics, such as transmittance and first-scatter event estimates [HMES20, HHCM21]. While effective in principle, such methods still require path-traced radiance samples as input: their cost grows with medium density, and the noise of these samples propagates into the reconstruction, leading to blurring and instability across realizations (Fig. 2 Denoiser).

In another research line, radiance-predicting neural networks replace path tracing for atmospheric clouds, albeit at considerably higher computational cost than MC denoisers [KMM 17] or requiring progressive inference updates to simulate full environment lighting [HYL 23] (Fig. 2).

Both lines of work still depend on simulating light transport through the volume at inference time, either through path samples or through per-sample network queries. This raises a simple question: are path samples needed at all? Prior work on surfaces shows that localized illumination effects can be inferred from auxiliary buffers alone [NAM<sup>∗</sup>17], and Deep Scattering shows that networks can learn volumetric transport [KMM<sup>∗</sup>17]. We build upon both insights: we keep the feature-based design of volumetric denoisers, but drop the radiance input entirely and regress the final radiance from features computed along primary rays.

We show that optical depth, multi-scale interaction depths, transmittance towards the light, environment illumination at the expected first interaction, and local medium properties form a nearsufficient representation of the transport solution. We formalize this as two testable assumptions, sufficiency and noise invariance, and learn the mapping from features to radiance with an attentionaugmented image-to-image network. Since the features are deterministic except for low-variance light-visibility estimates, the output is stable across realizations without requiring accumulation over frames.

We demonstrate real-time performance and high reconstruction accuracy in challenging volumetric cloud scenes (Fig. 1 and Fig. 2 Ours), where state-of-the-art MC denoisers struggle to maintain both stability and fine detail. By moving from sample denoising to primary-ray feature regression, our method offers a new pathway toward practical real-time volumetric light transport.

In summary, our contributions are:

• The formulation of volumetric rendering as a deterministic decoding from inexpensive transport features, replacing path sampling at inference time.

• A compact set of primary-ray features for participating media, which we show to be near-sufficient for inferring full multiple scattering in clouds.

• A real-time image-to-image network with local self-attention, which infers directional and environmental scattering separately and scales to high resolutions.

• A real-time cloud renderer simulating full radiance transport in unseen clouds and environments, showing high accuracy, low variance across realizations, and stable cost even for high densities.

## 2. Related Work

MC denoising for surfaces. Early denoisers handle the spatially varying noise of MC renderings with nonlinear image filters, such as bilateral and edge-avoiding filters [Lee83, RW94, DSHL10] or non-local means [RKZ12], often guided by G-buffers to preserve edges. Spatiotemporal variance-guided filtering [SKW 17] additionally reuses samples from past frames to improve temporal continuity. With the advent of deep learning, denoising has shifted from handcrafted filters to data-driven prediction [HY21, YGH<sup>∗</sup>25]. Learning the mapping from noisy features to filter parameters [KBS15] gave rise to two paradigms: radiance-prediction networks regress clean pixel values directly [CKS<sup>∗</sup>17, XZW<sup>∗</sup>19, LXS20], whereas kernel-prediction networks estimate spatially varying filter kernels that keep the output consistent with the input colors $[ \mathrm { B V M ^ { * } } 1 7 , \mathrm { V R M ^ { * } } \bar { 1 } 8 , \mathrm { I M F ^ { * } } 2 1 ] .$ . Sampling and temporal denoising networks can further be trained jointly $[ \mathrm { H M S } ^ { \ast } 2 0 ]$ These methods achieve near-reference quality by treating rendering as conditional image synthesis, but they are designed for surfaces and all require path-traced radiance as input.

Volumetric denoising and sample reuse. Volumetric path tracing converges far more slowly than its surface counterpart, since each radiance sample involves many scattering events along the ray. At low sample counts this yields severely noisy images, while auxiliary buffers that are effective for surfaces, such as depth or normals, become noisy or ill-defined in media $[ \mathrm { Y G H } ^ { * } 2 5$ , DFNB25]. Hofmann et al. [HMES20] were the first to extend neural MC denoising to volumetric path tracing of medical data, introducing a set of auxiliary features designed specifically for volumes and increasing the effective sample count through spatial and temporal filtering; this was later extended to sparse volumes [HHCM21]. Our features build on this idea. Subsequent work decomposes the image into a surface and a volume layer with dedicated denoisers and a network that learns to blend both [HHM23]. Xu et al. $[ \mathrm { X X Z ^ { * } } 2 5 ]$ decouple radiance samples according to their noise level and fuse them with auxiliary features in a lightweight network, reducing the bias of earlier decoupling strategies. In contrast, Iglesias-Guitian et al. [IGMM22] explicitly avoid auxiliary features and exploit temporal coherence instead, reprojecting linear models across frames and combining past and current estimates by weighted recursive least squares. Orthogonally, the quality of the input samples can be improved. Zhou et al. [ZZL<sup>∗</sup>17] fit Gaussian mixtures to single-scattered radiance to guide visibility-aware sampling in participating media, and spatiotemporal reservoir resampling $\mathrm { [ B W P ^ { * } 2 0 ] }$ has been extended to volumetric path space by ReSTIR-Volume [LWY21]. All these approaches still trace paths through the medium; our method requires none at inference time.

Radiance-free neural rendering. Concurrently with learned denoisers, Deep Shading $[ \mathrm { N A M } ^ { * } 1 7 ]$ showed that CNNs can synthesize the final appearance directly from screen-space G-buffers, circumventing hand-crafted shaders and requiring no radiance input at all. Subsequent work explicitly frames G-buffers as the input representation for learning light transport, positioning Gbuffer-based neural rendering as a modern alternative to precomputed radiance transfer [RBRD22]. G-buffers have since been combined with reflective shadow maps to reconstruct complex indirect lighting $[ \mathrm { U K K } ^ { * } 2 6 ]$ , used to synthesize global illumination including transparent objects [ZSS26], and shown to remain valuable even when the primary transport data comes from other sources $[ \mathrm { H S G } ^ { * } 2 2 ]$ . Beyond screen space, transformer-based models infer global illumination directly from a 3D scene description: RenderFormer $[ \mathrm { Z D P ^ { * } } 2 5 ]$ maps triangle tokens with their reflectance properties to images with full global illumination, and Xu et al. $[ \mathrm { X T W } ^ { * } 2 6 ]$ learn a generalizable light transport embedding from scene point clouds, queried through cross-attention to predict diffuse global illumination in unseen scenes. These approaches recover local and global illumination effects on surfaces, but none addresses the direct inference of global illumination in volumetric media. We transfer their central idea, radiance-free inference from auxiliary buffers, to participating media.

Learned volumetric light transport. Instead of denoising rendered images, neural networks can approximate components of volumetric light transport itself. Deep Scattering $\left[ \mathrm { K M M ^ { * } } 1 7 \right]$ predicts the radiance caused by multiple scattering in clouds from a hierarchical 3D descriptor of the density around each shading point, oriented towards the light source. Learned statistical path models replace long stochastic scattering chains with a surrogate based on a more structured probabilistic model of the local geometry [LHW21]. MRPNN [HYL<sup>∗</sup>23] brings Deep Scattering to real time with a lightweight multi-feature fusion network that combines precomputed transmittance fields with a frequency-sensitive density stencil. For environment illumination, however, it requires progressive sampling over light directions, repeating the full ray accumulation for each direction. Rather than evaluating a network per sample, Liang et al. $[ \mathrm { L Y G } ^ { * } 2 6 ]$ bake a compact low-rank latent approximation of the scattering operator, gaining performance and avoiding over-illumination artifacts, but at the cost of a baking step per volume. Similarly, Relightable Neural Assets [MLSH24] encode all shading and scattering of an asset in a feature grid decoded by a small MLP, an approach adapted to relight volumetric clouds under environment illumination within a rasterization pipeline [ODG26]; such representations are tied to the specific volume they were trained on. In contrast, our network operates in image space on features computed on the ${ \mathrm { f l y } } ,$ requires neither per-point 3D descriptors nor precomputed volumetric fields or pervolume training, and generalizes to unseen volumes.

## 3. Method

Fig. 3 gives an overview of our method. We first formulate rendering as a deterministic decoding of image features (Secs. 3.1–3.2), then define a set of inexpensive primary-ray and environment features for optically thick media (Sec. 3.3), and finally describe how the decoder is learned and structured to separate directional and environmental light (Secs. 3.4–3.5).

## 3.1. Physically-Based Volume Rendering

Let θ denote a scene parameterization including geometry, materials, illumination, and participating media. We define a physicallybased rendering operator R such that $I = R ( { \boldsymbol { \theta } } )$ corresponds to the solution of the light transport equation. Evaluating R is computationally expensive due to high-dimensional path integrals required for global illumination with multiple scattering.

Light transport in participating media is governed by the volume rendering equation, which accounts for attenuation, absorption, emission, and multiple scattering along a ray. Let $x _ { t } = x _ { 0 } - t { \boldsymbol { \omega } }$ denote a point along direction −ω from an origin x<sub>0</sub>. The outgoing radiance is given by

$$
L ( x _ { 0 } , \mathbf { \omega } ) = \int _ { 0 } ^ { d } { T ( x _ { 0 }  x _ { t } ) \sigma _ { t } ( x _ { t } ) [ \cdots ] d t } + T ( x _ { 0 }  x _ { d } ) L ( x _ { d } , \mathbf { \omega } ) ,
$$

where $\sigma _ { t }$ is the extinction coefficient and

$$
T ( x _ { a }  x _ { b } ) = \exp ( - \int _ { x _ { a } } ^ { x _ { b } } \sigma _ { t } ( x ) d x )
$$

denotes transmittance between two points in the medium.

![](images/9238354eeb67969c8f32bcc17150fe3f1618dbfc1a7420f6d51f67356ff51968.jpg)  
Figure 3: Feature extraction and light-transport inference. Primary rays are converted into compactfeatures describing optical depth, multiscale density crossings, transmittance toward directional and environmental illumination, and local scattering properties. Thesefeatures are organized into light-transport channels and processed by specialized neural models to infer the corresponding scattered-light components, which are subsequently combined into thefinal image.

The term $\left[ \cdots \right]$ represents the local source contribution,

$$
\begin{array} { r } { \Phi ( x _ { t } ) L _ { s } ( x _ { t } , \mathbf { \omega } ) + \left( 1 - \Phi ( x _ { t } ) \right) L _ { e } ( x _ { t } , \mathbf { \omega } ) , } \end{array}
$$

where $\boldsymbol { \Phi }$ is the single-scattering albedo, $L _ { e }$ denotes emitted radiance, and $L _ { s }$ is the in-scattered radiance. The latter integrates incident illumination over the sphere of directions:

$$
L _ { s } ( x _ { t } , \boldsymbol { \omega } ) = \int _ { S ^ { 2 } } \rho ( x _ { t } , \boldsymbol { \omega } , \boldsymbol { \omega } _ { i } ) L ( x _ { t } , \boldsymbol { \omega } _ { i } ) d \boldsymbol { \omega } _ { i } ,
$$

where $\rho$ is the phase function describing angular scattering. In this paper, media with emissive particles are not considered, $\mathrm { i . e . , }$ $L _ { e } ( x , \omega ) = 0$ . We employ the Henyey–Greenstein phase function, parameterized by an anisotropy coefficient $^ { g , }$ which provides an efficient approximation of highly forward-scattering volumetric transport.

## 3.2. Problem Formulation

We introduce a set of auxiliary forward operators $\{ F _ { k } \} _ { k = 1 } ^ { n }$ producing intermediate images

$$
\hat { \mathbf { F } } = \mathbf { F } ( \boldsymbol { \Theta } ; \boldsymbol { \xi } ) = \big ( F _ { 1 } ( \boldsymbol { \Theta } ; \boldsymbol { \xi } ) , \ldots , F _ { n } ( \boldsymbol { \Theta } ; \boldsymbol { \xi } ) \big ) , \qquad \boldsymbol { \xi } \sim p ( \boldsymbol { \xi } ) ,
$$

where $\xi$ denotes the random numbers consumed by the operators, drawn independently of the scene θ. Unlike $R ,$ each $F _ { k }$ is a computationally efficient transport projection derived from θ, encoding structured quantities such as transmittance, scattering integrals, or low-order illumination terms rather than approximations of full radiance.

We make the following assumptions:

(A1) Sufficiency. The rendered image depends on the scene only through the distribution of its features:

$$
\mathbf { F } ( \boldsymbol { \Theta } ; \boldsymbol { \xi } ) \overset { d } { = } \mathbf { F } ( \boldsymbol { \Theta } ^ { \prime } ; \boldsymbol { \xi } ) \Rightarrow R ( \boldsymbol { \Theta } ) = R ( \boldsymbol { \Theta } ^ { \prime } ) ,
$$

where $\underline { { \underline { { d } } } }$ denotes equality in distribution with respect to $\xi \sim p ( \xi )$ Hence, the scene parameters θ are not required to infer the image. Notably, (A1) does not require θ itself to be recoverable from the features: different scenes, e.g., different density fields, may share the same feature distribution, as long as they also share the same rendered image.

(A2) Noise-invariant decoding. There exists a deterministic mapping S that is insensitive to the stochastic estimation of the features, i.e., all realizations of the features of a scene decode to the same image:

$$
S \big ( \mathbf { F } ( \boldsymbol { \Theta } ; \boldsymbol { \xi } ) \big ) = S \big ( \mathbf { F } ( \boldsymbol { \Theta } ; \boldsymbol { \xi } ^ { \prime } ) \big ) = R ( \boldsymbol { \Theta } ) \qquad \mathrm { a l m o s t ~ s u r e l y . }
$$

Under (A1) and $( \mathbf { A } 2 ) ,$ , rendering reduces to a deterministic mapping from noisy auxiliary features to radiance. Both assumptions can be tested separately: violations of (A1) appear as a systematic bias that persists even when the decoder observes arbitrarily many realizations $\mathbf { F } ( \boldsymbol { \Theta } ; \boldsymbol { \xi } _ { 1 } ) , \mathbf { F } ( \boldsymbol { \Theta } ; \boldsymbol { \xi } _ { 2 } ) , \ldots$ . of the features, i.e., when their distribution is fully known, while violations of (A2) appear as a non-zero variance $\operatorname { V a r } _ { \boldsymbol { \xi } } \big [ S ( \mathbf { F } ( \boldsymbol { \Theta } ; \boldsymbol { \xi } ) ) \big ]$ across single realizations.

This formulation generalizes MC denoising, where a single operator $F _ { \mathrm { M C } } ( { \boldsymbol { \theta } } ; { \boldsymbol { \xi } } )$ provides an unbiased radiance estimate. Its distribution determines $R ( \Theta ) = \mathbb { E } _ { \xi } [ F _ { \mathrm { M C } } ( \Theta ; \xi ) ]$ , so (A1) holds trivially, and the entire difficulty lies in (A2). Our operators shift this balance: most features are deterministic and the remaining ones have low variance, making (A2) easy to satisfy, while none of them is a degraded version of I. Instead, each $F _ { k }$ provides a complementary, physically meaningful projection of light transport, and S performs radiance reconstruction from heterogeneous transport measurements rather than denoising of a single stochastic estimator.

## 3.3. Features

We now define the operators $\left\{ F _ { k } \right\}$ for volume rendering in optically thick media. All of them are evaluated along primary rays and describe attenuation, visibility, illumination, and local scattering properties (Fig. 3 Left).

Optical depth $F _ { \tau } .$ . We consider the accumulated extinction for each ray

$$
\tau ( d ) = \int _ { 0 } ^ { d } \sigma _ { t } ( x _ { t } ) d t ,
$$

which governs transmittance, expected free-flight distance, and scattering density. Rather than using transmittance directly, we employ a normalized encoding of a medium with reduced extinction

$$
{ \hat { \mathsf { \tau } } } ( d ) = 1 - { \frac { 1 } { 1 + 0 . 5 \tau ( d ) } } .
$$

This transformation suppresses the exponential saturation of $T$ in high-opacity regions, yielding a smoother signal that preserves relative variations in dense media and improves numerical conditioning during reconstruction.

Interaction depth $F _ { d } .$ . We estimate the depth of the first scattering event, conditioned on an interaction occurring along the ray. Let $\tau = \tau ( d )$ , then

$$
\tau _ { c } = 1 - \frac { \tau e ^ { - \tau } } { 1 - e ^ { - \tau } }
$$

is the expected optical depth of the first interaction, given that it occurs before $d .$ We define the interaction depth as the minimum distance $t _ { 1 }$ at which this optical depth is reached,

$$
\tau ( t _ { 1 } ) = \tau _ { c } .
$$

Unlike the expected distance, which may fall into empty space between separated structures, $t _ { 1 }$ always lies inside the medium. It removes the noise and discontinuities of the free-flight estimator and is always defined for $\tau > 0$

Crossing layers $F _ { c } .$ To capture volumetric structure across scales, we compute multiple interaction-depth fields using progressively reduced extinction levels. Each layer represents the first conditioned interaction in a progressively more transparent version of the medium, producing a continuous multi-scale profile of volumetric occupancy. Instead of using the depth sequence, we use the distance between two consecutive crossings. This value remains more stable to camera variations.

Transmittancefor the next-event estimation $F _ { n } .$ . For the first interaction point $x _ { t _ { 1 } }$ , we estimate the transmittance towards a directional light source $T ( x _ { t _ { 1 } }  x _ { l } )$ .

Environment illumination $F _ { e } .$ . We additionally compute incident illumination from the environment map convolved through transmittance and phase function:

$$
\int _ { S ^ { 2 } } T ( x _ { t _ { 1 } }  x _ { i } ) \rho ( x _ { t _ { 1 } } , \boldsymbol { \omega } , \boldsymbol { \omega } _ { i } ) L ( x _ { i } , \boldsymbol { \omega } _ { i } ) d \boldsymbol { \omega } _ { i } .
$$

We estimate this integral using a low-sample-count Monte Carlo estimator, with 8 samples as a practical trade-off between computational cost and stability; this number can be adjusted as needed (see Section 4).

Medium properties $F _ { m } .$ . At the first interaction point, we extract local scattering parameters including the Henyey–Greenstein anisotropy parameter g and the single-scattering albedo $\boldsymbol { \Phi } ( \boldsymbol { x } _ { t _ { 1 } } )$ .

Together, these auxiliary quantities define a structured representation of the medium that captures partial yet informative constraints on global light transport.

## 3.4. Learning the Decoder

Assumptions (A1) and (A2) postulate the existence of the decoder S, but provide no closed form for it. We therefore approximate S by

a parameterized model

$$
S _ { \Psi } : \mathbb { R } ^ { H \times W \times n } \to \mathbb { R } ^ { H \times W \times 3 } ,
$$

an image-to-image neural network with weights $\Psi$ that maps the n feature images of resolution $H \times W$ to the rendered image. The parameters are obtained by minimizing the expected reconstruction error over both the distribution of scenes and the stochastic source of the features:

$$
\Psi ^ { \star } = \arg \operatorname* { m i n } _ { \Psi } \mathbb { E } _ { \theta \sim p ( \theta ) } \mathbb { E } _ { \xi \sim p ( \xi ) } \Big [ \ell \big ( S _ { \Psi } ( \mathbf { F } ( \theta ; \xi ) ) , R ( \theta ) \big ) \Big ] ,
$$

where $\ell$ is a reconstruction loss. In practice, the outer expectation is approximated by a finite set of training scenes $\theta _ { i }$ with converged reference renderings $R ( \Theta _ { i } )$ , and the inner expectation by drawing several realizations $\xi$ for each scene during training.

Both expectations relate directly to our assumptions. The inner expectation pairs all realizations $\dot { \mathbf { F } } ( \boldsymbol { \theta } ; \boldsymbol { \xi } )$ of a scene with the same target $R ( { \boldsymbol { \theta } } )$ , and thus rewards outputs that are invariant to $\xi ,$ as required by (A2). The outer expectation restricts the sufficiency assumption (A1) to the scene distribution $p ( \Theta ) \colon S _ { \Psi }$ only needs to be accurate for plausible scenes, and may exploit regularities of $p ( \boldsymbol { \Theta } )$ e.g., typical cloud structures, to resolve ambiguities left open by the features.

## 3.5. Structured Decoder

The formulation above holds for any decoder. In practice, we exploit exact properties of the rendering operator R to reduce the complexity of the learning problem. Since these properties are satisfied by R itself, the true decoder S shares them, and imposing them on $S _ { \Psi }$ restricts the hypothesis space without excluding the solution.

Linearity in the illumination. We split the distant illumination into a smooth environment map $L _ { \mathrm { e n v } }$ and a dominant directional light ${ \mathfrak { o } } _ { l }$ with intensities e and l. Since light transport is linear in the incident radiance,

$$
I = e \left( T B + I _ { e } \right) + l I _ { l } ,
$$

where $T B$ is the environment transmitted along the primary ray, and $I e , I _ { l }$ are the radiance scattered towards the camera for a normalized environment and a unit directional light. T and $B$ are known from the primary-ray traversal, so only $I _ { e }$ and $I _ { l }$ need to be inferred. We learn them with two decoders, $S _ { e }$ and $S _ { l } ,$ each receiving the shared primary-ray features and the illumination feature of its component $( F _ { e }$ resp. $F _ { n } ) .$ . Factoring out e and l also removes the illumination scale from the inputs, and a single prediction covers arbitrary intensity ratios (Fig. 3 Right).

Independence across color channels. Without spectral coupling, each color channel is the solution of an independent transport problem. We therefore apply the same single-channel decoder to each channel, with the features of that channel as input. This shares weights across channels and reduces the output to a single channel. The decoder is agnostic to which medium properties vary across channels, since each channel is decoded from its own features. Which properties vary only affects the cost of feature extraction: a channel-dependent extinction requires a separate traversal of the volume per channel. In our experiments, we use a shared extinction and vary only the albedo and illumination per channel, so the density-dependent features are computed once and shared.

Decoupling by construction. A single model that jointly predicts all components and channels would have to learn these independences from data. This requires training cases that cover the combinations of illumination and channel-dependent medium properties, and additional losses that penalize the influence of one channel or component on the others. In our experiments, such joint models did not achieve the required independence. In addition, they required considerably more capacity, which reduced performance substantially. Separate single-channel decoders satisfy both properties by construction and keep each decoder small.

The specific architecture of the decoders, the loss ℓ, and the training procedure are described in Sec. 4.

## 4. Implementation

The rendering pipeline is implemented in Vulkan, with CUDAexported device memory used to directly use rendered buffers in the neural network inference stage, avoiding host-side synchronization overhead and ensuring a tightly coupled rendering–inference loop.

## 4.1. Feature Rendering

Our scenes consist of volumes scaled to fit in a $[ - 1 , 1 ] ^ { 3 }$ cube. We employ exhaustive ray-marching (step size 0.001u) to remove stochasticity for the primary ray features. We do a DDA on the coarse grid of majorants to skip empty regions when traversing the volume, and ray-march only non-zero voxels. A first traversal computes the optical depth τ. A second traversal starts from the first position with non-zero density, starting to determine the crossings, with scales ×4, ×1, ×0.25, ×0.1. Because density scales are sorted, we can handle all crossings with just a second traversal. Early termination is possible when all crossings have been determined.

For the first crossing depth, values are estimated for the transmittance towards the directional light, the environment samples, the anisotropy factor, and the single-scattering albedo.

To reduce variance in the transmitted direct-lighting estimate, we use deterministic ray marching over a short initial segment of radius 0.1u, followed by stochastic ratio tracking. For environment lighting, we importance-sample directions using a quadtree over the equirectangular environment map, with sampling densities proportional to luminance.

## 4.2. Network Architecture

The reconstruction operator S is parametrized using a U-Net backbone composed of residual blocks and attention-based feature mixing. Following HaloNet [VRS<sup>∗</sup>21], we incorporate local spatial self-attention at intermediate resolutions and at the bottleneck. We call our model HaloRT—Halo-attention-based Radiative Transfer model (see Fig. 4).

Since multiple scattering couples distant regions of the medium, a pixel’s radiance can depend on features at other image locations whose relevance varies with the scene, e.g., thin regions through which light enters the volume. Thus, we complement the convolutional backbone with local attention that relates features through content-dependent weights rather than fixed kernels (see Table 1).

![](images/00094f4611196ce626c4fcfd2d114f2ac67a222a98c36784f48bfe858c43d863.jpg)  
Figure 4: Architecture of our HaloRT model. Our U-Net downsamples 6 times the initial resolution and works with blocks of 64 channels in the bottleneck. Local window attention (Halo) is applied instead offull attention in 4 stages and the bottleneck.

Table 1: Scaling of inference-time (ms) with resolution for different 8-channel-in / 1-channel-out architectures, measured on an NVIDIA GeForce RTX 5090.
<table><tr><td>Architecture</td><td> $2 5 6 ^ { 2 }$ </td><td> $\overline { { 5 1 2 ^ { 2 } } }$ </td><td>10242</td><td>20482</td></tr><tr><td>HaloRT</td><td>1.411</td><td>1.563</td><td>2.914</td><td>10.528</td></tr><tr><td>HaloRT without attention</td><td>1.233</td><td>1.378</td><td>2.646</td><td>10.007</td></tr><tr><td>Full attn at same stages</td><td>1.446</td><td>1.618</td><td>4.700</td><td>31.659</td></tr><tr><td>CNN-based Autoencoder</td><td>1.029</td><td>1.132</td><td>1.835</td><td>6.830</td></tr></table>

Restricting attention to local spatial blocks achieves similar scalability with image resolution as CNN-based models used in classical denoisers (4th row in Table 1) and avoids the quadratic computational and memory costs of full spatial self-attention (3rd row of Table 1). At the bottleneck, the reduced spatial resolution still allows each local attention block to operate on features whose receptive fields span the entire input image, enabling image-wide information exchange at a small additional cost compared to attentionfree inference (2nd row in Table 1).

Notably, the network is never explicitly provided with the light direction. Nevertheless, it can reconstruct backlit regions effectively by relating the directional light transmittance to the other extracted features, exploiting the spatially coherent visibility pattern of the light source across the image.

## 4.3. Rendering and Dataset Generation

We render volumetric scenes using delta-tracking combined with next-event estimation (NEE). Light transport is simulated through stochastic free-flight sampling and a single scattering evaluation along the ray. At each scattering event, direct illumination is estimated via explicit light sampling of the environment, while indirect contribution is accumulated through MC integration with a single sample over the phase function that continues estimating higherorder scattered light. Only for the primary rays, the case with no interaction accumulates the environment light (transmitted background).

We further use ratio-tracking to estimate the transmittance, in combination with a coarse grid of majorants to accelerate delta-/ratio-tracking estimators.

L. Leonard & R. Westermann / Volumetric Light Transport Inference
<table><tr><td>Method</td><td>MAE↓</td><td> $\overline { { \mathrm { P S N R } \uparrow \mathrm { ( d B ) } } }$ </td><td>SSIM↑</td><td> $\overline { { \mathrm { { L P I P S \downarrow } } } }$ </td><td> $\overline { { { \mathrm { R M S } } { \cdot } { \mathrm { S D } } { \downarrow } } }$ </td><td> $\overline { { \mathrm { T i m e } \downarrow ( \mathrm { m s } ) } }$ </td></tr><tr><td>Denoiser</td><td> $\overline { { 0 . 0 0 3 9 1 \pm 0 . 0 0 1 7 9 } }$ </td><td> $\overline { { 4 1 . 6 2 \pm 3 . 0 5 } }$ </td><td> $\overline { { 0 . 9 6 8 7 \pm 0 . 0 2 5 0 } }$ </td><td> $\overline { { 0 . 0 9 0 1 \pm 0 . 0 6 5 2 } }$ </td><td> $\overline { { 0 . 0 0 6 2 8 \pm 0 . 0 0 3 3 0 } }$ </td><td> $\overline { { 2 3 . 8 4 \pm 9 . 6 1 } }$ </td></tr><tr><td>DenoiserX</td><td> $0 . 0 0 3 3 8 \pm 0 . 0 0 1 5 1$ </td><td> $\pm 4 . 2 3 \pm 2 . 8 0$ </td><td> $0 . 9 8 2 7 \pm 0 . 0 1 5 1$ </td><td> $0 . 0 5 1 7 \pm 0 . 0 4 2 0$ </td><td> $0 . 0 0 3 2 3 \pm 0 . 0 0 1 2 3$ </td><td> $2 4 . 7 6 \pm 9 . 9 8$ </td></tr><tr><td>HaloRT</td><td> $\mathbf { 0 . 0 0 3 2 4 } \pm \mathbf { 0 . 0 0 2 6 5 }$ </td><td> $4 2 . 7 9 \pm 3 . 9 1$ </td><td> $\mathbf { 0 . 9 8 7 4 } \pm \mathbf { 0 . 0 1 1 } 2$ </td><td> $\mathbf { 0 . 0 4 2 0 \pm 0 . 0 4 1 8 }$ </td><td> $\mathbf { 0 . 0 0 0 7 0 \pm 0 . 0 0 0 3 5 }$ </td><td> ${ \bf 1 0 . 6 6 \pm 0 . 5 2 }$ </td></tr></table>

Table 2: Quality and performance comparison, averaged over the evaluation dataset. Our attention-based HaloRT achieves the best overall performance exceptfor the PSNR, whereas our proposed DenoiserX remains competitive.

We generate a synthetic dataset composed of volumetric cloud scenes rendered under diverse lighting and medium configurations. Environment illumination is generated using 1000 HDR-like maps synthesized with Stable Diffusion [RBL 22], covering a wide range of natural and stylized lighting conditions. In addition, we consider 500 volumes of cloud formations from the Cloudy Project dataset [LTW25]. In total we synthesize 22000 instances with 4096 samples per pixel (spp) for the images.

Scenes are rendered with randomized camera poses, light directions, phase function anisotropy parameters, and scattering albedos. The latter are sampled in the high-scattering regime and may vary per color channel, enabling colored volumetric effects. For each instance we generate the scattered radiance from the environment, the scattered radiance from the light direction, the transmittance, and background. We render the images for a normalized (mean = 1.0) environment and a unitary light intensity since we can combine them with different scalings to cover a wider range of options.

## 4.4. Training

All models are trained using AdamW with identical hyperparameters for fair comparison between the CNN [HHCM21] and HaloRT. We use a learning rate exponentially decaying over 240K optimization steps from $\overline { { 1 0 ^ { - 3 } } } \mathrm { ~ t o ~ } \overline { { 1 0 ^ { - 6 } } }$ , then held constant for 120K extra steps to let the network fully converge at a low, stable rate.

The optimization objective combines an $\ell _ { 1 }$ reconstruction loss with a $\ell _ { 1 }$ of a logarithmic compression variant to better balance low- and high-intensity regions:

$$
\begin{array} { r l r } & { } & { c ( x ) = \mathrm { s g n } ( x ) \log ( 1 + ( 1 0 0 - 1 ) | x | ) / \log 1 0 0 } \\ & { } & { \mathcal { L } = 0 . 6 \| \hat { I } - I \| _ { 1 } + 0 . 4 \| c ( \hat { I } ) - c ( I ) \| _ { 1 } ~ } \end{array}
$$

Each step draws a batch of 16 instances from an in-memory, periodically refreshed training pool, split into micro-batches of 4 for the forward pass. For every micro-batch we accumulate gradients over a horizontal flip augmentation and, independently, over all three color channels of the spectral-independent network, backpropagating each of the resulting six passes separately with the loss rescaled accordingly. Each training instance stores several independently sampled noisy realizations of its stochastic conditioning feature (next-event-estimation sample for the direct branch, environment sample for the environment branch); one realization is drawn at random per instance at every step, so the same deterministic target is seen under different noise draws across training. The number of environment samples varies uniformly between 1 and 16 to make the model agnostic to different noise levels.

![](images/b47e07ce140cec5ac9f3acb22b46a399048b15af4a91d3485fb87804611b71a7.jpg)  
Figure 5: PSNR and LPIPS across the evaluation dataset sorted according to different scene parameters. Our method achieves superior LPIPS although PSNR penalizes global biases not present in denoisers.

## 5. Results

We evaluate three reconstruction approaches under identical rendering and evaluation conditions. The first baseline (Denoiser) is the CNN-based denoiser with 4-spp Monte Carlo radiance buffer proposed by Hofmann et al. [HMES20, HHCM21], which operates on noisy radiance estimates and statistical features. The second variant (DenoiserX) uses the same architecture but replaces the noisy features with our deterministic features, while retaining the 4-spp radiance buffer. Finally, our attention-based U-Net (HaloRT) operates exclusively on auxiliary transport features, without requiring path-traced radiance samples.

Experiment Setup. All experiments are conducted on an NVIDIA GeForce RTX 5090 GPU. We evaluate all methods on a held-out set of 200 test images (512×512) rendered for 15 left-out clouds never used for training. We expand the evaluation dataset by including additional environment illumination scales (×0.2) and directional light intensities (×3), as well as five distinct directional light colors.

Quantitative Results. We use Structural Similarity Index Measure (SSIM) [WBSS04], Peak Signal-to-Noise Ratio (PSNR), Mean Absolute Error (MAE), Root Mean Square of the pixel-wise predictive Standard Deviation (RMS-SD) as a measure of stability, and Learned Perceptual Image Patch Similarity (LPIPS) [ZIE<sup>∗</sup>18] for quantitative results (Table 2). Our method surpasses denoisers in all metrics except PSNR, more notably in stability and time.

We further organize the evaluation dataset according to lighting conditions and volume settings to examine how these parameters affect the performance of the three methods. Fig. 5 presents the corresponding metrics across the different configurations, allowing performance trends and sensitivities to be compared. While DenoiserX consistently achieves higher PSNR values than our method, both methods exhibit similar trends as the scene parameters vary. In contrast, HaloRT consistently achieves lower LPIPS scores, indicating better perceptual similarity to the reference images across all evaluated configurations. This suggests that, despite the lower pixel-wise fidelity measured by PSNR, our method better preserves perceptually relevant details and light transport characteristics across a wide range of scene conditions.

![](images/f15169f5d365df0ef163619a3b982448a373076607c2bdb4ed879d2aea168eff.jpg)  
Figure 6: Comparison of stability across 10 realizations, using maximum absolute pixel-wise deviation from the mean image, amplified by a factor of 16. Denoisers exhibit higher variability due to stochastic radiance inputs. HaloRT exhibits near-zero deviation with only 8 environment samples.

![](images/6494ed5a4a7e4a04a892cd4c1f3b89ccdaacb07f7e3c7db6611ef50aea03d199.jpg)  
Figure 7: Effect ofthe number ofenvironment samples on generation time, reconstruction accuracy, and stability, averaged over the evaluation dataset with varying environment lighting.

Stability. One of the key properties of our method is that the decoder is largely invariant to feature noise, meaning that stochasticity in the features does not propagate through the model. To demonstrate this, we measure the model’s stability as the standard deviation of the inferred image over a given number of realizations (Fig. 6). In particular, we show that only a few environment samples are needed to make the variation imperceptible, with additional samples providing negligible benefits (Fig. 7).

Performance analysis. Although the CNN used for classical denoising is computationally inexpensive, the radiance estimate it requires – computed using a few path traces through the volume – scales poorly with increasing cloud density (Fig. 8), as scattering events become more likely and require spawning additional rays. In contrast, since no paths are computed our runtime remains stable. Along primary rays and environment samples the workload is kept low, due to optimizations like empty-space skipping and fast ratio tracking in sparse volumes, and early termination once the optical depth saturates, along with Russian roulette in ratio tracking, in dense volumes.

![](images/1d1f5dd9cb8d7741897abb46b5e555c2c14b1b82dfe4b643cc475358819e57cc.jpg)  
Figure 8: Reconstruction times at different density scales and camera distances. Our network requires more inference time, but since our method does not depend on path sampling, it shows overall better scalability.

As the camera approaches the cloud, denoiser runtimes increase with screen coverage and, thus, with the number of paths to be evaluated. Our inexpensive auxiliary features in combination with the constant computational complexity of the inference step maintain stable runtimes and nearly constant FPS (Fig. 8).

Qualitative Results. Fig. 9 compares all three different variants qualitatively.

Fig. 10 compares our reconstructions with the path-traced references for a cloud lit from the front, the side, and the back. Starting from a base configuration, each row varies a single medium parameter: density, single-scattering albedo, and phase-function anisotropy. Our method reproduces the brightness and color of the multiply scattered light in all configurations, including the darker appearance of low-albedo media and the bright rim of strongly forward-scattering media under back lighting.

Directional Light Inference. In a final experiment, we compare our approach to MRPNN [HYL 23] using a single directional light. The experiment is carried out on an NVIDIA GeForce RTX 3090, since the MRPPN code does not execute on the RTX 5090. MRPNN is not designed to efficiently infer the radiance transport in an ambient light environment (see Fig. 2), but it also requires multiple, yet significantly less samples to generate a noise-free image in the presence of a single directional light (Fig. 11). Ours infers directional light transport with a single pass, producing an accurate, noise-free estimation that runs faster than a single pass of MRPNN.

Reference

![](images/4844683c159ecf0c415c2d10ab10dccdf6b5ca0dbe99af235d18c21bf26ee7b3.jpg)  
Figure 9: Qualitative comparison of HaloRT against baseline denoisers. Although PSNR is strongly affected by intensity biases, HaloRT better preservesfine details and captures important light transport effects.

![](images/1d6645d9f3c9ba1fe4822269d9daac8c4ddc30c3ea713cc3fb45527c504d4315.jpg)  
Figure 10: Qualitative comparison. Each pair shows the path-traced reference (left) and HaloRT (right), across variations in density, scattering albedo, and anisotropy under different lighting directions.

![](images/2f32ede48924c9f5f64547165ae8bab7c87d9ccf144362ae926687ff20bd518d.jpg)  
Figure 11: Using only one directional monochromatic light, MRPNN [HYL<sup>∗</sup>23] shows clearly visible noise at the boundaries. HaloRT’s inference is noise-free and executes faster than a single pass ofMRPNN.

## 6. Discussion and Limitations

Feature sufficiency. Our results support the view that, for clouds, inexpensive primary-ray features with additional environment samples carry most of the information needed to infer the full transport solution. HaloRT, which never observes a path-traced radiance sample, achieves better perceptual quality and stability than volumetric denoisers at a slightly lower PSNR (Table 2). The same features also improve the volumetric denoiser substantially when they replace its original auxiliary buffers (DenoiserX vs. Denoiser), which indicates that they encode transport information that a radiance-based reconstruction can also exploit.

Even in scenarios which are challenging for HaloRT, i.e., under normal environment lighting where global transport dominates and strong directional light, sufficiency (A1) of our proposed auxiliary feature set still holds reasonably well in practice. In these scenarios, the radiance samples used by a denoiser provide information that our features do not explicitly capture, i.e., the light transport depends on the volumetric structures along paths that leave the vicinity of the primary ray. Since (A1) is only required for plausible scenes $\Theta \sim p ( \Theta )$ , the decoder resolves the remaining ambiguities with regularities learned from the training clouds, and its accuracy is therefore tied to how well the training distribution covers the scenes encountered at inference time.

As shown in Fig. 9, as well as in Fig. 5, in the aforementioned scenarios HaloRT shows overall lower PSNR than DenoiserX. However, it is also seen that HaloRT can nevertheless recover features at far more detail, resulting in locally improved and overall tighter reconstructions.

Noise invariance. Since most features are deterministic and only the light-visibility features are estimated stochastically, the decoder becomes nearly invariant to the feature noise (A2). The standard deviation across realizations is less than a quarter of that of DenoiserX and one ninth of that of the Denoiser (Table 2). A comparison of temporal consistency between the methods can be seen in the supplementary video.

Cost. Unlike path-based reconstruction, our runtime remains stable across densities (Fig. 8). In addition, the network dominates the cost: with local attention it scales close to a convolutional network, but inference still takes about 10.5 ms per pass at $2 0 4 8 ^ { 2 }$ (Table 1), and one pass is required per illumination component and color channel.

Scope. We trained and evaluated our method only on cloud-like media from a single dataset, with the Henyey–Greenstein phase function, non-emissive media, and a single volume under distant illumination, consisting of a smooth environment map and one dominant directional light. Surfaces inside or around the medium, local light sources, and emission are not handled. We share the extinction across color channels. Channel-dependent extinction is supported by the formulation but requires one volume traversal per channel. Finally, our training environments are synthesized, and a larger set of captured HDR environments would provide a stronger test of generalization.

## 7. Conclusion and Future Work

We present a real-time method for volumetric light transport that removes path-traced radiance samples from inference entirely. Rather than denoising stochastic estimates of the image, we reconstruct full multiple scattering from a compact set of auxiliary features: optical depth, multi-scale interaction depths, transmittance towards the light, environment illumination at the expected first interaction, and local medium properties. We formalize this as deterministic decoding under two testable assumptions, feature sufficiency and noise-invariant decoding. We show that MC denoising is a special case in which sufficiency holds trivially and all of the difficulty lies in noise invariance. Our features shift this balance: noise invariance becomes easy to satisfy, while sufficiency holds approximately for clouds. By exploiting the linearity of light transport in the illumination and the independence of color channels, we learn the decoder as two single-channel HaloRT models, which use local self-attention to exchange information across the image at a cost that scales to high resolutions. On unseen clouds, our method outperforms volumetric denoisers in accuracy and stability, runs more than twice as fast than the feature-enhanced denoiser, and depends much less on medium density and camera distance.

Several directions remain open. Extending the features and training data to other media, such as smoke, fire, or subsurface scattering, and to scenes that combine media with surfaces and local lights would broaden applicability and test the limits of the sufficiency assumption. Temporal recurrence or cross-frame attention could improve coherence under camera motion without reintroducing a dependence on path sampling.

The remaining error has three sources, each leaving room for improvements. First, the features are only approximately sufficient (A1). Optimizing the hand-designed feature operators jointly with the decoder could narrow this gap. Second, the learned decoder S<sub>Ψ</sub> is limited by model capacity, training data, and optimization, which larger and more diverse datasets and architectures could reduce. Third, the visibility features remain stochastic (A2), and lowervariance estimators, such as deep shadow maps for the directional light, could reduce this residual noise at little cost.

## References

[Áfr26] ÁFRA A. T.: Intel<sup>®</sup> Open Image Denoise, 2026. https:// www.openimagedenoise.org. 2

[AK90] ARVO J., KIRK D.: Particle transport and image synthesis. In Proceedings of the 17th Annual Conference on Computer Graphics and Interactive Techniques (New York, NY, USA, aug 1990), SIG-

GRAPH ’90, Association for Computing Machinery, pp. 63–66. doi: 10.1145/97879.97886. 1

[BVM<sup>∗</sup>17] BAKO S., VOGELS T., MCWILLIAMS B., MEYER M., NOVÁK J., HARVILL A., SEN P., DEROSE T., ROUSSELLE F.: Kernelpredicting convolutional networks for denoising Monte Carlo renderings. ACM Transactions on Graphics 36, 4 (jul 2017). doi:10.1145/ 3072959.3073708. 1, 3

[BWP<sup>∗</sup>20] BITTERLI B., WYMAN C., PHARR M., SHIRLEY P., LEFOHN A., JAROSZ W.: Spatiotemporal reservoir resampling for realtime ray tracing with dynamic direct lighting. ACM Trans. Graph. 39, 4 (Aug. 2020). doi:10.1145/3386569.3392481. 3

[CKS<sup>∗</sup>17] CHAITANYA C. R. A., KAPLANYAN A. S., SCHIED C., SALVI M., LEFOHN A. E., NOWROUZEZAHRAI D., AILA T.: Interactive reconstruction of Monte Carlo image sequences using a recurrent denoising autoencoder. ACM Transactions on Graphics 36, 4 (jul 2017). doi:10.1145/3072959.3073601. 1, 2

[Coo86] COOK R. L.: Stochastic sampling in computer graphics. ACM Transactions on Graphics 5, 1 (jan 1986), 51–72. doi:10.1145/ 7529.8927. 1

[DFNB25] DENISOVA E., FRANCIA P., NARDI C., BOCCHI L.: Advancements in biomedical rendering: A survey on AI-based denoising techniques. Computers in Biology and Medicine 197 (2025), 110979. doi:10.1016/j.compbiomed.2025.110979. 3

[DSHL10] DAMMERTZ H., SEWTZ D., HANIKA J., LENSCH H. P. A.: Edge-avoiding à-trous wavelet transform for fast global illumination filtering. In Proceedings ofthe Conference on High Performance Graphics (Goslar, DEU, 2010), HPG ’10, Eurographics Association, pp. 67–75. doi:10.5555/1921479.1921491. 2

[HHCM21] HOFMANN N., HASSELGREN J., CLARBERG P., MUNKBERG J.: Interactive path tracing and reconstruction of sparse volumes. In ACM SIGGRAPH Symposium on Interactive 3D Graphics and Games (I3D) (March 2021), vol. 6. doi:10.1145/3585497. 2, 3, 7

[HHM23] HOFMANN N., HASSELGREN J., MUNKBERG J.: Joint neural denoising of surfaces and volumes. Proceedings of the ACM on Computer Graphics and Interactive Techniques 6, 1 (may 2023). doi: 10.1145/3585497. 3

[HMES20] HOFMANN N., MARTSCHINKE J., ENGEL K., STAM-MINGER M.: Neural denoising for path tracing of medical volumetric data. Proceedings of the ACM on Computer Graphics and Interactive Techniques 3, 2 (2020), 1–18. doi:10.1145/3406181. 2, 3, 7

[HMS<sup>∗</sup>20] HASSELGREN J., MUNKBERG J., SALVI M., PATNEY A., LEFOHN A. E.: Neural temporal adaptive sampling and denoising. Computer Graphics Forum 39, 2 (2020), 147–155. doi:10.1111/cgf. 13919. 3

[HSG<sup>∗</sup>22] HUANG T., SONG Y., GUO J., TAO C., ZONG Z., FU X., LI H., GUO Y.: Real-time deep radiance reconstruction from imperfect caches. Computer Graphics Forum 41, 7 (2022), 267–278. doi:10. 1111/cgf.14675. 3

[HY21] HUO Y., YOON S.-E.: A survey on deep learning-based Monte Carlo denoising. Computational Visual Media 7, 2 (2021), 169–185. doi:10.1007/s41095-021-0209-9. 2

[HYL<sup>∗</sup>23] HU J., YU C., LIU H., YAN L., WU Y., JIN X.: Deep realtime volumetric rendering using multi-feature fusion. In ACM SIG-GRAPH 2023 Conference Proceedings (New York, NY, USA, 2023), SIGGRAPH ’23, Association for Computing Machinery. doi:10. 1145/3588432.3591493. 2, 3, 8, 11

[IGMM22] IGLESIAS-GUITIAN J. A., MANE P., MOON B.: Real-time denoising of volumetric path tracing for direct volume rendering. IEEE Transactions on Visualization and Computer Graphics 28, 7 (2022), 2734–2747. doi:10.1109/TVCG.2020.3037680. 3

[IMF<sup>∗</sup>21] I ¸SIK M., MULLIA K., FISHER M., EISENMANN J., GHARBI M.: Interactive Monte Carlo denoising using affinity of neural features. ACM Transactions on Graphics 40, 4 (2021), 1–13. doi:10.1145/ 3450626.3459793. 1, 3

[Kaj86] KAJIYA J. T.: The rendering equation. In Proceedings ofthe 13th Annual Conference on Computer Graphics and Interactive Techniques (New York, NY, USA, 1986), SIGGRAPH ’86, Association for Computing Machinery, pp. 143–150. doi:10.1145/15922.15902. 1

[KBS15] KALANTARI N. K., BAKO S., SEN P.: A machine learning approach for filtering Monte Carlo noise. ACM Transactions on Graphics 34, 4 (jul 2015). doi:10.1145/2766977. 2

[KMM<sup>∗</sup>17] KALLWEIT S., MÜLLER T., MCWILLIAMS B., GROSS M., NOVÁK J.: Deep scattering: Rendering atmospheric clouds with radiance-predicting neural networks. ACM Transactions on Graphics 36, 6 (nov 2017). doi:10.1145/3130800.3130880. 2, 3

[Lee83] LEE J.-S.: Digital image smoothing and the sigma filter. Computer Vision, Graphics, and Image Processing 24, 2 (1983), 255–269. doi:10.1016/0734-189X(83)90047-6. 2

[LHW21] LEONARD L., HÖHLEIN K., WESTERMANN R.: Learning multiple-scattering solutions for sphere-tracing of volumetric subsurface effects. Computer Graphics Forum 40, 2 (2021), 165–178. doi:10. 1111/cgf.142623. 3

[LTW25] LEONARD L., THÜREY N., WESTERMANN R.: Light transport-aware diffusion posterior sampling for single-view reconstruction of 3d volumes. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2025). 7

[LWY21] LIN D., WYMAN C., YUKSEL C.: Fast volume rendering with spatiotemporal reservoir resampling. ACM Transactions on Graphics 40, 6 (2021), 279:1–279:18. doi:10.1145/3478513.3480499. 3

[LXS20] LU Y., XIE N., SHEN H. T.: DMCR-GAN: Adversarial denoising for Monte Carlo renderings with residual attention networks and hierarchical features modulation of auxiliary buffers. In SIGGRAPHAsia 2020 Technical Communications (2020). doi:10.1145/3410700. 3425426. 2

[LYG<sup>∗</sup>26] LIANG J., YUAN H., GAI M., WANG G., LI S.: Multi-feature radiance baking neural networks for instant volumetric rendering. In Proceedings ofthe Special Interest Group on Computer Graphics and Interactive Techniques, Conference Papers (New York, NY, USA, 2026), SIG-GRAPH Conference Papers ’26, Association for Computing Machinery. URL: https://doi.org/10.1145/3799902.3811179, doi:10.1145/3799902.3811179. 3

[MLSH24] MULLIA K., LUAN F., SUN X., HAŠAN M.: RNA: Relightable neural assets. ACM Transactions on Graphics (2024). doi: 10.1145/3695866. 3

[NAM<sup>∗</sup>17] NALBACH O., ARABADZHIYSKA E., MEHTA D., SEIDEL H.-P., RITSCHEL T.: Deep shading: Convolutional neural networks for screen space shading. Computer Graphics Forum 36, 4 (2017), 65–78. doi:10.1111/cgf.13225. 2, 3

[NGHJ18] NOVÁK J., GEORGIEV I., HANIKA J., JAROSZ W.: Monte Carlo methods for volumetric light transport simulation. Computer Graphics Forum 37, 2 (may 2018), 551–576. doi:10.1111/cgf. 13383. 1

[ODG26] OLAJOS R., DOGGETT M., GOSWAMI P.: Environmental volumetric neural shading of clouds for real-time rendering. Proc. ACM Comput. Graph. Interact. Tech. 9, 4 (June 2026). URL: https: //doi.org/10.1145/3820020, doi:10.1145/3820020. 3

[PJH23] PHARR M., JAKOB W., HUMPHREYS G.: Physically Based Rendering: From Theory to Implementation, 4th ed. The MIT Press, Cambridge, MA, USA, 2023. 1

[RBL<sup>∗</sup>22] ROMBACH R., BLATTMANN A., LORENZ D., ESSER P., OMMER B.: High-resolution image synthesis with latent diffusion models. In 2022 IEEE/CVF conference on computer vision and pattern recognition (CVPR) (2022), ieee, pp. 10674–10685. 7

[RBRD22] RAINER G., BOUSSEAU A., RITSCHEL T., DRETTAKIS G.: Neural precomputed radiance transfer. Computer Graphics Forum 41, 2 (2022), 365–378. doi:10.1111/cgf.14480. 3

[RKZ12] ROUSSELLE F., KNAUS C., ZWICKER M.: Adaptive rendering with non-local means filtering. ACM Transactions on Graphics 31, 6 (2012), 195:1–195:11. doi:10.1145/2366145.2366214. 2

[RW94] RUSHMEIER H. E., WARD G. J.: Energy preserving nonlinear filters. In Proceedings of the 21st Annual Conference on Computer Graphics and Interactive Techniques (New York, NY, USA, 1994), SIGGRAPH ’94, Association for Computing Machinery, pp. 131–138. doi:10.1145/192161.192189. 2

[SKW<sup>∗</sup>17] SCHIED C., KAPLANYAN A., WYMAN C., PATNEY A., CHAITANYA C. R. A., BURGESS J., LIU S., DACHSBACHER C., LEFOHN A., SALVI M.: Spatiotemporal variance-guided filtering: Real-time reconstruction for path-traced global illumination. In Proceedings of the ACM SIGGRAPH Symposium on High Performance Graphics (2017), HPG ’17, pp. 2:1–2:12. doi:10.1145/3105762. 3105770. 1, 2

[UKK<sup>∗</sup>26] USS W., KIM S., KALINSKI´ W., ANAND H., KUZNETSOV A., HARIHARA R.: Lightweight attention-based indirect illumination. In Proceedings of the Special Interest Group on Computer Graphics and Interactive Techniques Conference Posters (New York, NY, USA, 2026), SIGGRAPH Posters ’26, Association for Computing Machinery. URL: https://doi.org/10.1145/3799825.3818739, doi:10.1145/3799825.3818739. 3

[VRM<sup>∗</sup>18] VOGELS T., ROUSSELLE F., MCWILLIAMS B., RÖTHLIN G., HARVILL A., ADLER D., MEYER M., NOVÁK J.: Denoising with kernel prediction and asymmetric loss functions. ACM Transactions on Graphics (TOG) 37, 4 (2018), 1–15. doi:10.1145/3197517. 3201388. 1, 3

[VRS<sup>∗</sup>21] VASWANI A., RAMACHANDRAN P., SRINIVAS A., PARMAR N., HECHTMAN B., SHLENS J.: Scaling local self-attention for parameter efficient visual backbones. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (June 2021), pp. 12894–12904. 6

[ZZL<sup>∗</sup>17] ZHOU G., ZHU D., LI T., WANG Z., ZHOU Y.: On learning the visibility for joint importance sampling of low-order scattering. Neurocomputing 228 (2017), 97–105. Advanced Intelligent Computing: Theory and Applications. doi:10.1016/j.neucom.2016. 09.086. 3

[VVF24] VAVILALA V., VASANTH R., FORSYTH D.: Denoising Monte Carlo renders with diffusion models. In ACM SIGGRAPH 2024 Posters (2024), SIGGRAPH ’24, Association for Computing Machinery. doi: 10.1145/3641234.3671026. 1

[WBSS04] WANG Z., BOVIK A., SHEIKH H., SIMONCELLI E.: Image quality assessment: from error visibility to structural similarity. IEEE Transactions on Image Processing 13, 4 (2004), 600–612. doi:10. 1109/TIP.2003.819861. 7

[XXZ<sup>∗</sup>25] XU C., XU X., ZHANG J., LIU Y., CAO Y., ZHAO L.: Real-time neural denoising for volume rendering using dual-input feature fusion network. Computer Graphics Forum 44, 6 (2025), e70276. doi:10.1111/cgf.70276. 3

[XTW<sup>∗</sup>26] XU B., T M. V., WANG C., LI T.-M., WU L., WRONSKI B., RAMAMOORTHI R., SALVI M.: A generalizable light transport 3D embedding for global illumination. doi:10.1145/3799902. 3811095. 3

[XZW<sup>∗</sup>19] XU B., ZHANG J., WANG R., XU K., YANG Y.-L., LI C., TANG R.: Adversarial Monte Carlo denoising with conditioned auxiliary feature modulation. ACM Transactions on Graphics 38, 6 (2019), 224:1– 224:12. doi:10.1145/3355089.3356547. 2

[YGH<sup>∗</sup>25] YAN R., GUO H., HUANG L., XIAO N., LI S., WANG Y., LV Y., CHEN G.: A survey on deep learning for Monte Carlo path tracing. ACM Computing Surveys 58, 4 (2025), 1–38. doi:10.1145/ 3759487. 2, 3

[ZDP<sup>∗</sup>25] ZENG C., DONG Y., PEERS P., WU H., TONG X.: Renderformer: Transformer-based neural rendering of triangle meshes with global illumination. In Proceedings of the Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers (2025), pp. 1–11. 3

[ZIE<sup>∗</sup>18] ZHANG R., ISOLA P., EFROS A. A., SHECHTMAN E., WANG O.: The unreasonable effectiveness of deep features as a perceptual metric. In 2018 IEEE/CVF conference on computer vision and pattern recognition (2018), IEEE, pp. 586–595. 7

[ZSS26] ZHANG Z., SIMO-SERRA E.: Neural scene baking for permutation invariant transparency rendering with real-time global illumination. Computational Visual Media 12, 2 (2026), 321–335. doi: 10.26599/CVM.2025.9450433. 3