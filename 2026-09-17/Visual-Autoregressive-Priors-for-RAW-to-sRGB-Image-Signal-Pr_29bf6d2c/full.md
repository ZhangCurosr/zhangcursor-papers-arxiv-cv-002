# Visual Autoregressive Priors for RAW-to-sRGB Image Signal Processing

Tailai Chen<sup>1</sup>, Xiaotong Luo<sup>2</sup>, Yuan Gao<sup>1⋆</sup>, Xin Jin<sup>1</sup>, and Wenjun Zeng<sup>1</sup>

<sup>1</sup> OmniVision-IDT Joint Laboratory for Intelligent Image Sensing, Ningbo Key Laboratory of Spatial Intelligence and Digital Derivative, Ningbo Institute of Digital Twin, Eastern Institute of Technology, Ningbo; Zhejiang Key Laboratory of Industrial Intelligence and Digital Twin ygao@idt.eitech.edu.cn <sup>2</sup> The Hong Kong Polytechnic University

Abstract. RAW-to-sRGB image signal processing (ISP) must recover perceptually faithful colors and fine details from sensor measurements, often under imperfect spatial alignment and missing camera metadata. This paper presents, to the best of our knowledge, the first application of visual autoregressive (VAR) next-scale prediction over a discrete image codebook to the RAW-to-sRGB ISP task. We adapt a frozen 1.10 Bparameter VAR backbone for RAW-conditioned ISP with only 32.93 M trainable parameters (2.99%), and propose a frequency-decomposed color loss that separately supervises low-frequency tone via wavelet LL cosine similarity and chromatic edges via detail-band ℓ<sub>1</sub>. On the Zurich RAW-to-sRGB benchmark, the method improves PSNR-Y from 21.31 to 21.89 dB and reduces LPIPS from 0.276 to 0.218 on the full 1,204-image test set. Diagnostic experiments show that the VAR prior preserves structure well, but continuous color transfer remains the dominant bottleneck: oracle afine correction recovers 3.8 dB, while learned color heads yield marginal gains.

Keywords: RAW-to-sRGB ISP · visual autoregressive modeling · color correction · misaligned supervision

## 1 Introduction

Modern camera pipelines transform RAW sensor measurements into displayready sRGB images through a cascade of operations: demosaicing, denoising, white balance, color correction, gamma mapping, and sharpening. End-to-end neural ISP aims to replace this entire stack with a single learned mapping [3, 5], but the task is considerably more challenging than standard image restoration. In real paired datasets such as Zurich RAW-to-sRGB (ZRR), RAW inputs and sRGB targets are captured by diferent cameras with diferent optics, exposure behavior, and color rendering pipelines, and camera metadata may be absent entirely. The model must therefore handle texture reconstruction, spatially misaligned supervision, and cross-camera color transfer simultaneously.

This challenge has shaped the neural ISP literature. W-Net treats ZRR as a misaligned supervision problem with color-sensitive losses [13]. LiteISPNet introduces a global color mapping module for alignment against a color-adjusted reference [16]. AWNet and MW-ISPNet use wavelet and multi-scale components for detail preservation with enlarged receptive fields [1, 4]. FourierISP separates phase (structure) and amplitude (style) in the frequency domain [2]. RMFA-Net argues that RAW-specific preprocessing directly afects color fidelity [6]. These works establish a recurring principle: successful RAW-to-sRGB models need both spatial detail preservation and controlled color transfer.

Generative priors ofer a complementary angle. Difusion models have been applied: DifRAW conditions on LiteISPNet outputs [14], and ISPDifuser separates grayscale reconstruction from histogram-guided color mapping [11]. These report strong perceptual quality but require iterative sampling. Visual autoregressive modeling (VAR) [12] generates images through next-scale prediction over vector-quantized latents, ofering a coarse-to-fine prior naturally suited to ISP: the RAW image supplies scene content at all scales while the pretrained prior regularizes plausible sRGB structure. However, a discrete codebook is not obviously suited to continuous camera operations—white balance, exposure compensation, and tone curves—that are fundamentally smooth, per-pixel transformations.

We therefore ask: can a frozen VAR prior be adapted for RAW-to-sRGB ISP with a small trainable conditioning branch, and where does this adaptation fail? Our approach freezes the entire VAR backbone and VQ-VAE, training only conditioning embeddings and cross-attention modules (2.99% of total parameters). To handle misaligned supervision, we propose a frequency-decomposed color loss: a Haar wavelet decomposes outputs and targets, and we apply cosine similarity on the LL subband for global tone while using $\ell _ { 1 }$ on detail bands for chromatic edges. This design draws on the structure–color separation from W-Net [13] and FourierISP [2], but operates in the wavelet domain for better spatial locality under misalignment.

On ZRR, the method improves PSNR-Y by 0.58 dB and reduces LPIPS by 21% over the baseline without color supervision. However, diagnostic experiments reveal that remaining errors are dominated by brightness, contrast, and white-balance shifts rather than texture or structural artifacts. Oracle per-image afine color correction recovers 3.8 dB PSNR, demonstrating that the output already contains the necessary structural detail. Yet learned color heads yield only marginal or negative gains, confirming that robust continuous color transfer under misaligned supervision remains the key open problem for discrete generative priors in ISP.

Our contributions are:

– To the best of our knowledge, the first application of visual autoregressive modeling to RAW-to-sRGB ISP, with a parameter-eficient formulation adapting a frozen 1.10 B-parameter VAR backbone using only 32.93 M trainable parameters (2.99%).

– A frequency-decomposed color loss using wavelet-domain cosine similarity for tone and $\ell _ { 1 }$ for chromatic edges, robust to spatial misalignment in crosscamera datasets.

– Diagnostic experiments demonstrating that the VAR prior captures structure well, while continuous color transfer remains the main limitation, with oracle afine correction recovering 3.8 dB PSNR.

## 2 Related Work

Learned RAW-to-sRGB ISP. PyNET and the ZRR benchmark introduced a practical paired RAW–DSLR evaluation setting [5], while the Mobile AI 2021 challenge emphasized mobile ISP under resource constraints [3]. W-Net [13] addresses misalignment with a two-stage U-Net and color-sensitive loss. AWNet [1] uses wavelet-domain processing and global context, while MW-ISPNet [4] employs multi-level wavelet components. LiteISPNet [16] models geometric correspondence explicitly for learning with inaccurately aligned supervision. FourierISP [2] decouples structure and style via phase and amplitude separation. RMFA-Net [6] designs RAW-specific preprocessing to preserve color fidelity. Our work takes a diferent approach: probing how far a large pretrained discrete generative prior can be pushed for ISP via parameter-eficient adaptation.

Generative priors for ISP. DifRAW [14] conditions a difusion model on LiteISP-Net outputs to enhance perceptual quality. ISPDifuser [11] separates grayscale detail from histogram-guided color consistency, achieving strong ZRR results at higher iterative inference cost. Our work explores a diferent class of generative prior: a frozen next-scale autoregressive transformer with deterministic singlepass inference.

Discrete visual priors and VAR for low-level vision. Vector-quantized autoencoders [8] represent images through a discrete codebook and serve as a common substrate for generative modeling. VAR [12] redefines autoregression as next-scale prediction, enabling coarse-to-fine generation over VQ latents that naturally captures the hierarchical structure of natural images. Very recently, this paradigm has been extended to low-level vision: VARSR [9] adapts VAR for image super-resolution with prefix tokens and a difusion refiner, and RestoreVAR [10] extends VAR to all-in-one image restoration via cross-attention conditioning on degraded image latents. To the best of our knowledge, our work is the first to apply the VAR framework to the RAW-to-sRGB ISP task, which presents unique challenges of cross-camera color transfer and spatially misaligned supervision not encountered in standard super-resolution or restoration settings.

Color correction under misalignment. Color is central to ISP but entangled with exposure, white balance, tone curves, and the target camera’s rendering style. In misaligned datasets like ZRR, direct pixel-level supervision can degrade high-frequency details or encourage desaturated colors. Prior work addresses this through color-aware objectives [13], global context modules [1, 16], frequencydomain style separation [2], histogram-guided colorization [11], and sensor-aware tone modeling [6]. Our experiments separate the VAR branch’s discrete detail generation from frequency-domain color objectives and post-hoc color heads to isolate the chromatic mismatch contribution.

## 3 Method

Figure 1 illustrates the proposed framework. The pipeline consists of four stages: VQ-VAE tokenization, RAW conditioning, autoregressive code prediction with a frozen VAR backbone, and frequency-decomposed color supervision. During inference, the RAW condition guides the frozen model to predict multi-scale VQ codes, decoded into the final sRGB output in a single deterministic forward pass.

![](images/3ff3919520df46d0d6b918f288d34bf21a10ce78cfbb67f663b866b0615c19c2.jpg)  
Fig. 1: Overview of the proposed RAW-to-sRGB formulation. The frozen VQ-VAE tokenizes the target into multi-scale codes for training supervision. The frozen VAR transformer predicts codes conditioned on a RAW-derived signal through trainable adapters. A frequency-decomposed color loss supervises tone and chromatic edges in the wavelet domain.

## 3.1 VQ-VAE Tokenization

The foundation of our approach is a pretrained VQ-VAE [8] that maps continuous images into discrete code sequences. The encoder maps an input $x \in$ $\mathbb { R } ^ { H \times W \times 3 }$ to a spatial feature map, quantized element-wise against a learned codebook $\mathcal { C } = \{ e _ { k } \} _ { k = 1 } ^ { K }$ with K = 4096 entries and latent dimension $C _ { \mathrm { v a e } } = 3 2$

Following VAR [12], the VQ-VAE produces $S = 1 0$ scale levels with resolutions $\{ 1 { \times } 1 , 2 { \times } 2 , \ldots , 1 0 { \times } 1 0 \}$ , yielding $L = 6 8 0$ total tokens per image. The coarsest scale captures global color and layout, while finer scales add local detail and texture.

This multi-scale representation implies a fidelity ceiling: the VQ-VAE reconstruction defines the best any downstream autoregressive prediction can achieve. If the codebook cannot faithfully represent the target camera’s color gamut, the system inherits this limitation regardless of conditioning quality. We analyze this ceiling in Sec. 4.4.

## 3.2 RAW Conditioning

Given a RAW Bayer image $\boldsymbol { r } \in \mathbb { R } ^ { H _ { r } \times W _ { r } }$ with an RGGB color filter array, we construct a three-channel conditioning image $c \in \mathbb { R } ^ { 2 5 6 \times 2 5 6 \times 3 }$ by splitting the mosaic into R, G, B channels (averaging the two green pixels), then bilinearly resizing to $2 5 6 \times 2 5 6$ and normalizing to [−1, 1]. This packing prioritizes compatibility with the pretrained three-channel conditioning embedding. The two green pixels carry slightly diferent spatial information due to their Bayer grid ofset; a more sophisticated CFA-aware packing could preserve this, but we treat it as secondary relative to the color transfer problem.

## 3.3 Parameter-Eficient Adaptation

The target sRGB image x is tokenized by the frozen VQ-VAE encoder into multi-scale discrete code maps $\{ z _ { s } \} _ { s = 1 } ^ { S }$ . The VAR transformer predicts these codes autoregressively from coarse to fine, conditioned on c via cross-attention. At each scale $s ,$ the model receives all coarser-scale codes $z _ { < s }$ and predicts $z _ { s }$ in parallel across spatial positions.

We freeze all pretrained VAR parameters θ and train only the conditioning parameters $\phi$ (conditioning embedding and cross-attention projections):

$$
\mathcal { L } _ { \mathrm { A R } } = - \sum _ { s = 1 } ^ { S } \sum _ { i } \log p _ { \theta , \phi } ( z _ { s , i } \mid z _ { < s } , z _ { s , < i } , c ) ,\tag{1}
$$

where $z _ { s , i }$ is the target code at scale s and position i. This yields 32.93 M trainable parameters out of 1.10 B total (2.99%), preserving the pretrained visual prior while adapting to RAW-conditioned generation. The frozen VQ-VAE contributes an additional 108.95 M non-updated parameters.

## 3.4 Frequency-Decomposed Color Supervision

Direct pixel-level losses between decoded output and DSLR target are problematic in ZRR due to residual spatial misalignment. Small spatial ofsets create large pixel gradients that pull the model toward blurry, spatially averaged outputs. We therefore decompose supervision in the wavelet domain, which provides spatial locality and a natural separation between low-frequency color and midfrequency edges.

Given prediction xˆ and target x, a single-level Haar wavelet yields four subbands: approximation (LL) and detail (LH, HL, HH). We apply cosine similarity on LL for global tone and white balance:

$$
\mathcal { L } _ { \mathrm { L L } } = 1 - \frac { \langle \mathrm { L L } ( \hat { x } ) , \mathrm { L L } ( x ) \rangle } { \Vert \mathrm { L L } ( \hat { x } ) \Vert _ { 2 } \Vert \mathrm { L L } ( x ) \Vert _ { 2 } + \epsilon } ,\tag{2}
$$

and $\ell _ { 1 }$ on detail subbands for chromatic edges:

$$
\mathcal { L } _ { \mathrm { d e t a i l } } = \frac { \mu } { 3 } \sum _ { d \in \{ \mathrm { L H } , \mathrm { H L } , \mathrm { H H } \} } \| d ( \hat { x } ) - d ( x ) \| _ { 1 } ,\tag{3}
$$

with $\mu = 0 . 1$ to down-weight details relative to the LL tone loss.

The cosine similarity in $\mathcal { L } _ { \mathrm { L L } }$ is invariant to global brightness scaling, important because the smartphone and DSLR may have substantially diferent exposure levels. The LL downsampling absorbs small spatial ofsets that would cause large errors in a full-resolution pixel loss, while the detail-band $\ell _ { 1 }$ encourages preservation of chromatic edge structure without requiring exact alignment.

The total training objective is:

$$
\begin{array} { r } { \mathcal { L } = \mathcal { L } _ { \mathrm { A R } } + \lambda _ { \mathrm { f r e q } } ( \mathcal { L } _ { \mathrm { L L } } + \mathcal { L } _ { \mathrm { d e t a i l } } ) , } \end{array}\tag{4}
$$

with $\lambda _ { \mathrm { f r e q } } ~ = ~ 0 . 0 0 5$ , balancing discrete code prediction with continuous color supervision.

Diferentiable decoding during training. The image-domain loss does not backpropagate through an argmax operation. For each token logit vector $\ell _ { s , i }$ , we compute a soft assignment $p _ { s , i } = \mathrm { s o f t m a x } ( \ell _ { s , i } )$ and its expected codebook embedding $\bar { e } _ { s , i } = p _ { s , i } \mathcal { C }$ . The expected embeddings from all scales are accumulated by the frozen residual-quantization path and decoded by the frozen VQ-VAE to obtain ${ \hat { x } } .$ . Consequently, gradients from the wavelet losses pass through the decoder, the fixed codebook multiplication, and the softmax to the predicted logits and trainable conditioning parameters $\phi ;$ the VAR backbone, codebook, and VQ-VAE remain frozen. At inference, we replace the soft assignments with deterministic top-1 code selection.

Comparison with Fourier-domain approaches. Our wavelet decomposition shares conceptual ground with FourierISP [2], which separates phase (structure) and amplitude (style) globally. The key advantage of the wavelet approach is spatial locality: each coeficient reflects a local neighborhood rather than a global frequency bin. Under misalignment, a small shift causes bounded perturbations in local wavelet coeficients but large phase errors across the Fourier spectrum, making wavelet supervision more robust for cross-camera datasets.

## 3.5 Post-Hoc Color Diagnostics

To quantify how much of the remaining gap is attributable to color rather than structural errors, we evaluate three lightweight post-hoc color modules on the frozen model output y, intentionally constrained to prevent texture hallucination:

– Oracle afine: A per-image 3×4 afine matrix fit by least-squares from output to ground truth. Since it uses test-time ground truth, it is not deployable— it provides an upper bound on linear color-only correction.

– Low-frequency residual head: A small CNN with average-pooling bottleneck predicts a spatially smooth residual $\varDelta _ { \mathrm { l o w } }$ from the concatenation of y and $c ,$ producing $y ^ { \prime } = y + \varDelta _ { \mathrm { l o w } } ( y , c )$

– Adaptive CCM: A lightweight network predicts a per-image $3 \times 3$ matrix A and bias b from global average-pooled features, producing $y ^ { \prime } = A y + b ,$ trained with oracle-distilled supervision.

The contrast between oracle upper bound and learned modules reveals whether the bottleneck is structural or chromatic.

## 4 Experiments

We first describe the experimental setup, then present quantitative and qualitative results, analyze the VQ-VAE reconstruction ceiling, and finally conduct post-hoc color correction diagnostics to isolate the structure–color bottleneck.

## 4.1 Setup

Dataset. We evaluate on the Zurich RAW-to-sRGB (ZRR) dataset [5], pairing Huawei P20 Pro smartphone RAW captures with Canon 5D Mark IV DSLR images. The training set contains 46,839 paired patches $( 4 4 8 \times 4 4 8 )$ , and the test set contains 1,204 patches. Due to diferent lens assemblies, sensor sizes, and color pipelines, there is inherent spatial misalignment and substantial color mismatch between pairs, making ZRR particularly challenging.

Metrics. For all of our ablations, we report PSNR and SSIM on the luminance (Y) channel using one fixed evaluation script; this reduces the impact of crosscamera color shifts and enables controlled within-method comparisons. We also report LPIPS [15] for perceptual similarity and secondary no-reference scores CLIPIQA and MUSIQ, though we emphasize full-reference fidelity because noreference metrics can favor aesthetically pleasing but unfaithful reconstructions.

Architecture and training. The VAR backbone has depth 24, embedding dimension 1536, and 24 attention heads (1.10 B parameters). The VQ-VAE uses $K = 4 0 9 6$ codes with $C _ { \mathrm { v a e } } = 3 2$ , producing S = 10 scales with $L = 6 8 0$ tokens (108.95 M parameters). Both are entirely frozen; only conditioning embedding and cross-attention projections are trained (32.93 M, 2.99%). We train with AdamW [7] $\mathrm { ( l r = 1 0 ^ { - 4 } }$ , cosine annealing). At inference, deterministic top-k=1 decoding produces a single output in one forward pass.

## 4.2 Main Results

Table 1: Ablation on the full ZRR test set (1,204 images). Each row adds one component to the baseline.
<table><tr><td>Configuration</td><td>PSNR-Y↑</td><td>SSIM-Y↑</td><td>LPIPS↓</td><td>CLIPIQA↑</td><td>MUSIQ↑</td></tr><tr><td>Baseline</td><td>21.312</td><td>0.7272</td><td>0.2764</td><td>0.4538</td><td>47.80</td></tr><tr><td>+ Spatial color loss</td><td>21.681</td><td>0.7356</td><td>0.2543</td><td>0.4229</td><td>44.97</td></tr><tr><td>+ Freq-decomposed loss</td><td>21.891</td><td>0.7562</td><td>0.2176</td><td>0.4107</td><td>43.52</td></tr></table>

Table 1 presents the ablation. The baseline with only $\mathcal { L } _ { \mathrm { A R } }$ achieves 21.312 dB PSNR-Y. Adding spatial color loss improves PSNR-Y by 0.37 dB; replacing it with the frequency-decomposed loss yields 21.891 dB (+0.58 dB total), SSIM-Y 0.7562, and LPIPS 0.2176 (21.3% reduction). The monotonic decrease in noreference scores is expected: stronger DSLR-target supervision pulls outputs from generic aesthetic preferences—an acceptable trade-of for faithful reconstruction.

Table 2: Published ZRR landscape under nonidentical evaluation protocols. Priorwork values are RGB-channel results quoted from the corresponding papers, whereas ours is evaluated on the Y channel over the full test set. The two blocks provide context and must not be numerically ranked across protocols.
<table><tr><td>Method</td><td>Evaluation</td><td>PSNR↑ SSIM↑LPIPS↓</td><td></td></tr><tr><td>PyNet [5]</td><td>RGB, published 21.19</td><td>0.747</td><td>0.193</td></tr><tr><td>AWNet-R [1] RGB, published</td><td>21.42</td><td>0.748</td><td>0.198</td></tr><tr><td>AWNet-D [1] RGB, published</td><td>21.53</td><td>0.749</td><td>0.212</td></tr><tr><td>MW-ISPNet [4] RGB, published</td><td>21.42</td><td>0.754</td><td>0.213</td></tr><tr><td>LiteISPNet [16] RGB, published</td><td>21.55</td><td>0.749</td><td>0.187</td></tr><tr><td>FourierISP [2] RGB, published</td><td>21.65</td><td>0.755</td><td>0.182</td></tr><tr><td>DiffRAW [14] RGB, published</td><td>21.31</td><td>0.743</td><td>0.145</td></tr><tr><td>ISPDiffuser [11] RGB, published</td><td>21.77</td><td>0.754</td><td>0.157</td></tr><tr><td>Ours</td><td>Y, ours 21.89</td><td>0.756</td><td>0.218</td></tr></table>

Table 2 provides landscape context rather than a head-to-head ranking. Because the published baselines use RGB-channel PSNR/SSIM while our retained evaluation uses the Y channel, diferences between the two blocks are not directly attributable to model quality. Our quantitative claims therefore rest on the matched-protocol ablation in Table 1; a unified RGB re-evaluation of all methods remains necessary for strict comparison. The published LPIPS values nevertheless expose the main practical gap: our LPIPS (0.218) trails FourierISP (0.182) and difusion methods (DifRAW: 0.145; ISPDifuser: 0.157), consistent with the residual color and tone errors observed qualitatively.

## 4.3 Qualitative Analysis

![](images/cd61deb56245096a8767ec07925a3f9ada8a657cb278e1235c4c65e6c94e64b6.jpg)  
Fig. 2: Qualitative results on ZRR. Each row: RAW input (left), our output (middle), ground truth (right). The model preserves structure and edges; residual errors are global color, brightness, and contrast shifts.

Figure 2 shows representative outputs. The model recovers plausible scene structure and edge detail from heavily color-shifted RAW inputs. Object boundaries, texture patterns, and spatial layout are faithfully preserved. However, remaining errors are predominantly global: outputs often appear too gray, too bright, or with lower contrast than the DSLR target—characteristic of whitebalance and tone-curve mismatches rather than texture or structural failures. This visual pattern motivates the color correction diagnostics below.

## 4.4 VQ-VAE Reconstruction Ceiling

Before analyzing the autoregressive model’s errors, we characterize the VQ-VAE fidelity ceiling. Figure 3 compares ground-truth ZRR targets with their VQ-VAE encode-decode reconstructions (bypassing the VAR transformer). The frozen VQ-VAE preserves scene layout, boundaries, and color with high fidelity. Quantization artifacts appear primarily in smooth gradients (sky, walls, defocused backgrounds) where the discrete codebook cannot represent every tonal variation. These artifacts are substantially smaller than errors from autoregressive prediction, confirming that the K = 4096 codebook is not the bottleneck—the dominant error source is RAW-conditioned code selection.

![](images/0e4ce49ecb091c4c37bc0076ad2d35fda474677d0722198cdf96f4d40c3835a0.jpg)  
Fig. 3: VQ-VAE reconstruction ceiling. Each row: ground truth (left), VQ-VAE encodedecode reconstruction (right). The codebook preserves structure, color, and most detail; artifacts appear only in smooth gradient regions.

## 4.5 Color Correction Diagnostics

Table 3: Post-hoc color diagnostics on a 100-image ZRR subset. The oracle afine uses each test target to fit a per-image afine transform and is not deployable; it estimates the ceiling of color-only correction.
<table><tr><td>Setting</td><td>PSNR-Y↑SSIM-Y↑LPIPS↓</td><td></td><td></td></tr><tr><td>Oracle affine, before Oracle affine, after</td><td>21.129 24.950</td><td>0.7123 0.7269</td><td>0.2892 0.2815</td></tr><tr><td>Low-freq residual, before</td><td>21.681</td><td>0.7357</td><td>0.2543</td></tr><tr><td>Low-freq residual, after</td><td>21.694</td><td>0.7357</td><td>0.2543</td></tr><tr><td>Adaptive CCM, before</td><td>21.678</td><td>0.7356</td><td>0.2543</td></tr><tr><td>Adaptive CCM, after</td><td>21.391</td><td>0.7359</td><td>0.2648</td></tr></table>

Table 3 presents the central diagnostic result. The oracle afine raises PSNR-Y by 3.8 dB (21.13 to 24.95), confirming that the output already contains the structural detail for high-fidelity reconstruction—the missing ingredient is correct color and tone mapping.

Neither learned head realizes this potential. The low-frequency residual head provides only 0.013 dB, suggesting spatially smooth additive correction is insufficient. The adaptive CCM actually worsens LPIPS from 0.254 to 0.265, indicating overcorrection. We attribute this to: (1) substantial per-image variation in exposure, white balance, and illumination in ZRR; (2) misalignment limiting supervisory signal for precise color correspondences; and (3) teacher matrices computed against misaligned targets introducing systematic noise.

![](images/c6269865a56437d0226d67cb0a98beea4930f136274a9c2fe80db0f2a49a1a8b.jpg)  
Fig. 4: Oracle afine correction. Each row, left to right: RAW input, model output, oracle-corrected output, ground truth. The oracle recovers target color and contrast, confirming the output contains necessary structural detail.

Figure 4 provides visual evidence. The model output captures scene content faithfully but exhibits color and brightness shift. After oracle correction, the output closely matches the ground truth, confirming that the VAR prior has solved structure reconstruction and the remaining gap is a learnable but currently unsolved color transform.

## 5 Discussion

Our experiments demonstrate that VAR priors efectively capture image structure for ISP, producing outputs with coherent edges, textures, and layout. However, the discrete codebook should not be expected to simultaneously solve all continuous camera operations through code selection alone. This echoes the design principle shared by recent high-performing methods: FourierISP [2] separates structure-related phase from color-related amplitude, and ISPDifuser [11] assigns difusion to grayscale texture with a separate module for color consistency.

The 3.8 dB oracle gap (Table 3) represents a substantial opportunity. A natural next step is a color-modulated architecture in which a compact color state is predicted from RAW statistics and injected into the VQ decoder or VAR attention blocks through adaptive normalization, cross-attention bias, or lowrank codebook ofsets. This keeps the discrete prior responsible for geometry and texture while providing a continuous path for exposure, white balance, and target-camera style. The failure of post-hoc correction (Table 3) suggests this modulation should be trained jointly with the autoregressive objective, rather than applied after quantization and decoding.

Limitations. The model operates at 256×256, below the native resolution of many practical ISP applications; scaling would require tiling with overlap blending or a resolution-adapted VAR architecture. The frozen codebook was trained on ImageNet and may not optimally represent specific camera color gamuts; fine-tuning or expanding the codebook for ISP is a promising direction. We evaluate on a single benchmark (ZRR); generalization to other sensor–target pairs remains to be verified. Moreover, Table 2 combines results reported under different channel protocols and is contextual rather than a strict ranking; a unified evaluation is required for direct comparison. The 1.10 B-parameter backbone is impractical for mobile deployment, though it serves our diagnostic purpose.

## 6 Conclusion

We presented the first study of visual autoregressive priors for RAW-to-sRGB ISP. A frozen 1.10 B-parameter VAR backbone, adapted with only 32.93 M trainable parameters via conditioning embeddings and cross-attention, produces detailed sRGB reconstructions on ZRR. The frequency-decomposed color loss improves full-test fidelity to 21.89 dB PSNR-Y and 0.218 LPIPS. Oracle and learned color-correction experiments confirm that the remaining bottleneck is continuous color transfer rather than detail synthesis, with oracle afine correction recovering 3.8 dB. These findings motivate future VAR-based ISP architectures combining discrete structure generation with jointly trained continuous color modulation.

## Acknowledgments

This work was supported by the OmniVision-IDT Joint Laboratory for Intelligent Image Sensing under the project “Research on Frontier Technologies of Intelligent Image Sensing”. The computing for this research was supported by High Performance Computing Platform at Eastern Institute of Technology, Ningbo.

## References

1. Dai, L., Liu, X., Li, C., Chen, J.: AWNet: Attentive wavelet network for image ISP. In: ECCV Workshops. pp. 185–201 (2020)

2. He, X., Hu, T., Wang, G., Wang, Z., Wang, R., Zhang, Q., Yan, K., Chen, Z., Li, R., Xie, C., et al.: Enhancing RAW-to-sRGB with decoupled style structure in fourier domain. In: AAAI Conf. Artif. Intell. vol. 38, pp. 2130–2138 (2024)

3. Ignatov, A., Chiang, C.M., Kuo, H.K., Sycheva, A., Timofte, R., Chen, M.H., Lee, M.Y., Xu, Y.S., Tseng, Y., Xu, S., Guo, J., Chen, C.H., Hsyu, M.C., Tsai, W.C., Chen, C.W., Malivenko, G., Kwon, M., Lee, M., Yoo, J., Kang, C., Wang, S., Shaolong, Z., Dejun, H., Fen, X., Zhuang, F., Ma, Y., Peng, J., Wang, T., Song, F., Hsu, C.C., Chen, K.L., Wu, M.H., Chudasama, V., Prajapati, K., Patel, H., Sarvaiya, A., Upla, K., Raja, K., Ramachandra, R., Busch, C., de Stoutz, E.: Learned smartphone isp on mobile npus with deep learning, mobile ai 2021 challenge: Report. In: IEEE Conf. Comput. Vis. Pattern Recog. Worksh. pp. 2503– 2514 (2021)

4. Ignatov, A., Timofte, R., Zhang, Z., Liu, M., Wang, H., Zuo, W., Zhang, J., Zhang, R., Peng, Z., Ren, S., et al.: AIM 2020 challenge on learned image signal processing pipeline. In: ECCV Workshops. pp. 152–170 (2020)

5. Ignatov, A., Van Gool, L., Timofte, R.: Replacing mobile camera isp with a single deep learning model. arXiv preprint arXiv:2002.05509 (2020)

6. Li, F., Hou, W., Jia, P.: RMFA-Net: A neural ISP for real RAW to RGB image reconstruction. arXiv preprint arXiv:2406.11469 (2024)

7. Loshchilov, I., Hutter, F.: Decoupled weight decay regularization. In: Int. Conf. Learn. Represent. (2019)

8. van den Oord, A., Vinyals, O., Kavukcuoglu, K.: Neural discrete representation learning. In: Adv. Neural Inform. Process. Syst. pp. 6306–6315 (2017)

9. Qu, Y., Yuan, K., Hao, J., Zhao, K., Xie, Q., Sun, M., Zhou, C.: Visual autoregressive modeling for image super-resolution. In: Int. Conf. Mach. Learn. (2025)

10. Rajagopalan, S., Narayan, K., Patel, V.M.: RestoreVAR: Visual autoregressive generation for all-in-one image restoration. arXiv preprint arXiv:2505.18047 (2025)

11. Ren, Y., Jiang, H., Yang, M., Li, W., Liu, S.: ISPDifuser: Learning RAW-tosRGB mappings with texture-aware difusion models and histogram-guided color consistency. In: AAAI Conf. Artif. Intell. vol. 39, pp. 6722–6730 (2025)

12. Tian, K., Jiang, Y., Yuan, Z., Peng, B., Wang, L.: Visual autoregressive modeling: Scalable image generation via next-scale prediction. In: Adv. Neural Inform. Process. Syst. (2024)

13. Uhm, K.H., Kim, S.W., Ji, S.W., Cho, S.J., Hong, J.P., Ko, S.J.: W-net: Two-stage u-net with misaligned data for raw-to-rgb mapping. In: Int. Conf. Comput. Vis. Worksh. pp. 3636–3642 (2019)

14. Yi, M., Zhang, K., Liu, P., Zuo, T., Tian, J.: DifRAW: Leveraging difusion model to generate DSLR-comparable perceptual quality sRGB from smartphone RAW images. In: AAAI Conf. Artif. Intell. vol. 38, pp. 6711–6719 (2024)

15. Zhang, R., Isola, P., Efros, A.A., Shechtman, E., Wang, O.: The unreasonable efectiveness of deep features as a perceptual metric. In: IEEE Conf. Comput. Vis. Pattern Recog. pp. 586–595 (2018)

16. Zhang, Z., Wang, H., Liu, M., Wang, R., Zuo, W., Zhang, J.: Learning raw-to-srgb mappings with inaccurately aligned supervision. In: Int. Conf. Comput. Vis. pp. 4348–4358 (2021)