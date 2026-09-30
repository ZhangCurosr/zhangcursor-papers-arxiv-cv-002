# FastVR: Efficient Streaming Video Restoration with One-Step Diffusion

Xiaoxu Chen<sup>1,\*</sup> Qin Yang<sup>1,2,\*</sup> Haoran Bai<sup>1</sup> Sibin Deng<sup>1</sup> Ying Chen<sup>1,†</sup>

<sup>1</sup>Alibaba Group <sup>2</sup>Xidian University

Project Page https://chenxx89.github.io/projects/fastvr/

GitHub https://github.com/chenxx89/FastVR

Hugging Face https://huggingface.co/chenxx89/FastVR

![](images/2886ee69e094ef056ece12c6e5c4653eda1fa2282548cf9ef8599a1aca59ebb9.jpg)

![](images/b3dbe8993ba1a00a57458dd43e6ab2f4c2b48aa154093e38480aaeeea851f9d7.jpg)  
Peak GPU Memory

![](images/d12837a814d16936ce8f2b5a90119412b29aadb10896cd4a0e552ffe56fcb9ea.jpg)  
Figure 1 | Visual quality and inference efficiency of FastVR. Left: A restoration example and zoomedin comparisons with existing methods. Right: Inference speed and peak GPU memory usage when processing a 120-frame 1080p video on a single NVIDIA H20 GPU.

## Abstract

Diffusion-based video restoration recovers realistic details, but its practical deployment is limited by two efficiency bottlenecks: costly VAE encoding and decoding, and the quadratic cost of full self-attention in diffusion transformers (DiTs). This paper presents FastVR, a streaming video restoration framework built on a one-step diffusion model, which delivers strong restoration quality and temporal consistency while processing 1080p video at 11 FPS on a single H20 GPU. To improve inference efficiency, FastVR combines a lightweight VAE with chunk-wise causal attention, which substantially reduces the computational cost. During training, it further adopts velocity consistency regularization and continuous trajectory learning, which improve restoration quality. Extensive experiments show that FastVR is more efficient than the evaluated diffusion baselines while achieving state-of-the-art performance on synthetic and real-world benchmarks. We hope that this work supports further progress in the community.

## 1. Introduction

Video restoration aims to recover visually faithful and temporally consistent content from videos degraded by blur, noise, compression, downsampling, and other artifacts. In practical deployment, high resolution long videos generally need to be processed with low latency under limited computation and memory budgets. Although traditional reconstruction-based methods [4, 5] can achieve high accuracy, they tend to produce overly smooth textures when severe degradation removes high-frequency details. In recent years, diffusion-based video restoration [7, 16, 2, 28] has attracted considerable attention, since learned generative priors can restore realistic textures and fine details. Image diffusion priors improve spatial detail, but they do not model temporal dynamics explicitly and can therefore break temporal consistency across frames. Video diffusion models alleviate this limitation by modeling motion and appearance jointly. STAR [19] adapts a text-to-video prior to real-world video super-resolution, improving degradation removal while preserving temporal consistency. Vivid-VR [2] further shows that a large text-to-video DiT can generate photorealistic textures without sacrificing temporal consistency. These studies suggest that video diffusion priors provide a strong basis for recovering spatial detail and maintaining temporal consistency, although the gains come at a high computational cost.

The first factor that limits efficiency is the cost of iterative sampling. Conventional methods repeatedly evaluate a large denoising network, so latency grows roughly linearly with the number of sampling steps. Vivid-VR, for example, uses a large video diffusion backbone with 50 inference steps. Recent one-step methods have achieved notable progress. DOVE [7] fine-tunes a pretrained video diffusion model directly for one-step real-world video super-resolution and achieves a large speedup over multi-step baselines. However, the authors also observe that endpoint regression tends to yield over-smoothed results, which motivates an additional refinement stage in pixel space. SeedVR2 [16] combines progressive distillation with adversarial post-training in order to preserve restoration capability after compressing a multistep process into a single step. DUO-VSR [13] strengthens one-step optimization through distribution matching, feature-level adversarial supervision, and preference refinement. Although the one-step methods described above improve perceptual quality, they discard the progressive correction mechanism of diffusion sampling and remain expensive on high-resolution videos.

One-step modeling removes much of this cost, but two other system-level bottlenecks then become dominant. In the DiT, the cost of global self-attention grows quadratically with the number of tokens. Moreover, although latent compression by the VAE reduces the denoising cost, reconstruction through a large causal video VAE can dominate the overall latency. SeedVR2 [16] reports that the causal video VAE accounts for more than 95% of the total runtime on a 100-frame 720p video, and DUO-VSR [13] likewise identifies VAE processing as the main overhead in a one-step pipeline. Reducing the number of sampling steps is therefore not sufficient for high-resolution restoration. Recently, several streaming methods have adopted comprehensive solutions to further improve inference efficiency. FlashVSR [28] introduces causal sparse attention, KV cache, one-step distillation, and a lightweight decoder, reaching near real-time throughput on long videos. The analysis in FlashVSR also shows that VAE decoding becomes a dominant component once denoising is compressed into a single step, while dense attention remains inefficient at high resolution. Such results reshape the central research question: the challenge is no longer only to reduce the number of diffusion steps, but to achieve a better end-to-end balance among autoencoding cost, attention complexity, optimization stability, restoration fidelity, and temporal consistency.

We present FastVR, a one-step diffusion framework for high-quality streaming video restoration that targets the two costs which become dominant after sampling acceleration. At the representation level, FastVR adopts a lightweight VAE to reduce the cost of encoding degraded frames and decoding restored frames. At the denoising level, FastVR combines spatial tiling with chunk-wise causal attention to limit spatial computation and temporal attention context, reducing token interaction costs and supporting online processing without access to future chunks. To compensate for the limited use of progressive correction in a one-step mapping, FastVR constructs a continuous restoration trajectory between lowquality and high-quality videos, so that the model observes intermediate states along the transformation from a degraded video to a clean one rather than a single supervised endpoint. FastVR further imposes an explicit velocity consistency regularizer across sampled trajectory points, which encourages compatible restoration velocities at different sampling positions and constrains the learned vector field, thereby improving the optimization stability of one-step prediction. FastVR runs at 11 frames per second on 1080p video with a single NVIDIA H20 GPU, while maintaining strong restoration quality and temporal consistency. In summary, our main contributions are as follows:

• We design an efficient framework for high-resolution streaming video restoration, in which a lightweight VAE and chunk-wise causal attention reduce the encoding, decoding, and attention costs of one-step inference.

• We introduce explicit velocity consistency regularization into the learning of continuous restoration trajectories, which improves both the optimization stability and the detail recovery ability of the one-step model.

• Extensive experiments on synthetic and real-world benchmarks show that FastVR attains competitive restoration quality and temporal consistency while reaching 11 FPS on 1080p video with a single H20 GPU.

Figure 1 illustrates the restoration quality and inference efficiency of FastVR compared with existing methods.

## 2. Related Work

## 2.1. Diffusion-based Video Restoration.

Conventional VSR methods trained on synthetic or composite degradations [4, 5] can still struggle to recover realistic fine details under severe degradation. Diffusion models [9, 14] instead bring a strong generative prior to this task. This prior proves highly effective in image restoration [18, 24, 25], where realistic details can be faithfully synthesized. However, extending this success to video is not straightforward, as temporal consistency must be preserved in addition to spatial fidelity. Early attempts therefore attached temporal modules to pretrained image backbones: Upscale-A-Video [27] propagates latents along optical-flow trajectories, MGLD-VSR [20] steers sampling with motion-aware losses, and DiffVSR [10] augments the backbone with multi-scale temporal attention and a staged training scheme. However, the underlying 2D prior limits their robustness under severe spatiotemporal corruption. This limitation has driven a shift toward video-native Diffusion Transformers, whose pretraining on large-scale text-to-video data [21, 15] provides a stronger motion prior. Building on this foundation, SeedVR [17], STAR [19], and Vivid-VR [2] introduce restoration-specific architectures and objectives, and currently set the state of the art in detail fidelity.

## 2.2. Video Diffusion Acceleration.

Despite this fidelity, the iterative denoising of multi-step sampling imposes prohibitive inference latency, especially at high resolutions. To reduce this cost, a line of work condenses the sampling process into a single forward pass via distillation [23, 22], adversarial post-training [11], or rectified-flow techniques [12], which have recently been extended to video restoration. DOVE [7] trains a text-to-video model for direct one-step generation through a two-stage latent-to-pixel scheme, SeedVR2 [16] reaches a single step via progressive distillation and adversarial post-training, and DUO-VSR [13] unifies distribution matching with adversarial supervision through dual-stream distillation. Even after the sampling steps are compressed to one, the VAE and the bidirectional attention in the DiT become the dominant latency sources in high-resolution super-resolution. To address this, FlashVSR [28] couples one-step distillation with sparse causal attention and a compact decoder, forming the first diffusion-based streaming framework toward real-time VSR.

## 3. Method

FastVR is a one-step streaming restoration model built on Wan2.2-TI2V-5B [15]. As illustrated in Fig. 2, it combines a lightweight VAE with a chunk-wise causal attention pattern to reduce the two dominant costs of high-resolution video restoration: the cost of the VAE and the quadratic cost of bidirectional attention. During inference, a frozen lightweight encoder first maps a low-quality (LQ) video into a compact latent representation, the DiT predicts a restoration velocity in one forward pass per temporal chunk and spatial tile, and a frozen lightweight decoder reconstructs the restored video from the updated latents and the LQ video. To learn the restoration mapping, the DiT is trained in two stages using the frozen Wan VAE. Stage I learns a continuous restoration trajectory in the latent space and regularizes the predicted velocity along a model-induced trajectory. Stage II then optimizes the model under pixel-space supervision, targeting details that latent-space regression alone does not preserve.

![](images/f8472d6fee9d3ac02d5d7b8c9f4b759b6f04f483d1e57d0c1e3d2c077836a11c.jpg)  
Figure 2 | Overview of FastVR. Top: one-step streaming inference with a lightweight VAE and chunk-wise causal attention, with LQ frames conditioning the decoder. Bottom: two-stage DiT training with the frozen Wan VAE, using latent-space trajectory learning and velocity consistency in Stage I, followed by pixel-space $\ell _ { 1 }$ and DISTS supervision in Stage II.

## 3.1. Model Architecture

Lightweight VAE. To reduce the encoding and decoding overhead during inference, we adopt a lightweight VAE based on lightweight autoencoding [3] and conditional decoding [28]. Its encoder maps LQ frames into a latent representation compatible with the pretrained Wan DiT. The decoder reconstructs the video from the restored latents while using the LQ frames as an additional condition. This supplies structural information directly from the input and reduces the burden of reconstructing the video solely from compressed latents. We train the lightweight VAE separately from the DiT, using a combination of pixel-wise $\ell _ { 1 }$ and LPIPS perceptual losses [26] between the decoded output and the original HQ video, balancing reconstruction fidelity and perceptual quality. Both encoding and decoding support incremental temporal processing with cached features, allowing bounded intermediate memory during streaming inference. The lightweight VAE is frozen for inference, while the two-stage DiT training uses the original frozen Wan VAE encoder and decoder.

Chunk-wise causal attention. The pretrained Wan DiT employs bidirectional self-attention, whose complexity is quadratic in video length. We reduce this cost by partitioning the latent sequence into non-overlapping chunks of � temporal positions and restricting each chunk to attend only to itself and its immediate predecessor. Let � denote the number of latent frames and $i , j \in \{ 0 , \ldots , T - \overset { \cdot } { 1 } \}$ } the temporal indices of a query and a key, respectively. We define the temporal visibility mask $\mathbf { M } \in \{ 0 , 1 \} ^ { T \times T }$ as

$$
M _ { i , j } = \left\{ \begin{array} { l l } { 1 , } & { \lfloor i / k \rfloor - \lfloor j / k \rfloor \in \{ 0 , 1 \} , } \\ { 0 , } & { \mathrm { o t h e r w i s e } . } \end{array} \right.\tag{1}
$$

The mask is applied in every self-attention layer, preserving bidirectional attention within each chunk while limiting temporal context to one preceding chunk.

During training, the entire latent sequence is processed in parallel, with M enforcing the chunk-wise causal attention masks. During inference, we sequentially restore video chunks using a streaming inference scheme and employ KV caching for efficient computation. For this sequential inference scheme, with fixed chunk size � and spatial resolution, the temporal attention cost scales as $O ( T k )$ , compared with $O ( T ^ { 2 } )$ for bidirectional attention, while the KV-cache size remains $O ( k )$ independent of video length.

## 3.2. Two-Stage Latent–Pixel Training

We optimize FastVR in two stages while keeping all encoder and decoder parameters fixed. The first stage learns the LQ-to-HQ transformation as a continuous vector field in the latent space, whereas the second stage directly optimizes the decoded result produced by one-step prediction. This design first establishes the restoration mapping under efficient latent-space supervision and subsequently adapts the predicted latent to the reconstruction characteristics of the Wan VAE decoder.

Given an HQ video $\mathbf { x } _ { \mathrm { H Q } } ,$ we synthesize the paired LQ observation $\mathbf { x } _ { \mathrm { L Q } }$ online using the RealBasicVSR degradation pipeline [6]. The LQ and HQ videos are encoded by the frozen Wan VAE encoder [15]:

$$
\begin{array} { r } { \mathbf { z } _ { \mathrm { L Q } } = \mathcal { E } _ { \mathrm { W a n } } ( \mathbf { x } _ { \mathrm { L Q } } ) , \qquad \mathbf { z } _ { \mathrm { H Q } } = \mathcal { E } _ { \mathrm { W a n } } ( \mathbf { x } _ { \mathrm { H Q } } ) . } \end{array}\tag{2}
$$

Stage I: Latent-space trajectory learning. Unlike endpoint regression $[ 7 ] ,$ we supervise the restoration mapping over a continuous path between $\mathbf { z } _ { \mathrm { H Q } }$ and $\mathbf { z } _ { \mathrm { L Q } }$ . Let $\sigma _ { t }$ denote the noise level associated with timestep �. We parameterize the restoration trajectory such that $t = 0$ corresponds to the HQ endpoint $\mathbf { z } _ { \mathrm { H Q } }$ with $\sigma _ { 0 } = 0 ,$ whereas $t = \tau$ corresponds to the LQ endpoint $\mathbf { z } _ { \mathrm { L Q } }$ with noise level $\sigma _ { \tau }$ . The intermediate latent is defined as follows:

$$
{ \bf z } _ { t } = \frac { t } { \tau } { \bf z } _ { \mathrm { L Q } } + \left( 1 - \frac { t } { \tau } \right) { \bf z } _ { \mathrm { H Q } } .\tag{3}
$$

The corresponding velocity with respect to $\sigma _ { t }$ is

$$
{ \bf u } _ { t } = \frac { \partial { \bf z } _ { t } } { \partial \sigma _ { t } } = \frac { { \bf z } _ { \mathrm { L Q } } - { \bf z } _ { \mathrm { H Q } } } { \sigma _ { \tau } } .\tag{4}
$$

We train the DiT velocity predictor $\upsilon _ { \theta }$ using

$$
\mathcal { L } _ { \mathrm { t r a j } } = \mathbb { E } _ { \mathbf { x } _ { \mathrm { H Q } } , t } \left[ \| \nu _ { \theta } ( \mathbf { z } _ { t } , t , \mathbf { c } _ { \emptyset } ) - \mathbf { u } _ { t } \| _ { 2 } ^ { 2 } \right] ,\tag{5}
$$

where $\pmb { c } _ { \emptyset }$ denotes the empty-text condition. Random timestep sampling exposes the model to latent states with varying degradation levels, enabling it to learn restoration velocities throughout the LQ-to-HQ trajectory rather than only the endpoint mapping.

Although $\mathcal { L } _ { \mathrm { t r a j } }$ provides velocity supervision at states sampled from the reference trajectory, the latent update during inference is determined by the model’s own prediction. An inaccurate velocity may therefore displace the predicted latent from the reference trajectory and directly introduce errors into the one-step restoration result. To mitigate this discrepancy, we impose self-distilled velocity consistency along a model-induced trajectory.

Specifically, for a sampled timestep �, we draw $\smash { t ^ { \prime } \sim \mathcal { U } ( 0 , t ) }$ and perform an Euler update from $\sigma _ { t }$ to $\sigma _ { t ^ { \prime } . }$ where $\sigma _ { t ^ { \prime } } < \sigma _ { t } .$

$$
\begin{array} { r l } & { \mathbf { v } _ { t } = \nu _ { \theta } ( \mathbf { z } _ { t } , t , \mathbf { c } _ { \emptyset } ) , } \\ & { \widetilde { \mathbf { z } } _ { t ^ { \prime } } = \mathrm { s g } [ \mathbf { z } _ { t } + ( \sigma _ { t ^ { \prime } } - \sigma _ { t } ) \mathbf { v } _ { t } ] . } \end{array}\tag{6}
$$

Here, sg[·] denotes the stop-gradient operator. We then evaluate the velocity at the propagated state $\widetilde { \mathbf { z } } _ { t ^ { \prime } }$ and use the detached prediction as the self-distillation target:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { v c } } = \mathbb { E } _ { ( \mathbf { x } _ { \mathrm { L Q } } , \mathbf { x } _ { \mathrm { H Q } } ) , t , t ^ { \prime } } \left[ \left\| \mathbf { v } _ { t } - \mathrm { s g } \big [ \nu _ { \theta } \big ( \widetilde { \mathbf { z } } _ { t ^ { \prime } } , t ^ { \prime } , \mathbf { c } _ { \emptyset } \big ) \big ] \right\| _ { 2 } ^ { 2 } \right] . } \end{array}\tag{7}
$$

This objective encourages consistent velocity predictions between a reference state and the state reached by the model’s own update. The overall objective for Stage I is

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { I } } = \mathcal { L } _ { \mathrm { t r a j } } + \mathcal { L } _ { \mathrm { v c } } . } \end{array}\tag{8}
$$

Stage II: Pixel-space refinement. Stage I provides latent-space supervision but does not directly account for the reconstruction characteristics of the VAE decoder. Because the mapping from the latent space to the pixel space is nonlinear, a small latent-space error does not necessarily imply high-fidelity reconstruction in the pixel space. We therefore perform pixel-space fine-tuning on the decoded output of the one-step prediction.

Starting from $\mathbf { z } _ { \mathrm { L Q } }$ at $t = \tau$ , the restored latent is computed using the same update as that employed during inference:

$$
\widehat { \mathbf { z } } = \mathbf { z } _ { \mathrm { L Q } } - \sigma _ { \tau } \nu _ { \theta } \big ( \mathbf { z } _ { \mathrm { L Q } } , \tau , \mathbf { c } _ { \otimes } \big ) .\tag{9}
$$

The restored video is then obtained using the frozen Wan VAE decoder:

$$
{ \widehat { \mathbf { x } } } = G _ { W \mathrm { a n } } ( { \widehat { \mathbf { z } } } ) .\tag{10}
$$

We supervise the decoded output using an equally weighted combination of the pixel-wise $\ell _ { 1 }$ loss and the DISTS perceptual loss [8]:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { I I } } = \mathcal { L } _ { \ell _ { 1 } } ( \widehat { \mathbf { x } } , \mathbf { x } _ { \mathrm { H Q } } ) + \mathcal { L } _ { \mathrm { D I S T S } } ( \widehat { \mathbf { x } } , \mathbf { x } _ { \mathrm { H Q } } ) . } \end{array}\tag{11}
$$

The Wan VAE decoder remains frozen, while gradients are propagated through it to update the DiT parameters �. Stage II therefore aligns the latent prediction with the pixel-space reconstruction objective without modifying the inference architecture.

## 4. Experiments

In this section, we evaluate the proposed FastVR on synthetic and real-world benchmarks and compare it with state-of-the-art methods.

## 4.1. Implementation Details

Training dataset. To ensure high visual quality, we construct a training set of approximately 80,000 videos filtered based on resolution, scene transitions, and no-reference quality scores. During training, we resize each video so that its shorter side is 1024 pixels, then center-crop it to 1024 × 1024 pixels. In both training stages, we sample 45-frame clips with randomly selected temporal strides and use the RealBasicVSR degradation pipeline to synthesize the corresponding low-quality clips on the fly.

Optimization. We initialize the DiT from Wan2.2-TI2V-5B and use empty text prompts during training. We use the AdamW optimizer with $( \beta _ { 1 } , \beta _ { 2 } ) = ( 0 . 9 , 0 . 9 5 )$ and train in bfloat16 precision on 32 NVIDIA H100-80G GPUs, with a global batch size of 32. The learning rates are $2 \times 1 0 ^ { - 5 }$ and $1 \times 1 0 ^ { - 5 }$ for the first and second stages, respectively. We train for approximately 10,000 iterations in total across the two stages, using learning rate warm-up followed by cosine decay. We also use gradient checkpointing and apply gradient clipping with a threshold of 1.

Inference. We perform a single step from the point in the shifted schedule closest to timestep 399. We run the DiT with chunk-wise causal attention along the temporal dimension, using a chunk size of 3 latent frames. For spatial tiling, we use overlapping 64 × 64 latent tiles. The latency includes the execution time of all enabled modules but excludes file I/O and model loading.

## 4.2. Evaluation and Metrics.

We compare FastVR with state-of-the-art video restoration methods, including STAR [19], SeedVR-7B [17], Vivid-VR [2], DOVE [7], SATB-VR (one-step) [1], SeedVR2-7B [16], and the Tiny and Full variants of FlashVSR [28]. We use the same benchmarks as Vivid-VR, covering both synthetic datasets (SPMCS, UDM10, and YouHQ40) and real-world datasets (VideoLQ and UGC50). For real-world videos without ground-truth references, we use no-reference image quality metrics (NIQE, MANIQA, MUSIQ, and CLIP-IQA) and the video quality metric DOVER. For the synthetic benchmarks, we additionally report the full-reference metrics PSNR, SSIM, and LPIPS.

<table><tr><td>Datasets</td><td>Metrics</td><td>STAR</td><td>SeedVR Vivid-VR</td><td></td><td>DOVE</td><td>SATB-VR SeedVR2</td><td></td><td>FlashVSR- FlashVSR- Tiny</td><td>Full</td><td>Ours</td></tr><tr><td rowspan="7">SPMCS</td><td>PSNR ↑</td><td>24.18</td><td>24.08</td><td>21.73</td><td>24.80</td><td>24.18</td><td>26.07</td><td>23.57</td><td>23.44</td><td>23.50</td></tr><tr><td>SSIM ↑</td><td>0.720</td><td>0.689</td><td>0.604</td><td>0.754</td><td>0.707</td><td>0.777</td><td>0.675</td><td>0.670</td><td>0.662</td></tr><tr><td>LPIPS ↓</td><td>0.301</td><td>0.263</td><td>0.278</td><td>0.168</td><td>0.197</td><td>0.191</td><td>0.223</td><td>0.226</td><td>0.221</td></tr><tr><td>NIQE↓</td><td>7.058</td><td>4.514</td><td>3.457</td><td>4.031</td><td>4.047</td><td>4.969</td><td>3.496</td><td>3.278</td><td>3.505</td></tr><tr><td>MANIQA↑</td><td>0.229</td><td>0.315</td><td>0.410</td><td>0.346</td><td>0.384</td><td>0.305</td><td>0.361</td><td>0.381</td><td>0.400</td></tr><tr><td>MUSIQ↑</td><td>30.62</td><td>56.99</td><td>70.03</td><td>63.29</td><td>67.82</td><td>53.23</td><td>66.27</td><td>67.91</td><td>71.23</td></tr><tr><td>CLIP-IQA ↑</td><td>0.254</td><td>0.347</td><td>0.483</td><td>0.410</td><td>0.514</td><td>0.325</td><td>0.512</td><td>0.571</td><td>0.586</td></tr><tr><td></td><td>DOVER↑</td><td>4.266</td><td>9.779</td><td>11.35</td><td>9.898</td><td>10.65</td><td>8.625</td><td>10.33</td><td>10.38</td><td>11.71</td></tr><tr><td rowspan="8">UDM10</td><td>PSNR ↑</td><td>27.29</td><td>27.80</td><td>24.54</td><td>30.53</td><td>28.67</td><td>29.04</td><td>26.82</td><td>26.36</td><td>28.76</td></tr><tr><td>SSIM ↑</td><td>0.855</td><td>0.848</td><td>0.761</td><td>0.894</td><td>0.859</td><td>0.884</td><td>0.806</td><td>0.797</td><td>0.842</td></tr><tr><td>LPIPS↓</td><td>0.167</td><td>0.148</td><td>0.243</td><td>0.101</td><td>0.150</td><td>0.117</td><td>0.172</td><td>0.182</td><td>0.154</td></tr><tr><td>NIQE↓</td><td>6.072</td><td>5.345</td><td>4.046</td><td>5.055</td><td>4.283</td><td>5.641</td><td>3.941</td><td>3.779</td><td>3.742</td></tr><tr><td>MANIQA↑</td><td>0.260</td><td>0.264</td><td>0.359</td><td>0.296</td><td>0.381</td><td>0.262</td><td>0.341</td><td>0.364</td><td>0.381</td></tr><tr><td>MUSIQ↑</td><td>45.38</td><td>50.29</td><td>64.71</td><td>55.17</td><td>65.83</td><td>48.91</td><td>62.49</td><td>65.07</td><td>67.50</td></tr><tr><td>CLIP-IQA ↑</td><td>0.289</td><td>0.273</td><td>0.426</td><td>0.340</td><td>0.507</td><td>0.272</td><td>0.494</td><td>0.556</td><td>0.568</td></tr><tr><td>DOVER↑</td><td>9.454</td><td>9.349</td><td>11.97</td><td>10.41</td><td>10.98</td><td>8.752</td><td>11.52</td><td>11.60</td><td>11.86</td></tr><tr><td rowspan="8">YouHQ40</td><td>PSNR ↑</td><td>22.92</td><td>22.46</td><td>21.31</td><td>24.10</td><td>23.67</td><td>24.00</td><td>22.77</td><td>22.56</td><td>23.10</td></tr><tr><td>SSIM ↑</td><td>0.657</td><td>0.621</td><td>0.579</td><td>0.688</td><td>0.657</td><td>0.693</td><td>0.608</td><td>0.602</td><td>0.621</td></tr><tr><td>LPIPS↓</td><td>0.433</td><td>0.240</td><td>0.357</td><td>0.283</td><td>0.281</td><td>0.185</td><td>0.300</td><td>0.290</td><td>0.271</td></tr><tr><td>NIQE↓</td><td>6.744</td><td>4.243</td><td>3.410</td><td>4.456</td><td>4.004</td><td>4.576</td><td>3.603</td><td>3.465</td><td>3.207</td></tr><tr><td>MANIQA↑</td><td>0.240</td><td>0.315</td><td>0.372</td><td>0.304</td><td>0.354</td><td>0.314</td><td>0.347</td><td>0.367</td><td>0.380</td></tr><tr><td>MUSIQ↑</td><td>36.36</td><td>61.91</td><td>70.55</td><td>60.65</td><td>67.91</td><td>59.34</td><td>66.87</td><td>69.62</td><td>72.91</td></tr><tr><td>CLIP-IQA ↑</td><td>0.279</td><td>0.360</td><td>0.447</td><td>0.356</td><td>0.486</td><td>0.336</td><td>0.527</td><td>0.590</td><td>0.620</td></tr><tr><td>DOVER↑</td><td>7.868</td><td>14.00</td><td>14.61</td><td>12.52</td><td>13.25</td><td>12.80</td><td>13.70</td><td>13.84</td><td>14.57</td></tr><tr><td rowspan="5">VideoLQ</td><td>NIQE↓</td><td>5.789</td><td>4.994</td><td>4.371</td><td>5.049</td><td>4.260</td><td>5.674</td><td>4.060</td><td>3.892</td><td>3.759</td></tr><tr><td>MANIQA↑</td><td>0.271</td><td>0.223</td><td>0.319</td><td>0.272</td><td>0.356</td><td>0.221</td><td>0.278</td><td>0.299</td><td>0.361</td></tr><tr><td>MUSIQ↑</td><td>50.52</td><td>46.49</td><td>62.47</td><td>55.11</td><td>65.59</td><td>43.41</td><td>57.54</td><td>61.88</td><td>69.17</td></tr><tr><td>CLIP-IQA ↑</td><td>0.265</td><td>0.229</td><td>0.338</td><td>0.271</td><td>0.436</td><td>0.220</td><td>0.348</td><td>0.405</td><td>0.446</td></tr><tr><td>DOVER↑</td><td>8.758</td><td>7.240</td><td>9.743</td><td>8.780</td><td>9.577</td><td>6.331</td><td>8.954</td><td>9.360</td><td>9.761</td></tr><tr><td rowspan="5">UGC50</td><td>NIQE↓</td><td>5.754</td><td>5.662</td><td>4.361</td><td>5.493</td><td>4.672</td><td>6.230</td><td>4.083</td><td>3.887</td><td>3.891</td></tr><tr><td>MANIQA↑</td><td>0.325</td><td>0.262</td><td>0.376</td><td>0.320</td><td>0.402</td><td>0.253</td><td>0.354</td><td>0.372</td><td>0.379</td></tr><tr><td>MUSIQ↑</td><td>55.01</td><td>49.76</td><td>67.61</td><td>57.82</td><td>68.52</td><td>46.12</td><td>63.85</td><td>65.66</td><td>68.92</td></tr><tr><td>CLIP-IQA ↑</td><td>0.353</td><td>0.305</td><td>0.450</td><td>0.353</td><td>0.571</td><td>0.276</td><td>0.516</td><td>0.563</td><td>0.587</td></tr><tr><td>DOVER↑</td><td>10.92</td><td>10.47</td><td>14.46</td><td>11.84</td><td>13.40</td><td>8.209</td><td>13.40</td><td>13.29</td><td>13.86</td></tr></table>

Table 1 | Quantitative comparisons on benchmarks, including synthetic (SPMCS, UDM10, YouHQ40) and real-world (VideoLQ, UGC50) videos. The best and second-best results are marked in bold and underline, respectively. Ties at the displayed precision share the same rank.

## 4.3. Quantitative Results

Table 1 presents comparisons on five synthetic and real-world benchmarks. FastVR achieves the highest MUSIQ and CLIP-IQA scores on all five datasets and ranks first or second in DOVER. It also obtains the best results across all reported metrics on VideoLQ. On synthetic benchmarks, FastVR improves LPIPS over both FlashVSR variants, although DOVE and SeedVR2 generally retain advantages in full-reference metrics. Overall, these results demonstrate that FastVR delivers strong no-reference restoration quality across diverse degradations with a single diffusion step.

## 4.4. Qualitative Results

Figures 3 and 4 present visual comparisons on real-world and synthetic videos. On real-world footage, FastVR reconstructs clearer embroidery, hair strands, and horse-bridle details, while avoiding the pronounced texture artifacts visible in some competing results. It also preserves the subtitle characters more faithfully in the final example, where Vivid-VR and the FlashVSR variants introduce visible distortions. In the synthetic examples, FastVR recovers fine fur and fabric detail while maintaining clear object contours. Compared with the FlashVSR variants, it produces less granular textures in the illustrated face and knitted garment. Overall, these examples suggest that FastVR achieves a favorable balance between detail recovery and artifact suppression using one-step diffusion inference.

![](images/4476e6f670a4c4cfaa4e4315c1a3c4bf4308e407f1986c36662eb0f91b83d2a4.jpg)  
Figure 3 | Qualitative comparisons on real-world videos, showing clothing, landscapes, portraits, animals, and text. The boxed regions in the input frames are enlarged for comparison. Zoom in for details.

![](images/4a10db17ac1ef7e7108f0f30c865a6f0b716a1227606142ccd1242221e686b1a.jpg)  
Figure 4 | Qualitative comparisons on synthetically degraded videos, showing feathers, illustrated patterns, animal fur, and knitted fabric. The boxed regions in the input frames are enlarged for comparison. Zoom in for details.

## 5. Conclusion

We presented FastVR, a one-step diffusion framework for efficient streaming video restoration. By combining a lightweight VAE with chunk-wise causal attention and KV caching, FastVR reduces autoencoding and denoising overhead while keeping intermediate memory bounded during long-video inference. Its two-stage training combines continuous latent-space trajectory learning and velocity consistency regularization with pixel-space refinement. Experiments on synthetic and real-world benchmarks demonstrate strong no-reference quality and competitive visual detail, together with favorable inference speed and memory usage. These results highlight the potential of jointly designing efficient inference and restoration-oriented training for practical high-resolution video restoration.

## References

[1] Haoran Bai, Xiaoxu Chen, Xiaoyu Liu, Zongsheng Yue, Sibin Deng, Wangmeng Zuo, and Ying Chen. SATB-VR: Training few-step video restoration diffusion model using snr-aware trajectory blending. arXiv preprint arXiv:2606.28677, 2026.

[2] Haoran Bai, Xiaoxu Chen, Canqian Yang, Zongyao He, Sibin Deng, and Ying Chen. Vivid-VR: Distilling concepts from text-to-video diffusion transformer for photorealistic video restoration. arXiv preprint arXiv:2508.14483, 2025.

[3] Ollin Boer Bohan. Taehv: Tiny autoencoder for hunyuan video. https://github.com/madebyo llin/taehv, 2025.

[4] Kelvin C. K. Chan, Xintao Wang, Ke Yu, Chao Dong, and Chen Change Loy. BasicVSR: The search for essential components in video super-resolution and beyond. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4947–4956, 2021.

[5] Kelvin C. K. Chan, Shangchen Zhou, Xiangyu Xu, and Chen Change Loy. BasicVSR++: Improving video super-resolution with enhanced propagation and alignment. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5972–5981, 2022.

[6] Kelvin C.K. Chan, Shangchen Zhou, Xiangyu Xu, and Chen Change Loy. Investigating tradeoffs in real-world video super-resolution. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022.

[7] Zheng Chen, Zichen Zou, Kewei Zhang, Xiongfei Su, Xin Yuan, Yong Guo, and Yulun Zhang. DOVE: Efficient one-step diffusion model for real-world video super-resolution. arXiv preprint arXiv:2505.16239, 2025.

[8] Keyan Ding, Kede Ma, Shiqi Wang, and Eero P. Simoncelli. Image quality assessment: Unifying structure and texture similarity. IEEE Transactions on Pattern Analysis and Machine Intelligence, 44(5):2567–2581, 2020.

[9] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.

[10] Xiaohui Li, Yihao Liu, Shuo Cao, Ziyan Chen, Shaobin Zhuang, Xiangyu Chen, Yinan He, Yi Wang, and Yu Qiao. Diffvsr: Revealing an effective recipe for taming robust video super-resolution against complex degradations. In 2025 IEEE/CVF International Conference on Computer Vision (ICCV), pages 15319–15328. IEEE, 2025.

[11] Shanchuan Lin, Xin Xia, Yuxi Ren, Ceyuan Yang, Xuefeng Xiao, and Lu Jiang. Diffusion adversarial post-training for one-step video generation. In Proceedings of the 42nd International Conference on Machine Learning, volume 267, pages 37959–37974. PMLR, 2025.

[12] Xingchao Liu, Chengyue Gong, and qiang liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. In The Eleventh International Conference on Learning Representations, 2023.

[13] Zhengyao Lv, Menghan Xia, Xintao Wang, and Kwan-Yee K Wong. DUO-VSR: Dual-stream distillation for one-step video super-resolution. arXiv preprint arXiv:2603.22271, 2026.

[14] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456, 2020.

[15] Wan Team, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.

[16] Jianyi Wang, Shanchuan Lin, Zhijie Lin, Yuxi Ren, Meng Wei, Zongsheng Yue, Shangchen Zhou, Hao Chen, Yang Zhao, Ceyuan Yang, Xuefeng Xiao, Chen Change Loy, and Lu Jiang. SeedVR2: One-step video restoration via diffusion adversarial post-training. arXiv preprint arXiv:2506.05301, 2025.

[17] Jianyi Wang, Zhijie Lin, Meng Wei, Yang Zhao, Ceyuan Yang, Chen Change Loy, and Lu Jiang. Seedvr: Seeding infinity in diffusion transformer towards generic video restoration. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025.

[18] Jianyi Wang, Zongsheng Yue, Shangchen Zhou, Kelvin CK Chan, and Chen Change Loy. Exploiting diffusion prior for real-world image super-resolution. International Journal of Computer Vision, 132(12):5929–5949, 2024.

[19] Rui Xie, Yinhong Liu, Penghao Zhou, Chen Zhao, Jun Zhou, Kai Zhang, Zhenyu Zhang, Jian Yang, Zhenheng Yang, and Ying Tai. Star: Spatial-temporal augmentation with text-to-video models for real-world video super-resolution. In 2025 IEEE/CVF International Conference on Computer Vision (ICCV), pages 17108–17118. IEEE, 2025.

[20] Xi Yang, Chenhang He, Jianqi Ma, and Lei Zhang. Motion-guided latent diffusion for temporally consistent real-world video super-resolution. In European conference on computer vision, pages 224–242. Springer, 2024.

[21] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. In International Conference on Learning Representations, volume 2025, pages 83048–83077, 2025.

[22] Tianwei Yin, Michaël Gharbi, Taesung Park, Richard Zhang, Eli Shechtman, Fredo Durand, and William T Freeman. Improved distribution matching distillation for fast image synthesis. In Advances in neural information processing systems, volume 37, pages 47455–47487, 2024.

[23] Tianwei Yin, Michaël Gharbi, Richard Zhang, Eli Shechtman, Fredo Durand, William T Freeman, and Taesung Park. One-step diffusion with distribution matching distillation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6613–6623. IEEE, 2024.

[24] Fanghua Yu, Jinjin Gu, Zheyuan Li, Jinfan Hu, Xiangtao Kong, Xintao Wang, Jingwen He, Yu Qiao, and Chao Dong. Scaling up to excellence: Practicing model scaling for photo-realistic image restoration in the wild. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 25669–25680. IEEE, 2024.

[25] Zongsheng Yue, Kang Liao, and Chen Change Loy. Arbitrary-steps image super-resolution via diffusion inversion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 23153–23163. IEEE, 2025.

[26] Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 586–595, 2018.

[27] Shangchen Zhou, Peiqing Yang, Jianyi Wang, Yihang Luo, and Chen Change Loy. Upscale-A-Video: Temporal-consistent diffusion model for real-world video super-resolution. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2535–2545, 2024.

[28] Junhao Zhuang, Shi Guo, Xin Cai, Xiaohui Li, Yihao Liu, Chun Yuan, and Tianfan Xue. FlashVSR: Towards real-time diffusion-based streaming video super-resolution. arXiv preprint arXiv:2510.12747, 2025.