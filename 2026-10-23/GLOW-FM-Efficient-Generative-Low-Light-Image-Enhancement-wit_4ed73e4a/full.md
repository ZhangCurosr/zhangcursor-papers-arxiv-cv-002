# GLOW-FM: Efficient Generative Low-Light Image Enhancement with One-Step Latent Flow Matching

Ariel Lapid<sup>\*</sup> Yuval Haitman<sup>\*</sup> Reuven Peretz Arnon Netzer Idit Diamant<sup>†</sup> Arm AI Research (AAIR), Arm Ltd

## Abstract

Low-light image enhancement (LLIE) aims to recover wellexposed and perceptually faithful images from severely degraded observations. Recent diffusion-based generative methods achieve strong perceptual quality by leveraging powerful imagepriors, but their iterative sampling incurs substantial computational cost, limiting practical deployment. Despite recent efficiency gains in generative LLIE, high-quality enhancement with a singlefunction evaluation and low overall computation remains challenging, particularly under diverse real-world lowlight conditions. We introduce GLOW-FM, a one-step latent flow-matchingframeworkfor efficient generative LLIE. GLOW-FMperforms conditional transport in the latent space ofafrozen Tiny-VAE and leverages Terminal Velocity Matching (TVM) to enable enhancement with a singlefunction evaluation. Matched comparisons with alternative one-step objectives support TVM as an effective design choice, and, to the best of our knowledge, this is itsfirst adaptation to image restoration and specifically to LLIE. Extensive experiments on challenging real-world benchmarks demonstrate that GLOW-FM achieves competitive image quality and state-of-the-art cross-dataset generalization while requiring only a single function evaluation and 34 GFLOPs for the small variant. Thesefindings highlight the potential of GLOW-FM to bring generative LLIE closer to practical deployment through single-step inference and compact latent modeling. Code is available at https://github.com/armresearch/AAIR-GLOW-FM.

## 1. Introduction

Low-light image enhancement (LLIE) aims to recover well-exposed and visually pleasing images from observations captured under insufficient illumination. Real-world low-light images often exhibit a complex combination of degradations, including severe underexposure, sensor noise, color distortion, reduced contrast, detail loss, and local saturation, which vary significantly across cameras and imaging conditions, making LLIE a challenging problem in real-world imaging [6, 8, 28]. Beyond degrading visual quality, these artifacts also impair downstream computer vision tasks such as object detection and recognition, whose performance declines substantially on poorly exposed inputs [32]. Consequently, LLIE seeks not only to enhance image appearance but also to restore perceptually faithful content that supports reliable high-level vision applications [16, 40].

![](images/8e4a2e86b61ccfe0efd8bd091a0513ce0f6612efb1b40fbc8b8a9cee20dc80d5.jpg)  
Figure 1. GLOW-FM Performance. Comparison with efficient generative LLIE baselines on LSD-DLL. Metrics are min–max normalized across the compared methods, with lower-is-better metrics reversed so that farther from the center is always better; axis annotations show the corresponding absolute metric values. Full results are reported in Tab. 1, while Perception Generalization uses LSD-U NIQE from Tab. 2.

Regression-based LLIE methods provide strong distortion performance and efficient inference [4, 5, 7, 45, 54, 58, 59, 61, 62, 68, 69]. These approaches typically optimize distortionoriented objectives and are effective at recovering visibility and suppressing noise, but often struggle under severe degradation, where substantial information loss makes enhancement inherently ambiguous. Trained with pixel-wise objectives, they tend to produce overly smooth results and suppress fine details [3]. To address this limitation, generative models [17, 19, 21, 22, 55, 65, 66] have emerged as a powerful LLIE paradigm, learning the distribution of natural images and leveraging strong image priors to recover realistic textures and structures that are difficult to reconstruct with deterministic networks. These methods demonstrate substantial improvements in perceptual quality and robustness under severe degradations. However, these gains come at the cost of iterative sampling, resulting in high computational complexity and limiting practical deployment, particularly for high-resolution enhancement in resource-constrained applications such as smartphone camera image processing.

To address this limitation, several efficient generative LLIE frameworks have recently been proposed [20, 25, 70]. These methods significantly reduce the required number of function evaluations (NFEs) compared to conventional diffusion models. Nevertheless, they still rely on iterative refinement procedures and diffusion-inspired architectures whose computational complexity remains substantial. For example, WCDM [20] requires 10 NFEs and approximately 90 GFLOPs during inference, while ReDDiT [25] reduces sampling to 2 NFEs but still requires roughly 128 GFLOPs. Consequently, while recent generative LLIE approaches have substantially reduced sampling cost, achieving high-quality enhancement with a single inference step and a low overall computational budget remains an active challenge, particularly across diverse and demanding real-world low-light conditions.

In this work, we address this challenge by introducing an efficient Generative Low-Light image enhancement with one-step latent Flow Matching (GLOW-FM). Our approach leverages Terminal Velocity Matching (TVM) [71], which enables direct prediction of the normal-light image, eliminating the need for iterative sampling during inference. Comparisons with standard Flow Matching [30, 31], Consistency Flow Matching [63], and MeanFlow [12] establish Terminal Velocity Matching (TVM) [71] as a deliberate design choice for single-step LLIE. To the best of our knowledge, this is the first adaptation of TVM to image restoration, and specifically to LLIE. GLOW-FM learns a conditional transport that maps degraded low-light representations directly to their well-exposed counterparts in the latent space of a frozen Tiny-VAE [49]. Operating in this compact latent space substantially reduces computational cost while preserving the expressive power of generative modeling. Moreover, latent representations capture semantic image structure while suppressing dataset-specific variations, improving restoration and generalization [29, 48, 51].

Extensive experiments on challenging real-world low-light benchmarks [45, 54] demonstrate that GLOW-FM provides an effective generative prior for low-light enhancement while operating under small inference budgets. Despite requiring only a single NFE, our method achieves image quality competitive with recent efficient generative LLIE approaches and consistently exhibits strong cross-dataset generalization as shown in Fig. 1. These results suggest that latent flow matching, combined with one-step TVM estimation, offers a promising direction toward practical, high-quality, and computationally efficient low-light image enhancement.

Our main contributions are summarized as follows:

• We introduce GLOW-FM, an efficient generative framework for low-light image enhancement (LLIE) based on latent flow matching. By performing enhancement in a compact latent space with a lightweight Tiny-VAE and a one-step flow formulation, GLOW-FM achieves high restoration quality at low computational cost.

• We establish Terminal Velocity Matching (TVM) as the strongest overall one-step objective for LLIE through matched comparisons with Rectified Flow, Consistency Flow Matching, and MeanFlow. To the best of our knowledge, this is the first adaptation of TVM to image restoration, specifically LLIE.

• We achieve state-of-the-art cross-dataset generalization compared to generative and non-generative baselines, while maintaining competitive image quality and improved efficiency over existing generative LLIE methods.

## 2. Related Work

## 2.1. Generative Low-Light Image Enhancement

A large body of LLIE research focuses on directly regressing a normal-light image from a low-light input using Retinex decomposition [26], illumination estimation, and deep neural networks [4, 5, 45, 62]. While these approaches are computationally efficient, they often produce overly smooth results and struggle to capture the diversity of plausible normal-light reconstructions. The recent success of diffusion models [18] in image generation has motivated their adoption for high-quality low-light image enhancement [17, 19–21, 25, 65, 70]. Diff-Retinex [65] formulates enhancement as Retinex decomposition followed by conditional diffusion-based reconstruction of reflectance and illumination components. GSAD [19] introduces a global structure-aware diffusion framework that combines curvature-based regularization with uncertainty-guided optimization to preserve scene structure and focus restoration on challenging regions. Subsequent work has largely aimed to reduce the high computational cost of diffusion sampling. ReDDiT [25] uses trajectory matching distillation in a reflectance-aware residual space to reduce the NFEs required at inference. PyDiff [70] performs denoising in a coarse-to-fine resolution pyramid and employs a global correction module to mitigate brightness and color inconsistencies. WCDM [20] accelerates inference by conducting diffusion in the wavelet domain on low-frequency subbands while restoring high-frequency details through a dedicated branch. Reti-Diff [17] generates compact latent Retinex priors to guide a transformer-based restoration network, while LightenDiffusion [21] combines latent Retinex decomposition with diffusion for unsupervised LLIE. Related concurrent preprints, IllumFlow [57] and Consist-Retinex [60], further explore onestep generative LLIE through Retinex-based decomposition.

![](images/e94c2735f1869b9ea2405e5053e39608e30fabd2af145c99f4c8f59e66491288.jpg)  
Figure 2. Overview of GLOW-FM Training. Paired low-light and normally exposed images are encoded into a frozen autoencoder latent space. During training, the model learns both the instantaneous flow-matching velocity and a terminal-velocity displacement map between arbitrary times. During inference, the enhanced latent is obtained with a single step and decoded back to image space.

## 2.2. Generative Model Acceleration

Generative diffusion and flow models have achieved remarkable success across a wide range of image-generation tasks. Their strong priors have also motivated applications to image restoration, where the goal is to recover a clean image from a degraded observation. InDI [11] performs restoration through direct iteration, while IR-SDE [35] models the restoration trajectory using a mean-reverting stochastic process. Bridge-matching methods [46] further formalize transport between degraded and clean endpoint distributions, and flow-based approaches such as FlowIE [73] and PnP-Flow [37] extend this formulation to image restoration. Despite their strong restoration capabilities, these methods commonly require iterative sampling, resulting in substantial inference cost. Early acceleration approaches employed fast post-hoc samplers [23, 34, 47], reducing inference from hundreds of steps to tens, often at the expense of quality under aggressive acceleration. More recently, flow-based objectives have been developed specifically for efficient oneand few-step generation. Rectified Flow [31] learns straighter transport trajectories, while Consistency Flow Matching [63] regularizes the velocity field for few-step inference. ELIR [9] applies Consistency Flow Matching to image restoration in the latent space of a Tiny AutoEncoder. MeanFlow [12] learns mean transport trajectories, whereas TVM [71] learns finite displacements between arbitrary timesteps for high-fidelity one- and few-step generation. Despite these advances, efficient flow-based generation remains relatively unexplored for LLIE. Building upon TVM, we introduce a single-step latent flow-matching framework for efficient low-light enhancement.

## 3. Method

We formulate low-light image enhancement as a conditional flow-matching problem [30] between the low-light distribution $p _ { 1 } ( x _ { 1 } )$ and the normally exposed distribution $p _ { 0 } ( x _ { 0 } )$ . Given a paired sample $( x _ { 1 } , x _ { 0 } )$ , where $x _ { 1 } \sim p _ { 1 } ( x _ { 1 } )$ and $x _ { 0 } \sim p _ { 0 } ( x _ { 0 } )$ denote the low-light source and normal-light target, respectively, we follow the TVM time convention [71]: the source is placed at t = 1 and the target at $t = 0 ,$ opposite to the original Flow Matching convention. Enhancement therefore proceeds toward earlier terminal times, from t=1 to t=0.

Rather than learning this transport directly in pixel space, we perform it in the latent space of a frozen VAE with encoder E and decoder D. We denote the corresponding latents as $z _ { 1 } = \mathcal { E } ( x _ { 1 } )$ and $z _ { 0 } = \mathcal { E } ( x _ { 0 } )$ , and define intermediate states $z _ { t }$ along the flow-path. To enable efficient inference, we train a two-time conditioned model $F _ { \theta } ( z _ { t } , t , s )$ with Terminal Velocity Matching (TVM) [71], which learns displacements along the trajectory rather than relying only on numerical integration of an instantaneous velocity field. This allows the same latent flow model to support both one-step enhancement and few-step refinement. Beyond its reported strong one-step performance, we motivate TVM for restoration through theoretical analysis. Under the TVM regularity conditions [71, Theorem 1] and a Lipschitz assumption on the decoder, we derive a bound on the pixel-space Wasserstein-2 distance [41] between the restored-image and target normal-light distributions in terms of the TVM objectives and autoencoder reconstruction error. This links TVM training to pixel-space restoration; the complete theorem and proof are provided in Appendix B. We next describe TVM’s adaptation to our restoration setting, training and inference, and the design choices enabling efficient generation.

## 3.1. Terminal Velocity Matching

Flow Matching [30] learns an instantaneous velocity field and generates samples by numerically integrating this field over time. To accelerate this process, TVM directly parameterizes the finite transition between two distinct time steps, bypassing the need for small integration steps. For an ideal latent velocity field $u ( z _ { t } , t )$ the flow map from time t to an earlier time $s \leq t$ is given by:

$$
z _ { s } : = \psi ( z _ { t } , t , s ) = z _ { t } + \int _ { t } ^ { s } u ( z _ { r } , r ) d r .\tag{1}
$$

The corresponding net displacement integrates the velocity from t to s, while its derivative at the zero-displacement boundary recovers the instantaneous velocity:

$$
\begin{array} { r } { f ( z _ { t } , t , s ) = \int _ { t } ^ { s } u ( z _ { r } , r ) d r , \quad \partial _ { s } f ( z _ { t } , t , s ) | _ { s = t } = u ( z _ { t } , t ) . } \end{array}\tag{2}
$$

Following TVM, we parameterize the displacement with a two-time network. A scaling is used to enforce zero displacement at the boundary $s = t ,$ , where the instantaneous velocity is also recovered:

$$
\begin{array} { r l } & { f _ { \theta } ( z _ { t } , t , s ) = ( s - t ) F _ { \theta } ( z _ { t } , t , s ) , } \\ & { \left. u _ { \theta } ( z _ { t } , t ) : = \partial _ { s } f _ { \theta } ( z _ { t } , t , s ) \right| _ { s = t } = F _ { \theta } ( z _ { t } , t , t ) . } \end{array}\tag{3}
$$

The boundary relation above recovers standard flow matching when the displacement length goes to zero. For finite-step inference, however, the desired objective is the full displacement from time t to an earlier time s. A direct objective would regress this displacement by:

$$
\mathcal { L } _ { \mathrm { d i s p l } } { = } \mathbb { E } \left[ \left. f _ { \theta } ( z _ { t } , t , s ) - \int _ { t } ^ { s } u ( z _ { r } , r ) d r \right. _ { 2 } ^ { 2 } \right] ,\tag{4}
$$

but this target is unavailable because it requires the true velocity field along the entire trajectory. TVM avoids this explicit integration by using the terminal velocity, defined as the derivative of the displacement with respect to the terminal time s. In the ideal case, this derivative matches the velocity at the terminal point, i.e., $\partial _ { s } f ( z _ { t } , t , s ) = u ( z _ { s } , s )$ for any $s \leq t .$

Since the true terminal point $z _ { s }$ and velocity field are unavailable for the predicted displacement, we use the predicted terminal latent

$$
\hat { z } _ { s } = z _ { t } + f _ { \theta } ( z _ { t } , t , s ) ,\tag{5}
$$

and approximate the terminal velocity with the model velocity,

$$
\begin{array} { r } { \partial _ { s } f _ { \theta } ( z _ { t } , t , s ) \approx u _ { \theta } ( \hat { z } _ { s } , s ) = F _ { \theta } ( \hat { z } _ { s } , s , s ) . } \end{array}\tag{6}
$$

This provides a practical proxy for supervising finite displacements without explicit ODE integration, and it is the basis of the TVM loss described in the following section.

Algorithm 1 Training GLOW-FM   
Require: Paired images $( x _ { 1 } , x _ { 0 } )$ , noise scale $\sigma ,$ encoder $\mathcal { E } ,$   
weights θ, EMA weights <sup>¯</sup>θ   
1: for training step τ do   
2: Encode $z _ { 1 } = \mathcal { E } ( x _ { 1 } )$ and $z _ { 0 } = \mathcal { E } ( x _ { 0 } )$   
3: Sample $\epsilon \sim \mathcal { N } ( 0 , I )$ and set $\tilde { z } _ { 1 } = z _ { 1 } + \sigma \epsilon$   
4: Sample $t \mathrm { \sim } \mathcal { U } [ 0 , 1 ]$ and $s \sim \mathcal { U } [ 0 , t ]$   
5: Form $z _ { t } = ( 1 - t ) z _ { 0 } + t \tilde { z } _ { 1 }$ and $z _ { s } = ( 1 - s ) z _ { 0 } + s \tilde { z } _ { 1 }$   
6: Set target velocity $v = \tilde { z } _ { 1 } - z _ { 0 }$   
7: Compute $\mathcal { L } _ { \mathrm { F M } }$ using Eq. (9)   
8: Compute $\hat { z } _ { s } = z _ { t } + f _ { \theta } ( z _ { t } , t , s )$   
9: Compute target proxy $u _ { \bar { \theta } } ( \hat { z } _ { s } , s ) = F _ { \bar { \theta } } ( \mathrm { s g } ( \hat { z } _ { s } ) , s , s )$   
10: Compute ${ \mathcal { L } } _ { \mathrm { T V M } }$ using Eq. (11)   
11: Update θ with $\mathcal { L } _ { \mathrm { F M } } + \lambda \mathcal { L } _ { \mathrm { T V M } }$   
12: Update EMA parameters $\bar { \theta }$   
13: end for

Algorithm 2 Inference with GLOW-FM   
Require: Low-light image $x _ { 1 } , \mathcal { E } , \mathcal { D } _ { : }$ weights θ, NFEs M   
1: Initialize $\begin{array} { r } { z ^ { ( M ) } = \mathcal { E } ( x _ { 1 } ) ; } \end{array}$ choose grid $1 = t _ { M } > \cdots > t _ { 0 } = 0$   
2: for $m { = } M , M { - } 1 { , } { \ldots } 1$ do   
3: $\scriptstyle z ^ { ( m - 1 ) } = z ^ { ( m ) } + f _ { \theta } ( z ^ { ( m ) } , t _ { m } , t _ { m - 1 } )$   
4: end for   
5: Decode $\hat { x } _ { 0 } = \mathcal { D } ( z ^ { ( 0 ) } )$   
6: return enhanced image $\scriptstyle { \hat { x } } _ { 0 }$

## 3.2. Training

Fig. 2 illustrates the training procedure, and Algorithm 1 provides the detailed algorithmic steps. Given paired low-light and normally exposed images $( x _ { 1 } , x _ { 0 } )$ , we first encode them via the frozen VAE encoder as $z _ { 1 } = \mathcal { E } ( x _ { 1 } )$ and $z _ { 0 } = \mathcal { E } ( x _ { 0 } )$ Directly using the low-light latent endpoint can make the source distribution overly concentrated around the observed degraded embeddings, which may lead the model to learn a sharp and less stable transport. To smooth the source distribution and make it better supported over the target latent space, we follow stochastic-interpolant practice [1, 9] and perturb the source latent with small Gaussian noise:

$$
\begin{array} { r } { \tilde { z } _ { 1 } = z _ { 1 } + \sigma \epsilon , \qquad \epsilon \sim \mathcal { N } ( 0 , I ) . } \end{array}\tag{7}
$$

The perturbed latent $\tilde { z } _ { 1 }$ is then used as the source endpoint during training.

Flow-matching loss. We train on the linear latent flow path between the normally exposed target $z _ { \mathrm { 0 } }$ and the perturbed low-light source $\tilde { z } _ { 1 }$ . For sampled times $t { \sim } \mathcal { U } [ 0 , 1 ]$ and $s \sim \mathcal { U } [ 0 , t ] ,$ , we define

$$
z _ { t } = ( 1 - t ) z _ { 0 } + t \tilde { z } _ { 1 } , \qquad z _ { s } = ( 1 - s ) z _ { 0 } + s \tilde { z } _ { 1 } ,\tag{8}
$$

with the conditional flow-matching target: $v = \tilde { z } _ { 1 } - z _ { 0 }$ . The FM loss supervises the boundary case of the two-time model, where s=t and $F _ { \theta }$ represents the instantaneous velocity (Eq. (3)):

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { F M } } = \mathbb { E } _ { z _ { 0 } , z _ { 1 } , s , \epsilon } \left[ \| F _ { \theta } ( z _ { s } , s , s ) - v \| _ { 2 } ^ { 2 } \right] . } \end{array}\tag{9}
$$

Ground Truth  
ReDDiT  
PyDiff  
WCDM  
![](images/a7052f9582d17544b6c4bc1f6c6608d2360b222a9c62b5a64fe8c90fe5476c51.jpg)  
Figure 3. Qualitative comparison on LSD-DLL. Representative examples from LSD-DLL comparing low-light inputs, outputs from ReDDiT, PyDiff, WCDM, and GLOW-FM, and the corresponding ground-truth normal-light images.

This anchors the model to the standard flow-matching velocity and provides the velocity proxy used by TVM at terminal points.

Terminal-velocity loss. The TVM loss supervises finite displacements between two sampled times. For each pair (s,t) such that $s \leq t ,$ the displacement model $f _ { \boldsymbol { \theta } } ( z _ { t } , t , s )$ predicts a terminal latent $\hat { z } _ { s }$ as defined in Eq. (5). Following TVM [71], we train the displacement by matching its derivative with respect to the terminal time s to the velocity at the predicted terminal point. Using the parameterization in Eq. (3), this couples the derivative of the finite-displacement prediction, $\partial _ { s } f _ { \theta } ( z _ { t } , t , s )$ , with the boundary velocity branch $F _ { \theta } ( \hat { z } _ { s } , s , s )$ Since the true terminal velocity is unavailable, we use an EMA copy of the network as a target proxy:

$$
u _ { \bar { \theta } } ( \hat { z } _ { s } , s ) = F _ { \bar { \theta } } ( \mathrm { s g } ( \hat { z } _ { s } ) , s , s ) ,\tag{10}
$$

where $\bar { \theta }$ denotes the EMA parameters (teacher model) and sg(·) denotes stop-gradient. The terminal-velocity loss is

$$
\mathcal { L } _ { \mathrm { T V M } } = \mathbb { E } _ { z _ { 0 } , z _ { 1 } , t , s , \epsilon } \Big [ \| \partial _ { s } f _ { \theta } ( z _ { t } , t , s ) - \mathrm { s g } ( u _ { \theta } ( \hat { z } _ { s } , s ) ) \| _ { 2 } ^ { 2 } \Big ] .\tag{11}
$$

The derivative $\partial _ { s } f _ { \theta } ( z _ { t } , t , s )$ is computed with a Jacobian-vector product (JVP) with respect to the terminal time s. The EMA target and stop-gradient follow the practical biased estimator used in TVM: the target velocity is treated as fixed during the current update, which prevents gradients from flowing through both the predicted terminal point and the target branch. This stabilizes the proxy velocity used to supervise finite displacements.

Total objective. The final objective combines the flow-matching and TVM losses from Eqs. (9) and (11):

$$
\begin{array} { r } { \mathcal { L } = \mathcal { L } _ { \mathrm { F M } } + \lambda \mathcal { L } _ { \mathrm { T V M } } , } \end{array}\tag{12}
$$

where λ controls the relative contribution of the TVM term. In practice, we employ λ as a warm-up schedule, initially training the model with the FM objective to establish a stable velocity

field before gradually introducing the TVM loss. Additional schedule details are provided in Appendix C.

## 3.3. Inference

At inference time, we encode the low-light input as the initial latent $z ^ { ( M ) } = \mathcal { E } ( x _ { 1 } )$ at the source endpoint $t _ { M } = 1$ . We use the uniform time grid $t _ { m } = m / M , m = 0 , . . . , M .$ , traversed in decreasing order from $t _ { M } = 1 \mathrm { t o } t _ { 0 } = 0$ . For each m, the interval uses start time $t = t _ { m }$ and terminal time $s = t _ { m - 1 }$ , and the model predicts the displacement from the current latent at $t _ { m }$ to the next latent at $t _ { m - 1 }$ . Each interval requires one network evaluation; hence, the number of intervals M equals the number of NFEs. For one-step enhancement, M=1, so t=1 and $s { = } 0 ,$ and the model directly transports the encoded low-light latent to the enhanced endpoint. For few-step enhancement, the same displacement model is applied over multiple shorter intervals. The full inference procedure is summarized in Algorithm 2.

## 3.4. Efficient Latent Architecture

Compact Latent Representation. Operating in latent space is central to the efficiency of GLOW-FM. Instead of applying the flow model to full-resolution RGB images, the VAE encoder maps the input into a lower-dimensional representation with reduced spatial resolution. The learned transport therefore acts on a compact latent grid, lowering the cost of each function evaluation and making low-NFE inference more practical.

The latent space also provides a useful restoration prior. Since the VAE is pretrained on a broad natural-image distribution, its representation captures reusable image structure while suppressing some pixel-level noise and dataset-specific degradation patterns. Prior work has similarly shown that pretrained latent representations can support robust image restoration and domain generalization [29, 48, 51]. As a result, the flow model learns exposure and appearance corrections in a natural-image latent domain rather than relying solely on dataset-specific pixel statistics. For the autoencoder, inspired by [9], we use Tiny AutoEncoder for Stable Diffusion (TAESD) [49], a lightweight distilled alternative to the Stable Diffusion VAE (SD-VAE) [44]. TAESD preserves an SD-VAE-compatible latent interface while substantially reducing the cost of encoding and decoding. In our implementation, replacing SD-VAE with TAESD reduces encoding/decoding GFLOPs by 15× and parameter count by 33×, with minimal loss in end-to-end restoration performance. This is important for low-NFE generation, where the flow model requires only one or a few evaluations and the VAE cost becomes a significant part of the total inference budget.

<table><tr><td rowspan="2">Category</td><td rowspan="2">Method</td><td rowspan="2">NFEs↓ GFLOPs↓ Params (M)↓</td><td rowspan="2"></td><td rowspan="2"></td><td colspan="5">LSD-DLL</td><td colspan="5">UHD-LOL4K</td></tr><tr><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>MUSIQ↑</td><td>NIQE↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓ MUSIQ↑</td><td></td><td>NIQE↓</td></tr><tr><td rowspan="4">Non-Generative</td><td>TFFormer [45]</td><td>1</td><td>34</td><td>5.9</td><td>14.1</td><td>0.349</td><td>0.63</td><td>32.7</td><td>4.49</td><td>31.4</td><td>0.987</td><td>0.04</td><td>32.9</td><td>4.65</td></tr><tr><td>Retinexformer [5]</td><td>1</td><td>17</td><td>1.6</td><td>16.9</td><td>0.567</td><td>0.54</td><td>28.1</td><td>4.64</td><td>30.3</td><td>0.983</td><td>0.05</td><td>34.0</td><td>4.47</td></tr><tr><td>HVI-CIDNet [62]</td><td>1</td><td>8</td><td>2.0</td><td>15.2</td><td>0.495</td><td>0.56</td><td>23.5</td><td>4.42</td><td>33.7</td><td>0.989</td><td>0.02</td><td>33.1</td><td>5.03</td></tr><tr><td>Multinex [4]</td><td>1</td><td>3</td><td>0.04</td><td>15.6</td><td>0.501</td><td>0.63</td><td>21.6</td><td>7.73</td><td>24.8</td><td>0.958</td><td>0.09</td><td>32.0</td><td>4.72</td></tr><tr><td rowspan="3">Generative</td><td>PyDiff [70]</td><td>4</td><td>473</td><td>97.9</td><td>17.3</td><td>0.544</td><td>0.56</td><td>33.2</td><td>7.38</td><td>22.5</td><td>0.956</td><td>0.17</td><td>32.4</td><td>4.83</td></tr><tr><td>WCDM [20]</td><td>10</td><td>90</td><td>22.1</td><td>17.1</td><td>0.553</td><td>0.59</td><td>25.5</td><td>5.94</td><td>26.8</td><td>0.957</td><td>0.16</td><td>31.7</td><td>4.08</td></tr><tr><td>ReDDiT [25]</td><td>2</td><td>128</td><td>17.4</td><td>14.1</td><td>0.488</td><td>0.59</td><td>30.9</td><td>7.45</td><td>22.2</td><td>0.924</td><td>0.25</td><td>35.9</td><td>5.24</td></tr><tr><td rowspan="3">Ours</td><td>GLOW-FM-XL</td><td>1</td><td>48</td><td>124.0</td><td>20.3</td><td>0.548</td><td>0.48</td><td>36.0</td><td>2.92</td><td>31.7</td><td>0.969</td><td>0.12</td><td>35.2</td><td>4.59</td></tr><tr><td>GLOW-FM-M</td><td>1</td><td>36</td><td>21.8</td><td>19.5</td><td>0.531</td><td>0.50</td><td>35.5</td><td>2.86</td><td>30.6</td><td>0.961</td><td>0.14</td><td>35.1</td><td>4.63</td></tr><tr><td>GLOW-FM-S</td><td>1</td><td>34</td><td>9.1</td><td>19.7</td><td>0.516</td><td>0.52</td><td>37.1</td><td>2.70</td><td>30.3</td><td>0.962</td><td>0.14</td><td>34.9</td><td>4.62</td></tr></table>

Table 1. Results on paired low-light benchmarks. We evaluate on LSD-DLL and UHD-LOL4K benchmarks and report image quality and efficiency metrics. Best results are shown in bold and second-best results are underlined.

<table><tr><td>Category</td><td>Method</td><td>NFEs↓</td><td>GFLOPs↓</td><td>Params (M)↓</td><td>NIQE↓</td><td>MUSIQ↑</td></tr><tr><td rowspan="4">Non-Generative</td><td>TFFormer [45]</td><td>1</td><td>34</td><td>5.9</td><td>4.42</td><td>24.1</td></tr><tr><td>Retinexformer [5]</td><td>1</td><td>17</td><td>1.6</td><td>7.35</td><td>23.4</td></tr><tr><td>HVI-CIDNet [62]</td><td>1</td><td>8</td><td>2.0</td><td>6.26</td><td>26.5</td></tr><tr><td>Multinex [4]</td><td>1</td><td>3</td><td>0.04</td><td>6.96</td><td>27.2</td></tr><tr><td rowspan="3">Generative</td><td>PyDiff [70]</td><td>4</td><td>473</td><td>97.9</td><td>6.96</td><td>26.3</td></tr><tr><td>WCDM [20]</td><td>10</td><td>90</td><td>22.1</td><td>6.16</td><td>23.7</td></tr><tr><td>ReDDiT [25]</td><td>2</td><td>128</td><td>17.4</td><td>7.94</td><td>26.4</td></tr><tr><td rowspan="3">Ours</td><td>GLOW-FM-XL</td><td>1</td><td>48</td><td>124.0</td><td>4.33</td><td>28.4</td></tr><tr><td>GLOW-FM-M</td><td>1</td><td>36</td><td>21.8</td><td>4.49</td><td>28.3</td></tr><tr><td>GLOW-FM-S</td><td>1</td><td>34</td><td>9.1</td><td>4.99</td><td>27.7</td></tr></table>

Table 2. Results on LSD-U. Best results are shown in bold and second-best results are underlined.

HDiT two-time flow model. We implement the two-time conditioned flow model $F _ { \theta } ( z _ { t } , t , s )$ with an Hourglass Diffusion Transformer (HDiT) backbone. HDiT combines the multi-scale efficiency of U-Net-like architectures with the representation capacity of diffusion transformers [10]. Unlike standard DiT-style architectures [42], which apply global attention at every resolution and incur $O ( n ^ { 2 } )$ attention cost, HDiT restricts global attention to low-resolution stages and uses fixed-window neighborhood attention at higher resolutions, yielding linear $O ( n )$ scaling with the number of latent image tokens. This design is useful for GLOW-FM because latent restoration requires both global exposure/color correction and preservation of spatially aligned local content from the degraded input, while

keeping the two-time flow model efficient at inference.
<table><tr><td>Method</td><td></td><td></td><td>NFEs↓ GFLOPs↓ Params (M)↓</td><td>PSNR↑</td><td>MUSIQ↑ NIQE↓</td><td></td></tr><tr><td>PyDiff [70]</td><td>4</td><td>473</td><td>97.9</td><td>21.9</td><td>74.2</td><td>5.40</td></tr><tr><td>WCDM [20]</td><td>10</td><td>90</td><td>22.1</td><td>26.3</td><td>66.9</td><td>4.52</td></tr><tr><td>ReDDiT [25]</td><td>2</td><td>128</td><td>17.4</td><td>20.8</td><td>71.7</td><td>5.38</td></tr><tr><td>GLOW-FM-XL</td><td>1</td><td>48</td><td>124</td><td>21.8</td><td>70.4</td><td>3.71</td></tr><tr><td>GLOW-FM-M</td><td>1</td><td>36</td><td>21.8</td><td>21.3</td><td>71.5</td><td>3.79</td></tr><tr><td>GLOW-FM-S</td><td>1</td><td>34</td><td>9.1</td><td>21.2</td><td>71.3</td><td>3.70</td></tr></table>

Table 3. Results on LOL-v1. Best results are shown in bold and second-best results are underlined.

## 4. Experiments

## 4.1. Experimental Setup

Datasets. We evaluate GLOW-FM on both paired and unpaired real-world LLIE benchmarks. Our primary benchmark is LSD-DLL [45], a large-scale dataset designed to improve the realism and diversity of LLIE evaluation. Unlike many earlier low-light datasets [7, 15, 64], which are limited by synthetic darkening, controlled capture settings, or restricted sensor coverage, LSD-DLL [45] provides real low and normal-light image pairs captured across diverse scenes and devices in the wild using dynamic exposure calibration and multi-frame alignment to obtain more realistic paired data. As a result, it better captures phenomena such as sensor noise, spatially varying illumination, and color distortions that are often underrepresented in existing LLIE benchmarks. These properties motivated its use as the benchmark dataset in [8]. LSD-DLL also provides the LSD-U test split, which consists of unpaired low-light images captured with unseen devices and enables evaluation of generalization beyond the paired training distribution. To complement LSD-DLL, we additionally evaluate on UHD-LOL4K [54], a high-resolution LLIE benchmark, and LOL-v1 [7], one of the most widely used datasets in the LLIE literature.

Following [19, 20, 25], we further evaluate cross-dataset generalization by training on LOL-v2-Syn [64] and testing directly on five widely used unpaired real-world low-light datasets: DICM [27], LIME [14], MEF [36], NPE [53], and VV [50].

<table><tr><td rowspan="2">Category</td><td rowspan="2">Method</td><td rowspan="2">GFLOPs↓ Params (M)↓</td><td rowspan="2"></td><td colspan="2">DICM</td><td colspan="2">LIME</td><td colspan="2">MEF</td><td colspan="2">NPE</td><td colspan="2">VV</td><td colspan="2">Mean</td></tr><tr><td>NIQE↓</td><td>BRISQUE↓</td><td>NIQE↓</td><td>BRISQUE↓</td><td>NIQE↓</td><td>BRISQUE↓</td><td>NIQE↓</td><td>BRISQUE↓</td><td>NIQE↓</td><td>BRISQUE↓</td><td>NIQE↓</td><td>BRISQUE↓</td></tr><tr><td rowspan="9">Non-Generative</td><td>SNR-Aware [61]</td><td>24</td><td>39.1</td><td>6.17</td><td>33.2</td><td>5.96</td><td>34.9</td><td>4.33</td><td>27.9</td><td>6.51</td><td>36.9</td><td>11.89</td><td>80.0</td><td>6.97</td><td>42.6</td></tr><tr><td>DCC-Net [69]</td><td>64</td><td>13.2</td><td>3.44</td><td>27.0</td><td>4.26</td><td>25.4</td><td>4.13</td><td>30.2</td><td>3.51</td><td>25.1</td><td>3.55</td><td>26.2</td><td>3.78</td><td>26.8</td></tr><tr><td>Zero-DCE [13]</td><td>5</td><td>0.079</td><td>2.84</td><td>19.3</td><td>3.80</td><td>18.2</td><td>4.02</td><td>26.3</td><td>3.59</td><td>26.4</td><td>3.21</td><td>29.1</td><td>3.49</td><td>23.8</td></tr><tr><td>KinD++ [68]</td><td>371</td><td>8.3</td><td>3.10</td><td>20.5</td><td>4.72</td><td>23.3</td><td>4.20</td><td>34.1</td><td>3.79</td><td>27.6</td><td>2.68</td><td>26.5</td><td>3.70</td><td>26.4</td></tr><tr><td>CPGA-Net [58]</td><td>2</td><td>0.025</td><td>3.62</td><td>28.1</td><td>4.59</td><td>21.5</td><td>4.18</td><td>28.1</td><td>3.73</td><td>26.4</td><td>4.22</td><td>35.3</td><td>4.07</td><td>27.9</td></tr><tr><td>HVI-CIDNet [62]</td><td>8</td><td>2.0</td><td>3.02</td><td>19.0</td><td>3.82</td><td>14.2</td><td>3.82</td><td>14.1</td><td>3.43</td><td>22.5</td><td>3.25</td><td>23.3</td><td>3.47</td><td>18.6</td></tr><tr><td>Retinexformer [5]</td><td>17</td><td>1.6</td><td>3.37</td><td>9.5</td><td>3.86</td><td>12.5</td><td>3.82</td><td>14.9</td><td>3.35</td><td>16.4</td><td>2.73</td><td>14.6</td><td>3.42</td><td>13.6</td></tr><tr><td>TFFormer [45]</td><td>34</td><td>5.9</td><td>4.45</td><td>26.3</td><td>4.72</td><td>22.0</td><td>4.80</td><td>25.7</td><td>4.56</td><td>25.9</td><td>2.91</td><td>9.1</td><td>4.29</td><td>21.8</td></tr><tr><td>Multinex [4]</td><td>3</td><td>0.045</td><td>3.14</td><td>12.0</td><td>3.84</td><td>14.2</td><td>3.72</td><td>16.2</td><td>3.37</td><td>17.6</td><td>2.93</td><td>17.8</td><td>3.40</td><td>15.6</td></tr><tr><td rowspan="6">Generative</td><td>GSAD [19]</td><td>1277</td><td>17.2</td><td>2.93</td><td>18.1</td><td>3.61</td><td>13.3</td><td>3.58</td><td>14.6</td><td>3.37</td><td>21.2</td><td>3.22</td><td>23.9</td><td>3.34</td><td>18.2</td></tr><tr><td>PyDiff [70]</td><td>473</td><td>97.9</td><td>3.86</td><td>13.4</td><td>4.35</td><td>18.7</td><td>5.19</td><td>28.2</td><td>3.68</td><td>20.8</td><td>3.52</td><td>19.0</td><td>4.12</td><td>20.0</td></tr><tr><td>ReDDiT [25]</td><td>128</td><td>17.4</td><td>3.70</td><td>15.5</td><td>3.80</td><td>12.8</td><td>3.88</td><td>11.9</td><td>3.59</td><td>14.6</td><td>2.65</td><td>13.9</td><td>3.53</td><td>13.7</td></tr><tr><td>WCDM [20]</td><td>90</td><td>22.1</td><td>3.36</td><td>15.8</td><td>3.59</td><td>14.4</td><td>3.78</td><td>16.7</td><td>3.37</td><td>18.3</td><td>2.35</td><td>8.0</td><td>3.29</td><td>14.7</td></tr><tr><td>Reti-Diff [17]</td><td>175 92</td><td>26.1 27.8</td><td>3.10 3.37</td><td>7.8</td><td>3.66</td><td>12.0</td><td>3.68</td><td>15.33</td><td>3.39</td><td>17.0</td><td>2.88</td><td>16.8</td><td>3.34</td><td>13.8</td></tr><tr><td>LightenDiffusion [21]</td><td></td><td></td><td></td><td>19.5</td><td>4.02</td><td>17.1</td><td>4.00</td><td>23.8</td><td>3.33</td><td>20.8</td><td>3.43</td><td>27.4</td><td>3.63</td><td>21.7</td></tr><tr><td rowspan="3">Ours</td><td>GLOW-FM-XL</td><td>48</td><td>124.0</td><td>2.61</td><td>7.9</td><td>3.12</td><td>8.9</td><td>3.28</td><td>10.9</td><td>2.94</td><td>12.5</td><td>2.28</td><td>13.7</td><td>2.86</td><td>10.8</td></tr><tr><td>GLOW-FM-M</td><td>36</td><td>21.8 9.1</td><td>2.60</td><td>8.2</td><td>3.17</td><td>10.1</td><td>3.19</td><td>10.8</td><td>2.93</td><td>12.6</td><td>2.32</td><td>14.0</td><td>2.84</td><td>11.1</td></tr><tr><td>GLOW-FM-S</td><td>34</td><td></td><td>2.58</td><td>7.8</td><td>3.16</td><td>9.7</td><td>3.21</td><td>10.7</td><td>2.95</td><td>12.1</td><td>2.30</td><td>14.3</td><td>2.83</td><td>10.9</td></tr></table>

Table 4. Cross-dataset generalization benchmark. Results on DICM, LIME, MEF, NPE, and VV. Lower is better. Best results are shown in bold and second-best results are underlined.

Since these datasets do not provide paired reference images, they are evaluated using no-reference image quality metrics.

Evaluation metrics. For paired benchmarks, we evaluate both distortion and perceptual quality. Distortion is measured using PSNR and SSIM [56], while perceptual similarity is assessed using LPIPS [67]. We additionally evaluate perceptual image quality using the no-reference metrics NIQE [39] and MUSIQ [24]. For unpaired benchmarks, where ground-truth reference images are unavailable, we follow [20] and report the no-reference perceptual quality metrics NIQE [39] and BRISQUE [38]. Results are averaged over five evaluation seeds; full mean and STD results are provided in Appendix F. Implementation details. GLOW-FM employs a frozen Tiny-VAE [49] to obtain 16-channel latent image representations, while only the latent flow model is optimized during training. Unless otherwise specified, experiments use the GLOW-FM-XL configuration with a single NFE. We additionally evaluate medium (M) and small (S) configurations, which differ only in the capacity of the HDiT backbone. Source latents are perturbed with Gaussian noise using σ =0.01. Models are trained for approximately 100k optimization steps on 8 NVIDIA H100 GPUs. For LSD-DLL and UHD-LOL4K, we trained the baselines on the respective training sets, whereas all other results use the authors’ official code and checkpoints. Appendix D details these baseline training protocols together with our implementation, hyperparameters, and runtime and memory analyses.

## 4.2. Results on Paired Benchmarks

Tab. 1 compares GLOW-FM with representative non-generative and recent efficient generative LLIE methods on the paired LSD-DLL and UHD-LOL4K benchmarks. Across both datasets, GLOW-FM achieves a favorable quality–efficiency trade-off with only a single NFE during inference. On LSD-DLL, it obtains the strongest overall performance across several distortion and perceptual-quality metrics. On UHD-LOL4K, nongenerative methods remain stronger on some distortion-oriented metrics, while GLOW-FM maintains competitive image quality with substantially lower inference cost than the generative baselines. The medium and small variants further reduce computational cost while preserving strong image quality. Representative qualitative results on LSD-DLL are shown in Fig. 3.

On LOL-v1, GLOW-FM maintains a strong qualityefficiency tradeoff. As shown in Tab. 3, GLOW-FM achieves the best NIQE score among the compared methods with a single NFE. We hypothesize that its weaker distortion performance arises from the limited training data available in this benchmark. A controlled data-scaling experiment supporting this interpretation is presented in Fig. 5. Since paired evaluation does not fully capture robustness to unseen devices and low-light domains, we separately examine cross-dataset and unseen-device generalization in Sec. 4.3.

## 4.3. Generalization to Unseen Low-Light Data

Tab. 4 reports results on five widely used unpaired real-world low-light benchmarks: DICM, LIME, MEF, NPE, and VV. Following the protocol adopted in prior work [19, 25], GLOW-FM is trained on LOL-v2-Syn [64] and evaluated directly on the target datasets without additional fine-tuning. GLOW-FM achieves the best generative NIQE and BRISQUE results on nearly all evaluated benchmarks while operating at substantially lower computational cost. It also obtains the strongest mean performance across both metrics.

Tab. 2 reports results on the LSD-U subset from [8], which evaluates generalization to low-light images captured with unseen devices. Among all compared generative and non-generative methods, GLOW-FM achieves the best NIQE and MUSIQ scores with a single NFE. Moreover, all GLOW-FM variants achieve competitive no-reference image quality despite substantially lower computational cost than the competing generative methods. These results demonstrate strong generalization beyond the training distribution, indicating that the proposed latent flow formulation remains robust under real-world distribution shifts. Together with the cross-dataset results in Tab. 4, they demonstrate consistent generalization across unseen devices and diverse low-light domains.

![](images/83dcd66d81e8aa0cc8582cdfe0efd9c44e91a5e3a83b3033914a3569f9081a3d.jpg)

Figure 4. Qualitative generalization comparison. Results on unpaired real-world low-light benchmarks show that GLOW-FM recovers natura illumination while preserving scene structure across diverse inputs.  
![](images/56027d9a8c1d41c57c332d2153ddf971b683dddfd216aa35a8f1fa20f498e156.jpg)  
Figure 5. Effect of train-set size. Results on LSD-DLL for models trained with increasing train-set size.

Given the perception–distortion trade-off [3], we evaluate metrics from both families. Quantitative results and paired ground-truth comparisons in Fig. 3 indicate GLOW-FM offers a favorable balance among the compared methods, while unpaired examples in Fig. 4 show natural illumination and fewer artifacts; additional examples are provided in Appendix G.

## 4.4. Ablation Study

Comparison of one-step objectives. Tab. 5 compares TVM with MeanFlow [12], ConsistencyFM [63] and No-TVM (standard Flow Matching [30, 31]), with all other experimental settings held fixed. We additionally evaluate No-TVM at 25 NFEs. Among the matched TAESD one-step variants, TVM achieves the strongest overall distortion and perceptual performance. MeanFlow forms its JVP target from the initial-time velocity, whose sample-wise approximation can destabilize training [71]; this may be particularly problematic for LLIE, where the initial state is a degraded low-light image. ConsistencyFM enforces a velocity-consistent straight flow, which may restrict a single full-interval mapping for spatially heterogeneous restoration [9, 63]. TVM instead regularizes the displacement at its predicted terminal state, aligning the one-step mapping with the velocity near the enhanced endpoint [71].

Effect of training-set size. Fig. 5 isolates the effect of data availability by training GLOW-FM on progressively larger LSD-DLL subsets while keeping the architecture and training protocol fixed. Increasing the training set from approximately 0.2K to 3.0K image pairs improves PSNR monotonically from 16.8 to 20.3 dB and MUSIQ from 34.5 to 36.0. This trend supports our data-limited interpretation of the LOL-v1 distortion results and is consistent with prior observations that flow-based models benefit strongly from additional training data [72].

Effect of latent representation. Tab. 5 further evaluates the effect of the latent representation on GLOW-FM performance.

<table><tr><td>Variant</td><td>Latent</td><td>GFLOPs↓</td><td>Params (M)↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS.↓</td><td>MUSIQ↑</td><td>NIQE↓</td></tr><tr><td>Pixel Space (TVM)</td><td>None</td><td>1693</td><td>121</td><td>16.0</td><td>0.47</td><td>0.54</td><td>28.2</td><td>4.51</td></tr><tr><td>SD-VAE (TVM)</td><td>SD-VAE</td><td>471</td><td>205</td><td>19.6</td><td>0.50</td><td>0.47</td><td>40.6</td><td>3.37</td></tr><tr><td>No TVM (1 NFE)</td><td>TAESD</td><td>48</td><td>124</td><td>19.7</td><td>0.50</td><td>0.48</td><td>27.7</td><td>6.00</td></tr><tr><td>No TVM (25 NFE)</td><td>TAESD</td><td>415</td><td>124</td><td>20.1</td><td>0.54</td><td>0.37</td><td>35.6</td><td>3.00</td></tr><tr><td>ConsistencyFM</td><td>TAESD</td><td>48</td><td>124</td><td>19.4</td><td>0.52</td><td>0.50</td><td>32.8</td><td>3.53</td></tr><tr><td>MeanFlow</td><td>TAESD</td><td>48</td><td>124</td><td>19.5</td><td>0.52</td><td>0.49</td><td>31.7</td><td>3.59</td></tr><tr><td>GLOW-FM</td><td>TAESD</td><td>48</td><td>124</td><td>20.3</td><td>0.55</td><td>0.48</td><td>36.0</td><td>2.92</td></tr></table>

Table 5. Ablation study. Results on LSD-DLL assessing the latent representation and one-step training objective.

We compare the proposed TAESD latent representation against a larger SD-VAE and a direct pixel-space variant while keeping the velocity network fixed. TAESD achieves better PSNR, SSIM, and NIQE with nearly an order of magnitude less computation, whereas SD-VAE achieves better LPIPS and MUSIQ at substantially higher computational and parameter costs. Additional autoencoder-reconstruction analysis is provided in Appendix E. We further compare against a pixel-space variant that removes the latent representation entirely. Despite requiring significantly more computation, direct pixel-space modeling produces substantially weaker distortion and perceptual quality. These results indicate that the proposed latent-space formulation is not only computationally advantageous but also improves enhancement quality, suggesting that the compact TAESD representation provides an effective latent space for LLIE.

## 5. Conclusion

Low-light image enhancement remains challenging because noise, color shifts, and information loss make faithful restoration ambiguous. Although recent generative LLIE methods have reduced sampling cost, combining single-step inference, low overall computation, and robust performance across real-world conditions remains challenging.

In this work, we address this challenge with GLOW-FM, an efficient generative LLIE framework based on one-step latent flow matching. By combining efficiency-aware latent flow matching with Terminal Velocity Matching, GLOW-FM enables single-NFE enhancement while preserving generative priors for realistic restoration. Experiments show competitive quality, improved efficiency, and state-of-the-art cross-dataset generalization, suggesting a practical direction for robust and efficient generative low-light enhancement.

## References

[1] Michael Albergo and Eric Vanden-Eijnden. Building normalizing flows with stochastic interpolants. In ICLR 2023 Conference, 2023. 4

[2] Arm Editorial Team. Scalable matrix extension (SME) for Armv9 architecture enables AI innovation on the Arm CPU, 2024. 18

[3] Yochai Blau and Tomer Michaeli. The perception-distortion tradeoff. In Proceedings of the IEEE conference on computer vision andpattern recognition, pages 6228–6237, 2018. 1, 8

[4] Alexandru Brateanu, Tingting Mu, Codruta O Ancuti, and Cosmin Ancuti. Multinex: Lightweight low-light image enhancement via multi-prior retinex. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 29887–29896, 2026. 1, 2, 6, 7, 19

[5] Yuanhao Cai, Hao Bian, Jing Lin, Haoqian Wang, Radu Timofte, and Yulun Zhang. Retinexformer: One-stage retinex-based transformer for low-light image enhancement. In Proceedings of the IEEE/CVF international conference on computer vision, pages 12504–12513, 2023. 1, 2, 6, 7, 19

[6] Chen Chen, Qifeng Chen, Jia Xu, and Vladlen Koltun. Learning to see in the dark. In Proceedings ofthe IEEE conference on computer vision and pattern recognition, pages 3291–3300, 2018. 1

[7] Wei Chen, Wenjing Wang, Wenhan Yang, and Jiaying Liu. Deep retinex decomposition for low-light enhancement. In British Machine Vision Conference, 2018. 1, 6

[8] George Ciubotariu, S. M. A. Sharif, Abdur Rehman, Fayaz Ali Dharejo, Rizwan Ali Naqvi, Marcos V. Conde, Radu Timofte, et al. Low light image enhancement challenge at ntire 2026. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, pages 1741–1753, 2026. 1, 6, 7

[9] Elad Cohen, Idan Achituve, Idit Diamant, Arnon Netzer, and Hai Victor Habi. Efficient image restoration via latent consistency flow matching. In 36th British Machine Vision Conference 2025, BMVC 2025, Sheffield, UK, November 24-27, 2025. BMVA, 2025. 3, 4, 6, 8, 12, 13

[10] Katherine Crowson, Stefan Andreas Baumann, Alex Birch, Tanishq Mathew Abraham, Daniel Z Kaplan, and Enrico Shippole. Scalable high-resolution pixel-space image synthesis with hourglass diffusion transformers. In Forty-first International Conference on Machine Learning, 2024. 6

[11] Mauricio Delbracio and Peyman Milanfar. Inversion by direct iteration: An alternative to denoising diffusion for image restoration. Transactions on Machine Learning Research, 2023. 3

[12] Zhengyang Geng, Mingyang Deng, Xingjian Bai, Zico Kolter, and Kaiming He. Mean flows for one-step generative modeling. Advances in Neural Information Processing Systems, 38: 75460–75482, 2026. 2, 3, 8

[13] Chunle Guo, Chongyi Li, Jichang Guo, Chen Change Loy, Junhui Hou, Sam Kwong, and Runmin Cong. Zero-reference deep curve estimation for low-light image enhancement. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1780–1789, 2020. 7

[14] Xiaojie Guo, Yu Li, and Haibin Ling. Lime: Low-light image enhancement via illumination map estimation. IEEE Transactions on image processing, 26(2):982–993, 2016. 6

[15] Jiang Hai, Zhu Xuan, Ren Yang, Yutong Hao, Fengzhu Zou, Fang Lin, and Songchen Han. R2rnet: Low-light image enhancement via real-low to real-normal network. Journal of Visual Communication and Image Representation, 90:103712, 2023. 6

[16] Khurram Azeem Hashmi, Goutham Kallempudi, Didier Stricker, and Muhammad Zeshan Afzal. Featenhancer: Enhancing hierarchical features for object detection and beyond under low-light vision. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision (ICCV), pages 6725–6735, 2023. 1

[17] Chunming He, Chengyu Fang, Yulun Zhang, Longxiang Tang, Jinfa Huang, Kai Li, Xiu Li, Sina Farsiu, et al. Reti-diff: Illumination degradation image restoration with retinex-based latent diffusion model. In International Conference on Learning Representations, pages 43332–43352, 2025. 1, 2, 7

[18] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020. 2

[19] Jinhui Hou, Zhiyu Zhu, Junhui Hou, Hui Liu, Huanqiang Zeng, and Hui Yuan. Global structure-aware diffusion process for low-light image enhancement. Advances in Neural Information Processing Systems, 36:79734–79747, 2023. 1, 2, 6, 7

[20] Hai Jiang, Ao Luo, Haoqiang Fan, Songchen Han, and Shuaicheng Liu. Low-light image enhancement with waveletbased diffusion models. ACM Transactions on Graphics (TOG), 42(6):1–14, 2023. 2, 6, 7, 19

[21] Hai Jiang, Ao Luo, Xiaohong Liu, Songchen Han, and Shuaicheng Liu. Lightendiffusion: Unsupervised low-light image enhancement with latent-retinex diffusion models. In European Conference on Computer Vision, pages 161–179, 2024. 1, 2, 7

[22] Yifan Jiang, Xinyu Gong, Ding Liu, Yu Cheng, Chen Fang, Xiaohui Shen, Jianchao Yang, Pan Zhou, and Zhangyang Wang. Enlightengan: Deep light enhancement without paired supervision. IEEE Transactions on Image Processing, 30:2340–2349, 2021. 1

[23] Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusion-based generative models. Advances in neural information processing systems, 35: 26565–26577, 2022. 3

[24] Junjie Ke, Qifei Wang, Yilin Wang, Peyman Milanfar, and Feng Yang. Musiq: Multi-scale image quality transformer. In Proceedings of the IEEE/CVF international conference on computer vision, pages 5148–5157, 2021. 7

[25] Guanzhou Lan, Qianli Ma, Yuqi Yang, Zhigang Wang, Dong Wang, Xuelong Li, and Bin Zhao. Efficient diffusion as low light enhancer. In Proceedings ofthe Computer Vision and Pattern Recognition Conference (CVPR), pages 21277–21286, 2025. 2, 6, 7, 19

[26] Edwin H Land. The retinex theory of color vision. Scientific american, 237(6):108–129, 1977. 2

[27] Chulwoo Lee, Chul Lee, and Chang-Su Kim. Contrast enhancement based on layered difference representation of 2d histograms. IEEE Transactions on Image Processing, 22(12): 5372–5384, 2013. 6

[28] Chongyi Li, Chunle Guo, Linghao Han, Jun Jiang, Ming-Ming Cheng, Jinwei Gu, and Chen Change Loy. Low-light image and video enhancement using deep learning: A survey. IEEE transactions on pattern analysis and machine intelligence, 44 (12):9396–9416, 2021. 1

[29] Huaqiu Li, Yong Wang, Tongwen Huang, Hailang Huang, Haoqian Wang, and Xiangxiang Chu. Ld-rps: Zero-shot unified image restoration via latent diffusion recurrent posterior sampling. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pages 13684–13694, 2025. 2, 5

[30] Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matthew Le. Flow matching for generative modeling. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net, 2023. 2, 3, 4, 8, 12, 13

[31] Xingchao Liu, Chengyue Gong, and qiang liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. In The Eleventh International Conference on Learning Representations, 2023. 2, 3, 8

[32] Yuen Peng Loh and Chee Seng Chan. Getting to know low-light images with the exclusively dark dataset. Computer vision and image understanding, 178:30–42, 2019. 1

[33] Ilya Loshchilov and Frank Hutter. Sgdr: Stochastic gradient descent with warm restarts. arXiv preprint arXiv:1608.03983, 2016. 15

[34] Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. Dpm-solver: A fast ode solver for diffusion probabilistic model sampling in around 10 steps. Advances in neural information processing systems, 35:5775–5787, 2022. 3

[35] Ziwei Luo, Fredrik K. Gustafsson, Zheng Zhao, Jens Sjolund,¨ and Thomas B. Schon. Image restoration with mean-reverting¨ stochastic differential equations. In Proceedings ofthe 40th International Conference on Machine Learning, pages 23045–23066. PMLR, 2023. 3

[36] Kede Ma, Kai Zeng, and Zhou Wang. Perceptual quality assessment for multi-exposure image fusion. IEEE Transactions on Image Processing, 24(11):3345–3356, 2015. 6

[37] Segol ´ ene Martin, Anne Gagneux, Paul Hagemann, and Gabriele\` Steidl. Pnp-flow: Plug-and-play image restoration with flow matching. In International Conference on Learning Representations, pages 45466–45492, 2025. 3

[38] Anish Mittal, Anush Krishna Moorthy, and Alan Conrad Bovik. No-reference image quality assessment in the spatial domain. IEEE Transactions on image processing, 21(12):4695–4708, 2012. 7

[39] Anish Mittal, Rajiv Soundararajan, and Alan C Bovik. Making a “completely blind” image quality analyzer. IEEE Signal processing letters, 20(3):209–212, 2012. 7

[40] Igor Morawski, Yu-An Chen, Yu sheng Lin, and Winston H. Hsu. Nod: Taking a closer look at detection under extreme low-light conditions with night object detection dataset. In 32nd British Machine Vision Conference 2021, BMVC 2021, Online, November 22-25, 2021, page 76. BMVA Press, 2021. 1

[41] Victor M Panaretos and Yoav Zemel. Statistical aspects of wasserstein distances. Annual Review of Statistics and Its Application, 6(1):405–431, 2019. 3, 12

[42] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings ofthe IEEE/CVF international conference on computer vision, pages 4195–4205, 2023. 6

[43] PyTorch Contributors. ReduceLROnPlateau — PyTorch 2.13 documentation. https://docs.pytorch.org/docs/ 2.13/generated/torch.optim.lr\_scheduler. ReduceLROnPlateau.html, 2026. 15

[44] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bjorn Ommer. High-resolution image synthesis¨ with latent diffusion models. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10684–10695, 2022. 6, 14

[45] SMA Sharif, Abdur Rehman, Zain Ul Abidin, Fayaz Ali Dharejo, Radu Timofte, and Rizwan Ali Naqvi. Illuminating darkness: Learning to enhance low-light images in-the-wild. In Proceedings of the IEEE/CVF Winter Conference on Applications ofComputer Vision, pages 2263–2272, 2026. 1, 2, 6, 7, 19

[46] Yuyang Shi, Valentin De Bortoli, Andrew Campbell, and Arnaud Doucet. Diffusion schrodinger bridge matching.¨ Advances in neural informationprocessing systems, 36:62183–62223, 2023. 3

[47] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021. OpenReview.net, 2021. 3

[48] Xavier Thomas and Deepti Ghadiyaram. What’s in a latent? leveraging diffusion latent space for domain generalization. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 2183–2194, 2025. 2, 5

[49] Patrick von Platen, Suraj Patil, Anton Lozhkov, Pedro Cuenca, Nathan Lambert, Kashif Rasul, Mishig Davaadorj, Dhruv Nair, Sayak Paul, William Berman, Yiyi Xu, Steven Liu, and Thomas Wolf. Diffusers: State-of-the-art diffusion models. https://github.com/huggingface/diffusers, 2022. 2, 6, 7, 14

[50] Vassilios Vonikakis, Rigas Kouskouridas, and Antonios Gasteratos. On the evaluation of illumination compensation algorithms. Multimedia Tools and Applications, 77(8):9211–9231, 2018. 6

[51] Ziyu Wan, Bo Zhang, Dongdong Chen, Pan Zhang, Dong Chen, Jing Liao, and Fang Wen. Bringing old photos back to life. In proceedings of the IEEE/CVF conference on computer vision andpattern recognition, pages 2747–2757, 2020. 2, 5

[52] Jianyi Wang, Kelvin CK Chan, and Chen Change Loy. Exploring clip for assessing the look and feel of images. In Proceedings of the AAAI conference on artificial intelligence, pages 2555–2563, 2023. 16

[53] Shuhang Wang, Jin Zheng, Hai-Miao Hu, and Bo Li. Naturalness preserved enhancement algorithm for non-uniform illumination images. IEEE transactions on image processing, 22(9): 3538–3548, 2013. 6

[54] Tao Wang, Kaihao Zhang, Tianrun Shen, Wenhan Luo, Bjorn Stenger, and Tong Lu. Ultra-high-definition low-light image enhancement: A benchmark and transformer-based method. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 2654–2662, 2023. 1, 2, 6

[55] Yufei Wang, Yi Yu, Wenhan Yang, Lanqing Guo, Lap-Pui Chau, Alex C Kot, and Bihan Wen. Exposurediffusion: Learning to expose for low-light image enhancement. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pages 12438–12448, 2023. 1

[56] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing, 13 (4):600–612, 2004. 7

[57] Wenyang Wei, Yang Yang, Xixi Jia, Xiangchu Feng, Weiwei Wang, and Renzhen Wang. Illumflow: Illumination-adaptive

low-light enhancement via conditional rectified flow and retinex decomposition, 2025. 2

[58] Shyang-En Weng, Shaou-Gang Miaou, and Ricky Christanto. A lightweight low-light image enhancement network via channel prior and gamma correction. International Journal of Pattern Recognition and Artificial Intelligence, 39(12):2554013, 2025. 1, 7

[59] Wenhui Wu, Jian Weng, Pingping Zhang, Xu Wang, Wenhan Yang, and Jianmin Jiang. Uretinex-net: Retinex-based deep unfolding network for low-light image enhancement. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 5901–5910, 2022. 1

[60] Jian Xu, Wei Chen, Shigui Li, Delu Zeng, John Paisley, and Qibin Zhao. Consist-retinex: One-step noise-emphasized consistency training accelerates high-quality retinex enhancement, 2026. 2

[61] Xiaogang Xu, Ruixing Wang, Chi-Wing Fu, and Jiaya Jia. Snr-aware low-light image enhancement. In CVPR, 2022. 1, 7

[62] Qingsen Yan, Yixu Feng, Cheng Zhang, Guansong Pang, Kangbiao Shi, Peng Wu, Wei Dong, Jinqiu Sun, and Yanning Zhang. Hvi: A new color space for low-light image enhancement. In Proceedings of the computer vision and pattern recognition conference, pages 5678–5687, 2025. 1, 2, 6, 7, 19

[63] Ling Yang, Zixiang Zhang, Zhilong Zhang, Xingchao Liu, Minkai Xu, Wentao Zhang, Chenlin Meng, Stefano Ermon, and Bin Cui. Consistency flow matching: Defining straight flows with velocity consistency. arXiv preprint arXiv:2407.02398, 2024. 2, 3, 8

[64] Wenhan Yang, Wenjing Wang, Haofeng Huang, Shiqi Wang, and Jiaying Liu. Sparse gradient regularized deep retinex network for robust low-light image enhancement. IEEE Transactions on Image Processing, 30:2072–2086, 2021. 6, 7

[65] Xunpeng Yi, Han Xu, Hao Zhang, Linfeng Tang, and Jiayi Ma. Diff-retinex: Rethinking low-light image enhancement with a generative diffusion model. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 12302–12311, 2023. 1, 2

[66] Yuyang Yin, Dejia Xu, Chuangchuang Tan, Ping Liu, Yao Zhao, and Yunchao Wei. Cle diffusion: Controllable light enhancement diffusion model. In Proceedings of the 31st ACM International Conference on Multimedia, pages 8145–8156, 2023. 1

[67] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 586–595, 2018. 7

[68] Yonghua Zhang, Xiaojie Guo, Jiayi Ma, Wei Liu, and Jiawan Zhang. Beyond brightening low-light images. International journal ofcomputer vision, 129(4):1013–1037, 2021. 1, 7

[69] Zhao Zhang, Huan Zheng, Richang Hong, Mingliang Xu, Shuicheng Yan, and Meng Wang. Deep color consistent network for low-light image enhancement. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1899–1908, 2022. 1, 7

[70] Dewei Zhou, Zongxin Yang, and Yi Yang. Pyramid diffusion models for low-light image enhancement. In Proceedings of the Thirty-Second International Joint Conference on Artificial Intelligence, 2023. 2, 6, 7, 19

[71] Linqi Zhou, Mathias Parger, Ayaan Haque, and Jiaming Song. Terminal velocity matching. In The Fourteenth International Conference on Learning Representations, 2026. 2, 3, 5, 8, 12, 13

[72] Kaizhen Zhu, Mokai Pan, Zhechuan Yu, Jingya Wang, Jingyi Yu, and Ye Shi. Diffusion bridge or flow matching? a unifying framework and comparative analysis. arXiv preprint arXiv:2509.24531, 2025. 8

[73] Yixuan Zhu, Wenliang Zhao, Ao Li, Yansong Tang, Jie Zhou, and Jiwen Lu. Flowie: Efficient image enhancement via rectified flow. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13–22, 2024. 3

# GLOW-FM: Efficient Generative Low-Light Image Enhancement with One-Step Latent Flow Matching

Supplementary Material

## A. Supplementary Overview

This supplementary material provides additional details and results supporting the main paper and is organized as follows:

• Appendix B provides the pixel-space Wasserstein bound for latent TVM.

• Appendix C presents additional implementation details.

• Appendix D describes the training protocols used for the baseline methods.

• Appendix E provides further ablation studies.

• Appendix F reports additional quantitative results.

• Appendix G presents additional qualitative comparisons.

## B. Pixel-Space Wasserstein Bound for Latent TVM

Motivation, notation, and definitions. In the main paper, we motivate TVM not only by its strong one-step generation performance, but also through theoretical analysis. We make this point precise below by showing that the Wasserstein-2 distance [41] between the predicted normal-light image distribution and the true normal-light image distribution is controlled by the latent TVM training objective and the VAE reconstruction error. Thus, minimizing our latent TVM objective reduces an upper bound on the pixel-space distributional gap.

Let X and Z denote the image and latent spaces, respectively. Let $x _ { 0 } \sim p _ { x _ { 0 } }$ be a normal-light target image, and let $z _ { 0 } = \mathcal { E } ( x _ { 0 } )$ be its latent representation under the frozen VAE encoder E. The decoder is denoted by $\mathcal { D } : \mathcal { Z }  \mathcal { X }$ . The TVM flow model has trainable parameters θ and produces an estimated normal-light latent $\hat { z } _ { 0 } \sim p _ { \hat { z } _ { 0 } } ^ { \theta }$ whose decoded prediction is $\hat { x } _ { 0 } = \mathcal { D } ( \hat { z } _ { 0 } )$ . We also define the VAE reconstruction of the target image as $\widetilde { \boldsymbol { x } } _ { 0 } = \mathcal { D } ( \mathcal { E } ( \boldsymbol { x } _ { 0 } ) ) = \mathcal { D } ( \boldsymbol { z } _ { 0 } )$

We use # to denote the pushforward of a distribution. Hence,

$$
p _ { z _ { 0 } } = \mathcal { E } _ { \# } p _ { x _ { 0 } } , \qquad p _ { \hat { x } _ { 0 } } ^ { \theta } = \mathcal { D } _ { \# } p _ { \hat { z } _ { 0 } } ^ { \theta } , \qquad p _ { \tilde { x } _ { 0 } } = \mathcal { D } _ { \# } p _ { z _ { 0 } } .\tag{13}
$$

The variable $z _ { s }$ denotes a latent point at time s, $v _ { s }$ denotes the corresponding sample-wise conditional velocity, and $u ( z _ { s } , s )$ denotes the marginal latent velocity field. As in Flow Matching [30], $u ( z _ { s } , s ) = \mathbb { E } [ v _ { s } | z _ { s } ]$

We denote the terminal-velocity matching loss by $\mathcal { L } _ { \mathrm { T V M } } ^ { t , s } ( \theta )$ and the conditional flow-matching loss used in our training objective by $\mathcal { L } _ { \mathrm { F M } } ^ { s } ( \theta )$ . For compactness, define the combined per-time objective

$$
\begin{array} { r } { \begin{array} { r } { \mathcal { L } ^ { t , s } ( \theta ) = \mathcal { L } _ { \mathrm { T V M } } ^ { t , s } ( \theta ) + \mathcal { L } _ { \mathrm { F M } } ^ { s } ( \theta ) . } \end{array} } \end{array}\tag{14}
$$

The VAE reconstruction error is

$$
\epsilon _ { \mathrm { V A E } } = \mathbb { E } _ { \boldsymbol { x } _ { 0 } \sim p _ { \boldsymbol { x } _ { 0 } } } \| \boldsymbol { D } ( \boldsymbol { \mathcal { E } } ( \boldsymbol { x } _ { 0 } ) ) - \boldsymbol { x } _ { 0 } \| _ { 2 } ^ { 2 } .\tag{15}
$$

For distributions $\mu , \nu$ over a space $\mathcal { V } ,$ the Wasserstein-2 distance is

$$
W _ { 2 , y } ( \mu , \nu ) = \sqrt { \operatorname* { i n f } _ { \gamma \in \Pi ( \mu , \nu ) } \int _ { \mathcal { V } \times \mathcal { V } } \| y - y ^ { \prime } \| _ { 2 } ^ { 2 } d \gamma ( y , y ^ { \prime } ) } ,\tag{16}
$$

where $\scriptstyle \Pi ( \mu , \nu )$ is the set of couplings with marginals $\mu$ and ν. We will prove the following result:

Theorem 1 (Pixel-space Wasserstein bound for latent TVM). Assume that the TVM regularity conditions of[71, Theorem 1] hold in latent space, and thefrozen decoder D is L -Lipschitz on the relevant latent support. Then

$$
W _ { 2 , \mathscr { X } } ( p _ { \hat { x } _ { 0 } } ^ { \theta } , p _ { x _ { 0 } } ) \leq L _ { \mathscr { D } } \sqrt { \int _ { 0 } ^ { t } \lambda [ L ] ( s ) \mathscr { L } ^ { t , s } ( \theta ) d s } + \sqrt { \epsilon _ { \mathrm { V A E } } } ,\tag{17}
$$

where $\lambda [ \cdot ]$ is $a _ { , }$ functional of the Lipschitz constants $L ( s )$

Consequently, in the ideal regime where the weighted TVM/FM training objective converges to zero and the VAE reconstruction error is zero, the Wasserstein-2 bound also vanishes. For a fixed pretrained $\mathrm { V A E , ~ \epsilon _ { V A E } }$ remains a fixed reconstruction residual, while TVM training minimizes the trainable part of the bound.

Lemma 1 (Pixel-Space $W _ { 2 }$ Bound under a Lipschitz Decoder). Let D be L<sub>D</sub>-Lipschitz on the relevant latent support:

$$
\| \mathcal { D } ( z ) - \mathcal { D } ( z ^ { \prime } ) \| _ { 2 } \leq L _ { \mathcal { D } } \| z - z ^ { \prime } \| _ { 2 } .\tag{18}
$$

Then

$$
W _ { 2 , \mathcal { X } } ( p _ { \hat { x } _ { 0 } } ^ { \theta } , p _ { \tilde { x } _ { 0 } } ) { \le } L _ { \mathcal { D } } W _ { 2 , \mathcal { Z } } ( p _ { \hat { z } _ { 0 } } ^ { \theta } , p _ { z _ { 0 } } ) .\tag{19}
$$

Proof. Following the decoder pushforward argument used in $[ 9 ,$ Eq. (11)], we first write the Wasserstein-2 distance in pixel space:

$$
\begin{array} { r l } & { W _ { 2 , \mathcal { X } } ( p _ { \hat { x } _ { 0 } } ^ { \theta } , p _ { \tilde { x } _ { 0 } } ) } \\ & { = \sqrt { \underset { \gamma \in \Pi ( p _ { \hat { x } _ { 0 } } ^ { \theta } , p _ { \tilde { x } _ { 0 } } ) } { \operatorname* { i n f } } \int _ { \mathcal { X } \times \mathcal { X } } \| \hat { x } _ { 0 } - \tilde { x } _ { 0 } \| _ { 2 } ^ { 2 } d \gamma ( \hat { x } _ { 0 } , \tilde { x } _ { 0 } ) . } } \end{array}\tag{20}
$$

Using the pushforward relations induced by the decoder D in Eq. (13), together with the expectation form of the Wasserstein-2 distance [41] and the law of the unconscious statistician (LOTUS), the same distance can be expressed over latent-space couplings as

$$
\begin{array} { r l } & { W _ { 2 , \mathcal { X } } ( p _ { \hat { x } _ { 0 } } ^ { \theta } , p _ { \tilde { x } _ { 0 } } ) } \\ & { = \sqrt { \displaystyle \operatorname* { i n f } _ { \pi \in \Pi ( p _ { \hat { z } _ { 0 } } ^ { \theta } , p _ { z _ { 0 } } ) } \int _ { \mathcal { Z } \times \mathcal { Z } } \| \mathcal { D } ( \hat { z } _ { 0 } ) - \mathcal { D } ( z _ { 0 } ) \| _ { 2 } ^ { 2 } d \pi ( \hat { z } _ { 0 } , z _ { 0 } ) } . } \end{array}\tag{21}
$$

Using the Lipschitz property of D from Eq. (18) we get:

$$
\begin{array} { r l } & { W _ { 2 , \mathcal { X } } ( p _ { \hat { x } _ { 0 } } ^ { \theta } , p _ { \tilde { x } _ { 0 } } ) } \\ & { \leq \sqrt { L _ { \mathcal { D } _ { \pi \in \Pi ( p _ { \hat { z } _ { 0 } } ^ { \theta } , p _ { z _ { 0 } } ) } } ^ { 2 } \int _ { \mathcal { Z } \times \mathcal { Z } } \| \hat { z } _ { 0 } - z _ { 0 } \| _ { 2 } ^ { 2 } d \pi ( \hat { z } _ { 0 } , z _ { 0 } ) } } \\ & { = L _ { \mathcal { D } } W _ { 2 , \mathcal { Z } } ( p _ { \hat { z } _ { 0 } } ^ { \theta } , p _ { z _ { 0 } } ) . } \end{array}\tag{22}
$$

Lemma 2 (Constant-Free TVM Bound). Let $p _ { \hat { z } _ { 0 } } ^ { \theta }$ denote the latent distribution obtained by the TVM pushforward from time t to 0, and let $p _ { z _ { 0 } }$ denote the target normal-light latent distribution. Assume that the TVM regularity conditions of[71, Theorem 1] hold in latent space, then

$$
W _ { 2 , \mathcal { Z } } ^ { 2 } ( p _ { \hat { z } _ { 0 } } ^ { \theta } , p _ { z _ { 0 } } ) \leq \int _ { 0 } ^ { t } \lambda [ L ] ( s ) \big [ \mathcal { L } _ { \mathrm { T V M } } ^ { t , s } ( \theta ) + \mathcal { L } _ { \mathrm { F M } } ^ { s } ( \theta ) \big ] d s ,\tag{23}
$$

where $\lambda [ \cdot ]$ is afunctional ofthe Lipschitz constants $L ( s )$

Proof. The TVM proof [71, Eq. (56)] first obtains the following latent-space bound:

$$
\begin{array} { r l } & { W _ { 2 , \mathcal { Z } } ^ { 2 } ( p _ { \hat { z } _ { 0 } } ^ { \theta } , p _ { z _ { 0 } } ) } \\ & { \leq \displaystyle \int _ { 0 } ^ { t } \lambda [ L ] ( s ) \left[ \mathcal { L } _ { \mathrm { T V M } } ^ { t , s } ( \theta ) + \underbrace { { \mathbb E } _ { z _ { s } } \| u _ { \theta } ( z _ { s } , s ) - u ( z _ { s } , s ) \| _ { 2 } ^ { 2 } } _ { ( a ) } \right] d s . } \end{array}\tag{24}
$$

The first term is the terminal-velocity matching loss and the second term, denoted by (a), is the marginal flow-matching loss:

$$
\begin{array} { r } { ( a ) = \mathcal { L } _ { \mathrm { M F M } } ^ { s } ( \theta ) = \mathbb { E } _ { z _ { s } } \| u _ { \theta } ( z _ { s } , s ) - u ( z _ { s } , s ) \| _ { 2 } ^ { 2 } , } \end{array}\tag{25}
$$

where $u ( z _ { s } , s )$ is the marginal velocity field.

To connect (a) to the practical flow-matching objective used in our training, we use the Marginal Flow-Matching (MFM) / Conditional Flow-Matching (CFM) equivalence of Lipman et al. [30, Theorem 2].<sup>1</sup> In our notation, the conditional flow-matching loss is

$$
\mathcal { L } _ { \mathrm { F M } } ^ { s } ( \theta ) { = } \mathbb { E } _ { z _ { s } , v _ { s } } \| u _ { \theta } ( z _ { s } , s ) - v _ { s } \| _ { 2 } ^ { 2 } ,\tag{26}
$$

where the marginal velocity is the conditional expectation of the sample-wise velocity:

$$
u ( z _ { s } , s ) = \mathbb { E } [ v _ { s } | z _ { s } ] .\tag{27}
$$

Following the standard MFM/CFM relation [30, Theorem 2], add and subtract $u ( z _ { s } , s )$ :

$$
u _ { \theta } ( z _ { s } , s ) - v _ { s } = u _ { \theta } ( z _ { s } , s ) - u ( z _ { s } , s ) + u ( z _ { s } , s ) - v _ { s } .\tag{28}
$$

Expanding the squared norm and taking expectation gives

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { F M } } ^ { s } ( \theta ) = \mathcal { L } _ { \mathrm { M F M } } ^ { s } ( \theta ) + \mathbb { E } _ { z _ { s } , v _ { s } } \| u ( z _ { s } , s ) - v _ { s } \| _ { 2 } ^ { 2 } } \\ & { \qquad + 2 \mathbb { E } _ { z _ { s } , v _ { s } } [ \langle u _ { \theta } ( z _ { s } , s ) - u ( z _ { s } , s ) , u ( z _ { s } , s ) - v _ { s } \rangle ] . } \end{array}\tag{29}
$$

The cross term vanishes because

$$
\begin{array} { r } { \mathbb { E } [ u ( z _ { s } , s ) - v _ { s } \mid z _ { s } ] = u ( z _ { s } , s ) - \mathbb { E } [ v _ { s } \mid z _ { s } ] = 0 . } \end{array}\tag{30}
$$

Therefore,

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { F M } } ^ { s } ( \theta ) { = } \mathcal { L } _ { \mathrm { M F M } } ^ { s } ( \theta ) { + } \mathbb { E } _ { z _ { s } , v _ { s } } \| u ( z _ { s } , s ) { - } v _ { s } \| _ { 2 } ^ { 2 } . } \end{array}\tag{31}
$$

Since the second term is nonnegative, we obtain

$$
( a ) = \mathcal { L } _ { \mathrm { M F M } } ^ { s } ( \theta ) \leq \mathcal { L } _ { \mathrm { F M } } ^ { s } ( \theta ) .\tag{32}
$$

Substituting this inequality into Eq. (24), the marginal flow-matching term (a) is upper-bounded by the flow-matching training objective. Hence,

$$
W _ { 2 , \mathcal { Z } } ^ { 2 } ( p _ { \hat { z } _ { 0 } } ^ { \theta } , p _ { z _ { 0 } } ) \leq \int _ { 0 } ^ { t } \lambda [ L ] ( s ) \big [ \mathcal { L } _ { \mathrm { T V M } } ^ { t , s } ( \theta ) + \mathcal { L } _ { \mathrm { F M } } ^ { s } ( \theta ) \big ] d s .
$$

Lemma 3 (VAE Reconstruction Wasserstein Bound). Let $\tilde { { \boldsymbol { x } } } _ { 0 } = { \mathcal { D } } ( { \mathcal { E } } ( { \boldsymbol { x } } _ { 0 } ) )$ and let $p _ { \tilde { x } _ { 0 } } = ( \mathcal { D } \circ \mathcal { E } ) _ { \# } p _ { x _ { 0 } }$ . Assume the VAE reconstruction error isfinite:

$$
\epsilon _ { \mathrm { V A E } } = \mathbb { E } _ { \boldsymbol { x } _ { 0 } \sim p _ { \boldsymbol { x } _ { 0 } } } \| \mathcal { D } ( \mathcal { E } ( \boldsymbol { x } _ { 0 } ) ) - \boldsymbol { x } _ { 0 } \| _ { 2 } ^ { 2 } .\tag{33}
$$

Then

$$
W _ { 2 , \mathcal { X } } ( p _ { \tilde { x } _ { 0 } } , p _ { x _ { 0 } } ) { \le } \sqrt { \epsilon _ { \mathrm { V A E } } } .\tag{34}
$$

Proof. Following the reconstruction-to-Wasserstein argument used in [9, Eq. (10)], we start from the definition of the Wasserstein-2 distance:

$$
\begin{array} { r l } & { W _ { 2 , \mathcal { X } } ( p _ { \tilde { x } _ { 0 } } , p _ { x _ { 0 } } ) = } \\ & { \sqrt { \underset { \gamma \in \Pi ( p _ { \tilde { x } _ { 0 } } , p _ { x _ { 0 } } ) } { \operatorname* { i n f } } } \int _ { \mathcal { X } \times \mathcal { X } } \| \tilde { x } _ { 0 } - x _ { 0 } \| _ { 2 } ^ { 2 } d \gamma ( \tilde { x } _ { 0 } , x _ { 0 } ) . } \end{array}\tag{35}
$$

Consider the particular coupling induced by sampling x $\phantom { } _ { ) } \sim p _ { x _ { 0 } }$ and setting

$$
\begin{array} { r } { \tilde { x } _ { 0 } = \mathcal { D } ( \mathcal { E } ( x _ { 0 } ) ) . } \end{array}\tag{36}
$$

Equivalently, this coupling is

$$
\gamma _ { \mathrm { V A E } } = ( { \cal D } \circ { \mathcal E } , \mathrm { i d } ) _ { \# } p _ { x _ { 0 } } ,\tag{37}
$$

and it belongs to $\Pi ( p _ { \tilde { x } _ { 0 } } , p _ { x _ { 0 } } )$ . Since the infimum is no larger than the cost of any particular coupling,

$$
\begin{array} { l } { \displaystyle { W _ { 2 , \mathcal { X } } ( p _ { \tilde { x } _ { 0 } } , p _ { x _ { 0 } } ) \leq } } \\ { \displaystyle { \sqrt { \int _ { \mathcal { X } \times \mathcal { X } } \| \tilde { x } _ { 0 } - x _ { 0 } \| _ { 2 } ^ { 2 } d \gamma _ { \mathrm { V A E } } ( \tilde { x } _ { 0 } , x _ { 0 } ) } } . } \end{array}\tag{38}
$$

Using the definition of $\gamma _ { \mathrm { V A E } }$ , we can substitute $\widetilde { \boldsymbol { x } } _ { 0 } = \boldsymbol { D } ( \mathcal { E } ( \boldsymbol { x } _ { 0 } ) )$ 0 and marginalize over $\tilde { x } _ { 0 } \mathrm { : }$

$$
\begin{array} { r l } & { \displaystyle \int _ { \mathcal { X } \times \mathcal { X } } \| \tilde { x } _ { 0 } - x _ { 0 } \| _ { 2 } ^ { 2 } d \gamma _ { \mathrm { V A E } } ( \tilde { x } _ { 0 } , x _ { 0 } ) } \\ & { \displaystyle = \int _ { \mathcal { X } } \| \mathcal { D } ( \mathcal { E } ( x _ { 0 } ) ) - x _ { 0 } \| _ { 2 } ^ { 2 } d p _ { x _ { 0 } } ( x _ { 0 } ) } \\ & { \displaystyle = \mathbb { E } _ { x _ { 0 } \sim p _ { x _ { 0 } } } \big [ \| \mathcal { D } ( \mathcal { E } ( x _ { 0 } ) ) - x _ { 0 } \| _ { 2 } ^ { 2 } \big ] . } \end{array}\tag{39}
$$

Therefore,

$$
\begin{array} { r l } & { W _ { 2 , \mathcal { X } } ( p _ { \tilde { x } _ { 0 } } , p _ { x _ { 0 } } ) { \le } \sqrt { \mathbb { E } _ { { x _ { 0 } } \sim { p _ { x _ { 0 } } } } \| \mathcal { D } ( \mathcal { E } ( x _ { 0 } ) ) - x _ { 0 } \| _ { 2 } ^ { 2 } } } \\ & { \qquad = \sqrt { \epsilon _ { \mathrm { V A E } } } . } \end{array}\tag{40}
$$

Proof of Theorem 1. By the triangle inequality for the Wasserstein-2 metric,

$$
W _ { 2 , \mathcal { X } } ( p _ { \hat { x } _ { 0 } } ^ { \theta } , p _ { x _ { 0 } } ) { \leq } W _ { 2 , \mathcal { X } } ( p _ { \hat { x } _ { 0 } } ^ { \theta } , p _ { \tilde { x } _ { 0 } } ) { + } W _ { 2 , \mathcal { X } } ( p _ { \tilde { x } _ { 0 } } , p _ { x _ { 0 } } ) .\tag{41}
$$

Using Lemma 1,

$$
W _ { 2 , \mathcal { X } } ( p _ { \hat { x } _ { 0 } } ^ { \theta } , p _ { \tilde { x } _ { 0 } } ) { \le } L _ { \mathcal { D } } W _ { 2 , \mathcal { Z } } ( p _ { \hat { z } _ { 0 } } ^ { \theta } , p _ { z _ { 0 } } ) .\tag{42}
$$

Using Lemma 2,

$$
W _ { 2 , \mathcal { Z } } ( p _ { \hat { z } _ { 0 } } ^ { \theta } , p _ { z _ { 0 } } ) \leq \sqrt { \int _ { 0 } ^ { t } \lambda [ L ] ( s ) \mathcal { L } ^ { t , s } ( \theta ) d s } .\tag{43}
$$

Thus,

$$
W _ { 2 , \mathcal { X } } ( p _ { \hat { x } _ { 0 } } ^ { \theta } , p _ { \tilde { x } _ { 0 } } ) { \leq } L _ { \mathcal { D } } \sqrt { \int _ { 0 } ^ { t } \lambda [ L ] ( s ) \mathcal { L } ^ { t , s } ( \theta ) d s } .\tag{44}
$$

Using Lemma 3,

$$
W _ { 2 , \mathcal { X } } ( p _ { \tilde { x } _ { 0 } } , p _ { x _ { 0 } } ) { \le } \sqrt { \epsilon _ { \mathrm { V A E } } } .\tag{45}
$$

Substituting both bounds into the triangle inequality gives

$$
\begin{array} { r l r } {  { W _ { 2 , \mathcal { X } } ( p _ { \hat { x } _ { 0 } } ^ { \theta } , p _ { x _ { 0 } } ) } } \\ & { } & { \leq L _ { \mathcal { D } } \sqrt { \displaystyle \int _ { 0 } ^ { t } \lambda [ L ] ( s ) \mathcal { L } ^ { t , s } ( \theta ) d s } + \sqrt { \epsilon _ { \mathrm { V A E } } } . } \end{array}\tag{46}
$$

This proves the theorem.

## C. Additional Implementation Details

Training and inference. Unless otherwise specified, all experiments use a frozen Tiny-VAE encoder/decoder while optimizing only the latent flow model. Models are trained on 8 NVIDIA H100 GPUs with a per-GPU batch size of 2 (effective batch size 16) for approximately 100k optimization steps. Optimization uses AdamW with a learning rate of $1 0 ^ { - 5 }$ , betas (0.9,0.95), weight decay 0.01, gradient clipping at norm 1.0, and EMA decay 0.9999. The learning rate follows a 5k-step warmup with cosine decay to a minimum learning-rate ratio of 0.01. Training uses random 2048×2048 crops for LSD-DLL and UHD-LOL4K, and 384×384 crops for LOL-v1. Validation uses a batch size of 1 and is performed at the original image resolution. Source latents are perturbed with Gaussian noise using $\sigma = 0 . 0 1$ during training and $\sigma = 0$ during evaluation. Unless otherwise specified, all reported results use a single NFE.

HDiT configurations. Tab. 6 summarizes the implementation details of the HDiT backbone variants used in our experiments. Each HDiT backbone has two resolution levels: the first level uses $7 \times 7$ neighborhood attention (NA), and the second level uses global self-attention. In the table, Widths and Depths denote the channel width and the number of transformer blocks specified at each HDiT level, respectively; Map. width and Map. depth denote the width and depth of the conditioning/mapping network, and $d _ { \mathrm { h e a d } }$ denotes the attention head dimension.

Tiny-VAE latent representation. We use the official madebyollin/taesd3 [49] checkpoint as a frozen Tiny-VAE image tokenizer. TAESD is a lightweight autoencoder distilled from the Stable Diffusion VAE (SD-VAE) [44], providing a compact alternative for encoding and decoding images. RGB images are encoded into 16-channel latent representations, and the flow model operates entirely in this latent space. The encoder and decoder remain frozen throughout training, and no additional latent normalization is applied. During inference, the predicted latent representation is decoded using the same frozen Tiny-VAE decoder before computing image-space evaluation metrics.

TVM warmup schedule. The TVM loss weight is introduced gradually during training according to

$$
\begin{array} { r } { \mathcal { L } ( k ) = \mathcal { L } _ { \mathrm { F M } } + \lambda ( k ) \mathcal { L } _ { \mathrm { T V M } } , } \end{array}\tag{47}
$$

where the step-dependent TVM weight is defined as

$$
\lambda ( k ) = \left\{ \begin{array} { l l } { 0 , } & { k < W , } \\ { \displaystyle \lambda _ { \operatorname* { m a x } } \frac { k - W } { R } , } & { W \leq k < W + R , } \\ { \lambda _ { \operatorname* { m a x } } , } & { k \geq W + R . } \end{array} \right.\tag{48}
$$

Here, k denotes the optimizer step, W the warmup duration, R the linear ramp length, and $\lambda _ { \mathrm { m a x } }$ the final TVM weight. The TVM objective is disabled during warmup, introduced linearly over the ramp period, and kept fixed at $\lambda _ { \mathrm { m a x } }$ thereafter. We use $( W , R , \lambda _ { \mathrm { m a x } } ) = ( 5 \mathrm { k } , 1 0 \mathrm { k } , 1 . 0 )$ for LSD-DLL and (10k,20k,1.0) for UHD-LOL4K.

## D. Baseline Methods: Training Details

For LSD-DLL and UHD-LOL4K, we independently retrained every baseline on the corresponding training set using the authors’ official code and method-specific training recipe. We retained the official architecture, optimization settings, and training schedule, modifying only the dataset configuration as necessary. For LOL-v1, we used the official pretrained weights released by the respective authors whenever available. TFFormer was the only exception: because no official LOL-v1 checkpoint was available, we also trained it from scratch on the LOL-v1 training set. Unless otherwise stated, the training details below describe our retraining on LSD-DLL and UHD-LOL4K, and all models were trained using NVIDIA H100 GPUs.

<table><tr><td>Model</td><td>Input channels</td><td>Widths</td><td>Depths</td><td>Map. width</td><td>Map. depth</td><td>Patch</td><td> $d _ { \mathrm { h e a d } }$ </td><td>NA kernel</td></tr><tr><td>GLOW-FM-XL</td><td>16</td><td>(384,768)</td><td>(2,11)</td><td>768</td><td>2</td><td> $2 \times 2$ </td><td>64</td><td>7×7</td></tr><tr><td>GLOW-FM-M</td><td>16</td><td>(192,384)</td><td>(2,6)</td><td>384</td><td>2</td><td> $2 \times 2$ </td><td>64</td><td> $7 \times 7$ </td></tr><tr><td>GLOW-FM-S</td><td>16</td><td>(128,256)</td><td>(2,4)</td><td>256</td><td>2</td><td> $2 \times 2$ </td><td>64</td><td> $7 \times 7$ </td></tr></table>

Table 6. HDiT configurations used by GLOW-FM.

TFFormer. We trained a separate TFFormer model from scratch on each of LSD-DLL, UHD-LOL4K, and LOL-v1. TFFormer was optimized using an Adam optimizer with hyperparameters $\beta _ { 1 } = 0 . 9$ and $\beta _ { 2 } = 0 . 9 9$ . We set the initial learning rate to $1 0 ^ { - 4 }$ and adjusted it after 100,000 steps using PyTorch’s ReduceLROnPlateau scheduler [43]. We trained each model for 300,000 steps on a single GPU using a batch size of 24 and 128×128 RGB patches. Additionally, we used a low-light regularization coefficient of $\lambda _ { R } ~ = ~ 0 . 2$ for the combined reconstruction and luminance–chrominance objective.

HVI-CIDNet. HVI-CIDNet was trained with the Adam optimizer $( \beta _ { 1 } = 0 . 9 , \beta _ { 2 } = 0 . 9 9 9 )$ and $2 5 6 \times 2 5 6$ input crops using a single GPU for 1,000 epochs with a batch size of 4. The initial learning rate was $1 0 ^ { - 4 }$ and is then steadily decreased to $1 \times 1 0 ^ { - 7 }$ using a cosine annealing scheme [33] during training. We used the default HVI-CIDNet data augmentation pipeline, consisting of random cropping, random horizontal flipping, and random vertical flipping.

Retinexformer. We trained Retinexformer using the Adam optimizer with $\beta _ { 1 } = 0 . 9$ and $\beta _ { 2 } { = } 0 . 9 9 9$ , an initial learning rate $\operatorname { o f 2 } \times 1 0 ^ { - 4 }$ then steadily decreased to $1 \times 1 0 ^ { - 6 }$ by the cosine annealing scheme [33]. Training is set for $2 . 5 \times 1 0 ^ { 5 }$ iterations. We train with batch size 8 on randomly cropped 128×128 patches, and apply data augmentation of random rotation and flipping.

Multinex. We trained Multinex using the Adam optimizer with a cosine annealing learning-rate schedule [33] that decays from $2 \times 1 0 ^ { - 4 }$ to $1 \times 1 0 ^ { - 6 }$ . Multinex is trained from scratch for 150K iterations with a batch size of 8 and patch size of 256×256. Training data are augmented via random cropping, horizontal and vertical flipping, and random rotation.

PyDiff. PyDiff was trained from scratch using random $1 9 2 \times 2 8 8$ crops, horizontal flipping, and brightness augmentation with a factor sampled from [0.5,1.5]. We used Adam with an initial learning rate of $1 0 ^ { - 4 }$ and a batch size of 8 per GPU. The learning rate was multiplied by 0.5 after 50k, 75k, 100k, 150k, and 200k iterations. Training was conducted for 500k iterations using the official 2,000-step linear diffusion schedule, with variance endpoints of $1 0 ^ { - 6 }$ and $1 0 ^ { - 2 }$

WCDM. WCDM was trained from scratch using random 256 × 256 crops and a batch size of 12. The model was optimized with Adam using a learning rate of $1 0 ^ { - 4 }$ and zero weight decay. The learning rate was multiplied by 0.8 every 5,000 iterations, and an exponential moving average with a decay of 0.999 was maintained during training. Training was conducted for 1,000 epochs, following the official training schedule. We used the official 200-step linear diffusion schedule with variance endpoints of $1 0 ^ { - 4 }$ and $2 \times 1 0 ^ { - 2 }$

ReDDiT. For each of LSD-DLL and UHD-LOL4K, we followed the ReDDiT progressive-distillation procedure. We first trained the required dataset-specific GSAD teacher from scratch for 750 epochs in FP32 using random 96 × 96 crops. Training was performed across eight GPUs with a batch size of 8 per GPU, yielding an effective batch size of 64. We used Adam with a learning rate of $1 0 ^ { - 4 } .$ . The teacher objective combined the $\ell _ { 1 }$ reconstruction loss and the GSAD structural loss with weights of 10 and 1, respectively, and employed a 500-step linear diffusion schedule.

Starting from the dataset-specific teacher, we first trained an eight-step ReDDiT model and then progressively distilled it into four-step and two-step students. Each distillation stage was trained for 5,000 iterations using random 96×96 crops, a batch size of 16, Adam with a learning rate of $1 0 ^ { - 4 }$ , and an exponential moving average with a decay of 0.9999. The ReDDiT stages used a fine-grained diffusion-time discretization of T=512 with a linear $\beta \mathrm { \cdot }$ schedule whose endpoints were $1 0 ^ { - 4 }$ and $2 \times 1 0 ^ { - 2 }$

## E. Additional Ablation Studies

Extended NFE Ablation. Figures 6 and 7 analyze the effect of the number of function evaluations (NFEs). Figure 6 reports PSNR and NIQE, while Fig. 7 extends the evaluation to SSIM, LPIPS-VGG, LPIPS-Alex, MUSIQ, and CLIP-IQA. Across these metrics, the TVM-trained variants remain largely stable as the number of NFEs increases, whereas No-TVM improves substantially from one to 25 NFEs. The largest improvement is observed for LPIPS-VGG, for which No-TVM surpasses the TVM-trained model after only a few NFEs. The remaining metrics, including LPIPS-Alex, converge toward performance comparable to that of the TVM-trained model operating with a single NFE. As reported in Tab. 7, No-TVM at 25 NFEs achieves lower LPIPS; however, one-step TVM retains slightly better PSNR and NIQE with an 8.6× reduction in GFLOPs. Overall, these results are consistent with the designs of the respective objectives and demonstrate that TVM provides a favorable quality–efficiency trade-off by achieving competitive image quality with a single NFE.

![](images/5854d18c8da094f49382b68b80ea1cf396b606e3cb3176b97c944d49199d4163.jpg)  
Figure 6. NFE ablation. PSNR and NIQE on LSD-DLL as the number of function evaluations varies for GLOW-FM with and without TVM.

![](images/e74eda7bae293dd093132128cc6637ee3c50bfce559708c34da432277da0d661.jpg)

![](images/2fd7a02364021845e2e0f41f338b150e461a4413c5acda230f3fde0ef84aa021.jpg)

![](images/b1267542d5c1a2c0300af560e32be0b4144189f5ee63c0f871f9c110f34b9f34.jpg)

![](images/32ddd19482030de951f90d78f93be34b778f7a387647a1b3a9bf0aef433661b9.jpg)

![](images/2106ce8f5886adf03a7704e6dc769b08b4a9d5120a0b1b41aa9f44b4aa3c22a7.jpg)  
GLOW-FM-XL GLOW-FM-M GLOW-FM-S w/o TVM  
Figure 7. Extended NFE ablation. SSIM, LPIPS-VGG, LPIPS-Alex, MUSIQ, and CLIP-IQA on the LSD-DLL validation split as the number of function evaluations varies.

One-Step Methods across NFEs. Fig. 8 extends the matched one-step comparison by evaluating TVM, MeanFlow, ConsistencyFM, and No-TVM across different inference budgets. At one NFE, TVM provides the strongest joint distortion–perception performance. Its PSNR and NIQE remain stable as the number of NFEs increases, indicating that it already captures the enhancement trajectory in a single evaluation. No-TVM improves substantially with additional NFEs but requires many evaluations to approach one-step TVM. MeanFlow and ConsistencyFM improve NIQE with additional NFEs but remain lower in PSNR, reflecting a different distortion–perception trade-off. Overall, these results further support TVM for efficient one-step LLIE.

![](images/2d46cb3ce0b9f8fcc04d5622e050bfba1132bb9d8536eda2d062f18ed7a880f7.jpg)  
Figure 8. One-Step Methods across NFEs. PSNR and NIQE on LSD-DLL for TVM, MeanFlow, ConsistencyFM, and No-TVM as the number of NFEs varies.

Extended Ablation Study Table. Tab. 7 extends Tab. 5 from the main paper by adding CLIP-IQA [52] on LSD-DLL and reporting unseen-device generalization results on LSD-U.

Autoencoder Reconstruction Analysis. The extended architecture ablation in Tab. 7 first evaluates the effect of the latent autoencoder on end-to-end enhancement. Compared with the SD-VAE-based variant, the TAESD-based GLOW-FM achieves higher PSNR and SSIM and lower NIQE while reducing the computational cost from 471 to 48 GFLOPs and the parameter count from 205M to 124M. The SD-VAE variant achieves lower LPIPS and higher MUSIQ, reflecting a different distortion–perception trade-off.

To separate transport-model error from the reconstruction error introduced by the frozen autoencoder, we additionally evaluate direct target reconstruction, $\tilde { x } _ { 0 } = \mathcal { D } ( \mathcal { E } ( x _ { 0 } ) )$ , for TAESD and SD-VAE across the three paired benchmarks. This setting represents the performance obtained when the target encoding is recovered exactly and therefore provides a useful reconstruction reference, but not a strict bound on the output of GLOW-FM. In particular, PSNR, SSIM, and LPIPS measure correspondence with the paired target, whereas the no-reference metrics NIQE and MUSIQ assess perceptual image statistics and may improve even when reconstruction fidelity decreases. We therefore report NIQE and MUSIQ only as complementary diagnostics.

As shown in Tab. 8, direct TAESD reconstruction achieves higher PSNR and SSIM and lower LPIPS than SD-VAE reconstruction on all three datasets. This differs from the end-to-end comparison above, in which the SD-VAE variant achieves slightly better LPIPS, because direct reconstruction isolates the autoencoder from the transport model. On LSD-DLL, direct TAESD reconstruction obtains 29.49 dB PSNR and 0.069 LPIPS, compared with 20.3 dB and 0.48 for GLOW-FM. This substantial gap indicates considerable reconstruction headroom and suggests that TAESD reconstruction error is not the principal bottleneck in end-to-end enhancement performance.

<table><tr><td rowspan="2">Variant</td><td rowspan="2">Latent</td><td rowspan="2">GFLOPs↓ Params (M)↓</td><td rowspan="2"></td><td colspan="6">LSD-DLL</td><td colspan="3">LSD-U</td></tr><tr><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS.↓</td><td>MUSIQ↑</td><td>NIQE↓</td><td>CLIP-IQA↑</td><td>NIQE↓</td><td>MUSIQ↑</td><td>CLIP-IQA↑</td></tr><tr><td>Pixel Space (TVM)</td><td>None</td><td>1693</td><td>121</td><td>16.0</td><td>0.47</td><td>0.54</td><td>28.2</td><td>4.51</td><td>0.386</td><td>6.15</td><td>25.2</td><td>0.210</td></tr><tr><td>SD-VAE (TVM)</td><td>SD-VAE</td><td>471</td><td>205</td><td>19.6±3×10−2</td><td>0.504±3×10−3</td><td>0.472±3×10−2</td><td>40.58±1×10−1</td><td>3.372±1×10−3</td><td>0.609±8×10−4</td><td>4.35±1×10−3</td><td>30.4±6×10−3</td><td>0.362±2×10−4</td></tr><tr><td>No TVM (25 NFE)</td><td>TAESD</td><td>415</td><td>124</td><td>20.1</td><td>0.54</td><td>0.37</td><td>35.6</td><td>3.00</td><td>0.626</td><td>4.75</td><td>28.8</td><td>0.341</td></tr><tr><td>ConsistencyFM</td><td>TAESD</td><td>48</td><td>124</td><td>19.46</td><td>0.52</td><td>0.50</td><td>32.8</td><td>3.53</td><td>0.573</td><td>5.17</td><td>27.3</td><td>0.327</td></tr><tr><td>MeanFlow</td><td>TAESD</td><td>48</td><td>124</td><td>19.49</td><td>0.52</td><td>0.49</td><td>31.7</td><td>3.59</td><td>0.567</td><td>5.32</td><td>27.8</td><td>0.344</td></tr><tr><td>GLOW-FM</td><td>TAESD</td><td>48</td><td>124</td><td>20.3</td><td>0.55</td><td>0.48</td><td>36.0</td><td>2.92</td><td>0.625</td><td>4.33</td><td>28.4</td><td>0.367</td></tr></table>

Table 7. Extended ablation study. LSD-DLL and LSD-U results assessing the latent representation and flow-matching objective. SD-VAE results are reported as the mean ± sample standard deviation across five seeds. All other variants are deterministic and are therefore reported as single values. Best mean results are shown in bold.

<table><tr><td>Dataset</td><td>Setting</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>NIQE↓</td><td>MUSIQ↑</td></tr><tr><td rowspan="3">LOL-v1</td><td>SD-VAE Rec.</td><td>28.14</td><td>0.854</td><td>0.048</td><td>4.339</td><td>72.45</td></tr><tr><td>TAESD Rec.</td><td>30.33</td><td>0.909</td><td>0.031</td><td>3.735</td><td>72.70</td></tr><tr><td>GLOW-FM</td><td>21.8</td><td>0.828</td><td>0.29</td><td>3.71</td><td>70.4</td></tr><tr><td rowspan="3">LSD-DLL</td><td>SD-VAE Rec.</td><td>26.40</td><td>0.687</td><td>0.086</td><td>3.719</td><td>37.94</td></tr><tr><td>TAESD Rec.</td><td>29.49</td><td>0.816</td><td>0.069</td><td>3.350</td><td>35.77</td></tr><tr><td>GLOW-FM</td><td>20.3</td><td>0.548</td><td>0.48</td><td>2.92</td><td>36.0</td></tr><tr><td rowspan="3">UHD-LOL4K</td><td>SD-VAE Rec.</td><td>32.89</td><td>0.948</td><td>0.026</td><td>4.745</td><td>34.84</td></tr><tr><td>TAESD Rec.</td><td>36.11</td><td>0.981</td><td>0.019</td><td>4.552</td><td>34.48</td></tr><tr><td>GLOW-FM</td><td>31.7</td><td>0.969</td><td>0.12</td><td>4.59</td><td>35.2</td></tr></table>

Table 8. Autoencoder reconstruction analysis. Direct target reconstructions $\mathcal { D } ( \mathcal { E } ( x _ { 0 } ) )$ are compared with the corresponding GLOW-FM-XL enhancement outputs.

<table><tr><td>Autoencoder</td><td>LOL-v1</td><td>LSD-DLL</td><td>UHD-LOL4K</td></tr><tr><td>TAESD</td><td>1.181</td><td>1.792</td><td>0.266</td></tr><tr><td>SD-VAE</td><td>1.963</td><td>4.579</td><td>0.693</td></tr></table>

Table 9. Autoencoder reconstruction error. MSE↓ is reported in units of $\times 1 0 ^ { - 3 }$

Tab. 9 further reports the empirical autoencoder reconstruction term, $\epsilon _ { \mathrm { V A E } } .$ , measured using MSE. TAESD obtains MSE values of 1.181, 1.792, and 0.266, compared with 1.963, 4.579, and 0.693 for SD-VAE on LOL-v1, LSD-DLL, and UHD-LOL4K, respectively (all $\times 1 0 ^ { - 3 } )$ . Thus, TAESD yields a smaller measurable autoencoder reconstruction term on all three benchmarks. These values characterize only the autoencoder component of the $W _ { 2 }$ analysis from Theorem 1, rather than the complete bound, whose transport weighting and decoder-dependent constants are not estimated.

Although TAESD uses a substantially smaller encoder and decoder, parameter count alone does not determine the information retained by the latent representation. At each latent spatial location, TAESD encodes 16 channels, whereas SD-VAE encodes four. Because both use the same spatial downsampling factor, TAESD produces four times as many latent coefficients for a given input resolution. This higher latent dimensionality helps explain its stronger reconstruction performance despite its smaller model size.

![](images/dbfdaec7e33f6d17436022d381ff697a1c42cbf597e5839430a884f08203a3e0.jpg)  
Figure 9. TVM warmup ablation. PSNR and NIQE on LSD-DLL with and without TVM warmup. Curves are smoothed for visualization.

TVM Warmup Ablation. Fig. 9 studies the effect of gradually introducing the TVM objective during training. We compare the proposed warmup-ramp schedule against a variant trained without TVM warmup.

Without TVM warmup, the TVM objective is introduced before the flow-matching model has learned a meaningful low-to-normal-light transport. As a result, the displacement objective is optimized with respect to an immature velocity field, leading the optimization toward an inferior solution. In contrast, the proposed warmup schedule allows the flow-matching objective to first establish a reliable transport before introducing the TVM objective, resulting in improved PSNR and substantially better NIQE at convergence.

Effect of source-latent perturbation. During training, we perturb the low-light source latent as $\tilde { z } _ { 1 } = z _ { 1 } + \sigma \epsilon$ , where $\epsilon \sim \mathcal { N } ( 0 , I )$ . We evaluate four noise levels using distortion and perceptual metrics on the LSD-DLL validation set, unseendevice generalization on LSD-U, and cross-dataset mean NIQE on DICM, LIME, MEF, NPE, and VV, following the evaluation in main-paper Tab. 4. As shown in Tab. 10, σ =0.01 improves LSD-DLL PSNR and LPIPS over σ=0 and achieves the best generalization results on LSD-U and across the five cross-dataset benchmarks. Although σ=0.10 obtains better in-domain NIQE, σ =0.01 provides the best overall balance when jointly considering distortion, perceptual quality, and generalization.

<table><tr><td rowspan="2">Training σ</td><td colspan="2">LSD-DLL</td><td rowspan="2">LSD-U Cross-dataset NIQE↓</td></tr><tr><td>PSNR↑ LPIPS↓</td><td>NIQE↓</td></tr><tr><td>0.00</td><td>20.05 0.49</td><td>2.93</td><td>Mean NIQE↓ 4.72 2.87</td></tr><tr><td>0.01</td><td>20.30 0.48</td><td>2.92</td><td>4.33 2.86</td></tr><tr><td>0.05</td><td>19.91 0.50</td><td>2.90</td><td>4.88 2.90</td></tr><tr><td>0.10</td><td>19.62 0.51</td><td>2.84</td><td>4.67 2.98</td></tr></table>

Table 10. Effect of source-latent perturbation. LSD-U measures unseen-device generalization, while cross-dataset mean NIQE is averaged over DICM, LIME, MEF, NPE, and VV. Best results are shown in bold.

## F. Additional Quantitative Results

This section extends the quantitative comparisons presented in the main paper. Specifically, we report results over five evaluation seeds as mean±std and include CLIP-IQA for all benchmarks. For LOL-v1, we additionally report SSIM and LPIPS. The extended results for LOL-v1, LSD-DLL, UHD-LOL4K, and LSD-U are reported in Tabs. 11 to 14, respectively.

Runtime and Memory Analysis. Tab. 15 reports end-to-end inference latency and peak memory usage. All measurements use batch size one and report latency averaged over 100 timed inferences following 20 warm-up iterations. GPU measurements were obtained on a single NVIDIA H100 at resolutions from 256×256 to 4096×4096. We additionally evaluate 1024×1024 inference on an Apple M4 Max using ONNX with Arm SME/SME2 acceleration. SME is an Armv9-A instruction-set extension designed to accelerate matrix and vector processing [2].

On the H100, GLOW-FM-S achieves the lowest latency and peak allocated memory across all evaluated resolutions. At high resolutions, all GLOW-FM variants are substantially faster and require less GPU memory than the compared generative methods. Under Arm SME/SME2 execution, all GLOW-FM variants also achieve lower latency than the competing methods. These results demonstrate practical runtime efficiency across both server-class GPU acceleration and local, edge-oriented Arm execution.

## G. Additional Qualitative Results

This section presents additional qualitative comparisons across several benchmarks, including unseen-device generalization on LSD-U (Fig. 10), high-resolution enhancement on UHD-LOL4K (Fig. 11) and LSD-DLL (Fig. 13, Fig. 14), and lowresolution enhancement on LOL-v1 (Fig. 12). In Fig. 15, we further extend the qualitative generalization comparison shown in Fig. 4 of the main paper with magnified crops to enable closer inspection of dark regions and facilitate comparison with competing methods. Overall, the results show that GLOW-FM produces natural illumination, preserves scene structure and fine details, and avoids severe visual artifacts across diverse low-light inputs.

<table><tr><td>Method</td><td>NFEs↓ GFLOPs↓ Params (M)↓</td><td></td><td></td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>MUSIQ↑</td><td>NIQE↓</td><td>CLIP-IQA↑</td></tr><tr><td>Retinexformer [5]</td><td>1</td><td>17</td><td>1.6</td><td>30.3</td><td>0.983</td><td>0.05</td><td>34.0</td><td>4.47</td><td>0.286</td></tr><tr><td>HVI-CIDNet [62]</td><td>1</td><td>8</td><td>2.0</td><td>33.7</td><td>0.989</td><td>0.02</td><td>33.1</td><td>5.03</td><td>0.281</td></tr><tr><td>Multinex [4]</td><td>1</td><td>3</td><td>0.04</td><td>24.8</td><td>0.958</td><td>0.09</td><td>32.0</td><td>4.72</td><td>0.293</td></tr><tr><td>TFFormer [45]</td><td>1</td><td>34</td><td>5.9</td><td>31.4</td><td>0.987</td><td>0.04</td><td>32.9</td><td>4.65</td><td>0.267</td></tr><tr><td>PyDiff [70]</td><td>4</td><td>473</td><td>97.9</td><td></td><td></td><td></td><td></td><td></td><td> $2 2 . 5 \pm 3 \times 1 0 ^ { - 1 } 0 . 9 5 6 \pm 2 \times 1 0 ^ { - 3 } 0 . 1 7 \pm 6 \times 1 0 ^ { - 2 } 3 2 . 4 4 7 \pm 7 \times 1 0 ^ { - 1 } 4 . 8 3 \pm 4 \times 1 0 ^ { - 2 } 0 . 3 3 1 \pm 6 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>WCDM [20]</td><td>10</td><td>90</td><td>22.1</td><td></td><td></td><td></td><td></td><td></td><td> $2 6 . 8 \pm 5 \times 1 0 ^ { - 2 } 0 . 9 5 7 \pm 2 \times 1 0 ^ { - 5 } 0 . 1 6 \pm 3 \times 1 0 ^ { - 2 } 3 1 . 7 2 8 \pm 1 \times 1 0 ^ { - 1 } 4 . 0 8 \pm 4 \times 1 0 ^ { - 2 } 0 . 2 3 1 \pm 1 \times 1 0 ^ { - 2 }$ </td></tr><tr><td>ReDDiT [25]</td><td>2</td><td>128</td><td>17.4</td><td></td><td></td><td></td><td></td><td></td><td> $2 2 . 2 \pm 1 \times 1 0 ^ { - 1 } 0 . 9 2 4 \pm 4 \times 1 0 ^ { - 3 } 0 . 2 5 \pm 3 \times 1 0 ^ { - 3 } 3 5 . 9 3 3 \pm 2 \times 1 0 ^ { - 2 } 5 . 2 4 \pm 4 \times 1 0 ^ { - 3 } 0 . 3 0 1 \pm 2 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>GLOW-FM-XL</td><td>1</td><td>48</td><td>124</td><td>31.7</td><td>0.969</td><td>0.12</td><td>35.2</td><td>4.59</td><td>0.323</td></tr><tr><td>GLOW-FM-M</td><td>1</td><td>36</td><td>21.8</td><td>30.6</td><td>0.961</td><td>0.14</td><td>35.1</td><td>4.63</td><td>0.323</td></tr><tr><td>GLOW-FM-S</td><td>1</td><td>34</td><td>9.1</td><td>30.3</td><td>0.962</td><td>0.14</td><td>34.881</td><td>4.62</td><td>0.316</td></tr></table>

<table><tr><td>Method</td><td></td><td>NFEs↓ GFLOPs↓ Params (M)↓</td><td></td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>MUSIQ↑</td><td>NIQE↓</td><td>CLIP-IQA↑</td></tr><tr><td>PyDiff [70]</td><td>4</td><td>473</td><td>97.9</td><td> $2 1 . 9 { \pm } 2 { \times } 1 0 ^ { - 1 }$ </td><td> $\mathbf { 0 . 8 9 5 { \scriptstyle \pm 2 \times 1 0 } ^ { - 2 } }$ </td><td> $\mathbf { 0 . 1 1 \pm 3 \times 1 0 ^ { - 3 } }$ </td><td> $7 4 . 2 { \scriptstyle \pm 2 \times 1 0 ^ { - 1 } }$ </td><td> $5 . 4 0 { \scriptstyle \pm 1 \times 1 0 ^ { - 1 } }$ </td><td> $0 . 4 4 { \pm } 2 { \times } 1 0 ^ { - 2 }$ </td></tr><tr><td>WCDM [20]</td><td>10</td><td>90</td><td>22.1</td><td> ${ \bf 2 6 . 3 \pm 1 \times 1 0 ^ { - 2 } }$ </td><td> $\underline { { 0 . 8 8 7 } } \pm 2 \times 1 0 ^ { - 2 }$ </td><td> $\underline { { 0 . 1 2 } } \pm 6 { \times } 1 0 ^ { - 4 }$ </td><td> $6 6 . 9 { \pm } 1 { \times } 1 0 ^ { - 2 }$ </td><td> $4 . 5 2 { \pm } 1 { \times } 1 0 ^ { - 2 }$ </td><td> $0 . 4 0 { \scriptstyle \pm 3 \times 1 0 ^ { - 4 } }$ </td></tr><tr><td>ReDDiT [25]</td><td>2</td><td>128</td><td>17.4</td><td> $2 0 . 8 { \pm } 2 { \times } 1 0 ^ { - 1 }$ </td><td> $0 . 8 8 3 { \scriptstyle \pm 2 \times 1 0 ^ { - 3 } }$ </td><td> $0 . 2 0 { \scriptstyle \pm 2 \times 1 0 ^ { - 3 } }$ </td><td> $\phantom { 0 } { 7 1 . 7 \pm 7 \times 1 0 ^ { - 2 } }$ </td><td> $5 . 3 8 { \pm } 2 { \times } 1 0 ^ { - 2 }$ </td><td> $\mathbf { 0 . 7 0 { \pm } } 4 { \times } 1 0 ^ { - 3 }$ </td></tr><tr><td>GLOW-FM-XL</td><td>1</td><td>48</td><td>124</td><td>21.8</td><td>0.828</td><td>0.29</td><td>70.4</td><td>3.71</td><td>0.554</td></tr><tr><td>GLOW-FM-M</td><td>1</td><td>36</td><td>21.8</td><td>21.3</td><td>0.826</td><td>0.27</td><td>71.5</td><td>3.79</td><td>0.547</td></tr><tr><td>GLOW-FM-S</td><td>1</td><td>34</td><td>9.1</td><td>21.2</td><td>0.822</td><td>0.27</td><td>71.3</td><td>3.70</td><td>0.549</td></tr></table>

Table 11. Five-seed evaluation results on LOL-v1.
<table><tr><td>Method</td><td>NFEs↓ GFLOPs↓ Params (M)↓</td><td></td><td></td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>MUSIQ↑</td><td>NIQE↓</td><td>CLIP-IQA↑</td></tr><tr><td>Retinexformer [5]</td><td>1</td><td>17</td><td>1.6</td><td>16.9</td><td>0.567</td><td>0.54</td><td>28.1</td><td>4.64</td><td>0.32</td></tr><tr><td>HVI-CIDNet [62]</td><td>1</td><td>8</td><td>2.0</td><td>15.2</td><td>0.495</td><td>0.56</td><td>23.5</td><td>4.42</td><td>0.37</td></tr><tr><td>Multinex [4]</td><td>1</td><td>3</td><td>0.04</td><td>15.6</td><td>0.501</td><td>0.63</td><td>21.6</td><td>7.73</td><td>0.39</td></tr><tr><td>TFFormer [45]</td><td>1</td><td>34</td><td>5.9</td><td>14.1</td><td>0.349</td><td>0.63</td><td>32.7</td><td>4.49</td><td>0.45</td></tr><tr><td>PyDiff [70]</td><td>4</td><td>473</td><td>97.9</td><td>17.3±9×10−2</td><td>0.544±1×10−3</td><td>0.56±5×10−3</td><td>33.2±1×10−1</td><td> $7 . 3 8 { \pm } 7 { \times } 1 0 ^ { - 2 }$ </td><td> $0 . 4 2 5 { \scriptstyle \pm 2 \times 1 0 ^ { - 3 } }$ </td></tr><tr><td>WCDM [20]</td><td>10</td><td>90</td><td>22.1</td><td> $1 7 . 1 { \pm } 1 { \times } 1 0 ^ { - 1 }$ </td><td>0.553±2×10−3</td><td>0.59±7×10−4</td><td>25.5±3×10−²</td><td>5.94±4×10−2</td><td> $0 . 2 6 9 { \scriptstyle \pm 7 \times 1 0 ^ { - 4 } }$ </td></tr><tr><td>ReDDiT [25]</td><td>2</td><td>128</td><td>17.4</td><td> $1 4 . 1 { \pm } 7 { \times } 1 0 ^ { - 3 }$ </td><td>0.488±2×10−4</td><td> $0 . 5 9 { \scriptstyle \pm 2 \times 1 0 ^ { - 3 } }$ </td><td> $3 0 . 9 { \pm } 1 { \times } 1 0 ^ { - 1 }$ </td><td> $7 . 4 5 { \pm } 4 \times 1 0 ^ { - 2 }$ </td><td> $0 . 3 5 9 { \scriptstyle \pm 2 \times 1 0 ^ { - 3 } }$ </td></tr><tr><td>GLOW-FM-XL</td><td>1</td><td>48</td><td>124</td><td>20.3</td><td>0.548</td><td>0.48</td><td>36.0</td><td>2.92</td><td>0.625</td></tr><tr><td>GLOW-FM-M</td><td>1</td><td>36</td><td>21.8</td><td>19.5</td><td>0.531</td><td>0.50</td><td>35.5</td><td>2.86</td><td>0.619</td></tr><tr><td>GLOW-FM-S</td><td>1</td><td>34</td><td>9.1</td><td>19.7</td><td>0.516</td><td>0.52</td><td>37.1</td><td>2.70</td><td>0.613</td></tr></table>

Table 12. Five-seed evaluation results on LSD-DLL.

Table 13. Five-seed evaluation results on UHD-LOL4K.
<table><tr><td>Method</td><td></td><td></td><td>NFEs↓ GFLOPs↓ Params (M)↓</td><td>NIQE↓</td><td>MUSIQ↑</td><td>CLIP-IQA↑</td></tr><tr><td>Retinexformer [5]</td><td>1</td><td>17</td><td>1.6</td><td>7.35</td><td>23.4</td><td>0.205</td></tr><tr><td>HVI-CIDNet [62]</td><td>1</td><td>8</td><td>2.0</td><td>6.26</td><td>26.5</td><td>0.281</td></tr><tr><td>Multinex [4]</td><td>1</td><td>3</td><td>0.04</td><td>6.96</td><td>27.2</td><td>0.301</td></tr><tr><td>TFFormer [45]</td><td>1</td><td>34</td><td>5.9</td><td>4.42</td><td>24.1</td><td>0.271</td></tr><tr><td>PyDiff [70]</td><td>4</td><td>473</td><td>97.9</td><td> $6 . 9 6 { \pm } 5 { \times } 1 0 ^ { - 2 }$ </td><td> $2 6 . 3 { \pm } 2 { \times } 1 0 ^ { - 1 }$ </td><td> $0 . 2 7 6 { \pm } 1 { \times } 1 0 ^ { - 3 }$ </td></tr><tr><td>WCDM [20]</td><td>10</td><td>90</td><td>22.1</td><td> $6 . 1 6 { \pm } 2 { \times } 1 0 ^ { - 2 }$ </td><td> $2 3 . 6 9 2 { \scriptstyle \pm 3 \times 1 0 ^ { - 1 } }$ </td><td> $0 . 1 8 3 { \scriptstyle \pm 3 \times 1 0 ^ { - 3 } }$ </td></tr><tr><td>ReDDiT [25]</td><td>2</td><td>128</td><td>17.4</td><td> $7 . 9 4 { \pm } 2 { \times } 1 0 ^ { - 2 }$ </td><td> $2 6 . 3 7 1 { \scriptstyle \pm 5 \times 1 0 ^ { - 2 } }$ </td><td> $0 . 2 6 5 { \scriptstyle \pm 1 \times 1 0 ^ { - 3 } }$ </td></tr><tr><td>GLOW-FM-XL</td><td>1</td><td>48</td><td>124</td><td>4.33</td><td>28.4</td><td>0.367</td></tr><tr><td>GLOW-FM-M</td><td>1</td><td>36</td><td>21.8</td><td>4.49</td><td>28.3</td><td>0.358</td></tr><tr><td>GLOW-FM-S</td><td>1</td><td>34</td><td>9.1</td><td>4.99</td><td>27.7</td><td>0.347</td></tr></table>

Table 14. Five-seed evaluation results on LSD-U.

<table><tr><td rowspan="3">Method</td><td rowspan="3">NFE</td><td colspan="10">NVIDIA H100 GPU</td><td colspan="2">Apple M4 Max (SME/SME2)</td></tr><tr><td colspan="2">256×256</td><td colspan="2">512×512</td><td colspan="2">1024×1024</td><td colspan="2">2048×2048</td><td colspan="2">4096×4096</td><td colspan="2">1024×1024</td></tr><tr><td>Lat. (ms) Mem. (MB)</td><td></td><td>Lat. (ms) Mem. (MB)</td><td></td><td>Lat. (ms) Mem. (MB)</td><td></td><td>Lat. (ms) Mem. (MB)</td><td></td><td>Lat. (ms) Mem. (MB)</td><td></td><td>Lat. (ms)</td><td>Mem. (MB)</td></tr><tr><td>ReDDiT</td><td>2</td><td>31.4</td><td>291.3</td><td>65.7</td><td>859.1</td><td>286.8</td><td>3130.4</td><td>1791.4</td><td>12215.2</td><td>39704.3</td><td>48554.7</td><td>7484.1</td><td>25000.2</td></tr><tr><td>PyDiff</td><td>4</td><td>80.7</td><td>1099.9</td><td>191.7</td><td>1949.3</td><td>738.6</td><td>5346.7</td><td>4485.5</td><td>18936.2</td><td>80846.0</td><td>73294.4</td><td>23682.8</td><td>27056.0</td></tr><tr><td>WCDM</td><td>10</td><td>166.8</td><td>195.4</td><td>172.6</td><td>430.1</td><td>229.8</td><td>1369.1</td><td>974.7</td><td>5125.1</td><td>7606.8</td><td>35277.1</td><td>4987.1</td><td>4047.0</td></tr><tr><td>GLOW-FM-XL</td><td>1</td><td>40.9</td><td>559.3</td><td>45.8</td><td>708.3</td><td>65.0</td><td>1304.6</td><td>237.4</td><td>3689.6</td><td>1149.1</td><td>13229.6</td><td>3710.3</td><td>4557.8</td></tr><tr><td>GLOW-FM-M</td><td>1</td><td>31.5</td><td>165.5</td><td>37.1</td><td>314.5</td><td>59.8</td><td>910.8</td><td>197.8</td><td>3295.8</td><td>870.2</td><td>12835.8</td><td>2709.6</td><td>3526.0</td></tr><tr><td>GLOW-FM-S</td><td>1</td><td>27.6</td><td>116.4</td><td>32.7</td><td>265.4</td><td>53.8</td><td>861.7</td><td>192.3</td><td>3246.7</td><td>773.5</td><td>12786.7</td><td>1811.2</td><td>3367.2</td></tr></table>

Table 15. Runtime efficiency on an NVIDIA H100 GPU and Arm SME/SME2. We report mean end-to-end latency and peak memory usage with batch size one after 20 warm-up iterations and over 100 timed inferences. GPU results cover five input resolutions and use PyTorch peak allocated memory. SME/SME2 results are measured at 1024×1024 on an Apple M4 Max using ONNX, with memory measured as the peak resident-set size (RSS) of the inference process. Best results are shown in bold and second-best results are underlined.

![](images/8836cb87104f69521e41c6a49ba557b73281f883e0d8ac0812a9aada4d79d0c8.jpg)  
Figure 10. Qualitative results on LSD-U.

![](images/51a1b5c06961ac99eb43c8325825e1ed8506516ad45f79f037cbfe317cfa5f84.jpg)  
Figure 11. Qualitative results on UHD-LOL4K.

![](images/32dfa8d14b209594c9a2c3d29c31c79199d8dc8d2c998f37a702dd9def30d0f6.jpg)  
Figure 12. Qualitative results on LOL-v1.

![](images/706433baca1d7c2d6d7f4385958911197c706ddd890b52b8d167822a3c32d655.jpg)  
Figure 13. Qualitative results on LSD-DLL. Each full-image comparison row is followed by a magnified crop from the full-image.

![](images/5849a2728af81976725cdaace6ea1daeb42fa9e7faf6fee8e1962fe274c7bf93.jpg)  
Figure 14. Qualitative results on LSD-DLL. Each full-image comparison row is followed by a magnified crop from the full-image.

![](images/c551f90af1544d2217cf8cf3f5a4ddc219d7a5566906dba6ba63fe5b77bf8514.jpg)  
Figure 15. Qualitative comparisons on the unpaired LIME, DICM, MEF, NPE, and VV datasets. Each full-image comparison row is followed by a magnified crop from the full-image.