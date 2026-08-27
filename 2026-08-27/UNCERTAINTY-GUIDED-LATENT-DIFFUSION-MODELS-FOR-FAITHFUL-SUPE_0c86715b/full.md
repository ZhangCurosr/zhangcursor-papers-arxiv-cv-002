# UNCERTAINTY-GUIDED LATENT DIFFUSION MODELS FOR FAITHFUL SUPER RESOLUTION

Ren Wang

National Taiwan University

## ABSTRACT

The perception-distortion trade-off poses a fundamental challenge in single-image super-resolution (SR). Although diffusion-based SR methods excel at generating perceptually realistic images, achieving high fidelity remains a key limitation. Recent advances in diffusion-based SR have shown promise in improving fidelity, but these methods often compromise perceptual quality due to their high reliance on a high-fidelity image. To address this, we introduce UGDiff, a novel diffusion guidance paradigm designed to further improve the perception-distortion balance. In particular, we first estimate the reconstruction uncertainty of the latent features corresponding to a high-fidelity image. This uncertainty is then used to guide the diffusion process to selectively restore high-frequency details in high-uncertainty regions, while preserving fidelity elsewhere. Furthermore, our guidance method adaptively identifies the high-uncertainty regions by considering not only the estimated uncertainty but also the posterior variance of the diffusion sampler at each timestep. This relaxes the reliance on the high-fidelity image in the later stages of sampling, thereby achieving a better perception-distortion balance. Extensive experimental results demonstrate that our method performs favorably against state-of-the-art diffusionbased SR methods.

Index Terms— Super-resolution, perception-distortion trade-off, diffusion model, uncertainty

## 1. INTRODUCTION

Single-image super-resolution (SR) aims to recover a highresolution image $\mathbf { I } _ { \mathrm { h q } }$ from a low-resolution observation $\mathbf { I } _ { \mathrm { l q } } .$ Typically, the problem can be formulated as

$$
\mathbf { I } _ { \mathrm { h q } } = \underset { \mathbf { x } } { \arg \operatorname* { m i n } } \frac { 1 } { 2 \sigma ^ { 2 } } \left\| \mathbf { I } _ { \mathrm { l q } } - \mathcal { H } ( \mathbf { x } ) \right\| _ { 2 } ^ { 2 } + \lambda \mathcal { P } ( \mathbf { x } ) ,\tag{1}
$$

where H is the degradation model, σ is the associated noise level, and $\mathcal { P }$ is the image prior. The main challenge in optimizing this objective is that there are many possible solutions; some tend towards high-fidelity (i.e., optimizing

Yung-Yu Chuang

National Taiwan University, NTU AI-CoRE

the first term), while others lean towards perceptual quality (i.e., optimizing the second term). This is the well-known perception-distortion trade-off [1]. The motivation of this work is to achieve a better perceptual-distortion balance rather than merely optimizing for a single metric. Specifically, we refer to this objective as faithful super-resolution.

Recently, diffusion model-based SR methods [2, 3, 4, 5] have shown great capability of generating realistic highfrequency details to improve perceptual quality. However, achieving high fidelity is challenging because of the stochastic nature of diffusion models. Several prior studies have tried to address this problem. PASD [6] uses ControlNet [7] to better learn the guidance from the input image. DiffBIR [8] proposes steering the diffusion process towards a high-fidelity image through image gradient weighting. SUPIR [9] proposes a restoration-guided sampling that performs a weighted interpolation between the sampling point and the input image at each step. DADiff [10] integrates a regression-based restoration model into diffusion guidance. PiSA-SR [11] blends the outputs of pixel-level and semantic-level enhancement modules in the latent space. FaithDiff [12] proposes a feature alignment module to identify useful information from the input image. However, these methods often trade perceptual quality for fidelity due to their high reliance on the input image or a high-fidelity image.

In this work, we leverage uncertainty estimation for faithful super-resolution in latent diffusion models (LDMs) [13]. While the VAE [14] in LDMs also estimates uncertainty, it is typically ignored in practice due to the negligible KLdivergence loss during training. Consequently, we introduce a specialized encoder to estimate the uncertainty of a highfidelity image in the latent space. Following Han et al. [15], we propose a simple but effective $\mathcal { L } _ { 2 }$ loss to train the encoder. This uncertainty is then used to guide the diffusion process to selectively restore high-frequency details in high-uncertainty regions, while preserving fidelity elsewhere. Furthermore, we adaptively identify the high-uncertainty regions by incorporating not only the estimated uncertainty but also the posterior variance of the diffusion sampler. This enables our method to relax the reliance on the high-fidelity image in the later stages of sampling, thus improving the perception-distortion balance. Fig. 1 shows that our method achieves the best balance among all of the compared diffusion-based SR methods.

![](images/94d09ab9f2d51b006c40fc6c09d5f7e580aa36b7a9975ae47afc8a2918eff38a.jpg)  
Fig. 1. Perception-Distortion trade-off of diffusion-based SR methods on RealSR. The DiffBIR curve corresponds to various values of s, while ours varies γ at a fixed $s = 1 0 0$

Our contributions are summarized as follows:

• We propose a simple yet effective method to train an uncertainty estimation model in the latent space.

• We propose a novel guidance method to make the diffusion model selectively and adaptively restore highfrequency details in high-uncertainty regions.

• Extensive experiments demonstrate that our method strikes a better perception-distortion balance compared to state-of-the-art diffusion-based SR methods.

## 2. RELATED WORK

## 2.1. Diffusion-Based Image Super Resolution

SR3 [2] proposes to train a conditional diffusion model from scratch for super-resolution. SRDiff [3] and ResShift [4] propose to train diffusion models in residual space. IR-SDE [16] adds low-quality images to the initial noise during training, and lets the model be directly conditioned on the noised data at each step. StableSR [5] utilizes Stable Diffusion [13] as a strong prior for generating high-frequency information. PASD [6] leverages ControlNet [7] to better learn the guidance from input images. DiffBIR [8] uses a restored image instead of the input image as the condition of Control-Net, and proposes a restoration guidance to improve fidelity. SeeSR [17] and SUPIR [9] show that text prompts offer effective semantic guidance for super-resolution. FaithDiff [12] proposes replacing ControlNet with a feature alignment module. OSEDiff [18] and PiSA-SR [11] apply LoRA [19] to achieve one-step sampling for super-resolution. In this work, we choose DiffBIR as the baseline to build our method.

## 2.2. Uncertainty in Image Super Resolution

Kendall and Gal [20] give a fundamental framework of uncertainty in deep learning, where classification problems are based on entropy and regression problems are based on Gaussian likelihood. UDL [21] and UGAN [22] use uncertainty to identify the challenging pixels of an image and prioritize them for optimization. Liu et al. [23] use a similar framework, but operate in the frequency domain. Fang et al. [24] incorporate uncertainty learning into kernel estimation to improve the robustness of MAP-based SR. Han et al. [15] propose to train an uncertainty estimation model using optimal uncertainty when the mean is known, and integrate it into a VAE [14]-like framework for stochastic SR. UPSR [25] proposes uncertainty-guided noise weighting to enable region-specific noise level control, given that slight noise is more advantageous for reconstructing flat areas, and vice versa. BUFF [26] employs a Bayesian network to generate an uncertainty mask for diffusion models, which is used to weight the noise during training and condition the denoiser. Our work draws significant inspiration from Han et al. [15]; however, they neither used a diffusion model nor performed operations in the latent space.

## 3. METHOD

## 3.1. Preliminaries

We follow the diffusion formulation adopted by LDM [13]. Given a real image $\mathbf { x } _ { 0 } \sim q ( \mathbf { x } )$ and a pretrained encoder $\mathcal { E } ,$ LDM applies the forward process of DDPM [27] to the clean latent code $\mathbf { z } _ { 0 } = \mathcal { E } ( \mathbf { x } _ { 0 } )$ for $t \in \{ 1 , 2 , . . . , T \}$ by

$$
\begin{array} { r } { q ( \mathbf { z } _ { t } | \mathbf { z } _ { 0 } ) = \mathcal { N } ( \mathbf { z } _ { t } ; \sqrt { \bar { \alpha } _ { t } } \mathbf { z } _ { 0 } , ( 1 - \bar { \alpha } _ { t } ) \mathbf { I } ) , } \end{array}\tag{2}
$$

where $\mathbf { z } _ { t }$ is the noised latent code, $\begin{array} { r } { \bar { \alpha } _ { t } = \prod _ { i = 1 } ^ { t } ( 1 - \beta _ { i } ) } \end{array}$ , and $\beta _ { t }$ is the pre-defined variance at timestep t. Then, starting from $\mathbf { z } _ { T } \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$ , the reverse process is given by

$$
\begin{array} { r } { p ( \mathbf { z } _ { t - 1 } \vert \mathbf { z } _ { t } , \hat { \mathbf { z } } _ { 0 \vert t } ) = \mathcal { N } ( \mathbf { z } _ { t - 1 } ; \mu _ { \theta } ( \mathbf { z } _ { t } , \hat { \mathbf { z } } _ { 0 \vert t } ) , \sigma _ { t } ^ { 2 } \mathbf { I } ) , } \end{array}\tag{3}
$$

where

$$
\begin{array} { c } { \displaystyle \mu _ { \boldsymbol { \theta } } \big ( \mathbf { z } _ { t } , \hat { \mathbf { z } } _ { 0 \mid t } \big ) = \frac { \sqrt { \alpha _ { t } } \big ( 1 - \bar { \alpha } _ { t - 1 } \big ) } { 1 - \bar { \alpha } _ { t } } \mathbf { z } _ { t } + \frac { \sqrt { \bar { \alpha } _ { t - 1 } } \beta _ { t } } { 1 - \bar { \alpha } _ { t } } \hat { \mathbf { z } } _ { 0 \mid t } , } \\ { \displaystyle \sigma _ { t } ^ { 2 } = \frac { 1 - \bar { \alpha } _ { t - 1 } } { 1 - \bar { \alpha } _ { t } } \beta _ { t } , } \end{array}\tag{4}
$$

and

$$
\hat { \mathbf { z } } _ { 0 \mid t } = \frac { 1 } { \sqrt { \bar { \alpha } _ { t } } } \left( \mathbf { x } _ { t } - \sqrt { 1 - \bar { \alpha } _ { t } } \epsilon _ { \theta } ( \mathbf { z } _ { t } , t ) \right) ,\tag{5}
$$

where $\boldsymbol { \epsilon } _ { \boldsymbol { \theta } } ( \mathbf { z } _ { t } , t )$ is the noise estimated by an LDM model θ. Finally, we can obtain $\hat { \mathbf { x } } _ { 0 } = \mathcal { D } ( \hat { \mathbf { z } } _ { 0 \mid 1 } ) \sim q ( \mathbf { x } )$ given a pretrained decoder D. For SR, $\mu _ { \theta }$ is typically conditioned on $\mathcal { E } ( \mathbf { I } _ { \mathrm { l q } } ) , i . e .$ the estimated noise is $\epsilon _ { \theta } ( \mathbf { z } _ { t } , \mathcal { E } ( \mathbf { I } _ { \mathrm { l q } } ) , t )$

## 3.2. Uncertainty Estimation in Latent Space

Recall that while the pretrained encoder E also estimates uncertainty, it is typically ignored in practice due to the negligible KL-divergence loss during training. In this section, we describe how we train our uncertainty estimation model $\mathcal { E } ^ { \prime }$ in the latent space. Fig. 2 gives an overview of our method. For simplicity, we let $\mu , \pmb { \Sigma _ { \mathrm { u } } } \in \mathbb { R } ^ { d }$ respectively denote the mean and the diagonal entries of the covariance matrix, where $d = H \times W \times C$ . As in [20], we can train $\mathcal { E } ^ { \prime }$ by minimizing the Gaussian negative log-likelihood (NLL) as

![](images/fed120e6b0b47d0d94fe7fca8a2572dff51594b75013e27a063e104a6c68ff17.jpg)  
Fig. 2. Overview of our method. Given a high-fidelity image $\mathbf { I } _ { \mathrm { r m } }$ restored from the low-quality image $\mathbf { I } _ { \mathrm { l q } }$ by a regression-based restoration model $\mathcal { R } ,$ our uncertainty estimation model $\mathcal { E } ^ { \prime }$ produces a variance map $\Sigma _ { \mathrm { u } }$ specifically for $\pmb { \mu } = \mathcal { E } ( \mathbf { I } _ { \mathrm { r m } } )$ . During the reverse process of a pretrained latent diffusion model, we rectify $\mu _ { \boldsymbol { \theta } } ( \mathbf { z } _ { t } , \hat { \mathbf { z } } _ { 0 \mid t } )$ , i.e., the mean of the posterior distribution $p ( \mathbf { z } _ { t - 1 } | \mathbf { z } _ { t } , \hat { \mathbf { z } } _ { 0 | t } )$ , at each timestep t. Our guidance aims to push $\mu _ { \boldsymbol { \theta } } \big ( \mathbf { z } _ { t } , \hat { \mathbf { z } } _ { 0 \mid t } \big )$ to the direction of $\pmb { \mu }$ according to the uncertainty at each pixel. Repeating this process, we get the final restored image $\mathbf { I } _ { \mathrm { h q } }$ at timestep 0, which is expected to have plentiful details in textured regions while remaining robust against hallucinations and artifacts elsewhere.

$$
\mathcal { L } _ { \mathrm { n l l } } = \mathbb { E } \left[ \sum _ { i = 1 } ^ { d } \frac { ( \mathbf { z } _ { \mathrm { g t } , i } - \pmb { \mu } _ { i } ) ^ { 2 } } { 2 \pmb { \Sigma } _ { \mathrm { u } , i } } + \frac { 1 } { 2 } \ln \pmb { \Sigma } _ { \mathrm { u } , i } \right] ,\tag{6}
$$

where $\mathbf { z } _ { \mathrm { g t } } ~ = ~ \mathcal { E } ( \mathbf { I } _ { \mathrm { g t } } )$ for the ground-truth image $\mathbf { I } _ { \mathrm { g t } }$ . Typically, both $\pmb { \mu }$ and $\Sigma _ { \mathrm { u } }$ are estimated by the same model, requiring a non-trivial balance between the two loss terms during training. Inspired by Han et al. [15], if $\pmb { \mu }$ is estimated by an existing regression-based restoration model $\mathcal { R }$ , i.e., $\pmb { \mu } =$ $\mathcal { E } ( \mathcal { R } ( \mathbf { I } _ { \mathrm { l q } } ) ) = \mathcal { E } ( \mathbf { I } _ { \mathrm { r m } } )$ , we can have the optimal uncertainty as

$$
\pmb { \Sigma } _ { \mathrm { u } } ^ { \star } = \underset { \pmb { \Sigma } _ { \mathrm { u } } } { \arg \operatorname* { m i n } } \mathcal { L } _ { \mathrm { n l l } } = ( \mathbf { z } _ { \mathrm { g t } } - \pmb { \mu } ) \odot ( \mathbf { z } _ { \mathrm { g t } } - \pmb { \mu } ) .\tag{7}
$$

Then, we can train $\mathcal { E } ^ { \prime }$ by a simple $\mathcal { L } _ { 2 }$ loss as

$$
\mathcal { L } _ { \mathrm { u n c e r t a i n t y } } = \mathbb { E } \left[ \left. \pmb { \Sigma } _ { \mathrm { u } } - \pmb { \Sigma } _ { \mathrm { u } } ^ { \star } \right. _ { 2 } ^ { 2 } \right] ,\tag{8}
$$

where $\pmb { \Sigma } _ { \mathbf { u } } = \mathcal { E } ^ { \prime } ( \mathbf { I } _ { \mathrm { r m } } )$ . With this simple training objective, our model can focus on uncertainty estimation without compromising mean accuracy.

## 3.3. Uncertainty-Driven Diffusion Guidance

In this section, we describe how we use the estimated uncertainty to guide the sampling process. We start from the guidance proposed by DiffBIR [10], which is given by

$$
\mathcal { L } _ { \mathrm { d i f f b i r } } = \left| \left| \mathbf { 1 } - \operatorname { t a n h } ( \mathcal { G } ( \mathbf { I } _ { \mathrm { r m } } ) ) ) \odot \left( \mathcal { D } ( \hat { \mathbf { z } } _ { 0 \mid t } ) - \mathbf { I } _ { \mathrm { r m } } \right) \right| \right| _ { 2 } ^ { 2 } ,\tag{9}
$$

$$
\mathbf { z } _ { t - 1 } \sim \mathcal { N } ( \mu _ { \boldsymbol { \theta } } ( \mathbf { z } _ { t } , \hat { \mathbf { z } } _ { 0 | t } ) - s \nabla _ { \hat { \mathbf { z } } _ { 0 | t } } \mathcal { L } _ { \mathrm { d i f f b i r } } , \sigma _ { t } ^ { 2 } \mathbf { I } ) ,\tag{10}
$$

where $\mathcal { G }$ is an image gradient operator, $\mu _ { \theta }$ is conditioned on $\mu = \mathcal { E } ( \mathbf { I } _ { \mathrm { r m } } )$ , and s is the guidance scale. This guidance lets the sampling process move towards $\mathbf { I } _ { \mathrm { r m } }$ for low-textured areas and thus enhances the fidelity.

Our core idea is to make this guidance more adaptive according to the estimated uncertainty. Specifically, we form a product of Gaussians by multiplying the estimated uncertainty distribution by the posterior in Eq. (3), then obtain its maximizer by

$$
\begin{array} { r l } & { \mathbf z _ { t - 1 } ^ { \star } = \underset { \mathbf z _ { t - 1 } } { \arg \operatorname* { m a x } } p ( \mathbf z _ { t - 1 } | \mu , \boldsymbol { \Sigma } _ { \mathbf { u } } ^ { \prime } ) \cdot p ( \mathbf z _ { t - 1 } | \mathbf z _ { t } , \hat { \mathbf z } _ { 0 } | \boldsymbol { t } ) } \\ & { \quad \quad \quad = \underset { \mathbf z _ { t - 1 } } { \arg \operatorname* { m a x } } \mathcal N ( \mathbf z _ { t - 1 } ; \sqrt { \bar { \alpha } _ { t - 1 } } \mu , \bar { \alpha } _ { t - 1 } \operatorname { d i a g } ( \boldsymbol { \Sigma } _ { \mathbf { u } } ^ { \prime } ) ) } \\ & { \quad \quad \quad \quad \cdot \mathcal N ( \mathbf z _ { t - 1 } ; \mu \boldsymbol { \theta } ( \mathbf z _ { t } , \hat { \mathbf z } _ { 0 } | \boldsymbol { t } ) , \sigma _ { t } ^ { 2 } \mathbf I ) } \\ & { \quad \quad \quad = \frac { \sigma _ { t } ^ { 2 } \cdot \sqrt { \bar { \alpha } _ { t - 1 } } \mu + \bar { \alpha } _ { t - 1 } \boldsymbol { \Sigma } _ { \mathbf { u } } ^ { \prime } \cdot \mu _ { \boldsymbol { \theta } } \left( \mathbf z _ { t } , \hat { \mathbf z } _ { 0 } | \boldsymbol { t } \right) } { \sigma _ { t } ^ { 2 } \mathbf { 1 } + \bar { \alpha } _ { t - 1 } \boldsymbol { \Sigma } _ { \mathbf { u } } ^ { \prime } } , } \end{array}\tag{11}
$$

where $\pmb { \Sigma } _ { \mathrm { u } } ^ { \prime } = \gamma \pmb { \Sigma } _ { \mathrm { u } }$ and $\gamma \in \mathbb { R } ^ { + }$ is the uncertainty scale to control the perception-distortion balance (lower for better fidelity), and $\begin{array} { r } { \mathbb E [ \mathbf z _ { t - 1 } | \pmb { \mu } ] = \sqrt { \bar { \alpha } _ { t - 1 } } \pmb { \mu } } \end{array}$ is based on Eq. (2). Finally, our diffusion guidance is given by

$$
\mathcal { L } _ { \mathrm { u g d i f f } } = \left. \sqrt { \bar { \alpha } _ { t - 1 } } \hat { \mathbf { z } } _ { 0 \mid t } - \mathbf { z } _ { t - 1 } ^ { \star } \right. _ { 2 } ^ { 2 } ,\tag{12}
$$

which is substituted into Eq. (10) to replace ${ \mathcal { L } } _ { \mathrm { d i f f b i r } }$ , making the sampling process move towards $\mathbf { z } _ { t - 1 } ^ { \star }$ instead of $\mathbf { I } _ { \mathrm { r m } }$

It is worth noting that Eq. (11) actually means that we perform a linear combination on the mean of the two distributions. As we can see in Fig. 4, the relative ratio of the weighting $\rho = \sigma _ { t } ^ { 2 } / ( \bar { \alpha } _ { t - 1 } \sigma _ { \mathrm { u } } ^ { 2 } )$ increases as the uncertainty $\sigma _ { \mathrm { u } } ^ { 2 }$ decreases, causing the sampling to go towards the direction of $\mu .$ Furthermore, $\rho$ is typically very large at the early steps but decreases faster later, allowing our method to relax reliance on $\pmb { \mu }$ at the later steps and improve perceptual quality.

![](images/46a527f3be00c8b4366ee47005ed14342001c87e420745cb5ebfb733e9b3364a.jpg)  
Fig. 3. Comparison of 4× upsampling on DIV2K-Val (upper) and RealSR (bottom). We set $s = 1 0 0$ for all of our methods.

![](images/1e04593fd9c9b61b77bebde232ada83b36a75a7e6d0ce121dbfbd0e632da5604.jpg)  
Fig. 4. Relative ratio of weighting over time under different uncertainties $\sigma _ { \mathrm { u } } ^ { 2 } .$ . The ratio is defined by $\rho = \sigma _ { t } ^ { 2 } / ( \bar { \alpha } _ { t - 1 } \sigma _ { \mathrm { u } } ^ { 2 } )$ Note that the curves are depicted using the scaled linear noise scheduler with $\beta _ { 1 } = 8 . 5 \times 1 0 ^ { - 4 }$ and $\beta _ { T } = 1 . 2 \times 1 0 ^ { - 2 }$

## 4. EXPERIMENTS

## 4.1. Experiment Settings

Data preparation. We use the test datasets from StableSR [5]: a synthetic dataset DIV2K-Val [28] and a real dataset RealSR [29], both of which involve 4× upsampling from 128 × 128 to 512 × 512. We train our uncertainty estimation model $\mathcal { E } ^ { \prime }$ on LSDIR [30] and the first 10K face images from FFHQ [31], where the LQ-HQ pairs are generated by the degradation pipeline from Real-ESRGAN [32]. We also randomly generate 100 crops from DIV2K [28] for validation, denoted by DIV2K-Ours.

Implementation details. We build our method upon the DiffBIR [10] baseline. We use the ControlNet [7] model from DiffBIR, and the pretrained diffusion and VAE [14] models from Stable Diffusion v2.1 [13]. For the text encoder, we use the same negative prompts as DiffBIR and keep the positive prompts empty. We also adopt BSRNet [33] as our restoration model R. We use the VAE architecture and train from scratch our uncertainty estimation model $\mathcal { E } ^ { \prime }$ using AdamW [34] with a learning rate of $1 0 ^ { - 4 } .$ , a batch size of 128, and a crop size of 256 × 256 on 4 NVIDIA A100 GPUs for 90K iterations. For sampling, we use the spaced DDPM [35] sampler with the scaled linear noise scheduler, where $\beta _ { 1 } = 8 . 5 \times 1 0 ^ { - 4 }$ and $\beta _ { T } = 1 . 2 \times 1 0 ^ { - 2 }$ , for totally T = 50 steps.

Methods in comparison. We compare our methods with the state-of-the-art diffusion-based SR methods, including StableSR [5], ResShift [4], SeeSR [17], FaithDiff [12], SUPIR [9], PiSA-SR [11], and DiffBIR [8]. We provide all the numbers using the official code with default settings.

Evaluation metrics. There are 4 metrics used for evaluation. PSNR and SSIM are reference-based metrics for fidelity. LPIPS [36] is a reference-based metric, and NIQE [37] is a non-reference metric, both for perceptual quality.

Table 1. Quantitative comparisons of 4× upsampling. Underline indicates superior performance between DiffBIR and ours at comparable PSNR levels. Red and blue respectively indicate the best and the second best performance among all methods. We set s = 100 for all of our methods.
<table><tr><td>Dataset</td><td>Method</td><td>PSNR ↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>NIQE↓</td></tr><tr><td rowspan="5">DIV2K-Val</td><td>StableSR [5] ResShift [4] SeeSR [17]</td><td>21.6100 22.6640 21.9703</td><td>0.5267 0.5884 0.5592</td><td>0.3113 0.3077 0.3194</td><td>4.7581 6.9142 4.8097</td></tr><tr><td>FaithDiff [12] SUPIR [9] PiSA-SR [11]</td><td>21.7965 21.4792 22.2189</td><td>0.5348 0.5005 0.5624</td><td>0.3118 0.3625 0.2823</td><td>4.9149 6.3373 4.5577</td></tr><tr><td>DiffBIR [8] (s = 0.0) DiffBIR [8] (s = 0.15) DiffBIR [8] (s = 0.5)</td><td>21.4790 22.2730 22.7957</td><td>0.4969 0.5418 0.5707</td><td>0.3670 0.3413 0.3419</td><td>4.9934 5.2346</td></tr><tr><td>Ours (γ = 0.5) Ours (γ = 0.1)</td><td>21.5034 22.2350</td><td>0.5087</td><td>0.3510</td><td>5.6970 4.1650</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="7">RealSR</td><td></td><td></td><td>0.5471</td><td>0.3406</td><td>4.7852</td></tr><tr><td>Ours (γ = 0.01)</td><td>22.7391</td><td>0.5737</td><td>0.3430</td><td>5.3038</td></tr><tr><td>StableSR [5] ResShift [4]</td><td>23.1258</td><td>0.6722</td><td>0.3002</td><td>5.8812</td></tr><tr><td>SeeSR [17]</td><td>23.9576 23.5972</td><td>0.7049 0.6872</td><td>0.3279 0.3007</td><td>8.0707 5.3957</td></tr><tr><td>FaithDiff [12]</td><td>23.7191</td><td>0.6721</td><td>0.2887</td><td>5.3895</td></tr><tr><td>SUPIR [9]</td><td>23.4637</td><td>0.6326</td><td>0.3716</td><td></td></tr><tr><td>PiSA-SR [11]</td><td>23.9644</td><td>0.7089</td><td>0.2672</td><td>7.5991 5.5014</td></tr><tr><td></td><td>DiffBIR [8] (s = 0.0)</td><td>23.3298</td><td>0.6134</td><td>0.3652</td><td>5.8397</td></tr><tr><td>DiffBIR [8] (s = 0.3)</td><td></td><td>24.0969</td><td>0.6707</td><td>0.3279</td><td>6.2908</td></tr><tr><td>DiffBIR [8] (s = 1.0)</td><td></td><td>24.6003</td><td>0.7058</td><td>0.3105</td><td>6.8355</td></tr><tr><td>Ours (γ = 0.5)</td><td>23.4219</td><td></td><td>0.6301</td><td>0.3324</td><td>4.9070</td></tr><tr><td>Ours (γ = 0.1) Ours (γ = 0.01)</td><td></td><td>24.1528 24.5990</td><td>0.6700 0.6946</td><td>0.3170 0.3072</td><td>5.6148</td></tr></table>

## 4.2. Ablation Studies

Ablation study on mean estimation. To validate the importance of Eq. (8), we train an uncertainty estimation model ${ \mathcal { E } } _ { \mathrm { n l l } }$ using Eq. (6) and employ its mean estimation. Table 2 shows that using the mean estimation from BSRNet yields superior performance across all metrics on DIV2K-Ours, thereby justifying our focus on accurate uncertainty estimation.

Ablation study on guidance parameters. Table 3 shows the ablation study of guidance parameters on DIV2K-Ours. In DiffBIR, fidelity improves while perceptual quality decreases as the guidance scale s increases, but our method exhibits the opposite trend. This demonstrates a fundamental distinction between our method and DiffBIR: rather than merely pulling the results towards the restored image $\mathbf { I } _ { \mathrm { r m } } .$ , we alter the convergence characteristics of the sampling process. Therefore, we fix s = 100 because it achieves the optimal balance among all configurations with $\gamma = 1 . 0$ . With s fixed at 100, different values of $\gamma$ offer different trade-offs. Specifically, a smaller $\gamma$ results in a stronger pull towards the mean $\mu ,$ thereby improving fidelity, and vice versa. Given its clear influence on the perception-distortion trade-off, we adopt $\gamma$ as the primary parameter to control this balance throughout our experiments.

Table 2. Ablation study on mean estimation. Bold indicates superior performance. We conduct the experiments using DiffBIR [8] without guidance $( s = 0 )$
<table><tr><td> $\mathbf { I } _ { \mathrm { r m } }$ </td><td>PSNR ↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>NIQE↓</td></tr><tr><td> $\mathbf { B S R N e t } ( \mathbf { I } _ { \mathrm { l q } } )$ </td><td>20.2307</td><td>0.4222</td><td>0.4388</td><td>4.6974</td></tr><tr><td> $\mathcal { D } ( \mathcal { E } _ { \mathrm { n l l } } ( \dot { \mathbf { I } _ { \mathrm { l q } } } ) ) )$ </td><td>19.5111</td><td>0.3844</td><td>0.4854</td><td>5.6106</td></tr></table>

Table 3. Ablation study on guidance parameters. Bold indicates the best performance.
<table><tr><td>S</td><td>γ PSNR ↑</td><td>SSIM ↑</td><td>LPIPS↓</td><td>NIQE↓</td></tr><tr><td>1</td><td>1.0 20.2278</td><td>0.4222</td><td>0.4387</td><td>4.6801</td></tr><tr><td>10 1.0</td><td>20.2040</td><td>0.4219</td><td>0.4371</td><td>4.5573</td></tr><tr><td>100 1.0</td><td>19.8139</td><td>0.4113</td><td>0.4271</td><td>3.7287</td></tr><tr><td>200 1.0</td><td>18.9496</td><td>0.3779</td><td>0.4628</td><td>4.0655</td></tr><tr><td>100</td><td>1.0 19.8139</td><td>0.4113</td><td>0.4271</td><td>3.7287</td></tr><tr><td>100</td><td>0.5 20.2100</td><td>0.4340</td><td>0.4185</td><td>3.8541</td></tr><tr><td>100</td><td>0.1 20.9637</td><td>0.4780</td><td>0.4095</td><td>4.5425</td></tr><tr><td>100 0.01</td><td>21.4769</td><td>0.5085</td><td>0.4134</td><td>5.2745</td></tr></table>

## 4.3. Comparisons with State-of-the-Arts

Quantitative comparisons. Table 1 shows the quantitative comparisons. Given that DiffBIR serves as our baseline, we first compare our method against it at comparable PSNR levels. As demonstrated in the table, our method consistently achieves superior perceptual quality. Furthermore, our method performs favorably against other state-of-the-art methods, particularly in terms of NIQE scores.

Fig. 1 shows the perception-distortion trade-off of diffusionbased SR methods on RealSR using PSNR and NIQE. As we can see in the figure, our method achieves the best perceptiondistortion balance among all of the compared methods.

Qualitative comparisons. Fig. 3 shows the qualitative comparisons on DIV2K-Val and RealSR. Our method $( \gamma = 0 . 5 )$ yields visually similar results to DiffBIR $( s = 0 )$ when their PSNR values are comparable. If we decrease $\gamma$ from 0.5 to 0.1, we can avoid unnecessary high-frequency details particularly in flat regions, showing the effectiveness of our guidance method. Regarding other methods, PiSA-SR, SeeSR, and FaithDiff often introduce details in undesired regions, while ResShift’s results lack high-frequency details.

## 5. CONCLUSION

In this paper, we have presented a method for training an uncertainty estimation model in the latent space and leveraging this uncertainty to guide the diffusion process. The experimental results have demonstrated that our proposed method strikes a better perception-distortion balance than state-of-the-art methods. For future work, distilling the diffusion process towards single-step or extending our method to a broader range of tasks could be possible directions.

## 6. REFERENCES

[1] Y. Blau and T. Michaeli, “The perception-distortion tradeoff,” in CVPR, 2018.

[2] C. Saharia, J. Ho, W. Chan, T. Salimans, D. J. Fleet, and M. Norouzi, “Image super-resolution via iterative refinement,” IEEE TPAMI, vol. 45, no. 4, 2023.

[3] H. Li, Y. Yang, M. Chang, H. Feng, Z. Xu, Q. Li, and Y. Chen, “Srdiff: Single image super-resolution with diffusion probabilistic models,” Neurocomputing, vol. 479, 2022.

[4] Z. Yue, J. Wang, and C. C. Loy, “Efficient diffusion model for image restoration by residual shifting,” IEEE TPAMI, vol. 47, no. 1, 2025.

[5] J. Wang, Z. Yue, S. Zhou, K. C. K. Chan, and C. C. Loy, “Exploiting diffusion prior for real-world image super-resolution,” IJCV, vol. 132, 2024.

[6] T. Yang, R. Wu, P. Ren, X. Xie, and L. Zhang, “Pixel-aware stable diffusion for realistic image super-resolution and personalized stylization,” in ECCV, 2024.

[7] L. Zhang, A. Rao, and M. Agrawala, “Adding conditional control to text-to-image diffusion models,” in ICCV, 2023.

[8] X. Lin, J. He, Z. Chen, Z. Lyu, B. Dai, F. Yu, W. Ouyang, Y. Qiao, and C. Dong, “Diffbir: Towards blind image restoration with generative diffusion prior,” in ECCV, 2024.

[9] F. Yu, J. Gu, Z. Li, J. Hu, X. Kong, X. Wang, J. He, Y. Qiao, and C. Dong, “Scaling up to excellence: Practicing model scaling for photo-realistic image restoration in the wild,” in CVPR, 2024.

[10] S.-H. Lu, R. Wang, C.-C. Huang, and W.-C. Chiu, “Boosting diffusion guidance via learning degradation-aware models for blind super resolution,” in WACV, 2025.

[11] L. Sun, R. Wu, Z. Ma, S. Liu, Q. Yi, and L. Zhang, “Pixellevel and semantic-level adjustable super-resolution: A duallora approach,” in CVPR, 2025.

[12] J. Chen, J. Pan, and J. Dong, “Faithdiff: Unleashing diffusion priors for faithful image super-resolution,” in CVPR, 2025.

[13] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer, “High-resolution image synthesis with latent diffusion models,” in CVPR, 2022.

[14] D. P. Kingma and M. Welling, “Auto-encoding variational bayes,” in ICLR, 2014.

[15] D. Han, S. Hwang, H. Ahn, and M. Jeon, “An efficient uncertainty-driven learning for stochastic super-resolution,” IEEE Access, vol. 13, 2025.

[16] Z. Luo, F. K. Gustafsson, Z. Zhao, J. Sjolund, and T. B. Sch ¨ on,¨ “Image restoration with mean-reverting stochastic differential equations,” in ICML, 2023.

[17] R. Wu, T. Yang, L. Sun, Z. Zhang, S. Li, and L. Zhang, “Seesr: Towards semantics-aware real-world image super-resolution,” in CVPR, 2024.

[18] R. Wu, L. Sun, Z. Ma, and L. Zhang, “One-step effective diffusion network for real-world image super-resolution,” in NeurIPS, 2024.

[19] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen, “Lora: Low-rank adaptation of large language models,” in ICLR, 2022.

[20] A. Kendall and Y. Gal, “What uncertainties do we need in bayesian deep learning for computer vision?” in NIPS, 2017.

[21] Q. Ning, W. Dong, X. Li, J. Wu, and G. Shi, “Uncertaintydriven loss for single image super-resolution,” in NeurIPS, 2021.

[22] C. Ma, “Uncertainty-aware gan for single image super resolution,” in AAAI, 2024.

[23] T. Liu, J. Cheng, and S. Tan, “Spectral bayesian uncertainty for image super-resolution,” in CVPR, 2023.

[24] Z. Fang, W. Dong, X. Li, J. Wu, L. Li, and G. Shi, “Uncertainty learning in kernel estimation for multi-stage blind image superresolution,” in ECCV, 2022.

[25] L. Zhang, W. You, K. Shi, and S. Gu, “Uncertainty-guided perturbation for image super-resolution diffusion model,” in CVPR, 2025.

[26] Z. He, S. Zhang, R. Hu, Y. Shen, and Y. Zhang, “Buff: Bayesian uncertainty guided diffusion probabilistic model for single image super-resolution,” in AAAI, 2025.

[27] J. Ho, A. Jain, and P. Abbeel, “Denoising diffusion probabilistic models,” in NeurIPS, 2020.

[28] E. Agustsson and R. Timofte, “Ntire 2017 challenge on single image super-resolution: Dataset and study,” in CVPRW, 2017.

[29] J. Cai, H. Zeng, H. Yong, Z. Cao, and L. Zhang, “Toward realworld single image super-resolution: A new benchmark and a new model,” in ICCV, 2019.

[30] Y. Li, K. Zhang, J. Liang, J. Cao, C. Liu, R. Gong, Y. Zhang, H. Tang, Y. Liu, D. Demandolx, and et al, “Lsdir: A large scale dataset for image restoration,” in CVPR, 2023.

[31] T. Karras, S. Laine, and T. Aila, “A style-based generator architecture for generative adversarial networks,” in CVPR, 2019.

[32] X. Wang, L. Xie, C. Dong, and Y. Shan, “Real-esrgan: Training real-world blind super-resolution with pure synthetic data,” in ICCVW, 2021.

[33] K. Zhang, J. Liang, L. Van Gool, and R. Timofte, “Designing a practical degradation model for deep blind image superresolution,” in ICCV, 2021.

[34] I. Loshchilov and F. Hutter, “Decoupled weight decay regularization,” in ICLR, 2019.

[35] A. Nichol and P. Dhariwal, “Improved denoising diffusion probabilistic models,” in ICML, 2021.

[36] Z. Wang, A. Bovik, H. Sheikh, and E. Simoncelli, “Image quality assessment: from error visibility to structural similarity,” IEEE TIP, vol. 13, no. 4, 2004.

[37] L. Zhang, L. Zhang, and A. C. Bovik, “A feature-enriched completely blind image quality evaluator,” IEEE TIP, vol. 24, no. 8, 2015.

[38] P. Wei, Z. Xie, H. Lu, Z. Zhan, Q. Ye, W. Zuo, and L. Lin, “Component divide-and-conquer for real-world image superresolution,” in ECCV, 2020.

# UNCERTAINTY-GUIDED LATENT DIFFUSION MODELS FOR FAITHFUL SUPER RESOLUTION

# Supplementary Material

## A. ABLATION STUDY ON PARAMETERIZATION

Table 4 shows the ablation study of the parameterization of our uncertainty estimation model on DIV2K-Ours. By parameterization, we mean whether to use $\sigma _ { \mathrm { u } }$ or $\sigma _ { \mathrm { u } } ^ { 2 }$ as the output of the model. The parameterization with $\sigma _ { \mathrm { u } }$ performs better in terms of most metrics, so we choose it as our final solution, i.e., $\Sigma _ { \mathrm { u } } = \mathcal { E } ^ { \prime } ( \mathbf { I } _ { \mathrm { r m } } ) \odot \mathcal { E } ^ { \prime } ( \mathbf { I } _ { \mathrm { r m } } )$

Table 4. Ablation study on parameterization. Bold indicates the better performance. We set $s = 1 0 0$ and $\gamma = 0 . 1$ for the experiments.
<table><tr><td>Param.</td><td>PSNR↑</td><td>SSIM ↑</td><td>LPIPS↓</td><td>NIQE↓</td></tr><tr><td> $\sigma _ { \mathrm { u } } ^ { 2 }$ </td><td>20.7987</td><td>0.4676</td><td>0.4118</td><td>4.3496</td></tr><tr><td> $\sigma _ { \mathrm { u } }$ </td><td>20.9637</td><td>0.4780</td><td>0.4095</td><td>4.5425</td></tr></table>

## B. VISUAL COMPARISON ON UNCERTAINTY SCALE

We provide a comparison on the uncertainty scale $\gamma$ in Fig. 5. As shown in the upper part of the figure, a higher value of γ is preferred when richer high-frequency details are desired. In contrast, we can also use a lower γ to avoid unnecessary details, making the result look more like the restoration model BSRNet [33], as in the bottom part.

![](images/3900d167f46739ca0beb6fc5a33a19080766252dc6fb0b596e04ead3c75fef88.jpg)  
Fig. 5. Visual comparison under different uncertainty scales on DIV2K-Val (upper) and RealSR (bottom). We set $s = 1 0 0$ for all of our methods.

## C. IMPACT OF GUIDANCE ON THE SAMPLING PROCESS

Fig. 6 presents a qualitative and quantitative comparison of the sampling process, both with and without the proposed guidance. Recall that there are a total of 50 sampling steps; we select 6 steps within this range to visualize the corresponding $\mathcal { D } ( \hat { \mathbf { z } } _ { 0 \mid t } )$ . As illustrated in the figure, our proposed guidance suppresses unnecessary textures in low-uncertainty regions, thereby achieving higher PSNR eventually. Furthermore, the NIQE curve with our guidance is nearly identical to the baseline, demonstrating that our adaptive mechanism, which relaxes the reliance on µ at later sampling steps, effectively preserves perceptual quality.

![](images/b5b1b2801fc5e99d3301120b9b3c6112c7ba329e107278ce9206ec1666ffa509.jpg)  
Fig. 6. Visualization and analysis of the sampling process with and w/o our guidance. We set $s = 1 0 0$ and $\gamma = 0 . 1$ for the guidance.

## D. MORE QUALITATIVE COMPARISONS WITH STATE-OF-THE-ARTS

We provide more qualitative comparisons with state-of-the-arts in Figs. 7–8, where Fig. 7 uses the DRealSR [38] dataset from StableSR [5]. Consistent with the observations in Fig. 3 of the main paper, our method performs similiarly to DiffBIR [8] with $\gamma = 0 . 5$ . By setting $\gamma = 0 . 1$ , we can remove unnecessary texture details, and hence achieve a better perception-distortion balance than the compared methods.

## E. PROOFS

## E.1. Equation (7): Optimal Uncertainty

If $\mathbf { z } _ { \mathrm { g t } }$ and $\pmb { \mu }$ are given, the point-wise term in ${ \mathcal { L } } _ { \mathrm { n l l } }$ is deterministic, and we can set the gradient to zero as

$$
{ \begin{array} { r l } & { { \frac { \partial { \mathcal { L } } _ { \mathrm { n l l } } } { \partial \Sigma _ { \mathrm { u } } } } = { \frac { \partial } { \partial \Sigma _ { \mathrm { u } } } } \left[ \sum _ { i = 1 } ^ { d } { \frac { ( \mathbf { z } _ { \mathrm { g t } , i } - \mu _ { i } ) ^ { 2 } } { 2 \Sigma _ { \mathrm { u } , i } } } + { \frac { 1 } { 2 } } \ln \Sigma _ { \mathrm { u } , i } \right] } \\ & { \qquad = \left[ - { \frac { ( \mathbf { z } _ { \mathrm { g t } , i } - \mu _ { i } ) ^ { 2 } } { 2 \Sigma _ { \mathrm { u } , i } ^ { 2 } } } + { \frac { 1 } { 2 \Sigma _ { \mathrm { u } , i } } } \right] _ { i = 1 } ^ { d } } \\ & { \qquad = 0 . } \end{array} }
$$

Therefore, we have

$$
\pmb { \Sigma } _ { \mathrm { u } } ^ { \star } = \underset { \pmb { \Sigma } _ { \mathrm { u } } } { \arg \operatorname* { m i n } } \mathcal { L } _ { \mathrm { n l l } } = ( \mathbf { z } _ { \mathrm { g t } } - \pmb { \mu } ) \odot ( \mathbf { z } _ { \mathrm { g t } } - \pmb { \mu } ) .
$$

## E.2. Equation (11): Maximizer of the Product of Gaussians

To calculate

$$
\begin{array} { r l } & { \mathbf { z } _ { t - 1 } ^ { \star } = \underset { \mathbf { z } _ { t - 1 } } { \arg \operatorname* { m a x } } \mathcal { N } ( \mathbf { z } _ { t - 1 } ; \sqrt { \bar { \alpha } _ { t - 1 } } \mu , \bar { \alpha } _ { t - 1 } \operatorname { d i a g } ( \Sigma _ { \mathbf { u } } ^ { \prime } ) ) \cdot \mathcal { N } ( \mathbf { z } _ { t - 1 } ; \mu _ { \theta } ( \mathbf { z } _ { t } , \hat { \mathbf { z } } _ { 0 \mid t } ) , \sigma _ { t } ^ { 2 } \mathbf { I } ) } \\ & { \quad \quad \quad = \underset { \mathbf { z } _ { t - 1 } } { \arg \operatorname* { m i n } } \underset { i = 1 } { \overset { d } { \sum } } \frac { \big ( \mathbf { z } _ { t - 1 , i } - \sqrt { \bar { \alpha } _ { t - 1 } } \mu _ { i } \big ) ^ { 2 } } { 2 \bar { \alpha } _ { t - 1 } \Sigma _ { \mathbf { u } , i } ^ { \prime } } + \frac { \big ( \mathbf { z } _ { t - 1 , i } - \mu _ { \theta , i } \big ( \mathbf { z } _ { t } , \hat { \mathbf { z } } _ { 0 \mid t } \big ) \big ) ^ { 2 } } { 2 \sigma _ { t } ^ { 2 } } , } \end{array}
$$

we can set the gradient to zero as

$$
\begin{array} { r l } & { \displaystyle \frac { \partial } { \partial \mathbf { z } _ { t - 1 } } \left[ \sum _ { i = 1 } ^ { d } \frac { ( \mathbf { z } _ { t - 1 , i } - \sqrt { \bar { \alpha } _ { t - 1 } } \mu _ { i } ) ^ { 2 } } { 2 \bar { \alpha } _ { t - 1 } \Sigma _ { \mathrm { u } , i } ^ { \prime } } + \frac { ( \mathbf { z } _ { t - 1 , i } - \mu _ { \theta , i } ( \mathbf { z } _ { t } , \hat { \mathbf { z } } _ { 0 \mid t } ) ) ^ { 2 } } { 2 \sigma _ { t } ^ { 2 } } \right] } \\ & { = \mathbf { z } _ { t - 1 } \left( \frac { 1 } { \bar { \alpha } _ { t - 1 } \Sigma _ { \mathrm { u } } ^ { \prime } } + \frac { 1 } { \sigma _ { t } ^ { 2 } \mathbf { 1 } } \right) - \left( \frac { \sqrt { \bar { \alpha } _ { t - 1 } } \mu } { \bar { \alpha } _ { t - 1 } \Sigma _ { \mathrm { u } } ^ { \prime } } + \frac { \mu _ { \theta } ( \mathbf { z } _ { t } , \hat { \mathbf { z } } _ { 0 \mid t } ) } { \sigma _ { t } ^ { 2 } \mathbf { 1 } } \right) } \\ & { = \mathbf { 0 } . } \end{array}
$$

$$
\mathbf { z } _ { t - 1 } ^ { \star } = \frac { \sigma _ { t } ^ { 2 } \cdot \sqrt { \bar { \alpha } _ { t - 1 } } \mu + \bar { \alpha } _ { t - 1 } \Sigma _ { \mathrm { u } } ^ { \prime } \cdot \mu _ { \theta } ( \mathbf { z } _ { t } , \hat { \mathbf { z } } _ { 0 \mid t } ) } { \sigma _ { t } ^ { 2 } \mathbf { 1 } + \bar { \alpha } _ { t - 1 } \Sigma _ { \mathrm { u } } ^ { \prime } } .
$$

Therefore, we have

![](images/d50f225408d590396066e5afcaf5c621667aa4e275bac60af188c27f71947f6a.jpg)  
Fig. 7. Qualitative comparison of 4× upsampling on DRealSR. We set s = 100 for all of our methods.

![](images/5d53596201acedcc5c2ae855d6dc9c4321b59d201f3ea4e80877939cebd20afd.jpg)  
Fig. 8. Qualitative comparison of 4× upsampling on DIV2K-Val. We set s = 100 for all of our methods.