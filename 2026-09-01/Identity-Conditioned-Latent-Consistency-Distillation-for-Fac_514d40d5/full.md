# Identity-Conditioned Latent Consistency Distillation for Face Synthesis

Tiago Kienen Chaves<sup>∗</sup>, Bernardo Biesseck<sup>∗†</sup>, David Menotti<sup>∗</sup>

<sup>∗</sup>Department of Informatics, Federal University of Parana, Curitiba, Brazil´

<sup>†</sup>Federal Institute of Mato Grosso (IFMT), Pontes e Lacerda, MT, Brazil

{tiagokienen,menotti}@ufpr.br bernardo.biesseck@ifmt.edu.br

Abstract—Diffusion models have achieved strong results in high-fidelity image synthesis, but their iterative sampling process makes large-scale generation computationally expensive. This limitation is especially relevant when generating synthetic face datasets for face recognition, where a large number of subjects with many samples in different poses, expressions, ages, etc., are required. In this work, we show that identityconditioned face synthesis can be performed at a substantially lower computational cost by a latent Consistency Model with few iterations, without compromising image quality. For training, we distill knowledge from the foundation Diffusion Model Arc2Face (teacher) by adapting its original text-to-image pipeline to an embedding-to-face setting, replacing textual prompts with Arc-Face identity embeddings. Our distilled model (student) generates identity-conditioned face images with an average inference time of 0.4819 seconds per image, compared with 2.102 seconds for Arc2Face, resulting in a 4.36× speed-up. Quantitative results, based on FID scores, show that the distilled model remains competitive with Arc2Face across all evaluation protocols. On 100k generated images, it achieves near-parity on CelebA (13.921 vs. 12.928) and outperforms the teacher on WebFace42M (9.317 vs. 9.802). Further evaluations on Synth-500 and AgeDB show a moderate performance gap for the former but comparable results for the latter. These results indicate that Arc2Face can be accelerated through task-specific latent consistency distillation while preserving high image quality for large-scale synthetic face generation. Our proposal is publicly available at https://github. com/UFPR-IPASP-PR/FaceRec-IdentityConsistency.

## I. INTRODUCTION

Diffusion Models (DMs) have become a dominant paradigm for image generation due to their ability to synthesize highquality and diverse images [1], [2]. However, this quality is usually obtained through an iterative reverse process that requires multiple neural network evaluations. Even when sampling is performed in a compressed latent space, as in Latent Diffusion Models (LDMs) [2], inference remains significantly more expensive than a single forward pass. This cost becomes a practical limitation when the goal is not to generate a few examples, but to synthesize large datasets containing thousands or millions of images.

Synthetic face generation can support several practical applications: producing large-scale identity-balanced training sets, creating controlled benchmarks for failure analysis, generating multiple variations of the same identity under different visual conditions, enabling interactive tools for dataset design, among others.

![](images/c49f9b893ce4d33c9169841f35f4e06b8c7f17cff9fba76244532669762a807b.jpg)  
Fig. 1. Illustration of the synthetic face generation processes by a Diffusion Model, such as Arc2Face (teacher), and by our distilled Consistency Model (student). While a Diffusion Model takes 25 steps to generate a high-quality image, a Consistency Model can produce equivalent results in 2 or 3 steps.

In the context of face recognition, various synthetic face generation methods based on diffusion process have recently been proposed, such as Arc2Face [3], VIGface [4], DC-Face [5], Vec2Face [6], and IDiff-Face [7]. These methods are capable of producing large-scale synthetic datasets but also incur high computational cost. In such a scenario, reducing generation time is not merely a convenience; it determines whether synthetic data can be produced at a scale compatible with modern face recognition experiments.

To this end, Consistency Models (CMs) [8] were introduced as generative models capable of producing high-quality samples in one or a few steps, either by training from scratch or by knowledge distillation from pretrained diffusion models. Latent Consistency Models (LCMs) [9] extend this idea to the latent space of pretrained Latent Diffusion Models (LDMs) [2], also enabling few-step high-resolution generation. LCM-LoRA further reduces the training and storage cost by learning a Low-Rank Adaptation (LoRA) module that can be plugged into diffusion models [10], [11].

Although successfully deployed in text-to-image settings to generate stylized content (e.g., paintings, landscapes, or animated characters), these acceleration methods have only recently begun to be explored in face-related tasks, including controllable text-to-face generation and blind face restoration [12], [13]. However, these existing works primarily focus on image restoration or prompt matching and do not directly address the challenge of generating large-scale, identityconsistent face datasets across a variety of poses, expressions, and illumination conditions.

In this work, we bridge this gap by showing that a CM can effectively learn to generate high-quality identityconditioned face images, comparable to those produced by DMs but in a fraction of the steps. Our knowledge distillation experiments, using the foundational diffusion model Arc2Face [3] as the teacher, demonstrate that CMs can be applied to face-generation tasks that require large numbers of samples. Our distilled model substantially reduces inference time while maintaining competitive Frechet Inception Distance´ (FID scores) [14] relative to the diffusion teacher Arc2Face, demonstrating its potential for various applications.

## II. RELATED WORK AND BACKGROUND

In this section, we review related works and important concepts to this proposal.

## A. Diffusion and Latent Diffusion Models

Denoising Diffusion Probabilistic Models (DDPMs) [1] generate images by learning to reverse a gradual noising process. During inference, a sample is produced through an iterative denoising trajectory, typically requiring many evaluations of a neural denoiser. Denoising Diffusion Implicit Models (DDIMs) [15] and other numerical samplers reduce the number of steps, but generation is still fundamentally iterative. Latent Diffusion Models (LDMs) [2] reduce the computational burden by performing denoising in the latent space of an autoencoder rather than directly in pixel space. This strategy enables high-resolution synthesis with lower memory and compute cost, and provides the foundation for many Stable Diffusion models.

## B. Arc2Face

Arc2Face [3] is a diffusion model that produces new highresolution face images of a subject given its face embedding. To ensure identity consistency, it conditions face generation on embeddings extracted by a ResNet100 (R100) [16], pretrained on the Webface42M [17] dataset via the ArcFace [18] loss function. Specifically, the framework is built upon a latent diffusion architecture, replacing generic textual conditioning with an identity-conditioning pathway. While this design makes Arc2Face attractive for generating synthetic datasets for face recognition, its sampling cost scales heavily with the number of diffusion steps.

## C. Consistency Models and Latent Consistency Models

Consistency models learn mappings that send noisy samples at different noise levels to a consistent point on the data manifold [8]. They can be trained directly from data or distilled from pretrained diffusion models. In the distillation setting, the model learns to approximate the solution of the probability flow (PF) ordinary differential equation (ODE) associated with the teacher diffusion model. Latent Consistency Models apply this principle in the latent space of pretrained latent diffusion models, allowing high-quality generation with very few inference steps [9]. The LCM training objective combines a skipping-step strategy with guided distillation, allowing a student model to approximate the teacher’s reverse process over larger jumps in the denoising trajectory.

## D. LCM-LoRA and Adapter-Based Acceleration

LCM-LoRA extends latent consistency distillation by training a LoRA adapter rather than the full model [10]. LoRA reduces the number of trainable parameters by representing weight updates through low-rank matrices [11]. This makes LCM-LoRA attractive when the goal is to accelerate generic Stable-Diffusion-based models with low training and storage overhead. However, adapter-based acceleration is not necessarily optimal for specialized settings in which the conditioning signal and evaluation criteria differ from generic text-to-image generation. In identity-conditioned face synthesis, the central requirement is not only visual realism but also preservation of the identity encoded in the input ArcFace embedding.

## E. Consistency-Based Face Generation and Restoration

Although consistency-based methods have been explored in face-related contexts, most existing works do not address embedding-to-face distillation. ExpertGen [12], for example, studies controllable text-to-face generation with training-free expert guidance, while InterLCM [13] uses latent consistency models for blind face restoration. These works indicate that consistency-based models can be useful in facial domains. However, they differ from our objective: distilling an ArcFaceembedding-conditioned face generator into a fast model for identity-conditioned synthetic data generation.

## III. METHODOLOGY

The goal of our work is to investigate the potential of Consistency Models to generate realistic, identity-consistent face images of comparable quality to those produced by equivalent Diffusion Models, with fewer inference steps. To do this, we distill knowledge from the teacher DM Arc2Face, denoted by $\epsilon _ { \psi } .$ , to train our student Consistency Model architecture [8]. Our approach follows the Latent Consistency Distillation (LCD) [9] framework while replacing text conditioning with Arc2Face identity conditioning. This section describes the identity-conditioned latent representation, the consistency mapping objective, the EMA-based distillation loss, and the complete training procedure summarized in Algorithm 1.

Let I be an input real facial image, and $\textbf { \textit { e } } \in \ \mathbb { R } ^ { 5 1 2 }$ is its normalized face identity embedding extracted by an R100/ArcFace face recognition model, which is then mapped via a CLIP [19] text encoder to match the dimensions expected by the cross-attention layers of the Stable Diffusion U-Net [20] architecture:

$$
c = { \mathrm { P r o j e c t } } ( e ) .\tag{1}
$$

The frozen Arc2Face teacher network $\epsilon _ { \psi }$ operates within a latent space generated by a Variational Autoencoder (VAE) E, with the latent image $z _ { 0 } = \mathcal { E } ( I )$ . The forward diffusion process adds Gaussian noise to the latent vector over a discrete timeline, yielding a sample $z _ { t }$ at time step t. In each iteration, the teacher model yields a noise prediction $\boldsymbol { \epsilon } _ { \psi } ( z _ { t } , c , t )$

To enforce powerful classifier-free guidance during distillation, we define the guided noise prediction sequence using the Imagen-style formulation. Given a guidance parameter $w \sim [ \omega _ { \mathrm { m i n } } , \omega _ { \mathrm { m a x } } ]$ , the guided teacher prediction is defined linearly as:

$$
\tilde { \epsilon } \ast \psi ( z _ { t } , c , \emptyset , w , t ) = \epsilon \ast \psi ( z _ { t } , \emptyset , t ) + w \cdot [ \epsilon _ { \psi } ( z _ { t } , c , t ) - \epsilon _ { \psi } ( z _ { t } , \emptyset , t ) ]\tag{2}
$$

where ∅ represents the unconditional null token or empty text embedding sequence used to simulate unguided generation.

## A. Consistency Mapping and the Consistency Objective

Our student consistency network, parameterized by θ, aims to model a consistent function $f _ { \theta } ~ : ~ ( z _ { t } , w , c , t ) ~  ~ z _ { 0 }$ that maps any point along an ODE trajectory directly to its origin. Following the boundary conditions of consistency models, the model is formulated as:

$$
f _ { \theta } ( z _ { t } , w , c , t ) = c _ { \mathrm { s k i p } } ( t ) z _ { t } + c _ { \mathrm { o u t } } ( t ) F _ { \theta } ( z _ { t } , w , c , t ) ,\tag{3}
$$

where $c _ { \mathrm { s k i p } } ( t )$ and $c _ { \mathrm { o u t } } ( t )$ are analytical boundary functions ensuring $f _ { \theta } ( z _ { 0 } , w , c , 0 ) = z _ { 0 }$ , and $F _ { \theta }$ is a trainable UNet architecture parameterized by online weights θ. Unlike standard LCD, our student architecture explicitly ingests the conditional vector w via a sinusoidal Fourier embedding, mapping the guidance scale into a high-dimensional vector injected directly into the network blocks as a temporal condition.

Given a sub-sampled discrete time sequence path $\tau =$ $[ t _ { 1 } , t _ { 2 } , \dots , t _ { N } ]$ extracted from a total of $T$ diffusion steps $( \mathrm { e . g . }$ $N = 5 0 , T = 1 0 0 0 )$ , we sample adjacent indices $( n , n + k )$ to compute the trajectory matching objective, where $k = 1$ in our implementation.

Although $k = 1$ in the reduced sequence, each transition still corresponds to a larger jump in the original diffusion timeline. For example, with $N = 5 0$ reduced steps extracted from $T = 1 0 0 0$ diffusion steps, each adjacent transition in $\tau$ skips approximately $T / N$ original timesteps. For a given latent sample $z _ { t _ { n + k } }$ , a single-step DDIM solver estimation $\hat { z } * t _ { n } { } ^ { \Psi }$ is evaluated using the teacher’s CFG output:

$$
\hat { z } \ast { t _ { n } } ^ { \Psi } = \mathrm { D D I M } \left( z _ { t _ { n + k } } , \tilde { \epsilon } * \psi ( z * t _ { n + k } , c , \emptyset , w , t _ { n + k } ) , t _ { n + k } , t _ { n } \right)\tag{4}
$$

## B. Distillation Loss with Exponential Moving Average (EMA) Targets

To enforce training stability and protect the identityconditioned manifold from collapsing, we follow Consistency

```latex
Algorithm 1 Identity-Conditioned Latent Consistency Distil
lation (IC-LCD)
Require: Pre-trained Arc2Face Teacher U-Net $\epsilon _ { \psi }$
Require: Pre-trained VAE E
Require: Pre-trained ResNet100+ArcFace
Require: Time schedule $\tau = [ t _ { 1 } , t _ { 2 } , \dots , t _ { N } ]$
Require: Guidance boundaries $[ \omega _ { \mathrm { m i n } } , \omega _ { \mathrm { m a x } } ]$
Require: Huber loss function D ∗ Huber
Require: Initial Student parameters θ
Require: Target Student parameters $\theta ^ { - }  \theta ,$
Require: Learning rate η
Require: EMA decay $\mu$
1: repeat
2: $( e , z _ { 0 } ) \sim \mathcal { D }$
3: $c \gets { \mathrm { P r o j e c t } } ( e )$
4: $\ r \Theta \gets \mathrm { P r o j e c t } ( \mathbf { 0 } )$
5: $n \sim \mathcal { U } ( 1 , N - 1 )$
6: $t _ { n } \gets \tau [ n ]$ and $t * n + k \gets \tau [ n + 1 ]$
7: $w \sim \mathcal { U } ( \omega _ { \mathrm { m i n } } , \omega _ { \mathrm { m a x } } )$
8: $\epsilon \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$
9: $z _ { t _ { n + k } } \gets \alpha _ { t _ { n + k } } z _ { 0 } + \sigma _ { t _ { n + k } } \epsilon$
10: $\epsilon _ { \mathrm { u n c o n d } }  \epsilon _ { \psi } ( z _ { t _ { n + k } } , \emptyset , t _ { n + k } )$
11: $\epsilon _ { \mathrm { c o n d } }  \epsilon _ { \psi } ( z _ { t _ { n + k } } , c , t _ { n + k } )$
12: $\tilde { \epsilon }  \epsilon _ { \mathrm { u n c o n d } } + w \cdot ( \epsilon _ { \mathrm { c o n d } } - \epsilon _ { \mathrm { u n c o n d } } )$
13: $\hat { z } _ { t _ { n } } ^ { \Psi } \gets \mathrm { D D I M S o l v e r } ( z _ { t _ { n + k } } , \tilde { \epsilon } , t _ { n + k } , t _ { n } )$
14: $e _ { w } \gets \mathrm { G u i d a n c e E m b e d d i n g } ( w )$
15: $\hat { x } _ { \theta } \gets f _ { \theta } ( z _ { t _ { n + k } } , e _ { w } , c , t _ { n + k } )$
16: $\hat { x } _ { \theta ^ { - } } \gets f _ { \theta ^ { - } } ( \hat { z } _ { t _ { n } } ^ { \Psi } , e _ { w } , c , t _ { n } )$
17: $\mathcal { L }  \mathcal { D } _ { \mathrm { H u b e r } } ( \hat { x } _ { \theta } ^ {  } , \hat { x } _ { \theta ^ { - } } )$
18: $\theta  \theta - \eta \nabla _ { \theta } \mathcal { L } .$
19: $\theta ^ { - }  \mu \theta ^ { - } + ( 1 - \mu ) \theta ^ { \ \ast \epsilon }$
20: until converged
```

Models and Latent Consistency Distillation [8], [9] and employ an Exponential Moving Average (EMA) target student network parameterized by $\theta ^ { - }$

The online student parameters $\theta$ are updated by minimizing the distance between the online consistency mapping at the nosier step and the target consistency mapping at the resolved DDIM step.

We optimize the online parameters θ using either a Mean Squared Error $( { \mathcal { L } } _ { 2 } ) .$ or a Huber loss function:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { I C - L C D } } ( \theta ; \theta ^ { - } ) = \mathbb { E } _ { z _ { 0 } , c , w , n } \Big [ \mathcal { D } _ { \mathrm { H u b e r } } \big ( f _ { \theta } ( z _ { t _ { n + k } } , w , c , t _ { n + k } ) , } \\ { f _ { \theta ^ { - } } ( \hat { z } _ { t _ { n } } ^ { \Psi } , w , c , t _ { n } ) \big ) \Big ] . \quad } \end{array}\tag{5}
$$

Following each optimization step, the target network pa-. rameters $\theta ^ { - }$ are slowly updated via a momentum parameter $\mu , \mathrm { i . e . , } \theta ^ { - }  \mu \theta ^ { - } + ( 1 - \mu ) \theta .$

Algorithm 1 presents all distillation steps in detail.

## IV. EXPERIMENTS

This section presents our experimental setup, training and testing datasets, obtained results, and qualitative analysis.

## A. Training Data and Cache Precomputation

Due to time constraints, we used 150k face samples from the dataset WebFace4M [17], which contains approximately 4 million face images across 200k identities and is a subset of WebFace260M [17]. All samples in that dataset are originally aligned and cropped to 112×112 pixels using five landmarks: left eye, right eye, nose tip, left mouth, and right mouth. To ensure data dimensions were compatible with the teacher model Arc2Face, face images were resized to 512×512 pixels and their values were normalized from [0, 255] to [−1, 1].

To avoid repeatedly loading images, extracting identity embeddings, and encoding images into latents during training, we precompute and save to disk the required tensors before distillation. The resulting cache is saved in tensor shards containing:

$e \mathrm { : }$ ArcFace identity embeddings with shape [N, 512] computed from original 112 × 112 pixel images;

$z _ { \mathrm { 0 } } { \mathrm { : } }$ VAE latents with shape [N, 4, 64, 64] from $5 1 2 \times 5 1 2$ pixel images;

Our model was trained for 8 epochs with batch=2 using the optimizer AdamW with learning rate of 1e-5 on one GPU NVIDIA GeForce RTX 3090, which took approximately 28 hours.

After training, we compare our distilled consistency model with the original Arc2Face teacher. Both models receive R100/ArcFace identity embeddings as conditioning and generate new synthetic face images of such a subject at 512×512 pixels. The main evaluation focuses on three aspects: computational cost, distributional image quality measured by Frechet´ Inception Distance (FID) [14], and qualitative comparison.

FID is computed between two sets of images and measures the distance between their data distributions. Let $x ^ { r }$ and $x ^ { s }$ denote the 2048-dimensional features extracted from real and synthetic image samples, respectively, with a pretrained Inception-V3 [21] network from its intermediate activations layer pool3. Real means and covariances are computed as $\begin{array} { r c l } { \mu _ { r } } & { = } & { \frac { 1 } { N } \sum _ { i = 1 } ^ { N } f ( x _ { i } ^ { r } ) } \end{array}$ and $\begin{array} { r l } { \Sigma _ { r } } & { { } = } \end{array}$ $\textstyle { \frac { 1 } { N } } \sum _ { i = 1 } ^ { N } { \bigl ( } f ( x _ { i } ^ { r } ) - \mu _ { r } { \bigr ) } ( f ( x _ { i } ^ { r } ) - \mu _ { r } { \overset { . } { ) } } ^ { T }$ , respectively. Both $\mu _ { s }$ and $\Sigma _ { s }$ are computed analogously from the synthetic images. The FID is then defined as

$$
\begin{array} { r l r } {  { F I D ( \mu _ { r } , \Sigma _ { r } , \mu _ { s } , \Sigma _ { s } ) = } } \\ & { } & { \sqrt { \| \mu _ { r } - \mu _ { s } \| _ { 2 } ^ { 2 } + \operatorname { T r } ( \Sigma _ { r } + \Sigma _ { s } - 2 ( \Sigma _ { r } \Sigma _ { s } ) ^ { 1 / 2 } ) } . } \end{array}\tag{6}
$$

A lower FID indicates that the distribution of generated images is closer to the reference distribution in the metric’s feature space.

We evaluate two settings using FID. First, we generate 100k images from a random subset of identities from Web-Face4M [17], which we compare to CelebA [22] and another disjoint subset of 100k images from WebFace42M. This setting evaluates large-scale distributional quality both against a general face dataset, CelebA, and against a reference domain closer to the training and face-recognition data, WebFace42M. Second, we repeat the Arc2Face evaluation protocol, in which five synthetic images are generated for each synthetic identity in the datasets Synth-500 [3] and AgeDB [23]. This second setting is included to make our results directly comparable with the published Arc2Face protocol and to evaluate the model on both synthetic identities and real face images outside the WebFace training subset. Then, FID is computed between the corresponding samples.

![](images/bdd2cc092aa2ab1d36ffc7976ba8e0bb8b4c704dbec1ccff0c41ba75775cfeb9.jpg)

![](images/4c36e485b23dc5f8cc63bd4280f4272c8ba00c1ba7c892f6f44e87faf77a156f.jpg)  
Fig. 2. Real and Synthetic face images. The first two rows show samples from the real datasets Webface4M and CelebA, used in this work as quality reference. The last three rows show synthetic images generated by the teacher Diffusion Model Arc2Face, LCM-LoRA, and our distilled Consistency Model (CM) from real faces of the dataset Webface4M. Each image belongs to a distinct subject and shows the ability of all methods to generate faces of various ethnic groups.

TABLE I  
FID SCORES ACROSS ALL EVALUATED SETTINGS. LOWER IS BETTER.
<table><tr><td>Dataset</td><td>Arc2Face</td><td>LCM-LoRA</td><td>CM (ours)</td><td>Diff. to Arc2Face</td></tr><tr><td>CelebA</td><td>12.928</td><td>11.945</td><td>13.921</td><td>+0.993</td></tr><tr><td>WebFace42M</td><td>9.802</td><td>13.832</td><td>9.317</td><td>-0.485</td></tr><tr><td>Synth-500</td><td>5.673</td><td>11.4902</td><td>8.7039</td><td>+3.0309</td></tr><tr><td>AgeDB</td><td>6.628</td><td>13.0664</td><td>6.9720</td><td>+0.3440</td></tr></table>

## B. Quantitative Results

Table I presents the FID scores for all evaluated settings. Compared to CelebA [22], the consistency model achieves a slightly higher FID than the teacher Arc2Face, increasing from 12.928 to 13.921, which corresponds to an absolute difference of 0.993, or approximately 7.68%. Compared to WebFace42M [17], the consistency model achieves a lower FID than the teacher, reducing it from 9.802 to 9.317, an improvement of approximately 4.95%. Since Web-Face4M/WebFace42M are closer to the training domain, this suggests that the distilled model preserves the distributional characteristics of the teacher particularly well in the face recognition data domain.

Figure 2 qualitatively illustrates samples from the real reference datasets and from the generated images used in the 100k-image evaluation.

These FID results indicate that the few-step distilled model does not suffer a large distributional degradation relative to the diffusion teacher. The CelebA result shows a small loss relative to Arc2Face, while the WebFace42M result shows that the consistency model can even outperform the teacher on FID in a reference domain closer to the training data.

For the Arc2Face protocol, the consistency model obtains a higher FID than Arc2Face on Synth-500, indicating a larger gap in this setting. On AgeDB, the difference is smaller, with the proposed model obtaining 6.9720 compared with 6.628 for Arc2Face. Figure 3 shows qualitative samples generated under this protocol.

The Synth-500 result shows that the student still has limitations and may not fully reproduce the teacher’s distribution in all evaluation settings. One possible explanation is that Synth-500 is itself composed of synthetic identities, which may differ from the real-face distribution used during cache construction and distillation. This domain mismatch can make the distilled model less accurate in reproducing the teacher behavior under this protocol. However, the AgeDB result is close to the teacher’s, suggesting that the consistency model retains much of the teacher’s generation quality on real face images while substantially reducing computational cost.

## C. Computational Cost Comparison

Table II reports the average generation time per image. The distilled model reduces the average time from 2.102 seconds to 0.4819 seconds, corresponding to a 4.36× speed-up and a 77.1% reduction in per-image generation time. Because absolute runtime depends on hardware and implementation details, the most relevant observation is the relative reduction obtained under the same evaluation setup.

This result is important for practical face synthesis. For example, generating 500k images would require approximately 292 GPU-hours at 2.102 seconds per image, but approximately 67 GPU-hours at 0.4819 seconds per image, assuming the same single-image throughput. Thus, the distilled model makes large-scale synthetic dataset generation substantially more feasible.

## D. Qualitative Results

Qualitatively, the distilled model successfully generates recognizable face images from ArcFace embeddings. The generated samples preserve the central behavior expected from Arc2Face: the input is not a text prompt but a face recognition vector, and the output is a plausible face consistent with that vector. Visual inspection in Figures 2 and 3 shows that the student can synthesize diverse faces and does not collapse to a single appearance pattern.

TABLE II  
AVERAGE GENERATION TIME PER IMAGE.
<table><tr><td>Model</td><td>Time/image (s)</td><td>Relative speed</td></tr><tr><td>Arc2Face diffusion teacher</td><td>2.1020</td><td>1.00×</td></tr><tr><td>LCM-LoRA</td><td>0.1780</td><td>11.80×</td></tr><tr><td>Proposed consistency model</td><td>0.4819</td><td>4.36×</td></tr></table>

![](images/86fcb5b54d9467711e14db9ed6acf2b5548482f3b76fdc15df0ddc68dd70d9ac.jpg)

![](images/af3c258ba1d8ae5ced852a85f51897f8da196a31dd32543ac28b838b16648d64.jpg)  
Fig. 3. New generated face images from the datasets Synth-500 [3] and AgeDB [23], using three different models: Diffusion Model Arc2Face (teacher), LCM-LoRA, and our Consistency Model (CM).

## E. Analysis

The results support the hypothesis that identity-conditioned face synthesis can be accelerated through latent consistency distillation. The most direct gain is computational: the proposed model reduces generation time by more than four times. This has immediate practical consequences for synthetic data generation, where the total cost scales linearly with the number of generated images.

The FID results also suggest that the distilled model remains useful for large-scale generation. Although it is not uniformly better than Arc2Face, it remains close to the teacher in most settings and even obtains a lower FID against WebFace42M. This is relevant because WebFace-style data is directly connected to face recognition research. Therefore, for applications where a small quality trade-off is acceptable in exchange for a large reduction in cost, the proposed model is a practical alternative.

A full pretrained consistency model also has advantages over a generic LCM-LoRA adapter in this setting. LCM-

LoRA is attractive because it is lightweight and plug-andplay, but it is designed primarily as a universal acceleration module for Stable-Diffusion-like models. Our setting is more specialized: the conditioning signal is an ArcFace embedding, and the key requirement is identity preservation rather than only prompt-image alignment or visual appeal. By training the student directly on Arc2Face identity-conditioned trajectories, the model can learn the task-specific relationship between identity embeddings and generated faces.

At the same time, this work does not eliminate the usefulness of adapters. LCM-LoRA remains a strong baseline for future comparison because it requires fewer trainable parameters and is easier to train. The contribution of this paper is complementary: it investigates whether a full Arc2Facedistilled consistency model can serve as a specialized generator for identity-conditioned face synthesis.

## V. CONCLUSION

This paper presented a latent-consistency distillation approach to accelerate Arc2Face, an identity-conditioned face generator based on ArcFace embeddings. We adapted a text-to-image latent consistency distillation pipeline to the embedding-to-face setting, precomputed ArcFace identity embeddings and VAE latents for 150k WebFace4M identity samples, and trained a student model initialized from the Arc2Face teacher. The resulting model generates face images in 0.4819 seconds on average, compared with 2.1020 seconds for the diffusion teacher, achieving a 4.36× speed-up.

Quantitative results showed that the distilled model is competitive with Arc2Face. On 100k generated images, the proposed model achieves FID scores of 13.9210 on CelebA and 9.3170 on WebFace42M, compared with 12.9280 and 9.8020 for Arc2Face, respectively. In the Arc2Face protocol, the proposed model achieves 8.7039 on Synth-500 and 6.9720 on AgeDB, showing a larger gap on Synth-500 (+3.0309) but near-parity on AgeDB (+0.3440).

The main limitation of the current version is that evaluation is still centered on FID and runtime. Future work should include identity-specific metrics such as cosine similarity between conditioning embeddings and generated images, face detection rate, intra-identity diversity, inter-identity separability, and downstream face recognition performance after training with the generated data. Future ablations should evaluate the effect of the number of inference steps, guidance scale, training length, cache size, and the comparison between full-model distillation and LCM-LoRA adapters. Because we employed an embedding-to-face schema, our model is limited to ArcFace embeddings and will not work properly directly with text embeddings without retraining. Finally, because face synthesis can be misused, future versions should explicitly discuss safeguards, dataset governance, and responsible use of synthetic identity-conditioned face generation.

## ACKNOWLEDGMENT

This study was financed in part by the Coordenac¸ao˜ de Aperfeic¸oamento de Pessoal de N´ıvel Superior -

Brasil (CAPES), through the Programa de Excelenciaˆ Academica (PROEX) ˆ - Finance Code 001, and in part by the Conselho Nacional de Desenvolvimento Cient´ıfico e Tecnologico (CNPq)´ and Fundac¸ao Arauc˜ aria´ under grant #078/2026. The authors also thank the Federal Institute of Mato Grosso (IFMT), Pontes e Lacerda Campus, for supporting Bernardo Biesseck.

## REFERENCES

[1] J. Ho, A. Jain, and P. Abbeel, “Denoising diffusion probabilistic models,” in Advances in Neural Information Processing Systems (NeurIPS), 2020.

[2] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer, “Highresolution image synthesis with latent diffusion models,” in CVPR, 2022.

[3] F. P. Papantoniou, A. Lattas, S. Moschoglou, J. Deng, B. Kainz, and S. Zafeiriou, “Arc2face: A foundation model for id-consistent human faces,” in European Conf. on Computer Vision (ECCV), 2024.

[4] M. Kim, M.-C. Sagong, G. P. Nam, J. Cho, and I.-J. Kim, “Vigface: Virtual identity generation for privacy-free face recognition,” in ICCV, 2025.

[5] M. Kim, F. Liu, A. Jain, and X. Liu, “Dcface: Synthetic face generation with dual condition diffusion model,” in IEEE/CVF Conf. on Computer Vision and Pattern Recognition (CVPR), 2023.

[6] H. Wu, J. Singh, S. Tian, L. Zheng, and K. W. Bowyer, “Vec2face: Scaling face dataset generation with loosely constrained vectors,” in Int. Conf. on Learning Representations (ICLR), 2025.

[7] F. Boutros, J. H. Grebe, A. Kuijper, and N. Damer, “Idiff-face: Syntheticbased face recognition through fizzy identity-conditioned diffusion models,” in IEEE/CVF Int. Conf. on Computer Vision (ICCV), October 2023.

[8] Y. Song, P. Dhariwal, M. Chen, and I. Sutskever, “Consistency models,” in Int. Conf. on Machine Learning (ICML), 2023.

[9] S. Luo, Y. Tan, L. Huang, J. Li, and H. Zhao, “Latent consistency models: Synthesizing high-resolution images with few-step inference,” arXiv preprint arXiv:2310.04378, 2023.

[10] S. Luo, Y. Tan, S. Patil, D. Gu, P. von Platen, A. Passos, L. Huang, J. Li, and H. Zhao, “Lcm-lora: A universal stable-diffusion acceleration module,” arXiv preprint arXiv:2311.05556, 2023.

[11] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen, “Lora: Low-rank adaptation of large language models,” in Int. Conf. on Learning Representations (ICLR), 2022.

[12] L. Shi and Y. Fu, “Expertgen: Training-free expert guidance for controllable text-to-face generation,” arXiv preprint arXiv:2505.17256, 2025.

[13] S. Li, K. Wang, J. van de Weijer, F. S. Khan, C.-L. Guo, S. Yang, Y. Wang, J. Yang, and M.-M. Cheng, “InterLCM: Low-quality images as intermediate states of latent consistency models for effective blind face restoration,” in Int. Conf. on Learning Representations (ICLR), 2025.

[14] M. Heusel, H. Ramsauer, T. Unterthiner, B. Nessler, and S. Hochreiter, “GANs trained by a two time-scale update rule converge to a local nash equilibrium,” in Ad. in Neural Information Processing Systems, 2017.

[15] J. Song, C. Meng, and S. Ermon, “Denoising diffusion implicit models,” in Int. Conf. on Learning Representations (ICML), 2021.

[16] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in IEEE/CVF Conf. on Computer Vision and Pattern Recognition (CVPR), 2016.

[17] Z. Zhu, G. Huang, J. Deng, Y. Ye, J. Huang, X. Chen, J. Zhu, T. Yang, J. Lu, D. Du, and J. Zhou, “Webface260m: A benchmark unveiling the power of million-scale deep face recognition,” in CVPR, 2021.

[18] J. Deng, J. Guo, N. Xue, and S. Zafeiriou, “Arcface: Additive angular margin loss for deep face recognition,” in CVPR, 2019.

[19] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, G. Krueger, and I. Sutskever, “Learning transferable visual models from natural language supervision,” in Int. Conf. on Machine Learning (ICML), 2021.

[20] O. Ronneberger, P. Fischer, and T. Brox, “U-net: Convolutional networks for biomedical image segmentation,” in MICCAI, 2015.

[21] C. Szegedy, V. Vanhoucke, S. Ioffe, J. Shlens, and Z. Wojna, “Rethinking the inception architecture for computer vision,” in CVPR, 2016.

[22] Z. Liu, P. Luo, X. Wang, and X. Tang, “Deep learning face attributes in the wild,” in IEEE/CVF Int. Conf. on Computer Vision (ICCV), 2015.

[23] S. Moschoglou, A. Papaioannou, C. Sagonas, J. Deng, I. Kotsia, and S. Zafeiriou, “Agedb: The first manually collected, in-the-wild age database,” in CVPRW, 2017.