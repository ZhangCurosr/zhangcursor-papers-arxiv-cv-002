# Scale-Separated Conditioning for Style-Encoder-Free Difusion Stylization

Jingtao Zhang<sup>1⋆</sup>, Haorui Gao<sup>2⋆⋆</sup>, Youqing Liang<sup>2</sup>, and Zeming Liu<sup>3</sup>

<sup>1</sup> College of Computing, Georgia Institute of Technology, Atlanta, GA, USA jzhang957@gatech.edu

<sup>2</sup> Courant Institute of Mathematical Sciences, New York University, New York, NY, USA

<sup>3</sup> Department of Computer Science, Brown University, Providence, RI, USA

Abstract. Reference-based difusion stylization requires separating target geometry from transferable appearance. Existing tuning-based methods often rely on aligned content-style-target triplets or auxiliary visual encoders, which increases data cost and can transfer unintended scene structure from the style reference. We propose SEFS (Style-Encoder-Free Stylization), a style-encoder-free conditioning framework for difusion transformers. SEFS forms style tokens from stochastic low-resolution crops of single training images. This crop bottleneck preserves local appearance statistics such as palette, stroke, texture, and material, while reducing access to global layout cues. Target content is encoded by edge and segmentation cues and fused with the noisy latent through parametereficient trainable projections. We add style-to-denoising re-normalization for token-statistic alignment and cross-block skip fusion for spatial detail. SEFS trains on unpaired single images; the frozen difusion VAE is used only to place image conditions in the latent space. On artistic stylization benchmarks, SEFS improves content consistency and leakage diagnostics while retaining reference-style afinity, and ablations support the cropresolution, re-normalization, and skip-fusion choices. The code of SEFS will be made publicly available.

Keywords: Reference-based stylization · Difusion models · Style transfer

## 1 Introduction

Difusion-based text-to-image models have become a common backbone for visual generation, with rapid progress in latent difusion and difusion transformers [21,19,8,5,4]. Their pretrained priors support many conditioning tasks, including text prompts, spatial controls, identity references, inpainting or outpainting, and image edits [30,29,22,1,31,2,14,23]. Reference-based stylization is one such conditioning problem. Given a target content image and a style reference, the model must preserve the target geometry while transferring visual appearance such as palette, material, atmosphere, stroke pattern, illumination, and local texture. These attributes are not explicitly labeled in the reference: style is entangled with the reference image’s own objects, layout, and scene semantics. This makes stylized data collection, style representation, and evaluation dificult.

This entanglement leads to a common failure mode. If the reference representation is too rich, the model may copy non-transferable structure from the style image, causing content leakage. If it is too weak, the generated image preserves geometry but misses the reference appearance. Existing approaches occupy diferent points on this trade-of. Training-free methods manipulate inversion trajectories or attention features from the reference [26,6,10,20,13], which avoids additional training but can be slow when inversion is required, sensitive to inversion quality, and dependent on layer-wise intervention. Adapter-based methods introduce pretrained image encoders or prompt adapters [29,9], but the extracted embeddings are not guaranteed to isolate style from semantics and introduce additional parameters. Tuning-based methods can improve stylization, yet often depend on curated content-style-target triplets or large-scale paired construction [28], making the data pipeline expensive and limiting scalability.

SEFS uses a conditioning bottleneck rather than a stronger style encoder. A style reference contains transferable appearance at local scales, whereas its global spatial arrangement is usually non-transferable. We therefore construct style tokens from stochastic low-resolution crops of individual images. Cropping exposes local appearance statistics; resizing suppresses precise contours and global layout; and training across many single images discourages an image-identity shortcut. At inference time, the same style-token pathway is driven by an external style reference, while content is supplied by structural controls from the target image. This creates a same-image-to-cross-image generalization setting: training learns the scale-separated roles from unpaired single images, whereas evaluation tests fixed target-content and external-style-reference pairs.

We propose SEFS (Style-Encoder-Free Stylization), a style-encoder-free conditioning framework for difusion transformers. SEFS separates the conditioning pathway by scale and role: edge maps and segmentation maps provide targetcontent cues, while low-resolution style crops provide appearance tokens. These conditions are fused into the difusion transformer with parameter-eficient trainable projections and LoRA adaptation. Style-to-denoising re-normalization aligns the statistics of style tokens with the denoising token stream, and cross-block skip fusion reuses shallow structural features in later transformer blocks. The framework uses unpaired single images for training, avoids auxiliary style encoders, and remains parameter-eficient.

Our contributions are summarized as follows:

i) We introduce SEFS, a style-encoder-free difusion stylization framework that learns from unpaired single images rather than aligned content-style-target triplets. The method uses a scale-separated crop bottleneck to preserve local appearance statistics while suppressing global reference structure.

ii) We design a parameter-eficient trainable conditioning architecture that combines target edges, segmentation maps, and low-resolution style tokens inside a difusion transformer. Cross-block skip fusion improves content preservation by carrying shallow structural information to later denoising stages.

![](images/4738f29afe99221954b56fa077c50d5c6291bafce87b1798e786c24e73ac426e.jpg)  
Fig. 1. Representative comparisons with CSGO [28], StyleShot [9], and SEFS. Existing methods can drift from the target content or transfer reference-specific structure; scale separated conditioning helps SEFS preserve target layout while matching reference appearance.

iii) We introduce style-to-denoising re-normalization to stabilize heterogeneous token fusion. This module improves the crop-token pathway in our ablations and supports training using about 40k unpaired WikiArt images, without requiring curated triplets or a separate style-encoder training set.

iv) We evaluate SEFS against recent difusion stylization methods and conduct focused ablations on content conditions, crop resolution, style-encoder usage, re-normalization, and skip fusion. The results improve content scores and leakage diagnostics while retaining reference-style afinity.

## 2 Related Work

## 2.1 Difusion Models and Conditional Generation

Latent difusion models [21,19] and difusion transformers [18,8,5,4,32,24] provide strong priors for controllable image synthesis. Typical conditioning methods keep the generator largely fixed and add spatial adapters [30], image-prompt adapters [29], or low-rank tuning [11]. This makes adaptation eficient but creates a representation trade-of: conditions must guide generation without overriding target content or leaking unwanted semantics. In stylization, spatial maps preserve geometry but not appearance, whereas image embeddings can entangle style with reference identity, objects, and layout. SEFS keeps the parameter-eficient paradigm but replaces the auxiliary style encoder with a scale-separated image pathway.

## 2.2 Reference-Based Stylization

Arbitrary reference-based stylization remains dificult because style may include color, material, atmosphere, stroke statistics, and sometimes global structure. Classical neural style transfer separates content and style through feature statistics, as in AdaIN-style normalization [12], while transformer feed-forward methods improve perceptual quality and content preservation [7]. Difusion methods exploit stronger generative priors, but still need a style pathway that transfers appearance without copying reference layout.

Recent difusion stylization methods are commonly training-free, adapterbased, or tuning-based. Training-free approaches manipulate inversion trajectories, attention features, reference modulation, or sampling startpoints [26,6,10,20,13]. They avoid task-specific training but can be sensitive to inversion quality, layer choices, and attention reuse. Adapter-based methods use image-conditioned pathways or pretrained visual encoders [29,9], which capture rich appearance but may retain object category, layout, or reference identity. Tuning-based methods such as CSGO [28] improve stylization end-to-end, often with curated triplets or large construction pipelines.

SEFS is complementary to these directions. Instead of extracting a fullresolution style embedding with a pretrained encoder, it forms style tokens from stochastic low-resolution crops. This bottleneck preserves local transferable appearance while suppressing global reference geometry; combined with target structural cues and token re-normalization, it yields a difusion-transformer stylization framework trained without triplet supervision or an auxiliary visual style encoder.

## 3 The proposed SEFS

## 3.1 Preliminaries

Flow Matching. Let $p _ { 0 }$ and $p _ { 1 }$ denote the source and data distributions on $\mathbb { R } ^ { n }$ . An ODE-based generative model transports a sample $\mathbf { x } _ { 0 } \sim p _ { 0 }$ to $\mathbf { x } _ { 1 } \sim p _ { 1 }$ by integrating a time-dependent vector field:

$$
\frac { d { \mathbf x } _ { t } } { d t } = f _ { \theta } ( { \mathbf x } _ { t } , t ) , \quad t \in [ 0 , 1 ] .\tag{1}
$$

The learned vector field $f _ { \theta } : \mathbb { R } ^ { n } \times [ 0 , 1 ] \to \mathbb { R } ^ { n }$ defines the trajectory followed during generation.

Rectified Flow. Rectified Flow [16] trains the vector field on straight paths between a data sample and Gaussian noise. We use an equivalent reverse-time parameterization of the SD3 latent-space path [8], where $t = 0$ corresponds to noise and $t = 1$ corresponds to the data latent:

$$
\begin{array} { r } { \mathbf { z } _ { t } = ( 1 - t ) \boldsymbol { \epsilon } + t \mathbf { z } _ { 0 } , \quad \boldsymbol { \epsilon } \sim \mathcal { N } ( 0 , I ) , } \end{array}\tag{2}
$$

and minimize

$$
\mathcal { L } _ { \mathrm { R F } } = \mathbb { E } _ { t , \epsilon } \left[ \lambda _ { t } \lVert f _ { \theta } ( \mathbf { z } _ { t } , t ) - ( \mathbf { z } _ { 0 } - \epsilon ) \rVert _ { 2 } ^ { 2 } \right] ,\tag{3}
$$

where $\lambda _ { t }$ is the timestep-dependent loss weight.

![](images/e995990158ef8a9c7827d3ef58ff8c0628e29d4133c60607887079404c67f61b.jpg)  
Fig. 2. Framework of the proposed SEFS. The target image provides structural content conditions through Canny edges and SAM [15] segmentation, while the style pathway receives stochastic low-resolution crops that expose local appearance statistics and suppress global layout. Style tokens are concatenated with the denoising tokens and re-normalized using the denoising-token statistics $( \mu _ { \mathbf { d } } , v _ { \mathbf { d } } )$ and style-token statistics $( \mu _ { \mathbf { s } } , v _ { \mathbf { s } } )$ . The right part illustrates LoRA fine-tuning and cross-block skip fusion.

![](images/64ae92e8b630a3b373452e4ec28727f92b0c283e0651013bf8ff129310c769f8.jpg)  
Fig. 3. Training inputs of SEFS. Canny edges preserve fine structural boundaries, SAM masks preserve coarse region layout, and low-resolution style crops retain local appearance while reducing access to global reference geometry.

## 3.2 The proposed SEFS

We first describe how SEFS constructs content and style conditions from single images, then present the token projection, re-normalization, cross-block skip fusion, and training objective.

Condition Generation. Given a training image x, SEFS constructs a pseudo content-style training instance without requiring an external stylized target. For content, we use two complementary structural views. Edge maps such as Canny [3] and HED [27] are commonly used to provide high-frequency boundaries and local contours; we instantiate this view with Canny edges. SAM segmentation masks [15] provide region-level organization that is not captured by sparse edges alone, such as consistent regions over buildings, sky, or mountains. For style, we apply stochastic cropping to x and resize the crop to a low resolution. This operation preserves local appearance statistics such as color palette, texture, material, and stroke pattern, but removes much of the precise contour and global layout information that would otherwise encourage content leakage. During inference, the cropped training view is replaced by the user-provided style reference, processed through the same low-resolution style pathway.

Implementation details. During training, RandomCropResize samples a crop with ratio (0.1 ∼ 0.5) from the training image and resizes it to $6 4 \times 6 4$ to form the style view. During inference, we use a deterministic single-view protocol: the full style reference is resized to the same 64 × 64 style-token resolution rather than sampled repeatedly. Canny maps and SAM outputs are rendered as threechannel images before VAE encoding, matching the input interface of the copied SD3 patch-embedding branch. Text-conditioning settings are kept fixed across training and evaluation so that diferences primarily reflect the image-conditioning pathways. Although training constructs pseudo instances from a single image, the content and style signals are deliberately separated by role and scale: structural maps provide geometry, while the low-resolution style view provides appearance statistics with limited access to global layout. The cross-image evaluations in Fig. 4 and Table 2 therefore test whether this bottleneck transfers to arbitrary content-style pairs rather than only reconstructing the training image.

Condition projection. Let m $\in \mathbb { R } ^ { H \times W \times 3 }$ and $\mathbf { c } \in \mathbb { R } ^ { I \times W \times 3 }$ denote the SAM segmentation map and Canny edge map, respectively. Let $\mathbf { s } \in \mathbb { R } ^ { H _ { 1 } \times W _ { 1 } \times 3 }$ denote the low-resolution style view, where $H _ { 1 } ~ < ~ H$ and $W _ { 1 } < W$ . We use pretrained SD3 [8] as the base model and form the noisy latent by

$$
\mathbf { z } _ { t } = ( 1 - t ) \cdot \epsilon + t \cdot \mathbf { z } _ { 0 }\tag{4}
$$

where ${ \bf z } _ { 0 } = E n c o d e ( { \bf x } )$ and $E n c o d e ( \cdot )$ is the pretrained VAE encoder. We encode the content and style conditions in the same latent space: $\mathbf { m _ { z } } = E n c o d e ( \mathbf { m } )$ ${ \bf c _ { z } } = E n c o d e ( { \bf c } )$ , and $\mathbf { s _ { z } } = E n c o d e ( \mathbf { s } )$ . Thus, “style-encoder-free” means that SEFS does not introduce an auxiliary CLIP/IP-Adapter-style visual encoder for style extraction; all image conditions are encoded only through the frozen difusion VAE before entering the parameter-eficient trainable conditioning branch. After patchification with patch size $p$ and projection by the copied SD3 patch-embedding layer, the denoising latent, segmentation latent, and edge latent are represented as sequences of length L and hidden dimension $D ,$ while the style latent has length $L _ { 1 }$ . We concatenate content-related tokens along the channel dimension:

$$
\mathbf { z } _ { c a t } = C o n c a t ( \mathbf { z } _ { t } , \mathbf { m } _ { \mathbf { z } } , \mathbf { c } _ { \mathbf { z } } )\tag{5}
$$

where $\mathbf { z } _ { c a t } \in \mathbb { R } ^ { L \times 3 D }$ . A trainable projection $\mathbf { W } _ { c a t } \in \mathbb { R } ^ { 3 D \times D }$ maps the concatenated sequence back to the original hidden dimension. We initialize this projection

to preserve the pretrained denoising stream at the beginning of fine-tuning:

$$
\mathbf { W } _ { c a t } = \left( \begin{array} { l l l l } { 1 } & { 0 } & { \cdots } & { 0 } \\ { 0 } & { 1 } & { \cdots } & { 0 } \\ { \cdots } & { \cdots } & { \cdots } & { \cdots } \\ { 0 } & { 0 } & { \cdots } & { 1 } \\ { 0 } & { 0 } & { 0 } & { 0 } \\ { \cdots } & { \cdots } & { \cdots } & { \cdots } \\ { 0 } & { 0 } & { 0 } & { 0 } \end{array} \right) , \mathbf { z } = \mathbf { z } _ { c a t } \mathbf { W } _ { c a t }\tag{6}
$$

where $\mathbf { W } _ { c a t }$ is initialized by concatenating one $D \times D$ identity matrix and two $D \times D$ zero matrices. The style tokens are then concatenated with the projected denoising sequence along the token dimension, yielding $\mathbf { z } \in \mathbb { R } ^ { ( L + L _ { 1 } ) \times \hat { D } }$

Re-normalization. The denoising tokens and style tokens originate from diferent visual views and therefore have diferent activation statistics. Direct token concatenation can make the style stream either dominate generation or be ignored during early fine-tuning. To stabilize the fusion, after the first transformer block we divide the sequence into the denoising part $\mathbf { d } _ { z } = \mathbf { z } [ 0 : L ]$ and the style part $\mathbf s _ { z } = \mathbf z [ L : L + L _ { 1 } ]$ . We then align the style-token statistics to the denoisingtoken statistics:

$$
\tilde { \mathbf { s } } _ { \mathbf { z } } = \frac { \sqrt { v _ { \mathbf { d } } + \eta } } { \sqrt { v _ { \mathbf { s } } + \eta } } ( \mathbf { s } _ { \mathbf { z } } - \mu _ { \mathbf { s } } ) + \mu _ { \mathbf { d } }\tag{7}
$$

where $\mu _ { \mathbf { s } }$ and $v _ { \mathbf { s } }$ are the channel-wise mean and variance of the style tokens, µ<sub>d</sub> and $v _ { \mathbf { d } }$ are the corresponding statistics of the denoising tokens, and η is a small constant for numerical stability. The square roots convert variances into standard-deviation scales for feature-statistic matching. Similar to adaptive modulation in difusion transformers [18], we predict a residual scale and shift from the re-normalized style tokens:

$$
\begin{array} { r } { \gamma _ { \mathbf { s } } = \tilde { \mathbf { s } } _ { \mathbf { z } } \mathbf { W } _ { s c a l e } , \quad \beta _ { \mathbf { s } } = \tilde { \mathbf { s } } _ { \mathbf { z } } \mathbf { W } _ { s h i f t } , \quad \mathbf { s } _ { \mathbf { z } } = \tilde { \mathbf { s } } _ { \mathbf { z } } \odot \left( 1 + \gamma _ { \mathbf { s } } \right) + \beta _ { \mathbf { s } } } \end{array}\tag{8}
$$

where $\mathbf { W } _ { s c a l e } \in \mathbb { R } ^ { D \times D }$ and $\mathbf { W } _ { s h i f t } \in \mathbb { R } ^ { D \times D }$ are trainable parameters.

Skip-Connection across blocks. Transformer blocks at shallow depths preserve fine structural details, while deeper blocks tend to carry more semantic and global information. To improve content preservation, we introduce crossblock skip fusion in the latter half of the transformer. Let $\mathbf { z } ^ { ( i ) }$ denote the output sequence of block i in a model with N blocks. For later blocks, we concatenate the current sequence with its paired shallow feature along the channel dimension and project it back to D dimensions:

$$
\mathbf { W } _ { s k i p } ^ { ( i ) } = \left( \begin{array} { l l l l } { 1 } & { 0 } & { \cdots } & { 0 } \\ { 0 } & { 1 } & { \cdots } & { 0 } \\ { \cdots } & { \cdots } & { \cdots } & { \cdots } \\ { 0 } & { 0 } & { \cdots } & { 1 } \\ { 0 } & { 0 } & { 0 } & { 0 } \\ { \cdots } & { \cdots } & { \cdots } & { \cdots } \\ { 0 } & { 0 } & { 0 } & { 0 } \end{array} \right) , \mathbf { z } ^ { ( i ) } = \mathbf { z } ^ { ( i ) } \mathbf { W } _ { s k i p } ^ { ( i ) }\tag{9}
$$

The skip projection is initialized to pass through the current block output and gradually learn how much shallow structural information to reuse. The resulting sequence $\mathbf { z } ^ { ( i ) } \in \mathbb { R } ^ { ( L + L _ { 1 } ) \times D }$ is fed into the next block.

Objective function. The training objective follows the SD3 rectified-flow loss with additional conditions:

$$
\mathcal { L } _ { S E F S } = \lambda _ { t } \cdot | | f ( \mathbf { z } _ { t } , \mathbf { m } _ { \mathbf { z } } , \mathbf { c } _ { \mathbf { z } } , \mathbf { s } _ { \mathbf { z } } , t ; \theta ) - ( \mathbf { z } _ { 0 } - \epsilon ) | | _ { 2 } ^ { 2 }\tag{10}
$$

where $\lambda _ { t }$ balances diferent timesteps.

Algorithm summary. We summarize the training pipeline in Algorithm 1 and the block-wise forward pass in Algorithm 2. Algorithm 1 shows how each unpaired image is converted into structural content conditions and a low-resolution style view, encoded into latent tokens, and optimized with the rectified-flow objective. Algorithm 2 details the corresponding forward computation, including content-token projection, style-token re-normalization, and cross-block skip fusion. Here, $C o n C a t _ { D i m }$ and $C o n C a t _ { S e q }$ denote concatenation over the channel dimension and token dimension, respectively; at inference, the random crop is replaced by the user-provided style reference while the rest of the conditioning pipeline is unchanged.

Algorithm 1 Training pipeline of the proposed SEFS. RandomCropResize   
denotes stochastic cropping followed by low-resolution resizing.   
1: Input: Pretrained SD3 model $f _ { \theta }$ . Raw image x, text embeddings T. Pre   
trained SAM encoder $S A M ( \cdot )$ and VAE encoder Encode(·). Canny detector   
Canny(·), and Optimizer $O P T ( \cdot )$   
2: Init: $\Theta _ { \mathrm { t r a i n } }$ including LoRA, projection, re-normalization, and skip-fusion   
parameters   
3: Data: s ← RandomCropResize(x)   
4: Data: m $ S A M ( \mathbf { x } ) , \mathbf { c }  C a n n y ( \mathbf { x } )$   
5: Encode: $\mathbf { m _ { z } } , \mathbf { c _ { z } } , \mathbf { s _ { z } } , \mathbf { z } _ { 0 } \gets E n c o d e ( \mathbf { m } , \mathbf { c } , \mathbf { s } , \mathbf { x } )$   
6: Difusion: $\mathbf { z } _ { t } = ( 1 - t ) \cdot \epsilon + t \cdot \mathbf { z } _ { 0 } , \epsilon \sim \mathcal { N } ( 0 , \mathbf { I } )$   
7: Forward: $\mathcal { L } _ { S E F S } = | | f _ { \boldsymbol { \theta } } ( \mathbf { z } _ { t } , \mathbf { m } _ { \mathbf { z } } , \mathbf { c } _ { \mathbf { z } } , \mathbf { s } _ { \mathbf { z } } ) - ( \mathbf { z } _ { 0 } - \epsilon ) | | _ { 2 } ^ { 2 }$   
8: Backward: $\Theta _ { \mathrm { t r a i n } } = O P T ( \partial \mathcal { L } _ { S E F S } / \partial \Theta _ { \mathrm { t r a i n } } )$   
9: Return: SD3 model with trained LoRA and trainable conditioning modules

```tcl
Algorithm 2 Block-wise forward of the proposed SEFS.
1: Input: i-th block with LoRA weights $B l o c k ^ { ( i ) } ( \cdot ) , 0 \le i \le N - 1$ , and $N$ is
the number of blocks. Noisy image latent $\mathbf { z } _ { t } ,$ , style image latent $\mathbf { s _ { z } }$ , Canny
image latent $\mathbf { c _ { z } } ,$ and SAM image latent $\mathbf { m } _ { \mathbf { z } } .$ . Extra learnable parameters $\mathbf { W } _ { c a t }$
and $\mathbf { W } _ { s k i p } ^ { ( i ) } , \frac { \ d N } { \ d 2 } \leq i \leq N - 1$ . Shift and scale ${ \mathbf { W } _ { s h i f t } } , { \mathbf { W } _ { s c a l e } }$
2: Init: $j \gets \mathsf { 0 } , \mathsf { \bar { { s } } } k i p L i s t = [ ]$
3: Patchify: $\mathbf { z } _ { t } , \mathbf { s _ { z } } , \mathbf { c } _ { z } , \mathbf { m _ { z } } = P a t c h i f y ( \mathbf { z } _ { t } , \mathbf { s _ { z } } , \mathbf { c } _ { z } , \mathbf { m _ { z } } )$
4: Content Fusion: $\mathbf { z } = C o n C a t _ { D i m } ( \mathbf { z } _ { t } , \mathbf { c } _ { z } , \mathbf { m } _ { \mathbf { z } } ) \mathbf { W } _ { c a t }$
5: Style Fusion: ${ \bf z } ^ { ( 0 ) } = C o n C a t _ { S e q } ( { \bf z } , { \bf s _ { z } } )$
6: for $j \le N - 1$ do
7: if $j = = 0$ then
8: Store: skipList.append(z<sup>(j)</sup>)
9: Forward: $\mathbf { \bar { z } } ^ { ( j + 1 ) } \overset { \mathbf { \rho } } { = } B l o c k ^ { ( j ) } \big ( \mathbf { z } ^ { ( j ) } \big )$
10: Divide: $\mathbf { d _ { z } } , \mathbf { s _ { z } } \gets D i v i d e _ { S e q } ( \mathbf { z } ^ { ( j + 1 ) } )$
11: Statistics: $\mu _ { \mathbf { d } } , v _ { \mathbf { d } } \gets S t a t s ( \mathbf { d } _ { \mathbf { z } } )$
12: Statistics: $\mu _ { \mathbf { s } } , v _ { \mathbf { s } } \gets S t a t s ( \mathbf { s } _ { \mathbf { z } } )$
13: Standardization: $\begin{array} { r } { \mathbf { s } _ { z } \gets \frac { \mathbf { s } _ { \mathbf { z } } - \boldsymbol { \hat { \mu } } _ { \mathbf { s } } } { \sqrt { v _ { \mathbf { s } } + \eta } } , \eta = 1 0 ^ { - 5 } } \end{array}$
14: Re-Normalization: $\tilde { \mathbf { s } } _ { \mathbf { z } } \gets \mathbf { s } _ { z } \odot \sqrt { v _ { \mathbf { d } } + \eta } + \mu _ { \mathbf { d } }$
15: Scale $\mathbf { \sigma } / \mathbf { S h i f t } \mathbf { : } \ \mathbf { a _ { s } } \gets 1 + \tilde { \mathbf { s } } _ { \mathbf { z } } \mathbf { W } _ { s c a l e }$
16: Scale/Shift: $\mathbf { b _ { s } } \gets \tilde { \mathbf { s } } _ { \mathbf { z } } \mathbf { W } _ { s h i f t }$
17: Re-Scale: $\mathbf { s _ { z } } \gets \tilde { \mathbf { s } } _ { \mathbf { z } } \odot \mathbf { a } _ { \mathbf { s } } + \mathbf { b } _ { \mathbf { s } }$
18: Style-Fusion: $\mathbf { z } ^ { ( j + 1 ) } \gets C a t _ { S e q } ( \mathbf { d } _ { \mathbf { z } } , \mathbf { s } _ { \mathbf { z } } )$
19: else if $j \le ( N - 1 ) / 2$ then
20: Store: skipList.append $( \mathbf { z } ^ { ( j ) } )$
21: Forward: $\mathbf { \bar { z } } ^ { ( j + 1 ) } = B l o c k ^ { ( j ) } \big ( \mathbf { z } ^ { ( j ) } \big )$
22: else
23: Skip: $\begin{array} { r } { \mathbf { z } ^ { ( j ) } = C a t _ { D i m } \big ( \mathbf { z } ^ { ( j ) } , s k i p L i s t [ N - 1 - j ] \big ) } \end{array}$
24: Fusion: $\mathbf { z } ^ { ( j ) }  \mathbf { z } ^ { ( j ) } \mathbf { W } _ { s k i p } ^ { ( j ) }$
25: Forward: $\mathbf { z } ^ { ( j + 1 ) } = B l o c \dot { k } ^ { ( j ) } ( \mathbf { z } ^ { ( j ) } )$
26: end if
27: $j = j + 1$
28: end for
29: return $\mathbf { z } _ { ( N ) }$
```

Inference pipeline. Given a target content image r and a style reference image s, we compute the Canny edge map c and SAM segmentation map m from r. The style reference is passed through the same low-resolution style pathway used during training. We then encode m, $\mathbf { c } ,$ and s into latent conditions ${ \bf z } _ { \bf m } , { \bf z } _ { \bf c } ,$ and $\mathbf { z _ { s } } ,$ and denoise a randomly initialized latent conditioned on all three signals.

## 4 Experimental Results

Experimental setup. We use SD3-Medium as the base generator and train only LoRA layers, content/style projections, re-normalization scale/shift, and cross-block skip projections. A copied SD3 patch-embedding layer encodes $\mathbf { s } _ { z } , \mathbf { c } _ { z }$ and $\mathbf { m } _ { z } ;$ the original branch remains frozen.

Training uses WikiArt [25], AdamW [17], 30k steps, learning rate $2 \times 1 0 ^ { - 5 }$ weight decay 0.03, batch size 32, and 8 A100 GPUs. Evaluation protocol. The main comparison uses 1,000 fixed cross-image content-style pairs against StyleShot [9] and CSGO [28]. Canny/SAM are computed only from the target content image; style references enter through each method’s style-conditioning input. All methods use oficial checkpoints or recommended preprocessing, generate $5 1 2 \times 5 1 2$ images with 30 steps, and share prompts, seeds, and sampler settings. We use LoRA rank 16, SAM ViT-H masks, and Canny thresholds 100/200. Content DINO and CLIP-I measure target consistency; Style Ref. Sim. is CLIP image-image similarity to the reference; Leakage is a DINO diagnostic between edge-rendered outputs and style references, where lower is better; FID is computed against held-out WikiArt style-domain images. Human preference is the primary aggregate signal, and automatic metrics are diagnostics. Pseudo-triplet ablations in Tables 1 and 4 are used only for mechanism analysis.

Table 1. Content-condition and skip-fusion ablation on WikiArt. All variants use 30k steps, batch size 32, and learning rate $2 \times 1 0 ^ { - 5 }$
<table><tr><td>Skip-connection|SAM|Canny|</td><td></td><td></td><td>IS</td><td>FID</td><td>|CLIP-I</td></tr><tr><td>√</td><td></td><td></td><td>9.1788</td><td>48.7677</td><td>0.9063</td></tr><tr><td>X</td><td>√</td><td></td><td>9.1304</td><td>53.1521</td><td>0.8813</td></tr><tr><td>√</td><td>X</td><td></td><td>9.1543</td><td>49.3103</td><td>0.8983</td></tr><tr><td>V</td><td></td><td>X</td><td>9.0941</td><td>59.1620</td><td>0.8294</td></tr></table>

## 4.1 Main Comparisons

Fig. 4 compares SEFS with CSGO [28] and StyleShot [9]. Across diverse pairs, SEFS preserves target structure while transferring palette, texture, and stroke statistics. The references contain non-transferable foreground figures, village layouts, and line-art contours; baselines more often inherit these structures or distort target geometry.

Table 2 reports the main comparison. Standard baselines use their released pipelines. To isolate the content-control confound, StyleShot + Struct. Ctrl. keeps the StyleShot style encoder and injection pathway, adds the same target-derived Canny/SAM branch as SEFS, and trains only the added structural projection and LoRA layers with the same data, rank, and step budget. It does not use the SEFS low-resolution style-token pathway or re-normalization.

Style  
SEFS  
CSGO  
StyleShot  
Content  
SEFS  
CSGO  
StyleShot  
![](images/40695ea62d7272b74a97a04e69ac279996e62f894c69bd0033ede7af4ea49936.jpg)  
Fig. 4. Qualitative comparisons. SEFS better preserves target layout while transferring reference appearance; baselines either weaken style transfer or introduce reference induced content drift.

Table 2. Main comparison. StyleShot + Struct. Ctrl. gives StyleShot the same targetderived Canny/SAM controls as SEFS. Time is per image on A100. Lower Leakage/FID and higher content/style scores are better.
<table><tr><td>Method</td><td>|Content DINO|Content CLIP-I|Style Ref. Sim.|Leakage</td><td></td><td></td><td></td><td>FID</td><td>|Time</td></tr><tr><td>StyleShot</td><td>0.8017</td><td>0.8684</td><td>0.7462</td><td>0.4871</td><td>|56.8421|</td><td>4.8s</td></tr><tr><td>StyleShot + Struct. Ctrl.</td><td>0.8918</td><td>0.8971</td><td>0.7785</td><td>0.4472</td><td>51.9248</td><td>7.2s</td></tr><tr><td>CSGO</td><td>0.8098</td><td>0.8566</td><td>0.7397</td><td>0.4944</td><td>60.6377</td><td>6.3s</td></tr><tr><td>SEFS</td><td>0.9040</td><td>0.9026</td><td>0.7835</td><td>0.3931</td><td>49.5724</td><td>6.1s</td></tr></table>

The matched-control baseline narrows the content gap, confirming the value of structural conditions, but still has higher Leakage and worse FID despite similar content and Style Ref. Sim. scores. Thus, SEFS’s gain is not explained by Canny/SAM alone; under matched structure control, the low-resolution style bottleneck better suppresses non-transferable reference layout.

Leakage validation. We annotate 100 hard pairs with non-transferable objects or layouts. The DINO edge-rendering score correlates with human leakage ratings with Spearman $\rho = 0 . 6 2$ , while Style Ref. Sim. is weaker $( \rho = 0 . 2 7 )$ supporting Leakage as an auxiliary diagnostic. Eficiency. SEFS uses no auxiliary learned visual style encoder; image conditions pass through the frozen VAE, and online structural preprocessing is counted in end-to-end latency. It adds 42.7M trainable parameters, takes 3.9s model-only time on A100, and takes 6.1s endto-end with online Canny/SAM. Human preference. In blind forced-choice comparisons, 42 participants judged six content-style pairs per comparison, yielding 252 judgments for each method pair. Each question showed the same content image, style reference, and two anonymized outputs in random order.

Under the joint criterion of content preservation and style fidelity, participants preferred SEFS over StyleShot in 64.3% of comparisons (95% Wilson CI: 58.2– 69.9%), over CSGO in 72.2% (95% Wilson CI: 66.4–77.4%), and over StyleShot + Struct. Ctrl. in 62.7% (95% Wilson CI: 56.6–68.4%).

Table 3 gives cross-image mechanism controls. Removing or shufling style tokens preserves content but reduces style-reference similarity, showing that the style pathway carries transferable appearance. SEFS w/ Style Encoder keeps the same SD3 backbone, content branch, LoRA rank, data, and budget, but replaces low-resolution VAE style tokens with a frozen CLIP visual style encoder and trainable projection.

Table 3. Cross-image mechanism controls on the same 1,000 content-style pairs.
<table><tr><td>Variant</td><td>Content DINO|</td><td>|Style Ref. Sim.|Leakage</td><td></td><td>FID</td></tr><tr><td>Full SEFS</td><td>0.9040</td><td>0.7835</td><td>0.3931</td><td>49.5724</td></tr><tr><td>SEFS  $\mathrm { w } / \AA$  Style Encoder</td><td>0.8924</td><td>0.7946</td><td>0.4478</td><td>52.3861</td></tr><tr><td>w/o style tokens</td><td>0.9132</td><td>0.7048</td><td>0.3619</td><td>55.2846</td></tr><tr><td>shuffled style</td><td>0.9015</td><td>0.7162</td><td>0.3698</td><td>53.9173</td></tr><tr><td>w/o re-normalization</td><td>0.8618</td><td>0.7447</td><td>0.4326</td><td>58.6035</td></tr><tr><td>high-res style tokens</td><td>0.8654</td><td>0.7779</td><td>0.4612</td><td>55.9184</td></tr></table>

The architecture-matched control reaches slightly higher Style Ref. Sim. than SEFS, but also increases Leakage and FID, supporting the trade-of between reference afinity and reference-layout leakage.

## 4.2 Ablation Studies

Content Consistency. We sample 1,000 WikiArt test images and construct pseudo triplets with SAM, Canny [3], and a 64 × 64 random style crop, isolating content conditioning because Canny and SAM encode structure rather than appearance. For Canny-only or SAM-only variants, $\mathbf { W } _ { c a t } \in \mathbb { R } ^ { 2 D \times D }$ matches the input dimension.

Table 1 shows that the full model performs best. Removing Canny causes the largest FID and CLIP-I degradation, highlighting edge-level geometry; removing SAM has a smaller but consistent efect on coarse layout. Skip fusion further improves FID and CLIP-I.

Style Consistency. We analyze re-normalization and crop resolution with 10k-step variants. For the resolution ablation, the style-view resolution is varied during training. At evaluation time, all style views are first downsampled to 64×64 and then resized to the corresponding training resolution, so the comparison tests the learned resolution-specific pathway rather than giving higher-resolution variants extra reference information. The encoder variant replaces crop tokens with a pretrained CLIP visual encoder and trainable projection.

Table 4. Style-token ablation on WikiArt. “Style Res” is the style-view resolution; “Style Enc” replaces crop tokens with a pretrained CLIP visual encoder.
<table><tr><td>Re-Normalization|Style Res |Style Enc|</td><td></td><td></td><td>IS</td><td>FID</td><td>|CLIP-I</td></tr><tr><td>√</td><td> $6 4 \times 6 4$ </td><td>X</td><td>9.4697</td><td>54.1585</td><td>0.8821</td></tr><tr><td>X</td><td> $6 4 \times 6 4$ </td><td>X</td><td>9.0913</td><td>66.9413</td><td>0.8191</td></tr><tr><td>√</td><td> $1 2 8 \times 1 2 8$ </td><td>X</td><td>9.2169</td><td>56.9813</td><td>0.8535</td></tr><tr><td>√</td><td> $2 5 6 \times 2 5 6$ </td><td>X</td><td>9.3046</td><td>58.9413</td><td>0.8199</td></tr><tr><td>√</td><td> $2 5 6 \times 2 5 6$ </td><td>√</td><td>9.4039</td><td>57.0631</td><td>0.8517</td></tr></table>

Table 4 shows that removing re-normalization substantially degrades IS, FID, and CLIP-I. The $6 4 \times 6 4$ setting performs best, suggesting that learning with a stronger low-resolution bottleneck is beneficial; higher training resolutions weaken this bottleneck and make the style pathway more prone to reference-structure reliance. The CLIP-encoder variant does not outperform crop tokens, suggesting that a stronger generic visual representation is not necessarily a better style representation here.

## 5 Discussion

## 5.1 Methodology Comparisons

Relation to CSGO [28]. Both CSGO and SEFS are end-to-end tuning methods for difusion stylization. The main diference is the supervision and conditioning design. CSGO relies on pre-collected content-style-target triplets, whereas SEFS constructs pseudo training instances from unpaired single images. Architecturally, CSGO uses an image encoder for style extraction and a ControlNet-style branch for content control. SEFS instead uses low-resolution crop tokens for style, structural maps for content, and parameter-eficient LoRA/projection layers [11], reducing dependence on auxiliary encoders and triplet construction.

Relation to StyleShot [9]. StyleShot also aims to disentangle content and style, but does so through dedicated pretrained encoders and a two-stage training strategy. This design is practical for arbitrary style transfer, but the style representation still relies on learned encoder capacity and can retain reference identity, object semantics, or layout cues. SEFS pursues the same separation objective through data construction and token design rather than encoder capacity: content is represented by explicit structural conditions, while style is represented by low-resolution appearance tokens. This design avoids learning a separate style encoder and addresses content leakage through the scale of the style signal.

Limitations. Because SEFS deliberately compresses the style reference into low-resolution tokens, it is designed primarily for transferable local appearance statistics rather than exact global composition. It may therefore under-transfer styles whose identity depends on symbolic motifs, legible text, object-level shapes, or precise layout. This is the main trade-of of the proposed bottleneck: reducing reference-structure leakage can also suppress non-local style cues. The method also inherits errors from structural preprocessing, where inaccurate Canny edges or SAM masks can weaken content preservation. These limitations suggest future work on adaptive multi-scale style tokens and more robust structural conditioning.

## 6 Conclusion

We presented SEFS (Style-Encoder-Free Stylization), a style-encoder-free difusion stylization framework based on scale-separated conditioning. Instead of extracting style with an auxiliary visual encoder or relying on aligned triplets, SEFS learns style tokens from stochastic low-resolution crops of unpaired single images. With target-derived structural cues, re-normalization, and skip fusion, SEFS preserves target structure while transferring reference appearance. Cross-image comparisons, matched controls, and ablations indicate that the crop-token bottleneck reduces reference-layout leakage while retaining transferable style cues; the automatic metrics are used as diagnostics rather than complete definitions of stylization quality.

## References

1. Bertalmio, M., Sapiro, G., Caselles, V., Ballester, C.: Image inpainting. In: SIG GRAPH (2000)

2. Brooks, T., Holynski, A., Efros, A.A.: Instructpix2pix: Learning to follow image editing instructions. In: CVPR (2023)

3. Canny, J.: A computational approach to edge detection. TPAMI (1986)

4. Chen, J., Ge, C., Xie, E., Wu, Y., Yao, L., Ren, X., Wang, Z., Luo, P., Lu, H., Li, Z.: Pixart-sigma: Weak-to-strong training of difusion transformer for 4k text-to-image generation. In: ECCV. pp. 74–91 (2024)

5. Chen, J., Yu, J., Ge, C., Yao, L., Xie, E., Wu, Y., Wang, Z., Kwok, J., Luo, P., Lu, H., et al.: Pixart-alpha: Fast training of difusion transformer for photorealistic text-to-image synthesis. ICLR (2024)

6. Chung, J., Hyun, S., Heo, J.P.: Style injection in difusion: A training-free approach for adapting large-scale difusion models for style transfer. In: CVPR (2024)

7. Deng, Y., Tang, F., Dong, W., Ma, C., Pan, X., Wang, L., Xu, C.: Stytr2: Image style transfer with transformers. In: CVPR (2022)

8. Esser, P., Kulal, S., Blattmann, A., Entezari, R., Müller, J., Saini, H., Levi, Y., Lorenz, D., Sauer, A., Boesel, F., et al.: Scaling rectified flow transformers for high-resolution image synthesis. In: ICML (2024)

9. Gao, J., Sun, Y., Liu, Y., Tang, Y., Zeng, Y., Qi, D., Chen, K., Zhao, C.: Styleshot: A snapshot on any style. IEEE Transactions on Pattern Analysis and Machine Intelligence 48(2), 1215–1228 (2026)

10. Hertz, A., Voynov, A., Fruchter, S., Cohen-Or, D.: Style aligned image generation via shared attention. In: CVPR (2024)

11. Hu, E.J., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., Chen, W., et al.: Lora: Low-rank adaptation of large language models. In: ICLR (2022)

12. Huang, X., Belongie, S.: Arbitrary style transfer in real-time with adaptive instance normalization. In: ICCV (2017)

13. Jeong, J., Kim, J., Choi, Y., Lee, G., Uh, Y.: Visual style prompting with swapping self-attention. arXiv preprint arXiv:2402.12974 (2024)

14. Kawar, B., Zada, S., Lang, O., Tov, O., Chang, H., Dekel, T., Mosseri, I., Irani, M.: Imagic: Text-based real image editing with difusion models. In: CVPR (2023)

15. Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., Xiao, T., Whitehead, S., Berg, A.C., Lo, W.Y., et al.: Segment anything. In: ICCV (2023)

16. Liu, X., Gong, C., et al.: Flow straight and fast: Learning to generate and transfer data with rectified flow. In: ICLR (2023)

17. Loshchilov, I., Hutter, F.: Decoupled weight decay regularization. In: ICLR (2019)

18. Peebles, W., Xie, S.: Scalable difusion models with transformers. In: ICCV (2023)

19. Podell, D., English, Z., Lacey, K., Blattmann, A., Dockhorn, T., Müller, J., Penna, J., Rombach, R.: Sdxl: Improving latent difusion models for high-resolution image synthesis. In: ICLR (2024)

20. Qi, T., Fang, S., Wu, Y., Xie, H., Liu, J., Chen, L., He, Q., Zhang, Y.: Deadif: An eficient stylization difusion model with disentangled representations. In: CVPR (2024)

21. Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B.: High-resolution image synthesis with latent difusion models. In: CVPR (2022)

22. Ruiz, N., Li, Y., Jampani, V., Pritch, Y., Rubinstein, M., Aberman, K.: Dreambooth: Fine tuning text-to-image difusion models for subject-driven generation. In: CVPR (2023)

23. Shi, Y., Xue, C., Liew, J.H., Pan, J., Yan, H., Zhang, W., Tan, V.Y., Bai, S.: Dragdifusion: Harnessing difusion models for interactive point-based image editing. In: CVPR (2024)

24. Tan, S., Li, H., Zhang, J., Jia, X., Yang, X., Zhang, S., Zhang, Y.: SWIFT: Promptadaptive memory for eficient interactive long video generation. arXiv preprint arXiv:2605.09442 (2026)

25. Tan, W.R., Chan, C.S., Aguirre, H.E., Tanaka, K.: Ceci n’est pas une pipe: A deep convolutional network for fine-art paintings classification. In: ICIP (2016)

26. Wallace, B., Gokul, A., Naik, N.: Edict: Exact difusion inversion via coupled transformations. In: CVPR (2023)

27. Xie, S., Tu, Z.: Holistically-nested edge detection. In: ICCV (2015)

28. Xing, P., Wang, H., Sun, Y., Wang, Q., Bai, X., Ai, H., Huang, J.Y., Li, Z.: Csgo: Content-style composition in text-to-image generation. In: NeurIPS. pp. 111464–111504 (2025)

29. Ye, H., Zhang, J., Liu, S., Han, X., Yang, W.: Ip-adapter: Text compatible image prompt adapter for text-to-image difusion models. arXiv preprint arXiv:2308.06721 (2023)

30. Zhang, L., Rao, A., Agrawala, M.: Adding conditional control to text-to-image difusion models. In: ICCV (2023)

31. Zhang, S., Huang, J., Zhou, Q., Wang, F., Luo, J., Yan, J., et al.: Continuousmultiple image outpainting in one-step via positional query and a difusion-based approach. In: ICLR (2024)

32. Zheng, G., Chen, H., Li, H., Zhang, J., Yang, Z., Jia, X., Yang, X., Zhang, S., Zhang, Y.: Enhancing video physical consistency via role-aware joint training and modality-decoupled denoising. arXiv preprint arXiv:2607.04653 (2026)