# V-RAE: Rethinking Video Latent Spaces for Generation

Minghui Guo<sup>1</sup> Shengqiong Wu<sup>2</sup> Hao Fei<sup>2</sup>

<sup>1</sup>National University of Singapore <sup>2</sup>University of Oxford

Project Page: https://v-rae.github.io/

## Abstract

Latent video generation relies on autoencoders to define a compact space in which generative models operate. Although video autoencoder architectures have evolved substantially, their latent spaces are still optimized primarily for pixel-level reconstruction and provide limited high-level semantic organization. A reconstruction-optimal latent space, however, need not be well suited to generative modeling. We propose V-RAE, a video representation autoencoder that builds compact generative latents on top of frozen vision foundation model representations. A lightweight temporal pooling module removes temporal redundancy while preserving semantic structure, and a video decoder reconstructs continuous motion from the compressed features. We evaluate V-RAE with four representative frozen encoders on video reconstruction, semantic probing, and class-conditional generation. V-RAE achieves 2.13 rFVD on K600, outperforming all evaluated large-scale pretrained video VAEs. Its latents retain substantially more semantic information than conventional video tokenizer latents. Under matched generation settings, our best variant achieves gFVD scores of 117.86 and 19.16 on UCF101 and K600, respectively, while converging up to 6× faster. We further show that reconstruction quality alone is insufficient to characterize generative utility and introduce tFVD, a temporal-coherence diagnostic that correlates more reliably with downstream generation quality. Beyond video generation, V-RAE also improves future video prediction on Cityscapes over the Wan 2.2 VAE latent space under matched prediction settings. Taken together, the experiments show that frozen semantic representations can support video reconstruction, generation, and predictive modeling.

![](images/b20e636a3bd6ac5432ecc769039ea3ba9032d68b61bb86bc135e44d2f7ec332a.jpg)

![](images/e7a0093ff90630649901e1939b73729b5ef5979526242671c3ba1b0e6c282d67.jpg)  
Figure 1: V-RAE overview. Left: Normalized reconstruction, generation, compression, and semantic performance across video tokenizers; FVD axes (↓) are reversed so outward is better. Right: K600 gFVD convergence; V-RAE converges up to 6× faster than VAE-based latent spaces.

## Contents

1 Introduction 3   
2 Methodology 4   
2.1 Video Representation Autoencoder (V-RAE) 4   
2.2 V-RAE Learning Objective . 6   
2.3 V-RAE for Generation 6   
3 Experimental Settings 7   
4 Results and Discussion 8   
4.1 Reconstructing Videos from Semantic Latents . 8   
4.2 Generation from Semantic Latents 11   
4.3 Good Reconstruction Does Not Guarantee Good Generation 12   
4.4 Future Video Prediction in Semantic Latent Space 15   
5 Related Work 15   
6 Conclusion 17   
A Implementation Details 21   
A.1 V-RAE Architecture . 21   
A.2 Reconstruction Training Recipe 22   
A.3 Generation Training Recipe 23   
B Evaluation Protocols 25   
C Additional Qualitative Results 26   
D Limitations 26

## 1. Introduction

Modern video generation (Blattmann et al., 2023) is predominantly performed in a learned latent space produced by a video autoencoder, which compresses raw pixels into compact representations on which a diffusion- or autoregressive-based generative model operates (Peebles and Xie, 2023; Kondratyuk et al., 2023). Such compression is essential for reducing the computational cost of video generation, but it also imposes a strong inductive bias on the generative process, as the autoencoder determines the structure and information content of the latent space, thereby shaping the difficulty of generative modeling and placing an upper bound on the attainable generation quality. Most existing video autoencoders are optimized primarily for pixel-level reconstruction, including continuous VAEs (Wan et al., 2025; Kong et al., 2024; Yang et al., 2025), discrete tokenizers (Yu et al., 2023; Wang et al., 2024a), and query-based designs (Wang et al., 2025; Xiong et al., 2026). Despite their architectural differences, these methods largely prioritize local textures, colors, and fine-grained appearance details. While such objectives are effective for reconstructing observed pixels, optimizing reconstruction fidelity alone does not necessarily yield a latent space that is favorable for generative modeling (Yao et al., 2025; Skorokhodov et al., 2025; Xu et al., 2026). This mismatch is particularly consequential for video, where the generator must jointly model appearance, motion, and long-range temporal dependencies over long latent sequences.

Motivated by this limitation, some recent studies have begun to incorporate semantic information into latent video generation. At the autoencoder level, following related advances in image generation, Ma et al. (2025a) and Wu et al. (2025b) introduce semantic alignment objectives during latent-space learning. At the generator level, VideoREPA (Zhang et al., 2025) extends representation-alignment objectives such as the REPA loss (Yu et al., 2024; Leng et al., 2025) to video generation by aligning intermediate diffusion features with pretrained visual representations. Nevertheless, these approaches primarily use semantic features as auxiliary supervision for the autoencoder or generator, rather than directly defining the generative latent space. More recently, RAE (Zheng et al., 2025) and RAEv2 (Singh et al., 2026) directly adopt the feature space of a frozen image representation encoder as the generative latent space. The resulting latents are semantically rich, structurally coherent, and well-suited to diffusion modeling, while providing a shared representation for visual understanding and generation. Whether this paradigm can be effectively extended to video, with a frozen representation encoder supporting both high-fidelity reconstruction and effective video generation, remains largely unexplored.

This extension is non-trivial because video introduces three additional challenges. First, representation encoders produce temporally dense features. Image encoders provide no temporal compression, while the native video encoders considered here provide only 2× compression, far below the rate required to keep DiT training and inference affordable. Second, temporal coherence must be preserved in both compressed latents and decoded motion; otherwise, videos exhibit flicker and jitter, and the latent distribution becomes difficult for diffusion models to learn. Third, temporal compression must preserve the semantic structure inherited from the encoder, which is the central advantage of using representation features as generative latents.

To address these challenges, we propose V-RAE, a video representation autoencoder that uses frozen visual features as its latent space and adds learnable temporal compression and decoding for video generation. On the encoder side, a frozen VFM encoder integrates with a learnable lightweight temporal pooling module that removes temporal redundancy while preserving semantic structure. On the decoder side, we design a spatiotemporal Transformer decoder equipped with 3D RoPE (Su et al., 2021) to reconstruct temporally coherent videos from the compressed latents. A multi-frame unpatchify layer maps each temporal latent step to multiple consecutive frames, allowing the decoder to accommodate the effective temporal compression ratio of each representation encoder. Since the representation encoder remains frozen, the resulting latent space stays grounded in its pretrained semantic organization.

We evaluate V-RAE instantiated with four representative frozen encoders, i.e., DINOv3 (Siméoni et al., 2025), SigLIP2 (Tschannen et al., 2025), EUPE (Zhu et al., 2026), and V-JEPA 2.1 (Mur-Labadia et al., 2026), across reconstruction fidelity, semantic preservation, and generative quality. Figure 1 provides a holistic comparison between V-RAE and representative video tokenizers. For reconstruction, V-RAE with V-JEPA 2.1 attains 2.13 rFVD on K600, surpassing all evaluated large-scale pretrained video VAEs. On UCF101, its DINOv3-L variant reaches 6.12 rFVD, ranking second only to Wan2.1 VAE at 6.05. V-RAE latents also preserve substantially richer semantic information than conventional VAE latents across three probing benchmarks. In particular, V-RAE with DINOv3-L reaches 89.13% top-1 accuracy on UCF101, compared with only 30.83% for the strongest VAE baseline. The semantic richness of the learned latent space further translates into improved generation. Our best-performing variant achieves gFVD scores of 117.86 on UCF101 and 19.16 on K600, outperforming all evaluated VAE-based latent spaces while converging up to 6× faster under matched training settings. Across encoder choices, stronger semantic preservation generally coincides with better generation quality, suggesting that the semantic organization inherited from pretrained visual encoders contributes substantially to these gains. Interestingly, rFVD and gFVD produce markedly different rankings of video autoencoders, with Pearson correlations of 0.200 on UCF101 and 0.473 on K600, indicating that high reconstruction fidelity does not necessarily imply an effective latent space for generation. We characterize this discrepancy using tFVD, which evaluates temporal coherence by decoding interpolated trajectories in the latent space. Compared with rFVD, tFVD exhibits substantially stronger Pearson correlations with downstream generation quality (0.919 on K600, 0.621 on UCF101), making it a more reliable and informative diagnostic of the generative suitability of video latent spaces. Finally, we demonstrate that V-RAE provides an effective latent representation beyond video generation: on future video prediction with Cityscapes, it substantially outperforms the Wan2.2 VAE latent space under the same prediction architecture and training budget.

![](images/40cc0cf65e1f668a1884d1fd1cdee1fb8f20e564c6061c7df788c7f3f3358f8a.jpg)  
Figure 2: V-RAE architecture. A frozen visual representation encoder produces temporally dense features, which are compressed by a lightweight temporal pooling module. The figure depicts the chunk-wise causal decoder used with image encoders.

In summary, our contributions are threefold:

• We propose V-RAE, which builds a generative latent space on frozen vision representation encoders. Temporal pooling and a spatiotemporal Transformer decoder jointly address temporal compression, temporal coherence, and semantic preservation.

• We introduce tFVD, a generation-oriented diagnostic for video autoencoders, and show that reconstruction fidelity alone does not fully characterize the generative utility of a latent space.

• We present a systematic study across four encoders with distinct pretraining paradigms, demonstrating reconstruction competitive with large-scale video VAEs, better generation quality across the evaluated VAE latent spaces, and up to 6× faster convergence under matched comparisons.

## 2. Methodology

## 2.1. Video Representation Autoencoder (V-RAE)

As illustrated in Figure 2, V-RAE couples a frozen visual representation encoder with temporal pooling and a Transformer video decoder. Temporal pooling compresses visual features into a compact video latent sequence, which is then mapped back to pixels by the decoder. Only the temporal pooling module and video decoder are learned, while the pretrained representation encoder remains fixed.

Visual Representation Encoder. V-RAE is designed to operate with frozen visual representation encoders pretrained on either images or videos. For image-pretrained encoders, including DINOv3 (Siméoni et al., 2025), SigLIP2 (Tschannen et al., 2025), and EUPE (Zhu et al., 2026), we encode each video frame independently and organize the resulting spatial features along the temporal dimension. These encoders provide strong spatial and semantic representations, but do not natively model cross-frame temporal dependencies. V-RAE also supports video-native encoders that jointly encode multiple frames into spatiotemporal features. We instantiate this setting with V-JEPA 2.1-L (Mur-Labadia et al., 2026), which builds on predictive video pretraining (Assran et al., 2025) to produce dense and temporally consistent representations. This unified interface allows V-RAE to accommodate either frame-wise spatial features or native spatiotemporal features within the same subsequent compression and reconstruction framework.

Temporal Pooling. Visual representation encoders often produce temporally dense features with substantial redundancy, resulting in unnecessarily long latent sequences for generative modeling. We therefore apply a learnable temporal pooling module $\breve { \mathcal P }$ after the frozen encoder to compress adjacent features while preserving their semantic structure. Given an input video $\boldsymbol { X } \in \mathbb { R } ^ { T \times 3 \times H \times \dot { \boldsymbol { W } } }$ , the encoder produces $\boldsymbol { F } = \dot { \boldsymbol { \mathcal { E } } ( \boldsymbol { X } ) } \in \mathbb { R } ^ { T _ { E } \times C \times h \times w }$ , where $T _ { E }$ is the feature sequence length and $r _ { E } = T / T _ { E }$ is the encoder’s native temporal compression ratio. In our experiments, $r _ { E } = 1$ for DINOv3, SigLIP2, and EUPE, and $r _ { E } = 2$ for V-JEPA 2.1. The pooler introduces an additional compression ratio $r _ { P }$ by dividing � into non-overlapping temporal windows of length �� and mapping each window to a single latent feature map. We use temporal attention pooling throughout our main experiments; an architectural illustration and comparisons with alternative pooling designs are provided in Appendix $\mathrm { A . 1 }$ . The resulting latent representation is

$$
{ \cal Z } = { \mathcal P } ( { \cal F } ) = [ z _ { 0 } , \dots , z _ { T _ { { \cal Z } } - 1 } ] \in \mathbb { R } ^ { T _ { { \cal Z } } \times { \cal C } \times h \times w } , T _ { { \cal Z } } = T _ { E } / r _ { P } \ .\tag{1}
$$

The overall temporal compression ratio is therefore

$$
r _ { \mathrm { a l l } } = r _ { E } \cdot r _ { P } = { \frac { T } { T _ { Z } } } \ .\tag{2}
$$

After flattening the spatial grid, each latent feature map is represented as

$$
z _ { t } = \{ z _ { t , p } \} _ { p = 1 } ^ { h w } , \qquad z _ { t , p } \in \mathbb { R } ^ { C } ,\tag{3}
$$

where � indexes a spatial position. Similarly, we denote the encoder token at temporal position � and spatial position $p$ by $f _ { s , p } \in \mathbb { R } ^ { C }$ . For temporal attention pooling, the output token $\pmb { z } _ { t , p }$ is computed from the encoder tokens within the corresponding temporal window,

$$
\left\{ f _ { r _ { P } \cdot t + i , p } \right\} _ { i = 0 } ^ { r _ { P } - 1 } \mathrm { ~ . ~ }\tag{4}
$$

For notational simplicity, we omit the fixed output index � and spatial position $p ,$ and write $f _ { i } = f _ { r _ { P } \cdot t + i , p }$ For the �-th attention head, the key and value vectors are obtained as

$$
\pmb { k } _ { i } ^ { ( m ) } = \pmb { W } _ { K } ^ { ( m ) } \cdot \pmb { f } _ { i } , \qquad \pmb { \nu } _ { i } ^ { ( m ) } = \pmb { W } _ { V } ^ { ( m ) } \cdot \pmb { f } _ { i } .\tag{5}
$$

The attention weights and pooled features are then computed as

$$
\alpha _ { i } ^ { ( m ) } = { \mathrm { S o f t m a x } } _ { i } \left( \frac { ( \boldsymbol { q } ^ { ( m ) } ) ^ { \top } \cdot \boldsymbol { k } _ { i } ^ { ( m ) } } { \sqrt { d } } + \beta _ { i } ^ { ( m ) } \right) ~ , \qquad \boldsymbol { u } ^ { ( m ) } = \sum _ { i = 0 } ^ { r _ { P } - 1 } \alpha _ { i } ^ { ( m ) } \cdot \boldsymbol { v } _ { i } ^ { ( m ) } ~ ,\tag{6}
$$

where $\pmb q ^ { ( m ) } \in \mathbb { R } ^ { d }$ is a 1D learnable query vector shared across temporal windows and spatial positions, $\beta _ { i } ^ { ( m ) }$ is a learnable bias associated with the local temporal offset $i ,$ and $d = C / M$ is the dimension of each of the � attention heads. The pooled output token is

$$
z _ { t , p } = \mathrm { N o r m } ( W _ { O } \cdot { \mathrm { C o n c a t } } ( [ u ^ { ( m ) } ] _ { m = 1 } ^ { M } ) ) ~ ,\tag{7}
$$

where $W _ { O }$ denotes the output projection. The same operation is applied independently at every spatial position, reducing the temporal resolution without introducing additional mixing across the spatial grid. We initialize the learnable queries and temporal biases to zero and initialize the projection layers to preserve the input features. Consequently, the attention weights are uniform at the beginning of training, $\alpha _ { i } ^ { ( m ) } = 1 / r _ { P }$ , such that the attention aggregation initially reduces to temporal mean pooling before gradually learning content-adaptive temporal weighting.

V-RAE Decoder. The V-RAE decoder reconstructs the input video from the compressed latent representation, $\hat { X } = D ( Z )$ . As illustrated in Figure 2, it adopts the lightweight Transformer-decoder architecture of MAE (He et al., 2022), consisting of a stack of ViT blocks (Dosovitskiy et al., 2021). We extend RoPE (Su et al., 2021) to the temporal and two spatial dimensions, providing each latent token with explicit spatiotemporal positional information. The decoder attention pattern follows the temporal context available from the frozen representation encoder. For DINOv3-L, SigLIP2-L, and EUPE-B, which encode frames independently, we use chunk-wise causal self-attention: each temporal latent step and its associated spatial tokens form a chunk, tokens within a chunk interact bidirectionally, and each chunk can attend only to itself and preceding chunks. For V-JEPA 2.1-L, the encoder representations already aggregate information with non-causal full spatiotemporal attention. We therefore use full self-attention over all spatiotemporal latent tokens in its decoder, allowing every latent chunk to attend to both preceding and subsequent chunks. Finally, we modify the unpatchify layer to account for the overall temporal compression ratio $r _ { \mathrm { a l l } }$ Instead of reconstructing a single frame, each latent time step is mapped to $r _ { \mathrm { a l l } }$ consecutive frames. This multi-frame decoding formulation accommodates arbitrary combinations of native temporal compression in the representation encoder and additional compression introduced by the temporal pooling module.

## 2.2. V-RAE Learning Objective

The representation encoder remains frozen throughout V-RAE training, and only the temporal pooling module and decoder are optimized. The training objective combines an $L _ { 1 }$ reconstruction loss, a Learned Perceptual Image Patch Similarity (LPIPS) loss $\mathcal { L } _ { \mathrm { L P I P S } }$ (Zhang et al., 2018), and an adversarial loss L (Goodfellow et al., 2014). Following prior visual-tokenizer designs (Lu et al., 2025; Agarwal et al., 2025), we apply the Gram-matrix loss ${ \mathcal { L } } _ { \mathrm { G r a m } } .$ , which encourages the reconstructed features to match the statistics of the target features. The $L _ { 1 }$ and adversarial losses are computed over all reconstructed frames. The overall training objective is

$$
\mathcal { L } _ { \mathrm { r e c o n } } = \lambda _ { 1 } \mathcal { L } _ { 1 } + \lambda _ { \mathrm { l p i p s } } \mathcal { L } _ { \mathrm { L P I P S } } + \lambda _ { \mathrm { g a n } } \mathcal { L } _ { \mathrm { G A N } } + \lambda _ { \mathrm { g r a m } } \mathcal { L } _ { \mathrm { G r a m } } ~ .\tag{8}
$$

## 2.3. V-RAE for Generation

After reconstruction training, we freeze all V-RAE parameters and train a DiT (Peebles and Xie, 2023) in the resulting latent space using rectified flow (Liu et al., 2023). To accommodate the high dimensionality of the V-RAE latent space, we follow Zheng et al. (2025) and apply a dimension-dependent shift to the noise schedule. Specifically, we first sample � ∼ LogitNormal(0, 1) and transform it as

$$
\hat { \tau } = \frac { s \cdot \tau } { 1 + ( s - 1 ) \cdot \tau } , \qquad s = \sqrt { \frac { T _ { Z } \cdot h \cdot w \cdot C } { n } } ,\tag{9}
$$

where $n = 4 0 9 6$ is the reference latent dimension. Unlike the corresponding shift used for image RAEs, our scaling factor explicitly accounts for the temporal latent length $T _ { Z } .$ . Given a clean latent � and Gaussian noise $\epsilon ,$ we construct the noisy latent as

$$
Z _ { \hat { \tau } } = ( 1 - \hat { \tau } ) \cdot Z + \hat { \tau } \cdot \epsilon .\tag{10}
$$

We adopt a clean-latent prediction parameterization, in which the DiT directly predicts ${ \widehat { \pmb { Z } } } _ { \theta }$ from $\pmb { Z } _ { \hat { \tau } }$ and �ˆ. The clean-latent prediction is converted into a velocity estimate as

$$
\widehat { \pmb { \nu } } _ { \theta } = \frac { Z _ { \widehat { \tau } } - \widehat { Z } _ { \theta } } { \operatorname* { m a x } ( \widehat { \tau } , t _ { \epsilon } ) } \ ,\tag{11}
$$

where $t _ { \epsilon } = 0 . 0 5$ clamps the velocity-conversion denominator. The corresponding rectified flow target is $\pmb { \nu } = ( Z _ { \hat { \tau } } - \pmb { Z } ) ,$ /max $( \hat { \tau } , t _ { \epsilon } )$ , which reduces to $\epsilon - Z$ when $\hat { \tau } \geq t _ { \epsilon }$ . Following RAEv2 (Singh et al., 2026), rather than using a REPA loss (Yu et al., 2024), we attach an auxiliary clean-latent prediction head to the output of the 8-th transformer block. Let $\mathcal { L } _ { \mathrm { R F } } ^ { \mathrm { f u l l } }$ and $\mathcal { L } _ { \mathrm { R F } } ^ { \mathrm { b a s e } }$ denote the rectified flow losses computed from the final and auxiliary prediction heads, respectively. The overall training objective is

$$
\mathcal { L } _ { \mathrm { D i T } } = \mathcal { L } _ { \mathrm { R F } } ^ { \mathrm { f u l l } } + \mathcal { L } _ { \mathrm { R F } } ^ { \mathrm { b a s e } } .\tag{12}
$$

During inference, the auxiliary branch is retained to provide internal guidance during sampling. We solve the rectified flow trajectory using a 100-step Euler sampler and decode the generated latent sequence into video frames using the frozen V-RAE decoder.

Table 1: Reconstruction performance of video tokenizers on UCF101 and K600. Bold and underlined values indicate the best and second-best results across all methods, respectively.
<table><tr><td colspan="4">Tokenizer Configuration</td><td colspan="4">UCF101 test</td><td colspan="4">K600 val</td></tr><tr><td>Method</td><td>Causal</td><td>Comp.</td><td>Dim.</td><td>LPIPS↓</td><td>PSNR ↑</td><td>SSIM ↑</td><td>rFVD↓</td><td>LPIPS↓</td><td>PSNR ↑</td><td>SSIM ↑</td><td>rFVD ↓</td></tr><tr><td colspan="10">Large-Scale Pretrained Video Tokenizers</td></tr><tr><td>Wan2.1 VAE</td><td>√</td><td>4×8×8</td><td>16</td><td>0.068</td><td>34.09</td><td>0.937</td><td>6.05</td><td>0.052</td><td>35.23</td><td>0.946</td><td>3.58</td></tr><tr><td>Wan2.2 VAE</td><td>√</td><td> $4 \times 1 6 \times 1 6$ </td><td>48</td><td>0.067</td><td>34.16</td><td>0.942</td><td>12.20</td><td>0.049</td><td>35.73</td><td>0.951</td><td>4.76</td></tr><tr><td>HunyuanVideo VAE</td><td>√</td><td> $4 \times 8 \times 8$ </td><td>16</td><td>0.069</td><td>35.14</td><td>0.950</td><td>7.73</td><td>0.056</td><td>36.31</td><td>0.954</td><td>4.38</td></tr><tr><td>CogVideoX VAE</td><td>√</td><td>4×8×8</td><td>16</td><td>0.081</td><td>34.22</td><td>0.943</td><td>14.53</td><td>0.069</td><td>34.82</td><td>0.946</td><td>8.11</td></tr><tr><td>Cosmos-0.1 (CV4x8x8)</td><td>√</td><td> $4 \times 8 \times 8$ </td><td>16</td><td>0.109</td><td>32.57</td><td>0.925</td><td>17.01</td><td>0.100</td><td>33.55</td><td>0.928</td><td>9.88</td></tr><tr><td>AToken</td><td>x</td><td> $4 \times 1 6 \times 1 6$ </td><td>48</td><td>0.089</td><td>33.11</td><td>0.934</td><td>8.17</td><td>0.072</td><td>34.05</td><td>0.937</td><td>5.36</td></tr><tr><td colspan="10">Benchmark-Scale Video Tokenizers</td></tr><tr><td>Open-MAGVIT2</td><td></td><td> $4 \times 8 \times 8$ </td><td>18</td><td>0.140</td><td></td><td></td><td></td><td>0.147</td><td>27.74</td><td>0.859</td><td>17.16</td></tr><tr><td>OmniTokenizer VAE</td><td>√ √</td><td> $4 \times 8 \times 8$ </td><td>8</td><td>0.132</td><td>27.74 29.59</td><td>0.867 0.908</td><td>36.21 28.86</td><td>0.139</td><td>27.75</td><td>0.899</td><td>15.35</td></tr><tr><td>LARP-L-long</td><td>√</td><td>1024 tokens</td><td>16</td><td>0.233</td><td>25.32</td><td>0.799</td><td>142.29</td><td>0.236</td><td>26.00</td><td>0.801</td><td>125.00</td></tr><tr><td colspan="10">V-RAE Series (Ours)</td></tr><tr><td>V-RAE (DINOv3-L)</td><td>√</td><td> $4 \times 1 6 \times 1 6$ </td><td>1024</td><td>0.111</td><td>27.76</td><td>0.858</td><td>6.12</td><td>0.104</td><td>27.96</td><td>0.850</td><td>2.76</td></tr><tr><td>V-RAE (EUPE-B)</td><td>√</td><td> $4 \times 1 6 \times 1 6$ </td><td>768</td><td>0.122</td><td>26.76</td><td>0.829</td><td>8.05</td><td>0.114</td><td>27.30</td><td>0.829</td><td>3.36</td></tr><tr><td>V-RAE (SigLIP2-L)</td><td>√</td><td> $4 \times 1 6 \times 1 6$ </td><td>1024</td><td>0.142</td><td>26.41</td><td>0.815</td><td>9.83</td><td>0.134</td><td>27.05</td><td>0.816</td><td>3.38</td></tr><tr><td>V-RAE (V-JEPA2.1-L)</td><td>x</td><td> $4 \times 1 6 \times 1 6$ </td><td>1024</td><td>0.121</td><td>27.82</td><td>0.856</td><td>6.65</td><td>0.110</td><td>28.48</td><td>0.858</td><td>2.13</td></tr></table>

<sup>\*</sup> Evaluation protocol. We sample one 256 × 256 clip from the beginning of each evaluation video with a temporal stride of 3. V-RAE, AToken, and LARP-L-long use 16-frame inputs, while causal video VAEs with an independently encoded first frame use 17-frame inputs.

## 3. Experimental Settings

Datasets. Our main reconstruction and class-conditional generation benchmarks are UCF101 (Soomro et al., 2012) and Kinetics-600 (K600) (Carreira et al., 2018). For reconstruction, all V-RAE variants are trained on the union of the UCF101 and K600 training sets and evaluated on the UCF101 test set and K600 validation set, respectively. For class-conditional generation, we train a separate DiT on each dataset using 20-frame clips sampled at an interval of 3. Following Latte (Ma et al., 2025b), we evaluate UCF101 generation using 2,048 generated samples; for K600, the evaluation population and covariance statistics follow the selected FVD implementation. We additionally conduct semantic probing on UCF101, Something-Something V2 (SSv2) (Goyal et al., 2017), and Kinetics-400 (K400) (Kay et al., 2017) to assess the semantic information retained in the learned latents.

Implementation. (1) For reconstruction, we report LPIPS (Zhang et al., 2018), PSNR, SSIM (Wang et al., 2004), and rFVD. V-RAE and AToken (Lu et al., 2025) are evaluated on 16-frame clips at 256 × 256, whereas causal VAEs that encode the first frame separately are evaluated on 17-frame clips at the same resolution. Unlike LARP (Wang et al., 2025), which reports results on the training split, we extract evaluation clips from the beginning of videos in the UCF101 test set and K600 validation set. Clips are sampled at a temporal interval of 3 to expose larger motion changes, given FVD’s known bias toward frame-level appearance over temporal realism (Ge et al., 2024), while matching the Latte preprocessing protocol (Ma et al., 2025b) used in our generation experiments. This shared protocol supports the direct analysis in Section 4.3 of the relation between reconstruction FVD (rFVD) and generation FVD (gFVD), both derived from Fréchet Video Distance (Unterthiner et al., 2018). For the DINOv3- and EUPE-based V-RAE variants, we initialize the decoder from the corresponding RAEv2 checkpoint (Singh et al., 2026); the SigLIP2- and V-JEPA 2.1-based decoders are trained from scratch.

(2) For class-conditional generation, all tokenizer latent spaces use the same DiT backbone and spatial resolution. We adjust patchification according to each autoencoder’s compression ratio to maintain a fixed DiT sequence length of 1,280 tokens. Causal VAEs with a separately encoded first frame generate 17 frames directly, whereas V-RAE and AToken generate 20 frames and retain the first 17 for evaluation. Generation quality is measured by gFVD at 256 × 256. Detailed model, training, and evaluation

![](images/178843962c12431f1d8ff0803ce23cfd1c1d25ee36f14f11ef869547d29e657c.jpg)  
Figure 3: Temporal Reconstruction Error Difference (TRED) for RAEv2, Wan2.2 VAE, OmniTokenizer, and V-RAE on K600 validation examples. The heatmaps visualize temporally averaged TRED within high-frequency regions of interest, while the curves show spatially averaged TRED over consecutive frame transitions.

configurations are provided in Appendix A.

Baselines. We compare V-RAE with large-scale pretrained video autoencoders and tokenizers, including Wan2.1 VAE and Wan2.2 VAE (Wan et al., 2025), HunyuanVideo VAE (Kong et al., 2024), CogVideoX VAE (Yang et al., 2025), Cosmos VAE (Agarwal et al., 2025), AToken (Lu et al., 2025), MAGVIT-v2 (Yu et al., 2023; Luo et al., 2024), OmniTokenizer (Wang et al., 2024a), and LARP (Wang et al., 2025). Under the matched DiT setup described above, we compare both generation quality and convergence speed. We also include frame-wise RAEv2 (Singh et al., 2026) to isolate the effect of temporal modeling. Finally, we compare different temporal-pooling designs to study their effects on reconstruction quality and latent semantic quality.

## 4. Results and Discussion

## 4.1. Reconstructing Videos from Semantic Latents

Overall Reconstruction Performance. We first compare V-RAE with both large-scale pretrained and benchmark-scale video tokenizers. As shown in Table 1, V-RAE instantiated with DINOv3-L achieves an rFVD of 6.12 on UCF101, ranking second only to Wan2.1 VAE at 6.05. On K600, V-RAE with V-JEPA 2.1-L achieves an rFVD of 2.13, improving over the strongest video VAE baseline at 3.58 by 40.5%. Moreover, all four V-RAE variants outperform the strongest evaluated large-scale video VAE baseline on K600 and consistently outperform Open-MAGVIT2, OmniTokenizer, and LARP on both datasets. This comparison shows that representations extracted by frozen VFMs can form effective latent spaces for video reconstruction across datasets.

Comparing different representation encoders, the best reconstruction encoder is dataset-dependent: DINOv3-L performs best on UCF101, whereas V-JEPA 2.1-L performs best on K600. The strong results of all four variants indicate that V-RAE’s reconstruction capability is not specific to a single encoder or pretraining paradigm. They further suggest that both frame-wise semantic representations and video-native spatiotemporal representations provide effective reconstruction latents, while their relative advantages depend on the target data distribution.

Because V-RAE uses a frozen semantic encoder that is not optimized specifically for pixel-level reconstruction, it underperforms some large-scale video VAEs in terms of LPIPS, PSNR, and SSIM. Despite this, V-RAE remains highly competitive in rFVD. This result highlights a distinction between frame-level fidelity and video-level distributional similarity: stronger pixel-wise or perceptual reconstruction does not necessarily translate into a lower distributional discrepancy in spatiotemporal feature space. More importantly, it shows that high-quality video reconstruction can be achieved without re-optimizing the representation encoder for low-level pixel recovery.

To examine whether the improvement in rFVD is also reflected in local temporal stability, we introduce Temporal Reconstruction Error Difference (TRED), a diagnostic metric that quantifies frame-to-frame fluctuations in reconstruction error. Let $\mathbf { x } _ { t }$ and $\widehat { \mathbf { x } } _ { t }$ denote the ground-truth and reconstructed frames at time $t ,$ respectively, and let $\Omega _ { \mathrm { R O I } }$ denote a high-frequency region of interest selected from a K600 validation video. For each spatial location $\mathbf { p } = ( h , w ) \in$ Ω , we define the channel-averaged reconstruction error as $\begin{array} { r } { e _ { t } ( \mathbf { p } ) = \frac { 1 } { C } \sum _ { c = 1 } ^ { C } \hat { \left| x _ { t , c } ( \mathbf { p } ) - \hat { x } _ { t , c } ( \mathbf { p } ) \right| } } \end{array}$ . We summarize its temporal variation using a time-averaged spatial heatmap and a spatially averaged temporal curve:

$$
\begin{array} { r l r } & { \overline { { \mathrm { T R E D } } } _ { \mathrm { t } } ( \mathbf { p } ) = \displaystyle \frac { 1 } { T - 1 } \sum _ { t = 1 } ^ { T - 1 } \left| e _ { t + 1 } ( \mathbf { p } ) - e _ { t } ( \mathbf { p } ) \right| \ , } & { \mathbf { p } \in \Omega _ { \mathrm { R O I } } \ , } \\ & { \overline { { \mathrm { T R E D } } } _ { \mathrm { R O I } } ( t ) = \displaystyle \frac { 1 } { | \Omega _ { \mathrm { R O I } } | } \sum _ { \mathbf { p } \in \Omega _ { \mathrm { R O I } } } \left| e _ { t + 1 } ( \mathbf { p } ) - e _ { t } ( \mathbf { p } ) \right| \ , } & { t = 1 , \dots , T - 1 \ . } \end{array}\tag{13}
$$

As shown in Figure 3, both V-RAE variants exhibit markedly lower TRED and $\overline { { \mathrm { T R E D } } } _ { \mathrm { R O I } }$ values than the frame-wise RAEv2 baseline on the presented examples, while producing values close to those of Wan2.2 VAE and OmniTokenizer. These observations complement the global distributional comparison provided by rFVD with a localized diagnostic of reconstruction-error stability. In particular, they suggest that the explicit temporal modeling introduced in V-RAE reduces abrupt cross-frame error fluctuations and thereby alleviates visible reconstruction flicker.

The Impact of Temporal Pooling. We compare V-RAE with the frame-wise RAEv2 baseline to isolate the contribution of temporal modeling. As shown in Figure $^ { 4 , }$ each pair uses the same frozen image VFM (DINOv3-L, SigLIP2-L, or EUPE-B), thereby controlling for the underlying semantic representation. RAEv2 reconstructs frames independently, whereas V-RAE couples neighboring features through temporal pooling and decodes them with a causal video decoder. Together with the TRED visualization in Figure 3, these results highlight the importance of temporal pooling and cross-frame self-attention for improving temporal consistency and reducing flicker and jitter in reconstructed videos.

![](images/6a0cf45d6240374b1cc25e12fa8c75ed6f366170bc606d557012d09dde313389.jpg)  
Figure 4: Effect of temporal modeling on UCF101 reconstruction; lower rFVD is better.

Comparison of Temporal Pooling Designs. To isolate the effect of temporal pooling, we keep the frozen representation encoder, video decoder, and training recipe fixed, and vary only the pooling mechanism. Detailed structures of the four pooling designs are provided in Appendix A.1. The trade-offs in Figure 5 can therefore be attributed primarily to the temporal aggregation strategy. Mean pooling introduces no additional parameters and achieves the highest probing accuracy (91.82%), suggesting that uniform averaging largely preserves the semantic structure inherited from the pretrained encoder. However, it yields substantially worse reconstruction fidelity, with a 94.25 rFVD, as it cannot adapt its aggregation to content-dependent temporal variations. Convolutional pooling substantially reduces rFVD to 5.91 through learnable local temporal mixing, but its probing accuracy drops to 78.26%, indicating that improved reconstruction comes at the cost of semantic preservation. Increasing the pooling capacity does not resolve this trade-off: Q-former pooling uses 67M parameters, but neither improves reconstruction (6.87 rFVD) nor fully preserves semantic information (84.37% probing accuracy). Additional capacity alone is therefore insufficient to balance reconstruction and semantic preservation. In contrast, temporal attention pooling adaptively weights features according to their temporal content. With only 3M parameters, it achieves 6.12 rFVD while retaining 89.13% probing accuracy, providing a substantially better reconstruction-semantics balance than the other pooling strategies. We therefore adopt temporal attention pooling as the default configuration in all remaining experiments.

![](images/60349dab3787ac8986f29d1a10c25c2bd970e7af94e4c203ee0b5aff9d849c52.jpg)  
Figure 5: Reconstruction-semantics trade-off of temporal pooling designs on the UCF101 test set. Point labels report rFVD / probing accuracy / parameters; the horizontal axis is broken to retain resolution among the learnable designs.

Table 2: Top-1 probing accuracy of RAE- and VAE-based video tokenizers. Parentheses indicate the change from the corresponding frozen model.
<table><tr><td>Model</td><td>Temporal Compression</td><td colspan="2">UCF101 ↑</td><td colspan="2">SSv2↑</td><td colspan="2">K400 ↑</td></tr><tr><td>Probe Type</td><td>Ratio</td><td>Linear Probe Single</td><td>TTA</td><td>Attentive Probe Single</td><td>TTA</td><td>Attentive Probe Single</td><td>TTA</td></tr><tr><td colspan="8">RAE-based Video Autoencoders</td></tr><tr><td>DINOv3-L</td><td>1×</td><td>91.84</td><td>91.84</td><td>69.92</td><td>70.21</td><td>86.32</td><td>87.15</td></tr><tr><td>V-RAE (DINOv3-L)</td><td>4×</td><td>89.13 (-2.71)</td><td>88.94(-2.90)</td><td>66.55(-3.37)</td><td>67.19 (-3.02)</td><td>83.12 (-3.20)</td><td>84.36 (-2.79)</td></tr><tr><td>EUPE-B</td><td>1x</td><td>93.86</td><td>93.78</td><td>67.17</td><td>67.64</td><td>83.53</td><td>84.21</td></tr><tr><td>V-RAE (EUPE-B)</td><td>4x</td><td>90.16 (-3.70)</td><td>89.93 (-3.85)</td><td>65.67 (-1.50)</td><td>66.21 (-1.43)</td><td>82.21 (-1.32)</td><td>83.31 (-0.90)</td></tr><tr><td>SigLIP2-L</td><td>1×</td><td>94.05</td><td>93.55</td><td>68.54</td><td>68.80</td><td>86.00</td><td>86.93</td></tr><tr><td>V-RAE (SigLIP2-L)</td><td>4×</td><td>90.92 (-3.13)</td><td>90.88 (-2.67)</td><td>65.39 (-3.15)</td><td>66.08 (-2.72)</td><td>82.56 (-3.44)</td><td>83.91 (-3.02)</td></tr><tr><td>V-JEPA 2.1-L</td><td>2×</td><td>93.02</td><td>92.83</td><td>76.58</td><td>77.11</td><td>84.29</td><td>85.20</td></tr><tr><td>V-RAE (V-JEPA 2.1-L)</td><td>4×</td><td>86.65 (-6.37)</td><td>86.65 (-6.18)</td><td>72.91 (-3.67)</td><td>73.50 (-3.61)</td><td>80.70 (-3.60)</td><td>81.87 (-3.34)</td></tr><tr><td colspan="8">VAE-based Video Autoencoders</td></tr><tr><td>Wan2.2 VAE</td><td>4X</td><td>16.94</td><td>16.29</td><td>41.86</td><td>42.62</td><td>46.13</td><td>48.11</td></tr><tr><td>AToken</td><td>4×</td><td>30.83</td><td>30.29</td><td>45.05</td><td>45.95</td><td>53.27</td><td>55.32</td></tr><tr><td>CogVideoX VAE</td><td>4X</td><td>14.51</td><td>14.14</td><td>37.03</td><td>38.34</td><td>41.59</td><td>43.53</td></tr></table>

Semantic Preservation. We probe the temporally compressed generation tokens on UCF101, SSv2, and K400, following the DINOv3 protocol (Siméoni et al., 2025). Representative VAE-based video tokenizers are included for comparison.

As shown in Table 2, V-RAE latents retain substantially richer semantic information than conventional VAE latents. V-RAE achieves 90.92% on UCF101 with SigLIP2-L, 72.91% on SSv2 with V-JEPA 2.1, and 83.12% on K400 with DINOv3-L, compared with 30.83%, 45.05%, and 53.27% for the strongest evaluated VAE baseline, respectively. The gains across diverse benchmarks indicate that the compressed latents retain both appearance and motion semantics.

Temporal compression also largely preserves the semantics of the original encoder features: the three image-encoder variants remain within 3.85 percentage points of their frozen encoders across all datasets and evaluation settings. V-JEPA 2.1 shows a larger 6.37-point decrease on UCF101, but remains within 3.67 points on SSv2 and K400 while achieving the strongest V-RAE result on SSv2. Temporal attention pooling therefore substantially reduces temporal redundancy with only modest semantic degradation.

## FINDING 1 Temporal compression need not erase semantics.

The key is not pooling capacity but content-adaptive aggregation: an effective temporal compressor selectively removes temporal redundancy while preserving the semantic structure inherited from the pretrained encoder.

Table 3: Class-conditional video generation performance comparison on UCF101 and K600 using different video tokenizers and video VAEs as latent spaces.
<table><tr><td>Method</td><td>Resolution, Frames</td><td>Generator</td><td>#Tokens</td><td>UCF101 gFVD ↓</td><td>K600 gFVD ↓</td></tr><tr><td colspan="6">VAE-Based Video Autoencoders</td></tr><tr><td>Wan2.1 VAE</td><td> $2 5 6 \times 2 5 6 , 1 7$ </td><td>DiT XL/2</td><td>1280</td><td>148.20</td><td>53.75</td></tr><tr><td>Wan2.2 VAE</td><td> $2 5 6 \times 2 5 6 , 1 7$ </td><td>DiT XL/1</td><td>1280</td><td>154.64</td><td>52.25</td></tr><tr><td>HunyuanVideo VAE</td><td> $2 5 6 \times 2 5 6 , 1 7$ </td><td>DiT XL/2</td><td>1280</td><td>211.53</td><td>53.28</td></tr><tr><td>CogVideoX VAE</td><td> $2 5 6 \times 2 5 6 , 1 7$ </td><td>DiT XL/2</td><td>1280</td><td>159.20</td><td>51.83</td></tr><tr><td>Cosmos VAE (CV4x8x8)</td><td> $2 5 6 \times 2 5 6 , 1 7$ </td><td>DiT XL/2</td><td>1280</td><td>152.70</td><td>41.66</td></tr><tr><td>AToken</td><td> $2 5 6 \times 2 5 6 , 2 0 \to 1 7$ </td><td>DiT XL/1</td><td>1280</td><td>143.00</td><td>46.74</td></tr><tr><td colspan="6">V-RAE Series (Ours)</td></tr><tr><td>V-RAE (DINOv3-L)</td><td> $2 5 6 \times 2 5 6 , 2 0 \to 1 7$ </td><td> $\mathrm { D i T } ^ { \mathrm { { D H } } } \times \mathrm { L } / 1$ </td><td>1280</td><td>131.40</td><td>30.09</td></tr><tr><td> $\mathbf { V } – \mathbf { R } \mathbf { A } \mathbf { E } \left( { \mathrm { S i g L I P 2 – L } } \right)$ </td><td> $2 5 6 \times 2 5 6 , 2 0 \to 1 7$ </td><td> $\mathrm { D i T } ^ { \mathrm { { D H } } } \times \mathrm { L } / 1$ </td><td>1280</td><td>142.60</td><td>34.48</td></tr><tr><td>V-RAE (EUPE-B)</td><td> $2 5 6 \times 2 5 6 , 2 0 \to 1 7$ </td><td> $\mathrm { D i T } ^ { \mathrm { { D H } } } \times \mathrm { L } / 1$ </td><td>1280</td><td>125.98</td><td>24.77</td></tr><tr><td> ${ \bf V - R A E _ { \alpha } } \left( { \bf V - J E P A } \ : 2 . 1 { \mathrm { - L } } \right)$ </td><td> $2 5 6 \times 2 5 6 , 2 0 \to 1 7$ </td><td> $\mathrm { D i T } ^ { \mathrm { { D H } } } \times \mathrm { L } / 1$ </td><td>1280</td><td>117.86</td><td>19.16</td></tr></table>

<sup>\*</sup> Note: DiT<sup>DH</sup> denotes dual-head prediction. Following RAE (Zheng et al., 2025), which reports no benefit and sometimes degradation for lowdimensional VAE latents, VAE baselines retain single-head prediction. All methods use 1280 latent tokens; 20→17 denotes generating 20 frames and cropping to 17 for evaluation.

## 4.2. Generation from Semantic Latents

Overall Generation Performance. We then evaluate the same semantic latents as a space for classconditional video generation. Following the shared training configuration in Appendix A.3, we train a class-conditional DiT from scratch on UCF101 and K600 for each tokenizer and decode its generated latents with the corresponding decoder. We match the latent budget at 1,280 tokens across methods, thereby isolating the effect of the latent representation under comparable generation settings.

Table 3 shows that all four V-RAE variants outperform conventional video tokenizers on both datasets. V-RAE with V-JEPA 2.1 performs best, achieving 117.86 gFVD on UCF101 and 19.16 on K600, improvements of 25.14 and 22.50 points over the strongest non-V-RAE baselines. EUPE-B also achieves 125.98 and 24.77, confirming that the advantage extends across representation encoders. Figures 6 and 7 show improved visual integrity and cross-frame consistency; additional DINOv3-L and SigLIP2-L results appear in Appendix C.

Optimization Efficiency. Final gFVD captures generation quality only after training. To examine optimization efficiency, we additionally track gFVD throughout training under matched optimization settings. Figure 8 compares V-RAE with Wan2.2 VAE on UCF101. On UCF101, V-JEPA 2.1 and EUPE-B maintain lower gFVD than Wan2.2 VAE at every evaluated checkpoint. In particular, V-JEPA 2.1 reaches, in roughly 30K updates, a gFVD comparable to that attained by Wan2.2 VAE after 150K updates, corresponding to approximately 5× faster convergence. As shown in Figure 1, the advantage is more pronounced on K600. EUPE-B matches the gFVD reached by Wan2.2 VAE at 180K updates after only 30K updates. This corresponds to 6× faster convergence, while at 180K updates it outperforms Wan2.2 VAE by 24.1 gFVD points.

![](images/de07f538816b3d715bb977aff13deda0b29b261245eba1f89f873b58422581e3.jpg)  
Figure 8: gFVD convergence on UCF101 under matched training settings. V-RAE converges up to 5× faster than Wan2.2 VAE.

These convergence results complement the final scores in Table 3 and show that semantic latents are not only more effective generation targets but also substantially easier to optimize.

![](images/0e77593cc976c0038db417c80d482b35cd2d87b8893668671a0a6924042ea5e0.jpg)  
Figure 6: Qualitative class-conditional generation on UCF101. Dashed lines separate the baselines and V-RAE models; each strip shows three frames from one sample.

FINDING 2 Semantic organization makes video generation easier to learn.

Semantic latents expose objects, actions, and scene structure to the generator, allowing it to focus on modeling how visual states evolve rather than rediscovering semantics and dynamics from reconstruction-oriented codes.

## 4.3. Good Reconstruction Does Not Guarantee Good Generation

Reconstruction quality alone does not characterize whether a latent space is well suited to generative modeling. Reconstruction evaluates the deterministic path $\boldsymbol { X } \stackrel { \mathcal { E } } { \longrightarrow } \boldsymbol { F } \stackrel { \mathcal { P } } { \longrightarrow } \boldsymbol { Z } \stackrel { \mathcal { D } } { \longrightarrow } \boldsymbol { \widehat { X } } ,$ , where the decoder receives latents encoded from real videos. A generator faces a different challenge: it must learn the latent distribution and generate new latent samples that can be reliably decoded. Consequently, an autoencoder may reconstruct real samples faithfully while still inducing a latent space that is difficult for a downstream generator to model.

Our results expose this discrepancy directly. On UCF101, HunyuanVideo VAE achieves a lower rFVD than V-RAE with EUPE-B (7.73 vs. 8.05), but a substantially worse gFVD (211.53 vs. 125.98). On K600, Wan2.1 VAE and V-RAE with SigLIP2-L achieve similar rFVD scores (3.58 vs. 3.38), yet their gFVD scores differ substantially (53.75 vs. 34.48). Thus, fidelity at encoded data points does not by itself reveal how easily the surrounding latent space can be learned and decoded.

![](images/9952c6572a72bc0fe0cffac2d8453377357c3bb2fd73f67215e96b6a38e1a171.jpg)  
Figure 7: Qualitative class-conditional generation on Kinetics-600. Dashed lines separate the baselines and V-RAE models; each strip shows three frames from one sample.

![](images/ebc239c458d5f7c08f39dbc18d3dda35474bb474c0e3c772a90e9e430f0eb424.jpg)

![](images/eb2a27eb31eaddc59eb952e55e480bde150596438667dcbf1411bd123b1db08f.jpg)  
(b) Visualization of interpolated reconstructions  
Figure 9: Overview of tFVD. (a) Each interior latent is replaced by the midpoint of its two temporal neighbors, and the interpolated sequence is decoded to compute FVD against real clips. (b) Qualitative comparison of interpolated reconstructions from V-RAE and a conventional VAE.

Temporal Fréchet Video Distance (tFVD). To assess this property without training a complete generator, we introduce tFVD, a generation-oriented diagnostic that probes the local temporal geometry of a video latent space. The complete evaluation protocol is provided in Appendix B. Given a temporally ordered latent sequence $Z = \left[ z _ { 0 } , z _ { 1 } , \dots , z _ { L + 1 } \right]$ with � interior codes, where $\pmb { z } _ { 0 }$ is the first-frame latent for causal VAEs and the first-four-frame latent for V-RAE, we replace each interior code by the midpoint of its two temporal neighbors:

![](images/79d4910f585dd933e4a4b9d57a60573b3498ffb39519fe1c89a2bb199bfe1b69.jpg)

![](images/07c9bb74e64df2b64dfc9b884b9ecd87ed55596210b1ff48bd1de5f0e0881382.jpg)

![](images/06181c8d1840a6435fb27d5c9b0c8acb875d2890bb2021600d4185be2f487d29.jpg)  
Wan2.1 HunyuanVideo Cosmos V-RAE (DINOv3-L) V-RAE (EUPE-B) Wan2.2 CogVideoX AToken V-RAE (SigLIP2-L) V-RAE (V-JEPA 2.1)

![](images/f7ea9ce8767d76eabb593e7eee66cf908f9eac3c935323a45f3c8438f05bce7a.jpg)  
Figure 10: Metric-correlation analysis on UCF101 and K600, comparing reconstruction fidelity and temporal interpolation consistency against downstream generation quality.

$$
z _ { t } ^ { \prime } = \frac { 1 } { 2 } \left( z _ { t - 1 } + z _ { t + 1 } \right) \ , \qquad t = 1 , \ldots , L \ .\tag{14}
$$

We then form the perturbed latent sequence $\pmb { Z } ^ { \prime } = \left[ \pmb { z } _ { 1 } ^ { \prime } , \ldots , \pmb { z } _ { L } ^ { \prime } \right]$ , decode it, and compute FVD (Unterthiner et al., 2018) between the reconstructed clips and their corresponding real clips:

$$
\mathrm { \ t F V D = F V D \left( \{ \right)} X \} , \{ \mathcal { D } ( Z ^ { \prime } ) \}  \ .\tag{15}
$$

A lower tFVD indicates that local interpolations between temporally adjacent latent codes can be decoded into temporally coherent videos.

Unlike rFVD, this construction deliberately breaks the direct encode-decode path by requiring the decoder to process latent codes that are not directly produced by the encoder at corresponding time steps. It therefore serves as a controlled stress test rather than a full generative process. If the temporal latent trajectory is highly curved or discontinuous, its local midpoints may leave the region that decodes to natural videos, causing ghosting, flicker, or abrupt motion. A smoother temporal latent space should instead decode these interpolated codes into coherent intermediate motion, potentially making it more tolerant to latent prediction errors during generation.

Figure 9(b) illustrates the resulting difference. For the jumping sequence, V-RAE better preserves the subject, action trajectory, and scene structure across the interpolated frames, whereas the VAE exhibits stronger ghosting, texture corruption, and structural instability. Although both models can reconstruct encoded inputs, their behavior between encoded temporal states is markedly different.

Predicting Generation Performance. To assess whether autoencoder-level metrics predict downstream generation quality, we compare rFVD and tFVD with $\mathrm { g F V D }$ under matched generator settings. As shown in Figure 10, rFVD is substantially less correlated with gFVD: the Pearson correlations are $r = 0 . 2 0 0$ on UCF101 and $r = 0 . 4 7 3$ on K600. In contrast, tFVD increases the correlations to $r = 0 . 6 2 1$ and $r = 0 . 9 1 9 \AA$ respectively. The substantially stronger association across both datasets shows that temporal interpolation consistency provides a more informative estimate of generative utility than reconstruction fidelity alone.

The analysis shows that reconstruction fidelity alone is insufficient for evaluating a video tokenizer. A generation-friendly latent space must also be semantically organized, temporally smooth, and robust to small deviations from encoded trajectories. Together with the semantic probing and generation results, tFVD indicates that V-RAE benefits not merely from reconstructing videos with frozen representation features, but from organizing those features into a coherent latent space that is easier for a video generator to learn.

## FINDING 3 Generation-friendly latents require more than reconstruction fidelity.

A good video latent should remain temporally smooth and decodable under prediction errors. tFVD probes this latent temporal smoothness and decoder robustness more directly than rFVD.

Table 4: Future video prediction results on Cityscapes at 40k training steps. Both methods use the same conditional DiT and training budget.
<table><tr><td>Latent space</td><td>rFVD↓</td><td>tFVD↓</td><td> $\mathbf { g F I D \downarrow }$ </td><td> $\mathbf { g F V D \downarrow }$ </td></tr><tr><td>Wan2.2 VAE</td><td>7.0256</td><td>319.0233</td><td>15.02</td><td>144.47</td></tr><tr><td>V-RAE (EUPE-B)</td><td>29.2931</td><td>224.6040</td><td>11.52</td><td>111.36</td></tr></table>

## 4.4. Future Video Prediction in Semantic Latent Space

Beyond class-conditional video generation, we investigate whether V-RAE can provide a directly decodable latent state space for visual world modeling, i.e., predicting future clips based on context. Technically, let $\mathbf { \boldsymbol { x } } ^ { c }$ and $\mathbf { \boldsymbol { x } } ^ { f }$ denote the context and future clips, respectively. The prediction pipeline is formulated as

$$
\begin{array} { r } { \mathbf { Z } ^ { c } = \mathcal { P } ( \mathcal { E } ( \mathbf { X } ^ { c } ) ) , \qquad \mathbf { Z } ^ { f } = \mathcal { P } \Big ( \mathcal { E } \Big ( \mathbf { X } ^ { f } \Big ) \Big ) , \qquad \widehat { \mathbf { Z } } ^ { f } \sim p _ { \theta } \Big ( \mathbf { Z } ^ { f } \mid \mathbf { Z } ^ { c } \Big ) , \qquad \widehat { \mathbf { X } } ^ { f } = \mathcal { D } \Big ( \widehat { \mathbf { Z } } ^ { f } \Big ) , } \end{array}\tag{16}
$$

where ${ \bf Z } ^ { c }$ and ${ \bf Z } ^ { f }$ are the context and target future latents produced by the frozen representation encoder $\varepsilon$ followed by the temporal pooler ${ \mathcal { P } } ,$ and $\widehat { \mathbf { Z } } ^ { f }$ is the prediction sampled from the conditional latent DiT �� (·|·) given ${ \bf Z } ^ { c } .$ . The V-RAE decoder D then renders the predicted future clip $\widehat { \mathbf { X } } ^ { f }$ . This formulation jointly evaluates whether the latent space supports predictable future dynamics and whether predicted states can be decoded while preserving scene geometry, object identity, and motion trajectories.

Experimental Setup. We evaluate on the Cityscapes driving dataset (Cordts et al., 2016), conditioning on frames 4-15 and predicting frames 16-27 at a resolution of 432 × 768. V-RAE leverages EUPE-B (Zhu et al., 2026) as its representation encoder and is also fine-tuned on CoVLA (Arai et al., 2025) for high-resolution video reconstruction. We take the Wan2.2 VAE (Wan et al., 2025) as the conventional latent baseline. Both latent spaces share the same Transformer backbone, latent-token budget, training schedule, and sampling configuration, with only the latent-channel-dependent input and output projections adjusted. We compute gFID (Heusel et al., 2017) over $6 { , } 0 0 0$ future frames generated from 500 validation sequences, and gFVD (Unterthiner et al., 2018) over the corresponding 500 12-frame videos.

Results and Analysis. Under the same conditional DiT and training budget, V-RAE reduces gFID from 15.02 to 11.52 and gFVD from 144.47 to 111.36, demonstrating that its latent space is more effective for modeling future video distributions. Despite its worse reconstruction fidelity (29.2931 vs. 7.0256 rFVD), V-RAE achieves a substantially lower tFVD (224.6040 vs. 319.0233) together with superior future prediction. This reversal echoes Section 4.3: tFVD probes temporal smoothness and decoder robustness to off-trajectory latent states, properties that are more relevant than reconstruction fidelity when a predictor must generate approximate future codes. Because the downstream architecture and optimization budget are controlled, the results support attributing much of the performance gap to the choice of latent representation. Figure 11 provides consistent qualitative evidence: both methods preserve static road and building layouts, but the Wan2.2 baseline increasingly blurs vehicles, pedestrians, and bicycles and distorts their relative positions at longer horizons, whereas V-RAE better maintains object contours, lane structure, and instance separation. The lower tFVD and improved qualitative stability indicate that a semantically organized and temporally smooth latent space reduces structural and identity drift during future-state prediction, leading to more realistic and coherent videos.

## FINDING 4 Semantic latents form a directly decodable predictive state space.

A predictor learns transitions between semantic states, while the same frozen decoder renders predicted states into pixels, unifying future-state prediction and visual reconstruction within a single latent interface.

## 5. Related Work

Representation Learning for Generation. Pretrained visual representations have increasingly been incorporated into generative models. In the image domain, prior work aligns tokenizer latents with pretrained features (Ma et al., 2025a; Wu et al., 2025b), aligns generator features through auxiliary objectives (Yu et al., 2024; Leng et al., 2025; Wu et al., 2025a), or directly adopts frozen encoder features as the generative latent space (Zheng et al., 2025; Singh et al., 2026; Tong et al., 2026). In video, DERA (Guo et al., 2025) aligns tokenizer representations, while VideoREPA (Zhang et al., 2025) and Wu et al. (2025a) align diffusion features. Divot (Ge et al., 2025) models continuous video-tokenizer features for unified understanding and generation, but learns those features specifically as a video tokenizer rather than inheriting a frozen VFM feature space. To the best of our knowledge, V-RAE is the first to perform video generative modeling directly in a temporally compressed, high-dimensional semantic feature space inherited from frozen vision foundation models, with generated latents decoded into pixel-space videos.

![](images/4c42a3e7b41daa4c07cf8d19e7960763a01a6f883cc9c822a8fcff91051771f2.jpg)  
(b) Pedestrian-dominated intersection scene  
Figure 11: Qualitative comparison of future video prediction on Cityscapes. We show (a) a traffic scene dominated by vehicle motion and (b) an intersection scene containing pedestrians and bicycles. In each panel, the top row contains six context frames, followed by ground-truth future frames, predictions using the Wan2.2 VAE baseline, and V-RAE predictions. The three future frames correspond to successive latent chunks at $X _ { t + 4 } , X _ { t + 8 } .$ , and $X _ { t + 1 2 }$ . Red dashed boxes highlight dynamic regions where the two latent spaces differ most clearly.

Video Autoencoders. Video autoencoders provide the latent spaces used by modern video generators (Blattmann et al., 2023). Representative designs include continuous video VAEs (Wan et al., 2025; Kong et al., 2024; Yang et al., 2025; Wang et al., 2024b), discrete tokenizers (Yu et al., 2023; Wang et al., 2024a; Kondratyuk et al., 2023; Agarwal et al., 2025), and query-based tokenizers (Wang et al., 2025; Tan et al., 2025; Xiong et al., 2026). These approaches primarily optimize compact representations exclusively for pixel-level reconstruction, leaving their latent spaces devoid of semantic structure. In contrast, V-RAE starts from a frozen semantic representation space and learns temporal compression and pixel decoding, while preserving sufficient semantics and producing smooth temporal trajectories.

## 6. Conclusion

We introduce V-RAE, a video representation autoencoder that repurposes frozen visual representation models as semantically structured latent spaces for video reconstruction and generation. V-RAE achieves near-best reconstruction on UCF101 and outperforms all evaluated video autoencoders on K600, while retaining substantially richer semantic information than conventional VAE-based video tokenizers. When used as the latent space of a DiT, V-RAE further improves generation quality and converges up to 6× faster under matched training settings. Our analysis also reveals that reconstruction and generation metrics induce markedly different rankings of video autoencoders, indicating that pixel reconstruction quality alone is insufficient to characterize the suitability of a latent space for generation. To better diagnose this property, we introduced tFVD, which evaluates decoded trajectories between latent samples and more reliably reflects downstream generation performance than rFVD. Preserving the semantic and temporal organization of pretrained representations therefore provides an effective alternative to learning video latents solely through reconstruction objectives.

## References

Niket Agarwal, Arslan Ali, Maciej Bala, Yogesh Balaji, Erik Barker, Tiffany Cai, Prithvijit Chattopadhyay, Yongxin Chen, Yin Cui, Yifan Ding, et al. Cosmos world foundation model platform for physical ai. arXiv preprint arXiv:2501.03575, 2025.

Hidehisa Arai, Keita Miwa, Kento Sasaki, Kohei Watanabe, Yu Yamaguchi, Shunsuke Aoki, and Issei Yamamoto. CoVLA: Comprehensive vision-language-action dataset for autonomous driving. In Proceedings of the IEEE/CVF Winter Conference on Applications ofComputer Vision, pages 1933–1943, 2025.

Mido Assran, Adrien Bardes, David Fan, Quentin Garrido, Russell Howes, Mojtaba Komeili, Matthew Muckley, Ammar Rizvi, Claire Roberts, Koustuv Sinha, et al. V-JEPA 2: Self-supervised video models enable understanding, prediction and planning. arXiv preprint arXiv:2506.09985, 2025.

Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023.

Joao Carreira, Eric Noland, Andras Banki-Horvath, Chloe Hillier, and Andrew Zisserman. A short note about Kinetics-600. arXiv preprint arXiv:1808.01340, 2018.

Marius Cordts, Mohamed Omran, Sebastian Ramos, Timo Rehfeld, Markus Enzweiler, Rodrigo Benenson, Uwe Franke, Stefan Roth, and Bernt Schiele. The Cityscapes dataset for semantic urban scene understanding. In Proceedings ofthe IEEE Conference on Computer Vision and Pattern Recognition, pages 3213–3223, 2016.

Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In International Conference on Learning Representations, 2021.

Songwei Ge, Aniruddha Mahapatra, Gaurav Parmar, Jun-Yan Zhu, and Jia-Bin Huang. On the content bias in fréchet video distance. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024.

Yuying Ge, Yizhuo Li, Yixiao Ge, and Ying Shan. Divot: diffusion powers video tokenizer for comprehension and generation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 13606–13617, 2025.

Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. In Advances in Neural Information Processing Systems, volume 27, 2014.

Raghav Goyal, Samira Ebrahimi Kahou, Vincent Michalski, Joanna Materzynska, Susanne Westphal, Heuna Kim, Valentin Haenel, Ingo Fruend, Peter Yianilos, Moritz Mueller-Freitag, Florian Hoppe, Christian Thurau, Ingo Bax, and Roland Memisevic. The “something something” video database for learning and evaluating visual common sense. In Proceedings ofthe IEEE International Conference on Computer Vision, pages 5842–5850, 2017.

Pengbo Guo, Junke Wang, Zhen Xing, Chengxu Liu, Daoguo Dong, Xueming Qian, and Zuxuan Wu. Dera: Decoupled representation alignment for video tokenization. arXiv preprint arXiv:2512.04483, 2025.

Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, and Ross Girshick. Masked autoencoders are scalable vision learners. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 16000–16009, 2022.

Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. GANs trained by a two time-scale update rule converge to a local Nash equilibrium. In Advances in Neural Information Processing Systems, volume 30, 2017.

Will Kay, Joao Carreira, Karen Simonyan, Brian Zhang, Chloe Hillier, Sudheendra Vijayanarasimhan, Fabio Viola, Tim Green, Trevor Back, Paul Natsev, Mustafa Suleyman, and Andrew Zisserman. The kinetics human action video dataset. arXiv preprint arXiv:1705.06950, 2017.

Dan Kondratyuk, Lijun Yu, Xiuye Gu, José Lezama, Jonathan Huang, Grant Schindler, Rachel Hornung, Vighnesh Birodkar, Jimmy Yan, Ming-Chang Chiu, et al. Videopoet: A large language model for zero-shot video generation. arXiv preprint arXiv:2312.14125, 2023.

Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024.

Xingjian Leng, Jaskirat Singh, Yunzhong Hou, Zhenchang Xing, Saining Xie, and Liang Zheng. Repa-e: Unlocking vae for end-to-end tuning of latent diffusion transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 18262–18272, 2025.

Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. In International Conference on Learning Representations, 2023.

Jiasen Lu, Liangchen Song, Mingze Xu, Byeongjoo Ahn, Yanjun Wang, Chen Chen, Afshin Dehghan, and Yinfei Yang. Atoken: A unified tokenizer for vision. arXiv preprint arXiv:2509.14476, 2025.

Zhuoyan Luo, Fengyuan Shi, Yixiao Ge, Yujiu Yang, Limin Wang, and Ying Shan. Open-MAGVIT2: An open-source project toward democratizing auto-regressive visual generation. arXiv preprint arXiv:2409.04410, 2024.

Chuofan Ma, Yi Jiang, Junfeng Wu, Jihan Yang, Xin Yu, Zehuan Yuan, Bingyue Peng, and Xiaojuan Qi. Unitok: A unified tokenizer for visual generation and understanding. Advances in Neural Information Processing Systems, 38: 129274–129297, 2025a.

Xin Ma, Yaohui Wang, Gengyun Jia, Xinyuan Chen, Ziwei Liu, Yuan-Fang Li, Cunjian Chen, and Yu Qiao. Latte: Latent diffusion transformer for video generation. Transactions on Machine Learning Research, 2025b.

Lorenzo Mur-Labadia, Matthew Muckley, Amir Bar, Mahmoud Assran, Koustuv Sinha, Michael Rabbat, Yann LeCun, Nicolas Ballas, and Adrien Bardes. V-JEPA 2.1: Unlocking dense features in video self-supervised learning. arXiv preprint arXiv:2603.14482, 2026.

William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4195–4205, 2023.

Oriane Siméoni, Huy V. Vo, Maximilian Seitzer, et al. DINOv3. arXiv preprint arXiv:2508.10104, 2025.

Jaskirat Singh, Boyang Zheng, Zongze Wu, Richard Zhang, Eli Shechtman, and Saining Xie. Improved baselines with representation autoencoders. arXiv preprint arXiv:2605.18324, 2026.

Ivan Skorokhodov, Sharath Girish, Benran Hu, Willi Menapace, Yanyu Li, Rameen Abdal, Sergey Tulyakov, and Aliaksandr Siarohin. Improving the diffusability of autoencoders. arXiv preprint arXiv:2502.14831, 2025.

Khurram Soomro, Amir Roshan Zamir, and Mubarak Shah. UCF101: A dataset of 101 human actions classes from videos in the wild. arXiv preprint arXiv:1212.0402, 2012.

Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, and Yunfeng Liu. RoFormer: Enhanced transformer with rotary position embedding. arXiv preprint arXiv:2104.09864, 2021.

Zhentao Tan, Ben Xue, Jian Jia, Junhao Wang, Wencai Ye, Shaoyun Shi, Mingjie Sun, Wenjin Wu, Quan Chen, and Peng Jiang. Sweettok: Semantic-aware spatial-temporal tokenizer for compact video discretization. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pages 23541–23550, 2025.

Shengbang Tong, Boyang Zheng, Ziteng Wang, Bingda Tang, Nanye Ma, Ellis Brown, Jihan Yang, Rob Fergus, Yann LeCun, and Saining Xie. Scaling text-to-image diffusion transformers with representation autoencoders. arXiv preprint arXiv:2601.16208, 2026.

Michael Tschannen, Alexey Gritsenko, Xiao Wang, Muhammad Ferjad Naeem, Ibrahim Alabdulmohsin, Nikhil Parthasarathy, Talfan Evans, Lucas Beyer, Ye Xia, Basil Mustafa, et al. SigLIP 2: Multilingual vision-language encoders with improved semantic understanding, localization, and dense features. arXiv preprint arXiv:2502.14786, 2025.

Thomas Unterthiner, Sjoerd van Steenkiste, Karol Kurach, Raphaël Marinier, Marcin Michalski, and Sylvain Gelly. Towards accurate generative models of video: A new metric and challenges. arXiv preprint arXiv:1812.01717, 2018.

Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.

Hanyu Wang, Saksham Suri, Yixuan Ren, Hao Chen, and Abhinav Shrivastava. Larp: Tokenizing videos with a learned autoregressive generative prior. In International Conference on Learning Representations, volume 2025, pages 60671–60697, 2025.

Junke Wang, Yi Jiang, Zehuan Yuan, Binyue Peng, Zuxuan Wu, and Yu-Gang Jiang. Omnitokenizer: A joint image-video tokenizer for visual generation. Advances in Neural Information Processing Systems, 37:28281–28295, 2024a.

Yuchi Wang, Junliang Guo, Xinyi Xie, Tianyu He, Xu Sun, and Jiang Bian. Vidtwin: Video vae with decoupled structure and dynamics. arXiv preprint arXiv:2412.17726, 2024b.

Zhou Wang, Alan C. Bovik, Hamid R. Sheikh, and Eero P. Simoncelli. Image quality assessment: From error visibility to structural similarity. IEEE Transactions on Image Processing, 13(4):600–612, 2004.

Ge Wu, Shen Zhang, Ruijing Shi, Shanghua Gao, Zhenyuan Chen, Lei Wang, Zhaowei Chen, Hongcheng Gao, Yao Tang, Ming-Ming Cheng, et al. Representation entanglement for generation: Training diffusion transformers is much easier than you think. Advances in Neural Information Processing Systems, 38:7714–7743, 2025a.

Shengqiong Wu, Hao Fei, Xiangtai Li, Jiayi Ji, Hanwang Zhang, Tat-Seng Chua, and Shuicheng Yan. Towards semantic equivalence of tokenization in multimodal llm. In International Conference on Learning Representations, volume 2025, pages 93334–93358, 2025b.

Tianwei Xiong, Jun Hao Liew, Zilong Huang, Zhijie Lin, Jiashi Feng, and Xihui Liu. Evatok: Adaptive length video tokenization for efficient visual autoregressive generation. arXiv preprint arXiv:2603.12267, 2026.

Tongda Xu, Mingwei He, Shady Abu-Hussein, Jose Miguel Hernandez-Lobato, Haotian Zhang, Kai Zhao, Chao Zhou, Ya-Qin Zhang, and Yan Wang. Making reconstruction fid predictive of diffusion generation fid. arXiv preprint arXiv:2603.05630, 2026.

Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. In International Conference on Learning Representations, volume 2025, pages 83048–83077, 2025.

Jingfeng Yao, Bin Yang, and Xinggang Wang. Reconstruction vs. generation: Taming optimization dilemma in latent diffusion models. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 15703–15712. IEEE, 2025.

Lijun Yu, José Lezama, Nitesh B Gundavarapu, Luca Versari, Kihyuk Sohn, David Minnen, Yong Cheng, Vighnesh Birodkar, Agrim Gupta, Xiuye Gu, et al. Language model beats diffusion–tokenizer is key to visual generation. arXiv preprint arXiv:2310.05737, 2023.

Sihyun Yu, Sangkyung Kwak, Huiwon Jang, Jongheon Jeong, Jonathan Huang, Jinwoo Shin, and Saining Xie. Representation alignment for generation: Training diffusion transformers is easier than you think. arXiv preprint arXiv:2410.06940, 2024.

Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 586–595, 2018.

Xiangdong Zhang, Jiaqi Liao, Shaofeng Zhang, Fanqing Meng, Xiangpeng Wan, Junchi Yan, and Yu Cheng. Videorepa: Learning physics for video generation through relational alignment with foundation models. Advances in Neural Information Processing Systems, 38:122647–122676, 2025.

Boyang Zheng, Nanye Ma, Shengbang Tong, and Saining Xie. Diffusion transformers with representation autoencoders. arXiv preprint arXiv:2510.11690, 2025.

Chenchen Zhu, Saksham Suri, Cijo Jose, Maxime Oquab, Marc Szafraniec, Wei Wen, Yunyang Xiong, Patrick Labatut, Piotr Bojanowski, Raghuraman Krishnamoorthi, and Vikas Chandra. Efficient universal perception encoder. arXiv preprint arXiv:2603.22387, 2026.

## A. Implementation Details

We provide detailed model and training configurations for reproducibility. Section A.1 describes the V-RAE architecture and temporal pooling designs, Section A.2 specifies the reconstruction recipe and shared VideoMAE discriminator, and Section A.3 reports the generation architecture, training, sampling, and evaluation settings.

## A.1. V-RAE Architecture

Table 5: VFM-dependent V-RAE architecture. Block indices are zero-based; shared pooler and decoder details are described in text.
<table><tr><td>Configuration</td><td>DINOv3-L</td><td>SigLIP2-L</td><td>EUPE-B</td><td>V-JEPA 2.1-L</td></tr><tr><td colspan="5">Encoder settings</td></tr><tr><td>Backbone (blocks / width)</td><td>ViT-L/16 (24 / 1024)</td><td>ViT-L/16-256 (24 / 1024)</td><td>ViT-B/16 (12 / 768)</td><td>ViT-L/16 (24 / 1024)</td></tr><tr><td>Selected feature blocks</td><td>11,13,15,17,19,21,23</td><td>11,13,15,17,19,21,23</td><td>5,6,7,8,9,10,11</td><td>11,13,15,17,19,21,23</td></tr><tr><td>Patch / encoder tubelet</td><td>16 / 1</td><td>16 / 1</td><td>16 / 1</td><td>16 / 2</td></tr><tr><td>Input norm.</td><td>ImageNet-style</td><td>Native processor</td><td>ImageNet-style</td><td>ImageNet-style</td></tr><tr><td>Encoder temporal length</td><td>T</td><td>T</td><td>T</td><td>T/2</td></tr><tr><td colspan="5">Decoder settings</td></tr><tr><td>Temporal group size</td><td>4</td><td>4</td><td>4</td><td>2</td></tr><tr><td>Latent width C</td><td>1024</td><td>1024</td><td>768</td><td>1024</td></tr><tr><td>Decoder initialization</td><td>RAEv2 image decoder</td><td>Scratch</td><td>RAEv2 image decoder</td><td>Scratch</td></tr><tr><td>Decoder attention mask</td><td>Chunk-causal</td><td>Chunk-causal</td><td>Chunk-causal</td><td>Full (non-causal)</td></tr><tr><td>Trainable Params</td><td colspan="5">~420M</td></tr></table>

Representation encoders. We instantiate V-RAE with DINOv3-L, SigLIP2-L, EUPE-B, and V-JEPA 2.1-L and keep the representation encoder frozen throughout reconstruction training. DINOv3-L, SigLIP2-L, and V-JEPA 2.1-L use 1024-dimensional features, whereas EUPE-B uses 768-dimensional features. The three image encoders preserve the input temporal length, while the tubelet size of two in V-JEPA 2.1-L reduces it by half. We therefore pool groups of four image-encoder features or two V-JEPA features, producing the same temporal compression ratio of four for all variants. A 16-frame, 256 × 256 clip is represented by a $4 \times 1 6 \times \mathsf { \bar { 1 6 } }$ latent grid with channel width � given in Table 5.

Non-affine latent normalization in temporal pooling. The temporal attention operation is described in the main text; here we focus on the normalization applied to its output. Let ˜� denote the pooled feature. Before latent-noise augmentation, we apply LayerNorm without a learned scale or bias:

$$
z _ { \mathrm { c l e a n } } = \mathrm { L N } _ { \mathrm { n a } } ( \tilde { z } ) , \qquad z _ { \mathrm { t r a i n } } = z _ { \mathrm { c l e a n } } + \sigma \epsilon .\tag{17}
$$

This choice is important for noise-augmented reconstruction training. With an affine LayerNorm, $z _ { \mathrm { c l e a n } } =$ $\gamma \odot \mathrm { L N } ( \tilde { \mathfrak { z } } ) + \beta ,$ , the optimizer can increase � while the injected noise magnitude � remains fixed. The effective signal-to-noise ratio then grows approximately as $\gamma ^ { 2 } / \sigma ^ { 2 }$ , providing a shortcut that reduces the relative corruption without requiring the decoder to become robust to noise. We therefore fix $\gamma = 1$ and $\beta = 0$ . This removes the rescaling shortcut, keeps the latent scale calibrated, and forces the decoder to reconstruct under the intended noise level. With identity-initialized projections and zero-initialized queries and temporal biases, the pooler initially reduces to temporal mean pooling followed by this fixed normalization.

V-RAE decoder. All variants share a ViT-XL video decoder with width 1152, 28 transformer blocks, 16 attention heads, and a 4096-dimensional GELU feed-forward network. The decoder combines a learned 2D spatial embedding with 3D RoPE $( \theta = 1 0 , 0 0 0 )$ . The DINOv3-L, SigLIP2-L, and EUPE-B reconstruction models use chunk-causal attention: tokens interact freely within a latent chunk, while each chunk can attend only to itself and earlier chunks. The V-JEPA 2.1-L model instead uses non-causal full self-attention over all spatiotemporal latent tokens, consistent with the full temporal context provided by its frozen encoder. Each output token predicts a 4 × 16 × 16 × 3 = 3072-dimensional RGB tubelet, which is rearranged into four consecutive frames. For DINOv3-L and EUPE-B, compatible transformer weights are initialized from the corresponding RAEv2 image decoder, and the image prediction head is inflated along time; the other two decoders are trained from scratch.

![](images/d03f3ec8aced7dc135cf0834d4f64982a04e3d14c5bfc1fb67a44147e434cc21.jpg)  
Figure 12: Structures of the four temporal pooling designs. All modules reduce the encoder features to the same temporal latent length, differing only in how features within each temporal group are aggregated.

Temporal pooling designs. Temporal compression must remove frame-level redundancy without destroying the semantic geometry inherited from the frozen representation encoder. We study this trade-off along two design axes: whether aggregation is content-adaptive, and how strongly the pooler transforms the encoder feature space. For a fair comparison, we fix the DINOv3-L encoder, video decoder, training data, temporal compression ratio, and reconstruction objective. Every module maps one nonoverlapping temporal group to one latent feature map on the same spatial grid; only the aggregation operator changes. Figure 12 summarizes the resulting progression from fixed averaging to learned latent resampling.

The four designs expose different inductive biases. Mean pooling in Figure 12(a) is parameter-free and leaves the feature basis untouched, making it a useful reference for semantic preservation. However, it assigns the same weight to every frame and cannot distinguish a stable region from rapid motion or changes in appearance. The 3D convolution in Figure 12(b) instead learns a shared local spatiotemporal filter. This favors the reconstruction of local details, but its aggregation weights remain fixed after training, and convolution can modify the spatial organization of the pretrained features.

Temporal attention in Figure 12(c) retains the same spatial aggregation constraint while making the temporal weights depend on the input content. A single 1D query per head is shared across spatial positions and temporal groups, so the module learns which frames to retain without introducing a separate query grid or a deep feature transformation. Identity-initialized projections and zero-initialized queries and temporal biases make its initial behavior close to mean pooling; optimization can then depart from this conservative starting point only when reconstruction provides evidence for doing so. The final non-affine LayerNorm fixes the output scale for the noise-augmented decoder training described above.

Q-Former pooling in Figure 12(d) represents the opposite end of the capacity spectrum. Its learned 2D query grid is refined by four transformer blocks with 3D RoPE, allowing repeated spatial and temporal feature rewriting before producing the latent. This flexibility is useful when strong latent resampling is required, but is unnecessarily heavy for pooling short temporal groups and can move the output farther from the frozen encoder’s semantic feature space. The ablation in Figure 5 reflects these trade-offs: mean pooling best preserves probing accuracy but reconstructs poorly, convolutional pooling improves reconstruction at a substantial semantic cost, and the much larger Q-Former does not improve the overall balance. Temporal attention offers content-adaptive aggregation with only about 3M parameters and provides the best practical compromise, so we use it throughout the remaining experiments.

## A.2. Reconstruction Training Recipe

Reconstruction training. We jointly train the temporal pooler and video decoder while keeping the representation encoder frozen in evaluation mode. Training uses the union of UCF101 train split 1 and the

K600 training split. We sample 16-frame clips at an interval of 3 from a random temporal position, resize the short side to 256, and apply a random 256 × 256 crop and horizontal flip. DINOv3-L and EUPE-B are trained for 40 epochs, while SigLIP2-L and V-JEPA 2.1-L are trained for 50 epochs. The remaining shared training hyperparameters are listed in Table 6.

Table 6: Shared reconstruction training hyperparameters.
<table><tr><td>Configuration</td><td>Value</td></tr><tr><td>Global batch size</td><td>128</td></tr><tr><td>Optimizer</td><td>AdamW</td></tr><tr><td>Learning rate</td><td> $2 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Betas</td><td>(0.9,0.95)</td></tr><tr><td>Epsilon</td><td>10⁻8</td></tr><tr><td>Weight decay</td><td>0</td></tr><tr><td>Learning-rate schedule</td><td>Constant</td></tr><tr><td>Gradient clipping</td><td>Global norm 1.0</td></tr><tr><td>Precision</td><td>BF16</td></tr><tr><td>EMA decay</td><td>0.9995</td></tr><tr><td>L1 loss weight</td><td>1</td></tr><tr><td>LPIPS loss weight</td><td>1</td></tr><tr><td>GAN generator loss weight</td><td>0.3</td></tr><tr><td>Gram loss weight</td><td>100</td></tr></table>

Reconstruction objective. For each clip, the frozen encoder and temporal pooler produce a clean latent ${ \mathfrak { z } } _ { \mathrm { c l e a n } }$ . We perturb it during reconstruction training as $z _ { \mathrm { t r a i n } } = z _ { \mathrm { c l e a n } } + \sigma \epsilon ,$ where $\sigma \sim \mathcal { U } ( 0 , 0 . 8 )$ and $\epsilon \sim { \cal N } ( 0 , I )$ . The video decoder is optimized with L1, LPIPS, GAN, and Gram losses, whose weights are listed in Table 6. LPIPS is evaluated at the native 256 × 256 resolution, whereas the Gram branch uses $2 2 4 \times 2 2 4$ inputs; both perceptual branches cover all 16 frames. The latent perturbation is used only as a reconstruction-training augmentation; evaluation, latent-statistics estimation, and latent video generation all use clean latents.

Table 7: Shared VideoMAE discriminator configuration.
<table><tr><td>Configuration</td><td>Value</td></tr><tr><td>Backbone</td><td>Frozen VideoMAE-B</td></tr><tr><td>Feature taps</td><td>Embedding output and blocks 2, 5, 8, 11</td></tr><tr><td>Temporal head</td><td>Conv1d, kernel size 9</td></tr><tr><td>Normalization</td><td>Local batch normalization + spectral normalization</td></tr><tr><td>Adversarial start</td><td>Epoch 30</td></tr><tr><td>Update interval</td><td>One discriminator step per five generator steps</td></tr><tr><td>Optimizer</td><td>AdamW</td></tr><tr><td>Learning rate</td><td> $2 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>Learning-rate schedule</td><td>Constant</td></tr><tr><td>Betas</td><td>(0.9,0.95)</td></tr><tr><td>Weight decay</td><td>10⁻⁴</td></tr></table>

Adversarial objective. Beginning at epoch 30, we add the same VideoMAE-based adversarial objective for all four variants. The VideoMAE-B backbone remains frozen, and only the temporal Conv1d heads are optimized. We use a non-saturating generator loss and a smoothed non-saturating discriminator loss, both weighted by 0.3, and update the discriminator once every five generator steps. Translation, color, and cutout DiffAug are applied with probability 1.0 and cutout ratio 0.2. We additionally use LeCam regularization with weight 0.001 and logit-EMA decay 0.9995.

## A.3. Generation Training Recipe

DiT architecture. Our goal is to compare the latent spaces induced by different autoencoders, rather than the capacity of their downstream generators. We therefore use a high-capacity DiT to reduce the risk that limited generator expressiveness becomes the dominant bottleneck. The backbone and latent-token budget are matched across autoencoders, so differences in generation quality and convergence more directly reflect how readily each latent distribution can be modeled. Table 8 summarizes the shared latent DiT architecture and the task-specific configurations for UCF101, K600, and Cityscapes.

Table 8: Latent video DiT configurations for UCF101, K600, and Cityscapes.
<table><tr><td>Configuration</td><td>UCF101 generation</td><td>K600 generation</td><td>Cityscapes prediction</td></tr><tr><td colspan="4">Input and token geometry</td></tr><tr><td>RGB input</td><td>20 × 2562, interval 3</td><td>20 × 2562, interval 3</td><td>Frames 4–15 → 16–27, 432 × 768</td></tr><tr><td>Clean latent grid</td><td>5 × 16 × 16 × C</td><td>5 × 16 × 16 × C</td><td>3 × 27 × 48 × 768 each</td></tr><tr><td>Encoder / output tokens</td><td>1280 / 1280</td><td>1280 / 1280</td><td>7776 / 3888</td></tr><tr><td>Latent patch size</td><td>1×1</td><td>1×1</td><td>1×1</td></tr><tr><td colspan="4">DiT encoder and output decoder</td></tr><tr><td>Main encoder width / blocks</td><td>1536  /  28</td><td>1536 / 28</td><td>1152 / 28</td></tr><tr><td>Main encoder heads / head dim</td><td>24  /  64</td><td>24/64</td><td>16 / 72</td></tr><tr><td>Output decoder width / blocks</td><td>2048  /  2</td><td>2048 /  2</td><td>2048 / 2</td></tr><tr><td>Output decoder heads / head dim</td><td>16 / 128</td><td>16 / 128</td><td>16 / 128</td></tr><tr><td>Self-attention</td><td>Full spatiotemporal</td><td>Full spatiotemporal</td><td>Full spatiotemporal</td></tr><tr><td>Positional encoding</td><td>3D RoPE (θ = 10,000)</td><td>3D RoPE, (θ = 10,000)</td><td>3D RoPE, (θ = 10,000)</td></tr><tr><td>Block conditioning</td><td>AdaLN-Zero</td><td>AdaLN-Zero</td><td>AdaLN-Zero</td></tr><tr><td>Normalization / epsilon</td><td>RMSNorm / 10−6</td><td>RMSNorm / 10-6</td><td>RMSNorm / 10−6</td></tr><tr><td>FFN / MLP ratio</td><td>SwiGLU / 4.0</td><td>SwiGLU / 4.0</td><td>SwiGLU / 4.0</td></tr><tr><td>Time embedding</td><td>256-D Gaussian Fourier</td><td>256-D Gaussian Fourier</td><td>256-D Gaussian Fourier</td></tr><tr><td>Auxiliary base branch</td><td>Encoder block 8</td><td>Encoder block 8</td><td>Encoder block 8</td></tr><tr><td>Trainable parameters</td><td>1.368B</td><td>1.369B</td><td>0.843B</td></tr><tr><td colspan="4">Conditioning</td></tr><tr><td>Condition</td><td>Class label</td><td>Class label</td><td>Clean context latent</td></tr><tr><td>Number of classes</td><td>101</td><td>600</td><td></td></tr><tr><td>Condition dropout</td><td>0.1</td><td>0.1</td><td>0.1</td></tr><tr><td>Context embedding</td><td>一</td><td></td><td>Dropout → mean pool → 2-layer MLP</td></tr><tr><td colspan="4">Optimization</td></tr><tr><td>Global batch size</td><td>64</td><td>128</td><td>8</td></tr><tr><td>Optimizer / learning rate</td><td>AdamW / 10-4</td><td>AdamW / 10−4</td><td>AdamW / 3 × 10−4</td></tr><tr><td>Weight decay</td><td>0 Linear to 5 × 10−5</td><td>0 Linear to 5 × 10−5</td><td>0 Cosine to 10-4</td></tr><tr><td>LR schedule Warmup</td><td>10 epochs from zero</td><td>2 epochs from zero</td><td>1,490 updates from zero</td></tr><tr><td>Gradient clipping</td><td>Global norm 1.0</td><td>Global norm 1.0</td><td>Global norm 1.0</td></tr><tr><td>Precision</td><td>BF16</td><td>BF16</td><td>BF16</td></tr><tr><td>Training length / EMA</td><td>1700 epochs / 0.9995</td><td>100 epochs / 0.9995</td><td>40k steps / 0.9995</td></tr><tr><td colspan="4">Flow matching</td></tr><tr><td>Prediction parameterization Time distribution</td><td>x-prediction</td><td>x-prediction LogitNormal(0, 1)</td><td>x-prediction</td></tr><tr><td>Velocity denominator clamp te</td><td>LogitNormal(0, 1)</td><td></td><td>LogitNormal(0, 1)</td></tr><tr><td>Full / base loss weights</td><td>0.05</td><td>0.05</td><td>0.05 1.0 / 1.0</td></tr><tr><td></td><td>1.0 / 1.0</td><td>1.0 / 1.0</td><td></td></tr><tr><td colspan="4">Online sampling and evaluation</td></tr><tr><td>Online sampler / steps</td><td>Euler / 100</td><td>Euler / 100</td><td>Euler / 100</td></tr><tr><td>Online CFG / internal guidance</td><td>1.0 / 1.3</td><td>1.0 / 1.2</td><td>1.0 / 1.2</td></tr><tr><td>Evaluation population</td><td>2048 clips</td><td>50,000 clips</td><td>500 clips</td></tr><tr><td>Video metric</td><td>17-frame gFVD</td><td>17-frame gFVD</td><td>12-frame gFID + gFVD</td></tr></table>

For UCF101 and K600, all four representation encoders use the same class-conditional DiT backbone: a 28-block, 1536-dimensional transformer encoder and a two-block, 2048-dimensional output decoder. Only the input channel count and corresponding time shift change for EUPE-B. Cityscapes retains the 28-block depth but uses a 1152-dimensional encoder with 16 attention heads of dimension 72 and EUPE-B latents with 768 channels. It uses a clean-context/noisy-future token sequence; the output decoder predicts only the future tokens. All variants use full self-attention, per-head Q/K RMSNorm, 3D RoPE, AdaLN-Zero conditioning, RMSNorm, and SwiGLU.

For conditioning, UCF101 and K600 use class embeddings with classifier-free dropout probability 0.1. Cityscapes instead conditions on clean context latents; the context is randomly dropped with probability 0.1, mean pooled, and mapped by a two-layer MLP. A secondary prediction branch is attached after encoder block 8 and is used for the auxiliary training objective and internal guidance.

Training details. We freeze the complete V-RAE and compute latent statistics separately for each dataset and encoder. Cityscapes uses the CoVLA-fine-tuned EUPE-B V-RAE checkpoint and statistics computed from training future latents only. UCF101 trains for 1,700 epochs with batch size 64 and a 10-epoch warmup; K600 trains for 100 epochs with batch size 128 and a 2-epoch warmup. Both linearly decay the learning rate from $1 0 ^ { - 4 }$ to $\mathsf { \bar { 5 } } \times 1 0 ^ { - 5 }$ . Cityscapes instead runs for 40,000 steps with batch size 8, a 1,490-update linear warmup to $3 \times 1 0 ^ { - 4 }$ , and cosine decay to $1 0 ^ { - 4 }$ . All configurations use BF16 mixed precision and an EMA decay of 0.9995. For flow matching, given a normalized clean latent $\scriptstyle { \boldsymbol { x } } _ { 0 }$ and Gaussian noise �, we form the interpolated latent

$$
\begin{array} { r } { { \boldsymbol { x } } _ { t } = ( 1 - t ) \cdot { \boldsymbol { x } } _ { 0 } + t \cdot { \boldsymbol { \epsilon } } \ , \qquad t = \frac { s \cdot t _ { 0 } } { 1 + ( s - 1 ) \cdot t _ { 0 } } \ , \qquad t _ { 0 } \sim \mathrm { L o g i t N o r m a l } ( 0 , 1 ) \ . } \end{array}\tag{18}
$$

The time shift � is calculated from the latent dimensionality as defined in Equation (9). The DiT predicts $\scriptstyle { \boldsymbol { x } } _ { 0 }$ and converts the prediction to velocity before evaluating the flow-matching MSE. The final objective sums the main prediction loss and the auxiliary loss from encoder block 8 with unit weight for each term.

Sampling and evaluation. We sample all generation and prediction models from their EMA weights using a 100-step Euler solver from � = 1 to $t = 0 .$ . Online samples use CFG 1.0, with internal guidance 1.3 for UCF101 and 1.2 for K600 and Cityscapes active over � ∈ [0.10, 1.0]. Reported results are computed separately from online monitoring, using the benchmark-specific frame ranges and evaluation populations described below.

## B. Evaluation Protocols

Unless stated otherwise, we evaluate EMA checkpoints. The rFVD, tFVD, and gFVD metrics use clip features from an I3D network pretrained on Kinetics-400, whereas gFID uses Inception-v3 pool3 features. All Fréchet means and covariances are accumulated in float64.

Reconstruction. We evaluate on the UCF101 test set and the K600 validation set. Each clip is taken from the beginning of a video with a temporal interval of $3 ;$ we resize the short side and apply a center crop to 256 × 256. V-RAE and AToken use 16-frame clips, while causal VAEs that encode the first frame separately use 17-frame clips, following Table 1. K600 contributes 27,874 clips after removing 36 videos that are too short; no temporal padding is used. LPIPS, PSNR, and SSIM are evaluated over all valid frames, and rFVD is computed between the reconstructed and real clip distributions.

Temporal interpolation. For tFVD, we construct a temporally ordered sequence of six clean latent codes $\left[ \pmb { z } _ { 0 } , \dots , \pmb { z } _ { 5 } \right]$ using each tokenizer’s native temporal grouping. Here, �<sub>0</sub> is the separately encoded first-frame latent for causal VAEs and the latent chunk of the first four frames for V-RAE; later codes follow the tokenizer’s native temporal order. We replace the four interior codes with local midpoints:

$$
\widetilde { \pmb { Z } } = \left[ \frac { \pmb { z } _ { 0 } + \pmb { z } _ { 2 } } { 2 } , \frac { \pmb { z } _ { 1 } + \pmb { z } _ { 3 } } { 2 } , \frac { \pmb { z } _ { 2 } + \pmb { z } _ { 4 } } { 2 } , \frac { \pmb { z } _ { 3 } + \pmb { z } _ { 5 } } { 2 } \right] .\tag{19}
$$

The four interpolated codes are decoded from temporal position zero, producing a 16-frame clip. We compare it with the aligned ground-truth frames 4-19 of the input clip and compute tFVD using the same I3D implementation as rFVD.

Class-conditional generation. All tokenizer comparisons are evaluated on 17-frame videos at $2 5 6 \times 2 5 6$ V-RAE and AToken first generate 20 frames and retain the first 17 for metric computation; causal VAEs with a separate first-frame encoder generate 17 frames directly. UCF101 gFVD uses 2,048 generated clips, while K600 uses 50,000. Within each benchmark, every tokenizer uses the same evaluation population and the same 100-step Euler sampling protocol. V-RAE samples use EMA DiT weights and CFG 1.0; the internal-guidance scale is fixed within each reported comparison.

Future prediction. For Cityscapes, the model conditions on frames 4-15 and predicts frames 16-27 at 432 × 768 and 16 fps. We evaluate all 500 validation sequences. The same 12-frame predictions are used for both metrics: gFID is computed over the resulting 6,000 frames, and gFVD over the 500 video clips. Sampling uses the EMA model, 100 Euler steps, CFG 1.0, and internal guidance 1.2 over � ∈ [0.10, 1.0].

## C. Additional Qualitative Results

Figure 13 provides additional samples from the DINOv3-L and SigLIP2-L variants of V-RAE. Together with Figures 6 and 7, these results cover all four V-RAE representation encoders evaluated in our generation experiments.

![](images/2ca182944f58e9a702e778c1e29665ed9c1fc2563a1b606d6752492bb66ce68c.jpg)  
(b) Kinetics-600  
Figure 13: Class-conditional generation results for V-RAE with DINOv3-L and SigLIP2-L

## D. Limitations

We finally discuss several limitations of the current work that motivate future research. First, our experiments primarily focus on relatively controlled video benchmarks and moderate-scale generative models. While these settings allow us to isolate the effect of the latent representation, it remains unclear whether the advantages of V-RAE extend to large-scale open-domain text-to-video generation with larger datasets, higher resolutions, longer videos, and substantially larger DiT backbones. Second, although we evaluate several representative image- and video-based encoders, we do not systematically disentangle how different pretraining objectives, architectures, and spatiotemporal inductive biases affect reconstruction and generation. Finally, while our experiments reveal a strong association between semantic preservation and downstream generation quality, the precise relationship between semantic structure in the latent space and its generative utility remains to be better understood.