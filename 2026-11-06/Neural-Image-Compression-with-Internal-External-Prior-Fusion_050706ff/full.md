# Neural Image Compression with Internal-External Prior Fusion and Alignment

Tiancheng Si

School of Artificial Intelligence

Xidian University

Xi’an 710126, China

tiancheng@stu.xidian.edu.cn

Wenlong Xu   
School of Artificial Intelligence   
Xidian University   
Xi’an 710126, China   
xuwl@stu.xidian.edu.cn   
Dahua Gao   
School of Artificial Intelligence   
Xidian University   
Xi’an 710126, China   
dhgao@xidian.edu.cn

Abstract—Existing context models for learned image compression rely primarily on internal priors from the current image, overlooking common patterns and typical textures shared across natural images. To address this limitation, we propose an Internal-External Prior Fusion-Alignment (IE-PFA) context model built upon quadtree spatial-channel joint modeling, where a learnable dictionary shared by the encoder and decoder serves as an external prior. Specifically, an RWKV-based Spatial-Channel Prior Fusion (RSCPF) module efficiently fuses internal and external priors along spatial and channel dimensions with linear complexity, while a training-only Self-Distilled Prior Alignment (SDPA) mechanism aligns them without inference overhead. Moreover, our model supports variable-rate coding over 64 quality levels within a single network and is accelerated by custom CUDA operators. Experiments on seven benchmarks demonstrate state-of-the-art rate-distortion performance, achieving an average BD-Rate saving of 21.65% over VTM-22.0-Intra, with only 29.9M parameters and the fastest coding speed among advanced neural codecs (61 ms encoding and 52 ms decoding). Code is available at https://github.com/stcnjtech/DCVC-X-Intra. Index Terms—Image Compression, Prior Alignment, Prior Fusion, Self-Distillation

## I. INTRODUCTION

Neural image compression (NIC), driven by end-to-end ratedistortion optimization, has advanced rapidly and achieved performance competitive with or even superior to traditional codecs such as VVC [1]–[5]. In NIC, the entropy model estimates the probability distribution of quantized latents and plays a crucial role in determining the bitrate of the compressed bitstream. To improve entropy estimation, existing methods have developed various context models, including hyperprior-based models [2], spatial [3], [4] or channel-wise [5] autoregressive models, and joint spatial-channel context models [6]. These approaches have sought a better balance between compression performance and coding efficiency.

However, existing context models primarily rely on internal priors from the current image, such as decoded latents and hyperprior features, while overlooking the common patterns and typical textures shared across natural images [7]. Such dataset-level external priors contain rich global statistics and can complement distribution estimation when single-image information is insufficient [8], especially at low bitrates [9]–[11] or in complex texture regions [7]. Exploiting these priors, however, raises two challenges: (i) internal and external priors differ substantially in representation, requiring efficient deep fusion without heavy computational overhead; and (ii) the distribution shift between them may even degrade entropy estimation if not properly constrained.

![](images/a41c73abf9536e788f691c3e78fcfdb41d09889819e05eed18c591838cd9dbdf.jpg)  
Fig. 1. Overall architecture of our proposed model. The core is the Internal-External Prior Fusion-Alignment (IE-PFA) context model, which estimates the latent distribution by jointly exploiting internal priors and external priors provided by a learnable dictionary capturing common patterns of natural images.

To this end, we propose the Internal-External Prior Fusion-Alignment (IE-PFA) context model, which adopts a learnable dictionary shared between the encoder and decoder to provide external priors without transmitting any additional bits. Within a quadtree spatial-channel modeling framework, we design the RWKV-based Spatial-Channel Prior Fusion (RSCPF) module to efficiently fuse internal and external priors along both spatial and channel dimensions with linear complexity. To mitigate the distribution shift between the two priors, we further introduce a training-only Self-Distilled Prior Alignment (SDPA) mechanism, which uses mini-batch similarity structures as soft supervision to align the external prior with the internal one at zero inference cost. In addition, we use a threestage progressive training strategy to further promote effective fusion and alignment of the two priors, and introduce a shared rate-adjustment module bank to enable smooth variable-rate adjustment within a single model. Our main contributions are:

• We propose IE-PFA, which introduces external priors via a shared learnable dictionary and jointly estimates entropy parameters with internal priors, exploiting crossimage statistics without extra bits.

• We design the linear-complexity RSCPF fusion module and the training-only SDPA alignment mechanism, complemented by a three-stage progressive training strategy.

• Experiments show that our method, with only 29.9M parameters, achieves an average BD-Rate saving of 21.65% over VTM-22.0-Intra on seven benchmarks, outperforming advanced NIC methods while offering the fastest coding speed and smooth single-model variablerate adaptation.

![](images/cc2411099a76a08c79f6804f8aade45f22d03a20c01320e59c9d52d2ac19b8b2.jpg)  
Fig. 2. (a) The proposed IE-PFA context model with four branches based on quadtree spatial-channel joint modeling. Each branch e<sub>i</sub> mainly consists of (b) the RWKV-based Spatial-Channel Prior Fusion (RSCPF) module and (c) the Self-Distilled Prior Alignment (SDPA) mechanism.

## II. PROPOSED METHODS

## A. Overview

The overall architecture of the proposed codec, named DCVC-X-Intra, is illustrated in Fig. 1. The analysis transform g maps an input image x to the latent $y = g _ { a } ( x ) \in$ $\mathbb { R } ^ { C \times H \times W }$ , which is quantized as $\hat { y } = Q ( y )$ under estimated Gaussian parameters $( \mu , \sigma )$ and then losslessly entropy-coded; the synthesis transform $g _ { s }$ reconstructs $\hat { x } = g _ { s } ( \hat { y } )$ at the ${ \mathrm { d e } } -$ coder. For entropy modeling, the hyper-analysis transform $h _ { a }$ extracts side information $z = h _ { a } ( y )$ , coded with a factorized prior after quantization, and the hyper-synthesis transform $h _ { s }$ decodes zˆ into the hyperprior feature $\bar { \mathcal { F } _ { z } } \in \mathbb { R } ^ { C \times H \times W }$ and the spatial-wise quantization steps $Q _ { e n c } ^ { s p } , Q _ { d e c } ^ { s p } \in \mathbb { R } ^ { 1 \times H \times W }$

The proposed IE-PFA is the core of the entropy model. Based on quadtree spatial-channel joint modeling, it partitions the latent into four branches coded sequentially, and estimates entropy parameters by jointly leveraging two types of priors: the internal prior $\mathcal { F } _ { I , i }$ , formed by concatenating the decoded groups $\hat { y } _ { < i } = \{ \hat { y } _ { 0 } , \cdot \cdot \cdot , \hat { y } _ { i - 1 } \}$ with $\mathcal { F } _ { z } ,$ models intra-image dependencies; the external prior $\mathcal { F } _ { E } \in \mathbb { R } ^ { L \times C _ { d } }$ is a learnable dictionary shared by the encoder and decoder, capturing common patterns and typical textures of natural image datasets. The structural details of IE-PFA are presented in Sec. II-B. Moreover, unlike previous end-to-end schemes, we adopt a three-stage progressive training strategy (Sec. II-C) to fully fuse and align the two priors.

## B. Internal-External Prior Fusion-Alignment Context Model

Existing context models typically rely only on internal priors and fail to exploit dataset-level regularities [12]–[15]. IE-PFA (Fig. 2) introduces the external prior via a shared learnable dictionary: during training, it is aligned to the internal prior through a self-distillation loss, gradually learning the typical structures of natural images; at inference it in turn guides and refines the internal prior, providing a global complement when single-image information is insufficient. As shown in Fig. 2(a), IE-PFA first multiplies $\hat { y }$ by $Q _ { e n c } ^ { s p }$ to modulate its spatial magnitude, then codes the four quadtree groups sequentially. The i-th branch $e _ { i }$ takes $\mathcal { F } _ { I , i } \mathrm { ~ } ( \hat { y } _ { < i }$ is allzero when $i = 0 )$ and $\mathcal { F } _ { E }$ as input to estimate the entropy parameters of ${ \hat { y } } _ { i } { \mathrm { : } }$

$$
\mu _ { i } , \sigma _ { i } = e _ { i } \left( \hat { y } _ { < i } , \mathcal { F } _ { z } , \mathcal { F } _ { E } \right) .\tag{1}
$$

Each branch comprises the RSCPF module, which interactively fuses $\mathcal { F } _ { I , i }$ and $\mathcal { F } _ { E }$ into the fused features $\Phi _ { I , i }$ and $\Phi _ { E , i }$ , and the SDPA mechanism, which aligns the external fused feature to the internal one during training. The fused feature $\Phi _ { I , i }$ is then merged with $\hat { y } _ { < i }$ and $\mathcal { F } _ { z }$ and fed into the entropy parameter module to predict more accurate µ<sub>i</sub> and $\sigma _ { i }$ Finally, all decoded groups are concatenated and multiplied by $\boldsymbol { Q } _ { d e c } ^ { s p }$ to obtain the decoded latent.

RSCPF Module. As shown in Fig. 2(b), RSCPF, which is shared across branches to greatly reduce parameters, contains spatial-aware aggregation, spatial mixing, and channel mixing. The spatial-aware aggregation employs a lightweight spatial attention to dynamically weight each position of the internal prior, $X _ { s a , i } = S A ( \mathcal { F } _ { I , i } ) \odot \mathcal { F } _ { I , i } .$ , capturing its typical textures. Inspired by linear-complexity attention [16]–[21], the spatial mixing introduces the RWKV-based [18] bidirectional WKV (BiWKV) attention to efficiently capture long-range dependencies and fuse the spatial features of the two priors: $X _ { s a , i }$ first passes through an Omni-Shift [21] operation to capture 2D local context (reparameterized from multiple training-time kernels into a single kernel at inference), which is concatenated with $\mathcal { F } _ { E }$ and projected via LayerNorm into the receptance $R _ { s , i } .$ , key $K _ { s , i } ,$ and value $V _ { s , i }$ . BiWKV then computes the global attention:

$$
w k v _ { t } = \frac { \sum _ { j = 1 , j \neq t } ^ { T } e ^ { - ( \lvert t - j \rvert - 1 ) / T \cdot w + k _ { j } } v _ { j } + e ^ { u + k _ { t } } v _ { t } } { \sum _ { j = 1 , j \neq t } ^ { T } e ^ { - ( \lvert t - j \rvert - 1 ) / T \cdot w + k _ { j } } + e ^ { u + k _ { t } } } ,\tag{2}
$$

where $k _ { j }$ and $v _ { j }$ denote the key and value at position $j ;$ wkv is modulated by the sigmoid gate $\sigma ( R _ { s , i } )$ and then projected to the output $O _ { s , i } .$ The channel mixing analogously fuses channel-wise features. Its input $O _ { s , i }$ is split into $\Psi _ { I , i }$ and $\Psi _ { E , i }$ . After applying Omni-Shift to $\Psi _ { I , i } .$ the two branches are merged and projected to produce $\mathit { R _ { c , i } }$ and $K _ { c , i } ,$ while $V _ { c , i }$ is obtained via squared ReLU. Gating $V _ { c , i }$ by $\sigma ( R _ { c , i } )$ then yields $O _ { c , i }$ , which is split into fused priors $\Phi _ { I , i } \in \overline { { \mathbb { R } ^ { C \times H \times W } } }$ and $\Phi _ { E , i } \in \mathbb { R } ^ { C \times L }$

SDPA Mechanism. Motivated by the observation in knowledge distillation and multimodal alignment [22], [23] that similar semantics exhibit similar activation patterns, SDPA aligns the two priors via mini-batch similarity structures, as shown in Fig. 2(c). For a training mini-batch of size B, we compute the L2-normalized pairwise similarities:

$$
\left\{ \begin{array} { l l } { G _ { I , i } = \mathrm { L 2 N o r m } \left( \Phi _ { I , i } \cdot \Phi _ { I , i } ^ { T } \right) , } \\ { G _ { E , i } = \mathrm { L 2 N o r m } \left( \Phi _ { E , i } \cdot \Phi _ { E , i } ^ { T } \right) , } \end{array} \right.\tag{3}
$$

![](images/ee9d2cdf9645fcc63363e9d06e9f07538232656c7cfe24065b7f2f5a3f2a7d96.jpg)

![](images/19db1b8c918e52febe4f99fe2dae0474f68637de140009845469581ce3fb7d9d.jpg)

![](images/7fefb07fa7409db48a652b6bb607a6a193f8e3d741b4ea088cad79e8af50479e.jpg)  
Fig. 3. Rate-distortion curves of different methods on the Kodak, Tecnick, and HEVC C datasets.

where $G _ { I , i }$ and $G _ { E , i }$ are $B \times B$ matrices whose entries measure the activation similarity of image pairs. The alignment loss sums the discrepancies over the four branches:

$$
\mathcal { L } _ { S D P A } = \frac { 1 } { B ^ { 2 } } \sum _ { i = 0 } ^ { 3 } \left. \boldsymbol { G } _ { I , i } - \boldsymbol { G } _ { E , i } \right. _ { F } ^ { 2 } ,\tag{4}
$$

where $\| \cdot \| _ { F }$ is the Frobenius norm. Requiring no extra teacher network, the mini-batch similarity structure serves as soft supervision that aligns the two priors at the featuredistribution level and mitigates their distribution shift. The SDPA mechanism operates only during training and adds no inference overhead.

## C. Overall Loss

We adopt a three-stage progressive training strategy with the loss

$$
\mathcal { L } = \lambda \left( \Vert x - \hat { x } \Vert ^ { 2 } + \alpha \mathcal { L } _ { S D P A } \right) + \mathcal { R } ( \hat { y } ) + \mathcal { R } ( \hat { z } ) ,\tag{5}
$$

where $\mathcal { R } ( \cdot )$ denotes the bitrate and λ controls the ratedistortion trade-off. Stage 1 optimizes only the internal prior with the standard rate-distortion loss $( \alpha = 0 )$ . Stage 2 introduces the external prior: $g _ { a }$ and $h _ { a }$ are frozen, the remaining modules are trained, and the alignment term is added $( \alpha = 0 . 2 )$ so that the external prior is optimized toward the internal one. Stage 3 unfreezes all modules and fine-tunes end-to-end with the same joint loss as Stage 2, further strengthening prior fusion and alignment.

## D. Other Optimizations

To support smooth variable-rate adjustment in a single model rather than training one model per bitrate [6], we introduce a shared rate-adjustment module bank (see Fig. 1, left), which learns a series of factorized modules to model diverse distributions for different quantization levels $q p \in [ 0 , 6 3 ]$ According to $q p ,$ the channel-wise quantization parameters $q _ { e } ^ { c h } , q _ { d } ^ { c h }$ drawn from the bank modulate the channel magnitude of the latent, while the hyperprior module generates the spatialwise parameters $Q _ { e n c } ^ { s p } , Q _ { d e c } ^ { s p }$ for spatially adaptive granularity. During training, $q p$ is randomly sampled from the 64 levels, and the corresponding rate-distortion weight λ is log-linearly interpolated within a predefined interval $\mathbf { \bar { \rho } } [ \lambda _ { \operatorname* { m i n } } , \lambda _ { \operatorname* { m a x } } ]$ . In addition, we develop fused CUDA operators to reduce GPU memory I/O overhead and accelerate inference.

(a) Original  
(b) ELIC  
(c) DCVC-FM-Intra  
(d) Ours  
![](images/df64fb7972a29640eccd4c8819af9b04b55bacc041eaba813790b2190faa9efd.jpg)  
Fig. 4. Visual comparison on Kodak images at low bitrates.

## III. EXPERIMENTAL RESULTS

## A. Experimental Settings

Datasets. Following previous works [12], [13], [15], we train on the first 400K images of the ImageNet dataset [24] (shorter edge $\ge 2 5 6$ pixels). Evaluation is conducted on Kodak [25], CLIC Professional Validation [26], Tecnick [27], and the first 96 frames of each HEVC test sequence [28] (Classes B–E). Baselines and Metrics. We compare with VVC [29] (VTM-22.0-Intra: all-intra configuration, BT.709) and advanced neural codecs, including ELIC [12], STF [30], TCM [13], DCVC-DC-Intra [31], WeConvene [14], DCVC-FM-Intra [6], and AuxT [15], using BD-Rate [32] computed from RGB PSNR– bpp curves.

Training Details. We train on 4 A100 GPUs, with $2 5 6 \times 2 5 6$ random crops, a batch size of 16 per GPU, and MSE as the distortion loss. A single model covering 64 quality levels with $\lambda \in [ 0 . 0 0 0 6 7 , 0 . 1 5 6 ]$ enables variable-rate coding. We adopt a three-stage training strategy: 100 epochs with a learning rate of $1 \times 1 0 ^ { - 4 }$ , followed by two stages of 50 epochs each with $1 \times 1 0 ^ { - 5 }$

## B. Rate-Distortion Performance and Complexity

As shown in Table I, among the compared methods, our method achieves the lowest BD-Rate on five of the seven datasets and the lowest average BD-Rate (−21.65%). The rate-distortion curves on HEVC C in Fig. 3 show that our method outperforms the others at all tested bitrates, with a BD-Rate margin of 3.22 percentage points over the secondbest DCVC-FM-Intra, suggesting that our prior modeling particularly benefits complex content. Fig. 4 compares the reconstructions of two Kodak images at low bitrates. At comparable bitrates (within 0.002 bpp), DCVC-X-Intra achieves the highest PSNR on both images: 29.60 dB versus 29.01 dB (ELIC) and 29.39 dB (DCVC-FM-Intra) on kodim07, and 28.31 dB versus 27.98 dB and 28.15 dB on kodim21. On the window shutter of kodim07, ELIC and DCVC-FM-Intra blur part of the thin slats, whereas DCVC-X-Intra keeps them sharp and clearly separated. On the rocks of kodim21, ELIC produces a smeared and discolored patch, and DCVC-FM-Intra over-smooths the surface so that the vertical cracks are largely lost, while DCVC-X-Intra retains these cracks.

TABLE I  
BD-RATE (%) RELATIVE TO VTM-22.0-INTRA ON SEVEN DATASETS. AVG IS THEIR AVERAGE. GFLOPS ARE COMPUTED AT THE KODAK RESOLUTION (768 × 512), AND ENC./DEC. TIME (MS), INCLUDING ENTROPY CODING, IS AVERAGED OVER THE 24 KODAK IMAGES ON AN NVIDIA RTX 4090 GPU; † DENOTES OUR MODEL WITH CUSTOM CUDA OPERATORS. THE BEST AND SECOND-BEST RESULTS ARE IN RED BOLD AND BLUE, RESPECTIVELY.
<table><tr><td rowspan="2">Method</td><td colspan="8">BD-Rate over VTM-22.0-Intra (%)</td><td rowspan="2">GFLOPs</td><td rowspan="2">Params</td><td>Enc. (ms)</td><td>Dec.</td></tr><tr><td>Kodak</td><td>CLIC</td><td>Tecnick</td><td>HEVC B</td><td>HEVC C</td><td>HEVC D</td><td>HEVC E</td><td>AVG</td><td></td><td>(ms)</td></tr><tr><td>VTM-22.0-Intra</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td></td><td></td><td>一</td><td></td></tr><tr><td>ELIC [12]</td><td>-8.64</td><td>-13.79</td><td>-17.33</td><td>-16.17</td><td>-9.94</td><td>-8.07</td><td>-16.71</td><td>-12.95</td><td>371</td><td>41.9M</td><td>91</td><td>119</td></tr><tr><td>STF [30]</td><td>-8.22</td><td>-10.01</td><td>-11.00</td><td>-13.97</td><td>-6.67</td><td>-2.12</td><td>-14.96</td><td>-9.56</td><td>304</td><td>99.8M</td><td>102</td><td>131</td></tr><tr><td>TCM [13]</td><td>-11.70</td><td>-17.00</td><td>-17.03</td><td>-16.70</td><td>-8.06</td><td>-5.67</td><td>-15.26</td><td>-13.06</td><td>701</td><td>75.9M</td><td>142</td><td>151</td></tr><tr><td>DCVC-DC-Intra [31]</td><td>-14.87</td><td>-18.75</td><td>-21.41</td><td>-23.54</td><td>-11.38</td><td>-14.77</td><td>-29.77</td><td>-19.21</td><td>278</td><td>31.0M</td><td>78</td><td>66</td></tr><tr><td>WeConvene [14]</td><td>-13.37</td><td>-16.77</td><td>-17.73</td><td>-17.42</td><td>-10.52</td><td>-8.29</td><td>-16.79</td><td>-14.41</td><td>905</td><td>105.5M</td><td>275</td><td>271</td></tr><tr><td>DCVC-FM-Intra [6]</td><td>-15.02</td><td>-17.60</td><td>-19.94</td><td>-19.39</td><td>-15.79</td><td>-18.67</td><td>-25.91</td><td>-18.90</td><td>253</td><td>26.6M</td><td>71</td><td>62</td></tr><tr><td>AuxT [15]</td><td>-14.39</td><td> $- 1 7 . 7 6$ </td><td>-18.83</td><td>-16.72</td><td>-12.52</td><td>-11.21</td><td>-19.17</td><td>-15.80</td><td>216</td><td>46.2M</td><td>170</td><td>166</td></tr><tr><td>DCVC-X-Intra(Ours)</td><td>-17.33</td><td>-19.21</td><td>-21.97</td><td>-23.42</td><td>-19.01</td><td>-21.98</td><td>-28.60</td><td>-21.65</td><td>297</td><td>29.9M</td><td> $7 4 / 6 1 ^ { \dag }$ </td><td>64/52†</td></tr></table>

TABLE II

ABLATION STUDY ON KODAK: COMPONENTS ADDED CUMULATIVELY AND SDPA ON DCAE [7] (LEFT), AND DICTIONARY SIZE L (RIGHT). BD-RATE (%) IS RELATIVE TO VTM-22.0-INTRA; LATENCY (MS) IS THE TOTAL ENCODING AND DECODING TIME WITHOUT CUSTOM CUDA OPERATORS.
<table><tr><td>Method</td><td>BD-Rate</td></tr><tr><td>Baseline</td><td>-13.43</td></tr><tr><td>+ RSCPF</td><td>-15.84</td></tr><tr><td>+ Dictionary</td><td>-15.97</td></tr><tr><td>+ SDPA</td><td>-17.33</td></tr><tr><td>DCAE [7]</td><td>-21.24</td></tr><tr><td>+ SDPA</td><td>-23.67</td></tr></table>

<table><tr><td>L</td><td>BD-Rate</td><td>Latency</td></tr><tr><td>0</td><td>-15.84</td><td>124</td></tr><tr><td>32</td><td>-15.82</td><td>135</td></tr><tr><td>64</td><td> $- 1 6 . 4 7$ </td><td>136</td></tr><tr><td>128</td><td>-17.33</td><td>138</td></tr><tr><td>256</td><td>-17.27</td><td>138</td></tr><tr><td>512</td><td>-17.33</td><td>142</td></tr></table>

Coding times in Table I are averaged over all 24 Kodak images. Without custom CUDA operators, our model (74 ms/64 ms) is the second fastest, behind only DCVC-FM-Intra (71 ms/62 ms). With custom CUDA operators, the encoding and decoding times are further reduced to 61 ms and 52 ms, i.e., by 17.6% and 18.8%, respectively, making our method the fastest. Moreover, with only 29.9M parameters, fewer than most competing methods, our model achieves a favorable performance-complexity trade-off.

## C. Ablation Study

Effectiveness of RSCPF and SDPA. We build a baseline by removing all proposed modules and mechanisms (RSCPF, the external dictionary, and SDPA) and then add them back progressively until reaching the full DCVC-X-Intra. As shown in Table II (left), adding RSCPF improves the BD-Rate from −13.43% to −15.84%, and adding the dictionary without alignment brings only a marginal gain $( - 1 5 . 9 7 \% ) ;$ the external prior pays off once SDPA aligns it $( - 1 7 . 3 3 \% )$ . Fig. 5 further visualizes the priors: the fused prior $\Phi _ { I , i }$ responds more clearly to the main structures and textures, and the aligned prior $\Phi _ { I , i } ^ { \prime }$ focuses more on key structural and complex-texture regions, which qualitatively supports the joint effectiveness of RSCPF and SDPA for accurate entropy parameter estimation. Effect of the Dictionary Size. We further train models with dictionary sizes $L \in \{ 0 , 3 2 , 6 4 , 1 2 8 , 2 5 6 , 5 1 2 \}$ and report the BD-Rate and latency in Table II (right). Despite a slight drop at $L = 3 2 .$ , the performance generally improves with L and then saturates: the best BD-Rate is reached at $L = 1 2 8$ , whereas $L = 2 5 6$ brings no gain at the same latency and $L = 5 1 2$ only matches it with a higher latency (142 ms). Thus, L = 128 achieves a good balance between compression performance and computational cost.

![](images/ee0e2c84e33178617f6db6c6445704099ba61d1e6b63f821fce31e0db6c967b3.jpg)  
Fig. 5. Visualization of priors on Kodak. From left to right: raw image x, original internal prior $\mathcal { F } _ { I , i } ,$ unaligned fused prior $\Phi _ { I , i } .$ , and SDPA-aligned fused prior $\Phi _ { I , i } ^ { \prime }$

Generalization of SDPA to DCAE. DCAE [7] also builds its entropy model on a learned dictionary, but its parameter count (119.2M) and encoding latency exceed those of all methods in Table I, so we evaluate it separately. Keeping its architecture unchanged, we add SDPA only as a training loss and retrain it with $\lambda \in \{ 0 . 0 5 , 0 . 0 2 5 , 0 . 0 1 3 , 0 . 0 0 6 7 \}$ , improving its BD-Rate from −21.24% to −23.67% (Table II, left).

## IV. CONCLUSION AND DISCUSSION

In this paper, we propose a neural image compression model with an Internal-External Prior Fusion-Alignment (IE-PFA) context model, where an RWKV-based Spatial-Channel Prior Fusion (RSCPF) module fuses internal priors with a learnable external dictionary, and a Self-Distilled Prior Alignment (SDPA) mechanism aligns the two priors without extra inference cost. Together with a variable-rate design and custom CUDA operators, our model achieves state-of-the-art ratedistortion performance on seven benchmarks with only 29.9M parameters and the fastest coding speed among the compared methods. In future work, we will extend the external prior modeling to perceptual-oriented and video compression.

[1] J. Balle, V. Laparra, and E. P. Simoncelli, “End-to-end optimized image´ compression,” in International Conference on Learning Representations, 2017.

[2] J. Balle, D. Minnen, S. Singh, S. J. Hwang, and N. Johnston, “Variational´ image compression with a scale hyperprior,” in International Conference on Learning Representations, 2018.

[3] D. Minnen, J. Balle, and G. D. Toderici, “Joint autoregressive and´ hierarchical priors for learned image compression,” in Advances in Neural Information Processing Systems, vol. 31, 2018.

[4] Z. Cheng, H. Sun, M. Takeuchi, and J. Katto, “Learned image compression with discretized Gaussian mixture likelihoods and attention modules,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2020, pp. 7939–7948.

[5] D. Minnen and S. Singh, “Channel-wise autoregressive entropy models for learned image compression,” in Proceedings of the IEEE International Conference on Image Processing, 2020, pp. 3339–3343.

[6] J. Li, B. Li, and Y. Lu, “Neural video compression with feature modulation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 26 099–26 108.

[7] J. Lu, L. Zhang, X. Zhou, M. Li, W. Li, and S. Gu, “Learned image compression with dictionary-based entropy model,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025, pp. 12 850–12 859.

[8] A. van den Oord, O. Vinyals, and K. Kavukcuoglu, “Neural discrete representation learning,” in Advances in Neural Information Processing Systems, vol. 30, 2017.

[9] P. Esser, R. Rombach, and B. Ommer, “Taming transformers for highresolution image synthesis,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021, pp. 12 873– 12 883.

[10] Z. Jia, J. Li, B. Li, H. Li, and Y. Lu, “Generative latent coding for ultra-low bitrate image compression,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 26 088–26 098.

[11] L. Qi, Z. Jia, J. Li, B. Li, H. Li, and Y. Lu, “Generative latent coding for ultra-low bitrate image and video compression,” IEEE Transactions on Circuits and Systems for Video Technology, vol. 35, no. 10, pp. 10 500– 10 515, 2025.

[12] D. He, Z. Yang, W. Peng, R. Ma, H. Qin, and Y. Wang, “ELIC: Efficient learned image compression with unevenly grouped spacechannel contextual adaptive coding,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 5718–5727.

[13] J. Liu, H. Sun, and J. Katto, “Learned image compression with mixed Transformer-CNN architectures,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 14 388– 14 397.

[14] H. Fu, J. Liang, Z. Fang, J. Han, F. Liang, and G. Zhang, “WeConvene: Learned image compression with wavelet-domain convolution and entropy model,” in Proceedings of the European Conference on Computer Vision, 2024, pp. 37–53.

[15] H. Li et al., “On disentangled training for nonlinear transform in learned image compression,” in International Conference on Learning Representations, 2025.

[16] A. Gu and T. Dao, “Mamba: Linear-time sequence modeling with selective state spaces,” in Conference on Language Modeling, 2024.

[17] T. Dao and A. Gu, “Transformers are SSMs: Generalized models and efficient algorithms through structured state space duality,” in Proceedings of the 41st International Conference on Machine Learning (ICML), 2024, pp. 10 041–10 071.

[18] B. Peng et al., “RWKV: Reinventing RNNs for the transformer era,” in Findings of the Association for Computational Linguistics: EMNLP 2023, 2023, pp. 14 048–14 077.

[19] S. Zhai et al., “An attention free transformer,” arXiv:2105.14103, 2021.

[20] Y. Duan et al., “Vision-RWKV: Efficient and scalable visual perception with RWKV-like architectures,” in International Conference on Learning Representations, 2025.

[21] Z. Yang, J. Li, H. Zhang, D. Zhao, B. Wei, and Y. Xu, “Restore-RWKV: Efficient and effective medical image restoration with RWKV,” IEEE Journal of Biomedical and Health Informatics, vol. 30, no. 1, pp. 513– 526, 2026.

[22] F. Tung and G. Mori, “Similarity-preserving knowledge distillation,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2019, pp. 1365–1374.

[23] A. Radford et al., “Learning transferable visual models from natural language supervision,” in Proceedings of the 38th International Conference on Machine Learning (ICML), 2021, pp. 8748–8763.

[24] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei, “ImageNet: A large-scale hierarchical image database,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2009, pp. 248– 255.

[25] R. Franzen, “Kodak lossless true color image suite,” 1999.

[26] G. Toderici et al., “Workshop and challenge on learned image compression (CLIC2020),” 2020. [Online]. Available: https://archive. compression.cc

[27] N. Asuni and A. Giachetti, “TESTIMAGES: A large-scale archive for testing visual devices and basic image processing algorithms,” in Smart Tools and Apps for Graphics (STAG), 2014, pp. 63–70.

[28] G. J. Sullivan, J.-R. Ohm, W.-J. Han, and T. Wiegand, “Overview of the High Efficiency Video Coding (HEVC) standard,” IEEE Transactions on Circuits and Systems for Video Technology, vol. 22, no. 12, pp. 1649– 1668, 2012.

[29] B. Bross et al., “Overview of the Versatile Video Coding (VVC) standard and its applications,” IEEE Transactions on Circuits and Systems for Video Technology, vol. 31, no. 10, pp. 3736–3764, 2021.

[30] R. Zou, C. Song, and Z. Zhang, “The devil is in the details: Windowbased attention for image compression,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 17 492–17 501.

[31] J. Li, B. Li, and Y. Lu, “Neural video compression with diverse contexts,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 22 616–22 626.

[32] G. Bjøntegaard, “Calculation of average PSNR differences between RDcurves,” ITU SG16 Doc. VCEG-M33, 2001.