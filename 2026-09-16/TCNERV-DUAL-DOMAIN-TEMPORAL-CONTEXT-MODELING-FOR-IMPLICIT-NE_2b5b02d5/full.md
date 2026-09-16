# TCNERV: DUAL-DOMAIN TEMPORAL CONTEXT MODELING FOR IMPLICIT NEURAL VIDEO COMPRESSION

Xuezhi Xiang<sup>1,2,∗</sup> Yixin Zhao<sup>1</sup> Heqi Xiang<sup>3</sup> Jiayao Liu<sup>1</sup> Shanjun Zhang<sup>4</sup>

<sup>1</sup>Information and Communication Engineering, Harbin Engineering University, Harbin, 150001, China

<sup>2</sup>Key Laboratory of Advanced Marine Communication and Information Technology, Harbin, 150001, China

<sup>3</sup>Department of Computer Science, University of Toronto, Toronto, ON M5S 2E4, Canada

<sup>4</sup>The Department of Computer Science, Kanagawa University, Kanagawa, 221-8686, Japan

xiangxuezhi@hrbeu.edu.cn; yixinzhao@hrbeu.edu.cn

claire.xiang@mail.utoronto.ca; liujiayao@hrbeu.edu.cn; chiyoz01@kanagawa-u.ac.jp

## ABSTRACT

Video compression aims to minimize reconstruction distortion under a constrained bit rate. Existing video implicit neural representations (INRs) often decode frames independently, leaving intermediate features unconditioned on previous reconstructions and content embeddings without explicit temporal prediction. We propose TCNeRV, which exploits reconstructed context in both feature and embedding domains. Its multi-scale temporal-context fusion (MTCF) module injects gated historical features at multiple decoder scales, while temporal embedding-residual coding (TERC) predicts each content embedding and codes only its residual. With approximately 3M parameters, TCNeRV achieves an average PSNR of 36.08 dB on the UVG dataset, outperforming HNeRV-Boost by 2.20 dB. It reduces BD-rate by 22.06%, 66.73%, and 29.85% relative to HM, DCVC, and HiNeRV, respectively, demonstrating competitive rate-distortion performance with limited model capacity.

Index Terms— video compression, implicit neural representation, temporal context modeling, feature fusion, residual coding

## 1. INTRODUCTION

Video compression minimizes reconstruction distortion under a constrained bit rate. Conventional standards, including HEVC [1] and VVC [2], improve coding efficiency through block partitioning, prediction, transform quantization, and entropy coding. Learning-based codecs jointly optimize major coding components. DVC [3] introduced end-to-end motion and residual coding, DCVC [4] adopted conditional coding with temporal features, and DCVC-DC [5] employed diverse spatiotemporal contexts. However, their multiple motion, context, residual, and entropy-modeling sub-networks incur considerable complexity.

Implicit neural representations (INRs) instead parameterize signals using coordinate- or index-conditioned networks. Periodic activations [6] improve high-frequency fitting, while COIN [7] enables content-specific image compression. For video, NeRV [8] maps frame indices directly to frames, E-NeRV [9] disentangles spatial-temporal contexts, and HNeRV [10] introduces content-adaptive embeddings. HiNeRV [11] uses hierarchical multi-scale encoding, and HNeRV-Boost [12] improves feature-to-frame alignment through conditional decoding and temporal-aware modulation. Nevertheless, these methods largely represent frames independently. Temporal INR methods include FFNeRV [13], which propagates information using optical flow, and DNeRV [14], which separates spatial content and frame differences. Such mechanisms require additional branches, while unselective temporal fusion may propagate mismatched features under fast motion, occlusion, or scene changes.

INR compression further requires quantization and entropy coding. Early work compressed video-specific networks [15], while entropy-constrained representations [16] incorporated rate estimation. C3 [17] targets efficient decoding, NVRC [18] jointly optimizes representations, quantization, and entropy models, and GIViC [19] exploits generative priors and long-range dependencies. However, hybrid video INRs still encode frame-wise embeddings independently.

Existing approaches therefore have two limitations: intermediate decoding features cannot reuse structures and textures from previous reconstructions, and independently coded embeddings cannot remove temporally predictable information before quantization.

Inspired by the gated context selection of BiECVC [20], we propose multi-scale temporal-context fusion (MTCF), which aligns features from the previous reconstruction with two decoder stages and integrates them through channel-wise gated residual connections. Inspired by the autoregressive representation of NIRVANA [21], we further propose temporal embedding-residual coding (TERC), which predicts the current embedding from reconstructed context and encodes only its residual. MTCF reduces distortion in the decodingfeature domain, while TERC exploits redundancy in the embedding domain. Together, they constitute TCNeRV.

![](images/b566cae0813b9e1ae9b4dfe7881379d365414164961dd40e97ebca5e33a04302.jpg)  
Fig. 1. Overall architecture of TCNeRV. MTCF enhances current-frame structure and texture reconstruction through multi scale feature extraction and channel-wise gating; TERC predicts the current content embedding from historical reconstruction context and establishes an explicit temporal predictive-coding relationship among framewise content embeddings.

Our contributions are summarized as follows:

1. We propose TCNeRV, which constructs causal temporal context from the previous reconstructed frame and exploits inter-frame correlation in both decoding features and content embeddings.

2. We propose MTCF, which performs gated residual fusion at two decoder scales to improve structure and texture reconstruction without explicit motion estimation.

3. We propose TERC, which replaces full-embedding coding with frame-level embedding-prediction-residual coding before quantization.

4. Experiments on the UVG dataset validate TCNeRV. For PSNR-oriented compression, it reduces BD-rate by 22.06%, 66.73%, and 29.85% relative to HM, DCVC, and HiNeRV, respectively.

## 2. METHOD

## 2.1. Method Overview

TCNeRV builds on HNeRV-Boost [12] and represents a video using quantized content embeddings, temporal embeddings, and conditional-decoder parameters. As shown in Fig. 1, a shared ConvNeXt encoder [22] extracts the current content embedding. TERC predicts this embedding from the previous reconstruction and encodes only the prediction residual, thereby exploiting temporal redundancy. The recovered content and temporal embeddings are fed into a six-stage conditional decoder, where Stage 1 expands the channels and

Stages 2–6 progressively restore spatial resolution. At Stages 3 and 4, MTCF aligns the historical context with the current decoding features and performs gated fusion. The first frame uses zero context, while subsequent frames use the preceding reconstruction, forming a causal closed-loop process shared by the encoder and decoder.

## 2.2. Multi-Scale Temporal-Context Fusion (MTCF)

Adjacent frames share structures and textures, but historical cues may become unreliable under occlusion, fast motion, or scene changes. MTCF therefore extracts context at two decoder scales and integrates it through channel-wise gates.

As shown in Fig. 2, MTCF operates at Stages 3 and 4 with resolutions of 135 × 240 and $2 7 0 \times 4 8 0 .$ , respectively. Let $\mathbf { f } _ { i } ^ { \mathrm { d e c } }$ denote the feature generated by the corresponding SNeRV and TAT blocks [12]. The previous reconstruction $\mathbf { c } _ { t }$ is resized and transformed into a scale-aligned context feature:

$$
\mathbf { f } _ { i } ^ { \mathrm { c t x } } = \Phi _ { i } [ \mathrm { R e s i z e } ( \mathbf { c } _ { t } ; H _ { i } , W _ { i } ) ] ,\tag{1}
$$

where $( H _ { 3 } , W _ { 3 } ) = ( 1 3 5 , 2 4 0 )$ and $( H _ { 4 } , W _ { 4 } ) = ( 2 7 0 , 4 8 0 )$ Φ<sub>i</sub> comprises a $3 \times 3$ convolution, a GELU activation, and a $1 \times 1$ convolution, mapping the three input channels through max $\left( C _ { i } , 3 2 \right)$ channels to $C _ { i }$ channels. Learnable gates then control context fusion:

$$
\begin{array} { r } { \mathbf { a } _ { i } = \mathrm { s i g m o i d } ( \mathbf { g } _ { i } ) , \qquad \mathbf { f } _ { i } ^ { \mathrm { o u t } } = \mathbf { f } _ { i } ^ { \mathrm { d e c } } + \mathbf { a } _ { i } \odot \mathbf { f } _ { i } ^ { \mathrm { c t x } } , } \end{array}\tag{2}
$$

where $i \in \{ 3 , 4 \}$ and ⊙ denotes channel-wise multiplication. Stage 3 supplies layout and motion cues; Stage 4 enhances edges and textures; and the gates suppress unreliable context without altering the backbone structure.

## 2.3. Temporal Embedding-Residual Coding (TERC)

Frame-wise embeddings determine reconstructed content and contribute to the bitstream. Existing INR codecs encode them independently, leaving temporal redundancy unexploited. TERC instead predicts each embedding from the historical reconstruction and encodes only its residual.

Table 1. Video representation results on the UVG dataset. Six methods are compared at three model scales in terms of parameters, computation, and PSNR.
<table><tr><td>Model</td><td>Size</td><td>MACs</td><td>Beauty</td><td>Bosph.</td><td>Honey.</td><td>Jockey</td><td>Ready.</td><td>Shake.</td><td>Yacht.</td><td>Avg.</td></tr><tr><td>NeRV [8]</td><td>3.31M</td><td>227G</td><td>32.83</td><td>32.20</td><td>38.15</td><td>30.30</td><td>23.62</td><td>33.24</td><td>26.43</td><td>30.97</td></tr><tr><td>E-NeRV [9]</td><td>3.29M</td><td>230G</td><td>33.13</td><td>33.38</td><td>38.87</td><td>30.61</td><td>24.53</td><td>34.26</td><td>26.87</td><td>31.66</td></tr><tr><td>HNeRV [10]</td><td>3.26M</td><td>175G</td><td>33.56</td><td>35.03</td><td>39.28</td><td>31.58</td><td>25.45</td><td>34.89</td><td>28.98</td><td>32.68</td></tr><tr><td>HiNeRV [11]</td><td>3.19M</td><td>181G</td><td>34.08</td><td>38.58</td><td>39.71</td><td>36.10</td><td>31.53</td><td>35.85</td><td>30.95</td><td>35.26</td></tr><tr><td>HNeRV-Boost [12]</td><td>3.05M</td><td>131G</td><td>33.80</td><td>36.12</td><td>39.64</td><td>34.29</td><td>28.13</td><td>35.88</td><td>29.32</td><td>33.88</td></tr><tr><td>TCNeRV</td><td>3.06M</td><td>131G</td><td>34.06</td><td>39.68</td><td>39.62</td><td>36.49</td><td>33.58</td><td>36.02</td><td>33.11</td><td>36.08</td></tr><tr><td>NeRV</td><td>6.53M</td><td>228G</td><td>33.67</td><td>34.83</td><td>39.00</td><td>33.34</td><td>26.03</td><td>34.39</td><td>28.23</td><td>32.78</td></tr><tr><td>E-NeRV</td><td>6.54M</td><td>245G</td><td>33.97</td><td>35.83</td><td>39.75</td><td>33.56</td><td>26.94</td><td>35.57</td><td>28.79</td><td>33.49</td></tr><tr><td>HNeRV</td><td>6.40M</td><td>349G</td><td>33.99</td><td>36.45</td><td>39.56</td><td>33.56</td><td>27.38</td><td>35.93</td><td>30.48</td><td>33.91</td></tr><tr><td>HiNeRV</td><td>6.49M</td><td>368G</td><td>34.33</td><td>40.37</td><td>39.81</td><td>37.93</td><td>34.54</td><td>37.04</td><td>32.94</td><td>36.71</td></tr><tr><td>HNeRV-Boost</td><td>5.01M</td><td>288G</td><td>34.14</td><td>37.87</td><td>39.74</td><td>35.84</td><td>30.36</td><td>36.71</td><td>30.77</td><td>35.06</td></tr><tr><td>TCNeRV</td><td>5.04M</td><td>289G</td><td>34.21</td><td>40.38</td><td>39.74</td><td>37.34</td><td>34.89</td><td>36.74</td><td>34.03</td><td>36.76</td></tr><tr><td>NeRV</td><td>13.01M</td><td>230G</td><td>34.15</td><td>36.96</td><td>39.55</td><td>35.80</td><td>28.68</td><td>35.90</td><td>30.39</td><td>34.49</td></tr><tr><td>E-NeRV</td><td>13.02M</td><td>285G</td><td>34.25</td><td>37.61</td><td>39.74</td><td>35.45</td><td>29.17</td><td>36.97</td><td>30.76</td><td>34.85</td></tr><tr><td>HNeRV</td><td>12.87M</td><td>701G</td><td>34.30</td><td>37.96</td><td>39.73</td><td>35.47</td><td>29.67</td><td>37.16</td><td>32.31</td><td>35.23</td></tr><tr><td>HiNeRV</td><td>12.82M</td><td>718G</td><td>34.66</td><td>41.83</td><td>39.95</td><td>39.01</td><td>37.32</td><td>38.19</td><td>35.20</td><td>38.02</td></tr><tr><td>HNeRV-Boost</td><td>10.03M</td><td>700G</td><td>34.42</td><td>39.75</td><td>39.83</td><td>37.57</td><td>33.12</td><td>37.85</td><td>32.90</td><td>36.49</td></tr><tr><td>TCNeRV</td><td>10.08M</td><td>702G</td><td>34.35</td><td>41.31</td><td>39.82</td><td>38.17</td><td>36.42</td><td>37.76</td><td>35.37</td><td>37.60</td></tr></table>

![](images/3128c813244bf468203a8447e1aae7f498ea5657cdc04f12d2341ab76fcfb22e.jpg)  
Fig. 2. Architecture of MTCF. It leverages multi-level temporal context to enhance reconstruction and suppress unreliable historical information.

As shown in Fig. 3, the shared ConvNeXt encoder [22] extracts the target embedding $\mathbf { y } _ { t } = \mathcal { E } ( \mathbf { x } _ { t } ; \boldsymbol { \phi } )$ , where $C _ { e } = 1 6$ and $( H _ { e } , W _ { e } ) = ( 9 , 1 6 )$ . The current embedding is predicted as

$$
\mathbf { y } _ { t } ^ { \mathrm { p r e d } } = \left\{ \begin{array} { l l } { \mathbf { 0 } , } & { t = 0 , } \\ { \mathcal { P } ( \mathrm { s t o p g r a d } [ \mathcal { E } ( \mathbf { c } _ { t } ; \phi ) ] ) , } & { t \geq 1 . } \end{array} \right.\tag{3}
$$

where $\mathcal { P }$ is a $1 \times 1$ convolutional predictor. TERC quantizes and entropy codes only the residual $\mathbf { r } _ { t } = \mathbf { y } _ { t } - \mathbf { y } _ { t } ^ { \mathrm { p r e d } }$ . The

![](images/188e8d012b410b2dc75941c55725a7b8dd7c7e5e29abbdde41798ba2d362b08b.jpg)  
Fig. 3. Architecture of TERC. It predicts the embedding from historical context, codes the prediction residual, and thus establishes explicit frame-level predictive coding prior to quantization.

decoder reconstructs the embedding as

$$
\widetilde { \mathbf { y } } _ { t } = \mathrm { s t o p g r a d } \left( \mathbf { y } _ { t } ^ { \mathrm { p r e d } } \right) + \widehat { \mathbf { r } } _ { t } .\tag{4}
$$

The first frame therefore codes the complete embedding, while subsequent frames encode only temporally unpredictable information. Stop-gradient prevents cancellation between the prediction and residual branches. The training loss, entropy regularization, and compression fine-tuning follow HNeRV-Boost [12].

## 3. EXPERIMENTS

## 3.1. Experimental Configuration

We use the UVG [23] dataset, comprising seven $1 9 2 0 \times 1 0 8 0$ videos with 600 or 300 frames. Regression performance is evaluated using PSNR and MS-SSIM. Compression performance is characterized by Bpp together with PSNR and MS-SSIM, and BD-rate [24] is computed over the common quality range. We adopt HNeRV-Boost [12] with a ConvNeXt encoder [22] as the baseline. The base and embedding channel dimensions are 64 and 16, respectively. The upsampling scales are [5, 3, 2, 2, 2], and the last five stages contain [1, 1, 2, 2, 2] blocks. Regression models are trained for 300 epochs with Adan [25], a batch size of 1, an initial learning rate of $3 \times 1 0 ^ { - 3 }$ , 10% warm-up, and cosine decay. Compression models are fine-tuned for 100 epochs at $5 \times 1 0 ^ { - 4 }$ using 8-bit quantization, a global Gaussian entropy model, a rate-distortion coefficient of 0.05, and a target bit depth of 4 bits. Comparisons include x265 [26], HM-18.0 [27], DCVC variants [4, 5, 28], and INR codecs [8, 10, 11, 12, 18]. Our neural models are implemented in PyTorch and trained on NVIDIA TITAN RTX GPUs.

![](images/f32bbf4b31a4351d17b2791375bc3659d8a73a0c2c72dbc91cae3120c9ee2f2b.jpg)

![](images/2bc07abe7c98efa1f45e610c7e1907aa101a90572bc935d80e270442009bc334.jpg)  
Fig. 4. Rate-distortion comparison on the UVG dataset in terms of Bpp–PSNR and Bpp–MS-SSIM.

Table 2. BD-rate (%) of TCNeRV relative to the comparison methods on the UVG dataset.
<table><tr><td>Method</td><td>PSNR</td><td>MS-SSIM</td></tr><tr><td>x265 (veryslow) [26]</td><td>-58.76%</td><td>N/A</td></tr><tr><td>HM (RA) [27]</td><td>-22.06%</td><td>-42.83%</td></tr><tr><td>DCVC [4]</td><td>-66.73%</td><td>-44.51%</td></tr><tr><td>DCVC-DC [5]</td><td>8.31%</td><td>114.27%</td></tr><tr><td>DCVC-HEM [28]</td><td>-24.10%</td><td>N/A</td></tr><tr><td>HNeRV-Boost [12]</td><td>-57.26%</td><td>N/A</td></tr><tr><td>HiNeRV [11]</td><td>-29.85%</td><td>-8.74%</td></tr><tr><td>NVRC [18]</td><td>40.22%</td><td>68.61%</td></tr></table>

## 3.2. Video Regression

Table 1 reports the results on the UVG [23] dataset. With approximately 3M parameters, TCNeRV achieves 36.08 dB, exceeding HNeRV-Boost and HiNeRV by 2.20 and 0.82 dB, respectively. The corresponding gains over HNeRV-Boost at the medium and large scales are 1.70 and 1.11 dB, respectively. Larger improvements on high-motion sequences confirm the benefit of historical context.

Table 3. Ablation study of TCNeRV on the UVG dataset.
<table><tr><td>Model MTCF</td><td>TERC</td><td>Size</td><td>MACs</td><td>Avg.</td></tr><tr><td rowspan="4">Baseline [12]</td><td></td><td>3.05M</td><td>130.6G</td><td>33.88</td></tr><tr><td>√</td><td>3.06M</td><td>131.0G</td><td>35.96</td></tr><tr><td>√</td><td>3.05M</td><td>130.6G</td><td>34.08</td></tr><tr><td>√ √</td><td>3.06M</td><td>131.0G</td><td>36.08</td></tr></table>

## 3.3. Video Compression

Fig. 4 and Table 2 present the compression results on the UVG [23] dataset. TCNeRV reduces BD-rate by 22.06%, 66.73%, 24.10%, 57.26%, and 29.85% relative to HM, DCVC, DCVC-HEM, HNeRV-Boost, and HiNeRV, respectively. Its BD-rates relative to DCVC-DC and NVRC are +8.31% and +40.22%, indicating remaining performance gaps.

## 3.4. Ablation Study

Table 3 shows that MTCF improves the average PSNR from 33.88 to 35.96 dB while adding only 0.01M parameters and 0.4G MACs. TERC alone provides a 0.20 dB improvement, and the full model reaches 36.08 dB, exceeding the MTCFonly variant by 0.12 dB. These results indicate that MTCF provides the primary reconstruction gain, while TERC offers a modest additional improvement.

## 4. CONCLUSION

TCNeRV uses historical reconstructions for gated multi-scale decoding-feature fusion and content-embedding residual coding. With approximately 3M parameters, it reduces BD-rate by 57.26% relative to HNeRV-Boost, demonstrating competitiveness among implicit neural video codecs. Future work will address long-sequence errors, scene changes, random access, and rate-model optimization.

## 5. REFERENCES

[1] G. J. Sullivan, J.-R. Ohm, W.-J. Han, and T. Wiegand, “Overview of the high efficiency video coding (HEVC) standard,” IEEE Trans. Circuits Syst. Video Technol., vol. 22, no. 12, pp. 1649–1668, 2012.

[2] B. Bross et al, “Overview of the versatile video coding (VVC) standard and its applications,” IEEE Trans. Circuits Syst. Video Technol., vol. 31, no. 10, pp. 3736– 3764, 2021.

[3] G. Lu, W. Ouyang, D. Xu, X. Zhang, C. Cai, and Z. Gao, “DVC: An end-to-end deep video compression framework,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2019, pp. 11006–11015.

[4] J. Li, B. Li, and Y. Lu, “Deep contextual video compression,” in Adv. Neural Inf. Process. Syst. (NeurIPS), 2021, vol. 34, pp. 18114–18125.

[5] J. Li, B. Li, and Y. Lu, “Neural video compression with diverse contexts,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2023, pp. 22616–22626.

[6] V. Sitzmann, J. N. P. Martel, A. W. Bergman, D. B. Lindell, and G. Wetzstein, “Implicit neural representations with periodic activation functions,” in Adv. Neural Inf. Process. Syst. (NeurIPS), 2020, vol. 33, pp. 7462–7473.

[7] E. Dupont, A. Golinski, M. Alizadeh, Y. W. Teh, and A. Doucet, “COIN: Compression with implicit neural representations,” in Proc. Int. Conf. Learn. Represent. (ICLR), 2021.

[8] H. Chen, B. He, H. Wang, Y. Ren, S.-N. Lim, and A. Shrivastava, “NeRV: Neural representations for videos,” in Adv. Neural Inf. Process. Syst. (NeurIPS), 2021, vol. 34, pp. 21557–21568.

[9] Z. Li, M. Wang, H. Pi, K. Xu, J. Mei, and Y. Liu, “E-NeRV: Expedite neural video representation with disentangled spatial-temporal context,” in Proc. Eur. Conf. Comput. Vis. (ECCV), 2022, pp. 267–284.

[10] H. Chen, M. Gwilliam, S.-N. Lim, and A. Shrivastava, “HNeRV: A hybrid neural representation for videos,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2023, pp. 10270–10279.

[11] H. M. Kwan, G. Gao, F. Zhang, A. Gower, and D. Bull, “HiNeRV: Video compression with hierarchical encoding-based neural representation,” in Adv. Neural Inf. Process. Syst. (NeurIPS), 2023, vol. 36, pp. 72692– 72704.

[12] X. Zhang et al, “Boosting neural representations for videos with a conditional decoder,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2024, pp. 2556–2566.

[13] J. C. Lee, D. Rho, J. H. Ko, and E. Park, “FFNeRV: Flow-guided frame-wise neural representations for videos,” in Proc. ACM Int. Conf. Multimedia (ACM MM), 2023, pp. 7859–7870.

[14] Q. Zhao, M. S. Asif, and Z. Ma, “DNeRV: Modeling inherent dynamics via difference neural representation for videos,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2023, pp. 2031–2040.

[15] Y. Zhang, T. van Rozendaal, J. Brehmer, M. Nagel, and T. Cohen, “Implicit neural video compression,” in Proc. Int. Conf. Learn. Represent. (ICLR), 2022.

[16] C. Gomes, R. Azevedo, and C. Schroers, “Video compression with entropy-constrained neural representations,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2023, pp. 18497–18506.

[17] H. Kim, M. Bauer, L. Theis, J. R. Schwarz, and E. Dupont, “C3: High-performance and low-complexity neural compression from a single image or video,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2024, pp. 9347–9358.

[18] H. M. Kwan, G. Gao, F. Zhang, A. Gower, and D. Bull, “Neural video representation compression,” in Adv. Neural Inf. Process. Syst. (NeurIPS), 2024, vol. 37, pp. 132440–132462.

[19] G. Gao, S. Teng, T. Peng, F. Zhang, and D. Bull, “GIViC: Generative implicit video compression,” in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), 2025, pp. 17356–17367.

[20] W. Jiang, J. Li, K. Zhang, and L. Zhang, “BiECVC: Gated diversification of bidirectional contexts for learned video compression,” in Proc. ACM Int. Conf. Multimedia (ACM MM), 2025, pp. 7248–7257.

[21] S. R. et al, “NIRVANA: Neural implicit representations of videos with adaptive networks and autoregressive patch-wise modeling,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2023, pp. 14378– 14387.

[22] Z. Liu, H. Mao, C.-Y. Wu, C. Feichtenhofer, T. Darrell, and S. Xie, “A ConvNet for the 2020s,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2022, pp. 11976–11986.

[23] A. Mercat, M. Viitanen, and J. Vanne, “UVG dataset: 50/120fps 4K sequences for video codec analysis and development,” in Proc. ACM Multimedia Syst. Conf. (MMSys), 2020, pp. 297–302.

[24] G. Bjontegaard, “Calculation of average PSNR differences between RD-curves,” Tech. Rep. VCEG-M33, ITU-T Video Coding Experts Group, 2001.

[25] X. Xie, P. Zhou, H. Li, Z. Lin, and S. Yan, “Adan: Adaptive Nesterov momentum algorithm for faster optimizing deep models,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 46, no. 12, pp. 9508–9520, 2024.

[26] VideoLAN, “x265 HEVC encoder,” [Online]. Available: https://www.videolan.org/developers/x265.html. [Accessed: Sep. 14, 2026].

[27] C. Rosewarne, K. Sharman, R. Sjoberg, and G. Sullivan, “High efficiency video coding (HEVC) test model 16 (HM16) improved encoder description update 16,” Tech. Rep., Joint Video Experts Team, 2022.

[28] J. Li, B. Li, and Y. Lu, “Hybrid spatial-temporal entropy modelling for neural video compression,” in Proc. ACM Int. Conf. Multimedia (ACM MM), 2022, pp. 1503– 1511.