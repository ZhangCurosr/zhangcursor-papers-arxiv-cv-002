# SemMSA: Latent Semantic-Aided Robust Multimodal Sentiment Analysis with Incomplete Data

Wenhao Li<sup>1,2</sup>, Zhibin Wu<sup>1</sup>, Chong Xiao<sup>1</sup>, Qiangchang Wang<sup>1∗</sup> <sup>1</sup>Software School, Shandong University <sup>2</sup>Shenzhen Loop Area Institute

## Abstract

Recent research on Multimodal Sentiment Analysis (MSA) has focused on learning from language, visual, and acoustic modalities with incomplete data to infer human sentiment. Most studies typically compensate for missing information by reconstructing modality features or designing complicated fusion mechanisms. However, these methods still suffer from spurious generation and noisy guidance due to the lack of high-level semantic grounding in partially observed multimodal evidence. To address these issues, we propose SemMSA, a latent semantic-aided framework that constructs rich sentiment-relevant semantics with LLMs, fully integrating with all modalities via anchor-free spectral alignment. It mainly consists of Crossmodal Semantic Refinement (CSR) and Cross-modal Spectral Alignment (CSA). Specifically, CSR first adaptively extracts visual and acoustic representations by corresponding adapters to form a unified multimodal prefix with language in the frozen LLM embedding space. It then iteratively produces continuous discriminative semantic states through a token-efficient latent refinement process without decoding explicit text. Next, CSA simultaneously aligns the refined semantics with all modalities by enhancing the dominant spectral component of their kernel Gram matrix. This captures global nonlinear dependencies among all representations without relying on a predefined anchor modality. In addition, an instance-level spectral separation constraint preserves cross-sample discriminability and mitigates representation collapse. Extensive experiments on SIMS, MOSI, and MOSEI benchmarks demonstrate that SemMSA achieves state-of-the-art performance.

## 1 Introduction

Multimodal Sentiment Analysis (MSA) aims to learn a comprehensive understanding of human sentiment by modeling multimodal information from language, vision, and audio [1, 2]. Recent research has increasingly moved from controlled laboratory settings toward real-world multimodal scenarios and deployment [3–5]. However, multimodal observations are often incomplete due to noise, occlusion, sensor failure, or transmission instability [6–8]. In particular, intra-modal missingness, where partial tokens, frames, or acoustic segments are corrupted within each modality, frequently occurs in practice and significantly degrades models trained under complete data assumptions.

Recent studies have made significant progress, following two paradigms, as shown in Figure 1(a) and (b). Reconstruction-based methods [9–12] aim to recover complete features from partially observed inputs, alleviating information loss. However, they tend to reconstruct low-level feature patterns rather than high-level semantics that are critical for sentiment understanding. Moreover, the same text may correspond to multiple plausible vocal prosodies and facial expressions, resulting in hallucinating features that contradict the actual sentiment evidence. Fusion-based methods [7, 8, 6, 13, 14] instead directly integrate available modalities through complex cross-modal interaction modules to learn robust representations. However, the fused features still suffer from noisy observations and limited evidence. They may overfit modality co-occurrence patterns observed during training under severe missing-modality scenarios, leading to unstable guidance and degraded discrimination.

![](images/06e3a4ed000fdaf71f3fed7b280d3c9b90d3140cb030d742b4bc3dcb143e4b9d.jpg)  
Figure 1: Framework comparison between (a) reconstruction-based methods, (b) fusion-based methods, and (c) our SemMSA. Compared with existing methods, SemMSA refines and fully integrates sentiment-relevant semantic information with the LLM to compensate for incomplete data.

Human sentiment understanding often relies on reliable contextual cues and prior knowledge about emotional expression to interpret intentions, attitudes, and affective states under incomplete observations [15, 16]. This motivates semantic-level compensation, where the goal is to enrich incomplete multimodal representations with rich sentiment-relevant semantic information. Recent studies also indicate a broader shift from discriminative emotion recognition toward generative emotion understanding [17]. Large Language Models (LLMs), with strong contextual representation and semantic abstraction capabilities, provide a promising source of such high-level semantics. However, directly decoding textual descriptions or rationales introduces additional costly overhead. Therefore, an efficient mechanism is needed to exploit LLM-derived semantics in the hidden space.

Inspired by this perspective, we propose SemMSA, a latent semantic-aided framework with LLMs, as shown in Figure 1(c). First, Cross-modal Semantic Refinement (CSR) bridges heterogeneous non-language modalities and the frozen LLM embedding space through lightweight adapters. Each adapter uses learnable prompts with self- and cross-attention to adaptively aggregate reliable evidence from incomplete visual or acoustic sequences. The produced compact visual and acoustic prefix tokens are concatenated with language embedded in the frozen LLM space. CSR then recurrently appends the last hidden state of the LLM as a continuous latent token, yielding a sequence of sentiment-relevant semantic states. This provides token-efficient semantic compensation without explicit text decoding. Next, Cross-modal Spectral Alignment (CSA) establishes a kernel Gram matrix over the normalized semantic, language, visual, and acoustic representations for comprehensive cross-modal consistency alignment. It encourages all features to align simultaneously along a shared leading direction by enhancing the dominant spectral component of this matrix. This captures global nonlinear dependencies across representations, avoiding the instability from anchor modality under severe missingness. In addition, an instance-wise spectral separation constraint is imposed on the dominant eigenvectors to preserve cross-sample discriminability, mitigating representation collapse.

Overall, our contributions are summarized as follows:

• A SemMSA framework is proposed to provide high-level semantic compensation with LLMs and jointly align heterogeneous representations via an anchor-free spectral objective.

• CSR module adaptively aggregates multimodal data into the frozen LLM space and iteratively refines sentiment-relevant semantic states in a token-efficient latent process.

• CSA module aligns all representations simultaneously by enhancing the dominant spectral component they form, capturing global nonlinear relationships without anchor dependency.

• The proposed method consistently achieves state-of-the-art performance on MOSI, MOSEI, and SIMS under diverse missingness, significantly improving accuracy by 1.4% on average.

## 2 Related Work

## 2.1 MSA with Incomplete Data

Most MSA methods often assume that language, visual, and acoustic modalities are fully observed [18, 3, 19–21, 4, 22]. They learn unified representations by modeling intra- and inter-modal contextual dependencies. Once modalities are missing or corrupted, models trained under complete settings often degrade sharply. Recent fusion-based methods attempt to alleviate missing information through complex interaction and fusion, including multi-view correlation learning [23, 24], recurrent cross-modal translation [25, 26], graph-based dependency modeling [27], and modality-conditioned fusion [28]. Several dominance-guided approaches, such as LNLN [7] and P-RMF [8], further rely on high-quality primary modalities to anchor multimodal learning. Nevertheless, these methods may become unreliable under severe missingness [12]. The scarce representative features with noise in limited data may fail to capture sufficient discriminative information. Another line of work reconstructs missing modalities from observed ones, including cascaded residual auto-encoders [29], cycle-consistency-based recovery [25, 26], graph-based reconstruction [27], low-level feature restoration [10], and diffusion-based distribution recovery [12]. However, reconstruction-based methods mainly recover superficial statistical patterns rather than sentiment cues essential for sentiment judgment, while diffusion models also incur substantial computational costs. In contrast, SemMSA complements incomplete data with compact LLM-derived latent semantics and comprehensively integrates all representations through a consistent spectral alignment. To the best of our knowledge, this is the first attempt to utilize LLMs to perform semantic-level compensation for MSA.

## 2.2 Multimodal Alignment

Most existing methods [30–32] adopt CLIP-based pairwise contrastive learning [33]. This paradigm has inspired a series of works extending to additional modalities, such as audio-to-text [34], point cloud-to-text [35], and video-to-text [36]. However, aligning each representation only to a single anchor neglects the interactions among the remaining points, making it difficult to capture global structural relationships. This inherently ignores the complexity and richness of multiple modalities. The Gram matrix, which characterizes the mutual geometry among sets of vectors, has shown promise in theoretical analyses of deep learning networks [37] and various downstream tasks [38–40]. In contrast, SemMSA introduces a kernelized Gram matrix to capture nonlinear relationships, aligning all representations along a dominant spectral direction without anchor dependency.

## 3 Method

We begin by introducing the preprocessing of MSA, and then present two proposed components: (1) how cross-modal latent semantics are generated through CSR, and (2) how all representations are simultaneously aligned via CSA. Figure 2 shows the overview of the proposed SemMSA.

## 3.1 Multimodal Input

Given a multimodal sample for MSA, we consider three modalities of vision, audio, and language. Each modality with incomplete data is denoted by ${ \mathbf { U } } _ { m }$ , where $m \in \{ v , a , l \}$ . Following prior works [7–9], each modality is first processed by a widely used modality-specific encoder $f _ { \phi }$ with frozen parameters [41–43] to obtain a sequence representation $\mathbf { X } _ { m }$ as follows:

$$
\mathbf { X } _ { m } = f _ { \phi } ( \mathbf { U } _ { m } ) , \quad \mathbf { X } _ { m } \in \mathbb { R } ^ { T _ { m } \times d _ { m } } ,\tag{1}
$$

where $T _ { m }$ is the sequence length and $d _ { m }$ denotes the feature dimension of each embedding vector.

## 3.2 Cross-modal Semantic Refinement

Visual and Audio Adapter. To extract heterogeneous nonverbal features into the embedding space of a frozen LLM, visual and acoustic adapters with M learnable prompt embeddings $\mathbf { P } _ { m } \in \breve { \mathbb { R } ^ { M \times d _ { m } } }$ are introduced and updated by a lightweight transformer with B blocks. Due to the features $\mathbf { X } _ { m }$ of frozen encoders without temporal information, learnable positional embeddings $\mathbf { t } _ { m } \in \mathbb { R } ^ { T _ { m } \times d _ { m } }$ are injected into the frame or audio segment representations to obtain temporal order. $\mathbf { P } _ { m }$ first act as queries to perform cross-attention with ${ \bf { X } } _ { m }$ to extract modality evidence. These embeddings then interact with each other to capture intra-modality dependencies through self-attention.

![](images/c3d3abc85c4be9105f3596088d4439fff50972ea7517290a1e32d7715c35f189.jpg)  
Figure 2: Overview of the proposed SemMSA framework. Multimodal inputs with incomplete data are first extracted by encoders. Then, CSR maps visual and audio features via adapters and concatenates with language embedded in the frozen LLM space. Iterative hidden-state refinement is performed to produce compact sentiment-aware latent semantic states. Next, CSA jointly aligns the semantics with visual, acoustic, and language representations by constructing a kernel Gram matrix over normalized features and enhancing its dominant spectral component.

After B blocks, the output of the adapter consists of M modality-specific vectors, one per prompt embedding. They are linearly projected into the dimension of the LLM to obtain $\mathbf { Z } _ { v } , \mathbf { Z } _ { a } ^ { \mathbf { \bar { \nu } } } \in \mathbf { \bar { \mathbb { R } } } ^ { M \times d }$ The adapters are optimized to effectively bridge the output of the frozen encoders to the frozen LLM during training. In addition, the language $\mathbf { Z } _ { l } \in \mathbb { R } ^ { T \times d }$ is directly embedded by the LLM, where T is the length of the textual embedding sequence and d is the LLM embedding dimension.

Semantic Refinement. The multimodal input ${ \bf U } ^ { ( 0 ) } = [ { \bf Z } _ { l } ; { \bf Z } _ { v } ; { \bf Z } _ { a } ] \in \mathbb { R } ^ { ( T + 2 M ) \times d }$ to the frozen LLM is formed by concatenating the three modalities in a fixed order. $\mathbf { U } ^ { ( 0 ) }$ serves as the initial multimodal prefix, providing the frozen LLM with partially observed language, visual, and acoustic evidence in a unified embedding space. Instead of decoding explicit textual descriptions, CSR performs iterative hidden-state refinement directly in the continuous LLM space. Let $\mathcal { F } _ { \theta } ( \mathbf { U } ) \in$ $\bar { \mathbb { R } } ^ { | \mathbf { U } | \times d }$ denote the last-layer hidden states produced by the frozen LLM with parameters θ for an input sequence U. The hidden state at the final position is defined as:

$$
\mathbf { f } ( \mathbf { U } ) = \mathcal { F } _ { \boldsymbol { \theta } } ( \mathbf { U } ) _ { | \mathbf { U } | } \in \mathbb { R } ^ { d } ,\tag{2}
$$

where f(U) provides a contextual summary of the current multimodal prefix. Based on this representation, the k-th latent semantic state is constructed as

$$
\begin{array} { r } { \mathbf { z } _ { k } = \mathbf { f } \left( \mathbf { U } ^ { ( k - 1 ) } \right) , \quad \mathbf { U } ^ { ( k ) } = \left[ \mathbf { U } ^ { ( k - 1 ) } ; \mathbf { z } _ { k } \right] , \quad k = 1 , \ldots , O , } \end{array}\tag{3}
$$

where O is the number of refinement steps, $\mathbf { z } _ { k } \in \mathbb { R } ^ { d }$ denotes the continuous latent semantic state at step $k ,$ and $\mathbf { U } ^ { ( k ) } \in \mathbb { R } ^ { ( T + 2 M + k ) \times d }$ is the updated prefix after appending $\mathbf { z } _ { k }$ . Since $\mathbf { z } _ { k }$ has the same dimensionality as the LLM token embeddings, it can be appended as a continuous input token for the next refinement step. We use this recurrent operation as a hidden-space semantic refinement mechanism, rather than explicit textual generation. The resulting sequence of latent semantic states is

$$
{ \bf Z } = [ { \bf z } _ { 1 } ; { \bf z } _ { 2 } ; \cdot \cdot \cdot ; { \bf z } _ { O } ] \in \mathbb { R } ^ { O \times d } .\tag{4}
$$

These states provide a compact representation of sentiment-relevant information conditioned on the available multimodal evidence. By repeatedly applying the frozen LLM transformation to the updated prefix, CSR refines the hidden context while avoiding autoregressive decoding of natural-language descriptions. Finally, a compact sentiment-aware semantic representation H is obtained for final fusion and prediction by applying pooling followed by a linear projection with W and ${ \bf b } _ { \mathrm { s } }$

$$
\mathbf { H } _ { \mathrm { s } } = \mathbf { W } _ { \mathrm { s } } \operatorname { P o o l } ( \mathbf { Z } ) + \mathbf { b } _ { \mathrm { s } } , \qquad \mathbf { H } _ { \mathrm { s } } \in \mathbb { R } ^ { d ^ { \prime } } .\tag{5}
$$

## 3.3 Cross-modal Spectral Alignment

To achieve comprehensive consistency alignment among semantics, vision, audio, and language, Cross-modal Spectral Alignment (CSA) is proposed to improve global and nonlinear interactions among all representations. Unlike traditional pairwise contrastive objectives which align modality pairs through a predefined anchor, CSA measures the alignment by analyzing the spectral structure of their shared representation matrix. A stronger dominant spectral component indicates that different modalities are concentrated along a common latent direction.

Specifically, three modality representations ${ \bf { X } } _ { m }$ are first extracted and unified by their respective modality encoders to obtain $\mathbf { H } _ { m }$ with the $d ^ { \prime }$ dimension. Each encoder consists of a linear transformation layer followed by Transformer encoder layers. Next, they are transformed into normalized quadruples and construct the matrix $\mathbf { V } = ( \mathbf { h } _ { s } , \mathbf { \dot { h } } _ { v } , \mathbf { h } _ { a } , \mathbf { h } _ { l } )$ . The corresponding Gram matrix $\mathbf { G } \in \mathbb { R } ^ { 4 \times 4 }$ is first defined to reflect the pairwise similarity as follows:

$$
\mathbf { G } ( \mathbf { V } ) = \mathbf { V } ^ { \top } \mathbf { V } = { \left[ \begin{array} { l l l l l } { \left. \mathbf { h } _ { s } , \mathbf { h } _ { s } \right. } & { \left. \mathbf { h } _ { s } , \mathbf { h } _ { v } \right. } & { \left. \mathbf { h } _ { s } , \mathbf { h } _ { a } \right. } & { \left. \mathbf { h } _ { s } , \mathbf { h } _ { l } \right. } \\ { \left. \mathbf { h } _ { v } , \mathbf { h } _ { s } \right. } & { \left. \mathbf { h } _ { v } , \mathbf { h } _ { v } \right. } & { \left. \mathbf { h } _ { v } , \mathbf { h } _ { a } \right. } & { \left. \mathbf { h } _ { v } , \mathbf { h } _ { l } \right. } \\ { \left. \mathbf { h } _ { a } , \mathbf { h } _ { s } \right. } & { \left. \mathbf { h } _ { a } , \mathbf { h } _ { v } \right. } & { \left. \mathbf { h } _ { a } , \mathbf { h } _ { a } \right. } & { \left. \mathbf { h } _ { a } , \mathbf { h } _ { l } \right. } \\ { \left. \mathbf { h } _ { l } , \mathbf { h } _ { s } \right. } & { \left. \mathbf { h } _ { l } , \mathbf { h } _ { v } \right. } & { \left. \mathbf { h } _ { l } , \mathbf { h } _ { a } \right. } & { \left. \mathbf { h } _ { l } , \mathbf { h } _ { l } \right. } \end{array} \right] } , \quad \mathbf { G } _ { i j } = \langle \mathbf { h } _ { i } , \mathbf { h } _ { j } \rangle .\tag{6}
$$

To further learn nonlinear cross-modal dependency, this metric is extended into a high-dimensional Reproducing Kernel Hilbert Space (RKHS) using a Radial Basis Function (RBF) kernel mapping $\kappa ( \cdot , \cdot )$ The corresponding kernel Gram matrix $\mathbf { K } ( V )$ is defined as:

$$
{ \bf K } _ { i j } = \kappa ( { \bf h } _ { i } , { \bf h } _ { j } ) , \qquad \kappa ( { \bf h } _ { i } , { \bf h } _ { j } ) = \exp \left( - \frac { \| { \bf h } _ { i } - { \bf h } _ { j } \| _ { 2 } ^ { 2 } } { 2 \sigma ^ { 2 } } \right) .\tag{7}
$$

If the four representations of the same instance are sufficiently aligned in the kernel space, then the kernel Gram matrix should exhibit a significant low-rank structure. In the ideal case, all representations are mapped to the same shared semantic direction, and K degenerates into a rank-one matrix. Therefore, we perform eigendecomposition on K as follows:

$$
{ \bf K } = { \bf Q } { \bf \Lambda } { \bf Q } ^ { \top } , \qquad { \bf \Lambda } = \mathrm { d i a g } ( \lambda _ { 1 } , \lambda _ { 2 } , \lambda _ { 3 } , \lambda _ { 4 } ) ,\tag{8}
$$

where $\lambda _ { 1 } \geq \lambda _ { 2 } \geq \lambda _ { 3 } \geq \lambda _ { 4 } \geq 0$ . The largest eigenvalue $\lambda _ { 1 }$ represents the principal nonlinear semantic direction shared by the four representations. A higher proportion of it indicates that the representations are more concentrated in the same latent alignment subspace. Therefore, we regard the eigenvalues as logits and enhance the dominance of $\lambda _ { 1 }$ through a softmax-based spectral objective:

$$
\mathcal { L } _ { \mathrm { c s a } } = - \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \log \frac { \exp ( \lambda _ { 1 } ^ { i } / \tau ) } { \sum _ { j = 1 } ^ { 4 } \exp ( \lambda _ { j } ^ { i } / \tau ) } ,\tag{9}
$$

where $N$ is the batch size and τ denotes the temperature parameter. To prevent degenerate solutions where different instances collapse to the same semantic point, we introduce an instance-wise spectral separation regularizer. For the i-th instance, the dominant eigenvector $\mathbf { q } _ { 1 } ^ { i }$ associated with the largest eigenvalue $\lambda _ { 1 } ^ { i }$ of the kernel Gram matrix indicates the principal alignment pattern among the semantic, visual, acoustic, and language representations. Since $\mathbf { q } _ { 1 } ^ { i }$ only reflects representation-wise combination coefficients, we project it back to the high-dimensional representation space to obtain the dominant semantic direction ${ \bf u } _ { 1 } ^ { i }$ . The separation loss is then defined as:

$$
\mathbf { u } _ { 1 } ^ { i } = \frac { \mathbf { V } ^ { i } \mathbf { q } _ { 1 } ^ { i } } { \left\| \mathbf { V } ^ { i } \mathbf { q } _ { 1 } ^ { i } \right\| _ { 2 } } , \qquad \mathcal { L } _ { \mathrm { s e p } } = \frac { 1 } { N ( N - 1 ) } \sum _ { \substack { i = 1 } } ^ { N } \sum _ { \substack { j = 1 } } ^ { N } \left[ ( \mathbf { u } _ { 1 } ^ { i } ) ^ { \top } \mathbf { u } _ { 1 } ^ { j } \right] ^ { 2 } .\tag{10}
$$

where $\mathbf { V } ^ { i }$ contains the normalized representations of the i-th instance. This regularizer penalizes high similarity between dominant semantic directions of different instances, preserving inter-instance discriminability. These aligned embeddings are fused via element-wise summation to form the final multimodal representation H<sup>ˆ</sup> as follows:

$$
\hat { \mathbf { H } } = \mathbf { H } _ { s } + \mathbf { H } _ { v } + \mathbf { H } _ { a } + \mathbf { H } _ { l } .\tag{11}
$$

Table 1: The average results on MOSI and MOSEI datasets over missing rates from 0.0 to 0.9.
<table><tr><td></td><td colspan="6">MOSI</td><td colspan="6">MOSEI</td></tr><tr><td>Method</td><td>Acc-2</td><td>F1</td><td>Acc-5</td><td>Acc-7</td><td>MAE↓</td><td>Corr ↑</td><td>Acc-2</td><td>F1</td><td>Acc-5</td><td>Acc-7</td><td>MAE↓</td><td>Corr ↑</td></tr><tr><td>MISA [44]</td><td>70.33/71.49</td><td>70.00/71.28</td><td>33.08</td><td>29.85</td><td>1.085</td><td>0.524</td><td>75.82/71.27</td><td>68.73/63.85</td><td>39.39</td><td>40.84</td><td>0.780</td><td>0.503</td></tr><tr><td>Self-MM [45]</td><td>69.26/70.51</td><td>67.54/66.60</td><td>34.67</td><td>29.55</td><td>1.070</td><td>0.512</td><td>77.42/73.89</td><td>72.31/68.92</td><td>45.38</td><td>44.70</td><td>0.695</td><td>0.498</td></tr><tr><td>MMIM [46]</td><td>67.06/69.14</td><td>64.04/66.65</td><td>33.77</td><td>31.30</td><td>1.077</td><td>0.507</td><td>75.89/73.32</td><td>70.32/68.72</td><td>41.74</td><td>40.75</td><td>0.739</td><td>0.489</td></tr><tr><td>TETFN [47]</td><td>67.68/69.76</td><td>63.29/65.69</td><td>34.34</td><td>30.30</td><td>1.087</td><td>0.507</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>TFR-Net [9]</td><td>66.35/68.15</td><td>60.06/61.73</td><td>34.67</td><td>29.54</td><td>1.200</td><td>0.459</td><td>77.23/73.62</td><td>71.99/68.80</td><td>34.67</td><td>46.83</td><td>0.697</td><td>0.489</td></tr><tr><td>ALMT [4]</td><td>68.39/70.40</td><td>71.80/72.57</td><td>33.42</td><td>30.30</td><td>1.083</td><td>0.498</td><td>77.54/76.64</td><td>78.03/77.14</td><td>41.64</td><td>40.92</td><td>0.674</td><td>0.481</td></tr><tr><td>LNLN [7]</td><td>70.94/72.55</td><td>71.25/72.73</td><td>38.27</td><td>34.26</td><td>1.046</td><td>0.527</td><td>78.19/76.30</td><td>79.95/77.77</td><td>46.17</td><td>45.42</td><td>0.692</td><td>0.530</td></tr><tr><td>P-RMF [8]</td><td>71.53/72.81</td><td>71.69/72.93</td><td>38.50</td><td>34.19</td><td>1.038</td><td>0.525</td><td>78.83/78.14</td><td>80.39/79.33</td><td>45.87</td><td>44.63</td><td>0.658</td><td>0.589</td></tr><tr><td>TF-Mamba [48]</td><td>73.46/72.54</td><td>73.59/72.57</td><td>37.74</td><td>33.95</td><td>1.035</td><td>0.548</td><td>77.34/77.61</td><td>77.18/77.43</td><td>46.64</td><td>45.66</td><td>0.673</td><td>0.578</td></tr><tr><td>SemMSA</td><td>74.36/73.91</td><td>74.18/73.82</td><td>40.93</td><td>36.49</td><td>1.011</td><td>0.550</td><td>79.61/79.38</td><td>80.62/79.87</td><td>48.12</td><td>47.06</td><td>0.648</td><td>0.601</td></tr></table>

The fused feature H<sup>ˆ</sup> is finally fed into a linear classifier for sentiment prediction $\hat { y } = \mathrm { L i n e a r } ( \hat { \mathbf { H } } )$ Finally, the overall loss also includes Mean Squared Error (MSE) loss between the predicted score yˆ and the ground-truth sentiment label y, as follows:

$$
\mathcal { L } _ { \mathrm { t o t a l } } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \left. y ^ { i } - \hat { y } ^ { i } \right. _ { 2 } ^ { 2 } + \mathcal { L } _ { \mathrm { c s a } } + \mathcal { L } _ { \mathrm { s e p } } .\tag{12}
$$

## 4 Experiments

## 4.1 Experimental Details

Datasets. Extensive experiments are conducted on three standard MSA benchmarks: MOSI [49] and MOSEI [50] annotated with sentiment scores in [−3, +3]. SIMS [51] is a Chinese dataset annotated in [−1, +1]. The detailed dataset statistics are introduced in the Appendix.

Implementation Details. The proposed SemMSA is trained for 200 epochs with a batch size of 64 across all datasets. The AdamW optimizer [52] is used with a learning rate of $1 \times 1 0 ^ { - 4 }$ , together with warm-up, cosine annealing, and early stopping strategies. The input sequence length T is set to 8, and the hidden dimension d is set to 128. Qwen3-1.7B [53] is adopted as the frozen LLM to generate semantic features. The visual and acoustic Adapters use M = 8 learnable prompts and B = 2 blocks. Refinement step O in CSR is set to 4. The RBF kernel bandwidth σ = 1.0 and temperature parameter τ = 0.1 by default. All experiments are performed with an NVIDIA RTX 6000 Ada.

Missingness Settings and Evaluation Metrics. Following [7–9, 48], missing visual and acoustic segments are replaced with zero vectors and language tokens with [UNK]. Training employs instancewise Bernoulli masking, with 50% of samples kept complete. During testing, the missing rate is varied from 0 to 0.9 (step 0.1) and missing positions are independently sampled per modality. We report the average results for random seeds 1111, 1112, and 1113. For MOSI and MOSEI, we report seven-class accuracy (Acc-7), Acc-5, Acc-2, F1 score, Mean Absolute Error (MAE), and Pearson correlation coefficient (Corr). Acc-2 and F1 are reported sequentially in two forms of negative/non-negative and negative/positive. For SIMS, we report Acc-5, Acc-3, Acc-2, F1 score, MAE, and Corr.

## 4.2 Main Results

Intra-modal Missingness. Table 1 and 2 report comparisons on MOSI, MOSEI, and SIMS under intra-modal missing settings. SemMSA consistently achieves state-of-the-art performance. On MOSI, it improves Acc-5 and Acc-7 over the best baselines TF-Mamba [48] by 2.43% and 2.23%, respectively, while achieving the lowest MAE of 1.011. On MOSEI, SemMSA further improves Acc-2 to 79.61 and F1 to 80.62, demonstrating its scalability on larger-scale data. On SIMS, the improvement is

Table 2: The average results on SIMS dataset over missing rates from 0.0 to 0.9.
<table><tr><td>Method</td><td>Acc-2</td><td>F1</td><td>Acc-3</td><td>Acc-5</td><td>MAE↓</td><td>Corr ↑</td></tr><tr><td>MISA [44]</td><td>72.71</td><td>66.30</td><td>56.87</td><td>31.53</td><td>0.539</td><td>0.348</td></tr><tr><td>Self-MM [45]</td><td>72.81</td><td>68.43</td><td>56.75</td><td>32.28</td><td>0.508</td><td>0.376</td></tr><tr><td>MMIM [46]</td><td>69.86</td><td>66.21</td><td>56.76</td><td>31.81</td><td>0.544</td><td>0.339</td></tr><tr><td>TETFN [47]</td><td>73.58</td><td>68.67</td><td>56.91</td><td>33.42</td><td>0.505</td><td>0.387</td></tr><tr><td>TFR-Net [9]</td><td>68.13</td><td>58.70</td><td>52.89</td><td>26.52</td><td>0.661</td><td>0.169</td></tr><tr><td>ALMT [4]</td><td>71.85</td><td>76.21</td><td>56.47</td><td>34.16</td><td>0.509</td><td>0.372</td></tr><tr><td>LNLN [7]</td><td>73.15</td><td>69.18</td><td>57.32</td><td>34.68</td><td>0.515</td><td>0.387</td></tr><tr><td>P-RMF [8]</td><td>73.64</td><td>74.65</td><td>54.75</td><td>34.83</td><td>0.500</td><td>0.414</td></tr><tr><td>TF-Mamba [48]</td><td>74.68</td><td>72.20</td><td>55.51</td><td>34.46</td><td>0.512</td><td>0.386</td></tr><tr><td>SemMSA</td><td>75.46</td><td>77.50</td><td>58.84</td><td>35.68</td><td>0.474</td><td>0.501</td></tr></table>

![](images/1808665df34f1a59fe73ffeca1b7997cd107cfd2adc9d358f112eaaf7e7a560c.jpg)  
MISA  Self-MM  MMIM  TFR-Net → LNLN → P-RMF  TF-Mamba  SemMSA  
Figure 3: Performance visualization of F1 Score, MAE, Acc-2, and Corr across MOSI, MOSEI, and SIMS under missing rates from 0.0 to 0.9, where lower MAE denotes superior performance.

particularly clear, surpassing P-RMF [8] by 0.087 in Corr. Figure 3 further illustrates the performance trends of F1, Acc-2, MAE, and Corr. As the missing rate increases, most baseline methods exhibit clear performance degradation, struggling to maintain reliable representations. In contrast, SemMSA maintains consistently superior and more stable performance across missing rates. These results indicate that SemMSA can effectively complement incomplete multimodal observations with sentiment-relevant latent semantics and comprehensively align heterogeneous representations through global nonlinear alignment, leading to strong robustness and generalization under challenging missing-modality scenarios. Detailed results under each missing rate are provided in the Appendix.

Inter-modal Missingness. Following the same protocol in [13, 27, 55], we further remove entire modalities from MOSEI samples to evaluate the performance of SemMSA under intermodal missingness. The results are reported in Table 3 using F1 as the evaluation metric. It is worth noting that (1) all methods degrade significantly in unimodal scenarios, confirming the complementary roles of different modalities. (2) Language consistently outperforms visual and acoustic modalities, highlighting more explicit

Table 3: The F1 scores on the MOSEI dataset under varying inter-modal missing conditions.
<table><tr><td>Method</td><td>{1}</td><td>{a}</td><td>{v}</td><td>{l, a}</td><td>{l, v}</td><td>{a, v}</td><td>{l, a, v}</td><td>Avg.</td></tr><tr><td>Self-MM [45]</td><td>71.53</td><td>43.57</td><td>37.61</td><td>75.91</td><td>74.62</td><td>49.52</td><td>83.69</td><td>62.35</td></tr><tr><td>CubeMLP [54]</td><td>67.52</td><td>39.54</td><td>32.58</td><td>71.69</td><td>70.06</td><td>48.54</td><td>83.17</td><td>59.01</td></tr><tr><td>DMD [55]</td><td>70.26</td><td>46.18</td><td>39.84</td><td>74.78</td><td>72.45</td><td>52.70</td><td>84.78</td><td>63.00</td></tr><tr><td>MCTN [25]</td><td>75.50</td><td>62.72</td><td>59.46</td><td>76.64</td><td>77.13</td><td>64.84</td><td>81.75</td><td>71.15</td></tr><tr><td>TransM [56]</td><td>77.98</td><td>63.68</td><td>58.67</td><td>80.46</td><td>78.61</td><td>62.24</td><td>81.48</td><td>71.87</td></tr><tr><td>SMIL [57]</td><td>76.57</td><td>65.96</td><td>60.57</td><td>77.68</td><td>76.24</td><td>66.87</td><td>80.74</td><td>72.09</td></tr><tr><td>GCNet [27]</td><td>80.52</td><td>66.54</td><td>61.83</td><td>81.96</td><td>81.15</td><td>69.21</td><td>82.35</td><td>74.79</td></tr><tr><td>CorrKD [13]</td><td>80.76 66.09</td><td></td><td>62.30</td><td>81.74</td><td>81.28</td><td>71.92</td><td>82.16</td><td>75.18</td></tr><tr><td>SemMSA</td><td>83.61</td><td>66.59</td><td>65.91</td><td>85.17</td><td>84.96</td><td>72.68</td><td>86.32</td><td>77.89</td></tr></table>

sentiment semantics in textual information. (3) SemMSA achieves the best results under all intermodal missingness, outperforming the second-best CorrKD [13] by a significant margin of 2.71% on average. Improvements in the language missingness, such as {a,v}, demonstrate that SemMSA effectively leverages nonverbal evidence to obtain rich sentiment-relevant semantics by semantic refinement, fully integrating them via anchor-free spectral alignment.

Table 4: Ablation study of components and losses on MOSI and SIMS datasets.
<table><tr><td rowspan="2"></td><td rowspan="2">CSR Lcsa</td><td rowspan="2"> $\pmb { \mathcal { L } } _ { \mathbf { s e p } }$ </td><td colspan="6">MOSI</td><td colspan="6">SIMS</td></tr><tr><td>Acc-2</td><td>F1</td><td>Acc-5</td><td>Acc-7</td><td>MAE↓</td><td>Corr ↑</td><td>Acc-2</td><td>F1</td><td>Acc-3</td><td>Acc-5</td><td>MAE↓</td><td>Corr ↑</td></tr><tr><td></td><td></td><td></td><td>67.49/69.47</td><td>63.28/67.01</td><td>34.29</td><td>31.71</td><td>1.085</td><td>0.501</td><td>66.83</td><td>57.92</td><td>31.12</td><td>54.61</td><td>0.593</td><td>0.387</td></tr><tr><td>√</td><td></td><td></td><td>73.36/73.29</td><td>73.04/73.28</td><td>38.84</td><td>35.03</td><td>1.040</td><td>0.522</td><td>74.18</td><td>76.21</td><td>34.27</td><td>57.36</td><td>0.488</td><td>0.450</td></tr><tr><td>√</td><td>√</td><td></td><td>73.94/73.56</td><td>73.83/73.46</td><td>40.48</td><td>36.08</td><td>1.024</td><td>0.529</td><td>74.92</td><td>76.88</td><td>35.02</td><td>58.12</td><td>0.480</td><td>0.493</td></tr><tr><td>√</td><td>√</td><td>√</td><td>74.36/73.91</td><td>74.18/73.82</td><td>40.93</td><td>36.49</td><td>1.011</td><td>0.550</td><td>75.46</td><td>77.50</td><td>35.68</td><td>58.84</td><td>0.474</td><td>0.501</td></tr></table>

## 4.3 Model Analysis

Ablation Study. We conduct an ablation study on MOSI and SIMS to evaluate the effectiveness of the proposed CSR, $\mathcal { L } _ { c s a } ,$ and $\mathcal { L } _ { s e p } ,$ as shown in Table 4. First, introducing CSR alone brings substantial gains over the baseline on both datasets, improving Acc-2 and F1 by 5.68% and 11.11% on average. These indicate that rich sentiment-relevant semantics effectively compensate for incomplete inputs. Second, adding $\mathcal { L } _ { \mathrm { c s a } }$ further brings performance gains. MOSI Acc-5 increases from 38.84 to 40.48 and Acc-7 from 35.03 to 36.08, suggesting better cross-modal consistency. Finally, the best performance is achieved when $\mathcal { L } _ { s e p }$ is also incorporated. The full model consistently outperforms all ablated variants, highlighting the importance of semantic refinement for enhancing discriminative sentiment semantics, as well as the complementarity between intra-instance cross-modal coherence by Lcsa and inter-instance discriminability by Lsep.

Table 5: Comparison with different projection and alignment methods on SIMS dataset.  
(a) Projection mechanisms
<table><tr><td>Method</td><td>Acc-2</td><td>F1</td><td>Acc-5 Acc-3</td><td>MAE</td><td>Corr ↑</td></tr><tr><td>Linear</td><td>72.84</td><td>73.62</td><td>54.91</td><td>32.46 0.512</td><td>0.431</td></tr><tr><td>MLP</td><td>73.58</td><td>74.36</td><td>56.03</td><td>33.42 0.501</td><td>0.452</td></tr><tr><td>Trans. [58]</td><td>74.52</td><td>76.21</td><td>57.63</td><td>34.76 0.486</td><td>0.482</td></tr><tr><td>QFormer [59]</td><td>74.83</td><td>76.68</td><td>58.12</td><td>35.09 0.481</td><td>0.491</td></tr><tr><td>Ours</td><td>75.46</td><td>77.50</td><td>58.84</td><td>35.68 0.474</td><td>0.501</td></tr></table>

(b) Alignment mechanisms
<table><tr><td>Method</td><td>Acc-2</td><td>F1</td><td>Acc-5</td><td>Acc-3 MAE↓</td><td>Corr ↑</td></tr><tr><td>InfoNCE [60]</td><td>73.42</td><td>75.39</td><td>56.88</td><td>33.74 0.501</td><td>0.489</td></tr><tr><td>CMD [44]</td><td>73.77</td><td>75.18</td><td>56.92</td><td>34.21 0.503</td><td>0.487</td></tr><tr><td>PMRL [40]</td><td>74.63</td><td>75.98</td><td>57.62</td><td>34.58 0.493</td><td>0.499</td></tr><tr><td>Volume [39]</td><td>74.31</td><td>76.35</td><td>57.91</td><td>34.86 0.487</td><td>0.506</td></tr><tr><td>Ours</td><td>75.46</td><td>77.50</td><td>58.84</td><td>35.68 0.474</td><td>0.501</td></tr></table>

Comparison with Projection Methods. As shown in Table 5a, we compare different projection strategies by replacing our Adapter with representative alternatives. Directly projecting modality features into the LLM embedding space with linear or MLP leads to a clear performance drop, indicating that dimensional matching alone is insufficient for constructing effective multimodal prefixes. Despite the effectiveness of Standard Transformer (∼85M) [58] and Q-Former (∼100M) [59], they introduce substantially higher computational overhead. In contrast, our lightweight Adapter with only 4.7M parameters achieves the best overall results, indicating that it can compactly aggregate sequence-level modal evidence and adaptively select reliable clues under incomplete observations.

Comparison with Alignment Methods. As shown in Table 5b, we compare four cross-modal alignment strategies. The baseline is utilized without an alignment objective. InfoNCE [60] and CMD [44] rely on pairwise similarity, failing to model the joint structure across all three modalities, resulting in poor results. While the PMRL [40] and volume-based loss improve upon this, they remain limited to the linear space and anchor modality, struggling to capture complex semantic relationships and select a stable anchor under severe intra-modality missingness. In contrast, our method introduces a Reproducing Kernel Hilbert Space (RKHS) without anchor dependency to enable nonlinear and robust alignment, achieving the best results.

Table 6: Efficiency comparison on MOSI.
<table><tr><td>Method</td><td>Trainable Params</td><td>Task GFLOPs</td><td>Memory(GB)</td><td>Latency(ms)</td><td>Acc-2(%)</td><td>F1(%)</td></tr><tr><td>LNLN [7]</td><td>116M</td><td>9.0</td><td>12.8</td><td>25.1</td><td>70.94/72.55</td><td>71.25/72.73</td></tr><tr><td>P-RMF [8]</td><td>117M</td><td>9.7</td><td>13.5</td><td>67.0</td><td>71.53/72.81</td><td>71.69/72.93</td></tr><tr><td>SemMSA</td><td>113M</td><td>4.8</td><td>13.0</td><td>30.6</td><td>74.36/73.91</td><td>74.18/73.82</td></tr></table>

Comparison of Computational Overhead. Efficiency experiments are further conducted on MOSI using the same server configuration as LNLN [7] and P-RMF [8], where trainable parameters and GFLOPs include only the modules optimized for MSA. As shown in Table 6, SemMSA achieves the best performance while maintaining a comparable computation cost. It reduces the task GFLOPs by 50.5% compared with LNLN and inference time by 54.3% compared with P-RMF. Notably, the reported memory and latency reflect the entire pipeline with the frozen LLM, rather than just the trainable modules. The efficiency mainly comes from the token-efficient hidden-state refinement design. SemMSA avoids autoregressive decoding of explicit textual descriptions and appends only a small number of continuous latent states. These results indicate that SemMSA provides a favorable trade-off between robustness and inference cost under incomplete multimodal settings.

Visualization of Prediction Performance. To further evaluate the robustness of SemMSA, we visualize the confusion matrices on the MOSI dataset under missing rates of 0.1, 0.5, 0.7, and 0.9, as shown in Figure 4. Under high missing rates, LNLN tends to wrongly concentrate predictions around middle sentiment classes. In contrast, SemMSA maintains clearer diagonal patterns across sentiment categories. This demonstrates that the semantic-aware modeling in SemMSA facilitates the capture of discriminative sentiment cues from incomplete inputs, alleviating classification boundary degeneration for robust prediction under severe modality missingness.

![](images/463a5dac5bdf487ca37c7bb8c2281c77ade4f5c2f3eb5ae84f3b324cff6fe58b.jpg)

![](images/965c0e3d03b62c8fcc121c53d020445dfa100163d62fb97843270db5638f793e.jpg)

![](images/e2611a79bb6e053f708eaaed31eab5593ca8d7fdf6aeb09014b6ae34834ee3a5.jpg)

![](images/79c0834f3046378af89bb97655142beffd07971cc7344c5b722e98608b39717a.jpg)

![](images/3bf2240d632fa1cb85ab1df1e371f0a31ebf531ed1c45bcf4db4a28d391af66b.jpg)  
Missing Rate = 0.1

![](images/a36e3e5c06a19700627167ee02b4b048315733109becc1d040158b5a41638eaf.jpg)  
Missing Rate = 0.5

![](images/7dd73b2970432457f53861d04c460b8285417a32ba852bb5460969936a3d057c.jpg)  
Missing Rate = 0.7

![](images/d6359781d8d1d456e95129c8c0f43af116913407d75a7a11a373431136993adc.jpg)  
Missing Rate = 0.9  
Figure 4: Confusion matrices of SemMSA and LNLN on the MOSI, where 0-6 denote strongly negative, negative, weakly negative, neutral, weakly positive, positive, and strongly positive, respectively.

Table 7: Comparison with LLMs for prediction and semantic generation on MOSI and SIMS.
<table><tr><td rowspan="2">Model</td><td rowspan="2">Role</td><td colspan="6">MOSI</td><td colspan="6">SIMS</td></tr><tr><td>Acc-2</td><td>F1</td><td>Acc-5</td><td>Acc-7</td><td>MAE↓</td><td>Corr ↑</td><td>Acc-2</td><td>F1</td><td>Acc-5</td><td>Acc-3</td><td>MAE↓</td><td>Corr ↑</td></tr><tr><td>Qwen2.5-Omni-7B [61]</td><td>Prediction</td><td>69.84/71.36</td><td>69.92/71.58</td><td>36.94</td><td>33.21</td><td>1.086</td><td>0.497</td><td>70.83</td><td>70.26</td><td>52.74</td><td>31.92</td><td>0.548</td><td>0.341</td></tr><tr><td>Qwen3-1.7B [53]</td><td>Semantics</td><td>74.36/73.91</td><td>74.18/73.82</td><td>40.93</td><td>36.49</td><td>1.011</td><td>0.550</td><td>75.46</td><td>77.50</td><td>58.84</td><td>35.68</td><td>0.474</td><td>0.501</td></tr><tr><td>Llama3.1-8B [62]</td><td>Semantics</td><td>74.42/73.75</td><td>74.04/73.69</td><td>41.11</td><td>36.50</td><td>1.015</td><td>0.546</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Qwen3-8B [53]</td><td>Semantics</td><td>74.66/73.88</td><td>74.11/73.79</td><td>40.86</td><td>36.62</td><td>1.012</td><td>0.553</td><td>75.38</td><td>77.63</td><td>58.79</td><td>35.77</td><td>0.476</td><td>0.514</td></tr><tr><td>Qwen2.5-Omni-7B [61]</td><td>Semantics</td><td>74.52/74.03</td><td>74.26/73.76</td><td>41.05</td><td>36.55</td><td>1.008</td><td>0.548</td><td>75.59</td><td>77.42</td><td>58.96</td><td>35.61</td><td>0.471</td><td>0.512</td></tr></table>

The Effect of Different LLMs. Table 7 compares open-source Qwen3-1.7B, Llama-3.1-8B, Qwen3- 8B, and Qwen2.5-Omni-7B on MOSI and SIMS. For fair comparison, the direct prediction of LLMs strictly follows the same evaluation protocol and data splits for MSA with incomplete data. However, its performance is relatively limited with only 69.84/71.36 Acc-2 and 69.92/71.58 F1 on MOSI, showing the difficulty of adapting LLMs to sentiment prediction. In contrast, using LLMs for semantic generation brings clear improvements, since rich semantic information provides auxiliary high-level knowledge to compensate for scarce representations with missing data. Specifically, Qwen3-8B achieves the best F1 and Acc-3 on SIMS. Llama3.1-8B without support of Chinese and Qwen2.5-Omni-7B obtains superior results of Acc-5 and MAE on MOSI, respectively. These results show that SemMSA is robust across different LLMs and that LLMs are more effective as semantic generators than direct predictors for MSA with incomplete data.

## 5 Conclusion

In this paper, we propose SemMSA, a semantic-aided framework to enrich incomplete data with LLMderived sentiment-aware semantic representations and integrate all modalities through anchor-free spectral alignment. Specifically, we introduce Cross-modal Semantic Refinement (CSR), which maps visual and acoustic representations into the frozen LLM embedding space with lightweight Adapters. CSR then produces sentiment-relevant latent semantics iteratively through token-efficient hidden-state refinement, without decoding explicit text. We further propose Cross-modal Spectral Alignment (CSA), which jointly aligns semantic, language, visual, and acoustic representations by enhancing the dominant spectral component of their kernel Gram matrix. This captures global nonlinear crossmodal dependencies without relying on a predefined anchor modality, while preserving discriminative structure through instance-level spectral separation. Extensive experiments on SIMS, MOSI, and MOSEI datasets demonstrate the effectiveness of SemMSA, achieving state-of-the-art performance under diverse missing-modality settings.

## References

[1] Y. Fang, W. Huang, G. Wan, K. Su, and M. Ye, “Emoe: Modality-specific enhanced dynamic emotion experts,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 14 314– 14 324.

[2] S. Wu, D. He, X. Wang, L. Wang, and J. Dang, “Enriching multimodal sentiment analysis through textual emotional descriptions of visual-audio content,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 39, no. 2, 2025, pp. 1601–1609.

[3] Y.-H. H. Tsai, S. Bai, P. P. Liang, J. Z. Kolter, L.-P. Morency, and R. Salakhutdinov, “Multimodal transformer for unaligned multimodal language sequences,” in Proceedings of the 57th annual meeting of the association for computational linguistics, 2019, pp. 6558–6569.

[4] H. Zhang, Y. Wang, G. Yin, K. Liu, Y. Liu, and T. Yu, “Learning language-guided adaptive hyper-modality representation for multimodal sentiment analysis,” in The 2023 Conference on Empirical Methods in Natural Language Processing, 2023.

[5] P. Wang, Q. Zhou, Y. Wu, T. Chen, and J. Hu, “Dlf: Disentangled-language-focused multimodal sentiment analysis,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 39, 2025, pp. 21 180– 21 188.

[6] M. Li, D. Yang, Y. Lei, S. Wang, S. Wang, L. Su, K. Yang, Y. Wang, M. Sun, and L. Zhang, “A unified self-distillation framework for multimodal sentiment analysis with uncertain missing modalities,” in Proceedings ofthe AAAI conference on artificial intelligence, vol. 38, no. 9, 2024, pp. 10 074–10 082.

[7] H. Zhang, W. Wang, and T. Yu, “Towards robust multimodal sentiment analysis with incomplete data,” Advances in Neural Information Processing Systems, vol. 37, pp. 55 943–55 974, 2024.

[8] A. Zhu, M. Hu, X. Wang, J. Yang, Y. Tang, and N. An, “Proxy-driven robust multimodal sentiment analysis with incomplete data,” in Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 2025, pp. 22 123–22 138.

[9] Z. Yuan, W. Li, H. Xu, and W. Yu, “Transformer-based feature reconstruction network for robust multimodal sentiment analysis,” in Proceedings of the 29th ACM international conference on multimedia, 2021, pp. 4400–4407.

[10] L. Sun, Z. Lian, B. Liu, and J. Tao, “Efficient multimodal transformer with dual-level feature restoration for robust multimodal sentiment analysis,” IEEE Transactions on Affective Computing, vol. 15, no. 1, pp. 309–325, 2023.

[11] J. Zeng, T. Liu, and J. Zhou, “Tag-assisted multimodal sentiment analysis under uncertain missing modalities,” in Proceedings ofthe 45th International ACM SIGIR Conference on Research and Development in Information Retrieval, 2022, pp. 1545–1554.

[12] Y. Wang, Y. Li, and Z. Cui, “Incomplete multimodality-diffused emotion recognition,” Advances in Neural Information Processing Systems, vol. 36, pp. 17 117–17 128, 2023.

[13] M. Li, D. Yang, X. Zhao, S. Wang, Y. Wang, K. Yang, M. Sun, D. Kou, Z. Qian, and L. Zhang, “Correlationdecoupled knowledge distillation for multimodal sentiment analysis with incomplete modalities,” in Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 12 458– 12 468.

[14] D. Kim and T. Kim, “Missing modality prediction for unpaired multimodal learning via joint embedding of unimodal models,” in European Conference on Computer Vision. Springer, 2024, pp. 171–187.

[15] M. C. Camacho, A. N. Nielsen, D. Balser, E. Furtado, D. C. Steinberger, L. Fruchtman, J. P. Culver, C. M. Sylvester, and D. M. Barch, “Large-scale encoding of emotion concepts becomes increasingly similar between individuals from childhood to adolescence,” Nature neuroscience, vol. 26, no. 7, pp. 1256–1266, 2023.

[16] J. Ortega, Y. Murai, and D. Whitney, “Integration of affective cues in context-rich and dynamic scene varies across individuals,” Nature Communications, vol. 17, p. 786, 2025.

[17] Z. Lian, X. Peng, K. Xu, Z. Jia, X. Che, Z. Cheng, F. Ma, L. Cui, Y. Zhang, X. Liu et al., “Mer 2026: From discriminative emotion recognition to generative emotion understanding,” arXiv preprint arXiv:2604.19417, 2026.

[18] A. Zadeh, M. Chen, S. Poria, E. Cambria, and L.-P. Morency, “Tensor fusion network for multimodal sentiment analysis,” in Proceedings of the 2017 conference on empirical methods in natural language processing, 2017, pp. 1103–1114.

[19] S. Mai, S. Xing, and H. Hu, “Locally confined modality fusion network with a global perspective for multimodal human affective computing,” IEEE Transactions on Multimedia, vol. 22, pp. 122–137, 2019.

[20] F. Lv, X. Chen, Y. Huang, L. Duan, and G. Lin, “Progressive modality reinforcement for human multimodal emotion recognition from unaligned multimodal sequences,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2021, pp. 2554–2562.

[21] D. Yang, S. Huang, H. Kuang, Y. Du, and L. Zhang, “Disentangled representation learning for multimodal emotion recognition,” in Proceedings of the 30th ACM international conference on multimedia, 2022, pp. 1642–1651.

[22] H. Zhang, Y. Zhang, C. Ying, X. Tang, and T. Yu, “Improving task-specific multimodal sentiment analysis with general mllms via prompting,” in The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025.

[23] G. Andrew, R. Arora, J. Bilmes, and K. Livescu, “Deep canonical correlation analysis,” in International conference on machine learning. PMLR, 2013, pp. 1247–1255.

[24] W. Wang, R. Arora, K. Livescu, and J. Bilmes, “On deep multi-view representation learning,” in International conference on machine learning. PMLR, 2015, pp. 1083–1092.

[25] H. Pham, P. P. Liang, T. Manzini, L.-P. Morency, and B. Póczos, “Found in translation: Learning robust joint representations by cyclic translations between modalities,” in Proceedings ofthe AAAI conference on artificial intelligence, vol. 33, 2019, pp. 6892–6899.

[26] J. Zhao, R. Li, and Q. Jin, “Missing modality imagination network for emotion recognition with uncertain missing modalities,” in Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), 2021, pp. 2608–2618.

[27] Z. Lian, L. Chen, L. Sun, B. Liu, and J. Tao, “Gcnet: Graph completion network for incomplete multimodal learning in conversation,” IEEE Transactions on pattern analysis and machine intelligence, vol. 45, no. 7, pp. 8419–8432, 2023.

[28] Z. Liu, B. Zhou, D. Chu, Y. Sun, and L. Meng, “Modality translation-based multimodal sentiment analysis under uncertain missing modalities,” Information Fusion, vol. 101, p. 101973, 2024.

[29] L. Tran, X. Liu, J. Zhou, and R. Jin, “Missing modalities imputation via cascaded residual autoencoder,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2017, pp. 1405–1414.

[30] J. Li, D. Li, C. Xiong, and S. Hoi, “Blip: Bootstrapping language-image pre-training for unified visionlanguage understanding and generation,” in International conference on machine learning. PMLR, 2022, pp. 12 888–12 900.

[31] B. Zhu, Y. Niu, Y. Han, Y. Wu, and H. Zhang, “Prompt-aligned gradient for prompt tuning,” in Proceedings ofthe IEEE/CVF international conference on computer vision, 2023, pp. 15 659–15 669.

[32] X. Zhai, B. Mustafa, A. Kolesnikov, and L. Beyer, “Sigmoid loss for language image pre-training,” in Proceedings of the IEEE/CVF international conference on computer vision, 2023, pp. 11 975–11 986.

[33] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark et al., “Learning transferable visual models from natural language supervision,” in International conference on machine learning. PmLR, 2021, pp. 8748–8763.

[34] B. Elizalde, S. Deshmukh, M. Al Ismail, and H. Wang, “Clap learning audio concepts from natural language supervision,” in ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2023, pp. 1–5.

[35] R. Zhang, Z. Guo, W. Zhang, K. Li, X. Miao, B. Cui, Y. Qiao, P. Gao, and H. Li, “Pointclip: Point cloud understanding by clip,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 8552–8562.

[36] Y. Ma, G. Xu, X. Sun, M. Yan, J. Zhang, and R. Ji, “X-clip: End-to-end multi-grained contrastive learning for video-text retrieval,” in Proceedings of the 30th ACM international conference on multimedia, 2022, pp. 638–647.

[37] J. Pennington and P. Worah, “Nonlinear random matrix theory for deep learning,” Advances in neural information processing systems, vol. 30, 2017.

[38] I. Nejjar, Q. Wang, and O. Fink, “Dare-gram: Unsupervised domain adaptation regression by aligning inverse gram matrices,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2023, pp. 11 744–11 754.

[39] G. Cicchetti, E. Grassucci, L. Sigillo, and D. Comminiello, “Gramian multimodal representation learning and alignment,” in The Thirteenth International Conference on Learning Representations, 2025.

[40] X. Liu, X. Xia, S.-K. Ng, and T.-S. Chua, “Principled multimodal representation learning,” IEEE Transactions on Pattern Analysis and Machine Intelligence, 2026.

[41] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, “Bert: Pre-training of deep bidirectional transformers for language understanding,” in Proceedings of the 2019 conference of the North American chapter of the association for computational linguistics: human language technologies, volume 1 (long and short papers), 2019, pp. 4171–4186.

[42] B. McFee, C. Raffel, D. Liang, D. P. Ellis, M. McVicar, E. Battenberg, and O. Nieto, “librosa: Audio and music signal analysis in python.” SciPy, vol. 2015, pp. 18–24, 2015.

[43] T. Baltrušaitis, P. Robinson, and L.-P. Morency, “Openface: an open source facial behavior analysis toolkit,” in 2016 IEEE winter conference on applications ofcomputer vision (WACV). IEEE, 2016, pp. 1–10.

[44] D. Hazarika, R. Zimmermann, and S. Poria, “Misa: Modality-invariant and-specific representations for multimodal sentiment analysis,” in Proceedings ofthe 28th ACM international conference on multimedia, 2020, pp. 1122–1131.

[45] W. Yu, H. Xu, Z. Yuan, and J. Wu, “Learning modality-specific representations with self-supervised multi-task learning for multimodal sentiment analysis,” in Proceedings ofthe AAAI conference on artificial intelligence, vol. 35, 2021, pp. 10 790–10 797.

[46] W. Han, H. Chen, and S. Poria, “Improving multimodal fusion with hierarchical mutual information maximization for multimodal sentiment analysis,” in Proceedings of the 2021 conference on empirical methods in natural language processing, 2021, pp. 9180–9192.

[47] D. Wang, X. Guo, Y. Tian, J. Liu, L. He, and X. Luo, “Tetfn: A text enhanced transformer fusion network for multimodal sentiment analysis,” Pattern Recognition, vol. 136, p. 109259, 2023.

[48] X. Li, X. Cheng, D. Miao, X. Zhang, and Z. Li, “Tf-mamba: Text-enhanced fusion mamba with missing modalities for robust multimodal sentiment analysis,” arXiv preprint arXiv:2505.14329, 2025.

[49] A. Zadeh, R. Zellers, E. Pincus, and L.-P. Morency, “Mosi: multimodal corpus of sentiment intensity and subjectivity analysis in online opinion videos,” arXiv preprint arXiv:1606.06259, 2016.

[50] A. B. Zadeh, P. P. Liang, S. Poria, E. Cambria, and L.-P. Morency, “Multimodal language analysis in the wild: Cmu-mosei dataset and interpretable dynamic fusion graph,” in Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 2018, pp. 2236–2246.

[51] W. Yu, H. Xu, F. Meng, Y. Zhu, Y. Ma, J. Wu, J. Zou, and K. Yang, “Ch-sims: A chinese multimodal sentiment analysis dataset with fine-grained annotation of modality,” in Proceedings of the 58th annual meeting of the association for computational linguistics, 2020, pp. 3718–3727.

[52] I. Loshchilov and F. Hutter, “Decoupled weight decay regularization,” in International Conference on Learning Representations, 2019.

[53] A. Yang, A. Li, B. Yang, B. Zhang, B. Hui, B. Zheng, B. Yu, C. Gao, C. Huang, C. Lv et al., “Qwen3 technical report,” arXiv preprint arXiv:2505.09388, 2025.

[54] H. Sun, H. Wang, J. Liu, Y.-W. Chen, and L. Lin, “Cubemlp: An mlp-based model for multimodal sentiment analysis and depression estimation,” in Proceedings of the 30th ACM international conference on multimedia, 2022, pp. 3722–3729.

[55] Y. Li, Y. Wang, and Z. Cui, “Decoupled multimodal distilling for emotion recognition,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2023, pp. 6631–6640.

[56] Z. Wang, Z. Wan, and X. Wan, “Transmodality: An end2end fusion method with transformer for multimodal sentiment analysis,” in Proceedings of the web conference 2020, 2020, pp. 2514–2520.

[57] M. Ma, J. Ren, L. Zhao, S. Tulyakov, C. Wu, and X. Peng, “Smil: Multimodal learning with severely missing modality,” in Proceedings of the AAAI conference on artificial intelligence, vol. 35, 2021, pp. 2302–2310.

[58] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin, “Attention is all you need,” Advances in neural information processing systems, vol. 30, 2017.

[59] J. Li, D. Li, S. Savarese, and S. Hoi, “Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models,” in International conference on machine learning. PMLR, 2023, pp. 19 730–19 742.

[60] A. v. d. Oord, Y. Li, and O. Vinyals, “Representation learning with contrastive predictive coding,” arXiv preprint arXiv:1807.03748, 2018.

[61] J. Xu, Z. Guo, J. He, H. Hu, T. He, S. Bai, K. Chen et al., “Qwen2.5-omni technical report,” arXiv preprint arXiv:2503.20215, 2025.

[62] A. Grattafiori, A. Dubey, A. Jauhri, A. Pandey, A. Kadian, A. Al-Dahle et al., “The llama 3 herd of models,” arXiv preprint arXiv:2407.21783, 2024.

[63] X. Zhang, W. Wei, and S. Zou, “Modal feature optimization network with prompt for multimodal sentiment analysis,” in Proceedings of the 31st International Conference on Computational Linguistics, 2025, pp. 4611–4621.

[64] S. Mai, Y. Zeng, S. Zheng, and H. Hu, “Hybrid contrastive learning of tri-modal representation for multimodal sentiment analysis,” IEEE transactions on affective computing, vol. 14, no. 3, pp. 2276–2289, 2022.

[65] J. Yang, Y. Yu, D. Niu, W. Guo, and Y. Xu, “Confede: Contrastive feature decomposition for multimodal sentiment analysis,” in Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 2023, pp. 7617–7630.

[66] A. Zhu, M. Hu, X. Wang, J. Yang, Y. Tang, and F. Ren, “Kebr: Knowledge enhanced self-supervised balanced representation for multimodal sentiment analysis,” in Proceedings ofthe 32nd ACM International Conference on Multimedia, 2024, pp. 5732–5741.

[67] Y. Zhuang, Y. Zhang, Z. Hu, X. Zhang, J. Deng, and F. Ren, “Glomo: Global-local modal fusion for multimodal sentiment analysis,” in Proceedings ofthe 32nd ACM International Conference on Multimedia, 2024, pp. 1800–1809.

## A Appendix / supplemental material

## A.1 Training Algorithm

```latex
Algorithm 1: Training algorithm of the proposed SemMSA.
Input: Training set $\mathcal { D } _ { t r a i n } ,$ frozen modality encoders $f _ { \phi } ^ { m }$ , frozen LLM $F _ { \theta } ,$ visual/audio
Adapters $A _ { v } , A _ { a } ,$ modality encoders $\mathcal { E } _ { m } ,$ semantic projector $\{ W _ { s } , b _ { s } \}$ , classifier $\mathcal { C }$
while not converged do
1. Sample a batch $\boldsymbol { B } = \{ ( U _ { v } ^ { i } , U _ { a } ^ { i } , U _ { l } ^ { i } , y ^ { i } ) \} _ { i = 1 } ^ { N }$ from $\mathcal { D } _ { t r a i n }$ and apply instance-wise
missing-modality masking;
2. Extract modality features $X _ { m } ^ { i }$ according to Eq. 1;
3. Feed $X _ { v } ^ { i }$ and $\hat { X _ { a } ^ { i } }$ into visual/audio Adapters to obtain compact prefix tokens $Z _ { v } ^ { i }$ and $Z _ { a } ^ { i } ,$
and embed language input as $Z _ { l } ^ { i } ;$
4. Construct the initial multimodal prefix $U ^ { i , ( 0 ) } = [ Z _ { l } ^ { i } ; Z _ { v } ^ { i } ; Z _ { a } ^ { i } ]$ in the frozen LLM
embedding space;
for $k = 1$ to O do
5. Obtain the latent semantic state $z _ { k } ^ { i }$ from the final-position hidden state of $F _ { \theta } ,$ and
update $U ^ { i , ( k ) }$ according to Eq. 2 and Eq. 3;
6. Collect $Z ^ { i }$ and obtain the fused semantic representation according to Eq. 4 and $\operatorname { E q . } 5 ;$
7. Extract visual, acoustic, and language features to obtain $H _ { v } ^ { i } , H _ { a } ^ { i } .$ , and $\hat { H _ { l } ^ { i } }$
8. Normalize $H _ { s } ^ { i } , H _ { v } ^ { i } , H _ { a } ^ { i } , H _ { l } ^ { i }$ and construct $V ^ { i } = [ h _ { s } ^ { i } , h _ { v } ^ { i } , h _ { a } ^ { i } , h _ { l } ^ { i } ] ;$
9. Build the kernel Gram matrix $K ^ { i }$ and perform eigendecomposition according to
Eq. 6–Eq. 8;
10. Compute the cross-modal spectral alignment loss $\mathcal { L } _ { \mathrm { c s a } }$ by enhancing the dominance of $\lambda _ { 1 } ^ { i }$
according to Eq. 9;
11. Compute the spectral separation loss $\mathcal { L } _ { \mathrm { s e p } }$ from dominant semantic directions $\{ u _ { 1 } ^ { i } \} _ { i = : } ^ { N }$ 1
according to Eq. 10;
12. Fuse all representations as ${ \hat { H } } ^ { i }$ and predict sentiment score $\hat { y } ^ { i }$ according to Eq. 11;
13. Calculate $\bar { \mathcal { L } } _ { \mathrm { t o t a l } }$ with the sentiment regression loss, ${ \mathcal { L } } _ { \mathrm { c s a } } ,$ and $\mathcal { L } _ { \mathrm { s e p } }$ according to Eq. 12;
14. Update all trainable parameters via gradient backpropagation.
Output: Trained SemMSA model for MSA with incomplete data.
```

We describe the training procedure of SemMSA in Algorithm 1. Given incomplete multimodal inputs, SemMSA first extracts modality-specific features and maps visual and acoustic evidence into the frozen LLM embedding space through lightweight Adapters. Then, Cross-modal Semantic Refinement (CSR) iteratively appends hidden semantic states from the frozen LLM to produce compact sentiment-aware semantics without explicit text decoding. After that, Cross-modal Spectral Alignment (CSA) constructs a kernel Gram matrix over semantic, visual, acoustic, and language representations, and enhances its dominant spectral component for anchor-free multimodal alignment. An instance-level spectral separation loss is further introduced to preserve discriminability across samples. Finally, all representations are fused, and the model is optimized for sentiment prediction.

## A.2 Theoretical Analysis

In this section, we provide a theoretical analysis of the proposed Cross-modal Spectral Alignment (CSA). Different from pairwise contrastive alignment, CSA aligns semantic, visual, acoustic, and language representations simultaneously by analyzing the spectral structure of their kernel Gram matrix. The analysis shows that enhancing the dominant spectral component of the kernel Gram matrix encourages all modalities to concentrate around a shared nonlinear semantic direction in the RKHS, thereby achieving anchor-free cross-modal alignment.

## A.2.1 Preliminaries

For the i-th multimodal instance, SemMSA obtains four normalized representations:

$$
V ^ { i } = [ h _ { s } ^ { i } , h _ { v } ^ { i } , h _ { a } ^ { i } , h _ { l } ^ { i } ] ,\tag{13}
$$

where $h _ { s } ^ { i } , h _ { v } ^ { i } , h _ { a } ^ { i }$ , and $h _ { l } ^ { i }$ denote the semantic, visual, acoustic, and language representations, respectively. For simplicity, we omit the instance index i and denote the k modality representations as $\{ h _ { 1 } , \dot { h } _ { 2 } , \dots , \dot { h } _ { k } \}$ CSA maps these representations into an RKHS F through a nonlinear feature map $\psi ( \cdot )$ induced by a positive definite kernel $\kappa ( \cdot , \cdot )$

$$
\kappa ( h _ { p } , h _ { q } ) = \langle \psi ( h _ { p } ) , \psi ( h _ { q } ) \rangle _ { \mathcal { F } } .\tag{14}
$$

In SemMSA, we adopt the RBF kernel:

$$
\kappa ( h _ { p } , h _ { q } ) = \exp \left( { - \frac { \| h _ { p } - h _ { q } \| _ { 2 } ^ { 2 } } { 2 \sigma ^ { 2 } } } \right) .\tag{15}
$$

The corresponding kernel Gram matrix is defined as

$$
K = \left[ \begin{array} { c c c } { \kappa ( h _ { 1 } , h _ { 1 } ) } & { \cdots } & { \kappa ( h _ { 1 } , h _ { k } ) } \\ { \vdots } & { \ddots } & { \vdots } \\ { \kappa ( h _ { k } , h _ { 1 } ) } & { \cdots } & { \kappa ( h _ { k } , h _ { k } ) } \end{array} \right] \in \mathbb { R } ^ { k \times k } .\tag{16}
$$

Since the RBF kernel satisfies $\kappa ( h _ { p } , h _ { p } ) = 1$ , we have

$$
\operatorname { T r } ( K ) = k .\tag{17}
$$

As K is symmetric positive semi-definite, it admits the eigendecomposition:

$$
K = Q \Lambda Q ^ { \top } , \quad \Lambda = \mathrm { d i a g } ( \lambda _ { 1 } , \lambda _ { 2 } , \ldots , \lambda _ { k } ) ,\tag{18}
$$

where $\lambda _ { 1 } \geq \lambda _ { 2 } \geq \cdot \cdot \cdot \geq \lambda _ { k } \geq 0$ and $\sum _ { j = 1 } ^ { k } \lambda _ { j } = k$

## A.2.2 Kernelized Full Alignment and Rank-one Gram Matrix

Definition 1 (Kernelized full alignment). The modality representations $\{ h _ { j } \} _ { j = 1 } ^ { k }$ arefully aligned in the RKHS iftheir kernel embeddings are identical:

$$
\psi ( h _ { 1 } ) = \psi ( h _ { 2 } ) = \cdot \cdot \cdot = \psi ( h _ { k } ) .\tag{19}
$$

Lemma 1 (Kernelized full alignment ⇐⇒ rank-one kernel Gram matrix). Let $K \in \mathbb { R } ^ { k \times k }$ be the kernel Gram matrix constructed from $\{ \psi ( h _ { j } ) \} _ { j = 1 } ^ { k }$ with $\kappa ( h _ { j } , h _ { j } ) = 1$ . Then the following two statements are equivalent:

$$
\psi ( h _ { 1 } ) = \psi ( h _ { 2 } ) = \cdot \cdot \cdot = \psi ( h _ { k } ) ,\tag{20}
$$

and

$$
\operatorname { r a n k } ( K ) = 1 , \quad K = \mathbf { 1 1 } ^ { \top } .\tag{21}
$$

Proof. If all modality embeddings are fully aligned in the RKHS, then for any $p , q \in \{ 1 , \ldots , k \}$ },

$$
K _ { p q } = \langle \psi ( h _ { p } ) , \psi ( h _ { q } ) \rangle _ { \mathcal { F } } = \langle \psi ( h _ { p } ) , \psi ( h _ { p } ) \rangle _ { \mathcal { F } } = 1 ,\tag{22}
$$

where the last equality follows from $\kappa ( h _ { p } , h _ { p } ) = 1$ . Therefore, $K = \mathbf { 1 1 } ^ { \top }$ , which is a rank-one matrix.

Conversely, suppose rank $( K ) = 1$ and $\kappa ( h _ { j } , h _ { j } ) = 1$ for all $j .$ Since K is symmetric positive semi-definite, there exists a vector $r \in \mathbb { R } ^ { k }$ such that

$$
K = r r ^ { \top } .\tag{23}
$$

The diagonal constraint $K _ { j j } = 1$ implies $r _ { i } ^ { 2 } = 1$ for all $j$ . For the RBF kernel, all entries satisfy $K _ { p q } > 0$ . Thus $r _ { p } r _ { q } > 0$ for any p, q, meaning all elements of r share the same sign. Consequently,

$$
K _ { p q } = r _ { p } r _ { q } = 1 , \quad \forall p , q .\tag{24}
$$

Hence $K = { \bf 1 1 } ^ { \top }$ . Since

$$
| | \psi ( h _ { p } ) - \psi ( h _ { q } ) | | _ { \mathcal { F } } ^ { 2 } = K _ { p p } + K _ { q q } - 2 K _ { p q } = 1 + 1 - 2 = 0 ,\tag{25}
$$

we obtain $\psi ( h _ { p } ) = \psi ( h _ { q } )$ for all modality pairs. Therefore, all modalities are fully aligned in the RKHS. □

Lemma 1 shows that the goal of nonlinear cross-modal alignment can be transformed into encouraging the kernel Gram matrix to approach a rank-one structure. This provides the theoretical basis for CSA.

## A.2.3 Dominant Spectral Component Encourages Nonlinear Alignment

Theorem 1 (Dominant eigenvalue maximization promotes kernelized cross-modal alignment). Let K be the kernel Gram matrix of k normalized modality representations in the RKHS, with eigenvalues $\lambda _ { 1 } \geq \lambda _ { 2 } \geq \cdot \cdot \cdot \geq \lambda _ { k } \geq 0$ and $\textstyle \sum _ { j = 1 } ^ { k } \lambda _ { j } = k .$ . Then K is rank-one if and only if

$$
\lambda _ { 1 } = k , \quad \lambda _ { 2 } = \cdot \cdot \cdot = \lambda _ { k } = 0 .\tag{26}
$$

Moreover, increasing the dominance ratio $o f \lambda _ { 1 }$ reduces the residual spectral energy

$$
\sum _ { j = 2 } ^ { k } \lambda _ { j } = k - \lambda _ { 1 } ,\tag{27}
$$

thereby pushing K toward its optimal rank-one approximation.

Proof. Since K is positive semi-definite, all eigenvalues are non-negative. The rank of K equals the number of non-zero eigenvalues. Thus, K is rank-one if and only if exactly one eigenvalue is non-zero. Because $\begin{array} { r } { \mathrm { T r } ( K ) = \sum _ { j = 1 } ^ { k } \lambda _ { j } = k } \end{array}$ , the only possible rank-one spectrum is

$$
\lambda _ { 1 } = k , \quad \lambda _ { 2 } = \cdot \cdot \cdot = \lambda _ { k } = 0 .\tag{28}
$$

For the second statement, the residual spectral energy outside the dominant component is

$$
\sum _ { j = 2 } ^ { k } \lambda _ { j } = \sum _ { j = 1 } ^ { k } \lambda _ { j } - \lambda _ { 1 } = k - \lambda _ { 1 } .\tag{29}
$$

Therefore, increasing $\lambda _ { 1 }$ directly decreases the total energy of all non-dominant spectral components. When $\lambda _ { 1 }$ approaches $k ,$ the remaining eigenvalues approach zero, and K approaches a rank-one matrix. According to Lemma 1, this corresponds to full cross-modal alignment in the RKHS.

Theorem 1 justifies the CSA objective in SemMSA. Specifically, CSA treats the eigenvalues of K as logits and enhances the dominance of the largest eigenvalue:

$$
\mathcal { L } _ { \mathrm { c s a } } = - \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \log \frac { \exp ( \lambda _ { 1 } ^ { i } / \tau ) } { \sum _ { j = 1 } ^ { k } \exp ( \lambda _ { j } ^ { i } / \tau ) } .\tag{30}
$$

Minimizing $\mathcal { L } _ { \mathrm { c s a } }$ increases the relative dominance of $\lambda _ { 1 } ^ { i }$ over the remaining eigenvalues. As a result, the kernel Gram matrix becomes closer to a rank-one matrix, encouraging semantic, visual, acoustic, and language representations to align along a shared nonlinear semantic direction.

## A.2.4 Anchor-free Property

Pairwise contrastive learning usually selects one modality as the anchor and aligns the remaining modalities to it. In contrast, CSA does not require a predefined anchor modality. The dominant eigenvector $q _ { 1 }$ of K is determined by the joint spectral structure of all modalities:

$$
K q _ { 1 } = \lambda _ { 1 } q _ { 1 } .\tag{31}
$$

Thus, the shared alignment direction is induced by all modality representations rather than by any single modality. The corresponding dominant semantic direction in the representation space is computed as

$$
u _ { 1 } = \frac { V q _ { 1 } } { \| V q _ { 1 } \| _ { 2 } } .\tag{32}
$$

This direction adaptively summarizes the principal cross-modal consensus of each instance. Therefore, CSA performs any-to-any alignment by enhancing a data-dependent dominant spectral component, avoiding the instability caused by manually selecting an anchor modality.

Table 8: Statistics of the multimodal sentiment analysis datasets.
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Speaker</td><td rowspan="2">Clip</td><td colspan="3">Sample</td><td rowspan="2">Language</td></tr><tr><td>Train</td><td>Valid</td><td>Test</td></tr><tr><td>MOSI [49]</td><td>93</td><td>2,199</td><td>1,284</td><td>229</td><td>686</td><td>English</td></tr><tr><td>MOSEI [50]</td><td>1,000</td><td>22,856</td><td>16,326</td><td>1,871</td><td>4,659</td><td>English</td></tr><tr><td>SIMS [51]</td><td>474</td><td>2,281</td><td>1,368</td><td>456</td><td>457</td><td>Chinese</td></tr></table>

## A.2.5 Instance-level Spectral Separation

Although maximizing the dominant eigenvalue promotes intra-instance cross-modal alignment, it may lead to a degenerate solution where different instances collapse to a similar semantic direction. To avoid this issue, SemMSA introduces an instance-level spectral separation loss:

$$
\mathcal { L } _ { \mathrm { s e p } } = \frac { 1 } { N ( N - 1 ) } \sum _ { i = 1 } ^ { N } \sum _ { j = 1 \atop j \neq i } ^ { N } \left[ ( u _ { 1 } ^ { i } ) ^ { \top } u _ { 1 } ^ { j } \right] ^ { 2 } .\tag{33}
$$

This loss penalizes high similarity between dominant semantic directions from different instances. Therefore, CSA simultaneously encourages:

$$
\mathrm { i n t r a - i n s t a n c e \ a l i g n m e n t : } \lambda _ { 1 } ^ { i } \gg \lambda _ { 2 } ^ { i } , . . . , \lambda _ { k } ^ { i } ,\tag{34}
$$

and

$$
\mathrm { i n t e r - i n s t a n c e ~ s e p a r a t i o n : ~ } ( u _ { 1 } ^ { i } ) ^ { \top } u _ { 1 } ^ { j }  0 , \quad i \neq j .\tag{35}
$$

The former ensures that different modalities of the same sample are aligned in the RKHS, while the latter preserves discriminability across samples and mitigates representation collapse.

The above analysis indicates that CSA has three desirable theoretical properties. First, by constructing a kernel Gram matrix, CSA extends linear Gram-based alignment to nonlinear RKHS alignment, enabling the model to capture complex cross-modal dependencies. Second, by enhancing the dominant eigenvalue, CSA explicitly encourages the kernel Gram matrix to approach a rank-one structure, which corresponds to full alignment of semantic, visual, acoustic, and language representations in the RKHS. Third, by using the dominant eigenvector rather than a predefined modality as the alignment direction, CSA realizes anchor-free multimodal alignment. Together with the instancelevel spectral separation loss, CSA learns representations that are both cross-modally coherent and instance-discriminative.

## A.3 Experiments

According to the granularity of missing elements, MSA with incomplete data can be categorized into inter-modal and intra-modal missingness. The former removes entire modalities, whereas the latter corrupts partial language tokens, visual frames, or acoustic segments within each modality. Intra-modal missingness often occurs randomly and may co-exist across multiple modalities, causing fragmented emotional cues and cross-modal inconsistency. Consequently, intra-modal missingness is considered more complex, mainly focused in this work.

## A.3.1 Dataset Details

SemMSA is evaluated on three standard benchmarks of MOSI [49], MOSEI [50], and SIMS [51].   
Detailed statistics of all datasets are summarized in Table 8.

MOSI. CMU-MOSI is a widely used benchmark for multimodal sentiment analysis. It contains 2,199 opinion-level video segments collected from online videos, where each sample is associated with three modalities: language, vision, and audio. Following the standard split, the dataset is divided into 1,284 training samples, 229 validation samples, and 686 test samples. Each sample is annotated with a real-valued sentiment score ranging from −3 to +3, where −3 denotes strongly negative sentiment and +3 denotes strongly positive sentiment. Due to its relatively small scale and fine-grained sentiment annotations, MOSI is commonly used to evaluate the robustness and generalization ability of MSA models under limited data conditions.

Table 9: Hyperparameters used on different datasets.
<table><tr><td>Hyperparameter</td><td>MOSI</td><td>MOSEI</td><td>SIMS</td></tr><tr><td>Vector Length T</td><td>8</td><td>8</td><td>8</td></tr><tr><td>Dimension d</td><td>128</td><td>128</td><td>128</td></tr><tr><td>Batch Size</td><td>64</td><td>64</td><td>64</td></tr><tr><td>Learning Rate</td><td>1e-4</td><td>1e-4</td><td>1e-4</td></tr><tr><td>Refine Step</td><td>4</td><td>4</td><td>4</td></tr><tr><td>Optimizer</td><td>AdamW</td><td>AdamW</td><td>AdamW</td></tr><tr><td>Epochs</td><td>200</td><td>200</td><td>200</td></tr><tr><td>Warmup</td><td>√</td><td>√</td><td>√</td></tr><tr><td>Cosine Annealing</td><td>√</td><td>√</td><td>√</td></tr><tr><td>Early Stop</td><td>√</td><td>√</td><td>√</td></tr><tr><td>Seed</td><td>1111,1112,1113</td><td>1111,1112,1113</td><td>1111,1112,1113</td></tr></table>

MOSEI. CMU-MOSEI is a large-scale multimodal sentiment analysis dataset built from YouTube video clips. It contains 22,856 annotated video segments covering diverse speakers, topics, and expression styles. Each segment includes language, visual, and acoustic information, making it suitable for evaluating multimodal representation learning in realistic scenarios. The dataset is split into 16,326 training samples, 1,871 validation samples, and 4,659 test samples. Similar to MOSI, each sample is labeled with a continuous sentiment score in the range of [−3, +3], where lower values indicate more negative sentiment and higher values indicate more positive sentiment. Compared with MOSI, MOSEI provides a larger and more diverse evaluation setting, which allows a more reliable assessment of model scalability and robustness.

SIMS. CH-SIMS is a Chinese multimodal sentiment analysis dataset collected from movies and TV series. It contains 2,281 video clips with aligned language, visual, and acoustic modalities. The dataset is partitioned into 1,368 training samples, 456 validation samples, and 457 test samples. Each sample is manually annotated with a sentiment score ranging from −1 to +1, where negative values indicate negative sentiment and positive values indicate positive sentiment. Different from MOSI and MOSEI, SIMS provides fine-grained annotations in Chinese multimodal scenarios, making it useful for evaluating whether a model can generalize across languages, domains, and cultural contexts.

## A.3.2 Baselines and Experimental Setup

We conduct a fair comparison with existing advanced and state-of-the-art methods, including Intramodal missingness: MISA [44], Self-MM [45], MMIM [46], TETFN [47], TFR-Net [9], ALMT [4]. LNLN [7], P-RMF [8], and TF-Mamba [48]; as well as Inter-modal missingness: CubeMLP [54], DMD [55], MCTN [25], TransM [56], SMIL [57], GCNet [27], CorrKD [13]. The results of intramodal missing baselines are reported as in [7, 8, 48]. The results of inter-modal missing baselines are adopted as in [27, 13]. This ensures fairness and consistency in the same evaluation settings across all compared methods. Our SemMSA model is trained using the PyTorch framework on an NVIDIA RTX 6000 Ada with 48GB of memory. Following prior MSA works [7–9], each modality is first processed by frozen modality-specific encoders $f _ { \phi }$ . Language is encoded by BERT [41], audio features are extracted by Librosa [42], and visual features are obtained by OpenFace [43]. Details of the hyperparameters are shown in Table 9.

## A.3.3 Performance Comparison under Complete-Modality MSA

Table 10 reports the comparison with state-of-the-art methods on MOSI and MOSEI under the complete modality setting. Following the training and evaluation in prior complete MSA works [63, 22], SemMSA achieves the best overall performance on both datasets, showing that the proposed framework remains highly effective even when all language, visual, and acoustic modalities are available. On MOSI, SemMSA obtains 88.70% Acc-2, 88.56% F1, 50.97% Acc-7, 0.861 Corr, and 0.628 MAE, outperforming the strongest baseline KEBR by 1.43%, 1.31%, and 3.16% in Acc-2, F1, and Acc-7, respectively. It also reduces MAE from 0.683 to 0.628, indicating more accurate sentiment intensity regression. On MOSEI, SemMSA achieves 88.88% Acc-2, 87.70% F1, 55.90% Acc-7, 0.877 Corr, and 0.513 MAE, consistently surpassing previous methods across all metrics. In particular, compared with ConFEDE, SemMSA improves Acc-7 and Corr by 1.04% and 0.097, respectively. These results demonstrate that SemMSA is not limited to incomplete-modality scenarios.

Table 10: Comparison with state-of-the-art methods on MOSI and MOSEI datasets under the complete modality setting. Acc-2 and F1 are reported under the negative/positive protocol.
<table><tr><td rowspan="3">Method</td><td colspan="5">MOSI</td><td colspan="5">MOSEI</td></tr><tr><td>Acc-2</td><td>F1</td><td>Acc-7</td><td>Corr</td><td>MAE↓</td><td>Acc-2</td><td>F1</td><td>Acc-7</td><td>Corr</td><td>MAE↓</td></tr><tr><td></td><td>83.6</td><td>42.3</td><td>0.761</td><td>0.783</td><td>85.5</td><td>85.3</td><td>52.2</td><td>0.756</td><td>0.555</td></tr><tr><td>MISA [44] Self-MM [45]</td><td>83.4 84.9</td><td>84.9</td><td>45.3</td><td></td><td>0.738</td><td>85.15</td><td>84.90</td><td>53.87</td><td>0.765</td><td>0.531</td></tr><tr><td>HyCon [64]</td><td>85.2</td><td>85.1</td><td>46.6</td><td>0.790</td><td>0.713</td><td>85.4</td><td>85.6</td><td>52.8</td><td>0.776</td><td>0.601</td></tr><tr><td>ConFEDE [65]</td><td>85.5</td><td>85.5</td><td>42.3</td><td>0.784</td><td>0.742</td><td>85.82</td><td>85.83</td><td>54.86</td><td>0.780</td><td>0.522</td></tr><tr><td>KEBR [66]</td><td>87.27</td><td>87.25</td><td>47.81</td><td>0.819</td><td>0.683</td><td>86.74</td><td>86.68</td><td>54.37</td><td>0.799</td><td>0.517</td></tr><tr><td>GLoMo [67]</td><td>86.7</td><td>86.6</td><td>48.3</td><td>0.782</td><td>0.718</td><td>86.5</td><td>86.4</td><td>55.0</td><td>0.771</td><td>0.539</td></tr><tr><td>MFON [63]</td><td>86.9</td><td>86.9</td><td>44.9</td><td>0.797</td><td>0.725</td><td>86.32</td><td>86.29</td><td>53.72</td><td>0.780</td><td>0.528</td></tr><tr><td>MMSLF [22]</td><td>86.61</td><td>86.69</td><td></td><td>0.797</td><td>0.734</td><td>86.62</td><td>86.71</td><td></td><td>0.773</td><td>0.539</td></tr><tr><td>SemMSA</td><td>88.70</td><td>88.56</td><td>50.97</td><td>0.861</td><td>0.628</td><td>88.88</td><td>87.70</td><td>55.90</td><td>0.877</td><td>0.513</td></tr></table>

By using Cross-modal Semantic Refinement to introduce compact LLM-derived sentiment semantics and Cross-modal Spectral Alignment to enforce global anchor-free consistency among modalities, SemMSA learns more coherent and discriminative multimodal representations.

Table 11: Ablation study of the number of CSR refinement steps O on the SIMS dataset.
<table><tr><td>0</td><td> $\operatorname { A c c - } 2$ </td><td>F1</td><td> $\mathbf { A c c } { - 3 }$ </td><td>Acc-5</td><td>MAE↓</td><td>Corr</td></tr><tr><td>1</td><td>73.82</td><td>75.94</td><td>56.91</td><td>34.02</td><td>0.493</td><td>0.468</td></tr><tr><td>2</td><td>74.58</td><td>76.62</td><td>57.73</td><td>34.71</td><td>0.485</td><td>0.486</td></tr><tr><td>3</td><td>75.12</td><td>77.08</td><td>58.31</td><td>35.24</td><td>0.479</td><td>0.495</td></tr><tr><td>4</td><td>75.46</td><td>77.50</td><td>58.84</td><td>35.68</td><td>0.474</td><td>0.501</td></tr><tr><td>5</td><td>75.21</td><td>77.12</td><td>58.47</td><td>35.39</td><td>0.478</td><td>0.497</td></tr><tr><td>6</td><td>74.89</td><td>76.84</td><td>58.02</td><td>35.05</td><td>0.482</td><td>0.489</td></tr></table>

## A.3.4 Effect of Semantic Refinement Step

We further investigate the effect of the number of CSR refinement steps O on the SIMS dataset, as shown in Table 11. The refinement step O controls how many continuous latent semantic states are iteratively generated by the frozen LLM. When O = 1, the model only performs a single-step semantic refinement, which provides limited high-level sentiment information and leads to relatively inferior performance. As O increases from 1 to 4, the performance consistently improves across all metrics. This indicates that multiple refinement steps help CSR progressively enrich incomplete multimodal representations with more discriminative sentiment-aware semantics.

The best performance is achieved when $O = 4$ , with 75.46 Acc-2, 77.50 F1, 58.84 Acc-3, 35.68 Acc-5, 0.474 MAE, and 0.501 Corr. This demonstrates that a moderate number of latent semantic refinement steps can effectively balance semantic enhancement and representation compactness. However, when O is further increased to 5 or 6, the performance slightly decreases. A possible reason is that excessive refinement introduces redundant or noisy latent states, which may weaken the compactness of semantic representations and increase the difficulty of cross-modal alignment. Moreover, larger O also increases the computational cost because each refinement step requires an additional forward pass through the frozen LLM. Therefore, we set O = 4 as the default value in SemMSA, which provides the best trade-off between performance and efficiency.

## A.3.5 Robustness Evaluation under Intra-modal Missingness

Following previous work [7–9], we evaluate the robustness of SemMSA under intra-modal random missingness by varying the missing rate r from 0 to 0.9 with an interval of 0.1. We do not report the results at $r = 1 . 0$ , since this setting removes all information from each modality and thus provides little meaningful evidence for sentiment prediction. Tables 12, 13, and 14 present the detailed comparison results on MOSI, MOSEI, and SIMS, respectively.

Table 12: Details of robust comparison on MOSI with different random missing rates.
<table><tr><td colspan="6">Random Missing Rate r = 0</td><td></td></tr><tr><td>Method</td><td>Acc-2</td><td>F1</td><td>Acc-5</td><td>Acc-7</td><td>MAE↓</td><td>Corr</td></tr><tr><td>MISA</td><td>81.24/82.78</td><td>81.23/82.83</td><td>48.30</td><td>43.05</td><td>0.771</td><td>0.777</td></tr><tr><td>Self-MM</td><td>83.24/85.22</td><td>83.26/85.19</td><td>52.38</td><td>42.81</td><td>0.720</td><td>0.790</td></tr><tr><td>MMIM</td><td>81.97/83.43</td><td>81.94/83.43</td><td>49.85</td><td>45.92</td><td>0.744</td><td>0.778</td></tr><tr><td>TETFN</td><td>81.10/82.62</td><td>81.09/82.67</td><td>51.31</td><td>44.07</td><td>0.719</td><td>0.794</td></tr><tr><td>TFR-Net</td><td>81.68/83.64</td><td>81.61/83.57</td><td>47.91</td><td>40.82</td><td>0.805</td><td>0.760</td></tr><tr><td>LNLN</td><td>81.24/84.25</td><td>81.79/84.61</td><td>49.76</td><td>44.56</td><td>0.751</td><td>0.778</td></tr><tr><td>P-RMF</td><td>82.65/84.15</td><td>82.69/84.37</td><td>48.83</td><td>44.31</td><td>0.726</td><td>0.782</td></tr><tr><td>TF-Mamba SemMSA</td><td>81.63/83.69</td><td>81.58/83.71</td><td>50.58</td><td>44.31 45.85</td><td>0.762</td><td>0.774</td></tr><tr><td>86.18/87.80</td><td></td><td>85.99/87.71</td><td>53.16</td><td></td><td>0.657</td><td>0.846</td></tr><tr><td colspan="5">Random Missing Rate r = 0.1</td><td></td><td></td></tr><tr><td>MISA</td><td>79.01/80.18</td><td>78.97/80.21</td><td>46.21</td><td>40.28</td><td>0.847</td><td>0.721</td></tr><tr><td>Self-MM</td><td>80.03/81.40</td><td>80.03/81.19</td><td>49.03</td><td>40.33</td><td>0.812</td><td>0.728</td></tr><tr><td>MMIM</td><td>78.13/79.98</td><td>77.99/79.83</td><td>46.65</td><td>42.61</td><td>0.825</td><td>0.718</td></tr><tr><td>TETFN</td><td>78.91/80.59</td><td>78.79/80.55</td><td>46.84</td><td>40.67</td><td>0.805</td><td>0.731</td></tr><tr><td>TFR-Net</td><td>77.99/79.27</td><td>77.61/78.70</td><td>45.82</td><td></td><td>38.63 0.872</td><td>0.705</td></tr><tr><td>LNLN</td><td>78.43/81.20</td><td>79.04/81.62</td><td>47.91</td><td></td><td>42.37 0.820</td><td>0.724</td></tr><tr><td>P-RMF</td><td>81.34/82.62</td><td>81.35/82.89</td><td>47.52</td><td></td><td>42.13 0.800</td><td>0.730</td></tr><tr><td>TF-Mamba</td><td>80.03/81.86</td><td>79.97/81.87</td><td>48.40</td><td></td><td>42.86 0.824</td><td>0.732</td></tr><tr><td>SemMSA</td><td>82.96/84.89</td><td>82.77/84.74</td><td>49.51</td><td></td><td>43.07 0.761</td><td>0.767</td></tr><tr><td colspan="5">Random Missing Rate r = 0.2</td><td></td><td></td></tr><tr><td>MISA</td><td>76.34/77.54</td><td>76.30/77.58</td><td>41.55</td><td>36.25</td><td>0.939</td><td>0.654</td></tr><tr><td>Self-MM</td><td>76.48/78.15</td><td>76.51/77.76</td><td>43.98</td><td>36.64</td><td>0.901</td><td>0.660</td></tr><tr><td>MMIM TETFN</td><td>74.54/76.42</td><td>74.22/76.12</td><td>42.66</td><td>39.07</td><td>0.918</td><td>0.651</td></tr><tr><td>TFR-Net</td><td>75.60/77.49</td><td>75.35/77.35</td><td>41.79</td><td>35.81</td><td>0.910</td><td>0.657</td></tr><tr><td></td><td>73.52/74.70</td><td>72.70/73.57</td><td>40.13</td><td>34.70</td><td>0.987</td><td>0.622</td></tr><tr><td>LNLN</td><td>76.87/79.22</td><td>77.34/79.53</td><td>45.14</td><td>39.74</td><td>0.891</td><td>0.668</td></tr><tr><td>P-RMF</td><td>78.13/79.57</td><td>78.11/80.97</td><td>44.75</td><td>40.38</td><td>0.853</td><td>0.668</td></tr><tr><td>TF-Mamba</td><td>79.15/80.49</td><td>79.17/80.56</td><td>44.75</td><td>39.21</td><td>0.879</td><td>0.693</td></tr><tr><td>SemMSA</td><td>80.05/81.55</td><td>79.88/81.41</td><td>47.18</td><td></td><td>41.77 0.838</td><td>0.711</td></tr><tr><td colspan="5">Random Missing Rate r = 0.3</td><td></td><td></td></tr><tr><td>MISA</td><td>74.54/75.76</td><td>74.51/75.82</td><td>38.97</td><td></td><td>34.60 0.989</td><td>0.618</td></tr><tr><td>Self-MM</td><td>74.98/76.37</td><td>74.94/75.68</td><td>40.67</td><td>34.89</td><td>0.967</td><td>0.614</td></tr><tr><td>MMIM</td><td>71.91/74.08</td><td>71.28/73.47</td><td>40.43</td><td>36.83</td><td>0.974</td><td>0.612</td></tr><tr><td>TETFN</td><td>73.42/75.25</td><td>72.78/74.77</td><td>38.58</td><td>33.24</td><td>0.982</td><td>0.607</td></tr><tr><td>TFR-Net</td><td>71.28/72.36</td><td>69.58/70.12</td><td>38.34</td><td>32.55</td><td>1.065</td><td>0.572</td></tr><tr><td>LNLN</td><td>75.96/77.29</td><td>75.68/77.56</td><td>42.81</td><td>38.00</td><td>0.953</td><td>0.617</td></tr><tr><td>P-RMF</td><td>75.80/76.83</td><td>75.82/79.27</td><td>42.71</td><td>39.21</td><td>0.922</td><td>0.621</td></tr><tr><td>TF-Mamba</td><td>76.53/77.74</td><td>76.56/77.85</td><td>42.27</td><td>37.76</td><td>0.932</td><td>0.645</td></tr><tr><td>SemMSA</td><td>79.65/78.79</td><td>79.43/78.69</td><td>46.75</td><td>41.91</td><td>0.889</td><td>0.658</td></tr><tr><td colspan="5">Random Missing Rate r = 0.4</td><td></td><td></td></tr><tr><td>MISA</td><td>72.59/73.88</td><td>72.49/73.88</td><td></td><td>35.37</td><td>32.65 1.041</td><td>0.585</td></tr><tr><td>Self-MM</td><td>71.96/73.17</td><td>71.75/71.74</td><td>36.30</td><td>31.20</td><td>1.027</td><td>0.579</td></tr><tr><td>MMIM</td><td>68.90/70.84</td><td>67.80/69.69</td><td>35.76</td><td>33.38</td><td>1.034</td><td>0.576</td></tr><tr><td>TETFN</td><td>70.07/72.05</td><td>68.58/70.79</td><td>35.28</td><td>30.66</td><td>1.051</td><td>0.571</td></tr><tr><td>TFR-Net</td><td>67.74/68.75</td><td>64.41/64.71</td><td>35.76</td><td>30.17</td><td>1.142</td><td>0.537</td></tr><tr><td>LNLN</td><td>74.25/76.01</td><td>74.67/76.31</td><td>41.11</td><td>36.49</td><td>0.987</td><td>0.594</td></tr><tr><td>P-RMF</td><td>73.76/75.46</td><td>74.09/77.71</td><td>40.67</td><td>35.59</td><td>1.001</td><td>0.584</td></tr><tr><td>TF-Mamba</td><td>75.22/76.07</td><td>75.25/76.16</td><td>40.23</td><td>35.86</td><td>0.961</td><td>0.617</td></tr><tr><td>SemMSA</td><td>77.86/78.02</td><td>77.67/77.89</td><td>44.69</td><td>39.71</td><td>0.937</td><td>0.625</td></tr></table>

<table><tr><td colspan="6">Random Missing Rate r = 0.5</td><td></td></tr><tr><td>Method</td><td>Acc-2</td><td>F1</td><td>Acc-5</td><td>Acc-7</td><td>MAE↓</td><td>Corr</td></tr><tr><td>MISA</td><td>69.34/70.53</td><td>69.20/70.50</td><td>30.61</td><td>28.14</td><td>1.124</td><td>0.519</td></tr><tr><td>Self-MM</td><td>67.54/67.43</td><td>66.81/64.27</td><td>31.39</td><td>26.97</td><td>1.129</td><td>0.503</td></tr><tr><td>MMIM</td><td>66.52/68.09</td><td>64.59/66.15</td><td>29.89</td><td>28.23</td><td>1.128</td><td>0.501</td></tr><tr><td>TETFN</td><td>65.06/67.23</td><td>61.78/64.30</td><td>31.34</td><td>27.55</td><td>1.157</td><td>0.492</td></tr><tr><td>TFR-Net</td><td>63.02/64.83</td><td>56.64/58.04</td><td>30.71</td><td>25.85</td><td>1.270</td><td>0.443</td></tr><tr><td>LNLN</td><td>71.86/73.37</td><td>72.30/73.70</td><td>38.39</td><td>33.92</td><td>1.059</td><td>0.536</td></tr><tr><td>P-RMF</td><td>71.28/73.02</td><td>71.66/73.33</td><td>37.90</td><td>33.67</td><td>1.077</td><td>0.523</td></tr><tr><td>TF-Mamba SemMSA</td><td>74.20/75.00</td><td>74.27/75.15</td><td>37.46</td><td>33.67</td><td>1.044</td><td>0.557</td></tr><tr><td></td><td>74.79/75.58</td><td>74.62/75.48</td><td>40.61</td><td>36.07</td><td>1.025</td><td>0.558</td></tr><tr><td colspan="5">Random Missing Rate r = 0.6</td><td></td><td></td></tr><tr><td>MISA</td><td>65.84/66.97</td><td>65.69/66.94</td><td>27.13</td><td></td><td>24.68 1.200</td><td>0.441</td></tr><tr><td>Self-MM</td><td>63.36/63.47</td><td>62.07/58.94</td><td>27.31</td><td></td><td>24.34 1.209</td><td>0.425</td></tr><tr><td>MMIM</td><td>62.49/63.67</td><td>59.48/60.87</td><td>27.11</td><td>25.41</td><td>1.208</td><td>0.418</td></tr><tr><td>TETFN</td><td>61.23/63.42</td><td>56.08/58.68</td><td>27.99</td><td></td><td>25.12 1.238</td><td>0.417</td></tr><tr><td>TFR-Net</td><td>59.47/61.64</td><td>50.53/52.44</td><td>28.33</td><td></td><td>24.05 1.371</td><td>0.363</td></tr><tr><td>LNLN</td><td>67.69/69.00</td><td>67.99/69.19</td><td></td><td>34.35</td><td>30.37 1.147</td><td>0.458</td></tr><tr><td>P-RMF</td><td>67.35/68.75</td><td>67.64/68.64</td><td>33.24</td><td></td><td>29.30 1.147</td><td>0.432</td></tr><tr><td>TF-Mamba</td><td>68.37/68.60</td><td>68.45/68.79</td><td>33.53</td><td></td><td>30.76 1.127</td><td>0.487</td></tr><tr><td>SemMSA</td><td>70.71/69.03</td><td>70.58/69.19</td><td>37.69</td><td></td><td>34.18 1.139</td><td>0.447</td></tr><tr><td colspan="5">Random Missing Rate r = 0.7</td><td></td><td></td></tr><tr><td>MISA</td><td>63.89/65.09</td><td>63.74/65.07</td><td>23.27</td><td>21.14</td><td>1.257</td><td>0.381</td></tr><tr><td>Self-MM</td><td>61.46/61.74</td><td>58.97/55.11</td><td>23.81</td><td>20.70</td><td>1.271</td><td>0.339</td></tr><tr><td>MMIM</td><td>59.18/61.23</td><td>54.36/57.15</td><td>24.00</td><td>22.35</td><td>1.267</td><td>0.342</td></tr><tr><td>TETFN TFR-Net</td><td>58.65/61.13</td><td>50.77/53.79</td><td>25.27</td><td>23.13</td><td>1.293</td><td>0.337</td></tr><tr><td></td><td>57.34/59.91 65.01/65.95</td><td>45.48/48.41</td><td>26.92</td><td>23.71 27.79</td><td>1.454</td><td>0.276</td></tr><tr><td>LNLN</td><td></td><td>65.14/65.95</td><td>31.19</td><td></td><td>1.219</td><td>0.383</td></tr><tr><td>P-RMF</td><td>65.16/66.16</td><td>65.33/64.69</td><td>32.94</td><td>27.84</td><td>1.229</td><td>0.383</td></tr><tr><td>TF-Mamba</td><td>66.91/67.23</td><td>66.98/67.41</td><td>29.30</td><td>27.26</td><td>1.196</td><td>0.411</td></tr><tr><td>SemMSA</td><td>67.51/64.76</td><td>67.31/64.99</td><td>33.48</td><td>30.68</td><td>1.232</td><td>0.358</td></tr><tr><td colspan="5">Random Missing Rate r = 0.8</td><td></td><td></td></tr><tr><td>MISA</td><td>62.24/63.56</td><td>61.67/63.16</td><td>20.99</td><td></td><td>19.92 1.311</td><td>0.321</td></tr><tr><td>Self-MM</td><td>58.26/59.55</td><td>53.56/49.98</td><td>22.11</td><td>19.29</td><td>1.313</td><td>0.282</td></tr><tr><td>MMIM</td><td>55.30/58.53</td><td>47.89/52.46</td><td>21.77</td><td>20.26</td><td>1.312</td><td>0.287</td></tr><tr><td>TETFN</td><td>56.85/59.40</td><td>45.59/48.73</td><td>23.76</td><td>22.01</td><td>1.337</td><td>0.274</td></tr><tr><td>TFR-Net</td><td>55.98/58.49</td><td>41.88/44.70</td><td>27.70</td><td>23.23</td><td>1.497</td><td>0.155</td></tr><tr><td>LNLN</td><td>62.10/62.75</td><td>62.03/62.56</td><td>28.23</td><td>26.34</td><td>1.283</td><td>0.314</td></tr><tr><td>P-RMF</td><td>61.08/62.04</td><td>61.22/60.76</td><td>29.74</td><td>25.97</td><td>1.275</td><td>0.316</td></tr><tr><td>TF-Mamba</td><td>63.12/63.57</td><td>63.20/63.77</td><td>26.38</td><td>24.93</td><td>1.258</td><td>0.353</td></tr><tr><td>SemMSA</td><td>64.98/62.78</td><td>64.81/62.96</td><td>30.69</td><td>28.05</td><td>1.278</td><td>0.316</td></tr><tr><td colspan="5">Random Missing Rate r = 0.9</td><td></td><td></td></tr><tr><td>MISA</td><td>58.21/58.64</td><td>56.19/56.84</td><td></td><td>18.41</td><td>17.78 1.369</td><td>0.226</td></tr><tr><td>Self-MM</td><td>55.25/58.59</td><td>47.46/46.16</td><td>19.78</td><td>18.32</td><td>1.353</td><td>0.197</td></tr><tr><td>MMIM</td><td>51.65/55.29</td><td>40.89/47.33</td><td>19.53</td><td>18.95</td><td>1.357</td><td>0.186</td></tr><tr><td>TETFN</td><td>55.88/58.43</td><td>42.12/45.24</td><td>21.19</td><td>20.75</td><td>1.378</td><td>0.186</td></tr><tr><td>TFR-Net</td><td>55.44/57.93</td><td>40.18/43.01</td><td>25.12</td><td>21.67</td><td>1.534</td><td>0.155</td></tr><tr><td>LNLN</td><td>56.51/56.50</td><td>56.47/56.32</td><td>23.86</td><td>22.98</td><td>1.349</td><td>0.202</td></tr><tr><td>P-RMF TF-Mamba</td><td>58.75/59.45</td><td>59.01/56.66</td><td>26.68</td><td>23.49</td><td>1.346</td><td>0.212</td></tr><tr><td></td><td>60.20/60.37</td><td>60.30/60.59</td><td>24.49</td><td>22.89</td><td>1.363</td><td>0.215</td></tr><tr><td>SemMSA</td><td>58.97/55.95</td><td>58.79/55.18</td><td>25.59</td><td>23.67</td><td>1.357</td><td>0.217</td></tr></table>

On the MOSI dataset, SemMSA achieves the best overall performance under low and moderate missing rates, as shown in Table 12. When r = 0, SemMSA obtains 86.18/87.80 Acc-2, 85.99/87.71 F1, 53.16 Acc-5, 45.85 Acc-7, 0.657 MAE, and 0.846 Corr, outperforming all baselines by a clear margin. As the missing rate increases from 0.1 to 0.5, SemMSA remains highly stable and achieves the best results on most metrics. For example, at r = 0.5, SemMSA obtains 74.79/75.58 Acc-2, 74.62/75.48 F1, 40.61 Acc-5, 36.07 Acc-7, 1.025 MAE, and 0.558 Corr, outperforming strong baselines such as P-RMF and TF-Mamba on the majority of metrics. These results indicate that the LLM-derived latent semantics introduced by CSR can effectively compensate for missing modality information, while CSA further improves cross-modal consistency under incomplete observations. At extremely high missing rates, such as r = 0.8 and r = 0.9, all methods suffer from severe degradation. Although SemMSA is not always the best on every metric in these extreme cases, it still achieves competitive results, showing that the proposed semantic-aided framework remains robust even when most modality evidence is missing.

Table 13: Details of robust comparison on MOSEI with different random missing rates.
<table><tr><td colspan="6">Random Missing Rate r = 0</td><td></td></tr><tr><td>Method</td><td>Acc-2</td><td>F1</td><td>Acc-5</td><td>Acc-7</td><td>MAE↓</td><td>Corr</td></tr><tr><td>MISA</td><td>84.10/85.28</td><td>83.75/85.10</td><td>53.85</td><td>51.79</td><td>0.552</td><td>0.759</td></tr><tr><td>Self-MM</td><td>84.68/85.34</td><td>84.66/85.11</td><td>55.72</td><td>53.89</td><td>0.531</td><td>0.764</td></tr><tr><td>MMIM</td><td>81.65/83.53</td><td>81.41/83.39</td><td>53.04</td><td>50.76</td><td>0.576</td><td>0.724</td></tr><tr><td>TFR-Net</td><td>84.65/84.96</td><td>84.34/84.71</td><td>47.91</td><td>53.71</td><td>0.550</td><td>0.745</td></tr><tr><td>LNLN</td><td>83.61/84.14</td><td>84.02/84.53</td><td>51.94</td><td>50.66</td><td>0.572</td><td>0.735</td></tr><tr><td>P-RMF</td><td>83.62/85.20</td><td>83.68/85.48</td><td>52.09</td><td>49.77</td><td>0.539</td><td>0.767</td></tr><tr><td>TF-Mamba</td><td>82.89/83.82</td><td>82.92/83.71</td><td>53.83</td><td>52.26</td><td>0.556</td><td>0.748</td></tr><tr><td>SemMSA</td><td>84.48/86.41</td><td>84.53/86.32</td><td>54.74</td><td>52.95</td><td>0.516</td><td>0.799</td></tr><tr><td colspan="6">Random Missing Rate r = 0.1</td><td></td></tr><tr><td>MISA</td><td>82.28/82.21</td><td>80.79/81.28</td><td>51.34</td><td>50.13</td><td>0.598</td><td>0.722</td></tr><tr><td>Self-MM</td><td>83.79/83.03</td><td>83.23/82.43</td><td>53.18</td><td>51.80</td><td>0.564</td><td>0.725</td></tr><tr><td>MMIM</td><td>81.09/82.00</td><td>80.15/81.57</td><td>51.19</td><td>49.09</td><td>0.602</td><td>0.696</td></tr><tr><td>TFR-Net</td><td>83.31/82.92</td><td>82.40/82.25</td><td>45.82</td><td>52.29</td><td>0.573</td><td>0.715</td></tr><tr><td>LNLN</td><td>82.73/83.32</td><td>82.91/83.66</td><td>51.25</td><td>49.96</td><td>0.591</td><td>0.712</td></tr><tr><td>P-RMF</td><td>82.79/83.98</td><td>82.94/84.37</td><td>51.45</td><td>49.04</td><td>0.556</td><td>0.748</td></tr><tr><td>TF-Mamba</td><td>82.68/83.16</td><td>82.69/83.03</td><td>52.07</td><td>50.53</td><td>0.570</td><td>0.730</td></tr><tr><td>SemMSA</td><td>83.74/85.30</td><td>83.74/85.17</td><td>53.41</td><td>51.76</td><td>0.536</td><td>0.774</td></tr><tr><td colspan="6">Random Missing Rate r = 0.2</td><td></td></tr><tr><td>MISA</td><td>79.93/77.84</td><td>76.88/75.56</td><td>47.66</td><td>47.24</td><td>0.659</td><td>0.674</td></tr><tr><td>Self-MM</td><td>82.33/80.84</td><td>81.17/79.76</td><td>50.51</td><td>49.44</td><td>0.604</td><td>0.678</td></tr><tr><td>MMIM</td><td>79.66/79.93</td><td>77.68/79.08</td><td>47.99</td><td>46.27</td><td>0.642</td><td>0.653</td></tr><tr><td>TFR-Net</td><td>81.61/80.47</td><td>79.99/79.29</td><td>40.13</td><td>51.04</td><td>0.604</td><td>0.672</td></tr><tr><td>LNLN</td><td>81.68/81.70</td><td>81.89/81.95</td><td>49.95</td><td>48.75</td><td>0.616</td><td>0.677</td></tr><tr><td>P-RMF</td><td>82.25/82.97</td><td>82.58/83.37</td><td>49.35</td><td>47.91</td><td>0.576</td><td>0.722</td></tr><tr><td>TF-Mamba SemMSA</td><td>81.84/82.55</td><td>81.83/82.40</td><td>50.50</td><td>49.17</td><td>0.588</td><td>0.710</td></tr><tr><td></td><td>83.11/83.97</td><td>83.05/83.85</td><td>52.32</td><td>50.84</td><td>0.568</td><td>0.738</td></tr><tr><td colspan="6">Random Missing Rate r = 0.3</td><td></td></tr><tr><td>MISA</td><td>77.28/73.32</td><td>72.25/68.91</td><td>43.40</td><td>43.99</td><td>0.724</td><td>0.615</td></tr><tr><td>Self-MM</td><td>79.99/77.63</td><td>77.74/75.69</td><td>48.07</td><td>47.23</td><td>0.653</td><td>0.610</td></tr><tr><td>MMIM</td><td>77.79/77.08</td><td>74.49/75.46</td><td>44.73</td><td>43.25</td><td>0.690</td><td>0.597</td></tr><tr><td>TFR-Net</td><td>79.29/77.48</td><td>76.52/75.43</td><td>38.34</td><td>48.75</td><td>0.650</td><td>0.604</td></tr><tr><td>LNLN</td><td>80.45/80.11</td><td>80.91/80.44</td><td>48.40</td><td>47.36</td><td>0.648</td><td>0.629</td></tr><tr><td>P-RMF</td><td>80.88/81.26</td><td>81.40/81.86</td><td>47.78</td><td>45.95</td><td>0.611</td><td>0.683</td></tr><tr><td>TF-Mamba</td><td>81.22/80.82</td><td>81.10/80.58</td><td>49.00</td><td>47.89</td><td>0.613</td><td>0.675</td></tr><tr><td>SemMSA</td><td>81.59/82.56</td><td>81.47/82.41</td><td>50.58</td><td>49.27</td><td>0.604</td><td>0.692</td></tr><tr><td colspan="6">Random Missing Rate r = 0.4</td><td></td></tr><tr><td>MISA</td><td>75.04/70.46</td><td>67.93/64.02</td><td>39.53</td><td>40.87</td><td>0.780</td><td>0.561</td></tr><tr><td>Self-MM</td><td>78.09/75.02</td><td>74.48/72.01</td><td>45.04</td><td>44.40</td><td>0.694</td><td>0.554</td></tr><tr><td>MMIM</td><td>76.15/74.56</td><td>71.40/71.98</td><td>41.86</td><td>40.84</td><td>0.732</td><td>0.542</td></tr><tr><td>TFR-Net</td><td>77.65/74.74</td><td>73.71/71.67</td><td>35.76</td><td>46.70</td><td>0.688</td><td>0.548</td></tr><tr><td>LNLN</td><td>79.70/78.49</td><td>80.46/78.98</td><td>46.88</td><td>45.99</td><td>0.673</td><td>0.592</td></tr><tr><td>P-RMF TF-Mamba</td><td>79.76/79.97 80.02/80.02</td><td>80.58/80.74 79.80/79.80</td><td>46.73</td><td>45.59</td><td>0.631</td><td>0.653</td></tr><tr><td></td><td></td><td></td><td>47.80</td><td>46.73</td><td>0.639</td><td>0.638</td></tr><tr><td>SemMSA</td><td>81.07/81.58</td><td>80.86/81.38</td><td>49.07</td><td>47.94</td><td>0.614</td><td>0.664</td></tr></table>

<table><tr><td colspan="7">Random Missing Rate r = 0.5</td></tr><tr><td>Method</td><td>Acc-2</td><td>F1</td><td>Acc-5</td><td>Acc-7</td><td>MAE↓</td><td>Corr</td></tr><tr><td>MISA</td><td>73.21/67.38</td><td>64.14/58.38</td><td>36.05</td><td>38.12</td><td>0.834</td><td>0.492</td></tr><tr><td>Self-MM</td><td>75.81/71.97</td><td>70.38/67.40</td><td>43.14</td><td>42.70</td><td>0.733</td><td>0.477</td></tr><tr><td>MMIM</td><td>74.45/71.75</td><td>67.96/67.70</td><td>39.21</td><td>38.68</td><td>0.775</td><td>0.470</td></tr><tr><td>TFR-Net</td><td>75.69/71.53</td><td>70.07/66.88</td><td>30.71</td><td>45.00</td><td>0.730</td><td>0.471</td></tr><tr><td>LNLN</td><td>78.10/76.44</td><td>79.30/77.23</td><td>45.59</td><td>44.90</td><td>0.710</td><td>0.529</td></tr><tr><td>P-RMF</td><td>78.64/78.62</td><td>79.74/79.49</td><td>44.90</td><td>43.94</td><td>0.666</td><td>0.601</td></tr><tr><td>TF-Mamba</td><td>77.94/78.59</td><td>77.78/78.34</td><td>46.60</td><td>45.68</td><td>0.676</td><td>0.583</td></tr><tr><td>SemMSA</td><td>80.20/80.85</td><td>79.85/80.58</td><td>47.08</td><td>46.01</td><td>0.658</td><td>0.602</td></tr><tr><td colspan="7">Random Missing Rate r = 0.6</td></tr><tr><td>MISA</td><td>72.30/65.55</td><td>62.12/54.64</td><td>33.30</td><td>36.16</td><td>0.875</td><td>0.415</td></tr><tr><td>Self-MM</td><td>73.93/69.33</td><td>66.76/63.01</td><td>41.75</td><td>41.47</td><td>0.762</td><td>0.401</td></tr><tr><td>MMIM</td><td>73.16/68.83</td><td>65.43/63.09</td><td>37.48</td><td>37.13</td><td>0.808</td><td>0.402</td></tr><tr><td>TFR-Net</td><td>74.05/68.80</td><td>67.07/62.51</td><td>28.33</td><td>43.88</td><td>0.762</td><td>0.397</td></tr><tr><td>LNLN</td><td>76.50/73.82</td><td>78.33/75.03</td><td>44.00</td><td>43.52</td><td>0.736</td><td>0.471</td></tr><tr><td>P-RMF</td><td>77.44/76.11</td><td>78.97/77.58</td><td>43.12</td><td>42.26</td><td>0.703</td><td>0.545</td></tr><tr><td>TF-Mamba</td><td>75.77/75.89</td><td>75.59/75.63</td><td>44.73</td><td>43.96</td><td>0.709</td><td>0.534 0.541</td></tr><tr><td>SemMSA</td><td>78.81/79.04</td><td>79.18/78.60</td><td>46.13</td><td>45.36</td><td>0.691</td><td></td></tr><tr><td colspan="7">Random Missing Rate r = 0.7</td></tr><tr><td>MISA</td><td>71.71/64.28</td><td>60.65/51.82</td><td>31.21</td><td>34.54</td><td>0.906</td><td>0.344</td></tr><tr><td>Self-MM</td><td>72.55/66.79</td><td>63.45/58.05</td><td>40.12</td><td>39.93</td><td>0.786</td><td>0.329</td></tr><tr><td>MMIM</td><td>72.26/66.89</td><td>63.26/58.90</td><td>35.47</td><td>35.25</td><td>0.834</td><td>0.341</td></tr><tr><td>TFR-Net</td><td>72.77/66.64 74.74/71.55</td><td>64.02/58.32</td><td>26.92</td><td>42.91</td><td>0.786</td><td>0.322</td></tr><tr><td>LNLN</td><td></td><td>77.40/73.49</td><td>42.56</td><td>42.22</td><td>0.762</td><td>0.408</td></tr><tr><td>P-RMF</td><td>75.87/74.64</td><td>78.12/75.88</td><td>42.41</td><td>41.73</td><td>0.733</td><td>0.481</td></tr><tr><td>TF-Mamba</td><td>73.49/73.50</td><td>73.19/73.26</td><td>43.40</td><td>42.78</td><td>0.747</td><td>0.469</td></tr><tr><td>SemMSA</td><td>76.30/75.54</td><td>78.40/76.84</td><td>43.66</td><td>43.07</td><td>0.745</td><td>0.469</td></tr><tr><td colspan="7">Random Missing Rate r = 0.8</td></tr><tr><td>MISA</td><td>71.30/63.43</td><td>59.69/49.95</td><td>29.51</td><td>33.29</td><td>0.927</td><td>0.267</td></tr><tr><td>Self-MM</td><td>71.83/65.07</td><td>61.49/54.44</td><td>38.78</td><td>38.69</td><td>0.805</td><td>0.259</td></tr><tr><td>MMIM</td><td>71.57/64.97</td><td>61.45/54.76</td><td>33.71</td><td>33.64</td><td>0.858</td><td>0.269</td></tr><tr><td>TFR-Net</td><td>71.95/65.05</td><td>61.82/54.91</td><td>27.70</td><td>42.23</td><td>0.807</td><td>0.241</td></tr><tr><td>LNLN</td><td>72.86/68.62</td><td>76.80/71.83</td><td>40.97</td><td>40.76</td><td>0.791</td><td>0.325</td></tr><tr><td>P-RMF</td><td>74.46/70.75</td><td>77.91/73.03</td><td>41.00</td><td>40.46</td><td>0.764</td><td>0.401</td></tr><tr><td>TF-Mamba</td><td>71.60/70.34</td><td>71.27/70.16</td><td>40.93</td><td>40.37</td><td>0.786</td><td>0.408</td></tr><tr><td>SemMSA</td><td>74.13/70.41</td><td>77.65/73.03</td><td>43.38</td><td>42.90</td><td>0.753</td><td>0.419</td></tr><tr><td colspan="7">Random Missing Rate r = 0.9</td></tr><tr><td>MISA Self-MM</td><td>71.07/62.95</td><td>59.12/48.80</td><td>28.03</td><td>32.29</td><td>0.941</td><td>0.180</td></tr><tr><td>MMIM</td><td>71.24/63.85 71.10/63.69</td><td>59.72/51.32 59.99/51.26</td><td>37.50 32.67</td><td>37.46 32.61</td><td>0.821 0.877</td><td>0.188 0.197</td></tr><tr><td>TFR-Net</td><td>71.34/63.64</td><td>59.99/52.02</td><td>25.12</td><td>41.73</td><td>0.820</td><td>0.175</td></tr><tr><td>LNLN</td><td>71.51/64.83</td><td>77.52/70.60</td><td>40.19</td><td>40.10</td><td>0.820</td><td>0.221</td></tr><tr><td>P-RMF</td><td>72.59/67.86</td><td>77.95/71.51</td><td>39.90</td><td>39.62</td><td>0.805</td><td>0.289</td></tr><tr><td>TF-Mamba</td><td>68.68/64.75</td><td>68.18/64.87</td><td>37.56</td><td>37.24</td><td>0.851</td><td>0.291</td></tr><tr><td>SemMSA</td><td>72.73/68.18</td><td>77.53/70.57</td><td>40.85</td><td>40.52</td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>0.801</td><td>0.316</td></tr></table>

As shown in Table 13, SemMSA shows more consistent advantages across different missing rates on MOSEI dataset. Under the complete setting r = 0, SemMSA achieves the best Corr of 0.799 and the lowest MAE of 0.516, while maintaining competitive Acc-2 and F1 results. When the missing rate increases, SemMSA consistently outperforms most baselines. For instance, at r = 0.4, SemMSA obtains 81.07/81.58 Acc-2, 80.86/81.38 F1, 49.07 Acc-5, 47.94 Acc-7, 0.614 MAE, and 0.664 Corr, achieving the best performance on all metrics. At r = 0.6, SemMSA also achieves the best Acc-2, F1, Acc-5, Acc-7, and MAE, demonstrating strong robustness under moderate-to-severe missingness. Even at r = 0.9, where the available multimodal information is extremely limited, SemMSA obtains the best Acc-2, Acc-5, MAE, and Corr, verifying its ability to preserve reliable sentiment representations under highly incomplete conditions.

On the SIMS dataset, SemMSA also demonstrates strong robustness, especially in terms of F1, MAE, and Corr, as shown in Table 14. At r = 0, SemMSA achieves the best Acc-2, F1, Acc-5, MAE, and Corr, with 80.62 Acc-2, 80.61 F1, 42.83 Acc-5, 0.408 MAE, and 0.640 Corr. When the missing rate increases to r = 0.3, SemMSA still achieves 78.00 Acc-2, 78.77 F1, 41.29 Acc-5, 0.456 MAE, and 0.555 Corr, surpassing competing methods on most metrics. At higher missing rates, SemMSA remains particularly strong in F1 and Corr. For example, at r = 0.8, SemMSA achieves the best Acc-2, F1, Acc-3, MAE, and Corr, indicating that the proposed method can better preserve sentiment discriminative information in challenging Chinese multimodal sentiment scenarios. At r = 0.9,

Table 14: Details of robust comparison on SIMS with different random missing rates.
<table><tr><td colspan="7">Random Missing Rate r = 0</td><td colspan="7">Random Missing Rate r = 0.5</td></tr><tr><td>Method</td><td>Acc-2</td><td>F1</td><td>Acc-3</td><td>Acc-5</td><td>MAE↓</td><td>Corr</td><td>Method</td><td>Acc-2</td><td>F1</td><td>Acc-3</td><td>Acc-5</td><td>MAE↓</td><td>Corr</td></tr><tr><td>MISA</td><td>78.19</td><td>77.22</td><td>63.38</td><td>40.55</td><td>0.449</td><td>0.576</td><td>MISA</td><td>71.26</td><td>64.16</td><td>54.78</td><td>30.56</td><td>0.552</td><td>0.367</td></tr><tr><td>Self-MM</td><td>78.26</td><td>78.00</td><td>64.92</td><td>40.77</td><td>0.421</td><td>0.584</td><td>Self-MM</td><td>71.41</td><td>67.11</td><td>53.90</td><td>32.02</td><td>0.517</td><td>0.390</td></tr><tr><td>MMIM</td><td>75.42</td><td>73.10</td><td>60.69</td><td>37.42</td><td>0.475</td><td>0.528</td><td>MMIM</td><td>68.49</td><td>64.81</td><td>52.37</td><td>33.41</td><td>0.553</td><td>0.336</td></tr><tr><td>TETFN</td><td>80.23</td><td>79.25</td><td>65.86</td><td>41.94</td><td>0.424</td><td>0.589</td><td>TETFN</td><td>72.43</td><td>67.30</td><td>56.24</td><td>33.48</td><td>0.512</td><td>0.394</td></tr><tr><td>TFR-Net</td><td>69.15</td><td>58.44</td><td>54.12</td><td>33.85</td><td>0.562</td><td>0.254</td><td>TFR-Net</td><td>67.47</td><td>58.66</td><td>52.37</td><td>24.65</td><td>0.685</td><td>0.171</td></tr><tr><td>P-RMF</td><td>78.34</td><td>79.69</td><td>60.61</td><td>38.95</td><td>0.441</td><td>0.550</td><td>P-RMF</td><td>73.61</td><td>74.83</td><td>54.83</td><td>34.79</td><td>0.495</td><td>0.404</td></tr><tr><td>TF-Mamba</td><td>79.65</td><td>78.92</td><td>61.93</td><td>37.86</td><td>0.441</td><td>0.548</td><td>TF-Mamba</td><td>75.71</td><td>73.61</td><td>58.64</td><td>37.20</td><td>0.495</td><td>0.424</td></tr><tr><td>SemMSA</td><td>80.62</td><td>80.61</td><td>65.15</td><td>42.83</td><td>0.408</td><td>0.640</td><td>SemMSA</td><td>73.84</td><td>76.27</td><td>56.60</td><td>38.21</td><td>0.486</td><td>0.480</td></tr><tr><td colspan="8">Random Missing Rate r = 0.1</td><td colspan="6">Random Missing Rate r = 0.6</td></tr><tr><td>MISA</td><td>77.39</td><td>75.82</td><td>63.02</td><td>38.88</td><td>0.461</td><td>0.561</td><td>MISA</td><td>70.46</td><td>61.81</td><td>53.97</td><td>27.72</td><td>0.578</td><td>0.286</td></tr><tr><td>Self-MM</td><td>77.32</td><td>76.76</td><td>63.53</td><td>40.26</td><td>0.433</td><td>0.563</td><td>Self-MM</td><td>70.02</td><td>64.21</td><td>51.86</td><td>29.10</td><td>0.548</td><td>0.313</td></tr><tr><td>MMIM</td><td>74.25</td><td>72.08</td><td>60.90</td><td>37.27</td><td>0.473</td><td>0.529</td><td>MMIM</td><td>67.91</td><td>63.86</td><td>49.31</td><td>29.18</td><td>0.578</td><td>0.270</td></tr><tr><td>TETFN</td><td>78.92</td><td>77.70</td><td>64.62</td><td>41.36</td><td>0.432</td><td>0.578</td><td>TETFN</td><td>70.97</td><td>64.19</td><td>53.54</td><td>29.90</td><td>0.545</td><td>0.309</td></tr><tr><td>TFR-Net</td><td>68.85</td><td>59.38</td><td>53.25</td><td>30.12</td><td>0.596</td><td>0.203</td><td>TFR-Net</td><td>67.03</td><td>58.30</td><td>52.59</td><td>24.80</td><td>0.696</td><td>0.157</td></tr><tr><td>P-RMF</td><td>77.24</td><td>78.88</td><td>59.52</td><td>37.72</td><td>0.454</td><td>0.530</td><td>P-RMF</td><td>72.12</td><td>73.32</td><td>53.05</td><td>33.92</td><td>0.515</td><td>0.383</td></tr><tr><td>TF-Mamba</td><td>79.65</td><td>78.79</td><td>62.36</td><td>36.98</td><td>0.445</td><td>0.550</td><td>TF-Mamba</td><td>73.09</td><td>69.44</td><td>54.92</td><td>32.82</td><td>0.523</td><td>0.370</td></tr><tr><td>SemMSA</td><td>79.54</td><td>79.83</td><td>63.39</td><td>43.05</td><td>0.425</td><td>0.625</td><td>SemMSA</td><td>72.96</td><td>75.94</td><td>55.74</td><td>33.64</td><td>0.495</td><td>0.455</td></tr><tr><td colspan="8">Random Missing Rate r = 0.2</td><td colspan="6">Random Missing Rate r = 0.7</td></tr><tr><td>MISA</td><td>74.33</td><td>71.70</td><td>59.23</td><td>38.15</td><td>0.489</td><td>0.490</td><td>MISA</td><td>69.95</td><td>59.54</td><td>52.52</td><td>24.87</td><td>0.601</td><td>0.167</td></tr><tr><td>Self-MM</td><td>74.98</td><td>73.71</td><td>61.71</td><td>38.37</td><td>0.464</td><td>0.500</td><td>Self-MM</td><td>69.58</td><td>62.28</td><td>50.62</td><td>25.53</td><td>0.571</td><td>0.198</td></tr><tr><td>MMIM</td><td>72.36</td><td>69.80</td><td>57.33</td><td>37.27</td><td>0.504</td><td>0.460</td><td>MMIM</td><td>66.89</td><td>62.23</td><td>46.53</td><td>28.59</td><td>0.595</td><td>0.190</td></tr><tr><td>TETFN</td><td>75.49</td><td>73.59</td><td>61.56</td><td>39.46</td><td>0.457</td><td>0.527</td><td>TETFN</td><td>69.29</td><td>61.09</td><td>51.06</td><td>27.86</td><td>0.572</td><td>0.190</td></tr><tr><td>TFR-Net</td><td>68.64</td><td>59.74</td><td>53.61</td><td>29.03</td><td>0.619</td><td>0.191</td><td>TFR-Net</td><td>67.18</td><td>58.15</td><td>52.30</td><td>23.78</td><td>0.707</td><td>0.163</td></tr><tr><td>P-RMF</td><td>76.23</td><td>77.46</td><td>59.30</td><td>37.05</td><td>0.460</td><td>0.512</td><td>P-RMF</td><td>71.58</td><td>72.22</td><td>52.52</td><td>32.23</td><td>0.531</td><td>0.369</td></tr><tr><td>TF-Mamba</td><td>78.77</td><td>77.88</td><td>61.05</td><td>38.29</td><td>0.459</td><td>0.507</td><td>TF-Mamba</td><td>71.77</td><td>67.46</td><td>48.80</td><td>28.45</td><td>0.570</td><td>0.248</td></tr><tr><td>SemMSA</td><td>77.78</td><td>78.36</td><td>62.52</td><td>42.16</td><td>0.441</td><td>0.590</td><td>SemMSA</td><td>72.09</td><td>75.78</td><td>55.52</td><td>31.43</td><td>0.510</td><td>0.425</td></tr><tr><td colspan="8">Random Missing Rate r = 0.3</td><td colspan="6">Random Missing Rate r = 0.8</td></tr><tr><td></td><td>74.11</td><td>70.40</td><td>59.30</td><td></td><td></td><td>0.464</td><td>MISA</td><td>69.37</td><td>57.82</td><td>52.22</td><td>22.69</td><td></td><td></td></tr><tr><td>MISA Self-MM</td><td>74.76</td><td>72.85</td><td>59.81</td><td>36.40 37.93</td><td>0.505 0.474</td><td>0.487</td><td>Self-MM</td><td>69.51</td><td>60.68</td><td>50.77</td><td>22.03</td><td>0.610 0.585</td><td>0.092 0.138</td></tr><tr><td>MMIM</td><td>72.36</td><td>69.52</td><td>58.06</td><td>37.71</td><td>0.512</td><td>0.436</td><td>MMIM</td><td>65.28</td><td>60.53</td><td>45.35</td><td>22.32</td><td>0.607</td><td>0.145</td></tr><tr><td>TETFN</td><td>75.86</td><td>73.28</td><td>61.92</td><td>38.80</td><td>0.463</td><td>0.521</td><td>TETFN</td><td>69.88</td></table>

SemMSA still obtains the highest Acc-2, F1, MAE, and Corr, further confirming its robustness under severe intra-modal missingness.

The superior robustness of SemMSA can be attributed to two key designs. First, Cross-modal Semantic Refinement enriches incomplete multimodal inputs with compact latent sentiment semantics from the frozen LLM, which helps alleviate the information loss caused by missing visual, acoustic, or textual tokens. Second, Cross-modal Spectral Alignment jointly aligns semantic, visual, acoustic, and language representations through a kernel Gram matrix, enabling anchor-free modeling of global nonlinear cross-modal dependencies. By combining semantic compensation with spectral alignment, SemMSA learns representations that are both sentiment-aware and cross-modally coherent, leading to stable and robust performance across different datasets and missing rates.

![](images/614541a660ba01e83f8da154278138e8187acc589805208467484c7cd4cbf169.jpg)  
Figure 5: Case study on the MOSI testing set, where SemMSA corrects challenging examples misclassified by P-RMF under incomplete multimodal inputs.

## A.3.6 Case Study

To intuitively evaluate the robustness of SemMSA, we visualize three testing examples from the MOSI dataset in Figure 5, where the red dashed boxes denote missing or corrupted language, visual, and acoustic regions. In the first case, both P-RMF [8] and SemMSA correctly predict the negative sentiment, showing that the remaining multimodal cues are still sufficient for sentiment recognition. However, in the second case, P-RMF incorrectly predicts positive sentiment, possibly because it is misled by local expressions such as “kick ass” while failing to capture the overall negative meaning conveyed by “you don’t like mayonnaise” under missing textual, facial, and acoustic cues. SemMSA correctly identifies the negative sentiment by leveraging latent sentiment semantics to compensate for incomplete evidence. These results suggest that Cross-modal Semantic Refinement helps recover sentiment-relevant semantics from incomplete inputs, while Cross-modal Spectral Alignment further enforces global consistency among language, visual, acoustic, and semantic representations, enabling accurate prediction under challenging intra-modal missingness.

## A.4 Limitations

While SemMSA demonstrates strong performance across a variety of MSA benchmarks, several limitations remain. First, our experiments are conducted on widely used MSA benchmarks, including MOSI, MOSEI, and SIMS. These datasets mainly focus on sentiment analysis in video-based opinion scenarios. The generalization ability of SemMSA to broader multimodal affective understanding tasks, such as emotion recognition in conversation, multimodal sarcasm detection, or real-world long-form human interaction analysis, remains to be further explored. Second, SemMSA mainly exploits the hidden states of the frozen LLM as continuous semantic representations, but the semantic refinement process remains implicit. Since no explicit textual rationale is decoded, the generated latent semantic states are less interpretable than natural-language explanations. Finally, the current CSA module aligns semantic, visual, acoustic, and language representations through a kernel Gram matrix within each instance. Although the instance-level spectral separation loss helps preserve discriminability, the alignment quality may still be affected when most available modality cues are unreliable. In such cases, the refined semantics may not fully compensate for the missing evidence.

## A.5 Societal Impact

SemMSA aims to improve robust multimodal sentiment analysis under incomplete data, which has potential benefits for real-world human-centered applications. By effectively leveraging language, visual, and acoustic cues even when some modalities are missing or corrupted, SemMSA can support more reliable affective understanding in scenarios such as online education, human-computer interaction, healthcare assistance, customer feedback analysis, and intelligent social platforms. Its semantic-aided design may help systems better interpret user attitudes and emotional states under noisy or imperfect sensing conditions, thereby improving accessibility and interaction quality. However, multimodal sentiment analysis also raises several societal concerns. First, sentiment prediction involves sensitive human-centered information, and the use of visual, acoustic, and language data may introduce privacy risks if deployed without appropriate consent, anonymization, and data protection mechanisms. Second, models trained on existing multimodal datasets may inherit demographic, cultural, linguistic, or contextual biases, potentially leading to unfair or inaccurate predictions for underrepresented groups. Third, incorrect sentiment predictions may negatively affect users if applied in high-stakes domains such as hiring and legal decision-making.