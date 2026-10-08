# MIXTURE OF LAYERS: Dynamic Layer Routing for Visual Reasoning

Jeonghwan Kim<sup>1</sup>∗, Sofia Stoica<sup>1</sup>∗, Jiwan Chung<sup>2</sup>, Ansel Blume<sup>1</sup>, Hyeonjeong Ha<sup>1</sup>, Zhenhailong Wang<sup>1</sup>, Xin Luna Dong<sup>3</sup>, Heng Ji<sup>1</sup> <sup>1</sup>University of Illinois Urbana-Champaign, <sup>2</sup>Yonsei University, <sup>3</sup>Meta Reality Labs {jk100, sstoica2, hengji}@illinois.edu

## Abstract

Pre-trained vision encoders contain layer-wise visual representations that differ in spatial granularity, semantic abstraction, and sensitivity to local details. However, most Multimodal Large Language Models (MLLMs) rely on only the final or penultimate vision encoder representations or fixed aggregation rules, making visual abstraction largely query-agnostic and limiting access to fine-grained cues such as small objects, spatial details, text, and subtle visual attributes. In this work, we propose MIXTURE OF LAYERS (MOL), an instruction-conditioned layer routing approach at visual patch level that dynamically aggregates queryrelevant latent representations from intermediate vision encoder layers. Given a text query, MOL predicts routing probabilities over vision encoder layers and performs a top-k sparse aggregation over selected hidden states at either the image level, patch level, or through a hybrid routing mechanism. In doing so, MOL enables query-adaptive access to layer-specific visual features forfine-grained visual reasoning. Our experiments across 7 fine-grained visual reasoning tasks demonstrate substantial performance improvements, especially across fine-grained visual grounding and understanding tasks such as +18.9% improvement on V\* in overall accuracy, +4.5% on HRBench4K, and +16.3% on CharXiv compared to the baseline MLLMs, without resorting to multi-resolution inputs, simple interleaving of multiple vision encoders, or increasing the number of patch tokens. We study vision encoders’ receptive field scales across different layers and their sampling behaviors to provide an in-depth analysis of why layer-wise sampling is helpful, demonstrating that conditional visual representations are a key step towards better visual perception and reasoning in MLLMs. Our project page is available at https://wjdghks950.github.io/mol.github.io/.

## 1 Introduction

Multimodal Large Language Models (MLLMs) process text and images as input, with designs that typically contain three components: one or more pre-trained vision encoders [40, 55, 38, 17, 20], a vision-language connector [36, 25, 42, 10, 28, 29, 5] that maps the visual features extracted from the vision encoder to the pre-trained Large Language Model (LLM) space [32, 47, 16], and the backbone LLM that takes as input the visual patches and textual tokens to perform complex visual understanding and reasoning. Despite their strong image-text understanding capabilities, many existing MLLMs typically assume that the later layer (e.g., final or penultimate) vision encoder representations provide a sufficiently universal abstraction for all downstream tasks as they are strongly aligned with semantic and language-level supervision during training.

Such assumptions, however, become especially limiting for fine-grained multimodal perception and reasoning, where answers often depend on small objects, localized text, subtle attributes, counting, or spatial relations that may not be preserved in a single final-layer representation. We argue that such fixed set of later layers provide limited, query-agnostic access to such granular visual information, even though prior analyses show that spatial and semantic information is distributed differently across vision-encoder layers [14, 15, 7, 23, 3]. Even query-conditioned visual extraction approaches such as Q-Former-based architectures (e.g., BLIP-2 [28], InstructBLIP [10]) or Perceiver Resampler-based designs (e.g., Flamingo [1]) primarily operate on fixed final-layer vision representations rather than dynamically selecting among intermediate-layer features. Recent analyses show that pretrained vision encoders such as CLIP [40] and DINOv2 [38] encode diverse visual features across layers [14, 7, 23], with the most useful layer varying by downstream task [3]. While recent methods exploit intermediate-layer representations [52, 4, 34, 6, 30, 37, 33], they typically rely on manually selected fixed layers [18, 34, 4, 6], increase the number of injected visual tokens [52, 34], or conduct weighted averaging across the layers [33]. More importantly, these choices are largely text-instruction agnostic: the same set of representations are used regardless of visual reasoning task (i.e., prompts). Finally, such static treatment also contrasts with biological vision, where the brain appears to selectively emphasize task-relevant levels of this hierarchy rather than use all representations uniformly [44] . This suggests that effective visual processing in MLLMs should dynamically select the representations most relevant to the current instruction, rather than simply exposing the LLM to more visual features or fixed end-layer representations.

Inspired by this principle, we propose MIXTURE OF LAYERS (MOL), an instruction-guided layer routing mechanism that dynamically selects intermediate-layer visual patch representations from MLLM vision encoders. Rather than relying on a fixed set of visual features, MOL predicts instruction-conditioned routing distributions over vision-encoder layers, enabling different image regions and tasks to access different levels of visual abstraction. Conceptually, the router treats intermediate vision layers as a set of candidate representations and selectively aggregates the most useful ones for the current query. We instantiate MOL with three routing variants: layer-level routing $( \mathbf { M o L } _ { l a y e r } )$ , patch-level routing $( \mathbf { M o L } _ { p a t c h } )$ , and hybrid routing $( \mathbf { M o L } _ { h y b r i d } )$ , which combines the preceding two variants. Specifically, $\mathbf { M o L } _ { l a y e r }$ performs image-level routing across the layers of the pre-trained vision encoder, while $\mathbf { M o L } _ { p a t c h }$ predicts a separate layer distribution for each visual patch, allowing different image regions to draw from different abstraction levels within the vision hierarchy. $\mathbf { M o L } _ { h y b r i d }$ combines the two routing schemes through a weighted product-ofexperts [21] and uses their disagreement to gate a reserve layer (§3.1), balancing local fine-grained evidence with broader semantic context. MOL is particularly effective in fine-grained visual reasoning, showing significant improvements on downstream tasks such as $V ^ { * }$ and CharXiv with gains of 18.9% and 16.3%, respectively. In practice, the added cost is small relative to the overall model: the router contributes negligible parameter overhead (<0.02% over the 13B base model) while leaving token-by-token language decoding unchanged.

Our work makes three key contributions:

• We introduce a novel routing formulation that adaptively selects intermediate visionencoder layer representations conditioned on the input instruction, supporting both imagelevel $\mathbf { M o L } _ { l a y e r }$ and patch-level $\mathbf { M o L } _ { p a t c h }$ decisions.

• We develop a hybrid routing mechanism $\mathbf { M o L } _ { h y b r i d }$ with a disagreement-gated reserve layer (§3.1) that balances local fine-grained evidence and global semantic context while routing across vision encoder layers.

• Across seven different fine-grained visual reasoning benchmarks, MOL substantially improves fine-grained visual reasoning without increasing the number of visual tokens passed into the LLM or relying on multiple vision encoders, making MOL complementary to existing MLLM scaling approaches.

## 2 Related Work

Multimodal Large Language Models Multimodal Large Language Models (MLLMs) ingest visual patch tokens and textual prompts to generate textual responses [36, 56, 27, 2, 10, 28, 11]. Most MLLMs rely on a pre-trained vision encoder, a vision-language connector that projects visual features into the LLM input space, and a pre-trained LLM backbone [36, 56, 27, 2, 11]. While architectures such as BLIP-2 [28], InstructBLIP [10], and Flamingo [1] introduce text-conditioned querying modules such as Q-Former or Perceiver Resampler, they still operate on fixed final or penultimatelayer vision representations. Similarly, visual abstractors and resamplers in models such as mPLUG-

OWL [53] and Honeybee [5] compress or transform already-extracted visual embeddings rather than dynamically selecting intermediate-layer features.

Consequently, most MLLMs continue to rely on the final or penultimate layers of pre-trained vision encoders [40, 55, 38] under the assumption that these representations sufficiently capture the visual information required for downstream reasoning tasks. In contrast, MOL performs instructionconditioned routing across intermediate vision-encoder layers, enabling different image regions and tasks to access different levels of visual abstraction. We leave aside vision encoder-agnostic frameworks such as Emu-3.5 [9] and SOLO [8], since they do not integrate pre-trained vision encoders.

Intermediate Layer Sampling from Pre-trained Vision Encoders Prior work evaluating the beneficial effect of intermediate visual features on MLLMs’ visual reasoning [4, 52, 34, 6, 30] shares a similar motivation to our proposed method: look into the intermediate layers’ hidden states to retrieve useful visual information. However, they focus on either rule-based selection or visual feature-only sampling strategies, limiting their applicability to novel, fine-grained downstream tasks. MOL overcomes prior static approaches to intermediate representations through instruction-guided sampling at multiple levels of representational granularity (i.e., patch- and layer-level). Concurrent work such as TGIF [33] predicts layer weights from the query through an MLP and weighted averages all vision layers. In contrast, $\scriptstyle \mathrm { M o L _ { l a y e r } }$ directly measures query-layer representation compatibility and performs sparse top-k routing, allowing the selected abstraction depth to depend on the visual evidence present in the input rather than only on the semantics of the prompt.

Fine-grained Visual Perception and Reasoning Most widely used visual reasoning benchmarks (e.g., MMMU [54], TextVQA [43], POPE [31], GQA [22]) primarily evaluate coarse-grained understanding and do not stress fine-grained perception in dense, high-resolution scenes critical in realworld tasks such as industrial inspection or document/diagram understanding. Recent benchmarks such as $V ^ { * }$ [50], HRBench [48], NaturalBench [26], RealWorldQA [51], and CharXiv [49] address this gap by requiring models to reason over localized text, small objects, and subtle attributes.

Prior approaches to improving fine-grained reasoning largely focus on increasing the visual token budget via multi-resolution or native-resolution inputs [27, 37, 2], or concatenating patch embeddings from multiple pre-trained vision encoders [42, 46, 23]. By contrast, MOL improves finegrained reasoning by refining the representation of each patch through instruction-conditioned selection of intermediate-layer features, without introducing additional visual tokens or simply concatenating features from multiple vision encoders [42, 46, 23]. This positions MOL as a simple, complementary approach to enhance visual representation quality without scaling input size, allowing it to be integrated with existing multi-patch, high-resolution approaches.

## 3 MIXTURE OF LAYERS: Selective Visual Representations for Fine-Grained Visual Reasoning

## 3.1 Preliminary

We propose MIXTURE OF LAYERS (MOL), an instruction-aligned layer routing mechanism that dynamically selects the most useful vision layers for each input image and text instruction. MOL can be seamlessly attached to different MLLM backbones without changing the language model architecture. Suppose we have aligned vision and text encoders as in dual-encoder architectures like CLIP [40], SigLIP [55], or DINOv2 w/ Txt [24]. Define the vision encoder’s hidden states for its L layers as:

$$
{ \bf H } ^ { 1 } , { \bf H } ^ { 2 } , \ldots , { \bf H } ^ { L } , \qquad { \bf H } ^ { \ell } \in \mathbb { R } ^ { N \times d _ { v } } ,\tag{1}
$$

where $N$ is the number of visual tokens and $d _ { v }$ is the vision encoder hidden dimension. Let the vision encoder’s corresponding text encoder<sup>2</sup> path produce the prompt embedding sequence

$$
\mathbf { T } = [ \mathbf { t } _ { 1 } , \dots , \mathbf { t } _ { M } ] \in \mathbb { R } ^ { M \times d _ { t } } , \quad \bar { \mathbf { t } } = \mathrm { P o o l } ( \mathbf { T } ) \in \mathbb { R } ^ { d _ { t } }\tag{2}
$$

where T is a sequence of M embedded text tokens of dimension $d _ { t }$ and Pool is an arbitrary pooling operator. We use the pooled text representation <sup>¯</sup>t as a query to score vision-layer features for routing:

$$
{ \bf q } = W _ { Q } { \bar { \bf t } } , \qquad { \bf k } _ { n } ^ { \ell } = W _ { K } { \bf h } _ { n } ^ { \ell } , \qquad s _ { n , \ell } ^ { ( p ) } = { \frac { { \bf q } ^ { \top } { \bf k } _ { n } ^ { \ell } } { \sqrt { d _ { v } } } } .\tag{3}
$$

![](images/0346b440c605dd596e7d2f60874f2f99e6649fee5e28aec8f2a0f96a62dd4e02.jpg)  
Figure 1: Overview of MIXTURE OF LAYERS. This figure illustrates a fine-grained visual reasoning scenario where the MLLM should ground its reasoning based on visual representations that accurately capture the queryrelevant visual features. MOL can be divided into three complementary variants: $\mathbf { ( i ) } \mathbf { M o L } _ { l a y e r }$ that generates a query-guided (t<sup>¯</sup>) layer-level probability distribution over the input image; (ii) $\mathbf { M o L } _ { p a t c h } .$ , which generates a query-guided distribution over layers for each visual patch; (iii) $\mathbf { M o L } _ { h y b r i d }$ that measures the disagreement between the $\mathbf { M o L } _ { l a y e r }$ and $\mathbf { M o L } _ { p a t c h }$ and assigns gating weights to the reserve layer (§3.1) representations, which subsequently are pushed through weighted linear combination to form the final patch embedding sequence. We leave out some arrows (e.g., text query to LLM backbone) for brevity.

where $\mathbf { h } _ { n } ^ { \ell }$ is the hidden state for each layer ℓ and patch token n. We pass these hidden states into our routing mechanism, which treats each vision layer as an expert, similar to MoE [41]. Among the L layers of the pre-trained vision encoder, we optionally designate a “reserve layer” $r \in \{ 1 , \ldots , L \}$ as a stable fallback when routing is uncertain or patch-level evidence is noisy; these layers are the last or penultimate layers originally selected by the backbone MLLMs. Reserve layers, as we show in Section 4.4, provide an essential anchor representation to enhance visual reasoning in MOL. For variants with a reserve mechanism, the router selects the top-k non-reserve layers $S _ { n }$ and combines them with a reserve layer r through a learned gate:

$$
\tilde { \mathbf { h } } _ { n } = ( 1 - \gamma _ { n } ) \sum _ { \ell \in S _ { n } } \hat { q } _ { n , \ell } \mathbf { h } _ { n } ^ { \ell } + \gamma _ { n } \mathbf { h } _ { n } ^ { r } , \qquad \hat { q } _ { n , \ell } = \frac { q _ { n , \ell } } { \sum _ { j \in S _ { n } } q _ { n , j } } .\tag{4}
$$

Here, $\gamma _ { n } ~ \in ~ [ 0 , 1 ]$ is the learned reserve-layer gate for patch n, and $q _ { n , \ell }$ denotes the router distribution over non-reserve layers. This prevents the model from collapsing to potentially erroneous layers while still allowing it to exploit specialized intermediate representations when appropriate. We denote the set of non-reserve layers by $\mathcal { L } _ { \mathrm { n o n r e s } } ~ = ~ \{ 1 , \dots , L \} ~ \backslash ~ \{ r \}$ . The final routing distribution, therefore, is decomposed into reserve and non-reserve mass: $p _ { n , r } \ = \ \gamma _ { n } , \qquad p _ { n , \ell } \ =$ $( 1 - \gamma _ { n } ) q _ { n , \ell } , \quad \ell \in \mathcal { L } _ { \mathrm { n o n r e s } }$ , We subsequently train the model with the standard autoregressive loss, $\mathcal { L } _ { \mathrm { L M } }$ , and an auxiliary load-balancing loss, ${ \mathcal { L } } _ { \mathrm { l b } }$ , adopted from Switch transformers [13], to prevent router collapse by encouraging balanced usage of vision encoder layers; $\lambda _ { \mathrm { l b } }$ controls its strength.: $\mathcal { L } = \mathcal { L } _ { \mathrm { L M } } + \lambda _ { \mathrm { l b } } \dot { \mathcal { L } } _ { \mathrm { l b } }$ We further elaborate on the training details in the Appendix.

## 3.2 MIXTURE OF LAYERS

There are three different ways to calculate the image features using our approach:

Layer-level Sampling $( \mathbf { M o L } _ { l a y e r } )$ The simplest variant performs one shared routing decision for the whole image. We first summarize each layer by average-pooling its visual tokens $\bar { \mathbf { h } } ^ { \ell } =$ $\begin{array} { r } { \frac { 1 } { N } \sum _ { n = 1 } ^ { N } \mathbf { h } _ { n } ^ { \ell } } \end{array}$ . Using the text query <sup>¯</sup>t, the router computes a layer distribution $p _ { \ell }$ over $\ell \in \{ 1 , \ldots , L \}$ selects the top-k layers $S = \mathrm { \bar { T o p K } } ( \mathbf { p } , k )$ , and renormalizes their probabilities as $\begin{array} { r } { \hat { p } _ { \ell } = p _ { \ell } / \sum _ { j \in S } p _ { j } } \end{array}$ The routed image representation is then written as: $\begin{array} { r } { \tilde { \mathbf { H } } = \sum _ { \ell \in \mathcal { S } } \hat { p } _ { \ell } \mathbf { H } ^ { \ell } } \end{array}$ Thus, $\mathbf { M o L } _ { l a y e r }$ applies the same selected layers and weights to all patches in the image.

Patch-level Sampling $( \mathbf { M o L } _ { p a t c h } )$ Patch-level routing instead makes a separate layer selection decision for each visual token. For patch n, the router computes a text-conditioned distribution $p _ { n , \ell }$ over layers, selects $S _ { n } = \mathrm { T o p } \bar { \mathrm { K } } ( \mathbf { p } _ { n , : } , k )$ , and renormalizes the selected weights as $\hat { p } _ { n , \ell } =$ $\textstyle p _ { n , \ell } / \sum _ { j \in S _ { n } } p _ { n , j }$ . The routed patch representation is $\begin{array} { r } { \tilde { \mathbf { h } } _ { n } = \sum _ { \ell \in \cal S _ { n } } \hat { p } _ { n , \ell } \mathbf { h } _ { n } ^ { \ell } } \end{array}$

This allows different visual patch tokens $( \mathrm { i . e . }$ ., image regions) to draw from different vision-layer abstractions while keeping the number of output patch tokens unchanged.

Hybrid Sampling $( \mathbf { M o L } _ { h y b r i d } )$ $\mathbf { M o L } _ { h y b r i d }$ unifies layer-level and patch-level sampling to combine the stability of global routing with the flexibility of local routing, improving adaptability across downstream tasks. We first compute a global routing distribution $p _ { \ell } ^ { ( \bar { g } ) }$ and patch-level, local routing distribution $p _ { n , \ell } ^ { ( p ) }$ as

$$
p _ { \ell } ^ { ( g ) } = \mathrm { s o f t m a x } \left( \frac { s _ { \ell } ^ { ( g ) } } { \tau } \right) , \quad p _ { n , \ell } ^ { ( p ) } = \mathrm { s o f t m a x } _ { \ell } \left( \frac { s _ { n , \ell } ^ { ( p ) } } { \tau } \right) .\tag{5}
$$

Here, $p _ { \ell } ^ { ( g ) }$ is the image-level routing probability for layer $\ell ,$ and $p _ { n , \ell } ^ { ( p ) }$ is the patch-level routing probability for patch n and layer ℓ. We then predict a text prompt–dependent mixing weight: $\beta =$ $\mathbf { \dot { \sigma } } \sigma ( \mathbf { w } _ { \beta } ^ { \top } \bar { \mathbf { t } } + \mathbf { \dot { b } } _ { \beta } ) . \mathbf { \dot { \beta } } \in [ 0 , 1 ]$ controls the interpolation between global and local routing, $\mathbf { w } _ { \beta }$ and $b _ { \beta }$ are learned parameters, and $\sigma ( \cdot )$ is the sigmoid function. The hybrid non-reserve routing distribution is computed via a weighted product-of-experts [21]: $q _ { n , \ell } \propto \overline { { \left( p _ { \ell } ^ { ( g ) } \right) ^ { 1 - \beta } \left( p _ { n , \ell } ^ { ( p ) } \right) ^ { \beta } } }$ . Here, $q _ { n , \ell }$ is the fused non-reserve routing probability for patch n and layer $\ell .$ Equivalently, in log-space, this would be written as follows:

$$
\log q _ { n , \ell } = ( 1 - \beta ) \log p _ { \ell } ^ { ( g ) } + \beta \log p _ { n , \ell } ^ { ( p ) } - \log Z _ { n } ,\tag{6}
$$

where $Z _ { n }$ is the normalization constant for patch $n .$

As discussed above (§3.1), we also use a reserve layer, $r ,$ a penultimate or last layer of the vision tower, as a fallback layer. To determine when to fallback or not, we measure the disagreement between local and global routing with

$$
d _ { n } = \mathrm { K L } \left( p _ { n , : } ^ { ( p ) } \parallel p ^ { ( g ) } \right) = \sum _ { \ell \neq r } p _ { n , \ell } ^ { ( p ) } \left( \log p _ { n , \ell } ^ { ( p ) } - \log p _ { \ell } ^ { ( g ) } \right) .\tag{7}
$$

where $d _ { n }$ is the disagreement score for patch $n , \mathrm { K L } ( \cdot \| \cdot )$ denotes KL divergence, and r is the reserve layer index. Then, we map disagreement to a reserve layer probability as $\gamma _ { n } ~ = ~ \sigma ( w _ { d } d _ { n } + b _ { d } )$ wherein $\gamma _ { n } ~ \in ~ [ 0 , 1 ]$ is the reserve weight for patch $n ,$ , and $w _ { d } , b _ { d }$ are learned scalar parameters. Intuitively, the higher the $d _ { n }$ , i.e., the disagreement between the layer- and patch-level distributions, the more we rely on the reserve layer and vice versa. Letting $\dot { S } _ { n } = \mathrm { T o p K } ( \mathbf { q } _ { n , : } , k )$ denote the selected non-reserve layers, the final patch representation is

$$
\tilde { \mathbf { h } } _ { n } = ( 1 - \gamma _ { n } ) \sum _ { \ell \in S _ { n } } \hat { q } _ { n , \ell } \mathbf { h } _ { n } ^ { \ell } + \gamma _ { n } \mathbf { h } _ { n } ^ { r } , \quad \mathrm { w h e r e } \quad \hat { q } _ { n , \ell } = \frac { q _ { n , \ell } } { \sum _ { j \in S _ { n } } q _ { n , j } } .\tag{8}
$$

Here, $\hat { q } _ { n , \ell }$ denotes the renormalized hybrid routing weight, and $\mathbf { h } _ { n } ^ { r }$ is the representation of patch n from the reserve layer r.

Sampling from Layers of Multiple Vision Encoders Unlike prior works that concatenate or interleave last-layer features from multiple vision encoders [46, 45], MOL can first route within each vision encoder and then fuse the routed representations across vision encoders. For each vision encoder m, we apply one of the proposed routing variants, $\alpha _ { m } \in$ {layer, patch, hybrid}, to obtain

$$
\begin{array} { r } { \tilde { \mathbf { H } } ^ { ( m ) } = \mathrm { M o L } _ { \alpha _ { m } } ^ { ( m ) } \left( \{ \mathbf { H } ^ { ( m , \ell ) } \} _ { \ell = 1 } ^ { L _ { m } } , \bar { \mathbf { t } } ^ { ( m ) } \right) , } \end{array}\tag{9}
$$

where $\bar { \mathbf { t } } ^ { ( m ) }$ is the vision encoder-aligned text representation. For instance, in the $\mathrm { C L I P } + \mathrm { D I N O v } 2$ w/ Txt setting in Table 3, we apply $\mathbf { M o L } _ { h y b r i d }$ independently to both vision encoders and fuse their routed representations through concatenation and upsampling. After routing, we project both branches into the language model’s hidden space: $\begin{array} { r } { { \bf Z } ^ { c } = \Pi _ { c } ( \tilde { \bf H } ^ { c } ) , \qquad { \bf Z } ^ { d } = \Pi _ { d } ( \tilde { \bf H } ^ { d } ) } \end{array}$ , where c and d denote CLIP and $\bar { \bf D I N O v } 2$ , respectively, and $\Pi _ { c }$ and $\Pi _ { d }$ are the corresponding vision-language projectors. Here, $\mathbf { Z } ^ { c } \in \mathbb { R } ^ { N _ { c } \times d _ { \mathrm { L L M } } ^ { \bullet } }$ and $\dot { \mathbf { Z } } ^ { d } \in \mathbb { R } ^ { \tilde { N } _ { d } \times d _ { \mathrm { L L M } } }$ , where $N _ { c }$ and $N _ { d }$ are the numbers of patch tokens from each vision encoder, and d<sub>LLM</sub> is the language model’s hidden dimension. Since the two vision encoders produce different numbers of patch tokens, we first upsample the projected DINOv2 features to the CLIP token-grid size: $\hat { \mathbf { Z } } ^ { d } = \operatorname { U p s a m p l e } _ { N _ { d } \to N _ { c } } ( \mathbf { Z } ^ { d } ) \in \mathbb { R } ^ { N _ { c } \times d _ { \mathrm { L L M } } }$ , where

Table 1: Performance across backbones and router types on fine-grained visual reasoning benchmarks. Metrics reported are accuracy. V∗ benchmark is broken into its constituent sub-categories for detailed analysis. Best ∆ denotes the maximum improvement (in absolute accuracy points) achieved by any MOL variant over the corresponding backbone baseline for each metric. We include $\mathbf { \bar { M o L } } _ { \mathrm { l a y e r } }$ for the Vicuna-13B settings with CLIP and DINOv2 w/ Txt as global-routing comparisons. The multi-encoder results are reported separately in Table 3.
<table><tr><td>Method</td><td>all</td><td>GPT4V-hard</td><td>V* OCR</td><td>direct attr.</td><td>rel. pos.</td><td>MMStar</td><td>HRBench8K</td><td>HRBench4K</td><td>RealWorldQA</td><td>NaturalBench</td><td>CharXiv</td></tr><tr><td colspan="10">LLaVA-v1.5-13B (Vicuna-13B + CLIP) [35, 40]</td></tr><tr><td>LLaVA-v1.5-13B</td><td>71.43</td><td>52.94</td><td>50.00</td><td>93.42</td><td>30.00</td><td></td><td>36.88</td><td>43.75</td><td>51.64</td><td>68.93</td><td>23.84</td></tr><tr><td>MoLlaver</td><td>76.89</td><td>58.82</td><td>53.33</td><td>65.22 75.65</td><td>92.11</td><td>29.53</td><td>38.00</td><td>40.62</td><td>52.35</td><td>69.71</td><td>26.25</td></tr><tr><td> $\scriptstyle \mathbf { M o L } _ { \mathrm { p a t c h } }$ </td><td>84.87</td><td>58.82</td><td>60.00</td><td>86.09</td><td>98.68</td><td>30.13</td><td>38.50</td><td>42.25</td><td>51.82</td><td>69.41</td><td>29.17</td></tr><tr><td>MoLhybrid</td><td>84.03</td><td>64.71</td><td>56.67</td><td>85.22</td><td>97.37</td><td>31.07</td><td>38.62</td><td>43.88</td><td>53.14</td><td>70.36</td><td>25.56</td></tr><tr><td>Best ∆</td><td>+13.44</td><td>+11.77</td><td>+10.00</td><td>+20.87</td><td>+5.26</td><td>+1.07</td><td>+1.74</td><td>+0.13</td><td>+1.50</td><td>+1.43</td><td>+5.33</td></tr><tr><td colspan="10">Vicuna-13B + DINOv2 w/ Txt [35, 24]</td></tr><tr><td>Vicuna-13B + DINOv2</td><td>65.55</td><td>76.47</td><td></td><td></td><td></td><td></td><td></td><td>35.38</td><td>50.46</td><td>68.29</td><td>22.92</td></tr><tr><td>MoLlayer</td><td>67.65</td><td>52.94</td><td>40.00 36.67</td><td>67.83</td><td>69.74</td><td>28.87 27.80</td><td>32.75 34.12</td><td>35.38</td><td>50.46</td><td>67.92</td><td>25.94</td></tr><tr><td>MoLpatch</td><td>76.05</td><td>64.71</td><td>40.00</td><td>60.87</td><td>93.42 94.74</td><td>28.47</td><td>33.37</td><td>34.88</td><td>51.02</td><td>66.03</td><td>17.71</td></tr><tr><td>MoLhybrid</td><td>84.45</td><td>70.59</td><td>50.00</td><td>74.78</td><td>98.68</td><td>30.13</td><td>36.50</td><td>39.87</td><td>52.29</td><td>67.89</td><td>26.92</td></tr><tr><td>Best ∆</td><td>+18.90</td><td>-5.88</td><td>+10.00</td><td>86.09 +18.26</td><td>+28.94</td><td>+1.26</td><td>+3.75</td><td>+4.49</td><td>+1.83</td><td>-0.40</td><td>+4.00</td></tr><tr><td colspan="10">Llama-3-8B-V (LLama-3-8B-Instruct + SigLIP) [16, 55]</td></tr><tr><td>Llama-3-8B-V</td><td>54.62</td><td>47.06</td><td></td><td></td><td></td><td>37.73</td><td>44.00</td><td>51.12</td><td>59.87</td><td>74.75</td><td>25.05</td></tr><tr><td>MoLpatch</td><td>52.52</td><td>41.18</td><td>66.67</td><td>56.52</td><td>48.68</td><td></td><td>40.00</td><td>48.88</td><td>58.92</td><td>75.02</td><td>26.16</td></tr><tr><td>MoLhybrid</td><td>63.03</td><td>47.06</td><td>43.33</td><td>60.00</td><td>47.37 50.00</td><td>37.07 33.40</td><td>42.13</td><td>45.00</td><td>45.62</td><td>72.05</td><td>26.82</td></tr><tr><td>Best ∆</td><td>+8.42</td><td>+0.00</td><td>76.67 +10.00</td><td>70.43 +13.91</td><td>+1.32</td><td>-0.66</td><td>-1.87</td><td>-2.24</td><td>-0.95</td><td>+0.27</td><td>+1.77</td></tr><tr><td colspan="10">Phi-1.5-1.3B + SigLIP [32, 55]</td></tr><tr><td>Phi-1.5-1.3B</td><td>42.01</td><td>23.52</td><td>26.66</td><td>35.65</td><td>61.84</td><td>34.93</td><td>32.00</td><td>36.50</td><td>34.93</td><td>67.31</td><td>4.07</td></tr><tr><td>MoLpatch MoLhybrid</td><td>39.92 50.84</td><td>29.41</td><td>50.00</td><td>28.70</td><td>55.26</td><td>33.60</td><td>30.63</td><td>33.12</td><td>38.82</td><td>60.86</td><td>19.65</td></tr><tr><td>Best Δ</td><td>+8.83</td><td>23.52</td><td>43.33</td><td>42.60</td><td>72.36</td><td>34.93</td><td>32.00</td><td>37.25</td><td>39.74</td><td>68.00</td><td>20.46</td></tr><tr><td></td><td></td><td>+0.00</td><td>+16.67</td><td>+6.95</td><td>+10.52</td><td>+0.00</td><td>+0.00</td><td>+0.75</td><td>+4.81</td><td>+0.69</td><td>+16.39</td></tr></table>

Table 2: Performance comparison against existing state-of-the-art methods on multi-layer visual feature integration. With all the existing baselines providing LLaVA-baseline checkpoints, we use LLaVA-v1.5- 13B [35] as our baseline in this experiment.
<table><tr><td rowspan="2">Method</td><td colspan="5">V*</td><td rowspan="2">MMStar</td><td rowspan="2">HRBench8K</td><td rowspan="2">HRBench4K</td><td rowspan="2">RealWorldQA</td><td rowspan="2">NaturalBench</td><td rowspan="2">CharXiv</td></tr><tr><td>all</td><td>GPT4V-hard</td><td>OCR</td><td>direct attr.</td><td>rel. pos.</td></tr><tr><td>LLaVA-v1.5-13B</td><td>71.43</td><td>52.94</td><td>50.00</td><td>65.22</td><td>93.42</td><td>30.00</td><td>36.88</td><td>43.75</td><td>51.64</td><td>68.93</td><td>23.84</td></tr><tr><td>TokenPacker [29]</td><td>73.53</td><td>47.06</td><td>63.33</td><td>66.09</td><td>93.74</td><td>29.07</td><td>38.00</td><td>43.38</td><td>54.12</td><td>67.63</td><td>22.76</td></tr><tr><td>MLVF [34]</td><td>78.92</td><td>45.88</td><td>54.00</td><td>79.13</td><td>96.05</td><td>29.84</td><td>36.88</td><td>42.50</td><td>51.90</td><td>69.80</td><td>24.72</td></tr><tr><td>DeepStack-L [37]</td><td>65.13</td><td>64.71</td><td>46.67</td><td>56.52</td><td>85.53</td><td>29.26</td><td>37.63</td><td>43.75</td><td>57.77</td><td>69.10</td><td>21.31</td></tr><tr><td>Dense Connector [52]</td><td>81.35</td><td>47.05</td><td>56.67</td><td>83.47</td><td>98.68</td><td>30.06</td><td>37.12</td><td>43.75</td><td>52.41</td><td>71.41</td><td>25.23</td></tr><tr><td>IGVA [34]</td><td>70.38</td><td>45.12</td><td>52.20</td><td>66.79</td><td>88.60</td><td>29.26</td><td>36.17</td><td>42.10</td><td>51.85</td><td>67.63</td><td>23.89</td></tr><tr><td>TGIF [33]</td><td>63.87</td><td>52.94</td><td>50.00</td><td>56.52</td><td>82.89</td><td>29.07</td><td>33.75</td><td>40.63</td><td>54.38</td><td>69.20</td><td>20.00</td></tr><tr><td>MoLlayer</td><td>76.89</td><td>58.82</td><td>53.33</td><td>75.65</td><td>92.11</td><td>29.53</td><td>38.00</td><td>40.62</td><td>52.35</td><td>69.71</td><td>27.85</td></tr><tr><td>MoLpatch</td><td>84.87</td><td>58.82</td><td>60.00</td><td>86.09</td><td>98.68</td><td>30.13</td><td>38.50</td><td>42.25</td><td>51.82</td><td>69.41</td><td>28.65</td></tr><tr><td>MoLhybrid</td><td>84.03</td><td>64.71</td><td>56.67</td><td>85.22</td><td>97.37</td><td>30.47</td><td>38.62</td><td>43.88</td><td>53.14</td><td>70.36</td><td>26.97</td></tr></table>

Table 3: Comparison of single- and multi-encoder routing with a Vicuna-13B backbone. Interleaved-MoF interleaves CLIP and DINOv2 patch inputs. MOL routes within one or both encoders; the dual-encoder variant uses concatenation/upsampling followed by a learned projector. Best ∆ reports the largest improvement among the three MOL configurations over Interleaved-MoF, computed from the displayed scores in percentage points. denotes DINOv2 with an aligned text encoder.
<table><tr><td>Method</td><td>Visual Input</td><td colspan="2">GPT4V-hard</td><td colspan="2">V* OCR</td><td colspan="2"></td><td rowspan="2">MMStar HRBench8K</td><td rowspan="2">HRBench4K</td><td rowspan="2">RealWorldQA</td><td rowspan="2">NaturalBench</td><td rowspan="2">CharXiv</td></tr><tr><td></td><td></td><td>all</td><td></td><td></td><td>direct attr.</td><td> $r e l . \ p o s .$ </td><td></td></tr><tr><td>Interleaved-MoF [46]</td><td>CLIP + DINOv2</td><td>82.35</td><td>58.82</td><td>53.33</td><td>82.61</td><td>98.68</td><td>31.87</td><td>35.75</td><td>44.12</td><td>57.77</td><td>67.26</td><td>26.73</td></tr><tr><td>MOLhybrid</td><td>CLIP</td><td>84.03</td><td>64.71</td><td>56.67</td><td>85.22</td><td>97.37</td><td>31.07</td><td>38.62</td><td>43.88</td><td>53.14</td><td>70.36</td><td>25.56</td></tr><tr><td>MoLhybrid</td><td>DINOv2†</td><td>84.45</td><td>70.59</td><td>50.00</td><td>86.09</td><td>98.68</td><td>30.13</td><td>36.50</td><td>39.87</td><td>52.29</td><td>67.89</td><td>26.92</td></tr><tr><td>MoLhybrid - Dual Enc.</td><td>CLIP + DINOv2†</td><td>90.34</td><td>82.35</td><td>66.67</td><td>92.17</td><td>98.68</td><td>32.00</td><td>37.62</td><td>43.50</td><td>52.55</td><td>69.72</td><td>31.13</td></tr><tr><td>Best ∆</td><td></td><td>+7.99</td><td>+23.53</td><td>+13.34</td><td>+9.56</td><td>+0.00</td><td>+0.13</td><td>+2.87</td><td>-0.24</td><td>-4.63</td><td>+3.10</td><td>+4.40</td></tr></table>

Upsample uses bilinear interpolation. In our setting, the DINOv2 and CLIP branches produce 256 and 576 patch tokens, respectively. We subsequently concatenate the two representations along the feature dimension and pass them through a learned linear projector:

$$
\mathbf { F } = [ \mathbf { Z } ^ { c }  \hat { \mathbf { Z } } ^ { d }  \mathbf { W } _ { f } + \mathbf { 1 } _ { N _ { c } } \mathbf { b } _ { f } ^ { \top } , \qquad \mathbf { F } \in \mathbb { R } ^ { N _ { c } \times d _ { \mathrm { L L M } } } ,\tag{10}
$$

where ∥ denotes feature concatenation, $\mathbf { W } _ { f } ~ \in ~ \mathbb { R } ^ { 2 d _ { \mathrm { L L M } } \times d _ { \mathrm { L L M } } }$ and $\mathbf { b } _ { f } \in \mathbb { R } ^ { d _ { \mathrm { L L M } } }$ are learned parameters, and $\mathbf { 1 } _ { N _ { c } }$ broadcasts the bias across patch tokens. This fusion allows the model to combine query-relevant representations from both vision encoders while preserving the CLIP branch’s output token length. The resulting patch embedding sequence F is then passed to the language model.

## 4 Experiments

In this section, we elaborate on the experimental setup of the backbone multimodal LLMs (MLLMs) we apply MIXTURE OF LAYERS (MOL) to, discuss the downstream fine-grained visual understanding and reasoning benchmarks, detail the analytical frameworks employed to study layer-wise sampling behaviors, and demonstrate how MOL sampled layer representations enhance fine-grained visual understanding in MLLMs.

Models Our work builds MOL in a fully reproducible MLLM training framework built on top of [35, 19]. This framework provides both the vision-language connector pre-training and multimodal instruction-tuning stages, unlike open-weight-only models [2, 16] where the underlying training pipeline data is not fully available. Namely, in this work we use LLaVA-1.5-13B [35], Llama-3-8B-$\bar { \mathrm { ~ V ~ } } [ 1 6 ]$ , and Phi-1.5-1.3B[32] + SigLIP [55] as experimental baselines, all of which are open-sourced by the Bunny framework [19]. The pre-trained vision encoders span CLIP [40], DINOv2 w/ Txt[38], and SigLIP [55]. For DINOv2 w/ Txt we pair it with Vicuna-13B, which is equivalent of a LLaVA v1.5-13B with DINOv2 w/ Txt as the vision encoder. For additional training details, please refer to Appendix.

Dataset We adopt the exact pre-training and instruction-tuning datasets provided by the individual backbone MLLMs [35, 11, 19] to ensure a fair comparison. For fine-grained visual understanding benchmarks, we employ V\* [50], HRBench4K, HRBench8K [48], NaturalBench [26], Real-WorldQA [51], and CharXiv [49] to evaluate the baseline and MOL-enhanced capabilities of different MLLMs.

## 4.1 Effect of MIXTURE OF LAYERS on Model Perception

Fine-grained Visual Understanding Table 1 shows that MOL improves fine-grained visual reasoning across multiple backbone architectures and vision encoders, across different parameter sizes (13B, 8B, 1.5B). On the primary LLaVA-v1.5-13B setting, MOL yields consistent gains over the baseline, with especially large improvements on V\* overall (+13.44), direct attribute recognition (+20.87), OCR (+10.00), and CharXiv (+5.33). Similar gains appear with DINOv2 w/ Txt, where $\mathbf { M o L } _ { \mathrm { h y b r i d } }$ improves $V ^ { * } \log + 1 8 . 9 0$ and relative position by +28.94, suggesting that adaptive layer routing is not specific to CLIP-based vision encoders. The gains are also visible in smaller or different backbones: for Phi-1.5-1.3B, MOL substantially improves OCR (+16.67), relative position (+10.52), and CharXiv (+16.39). Across settings, $\mathbf { M o L } _ { \mathrm { p a t c h } }$ is often strongest on localized perceptual categories, while $\mathbf { M o L } _ { \mathrm { h y b r i d } }$ tends to perform better on benchmarks requiring both local grounding and broader reasoning. These results suggest that selecting task-relevant intermediate layers is a general mechanism for improving fine-grained perception, rather than a backbone-specific optimization.

Comparison against State-of-the-Art In Table 2, we compare MOL with prior approaches that aggregate intermediate visual features via multi-resolution stacking (DeepStack [37]), token compression and fusion (TokenPacker [29]), or dense cross-layer connections (MLVF [34], Dense Connector [52]). Unlike these fixed aggregation schemes, MOL adaptively routes vision-layer representations conditioned on the input instruction. MOL consistently improves fine-grained visual reasoning. $\mathbf { M o L } _ { \mathrm { p a t c h } }$ achieves the best overall $V ^ { * }$ performance, outperforming Dense Connector by 3.52 points, with particularly strong gains on localized categories such as OCR and direct attribute. $\mathbf { M o L } _ { \mathrm { h y b r i d } }$ further improves performance on more challenging benchmarks (e.g., GPT4V-hard, MM-Star, HRBench), indicating the benefit of combining global and patch-level routing. Importantly, MOL operates with a single vision encoder yet matches or outperforms prior multi-encoder approaches (e.g., Interleaved-MoF [46]) as shown in Table 3, demonstrating that adaptive layer routing can be more effective than increasing visual diversity through simple interleaving of additional encoder features or more patch tokens.

Coarse-grained Visual Understanding Although MOL targets fine-grained perception, we also evaluate standard VQA benchmarks in Table 4. MOL largely preserves coarse-grained reasoning performance, with small gains on POPE, GQA, and MMMU, and minimal change on TextVQA. We include the full results here and defer additional analysis to the Appendix.

Table 4: Coarse-grained VQA performance.
<table><tr><td>Model</td><td>POPE</td><td>GQA TextVQA MMMU</td><td></td></tr><tr><td>LLaVA-v1.5-13B (Vicuna-13B + CLIP) [35, 40]</td><td colspan="3"></td></tr><tr><td> $\mathbf { L L a V } \mathbf { A – v } \mathbf { l } . 5 \mathbf { – } 1 3 \mathbf { B }$ </td><td>85.60 63.26</td><td>61.25</td><td>36.89</td></tr><tr><td>MoLpatch</td><td>86.23</td><td>63.73 61.09</td><td>38.22</td></tr><tr><td>MoLhybrid</td><td>86.47</td><td>63.75 61.11</td><td>37.00</td></tr><tr><td>Best </td><td>+0.87</td><td>+0.49</td><td>+1.33</td></tr><tr><td colspan="4">Vicuna-13B + DINOv2 w/ Txt [35, 24]</td></tr><tr><td>Vicuna-13B + DINOv2</td><td>84.05</td><td>61.46</td><td>34.67</td></tr><tr><td> $\mathsf { M o L } _ { \mathrm { p a t c h } }$ </td><td>84.12 59.96</td><td>46.87 47.12</td><td>34.11</td></tr><tr><td> $\scriptstyle \mathbf { M o L } _ { \mathrm { h y b r i d } } ^ { \mathrm { ~ } }$ </td><td>84.08 60.57</td><td>46.80</td><td>34.67</td></tr><tr><td>Best △</td><td>+0.07</td><td>-0.89 +0.25</td><td>+0.00</td></tr></table>

![](images/6bbd29c310bd6712413f6da3c1609a9a98fbf83878b5238efb8996acc45a1e9a.jpg)  
Figure 2: Layer selection behaviors of MIXTURE OF LAYERS. Average normalized routing probability across vision-encoder layers for LLaVA-v1.5-13B on V\*Bench, MMStar, and GQA categories. Background intensity shows category-normalized receptive field (RF) [39]; lighter regions indicate more localized layers. Markers denote selected top-k layers (k = 4). DINOv2 variants are shown in Appendix Figure 4.

## 4.2 MIXTURE OF LAYERS with Multiple Vision Encoders

Setup We extend hybrid routing to a Vicuna-13B backbone with CLIP and DINOv2 w/ Txt, applying a separate text-conditioned router to each vision encoder. The evaluated configuration selects one non-reserve layer per patch (k = 1) and combines it with the encoder’s reserve representation using the disagreement gate. Each routed stream is projected to the language model’s hidden dimension. The concatenation/upsampling fusion resizes the DINOv2 stream to the CLIP token-grid size, concatenates the two streams along the feature dimension, and applies a learned linear projection to produce 576 visual tokens. Table 3 compares this configuration with Interleaved-MoF [46], which also uses a Vicuna-13B backbone. This is a comparison between complete systems; the pretrained encoders and fusion mechanisms are not a controlled routing-only ablation.

Results The multi-encoder configuration achieves 90.34 on V\*, compared with 82.35 for Interleaved-MoF. The largest gain is on GPT4V-hard (82.35 versus 58.82), followed by OCR (66.67 versus 53.33) and direct attributes (92.17 versus 82.61); relative-position accuracy is tied at 98.68. These changes correspond to 19 additional correct answers out of 238: four on GPT4V-hard, four on OCR, and eleven on direct attributes. The same checkpoint achieves 31.13 on CharXiv. The remaining reported comparisons are mixed, including lower HRBench4K and RealWorldQA scores. We therefore do not infer a uniform advantage across benchmarks from this comparison.

Interpretation and Insights A plausible explanation is that routing and encoder diversity address different sources of lost visual information. Routing exposes query-relevant intermediate representations within each encoder, while the two encoders supply different learned representations of the image. The learned channel projection gives the language model access to both routed streams without extending the sequence beyond the CLIP token count. Combining an image-level routing preference with patch-specific evidence also allows the selected abstraction to vary across regions, while the reserve representation provides an anchor. The concentration of gains in OCR, attributes, and challenging visual questions is consistent with this explanation; the unchanged relative-position result suggests that extra visual diversity is less useful when the baseline is already near ceiling. However, this comparison does not establish which component causes the gains. A matched norouter model with the same fusion, training schedule, and checkpoint-selection rule is needed to isolate routing from fusion and encoder choice.

## 4.3 Analysis of Sampling Behavior

Layer-wise Sampling Behavior Figure 2 shows that MOL learns non-uniform, task-dependent routing patterns across benchmarks and question categories. The layer-level router $( \mathbf { M o L } _ { l a y e r } )$ which assigns one distribution to the entire image consistently prioritizes early layers, especially for categories requiring localized evidence such as OCR, direct attributes, and relative position. This is because global routing must rely on features that are consistently informative across the whole image, and early layers preserve fine-grained spatial details before they are aggregated. By contrast, the patch-level router $( \mathbf { M o L } _ { p a t c h } )$ places more mass on later layers. Since it computes routing independently for each patch, it can leverage semantically enriched representations that encode broader context, allowing individual regions to be interpreted relative to the overall scene and query. As a result, later-layer selections often reflect patches that benefit from contextualized, higher-level features. The hybrid router $( \mathbf { M o L } _ { h y b r i d } )$ balances these two behaviors by combining global and patch-level routing, yielding more consistent mid-to-late layer usage. Overall, these patterns suggest that global routing favors detail-preserving features, while patch-level routing can exploit contextualized representations, motivating the need for both.

![](images/2d2f798b32e7fdbb110f0f219240635e4cfd76ffeca3f05fec6965ba8f4780fd.jpg)  
Figure 3: Self-attention heatmap visualization of the selected layers aggregated across all the queries of the layers on the selected layers of CLIP-ViT-L14. For additional qualitative analysis on the selected layers of the DINOv2 encoder, please refer to Figure 5.

Table 5: Ablation on vision encoder-aligned text encoder. $\mathbf { M o L } _ { i m a g e }$ uses image-only routing, where the pooled embeddings of each vision encoder layer is converted to logits to form a distribution over the layers, while $\mathbf { M o L } _ { h y b r i d }$ uses text-conditioned routing. We denote the improvement / degradation compared to the baseline model in green / red. We use a Vicuna-13B + CLIP backbone.
<table><tr><td>Model</td><td>V*</td><td>GQA</td><td>TextVQA</td><td>MMMU</td></tr><tr><td> $\mathbf { M o L } _ { \mathrm { i m a g e } }$ </td><td> $7 7 . 3 1 \ ( + 5 . 8 8 )$ </td><td> $6 2 . 9 4 \ ( - 0 . 3 2 )$ </td><td>58.17 (-3.08)</td><td> $3 6 . 0 0 ( - 0 . 8 9 )$ </td></tr><tr><td> $\mathbf { M o L } _ { \mathrm { h y b r i d } }$ </td><td>84.03 (+12.60)</td><td>63.75 (+0.49)</td><td>61.11 (-0.14)</td><td> ${ \bf 3 7 . 0 0 } \left( + 0 . 1 1 \right)$ </td></tr></table>

Table 6: Ablations on routing design. (a) Removing the reserve layer degrades performance, indicating its importance as a fallback when routing is uncertain. In tandem, having an anchor (reserve) layer with complementary k layers helps MLLMs perform grounded fine-grained visual reasoning. (b) Varying k shows that selecting a small set of layers suffices for strong fine-grained reasoning.
<table><tr><td colspan="2"></td><td> $\mathbf { M o L } _ { \mathrm { h y b r i d } }$ </td></tr><tr><td>Benchmark</td><td> $\mathbf { M o L } _ { \mathrm { h y b r i d } }$ </td><td>No Reserve</td></tr><tr><td>V*</td><td>82.35</td><td>78.57 (-3.78%)</td></tr><tr><td>HRBench8K</td><td>38.62</td><td>35.50 (-3.12%)</td></tr><tr><td>HRBench4K</td><td>43.88</td><td>38.75 (-5.13%)</td></tr><tr><td>POPE</td><td>86.50</td><td>82.10 (-4.40%)</td></tr><tr><td>MMMU</td><td>37.00</td><td>34.00 (-3.00%)</td></tr></table>

<table><tr><td rowspan="2">Benchmark</td><td colspan="4"> $\mathbf { M o L } _ { \mathrm { p a t c h } }$ </td><td colspan="4"> $\mathbf { M o L } _ { \mathrm { h y b r i d } }$ </td></tr><tr><td>k = 1</td><td>k = 2</td><td>k = 4</td><td>Best</td><td>k = 1</td><td>k = 2</td><td>k = 4</td><td>Best</td></tr><tr><td>V*</td><td>76.89</td><td>73.95</td><td>79.41</td><td>79.41</td><td>84.03</td><td>76.89</td><td>67.65</td><td>84.03</td></tr><tr><td>HRBench8K</td><td>39.38</td><td>36.50</td><td>37.25</td><td>39.38</td><td>38.62</td><td>38.25</td><td>34.88</td><td>38.62</td></tr><tr><td>HRBench4K</td><td>43.12</td><td>42.75</td><td>43.75</td><td>43.75</td><td>43.88</td><td>44.00</td><td>43.50</td><td>44.00</td></tr><tr><td>MMStar</td><td>30.13</td><td>31.93</td><td>29.93</td><td>31.93</td><td>30.47</td><td>30.00</td><td>29.73</td><td>30.47</td></tr><tr><td>NaturalBench</td><td>69.41</td><td>68.07</td><td>68.61</td><td>69.41</td><td>69.96</td><td>68.91</td><td>70.45</td><td>70.45</td></tr><tr><td>CharXiv</td><td>28.65</td><td>28.47</td><td>27.94</td><td>28.65</td><td>26.13</td><td>26.25</td><td>26.97</td><td>26.97</td></tr><tr><td>RealWorldQA</td><td>55.42</td><td>54.90</td><td>55.16</td><td>55.42</td><td>55.03</td><td>56.60</td><td>56.34</td><td>56.60</td></tr></table>

(a) Effect of removing the reserve layer.  
(b) Effect of k on fine-grained visual reasoning.

Receptive Field Analysis across Different Layers To gain a deeper understanding of the routing behavior in light of the above layer-wise sampling behiavor analysis, we analyze the receptive field (RF) of each layer using the average self-attention distance [12]: $\begin{array} { r } { \mathrm { R F } ^ { \ell } = \mathbb { E } _ { i } \left[ \sum _ { j } A _ { i , j } ^ { \ell } d ( i , j ) \right] } \end{array}$ . RF increases with depth: early layers capture local interactions, while later layers aggregate broader spatial context. This provides a mechanistic explanation for the routing patterns. $\mathbf { M o L } _ { \mathrm { l a y e r } }$ favors early, small-RF layers because global routing must rely on detail-preserving features that are consistently informative across the image. In contrast, $\mathbf { M o L } _ { \mathrm { p a t c h } }$ often selects later, large-RF layers, since patch-wise routing can leverage contextualized representations to interpret each region relative to the scene and query. $\mathbf { M o L } _ { \mathrm { h y b r i d } }$ combines both regimes, enabling access to both local evidence and global context. Thus, RF analysis complements the routing observations by showing that MOL adapts to the inherent locality-context trade-off encoded across vision layers.

## 4.4 Ablations

Role of Vision Encoder-Aligned Text Encoders Table 5 studies whether routing should be conditioned on the text encoder aligned with the vision encoder. We compare $\mathbf { M o L } _ { i m a g e }$ , which predicts layer-routing probabilities from vision hidden states alone, against the text-conditioned $\mathbf { M o L } _ { h y b r i d } .$ Vision-only routing substantially underperforms the baseline, especially on $V ^ { * }$ , where it drops by 9.16 points. We hypothesize that image-only routing lacks a task-specific signal for determining which layer representations are relevant, making the router more prone to selecting inconsistent or semantically mismatched layers during joint optimization. In contrast, $\mathbf { M o L } _ { h y b r i d }$ improves $V ^ { * }$ by 13.44 points and yields small gains on GQA and MMMU while largely preserving TextVQA performance. These results suggest that layer selection should not be treated as an image-only decision: the input instruction provides an essential signal for selecting which visual abstractions are relevant for the current task.

Role of Reserve Layers and the Number of Layers Table 6 highlights two key properties of MOL. First, removing the reserve layer consistently degrades performance (3-5 points across benchmarks), indicating that it acts as a stable anchor that preserves strong, text-aligned representations when routing over intermediate layers is uncertain or noisy. Second, varying k shows that performance does not benefit from aggregating many layers; instead, small k values (e.g., $k = 1 \mathrm { o r } k = 4 )$ are sufficient and often optimal. Together, these results suggest that effective fine-grained visual reasoning arises from a balance between stability and selectivity: the reserve layer provides robustness, while sparse routing over a few task-relevant layers avoids introducing noise from irrelevant representations.

Qualitative Analysis on the Self-Attention Layers of Vision Encoders Figure 3 shows that MOL selects layers whose self-attention maps align with query-relevant regions. For localized tasks such as OCR and attribute recognition, mid-to-early layers (< layers 14-15) attend to finegrained details (e.g., text on signs, small objects), while later layers highlight broader semantic regions (e.g., people, objects). For spatial reasoning queries, attention is distributed across multiple relevant regions, enabling the model to resolve relationships such as left/right positioning. We defer Limitations and Broader Impacts to Appendix A due to lack of space.

## 5 Conclusion

We introduced MIXTURE OF LAYERS (MOL), an instruction-guided layer routing framework for MLLMs that dynamically selects intermediate vision-encoder representations for fine-grained visual reasoning conditioned on the input textual instruction. Instead of relying on fixed final-layer features or increasing the visual token budget, MOL selects task-relevant patch representations through layerlevel, patch-level, and hybrid routing for both a single and dual vision encoder schemes. Across fine-grained visual understanding benchmarks, MOL consistently improves over backbone MLLMs and prior static feature-integration methods, while remaining competitive on conventional coarsegrained VQA tasks. Our analysis further shows that different routing strategies exploit different regions of the vision-encoder hierarchy: layer-level routing favors localized early layers, patchlevel routing leverages contextualized later layers, and hybrid routing effectively combines both regimes to reap the benefit of both worlds. Overall, MOL demonstrates that dynamically selecting the appropriate visual abstraction can be a more effective alternative to scaling visual inputs or adding multiple vision encoders for improving MLLM perception and reasoning.

## 6 Acknowledgements

This research is based upon work supported by DARPA ITM Program No. FA8650-23-C-7316, the Office of the Director of National Intelligence (ODNI), Intelligence Advanced Research Projects Activity (IARPA), via 560000C260018, CapitalOne-Illinois Center for Generative AI Safety, Knowledge Systems, and Cybersecurity (ASKS), Amazon-Illinois Center on AI for Interactive Conversational Experiences (AICE). The views and conclusions contained herein are those of the authors and should not be interpreted as necessarily representing the official policies, either expressed or implied, of DARPA, IARPA, ODNI, DARPA, or the U.S. Government. The U.S. Government is authorized to reproduce and distribute reprints for governmental purposes notwithstanding any copyright annotation therein. This research used the Delta and DeltaAI advanced computing and data resources, which are supported by the National Science Foundation (award OAC 2320345 and award OAC 2005572) and the State of Illinois. Delta and DeltaAI are joint efforts of the University of Illinois Urbana-Champaign and its National Center for Supercomputing Applications. The AI Research Institutes program by National Science Foundation and the Institute of Education Sciences, U.S. Department of Education through Award 2229873 - AI Institute for Transforming Education for Children with Speech and Language Processing Challenges.

## References

[1] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, Roman Ring, Eliza Rutherford, Serkan Cabi, Tengda Han, Zhitao Gong, Sina Samangooei, Marianne Monteiro, Jacob Menick, Sebastian Borgeaud, Andrew Brock, Aida Nematzadeh, Sahand Sharifzadeh, Mikolaj Binkowski, Ricardo Barreira, Oriol Vinyals, Andrew Zisserman, and Karen Simonyan. Flamingo: a visual language model for few-shot learning. In Alice H. Oh, Alekh Agarwal, Danielle Belgrave, and Kyunghyun Cho, editors, Advances in Neural Information Processing Systems, 2022.

[2] Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, et al. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025.

[3] Daniel Bolya, Po-Yao Huang, Peize Sun, Jang Hyun Cho, Andrea Madotto, Chen Wei, Tengyu Ma, Jiale Zhi, Jathushan Rajasegaran, Hanoona Rasheed, Junke Wang, Marco Monteiro, Hu Xu, Shiyu Dong, Nikhila Ravi, Daniel Li, Piotr Dollár, and Christoph Feichtenhofer. Perception encoder: The best visual embeddings are not at the output of the network, 2025.

[4] Yue Cao, Yangzhou Liu, Zhe Chen, Guangchen Shi, Wenhai Wang, Danhuai Zhao, and Tong Lu. Mmfuser: Multimodal multi-layer feature fuser for fine-grained vision-language understanding. arXiv preprint arXiv:2410.11829, 2024.

[5] Junbum Cha, Wooyoung Kang, Jonghwan Mun, and Byungseok Roh. Honeybee: Localityenhanced projector for multimodal llm. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13817–13827, 2024.

[6] Haoran Chen, Junyan Lin, Xinghao Chen, Yue Fan, Jianfeng Dong, Xin Jin, Hui Su, Jinlan Fu, and Xiaoyu Shen. Multimodal language models see better when they look shallower. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 6688–6706, 2025.

[7] Haozhe Chen, Junfeng Yang, Carl Vondrick, and Chengzhi Mao. INViTE: INterpret and control vision-language models with text explanations. In The Twelfth International Conference on Learning Representations, 2024.

[8] Yangyi Chen, Xingyao Wang, Hao Peng, and Heng Ji. Solo: A single transformer for scalable vision-language modeling. Transactions on Machine Learning Research, 2024, 2024.

[9] Yufeng Cui, Honghao Chen, Haoge Deng, Xu Huang, Xinghang Li, Jirong Liu, Yang Liu, Zhuoyan Luo, Jinsheng Wang, Wenxuan Wang, et al. Emu3. 5: Native multimodal models are world learners. arXiv preprint arXiv:2510.26583, 2025.

[10] Wenliang Dai, Junnan Li, Dongxu Li, Anthony Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale Fung, and Steven Hoi. InstructBLIP: Towards general-purpose vision-language models with instruction tuning. In Thirty-seventh Conference on Neural Information Processing Systems, 2023.

[11] Matt Deitke, Christopher Clark, Sangho Lee, Rohun Tripathi, Yue Yang, Jae Sung Park, Mohammadreza Salehi, Niklas Muennighoff, Kyle Lo, Luca Soldaini, et al. Molmo and pixmo: Open weights and open data for state-of-the-art vision-language models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 91–104, 2025.

[12] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In International Conference on Learning Representations, 2021.

[13] William Fedus, Barret Zoph, and Noam Shazeer. Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity. Journal of Machine Learning Research, 23(120):1–39, 2022.

[14] Yossi Gandelsman, Alexei A Efros, and Jacob Steinhardt. Interpreting clip’s image representation via text-based decomposition. In The Twelfth International Conference on Learning Representations, 2024.

[15] Amin Ghiasi, Hamid Kazemi, Eitan Borgnia, Steven Reich, Manli Shu, Micah Goldblum, Andrew Gordon Wilson, and Tom Goldstein. What do vision transformers learn? a visual exploration. arXiv preprint arXiv:2212.06727, 2022.

[16] Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.

[17] Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross Girshick. Momentum contrast for unsupervised visual representation learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 9729–9738, 2020.

[18] Liqi He, Zuchao Li, Xiantao Cai, and Ping Wang. Multi-modal latent space learning for chainof-thought reasoning in language models. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 38, pages 18180–18187, 2024.

[19] Muyang He, Yexin Liu, Boya Wu, Jianhao Yuan, Yueze Wang, Tiejun Huang, and Bo Zhao. Efficient multimodal learning from data-centric perspective. arXiv preprint arXiv:2402.11530, 2024.

[20] Greg Heinrich, Mike Ranzinger, Hongxu Yin, Yao Lu, Jan Kautz, Andrew Tao, Bryan Catanzaro, and Pavlo Molchanov. Radiov2. 5: Improved baselines for agglomerative vision foundation models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 22487–22497, 2025.

[21] Geoffrey E Hinton. Products of experts. In Proceedings ofthe Ninth International Conference on Artificial Neural Networks (ICANN’99), volume 1, pages 1–6. IEE, 1999.

[22] Drew A Hudson and Christopher D Manning. Gqa: A new dataset for real-world visual reasoning and compositional question answering. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6700–6709, 2019.

[23] Dongsheng Jiang, Yuchen Liu, Songlin Liu, XIAOPENG ZHANG, Jin Li, Hongkai Xiong, and Qi Tian. From CLIP to DINO: Visual encoders shout in multi-modal large language models, 2024.

[24] Cijo Jose, Théo Moutakanni, Dahyun Kang, Federico Baldassarre, Timothée Darcet, Hu Xu, Daniel Li, Marc Szafraniec, Michaël Ramamonjisoa, Maxime Oquab, et al. Dinov2 meets text: A unified framework for image-and pixel-level vision-language alignment. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 24905–24916, 2025.

[25] Jing Yu Koh, Ruslan Salakhutdinov, and Daniel Fried. Grounding language models to images for multimodal inputs and outputs. In International Conference on Machine Learning, pages 17283–17300. PMLR, 2023.

[26] Baiqi Li, Zhiqiu Lin, Wenxuan Peng, Jean de Dieu Nyandwi, Daniel Jiang, Zixian Ma, Simran Khanuja, Ranjay Krishna, Graham Neubig, and Deva Ramanan. Naturalbench: Evaluating vision-language models on natural adversarial samples. Advances in Neural Information Processing Systems, 37:17044–17068, 2024.

[27] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, and Chunyuan Li. LLaVA-onevision: Easy visual task transfer. Transactions on Machine Learning Research, 2025.

[28] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping languageimage pre-training with frozen image encoders and large language models. In International conference on machine learning, pages 19730–19742. PMLR, 2023.

[29] Wentong Li, Yuqian Yuan, Jian Liu, Dongqi Tang, Song Wang, Jie Qin, Jianke Zhu, and Lei Zhang. Tokenpacker: Efficient visual projector for multimodal llm. International Journal of Computer Vision, 133(10):6794–6812, 2025.

[30] Xu Li, Yi Zheng, Haotian Chen, Xiaolei Chen, Yuxuan Liang, Chenghang Lai, Bin Li, and Xiangyang Xue. Instruction-guided fusion of multi-layer visual features in large vision-language models. Pattern Recognition, 170:111932, 2026.

[31] Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Wayne Xin Zhao, and Ji-Rong Wen. Evaluating object hallucination in large vision-language models. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 292–305, 2023.

[32] Yuanzhi Li, Sébastien Bubeck, Ronen Eldan, Allie Del Giorno, Suriya Gunasekar, and Yin Tat Lee. Textbooks are all you need ii: phi-1.5 technical report. arXiv preprint arXiv:2309.05463, 2023.

[33] Chenchen Lin, Sanbao Su, Rachel Luo, Yuxiao Chen, Yan Wang, Marco Pavone, and Fei Miao. Text-guided layer fusion mitigates hallucination in multimodal llms. arXiv preprint arXiv:2601.03100, 2026.

[34] Junyan Lin, Haoran Chen, Yue Fan, Yingqi Fan, Xin Jin, Hui Su, Jinlan Fu, and Xiaoyu Shen. Multi-layer visual feature fusion in multimodal llms: Methods, analysis, and best practices. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 4156–4166, 2025.

[35] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. In Proceedings ofthe IEEE/CVF conference on computer vision andpattern recognition, pages 26296–26306, 2024.

[36] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in neural information processing systems, 36:34892–34916, 2023.

[37] Lingchen Meng, Jianwei Yang, Rui Tian, Xiyang Dai, Zuxuan Wu, Jianfeng Gao, and Yu-Gang Jiang. Deepstack: Deeply stacking visual tokens is surprisingly simple and effective for lmms. Advances in Neural Information Processing Systems, 37:23464–23487, 2024.

[38] Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy V Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel HAZIZA, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. Transactions on Machine Learning Research, 2024.

[39] Namuk Park, Wonjae Kim, Byeongho Heo, Taekyung Kim, and Sangdoo Yun. What do selfsupervised vision transformers learn? In The Eleventh International Conference on Learning Representations, 2023.

[40] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021.

[41] Noam Shazeer, \*Azalia Mirhoseini, \*Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, and Jeff Dean. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. In International Conference on Learning Representations, 2017.

[42] Min Shi, Fuxiao Liu, Shihao Wang, Shijia Liao, Subhashree Radhakrishnan, Yilin Zhao, De-An Huang, Hongxu Yin, Karan Sapra, Yaser Yacoob, Humphrey Shi, Bryan Catanzaro, Andrew Tao, Jan Kautz, Zhiding Yu, and Guilin Liu. Eagle: Exploring the design space for multimodal LLMs with mixture of encoders. In The Thirteenth International Conference on Learning Representations, 2025.

[43] Amanpreet Singh, Vivek Natarjan, Meet Shah, Yu Jiang, Xinlei Chen, Devi Parikh, and Marcus Rohrbach. Towards vqa models that can read. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 8317–8326, 2019.

[44] Adam Steel, Brenda D. Garcia, Kala Goyal, Anna Mynick, and Caroline E. Robertson. Scene perception and visuospatial memory converge at the anterior edge of visually responsive cortex. Journal ofNeuroscience, 43(31):5723–5737, 2023.

[45] Shengbang Tong, Ellis Brown, Penghao Wu, Sanghyun Woo, Manoj Middepogu, Sai C Akula, Jihan Yang, Shusheng Yang, Adithya Iyer, Xichen Pan, et al. Cambrian-1: A fully open, visioncentric exploration of multimodal llms. Advances in Neural Information Processing Systems, 37:87310–87356, 2024.

[46] Shengbang Tong, Zhuang Liu, Yuexiang Zhai, Yi Ma, Yann LeCun, and Saining Xie. Eyes wide shut? exploring the visual shortcomings of multimodal llms. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 9568– 9578, June 2024.

[47] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.

[48] Wenbin Wang, Liang Ding, Minyan Zeng, Xiabin Zhou, Li Shen, Yong Luo, Wei Yu, and Dacheng Tao. Divide, conquer and combine: A training-free framework for high-resolution image perception in multimodal large language models. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 7907–7915, 2025.

[49] Zirui Wang, Mengzhou Xia, Luxi He, Howard Chen, Yitao Liu, Richard Zhu, Kaiqu Liang, Xindi Wu, Haotian Liu, Sadhika Malladi, et al. Charxiv: Charting gaps in realistic chart understanding in multimodal llms. Advances in Neural Information Processing Systems, 37:113569– 113697, 2024.

[50] Penghao Wu and Saining Xie. V?: Guided visual search as a core mechanism in multimodal llms. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13084–13094, 2024.

[51] xAI. Realworldqa: A real-world reasoning benchmark for multimodal llms, 2024.

[52] Huanjin Yao, Wenhao Wu, Taojiannan Yang, YuXin Song, Mengxi Zhang, Haocheng Feng, Yifan Sun, Zhiheng Li, Wanli Ouyang, and Jingdong Wang. Dense connector for mllms. Advances in Neural Information Processing Systems, 37:33108–33140, 2024.

[53] Qinghao Ye, Haiyang Xu, Guohai Xu, Jiabo Ye, Ming Yan, Yiyang Zhou, Junyang Wang, Anwen Hu, Pengcheng Shi, Yaya Shi, et al. mplug-owl: Modularization empowers large language models with multimodality. arXiv preprint arXiv:2304.14178, 2023.

[54] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, et al. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pages 9556–9567, 2024.

[55] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In Proceedings of the IEEE/CVF international conference on computer vision, pages 11975–11986, 2023.

[56] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. MiniGPT-4: Enhancing vision-language understanding with advanced large language models. In The Twelfth International Conference on Learning Representations, 2024.

## Appendix

How to read this appendix. Section A states the scope, limitations, and broader impacts of the work. Sections B–C document how the models and evaluations were run. Sections D–F provide the extended empirical evidence behind the main-paper results. The links in the roadmap jump directly to each section.

<table><tr><td>Appendix Section</td><td>Contents</td></tr><tr><td>A. Limitations and Broader Impacts</td><td>Scope of claims, methodological limitations, expected benefits, risks, and responsible-use considerations.</td></tr><tr><td>B. Training Details</td><td>Hardware, two-stage training, optimization settings, objective, and router naming.</td></tr><tr><td>C. Evaluation Details</td><td>Benchmark scripts, output files, score aggregation, and missing- entry conventions.</td></tr><tr><td>mark Results</td><td>D. Additional Bench- Full benchmark matrix, k sweep, and reserve-layer ablation.</td></tr><tr><td>Analyses</td><td>E. Router and RF Layer-selection behavior, receptive-field overlays, router usage di- agnostics, and DINOv2 placeholders.</td></tr><tr><td>ples</td><td>F. Qualitative Exam- Selected attention panels plus a LLaVA + DINOv2 heatmap anal- ysis.</td></tr><tr><td>G. Notes</td><td>Reproducibility Remaining caveats, source artifacts, and checkpoint-config con- ventions.</td></tr></table>

## A Limitations and Broader Impacts

Scope of the empirical claims. MIXTURE OF LAYERS is designed to improve the visual representation interface between a pretrained vision encoder and a multimodal language model by routing over intermediate vision-layer features. The central claim is therefore not that MOL universally improves every multimodal task, but that instruction-conditioned access to task-relevant intermediate representations improves fine-grained visual reasoning while largely preserving standard VQA performance. This distinction matters: the strongest gains in our experiments appear on benchmarks that require localized evidence, such as V∗Bench, CharXiv, HRBench, OCR-style questions, direct attribute recognition, and relative-position reasoning. On more conventional or coarse-grained VQA benchmarks, the gains are smaller and sometimes close to neutral. We view this as a useful boundary on the contribution rather than a weakness of the evaluation: MOL is targeted at perception bottlenecks where the final vision layer can discard details needed by the language model.

Architectural limitations. MOL assumes access to the hidden states of a pretrained vision encoder and, for the text-conditioned variants, a text encoder aligned with that vision tower. This makes the method natural for open and reproducible MLLM training pipelines such as LLaVA- and Bunny-style models, but less directly applicable to vision encoder-free MLLM architectures [8, 9] or models that expose only final visual embeddings. The router also introduces additional trainable parameters, routing hyperparameters such as k, and a reserve-layer design choice. Our ablations show that a small number of selected layers and the reserve layer are important, but these choices may need to be retuned for substantially different vision encoders, tokenization schemes, resolutions, or downstream domains.

Evaluation limitations. The experiments cover several backbones, parameter scales, and vision encoders, including CLIP, SigLIP, and DINOv2 w/ Txt, but they do not exhaust the space of modern MLLM architectures. We do not claim that the exact best router variant transfers unchanged to every backbone: $\mathbf { M o L } _ { \mathrm { p a t c h } }$ tends to help localized perceptual categories, while $\mathbf { M o L } _ { \mathrm { h y b r i d } }$ is more robust when both local grounding and broader context are needed. We also report single-run results for most large-scale training settings because training 8B–13B multimodal models is computationally expensive. The breadth of benchmarks and ablations partially mitigates this limitation, but future work should add repeated runs, stronger uncertainty estimates, and evaluation on additional domains such as video, medical imagery, remote sensing, and document-heavy workflows.

Interpretability limitations. Router probabilities, receptive-field overlays, and selected-layer attention maps provide useful diagnostics for understanding how MOL uses intermediate representations. They should not, however, be interpreted as complete causal explanations of the model’s answer. A selected layer may contribute useful features even when its attention map is diffuse, and visually plausible attention can still coincide with an incorrect answer. We therefore use these analyses as evidence about routing behavior and locality-context trade-offs, not as a guarantee that the final language-model reasoning is faithful or safe.

Broader positive impacts. Improved fine-grained visual reasoning can benefit applications that require careful visual grounding, such as assistive image understanding, chart and figure interpretation, educational tools, accessibility support, and scientific document analysis. A practical advantage of MOL is that it can improve access to useful visual abstractions without necessarily increasing the number of visual tokens or requiring multiple vision encoders. This may offer a more computeconscious path than simply scaling image resolution, concatenating additional towers, or passing substantially longer visual sequences to the language model.

Risks and misuse. The same fine-grained perception improvements can also increase risks when applied to sensitive visual data. Better OCR, attribute recognition, and small-object grounding may make it easier to extract private information from images, analyze people or locations without consent, or strengthen surveillance-style applications. MOL also does not remove inherited biases from the underlying language model, vision encoder, or instruction-tuning data. Since the method improves perception rather than safety alignment, it should not be treated as sufficient for high-stakes deployment in domains such as medical diagnosis, legal decision-making, hiring, biometric identification, or safety-critical robotics.

<table><tr><td>Backbone</td><td>Stage</td><td>Data</td><td>ZeRO</td><td>Epochs</td><td>Global Batch</td><td>Projector LR</td><td>Router LR</td></tr><tr><td> $\mathrm { V i c u n a } { - } 1 3 \mathrm { B } + \mathrm { C L I P }$ </td><td>Pretrain</td><td>BLIP/LAION-CC-SBU 558K</td><td>2</td><td>1</td><td>256</td><td> $1 \times 1 0 ^ { - 3 }$ </td><td> $1 \times 1 0 ^ { - 5 } – 5 \times 1 0 ^ { - 5 }$ </td></tr><tr><td> $\mathrm { V i c u n a } { - } 1 3 \mathrm { B } + \mathrm { C L I P }$ </td><td>Finetune</td><td>LLaVA-v1.5 mix 665K</td><td>3</td><td>1</td><td>128</td><td> $2 \times 1 0 ^ { - 5 }$ </td><td> $2 \times 1 0 ^ { - 5 }$ </td></tr><tr><td> $\mathrm { L l a m a } { - } 3 { - } 8 \mathrm { B } + \mathrm { S i g L I P }$ </td><td>Pretrain</td><td>Bunny pretrain 2M</td><td>2</td><td>1</td><td>128</td><td> $1 \times 1 0 ^ { - 3 }$ </td><td> $1 \times 1 0 ^ { - 4 }$ </td></tr><tr><td> $\mathrm { L l a m a } { - } 3 { - } 8 \mathrm { B } + \mathrm { S i g L I P }$ </td><td>Finetune</td><td>Bunny instruction 695K</td><td>3</td><td>1</td><td>128</td><td> $2 \times 1 0 ^ { - 5 } – 2 \times 1 0 ^ { - 4 }$ </td><td> $2 \times 1 0 ^ { - 5 } – 2 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Phi  $\cdot 1 . 5 + \mathrm { S i g L I P }$ </td><td>Pretrain</td><td>Bunny pretrain 2M</td><td>2</td><td>1</td><td>256</td><td> $5 \times 1 0 ^ { - 4 }$ </td><td> $5 \times 1 0 ^ { - 4 }$ </td></tr><tr><td> $\mathrm { P h i - } 1 . 5 + \mathrm { S i g L I P }$ </td><td>Finetune</td><td>Bunny instruction 695K</td><td>3</td><td>1</td><td>128</td><td> $5 { \times } 1 0 ^ { - 5 } – 2 { \times } 1 0 ^ { - 4 }$ </td><td> $5 { \times } 1 0 ^ { - 5 } – 2 { \times } 1 0 ^ { - 4 }$ </td></tr></table>

Table 7: Training setup. The Vicuna-13B + CLIP setting follows the LLaVA-v1.5 two-stage training recipe. The Llama-3-8B + SigLIP and $\mathrm { P h i - } 1 . 5 + \mathrm { S i g L I P }$ settings follow the Bunny two-stage recipe and use LoRA during instruction tuning.

<table><tr><td>Variant</td><td>Implementation key</td><td>Routing granularity</td><td>k</td><td>Reserve layer</td><td>Reserve mode</td><td>Text conditioning</td></tr><tr><td> $\mathbf { M o L } _ { \mathrm { l a y e r } }$ </td><td>attn-token</td><td>image-level layer set</td><td>1-4</td><td> $- 2$ </td><td>local or scaled gate</td><td>attention</td></tr><tr><td> $\bf { M o L } _ { \mathrm { { p a t c h } } }$ </td><td>patch_layer</td><td>patch-specific layer set</td><td>1-4</td><td> $- 2$ </td><td>local gate</td><td>attention</td></tr><tr><td> $\mathbf { M o L } _ { \mathrm { h y b r i d } }$ </td><td>hybrid</td><td>global-local fused routing</td><td>1-4</td><td>-2</td><td>disagreement gate</td><td>attention</td></tr></table>

Table 8: Router configuration summary. The reserve layer is the penultimate vision-encoder layer used as an anchor representation. Appendix tables use normalized method names rather than implementation keys.

Responsible-use considerations. We frame MOL as a representation-routing method for research on multimodal perception, not as a complete deployment system. Systems built on top of this work should include privacy-aware data handling, domain-specific evaluation, bias and robustness audits, and human oversight in high-stakes settings. Releasing training details, normalized router names, checkpoint-configuration conventions, and analysis scripts is intended to make the empirical claims easier to audit and to help future work identify both successes and failure modes rather than treating improved benchmark scores as a blanket guarantee of reliability.

## B Training Details

All models were trained on a single server with four NVIDIA H100 NVL GPUs, each with 95830 MiB of device memory. We use DeepSpeed ZeRO-2 for the vision-language connector pretraining stage and DeepSpeed ZeRO-3 for the multimodal instruction-tuning stage. Unless otherwise stated, training uses bf16, tf32, gradient checkpointing, a cosine learning-rate schedule, warmup ratio 0.03, weight decay 0, maximum sequence length 2048, and lazy image preprocessing.

The training objective is the standard autoregressive language-modeling loss with an auxiliary router load-balancing loss:

$$
\begin{array} { r } { \mathcal { L } = \mathcal { L } _ { \mathrm { L M } } + \lambda _ { \mathrm { l b } } \mathcal { L } _ { \mathrm { l b } } . } \end{array}\tag{11}
$$

The default load-balancing coefficient is $\lambda _ { \mathrm { l b } } = 1 0 ^ { - 2 }$ . During training, the router temperature is linearly annealed from 2.0 to 1.0. We save router statistics during training for the analysis in $\mathsf { A p - }$ pendix E.

Router naming. We group the attn-token implementation variants as $\begin{array} { r } { { \bf M o L } _ { \mathrm { l a y e r } } . } \end{array}$ , and report the better-performing layer-level variant for each setting. The patch\_layer implementation corresponds to $\mathbf { M o L } _ { \mathrm { p a t c h } }$ , and the hybrid implementation corresponds to $\mathbf { M o L } _ { \mathrm { h y b r i d } }$

Table 7 summarizes the hardware and optimization recipe used for each model family. The main distinction is between the Vicuna/LLaVA training path, which follows the LLaVA-v1.5 data mixture, and the Bunny training path, which uses the Bunny pretraining and instruction-tuning data for the Llama-3 and Phi backbones. We report global batch sizes after accounting for the four GPUs and gradient accumulation.

Table 8 gives the naming bridge between implementation keys in checkpoints and the method names used throughout the paper. This is important because the experiments include multiple layer-level router implementations. We treat them as the same conceptual family, $\mathbf { M o L } _ { \mathrm { l a y e r } }$ , when summarizing broad benchmark results.

## C Evaluation Details

We evaluate with benchmark-specific scripts under scripts/v1\_5/eval. Each run writes predictions, per-example details, and a summary file under playground/data/eval/BENCHMARK/answers. Unless the benchmark defines a specialized metric, we report accuracy. For V∗Bench, HRBench4K/8K, MMStar, NaturalBench, RealWorldQA, CharXiv, and the conventional VQA benchmarks, scores are percentages. For benchmarks with multiple splits or categories, the reported number is the official aggregate used by the corresponding evaluation script.

<table><tr><td>Method</td><td>MMBench</td><td>MMMU</td><td>MMStar</td><td>MMVet</td><td>MMVP</td><td>CV-Bench</td><td>NaturalBench</td><td>CharXiv</td><td>CountBenchQA</td><td>GQA</td><td>POPE</td><td>TextVQA</td></tr><tr><td>LLaVA-v1.5-13B</td><td>76.49</td><td>36.89</td><td>30.00</td><td>41.11</td><td>65.33</td><td>62.60</td><td>68.93</td><td>25.20</td><td>50.71</td><td>63.26</td><td>85.60</td><td>61.25</td></tr><tr><td>MoI  $\begin{array} { l } { { \bf { M O L } } { \bf { l a y e r } } } \\ { { \bf { M O L } } _ { \bf { \Omega } } } \end{array}$ </td><td>75.81</td><td>36.56</td><td>29.53</td><td></td><td>64.67</td><td>57.88</td><td>69.71</td><td>27.85</td><td>48.68</td><td>63.43</td><td>86.53</td><td>60.64</td></tr><tr><td></td><td>76.40</td><td>38.22</td><td>31.93</td><td></td><td>65.00</td><td>63.28</td><td>69.41</td><td>30.89</td><td>48.68</td><td>63.73 63.75</td><td>86.23 86.47</td><td>61.09</td></tr><tr><td> $\mathbf { M o L } _ { \mathrm { h y b r i d } } ^ {  }$ </td><td>75.94</td><td>37.00</td><td>31.07</td><td>41.88</td><td>65.67</td><td>62.46</td><td>70.45</td><td>26.97</td><td>51.93</td><td></td><td></td><td>61.11</td></tr></table>

Table 9: Full benchmark matrix for the primary LLaVA-v1.5 family, part 1. $\mathbf { M o L } _ { \mathrm { l a y e r } }$ is the per-metric best of the attn-layer and attn-token layer-level implementations. $\mathbf { M o L } _ { \mathrm { p a t c h } }$ and $\mathbf { M o L } _ { \mathrm { h y b r i d } }$ are the best available scores within their normalized method families.

<table><tr><td>Method</td><td>DocVQA</td><td>MME-RW</td><td>HallusionBench</td><td>RealWorldQA</td><td>ChartQA</td><td>HRBench4K</td><td>HRBench8K</td><td>V*Bench</td><td>DTD</td></tr><tr><td>LLaVA-v1.5-13B</td><td>23.98</td><td>32.33</td><td>51.64</td><td>55.29</td><td>20.28</td><td>43.75</td><td>36.88</td><td>71.43</td><td>35.41</td></tr><tr><td> $\mathbf { M o L a y e r }$ </td><td>24.68</td><td>30.39</td><td>55.09</td><td>55.82</td><td>20.56</td><td>41.38</td><td>38.00</td><td>76.89</td><td>31.94</td></tr><tr><td>MoLpatch</td><td>25.28</td><td>33.18</td><td>52.97</td><td>55.42</td><td>21.08</td><td>43.75</td><td>39.38</td><td>84.87</td><td>=</td></tr><tr><td>MoLhybrid</td><td>25.21</td><td>32.34</td><td>53.23</td><td>56.60</td><td>22.44</td><td>44.00</td><td>38.62</td><td>84.03</td><td>1</td></tr></table>

Table 10: Full benchmark matrix for the primary LLaVA-v1.5 family, part 2. MME-RW denotes MME-RealWorld. Empty entries indicate unavailable scores in the generated benchmark table.

<table><tr><td rowspan="2">Benchmark</td><td colspan="4"> $\mathbf { M o L } _ { \mathrm { p a t c h } }$ </td><td colspan="4"> $\mathbf { M o L } _ { \mathrm { h y b r i d } }$ </td></tr><tr><td> $k = 1$ </td><td> $k = 2$ </td><td> $k = 4$ </td><td>Best</td><td> $k = 1$ </td><td> $k = 2$ </td><td> $k = 4$ </td><td>Best</td></tr><tr><td>V*Bench</td><td>76.89</td><td>73.95</td><td>79.41</td><td>79.41</td><td>84.03</td><td>76.89</td><td>67.65</td><td>84.03</td></tr><tr><td>HRBench8K</td><td>39.38</td><td>36.50</td><td>37.25</td><td>39.38</td><td>38.62</td><td>38.25</td><td>34.88</td><td>38.62</td></tr><tr><td>HRBench4K</td><td>43.12</td><td>42.75</td><td>43.75</td><td>43.75</td><td>43.88</td><td>44.00</td><td>43.50</td><td>44.00</td></tr><tr><td>MMStar</td><td>30.13</td><td>31.93</td><td>29.93</td><td>31.93</td><td>30.47</td><td>30.00</td><td>29.73</td><td>30.47</td></tr><tr><td>NaturalBench</td><td>69.41</td><td>68.07</td><td>68.61</td><td>69.41</td><td>69.96</td><td>68.91</td><td>70.45</td><td>70.45</td></tr><tr><td>CharXiv</td><td>28.65</td><td>28.47</td><td>27.94</td><td>28.65</td><td>26.13</td><td>26.25</td><td>26.97</td><td>26.97</td></tr><tr><td>RealWorldQA</td><td>55.42</td><td>54.90</td><td>55.16</td><td>55.42</td><td>55.03</td><td>56.60</td><td>56.34</td><td>56.60</td></tr></table>

Table 11: Fine-grained visual reasoning under different k values. A small selected set of layers is usually sufficient; the best k varies by task and routing family.

The broad benchmark matrix in Appendix D is generated from playground/data/eval/benchmark\_overall\_table.md. Empty entries indicate that the run was not available in the generated table. We omit columns that were entirely empty for the selected LLaVA-v1.5 family rows.

## D Additional Benchmark Results

Tables 9 and 10 expand the main-paper results to a broader set of benchmarks for the primary LLaVAv1.5 family. The table is split only for readability. These numbers are intended to show whether MOL preserves general multimodal ability while improving fine-grained visual reasoning. The strongest gains concentrate on fine-grained benchmarks such as V∗Bench, CharXiv, HRBench, and MMStar, while conventional VQA-style benchmarks usually remain close to the baseline.

Table 11 isolates the effect of the number of selected layers. The main pattern is that selecting more layers is not automatically better. A small k often gives the best result, which supports the interpretation that the router should selectively retrieve a few useful abstraction levels instead of averaging many layers indiscriminately.

Table 12 tests the reserve layer directly. Removing it hurts V∗Bench, HRBench, POPE, and MMMU, indicating that the reserve representation acts as a stable fallback when the routed intermediate layers are noisy or uncertain.

<table><tr><td>Benchmark</td><td> $\mathbf { M o L } _ { \mathrm { h y b r i d } }$ </td><td>MoLhybrid No Reserve</td></tr><tr><td>V*Bench</td><td>82.35</td><td>78.57 (-3.78)</td></tr><tr><td>HRBench8K</td><td>38.62</td><td>35.50 (-3.12)</td></tr><tr><td>HRBench4K</td><td>43.88</td><td>38.75 (-5.13)</td></tr><tr><td>POPE</td><td>86.50</td><td>82.10 (-4.40)</td></tr><tr><td>MMMU</td><td>37.00</td><td>34.00 (-3.00)</td></tr></table>

Table 12: Reserve-layer ablation. Removing the reserve layer consistently hurts performance, supporting its role as a stable anchor representation.

## E Additional Router and Receptive Field Analyses

![](images/7db44c598843b6d6527283bfc04c72c71497c95263bfd57ccd341e5a024570a7.jpg)  
Figure 4: DINOv2 layer sampling with receptive-field background on V∗Bench, MMStar, and GQA. Curves show the normalized routing probability assigned to each DINOv2 vision-encoder layer for $\bf M o L _ { \mathrm { p a t c h } }$ and $\mathbf { M o L } _ { \mathrm { h y b r i d } }$ across question categories. Background intensity denotes the category-normalized receptive field (RF), where lighter regions indicate more localized layers and darker regions indicate broader spatial context. Type-wise markers indicate the selected top-k layers.

DINOv2 RF-background Figure 4 shows that DINOv2 exhibits category-dependent routing patterns consistent with the CLIP analysis in the main paper. $\mathbf { M o L } _ { \mathrm { p a t c h } }$ often assigns high probability to early layers, especially for V∗Bench and perceptual MMStar/GQA categories, suggesting that localized DINOv2 features remain useful for fine-grained cues such as OCR, attributes, and relative position. In contrast, $\mathbf { M o L } _ { \mathrm { h y b r i d } }$ places more mass on mid-to-late layers for categories requiring broader reasoning, such as logical reasoning, math, and science/technology in MMStar. This supports the role of hybrid routing: it preserves access to local evidence while selectively relying on deeper, larger-RF layers when the query requires more semantic or relational context.

## F Additional Qualitative Examples

Qualitative Analysis on the DINOv2 Attention Maps Figure 5 shows that MOL with a DI-NOv2 encoder consistently attends to localized, query-relevant regions for fine-grained visual reasoning. For OCR-heavy examples (e.g., reading the conference name on a mug or identifying text on a banner), the selected layers concentrate attention tightly around the relevant textual regions, despite cluttered backgrounds and small font sizes. In counting and attribute-centric examples, such as identifying the jersey number of a basketball player or recognizing the brand logo on a bag, the router selects layers whose attention maps isolate the discriminative object regions while suppressing surrounding distractions. Interestingly, even in semantically broader scenes (e.g., identifying the instrument held by the ape), the selected layers maintain strong localized grounding on the queried object rather than diffusing attention across the full scene. These patterns suggest that MOL with DINOv2 learns to exploit intermediate-layer representations that preserve both fine-grained local structure and semantically meaningful object boundaries, enabling grounded visual reasoning without requiring additional visual tokens or multiple encoders.

![](images/62f64aa9faf7b7d5a942708700cca0b851a7d295fcf71eb08b2e692328aa358d.jpg)  
Figure 5: Self-attention heatmap visualization of the selected layers aggregated across all the queries of the layers for DINOv2 encoder for the selected layers.

## G Reproducibility Notes

The appendix reports the generated result artifacts available under playground/data/eval. Because the main experiments train large MLLMs, we do not report multiple random seeds for every model and benchmark. We instead prioritize breadth across backbones, vision encoders, routing variants, and benchmarks. Checkpoint config.json files are treated as the source of truth for router mode, k, reserve-layer settings, and learning-rate metadata when run-script names and final checkpoint labels differ.

For reproducibility, final paper figures used by this appendix are copied into paper\_draft\_latex/figures/appendix. The source analysis artifacts remain in playground/data/eval, including the benchmark matrix, router/RF plots, reserve comparisons, and resolution analyses.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper’s contributions and scope?

Answer: [Yes].

Justification: The abstract and introduction state the paper’s main contribution, MIXTURE OF LAYERS, and scope the claims around fine-grained visual reasoning. The reported claims are supported by the experimental results in Section 4.

Guidelines:

• The answer [N/A] means that the abstract and introduction do not include the claims made in the paper.

• The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A [No] or [N/A] answer to this question will not be perceived well by the reviewers.

• The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.

• It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes].

Justification: We discuss limitations in the Appendix, including the empirical scope of the evaluation and remaining directions for extending MIXTURE OF LAYERS to broader backbones, modalities, and settings.

Guidelines:

• The answer [N/A] means that the paper has no limitation while the answer [No] means that the paper has limitations, but those are not discussed in the paper.

• The authors are encouraged to create a separate “Limitations” section in their paper.

• The paper should point out any strong assumptions and how robust the results are to violations of these assumptions.

• The authors should reflect on the scope of the claims made.

• The authors should reflect on the factors that influence the performance of the approach.

• The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.

• If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.

• Reviewers will be specifically instructed to not penalize honesty concerning limitations.

## 3. Theory assumptions and proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

Answer: [N/A].

Justification: This paper does not present theoretical results or formal proofs.

Guidelines:

• The answer [N/A] means that the paper does not include theoretical results.

• All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.

• All assumptions should be clearly stated or referenced in the statement of any theorems.

• The proofs can either appear in the main paper or the supplemental material.

• Any informal proof provided in the core of the paper should be complemented by formal proofs.

• Theorems and Lemmas that the proof relies upon should be properly referenced.

## 4. Experimental result reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper?

Answer: [Yes].

Justification: We describe the model architecture, routing variants, training setup, datasets, and evaluation benchmarks in the main paper and Appendix. We will release code and instructions to reproduce the main experimental results.

Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• If the paper includes experiments, a [No] answer to this question will not be perceived well by the reviewers.

• If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.

• Reproducibility can be provided via code, data, instructions, model checkpoints, hosted models, or other appropriate means.

• NeurIPS requires submissions to provide some reasonable avenue for reproducibility.

## 5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?

Answer: [Yes].

Justification: All datasets used in the paper are publicly available, and we will release the codebase with instructions for training and evaluation. At submission time, any released materials will be anonymized as required.

Guidelines:

• The answer [N/A] means that paper does not include experiments requiring code.

• Please see the NeurIPS code and data submission guidelines for more details.

• While code and data release is encouraged, [No] is acceptable with justification.

• The instructions should contain the exact command and environment needed to reproduce the results.

• The authors should provide instructions on data access and preparation.

• The authors should provide scripts to reproduce all experimental results.

• At submission time, released materials should be anonymized if applicable.

## 6. Experimental setting/details

Question: Does the paper specify all the training and test details necessary to understand the results?

Answer: [Yes].

Justification: The main paper describes the backbone models, vision encoders, datasets, benchmarks, and routing variants. Additional training details, hyperparameters, and evaluation settings are provided in the Appendix.

Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• The experimental setting should be presented in the core of the paper to a level of detail necessary to appreciate the results.

• The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [No].

Justification: We do not report statistical significance tests or error bars. The experiments are computationally expensive, and our comparisons follow standard benchmark evaluation protocols for MLLMs.

## Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• The authors should answer [Yes] if the results are accompanied by error bars, confidence intervals, or statistical significance tests.

• The factors of variability captured by the error bars should be clearly stated.

• The method for calculating the error bars should be explained.

• The assumptions made should be given.

• It should be clear whether the error bar is standard deviation or standard error.

• For asymmetric distributions, authors should avoid inappropriate symmetric error bars.

• If error bars are reported, the text should explain how they were calculated.

## 8. Experiments compute resources

Question: For each experiment, does the paper provide sufficient information on the computer resources needed to reproduce the experiments?

Answer: [Yes].

Justification: We provide compute resource details, including GPU type, memory, and training/evaluation cost, in the Appendix.

Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• The paper should indicate the type of compute workers, including relevant memory and storage.

• The paper should provide the amount of compute required for individual runs and estimate total compute.

• The paper should disclose whether the full research project required more compute than the experiments reported.

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics?

Answer: [Yes].

Justification: The research uses publicly available datasets and models, does not involve human subjects, and does not require deviations from the NeurIPS Code of Ethics.

Guidelines:

• The answer [N/A] means that the authors have not reviewed the NeurIPS Code of Ethics.

• If the authors answer [No], they should explain the special circumstances that require a deviation.

• The authors should make sure to preserve anonymity.

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [N/A].

Justification: This work is foundational research on visual representation routing in MLLMs and is not tied to a specific deployment or application domain. We do not identify a direct societal impact beyond the general impacts associated with improving multimodal model capabilities.

## Guidelines:

• The answer [N/A] means that there is no societal impact of the work performed.

• If the authors answer [N/A] or [No], they should explain why.

• Examples of negative societal impacts include malicious uses, fairness considerations, privacy considerations, and security considerations.

• Foundational research may not need application-specific broader impact discussion unless there is a direct path to harm.

• Authors should consider harms from intended use, incorrect results, and misuse.

• If there are negative societal impacts, authors may discuss mitigation strategies.

## 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse?

Answer: [N/A].

Justification: The paper does not release a new high-risk dataset or pretrained generative model. The released code is intended to reproduce the proposed routing method and experiments using existing public assets.

Guidelines:

• The answer [N/A] means that the paper poses no such risks.

• Released models with high misuse risk should be released with necessary safeguards.

• Datasets scraped from the Internet could pose safety risks.

• Authors are encouraged to make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets used in the paper properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes].

Justification: We cite the original papers for all datasets, models, and codebases used in this work. We use publicly available assets and will document their licenses and terms of use in the Appendix or released codebase.

Guidelines:

• The answer [N/A] means that the paper does not use existing assets.

• The authors should cite the original paper that produced the code package or dataset.

• The authors should state which version of the asset is used and, if possible, include a URL.

• The name of the license should be included for each asset.

• For scraped data, copyright and terms of service should be provided.

• If assets are released, license, copyright information, and terms of use should be provided.

• For repackaged datasets, both original and derived licenses should be provided.

• If license information is unavailable, authors are encouraged to contact asset creators.

## 13. New assets

Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?

Answer: [Yes].

Justification: We introduce and release the MIXTURE OF LAYERS codebase. The released asset will include documentation, setup instructions, training and evaluation scripts, and license information.

Guidelines:

• The answer [N/A] means that the paper does not release new assets.

• Researchers should communicate details of the dataset/code/model through documentation.

• The paper should discuss whether and how consent was obtained from people whose asset is used.

• At submission time, assets should be anonymized if applicable.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation?

Answer: [N/A].

Justification: This paper does not involve crowdsourcing or research with human subjects. Guidelines:

• The answer [N/A] means that the paper does not involve crowdsourcing nor research with human subjects.

• Including this information in supplemental material is fine.

• According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least minimum wage.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether IRB approvals were obtained?

Answer: [N/A].

Justification: This paper does not involve crowdsourcing or research with human subjects, so IRB approval is not applicable.

Guidelines:

• The answer [N/A] means that the paper does not involve crowdsourcing nor research with human subjects.

• IRB approval or equivalent may be required depending on the country or institution.

• Authors should adhere to the NeurIPS Code of Ethics and institutional guidelines.

• For initial submissions, do not include information that would break anonymity.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research?

Answer: [N/A].

Justification: The paper studies MLLM architectures, but LLMs are not used as a nonstandard research assistant or data-generation component in the core methodology. Any use of LLMs for writing, editing, or formatting does not affect the scientific contribution.

Guidelines:

• The answer [N/A] means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.

• Please refer to the LLM policy in the NeurIPS handbook for what should or should not be described.