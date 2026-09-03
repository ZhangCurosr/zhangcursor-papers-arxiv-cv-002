# RVSD: Retrieval Vision Sparse Decoding for Mitigating Visual Hallucinations in Large Vision-Language Models

Canjie Liu<sup>1</sup>, Jiawen Kang<sup>†1</sup>, Jinbo Wen<sup>2</sup>, Zishao Zhong<sup>3</sup>

<sup>1</sup> School of Automation, Guangdong University of Technology, Guangzhou, China <sup>2</sup> Department of Computer Science, City University of Hong Kong, Hong Kong, China <sup>3</sup> The Second Affiliated Hospital of Guangzhou University of Chinese Medicine, Guangzhou, China 2112404359@mail2.gdut.edu.cn, kavinkang@gdut.edu.cn jinbowen2-c@my.cityu.edu.hk, zhongzishao@gzucm.edu.cn

## Abstract

Large vision-language models have achieved remarkable success in vision-language tasks. However, they remain prone to Visual Hallucinations (VHs), undermining their reliability in real-world applications. Existing solutions typically require curated datasets, additional training, or multi-round decoding, resulting in considerable computational overhead. In this paper, we propose RVSD (Retrieval Vision Sparse Decoding), a training-free and plugand-play decoding framework that, for the first time, unifies token sparsification and Semantic-Space Visual Retrieval (SSVR) within a single decoding pass. Within RVSD, we introduce a semantics-directed token selection strategy that selectively sparsifies redundant tokens while preserving critical visual information. We further propose the SSVR mechanism, which reformulates visual compensation as an on-demand cross-modal retrieval process within a shared semantic space. Extensive experiments demonstrate that RVSD achieves state-of-the-art performance in mitigating VHs while maintaining robust suppression capabilities under long-context generation settings. Our code is available here.<sup>1</sup>

## 1 Introduction

Large Vision-Language Models (LVLMs) have recently demonstrated impressive capabilities across a wide range of vision-language tasks (Liu et al., 2024b; Bai et al., 2023; Liu et al., 2024c), such as image captioning, visual question answering, and open-ended visual reasoning. Despite their rapid progress, LVLMs remain highly susceptible to Visual Hallucinations (VHs) (Bai et al., 2024; Liu et al., 2024a), where generated responses contradict or lack grounding in the actual visual content. This fundamental flaw seriously compromises the reliability of LVLMs in real-world applications (Wang et al., 2023b), where misinterpretation of visual data can lead to significant consequences.

![](images/5c422cc35fa97bb305b9007a90f50a58c00b9140a3a009e17dd3ed7367e21044.jpg)  
Figure 1: Comparison of representative paradigms for mitigating VHs in LVLMs: (a) training-based alignment and RAG (Sun et al., 2024), (b) decoding interventions (Leng et al., 2024; Favero et al., 2024), (c) sparse decoders (Chen et al., 2024; Zhang et al., 2025), and (d) our RVSD, which turns pruned tokens into a retrievable memory bank and restores grounding on demand in a single forward pass.

Existing solutions can be categorized into four paradigms, which we compare with RVSD in Fig. 1. Training-based alignment (Sun et al., 2024) mitigates VHs via additional supervision and preference alignment, but requires expensive retraining and annotated data. Retrieval-augmented generation (Lewis et al., 2020) grounds generation using external knowledge bases, yet introduces substantial storage and retrieval overhead. Decoding-side interventions (Liu et al., 2025) adjust token distributions during inference, but often require iterative decoding procedures that incur considerable latency. More recently, sparsification-based decoding (Dang et al., 2026) has emerged as a promising direction that simultaneously accelerates inference and suppresses distracting visual tokens, offering an attractive trade-off between efficiency and VH mitigation for LVLMs.

![](images/54a0da691f145bf46b177fafa0124f0f9047e9aa6d60d2ebe66ffd56b62e7150.jpg)  
Figure 2: Sparsification-hallucination paradox, exemplified by the representative sparse decoder VASparse (Zhuang et al., 2025) on LLaVA-1.5 using the Caption Hallucination Assessment with Image Relevance (CHAIR) dataset.

However, existing sparse decoding methods remain largely modality-agnostic, often pruning visual tokens that appear statistically redundant yet contain semantically critical information (Zhuang et al., 2025). Once such fine-grained visual tokens are discarded, the resulting information loss becomes irreversible. Our empirical analysis further reveals that the hallucination rates of State-of-the-Art (SOTA) sparse decoders (Zhuang et al., 2025) increase substantially as generation length grows, as shown in Fig. 2. We refer to this phenomenon as the sparsification-hallucination paradox, highlighting the fundamental tension between inference efficiency and reliable visual grounding in LVLMs.

In this paper, we propose RVSD (Retrieval Vision Sparse Decoding), a training-free and plugand-play decoding framework for mitigating VHs in LVLMs. RVSD addresses the sparsificationhallucination paradox by combining semanticsdirected token sparsification with on-demand visual compensation in a unified single-pass inference process. Our contributions are as follows:

• We propose RVSD, a novel training-free and plug-and-play decoding framework that addresses the sparsification-hallucination paradox. RVSD jointly integrates token sparsification and dynamic visual compensation within a single inference pass, achieving an effective balance between inference efficiency and visual faithfulness.

• We develop a semantics-directed token selection strategy that selectively distinguishes redundant tokens from visually critical ones. Unlike prior modality-agnostic pruning methods, the proposed strategy performs targeted sparsification while preserving essential visual grounding information.

• We propose a Semantic-Space Visual Retrieval (SSVR) mechanism that performs ondemand cross-modal retrieval in a shared semantic space. When visual grounding becomes insufficient during generation, SSVR dynamically retrieves relevant visual evidence from a deferred visual memory bank to recalibrate hallucinated output distributions.

• Extensive experiments across multiple prominent benchmarks demonstrate that RVSD achieves SOTA performance, simultaneously improving both generation quality and decoding efficiency while exhibiting strong robustness under long-context generation settings.

## 2 Related Work

Large Vision-Language Models. Building upon the rapid advances in large language models (Zhao et al., 2026), recent research has extended them to the multimodal domain, giving rise to LVLMs (Liu et al., 2023, 2024b,c). LVLMs are typically trained via large-scale vision-language pretraining followed by instruction-based fine-tuning, while subsequent studies further align generation behavior with human preferences via Reinforcement Learning from Human Feedback (RLHF) (Sun et al., 2024) and preference optimization techniques. Despite their strong multimodal capabilities, LVLMs remain prone to VHs (Bai et al., 2024; Yin et al., 2024; Wu et al., 2023), severely undermining the reliability of LVLMs in real-world applications.

Visual Hallucination Mitigation. Existing efforts mitigate VHs from several perspectives. Trainingcentric approaches (Sun et al., 2024) improve multimodal alignment through additional supervision, RLHF, or preference optimization techniques. Decoding-time methods intervene during inference, where contrastive strategies, such as VCD (Leng et al., 2024) and M3ID (Favero et al., 2024), suppress linguistic priors via perturbed visual inputs, while attention-guided methods, including VTI (Liu et al., 2025), AvisC (Woo et al., 2025), and SHARP (Wu et al., 2025), enhance attention toward visually relevant regions. Adaptive visual reinforcement methods (Zhu et al., 2026) further strengthen visual evidence on demand during generation. Closest to our setting, RVCD (Lee and Song, 2025) performs retrieval-based contrastive decoding by referencing an offline database of generated single-concept images at the logit level. In contrast, RVSD requires no external image store, detector, or draft decoding pass, and retrieves the model’s own deferred visual tokens directly in the semantic space of a single forward pass. Despite their different formulations, these methods still rely on dense visual processing throughout decoding, leaving substantial room for improving efficiencyaware visual grounding.

![](images/fa063493b51b275f04b06836ab47d41a8e36b1162d29ec0ade73e01a3e1eb007.jpg)  
Figure 3: Overview of the proposed RVSD framework, which consists of two stages. Stage 1 performs semantics-directed token selection, partitioning visual tokens into an active sparse set and a deferred memory bank through cross-modal attention. Stage 2 applies the SSVR mechanism to retrieve on-demand visual evidence from the deferred bank and re-anchor decoding within a single forward pass, transforming hallucinated content from the original and sparse decoders into faithful generation.

Sparse Decoding for Efficient Inference. Token sparsification has been utilized to improve both inference efficiency and visual faithfulness (Wu et al., 2026). General-purpose accelerators, such as SparseVLM (Zhang et al., 2025) and FastV (Chen et al., 2024), prune visual tokens using modality-agnostic criteria primarily designed for acceleration. Recent methods, such as VASparse (Zhuang et al., 2025), LTS-FS (Dang et al., 2026), PruneHal (Sun et al., 2025), MINT (Wang et al., 2025), and EAZY (Che et al., 2025), further explore hallucination-aware sparsification by enabling LVLMs to focus on task-relevant visual tokens. Nevertheless, existing methods still perform irreversible pruning, which may remove fine-grained visual evidence. Unlike prior methods, RVSD reorganizes discarded tokens into a deferred visual memory bank, enabling ondemand restoration of visual grounding within the same decoding pass without additional training or iterative decoding.

## 3 Problem Formulation

Given an LVLM $\mathcal { M } _ { \theta } ^ { \mathrm { M L L M } }$ parameterized by θ, the model consists of a text embedding layer, a vision encoder, a vision-language interface module, a decoder with L transformer layers, and an affine output layer $\varsigma ( \cdot )$ for next-token prediction. For an image-grounded generation task with textual query x and input image v, the vision encoder first extracts visual features from v, which are then transformed into aligned visual tokens $z _ { v }$ via a Multi-Layer Perceptron (MLP) or Q-Former interface. The visual tokens $z _ { v }$ are concatenated with the query x and fed into the decoder to autoregressively generate the response $y ,$ given by

$$
y _ { t } \sim p _ { \theta } ( \cdot | v , x , y _ { < t } ) \propto \mathrm { s o f t m a x } ( f _ { \theta } ( \cdot | v , x , y _ { < t } ) ) ,\tag{1}
$$

where $y _ { t }$ denotes the token generated at step $t , y _ { < t }$ denotes the previously generated prefix, and $f _ { \theta }$ represents the unnormalized logits from ${ \mathcal { M } } _ { \theta } ^ { \mathrm { M L L M } }$

When the generated response $y$ contradicts or is unsupported by the input image $v ,$ the LVLM is considered to exhibit VHs. Accordingly, the objective of VH mitigation is to suppress unfaithful generations while preserving overall response quality. In the following, we present RVSD, which improves visual faithfulness during inference with minimal computational overhead.

## 4 Methodology

As illustrated in Fig. 3, we present the architecture of the proposed RVSD. In the following, we describe its single-pass inference process and two key components: semantics-directed token selection and the SSVR mechanism.

## 4.1 Single Forward Pass

Let the visual token bank be denoted by $Z _ { v }$ $[ z _ { 1 } , \dots , z _ { N _ { v } } ] \ \in \ \mathbb { R } ^ { N _ { v } \times D }$ , where $N _ { v }$ is the total number of visual tokens and D is the hidden dimension. The decoder hidden states at decoding step t in layer l are denoted by $H _ { t } ^ { ( l ) } = [ h _ { 1 } ^ { ( l ) } , \ldots , h _ { t } ^ { ( l ) } ]$ in which each $h _ { i } ^ { ( l ) } \in \mathbb { R } ^ { D }$ denotes the hidden state at textual position i. Thus, the text-to-vision attention weight $A _ { i , j } ^ { ( l ) }$ from the i-th text token to the j-th visual token at layer l is defined as

$$
A _ { i , j } ^ { ( l ) } = \frac { \exp { \left( ( W _ { Q } ^ { ( l ) } h _ { i } ^ { ( l ) } ) ^ { \top } ( W _ { K } ^ { ( l ) } z _ { j } ) / \sqrt { D } \right) } } { \sum _ { r = 1 } ^ { N _ { v } } \exp { \left( ( W _ { Q } ^ { ( l ) } h _ { i } ^ { ( l ) } ) ^ { \top } ( W _ { K } ^ { ( l ) } z _ { r } ) / \sqrt { D } \right) } } ,\tag{2}
$$

where $W _ { Q } ^ { ( l ) } , W _ { K } ^ { ( l ) }$ denote the query and key projection matrices at layer l, respectively.

At each decoding step, RVSD maintains two complementary visual token sets: an active sparse set $S _ { t }$ for standard decoding and a deferred set $\mathcal { P } _ { t }$ that forms a deferred visual memory bank, where $\mathcal { S } _ { t } \cup \mathcal { P } _ { t } = 1 , \ldots , N _ { v }$ and $S _ { t } \cap \mathcal { P } _ { t } = \mathcal { O }$ . Decoding is performed over $S _ { t }$ , while $\mathcal { P } _ { t }$ remains semantically retrievable and provides relevant visual evidence on demand. Based on Eq. (1), RVSD autoregressively predicts the token at generation step t as

$$
p _ { \theta } ^ { \mathrm { R V S D } } = \operatorname { s o f t m a x } \left( f _ { \theta } \big ( \cdot | Z _ { v } [ S _ { t } ] , x , y _ { < t } ; \Delta _ { t } \big ) \right) ,\tag{3}
$$

where $\Delta _ { t }$ denotes an optional SSVR branch activated only when visual grounding becomes insufficient. This design preserves decoding efficiency under normal generation while enabling querydriven visual retrieval during uncertain steps, without requiring multi-round decoding. Algorithm 1 presents the overall single forward pass of RVSD. The dynamic sparsity module for creating $S _ { t }$ and $\mathcal { P } _ { t }$ and the SSVR module realizing $\Delta _ { t }$ are detailed in Sections 4.2 and 4.3, respectively.

## 4.2 Dynamic Sparsity

The objective of dynamic sparsity is to retain visual tokens most relevant to the salient textual semantics in the current generation trajectory. Rather than performing modality-agnostic pruning, RVSD explicitly scores visual tokens by aggregating crossmodal attention from salient textual positions. At decoding step t, we first compute textual saliency over generated tokens $\{ 1 , \ldots , t - 1 \}$ as

Algorithm 1 Single Forward Pass of RVSD   
Require: An LVLM M<sub>θ</sub>, an image v, and a query x   
1: Encode visual tokens: $Z _ { v }  \breve { \mathcal { M } } _ { \theta } ^ { \mathrm { e n c } } ( v )$ ; initialize $\mathbf { \tilde { s } ^ { 0 }  0 }$   
2: for $t = 1 , 2 , \ldots$ do   
3: Select token subset $\mathcal { T } _ { t }$ via $\beta ^ { t }$ using Eq. (4)   
4: Update smoothed saliency $\mathit { \Pi } _ { \widetilde { s } ^ { t } }$ from $s ^ { t }$ using Eq. (6)   
5: Partition visual tokens into $S _ { t }$ and $\mathcal { P } _ { t }$ using Eq. (7)   
6: trigger TRUE   
7: for $\bar { l } = l _ { \mathrm { s } }$ to $l _ { \mathrm { e } }$ do   
8: Compute uncertainty score $u _ { t } ^ { ( l ) }$ using Eq. (8)   
9: if trigger and $u _ { t } ^ { ( l ) } > \gamma$ then   
10: Retrieve deferred tokens from $Z _ { v } [ \mathcal { P } _ { t } ]$ via   
query $q _ { t } ^ { ( l ) }$ using Eq. (9)   
11: Aggregate retrieved features $\tilde { Z } _ { v }$ from top-k<sub>c</sub>   
candidates using Eq. (10)   
12: Construct adaptive adapter weights $W _ { 1 , 2 } ^ { \mathrm { a d p t } }$ us  
ing Eq. (11)   
13: Fuse retrieved visual evidence using Eq. (13)   
14: trigger ← FALSE   
15: end if   
16: end for   
17: Predict token y<sub>t</sub> over $Z _ { v } [ S _ { t } ]$ using Eq. (3)   
18: end for   
19: return Response sequence y

$$
\beta _ { i } ^ { t } = \frac { \exp ( \| h _ { i } ^ { ( L ) } \| _ { 2 } / \tau ) } { \sum _ { r = 1 } ^ { t - 1 } \exp ( \| h _ { r } ^ { ( L ) } \| _ { 2 } / \tau ) } ,\tag{4}
$$

where $\| \cdot \| _ { 2 }$ denotes the $\ell _ { 2 }$ norm, $h _ { i } ^ { ( L ) }$ is the lastlayer hidden state of the i-th token, and τ is a temperature parameter. We then select the major textual token set $\mathcal { T } _ { t } = \mathrm { T o p K } ( \beta ^ { t } , m )$ , where m is the number of major textual tokens retained. Given a layer subset $\mathcal { L } _ { s }$ for stable attention aggregation, we define the visual relevance score $\boldsymbol { s } _ { j } ^ { t }$ as

$$
s _ { j } ^ { t } = \sum _ { i \in \mathcal { T } _ { t } } \beta _ { i } ^ { t } \Bigl ( \frac { 1 } { | \mathscr { L } _ { s } | } \sum _ { l \in \mathscr { L } _ { s } } A _ { i , j } ^ { ( l ) } \Bigr ) .\tag{5}
$$

To suppress step-wise noise, we apply temporal smoothing, which is expressed as

$$
\tilde { s } ^ { t } = \eta \tilde { s } ^ { t - 1 } + ( 1 - \eta ) s ^ { t } ,\tag{6}
$$

where $\eta \in [ 0 , 1 )$ is the smoothing coefficient, and $\tilde { s } ^ { t }$ represents the smoothed relevance vector at step t. We then construct the active sparse set as

$$
\begin{array} { r } { S _ { t } = \mathrm { T o p K } ( \tilde { s } ^ { t } , k ) , \mathcal { P } _ { t } = \{ 1 , . . . , N _ { v } \} \setminus { S } _ { t } , } \end{array}\tag{7}
$$

where k is the budget that controls the size of the retained sparse set. Therefore, RVSD retains the visual tokens most relevant to the current decoding context while deferring the remaining tokens for potential later retrieval. With $k \ll N _ { v }$ , the perlayer attention complexity is reduced from $\mathcal { O } ( ( t +$ $\dot { N _ { v } } ) ^ { 2 } )$ to ${ \mathcal { O } } ( ( t + k ) ^ { 2 } )$ .

## 4.3 Semantic-Space Visual Retrieval

Dynamic sparsity alone may still discard finegrained visual evidence required at later decoding stages. To address this issue, RVSD reformulates visual compensation as on-demand cross-modal retrieval in a shared semantic space. Instead of directly restoring pruned features, RVSD treats the deferred set $\mathcal { P } _ { t }$ as a deferred visual memory bank and uses the current decoder state as a semantic query to retrieve relevant visual evidence whenever grounding becomes insufficient.

## 4.3.1 Uncertainty-triggered retrieval gate

At decoding step t, we scan intermediate layers and compute the normalized predictive uncertainty as

$$
u _ { t } ^ { ( l ) } = - \frac { 1 } { \log N } \sum _ { n = 1 } ^ { N } p _ { \theta , n } ^ { ( l ) } \log p _ { \theta , n } ^ { ( l ) } ,\tag{8}
$$

where $N$ is the vocabulary size, and $p _ { \theta , n } ^ { ( l ) }$ represents the predicted probability of the n-th vocabulary token derived from the layer-l hidden state. When $u _ { t } ^ { ( l ) } > \gamma$ for $l \in [ l _ { \mathrm { s } } , l _ { \mathrm { e } } ]$ , where $\gamma$ is a predefined entropy threshold and $[ l _ { \mathrm { s } } , l _ { \mathrm { e } } ]$ denotes the intermediatelayer scanning window, the model triggers a single retrieval call at this step, indicating that the current generation trajectory has insufficient visual grounding and requires re-anchoring.

## 4.3.2 Cross-modal retrieval

Once triggered, we project the current decoder hidden state into the shared visual-text semantic space and use it as a query $q _ { t } ^ { ( l ) } = h _ { t } ^ { ( l ) } [ - 1 , : ]$ , i.e., the hidden state at the last token position of layer l, to query the deferred visual memory bank:

$$
r _ { j } = \frac { q _ { t } ^ { ( l ) } z _ { j } ^ { \top } } { \sqrt { D } } , j \in { \mathcal P } _ { t } , w = \mathrm { s o f t m a x } ( r ) ,\tag{9}
$$

where r measures the semantic relevance between $r _ { j }$ the textual query and each deferred visual token, and w defines a normalized retrieval distribution over $\mathcal { P } _ { t }$ . With $k _ { c } = \mathrm { m i n } ( K , | \mathcal { P } _ { t } | )$ , where K is a predefined upper bound, we select the $\mathrm { t o p } \mathrm { - } k _ { c }$ tokens $( \tilde { w } , \mathcal { T } _ { t } ) = \mathrm { T o p K } ( w , k _ { c } )$ , where w˜ denotes the corresponding weights, and $\mathcal { T } _ { t } \subseteq \mathcal { P } _ { t }$ denotes their indices in the deferred memory bank. The weights are renormalized and aggregated into a query-conditioned visual representation, which is expressed as

$$
\begin{array} { r } { \tilde { Z } _ { v } = \mathrm { d i a g } ( \tilde { w } ) Z _ { v } [ \mathcal { T } _ { t } ] , } \end{array}\tag{10}
$$

where diag( ˜w) denotes a diagonal matrix whose main diagonal is formed by w˜.

## 4.3.3 Lightweight injection

To inject the retrieved evidence into the decoding stream without disturbing the model’s intrinsic distribution, we introduce a transient adapter whose weights are scale-matched to the host Feed-Forward Network (FFN), expressed as

$$
W _ { 1 } ^ { \mathrm { a d p t } } = \frac { s _ { \mathrm { u p } } } { \tilde { s } _ { v } } \tilde { Z } _ { v } , \ W _ { 2 } ^ { \mathrm { a d p t } } = \frac { s _ { \mathrm { d o w n } } } { \tilde { s } _ { v } } \tilde { Z } _ { v } ^ { \top } .\tag{11}
$$

Here, $W _ { 1 } ^ { \mathrm { a d p t } }$ and $W _ { 2 } ^ { \mathrm { a d p t } }$ denote the up- and downprojection weights of the transient adapter, respectively. $W _ { \mathrm { u p } }$ and $W _ { \mathrm { d o w n } }$ denote the corresponding up- and down-projection matrices of the host FFN, respectively. $s _ { \mathrm { u p } } =$ mean $( | W _ { \mathrm { u p } } | )$ $s _ { \mathrm { d o w n } } =$ mean $\left( | W _ { \mathrm { d o w n } } | \right)$ , and $\tilde { s } _ { v } ~ = ~ \mathrm { m e a n } ( | \tilde { Z } _ { v } | )$ denote the mean absolute magnitudes used for amplitude alignment. The adapter performs a low-rank projection that re-grounds the textual representation using the retrieved visual evidence, expressed as

$$
\begin{array} { r } { \mathcal { G } ( x ) = x ( W _ { 1 } ^ { \mathrm { a d p t } } ) ^ { \top } ( W _ { 2 } ^ { \mathrm { a d p t } } ) ^ { \top } = \frac { s _ { \mathrm { u p } } s _ { \mathrm { d o w n } } x \tilde { Z } _ { v } ^ { \top } \tilde { Z } _ { v } } { \tilde { s } _ { v } ^ { 2 } } , } \end{array}\tag{12}
$$

where x denotes the input residual-stream hidden state fed into the adapter. After amplitude normalization, the retrieved visual signal aˆ is fused with the block-output residual stream $h _ { t } ^ { ( l + 1 ) }$ , i.e., the hidden state after FFN residual addition in layer l that feeds into the attention sublayer of layer l+1:

$$
o = ( 1 - \alpha ) \cdot h _ { t } ^ { ( l + 1 ) } + \alpha \cdot \hat { a } ,\tag{13}
$$

where o denotes the fused hidden state passed to layer l+1, and $\alpha \in [ 0 , 1 ]$ denotes the injection ratio that balances the original residual stream and the retrieved visual evidence. After retrieval, the trigger is reset, and the transient adapter is released. This enables parameter-free and on-demand visual retrieval in the semantic space, i.e., visual evidence is retrieved only when grounding fails, and the overhead is removed after each step, ensuring efficiency while mitigating VHs.

## 5 Experiments

## 5.1 Experimental Settings

Benchmarks. We evaluate RVSD on five complementary benchmarks. For discriminative hallucination, we use POPE (Li et al., 2023) with random, popular, and adversarial splits, the MME hallucination subset (Fu et al., 2026) covering object-level and attribute-level probes, and the discrimination subset of AMBER (Wang et al., 2023a), reporting accuracy and F1 scores. For generative hallucination, we evaluate on CHAIR (Rohrbach et al., 2018) over 500 MS-COCO images and the AM-BER generative subset, and also report AMBER discrimination accuracy and F1 scores for a unified view of caption-level performance. To assess whether VH mitigation affects general capability, we further evaluate on MM-Vet (Yu et al., 2024), which covers six vision-language abilities and uses GPT-4 as the evaluator.

<table><tr><td rowspan=1 colspan=13>Random            Popular          Adversarial           AverageModels       MethodsAcc (%)↑ F1 (%)↑ Acc (%)↑ F1 (%)↑ Acc (%)↑ F1 (%)↑ Acc (%)↑ F1 (%)↑</td></tr><tr><td rowspan=1 colspan=13>Vanilla                   83.8 ↑0.084.2 ↑0.082.0 ↑0.082.6 ↑0.075.8 ↑0.078.1 ↑0.080.5 ↑0.081.6 ↑0.0</td></tr><tr><td rowspan=3 colspan=3>+VCD (Leng et al., 2024)   85.0 ↑1.2+M3ID (Favero et al., 2024)LLaVA-1.5+VTI (Liu et al., 2025)      83.0 ↓0.8</td><td rowspan=1 colspan=1>84.2 ↑0.0</td><td rowspan=1 colspan=1>82.1 ↑0.1</td><td rowspan=1 colspan=2>83.2 ↑0.6</td><td rowspan=1 colspan=2>76.3 ↑0.5</td><td rowspan=1 colspan=1>78.7 ↑0.6</td><td rowspan=1 colspan=1>81.1 ↑</td><td rowspan=1 colspan=1>0.6</td><td rowspan=1 colspan=1>82.0 ↑0.4</td></tr><tr><td rowspan=1 colspan=2>86.1 ↑2.3</td><td rowspan=1 colspan=1>85.0 ↑0.8</td><td rowspan=1 colspan=1>82.8 ↑0.8</td><td rowspan=1 colspan=1>84.1 ↑</td><td rowspan=1 colspan=1>1.5</td><td rowspan=1 colspan=2>77.1 ↑1.3</td><td rowspan=1 colspan=1>78.9 ↑0.8</td><td rowspan=1 colspan=1>82.0 ↑</td><td rowspan=1 colspan=1>1.5</td><td rowspan=1 colspan=1>82.7 ↑1.1</td></tr><tr><td rowspan=1 colspan=2>83.0 ↓0.8</td><td rowspan=1 colspan=1>83.6↓0.6</td><td rowspan=1 colspan=1>80.4↓1.6</td><td rowspan=1 colspan=1>81.8 ↓</td><td rowspan=1 colspan=1>0.8</td><td rowspan=1 colspan=2>76.0 ↑0.2</td><td rowspan=1 colspan=1>78.8 ↑0.7</td><td rowspan=1 colspan=1>79.8 ↓</td><td rowspan=1 colspan=1>0.7</td><td rowspan=1 colspan=1>81.4 ↓0.2</td></tr><tr><td rowspan=1 colspan=3>+AvisC (Woo et al., 2025)   82.3 ↓1.5</td><td rowspan=1 colspan=1>83.5 ↓0.7</td><td rowspan=1 colspan=1>78.2 ↓3.8</td><td rowspan=1 colspan=1>80.5 ↓</td><td rowspan=1 colspan=1>2.1</td><td rowspan=1 colspan=2>74.2 ↓1.6</td><td rowspan=1 colspan=1>77.7 ↓0.4</td><td rowspan=1 colspan=1>78.2 ↓</td><td rowspan=1 colspan=1>2.3</td><td rowspan=1 colspan=1>80.6↓1.0</td></tr><tr><td rowspan=1 colspan=3>+RVSD (Ours)            87.3 ↑3.5</td><td rowspan=1 colspan=1>86.3 ↑2.1</td><td rowspan=1 colspan=1>86.6 ↑4.6</td><td rowspan=1 colspan=2>85.3 ↑2.7</td><td rowspan=1 colspan=2>84.7 ↑8.9</td><td rowspan=1 colspan=1>83.5 ↑5.4</td><td rowspan=1 colspan=1>86.2 ↑</td><td rowspan=1 colspan=1>5.7</td><td rowspan=1 colspan=1>85.0 ↑3.4</td></tr><tr><td rowspan=1 colspan=3>Vanilla                   84.4 ↑0.0</td><td rowspan=1 colspan=1>82.3 ↑0.0</td><td rowspan=1 colspan=1>83.2 ↑0.0</td><td rowspan=1 colspan=1>81.5 ↑</td><td rowspan=1 colspan=1>0.0</td><td rowspan=1 colspan=2>79.5 ↑0.0</td><td rowspan=1 colspan=1>78.1 ↑0.0</td><td rowspan=1 colspan=1>82.4 ↑</td><td rowspan=1 colspan=1>0.0</td><td rowspan=1 colspan=1>80.6 ↑0.0</td></tr><tr><td rowspan=1 colspan=1>+VCD (Leng et al., 2024)</td><td rowspan=1 colspan=1>86.0 ↑</td><td rowspan=1 colspan=1>1.6</td><td rowspan=1 colspan=1>84.3 ↑2.0</td><td rowspan=1 colspan=1>84.5 ↑1.3</td><td rowspan=1 colspan=1>82.9 ↑</td><td rowspan=1 colspan=1>1.4</td><td rowspan=1 colspan=2>80.9 ↑1.4</td><td rowspan=1 colspan=1>79.7 ↑1.6</td><td rowspan=1 colspan=1>83.8 ↑</td><td rowspan=1 colspan=1>1.4</td><td rowspan=1 colspan=1>82.3 ↑1.7</td></tr><tr><td rowspan=2 colspan=3>+M3ID (Favero et al., 2024) 85.5 ↑1.1LLaVA-NEXT+VTI (Liu et al., 2025)      84.8 ↑0.4</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>83.6 ↑1.3</td><td rowspan=1 colspan=2>84.2 ↑1.0</td><td rowspan=1 colspan=1>82.4 ↑</td><td rowspan=1 colspan=1>0.9</td><td rowspan=1 colspan=1>80.6 ↑1.1</td><td rowspan=1 colspan=2>79.2 ↑1.1</td><td rowspan=1 colspan=1>83.4 ↑</td></tr><tr><td rowspan=1 colspan=1>83.1 ↑0.8</td><td rowspan=1 colspan=1>81.8 ↓1.4</td><td rowspan=1 colspan=1>81.6 ↑</td><td rowspan=1 colspan=1>0.1</td><td rowspan=1 colspan=2>79.0 ↓0.5</td><td rowspan=1 colspan=1>78.1 ↑0.0</td><td rowspan=1 colspan=1>81.9 ↓</td><td rowspan=1 colspan=1>0.5</td><td rowspan=1 colspan=1>80.9 ↑0.3</td></tr><tr><td rowspan=1 colspan=3>+AvisC (Woo et al., 2025)   85.2 ↑0.8</td><td rowspan=1 colspan=1>82.8 ↑0.5</td><td rowspan=1 colspan=1>83.9 ↑0.7</td><td rowspan=1 colspan=1>81.6 ↑</td><td rowspan=1 colspan=1>0.1</td><td rowspan=1 colspan=2>81.8 ↑2.3</td><td rowspan=1 colspan=1>80.2 ↑2.1</td><td rowspan=1 colspan=1>83.6 ↑</td><td rowspan=1 colspan=1>1.2</td><td rowspan=1 colspan=1>81.5 ↑0.9</td></tr><tr><td rowspan=1 colspan=3>+RVSD (Ours)           88.5 ↑4.1</td><td rowspan=1 colspan=1>87.6 ↑5.3</td><td rowspan=1 colspan=1>87.9 ↑4.7</td><td rowspan=1 colspan=1>86.7 ↑</td><td rowspan=1 colspan=1>5.2</td><td rowspan=1 colspan=2>86.5 ↑7.0</td><td rowspan=1 colspan=1>85.4 ↑7.3</td><td rowspan=1 colspan=1>87.6 ↑</td><td rowspan=1 colspan=1>5.2</td><td rowspan=1 colspan=1>86.6 ↑6.0</td></tr><tr><td rowspan=3 colspan=3>Vanilla                   84.9 ↑0.0</td><td rowspan=3 colspan=1>82.9 ↑0.0</td><td rowspan=3 colspan=1>84.0 ↑0.0</td><td rowspan=3 colspan=2>81.9 ↑0.0</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan=2 colspan=2>82.1 ↑0.0</td><td></td><td></td><td></td><td></td></tr><tr><td rowspan=1 colspan=1>80.2 ↑0.0</td><td rowspan=1 colspan=2>83.7 ↑0.0</td><td rowspan=1 colspan=1>81.7 ↑0.0</td></tr><tr><td rowspan=1 colspan=1>+VCD (Leng et al., 2024)</td><td rowspan=1 colspan=2>85.5 ↑0.6</td><td rowspan=1 colspan=1>83.6 ↑0.7</td><td rowspan=1 colspan=1>84.9 ↑0.9</td><td rowspan=1 colspan=1>83.6 ↑</td><td rowspan=1 colspan=1>1.7</td><td rowspan=1 colspan=1>84.0 ↑</td><td rowspan=1 colspan=1>1.9</td><td rowspan=1 colspan=1>82.0 ↑1.8</td><td rowspan=1 colspan=1>84.8 ↑</td><td rowspan=1 colspan=1>1.1</td><td rowspan=1 colspan=1>83.1 ↑1.4</td></tr><tr><td rowspan=2 colspan=1>+M3ID (Favero et al., 2024)Qwen-VL+VTI (Liu et al., 2025)</td><td rowspan=1 colspan=2>85.3 ↑0.4</td><td rowspan=1 colspan=1>83.4 ↑0.5</td><td rowspan=1 colspan=1>84.2 ↑0.2</td><td rowspan=1 colspan=1>82.7 ↑</td><td rowspan=1 colspan=1>0.8</td><td rowspan=1 colspan=1>83.2 ↑</td><td rowspan=1 colspan=1>1.1</td><td rowspan=1 colspan=1>80.8 ↑0.6</td><td rowspan=1 colspan=1>84.2 ↑</td><td rowspan=1 colspan=1>0.5</td><td rowspan=1 colspan=1>82.3 ↑0.6</td></tr><tr><td rowspan=1 colspan=2>85.3 ↑0.4</td><td rowspan=1 colspan=1>83.5 ↑0.6</td><td rowspan=1 colspan=1>83.0↓1.0</td><td rowspan=1 colspan=1>82.3 ↑</td><td rowspan=1 colspan=1>0.4</td><td rowspan=1 colspan=1>83.2 ↑</td><td rowspan=1 colspan=1>1.1</td><td rowspan=1 colspan=1>81.8 ↑1.6</td><td rowspan=1 colspan=1>83.8 ↑</td><td rowspan=1 colspan=1>0.1</td><td rowspan=1 colspan=1>82.5 ↑0.8</td></tr><tr><td rowspan=1 colspan=1>+AvisC (Woo et al., 2025)</td><td rowspan=1 colspan=2>82.9 ↓2.0</td><td rowspan=1 colspan=1>80.0 ↓2.9</td><td rowspan=1 colspan=1>82.8 ↓1.2</td><td rowspan=1 colspan=1>80.1 ↓</td><td rowspan=1 colspan=1>1.8</td><td rowspan=1 colspan=1>81.2 ↓</td><td rowspan=1 colspan=1>0.9</td><td rowspan=1 colspan=1>78.5 ↓1.7</td><td rowspan=1 colspan=1>82.3 ↓</td><td rowspan=1 colspan=1>1.4</td><td rowspan=1 colspan=1>79.5 ↓2.2</td></tr><tr><td rowspan=1 colspan=1>+RVSD (Ours)</td><td rowspan=1 colspan=2>86.2 ↑1.3</td><td rowspan=1 colspan=1>84.7 ↑1.8</td><td rowspan=1 colspan=1>85.3 ↑1.3</td><td rowspan=1 colspan=1>83.3 ↑</td><td rowspan=1 colspan=1>1.4</td><td rowspan=1 colspan=2>84.5 ↑2.4</td><td rowspan=1 colspan=1>82.6 ↑2.4</td><td rowspan=1 colspan=1>85.3 ↑</td><td rowspan=1 colspan=1>1.6</td><td rowspan=1 colspan=1>83.5 ↑1.8</td></tr></table>

Table 1: Performance comparison of RVSD with the baselines on the POPE-MSCOCO discrimination benchmark across the random, popular, and adversarial splits. The average column reports the mean accuracy and F1 scores over all splits. Green (•) and red (•) indicate improvements and degradations relative to the base model, respectively, while (↑/↓) denotes the absolute change. Best results are highlighted in bold.

Models. We evaluate RVSD on the 7B variants of LLaVA-1.5 (Liu et al., 2024b), LLaVA-NEXT (Liu et al., 2024c), and Qwen-VL (Bai et al., 2023). LLaVA-1.5 adopts a two-layer MLP for vision-language alignment, and LLaVA-NEXT further introduces the AnyRes mechanism to support higher-resolution visual understanding, while Qwen-VL instead couples a cross-attentionbased vision-language adapter with a non-LLaMA language backbone. Together, they cover both standard- and high-resolution settings as well as heterogeneous vision-language interfaces, enabling a comprehensive evaluation of RVSD. Moreover,

RVSD is model-agnostic and can be seamlessly integrated into different LVLM architectures.

Baselines. We compare RVSD with representative training-free baselines from three categories: (i) contrastive decoders, including VCD (Leng et al., 2024) and M3ID (Favero et al., 2024); (ii) the attention-intervention method AvisC (Woo et al., 2025); and (iii) the latent-space steering method VTI (Liu et al., 2025). All baselines leverage the hyperparameters recommended in their original papers under the same decoding budget, while the vanilla LVLM with sampling decoding serves as the base model.

## 5.2 Experimental Results

Results on POPE. In Table 1, we see that RVSD achieves the best performance across all POPE splits on the three backbones, obtaining the highest accuracy on every split and the highest F1 scores in almost all cases. On LLaVA-1.5-7B, RVSD improves the average Accuracy/F1 from 80.5/81.6 to 86.2/85.0. On LLaVA-NEXT, the scores increase from 82.4/80.6 to 87.6/86.6, and on Qwen-VL, from 83.7/81.7 to 85.3/83.5. The largest gains are observed on the adversarial split, with improvements of +8.9/+5.4, +7.0/+7.3, and +2.4/+2.4 on the three backbones, respectively. In contrast, VCD and M3ID yield only limited improvements, while VTI and AvisC occasionally degrade performance. These results demonstrate the effectiveness of combining semantics-directed sparsification with ondemand retrieval for reliable visual grounding.

![](images/8a20b59c8860b08d08ce57e969397fe20ee8f1caf77d4ee6ee1c45d1324839d7.jpg)  
Figure 4: Per-dimension MM-Vet scores of Vanilla decoding and RVSD on LLaVA-1.5, LLaVA-NEXT, and Qwen-VL, covering the Rec, OCR, Know, Gen, and Spat capability dimensions as well as the overall total score. RVSD improves every reported dimension on all three backbones.

Results on MM-Vet. To verify that VH mitigation does not compromise general multimodal capability, we evaluate RVSD on MM-Vet and report the Rec, OCR, Know, Gen, and Spat dimensions together with the overall total score in Fig. 4. We observe that RVSD improves the total score over the vanilla model on all three backbones, from 30.0 to 33.1 on LLaVA-1.5, from 42.1 to 42.7 on LLaVA-NEXT, and from 36.7 to 38.4 on Qwen-VL, with gains observed across every reported dimension. The improvements are most pronounced on Know, suggesting that restoring pruned visual evidence also benefits knowledge-intensive reasoning. The results demonstrate that RVSD mitigates VHs while preserving general multimodal competence.

Results on MME. As shown in Fig. 5, RVSD obtains the highest total score on the MME hallucination subset across all three backbones, reaching 651.7 on LLaVA-1.5, 668.3 on LLaVA-NEXT, and 593.4 on Qwen-VL, and consistently surpassing the vanilla decoder on the total axis of every panel. On the two LLaVA backbones, the improvements are concentrated on the attribute-level probes, where RVSD achieves the highest scores on both Color and Position, whereas the scores of most baselines decline markedly on Position for LLaVA-NEXT. On Qwen-VL, the gain is instead driven by object Existence, while Count and Position remain below the strongest baselines. These results further demonstrate the effectiveness of RVSD in mitigating fine-grained VHs.

Results on CHAIR and AMBER. As shown in Table 2, we further evaluate caption-level behavior on CHAIR and AMBER, jointly assessing discrimination (Accuracy/F1) and generative hallucination (CHAIR , CHAIR , AMBER CHAIR, HalRate, Score). We observe that RVSD achieves the best discrimination performance and the highest AM-BER score across all three backbones, and obtains the lowest CHAIR on the two LLaVA backbones and the lowest CHAIR<sub>S</sub> on Qwen-VL, indicating effective suppression of object-level hallucinations during long-form generation. In contrast, existing baselines provide only limited or inconsistent improvements. Overall, RVSD consistently improves both faithfulness and caption quality across different LVLM architectures, demonstrating the effectiveness of combining semantics-directed sparsification with on-demand visual retrieval.

## 5.3 Ablation Studies

Mechanism Contribution. To evaluate the contribution of each component, we conduct an ablation study on the full MME benchmark using LLaVA-1.5 by separately removing the semantics-directed sparse selection (w/o Sparse) and the SSVR module (w/o SSVR). As shown in Table 3, removing either component degrades the overall total score, demonstrating that both modules are essential to RVSD. In particular, removing SSVR causes the largest performance drop, reducing Cognition from 371.79 to 348.21 and yielding the greatest decline in the total score, highlighting the importance of on-demand visual retrieval for restoring visual grounding during reasoning-intensive decoding. In contrast, removing sparse selection slightly improves the perception score by retaining all visual tokens, but noticeably reduces the cognition score by 14.29, indicating that dense decoding introduces redundant visual information that weakens effective grounding. Overall, RVSD achieves the best total and cognition scores, demonstrating that semantics-directed sparsification and SSVR complement each other to improve visual grounding.

## 5.4 Inference Latency

As shown in Table 4, we further benchmark inference efficiency under the same decoding budget. We observe that RVSD is the only method that maintains the same latency as the Vanilla baseline (1.00×) while simultaneously reducing memory usage by 2.2%, Tera Floating-Point Operations Per Second (TFLOPs) by 15.6%, and Time-to-First-

![](images/3e457aa848349164e2a196cd26d2bd8b902594238277b51ba5002988ea2feaef.jpg)  
Figure 5: Performance comparison of RVSD with the baselines on the MME hallucination subset for (a) LLaVA-1.5, (b) LLaVA-NEXT, and (c) Qwen-VL, covering object-level (Existence, Count) and attribute-level (Position, Color) probes, with Total denoting their sum. Values annotated in red are those of RVSD, and each axis is scaled independently.

<table><tr><td rowspan="2">Models</td><td rowspan="2">Methods</td><td colspan="4">CHAIR</td><td colspan="5">AMBER</td></tr><tr><td>CHAIRS↓</td><td>CHAIRI↓</td><td>Recall↑</td><td>Length</td><td>Acc (%)↑</td><td>F1 (%)↑</td><td>CHAIR↓</td><td>HalRate↓</td><td>Score↑</td></tr><tr><td rowspan="5">LLaVA-1.5</td><td>Vanilla</td><td>52.8 ↓0.0</td><td>15.9 ↓0.0</td><td>77.3 ↑0.0</td><td>93.4</td><td>67.0 ↑0.0</td><td>71.0 ↑0.0</td><td>12.0 ↓0.0</td><td>51.0 ↓0.0</td><td>79.5 ↑0.0</td></tr><tr><td>+VCD (Leng et al., 2024)</td><td>51.0↓1.8</td><td>14.9↓1.0</td><td>77.2 ↓0.1</td><td>101.9</td><td>67.3 ↑0.3</td><td>71.1 ↑0.1</td><td>10.0 ↓2.0</td><td>43.6↓7.4</td><td>80.6 ↑1.1</td></tr><tr><td>+M3ID (Favero et al., 2024)</td><td>56.2 ↑3.4</td><td>17.0 ↑1.1</td><td>79.3 ↑2.0</td><td>97.1</td><td>67.3 ↑0.3</td><td>70.9 ↓0.1</td><td>9.1 ↓2.9</td><td>42.8 ↓8.2</td><td>80.9 ↑1.4</td></tr><tr><td>+AvisC (Woo et al., 2025)</td><td>44.0↓8.8</td><td>13.7 ↓2.2</td><td>72.9 ↓4.4</td><td>89.8</td><td>70.7 ↑3.7</td><td>74.2 ↑3.2</td><td>11.8 ↓0.2</td><td>51.2 ↑0.2</td><td>81.2 ↑1.7</td></tr><tr><td>+VTI (Liu et al., 2025)</td><td>36.8↓16.</td><td>13.6 ↓2.3</td><td>66.6↓10.</td><td>66.7</td><td>66.5 ↓0.5</td><td>70.6 ↓0.4</td><td>6.9 ↓5.1</td><td>27.0 ↓24.</td><td>81.9 ↑2.4</td></tr><tr><td></td><td>+RVSD (Ours)</td><td>38.4↓14.</td><td>10.3 ↓5.6</td><td>74.4 ↓2.9</td><td>90.6</td><td>73.6 ↑6.6</td><td>76.9 ↑5.9</td><td>7.3 ↓4.7</td><td>31.2↓19.</td><td>84.8 ↑5.3</td></tr><tr><td rowspan="5">LLaVA-NEXT</td><td>Vanilla</td><td>35.8↓0.0</td><td>12.0 ↓0.0</td><td>59.5 ↑0.0</td><td>179.0</td><td>72.9 ↑0.0</td><td>78.6↑0.0</td><td>12.0 ↓0.0</td><td>59.6 ↓0.0</td><td>83.3 ↑0.0</td></tr><tr><td>+VCD (Leng et al., 2024)</td><td>40.2 ↑4.4</td><td>10.7 ↓1.3</td><td>62.1 ↑2.6</td><td>171.2</td><td>74.3 ↑1.4</td><td>79.6 ↑1.0</td><td>11.8 ↓0.2</td><td>58.6↓1.0</td><td>83.9 ↑0.6</td></tr><tr><td>+M3ID (Favero et al., 2024)</td><td>35.2 ↓0.6</td><td>10.3 ↓1.7</td><td>61.4 ↑1.9</td><td>152.7</td><td>75.0 ↑2.1</td><td>80.0 ↑1.4</td><td>10.2↓1.8</td><td>52.8 ↓6.8</td><td>84.9 ↑1.6</td></tr><tr><td>+AvisC (Woo et al., 2025)</td><td>40.4 ↑4.6</td><td>12.4 ↑0.4</td><td>60.0 ↑0.5</td><td>183.8</td><td>77.7 ↑4.8</td><td>82.1 ↑3.5</td><td>14.2 ↑2.2</td><td>65.2 ↑5.6</td><td>84.0 ↑0.7</td></tr><tr><td>+VTI (Liu et al., 2025)</td><td>26.4↓9.4</td><td>7.7 ↓4.3</td><td>58.5 ↓1.0</td><td>187.6</td><td>75.2 ↑2.3</td><td>81.9 ↑3.3</td><td>10.3 ↓1.7</td><td>58.3 ↓1.3</td><td>87.1 ↑3.8</td></tr><tr><td></td><td>+RVSD (Ours)</td><td>26.2 ↓9.6</td><td>6.7 ↓5.3</td><td>63.7 ↑4.2</td><td>185.0</td><td>80.0 ↑7.1</td><td>83.8 ↑5.2</td><td>8.9 ↓3.1</td><td>47.9↓11.</td><td>87.5 ↑4.2</td></tr><tr><td rowspan="5">Qwen-VL</td><td>Vanilla</td><td>2.8 ↓0.0</td><td>3.0 ↓0.0</td><td>31.0 ↑0.0</td><td>5.3</td><td>82.9 ↑0.0</td><td>86.9 ↑0.0</td><td>4.8 ↓0.0</td><td>9.2 ↓0.0</td><td>91.1 ↑0.0</td></tr><tr><td>+VCD (Leng et al., 2024)</td><td>1.4↓1.4</td><td>1.2 ↓1.8</td><td>30.8 ↓0.2</td><td>4.0</td><td>84.1 ↑1.2</td><td>87.9↑1.0</td><td>3.5 ↓1.3</td><td>7.7 ↓1.5</td><td>92.2 ↑1.1</td></tr><tr><td>+M3ID (Favero et al., 2024)</td><td>1.7 ↓1.1</td><td>1.3 ↓1.7</td><td>31.8 ↑0.8</td><td>3.4</td><td>84.0 ↑1.1</td><td>87.1 ↑0.2</td><td>3.7↓1.1</td><td>8.2↓1.0</td><td>91.7 ↑0.6</td></tr><tr><td>+AvisC (Woo et al., 2025)</td><td>1.6↓1.2</td><td>1.6↓1.4</td><td>32.0 ↑1.0</td><td>4.4</td><td>84.1 ↑1.2 83.5 ↑0.6</td><td>87.9 ↑1.0 87.3 ↑0.4</td><td>4.3 ↓0.5</td><td>8.6↓0.6</td><td>91.8 ↑0.7</td></tr><tr><td>+VTI (Liu et al., 2025)</td><td>1.9↓0.9</td><td>1.9↓1.1</td><td>30.2 ↓0.8</td><td>4.5</td><td></td><td></td><td>3.1 ↓1.7</td><td>6.8 ↓2.4</td><td>92.4 ↑1.3</td></tr><tr><td></td><td>+RVSD (Ours)</td><td>1.2↓1.6</td><td>1.6 ↓1.4</td><td>31.9 ↑0.9</td><td>5.9</td><td>84.3 ↑1.4</td><td>88.6 ↑1.7</td><td>3.4↓1.4</td><td>7.3 ↓1.9</td><td>92.6 ↑1.5</td></tr></table>

Table 2: Performance comparison on the CHAIR benchmark (CHAIR<sub>S</sub>, CHAIR<sub>I</sub>, Recall, Length) and the AMBER benchmark for open-ended captioning, where the AMBER block jointly reports discrimination metrics (Accuracy, F1) and generative metrics (CHAIR, HalRate, Score). Best results are highlighted in bold.

<table><tr><td>Methods</td><td colspan="3">Perception↑ Cognition↑</td></tr><tr><td>RVSD</td><td>1511.40</td><td>371.79</td><td>1883.19</td></tr><tr><td>w/o Sparse</td><td>1514.60</td><td>357.50</td><td>1872.10</td></tr><tr><td>w/o SSVR</td><td>1520.36</td><td>348.21</td><td>1868.57</td></tr></table>

Table 3: Component-level ablation study on the full MME benchmark using LLaVA-1.5, evaluating the contributions of the semantics-directed sparse selection module and the SSVR module. Best results are highlighted in bold.

Token (TTFT) by 20.2%. In contrast, contrastive decoding methods (VCD and M3ID) introduce over 2× latency overhead due to auxiliary forward passes, while AvisC incurs the highest cost from dual forward inference and additional attention extraction. Although VTI is relatively lightweight, it still increases latency by 1.09×. These results demonstrate that RVSD effectively improves visual grounding without sacrificing inference efficiency, benefiting from semantics-directed sparsification and the uncertainty-triggered retrieval mechanism that is activated only when necessary.

## 6 Conclusion

In this paper, we have proposed RVSD, a trainingfree and plug-and-play decoding framework that, for the first time, resolves the previously overlooked sparsification-hallucination paradox in existing sparse decoders. By integrating semanticsdirected token selection with the SSVR mechanism within a single decoding pass, RVSD transforms irreversible visual token pruning into reversible and on-demand cross-modal retrieval, enabling visual grounding to be restored only when necessary. Extensive experiments on five representative benchmarks across three LVLM backbones demonstrate that RVSD achieves SOTA performance in VH mitigation while simultaneously improving decoding efficiency and maintaining robustness under long-context generation, where existing sparse and decoding-based methods often deteriorate.

<table><tr><td>Methods</td><td>Latency ↓</td><td>Memory↓</td><td>TFLOPs↓</td><td>TTFT ↓</td></tr><tr><td>Vanilla</td><td>61.9</td><td>14334</td><td>4.36</td><td>243.8</td></tr><tr><td>+VCD</td><td>131.6</td><td>14979</td><td>8.72</td><td>445.4</td></tr><tr><td>+M3ID</td><td>132.5</td><td>14977</td><td>8.72</td><td>477.8</td></tr><tr><td>+AvisC</td><td>173.6</td><td>15501</td><td>8.45</td><td>725.2</td></tr><tr><td>+VTI</td><td>67.3</td><td>14363</td><td>4.36</td><td>266.3</td></tr><tr><td>+RVSD</td><td>62.0</td><td>14015</td><td>3.68</td><td>194.6</td></tr></table>

Table 4: Inference efficiency comparison between RVSD and representative decoding-side VH mitigation methods, reporting per-token decoding latency (ms/token), peak GPU memory usage (MB), TFLOPs (T), and TTFT (ms). Best results are highlighted in bold.

For future work, we will extend RVSD to more challenging multimodal settings, such as long-form video understanding, where extended temporal dependencies and richer modality interactions further exacerbate the sparsification-hallucination paradox. We hope this work provides a promising direction toward trustworthy and efficient LVLM inference.

## Limitations

While RVSD shows promising results, several limitations remain. Firstly, the uncertainty-triggered retrieval gate in SSVR relies on a predefined entropy threshold to determine when cross-modal retrieval should be activated. Although RVSD remains robust across a reasonable range of threshold values, a more adaptive gating strategy that dynamically adjusts the trigger according to input complexity can further reduce unnecessary retrieval calls and improve efficiency on simpler queries.

Additionally, the current implementation uses a simple dot-product similarity to retrieve evidence from the deferred visual memory bank. More advanced retrieval mechanisms, such as learned semantic projections or multi-head cross-modal attention, may further improve performance on finegrained tasks. Although our evaluation already spans two distinct architectural families, i.e., the LLaVA series and Qwen-VL, we also plan to extend RVSD to larger-scale and more heterogeneous LVLMs to further evaluate its generality.

## Ethical Considerations

This work aims to improve the trustworthiness of LVLMs by mitigating VHs, thereby contributing to the safer deployment of multimodal artificial intelligence systems. RVSD is a training-free and inference-only framework that requires no additional data collection or human annotation. All datasets (i.e., POPE, MME, AMBER, CHAIR, MM-Vet, MSCOCO, A-OKVQA, and GQA) and backbone models (LLaVA-1.5, LLaVA-NEXT, and Qwen-VL) used in this study are publicly available for academic research and are used strictly in accordance with their original licenses. While RVSD substantially reduces VHs, it does not eliminate them. Therefore, we recommend pairing it with downstream verification or human oversight before deployment in safety-critical applications such as medical diagnosis or assistive technologies.

## Acknowledgements

This work was supported in part by the National Natural Science Foundation of China (NSFC) under Grant No. 62572132, and the Guangdong Basic and Applied Basic Research Foundation under Grant No. 2025A1515010137.

## References

Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. 2023. Qwen-vl: A frontier large vision-language model with versatile abilities. arXiv preprint arXiv:2308.12966.

Zechen Bai, Pichao Wang, Tianjun Xiao, Tong He, Zongbo Han, Zheng Zhang, and Mike Zheng Shou. 2024. Hallucination of multimodal large language models: A survey. arXiv preprint arXiv:2404.18930.

Liwei Che, Tony Qingze Liu, Jing Jia, Weiyi Qin, Ruixiang Tang, and Vladimir Pavlovic. 2025. Hallucinatory image tokens: A training-free eazy approach to detecting and mitigating object hallucinations in lvlms. In 2025 IEEE/CVF International Conference on Computer Vision (ICCV), pages 21635–21644. IEEE.

Liang Chen, Haozhe Zhao, Tianyu Liu, Shuai Bai, Junyang Lin, Chang Zhou, and Baobao Chang. 2024. An image is worth 1/2 tokens after layer 2: Plug-and-play

inference acceleration for large vision-language models. In European Conference on Computer Vision, pages 19–35. Springer.

Tiantian Dang, Chao Bi, Shufan Shen, Jinzhe Liu, Qingming Huang, and Shuhui Wang. 2026. Locatethen-sparsify: Attribution guided sparse strategy for visual hallucination mitigation. arXiv preprint arXiv:2603.16284.

Alessandro Favero, Luca Zancato, Matthew Trager, Siddharth Choudhary, Pramuditha Perera, Alessandro Achille, Ashwin Swaminathan, and Stefano Soatto. 2024. Multi-modal hallucination control by visual information grounding. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 14303–14312. IEEE.

Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Jinrui Yang, Xiawu Zheng, Ke Li, Xing Sun, and 1 others. 2026. Mme: A comprehensive evaluation benchmark for multimodal large language models. Advances in Neural Information Processing Systems, 38.

Jihoon Lee and Min Song. 2025. Retrieval visual contrastive decoding to mitigate object hallucinations in large vision-language models. In Findings of the Associationfor Computational Linguistics: ACL 2025, pages 8200–8219.

Sicong Leng, Hang Zhang, Guanzheng Chen, Xin Li, Shijian Lu, Chunyan Miao, and Lidong Bing. 2024. Mitigating object hallucinations in large visionlanguage models through visual contrastive decoding. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 13872– 13882. IEEE.

Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, and 1 others. 2020. Retrieval-augmented generation for knowledge-intensive nlp tasks. Advances in neural information processing systems, 33:9459– 9474.

Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Xin Zhao, and Ji-Rong Wen. 2023. Evaluating object hallucination in large vision-language models. In Proceedings of the 2023 conference on empirical methods in natural language processing, pages 292– 305.

Hanchao Liu, Wenyuan Xue, Yifei Chen, Dapeng Chen, Xiutian Zhao, Ke Wang, Liping Hou, Rongjun Li, and Wei Peng. 2024a. A survey on hallucination in large vision-language models. arXiv preprint arXiv:2402.00253.

Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. 2024b. Improved baselines with visual instruction tuning. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 26286–26296. IEEE.

Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. 2024c. Llavanext: Improved reasoning, ocr, and world knowledge.

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. 2023. Visual instruction tuning. Advances in neural information processing systems, 36:34892– 34916.

Sheng Liu, Haotian Ye, and James Y Zou. 2025. Reducing hallucinations in large vision-language models via latent space steering. In International Conference on Learning Representations, volume 2025, pages 72402–72419.

Anna Rohrbach, Lisa Anne Hendricks, Kaylee Burns, Trevor Darrell, and Kate Saenko. 2018. Object hallucination in image captioning. In Proceedings of the 2018 conference on empirical methods in natural language processing, pages 4035–4045.

Fengyuan Sun, Hui Chen, Xinhao Xu, Dandan Zheng, Jingdong Chen, Jun Zhou, Jungong Han, and Guiguang Ding. 2025. Prunehal: Reducing hallucinations in multi-modal large language models through adaptive kv cache pruning. arXiv preprint arXiv:2510.19183.

Zhiqing Sun, Sheng Shen, Shengcao Cao, Haotian Liu, Chunyuan Li, Yikang Shen, Chuang Gan, Liangyan Gui, Yu-Xiong Wang, Yiming Yang, and 1 others. 2024. Aligning large multimodal models with factually augmented rlhf. In Findings of the association for computational linguistics: ACL 2024, pages 13088–13110.

Chao Wang, Jianming Yang, and Yang Zhou. 2025. Mint: Mitigating hallucinations in large visionlanguage models via token reduction. arXiv preprint arXiv:2502.00717.

Junyang Wang, Yuhang Wang, Guohai Xu, Jing Zhang, Yukai Gu, Haitao Jia, Jiaqi Wang, Haiyang Xu, Ming Yan, Ji Zhang, and 1 others. 2023a. Amber: An llmfree multi-dimensional benchmark for mllms hallucination evaluation. arXiv preprint arXiv:2311.07397.

Sheng Wang, Zihao Zhao, Xi Ouyang, Qian Wang, and Dinggang Shen. 2023b. Chatcad: Interactive computer-aided diagnosis on medical image using large language models. arXiv preprint arXiv:2302.07257.

Sangmin Woo, Donguk Kim, Jaehyuk Jang, Yubin Choi, and Changick Kim. 2025. Don’t miss the forest for the trees: Attentional vision calibration for large vision language models. In Findings ofthe Association for Computational Linguistics: ACL 2025, pages 1927–1951.

Jiayang Wu, Wensheng Gan, Zefeng Chen, Shicheng Wan, and Philip S Yu. 2023. Multimodal large language models: A survey. In 2023 IEEE international conference on big data (bigdata), pages 2247–2256. IEEE.

Junfei Wu, Yue Ding, Guofan Liu, Tianze Xia, Ziyue Huang, Dianbo Sui, Qiang Liu, Shu Wu, Liang Wang, and Tieniu Tan. 2025. Sharp: Steering hallucination in lvlms via representation engineering. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 14357–14372.

Qiong Wu, Wenhao Lin, Yiyi Zhou, Weihao Ye, Zhanpeng Zeng, Xiaoshuai Sun, and Rongrong Ji. 2026. Accelerating multimodal large language models via dynamic visual-token exit and the empirical findings. Advances in Neural Information Processing Systems, 38:168378–168403.

Shukang Yin, Chaoyou Fu, Sirui Zhao, Ke Li, Xing Sun, Tong Xu, and Enhong Chen. 2024. A survey on multimodal large language models. National Science Review, 11(12).

Weihao Yu, Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Zicheng Liu, Xinchao Wang, and Lijuan Wang. 2024. MM-vet: Evaluating large multimodal models for integrated capabilities. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 57730–57754.

Yuan Zhang, Chun-Kai Fan, Junpeng Ma, Wenzhao Zheng, Tao Huang, Kuan Cheng, Denis A Gudovskiy, Tomoyuki Okuno, Yohei Nakata, Kurt Keutzer, and Shanghang Zhang. 2025. SparseVLM: Visual token sparsification for efficient vision-language model inference. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 74840–74857.

Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Zican Dong, Yupeng Hou, Beichen Zhang, Yingqian Min, Junjie Zhang, Peiyu Liu, and 1 others. 2026. A survey of large language models. Frontiers of Computer Science, 20(12):2012627.

Xingyu Zhu, Kesen Zhao, Liang Yi, Shuo Wang, Zhicai Wang, Beier Zhu, Hanwang Zhang, and Xiangnan He. 2026. Look carefully: Adaptive visual reinforcements in multimodal large language models for hallucination mitigation. In International Conference on Learning Representations, volume 2026, pages 81235–81256.

Xianwei Zhuang, Zhihong Zhu, Yuxin Xie, Liming Liang, and Yuexian Zou. 2025. Vasparse: Towards efficient visual hallucination mitigation via visualaware token sparsification. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 4189–4199. IEEE.

## Appendix

## A More Experiments and Results

Results on POPE A-OKVQA and GQA. To further evaluate the cross-dataset generalization of RVSD, we conduct additional experiments on the POPE A-OKVQA and GQA subsets, as shown in Tables 5 and 6. Compared with POPE-MSCOCO (refer to Table 1), these datasets contain denser scenes and stronger object co-occurrence biases, making the popular and adversarial splits more challenging. RVSD achieves the best accuracy across all splits on all three backbones and the best F1 scores on the two LLaVA backbones, with the largest improvements observed on the adversarial split. In particular, RVSD improves Accuracy/F1 by 11.6/5.4 on LLaVA-1.5 A-OKVQA and by 10.9/3.8 on LLaVA-1.5 GQA, demonstrating strong robustness against misleading linguistic priors. On Qwen-VL, RVSD still obtains the highest accuracy on every split, improving the adversarial accuracy by 2.4 on A-OKVQA and by 4.4 on GQA, while its F1 scores remain comparable to the strongest contrastive baseline. In contrast, existing baselines provide only limited or inconsistent improvements, while AvisC notably degrades the average performance on both A-OKVQA and GQA. These results demonstrate that the effectiveness of RVSD generalizes consistently across different POPE datasets and is not dependent on a specific data distribution.

## B Detailed Experimental Setup

## B.1 Datasets and Benchmarks

We provide additional details for the five benchmarks used in the main paper, all of which are publicly released and standard in the VH literature. POPE (Li et al., 2023) evaluates object-level hallucination through binary “Yes/No” existence questions and reports accuracy and F1 scores under three sampling strategies, i.e., random, popular, and adversarial splits. The popular split samples frequently co-occurring object categories. In contrast, the adversarial split selects objects with strong co-occurrence priors relative to the image, making it particularly challenging for VH mitigation. We evaluate POPE on three data sources, namely MSCOCO (Rohrbach et al., 2018), A-OKVQA, and GQA, each containing 3,000 yes/no queries.

MME Hallucination Subset (Fu et al., 2026) evaluates hallucination through four probing tasks, including object-level probes (Existence, Count) and attribute-level probes (Position, Color). Each task contains 30 binary questions and is evaluated using Acc/Acc+, where Acc+ requires both yes/no questions associated with the same image to be answered correctly. The total score is computed as the sum of the four subtask scores.

AMBER (Wang et al., 2023a) contains both a discrimination subset and a generative subset. The discrimination subset uses binary yes/no questions and is evaluated with accuracy and F1 scores, while the generative subset evaluates open-ended captioning using CHAIR, HalRate, and a holistic Score that jointly measures caption coverage and hallucination penalty. AMBER covers existence-, attribute-, and relation-level hallucinations, with a particular emphasis on long-form caption generation.

CHAIR (Rohrbach et al., 2018) evaluates object hallucination in generated captions by extracting mentioned objects and matching them against MSCOCO ground-truth annotations. It reports two metrics, which are given by

$$
\mathrm { C H A I R } _ { I } = \frac { \lvert \{ \mathrm { h a l l u c i n a t e d o b j e c t s } \} \rvert } { \lvert \{ \mathrm { a l l m e n t i o n e d o b j e c t s } \} \rvert } ,\tag{14}
$$

$$
\mathrm { C H A I R } _ { S } = \frac { \left| \left\{ \mathrm { c a p t i o n s w / h a l l u c i n a t i o n } \right\} \right| } { \left| \left\{ \mathrm { a l l c a p t i o n s } \right\} \right| } .\tag{15}
$$

Following (Liu et al., 2025), we randomly sample 500 MSCOCO Val2014 images and prompt the model with “Please describe this image in detail” to elicit long-form responses.

MM-Vet (Yu et al., 2024) consists of 218 evaluation samples designed to probe six vision-language capabilities: Recognition (Rec), OCR, Knowledge (Know), Generation (Gen), Spatial reasoning (Spat), and Math. We follow the official evaluation protocol and use GPT-4 as the judge to assign a continuous score for each response, with the total score computed as the mean across all dimensions.

## B.2 Baselines

We compare RVSD against four representative training-free decoders spanning three paradigms. All baselines follow the recommended hyperparameters and are evaluated under the same decoding budget as RVSD for a fair comparison.

VCD (Visual Contrastive Decoding) (Leng et al., 2024) introduces a perturbed visual input by adding diffusion-style Gaussian noise and performs contrastive decoding between the clean and corrupted visual conditions, thereby suppressing unimodal language priors.

<table><tr><td rowspan=1 colspan=12>Random            Popular          Adversarial           AverageModels       MethodsAcc (%)↑ F1 (%)↑ Acc (%)↑ F1 (%)↑ Acc (%)↑ F1 (%)↑ Acc (%)↑ F1 (%)↑</td></tr><tr><td rowspan=1 colspan=7>Vanilla                   81.8 ↑0.083.5 ↑0.075.3 ↑0.078.7 ↑0.0</td><td rowspan=1 colspan=5>67.4 ↑0.0 73.7 ↑0.0 74.8 ↑0.0 78.6 ↑0.0</td></tr><tr><td rowspan=3 colspan=3>+VCD (Leng et al., 2024)   81.2 ↓0.6+M3ID (Favero et al., 2024) 82.9 ↑1.1LLaVA-1.5+AvisC (Woo et al., 2025)   79.1 ↓2.7</td><td rowspan=1 colspan=1>83.2 ↓0.3</td><td rowspan=1 colspan=1>74.7 ↓0.6</td><td rowspan=1 colspan=2>78.5 ↓0.2</td><td rowspan=1 colspan=1>68.1 ↑0.7</td><td rowspan=1 colspan=1>74.6 ↑0.9</td><td rowspan=1 colspan=2>74.7 ↓0.1</td><td rowspan=1 colspan=1>78.8 ↑0.2</td></tr><tr><td rowspan=1 colspan=1>84.6 ↑1.1</td><td rowspan=1 colspan=1>75.8 ↑0.5</td><td rowspan=1 colspan=1>79.4 ↑</td><td rowspan=1 colspan=1>0.7</td><td rowspan=1 colspan=1>68.3 ↑0.9</td><td rowspan=1 colspan=1>74.6 ↑0.9</td><td rowspan=1 colspan=1>75.7 ↑</td><td rowspan=1 colspan=1>0.9</td><td rowspan=1 colspan=1>79.5 ↑0.9</td></tr><tr><td rowspan=1 colspan=1>82.4 ↓1.1</td><td rowspan=1 colspan=1>71.8 ↓3.5</td><td rowspan=1 colspan=2>77.2 ↓1.5</td><td rowspan=1 colspan=1>64.4 ↓3.0</td><td rowspan=1 colspan=1>73.0 ↓0.7</td><td rowspan=1 colspan=1>71.8 ↓</td><td rowspan=1 colspan=1>3.0</td><td rowspan=1 colspan=1>77.5 ↓1.1</td></tr><tr><td rowspan=1 colspan=3>+VTI (Liu et al., 2025)      83.0 ↑1.2</td><td rowspan=1 colspan=1>83.6 ↑0.1</td><td rowspan=1 colspan=1>76.1 ↑0.8</td><td rowspan=1 colspan=2>79.5 ↑0.8</td><td rowspan=1 colspan=1>68.2 ↑0.8</td><td rowspan=1 colspan=1>74.6 ↑0.9</td><td rowspan=1 colspan=1>75.8 ↑</td><td rowspan=1 colspan=1>1.0</td><td rowspan=1 colspan=1>79.2 ↑0.6</td></tr><tr><td rowspan=1 colspan=3>+RVSD (Ours)            87.5 ↑5.7</td><td rowspan=1 colspan=1>86.4 ↑2.9</td><td rowspan=1 colspan=1>85.4 ↑10.</td><td rowspan=1 colspan=2>84.5 ↑5.8</td><td rowspan=1 colspan=1>79.0 ↑11.</td><td rowspan=1 colspan=1>79.1 ↑5.4</td><td rowspan=1 colspan=1>84.0 ↑</td><td rowspan=1 colspan=1>9.2</td><td rowspan=1 colspan=1>83.3 ↑4.7</td></tr><tr><td rowspan=1 colspan=3>Vanilla                   83.8 ↑0.0</td><td rowspan=1 colspan=1>83.0 ↑0.0</td><td rowspan=1 colspan=1>81.4 ↑0.0</td><td rowspan=1 colspan=2>80.8 ↑0.0</td><td rowspan=1 colspan=1>73.2 ↑0.0</td><td rowspan=1 colspan=1>74.5 ↑0.0</td><td rowspan=1 colspan=2>79.5 ↑0.0</td><td rowspan=1 colspan=1>79.4 ↑0.0</td></tr><tr><td rowspan=1 colspan=1>+VCD (Leng et al., 2024)</td><td rowspan=1 colspan=1>84.8 ↑</td><td rowspan=1 colspan=1>1.0</td><td rowspan=1 colspan=1>83.9 ↑0.9</td><td rowspan=1 colspan=1>81.5 ↑0.1</td><td rowspan=1 colspan=1>81.2 ↑</td><td rowspan=1 colspan=1>0.4</td><td rowspan=1 colspan=1>74.7 ↑1.5</td><td rowspan=1 colspan=1>76.0 ↑1.5</td><td rowspan=1 colspan=1>80.3 ↑</td><td rowspan=1 colspan=1>0.8</td><td rowspan=1 colspan=1>80.4 ↑1.0</td></tr><tr><td rowspan=2 colspan=3>+M3ID (Favero et al., 2024) 85.3 ↑1.5LLaVA-NEXT+AvisC (Woo et al., 2025)   85.8 ↑2.0</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>84.3 ↑1.3</td><td rowspan=1 colspan=2>82.2 ↑0.8</td><td rowspan=1 colspan=1>81.6 ↑</td><td rowspan=1 colspan=1>0.8</td><td rowspan=1 colspan=2>74.6 ↑1.4</td><td rowspan=1 colspan=1>75.7 ↑1.2</td></tr><tr><td rowspan=1 colspan=1>84.4 ↑1.4</td><td rowspan=1 colspan=1>83.9 ↑2.5</td><td rowspan=1 colspan=1>82.8 ↑</td><td rowspan=1 colspan=1>2.0</td><td rowspan=1 colspan=1>75.6 ↑2.4</td><td rowspan=1 colspan=1>76.6 ↑2.1</td><td rowspan=1 colspan=1>81.8 ↑</td><td rowspan=1 colspan=1>2.3</td><td rowspan=1 colspan=1>81.3 ↑1.9</td></tr><tr><td rowspan=1 colspan=3>+VTI (Liu et al., 2025)     84.8 ↑1.0</td><td rowspan=1 colspan=1>84.2 ↑1.2</td><td rowspan=1 colspan=1>80.4↓1.0</td><td rowspan=1 colspan=1>79.7 ↓</td><td rowspan=1 colspan=1>1.1</td><td rowspan=1 colspan=1>72.8 ↓0.4</td><td rowspan=1 colspan=1>74.3 ↓0.2</td><td rowspan=1 colspan=1>79.3 ↓</td><td rowspan=1 colspan=1>0.2</td><td rowspan=1 colspan=1>79.4 ↑0.0</td></tr><tr><td rowspan=1 colspan=3>+RVSD (Ours)            90.1 ↑6.3</td><td rowspan=1 colspan=1>89.6 ↑6.6</td><td rowspan=1 colspan=1>88.6 ↑7.2</td><td rowspan=1 colspan=1>88.2 ↑</td><td rowspan=1 colspan=1>7.4</td><td rowspan=1 colspan=1>81.8 ↑8.6</td><td rowspan=1 colspan=1>82.4 ↑7.9</td><td rowspan=1 colspan=1>86.8 ↑</td><td rowspan=1 colspan=1>7.3</td><td rowspan=1 colspan=1>86.7 ↑7.3</td></tr><tr><td rowspan=3 colspan=3>Vanilla                   86.8 ↑0.0</td><td rowspan=3 colspan=1>85.8 ↑0.0</td><td rowspan=3 colspan=1>85.6 ↑0.0</td><td rowspan=3 colspan=2>84.7 ↑0.0</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan=2 colspan=1>80.4 ↑0.0</td><td rowspan=2 colspan=1>80.5 ↑0.0</td><td rowspan=2 colspan=1>84.3 ↑</td><td></td><td></td></tr><tr><td rowspan=1 colspan=1>0.0</td><td rowspan=1 colspan=1>83.7 ↑0.0</td></tr><tr><td rowspan=1 colspan=1>+VCD (Leng et al., 2024)</td><td rowspan=1 colspan=2>87.4 ↑0.6</td><td rowspan=1 colspan=1>86.6 ↑0.8</td><td rowspan=1 colspan=1>86.3 ↑0.7</td><td rowspan=1 colspan=1>85.1 ↑</td><td rowspan=1 colspan=1>0.4</td><td rowspan=1 colspan=1>80.7 ↑0.3</td><td rowspan=1 colspan=1>80.8 ↑0.3</td><td rowspan=1 colspan=1>84.8 ↑</td><td rowspan=1 colspan=1>0.5</td><td rowspan=1 colspan=1>84.2 ↑0.5</td></tr><tr><td rowspan=2 colspan=1>+M3ID (Favero et al., 2024)Qwen-VL+AvisC (Woo et al., 2025)</td><td rowspan=1 colspan=2>87.1 ↑0.3</td><td rowspan=1 colspan=1>85.9 ↑0.1</td><td rowspan=1 colspan=1>85.9 ↑0.3</td><td rowspan=1 colspan=1>84.6↓</td><td rowspan=1 colspan=1>0.1</td><td rowspan=1 colspan=1>80.5 ↑0.1</td><td rowspan=1 colspan=1>80.4 ↓0.1</td><td rowspan=1 colspan=1>84.5 ↑</td><td rowspan=1 colspan=1>0.2</td><td rowspan=1 colspan=1>83.6 ↓</td></tr><tr><td rowspan=1 colspan=2>84.7 ↓2.1</td><td rowspan=1 colspan=1>83.0 ↓2.8</td><td rowspan=1 colspan=1>83.9↓1.7</td><td rowspan=1 colspan=1>82.4 ↓</td><td rowspan=1 colspan=1>2.3</td><td rowspan=1 colspan=1>78.1 ↓2.3</td><td rowspan=1 colspan=1>77.3 ↓3.2</td><td rowspan=1 colspan=1>82.2 ↓</td><td rowspan=1 colspan=1>2.1</td><td rowspan=1 colspan=1>2.8</td></tr><tr><td rowspan=1 colspan=1>+VTI (Liu et al., 2025)</td><td rowspan=1 colspan=2>85.0↓1.8</td><td rowspan=1 colspan=1>85.3 ↓0.5</td><td rowspan=1 colspan=1>82.5 ↓3.1</td><td rowspan=1 colspan=1>83.3 ↓</td><td rowspan=1 colspan=1>1.4</td><td rowspan=1 colspan=1>74.5 ↓5.9</td><td rowspan=1 colspan=1>75.0 ↓5.5</td><td rowspan=1 colspan=1>80.7 ↓</td><td rowspan=1 colspan=1>3.6</td><td rowspan=1 colspan=1>81.2 ↓</td></tr><tr><td rowspan=1 colspan=3>+RVSD (Ours)            87.4 ↑0.6</td><td rowspan=1 colspan=1>86.2 ↑0.4</td><td rowspan=1 colspan=1>87.7 ↑2.1</td><td rowspan=1 colspan=1>84.5 ↓</td><td rowspan=1 colspan=1>0.2</td><td rowspan=1 colspan=1>82.8 ↑2.4</td><td rowspan=1 colspan=1>81.1 ↑0.6</td><td rowspan=1 colspan=1>86.0 ↑</td><td rowspan=1 colspan=1>1.7</td><td rowspan=1 colspan=1>83.9 ↑0.2</td></tr></table>

Table 5: Performance comparison on the POPE discrimination benchmark over the A-OKVQA dataset. The average column reports the mean accuracy and F1 scores across the random, popular, and adversarial splits. Green (•) and red (•) indicate improvements and degradations relative to the base model (sampling baseline), respectively, while (↑/↓) denotes the absolute change. Best results within each model block are highlighted in bold.

M3ID (Multi-Modal Mutual-Information Decoding) (Favero et al., 2024) reweights output logits using a mutual-information-based score between visual and textual modalities, aiming to enhance visual grounding during generation.

AvisC (Attentional Vision Calibration) (Woo et al., 2025) identifies high-attention “blind tokens” that dominate cross-modal attention and recalibrates predictions via dual forward passes with and without these tokens.

VTI (Visual Token Intervention) (Liu et al., 2025) steers latent representations in the large language model hidden space along a pre-computed direction associated with improved visual grounding.

## B.3 Models

We instantiate RVSD on three LVLMs drawn from two distinct architectural families. LLaVA-1.5- 7B (Liu et al., 2024b) adopts CLIP-ViT-L/14-336px as the visual encoder, a two-layer MLP as the cross-modal projector, and Vicuna-7B-v1.5 as the language backbone, producing 576 visual tokens per image. LLaVA-NEXT-7B (Liu et al., 2024c) extends LLaVA-1.5 with the AnyRes mechanism, which partitions the input image into up to four high-resolution sub-images, increasing the number of visual tokens to as many as 2,880 and substantially amplifying the burden on visual grounding. Qwen-VL (Bai et al., 2023) instead couples OpenCLIP-ViT-bigG with a single-layer positionaware cross-attention adapter that compresses the visual feature map into a fixed set of 256 visual tokens, and adopts Qwen-7B as the language backbone, differing from the LLaVA series in both the vision-language interface and the language backbone. Overall, the backbones cover standard and high-resolution settings as well as both MLPprojection and cross-attention resampler interfaces, and demonstrate that RVSD is model-agnostic.

## C Implementation Details

Hyperparameter Settings of RVSD. RVSD is implemented in PyTorch on top of the official LLaVA codebase and introduces no additional trainable parameters. Table 7 summarizes the default hyperparameter configuration used across all experiments unless otherwise specified. We adopt a unified pruning layer set ${ \mathcal L } _ { s } ~ = ~ 2 , 6 .$ , 15 for all three backbones, where the visual-token bank is progressively pruned at these layers. For LLaVA-1.5-7B, the sparse-set budget is set to k = 192, approximately one third of $N _ { v } = 5 7 6 .$ , resulting in a 67% reduction in visual tokens. For LLaVA-NEXT-7B, the budget is rescaled according to the image shape to maintain a consistent retention ratio under the AnyRes high-resolution setting. For Qwen-VL, whose cross-attention adapter emits only $N _ { v } = 2 5 6$ visual tokens, the budget starts from k = 192 and is progressively tightened to [133, 89, 52] at the three pruning layers, yielding a final retention ratio comparable to that of LLaVA-1.5. The number of major textual tokens is defined as $m = \lfloor | y < t | / 2 \rfloor$ , which selects the most recent half of the generated prefix for visual-saliency aggregation and ensures a length-adaptive allocation consistent with the Top-K formulation in the main paper. Temporal smoothing in Eq. (6) is disabled by setting $\eta = 0 , \mathrm { i . e . , } \tilde { s } ^ { t } = s ^ { t }$ . For SSVR, the entropy threshold $\gamma _ { : }$ , the intermediate-layer window $[ l _ { \mathrm { s } } , l _ { \mathrm { e } } ]$ the retrieval bound K, and the injection ratio α are tuned per backbone to accommodate differences in visual-token granularity across LLaVA-1.5 (single 24×24 grid), LLaVA-NEXT (multi-granularity AnyRes grids), and Qwen-VL (256 resampled tokens without an explicit spatial grid).

<table><tr><td rowspan="2">Models</td><td rowspan="2">Methods</td><td colspan="2">Random</td><td colspan="2">Popular</td><td colspan="2">Adversarial</td><td colspan="2">Average</td></tr><tr><td>Acc (%)↑</td><td>F1 (%)↑</td><td>Acc (%)↑</td><td>F1 (%)↑</td><td>Acc (%)↑</td><td>F1 (%)↑</td><td>Acc (%)↑</td><td>F1 (%)↑</td></tr><tr><td rowspan="5">LLaVA-1.5</td><td>Vanilla</td><td>81.6 ↑0.0</td><td>83.5 ↑0.0</td><td>73.1 ↑0.0</td><td>77.5 ↑0.0</td><td>68.0 ↑0.0</td><td>74.5 ↑0.0</td><td>74.2 ↑0.0</td><td>78.5 ↑0.0</td></tr><tr><td>+VCD (Leng et al., 2024)</td><td>82.2 ↑0.6</td><td>84.1 ↑0.6</td><td>71.5 ↓1.6</td><td>76.8 ↓0.7</td><td>67.6 ↓0.4</td><td>74.5 ↑0.0</td><td>73.8 ↓0.4</td><td>78.5 ↑0.0</td></tr><tr><td>+M3ID (Favero et al., 2024)</td><td>83.3 ↑1.7</td><td>84.5 ↑1.0</td><td>72.3 ↓0.8</td><td>77.1 ↓0.4</td><td>67.2 ↓0.8</td><td>74.0 ↓0.5</td><td>74.3 ↑0.1</td><td>78.5 ↑0.0</td></tr><tr><td>+AvisC (Woo et al., 2025)</td><td>79.0 ↓2.6</td><td>82.2 ↓1.3</td><td>67.4 ↓5.7</td><td>74.8 ↓2.7</td><td>64.1 ↓3.9</td><td>72.9 ↓1.6</td><td>70.2 ↓4.0</td><td>76.6↓1.9</td></tr><tr><td>+VTI (Liu et al., 2025)</td><td>79.8 ↓1.8</td><td>82.3 ↓1.2</td><td>73.5 ↑0.4</td><td>78.0 ↑0.5</td><td>68.0 ↑0.0</td><td>74.6 ↑0.1</td><td>73.8 ↓0.4</td><td>78.3 ↓0.2</td></tr><tr><td></td><td>+RVSD (Ours)</td><td>85.3 ↑3.7</td><td>83.8 ↑0.3</td><td>81.3 ↑8.2</td><td>80.3 ↑2.8</td><td>78.9 ↑11.</td><td>78.3 ↑3.8</td><td>81.8 ↑7.6</td><td>80.8 ↑2.3</td></tr><tr><td rowspan="5">LLaVA-NEXT</td><td>Vanilla</td><td>83.1 ↑0.0</td><td>82.5 ↑0.0</td><td>78.5 ↑0.0</td><td>78.5 ↑0.0</td><td>73.3 ↑0.0</td><td>74.5 ↑0.0</td><td>78.3 ↑0.0</td><td>78.5 ↑0.0</td></tr><tr><td>+VCD (Leng et al., 2024)</td><td>83.4 ↑0.3</td><td>83.4 ↑0.9</td><td>78.2 ↓0.3</td><td>78.6 ↑0.1</td><td>74.2 ↑0.9</td><td>75.6 ↑1.1</td><td>78.6 ↑0.3</td><td>79.2 ↑0.7</td></tr><tr><td>+M3ID (Favero et al., 2024)</td><td>84.3 ↑1.2</td><td>84.3 ↑1.8</td><td>78.8 ↑0.3</td><td>78.9 ↑0.4</td><td>74.2 ↑0.9</td><td>75.4 ↑0.9</td><td>79.1 ↑0.8</td><td>79.5 ↑1.0</td></tr><tr><td>+AvisC (Woo et al., 2025)</td><td>84.9 ↑1.8</td><td>83.5 ↑1.0</td><td>80.5 ↑2.0</td><td>79.5 ↑1.0</td><td>73.2 ↓0.1</td><td>74.3 ↓0.2</td><td>79.5 ↑1.2</td><td>79.1 ↑0.6</td></tr><tr><td>+VTI (Liu et al., 2025)</td><td>83.3 ↑0.2</td><td>82.6 ↑0.1</td><td>78.6 ↑0.1</td><td>78.2 ↓0.3</td><td>73.2 ↓0.1</td><td>74.3 ↓0.2</td><td>78.4 ↑0.1</td><td>78.4 ↓0.1</td></tr><tr><td></td><td>+RVSD (Ours)</td><td>88.2 ↑5.1</td><td>87.2 ↑4.7</td><td>84.9 ↑6.4</td><td>84.3 ↑5.8</td><td>81.6 ↑8.3</td><td>81.4 ↑6.9</td><td>84.9 ↑6.6</td><td>84.3 ↑5.8</td></tr><tr><td rowspan="5">Qwen-VL</td><td>Vanilla</td><td>81.3 ↑0.0</td><td>79.2 ↑0.0</td><td>75.9 ↑0.0</td><td>74.9 ↑0.0</td><td>75.5 ↑0.0</td><td>74.4 ↑0.0</td><td>77.6 ↑0.0</td><td>76.2 ↑0.0</td></tr><tr><td>+VCD (Leng et al., 2024)</td><td>82.0 ↑0.7</td><td>80.5 ↑1.3</td><td>75.9 ↑0.0</td><td>75.6 ↑0.7</td><td>76.7 ↑1.2</td><td>76.2 ↑1.8</td><td>78.2 ↑0.6</td><td>77.4 ↑1.2</td></tr><tr><td>+M3ID (Favero et al., 2024)</td><td>82.4 ↑1.1</td><td>79.7 ↑0.5</td><td>76.8 ↑0.9</td><td>77.0 ↑2.1</td><td>77.1 ↑1.6</td><td>76.7 ↑2.3</td><td>78.8 ↑1.2</td><td>77.8 ↑1.6</td></tr><tr><td>+AvisC (Woo et al., 2025)</td><td>80.5 ↓0.8</td><td>77.8 ↓1.4</td><td>74.2 ↓1.7</td><td>72.3 ↓2.6</td><td>75.5 ↑0.0</td><td>73.4↓1.0</td><td>76.7 ↓0.9</td><td>74.5 ↓1.7</td></tr><tr><td>+VTI (Liu et al., 2025)</td><td>82.0 ↑0.7</td><td>82.4 ↑3.2</td><td>76.5 ↑0.6</td><td>77.9 ↑3.0</td><td>71.0 ↓4.5</td><td>74.5 ↑0.1</td><td>76.5 ↓1.1</td><td>78.3 ↑2.1</td></tr><tr><td></td><td>+RVSD (Ours)</td><td>84.3 ↑3.0</td><td>81.7 ↑2.5</td><td>82.7 ↑6.8</td><td>80.3 ↑5.4</td><td>79.9 ↑4.4</td><td>77.8 ↑3.4</td><td>82.3 ↑4.7</td><td>79.9 ↑3.7</td></tr></table>

Table 6: Performance comparison on the POPE discrimination benchmark over the GQA dataset. The average column reports the mean accuracy and F1 scores across the random, popular, and adversarial splits. Green (•) and red (•) indicate improvements and degradations relative to the base model (sampling baseline), respectively, while (↑/↓) denotes the absolute change. Best results within each model block are highlighted in bold.
<table><tr><td>Symbols</td><td>Descriptions</td><td>LLaVA-1.5</td><td>LLaVA-NEXT</td><td>Qwen-VL</td></tr><tr><td>k</td><td>Sparse-set budget</td><td>192</td><td>Scaled by image_shape</td><td>192 (tier) → [133, 89, 52]</td></tr><tr><td>m</td><td>Major textual tokens</td><td> $\lfloor \left| y _ { < t } \right| / 2 \rfloor$ </td><td> $\lfloor \left| y _ { < t } \right| / 2 \rfloor$ </td><td> $\lfloor \left| y _ { < t } \right| / 2 \rfloor$ </td></tr><tr><td> $\tau$ </td><td>Saliency softmax temperature</td><td>1.0</td><td>1.0</td><td>1.0</td></tr><tr><td> $\eta$ </td><td>Temporal smoothing coefficient</td><td>0</td><td>0</td><td>0</td></tr><tr><td> $\mathcal { L } _ { s }$ </td><td>Pruning layer set</td><td>{2, 6, 15}</td><td>{2, 6, 15}</td><td>{2, 6, 15}</td></tr><tr><td> $\gamma$ </td><td>Entropy trigger threshold</td><td>0.75</td><td>0.5</td><td>0.75</td></tr><tr><td> $[ l _ { \mathrm { s } } , l _ { \mathrm { e } } ]$ </td><td>SSVR layer scanning window</td><td>[6,27]</td><td>[5,12]</td><td>[5,16]</td></tr><tr><td> $K$ </td><td>Retrieval top-K bound</td><td>128</td><td>7</td><td>57</td></tr><tr><td>α</td><td>Injection ratio</td><td>0.2</td><td>0.09</td><td>0.2</td></tr></table>

Table 7: Default hyperparameter settings of RVSD used across all experiments unless otherwise specified.

Decoding Protocol. Following (Leng et al., 2024; Liu et al., 2025), all methods, including RVSD and baselines, are evaluated under a unified decoding setup using nucleus sampling with temperature $T = 1 . 0 , \mathrm { t o p } { \cdot } p = 0 . 9$ , no repetition penalty, and batch size 1 for a fair comparison. The generation budget, i.e., the maximum number of new tokens, is benchmark-dependent: It is set to 2 for binary yes/no tasks in POPE and the MME hallucination subset, and to 1,024 for open-ended generation benchmarks including AMBER (generative), MM-Vet, and CHAIR. For the inference latency evaluation in Table 4, we follow the protocol specified in the table caption, setting the maximum number of new tokens to 32 with batch size 1.

![](images/aa0e13538f4891f21eac8917eddb0bde7812b99ab2b6cbc6307288f9b91a5610.jpg)  
Figure 6: Subtask-level ∆ heatmap on MME under nonbaseline $( \alpha , k _ { c } )$ configurations, where $\Delta = \mathrm { s c o r e } ( \mathrm { c f g } ) -$ score(base) is computed relative to the baseline setting $( \alpha =$ $0 . 2 0 , \dot { k } _ { c } = \dot { 1 } 2 8 )$

Implementation Details. We conduct all experiments on a single NVIDIA RTX A6000 GPU (48 GB) with NVIDIA driver 535.183.01 and CUDA 12.2 on Ubuntu 22.04. The software stack includes PyTorch 2.0 and Hugging Face Transformers 4.40.

## D Hyperparameter Sensitivity Analysis

Subtask-Level Sensitivity on MME. We first conduct a one-factor-at-a-time analysis around the baseline $( \alpha = 0 . 2 0 , k _ { c } = 1 2 8 )$ on MME with LLaVA-1.5, where α denotes the retrieval injection ratio and $k _ { c }$ the top-k retrieval size. As shown in Fig. $^ { 6 , }$ α is the primary driver of subtask-level variations: Small values of α (0.05/0.10) consistently improve OCR and translation by up to +7.5, whereas a larger value $( \alpha = 0 . 5 0 )$ leads to broad degradations across OCR, commonsense, counting, and color reasoning. In contrast, varying $k _ { c }$ within [32, 256] yields negligible changes across all subtasks. This analysis identifies $( \alpha = 0 . 1 0 , k _ { c }$ = 128) as a stable operating point and further reveals that the aggregate MME score can obscure task-specific regressions, particularly in reasoningintensive categories.

Hyperparameter Sensitivity. To complement the subtask-level analysis above, we further investigate the sensitivity of RVSD to its four key hyperparameters on the POPE-MSCOCO adversarial split with LLaVA-1.5, which is the most discriminative setting for hallucination evaluation. We perform a one-factor-at-a-time analysis, varying each hyperparameter while keeping the others fixed at their default values in Table 7, and the results are reported in Table 8.

<table><tr><td>Hyperparameters</td><td>Setting</td><td>Acc (%)↑</td><td>F1 (%) ↑</td></tr><tr><td rowspan="5">Injection ratio α</td><td>0.05</td><td>84.7</td><td>83.4</td></tr><tr><td>0.10</td><td>84.7</td><td>83.4</td></tr><tr><td>0.20</td><td>84.7</td><td>83.5</td></tr><tr><td>0.50</td><td>84.4</td><td>83.2</td></tr><tr><td>32</td><td>84.6</td><td>83.4</td></tr><tr><td rowspan="3">Retrieval bound K</td><td>64</td><td>84.6</td><td>83.4</td></tr><tr><td>128</td><td>84.7</td><td>83.5</td></tr><tr><td>256</td><td>84.6</td><td>83.4</td></tr><tr><td rowspan="3">Entropy threshold γ</td><td>0.3</td><td>84.7</td><td>83.4</td></tr><tr><td>0.5</td><td>84.7</td><td>83.4</td></tr><tr><td>0.7</td><td>84.6</td><td>83.4</td></tr><tr><td rowspan="4">Sparse budget k</td><td>96</td><td>83.0</td><td>81.1</td></tr><tr><td>192</td><td>84.7</td><td>83.5</td></tr><tr><td>288</td><td>85.0</td><td>84.1</td></tr><tr><td> $5 7 6 \ : ( = N _ { v } )$ </td><td>85.0</td><td>84.1</td></tr></table>

Table 8: Hyperparameter sensitivity analysis on the POPE-MSCOCO Adversarial split with LLaVA-1.5. The best result in each block is highlighted in bold.

Effect of Injection Ratio α. RVSD is robust to the injection ratio α in the practical range [0.05, 0.20], where Accuracy/F1 remains stable at 84.7/83.4-84.7/83.5. Increasing α to 0.50 overinjects retrieved evidence relative to the residualstream magnitude, leading to a mild degradation to 84.4/83.2. This behavior is consistent with the amplitude-alignment design in Eq. (11), where the retrieval branch is intended to re-anchor rather than overwrite the host representation.

Effect of Retrieval Bound K. Accuracy/F1 scores vary by less than 0.1 across the retrieval bound $K ~ \in ~ \{ 3 2 , 6 4 , 1 2 8 , 2 5 6 \}$ , indicating strong robustness of the softmax-weighted aggregation in Eq. (9). Irrelevant entries in the deferred memory bank receive near-zero weights and are effectively suppressed, such that increasing K neither introduces noise nor yields additional benefit.

Effect of Entropy Threshold γ. RVSD is stable across $\gamma \in { 0 . 3 , 0 . 5 , 0 . 7 }$ , with the accuracy score remaining at 84.7 for the two lower thresholds and decreasing by only 0.1 at $\gamma = 0 . 7$ . This suggests that, on the POPE adversarial split, the SSVR gate activates on a well-separated subset of uncertain steps, and moderate variations in γ do not materially affect which steps are re-anchored.

Effect of Sparse Budget k. The sparse budget k is the most influential among the four hyperparameters. A small budget $( k \ = \ 9 6$ , approximately one-sixth of $N _ { v } )$ removes too much finegrained visual evidence and reduces Accuracy/F1 to 83.0/81.1. Increasing k to 288 or 576 slightly improves performance to 85.0/84.1. The default setting $\begin{array} { l } { ( k ~ = ~ 1 9 2 ) } \end{array}$ achieves competitive Accuracy/F1 (84.7/83.5) while maintaining a substantially lower attention cost than $k = 5 7 6 .$ , and is therefore adopted as the efficiency–faithfulness trade-off point reported in Table 4. This trend empirically supports the claim in Section 4.2 that visual grounding in RVSD is primarily driven by semantics-directed selection rather than the sheer number of retained tokens.

## E Algorithmic Complexity

We analyze the asymptotic overhead introduced by RVSD over vanilla decoding. Let D denote the hidden dimension, $N _ { v }$ the number of visual tokens, t the current decoding step, and k the sparse-set budget. The dominant cost of vanilla decoding at step t arises from per-layer self-attention, with complexity $\mathcal { O } ( ( t + N _ { v } ) ^ { 2 } D )$ . In RVSD, this is reduced to ${ \mathcal { O } } ( ( t + k ) ^ { 2 } D )$ through the sparse set $S _ { t }$ , leading to computational savings when $k \ll N _ { v }$

The additional overhead of RVSD consists of three components. First, the textual saliency computation in Eq. (4) and the relevance scoring in Eq. (5) incur costs of $\mathcal { O } ( t )$ and $\mathcal { O } ( m N _ { v } \vert \mathcal { L } _ { s } \vert )$ , respectively. Both are linear and negligible compared with the quadratic attention term. Second, SSVR retrieval in Eq. (9) requires $\mathcal { O } ( | \mathcal { P } _ { t } | D )$ for similarity computation and $\mathcal { O } ( | \mathcal { P } _ { t } | \log K )$ for top-K selection, which remains dominated by selfattention. Third, the residual-stream adapter in Eq. (11) and the fusion step in Eq. (13) incur $\mathcal { O } ( K D )$ per triggered step. However, the entropybased gate in Eq. (8) activates retrieval only on a small subset of steps. Overall, the additional perstep cost is bounded by $\mathcal { O } ( N _ { v } D )$ and is negligible relative to attention. Empirically, Table 4 further shows that RVSD reduces per-token latency, maximum memory, TFLOPs, and TTFT compared with vanilla decoding, a profile not achieved by previous decoding-side methods for VH mitigation.

## F License of Artifacts

We use all datasets and models strictly for academic research in accordance with their original licenses and terms. MSCOCO is released under the Creative Commons Attribution 4.0 License, and POPE is released under the MIT License. AM-BER, MME, and MM-Vet are used under their respective official non-commercial academic research licenses. LLaVA-1.5 and LLaVA-NEXT are released under the Apache 2.0 License, while their LLaMA-2/Vicuna backbones are subject to the Meta LLaMA-2 community license. Qwen-VL is released under the Tongyi Qianwen License, which allows use for academic research. We do not redistribute any of these resources beyond the permissions granted by their licenses, and we will release our RVSD implementation under a permissive open-source license upon publication.

## G Case Study

To provide an intuitive understanding of how RVSD mitigates VHs during generation, we present qualitative case studies in Figs. 7 and 8. Compared with the vanilla decoder and representative sparse decoding baselines, RVSD produces captions that are more faithfully grounded in the visual content, successfully suppressing hallucinated objects, attributes, and spatial relations that are absent from the input image. As shown in Fig. 7, RVSD consistently mitigates object- and attributelevel hallucinations across diverse scenes, while Fig. 8 further demonstrates its effectiveness under more challenging long-form generation settings with denser visual content and finer-grained semantic dependencies. These cases collectively demonstrate that the semantics-directed token selection effectively preserves visually critical evidence, while the SSVR mechanism re-anchors decoding through on-demand cross-modal retrieval whenever visual grounding becomes insufficient.

## H Usage of AI Assistant

We used GPT-4o exclusively for the language polishing of this manuscript, including grammar correction and sentence-level rephrasing. All generated outputs were manually reviewed and edited to ensure technical accuracy. No part of the research ideation, method design, experimental analysis, or interpretation of the results involved AI assistance.

![](images/87982a1fe20e208fe39c1e2024501e335bfe9b48b7ab14c5b9d87b2b0da20e25.jpg)  
Question: Is this artwork displayed in kupferstichkabinett, gotha?

Ground Truth Answer: No

LLaVA-1.5 (Baseline): Yes

+VCD: Yes

+AVISC: Yes

+VTI: Yes

![](images/6b8f552959e50b9e440f3cab70a02e9aef08cb9be8aa669e1fb18cb93cf52aa0.jpg)

+RVSD (Ours): No

Question: Is this artwork created by wit, jacob de?

Ground Truth Answer: No

+VCD: Yes

LLaVA-1.5 (Baseline): No

+AVISC: Yes

+VTI: No

+RVSD (Ours): No

![](images/d78a56f75177f99d64b79c056b94c40fb29ac22aae7a95ab645f8088f0df9574.jpg)

![](images/b3c03ddb15721ccf28b6b3419ce5e898caaf5b6d90c937d5186e3df6c33f183e.jpg)  
Question: Is this a picture of Kartause Ittingen?

![](images/25c383988f6dea7f4e8ef829ac7bf7fcd79c91b492896679af78efdf934832d9.jpg)

Ground Truth Answer: No

![](images/03ef60556b45b957ee48c7ef81d07024d3863fc3ec404c5ce038add53b33be1d.jpg)  
Question: Is this a photo of Huxi Mosque?

Ground Truth Answer: No

Question: Is this movie originated from the country or region of usa?

LLaVA-1.5 (Baseline): Yes

+VTI: Yes

+VCD: Yes

+AVISC: Yes

Ground Truth Answer: No

LLaVA-1.5 (Baseline): Yes

+RVSD (Ours): No

Question: Is this picture captured in a place of swamp?

Ground Truth Answer: No

LLaVA-1.5 (Baseline): No

+VCD: No

+AVISC: No

+VTI: Yes

LLaVA-1.5 (Baseline): Yes

+RVSD (Ours) : No

![](images/6625116fb76b094a3565cc758e1383a036078a68223af69a541f5b874032fa77.jpg)  
Question: Is this artwork created by witte, emanuel de?

Ground Truth Answer: No

+VCD: Yes

+AVISC: Yes

+VCD: Yes

LLaVA-1.5 (Baseline): Yes

+AVISC: Yes

+VCD: Yes

+AVISC: Yes

+VTI: Yes

+VTI: Yes

+RVSD (Ours): No

![](images/b874806a4555d36778c78001a27bee741c26a5ce2830beaee424f3cd2ad5914b.jpg)

+RVSD (Ours): No

+VTI: No

Question: Is there a cow in the image?

![](images/e29e89d55dbab17e917bef6f4cf721b0b3288302cbca237b4896f44206a4d759.jpg)  
Question: Is there a bottle in the image?

Ground Truth Answer: No

+RVSD (Ours): No

Ground Truth Answer: No

![](images/41f9fd396103054ae6df4462f00a04a0bbe67c0ee141d653ac77edbfbd5ef9f0.jpg)

Question: Is there an adult in the image?

LLaVA-1.5 (baseline): Yes

Ground Truth Answer: No

+VCD: Yes

LLaVA-1.5 (baseline): Yes

+VCD: Yes

LLaVA-1.5 (baseline): Yes

+AVISC: Yes

+AVISC: Yes

+VCD: Yes

+AVISC: Yes

+VTI: Yes

+VTI: Yes

+VTI: Yes

+RVSD (Ours): No

+RVSD (Ours): No

+RVSD (Ours): No

Figure 7: Qualitative case studies of RVSD compared with the vanilla decoder and baseline methods. Hallucinated content is highlighted in red, while faithful content grounded in the input image is highlighted in green. RVSD consistently produces visually faithful descriptions across diverse scenes.

![](images/ce73cf8732d20b538dd1b2042f93e6d4d3426c3f90cf021ef0a2611ea81eab9c.jpg)  
Figure 8: Additional qualitative case studies of RVSD on more challenging long-form generation samples. Hallucinated content is highlighted in red, while faithful content grounded in the input image is highlighted in green. RVSD continues to deliver visually faithful generations even under denser scenes and finer-grained semantic dependencies, where baseline decoders frequently exhibit object-, attribute-, and relation-level hallucinations.