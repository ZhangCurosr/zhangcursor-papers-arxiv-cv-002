# MARS: What Retrieval Signals Are Hidden in Multimodal Large Language Models for Text-Video Retrieval?

Uicheol Jung, Juyoung Hong, Geuntaek Lim, Yukyung Choi

Sejong University, Seoul, Republic of Korea

{ucjung,ykchoi}@rcv.sejong.ac.kr

## Abstract

Text-video retrieval requires representations that can distinguish videos with similar scenes, actions, and temporal patterns. Recent multimodal large language models have been adapted as embedding models, but they often represent each input using a single token from the final layer. This can compress diverse video-text cues into a single vector and limit finegrained retrieval. To address this limitation, we propose MARS, a multi-layer and multi-slot embedding framework for textvideo retrieval. MARS constructs multiple adaptive representation slots by combining hidden states from diferent decoder layers, compares corresponding text and video slots, and aggregates their similarities for retrieval. To better handle confusing candidates, we further introduce a hard-negative-aware slot specialization objective that encourages the slots to capture discriminative matching cues. Experiments on four text-video retrieval benchmarks show that MARS achieves state-ofthe-art results in both direct similaritybased retrieval and reranking settings. Ablation studies and analyses demonstrate that multi-layer fusion, multiple slots, and hardnegative-aware slot specialization provide complementary gains. Code is available at https://github.com/sejong-rcv/MARS.

## 1 Introduction

Text-video retrieval aims to align sentences and videos in a shared representation space based on their semantic similarity (Miech et al., 2019; Dong et al., 2022; Gabeur et al., 2020; Bain et al., 2021). With the rapid growth of video data, this task has become important for retrieving relevant video content through natural language queries.

Recently, MLLMs have attracted increasing attention as general-purpose embedding models (Jiang et al., 2024a,b, 2025; Zhang et al., 2025b; Liu et al., 2025). Rather than generating text, these methods obtain embeddings from the hidden state of a specific token (e.g., the final or ⟨EOS⟩ token) given a prompted input. For example, E5-V adopts an in one word prompting strategy inspired by PromptEOL (Jiang et al., 2024a), such as <image> Summary above image in one word:, to obtain a unified multimodal representation (Jiang et al., 2024b). These studies show that MLLMs can be used as representation models for multimodal inputs.

However, adapting MLLM-based methods to text-video retrieval remains underexplored. While extracting a single token representation from the final layer is efective for generalpurpose tasks (Jiang et al., 2024b; Zhang et al., 2025b), it creates a severe information bottleneck for complex video content. Because videos inherently contain multi-granular and compositional cues (e.g., objects, actions, and temporal dynamics), retrieval embeddings must preserve fine-grained semantic information to diferentiate visually similar scenes. This limitation motivates us to rethink embedding extraction from two perspectives: leveraging distributed information across multiple decoder layers, and preserving complementary cues without collapsing them into a single vector.

First, although final-layer representations are widely used, recent studies indicate they are not always optimal for downstream embedding tasks (Skean et al., 2025; Tang and Yang, 2024). Useful embedding signals are often distributed across multiple decoder layers rather than concentrated at the end. Accordingly, text-video retrieval can benefit from aggregating this multi-layer evidence instead of relying solely on the final output.

Second, extracted cues should be preserved rather than collapsed into a single vector. Prior work on fine-grained retrieval demonstrates that global representations alone struggle to capture detailed matching signals like objects and actions (Chen et al., 2020a; Ma et al., 2022). Forcing these diverse cues into a single representation inevitably entangles them, obscuring critical semantic details and degrading performance on similar candidates. Therefore, to maintain fine-grained discriminability, these cues must be preserved and compared separately before computing the final retrieval.

To address these limitations, we propose MARS (Multi-layer Adaptive Representation Slots). Rather than extracting an embedding from a single layer, MARS constructs each slot by fusing hidden states from multiple decoder layers with slot-specific layer weights. This allows diferent slots to emphasize diferent layerwise evidence. Instead of compressing all information into one global vector, MARS preserves complementary cues through multiple adaptive representation slots and compares corresponding slots before aggregating their similarities. We further introduce a hard-negative-aware slot specialization loss and a diversity loss to encourage the slots to capture distinct matching signals. As a result, MARS can better distinguish candidates that difer in fine-grained semantic details.

To summarize, our contributions are as follows:

• We revisit embedding extraction for MLLM-based text-video retrieval and propose MARS, which constructs adaptive representation slots by fusing layer-wise representations.

• We validate MARS on four benchmarks, achieving state-of-the-art results in both direct retrieval and reranking settings.

• We provide in-depth analyses to elucidate how multi-layer evidence aggregation and complementary slot-wise matching contribute to the performance gains of MARS.

## 2 Related Work

## 2.1 Text-Video Retrieval

A major line of work extends CLIP-based representations to the video domain through frame aggregation, temporal modeling, and fine-grained alignment (Radford et al., 2021; Luo et al., 2022; Xue et al., 2023; Ma et al., 2022; Wu et al., 2023; Wang et al., 2023c; Shen et al., 2025; Jung et al., 2026). In parallel, video foundation models have improved text-video retrieval through large-scale pre-training (Li et al., 2023b; Wang et al., 2022, 2024b). However, these gains often rely on heavy crossmodal decoders or reranking (Li et al., 2023b; Wang et al., 2024b; Ko et al., 2025). In contrast, our work follows an eficient dual-encoder retrieval structure, where videos and texts are encoded independently and ranked directly by representation similarity without additional pairwise scoring.

## 2.2 MLLMs as Multimodal Embedders

Recent studies have adapted MLLMs into multimodal embedders through prompt-based extraction (Jiang et al., 2024b; Liu et al., 2025) and contrastive or data-centric training (Jiang et al., 2025; Lin et al., 2025; Zhang et al., 2025b; Chen et al., 2025; Zhou et al., 2025; Meng et al., 2026). Despite methodological diferences, these approaches predominantly derive the embedding from a single token (e.g., the final or ⟨EOS⟩ token). This single-token extraction fails to fully exploit the multi-layered representations within MLLMs, inherently limiting their ability to capture fine-grained cross-modal correspondences. Instead of collapsing information into a single token from a single layer, our work constructs multiple layer-fused adaptive representation slots, preserving detailed semantic cues for accurate slot-wise matching.

## 3 Method

In this section, we introduce MARS, as illustrated in Figure 1. We first formulate slotbased text-video retrieval setting in Sec. 3.1. Next, Sec. 3.2 presents the main components of MARS, including representation token prompting, layer-fused slot construction, slot-wise matching, and hard-negative-aware slot specialization. Finally, Sec. 3.3 defines the overall training objective.

## 3.1 Problem Formulation

Given a batch of paired texts and videos B = $\{ ( t _ { i } , v _ { i } ) \} _ { i = 1 } ^ { B }$ , text-video retrieval aims to assign higher similarity scores to matched text-video pairs than to mismatched pairs. We denote the similarity between text $t _ { i }$ and video $v _ { j }$ by $S [ i , j ]$ , forming a score matrix $S \in \mathbb { R } ^ { B \times \check { B } }$ where rows correspond to texts and columns correspond to videos. For a unified notation, we let $a \in \{ t , v \}$ denote the query side, with $S ^ { t } = S$ for text-to-video retrieval and $S ^ { v } = S ^ { \top }$ for video-to-text retrieval.

![](images/eef2b76bd5be929cc45e9f362cc53e69efc407ec369ae57fe601ff0bf19b3197.jpg)  
Figure 1: Overview of MARS. (a) The overall framework constructs multiple representation slots by appending adaptive representation tokens to the prompt and fusing their hidden states across decoder layers. (b) The resulting slots are matched between text and video to compute the retrieval score. (c) A diversity objective reduces redundancy among slots from the same input. (d) Hard-negative-aware slot specialization further strengthens discrimination against confusing negatives.

## 3.2 MARS (Multi-layer Adaptive Representation Slots)

By constructing multiple adaptive representation slots from hidden states across decoder layers, MARS preserves complementary finegrained cues that may be weakened when each input is compressed into a single global vector.

Representation token prompting. For each input, we construct a chat-formatted prompt and place M learnable adaptive representation tokens in the model response field, where each token corresponds to one adaptive representation slot. The prompt template is illustrated as follows:

system: You are a helpful assistant.   
user: {caption or video input}   
Represent this {text|video} in detail.   
assistant: <slot 1> <slot 2> · · · <slot M>

The adaptive representation tokens are shared across text and video inputs. We denote them as $\{ \mathbf { e } _ { m } ^ { \mathrm { s l o t } } \} _ { m = 1 } ^ { M }$ , where $\mathbf { e } _ { m } ^ { \mathrm { s l o t } } \in \mathbb { R } ^ { D }$

Since the MLLM follows causal attention, we use the hidden states immediately preceding the adaptive representation tokens as slot representations. Let $\mathbf { h } _ { i , m , t } ^ { ( n ) } , \mathbf { h } _ { j , m , v } ^ { ( n ) } \in \mathbb { R } ^ { D }$ denote the hidden states from the n-th decoder layer for the m-th slot of text sample i and video sample $j ,$ respectively.

Multi-layer adaptive slot construction. To construct each slot from hidden states across multiple decoder layers, we apply slot-wise weighted layer fusion. For the m-th slot, we use a learnable vector ${ \mathbf w } _ { m } \in \mathbb { R } ^ { N }$ to assign weights to diferent decoder layers. The normalized layer weight $\alpha _ { m , n }$ is computed as:

$$
\alpha _ { m , n } = \frac { \exp ( w _ { m , n } ) } { \sum _ { n ^ { \prime } = 1 } ^ { N } \exp ( w _ { m , n ^ { \prime } } ) } .\tag{1}
$$

The hidden states from diferent layers are then fused to obtain the text and video slot embeddings:

$$
\mathbf { z } _ { i , m } ^ { t } = \sum _ { n = 1 } ^ { N } \alpha _ { m , n } \mathbf { h } _ { i , m , t } ^ { ( n ) } , \quad \mathbf { z } _ { j , m } ^ { v } = \sum _ { n = 1 } ^ { N } \alpha _ { m , n } \mathbf { h } _ { j , m , v } ^ { ( n ) } .\tag{2}
$$

We apply L<sub>2</sub>-normalization to each slot embedding, denoted by zˆ.

Slot-wise matching. After constructing the normalized text and video slot embeddings, we compare them in a slot-aligned manner. For the m-th slot, we compute the slot-wise cosine similarity matrix:

$$
S _ { m } [ i , j ] = \left. \hat { \mathbf { z } } _ { i , m } ^ { t } , \hat { \mathbf { z } } _ { j , m } ^ { v } \right. .
$$

Applying the same notation to each slot-wise score matrix, we define $S _ { m } ^ { t } = S _ { m }$ and $S _ { m } ^ { v } = S _ { m } ^ { \top }$ We then aggregate the slot-wise similarities across M slots to obtain the final similarity score. We use uniform aggregation, which is simple and performs consistently across benchmarks, as shown in Appendix C. Accordingly, with slot weights $\beta _ { m } = 1 / M$ , the aggregated score is defined as:

$$
S [ i , j ] = \sum _ { m = 1 } ^ { M } \beta _ { m } S _ { m } [ i , j ] .\tag{3}
$$

Hard-negative-aware slot specialization. Hard negatives are commonly used to strengthen cross-modal alignment by providing confusing mismatched samples (Chen et al., 2020b; Li et al., 2021). While prior methods mainly use them as pair-level training samples, we leverage them as slot-level signals for specialization. We therefore introduce a hard-negativeaware slot specialization objective that encourages the most discriminative slot to separate the positive pair from the confusing negative. For each query side $a \in \{ t , v \}$ , we select the hardest negative using the corresponding score matrix:

$$
j _ { a } ^ { * } ( i ) = \arg \operatorname* { m a x } _ { j \neq i } S ^ { a } [ i , j ] .\tag{4}
$$

For each selected hard negative, we define the slot-wise positive-negative gap as:

$$
\Delta _ { m } ^ { a } ( i ) = S _ { m } ^ { a } [ i , i ] - S _ { m } ^ { a } [ i , j _ { a } ^ { * } ( i ) ] .\tag{5}
$$

Here, the slot-wise gap measures how much the positive pair is separated from the selected hard negative at slot m. We then take the largest slot-wise gap across slots:

$$
\mathrm { g a p } ^ { a } ( i ) = \operatorname* { m a x } _ { 1 \leq m \leq M } \Delta _ { m } ^ { a } ( i ) .\tag{6}
$$

This formulation selects the most discriminative slot for each hard negative; since the

selected slot can vary across negatives, diferent slots are encouraged to specialize. The queryside hinge loss is defined as:

$$
\mathcal { L } _ { \mathrm { h n } } ^ { a } = \frac { 1 } { B } \sum _ { i = 1 } ^ { B } \operatorname* { m a x } ( 0 , \delta - \mathrm { g a p } ^ { a } ( i ) ) ,\tag{7}
$$

where δ is the margin. When the gap is smaller than $\delta ,$ this term promotes a larger separation between the positive pair and the selected hard negative. The final hard-negative-aware slot specialization loss is obtained by averaging over the text and video query sides:

$$
\mathcal { L } _ { \mathrm { h n } } = \frac { 1 } { 2 } \sum _ { a \in \{ t , v \} } \mathcal { L } _ { \mathrm { h n } } ^ { a } .\tag{8}
$$

## 3.3 Training Objective

The model is trained with three objectives: symmetric contrastive alignment, slot diversity regularization, and hard-negative-aware slot specialization. For each side $a \ \in \ \{ t , v \}$ , the contrastive loss is:

$$
\mathcal { L } _ { \mathrm { N C E } } ^ { a } = - \frac { 1 } { B } \sum _ { i = 1 } ^ { B } \log \frac { \exp ( \gamma S ^ { a } [ i , i ] ) } { \sum _ { j = 1 } ^ { B } \exp ( \gamma S ^ { a } [ i , j ] ) } .\tag{9}
$$

The symmetric InfoNCE loss is defined as:

$$
\mathcal { L } _ { \mathrm { N C E } } = \frac { 1 } { 2 } \sum _ { a \in \{ t , v \} } \mathcal { L } _ { \mathrm { N C E } } ^ { a } ,\tag{10}
$$

where $\gamma$ is the scaling factor. Since multiple slots are extracted from the same input, diferent slots can become redundant. We therefore apply a squared cosine similarity regularizer within each side:

$$
\mathcal { L } _ { \mathrm { d i v } } ^ { a } = \frac { 1 } { B M ( M - 1 ) } \sum _ { i = 1 } ^ { B } \sum _ { m \neq m ^ { \prime } } \left[ ( \hat { \mathbf { z } } _ { i , m } ^ { a } ) ^ { \top } \hat { \mathbf { z } } _ { i , m ^ { \prime } } ^ { a } \right] ^ { 2 } .\tag{11}
$$

The total diversity loss is defined as:

$$
\mathcal { L } _ { \mathrm { d i v } } = \frac { 1 } { 2 } \sum _ { a \in \{ t , v \} } \mathcal { L } _ { \mathrm { d i v } } ^ { a } .\tag{12}
$$

This term penalizes high similarity between slots from the same input and encourages the slots to encode distinct retrieval-relevant information.

Together with the hard-negative-aware slot specialization loss defined above, the final training objective is:

$$
\mathcal { L } = \mathcal { L } _ { \mathrm { N C E } } + \lambda _ { \mathrm { d i v } } \mathcal { L } _ { \mathrm { d i v } } + \lambda _ { \mathrm { h n } } \mathcal { L } _ { \mathrm { h n } } ,\tag{13}
$$

where $\lambda _ { \mathrm { d i v } }$ and $\lambda _ { \mathrm { h n } }$ are loss weights.

Table 1: Comparison with representative text-video retrieval methods on four benchmarks. T2V and V2T denote text-to-video and video-to-text retrieval, respectively. mR@1 denotes the mean R@1 across the four datasets. <sup>†</sup> denotes our reproduced InternVideo2-1B result. <sup>∗</sup> indicates the use of the DSL post-processing operation. MARS-R denotes MARS with reranking.
<table><tr><td rowspan="3">Method</td><td colspan="2">DiDeMo</td><td colspan="2"></td><td colspan="2">ActivityNet</td><td colspan="2">LSMDC</td><td colspan="2">MSR-VTT</td><td colspan="2">Avg.</td></tr><tr><td>T2V</td><td></td><td>V2T</td><td>T2V</td><td>V2T</td><td>T2V</td><td>V2T</td><td></td><td>T2V</td><td>V2T</td><td>T2V</td><td>V2T</td></tr><tr><td>|R@1 R@5 R@10|R@1 R@5 R@10|R@1 R@5 R@10|R@1 R@5 R@10|R@1 R@5 R@10|R@1 R@5 R@10|R@1 R@5 R@10|R@1 R@5 R@10|mR@1 mR@1</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="9">Direct similarity-based retrieval</td><td></td><td></td><td></td></tr><tr><td>CLIP4Clip (Luo et al., 2022)</td><td>42.8 68.5 79.2</td><td>42.5 70.6 80.2</td><td>40.5 72.4</td><td>83.4 42.6</td><td>73.4 85.6</td><td>21.6 41.8 49.8</td><td>20.9 40.7 49.1</td><td>44.5 71.4</td><td>81.6 43.1</td><td>70.5 81.2</td><td>37.4</td><td>37.3</td></tr><tr><td>CLIP-ViP (Xue et al., 2023)</td><td>48.6 77.1 84.4</td><td></td><td>51.1</td><td>78.4 88.3</td><td></td><td>25.6 45.3 54.4</td><td></td><td>50.1</td><td>74.8 84.6</td><td></td><td>43.9</td><td></td></tr><tr><td>ViCLIP (Wang et al., 2023b)</td><td>49.4</td><td>50.2</td><td>49.8</td><td></td><td>48.1</td><td>33.0</td><td>32.5</td><td>52.5</td><td></td><td>51.8</td><td>46.2</td><td>45.6</td></tr><tr><td>Video-ColBERT (Reddy et al., 2025) InternVideo (Wang et al., 2022)</td><td>48.2 75.1 83.7 82.4 88.9</td><td></td><td>45.5 62.2</td><td>74.6 85.5 93.2</td><td>62.8 86.2 93.3</td><td>34.0 53.7</td><td>34.9 54.6</td><td>48.1 63.1 55.2</td><td>74.9 83.9 79.6 87.5</td><td>57.9 79.2 86.4</td><td>52.3</td><td>53.7</td></tr><tr><td>MARS (Ours)</td><td>57.9 79.7</td><td>59.1 81.8</td><td>89.0</td><td>85.9</td><td>92.6</td><td></td><td>62.9 78.8 50.0</td><td>79.0 61.6</td><td>83.6 89.1</td><td>59.0 85.0</td><td></td><td></td></tr><tr><td>MARS* (Ours)</td><td>92.3 95.5 84.6 94.6 97.0</td><td>75.7 92.4 84.2 94.4</td><td>95.4 75.7 97.1 82.1</td><td>93.0 96.8 95.1 98.0</td><td>73.2 97.0 82.1 95.6 98.1</td><td>51.0 70.9 54.4 73.9</td><td>71.5 82.0 54.9 74.4</td><td>81.4 65.5</td><td>85.8 91.1</td><td>66.8 86.7 92.7</td><td>91.0 67.0 71.6</td><td>64.5 72.0</td></tr><tr><td colspan="9">Cross-modal matching and reranking</td><td></td><td></td><td></td><td></td></tr><tr><td>UMT (Li et al., 2023b)</td><td></td><td></td><td></td><td></td><td></td><td></td><td>73.0</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>InternVideo2 1B†* (Wang et al., 2024b)</td><td>70.4 90.1 93.5 72.8 90.3 93.9</td><td>67.9 88.6 69.1 89.1</td><td>93.0 66.8 93.7 66.2</td><td>89.1 94.9 88.5 94.3</td><td>64.4 89.1 94.8 62.8 86.7 93.5</td><td>43.0 65.5 42.7 64.7</td><td>41.4 64.3 72.9 42.7 63.7</td><td>71.5 58.8 71.3 58.7</td><td>81.0 87.1 80.2 87.0</td><td>58.6 81.6 86.5 56.8 79.3 85.6</td><td>59.8 60.1</td><td>58.1 57.8</td></tr><tr><td>InternVideo2 6B* (Wang et al., 2024b)</td><td>74.2</td><td>71.9</td><td>74.1</td><td></td><td>68.7</td><td>46.4</td><td>46.7</td><td>62.8</td><td></td><td>60.2</td><td>64.4</td><td>61.9</td></tr><tr><td>BLiM (Ko et al., 2025)</td><td>86.4 95.6 96.4</td><td>82.8 95.6</td><td>96.4 81.0</td><td>94.2 96.6</td><td>74.4 92.6 96.2</td><td>55.7 73.1</td><td>78.2 49.1 71.0</td><td>77.1 64.7</td><td>83.9 88.2</td><td>62.2 82.7 87.0</td><td>72.0</td><td>67.1</td></tr><tr><td>MARS-R (Ours)</td><td>87.6 96.1 96.8</td><td>82.2 94.9</td><td>96.5 83.5 95.4</td><td>97.9</td><td>77.9 94.3</td><td>97.6 56.1 74.1</td><td>80.9 50.0 71.5</td><td>79.0</td><td>|65.8 84.8 90.0</td><td>61.7 85.1</td><td>91.1 73.2</td><td>67.9</td></tr></table>

Table 2: Component ablation. LF and Slots denote multi-layer fusion and adaptive representation slots.  
Table 3: Efect of the number of adaptive representation slots.
<table><tr><td colspan="4">Components</td><td colspan="2">DiDeMo</td><td colspan="2">ActivityNet</td><td colspan="2">LSMDC</td><td colspan="2">MSR-VTT</td></tr><tr><td>LF</td><td>Slots</td><td> ${ \mathcal { L } } _ { \mathrm { d i v } }$ </td><td> ${ \mathcal { L } } _ { \mathrm { h n } }$ </td><td>T2V</td><td>V2T</td><td>T2V</td><td>V2T</td><td>T2V</td><td>V2T</td><td>T2V</td><td>V2T</td></tr><tr><td></td><td></td><td></td><td></td><td>72.9</td><td>71.5</td><td>69.5</td><td>66.0</td><td>47.5</td><td>47.1</td><td>58.3</td><td>56.7</td></tr><tr><td>√</td><td></td><td></td><td></td><td>76.6</td><td>74.9</td><td>72.7</td><td>70.0</td><td>47.9</td><td>46.6</td><td>59.8</td><td>57.9</td></tr><tr><td>V</td><td>√</td><td></td><td></td><td>77.4</td><td>75.0</td><td>75.6</td><td>73.6</td><td>49.9</td><td>50.2</td><td>60.8</td><td>57.9</td></tr><tr><td>√</td><td>√</td><td>√</td><td></td><td>77.5</td><td>75.0</td><td>75.7</td><td>73.2</td><td>50.9</td><td>49.7</td><td>61.1</td><td>59.0</td></tr><tr><td></td><td></td><td></td><td></td><td>79.7 75.7</td><td></td><td>75.7</td><td>73.2</td><td>51.0</td><td>50.0</td><td>61.6</td><td>59.0</td></tr></table>

<table><tr><td>Slots M</td><td>T2V</td><td>V2T</td><td>Avg.</td></tr><tr><td>2</td><td>76.5</td><td>74.8</td><td>75.6</td></tr><tr><td>3</td><td>78.8</td><td>73.9</td><td>76.3</td></tr><tr><td>4</td><td>79.7</td><td>75.7</td><td>77.7</td></tr><tr><td>5</td><td>79.9</td><td>74.9</td><td>77.4</td></tr><tr><td>6</td><td>77.3</td><td>74.6</td><td>75.9</td></tr></table>

## 4 Experiments

## 4.1 Experimental Setup

Datasets and metrics. We use DiDeMo, ActivityNet, LSMDC, and MSR-VTT (Hendricks et al., 2017; Krishna et al., 2017; Rohrbach et al., 2017; Xu et al., 2016), covering diverse domains, video lengths, and caption styles. For each benchmark, we report Recall@K (R@1, R@5, and R@10) for text-tovideo (T2V) and video-to-text (V2T) retrieval, along with mR@1 across the four benchmarks. Dataset statistics and preprocessing details are provided in Appendix A.

Implementation details. MARS uses VideoChat-Flash-Qwen2-7B (Li et al., 2025) as the MLLM, containing a UMT-L vision encoder (Li et al., 2023b), a linear projection, and a 28-layer Qwen2 (Yang et al., 2024). Pretrained weights remain frozen during training. We only update adaptive representation tokens, slot-wise layer-fusion weights, and LoRA (Hu et al., 2022) parameters (applied to the projection and Qwen2). Unless specified, we set M = 4 for DiDeMo, ActivityNet, and LSMDC, and M = 3 for MSR-VTT, with $\lambda _ { \mathrm { d i v } } ~ = ~ 0 . 1 , ~ \lambda _ { \mathrm { h n } } ~ = ~ 0 . 0 5$ , and δ = 0.04. Additional details are provided in Appendix B.

## 4.2 Main Results

The compared methods fall into two retrieval protocols: direct similarity-based retrieval and cross-modal matching or reranking. Direct similarity methods encode videos and texts independently, and rank candidates by representation similarity. In contrast, cross-modal matching and reranking methods retrieve candidates via embedding similarity and refine scores through additional cross-modal interaction or candidate-level reranking. Specifically, MARS-R first retrieves candidates using MARS and then reranks them with BLiM.

As shown in Table 1, MARS achieves the strongest performance among direct similaritybased retrieval methods across four benchmarks. Without cross-modal reranking, MARS outperforms InternVideo2-6B<sup>∗</sup>, which relies on cross-modal matching, while retaining a direct similarity-based retrieval structure, achieving 67.0 T2V mR@1 and 64.5 V2T mR@1. This suggests that multi-layer evidence aggregation and complementary slot-wise matching produce stronger retrieval embeddings for direct matching. Furthermore, with the same DSL (Cheng et al., 2021) post-processing, which calibrates retrieval scores using dual softmax normalization, MARS<sup>∗</sup> improves to 71.6 T2V mR@1 and 72.0 V2T mR@1, indicating that the score distribution produced by MARS also benefits from score calibration. When candidate-level reranking is applied, MARS-R achieves the best T2V mR@1 of 73.2 and improves over BLiM in both average T2V and V2T performance.

![](images/c389eae0ff0981ed7e14a8aae2c42f80ad3c8c97012b233aa8c6ccec71aa13a6.jpg)  
Figure 2: Layer-wise retrieval performance. (a) evaluates the final-layer baseline across decoder layers. (b) reports the multi-layer fusion model, where bars show the learned layer-fusion weights. The dashed line marks the final layer.

![](images/54ca67afebba9a58fd2a9f991847766b1dc32930e8159f1776c8cd02d67eae95.jpg)  
Figure 3: Slot-level retrieval analysis across benchmarks. Uniform aggregation outperforms each individual slot, and slot-exclusive rank-1 cases indicate complementary retrieval behavior among slots.

## 4.3 Ablation Studies

We conduct ablations across four benchmarks to isolate each component of MARS. Tables 2 and 3 report R@1 for T2V and V2T retrieval. Table 2 starts from a final-layer single-slot baseline and progressively adds layer fusion, representation slots, slot diversity regularization, and hard-negative-aware slot specialization. Multi-layer fusion: This yields the largest individual gain, increasing the average R@1 across the eight T2V/V2T settings from 61.2 to 63.3. This supports our motivation that retrieval cues are distributed across decoder depths rather than concentrated in the final layer. Adaptive representation slots: Adding multiple slots further raises the average R@1 to 65.1, showing that they preserve complementary matching information that may be overly compressed in a single representation. Slot diversity regularization: This provides a modest additional gain, suggesting that reducing redundancy among slots helps maintain distinct retrieval cues. Hard-negative-aware slot specialization: This final component improves the full model to the best average R@1 of

![](images/06929c1e5f767760a156ac6f19eac46a591f59e7983b5750a285f7bcaa3ff5e5.jpg)

![](images/123b58ccbc160552db08406eee678bdc333c90a281ebe13f655827004c0dadbe.jpg)

![](images/38b0b41798e40ee9a59d1960907f2ebd58d01706a375277838898be32d66a172.jpg)

![](images/488223f3c8691eb1310c619e677791bd79845a3eeb2d57913590b5549302b537.jpg)  
Pairwise Slot Cosine Similarity

![](images/949621279fb3b98dd586bdab51e069eb8397c5f40e6caa0594259d49c297988a.jpg)

![](images/b415ee0bd9b964d36a6b9394b716f3e2034114f2695fe712ecd3cbfea2afa597.jpg)

![](images/7da2cd50e07ef7dbf053297b1f19d8d30bbfd3d3cb1ab29970848c577c4ef6bb.jpg)

![](images/1e564253177166bd5c10a1e8b38b2d09e6f1b895b017dc3ead7fc02ce994ab67.jpg)  
Figure 4: Diversity analysis of adaptive representation slots. Pairwise top-1 agreement and slot embedding cosine similarity show that the learned slots make diferent retrieval decisions and remain distinct in the embedding space.

65.7, with particularly strong gains on DiDeMo and consistent improvements in T2V retrieval across all four benchmarks. Together, these results show that the proposed components provide complementary gains over the final-layer single-slot baseline.

We further vary the number of slots M in Table 3. The average R@1 peaks at 77.7 with M = 4 (up from 75.6 at M = 2), but drops with more slots. This indicates that a moderate number of slots is optimal, whereas excessive slots may introduce redundant or less discriminative representations.

## 4.4 Dissecting Representation Slots

Beyond performance, we analyze why representation slots improve text-video retrieval. We probe three properties that underlie this improvement: 1) layer-wise evidence: whether retrieval-relevant cues are distributed across decoder layers rather than concentrated in the final layer; 2) slot complementarity: whether diferent slots capture complementary matching cues; and 3) slot distinctness: whether the learned slots remain distinct in retrieval behavior and embedding space.

Q1. Where is retrieval evidence encoded across decoder layers? Figure 2 compares single-layer retrieval performance across decoder layers between the final-layer baseline and the multi-layer fusion model. In the finallayer baseline, R@1 increases toward the final decoder layer, where the contrastive objective is directly applied. However, this pattern changes when multi-layer fusion is trained. The strongest single-layer performance is no longer limited to the final layer; instead, several upper decoder layers achieve comparable R@1.

Interestingly, the learned fusion weights do not simply follow the single-layer R@1 curve. Some layers with lower individual R@1 still receive non-negligible weights, suggesting that they provide information that is useful when combined with other layers. This indicates that retrieval evidence is distributed across decoder layers, supporting our design choice of constructing slot embeddings from layer-fused representations rather than relying only on the final decoder layer.

Q2. Do multiple slots provide complementary matching cues? We next examine whether the learned slots capture complementary retrieval evidence. Figure 3 compares the single-slot T2V performance of each slot with uniform aggregation. Across all benchmarks, aggregation significantly outperforms the strongest individual slot (R@1 gains: +12.3 on DiDeMo, +10.2 ActivityNet, +11.5 LSMDC, +9.6 MSR-VTT), indicating the final score benefits from combined signals rather than a single dominant slot. This indicates that the final retrieval score is not dominated by a single slot, but benefits from combining multiple slot-wise matching signals.

Table 4: Computational cost comparison across four benchmarks. Ofline cost is measured per video, while online cost is measured per query for each benchmark. Total computation is reported in million GFLOPs (M GFLOPs), and inference time is measured on a single NVIDIA A100 GPU.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Offline GF/video</td><td colspan="4">Online GF/query</td><td rowspan="2">Total (M GFLOPs)</td><td rowspan="2">Time</td><td rowspan="2">mR@1</td></tr><tr><td>DiDeMo</td><td>ActivityNet</td><td>LSMDC</td><td>MSR-VTT</td></tr><tr><td>UMT</td><td>267.8</td><td>1,740.8</td><td>2,683.4</td><td>2,089.2</td><td>1,395.2</td><td>20.5</td><td>2,351.8s</td><td>59.8</td></tr><tr><td>InternVideo2-1B</td><td>2,542.6</td><td>4,576.1</td><td>5,661.3</td><td>4,977.5</td><td>4,177.4</td><td>61.7</td><td>8,717.4s</td><td>60.1</td></tr><tr><td>InternVideo2-6B</td><td>13,384.1</td><td>2,857.2</td><td>3,942.5</td><td>3,258.7</td><td>2,458.5</td><td>134.0</td><td>10,424.3s</td><td>64.4</td></tr><tr><td>MARS (Ours)</td><td>5,657.7</td><td>322.3</td><td>1,031.8</td><td>659.3</td><td>274.3</td><td>51.1</td><td>3,626.1s</td><td>67.0</td></tr></table>

Table 5: Generalization of MARS across MLLM backbones and model scales. Each entry reports T2V / V2T R@1. † denotes the backbone equipped with MARS. ∆ denotes the average absolute R@1 improvement over the corresponding base model across both retrieval directions and all four benchmarks.
<table><tr><td>Model</td><td>DiDeMo</td><td>ActivityNet</td><td>LSMDC</td><td></td><td>MSR-VTT</td><td>Avg.</td><td></td><td>∆</td></tr><tr><td>VideoLLaMA3-7B</td><td>61.0 56.8</td><td>60.4 58.6</td><td>38.8</td><td>36.9</td><td>54.1 53.5</td><td>53.6</td><td>51.5</td><td></td></tr><tr><td>VideoLLaMA3-7B† (Ours)</td><td>63.8 61.1</td><td>66.4 64.7</td><td>39.6</td><td>38.4</td><td>56.1 54.3</td><td>56.5</td><td>54.6</td><td>+3.0</td></tr><tr><td>Qwen2-VL-7B</td><td>57.7 /56.8</td><td>53.3 52.4</td><td>34.3</td><td>32.0</td><td>54.8 52.8</td><td>50.0</td><td>/48.5</td><td></td></tr><tr><td>Qwen2-VL-7B† (Ours)</td><td>62.1 60.8</td><td>58.9 57.7</td><td>36.1</td><td>32.3</td><td>56.3 54.7</td><td>53.4</td><td>51.4</td><td>+3.1</td></tr><tr><td>Qwen2-VL-2B</td><td>54.3 52.9</td><td>48.5 46.6</td><td>30.5</td><td>28.7</td><td>52.3 48.1</td><td>46.4</td><td>/44.1</td><td></td></tr><tr><td>Qwen2-VL-2B† (Ours)</td><td>57.1 56.2</td><td>51.9 50.5</td><td>31.4</td><td>28.9</td><td>53.3</td><td>50.6 48.4</td><td>46.6</td><td>+2.3</td></tr></table>

To examine whether this aggregation gain reflects complementary successes across slots, we count slot-exclusive rank-1 cases, where only one slot retrieves the correct video at rank one. Such cases appear consistently across benchmarks (123 on DiDeMo, 785 ActivityNet, 191 LSMDC, 209 MSR-VTT). This shows that diferent slots can recover diferent correct matches rather than simply duplicating the strongest slot. Therefore, a slot should not be judged only by its standalone accuracy; even a weaker slot can contribute by resolving retrieval cases that other slots miss.

Q3. Do learned slots form distinct representations? Figure 4 examines whether the learned slots collapse into similar representations. The mean of-diagonal agreement remains well below full agreement across benchmarks (0.537 on DiDeMo, 0.429 ActivityNet, 0.206 LSMDC, 0.419 MSR-VTT), showing that diferent slots often produce diferent topranked candidates rather than repeatedly retrieving the same video.

The bottom row reports pairwise cosine similarity between video-side slot embeddings. The of-diagonal similarities remain generally small, indicating that the slots are also separated in the embedding space. Together, the retrievallevel agreement and embedding-space analyses show that the learned slots do not collapse into redundant copies, but maintain distinct representations that support multi-slot retrieval.

## 4.5 Computational Cost Analysis

Since MARS builds on an MLLM, inference cost is an important practical consideration. We analyze the computational cost across four benchmarks by separating ofline video encoding from online retrieval-time computation, as reported in Table 4. The ofline cost comes from video encoding, which can be precomputed and stored, whereas the online cost is incurred per text query for text encoding and similarity computation.

After video representations are precomputed, MARS follows a dual-encoder retrieval structure: it encodes each text query once and computes similarity scores against stored video representations. By contrast, UMT and InternVideo2 include cross-modal matching components in their retrieval pipelines, resulting in substantially higher online computation. MARS achieves the lowest online cost across all four benchmarks, while obtaining the highest average T2V R@1 of 67.0. Compared with InternVideo2, MARS also requires less total computation and evaluation time despite achieving stronger retrieval performance. These results show that MARS provides an efective balance between retrieval accuracy and online computational eficiency.

Table 6: Text-image retrieval on COCO and Flickr30K. All values are R@1.
<table><tr><td>Method</td><td colspan="2">COCO (FT)</td><td colspan="2">Flickr30K (ZS)</td></tr><tr><td></td><td>T2I</td><td>I2T</td><td>T2I</td><td>I2T</td></tr><tr><td>Direct retrieval</td><td></td><td></td><td></td><td></td></tr><tr><td>BEiT-3 (Wang et al., 2023a)</td><td>65.1</td><td>82.7</td><td>89.1</td><td>97.5</td></tr><tr><td>MARS (Ours)</td><td>66.3</td><td>83.9</td><td>89.1</td><td>97.0</td></tr><tr><td>Fusion reranking</td><td></td><td></td><td></td><td></td></tr><tr><td>ALBEF (Li et al., 2021)</td><td>60.7</td><td>77.6</td><td>82.8</td><td>94.1</td></tr><tr><td>BLIP (Li et al., 2022)</td><td>65.1</td><td>82.4</td><td>86.7</td><td>96.7</td></tr><tr><td>BLIP-2 (Li et al., 2023a)</td><td>68.3</td><td>85.4</td><td>89.7</td><td>97.6</td></tr></table>

## 4.6 Generalization Across MLLM Backbones

To examine whether the efectiveness of MARS generalizes across diferent architectures, we evaluate it on three distinct MLLM backbones: VideoLLaMA3-7B (Zhang et al., 2025a) and Qwen2-VL (Wang et al., 2024a) (7B and 2B). For each backbone, we compare MARS against its corresponding single-representation baseline. As shown in Table 5, MARS consistently improves both T2V and V2T R@1 across all four benchmarks. Notably, it yields consistent absolute improvements in average R@1 ranging from 2.0% to 3.4% across VideoLLaMA3-7B, Qwen2-VL-7B, and Qwen2-VL-2B for both retrieval directions. These robust enhancements confirm that MARS is not tied to a specific backbone, but serves as a general mechanism for extracting stronger retrieval representations from diverse MLLMs.

## 4.7 Generalization to Text–Image Retrieval

Although our main experiments focus on textvideo retrieval, the MARS framework is not restricted to video inputs. To examine its applicability beyond the video domain, we evaluate MARS on text-image retrieval using COCO (Lin et al., 2014) (fine-tuned) and Flickr30K (Plummer et al., 2015) (zero-shot), following the Karpathy split (Karpathy and Fei-Fei, 2015) adopted by BEiT-3. As shown in Table 6, MARS achieves R@1 scores 1.2 points higher than BEiT-3 on COCO in both T2I and I2T directions. On Flickr30K, MARS maintains highly competitive zero-shot performance. Although BLIP-2 achieves higher scores with an additional cross-modal reranking stage, MARS remains competitive using only direct similarity-based retrieval. These results indicate that the core MARS mechanism generalizes well beyond the video domain, serving as a powerful representation extractor for textimage retrieval as well.

![](images/c6be00068f9edbfad000ea8774aec81a01e0ea90d3a7e791bab8a3a6d51dd8fc.jpg)  
Figure 5: Qualitative hard-negative comparison. Unlike the baseline, MARS correctly ranks the ground truth above a visually similar negative.

## 4.8 Qualitative Results

Figure 5 shows a qualitative hard-negative case from DiDeMo. For the same text query, the final-layer single-slot baseline ranks a visually plausible negative above the groundtruth video. We use this example to examine whether MARS can correct confusion between candidates with similar scene-level content. The query contains several fine-grained cues, including the stroller, the girl’s movement, and the directional motions of nearby women. While the baseline assigns a higher score to the hard negative, MARS reverses this ordering and retrieves the ground-truth video at rank one. This example illustrates that MARS can better distinguish the ground-truth video from a confusing negative using the final score.

## 5 Conclusion

In this paper, we introduced MARS, a multilayer and multi-slot embedding framework for text-video retrieval with MLLMs. Through this design, experiments on four benchmarks show that MARS achieves strong retrieval performance while retaining a direct similarity-based retrieval structure. Further ablations and analyses confirm that multi-layer fusion, adaptive representation slots, and hard-negative-aware specialization provide complementary gains. These findings suggest that richer representation extraction from MLLMs is promising for fine-grained retrieval. Future work can explore adaptive slot interactions, temporal reasoning, and broader retrieval scenarios.

## Limitations

This work has several limitations. First, although MARS is evaluated across multiple established text-video retrieval benchmarks, these datasets may not fully reflect opendomain scenarios involving substantially longer videos, noisy descriptions, or diverse user queries. Second, while we further demonstrate the generalization of MARS from text-video to text-image retrieval, its applicability to other modalities, such as audio-language or 3Dlanguage retrieval, remains unexplored. Third, our study focuses on video-level retrieval and does not explicitly address temporal grounding or moment-level retrieval. Extending MARS to these broader retrieval settings represents a promising direction for future work.

Artifact licenses. We use publicly available benchmark datasets and pretrained models for research purposes, following their respective licenses and terms of use.

## Acknowledgements

This work was partly supported by Institute of Information and communications Technology Planning & Evaluation (IITP) under the Development of Multimodal Data Input-Based Search Augmentation Generation Technology (IITP-2026-RS-2024-00455244, 50%), the Leading Generative AI Human Resources Development (IITP-2026-RS-2026-25544647, 25%) and the Artificial Intelligence Innovation Human Resources Development (IITP-2026-RS-2026- 25549817, 25%), grant funded by the Korea government(MSIT).

## References

Max Bain, Arsha Nagrani, G¨ul Varol, and Andrew Zisserman. 2021. Frozen in time: A joint video and image encoder for end-to-end retrieval. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 1728–1738.

Haonan Chen, Liang Wang, Nan Yang, Yutao Zhu, Ziliang Zhao, Furu Wei, and Zhicheng Dou. 2025. mmE5: Improving multimodal multilingual embeddings via high-quality synthetic data. In Findings of the Association for Computational Linguistics: ACL 2025, pages 8254–8275. Associ ation for Computational Linguistics.

Shizhe Chen, Yida Zhao, Qin Jin, and Qi Wu. 2020a. Fine-grained video-text retrieval with

hierarchical graph reasoning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10638–10647.

Tianqi Chen, Bing Xu, Chiyuan Zhang, and Carlos Guestrin. 2016. Training deep nets with sublinear memory cost. arXiv preprint arXiv:1604.06174.

Yen-Chun Chen, Linjie Li, Licheng Yu, Ahmed El Kholy, Faisal Ahmed, Zhe Gan, Yu Cheng, and Jingjing Liu. 2020b. UNITER: UNiversal Image-TExt Representation Learning. In Computer Vision – ECCV 2020, volume 12375 of Lecture Notes in Computer Science, pages 104– 120. Springer.

Xing Cheng, Hezheng Lin, Xiangyu Wu, Fan Yang, and Dong Shen. 2021. Improving videotext retrieval by multi-stream corpus alignment and dual softmax loss. arXiv preprint arXiv:2109.04290.

Jianfeng Dong, Xirong Li, Chaoxi Xu, Xun Yang, Gang Yang, Xun Wang, and Meng Wang. 2022. Dual encoding for video retrieval by text. IEEE Transactions on Pattern Analysis and Machine Intelligence, 44(8):4065–4080.

Valentin Gabeur, Chen Sun, Karteek Alahari, and Cordelia Schmid. 2020. Multi-modal transformer for video retrieval. In Computer Vision – ECCV 2020, volume 12349 of Lecture Notes in Computer Science, pages 214–229. Springer.

Lisa Anne Hendricks, Oliver Wang, Eli Shechtman, Josef Sivic, Trevor Darrell, and Bryan Russell. 2017. Localizing moments in video with natural language. In Proceedings of the IEEE International Conference on Computer Vision, pages 5803–5812.

Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2022. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations.

Ting Jiang, Shaohan Huang, Zhongzhi Luan, Deqing Wang, and Fuzhen Zhuang. 2024a. Scaling sentence embeddings with large language models. In Findings of the Association for Computational Linguistics: EMNLP 2024, pages 3182–3196, Miami, Florida, USA. Association for Computational Linguistics.

Ting Jiang, Minghui Song, Zihan Zhang, Haizhen Huang, Weiwei Deng, Feng Sun, Qi Zhang, Deqing Wang, and Fuzhen Zhuang. 2024b. E5-V: Universal embeddings with multimodal large language models. arXiv preprint arXiv:2407.12580.

Ziyan Jiang, Rui Meng, Xinyi Yang, Semih Yavuz, Yingbo Zhou, and Wenhu Chen. 2025. VLM2Vec: Training vision-language models for massive multimodal embedding tasks. In International Conference on Learning Representations.

Uicheol Jung, Juyoung Hong, Hojung Kwon, and Yukyung Choi. 2026. TAME: Temporal-aware mixture-of-experts for text-video retrieval. IEEE Access, 14:16188–16203.

Andrej Karpathy and Li Fei-Fei. 2015. Deep visualsemantic alignments for generating image descriptions. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 3128–3137.

Dohwan Ko, Ji Soo Lee, Minhyuk Choi, Zihang Meng, and Hyunwoo J. Kim. 2025. Bidirectional likelihood estimation with multi-modal large language models for text-video retrieval. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 22263–22273.

Ranjay Krishna, Kenji Hata, Frederic Ren, Li Fei-Fei, and Juan Carlos Niebles. 2017. Densecaptioning events in videos. In Proceedings of the IEEE International Conference on Computer Vision, pages 706–715.

Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. 2023a. BLIP-2: Bootstrapping languageimage pre-training with frozen image encoders and large language models. In Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pages 19730–19742. PMLR.

Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. 2022. BLIP: Bootstrapping language-image pre-training for unified vision-language understanding and generation. In Proceedings of the 39th International Conference on Machine Learning, volume 162 of Proceedings of Machine Learning Research, pages 12888–12900. PMLR.

Junnan Li, Ramprasaath R. Selvaraju, Akhilesh D. Gotmare, Shafiq Joty, Caiming Xiong, and Steven C.H. Hoi. 2021. Align before fuse: Vision and language representation learning with momentum distillation. In Advances in Neural Information Processing Systems, volume 34, pages 9694–9705.

Kunchang Li, Yali Wang, Yizhuo Li, Yi Wang, Yinan He, Limin Wang, and Yu Qiao. 2023b. Unmasked teacher: Towards training-eficient video foundation models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 19948–19960.

Xinhao Li, Yi Wang, Jiashuo Yu, Xiangyu Zeng, Yuhan Zhu, Haian Huang, Jianfei Gao, Kunchang Li, Yinan He, Chenting Wang, Yu Qiao, Yali Wang, and Limin Wang. 2025. VideoChat-Flash: Hierarchical compression for long-context video modeling. arXiv preprint arXiv:2501.00574.

Sheng-Chieh Lin, Chankyu Lee, Mohammad Shoeybi, Jimmy Lin, Bryan Catanzaro, and Wei Ping. 2025. MM-Embed: Universal multimodal retrieval with multimodal LLMs. In International Conference on Learning Representations.

Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Doll´ar, and C. Lawrence Zitnick. 2014. Microsoft coco: Common objects in context. In Computer Vision – ECCV 2014, pages 740–755. Springer.

Yang Liu, Samuel Albanie, Arsha Nagrani, and Andrew Zisserman. 2019. Use what you have: Video retrieval using representations from collaborative experts. In Proceedings of the British Machine Vision Conference.

Yikun Liu, Yajie Zhang, Jiayin Cai, Xiaolong Jiang, Yao Hu, Jiangchao Yao, Yanfeng Wang, and Weidi Xie. 2025. LamRA: Large multimodal model as your advanced retrieval assistant. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4015–4025.

Huaishao Luo, Lei Ji, Ming Zhong, Yang Chen, Wen Lei, Nan Duan, and Tianrui Li. 2022. CLIP4Clip: An empirical study of CLIP for end to end video clip retrieval and captioning. Neurocomputing, 508:293–304.

Yiwei Ma, Guohai Xu, Xiaoshuai Sun, Ming Yan, Ji Zhang, and Rongrong Ji. 2022. X-CLIP: End-to-end multi-grained contrastive learning for video-text retrieval. In Proceedings of the 30th ACM International Conference on Multimedia, pages 638–647.

Rui Meng, Ziyan Jiang, Ye Liu, Mingyi Su, Xinyi Yang, Yuepeng Fu, Can Qin, Zeyuan Chen, Ran Xu, Caiming Xiong, Yingbo Zhou, Wenhu Chen, and Semih Yavuz. 2026. VLM2Vec-V2: Advancing multimodal embedding for videos, images, and visual documents. Transactions on Machine Learning Research.

Antoine Miech, Dimitri Zhukov, Jean-Baptiste Alayrac, Makarand Tapaswi, Ivan Laptev, and Josef Sivic. 2019. HowTo100M: Learning a text-video embedding by watching hundred million narrated video clips. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 2630–2640.

Bryan A. Plummer, Liwei Wang, Chris M. Cervantes, Juan C. Caicedo, Julia Hockenmaier, and Svetlana Lazebnik. 2015. Flickr30k entities: Collecting region-to-phrase correspondences for richer image-to-sentence models. In Proceedings of the IEEE International Conference on Computer Vision, pages 2641–2649.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. 2021. Learning transferable visual models from natural language supervision. In Proceedings of the 38th International Conference on Machine Learning, pages 8748–8763.

Arun Reddy, Alexander Martin, Eugene Yang, Andrew Yates, Kate Sanders, Kenton Murray, Reno Kriz, Celso M. de Melo, Benjamin Van Durme, and Rama Chellappa. 2025. Video-ColBERT: Contextualized late interaction for text-to-video retrieval. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 19691–19701.

Anna Rohrbach, Atousa Torabi, Marcus Rohrbach, Niket Tandon, Christopher Pal, Hugo Larochelle, Aaron Courville, and Bernt Schiele. 2017. Movie description. International Journal of Computer Vision, 123(1):94–120.

Leqi Shen, Tianxiang Hao, Tao He, Sicheng Zhao, Yifeng Zhang, Pengzhang Liu, Yongjun Bao, and Guiguang Ding. 2025. TempMe: Video temporal token merging for eficient text-video retrieval. In International Conference on Learning Representations.

Oscar Skean, Md Rifat Arefin, Dan Zhao, Niket Nikul Patel, Jalal Naghiyev, Yann Lecun, and Ravid Shwartz-Ziv. 2025. Layer by layer: Uncovering hidden representations in language models. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 55854–55875. PMLR.

Yixuan Tang and Yi Yang. 2024. Pooling and attention: What are efective designs for LLMbased embedding models? arXiv preprint arXiv:2409.02727.

Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Yang Fan, Kai Dang, Mengfei Du, Xuancheng Ren, Rui Men, Dayiheng Liu, Chang Zhou, Jingren Zhou, and Junyang Lin. 2024a. Qwen2-VL: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191.

Wenhui Wang, Hangbo Bao, Li Dong, Johan Bjorck, Zhiliang Peng, Qiang Liu, Kriti Aggarwal, Owais Khan Mohammed, Saksham Singhal, Subhojit Som, and Furu Wei. 2023a. Image as a foreign language: BEiT pretraining for vision and vision-language tasks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 19175–19186.

Yi Wang, Yinan He, Yizhuo Li, Kunchang Li, Jiashuo Yu, Xin Ma, Xinhao Li, Guo Chen, Xinyuan Chen, Yaohui Wang, Conghui He, Ping Luo, Ziwei Liu, Yali Wang, Limin Wang, and Yu Qiao. 2023b. InternVid: A large-scale videotext dataset for multimodal understanding and generation. arXiv preprint arXiv:2307.06942.

Yi Wang, Kunchang Li, Xinhao Li, Jiashuo Yu, Yinan He, Chenting Wang, Guo Chen, Baoqi Pei, Ziang Yan, Rongkun Zheng, Jilan Xu, Zun

Wang, Yansong Shi, Tianxiang Jiang, Songze Li, Hongjie Zhang, Yifei Huang, Yu Qiao, Yali Wang, and Limin Wang. 2024b. InternVideo2: Scaling foundation models for multimodal video understanding. In Proceedings of the European Conference on Computer Vision.

Yi Wang, Kunchang Li, Yizhuo Li, Yinan He, Bingkun Huang, Zhiyu Zhao, Hongjie Zhang, Jilan Xu, Yi Liu, Zun Wang, Sen Xing, Guo Chen, Junting Pan, Jiashuo Yu, Yali Wang, Limin Wang, and Yu Qiao. 2022. InternVideo: General video foundation models via generative and discriminative learning. arXiv preprint arXiv:2212.03191.

Ziyang Wang, Yi-Lin Sung, Feng Cheng, Gedas Bertasius, and Mohit Bansal. 2023c. Unified coarse-to-fine alignment for video-text retrieval. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 2816– 2827.

Wenhao Wu, Haipeng Luo, Bo Fang, Jingdong Wang, and Wanli Ouyang. 2023. Cap4Video: What can auxiliary captions do for text-video retrieval? In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10704–10713.

Jun Xu, Tao Mei, Ting Yao, and Yong Rui. 2016. MSR-VTT: A large video description dataset for bridging video and language. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 5288–5296.

Hongwei Xue, Yuchong Sun, Bei Liu, Jianlong Fu, Ruihua Song, Houqiang Li, and Jiebo Luo. 2023. CLIP-ViP: Adapting pre-trained imagetext model to video-language representation alignment. In International Conference on Learning Representations.

An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Zhou, Chengpeng Li, Chengyuan Li, Dayiheng Liu, Fei Huang, Guanting Dong, Haoran Wei, Huan Lin, Jialong Tang, Jialin Wang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Ma, and 43 others. 2024. Qwen2 technical report. arXiv preprint arXiv:2407.10671.

Youngjae Yu, Jongseok Kim, and Gunhee Kim. 2018. A joint sequence fusion model for video question answering and retrieval. In Proceedings of the European Conference on Computer Vision, page 487–503.

Boqiang Zhang, Kehan Li, Zesen Cheng, Zhiqiang Hu, Yuqian Yuan, Guanzheng Chen, Sicong Leng, Yuming Jiang, Hang Zhang, Xin Li, Peng Jin, Wenqi Zhang, Fan Wang, Lidong Bing, and Deli Zhao. 2025a. VideoLLaMA 3: Frontier multimodal foundation models for image and video understanding. arXiv preprint arXiv:2501.13106.

Xin Zhang, Yanzhao Zhang, Wen Xie, Mingxin Li, Ziqi Dai, Dingkun Long, Pengjun Xie, Meishan Zhang, Wenjie Li, and Min Zhang. 2025b. Bridging modalities: Improving universal multimodal retrieval by multimodal large language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9274–9285.

Junjie Zhou, Yongping Xiong, Zheng Liu, Ze Liu, Shitao Xiao, Yueze Wang, Bo Zhao, Chen Jason Zhang, and Defu Lian. 2025. MegaPairs: Massive data synthesis for universal multimodal retrieval. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 19076–19095, Vienna, Austria. Association for Computational Linguistics.

## Appendix Overview

• In Appendix A, we provide dataset details for the text-video retrieval benchmarks used in our experiments.

• In Appendix B, we describe the implementation details, including model architecture, trainable parameters, and training configuration.

• In Appendix C, we analyze slot aggregation strategies and justify the use of uniform aggregation.

• In Appendix D, we present additional analyses of adaptive representation slots using slot geometry diagnostics.

## A Dataset Details

## A.1 Text-Video Retrieval

We provide dataset-specific details on preprocessing, text construction, and split statistics for the text-video retrieval benchmarks used in our experiments. The resulting statistics are summarized in Table A1.

DiDeMo. Distinct Describable Moments (DiDeMo) (Hendricks et al., 2017) contains short videos with multiple moment-level descriptions. As in prior text-video retrieval studies (Liu et al., 2019; Bain et al., 2021; Luo et al., 2022; Ma et al., 2022), we concatenate all captions associated with the same video and formulate the task as paragraph-video retrieval. In our processed split, the training set contains 8,381 video-text pairs and the test set contains 1,003 video-text pairs. Each video has 3.9 captions on average in the training split. The captions are merged into a single text input using whitespace concatenation.

ActivityNet Captions. ActivityNet Captions (Krishna et al., 2017) contains YouTube videos annotated with dense event descriptions. We use the val 1 split for evaluation and construct paragraph-level text queries by concatenating all captions associated with the same video, consistent with standard text-video retrieval protocols (Liu et al., 2019; Bain et al., 2021; Luo et al., 2022; Ma et al., 2022). The training set contains 10,009 video-text pairs, and the test set contains 4,917 video-text pairs. Each video has 3.7 captions on average in the training split. Compared with DiDeMo, ActivityNet contains longer videos and multiple event-level captions per video.

LSMDC. The Large Scale Movie Description Challenge (LSMDC) (Rohrbach et al., 2017) consists of short movie clips paired with captions from movie scripts or descriptive video services. We follow the standard text-video retrieval setup used in previous works (Liu et al., 2019; Bain et al., 2021; Luo et al., 2022; Ma et al., 2022) and evaluate the model on the 1,000-sample test split. Our processed training set contains 101,055 samples, and the test split contains 1,000 samples.

MSR-VTT. MSR-VTT (Xu et al., 2016) contains 10K video clips from diverse categories, where each training video is annotated with multiple captions. We adopt the standard 1K-A test protocol (Yu et al., 2018; Liu et al., 2019), using 9,000 training videos with 20 captions per video, resulting in 180,000 training text-video pairs. The test split contains 1,000 videos. In our dataloader, each caption is treated as an individual text input during training, while the video feature is shared across captions from the same video.

## B Implementation Details

Model architecture and trainable parameters. MARS is built on VideoChat-Flash-Qwen2-7B (Li et al., 2025). This MLLM combines a UMT-L vision encoder (Li et al., 2023b), a linear multimodal projection layer, and a Qwen2 language model (Yang et al., 2024). The language model contains 28 decoder layers. Following the prompt design in Sec. 3, we insert adaptive representation tokens into the response field for both video and text inputs. We use the hidden states of these tokens from all decoder layers for layer fusion. The number of slots is set to four for DiDeMo, ActivityNet, and LSMDC, and three for MSR-VTT.

We fine-tune the model in a parametereficient manner. LoRA (Hu et al., 2022) is applied to the attention and MLP projection modules of the language model, as well as to the multimodal projection layer. We set the LoRA rank to 128, the LoRA scaling factor to 256, and the dropout rate to 0.05. The trainable parameters consist of the LoRA parameters, the adaptive representation tokens, and the slotwise layer-fusion weights. All other pretrained parameters are frozen during training.

Table A1: Dataset statistics used in our text-video retrieval experiments. DiDeMo and ActivityNet use paragraph-level text inputs by concatenating multiple captions of the same video, while LSMDC and MSR-VTT use one caption per training sample.
<table><tr><td>Dataset</td><td>Train Samples</td><td>Test Samples</td><td>Caption Format</td><td>Captions / Video</td></tr><tr><td>DiDeMo</td><td>8,381</td><td>1,003</td><td>list of captions</td><td>3.9</td></tr><tr><td>ActivityNet</td><td>10,009</td><td>4,917</td><td>list of captions</td><td>3.7</td></tr><tr><td>LSMDC</td><td>101,055</td><td>1,000</td><td>single caption</td><td>1.0</td></tr><tr><td>MSR-VTT</td><td>180,000</td><td>1,000</td><td>single caption</td><td>20.0</td></tr></table>

Table A2: Training settings for each text-video retrieval dataset. Shared settings are applied across all benchmarks, while dataset-specific settings such as the number of slots, total epochs, learning rate, and efective batch size are reported separately.
<table><tr><td></td><td>DiDeMo</td><td>ActivityNet</td><td>LSMDC</td><td>MSR-VTT</td></tr><tr><td>Optimizer AdamW betas  $( \beta _ { 1 } , \beta _ { 2 } )$ </td><td></td><td>AdamW (0.9,0.95)</td><td></td><td></td></tr><tr><td>Weight decay Warmup epochs</td><td></td><td>0.05</td><td>1</td><td></td></tr><tr><td>Input frames</td><td></td><td>16</td><td></td><td></td></tr><tr><td> $\lambda _ { \mathrm { d i v } }$ </td><td></td><td>0.1</td><td></td><td></td></tr><tr><td> $\lambda _ { \mathrm { h n } }$ </td><td></td><td>0.05</td><td></td><td></td></tr><tr><td>Hard-negative margin δ</td><td></td><td>0.04</td><td></td><td></td></tr><tr><td>Number of slots M</td><td></td><td></td><td></td><td></td></tr><tr><td>Total epochs</td><td>4</td><td>4</td><td>4</td><td>3</td></tr><tr><td>Learning rate</td><td>5</td><td>5</td><td>3</td><td>3</td></tr><tr><td>Effective batch size</td><td>2e-4 320</td><td>2e-4 320</td><td>2e-4 320</td><td>1e-4 512</td></tr></table>

Training configuration. We sample 16 frames from each video in all experiments. The learning rate is linearly warmed up during the first epoch. To reduce activation memory, we use gradient checkpointing (Chen et al., 2016), which allows larger efective batch sizes for contrastive learning. All models are trained on two NVIDIA A100 40GB GPUs. Dataset-specific hyperparameters are reported in Table A2.

## C Slot Aggregation Analysis

MARS computes the final retrieval score by aggregating the similarities of corresponding textvideo slots. In the main method, we use uniform aggregation, where each slot contributes equally to the final score. This design avoids introducing additional slot-weighting parameters and preserves the contribution of all adaptive representation slots. Table A3 compares uniform aggregation with two alternative strategies. The learnable strategy assigns trainable weights to diferent slots, while MaxSim selects the strongest slot-level similarity. Uniform aggregation achieves the best average performance across the eight T2V/V2T settings, with an average R@1 of 65.7, compared with 65.4 for learnable aggregation and 65.1 for MaxSim. These results suggest that the slots encode complementary retrieval cues, and that a simple average combines them more robustly than either learned weighting or maximum-based selection.

Table A3: Ablation on slot aggregation strategies.
<table><tr><td rowspan="2">Aggregation</td><td colspan="2">DiDeMo</td><td colspan="2">ActivityNet</td><td colspan="2">LSMDC</td><td colspan="2">MSR-VTT</td><td rowspan="2">Avg.</td></tr><tr><td>T2V</td><td>V2T</td><td>T2V</td><td>V2T</td><td>T2V</td><td>V2T</td><td>|T2V</td><td>V2T |</td></tr><tr><td>Learnable</td><td>78.6</td><td>74.4</td><td>74.8</td><td>73.4</td><td>50.8</td><td>50.2</td><td>61.1</td><td>59.5</td><td>65.4</td></tr><tr><td>MaxSim</td><td>78.2</td><td>73.9</td><td>74.6</td><td>72.7</td><td>50.4</td><td>50.3</td><td>60.8</td><td>59.7</td><td>65.1</td></tr><tr><td>Uniform</td><td>79.7</td><td>75.7</td><td>75.7</td><td>73.2</td><td>51.0</td><td>50.0</td><td>61.6</td><td>59.0</td><td>65.7</td></tr></table>

## D Additional Analysis of Adaptive Representation Slots

We provide additional analyses of adaptive representation slots. These results complement the main analysis in Sec. 4.4 by examining the geometry of slot embeddings from additional perspectives. All analyses follow the same notation as Sec. 3, where each input is represented by multiple adaptive representation slots and the final retrieval score is computed by uniform aggregation of the corresponding slot-wise similarities.

Per-slot effective rank  
![](images/f22d90c85a2007387e1db9603f8d61bd865aca2a235f7eba080efcd13f97bce7.jpg)

![](images/08a7dce2a1d59a9d729335a7439c305fd11ef281cc9c95596e3b8d81ee5d9038.jpg)

![](images/9c54521d446bc2ffbb35ee8dfff3a8bd5c4c412d9a99a41ffc59c7ffdf55f169.jpg)

![](images/cacd4bbe0b71e426791c5201f70970f4d7e709f975a93f8d2742ffeaea9cab5d.jpg)

![](images/51091144044fc3b4dafcaef806cad1cd7af613bb87a2c44d187d2ddfe1699497.jpg)

![](images/baa9b5a145f5102ae20f1a91b24cea67da233681864f30c151276e60e80efe61.jpg)

![](images/bf5dac83ce31632f2a077ba1ef4b54a2b29609f5e4ae3f1ea89a8ae35ae29aa7.jpg)

![](images/77b504486cbcaa496939d4bcf4cd1f68ed787d3a182fe2b82658126704b4c413.jpg)  
Figure A1: Additional geometry diagnostics of adaptive representation slots. The first panel reports linear CKA between slot embedding matrices, which measures sample-level structural similarity between slots. The second panel reports the efective rank of each slot embedding matrix, where a higher efective rank indicates that the slot uses a broader set of embedding directions.

## D.1 Slot Geometry Diagnostics

We further examine the geometry of adaptive representation slots. These diagnostics are not used as retrieval metrics, but provide additional evidence about how the slots relate to one another in the embedding space. Since pairwise cosine similarity is discussed in Sec. 4.4, we focus here on two complementary diagnostics: linear CKA and efective rank.

Pairwise cosine similarity compares individual vector directions, but does not fully describe whether two slots encode similar samplelevel structures. We therefore compute linear CKA between slot embedding matrices. As shown in the first panel of Figure A1, the structural similarity between slots varies across benchmarks. This suggests that slot distinctness can depend on dataset characteristics. Together with the pairwise cosine analysis in Sec. 4.4, the CKA analysis provides a complementary view of how adaptive representation slots difer beyond their vector directions.

slot embedding matrix. The efective rank is computed from the entropy of the normalized squared singular values. It reflects how broadly a slot uses the available embedding dimensions. The second panel of Figure A1 shows that diferent slots can have diferent efective ranks. This indicates that adaptive representation slots do not necessarily use the embedding space in the same way. Combined with the retrieval-side analyses in Sec. 4.4, these geometry diagnostics suggest that the slots form distinct representation patterns.

We also measure the efective rank of each