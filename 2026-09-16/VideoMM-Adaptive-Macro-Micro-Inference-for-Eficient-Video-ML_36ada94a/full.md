# VideoMM: Adaptive Macro-Micro Inference for Eficient Video MLLMs

Haoyu Guo<sup>1,3,∗</sup>, Yuan Feng<sup>2,3,∗</sup>, Junlin Lv<sup>2,3</sup>, Mingjun Xiao<sup>2,3</sup>, S Kevin Zhou<sup>1,3</sup>, Xike Xie<sup>1,3,†</sup>

<sup>1</sup>School of Biomedical Engineering, University of Science and Technology of China

<sup>2</sup>School of Computer Science, USTC

<sup>3</sup>Data Darkness Lab, MIRACLE Center, Suzhou Institute for Advanced Research China

## Abstract

Scaling Multimodal Large Language Models (MLLMs) to long-form video understanding is bottlenecked by the explosion of visual tokens, which saturates context windows and incurs prohibitive costs. Current solutions predominantly rely on auxiliary models for token reduction but face a fundamental dilemma: lightweight encoder-driven approaches often overlook critical semantic information, whereas heavyweight MLLM-driven reduction negates the eficiency gains. In this work, we identify a more fundamental inefficiency underlying this dilemma: while fine-grained visual details are essential for detailed understanding, they are largely redundant for the preliminary task of selecting semantically relevant regions. Motivated by this, we introduce VideoMM, which marks a paradigm shift from model-centric downsizing to adaptive percep tual granularity. Specifically, our framework decouples selection from reasoning by executing semantic filtering on a cost-efective Macro Proxy (derived from downscaled frames), and projecting the selected regions onto high-fidelity Micro Tokens for detailed understanding only when necessary. Extensive evaluations show that VideoMM significantly outperforms existing solutions. It achieves a 6.13× speedup and a 7.4% accuracy gain over full-context baselines on LongVideoBench, and further accelerates inference by 2.73× over current leading methods, establishing a highly scalable paradigm for long-video understanding. Our code is available at: https://github.com/adfh917k/VideoMM.

## 1 Introduction

Multimodal Large Language Models (MLLMs) [2, 24, 26, 27, 44] have recently achieved remarkable success in interpreting static images and short video clips. However, extending these capabilities to long-form video understanding remains a critical challenge. While long-video analysis enables high-value applications—such as video QA, temporal event localization, and long-horizon action recognition [1, 19, 21]—it introduces severe scalability bottlenecks. The primary constraint is the massive influx of visual tokens: for instance, a single 10-minute 720p video sampled at 1 fps generates over 300k tokens in the widely adopted Qwen-3-VL model. This volume rapidly exhausts the limited context windows of current models, leading to prohibitive computational costs.

To mitigate this, existing research has focused on reducing the number of visual tokens via auxiliary models prior to full inference, yet these approaches face a fundamental dilemma. Lightweight encoder-driven methods [3, 22, 38] are eficient but often discard semantically critical information due to limited semantic awareness, while MLLM-driven selection [42] improves accuracy at the cost of substantial computational overhead. As a result, existing model-centric downsizing methods struggle to reconcile the tradeof between accuracy and eficiency in token selection. We argue that this bottleneck arises from a more fundamental redundancy: in long-video understanding, diferent stages of inference require fundamentally diferent levels of perceptual fidelity.

![](images/7b66ba01b5e5e7fc4f581c6192fd4a870ff3eaaf8dce80303eaf4594e3fd3848.jpg)  
(a) Qwen2.5-VL-7B

![](images/6a6be16a5816c4b1a30dde4fd17f41820bd5553d246fb19ed11e480810cc8d25.jpg)

![](images/efb2bb0f933782b16700824d6882d290a3e65816890320f9b58962cb7587054f.jpg)  
(b) GLM-4.1V-9B  
(c) Qwen3-VL-8B  
Figure 1: VideoMM sets a new accuracy–eficiency frontier on LongVideoBench, achieving a 6.13× inference speedup with a 7.4% accuracy gain over the vanilla baseline, and a 2.73× speedup over FlexSelect without accuracy loss.

While high-fidelity visual details are essential for precise semantic understanding, they are largely unnecessary for the preliminary task of identifying which regions are relevant. Consequently, performing semantic filtering directly on original high-resolution visual tokens is inherently wasteful.

Drawing inspiration from the coarse-to-fine nature of human perception-where one first grasps the global context (macro) before focusing on specific details (micro)—we propose VideoMM, a macro-micro paradigm that decouples semantic localization from fine-grained reasoning. VideoMM initially identifies relevant regions using a lightweight Macro Proxy derived from downscaled frames. Subsequently, it employs a consensus mechanism to govern adaptive computation: high-resolution Micro Tokens are selectively activated only when the necessity for fine-grained inference is confirmed. By strictly adhering to this coarse-to-fine progression, VideoMM ensures that heavy computation is reserved solely for regions requiring granular scrutiny, thereby minimizing redundant computation without compromising semantic fidelity. Extensive evaluations across three benchmarks validate the efectiveness of our approach. Notably, as shown in Figure 1, VideoMM achieves an average speedup of 6.13× and an accuracy gain of 7.0% over vanilla baselines and accelerates state-of-the-art methods by over 2.73× with comparable accuracy on the challenging LongVideoBench. To summarize, our contributions are as follows:

• We identify a fundamental dilemma in existing token reduction methods rooted in model downsizing, which struggle to balance accuracy with eficiency. To resolve this, we propose a paradigm shift from model scale to perceptual granularity, exploiting the coarse-to-fine nature of visual information to reconcile the conflict between selection pre cision and overhead.

• We propose VideoMM, a macro-micro paradigm mimicking human coarse-to-fine perception: Grouped Selection with Macro-view Proxy decouples token selection from dense processing via lightweight proxies, while Adaptive Macro-Micro Inference dynamically recruits specific details only when necessary for ambiguous cases.

• Extensive evaluations validate the efectiveness ofVideoMM, achieving a 6.1× speedup and a 7.0% accuracy gain on LongVideoBench. Through further analysis of macro-micro variants, we demonstrate the intrinsic eficacy of this perceptual granularity-oriented paradigm, establishing a foundational and promising direction that moves beyond the existing model-centric perspective.

## 2 Related Works

Given the escalating visual token counts in video MLLMs, token reduction is essential for eficient inference. Current strategies typically employ an additional downscaling model to prune unimpor tant tokens prior to full inference. Depending on the downscaling model used, these strategies fall into two main categories: (1) lightweight Encoder-Driven methods [3, 22, 25, 38] selects tokens using only the ViT encoder. For example, VisionZip [38] first selects a sub set of dominant visual tokens using ViT attention and then merges the remaining less important tokens through averaging based on embedding similarity, thereby achieving efective token reduction. In contrast, VidCom [22] uses the diversity of ViT embeddings to represent frame-wise importance, subsequently applying difer ent token compression ratios to diferent frames. However, these methods are prone to erroneous pruning, as lightweight ViTs lack the capacity to fully capture complex video semantics. (2) Recent heavyweight MLLM-driven methods [7, 41, 42] further leverage a small-scale MLLM to more accurately detect and preserve essential visual tokens. For instance, the current SOTA method, FlexSelect [42], employs a reduced-size MLLM to identify significant tokens via cross-modal attention, subsequently feeding only these selected tokens and the text query into the target MLLM for inference. While this approach improves accuracy, it remains computationally expensive; the overhead introduced by the auxiliary MLLM efectively negates the latency gains achieved through token reduction.

Overall, the prevailing emphasis on the downscaling model’s size for fast token reduction results in an accuracy–eficiency dilemma. In contrast, VideoMM introduces a paradigm shift by shifting the focus from model scale to perceptual granularity. By constructing a macro-level proxy for fast token reduction, VideoMM substantially reduces the computational overhead of this process—even when using a MLLM to guide compression—without sacrificing semantic depth. Empirically, VideoMM achieves faster inference than lightweight encoder-driven methods while preserving the accuracy advantages of MLLM-driven strategies. <sup>1</sup>

## 3 Methods

We consider standard MLLMs generally consisting of two primary components: a vision transformer encoder (ViT), and a large language model �. Given an input video $\mathbf { V } \in \dot { \mathbb { R } } ^ { T \times H \times W \times 3 } ,$ , a ViT encoder directly processes the frames into a sequence of visual tokens:

$$
\mathbf { X } _ { V } = \mathrm { V i T } ( \mathbf { V } ) \in \mathbb { R } ^ { N \times D } ,\tag{1}
$$

where $N = T \times H _ { p } \times W _ { p }$ is the total number of visual tokens, � is the number of sampled video frames, $H _ { p } , W _ { p }$ denotes the patched height and patched width respectively, and � is the hidden dimension of the ViT. Finally, the visual tokens $\mathbf { X } _ { V }$ are concatenated with the text query tokens $\mathbf { X } _ { Q }$ . The LLM � generates the response $\mathbf { Y } = \{ y _ { i } \} _ { i = } ^ { L }$ 1 autoregressively by modeling the conditional probability:

$$
\ L _ { \hat { P } } ( \mathbf { Y } \mid \mathbf { X } _ { V } , \mathbf { X } _ { Q } ) = \prod _ { i = 1 } ^ { L } \ L _ { \hat { P } } ( y _ { i } \mid \mathbf { X } _ { V } , \mathbf { X } _ { Q } , y _ { < i } ) ,
$$

where $y _ { < i }$ represents the tokens generated prior to step �.

## 3.1 Overall Architecture of VideoMM

We present VideoMM, a framework that operationalizes the coarseto-fine nature ofhuman perception to resolve the eficiency-accuracy dilemma in long-form video understanding. The core insight driving our architecture is that while long videos contain massive tokens, semantic density is sparse: a global macro-view is often suficient for general reasoning, whereas micro details are only necessary for specific, ambiguous segments. As illustrated in Figure 2, VideoMM implements this philosophy through pipelined collaboration between two distinct stages. (1) Grouped Selection with Macro Proxy: This stage acts as a lightweight semantic scout. By operating on a spatially downscaled proxy, it swiftly isolates semantically relevant temporal regions under MLLM guidance, thereby circumventing the prohibitive computational cost of processing raw high-resolution frames. (2) Adaptive Macro-Micro Inference: This stage mimics the human cognitive process of attentional zooming, where fine-grained details are recruited only when global perception proves ambiguous. Instead of indiscriminately processing high-resolution tokens, the model assesses the suficiency of macro-view via a consensus mechanism. If the global context yields a confident answer, inference terminates early; the model activates high-resolution micro tokens only when necessary to resolve uncertainty. This hierarchical design ensures that computational resources are allocated precisely where needed, achieving substantial inference acceleration without compromising the model’s ability to resolve subtle visual details.

## 3.2 Stage I: Grouped Selection with Macro Proxy

To balance the precision of MLLM-guided token selection with computational eficiency, we operate on a spatially downscaled representation of the input video. This strategy is grounded in the insight that downscaling significantly reduces data volume while preserving the spatial consistency of critical features. By identifying salient regions on this lightweight proxy, we substantially minimize the computational overhead of the selection process. Please refer to Algorithm 1 for the Stage I pseudocode.

![](images/4286455a6957057d254f5ed8d6b9d564f198f52fd66dfe78d15d7fad8db5ee6b.jpg)  
Figure 2: VideoMM Framework. (a) Stage 1: Grouped Selection with Macro-view Proxy eficiently identifies informative regions. (b) Stage 2: Adaptive Macro-Micro Inference assesses semantic suficiency via consensus, integrating Micro-view video only when necessary.

Spatial Downscaling to Macro Proxy. Given an input video $\mathbf { V } \in \mathbb { R } ^ { T \times H \times W \times 3 }$ , we generate a macro proxy $\mathbf { V } ^ { \prime } \in \mathbb { R } ^ { T \times \frac { H } { k } \times \frac { W } { k } \times 3 } \mathrm { \ b y }$ uniformly downscaling spatial dimensions by a factor <sup>2</sup> of �. This transformation reduces the token count by $k ^ { 2 } ;$

$$
\mathbf { X } _ { \mathbf { V } ^ { \prime } } = \mathrm { V i T } ( \mathbf { V } ^ { \prime } ) \in \mathbb { R } ^ { N / k ^ { 2 } \times D }
$$

By replacing V with V<sup>′</sup> for selection, we drastically reduce the number of visual tokens by a factor of $k ^ { 2 }$ which facilitates subsequent MLLM-guided selection while preserving the global layout of salient features.

Grouped Selection via MLLM Guidance. To identify informative tokens within context constraints, we process the macro-view proxy $\mathbf { V } ^ { \prime }$ using an interleaved grouping strategy. We partition the � frames (denoted as $f _ { 1 } , f _ { 2 } , \ldots , f _ { T } )$ into � distinct groups:

$$
{ \mathcal { F } } _ { j } = \{ f _ { i } \mid i \in [ 1 , T ] , i \equiv j { \pmod { G } } \} ,\tag{2}
$$

where $j \in \{ 0 , . . . , G - 1 \}$ . This decomposition ensures that each group functions as a sparse, global representation, enabling the MLLM to form an independent yet comprehensive understanding of the video content within each group.

Algorithm 1 Stage I: Grouped Selection with Macro Proxy   
Require: Video $\mathbf { V } \in \mathbb { R } ^ { T \times H \times W \times 3 } ,$ , Query �, Scale �, Groups �, To  
ken budget �   
Ensure: Macro-level token set ${ \mathcal { T } } ^ { \prime }$   
1. Spatial Downscaling to Macro Proxy   
1: V<sup>′</sup> ← Downscale(V, �) ⊲ ${ \bf { V } } ^ { \prime } \in \mathbb { R } ^ { T \times \frac { H } { k } \times \frac { W } { k } \times \mathfrak { T } }$   
2: $\mathbf { \boldsymbol { X } } _ { \mathbf { \boldsymbol { V } } ^ { \prime } } \gets \mathrm { \nabla { \ V i T } } ( \mathbf { \boldsymbol { V } } ^ { \prime } )$ ⊲ Token count reduced by $k ^ { 2 }$   
3: $\mathcal { T } ^ { \prime }  \emptyset$   
2. Grouped Selection via MLLM Guidance   
4: Interleave frames $f _ { 1 : T }$ into � groups $\{ \mathcal { F } _ { j } \}$   
5: for each group $\mathcal { F } _ { j }$ do   
6: Extract cross-modal $A _ { q , i } ^ { ( l , h ) }$ for visual token �   
7: �<sub>�</sub> = max�<sub>∈</sub> $\mathcal { L } _ { \mathrm { m i d } }$ maxℎ ma ${ \mathfrak { c } } _ { q } A _ { q , i } ^ { ( l , h ) }$   
8: $\begin{array} { r } { s _ { f } \gets \sum _ { i \in f } s _ { i } } \end{array}$ for each frame $\dot { \boldsymbol { f } } \in \mathcal { F } _ { j }$   
9: // Retain top-50% frames   
10: $\mathcal { F } _ { j } ^ { \prime }$ ← TopRatio $( \mathcal { F } _ { j } ,$ �, key = �<sub>�</sub>)   
11: // Select top tokens from retained frames   
12: $\mathcal { T } _ { j } ^ { \prime } \gets \mathrm { T o p K } ( \{ \mathbf { X } _ { V ^ { \prime } } \in \mathcal { F } _ { j } ^ { \prime } \} , B / G , \mathrm { k e y } = s _ { i } )$   
13: $\bar { \mathcal { T } } ^ { \prime }  \mathcal { T } ^ { \prime } \cup \mathcal { T } _ { j } ^ { \prime }$   
14: end for   
15: return ${ \mathcal { T } } ^ { \prime }$

Within each group, we quantify the semantic relevance of visual tokens based on the cross-modal attention weights from the text query $\mathbf { X } _ { Q } \in \mathbb { R } ^ { N _ { Q } \times D }$ . Specifically, for a head ℎ in layer �, let $\mathbf { W } _ { Q } ^ { ( l , h ) } , \mathbf { W } _ { K } ^ { ( l , h ) } \in \mathbb { R } ^ { D \times d _ { h } }$ be the projection matrices, where $d _ { h }$ is the head dimension. The attention weights from text to visual tokens are computed as:

![](images/aabab07161ffd1e45344bf4c051872a3f85e7209c00c44c4d3b909ff00e2b4df.jpg)  
Figure 3: Illustration of Stage II: Adaptive Macro-Micro Inference. Top: In unambiguous cases, the voting mechanism across macro groups reaches a consensus, enabling the model to generate an immediate response. Bottom: When macro-views diverg (indicating ambiguity), VideoMM maps the selected macro regions to micro-level tokens for high-resolution refinement.

$$
\begin{array} { r l } & { \mathbf { A } ^ { ( l , h ) } \in \mathbb { R } ^ { N _ { Q } \times N / k ^ { 2 } } } \\ & { = \mathrm { S o f t m a x } \left( \frac { ( \mathbf { X } _ { Q } \mathbf { W } _ { Q } ^ { ( l , h ) } ) ( \mathbf { X } _ { V ^ { \prime } } \mathbf { W } _ { K } ^ { ( l , h ) } ) ^ { \top } } { \sqrt { d _ { h } } } \right) } \end{array}\tag{3}
$$

where $\mathbf { A } _ { q , i } ^ { ( l , h ) }$ represents the attention weight from the �-th text token to the �-th visual token. We focus on intermediate layers $\mathcal { L } _ { \mathrm { m i d } } ,$ as recent studies on long-context LLM inference [10, 28, 34, 36] suggest that semantic retrieval is predominantly localized in the middle layers. <sup>3</sup> Inspired by the design of the recent worst-case importance indicator [9], we employ maximum aggregation to distill these signals into a robust importance score �<sub>�</sub>

$$
s _ { i } = \operatorname* { m a x } _ { l \in \mathcal { L } _ { \mathrm { m i d } } } \operatorname* { m a x } _ { h } \operatorname* { m a x } _ { q } A _ { q , i } ^ { ( l , h ) }\tag{4}
$$

With the relevance scores �<sub>�</sub> established, we implement a hierarchical selection process to filter redundancy at both the frame and token levels. First, at the frame level, we aggregate the importance of each frame � by summing its constituent token scores, denoted as $\begin{array} { r } { S _ { f } = \sum _ { i \in f } s _ { i } } \end{array}$ . For each group $\mathcal { F } _ { j }$ , we retain the top � fraction of frames (where $\alpha = 0 . 5$ in our experiments) with the highest $S _ { f }$ values, forming a refined subset $\mathcal { F } _ { j } ^ { \prime } . ^ { 4 }$ Subsequently, at the token level, we select the � tokens uniformly from each frame groups to construct the final macro-level token set ${ \mathcal { T } } ^ { \prime } { : }$

$$
\mathcal { T } ^ { \prime } = \bigcup _ { j = 0 } ^ { G - 1 } \operatorname { T o p K } ( \{ \mathbf { X } _ { V ^ { \prime } } \in \mathcal { F } _ { j } ^ { \prime } \} , B / G , \operatorname { k e y } = s _ { i } )\tag{5}
$$

Algorithm 2 Stage II: Adaptive Macro-Micro Inference   
Require: Macro token set T<sup>′</sup> (from Stage I), Query �, Downscale   
factor �, Frame Group ${ \mathcal { F } } _ { j } ,$ , Model M   
Ensure: Final Answer �   
1. Macro-Consensus Verification   
�<sub>response</sub> ← {M (T<sup>′</sup>, �)}   
for each group F do   
Extract a subset $\mathcal { T } _ { j } ^ { \prime } \subseteq \mathcal { T } ^ { \prime }$ belonging to group F<sub>�</sub>   
$a n s _ { j } \gets M ( \mathcal { T } _ { j } , Q )$ ⊲ Generate answer per group   
Add ���<sub>�</sub> to �<sub>response</sub>   
end for   
if All responses in �<sub>response</sub> are consistent then   
return the consensus result from �<sub>response</sub>   
end if   
2. Micro-Detail Recruitment   
Obtain raw frames V<sup>ˆ</sup> associated with ${ \mathcal { T } } ^ { \prime }$   
Encode micro features: $\mathbf { X } _ { \hat { V } } \gets \mathrm { V i T } ( \hat { \mathbf { V } } )$   
// Macro-to-Micro token mapping via Eq. 7   
T ← Gather( $\mathbf { \Delta } _ { \mathbf { X } _ { \hat { V } } } , \mathcal { T } ^ { \prime } )$   
return M(T, �)

## 3.3 Stage II: Adaptive Macro-Micro Inference

While Stage I identifies where to look, Stage II determines how deep to look. As illustrated in Figure 3, this stage is grounded in the principle of computational eficiency: most video queries can be answered via global semantic cues (Macro), while only a fraction require high-resolution scrutiny (Micro). To exploit this, VideoMM employs an adaptive granularity inference path, prioritizing lowcost macro inference and escalating to high-cost micro reasoning only when ambiguity is detected. Please refer to Algorithm 2 for the Stage II pseudocode.

Macro-Consensus Verification. We employ a Consensus-Based Voting mechanism [30] to evaluate the suficiency of the macroview. Each group $\mathcal { F } _ { j }$ , alongside the jointly selected macro token set ${ \mathcal { T } } ^ { \prime }$ (treated as an additional independent group), processed independently, acts as a weak estimator generating a preliminary response. We posit that prediction invariance across disjoint temporal subsets serves as a robust proxy for confidence. If all independent groups converge on a consensus, macro-level semantics are deemed sufi-$\mathrm { c i e n t } ^ { 5 }$ . In such cases, the model terminates inference early using the consensus result, bypassing the overhead of micro-level processing. 6

Micro-Detail Recruitment. When macro-views diverge—indicating ambiguity and the need for finer detail—, VideoMM activates microlevel refinement. This process necessitates encoding the relevant raw video frames into micro tokens:

$$
\mathbf { X } _ { \hat { V } } = \mathrm { V i T } ( { \hat { \mathbf { V } } } ) \in \mathbb { R } ^ { \alpha N \times D }\tag{6}
$$

where $\hat { \mathbf { V } }$ denotes the raw video input constructed from the frames selected in the first stage, consisting of only the �-fraction of the original frames. This efectively controls the computational overhead of ViT re-encoding for these micro tokens. We then map the critical macro tokens ${ \check { \mathcal { T } } } ^ { \prime }$ selected in Stage I back to their corresponding micro-level representations $\mathcal { T }$ . Specifically, for any macro token in $\mathbf { \bar { \mathbf { \nabla } } } _ { \mathcal { T } ^ { \prime } }$ identified by $( t , h ^ { \prime } , w ^ { \prime } )$ in frame and spatial coordinates, it corresponds to a micro-token set from the newly encoded $\mathbf { X } _ { \hat { V } }$ within the same spatial region:

$$
\mathcal { T } _ { ( t , h , w ) } = \left\{ \mathbf { X } _ { \hat { V } } [ t , h , w ] \ | \begin{array} { l } { \left\{ h ^ { \prime } k \leq h < ( h ^ { \prime } + 1 ) k , \right. } \\ { \left. w ^ { \prime } k \leq w < ( w ^ { \prime } + 1 ) k \right\} . } \end{array} \right.\tag{7}
$$

The complete micro token set $\mathcal { T }$ is then the union of these subsets: $\mathcal { T } = \bigcup \mathcal { T } _ { ( t , h , w ) }$ , which is then fed into the models for the definitive response. This paradigm balances eficiency and accuracy by using the macro stage as a filter for noise and the micro stage as a specialized solver for detail.

## 3.4 Eficiency Analysis

To quantify the eficiency gains of VideoMM, we analyze its asymptotic complexity relative to standard MLLM inference. We focus on the self-attention mechanism, which constitutes the primary computational bottleneck in long-video understanding. Let � denote the number of visual tokens in raw high-resolution video. Given that � scales significantly with video duration, we analyze complexity with respect to �, treating text sequence length as negligible.

Complexity Derivation. Standard global attention generally scales quadratically, i.e., $O ( N ^ { 2 } )$ . VideoMM optimizes this via three key parameters: the macro downscaling factor $k ,$ the number of temporal groups $G ,$ and the macro-selection budget � (introduced in Stage I) combined with the macro-consensus probability � (governed by Stage II). The expected computational cost of VideoMM, $\mathbb { E } [ \Omega _ { \mathrm { { O u r s } } } ]$ is the sum of two components:

(1) Macro Processing (Stage I & Stage II-Consensus): Both Grouped Selection (Stage I) and Macro-Consensus Verification (Stage II) operate on shared Macro Proxy. Here, token count is reduced to $N / k ^ { 2 }$ and divided into � groups. Since attention is restricted within these groups for relevance estimation $( s _ { i } )$ and consensus voting, the computational overhead is: $\begin{array} { r } { G \cdot \left( \frac { N / k ^ { 2 } } { G } \right) ^ { 2 } = \frac { N ^ { 2 } } { G k ^ { 4 } } } \end{array}$

(2) Micro Processing (Stage II-Refinement): This term corresponds to the conditional activation of the Micro-Detail Recruitment phase in Stage II. Let $\rho = ( \boldsymbol { B } \cdot \boldsymbol { k } ^ { 2 } ) / N$ denote the ratio of selected high-resolution tokens. Crucially, this computation is only incurred when the consensus mechanism fails. Let $\beta$ denote the probability of successfully reaching a consensus (i.e., early exit). The expected cost is: $( 1 - \beta ) ( \rho N ) ^ { 2 }$

Summing these terms, the total expected complexity is:

$$
\mathbb { E } [ \Omega _ { \mathrm { O u r s } } ] = O \left( N ^ { 2 } \left[ \underbrace { \frac { 1 } { G k ^ { 4 } } } _ { \mathrm { M a c r o ~ P r o x y } } + \underbrace { ( 1 - \beta ) \rho ^ { 2 } } _ { \mathrm { M i c r o ~ R e f i n e m e n t } } \right] \right) .\tag{8}
$$

Equation 8 demonstrates the transformation of quadratic dependency into an eficient form via two benefits: (1) Grouping and Downscaling: The term $\textstyle { \frac { 1 } { G k ^ { 4 } } }$ represents the structural cost reduction, where � and � provide quartic and linear complexity drops, respectively. (2) Adaptive Inference: The term $( 1 - \beta ) \rho ^ { 2 }$ captures the gain from adaptive eficiency. Empirically, high early-exit rates $( \beta \approx 7 0 \% )$ and extreme sparsity $( \rho < 3 \% )$ confine heavy computation strictly to ambiguous instances, preserving detail with minimal overhead.

## 4 Experiments

## 4.1 Settings

Models. To demonstrate the universality of our framework, we conduct evaluations on three leading open-source MLLMs: Qwen2.5- VL-7B [2] and GLM-4.1-VL-9B-Thinking [27] support 64K context windows, while Qwen3-VL-8B [26] extends to 256K natively. For GLM-4.1-VL-9B-Thinking, we disable its thinking mode to force direct responses and keep evaluation costs manageable. These models were selected as state-of-the-art for long-context video understanding; furthermore, their dynamic-resolution ViTs naturally support varied input resolutions, inherently aligning with our hierarchical macro-micro paradigm.

Baselines. We benchmark our approach against three representative token reduction methods, categorized by their computational characteristics: (i) lightweight encoder-driven methods and (ii) heavyweight MLLM-driven methods. For the lightweight category, we evaluate VisionZip [38], which utilizes ViT attention for selection, and VidCom [22], which employ token uniqueness for compression. For the heavyweight category, we utilize FlexSelect [42] as a representative baseline. To ensure a fair comparison and strictly control computational costs, we adopt a unified token budget across all baseline categories. Specifically, for both lightweight encoderdriven methods [22, 38] and the heavyweight MLLM-driven FlexSelect [42], we retain a fixed 8,192 visual tokens, following common practice. For VideoMM $( G = 4 , k = 2 )$ , we align our Micro token budget with this unified 8,192 limit; consequently, the Macro budget is set to 2,048 (8, 192/4).

Table 1: Performance comparison on video understanding benchmarks. Throughput denotes inference speed in samples per minute. The best two results are bolded. VideoMM demonstrates leading performance in both accuracy and throughput, whereas other methods typically prioritize one over the other.
<table><tr><td rowspan="2">Method</td><td colspan="2">LongVideoBench</td><td colspan="2">VideoMME</td><td colspan="2">LVBench</td><td colspan="2">Average</td></tr><tr><td>Accuracy</td><td>Throughput</td><td>Accuracy</td><td>Throughput</td><td>Accuracy</td><td>Throughput</td><td>Accuracy</td><td>Throughput</td></tr><tr><td colspan="9">Qwen2.5-VL-7B</td></tr><tr><td>Vanilla</td><td>57.67</td><td>0.77</td><td>61.56</td><td>0.85</td><td>39.57</td><td>0.65</td><td>52.93</td><td>0.76</td></tr><tr><td>VisionZip</td><td>58.23 (↑1.0%)</td><td>3.63 (4.7x)</td><td>62.14 (↑0.9%)</td><td>3.84 (4.5x)</td><td>41.38 (↑4.6%)</td><td>3.10 (4.8x)</td><td>53.92 (↑1.9%)</td><td>3.52 (4.6x)</td></tr><tr><td>VidCom</td><td>54.60 (↓5.3%)</td><td>4.27 (5.5x)</td><td>61.56 (-0.0%)</td><td>4.42 (5.2x)</td><td>40.80 (↑3.1%)</td><td>4.32 (6.6x)</td><td>52.32 (↓1.2%)</td><td>4.34 (5.7x)</td></tr><tr><td>Flexselect</td><td>64.10 (↑11.1%)</td><td>1.83 (2.4x)</td><td>68.85 (↑11.8%)</td><td>1.84 (2.2x)</td><td>51.45 (↑30.0%)</td><td>1.50 (2.3x)</td><td>61.47 (↑16.1%)</td><td>1.72 (2.3x)</td></tr><tr><td>VideoMM</td><td>64.70 (↑12.2%)</td><td>5.16 (6.7x)</td><td>68.41 (↑11.1%)</td><td>5.47 (6.4x)</td><td>50.94 (↑28.7%)</td><td>4.20 (6.5x)</td><td>61.35 (↑15.9%)</td><td>4.94 (6.5x)</td></tr><tr><td colspan="9">GLM-4.1V-9B</td></tr><tr><td>Vanilla</td><td>58.98</td><td>0.51</td><td>66.67</td><td>0.58</td><td>47.90</td><td>0.42</td><td>57.85</td><td>0.50</td></tr><tr><td>VisionZip</td><td>55.35 (↓6.2%)</td><td>4.10 (8.0x)</td><td>59.55 (↓10.7%)</td><td>4.30 (7.4x)</td><td>36.41 (↓24.0%)</td><td>3.60 (8.6x)</td><td>50.44 (↓12.8%)</td><td>4.00 (8.0x)</td></tr><tr><td>VidCom</td><td>51.68 (↓12.4%)</td><td>3.50 (6.9x)</td><td>56.48 (↓15.3%)</td><td>3.70 (6.4x)</td><td>37.70 (↓21.3%)</td><td>2.96 (7.0x)</td><td>48.62 (↓16.0%)</td><td>3.39 (6.8x)</td></tr><tr><td>Flexselect VideoMM</td><td>60.66 (↑2.8%)</td><td>1.47 (2.9x)</td><td>65.15 (↓2.3%)</td><td>1.57 (2.7x)</td><td>50.16 (↑4.7%)</td><td>1.28 (3.0x)</td><td>58.66 (↑1.4%)</td><td>1.44 (2.9x)</td></tr><tr><td></td><td>64.32 (↑9.1%)</td><td>3.77 (7.4x)</td><td>67.37 (↑1.0%)</td><td>4.21 (7.3x)</td><td>49.52 (↑3.4%)</td><td>2.87 (6.8x)</td><td>60.40 (↑4.4%)</td><td>3.62 (7.2x)</td></tr><tr><td colspan="9">Qwen3-VL-8B</td></tr><tr><td>Vanilla</td><td>66.94</td><td>0.85</td><td>71.78</td><td>0.97</td><td>54.1</td><td>0.70</td><td>64.27</td><td>0.84</td></tr><tr><td>VisionZip VidCom</td><td>57.97 (↓13.4%)</td><td>2.47 (2.9x)</td><td>65.55 (↓8.7%)</td><td>2.43 (2.5x)</td><td>40.22 (↓25.7%)</td><td>1.92 (2.7x)</td><td>54.58 (↓15.1%)</td><td>2.27 (2.7x)</td></tr><tr><td>Flexselect</td><td>59.16 (↓11.6%)</td><td>4.39 (5.2x)</td><td>64.59 (↓10.0%)</td><td>3.04 (3.1x)</td><td>44.03 (↓18.6%)</td><td>2.39 (3.4x)</td><td>55.93 (↓13.0%)</td><td>3.27 (3.9x)</td></tr><tr><td>VideoMM</td><td>67.17 (↑0.3%)</td><td>1.30 (1.5x)</td><td>72.19 (↑0.6%)</td><td>1.32 (1.4x)</td><td>55.20 (↑2.0%)</td><td>0.97 (1.4x)</td><td>64.85 (↑0.9%)</td><td>1.20 (1.4x)</td></tr><tr><td></td><td>67.54 (↑0.9%)</td><td>3.69 (4.3x)</td><td>71.67 (↓0.2%)</td><td>3.59 (3.7x)</td><td>54.36 (↑0.5%)</td><td>2.50 (3.6x)</td><td>64.52 (↑0.4%)</td><td>3.26 (3.9x)</td></tr></table>

Table 2: Comparison on long-video benchmarks under comparable throughput (FlexSelect reduced to 256 frames; Acc.: Accuracy (%); Thpt.: Throughput (samples/min))
<table><tr><td rowspan="2">Method</td><td colspan="2">LongVideoBench</td><td colspan="2">LVBench</td><td colspan="2">Average</td></tr><tr><td>Acc.</td><td>Thpt.</td><td>Acc.</td><td>Thpt.</td><td>Acc.</td><td>Thpt.</td></tr><tr><td colspan="7">Qwen2.5-VL-7B</td></tr><tr><td>Vanilla (512)</td><td>57.67</td><td>0.77</td><td>39.57</td><td>0.65</td><td>48.62</td><td>0.71</td></tr><tr><td>FlexSelect (256) VideoMM (512)</td><td>62.30 64.70</td><td>3.27 5.16</td><td>46.93 50.94</td><td>2.87 4.20</td><td>54.62 57.82</td><td>3.07 4.68</td></tr><tr><td colspan="7">GLM-4.1V-9B</td></tr><tr><td>Vanilla (512)</td><td>58.98</td><td>0.51</td><td>47.90</td><td>0.42</td><td>53.44</td><td>0.47</td></tr><tr><td>FlexSelect (256)</td><td>63.13</td><td>3.35</td><td>48.16</td><td>2.97</td><td>55.65</td><td>3.16</td></tr><tr><td>VideoMM (512)</td><td>64.32</td><td>3.77</td><td>49.52</td><td>2.87</td><td>56.92</td><td>3.32</td></tr><tr><td colspan="7">Qwen3-VL-8B</td></tr><tr><td>Vanilla (512)</td><td>66.94</td><td>0.85</td><td>54.10</td><td>0.70</td><td>60.52</td><td>0.78</td></tr><tr><td>FlexSelect (256)</td><td>65.82</td><td>3.70</td><td>50.36</td><td>3.06</td><td>58.09</td><td>3.38</td></tr><tr><td>VideoMM (512)</td><td>67.54</td><td>3.69</td><td>54.36</td><td>2.50</td><td>60.95</td><td>3.10</td></tr></table>

Benchmarks. Our evaluation employs two widely adopted longvideo benchmarks: LongVideoBench[33], focusing on fine-grained retrieval and complex reasoning within long-context videos, and LVBench[29] for testing the capability over hour-long video analysis. Furthermore, we assess multi-scale temporal understanding using VideoMME[12]. We sample videos uniformly at 1 FPS, up to a maximum of 512 frames. All evaluations use the LMMs-Eval framework[40] following standard protocols.

![](images/c72196f9f270a1b949fd1070531e0adf5b888cb667134a16c34b618375b4fece.jpg)  
Figure 4: Impact of input frame counts on LongVideoBench using Qwen3-VL-8B. The solid lines represent accuracy (left axis), while the transparent bars represent throughput (right axis).

## 4.2 Main Results

Table 1 presents a comparative evaluation of accuracy and through put for VideoMM against various token compression methods across multiple models and benchmarks. Overall, Encoder-driven methods excel in throughput, whereas MLLM-driven methods achieve higher accuracy. In contrast, VideoMM demonstrates superiority in both accuracy and eficiency, enabled by its novel macro-micro perceptual granularity. Detailed performance on the three benchmarks are provided in the appendix.

Comparison with Encoder-Driven Methods.

By leveraging macro-level MLLM-guided token selection, VideoMM significantly outperforms lightweight encoder-driven methods (e.g., VisionZip, VidCom) in both accuracy and throughput. For instance,

![](images/b11d3468185185aa4db53856272ee79af72d7eb618d1522d5331adc0f62ce636.jpg)  
Figure 5: Accuracy-eficiency trade-ofs across VideoMM fam ily variants (Qwen3-VL-8B, LongVideoBench). Legend colors from dark to light correspond to frame counts of 512, 386, 256, and 128, respectively. The VideoMM family establishes a superior Pareto frontier, validating the intrinsic eficacy of the proposed macro-micro paradigm.

on Qwen3-VL-8B and Qwen2.5-VL-7B, VideoMM attains average scores of 64.52 and 61.35 across three benchmarks, substantially outperforming VidCom’s 55.93 and 52.32, respectively.Moreover, even in terms of throughput—where encoder-driven methods typ ically excel—VideoMM achieves superior performance due to its macro-micro paradigm. On the same models, VideoMM reaches average throughputs of 3.26 and 4.94 samples/min, delivering comparable or superior overall eficiency compared to VidCom’s 3.27 and 4.34.

Comparison with MLLM-Driven methods. By employing Adap tive Macro-Micro Inference, VideoMM achieves a substantial boost in throughput while preserving accuracy on par with FlexSelect. For example, on Qwen3 and GLM, VideoMM achieves average scores of 64.52 and 60.40 across three benchmarks, close to FlexSelect’s 64.85 and 58.66, respectively. Crucially, VideoMM demonstrates a significant throughput advantage: on Qwen3, it reaches 3.26 samples/min—surpassing FlexSelect’s 1.20—which corresponds to a 3.9× speedup over the vanilla. This gap is even more pronounced on GLM 4.1V-9B, where VideoMM achieves 3.62 samples/min compared to FlexSelect’s 1.44, yielding a 7.2× speedup over vanilla. In real-world long-video understanding, balancing accuracy and throughput of ten necessitates adjusting input frame counts to meet deployment constraints. Thus, in Table 2, we focus specifically on two long video benchmarks (LongVideoBench and LVBench), employing frame reduction on FlexSelect (to 256 frames) to compare accuracy under iso-throughput conditions. Overall, our method consistently achieves higher accuracy at comparable throughput. For instance, with Qwen2.5-VL-7B, VideoMM outperforms FlexSelect by 4.7 point in average accuracy (59.32 vs. 54.62) while simultaneously increasing throughput. This advantage extends to GLM-4.1V-9B and Qwen3-VL-8B, where our method maintains higher accuracy at comparable inference speeds.

Table 3: Early-exit statistics � (LongVideoBench, Qwen3-VL-8B).
<table><tr><td>Frame count</td><td>128</td><td>256</td><td>384</td><td>512</td></tr><tr><td>VideoMM⁻</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td></tr><tr><td>VideoMM</td><td>65.7%</td><td>69.6%</td><td>70.8%</td><td>73.3%</td></tr><tr><td>VideoMM⁺</td><td>88.6%</td><td>89.3%</td><td>88.6%</td><td>88.6%</td></tr></table>

## 4.3 Analysis Experiments

Detail discussion of Adaptive Computation. VideoMM employs an adaptive computation mechanism within a macro-micro hierarchical framework, dynamically allocating computational resources based on sample complexity. To investigate the accuracy-eficiency trade-of from this mechanism, we instantiate two variants by modulating the adaptive strategy <sup>7</sup>: (1) VideoMM<sup>+</sup> (Eficiency-Prioritized): This variant adopts a majority-voting strategy for Macro-Consensus Verification, allowing early exit upon majority agreement. This significantly increases the probability of early termination at the costeficient macro level compared to the standard VideoMM, which requires unanimous consensus. (2) VideoMM<sup>−</sup> (Non-Adaptive Ablation): Conversely, this variant enforces all samples to undergo fine-grained inference at the micro level, completely disabling the adaptive computation mechanism. Figure 5 illustrates the accuracyeficiency trade-of across varying frame inputs, while Table 3 details the early-exit statistics �. Overall, as we transition from VideoMM<sup>−</sup> to VideoMM and finally VideoMM<sup>+</sup>, both the early-exit probability � and inference speed progressively increase, though potentially at the cost of marginal accuracy drops. Considering both accuracy and eficiency, both adaptive variants (VideoMM and VideoMM<sup>+</sup>) yield a superior trade-of compared to the nonadaptive VideoMM<sup>−</sup>, occupying the optimal upper-right region of the plot. Between the two, the choice depends on specific application constraints. For instance, with 512-frame inputs, VideoMM prioritizes higher accuracy (67.54) with moderate throughput (3.69), whereas VideoMM<sup>+</sup> achieves significantly higher throughput (4.90) with slightly lower accuracy (66.87). Overall, the VideoMM family establishes a new optimal frontier on accuracy-throughput land scape. This success validates the intrinsic eficacy of macro-micro perceptual granularity paradigm, ofering a novel direction that moves beyond the prevailing model-centric perspective.

Influence of Input Frames. As shown in Figure 4, we further evaluate the performance of VideoMM on the challenging LongVideoBench benchmark using varying input frame count to simulate diverse real-world deployment constraints. Typically, increasing the number of frames improves long-video understanding accuracy but incurs a substantial token overhead, thereby significantly reducing throughput. However, VideoMM efectively mitigates the trade-of between eficiency and accuracy, maintaining superior accuracy while markedly improving throughput across all input frame settings. For instance, with 512 frames, VideoMM attains a superior accuracy of 67.54% and achieves speedups of 4.3× over vanilla and 2.8× over FlexSelect. Notably, the throughput of VideoMM at 512 frames is comparable to that of FlexSelect at only 256 frames. Consequently, in practical applications, Video-MM enables the eficient

Table 4: Impact of downscale factor �. Evaluated on LongVideoBench using Qwen3-VL-8B. Throughput is measured in samples per min. Best results are highlighted in bold.
<table><tr><td>Method</td><td>Accuracy</td><td>Throughput</td></tr><tr><td>Vanilla</td><td>66.94</td><td>0.85</td></tr><tr><td>VisionZip VidCom Flexselect</td><td>57.97 (↓13.4%) 59.16 (↓11.6%) 67.17 (↑0.3%)</td><td>2.47 (2.9x) 4.39 (5.2x) 1.30 (1.5x)</td></tr><tr><td>VideoMM, k=2 VideoMM, k=3</td><td>67.54 (↑0.9%) 65.89 (↓1.6%)</td><td>3.69 (4.3x) 6.74 (7.9x)</td></tr></table>

processing of longer video inputs, thereby unlocking superior understanding capabilities from these extended contexts.

Influence of Downscale factor �. In our main experiments, we set the downscale factor to $k = 2 ,$ reducing the first-stage macro proxy tokens to one-quarter of the original count to maintain high accuracy. Here, we investigate a more aggressive downscaling strategy by setting � = 3—which reduces tokens to 1/9—to evaluate the impact on the accuracy-throughput trade-of. As shown in Table 4, increasing downscale factor � = 3 naturally incurs a slight accuracy cost compared to � = 2 (65.89 vs. 67.54). However, this yields a substantial gain in computational speed, nearly doubling the throughput from 3.69 to 6.74 samples per second. Crucially, this aggressive � = 3 configuration establishes a superior Pareto frontier for highly eficient long-video understanding. When benchmarked against lightweight Encoder-driven methods (e.g., VisionZip and VidCom), VideoMM with � = 3 demonstrates comprehensive dominance: it not only retains higher accuracy (65.89 vs. ≈64.0) but also achieves drastically higher throughput (7.9× vs. ≈2.2×). The extreme downscaling strategy remains highly efective primarily because our Stage II consensus mechanism acts as a reliable safety net; when severe perceptual ambiguity is induced by the � = 3 compression, the resulting macro-level divergence typically triggers a fallback to micro-level refinement, mitigating the risk of catastrophic reasoning failures. This result confirms the robustness of our macro-proxy mechanism even under aggressive downscaling. Consequently, the downscale factor serves as a practical control knob, allowing users to seamlessly transition between high-precision and high-speed regimes to meet diverse deployment constraints.

Task-Aware Behavior of Adaptive Computation. To further investigate the dynamics of our macro-micro hierarchical framework, we quantitatively analyze the correlation between the macrolevel early-exit probability (�) and specific task categories on the LongVideoBench. Following the task taxonomy introduced in the original benchmark [33], the adaptive mechanism exhibits distinct routing preferences depending on the semantic requirements of the query. For perception-centric tasks (Level-1) characterized by explicit temporal anchors—such as Text-Referred Event (T2E) and Object-Referred Event (O2E)—VideoMM triggers early exits at a consistently high rate (ranging from 76% to 83%). Circumventing the fine-grained micro level in these localized tasks significantly reduces inference latency. This phenomenon suggests that coarsegrained proxy tokens are highly efective for extracting localized semantics, whereas enforcing full-resolution inference may inadvertently cause the model to over-assimilate irrelevant visual noise. Conversely, for relation-centric tasks (Level-2) that necessitate global temporal logic—such as Sequence of Scenes (SSS) and Object Before/After Object (O3O)—the early-exit rate decreases to approximately 60%. In these complex scenarios, the macro-level consensus naturally diverges. This divergence inherently stems from the fact that the spatial downsampling within the macro proxy often obscures subtle visual state transitions and fine-grained action boundaries essential for stringent temporal deduction. Recognizing this semantic deficit, the framework conservatively routes the samples to the micro level for comprehensive tracking, ensuring lossless visual features for long-range dependencies. Consequently, the macro-micro paradigm functions as an implicit task-aware router: it accelerates and filters noise for localized queries while prudently preserving high-resolution computational capacity for long-span reasoning. From a system optimization perspective, this dynamic routing profoundly regulates the inference memory footprint.

Table 5: Quantitative analysis of early-exit behaviors across representative tasks on LongVideoBench. The macro-micro paradigm naturally exhibits higher early-exit rates for localized perception tasks, while conservatively maintaining deep inference (lowest �) for complex relation tasks.
<table><tr><td>Task Level</td><td>Task</td><td>Early-Exit Rate (β)</td></tr><tr><td rowspan="3">Perception (Level-1)</td><td>T2O (Text-Obj)</td><td>76.32%</td></tr><tr><td>T2E (Text-Event)</td><td>83.08%</td></tr><tr><td>O2E (Obj-Event)</td><td>82.76%</td></tr><tr><td rowspan="3">Relation (Level-2)</td><td>T3O (Text-Obj)</td><td>63.51%</td></tr><tr><td>SSS (Scene Seq)</td><td>62.89%</td></tr><tr><td>O3O (Obj-Obj)</td><td>60.61%</td></tr></table>

## 5 Conclusion

In this work, we identified a fundamental dilemma in eficient longform video understanding: existing token reduction methods rely on model-centric downsizing for token selection, failing to reconcile the trade-of between selection precision and computational overhead. To address this, we proposed VideoMM, a novel framework that orchestrates a paradigm shift from model scale to adaptive perceptual granularity. By adhering to a coarse-to-fine philosophy, VideoMM decouples semantic filtering from dense processing via a cost-efective Macro Proxy, while the Adaptive Macro-Micro Inference mechanism dynamically recruits high-fidelity tokens only when necessary.Extensive evaluations on LongVideoBench and other benchmarks demonstrate that VideoMM not only breaks the eficiency bottleneck—achieving a 6.13× speedup—but also improves accuracy by over 7% compared to full-context baselines. Furthermore, our in-depth analysis of multiple variants under the Macro-Micro design demonstrates that the VideoMM family estab lishes a superior Pareto frontier compared to prior arts, empirically confirming the inherent efectiveness of the hierarchical perceptual granularity paradigm. We believe this promising perspective opens a new avenue for eficient video understanding, facilitating the real-world deployment of multimodal systems in long-context scenarios.

## References

[1] Ali K. AlShami, Ryan Rabinowitz, Khang Lam, Yousra Shleibik, Melkamu Mersha, Terrance Boult, and Jugal Kalita. 2024. SMART-vision: survey of modern action recognition techniques in vision. Multimedia Tools and Applications 84, 27 (Dec. 2024), 32705–32776. doi:10.1007/s11042-024-20484-5

[2] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. 2025. Qwen2.5-VL Technical Report. arXiv preprint arXiv:2502.13923 (2025).

[3] Daniel Bolya, Cheng-Yang Fu, Xiaoliang Dai, Peizhao Zhang, Christoph Feichtenhofer, and Judy Hofman. 2023. Token Merging: Your ViT but Faster. In International Conference on Learning Representations.

[4] Zefan Cai, Yichi Zhang, Bofei Gao, Yuliang Liu, Tianyu Liu, Keming Lu, Wayne Xiong, Yue Dong, Baobao Chang, Junjie Hu, and Xiao Wen. 2024. Pyramidkv: Dynamic kv cache compression based on pyramidal information funneling. arXiv preprint arXiv:2406.02069 (2024).

[5] Boyu Chen, Zhengrong Yue, Siran Chen, Zikang Wang, Yang Liu, Peng Li, and Yali Wang. 2025. LVAgent: Long Video Understanding by Multi-Round Dynamica Collaboration of MLLM Agents. arXiv:2503.10200 [cs.CV] https://arxiv.org/abs/ 2503.10200

[6] Charlie Chen, Sebastian Borgeaud, Geofrey Irving, Jean-Baptiste Lespiau, Laurent Sifre, and John Jumper. 2023. Accelerating Large Language Model Decoding with Speculative Sampling. arXiv:2302.01318 [cs.CL] https://arxiv.org/abs/2302. 01318

[7] Liang Chen, Haozhe Zhao, Tianyu Liu, Shuai Bai, Junyang Lin, Chang Zhou, and Baobao Chang. 2024. An Image is Worth 1/2 Tokens After Layer 2: Plug-and-Play Inference Acceleration for Large Vision-Language Models. arXiv:2403.06764 [cs.CV] https://arxiv.org/abs/2403.06764

[8] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, Bin Li, Ping Luo, Tong Lu, Yu Qiao, and Jifeng Dai. 2024. Intern VL: Scaling up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 24185–24198. doi:10.1109/ CVPR52733.2024.02283

[9] Yuan Feng, Haoyu Guo, JunLin Lv, S Kevin Zhou, and Xike Xie. 2025. Taming the Fragility of KV Cache Eviction in LLM Inference. arXiv preprint arXiv:2510.13334 (2025).

[10] Yuan Feng, Junlin Lv, Yukun Cao, Xike Xie, and S Kevin Zhou. 2024. Ada-kv: Optimizing kv cache eviction by adaptive budget allocation for eficient llm inference. arXiv preprint arXiv:2407.11550 (2024).

[11] Yuan Feng, Junlin Lv, Yukun Cao, Xike Xie, and S Kevin Zhou. 2025. Identify critical kv cache in llm inference from an output perturbation perspective. arXiv preprint arXiv:2502.03805 (2025).

[12] Chaoyou Fu, Yuhan Dai, Yongdong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, et al. 2025. Video mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis. In CVPR.

[13] Yu Fu, Zefan Cai, Abedelkadir Asi, Wayne Xiong, Yue Dong, and Wen Xiao. 2024. Not All Heads Matter: A Head-Level KV Cache Compression Method with Integrated Retrieval and Reasoning. arXiv preprint arXiv:2410.19258 (2024).

[14] Yicheng Ji, Jun Zhang, Heming Xia, Jinpeng Chen, Lidan Shou, Gang Chen, and Huan Li. 2025. Specvlm: Enhancing speculative decoding of video llms via verifier-guided token pruning. In Proceedings ofthe 2025 Conference on Empirical Methods in Natural Language Processing. 7216–7230.

[15] Huiqiang Jiang, Yucheng Li, Chengruidong Zhang, Qianhui Wu, Xufang Luo, Surin Ahn, Zhenhua Han, Amir H. Abdi, Dongsheng Li, Chin-Yew Lin, Yuqing Yang, and Lili Qiu. 2024. MInference 1.0: Accelerating Pre-filling for Long-Context LLMs via Dynamic Sparse Attention. In The Thirty-eighth Annual Conference on Neural Information Processing Systems. https://openreview.net/forum?id= fPBACAbqSN

[16] Yaniv Leviathan, Matan Kalman, and Yossi Matias. 2023. Fast Inference from Transformers via Speculative Decoding. arXiv:2211.17192 [cs.LG] https://arxiv. org/abs/2211.17192

[17] Yuhong Li, Yingbing Huang, Bowen Yang, Bharat Venkitesh, Acyr Locatelli, Hanchen Ye, Tianle Cai, Patrick Lewis, and Deming Chen. 2024. SnapKV: LLM Knows What You are Looking for Before Generation. In The Thirty-eighth Annual Conference on Neural Information Processing Systems. https://openreview.net/ forum?id=poE54GOq2l

[18] Yucheng Li, Huiqiang Jiang, Chengruidong Zhang, Qianhui Wu, Xufang Luo, Surin Ahn, Amir H. Abdi, Dongsheng Li, Jianfeng Gao, Yuqing Yang, and Lili Qiu. 2025. MMInference: Accelerating Pre-filling for Long-Context VLMs via Modality-Aware Permutation Sparse Attention. arXiv:2504.16083 [cs.CV] https: //arxiv.org/abs/2504.16083

[19] Huabin Liu, Filip Ilievski, and Cees G. M. Snoek. 2025. Commonsense Video Question Answering through Video-Grounded Entailment Tree Reasoning. In

Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 3262–3271.

[20] Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. 2024. LLaVA-NeXT: Improved reasoning, OCR, and world knowledge. https://llava-vl.github.io/blog/2024-01-30-llava-next/

[21] Xiaolong Liu, Yao Hu, Song Bai, Fei Ding, Xiang Bai, and Philip H. S. Torr. 2021. Multi-Shot Temporal Event Localization: A Benchmark. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 12596–12606.

[22] Xuyang Liu, Yiyu Wang, Junpeng Ma, and Linfeng Zhang. 2025. Video Compression Commander: Plug-and-Play Inference Acceleration for Video Large Language Models. In Proceedings ofthe 2025 Conference on Empirical Methods in Natural Language Processing. Association for Computational Linguistics, Suzhou, China, 1910–1924. doi:10.18653/v1/2025.emnlp-main.98

[23] Jiaming Tang, Yilong Zhao, Kan Zhu, Guangxuan Xiao, Baris Kasikci, and Song Han. 2024. Quest: Query-Aware Sparsity for Eficient Long-Context LLM Infer ence.

[24] Yunlong Tang, Jing Bi, Siting Xu, Luchuan Song, Susan Liang, Teng Wang, Daoan Zhang, Jie An, Jingyang Lin, Rongyi Zhu, Ali Vosoughi, Chao Huang, Zeliang Zhang, Pinxin Liu, Mingqian Feng, Feng Zheng, Jianguo Zhang, Ping Luo, Jiebo Luo, and Chenliang Xu. 2025. Video Understanding with Large Language Models: A Survey. IEEE Transactions on Circuits and Systems for Video Technology (2025), 1–1. doi:10.1109/TCSVT.2025.3566695

[25] Keda Tao, Can Qin, Haoxuan You, Yang Sui, and Huan Wang. 2025. DyCoke: Dynamic Compression of Tokens for Fast Video Large Language Models. In Proceedings ofthe Computer Vision and Pattern Recognition Conference. 18992– 19001.

[26] Qwen Team. 2025. Qwen3 Technical Report. arXiv:2505.09388 [cs.CL] https: //arxiv.org/abs/2505.09388

[27] V Team, Wenyi Hong, Wenmeng Yu, Xiaotao Gu, Guo Wang, Guobing Gan, Haomiao Tang, Jiale Cheng, Ji Qi, Junhui Ji, Lihang Pan, Shuaiqi Duan, Weihan Wang, Yan Wang, Yean Cheng, Zehai He, Zhe Su, Zhen Yang, Ziyang Pan, Aohan Zeng, Baoxu Wang, Bin Chen, Boyan Shi, Changyu Pang, Chenhui Zhang, Da Yin, Fan Yang, Guoqing Chen, Jiazheng Xu, Jiale Zhu, Jiali Chen, Jing Chen, Jinhao Chen, Jinghao Lin, Jinjiang Wang, Junjie Chen, Leqi Lei, Letian Gong, Leyi Pan, Mingdao Liu, Mingde Xu, Mingzhi Zhang, Qinkai Zheng, Sheng Yang, Shi Zhong, Shiyu Huang, Shuyuan Zhao, Siyan Xue, Shangqin Tu, Shengbiao Meng, Tianshu Zhang, Tianwei Luo, Tianxiang Hao, Tianyu Tong, Wenkai Li, Wei Jia, Xiao Liu, Xiaohan Zhang, Xin Lyu, Xinyue Fan, Xuancheng Huang, Yanling Wang, Yadong Xue, Yanfeng Wang, Yanzi Wang, Yifan An, Yifan Du, Yiming Shi, Yiheng Huang, Yilin Niu, Yuan Wang, Yuanchang Yue, Yuchen Li, Yutao Zhang, Yuting Wang, Yu Wang, Yuxuan Zhang, Zhao Xue, Zhenyu Hou, Zhengxiao Du, Zihan Wang, Peng Zhang, Debing Liu, Bin Xu, Juanzi Li, Minlie Huang, Yuxiao Dong, and Jie Tang. 2025. GLM-4.5V and GLM-4.1V-Thinking: Towards Versatile Multimodal Reasoning with Scalable Reinforcement Learning. arXiv:2507.01006 [cs.CV] https://arxiv.org/abs/2507.01006

[28] Jiahui Wang, Zuyan Liu, Yongming Rao, and Jiwen Lu. 2025. SparseMM: Head Sparsity Emerges from Visual Concept Responses in MLLMs. arXiv preprint arXiv:2506.05344 (2025).

[29] Weihan Wang, Zehai He, Wenyi Hong, Yean Cheng, Xiaohan Zhang, Ji Qi, Shiyu Huang, Bin Xu, Yuxiao Dong, Ming Ding, and Jie Tang. 2024. LVBench: An Extreme Long Video Understanding Benchmark. arXiv:2406.08035 [cs.CV]

[30] Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. 2023. Self-Consistency Improves Chain of Thought Reasoning in Language Models. arXiv:2203.11171 [cs.CL] https://arxiv.org/abs/2203.11171

[31] Zikang Wang, Boyu Chen, Zhengrong Yue, Yi Wang, Yu Qiao, Limin Wang, and Yali Wang. 2026. VideoChat-A1: Thinking with Long Videos by Chain-of-Shot Reasoning. arXiv:2506.06097 [cs.CV] https://arxiv.org/abs/2506.06097

[32] Zheng Wang, Haoran Chen, Haoxuan Qin, Zhipeng Wei, Tianwen Qian, and Cong Bai. 2026. Think, Then Verify: A Hypothesis-Verification Multi-Agent Framework for Long Video Understanding. arXiv:2603.04977 [cs.CV] https: //arxiv.org/abs/2603.04977

[33] Haoning Wu, Dongxu Li, Bei Chen, and Junnan Li. 2024. LongVideoBench: A Benchmark for Long-context Interleaved Video-Language Understanding. arXiv:2407.15754 [cs.CV] https://arxiv.org/abs/2407.15754

[34] Wenhao Wu, Yizhong Wang, Guangxuan Xiao, Hao Peng, and Yao Fu. 2024. Retrieval Head Mechanistically Explains Long-Context Factuality. arXiv:2404.15574 [cs.CL] https://arxiv.org/abs/2404.15574

[35] Chaojun Xiao, Pengle Zhang, Xu Han, Guangxuan Xiao, Yankai Lin, Zhengyan Zhang, Zhiyuan Liu, and Maosong Sun. 2024. InfLLM: Training-Free Long-Context Extrapolation for LLMs with an Eficient Context Memory. arXiv:2402.04617 [cs.CL] https://arxiv.org/abs/2402.04617

[36] Guangxuan Xiao, Jiaming Tang, Jingwei Zuo, Junxian Guo, Shang Yang, Haotian Tang, Yao Fu, and Song Han. 2024. DuoAttention: Eficient Long-Context LLM Inference with Retrieval and Streaming Heads. arXiv:2410.10819 [cs.CL] https: //arxiv.org/abs/2410.10819

[37] Haiyang Yan, Hongyun Zhou, Peng Xu, Xiaoxue Feng, and Mengyi Liu. 2026. Symphony: A Cognitively-Inspired Multi-Agent System for Long-Video Understanding. arXiv:2603.17307 [cs.CV] https://arxiv.org/abs/2603.17307

[38] Senqiao Yang, Yukang Chen, Zhuotao Tian, Chengyao Wang, Jingyao Li, Bei Yu, and Jiaya Jia. 2024. VisionZip: Longer is Better but Not Necessary in Vision Language Models. arXiv preprint arXiv:2412.04467 (2024).

[39] Jun Zhang, Jue Wang, Huan Li, Lidan Shou, Ke Chen, Gang Chen, and Sharad Mehrotra. 2024. Draft& verify: Lossless large language model acceleration via self-speculative decoding. In Proceedings ofthe 62nd Annual Meeting ofthe Association for Computational Linguistics (Volume 1: Long Papers). 11263–11282.

[40] Kaichen Zhang, Bo Li, Peiyuan Zhang, Fanyi Pu, Joshua Adrian Cahyono, Kairui Hu, Shuai Liu, Yuanhan Zhang, Jingkang Yang, Chunyuan Li, and Ziwei Liu. 2024. LMMs-Eval: Reality Check on the Evaluation of Large Multimodal Models. arXiv:2407.12772 [cs.CL] https://arxiv.org/abs/2407.12772

[41] Yuan Zhang, Chun-Kai Fan, Junpeng Ma, Wenzhao Zheng, Tao Huang, Kuan Cheng, Denis Gudovskiy, Tomoyuki Okuno, Yohei Nakata, Kurt Keutzer, et al. 2025. SparseVLM: Visual Token Sparsification for Eficient Vision-Language Model Inference. In International Conference on Machine Learning.

[42] Yunzhu zhang, Yu Lu, Tianyi Wang, Fengyun Rao, Yi Yang, and Linchao Zhu. 2025. FlexSelect: Flexible Token Selection for Eficient Long Video Understanding. In The Thirty-ninth Annual Conference on Neural Information Processing Systems. https://openreview.net/forum?id=0D3ja9s17M

[43] Zhenyu Zhang, Ying Sheng, Tianyi Zhou, Tianlong Chen, Lianmin Zheng, Ruisi Cai, Zhao Song, Yuandong Tian, Christopher Ré, Clark Barrett, Zhangyang Wang, and Beidi Chen. 2023. H O: Heavy-Hitter Oracle for Eficient Generative Inference of Large Language Models. https://arxiv.org/abs/2306.14048

[44] Orr Zohar, Xiaohan Wang, Yann Dubois, Nikhil Mehta, Tong Xiao, Philippe Hansen-Estruch, Licheng Yu, Xiaofang Wang, Felix Juefei-Xu, Ning Zhang, Serena Yeung-Levy, and Xide Xia. 2024. Apollo: An Exploration of Video Understanding in Large Multimodal Models. arXiv:2412.10360 [cs.CV] https: //arxiv.org/abs/2412.10360

## A Additional Related Works.

MLLMs typically encode visual inputs as discrete visual tokens. Early approaches were restricted to processing videos at fixed resolutions, such as InternVL [8] and LLaVA-NeXT [20], which hinders the comprehension of content containing diverse semantic details. Most recent advancements support native dynamic resolution, enabling direct, patch-wise partitioning of inputs at arbitrary resolutions. Indeed, this capability has emerged as the prevailing paradigm in contemporary MLLMs. Leading open-source mllms, including Qwen3-VL [26] and GLM-4V [27], have adopted this paradigm, enabling direct patch-wise tokenization of visual inputs at arbitrary resolutions. This capability of mllms to handle videos at arbitrary native resolutions serves as the foundation for our VideoMM. Leveraging this flexibility, VideoMM adaptively modulates video resolution to coordinate video inference on a coarse-to-fine basis, enhancing comprehensive video understanding.

Beyond token reduction, a significant body of work addresses long-context LLM inference through complementary mechanisms that can be synergistically integrated with our approach.

KV Cache Eviction accelerates the decoding process by selectively pruning internally retained KV caches. Predominantly operating in long text scenarios, these methods optimize compression by assessing cache importance [9, 11, 17, 43] or employing dynamic budget allocation [4, 10, 13]. Recent researches have been extended to MLLMs: notably, SparseMM [28] incorporates inter-head allocation into multimodal cache management. However, since these methods compress the cache only after processing all tokens to alleviate memory pressure, they do not reduce the token count itself. Consequently, they are orthogonal to token reduction strategies.

Sparse attention methods [15, 18, 23, 35] accelerate inference by retaining the full KV cache while selecting only a critical subset for computation. However, as these approaches do not lower the number of processed tokens, they remain orthogonal to token reduction methods. Future work could combine lightweight token reduction with sparse attention to further minimize inference overhead.

Speculative decoding [6, 16, 39] accelerates inference by utilizing a small model to draft outputs, which are then verified by a larger model, thereby improving decoding eficiency through collaborative inference. This technique has recently been extended to MLLMs. For example, SpecVLM [14] enhances decoding eficiency by compressing visual tokens and passing them to a smaller model for drafting, followed by verification from a larger model. This demonstrates that visual token reduction is orthogonal to the spec ulative decoding paradigm. Future work could explore integrating these two approaches more closely.

Agent-based Video Understanding [5, 31, 32, 37] tackles longvideo redundancy at a semantic level through interactive reasoning. LVAgent [5] introduces a multi-round dynamic collaboration framework among multiple MLLM agents, filtering out suboptimal reasoning through iterative selection, perception, action, and reflection. VideoChat-A1 [31] emphasizes the inherent shot-based structure of videos by proposing a Chain-of-Shot reasoning paradigm. It progressively selects relevant shots, partitions them into fine-grained subshots via feature clustering, and evaluates reasoning confidence to iteratively refine the temporal context. VideoHV-Agent [32] reformulates long-video question answering as a hypothesis–verification process: a Thinker converts candidate answers into testable hypotheses, a Judge derives discriminative clues, and a Verifier localizes and examines targeted video evidence before an Answer agent integrates the results. Symphony [37] employs a cognitively inspired multi-agent architecture that coordinates task decomposition, evidence grounding, visual perception, and reflection for long-video reasoning. Future work could explore synergizing agentbased semantic exploration with visual token reduction methods for optimal long-video inference eficiency.

Table 6: Detailed performance comparison on LongVideoBench. The best two results are bolded.
<table><tr><td rowspan="2">Method</td><td>LongVideoBench</td><td colspan="10"></td><td colspan="7"></td></tr><tr><td>E2O</td><td>E3E</td><td>O2E</td><td>030</td><td>S2A</td><td>S2E</td><td>S2O</td><td>SAA</td><td>SOS SSS</td><td></td><td>T2A</td><td>T2E</td><td>T2O</td><td>T3E</td><td>T3O</td><td>TAA</td><td>TOS</td><td>Avg.</td></tr><tr><td colspan="10">Qwen2.5-VL-7B</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Vanilla</td><td>67.69</td><td>56.38</td><td>62.07</td><td>53.03</td><td>67.05</td><td>66.67</td><td>61.11</td><td>56.94</td><td>66.67</td><td>40.21</td><td>62.03</td><td>64.62</td><td>55.26</td><td>50.68</td><td>52.70</td><td>58.54</td><td>39.73</td><td>57.67</td></tr><tr><td>VisionZip</td><td>73.44</td><td>61.96</td><td>63.22</td><td>60.61</td><td>67.05</td><td>68.82</td><td>51.39</td><td>52.78</td><td>69.14</td><td>43.16</td><td>60.76</td><td>64.62</td><td>52.63</td><td>49.32</td><td>56.76</td><td>54.32</td><td>39.73</td><td>58.23</td></tr><tr><td>VidCom</td><td>61.54</td><td>61.70</td><td>59.77</td><td>48.48</td><td>64.77</td><td>70.97</td><td>52.78</td><td>51.39</td><td>61.73</td><td>36.08</td><td>58.23</td><td>66.15</td><td>51.32</td><td>41.10</td><td>48.65</td><td>57.32</td><td>32.88</td><td>54.60</td></tr><tr><td>Flexselect</td><td>73.85</td><td>73.40</td><td>67.82</td><td>66.67</td><td>77.27</td><td>75.27</td><td>65.28</td><td>61.11</td><td>69.14</td><td>53.61</td><td>68.35</td><td>67.69</td><td>68.42</td><td>45.21</td><td>64.86</td><td>50.00</td><td>38.36</td><td>64.10</td></tr><tr><td>VideoMM</td><td>76.92</td><td>73.40</td><td>68.97</td><td>56.06</td><td>79.55</td><td>73.12</td><td>65.28</td><td>63.89</td><td>69.14</td><td>54.64</td><td>70.89</td><td>67.69</td><td>68.42</td><td>52.05</td><td>59.46</td><td>53.66</td><td>42.47</td><td>64.70</td></tr><tr><td colspan="10">GLM-4.1V-9B</td><td colspan="8"></td></tr><tr><td>Vanilla</td><td>58.46</td><td>67.02</td><td>72.09</td><td>57.58</td><td>75.00</td><td>62.37</td><td>55.56</td><td>56.94</td><td>64.20</td><td>38.14</td><td>67.09</td><td>61.54</td><td>55.26</td><td>50.68</td><td>55.41</td><td>62.20</td><td>39.73</td><td>58.98</td></tr><tr><td>VisionZip</td><td>53.23</td><td>61.54</td><td>62.07</td><td>52.38</td><td>66.27</td><td>61.11</td><td>54.29</td><td>49.30</td><td>67.50</td><td>37.89</td><td>51.32</td><td>60.32</td><td>51.39</td><td>53.52</td><td>55.41</td><td>62.50</td><td>38.03</td><td>55.35</td></tr><tr><td>VidCom</td><td>55.38</td><td>60.64</td><td>56.32</td><td>45.45</td><td>60.23</td><td>61.29</td><td>44.44</td><td>51.39</td><td>58.02 32.99</td><td></td><td>55.70</td><td>53.85</td><td>50.00</td><td>45.21</td><td>47.30</td><td>57.32</td><td>39.73</td><td>51.68</td></tr><tr><td>Flexselect</td><td>56.92</td><td>63.83</td><td>63.22</td><td>62.12</td><td>73.86</td><td>67.74</td><td>59.72</td><td>58.33</td><td>67.90</td><td>45.36</td><td>77.22</td><td>60.00</td><td>52.63</td><td>57.53</td><td>55.41</td><td>69.51</td><td>35.62</td><td>60.66</td></tr><tr><td>VideoMM</td><td>69.23</td><td>74.47</td><td>71.26</td><td>60.61</td><td>80.68</td><td>72.04</td><td>62.50</td><td>55.56</td><td>72.84</td><td>46.39</td><td>72.15</td><td>67.69</td><td>60.53</td><td>54.79</td><td>59.46</td><td>69.51</td><td>38.36</td><td>64.32</td></tr><tr><td colspan="10">Qwen3-VL-8B</td><td colspan="8"></td></tr><tr><td></td><td>72.31</td><td>78.72</td><td>72.41</td><td>60.61</td><td>82.95</td><td>68.82</td><td>72.22</td><td>65.28</td><td>70.37</td><td>57.73</td><td>74.68</td><td>66.15</td><td>71.05</td><td>53.42</td><td>60.81</td><td>59.76</td><td>45.21</td><td>66.94</td></tr><tr><td>VisionZip VidCom</td><td>59.68</td><td>71.91</td><td>67.06</td><td>58.06</td><td>71.25</td><td>65.17</td><td>55.71</td><td>58.33</td><td>64.10</td><td>49.47</td><td>58.67</td><td>63.49</td><td>63.89</td><td>42.03</td><td>51.43</td><td>57.50</td><td>39.44</td><td>58.97</td></tr><tr><td>Flexselect</td><td>67.69</td><td>65.96</td><td>64.37</td><td>56.06</td><td>70.45</td><td>70.97</td><td>61.11</td><td>54.17</td><td>64.20</td><td>43.30</td><td>64.56</td><td>64.62</td><td>65.79</td><td>43.84</td><td>56.76</td><td>51.22</td><td>38.36</td><td>59.16</td></tr><tr><td>VideoMM</td><td>73.85 70.77</td><td>76.60 77.66</td><td>68.97</td><td>69.70</td><td>81.82 80.68</td><td>72.04 70.97</td><td>65.28</td><td>66.67</td><td>70.37 71.60</td><td>58.76</td><td>78.48</td><td>66.15</td><td>75.00 73.68</td><td>49.32</td><td>63.51</td><td>58.54</td><td>42.47</td><td>67.17</td></tr><tr><td></td><td></td><td></td><td>72.41</td><td>68.18</td><td></td><td></td><td>69.44</td><td>63.89</td><td></td><td>54.64</td><td>75.95</td><td>73.85</td><td></td><td>47.95</td><td>64.86</td><td>62.20</td><td>46.58</td><td>67.54</td></tr></table>

Table 7: Detailed performance comparison on LVBench. The best two results are bolded.
<table><tr><td rowspan="2">Method</td><td colspan="7">LVBench</td></tr><tr><td>Key Info. Retrieval</td><td>Event Understanding</td><td>Summarization</td><td>Entity Recognition</td><td>Reasoning</td><td>Temporal Grounding</td><td>Avg.</td></tr><tr><td colspan="8">Qwen2.5-VL-7B</td></tr><tr><td>Vanilla</td><td>45.02</td><td>36.79</td><td>32.76</td><td>39.44</td><td>39.80</td><td>30.91</td><td>39.57</td></tr><tr><td>VisionZip</td><td>41.92</td><td>38.79</td><td>32.76</td><td>41.95</td><td>42.29</td><td>36.82</td><td>41.38</td></tr><tr><td>VidCom</td><td>47.08</td><td>39.10</td><td>36.21</td><td>39.14</td><td>36.32</td><td>36.82</td><td>40.80</td></tr><tr><td>Flexselect</td><td>58.76</td><td>47.14</td><td>44.83</td><td>51.99</td><td>47.26</td><td>39.55</td><td>51.45</td></tr><tr><td>VideoMM</td><td>58.42</td><td>47.45</td><td>39.66</td><td>51.40</td><td>49.75</td><td>41.36</td><td>50.94</td></tr><tr><td colspan="8">GLM-4.1V-9B</td></tr><tr><td>Vanilla</td><td>52.58</td><td>45.75</td><td>39.66</td><td>47.71</td><td>47.76</td><td>36.82</td><td>47.90</td></tr><tr><td>VisionZip</td><td>35.05</td><td>35.39</td><td>31.03</td><td>37.08</td><td>37.31</td><td>27.73</td><td>36.41</td></tr><tr><td>VidCom</td><td>46.05</td><td>36.32</td><td>32.76</td><td>34.12</td><td>43.28</td><td>39.55</td><td>37.70</td></tr><tr><td>Flexselect</td><td>55.33</td><td>48.69</td><td>31.03</td><td>50.66</td><td>48.76</td><td>41.82</td><td>50.16</td></tr><tr><td>VideoMM</td><td>53.95</td><td>48.22</td><td>29.31</td><td>49.78</td><td>49.75</td><td>40.91</td><td>49.52</td></tr><tr><td colspan="8">Qwen3-VL-8B</td></tr><tr><td>Vanilla</td><td>60.82</td><td>51.47</td><td>32.76</td><td>56.72</td><td>44.78</td><td>47.73</td><td>54.10</td></tr><tr><td>VisionZip</td><td>39.52</td><td>41.27</td><td>27.59</td><td>41.06</td><td>37.31</td><td>35.45</td><td>40.22</td></tr><tr><td>VidCom</td><td>52.23</td><td>43.28</td><td>29.31</td><td>44.02</td><td>39.30</td><td>40.00</td><td>44.03</td></tr><tr><td>Flexselect</td><td>63.23</td><td>52.55</td><td>34.48</td><td>56.28</td><td>52.74</td><td>46.82</td><td>55.20</td></tr><tr><td>VideoMM</td><td>58.42</td><td>52.86</td><td>31.03</td><td>57.16</td><td>48.26</td><td>45.45</td><td>54.36</td></tr></table>

## B Benchmark Details.

Below, we provide a detailed overview of the video understanding benchmarks utilized in the experiments.

Table 8: Detailed performance comparison on VideoMME. The best two results are bolded.
<table><tr><td rowspan="2">Method</td><td colspan="10">VideoMME</td></tr><tr><td>Counting Problem Info. Synopsis Object Recog.</td><td colspan="10">Action Reason. Object Reason. Temporal Percep. Attribute Percep.</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>Qwen2.5-VL-7B</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Vanilla</td><td>40.67</td><td></td><td>68.36</td><td>53.33</td><td>69.09</td><td>72.52</td><td>41.24</td><td>63.58</td><td>68.35</td><td>62.96</td><td>80.36</td><td>61.56</td></tr><tr><td>VisionZip</td><td>37.31</td><td>78.02</td><td>70.06</td><td>55.44</td><td>80.00</td><td>73.87</td><td>50.28</td><td>62.62</td><td>64.75</td><td>59.26</td><td>75.00</td><td>62.07</td></tr><tr><td>VidCom</td><td>38.06</td><td>74.30</td><td>67.80</td><td>56.14</td><td>74.55</td><td>67.57</td><td>49.15</td><td>65.18</td><td>71.22</td><td>61.11</td><td>69.64</td><td>61.56</td></tr><tr><td>Flexselect VideoMM</td><td>45.90 46.64</td><td>81.11 81.42</td><td>74.58</td><td>63.51</td><td>74.55</td><td>79.73</td><td>61.02</td><td>68.37</td><td>79.86</td><td>64.81</td><td>78.57</td><td>68.85</td></tr><tr><td></td><td></td><td></td><td>72.32</td><td>59.65</td><td>72.73</td><td>77.93</td><td>60.45</td><td>69.97</td><td>79.14</td><td>66.67</td><td>80.36</td><td>68.41</td></tr><tr><td colspan="10">GLM-4.1V-9B 63.22</td></tr><tr><td>Vanilla</td><td>44.78</td><td>79.57</td><td>75.14</td><td>55.79</td><td>70.91</td><td>77.48</td><td>54.24</td><td>67.09</td><td>80.58</td><td>68.52</td><td>80.36</td><td>66.67</td></tr><tr><td>VisionZip</td><td>35.07</td><td>73.07</td><td></td><td></td><td></td><td>73.42</td><td>45.20</td><td>62.30</td><td>58.99</td><td>64.81</td><td>75.00</td><td>59.48</td></tr><tr><td>VidCom</td><td>35.07</td><td>72.45</td><td>66.10 56.78</td><td>55.09 47.37</td><td>65.45 56.36</td><td>63.96</td><td>47.46</td><td>56.23</td><td>73.38</td><td>55.56</td><td>64.29</td><td>56.48</td></tr><tr><td>Flexselect VideoMM</td><td>43.28 42.54</td><td>77.09</td><td>70.90</td><td>50.88</td><td>63.64</td><td>75.68</td><td>54.24</td><td>67.73</td><td>80.58</td><td>66.67</td><td>83.93</td><td>65.15</td></tr><tr><td>59.65</td><td>79.26</td><td>73.73</td><td></td><td>64.32 64.32 70.91</td><td colspan="3">78.38 57.06</td><td>68.05</td><td>82.01</td><td>70.37</td><td>83.93</td><td>67.37</td></tr><tr><td colspan="10">69.38</td></tr><tr><td>Vanilla</td><td></td><td></td><td>76.84</td><td></td><td>Qwen3-VL-8B</td><td></td><td>63.84</td><td>70.61</td><td>82.73</td><td>72.22</td><td></td><td>71.78</td></tr><tr><td>VisionZip</td><td>52.61 41.04</td><td>83.59</td><td></td><td>63.16</td><td>81.82</td><td>81.08</td><td></td><td></td><td></td><td></td><td>83.93</td><td></td></tr><tr><td>VidCom</td><td>45.52</td><td>79.57</td><td>69.21</td><td>60.35</td><td>74.55</td><td>80.18</td><td>54.24</td><td>66.45</td><td>64.03</td><td>70.37</td><td>80.36</td><td>65.48 64.59</td></tr><tr><td>Flexselect</td><td>50.75</td><td>78.33</td><td>70.62</td><td>56.84</td><td>70.91</td><td>73.42</td><td>56.50</td><td>63.58 69.65</td><td>66.19</td><td>68.52</td><td>73.21 82.14</td><td>72.19</td></tr><tr><td>VideoMM</td><td>50.37</td><td>84.21 83.59</td><td>79.38 75.42</td><td>63.16 69.60 63.51 69.38</td><td>76.36 83.64</td><td>81.98 82.43</td><td>67.80 65.54</td><td>71.88</td><td>82.73 80.58</td><td>75.93 72.22</td><td>82.14</td><td>71.67</td></tr></table>

(1) LongVideoBench [33] is a comprehensive video questionanswering benchmark comprising 3,763 web-collected videos (up to one hour) with 6,678 human-annotated multiplechoice questions across 17 fine-grained categories. It introduces a referring reasoning task where models must retrieve and reason over relevant multimodal information from specified video contexts.

(2) VideoMME [12] is a full-spectrum multi-modal evaluation benchmark for MLLMs comprising 900 videos (254 hours) with 2,700 question-answer pairs. It spans 6 visual domains, covers short- to long-form videos (11 seconds to 1 hour), and includes multi-modal inputs (frames, subtitles, audio).

(3) LVBench [29] is a benchmark specifically designed for long video understanding, created to address the gap in evaluating MLLMs for real-world applications requiring comprehension of videos spanning several hours. It comprises publicly sourced videos and includes diverse tasks focused on long video comprehension and information ex traction.

## C Model Details

Our evaluation employs three models, all of which support dynamic resolution:

• Qwen2.5-VL-7B-Instruct [2] builds upon the Qwen2- VL architecture, introducing significant advancements in visual-language understanding. Key enhancements include superior comprehension ofcomplex visual elements such as text, charts, and layouts, alongside agentic capabilities for dynamic tool direction. Architecturally, it extends dynamic resolution to the temporal dimension through dynamic Frame Per Second sampling, complemented by mRoPE updates that facilitate learning of temporal sequences and precise event identification within videos.

• GLM4.1V-9B-Thinking [27]: is a powerful Mllm designed for advanced general-purpose multimodal reasoning. At its core is a reasoning-centric training framework that leverages a robust vision foundation model, initially established through large-scale pre-training. This foundation’s full potential is subsequently unlocked via Reinforcement Learning, significantly enhancing the model’s comprehensive capabilities. It also supports dynamic-resolution video inputs.

Table 9: Impact of Lite Models on LongVideoBench based on Qwen3-VL-8B. Acc.: Accuracy (%); Thpt.: Throughput (samples/min))
<table><tr><td>Method</td><td></td><td>Accuracy Throughput</td></tr><tr><td>Vanilla</td><td>66.94</td><td>0.85</td></tr><tr><td>VideoMM</td><td>67.54</td><td>3.69</td></tr></table>

• Qwen3-VL-8B-Instruct [26] stands as the most powerful lite mllm in the Qwen series to date, featuring comprehensive upgrades. It incorporates a novel Interleaved-MRoPE, which provides full-frequency allocation across time, width, and height through robust positional embeddings, thereby enhancing long-horizon video reasoning. Additionally, it employs Text–Timestamp Alignment, moving beyond T-RoPE to achieve precise, timestamp-grounded event localization for stronger video temporal modeling.

## D Accelerating Token Compression Using Lightweight MLLMs

While VideoMM primarily focuses on optimizing representation granularity, its framework is orthogonal to—and can be efectively combined with—model-centric downsizing strategies. In such a configuration, a lightweight model with fewer parameters conducts the initial token selection, streamlining the process before a larger MLLM performs the final inference. We try a direct integration. To demonstrate this flexibility, we instantiate a variant named VideoMM-Lite, where we directly employ the of-the-shelf Qwen3-VL-4B for token selection and the Qwen3-VL-8B for the final inference, avoiding additional training overhead <sup>8</sup>. As shown in Table 9, compared to the standard VideoMM, VideoMM-Lite increases throughput from 3.69 to 4.19 samples/mins, with only a marginal accuracy dip from 67.54% to 66.34%. This result confirms that integrating a smaller parameter model for selection is a viable pathway to further enhance throughput with minimal impact on performance.

Table 10: Impact of multi-stages of VideoMM on LongVideoBench.
<table><tr><td>Method</td><td>Accuracy</td><td>Throughput</td></tr><tr><td>Vanilla</td><td>66.94</td><td>0.85</td></tr><tr><td>VideoMM (Two Stage)</td><td>67.54</td><td>3.69</td></tr><tr><td>VideoMM-multi (Three Stage)</td><td>64.55</td><td>5.91</td></tr></table>

Accuracy represents the percentage of correct predictions, while Throughout denotes the number of samples processed per second.

Table 11: Comparison of VideoMM and FlexSelect on VideoMME using Qwen-VL models
<table><tr><td>Method</td><td>Acc.</td><td>Thpt.</td></tr><tr><td>Qwen2.5-VL-7B</td><td></td><td></td></tr><tr><td>Vanilla (512) FlexSelect (256) VideoMM (512)</td><td>61.56 68.26 68.41</td><td>0.85 3.54 5.47</td></tr><tr><td>GLM-4.1V-9B</td><td></td><td></td></tr><tr><td>Vanilla (512) FlexSelect (256) VideoMM (512)</td><td>66.67 69.56 67.37</td><td>0.58 3.25 4.21</td></tr><tr><td>Qwen3-VL-8B</td><td></td><td></td></tr><tr><td>Vanilla (512)</td><td>71.78</td><td>0.97</td></tr><tr><td>FlexSelect (256)</td><td>71.48</td><td>3.54</td></tr><tr><td>VideoMM (512)</td><td>71.67</td><td>3.59</td></tr></table>

Acc.: Accuracy (%); Thpt.: Throughput (samples/min)

## E Three-Stage Micro-Micro Exploration.

To further demonstrate the extensibility of our framework, we introduce VideoMM-multi, a variant that expands the original two-stage design into a three-stage cascade. This is achieved by inserting an additional, coarser-grained hierarchical level prior to the standard Macro Proxy. Specifically, we implement an Ultra-Macro Proxy with an aggressive downsampling factor of $k ^ { 2 }$ (compared to the standard factor �), designed to filter out the most obvious non-essential regions with extreme eficiency before passing the remaining candidates to the Macro level.We evaluated this threestage design (� = 2) using Qwen3-VL-8B on LongVideoBench. As shown in Table 10, the comparison reveals a distinct trade-of between granular precision and computational speed. The standard two-stage VideoMM maintains superior semantic retention, achieving a higher accuracy of 67.54 compared to VideoMM-multi’s 64.55.

Conversely, by ofloading the majority ofworkload to the new Ultra-Macro tier, VideoMM-multi drastically reduces the computational burden, increasing throughput to 5.91 samples/min—significantly outperforming the standard VideoMM’s 3.69 samples/min. These results confirm that dynamically adjusting the depth of the macromicro hierarchy serves as a powerful lever for tuning the accuracyeficiency trade-of.