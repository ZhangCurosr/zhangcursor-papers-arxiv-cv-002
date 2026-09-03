# TAME: Temporal-Aware Mixture-of-Experts for Text-Video Retrieval

Uicheol Jung<sup>1</sup> Juyoung Hong<sup>1</sup>

<sup>1</sup>Sejong University, Seoul, Republic of Korea

ucjung@rcv.sejong.ac.kr

Hojung Kwon<sup>2</sup> Yukyung Choi<sup>1</sup>

<sup>2</sup>Wisenut, Seongnam, Republic of Korea

ykchoi@rcv.sejong.ac.kr

## Abstract

Text–Video Retrieval (TVR) retrieves videos that match a natural-language query, but extending image–text models such as CLIP to videos isfundamentally limited by the lack oftemporal modeling. Videos exhibitframe-wise heterogeneity in appearance and motion, and compressing allframes into a single representation often obscures temporal structure and semantic transitions. To address this, we propose Temporal-Aware Mixture-of-Experts for Text-Video Retrieval (TAME), a CLIP-basedframework thatjointly models frame-level structure and temporal relations. First, we integrate sparse Mixture-of-Experts (MoE) layers into both CLIP encoders and apply frame-consistent routing on the vision branch so that experts specialize according toframelevel visual patterns while preserving the original vision– language alignment. Second, we introduce Frame–Temporal (FT) tokens that aggregate global cross-frame information and feed it back to each frame, enabling the visual encoder to capture long-range temporal dependencies without harming local details. Third, we design a Cross-Temporal Interaction and Aggregation (CTIA) module that refines framewise sentence–video similarities through staged temporal filtering andfusion. Experiments on standard TVR benchmarks show that TAME consistently improves over CLIPbased baselines. On MSR-VTT, it improves R@1 by 4.0 over CLIP4Clip, and also achieves consistent gains on DiDeMo, MSVD, LSMDC, and ActivityNet. The code is available at https://github.com/sejong-rcv/TAME.

## 1. Introduction

With the explosive growth of video content across social platforms, entertainment services, and multimedia archives, the ability to retrieve relevant videos from natural-language queries, known as Text-Video Retrieval (TVR), has become increasingly critical. Unlike traditional metadata-based search, TVR demands fine-grained multimodal understanding, where both visual signals and linguistic semantics must be aligned at the appropriate spatiotemporal granularity. This capability is essential for applications such as content recommendation, video understanding, and large-scale retrieval systems deployed in industrial settings.

Recent TVR research has been accelerated by CLIP-style dual encoders [1], which leverage strong image–text alignment learned from large-scale contrastive pretraining. Previous studies [2–4] demonstrate that transferring image–text representations to the video domain provides a powerful foundation for retrieval. However, as illustrated in Fig. 1(a), conventional CLIP-based approaches aggregate all frame embeddings into a single video representation, relying primarily on mean pooling. This strategy overlooks temporal structure, compresses heterogeneous frame-level patterns, and provides no mechanism to adaptively weight frames based on semantic relevance. Consequently, CLIP-based models struggle to fully capture the temporal progression and semantic variability that characterize videos.

Recent and representative studies [5–9] attempt to mitigate this issue by introducing temporal aggregation/attention, cross-frame token interaction, or token selection/clustering modules. Yet, these approaches still rely on a single dense encoder pathway, forcing all frames through the same feature transformation. This architectural bottleneck limits representational capacity and prevents the model from adapting to heterogeneous visual patterns—such as static versus dynamic scenes, background shifts, or object composition changes across frames. The lack of specialization becomes more pronounced as video content becomes more visually diverse.

To address these challenges, we propose Temporal-Aware Mixture-of-Experts for Text-Video Retrieval (TAME), a unified framework designed to capture frame-level variability and improve temporal aggregation for retrieval. As illustrated in Fig. 1(b), TAME replaces dense FFNs in both the text and vision encoders with sparse Mixture-of-Experts (MoE) layers, enabling frame-consistent expert routing that expands representational diversity while preserving CLIP alignment. Importantly, extending MoE to the text encoder supports semantic specialization on the language side, allowing the model to capture diverse linguistic patterns and maintain balanced cross-modal capacity. In addition, we introduce Frame–Temporal (FT) tokens that propagate compact temporal context across frames while keeping patch-level spatial structure intact, and we design a Cross-Temporal Interaction and Aggregation (CTIA) module that refines frame–sentence similarity into a coherent video-level score.

![](images/69b0144ca245c025b623f70a471b64fd3e16f1b942f79a03ad3e889a2c415856.jpg)  
Figure 1. Comparison between previous method and ours. (a) Previous methods encode all frames with a single encoder and average them uniformly, which overlooks temporal dependency and frame importance. (b) Our TAME performs frame-wise expert routing to model visual diversity and employs Cross-Temporal Interaction and Aggregation (CTIA) to adaptively weight and integrate frame representations according to their temporal context, yielding a robust video-level feature.

Our key contributions are summarized as follows:

1. Sparse Mixture-of-Experts architecture tailored for TVR. We introduce a sparse MoE framework specifically tailored for Text–Video Retrieval. By integrating MoE layers into both the text and visual encoders and adopting frame-consistent routing, the model captures heterogeneous frame-level visual patterns that are difficult to model using a single shared encoder.

2. Frame–Temporal tokens for lightweight temporal propagation. We propose FT tokens that encode compact temporal context and interact with frame and patch embeddings, enabling the model to share temporal information across frames without modifying spatial token structure.

3. Sentence-conditioned cross-temporal aggregation. To address temporal inconsistency introduced by framewise expert routing, we propose a Cross-Temporal Interaction and Aggregation (CTIA) module that refines frame–sentence similarity into a coherent video-level representation.

4. Consistent performance gains across TVR benchmarks. Evaluation on MSR-VTT, DiDeMo, MSVD, LSMDC, and ActivityNet shows that the proposed architecture consistently improves over CLIP-based baselines.

Ultimately, TAME retains CLIP’s strong vision–language alignment while resolving its inherent temporal limitations, achieving more precise and resilient text–video matching.

## 2. Related Work

## 2.1. CLIP-based Text–Video Dual Encoders

Recent text–video retrieval (TVR) pipelines commonly adopt CLIP-style dual encoders that map text and video into a shared embedding space and compute similarities for ranking. Representative approaches such as CLIP4Clip [2], CLIP2Video [10], CenterCLIP [7], X-CLIP [3], TS2- Net [5], X-Pool [4], CLIP-ViP [6], Prompt Switch [11], Cap4Video [12], TeachCLIP [8] differ in how they pool frame features, parameterize sentence–video alignment, and incorporate lightweight temporal modules, while sharing a pretrained CLIP [1] backbone and a similarity head over sentence and frame-level features. However, a recurring limitation in this line is that video-level decisions are still often derived from relatively shallow temporal aggregation (e.g., pooling or local temporal blocks), which weakens cross-frame interaction and makes the final score vulnerable to temporally noisy frames that are irrelevant to the query. This issue becomes more pronounced when the model must decide which frames constitute valid evidence for the sentence, since naive pooling can over-count spurious matches and under-utilize temporally consistent cues. In addition, while improving representation capacity could alleviate such brittleness, dense scaling of the encoders is expensive and does not directly guarantee better video-level evidence consolidation. Motivated by these gaps, we introduce a sparse Mixture-of-Experts (MoE) into the CLIP encoders to expand representational capacity with low active compute via sparse gating, and we further propose a Cross-Temporal Interaction & Aggregation (CTIA) module that explicitly consolidates per-frame sentence–frame scores into a single video-level logit by propagating evidence across neighboring frames and encouraging temporal consistency, thereby making the retrieval decision less sensitive to frame-level noise.

![](images/01bf9996d5b996da9fb393fd7e620e1490648480a57013b3172efca03814d890.jpg)  
Figure 2. TAME overview. Based on CLIP dual encoders, we introduce three add-on modules to enhance text–video retrieval. (i) MoE-Enhanced CLIP: we apply sparse upcycling that replaces selected FFN blocks with MoE layers to expand encoding capacity; experts are initialized from the dense checkpoint, while routers are randomly initialized. (ii) Frame–Temporal (FT) tokens: videos are sampled into frames and encoded into frame-level representations, which are further refined by injecting lightweight cross-frame context in the visual branch. (iii) CTIA: the CTIA module performs sentence-conditioned frame alignment and aggregates frame evidence to estimate the fina frame-aware similarity. In addition to the frame-aware branch, frame representations are mean-pooled to form a video representation for global alignment with the sentence embedding.

## 2.2. Mixture-of-Experts in Vision and Language

Deep learning has shown that scaling model, data, and compute improves within-distribution generalization. Early studies consistently reported reductions in test-in-distribution loss as scale increases [13]. In parallel, large language and multimodal models [14], [1], [15] showed broad transfer and prompt-based reasoning, and “scaling-law” studies formalized power-law links between model/data/compute and test loss [16], [17]. However, for many large dense models in practice, memory/communication/power costs grow steeply with scale, often reducing training efficiency. To mitigate these limitations, sparsely activated Mixture-of-Experts (MoE) architectures [18] have gained attention: a router selects a small subset of experts per input, thereby expanding total parameter capacity while aiming to keep per-token active compute roughly constant under typical capacity settings.

In vision, V-MoE [19] extends sparse MoE to Vision Transformers by replacing selected FFN blocks with expert pools and routing patch tokens to a small number of experts, enabling substantial capacity scaling with nearly constant active computation. For videos, recent works explore applying conditional routing to spatiotemporal transformers (e.g., ViViT-style backbones) to dynamically allocate computation across visual tokens under a compute budget [20].

Recent attempts to apply MoE to CLIP further support this direction: CLIP-MoE [21] uses multiplet upcycling to lightly fine-tune complementary CLIP branches as experts and combine them via MoE, showing consistent gains across several public zero-shot classification/retrieval benchmarks and as an MLLM backbone without substantially increasing compute; CLIP-UP [22] provides a simple, efficient recipe to upcycle a pretrained dense CLIP into a sparse MoE, reusing existing weights to achieve a stronger performance–cost balance under modest additional training.

Different from standard vision/video MoEs that primarily target recognition accuracy–efficiency trade-offs, our setting is text–video retrieval with a CLIP-style dual-encoder, where routing should preserve cross-modal alignment. ${ \mathrm { A c } } -$ cordingly, TAME designs routing signals tailored to retrieval (text-token routing in the text branch and frame-wise routing in the visual branch) while maintaining the pretrained vision–language alignment.

## 2.3. Hashing-based Content-Based Medical Image Retrieval

Beyond CLIP-style TVR, a parallel retrieval literature studies compact representations for efficient indexing, particularly hash-code based content-based image retrieval (CBIR). In medical imaging, a number of methods learn discriminative deep features and generate hash codes to enable fast search in large-scale archives, including deep feature selection guided Siamese hashing for medical image retrieval [23], class-driven medical image retrieval using hash codes of deep features [24], and attention-based end-to-end CNN frameworks for X-ray image retrieval [25]. These works highlight a complementary set of challenges and solutions: they typically focus on single-modality image-to-image retrieval and emphasize compact/binary hashing for storage and efficient indexing, but they do not address cross-modal semantic alignment (text–vision) nor the temporal evidence aggregation that is intrinsic to videos. Consequently, although hashing-based medical CBIR methods are relevant as retrieval-oriented representation learning, their core gaps relative to our setting include limited support for cross-modal grounding and the absence of mechanisms for consolidating temporally distributed evidence. In contrast, our work targets cross-modal text–video retrieval and addresses the video-specific decision bottleneck by combining capacityefficient scaling (sparse MoE in CLIP encoders) with temporally consistent evidence aggregation (CTIA), which directly improves video-level ranking robustness rather than optimizing binary codes for single-image indexing.

## 3. Method

In this section, we introduce the overall methodology of the proposed framework TAME. The entire architecture is illustrated in Fig. 2. First, Sec. 3.1 describes CLIP-based dual encoders that extract aligned text and frame-level video representations. Then, Sec. 3.2 introduces MoE-Enhanced CLIP, which replaces selected dense FFNs with sparse experts to enhance representation capacity. Next, Sec. 3.3 presents Frame–Temporal (FT) tokens, a lightweight temporal operator that injects cross-frame context into the visual representations before computing framewise similarities. Sec. 3.4 then introduces CTIA, which performs sentence-conditioned aggregation over frames to complement the baseline similarity. Finally, Sec. 3.5 specifies the training objective and joint optimization of the contrastive and auxiliary losses.

Formulation overview and linkage to our contributions. We consider text–video retrieval with a text query $T$ and a video $V ~ = ~ \{ I _ { n } \} _ { n = 1 } ^ { F }$ consisting of $F$ sampled frames. The CLIP text encoder outputs token embeddings ${ \textbf { e } } \in$ $\mathbb { R } ^ { ( L _ { t } + 1 ) \times C }$ , and we use the [EOS] embedding as the global sentence representation, denoted by $q = e _ { [ \mathrm { E O S } ] } \in \mathbb { R } ^ { C }$ . The visual encoder maps each frame $I _ { n }$ to a frame-level representation $f _ { n } \in \mathbb { R } ^ { C }$ (the per-frame [CLS] embedding), yielding the frame sequence ${ \bf F } = \{ f _ { n } \} _ { n = 1 } ^ { - } \dot { \in } \mathbb { R } ^ { F \times C }$ . We compute framewise sentence–frame evidence s $\in \mathbb { R } ^ { F }$ by $s _ { n } = \cos ( q , f _ { n } )$ and define $\mathbf { s } = ( s _ { 1 } , \ldots , s _ { F } )$ . This framewise evidence quantifies which frames are directly relevant to the query and serves as the input signal for CTIA to model cross-frame interactions and perform decision-level temporal evidence aggregation.

Prior CLIP4Clip-style baselines primarily make the retrieval decision using a video-level similarity computed from an aggregated video embedding. In our framework, we additionally exploit the framewise evidence s and define the final retrieval logit as a function of both the sentence and the frame sequence, $\mathrm { i . e . , } S _ { \mathrm { f i n a l } } = \Phi ( q , { \bf F } )$ . We instantiate $\Phi ( \cdot )$ via CTIA by defining a sentence-conditioned aggregation function $\mathrm { A g g } ( \mathbf { s } , \mathbf { F } )$ and combining its output with the baseline similarity to obtain $S _ { \mathrm { f i n a l } }$ (see Sec. 3.4 for the explicit formulation).

This formulation clarifies how each proposed component contributes. MoE-Enhanced CLIP modifies the encoders that produce $q$ and F by replacing selected dense FFNs with sparse experts under top- $. K$ routing (Sec. 3.2). Frame– Temporal (FT) tokens refine F by injecting cross-frame context in the visual branch prior to computing s (Sec. 3.3). CTIA defines $\mathrm { A g g ( \cdot ) }$ to transform s into frame weights and a decision-level score used in $S _ { \mathrm { f i n a l } } \left( \mathrm { S e c } . 3 . 4 \right)$ . Finally, we use $S _ { \mathrm { f i n a l } }$ as the similarity logit to construct the batch-wise matrix $S _ { i j } = S _ { \mathrm { f i n a l } } ( v _ { i } , q _ { j } )$ and optimize the retrieval objective (Sec. 3.5).

## 3.1. CLIP Encoder

Following prior work [2], we adopt a dual-encoder architecture based on the pre-trained CLIP [1] model to extract aligned text–video representations within a shared embedding space.

Text Encoder. Given a text query T, the CLIP text encoder processes this sequence and outputs a sequence of token embeddings $\mathbf { e } = \{ \bar { e } _ { 1 } , e _ { 2 } , . . . , e _ { L _ { t } } , \bar { e } _ { [ \mathrm { E O S } ] } \} \in \mathbf { \bar { \mathbb { R } } } ^ { ( L _ { t } + 1 ) \times C }$ where $L _ { t }$ denotes the number of textual tokens in T, and $C$ is the feature dimension of each token embedding. Here, $e _ { \left[ \mathrm { E O S } \right] } \in \mathbb { R } ^ { C }$ corresponds to the embedding of the special [EOS] token, which is used as the global sentence representation.

Visual Encoder. Given a video V consisting of $F$ sampled frames, each frame is processed by the CLIP visual encoder. For the n-th frame, we partition it into M patches and prepend a [CLS] token, resulting in a set of patch embeddings $P _ { n } ~ = ~ \{ p _ { n } ^ { 1 } , p _ { n } ^ { 2 } , \ldots , p _ { n } ^ { M } \} ~ \in ~ \mathbb { R } ^ { M \times C }$ , where all visual tokens share the same feature dimension C as textual tokens. The embedding of the [CLS] token serves as the frame-level representation $f _ { n } \in \mathbb { R } ^ { C }$ . Aggregating all frame embeddings yields the video-level sequence $\mathbf { F } = \{ f _ { 1 } , f _ { 2 } , \dotsc , f _ { F } \} \in \mathbf { \bar { \mathbb { R } } } ^ { \bar { F } \times C }$

![](images/27a4dd605c7b459f894dd0cdd92e8b5f6c3cafa54245dd9f8e045bc447526134.jpg)  
Figure 3. Sparse upcycling (MoE-Enhanced CLIP). We replace the pretrained image-layer FFN (MLP) with a sparse MoE layer composed of E cloned MLP experts. The expert weights are copied from the dense checkpoint to preserve CLIP initialization, while the router is randomly initialized to enable expert specialization during training.

Both textual and visual embeddings are further mapped into a shared embedding space, allowing direct similarity computation between the global sentence embedding e<sub>[EOS]</sub> and the video-level sequence F for downstream similarity computation. This CLIP-based feature extraction serves as the backbone for our proposed MoE-Enhanced CLIP, which augments the encoder layers with MoE modules to better capture diverse frame-level semantics and temporal dependencies.

## 3.2. MoE-Enhanced CLIP

In Text-Video Retrieval, the encoders must handle highly diverse visual and linguistic patterns. Accordingly, it is important to enhance $\mathrm { C L I P } \ ' _ { \mathrm { s } }$ encoding capacity while preserving its pretrained vision–language alignment. Sparse Mixture-of-Experts (MoE) offers a principled way to achieve this [19, 21, 22, 26]: it replaces the dense feed-forward network (FFN) with a set of parallel MLP experts and routes each token to only a small subset of them, thereby expanding representational capacity without increasing computation.

To this end, we follow prior MoE-augmented Transformer studies [21, 22] and integrate sparse MoE layers into both CLIP encoders. Among the L Transformer layers, we replace a subset of the FFNs with MoE layers, each consisting of $N _ { E }$ position-wise MLP experts $\{ E _ { i } \} _ { i = 1 } ^ { N _ { E } }$ that take the role of the original FFN.

A lightweight router $G ( \cdot )$ assigns each input token a score for every expert. We select only the top-K experts to enforce sparsity, normalize the selected scores with a softmax, and obtain the MoE output as a weighted sum of the selected expert outputs. Following [21, 22], each expert is initialized by copying the pretrained CLIP FFN, while the router is randomly initialized. This initialization maintains compatibility with the pretrained checkpoint and allows experts to gradually specialize during training. The overall MoE block architecture is illustrated in Fig. 3.

Text Branch. Text-side routing is performed per token. For a given MoE layer, let $h _ { j } \in \mathbb { R } ^ { C }$ denote the representation of the j-th token. The text router $G _ { t } ( \cdot )$ produces expert logits for $h _ { j }$ , from which only the top-K entries are retained and normalized. We define the resulting sparse gating vector as:

$$
\theta _ { j } = \mathrm { s o f t m a x } ( \mathrm { T o p K } ( G _ { t } ( h _ { j } ) ) ) .\tag{1}
$$

Here, $\theta _ { j , i }$ denotes the routing weight assigned to expert $E _ { i }$ for token j. The MoE output is then computed as a gated mixture over the experts:

$$
y _ { j } = \mathrm { M o E } ( h _ { j } ) = \sum _ { i = 1 } ^ { N _ { E } } \theta _ { j , i } \cdot E _ { i } ( h _ { j } ) .\tag{2}
$$

Visual Branch. Unlike text, video inputs contain dense spatial patches within each frame, and vanilla MoE routing assigns experts independently to every patch. Such patchlevel routing often results in inconsistent expert selection within the same frame, failing to capture coherent framelevel semantics. To mitigate this, we introduceframe-wise routing. For frame $n ,$ let $\tilde { f } _ { n } \in \mathbb { R } ^ { C }$ denote the frame-level [CLS] token, and let $X _ { n } ~ \in ~ \mathbb { R } ^ { M \times C }$ denote its $M$ patch tokens. The video router $G _ { v } ( \cdot )$ produces expert logits from ${ \tilde { f } } _ { n } .$ , and the resulting sparse gating vector is defined as:

$$
\theta _ { n } = \mathrm { s o f t m a x } \Big ( \mathrm { T o p K } \Big ( G _ { v } ( \tilde { f } _ { n } ) \Big ) \Big ) .\tag{3}
$$

The same gating weights are then applied uniformly to all patch tokens, and the MoE output for frame n is computed as:

$$
y _ { n } = \mathrm { M o E } ( X _ { n } ) = \sum _ { i = 1 } ^ { N _ { E } } \theta _ { n , i } \cdot E _ { i } ( X _ { n } ) .\tag{4}
$$

This frame-consistent routing enforces intra-frame coherence, suppresses patch-level routing noise, and enables experts to specialize in distinct frame-level visual patterns, which is crucial for modeling temporal structure.

Auxiliary Loss. While the routing schemes above determine expert selection, they do not enforce balanced expert utilization. Following the Switch Transformer [26], we adopt an auxiliary load-balancing loss to prevent routing collapse. For an MoE layer, let z denote an input token and $q _ { i } ( z )$ the router probability for expert $E _ { i }$ . The loss $\mathcal { L } _ { \mathrm { a u x } }$ is defined as:

$$
\mathcal { L } _ { \mathrm { a u x } } = \boldsymbol { \alpha } \cdot \boldsymbol { N _ { E } } \sum _ { i = 1 } ^ { N _ { E } } \rho _ { i } \pi _ { i } ,\tag{5}
$$

where the hard routing frequency $\rho _ { i }$ and the mean router probability $\pi _ { i }$ are given by:

$$
\rho _ { i } = \frac { 1 } { | B | } \sum _ { z \in B } \mathbf { 1 } \biggl \{ \underset { j } { \arg \operatorname* { m a x } } q _ { j } ( z ) = i \biggr \} ,\tag{6}
$$

$$
\pi _ { i } = \frac { 1 } { | B | } \sum _ { z \in B } q _ { i } ( z ) .\tag{7}
$$

Here, B denotes the set of routed tokens in a batch. Following prior work [26], we set $\alpha = 0 . 0 1$ . Since both the text and video encoders contain MoE layers—with token-wise routing for text and frame-wise routing for video—we compute the auxiliary load-balancing loss separately for each encoder. We denote these losses as $\bar { \mathcal { L } } _ { \mathrm { a u x } } ^ { T }$ and $\mathcal { L } _ { \mathrm { a u x } } ^ { V } .$ , respectively.

## 3.3. Frame-Temporal Tokens for Vision MoE

CLIP’s vision encoder processes each frame independently, which prevents it from capturing temporal continuity (e.g., actions unfolding over time) and cross-frame dependencies such as object re-appearance or scene transitions. This limitation is problematic for video understanding, where global temporal context is essential. To address this, we introduce a set of Frame–Temporal (FT) tokens, inspired by the video proxy in CLIP-VIP [6]. For each frame n, we insert H learnable FT tokens $t _ { n } = \{ t _ { n } ^ { 1 } , \ldots , t _ { n } ^ { H } \}$ between the frame-level [CLS] token $f _ { n }$ and the patch tokens $P _ { n }$ . These FT tokens serve as compact carriers of inter-frame context and later accumulate global information through both local and global attention mechanisms. The initial input sequence for frame n is then formed as:

$$
\begin{array} { r } { { \bf T } _ { n } ^ { ( 0 ) } = \mathrm { c o n c a t } \big ( \ : f _ { n } , \ : t _ { n } , \ : P _ { n } \big ) , } \end{array}\tag{8}
$$

where $\mathbf { T } _ { n } ^ { ( 0 ) }$ contains one frame-level CLS token, H FT tokens, and M patch tokens.

Non-MoE blocks follow the standard CLIP Transformer, refining intra-frame context through local self-attention. In contrast, MoE blocks incorporate both local and global context. We first apply frame-wise multi-head self-attention to obtain locally updated tokens:

$$
\left( f _ { n } ^ { \prime } , t _ { n } ^ { \prime } , P _ { n } ^ { \prime } \right) = \mathrm { M u l t i H e a d A t t n } \left( \mathbf { T } _ { n } ^ { \left( \ell - 1 \right) } \right) ,\tag{9}
$$

where $f _ { n } ^ { \prime }$ is the updated [CLS] token, $t _ { n } ^ { \prime }$ are the FT tokens, and $P _ { n } ^ { \prime }$ are the updated patch tokens for frame n.

Global attention over FT tokens (implementation detail). After the local self-attention in Eq. (9), we obtain $f _ { n } ^ { \prime } \in \mathbb { R } ^ { C }$ $t _ { n } ^ { \prime } \in \mathbb { R } ^ { H \times C }$ , and $P _ { n } ^ { \prime } \in \mathbb { R } ^ { M \times C }$ for each frame n, where $f _ { n } ^ { \prime }$ is the frame-level [CLS] token, $t _ { n } ^ { \prime }$ are the H FT tokens, and $P _ { n } ^ { \prime }$ are the patch tokens. To inject cross-frame context while keeping the [CLS] token unchanged (so that the core frame-level representative is not altered by global mixing), we exclude $f _ { n } ^ { \prime }$ and build a global token sequence by concatenating FT tokens and patch tokens across F frames:

$$
U = \mathrm { c o n c a t } \big ( \{ [ t _ { n } ^ { \prime } ; P _ { n } ^ { \prime } ] \} _ { n = 1 } ^ { F } \big ) \in \mathbb { R } ^ { F ( H + M ) \times C } .\tag{10}
$$

We then apply global multi-head self-attention over $U .$ Specifically, we compute the projected queries, keys, and values as:

$$
Q = U W _ { Q } , \quad K = U W _ { K } , \quad V = U W _ { V } ,\tag{11}
$$

where $W _ { Q } , W _ { K } , W _ { V } \ \in \ \mathbb { R } ^ { C \times C }$ (implemented in a multihead manner with head dimension d). The global attention output is given by:

$$
\hat { U } = \mathrm { s o f t m a x } \left( \frac { Q K ^ { \top } } { \sqrt { d } } \right) V \in \mathbb { R } ^ { F ( H + M ) \times C } .\tag{12}
$$

In a minibatch, this global attention is computed independently for each video sample, i.e., tokens from different videos do not attend to each other.

Updating FT tokens only. We reshape $\hat { U }$ back to per-frame outputs $\check { u } _ { n } \in \mathbb { R } ^ { ( H + M ) \times \check { C } }$ and update only the FT tokens via a residual:

$$
\bar { t } _ { n } = \mathrm { L N } ( t _ { n } ^ { \prime } + \hat { u } _ { n } [ 1 { : } H ] ) \in \mathbb { R } ^ { H \times C } ,\tag{13}
$$

while keeping the patch tokens unchanged in this global step, i.e., $\bar { P } _ { n } = P _ { n } ^ { \prime }$ . Moreover, updating only FT tokens restricts the pathway through which global information enters the frame representation, preventing direct injection into patch tokens and thereby preserving CLIP’s pretrained patch-level semantics. Within a video, interactions between unrelated frames are further moderated by the content-based nature of attention, which learns to assign low weights to semantically irrelevant pairs.

Complexity (layer-wise and branch-wise). The global attention over $U$ costs $O \big ( ( F ( H { + } M ) ) ^ { 2 } C \big )$ per video and per layer. Importantly, we do not apply this global attention to all Transformer layers. Instead, we deliberately apply global attention over frames only to a small subset of MoEenabled layers in the visual encoder (e.g., the top $L _ { G } ^ { V } { = } 2$ backbone layers), and we do not apply global attention over frames in the text branch $( L _ { G } ^ { T } \mathrm { = } 0 )$ . This design injects crossframe context where we expand capacity with MoE, while keeping the overall compute overhead small.

Let $S = F ( H { + } M )$ denote the length of the global sequence U. Then, the total additional cost of global attention over a minibatch of size B is:

$$
\begin{array} { r l } & { C _ { \mathrm { g l o b a l } } = O ( B \cdot L _ { G } ^ { v } \cdot S ^ { 2 } \cdot C ) } \\ & { \qquad = O \big ( B \cdot L _ { G } ^ { v } \cdot ( F ( H + M ) ) ^ { 2 } \cdot C \big ) . } \end{array}\tag{14}
$$

and $\mathcal { C } _ { \mathrm { g l o b a l } } ^ { T } = 0$ for the text branch. Therefore, compared to applying global-over-frames at every layer of both encoders, our design reduces the added cost proportionally to the small ratio $L _ { G } ^ { V } / ( L _ { V } + L _ { T } )$

![](images/1e4da60f51ed6ba09d915f101ab768ec601e9ea92a6c54a95ee2f6c347cd6f89.jpg)  
Figure 4. CTIA (Cross-Temporal Interaction & Aggregation) computes the base similarity r from the sentence vector $\mathbf { q }$ and frame vectors F, then applies Gaussian smoothing to r to obtain $r _ { \mathrm { c o n v } }$ and generates $r _ { \mathrm { g r a p h } }$ through propagation on the frame–frame similarity graph. It finally fuses these three signals to produce the frame– sentence score $S _ { f , t }$

## 3.4. Cross-Temporal Interaction & Aggregation (CTIA)

As designed in the previous section, the MoE layers enhance representational capacity by enabling expert specialization at the frame and token levels. However, while the vision-side MoE effectively refines each frame independently, it still lacks an explicit mechanism for inter-frame alignment and temporal consolidation, which are essential for building a coherent temporal stream. To address this limitation, we introduce Cross-Temporal Interaction and Aggregation (CTIA), which injects sentence-conditioned signals into frames and aggregates them into a unified decision score. To explicitly connect framewise similarities to the final retrieval decision, we formulate CTIA as a sentence-conditioned aggregation function $S _ { \mathrm { C T I A } } = \mathrm { A g g } ( \mathbf { s } , \mathbf { F } )$ that transforms the framewise evidence s into weights over frames and produces a decisionlevel score used in $S _ { \mathrm { f i n a l } }$

CTIA is built upon three complementary relevance signals: (i) the raw query-conditioned evidence r, (ii) its locally smoothed variant $\mathbf { r } _ { \mathrm { c o n v } }$ that enforces short-range temporal coherence, and (iii) the graph-propagated signal $\bf { r } _ { \mathrm { g r a p h } }$ that transfers relevance across semantically similar but temporally distant frames. We fuse these signals linearly for interpretability and stable optimization; ablations that remove each term (via β-term configurations) are provided in Table 7.

We first compute a framewise cosine similarity vector $\mathbf { s } = ( s _ { 1 } , s _ { 2 } , \dotsc , s _ { F } ) \in \mathbb { R } ^ { F }$ between each frame representation $f _ { n }$ and the sentence embedding $q ( \mathrm { i . e . , } q = e _ { [ \mathrm { E O S } ] } ) .$ Based on this framewise evidence, CTIA (Fig. 4) derives the relevance signals $\mathbf { r } , \mathbf { r } _ { \mathrm { c o n v } }$ , and $\bf { r } _ { \mathrm { g r a p h } }$ , and produces the final frame weights $\mathbf { w } = ( w _ { 1 } , \hdots , \bar { w } _ { F } ) \in \mathbb { R } ^ { F }$ through four stages: sharpening, local Gaussian smoothing, non-local frame-graph propagation, and learnable fusion.

(1) Sharpening. We apply a softmax along the frame axis to obtain per-frame relevance $\mathbf { r } \in \mathbb { R } ^ { F }$

$$
\mathbf { r } = \mathrm { s o f t m a x } \left( \frac { \mathbf { s } } { \tau _ { s } } \right) ,\tag{15}
$$

where $\tau _ { s }$ is the temperature parameter.

(2) Local Gaussian smoothing. We smooth r with a normalized 1D Gaussian kernel $\boldsymbol { k } ~ \in \mathbb { R } ^ { K }$ (odd K, width σ) to encode short-range temporal coherence. The kernel is defined as:

$$
k [ u ] \propto \exp \left( - \frac { 1 } { 2 } \left( \frac { u } { \sigma } \right) ^ { 2 } \right) .\tag{16}
$$

The locally smoothed relevance vector $\mathbf { r } _ { \mathrm { c o n v } }$ is computed by:

$$
\mathbf { r } _ { \mathrm { c o n v } } = \mathbf { r } * k \in \mathbb { R } ^ { F } .\tag{17}
$$

This operation smooths out fluctuations in frame relevance by averaging neighboring frames. However, local smoothing alone cannot connect repeated or visually similar scenes that are temporally distant, motivating a complementary nonlocal step.

(3) Non-local frame-graph propagation. To complement the limitations of local smoothing, we propagate relevance over a content-aware frame graph. Using the frame feature $\mathbf { F } \in \mathbb { R } ^ { F \times C }$ , we compute a frame-to-frame affinity matrix $\mathbf { A } \in \mathbb { R } ^ { F \times F }$ as:

$$
\mathbf { A } = \mathrm { s o f t m a x } \left( \frac { \mathbf { F } \mathbf { F } ^ { \top } } { \tau _ { g } } \right) ,\tag{18}
$$

where $\tau _ { g }$ is a temperature parameter. To prevent oversmoothing, we add a self-loop with a balancing coefficient $\gamma \in [ 0 , 1 ]$

$$
\mathbf { A }  \gamma I + ( 1 - \gamma ) \mathbf { A } ,\tag{19}
$$

where I denotes the identity matrix. The non-local relevance $\mathbf { r } _ { \mathrm { g r a p h } }$ is then obtained by propagating r over the graph:

$$
\mathbf { r } _ { \mathrm { g r a p h } } = \mathbf { A } \mathbf { r } .\tag{20}
$$

Unlike local convolution, graph propagation connects frames that are semantically similar regardless of temporal distance, reinforcing repeated or thematically related content while the self-loop term $\gamma I$ prevents over-diffusion.

(4) Learnable fusion and final weighting. We combine the three relevance signals using learnable weights $\beta =$ $\mathrm { R e L U } ( [ \beta _ { r } , \beta _ { c } , \beta _ { g } ] )$ and a fusion scale $\lambda _ { g } \geq 0$ . The fused relevance vector u is:

$$
\mathbf { u } = \beta _ { r } \mathbf { r } ~ + ~ \beta _ { c } \mathbf { r } _ { \mathrm { c o n v } } ~ + ~ \beta _ { g } \lambda _ { g } \mathbf { r } _ { \mathrm { g r a p h } } .\tag{21}
$$

We then compute the final frame weights $\mathbf { w } \in \mathbb { R } ^ { F }$ using a temperature-controlled softmax:

$$
\mathbf { w } = \mathrm { s o f t m a x } \bigg ( \frac { \mathbf { u } } { \tau _ { w } } \bigg ) .\tag{22}
$$

Given the framewise similarity vector $\mathbf { s } = ( s _ { 1 } , s _ { 2 } , \ldots , s _ { F } )$ the CTIA score $S _ { \mathrm { C T I A } }$ is computed as a weighted sum:

$$
S _ { \mathrm { C T I A } } = \mathbf { w } ^ { \top } \mathbf { s } = \sum _ { i = 1 } ^ { F } w _ { i } s _ { i } .\tag{23}
$$

Following CLIP4Clip [2], we use a temporal encoder to aggregate the frame features into a video feature v. We then compute the cosine similarity between the video representation v and the sentence representation $e _ { [ \mathrm { E O S } ] }$ to obtain the baseline text–video score, which we denote as $S _ { \mathrm { b a s e } }$ . The final retrieval score $S _ { \mathrm { f i n a l } }$ is computed as the average of the video-level similarity and our CTIA similarity:

$$
S _ { \mathrm { f i n a l } } = \frac { 1 } { 2 } \left( S _ { \mathrm { b a s e } } + S _ { \mathrm { C T I A } } \right) .\tag{24}
$$

In our default setting, we optimize the entire framework end-to-end using the objective in Sec. 3.5.

## 3.5. Objective

Given a minibatch $\{ ( v _ { i } , q _ { i } ) \} _ { i = 1 } ^ { B }$ , where $v _ { i }$ denotes the i-th video and $q _ { i }$ the corresponding text query, we construct a similarity matrix $\mathbf { S } \in \bar { \mathbb { R } ^ { B \times B } }$ , where each entry represents the final video–text similarity:

$$
S _ { i j } = S _ { \mathrm { f i n a l } } ( v _ { i } , q _ { j } ) .\tag{25}
$$

We then optimize the bidirectional InfoNCE losses [37]:

$$
\begin{array} { r l } & { \mathcal { L } _ { q \to v } = - \displaystyle \frac { 1 } { B } \sum _ { i = 1 } ^ { B } \log \frac { \exp ( S _ { i i } / \tau ) } { \sum _ { j = 1 } ^ { B } \exp ( S _ { i j } / \tau ) } , } \\ & { \mathcal { L } _ { v \to q } = - \displaystyle \frac { 1 } { B } \sum _ { i = 1 } ^ { B } \log \frac { \exp ( S _ { i i } / \tau ) } { \sum _ { j = 1 } ^ { B } \exp ( S _ { j i } / \tau ) } , } \\ & { \mathcal { L } _ { v q } = \displaystyle \frac { 1 } { 2 } ( \mathcal { L } _ { q \to v } + \mathcal { L } _ { v \to q } ) . } \end{array}\tag{26}
$$

Here, τ is the temperature hyperparameter for scaling.

Additionally, we apply a batch-hard margin loss [38, 39] to further separate each positive pair from its most confusing negative. For each anchor i, we denote the positive similarity as $p _ { i } = S _ { i i }$ , and identify the hardest negative as $n _ { i } = \operatorname* { m a x } _ { j \neq i } S _ { i j }$ . The loss penalizes cases where the hardest negative becomes closer than the positive by a margin m:

$$
\mathcal { L } _ { \mathrm { h a r d } } = \frac { 1 } { B } \sum _ { i = 1 } ^ { B } \left[ m + n _ { i } - p _ { i } \right] _ { + } , \qquad [ x ] _ { + } = \operatorname* { m a x } ( x , 0 ) .\tag{27}
$$

Here, $m > 0$ is the margin hyperparameter that enforces a minimum separation between the positive and its hardest negative.

Finally, the overall training objective combines the bidirectional InfoNCE loss, the MoE load-balancing regularizers, and the hard negative margin loss:

$$
{ \mathcal { L } } _ { \mathrm { t o t a l } } = { \mathcal { L } } _ { v q } ~ + ~ \lambda \big ( { \mathcal { L } } _ { \mathrm { a u x } } ^ { T } + { \mathcal { L } } _ { \mathrm { a u x } } ^ { V } \big ) ~ + ~ { \mathcal { L } } _ { \mathrm { h a r d } } .\tag{28}
$$

Here, λ is a hyperparameter.

## 4. Experiments

## 4.1. Datasets

We evaluate on five widely used text–video retrieval benchmarks and, unless otherwise noted, follow standard splits and protocols. MSR-VTT [40] contains 10,000 web video clips, each annotated with 20 sentences; following [41], we use 9,000 clips for training and report results on the standard 1k-A test split. DiDeMo [42] includes 10,000 Flickr videos with 40,000 sentence descriptions, and we concatenate all descriptions of a video into a single paragraph and perform paragraph-to-video retrieval. MSVD [43] comprises 1,970 YouTube videos with roughly 40 crowd-sourced captions per video, and we adopt the 1,200/100/670 train/val/test split. LSMDC [44] is a movie-sourced benchmark with 118,081 short clips (2–30 seconds) collected from 202 movies; following [2], [4], we use 7,408 videos for validation and 1,000 videos for testing. ActivityNet [45] contains around 20,000 YouTube videos; following prior videoparagraph retrieval protocols [46], [41], we concatenate all descriptions of a video into a single paragraph and evaluate on the ’val1’ split.

## 4.2. Evaluation

We report results for both text-to-video (T→V) and video-totext (V→T) retrieval. We evaluate using Recall@K (R@1, R@5, R@10), Median Rank (MdR), and Mean Rank (MnR). R@K is the fraction of queries whose first correct match appears within the top-K, and MdR/MnR are the median and mean rank, respectively, of that first correct match.

## 4.3. Implementation details.

Following CLIP4Clip [2], we initialize the text and video encoders with the pretrained CLIP (ViT-B/32) weights [1]. We optimize with Adam [47] and decay the learning rate using a cosine schedule [48]. The initial learning rate is set to $1 \times 1 0 ^ { - 7 }$ for the text and video encoders (including linear projections), and to $1 \times 1 0 ^ { - 4 }$ for the other modules. For MSR-VTT, MSVD and LSMDC we set the maximum token length to 32 and the maximum number of frames per video to 12. For DiDeMo and ActivityNet, we use 64 tokens and 64 frames. The batch size and number of training epochs are 128 and 5 for MSR-VTT, 300 and 5 for MSVD, LSMDC 64 and 20 for DiDeMo and 64 and 25 for ActivityNet. For MoE-Enhanced CLIP, we replace the FFNs of the last two Transformer layers ({11, 12}) in both encoders with sparse MoE blocks using $N _ { E } { = } 4$ experts and Top-K routing (K=2 for text and K=1 for vision). We set the number of Frame–Temporal (FT) tokens to H=2. Experiments are conducted on four NVIDIA RTX A6000 GPUs. On the MSR-VTT 1k-A split, training requires approximately 4 hours for 5 epochs. During inference, computing the query– video similarity matrix takes 20.84 s in total (20.8 ms per query).

Table 1. Retrieval performance on the MSR-VTT-9K dataset. The symbol \* indicates the use of the DSL [27] post-processing operation in the corresponding methods, while the symbol \*\* indicates that QB-Norm [28] is additionally applied, and the symbol † denotes the results obtained from the source code provided by the authors. Gray-shaded entries indicate results taken from large-scale foundation models with substantially different pretraining scale/objectives; for InternVideo2, only R@1 is reported in the source table, and other metrics are not available (shown as “–”).
<table><tr><td rowspan="2">Method</td><td colspan="5">Text ⇒ Video</td><td colspan="5">Video ⇒ Text</td></tr><tr><td>R@1↑</td><td>R@5↑</td><td>R@10↑</td><td>MdR↓</td><td>MnR↓</td><td>R@1↑</td><td>R@5↑</td><td>R@10↑</td><td>MdR↓</td><td>MnR↓</td></tr><tr><td>CLIP-ViT-B/32</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>CLIP4Clip† [2]</td><td>44.0</td><td>72.0</td><td>81.6</td><td>2.0</td><td>16.0</td><td>43.9</td><td>70.6</td><td>79.8</td><td>2.0</td><td>12.0</td></tr><tr><td>CAMoE [27]</td><td>44.6</td><td>72.6</td><td>81.8</td><td>2.0</td><td>13.3</td><td>45.1</td><td>72.4</td><td>83.1</td><td>2.0</td><td>10.0</td></tr><tr><td>CAMoE* [27]</td><td>47.3</td><td>74.2</td><td>84.5</td><td>2.0</td><td>11.9</td><td>49.1</td><td>74.3</td><td>84.3</td><td>2.0</td><td>9.9</td></tr><tr><td>CLIP2Video [10]</td><td>45.6</td><td>72.6</td><td>81.7</td><td>2.0</td><td>14.6</td><td>43.5</td><td>72.3</td><td>82.1</td><td>2.0</td><td>10.2</td></tr><tr><td>CenterCLIP [7]</td><td>44.2</td><td>71.6</td><td>82.1</td><td>2.0</td><td>15.1</td><td>42.8</td><td>71.7</td><td>82.2</td><td>2.0</td><td>10.9</td></tr><tr><td>X-Pool [4]</td><td>46.9</td><td>72.8</td><td>82.2</td><td>2.0</td><td>14.3</td><td>44.4</td><td>73.3</td><td>84.0</td><td>2.0</td><td>9.0</td></tr><tr><td>TS2-Net [5]</td><td>47.0</td><td>74.5</td><td>83.8</td><td>2.0</td><td>13.0</td><td>45.3</td><td>74.1</td><td>83.7</td><td>2.0</td><td>9.2</td></tr><tr><td>X-CLIP [3]</td><td>46.1</td><td>73.0</td><td>83.1</td><td>2.0</td><td>13.2</td><td>46.8</td><td>73.3</td><td>84.0</td><td>2.0</td><td>9.1</td></tr><tr><td>QB-Norm** [28]</td><td>47.2</td><td>73.0</td><td>83.0</td><td>2.0</td><td>一</td><td>一</td><td>一</td><td>一</td><td>一</td><td>一</td></tr><tr><td>DRL [29]</td><td>47.4</td><td>74.6</td><td>83.8</td><td>2.0</td><td>一</td><td>一</td><td>一</td><td>一</td><td>一</td><td>一</td></tr><tr><td>CLIP-VIP [6]</td><td>46.5</td><td>72.1</td><td>82.5</td><td>一</td><td>一</td><td>一</td><td>一</td><td>一</td><td>一</td><td>一</td></tr><tr><td>STAN [30]</td><td>46.9</td><td>72.8</td><td>82.8</td><td>2.0</td><td>一</td><td>一</td><td>一</td><td>一</td><td>一</td><td>一</td></tr><tr><td>UATVR [31]</td><td>47.5</td><td>73.9</td><td>83.5</td><td>2.0</td><td>12.3</td><td>46.9</td><td>73.8</td><td>83.8</td><td>2.0</td><td>8.6</td></tr><tr><td>UATVR* [31]</td><td>49.8</td><td>76.1</td><td>85.5</td><td>2.0</td><td>12.9</td><td>51.1</td><td>74.8</td><td>85.1</td><td>1.0</td><td>8.3</td></tr><tr><td>ProST [32]</td><td>48.2</td><td>74.6</td><td>83.4</td><td>2.0</td><td>12.4</td><td>46.3</td><td>74.2</td><td>83.2</td><td>2.0</td><td>8.7</td></tr><tr><td>Prompt Switch [11]</td><td>47.8</td><td>73.9</td><td>82.2</td><td>一</td><td>14.1</td><td>46.0</td><td>74.3</td><td>84.8</td><td>一</td><td>8.5</td></tr><tr><td>TeachCLIP [8]</td><td>46.8</td><td>74.3</td><td>一</td><td>一</td><td>一</td><td></td><td></td><td></td><td>一</td><td>一</td></tr><tr><td>EERCF [33]</td><td>47.8</td><td>74.1</td><td>84.1</td><td>一</td><td>一</td><td>44.7</td><td>74.2</td><td>83.9</td><td>一</td><td>一</td></tr><tr><td>LSDO [34]</td><td>49.1</td><td>75.2</td><td>84.2</td><td>2.0</td><td>12.0</td><td>一</td><td></td><td></td><td>一</td><td>一</td></tr><tr><td>GLSCL [35]</td><td>48.1</td><td>73.9</td><td>82.3</td><td>2.0</td><td>13.8</td><td>47.1</td><td>73.0</td><td>83.2</td><td>2.0</td><td>10.3</td></tr><tr><td>GLSCL** [35]</td><td>50.0</td><td>73.7</td><td>82.8</td><td>1.5</td><td>14.2</td><td>49.4</td><td>74.5</td><td>83.9</td><td>2.0</td><td>9.6</td></tr><tr><td>InternVideo2s2-6B [36]</td><td>62.8</td><td></td><td></td><td></td><td></td><td>60.2</td><td></td><td></td><td></td><td></td></tr><tr><td>TAME</td><td>48.0</td><td>74.1</td><td>81.6</td><td>2.0</td><td>14.2</td><td>46.9</td><td>72.7</td><td>82.1</td><td>2.0</td><td>10.6</td></tr><tr><td>TAME*</td><td>51.1</td><td>75.8</td><td>84.5</td><td>1.0</td><td>10.9</td><td>52.1</td><td>76.5</td><td>83.3</td><td>1.0</td><td>8.5</td></tr></table>

Reporting protocol. Unless specified, we report raw similarities. When indicated $( \mathrm { e . g . , \vec { \cdot } \mathrm { e . s 7 } } )$ , we apply evaluation-time DSL [27] with temperature $\tau _ { \mathrm { d s l } }$ , which reweights similarities via softmax along both the query and video dimensions and combines the two weights into the final score. Notably, DSL is applied only at inference time and does not affect training.

## 4.4. Performance Comparisons

We compare our model against methods that use CLIP-ViT as the backbone. On MSR-VTT (Table 1), our model achieves an improvement of +4.0 R@1 over the CLIP4Clip [2] baseline. Compared to X-CLIP, which employs multi-level alignment (video–sentence, video–word, frame–sentence, frame–word), our approach relies only on video–sentence and frame–sentence alignment yet achieves a +1.9 R@1, underscoring the effectiveness of CTIA-based temporal alignment. Applying the DSL post-processing further boosts performance across all metrics.

As reported in Table 2, we present results on the DiDeMo, MSVD, LSMDC, and ActivityNet datasets. Our model demonstrates stable generalization across these datasets, yielding consistent improvements over the baseline. In particular, on DiDeMo, R@1 improves by +2.7 over the baseline. On MSVD, R@1 improves by +1.4 over the baseline. For LSMDC, our model achieves a +1.5 improvement in R@1 over CLIP4Clip, increasing from 22.6 to 24.1, while also reducing MnR from 61.0 to 57.6. Furthermore, ActivityNet, a long-form benchmark, shows a +2.2 improvement in R@1 over CLIP4Clip, improving from 40.5 to 42.7. It also yields consistent improvements at higher cutoffs, with R@5 reaching 73.0 and R@10 reaching 84.6. Finally, we confirm that applying DSL post-processing yields additional performance gains across all four datasets: R@1 increases by +3.9 on DiDeMo (45.5 → 49.4), by +2.1 on MSVD (46.6 → 48.7), and by +6.2 on ActivityNet (42.7 → 48.9). On LSMDC, while R@1 slightly decreases (24.1 → 23.4), high-recall metrics improve (R@5: 41.4 → 43.3; R@10: 51.3 → 51.9) together with a lower MnR (57.6 → 56.0), indicating that DSL provides additional ranking benefits in terms of broader recall.

Table 2. Text-to-video retrieval results on DiDeMo, MSVD, LSMDC, and ActivityNet. We report standard recall metrics (R@1/R@5/R@10; higher is better) and mean rank (MnR; lower is better) under the official evaluation protocols of each benchmark. Missing metrics that are not reported in the original papers are marked as “-”. The symbol \* indicates the use of the DSL [27].
<table><tr><td rowspan="2">Method</td><td colspan="4">DiDeMo</td><td colspan="4">MSVD</td><td colspan="4">LSMDC</td><td colspan="4">ActivityNet</td></tr><tr><td>|R@1↑</td><td>R@5↑</td><td>R@10↑</td><td>MnR↓</td><td>R@1↑</td><td>R@5↑</td><td>R@10↑</td><td>MnR↓</td><td>R@1↑</td><td>R@5↑</td><td>R@10↑</td><td>MnR↓</td><td>R@1↑</td><td>R@5↑</td><td>R@10↑</td><td>MnR↓</td></tr><tr><td>CLIP4Clip [2]</td><td>42.8</td><td>68.5</td><td>79.2</td><td>18.9</td><td>45.2</td><td>75.5</td><td>84.3</td><td>10.3</td><td>22.6</td><td>41.0</td><td>49.1</td><td>61.0</td><td>40.5</td><td>72.4</td><td>83.6</td><td>7.5</td></tr><tr><td>CenterCLIP [7]</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>21.4</td><td>39.7</td><td>49.4</td><td>55.9</td><td>43.5</td><td>75.0</td><td>85.9</td><td>6.9</td></tr><tr><td>CAMoE [27]</td><td>43.8</td><td>71.4</td><td></td><td></td><td>46.9</td><td>76.1</td><td></td><td>9.8</td><td>22.5</td><td>42.6</td><td>50.9</td><td>56.5</td><td></td><td></td><td></td><td></td></tr><tr><td>CLIP2Video [10]</td><td></td><td></td><td></td><td></td><td>47.0</td><td>76.8</td><td>85.9</td><td>9.6</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>X-Pool [4]</td><td></td><td></td><td></td><td></td><td>47.2</td><td>77.4</td><td>86.0</td><td>9.3</td><td>25.2</td><td>43.7</td><td>53.5</td><td>53.2</td><td></td><td></td><td></td><td></td></tr><tr><td>TS2-Net [5]</td><td>41.8</td><td>71.6</td><td>82.0</td><td>14.8</td><td></td><td></td><td></td><td>1</td><td>23.4</td><td>42.3</td><td>50.9</td><td>56.9</td><td>41.0</td><td>73.6</td><td>84.5</td><td>8.4</td></tr><tr><td>X-CLIP [3]</td><td>45.2</td><td>74.0</td><td></td><td>14.6</td><td>47.1</td><td>77.8</td><td></td><td>9.5</td><td>23.3</td><td>43.0</td><td></td><td>56.0</td><td>44.3</td><td>74.1</td><td></td><td>7.9</td></tr><tr><td>CLIP-VIP [6]</td><td>40.6</td><td>70.4</td><td>79.3</td><td>15.1</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>UATVR [31]</td><td>43.1</td><td>71.8</td><td>82.3</td><td>15.1</td><td>46.0</td><td>76.3</td><td>85.1</td><td>10.4</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ProST [32]</td><td>44.9</td><td>72.7</td><td>82.7</td><td>13.7</td><td></td><td></td><td></td><td></td><td>24.1</td><td>42.5</td><td>51.6</td><td>54.6</td><td></td><td></td><td></td><td></td></tr><tr><td>Prompt Switch [11]</td><td></td><td></td><td></td><td></td><td>47.1</td><td>76.9</td><td>86.1</td><td>9.5</td><td>23.1</td><td>41.7</td><td>50.5</td><td>56.8</td><td></td><td></td><td></td><td></td></tr><tr><td>TeachCLIP [8]</td><td>43.7</td><td>71.2</td><td></td><td></td><td>47.4</td><td>77.3</td><td></td><td></td><td></td><td></td><td></td><td></td><td>42.2</td><td>72.7</td><td></td><td></td></tr><tr><td>EERCF [33]</td><td>44.7</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>43.1</td><td>74.5</td><td></td><td></td></tr><tr><td>LSDO [34]</td><td>46.7</td><td>71.1 75.0</td><td>81.8 82.9</td><td>14.5</td><td>44.6</td><td>75.4</td><td>84.8</td><td>10.3</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>7.1</td></tr><tr><td>GLSCL [35]</td><td></td><td></td><td></td><td>13.0</td><td>47.5</td><td>76.3</td><td>86.1</td><td>9.8</td><td>23.4</td><td>42.4</td><td>49.8</td><td>56.2</td><td>40.6</td><td>71.8</td><td>84.1</td><td></td></tr><tr><td>TAME</td><td>45.5</td><td>72.9</td><td>80.8</td><td>14.8</td><td>46.6</td><td>76.8</td><td>85.4</td><td>9.9</td><td>24.1</td><td>41.4</td><td>51.3</td><td>57.6</td><td>42.7</td><td>73.0</td><td>84.6</td><td>7.6</td></tr><tr><td>TAME *</td><td>49.4</td><td>75.1</td><td>83.0</td><td>12.9</td><td>48.7</td><td>77.9</td><td>85.9</td><td>10.4</td><td>23.4</td><td>43.3</td><td>51.9</td><td>56.0</td><td>48.9</td><td>77.0</td><td>87.0</td><td>6.4</td></tr></table>

Table 3. Ablation study of the proposed components. We progressively incorporate each module into the same CLIP backbone under identical settings (MoE: sparse upcycling; LBL: load balancing; FT: Frame–Temporal tokens; CFA: cross-frame attention (global attention over FT tokens); CTIA: sentence-conditioned temporal aggregation).
<table><tr><td rowspan="2">Configuration</td><td colspan="3">R@K↑</td><td rowspan="2">MnR↓</td></tr><tr><td>R@1</td><td>R@5</td><td>R@10</td></tr><tr><td>(1) CLIP4Clip baseline</td><td>44.0</td><td>72.0</td><td>81.6</td><td>16.0</td></tr><tr><td>(2) (1) + MoE</td><td>43.0</td><td>70.6</td><td>80.4</td><td>16.1</td></tr><tr><td>(3) (2) + LBL</td><td>43.8</td><td>71.1</td><td>80.0</td><td>16.0</td></tr><tr><td>(4) (3) + FT</td><td>42.8</td><td>71.4</td><td>80.8</td><td>16.4</td></tr><tr><td>(5) (4) + CFA</td><td>42.9</td><td>71.2</td><td>81.2</td><td>16.1</td></tr><tr><td>(6) (5) + CTIA</td><td>47.5</td><td>72.5</td><td>82.0</td><td>13.2</td></tr><tr><td>(7) (6) + Hard-negative loss</td><td>48.0</td><td>74.1</td><td>81.6</td><td>14.2</td></tr></table>

## 4.5. Ablation Study

We conduct ablations on MSR-VTT (1k-A split) to analyze the contribution of each component. The corresponding results are summarized in Table 3. Specifically, the ablation study evaluates the following components: (a) the CLIP4Clip baseline, (b) MoE in both encoders, (c) Load-balancing loss (LBL) for MoE routing, (d) Frame–Temporal (FT) tokens inserted before each visual MoE layer, (e) Cross-frame attention (CFA; global attention over FT tokens), (f) CTIA for cross-temporal interaction/aggregation, and (g) Hard-negative margin loss. Unless otherwise stated, all settings (backbone, training epochs, optimizer, and data pre/post-processing) are fixed across rows.

Starting from the CLIP4Clip baseline (Row 1), we first replace a subset of feed-forward layers in both the text and vision encoders with Mixture-of-Experts (MoE) layers (Row 2). Simply inserting MoE without any additional regularization does not yield an immediate improvement in retrieval quality. We interpret this as early evidence that naively adding sparsely activated experts can introduce routing instability and inconsistent specialization across frames and tokens, which hurts retrieval alignment.

To make this instability more explicit, Fig. 5 tracks routing imbalance over training for MoE placements {11, 12}, and Fig. 6 visualizes the expert assignment ratio for each MoE layer as bar charts over experts. We report three imbalance indicators: (i) top-1 share max<sub>i</sub> p<sub>i</sub>, which measures the dominance of the most frequently selected expert, (ii) normalized entropy H(p)/ log E (closer to 1 indicates more uniform routing), and (iii) coefficient of variation (CV) $\sigma ( \boldsymbol { p } ) / \mu ( \boldsymbol { p } )$ . Unless noted otherwise, the MoE layer {12} refers to the top CLIP backbone block and {11} denotes the second-to-top block. The diagnostics reveal a pronounced layer-selective routing collapse in Row 2, dominated by the text encoder at {12}: early in training, a single expert becomes highly dominant (top-1 share 0.75 ∼ 0.80), accompanied by reduced entropy and large dispersion (e.g., $H ( p ) / \log E \approx 0 . 5 0 \sim 0 . 5 6 , \mathrm { C V } \approx 1 . 1 7 \sim 1 . 2 8 )$ . By comparison, the {11} MoE layer is less extreme but still noticeably imbalanced (top-1 ≈ 0.50 with $\mathrm { C V } \approx 0 . 6 4 \sim 0 . 6 7 )$ indicating that the instability is amplified near the top layer rather than being uniform across MoE placements. On the vision side, the imbalance at {12} is milder (typical top-1 ≈ 0.40 with $H ( p ) / \log E \approx 0 . 8 4 \sim 0 . 8 6 )$ . Fig. 6 provides a complementary qualitative snapshot: the expert-ratio bar charts reveal that Row 2 (no LBL) assigns a disproportionate fraction of tokens to a single expert in the text MoE at {12}, while the remaining experts receive markedly smaller shares. This visual evidence is consistent with the imbalance trends quantified in Fig. 5 (e.g., elevated top-1 share and reduced entropy). Overall, the results indicate that tokenwise routing in the text encoder—particularly at the top MoE layer {12}—is a major contributor to the pronounced performance drop observed when MoE is introduced without stabilization/regularization.

![](images/961c23839ae2358e9c2195b92a911d9e21c74ac48bb396bb122efe160f8b10bb.jpg)  
Figure 5. Routing distribution metrics over training for MoE placements {11, 12}. We plot the top-1 share max<sub>i</sub> p<sub>i</sub>, normalized entropy $H ( p ) / \log E ,$ and coefficient of variation $\sigma ( \boldsymbol { p } ) / \mu ( \boldsymbol { p } )$ , where p<sub>i</sub> denotes the average routing probability assigned to expert i and E is the number of experts. Without the auxiliary load-balancing loss (Row2), the routing distribution becomes persistently skewed, indicating layer-selective collapse. With load-balancing (Row3), the routing distribution remains stable and close to uniform throughout training.

To address this, we add a load-balancing loss (LBL) on the expert router (Row 3). LBL encourages more uniform expert utilization and prevents the router from collapsing onto a few experts. With this regularization, performance stabilizes and the distribution of active experts becomes more balanced. In other words, balanced expert assignment is important for maintaining stable text–video alignment.

![](images/f6d8011384d8ee92075d8fcb974d6d56fa27b38629b8ac92076e2d5c7faa67df.jpg)  
Figure 6. Expert assignment ratios for MoE placements {11, 12}. We visualize the per-expert routing distribution $\boldsymbol { p } \in \mathbb { R } ^ { E }$ for both text and vision encoders during the initial fine-tuning stage. Row 2 (no LBL) exhibits strong routing concentration, most prominently in the text MoE at {12}, whereas Row 3 (with LBL) yields near-uniform utilization across experts. This figure reports the corresponding imbalance trajectories over training.

This stabilization is also confirmed by the routing diagnostics: with LBL (Row 3), routing at the same upper-layer MoE placements {11, 12} remains near-uniform throughout training. In particular, both {11} and {12} exhibit entrop $_ { \mathrm { { n o r m } } } \approx 0 . 9 9 \sim 1 . 0 0$ with low top-1 share and low CV, indicating balanced expert utilization and preventing router collapse. This consistent behavior supports that LBL stabilizes optimization and mitigates the early degradation observed in Row 2.

We note that Row 3 does not yet surpass the dense CLIP4Clip baseline (Row 1), even though routing collapse is effectively mitigated. This is expected: LBL mainly serves as a stabilizer that prevents degenerate expert utilization and ensures consistent optimization, rather than directly boosting retrieval alignment on its own. In the early fine-tuning stage on MSR-VTT, MoE also introduces additional degrees of freedom (router and expert specialization), which can delay immediate gains. Accordingly, the severe early degradation in Row 2 is best explained by unstabilized routing collapse, whereas Row 3 should be viewed as a necessary intermediate step that makes the MoE backbone reliably trainable. In particular, without LBL the top-layer text router quickly saturates to a single expert, leading to highly non-smooth gradients and unstable token-level alignment early in training (Fig. 5–6).

Building on this stabilized MoE backbone, we next assess the incremental impact of temporal modeling (Rows 4–5). We then insert Frame–Temporal (FT) tokens before each visual MoE layer (Row 4) so that each frame representation is augmented with lightweight temporal tokens. However, using FT tokens alone (Row 4) does not materially improve retrieval quality, since FT tokens enrich each frame in isolation but does not explicitly exchange information across frames. To address this limitation, we add cross-frame attention (CFA) (Row 5), which allows these tokens to exchange information across frames. CFA stabilizes performance and prevents further degradation in rank metrics, indicating that explicit inter-frame exchange is important for temporal modeling.

Row 6 introduces our Cross-Temporal Interaction & Aggregation (CTIA) module, which aggregates sentence–frame similarity over time and produces a video-level score. At this stage we observe a clear improvement over the baseline: the model ranks relevant videos higher on average. Taken together with Rows 4–5, these results suggest that strengthening frame-level representations (via FT and CFA) is helpful but not sufficient on its own; the retrieval model benefits most when those per-frame signals are subsequently organized across time. CTIA performs this temporal organization by aggregating and propagating evidence across frames to produce a more coherent video-level match. Rather than relying only on a single pooled embedding, it highlights the frames that are most relevant to the text query and links them into a consistent trajectory. Finally, Row 7 adds a hard negative margin loss, which penalizes cases where non-matching videos are scored too close to the ground-truth match. This further sharpens retrieval by explicitly separating hard distractors, especially among visually similar candidates.

Table 4. Analysis of the effect of MoE layer placement by varying only the encoder layers in which FFNs are upcycled to MoE, while fixing E, top-K routing, and all other settings.
<table><tr><td>Layers</td><td>R@1↑</td><td>R@5↑</td><td>R@10↑</td><td>MnR↓</td></tr><tr><td>{12}</td><td>47.9</td><td>73.7</td><td>82.2</td><td>14.2</td></tr><tr><td>{9,11}</td><td>47.4</td><td>73.7</td><td>81.9</td><td>14.1</td></tr><tr><td>{11,12} (def.)</td><td>48.0</td><td>74.1</td><td>81.6</td><td>14.2</td></tr><tr><td>{8,10,12}</td><td>46.4</td><td>72.0</td><td>81.7</td><td>13.4</td></tr><tr><td>{10,11,12}</td><td>45.9</td><td>73.3</td><td>81.5</td><td>14.0</td></tr><tr><td>{7-12}</td><td>45.8</td><td>73.6</td><td>82.1</td><td>13.9</td></tr></table>

Table 5. Effect of the number of MoE experts, measured by varying only E in the upcycled FFNs, while keeping the MoE placement, top-K routing, and all other settings fixed.
<table><tr><td> $N _ { E }$ </td><td>R@1↑</td><td>R@5↑</td><td>R@10↑</td><td>MdR↓</td><td>MnR↓</td></tr><tr><td>2</td><td>46.7</td><td>72.2</td><td>82.7</td><td>2.0</td><td>13.9</td></tr><tr><td>4</td><td>48.0</td><td>74.1</td><td>81.6</td><td>2.0</td><td>14.2</td></tr><tr><td>6</td><td>46.8</td><td>72.3</td><td>82.0</td><td>2.0</td><td>14.3</td></tr><tr><td>8</td><td>46.4</td><td>73.1</td><td>81.7</td><td>2.0</td><td>13.6</td></tr></table>

Table 6. Effect of the number of Frame–Temporal Tokens. The number of tokens inserted per frame is varied to control temporal context modeling, while keeping the CLIP backbone and other modules fixed.
<table><tr><td>FT</td><td>R@1↑</td><td>R@5↑</td><td>R@10↑</td><td>MdR↓</td><td>MnR↓</td></tr><tr><td>0</td><td>47.4</td><td>73.7</td><td>81.4</td><td>2.0</td><td>13.4</td></tr><tr><td>1</td><td>47.1</td><td>72.6</td><td>81.5</td><td>2.0</td><td>14.4</td></tr><tr><td>2</td><td>48.0</td><td>74.1</td><td>81.6</td><td>2.0</td><td>14.2</td></tr><tr><td>3</td><td>47.1</td><td>73.0</td><td>82.1</td><td>2.0</td><td>13.7</td></tr></table>

Table 7. CTIA β-term ablation on MSR-VTT (1k-A), evaluating the contribution of each relevance component (r, r<sub>conv</sub>, r<sub>graph</sub>) and their combinations.
<table><tr><td rowspan=2 colspan=1>β-termβr βc  $\beta _ { g }$ </td><td rowspan=2 colspan=1>T→VR@1↑R@5↑MnR↓</td><td rowspan=1 colspan=1>V→T</td></tr><tr><td rowspan=1 colspan=1>R@1↑R@5↑MnR↓</td></tr><tr><td rowspan=1 colspan=1>1 1  1</td><td rowspan=1 colspan=1>48.0  74.1   14.2</td><td rowspan=1 colspan=1>46.9  72.7  10.6</td></tr><tr><td rowspan=1 colspan=1>1 0 0</td><td rowspan=1 colspan=1>47.6  72.7   14.3</td><td rowspan=1 colspan=1>46.9  73.8   9.7</td></tr><tr><td rowspan=1 colspan=1>0 1 0</td><td rowspan=1 colspan=1>47.7  73.4  13.5</td><td rowspan=1 colspan=1>47.2  73.0   9.9</td></tr><tr><td rowspan=1 colspan=1>0 0 1</td><td rowspan=1 colspan=1>47.1   72.8  14.0</td><td rowspan=1 colspan=1>47.5  73.0  10.4</td></tr><tr><td rowspan=1 colspan=1>1 1  0</td><td rowspan=1 colspan=1>47.3  72.9   13.6</td><td rowspan=1 colspan=1>45.8  72.9   9.9</td></tr><tr><td rowspan=1 colspan=1>1 0 1</td><td rowspan=1 colspan=1>47.0  72.9   13.8</td><td rowspan=1 colspan=1>46.7  71.9  10.8</td></tr><tr><td rowspan=1 colspan=1>0 1 1</td><td rowspan=1 colspan=1>46.5  73.1  13.4</td><td rowspan=1 colspan=1>46.7  72.8  10.0</td></tr></table>

## 4.6. Ablation on MoE Layers and Experts

We evaluate different placements of the MoE layers in both the text and vision encoders, as shown in Table 4. Concentrating MoE at the top two backbone layers {11, 12} provides the most balanced performance in terms of R@1 and R@5 $( \mathrm { R } @ 1 { = } 4 8 . 0 , \mathrm { R } @ 5 { = } 7 4 . 1 )$ ). Using only the top layer {12} yields a close R@1 of 47.9 but a slightly lower R@10. By contrast, distributing MoE more broadly (e.g., {7–12}, {8, 10, 12}) does not improve recall and tends to increase variance in mean rank (MnR). These results suggest that lower layers mainly encode low-level patterns, whereas upper layers provide more disentangled semantic information where expert modules can specialize most effectively. Therefore, rather than distributing MoE throughout the backbone, concentrating it near the upper layers is a more effective design choice for leveraging expert capacity at the representation level.

Table 5 presents the ablation on the number of experts. We observe that using four experts yields the most balanced performance overall, while changing the count to 2, 6, or 8 does not lead to consistent improvements. In all experiments, we set the number of experts in the visual and text encoders to be identical, and the setting $N _ { E } = 4$ is used as the default configuration.

## 4.7. Ablation on Frame–Temporal Tokens

Table 6 shows that using two FT tokens yields the strongest R@1/R@5 (48.0/74.1 with MdR=2.0) and the most balanced overall profile, while one token performs slightly lower (notably higher MnR) and three tokens raise R@10 (82.1) without consistent gains in R@1/R@5, suggesting FT=2 as a practical default.

## 4.8. Ablation on CTIA β-Terms

Table 7 ablates the CTIA fusion terms by enabling each relevance component $( { \bf r } , { \bf r } _ { \mathrm { c o n v } } , { \bf r } _ { \mathrm { g r a p h } } )$ with binary coefficients $( \beta _ { r } , \beta _ { c } , \beta _ { g } )$ . Using all terms jointly (1, 1, 1) achieves the best Text→Video performance (48.0/74.1 in R@1/R@5), indicating that raw evidence, local smoothing, and graph propagation provide complementary cues. Single-term settings remain competitive but show different tendencies: r yields the best MnR on Text→Video among single terms (13.5), while $\bf { r } _ { \mathrm { g r a p h } }$ gives the best Video→Text R@1 (47.5). Overall, (1, 1, 1) provides the most balanced bidirectional profile and is used as the default.

## 4.9. Unified Model-Scale and Compute Comparison

Table 8 provides a unified comparison between representative unified video–language models (All-in-One [49], ALPRO [50]) and CLIP-based dual-encoder approaches (CLIP4Clip, TAME) in terms of model scale (Params), compute (GFLOPs/sample), and retrieval performance on MSR-VTT (1k-A) and DiDeMo. FLOPs are computed with fvcore (FlopCountAnalysis) under each repository’s retrieval configuration, and the profiling frame counts follow common practice (All-in-One=3, ALPRO=8, CLIP4Clip/TAME=12). Importantly, while our MoE design increases the total number of trainable parameters by introducing multiple experts, sparse Top-K routing activates only a small subset of experts per token/sample; therefore, the per-sample compute does not scale linearly with the total number of experts E, and is instead primarily governed by K and the number of MoE-inserted blocks (as well as any additional token/mixing modules). Under this unified setting, the unified baselines exhibit a markedly higher compute cost (e.g., ALPRO: 208.36 GFLOPs/sample) yet lower reported R@K than CLIP4Clip and TAME. In contrast, TAME achieves consistent gains over CLIP4Clip with a moderate increase in scale/compute (163.93M/54.12 → 205.40M/77.59), improving MSR-VTT R@1 from 44.0 to 48.0 and reducing MnR from 16.0 to 14.2, while also improving DiDeMo R@1 from 42.8 to 45.5 and MnR from 18.9 to 14.8. Moreover, TAME outperforms the larger unified baseline ALPRO while requiring ∼2.7× fewer GFLOPs per sample (208.36 → 77.59), demonstrating a favorable accuracy–compute trade-off.

Table 8. Unified comparison of model scale/compute and retrieval performance on MSR-VTT (1k-A) and DiDeMo. FLOPs are computed with fvcore (FlopCountAnalysis) using a profiling batch that follows each repository’s retrieval configuration. GFLOPs/sample is computed by dividing the total FLOPs of one forward pass (for a profiling batch) by the batch size. Frames indicate the number of frames used in profiling (All-in-One=3, ALPRO=8, CLIP4Clip/TAME=12). “–” indicates metrics not reported.
<table><tr><td>Model</td><td></td><td>Frames Params (M)</td><td>GFLOPs/sample</td><td colspan="4">MSR-VTT (1k-A)</td><td colspan="4">DiDeMo</td></tr><tr><td></td><td></td><td></td><td></td><td>R@1↑</td><td>R@5↑</td><td>R@10↑</td><td>MnR↓</td><td>R@1↑</td><td>R@5↑</td><td>R@10↑</td><td>MnR↓</td></tr><tr><td>All-in-One [49]</td><td>3</td><td>109.87</td><td>63.91</td><td>37.9</td><td>68.1</td><td>77.1</td><td>1</td><td>32.7</td><td>61.4</td><td>73.5</td><td>一</td></tr><tr><td>ALPRO [50]</td><td>8</td><td>231.48</td><td>208.36</td><td>33.9</td><td>60.7</td><td>73.2</td><td></td><td>35.9</td><td>67.5</td><td>78.8</td><td></td></tr><tr><td>CLIP4Clip</td><td>12</td><td>163.93</td><td>54.12</td><td>44.0</td><td>72.0</td><td>81.6</td><td>16.0</td><td>42.8</td><td>68.5</td><td>79.2</td><td>18.9</td></tr><tr><td>TAME</td><td>12</td><td>205.40</td><td>77.59</td><td>48.0</td><td>74.1</td><td>81.6</td><td>14.2</td><td>45.5</td><td>72.9</td><td>80.8</td><td>14.8</td></tr></table>

Table 9. Frame permutation ablation on MSR-VTT (1k-A): Text→Video (T→V) raw performance.
<table><tr><td rowspan="2">Model</td><td colspan="4">Ordinary</td><td colspan="5">Reverse</td><td colspan="5"></td></tr><tr><td>R@1↑</td><td>R@5↑</td><td>R@10↑</td><td>MnR↓</td><td>R@1↑</td><td>R@5↑</td><td>R@10↑</td><td>MnR↓</td><td>∆R@1</td><td>R@1↑</td><td>R@5↑</td><td>R@10↑</td><td>MnR↓</td><td>∆R@1</td></tr><tr><td>CLIP4Clip</td><td>44.0</td><td>72.0</td><td>81.6</td><td>16.0</td><td>43.6</td><td>71.9</td><td>82.0</td><td>16.1</td><td>-0.9%</td><td>43.6</td><td>72.0</td><td>82.0</td><td>16.1</td><td>-0.9%</td></tr><tr><td>TAME</td><td>48.0</td><td>74.1</td><td>81.6</td><td>14.2</td><td>47.1</td><td>73.9</td><td>81.6</td><td>14.2</td><td>-1.9%</td><td>47.6</td><td>74.0</td><td>81.7</td><td>14.3</td><td>-0.8%</td></tr></table>

## 4.10. Frame Permutation Ablation (Temporal-Order Robustness)

Table 9 reports a frame permutation ablation on MSR-VTT (1k-A) under the raw T→V protocol. We fix the sampled frame set for each video and permute only the temporal order at inference (ordinary/reverse/random), thereby isolating the effect of frame ordering from changes in visual evidence. Overall, both methods show limited sensitivity to order perturbations; however, TAME exhibits a slightly larger degradation under the fully reversed order. Specifically, CLIP4Clip decreases by 0.9% in R@1 (44.0→43.6), whereas TAME decreases by 1.9% (48.0→47.1). In contrast, under random permutation, the R@1 change of TAME is small (-0.8%, 48.0→47.6) and comparable to the baseline (-0.9%). These results indicate that performance degradation remains limited under temporal-order perturbations, and the overall effect of our method is largely preserved.

This trend is consistent with the design of TAME: we perform content-based cross-frame interaction and aggregation (global mixing over FT tokens and CTIA), while still being able to exploit directional temporal cues to some extent, which can explain why a complete reversal yields a slightly larger drop than a random shuffle. Moreover, under the widely used sparse-frame MSR-VTT protocol, many queries emphasize overall video semantics rather than strict event chronology; therefore, this ablation is best interpreted as a robustness-to-order test (with mild order sensitivity).

## 4.11. Masked-Frame Ablation (Robustness to Missing Temporal Evidence)

Table 10 reports a masked-frame (black-out) ablation on MSR-VTT (1k-A) under the raw T→V protocol. We keep the sampled frame set and the retrieval pipeline unchanged, and apply corruption only at inference by randomly selecting a portion of the sampled frames and replacing them with a black-out mask (mask ratio ∈ {0.1, 0.2, 0.3, 0.4, 0.5, 0.6}). This test evaluates robustness to missing/occluded temporal evidence, rather than temporal-order sensitivity.

As the mask ratio increases, both methods degrade (lower R@K and higher MnR), with a noticeably larger drop beyond 0.4. Importantly, TAME maintains higher absolute performance than CLIP4Clip at all mask ratios. Moreover, under heavy masking, TAME shows a smaller relative R@1 drop compared to the unmasked setting: at 0.5, CLIP4Clip decreases by 11.73% (44.0→38.84), whereas TAME decreases by 9.63% (48.0→43.38); at 0.6, CLIP4Clip decreases by 18.91% (44.0→35.68) versus 16.29% for TAME (48.0→40.18). Ranking quality is also better preserved by TAME under severe corruption, as reflected by a smaller MnR increase at 0.6 (+5.24 for CLIP4Clip vs. +3.40 for TAME).

Table 10. Masked-frame (black-out) test on MSR-VTT (1k-A): Text→Video (T→V) average raw performance under different mask ratios.
<table><tr><td rowspan="2">Mask ratio</td><td colspan="5">CLIP4Clip</td><td colspan="5">TAME</td></tr><tr><td>R@1↑</td><td>R@5↑</td><td>R@10↑</td><td>MnR↓</td><td>∆R@1 (vs. 0.0)</td><td>R@1↑</td><td>R@5↑</td><td>R@10↑</td><td>MnR↓</td><td>∆R@1 (vs. 0.0)</td></tr><tr><td>0.0</td><td>44.00</td><td>72.00</td><td>81.60</td><td>16.00</td><td></td><td>48.00</td><td>74.10</td><td>81.60</td><td>14.20</td><td></td></tr><tr><td>0.1</td><td>43.20</td><td>70.96</td><td>81.30</td><td>16.06</td><td>-1.82%</td><td>46.48</td><td>73.52</td><td>81.54</td><td>14.28</td><td>-3.17%</td></tr><tr><td>0.2</td><td>42.34</td><td>70.40</td><td>80.54</td><td>16.34</td><td>-3.77%</td><td>46.10</td><td>72.60</td><td>81.20</td><td>14.40</td><td>-3.96%</td></tr><tr><td>0.3</td><td>42.02</td><td>69.36</td><td>79.30</td><td>17.04</td><td>-4.50%</td><td>45.52</td><td>71.44</td><td>81.24</td><td>14.88</td><td>-5.17%</td></tr><tr><td>0.4</td><td>40.10</td><td>68.08</td><td>77.82</td><td>18.30</td><td>-8.86%</td><td>44.10</td><td>70.36</td><td>80.28</td><td>15.50</td><td>-8.13%</td></tr><tr><td>0.5</td><td>38.84</td><td>66.60</td><td>77.16</td><td>18.92</td><td>-11.73%</td><td>43.38</td><td>70.80</td><td>80.08</td><td>16.08</td><td>-9.63%</td></tr><tr><td>0.6</td><td>35.68</td><td>62.00</td><td>72.88</td><td>24.16</td><td>-18.91%</td><td>40.18</td><td>68.14</td><td>78.12</td><td>19.48</td><td>-16.29%</td></tr></table>

These results suggest that TAME is less sensitive when a substantial portion of frame-level evidence is removed. Our final similarity integrates a global video–text baseline score with a framewise evidence aggregation score, which reduces over-reliance on any single frame cue and enables more stable consolidation of the remaining informative frames under partial evidence removal. In addition, blacked-out frames typically yield weak similarity signals due to the loss of visual content, and thus receive low influence during framewise consolidation, helping maintain ranking stability when temporal evidence is partially missing.

## 5. Conclusion

In this paper, we presented TAME, a temporal-aware extension of CLIP for text–video retrieval that bridges imagelevel pre-training and video-level semantics. Building on a CLIP backbone, TAME couples frame-wise Mixture-of-Experts (MoE) layers with frame-consistent routing, Frame– Temporal (FT) tokens, and a cross-temporal interaction and aggregation mechanism (CTIA) to model temporal structure without sacrificing stable vision–language alignment.

Extensive experiments on TVR benchmarks demonstrate that TAME consistently outperforms CLIP-based baselines, with notable gains on MSR-VTT and additional improvements on DiDeMo, MSVD, LSMDC and ActivityNet. These results suggest that the consistent performance improvements observed across diverse benchmarks with different characteristics indicate our method is not a dataset-specific technique, but rather a generalizable approach.

Limitations and Future Work. TAME improves retrieval by expanding encoder capacity via sparse MoE and organizing frame evidence via lightweight temporal tokens and CTIA, but it introduces trade-offs in routing stability and temporal-token overhead. Our ablations suggest that MoE benefits require careful stabilization (e.g., load balancing), and temporal mixing must be applied selectively to control memory cost. Moreover, CTIA is efficient and interpretable, yet it relies on the quality of framewise similarities and a small set of hyperparameters. Future work will focus on more efficient temporal mixing for long videos, more robust/adaptive routing, and dynamic compute allocation (e.g., learned frame sampling or token budgeting), as well as extending CTIA-style aggregation to temporally grounded or streaming retrieval. In addition, we will explore efficiencyoriented model compression strategies such as pruning, quantization, and knowledge distillation to develop lightweight variants of TAME that retain high retrieval accuracy with reduced model size and inference cost.

## Acknowledgements

This work was partly supported by the Institute of Information and Communications Technology Planning and Evaluation (IITP) grant funded by the Korea government (MSIT) under Grant No. 2710017875 (Development of multimodal data input-based search augmentation generation technology, 50%) and Grant No. IITP-2025-RS-2023-00254529 (Metaverse support program to nurture the best talents, 25%), and by the National Research Foundation of Korea (NRF) grant funded by the Korea government (MSIT) under Grant No. RS-2025-00553785 (Research on the core technology of data integration and augmentation for OTT user analysis, 25%).

## References

[1] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, G. Krueger, and I. Sutskever, “Learning transferable visual models from natural language supervision,” in Proceedings

of the 38th International Conference on Machine Learning, vol. 139 of Proceedings of Machine Learning Research, pp. 8748–8763, 2021.

[2] H. Luo, L. Ji, M. Zhong, Y. Chen, W. Lei, N. Duan, and T. Li, “Clip4clip: An empirical study of clip for end to end video clip retrieval and captioning,” Neurocomputing, vol. 508, pp. 293– 304, 2022.

[3] Y. Ma, G. Xu, X. Sun, M. Yan, J. Zhang, and R. Ji, “Xclip: End-to-end multi-grained contrastive learning for videotext retrieval,” in Proceedings of the 30th ACM International Conference on Multimedia, MM ’22, (New York, NY, USA), pp. 638–647, Association for Computing Machinery, 2022.

[4] S. K. Gorti, N. Vouitsis, J. Ma, K. Golestan, M. Volkovs, A. Garg, and G. Yu, “X-pool: Cross-modal language-video attention for text-video retrieval,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 5006–5015, June 2022.

[5] Y. Liu, P. Xiong, L. Xu, S. Cao, and Q. Jin, “Ts2-net: Token shift and selection transformer for text-video retrieval,” in Computer Vision – ECCV 2022 (S. Avidan, G. Brostow, M. Cisse, G. M. Farinella, and T. Hassner, eds.), (Cham),´ pp. 319–335, Springer Nature Switzerland, 2022.

[6] H. Xue, Y. Sun, B. Liu, J. Fu, R. Song, H. Li, and J. Luo, “Clip-vip: Adapting pre-trained image-text model to video-language representation alignment,” arXiv preprint arXiv:2209.06430, 2022.

[7] S. Zhao, L. Zhu, X. Wang, and Y. Yang, “Centerclip: Token clustering for efficient text-video retrieval,” in Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval, SIGIR ’22, (New York, NY, USA), pp. 970–981, Association for Computing Machinery, 2022.

[8] K. Tian, R. Zhao, Z. Xin, B. Lan, and X. Li, “Holistic features are almost sufficient for text-to-video retrieval,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 17138–17147, 2024.

[9] X. Jin, B. Zhang, W. Gong, K. Xu, X. Deng, P. Wang, Z. Zhang, X. Shen, and J. Feng, “Mv-adapter: Multimodal video transfer learning for video text retrieval,” in Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 27144–27153, 2024.

[10] H. Fang, P. Xiong, L. Xu, and Y. Chen, “Clip2video: Mastering video-text retrieval via image clip,” arXiv preprint arXiv:2106.11097, 2021.

[11] C. Deng, Q. Chen, P. Qin, D. Chen, and Q. Wu, “Prompt switch: Efficient clip adaptation for text-video retrieval,” in Proceedings ofthe IEEE/CVF International Conference on Computer Vision (ICCV), pp. 15648–15658, October 2023.

[12] W. Wu, H. Luo, B. Fang, J. Wang, and W. Ouyang, “Cap4video: What can auxiliary captions do for text-video retrieval?,” in Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 10704– 10713, June 2023.

[13] J. Hestness, S. Narang, N. Ardalani, G. Diamos, H. Jun, H. Kianinejad, M. M. A. Patwary, Y. Yang, and Y. Zhou, “Deep learning scaling is predictable, empirically,” arXiv preprint arXiv:1712.00409, 2017.

[14] T. Brown, B. Mann, N. Ryder, M. Subbiah, J. D. Kaplan, P. Dhariwal, A. Neelakantan, P. Shyam, G. Sastry, A. Askell, S. Agarwal, A. Herbert-Voss, G. Krueger, T. Henighan, R. Child, A. Ramesh, D. Ziegler, J. Wu, C. Winter, C. Hesse, M. Chen, E. Sigler, M. Litwin, S. Gray, B. Chess, J. Clark, C. Berner, S. McCandlish, A. Radford, I. Sutskever, and D. Amodei, “Language models are few-shot learners,” in Advances in Neural Information Processing Systems (H. Larochelle, M. Ranzato, R. Hadsell, M. Balcan, and H. Lin, eds.), vol. 33, pp. 1877–1901, Curran Associates, Inc., 2020.

[15] J.-B. Alayrac, J. Donahue, P. Luc, A. Miech, I. Barr, Y. Hasson, K. Lenc, A. Mensch, K. Millican, M. Reynolds, R. Ring, E. Rutherford, S. Cabi, T. Han, Z. Gong, S. Samangooei, M. Monteiro, J. L. Menick, S. Borgeaud, A. Brock, A. Nematzadeh, S. Sharifzadeh, M. a. Binkowski, R. Barreira,´ O. Vinyals, A. Zisserman, and K. Simonyan, “Flamingo: a visual language model for few-shot learning,” in Advances in Neural Information Processing Systems (S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh, eds.), vol. 35, pp. 23716–23736, Curran Associates, Inc., 2022.

[16] J. Kaplan, S. McCandlish, T. Henighan, T. B. Brown, B. Chess, R. Child, S. Gray, A. Radford, J. Wu, and D. Amodei, “Scaling laws for neural language models,” arXiv preprint arXiv:2001.08361, 2020.

[17] J. Hoffmann, S. Borgeaud, A. Mensch, E. Buchatskaya, T. Cai, E. Rutherford, D. de Las Casas, L. A. Hendricks, J. Welbl, A. Clark, T. Hennigan, E. Noland, K. Millican, G. van den Driessche, B. Damoc, A. Guy, S. Osindero, K. Simonyan, E. Elsen, O. Vinyals, J. Rae, and L. Sifre, “An empirical analysis of compute-optimal large language model training,” in Advances in Neural Information Processing Systems (S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh, eds.), vol. 35, pp. 30016–30030, Curran Associates, Inc., 2022.

[18] N. Shazeer, A. Mirhoseini, K. Maziarz, A. Davis, Q. Le, G. Hinton, and J. Dean, “Outrageously large neural networks: The sparsely-gated mixture-of-experts layer,” arXiv preprint arXiv:1701.06538, 2017.

[19] C. Riquelme, J. Puigcerver, B. Mustafa, M. Neumann, R. Jenatton, A. Susano Pinto, D. Keysers, and N. Houlsby, “Scaling vision with sparse mixture of experts,” in Advances in Neural Information Processing Systems (M. Ranzato, A. Beygelzimer, Y. Dauphin, P. Liang, and J. W. Vaughan, eds.), vol. 34, pp. 8583–8595, Curran Associates, Inc., 2021.

[20] G. Jain, N. Hegde, A. Kusupati, A. Nagrani, S. Buch, P. Jain, A. Arnab, and S. Paul, “Mixture of nested experts: Adaptive processing of visual tokens,” Advances in Neural Information Processing Systems, vol. 37, pp. 58480–58497, 2024.

[21] J. Zhang, X. Qu, T. Zhu, and Y. Cheng, “CLIP-MoE: Towards building mixture of experts for CLIP with diversified multiplet upcycling,” in Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing (C. Christodoulopoulos, T. Chakraborty, C. Rose, and V. Peng, eds.), (Suzhou, China), pp. 5406–5419, Association for Computational Linguistics, Nov. 2025.

[22] X. Wang, C. Chen, Y. Yang, H.-Y. Chen, B. Zhang, A. Pal, X. Zhu, and X. Du, “CLIP-UP: A simple and efficient mixture-

of-experts CLIP training recipe with sparse upcycling,” in Findings of the Association for Computational Linguistics: EMNLP 2025 (C. Christodoulopoulos, T. Chakraborty, C. Rose, and V. Peng, eds.), (Suzhou, China), pp. 21186– 21200, Association for Computational Linguistics, Nov. 2025.

[23] S¸. Ozt <sup>¨</sup> urk, “Hash code generation using deep feature selec-¨ tion guided siamese network for content based medical image retrieval,” GAZI UNIVERSITY JOURNAL OF SCIENCE, vol. 34, pp. 733–746, 08 2021.

[24] S¸ . Ozt <sup>¨</sup> urk, “Class-driven content-based medical image re-¨ trieval using hash codes of deep features,” Biomedical Signal Processing and Control, vol. 68, p. 102601, 2021.

[25] S¸ . Ozt<sup>¨</sup> urk, A. Alhudhaif, and K. Polat, “Attention-based end-¨ to-end cnn framework for content-based x-ray imageretrieval,” Turkish Journal ofElectrical Engineering and Computer Sciences, vol. 29, no. 8, pp. 2680–2693, 2021.

[26] W. Fedus, B. Zoph, and N. Shazeer, “Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity,” Journal of Machine Learning Research, vol. 23, no. 120, pp. 1–39, 2022.

[27] X. Cheng, H. Lin, X. Wu, F. Yang, and D. Shen, “Improving video-text retrieval by multi-stream corpus alignment and dual softmax loss,” arXiv preprint arXiv:2109.04290, 2021.

[28] S.-V. Bogolin, I. Croitoru, H. Jin, Y. Liu, and S. Albanie, “Cross modal retrieval with querybank normalisation,” in 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 5184–5195, IEEE Computer Society, 2022.

[29] Q. Wang, Y. Zhang, Y. Zheng, P. Pan, and X.-S. Hua, “Disentangled representation learning for text-video retrieval,” arXiv preprint arXiv:2203.07111, 2022.

[30] R. Liu, J. Huang, G. Li, J. Feng, X. Wu, and T. H. Li, “Revisiting temporal modeling for clip-based image-to-video knowledge transferring,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 6555–6564, 2023.

[31] B. Fang, W. Wu, C. Liu, Y. Zhou, Y. Song, W. Wang, X. Shu, X. Ji, and J. Wang, “Uatvr: Uncertainty-adaptive text-video retrieval,” in 2023 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 13677–13687, IEEE, 2023.

[32] P. Li, C.-W. Xie, L. Zhao, H. Xie, J. Ge, Y. Zheng, D. Zhao, and Y. Zhang, “Progressive spatio-temporal prototype matching for text-video retrieval,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 4100–4110, October 2023.

[33] K. Tian, Y. Cheng, Y. Liu, X. Hou, Q. Chen, and H. Li, “Towards efficient and effective text-to-video retrieval with coarse-to-fine visual representation learning,” in Proceedings ofthe AAAI Conference on Artificial Intelligence, pp. 5207– 5214, March 2024.

[34] Y. Zheng, B. Huang, Z. Chen, and D. Yu, “Enhancing textvideo retrieval performance with low-salient but discriminative objects,” IEEE Transactions on Image Processing, vol. 34, pp. 581–593, 2025.

[35] H. Zhang, P. Zeng, L. Gao, J. Song, Y. Duan, X. Lyu, and H. T. Shen, “Text-video retrieval with global-localsemantic consistent learning,” IEEE Transactions on Image Processing, vol. 34, pp. 3463–3474, 2025.

[36] Y. Wang, K. Li, X. Li, J. Yu, Y. He, G. Chen, B. Pei, R. Zheng, Z. Wang, Y. Shi, T. Jiang, S. Li, J. Xu, H. Zhang, Y. Huang, Y. Qiao, Y. Wang, and L. Wang, “Internvideo2: Scaling foundation models for multimodal video understanding,” in Computer Vision – ECCV 2024, pp. 396–416, 2025.

[37] A. v. d. Oord, Y. Li, and O. Vinyals, “Representation learning with contrastive predictive coding,” arXiv preprint arXiv:1807.03748, 2018.

[38] A. Hermans, L. Beyer, and B. Leibe, “In defense of the triplet loss for person re-identification,” arXiv preprint arXiv:1703.07737, 2017.

[39] W. Liu, X. Tang, and X. Ren, “Trajectory smoothing constraint and hard negative mining for distractor-aware regression tracking,” IEEE Access, vol. 7, pp. 84658–84667, 2019.

[40] J. Xu, T. Mei, T. Yao, and Y. Rui, “Msr-vtt: A large video description dataset for bridging video and language,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), June 2016.

[41] V. Gabeur, C. Sun, K. Alahari, and C. Schmid, “Multi-modal transformer for video retrieval,” in European Conference on Computer Vision, pp. 214–229, Springer, 2020.

[42] L. Anne Hendricks, O. Wang, E. Shechtman, J. Sivic, T. Darrell, and B. Russell, “Localizing moments in video with natural language,” in Proceedings of the IEEE International Conference on Computer Vision (ICCV), Oct 2017.

[43] D. Chen and W. Dolan, “Collecting highly parallel data for paraphrase evaluation,” in Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies (D. Lin, Y. Matsumoto, and R. Mihalcea, eds.), (Portland, Oregon, USA), pp. 190–200, Association for Computational Linguistics, June 2011.

[44] A. Rohrbach, M. Rohrbach, N. Tandon, and B. Schiele, “A dataset for movie description,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), June 2015.

[45] F. Caba Heilbron, V. Escorcia, B. Ghanem, and J. Carlos Niebles, “Activitynet: A large-scale video benchmark for human activity understanding,” in Proceedings of the ieee conference on computer vision and pattern recognition, pp. 961–970, 2015.

[46] B. Zhang, H. Hu, and F. Sha, “Cross-modal and hierarchical modeling of video and text,” in Proceedings of the european conference on computer vision (ECCV), pp. 374–390, 2018.

[47] D. P. Kingma, “Adam: A method for stochastic optimization,” arXiv preprint arXiv:1412.6980, 2014.

[48] I. Loshchilov and F. Hutter, “Sgdr: Stochastic gradient descent with warm restarts,” arXiv preprint arXiv:1608.03983, 2016.

[49] A. J. Wang, Y. Ge, R. Yan, Y. Ge, X. Lin, G. Cai, J. Wu, Y. Shan, X. Qie, and M. Z. Shou, “All in one: Exploring unified video-language pre-training,” arXiv preprint arXiv:2203.07303, 2022.

[50] D. Li, J. Li, H. Li, J. C. Niebles, and S. C. Hoi, “Align and prompt: Video-and-language pre-training with entity prompts,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 4953–4963, 2022.