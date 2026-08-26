![](images/68fc1c8fa05ce8c85c0fc40191cff3cb9b6ee153d0b028bbab322cf2788def96.jpg)

# WeMM-Embedding: WeChat Multi-Modal Embedding Technical Report

Junjie Zhou<sup>∗</sup>, Ke Mei<sup>∗†</sup>, Lei Li<sup>∗</sup>, Tianyi Wang, Fengyun Rao<sup>‡</sup>, Jing LYU WeChat Vision, Tencent Inc.

## Abstract

Universal multimodal embeddings are becoming a core component of modern AI systems, enabling heterogeneous content to be represented in a shared space for applications such as retrieval, recom mendation, classification, and agentic systems. In this report, we present WeMM-Embedding, a family of universal multimodal embedding models supporting text, images, videos, visual documents, and arbitrarily interleaved multimodal inputs with flexible output dimensions. The family comprises 2B, 4B, and 9B variants and is trained in two stages: a large-scale multimodal align ment stage, followed by a refinement stage using curated data, fine-grained relevance supervision, and cross-scale knowledge transfer. Across extensive evaluations, WeMM-Embedding achieves leading performance on multiple public benchmarks. Notably, the 2B variant already surpasses the previously leading 8B open-source baseline on MMEB-v2, while the 9B variant further achieves a new state-of-the-art overall score of 80.6. WeMM-Embedding also demonstrates strong practical performance across WeChat applications, with substantial gains on a 26-task in-house benchmark and consistent improvements across 14 online A/B tests. It has been deployed at scale across recommendation and search applications, including WeChat Channels, Oficial Accounts, Moments, and e-commerce services. We have released the model weights and code to facilitate future research.

https://huggingface.co/collections/tencent/wemm-embedding https://github.com/Tencent/WeMM-Embedding

Date: August 26, 2026

![](images/e279c030f55ffc135f1e69c3f048612c576f8180e483fbc757dd42938d351e64.jpg)  
 % #&"  % #&"   % #&"   , &	% #&" , &	% #&"  %#&#% #&" %.'&'+ '-" \*\$)#%'\$	

![](images/1aa3db21bf945bac3f61e2b35219de13dab46f8a5030af6872840df938f4aa2e.jpg)  
Figure 1 Performance and efficiency overview of WeMM-Embedding. Left: MMEB-v2 overall performance across diferent model sizes, compared with representative baselines. Right: Aggregate performance on the MMEB-v2 image and video subsets, a 12-dataset cross-modal retrieval suite, and the 26-task in-house benchmark.

## 1 Introduction

Multimodal embedding models have become fundamental components of modern AI systems, owing to their ability to map heterogeneous inputs, such as text, images, and videos, into a shared dense representation space. Such representations support a broad range of downstream applications, including classification [12, 20], any-to-any retrieval [2, 25, 48], clustering [43], recommendation [3, 13], and agentic systems [9, 14, 46]. Early CLIP-style models [33, 37, 49] established the efectiveness of large-scale cross-modal alignment through modality-specific dual encoders. However, their modality-specific encoding pathways do not naturall support the joint representation of inputs that combine multiple modalities, such as interleaved text–image documents, composed multimodal queries, or videos paired with transcripts. Subsequent work incorporated pretrained visual encoders as visual tokenizers for text encoders, extending this paradigm to broader any-to-any matching [51, 54]. Nevertheless, these encoder-based approaches remain limited in generality as multimodal embedding expands toward increasingly diverse tasks and input compositions.

The emergence of Multimodal Large Language Models (MLLMs) [1, 8, 24] has driven rapid progress in universal multimodal embedding models. MLLMs naturally support arbitrary interleaved combinations of text, images, and videos, while their broad pretrained capabilities provide a strong foundation for representation learning across diverse tasks. Motivated by these advantages, early studies explored adapting the hidden states of MLLMs into general-purpose embeddings and demonstrated the feasibility of continued contrastive training on paired multimodal data [19, 23]. More recent advances in large-scale paired multimodal data synthesis and knowledge distillation have further improved model performance, making MLLM-based embedding an increasingly prominent paradigm for universal multimodal representation learning [4, 22, 50, 52].

In this work, we introduce WeMM-Embedding, a family of universal multimodal embedding models spanning 2B, 4B, and 9B scales. To build a strong and general-purpose multimodal embedding family, we devise a two-stage training strategy that progressively moves from broad multimodal alignment to finer-grained relevance learning. In the first stage, the models are trained on several hundred million heterogeneous pairs spanning diverse modalities, tasks, and domains, establishing broad multimodal coverage and a strong initial representation space. In the second stage, we further refine the models on a carefully curated corpus with improved semantic balance, higher data quality, and more challenging negatives, while introducing richer relevance supervision and cross-scale knowledge transfer to strengthen fine-grained matching. This progressive training strategy allows WeMM-Embedding to retain broad capability coverage while continuously improving relevance modeling and representation quality.

WeMM-Embedding achieves state-of-the-art performance across a broad range of public benchmarks and shows strong practical performance in real-world applications. As summarized in Figure 1, the compact 2B variant already surpasses previously leading 8B open-source baselines on MMEB-v2 [28], highlighting the strong parameter eficiency of WeMM-Embedding. Scaling to 9B further raises the overall score to 80.6, ranking first on the oficial MMEB-v2 leaderboard and outperforming all listed open-source and proprietary models.<sup>1</sup> On the cross-modal retrieval suite reported in Gemini Embedding 2 [35], the 2B model also compares favorably with leading proprietary models. Beyond public benchmark evaluations, WeMM-Embedding also demonstrates strong practical performance, with substantial gains on a 26-task in-house benchmark and consistent improvements across 14 online A/B tests across WeChat applications. It has also been deployed at scale across recommendation and search applications, including WeChat Channels, Oficial Accounts, Moments, and e-commerce services. Together, these results establish a new performance–eficiency frontier for universal multimodal embedding models and demonstrate strong practical value in large-scale deployment.

## 2 Data Construction

To support general-purpose multimodal representation learning, we collect and synthesize several hundred million training examples spanning diverse modalities, domains, and tasks. We express these heterogeneous examples in a unified pair-based format, allowing them to be incorporated into a common training framework while preserving their original supervision signals. We further construct a curated collection that complements the large-scale data with greater semantic diversity, higher data quality, and more informative supervision.

![](images/2dcc6661f27b363db3819560518c94a168ceecf463e01f32942b6649d0776ccc.jpg)  
Figure 2 Overview of our multimodal training data. Major data families and representative coverage across diverse task settings and content domains.

## 2.1 Large-Scale Training Data Collection

Our training data are collected from public datasets and web-scale weakly supervised sources, and are further supplemented with task-oriented synthetic data and in-house collections. The resulting data span text, images, videos, and their interleaved combinations across diverse domains and task settings.

Unified Pair-Based Format. Despite substantial variation in input structures and supervision signals, these heterogeneous tasks can all be formulated as matching a source instance against one or more target candidates. Accordingly, we represent each training example in a unified pair-based format:

$$
z _ { i } = ( I _ { i } , q _ { i } , c _ { i } , \mathcal { N } _ { i } , y _ { i } ) ,\tag{1}
$$

where $I _ { i }$ denotes an optional task-specific instruction, $q _ { i }$ denotes the source instance, $c _ { i }$ denotes the paired target, ${ \mathcal { N } } _ { i }$ denotes an optional set of explicit hard negatives, and $y _ { i }$ denotes an optional graded relevance score. Both $q _ { i }$ and $c _ { i }$ may contain text, images, videos, or interleaved combinations of these modalities. Each element of ${ \mathcal { N } } _ { i }$ is drawn from the same target candidate space as $c _ { i }$ and has the corresponding target-side modality structure. When provided, $I _ { i }$ specifies the intended matching relation and helps distinguish tasks with similar input modalities. For standard source–target pairs, $c _ { i }$ serves as the positive target, while both ${ \mathcal { N } } _ { i }$ and $y _ { i }$ may be omitted. When explicit hard negatives are available, ${ \mathcal { N } } _ { i }$ contains candidates that are related to $q _ { i }$ but less relevant than $c _ { i } .$ . For datasets with graded relevance annotations, $y _ { i }$ retains the original ordinal or continuous relevance score assigned to the pair $( q _ { i } , c _ { i } )$ and is subsequently used to construct relative ordering constraints during training.

Under this formulation, examples from diferent tasks share a common source–target structure and can be organized into task-specific batches within a unified multi-task training pipeline. For standard pairwise data, targets from other examples in the same batch serve as in-batch negatives. For tasks with shared target spaces, such as classification, repeated targets may introduce false negatives; these collisions are excluded through semantic target masking, as detailed in Section 3.2.

Data Coverage and Composition. As shown in Figure 2, our training data cover diverse forms of supervision, task settings, and content domains. The training corpus includes several major types of data and supervision, with representative examples illustrated in the figure. We describe the main components below:

• Weakly Supervised Pairs. We collect large-scale image–text and video–text pairs from public datasets and web-scale weakly supervised sources, where visual and textual content are associated through naturally occurring correspondence rather than explicit visual descriptions. These pairs provide broad but relatively coarse supervision across diverse visual content.

• Caption Pairs. We pair images and videos with captions that explicitly describe their visual content, ranging from concise summaries to detailed descriptions of entities, attributes, relations, spatial context, actions, and events. Compared with weakly supervised pairs, these examples provide more direct and fine-grained visual–language correspondence.

• Retrieval Pairs. Retrieval examples associate queries with relevant candidates across text, images, videos, and interleaved multimodal inputs. They range from conventional unimodal and cross-modal matching to more challenging settings involving composed queries, reasoning, instructions, long contexts, spatial grounding, temporal localization, and agent-related retrieval.

• Classification Pairs. We reformulate classification datasets as source–label pairs, where the target is represented by either a class name or a natural-language description of the corresponding category. These examples cover a variety of image- and video-based recognition tasks, including object, scene, and action classification.

• Multimodal Question-Answer Pairs. We pair textual or multimodal questions with answer targets across a range of capabilities, including visual perception, relational and spatial understanding, optical character recognition, knowledge-intensive understanding, reasoning, document and chart comprehension, and event understanding.

• Graded Relevance Pairs. In the large-scale collection, graded supervision takes the form of manually assigned discrete relevance levels for source–target pairs. These labels distinguish multiple degrees of relevance beyond binary matching and support ranking-oriented training [17]. Reranker-derived relevance scores are introduced separately during the subsequent data curation stage.

## 2.2 Curated Data Construction

Alongside the large-scale collection, we construct a curated dataset approximately one tenth its size. The curated dataset focuses on improving semantic balance and data quality while introducing more informative supervision. Its construction includes Semantic-ID-guided resampling, quality control, and selective hardnegative enrichment.

Semantic-ID-Guided Resampling. Although the large-scale collection covers a broad range of domains and tasks, its semantic distribution remains skewed toward frequent patterns. Inspired by Semantic IDs [34], we derive a discrete identifier for each source–target pair to characterize this distribution and guide resampling. For each pair, the side with the longer serialized token sequence is encoded using an intermediate checkpoint of WeMM-Embedding. We then fit a three-level residual k-means quantizer (RQ-KMeans) [26] to the resulting representations. Each representation is subsequently mapped through the learned codebooks to obtain a three-element Semantic ID. We use the assignment density at each codebook level to guide resampling. Examples associated with densely populated codes are sampled at lower rates, whereas those mapped to less populated codes are retained at higher rates. This reduces repeated exposure to frequent semantic patterns without enforcing a uniform distribution over Semantic IDs.

Data Quality Refinement. The resampled training examples are further refined using a multimodal large language model. Based on the corresponding task context, the model assesses whether each source–target pair reflects the intended matching relation and filters out mismatched examples. We also refine the textual fields where necessary. For example, weakly supervised image–text and video–text pairs ofer broad coverage but may contain noisy or factually inaccurate descriptions; in such cases, the model corrects factual inconsistencies while preserving the original alt-text style and level of detail.

Hard-Negative Construction. A subset of the refined training examples is further enriched with explicit hard negatives to improve discrimination among semantically similar candidates. The construction procedure varies with the target modality. For text targets, multimodal large language models generate plausible but incorrect candidates based on the source and positive target. For image and video targets, intermediate checkpoints of our WeMM-Embedding models are used to retrieve semantically similar candidates from taskspecific candidate pools. For a smaller subset of the mined candidate sets, reranking models are further used to assign relevance scores, providing finer-grained supervision among dificult candidates. The corresponding training objective is introduced in Section 3.2.

## 3 Modeling and Training Strategy

WeMM-Embedding comprises three universal multimodal embedding models with 2B, 4B, and 9B parameters, built on the corresponding natively multimodal Qwen3.5 backbones [39]. This section first introduces how heterogeneous multimodal inputs are encoded into dense representations, and then presents the two-stage strategy used to train the model family.

## 3.1 Modeling

WeMM-Embedding is built upon the Qwen3.5 architecture. Benefiting from its native support for heterogeneous and interleaved multimodal inputs, WeMM-Embedding can encode arbitrary combinations of text, images, and videos into unified dense representations. We adopt last-token pooling by appending a dedicated <embedding> token to the input sequence and using its final-layer hidden state as the output representation.

Formally, an input instance may contain an optional task-specific instruction together with an ordered sequence of multimodal segments:

$$
\mathcal { D } = \langle I _ { \mathrm { i n s t } } , x _ { 1 } , x _ { 2 } , \ldots , x _ { m } \rangle ,\tag{2}
$$

where $I _ { \mathrm { i n s t } }$ denotes the optional instruction and each $x _ { i }$ may correspond to text, an image, or a video. Textual segments are converted into token embeddings through the native tokenizer and embedding layer, while visual segments are converted into visual token representations through the native visual processing pipeline. The textual and visual tokens are arranged according to the original order of the input segments, followed by a dedicated <embedding> token:

$$
{ \bf S } = \left[ { \bf z } _ { 1 } , { \bf z } _ { 2 } , \ldots , { \bf z } _ { N } , { \bf z } _ { \mathrm { e m b } } \right] ,\tag{3}
$$

where each $\mathbf { z } _ { i }$ denotes a textual or visual token embedding, and $\mathbf { z } _ { \mathrm { e m b } }$ denotes the token embedding of <embedding>. The resulting multimodal sequence is then processed by the LLM backbone $G _ { \theta }$

$$
\mathbf { H } = G _ { \theta } ( \mathbf { S } ) = \left[ \mathbf { h } _ { 1 } , \mathbf { h } _ { 2 } , \dots , \mathbf { h } _ { N } , \mathbf { h } _ { \mathrm { e m b } } \right] .\tag{4}
$$

By default, the <embedding> token is placed at the end of the sequence and attends to all preceding textual and visual content under the native causal attention mask. The same causal formulation also allows multiple <embedding> tokens to be inserted at diferent sequence positions. For example, when a video is followed by its automatic speech recognition (ASR) transcript, placing one token after the video tokens and another at the end of the sequence enables video-only and joint video–text representations to be extracted within a single forward pass, supporting downstream applications with diferent modality requirements.

The final-layer hidden state corresponding to each <embedding> token is L2-normalized to obtain the output representation:

$$
\mathbf { e } _ { \mathcal { D } } = \frac { \mathbf { h } _ { \mathrm { e m b } } } { \| \mathbf { h } _ { \mathrm { e m b } } \| _ { 2 } } .\tag{5}
$$

In addition, WeMM-Embedding supports flexible embedding dimensions through Matryoshka Representation Learning (MRL) [21]. Given the final hidden state $\mathbf { h } _ { \mathrm { e m b } } \in \mathbb { R } ^ { D }$ , an embedding of dimension $d \leq D$ is obtained by retaining its first d dimensions and applying L2 normalization:

$$
\mathbf { e } _ { \mathcal { D } } ^ { ( d ) } = \frac { \mathbf { h } _ { \mathrm { e m b } , 1 : d } } { \left\| \mathbf { h } _ { \mathrm { e m b } , 1 : d } \right\| _ { 2 } } , \qquad d \in { \mathcal { D } } _ { \mathrm { M R L } } ,\tag{6}
$$

where $\mathcal { D } _ { \mathrm { M R I } }$ denotes the predefined set of supported embedding dimensions. At inference time, embeddings at all supported dimensions can be obtained from a single forward pass through prefix truncation and re-normalization.

## 3.2 Training Strategy

We train WeMM-Embedding using a two-stage strategy. The first stage establishes a general multimodal embedding space through large-scale multi-task alignment. The second stage continues training on curated data, combining contrastive learning with explicit hard negatives, selective reranker supervision, and embedding distillation from a larger model. Both stages operate on the unified pair-based representation introduced in Section 2.1, while each batch uses the objective supported by its supervision signals.

## 3.2.1 Stage 1: Large-Scale Multimodal Alignment

In the first stage, we train WeMM-Embedding on the large-scale collection described in Section 2.1, comprising several hundred million source–target pairs across diverse modalities, domains, and tasks. Each batch is constructed from a consistent data source, while batches from diferent tasks are interleaved throughout training. Standard paired examples are optimized with contrastive learning [30], whereas examples carrying native graded relevance annotations use a score-gap-weighted CoSENT-style ranking objective [16].

Contrastive Learning. For standard paired data, we optimize source–target alignment using an InfoNCE objective [30]. Each batch is drawn from a single dataset, keeping the task definition and candidate space consistent within the batch. For each source, its paired target is treated as the positive, while targets associated with other examples serve as in-batch negatives. When explicit hard negatives are available, they are incorporated into the same negative pool. This construction provides informative task-consistent negatives, but may also introduce collisions when diferent examples contain nearly identical sources or targets. We therefore apply duplicate-aware masking on both sides of the pair before computing the loss.

Let $\mathcal { C } _ { j } = \{ c _ { j } ^ { + } \} \cup \mathcal { N } _ { j }$ denote the candidates associated with source $q _ { j }$ , where ${ \mathcal { N } } _ { j }$ is empty for ordinary paired examples. Given a batch of B source–target pairs, the contrastive objective is defined as

$$
\mathcal { L } _ { \mathrm { C L } } = - \frac { 1 } { B } \sum _ { i = 1 } ^ { B } \log \frac { \exp \left( s ( q _ { i } , c _ { i } ^ { + } ) / \tau \right) } { \exp \left( s ( q _ { i } , c _ { i } ^ { + } ) / \tau \right) + \displaystyle \sum _ { j = 1 } ^ { B } \sum _ { c \in \mathcal { C } _ { j } \backslash \{ c _ { i } ^ { + } \} } M _ { i , j , c } \exp \left( s ( q _ { i } , c ) / \tau \right) } ,\tag{7}
$$

where $s ( \cdot , \cdot )$ denotes cosine similarity between normalized embeddings and $\tau$ is a learnable temperature parameter. The duplicate-aware mask is defined as

$$
M _ { i , j , c } = \left\{ \begin{array} { l l } { 0 , } & { \mathrm { i f ~ } j \neq i \mathrm { ~ a n d ~ } s ( q _ { i } , q _ { j } ) > \tau _ { \mathrm { d u p } } , } \\ { \mathrm { ~ o r ~ i f ~ } s ( c _ { i } ^ { + } , c ) > \tau _ { \mathrm { d u p } } , } \\ { 1 , } & { \mathrm { o t h e r w i s e , } } \end{array} \right.\tag{8}
$$

where $\tau _ { \mathrm { d u p } }$ is the similarity threshold used to identify near-duplicate representations. Source-side masking excludes all candidates associated with a near-duplicate source, while target-side masking excludes candidates that closely match the current positive target. The latter applies to both in-batch targets and explicit hard negatives.

Graded Relevance Learning. Within the large-scale collection, a subset of source–target pairs is annotated with manually assigned discrete relevance levels. Such annotations are particularly useful for item-toitem retrieval and recommendation-oriented scenarios, where diferent pairs may exhibit varying degrees of relatedness. Treating these examples as ordinary positive pairs under the contrastive objective would ignore this graded structure. We therefore adopt a score-gap-weighted CoSENT-style objective [16], which optimizes the relative ordering of pairwise similarities and assigns greater importance to comparisons with larger relevance gaps.

Formally, given a graded-relevance batch

$$
\boldsymbol { B } _ { \mathrm { r e l } } = \left\{ \left( q _ { i } , c _ { i } , y _ { i } \right) \right\} _ { i = 1 } ^ { G } ,\tag{9}
$$

let $s _ { i } = s ( q _ { i } , c _ { i } )$ denote the predicted similarity of the i-th source–target pair. For two examples i and j, we define the relevance-gap weight as

$$
w _ { i j } = \operatorname* { m a x } \left( \left| y _ { i } - y _ { j } \right| , \epsilon \right) ,\tag{10}
$$

where ϵ is a small positive constant. The ranking loss associated with example i is

$$
\begin{array} { r l r } {  { \mathcal { L } _ { i } ^ { \mathrm { R e l } } = \log \Bigg [ 1 + \sum _ { j : y _ { i } > y _ { j } } w _ { i j } \exp ( \gamma [ s _ { j } - s _ { i } ] ) } } \\ & { } & { + \sum _ { j : y _ { j } > y _ { i } } w _ { i j } \exp ( \gamma [ s _ { i } - s _ { j } ] ) \Bigg ] , } \end{array}\tag{11}
$$

and the batch objective is

$$
\mathcal { L } _ { \mathrm { R e l } } = \frac { 1 } { G } \sum _ { i = 1 } ^ { G } \mathcal { L } _ { i } ^ { \mathrm { R e l } } .\tag{12}
$$

This objective encourages pairs with higher relevance levels to receive higher similarities and gives greater weight to comparisons with larger label gaps. Pairs with equal labels impose no ordering constraint.

Matryoshka Representation Learning. We apply Matryoshka Representation Learning [21] to both contrastive and graded-relevance training. For each batch, the objective associated with its supervision is evaluated independently at every supported embedding dimension:

$$
\mathcal { L } _ { X } ^ { \mathrm { M R L } } = \sum _ { d \in \mathcal { D } _ { \mathrm { M R L } } } \alpha _ { d } \mathcal { L } _ { X } ^ { ( d ) } , \qquad X \in \{ \mathrm { C L } , \mathrm { R e l } \} ,\tag{13}
$$

where $\mathcal { L } _ { X } ^ { ( d ) }$ is computed using embeddings independently truncated and normalized at dimension $d ,$ and $\alpha _ { d }$ denotes the corresponding loss weight. Thus, each interleaved batch is optimized using the multi-dimensional form of its corresponding task objective.

## 3.2.2 Stage 2: Curated Fine-Tuning and Distillation

Following large-scale multimodal alignment, WeMM-Embedding is further trained on the curated dataset described in Section 2.2. The contrastive and graded-relevance objectives remain in use, while additional supervision is provided by reranking models and a larger embedding teacher. Reranking models provide query-specific ordering signals over mined candidates for selected tasks, whereas the embedding teacher transfers the similarity structure induced within each training batch.

Reranker Supervision. We train dedicated multimodal rerankers to provide fine-grained ordering supervision over query-specific candidate sets, each containing an annotated positive target and mined hard negatives. Building on the score-gap-weighted CoSENT formulation used for graded relevance learning in Section 3.2.1, we replace the manually assigned relevance levels with reranker scores and construct comparisons only among candidates associated with the same query. Specifically, for source $q _ { b }$ with candidate set $\mathcal { C } _ { b } = \{ c _ { b , 1 } , \ldots , c _ { b , K } \}$ the reranker scores $\hat { y } _ { b , k }$ induce the ordered pair set

$$
\mathcal { \widehat { P } } _ { b } = \{ ( i , j ) \mid \hat { y } _ { b , i } > \hat { y } _ { b , j } \} .\tag{14}
$$

The corresponding objective is

$$
\mathcal { L } _ { \mathrm { R a n k } } = \frac { 1 } { B } \sum _ { b = 1 } ^ { B } \log \left[ 1 + \sum _ { ( i , j ) \in \widehat { \mathcal { P } } _ { b } } \omega _ { b , i j } \exp \left( \gamma \left[ s ( q _ { b } , c _ { b , j } ) - s ( q _ { b } , c _ { b , i } ) \right] \right) \right] ,\tag{15}
$$

where $\omega _ { b , i j } = \mathrm { m a x } ( \left| \hat { y } _ { b , i } - \hat { y } _ { b , j } \right| , \epsilon )$ weights each comparison according to the reranker score gap. Rerankerscored batches use $\mathcal { L } _ { \mathrm { R a n k } }$ in place of the standard contrastive objective.

Empirically, we find that reranking candidates retrieved by our embedding models does not consistently improve performance across multimodal tasks. Similar to observations reported in prior work [22], stable gains are observed only on a limited subset of tasks. We therefore restrict reranker supervision to settings where it yields reliable improvements.

Embedding Distillation. To obtain teacher supervision with broader coverage, we additionally distill from a larger model in the same WeMM-Embedding family. Unlike reranker supervision, which relies on task-specific models and preconstructed candidate sets, embedding distillation derives online soft targets from the batch-wise source–target similarity distributions produced by the teacher. It therefore requires no additional ofline relevance annotations and can be applied across heterogeneous batch types.

Formally, for a batch containing $B$ sources and a target pool of size $K ,$ , let $\mathbf { A } ^ { T } , \mathbf { A } ^ { S } \in \mathbb { R } ^ { B \times K }$ denote the source-to-target similarity matrices produced by the teacher and student:

$$
A _ { i j } ^ { T } = \frac { s _ { T } ( q _ { i } , c _ { j } ) } { \tau _ { T } } , \qquad A _ { i j } ^ { S } = \frac { s _ { S } ( q _ { i } , c _ { j } ) } { \tau _ { S } } ,\tag{16}
$$

where $\tau _ { T }$ and $\tau _ { S }$ are the teacher and student temperatures. We further construct the reverse similarity matrices between the positive targets and the sources:

$$
\overline { { A } } _ { i j } ^ { T } = \frac { s _ { T } ( c _ { i } ^ { + } , q _ { j } ) } { \tau _ { T } } , \qquad \overline { { A } } _ { i j } ^ { S } = \frac { s _ { S } ( c _ { i } ^ { + } , q _ { j } ) } { \tau _ { S } } .\tag{17}
$$

The corresponding row-wise relation distributions are

$$
\mathbf { P } _ { T , i } ^ { q  c } = \mathrm { S o f t m a x } ( \mathbf { A } _ { i , : } ^ { T } ) , \qquad \mathbf { P } _ { S , i } ^ { q  c } = \mathrm { S o f t m a x } ( \mathbf { A } _ { i , : } ^ { S } ) ,\tag{18}
$$

$$
\mathbf { P } _ { T , i } ^ { c  q } = \mathrm { S o f t m a x } ( \overline { { \mathbf { A } } } _ { i , : } ^ { T } ) , \qquad \mathbf { P } _ { S , i } ^ { c  q } = \mathrm { S o f t m a x } ( \overline { { \mathbf { A } } } _ { i , : } ^ { S } ) .\tag{19}
$$

We define the bidirectional embedding-distillation objective as

$$
\mathcal { L } _ { \mathrm { E m b } } = \frac { 1 } { 2 B } \sum _ { i = 1 } ^ { B } [ D _ { \mathrm { K L } } ( \mathbf { P } _ { T , i } ^ { q  c } \Vert \mathbf { P } _ { S , i } ^ { q  c } ) + D _ { \mathrm { K L } } ( \mathbf { P } _ { T , i } ^ { c  q } \Vert \mathbf { P } _ { S , i } ^ { c  q } ) ] .\tag{20}
$$

The two terms align the teacher and student similarity distributions in the source-to-target and target-to-source directions, respectively. Unlike the one-hot supervision used in standard contrastive learning, the teacher distributions retain relative similarity diferences among candidates, providing softer and more structured targets for transferring semantic relations. Empirically, this supervision is particularly valuable for our compact WeMM-Embedding variants and contributes substantially to their performance gains.

Training Configuration. For each Stage 2 batch, the task objective is selected according to its available supervision:

$$
\mathcal { L } _ { \mathrm { T a s k } } = \left\{ \begin{array} { l l } { \mathcal { L } _ { \mathrm { C L } } ^ { \mathrm { M R L } } , } & { \mathrm { f o r ~ s t a n d a r d ~ p a i r e d ~ o r ~ h a r d - n e g a t i v e ~ b a t c h e s } , } \\ { \mathcal { L } _ { \mathrm { R e l } } ^ { \mathrm { M R L } } , } & { \mathrm { f o r ~ g r a d e d - r e l e v a n c e ~ b a t c h e s } , } \\ { \mathcal { L } _ { \mathrm { R a n k } } ^ { \mathrm { M R L } } , } & { \mathrm { f o r ~ r e r a n k e r - s c o r e d ~ b a t c h e s } . } \end{array} \right.\tag{21}
$$

Reranker-scored batches use the ranking objective in place of the standard contrastive objective. Embedding distillation, by contrast, is added to the selected task objective whenever a larger embedding teacher is available. The overall Stage 2 objective is therefore

$$
{ \mathcal { L } } _ { \mathrm { S t a g e 2 } } = { \mathcal { L } } _ { \mathrm { T a s k } } + \lambda _ { \mathrm { E m b } } { \mathcal { L } } _ { \mathrm { E m b } } ,\tag{22}
$$

where $\lambda _ { \mathrm { E m b } }$ denotes the weight assigned to the embedding-distillation loss.

For the 2B and 4B variants, the frozen 9B WeMM-Embedding model serves as the embedding teacher during Stage 2. Since no larger embedding teacher is available for the 9B variant, we instead train multiple specialized Stage 2 variants using complementary data mixtures and training configurations, and combine them through model merging [47] to obtain the final 9B model.

## 4 Experiments and Analysis

We evaluate WeMM-Embedding across a broad range of public and in-house benchmarks covering image, video, visual-document, text, and agent-oriented retrieval. Our experiments cover the MMEB benchmark series [15, 28], a collection of widely used cross-modal retrieval benchmarks reported in the Gemini Embedding 2 study [35], and an in-house benchmark comprising 26 tasks derived from real-world applications within WeChat. We compare WeMM-Embedding with representative state-of-the-art multimodal embedding models, including both open-source and proprietary commercial models.

## 4.1 Performance on the MMEB Series

The MMEB series provides a comprehensive evaluation across diverse modalities, task formulations, and retrieval scenarios. Specifically, MMEB-v2 [28] evaluates general multimodal embedding capabilities across images, videos, and visual documents, while MMEB-v3 [15] further extends the evaluation to audio, complex text retrieval, and agent-centric scenarios. In this section, we follow the benchmark-defined evaluation metrics, using Hit@1 for image, video, audio, and agent tasks and NDCG@5 for text and visual-document tasks.

## 4.1.1 General Multimodal Representation

We first report the performance of WeMM-Embedding on MMEB-v2, which comprises 78 datasets spanning images, videos, and visual documents and covers diverse tasks including classification, question answering, retrieval, and visual grounding [28]. As shown in Table 1, WeMM-Embedding-2B achieves an overall score of 77.9, outperforming Qwen3-VL-Embedding-2B [22] and DME-2B [6] by 4.7 and 3.1 points, respectively, and slightly surpassing Qwen3-VL-Embedding-8B. The 4B variant further improves the score to 79.2, outperforming all compared 8B–9B baselines. Scaling to 9B further raises the overall score to 80.6, ranking first on the oficial MMEB-v2 leaderboard and outperforming all listed open-source and proprietary models.

<table><tr><td rowspan="2">Model</td><td rowspan="2">Size AVG</td><td rowspan="2"></td><td colspan="5">Image</td><td colspan="5">Video</td><td rowspan="2">VisDoc</td></tr><tr><td>Overall</td><td>CLS</td><td>QA</td><td>Ret</td><td>GD</td><td>Overall</td><td>CLS</td><td>QA</td><td>V-Ret</td><td>M-Ret</td></tr><tr><td># datasets →</td><td></td><td>78</td><td>36</td><td>10</td><td>10</td><td>12</td><td>4</td><td>18</td><td>5</td><td>5</td><td>5</td><td>3</td><td>24</td></tr><tr><td>VLM2Vec [19]</td><td>2B</td><td>47.8</td><td>59.7</td><td>58.7</td><td>49.3</td><td>65.0</td><td>72.9</td><td>29.0</td><td>33.4</td><td>30.5</td><td>20.6</td><td>33.0</td><td>44.0</td></tr><tr><td>GME [50]</td><td>2B</td><td>55.4</td><td>51.9</td><td>54.4</td><td>29.9</td><td>66.9</td><td>55.5</td><td>33.9</td><td>34.9</td><td>42.0</td><td>25.6</td><td>32.4</td><td>76.8</td></tr><tr><td>VLM2Vec-V2 [28]</td><td>2B</td><td>59.3</td><td>64.9</td><td>62.9</td><td>56.3</td><td>69.5</td><td>77.3</td><td>34.9</td><td>39.3</td><td>34.3</td><td>28.8</td><td>38.5</td><td>69.2</td></tr><tr><td>Ops-MM-embedding-v1</td><td>2B</td><td>64.6</td><td>69.0</td><td>68.1</td><td>65.1</td><td>69.2</td><td>80.9</td><td>47.6</td><td>53.6</td><td>55.7</td><td>41.8</td><td>33.7</td><td>70.8</td></tr><tr><td>RzenEmbed [18]</td><td>2B</td><td>67.2</td><td>72.3</td><td>68.5</td><td>66.3</td><td>74.5</td><td>90.3</td><td>47.3</td><td>50.4</td><td>49.7</td><td>46.6</td><td>38.9</td><td>74.5</td></tr><tr><td>Qwen3-VL-Embedding [22]</td><td>2B</td><td>73.2</td><td>75.0</td><td>70.3</td><td>74.3</td><td>74.9</td><td>88.6</td><td>61.9</td><td>71.9</td><td>64.9</td><td>53.9</td><td>53.3</td><td>79.2</td></tr><tr><td>DME-Small [6]†</td><td>2B</td><td>74.8</td><td>75.9</td><td>68.6</td><td>76.2</td><td>75.6</td><td>94.3</td><td>65.6</td><td>84.5</td><td>61.9</td><td>55.5</td><td>57.4</td><td>79.9</td></tr><tr><td>WeMM-Embedding</td><td>2B</td><td>77.9</td><td>79.6</td><td>74.7</td><td>79.2</td><td>79.3</td><td>94.0</td><td>70.8</td><td>84.9</td><td>71.7</td><td>63.4</td><td>58.4</td><td>80.7</td></tr><tr><td>WeMM-Embedding</td><td>4B</td><td>79.2</td><td>80.8</td><td>75.1</td><td>81.0</td><td>80.8</td><td>95.0</td><td>72.1</td><td>86.8</td><td>72.5</td><td>65.2</td><td>58.6</td><td>82.0</td></tr><tr><td>VLM2Vec [19]</td><td>8B</td><td>53.2</td><td>65.5</td><td>62.7</td><td>56.9</td><td>69.4</td><td>82.2</td><td>34.0</td><td>39.1</td><td>30.0</td><td>29.0</td><td>40.6</td><td>49.1</td></tr><tr><td>GME [50]</td><td>8B</td><td>59.2</td><td>56.0</td><td>57.7</td><td>34.7</td><td>71.2</td><td>59.3</td><td>38.6</td><td>37.4</td><td>50.4</td><td>28.4</td><td>38.2</td><td>79.3</td></tr><tr><td>Ops-MM-embedding-v1</td><td>8B</td><td>68.9</td><td>72.7</td><td>69.7</td><td>69.6</td><td>73.1</td><td>87.2</td><td>53.8</td><td>59.7</td><td>62.2</td><td>45.7</td><td>43.2</td><td>74.4</td></tr><tr><td>RzenEmbed [18]</td><td>8B</td><td>72.9</td><td>75.9</td><td>70.6</td><td>71.7</td><td>78.5</td><td>92.1</td><td>55.7</td><td>58.8</td><td>63.5</td><td>51.0</td><td>45.5</td><td>81.3</td></tr><tr><td>IFM-TTE [1i]</td><td>8B</td><td>74.1</td><td>77.9</td><td>76.7</td><td>78.5</td><td>74.6</td><td>89.3</td><td>59.2</td><td>60.5</td><td>67.9</td><td>51.7</td><td>54.9</td><td>79.5</td></tr><tr><td>Qwen3-VL-Embedding [22]</td><td>8B</td><td>77.8</td><td>80.1</td><td>74.2</td><td>81.1</td><td>80.2</td><td>92.3</td><td>67.1</td><td>78.4</td><td>71.0</td><td>58.7</td><td>56.1</td><td>82.4</td></tr><tr><td>DME-Medium [6]†</td><td>9B</td><td>78.4</td><td>79.8</td><td>74.5</td><td>80.9</td><td>78.2</td><td>94.6</td><td>70.8</td><td>87.7</td><td>71.0</td><td>61.0</td><td>58.5</td><td>82.0</td></tr><tr><td>WeMM-Embedding</td><td>9B</td><td>80.6</td><td>81.9</td><td>76.2</td><td>82.2</td><td>81.7</td><td>95.6</td><td>74.3</td><td>87.4</td><td>77.7</td><td>67.4</td><td>58.5</td><td>83.3</td></tr><tr><td>Seed-1.6-Embedding*</td><td></td><td>76.9</td><td>78.0</td><td>75.1</td><td>74.9</td><td>79.3</td><td>89.1</td><td>67.7</td><td>85.2</td><td>66.7</td><td>59.1</td><td>54.8</td><td>82.2</td></tr><tr><td>QQMM-embed-v4†</td><td></td><td>78.3</td><td>82.0</td><td>76.8</td><td>83.3</td><td>80.7</td><td>96.1</td><td>67.4</td><td>77.8</td><td>71.9</td><td>60.0</td><td>54.9</td><td>80.9</td></tr><tr><td>Octen-VL-Large*</td><td></td><td>80.1</td><td>81.9</td><td>75.7</td><td>84.5</td><td>80.6</td><td>94.4</td><td>76.0</td><td>87.1</td><td>82.1</td><td>68.0</td><td>60.4</td><td>80.5</td></tr><tr><td>DME-Large [6]†</td><td></td><td>80.2</td><td>81.1</td><td>75.4</td><td>82.9</td><td>79.6</td><td>95.2</td><td>74.4</td><td>90.1</td><td>75.7</td><td>64.5</td><td>62.2</td><td>83.4</td></tr></table>

Table 1 Benchmarking results on MMEB-v2. Baseline results are taken from the oficial MMEB leaderboard. <sup>†</sup>Closedsource leaderboard submission without publicly released model weights or a public inference endpoint. <sup>⋆</sup>Proprietary commercial model with an undisclosed parameter count. CLS: classification; QA: question answering; Ret: retrieval; GD: visual grounding; V-Ret: video retrieval; M-Ret: moment retrieval.

## 4.1.2 Text and Agent-Centric Tasks

We further evaluate WeMM-Embedding on MMEB-v3 [15], which extends the benchmark with complex text retrieval, agent-centric tasks, audio evaluation, and the MCMR image-retrieval task. Among the newly added tasks, 53 text tasks cover reasoning retrieval, instruction following, long-context retrieval, multi-condition retrieval, and general retrieval, while 47 agent tasks cover tool, GUI, and memory retrieval. When computing V3-All, unsupported tasks are assigned a score of zero; accordingly, the 11 audio tasks are scored as zero for the current WeMM-Embedding models, which do not support audio input. Baseline per-task results on the newly added tasks are obtained from the MMEB-v3 paper or score files submitted to the oficial MMEB leaderboard.<sup>2</sup> For VLM2Vec [19], VLM2Vec-V2 [28], GME [50], and Qwen3-VL-Embedding [22], we recompute V3-All by combining their originally reported MMEB-v2 results with the corresponding results on the newly added MMEB-v3 tasks.

As shown in Table 2, WeMM-Embedding-2B already outperforms all compared baselines on MMEB-v3, achieving 56.0 on V3-All. It also attains the highest scores among all baselines on the Text and Agent task groups, with 45.3 and 45.1, respectively. The 4B and 9B variants achieve V3-All scores of 58.2 and 59.5, respectively, further widening the gap over existing models.

## 4.2 Cross-Modal Retrieval Evaluation

The cross-modal evaluation comprises 12 widely used public benchmarks that measure semantic alignment between text and visual content across images, videos, and visual documents. For the three proprietary models, Gemini Embedding 2 [35], Amazon Nova MME [32], and Voyage Multimodal 3.5 [40], we use the results reported in the Gemini Embedding 2 technical report [35]. Qwen3-VL-Embedding and WeMM-Embedding are evaluated by us on the corresponding public evaluation sets using the metric specified for each benchmark.

<table><tr><td rowspan="2">Model</td><td rowspan="2"></td><td rowspan="2">Size V3-All</td><td colspan="6">Text</td><td colspan="4">Agent</td><td rowspan="2"></td><td rowspan="2">MCMR Audio</td></tr><tr><td>Overall</td><td>RR</td><td>IF</td><td>LC</td><td>MC</td><td>GR</td><td>Overall</td><td>Tool</td><td>GUI</td><td>Memory</td></tr><tr><td># tasks →</td><td></td><td>190</td><td>53</td><td>20</td><td>9</td><td>6</td><td>5</td><td>13</td><td>47</td><td>35</td><td>8</td><td>4</td><td>1</td><td>11</td></tr><tr><td>VLM2Vec-V2 [28]</td><td>2B</td><td>38.3</td><td>24.5</td><td>7.8</td><td>29.2</td><td>11.5</td><td>50.3</td><td>41.2</td><td>28.7</td><td>27.6</td><td>36.2</td><td>23.3</td><td>4.1</td><td>0.0</td></tr><tr><td>Omni-Embed-Nemotron [45]</td><td>3B</td><td>43.5</td><td>39.2</td><td>17.2</td><td>42.8</td><td>40.0</td><td>69.7</td><td>56.9</td><td>36.5</td><td>38.1</td><td>32.0</td><td>32.2</td><td>26.1</td><td>36.5</td></tr><tr><td>E5-Omni [5]</td><td>3B</td><td>44.6</td><td>26.7</td><td>11.5</td><td>44.4</td><td>8.3</td><td>60.7</td><td>34.4</td><td>36.9</td><td>37.7</td><td>35.6</td><td>32.3</td><td>31.9</td><td>30.8</td></tr><tr><td>Qwen3-VL-Embedding [22]</td><td>2B</td><td>50.9</td><td>39.2</td><td>16.6</td><td>40.6</td><td>53.1</td><td>61.2</td><td>58.0</td><td>39.3</td><td>42.6</td><td>30.4</td><td>28.4</td><td>42.0</td><td>0.0</td></tr><tr><td>WeMM-Embedding</td><td>2B</td><td>56.0</td><td>45.3</td><td>26.3</td><td>49.3</td><td>61.9</td><td>59.7</td><td>58.7</td><td>45.1</td><td>46.2</td><td>38.9</td><td>47.2</td><td>42.5</td><td>0.0</td></tr><tr><td>WeMM-Embedding</td><td>4B</td><td>58.2</td><td>47.9</td><td>30.0</td><td>52.0</td><td>62.4</td><td>60.1</td><td>61.1</td><td>49.0</td><td>50.6</td><td>41.5</td><td>50.4</td><td>41.9</td><td>0.0</td></tr><tr><td>WAVE [38]</td><td>7B</td><td>26.3</td><td>13.7</td><td>5.9</td><td>31.3</td><td>2.6</td><td>22.9</td><td>18.6</td><td>11.3</td><td>11.9</td><td>11.8</td><td>5.7</td><td>8.9</td><td>31.8</td></tr><tr><td>VLM2Vec [19]</td><td>8B</td><td>32.9</td><td>22.2</td><td>7.2</td><td>28.1</td><td>5.9</td><td>40.9</td><td>41.7</td><td>19.7</td><td>19.8</td><td>21.4</td><td>15.9</td><td>0.9</td><td>0.0</td></tr><tr><td>LCO-Embedding-Omni [42]</td><td>7B</td><td>40.6</td><td>32.4</td><td>9.5</td><td>52.0</td><td>12.6</td><td>61.4</td><td>52.1</td><td>27.8</td><td>29.0</td><td>25.0</td><td>23.0</td><td>20.0</td><td>43.2</td></tr><tr><td>GME [50]</td><td>8B</td><td>43.6</td><td>37.1</td><td>12.5</td><td>52.4</td><td>17.8</td><td>59.0</td><td>62.5</td><td>35.6</td><td>39.0</td><td>30.0</td><td>17.1</td><td>27.3</td><td>0.0</td></tr><tr><td>E5-Omni [5]</td><td>7B</td><td>47.1</td><td>26.9</td><td>11.1</td><td>42.1</td><td>9.7</td><td>56.9</td><td>35.9</td><td>36.7</td><td>37.4</td><td>38.0</td><td>27.3</td><td>41.1</td><td>43.0</td></tr><tr><td>Tianmu-Emb-Uni</td><td>8B</td><td>53.3</td><td>43.6</td><td>20.7</td><td>47.1</td><td>58.5</td><td>63.4</td><td>62.0</td><td>39.4</td><td>42.2</td><td>35.6</td><td>22.6</td><td>38.8</td><td>38.9</td></tr><tr><td>Qwen3-VL-Embedding [22]</td><td>8B 9B</td><td>53.5</td><td>42.5</td><td>18.2</td><td>44.8</td><td>58.0</td><td>61.2</td><td>61.2</td><td>38.4</td><td>41.3</td><td>33.5</td><td>22.8</td><td>38.0</td><td>0.0</td></tr><tr><td>WeMM-Embedding</td><td></td><td>59.5</td><td>48.8</td><td>31.8</td><td>52.8</td><td>64.5</td><td>58.3</td><td>61.1</td><td>51.0</td><td>53.0</td><td>43.3</td><td>48.9</td><td>49.3</td><td>0.0</td></tr></table>

Table 2 Evaluation results on MMEB-v3. V3-All averages all 190 tasks, comprising the 78 MMEB-v2 tasks, 53 Text tasks, 47 Agent tasks, 11 Audio tasks, and MCMR [10]. Following the evaluation protocol defined in the MMEB-v3 paper [15], Text results are reported using NDCG@5. Unsupported tasks are assigned a score of zero. RR: reasoning retrieval; IF: instruction following; LC: long-context retrieval; MC: multi-condition retrieval; GR: general retrieval.

As shown in Table 3, WeMM-Embedding-2B achieves an overall score of 79.8 across the 12 tasks, outperforming all compared open-source baselines and comparing favorably with leading proprietary models. Scaling the model to 4B and 9B further improves the overall score to 80.8 and 81.7, respectively.
<table><tr><td></td><td></td><td>Gemini</td><td>Amazon Embedding 2† Nova MME† Multimodal 3.5†</td><td>Voyage</td><td>Qwen3-VL- Embedding</td><td>WeMM- Embedding</td><td>WeMM- Embedding</td><td>WeMM- Embedding</td></tr><tr><td>Size</td><td></td><td>一</td><td>一</td><td>=</td><td>8B</td><td>2B</td><td>4B</td><td>9B</td></tr><tr><td rowspan="5">Text → Image Recall@1</td><td>MSCOCO [7]</td><td>62.9</td><td>57.2</td><td>58.1</td><td>58.7</td><td>62.6</td><td>63.5</td><td>64.6</td></tr><tr><td>Flickr30k [31]</td><td>89.1</td><td>81.6</td><td>89.9</td><td>84.6</td><td>86.9</td><td>87.0</td><td>87.8</td></tr><tr><td>DOCCI [29]</td><td>93.4</td><td>84.0</td><td>83.8</td><td>95.0</td><td>94.0</td><td>95.8</td><td>96.7</td></tr><tr><td>TextCaps [36]</td><td>89.6</td><td>76.0</td><td>79.4</td><td>79.3</td><td>82.5</td><td>85.8</td><td>85.2</td></tr><tr><td>Mean</td><td>83.8</td><td>74.7</td><td>77.8</td><td>79.4</td><td>81.5</td><td>83.0</td><td>83.6</td></tr><tr><td rowspan="5">Image → Text Recall@1</td><td>MSCOCO [7]</td><td>78.8</td><td>68.3</td><td>74.5</td><td>71.1</td><td>78.8</td><td>79.4</td><td>79.3</td></tr><tr><td>Flickr30k [31]</td><td>97.4</td><td>87.5</td><td>94.5</td><td>90.7</td><td>95.6</td><td>96.6</td><td>94.4</td></tr><tr><td>DOCCI [29]</td><td>91.3</td><td>76.5</td><td>77.4</td><td>92.8</td><td>90.6</td><td>93.7</td><td>92.8</td></tr><tr><td>TextCaps [36]</td><td>97.4</td><td>88.9</td><td>88.6</td><td>91.9</td><td>93.3</td><td>96.0</td><td>94.8</td></tr><tr><td>Mean</td><td>91.2</td><td>80.3</td><td>83.8</td><td>86.6</td><td>89.6</td><td>91.4</td><td>90.3</td></tr><tr><td rowspan="4">Text → Video NDCG@10</td><td>VATEX [41]</td><td>68.8</td><td>60.3</td><td>55.2</td><td>68.5</td><td>74.6</td><td>75.9</td><td>78.4</td></tr><tr><td>MSR-VTT [44]</td><td>68.0</td><td>67.0</td><td>63.0</td><td>70.6</td><td>72.7</td><td>71.7</td><td>74.1</td></tr><tr><td>YouCook2 [53]</td><td>52.5</td><td>34.7</td><td>31.4</td><td>54.7</td><td>63.9</td><td>62.2</td><td>66.3</td></tr><tr><td>Mean</td><td>63.1</td><td>54.0</td><td>49.9</td><td>64.6</td><td>70.4</td><td>69.9</td><td>72.9</td></tr><tr><td>NDCG@10</td><td>ViDoRe V2 [27]</td><td>64.9</td><td>60.6</td><td>65.5</td><td>62.8</td><td>61.4</td><td>62.1</td><td>66.3</td></tr><tr><td>AVG</td><td></td><td>79.5</td><td>70.2</td><td>71.8</td><td>76.7</td><td>79.8</td><td>80.8</td><td>81.7</td></tr></table>

Table 3 Cross-modal retrieval results on 12 public benchmarks. AVG denotes the average across the 12 datasets. <sup>†</sup>Proprietary commercial model with an undisclosed parameter count.

## 4.3 In-House Evaluation

We further evaluate WeMM-Embedding on an in-house benchmark comprising 26 tasks derived from real-world applications within WeChat. The benchmark covers five categories: classification, search, cross-domain content matching, article relevance, and video relevance. As shown in Table 4, WeMM-Embedding-2B substantially outperforms the representative open-source baseline and achieves higher scores across all five categories.

Beyond ofline evaluation, WeMM-Embedding has been deployed at scale in recommendation systems spanning

<table><tr><td>Model</td><td>Size</td><td>AVG</td><td>Classification</td><td>Search</td><td>Cross-DM</td><td>Article Rel.</td><td>Video Rel.</td></tr><tr><td># tasks →</td><td></td><td>26</td><td>4</td><td>7</td><td>4</td><td>4</td><td>7</td></tr><tr><td>Qwen3-VL-Embedding [22]</td><td>2B</td><td>60.9</td><td>60.1</td><td>51.9</td><td>55.5</td><td>75.5</td><td>64.9</td></tr><tr><td>WeMM-Embedding</td><td>2B</td><td>72.0</td><td>72.6</td><td>65.5</td><td>73.7</td><td>86.4</td><td>68.8</td></tr></table>

Table 4 Evaluation results on the in-house benchmark. AVG denotes the average over all 26 tasks. Cross-DM denotes cross-domain content matching; Article Rel. denotes article relevance; Video Rel. denotes video relevance.

WeChat Oficial Accounts, WeChat Channels, and e-commerce content. Its multimodal representations combine complementary signals from text, cover images, and video frames, and are used at multiple stages of the recommendation pipeline, including candidate retrieval, ranking feature construction, user sequence modeling, and cross-domain content understanding. Semantic IDs derived from these representations further provide compact discrete features for indexing and sequence modeling. To date, WeMM-Embedding has delivered consistent gains in 14 online $\mathrm { A } / \mathrm { B }$ tests across these systems, with the corresponding improvements subsequently rolled out in production. These deployments have improved content matching, recommendation quality, and user engagement, with notable gains for long-tail and newly published content.

WeMM-Embedding has also been deployed in WeChat search, supporting semantic retrieval across diverse content sources including WeChat Channels videos, WeChat Oficial Accounts articles, and WeChat Moments. The model supports both unimodal and cross-modal retrieval and improves semantic relevance and retrieval quality across text, image, and video content. Taken together, the in-house evaluation and production deployments show that the strong performance of WeMM-Embedding extends beyond public benchmarks to diverse real-world tasks and large-scale recommendation and search systems.

## 4.4 Further Analysis

In this section, we first analyze the performance of the learned Matryoshka representations across diferent embedding dimensions, then revisit key Stage-1 design choices, and finally present a cumulative analysis of the strategies adopted for Stage-2 training.

## 4.4.1 MRL Analysis

WeMM-Embedding incorporates Matryoshka Representation Learning (MRL) [21], allowing a single model to produce nested representations at multiple embedding dimensions. We evaluate the 2B model at dimensions ranging from 64 to 2,048 on the corresponding task subsets of MMEB-v2 to examine how dimensionality reduction afects diferent modalities and task types. For each evaluation group, we report the proportion of performance retained relative to its result at 2,048 dimensions.

As shown in the left panel of Figure 3, performance on image and video tasks follows closely matched trends as the embedding dimension decreases. At 256 dimensions, the model retains 98.7% of its 2,048-dimensional performance on both image and video tasks; the corresponding retention rates rise to 99.2% and 98.8% at 512 dimensions, respectively. Visual-document tasks exhibit greater sensitivity to dimensionality reduction, which may reflect the higher information density of visual documents, particularly their dense textual content.

The right panel of Figure 3 further shows diferent sensitivity patterns across task types. Classification is the least sensitive to dimensionality reduction, question answering exhibits moderate degradation, and retrieval is afected most substantially, especially at 64 and 128 dimensions. Once the embedding dimension reaches 256, however, all three task groups retain more than 97% of their respective full-dimensional performance, and the gains from further increasing the dimension become progressively smaller. Taken together, these results suggest that 256- or 512-dimensional representations provide practical choices for eficiency-sensitive applications while preserving most of the performance achieved at 2,048 dimensions.

![](images/abd9b3f55cf2c073e278f3a60464dc395515429969899a437751792a19be48de.jpg)

![](images/4bc3500af12a6f3d4247abbb8b3d3d01cb4add5ee26fc15b8b8e0db1322a58b9.jpg)  
Figure 3 MRL analysis of WeMM-Embedding-2B on MMEB-v2. Left: Performance retained on the image, video, and visual-document subsets across embedding dimensions. Right: Performance retained for classification (CLS), question answering (QA), and retrieval (RET), each averaged over the corresponding image and video tasks.

## 4.4.2 Stage-1 Design Analysis

We conduct a small-scale ablation study with a 2B model to examine some of the key Stage-1 design choices, including task-specific instructions, task-consistent batching, and duplicate-aware masking. The resulting variants are evaluated on MMEB-v2 under a common experimental protocol.

<table><tr><td>Stage-1 Configuration</td><td>AVG</td><td>Image</td><td>Video</td><td>VisDoc</td></tr><tr><td>Full configuration</td><td>71.9</td><td>76.1</td><td>59.1</td><td>75.3</td></tr><tr><td>w/o task-specific instructions</td><td>71.1</td><td>75.7</td><td>58.2</td><td>73.8</td></tr><tr><td>w/o task-consistent batching</td><td>68.5</td><td>71.5</td><td>56.3</td><td>73.1</td></tr><tr><td>w/o duplicate-aware masking</td><td>71.4</td><td>75.3</td><td>58.6</td><td>75.1</td></tr></table>

Table 5 Results on MMEB-v2 from a small-scale Stage-1 ablation study using a 2B model. AVG denotes the average across all 78 MMEB-v2 tasks.

As shown in Table 5, removing task-specific instructions leads to a 0.8-point decrease in the overall score, with the largest degradation observed on visual-document tasks. This result indicates that explicit instructions help the model capture task-specific matching objectives across heterogeneous tasks. Task-consistent batching has the largest impact among the evaluated designs. Replacing it with mixed sampling results in a 3.4-point decrease in overall performance, suggesting that batches constructed from a consistent task and candidate space provide more informative in-batch negatives. Duplicate-aware masking also contributes to performance, as its removal leads to a 0.5-point decrease in the overall score. By filtering duplicate or semantically equivalent targets from the negative set, it makes the unified contrastive objective better suited to tasks with repeated labels, such as classification.

## 4.4.3 Cumulative Stage-2 Analysis

We further conduct a cumulative study with the 2B model to examine several key strategies used in Stage-2 training, including curated data, reranker supervision, embedding-teacher distillation, and an expanded visual input budget. Starting from the Stage-1 checkpoint, these strategies are introduced sequentially, and each resulting configuration is evaluated on MMEB-v2.

As shown in Table 6, cumulatively incorporating our Stage-2 strategies improves the overall score by 2.2 points. First, curated data yields clear gains on both image and video tasks, highlighting the value of balanced resampling, quality refinement, and hard-negative enrichment when constructing a compact Stage-2 training set from the large-scale multimodal corpus. Building on this, reranker supervision provides an additional improvement, with its largest benefit observed on visual-document tasks. Embedding-teacher distillation further improves performance across all three domains, demonstrating that dense similarity supervision transfers efectively across heterogeneous tasks and contributes to the strong performance of the compact model. Finally, expanding the visual input budget through higher-resolution inputs and denser video frame sampling brings a further improvement, particularly on video tasks. Overall, these substantial gains suggest that carefully designed supervision signals remain essential for further refining universal multimodal embedding models once large-scale multimodal alignment has been established.

<table><tr><td>Cumulative Configuration</td><td>AVG</td><td>Image</td><td>Video</td><td>VisDoc</td></tr><tr><td>Stage-1 checkpoint</td><td>75.7</td><td>77.7</td><td>67.5</td><td>78.9</td></tr><tr><td>+ curated data</td><td>76.6</td><td>78.9</td><td>69.2</td><td>78.5</td></tr><tr><td>+ reranker supervision</td><td>76.7</td><td>78.8</td><td>69.4</td><td>79.1</td></tr><tr><td>+ embedding-teacher distillation</td><td>77.6</td><td>79.4</td><td>70.0</td><td>80.4</td></tr><tr><td>+ expanded visual input budget (final)</td><td>77.9</td><td>79.6</td><td>70.8</td><td>80.7</td></tr></table>

Table 6 Cumulative Stage-2 results of the 2B model on MMEB-v2. Each row reports the configuration after introducing the listed strategy. AVG denotes the average across all 78 MMEB-v2 tasks.

## 5 Conclusion and Future Work

In this report, we introduced WeMM-Embedding, a family of universal multimodal embedding models spanning multiple scales. The family is trained with a two-stage strategy that progresses from broad multimodal alignment to finer-grained representation learning through curated data, richer relevance supervision, and cross-scale knowledge transfer. Extensive evaluations across public benchmarks demonstrate that WeMM-Embedding achieves state-of-the-art performance and establishes a new performance–eficiency frontier. Moreover, it delivers consistent gains on the 26-task in-house benchmark and across 14 online A/B tests. WeMM-Embedding has also been deployed at scale across multiple recommendation and search systems within WeChat, further demonstrating its practical value. In the future, we will continue to extend WeMM-Embedding toward omni-modal inputs, scale the model family to larger sizes, and further improve data curation and fine-grained relevance supervision.

## References

[1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.

[2] Yingshan Chang, Mridu Narang, Hisami Suzuki, Guihong Cao, Jianfeng Gao, and Yonatan Bisk. Webqa: Multihop and multimodal qa. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 16495–16504, 2022.

[3] Ben Chen, Xian Guo, Siyuan Wang, Zihan Liang, Yue Lv, Yufei Ma, Xinlong Xiao, Bowen Xue, Xuxin Zhang, Ying Yang, et al. Onesearch: A preliminary exploration of the unified end-to-end generative framework for e-commerce search. arXiv preprint arXiv:2509.03236, 2025.

[4] Haonan Chen, Liang Wang, Nan Yang, Yutao Zhu, Ziliang Zhao, Furu Wei, and Zhicheng Dou. mme5: Improving multimodal multilingual embeddings via high-quality synthetic data. arXiv preprint arXiv:2502.08468, 2025.

[5] Haonan Chen, Sicheng Gao, Radu Timofte, Tetsuya Sakai, and Zhicheng Dou. e5-omni: Explicit cross-modal alignment for omni-modal embeddings. In Findings of the Association for Computational Linguistics: ACL 2026, pages 19430–19443, 2026.

[6] Haonan Chen, Chu Li, Zhicheng Wang, Yuanwei Liu, Yuanjiang Wang, Shaohua Jiang, and Zhicheng Dou. Douyin multimodal embedding model technical report, 2026. URL https://arxiv.org/abs/2608.02148.

[7] Xinlei Chen, Hao Fang, Tsung-Yi Lin, Ramakrishna Vedantam, Saurabh Gupta, Piotr Dollár, and C Lawrence Zitnick. Microsoft coco captions: Data collection and evaluation server. arXiv preprint arXiv:1504.00325, 2015.

[8] Zhe Chen, Weiyun Wang, Hao Tian, Shenglong Ye, Zhangwei Gao, Erfei Cui, Wenwen Tong, Kongzhi Hu, Jiapeng Luo, Zheng Ma, et al. How far are we to gpt-4v? closing the gap to commercial multimodal models with open-source suites. arXiv preprint arXiv:2404.16821, 2024.

[9] Prateek Chhikara, Dev Khant, Saket Aryan, Taranjeet Singh, and Deshraj Yadav. Mem0: Building productionready ai agents with scalable long-term memory. arXiv preprint arXiv:2504.19413, 2025.

[10] Wei Chow, Yuan Gao, Linfeng Li, Xian Wang, Qi Xu, Hang Song, Lingdong Kong, Ran Zhou, Yi Zeng, Yidong Cai, et al. Merit: Multilingual semantic retrieval with interleaved multi-condition query. Advances in Neural Information Processing Systems, 38:74806–74867, 2025.

[11] Xuanming Cui, Jianpeng Cheng, Hong-you Chen, Satya Narayan Shukla, Abhijeet Awasthi, Xichen Pan, Chaitanya Ahuja, Shlok Mishra, Taipeng Tian, Qi Guo, et al. Think then embed: Generative context improves multimodal embedding. In International Conference on Learning Representations, volume 2026, pages 2690–2709, 2026.

[12] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pages 248–255. Ieee, 2009.

[13] Jiaxin Deng, Shiyao Wang, Kuo Cai, Lejian Ren, Qigen Hu, Weifeng Ding, Qiang Luo, and Guorui Zhou. Onerec: Unifying retrieve and rank with generative recommender and iterative preference alignment. CoRR, abs/2502.18965, 2025. doi: 10.48550/ARXIV.2502.18965. URL https://doi.org/10.48550/arXiv.2502.18965.

[14] Wenbo Hu, Jia-Chen Gu, Zi-Yi Dou, Mohsen Fayyaz, Pan Lu, Kai-Wei Chang, and Nanyun Peng. Mrag-bench: Vision-centric evaluation for retrieval-augmented multimodal models. arXiv preprint arXiv:2410.08182, 2024.

[15] Haohang Huang, Xuan Lu, Mingyi Su, Xuan Zhang, Ziyan Jiang, Ping Nie, Kai Zou, Tomas Pfister, Wenhu Chen, Wei Zhang, et al. Mmeb-v3: Measuring the performance gaps of omni-modality embedding models. arXiv preprint arXiv:2604.23321, 2026.

[16] Xiang Huang, Hao Peng, Dongcheng Zou, Zhiwei Liu, Jianxin Li, Kay Liu, Jia Wu, Jianlin Su, and Philip S. Yu. Cosent: Consistent sentence embedding via similarity ranking. IEEE ACM Trans. Audio Speech Lang. Process., 32:2800–2813, 2024. doi: 10.1109/TASLP.2024.3402087. URL https://doi.org/10.1109/TASLP.2024.3402087.

[17] Kalervo Järvelin and Jaana Kekäläinen. Cumulated gain-based evaluation of IR techniques. ACM Trans. Inf. Syst., 20(4):422–446, 2002. doi: 10.1145/582415.582418. URL http://doi.acm.org/10.1145/582415.582418.

[18] Weijian Jian, Yajun Zhang, Dawei Liang, Chunyu Xie, Yixiao He, Dawei Leng, and Yuhui Yin. Rzenembed: Towards comprehensive multimodal retrieval. arXiv preprint arXiv:2510.27350, 2025.

[19] Ziyan Jiang, Rui Meng, Xinyi Yang, Semih Yavuz, Yingbo Zhou, and Wenhu Chen. Vlm2vec: Training visionlanguage models for massive multimodal embedding tasks. arXiv preprint arXiv:2410.05160, 2024.

[20] Will Kay, Joao Carreira, Karen Simonyan, Brian Zhang, Chloe Hillier, Sudheendra Vijayanarasimhan, Fabio Viola, Tim Green, Trevor Back, Paul Natsev, et al. The kinetics human action video dataset. arXiv preprint arXiv:1705.06950, 2017.

[21] Aditya Kusupati, Gantavya Bhatt, Aniket Rege, Matthew Wallingford, Aditya Sinha, Vivek Ramanujan, William Howard-Snyder, Kaifeng Chen, Sham Kakade, Prateek Jain, et al. Matryoshka representation learning. Advances in Neural Information Processing Systems, 35:30233–30249, 2022.

[22] Mingxin Li, Yanzhao Zhang, Dingkun Long, Keqin Chen, Sibo Song, Shuai Bai, Zhibo Yang, Pengjun Xie, An Yang, Dayiheng Liu, Jingren Zhou, and Junyang Lin. Qwen3-vl-embedding and qwen3-vl-reranker: A unified framework for state-of-the-art multimodal retrieval and ranking. CoRR, abs/2601.04720, 2026. doi: 10.48550/ARXIV.2601.04720. URL https://doi.org/10.48550/arXiv.2601.04720.

[23] Sheng-Chieh Lin, Chankyu Lee, Mohammad Shoeybi, Jimmy Lin, Bryan Catanzaro, and Wei Ping. Mm-embed: Universal multimodal retrieval with multimodal llms. arXiv preprint arXiv:2411.02571, 2024.

[24] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In Alice Oh, Tristan Naumann, Amir Globerson, Kate Saenko, Moritz Hardt, and Sergey Levine, editors, Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023, 2023. URL http://papers.nips.cc/paper\_files/paper/2023/ hash/6dcf277ea32ce3288914faf369fe6de0-Abstract-Conference.html.

[25] Zheyuan Liu, Cristian Rodriguez-Opazo, Damien Teney, and Stephen Gould. Image retrieval on real-life images with pre-trained vision-and-language models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 2125–2134, 2021.

[26] Xinchen Luo, Jiangxia Cao, Tianyu Sun, Jinkai Yu, Rui Huang, Wei Yuan, Hezheng Lin, Yichen Zheng, Shiyao Wang, Qigen Hu, Changqing Qiu, Jiaqi Zhang, Xu Zhang, Zhiheng Yan, Jingming Zhang, Simin Zhang, Mingxing Wen, Zhaojie Liu, and Guorui Zhou. QARM: quantitative alignment multi-modal recommendation at kuaishou. In Meeyoung Cha, Chanyoung Park, Noseong Park, Carl Yang, Senjuti Basu Roy, Jessie Li, Jaap Kamps, Kijung Shin, Bryan Hooi, and Lifang He, editors, Proceedings of the 34th ACM International Conference on Information and Knowledge Management, CIKM 2025, Seoul, Republic of Korea, November 10-14, 2025, pages 5915–5922. ACM, 2025. doi: 10.1145/3746252.3761502. URL https://doi.org/10.1145/3746252.3761502.

[27] Quentin Macé, António Loison, and Manuel Faysse. Vidore benchmark v2: Raising the bar for visual retrieval. arXiv preprint arXiv:2505.17166, 2025.

[28] Rui Meng, Ziyan Jiang, Ye Liu, Mingyi Su, Xinyi Yang, Yuepeng Fu, Can Qin, Zeyuan Chen, Ran Xu, Caiming Xiong, et al. Vlm2vec-v2: Advancing multimodal embedding for videos, images, and visual documents. arXiv preprint arXiv:2507.04590, 2025.

[29] Yasumasa Onoe, Sunayana Rane, Zachary Berger, Yonatan Bitton, Jaemin Cho, Roopal Garg, Alexander Ku, Zarana Parekh, Jordi Pont-Tuset, Garrett Tanzer, et al. Docci: Descriptions of connected and contrasting images. In European Conference on Computer Vision, pages 291–309. Springer, 2024.

[30] Aaron van den Oord, Yazhe Li, and Oriol Vinyals. Representation learning with contrastive predictive coding. arXiv preprint arXiv:1807.03748, 2018.

[31] Bryan A Plummer, Liwei Wang, Chris M Cervantes, Juan C Caicedo, Julia Hockenmaier, and Svetlana Lazebnik. Flickr30k entities: Collecting region-to-phrase correspondences for richer image-to-sentence models. In Proceedings of the IEEE international conference on computer vision, pages 2641–2649, 2015.

[32] Danilo Poccia. Amazon Nova Multimodal Embeddings: State-of-the-art embedding model for agentic RAG and semantic search, October 2025. URL https://aws.amazon.com/blogs/aws/ amazon-nova-multimodal-embeddings-now-available-in-amazon-bedrock/.

[33] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021.

[34] Shashank Rajput, Nikhil Mehta, Anima Singh, Raghunandan Hulikal Keshavan, Trung Vu, Lukasz Heldt, Lichan Hong, Yi Tay, Vinh Tran, Jonah Samost, et al. Recommender systems with generative retrieval. Advances in Neural Information Processing Systems, 36:10299–10315, 2023.

[35] Madhuri Shanbhogue, Zhe Li, Shanfeng Zhang, Gustavo Hernández Ábrego, Shih-Cheng Huang, Aashi Jain, Daniel Salz, Sonam Goenka, Chaitra Hegde, Ji Ma, et al. Gemini embedding 2: A native multimodal embedding model from gemini. arXiv preprint arXiv:2605.27295, 2026.

[36] Oleksii Sidorov, Ronghang Hu, Marcus Rohrbach, and Amanpreet Singh. Textcaps: a dataset for image captioning with reading comprehension. In European conference on computer vision, pages 742–758. Springer, 2020.

[37] Quan Sun, Yuxin Fang, Ledell Wu, Xinlong Wang, and Yue Cao. Eva-clip: Improved training techniques for clip at scale. arXiv preprint arXiv:2303.15389, 2023.

[38] Changli Tang, Qinfan Xiao, Ke Mei, Tianyi Wang, Fengyun Rao, and Chao Zhang. Wave: learning unified & versatile audio-visual embeddings with multimodal llm. In International Conference on Learning Representations, volume 2026, pages 61596–61612, 2026.

[39] Qwen Team. Qwen3.5: Accelerating productivity with native multimodal agents, February 2026. URL https: //qwen.ai/blog?id=qwen3.5.

[40] Voyage AI. voyage-multimodal-3.5: A new multimodal retrieval frontier with video support, January 2026. URL https://blog.voyageai.com/2026/01/15/voyage-multimodal-3-5/.

[41] Xin Wang, Jiawei Wu, Junkun Chen, Lei Li, Yuan-Fang Wang, and William Yang Wang. Vatex: A large-scale, high-quality multilingual dataset for video-and-language research. In 2019 IEEE/CVF International Conference on Computer Vision (ICCV), pages 4580–4590. IEEE, 2019.

[42] Chenghao Xiao, Hou Pong Ken Chan, Hao Zhang, Weiwen Xu, Mahani Aljunied, and Yu Rong. Scaling language-centric omnimodal representation learning. Advances in Neural Information Processing Systems, 38: 158370–158401, 2025.

[43] Chenghao Xiao, Isaac Chung, Imene Kerboua, Jamie Stirling, Xin Zhang, Márton Kardos, Roman Solomatin, Noura Al Moubayed, Kenneth Enevoldsen, and Niklas Muennighof. Mieb: Massive image embedding benchmark. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 22187–22198, 2025.

[44] Jun Xu, Tao Mei, Ting Yao, and Yong Rui. Msr-vtt: A large video description dataset for bridging video and language. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 5288–5296, 2016.

[45] Mengyao Xu, Wenfei Zhou, Yauhen Babakhin, Gabriel Moreira, Ronay Ak, Radek Osmulski, Bo Liu, Even Oldridge, and Benedikt Schiferer. Omni-embed-nemotron: A unified multimodal retrieval model for text, image, audio, and video. arXiv preprint arXiv:2510.03458, 2025.

[46] Wujiang Xu, Zujie Liang, Kai Mei, Hang Gao, Juntao Tan, and Yongfeng Zhang. A-mem: Agentic memory for llm agents. Advances in Neural Information Processing Systems, 38:17577–17604, 2025.

[47] Prateek Yadav, Derek Tam, Leshem Choshen, Colin A Rafel, and Mohit Bansal. Ties-merging: Resolving interference when merging models. Advances in Neural Information Processing Systems, 36:7093–7115, 2023.

[48] Huaying Yuan, Jian Ni, Zheng Liu, Yueze Wang, Junjie Zhou, Zhengyang Liang, Bo Zhao, Zhao Cao, Ji-Rong Wen, and Zhicheng Dou. Momentseeker: A task-oriented benchmark for long-video moment retrieval. Advances in Neural Information Processing Systems, 38, 2025.

[49] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 11975–11986, 2023.

[50] Xin Zhang, Yanzhao Zhang, Wen Xie, Mingxin Li, Ziqi Dai, Dingkun Long, Pengjun Xie, Meishan Zhang, Wenjie Li, and Min Zhang. Bridging modalities: Improving universal multimodal retrieval by multimodal large language models. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 9274–9285. IEEE, 2025.

[51] Junjie Zhou, Zheng Liu, Shitao Xiao, Bo Zhao, and Yongping Xiong. Vista: Visualized text embedding for universal multi-modal retrieval. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 3185–3200, 2024.

[52] Junjie Zhou, Yongping Xiong, Zheng Liu, Ze Liu, Shitao Xiao, Yueze Wang, Bo Zhao, Chen Jason Zhang, and Defu Lian. Megapairs: Massive data synthesis for universal multimodal retrieval. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 19076–19095, 2025.

[53] Luowei Zhou, Chenliang Xu, and Jason Corso. Towards automatic learning of procedures from web instructional videos. In Proceedings of the AAAI conference on artificial intelligence, volume 32, 2018.

[54] Tianshuo Zhou, Sen Mei, Xinze Li, Zhenghao Liu, Chenyan Xiong, Zhiyuan Liu, Yu Gu, and Ge Yu. Marvel: unlocking the multi-modal capability of dense retrieval via visual module plugin. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 14608–14624, 2024.