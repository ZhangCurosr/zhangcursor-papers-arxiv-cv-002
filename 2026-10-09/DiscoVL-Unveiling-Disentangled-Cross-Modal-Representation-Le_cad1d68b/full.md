# DiscoVL: Unveiling Disentangled Cross-Modal Representation Learning via Orthogonal Adversarial Regularization for Vision-Language Models

Mengping Dong<sup>1\*</sup> , Jinbao Li<sup>1\*</sup> , and Fei Li<sup>2†</sup>

<sup>1</sup> Shandong Artificial Intelligence Institute, Qilu University of Technology (Shandong Academy of Sciences), Jinan, China {dongmengping,lijb}@qlu.edu.cn 2 University of Florida, Florida, USA leefly072@126.com

Abstract. Pre-trained vision-language models excel across varied perception tasks, but adapting them to novel downstream settings without sacrificing generalization remains non-trivial. Existing parametereficient prompt learning method often yields inconsistent representations and fails to account for semantic distribution shifts. In this work, we present DiscoVL, a disentangled cross-modal representation learning framework that couples orthogonal adversarial regularization with structured cross-modal alignment for vision-language models. To address the insuficient cross-modal interaction, our DiscoVL designs a multi-branch low-rank residual aligner that decomposes representations into subspaces and enables bidirectional cross-modal feedback between visual and textual streams at each layer. Furthermore, while conventional triplet constraints overfit features to class centroids, we design an orthogonal regularization for adversarial triplet loss, which prevents centroid collapse and substantially boosts generalization. Evaluations on 15 benchmarks demonstrate that DiscoVL delivers consistent improvements over stateof-the-art methods for base-to-novel generalization, cross-dataset evaluation, and few-shot learning.

Keywords: Vision-language models · Representation learning · Zeroshot learning · Few-shot learning

## 1 Introduction

Vision-language models (VLMs) such as CLIP [26] learn joint representations from millions of image-text pairs and exhibit impressive transfer to tasks ranging from zero-shot classification [10] to open-vocabulary detection [5] and segmentation [20]. Prompt learning [38] has emerged as the dominant parameter-eficient strategy, which freezes the backbone and learns a small set of tunable tokens, avoiding the cost and forgetting risk of full fine-tuning.

A fundamental but largely overlooked problem is that existing VLMs adaptation methods learn entangled representations in which class-specific discriminative cues and class-agnostic transferable knowledge are tightly intertwined. Representative approaches, including $\mathrm { C o O p / C o C o O p }$ [40, 41] and PromptSRC [16], regularize prompt learning using textual prototypes or handcrafted priors. However, the resulting representations still entangle task-relevant patterns with broad semantic priors inherited from pre-training. Recent multi-modal prompt learners, such as AAPL [17], HiCroPL [39], and MMRL [8], aim to stabilize training by injecting attributes, routing knowledge flows, or stacking multiple representation aligners, yet they still optimize a single entangled feature space without explicitly separating these two types of information.

This entanglement leads to three consequences: 1) class-specific adaptation contaminates the class-agnostic subspace, causing the representations to drift away from the pre-trained feature and eroding zero-shot transfer [36]; 2) joint-space projections ignore modality-specific distribution gaps, producing inconsistent crossmodal representations that amplify this drift across layers; and 3) representation tokens rapidly collapse onto class centroids, destroying the classagnostic structure needed for novelclass generalization.

![](images/09bbfcb7df112cbe074d3f4c6ec1bd9cd9cb0ece1b19ebdab0f8b12f7077e7dc.jpg)

We address this root cause with DiscoVL, a Disentangled cross-modal representation learning framework for V ision-Language models adaptation. The key idea is to explicitly factorize

Fig. 1: Comparison of the harmonic mean between previous state-of-the-art method and our DiscoVL across 11 diverse datasets for base-to-novel generalization.

the adapted representation space into a shared, class-agnostic subspace that preserves CLIP’s transferable semantics and a task-specific subspace that captures downstream discriminative cues. This disentanglement is realized through two complementary modules. First, we propose a Disentangled Cross-modal Representation Aligner (DCRA). It employs a multi-branch residual architecture with subspace separation and a bidirectional cross-modal feedback loop. This design progressively fuses textual and visual cues across layers while explicitly decoupling shared and task-specific information, thereby alleviating cross-modal and cross-layer inconsistencies. Second, an Orthogonal Adversarial Representation Learning (OARL) module enforces class-agnostic constraints in the representation space through orthogonal regularization and adversarial triplet objectives, preventing representation tokens from collapsing onto class centroids and preserving the pre-trained semantic representation. Extensive experiments on 11 benchmarks, as shown in Fig. 1, demonstrate that DiscoVL consistently surpasses prior prompt-learning methods on base-to-novel generalization.

We summarize our main contributions as follows:

We identify representation entanglement as a core bottleneck of existing VLMs prompt-learning methods, and propose disentangled cross-modal representation learning as a principled solution.

A disentangled cross-modal representation aligner (DCRA) with multi-branch low-rank subspaces and bidirectional cross-modal feedback, which separates shared and task-specific representations while capturing complementary modality cues across layers.

Orthogonal adversarial representation learning (OARL) that preserves classagnostic knowledge through orthogonal triplet tokens and adversarial constraints, efectively suppressing token collapse.

Comprehensive evaluations on 15 datasets showing consistent, significant gains over state-of-the-art prompt-learning baselines, with dedicated disentanglement analyses confirming the intended subspace separation.

## 2 Related work

Vision-language models. Vision-language models (VLMs) have rapidly become the backbone of multi-modal perception, owing to their ability to contrastively align paired image-text samples in a shared latent space [30]. CLIP [26] demonstrated that large-scale supervision, 400M image-text pairs, enables strong zero-shot generalization, and ALIGN [14] further confirmed the scaling law with 1.8B weakly curated pairs. Subsequent pipelines such as BLIP [19] and LLaVA [21] inherit CLIP-like visual encoders while augmenting them with frozen LLM decoders to support open-ended reasoning. However, when these models are adapted to downstream domains, class-specific and class-agnostic information becomes entangled in a single feature space, causing semantic drift and degraded cross-domain transfer. DiscoVL addresses this by explicitly disentangling the two types of information through a cross-modal aligner and orthogonal adversarial regularization.

Fine-tuning VLMs. Prompt engineering, first explored for LLM adaptation, has emerged as the dominant parameter-eficient strategy for tailoring VLMs. Early CLIP deployments relied on carefully designed textual templates [26], whereas later work learns prompt embeddings in the continuous token space [10]. $\mathrm { K g C o O p }$ [36] anchors soft prompts with a handcrafted prior to combat forgetting, while MaPLe [15] injects branch-aware prompts into the dual vision-language towers to capture heterogeneous cues. PromptSRC [16] introduces self-regulating constraints to stabilize prompt optimization, and Co-Prompt [28] enforces predictive consistency between frozen and trainable heads. Beyond generic prompts, TCP [37] derives class-aware representations via textual knowledge embeddings, TextRefiner [34] augments prompts with cached visual descriptions, and SurPL [22] synthesizes target-aware text features through a lightweight surrogate generator. These methods all operate within an entangled feature space where class-specific adaptation inevitably corrupts class-agnostic semantics. DiscoVL difers fundamentally: it factorizes the representation space into disentangled shared and task-specific subspaces and enforces their separation with orthogonal adversarial regularization, addressing the root cause of cross-domain degradation rather than its symptoms.

![](images/92ff93d8af8f4cb103116b70cdc5568dea11d7d04891cce73a78b396e2eb2350.jpg)  
Fig. 2: Overview of DiscoVL, which disentangles representations into shared and taskspecific subspaces. DCRA separates and aligns cross-modal cues via multi-branch residual structure and bidirectional feedback, while OARL enforces orthogonal adversarial constraints to prevent representation tokens from collapsing onto class centroids.

## 3 Methodology

## 3.1 Preliminaries

Contrastive language-image pre-training. CLIP [26] comprises an image encoder I and a text encoder T, each implemented as L stacked transformer blocks $\{ I _ { \ell } \} _ { \ell = 1 } ^ { L }$ and $\{ T _ { \ell } \} _ { \ell = 1 } ^ { L }$ . Given an image $\boldsymbol { x } \in \mathbb { R } ^ { H \times W \times 3 }$ , we partition it into P patches and linearly project them to obtain $V _ { 0 } \in \mathbb { R } ^ { P \times d _ { i } }$ , where $d _ { i }$ denotes the embedding dimension. A learnable class token $k _ { 0 }$ is prepended, and positional encoding is added before propagating the token sequence through the vision transformer $[ k _ { \ell } , V _ { \ell } ] = I _ { \ell } ( [ k _ { \ell - 1 } , V _ { \ell - 1 } ] ) , \ell = 1 , \dots , L$ . The final class token $k _ { L }$ summarizes the entire image and is mapped into the shared vision-language space through the patch projection layer $Z _ { i m g } ,$ yielding the image embedding $v = Z _ { i m g } ( k _ { L } )$ . For the text branch, a prompt template is tokenized into $U$ tokens, which are embedded as $S _ { 0 } \in \mathbb { R } ^ { U \times d _ { t } }$ with positional encoding, and $d _ { t }$ is the embedding dimension. Beginning-of-text and end-of-text tokens $\left( h _ { 0 } , g _ { 0 } \right)$ are appended, and the sequence is processed as $\begin{array} { r } { [ h _ { \ell } , S _ { \ell } , g _ { \ell } ] = T _ { \ell } ( [ h _ { \ell - 1 } , S _ { \ell - 1 } , g _ { \ell - 1 } ] ) , \ell = } \end{array}$ $1 , \ldots , L$ . The terminal token $g _ { L }$ is projected with $\boldsymbol { Z } _ { t x t } ~ \mathrm { t o }$ produce $t \in \mathbb { R } ^ { d }$ . Given an image embedding $v \in \mathbb { R } ^ { d }$ and class-specific text embeddings $\{ t _ { k } \} _ { k = 1 } ^ { K }$ for K classes, CLIP predicts the class posterior via

```latex
Algorithm 1 Training DiscoVL (DCRA + OARL)
1: Input: Pre-trained CLIP image encoder I, text encoder T, dataset $\mathcal { D } ,$ representa
tion tokens $A ,$ injection layer $Q$
2: Initialize: DCRA parameters $\{ F _ { \mathrm { s h a r e d } } ^ { m } , J _ { \ell } ^ { e } , D _ { \ell } ^ { e } , g ( \cdot ) , \rho ( \ell ) \}$ , projectors $Z _ { \mathrm { i m g } } ^ { k } , \ Z _ { \mathrm { i m g } } ^ { a } ,$
$Z _ { T 2 I } , Z _ { I 2 T }$
3: for each training iteration do
4: Sample mini-batch $\{ ( x _ { i } , y _ { i } ) \}$ from D
5: Encode text prompts with T to obtain text feature t and text tokens
6: Encode image $x _ { i }$ with I and inject representation tokens A from layer $Q _ { \checkmark }$
7: Apply DCRA to obtain image class token $k _ { L } ,$ , representation tokens $\{ A _ { \ell } ^ { i ^ { \prime } } \}$
8: Compute representation feature $\begin{array} { r } { a _ { L } = \frac { 1 } { M } \sum _ { j = 1 } ^ { M } A _ { L , j } ^ { i ^ { \prime } } } \end{array}$
9: Project features $f _ { k } = Z _ { \mathrm { i m g } } ^ { k } ( k _ { L } ) , f _ { a } = Z _ { \mathrm { i m g } } ^ { a } ( a _ { L } )$
10: Compute cross-modal logits via cosine similarity between $( f _ { k } , t )$ and $( f _ { a } , t )$
11: Compute adversarial triplet loss $L _ { \mathrm { a t r } } ^ { k } , L _ { \mathrm { a t } 1 } ^ { k ^ { \prime } }$
12: Compute orthogonal regularization $\tilde { L } _ { \mathrm { o r t h o } } ^ { k } , L _ { \mathrm { o r t h o } } ^ { k ^ { \prime } }$
13: Compute classification losses $L _ { \mathrm { c e } } ^ { k } , L _ { \mathrm { c e } } ^ { a }$
14: Compute cosine consistency with frozen CLIP $L _ { \mathrm { { c o s } } } ^ { i } , L _ { \mathrm { { c o s } } } ^ { l }$
15: Update trainable parameters by minimizing $L _ { \mathrm { t o t a l } }$
16: end for
```

$$
P ( y = k \mid v ) = \frac { \exp ( s i m ( v , t _ { k } ) / \tau ) } { \sum _ { i = 1 } ^ { K } \exp ( s i m ( v , t _ { i } ) / \tau ) }\tag{1}
$$

where sim(·) denotes cosine similarity and $\tau$ is the temperature coeficient. The DiscoVL training procedure is outlined in Algorithm 1.

Multi-modal representation learning. MMRL [8] augments CLIP with a learnable representation space $\mathcal { A }$ that injects shared tokens into upper transformer layers of encoders. Given representation tokens $A \in \mathbb { R } ^ { M \times d _ { a } }$ , where M is the token number and $d _ { a }$ is the representation space dimension. A lightweight projector $\mathcal { G } ( \cdot )$ maps them into modality-specific latent spaces, enabling crossmodal interaction without altering the frozen lower layers:

$$
A ^ { i } = \{ A _ { \ell } ^ { i } \} _ { \ell = Q - 1 } ^ { L - 1 } , A _ { \ell } ^ { i } = \mathcal { G } _ { \ell } ^ { i } ( A ) , A ^ { l } = \{ A _ { \ell } ^ { l } \} _ { \ell = Q - 1 } ^ { L - 1 } , A _ { \ell } ^ { l } = \mathcal { G } _ { \ell } ^ { l } ( A )\tag{2}
$$

where $A _ { \ell } ^ { i } \in \mathbb { R } ^ { M \times d _ { i } }$ and $A _ { \ell } ^ { l } \in \mathbb { R } ^ { M \times d _ { t } }$ denote the injected tokens for the image and text branches at layer $\ell + 1$ , and Q marks the first layer that receives the additional context. Restricting the injections to higher layers preserves low-level priors learned during contrastive pre-training while allowing high-level semantics

to adapt. For image encoder I:

$$
[ k _ { \ell } , V _ { \ell } ] = I _ { \ell } ( k _ { \ell - 1 } , V _ { \ell - 1 } ) , \ell = 1 , \dots , Q - 1\tag{3}
$$

$$
[ k _ { \ell } , \ ⨏ _ { - } , \ V _ { \ell } ] = I _ { \ell } \big ( [ k _ { \ell - 1 } , A _ { \ell - 1 } ^ { i } , V _ { \ell - 1 } ] \big ) , \quad \ell = Q , \dots , L - 1\tag{4}
$$

$$
[ k _ { \ell } , A _ { \ell } ^ { i } , V _ { \ell } ] = I _ { \ell } ( [ k _ { \ell - 1 } , A _ { \ell - 1 } ^ { i } , V _ { \ell - 1 } ] ) , \quad \ell = L\tag{5}
$$

For text encoder $T , A _ { \ell } ^ { l }$ is prepended to the original sequence.

$$
[ h _ { \ell } , S _ { \ell } , g _ { \ell } ] = T _ { \ell } ( [ h _ { \ell - 1 } , S _ { \ell - 1 } , g _ { \ell - 1 } ] ) , \quad \ell = 1 , \dots , Q - 1\tag{6}
$$

$$
[ h _ { \ell } , \_ , S _ { \ell } , g _ { \ell } ] = T _ { \ell } ( [ h _ { \ell - 1 } , A _ { \ell - 1 } ^ { l } , S _ { \ell - 1 } , g _ { \ell - 1 } ] ) , \quad \ell = Q , \ldots , L - 1\tag{7}
$$

$$
h _ { \ell } , A _ { \ell } ^ { l } , S _ { \ell } , g _ { \ell } \big ] = T _ { \ell } \big ( \big [ h _ { \ell - 1 } , A _ { \ell - 1 } ^ { l } , S _ { \ell - 1 } , g _ { \ell - 1 } \big ] \big ) , \quad \ell = L\tag{8}
$$

## 3.2 Disentangled cross-modal representation aligner

Fig. 2 summarizes the overview of DiscoVL. The images are tokenized into a sequence of patches, which are processed through a frozen CLIP vision backbone. Starting from layer $Q { \mathrm { . } }$ , class prompts are encoded by the text encoder, and both modalities meet in a lightweight projector $\mathcal { G } ( \cdot )$ that lives entirely in the representation space. Cosine similarities between projected tokens deliver the logits that are optimized jointly with the orthogonal adversarial learning objective. Concretely, MMRL++ [9] factorizes each projector into (i) a globally shared kernel that captures layer-invariant knowledge and (ii) a lightweight residual that absorbs layer-specific nuances. For modality $m \in \{ i , l \}$ and layers $\ell \in \{ Q - 1 , \ldots , L - 1 \}$ , the factorization of projection weights is expressed as:

$$
F _ { \ell } ^ { m } = F _ { \mathrm { s h a r e d } } ^ { m } + J _ { \ell } D _ { \ell } , J _ { \ell } \in \mathbb { R } ^ { d _ { a } \times a _ { 1 } } , D _ { \ell } \in \mathbb { R } ^ { a _ { 1 } \times d _ { m } }\tag{9}
$$

where $a _ { 1 }$ denotes the rank of the low-rank factorization. $F _ { \mathrm { s h a r e d } } ^ { m } \in \mathbb { R } ^ { d _ { a } \times d _ { m } }$ is the globally shared component across layers, while $J _ { \ell } D _ { \ell }$ captures the layerspecific residual knowledge. To better focus on the most relevant subspaces for each layer and modality, we extend this approach to a multi-branch residual aligner, as depicted in ${ \mathrm { F i g . } }$ . 3a. This extension allows each layer to dynamically select diferent subspaces depending on the input representation, improving the model’s capacity to capture diverse aspects of the data. The residual component is now represented as a weighted sum over $E$ subspaces, where each subspace is associated with a distinct low-rank factorization:

$$
\varDelta F _ { \ell } ^ { m } = \sum _ { e = 1 } ^ { E } o _ { \ell } ^ { e } \left( J _ { \ell } ^ { e } D _ { \ell } ^ { e } \right) , J _ { \ell } ^ { e } \in \mathbb { R } ^ { d _ { a } \times a _ { 1 } } , D _ { \ell } ^ { e } \in \mathbb { R } ^ { a _ { 1 } \times d _ { m } }\tag{10}
$$

where $J _ { \ell } ^ { e } , D _ { \ell } ^ { e }$ are branch-specific low-rank components and $o _ { \ell } ^ { e }$ are attention weights that are dynamically predicted by a gating mechanism.

$$
o _ { \ell } = { \mathit { S o f t m a x } } { \left( g ( F _ { \ell } ^ { m } ) \right) }\tag{11}
$$

where $g ( \cdot )$ is a learned function that outputs the attention weights based on the current representation $F _ { \ell } ^ { m }$ . This gating mechanism allows the model to adaptively emphasize the most relevant subspace for each layer by assigning higher attention to subspaces that are most aligned with the current data. The designing of multiple branches enables the model to focus on distinct subspaces for each layer, allowing it to capture more complex and diverse relationships within and between modalities, while maintaining a compact and eficient parameterization. For the patch projection layer of representation, we define its weights as:

$$
F _ { a } = F _ { i } ^ { k } + J _ { a } D _ { a } , J _ { a } \in \mathbb { R } ^ { d _ { i } \times a _ { 2 } } , D _ { a } \in \mathbb { R } ^ { a _ { 2 } \times d }\tag{12}
$$

To further tighten alignment, DCRA injects bi-directional feedback between diferent modalities. In the lower half of the stack we enrich visual tokens with textual cues through a projection $Z _ { T 2 I } .$ , controlled by a learned fusion gate $\rho ( \ell )$

$$
A _ { \ell } ^ { i ^ { \prime } } = ( 1 - \rho ( \ell ) ) F _ { \ell } ^ { i } ( A _ { \ell } ^ { i } ) + \rho ( \ell ) Z _ { T 2 I } ( F _ { \ell } ^ { l } ( A _ { \ell } ^ { l } ) )\tag{13}
$$

In the upper half, we reverse the process by sending visual summaries back to the text stream through $Z _ { I 2 T }$

$$
A _ { \ell } ^ { l ^ { ' } } = ( 1 - \rho ( \ell ) ) F _ { \ell } ^ { l } ( A _ { \ell } ^ { l } ) + \rho ( \ell ) Z _ { I 2 T } ( F _ { \ell } ^ { i } ( A _ { \ell } ^ { i ^ { ' } } ) )\tag{14}
$$

The fusion gate $\rho ( \ell )$ is a learned function that enables the model to determine, at each layer $\ell ,$ the balance between preserving the original modality and fusing it with the other modality. This adaptive mechanism allows DiscoVL to focus on the most discriminative representation while prioritizing the modality that provides the most relevant information at each depth.

![](images/874e50d106b99dc7b9c13a2a3cc5a3a9304e4be35887e4b3e571ccd206aef598.jpg)  
(a)

![](images/df1492c27d25ea4a1698c62a906197d32c056439e51d4cee5897c1c353a16dd4.jpg)  
(b)  
Fig. 3: Main component. (a) Disentangled cross-modal representation aligner, which contains a multi-branch residual aligner and bi-directional feedback between diferent image-text modalities. (b) Adversarial triplet regularization. The red dashed box is vanilla adversarial triplet regularization.

## 3.3 Orthogonal adversarial representation learning

Adversarial triplet learning [17] remains a strong mechanism for separating class semantics, but in practice we observe that representation tokens quickly collapse toward class centroids, which limits their ability to generalize to new categories. To solve the issue, we design an adversarial representation triplet loss that explicitly regulates cross-class interactions between image features and representation features. Formally, let $f ^ { k }$ denote the image feature of class k obtained from the image encoder, and $f ^ { a }$ its corresponding adversarial representation from the unified embedding space. As depicted in Fig. 3b, for each anchor $f ^ { k }$ belonging to class $k ,$ we construct positive/negative pairs as: (1) Positive: the aligned representation feature $f ^ { a }$ from the same class k; (2) Negative: an image feature $f ^ { k ^ { \prime } }$ sampled from a distinct class $\boldsymbol { k } ^ { \prime } \neq \boldsymbol { k }$ . The loss is defined as follows, where m is a margin parameter.

$$
L _ { \mathrm { a t r } } ^ { k } = \operatorname* { m a x } { ( 0 , \ \lVert f ^ { k } - f ^ { a } \rVert _ { 2 } ^ { 2 } - \lVert f ^ { k } - f ^ { k ^ { \prime } } \rVert _ { 2 } ^ { 2 } + m ) } ,
$$

$$
L _ { \mathrm { a t r } } ^ { k ^ { \prime } } = \operatorname* { m a x } \left( 0 , \ \lVert f ^ { k ^ { \prime } } - f ^ { a ^ { \prime } } \rVert _ { 2 } ^ { 2 } - \lVert f ^ { k ^ { \prime } } - f ^ { k } \rVert _ { 2 } ^ { 2 } + m \right)\tag{15}
$$

(16)

The adversarial constraint aligns each image with its representation to ensure semantic consistency while separating features of diferent classes to prevent class-specific entanglement. However, redundant features can still be exploited by adversarial perturbations, which manipulate multiple correlated features simultaneously and cause large output deviations. Orthogonal regularization promotes feature independence, reducing redundancy and ensuring that each feature captures distinct information.

$$
L _ { \mathrm { o r t h o } } ^ { k } = \zeta \cdot \left\| I - { \frac { X ^ { k } ( X ^ { k } ) ^ { T } } { \| X ^ { k } \| ^ { 2 } } } \right\| _ { F } ^ { 2 }\tag{17}
$$

where $\zeta$ is the regularization hyperparameter. $X ^ { k } \in \mathbb { R } ^ { M \times d _ { i } }$ is the matrix representing the $f _ { k } , ~ I$ is the identity matrix. $\| \cdot \| _ { F }$ is the Frobenius norm, which represents the square root of the sum of squares of all elements in the matrix.

In our framework, representation tokens are designed to capture datasetspecific adaptations, while the class token preserves the pre-trained knowledge from CLIP. During training, we optimize both the representation tokens and the original class token, but prioritize the representation tokens to adapt them to the target dataset while retaining pre-trained knowledge. Specifically, for the image encoder I, processing an input through $L$ transformer layers yields a class token $k _ { L } ~ \in ~ \mathbb { R } ^ { d _ { i } }$ and a set of representation tokens $A _ { \ell } ^ { i ^ { ' } } ~ \in ~ \mathbb { R } ^ { M \times d _ { i } }$ . The final representation feature $a _ { L }$ is computed by averaging across the M tokens. Both class and representation tokens are projected into the shared vision-language latent space via patch projection layers:

$$
f _ { k } = Z _ { i m g } ^ { k } ( k _ { L } ) , f _ { a } = Z _ { i m g } ^ { a } ( a _ { L } )\tag{18}
$$

where $Z _ { i m g } ^ { k }$ is the frozen CLIP patch projection for class features, and $Z _ { i m g } ^ { a }$ is a trainable projection for representation features. For the text encoder $T$ , we map

the end-of-text token $g _ { L }$ into the shared vision-language space $t = Z _ { t x t } ( g _ { L } )$ and compute cross-entropy losses:

$$
L _ { \mathrm { c e } } ^ { k } = - \sum _ { k = 1 } ^ { K } y _ { k } \log p ( y = k \mid f _ { k } ) , L _ { \mathrm { c e } } ^ { a } = - \sum _ { k = 1 } ^ { K } y _ { k } \log p ( y = k \mid f _ { a } )\tag{19}
$$

where $y _ { k }$ is an indicator variable taking the value 1 if the image belongs to class $k ,$ and 0 otherwise. To further preserve the generalization ability of class features, we maximize the cosine similarity between the updated features $( f _ { k } , t )$ and the frozen CLIP features $( v , t _ { 0 } )$ :

$$
L _ { \mathrm { c o s } } ^ { i } = 1 - \frac { f _ { k } \cdot v } { \| f _ { k } \| \| v \| } , \quad L _ { \mathrm { c o s } } ^ { l } = 1 - \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \frac { t _ { k } \cdot t _ { 0 } ^ { k } } { | t _ { k } | | t _ { 0 } ^ { k } | }\tag{20}
$$

The total loss function are as follows:

$$
L _ { \mathrm { t o t a l } } = \alpha L _ { \mathrm { c e } } ^ { k } + ( 1 - \alpha ) L _ { \mathrm { c e } } ^ { a } + \beta ( L _ { \mathrm { c o s } } ^ { i } + L _ { \mathrm { c o s } } ^ { l } ) + \gamma ( L _ { \mathrm { a t r } } ^ { k } + L _ { \mathrm { a t r } } ^ { k ^ { \prime } } + L _ { \mathrm { o r t h o } } ^ { k } + L _ { \mathrm { o r t h o } } ^ { k ^ { \prime } } )\tag{21}
$$

where $\alpha , \beta ,$ and $\gamma$ are hyper-parameters controlling the relative weights of classification, generalization, and orthogonal adversarial triplet regularization. We combine the class features and the task-specific representation features to compute the probability for the base class [8]:

$$
p ( y = k \mid x ) = \alpha \cdot p ( y = k \mid f _ { k } ) + ( 1 - \alpha ) \cdot p ( y = k \mid f _ { a } )\tag{22}
$$

For novel classes, only the class token is used for prediction, as it retains the general knowledge of the pre-trained CLIP. Adversarial triplet learning representation tokens adapt to the dataset and are robust across classes, while the class token maintains generalization for new classes.

$$
p ( y = k \mid x ) = p ( y = k \mid f _ { k } )\tag{23}
$$

## 4 Experiments

## 4.1 Experimental setup

Evaluation and datasets. We follow the previous work [8, 16] to thoroughly evaluate DiscoVL across four tasks with several state-of-the-art methods: baseto-novel generalization, cross-dataset evaluation, few-shot learning, and domain generalization. For those benchmarks, we use the same datasets as followed by previous works [8,16]. For base-to-novel generalization, cross-dataset evaluation, and few-shot learning, we conduct experiments on 11 datasets from diverse domains: generic object classification on ImageNet [4] and Caltech101 [7]; finegrained classification on Oxford-Pets [25], StanfordCars [18], Flowers102 [24], Food101 [1], and FGVCAircraft [23]; scene recognition on SUN397 [33]; action recognition on UCF101 [29]; texture classification on DTD [3]; and satellite imagery recognition on EuroSAT [11]. We evaluate DiscoVL’s performance in domain generalization by testing it on four variants of ImageNet: ImageNet-V2 [27], ImageNet-Sketch [31], ImageNet-A [13], and ImageNet-R [12]. Although these datasets share the same class structure as ImageNet, they exhibit significant diferences in data distribution.

Table 1: Base-to-novel generalization results. HM indicates the harmonic mean. The best and second best results are marked in bold and underline. DiscoVL consistently improves base class performance while preserving generalization to novel classes.
<table><tr><td>Method</td><td></td><td></td><td>| Base Novel| HM</td><td>Method</td><td></td><td>| Base Novel | HM</td><td></td><td>Method</td><td>| Base Novel | HM</td><td>Method</td><td></td><td>| Base Novel</td><td></td><td>HM</td></tr><tr><td>CoOp [41] CoCoOp [40] KgCoOp [36]</td><td></td><td>82.69 63.22 80.47</td><td>71.66 71.69 75.83</td><td></td><td>CoOp [41] CoCoOp [40]</td><td>98.0089.81 97.96 93.81</td><td>93.73 95.84</td><td>CoOp [41] CoCoOp [40]</td><td>79.44 41.18 77.01 56.00</td><td>54.24 64.85</td><td>CoOp [41] CoCoOp [40]</td><td></td><td>92.19 54.74 60.04</td><td></td><td>68.69 71.21</td></tr><tr><td>MaPLe [15] PromptSRC [16]</td><td></td><td>80.73 73.60 82.28 84.26</td><td>75.14 76.10</td><td>77.00 78.55 79.97</td><td>KgCoOp [36] MaPLe [15] PromptSRC [16]</td><td>97.72 94.39 97.74 94.36 98.10 94.03</td><td></td><td>96.03 96.02</td><td>KgCoOp [36] MaPLe [15] PromptSRC [16]</td><td>77.55 54.99 64.35 80.36 59.18 68.16 83.37 62.97 71.75</td><td></td><td>KgCoOp [36] MaPLe [15] PromptSRC [16]</td><td>87.49 85.64 94.07</td><td>64.34 73.23</td><td>73.48 82.35</td></tr><tr><td>TCP [37] MMA [35]</td><td></td><td>84.13 83.20</td><td>75.36 76.80</td><td>79.51 79.87</td><td>TCP [37] MMA [35]</td><td></td><td>98.23 94.67 98.40 94.00</td><td>96.02 96.42</td><td>TCP [37] MMA [35]</td><td>82.77 58.07 68.25 83.20 73.38</td><td></td><td>TCP [37] MMA [35]</td><td>92.90 91.63</td><td>73.90 74.73</td><td>82.32 82.32</td></tr><tr><td>AAPL [17]</td><td></td><td>80.27</td><td>72.17</td><td>76.01</td><td>AAPL [17]</td><td></td><td>97.87 95.10</td><td>96.15 96.46</td><td>AAPL [17]</td><td>65.63 73.90 53.43</td><td></td><td>AAPL [17]</td><td>85.46</td><td>82.34</td><td>83.87</td></tr><tr><td>CoPrompt [28]</td><td></td><td>84.00 77.23</td><td></td><td>80.48</td><td>CoPrompt [28]</td><td>98.27 94.90</td><td></td><td>96.55</td><td>CoPrompt [28]</td><td>83.13 64.73</td><td>62.02 72.79</td><td>CoPrompt [28]</td><td>87.00 94.60</td><td>66.30</td><td>75.25</td></tr><tr><td>TextRefiner [34]</td><td></td><td>79.74 74.32</td><td></td><td>76.94</td><td>TextRefiner [34]</td><td>98.13 94.43</td><td></td><td></td><td>TextRefiner [34]</td><td>75.35 58.09</td><td>65.60</td><td>TextRefiner [34]</td><td></td><td>78.57</td><td>85.84</td></tr><tr><td>RAda [2]</td><td></td><td>84.32 76.25</td><td></td><td>80.08</td><td>RAda [2]</td><td></td><td>98.06 93.56</td><td>96.24 95.76</td><td>RAda [2]</td><td>82.06 67.15</td><td>73.86</td><td>RAda [2]</td><td>74.57</td><td>72.82</td><td>73.68 84.20</td></tr><tr><td>2SFS [6]</td><td></td><td>85.55 75.48</td><td></td><td>80.20</td><td>2SFS [6]</td><td></td><td>98.71 94.43</td><td>96.52</td><td>2SFS [6]</td><td>84.60 65.01</td><td>73.52</td><td>2SFS [6]</td><td>96.91 67.09</td><td>96.48 74.69</td><td>79.29</td></tr><tr><td>SkipT. [32] MMRL [8]</td><td></td><td>85.04 85.68</td><td>77.53 77.16</td><td>81.11 81.20</td><td>SkipT. [32] MMRL [8]</td><td></td><td>98.50 95.33</td><td>96.89</td><td>SkipT. [32]</td><td>83.77 67.23</td><td>74.59</td><td>SkipT. [32]</td><td>92.47 83.00</td><td></td><td>87.48</td></tr><tr><td>DiscoVL (Ours)|85.98 77.61|81.58</td><td></td><td></td><td></td><td></td><td>DiscoVL (Ours)|99.13 94.97 |97.01</td><td></td><td>98.97 94.50</td><td>96.68</td><td>MMRL [8] DiscoVL (Ours)|85.77 65.63</td><td>85.67 65.00</td><td>73.82 74.36</td><td>MMRL [8] DiscoVL (Ours)|</td><td>95.60 80.17</td><td></td><td>87.21</td></tr><tr><td>Δ (a) Average</td><td>+0.30 +0.45</td><td></td><td></td><td>+0.38</td><td>Δ (b) Caltech101</td><td></td><td>+0.16 +0.47</td><td>+0.33</td><td>Δ</td><td>+0.10 +0.63 (c) DTD</td><td>+0.54</td><td>Δ</td><td>+1.20 +1.03</td><td>96.80 81.20</td><td>|88.32 +1.11</td></tr><tr><td colspan="4"></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>(d) EuroSAT</td><td></td><td></td></tr><tr><td>Method CoOp [41] CoCoOp [40]</td><td></td><td>| Base Novel| HM 40.44 22.30 33.41 23.71</td><td>28.75</td><td>27.74</td><td>Method CoOp [41] CoCoOp [40]</td><td>| Base Novel| HM 88.33 82.26</td><td>90.70 91.29</td><td>85.19 90.99</td><td>Method CoOp [41] CoCoOp [40]</td><td>| Base Novel | HM 76.47 67.88</td><td>71.92</td><td>Method CoOp [41] CoCoOp [40]</td><td>| Base Novel| 97.60 59.67</td><td></td><td>HM 74.06</td></tr><tr><td>KgCoOp [36] MaPLe [15] PromptSRC [16]</td><td>36.21</td><td>33.55 37.44 35.61</td><td>34.83 36.50</td><td></td><td>KgCoOp [36] MaPLe [15]</td><td>90.50 91.70 90.71</td><td>92.05</td><td>91.09 91.38</td><td>KgCoOp [36] MaPLe [15]</td><td>75.98 70.43 75.83 69.96 76.66 70.54</td><td>73.10 72.78 73.47</td><td>KgCoOp [36] MaPLe [15]</td><td>95.00 95.92</td><td>94.87 71.75 74.73 72.46</td><td>81.71 83.65 82.56</td></tr><tr><td>TCP [37] MMA [35]</td><td>41.97</td><td>42.73 37.87 34.43</td><td>40.15 37.83</td><td></td><td>PromptSRC [16] TCP [37]</td><td>90.67 90.57</td><td>91.53 91.37</td><td>91.10 90.97</td><td>PromptSRC [16] TCP [37]</td><td>77.60 70.73 77.27 69.87</td><td>74.01 73.38</td><td>PromptSRC [16] TCP [37]</td><td>98.07 97.73</td><td>76.50 75.57</td><td>85.95 85.23</td></tr><tr><td>AAPL [17] CoPrompt [28]</td><td>34.07</td><td>40.57 36.33 24.17</td><td>38.33 28.28</td><td></td><td>MMA [35] AAPL [17]</td><td>90.70 91.60</td><td>90.13 91.30</td><td>90.71 91.15</td><td>MMA [35] AAPL [17]</td><td>77.31 71.00 76.53 70.57</td><td>74.02 73.43</td><td>MMA [35] AAPL [17]</td><td>97.77 95.10</td><td>75.93 70.63 76.60</td><td>85.48 81.06</td></tr><tr><td>TextRefiner [34]</td><td></td><td>40.20 39.33 35.35 35.87</td><td>39.76</td><td>35.61</td><td>CoPrompt [28] TextRefiner [34]</td><td>90.73 92.07</td><td>90.88 91.43</td><td>91.40 91.15</td><td>CoPrompt [28] TextRefiner [34]</td><td>77.67 71.27 76.84 70.54</td><td>74.33 73.56</td><td>CoPrompt [28] TextRefiner [34]</td><td>97.27 95.92</td><td>74.33</td><td>85.71 83.76</td></tr><tr><td>RAda [2] 2SFS [6]</td><td>41.72 47.48</td><td>38.09 35.51</td><td>39.82</td><td></td><td>RAda [2] 2SFS [6]</td><td>90.35 91.49</td><td>89.11 91.34</td><td>90.92</td><td>RAda [2] 2SFS [6]</td><td>77.96 70.23</td><td>73.89 74.20</td><td>RAda [2] 2SFS [6]</td><td>97.74</td><td>75.32 76.17</td><td>85.08 85.83</td></tr><tr><td>SkipT. [32] MMRL [8]</td><td>45.37 46.30</td><td>37.13</td><td>40.63 40.84</td><td></td><td>SkipT. [32]</td><td>90.67 92.03</td><td></td><td>90.21 91.34</td><td>SkipT. [32]</td><td>77.71 70.99 77.73 70.40</td><td>73.89</td><td>SkipT. [32]</td><td>98.29</td><td>98.57 75.80</td><td>85.70</td></tr><tr><td>DiscoVL (Ours)| Δ</td><td>46.40</td><td>37.70</td><td>37.03 41.15 |41.60</td><td></td><td>MMRL [8] DiscoVL (Ours)|</td><td>90.57</td><td>91.50 90.73 91.67</td><td>91.03 91.20</td><td>MMRL [8] DiscoVL (Ours)|78.07 71.63|74.71</td><td>77.90 71.30</td><td>74.45</td><td>MMRL [8] DiscoVL (Ours)|</td><td>98.97 77.27 |98.90 77.70|87.03</td><td></td><td>86.78</td></tr><tr><td>(e) FGVCAircraft</td><td></td><td>+0.10 +0.67</td><td>+0.45</td><td></td><td>Δ</td><td>+0.16 +0.17</td><td></td><td>+0.17</td><td>Δ</td><td>+0.17 +0.33</td><td>+0.26</td><td>Δ</td><td>-0.07 +0.43</td><td></td><td>+0.25</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td>(f) Food101</td><td></td><td></td><td>(g) ImageNet</td><td></td><td></td><td></td><td>(h) Flowers102</td><td></td><td></td></tr><tr><td>Method</td><td></td><td>| Base Novel| HM</td><td></td><td></td><td>Method</td><td>| Base Novel| HM</td><td></td><td></td><td>Method</td><td>| Base Novel | HM</td><td></td><td>Method</td><td></td><td>|Base Novel</td><td>HM</td></tr><tr><td>CoOp [41] CoCoOp [40] KgCoOp [36]</td><td></td><td>93.67 95.29 95.20 97.69</td><td>94.47 96.43</td><td></td><td>CoOp [41] CoCoOp [40]</td><td>78.12 60.40 70.49</td><td>73.59</td><td>68.13 72.01</td><td>CoOp [41] CoCoOp [40]</td><td>80.60 65.89 79.74 76.86</td><td>72.51 78.27</td><td>CoOp [41] CoCoOp [40]</td><td>84.69 82.33</td><td>56.05 73.45</td><td>67.46</td></tr><tr><td>MaPLe [15] PromptSRC [16]</td><td>95.43 97.76</td><td>94.65 97.76</td><td>96.18</td><td></td><td>KgCoOp [36] MaPLe [15]</td><td>71.76</td><td>75.04 74.00</td><td>73.36 73.47</td><td>KgCoOp [36] MaPLe [15]</td><td>80.29 76.53 80.82</td><td>78.36 79.75</td><td>KgCoOp [36]</td><td>82.89</td><td>76.67</td><td>77.64 79.65 80.77</td></tr><tr><td></td><td></td><td>95.33 97.30</td><td>96.58</td><td></td><td>PromptSRC [16]</td><td>72.94 78.27 74.97</td><td></td><td>76.58</td><td>PromptSRC [16]</td><td>78.70 82.67 78.47</td><td>80.52</td><td>MaPLe [15] PromptSRC [16]</td><td>83.00</td><td>78.66 78.80 80.77 80.03 82.20</td><td>82.74 83.83</td></tr><tr><td>TCP [37] MMA [35] AAPL [17]</td><td>94.67 95.40</td><td>97.20 98.07</td><td>96.30 95.92</td><td>96.72</td><td>TCP [37] MMA [35]</td><td>80.80 74.13</td><td>78.50 73.10</td><td>77.32 75.70</td><td>TCP [37] MMA [35]</td><td>82.63 78.20 82.27 78.57</td><td>80.35 80.38</td><td>TCP [37] MMA [35]</td><td>87.10 87.13 86.23</td></table>

(i) OxfordPets  
(j) StanfordCars  
(k) SUN397

Implementation details. We follow prior studies [8,16] and adopt a 16-shot learning setting across all experiments, except for the few-shot learning tasks. The ViT-B/16 variant of the CLIP model serves as the visual backbone for all experimental setups [8]. For each test image, we use the hand-crafted text prompt "a photo of a [class]." Optimization is performed using the AdamW optimizer with an initial learning rate of 0.001. Training on ImageNet for the base-tonovel generalization task spans 5 epochs, whereas training on the remaining datasets is conducted over 10 epochs. For cross-dataset evaluation and domain generalization tasks, we perform training for a single epoch on ImageNet. We report the base-class accuracy (%), denoted as Base, the novel-class accuracy (%), denoted as Novel, and their harmonic mean (%), denoted as HM, to compare the efectiveness of diferent methods. In the few-shot learning tasks, training is carried out for 5 epochs on ImageNet and 50 epochs for other datasets for 1, 2, 4, 8, 16 shot. We test the performance of models after training, the average accuracy is reported over three independent runs. We introduce M = 5 representation tokens starting from the Q-th Transformer layer, with Q = 6. The low rank dimensions $a _ { 1 }$ and $a _ { 2 }$ are set to 4 and 64, respectively. The parameter α is fixed at 0.7, and the parameter of $\beta$ is set to 0.5, the parameter γ is 0.3.

Table 2: Comparisons with state-of-the-art methods on cross-dataset evaluation. Bold values indicate the best results. Overall, DiscoVL provides the highest average accuracy, indicating better generalization.
<table><tr><td rowspan="2">Method</td><td colspan="3">Source</td><td colspan="9">Target dataset</td></tr><tr><td>|ImageNet Caltech101</td><td></td><td>Pets</td><td>Cars</td><td>Flowers102 Food101 Aircraft SUN397 DTD EuroSAT UCF101 Average</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>CoOp [41]</td><td>71.51</td><td>93.70</td><td>89.14 64.51</td><td></td><td>68.71</td><td>85.30</td><td>18.47</td><td>64.15</td><td>41.90</td><td>46.39</td><td>66.55</td><td>63.88</td></tr><tr><td>CoCoOp [40]</td><td>71.02</td><td>94.43</td><td>90.14 65.32</td><td></td><td>71.88</td><td>86.06</td><td>22.94</td><td>67.36</td><td>45.73</td><td>45.37</td><td>68.21</td><td>65.74</td></tr><tr><td>MaPLe [15]</td><td>70.72</td><td>93.53</td><td>90.49 65.57</td><td></td><td>72.23</td><td>86.20</td><td>24.74</td><td>67.01</td><td>46.49</td><td>48.06</td><td>68.69</td><td>66.30</td></tr><tr><td>PromptSRC [16]</td><td>71.27</td><td>93.60</td><td>90.25 65.70</td><td></td><td>70.25</td><td>86.15</td><td>23.90</td><td>67.10</td><td>46.87</td><td>45.50</td><td>68.75</td><td>65.81</td></tr><tr><td>TCP [37]</td><td>71.40</td><td>93.97</td><td>91.25</td><td>64.69</td><td>71.21</td><td>86.69</td><td>23.45</td><td>67.15</td><td>44.35</td><td>51.45</td><td>68.73</td><td>66.29</td></tr><tr><td>MMA [35]</td><td>71.00</td><td>93.80</td><td>90.30</td><td>66.13</td><td>72.07</td><td>86.12</td><td>25.33</td><td>68.17</td><td>46.57</td><td>49.24</td><td>68.32</td><td>66.61</td></tr><tr><td>AAPL [17]</td><td>71.37</td><td>94.17</td><td>90.73</td><td>65.10</td><td>71.67</td><td>86.00</td><td>23.03</td><td>66.80</td><td>44.80</td><td>41.83</td><td>69.30</td><td>65.34</td></tr><tr><td>CoPrompt [28]</td><td>70.80</td><td>94.50</td><td>90.73</td><td>65.67</td><td>72.30</td><td>86.43</td><td>24.00</td><td>67.57</td><td>47.07</td><td>51.90</td><td>69.73</td><td>67.00</td></tr><tr><td>SkipT. [32]</td><td>72.77</td><td>93.43</td><td>90.10</td><td>65.37</td><td>71.97</td><td>86.17</td><td>25.13</td><td>67.33</td><td>48.00</td><td>54.27</td><td>68.23</td><td>67.00</td></tr><tr><td>MMRL [8]</td><td>72.03</td><td>94.67</td><td>91.43</td><td>66.10</td><td>72.77</td><td>86.40</td><td>26.30</td><td>67.57</td><td>45.90</td><td>53.10</td><td>68.27</td><td>67.25</td></tr><tr><td>DiscoVL (Ours)|</td><td>72.10</td><td>94.73</td><td>91.90 66.30</td><td></td><td>72.87</td><td>86.70</td><td>26.47</td><td>67.80</td><td>46.13</td><td>53.07</td><td>68.80</td><td>67.48</td></tr><tr><td>Δ</td><td>+0.07</td><td>+0.06</td><td>+0.47 +0.20</td><td></td><td>+0.10</td><td>+0.30</td><td>+0.17</td><td>+0.23</td><td>+0.23</td><td>-0.03</td><td>+0.53</td><td>+0.23</td></tr></table>

## 4.2 Base-to-novel generalization

As reported in Table 1, DiscoVL achieves competitiveness across all key metrics, with improvements of 0.30%, 0.45%, and 0.38% on base, novel, and HM, respectively, over the previous best model, MMRL. DiscoVL provides a greater HM than all competitors in 7 out of 11 benchmarks and outperforms all competitors on average across datasets. Notably, on EuroSAT, DiscoVL improves HM by 1.11%. Although it underperforms Skip Tuning by 1.80% in novel-class accuracy, it achieves a 4.33% gain in base-class accuracy, leading to a higher overall HM and a more balanced base–novel trade-of.

## 4.3 Cross-dataset evaluation

As summarized in Table 2, DiscoVL achieves the competitive accuracy on ImageNet and shows superior performance in 6 out of the 10 datasets. DiscoVL demonstrates superior performance, with improvements of 0.48%, 0.48% and 0.23% for CoPrompt, Skip Tuning and MMRL, respectively. This highlights that the performance gain primarily arises from our disentangled cross-modal representation aligner, which alleviates representation inconsistency across layers and enhances robustness to distribution shifts.

## 4.4 Domain generalization

We evaluate the robustness of diferent methods on out-of-distribution datasets by training models on the ImageNet source dataset and evaluating their performance on four ImageNet variants with domain shifts. As shown in Table 3, our DiscoVL demonstrates significant performance across all scenarios. DiscoVL achieves the best performance on ImageNet-A and improves the average target accuracy over PromptSRC by 0.18%. Furthermore, DiscoVL achieves the largest relative improvement of 0.47% on ImageNet-R, which contains renditions with varied artistic styles, suggesting that DiscoVL efectively captures high-level semantic features that are invariant to style changes.

Table 3: Comparisons with state-of-the-art methods on domain generalization. On an average, DiscoVL achieves consistent improvement.
<table><tr><td rowspan="2">Method</td><td>Source</td><td colspan="5">Target dataset</td></tr><tr><td></td><td colspan="6">[ImageNet ImageNetV2 ImageNet-Sk ImageNet-A ImageNet-R Average</td></tr><tr><td>CLIP [26]</td><td>66.73</td><td>60.83</td><td>46.15</td><td>47.77</td><td>73.96</td><td></td><td>57.18</td></tr><tr><td>CoOp [41]</td><td>71.51</td><td>64.20</td><td>47.99</td><td>49.71</td><td>75.21</td><td></td><td>59.28</td></tr><tr><td>CoCoOp [40]</td><td>71.02</td><td>64.07</td><td>48.75</td><td>50.63</td><td></td><td>76.18</td><td>59.91</td></tr><tr><td>MaPLe [15]</td><td>70.72</td><td>64.07</td><td>49.15</td><td>50.90</td><td></td><td>76.98</td><td>60.27</td></tr><tr><td>PromptSRC [16]</td><td>71.27</td><td>64.35</td><td>49.55</td><td>50.90</td><td></td><td>77.80</td><td>60.65</td></tr><tr><td>MMA [35]</td><td>71.00</td><td>64.33</td><td>49.13</td><td>51.12</td><td></td><td>77.32</td><td>60.48</td></tr><tr><td>AAPL [17]</td><td>71.37</td><td>64.20</td><td>48.80</td><td>50.60</td><td></td><td>76.87</td><td>60.12</td></tr><tr><td>CoPrompt [28]</td><td>70.80</td><td>64.25</td><td>49.43</td><td>50.50</td><td></td><td>77.51</td><td>60.42</td></tr><tr><td>TextRefiner [34]</td><td>72.06</td><td>65.02</td><td>48.58</td><td>49.77</td><td></td><td>76.30</td><td>59.92</td></tr><tr><td>SkipT. [32]</td><td>72.77</td><td>65.67</td><td>49.73</td><td>51.13</td><td></td><td>78.27</td><td>61.20</td></tr><tr><td>MMRL [8]</td><td>72.03</td><td>64.47</td><td>49.17</td><td>51.20</td><td></td><td>77.53</td><td>60.59</td></tr><tr><td rowspan="2">DiscoVL (Ours) Δ</td><td>72.10</td><td>64.60</td><td>49.40</td><td>51.30</td><td></td><td>78.00</td><td>60.83</td></tr><tr><td>+0.07</td><td>+0.13</td><td>+0.23</td><td>+0.10</td><td></td><td>+0.47</td><td>+0.24</td></tr></table>

## 4.5 Few-shot learning

This evaluation examines the model’s transfer learning capability in limited-data scenarios. Our model is trained on subsets of the training data with 1, 2, 4, 8, and 16-shots per class and subsequently evaluated on the full test sets. As shown in Fig. 4, under extreme few-shot conditions (1/2-shot), DiscoVL demonstrates strong advantages, achieving 73.48% and 76.63%, respectively, surpassing the previous best methods. In the 16-shot setting, DiscoVL continues to achieve the highest performance at 84.58%, exceeding PromptSRC and Skip Tuning by 1.71% and 0.67%, respectively.

![](images/99105acbcba6eaf9f8ce7ff91e322381901d24981bc6ec33e0fb53a96765175c.jpg)  
Fig. 4: Comparison with previous few-shot fine-tuning methods on 11 datasets under diferent shot settings. Our DiscoVL achieves new state-of-the-art performance.

## 4.6 Ablation study

Core components. As shown in Table 4a, each component progressively contributes to performance improvement. The multi-branch residual aligner and bidirectional feedback (DCRA) improve HM from 81.20% to 81.32%, while introducing orthogonal adversarial representation learning further increases HM to 81.58%. Both modules consistently improve performance when applied individually, and their combination yields the best generalization.

Loss function. As shown in Table 4b, each loss component contributes incrementally: $L _ { \mathrm { a t r } } ^ { k }$ improves HM by +0.20%, adding $L _ { \mathrm { a t r } } ^ { k ^ { \prime } }$ yields a further +0.07%, and incorporating orthogonal regularization $L _ { \mathrm { o r t h o } }$ achieves the best HM of 78.42%, confirming that all terms are complementary.

Table 4: Ablation study of DiscoVL components (average over 11 datasets) and loss function contributions (StanfordCars).
<table><tr><td>Method</td><td>Base Novel</td><td>HM</td></tr><tr><td>Baseline</td><td>85.68 77.16</td><td>81.20</td></tr><tr><td>+ Multi-branch res aligner</td><td>85.70 77.23</td><td>81.24</td></tr><tr><td>+ Bidirectional feedback</td><td>85.77 77.3081.32</td><td></td></tr><tr><td>+ Adversarial representation</td><td>85.84 77.5081.46</td><td></td></tr><tr><td>+ Orthogonal regularization</td><td>85.98 77.61 81.58</td><td></td></tr></table>

(a) Main components

<table><tr><td>Baseline  $L _ { \mathrm { a t 1 } } ^ { k }$ </td><td> $L _ { \mathrm { a t r } } ^ { k ^ { \prime } }$ </td><td> $L _ { \mathrm { o r t h o } } ^ { k } { + } L _ { \mathrm { o r t h o } } ^ { k ^ { \prime } }$ </td><td>Base Novel</td><td>HM</td></tr><tr><td>√</td><td>× ×</td><td>×</td><td>81.30</td><td>75.07 78.06</td></tr><tr><td>√</td><td>√ X</td><td>X</td><td>81.57 75.20</td><td>78.26</td></tr><tr><td>√</td><td>√ √</td><td>×</td><td></td><td>81.7075.23 78.33</td></tr><tr><td>√</td><td>√</td><td>√ √</td><td></td><td>81.80 75.30 78.42</td></tr></table>

(b) Loss function

## 4.7 Visualization

T-SNE. As depicted in Fig. 5, we perform t-SNE visualizations on the EuroSAT to evaluate the efectiveness of DiscoVL. The results show clear class-wise clustering, indicating that the learned fea ture representations are highly discriminative. This is attributed to disentangled cross-modal representation aligner, which progressively integrates informa tion from diferent modalities, mitigates inter-modal heterogeneity, and enhances

![](images/05789a34cfdc7bb83c77589551de3f07376babcd4620b9d35923e50714aeccda.jpg)

![](images/eaf1fad628527dafdf659f1385210bee83974af3476017ce8786d9e43a577ac2.jpg)  
(a) Baseline  
(b) DiscoVL  
Fig. 5: t-SNE visualization. Diferent colors denote diferent classes.

the model’s balanced understanding of semantic information. Specifically, the disentanglement of shared and task-specific features ensures that common crossmodal knowledge is preserved while modality-unique characteristics are efectively isolated. Furthermore, orthogonal and adversarial regularization reduce feature redundancy and enforce feature independence.

Grad-CAM. In Fig. 6, the shared branch yields broad, difuse activations that capture class-agnostic contextual cues, while the task-specific branch produces more focused responses, pinpointing the most discriminative local regions for the target category. Notably, our DiscoVL achieves the most refined and con

![](images/ad4571b413f1f6f26c5323211d56692d417459532cea8a8d48d1379a6c7a2238.jpg)  
Fig. 6: Grad-CAM visualization.

centrated attention map. This demonstrates its superior ability to synergistically integrate global context and local details, outperforming the baseline, whose attention remains comparatively scattered and less definitive.

## 4.8 Training stability and eficiency

We assess the training stability of DiscoVL by reporting standard deviations across three random seeds, as shown in Table 5. DiscoVL demonstrates more stability than CoPrompt and MMRL in the base class on the ImageNet dataset. In the novel class, DiscoVLs’ stability is slightly lower than that of CoPrompt but shows improvement. On the EuroSAT dataset, DiscoVL achieves the lowest standard deviation, indicating higher stability. In the novel class, although the standard deviation increases compared to other methods, it still maintains relatively consistent performance. The OARL and DCRA mechanisms introduced in DiscoVL efectively enhance training stability, leading to more reliable and consistent adaptation performance across diferent seed conditions.

Table 6 compares the computational eficiency on ImageNet. Overall, DiscoVL achieves a superior accuracy-eficiency trade-of compared to MMRL. Specifically, it reduces the number of parameters by 55.7%, lowers latency from 6.31 ms to 6.11 ms, improves throughput, and slightly decreases FLOPs. Despite its significantly reduced model complexity, DiscoVL still improves HM by 0.26%, indicating that the proposed design enhances computational eficiency without sacrificing accuracy and even yields marginally better generalization.

Table 5: Training stability comparison, measured by mean ± standard deviation (%). ⋆ means reproduced by ourselves.
<table><tr><td rowspan="2">Method</td><td colspan="3">ImageNet</td><td colspan="3">EuroSAT</td></tr><tr><td>Base</td><td>Novel</td><td>HM</td><td>Base</td><td>Novel</td><td>HM</td></tr><tr><td>Coprompt*</td><td>76.97 ± 0.57 71.10 ± 0.00 73.92</td><td></td><td></td><td>93.67 ± 1.0877.73 ± 8.16 84.96</td><td></td><td></td></tr><tr><td>MMRL</td><td> $7 7 . 9 0 \pm 0 . 0 8$ </td><td> $7 1 . 3 0 \pm 0 . 2 8$ </td><td>74.45</td><td> $9 5 . 6 0 \pm 0 . 3 3$ </td><td>80.17 ± 5.05 87.21</td><td></td></tr><tr><td>SkipT.*</td><td> $7 7 . 7 7 \pm { \bf 0 . 0 9 }$ </td><td> $7 0 . 4 0 \pm 0 . 2 2$ </td><td>73.87</td><td> $9 2 . 6 0 \pm 1 . 2 8$ </td><td>82.10 ± 2.91 87.03</td><td></td></tr><tr><td>DiscoVL (Ours)</td><td> $7 8 . 0 7 \pm 0 . 1 3$ </td><td> $7 1 . 6 3 \pm 0 . 0 5$ </td><td></td><td>74.71|96.80 ± 0.21 81.20 ± 2.99 88.32</td><td></td><td></td></tr></table>

Table 6: Eficiency on a single NVIDIA RTX 3060 GPU.
<table><tr><td>Method</td><td>Param. (M) Lat.(ms)</td><td></td><td>FPS</td><td>FLOPs (G) Train time (min) HM (%)</td><td></td><td></td></tr><tr><td>MMRL</td><td>4.99</td><td>6.31</td><td>158.44</td><td>21.58</td><td>156</td><td>74.45</td></tr><tr><td>DiscoVL (Ours) 2.21(−2.78)</td><td></td><td></td><td>6.11(−0.20) 163.68(+5.24) 21.50(−0.08) 147(−9)</td><td></td><td></td><td>74.71(+0.26)</td></tr></table>

## 5 Conclusion

We identify representation entanglement, the conflation of class-specific and class-agnostic information, as a principal bottleneck of existing VLMs promptlearning methods and propose DiscoVL, a disentangled cross-modal representation learning framework that explicitly factorizes the adapted feature space into shared and task-specific subspaces. DCRA separates and progressively aligns cross-modal cues through multi-branch residual structure and bidirectional feedback, while OARL preserves the class-agnostic representation via orthogonal adversarial constraints that suppress token collapse. Across 15 benchmarks, we observe consistent performance gains on both base and novel classes. Dedicated disentanglement analyses further confirm that the separated components are genuinely distinct and functionally complementary. Future work will explore coupling DCRA with dynamic prompt schedulers and extending orthogonal adversarial learning to sequential or streaming settings.

## Acknowledgements

This work was supported by the Key R & D Program of Shandong Province, China under Grant No.2025CXPT094, 20 Guidelines for New Colleges in Jinan City under Grant No. 202333044, Qilu University of Technology (Shandong Academy of Sciences) Major Innovation Project of Science, Education and Industry Integration Pilot Project 2025ZDZX12, and the Natural Science Foundation of Shandong Province Young Scientists Fund under Grant No. ZR2026QC1572.

## References

1. Bossard, L., Guillaumin, M., Van Gool, L.: Food-101–mining discriminative components with random forests. In: European Conference on Computer Vision. pp. 446–461 (2014)

2. Chen, L., Ahmad, G.S., Yao, T., Liu, L., Shen, Z.: One last attention for your vision-language model. In: International Conference on Computer Vision. pp. 1464– 1473 (2025)

3. Cimpoi, M., Maji, S., Kokkinos, I., Mohamed, S., Vedaldi, A.: Describing textures in the wild. In: Computer Vision and Pattern Recognition. pp. 3606–3613 (2014)

4. Deng, J., Dong, W., Socher, R., Li, L.J., Li, K., Fei-Fei, L.: Imagenet: A largescale hierarchical image database. In: Computer Vision and Pattern Recognition. pp. 248–255 (2009)

5. Du, Y., Wei, F., Zhang, Z., Shi, M., Gao, Y., Li, G.: Learning to prompt for openvocabulary object detection with vision-language model. In: Computer Vision and Pattern Recognition. pp. 14084–14093 (2022)

6. Farina, M., Mancini, M., Iacca, G., Ricci, E.: Rethinking few-shot adaptation of vision-language models in two stages. In: Computer Vision and Pattern Recognition. pp. 29989–29998 (2025)

7. Fei-Fei, L., Fergus, R., Perona, P.: Learning generative visual models from few training examples: An incremental bayesian approach tested on 101 object categories. In: Computer Vision and Pattern Recognition Workshop. pp. 178–178 (2004)

8. Guo, Y., Gu, X.: Mmrl: Multi-modal representation learning for vision-language models. In: Computer Vision and Pattern Recognition. pp. 25015–25025 (2025)

9. Guo, Y., Gu, X.: Mmrl++: Parameter-eficient and interaction-aware representation learning for vision-language models. International Journal of Computer Vision 134(1), 11 (2026)

10. Hao, F., He, F., Wu, F., Wang, T., Song, C., Cheng, J.: Task-aware clustering for prompting vision-language models. In: Computer Vision and Pattern Recognition. pp. 14745–14755 (2025)

11. Helber, P., Bischke, B., Dengel, A., Borth, D.: Eurosat: A novel dataset and deep learning benchmark for land use and land cover classification. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing 12(7), 2217– 2226 (2019)

12. Hendrycks, D., Basart, S., Mu, N., Kadavath, S., Wang, F., Dorundo, E., Desai, R., Zhu, T., Parajuli, S., Guo, M., et al.: The many faces of robustness: A critical analysis of out-of-distribution generalization. In: International Conference on Computer Vision. pp. 8340–8349 (2021)

13. Hendrycks, D., Zhao, K., Basart, S., Steinhardt, J., Song, D.: Natural adversarial examples. In: Computer Vision and Pattern Recognition. pp. 15262–15271 (2021)

14. Jia, C., Yang, Y., Xia, Y., Chen, Y.T., Parekh, Z., Pham, H., Le, Q., Sung, Y.H., Li, Z., Duerig, T.: Scaling up visual and vision-language representation learning with noisy text supervision. In: International Conference on Machine Learning. pp. 4904–4916 (2021)

15. Khattak, M.U., Rasheed, H., Maaz, M., Khan, S., Khan, F.S.: Maple: Multi-modal prompt learning. In: Computer Vision and Pattern Recognition. pp. 19113–19122 (2023)

16. Khattak, M.U., Wasim, S.T., Naseer, M., Khan, S., Yang, M.H., Khan, F.S.: Selfregulating prompts: Foundational model adaptation without forgetting. In: International Conference on Computer Vision. pp. 15190–15200 (2023)

17. Kim, G., Kim, S., Lee, S.: Aapl: Adding attributes to prompt learning for visionlanguage models. In: Computer Vision and Pattern Recognition. pp. 1572–1582 (2024)

18. Krause, J., Stark, M., Deng, J., Fei-Fei, L.: 3d object representations for finegrained categorization. In: International Conference on Computer Vision Workshops. pp. 554–561 (2013)

19. Li, J., Li, D., Xiong, C., Hoi, S.: Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation. In: International Conference on Machine Learning. pp. 12888–12900 (2022)

20. Liang, F., Wu, B., Dai, X., Li, K., Zhao, Y., Zhang, H., Zhang, P., Vajda, P., Marculescu, D.: Open-vocabulary semantic segmentation with mask-adapted clip. In: Computer Vision and Pattern Recognition. pp. 7061–7070 (2023)

21. Liu, H., Li, C., Wu, Q., Lee, Y.J.: Visual instruction tuning. Advances in Neural Information Processing Systems 36, 34892–34916 (2023)

22. Liu, L., Wang, N., Yang, X., Gao, X., Liu, T.: Surrogate prompt learning: Towards eficient and diverse prompt learning for vision-language models. In: International Conference on Machine Learning (2025)

23. Maji, S., Rahtu, E., Kannala, J., Blaschko, M., Vedaldi, A.: Fine-grained visual classification of aircraft. arXiv preprint arXiv:1306.5151 (2013)

24. Nilsback, M.E., Zisserman, A.: Automated flower classification over a large number of classes. In: Computer Vision, Graphics & Image Processing. pp. 722–729 (2008)

25. Parkhi, O.M., Vedaldi, A., Zisserman, A., Jawahar, C.: Cats and dogs. In: Computer Vision and Pattern Recognition. pp. 3498–3505 (2012)

26. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al.: Learning transferable visual models from natural language supervision. In: International Conference on Machine Learning. pp. 8748–8763 (2021)

27. Recht, B., Roelofs, R., Schmidt, L., Shankar, V.: Do imagenet classifiers generalize to imagenet? In: International Conference on Machine Learning. pp. 5389–5400 (2019)

28. Roy, S., Etemad, A.: Consistency-guided prompt learning for vision-language models. In: International Conference on Learning Representations (2024)

29. Soomro, K.: Ucf101: A dataset of 101 human actions classes from videos in the wild. arXiv preprint arXiv:1212.0402 (2012)

30. Vasu, P.K.A., Faghri, F., Li, C.L., Koc, C., True, N., Antony, A., Santhanam, G., Gabriel, J., Grasch, P., Tuzel, O., et al.: Fastvlm: Eficient vision encoding for vision language models. In: Computer Vision and Pattern Recognition Conference. pp. 19769–19780 (2025)

31. Wang, H., Ge, S., Lipton, Z., Xing, E.P.: Learning robust global representations by penalizing local predictive power. Advances in Neural Information Processing Systems 32 (2019)

32. Wu, S., Zhang, J., Zeng, P., Gao, L., Song, J., Shen, H.T.: Skip tuning: Pretrained vision-language models are efective and eficient adapters themselves. In: Computer Vision and Pattern Recognition. pp. 14723–14732 (2025)

33. Xiao, J., Hays, J., Ehinger, K.A., Oliva, A., Torralba, A.: Sun database: Large-scale scene recognition from abbey to zoo. In: Computer Vision and Pattern Recognition. pp. 3485–3492 (2010)

34. Xie, J., Zhang, Y., Peng, J., Huang, Z., Cao, L.: Textrefiner: Internal visual feature as eficient refiner for vision-language models prompt tuning. In: AAAI Conference on Artificial Intelligence. vol. 39, pp. 8718–8726 (2025)

35. Yang, L., Zhang, R.Y., Wang, Y., Xie, X.: Mma: Multi-modal adapter for visionlanguage models. In: Computer Vision and Pattern Recognition. pp. 23826–23837 (2024)

36. Yao, H., Zhang, R., Xu, C.: Visual-language prompt tuning with knowledge-guided context optimization. In: Computer Vision and Pattern Recognition. pp. 6757–6767 (2023)

37. Yao, H., Zhang, R., Xu, C.: Tcp: Textual-based class-aware prompt tuning for visual-language model. In: Computer Vision and Pattern Recognition. pp. 23438– 23448 (2024)

38. Zhao, C., Wang, Y., Jiang, X., Shen, Y., Song, K., Li, D., Miao, D.: Learning domain invariant prompt for vision-language models. IEEE Transactions on Image Processing 33, 1348–1360 (2024)

39. Zheng, H., Yang, S., He, Z., Yang, J., Huang, Z.: Hierarchical cross-modal prompt learning for vision-language models. In: International Conference on Computer Vision. pp. 1891–1901 (2025)

40. Zhou, K., Yang, J., Loy, C.C., Liu, Z.: Conditional prompt learning for visionlanguage models. In: Computer Vision and Pattern Recognition. pp. 16816–16825 (2022)

41. Zhou, K., Yang, J., Loy, C.C., Liu, Z.: Learning to prompt for vision-language models. International Journal of Computer Vision 130(9), 2337–2348 (2022)

# Supplementary Material

## S1 Hyperparameter Analysis

Efectiveness of value $\gamma ,$ α and $\beta .$ . We perform an ablation study on the balancing parameter $\gamma$ in the orthogonal adversarial regularization. From the results in Table S1a, the optimal $\gamma$ is 0.3, while values of 0.2 and 0.4 result in slight performance degradation or no further improvement, respectively, indicating that overly strong regularization yields no additional benefits. The parameter α regulates the learning process by balancing the contributions of representation token features and class token features. As presented in Table S1b, performance generally improves with increasing α, with the HM peaking at 84.46% when α is 0.7. The penalty coeficient $\beta$ controls the strength of regularization, promoting alignment between class token features and CLIP’s fixed feature space. As depicted in Table S1c, raising $\beta$ from 0.0 to 0.5 slightly improves both base and novel class accuracy, yielding a peak HM of 41.60. Further increasing β to 2.0 results in a marginal performance drop.

Table S1: Ablation studies on $\gamma ,$ α and $\beta .$
<table><tr><td>γ</td><td>Base</td><td>Novel</td><td>HM</td><td>α</td><td>Base Novel</td><td>HM</td><td>β</td><td>Base</td><td>Novel</td><td>HM</td></tr><tr><td>0.1</td><td>90.67</td><td>91.60</td><td>91.13</td><td>0.0</td><td>88.53</td><td>80.47 84.31</td><td>0.0</td><td>46.37</td><td>37.20</td><td>41.29</td></tr><tr><td>0.2</td><td></td><td></td><td>90.40 91.57 90.98</td><td>0.3</td><td>88.40</td><td>80.23 84.12</td><td>0.2</td><td></td><td>46.4037.53</td><td>41.50</td></tr><tr><td>0.3</td><td></td><td></td><td>90.73 91.67 91.20</td><td>0.5</td><td>88.60</td><td>80.50 84.36</td><td>0.5</td><td>46.40 37.70 41.60</td><td></td><td></td></tr><tr><td>0.4</td><td></td><td></td><td>90.50 91.60 91.10</td><td>0.7</td><td>88.67 80.63 84.46</td><td></td><td>2.0</td><td></td><td>46.37 37.60 41.53</td><td></td></tr><tr><td colspan="3">(a) Food101</td><td></td><td>(b) UCF101</td><td></td><td></td><td>(c) FGVCAircraft</td><td></td><td></td><td></td></tr></table>

Regularization strategies. We explore regularization strategies to align class tokens with frozen CLIP features on EuroSAT. As summarized in Table S2, cosine regularization achieves the best performance with an HM of 88.32%. $L _ { 1 }$ regularization attains the lowest HM, while MSE regularization provides a competitive HM.

Efect of branch E. Table S3 on EuroSAT shows E = 4 is optimal, and $E = 2$ underperforms across all metrics, and $E = 6$ yields no further gain while adding cost. We adopt $E = 4$ as the default.

Table S2: Regularization  
Table S3: Value of E
<table><tr><td>Regularization| Base</td><td>Novel</td><td>HM</td></tr><tr><td>Cosine</td><td>96.80 81.20 88.32</td><td></td></tr><tr><td>L1</td><td>96.5080.87</td><td>88.00</td></tr><tr><td>MSE</td><td>96.6381.00</td><td>88.13</td></tr></table>

<table><tr><td>E Base</td><td>Novel</td><td>HM</td></tr><tr><td>2 96.61</td><td>81.07</td><td>88.17</td></tr><tr><td></td><td>4 96.80 81.20 88.32</td><td></td></tr><tr><td></td><td>6 96.47 80.93 88.02</td><td></td></tr></table>

Gate behavior analysis. As shown in Table S4, the fusion weights remain stable between 0.723 and 0.731 across all layers, indicating a strong cross-modal fusion strategy, where approximately 73% of the information is derived from the other modality. This consistently high gating value demonstrates efective bidirectional information flow and stable fusion strength between the text and visual modalities across diferent network depths.

Table S4: Fusion weights analysis.
<table><tr><td>Layer index</td><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td><td>7</td><td>mean</td></tr><tr><td>Fusion weights 0.7276 0.7273 0.7228 0.7294 0.7276 0.7276 0.7306 0.7276</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

## S2 Disentanglement Analysis

Separability. As shown in Table S5, the per-layer mean absolute cosine similarity between the shared and task-specific branches remains below 0.10 across all layers, demonstrating that their features are highly orthogonal with minimal overlap. The task-specific branch consistently outperforms the shared branch in linear probe accuracy, indicating that it encodes class-discriminative information, whereas the shared branch preserves class-agnostic representations. The stability of this trend across layers substantiates that DCRA achieves robust functional disentanglement at varying network depths.

Table S5: Separability across diferent layers (averaged over 11 datasets). ’Cos. Sim.’ is the mean absolute cosine similarity between shared and task-specific branch features. ’Probe Acc.’ is the base-class accuracy of a frozen linear probe trained on each branch independently.
<table><tr><td rowspan=1 colspan=3>Layer|Cos. Sim. (↓)|Shared Probe Acc. (%) Task-Specific Probe Acc. (%)</td></tr><tr><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>0.08</td><td rowspan=1 colspan=1>67.42                 81.35</td></tr><tr><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>0.07</td><td rowspan=1 colspan=1>68.11                 82.04</td></tr><tr><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>0.09</td><td rowspan=1 colspan=1>67.89                 82.47</td></tr><tr><td rowspan=1 colspan=1>9</td><td rowspan=1 colspan=1>0.06</td><td rowspan=1 colspan=1>68.53                 82.71</td></tr><tr><td rowspan=1 colspan=1>10</td><td rowspan=1 colspan=1>0.08</td><td rowspan=1 colspan=1>68.20                 82.13</td></tr><tr><td rowspan=1 colspan=1>11</td><td rowspan=1 colspan=1>0.07</td><td rowspan=1 colspan=1>68.74                 81.96</td></tr></table>

Branch intervention. Table S6 provides the branch intervention results. Zeroing-out the task-specific branch drops base-class accuracy by 6.2% but novel by only 1.2% in ImageNet; masking the shared branch barely afects base (−0.9%) but hurts novel generalization (−4.4%), validating functional disentanglement. This trend holds consistently across all three datasets, validating the functional disentanglement of the two branches.

Table S6: Branch intervention results. ’Full’ denotes the proposed DiscoVL. ’\Task’ and ’\Shared’ denote the task-specific and shared branches, respectively.
<table><tr><td>Setting</td><td>ImageNet Base Novel HM</td><td>EuroSAT Base Novel HM</td><td>UCF101 Base Novel HM</td></tr><tr><td>Full</td><td>|78.07 71.63 74.71|96.8081.2088.32|88.67 80.6384.46</td><td></td><td></td></tr><tr><td>Task</td><td>71.90 70.47 71.18</td><td>89.53 79.80 84.39</td><td>81.20 79.43 80.31</td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td>Shared</td><td>|77.13 67.20 71.82</td><td>|96.07 76.5385.19|</td><td>87.93 76.10 81.59</td></tr></table>