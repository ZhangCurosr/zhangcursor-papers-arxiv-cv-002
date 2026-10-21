# In PRAISE of Attributes: Grounding Retrieval Geometry with Query-Conditioned Subspaces for Person Re-Identification

Rajarshi Bhattacharya<sup>1\*</sup> Aryan Shukla<sup>1</sup> Shakeeb Murtaza<sup>1</sup> Christian Desrosiers<sup>1</sup> Jose Dolz<sup>1</sup> Maguelonne Heritier<sup>2</sup> Eric Granger<sup>1</sup>

<sup>1</sup>LIVIA, École de technologie supérieure <sup>2</sup>Genetec Inc. <sup>§</sup>Code

## Abstract

Person re-identification (ReID) ranks an image gallery using either an image query or a textual query. In image-to-image (I2I) ReID, viewpoint, illumination, occlusion, and background changes can make distractors appear more similar than the true match. In text-to-image (T2I) ReID, incomplete or ambiguous descriptions may omit key visual evidence. We study this problem for vision-language model (VLM) based ReID, where attributes such as clothing color and carried objects provide useful fine-grained matching cues. However, directly using attribute-prompt similarities as retrieval scores is unreliable because templated prompts can be biased, correlated, and poorly calibrated. We therefore propose PRompt-guided Attribute-based Inference-time Similarity Enhancement (PRAISE), a query-side test-time adaptation framework for VLM-based ReID. Instead of treating prompts as scores, PRAISE uses a transferable attributeprompt bank to construct an Attribute-Conditioned Subspace (ACS): a query-dependent low-rank subspace induced by the most compatible attribute prompts. ACS provides a controlled attribute-conditioned geometryfor reshaping query– gallery affinities and guiding lightweight inference-time updates without identity or attribute labels. Experiments across multiple I2I- and T2I-ReID benchmarks show that PRAISE consistently improves VLM retrieval backbones and outperforms both generic and ReID-specific TTA baselines.

## 1. Introduction

Person re-identification (ReID) aims to retrieve images of the same identity across cameras and time. It is commonly studied in two retrieval settings: image-to-image (I2I) ReID, where an image query is matched against an image gallery, and text-to-image (T2I) ReID, where a natural-language description is used to retrieve the corresponding person image.

![](images/b64a4d408751812488488d2fd87e4035bd37b8313f4ac835d696c09d4c6447f2.jpg)  
(a)  
(b)

![](images/1ea33d6a5dbbc3eb1adb951a20e7bae6cc0d8de733cef85e3c108f4d20568d96.jpg)  
(c)  
Figure 1. Motivation and overview of PRAISE. (a) In I2I-ReID, raw visual similarity can retrieve distractors caused by camera, pose, or background bias. (b) In T2I-ReID, incomplete text descriptions can produce partial matches that share only some mentioned attributes. (c) PRAISE addresses both cases by selecting query-relevant attribute directions, orthogonalizing them into an Attribute-Conditioned Subspace (ACS), and using this controlled subspace to reshape query-gallery affinities during test-time.

Despite steady progress in both settings using image and text supervision [27, 58], test-time retrieval remains sensitive to nuisance factors. In I2I-ReID, viewpoint changes, pose variation, blur, illumination, occlusion, and background clutter can make visually plausible distractors outrank the true match [40]. In T2I-ReID, retrieval can fail for a complementary reason: natural-language descriptions are often incomplete, ambiguous, or noisy, and may omit discriminative visual evidence [35]. Thus, although the query modality differs, both settings suffer from insufficiently reliable querygallery evidence at test time.

A natural way to strengthen this evidence is to use semantic attributes as an auxiliary matching signal. If a query and a gallery image depict the same identity, their match should be supported not only by raw embedding similarity, but also by agreement over stable person-centric cues such as clothing color, clothing type, carried objects, hair, or accessories. Conversely, if two samples correspond to different identities, at least some of these cues are likely to disagree. We refer to such cues as attributes: attributes that are visually observable, likely to remain meaningful across camera views, and expressible in both image and text queries. These attributes are not intended to uniquely identify a person, but they can provide useful corrective evidence for rejecting implausible matches and reinforcing plausible ones.

Attribute-based cues have been widely explored in person retrieval, including attribute-based ReID and text-based person search [6, 13, 45, 52]. Prior methods typically use attributes as predicted labels, auxiliary supervision, or learned cross-modal embeddings, relying on training or fine-tuning to calibrate visual evidence with attribute semantics. In contrast, our work considers a more realistic setting in which no target-domain identity labels or attribute annotations are available, and no additional attribute classifier can be trained. Under these constraints, we construct a transferable bank ofattributes that capture visually observable, person-centric cues, such as clothing color and type, hair, and accessories, that remain stable across camera views and can be consistently expressed in both image and text queries. VLMs [15, 27, 58] can encode these cues as naturallanguage prompts, allowing textual semantics to be reused in I2I-ReID, where the text branch is typically discarded at inference [27], while naturally supporting T2I-ReID. A simple approach is to compare query and gallery responses to this prompt bank and fuse the resulting attribute similarity with identity affinity. However, we show that this direct fusion is unreliable because attribute prompts are coarse, shared across identities, correlated in VLM text space, and biased by prompt templates. Consequently, raw attribute scores are often poorly calibrated and can amplify generic distractors instead of improving retrieval.

This motivates a different test-time use of attributes. Instead of acting as direct matching scores, prompts should guide the adaptation geometry itself. To this end, we propose PRAISE: PRompt-guided Attribute-based Inferencetime Similarity Enhancement. For each query, PRAISE selects the most compatible attribute prompts and orthogonalizes their directions to form an Attribute-Conditioned Subspace (ACS). Query-gallery comparisons are then reshaped within this low-rank, query-conditioned subspace, allowing attributes to provide structured auxiliary evidence without relying on brittle prompt-profile matching. The resulting ACS affinity is combined with the standard identity affinity and used to drive a lightweight test-time adaptation objective that updates only normalization parameters. We assume a transductive retrieval setting where the gallery is available at test time, following prior works [1, 43]. The same mechanism applies to both I2I-ReID and T2I-ReID, differing only in whether the query representation is produced by the visual or text encoder.

Our contributions are summarized as follows: (1) PRAISE, a test-time adaptation method for ReID that improves query-gallery ranking by combining identity affinity with prompt-guided attribute evidence while updating only lightweight normalization parameters. (2) The Attribute-Conditioned Subspace (ACS), a test-time-compatible mechanism that uses attribute prompts as query-conditioned geometric directions rather than direct matching scores, mitigating prompt-similarity bias during retrieval. (3) A label-free, transferable attribute prompt-bank construction protocol for ReID. The bank captures stable person-centric cues, is preencoded once, requires no target-domain identity or attribute labels, and supports both I2I and T2I retrieval.

## 2. Related Works

Image-to-Image (I2I) Re-identification. I2I-ReID methods were developed as the principal form of Re-identification in different scenarios such as ordinary/traditional ReID [9, 17, 39, 51], clothes-changing ReID [7, 12, 24, 26], Visual Infrared-ReID [19, 33, 44, 53, 54, 57] for low lighting conditions, etc. Recently, models such as CLIP-ReID [27] have emerged, which use the superior representation power of VLMs in order to perform Re-identification. Generalpurpose models such as PLIP [58] have also been proposed, which are essentially VLMs intended for use in ReID settings for both I2I and T2I (Text-to-image) ReID. However, existing VLM-based ReID methods largely underutilize multimodal alignment by relying on cosine similarity between visual embeddings. We argue that this image-only approach overlooks the text branch as a stable semantic prior for mitigating domain shifts, such as camera bias, pose variation [40], and visual noise.

Test-time Adaptation of VLMs. Several methods have been proposed to adapt VLMs to unseen data streams. CLI-PArTT [8] introduces a cross-modal pseudo-labeling loss that refines image-wise prompts, while WATT [31] utilizes prompt ensembling to stabilize retrieval. However, these loss functions often remain heuristically decoupled from the original contrastive geometry of the backbone. Alternative paradigms, such as TPT [38], employ prompt tuning to optimize context tokens at test time, but reliance on multiple stochastic augmentations introduces significant computational overhead. Gradient-free alternatives like TDA [18] utilize cache-based pseudo-labeling to simulate few-shot episodes, yet struggle under severe out-of-distribution shifts where the cached support becomes unreliable. In contrast to classification-centric TTA methods, we propose PRAISE, which modulates adaptation geometry through an attributeconditioned subspace. By updating only a small set of normalization parameters, PRAISE enables robust openset adaptation without unstable global updates or costly augmentation-based tuning.

Text-to-Image (T2I) Re-identification. Existing T2I-ReID methods [5, 14, 35, 41, 47, 49, 56] predominantly follow a tripartite taxonomy based on architectural design: (i) unimodal alignment via modality-specific encoders $( \mathrm { e . g . }$ ResNet [10], BERT [46]), (ii) zero-shot retrieval using general-purpose vision-language models like CLIP [37], and (iii) domain-specific multimodal pre-training tailored for ReID. While recent advancements in cross-modal embedding models have bridged the high-level semantic gap, these offline frameworks remain constrained by a static internal knowledge base, making them brittle against dynamic or underspecified text queries during inference. Recent literature has explored interactive paradigms, such as THI [36], which utilizes multi-round visual question answering with MLLMs to refine query intent. However, these methods often incur high computational cost and suffer from label leakage or error accumulation. In contrast, PRAISE uses an attributeconditioned subspace (ACS), aligning visual features with textual prompts via lightweight normalization updates. This enables efficient open-set adaptation without multi-round refinement.

## 3. Background: Attribute-Guided Retrieval

Notation. Denoting as $\mathcal { D } _ { g } = \{ ( x _ { j } , p _ { j } , c _ { j } ) \} _ { j = 1 } ^ { N _ { g } }$ the gallery where $x _ { j }$ is a gallery image, $p _ { j }$ is a person label and $c _ { j }$ is a camera label<sup>1</sup>, we precompute gallery embeddings: ${ \bf Z } _ { g } = \mathcal { F } _ { v } ( { \bf X } _ { g } ) \ \in \ \mathbb { R } ^ { N _ { g } \times d }$ , where each row $\mathbf { z } _ { g , j }$ is $\ell _ { 2 } \cdot$ -normalized. In practice, $\mathbf { Z } _ { g }$ is cached offline, once per test set, to avoid excessive computational loads. Next, let $\mathcal { P } = \{ s _ { \ell } \} _ { \ell = 1 } ^ { L }$ be a bank of attribute prompts spanning multiple attribute categories. The specific design of the prompt bank will be discussed later. Encoded prompts are denoted as $\mathbf { Z } _ { t } = \mathcal { F } _ { t } ( \mathcal { P } ) \in \mathbb { R } ^ { L \times d }$ with a row-wise $\ell _ { 2 }$ normalization.

The central goal of ReID is to rank a gallery for a given query such that images of the same identity appear near the top of the list. In most modern VLM-based pipelines, this is done by comparing normalized embeddings from a visual encoder $\mathcal { F } _ { v }$ and/or text encoder $\mathcal { F } _ { t }$ . For a query $( x _ { i } .$ image or text), the query representation is obtained as $\mathbf { z } _ { i } = \mathcal { F } _ { v } ( x _ { i } )$ or $\mathbf { z } _ { i } = \mathcal { F } _ { t } ( x _ { i } )$ depending on query modality. Gallery images are encoded as $\mathbf { Z } _ { g } = \mathcal { F } _ { v } ( \mathbf { X } _ { g } )$ , where each row $\mathbf { z } _ { g , j }$ corresponds to one gallery image. The standard retrieval affinity is then computed by cosine similarity:

$$
\begin{array} { r } { \mathbf { A } _ { \mathrm { i d } } [ i , j ] = \tilde { \mathbf { z } } _ { i } ^ { \top } \tilde { \mathbf { z } } _ { g , j } , } \end{array}\tag{1}
$$

where $\tilde { \mathbf { z } } = \mathbf { z } / ( \| \mathbf { z } \| _ { 2 } + \epsilon )$ . This identity score is simple and effective, but it is also vulnerable to nuisance factors such as viewpoint, illumination, camera style, background clutter,

occlusion, and incomplete textual descriptions. These factors can cause visually plausible distractors to receive high similarity even when they do not share the same identity.

## 3.1. Attribute Scores as Corrective Retrieval Signal

A natural way to improve retrieval robustness is to introduce semantic attributes as an additional source of evidence. Attributes such as clothing color, sleeve length, lower-body clothing type, gender, hair, carried objects, and accessories are not sufficient to uniquely identify a person, but they often provide useful constraints on whether a query-gallery pair is plausible. Following this intuition, an established design pattern is score fusion [2, 22]. The retrieval system computes two affinities for each query-gallery pair: an identity or appearance affinity $\mathbf { A } _ { \mathrm { i d } }$ , and an attribute-based affinity $\mathbf { A } _ { \mathrm { a t t } } .$ . The final matching score is then formed as a weighted combination:

$$
\mathbf { A } [ i , j ] = \mathbf { A } _ { \mathrm { i d } } [ i , j ] + \lambda _ { \mathrm { a t t } } \mathbf { A } _ { \mathrm { a t t } } [ i , j ] ,\tag{2}
$$

where $\lambda _ { \mathrm { a t t } } \geq 0$ controls the contribution of the attribute branch. Conceptually, $\mathbf { A } _ { \mathrm { i d } }$ captures the raw embedding-level similarity between the query and gallery image, while $\mathbf { A } _ { \mathrm { a t t } }$ acts as a corrective semantic score. The identity score can come from an existing ReID backbone, while the attribute score can be computed using an auxiliary attribute classifier, a text-attribute model, or a VLM prompt bank. A high attribute score should promote candidates that share queryconsistent semantic evidence, while a low attribute score should suppress candidates whose appearance similarity is likely caused by nuisance correlations.

Let $\mathcal { P } = \{ s _ { \ell } \} _ { \ell = 1 } ^ { L }$ denote a bank of natural-language attribute prompts, where each prompt describes one semantic attribute, for example $\mathbf { \ddot { a } }$ photo of a person wearing a red $\mathrm { s h i r t } ^ { \prime \prime } \qquad \mathrm { o r } \qquad \mathrm { \ " { a } }$ ph $\mathsf { \Omega } _ { \mathsf { I } } \mathsf { o } \mathsf { t } \circ \mathsf { \Omega } \circ \mathsf { f } \ : \mathsf { \Omega } \mathsf { a }$ person carrying a backpac $\mathrm { : k } ^ { \prime \prime }$ . These prompts are encoded by the text encoder:

$$
\mathbf { Z } _ { t } = \mathcal { F } _ { t } ( \mathcal { P } ) \in \mathbb { R } ^ { L \times d } ,\tag{3}
$$

with row-wise $\ell _ { 2 }$ normalization to get $\bar { \mathbf Z } _ { t }$ . For a query embedding $\mathbf { z } _ { i } ,$ its similarity to the attribute bank is ${ \mathbf u } _ { i } = \bar { \mathbf Z } _ { t } { \mathbf z } _ { i } ^ { \top } \in$ $\mathbb { R } ^ { L }$ . Similarly, for a gallery embedding ${ \bf z } _ { g , j }$ , its attribute similarity profile is $\mathbf { v } _ { j } = \bar { \mathbf { Z } } _ { t } \mathbf { z } _ { g , j } ^ { \top } \in \mathbb { R } ^ { L }$ . The vectors $\mathbf { u } _ { i }$ and $\mathbf { v } _ { j }$ can be interpreted as soft attribute profiles. Each dimension measures how compatible the query or gallery sample is with one prompt in the attribute bank. Therefore, if the query and gallery image correspond to the same identity, one may expect their attribute profiles to be similar. This leads to a direct attribute similarity:

$$
\begin{array} { r } { \mathbf { A } _ { \mathrm { a t t } } ^ { \mathrm { d i r e c t } } [ i , j ] = \tilde { \mathbf { u } } _ { i } ^ { \top } \tilde { \mathbf { v } } _ { j } , } \end{array}\tag{4}
$$

where $\tilde { \mathbf { u } } _ { i } { = } \mathbf { u } _ { i } / ( \lVert \mathbf { u } _ { i } \rVert _ { 2 } { + } \epsilon )$ and $\tilde { \mathbf { v } } _ { j } = \mathbf { v } _ { j } / ( \lVert \mathbf { v } _ { j } \rVert _ { 2 } + \epsilon )$ . In this view, $\mathbf { A } _ { \mathrm { a t t } } ^ { \mathrm { d i r e c t } } [ i , j ]$ measures whether the query and gallery induce similar responses over the same set of attribute prompts.

![](images/9ed439198686aab0113d20231a7b3609f2c279ad23a8d92a3cb9edd1da337da1.jpg)  
(a) Without ACS

![](images/3c6d95cbea5a13bbca21cdf27d87389b8f32c5b733ea6f347f2a8bfb38ce264f.jpg)  
(b) With ACS  
Figure 2. Similarity distributions for matching and non-matching query-gallery pairs on Market-1501. Direct prompt-profile similarity (a) is saturated for both groups, whereas ACS (b) yields clearer separation between matches and distractors.

It is a query-gallery similarity computed not in the original image/text embedding space, but in the prompt-response space.

The appeal of this construction is that it converts attributes into a retrieval-compatible score without requiring explicit attribute labels. It also applies to both I2I and T2I ReID. In I2I ReID, both $\mathbf { u } _ { i }$ and $\mathbf { v } _ { j }$ are obtained by comparing image embeddings to text prompts. In T2I ReID, $\mathbf { u } _ { i }$ is obtained from a text-query embedding and $\mathbf { v } _ { j }$ from a gallery-image embedding, but both are still expressed as responses to the same prompt bank. Thus, direct attribute similarity provides a unified mechanism for using semantic attributes across both retrieval settings.

## 3.2. Calibration of Prompt-Response Similarity

Although direct attribute similarity offers a simple auxiliary retrieval score, it assumes that prompt activations form a reliable matching space. This is unsuitable for ReID because: (1) Identity Closeness: coarse attributes $( \mathrm { e . g . }$ , “dark hair”) are shared by many identities and are not identity-specific; (2) Template Bias: responses may reflect shared prompt templates rather than visual evidence; e.g., “a photo ofa person wearing a red shirt” and “a photo of a person wearing a blue shirt” differ only in the color token but share almost the entire sentence structure. [3, 32] (3) Attribute Collision: semantically close captions can produce highly correlated activations; e.g., red/pink/magenta, pants/trousers/long pants, or bag/backpack/shoulder bag [16, 42]. As a result, $\mathbf { A } _ { \mathrm { a t t } } ^ { \mathrm { d i r e c t } }$ may have a narrow or poorly structured distribution, making true matches and distractors hard to separate and misaligning it with the raw identity affinity $\mathbf { A } _ { \mathrm { i d } }$

This behavior is illustrated in Figure 2. The direct attribute-derived affinity exhibits a markedly different distribution from the baseline identity affinity, indicating weak agreement between the two matching views. Under such mismatch, additive fusion becomes unstable: the attribute term is intended to correct $\mathbf { A } _ { \mathrm { i d } } ,$ but can instead amplify distractors that merely share generic attributes with the query.

This suggests that attribute prompts should not be used as unconstrained scoring functions. Instead, they should define a more controlled comparison space in which identity-level affinity remains the primary signal, while prompt-template bias is reduced and correlated attribute captions are prevented from dominating the score. Such a formulation would preserve the corrective value of attributes without allowing coarse or colliding prompt responses to destabilize retrieval.

## 4. Attribute conditioned subspace construction and PRAISE

Given a frozen VLM and a target-domain test set (queries and gallery), we propose PRAISE, a TTA method that leverages noisy natural-language attribute prompts to recalibrate the embedding space under domain shift to improve querygallery ranking for both I2I-ReID and T2I-ReID. PRAISE is a TTA method that performs the following key steps: (1) precompute a bank of attribute prompt embeddings, (2) construct a query-conditioned attribute subspace using the prompt bank, and (3) adapt a small subset of backbone parameters at test time, to improve retrieval performance. Please refer to Figure 3 for an overview of the methodology.

## 4.1. Prompt Bank Construction

The quality of the attribute bank is important because PRAISE relies on prompt-induced semantic directions rather than supervised attribute labels. We therefore construct $\mathcal { P }$ through a data-driven but label-free procedure that extracts recurring pedestrian attributes while keeping the vocabulary compact and reproducible.

• Captioning. We use Qwen2.5-VL-7B-Instruct [50] to generate short person-centric descriptions for pedestrian images. The dataset can be any standard ReID dataset. These descriptions are used only to mine recurring attributes, not as identity annotations.

• Attribute grouping. We map the generated descriptions into coarse pedestrian categories, including hair, gender, upper-/lower-body clothing color, clothing type, and accessories.

• Vocabulary cleaning. Within each category, we keep unique attribute phrases and remove duplicate or nearidentical variants, yielding a compact set of attribute suffixes. Each suffix is inserted into the shared template $s _ { \ell } = \ " \mathtt { A }$ photo of [ATT]” to form $\mathcal { P } = \{ s _ { \ell } \} _ { \ell = 1 } ^ { L }$ The full suffix list is provided in the Supplementary Material.

• Prompt encoding and centering. We encode the bank as $\mathbf { Z } _ { t } = \mathcal { F } _ { t } ( \mathcal { P } ) \in \mathbb { R } ^ { L \times d }$ and apply row-wise $\ell _ { 2 }$ normalization. To reduce shared-template bias, we also encode a neutral prompt s , e.g., “a photo ofa person”, and use centered directions $\bar { \mathbf { z } } _ { t , \ell } = \mathbf { z } _ { t , \ell } - \mathbf { z } _ { t , 0 }$ for $\ell \in \{ 1 , \ldots , L \}$

Unless otherwise stated, we mine the attribute bank once from the Market-1501 training split and freeze it for all experiments. The same centered bank $\bar { \mathbf Z } _ { t }$ is reused across all evaluation datasets and is not adapted using target identity labels, target attribute labels, or dataset-specific supervision. The resulting bank should therefore be understood as a fixed, label-free, transferable pedestrian attribute vocabulary rather than a dataset-specific classifier. In the Supplementary Material, we report prompt-bank source ablations showing that

PRAISE remains stable when the bank is mined from different ReID datasets.

## 4.2. Attribute-Conditioned Subspace (ACS) Construction

We now define the attribute-conditioned subspace (ACS) that will be used by PRAISE. For a given query, ACS is a low-rank, query-conditioned subspace spanned by the text directions of the most compatible attribute prompts. Rather than treating prompt similarities as final matching scores, ACS uses prompts to identify a small set of semantic directions along which query-gallery comparisons can be made more reliably. Thus, the prompt bank acts as the source of candidate attribute directions from which the query-specific subspace is constructed.

Finding the base attribute directions. For each query embedding $\mathbf { z } _ { i }$ (row i of Z), we compute prompt activations using the prompt bank: $\mathbf { u } _ { i } = \bar { \mathbf { Z } } _ { t } \mathbf { z } _ { i } ^ { \top } \in \mathbb { R } ^ { L }$ . Please note that the query embedding (Z) is the encoder representation of a query image batch $( \mathbf { X } _ { q } )$ in case of I2IReID $( \mathrm { i } . \mathrm { e } . , \mathcal { F } _ { v } ( \mathbf { X } _ { q } ) )$ or a query text description $( \mathcal { P } _ { q } )$ in case of T2IReID (i.e., $\mathcal { F } _ { t } ( \mathcal { P } _ { q } ) )$ . We then select a small index set of the most activated prompts:

$$
\begin{array} { r } { A _ { i } = \mathrm { T o p R } ( \mathbf { u } _ { i } ) , \qquad | A _ { i } | = r , \qquad r \ll L . } \end{array}
$$

This selection is treated as an index operation, i.e., stopgradient through TopR. This allows us to find a set of attribute direction vectors:

$$
\mathbf { U } _ { i } = \left[ \bar { \mathbf { z } } _ { t , \ell _ { 1 } } \cdot \cdot \cdot \bar { \mathbf { z } } _ { t , \ell _ { r } } \right] ^ { \top } \in \mathbb { R } ^ { r \times d } , \qquad \ell _ { k } \in \mathcal { A } _ { i } .
$$

Constructing the query-conditioned basis. While centering reduces template bias, attribute collision remains unresolved: selected prompt directions can still be redundant or unevenly scaled, causing correlated attributes to over-count the same cue. We therefore orthonormalize these directions into a controlled basis, so that projection-based comparisons are less sensitive to redundancy and each axis contributes independently. This is done via thin QR decomposition as follows:

$$
\mathbf { U } _ { i } ^ { \top } = \mathbf { Q } _ { i } \mathbf { R } _ { i } , \qquad \mathbf { Q } _ { i } \in \mathbb { R } ^ { d \times m } , \quad \mathbf { Q } _ { i } ^ { \top } \mathbf { Q } _ { i } = I _ { m } , \quad m \leq r .
$$

We refer to $\mathbf { Q } _ { i }$ as the Attribute-Conditioned Subspace (ACS) basis for query i.

Projection operators. $\mathbf { Q } _ { i }$ can now be decomposed into two sets of projection operators, one component parallel to the direction vectors and the other one perpendicular. We define these subspace projectors as: $\Pi _ { i } ^ { \parallel } \ = \ \mathbf { Q } _ { i } \mathbf { Q } _ { i } ^ { \top }$ and $\Pi _ { i } ^ { \perp } \ =$ $I - \Pi _ { i } ^ { \parallel }$ . We only need the subspace parallel projections and therefore, the projected features can be written as: $\mathbf { z } _ { i } ^ { \parallel } =$ $\pmb { \Pi } _ { i } ^ { \parallel } \mathbf { z } _ { i }$ for the query and, $\mathbf { z } _ { g , j } ^ { \parallel } = \Pi _ { i } ^ { \parallel } \mathbf { z } _ { g , j }$ for the gallery. Note that we will use the $\ell _ { 2 }$ -norm forms of the projected features frequently from this point onwards, and we denote them as:

Algorithm 1 Efficient ACS affinity computation   
Require: Normalized query embedding $\tilde { \mathbf { z } } _ { i } ,$ normalized gallery embed  
dings $\tilde { \mathbf { Z } } _ { g } ,$ centered prompt bank $\bar { \mathbf { Z } } _ { t } ,$ , selection size r   
Ensure: $\mathsf { A C S }$ affinity scores ${ \bf \cal A } ^ { \parallel } [ i , : ]$   
1: Query-attribute similarities: u<sub>i</sub> ← $\bar { \mathbf { Z } } _ { t } \mathbf { z } _ { i } ^ { \top }$   
2: Top-r indexing: $\mathcal { A } _ { i } \gets \mathrm { T o p R } ( \mathbf { u } _ { i } ) .$ , with $| { \mathcal { A } } _ { i } | = r$   
3: Forming direction vectors: $\mathbf { U } _ { i }  [ \bar { \mathbf { z } } _ { t , \ell _ { 1 } } \cdot \cdot \cdot \bar { \mathbf { z } } _ { t , \ell _ { r } } ] ^ { \top } , \ell _ { m } \in \mathcal { A } _ { i }$   
4: Orthogonalization: $\mathbf { U } _ { i } ^ { \top } = \mathbf { Q } _ { i } \mathbf { R } _ { i } ,$ with $\mathbf { Q } _ { i } ^ { \top } \mathbf { Q } _ { i } = \mathbf { I }$   
5: Compute ACS projections: $\mathbf { c } _ { i } \gets \mathbf { z } _ { i } ^ { \top } \mathbf { Q } _ { i }$ and $\mathbf { C } _ { g } \gets \mathbf { Z } _ { g } \mathbf { Q } _ { i }$   
6: Normalize projections: $\tilde { \mathbf { c } } _ { i } \gets \frac { \mathbf { c } _ { i } } { \lVert \mathbf { c } _ { i } \rVert _ { 2 } + \epsilon }$ and $\tilde { \mathbf { C } } _ { g } \gets \mathrm { r o w - n o r m } ( \mathbf { C } _ { g } )$   
7: Compute ACS affinity: $\mathbf { A } ^ { | | } [ i , : ] \longleftarrow \tilde { \mathbf { c } } _ { i } \tilde { \mathbf { C } } _ { g } ^ { \top }$   
8: return $\mathbf { A } ^ { \parallel } [ i , : ]$

$\tilde { \textbf { z } } = \frac { \textbf { z } } { | | \mathbf { z } | | _ { 2 } + \epsilon }$ . This makes computation of cosine similarities much easier by using only an inner product.

Implementation Note: We never explicitly form $\boldsymbol { \Pi } _ { i } ^ { \parallel } \in$ $\mathbb { R } ^ { d \times d }$ . Given $\mathbf { Q } _ { i } ~ \in ~ \mathbb { R } ^ { d \times k _ { i } }$ with orthonormal columns, projection is applied as: $\pmb { \Pi } _ { i } ^ { \parallel } \mathbf { z } = \mathbf { Q } _ { i } ( \mathbf { Q } _ { i } ^ { \top } \mathbf { z } )$ and $\Pi _ { i } ^ { \bot } \mathbf { z } =$ $\mathbf { z } - \mathbf { Q } _ { i } ( \mathbf { Q } _ { i } ^ { \top } \mathbf { z } )$ , which costs $\mathcal { O } ( d k _ { i } )$ rather than $\mathcal { O } ( d ^ { 2 } ) . ^ { 2 }$ <sup>2</sup> The full process is summarized in Algorithm 1.

ACS score computation. We now define the subspace (attribute) affinity as a measure of alignment towards the specific set of direction vectors:

$$
\mathbf { A } ^ { \parallel } [ i , j ] = ( \tilde { \mathbf { z } } _ { i } ^ { \parallel } ) ^ { \top } \tilde { \mathbf { z } } _ { g , j } ^ { \parallel } .\tag{5}
$$

In practice, this can be computed simply using $\begin{array} { r l } { \mathbf { c } _ { i } } & { { } = } \end{array}$ ${ \bf z } _ { i } ^ { \top } { \bf Q } _ { i }$ and ${ \bf c } _ { g , j } = { \bf z } _ { g , j } ^ { \top } { \bf Q } _ { i } , \ell _ { 2 }$ -normalizing them, and then taking a dot product. We then use a single scalar $\alpha _ { \mathrm { a c s } } \geq 0$ to blend the identity affinity with the attribute-subspace affinity. The ACS score is therefore derived as a linear combination of both affinities. The final ACS affinity used for computing inference time loss is:

$$
\mathbf { A } [ i , j ] = \mathbf { A } _ { \mathrm { i d } } [ i , j ] + \alpha _ { \mathrm { a c s } } \mathbf { A } ^ { \parallel } [ i , j ] .\tag{6}
$$

## 4.3. Test-Time Adaptation Objective

Retrieval distribution. Given logits $\mathbf { A } \in \mathbb { R } ^ { B \times N _ { g } }$ , we define a retrieval distribution per query: $\mathbf { p } _ { i } = \mathrm { s o f t m a x } ( \mathbf { A } [ i ,$ $] / \tau )$ , where $\tau > 0$ is a temperature value that controls the shape of the softmax distribution. Gallery and prompt-bank embeddings are kept fixed as source-space anchors and only the query-side encoder path is updated.

Top-k candidate-set likelihood. Let $\mathcal { K } _ { i } = \mathrm { T o p K } ( \mathbf { p } _ { i } )$ denote indices of top-k probabilities. We minimize a top-k negative log-likelihood:

$$
\mathcal { L } _ { \mathrm { T T A } } = - \frac { 1 } { B } \sum _ { i = 1 } ^ { B } \sum _ { j \in \mathcal { K } _ { i } } \log \mathbf { p } _ { i } ( j ) .\tag{7}
$$

<sup>2</sup>Equivalence: Since $\begin{array} { l } { \mathbf { Q } _ { i } ^ { \top } \mathbf { Q } _ { i } } \end{array} = \ I _ { k _ { i } } ,$ for any vectors ${ \mathbf z } _ { a } , { \mathbf z } _ { b } \in \left( \begin{array} { l l } { \quad } \end{array} \right)$ $\begin{array} { r } { \mathbb { R } ^ { d } \colon \ ( \pmb { \Pi } _ { i } ^ { \parallel } \mathbf { z } _ { a } ) ^ { \top } ( \pmb { \Pi } _ { i } ^ { \parallel } \mathbf { z } _ { b } ) \ = \ ( \mathbf { Q } _ { i } ^ { \top } \mathbf { z } _ { a } ) ^ { \top } ( \mathbf { Q } _ { i } ^ { \top } \mathbf { z } _ { b } ) } \end{array}$ such that $\| \boldsymbol { \Pi } _ { i } ^ { \parallel } \mathbf { z } _ { a } \| _ { 2 } ~ =$ $\| \mathbf { Q } _ { i } ^ { \top } \mathbf { z } _ { a } \| _ { 2 }$

![](images/e3ea506aac4f754611dd495897cff2aa6d4dfaab0ede6de68248c496cbd76853.jpg)  
Figure 3. Method overview. PRAISE uses attribute prompts to construct a query-specific semantic subspace that stabilizes query-side adaptation under domain shift. At test time, it performs query-side episodic TTA with a fixed source-space gallery index. Gallery and prompt-bank embeddings are computed once and kept fixed; gradients update only the query-side normalization affine parameters.

This encourages the retrieval distribution to become concentrated on a small candidate set, which empirically stabilizes adaptation under large galleries. Please refer to the Supplementary Materials for a theoretically grounded motivation for the loss function.

Adaptation settings. We adapt only normalization parameters (affine scale and bias) within $\mathcal { F } _ { v } \colon$ θ = {norm affine parameters in $\mathcal { F } _ { v } \big \}$ , keeping all other weights frozen. This preserves pretrained semantics and reduces overfitting at test time. We adopt an episodic TTA, i.e., for each test batch, we reset $\theta  \theta _ { 0 }$ (pretrained), perform $T$ gradient steps on ${ \mathcal { L } } ,$ and output the adapted embeddings for evaluation. Note that in case of Text-to-Image ReID, only the adaptation parameters become $\theta _ { t } =$ {norm affine parameters in $\mathcal { F } _ { t } \}$ . The rest of the pipeline follows as is, with the same logic.

## 5. Results and Discussion

## 5.1. Experimental Protocol

We evaluate PRAISE in two different settings: Image-to-Image (I2I) ReID and Text-to-Image (T2I) ReID. We use MSMT17 [48] and Market-1501 [55] for I2I, and CUHK-PEDES [25] and RSTP-ReID [56] for T2I. At test time, all methods are given (i) the query (image or text) and (ii) the unlabeled target gallery. We pre-encode all gallery features once and reuse them for all methods for efficiency. Furthermore, we ensure that all methods are granted the same access to unlabeled gallery features under the same episodic/reset protocol as our approach. All runtimes are measured on an NVIDIA RTX A6000 GPU (single GPU) with identical implementation settings (batch size, mixed precision), and exclude the one-time cost of gallery feature pre-encoding. Adam optimizer has been used throughout, for all experiments.

Table 1. Effect on different backbones. Backbones evaluated on I2IReID and T2IReID datasets with PRAISE applied (score on left side of $" \cdot \cdot \cdot \cdot$ is without adaptation and score on right is after applying PRAISE). Each dataset reports mAP and Rank-1 Accuracy (R1).
<table><tr><td rowspan="3">Backbone</td><td colspan="6">I2IReID</td><td colspan="4">T2IReID</td></tr><tr><td colspan="2">MSMT17 mAP</td><td rowspan="2"></td><td colspan="2">Market-1501</td><td colspan="2">CUHK-Pedes</td><td colspan="2"></td></tr><tr><td></td><td>R1</td><td>mAP</td><td>R1</td><td>mAP</td><td>R1</td><td>mAP</td><td>RSTPReID R1</td></tr><tr><td rowspan="2">CLIP ViT-B/16 [37] DINO.txt [15]</td><td>1.1/</td><td>1.4</td><td>5.6/ 6.7</td><td>3.5/ 4.3</td><td>11.9/14.0</td><td>7.1/ 9.1</td><td>8.9/11.7</td><td></td><td>8.1/ 9.2</td><td>11.2/12.9</td></tr><tr><td>0.7/ 1.8</td><td></td><td>5.2/ 9.4</td><td>1.9/ 3.0</td><td>7.1/10.1</td><td>7.3/</td><td>8.8</td><td>9.1/12.8</td><td>10.3/13.1</td><td>11.9/14.6</td></tr><tr><td rowspan="2">CLIPReID [27] RDE [35]</td><td>20.8/24.7</td><td></td><td>45.9/52.8</td><td>37.8/42.1</td><td>63.3/69.4</td><td>8.1/ 9.9</td><td></td><td>8.2/11.4</td><td>12.2/13.8</td><td>13.5/15.1</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>33.5/38.1</td><td></td><td>36.8/42.8</td><td>41.2/46.3</td><td>54.9/60.9</td></tr><tr><td rowspan="2">ICL [36] PLIP [58]</td><td></td><td></td><td></td><td></td><td>-</td><td>26.8/30.0</td><td></td><td>41.0/47.4</td><td>37.5/42.1</td><td>62.1/66.3</td></tr><tr><td>10.5/18.2</td><td></td><td>34.2/38.5</td><td>35.0/46.5</td><td>62.6/71.1</td><td>46.4/49.6</td><td></td><td>55.9/58.4</td><td>23.1/27.6</td><td>34.5/37.6</td></tr></table>

## 5.2. Impact on different backbones

We evaluate PRAISE across different VLM backbones in Table 1. Only CLIP [37], Dino.txt [15] and PLIP [58] are taken by their off-the-shelf training weights. For the other methods, we avoid fully supervised training on the datasets we test on, in order to effectively evaluate test-time adaptation strategies. This is why, most of the scores on the table appear to be very poor.<sup>3</sup> For CLIP-ReID [27], RDE [35] and ICL [36], we test on the dataset specified in the column header, upon training on the other dataset in the same task category (I2I or T2I). Off-the-shelf CLIP [37] and Dino.txt [15] perform poorly on ReID and are clearly unsuitable for a meaningful evaluation of PRAISE. We find that PRAISE consistently improves performance for all the state-of-the-art ReID methods compared. [27, 35, 36]. PLIP [58], being a general-purpose ReID-oriented VLM, achieves strong performance across both I2I and T2I settings. While PRAISE improves each backbone, we use PLIP as the default backbone due to its consistent performance across all evaluated tasks.

Table 2. Comparison of TTA methods on I2IReID benchmarks using a PLIP [58] backbone. Each dataset reports mAP and Rank-1 Accuracy (R1).
<table><tr><td rowspan="2">TTA Method</td><td rowspan="2"> $\mathbf { T y p e }$ </td><td colspan="4">Dataset</td><td rowspan="2">Time (ms) /sample</td></tr><tr><td>MSMT17</td><td>R1</td><td>Market-1501</td><td>R1</td></tr><tr><td>No-adapt (Source) [NeurIPS&#x27;24]</td><td>一</td><td>mAP 10.5</td><td>34.2</td><td>mAP</td><td>62.6</td><td>1.8</td></tr><tr><td></td><td rowspan="4">Classifier logit</td><td></td><td></td><td>35.0</td><td></td><td>16.2</td></tr><tr><td>TENT [43] [ICLR&#x27;21] SHOT [28] [ICML&#x27;20]</td><td>9.2 9.9</td><td>29.8 30.2</td><td>29.3 28.9</td><td>52.7 56.8</td><td>63.8</td></tr><tr><td>SAR [30] [ICLR&#x27;23]</td><td>10.0</td><td>31.1</td><td>33.2</td><td>63.3</td><td>91.4</td></tr><tr><td>ETA [29] [ICML/22]</td><td>9.8</td><td>30.8</td><td>32.2</td><td>58.9</td><td>47.1</td></tr><tr><td>DeYO [21] [ICLR&#x27;24]</td><td></td><td>10.8</td><td>34.9</td><td>34.2</td><td>64.0</td><td>51.2</td></tr><tr><td>TPT [38] [NeurIPS&#x27;22]</td><td>VLM</td><td>11.7</td><td>35.2</td><td>35.9</td><td>63.2</td><td>1189.1</td></tr><tr><td>TDA [18] [CVPR·24]</td><td>based</td><td>10.7</td><td>34.2</td><td>35.6</td><td>63.1</td><td>3.4</td></tr><tr><td>TEMP [1] [ICNN&#x27;24]</td><td>ReID /</td><td>11.8</td><td>36.1</td><td></td><td></td><td>20.1</td></tr><tr><td>Anchor-ReID [4] [SCIA&#x27;23]</td><td rowspan="3">retrieval</td><td>7.9</td><td>24.3</td><td>37.1 31.2</td><td>65.1 45.1</td><td>6.8</td></tr><tr><td>Norm [40] [ICLR&#x27;25]</td><td>14.8</td><td>40.2</td><td>41.0</td><td>67.3</td><td>2.2</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Direct Score Fusion (λatt=0.1) PRAISE (Ours)</td><td></td><td>12.8</td><td>33.0</td><td>39.2</td><td>60.3</td><td>38.1</td></tr><tr><td></td><td></td><td>18.2</td><td>38.5</td><td>46.5</td><td>71.1</td><td>42.8</td></tr><tr><td>PRAISE + Norm</td><td></td><td>19.5</td><td>40.4</td><td>48.1</td><td>72.5</td><td>43.1</td></tr></table>

## 5.3. Performance on Re-identification

How does PRAISE perform on I2I-ReID? A consistent trend is that classifier-oriented TTA methods often degrade performance compared to no-adapt (e.g., TENT/SHOT/SAR/ETA) [28–30, 43], with only minor gains from DeYO [21]. VLM-based TTA methods [18, 38] are more stable, while the strongest baselines are ReID-specific methods such as TEMP [1] and Norm [40]. We find that direct score fusion has some limited semantic improvement but the poor calibration leads to a poorer Rank-1. PRAISE substantially improves mAP over all baselines and also yields consistent Rank-1 gains. Although TDA and Norm are faster because they are backpropagation-free, PRAISE provides a strong runtime-accuracy tradeoff. Moreover, since Norm is post-hoc, it can be applied on top of PRAISE with only a 0.3ms/sample overhead, further improving performance.

How does PRAISE perform on T2I-ReID? A similar trend holds for T2I-ReID: classifier-logit methods such as TENT and SHOT [28, 43] give only marginal gains, while VLM classification-based methods such as TPT, TDA, WATT, CLI-PArTT and CLIPTTA [8, 18, 20, 31, 38] are often unstable and can degrade retrieval. TCR [23] is the strongest retrievalspecific baseline, but PRAISE achieves the best results on both datasets, improving CUHK-Pedes from 47.6/57.0 to 49.6/58.4 and RSTPReID from 25.8/36.6 to 27.6/37.6 in mAP/R1. Although slower than no-adapt and TCR, PRAISE remains much faster than augmentation-heavy methods such as TPT and WATT, while providing stronger retrieval accuracy under incomplete or ambiguous text queries. Similar to I2I-ReID, we find direct score fusion has a positive effect on the overall retrieval but leads to the ranking accuracy going down due to its improper calibration.

Note. We find that our prompt bank construction strategy is transferable to other datasets as well, with minimal performance changes when a differently sourced prompt bank is used. Furthermore, we find that the traditional test time adaptation methods are more suited to operating under corruption and image-noise based shifts where entropy based sharpening improves the retrieval geometry. Due to space restrictions in the main manuscript, a detailed summary with results can befound in the Supplementary Materials.

Table 3. Comparison of TTA methods on T2IReID benchmarks on a PLIP [58] backbone. Each dataset reports mAP and Rank-1 Accuracy (R1).
<table><tr><td rowspan="2">TTA Method</td><td rowspan="2"> $\mathbf { T y p e }$ </td><td colspan="4">Dataset</td><td rowspan="2">Time (ms) /sample</td></tr><tr><td>CUHK-Pedes mAP</td><td>R1</td><td>RSTPReID mAP</td><td>R1</td></tr><tr><td>No-adapt (Source) [NeurIPS&#x27;24]</td><td>一</td><td>46.4</td><td>55.9</td><td>23.1</td><td>34.5</td><td>12.9</td></tr><tr><td>TENT [43] [ICLR&#x27;21]</td><td>一 Classifier</td><td>46.4</td><td>56.2</td><td>23.3</td><td>34.2</td><td>81.8</td></tr><tr><td>SHOT [28] [ICML/20]</td><td>logit</td><td>47.1</td><td>56.6</td><td>24.2</td><td>34.0</td><td>179.1</td></tr><tr><td>TPT [38] [NeurIPS&#x27;22]</td><td rowspan="6">VLM</td><td>46.5</td><td>55.9</td><td>23.1</td><td>34.6</td><td>4193.4</td></tr><tr><td>TDA [18] [CVPR·24]</td><td>39.2</td><td>48.9</td><td>16.3</td><td>26.8</td><td>29.8</td></tr><tr><td>WATT (S) [31] [NeurIPS&#x27;24]</td><td>39.6</td><td>36.8</td><td>20.0</td><td>27.5</td><td>1339.3</td></tr><tr><td>WATT (P) [31] [NeurIPS&#x27;24]</td><td>40.1</td><td>47.6</td><td>20.2</td><td>7.8</td><td>1398.8</td></tr><tr><td>CLIPArTT [8] [WACV&quot;25]</td><td>37.5</td><td>44.8</td><td>20.0</td><td>27.9</td><td>605.2</td></tr><tr><td>CLIPTTA [20] [NeurIPS&#x27;25]</td><td>41.4</td><td>49.4</td><td>22.1</td><td>31.7</td><td>592.8</td></tr><tr><td>TCR [23] [ICLR°25]</td><td>Retrieval</td><td>47.6</td><td>57.0</td><td>25.8</td><td>36.6</td><td>120.8</td></tr><tr><td>Direct Score Fusion (λatt=0.1)</td><td></td><td>48.8</td><td>52.1</td><td>24.5</td><td>32.9</td><td>132.4</td></tr><tr><td>PRAISE (Ours)</td><td></td><td>49.6</td><td>58.4</td><td>27.6</td><td>37.6</td><td>160.3</td></tr></table>

## 5.4. Hyperparameter Sensitivity

This section is aimed at measuring how sensitive the performance of PRAISE is when the various hyperparameters are varied. While the sensitivity plots identify the bestperforming values on Market-1501, all experiments in the manuscript were run using the same set of hyperparameters to show that PRAISE is sufficiently robust to a hyperparameter set which is not explicitly optimized for a specific dataset.

Effect of $\alpha _ { \mathrm { a c s } }$ (Figure 4a): $\alpha _ { \mathrm { a c s } } { = } 0 . 1$ gives the best tradeoff, achieving peak Rank-1 and near-peak mAP. Larger values gradually degrade performance, suggesting that overly strong ACS weighting can over-constrain the embedding with noisy or partial attribute cues and harm fine-grained identity separation.

Effect of TopR (Figure 4b): Performance peaks around R=5, which matches the number of main attribute categories used in our prompt bank. Smaller R under-represents semantic diversity, while larger R introduces redundant or noisy directions that dilute the attribute-conditioned subspace.

Effect of Top-k (Figure 4c): Performance improves quickly from very small k and then saturates, with the best Rank-1 around k=50. This reflects a selection trade-off: too small k gives unstable pseudo-neighborhoods, whereas too large k admits more false positives and weakens discrimination.

Effect of optimization steps (Figure 4d): Performance improves rapidly during the first test-time updates and then saturates after roughly 10 steps. Since runtime grows nearly linearly with the number of steps, additional updates provide limited accuracy gains relative to their computational cost.

Number of attributes vs TopR (Figure 4e): We find that accuracy generally improves with more attribute categories and moderately larger TopR, but the interaction between the two is not strictly monotonic. This implies that diverse attribute categories provide complementary directions, while overly large TopR can add redundant prompts once sufficient semantic coverage is available.

![](images/1647461c074753d40f2965942414c416f5fe76e993968d3b7b0fd950c47968b7.jpg)  
(a) α<sub>acs</sub>

![](images/2e56eb96c21d88484345dd0173b6ea6d17ac64e1d2c10f2b162d83991b588d1e.jpg)  
(b) TopR

![](images/7b520286adcb9f71fb240e902d75da8c063110e9abb858e794a1c8337f0896ba.jpg)  
(c) Top-k

![](images/65369b3ec2b0aace535bb2ab34212de8c37e4191caa1077bccdaea574c7432f4.jpg)  
(d) Steps per batch

![](images/2c6a8ae765cd487828f1bcef41273a0bccc92b633cae5ee8b0073f1f44973154.jpg)  
(e) TopR vs number of attribute categories.  
Figure 4. Hyperparameter Sensitivity. Performance with respect to varying hyperparameters. Tested on Market-1501.

## 5.5. Ablation Study

Table 4 isolates the contributions of ACS, query conditioning, orthogonalization, score composition, and the adaptation objective. We also compare ACS with fixed low-rank PCA projections and stronger calibrated score-fusion baselines.

Component Ablations. Using ACS only as a frozen score gives little improvement over the base model, while adaptation without ACS provides only modest gains. In contrast, combining query-side adaptation with ACS improves Market-1501 from 35.0/62.6 to 46.5/71.1 mAP/R1 and RST-PReID from 23.1/34.5 to 27.6/37.6. Thus, ACS is not effective as a standalone re-ranking signal; its main benefit arises from guiding the adaptation process with query-conditioned semantic geometry.

Subspace and Score-Fusion Ablations. Shuffling prompts, using a constant ACS, or replacing ACS with a random subspace substantially degrades performance, confirming the importance of both semantic grounding and query conditioning. Removing QR while keeping the same selected directions also reduces performance to 44.2/68.1 on Market-1501 and 26.4/36.8 on RSTPReID, showing that orthogonalization is beneficial. Motivated by a denoising interpretation, we further compare matched-rank PCA subspaces from the prompt bank and gallery features. Both are competitive in mAP, reaching 45.3/63.0 and 45.2/62.9 on Market-1501, but remain below ACS, especially in Rank-1 accuracy, with the same ordering on RSTPReID. Thus, low-rank projection itself is useful, while query-conditioned semantic selection better resolves top-ranked matches. Stronger scorefusion baselines show a similar trend. Standardized and temperature-calibrated fusion improve over direct fusion, but the best variant reaches only 41.5/61.8 and 25.1/33.9, remaining clearly below PRAISE.

Adaptation Objective. We compare Eq. (7) with several alternatives, including top-k entropy minimization<sup>4</sup>. Top-1 NLL and full-gallery entropy reach 42.7/58.2 and 41.1/61.0 on Market-1501, while top-k entropy improves to 45.3/63.1. The proposed set likelihood remains best at 46.5/71.1, with

Table 4. Ablations. All rows use the same TopR, rank r, number of adaptation steps, and $\alpha _ { \mathrm { a c s } } ,$ unless stated otherwise. I2I-ReID results are reported on Market-1501 and T2I-ReID results on RSTPReID.

<table><tr><td rowspan="2">Group</td><td rowspan="2">Variant</td><td colspan="2">I2I-ReID</td><td colspan="2">T2I-ReID</td></tr><tr><td>mAP</td><td>R1</td><td>mAP</td><td>R1</td></tr><tr><td rowspan="4">Component Ablation</td><td>No adapt</td><td>35.0</td><td>62.6</td><td>23.1</td><td>34.5</td></tr><tr><td>ACS only (No-adaptation, αacs=0.1)</td><td>35.2</td><td>61.8</td><td>23.6</td><td>34.1</td></tr><tr><td>Adaptation w/o ACS (αacs=0)</td><td>36.8</td><td>65.0</td><td>23.5</td><td>35.4</td></tr><tr><td>ACS adaptation (i.e. PRAISE)</td><td>46.5</td><td>71.1</td><td>27.6</td><td>37.6</td></tr><tr><td rowspan="6">Subspace Ablation</td><td>Shuffled prompts across queries</td><td>37.8</td><td>62.8</td><td>23.5</td><td>35.3</td></tr><tr><td>Constant ACS subspace</td><td>36.7</td><td>62.6</td><td>23.5</td><td>35.1</td></tr><tr><td>Random subspace</td><td>35.1</td><td>62.9</td><td>24.2</td><td>34.8</td></tr><tr><td>ACS w/o QR orthogonalization</td><td>44.2</td><td>68.1</td><td>26.4</td><td>36.8</td></tr><tr><td>Global prompt PCA (rank r=5)</td><td>45.3</td><td>63.0</td><td>27.0</td><td>36.9</td></tr><tr><td>Gallery feature PCA (rank r=5)</td><td>45.2</td><td>62.9</td><td>26.8</td><td>36.8</td></tr><tr><td rowspan="3">Score Fusion Control</td><td>Direct score fusion</td><td>39.2</td><td>60.3</td><td>24.5</td><td>32.9</td></tr><tr><td>Standardized (z-score) fusion</td><td>41.4</td><td>61.1</td><td>24.8</td><td>33.1</td></tr><tr><td>Temperature-calibrated probability fusion</td><td>41.5</td><td>61.8</td><td>25.1</td><td>33.9</td></tr><tr><td rowspan="5">Adaptation Objective</td><td>Top-1 pseudo-label NLL</td><td>42.7</td><td>58.2</td><td>24.8</td><td>34.6</td></tr><tr><td>Full retrieval entropy (no top-k)</td><td>41.1</td><td>61.0</td><td>25.1</td><td>34.7</td></tr><tr><td>Top-k entropy</td><td>45.3</td><td>63.1</td><td>26.8</td><td>35.9</td></tr><tr><td>Raw top-k similarity (w/o softmax)</td><td>6.2</td><td>11.5</td><td>4.1</td><td>9.3</td></tr><tr><td>Proposed top-k set likelihood (Eq. 7)</td><td>46.5</td><td>71.1</td><td>27.6</td><td>37.6</td></tr><tr><td rowspan="4">Template Bias Control</td><td>Template Corruption, Attributes Preserved</td><td>46.1</td><td>69.7</td><td>27.3</td><td>37.2</td></tr><tr><td>Attribute tokens only</td><td>44.2</td><td>69.1</td><td>26.7</td><td>37.8</td></tr><tr><td>w/o Subtraction</td><td>40.7</td><td>62.8</td><td>24.3</td><td>36.6</td></tr><tr><td>w/o Subtraction + Template Corruption</td><td>40.5</td><td>62.7</td><td>24.4</td><td>36.6</td></tr></table>

the same trend on RSTPReID (27.6/37.6 versus 26.8/35.9). Unlike top-k entropy, Eq. (7) promotes the selected neighborhood collectively rather than explicitly sharpening probabilities within it, avoiding premature commitment to individua candidates. Raw top-k similarity without softmax performs poorly, showing the importance of gallery-relative normalization.

Template Bias Control. Corrupting the shared template while preserving attribute content leaves performance nearly unchanged, showing limited sensitivity to exact phrasing. Attribute tokens alone cause a moderate drop, while removing neutral-feature subtraction reduces Market-1501 mAP from 46.5 to 40.7. Additional template corruption without subtraction changes little, indicating that centering is the main mechanism suppressing shared template bias.

## 6. Conclusion

Limitations. PRAISE uses a fixed standardized attribute bank, so its effectiveness depends on the coverage of the prompt vocabulary and may be limited for rare or highly fine-grained appearance cues. Furthermore, when the initial VLM representation is very weak or the query is severely occluded/noisy, the selected attribute directions may be unreliable, limiting the benefit of ACS. This has been analyzed and reported in the Supplementary Materials.

We introduced PRAISE, a label-free TTA framework for VLM-based ReID across I2I and T2I retrieval. Our results show that attribute prompts are unreliable as direct matching scores, but become effective when used as query-conditioned directions for guiding retrieval adaptation. This provides a lightweight way to inject attribute consistency at test time without identity or attribute labels. Future work will explore query-adaptive prompt banks to improve coverage beyond a fixed bank of pedestrian attributes.

## References

[1] Kazuki Adachi, Shohei Enomoto, Taku Sasaki, and Shin’Ya Yamaguchi. Test-time similarity modification for person reidentification toward temporal distribution shift. In 2024 International Joint Conference on Neural Networks (IJCNN), pages 1–8. IEEE, 2024. 2, 7, 19, 20

[2] Surbhi Aggarwal, Venkatesh Babu Radhakrishnan, and Anirban Chakraborty. Text-based person search via attribute-aided matching. In Proceedings ofthe IEEE/CVF winter conference on applications ofcomputer vision, pages 2617–2625, 2020. 3

[3] James Urquhart Allingham, Jie Ren, Michael W Dusenberry, Xiuye Gu, Yin Cui, Dustin Tran, Jeremiah Zhe Liu, and Balaji Lakshminarayanan. A simple zero-shot prompt weighting technique to improve prompt ensembling in text-image models. In International Conference on Machine Learning, pages 547–568. PMLR, 2023. 4

[4] Mohammed Almansoori, Mustansar Fiaz, and Hisham Cholakkal. Anchor-reid: A test time adaptation for person re-identification. In Scandinavian Conference on Image Analysis, pages 599–612. Springer, 2023. 7, 19

[5] Yang Bai, Min Cao, Daming Gao, Ziqiang Cao, Chen Chen, Zhenfeng Fan, Liqiang Nie, and Min Zhang. Rasa: Relation and sensitivity aware representation learning for text-based person search. arXiv preprint arXiv:2305.13653, 2023. 3

[6] Chanho Eom, Geon Lee, Kyunghwan Cho, Hyeonseok Jung, Moonsub Jin, and Bumsub Ham. Cerberus: Attribute-based person re-identification using semantic ids. Expert Systems with Applications, 259:125320, 2025. 2

[7] Xinqian Gu, Hong Chang, Bingpeng Ma, Shutao Bai, Shiguang Shan, and Xilin Chen. Clothes-changing person re-identification with rgb modality only. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 1060–1069, 2022. 2

[8] Gustavo Adolfo Vargas Hakim, David Osowiechi, Mehrdad Noori, Milad Cheraghalikhani, Ali Bahri, Moslem Yazdanpanah, Ismail Ben Ayed, and Christian Desrosiers. Clipartt: Light-weight adaptation of clip to new domains at test time. arXiv preprint arXiv:2405.00754, 2024. 2, 7, 21

[9] Albert Haque, Alexandre Alahi, and Li Fei-Fei. Recurrent attention models for depth-based person identification. In Proceedings ofthe IEEE conference on computer vision and pattern recognition, pages 1229–1238, 2016. 2

[10] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 770–778, 2016. 3

[11] Dan Hendrycks and Thomas Dietterich. Benchmarking neural network robustness to common corruptions and perturbations. arXiv preprint arXiv:1903.12261, 2019. 18

[12] Yan Huang, Qiang Wu, JingSong Xu, Yi Zhong, and ZhaoXiang Zhang. Clothing status awareness for long-term person re-identification. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 11895–11904, 2021. 2

[13] Yan Huang, Zhang Zhang, Qiang Wu, Yi Zhong, and Liang Wang. Attribute-guided pedestrian retrieval: Bridging person re-id with internal attribute variability. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 17689–17699, 2024. 2

[14] Ding Jiang and Mang Ye. Cross-modal implicit relation reasoning and aligning for text-to-image person retrieval. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 2787–2797, 2023. 3

[15] Cijo Jose, Théo Moutakanni, Dahyun Kang, Federico Baldassarre, Timothée Darcet, Hu Xu, Daniel Li, Marc Szafraniec, Michaël Ramamonjisoa, Maxime Oquab, et al. Dinov2 meets text: A unified framework for image-and pixel-level visionlanguage alignment. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 24905–24916, 2025. 2, 6

[16] Sehun Jung and Hyang-won Lee. Learning generalizable prompt for clip with class similarity knowledge. arXiv preprint arXiv:2502.11969, 2025. 4

[17] Nikolaos Karianakis, Zicheng Liu, Yinpeng Chen, and Stefano Soatto. Reinforced temporal attention and split-rate transfer for depth-based person re-identification. In Proceedings of the European conference on computer vision (ECCV), pages 715–733, 2018. 2

[18] Adilbek Karmanov, Dayan Guan, Shijian Lu, Abdulmotaleb El Saddik, and Eric Xing. Efficient test-time adaptation of vision-language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14162–14171, 2024. 2, 7, 19, 20, 21

[19] Vladimir V Kniaz, Vladimir A Knyaz, Jirí Hladuvka, Walter G Kropatsch, and Vladimir Mizginov. Thermalgan: Multimodal color-to-thermal image translation for person reidentification in multispectral dataset. In Proceedings ofthe European Conference on Computer Vision (ECCV), pages 0–0, 2018. 2

[20] Marc Lafon, Gustavo Adolfo Vargas Hakim, Clément Rambour, Christian Desrosier, and Nicolas Thome. Cliptta: Robust contrastive vision-language test-time adaptation. arXiv preprint arXiv:2507.14312, 2025. 7, 18, 21

[21] Jonghyun Lee, Dahuin Jung, Saehyung Lee, Junsung Park, Juhyeon Shin, Uiwon Hwang, and Sungroh Yoon. Entropy is not enough for test-time adaptation: From the perspective of disentangled factors. In The Twelfth International Conference on Learning Representations, 2024. 7, 19, 20

[22] Youngjun Lee, Doyoung Kim, Junhyeok Kang, Jihwan Bang, Hwanjun Song, and Jae-Gil Lee. Ra-tta: Retrieval-augmented test-time adaptation for vision-language models. In The Thirteenth International Conference on Learning Representations, 2025. 3

[23] Haobin Li, Peng Hu, Qianjun Zhang, Xi Peng, Xiting Liu, and Mouxing Yang. Test-time adaptation for cross-modal retrieval with query shift. arXiv preprint arXiv:2410.15624, 2024. 7, 21

[24] Junjie Li, Guanshuo Wang, Fufu Yu, Yichao Yan, Qiong Jia, Shouhong Ding, Xingdong Sheng, Yunhui Liu, and Xiaokang Yang. Rethinking clothes changing person reid: Conflicts, synthesis, and optimization. arXiv preprint arXiv:2404.12611, 2024. 2

[25] Shuang Li, Tong Xiao, Hongsheng Li, Bolei Zhou, Dayu Yue, and Xiaogang Wang. Person search with natural language description. arXiv preprint arXiv:1702.05729, 2017. 6

[26] Shihua Li, Haobin Chen, Shijie Yu, Zhiqun He, Feng Zhu, Rui Zhao, Jie Chen, and Yu Qiao. Cocas+: Large-scale clothes-changing person re-identification with clothes templates. IEEE Transactions on Circuits and Systems for Video Technology, 33(4):1839–1853, 2022. 2

[27] Siyuan Li, Li Sun, and Qingli Li. Clip-reid: exploiting visionlanguage model for image re-identification without concrete text labels. In Proceedings ofthe AAAI Conference on Artificial Intelligence, pages 1405–1413, 2023. 1, 2, 6, 20

[28] Jian Liang, Dapeng Hu, and Jiashi Feng. Do we really need to access the source data? source hypothesis transfer for unsupervised domain adaptation. In Proceedings ofthe 37th International Conference on Machine Learning. JMLR.org, 2020. 7, 19, 20, 21

[29] Shuaicheng Niu, Jiaxiang Wu, Yifan Zhang, Yaofo Chen, Shijian Zheng, Peilin Zhao, and Mingkui Tan. Efficient testtime model adaptation without forgetting. In International conference on machine learning, pages 16888–16905. PMLR, 2022. 7, 13, 19, 20

[30] Shuaicheng Niu, Jiaxiang Wu, Yifan Zhang, Zhiquan Wen, Yaofo Chen, Peilin Zhao, and Mingkui Tan. Towards stable test-time adaptation in dynamic wild world. In Internetional Conference on Learning Representations, 2023. 7, 19, 20

[31] David Osowiechi, Mehrdad Noori, Gustavo Vargas Hakim, Moslem Yazdanpanah, Ali Bahri, Milad Cheraghalikhani, Sahar Dastani, Farzad Beizaee, Ismail Ayed, and Christian Desrosiers. Watt: Weight average test time adaptation of clip. Advances in neural information processing systems, 37: 48015–48044, 2024. 2, 7, 21

[32] Sarah Parisot, Yongxin Yang, and Steven McDonagh. Learning to name classes for vision and language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 23477–23486, 2023. 4

[33] Hyunjong Park, Sanghoon Lee, Junghyup Lee, and Bumsub Ham. Learning by aligning: Visible-infrared person reidentification using cross-modal correspondences. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 12046–12055, 2021. 2

[34] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. Pytorch: An imperative style, high-performance deep learning library. Advances in neural information processing systems, 32, 2019. 16

[35] Yang Qin, Yingke Chen, Dezhong Peng, Xi Peng, Joey Tianyi Zhou, and Peng Hu. Noisy-correspondence learning for

text-to-image person re-identification. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 27197–27206, 2024. 1, 3, 6, 20, 21

[36] Yang Qin, Chao Chen, Zhihang Fu, Dezhong Peng, Xi Peng, and Peng Hu. Human-centered interactive learning via mllms for text-to-image person re-identification. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 14390–14399, 2025. 3, 6

[37] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021. 3, 6

[38] Manli Shu, Weili Nie, De-An Huang, Zhiding Yu, Tom Goldstein, Anima Anandkumar, and Chaowei Xiao. Test-time prompt tuning for zero-shot generalization in vision-language models. Advances in Neural Information Processing Systems, 35:14274–14289, 2022. 2, 7, 13, 18, 19, 20, 21

[39] Jianlou Si, Honggang Zhang, Chun-Guang Li, Jason Kuen, Xiangfei Kong, Alex C Kot, and Gang Wang. Dual attention matching network for context-aware feature sequence based person re-identification. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 5363–5372, 2018. 2

[40] Myungseo Song, Jin-Woo Park, and Jong-Seok Lee. Exploring the camera bias of person re-identification. arXiv preprint arXiv:2502.10195, 2025. 1, 2, 7, 19, 20, 22, 23

[41] Wentan Tan, Changxing Ding, Jiayu Jiang, Fei Wang, Yibing Zhan, and Dapeng Tao. Harnessing the power of mllms for transferable text-to-image person reid. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 17127–17137, 2024. 3

[42] Xinyu Tian, Shu Zou, Zhaoyuan Yang, and Jing Zhang. Argue: Attribute-guided prompt tuning for vision-language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 28578–28587, 2024. 4

[43] Dequan Wang, Evan Shelhamer, Shaoteng Liu, Bruno Olshausen, and Trevor Darrell. Tent: Fully test-time adaptation by entropy minimization. arXiv preprint arXiv:2006.10726, 2020. 2, 7, 13, 18, 19, 20, 21

[44] Guan’an Wang, Tianzhu Zhang, Jian Cheng, Si Liu, Yang Yang, and Zengguang Hou. Rgb-infrared cross-modality person re-identification via joint pixel and feature alignment. In The IEEE International Conference on Computer Vision (ICCV), 2019. 2

[45] Xiao Wang, Qian Zhu, Shujuan Wu, Bo Jiang, and Shiliang Zhang. When person re-identification meets event camera: a benchmark dataset and an attribute-guided re-identification framework. In Proceedings ofthe AAAI Conference on Artificial Intelligence, pages 10172–10180, 2026. 2

[46] Yansong Wang, Yundong Sun, Yansheng Fu, Dongjie Zhu, and Zhaoshuo Tian. Spectrum-bert: pretraining of deep bidirectional transformers for spectral classification of chinese liquors. IEEE Transactions on Instrumentation and Measurement, 73:1–13, 2024. 3

[47] Zhe Wang, Zhiyuan Fang, Jun Wang, and Yezhou Yang. Vitaa: Visual-textual attributes alignment in person search by natural

language. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XII 16, pages 402–420. Springer, 2020. 3

[48] Longhui Wei, Shiliang Zhang, Wen Gao, and Qi Tian. Person transfer gan to bridge domain gap for person re-identification. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 79–88, 2018. 6

[49] Yushuang Wu, Zizheng Yan, Xiaoguang Han, Guanbin Li, Changqing Zou, and Shuguang Cui. Lapscore: languageguided person search via color reasoning. In Proceedings of the IEEE/CVF international conference on computer vision, pages 1624–1633, 2021. 3

[50] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025. 4, 16

[51] Fan Yang, Ke Yan, Shijian Lu, Huizhu Jia, Xiaodong Xie, and Wen Gao. Attention driven person re-identification. Pattern Recognition, 86:143–155, 2019. 2

[52] Shuyu Yang, Yinan Zhou, Zhedong Zheng, Yaxiong Wang, Li Zhu, and Yujiao Wu. Towards unified text-based person retrieval: A large-scale multi-attribute and language search benchmark. In Proceedings of the 31st ACM international conference on multimedia, pages 4492–4501, 2023. 2

[53] M. Ye, X. Lan, Z. Wang, and P. C. Yuen. Bi-directional center-constrained top-ranking for visible thermal person reidentification. IEEE Transactions on Information Forensics and Security, 15:407–419, 2020. 2

[54] Mang Ye, Jianbing Shen, Gaojie Lin, Tao Xiang, Ling Shao, and Steven C. H. Hoi. Deep learning for person re-identification: A survey and outlook. arXiv preprint arXiv:2001.04193, 2020. 2

[55] Liang Zheng, Liyue Shen, Lu Tian, Shengjin Wang, Jingdong Wang, and Qi Tian. Scalable person re-identification: A benchmark. In Proceedings ofthe IEEE international conference on computer vision, pages 1116–1124, 2015. 6, 13, 24

[56] Aichun Zhu, Zijie Wang, Yifeng Li, Xili Wan, Jing Jin, Tian Wang, Fangqiang Hu, and Gang Hua. Dssl: Deep surroundings-person separation learning for text-based person retrieval. In Proceedings of the 29th ACM international conference on multimedia, pages 209–217, 2021. 3, 6, 13, 25

[57] Yuanxin Zhu, Zhao Yang, Li Wang, Sai Zhao, Xiao Hu, and Dapeng Tao. Hetero-center loss for cross-modality person re-identification. Neurocomputing, 386:97–109, 2020. 2

[58] Jialong Zuo, Jiahao Hong, Feng Zhang, Changqian Yu, Hanyu Zhou, Changxin Gao, Nong Sang, and Jingdong Wang. Plip: Language-image pre-training for person representation learning. Advances in Neural Information Processing Systems, 37: 45666–45702, 2024. 1, 2, 6, 7, 19

# Supplementary Material

In PRAISE of Attributes: Grounding Retrieval Geometry with Query-Conditioned Subspaces for Person Re-Identification Code and Attribute Bank: <sup>§</sup>Github

## A. Geometric Motivation for Attribute-Conditioned Retrieval

Prompt directions as biased semantic displacements. The central role of ACS is to use attribute prompts as directions that shape the retrieval geometry, rather than as calibrated attribute classifiers. To motivate this design, consider a normalized prompt embedding $\mathbf { z } _ { t , \ell } = \mathcal { F } _ { t } ( s _ { \ell } )$ . Because all prompts share a similar person-centric template, their embeddings contain both an attribute-specific component and a common template component. We model this as

$$
\mathbf { z } _ { t , \ell } = \mathbf { b } + \mathbf { a } _ { \ell } + \mathbf { \epsilon } \epsilon ,\tag{8}
$$

where b denotes the shared person/template component, $\mathbf { a } _ { \ell }$ denotes the semantic displacement induced by the attribute suffix, and $\epsilon _ { \ell }$ captures lexical and encoder noise. For a query embedding $\mathbf { z } _ { i }$ , direct prompt activation gives:

$$
\mathbf { z } _ { i } ^ { \top } \mathbf { z } _ { t , \ell } = \mathbf { z } _ { i } ^ { \top } \mathbf { b } + \mathbf { z } _ { i } ^ { \top } \mathbf { a } _ { \ell } + \mathbf { z } _ { i } ^ { \top } \boldsymbol { \epsilon } _ { \ell } .\tag{9}
$$

The first term is shared across many prompts and can dominate the attribute-specific term. Thus, raw prompt responses may be strongly affected by template bias and should not be treated as calibrated matching scores.

To reduce this shared component, PRAISE subtracts the neutral prompt embedding $\mathbf { z } _ { t , 0 } = \mathcal { F } _ { t } ( s _ { 0 } )$ and uses centered directions

$$
\bar { \mathbf { z } } _ { t , \ell } = \mathbf { z } _ { t , \ell } - \mathbf { z } _ { t , 0 } .\tag{10}
$$

If the neutral prompt captures the common person/template component, i.e., $\mathbf { z } _ { t , 0 }$ ≈ b, then

$$
\begin{array} { r } { \bar { \mathbf { z } } _ { t , \ell } \approx \mathbf { a } _ { \ell } + \boldsymbol { \epsilon } _ { \ell } - \boldsymbol { \epsilon } _ { 0 } . } \end{array}\tag{11}
$$

Therefore, prompt centering can be interpreted as a firstorder removal of template bias, leaving a direction that better approximates the attribute-specific semantic displacement.

Why direct attribute-profile fusion is unstable. Let the query and gallery prompt-response profiles be $\mathbf { u } _ { i } = \bar { \mathbf { Z } } _ { t } \mathbf { z } _ { i } ^ { \top }$ and $\mathbf { v } _ { j } = \bar { \mathbf Z } _ { t } \mathbf { z } _ { g , j } ^ { \top }$ . Ignoring the final normalization in Equation 4, their inner product can be written as

$$
\mathbf { u } _ { i } ^ { \top } \mathbf { v } _ { j } = \mathbf { z } _ { i } ^ { \top } \bar { \mathbf { Z } } _ { t } ^ { \top } \bar { \mathbf { Z } } _ { t } \mathbf { z } _ { g , j } .\tag{12}
$$

Thus, direct attribute-profile similarity is equivalent to comparing query and gallery embeddings under the fixed metric $\mathbf { M } _ { t } = \hat { \mathbf { Z } } _ { t } ^ { \top } \hat { \mathbf { Z } } _ { t }$ . This metric depends on the full prompt bank, including redundant and correlated prompts. If many prompts encode nearby concepts, such as similar colors or clothing phrases, then ${ { \bf { M } } _ { t } }$ over-weights those correlated directions. Consequently, $\mathbf { A } _ { \mathrm { a t t } } ^ { \mathrm { d i r e c t } }$ can amplify coarse attribute overlap while failing to separate true identity matches from distractors that share common attributes. This motivates replacing global prompt-profile fusion with a queryconditioned subspace comparison.

ACS as a query-conditioned semantic projector. For each query $\mathbf { z } _ { i } ,$ PRAISE selects the most compatible centered prompt directions,

$$
\begin{array} { r } { A _ { i } = \mathrm { T o p R } ( \bar { \mathbf Z } _ { t } \mathbf { z } _ { i } ^ { \top } ) , \qquad | A _ { i } | = r . } \end{array}\tag{13}
$$

This estimates a small set of active person-centric factors for the query. Since only a few attributes are usually relevant for a particular person image or text description, using the full prompt bank may introduce irrelevant directions. In contrast, $A _ { i }$ defines a local semantic coordinate system specific to query i.

Let

$$
\mathbf { U } _ { i } = \left[ \bar { \mathbf { z } } _ { t , \ell _ { 1 } } \cdot \cdot \cdot \bar { \mathbf { z } } _ { t , \ell _ { r } } \right] ^ { \top } \in \mathbb { R } ^ { r \times d } , \qquad \ell _ { k } \in \mathcal { A } _ { i } .\tag{14}
$$

The ACS basis is then obtained by thin QR decomposition:

$$
\begin{array} { r } { ( \mathbf { U } _ { i } ^ { w } ) ^ { \top } = \mathbf { Q } _ { i } \mathbf { R } _ { i } , \qquad \mathbf { Q } _ { i } ^ { \top } \mathbf { Q } _ { i } = I _ { m } , \qquad m \leq r . } \end{array}\tag{15}
$$

The important object is the projector

$$
\begin{array} { r } { \Pi _ { i } ^ { \parallel } = \mathbf { Q } _ { i } \mathbf { Q } _ { i } ^ { \top } . } \end{array}\tag{16}
$$

QR does not introduce new semantic information; it only constructs an orthonormal basis for the span of the selected attribute directions. Hence, ACS depends on the selected semantic span, not on the arbitrary scale, ordering, or redundancy of the prompts. In particular, for any orthogonal matrix $\mathbf { O } \in \mathbb { R } ^ { m \times m }$

$$
\begin{array} { r } { ( \mathbf { Q } _ { i } \mathbf { O } ) ( \mathbf { Q } _ { i } \mathbf { O } ) ^ { \top } = \mathbf { Q } _ { i } \mathbf { Q } _ { i } ^ { \top } = \Pi _ { i } ^ { \parallel } . } \end{array}\tag{17}
$$

Therefore, the induced projection is invariant to the particular orthonormal basis returned by QR.

Projected similarity as attribute-consistent evidence. The ACS affinity compares query and gallery features only through coordinates inside the query-conditioned semantic subspace:

$$
\begin{array} { r } { \mathbf { A } ^ { \parallel } [ i , j ] = ( \tilde { \mathbf { z } } _ { i } ^ { \parallel } ) ^ { \top } \tilde { \mathbf { z } } _ { g , j } ^ { \parallel } , \qquad \mathbf { z } _ { i } ^ { \parallel } = \Pi _ { i } ^ { \parallel } \mathbf { z } _ { i } , \quad \mathbf { z } _ { g , j } ^ { \parallel } = \Pi _ { i } ^ { \parallel } \mathbf { z } _ { g , j } . } \end{array}\tag{18}
$$

Equivalently, since $\mathbf { Q } _ { i } ^ { \top } \mathbf { Q } _ { i } = I _ { m }$ , this can be computed using subspace coordinates

$$
\begin{array} { r } { \mathbf { c } _ { i } = \mathbf { z } _ { i } ^ { \top } \mathbf { Q } _ { i } , \qquad \mathbf { c } _ { g , j } = \mathbf { z } _ { g , j } ^ { \top } \mathbf { Q } _ { i } , } \end{array}\tag{19}
$$

followed by $\ell _ { 2 }$ normalization and an inner product. This shows that ACS is not a prompt-response matching score. Instead, it is a projected cosine similarity under a queryconditioned semantic projector.

This distinction is important for ReID. Attributes such as clothing color, gender, or accessories are not identity labels; many identities may share them. Therefore, A<sup>∥</sup> should not replace the identity affinity $\mathbf { A } _ { \mathrm { i d } } .$ . Instead, it provides auxiliary evidence that checks whether a candidate gallery image agrees with the query along the selected semantic factors. The final affinity is therefore

$$
\mathbf { A } [ i , j ] = \mathbf { A } _ { \mathrm { i d } } [ i , j ] + \alpha _ { \mathrm { a c s } } \mathbf { A } ^ { \parallel } [ i , j ] .\tag{20}
$$

A small $\alpha _ { \mathrm { a c s } }$ makes ACS act as a semantic regularizer on the identity geometry: candidates must remain close in the original embedding space, but are additionally favored when they agree along query-relevant attribute directions.

## B. Motivation for the Loss function

![](images/733e427acb4037e8ed14a5eb2c96c741d0505a5314c7b3899997095a11693743.jpg)  
(a) Error rate vs Top-1 Cosine Distance

![](images/c55da0fe3fd11edc166eea86e5701f59b5928d6412250c8b826edfb0dd304194.jpg)  
(b) Error rate vs Feature Entropy  
Figure 5. Retrieval Top-1 Error-rate distribution across entropy and cosine distance. Curves computed on the Market-1501 [55] dataset for I2I-ReID and RSTPReID [56] for T2I-ReID.

We begin this section with a key observation in our loss function (Eq. 7, Main manuscript), that the loss essentially is a minimization of cosine distance within top-k gallery neighbours. Critically, most common TTA methods employ a direct feature level entropy [29, 38, 43]. We first consider an empirical analysis to observe trends in retrieval error rate versus (1) cosine distance and (2) feature entropy. Thereafter, we use these findings to develop a theoretical explanation for why direct entropy is ill-posed as a potential retrieval uncertainty surrogate, and an alternative loss function that is suitable for ReID.

## B.1. Empirical evidence: why feature-level entropy is a weak proxy for ReID uncertainty

We report two diagnostic curves (Figure 5) that relate the test error rate to two candidate “confidence” signals computed from the frozen retrieval model: (i) the cosine distance ofthe top-1 match between query and gallery, and (ii) the feature level entropy of a prediction distribution. The distance plot exhibits a clear, smooth, and largely monotonic trend: as the top-1 cosine distance increases, the error rate increases substantially. This behavior is consistent with the ReID geometry, i.e. when the closest gallery point is far in the embedding space, the query is less likely to have a correct nearest neighbor. In contrast, the entropy plot shows a noisier and partially non-monotonic relationship. In particular, there exist regimes where entropy is relatively low but the error rate is high, and also regimes where entropy grows without a predictable change in error. This implies that direct feature level entropy is miscalibrated as an uncertainty indicator fo ReID.

Entropy minimization implicitly encourages peaky predictions. In ReID test-time evaluation, identities are disjoint from training, and the correct notion of confidence is whether the query embedding is near its true neighbor(s) in $\mathbf { Z } _ { g } .$ , not whether it yields a confident classification under some surrogate label space. Concretely, entropy is typically defined over a distribution formed with respect to a classifier head trained on source identities. In case of ReID/Retrieval, the peaky predictions can occur for the wrong reasons. For example, a model can become highly confident in a direction that is unrelated to the correct gallery neighborhood, yielding low entropy but high error. This matches the empirical curve where low entropy does not guarantee low error.

Therefore, directly minimizing feature-level entropy is risky in ReID because it can over-emphasize “peakedness” (confidence) rather than neighborhood correctness. Under domain shift, this may accelerate drift toward spurious modes (overconfident wrong matches), which is exactly the kind of collapse we aim to avoid in test-time adaptation.

## B.2. Theoretical justification: distance as an uncertainty surrogate for nearest-neighbor retrieval

We provide an intuition consistent with the observed monotonicity in the distance plot.

Proposition B.1 (Distance as an uncertainty surrogate). Let $\mathbf { z } _ { i }$ be a query feature and $\mathbf { z } _ { g , j }$ be a gallery feature in $\mathbb { R } ^ { d } . ^ { 5 }$ Consider a nonnegative function u(·) that models uncertainty as a function of the Cosine distance between features, $i . e .$

$$
u ( d ) \quad w i t h \quad d = 1 - \mathbf { z } _ { i } ^ { \top } \mathbf { z } _ { g , j } .
$$

A natural uncertainty surrogate satisfies: (i) u(0) = 0 (certainty at zero distance) and (ii) u is non-decreasing (larger distance ⇒ larger uncertainty).

![](images/53779408938365d1217071a235afaab82e66527d6409ccc2cba0639591d48ec9.jpg)

![](images/7a6b93c977b1c79fb0deb78fe4836b5e8de9bfdb5d1ad25856f30daf1c82700a.jpg)  
(a) Query-Gallery Embedding Geometry  
(b) e(r) vs r expected curve  
Figure 6. Understanding the nature of the embedding space between a query and gallery, and a theoretical interpretation of Figure 5 where error rate is a surrogate for $e ( r )$

Interpretation. When the closest gallery neighbor to $\mathbf { z } _ { i }$ lies far away, there is more room for an incorrect identity to become the nearest neighbor under small perturbations (domain shift, camera style changes, background clutter). Hence, larger distance should correspond to higher retrieval uncertainty, consistent with the monotonic trend in the cosinedistance plot.

Proposition B.2 (Nearest-neighbor error is non-decreasing with distance). Fix a gallery set $\{ \mathbf { z } _ { g , j } \} _ { j = 1 } ^ { N _ { g } } \subset \mathbb { R } ^ { d }$ where $\mathbf { z } _ { g , 1 }$ denotes the unique correct gallery embedding for query $\mathbf { z } _ { i }$ and all other points are incorrect. Consider the 1-NN decision rule ${ \hat { \jmath } } ( \mathbf { z } _ { i } ) = \arg$ min<sub>j</sub> $\left( 1 - \mathbf { z } _ { i } ^ { \top } \mathbf { z } _ { g , j } \right)$ . Ifthe conditional distribution of $\mathbf { \dot { z } } _ { i }$ given $1 - \mathbf { z } _ { i } ^ { \top } \mathbf { z } _ { g , j } = r$ is isotropic around ${ \mathbf { z } } _ { g , 1 } ,$ then the probability oferror

$$
e ( r ) = { \mathrm { P r } } \left[ \hat { \jmath } ( \mathbf { z } _ { i } ) \neq 1 \ | \ | \mathbf { z } _ { i } - \mathbf { z } _ { g , 1 } | | _ { 2 } = r \right]
$$

is a non-decreasingfunction of $\dot { } r .$

Discussion. As illustrated in Figure 6, for fixed gallery points, the set of query locations that are correctly classified by 1-NN forms a Voronoi region around $\mathbf { z } _ { g , 1 }$ . Intersecting this region with a sphere of radius r centered at $\mathbf { z } _ { g , 1 }$ yields the set of directions from which the correct point remains nearest. As r increases, this intersection cannot gain new directions, and thus the measure of “correct directions” does not increase. Under isotropy, this implies $e ( r )$ is nondecreasing. This formalizes the key empirical observation: larger nearest-neighbor distance implies a higher chance of top-1 retrieval error.

Our ACS affinity $\mathbf { A } [ i , j ]$ in (6) is still a geometric similarity between query and gallery features (augmented by the ACS-induced subspace alignment). Thus, sharpening retrieval under A is aligned with increasing confidence in a distance-consistent way, unlike feature-level entropy that may be defined over a mismatched label space.

## B.3. Pitfalls of using cosine distance directly as loss

A naive objective would be to reduce the distance to the   
current best match, $\mathrm { e . g . }$ •,

$$
\operatorname* { m i n } _ { \theta } \ \frac { 1 } { B } \sum _ { i = 1 } ^ { B } \big ( 1 - \mathbf { A } [ i , \hat { \ j } _ { i } ] \big ) , \qquad \hat { \ j } _ { i } = \arg \operatorname* { m a x } _ { j } \mathbf { A } [ i , j ] .
$$

However, this is unstable for large-scale retrieval: (1) it depends on the discrete arg max index ${ \hat { \jmath } } _ { i }$ , which can switch abruptly across steps, producing high-variance gradients; (2) it over-commits to a single neighbor early, making the method vulnerable to confirmation bias under domain shift (a wrong early top-1 can be reinforced aggressively); and (3) its gradients do not explicitly control the relative separation between a small candidate set and the rest of the gallery.

Instead, we define the retrieval distribution: $\begin{array} { r l } { \mathbf { p } _ { i } } & { { } = } \end{array}$ softmax $( \mathbf { A } [ i , : ] / \tau )$ , which replaces hard assignment with a differentiable relaxation. The exponential map amplifies meaningful logit gaps while remaining smooth, and the temperature τ explicitly controls sharpness, i.e. high τ yields conservative updates and lower τ yields stronger sharpening. This improves optimization stability and makes the loss less sensitive to small perturbations in A.

Using − log p<sub>i</sub>(j) yields well-behaved gradients as it focuses optimization on increasing probability mass rather than unboundedly increasing raw similarities, and it naturally operates on relative scores across the gallery. In practice, this reduces the tendency to exploit trivial scaling effects and provides consistent gradients even when similarities saturate.

A full entropy minimization over all $N _ { g }$ gallery entries encourages the entire distribution to become extremely peaky, which can collapse to an arbitrary wrong index when the model is miscalibrated (as suggested by the entropy plot). We therefore minimize only over the head:

$$
{ \mathcal { L } } _ { \mathrm { T T A } } = - { \frac { 1 } { B } } \sum _ { i = 1 } ^ { B } \sum _ { j \in K _ { i } } \log \mathbf { p } _ { i } ( j ) \qquad K _ { i } = \mathrm { T o p K } ( \mathbf { p } _ { i } ) .\tag{21}
$$

This has three stabilizing effects: (i) it avoids overpenalizing the long tail of the gallery, which is dominated by many near-zero probabilities and can inject large aggregate gradient noise when $N _ { g }$ is large; (ii) it reduces confirmation bias by allowing a small candidate set to remain plausible rather than forcing a single winner too early; and (iii) it mitigates catastrophic collapse: even if the current top-1 is wrong, concentrating probability mass over several top candidates is empirically more robust and permits recovery across adaptation steps.

## C. Differentiability of the Loss

PRAISE frequently interleaves discrete retrieval operators (Top-k/Top-r) with continuous projection and normalization steps. As a result, it is important to assess differentiability to make explicit which components support gradient-based test-time optimization and which are treated as stop-gradient index selections.

Setup. Let θ denote the adapted parameters (the affine scale and bias in normalization layers) in the encoder being adapted. For a query batch $\mathbf { X } _ { q }$ of size $B ,$ the query embeddings are

$$
\mathbf { Z } ( \theta ) = { \mathcal { F } } _ { v } ( \mathbf { X } _ { q } ; \theta ) \in \mathbb { R } ^ { B \times d } , \qquad \mathbf { z } _ { i } ( \theta ) \in \mathbb { R } ^ { d } { \mathrm { ~ i s ~ r o w ~ } } i .
$$

We treat cached gallery embeddings $\mathbf { Z } _ { g } \in \mathbb { R } ^ { N _ { g } \times d }$ as constants during TTA $( \mathrm { i } . \mathrm { e } . , \partial \mathbf { Z } _ { g } / \partial \theta = 0 )$ , consistent with the implementation where $\mathbf { Z } _ { g }$ is precomputed once per test set.

Given logits $\mathbf { A } ( \theta ) \in \bar { \mathbb { R } } ^ { B \times \mathbf { \bar { N } } _ { g } }$ , the gradient follows by the chain rule:

$$
\nabla _ { \boldsymbol { \theta } } \mathcal { L } = \sum _ { i = 1 } ^ { B } \left( \frac { \partial \mathbf { z } _ { i } } { \partial \boldsymbol { \theta } } \right) ^ { \top } \left( \frac { \partial \mathcal { L } } { \partial \mathbf { z } _ { i } } \right) .
$$

Thus it suffices to characterize $\partial { \mathcal { L } } / \partial \mathbf { A }$ and ${ \partial { \bf A } } / { \partial { \bf z } _ { i } }$

## C.1. Differentiability of the retrieval loss

For each query $i ,$ define logits $\mathbf { s } _ { i } = \mathbf { A } [ i , : ] \in \mathbb { R } ^ { N _ { g } }$ and retrieval distribution

$$
\mathbf { p } _ { i } = \mathrm { s o f t m a x } ( \mathbf { s } _ { i } / \tau ) , \qquad p _ { i } ( j ) = { \frac { \exp ( { s _ { i , j } / \tau } ) } { \sum _ { m = 1 } ^ { N _ { g } } \exp ( { s _ { i , m } / \tau } ) } } .
$$

The loss function has been defined in Equation 21. This objective is piecewise smooth in $\mathbf { s } _ { i } \mathbf { : }$ it is smooth whenever the set $\kappa _ { i }$ is unchanged $( \mathrm { i . e . }$ , away from boundary events / ties). In implementation, $\kappa _ { i }$ is treated as an index set (stop-gradient through TopK).

Now, let $m _ { i } ( \ell ) = \mathbb { I } [ \ell \in \mathcal { K } _ { i } ]$ . For a fixed $\kappa _ { i }$

$$
\frac { \partial \mathcal { L } _ { \mathrm { T T A } } } { \partial s _ { i , \ell } } = \frac { 1 } { B \tau } \left( k p _ { i , \ell } - m _ { i } ( \ell ) \right) , \qquad \ell \in \{ 1 , \dots , N _ { g } \} .
$$

This follows from ∂ log $p _ { i , j } / \partial s _ { i , \ell } = \frac { 1 } { \tau } ( \mathbb { I } [ \ell = j ] - p _ { i , \ell } )$ (derivative of softmax function) and summing over $j \in \mathcal { K } _ { i } .$

At ties around the k-th threshold, TopK is not differentiable. The loss remains well-defined, and the above expression corresponds to a valid subgradient under a deterministic tie-breaking rule.

## C.2. ACS affinity as a differentiable map

Recap Given centered prompt directions $\bar { \mathbf Z } _ { t } \in \mathbb R ^ { L \times d }$ (constant) and a query embedding $\mathbf { z } _ { i } .$ , prompt activations are $\mathbf { u } _ { i } = \bar { \mathbf { Z } } _ { t } \mathbf { z } _ { i } ^ { \top } \in \mathbb { R } ^ { L }$ and

$$
\begin{array} { r } { A _ { i } = \mathrm { T o p R } ( \mathbf { u } _ { i } ) , \qquad | A _ { i } | = r \ll L . } \end{array}
$$

We stack the selected directions

$$
\mathbf { U } _ { i } = \left[ \bar { \mathbf { z } } _ { t , \ell _ { 1 } } \cdot \cdot \cdot \bar { \mathbf { z } } _ { t , \ell _ { r } } \right] ^ { \top } \in \mathbb { R } ^ { r \times d } , \qquad \ell _ { k } \in \mathcal { A } _ { i } ,
$$

and obtain a thin QR factorization

$$
\mathbf { U } _ { i } ^ { \top } = \mathbf { Q } _ { i } \mathbf { R } _ { i } , \qquad \mathbf { Q } _ { i } \in \mathbb { R } ^ { d \times m } , \quad \mathbf { Q } _ { i } ^ { \top } \mathbf { Q } _ { i } = I _ { m } , \quad m \leq r .
$$

The ACS projector is $\pmb { \Pi } _ { i } ^ { \parallel } = \mathbf { Q } _ { i } \mathbf { Q } _ { i } ^ { \top }$

As stated in the method, A<sub>i</sub> is a discrete index set and we do not backpropagate through TopR. In our implementation we also treat $\mathbf { Q } _ { i }$ <sub>i</sub> as constant w.r.t. $\mathbf { z } _ { i }$ during differentiation, i.e., $\begin{array} { r } { \frac { \partial \mathbf Q _ { i } } { \partial \mathbf z _ { i } } = 0 , } \end{array}$ , to avoid unstable gradients through QR under changing index sets. Therefore, gradients propagate only through subsequent matrix products, projections, normalizations, and dot-products.

## C.3. Projection, normalization, and subspace affinity

Projected features. With $\boldsymbol { \Pi } _ { i } ^ { \parallel }$ fixed (given $\mathbf { Q } _ { i } )$ , projected query and gallery features are

$$
\begin{array} { r } { \mathbf { z } _ { i } ^ { \parallel } = \boldsymbol { \Pi } _ { i } ^ { \parallel } \mathbf { z } _ { i } , \qquad \mathbf { z } _ { g , j } ^ { \parallel } = \boldsymbol { \Pi } _ { i } ^ { \parallel } \mathbf { z } _ { g , j } , } \end{array}
$$

and hence

$$
\frac { \partial \mathbf { z } _ { i } ^ { \parallel } } { \partial \mathbf { z } _ { i } } = \boldsymbol { \Pi } _ { i } ^ { \parallel } .
$$

We are able to treat $\boldsymbol { \Pi } _ { i } ^ { \parallel }$ as relatively constant with respect to $\mathbf { z } _ { i } ,$ due to the stop-gradient in the ACS basis construction.

Stabilized normalization. We use $\tilde { \mathbf { z } } = \frac { \mathbf { z } } { \| \mathbf { z } \| _ { 2 } + \epsilon }$ . For any nonzero $v \in \mathbb { R } ^ { d }$ , let $r ( \mathbf { v } ) = \| \mathbf { v } \| _ { 2 } + \epsilon$ and $\widetilde { \mathbf { v } } = \mathbf { v } / r ( \mathbf { v } )$ . A convenient Jacobian form is

$$
\frac { \partial  { \widetilde { \mathbf { v } } } } { \partial  { \mathbf { v } } } = \frac { 1 } { r (  { \mathbf { v } } ) } \left( I - \frac {  { \mathbf { v v } } ^ { \top } } { \|  { \mathbf { v } } \| _ { 2 } r (  { \mathbf { v } } ) } \right) ,
$$

which reduces to $\begin{array} { r } { \frac { 1 } { \| \mathbf { v } \| _ { 2 } } \left( I - \tilde { \mathbf { v } } \tilde { \mathbf { v } } ^ { \top } \right) } \end{array}$ when $\epsilon  0$ and $\| \mathbf { v } \| _ { 2 } >$ 0. Adding $\epsilon > 0$ makes the denominator $r ( \mathbf { v } )$ bounded away from zero, so $\tilde { \mathbf { v } } = \mathbf { v } / r ( \mathbf { v } )$ does not explode when $\| \mathbf { v } \| _ { 2 }$ is very small (e.g., after projection). Consequently, the Jacobian terms that scale as $1 / r ( \mathbf { v } )$ and $1 / ( \| \mathbf { v } \| _ { 2 } r ( \mathbf { v } ) )$ remain numerically controlled, preventing gradient blow-up near near-zero feature norms.

Subspace affinity. The ACS subspace affinity is

$$
\mathbf { A } ^ { \parallel } [ i , j ] = ( \tilde { \mathbf { z } } _ { i } ^ { \parallel } ) ^ { \top } \tilde { \mathbf { z } } _ { g , j } ^ { \parallel } , \qquad \tilde { \mathbf { z } } _ { i } ^ { \parallel } = \mathrm { n o r m } ( \mathbf { z } _ { i } ^ { \parallel } ) , \quad \tilde { \mathbf { z } } _ { g , j } ^ { \parallel } = \mathrm { n o r m } ( \mathbf { z } _ { g , j } ^ { \parallel } ) .
$$

Since $\mathbf { Z } _ { g }$ is cached, $\tilde { \mathbf { z } } _ { g , j } ^ { \parallel }$ is constant w.r.t. θ. Let $v _ { i } = \mathbf { z } _ { i } ^ { \parallel }$ Then

$$
\frac { \partial { \bf A } ^ { \parallel } [ i , j ] } { \partial v _ { i } } = \left( \frac { \partial \tilde { v } _ { i } } { \partial v _ { i } } \right) ^ { \top } \tilde { \bf z } _ { g , j } ^ { \parallel } , \qquad \frac { \partial { \bf A } ^ { \parallel } [ i , j ] } { \partial { \bf z } _ { i } } = { \bf \Pi } ^ { \parallel } _ { i } \left( \frac { \partial \tilde { v } _ { i } } { \partial v _ { i } } \right) ^ { \top } \tilde { \bf z } _ { g , j } ^ { \parallel } .
$$

## C.4. Gradient of the final ACS logits

The baseline identity affinity is

$$
\mathbf { A } _ { \mathrm { i d } } [ i , j ] = ( \tilde { \mathbf { z } } _ { i } ) ^ { \top } \tilde { \mathbf { z } } _ { g , j } ,
$$

with $\tilde { \mathbf { z } } _ { g , j }$ constant and $\tilde { \mathbf { z } } _ { i } = \mathrm { n o r m } ( \mathbf { z } _ { i } )$ . Therefore,

$$
\frac { \partial { \bf A } _ { \mathrm { i d } } [ i , j ] } { \partial { \bf z } _ { i } } = \left( \frac { \partial \tilde { \bf z } _ { i } } { \partial { \bf z } _ { i } } \right) ^ { \top } \tilde { \bf z } _ { g , j } .
$$

The final logits used by the loss are

$$
\mathbf { A } [ i , j ] = \mathbf { A } _ { \mathrm { i d } } [ i , j ] + \alpha _ { \mathrm { a c s } } \mathbf { A } ^ { \parallel } [ i , j ] .
$$

Thus

$$
\frac { \partial \mathbf { A } [ i , j ] } { \partial \mathbf { z } _ { i } } = \frac { \partial \mathbf { A } _ { \mathrm { i d } } [ i , j ] } { \partial \mathbf { z } _ { i } } + \alpha _ { \mathrm { a c s } } \frac { \partial \mathbf { A } ^ { \parallel } [ i , j ] } { \partial \mathbf { z } _ { i } } .
$$

## C.5. End-to-end gradient to the adapted parameters summary

Let $\begin{array} { r } { \mathbf { g } _ { i } = \frac { \partial \mathcal { L } _ { \mathrm { T T A } } } { \partial \mathbf { s } _ { i } } \in \mathbb { R } ^ { N _ { g } } } \end{array}$ be the gradient w.r.t. logits for query i (Sec. C.1). Then

$$
\frac { \partial \mathcal { L } _ { \mathrm { T T A } } } { \partial \mathbf { z } _ { i } } = \sum _ { j = 1 } ^ { N _ { g } } g _ { i , j } \frac { \partial \mathbf { A } [ i , j ] } { \partial \mathbf { z } _ { i } } .
$$

Finally, since $\mathbf { z } _ { i } = \mathcal { F } _ { v } ( x _ { i } ; \theta )$ for I2I-ReID (or $\mathbf { z } _ { i } = \mathcal { F } _ { t } ( \cdot ; \theta _ { t } )$ for T2I-ReID), we have established that gradients exist and are well defined for each step of PRAISE which can be computed by autograd algorithm [34]. Despite the stop-gradient and several top level indexing operations, the objective is differentiable almost everywhere and gradients propagate through: (1) matrix products for $\mathbf { u } _ { i }$ (used only for selection), (2) linear projection $\mathbf { z } \mapsto \Pi _ { i } ^ { \parallel } \mathbf { z } , ( 3 )$ stabilized normalization ${ \mathbf z } \mapsto \tilde { { \mathbf z } } , ( 4 )$ inner products defining $\mathbf { A } _ { \mathrm { i d } }$ and $\mathbf { A } ^ { \parallel }$ , (5) softmax, and (piecewise constant) top-k indexing in the loss.

## D. Full prompt bank constructions details

This section describes the construction of the attribute prompt bank used by PRAISE. The overall pipeline is shown in Figure 7. Given a set of unlabeled pedestrian images, we first use a vision-language model to extract compact appearance descriptions, then group the extracted cues into predefined pedestrian attribute categories, clean duplicate or noisy phrases, and finally insert the resulting attribute suffixes into a shared prompt template. Importantly, the procedure does not use identity labels, attribute annotations for any kind of target-domain supervision. The resulting bank is a fixed transferable vocabulary of pedestrian appearance cues, which is encoded once and reused by PRAISE during test-time adaptation.

## D.1. Input image collection and sampling

Let $\mathcal { X } _ { \mathrm { s r c } } = \{ x _ { n } \} _ { n = } ^ { N }$ denote a collection of pedestrian images used only for prompt-bank construction. In our default setting, $\mathcal { X } _ { \mathrm { s r c } }$ is taken from the Market-1501 training split. For the prompt-bank source ablation, we repeat the same procedure using images from MSMT17, CUHK-PEDES, RSTP-ReID, and a manually specified prompt list. When person identifiers are available in the file names or dataset metadata, we sample at most a fixed number of images per identity to avoid over-representing identities with many camera views. These identifiers are used only for balanced image sampling and are never used as identity supervision for training or adaptation.

![](images/8bd2fed7184fcba70c5689245a87f47c34e0ab2a9fd0b8c537f167ddb442ab95.jpg)  
Figure 7. Prompt-bank construction pipeline. (a) A VLM is used to caption unlabeled pedestrian images and extract recurring appearance cues. These cues are grouped into semantic categories and cleaned to remove duplicate or near-identical phrases. (b) The cleaned attribute suffixes are inserted into a shared template to form the final prompt bank $\mathcal { P } = \{ s _ { \ell } \} _ { \ell = 1 } ^ { L }$ , together with a neutral prompt $s _ { 0 }$ used for prompt centering.

## D.2. VLM-based appearance extraction

We use Qwen2.5-VL-7B-Instruct [50] to extract short, visually grounded pedestrian attributes from each sampled image. The instruction asks the model to ignore background, pose, action, camera information, brand names, and identityspecific information, and to return only compact appearance attributes. The output is constrained to a JSON-like schema so that the extracted cues can be parsed automatically and aggregated across the dataset.

## VLM appearance-extraction instruction

Extract only visible pedestrian   
appearance attributes from this cropped   
person image.   
Return ONLY valid JSON with these keys:   
{   
“upper\_color”: “string or list of short   
strings; use unknown if not visible”,   
“upper\_type”: “string or list of short   
strings; use unknown if not visible”,   
“lower\_color”: “string or list of short   
strings; use unknown if not visible”,   
“lower\_type”: “string or list of short   
strings; use unknown if not visible”,   
“shoes”: “string or list of short   
strings; use unknown if not visible”,   
“bag”: “string or list of short strings;   
use unknown if not visible”,   
“accessory”: “string or list of short   
strings; use unknown if not visible”,   
“hair”: “string or list of short   
strings; use unknown if not visible”   
}   
Rules: use lowercase; keep values short;   
do not mention background, pose, action,   
camera, brands, identity, or image   
quality; use “unknown” if a cue is absent   
or unclear; output JSON only.

For analyses involving sensitive or demographic-like prompts, we additionally allow the keys age and gender. However, these categories are not required by the method. In the attribute-category ablation, we explicitly evaluate the effect of removing age and gender prompts to verify that PRAISE does not depend on such cues.

## D.3. Attribute grouping

The parsed VLM outputs are grouped into a small set of pedestrian-centric categories. For the default bank, we use appearance categories that are commonly visible in standard short-term ReID. When included for analysis, the additional categories are {age, gender}. The purpose of grouping is not to predict supervised attributes, but to organize the mined vocabulary into interpretable categories from which query-conditioned ACS directions can be selected.

## Attribute grouping schema

```csv
Category Example extracted values
upper_color red, black, white, blue, green
upper_type t-shirt, shirt, jacket, hoodie, coat
lower_color black, blue, gray, brown, white
lower_type pants, jeans, shorts, skirt, dress
shoes white shoes, black shoes, sneakers
bag backpack, handbag, shoulder bag
accessory hat, glasses, umbrella
hair short, long, black, brown
age young, adult, elderly
gender man, woman, person
```

## D.4. Vocabulary cleaning and de-duplication

The raw VLM outputs contain synonymous, near-duplicate, and occasionally noisy phrases. We therefore normalize each extracted value before adding it to the vocabulary. The cleaning stage lowercases all strings, removes punctuation and filler words, discards unknown or non-visible values, and merges common synonyms. For example, grey is canonicalized to gray, trousers to pants, and back pack to backpack. Long phrases are discarded because they usually indicate caption spillover rather than atomic attributes.

After normalization, we count the frequency of each attribute phrase within each category. Attributes that occur fewer than a threshold $m _ { \mathrm { m i n } }$ are removed, and we optionally keep only the top M phrases per category. This produces a compact bank that covers recurring pedestrian cues while suppressing rare or noisy descriptions. Formally, for category $c \in { \mathcal { C } } .$ , we retain

$$
\mathcal { A } _ { c } = \left\{ a \ : \ \mathrm { c o u n t } ( a , c ) \geq m _ { \mathrm { m i n } } \right\} ,\tag{22}
$$

where a is a cleaned attribute suffix.

Vocabulary cleaning examples   
Raw phrase Cleaning operation Canonical phrase   
grey shirt spelling normalization gray shirt   
back pack synonym merge backpack   
trousers synonym merge pants   
not visible unknown-value removal discarded   
person wearing a red shirt filler removal red shirt

## D.5. Prompt synthesis

The cleaned attribute suffixes are converted into naturallanguage prompts using a shared template. For most categories, the generic form is

$$
s _ { \ell } = { } ^ { \mathrm { * } } \mathrm { A p h o t o ~ o f ~ a ~ p e r s o n ~ w i t h ~ [ A T T ] } . ^ { \mathrm { , } }\tag{23}
$$

where [ATT] is replaced by the cleaned attribute suffix. For grammatical consistency, we use category-specific templates for clothing colors, clothing types, shoes, bags, accessories, and hair. Examples are shown below.

## Prompt synthesis templates

Category Template   
upper\_color A photo of a person wearing a [ATT] upper-body garment.   
upper\_type A photo of a person wearing [ATT].   
lower\_color A photo of a person wearing a [ATT] lower-body garment.   
lower\_type A photo of a person wearing [ATT].   
shoes A photo of a person wearing [ATT] shoes.   
bag A photo of a person carrying [ATT].   
accessory A photo of a person with [ATT].   
hair A photo of a person with [ATT] hair.

The final prompt bank is

$$
\mathcal { P } = \{ s _ { \ell } \} _ { \ell = 1 } ^ { L } ,\tag{24}
$$

where each $s _ { \ell }$ corresponds to one retained attribute phrase. We also define a neutral prompt

$$
s _ { 0 } = \mathrm { { ^ { * } A p h o t o o f a p e r s o n . ^ { \prime } } }\tag{25}
$$

During PRAISE, all prompts are encoded by the frozen text encoder and centered by subtracting the neutral prompt embedding:

$$
\bar { \mathbf { z } } _ { t , \ell } = \mathbf { z } _ { t , \ell } - \mathbf { z } _ { t , 0 } .\tag{26}
$$

This centering step reduces shared-template bias and makes the prompt bank behave as a set of attribute directions rather than a set of independent prompt classifiers.

## D.6. Prompt-bank construction and implementation details

Attribute mining. We construct the attribute-prompt bank once as an offline preprocessing step using Qwen2.5-VL-7B-Instruct. For each ReID dataset, attributes are mined exclusively from its training-image partition; test queries and gallery images are never used for prompt-bank construction. For Market-1501, we sample at most four training images per identity and cap the total mining set at 3,000 images. Sampling is performed with a fixed random seed of 0. Training-set identity identifiers are used only to obtain approximately balanced image sampling across identities; identity labels are never provided to the VLM, do not determine the extracted vocabulary, and are not used during PRAISE test-time adaptation. Unless otherwise stated, junk/background identities are excluded.

VLM inference. Images are processed independently with Qwen2.5-VL-7B-Instruct in evaluation mode using bfloat16 precision on a single GPU. Generation is deterministic with a maximum of 192 newly generated tokens per image. The VLM is instructed to return only a JSON object containing short, visually observable pedestrian attributes and to output unknown when an attribute is not reliably visible. We explicitly prohibit descriptions of background, pose, action, camera, identity, brands, or image quality. By default, the mined vocabulary covers eight person-centric categories: upper-body color, upper-body clothing type, lower-body color, lower-body clothing type, shoes, bags, accessories, and hair. We do not include the optional age or gender categories in the experiments reported in this work.

Vocabulary cleaning and prompt generation. The raw VLM outputs are parsed and canonicalized before constructing the bank. Values marked as unknown/unclear are discarded, textual variants are normalized, common color and clothing synonyms are merged, and excessively long attribute phrases are removed. We retain an attribute only if it occurs in at least five mined images and retain at most 40 attributes per category. Duplicate prompts are subsequently removed. Each surviving attribute is converted into a category-specific natural-language template; for example, an upper-body color is expressed as “A photo of a person wearing a {attribute} upper-body garment,” while a bag attribute uses “A photo of a person carrying {attribute}.” The resulting bank contains L = 57 unique prompts for Market-1501. The complete vocabulary and occurrence counts are provided in the released prompt-bank files.

Reproducibility and computational cost. For Market-1501, we sample up to four training images per identity, with a maximum of 3,000 images used for prompt mining. Attributes are retained only if they are observed in at least five sampled images, and the vocabulary is capped at 40 attributes per semantic category. All sampling is performed with a fixed random seed of 0. The complete mining procedure requires approximately 102 minutes ( 12 minutes for loading model shards and 90 minutes for analyzing 3000 images from the training dataset) on an NVIDIA RTX A6000 GPU. This cost is incurred only once when constructing the reusable semantic vocabulary and is therefore reported separately from online PRAISE inference. Once constructed, the prompt bank is stored and reused for all subsequent queries; prompt embeddings are likewise precomputed before testtime adaptation. Additional experiments in subsection E.2 show that banks mined from a different source dataset remain effective, indicating that PRAISE does not rely on a dataset-specific prompt vocabulary.

## E. Additional Experiments

## E.1. Effectiveness of PRAISE in addressing image corruption domain shifts

Table 5 evaluates I2I-ReID on Market-1501 under five common corruptions, taking insights from common TTA evaluations in [20, 38, 43], originally proposed in [11]. Two regimes emerge: (i) noise corruptions (Gaussian/Poisson) where several generic entropy-based TTA methods become comparatively stronger than in the clean-domain table, and (ii) blur/brightness corruptions where relative rankings more closely match the main-paper trends. The application of these corruptions can be summarized as follows:

Table 5. Image Corruptions and Noise. Comparison of TTA methods on I2IReID using a PLIP [58] backbone. Results are reported on Market-1501 under five corruptions (mAP / R1).
<table><tr><td rowspan="2">TTA Method</td><td colspan="10">Market-1501 (Corrupted)</td></tr><tr><td>Gaussian Noise mAP</td><td>R1</td><td>Poisson Noise mAP</td><td>R1</td><td>Defocus Blur mAP</td><td>R1</td><td>Motion Blur mAP</td><td>R1</td><td>Brightness mAP</td><td>R1</td></tr><tr><td>No-adapt (Source) [NeurlPS&#x27;24]</td><td>24.2</td><td>50.7</td><td>14.5</td><td>37.5</td><td>13.7</td><td>35.7</td><td>8.9</td><td>26.1</td><td>33.5</td><td>60.9</td></tr><tr><td>TENT [43] [ICLR&#x27;21]</td><td>23.4</td><td>48.0</td><td>9.0</td><td>22.6</td><td>11.5</td><td>34.2</td><td>6.9</td><td>19.9</td><td>29.7</td><td>55.3</td></tr><tr><td>SHOT [28] [ICML/20]</td><td>23.8</td><td>51.1</td><td>15.7</td><td>36.3</td><td>13.2</td><td>33.6</td><td>1.9</td><td>4.9</td><td>28.5</td><td>55.5</td></tr><tr><td>SAR [30] [ICLR&#x27;23]</td><td>26.6</td><td>52.2</td><td>16.8</td><td>38.8</td><td>14.2</td><td>35.0</td><td>8.7</td><td>24.3</td><td>31.8</td><td>58.1</td></tr><tr><td>ETA [29] [ICML/22]</td><td>25.1</td><td>50.8</td><td>16.4</td><td>39.1</td><td>14.0</td><td>34.8</td><td>9.3</td><td>26.1</td><td>31.1</td><td>57.7</td></tr><tr><td>DeYO [21] [ICLR&#x27;24]</td><td>27.4</td><td>54.0</td><td>18.2</td><td>41.8</td><td>14.6</td><td>37.5</td><td>10.2</td><td>28.9</td><td>33.6</td><td>61.2</td></tr><tr><td>TPT [38] [NeurIPS&#x27;22]</td><td>26.7</td><td>53.0</td><td>15.1</td><td>35.9</td><td>15.1</td><td>38.2</td><td>9.1</td><td>23.6</td><td>34.1</td><td>61.2</td></tr><tr><td>TDA [18] [CVPR&#x27;24]</td><td>26.7</td><td>53.8</td><td>16.2</td><td>39.4</td><td>13.7</td><td>35.9</td><td>8.9</td><td>26.1</td><td>33.4</td><td>60.8</td></tr><tr><td>TEMP [1] [ICNN&#x27;24]</td><td>25.1</td><td>51.2</td><td>15.1</td><td>35.7</td><td>14.1</td><td>35.9</td><td>9.1</td><td>26.2</td><td>33.6</td><td>61.1</td></tr><tr><td>Anchor-ReID [4] [SCIA&#x27;23]</td><td>11.0</td><td>24.8</td><td>5.9</td><td>19.2</td><td>8.5</td><td>16.8</td><td>5.5</td><td>13.4</td><td>22.2</td><td>32.3</td></tr><tr><td>Norm [40] [ICLR&#x27;25]</td><td>24.8</td><td>50.9</td><td>14.6</td><td>37.8</td><td>14.5</td><td>36.1</td><td>9.4</td><td>26.8</td><td>35.1</td><td>62.9</td></tr><tr><td>PRAISE (Ours)</td><td>27.7</td><td>55.1</td><td>15.3</td><td>38.9</td><td>23.9</td><td>41.8</td><td>15.7</td><td>33.4</td><td>44.8</td><td>69.8</td></tr></table>

• Gaussian noise. We convert the image to a tensor in $[ 0 , 1 ]$ we add i.i.d. Gaussian noise $\mathcal { N } ( 0 , \bar { \sigma } ^ { 2 } )$ per pixel/channel with $\sigma = 0 . 0 6$ , and we clamp back to [0, 1].

• Poisson noise. We convert the image to a tensor $x \in [ 0 , 1 ]$ we scale intensities to pseudo-counts $x \cdot \lambda$ with $\lambda = 2 0$ we sample elementwise Poisson noise and rescale by $1 / \lambda ,$ and we clamp to [0, 1].

• Defocus blur. We apply PIL Gaussian blur with radius r = 3.0.

• Motion blur. We build a normalized k×k line kernel with k = 13 at a randomly chosen angle in {0, 45, 90, 135}, we reflect-pad the image, and we convolve per-channel using grouped convolution, then clamp to [0, 1].

• Brightness. We apply a brightness transform with factor $f \sim \mathcal { U } ( 0 . 5 5 , 1 . 4 5 )$ using PIL brightness enhancement.

Under additive noise, the dominant failure mode is often feature dispersion (reduced similarity gaps) rather than semantic misalignment. Entropy-based adaptation methods can then act as effective confidence restorers: by sharpening the retrieval posterior they partially recover top-neighbor separation, which explains their improved mAP/R1 under Gaussian and Poisson noise compared to their clean-domain behavior. This is especially visible for SAR, ETA and DeYO, which surpass or match the no-adapt baseline under both noise types (e.g., DeYO reaches 27.4/54.0 on Gaussian and 18.2/41.8 on Poisson, while the baseline is 24.2/50.7 and 14.5/37.5). In other words, when corruptions are mostly non-semantic and approximately independent and identically distributed, entropy filtering tends to select “still-recoverable” queries, and sharpening helps re-separate nearest neighbors.

Some methods exhibit extreme behavior under specific corruptions (e.g., SHOT collapses under Motion Blur with 1.9/4.9). This is plausible because motion blur introduces strong, structured high-frequency attenuation and spatial streaking, which can systematically break pseudolabel/consistency assumptions and thus updates can become dominated by incorrect neighbors, leading to rapid confirmation-bias collapse. Conversely, methods can appear unusually strong under noise because the corruption perturbs features in a way that preserves coarse identity structure and the TTA objective can then effectively “denoise” the retrieval posterior by amplifying pre-existing similarity gaps.

Table 6. Effect of attribute prompt-bank source. We evaluate PRAISE using the same fixed hyperparameters while varying only the source used to construct the attribute prompt bank. The manual bank is hand-designed, while the remaining banks are mined from the corresponding ReID datasets using the same label-free captioning and cleaning protocol. Results show that PRAISE is not tied to a specific prompt-bank source and remains stable across transferable pedestrian attribute vocabularies.
<table><tr><td rowspan="2">Prompt-bank source</td><td colspan="2">MSMT17</td><td colspan="2">Market-1501</td><td colspan="2">CUHK-PEDES</td><td colspan="2">RSTP-ReID</td></tr><tr><td>mAP</td><td>R1</td><td>mAP</td><td>R1</td><td>mAP</td><td>R1</td><td>mAP</td><td>R1</td></tr><tr><td>No-adapt source model</td><td>10.5</td><td>34.2</td><td>35.0</td><td>62.6</td><td>46.4</td><td>55.9</td><td>23.1</td><td>34.5</td></tr><tr><td>Generic prompt bank Market-1501 mined bank</td><td>17.8</td><td>36.3</td><td>45.9</td><td>70.1</td><td>48.7</td><td>56.1</td><td>27.1</td><td>36.1</td></tr><tr><td>MSMT17 mined bank</td><td>18.2</td><td>38.5</td><td>46.5</td><td>71.1</td><td>49.6</td><td>58.4</td><td>27.6</td><td>37.6</td></tr><tr><td></td><td>18.4</td><td>39.1</td><td>46.5</td><td>70.1</td><td>49.2</td><td>57.6</td><td>26.9</td><td>37.0</td></tr><tr><td>CUHK-PEDES mined bank</td><td>17.9</td><td>36.5</td><td>45.4</td><td>69.7</td><td>49.4</td><td>58.1</td><td>27.6</td><td>37.5</td></tr><tr><td>RSTP-ReID mined bank</td><td>17.8</td><td>36.3</td><td>45.3</td><td>69.7</td><td>49.1</td><td>57.7</td><td>27.2</td><td>36.9</td></tr></table>

For Defocus Blur, Motion Blur, and Brightness, the ordering becomes more consistent with the main paper: generic entropy-minimization baselines provide only modest gains (or can degrade), while retrieval-aligned objectives are more reliable. Notably, PRAISE improves substantially on the harder structured corruptions (e.g., Defocus Blur 23.9/41.8 and Motion Blur 15.7/33.4), where naive sharpening is more likely to overfit to spurious top-1 matches. This is consistent with the view that blur/illumination changes induce more semantics-affecting shifts (texture suppression, edge degradation, global intensity shifts) that demand retrieval-specific inductive biases, making the robustness trends closer to those observed in the evaluations of the main manuscript.

## E.2. Attribute bank Source and Category Ablations

Impact of attribute source. Table 6 studies whether the performance of PRAISE depends on the dataset used to construct the attribute prompt bank. Across all four benchmarks, replacing the default Market-1501 mined bank with manually designed prompts or banks mined from MSMT17, CUHK-PEDES, and RSTP-ReID leads to only small performance changes. All prompt banks consistently improve over the no-adapt source model, indicating that the gains of PRAISE do not come from a dataset-specific vocabulary or from overfitting the prompt bank to a particular benchmark. We observe a slight advantage for prompt banks mined from I2I-ReID datasets, especially Market-1501 and MSMT17. This is likely because these datasets contain surveillance-style pedestrian crops collected across multiple cameras, viewpoints, poses, and illumination conditions, which closely matches the visual gallery distribution used during ReID retrieval. As a result, the mined attributes tend to capture common visually grounded pedestrian cues, such as clothing color, clothing type, bags, shoes, and hair, that transfer well across both I2I and T2I settings. In contrast, banks mined from T2I-ReID datasets remain competitive but can reflect the visual and linguistic biases of text-person-search datasets, where some cues may be more description-oriented or less diverse in surveillance appearance. Nevertheless, the small variation across rows shows that PRAISE is robust to the prompt-bank source and only requires a transferable pedestrian attribute vocabulary rather than a dataset-specific prompt set.

Table 7. Attribute-category ablation. We evaluate PRAISE using the same Market-1501 mined prompt bank and fixed hyperparameters, while removing one group of attributes at a time. This isolates which attribute categories contribute most to the query-conditioned ACS geometry.
<table><tr><td rowspan="2">Prompt-bank variant</td><td colspan="2">Market-1501</td><td colspan="2">RSTP-ReID</td></tr><tr><td>mAP</td><td>R1</td><td>mAP</td><td>R1</td></tr><tr><td>No-adapt source model</td><td>35.0</td><td>62.6</td><td>23.1</td><td>34.5</td></tr><tr><td>Full prompt bank</td><td>46.5</td><td>71.1</td><td>27.6</td><td>37.6</td></tr><tr><td>w/o upper-body color w/o lower-body color</td><td>45.8</td><td>68.1</td><td>26.1</td><td>35.2</td></tr><tr><td></td><td>45.9</td><td>69.3</td><td>27.0</td><td>36.9</td></tr><tr><td>w/o upper-body clothing</td><td>46.0</td><td>69.5</td><td>27.2</td><td>37.4</td></tr><tr><td>w/o lower-body clothing</td><td>46.3</td><td>70.1</td><td>27.5</td><td>37.5</td></tr><tr><td>w/o bag/accessories</td><td>45.5</td><td>68.2</td><td>25.5</td><td>34.3</td></tr><tr><td>w/o hair</td><td>46.5</td><td>70.3</td><td>27.7</td><td>37.8</td></tr><tr><td>w/o age</td><td>46.7</td><td>70.8</td><td>27.6</td><td>37.6</td></tr><tr><td>w/o gender</td><td>46.4</td><td>70.2</td><td>27.5</td><td>37.6</td></tr></table>

Impact of attribute categories. Table 7 studies the contribution of different attribute groups by removing one category at a time from the Market-1501 mined prompt bank. All ablated variants remain well above the no-adapt source model, showing that PRAISE does not depend on a single attribute type and instead benefits from a redundant set of complementary pedestrian cues. Among the categories, upper-body color and bag/accessory prompts have the largest effect: removing upper-body color reduces performance on both Market-1501 and RSTP-ReID, while removing bag/accessory prompts produces one of the strongest drops, especially on RSTP-ReID. This is consistent with standard ReID behavior, where upper clothing color is often the most visible short-term identity cue, and carried objects or accessories provide discriminative evidence when present. In contrast, removing hair has negligible effect, suggesting that hair cues are either less reliably visible or already captured indirectly by other appearance prompts. Importantly, removing age or gender does not degrade performance and in some cases slightly improves it, indicating that PRAISE does not rely on demographic-like attributes. This supports the use of a safer prompt bank focused primarily on clothing, color, and carried-object cues in applications surrounding sensitive private information.

Table 8. Comparison of TTA methods on I2I-ReID benchmarks using a source-trained CLIP-ReID [27] image encoder. All methods are evaluated under the same PRAISE-compatible cosine-retrieval protocol, with fixed gallery embeddings and no target identity labels.
<table><tr><td rowspan="3">TTA Method</td><td colspan="8">I2I-ReID</td></tr><tr><td colspan="4">Train: MSMT17</td><td colspan="4">Train: Market-1501</td></tr><tr><td>Test: MSMT17 Test: Market-1501 mAP</td><td>R1</td><td>mAP</td><td>R1</td><td>mAP</td><td>R1</td><td>mAP</td><td>Test: MSMT17 Test: Market-1501 R1</td></tr><tr><td>No-adapt (Source)</td><td>57.4</td><td>79.9</td><td>37.8</td><td>63.3</td><td>21.0</td><td>46.9</td><td>83.6</td><td>92.6</td></tr><tr><td>TENT [43] [ICLR&#x27;21]</td><td>55.2</td><td>75.4</td><td>35.2</td><td>60.5</td><td>18.6</td><td>44.2</td><td>82.1</td><td>91.1</td></tr><tr><td>SHOT [28] [ICML&#x27;20]</td><td>49.7</td><td>68.4</td><td>32.1</td><td>57.6</td><td>14.6</td><td>41.2</td><td>80.6</td><td>89.8</td></tr><tr><td>SAR [30] [ICLR&#x27;23]</td><td>55.3</td><td>76.8</td><td>37.6</td><td>63.2</td><td>19.7</td><td>45.2</td><td>83.1</td><td>92.4</td></tr><tr><td>ETA [29] [ICML&#x27;22]</td><td>55.2</td><td>76.8</td><td>37.1</td><td>62.8</td><td>20.1</td><td>45.8</td><td>82.7</td><td>92.2</td></tr><tr><td>DeYO [21] [ICLR&#x27;24]</td><td>55.3</td><td>76.9</td><td>37.9</td><td>63.3</td><td>19.8</td><td>41.3</td><td>83.8</td><td>92.7</td></tr><tr><td>TPT [38] [NeurIPS&#x27;22]</td><td>57.1</td><td>79.5</td><td>38.1</td><td>63.9</td><td>20.2</td><td>46.1</td><td>83.9</td><td>92.8</td></tr><tr><td>TDA [18] [CVPR·24]</td><td>55.7</td><td>78.8</td><td>37.9</td><td>63.5</td><td>20.1</td><td>45.8</td><td>83.6</td><td>92.7</td></tr><tr><td>TEMP [1] [IJCNN&#x27;24]</td><td>55.1</td><td>75.3</td><td>38.9</td><td>65.1</td><td>18.9</td><td>44.7</td><td>84.2</td><td>93.1</td></tr><tr><td>Norm [40] [ICLR&#x27;25]</td><td>57.2</td><td>79.5</td><td>41.1</td><td>65.8</td><td>20.7</td><td>46.4</td><td>85.0</td><td>93.4</td></tr><tr><td>PRAISE (Ours)</td><td>63.1</td><td>83.8</td><td>42.4</td><td>65.1</td><td>24.3</td><td>50.9</td><td>86.3</td><td>94.3</td></tr></table>

## E.3. Extended Performance on CLIP-ReID[27] and RDE [35]

Evaluation protocol. The absolute no-adapt scores in Table 8 and Table 9 should not be interpreted as reproduced numbers from the original papers. Our goal is to evaluate test-time adaptation under a common retrieval interface that is compatible with PRAISE. Specifically, PRAISE requires query, gallery, and text-prompt representations to lie in the same VLM embedding space so that prompt-induced directions can be used to construct ACS. We therefore use the CLIP image embedding with cosine similarity for retrieval, without the task-specific ReID head used in the original supervised evaluation. This changes the absolute no-adapt performance relative to the original paper, but it is applied uniformly to all methods in the table. Thus, the comparison measures how each TTA method modifies the same fixedgallery cosine retrieval geometry, rather than comparing against separately optimized task-specific heads.

How does PRAISE perform on CLIP-ReID [27] backbone. Table 8 evaluates PRAISE in a more strongly supervised setting using a source-trained CLIP-ReID backbone. Since the backbone has already been trained for identity discrimination, the source model provides a much stronger retrieval geometry than the generic PLIP setting, leaving less room for test-time adaptation. Under this setting, several generic TTA objectives degrade performance, especially TENT, SHOT, SAR, ETA, and DeYO. This behavior is expected in open-set ReID: entropy-style sharpening can reinforce the model’s current nearest-neighbor structure, which may already be highly confident but still biased by camera, pose, or dataset-specific appearance correlations. As a result, directly sharpening the retrieval distribution can over-concentrate probability mass around visually similar distractors rather than improving identity ranking. In contrast, PRAISE consistently improves the source-trained CLIP-ReID model across both source and cross-dataset evaluations. The gains are moderate compared to the genericbackbone setting, but this is expected because the supervised backbone already produces strong identity features. Importantly, PRAISE improves both source-domain performance and cross-dataset transfer, indicating that the ACS-guided adaptation does not simply compensate for a weak backbone. Instead, the attribute-conditioned subspace provides complementary semantic structure that can refine an already discriminative ReID embedding. Norm remains a strong post-hoc baseline and achieves the best Rank-1 in one cross-dataset case, but PRAISE obtains the best mAP in all settings and the best or second-best Rank-1 across the table. These results show that PRAISE remains useful even when applied on top of a competitive source-trained ReID backbone, addressing the concern that the improvements are specific to weaker or generic VLM representations.

Table 9. Comparison of TTA methods on T2I-ReID benchmarks using source-trained RDE [35] backbones. All methods are evaluated under the same PRAISE-compatible cosine-retrieval protocol, with fixed gallery embeddings and no target identity labels.
<table><tr><td rowspan="3">TTA Method</td><td colspan="6">1 T2I-ReID</td></tr><tr><td colspan="3">Train: CUHK-PEDES</td><td colspan="3">Train: RSTP-ReID</td></tr><tr><td>Test: CUHK mAP</td><td>R1 mAP</td><td>Test: RSTP R1</td><td>Test: CUHK mAP R1</td><td>mAP</td><td>Test: RSTP R1</td></tr><tr><td>No-adapt (Source) [NeurIPS&#x27;24]</td><td>64.2</td><td>72.7</td><td>41.2 54.9</td><td>33.5</td><td>36.8</td><td>47.9 62.1</td></tr><tr><td>TENT [43] [ICLR&#x27;21]</td><td>64.0</td><td>72.8</td><td>41.0 53.4</td><td>33.1</td><td>36.4</td><td>47.5 62.0</td></tr><tr><td>SHOT [28] [ICML&#x27;20]</td><td>64.5</td><td>72.9</td><td>41.3 54.7</td><td>33.8</td><td>37.1</td><td>48.3 62.5</td></tr><tr><td>TPT [38] [NeurIPS&#x27;22]</td><td>64.2</td><td>72.9</td><td>41.4 55.1</td><td>33.6</td><td>36.9</td><td>48.1 62.5</td></tr><tr><td>TDA [18] [CVPR&#x27;24]</td><td>59.1</td><td>68.7</td><td>38.4 51.2</td><td>31.2</td><td>35.7</td><td>46.1 61.1</td></tr><tr><td>WATT (S) [31] [NeurIPS&#x27;24]</td><td>59.8</td><td>69.2 39.1</td><td>51.7</td><td>31.3</td><td>35.8 46.9</td><td>61.4</td></tr><tr><td>WATT (P) [31] [NeurIPS&#x27;24]</td><td>61.2</td><td>70.2</td><td>39.9 52.2</td><td>31.6</td><td>36.0 46.4</td><td>61.8</td></tr><tr><td>CLIPArTT [8] [WACV&#x27;25]</td><td>59.6</td><td>69.1</td><td>38.7 51.1</td><td>31.1</td><td>35.5 46.2</td><td>60.9</td></tr><tr><td>CLIPTTA [20] [NeurIPS&#x27;25]</td><td>62.6</td><td>71.2</td><td>40.2 52.7</td><td>31.4</td><td>35.9 46.5</td><td>61.8</td></tr><tr><td>TCR [23] [ICLR&#x27;25]</td><td>64.6</td><td>73.1</td><td>42.5 55.7</td><td>34.1</td><td>37.3</td><td>48.5 63.1</td></tr><tr><td>PRAISE (Ours)</td><td>67.1</td><td>75.0</td><td>46.3 60.9</td><td>38.1</td><td>42.8</td><td>50.6 65.5</td></tr></table>

How does PRAISE perform on RDE [35] backbone. Table 9 evaluates PRAISE on source-trained RDE backbones under both source-domain and cross-dataset T2I-ReID settings. Compared with the PLIP setting in the main paper, the no-adapt RDE model already provides a much stronger retrieval representation, so the available room for test-time adaptation is smaller. This is reflected in the relatively mild gains of most generic TTA methods, and in several cases their performance degradation. In particular, classificationoriented or generic VLM adaptation methods are not necessarily aligned with the fixed-gallery ranking objective of T2I-ReID, and their sharpening or prompt-adaptation objectives can disrupt an already discriminative retrieval embedding.

Among prior methods, [23] is the strongest baseline and achieves the second-best result across all evaluated metrics. Nevertheless, PRAISE consistently improves over both the no-adapt RDE model and all TTA baselines. The gains are moderate on source-domain evaluation, where the supervised RDE representation is already well calibrated, and larger under cross-dataset transfer, where query-gallery alignment is more affected by dataset shift. This shows that ACSguided adaptation is not limited to weak or generic VLM backbones; it remains complementary to a source-trained T2I-ReID model by refining the retrieval geometry using query-conditioned attribute directions.

![](images/14ba701348685181139045e38b682518c148774d36758270a274d13a2d342dcb.jpg)  
Figure 8. Gallery-size scalability of PRAISE on MSMT17 with PLIP. Peak GPU-memory and adaptation-latency increases are reported relative to the 1k-gallery setting. The gallery is increased from 1,000 to the full 82,161 images while retaining the default adaptation batch size of 32. Even at the full gallery, peak memory increases by only 9.0%, while amortized per-query latency increases by 15.5%.

## E.4. Gallery-size scalability

To evaluate the scalability of PRAISE with respect to gallery size, we conduct an additional experiment on MSMT17 using the PLIP backbone. MSMT17 is particularly suitable for this analysis because its large gallery allows us to evaluate PRAISE over a substantially wider range of gallery sizes than the other benchmarks. We progressively increase the gallery from 1,000 images to the full 82,161 images while keeping the adaptation configuration fixed, including the default query batch size of 32.

Figure 8 reports the relative increase in peak GPU memory and adaptation latency with respect to the 1k-gallery setting. We normalize both quantities to their respective values at 1,000 gallery images to make their scaling behavior directly comparable despite having different physical units. Peak GPU memory increases only modestly as the gallery grows: from 4,179 MB at 1k images to 4,555 MB for the complete 82,161-image gallery, corresponding to an increase of approximately 9.0%. Runtime exhibits similarly gradual scaling. The measured adaptation throughput decreases from 3.95 to 3.42 batches/s between the 1k and full-gallery settings. Since each adaptation iteration processes a batch of 32 queries, this corresponds to an increase in amortized latency from approximately 7.91 to 9.14 ms/query, or 15.5%.

![](images/3047eb4d7a3cf16a944fcdfcdb479235a7f03465f6af5c6f72410a6489ace7d3.jpg)  
Figure 9. Sensitivity to initial retrieval quality on Market-1501. Queries are grouped by the initial rank of the first valid crosscamera match under the frozen PLIP baseline. Bars show the percentage of queries whose first-correct rank improves, remains unchanged, or degrades after PRAISE. The line reports mean ∆RR, where positive values indicate improved retrieval.

These results indicate that the gallery-dependent operations in PRAISE introduce relatively small additional memory and computational overhead even when the gallery size is increased by more than 80×. Importantly, these measurements correspond only to online test-time adaptation and retrieval; one-time preprocessing operations such as gallery feature extraction and prompt-bank construction are reported separately and are not included in this scalability analysis.

## E.5. Sensitivity to Initial Retrieval Quality

A potential failure mode of query-side adaptation is confirmation bias, where an incorrect initial neighborhood is reinforced during optimization. To study this effect, we stratify Market-1501 queries according to the initial rank of the first valid cross-camera match under the frozen PLIP baseline. We consider six groups: Rank-1, ranks 2–5, 6–10, 11–20, 21–50, and 51+. For each group, we record whether PRAISE improves, preserves, or degrades the rank of the first correct match. We also report the change in reciprocal rank,

$$
\Delta \mathrm { R R } = \frac { 1 } { r _ { \mathrm { P R A I S E } } } - \frac { 1 } { r _ { \mathrm { b a s e } } } ,
$$

which emphasizes changes near the top of the retrieval list while reducing the influence of very large absolute rank changes.

As shown in Figure 9, PRAISE is not restricted to queries whose correct identity already appears in the initial retrieval head. Excluding queries that are already correct at Rank-1, a majority of queries improve in every initial-rank group. When the first correct match lies outside the initial top-5 neighborhood, PRAISE improves its rank for 63.4% of queries. Among these cases, 30.2% are moved into the final top-5 and 13.3% reach Rank-1. The median rank gain is six positions and the mean reciprocal-rank change is +0.164. Positive mean ∆RR is observed across all non-Rank-1 groups.

Table 10. Sensitivity to the direct attribute-fusion weight $\lambda _ { \mathrm { { a t t } } }$ Results are reported as mAP/R1.
<table><tr><td> $\lambda _ { \mathrm { { a t t } } }$ </td><td>Market-1501</td><td>RSTPReID</td></tr><tr><td>0.0</td><td>36.8/65.0</td><td>23.5/35.4</td></tr><tr><td>0.1</td><td>39.2/60.3</td><td>24.5/32.9</td></tr><tr><td>0.2</td><td>38.9/60.1</td><td>24.3/32.5</td></tr><tr><td>0.5</td><td>37.5/58.7</td><td>23.7/30.8</td></tr><tr><td>1.0</td><td>35.1/54.5</td><td>23.1/28.3</td></tr></table>

![](images/51df92f34afd4cb05972fe7d87ef5f4c4b80388c479f67685493e9a582dabce5.jpg)  
Figure 10. NMI comparison across methods.

For queries already correct at Rank-1, PRAISE preserves Rank-1 in 83.9% of cases. Since these queries cannot improve further in first-match rank, any change can only preserve or degrade their position. Overall, these results show that PRAISE does not simply reinforce candidates already present in the initial top-k neighborhood and can recover a substantial fraction of initially poor retrievals. Degradation remains possible for unreliable neighborhoods, which we retain as a limitation of query-side test-time adaptation.

## E.6. Sensitivity to Direct Fusion Weight

To assess whether the weakness of direct attribute-score fusion is caused by the choice of fusion weight, we sweep $\lambda _ { \mathrm { a t t } }$ in Eq. (2) over {0, 0.1, 0.2, 0.5, 1.0}. Here, $\lambda _ { \mathrm { a t t } } = 0$ corresponds to identity-only adaptation without attribute-score fusion. As shown in Table 10, a small attribute contribution improves mAP, with $\lambda _ { \mathrm { a t t } } = 0 . 1$ reaching 39.2 on Market-1501 and 24.5 on RSTPReID. However, Rank-1 accuracy decreases from 65.0 to 60.3 and from 35.4 to 32.9, respectively. Increasing the fusion weight further degrades both metrics. Thus, direct attribute scores contain useful retrieva information, but no tested fusion weight consistently improves both mAP and Rank-1 over the identity-only setting. All settings also remain well below PRAISE, supporting the use of attributes as query-conditioned geometric guidance rather than as directly added retrieval scores.

## E.7. Impact on camera bias

We want to assess whether our approach effectively mitigates camera bias. To measure this, we employ a mutual information (MI) based evaluation following the protocol suggested in [40]. The rationale behind this experiment is that, if feature representations exhibit reduced dependence on camera labels, the mutual information between features and camera identifiers should correspondingly decrease. Figure 10 presents the outcomes of this analysis. The results indicate that both Camera Normalization [40] and PRAISE achieve the lowest normalized mutual information (NMI) scores, suggesting a comparatively weaker association between features and camera labels. Camera Normalization proves to be more effective when it comes to dealing with camera bias. While these findings do not constitute definitive evidence for the complete elimination of camera bias, they provide strong empirical support that our approach substantially attenuates such bias in the learned feature representations.

## E.8. Visualization: Failure Cases

Figure 11 presents qualitative I2I-ReID examples on Market-1501. In the success cases among the first five rows, the adapted rankings better prioritize the true identity by suppressing visually similar distractors. This is visible when the source model retrieves near-duplicates with mismatched clothing patterns or accessories, while the adapted predictions shift more correct matches into the top positions.

The failure cases suggest a characteristic failure mode of online adaptation with attribute guidance. When the initial attribute-driven affinity is incorrect, the subsequent sharpening can reinforce the wrong neighborhood and bias the update toward an incorrect set of gallery candidates. In (h) and (j), even though the query depicts a woman, the adapted retrieval settles on men with similar color cues. This is consistent with an erroneous early gender-related attribute estimate that redirects the ACS subspace toward spurious cues. Overall, the attribute directions are often beneficial, but they can occasionally sway retrieval by fixating on odd or weakly discriminative cues, leading to self-reinforced mistakes after adaptation.

Figure 12 shows analogous behavior for T2I-ReID on RSTPReid. Some challenging queries remain correct after adaptation. In (c) and (d), the descriptions are visually ambiguous and still the adapted predictions maintain or improve the correct matches, indicating that ACS can help focus similarity on text-consistent visual evidence rather than incidental distractors. Failure cases for T2I arise when adaptation aggravates noisy correspondence between the text and the image set. In (f) and (i), the adapted rankings drift further from the correct identity, which is consistent with sharpening amplifying an initially misaligned top set. Compared to I2I, we observe that gender-related errors are better contained in T2I, likely because explicit textual descriptors provide a stronger constraint that reduces the chance of flipping into an incorrect gender-driven neighborhood.

## F. Additional Implementation Details

## F.1. Sharpness-Aware and Reliable Adaptation (SAR)

• Idea. Apply entropy-minimization TTA only on reliable queries; use a SAM-style sharpness-aware update to avoid brittle minima.

• ReID posterior. Treat gallery indices as transient pseudo-classes: $\begin{array} { r l } { \mathcal { P } } & { { } = } \end{array}$ softmax $( \mathbf { A } _ { \mathrm { i d } } / \tau _ { \mathrm { i d } } )$ and $\begin{array} { r l } { H _ { i } } & { { } = } \end{array}$ $\begin{array} { r } { - \sum _ { j } \mathcal { P } [ i , j ] \log \mathcal { P } [ i , j ] } \end{array}$

• Efficiency. Compute entropy on a top-K candidate set per query: $\mathcal { N } _ { K } ( i ) = \mathrm { T o p K } ( \mathbf { A } _ { \mathrm { i d } } [ i , : ] )$ , renormalize on $\mathcal { N } _ { K } ( i )$

• Filtering. Update only if $H _ { i } < \gamma ;$ skip step if no samples pass.

• Optimization. Adapt only normalization affine parameters in $\mathcal { F } _ { v } ; \mathbf { S A M }$ perturbation $\epsilon ( \theta ) = \rho g / ( | | g | | _ { 2 } + \varepsilon )$ with two-pass update.

• Stability. Episodic resets per batch (or EMA-based recovery in non-episodic variants).

## F.2. Efficient Test-Time Adaptation (ETA/EATA)

• Idea. Entropy-based TTA with two filters: (i) keep confident queries; (ii) drop redundant queries whose posteriors look like a running mean.

• Posterior/entropy. Same retrieval posterior as SAR (optionally top-K renormalized) to compute $H _ { i }$

• Filter 1 (reliability). Keep ${ { \mathcal { T } } _ { 1 } } = \{ i : H _ { i } < \gamma \}$

• Filter 2 (redundancy). Maintain running mean posterior p¯; keep $\mathcal { T } _ { 2 } = \{ i \in \mathcal { T } _ { 1 } : | \cos ( \mathcal { P } [ i , : ] , \bar { \mathbf { p } } ) | < \delta \}$ (implemented on a shared candidate support when using top-K).

• Loss. Reweighted entropy: $w _ { i } ~ = ~ \mathrm { e x p } ( - ( H _ { i } - \gamma ) )$ $\begin{array} { r } { \mathcal { L } _ { \mathrm { E T A } } = \frac { 1 } { \sum _ { i \in \mathcal { Z } \mathrm { 2 } } w _ { i } } \sum _ { i \in \mathcal { T } _ { 2 } } w _ { i } H _ { i } . } \end{array}$

• What is tuned. Only normalization affine parameters in $\mathcal { F } _ { v }$ (episodic reset recommended).

## F.3. DeYO for ReID (Entropy + PLPD filtering)

• Idea. Keep confident queries (entropy filter) and further keep those whose top prediction is sensitive to a mild corruption (PLPD), which aims to discard unreliable gradients.

• Posterior. $\mathcal { P } = \mathrm { s o f t m a x } ( \mathbf { A } _ { \mathrm { i d } } / \tau _ { \mathrm { i d } } )$ (often computed on top-K indices for speed).

• Entropy filter. ${ \mathcal { T } } _ { \mathrm { e n t } } = \{ i : H _ { i } < \gamma _ { \mathrm { E n t } } \}$

• PLPD. Apply corruption T, compute ${ \mathcal { P } } ^ { \prime }$ , take $j _ { i } ^ { \star } ~ =$ arg max<sub>j</sub> $\mathcal { P } [ i , j ]$ , and $\mathrm { P L P D } _ { i } = \mathcal { P } [ i , j _ { i } ^ { \star } ] - \mathcal { P } ^ { \prime } [ i , j _ { i } ^ { \star } ]$

• PLPD filter. Keep $\mathcal { T } _ { \mathrm { p l p d } } = \{ i \in \mathcal { T } _ { \mathrm { e n t } } : \mathrm { P L P D } _ { i } > \gamma _ { \mathrm { P L P D } } \}$

• Loss. Reweighted entropy on $\mathcal { T } _ { \mathrm { p l p d } }$ with weights combining entropy-margin and PLPD terms.

• What is tuned. Only normalization affine parameters in ${ \mathcal { F } } _ { v } ;$ use mild $\tau$ to avoid uniform-collapse of $\mathcal { P } ^ { \prime }$

![](images/73a3794048e3d5327244f3a18762d6ded2dcd2670dd2a13dbe95d5cbe2d1c04c.jpg)  
Figure 11. Some success and failure cases for I2I-ReID from Market-1501 [55] dataset.

## F.4. Test-time Prompt Tuning (TPT) for ReID (I2I and T2I)

• Idea. Adapt an attribute prompt bank (not the encoders) by sharpening prompt posteriors across stochastic views, then use prompt-induced attribute similarity as an auxiliary retrieval signal.

• Prompts. Precompute $\mathbf { Z } _ { t } ^ { ( 0 ) } = \mathcal { F } _ { t } ( \mathcal { T } ( S ) )$ and optionally center with neutral $s _ { \mathrm { 0 } } ;$ cache centered/normalized bank.

• What is tuned. Prompt-bank adapter $\Delta ( \phi )$ (often lowrank), producing $\mathbf { Z } _ { t } ( \phi ) = \mathrm { N o r m } ( \widetilde { \mathbf { Z } } _ { t } ^ { ( 0 ) } + \Delta ( \phi ) )$

• TPT loss. For M views (image augs for I2I; caption templates/perturbations for T2I), form $\begin{array} { r l } { \mathbf { q } ^ { ( m ) } } & { { } \stackrel { } { = } } \end{array}$ softmax $( \mathbf { \bar { Z } } ^ { ( m ) } \bar { \mathbf { Z } } _ { t } ( \phi ) ^ { \top } / \tau _ { t } )$ and minimize $H ( \bar { \bf q } )$ with $\bar { \bf q } =$ $\begin{array} { r } { \frac { 1 } { M } \sum _ { m } \dot { \bf q } ^ { ( m ) } ( \mathrm { o p t i o n a l l y } } \end{array}$ keep low-entropy views only).

• Retrieval. Compute $\mathbf { A } _ { \mathrm { i d } } = \mathbf { Z } _ { q } \mathbf { Z } _ { q } ^ { \top }$ and attribute similarity: $\mathbf { A } _ { \mathrm { a t t } } = ( \mathrm { s o f t m a x } ( \mathbf { Z } _ { q } \mathbf { Z } _ { t } ( \phi ) ^ { \top } / \tau _ { t } ) ) ( \mathrm { s o f t m a x } ( \mathbf { Z } _ { g } \mathbf { Z } _ { t } ( \phi ) ^ { \top } / \tau _ { t } ) ) ^ { \top } .$

![](images/9b0697372011168c3c4395c081cde5af0451ec6e8528e5f6d659ce6095ac731a.jpg)  
Figure 12. Some success and failure cases for T2I-ReID from RSTPReID [56] dataset.

fuse $\mathbf { A } = \mathbf { A } _ { \mathrm { i d } } + \beta \mathbf { A } _ { \mathrm { a t t } } .$

• Stability/efficiency. Episodic reset of $\phi ;$ chunked $\mathbf { Z } _ { g } \mathbf { Z } _ { t } ( \phi ) ^ { \top }$ for large $N _ { g } .$

## F.5. Training-free Dynamic Adapter (TDA) for T2I ReID

• Idea. No gradients: maintain positive/negative caches of confident retrieval outcomes and correct logits by

similarity-weighted scatter add/subtract.

• Base logits. $\mathbf { s } = ( q G ^ { \top } ) / \tau _ { \mathrm { i d } }$ for text query features q and cached gallery G.

• Gating. Use (top-k) retrieval entropy to decide whether a query is reliable enough to write to the positive cache, or uncertain enough to write distractors to the negative cache.

• Logit correction. Add weighted votes to cached positive indices and suppress cached negative candidate sets using

$$
A ( \sin ) = \alpha \exp ( - \beta ( 1 - \sin ) ) .
$$

• Practical. Episodic clears caches per batch; non-episodic accumulates over stream; gate thresholds dominate behavior.

## F.6. TCR for T2I ReID (Query-shift adaptation with constraints)

• Idea. Adapt the text encoder (query side) using retrievalderived pseudo-positives and constraints that enforce source-like cross-modal gap and text uniformity.

• What is tuned. Only LayerNorm affine parameters in $\mathcal { F } _ { t }$ (image encoder and gallery are fixed).

• Pseudo-pairs. Retrieve top gallery item per caption and gather corresponding image features to form an in-batch $B \times B ^ { \prime }$ “refined” similarity.

• Reliability. Use entropy of the $B \times B$ refined posterior as confidence (low entropy = stable pseudo-positive).

• Constraints. Maintain a small queue of source-like items to estimate an adaptive entropy threshold and a target modality-gap statistic; optimize a weighted sum of noiserobust entropy, text-uniformity, and gap-matching terms.

## F.7. WATT for T2I ReID (Weight-averaged TTA)

• Idea. Run short TTA for multiple caption-template “views” and average the resulting adapted normalization parameters (and optionally ensemble embeddings) to reduce template sensitivity.

• What is tuned. Only normalization affine parameters in $\mathcal { F } _ { t } ;$ force dropout to eval mode.

• Objective. Define a posterior over gallery indices via p = softmax ${ \cdot } ( z _ { q } G ^ { \top } / \tau _ { \mathrm { i d } } )$ (or ACS variant) and optimize an unsupervised sharpening/soft-target loss.

• Averaging. Snapshot parameters per template and average (parallel) or apply sequentially; evaluate using the final averaged state.

## F.8. CLIPTTA-style TTA for T2I ReID (Contrastive with pseudo-positives)

• Idea. Build pseudo-positive image prototypes by retrieval and optimize a CLIP-style in-batch contrastive objective on caption–pseudoimage pairs.

• What is tuned. Only normalization affine parameters in $\mathcal { F } _ { t }$ (gallery and image encoder fixed).

• Pseudo-positives. From $S ~ = ~ z _ { q } G ^ { \top }$ , select top-1 or weighted top-k gallery features per caption (stop-grad through selection).

• Loss. Contrastive loss on the resulting $B \times B$ logits; fallback entropy-based objectives when $B = 1$ and optional transductive soft-target variants for stability.

• Optional memory. FIFO queue of past pseudo-positives can be re-encoded under current $\mathcal { F } _ { t }$ to stabilize updates