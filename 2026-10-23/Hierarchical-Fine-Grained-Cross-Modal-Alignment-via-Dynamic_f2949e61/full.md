# Hierarchical Fine-Grained Cross-Modal Alignment via Dynamic Granularity Discovery and Optimal Transport

Pengfeng Song<sup>1</sup>, Shaozhong Zhang<sup>1\*</sup>, Pengfei Shao<sup>1</sup>

<sup>1</sup>College of Information and Intelligent Engineering,Zhejiang Wanli University,Ningbo,315100,Zhejiang,China.

\*Corresponding author(s). E-mail(s): dlut z88@163.com; Contributing authors: sungkites@163.com;

## Abstract

Fine-grained image-text retrieval requires a model not only to assess global semantic consistency between an image and a text, but also to model visual evidence for local textual semantics, including objects, attributes, quantities, actions, and relations, so as to distinguish semantically similar candidates. Existing methods commonly use fixed image patches, detected regions, or text tokens as the basic matching units and then aggregate local interactions through max pooling, cross attention, or similarity accumulation. Such strategies remain limited in semantic unit construction and global matching consistency. This paper proposes a hierarchical fine-grained cross-modal alignment framework, termed HFCMA. The framework progressively transforms low-level patch and token representations into adaptive visual regions and textual phrases, and then performs structured regionphrase matching within a global retrieval scoring process. Specifically, HFCMA first adopts dynamic granularity discovery (DGD) to aggregate raw patches and tokens into semantically compact visual regions and textual phrases. It then formulates region-phrase matching as an entropy-regularized optimal transport problem, thereby estimating a structured alignment matrix under global marginal constraints. To reduce background noise and local semantic drift, HFCMA further introduces background debiasing, textual importance weighting, and prototype consistency regularization. Experiments on Flickr30K and MS-COCO show that HFCMA efectively combines dynamic semantic unit construction with OT-based structured matching under a dual-encoder retrieval framework. The resulting fine-grained score provides structured local alignment information for candidate ranking and improves the discrimination of subtle semantic diferences.

Keywords: Image-text retrieval, Fine-grained cross-modal alignment, Dynamic granularity discovery, Optimal transport, Local semantic representation

## 1 Introduction

Large-scale image-text contrastive learning has substantially advanced visual-language representation learning. CLIP and ALIGN demonstrate that global alignment over massive image-text pairs can learn transferable multimodal representations [1, 2]. Subsequent work, including ViLBERT, LXMERT, and VL-BERT, improves crossmodal interaction through fusion-style pre-training [3–5]. OSCAR, ALBEF, and X-VLM further introduce object semantics, momentum distillation, and multi-grained visual concepts to strengthen image-text alignment [6–8]. Meanwhile, ViLT, BLIP, BLIP-2, CoCa, and SigLIP extend visual-language pre-training from region-free modeling, generative learning, frozen-encoder bridging, and loss design [9–13]. These advances have moved image-text retrieval from handcrafted features and region matching toward scalable pre-training frameworks.

Despite this progress, fine-grained image-text retrieval remains a weak aspect of current visual-language models. A global vector similarity is eficient for large-scale retrieval, but it is often insuficient when semantically similar candidates must be distinguished. For example, two images may contain the same object categories but difer in number, color, action, or spatial relation. A sentence may also describe only part of a visual scene rather than the whole image. In these cases, a retrieval model needs more reliable cross-modal correspondences at the levels of regions, attributes, and relations. Prior methods such as SCAN, VSR, UNITER, FILIP, SPARC, NAAF, CHAN, MSRM, CORA, and LAPS show that local interaction is important for improving the detailed discrimination of image-text retrieval [14–23].

Existing fine-grained approaches often build local interaction over backboneproduced patches, detector-based regions, tokens, words, or language-guided selected patches, and then design attention, graph reasoning, or similarity aggregation strategies for cross-modal matching. Although these designs improve local discrimination, the construction of matching units is still frequently constrained by fixed tokenization, external detectors, or predefined selection rules. As a result, the semantic granularity of local units may not be fully adaptive to each image-text pair. However, in real imagetext pairs, a phrase may correspond to multiple adjacent patches, and a visual region may simultaneously carry object, attribute, and relation cues. If the matching units are semantically fragmented, subsequent local interaction can only partially remedy the mismatch. Recent studies on GroupViT, Slot Attention, TokenLearner, and Token Merging suggest that visual or textual representations can be adaptively organized into more compact semantic units [29–32]. This provides a diferent perspective for fine-grained retrieval: the matching granularity should not be treated merely as a by-product of the backbone, but should be explicitly modeled.

Local matching also requires global consistency. Max pooling and cross attention often select the most similar local counterpart independently for each element, which may overlook the joint constraint between a set of visual regions and a set of textual phrases. Optimal transport (OT) provides a principled formulation for structured set matching: given marginal distributions, all local correspondences are optimized jointly rather than independently [33, 34]. Although OT has been used in distribution alignment, word mover distance, domain adaptation, and vision-language prompt learning [35–39], how to combine dynamic semantic unit discovery with OT matching for fine-grained image-text retrieval remains under-explored.

Motivated by these observations, this paper proposes HFCMA, a hierarchical finegrained cross-modal alignment framework. HFCMA addresses two questions: how to adaptively construct suitable visual regions and textual phrases for each image-text sample, and how to establish globally consistent region-phrase matching among these dynamic semantic units. HFCMA adopts a frozen CLIP-style dual encoder as the feature backbone. It first uses a dynamic granularity discovery module to organize patches and tokens into semantically more complete local units. It then solves a structured transport matrix with entropy-regularized OT and combines the resulting fine-grained score with the global similarity for candidate ranking.

The main contributions are summarized as follows.

• We propose a dynamic granularity discovery mechanism for image-text retrieval, which bridges low-level patch/token representations and higher-level region-phrase semantic units, transforming fixed local matching into sample-adaptive region-phrase alignment.

• We formulate fine-grained cross-modal matching as entropy-regularized optimal transport over the discovered semantic units, enabling region-phrase correspondences to be jointly optimized under global marginal constraints.

• We introduce textual importance weighting, background debiasing, and prototype consistency regularization to improve the stability and discriminability of local semantic units.

• We conduct comprehensive experiments covering standard retrieval, module ablation, matching mechanism comparison, parameter sensitivity, inference eficiency, crossdataset transfer, and visualization, showing both the efectiveness and the applicable boundary of HFCMA.

## 2 Related Work

## 2.1 Visual-Language Pre-Training and Image-Text Retrieval

Visual-language pre-training has evolved from fusion-based interaction to dual-encoder contrastive learning. ViLBERT, LXMERT, and VL-BERT jointly encode visual regions and text with cross-modal Transformers and achieve early progress on vision-language tasks [3–5]. OSCAR, UNITER, ALBEF, and X-VLM improve cross-modal semantics through object tags, alignment objectives, momentum distillation, and multi-grained visual concepts [6–8, 17]. These methods provide strong interaction ability, but their pairwise fusion cost is high when many candidates must be evaluated.

CLIP and ALIGN encode images and texts into global vectors, making largescale retrieval eficient through ofline indexing and fast inner-product search [1, 2]. Later models, including BLIP, BLIP-2, CoCa, and SigLIP, strengthen visual-language foundation models through generative pre-training, frozen encoder bridging, contrastivegenerative modeling, and sigmoid loss [10–13]. However, the eficiency of dual encoders comes with a compression of local details. Global embeddings may not preserve object position, attribute modification, and relational cues completely. Therefore, this work focuses on enhancing fine-grained alignment while keeping the eficiency advantage of dual-encoder retrieval.

## 2.2 Fine-Grained Cross-Modal Alignment

Fine-grained image-text matching usually builds local interaction at region-word, patchtoken, or phrase-region levels. SCAN models region-word correspondence with stacked cross attention [14]. VSR and GraDual exploit visual or cross-modal graph reasoning for local relation modeling [15, 40]. UNITER introduces word-region alignment objectives [17], while FILIP pushes alignment down to the token level [16]. Recent methods further study negative-aware attention, hard alignment, multilateral semantic relations, compositional relation modeling, and language-guided patch slimming, such as NAAF, CHAN, MSRM, CORA, and LAPS [19–23]. Related studies also explore context-aware attention, graph matching, improved visual semantic reasoning, causal attention, and granularity-aware uncertain region modeling for fine-grained retrieval [24–28].

These studies demonstrate that fine-grained cues are important for distinguishing semantically similar image-text samples. Nevertheless, most methods still regard local matching units as fixed inputs and mainly focus on how to aggregate local similarities. Fine-grained alignment should also answer what units should be matched. If patches or tokens do not form complete semantic units, local interaction may amplify accidental high similarities among fragmented elements. Diferent from these methods, HFCMA treats local semantic unit construction as an explicit part of retrieval-oriented matching. Instead of only improving the aggregation of local similarities, it first discovers sampleadaptive region and phrase units and then aligns them through a globally constrained transport plan.

## 2.3 Dynamic Semantic Grouping and Adaptive Representation

Dynamic semantic grouping aims to form compact and semantically coherent representations from low-level features. Slot Attention learns object-centric representations through competitive soft assignment [30]. Set Transformer and Perceiver show that attention-based architectures can process variable-length sets and compress highdimensional inputs [41–43]. DETR decodes structured detection results with object queries [44]. GroupViT demonstrates that visual patches can gradually form semantic groups under text supervision [29]. TokenLearner and Token Merging reduce redundant representations through adaptive token selection or merging [31, 32].

Most of these methods are designed for detection, segmentation, video understanding, or eficient inference rather than image-text retrieval. These studies motivate adaptive semantic organization, but they are not designed to optimize dynamically grouped units for bidirectional image-text retrieval. HFCMA adapts this idea to retrieval by learning adaptive visual and textual semantic units under image-text supervision and aligning them with OT-based structured matching.

## 2.4 Optimal Transport and Structured Matching

Optimal transport minimizes the overall transportation cost between sets or distributions and can jointly optimize a matching matrix under marginal constraints. The entropy-regularized formulation and Sinkhorn iteration introduced by Cuturi make OT eficient and diferentiable for deep learning [33]. From Earth Mover’s Distance to Word Mover’s Distance and domain adaptation, OT has become a useful tool for structured alignment [36–39]. In visual-language learning, PLOT applies OT to prompt learning for cross-modal transfer alignment [35].

However, existing OT-based methods are usually applied over fixed-granularity features, where the transport objects are predefined patches, tokens, or prompts. HFCMA places dynamic granularity discovery before OT, so that the transport matrix acts on more compact visual regions and textual phrases. This design improves both the semantic quality of matching units and the global consistency of local alignment.

## 3 Method

## 3.1 Problem Definition and Framework

Given an image I and a text T, image-text retrieval learns a matching function S(I, T) such that true image-text pairs receive higher scores than mismatched pairs. A frozen image encoder $E _ { v }$ and a frozen text encoder $E _ { t }$ produce global features and low-level local features:

$$
\begin{array} { r } { \mathbf { g } ^ { v } , \mathbf { X } ^ { v } = E _ { v } ( I ) , \qquad \mathbf { g } ^ { t } , \mathbf { X } ^ { t } = E _ { t } ( T ) , } \end{array}\tag{1}
$$

where $\mathbf { X } ^ { v } = \{ \mathbf { x } _ { i } ^ { v } \} _ { i = 1 } ^ { N _ { v } }$ denotes patch-level visual features and $\mathbf { X } ^ { t } = \{ \mathbf { x } _ { j } ^ { t } \} _ { j = 1 } ^ { N _ { t } }$ denotes token-level textual features. Instead of directly matching $\mathbf { X } ^ { v }$ and $\mathbf { X } ^ { t }$ , HFCMA first organizes them into semantically more complete visual regions and textual phrases, and then solves a structured cross-modal matching problem. The CLIP-style image and text encoders are kept frozen throughout training. HFCMA only learns lightweight projection layers, dynamic granularity discovery modules, prototype parameters, and score fusion parameters. This preserves the eficiency of dual-encoder retrieval while adding structured local evidence for candidate ranking. In HFCMA, the hierarchical design refers to this progressive pipeline from low-level patch/token representations, to adaptive region-phrase semantic units, and then to structured OT matching combined with global retrieval scoring. Fig. 1 shows the overall framework.

The global similarity is computed as

$$
S _ { g } ( I , T ) = \cos ( \mathbf { g } ^ { v } , \mathbf { g } ^ { t } ) .\tag{2}
$$

The fine-grained module produces visual semantic units $\mathbf { V } = \{ \mathbf { v } _ { k } \} _ { k = 1 } ^ { K _ { v } }$ and textual semantic units $\mathbf { P } = \{ \mathbf { p } _ { l } \} _ { l = 1 } ^ { K _ { t } }$ , where $K _ { v }$ and $K _ { t }$ are the efective numbers of activated units. The final matching score is

$$
S ( I , T ) = S _ { g } ( I , T ) + \beta S _ { f g } ( I , T ) ,\tag{3}
$$

![](images/9332d34aa5a11062a772c7bf51d1718c45c37645c615dcc418590616c7790072.jpg)  
Fig. 1 Overall framework of HFCMA. A frozen CLIP-style dual encoder extracts visual patch and textual token features. Dynamic granularity discovery organizes them into adaptive visual and textual semantic units, followed by visual background debiasing and phrase importance weighting. The refined units are matched by entropy-regularized OT, and the resulting fine-grained score is combined with global CLIP similarity for final image-text retrieval.

where $S _ { f g }$ is the OT-based fine-grained score and $\beta$ controls the contribution of local alignment.

## 3.2 Dynamic Granularity Discovery

The goal of dynamic granularity discovery is to convert fixed patch/token sequences into semantic units closer to visual regions and textual phrases. For the visual branch, the model introduces $M _ { v }$ learnable visual slots $\mathbf { Q } ^ { v } = \bar { \{ \mathbf { q } _ { k } ^ { v } \} } _ { k = 1 } ^ { M _ { v } }$ . The soft assignment between patch i and visual slot k is computed by

$$
A _ { i k } ^ { v } = \mathrm { s o f t m a x } _ { k } \left( \frac { \phi ( \mathbf { x } _ { i } ^ { v } ) ^ { \top } \psi ( \mathbf { q } _ { k } ^ { v } ) } { \tau _ { v } } \right) ,\tag{4}
$$

where $\phi ( \cdot )$ and $\psi ( \cdot )$ are linear projections and $\tau _ { v }$ is the assignment temperature. The k-th visual semantic unit is obtained by weighted aggregation:

$$
{ \bf v } _ { k } = \frac { \sum _ { i = 1 } ^ { N _ { v } } A _ { i k } ^ { v } { \bf x } _ { i } ^ { v } } { \sum _ { i = 1 } ^ { N _ { v } } A _ { i k } ^ { v } + \epsilon _ { 0 } } .\tag{5}
$$

The text branch adopts the same mechanism with textual slots $\mathbf { Q } ^ { t } = \{ \mathbf { q } _ { l } ^ { t } \} _ { l = 1 } ^ { M _ { t } }$ and produces phrase units $\mathbf { P } = \{ \mathbf { p } _ { l } \} _ { l = 1 } ^ { K _ { t } }$ . Compared with individual tokens, phrase units can better represent entity combinations, attribute modifiers, and relational fragments, such as “two zebras”, “standing next $\mathrm { t o } ^ { \dag }$ , or “on a green field”. Fig. 2 illustrates the conversion from fixed-granularity inputs to dynamic semantic units.

![](images/b8545d91d3d25861c36f672667f04a2423e87d0cbae249e7778c72ffb94b0667.jpg)  
Fig. 2 Illustration of dynamic granularity discovery. Patch and token sequences are softly assigned to learnable visual and phrase slots, producing compact semantic units without requiring bounding-box or phrase-boundary annotations. Sparse and diverse regularization encourages sample-adaptive slo activation and reduces redundant local units.

To control the number of efective slots and reduce repeated semantic responses, a lightweight granularity regularizer is used:

$$
\mathcal { L } _ { g r a n } = \mathcal { L } _ { c m p } + \lambda _ { s } \mathcal { L } _ { s p a r s e } + \lambda _ { d } \mathcal { L } _ { d i v } .\tag{6}
$$

For modality $m \in \{ v , t \}$ , let $\mathbf { z } _ { i } ^ { m }$ denote an input local feature, ${ \bf u } _ { k } ^ { m }$ denote a discovered semantic unit, and $A _ { i k } ^ { m }$ denote the corresponding soft assignment. The compactness term is written as

$$
\mathcal { L } _ { c m p } = \sum _ { m } \frac { 1 } { N _ { m } } \sum _ { i = 1 } ^ { N _ { m } } \sum _ { k = 1 } ^ { M _ { m } } A _ { i k } ^ { m } \left. \mathbf { z } _ { i } ^ { m } - \mathbf { u } _ { k } ^ { m } \right. _ { 2 } ^ { 2 } .\tag{7}
$$

The slot activation is $\begin{array} { r } { \rho _ { k } ^ { m } = \frac { 1 } { { \cal N } _ { m } } \sum _ { i } A _ { i k } ^ { m } } \end{array}$ . The sparsity term penalizes high activation entropy,

$$
\mathcal { L } _ { s p a r s e } = - \sum _ { m } \sum _ { k = 1 } ^ { M _ { m } } \rho _ { k } ^ { m } \log ( \rho _ { k } ^ { m } + \epsilon _ { 0 } ) ,\tag{8}
$$

so that only content-relevant slots remain active. The diversity term reduces repeated slot responses:

$$
\mathcal { L } _ { d i v } = \sum _ { m } \frac { 1 } { M _ { m } ( M _ { m } - 1 ) } \sum _ { k \neq k ^ { \prime } } \operatorname* { m a x } \bigl ( 0 , \cos ( \mathbf { u } _ { k } ^ { m } , \mathbf { u } _ { k ^ { \prime } } ^ { m } ) - \eta \bigr ) .\tag{9}
$$

Together, these terms encourage compact, selective, and non-redundant semantic units. They serve the retrieval objective and do not require a slot to correspond exactly to a human-annotated region or phrase boundary.

![](images/22cdd68da53aacd712f0f5a37e56cce4f74b44d13d48caedb061ec88ca2e5ab1.jpg)  
Fig. 3 OT-based structured matching between discovered visual and textual slots. HFCMA builds a cross-modal cost matrix, estimates a transport plan with Sinkhorn iteration under visual and textual marginal constraints, and converts the transport cost into a fine-grained score for reranking together with the global similarity.

## 3.3 OT-based Structured Region-Phrase Matching

After obtaining visual units V and phrase units P, fine-grained matching is formulated as structured alignment between two semantic sets. The region-phrase cost matrix is defined by cosine distance:

$$
\begin{array} { r } { C _ { k l } = 1 - \cos ( { \mathbf v } _ { k } , { \mathbf p } _ { l } ) . } \end{array}\tag{10}
$$

HFCMA then solves an entropy-regularized OT problem:

$$
\Pi ^ { * } = \arg \operatorname* { m i n } _ { \Pi \in \mathcal { U } ( \mathbf { a } , \mathbf { b } ) } \langle \Pi , \mathbf { C } \rangle + \varepsilon \sum _ { k , l } \Pi _ { k l } ( \log \Pi _ { k l } - 1 ) ,\tag{11}
$$

where $\Pi _ { k l }$ denotes the soft matching strength between visual unit k and phrase unit l, a and b are marginal distributions, and ε controls the smoothness of the transport plan. The base setting normalizes the marginals over active semantic units, and the textual side can be further reweighted by phrase importance as described in Sec. 3.4. The problem is eficiently solved by Sinkhorn iteration, which alternately rescales rows and columns of the transport matrix to satisfy the marginal constraints.

The fine-grained score is defined as the negative transport cost:

$$
S _ { f g } ( I , T ) = - \sum _ { k = 1 } ^ { K _ { v } } \sum _ { l = 1 } ^ { K _ { t } } \Pi _ { k l } ^ { * } C _ { k l } .\tag{12}
$$

Fig. 3 shows the structured matching process. Unlike hard assignment, OT preserves uncertainty in local correspondence. Unlike max pooling, OT reduces the risk that multiple phrases compete for the same highly similar visual region. It is therefore more suitable for many-to-many fine-grained matching.

The main computational cost comes from Sinkhorn matching for candidate pairs. If the number of visual slots is $K _ { v } ,$ the number of text slots is $K _ { t } .$ , and the number of

Sinkhorn iterations is $\boldsymbol { n } _ { s } ,$ the major complexity for a candidate pair is

$$
\mathcal { O } ( K _ { v } K _ { t } n _ { s } ) , \qquad K _ { v } \ll N _ { v } , \ K _ { t } \ll N _ { t } .\tag{13}
$$

Because $K _ { v }$ and $K _ { t }$ are much smaller than the original patch/token lengths, the structured matching cost remains moderate and can be used as a lightweight reranking head after dual-encoder retrieval.

## 3.4 Debiasing, Importance Weighting, and Prototype Consistency

Background co-occurrence may create shortcut correlations in image-text matching. HFCMA therefore introduces a background debiasing term that penalizes high similarity between low-response visual slots and the textual global feature:

$$
\mathcal { L } _ { d e b i a s } = \frac { 1 } { B } \sum _ { b = 1 } ^ { B } \operatorname* { m a x } ( 0 , \cos ( \mathbf { v } _ { b g } ^ { ( b ) } , \mathbf { g } _ { t } ^ { ( b ) } ) - \delta ) ,\tag{14}
$$

where $\mathbf { v } _ { b g }$ is the aggregated low-response visual unit and δ is a margin.

Textual importance weighting further emphasizes entity, attribute, quantity, and relation phrases. For the l-th phrase unit, an importance gate is computed as

$$
\begin{array} { r } { w _ { l } ^ { t } = \sigma ( \mathbf w _ { a } ^ { \top } \mathbf p _ { l } + b _ { a } ) , \qquad \hat { \mathbf p } _ { l } = w _ { l } ^ { t } \mathbf p _ { l } , } \end{array}\tag{15}
$$

where $\sigma ( \cdot )$ is the sigmoid function. The weighted phrase representation $\hat { \mathbf { p } } _ { l }$ is used to construct the region-phrase cost matrix. The normalized weights can also define the textual marginal, $\begin{array} { r } { b _ { l } = w _ { l } ^ { t } / \sum _ { l ^ { \prime } } w _ { l ^ { \prime } } ^ { t } } \end{array}$ , so that informative phrases receive a larger matching budget in the transport plan.

To improve cross-sample stability, a prototype consistency loss, inspired by prototypical representation learning and balanced clustering [45, 46], is imposed over dynamic semantic units:

$$
\mathcal { L } _ { p r o t o } = - \frac { 1 } { M } \sum _ { m = 1 } ^ { M } \sum _ { r = 1 } ^ { R } \tilde { z } _ { m r } \log z _ { m r } .\tag{16}
$$

Here $z _ { m r }$ is the predicted prototype distribution and $\tilde { z } _ { m r }$ is a balanced stop-gradient target.

## 3.5 Training Objective and Inference

The full objective is

$$
\begin{array} { r } { \mathcal { L } = \mathcal { L } _ { c o n } + \lambda _ { g } \mathcal { L } _ { g r a n } + \lambda _ { b } \mathcal { L } _ { d e b i a s } + \lambda _ { p } \mathcal { L } _ { p r o t o } + \lambda _ { c } \mathcal { L } _ { c o n s } , } \end{array}\tag{17}
$$

where $\mathcal { L } _ { c o n }$ is the bidirectional image-text contrastive loss, and $\mathcal { L } _ { c o n s }$ maintains consistency between local semantic units and global representations.

During inference, the frozen dual encoder first retrieves a candidate set using the global similarity $S _ { g }$ . HFCMA then applies dynamic granularity discovery and OT-based region-phrase matching only to the candidate pairs, and ranks them by S(I, T). This design keeps the indexable dual-encoder retrieval pipeline while adding fine-grained local evidence at the reranking stage.

## 4 Experiments

## 4.1 Experimental Setup

Experiments are conducted on Flickr30K and MS-COCO [47, 48]. Flickr30K follows the 1K test protocol. MS-COCO is evaluated under both the 5K full test setting and the 1K five-fold average setting. MS-COCO contains diverse and sometimes semantically close captions, a property also discussed in extended caption similarity studies [49]. We report image-to-text (I2T) and text-to-image (T2I) Recall@1, Recall@5, Recall@10, and RSUM, which is the sum of the six recall metrics.

Unless otherwise specified, the backbone is OpenAI CLIP ViT-L/14 and is kept frozen. Only the lightweight HFCMA head is trained. The slot number denotes the number of dynamic semantic units used in DGD. For MS-COCO, the default setting uses 12 visual/text slots, 512 hidden dimensions, 48 prototypes, 15 Sinkhorn iterations, and an OT entropy coeficient of 0.07. For Flickr30K, it uses 8 visual/text slots, 256 hidden dimensions, 32 prototypes, 10 Sinkhorn iterations, and an OT entropy coeficient of 0.10. The experiments are conducted on an NVIDIA A100 GPU.

## 4.2 Main Retrieval Results

Table 1 shows that HFCMA achieves competitive retrieval performance with a frozen CLIP backbone and a lightweight matching head. On Flickr30K-1K, Full HFCMA obtains an RSUM of 535.78, while HFCMA-Fusion-Calib reaches 542.92. This result is clearly higher than classical and recent baselines such as VSE++, SCAN, VSRN, IMRAM, NAAF, CHAN, HREM, and EISIN, and is close to LAPS. On MS-COCO 1K, HFCMA-Fusion-Calib reaches an RSUM of 532.62, comparable to CHAN and higher than HREM and EISIN. On the larger MS-COCO 5K split, it achieves 442.27, higher than NAAF and EISIN and close to CHAN and HREM.

The advantage of HFCMA is particularly reflected in top-ranked retrieval. For Flickr30K-1K, HFCMA-Fusion-Calib achieves 87.0 I2T R@1, exceeding LAPS at 85.1. This indicates that dynamic semantic units and OT-based matching are efective for determining the most plausible caption when several candidates are globally similar. The reason is not merely an increase in local interaction, but a change in the matching objects and the matching constraint. DGD reduces fragmented patch/token noise by constructing compact region-phrase units, while OT prevents local phrases from independently selecting the same visually salient region. Such structured local evidence is especially useful for R@1, where only the first-ranked candidate matters.

Table 1 Main retrieval results on Flickr30K-1K and MS-COCO.
<table><tr><td rowspan=1 colspan=13>Benchmark   Method                       I2T R@1 R@5R@10T2I R@1 R@5R@10RSUM</td></tr><tr><td rowspan=1 colspan=13>VSE++ (2018)[50]                52.980.5 87.2     39.670.1 79.5 409.8</td></tr><tr><td rowspan=1 colspan=7>SCAN (2018) 14                  67.4</td><td rowspan=1 colspan=1>90.3</td><td rowspan=1 colspan=5>95.8     48.677.785.2 465.0</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=6>VSRN (2019) 15]                  71.3</td><td rowspan=1 colspan=1>90.6</td><td rowspan=1 colspan=5>96.0    54.7 81.8 88.2 482.6</td></tr><tr><td rowspan=6 colspan=4>NAAF (2022)Flickr30K-1KCHAN (2023)HREM (2023)LAPS (2024)EISIN (2025)</td><td rowspan=1 colspan=7>[51]               74.1</td><td rowspan=1 colspan=1>93.0</td><td rowspan=1 colspan=1>96.6    53.979.487.2 484.2</td></tr><tr><td rowspan=1 colspan=3>19                  81.9</td><td rowspan=1 colspan=1>96.1</td><td rowspan=1 colspan=5>98.3    61.085.390.6 513.2</td></tr><tr><td rowspan=1 colspan=3>[20]                 80.6</td><td rowspan=1 colspan=1>96.1</td><td rowspan=1 colspan=3>97.8    63.987.5</td><td rowspan=1 colspan=2>92.6 518.5</td></tr><tr><td rowspan=1 colspan=3>[52]                 83.3</td><td rowspan=1 colspan=1>96.0</td><td rowspan=1 colspan=3>98.1    63.5 87.1</td><td rowspan=1 colspan=2>92.4 520.4</td></tr><tr><td rowspan=1 colspan=3>[23]                  85.1</td><td rowspan=1 colspan=1>97.7</td><td rowspan=1 colspan=3>99.2    74.093.0</td><td rowspan=1 colspan=2>96.3 545.3</td></tr><tr><td rowspan=1 colspan=3>[53]                  83.3</td><td rowspan=1 colspan=1>96.2</td><td rowspan=1 colspan=3>98.3    63.687.3</td><td rowspan=1 colspan=2>92.0 520.7</td></tr><tr><td rowspan=2 colspan=7>Full HFCMÁ (ours)HFCMA-Fusion-Calib (ours)    87.0</td><td rowspan=1 colspan=4>(ours)86.6</td><td rowspan=1 colspan=1>98.3</td><td rowspan=1 colspan=1>99.4   67.6289.28 94.585535.78</td></tr><tr><td rowspan=1 colspan=1>98.0</td><td rowspan=1 colspan=5>99.4   72.2091.14 95.18542.92</td></tr><tr><td rowspan=1 colspan=4>VSE++ (2018) [</td><td rowspan=1 colspan=3>50]               64.6</td><td rowspan=1 colspan=1>90.0</td><td rowspan=1 colspan=5>95.7    52.084.392.0 478.6</td></tr><tr><td rowspan=1 colspan=4>SCAN (2018)</td><td rowspan=1 colspan=1>[14]</td><td rowspan=1 colspan=2>72.79</td><td rowspan=1 colspan=1>4.8</td><td rowspan=1 colspan=2>98.4    58.88</td><td rowspan=1 colspan=1>8.4</td><td rowspan=1 colspan=1>94.8</td><td rowspan=1 colspan=1>507.9</td></tr><tr><td rowspan=1 colspan=1></td><td></td><td rowspan=1 colspan=2>VSRN (2019)</td><td rowspan=1 colspan=1>15</td><td rowspan=1 colspan=2>76.2</td><td rowspan=1 colspan=1>94.8</td><td rowspan=1 colspan=2>98.2    62.88</td><td rowspan=1 colspan=1>9.7</td><td rowspan=1 colspan=1>95.1</td><td rowspan=1 colspan=1>516.8</td></tr><tr><td rowspan=1 colspan=4>IMRAM (2020)</td><td rowspan=1 colspan=3>[51]               76.7</td><td rowspan=1 colspan=1>95.6</td><td rowspan=1 colspan=2>98.5    61.7</td><td rowspan=1 colspan=1>89.1</td><td rowspan=1 colspan=1>95.0</td><td rowspan=1 colspan=1>516.6</td></tr><tr><td rowspan=2 colspan=4>NAAF (2022)MS-COCO 1KCHAN (2023)</td><td rowspan=1 colspan=3>19                  80.5</td><td rowspan=1 colspan=1>96.5</td><td rowspan=1 colspan=2>98.8    64.1</td><td rowspan=1 colspan=1>90.7</td><td rowspan=1 colspan=1>96.5</td><td rowspan=1 colspan=1>527.2</td></tr><tr><td rowspan=1 colspan=3>[20]                 81.4</td><td rowspan=1 colspan=1>96.9</td><td rowspan=1 colspan=2>98.9    66.5</td><td rowspan=1 colspan=1>92.1</td><td rowspan=1 colspan=1>96.7</td><td rowspan=1 colspan=1>532.6</td></tr><tr><td rowspan=2 colspan=3></td><td rowspan=1 colspan=1>HREM (2023)</td><td rowspan=1 colspan=3>[52                 81.1</td><td rowspan=1 colspan=1>96.6</td><td rowspan=1 colspan=2>98.9    66.1</td><td rowspan=1 colspan=1>91.6</td><td rowspan=1 colspan=1>96.5</td><td rowspan=1 colspan=1>530.7</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>LAPS (2024)</td><td rowspan=1 colspan=3>[23]                  84.1</td><td rowspan=1 colspan=1>97.4</td><td rowspan=1 colspan=1>99.2</td><td rowspan=1 colspan=1>72.1</td><td rowspan=1 colspan=1>93.9</td><td rowspan=1 colspan=1>97.4</td><td rowspan=1 colspan=1>544.1</td></tr><tr><td rowspan=1 colspan=1></td><td></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>EISIN (2025)</td><td rowspan=1 colspan=3>[53]                  79.3</td><td rowspan=1 colspan=1>95.9</td><td rowspan=1 colspan=1>98.5</td><td rowspan=1 colspan=1>63.9</td><td rowspan=1 colspan=1>90.8</td><td rowspan=1 colspan=1>95.8</td><td rowspan=1 colspan=1>524.1</td></tr><tr><td rowspan=1 colspan=1></td><td></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Full HFCMÁ (o</td><td rowspan=1 colspan=3>urs)             81.389</td><td rowspan=1 colspan=1>6.58 9</td><td rowspan=1 colspan=1>8.50</td><td rowspan=1 colspan=1>62.568</td><td rowspan=1 colspan=1>8.04 9</td><td rowspan=1 colspan=1>5.89</td><td rowspan=1 colspan=1>522.95</td></tr><tr><td rowspan=1 colspan=3></td><td rowspan=1 colspan=5>HFCMA-Fusion-Calib (ours)    81.7297.50 9</td><td rowspan=1 colspan=5>8.82   66.1891.52 96.885532.62</td></tr><tr><td rowspan=1 colspan=4>VSE++ (2018)</td><td rowspan=1 colspan=3>[50]                41.3</td><td rowspan=1 colspan=1>71.1</td><td rowspan=1 colspan=4>81.2    30.3 59.472.4</td><td rowspan=1 colspan=1>355.7</td></tr><tr><td rowspan=1 colspan=4>SCAN (2018)</td><td rowspan=1 colspan=1>14]</td><td rowspan=1 colspan=2>50.4</td><td rowspan=1 colspan=1>82.2</td><td rowspan=1 colspan=2>90.0    38.6</td><td rowspan=1 colspan=1>69.3</td><td rowspan=1 colspan=1>80.4</td><td rowspan=1 colspan=1>410.9</td></tr><tr><td rowspan=1 colspan=4>VSRN (2019)</td><td rowspan=1 colspan=1>15</td><td rowspan=1 colspan=2>53.08</td><td rowspan=1 colspan=1>1.1</td><td rowspan=1 colspan=2>89.4     40.5</td><td rowspan=1 colspan=1>70.6</td><td rowspan=1 colspan=1>81.1</td><td rowspan=1 colspan=1>415.7</td></tr><tr><td rowspan=1 colspan=4>IMRAM (2020)</td><td rowspan=1 colspan=1>[5</td><td rowspan=1 colspan=2>1]               53.78</td><td rowspan=1 colspan=1>3.2</td><td rowspan=1 colspan=2>91.0    39.7</td><td rowspan=1 colspan=1>69.1</td><td rowspan=1 colspan=1>79.8</td><td rowspan=1 colspan=1>416.5</td></tr><tr><td rowspan=2 colspan=4>NAAF (2022)MS-COCO 5KCHAN(2023)</td><td rowspan=1 colspan=1>[19]</td><td rowspan=1 colspan=2>58.98</td><td rowspan=1 colspan=1>5.2</td><td rowspan=1 colspan=2>92.0    42.57</td><td rowspan=1 colspan=1>0.9</td><td rowspan=1 colspan=1>81.4</td><td rowspan=1 colspan=1>430.9</td></tr><tr><td rowspan=1 colspan=3>[20]                 59.8</td><td rowspan=1 colspan=1>87.2</td><td rowspan=1 colspan=2>93.3    44.97</td><td rowspan=1 colspan=1>4.5</td><td rowspan=1 colspan=1>84.2</td><td rowspan=1 colspan=1>443.9</td></tr><tr><td rowspan=2 colspan=4>HREM(2023)LAPS (2024)</td><td rowspan=1 colspan=3>[52]                 62.3</td><td rowspan=1 colspan=1>87.6</td><td rowspan=1 colspan=2>93.4    43.9</td><td rowspan=1 colspan=1>73.6</td><td rowspan=1 colspan=1>83.3</td><td rowspan=1 colspan=1>444.1</td></tr><tr><td rowspan=1 colspan=1>LAPS (2024)</td><td rowspan=1 colspan=2>[23]</td><td rowspan=1 colspan=1>67.1</td><td rowspan=1 colspan=1>88.6</td><td rowspan=1 colspan=2>94.3     53.07</td><td rowspan=1 colspan=1>9.5</td><td rowspan=1 colspan=2>87.6 470.1</td></tr><tr><td rowspan=1 colspan=3></td><td rowspan=1 colspan=1>EISIN (2025)</td><td rowspan=1 colspan=2>[53]</td><td rowspan=1 colspan=1>59.6</td><td rowspan=1 colspan=6>84.4 91.2    42.072.382.5 432.1</td></tr><tr><td rowspan=1 colspan=4>Full HFCMÁ (o</td><td rowspan=1 colspan=2>urs)</td><td rowspan=1 colspan=7>64.7885.76 91.98   46.9370.72 79.42439.59</td></tr><tr><td rowspan=1 colspan=13>HFCMA-Fusion-Calib (ours)    64.7085.52 90.94   47.8672.3780.88442.27</td></tr></table>

## 4.3 Ablation Study

Table 2 verifies the contributions of the main components. On COCO 5K, DGD alone improves RSUM from 390.97 to 407.99, indicating that adaptive semantic units already provide useful local evidence. Adding OT matching further increases RSUM to 425.00, showing that structured region-phrase matching is more efective than merely constructing local units. Debiasing with textual importance weighting and prototype consistency continue to improve the result to 433.74 and 439.59. Similar trends can be observed on COCO 1K and Flickr30K, which confirms the complementarity between semantic unit discovery and structured matching.

## 4.4 Comparison of Fine-Grained Matching Mechanisms

To directly examine whether a fine-grained mechanism can push correct candidates toward the top of the ranking list, we evaluate candidate promotion. A frozen CLIP global similarity first produces the initial ranking. Queries whose true matches are already in Top-128 but not yet in Top-10, Top-5, or Top-1 are selected. Diferent local matching mechanisms are then used to rerank the candidate set. Table 3 and Fig. 4 show that OT matching consistently achieves the highest promotion success rate. The improvement is larger at Top-10 than at Top-1, which is expected because moving a candidate into the first position is more dificult and requires highly reliable global-local consistency.

Table 2 Core module ablation results.
<table><tr><td>Dataset</td><td>Variant</td><td>I2T R@1 T2I R@1 RSUM</td><td></td><td></td><td>∆RSUM</td></tr><tr><td></td><td>COCO 5K CLIP baseline</td><td>56.34</td><td>36.50</td><td>390.97</td><td>0.00</td></tr><tr><td>COCO 5K DGD only</td><td></td><td>59.29</td><td>40.15</td><td>407.99</td><td>+17.02</td></tr><tr><td></td><td>COCO 5K + OT matching</td><td>62.25</td><td>43.80</td><td>425.00</td><td>+34.03</td></tr><tr><td>COCO 5K</td><td>+ background debiasing + textual importance weighting</td><td>63.77</td><td>45.68</td><td>433.74</td><td>+42.77</td></tr><tr><td>COCO 5K</td><td>+ prototype consistency (Full HFCMA)</td><td>64.78</td><td>46.93</td><td>439.59</td><td>+48.62</td></tr><tr><td>COCO 1K CLIP baseline</td><td></td><td>74.22</td><td>55.44</td><td>493.13</td><td>0.00</td></tr><tr><td>COCO 1K DGD only</td><td></td><td>76.73</td><td>57.93</td><td>503.57</td><td>+10.44</td></tr><tr><td>COCO 1K + OT matching</td><td></td><td>79.23</td><td>60.42</td><td>514.00</td><td>+20.87</td></tr><tr><td>COCO 1K</td><td>+ background debiasing</td><td>80.52</td><td>61.71</td><td>519.37</td><td>+26.24</td></tr><tr><td>COCO 1K</td><td>+ textual importance weighting + prototype consistency (Full HFCMA)</td><td>81.38</td><td>62.56</td><td>522.95</td><td>+29.82</td></tr><tr><td>Flickr30K</td><td>CLIP baseline</td><td>85.20</td><td>64.92</td><td>525.94</td><td>0.00</td></tr><tr><td>Flickr30K</td><td>DGD only</td><td>85.69</td><td>65.86</td><td>529.38</td><td>+3.44</td></tr><tr><td>Flickr30K</td><td>+ OT matching</td><td>86.18</td><td>66.81</td><td>532.83</td><td>+6.89</td></tr><tr><td>Flickr30K</td><td>+ background debiasing</td><td>86.43</td><td>67.30</td><td>534.60</td><td>+8.66</td></tr><tr><td>Flickr30K</td><td>+ textual importance weighting + prototype consistency (Full HFCMA)</td><td>86.60</td><td>67.62</td><td>535.78</td><td>+9.84</td></tr></table>

Table 3 Candidate promotion success rates (%) of diferent fine-grained matching mechanisms under the Top-128 candidate pool.
<table><tr><td>Dataset</td><td>Mechanism</td><td>Top10</td><td>Top5</td><td>Top1</td><td>Gain</td></tr><tr><td>Flickr30K Flickr30K</td><td>Max pooling Smooth Chamfer</td><td>32.99 36.62</td><td>26.29 27.70</td><td>13.57</td><td>0.00</td></tr><tr><td>Flickr30K</td><td>Hard assignment</td><td>39.74</td><td>28.48</td><td>14.85 15.75</td><td>+3.63 +6.75</td></tr><tr><td>Flickr30K</td><td>OT matching</td><td>43.64</td><td>31.77</td><td>17.99</td><td>+10.65</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>MS-COCO 5K Max pooling</td><td>25.30</td><td>20.22</td><td>12.22</td><td>0.00</td></tr><tr><td></td><td>MS-COCO 5K Smooth Chamfer</td><td>27.93</td><td>21.76</td><td>13.02</td><td>+2.63</td></tr><tr><td></td><td></td><td>30.08</td><td></td><td></td><td></td></tr><tr><td></td><td>MS-COCO 5K Hard assignment</td><td></td><td>23.70</td><td>14.19</td><td>+4.78</td></tr><tr><td></td><td>MS-COCO 5K OT matching</td><td></td><td></td><td>33.71 26.54 15.39</td><td>+8.41</td></tr></table>

## 4.5 Parameter Sensitivity and Inference Eficiency

Table 4 summarizes the sensitivity of six key parameters. Here K denotes the number of dynamic semantic slots rather than a manually defined scale. It has a clear efective range: too few slots compress multiple entities and relations into a small number of units, while too many slots introduce redundancy. The prototype number controls the capacity of cross-sample semantic priors. The OT entropy coeficient shows a middlerange optimum: a too small value makes the transport plan close to hard matching, while a too large value over-smooths the alignment. Sinkhorn iteration becomes stable after about 10–15 steps. The fusion weight $\beta$ and the reranking size $K _ { r }$ reflect how HFCMA interacts with dual-encoder retrieval. A moderate $\beta$ and a candidate range of 128–256 balance ranking gain and computational cost.

![](images/17e661bb223edff199465a0ab1506e9040032972ae2654724dc7abe1dc0e234e.jpg)  
Fig. 4 Candidate promotion success-rate curves under diferent fine-grained matching mechanisms.

Table 4 Numerical summary of key hyper-parameter sensitivity.
<table><tr><td>Parameter</td><td>Search range</td><td></td><td>Points Recommended value</td><td>Best RSUM</td><td>Endpoint drop</td></tr><tr><td>Slot number K 1–48</td><td></td><td>14</td><td>Flickr 8, COCO 12 535.78 / 522.95</td><td>439.59</td><td>67.19 85.58</td></tr><tr><td>Prototype number P</td><td>4-768</td><td>14</td><td>Flickr 32, COCO 48</td><td>535.78 / 522.95 / 47.49 / 117.75 439.59</td><td></td></tr><tr><td>OT coefficient 0.001-1.00 ε</td><td></td><td>14</td><td>Flickr 0.10, COCO 0.07</td><td>535.78 / 522.95 439.59</td><td>112.55 / 105.68</td></tr><tr><td>Sinkhorn steps 1–50  $n _ { s }$ </td><td></td><td>14</td><td>10-15</td><td>535.78 / 522.95 / 439.59</td><td>52.55 / 0.98</td></tr><tr><td>Fusion weight  $\beta$ </td><td>0.00-1.50</td><td>16</td><td>0.10-0.20</td><td>535.78 / 522.95 / 48.62 / 192.75 439.59</td><td></td></tr><tr><td>Rerank size  $K _ { r }$ </td><td>4-1024</td><td>15</td><td>128-256</td><td>535.78 / 522.95 / 439.59</td><td>52.92 /  7.52</td></tr></table>

Table 5 Inference eficiency of HFCMA.
<table><tr><td>Stage</td><td>Candidates Params</td><td>(M)</td><td>Time </td><td>(ms) Memory (MB)</td></tr><tr><td>CLIP encoding</td><td>0</td><td>427.62</td><td>97.01 73.12</td><td>2299.90</td></tr><tr><td>HFCMA head</td><td>1</td><td>1.35</td><td>0.097</td><td>2299.90</td></tr><tr><td>OT solver</td><td>1</td><td>0.00</td><td>0.049</td><td></td></tr><tr><td>Top-64 rerank</td><td>64</td><td>1.35</td><td>6.24</td><td>2299.90</td></tr><tr><td>Top-128 rerank</td><td>128</td><td>1.35</td><td>12.47</td><td>2299.90</td></tr><tr><td>Top-256 rerank</td><td>256</td><td>1.35</td><td>24.94</td><td>2299.90</td></tr></table>

Table 5 shows that the additional parameters of the HFCMA head are only 1.35M. The per-candidate overhead of the OT solver is small. Even for Top-128 reranking, the extra inference time is about 12.47 ms under the tested setting. Therefore, HFCMA can act as a lightweight fine-grained ranking module after global retrieval.

![](images/2eea0d7a246a13d4d128e570d19a54c3b8b6b5514207029b1c660dbf6dfc2343.jpg)  
Fig. 5 Key parameter sensitivity analysis.

## 4.6 Cross-Dataset Generalization

Table 6 reports cross-dataset transfer results. Training on MS-COCO and testing on Flickr30K brings clear improvements over the CLIP baseline, indicating that the learned region-phrase alignment can transfer to a smaller retrieval benchmark. The reverse direction, from Flickr30K to MS-COCO 5K, is more challenging because the test collection is larger and contains more diverse image-text descriptions. Even in this setting, HFCMA-Fusion-Calib still improves RSUM from 390.86 to 414.44. These results suggest that HFCMA has a certain level of cross-dataset adaptability, while its efect is influenced by diferences between training and test distributions.

Table 6 Cross-dataset generalization analysis.
<table><tr><td>Training set</td><td>Test set</td><td>Method</td><td>I2T R@1</td><td>R@5</td><td>R@10</td><td>T2I R@1</td><td>R@5</td><td>R@10</td><td>RSUM</td></tr><tr><td>MS-COCO 5K</td><td>Flickr30K</td><td>CLIP baseline</td><td>85.10</td><td>97.40</td><td>99.20</td><td>64.90</td><td>87.28</td><td>92.00</td><td>525.88</td></tr><tr><td>MS-COCO 5K</td><td>Flickr30K</td><td>Full HFCMA</td><td>86.40</td><td>98.30</td><td>98.70</td><td>68.04</td><td>88.66</td><td>91.88</td><td>531.98</td></tr><tr><td>MS-COCO 5K</td><td>Flickr30K</td><td>HFCMA-Fusion-Calib</td><td>88.70</td><td>98.20</td><td>98.60</td><td>73.78</td><td>90.60</td><td>92.56</td><td>542.44</td></tr><tr><td>Flickr30K</td><td>MS-COCO 5K</td><td>CLIP baseline</td><td>56.36</td><td>79.26</td><td>86.58</td><td>36.51</td><td>61.02</td><td>71.13</td><td>390.86</td></tr><tr><td>Flickr30K</td><td>MS-COCO 5K</td><td>Full HFCMA</td><td>56.64</td><td>79.34</td><td>87.00</td><td>37.06</td><td>61.63</td><td>71.62</td><td>393.29</td></tr><tr><td>Flickr30K</td><td>MS-COCO 5K</td><td>HFCMA-Fusion-Calib</td><td>58.58</td><td>81.26</td><td>88.12</td><td>43.12</td><td>67.36</td><td>76.00</td><td>414.44</td></tr></table>

## 4.7 Visualization Analysis

Visualization is used to inspect whether HFCMA provides interpretable local evidence. Fig. 6 presents representative ranking-improvement cases. The corrected candidates are visually close to the original CLIP top results, but difer in attributes, quantities, local objects, or spatial relations. The dynamic semantic units make these discriminative fragments explicit, and the OT transport plan assigns them under a global matching constraint. This explains why HFCMA can move some correct candidates from the candidate pool to the top positions.

![](images/fe44c11e19f20010672976c253311c29b278008d057137b55f8a181d05b17ac0.jpg)  
Fig. 6 Visualization of ranking-improvement cases.

Fig. 7 shows the local alignment evidence behind the ranking changes. Active visual slots tend to cover semantically meaningful regions, while textual slots correspond to phrases involving objects, attributes, and relations. The transport matrix concentrates on plausible region-phrase pairs while retaining soft uncertainty, which is consistent with the role of OT as a structured many-to-many matching mechanism.

![](images/324ac176db4114eaea79f2d452e6eaee4195f9e863e8b15d38c63376b78a05f8.jpg)  
Fig. 7 Dynamic visual slots and local alignment evidence from the OT transport matrix.

Fig. 8 further reports ranking-degradation and failure cases. These examples usually contain crowded scenes, overlapping objects, weakly localized attributes, or nearduplicate candidates. In such cases, dynamic slots may respond to background or visually salient but semantically irrelevant regions, and the transport plan may amplify this imperfect local evidence. The failure analysis indicates that HFCMA is most reliable when the caption contains local cues that can be grounded by stable region-phrase correspondences.

## 5 Discussion

The experimental results indicate that the core issue in fine-grained image-text retrieval is not only how to aggregate local similarities, but also what local units should be matched. Fixed patches and tokens are computationally convenient, but they are not always consistent with semantic units. Dynamic granularity discovery organizes local features into compact visual regions and textual phrases, allowing subsequent matching to operate on units that better represent entities, attributes, and relations. OT matching further places these units under a unified transport constraint, avoiding independent local maximum selection. In this sense, the hierarchical nature of HFCMA lies in the progressive transition from low-level patch/token features through adaptive region-phrase semantic units to structured cross-modal matching combined with global retrieval scoring.

![](images/41a95931675d6d025ad0c679377e1b3b4816c3731a428c215026bb0e1a9ea037.jpg)  
Fig. 8 Visualization of ranking-degradation and failure cases.

The ablation results provide the most direct structural evidence: DGD brings stable improvement, OT strengthens structured matching, and auxiliary regularization further improves stability. Fusion and calibration are useful in practical retrieval because local fine-grained scores should cooperate with global semantic anchors rather than replace them. This point is important for large-scale datasets such as MS-COCO, where many candidates are visually or semantically close.

Several limitations remain. First, dynamic slots are not explicit detection boxes or segmentation masks, and semantic drift may occur in crowded scenes, occlusion, or near-duplicate objects. Second, calibration depends on the candidate distribution and should be fixed by validation protocols rather than repeatedly tuned on the test set. Third, diferent CLIP backbones may influence the gain of the fine-grained head. Finally, current experiments focus on English short-text retrieval benchmarks, whose brittleness and evaluation sensitivity have also been noted in recent benchmark analyses [54]. Future work may extend the framework to multilingual descriptions, long-form narratives, and noisy open-domain image-text collections.

## 6 Conclusion

This paper proposes HFCMA, a hierarchical fine-grained cross-modal alignment framework for image-text retrieval. The method adaptively constructs visual regions and textual phrases through dynamic granularity discovery, and solves a globally constrained region-phrase transport matrix with entropy-regularized optimal transport. Compared with methods that rely on fixed patches/tokens and heuristic local aggregation, HFCMA improves both the matching units and the matching mechanism.

Experiments on Flickr30K and MS-COCO show that HFCMA can enhance retrieval performance under a frozen CLIP dual-encoder framework with only a lightweight additional head. Ablation, matching mechanism comparison, parameter sensitivity, eficiency analysis, cross-dataset generalization, and visualization further verify the complementary role of dynamic semantic unit construction and OT-based structured matching. The main conclusion is that adaptive semantic unit construction combined with structured set matching provides a feasible path for improving fine-grained discrimination in image-text retrieval.

## Acknowledgements

This work was supported by the Zhejiang Provincial Natural Science Foundation of China (Grant No. LY23G020007).

## Competing Interests

The authors declare that they have no competing interests.

## Data and Code Availability

The Flickr30K and MS-COCO datasets used in this study are publicly available from their respective providers. The HFCMA code is available at https://github.com/ sungkites/HFCMA.

## References

[1] Radford A, Kim J W, Hallacy C, et al. Learning transferable visual models from natural language supervision. In: Proceedings of the International Conference on Machine Learning, 2021: 8748–8763.

[2] Jia C, Yang Y, Xia Y, et al. Scaling up visual and vision-language representation learning with noisy text supervision. In: Proceedings of the International Conference on Machine Learning, 2021.

[3] Lu J, Batra D, Parikh D, et al. ViLBERT: Pretraining task-agnostic visiolinguistic representations for vision-and-language tasks. In: Advances in Neural Information Processing Systems, 2019.

[4] Tan H, Bansal M. LXMERT: Learning cross-modality encoder representations from transformers. In: Proceedings of EMNLP-IJCNLP, 2019.

[5] Su W, Zhu X, Cao Y, et al. VL-BERT: Pre-training of generic visual-linguistic representations. In: International Conference on Learning Representations, 2020.

[6] Li X, Yin X, Li C, et al. OSCAR: Object-semantics aligned pre-training for visionlanguage tasks. In: Proceedings of the European Conference on Computer Vision, 2020: 121–137.

[7] Li J, Selvaraju R R, Gotmare A D, et al. Align before fuse: Vision and language representation learning with momentum distillation. In: Advances in Neural Information Processing Systems, 2021: 9694–9705.

[8] Zeng Y, Zhang X, Li H. Multi-grained vision language pre-training: Aligning texts with visual concepts. In: Proceedings of the International Conference on Machine Learning, 2022: 25994–26009.

[9] Kim W, Son B, Kim I. ViLT: Vision-and-language transformer without convolution or region supervision. In: Proceedings of the International Conference on Machine Learning, 2021: 5583–5594.

[10] Li J, Li D, Xiong C, et al. BLIP: Bootstrapping language-image pre-training for unified vision-language understanding and generation. In: Proceedings of the International Conference on Machine Learning, 2022: 12888–12900.

[11] Li J, Li D, Savarese S, et al. BLIP-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In: Proceedings of the International Conference on Machine Learning, 2023: 19730–19742.

[12] Yu J, Wang Z, Vasudevan V, et al. CoCa: Contrastive captioners are image-text foundation models. arXiv:2205.01917, 2022.

[13] Zhai X, Mustafa B, Kolesnikov A, et al. Sigmoid loss for language image pretraining. In: Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023: 11975–11986.

[14] Lee K H, Chen X, Hua G, et al. Stacked cross attention for image-text matching. In: Proceedings of the European Conference on Computer Vision, 2018: 201–216.

[15] Li K, Zhang Y, Li K, et al. Visual semantic reasoning for image-text matching. In: Proceedings of the IEEE/CVF International Conference on Computer Vision, 2019: 4654–4662.

[16] Yao L, Huang R, Hou L, et al. FILIP: Fine-grained interactive language-image pre-training. In: International Conference on Learning Representations, 2022.

[17] Chen Y C, Li L, Yu L, et al. UNITER: Universal image-text representation learning. In: Proceedings of the European Conference on Computer Vision, 2020: 104–120.

[18] Bica I, Ilic A, Bauer M, et al. Improving fine-grained understanding in image-text pre-training. In: Proceedings of the International Conference on Machine Learning, 2024: 3974–3995.

[19] Zhang K, Mao Z, Wang Q, et al. Negative-aware attention framework for imagetext matching. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022: 15661–15670.

[20] Pan Z, Wu F, Zhang B. Fine-grained image-text matching by cross-modal hard aligning network. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023: 19275–19284.

[21] Wang Z, Gao Z, Guo K, et al. Multilateral semantic relations modeling for image text retrieval. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023: 2830–2839.

[22] Pham K, Huynh C, Lim S N, et al. Composing object relations and attributes for image-text matching. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024: 14354–14363.

[23] Fu Z, Zhang L, Xia H, et al. Linguistic-aware patch slimming framework for finegrained cross-modal alignment. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024: 26307–26316.

[24] Zhang Q, Lei Z, Zhang Z, Li S Z. Context-aware attention network for image-text retrieval. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2020: 3536–3545.

[25] Cheng Y, Zhu X, Qian J, Wen F, Liu P. Cross-modal graph matching network for image-text retrieval. ACM Transactions on Multimedia Computing, Communications, and Applications, 2022, 18: 1–23.

[26] Yuan H, Huang Y, Zhang D, Chen Z, Cheng W, Wang L. VSR++: Improving visual semantic reasoning for fine-grained image-text matching. In: Proceedings of the International Conference on Pattern Recognition, 2021.

[27] Yang X, Zhang H, Qi G, et al. Causal attention for vision-language tasks. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021: 9847–9857.

[28] Liu J, Zhou H, Liu Y, et al. Cross-modal fine-grained alignment via granularityaware and region-uncertain modeling. In: Proceedings of the AAAI Conference on Artificial Intelligence, 2026.

[29] Xu J, De Mello S, Liu S, et al. GroupViT: Semantic segmentation emerges from text supervision. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022: 18134–18144.

[30] Locatello F, Weissenborn D, Unterthiner T, et al. Object-centric learning with slot attention. In: Advances in Neural Information Processing Systems, 2020.

[31] Ryoo M S, Piergiovanni A, Arnab A, et al. TokenLearner: Adaptive space-time tokenization for videos. In: Advances in Neural Information Processing Systems, 2021: 12786–12797.

[32] Bolya D, Fu C Y, Dai X, et al. Token merging: Your ViT but faster. In: International Conference on Learning Representations, 2023.

[33] Cuturi M. Sinkhorn distances: Lightspeed computation of optimal transport. In: Advances in Neural Information Processing Systems, 2013: 2292–2300.

[34] Peyre G, Cuturi M. Computational optimal transport. Foundations and Trends in Machine Learning, 2019, 11(5–6): 355–607.

[35] Chen G, Yao W, Song X, et al. PLOT: Prompt learning with optimal transport for vision-language models. In: International Conference on Learning Representations, 2023.

[36] Rubner Y, Tomasi C, Guibas L J. The earth mover’s distance as a metric for image retrieval. International Journal of Computer Vision, 2000, 40(2): 99–121.

[37] Kusner M J, Sun Y, Kolkin N I, et al. From word embeddings to document distances. In: Proceedings of the International Conference on Machine Learning, 2015: 957–966.

[38] Courty N, Flamary R, Tuia D, et al. Optimal transport for domain adaptation. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2017, 39(9): 1853–1865.

[39] Damodaran B B, Kellenberger B, Flamary R, et al. DeepJDOT: Deep joint distribution optimal transport for unsupervised domain adaptation. In: Proceedings of the European Conference on Computer Vision, 2018: 447–463.

[40] Long S, Han S C, Wan X, et al. GraDual: Graph-based dual-modal representation for image-text matching. In: Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, 2022: 3459–3468.

[41] Lee J, Lee Y, Kim J, et al. Set transformer: A framework for attention-based permutation-invariant neural networks. In: Proceedings of the International Conference on Machine Learning, 2019: 3744–3753.

[42] Jaegle A, Gimeno F, Brock A, et al. Perceiver: General perception with iterative attention. In: Proceedings of the International Conference on Machine Learning, 2021: 4651–4664.

[43] Jaegle A, Borgeaud S, Alayrac J B, et al. Perceiver IO: A general architecture for structured inputs and outputs. In: International Conference on Learning Representations, 2022.

[44] Carion N, Massa F, Synnaeve G, et al. End-to-end object detection with transformers. In: Proceedings of the European Conference on Computer Vision, 2020.

[45] Snell J, Swersky K, Zemel R S. Prototypical networks for few-shot learning. In: Advances in Neural Information Processing Systems, 2017.

[46] Caron M, Misra I, Mairal J, et al. Unsupervised learning of visual features by contrasting cluster assignments. In: Advances in Neural Information Processing Systems, 2020.

[47] Plummer B A, Wang L, Cervantes C M, et al. Flickr30k Entities: Collecting regionto-phrase correspondences for richer image-to-sentence models. In: Proceedings of the IEEE International Conference on Computer Vision, 2015.

[48] Lin T Y, Maire M, Belongie S, et al. Microsoft COCO: Common objects in context. In: Proceedings of the European Conference on Computer Vision, 2014.

[49] Parekh Z, Baldridge J, Cer D, et al. Crisscrossed captions: Extended intramodal and intermodal semantic similarity judgments for MS-COCO. Transactions of the Association for Computational Linguistics, 2021, 9: 318–333.

[50] Faghri F, Fleet D J, Kiros J R, et al. VSE++: Improving visual-semantic embeddings with hard negatives. In: Proceedings of the British Machine Vision Conference, 2018: 12.1–12.13.

[51] Chen H, Ding G, Liu X, Lin Z, Liu J, Han J. IMRAM: Iterative matching with recurrent attention memory for cross-modal image-text retrieval. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2020:

12655–12663.

[52] Fu Z, Mao Z, Song Y, Zhang Y. Learning semantic relationship among instances for image-text matching. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023: 15159–15168.

[53] Peng S, Wang Z, Liu J, Zhang C, Jia L. Fine-grained local and global semantic fusion for multimodal image-text retrieval. Big Data and Cognitive Computing, 2025, 9(3): 53.

[54] Hendriksen M, Odijk D, de Rijke M. Assessing brittleness of image-text retrieval benchmarks. arXiv:2407.15239, 2024.