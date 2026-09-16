# Semantic-Spatial Agreement Verification for Mitigating Object Hallucination in Multimodal Large Language Models

Ziheng Ren, Qian Gao, Jun Fan, Guohui Ding, Zhenyu Yang, and Yuteng Xiao

Abstract—Multimodal large language models generate naturallanguage responses from visual inputs, yet may mention objects absent from an image. In medication assistance, accessible perception, and environmental decision-making, such hallucinations can create real-world safety risks. We propose Semantic-Spatial Agreement Verification (SSAV), a training-free method for verifying object claims. A visually grounded claim should remain stable across semantically equivalent queries and repeatedly localize to the same image region. SSAV aggregates multiple prompts to estimate semantic support and reduce sensitivity to query wording. Query Induced Regional Verification (QIRV) combines cross-query region persistence, spatial overlap, and relative candidate dominance to identify isolated high responses and dispersed localizations. A geometric mean fuses semantic and spatial evidence, lowering the verification score when either branch lacks support. Experiments on three base models and multiple evaluation protocols show that SSAV effectively mitigates object hallucination. On LLaVA-1.5-7B, accuracy averaged across COCO, A-OKVQA, and GQA improves by 1.81 and 3.17 percentage points under POPE Popular and Adversarial, respectively, while CHAIRs decreases from 49.40% to 32.80%. These results show that cross-query semantic stability and regional consistency provide interpretable external visual evidence for object claims.

Index Terms—Multimodal large language models, object hallucination, open-vocabulary object detection, semantic consistency, spatial consistency, training-free inference.

## I. INTRODUCTION

ULTIMODAL large language models (MLLMs) typconnector, and an autoregressive language model. Visual encoders commonly adopt CLIP-family Vision Transformers [1], [2] to encode an input image into visual tokens. Crossmodal connectors then project, compress, and align visual features with the representation space of the language model through linear projections, multilayer perceptrons, or Q-Former architectures [3]. For example, LLaVA connects a CLIP visual encoder to a language model through a projection module [4], whereas BLIP-2, InstructBLIP, and MiniGPT-4 use a Q-Former to extract and compress visual information relevant to language generation [3], [5], [6]. These architectures support image-based question answering, image captioning, and multiturn interaction, yet responses can still mention objects absent from the image after visual encoding, cross-modal alignment, and autoregressive generation. Such object hallucinations make it difficult for users to judge whether a response is trustworthy. A visually impaired user may rely on a visual assistant to distinguish medicines with similar appearances but different indications or dosages; confusing one medicine with another, or falsely claiming that the target medicine is present, may cause medication errors. Likewise, missing an obstacle or inventing safety equipment in accessible navigation, industrial inspection, or emergency response may prompt unsafe actions [7]. Object hallucination is therefore not ordinary linguistic noise but a safety problem that can be amplified along an error chain from perception to judgment and action. Mitigating hallucination is consequently a prerequisite for deploying MLLMs in highreliability settings.

Existing research improves visual faithfulness through two main routes: training-stage alignment and inference-time intervention. Training-stage methods use hallucination-aware visual instruction data, hard negatives, or fine-grained image descriptions to supervise all or part of a model. Some further introduce vision–language alignment losses, contrastive objectives, reinforcement learning from human feedback, or direct preference optimization [8], [9]. These approaches reduce overreliance on linguistic co-occurrence patterns at the data and parameter levels, but the cost of high-quality data construction, preference annotation, and model updating limits rapid transfer. Inference-time methods instead reweight or select visual tokens, intervene in attention or hidden states, calibrate candidate distributions, perform contrastive decoding, or apply retrospective self-correction and post-generation revision [10]– [13]. They avoid retraining but often require access to specific network layers, attention structures, or autoregressive decoding interfaces; some also rely on additional sampling, backtracking, or multiple generation rounds. More importantly, internal confidence or attention strength does not inherently establish that an object is present. Once a model is influenced by linguistic priors, redistributing information within the same generation system may preserve biases that share the same source as the original error.

A complementary approach is to verify an object claim with an independent visual model. Open-vocabulary object detectors accept textual queries and return matched local regions with corresponding scores [14], [15], thereby providing a natural interface for object-level external verification. Compared with direct intervention in the internal state of an MLLM, this route reconnects a language response to localizable image regions and makes the visual basis of a judgment easier to interpret. However, an open-vocabulary detector is not an infallible fact judge. Different expressions of the same object can change text–region matching scores, making a decision based on one query and a fixed threshold sensitive to wording. Background textures, local contours, co-occurring objects, or excessively broad candidate boxes can also yield isolated high responses. Even when the maximum score exceeds a threshold, the response may not correspond to a stable object instance. A single-query maximum therefore represents only one local matching result and cannot establish whether an object claim is repeatedly supported by the same visual evidence.

We revisit external object verification from the perspective of cross-query consistency. If an object is present, semantically equivalent queries may produce different absolute scores, but they should provide reproducible semantic support overall and repeatedly localize to the same or highly overlapping regions. Conversely, a spurious response triggered by particular wording, background texture, or an incidental local feature is more likely to vary across prompts or disperse across unrelated regions. Object-presence evidence should therefore not be compressed into one scalar from a single detection. It should jointly capture semantic stability across prompts and spatial consistency across query-induced localizations. These two dimensions respectively ask whether different expressions consistently support the concept and whether such support originates from the same candidate instance, forming a stricter visual-verification condition than a single-query threshold.

Based on this insight, we propose Semantic-Spatial Agree ment Verification (SSAV). For each object claim, SSAV constructs four semantically equivalent queries and retains high-scoring candidates returned for each query by an openvocabulary detector. The semantic branch aggregates the strongest matching scores across queries to reduce response fluctuations caused by a single template. The spatial branch introduces Query-Induced Regional Verification (QIRV), which treats the preferred box of each query as a node, connects boxes according to intersection over union, and jointly measures candidate persistence, spatial overlap, and the dominance of the preferred candidate over the runner-up within the largest connected component. Semantic and spatial evidence are fused by a geometric mean so that a clear deficiency in either dimension lowers the final verification score. For objectexistence question answering, SSAV calibrates the original Yes– No logit margin with external evidence rather than replacing the MLLM prediction. For open-ended image description, the same evidence verifies object claims in the generated response, allowing a common verification mechanism to support different output forms.

The main contributions are as follows:

• We decompose unreliable single-query external object verification into prompt-induced variation in semantic responses and inconsistency among candidate regions, and formulate cross-query semantic-spatial agreement as evidence for visual support.

• We develop SSAV and its spatial branch QIRV, which jointly verify whether different queries point to the same object instance through candidate persistence, spatial overlap, and candidate dominance, complementing multiprompt semantic evidence.

We evaluate SSAV on three MLLMs with different architectures, LLaVA-1.5-7B, mPLUG-Owl2, and MiniGPT-4 Vicuna-13B. Closed-set object-existence judgments are evaluated on the COCO, A-OKVQA, and GQA sources of POPE under Random, Popular, and Adversarial sampling and on paired Yes– No decisions in MME-Existence. Sentence-level and instancelevel object hallucinations in open-ended descriptions are evaluated with CHAIR on 500 COCO images. Core-component ablations, QIRV evidence-correspondence shuffling, fusionstrength sensitivity, and qualitative cases further examine the mechanisms of multi-prompt semantic support and spatial verification. Across the three base models, SSAV improves objectexistence judgments and reduces both CHAIRs and CHAIRi. For LLaVA-1.5-7B, average POPE Accuracy improves by 1.81 and 3.17 percentage points under Popular and Adversarial sampling, respectively, while CHAIRs and CHAIRi decrease from 49.40% and 13.82% to 32.80% and 8.22%. These crossmodel, cross-source, cross-task, and mechanism-level results show that SSAV effectively mitigates object hallucination in MLLMs.

## II. RELATED WORK

## A. Object Hallucination and Mitigation in MLLMs

Object hallucination denotes an MLLM claim unsupported by the input image, including mentions of absent objects, incorrect categories, or attributes, counts, actions, and relations generated around nonexistent objects [16], [17]. Unlike factual hallucination in text-only settings, it can be checked directly against the image and reflects a mismatch between model output and visual evidence. Object hallucination is commonly associated with weakened transmission of fine-grained visual information and strong linguistic priors. Visual projection, compression, and insufficient cross-modal alignment may weaken object evidence, while previously generated text, frequent co-occurrences, and linguistic momentum increasingly influence autoregressive decoding. The resulting response can remain fluent despite lacking stable visual support [10], [18].

Existing mitigation methods use either training-stage optimization or training-free inference intervention. Training-stage approaches employ hallucination-aware data, hard negatives, regional descriptions, alignment or contrastive objectives, and human-feedback or preference optimization, but require additional annotation, model updates, and computation. At inference time, visual contrastive decoding compares perturbed visual inputs or generation distributions [10]; DeCo and DeGF apply distribution correction or generative feedback [12], [19]; PAI, CCA, and VASparse redistribute visual tokens or attention [20]–[22]; KVSmooth smooths key–value caches [23]; and CIPHER and TruthPrInt intervene along hallucination- or truthfulness-related directions [24], [25]. Other approaches construct contrastive views, compare instruction-conditioned distributions, introspect decoding states, or revise generated descriptions [26]–[30]. Together, these methods address underused visual information through decoding distributions, attention structures, and latent representations.

Training-stage optimization depends on data and model updates, whereas internal inference intervention usually requires access to specific attention layers, hidden states, visual tokens, or next-token distributions; some methods also perform multiple forward passes, repeated sampling, or backtracking. Moreover, internal confidence, attention magnitude, or a latent direction characterizes the model’s own response state and does not inherently establish that an object is present, while overly strong intervention may also suppress visually grounded objects. SSAV instead leaves the parameters, internal representations, and autoregressive decoding of the MLLM unchanged. After an object claim is formed, it calls an independent visual model to provide localizable evidence and calibrates the original decision margin in closed-set tasks. The external model does not replace the MLLM; it verifies the object claim with independent visual evidence, reducing dependence on a particular internal network structure.

## B. External Visual Verification and Post-Generation Revision

External visual verification separates factual checking from the original generator. A typical pipeline extracts objects, attributes, and relations from an answer or caption; invokes an object detector, visual question-answering model, or other visual tool; and then deletes, rewrites, or regenerates unsupported content. Compared with self-checking based only on language probabilities or generation confidence, external verification reconnects textual claims to image content and provides a more interpretable visual basis. Prior work has used objectwise questions, multi-round visual question answering, region cropping, and local re-identification. Other approaches use textto-image models to construct external feedback that guides correction during generation [12], [13].

External tools do not guarantee reliable factual decisions. Multi-round verification can inherit verifier bias and propagate early errors, while a generative visual question-answering verifier may itself hallucinate. Single-threshold post-processing can mistake detection confidence for object-presence probability and ignore query wording, category coverage, and candidateregion quality. The central issue is therefore how to construct stable, repeatable, and interpretable object-presence evidence.

SSAV verifies object claims with independent visual evidence without multi-round MLLM reasoning or rule-based deletion of low-confidence objects. It converts each claim into semantically equivalent queries and jointly analyzes detector scores and candidate regions. The semantic branch evaluates cross-query conceptual support, whereas the spatial branch determines whether responses repeatedly localize to the same region. External evidence is therefore derived from consistency across queries rather than a single model answer or local match.

C. Open-Vocabulary Object Detection and Regional Consistency

Open-vocabulary object detection jointly encodes image regions and text queries, directly aligning category names or natural-language phrases with candidate boxes [14], [15], [31], [32]. Representative developments further learn detectionspecific prompts or region–text representations, expand detector vocabularies with image-level supervision, reuse frozen vision– language models, and scale open-vocabulary self-training [33]– [38]. Unlike a detector with a fixed category set, it can respond to open-form object claims without retraining a classification head for each verification target, making it a natural interface for object-level verification of MLLM outputs. A text–region matching score describes how strongly a candidate region responds to the query, while the box localizes the spatial origin of that response. Open-vocabulary detection, visual grounding, and region cropping have consequently been used to check generated content and provide regional visual evidence [13].

Detection scores are generally intended for candidate ranking rather than as uniformly calibrated object-presence probabilities. Querying the same object with a class name, a noun phrase, or a scene-conditioned expression can substantially change the text embedding and matching score. Background texture, neighboring objects, and overly broad candidate boxes can also receive high responses. A single-query maximum therefore establishes only a strong match between one expression and one region; it does not show that the object claim remains supported as the expression changes. If the strongest responses to different queries fall in nonoverlapping regions, even individually high scores are difficult to interpret as repeated evidence for the same visual instance.

Regional consistency provides a complementary signal for separating stable instance responses from incidental local activations. When semantically equivalent queries repeatedly localize to the same object, their candidate boxes should form a region cluster with high query coverage and substantial internal overlap. Responses triggered by background content or queryspecific spurious matches are more likely to disperse across locations. The advantage of the highest-scoring candidate over the runner-up further indicates whether a query produces a clear local dominant response. QIRV therefore combines region persistence, spatial consistency, and within-query candidate dominance and fuses this spatial evidence with multi-prompt semantic support. Rather than asking whether one box exceeds a fixed threshold, it asks whether equivalent queries agree in both response strength and spatial instance.

## III. METHOD

## A. Problem Formulation and Method Overview

Given an input image I, a question q, and a frozen multimodal large language model M, the base model generates $y \ = \ M ( I , q )$ . We focus on claims c in the response that can be reduced to whether an object exists, and seek to determine whether each claim has sufficient and stable image evidence without updating the base model. In closed-set object existence question answering, the target object is obtained directly from the question. In open-ended image description, a set of object claims C(y) is extracted from the generated text using a predefined category vocabulary and synonym mapping. Claim extraction identifies verification targets only and does not participate in evidence scoring.

![](images/80cbc2bc32f38017c12fba119e68023e60f4b5f5943d6d7c69acdfa3eb97e8c2.jpg)  
Fig. 1. Overview of Semantic-Spatial Agreement Verification (SSAV). An object claim is extracted from the MLLM response and converted into four semantically equivalent queries. A frozen open-vocabulary verifier returns candidate regions for each query. SSAV combines multi-query semantic support with QIRV spatial evidence, which measures region persistence, spatial overlap, and candidate dominance. The fused evidence calibrates the Yes–No margin in closed-set VQA or filters unsupported object claims in open-ended captioning.

To obtain visual evidence independent of the internal state of the base model, we use a frozen open-vocabulary detector D [14] as an auxiliary verifier. Given image I and an object query, D returns candidate regions and their text–region matching scores. The external detector neither replaces the base model nor turns a single detection maximum directly into an object-presence probability. Instead, it provides comparable regional responses that allow a claim to be evaluated along two dimensions: whether support persists across expressions and whether different expressions point to the same instance.

The SSAV pipeline is shown in Fig. 1. First, the multiprompt semantic-support branch constructs fixed semantically equivalent queries for claim c and aggregates the highest matching score from each query, reducing variation caused by one wording. Second, Query-Induced Regional Verification (QIRV) builds a region-relation graph from the preferred boxes produced by different queries and tests whether the responses bind to the same visual instance through candidate persistence, spatial consistency, and within-query candidate dominance. The two forms of evidence are fused by a geometric mean into an object-level verification score $S _ { \mathrm { s s a v } }$ . Closed-set object-existence question answering and open-ended image description share this evidence-extraction process but use different decision rules: the former calibrates the base model’s Yes–No logit margin, whereas the latter filters unsupported object claims with a claim-level threshold.

Figure 2 presents the empirical observations motivating the method. Aggregating equivalent queries improves the stability of object-presence evidence; present-object claims more often localize repeatedly to the same region across queries; and QIRV discriminates object presence better than any individual spatial component. These observations motivate the multi-prompt semantic-support and spatial-instance verification designs described below.

![](images/e30da523c091856c4330c173af42e31977922faf095e4c2ec20c7993014c034a.jpg)

Equivalent wording changes single-query evidence  
![](images/c82252b967514a594933d1ad847da0579dab77e766cee5210bf16b9a174cb20e.jpg)

b Region persistence across four queries  
![](images/b4bfef237d94e34506e8e185633e5280a1fd9574c9500de93beffafc31fd5460.jpg)

c Spatial evidence and its integration  
![](images/56fd773f01dd07febd04d3a68cfd11b18551fe8aabb6dfc3cac5139370700b0c.jpg)  
Fig. 2. Empirical motivation for semantic-spatial verification in SSAV. (a) Object-presence AUROC across different numbers of semantically equivalent queries; the line and shaded region show the mean and range across prompt combinations. (b) Distribution of the largest region-cluster size for presentand absent-object claims. (c) Object-presence AUROC of QIRV and its three spatial components, with image-clustered 95% bootstrap confidence intervals.

## B. Multi-Prompt Semantic Support

The matching score of an open-vocabulary detector is affected by query wording. Even when several text queries express the same object concept, they may activate regional responses of different strengths. Templates can also introduce different linguistic and contextual biases, making a high response to one query insufficient as stable visual support. Verification based only on a class-name query can therefore depend excessively on one expression [39]. To reduce this prompt sensitivity, we construct a query set $\mathcal { P } ( c )$ containing $K = 4$ fixed templates for each object claim c:

$$
\begin{array} { r } { \mathcal { P } ( c ) = \{ c , \mathrm { ~ a ~ p h o t o ~ o f ~ } c , \mathrm { ~ a n ~ i m a g e ~ c o n t a i n i n g ~ } c , } \\ { c \mathrm { ~ i n ~ t h e ~ s c e n e } \} . \qquad \quad } \end{array}\tag{1}
$$

All templates are fixed before testing and are not selected for a dataset, base model, or individual sample during evaluation. To prevent multiple text queries from interacting within one detector call, each query is submitted independently. For the jth query $p _ { j } \in \mathcal { P } ( c )$ , the detector retains the top $N$ candidate regions in descending order of matching score:

$$
\begin{array} { r } { D ( I , p _ { j } ) = \left\{ \left( b _ { j } ^ { ( n ) } , s _ { j } ^ { ( n ) } \right) \right\} _ { n = 1 } ^ { N } , \quad s _ { j } ^ { ( 1 ) } \geq s _ { j } ^ { ( 2 ) } \geq \cdots \geq s _ { j } ^ { ( N ) } . } \end{array}\tag{2}
$$

Here, $b _ { j } ^ { ( n ) }$ is the nth candidate box and $s _ { j } ^ { ( n ) } \in [ 0 , 1 ]$ is its text–region matching score. The semantic branch averages the preferred candidate score from each query to obtain multiprompt semantic support:

$$
S _ { \mathrm { s e m } } ( c ) = \frac { 1 } { K } \sum _ { j = 1 } ^ { K } s _ { j } ^ { ( 1 ) } .\tag{3}
$$

$S _ { \mathrm { s e m } }$ does not require identical scores across queries. Instead, cross-query aggregation reduces the influence of an anomalous response from a single template. Support is retained when a claim receives strong responses under several equivalent expressions, whereas a score triggered only by particular wording is attenuated after aggregation. Semantic support, however, captures only cross-query response strength and cannot determine whether the responses originate from the same object, different objects, or a broad background region. We therefore verify spatial instance consistency using the relationships among candidate boxes.

## C. Query-Induced Regional Verification

A high semantic score does not necessarily provide reliable instance-level evidence. Different queries may yield high scores at different image locations or jointly activate a broad background region, and averaging their scores ignores these localization differences. QIRV therefore treats the preferred candidate from each query as a query-induced region hypothesis and tests whether the resulting regions consistently correspond to the same spatial instance.

Specifically, QIRV takes the K preferred boxes $\{ b _ { j } ^ { ( 1 ) } \} _ { j = 1 } ^ { K }$ <sub>1</sub> as nodes and constructs an undirected graph $G = ( V , E )$ . For any two queries i and $j ,$ an edge is added if the intersection over union of their preferred boxes is no smaller than threshold $\eta \colon$

$$
\begin{array} { r } { ( i , j ) \in E \iff \mathrm { I o U } \Big ( b _ { i } ^ { ( 1 ) } , b _ { j } ^ { ( 1 ) } \Big ) \geq \eta . } \end{array}\tag{4}
$$

We set $\eta = 0 . 5$ and denote the largest connected component of $G$ by ${ \mathcal { C } } ^ { \star }$ . Connected components organize regional responses from different queries into candidate instance clusters, and the largest component represents the spatial instance supported by the greatest number of queries. If several queries repeatedly point to the same object, their preferred boxes should concentrate in one connected component and exhibit substantial overlap. QIRV consequently computes candidate persistence, spatial consistency, and within-query candidate dominance in sequence.

Candidate persistence measures the proportion of queries covered by the largest connected component:

$$
\rho = \frac { | \mathcal { C } ^ { \star } | } { K } .\tag{5}
$$

When all queries point to mutually connected regions, $\rho = 1 ;$ when only a few queries share a spatial response, $\rho$ decreases. Thus, $\rho$ measures the persistence of a candidate instance under changes in query wording rather than the confidence of a single box.

Candidate persistence counts the queries covered by the component but does not capture its geometric compactness. When $| { \mathcal { C } } ^ { \star } | > 1 ,$ , we define spatial consistency u as the mean pairwise IoU among all boxes in the largest connected component:

$$
u = \frac { 2 } { | \mathscr { C } ^ { \star } | ( | \mathscr { C } ^ { \star } | - 1 ) } \sum _ { i , j \in \mathscr { C } ^ { \star } } \mathrm { I o U } \Big ( b _ { i } ^ { ( 1 ) } , b _ { j } ^ { ( 1 ) } \Big ) .\tag{6}
$$

If the largest connected component contains only one node, no box pair is available. We set $u = 1 / K$ to retain limited support for an isolated response while preventing it from being interpreted as sufficient spatial consistency. Hence, $\rho$ and u respectively describe how many queries support one candidate instance and how tightly those supports align in space.

Even when preferred boxes from several queries overlap, the detector may assign similar scores to multiple regions within each query. The preferred candidate then lacks a clear advantage and remains spatially ambiguous. To quantify this local competition, QIRV compares the preferred and runner-up candidates for each query. Let their scores be $s _ { j } ^ { ( 1 ) }$ and $s _ { j } ^ { ( 2 ) } \big |$ within-query candidate dominance is defined as

$$
d _ { j } = \sigma \Big ( \log \mathrm { i t } \Big ( s _ { j } ^ { ( 1 ) } \Big ) - \log \mathrm { i t } \Big ( s _ { j } ^ { ( 2 ) } \Big ) \Big ) ,\tag{7}
$$

where $\sigma ( \cdot )$ is the sigmoid function. Detection scores are clipped to $[ \varepsilon , 1 - \varepsilon ]$ before the logit to avoid numerical overflow. $d _ { j }$ increases when the preferred candidate clearly exceeds the runner-up and approaches a neutral level when their scores are similar. Subsequent aggregation uses only nodes in $\mathcal { C ^ { \star } }$ and the corresponding $d _ { j }$ , keeping within-query competition aligned with the selected instance cluster.

Combining these factors, QIRV spatial evidence is

$$
S _ { \mathrm { s p a } } ( c ) = \rho u \frac { 1 } { | \mathcal { C } ^ { \star } | } \sum _ { j \in \mathcal { C } ^ { \star } } d _ { j } .\tag{8}
$$

This product jointly captures cross-query instance coverage, geometric overlap within the component, and relative withinquery dominance. An isolated high score is jointly limited by $\rho$ and the single-node value of $u ;$ dispersed localization reduces $u ;$ and ambiguity between the preferred and runner-up candidates reduces $d _ { j }$ . QIRV does not predict the category again. Instead, it uses the candidate structure already produced by the detector to test whether a category response binds stably to a specific visual instance.

## D. Semantic-Spatial Evidence Fusion

Multi-prompt semantic support measures whether an object concept repeatedly receives strong responses under equivalent expressions, whereas QIRV measures whether those responses point to a stable spatial instance. The two forms of evidence correspond to response strength and instance consistency; a deficiency in either may indicate inadequate visual support. Given this complementarity, we use their geometric mean as the final verification score:

$$
S _ { \mathrm { s s a v } } ( c ) = \sqrt { S _ { \mathrm { s e m } } ( c ) S _ { \mathrm { s p a } } ( c ) } .\tag{9}
$$

The geometric mean treats the branches symmetrically and lets the lower value directly constrain the result. If queries receive high scores but localize to dispersed regions, $S _ { \mathrm { s p a } }$ reduces the fused score. If boxes overlap strongly but the overall matching strength is weak, $S _ { \mathrm { s e m } }$ likewise limits the result. A high $ { S _ { \mathrm { s s a v } } }$ therefore requires both cross-query semantic support and spatial-instance support. The fusion introduces no learnable parameters, and the resulting object-level evidence is passed to the corresponding task-inference procedure.

## E. Task Inference With SSAV Evidence

SSAV produces a unified visual-evidence score $S _ { \mathrm { s s a v } }$ for each object claim. To adapt this evidence to different output forms, we calibrate the base model’s Yes–No logit margin in closed-set object-existence question answering and filter object claims from generated text in open-ended image description. Both tasks share the same evidence-extraction process and differ only in how the evidence is used during inference.

1) Closed-Set Object-Existence Question Answering: Let the logits assigned by the base model to the Yes and No answer tokens be $\ell _ { \mathrm { y e s } }$ and $\ell _ { \mathrm { n o } }$ . The original decision margin is

$$
m _ { \mathrm { b a s e } } = \ell _ { \mathrm { y e s } } - \ell _ { \mathrm { n o } } .\tag{10}
$$

SSAV does not overwrite the base prediction with the external detector output. Object-level evidence instead calibrates the original margin:

$$
m _ { \mathrm { s s a v } } = m _ { \mathrm { b a s e } } + \lambda _ { \mathrm { s s a v } } \left( S _ { \mathrm { s s a v } } - \tau _ { \mathrm { s s a v } } \right) , \quad \widehat { y } = \mathbb { I } [ m _ { \mathrm { s s a v } } \geq 0 ] .\tag{11}
$$

Here, $\lambda _ { \mathrm { s s a v } } \geq 0$ controls the strength of external-evidence calibration, and $\tau _ { \mathrm { s s a v } }$ is the center threshold of the external evidence. When $S _ { \mathrm { s s a v } } > \tau _ { \mathrm { s s a v } }$ , the calibration term favors Yes; when $S _ { \mathrm { s s a v } } < \tau _ { \mathrm { s s a v } }$ , it favors No. The final decision jointly depends on the original base-model margin and the strength of the external evidence.

2) Open-Ended Image Description: Given the base-model output y, we first extract the object-claim set $\mathcal { C } ( y )$ and compute $S _ { \mathrm { s s a v } } ( c )$ for every claim. Because generated text provides no Yes–No margin corresponding to an individual claim, a frozen claim-level threshold $\tau _ { \mathrm { c a p } }$ partitions claims into retained and rejected sets:

$$
\begin{array} { r } { \mathcal { C } _ { \mathrm { k e e p } } = \{ c \in \mathcal { C } ( y ) \mid S _ { \mathrm { s s a v } } ( c ) \geq \tau _ { \mathrm { c a p } } \} , } \end{array}
$$

$$
{ \mathcal { C } } _ { \mathrm { r e j e c t } } = { \mathcal { C } } ( y ) \setminus { \mathcal { C } } _ { \mathrm { k e e p } } .\tag{12}
$$

(13)

For claims in $\mathcal { C } _ { \mathrm { r e j e c t } }$ , deterministic editing rules remove the corresponding object phrase or neutralize it when direct deletion would disrupt sentence structure. Claims that are not rejected and all other text are left unchanged. This procedure revises only object claims already present in the generated text and does not trigger another generation by the base model. Object claims are identified using a task-defined category vocabulary and synonym mapping independently of subsequent SSAV scoring; the vocabulary can therefore be replaced for a different category space without changing the verification process.

Algorithm 1 summarizes the complete procedure for one object claim. For an open-ended description containing multiple claims, evidence extraction is repeated for each claim, after which the original text is edited once using all verification outcomes.

## IV. EXPERIMENTS

## A. Experimental Setup

1) Base Models and Implementation Details: We evaluate SSAV on three MLLMs with representative architectures and language backbones: LLaVA-1.5-7B, mPLUG-Owl2, and MiniGPT-4 Vicuna-13B [6], [40], [41]. We use public implementations of all base models. During inference, SSAV obtains external visual evidence from a frozen OWL-ViT-base-patch32 [14]; neither the base model nor the open-vocabulary detector is updated. The same query templates, candidate count, regionrelation construction, and semantic-spatial fusion rule are used across all base models and remain fixed for every evaluation sample.

We compare SSAV with representative methods. Vanilla denotes the original base-model output without hallucination mitigation. VCD, OPERA, and DeCo suppress linguistic priors by modifying decoding [10], [11], [19]. MemVR, ClearSight, and REVIS enhance visual representations through visualinformation reinjection or hidden-state intervention [42]–[44]. CCA, VASparse, PAI, and MiddleLayer primarily redistribute visual tokens or attention [20]–[22], [45]. PM enhances target perception through local image magnification [46]. DeGF revises answers through generative feedback [12], and PTI intervenes in the multimodal KV cache during prefilling [47].

The hyperparameter $\lambda _ { \mathrm { s s a v } }$ is set to 3, 1, and 8 for LLaVA-1.5- 7B, mPLUG-Owl2, and MiniGPT-4 Vicuna-13B, respectively. Unless otherwise stated, these settings are retained across all benchmark evaluations.

2) Evaluation Benchmarks and Metrics: We evaluate object hallucination in both closed-set object-existence question answering and open-ended image description. Closed-set evaluation uses POPE [17] and MME-Existence. POPE includes three data sources—COCO, A-OKVQA, and GQA [48]–[50]—with Random, Popular, and Adversarial negative-sampling settings for each. We report Accuracy and F1. For each sampling setting, we take the arithmetic mean over the three data sources

Algorithm 1: Semantic-Spatial Agreement Verification (SSAV)

Input: image I, question q, frozen base model M, open-vocabulary detector D,   
fixed query-template set P, candidate count N, IoU threshold eta,   
and fixed task-specific decision parameters   
Output: SSAV-calibrated or filtered response y'   
1: y <- M(I,q)   
2: Extract object claim c from q or y   
3: Construct K queries P(c) from c   
4: for each query p\_j in P(c) do   
5: Independently evaluate D(I,p\_j) and retain the top-N boxes and scores   
6: Record (b\_jˆ(1),s\_jˆ(1)) and the runner-up score $\hat { \mathbf { s } } _ { - } \hat { \mathbf { \jmath } } \cdot \mathbf { \jmath } ( 2 )$   
7: end for   
8: S\_sem <- (1/K) sum\_j s\_jˆ(1)   
9: Build graph G from the IoU between preferred candidate boxes   
10: Extract the largest component C<sub>\*</sub> and compute rho, u, and {d\_j | j in C<sub>\*</sub>}   
11: S\_spa <- rho <sub>\*</sub> u <sub>\*</sub> (1/|C<sub>\*</sub>|) sum\_(j in C<sub>\*</sub>) d\_j   
12: S\_ssav <- sqrt(S\_sem <sub>\*</sub> S\_spa)   
13: if the task is closed-set Yes-No VQA then   
14: m\_base <- ell\_yes - ell\_no   
15: m\_ssav <- m\_base + lambda\_ssav(S\_ssav - tau\_ssav)   
16: y' <- I[m\_ssav >= 0]   
17: else   
18: Partition c as retained or rejected using tau\_cap   
19: Edit the original caption deterministically to obtain $\boldsymbol { \ Y } ^ { \prime }$   
20: end if   
21: return $\boldsymbol { \ Y } ^ { \prime }$

and separately summarize Random, Popular, and Adversarial. Random primarily reflects ordinary object-existence judgment, whereas Popular and Adversarial further test hallucination under frequent and confusable negative objects.

For MME-Existence, we use the official object-existence subtask [51] and report Accuracy, paired accuracy (Acc+), and their sum, the MME Score. The complete MME benchmark also contains counting, position, color, optical-character-recognition, and knowledge-reasoning tasks, whereas SSAV targets whether an object claim is supported by image content. We therefore use the Existence subtask, which directly matches the definition of object hallucination, and pair it with POPE as a closed-set evaluation without conflating changes in other abilities.

Open-ended image description is evaluated with CHAIR [16]. All models generate captions for the same 500 COCO images with a maximum of 512 newly generated tokens. CHAIRs measures the proportion of sentences containing at least one hallucinated object, and CHAIRi measures the proportion of hallucinated instances among generated object mentions; lower values are better. Recall measures coverage of real objects in the generated descriptions, with higher values indicating better coverage.

## B. Experimental Results

Using identical model versions and complete standard protocols, we compare SSAV with the original base models and representative hallucination-mitigation methods.

Table I reports each sampling setting as the arithmetic mean over COCO, A-OKVQA, and GQA. On LLaVA-1.5-7B, SSAV raises Accuracy from 84.76% to 86.57% under Popular sampling and from 79.91% to 83.08% under Adversarial sampling; the corresponding F1 scores increase from 85.62% to 85.66% and from 81.85% to 82.60%. Random-sampling Accuracy and F1 change from 90.33% and 90.27% to 88.54% and 87.49%, respectively. Popular and Adversarial negatives contain frequent or strongly co-occurring confusable objects, for which the base model is more susceptible to linguistic priors. By requiring stable responses under equivalent queries and repeated localization to the same region, SSAV suppresses claims without consistent visual correspondences and yields larger gains in these difficult settings. In contrast, the strong Vanilla baseline under Random sampling leaves less room for correction; weakly supported true objects may be overcorrected, consistent with the recall trend in Fig. 3(b).

SSAV obtains the highest Accuracy and F1 within the LLaVA-1.5-7B and mPLUG-Owl2 groups under both Popular and Adversarial sampling. On mPLUG-Owl2, Adversarial Accuracy and F1 improve by 12.98 and 6.11 percentage points over Vanilla; the corresponding gains on MiniGPT-4 Vicuna 13B are 12.42 and 6.03 points. These larger gains indicate that independent semantic-spatial evidence is more beneficial when the base model distinguishes object presence from absence less reliably. Improvements across three language backbones in the difficult settings also show that the effect does not depend on a particular network layer or decoding interface.

SSAV achieves the highest MME Score in Table II for all three base models. For LLaVA-1.5-7B and mPLUG-Owl2, the score rises by 5 points from already strong baselines, with a larger gain in Acc+ than in Accuracy. Acc+ requires both paired questions for the same image to be answered correctly, so the pattern indicates that SSAV not only corrects individual existence judgments but also reduces inconsistency across paired Yes–No questions. For MiniGPT-4 Vicuna-13B, Accuracy increases from 78.33% to 98.33%, Acc+ from 56.67% to 96.67%, and the score from 135 to 195. The especially large Acc+ gain suggests that external semantic-spatial evidence is particularly effective at correcting systematic errors in paired questions. Together with the cross-source POPE evaluation, this result shows that SSAV’s benefit for short-form object-existence decisions is not confined to one sampling protocol.

SSAV reduces both sentence-level and instance-level object hallucination for all three base models. On LLaVA-1.5-7B, CHAIRs decreases from 49.40% to 32.80% and CHAIRi from 13.82% to 8.22%. On mPLUG-Owl2, the metrics decrease from 60.00% and 17.32% to 36.00% and 10.14%; on MiniGPT-4 Vicuna-13B, they decrease from 32.00% and 9.23% to

TABLE I  
COMPARISON WITH REPRESENTATIVE METHODS ON POPE. FOR EACH SAMPLING SETTING, ACCURACY AND F1 ARE AVERAGED ACROSS COCO, A-OKVQA, AND GQA. <sup>†</sup> DENOTES RESULTS REPORTED BY SUBSEQUENT STUDIES USING THE SAME MODEL BACKBONE AND EVALUATION PROTOCOL.
<table><tr><td rowspan="2">Model</td><td rowspan="2">Method</td><td colspan="2">Random</td><td colspan="2">Popular</td><td colspan="2">Adversarial</td></tr><tr><td>Acc ↑</td><td>F1 ↑</td><td>Acc ↑</td><td>F1 ↑</td><td>Acc ↑1</td><td>F1 ↑</td></tr><tr><td rowspan="9">LLaVA-1.5-7B</td><td>Vanilla</td><td>90.33</td><td>90.27</td><td>84.76</td><td>85.62</td><td>79.91</td><td>81.85</td></tr><tr><td>VCD{† [52]</td><td>82.31</td><td>83.83</td><td>76.59</td><td>79.82</td><td>71.20</td><td>76.14</td></tr><tr><td>HALC† [52]</td><td>87.00</td><td>87.81</td><td>80.31</td><td>82.76</td><td>72.68</td><td>77.49</td></tr><tr><td>Energy [52]</td><td>88.49</td><td>87.90</td><td>84.33</td><td>84.30</td><td>79.98</td><td>80.79</td></tr><tr><td>DoLa† [53]</td><td>84.78</td><td>84.19</td><td>79.75</td><td>80.61</td><td>76.32</td><td>76.16</td></tr><tr><td>OPERA† [53]</td><td>87.53</td><td>86.45</td><td>84.21</td><td>83.50</td><td>80.88</td><td>80.69</td></tr><tr><td>AGLA [53]</td><td>88.54</td><td>87.71</td><td>85.14</td><td>84.68</td><td>81.13</td><td>81.36</td></tr><tr><td>ICD† [54]</td><td>87.82</td><td>86.73</td><td>84.72</td><td>83.94</td><td>80.98</td><td>80.78</td></tr><tr><td>PAI† [54]</td><td>88.32</td><td>88.01</td><td>84.69</td><td>85.61</td><td>79.18</td><td>80.38</td></tr><tr><td rowspan="6">mPLUG-Owl2</td><td>SSAV</td><td>88.54</td><td>87.49</td><td>86.57</td><td>85.66</td><td>83.08</td><td>82.60</td></tr><tr><td>Vanilla VCD† [52]</td><td>80.33</td><td>83.07</td><td>71.83</td><td>77.33</td><td>67.80</td><td>75.13</td></tr><tr><td>HALC† [52]</td><td>79.70 81.93</td><td>82.05</td><td>72.83</td><td>77.34</td><td>69.41</td><td>75.23</td></tr><tr><td>Energy [52]</td><td>87.10</td><td>84.03 86.01</td><td>74.62 83.22</td><td>78.94</td><td>69.68</td><td>75.87</td></tr><tr><td>MVP [55]</td><td>90.14</td><td>90.10</td><td>82.49</td><td>82.60 83.32</td><td>80.05</td><td>80.00</td></tr><tr><td>SSAV</td><td>88.10</td><td>87.44</td><td>84.58</td><td>84.34</td><td>77.70 80.78</td><td>79.60 81.24</td></tr><tr><td rowspan="3">MiniGPT-4 Vicuna-13B</td><td>Vanilla</td><td>76.06</td><td>80.80</td><td>70.50</td><td>75.09</td><td>64.76</td><td>71.66</td></tr><tr><td>Regular (POPE) [17]</td><td>74.54</td><td>72.82</td><td>68.83</td><td>68.81</td><td>65.05</td><td>66.15</td></tr><tr><td>SSAV</td><td>85.82</td><td>84.87</td><td>83.69</td><td>82.97</td><td>77.18</td><td>77.69</td></tr></table>

TABLE II

SAME-PROTOCOL COMPARISON WITH REPRESENTATIVE METHODS ON MME-EXISTENCE. MME SCORE IS THE SUM OF ACCURACY AND ACC+. AN EM DASH INDICATES THAT ACCURACY AND ACC+ WERE NOT REPORTED SEPARATELY. <sup>†</sup> DENOTES RESULTS REPORTED BY SUBSEQUENT STUDIES USING THE SAME MODEL BACKBONE AND EVALUATION PROTOCOL.
<table><tr><td>Model</td><td>Method</td><td>Accuracy ↑</td><td>Acc+ ↑</td><td>MME Score ↑</td></tr><tr><td rowspan="9">LLaVA-1.5-7B</td><td>Vanilla</td><td>96.67</td><td>93.33</td><td>190.00</td></tr><tr><td>VCD† [52]</td><td></td><td></td><td>170.00</td></tr><tr><td>HALC† [52]</td><td></td><td></td><td>190.00</td></tr><tr><td>DoLa† [53]</td><td></td><td></td><td>175.00</td></tr><tr><td>OPERA† [53]</td><td></td><td></td><td>175.00</td></tr><tr><td>AGLA [53]</td><td></td><td></td><td>180.00</td></tr><tr><td>LURE† [56]</td><td>93.33</td><td>86.67</td><td>180.00</td></tr><tr><td>LogicCheckGPT [56]</td><td>96.67</td><td>93.33</td><td>190.00</td></tr><tr><td>SSAV</td><td>98.33</td><td>96.67</td><td>195.00</td></tr><tr><td rowspan="6">mPLUG-Owl2</td><td>Vanilla</td><td>91.67</td><td>83.33</td><td>175.00</td></tr><tr><td>VCD† [55]</td><td></td><td></td><td>170.00</td></tr><tr><td>OPERA† [55]</td><td></td><td></td><td>173.33</td></tr><tr><td>DoLa† [57]</td><td></td><td></td><td>167.00</td></tr><tr><td>HALC† [57]</td><td></td><td></td><td>167.00</td></tr><tr><td>VASparse† [57]</td><td></td><td></td><td>175.00</td></tr><tr><td rowspan="6">MiniGPT-4 Vicuna-13B</td><td>SSAV</td><td>93.33</td><td>86.67</td><td>180.00</td></tr><tr><td>Vanilla</td><td>78.33</td><td>56.67</td><td>135.00</td></tr><tr><td>LRV-Instruction† [56]</td><td>83.33</td><td>66.67</td><td>150.00</td></tr><tr><td>SelfCheck† [56]</td><td>80.00</td><td>60.00</td><td>140.00</td></tr><tr><td>LURE† [56]</td><td>85.00</td><td>70.00</td><td>155.00</td></tr><tr><td>LogicCheckGPT [56] SSAV</td><td>86.67 98.33</td><td>73.33 96.67</td><td>160.00 195.00</td></tr></table>

19.40% and 5.98%. Their simultaneous reduction shows that SSAV reduces hallucination at both the sentence and objectinstance levels, rather than merely removing a small number of concentrated errors. mPLUG-Owl2 has the highest baseline CHAIRs and the largest absolute reduction, indicating more room for object-level verification when the original captions hallucinate more severely. Continued gains on MiniGPT-4, whose baseline scores are lower, show that the mechanism does not depend on a high initial hallucination rate. Together with the matched-random deletion control in Table IV, these results indicate that the gain comes from evidence-based selection of claims rather than simple caption shortening.

## C. Ablation Studies

To analyze the contribution of each SSAV component to object-hallucination mitigation, we conduct detailed ablations on LLaVA-1.5-7B under different evidence-construction and fusion settings. All variants use the same base-model outputs, OWL-ViT candidate responses, and evaluation samples. Query counts and evidence construction follow the definition of each variant, while all other implementation conditions remain unchanged; decision parameters remain fixed throughout evaluation.

1) Ablation of Core Components: The core-component ablation covers multi-prompt semantic support, QIRV spatial verification, and semantic-spatial fusion. For closed-set objectexistence question answering, we compare five settings. Vanilla uses the original base-model decision without external visual evidence. Single Query uses only the top-1 response score to the class-name query. Prompt Mean averages top-1 scores across four semantically equivalent queries and retains only multiprompt semantic support. Spatial QIRV uses only the spatial evidence composed of candidate persistence, spatial consistency, and within-query candidate dominance. Full SSAV fuses multiprompt semantic support and QIRV with a geometric mean.

TABLE III  
SAME-PROTOCOL COMPARISON WITH REPRESENTATIVE METHODS ON CHAIR. ALL RESULTS USE 500 COCO IMAGES AND A MAXIMUM GENERATION LENGTH OF 512 NEW TOKENS. LOWER CHAIRS AND CHAIRI ARE BETTER. <sup>†</sup> DENOTES RESULTS REPORTED BY SUBSEQUENT STUDIES USING THE SAME MODEL BACKBONE AND GENERATION PROTOCOL.
<table><tr><td>Model</td><td>Method</td><td>CHAIRs↓</td><td>CHAIRi ↓</td></tr><tr><td rowspan="10">LLaVA-1.5-7B</td><td>Vanilla</td><td>49.40</td><td>13.82</td></tr><tr><td>Nucleus† [58]</td><td>54.00</td><td>16.10</td></tr><tr><td>Nucleus+INTER [58]</td><td>51.80</td><td>14.10</td></tr><tr><td>Beam† [58]</td><td>48.80</td><td>13.90</td></tr><tr><td>Beam+INTER [58]</td><td>46.40</td><td>13.40</td></tr><tr><td>VCD [58]</td><td>53.80</td><td>16.00</td></tr><tr><td>VCD+INTER [58]</td><td>56.00</td><td>15.70</td></tr><tr><td>OPERA† [58]</td><td>45.40</td><td>13.80</td></tr><tr><td>OPERA+INTER [58]</td><td>47.00</td><td>13.60</td></tr><tr><td>SSAV</td><td>32.80</td><td>8.22</td></tr><tr><td rowspan="10">mPLUG-Owl2</td><td>Vanilla</td><td>60.00</td><td>17.32</td></tr><tr><td>Nucleus† [58]</td><td>60.80</td><td>20.10</td></tr><tr><td>Nucleus+INTER [58]</td><td>59.40</td><td>19.30</td></tr><tr><td>Beam† [58]</td><td>56.40</td><td>17.90</td></tr><tr><td>Beam+INTER [58]</td><td>53.40</td><td>17.20</td></tr><tr><td>VCD[58]</td><td>62.80</td><td>20.50</td></tr><tr><td>VCD+INTER [58]</td><td>60.40</td><td>20.50</td></tr><tr><td>OPERA†[58]</td><td>55.20</td><td>16.10</td></tr><tr><td>OPERA+INTER [58]</td><td>52.50</td><td></td></tr><tr><td>SSAV</td><td>36.00</td><td>15.90 10.14</td></tr><tr><td rowspan="2">MiniGPT-4 Vicuna-13B</td><td></td><td></td><td></td></tr><tr><td>Vanilla SSAV</td><td>32.00 19.40</td><td>9.23 5.98</td></tr></table>

Each comparison addresses a specific question. Single Query versus Prompt Mean tests whether cross-query aggregation reduces response variation from one wording. Prompt Mean versus full SSAV measures the contribution of spatial-instance consistency beyond semantic response strength. Spatial QIRV versus full SSAV tests whether semantic support further improves a decision based only on region structure. Vanilla provides a common reference for the overall effect of each form of external evidence. Closed-set ablations report Accuracy and F1 on POPE; open-ended captioning compares Vanilla, Prompt Mean, Spatial QIRV, and full SSAV with CHAIRs and CHAIRi. We additionally include a Matched Random control for captioning, which randomly deletes the same number of object claims as full SSAV using the same editing procedure, testing whether improvements arise merely from the deletion count.

Effect of multi-prompt aggregation and spatial evidence. Single Query obtains 80.20% Accuracy and 80.90% F1 on GQA Adversarial. Prompt Mean slightly raises Accuracy to 80.47% but lowers F1 to 80.43%. Thus, aggregating queries reduces dependence on one wording but does not by itself guarantee simultaneous gains in both metrics; score averaging cannot determine whether high responses from different queries originate from the same instance. Spatial QIRV reaches 81.23% Accuracy and 81.62% F1, showing that cross-query regional correspondence contributes information beyond response strength. Full SSAV further reaches 81.97%

TABLE IV

ABLATION OF CORE COMPONENTS ON LLAVA-1.5-7B (%). POPE RESULTS ARE REPORTED ON THE GQA ADVERSARIAL SUBSET. LOWER CHAIRS AND CHAIRI ARE BETTER, AND MME SCORE IS THE SUM OF ACCURACY AND ACC+. AN EM DASH DENOTES AN INAPPLICABLE SETTING; BOLD INDICATES THE BEST RESULT IN EACH COLUMN.

(a) POPE and CHAIR
<table><tr><td>Method</td><td>GQA Acc ↑ GQA F1 ↑</td><td></td><td>CHAIRs↓</td><td>CHAIRi↓</td></tr><tr><td>Vanilla</td><td>77.73</td><td>80.54</td><td>49.40</td><td>13.82</td></tr><tr><td>Single Query</td><td>80.20</td><td>80.90</td><td></td><td></td></tr><tr><td>Prompt Mean</td><td>80.47</td><td>80.43</td><td>33.00</td><td>8.82</td></tr><tr><td>Spatial QIRV</td><td>81.23</td><td>81.62</td><td>32.80</td><td>8.84</td></tr><tr><td>SSAV</td><td>81.97</td><td>81.78</td><td>32.80</td><td>8.22</td></tr><tr><td>Matched Random</td><td></td><td></td><td>43.40</td><td>12.49</td></tr></table>

<table><tr><td>Method</td><td>Acc ↑</td><td>Acc+ ↑</td><td>Score ↑</td></tr><tr><td>Vanilla</td><td>96.67</td><td>93.33</td><td>190.00</td></tr><tr><td>Single Query</td><td>96.67</td><td>93.33</td><td>190.00</td></tr><tr><td>Prompt Mean</td><td>96.67</td><td>93.33</td><td>190.00</td></tr><tr><td>Spatial QIRV</td><td>96.67</td><td>93.33</td><td>190.00</td></tr><tr><td>SSAV</td><td>98.33</td><td>96.67</td><td>195.00</td></tr><tr><td>Matched Random</td><td></td><td></td><td></td></tr></table>

Accuracy and 81.78% F1, supporting the complementarity of semantic support and spatial-instance consistency.

Fusion gains in open-ended descriptions and paired judgments. On CHAIR, Prompt Mean, Spatial QIRV, and full SSAV have similar CHAIRs, but full SSAV yields the lowest CHAIRi at 8.22%, compared with 8.82% and 8.84% for Prompt Mean and Spatial QIRV. The fusion benefit therefore appears mainly in reducing hallucinated object instances after each individual branch has already removed the dominant sentencelevel errors. On MME-Existence, Single Query, Prompt Mean, and Spatial QIRV all remain at 190, while only full SSAV reaches 195, further indicating that strict paired judgments benefit from joint support in response strength and regional consistency.

2) QIRV Evidence Correspondence Analysis: The corecomponent ablation establishes whether spatial evidence contributes to the final decision, but it does not determine whether QIRV’s discriminative power derives from sample-specific spatial information rather than its score distribution or a datasetlevel bias. We therefore progressively shuffle QIRV evidence to test whether correct correspondence between an object claim and its image region is necessary for effective discrimination.

The experiment is conducted on the POPE Adversarial subsets. QIRV evidence is shuffled only within each dataset at ratios of 0%, 25%, 50%, 75%, and 100%. For ratios from 25% to 100%, we use random seeds 0, 1, 2, 3, and 4 and report the mean and standard deviation over five runs. Shuffling preserves the marginal QIRV score distribution within each dataset while progressively breaking its correspondence to individual image– claim samples. No task-decision threshold is used; we directly report ROC-AUC and PR-AUC, with PR-AUC computed as Average Precision.

As shown in Fig. 3(a), without shuffling, ROC-AUC and PR-AUC are 63.32% and 67.29%, respectively. At 25% shuffling, they decrease to $5 9 . 9 5 \% \pm 0 . 5 6 \%$ and 62.19%±0.42%. At 50% and 75%, ROC-AUC further decreases to $5 6 . 6 7 \% \pm 0 . 4 5 \%$ and $5 2 . 8 5 \% \pm 0 . 8 9 \%$ , while PR-AUC falls to $5 7 . 8 4 \% \pm 0 . 5 1 \%$ and $5 3 . 5 2 \% \pm 0 . 7 7 \%$ . With fully shuffled QIRV evidence, ROC-AUC and PR-AUC are only $5 0 . 4 3 \% \pm 0 . 3 8 \%$ and $5 0 . 7 8 \% \pm$ 0.42%, approaching random discrimination.

Both AUC metrics decrease continuously as the spatialevidence correspondence is progressively destroyed. Relative to fully aligned QIRV, complete shuffling lowers ROC-AUC by 12.89 percentage points and PR-AUC by 16.51 points. Because the permutation preserves the marginal QIRV score distribution within each dataset, these results show that QIRV contains sample-specific spatial information. Its discrimination depends on correct matching between an object claim and the corresponding image region and cannot be explained by the overall score distribution alone.

3) Sensitivity to Fusion Strength: To analyze the effect of external-evidence calibration strength on closed-set object judgments, we vary $\lambda _ { \mathrm { s s a v } } \in \{ 0 , 1 , 2 , 3 , 4 , 5 \}$ while holding all other settings fixed. $\lambda _ { \mathrm { s s a v } } ~ = ~ 0$ corresponds to the original decision without the SSAV calibration term.

As shown in Fig. 3(b), increasing $\lambda _ { \mathrm { s s a v } }$ from 0 to 1 raises Accuracy from 77.73% to 80.70% and F1 from 80.54% to 82.20%. $\mathrm { { A t } } \lambda _ { \mathrm { s s a v } } = 2$ , they reach 82.10% and 82.59%. Over this range, the number of corrected errors grows from 0 to 240, while 109 new errors are introduced, yielding a net correction of 131. Moderate calibration therefore preferentially fixes original decisions lacking visual support. At larger strengths, corrected samples continue to increase, but harmed samples grow faster: the net correction is 127 at $\lambda _ { \mathrm { s s a v } } = 3$ and 4 and drops to 86 at $\lambda _ { \mathrm { s s a v } } = 5 .$ . Recall simultaneously falls from 84.93% at $\lambda _ { \mathrm { s s a v } } = 2$ to 72.13% at $\lambda _ { \mathrm { s s a v } } = 5$ . The rise-and-fall pattern in Accuracy and F1 therefore reflects over-calibration rather than failure of the external evidence: an excessive coefficient overwhelms the base-model margin and changes some true positives to No. Calibration strength must balance false-positive reduction against positive-class recall.

Figure 4 illustrates claim-level revision by SSAV in four complementary scenarios. In case (a), the base model correctly identifies the dog, bicycle, and pedestrians but additionally states that some pedestrians carry handbags, an unsupported background-object claim. SSAV removes only the handbag phrase and preserves the remaining street-scene description. In case (b), a bench is linguistically plausible in a scene containing zebras, trees, and a natural environment, but the object is absent from the image. SSAV removes the bench while retaining the zebra count, positions, and background trees. In case (c), the action context of a baseball game induces the model to state that a player is preparing to catch a ball, although no visible sportsball evidence is present. SSAV removes the corresponding object claim while retaining the visible players, bat, and gloves. Case (d) further demonstrates multi-object hallucination in a complex indoor scene: the base model mentions a sink, bowls, and a potted plant, whereas SSAV removes these unsupported objects while retaining the people, refrigerator, oven, and wall clock. The four cases respectively cover an accessory in a complex background, a co-occurring object induced by scene priors, an implicit object induced by action context, and multiple objects in an indoor scene. They show that semanticspatial verification can revise local claims arising from different hallucination sources rather than truncating the description as a whole.

b  
QIRV depends on evidence correspondence  
![](images/662c34fc4ef293f472631fd4988da179e02165ce44533ad4e3edf53887b44374.jpg)

Calibration strength shows a peaked response  
![](images/11ec381db9354091f0613ed1417c98e3bc37afa5ffe0067db07b34531423c4d0.jpg)  
Fig. 3. QIRV evidence correspondence and calibration-strength sensitivity. (a) QIRV evidence is progressively shuffled within each of the three POPE Adversarial subsets while preserving the marginal score distribution. Points show macro-averaged ROC-AUC and PR-AUC; error bars denote the standard deviation over five random seeds, except for the deterministic 0% setting. (b) Accuracy and F1 of LLaVA-1.5-7B as λ varies, with all other parameters fixed.

![](images/e4b177c5eedf448be5645ca7b0e76bc8da7b4ccf5429c19b27c16c4be616cc43.jpg)  
Fig. 4. Qualitative examples of SSAV on CHAIR. Hallucinated claims in the vanilla response are highlighted in red, and descriptions retained after SSAV filtering are highlighted in green. The four cases illustrate background-, scene-prior-, action-induced, and multi-object indoor hallucinations.

## D. Design Choices and Robustness Analysis

We compare fusion operators, IoU thresholds, and candidatedominance formulations using 1,531 unambiguous claims from the complete CHAIR500 generations, including 1,158 present object and 373 hallucinated claims; 38 claims with ambiguous category mappings are excluded. Confidence intervals use 1,000 bootstrap resamples clustered by image. For controlled ranking comparison, CHAIR metrics use a fixed rejection budget of 464 claims, matching calibrated full SSAV. These results therefore compare variants under a matched rejection budget rather than after independent calibration.

1) Comparison of Fusion Operators: The arithmetic mean obtains a claim-level ROC-AUC of 81.23%, 3.39 points below the geometric mean, with a paired-bootstrap 95% confidence interval of $[ - 4 . 5 9 , - 2 . 1 8 ]$ . Its PR-AUC is 1.06 points lower, with an interval of $[ - 1 . 4 9 , - 0 . 6 5 ]$ , showing that linear compensation can allow one branch to mask insufficient evidence from the other. The geometric mean, harmonic mean, and minimum provide similar discrimination, with no stable paired advantage over the geometric mean. We retain the geometric mean as a smooth, symmetric, and parameter-free soft conjunction of semantic and spatial evidence.

TABLE V  
COMPARISON OF SEMANTIC-SPATIAL FUSION OPERATORS. CLAIM-LEVELAUCS ARE COMPUTED ON 1,531 UNAMBIGUOUS OBJECT CLAIMS; CHAIRRESULTS USE A MATCHED REJECTION BUDGET OF 464 CLAIMS.
<table><tr><td>Fusion operator</td><td>ROC-AUC ↑</td><td>PR-AUC ↑</td><td>CHAIRs ↓</td><td>CHAIRi ↓</td><td>Recall ↑</td></tr><tr><td>Arithmetic mean</td><td>81.23</td><td>93.15</td><td>34.40</td><td>8.91</td><td>74.86</td></tr><tr><td>Geometric mean (SSAV)</td><td>84.61</td><td>94.21</td><td>32.80</td><td>8.22</td><td>75.69</td></tr><tr><td>Harmonic mean</td><td>84.63</td><td>94.27</td><td>32.00</td><td>8.22</td><td>75.56</td></tr><tr><td>Minimum</td><td>84.62</td><td>94.33</td><td>31.80</td><td>8.34</td><td>75.56</td></tr></table>

TABLE VI  
SENSITIVITY TO THE IOU THRESHOLD η. CHAIR RESULTS USE THE SAME MATCHED REJECTION BUDGET OF 464 CLAIMS.
<table><tr><td>IoU threshold η</td><td>ROC-AUC ↑</td><td>PR-AUC ↑</td><td>CHAIRs↓</td><td>CHAIRi ↓</td><td>Recall ↑</td></tr><tr><td>0.30</td><td>84.65</td><td>94.22</td><td>32.80</td><td>8.22</td><td>75.69</td></tr><tr><td>0.40</td><td>84.66</td><td>94.22</td><td>32.80</td><td>8.22</td><td>75.62</td></tr><tr><td>0.50 (default)</td><td>84.61</td><td>94.21</td><td>32.80</td><td>8.22</td><td>75.69</td></tr><tr><td>0.60</td><td>84.59</td><td>94.20</td><td>32.80</td><td>8.22</td><td>75.62</td></tr><tr><td>0.70</td><td>84.57</td><td>94.20</td><td>32.80</td><td>8.22</td><td>75.62</td></tr></table>

TABLE VII

COMPARISON OF CANDIDATE-DOMINANCE FORMULATIONS. CHAIR RESULTS USE A MATCHED REJECTION BUDGET OF 464 CLAIMS.
<table><tr><td>Dominance formulation</td><td>ROC-AUC ↑</td><td>PR-AUC ↑</td><td>CHAIRs ↓</td><td>CHAIRi ↓</td><td>Recall ↑</td></tr><tr><td>Logit-sigmoid margin (SSAV)</td><td>84.61</td><td>94.21</td><td>32.80</td><td>8.22</td><td>75.69</td></tr><tr><td>Raw score gap</td><td>82.91</td><td>93.55</td><td>32.40</td><td>8.37</td><td>76.01</td></tr><tr><td>Relative score gap</td><td>80.09</td><td>92.57</td><td>33.80</td><td>8.87</td><td>75.50</td></tr><tr><td>Score ratio</td><td>84.66</td><td>94.23</td><td>33.00</td><td>8.22</td><td>75.69</td></tr><tr><td>Rank-only margin</td><td>84.62</td><td>94.22</td><td>33.40</td><td>8.37</td><td>75.69</td></tr></table>

TABLE VIII

INFERENCE EFFICIENCY ON THE COMPLETE CHAIR500 EVALUATION SET. WALL-CLOCK TIME EXCLUDES MODEL LOADING AND WARM-UP.
<table><tr><td>Stage / Method</td><td>Total time</td><td>Latency per image</td><td>Peak reserved GPU memory</td></tr><tr><td>Vanilla LLaVA generation</td><td>00:19:37</td><td>2.3550 s</td><td>14.178 GB</td></tr><tr><td>SSAV external verification</td><td>00:03:07</td><td>0.3744 s</td><td>0.666 GB</td></tr><tr><td>Full SSAV pipeline</td><td>00:22:45</td><td>2.7306 s</td><td>14.178 GB</td></tr></table>

2) Sensitivity to the IoU Threshold: As η varies from 0.3 to 0.7, claim-level ROC-AUC and PR-AUC change by at most 0.09 and 0.02 percentage points, respectively. Under a matched rejection budget, CHAIRs and CHAIRi remain fixed at 32.80% and 8.22%, while Recall ranges only from 75.62% to 75.69%. These results demonstrate stable performance across the evaluated IoU thresholds, including the default η = 0.5.

3) Comparison of Candidate-Dominance Formulations: The raw and relative score gaps reduce ROC-AUC to 82.91% and 80.09%, with paired-bootstrap differences of −1.71 points (95% confidence interval [−2.28, −1.20]) and −4.52 points ([−5.63, −3.48]) from the logit-sigmoid margin. Thus, rawscore differences provide less stable claim-level ranking under the current detector-score distribution. The score ratio and rank-only margin achieve similar AUCs but no consistent advantage across the controlled CHAIR metrics. We retain the logit-sigmoid margin because it represents preferred-torunner-up separation in a bounded form and provides a stable, interpretable trade-off among claim-level discrimination, CHAIR, and Recall.

## E. Efficiency Analysis

Evaluated on the complete 500-image CHAIR set using an NVIDIA GeForce RTX 4090, Vanilla LLaVA requires 19 min 37 s, or 2.3550 s per image. The 1,569 generated object claims require 6,276 OWL-ViT calls. Verification, claim-level decisions, and text editing add 3 min 07 s, yielding a total of 22 min 45 s, or 2.7306 s per image. SSAV therefore requires 1.159× the Vanilla runtime, corresponding to a 15.9% overhead. The verifier alone reserves 0.666 GB; because generation and verification run sequentially, the pipeline peak remains 14.178 GB. Measurements exclude model loading and warm-up.

## V. CONCLUSION

We investigate how an independent visual model can provide reliable external verification for object claims in MLLM responses. A single high response from an open-vocabulary detector can be affected by query wording and incidental local matches and therefore cannot be treated directly as sufficient evidence of object presence. By contrast, an object claim supported by real image content should maintain relatively stable responses under semantically equivalent queries and continue to point to the same visual instance. Based on this observation, we propose Semantic-Spatial Agreement Verification (SSAV). The method estimates semantic support through multi-prompt aggregation and introduces QIRV to jointly characterize candidate persistence, spatial overlap, and within-query candidate dominance, after which semantic and spatial evidence are fused into a unified object-level verification score. SSAV calibrates closed-set object-existence judgments and filters claims in open-ended image descriptions without updating the base model or the open-vocabulary detector. Experiments across multiple models and benchmarks show that SSAV effectively mitigates object hallucination in MLLMs.

SSAV’s verification performance remains limited by the perceptual capability of the external detector and incurs additional inference cost. Because QIRV uses each query’s preferred candidate region, small, occluded, or repeatedcategory objects may cause equivalent queries to localize to different real instances, weakening spatial evidence. Future work will examine alternative and multiple detectors to reduce detector-specific bias, extend Top-1 fixed-IoU verification to Top-M candidates, adaptive spatial relations, and multi-instance matching, assess robustness to synonymous and multilingual queries, and reduce verification cost through batched queries, candidate caching, and shared visual features.

## REFERENCES

[1] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, G. Krueger, and I. Sutskever, “Learning Transferable Visual Models From Natural Language Supervi sion,” in Proceedings of the 38th International Conference on Machine Learning. PMLR, 2021, pp. 8748–8763.

[2] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, J. Uszkoreit, and N. Houlsby, “An image is worth 16x16 words: Transformers for image recognition at scale,” in International Conference on Learning Representations, 2021.

[3] J. Li, D. Li, S. Savarese, and S. C. H. Hoi, “BLIP-2: Bootstrapping language-image pre-training with frozen image encoders and large language models,” in Proceedings of the 40th International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, vol. 202. PMLR, 2023, pp. 19 730–19 742.

[4] H. Liu, C. Li, Q. Wu, and Y. J. Lee, “Visual instruction tuning,” in Advances in Neural Information Processing Systems, A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine, Eds., vol. 36. Curran Associates, Inc., 2023, pp. 34 892–34 916.

[5] W. Dai, J. Li, D. Li, A. Tiong, J. Zhao, W. Wang, B. Li, P. N. Fung, and S. Hoi, “InstructBLIP: Towards General-purpose Vision-Language Models with Instruction Tuning,” Advances in Neural Information Processing Systems, vol. 36, pp. 49 250–49 267, 2023.

[6] D. Zhu, J. Chen, X. Shen, X. Li, and M. Elhoseiny, “MiniGPT-4: Enhancing Vision-Language Understanding with Advanced Large Language Models,” in The Twelfth International Conference on Learning Representations, 2023.

[7] Y. Tang, Y. Fang, T. Wang, L. Sun, and L. Chen, ““This Is My Fault”, Really? Understanding Blind and Low-Vision People’s Perception of Hallucination in Large Vision Language Models,” in Proceedings of the 38th Annual ACM Symposium on User Interface Software and Technology. New York, NY, USA: Association for Computing Machinery, 2025, pp. 1–20.

[8] F. Liu, K. Lin, L. Li, J. Wang, Y. Yacoob, and L. Wang, “Mitigating hallucination in large multi-modal models via robust instruction tuning,” in The Twelfth International Conference on Learning Representations, 2024.

[9] L. Li, Z. Xie, M. Li, S. Chen, P. Wang, L. Chen, Y. Yang, B. Wang, L. Kong, and Q. Liu, “VLFeedback: A large-scale AI feedback dataset for large vision-language models alignment,” in Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing. Association for Computational Linguistics, 2024, pp. 6227–6246.

[10] S. Leng, H. Zhang, G. Chen, X. Li, S. Lu, C. Miao, and L. Bing, “Mitigating object hallucinations in large vision-language models through visual contrastive decoding,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024, pp. 13 872– 13 882.

[11] Q. Huang, X. Dong, P. Zhang, B. Wang, C. He, J. Wang, D. Lin, W. Zhang, and N. Yu, “OPERA: Alleviating Hallucination in Multi-Modal Large Language Models via Over-Trust Penalty and Retrospection-Allocation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 13 418–13 427.

[12] C. Zhang, Z. Wan, Z. Kan, M. Q. Ma, S. Stepputtis, D. Ramanan, R. Salakhutdinov, L.-P. Morency, K. P. Sycara, and Y. Xie, “Selfcorrecting decoding with generative feedback for mitigating hallucinations in large vision-language models,” in International Conference on Learning Representations, 2025.

[13] S. Yin, C. Fu, S. Zhao, T. Xu, H. Wang, D. Sui, Y. Shen, K. Li, X. Sun, and E. Chen, “Woodpecker: Hallucination correction for multimodal large language models,” 2023. [Online]. Available: https://arxiv.org/abs/2310.16045

[14] M. Minderer, A. Gritsenko, A. Stone, M. Neumann, D. Weissenborn, A. Dosovitskiy, A. Mahendran, A. Arnab, M. Dehghani, Z. Shen, X. Wang, X. Zhai, T. Kipf, and N. Houlsby, “Simple open-vocabulary object detection with vision transformers,” in European Conference on Computer Vision. Springer, 2022, pp. 728–755.

[15] S. Liu, Z. Zeng, T. Ren, F. Li, H. Zhang, J. Yang, Q. Jiang, C. Li, J. Yang, H. Su, J. Zhu, and L. Zhang, “Grounding DINO: Marrying DINO with grounded pre-training for open-set object detection,” in European Conference on Computer Vision. Springer, 2024, pp. 38–55.

[16] A. Rohrbach, L. A. Hendricks, K. Burns, T. Darrell, and K. Saenko, “Object Hallucination in Image Captioning,” in Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, E. Riloff, D. Chiang, J. Hockenmaier, and J. Tsujii, Eds. Association for Computational Linguistics, 2018, pp. 4035–4045.

[17] Y. Li, Y. Du, K. Zhou, J. Wang, X. Zhao, and J.-R. Wen, “Evaluating Object Hallucination in Large Vision-Language Models,” in Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, H. Bouamor, J. Pino, and K. Bali, Eds. Association for Computational Linguistics, 2023, pp. 292–305.

[18] H. Seo, D. U. Kang, H. Cho, J. Lee, and S. Y. Chun, “On epistemic uncertainty of visual tokens for object hallucinations in large visionlanguage models,” in Advances in Neural Information Processing Systems, 2025.

[19] C. Wang, X. Chen, N. Zhang, B. Tian, H. Xu, S. Deng, and H. Chen, “MLLM can see? Dynamic correction decoding for hallucination mitiga tion,” in International Conference on Learning Representations, 2025.

[20] S. Liu, K. Zheng, and W. Chen, “Paying more attention to image: A training-free method for alleviating hallucination in lvlms,” in European Conference on Computer Vision (ECCV). Springer, 2024.

[21] Y. Xing, Y. Li, I. Laptev, and S. Lu, “Mitigating object hallucination via concentric causal attention,” in Advances in Neural Information Processing Systems, 2024.

[22] X. Zhuang, Z. Zhu, Y. Xie, L. Liang, and Y. Zou, “VASparse: Towards efficient visual hallucination mitigation via visual-aware token sparsification,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025.

[23] S. Jiang, F. Chen, X. Zhang, and K. He, “KVSmooth: Mitigating hallucination in multi-modal large language models through key-value smoothing,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2026.

[24] H. Dastmalchi, A. An, A. Cheraghian, and H. Barzamini, “Fighting hallucinations with counterfactuals: Diffusion-guided perturbations for

LVLM hallucination suppression,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2026.

[25] J. Duan, F. Kong, H. Cheng, J. Diffenderfer, B. Kailkhura, L. Sun, X. Zhu, X. Shi, and K. Xu, “TruthPrInt: Mitigating large vision-language models object hallucination via latent truthful-guided pre-intervention,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025.

[26] Z. Chen, Z. Zhao, H. Luo, H. Yao, B. Li, and J. Zhou, “HALC: Object hallucination reduction via adaptive focal-contrast decoding,” in Proceedings of the 41st International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, vol. 235. PMLR, 2024, pp. 7824–7846. [Online]. Available: https://proceedings.mlr.press/v235/chen24bi.html

[27] Y. Park, D. Lee, J. Choe, and B. Chang, “ConVis: Contrastive decoding with hallucination visualization for mitigating hallucinations in multimodal large language models,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 39, no. 6, 2025, pp. 6434–6442.

[28] X. Wang, J. Pan, L. Ding, and C. Biemann, “Mitigating hallucinations in large vision-language models with instruction contrastive decoding,” in Findings of the Association for Computational Linguistics: ACL 2024, 2024, pp. 15 840–15 853.

[29] F. Huo, W. Xu, Z. Zhang, H. Wang, Z. Chen, and P. Zhao, “Self-introspective decoding: Alleviating hallucinations for large vision-language models,” in The Thirteenth International Conference on Learning Representations, 2025. [Online]. Available: https: //openreview.net/forum?id=rsZwwjYHuD

[30] Y. Zhou, C. Cui, J. Yoon, L. Zhang, Z. Deng, C. Finn, M. Bansal, and H. Yao, “Analyzing and mitigating object hallucination in large vision-language models,” in The Twelfth International Conference on Learning Representations, 2024. [Online]. Available: https://openreview.net/forum?id=oZDJKTlOUe

[31] X. Gu, T.-Y. Lin, W. Kuo, and Y. Cui, “Open-vocabulary object detection via vision and language knowledge distillation,” in International Conference on Learning Representations, 2022.

[32] L. H. Li, P. Zhang, H. Zhang, J. Yang, C. Li, Y. Zhong, L. Wang, L. Yuan, L. Zhang, J.-N. Hwang, K.-W. Chang, and J. Gao, “Grounded languageimage pre-training,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 10 965–10 975.

[33] Y. Du, F. Wei, Z. Zhang, M. Shi, Y. Gao, and G. Li, “Learning to prompt for open-vocabulary object detection with vision-language model,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 14 064–14 073.

[34] Y. Zhong, J. Yang, P. Zhang, C. Li, N. Codella, L. H. Li, L. Zhou, X. Dai, L. Yuan, Y. Li, and J. Gao, “RegionCLIP: Region-based language-image pretraining,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 16 772–16 782.

[35] X. Zhou, R. Girdhar, A. Joulin, P. Krahenbuhl, and I. Misra, “Detecting twenty-thousand classes using image-level supervision,” in Computer Vision – ECCV 2022. Springer, 2022, pp. 350–368.

[36] W. Kuo, Y. Cui, X. Gu, A. J. Piergiovanni, and A. Angelova, “F-VLM: Open-vocabulary object detection upon frozen vision and language models,” in The Eleventh International Conference on Learning Representations, 2023. [Online]. Available: https: //arxiv.org/abs/2209.15639

[37] M. Minderer, A. A. Gritsenko, and N. Houlsby, “Scaling openvocabulary object detection,” in Advances in Neural Information Processing Systems, vol. 36, 2023. [Online]. Available: https: //openreview.net/forum?id=mQPNcBWjGc

[38] J. Wu, X. Li, S. Xu, H. Yuan, H. Ding, Y. Yang, X. Li, J. Zhang, Y. Tong, X. Jiang, B. Ghanem, and D. Tao, “Towards open vocabulary learning: A survey,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 46, no. 7, pp. 5092–5113, 2024.

[39] Z. Zhang, V. Q. Truong, and M. Hoai, “Low-rank prompt adaptation for open-vocabulary object detection,” in Proceedings of the IEEE/CVF International Conference on Computer Vision Workshops, 2025, pp. 4263– 4274.

[40] H. Liu, C. Li, Y. Li, and Y. J. Lee, “Improved Baselines with Visual Instruction Tuning,” in 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2024, pp. 26 286–26 296.

[41] Q. Ye, H. Xu, J. Ye, M. Yan, A. Hu, H. Liu, Q. Qian, J. Zhang, and F. Huang, “mPLUG-Owl2: Revolutionizing Multi-modal Large Language Model with Modality Collaboration,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 13 040–13 051.

[42] X. Zou, Y. Wang, Y. Yan, Y. Lyu, K. Zheng, S. Huang, J. Chen, P. Jiang, J. Liu, C. Tang, and X. Hu, “Look Twice Before You Answer: Memory Space Visual Retracing for Hallucination Mitigation in Multimodal Large

Language Models,” in Forty-Second International Conference on Machine Learning, 2025.

[43] H. Yin, G. Si, and Z. Wang, “ClearSight: Visual signal enhancement for object hallucination mitigation in multimodal large language models,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025.

[44] J. Wu, W. Shi, H. Shen, P. Qi, K. Tang, Z. Huang, B. Wang, and Z. Yang, “REVIS: Sparse latent steering to mitigate object hallucination in large vision-language models,” 2026. [Online]. Available: https://arxiv.org/abs/2602.11824

[45] Z. Jiang, J. Chen, B. Zhu, T. Luo, Y. Shen, and X. Yang, “Devils in middle layers of large vision-language models: Interpreting, detecting and mitigating object hallucinations via attention lens,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025, pp. 25 004–25 014.

[46] S. Mao, C. Zhang, and W. Cai, “Through the magnifying glass: Adaptive perception magnification for hallucination-free VLM decoding,” in Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). San Diego, California, United States: Association for Computational Linguistics, Jul. 2026, pp. 44 480–44 501.

[47] C. Zhang, C. Sun, X. Jiang, W. Li, and X. Tian, “Prefill-time intervention for mitigating hallucination in large vision-language models,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2026.

[48] T.-Y. Lin, M. Maire, S. Belongie, J. Hays, P. Perona, D. Ramanan, P. Dollar, and C. L. Zitnick, “Microsoft COCO: Common Objects ´ in Context,” in Computer Vision – ECCV 2014, D. Fleet, T. Pajdla, B. Schiele, and T. Tuytelaars, Eds. Springer International Publishing, 2014, pp. 740–755.

[49] D. Schwenk, A. Khandelwal, C. Clark, K. Marino, and R. Mottaghi, “A-OKVQA: A Benchmark for Visual Question Answering Using World Knowledge,” in Computer Vision – ECCV 2022, S. Avidan, G. Brostow, M. Cisse, G. M. Farinella, and T. Hassner, Eds. Springer Nature ´ Switzerland, 2022, pp. 146–162.

[50] D. A. Hudson and C. D. Manning, “GQA: A New Dataset for Real-World Visual Reasoning and Compositional Question Answering,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2019, pp. 6700–6709.

[51] C. Fu, P. Chen, Y. Shen, Y. Qin, M. Zhang, X. Lin, J. Yang, X. Zheng, K. Li, X. Sun, Y. Wu, and R. Ji, “MME: A comprehensive evaluation benchmark for multimodal large language models,” in Advances in Neural Information Processing Systems: Datasets and Benchmarks Track, 2025.

[52] X. Liu, A. Deng, and C. Zach, “Energy-guided decoding for object hallucination mitigation,” 2025. [Online]. Available: https: //arxiv.org/abs/2507.07731

[53] W. An, F. Tian, S. Leng, J. Nie, H. Lin, Q. Wang, G. Dai, P. Chen, and S. Lu, “AGLA: Mitigating object hallucinations in large vision-language models with assembly of global and local attention,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025.

[54] J. Li, J. Zhang, Z. Jie, L. Ma, M. Li, X. Luo, and G. Li, “Crossmodal attention calibration for LVLM hallucination mitigation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2026, pp. 40 186–40 196.

[55] X. Qu, J. Sun, W. Wei, D. Liu, J. Dong, and Y. Cheng, “Look, compare, decide: Alleviating hallucination in large vision-language models via multi-view multi-path reasoning,” in Proceedings of the 31st International Conference on Computational Linguistics. Association for Computational Linguistics, 2025, pp. 4428–4441.

[56] J. Wu, Q. Liu, D. Wang, J. Zhang, S. Wu, L. Wang, and T. Tan, “Logical closed loop: Uncovering object hallucinations in large vision-language models,” in Findings of the Association for Computational Linguistics: ACL 2024. Association for Computational Linguistics, 2024, pp. 6944– 6962.

[57] B. Chen, X. Lyu, L. Gao, J. Song, and H. T. Shen, “Attention hijackers: Detect and disentangle attention hijacking in LVLMs for hallucination mitigation,” 2025. [Online]. Available: https://arxiv.org/abs/2503.08216

[58] X. Dong, S. Dong, J. Wang, J. Huang, L. Zhou, Z. Sun, L. Jing, J. Lan, X. Zhu, and B. Zheng, “INTER: Mitigating hallucination in large visionlanguage models by interaction guidance sampling,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025, pp. 2534–2544.