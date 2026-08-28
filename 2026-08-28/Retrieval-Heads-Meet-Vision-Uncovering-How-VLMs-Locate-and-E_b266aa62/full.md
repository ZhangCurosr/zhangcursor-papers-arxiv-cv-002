# Retrieval Heads Meet Vision: Uncovering How VLMs Locate and Extract Visual Information

Chanho Park<sup>1</sup> Daehyeon Choi<sup>1</sup> Jihyun Lee<sup>2</sup> Minhyuk Sung<sup>1</sup> <sup>1</sup>KAIST <sup>2</sup>Independent Researcher {charlieppark, daehyeonchoi, mhsung}@kaist.ac.kr jihyunlee1027@gmail.com

![](images/ffe515407d7e6684a878ab61c6583b7ce362b6318f0a13a77801662fe5532387.jpg)  
Figure 1: Visual Retrieval Heads mediate visual reference resolution in VLMs. We identify a sparse set of attention heads, Visual Retrieval Heads (VRHs), that are causally necessary for resolving the image region referred to by the text prompt. On visual grounding (left), masking the VRHs redirects prediction-stage attention away from the ground-truth referent and causes the model to output a delocalized bounding box. The same heads, discovered only from grounding, also transfer to visual question answering (right): masking them removes attention to the referred object and leads to a fluent but visually unfaithful answer. Yellow boxes denote ground-truth referents; dashed yellow ellipses indicate regions where attention vanishes after VRH masking.

## Abstract

Vision-language models (VLMs) can locate an image region referred to by a text prompt and route the corresponding visual evidence to the output, yet the internal mechanism behind this behavior is not understood. Inspired by retrieval heads in large language models, we ask whether VLMs contain an analogous mechanism for visual retrieval. We answer affirmatively by introducing Visual Retrieval Heads (VRHs), a small subset of attention heads (about 1.7–2.6%) that are causally responsible for grounding text descriptions to image regions. To find them, we recast existing head-scoring methods under a unified design space over query tokens, key aggregation, and cross-sample aggregation. We then show that scoring attention from output prediction tokens with a sum over the ground-truth referent region most reliably identifies causal heads. Across eleven VLMs and five referringexpression benchmarks, masking only the top 20 VRHs reduces grounding accuracy by up to 80 percentage points, while masking the same number of random heads has little effect. Beyond replicating the causal-sparse-universal triad established for text retrieval heads, VRHs exhibit several properties not previously reported: they generalize across visual reference tasks, remaining causal on attribute, spatial, counting, and visual-math benchmarks despite being discovered through boundingbox prediction; they are functionally specific, preserving output format while corrupting localization; and they are architecturally shared, transferring causally across VLMs that share an LLM backbone but differ in vision encoder, projector, and instruction tuning.

## 1 Introduction

How do foundation models locate a specific piece of information buried in a long input context? This “needle-in-a-haystack” question has recently been answered, at least partially, for large language models (LLMs): a small set of attention heads called retrieval heads [1] is largely responsible for routing relevant input tokens to the output. These heads exhibit three properties that together make them a meaningful unit of analysis. They are causal: ablating them collapses long-context retrieval while leaving fluency intact. They are sparse: fewer than 5% of all heads carry the behavior. And they are universal: they appear across model families and scales, suggesting that long-context retrieval converges onto a shared internal mechanism rather than an idiosyncratic implementation.

This observation motivates our key question: whether Vision-Language Models (VLMs) contain an analogous mechanism for visual retrieval—locating an image region referred to by a text prompt and routing it to the output. Existing extensions of retrieval-head analysis to VLMs do not yet answer it. OCR Heads [2] and VERA [3] identify heads that retrieve textual evidence from images, but only when the image is itself a rendering of text; the retrieval target is still a string of characters, not a general visual concept. Localization Heads [4] move beyond rendered text and aim to identify heads whose attention maps align with referred objects, but the analysis stops at attention shape: the heads are not characterized by causal necessity.

In this work we ask whether causal, universal, and sparse retrieval heads exist in VLMs for general image retrieval, and how to find them. We answer affirmatively by introducing Visual Retrieval Heads (VRHs), a small subset of attention heads that are causally responsible for grounding text descriptions to image regions. We show that these heads share the defining properties of their text-only counterparts.

We use referring-expression visual grounding [5] as our probing task: given an image and a referring expression, the model must output a bounding box for the referent. This task makes the retrieval target spatially explicit; the ground-truth bounding box defines exactly which visual tokens the model needs to consult. This lets us define a head’s contribution to retrieval by how strongly it attends from the model’s prediction-stage tokens to the visual tokens inside that box.

Beyond identifying VRHs, our contributions are three-fold. First, on the methodology side, we observe that existing head-scoring methods can all be written in a single form, with a per-head score obtained by aggregating attention weights over a set of key tokens (the tokens to be retrieved) for each query token, and then aggregating these per-query scores across all queries. Under this view, prior methods differ along three axes: the query token set (input prompt tokens vs. output prediction tokens), the key aggregation (an argmax indicator for whether the most-attended key lies in the retrieval region vs. a summation of attention weights over the region), and the cross-sample aggregation (averaging scores across samples vs. voting). For visual grounding, we let the key tokens be the visual tokens that overlap the ground-truth bounding box, and we search this design space directly, scoring each variant by the drop in grounding accuracy when its top-ranked heads are masked. We find that the choice of key aggregation and cross-sample aggregation matters little, but the choice of query tokens matters a great deal: scoring at the output prediction stage with a sum over the retrieval region consistently identifies the most causally important heads. We adopt it as our default scoring function.

Second, on the empirical side, we demonstrate the causality, sparsity, and universality of the VRHs identified by our scoring function through comprehensive experiments across eleven VLMs built on eight language-model backbones [6–9, 27–33] and five grounding benchmarks (RefCOCO/+/g [5, 10], RefSpatialBench [11], Toloka [12]). Masking the top 20 VRHs—about 1.7–2.6% of all heads— reduces grounding accuracy by up to 80 percentage points, while masking the same number of random heads has negligible effect, demonstrating their causal role.

Third, most importantly, our characterization of VRHs reveals four properties that distinguish them from previously studied heads. (i) VRHs generalize across visual reference tasks: although we discover them through bounding-box prediction, they remain causal on attribute, spatial reasoning, counting, and visual math benchmarks (VAW, Spatial457, SpatialRGPT, CV-Bench, MMStar, Count-BenchQA, MathVista) [13–19], suggesting that the mechanism they implement is visual reference resolution in general, not object grounding in particular. (ii) VRHs are functionally specific: masking them preserves output-format validity but the predicted box is delocalized, the visual analogue of the “fluent-but-not-factual” failure mode reported for text retrieval heads. (iii) VRHs are architecturally shared: VRHs found in one VLM remain causally critical when masked in another VLM that shares the same LLM backbone but differs in vision encoder, projector, and instruction-tuning recipe. (iv) VRHs are stably detectable: results are consistent across output-token choices and scoring samples, with as few as 5 to 10 grounding examples recovering nearly the same top-ranked set as the full 200-example set.

## 2 Related Work

We discuss prior works on functionally specialized attention heads in large language models (LLMs) and vision-language models (VLMs). Specifically, we focus on methods for identifying heads specialized for retrieval-related tasks, which are most closely related to our setting.

Most existing methods in this domain identify such specialized heads by quantifying the importance of each head based on inference-time attention patterns [1, 20, 2, 3]. A pioneering work along this direction is Retrieval Heads [1], which detects heads specialized for text retrieval in LLMs by measuring how often the output answer token most attends to the correct input token to be retrieved, i.e., the needle token. QRHead [20] is a follow-up work showing that analyzing attention patterns from input prompt tokens (cf. output tokens in Retrieval Heads) to the input retrieval source tokens is more effective for detecting specialized heads in real-world scenarios.

Beyond text-only retrieval in LLMs, OCR Heads [2] is one of the earliest works to extend the headidentification method of Retrieval Heads to vision-language models (VLMs), discovering specialized heads for optical character recognition. Similarly, VERA [3] identifies VLM heads specialized for long-context visual document retrieval. However, both of these VLM-based specialized heads are limited to visual text documents as the input retrieval source. The first work on specialized heads for visual retrieval from general images is Localization Heads [4], but it does not verify the causality of the detected heads, i.e., their causal effect on retrieval performance, which is a key property of retrieval specialized heads emphasized in all the aforementioned works [1, 20, 2, 3].

In Tab. 1, we summarize the characteristics of these prior works. To the best of our knowledge, our work is the first to identify a set of attention heads specialized for visual retrieval from general images while exhibiting causality in the model’s visual retrieval performance. Note that we further discuss and compare the methodological differences among these prior works in Sec. 3.2.

Table 1: Comparison of prior works on retrieval-related specialized heads.
<table><tr><td></td><td>Retrieval Heads [1]</td><td>QRHead [20]</td><td>OCR Heads [2]</td><td>VERA [3]</td><td>Loc. Heads [4]</td><td>VRH (Ours)</td></tr><tr><td>Input Source</td><td>Text</td><td>Text</td><td>Visual document</td><td>Visual document</td><td>General image</td><td>General image</td></tr><tr><td>Causality</td><td>√</td><td>√</td><td>√</td><td>√</td><td>x</td><td>√</td></tr></table>

## 3 Toward Identifying Visual Retrieval Heads

We introduce our method for identifying Visual Retrieval Heads (VRHs), a subset of attention heads within vision-language models (VLMs) specialized for retrieving visual information relevant to an input text prompt. Our detected VRHs exhibit important desired properties of functionally specialized heads: (1) causality: VRHs are causally responsible for visual retrieval performance, such that masking these heads significantly degrades the VLM’s retrieval behavior; (2) universality: VRHs can be found across multiple VLM families and benchmarks; and (3) sparsity: VRHs constitute only a small subset of attention heads, revealing that visual retrieval behavior in VLMs is concentrated in a few heads. To the best of our knowledge, ours is the first work to discover specialized heads in VLMs that are causally responsible for visual retrieval from general images.

In what follows, we first introduce our setup, where visual grounding is used as our probing task (Sec. 3.1). We then show that existing head-scoring methods, i.e., methods for quantifying the importance of individual heads for a target functionality, can be expressed under a unified formulation. This formulation allows us to systematically explore the design space of scoring methods for detecting VRHs (Sec. 3.2). Based on this analysis, we propose our VRH scoring method by holistically examining different scoring method variants through causal validation, i.e., by measuring how much visual grounding accuracy degrades when the detected heads are masked (Sec. 3.3).

## 3.1 Visual Grounding as Probing Task

We use visual grounding as a probing task for analyzing the visual retrieval behavior of VLMs and detecting VRHs. In this task, a VLM is asked to localize the object in the input image referred to by the input text prompt. More specifically, each sample in a visual grounding dataset is represented as $\boldsymbol { s } = ( I _ { s } , q _ { s } , B _ { s } )$ , where $I _ { s }$ is an image, $q _ { s }$ is a referring expression, and $B _ { s }$ is the ground-truth bounding box of the referred object. Given $( I _ { s } , q _ { s } )$ , the VLM generates a text token sequence that is parsed into a predicted box $\hat { B } _ { s }$ . A prediction is judged correct if its intersection-over-union (IoU) with the ground-truth box exceeds a threshold, IoU $\begin{array} { r } { \lceil ( \hat { B } _ { s } , B _ { s } ) \geq \tau , } \end{array}$ , where τ is typically set to 0.5.

## 3.2 Unified Formulation of Head-Scoring Methods

To effectively design a method to detect VRHs responsible for this visual grounding task, we first review prior works on detecting functionally specialized heads in other retrieval-related tasks in LLMs and VLMs $( \mathrm { e . g . }$ , text retrieval [1, 20], optical character recognition [2, 3]). The key idea in these works is to introduce a head-scoring method to quantify the importance of each attention head for the target task and detect the specialized heads based on it. Here, the importance of each head is typically scored based on its attention weights computed between: (1) a set ofquery tokens $( \mathcal { Q } _ { s } )$ where retrieval behavior is expected to occur, and (2) a set ofkey tokens $( \mathcal { K } _ { s } )$ that correspond to the correct evidence region to be retrieved from the input. In our visual grounding setting, this evidence region is the annotated target region; therefore, we instantiate $\kappa _ { s }$ as the set of visual tokens whose patches overlap with the annotated bounding box $B _ { s }$

For example, existing text retrieval head detection methods in LLMs [1, 20] assign high importance to heads that place strong attention from either output tokens [1] (which express the retrieval prediction) or input prompt tokens [20] (which specify what should be retrieved) onto input tokens corresponding to the correct retrieval region.

More specifically, let $\bar { \Theta } _ { l , h }$ denote the importance score of head h at layer l, computed by aggregating the per-sample head importance scores $\Theta _ { l , h } ( s )$ over samples $s \in S$ in the scoring dataset. In most of the existing methods [1, 20, 2, 3], both the final head score $\bar { \Theta } _ { l , h }$ and the per-sample head score $\Theta _ { l , h } ( s )$ can be expressed under the following unified formulations:

$$
\bar { \Theta } _ { l , h } = \Omega _ { s \in { \cal S } } \underbrace { \left[ \Theta _ { l , h } ( s ) \right] } _ { \mathrm { p e r - s a m p l e ~ s c o r e } } , \quad \Theta _ { l , h } ( s ) = \frac { 1 } { \left| \mathcal { Q } _ { s } \right| } \sum _ { t \in \mathcal { Q } _ { s } } \Phi _ { j \in \mathcal { K } _ { s } } \left[ \alpha _ { l , h } ( t , j ) \right] ,\tag{1}
$$

where $\Omega _ { s \in \mathcal { S } }$ denotes a cross-sample aggregation function that aggregates the per-sample scores $\Theta _ { l , h } ( s )$ over all samples $s \in S$ . In the above equation, each per-sample score $\Theta _ { l , h } ( s )$ is computed by summarizing the attention weights $\alpha _ { l , h } ( t , j )$ , which reflect how strongly head $( l , h )$ attends from a set of query tokens $t \in \mathcal { Q } _ { s }$ to a set of key tokens $j \in \mathcal { K } _ { s }$ . To this end, the key aggregation function $\Phi _ { j \in \mathcal { K } _ { s } }$ first aggregates the attention weights over the set of key tokens into a scalar per-query score, and these per-query scores are then averaged over all selected query tokens to finally obtain the per-sample score $\Theta _ { l , h } ( s )$

Under this unified formulation, existing methods [1, 20, 2, 3] mainly differ in their choices of query token set $\mathcal { Q } _ { s }$ , key aggregation function $\Phi ,$ and cross-sample aggregation function Ω. We next briefly introduce the common design choices for these variables used in prior work [1, 20, 3, 4, 2].

• Query token set $\mathcal { Q } _ { s } \mathrm { : }$ : (i) Input query tokens $\mathcal { Q } _ { s } ^ { \mathrm { i n } }$ , which specify the retrieval target, or (ii) output tokens $\mathcal { Q } _ { s } ^ { \mathrm { o u t } }$ , where the retrieved information is generated.

• Key aggregation function Φ: (i) An argmax indicator $\Phi ^ { \mathrm { a r g m a x } }$ , which returns a binary score for whether the most-attended visible key token falls within $\displaystyle \kappa _ { s } ,$ , or (ii) a region-sum score $\Phi ^ { \mathrm { { \dot { s u m } } } }$ , which returns the total attention mass assigned to key tokens in $\kappa _ { s }$

• Cross-sample aggregation function Ω: (i) average aggregation $\Omega ^ { \mathrm { a v g } }$ , which averages per-sample scores across the dataset, and (ii) selection-frequency aggregation $\Omega ^ { \mathrm { f r e q } }$ , which ranks heads by how often they are selected across samples.

Tab. 2 summarizes how existing methods instantiate these choices. Based on this unified view, we define a design space for detecting VRHs: we construct $2 ^ { 3 }$ head-scoring variants by varying the query-token set $\mathcal { Q } _ { s } ,$ key aggregation $\Phi _ { i }$ , and cross-sample aggregation Ω according to the common choices above.

Table 2: Design choices in head-scoring methods. We summarize how existing methods instantiate the main components of Eq. 1. For $\mathcal { Q } _ { s } ,$ some methods use only a few representative tokens from the indicated input or output token set, i.e., $\mathcal { Q } _ { s } ^ { \mathrm { i n } }$ or $\mathcal { Q } _ { s } ^ { \circ \mathrm { u t } }$ ; we slightly abuse notation for abstraction. <sup>†</sup>Localization Heads is a special case that does not directly use key aggregation, but instead selects heads based on the spatial entropy of attention weights.
<table><tr><td></td><td>Retrieval Heads [1] QRHead [20] OCR Heads [2] VERA [3] Loc. Head [4]</td><td></td><td></td><td></td><td></td></tr><tr><td>Query Token Set  $\mathcal { Q } _ { s }$ </td><td> $\mathcal { Q } _ { s } ^ { \mathrm { { o u t } } }$ </td><td> $\mathcal { Q } _ { s } ^ { \mathrm { i n } }$ </td><td> $\mathcal { Q } _ { s } ^ { \mathrm { { o u t } } }$ </td><td> $\mathcal { Q } _ { s } ^ { \circ \mathrm { u t } }$ </td><td> $\mathcal { Q } _ { s } ^ { \mathrm { i n } }$ </td></tr><tr><td>Key Aggregation Φ</td><td> $\Phi ^ { \mathrm { a r g m a x } }$ </td><td> $\Phi ^ { \mathsf { s u m } }$ </td><td> $\Phi ^ { \mathrm { a r g m a x } }$ </td><td> $\Phi ^ { \mathsf { s u m } }$ </td><td> $\mathrm { { N } / A ^ { \dag } }$ </td></tr><tr><td>Sample Aggregation Ω</td><td> $\Omega ^ { \tt a v g }$ </td><td> $\Omega ^ { \tt a v g }$ </td><td> $\Omega ^ { \tt a v g }$ </td><td> $\Omega ^ { \tt a v g }$ </td><td> $\Omega ^ { \tt f r e q }$ </td></tr></table>

## 3.3 Causal Validation of Head-Scoring Variants

The unified scoring formulation above defines a family of candidate scoring methods for identifying VRHs, depending on the choice of the set of query tokens, key aggregation, and cross-sample aggregation. While these scores measure how strongly a head’s attention aligns with the target visual region, attention alignment alone does not establish that the head is functionally necessary for visual retrieval. As emphasized in prior work [20, 1–3], causality is essential for interpreting a head as a meaningful functional unit rather than an attention-correlated artifact. We therefore perform causal validation: for each scoring variant, we select the top-ranked heads, mask them during inference, and measure the resulting drop in visual grounding accuracy.

We perform this causal validation and empirically choose $( \mathcal { Q } _ { s } ^ { \mathrm { o u t } } , \Phi ^ { \mathrm { s u m } } , \Omega ^ { \mathrm { m e a n } } )$ as the default design for VRH detection. Fig. 2 reports this validation across four VLMs [6–9] on RefCOCO [5], ablating one design component at a time and measuring whether the resulting top-ranked heads are causally important for visual grounding. Masking heads detected by the default variant consistently causes large accuracy drops relative to the unmasked model $( k = 0 )$ , showing that this score identifies heads that are causally important for visual grounding. The ablations further show that the query-token set is the most important design choice: variants that score attention from output query tokens identify heads whose removal causes substantially larger and more consistent accuracy drops than variants based on input query tokens. This suggests that the strongest causal retrieval signal emerges when the model is autoregressively producing the grounded answer, rather than when it only encodes the referring expression in the prompt.

![](images/3c4b614c428e2788e0f3936b7807c7f6452fd71291e657e9783fb870e964fafe.jpg)  
Figure 2: Causal validation of head-scoring variants on RefCOCO [5] across four VLMs [6–9]. We mask the top-20 heads detected by each variant and report visual grounding accuracy; lower accuracy indicates that the detected heads are more causally important. Ours denotes our default variant $\left( \mathcal { Q } _ { s } ^ { \mathrm { o u t } } , \Phi ^ { \mathrm { s u m } } , \Omega ^ { \mathrm { m e a n } } \right)$ . The $\mathcal { Q } _ { s } , \Phi ,$ , and Ω ablations each change one component of this default design at a time, while Random masks 20 randomly selected heads. The dashed line shows the unmasked model $( k = 0 )$ , and $\Delta$ denotes the accuracy gap relative to Ours.

Comparison with existing specialized heads. We next ask whether VRHs simply overlap with specialized heads identified by prior methods, or instead reveal a distinct mechanism for visual reference resolution in general images. As discussed in Sec. 2, most existing methods focus on text retrieval [1, 20] or visual retrieval from document-like images [2, 3]. Localization Heads [4] move beyond rendered text toward general images, but characterize heads by spatial concentration rather than by their functional role. In contrast, VRHs are identified by directly scoring how strongly each head attends from output tokens to the target image region, and are validated by their causal effect on visual retrieval performance. Using the same causal validation protocol on RefCOCO [5], Fig. 3 shows that masking VRHs causes the largest degradation in grounding accuracy across four VLMs, substantially exceeding the effect of masking random heads or those identified by prior methods.

![](images/20ab5d23a7e084a1a679439bedd32b3d19529a1ca5e7f6727952c6fa103e71c3.jpg)  
Figure 3: Comparison with existing heads. We compare VRHs with Retrieval Heads [1], OCR Heads [2], VERA [3], Localization Heads [4], and randomly selected heads on the RefCOCO visual grounding benchmark [5]. The x-axis denotes the number of masked heads (k). Masking the top-k VRHs causes substantially larger drops in grounding accuracy than masking these baselines, indicating that VRHs capture a distinct set of heads causally important for visual grounding.

## 4 Visual Retrieval Heads Are Universal and Sparse

Sec. 3.3 establishes that our scoring method identifies heads that are causally important for visual grounding. We now ask a stronger question: are these heads merely an artifact of a particular model, dataset, or scoring setup, or do they reveal a general mechanism by which VLMs perform visual retrieval? Following the criteria used to characterize retrieval heads in LLMs [1, 20], we evaluate whether VRHs are both universal and sparse.

Universality. VRHs are universal, consistently emerging as a causal visual-retrieval unit across diverse model families and grounding benchmarks. We evaluate eleven VLMs (Fig. 6) built on eight language-model backbones (Qwen2.5, Qwen3, Llama-3, Gemma 2, DeepSeekMoE, InternLM2.5, GLM and Phi [6–9, 27–33]) on five grounding datasets spanning three benchmark families: (RefCOCO/+/g [5, 10], RefSpatialBench [11], and Toloka [12]). As shown in Fig. 4, masking the top-ranked VRHs consistently causes large drops in grounding accuracy across all model–benchmark pairs, whereas masking the same number of random heads has little effect. Fig. 6 summarizes the same contrast at k=20 on RefCOCO for every evaluated VLM, where the seven models beyond the four studied in depth are evaluated on a separate detection and evaluation split of the benchmark (Sec. J); the same detection and masking procedure collapses grounding in all of them. This indicates that, despite differences in architecture, training recipe, and benchmark distribution, diverse VLMs rely on a sparse set of attention heads for correct visual reference behavior.

Sparsity. VRHs are highly sparse, suggesting that visual retrieval is concentrated in a small set of specialized heads rather than diffusely implemented throughout the transformer. Fig. 5 shows the distribution of VRH scores on RefCOCO across the four VLMs. Fewer than 15% of heads receive a non-negligible score (above $1 0 ^ { - 3 } )$ , and only 0–4% receive a high score (above $5 \times 1 0 ^ { - 3 } )$ , depending on the model. Despite occupying only a small fraction of each model, these heads have a large causal effect: masking them is sufficient to substantially disrupt grounding. Thus, visual retrieval in VLMs mirrors the sparse-but-causal organization of retrieval heads in LLMs [1].

![](images/39b25ebf63beb102f37526b03e9e4f4f6643a6ae205ed41360755cdd2689ad96.jpg)  
Figure 5: Sparsity of VRHs. Distribution of head scores for VRH detection on RefCOCO [5] across four VLMs [6–9]. Only a small fraction of heads receive nonnegligible scores.

## 5 Characterizing Visual Retrieval Heads

In previous sections (Sec. 3–4), we established that VRHs are causal, universal, and sparse—properties shared with existing retrieval heads for text or visual text [1–3, 20]. We now show that VRHs exhibit important additional properties: (i) VRHs generalize across visual reference tasks beyond the grounding task used to detect them (Sec. 5.1); (ii) VRHs are functionally specific to visual retrieval, selectively disrupting reference resolution without degrading general generation (Sec. 5.2); (iii) VRHs are architecturally shared across VLMs built on the same LLM backbone, indicating they reside in a reusable language-model circuit (Sec. 5.3); and (iv) VRHs are stably detectable, remaining consistent across output-token choices and requiring only a handful of annotated examples (Sec. 5.4). Note that, while we mainly focus on characterizing the properties of VRHs in this section, we kindly refer readers to Sec. B in the Appendix, where we also show that VRHs can also be utilized to improve visual grounding accuracy via direct preference optimization (DPO).

![](images/18e605a9207b5f7161e88ff2c5ca7d8921c3da6a6ddc14b96e4006bc4299d4f6.jpg)

Figure 4: Universality of VRHs across models and benchmarks. We report visual grounding accuracy after masking the top-k VRHs across four VLMs and three benchmark families: Ref-COCO/+/g [5, 10], RefSpatialBench [11], and Toloka [12]. The x-axis denotes the number of masked heads (k). Masking only a few VRHs consistently causes large accuracy drops, while masking random heads has little effect. This broad causal effect shows that VRHs are not architecture- or dataset-specific, but instead capture a shared attention-head mechanism for visual grounding.  
![](images/39f425b2155ebf276dae9a873c6e8255402f6ff5c534eea7771376851ddaaa80.jpg)  
Figure 6: VRHs are universal across VLMs. Visual grounding accuracy on RefCOCO [5] for all eleven evaluated VLMs, before masking, after masking 20 random heads, and after masking the top-20 VRHs. Masking the VRHs causes a large accuracy drop in every model, whereas masking the same number of random heads has little effect. Exact values, evaluation splits and per-model protocols are given in Sec. J.

## 5.1 VRHs Generalize Across Visual Reference Tasks

We first show that VRHs effectively generalize beyond visual grounding: they implement a more general mechanism for visual reference resolution required by diverse vision question-answering tasks involving attributes, spatial reasoning, counting, vision-centric understanding, and visual math. This cross-task generalization importantly distinguishes VRHs from specialized heads studied in prior work on LLMs and VLMs [1, 20, 2, 3], which are typically identified and evaluated within a single task, leaving open whether the discovered heads support a broader cognitive function or only the particular evaluation format used to find them.

We validate this by taking VRHs detected on a visual grounding benchmark, RefCOCO [5], and testing their causal effects on seven VQA benchmarks spanning the aforementioned tasks: VAW [13], Spatial457 [14], SpatialRGPT [15], CV-Bench [16], MMStar [17], CountBenchQA [18], and Math-Vista [19]. As shown in Tab. 3, masking VRHs consistently degrades performance across all seven benchmarks and all four VLMs, while masking the same number of random heads has little effect. These results reveal that, although VRHs are discovered from visual grounding prompts that require generating bounding-box coordinates for the referred object, they capture a general visual reference mechanism required across diverse tasks.

Table 3: Generalization across visual reference tasks. Masking the top-20 VRHs detected on RefCOCO [5] consistently degrades performance on seven VQA benchmarks [13–19] spanning attributes, spatial reasoning, counting, vision-centric understanding, and visual math, while randomhead masking has little effect.
<table><tr><td>Model</td><td>Condition</td><td>VAW</td><td>Spatial457</td><td>SpatialRGPT</td><td>CV-Bench</td><td>MMStar</td><td>CountBenchQA</td><td>MathVista</td></tr><tr><td rowspan="4">Qwen2.5-VL-7B</td><td>Baseline (k=0)</td><td>63.8</td><td>68.2</td><td>36.1</td><td>75.1</td><td>58.3</td><td>83.9</td><td>71.5</td></tr><tr><td>Random (k=20)</td><td>63.7 (-0.1)</td><td>67.2 (-1.0)</td><td>39.5 (+3.4)</td><td>71.9 (-3.2)</td><td>56.8 (-1.5)</td><td>84.1 (+0.2)</td><td>72.0 (+0.5)</td></tr><tr><td>VRH (k=20)</td><td>40.5 (-23.3)</td><td>36.0 (-32.2)</td><td>21.8 (-14.3)</td><td>53.3 (-21.8)</td><td>47.9 (-10.4)</td><td>61.5 (-22.4)</td><td>61.7 (-9.8)</td></tr><tr><td>Baseline (k=0)</td><td>74.8</td><td>78.2</td><td>50.8</td><td>87.6</td><td>63.0</td><td>88.4</td><td>66.5</td></tr><tr><td rowspan="3">Qwen3-VL-8B</td><td>Random (k=20)</td><td>74.3 (-0.5)</td><td>76.4 (-1.8)</td><td>46.2 (-4.6)</td><td>87.1 (-0.5)</td><td>63.3 (+0.3)</td><td>89.6 (+1.2)</td><td>69.4 (+2.9)</td></tr><tr><td>VRH (k=20)</td><td>51.0 (-23.8)</td><td>36.9 (-41.3)</td><td>22.9 (-27.9)</td><td>68.7 (-18.9)</td><td>47.7 (-15.3)</td><td>74.5 (-13.9)</td><td>43.0 (-23.5)</td></tr><tr><td>Baseline (k=0)</td><td>73.1</td><td>73.1</td><td>50.0</td><td>85.9</td><td>67.9</td><td>81.3</td><td>80.0</td></tr><tr><td rowspan="3">InternVL3-8B</td><td>Random (k=20)</td><td>71.9 (-1.2)</td><td>72.4 (-0.7)</td><td>49.6 (-0.4)</td><td>86.7 (+0.8)</td><td>67.7 (-0.2)</td><td>81.9 (+0.6)</td><td>79.8 (-0.2)</td></tr><tr><td>VRH(k=20)</td><td>47.6 (-25.5)</td><td>37.3 (-35.8)</td><td>38.0 (-12.0)</td><td>69.3 (-16.6)</td><td>47.6 (-20.3)</td><td>60.1 (-21.2)</td><td>65.7 (-14.3)</td></tr><tr><td>Baseline (k=0)</td><td>72.6</td><td>66.3</td><td>50.4</td><td>80.3</td><td>64.9</td><td>78.0</td><td>78.9</td></tr><tr><td rowspan="3">InternVL3.5-8B</td><td>Random (k=20)</td><td>72.0 (-0.6)</td><td>64.3 (-2.0)</td><td>51.5 (+1.1)</td><td>82.8 (+2.5)</td><td>64.4 (-0.5)</td><td>79.0 (+1.0)</td><td>79.4 (+0.5)</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>VRH (k=20)</td><td>54.4 (-18.2)</td><td>39.3 (-27.0)</td><td>41.0 (-9.4)</td><td>68.6 (-11.7)</td><td>51.5 (-13.4)</td><td>65.2 (-12.8)</td><td>66.3 (-12.6)</td></tr></table>

## 5.2 VRHs are Functionally Specific

We next show that VRHs are functionally specific: masking them does not simply degrade generation, but selectively disrupts visual reference resolution. For example, on the visual grounding task, we observe that masking VRHs leads to mislocalized bounding-box predictions, while the bounding-box syntax remains correct. In Fig. 8a, we further decompose visual-grounding failures under VRH masking versus random-head masking into two types: (a) parse failure, where the output does not preserve the bounding-box syntax and cannot be parsed, and (b) mislocation, where the output is a syntactically valid bounding box but is spatially mislocalized.

We find that masking VRHs significantly increases mislocation rates while parse failures remain rare. This contrast mirrors the “fluent-but-not-factual” failure mode reported for text retrieval heads [1, 20]: masking the specialized heads preserves the surface form of the output, but corrupts the factuality of the evidence-grounded content. Fig. 1 illustrates this failure mode qualitatively: masked models produce syntactically valid bounding boxes that localize incorrect regions.

## 5.3 VRHs are Architecturally Shared

We next show that VRHs are not specific to a single VLM instantiation, but reflect an architecturally shared mechanism within the LLM backbone. In particular, VRHs detected in one VLM remain meaningful in another VLM that shares the same LLM backbone, even when the vision encoder, projector, and instruction tuning differ. We test this with pairs of same-backbone VLMs in two ways. First, we compare the layer-head score map, where each entry at layer l and head h records the dataset-level VRH score $\bar { \Theta } _ { l , h }$ . As shown in Fig. 7(a), same-backbone VLMs exhibit similar highscoring layer-head locations, with substantial top-20 overlap and high rank correlation. Second, we perform cross-model masking: we detect the top-20 VRHs in one VLM and mask the same layer-head indices in another. As shown in Fig. 7(b), transferred VRHs cause a large grounding-accuracy drop in the target model, approaching the effect of masking its own VRHs and far exceeding random-head masking. The same holds for the third pair in Fig. 7, Magma-8B [27] and LLaVA-NeXT-Llama3- 8B [28], where cross-masking the top-20 VRHs reduces grounding accuracy from 68.8% to 0.0% and from 85.8% to 5.8% respectively. In summary, these results show that VRHs are architecturally shared across same-backbone VLMs, suggesting that visual retrieval relies on a reusable circuit in the language backbone rather than an idiosyncratic VLM-specific implementation.

![](images/095494e5f0318c3d66f2dcd548e8a2d18d039da5b923da15aa591814c94796f6.jpg)

![](images/8d9a5f4cafbedf74a74bb4dcbead55e8de88fd9e2d5d65b436a4a867b7df99ae.jpg)  
Top-20 overlap: 15, Spearman : 0.909

![](images/01a025d868269299d597d90238203b84a46921953445712d7af9979953620a4b.jpg)

![](images/5194920f3f405a0037972620a0dc15b1499bb5f70809d18dab2c5e34c2b676ec.jpg)  
Top-20 overlap: 18, Spearman : 0.89

![](images/39a3e13815eb08b249454eff7a0751f1353c74404185038d32b20bc76144ef02.jpg)

Top-20 overlap: 14, Spearman : 0.900  
![](images/1b33c669a017ad218924602c5c0943becf0b66d6d9679607288b261835d0964b.jpg)

![](images/eedba0b8bc2bfaa1ebcf5c3d5e4021436021b952b062ff48f72424607b6727fb.jpg)

![](images/d8c5a2d5c0b8e7aef9d23a6b6cfee143b4bf93478e1f8144d128c2490a2ed343.jpg)

![](images/e918804588724ce3129663c9ca0f7467921261d868486f56371628ef587e9202.jpg)

![](images/8ea788f528c0bcae7f5108b87968614bf7fab05518b817c41fce34edfacb4dd1.jpg)  
Random heads Own VRHs Transferred VRHs

![](images/fed3bfddff19f4110b1507a299e3283e72e838eaa65c021f45ca418ea1883ad7.jpg)

![](images/3f76bef622294c3beb0f1e1f4314533fed0853670ce9bbe28ba641da77ae21fd.jpg)

Figure 7: VRH detection consistency across same-backbone VLMs. Each block is a pair of VLMs that share a language backbone but differ in vision encoder, projector and instruction tuning: Qwen2.5-VL-7B [6] with InternVL3-8B [8], which share a Qwen2.5-7B backbone; Qwen3-VL-8B [7] with InternVL3.5-8B [9], which share a Qwen3-8B backbone; and Magma-8B [27] with LLaVA-NeXT-Llama3-8B [28], which share a Llama-3 backbone. Top row: Layer-head score maps, where each cell denotes the VRH score $\bar { \Theta } _ { l , h }$ at layer l and head h; high-scoring heads appear at similar layer-head locations across same-backbone VLMs. Bottom row: Cross-model masking results, where the top-k VRHs identified in one model are masked at the same layer-head indices in the other model. Own VRHs are the heads detected in the panel’s own model and transferred VRHs those detected in its partner. The x-axis denotes the number of masked heads (k).

## 5.4 VRHs are Stably Detectable

We finally show that VRHs can be detected reliably across two key sources of variation in the scoring variables: the choice of output query tokens and the number of annotated grounding examples.

Robustness across output tokens. VRH detection is stable across output-token query choices: heads identified using different subsets of output tokens are equally causal. To test this, we select the top-20 VRHs under different query-token choices—a single random output token or all output tokens—and measure grounding accuracy after masking them. As shown in Fig. 8b, both choices cause a comparable accuracy drop (7.2% and 7.5%, respectively), while masking the same number of random heads barely affects performance (86.0%). This confirms that our head-scoring method consistently identifies causally important VRHs regardless of the specific output-token query set.

![](images/9923ae6e3013295fb14413b738bc6dcdd997f86abdf16bf65b048979860c5ec2.jpg)  
(a) Functional specificity

![](images/6ca3f4e41c11ed112a4e5c48021ca859ac671d4725b1a57a7d8e94357863b2c0.jpg)  
(b) Output-token robustness

![](images/ea6607298bdc4caee73ded56ee233d662f7b651f9c5197c531827b1bda33f458.jpg)  
(c) Sample-size robustness

Figure 8: VRHs are functionally specific and stably detectable. All experiments use Qwen2.5-VL-7B [6] on RefCOCO [5] with k=20 masked heads unless otherwise noted. (a) Visual grounding failure decomposition under VRH masking vs. random-head masking. Mislocation denotes syntactically valid but mislocalized box prediction, while parsefailure denotes output that cannot be parsed as a bounding box. (b) Grounding accuracy when masking VRHs detected from a single random output token, all output tokens, or random heads. (c) Grounding accuracy when masking VRHs detected with varying numbers of scoring samples n.

Robustness across sample size. VRH detection is highly sample-efficient: only a few grounding annotations are needed to recover the same top-ranked heads as a much larger scoring set. This indicates that VRHs exhibit a strong, repeatable attention signature, rather than a weak pattern that emerges only after extensive averaging. To quantify this, we vary the number of samples used to estimate $\bar { \Theta } _ { l , h }$ and compare each resulting top-20 head set with the reference set computed from 200 examples. As shown in Fig. 8c, as few as 5–10 examples recover nearly the same top-ranked VRHs as the full scoring set. This makes VRH detection practical: a small number of bounding-box annotations is sufficient to reliably identify VRHs.

## 6 Conclusion

We introduce Visual Retrieval Heads (VRHs), a sparse set of attention heads that are causally necessary for correct visual reference behavior. Using visual grounding as a probing task, we identify VRHs by scoring attention from output tokens to ground-truth visual regions, and validate their causal role through targeted masking across multiple VLMs and grounding benchmarks. These analyses show that VRHs are sparse, causal, and universal. Importantly, their role extends beyond grounding: the same heads are causally important for attribute recognition, spatial reasoning, counting, and visual math, suggesting that VRHs support visual reference resolution generally.

Limitation and Future Work. (i) Our analysis identifies VRHs through attention weights, following prior retrieval-head methods. While sufficient to reveal sparse, causal, and general visual-reference heads, this signal primarily captures where visual evidence is routed from, rather than what information is read out or transformed. Future work could combine VRH detection with activation patching [21], or causal mediation [22] for a finer-grained mechanistic account. (ii) Our experiments focus on single-image VLM inputs as a controlled setting for isolating visual reference behavior. Extending VRH analysis to multi-image, video, and interleaved image-text contexts could further reveal how VLMs route visual evidence across richer multimodal inputs, with potential applications in efficient visual-context selection [23] and multimodal cache compression [24].

## References

[1] Wenhao Wu, Yizhong Wang, Guangxuan Xiao, Hao Peng, and Yao Fu. Retrieval head mechanistically explains long-context factuality. In ICLR, 2025.

[2] Ingeol Baek, Hwan Chang, Sunghyun Ryu, and Hwanhee Lee. How do large vision-language models see text in image? unveiling the distinctive role of ocr heads. In EMNLP, 2025.

[3] Rongcan Pei, Huan Li, Fang Guo, and Qi Zhu. Vera: Identifying and leveraging visual evidence retrieval heads in long-context understanding. arXiv preprint arXiv:2602.10146, 2026.

[4] Seil Kang, Jinyeong Kim, Junhyeok Kim, and Seong Jae Hwang. Your large vision-language model only needs a few attention heads for visual grounding. In CVPR, 2025.

[5] Licheng Yu, Patrick Poirson, Shan Yang, Alexander C Berg, and Tamara L Berg. Modeling context in referring expressions. In ECCV, 2016.

[6] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2.5-VL technical report. arXiv preprint arXiv:2502.13923, 2025.

[7] Shuai Bai, Yuxuan Cai, Ruizhe Chen, et al. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025.

[8] Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Yuchen Duan, Hao Tian, Weijie Su, Jie Shao, et al. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models. arXiv preprint arXiv:2504.10479, 2025.

[9] Weiyun Wang, Zhangwei Gao, Lixin Gu, Hengjun Pu, Long Cui, Xingguang Wei, Zhaoyang Liu, Linglin Jing, Shenglong Ye, Jie Shao, et al. Internvl3.5: Advancing open-source multimodal models in versatility, reasoning, and efficiency. arXiv preprint arXiv:2508.18265, 2025.

[10] Junhua Mao, Jonathan Huang, Alexander Toshev, Oana Camburu, Alan Yuille, and Kevin Murphy. Generation and comprehension of unambiguous object descriptions. In CVPR, 2016.

[11] Enshen Zhou, Jingkun An, Cheng Chi, Yi Han, Shanyu Rong, Chi Zhang, Pengwei Wang, Zhongyuan Wang, Tiejun Huang, Lu Sheng, et al. Roborefer: Towards spatial referring with reasoning in visionlanguage models for robotics. In NeurIPS, 2025.

[12] Dmitry Ustalov, Nikita Pavlichenko, Sergey Koshelev, Daniil Likhobaba, and Alisa Smirnova. Toloka visual question answering benchmark. arXiv preprint arXiv:2309.16511, 2023.

[13] Khoi Pham, Kushal Kafle, Zhe Lin, Zhihong Ding, Scott Cohen, Quan Tran, and Abhinav Shrivastava. Learning to predict visual attributes in the wild. In CVPR, 2021.

[14] Xingrui Wang, Wufei Ma, Tiezheng Zhang, Celso M de Melo, Jieneng Chen, and Alan Yuille. Spatial457: A diagnostic benchmark for 6d spatial reasoning of large mutimodal models. In CVPR, 2025.

[15] An-Chieh Cheng, Hongxu Yin, Yang Fu, Qiushan Guo, Ruihan Yang, Jan Kautz, Xiaolong Wang, and Sifei Liu. Spatialrgpt: Grounded spatial reasoning in vision-language models. In NeurIPS, 2024.

[16] Shengbang Tong, Ellis Brown, Penghao Wu, Sanghyun Woo, Manoj Middepogu, Sai C Akula, Jihan Yang, Shusheng Yang, Adithya Iyer, Xichen Pan, et al. Cambrian-1: A fully open, vision-centric exploration of multimodal llms. In NeurIPS, 2024.

[17] Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Jiaqi Wang, Yu Qiao, Dahua Lin, et al. Are we on the right way for evaluating large vision-language models? In NeurIPS, 2024.

[18] Roni Paiss, Ariel Ephrat, Omer Tov, Shiran Zada, Inbar Mosseri, Michal Irani, and Tali Dekel. Teaching clip to count to ten. In ICCV, 2023.

[19] Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. In ICLR, 2024.

[20] Wuwei Zhang, Fangcong Yin, Howard Yen, Danqi Chen, and Xi Ye. Query-focused retrieval heads improve long-context reasoning and re-ranking. In EMNLP, 2025.

[21] Nicholas Goldowsky-Dill, Chris MacLeod, Lucas Sato, and Aryaman Arora. Localizing model behavior with path patching. arXiv preprint arXiv:2304.05969, 2023.

[22] Qiming Li, Zekai Ye, Xiaocheng Feng, Weihong Zhong, Weitao Ma, and Xiachong Feng. Causal tracing of object representations in large vision language models: Mechanistic interpretability and hallucination mitigation. In AAAI, 2026.

[23] Fengyun Wang, Sicheng Yu, Jiawei Wu, Jinhui Tang, Hanwang Zhang, and Qianru Sun. 3d question answering via only 2d vision-language models. In ICML, 2025.

[24] Suyu Ge, Yunan Zhang, Liyuan Liu, Minjia Zhang, Jiawei Han, and Jianfeng Gao. Model tells you what to discard: Adaptive KV cache compression for LLMs. In ICLR, 2024.

[25] Youmi Ma and Naoaki Okazaki. From interpretability to performance: Optimizing retrieval heads for long-context language models. In Findings of ACL, 2026.

[26] Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. In NeurIPS, 2023.

[27] Jianwei Yang, Reuben Tan, Qianhui Wu, Ruijie Zheng, Baolin Peng, Yongyuan Liang, Yu Gu, Mu Cai, Seonghyeon Ye, Joel Jang, Yuquan Deng, Lars Liden, and Jianfeng Gao. Magma: A foundation model for multimodal ai agents. In CVPR, 2025.

[28] Bo Li, Kaichen Zhang, Hao Zhang, Dong Guo, Renrui Zhang, Feng Li, Yuanhan Zhang, Ziwei Liu, and Chunyuan Li. Llava-next: Stronger llms supercharge multimodal capabilities in the wild, 2024. https://llava-vl.github.io/blog/2024-05-10-llava-next-stronger-llms/.

[29] Andreas Steiner, André Susano Pinto, Michael Tschannen, Daniel Keysers, Xiao Wang, Yonatan Bitton, Alexey Gritsenko, Matthias Minderer, Anthony Sherbondy, Shangbang Long, Siyang Qin, Reeve Ingle, Emanuele Bugliarello, Sahar Kazemzadeh, Thomas Mesnard, Ibrahim Alabdulmohsin, Lucas Beyer, and Xiaohua Zhai. Paligemma 2: A family of versatile vlms for transfer. arXiv preprint arXiv:2412.03555, 2024.

[30] Zhiyu Wu, Xiaokang Chen, Zizheng Pan, Xingchao Liu, Wen Liu, Damai Dai, Huazuo Gao, Yiyang Ma, Chengyue Wu, Bingxuan Wang, Zhenda Xie, Yu Wu, Kai Hu, Jiawei Wang, Yaofeng Sun, Yukun Li, Yishi Piao, Kang Guan, Aixin Liu, Xin Xie, Yuxiang You, Kai Dong, Xingkai Yu, Haowei Zhang, Liang Zhao, Yisong Wang, and Chong Ruan. Deepseek-vl2: Mixture-of-experts vision-language models for advanced multimodal understanding. arXiv preprint arXiv:2412.10302, 2024.

[31] Zhe Chen, Weiyun Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Erfei Cui, Jinguo Zhu, Shenglong Ye, Hao Tian, Zhaoyang Liu, et al. Expanding performance boundaries of open-source multimodal models with model, data, and test-time scaling. arXiv preprint arXiv:2412.05271, 2024.

[32] Z.ai. Glm-4.6v: Open source multimodal models with native function calling. Z.ai Technical Blog, 2025.

[33] Jyoti Aneja et al. Phi-4-reasoning-vision-15b technical report. arXiv preprint arXiv:2603.03975, 2026.

## A Additional Qualitative Results

## A.1 Visual Grounding

In Figure A1, we provide additional visualizations that complement Fig. 1 in the main paper. Specifically, we visualize the average attention map under three attention-head settings: (i) all heads, (ii) VRHs, and (iii) all heads excluding VRHs, on two grounding benchmarks: RefCOCO [5] and Toloka [12]. Across all examples, the all-heads attention concentrates on the ground-truth referent (column ii), and this concentration is preserved when only VRHs are kept (column iii) but largely lost when VRHs are masked (column iv). VRHs are therefore the heads responsible for routing attention to the referent: under VRH masking, attention on the referent vanishes and the predicted bounding box becomes mislocalized.

Input

All heads

VRHs

All heads except VRHs

![](images/2a040c4c4874fab02eb047e4af1e098e3933e6a8962a85420360c8c34cd44c18.jpg)  
Figure A1: Attention map visualizations on visual grounding tasks. From left to right, the columns show the input image and the mean attention maps over: (i) all heads, (ii) the top-20 VRHs, and (iii) all heads excluding VRHs. Green and red boxes indicate ground-truth and predicted bounding boxes, respectively.

## A.2 Visual Question Answering

Figure A2 shows analogous visualizations for visual question-answering (VQA) benchmarks: VAW [13] and Spatial457 [14]. As in the grounding case, attention on the visual evidence is preserved when only VRHs are kept but lost when they are masked: the model’s answer remains fluent but becomes visually unfaithful, the “fluent-but-not-factual” failure mode discussed in Sec. 5.2.

"What color is the tarp?"  
![](images/bec2ca55d1586c0b20643b7fce9eb6fc72ca2227355d50acd8178242c2769280.jpg)

![](images/8a8c2b03754c172eaa30bd290adf4783150757e32865d76f3b36ddb54fc2bce0.jpg)  
"What color is the banner?"

![](images/6b72b9650bae7085cfaba80cdb0d8ad1a23aa8419c5664aeeebdb5fdec6447c3.jpg)

![](images/1113fdf9d70d9631083045d4ee5c7249572baae4a7a1bcb25f64d334f4ef9d3d.jpg)  
Reasoning: : The tarp appears to be a shade of brown or tan, which is not one of the provided color options. Answer: : orange

![](images/f059984ec7d7500a87bda7f9c83164bed1ece882f8e15a8e84c433c650cdeedb.jpg)

![](images/736f20e8b5c4c10d51f8f0dc5051a3258361c7309a4932fc75b253d1e88aaaf5.jpg)

(Correct)

![](images/8b2585b65a4664c86737b3487e42d643debe60444c021f0f188700c5828a929b.jpg)  
Reasoning: The giraffe is standing in front of a building with a wooden structure that appears to be a part of a zoo or wildlife park. The tarp is not visible in the image. Answer: brown  
Reasoning: The banners in the image appears to be blue Answer: blue  
Reasoning: The image shows a street scene with people walking and sitting. There's a banner hanging from a building, but which the text is not clearly visible. Answer: brown

(Correct)

(Wrong)

(Wrong)  
"What color is the utility bike?"  
![](images/4d7e8871064c326ac7fe8a0c3d9b3fd4c87ca3d296caa099c9e1b9a242c52bdf.jpg)

Reasoning: The utility bike is located on the left side of the image. It has a purple frame and black wheels. The color of the utility bike is purple. Answer: Purple

![](images/0cb117bb6aefc6f76e4bdb01c1b15d06d41610bd949ee2cf7d9b7a09f3ba69ac.jpg)  
Reasoning: The image shows various vehicles and objects, including but the utility bike is not clearly visible. Answer: The utility bike is not clearly visible in the image.

![](images/2f4e46299741db75b1c98be416400f0a79373627ad2ef575e97bfa887df9d334.jpg)  
"Are any purple jets visible?"  
Reasoning: The image contains a small airplane with a purple body and pink wings, which matches the description purple jets visible? query Answer: Yes

![](images/372647d49303ef85747883df77b0c291f67df64c390244e3d308cd99c3453de1.jpg)  
Reasoning: The image shows various vehicles and objectss, including but a purple jet is not present. Answer: No  
(Wrong)  
(Correct)  
(Wrong)  
(Correct)

Figure A2: Attention map visualizations on visual question-answering tasks. Each panel shows the input image (upper left), and mean attention maps over: (i) all heads (lower left), (ii) the top-20 VRHs (upper right), and (iii) all heads excluding VRHs (lower right). Green boxes indicate the ground-truth target regions.

## B Application: Direct Preference Optimization Utilizing Visual Retrieval Heads

While the main paper focuses on discovering VRHs and characterizing their properties, the heads we identify are also actionable: they can be used as targets for downstream optimization. Specifically, we follow the recipe of Ma et al. [25], who propose optimizing retrieval heads via DPO [26] using positive/negative policies obtained from masked versus unmasked models.

## B.1 Method

We adapt their recipe to the visual setting:

• Positive policy $\pi ^ { + }$ : the original (unmasked) VLM.

• Negative policy $\pi ^ { - }$ : the same VLM with the top-20 VRHs masked.

• DPO data: for each grounding example $s = ( I _ { s } , q _ { s } ) \in \mathcal { D } _ { \operatorname { t r a i n } }$ , we sample $y _ { w } \sim \pi ^ { + } ( \cdot \mid I _ { s } , q _ { s } )$ as the chosen response and $y _ { l } \sim \pi ^ { - } ( \cdot \mid I _ { s } , q _ { s } )$ as the rejected response.

• Training: we fine-tune the unmasked VLM on these pairs with the standard DPO loss [26] with full parameter update, $\mathrm { l r } \mathrm { = } 2 e \mathrm { ~ - ~ } 7$ , for 4 epochs on 6,000 pairs from RefCOCO/+/g train split.

As a control, we train a separate model with random-head-masked negatives, sampling 20 random heads (excluding first-layer heads) for each example.

The intuition mirrors the LLM case: the negative policy concentrates exactly the failure mode we want to suppress (visually-unfaithful grounding driven by ablated VRHs), so DPO sharpens the model in the direction of stronger reliance on those same heads.

## B.2 Results

As shown in Fig. A3, we observe modest but consistent gains, with an average improvement of 0.78 percentage-points across three visual grounding benchmarks [5, 10]. This suggests that VRHs are not merely an analytic abstraction, but a mechanistic substrate that can be directly optimized. We view this as an initial application; using VRHs for routing decisions, KV-cache compression analogous to the LLM setting [1], or test-time intervention are promising directions for future work.

![](images/63f617061dbe35fca8e5f0699fcabea367e64880754f841c70f4ff9bf51b826e.jpg)

![](images/2369a8a099cdc86f10db316726e758e7c70dc89b17c3897482b98cc8936ad4dc.jpg)

![](images/fab34bd1d6e73afb6efbc83cfcc3726bc0c0f51f62cb36d84e7e0696e7d2c684.jpg)  
Figure A3: Grounding accuracy comparisons with DPO under varying negative policies. We compare two DPO variants that use either random-head dropping or VRH dropping to construct negative responses. Using VRH dropping as the negative policy yields stronger downstream improvements than random-head dropping, suggesting that perturbing causally relevant heads provides a more informative preference signal for optimization.

## C Full Quantitative Results

The main paper reports averaged results over RefCOCO/+/g [5, 10]. Here, we provide the full per-dataset breakdown and per-seed variability.

## C.1 Per-dataset Visual Grounding Results under Head Masking

Table A1 reports visual grounding accuracy separately on five referring-expression visual grounding benchmarks: RefCOCO, RefCOCO+, RefCOCOg, Toloka, and RefSpatialBench [5, 10, 12, 11], for all four VLMs and k ∈ {0, 1, 3, 5, 10, 20} under VRH masking and random-head masking.

Table A1: Per-dataset visual grounding accuracy.
<table><tr><td>Model</td><td>Method</td><td>k=0</td><td>k=1</td><td>k=3</td><td>k=5</td><td>k=10</td><td>k=20</td></tr><tr><td colspan="8">RefCOCO [5]</td></tr><tr><td>Qwen2.5-VL-7B</td><td>Random VRH</td><td>86.2 86.2</td><td>86.2 85.1</td><td>86.4 83.5</td><td>86.2 79.3</td><td>85.9 9.7</td><td>86.1 7.5</td></tr><tr><td>Qwen3-VL-8B</td><td>Random VRH</td><td>91.3 91.3</td><td>91.3 91.0</td><td>91.2</td><td>91.3</td><td>91.1 33.4</td><td>91.1 20.4</td></tr><tr><td>InternVL3-8B</td><td>Random</td><td>93.6</td><td>93.6</td><td>90.4 93.7</td><td>84.1 93.3</td><td>92.9</td><td>92.5</td></tr><tr><td>InternVL3.5-8B</td><td>VRH Random VRH</td><td>93.6 90.5 90.5</td><td>93.6 90.5</td><td>89.0 90.5</td><td>50.7 90.5</td><td>12.4 90.0</td><td>8.8 88.9</td></tr><tr><td colspan="8">RefCOCOg [10]</td></tr><tr><td>Qwen2.5-VL-7B</td><td>Random VRH</td><td>82.8 82.8</td><td>82.9 82.5</td><td>82.9 79.9</td><td>83.1 76.1</td><td>82.7 11.8</td><td>82.5 9.2</td></tr><tr><td>Qwen3-VL-8B</td><td>Random VRH</td><td>87.7 87.7</td><td>87.8 87.5</td><td>87.7 86.6</td><td>87.2 81.8</td><td>87.4</td><td>87.3 23.5</td></tr><tr><td>InternVL3-8B</td><td>Random VRH</td><td>90.0 90.0</td><td>89.9</td><td>90.0</td><td>89.8</td><td>30.6 89.4</td><td>88.9</td></tr><tr><td>InternVL3.5-8B</td><td>Random VRH</td><td>86.8 86.8</td><td>90.0 87.0</td><td>62.4 86.9</td><td>56.1 86.9</td><td>17.0 86.8</td><td>12.1 85.7 32.1</td></tr><tr><td colspan="8">RefCOCO+ [5]</td></tr><tr><td>Qwen2.5-VL-7B</td><td>Random VRH</td><td>78.5 78.5</td><td>78.4</td><td>78.3 75.3</td><td>78.7 71.5</td><td>78.0</td><td>78.3 7.5</td></tr><tr><td>Qwen3-VL-8B</td><td>Random VRH</td><td>84.2 84.2</td><td>84.2</td><td>84.2</td><td>84.0</td><td>9.4 83.7</td><td>83.8</td></tr><tr><td>InternVL3-8B</td><td>Random</td><td>86.8</td><td>83.9 86.8</td><td>83.0 87.0</td><td>78.2 86.8</td><td>27.8 86.2</td><td>17.0 86.0</td></tr><tr><td>InternVL3.5-8B</td><td>VRH Random</td><td>86.8 83.4</td><td>84.4 83.3</td><td>81.5 83.4</td><td>45.4 83.4</td><td>11.2 83.4</td><td>8.0 82.2</td></tr><tr><td></td><td>VRH</td><td>83.4</td><td>82.4</td><td>81.0</td><td>75.1</td><td>52.2</td><td>25.4</td></tr><tr><td colspan="8">Toloka [12]</td></tr><tr><td>Qwen2.5-VL-7B</td><td>Random VRH</td><td>67.4 67.4</td><td>67.2</td><td>67.4</td><td>66.9</td><td>67.6</td><td>67.2 1.1</td></tr><tr><td>Qwen3-VL-8B</td><td>Random</td><td>77.7</td><td>65.0 77.3</td><td>53.0 77.7</td><td>7.0 77.6</td><td>2.2 76.8</td><td>76.5</td></tr><tr><td></td><td>VRH</td><td>77.7</td><td>77.4</td><td>76.2</td><td>48.0</td><td>15.6</td><td>5.0</td></tr><tr><td>InternVL3-8B</td><td>Random</td><td>81.2</td><td>81.2</td><td>82.0</td><td>81.8</td><td>80.7</td><td>80.2</td></tr><tr><td>InternVL3.5-8B</td><td>VRH</td><td>81.2</td><td>80.4</td><td>77.2</td><td>29.9</td><td>6.0</td><td>1.6</td></tr><tr><td></td><td>Random</td><td>67.0</td><td>66.9</td><td>66.2</td><td>66.7</td><td>67.0</td><td>61.3 10.3</td></tr><tr><td></td><td>VRH</td><td>67.0</td><td>66.3</td><td>64.0</td><td>59.9</td><td>27.0</td><td></td></tr><tr><td colspan="8">RefSpatialBench [11]</td></tr><tr><td>Qwen2.5-VL-7B</td><td>Random</td><td>43.0</td><td>43.0</td><td>37.0</td><td>42.0</td><td>40.0</td><td>36.0</td></tr><tr><td>Qwen3-VL-8B</td><td>VRH</td><td>43.0</td><td>35.0</td><td>27.0</td><td>12.0</td><td>5.0</td><td>0.0</td></tr><tr><td></td><td>Random VRH</td><td>47.0</td><td>46.0</td><td>49.0</td><td>45.0</td><td>46.0</td><td>46.0 6.0</td></tr><tr><td></td><td></td><td>47.0</td><td>51.0</td><td>51.0</td><td>33.0</td><td>8.0</td><td></td></tr><tr><td>InternVL3-8B</td><td>Random</td><td>39.0</td><td>37.0</td><td>39.0</td><td>40.0</td><td>42.0</td><td>40.0</td></tr><tr><td></td><td>VRH</td><td>39.0</td><td>37.0</td><td>37.0</td><td>27.0</td><td>6.0</td><td>6.0</td></tr><tr><td>InternVL3.5-8B</td><td>Random</td><td>44.0</td><td>41.0</td><td>44.0</td><td>42.0</td><td>42.0</td><td>34.0</td></tr><tr><td></td><td>VRH</td><td>44.0</td><td>40.0</td><td>24.0</td><td>20.0</td><td>11.0</td><td>5.0</td></tr></table>

## D Detected Lists of Visual Retrieval Heads

For reproducibility, we list the top-20 VRH (layer, head) indices identified by our scoring method for each (VLM, grounding-benchmark) pair. The lists are ordered by descending head score $\bar { \Theta } _ { l , h }$

## D.1 Top-20 VRHs On RefCOCO [5]

Table A2: Top-20 VRHs detected on RefCOCO [5]. Each entry is (l, h).
<table><tr><td>Model</td><td>Top-20 VRH (layer, head)</td></tr><tr><td>Qwen2.5-VL-7B</td><td>(19,17), (16,0), (18,16), (16,20), (19,22), (19,20), (16,1), (19,23), (19,15), (17,18), (16,7), (14,11), (14,16), (16,6), (16,17), (19,26), (19,19), (14,23), (17,14), (18,15)</td></tr><tr><td>Qwen3-VL-8B</td><td>(20,15), (21,10), (18,19), (18,15), (21,11), (21,18), (17,27), (17,4), (19,3), (22,4), (21,8), (20,8), (18,5), (23,13), (20,17), (20,18), (23,30), (21,25), (20,16), (23,10)</td></tr><tr><td>InternVL3-8B</td><td>(16,20), (16,0), (19,22), (16,1), (19,23), (19,17), (19,20), (16,6), (14,11), (19,15), (16,4), (17,24), (18,16), (16,16), (16,17), (20,21), (16,7), (19,19), (14,16), (14,8)</td></tr><tr><td>InternVL3.5-8B</td><td>(21,11), (20,15), (21,10), (19,3), (18,15), (17,4), (21,18), (18,19), (17,27), (20,17), (24,13), (21,8), (20,18), (20,16), (21,25), (20,8), (18,5), (22,4), (18,21), (23,30)</td></tr></table>

## D.2 Top-20 VRHs On RefCOCO+ [5]

Table A3: Top-20 VRHs detected on RefCOCO+ [5]. Each entry is (l, h).
<table><tr><td>Model</td><td>Top-20 VRH (layer, head)</td></tr><tr><td>Qwen2.5-VL-7B</td><td>(19,17), (16,0), (16,20), (18,16), (19,22), (19,20), (16,1), (19,15), (19,23), (17,18), (14,11), (16,7), (14,16), (16,6), (16,17), (19,19), (19,26), (17,24), (14,23), (17,14)</td></tr><tr><td>Qwen3-VL-8B</td><td>(20,15), (21,10), (18,19), (18,15), (17,27), (21,11), (21,18), (17,4), (19,3), (20,8), (22,4), (21,8), (20,17), (20,18), (23,13), (21,25), (18,5), (23,30), (20,16), (23,10)</td></tr><tr><td>InternVL3-8B</td><td>(16,0), (16,20), (19,22), (19,23), (16,1), (19,17), (19,20), (16,6), (19,15), (14,11), (16,4), (17,24), (18,16), (20,21), (16,7), (16,16), (16,17), (14,8), (14,16), (19,19)</td></tr><tr><td>InternVL3.5-8B</td><td>(21,11), (21,10), (20,15), (19,3), (18,15), (21,18), (17,4), (18,19), (17,27), (20,17), (21,8), (20,18), (20,16), (24,13), (20,8), (22,4), (21,25), (17,26), (23,30), (23,10)</td></tr></table>

## D.3 Top-20 VRHs On RefCOCOg [10]

Table A4: Top-20 VRHs detected on RefCOCOg [10]. Each entry is (l, h).
<table><tr><td>Model</td><td>Top-20 VRH (layer, head)</td></tr><tr><td>Qwen2.5-VL-7B</td><td>(19,17), (16,0), (16,20), (19,22), (18,16), (16,1), (19,20), (19,23), (19,15), (14,11), (17,18), (16,7), (14,16), (16,6), (16,17), (19,26), (14,23), (19,19), (17,14), (18,15)</td></tr><tr><td>Qwen3-VL-8B</td><td>(20,15), (18,19), (21,10), (17,27), (18,15), (21,18), (21,11), (17,4), (19,3), (22,4), (21,8), (20,8), (18,5), (20,17), (20,18), (21,25), (23,30), (23,13), (20,16), (17,24)</td></tr><tr><td>InternVL3-8B</td><td>(16,20), (16,0), (16,1), (19,22), (19,17), (19,23), (14,11), (16,6), (19,20), (16,4), (19,15), (18,16), (16,17), (16,16), (17,24), (14,8), (14,16), (16,7), (20,21), (14,14)</td></tr><tr><td>InternVL3.5-8B</td><td>(21,11), (20,15), (21,10), (18,15), (19,3), (17,4), (21,18), (18,19), (17,27), (20,17), (20,18), (20,16), (21,8), (24,13), (19,13), (20,8), (18,21), (17,26), (22,4), (18,5)</td></tr></table>

## D.4 Top-20 VRHs On Toloka [12]

Table A5: Top-20 VRHs detected on Toloka [12]. Each entry is (l, h).
<table><tr><td>Model</td><td>Top-20 VRH (layer, head)</td></tr><tr><td>Qwen2.5-VL-7B</td><td>(19,17), (16,0), (19,22), (16,20), (19,15), (19,20), (19,23), (16,1), (18,16), (17,18), (19,19), (16,6), (19,26), (14,11), (19,25), (16,17), (16,16), (16,18), (16,4), (20,21)</td></tr><tr><td>Qwen3-VL-8B</td><td>(21,10), (20,15), (18,19), (21,11), (17,4), (18,15), (23,30), (21,8), (23,13), (22,4), (21,18), (20,17), (17,27), (23,10), (20,16), (20,8), (20,18), (19,3), (24,16), (21,31)</td></tr><tr><td>InternVL3-8B</td><td>(16,20), (19,17), (16,1), (19,20), (16,0), (19,15), (19,22), (16,17), (14,11), (19,23), (18,16), (16,16), (16,6), (20,21), (14,8), (19,19), (16,7), (17,24), (16,18), (21,25)</td></tr><tr><td>InternVL3.5-8B</td><td>(18,19), (21,11), (17,27), (21,10), (20,15), (18,15), (20,17), (17,4), (24,13), (23,30), (21,8), (19,3), (20,16), (21,18), (20,8), (23,10), (20,23), (18,21), (19,13), (21,9)</td></tr></table>

## D.5 Top-20 VRHs On RefSpatialBench [11]

Table A6: Top-20 VRHs detected on RefSpatialBench [11]. Each entry is (l, h).
<table><tr><td>Model</td><td>Top-20 VRH (layer, head)</td></tr><tr><td>Qwen2.5-VL-7B</td><td>(19,17), (19,22), (19,20), (19,15), (16,0), (16,20), (19,23), (16,1), (19,19), (18,16), (17,18), (19,26), (20,21), (16,16), (16,6), (16,4), (14,11), (19,25), (16,7), (20,23)</td></tr><tr><td>Qwen3-VL-8B</td><td>(21,10), (21,11), (20,15), (21,8), (23,30), (17,4), (18,19), (18,15), (21,18), (23,13), (23,10), (22,4), (20,17), (19,3), (20,16), (17,27), (20,8), (23,14), (25,10), (21,31)</td></tr><tr><td>InternVL3-8B</td><td>(19,17), (16,1), (16,20), (19,22), (19,15), (19,20), (19,23), (16,0), (14,11), (20,21), (17,24), (19,19), (16,6), (18,16), (16,17), (16,7), (16,16), (21,25), (21,5), (16,4)</td></tr><tr><td>InternVL3.5-8B</td><td>(21,11), (21,10), (21,8), (18,15), (19,3), (18,19), (21,18), (20,17), (17,4), (20,15), (17,27), (20,8), (24,13), (23,30), (20,23), (20,16), (21,25), (23,10), (20,21), (21,9)</td></tr></table>

## E Experimental Setup

We provide additional details on the VLMs, benchmarks, and head-masking procedure used throughout the paper.

## E.1 Vision-Language Models

We evaluate our method on four open-source VLMs spanning two model families: Qwen2.5-VL-7B [6], Qwen3-VL-8B [7], InternVL3-8B [8], and InternVL3.5-8B [9]. These four models form two same-backbone pairs that differ in vision encoder, projector, and instruction-tuning recipe. This controlled pairing is what enables the cross-model VRH transfer analysis in Sec. 5.3 of the main paper.

Table A7 summarizes the architectural specifications relevant to our analysis: the number of layers L, attention heads per layer H, and total head count L·H that appears in the denominator of our sparsity ratios $( \mathrm { e . g . , \ } ^ {  } 2 0 / 7 8 4 = 2 . 6 \% ^ { , , }$ for Qwen2.5-VL-7B).

Table A7: Architectural details of the four evaluated VLMs.
<table><tr><td>Model</td><td># Layers L</td><td># Heads/layer H</td><td>Total heads</td><td>Language backbone</td><td>Vision encoder</td><td>Projector</td></tr><tr><td>Qwen2.5-VL-7B</td><td>28</td><td>28</td><td>784</td><td>Qwen2.5-7B</td><td>Qwen ViT</td><td>MLP merger</td></tr><tr><td>Qwen3-VL-8B</td><td>36</td><td>32</td><td>1152</td><td>Qwen3-8B</td><td>Qwen ViT</td><td>MLP merger + DeepStack</td></tr><tr><td>InternVL3-8B</td><td>28</td><td>28</td><td>784</td><td>Qwen2.5-7B</td><td>InternViT-300M</td><td>2-layer MLP</td></tr><tr><td>InternVL3.5-8B</td><td>36</td><td>32</td><td>1152</td><td>Qwen3-8B</td><td>InternViT-300M</td><td>2-layer MLP</td></tr></table>

## E.2 Benchmarks

Visual grounding benchmarks (head detection & main causality evaluation). We use five referring-expression visual grounding datasets, grouped into three families:

• RefCOCO/+/g [5, 10]: standard COCO-derived referring-expression benchmarks. We evaluate on the validation splits and use 8,811, 3,804, 7,573 examples per split.

• RefSpatialBench [11]: a referring-expression benchmark emphasizing spatial-relation expressions. We use 100 examples (full set).

• Toloka VQA [12]: a crowd-sourced visual referring benchmark with diverse expression styles. We use 1705 examples (full set).

For all grounding datasets, a prediction is judged correct if IoU $( \hat { B } _ { s } , B _ { s } ) \ge \tau = 0 . 5$ (consistent with the main paper).

VQA benchmarks (cross-task generalization in Sec. 5.1). We further evaluate masked VLMs on seven VQA benchmarks spanning attribute prediction (VAW [13], Spatial457 [14]), spatial reasoning (SpatialRGPT [15], CV-Bench [16]), vision-centric understanding (MMStar [17]), counting (CountBenchQA [18]), and visual math (MathVista [19]).

• VAW [13]: We manually convert VAW into a multiple-choice question (MCQ) format and evaluate on 6,309 samples.

• Spatial457 [14]: We use the L1 (Single Object) portion and evaluate on 670 samples.

• SpatialRGPT [15]: We use the quantitative object-size estimation subset and evaluate on 266 samples.

• CV-Bench [16]: We use the full benchmark and evaluate on 2,638 samples.

• MMStar [17]: We use the minitest split and evaluate on 1,500 samples.

• CountBenchQA [18]: We use the full set of converted QA and evaluate on 491 samples.

• MathVista [19]: We use the MCQ split and evaluate on 1,000 samples.

## E.3 Head Masking Procedure

To evaluate the causal contribution of a head (l, h), we ablate it during the forward pass by zeroing the attention output of head (l, h) before it is added to the residual stream, leaving all other heads, MLPs, and layer norms untouched. This is the same masking convention used by Wu et al. [1] for text retrieval heads, ensuring our results are directly comparable.

Generation settings. All evaluations use greedy decoding and the model’s default chat template. For grounding, the bounding box is parsed from the generated text using the model-specific output format (e.g. "bbox\_2d": [x1, y1, x2, y2] for Qwen-VL families).

## E.4 Compute

All experiments were conducted on a single NVIDIA RTX A6000 (48 GB VRAM).

## F Ablation on the Combined Query Token Set

Sec. 3.3 of the main paper ablates $\mathcal { Q } _ { s } ^ { \mathrm { i n } }$ and $\mathcal { Q } _ { s } ^ { \mathrm { o u t } }$ separately. Here we additionally experiment with a VRH detection setting that uses their union $\mathcal { Q } _ { s } ^ { \mathrm { i n } } \cup \dot { \mathcal { Q } } _ { s } ^ { \mathrm { o u t } }$ , and compare the causal effectiveness of the resulting heads with that of our original VRHs in Tab. A8. Because this setting uses a superset of the query tokens compared to our original $\mathcal { Q } _ { s } ^ { \mathrm { o u t } }$ setting, it achieves comparable causal effectiveness for some models. However, our original setting still performs better on three models, Qwen3-VL, InternVL3 and InternVL3.5, with a particularly noticeable gap on Qwen3-VL. These results suggest that our original setting is more robust across models while also being more efficient, as it uses substantially fewer query tokens for head detection.

Table A8: Visual grounding accuracy on RefCOCO [5] after masking the top-20 VRHs detected using $\mathcal { Q } _ { s } ^ { \mathrm { i n } } \cup \mathcal { Q } _ { s } ^ { \mathrm { o u t } }$ versus ${ \mathcal { Q } } _ { s } ^ { \mathrm { { o u t } } }$ (ours) across four VLMs.
<table><tr><td></td><td>Qwen2.5-VL-7B</td><td>Qwen3-VL-8B</td><td>InternVL3-8B</td><td>InternVL3.5-8B</td></tr><tr><td>Original model</td><td>86.11</td><td>91.27</td><td>93.59</td><td>90.48</td></tr><tr><td>w/ Masking VRHs  $( \mathcal { Q } _ { s } ^ { \mathrm { i n } } \cup \mathcal { Q } _ { s } ^ { \mathrm { o u t } } )$ </td><td>7.14</td><td>29.85</td><td>8.98</td><td>32.94</td></tr><tr><td>w/ Masking VRHs  $( \mathcal { Q } _ { s } ^ { \mathrm { o u t } } ; \mathrm { o u r s } )$ </td><td>7.48</td><td>20.39</td><td>8.76</td><td>32.03</td></tr></table>

## G Sensitivity to the Number of Masked Heads

Fig. 2 of the main paper reports the scoring-variant ablation at k=20. Tab. A9 reports the same ablation for $k \in \{ 5 , 1 0 , 2 \bar { 0 } , \bar { 5 } 0 \}$ across the four VLMs, comparing our design choice $( \mathcal { Q } _ { s } ^ { \mathrm { o u t } } , \Phi ^ { \mathrm { s u m } } , \Omega ^ { \mathrm { m e a n } } )$ against variants obtained by flipping one design component at a time and against random-head masking. At k=5 the masking effect is relatively weak or inconsistent across models, which makes the relative ranking among scoring variants less conclusive. Once k increases beyond 5, our design choice consistently ranks either first or second in terms of causal effect across $k \in \{ 1 0 , 2 0 , 5 0 \}$ Notably, masking only the top-10 VRHs already causes substantial degradation across all scoring variants, showing that a small subset of VRHs is sufficient to strongly impair grounding. We use k=20 in the main experiments, following the convention established in the retrieval-head literature [1, 2]; Tab. A9 shows that our main conclusion remains largely robust for $k \geq 1 0$

Table A9: Scoring ablation for varying numbers of masked heads $k \in \{ 5 , 1 0 , 2 0 , 5 0 \}$ across four VLMs. We report RefCOCO [5] visual grounding accuracy after masking random heads, our detected VRHs, or VRHs obtained by flipping each scoring design choice. Bold and underline mark the largest and the second-largest causal effect among the four scoring variants in each row.
<table><tr><td></td><td>Random heads</td><td>VRHs (Ours)</td><td>Flipped Qs</td><td>Flipped Φ</td><td>Flipped Ω</td></tr><tr><td colspan="6">(a) Qwen2.5-VL-7B</td></tr><tr><td>k=0 (original model)</td><td>86.11</td><td>86.11</td><td>86.11</td><td>86.11</td><td>86.11</td></tr><tr><td>k=5</td><td>86.16</td><td>79.29</td><td>56.08</td><td>63.28</td><td>63.14</td></tr><tr><td>k=10</td><td>85.79</td><td>9.63</td><td>10.52</td><td>8.87</td><td>9.63</td></tr><tr><td>k=20</td><td>85.97</td><td>7.48</td><td>7.68</td><td>8.13</td><td>8.01</td></tr><tr><td>k=50</td><td>84.67</td><td>6.63</td><td>9.85</td><td>7.21</td><td>8.70</td></tr><tr><td colspan="6">(b) Qwen3-VL-8B</td></tr><tr><td>k=0</td><td>91.27</td><td>91.27</td><td>91.27</td><td>91.27</td><td>91.27</td></tr><tr><td>k=5</td><td>91.26</td><td>84.14</td><td>79.90</td><td>57.03</td><td>84.24</td></tr><tr><td>k=10</td><td>91.12</td><td>33.42</td><td>80.47</td><td>33.60</td><td>32.63</td></tr><tr><td>k=20</td><td>91.09</td><td>20.39</td><td>51.10</td><td>33.61</td><td>21.96</td></tr><tr><td>k=50</td><td>89.97</td><td>9.29</td><td>7.20</td><td>14.17</td><td>13.11</td></tr><tr><td colspan="6">(c) InternVL3-8B</td></tr><tr><td>k=0</td><td>93.59</td><td>93.59</td><td>93.59</td><td>93.59</td><td>93.59</td></tr><tr><td>k=5</td><td>93.28</td><td>50.62</td><td>72.06</td><td>50.62</td><td>56.16</td></tr><tr><td>k=10</td><td>92.90</td><td>12.39</td><td>39.05</td><td>14.20</td><td>14.20</td></tr><tr><td>k=20</td><td>92.46</td><td>8.76</td><td>15.83</td><td>7.89</td><td>9.93</td></tr><tr><td>k=50</td><td>90.98</td><td>8.31</td><td>13.90</td><td>7.35</td><td>8.87</td></tr><tr><td colspan="6">(d) InternVL3.5-8B</td></tr><tr><td>k=0</td><td>90.48</td><td>90.48</td><td>90.48</td><td>90.48</td><td>90.48</td></tr><tr><td>k=5</td><td>90.45</td><td>83.52</td><td>86.94</td><td>82.22</td><td>83.52</td></tr><tr><td>k=10</td><td>89.95</td><td>64.48</td><td>71.50</td><td>69.83</td><td>64.48</td></tr><tr><td>k=20</td><td>88.86</td><td>32.03</td><td>45.66</td><td>63.96</td><td>37.56</td></tr><tr><td>k=50</td><td>88.09</td><td>16.98</td><td>10.42</td><td>23.49</td><td>19.73</td></tr></table>

## H Head Detection and Evaluation Splits

Split protocol. For each grounding benchmark we use a fixed detection set for head scoring and evaluate the masked models on the remaining samples. On RefCOCO [5] the detection set consists of 200 samples drawn from the validation split and the evaluation set is the remaining 8,611 samples, so the two sets are disjoint by construction. In our original RefCOCO setup we used 200 samples for head detection and the full set of 8,811 samples for evaluation. Because the detection set constitutes only a small portion of the full dataset, we did not exclude it from evaluation, as we anticipated that it would have a minimal effect on the results. Tab. A10 reports causal validation results both with and without these 200 head-scoring samples; the results do not meaningfully differ. We nevertheless use disjoint head detection and evaluation sets as the default protocol.

We also note that Tab. 3 of the main paper includes cross-task causal evaluations in which heads detected using 200 RefCOCO samples are evaluated on separate, non-visual-grounding benchmarks, effectively alleviating concerns about similarity between the head scoring and evaluation sets.

Table A10: Visual grounding accuracy on RefCOCO [5] after masking the top-20 VRHs, evaluated with and without the 200 head-detection samples.
<table><tr><td>Evaluation set</td><td>Qwen2.5-VL-7B</td><td>Qwen3-VL-8B</td><td>InternVL3-8B</td><td>InternVL3.5-8B</td></tr><tr><td>w/ head detection samples (n=8,811)</td><td>7.46</td><td>20.44</td><td>8.76</td><td>32.14</td></tr><tr><td>w/o head detection samples  $_ { ( n = 8 , 6 1 1 ) }$ </td><td>7.48</td><td>20.39</td><td>8.76</td><td>32.03</td></tr></table>

## I Detection Cost and Fused Attention

VRH detection requires explicitly extracting attention weights over visual tokens, so its cost grows with the number of visual tokens. Tab. A11 reports the peak memory usage and computation time of VRH detection while varying the number of visual tokens by changing the input resolution or the number of input images. To examine how memory and computational costs interact with the use of fused-attention kernels, we report results using both eager and fused attention. As the number of visual tokens increases from 256 to 512 and 3,328, both memory usage and computation time increase, with the growth being substantially more pronounced under eager attention. In particular, at 3,328 visual tokens, eager attention requires 34.59 GiB and 4.86 s/sample, compared with 16.01 GiB and 2.31 s/sample using fused attention. These results show that fused-attention kernels substantially mitigate the memory and computational overhead associated with scaling to larger numbers of visual tokens. This is because eager attention materializes dense attention maps whose memory grows approximately quadratically with sequence length, whereas fused attention avoids retaining the full maps by recomputing only the query rows needed for VRH scoring.

Table A11: Peak memory usage and computation time of VRH detection with varying numbers of visual tokens. We use InternVL3-8B [8] with dynamic tiling as the base model and conduct all measurements on a single 48 GB A6000 GPU.
<table><tr><td>Setting</td><td># Visual tokens</td><td>Peak memory (GiB) (Eager / Fused Att.)</td><td>Time (s/sample) (Eager / Fused Att.)</td></tr><tr><td>Low resolution (448×448)</td><td>256</td><td>15.09 / 14.92</td><td> $1 . 7 7 \pm 0 . 4 7 / 1 . 4 1 \pm 0 . 2 1$ </td></tr><tr><td>Low resolution (448×448), 2 images</td><td>512</td><td>15.55 / 15.01</td><td> $2 . 3 3 \pm { 0 . 9 9 } / { 1 . 5 6 } \pm { 0 . 3 4 }$ </td></tr><tr><td>High resolution (1792×1344)</td><td>3,328</td><td>34.59 / 16.01</td><td> $4 . 8 6 \pm 0 . 7 7 / 2 . 3 1 \pm 0 . 2 9$ </td></tr></table>

## J Evaluated VLMs: Protocols and Per-Model Results

This section provides the protocol and the exact values behind Fig. 6 and Fig. 7 of the main paper.

Protocol. The four VLMs studied throughout the paper follow the split of Sec. H. For the seven further VLMs, VRHs are detected on a 200-sample RefCOCO [5] detection set using the same scoring function, and the masked models are evaluated on a disjoint 500-sample evaluation set of the same benchmark. The random-head control masks the same number of heads, k=20, drawn outside the detected set. Tab. A12 lists their exact accuracies as plotted in Fig. 6.

Table A12: Exact values plotted in Fig. 6 for the seven VLMs introduced in addition to the four studied throughout the paper. Visual grounding accuracy on RefCOCO [5] after masking the top-20 VRHs versus the same number of random heads. GLM-4.6V-Flash and Phi-4-Reasoning-Vision were released after the main experiments of this work were completed.
<table><tr><td>Condition Language backbone</td><td>Magma-8B [27] Llama-3</td><td>LLaVA-NeXT- Llama3-8B [28] Llama-3</td><td>PaliGemma2- 10B [29] Gemma 2</td><td>DeepSeek-VL2- Small [30] DeepSeekMoE</td><td>InternVL2.5- 8B [31] InternLM2.5</td><td>GLM-4.6V- Flash [32] GLM</td><td>Phi-4-Reasoning- Vision [33] Phi</td></tr><tr><td>Baseline (k=0)</td><td>68.8</td><td>85.8</td><td>73.6</td><td>93.0</td><td>90.0</td><td>86.4</td><td>74.2</td></tr><tr><td>Random (k=20)</td><td>64.0</td><td>77.8</td><td>73.2</td><td>91.6</td><td>86.8</td><td>85.8</td><td>73.8</td></tr><tr><td>VRH (k=20)</td><td>7.8</td><td>5.0</td><td>11.6</td><td>21.4</td><td>11.4</td><td>10.0</td><td>18.0</td></tr></table>

Cross-model transfer for the Llama-3 pair. Magma-8B [27] and LLaVA-NeXT-Llama3-8B [28] share Llama-3-based language backbones, making them a third same-backbone pair for the analysis of Sec. 5.3; their score maps and the full k-sweep are plotted in Fig. 7 of the main paper. As shown in Tab. A13, cross-masking VRHs causes a significant drop in visual grounding accuracy in both directions, supporting the claim that VRHs are transferable across models with the same LLM backbone in non-Qwen families as well.

Table A13: Cross-model transferability of VRHs between Magma-8B [27] and LLaVA-NeXT-Llama3-8B [28]. We report RefCOCO [5] visual grounding accuracy before and after masking the top-20 VRHs identified in the other model.
<table><tr><td></td><td>Magma-8B</td><td>LLaVA-NeXT-Llama3-8B</td></tr><tr><td>Original model</td><td>68.8</td><td>85.8</td></tr><tr><td>w/ Masking own top-20 VRHs</td><td>7.8</td><td>5.0</td></tr><tr><td>w/ Cross-masking VRHs</td><td>0.0</td><td>5.8</td></tr><tr><td>w/ Masking random heads</td><td>64.0</td><td>77.8</td></tr></table>